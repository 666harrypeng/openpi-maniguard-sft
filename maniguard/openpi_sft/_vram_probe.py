"""Print VRAM at the three milestones that decide whether ``batch_size`` fits.

Sizing ``batch_size`` is the one thing you cannot derive on paper, and the naive
reading -- "load a batch and see how much it costs" -- measures the wrong thing:
the batch tensors are tiny (batch x 2 cams x 256x256x3 bytes is a few MiB). What
actually scales with ``batch_size`` is the forward/backward **activation**
memory, and that only exists once a step has run. So the probe reports:

  1. first batch on device      -- the batch tensors alone (expect: small)
  2. train state initialized    -- + model params and optimizer state
  3. first training step done   -- + activations. This **peak** is the number
                                   that decides whether ``batch_size`` fits.

Step 3 also prints the breakdown (model+optimizer vs activations) and the
remaining headroom, which is what you need to raise or lower ``batch_size``.

openpi's ``scripts/train.py`` is a pristine upstream file, so this attaches to
three stable anchors inside it instead of editing it:

  * ``logging.info("Initialized data loader: ...")`` -- emitted right after
    ``batch = next(data_iter)``, i.e. the first batch is on the device.
  * ``logging.info("Initialized train state: ...")`` -- emitted right after
    ``jax.block_until_ready(train_state)``, i.e. params + optimizer are resident.
  * the first ``wandb.log(...)`` call -- reached only after the first
    ``ptrain_step``, and the ``jax.device_get`` just above it has already forced
    that step to complete, so the peak is real by then.

FAIL-SAFE BY DESIGN: every hook wraps the original and swallows its own errors,
so a probe failure can never break or slow a training run. If an anchor ever
disappears upstream the probe simply never fires; an ``atexit`` tripwire then
warns that a milestone was missed, so this file gets fixed rather than silently
rotting.

Set ``MANIGUARD_VRAM_PROBE=0`` to disable.

Mirrors ``_augmax_patch`` / ``_lerobot_video_patch`` -- ManiGuard additions stay
out of the upstream tree.
"""

from __future__ import annotations

import logging
import os

_GIB = 1024**3

# Milestone bookkeeping: label -> in-use bytes at that milestone.
_MARKS: dict[str, int] = {}
_seen: set[str] = set()


def _jax_mem() -> tuple[int, int, int] | None:
    """(in_use, peak, limit) bytes from the JAX allocator on device 0."""
    try:
        import jax

        stats = jax.devices()[0].memory_stats()
    except Exception:
        return None
    if not stats:
        return None
    return (
        int(stats.get("bytes_in_use", 0)),
        int(stats.get("peak_bytes_in_use", 0)),
        int(stats.get("bytes_limit", 0)),
    )


def _device_mem() -> tuple[int, int] | None:
    """(used, total) MiB for the whole card, per nvidia-smi.

    Complements the JAX allocator view: this also counts the CUDA context and
    anything else resident on the card, so it is the number to compare against
    the card's advertised capacity.
    """
    import subprocess

    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        ).stdout.strip().splitlines()[0]
        used, total = (int(x.strip()) for x in out.split(","))
    except Exception:
        return None
    return used, total


def _wandb_summary(values: dict[str, float]) -> None:
    """Mirror the milestone numbers into the wandb run summary (visible in the UI).

    openpi calls wandb.init before the data loader, so the run exists by the time
    any milestone fires; with wandb disabled the summary write is a no-op.
    """
    try:
        import wandb

        if wandb.run is not None:
            for k, v in values.items():
                wandb.run.summary[k] = round(v, 2)
    except Exception:  # a probe must never break the run
        pass


def _report(key: str, title: str) -> None:
    mem = _jax_mem()
    if mem is None:
        return
    in_use, peak, limit = mem
    _MARKS[key] = in_use

    print(f"[vram] ===== {title} " + "=" * max(0, 46 - len(title)), flush=True)
    line = f"[vram]   jax in-use : {in_use / _GIB:7.2f} GiB"
    if _MARKS.get("_prev") is not None:
        line += f"   (delta {(in_use - _MARKS['_prev']) / _GIB:+7.2f} GiB)"
    line += f"   peak : {peak / _GIB:7.2f} GiB"
    print(line, flush=True)
    _MARKS["_prev"] = in_use

    dev = _device_mem()
    if dev is not None:
        used, total = dev
        print(
            f"[vram]   device     : {used / 1024:7.2f} / {total / 1024:.2f} GiB used "
            f"(whole card, incl. CUDA context)",
            flush=True,
        )

    _wandb_summary({f"vram/{key}_in_use_gib": in_use / _GIB, "vram/peak_gib": peak / _GIB})

    # At the last milestone, turn the raw numbers into the batch-sizing answer.
    if key == "step" and "state" in _MARKS and "batch" in _MARKS:
        model_opt = _MARKS["state"] - _MARKS["batch"]
        activations = max(0, peak - _MARKS["state"])
        print(
            f"[vram]   -> batch tensors {_MARKS['batch'] / _GIB:.2f} GiB | "
            f"model+optimizer {model_opt / _GIB:.2f} GiB | "
            f"activations+workspace ~{activations / _GIB:.2f} GiB",
            flush=True,
        )
        _wandb_summary({
            "vram/batch_tensors_gib": _MARKS["batch"] / _GIB,
            "vram/model_optimizer_gib": model_opt / _GIB,
            "vram/activations_workspace_gib": activations / _GIB,
        })
        if dev is not None:
            used, total = dev
            free = (total - used) / 1024
            print(
                f"[vram]   -> headroom at this batch_size: ~{free:.2f} GiB "
                f"({100.0 * free / (total / 1024):.0f}% of the card still free)",
                flush=True,
            )
    print("[vram] " + "=" * 52, flush=True)


# Anchor substring -> (bookkeeping key, printed title). Both strings are emitted
# by openpi's train.py at exactly the moment we want to measure.
_LOG_ANCHORS: dict[str, tuple[str, str]] = {
    "Initialized data loader": ("batch", "1/3  first batch on device"),
    "Initialized train state": ("state", "2/3  model + optimizer on device"),
}


def _hook_logging() -> None:
    """Measure when openpi announces the data loader / train state are ready."""
    original = logging.info
    if getattr(original, "_maniguard_vram", False):
        return

    def info(msg, *args, **kwargs):  # noqa: ANN001, ANN202
        original(msg, *args, **kwargs)
        try:
            text = str(msg)
            for anchor, (key, title) in _LOG_ANCHORS.items():
                if anchor in text and anchor not in _seen:
                    _seen.add(anchor)
                    _report(key, title)
        except Exception:  # a probe must never break the run
            pass

    info._maniguard_vram = True  # type: ignore[attr-defined]
    logging.info = info  # type: ignore[assignment]


def _wrap_wandb_log(wandb) -> None:  # noqa: ANN001
    """Wrap wandb.log so its first call reports the post-first-step peak."""
    original = wandb.log
    if getattr(original, "_maniguard_vram", False):
        return

    def log(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        # openpi's main() wandb.log()s sample images right after the data loader,
        # BEFORE the train state exists -- only a log made after the train-state
        # milestone can be the post-first-step one, so gate on that.
        if "Initialized train state" in _seen and "first_step" not in _seen:
            _seen.add("first_step")
            try:
                _report("step", "3/3  first train step done (PEAK)")
            except Exception:
                pass
        return original(*args, **kwargs)

    log._maniguard_vram = True  # type: ignore[attr-defined]
    wandb.log = log


def _hook_wandb() -> None:
    """Measure on the first wandb.log -- the first train step has completed by then.

    ``wandb.init()`` REBINDS ``wandb.log`` to the live run's method (before init it
    is only a placeholder that raises), which silently drops a wrapper installed
    beforehand. openpi's ``init_wandb`` always calls ``wandb.init`` -- with
    ``mode="disabled"`` when wandb is off -- so the wrapper is re-applied *after*
    every init instead of only once at import.
    """
    try:
        import wandb
    except Exception:
        return

    _wrap_wandb_log(wandb)  # in case init already ran

    original_init = wandb.init
    if getattr(original_init, "_maniguard_vram", False):
        return

    def init(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        result = original_init(*args, **kwargs)
        try:
            _wrap_wandb_log(wandb)
        except Exception:
            pass
        return result

    init._maniguard_vram = True  # type: ignore[attr-defined]
    wandb.init = init


def _tripwire() -> None:
    """Warn (never raise) if a milestone never fired -- i.e. an anchor rotted."""
    missing = [k for k in (*_LOG_ANCHORS, "first_step") if k not in _seen]
    if missing and _seen:  # _seen empty => the run never got started; not our problem
        logging.warning(
            "VRAM probe: milestone(s) never fired: %s -- openpi's train.py anchors may "
            "have changed; update maniguard.openpi_sft._vram_probe.",
            ", ".join(missing),
        )


def apply() -> None:
    """Install the probe (no-op when MANIGUARD_VRAM_PROBE=0)."""
    if os.environ.get("MANIGUARD_VRAM_PROBE", "1") == "0":
        return
    import atexit

    _hook_logging()
    _hook_wandb()
    atexit.register(_tripwire)
