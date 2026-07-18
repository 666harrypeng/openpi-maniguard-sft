# CLAUDE.md

This repository is [openpi](https://github.com/Physical-Intelligence/openpi) extended with pi0.5
LoRA supervised-fine-tuning (SFT) configs for six manipulation task families. The full training
guide is **[SFT.md](SFT.md)** — read it before running anything (it has the per-family task table).

## What this repo does

Fine-tunes pi0.5 (warm-started from `pi05_base`, LoRA) on joint-space demonstration datasets.
Six configs (one per family — clutter, cabinet, stack, jar, lid, dusty):

- `pi05-base_datagen_v1_clutter_joint_2cam_lora`
- `pi05-base_datagen_v1_cabinet_joint_2cam_lora`
- `pi05-base_datagen_v1_stack_joint_2cam_lora`
- `pi05-base_datagen_v1_jar_joint_2cam_lora`
- `pi05-base_datagen_v1_lid_joint_2cam_lora`
- `pi05-base_datagen_v1_dusty_joint_2cam_lora`

Config definitions live in `maniguard/openpi_sft/train_configs.py`; each config's dataset + task are
in the [SFT.md](SFT.md) table. Norm-stats: a family with stats committed under `norm_stats/`
starts instantly; a family shipped config-only has them computed once before its first training
and cached in `outputs/norm_stats/` for every later run (a one-time CPU step, no manual action).
Either way you just pass `--config`.

## Running SFT

The recommended path is the container (`docker/sft.Dockerfile`):

```bash
docker build -t maniguard-sft -f docker/sft.Dockerfile .
docker run --gpus all -e HF_TOKEN -e WANDB_API_KEY -v "$PWD/outputs:/app/outputs" \
  maniguard-sft --config <config-name>
```

`HF_TOKEN` (write access to the model repos) and `WANDB_API_KEY` are required and passed at run
time. Each run smoke-tests, trains, and streams checkpoints to the config's Hugging Face model
repo. The non-container (uv) path and all options are in [SFT.md](SFT.md).

## Before launching on a new machine — verify these

Objectively check the actual machine rather than assuming the defaults fit it:

1. **CPU vs `num_workers`**: run `nproc` / `lscpu` and confirm the config's `num_workers=48`
   sits below the *physical* core count; lower it with `--num-workers` if not (oversubscribing
   cores makes the dataloader thrash and stall).
2. **XLA env is clean**: `env | grep XLA_PYTHON_CLIENT` should print nothing. JAX's defaults are
   the correct configuration here; in particular `XLA_PYTHON_CLIENT_PREALLOCATE=false` causes
   large-batch OOMs and must not be set.
3. **Disk**: clear stale local checkpoints from earlier runs (`outputs/sft_runs/*/checkpoints/`)
   — each finished run holds 4 step dirs of ~9 GB. Do **NOT** delete downloaded datasets
   (`outputs/hf/lerobot/`) or computed norm-stats (`outputs/norm_stats/`): both are reused as-is
   by later runs (norm-stats are deterministic dataset statistics — recomputing them wastes
   hours and changes nothing).
4. **Tokens**: `HF_TOKEN` (write access to the model repos) and `WANDB_API_KEY` are exported.

## One run, all 8 GPUs (data parallel)

Every config trains as **one job that owns all visible GPUs** — validated end-to-end on an
8-GPU node. `batch_size=256` is the GLOBAL batch: JAX shards it across the cards (32
samples/card), replicates the model on each (`fsdp_devices=1` — the train state is ~11 GiB, so
parameter sharding would solve a non-problem and pay per-layer collectives for it), and
all-reduces the trainable-parameter gradients once per step (ms-scale on NVLink). Measured
parallel efficiency is ~100%: the per-card step time is identical to a single-card run at
batch 32, i.e. 8 cards are 8× as fast. 32 samples/card is the measured GPU-saturation point —
per-card peak is ~23 GiB (≈22 GiB activations+model), and larger per-card batches add step time
but zero throughput. Faster cards speed this config up automatically — throughput scales with
per-card compute, so no per-hardware tuning is needed.

**No XLA memory env is needed** — JAX's default preallocation (a 75% pool of each card) holds
the ~23 GiB per-card peak with plenty of headroom on any modern data-center card. Do not set
`XLA_PYTHON_CLIENT_PREALLOCATE=false` (the on-demand allocator fragments and OOMs at large
batches where the pooled default is fine); if the shell environment sets any
`XLA_PYTHON_CLIENT_*` variable, unset it and use the defaults.

`batch_size`, the step counts and `peak_lr` are coupled — `num_train_steps = frames * 2 / 256`,
`peak_lr = 7e-5` (the value proven healthy at global batch 256), and
`decay_steps == num_train_steps` (enforced in `register()`: a decay that outlives the run stops
training mid-anneal at a far-too-high LR). Prefer the shipped values; if you change `--batch`,
recompute all of them.

### Checkpoints: 4 per run, latest run wins

`save_interval = keep_period = ceil(steps/4)` — a checkpoint lands every **half epoch** and every
one is a keeper, so a 2-epoch run yields exactly 4 checkpoints (0.5 / 1.0 / 1.5 / 2.0 epochs; a
3-epoch run would naturally yield 6 — the half-epoch interval does not depend on the epoch
count). The watcher streams each to the config's HF repo as it finalizes; locally a step dir also
holds `train_state/` (optimizer state, ~9 GB total with params), which is never uploaded.

Re-runs are **latest-run-wins**, both locally and on HF: launching without `--resume` overwrites
the exp's local checkpoints and first clears the HF repo's step folders. The HF clear is
load-bearing, not cosmetic — a re-run's checkpoints have identical file names and sizes to the
old ones, so without it the already-pushed dedup would silently keep the OLD run's weights on
HF. `--resume` continues the same logical run and keeps both.

### Dataset fetch (first run of a family downloads it)

`run_sft.sh` ensures the dataset is fully on local disk BEFORE training starts
(`fetch_dataset.py`): a completion marker makes later runs start instantly with zero Hub
traffic; a missing dataset is downloaded from the Hub with resume + backoff (the Hub
rate-limits these many-small-file datasets — a throttled fetch keeps resuming at full speed and
is bounded to a few hours of retrying before failing loud). Training never begins with a
partial dataset, which rules out the mid-training stall-on-Hub-fetch failure mode.

### Dataloader throughput

Training is a single JAX process with one DataLoader whose `num_workers` worker processes decode
the dataset's video frames for all 8 cards.

- **Per-worker thread caps** — `run_sft.sh` and `docker/sft.Dockerfile` set
  `OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=NUMEXPR_NUM_THREADS=1`. Without this,
  every worker's BLAS/OMP grabs all cores, the workers oversubscribe the CPU and thrash, and the
  dataloader stalls for minutes at a time. Confirm: `echo $OMP_NUM_THREADS` in the run prints `1`.
- **`num_workers` (default 48)** — sized for a host with ~96+ physical cores; one worker
  supplies ~13 samples/s of decoded video, so 48 workers (~620 samples/s) feed even the fastest
  current cards' consumption with ~3× headroom. **It must stay below the machine's physical core
  count** (`nproc` / `lscpu` — physical cores, not hyperthreads); on a smaller host lower it via
  `--num-workers <n>`. A pure perf knob with no effect on training dynamics.

**If the GPUs sawtooth** — bursts of ~100% utilization separated by multi-minute idle stalls —
the dataloader is starving. Note what that is *not*: the datasets are H.264 256×256 @ 30 fps and
decode is cheap (a random-access frame read costs tens of milliseconds; sequential decode runs at
tens of thousands of fps). With the thread caps in effect, **decode is not the bottleneck and
adding workers will not help.** Look at IO instead: is the LeRobot dataset actually on a local
disk, or on a network filesystem? `py-spy dump --pid <a worker pid>` during a stall shows exactly
where the worker is blocked.
