# CLAUDE.md

This repository is [openpi](https://github.com/Physical-Intelligence/openpi) extended with pi0.5
LoRA supervised-fine-tuning (SFT) configs for four manipulation task families. The full training
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
in the [SFT.md](SFT.md) table. Norm-stats: **clutter and cabinet** have theirs committed under
`norm_stats/` (read automatically — instant start); **stack, jar, lid and dusty** ship config-only,
so `run_sft.sh` computes their norm-stats on the fly before the first training (a one-time CPU step,
no manual action). Either way you just pass `--config`.

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

## Before launching: verify the config against this node

The hyperparameters in `train_configs.py` are defaults tuned for a particular machine. **Always
double-check this node's actual compute resources (GPU count and memory, CPU) and adjust the
config before a full run** — most importantly `batch_size`, `num_workers`, and `fsdp_devices`.

### Dataloader throughput (verify these against THIS node before trusting them)

Training is a single JAX process with one DataLoader whose `num_workers` worker processes decode
the dataset's video frames. Two settings govern whether the GPUs stay fed; both ship with
defaults that you should **re-derive for this node's CPU rather than assume**:

- **Per-worker thread caps** — `run_sft.sh` and `docker/sft.Dockerfile` set
  `OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=NUMEXPR_NUM_THREADS=1`. Without this,
  every worker's BLAS/OMP tries to grab all cores, the workers oversubscribe the CPU and thrash,
  and the dataloader stalls for minutes at a time. Confirm it is actually in effect on this node:
  `echo $OMP_NUM_THREADS` inside the run should print `1`.
- **`num_workers` (default 72)** — with threads capped, this is what decides how many cores decode
  in parallel. 72 assumes a host with roughly 96 physical cores (fill the cores, leave headroom
  for the JAX host process). **Check this node's real core count** (`nproc` / `lscpu`) and adjust:
  aim for a bit below the PHYSICAL core count, not the logical/hyperthread count. Override without
  editing the config via `--num-workers <n>`.

**Sanity-check that it's actually working**, don't just trust the numbers: after launch the
"Initialized data loader" line should appear within ~1-2 minutes (a much longer init means the
workers are still thrashing — the thread caps are not taking effect), and `nvidia-smi` GPU
utilization should be steady/high rather than a sawtooth of bursts and multi-minute idle stalls.
If it is still a sawtooth **even with the caps confirmed and `num_workers` sized to the cores**,
the bottleneck is the video-decode itself (the datasets are stored as AV1, which is slow to
random-access decode on CPU); the fix then is GPU/NVDEC decode or transcoding the dataset to
H.264, not more workers. In short: treat the shipped values as a starting point and **objectively
verify they are right for this specific machine.**
