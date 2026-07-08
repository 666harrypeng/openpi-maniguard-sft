# CLAUDE.md

This repository is [openpi](https://github.com/Physical-Intelligence/openpi) extended with pi0.5
LoRA supervised-fine-tuning (SFT) configs for four manipulation task families. The full training
guide is **[SFT.md](SFT.md)** — read it before running anything (it has the per-family task table).

## What this repo does

Fine-tunes pi0.5 (warm-started from `pi05_base`, LoRA) on joint-space demonstration datasets.
Four configs (one per family — clutter, cabinet, stack, jar):

- `pi05-base_datagen_v1_clutter_joint_2cam_lora`
- `pi05-base_datagen_v1_cabinet_joint_2cam_lora`
- `pi05-base_datagen_v1_stack_joint_2cam_lora`
- `pi05-base_datagen_v1_jar_joint_2cam_lora`

Config definitions live in `maniguard/openpi_sft/train_configs.py`; each config's dataset + task are
in the [SFT.md](SFT.md) table. Norm-stats: **clutter and cabinet** have theirs committed under
`norm_stats/` (read automatically — instant start); **stack and jar** ship config-only, so
`run_sft.sh` computes their norm-stats on the fly before the first training (a one-time CPU step, no
manual action). Either way you just pass `--config`.

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
