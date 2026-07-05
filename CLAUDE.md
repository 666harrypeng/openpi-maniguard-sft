# CLAUDE.md

This repository is [openpi](https://github.com/Physical-Intelligence/openpi) extended with pi0.5
LoRA supervised-fine-tuning (SFT) configs for two manipulation task families. The full training
guide is **[SFT.md](SFT.md)** — read it before running anything.

## What this repo does

Fine-tunes pi0.5 (warm-started from `pi05_base`, LoRA) on joint-space demonstration datasets.
Two configs:

- `pi05-base_datagen_v1_clutter_joint_2cam_lora`
- `pi05-base_datagen_v1_cabinet_joint_2cam_lora`

Config definitions live in `maniguard/openpi_sft/train_configs.py`. Norm-stats are committed under
`norm_stats/` and read automatically — no separate compute step.

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
