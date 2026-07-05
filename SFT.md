# pi0.5 LoRA SFT

This repo is [openpi](https://github.com/Physical-Intelligence/openpi) with added pi0.5
LoRA supervised-fine-tuning configs for two manipulation task families, trained from the
`pi05_base` checkpoint on joint-space demonstration datasets.

| Config | Dataset | Task |
|---|---|---|
| `pi05-base_datagen_v1_clutter_joint_2cam_lora` | [`IDEAS-Lab-Northwestern/datagen-clutter-v1-joint-5cam`](https://huggingface.co/datasets/IDEAS-Lab-Northwestern/datagen-clutter-v1-joint-5cam) | pick the target out of a cluttered tabletop into the goal |
| `pi05-base_datagen_v1_cabinet_joint_2cam_lora` | [`IDEAS-Lab-Northwestern/datagen-cabinet-v1-joint-5cam`](https://huggingface.co/datasets/IDEAS-Lab-Northwestern/datagen-cabinet-v1-joint-5cam) | open a drawer, place the object inside, close it |

Data: LeRobot v2.1, FrankaPanda absolute joint, 5 camera streams (the external `image_left`
overview + `wrist_image` are consumed; joints are converted to per-step deltas at train time).
Model: pi0.5, LoRA (`gemma_2b` r16 + `gemma_300m` r32), `discrete_state_input`.

## Environment

Dependencies are managed with [uv](https://docs.astral.sh/uv/) exactly as upstream openpi:

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

Or build the container (recommended on a cluster) — see [Docker](#docker) below. The GPU
kernel driver + `nvidia-container-toolkit` must already be present on the host; the CUDA
runtime and all Python dependencies are installed inside the container/venv.

## Train

Set `HF_TOKEN` (write access to the model repos) and `WANDB_API_KEY`, then:

```bash
uv run tools/openpi_sft/run_sft.sh --config pi05-base_datagen_v1_clutter_joint_2cam_lora
uv run tools/openpi_sft/run_sft.sh --config pi05-base_datagen_v1_cabinet_joint_2cam_lora
```

`run_sft.sh` runs a 100-step smoke test, then full training, streaming each checkpoint to the
config's model repo (`IDEAS-Lab-Northwestern/pi05-base-datagen-v1-<family>-joint-2cam-lora`) as
it finalizes. Norm-stats are committed under `norm_stats/` and read automatically — no compute
step. Run artifacts (checkpoints, logs) go under `outputs/sft_runs/<exp>/`.

Only `--config` is required; `--exp`, the push repo, and run length default from the config.
Common overrides: `--batch N`, `--steps N`, `--no-push`, `--smoke-only`. Recompute norm-stats
against this exact openpi with `--norm-stats`.

### Throughput

`num_workers` (dataloader workers, in the config) is the main throughput knob — if the GPUs
wait on data, raise it toward the node's CPU-core count and cap thread oversubscription with
`OMP_NUM_THREADS` (e.g. `export OMP_NUM_THREADS=4`) so the data workers don't thrash the CPU.

## Docker

```bash
docker build -t maniguard-sft -f docker/sft.Dockerfile .
docker run --gpus all -e HF_TOKEN -e WANDB_API_KEY \
  -v "$PWD/outputs:/app/outputs" maniguard-sft \
  --config pi05-base_datagen_v1_clutter_joint_2cam_lora
```

The image bakes in the code + norm-stats; secrets are passed at run time. `outputs/` is mounted
so logs/checkpoints persist on the host. Equivalent: `docker compose -f docker/sft.compose.yml
run --rm sft --config <name>`.

## Adding a family

Append a `TrainConfig` block to `maniguard/openpi_sft/train_configs.py` and drop its
`norm_stats/<config>/<repo_id>/norm_stats.json` into place; it becomes launchable by name.
