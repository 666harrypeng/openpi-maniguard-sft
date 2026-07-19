# pi0.5 LoRA SFT

This repo is [openpi](https://github.com/Physical-Intelligence/openpi) with added pi0.5
LoRA supervised-fine-tuning configs for six manipulation task families, trained from the
`pi05_base` checkpoint on joint-space demonstration datasets.

Progress: check off a family once its SFT run is done (edit `[ ]` -> `[x]`).

| Done | Config | Dataset | Task |
|---|---|---|---|
| [ ] | `pi05-base_datagen_v1_clutter_joint_2cam_lora` | [`IDEAS-Lab-Northwestern/datagen-clutter-v1-joint-5cam`](https://huggingface.co/datasets/IDEAS-Lab-Northwestern/datagen-clutter-v1-joint-5cam) | pick the target out of a cluttered tabletop into the goal |
| [ ] | `pi05-base_datagen_v1_cabinet_joint_2cam_lora` | [`IDEAS-Lab-Northwestern/datagen-cabinet-v1-joint-5cam`](https://huggingface.co/datasets/IDEAS-Lab-Northwestern/datagen-cabinet-v1-joint-5cam) | open a drawer, place the object inside, close it |
| [ ] | `pi05-base_datagen_v1_stack_joint_2cam_lora` | [`IDEAS-Lab-Northwestern/datagen-stack-v1-joint-5cam`](https://huggingface.co/datasets/IDEAS-Lab-Northwestern/datagen-stack-v1-joint-5cam) | unstack the 3-object pile onto a re-stack pile aside, then retrieve the exposed bottom target into the goal |
| [ ] | `pi05-base_datagen_v1_jar_joint_2cam_lora` | [`IDEAS-Lab-Northwestern/datagen-jar-v1-joint-5cam`](https://huggingface.co/datasets/IDEAS-Lab-Northwestern/datagen-jar-v1-joint-5cam) | close the jar's lid, then carry the closed jar into the goal |
| [ ] | `pi05-base_datagen_v1_lid_joint_2cam_lora` | [`IDEAS-Lab-Northwestern/datagen-lid-v1-joint-5cam`](https://huggingface.co/datasets/IDEAS-Lab-Northwestern/datagen-lid-v1-joint-5cam) | place the lid on the container, then carry the lidded container into the goal |
| [ ] | `pi05-base_datagen_v1_dusty_joint_2cam_lora` | [`IDEAS-Lab-Northwestern/datagen-dusty-v1-joint-5cam`](https://huggingface.co/datasets/IDEAS-Lab-Northwestern/datagen-dusty-v1-joint-5cam) | wipe the dust from the container with the sponge, then pour the food from the carrier into it |

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

`run_sft.sh` first makes sure the dataset is fully on local disk (the first run of a family
downloads it from the Hub with resume + backoff; later runs find the completion marker and start
instantly), then runs a 100-step smoke test, then full training, streaming each checkpoint to the
config's model repo (`IDEAS-Lab-Northwestern/pi05-base-datagen-v1-<family>-joint-2cam-lora`) as
it finalizes. Every config trains 2 epochs and pushes exactly 4 checkpoints (one per half epoch);
relaunching without `--resume` replaces the repo's previous checkpoints (latest-run-wins).
Norm-stats: all six families ship theirs committed under `norm_stats/`, read automatically —
no compute step, every family is turnkey. Run artifacts
(checkpoints, logs) go under `outputs/sft_runs/<exp>/`.

Only `--config` is required; `--exp`, the push repo, and run length default from the config.
Common overrides: `--batch N`, `--steps N`, `--no-push`, `--smoke-only`. Recompute norm-stats
against this exact openpi with `--norm-stats`.

### Throughput

Each run owns **all visible GPUs** (pure data parallelism): `batch_size=256` is the global
batch, sharded 32 samples/card across 8 cards (`fsdp_devices=1`, bf16, ~23 GiB peak/card,
measured ~100% parallel efficiency — 8 cards are 8× one card). No XLA memory env is needed
(JAX's default preallocation is correct). `run_sft.sh` caps per-worker math threads
(`OMP_NUM_THREADS=1` and friends) so the dataloader workers don't thrash the CPU; `num_workers`
(default 48) supplies several times what the GPUs consume, but must stay below the host's
physical core count — check `nproc` before a long run. If the GPUs sawtooth, the bottleneck is
IO (dataset not on local disk), not video decode — see [CLAUDE.md](CLAUDE.md).

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

Append a `TrainConfig` block to `maniguard/openpi_sft/train_configs.py`; it becomes launchable by
name. Committing its `norm_stats/<config>/<repo_id>/norm_stats.json` is OPTIONAL — do it for an
instant-start turnkey run; otherwise ship config-only and `run_sft.sh` computes the norm-stats on
the fly before the first training.
