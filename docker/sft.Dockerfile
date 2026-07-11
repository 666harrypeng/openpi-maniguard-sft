# Training image for the pi0.5 LoRA SFT configs in this repo.
# Inherits openpi's serve-image dependency recipe (base image, uv, python,
# uv.lock) and adds the project code + a training entrypoint. Uses the JAX
# trainer (scripts/train.py).
#
# Build: docker build -t maniguard-sft -f docker/sft.Dockerfile .
# Run:   docker run --gpus all -e HF_TOKEN -e WANDB_API_KEY \
#          -v "$PWD/outputs:/app/outputs" maniguard-sft \
#          --config pi05-base_datagen_v1_clutter_joint_2cam_lora

FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04@sha256:2d913b09e6be8387e1a10976933642c73c840c0b735f0bf3c28d97fc9bc422e0
COPY --from=ghcr.io/astral-sh/uv:0.5.1 /uv /uvx /bin/

WORKDIR /app

# git-lfs: LeRobot is pulled as a git dependency during the sync.
RUN apt-get update && apt-get install -y git git-lfs build-essential clang && \
    rm -rf /var/lib/apt/lists/*

# uv: copy link mode (volume-safe), venv written outside /app, never mutate the
# lockfile, and the venv on PATH so `python` resolves to the venv interpreter.
ENV UV_LINK_MODE=copy
ENV UV_PROJECT_ENVIRONMENT=/.venv
ENV UV_FROZEN=1
ENV PATH="/.venv/bin:${PATH}"
# Cap per-worker math threads so the dataloader's num_workers processes don't
# oversubscribe the CPU (see run_sft.sh for the full rationale). run_sft.sh also
# sets these with := defaults; the ENV makes them authoritative for any entry.
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1

# Install dependencies first, in a cached layer that only busts when the
# lockfile / client package changes.
COPY pyproject.toml uv.lock ./
COPY packages/openpi-client/pyproject.toml packages/openpi-client/pyproject.toml
COPY packages/openpi-client/src packages/openpi-client/src
RUN uv venv --python 3.11.9 $UV_PROJECT_ENVIRONMENT && \
    GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen --no-install-project --no-dev

# NOTE: openpi's PyTorch trainer needs a transformers_replace copy step here;
# this image uses the JAX trainer (scripts/train.py), which does not.

# Copy the rest of the project (openpi src, maniguard configs, tools, norm_stats)
# and install openpi so `import openpi` resolves; maniguard resolves via the
# launcher's sys.path insert.
COPY . /app
RUN GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen --no-dev

ENTRYPOINT ["bash", "tools/openpi_sft/run_sft.sh"]
