"""ManiGuard pi0.5 LoRA SFT TrainConfigs, registered into pristine openpi.

Two task families (clutter pick-and-place, cabinet pickup), each a fully inline
``TrainConfig`` (model + freeze_filter written out in full, no shared builders)
so the whole recipe is readable at a glance. ``register()`` inserts them into
openpi's ``_CONFIGS_DICT`` at import time, so openpi's ``scripts/train.py`` /
``scripts/compute_norm_stats.py`` resolve them by name when launched via the
wrappers in ``tools/openpi_sft/``; openpi itself is never edited.

JointController pipeline: both configs use ``Sim2CamLiberoDataConfig`` with
``use_delta_joint_actions=True`` (absolute-joint datasets; 7 arm joints ->
per-step delta, gripper absolute) and warm-start from ``pi05_base``.

Scale: batch 128, full data parallelism (``fsdp_devices=1``, model replicated
per device) with ``dtype=bfloat16`` (stable without FSDP parameter sharding).
Steps cover ~2 epochs of each dataset; ``decay_steps == num_train_steps``;
``keep_period = num_train_steps // 5``.
"""

from __future__ import annotations

import openpi.models.pi0_config as pi0_config
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
from openpi.training.config import DataConfig, TrainConfig

from maniguard.openpi_sft.data_configs import Sim2CamLiberoDataConfig

_PI05_BASE = "gs://openpi-assets/checkpoints/pi05_base/params"


def _build_configs() -> list[TrainConfig]:
    return [
        # Sim pnp-clutter (pick the target object out of a cluttered tabletop and
        # move it into the green goal sphere), LIBERO 2-cam, JOINT controller.
        # Dataset: IDEAS-Lab-Northwestern/datagen-clutter-v1-joint-5cam
        #   5-cam rendered (image_opposite/left/right/left_shoulder + wrist_image)
        #   but consumed 2-cam: external_cam="left" overview + wrist_image; the
        #   other views dropped, pi0.5's third image slot zero-filled + masked.
        #   8-D joint state + 8-D absolute-joint action; use_delta_joint_actions=True.
        # warm-start = pi05_base. discrete_state_input=True (pi0.5: the 8-D robot
        #   state is discretized + tokenized into the language prefix).
        # Scale: 2 epochs over the 901,520-frame set at batch 128
        #   (901_520 * 2 / 128 = 14,086 -> rounded up to 14,100).
        TrainConfig(
            name="pi05-base_datagen_v1_clutter_joint_2cam_lora",
            project_name="maniguard-sft",
            policy_metadata={
                "hf_repo": "IDEAS-Lab-Northwestern/pi05-base-datagen-v1-clutter-joint-2cam-lora",
                "hf_private": False,
                "default_exp": "datagen_v1_clutter_joint_2cam",
            },
            model=pi0_config.Pi0Config(
                pi05=True,
                action_dim=32,
                action_horizon=16,
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora",
                dtype="bfloat16",
                discrete_state_input=True,
            ),
            data=Sim2CamLiberoDataConfig(
                repo_id="IDEAS-Lab-Northwestern/datagen-clutter-v1-joint-5cam",
                base_config=DataConfig(prompt_from_task=True),
                use_delta_joint_actions=True,
                external_cam="left",
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader(_PI05_BASE),
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=500,
                peak_lr=7e-5,
                decay_steps=14_100,
                decay_lr=7e-6,
            ),
            num_train_steps=14_100,
            batch_size=128,
            num_workers=48,  # dataloader workers -- primary throughput knob
            log_interval=100,
            fsdp_devices=1,  # full data parallelism, model replicated per device
            keep_period=2_820,  # steps // 5 -> ~5 evenly-spaced checkpoints
            freeze_filter=pi0_config.Pi0Config(
                pi05=True,
                action_dim=32,
                action_horizon=16,
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora",
            ).get_freeze_filter(),
            ema_decay=None,
        ),
        # Sim cabinet-pickup (open the table-top cabinet drawer, put the target
        # object inside, and close it without knocking anything over), LIBERO 2-cam,
        # JOINT controller. Dataset: IDEAS-Lab-Northwestern/datagen-cabinet-v1-joint-5cam
        #   (same 5-cam->2-cam consumption + joint action semantics as clutter above).
        # Scale: 2 epochs over the 4,172,962-frame set at batch 128
        #   (4_172_962 * 2 / 128 = 65,203 -> rounded up to 65,250).
        TrainConfig(
            name="pi05-base_datagen_v1_cabinet_joint_2cam_lora",
            project_name="maniguard-sft",
            policy_metadata={
                "hf_repo": "IDEAS-Lab-Northwestern/pi05-base-datagen-v1-cabinet-joint-2cam-lora",
                "hf_private": False,
                "default_exp": "datagen_v1_cabinet_joint_2cam",
            },
            model=pi0_config.Pi0Config(
                pi05=True,
                action_dim=32,
                action_horizon=16,
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora",
                dtype="bfloat16",
                discrete_state_input=True,
            ),
            data=Sim2CamLiberoDataConfig(
                repo_id="IDEAS-Lab-Northwestern/datagen-cabinet-v1-joint-5cam",
                base_config=DataConfig(prompt_from_task=True),
                use_delta_joint_actions=True,
                external_cam="left",
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader(_PI05_BASE),
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=2_000,
                peak_lr=7e-5,
                decay_steps=65_250,
                decay_lr=7e-6,
            ),
            num_train_steps=65_250,
            batch_size=128,
            num_workers=48,  # dataloader workers -- primary throughput knob
            log_interval=100,
            fsdp_devices=1,  # full data parallelism, model replicated per device
            keep_period=13_050,  # steps // 5 -> ~5 evenly-spaced checkpoints
            freeze_filter=pi0_config.Pi0Config(
                pi05=True,
                action_dim=32,
                action_horizon=16,
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora",
            ).get_freeze_filter(),
            ema_decay=None,
        ),
        # Sim stack-retrieve (unstack the 3 same-object top pile onto a re-stack
        # pile aside, then retrieve the exposed bottom target into the green goal
        # sphere), LIBERO 2-cam, JOINT controller.
        # Dataset: IDEAS-Lab-Northwestern/datagen-stack-v1-joint-5cam (28 base
        #   tasks x 40 = 1120 demos; same 5-cam->2-cam + joint semantics as above).
        # Scale: 2 epochs over the 2,652,083-frame set at batch 128
        #   (2_652_083 * 2 / 128 = 41,439 -> rounded up to 41,500).
        TrainConfig(
            name="pi05-base_datagen_v1_stack_joint_2cam_lora",
            project_name="maniguard-sft",
            policy_metadata={
                "hf_repo": "IDEAS-Lab-Northwestern/pi05-base-datagen-v1-stack-joint-2cam-lora",
                "hf_private": False,
                "default_exp": "datagen_v1_stack_joint_2cam",
            },
            model=pi0_config.Pi0Config(
                pi05=True,
                action_dim=32,
                action_horizon=16,
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora",
                dtype="bfloat16",
                discrete_state_input=True,
            ),
            data=Sim2CamLiberoDataConfig(
                repo_id="IDEAS-Lab-Northwestern/datagen-stack-v1-joint-5cam",
                base_config=DataConfig(prompt_from_task=True),
                use_delta_joint_actions=True,
                external_cam="left",
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader(_PI05_BASE),
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=1_250,
                peak_lr=7e-5,
                decay_steps=41_500,
                decay_lr=7e-6,
            ),
            num_train_steps=41_500,
            batch_size=128,
            num_workers=48,  # dataloader workers -- primary throughput knob
            log_interval=100,
            fsdp_devices=1,  # full data parallelism, model replicated per device
            keep_period=8_300,  # steps // 5 -> ~5 evenly-spaced checkpoints
            freeze_filter=pi0_config.Pi0Config(
                pi05=True,
                action_dim=32,
                action_horizon=16,
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora",
            ).get_freeze_filter(),
            ema_decay=None,
        ),
        # Sim jar-transport (close a hinged jar's lid, then carry the closed jar
        # into the green goal sphere on the table), LIBERO 2-cam, JOINT controller.
        # Dataset: IDEAS-Lab-Northwestern/datagen-jar-v1-joint-5cam (26 base
        #   tasks x 40 = 1040 demos; same 5-cam->2-cam + joint semantics as above).
        # Scale: 2 epochs over the 946,870-frame set at batch 128
        #   (946_870 * 2 / 128 = 14,795 -> rounded up to 14,800).
        TrainConfig(
            name="pi05-base_datagen_v1_jar_joint_2cam_lora",
            project_name="maniguard-sft",
            policy_metadata={
                "hf_repo": "IDEAS-Lab-Northwestern/pi05-base-datagen-v1-jar-joint-2cam-lora",
                "hf_private": False,
                "default_exp": "datagen_v1_jar_joint_2cam",
            },
            model=pi0_config.Pi0Config(
                pi05=True,
                action_dim=32,
                action_horizon=16,
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora",
                dtype="bfloat16",
                discrete_state_input=True,
            ),
            data=Sim2CamLiberoDataConfig(
                repo_id="IDEAS-Lab-Northwestern/datagen-jar-v1-joint-5cam",
                base_config=DataConfig(prompt_from_task=True),
                use_delta_joint_actions=True,
                external_cam="left",
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader(_PI05_BASE),
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=450,
                peak_lr=7e-5,
                decay_steps=14_800,
                decay_lr=7e-6,
            ),
            num_train_steps=14_800,
            batch_size=128,
            num_workers=48,  # dataloader workers -- primary throughput knob
            log_interval=100,
            fsdp_devices=1,  # full data parallelism, model replicated per device
            keep_period=2_960,  # steps // 5 -> ~5 evenly-spaced checkpoints
            freeze_filter=pi0_config.Pi0Config(
                pi05=True,
                action_dim=32,
                action_horizon=16,
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora",
            ).get_freeze_filter(),
            ema_decay=None,
        ),
    ]


def register() -> None:
    """Insert the ManiGuard TrainConfigs into openpi's ``_CONFIGS_DICT``.

    Idempotent: re-registering overwrites by name. Must run before openpi's
    ``config.cli()`` / ``get_config()`` are called (the wrappers in
    ``tools/openpi_sft/`` import this package first, which triggers it).
    """
    from openpi.training.config import _CONFIGS_DICT

    for cfg in _build_configs():
        _CONFIGS_DICT[cfg.name] = cfg
