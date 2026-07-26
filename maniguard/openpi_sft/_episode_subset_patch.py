"""Load-time per-base-task episode subsetting for the data-scaling ablation.

Wraps ``openpi.training.data_loader.create_torch_dataset``: when the active
``DataConfig`` carries an ``episode_fraction`` (set by
``Sim2CamLiberoDataConfig(episode_fraction=...)`` via ``SubsetDataConfig``),
the LeRobot dataset is constructed with its native ``episodes=[...]`` filter so
only the selected episodes are loaded — the on-disk dataset stays READ-ONLY and
untouched. Fraction-less configs take openpi's original code path unchanged.

Selection rule (the finalized datagen layout, verified on clutter + cabinet):
every base task is stored as a CONSECUTIVE, task-homogeneous block of exactly
``EPISODES_PER_BASE_TASK`` (40) episodes. A fraction f keeps the FIRST
``ceil(40 * f)`` episodes of EVERY block (0.2 -> 8/40, 0.5 -> 20/40,
0.8 -> 32/40): task coverage is unchanged, only demos-per-task shrink. Both
layout assumptions are hard-asserted at load time (block count + per-block task
homogeneity) so a dataset that violates them aborts loudly instead of training
on a silently wrong subset.

The norm-stats pass (``scripts/compute_norm_stats.py``) builds its dataloader
through this same function, so each fraction config computes normalization
statistics on ITS OWN subset, stored under its own config name.
"""

from __future__ import annotations

import logging
import math

logger = logging.getLogger(__name__)

# The finalized datagen datasets store exactly 40 episodes per base task,
# written consecutively (task-homogeneous 40-blocks). Asserted at load time.
EPISODES_PER_BASE_TASK = 40


def select_episode_subset(episodes_meta: dict, fraction: float) -> list[int]:
    """Per-base-task first-``ceil(40*fraction)`` episode indices.

    ``episodes_meta``: LeRobotDatasetMetadata.episodes — {episode_index: record}
    with record["tasks"] (list of task strings) and record["length"].
    """
    n = len(episodes_meta)
    if n % EPISODES_PER_BASE_TASK != 0:
        raise ValueError(
            f"episode_fraction requires the {EPISODES_PER_BASE_TASK}-per-base-task layout; "
            f"dataset has {n} episodes (not a multiple of {EPISODES_PER_BASE_TASK})."
        )
    ordered = [episodes_meta[i] for i in sorted(episodes_meta)]
    keep = math.ceil(EPISODES_PER_BASE_TASK * fraction)
    selected: list[int] = []
    for b in range(n // EPISODES_PER_BASE_TASK):
        block = ordered[b * EPISODES_PER_BASE_TASK : (b + 1) * EPISODES_PER_BASE_TASK]
        tasks = {t for rec in block for t in (rec["tasks"] if isinstance(rec["tasks"], list) else [rec["tasks"]])}
        if len(tasks) != 1:
            raise ValueError(
                f"episode block {b} (episodes {block[0]['episode_index']}..{block[-1]['episode_index']}) "
                f"mixes {len(tasks)} task strings — not the per-base-task layout; refusing to subset."
            )
        selected.extend(rec["episode_index"] for rec in block[:keep])
    return selected


def apply() -> None:
    """Install the wrapper around openpi's create_torch_dataset (idempotent)."""
    import openpi.training.data_loader as _dl

    if getattr(_dl.create_torch_dataset, "_maniguard_subset_patch", False):
        return
    _orig = _dl.create_torch_dataset

    def create_torch_dataset(data_config, action_horizon, model_config):
        fraction = getattr(data_config, "episode_fraction", None)
        if fraction is None:
            return _orig(data_config, action_horizon, model_config)

        # Mirror of openpi's original body, plus the episodes= filter.
        import lerobot.common.datasets.lerobot_dataset as lerobot_dataset

        import openpi.transforms as _transforms

        repo_id = data_config.repo_id
        if repo_id is None:
            raise ValueError("Repo ID is not set. Cannot create dataset.")
        dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
        episodes = select_episode_subset(dataset_meta.episodes, fraction)
        frames = sum(dataset_meta.episodes[i]["length"] for i in episodes)
        logger.info(
            "[episode_subset] %s: fraction=%.2f -> %d/%d episodes "
            "(first %d of every %d-block), %d frames",
            repo_id, fraction, len(episodes), len(dataset_meta.episodes),
            math.ceil(EPISODES_PER_BASE_TASK * fraction), EPISODES_PER_BASE_TASK, frames,
        )
        dataset = lerobot_dataset.LeRobotDataset(
            repo_id,
            episodes=episodes,
            delta_timestamps={
                key: [t / dataset_meta.fps for t in range(action_horizon)]
                for key in data_config.action_sequence_keys
            },
        )
        if data_config.prompt_from_task:
            dataset = _dl.TransformedDataset(
                dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)]
            )
        return dataset

    create_torch_dataset._maniguard_subset_patch = True  # type: ignore[attr-defined]
    _dl.create_torch_dataset = create_torch_dataset
