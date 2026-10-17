from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_CHECKPOINT_NAME = "checkpoint.json"
_CHECKPOINT_TMP = "checkpoint.json.tmp"


def save_checkpoint(checkpoint_dir: Path, data: dict[str, Any]) -> None:
    """Atomically write checkpoint data to disk.

    Writes to a temporary file first, then replaces the checkpoint in one
    os.replace() call so a mid-write kill cannot corrupt the checkpoint.
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = checkpoint_dir / _CHECKPOINT_TMP
    dst_path = checkpoint_dir / _CHECKPOINT_NAME
    with tmp_path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh)
    tmp_path.replace(dst_path)


def load_checkpoint(checkpoint_dir: Path) -> dict[str, Any] | None:
    """Load checkpoint from disk; returns None if none exists."""
    path = checkpoint_dir / _CHECKPOINT_NAME
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def delete_checkpoint(checkpoint_dir: Path) -> None:
    """Remove checkpoint file after a stage completes successfully."""
    path = checkpoint_dir / _CHECKPOINT_NAME
    if path.exists():
        path.unlink()


def reseed_for_resume(seed: int, start_batch_index: int) -> int:
    """Reseed every random generator before continuing a resumed stage.

    The checkpoint does not store generator state, and the CLI seeds all
    generators with the global seed at start-up. Without reseeding, a resumed
    stage would redraw the channel, noise and bit realisations of the first
    batches. Offsetting the seed by the resume batch index gives the remaining
    batches fresh draws while keeping a resumed run reproducible.
    """
    import random

    import numpy as np
    import sionna
    import tensorflow as tf

    resume_seed = int(seed) + int(start_batch_index)
    random.seed(resume_seed)
    np.random.seed(resume_seed)
    tf.random.set_seed(resume_seed)
    sionna.phy.config.seed = resume_seed
    return resume_seed
