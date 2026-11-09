"""One random source for the PyTorch stack (Phase 4).

Every draw in the simulation (channel, noise, positions, bits) comes from
Sionna's per-device ``torch.Generator``, so ``seed_all`` makes a run
reproducible end to end, as ``tf.random.set_seed`` did on the TensorFlow stack.
"""

from __future__ import annotations

import random
from contextlib import contextmanager

import numpy as np
import torch


def rng(device: str | None = None) -> torch.Generator:
    from sionna.phy import config

    return config.torch_rng(device)


def seed_all(seed: int) -> None:
    from sionna.phy import config

    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    config.seed = seed


# Seed of the pilot sequences. Sionna 1.2.1 drew Kronecker pilots from a fixed
# seed (0); Sionna 2.x draws them from the shared generator, which would give
# every Model its own pilots and shift every later draw of the run.
PILOT_SEED = 0


@contextmanager
def isolated_draws(seed: int, device: str | None = None):
    """Run a block on a fixed seed without moving the shared generator."""
    g = rng(device)
    state = g.get_state()
    g.manual_seed(int(seed))
    try:
        yield
    finally:
        g.set_state(state)
