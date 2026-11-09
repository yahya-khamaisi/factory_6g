import numpy as np
import torch

from factory6g.sim.rng import rng, seed_all
from factory6g.sim.stages.checkpoint import reseed_for_resume


def _draw():
    return float(torch.rand([], generator=rng('cpu'))), float(np.random.rand())


def test_resume_reseed_gives_fresh_but_reproducible_draws():
    seed_all(42)
    fresh_start = _draw()

    assert reseed_for_resume(42, 745) == 787
    resumed = _draw()
    assert resumed != fresh_start

    reseed_for_resume(42, 745)
    assert _draw() == resumed
