"""The neural estimator must be conditioned on the Eb/No its training data used."""

import numpy as np
import pytest
from sionna.phy.utils import ebnodb2no

from factory6g.components.estimators.neural_estimator import NeuralChannelEstimator
from factory6g.models.model import Model
from factory6g.sim.config import load_config


@pytest.mark.parametrize("ebno_db", [0.0, 6.0, 20.0])
def test_ebno_recovered_from_noise_matches_operating_point(ebno_db):
    cfg = load_config("config/config.json").system_runtime_config
    rg = Model(cfg, estimator_type="ls")._rg
    est = NeuralChannelEstimator(rg, config=cfg, model_path="does/not/exist.keras")
    no = ebnodb2no(ebno_db, cfg["num_bits_per_symbol"], cfg["coderate"], rg)
    assert float(est.ebno_db_from_noise(no)) == pytest.approx(ebno_db, abs=1e-3)
