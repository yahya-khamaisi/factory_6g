"""Analytic anchors: the real channel and receiver code against closed-form results.

These pin the PHY to textbook results before any method comparison is trusted:

* the v2 Rayleigh channel has unit mean energy per antenna;
* uncoded QPSK over AWGN with 8-antenna combining matches Q(sqrt(2 L Eb/N0));
* uncoded QPSK over Rayleigh with 8-branch combining matches the closed-form
  L-branch MRC BER (Proakis, Digital Communications, eq. 14.4-15), at two
  noise levels, which also checks the diversity slope.

Hard decisions on the equalised symbols of a perfect-CSI receiver give the
uncoded BER; with one terminal the LMMSE equaliser is MRC up to a positive
scale, so the decisions are the MRC decisions. The terminal's large-scale gain
is pinned to 0 dB so only small-scale fading reaches the receiver.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from factory6g.models.model import Model
from factory6g.sim.config import load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
NUM_ANT = 8

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not (REPO_ROOT / "data" / "rt_channel_bank.npz").exists(), reason="the v2 channel draws positions from the bank"),
]


def _model(tmp_path: Path, channel: str) -> Model:
    raw = json.loads((REPO_ROOT / "config" / "config.json").read_text(encoding="utf-8"))
    raw["system"].update({"num_ut": 1, "num_bs_ant": NUM_ANT, "channel_model_type": channel, "large_scale_model": "hall"})
    path = tmp_path / "anchor_config.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    config = load_config(path)
    model = Model(config=config.system_runtime_config, estimator_type="ls", perfect_csi=True)
    if model._channel._large_scale is not None:
        model._channel._large_scale.statistical_gain = lambda idx: tf.ones([1], tf.float32)
    return model


def _uncoded_ber(model: Model, ebno_db: float, *, batches: int, batch_size: int) -> tuple[float, float]:
    errors = bits = 0
    for _ in range(batches):
        ctx = model.prepare_batch_context(batch_size=batch_size, ebno_db=ebno_db, include_feedback=False)
        out = model.run_batch(ctx, include_details=True)
        x, x_hat = np.asarray(out["qam"]), np.asarray(out["qam_hat"])
        errors += int(np.sum(np.sign(x.real) != np.sign(x_hat.real)) + np.sum(np.sign(x.imag) != np.sign(x_hat.imag)))
        bits += 2 * x.size
        no = float(ctx.noise_variance)
    return errors / bits, no


def _q(x: float) -> float:
    return 0.5 * math.erfc(x / math.sqrt(2.0))


def mrc_rayleigh_ber(gamma_b: float, branches: int) -> float:
    """Average BER of BPSK (each QPSK bit) with L-branch MRC in i.i.d. Rayleigh fading."""
    mu = math.sqrt(gamma_b / (1.0 + gamma_b))
    total = sum(math.comb(branches - 1 + k, k) * ((1 + mu) / 2) ** k for k in range(branches))
    return ((1 - mu) / 2) ** branches * total


def test_closed_form_reduces_to_single_branch():
    gamma = 3.0
    assert mrc_rayleigh_ber(gamma, 1) == pytest.approx(0.5 * (1 - math.sqrt(gamma / (1 + gamma))))


def test_rayleigh_channel_has_unit_energy_per_antenna(tmp_path):
    model = _model(tmp_path, "rayleigh")
    ctx = model.prepare_batch_context(batch_size=500, ebno_db=0.0, include_feedback=False)
    energy = np.abs(np.asarray(ctx.h_freq)[:, 0, :, 0, 0, 0, 0]) ** 2  # one value per slot and antenna (block fading)
    assert energy.size == 500 * NUM_ANT
    assert energy.mean() == pytest.approx(1.0, abs=0.06)
    # Rayleigh amplitude: the per-antenna energy is exponential, so its variance equals its mean squared.
    assert energy.var() == pytest.approx(1.0, abs=0.15)


def test_awgn_combining_matches_closed_form(tmp_path):
    model = _model(tmp_path, "awgn")
    ber, no = _uncoded_ber(model, 0.0, batches=2, batch_size=100)
    expected = _q(math.sqrt(2.0 * NUM_ANT / (2.0 * no)))  # Eb/N0 per branch is 1/(2 no) for unit-energy QPSK
    assert ber == pytest.approx(expected, rel=0.10)


@pytest.mark.parametrize("ebno_db", [0.0, 3.0])
def test_rayleigh_mrc_matches_closed_form(tmp_path, ebno_db):
    model = _model(tmp_path, "rayleigh")
    ber, no = _uncoded_ber(model, ebno_db, batches=6, batch_size=500)
    expected = mrc_rayleigh_ber(1.0 / (2.0 * no), NUM_ANT)
    # Fading dominates the Monte Carlo spread: 3000 independent draws keep it near 10 %.
    assert ber == pytest.approx(expected, rel=0.20)
