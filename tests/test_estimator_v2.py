"""System model v2: per-terminal neural conditioning and per-terminal adaptive branch."""

from __future__ import annotations

import json

import numpy as np
import pytest
import tensorflow as tf
from sionna.phy.ofdm import ResourceGrid

from factory6g.components.estimators import AdaptiveHybridChannelEstimator, LMMSEChannelEstimator
from factory6g.components.estimators.dft_estimator import DFTChannelEstimator
from factory6g.components.estimators.neural_estimator import (
    COND_BATCH_EBNO,
    LEGACY_CONDITIONING,
    LS_ENERGY_CONDITIONING,
    NeuralChannelEstimator,
    effective_ebno_db_np,
    load_conditioning,
)

# [batch, rx, rx_ant, tx, stream, symbols, subcarriers]
SHAPE = (2, 1, 3, 2, 1, 4, 16)


def _tiny_resource_grid() -> ResourceGrid:
    return ResourceGrid(
        num_ofdm_symbols=4,
        fft_size=16,
        subcarrier_spacing=30000.0,
        num_tx=1,
        num_streams_per_tx=1,
        cyclic_prefix_length=4,
        pilot_pattern="kronecker",
        pilot_ofdm_symbol_indices=[1, 3],
    )


def _two_terminal_ls(power_db: tuple[float, float], seed: int = 7) -> np.ndarray:
    """Unit-modulus random phases scaled per terminal, so each grid has exactly that power."""
    rng = np.random.default_rng(seed)
    h = np.exp(1j * rng.uniform(0, 2 * np.pi, SHAPE)).astype(np.complex64)
    scale = np.sqrt(10.0 ** (np.asarray(power_db) / 10.0)).reshape(1, 1, 1, 2, 1, 1, 1)
    return (h * scale).astype(np.complex64)


def test_effective_ebno_adds_each_terminals_power():
    h_ls = _two_terminal_ls((0.0, 10.0))
    cond = effective_ebno_db_np(h_ls, err_var_mean=0.0, ebno_db=5.0)
    assert cond.shape == SHAPE[:-2]
    np.testing.assert_allclose(cond[:, :, :, 0, :], 5.0, atol=1e-4)
    np.testing.assert_allclose(cond[:, :, :, 1, :], 15.0, atol=1e-4)


def test_effective_ebno_subtracts_ls_noise_and_floors():
    h_ls = _two_terminal_ls((3.0103, -40.0))  # 2.0 and 1e-4 linear
    cond = effective_ebno_db_np(h_ls, err_var_mean=1.0, ebno_db=0.0)
    # 2.0 - 1.0 = 1.0 -> 0 dB; the second terminal is floored at -30 dB.
    np.testing.assert_allclose(cond[:, :, :, 0, :], 0.0, atol=1e-3)
    np.testing.assert_allclose(cond[:, :, :, 1, :], -30.0, atol=1e-3)


def test_tensorflow_conditioning_matches_numpy():
    rg = _tiny_resource_grid()
    est = NeuralChannelEstimator(rg, model_path="does/not/exist.keras")
    h_ls = _two_terminal_ls((-2.0, 7.0))
    err = np.full(SHAPE, 0.1, dtype=np.float32)
    expected = effective_ebno_db_np(h_ls, err_var_mean=0.1, ebno_db=4.0)
    got = est.effective_ebno_db(tf.constant(h_ls), tf.constant(err), tf.constant(4.0)).numpy()
    np.testing.assert_allclose(got, expected, atol=1e-4)


def test_conditioning_is_read_from_metadata(tmp_path):
    model_path = str(tmp_path / "net.keras")
    assert load_conditioning(model_path) == LEGACY_CONDITIONING
    with open(model_path + ".metadata.json", "w") as fh:
        json.dump({"conditioning": LS_ENERGY_CONDITIONING}, fh)
    assert load_conditioning(model_path) == LS_ENERGY_CONDITIONING
    with open(model_path + ".metadata.json", "w") as fh:
        json.dump({"test_nmse": []}, fh)  # older metadata without the key
    assert load_conditioning(model_path)["input"] == COND_BATCH_EBNO


def test_adaptive_branch_is_chosen_per_terminal():
    rg = _tiny_resource_grid()
    est = AdaptiveHybridChannelEstimator(rg, quality_low=3.0, quality_high=12.0)
    no = tf.constant(1.0, tf.float32)
    # Terminal 0 at quality 1 (low), terminal 1 at quality 100 (high).
    h_ls = tf.constant(_two_terminal_ls((0.0, 20.0)))
    err = tf.ones(SHAPE, tf.float32)
    h_hat, err_var, branches = est.estimate_from_ls(h_ls, err, no)
    assert branches == ["low", "high"]

    h_lmmse, e_lmmse = LMMSEChannelEstimator(rg).estimate_from_ls(h_ls, err, no)
    h_dft, e_dft = DFTChannelEstimator(rg).estimate_from_ls(h_ls, err)
    np.testing.assert_allclose(h_hat.numpy()[:, :, :, 0], h_lmmse.numpy()[:, :, :, 0], atol=1e-6)
    np.testing.assert_allclose(h_hat.numpy()[:, :, :, 1], h_dft.numpy()[:, :, :, 1], atol=1e-6)
    ev = np.broadcast_to(err_var.numpy(), SHAPE)
    np.testing.assert_allclose(ev[:, :, :, 0], np.broadcast_to(e_lmmse.numpy(), SHAPE)[:, :, :, 0], atol=1e-6)
    np.testing.assert_allclose(ev[:, :, :, 1], np.broadcast_to(e_dft.numpy(), SHAPE)[:, :, :, 1], atol=1e-6)


def test_adaptive_single_branch_batch_matches_pure_estimator():
    rg = _tiny_resource_grid()
    est = AdaptiveHybridChannelEstimator(rg, quality_low=3.0, quality_high=12.0)
    h_ls = tf.constant(_two_terminal_ls((0.0, 1.0)))
    err = tf.ones(SHAPE, tf.float32)
    no = tf.constant(1.0, tf.float32)
    h_hat, _, branches = est.estimate_from_ls(h_ls, err, no)
    assert branches == ["low", "low"]
    h_lmmse, _ = LMMSEChannelEstimator(rg).estimate_from_ls(h_ls, err, no)
    np.testing.assert_array_equal(h_hat.numpy(), h_lmmse.numpy())


@pytest.mark.parametrize("per_terminal", [True, False])
def test_adaptive_v1_rule_still_available(per_terminal):
    rg = _tiny_resource_grid()
    est = AdaptiveHybridChannelEstimator(rg, per_terminal_branch=per_terminal)
    h_ls = tf.constant(_two_terminal_ls((0.0, 20.0)))
    _, _, branches = est.estimate_from_ls(h_ls, tf.ones(SHAPE, tf.float32), tf.constant(1.0, tf.float32))
    assert isinstance(branches, list) == per_terminal
