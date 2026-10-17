"""System model v2: persistent per-batch terminal positions and large-scale gains."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf
from sionna.phy.ofdm import ResourceGrid

from factory6g.components.antenna import AntennaConfig
from factory6g.components.channel import ChannelModel
from factory6g.components.large_scale import HallLargeScale, inf_sl_path_loss_db

BANK = Path("data/rt_channel_bank.npz")
pytestmark = pytest.mark.skipif(not BANK.exists(), reason="ray-traced channel bank not built")


def test_inf_sl_path_loss_matches_tr38901_table():
    # 10 m at 3.5 GHz: LOS 31.84 + 21.5 + 19 log10(3.5); NLOS 33 + 25.5 + 20 log10(3.5).
    los, nlos = inf_sl_path_loss_db(np.array([10.0, 10.0]), np.array([True, False]), 3.5e9)
    assert los == pytest.approx(31.84 + 21.50 + 19.0 * np.log10(3.5))
    assert nlos == pytest.approx(33.0 + 25.50 + 20.0 * np.log10(3.5))
    # NLOS never below LOS.
    d = np.linspace(1, 20, 50)
    assert np.all(inf_sl_path_loss_db(d, np.zeros(50, bool), 3.5e9) >= inf_sl_path_loss_db(d, np.ones(50, bool), 3.5e9))


def _channel(model: str, split: str = "eval") -> ChannelModel:
    cfg = {
        "channel_model_type": model, "rt_channel_bank": str(BANK), "rt_bank_split": split,
        "large_scale_model": "hall", "num_bs_ant": 8, "num_ut": 4, "num_ut_ant": 1,
        "direction": "uplink", "carrier_frequency": 3.5e9, "tx_polarization": "cross",
        "rx_polarization": "V", "scenario": "umi", "o2i_model": "low",
    }
    rg = ResourceGrid(num_ofdm_symbols=14, fft_size=128, subcarrier_spacing=30e3, num_tx=4,
                      num_streams_per_tx=1, cyclic_prefix_length=20, pilot_pattern="kronecker",
                      pilot_ofdm_symbol_indices=[2, 11])
    return ChannelModel(cfg, AntennaConfig(cfg), rg)


def test_raytraced_hall_is_static_over_batch_and_follows_traced_gain():
    tf.random.set_seed(1)
    ch = _channel("raytraced")
    h = ch.sample_frequency_response(6).numpy()
    assert h.shape == (6, 1, 8, 4, 1, 14, 128)
    np.testing.assert_allclose(h[0], h[5])                 # same positions in every slot
    np.testing.assert_allclose(h[..., 0, :], h[..., 13, :])  # static over the slot
    energy_db = 10 * np.log10(np.mean(np.abs(h[0, 0, :, :, 0, 0, :]) ** 2, axis=(0, 2)))
    np.testing.assert_allclose(energy_db, ch.last_large_scale["gain_db"], atol=1e-3)
    assert len({tuple(p) for p in ch.last_large_scale["positions"]}) == 4


@pytest.mark.parametrize("model", ["rayleigh", "rician", "tr38901"])
def test_statistical_hall_gain_persists_and_sets_per_terminal_energy(model):
    tf.random.set_seed(2)
    ch = _channel(model)
    h = ch.sample_frequency_response(40).numpy()
    gain_db = ch.last_large_scale["gain_db"]
    energy = np.mean(np.abs(h) ** 2, axis=(0, 1, 2, 4, 5, 6))    # [U], batch mean
    # Batch-mean energies follow the persistent gains up to small-scale averaging.
    np.testing.assert_allclose(10 * np.log10(energy / energy.mean()),
                               gain_db - 10 * np.log10(np.mean(10 ** (gain_db / 10))), atol=1.5)
    # Small-scale fading is not normalised per slot: per-slot link energy varies.
    per_slot = np.mean(np.abs(h) ** 2, axis=(1, 2, 4, 5, 6))      # [B, U]
    if model != "tr38901":
        assert np.std(per_slot[:, 0]) / np.mean(per_slot[:, 0]) > 0.1


def test_reference_puts_average_terminal_at_nominal_ebno():
    tf.random.set_seed(3)
    ls = HallLargeScale(BANK, "eval", num_ut=4, carrier_frequency_hz=3.5e9)
    g = np.concatenate([ls.statistical_gain(ls.draw_indices()).numpy() for _ in range(1500)])
    assert np.mean(g) == pytest.approx(1.0, rel=0.1)
    spread = 10 * np.log10(np.percentile(g, 90) / np.percentile(g, 10))
    assert spread > 6.0  # terminals differ by several dB


def test_train_and_eval_share_one_reference():
    a = HallLargeScale(BANK, "train", num_ut=4, carrier_frequency_hz=3.5e9)
    b = HallLargeScale(BANK, "eval", num_ut=4, carrier_frequency_hz=3.5e9)
    assert a.g_ref_statistical == b.g_ref_statistical
    assert a.g_ref_raytraced == b.g_ref_raytraced
