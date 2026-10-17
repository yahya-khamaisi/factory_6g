from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf
from sionna.phy.ofdm import ResourceGrid

from factory6g.components import raytracing as rt
from factory6g.components.antenna import AntennaConfig
from factory6g.components.channel import ChannelModel
from factory6g.sim.config import ConfigError, load_config
from factory6g.visualization import thesis_methodology_figures as figs

from .conftest import make_tiny_config, write_config


def test_machine_layout_matches_thesis_plan_figure():
    traced = rt.sample_machine_layout(
        figs.ROOM_DIMENSIONS,
        num_machines=figs.NUM_MACHINES,
        size_range=figs.MACHINE_SIZE_RANGE,
        padding=figs.ROOM_PADDING,
        seed=figs.MACHINE_SEED,
    )
    keys = ("x", "y", "sx", "sy", "sz")
    assert [{k: m[k] for k in keys} for m in traced] == [{k: m[k] for k in keys} for m in figs.sample_machine_layout()]


def test_terminal_positions_avoid_machines():
    machines = [{"x": 0.0, "y": 0.0, "z": 0.0, "sx": 4.0, "sy": 4.0, "sz": 1.0}]
    pos = rt.sample_terminal_positions(
        np.random.default_rng(0), 500,
        room_dimensions=(15.0, 15.0, 5.0), boundary_padding=1.0, height=1.0, machines=machines,
    )
    assert pos.shape == (500, 3)
    assert np.all((np.abs(pos[:, 0]) > 2.3) | (np.abs(pos[:, 1]) > 2.3))
    assert np.all(np.abs(pos[:, :2]) <= 6.5)


def test_checkerboard_split_keeps_guard_distance():
    rng = np.random.default_rng(1)
    pos = np.column_stack([rng.uniform(-7, 7, 4000), rng.uniform(-7, 7, 4000), np.ones(4000)])
    split = rt.checkerboard_split(pos, cell=1.5, guard=0.25)
    train, test = pos[split == 0, :2], pos[split == 1, :2]
    assert len(train) and len(test)
    d = np.min(np.linalg.norm(test[:, None, :] - train[None, :, :], axis=-1))
    assert d >= 0.5 - 1e-9


def _write_bank(path, n=32, m=8, f=128):
    rng = np.random.default_rng(3)
    h = (rng.normal(size=(n, m, f)) + 1j * rng.normal(size=(n, m, f))).astype(np.complex64) * 1e-3
    np.savez(path, train_h=h, eval_h=2 * h)


def _channel(tmp_path, split="eval"):
    bank = tmp_path / "bank.npz"
    _write_bank(bank)
    cfg = {
        "channel_model_type": "raytraced", "rt_channel_bank": str(bank), "rt_bank_split": split,
        "large_scale_model": "none",
        "num_bs_ant": 8, "num_ut": 4, "num_ut_ant": 1, "direction": "uplink",
        "carrier_frequency": 3.5e9, "tx_polarization": "cross", "rx_polarization": "V",
    }
    rg = ResourceGrid(num_ofdm_symbols=14, fft_size=128, subcarrier_spacing=30e3, num_tx=4,
                      num_streams_per_tx=1, cyclic_prefix_length=20, pilot_pattern="kronecker",
                      pilot_ofdm_symbol_indices=[2, 11])
    return ChannelModel(cfg, AntennaConfig(cfg), rg)


def test_v1_raytraced_channel_shape_and_link_normalisation(tmp_path):
    tf.random.set_seed(0)
    h = _channel(tmp_path).sample_frequency_response(5).numpy()
    assert h.shape == (5, 1, 8, 4, 1, 14, 128)
    energy = np.mean(np.abs(h) ** 2, axis=(2, 4, 5, 6))
    np.testing.assert_allclose(energy, 1.0, rtol=1e-4)
    # Static terminals: identical response on every OFDM symbol.
    np.testing.assert_allclose(h[..., 0, :], h[..., 13, :])


def test_raytraced_channel_rejects_unknown_split(tmp_path):
    with pytest.raises(KeyError):
        _channel(tmp_path, split="validation")


def test_config_accepts_raytraced_and_validates_split(tmp_path):
    data = make_tiny_config(str(tmp_path / "results"))
    data["system"]["channel_model_type"] = "raytraced"
    config = load_config(write_config(tmp_path, data))
    assert config.system_runtime_config["rt_bank_split"] == "eval"
    data["system"]["rt_bank_split"] = "test"
    with pytest.raises(ConfigError):
        load_config(write_config(tmp_path, data))
