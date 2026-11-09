from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import torch

from factory6g.models.resource_manager import create_resource_manager
from factory6g.sim.config import load_config
from factory6g.sim.types import ResourceManagerFeedback


def _make_feedback() -> ResourceManagerFeedback:
    h_hat = np.zeros((1, 1, 1, 4, 1, 1, 4), dtype=np.complex64)
    user_vectors = np.array(
        [
            [2.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [1.5 + 0.0j, 0.2 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 1.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.5 + 0.0j, 0.5 + 0.0j, 0.1 + 0.0j],
        ],
        dtype=np.complex64,
    )
    for ut_idx in range(4):
        h_hat[0, 0, 0, ut_idx, 0, 0, :] = user_vectors[ut_idx]
    err_var = np.zeros_like(h_hat, dtype=np.float32)
    return ResourceManagerFeedback(h_hat=torch.as_tensor(h_hat), err_var=torch.as_tensor(err_var))


def test_resource_manager_config_parses_new_fields_without_conftest():
    config_data = {
        "simulation": {
            "gpu_id": 0,
            "force_cpu": True,
            "log_level": "INFO",
            "seed": 1,
            "output_dir": "results",
            "plot_results": False,
        },
        "monte_carlo": {
            "batch_size": 1,
            "min_batches": 1,
            "max_batches": 1,
            "target_block_errors": None,
            "target_ber": None,
            "confidence_level": 0.95,
            "min_total_bits": 0,
            "ebno_min": 0.0,
            "ebno_max": 0.0,
            "ebno_step": 1.0,
        },
        "estimators": {"enabled": ["ls"], "kwargs": {}},
        "resource_managers": {
            "enabled": ["wmmse", "queue_aware", "drl"],
            "cnn_model_path": None,
            "drl_model_path": "models/policy.h5",
            "num_active_users": 2,
            "kwargs": {
                "wmmse": {"iterations": 8},
                "queue_aware": {"arrival_rate": 0.4},
            },
        },
        "factory_scenario": {
            "room_dimensions": [10.0, 10.0, 4.0],
            "num_machines": 1,
            "machine_size_range": [[1.0, 1.5], [1.0, 1.5], [1.0, 1.5]],
            "materials": {
                "metal": {
                    "name": "factory_metal",
                    "relative_permittivity": 1.0,
                    "conductivity": 1e7,
                },
                "concrete": {
                    "name": "factory_concrete",
                    "relative_permittivity": 7.0,
                    "conductivity": 0.1,
                },
            },
        },
        "system": {
            "carrier_frequency": 3.5e9,
            "fft_size": 32,
            "subcarrier_spacing": 30000.0,
            "num_ofdm_symbols": 14,
            "cyclic_prefix_length": 20,
            "pilot_ofdm_symbol_indices": [2, 11],
            "num_bs_ant": 4,
            "num_ut": 4,
            "num_ut_ant": 1,
            "num_bits_per_symbol": 2,
            "coderate": 0.5,
            "num_decoding_iter": 2,
            "channel_model_type": "tr38901",
            "scenario": "umi",
            "direction": "uplink",
            "o2i_model": "low",
            "enable_pathloss": False,
            "enable_shadow_fading": False,
            "min_ut_velocity": 0.0,
            "max_ut_velocity": 0.0,
        },
        "ray_tracing": {"max_depth": 2, "samples_per_src": 128, "max_paths": 8},
        "transceiver": {
            "tx_height_offset": 1.0,
            "rx_height": 1.0,
            "antenna_spacing": 0.5,
            "tx_pattern": "tr38901",
            "tx_polarization": "cross",
            "rx_pattern": "iso",
            "rx_polarization": "V",
            "wall_thickness": 0.2,
            "room_padding": 1.0,
            "rx_boundary_padding": 1.0,
        },
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        config_path = Path(tmp_dir) / "config.json"
        config_path.write_text(json.dumps(config_data), encoding="utf-8")
        config = load_config(config_path)

    assert config.resource_managers.drl_model_path == "models/policy.h5"
    assert config.resource_managers.kwargs["wmmse"]["iterations"] == 8
    assert config.resource_managers.kwargs["queue_aware"]["arrival_rate"] == 0.4


def test_wmmse_manager_returns_two_active_users_with_power():
    manager = create_resource_manager(
        "wmmse",
        num_ut=4,
        num_active=2,
        cnn_model_path=None,
        drl_model_path=None,
        manager_kwargs={"iterations": 6},
    )
    directives = manager.get_runtime_directives({"num_ut": 4}, ebno_db=5.0, feedback=_make_feedback())

    assert directives.active_ut_mask is not None
    assert directives.per_ut_power is not None
    assert sum(directives.active_ut_mask) == 2
    assert directives.active_ut_mask[0] == 1
    assert len(directives.per_ut_power) == 4
    assert all(0.0 <= value <= 1.0 for value in directives.per_ut_power)
    assert any(value > 0.0 for value in directives.per_ut_power)


def test_queue_aware_manager_uses_virtual_arrival_pressure_without_feedback():
    manager = create_resource_manager(
        "queue_aware",
        num_ut=4,
        num_active=1,
        cnn_model_path=None,
        drl_model_path=None,
        manager_kwargs={"arrival_rate": [0.0, 1.0, 0.0, 0.0], "utility_weight": 0.0},
    )
    directives = manager.get_runtime_directives({"num_ut": 4}, ebno_db=0.0, feedback=None)

    assert directives.active_ut_mask == [0, 1, 0, 0]
    assert directives.per_ut_power == [0.0, 1.0, 0.0, 0.0]


def test_drl_manager_heuristic_fallback_returns_valid_directives():
    manager = create_resource_manager(
        "drl",
        num_ut=4,
        num_active=2,
        cnn_model_path=None,
        drl_model_path=None,
        manager_kwargs={"temperature": 0.5, "fairness_weight": 0.4},
    )
    directives = manager.get_runtime_directives({"num_ut": 4}, ebno_db=5.0, feedback=_make_feedback())

    assert directives.active_ut_mask is not None
    assert directives.per_ut_power is not None
    assert sum(directives.active_ut_mask) == 2
    assert directives.active_ut_mask[0] == 1
    assert len(directives.per_ut_power) == 4
    assert all(0.0 <= value <= 1.0 for value in directives.per_ut_power)


def _pf_manager():
    return create_resource_manager(
        "pf",
        num_ut=4,
        num_active=2,
        cnn_model_path=None,
        drl_model_path=None,
        manager_kwargs={},
    )


def test_pf_rate_history_is_kept_per_operating_point():
    """PF state must not leak between Eb/No points.

    A shared history would make the decision at one operating point depend on
    how many batches an unrelated point had already consumed.
    """
    manager = _pf_manager()
    config = {"num_ut": 4}
    feedback = _make_feedback()

    for _ in range(5):
        manager.get_runtime_directives(config, ebno_db=0.0, feedback=feedback)

    assert manager.avg_rates_for(0.0) is not None
    assert manager.avg_rates_for(10.0) is None

    fresh = _pf_manager()
    warmed = manager.get_runtime_directives(config, ebno_db=10.0, feedback=feedback)
    cold = fresh.get_runtime_directives(config, ebno_db=10.0, feedback=feedback)
    assert warmed.active_ut_mask == cold.active_ut_mask

    np.testing.assert_allclose(
        manager.avg_rates_for(10.0),
        fresh.avg_rates_for(10.0),
    )


def test_pf_sweep_order_does_not_change_decisions():
    config = {"num_ut": 4}
    feedback = _make_feedback()
    points = [0.0, 4.0, 8.0]

    forward = _pf_manager()
    for _ in range(3):
        for ebno in points:
            forward.get_runtime_directives(config, ebno_db=ebno, feedback=feedback)

    reverse = _pf_manager()
    for _ in range(3):
        for ebno in reversed(points):
            reverse.get_runtime_directives(config, ebno_db=ebno, feedback=feedback)

    for ebno in points:
        np.testing.assert_allclose(
            forward.avg_rates_for(ebno),
            reverse.avg_rates_for(ebno),
            err_msg=f"PF history at {ebno} dB depends on sweep order",
        )
