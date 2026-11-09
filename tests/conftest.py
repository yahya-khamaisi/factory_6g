from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import torch

from factory6g.sim.rng import seed_all


def set_all_seeds(seed: int = 1234) -> None:
    seed_all(seed)


def make_tiny_config(output_dir: str) -> dict:
    return {
        "simulation": {
            "gpu_id": 0,
            "force_cpu": True,
            "log_level": "INFO",
            "seed": 1234,
            "output_dir": output_dir,
            "plot_results": False,
        },
        "monte_carlo": {
            "batch_size": 1,
            "min_batches": 1,
            "max_batches": 1,
            "target_block_errors": None,
            "target_ber": None,
            "stop_policy": "sweep",
            "confidence_level": 0.95,
            "min_total_bits": 0,
            "ebno_min": 0.0,
            "ebno_max": 0.0,
            "ebno_step": 1.0,
        },
        "estimators": {
            "enabled": ["ls"],
            "kwargs": {},
        },
        "resource_managers": {
            "enabled": ["static", "round_robin"],
            "cnn_model_path": None,
            "drl_model_path": None,
            "num_active_users": 1,
            "kwargs": {},
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
            "num_ut": 2,
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
        "ray_tracing": {
            "max_depth": 2,
            "samples_per_src": 128,
            "max_paths": 8,
        },
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


def write_config(tmp_path: Path, config_data: dict) -> Path:
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config_data, indent=2), encoding="utf-8")
    return config_path


@pytest.fixture(autouse=True)
def _reset_rng_state():
    set_all_seeds(1234)
    yield
