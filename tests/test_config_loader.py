from __future__ import annotations

import pytest

from factory6g.sim.config import ConfigError, load_config

from .conftest import make_tiny_config, write_config


def test_valid_config_parses_into_normalized_object(tmp_path):
    config_path = write_config(tmp_path, make_tiny_config(str(tmp_path / "results")))
    config = load_config(config_path)
    assert config.system.scenario == "umi"
    assert config.monte_carlo.ebno_db_range == [0.0]
    assert config.monte_carlo.stop_policy == "sweep"
    assert config.system_runtime_config["tx_pattern"] == "tr38901"


def test_missing_required_keys_fail_fast(tmp_path):
    config_data = make_tiny_config(str(tmp_path / "results"))
    del config_data["system"]["scenario"]
    config_path = write_config(tmp_path, config_data)
    with pytest.raises(ConfigError):
        load_config(config_path)


def test_unknown_keys_fail_fast(tmp_path):
    config_data = make_tiny_config(str(tmp_path / "results"))
    config_data["simulation"]["legacy"] = True
    config_path = write_config(tmp_path, config_data)
    with pytest.raises(ConfigError):
        load_config(config_path)


def test_locked_5g_scenario_rejects_other_values(tmp_path):
    config_data = make_tiny_config(str(tmp_path / "results"))
    config_data["system"]["scenario"] = "rma"
    config_path = write_config(tmp_path, config_data)
    with pytest.raises(ConfigError):
        load_config(config_path)


def test_simulation_targets_are_rejected(tmp_path):
    config_data = make_tiny_config(str(tmp_path / "results"))
    config_data["simulation"]["targets"] = ["estimators"]
    config_path = write_config(tmp_path, config_data)
    with pytest.raises(ConfigError):
        load_config(config_path)


def test_non_5g_radio_preset_values_are_rejected(tmp_path):
    config_data = make_tiny_config(str(tmp_path / "results"))
    # Subcarrier spacing is part of the locked 5G NR numerology.
    config_data["system"]["subcarrier_spacing"] = 15000.0
    config_path = write_config(tmp_path, config_data)
    with pytest.raises(ConfigError):
        load_config(config_path)


def test_supported_channel_models_are_accepted(tmp_path):
    for channel in ("rayleigh", "rician", "tr38901"):
        config_data = make_tiny_config(str(tmp_path / f"results_{channel}"))
        config_data["system"]["channel_model_type"] = channel
        config_path = write_config(tmp_path, config_data)
        assert load_config(config_path).system.channel_model_type == channel


def test_non_5g_tx_pattern_is_rejected(tmp_path):
    config_data = make_tiny_config(str(tmp_path / "results"))
    config_data["transceiver"]["tx_pattern"] = "dipole"
    config_path = write_config(tmp_path, config_data)
    with pytest.raises(ConfigError):
        load_config(config_path)


def test_resource_manager_kwargs_and_model_paths_parse(tmp_path):
    config_data = make_tiny_config(str(tmp_path / "results"))
    config_data["resource_managers"]["drl_model_path"] = "models/drl_policy.h5"
    config_data["resource_managers"]["kwargs"] = {
        "wmmse": {"iterations": 8},
        "queue_aware": {"arrival_rate": 0.4},
    }
    config_path = write_config(tmp_path, config_data)
    config = load_config(config_path)
    assert config.resource_managers.drl_model_path == "models/drl_policy.h5"
    assert config.resource_managers.kwargs["wmmse"]["iterations"] == 8
    assert config.resource_managers.kwargs["queue_aware"]["arrival_rate"] == 0.4


def test_threshold_stop_policy_requires_target_ber(tmp_path):
    config_data = make_tiny_config(str(tmp_path / "results"))
    config_data["monte_carlo"]["stop_policy"] = "threshold"
    config_data["monte_carlo"]["target_ber"] = None
    config_path = write_config(tmp_path, config_data)
    with pytest.raises(ConfigError):
        load_config(config_path)


def test_threshold_stop_policy_with_target_ber_parses(tmp_path):
    config_data = make_tiny_config(str(tmp_path / "results"))
    config_data["monte_carlo"]["stop_policy"] = "threshold"
    config_data["monte_carlo"]["target_ber"] = 1e-5
    config_path = write_config(tmp_path, config_data)
    config = load_config(config_path)
    assert config.monte_carlo.stop_policy == "threshold"
    assert config.monte_carlo.target_ber == 1e-5
