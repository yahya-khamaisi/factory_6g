from __future__ import annotations

import numpy as np
import pytest
import torch

from factory6g.models.resource_manager import ResourceDirectives
from factory6g.sim.config import ConfigError, load_config
from factory6g.sim.industry import IndustryConfig, IndustryScenario
from factory6g.sim.stages import resource_managers as rm_stage
from factory6g.sim.trace import TRACE_FILENAME, read_trace

from .conftest import make_tiny_config, set_all_seeds, write_config

RAW = {
    "device_classes": ["safety_wearable", "agv_control"],
    "body_blockage": {"probability": 0.5, "attenuation_db": 20.0},
    "events": [
        {"type": "bs_port_fault", "start": 2, "end": 3, "ports": [0, 1]},
        {"type": "ut_blockage", "start": 1, "end": 1, "ut": 1, "attenuation_db": 10.0},
        {"type": "interference_burst", "start": 4, "end": 4, "sinr_loss_db": 6.0},
    ],
}


def _scenario(raw=RAW, seed=1) -> IndustryScenario:
    return IndustryScenario(IndustryConfig.from_dict(raw), num_ut=2, num_bs_ant=4, seed=seed)


def test_config_validation():
    with pytest.raises(ValueError, match="Unknown device class"):
        IndustryConfig.from_dict({"device_classes": ["toaster"]})
    with pytest.raises(ValueError, match="Unknown event type"):
        IndustryConfig.from_dict({"device_classes": ["agv_control"], "events": [{"type": "meteor", "start": 0, "end": 1}]})
    with pytest.raises(ValueError, match="terminals"):
        IndustryScenario(IndustryConfig.from_dict({"device_classes": ["agv_control"]}), num_ut=2, num_bs_ant=4, seed=0)


def test_batch_state_is_repeatable_and_only_blocks_body_worn_devices():
    a, b = _scenario(), _scenario()
    states = [a.batch_state(i) for i in range(40)]
    assert states == [b.batch_state(i) for i in range(40)]
    assert all(not s["blocked"][1] for s in states)  # the AGV is not body-worn
    assert 5 < sum(s["blocked"][0] for s in states) < 35  # about half the batches for the wearable
    assert states[2]["ports"] == [0.0, 0.0, 1.0, 1.0] and "bs_port_fault" in states[2]["events"]
    assert states[1]["attenuation_db"][1] == pytest.approx(10.0)
    assert states[4]["sinr_loss_db"] == pytest.approx(6.0)


def test_channel_hook_scales_the_right_terminal_and_ports():
    scen = _scenario({"device_classes": ["agv_control", "agv_control"], "events": [{"type": "bs_port_fault", "start": 0, "end": 0, "ports": [3]}, {"type": "ut_blockage", "start": 0, "end": 0, "ut": 0, "attenuation_db": 20.0}]})
    h = torch.ones([1, 1, 4, 2, 1, 2, 3], dtype=torch.complex64)
    out = np.abs(scen.channel_hook(scen.batch_state(0))(h).numpy())
    assert out[0, 0, 3].max() == 0.0
    assert out[0, 0, 0, 0].max() == pytest.approx(0.1)
    assert out[0, 0, 0, 1].max() == pytest.approx(1.0)


def test_power_cap_for_body_worn_classes():
    scen = _scenario()
    capped = scen.cap_power(ResourceDirectives(active_ut_mask=[1, 1], per_ut_power=None))
    assert capped.per_ut_power == [0.5, 1.0]
    capped = scen.cap_power(ResourceDirectives(active_ut_mask=[1, 0], per_ut_power=[1.0, 0.0]))
    assert capped.per_ut_power == [0.5, 0.0]


def test_config_section_is_optional(tmp_path):
    data = make_tiny_config(str(tmp_path / "r"))
    assert load_config(write_config(tmp_path, data)).industry is None
    data["industry"] = {"device_classes": ["toaster", "agv_control"]}
    with pytest.raises(ConfigError):
        load_config(write_config(tmp_path, data))


def _run(tmp_path, industry):
    tmp_path.mkdir(parents=True, exist_ok=True)
    data = make_tiny_config(str(tmp_path / "results"))
    data["simulation"]["trace"] = True
    data["monte_carlo"].update({"min_batches": 3, "max_batches": 3, "ebno_min": 6.0, "ebno_max": 6.0})
    data["resource_managers"]["enabled"] = ["round_robin"]
    if industry is not None:
        data["industry"] = industry
    config = load_config(write_config(tmp_path, data))
    set_all_seeds(1234)
    return rm_stage.run_resource_manager_stage(config, checkpoint_dir=tmp_path / "stage")


@pytest.mark.slow
def test_quiet_scenario_leaves_the_real_stage_unchanged_and_a_full_port_fault_breaks_it(tmp_path):
    quiet = {"device_classes": ["agv_control", "agv_control"]}
    base = _run(tmp_path / "a", None)
    same = _run(tmp_path / "b", quiet)
    assert same["methods"]["round_robin"]["bit_errors"] == base["methods"]["round_robin"]["bit_errors"]
    broken = _run(tmp_path / "c", {**quiet, "events": [{"type": "bs_port_fault", "start": 0, "end": 2, "ports": [0, 1, 2, 3]}]})
    assert broken["methods"]["round_robin"]["ber"][0] > 0.3  # no signal reaches the base station
    trace = read_trace(tmp_path / "c" / "stage" / TRACE_FILENAME)
    assert trace["header"]["meta"]["industry"]["device_classes"] == ["agv_control", "agv_control"]
    assert trace["contexts"][0]["industry"]["ports"] == [0.0, 0.0, 0.0, 0.0]
