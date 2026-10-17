"""Modelled latency, allocated transmit power and Jain's fairness index."""

import math

import numpy as np
import pytest

from factory6g.models.model import Model
from factory6g.models.resource_manager import ResourceDirectives
from factory6g.sim.config import load_config
from factory6g.sim.stages.common import jain_fairness_index, per_ut_delivered_bits


def test_jain_index_extremes():
    assert jain_fairness_index([5, 5, 5, 5]) == pytest.approx(1.0)
    assert jain_fairness_index([8, 0, 0, 0]) == pytest.approx(0.25)
    assert jain_fairness_index([4, 4, 0, 0]) == pytest.approx(0.5)
    assert math.isnan(jain_fairness_index([0, 0, 0, 0]))


def test_per_ut_delivered_bits_counts_correct_bits_and_zeroes_masked():
    bits = np.zeros((2, 3, 1, 4), dtype=np.float32)
    bits_hat = bits.copy()
    bits_hat[0, 1, 0, 0] = 1.0  # one error at terminal 1
    assert per_ut_delivered_bits(bits, bits_hat) == [8, 7, 8]
    assert per_ut_delivered_bits(bits, bits_hat, ut_mask=[1, 0, 1]) == [8, 0, 8]


def test_allocated_power_and_modelled_latency():
    cfg = load_config("config/config.json").system_runtime_config
    m = Model(cfg, estimator_type="ls")
    p_max = 10 ** ((23.0 - 30.0) / 10.0)
    d = ResourceDirectives(active_ut_mask=[1, 1, 0, 0], per_ut_power=[1.0, 0.5, 1.0, 1.0])
    assert m._allocated_tx_power_w(d) == pytest.approx(1.5 * p_max)
    slot = m._estimate_air_interface_latency()
    assert m._modelled_latency_sec(slot) == pytest.approx(2.0 * slot)


def test_channel_error_stats_and_nmse():
    from factory6g.sim.stages.common import channel_error_stats, nmse_db

    h = np.ones((2, 3), dtype=np.complex64)
    h_hat = h + 0.1
    stats = channel_error_stats(h, h_hat)
    assert stats["channel_energy"] == pytest.approx(6.0)
    assert stats["sq_error"] == pytest.approx(0.06, rel=1e-5)
    assert stats["num_elements"] == 6
    assert nmse_db(stats["sq_error"], stats["channel_energy"]) == pytest.approx(-20.0, abs=1e-3)
    assert math.isnan(nmse_db(1.0, 0.0))
    with pytest.raises(ValueError):
        channel_error_stats(h, h_hat[:1])
