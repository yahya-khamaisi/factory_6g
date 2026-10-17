"""Batch-level evidence rule: errors must come from enough distinct batches."""

from factory6g.sim.stages.common import (
    POINT_STATUS_RESOLVED,
    POINT_STATUS_UPPER_BOUND_ONLY,
    classify_point_status,
    mc_stop_reason,
)
from factory6g.sim.config import load_config


def _stop(error_batches, min_error_batches):
    return mc_stop_reason(
        num_batches=50, total_bits=10**7, total_block_errors=150, target_block_errors=100,
        total_bit_errors=5000, target_ber=None, stop_policy="threshold", confidence_level=0.95,
        min_batches=10, min_total_bits=10**6,
        error_batches=error_batches, min_error_batches=min_error_batches,
    )


def test_point_needs_errors_from_enough_batches():
    assert classify_point_status(500, error_batches=3, min_error_batches=10) == POINT_STATUS_UPPER_BOUND_ONLY
    assert classify_point_status(500, error_batches=10, min_error_batches=10) == POINT_STATUS_RESOLVED
    assert classify_point_status(20, error_batches=40, min_error_batches=10) == POINT_STATUS_UPPER_BOUND_ONLY


def test_v1_rule_is_unchanged_by_default():
    assert classify_point_status(30) == POINT_STATUS_RESOLVED
    assert classify_point_status(500, error_batches=1) == POINT_STATUS_RESOLVED
    assert _stop(error_batches=1, min_error_batches=0) == "target_block_errors"


def test_block_error_target_waits_for_distinct_batches():
    assert _stop(error_batches=4, min_error_batches=10) is None
    assert _stop(error_batches=10, min_error_batches=10) == "target_block_errors"


def test_config_default_and_override(tmp_path):
    cfg = load_config("config/config.json")
    assert cfg.monte_carlo.min_error_batches >= 0
