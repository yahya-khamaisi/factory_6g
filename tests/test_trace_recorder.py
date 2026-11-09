from __future__ import annotations

import gzip
import json
from types import SimpleNamespace

import numpy as np
import pytest

from factory6g.sim.config import load_config
from factory6g.sim.stages import estimators as estimator_stage
from factory6g.sim.stages import resource_managers as rm_stage
from factory6g.sim.trace import (
    TRACE_FILENAME,
    NullTraceRecorder,
    TraceRecorder,
    make_trace_recorder,
    per_ut_error_stats,
    read_trace,
)

from .conftest import make_tiny_config, set_all_seeds, write_config


def test_per_ut_error_stats_counts_bits_and_blocks_per_terminal():
    bits = np.zeros((2, 3, 1, 8), dtype=np.int32)  # batch, terminal, stream, codeword bits
    bits_hat = bits.copy()
    bits_hat[0, 0, 0, :3] = 1  # terminal 0: 3 bit errors in one codeword
    bits_hat[1, 2, 0, 0] = 1  # terminal 2: 1 bit error
    stats = per_ut_error_stats(bits, bits_hat)
    assert stats["bit_errors"] == [3, 0, 1]
    assert stats["block_errors"] == [1, 0, 1]
    assert stats["bits"] == [16, 16, 16]
    assert stats["blocks"] == [2, 2, 2]


def test_disabled_recorder_writes_nothing(tmp_path):
    recorder = make_trace_recorder(False, tmp_path, stage="estimators", meta={})
    assert isinstance(recorder, NullTraceRecorder)
    recorder.record_context(0, 0.0, None)
    recorder.flush()
    assert not (tmp_path / TRACE_FILENAME).exists()


def _result(num_ut: int = 2, errors_on: int | None = None) -> dict:
    bits = np.zeros((1, num_ut, 1, 4), dtype=np.int32)
    bits_hat = bits.copy()
    if errors_on is not None:
        bits_hat[0, errors_on, 0, 0] = 1
    return {"bits": bits, "bits_hat": bits_hat, "err_var_mean": 0.25, "decoder_iterations": np.array([3, 5])}


def test_recorder_round_trip_masks_inactive_terminals(tmp_path):
    recorder = TraceRecorder(tmp_path, stage="resource_managers", meta={"methods": ["pf"]})
    large_scale = {"positions": np.array([[1.0, 2.0, 1.0], [3.0, 4.0, 1.0]]), "los": np.array([True, False]), "gain_db": np.array([-1.5, -9.25])}
    recorder.record_context(0, 6.0, large_scale)
    recorder.record_method(0, 6.0, "pf", result=_result(errors_on=1), active_mask=[0, 1], per_ut_power=[0.0, 0.5], runtime_sec=0.01)
    assert read_trace(recorder.path)["methods"] == []  # nothing is written before the batch is flushed
    recorder.flush()

    trace = read_trace(recorder.path)
    assert trace["header"]["stage"] == "resource_managers"
    assert trace["contexts"][0]["los"] == [True, False]
    assert trace["contexts"][0]["gain_db"] == [-1.5, -9.25]
    record = trace["methods"][0]
    assert record["active"] == [False, True]
    assert record["bit_errors"] == [None, 1]
    assert record["block_errors"] == [None, 1]
    assert record["power"] == [0.0, 0.5]
    assert record["decoder_iterations_mean"] == 4.0


def test_resume_drops_records_of_repeated_batches(tmp_path):
    recorder = TraceRecorder(tmp_path, stage="estimators", meta={})
    for batch in range(3):
        recorder.record_method(batch, 0.0, "ls", result=_result(), active_mask=[1, 1], per_ut_power=None, runtime_sec=0.0)
        recorder.flush()
    resumed = TraceRecorder(tmp_path, stage="estimators", meta={}, resume_from_batch=2)
    resumed.record_method(2, 0.0, "ls", result=_result(), active_mask=[1, 1], per_ut_power=None, runtime_sec=0.0)
    resumed.flush()
    batches = [r["batch"] for r in read_trace(resumed.path)["methods"]]
    assert batches == [0, 1, 2]


class _FakeModel:
    """Deterministic stand-in: errors depend only on the method and the batch counter."""

    def __init__(self, *, config, perfect_csi=False, estimator_type="ls", estimator_kwargs=None):
        del perfect_csi, estimator_kwargs
        self.num_ut = int(config.get("num_ut", 2))
        self.estimator_type = estimator_type
        self.calls = 0
        self.last_large_scale = {
            "positions": np.zeros((self.num_ut, 3)),
            "los": np.ones(self.num_ut, dtype=bool),
            "gain_db": np.zeros(self.num_ut),
        }

    def prepare_batch_context(self, batch_size, ebno_db, include_feedback):
        del include_feedback
        return SimpleNamespace(batch_size=batch_size, ebno_db=float(ebno_db), feedback=None)

    def run_batch(self, context, directives=None, include_details=True):
        del context, directives, include_details
        self.calls += 1
        bits = np.zeros((2, self.num_ut, 1, 16), dtype=np.int32)
        bits_hat = bits.copy()
        bits_hat.reshape(-1)[: (self.calls % 5) + 1] = 1
        return {"bits": bits, "bits_hat": bits_hat, "runtime_latency_sec": 1e-4, "energy_joules": 1e-6}


# Wall-clock measurements, and the energy figures derived from them, differ between
# any two runs; every other metric must match exactly.
_WALL_CLOCK_KEYS = {"runtime_sec", "latency_ms", "energy_joules_per_batch", "avg_power_w"}


def _comparable(value):
    if isinstance(value, float) and value != value:
        return "nan"
    if isinstance(value, list):
        return [_comparable(v) for v in value]
    return value


def _strip_runtime(result: dict) -> dict:
    methods = {
        name: {k: _comparable(v) for k, v in metrics.items() if k not in _WALL_CLOCK_KEYS}
        for name, metrics in result["methods"].items()
    }
    return {"ebno_db_range": result["ebno_db_range"], "methods": methods}


def _tiny(tmp_path, *, trace: bool, batches: int = 3):
    data = make_tiny_config(str(tmp_path / "results"))
    data["simulation"]["trace"] = trace
    data["monte_carlo"].update({"min_batches": batches, "max_batches": batches, "ebno_max": 3.0, "ebno_step": 3.0})
    data["estimators"]["enabled"] = ["ls", "lmmse"]
    return load_config(write_config(tmp_path, data))


@pytest.mark.parametrize("stage_module, runner", [
    (estimator_stage, "run_estimator_stage"),
    (rm_stage, "run_resource_manager_stage"),
])
def test_tracing_leaves_stage_results_unchanged(tmp_path, monkeypatch, stage_module, runner):
    monkeypatch.setattr(stage_module, "Model", _FakeModel)
    off_dir, on_dir = tmp_path / "off", tmp_path / "on"
    off_dir.mkdir()
    on_dir.mkdir()
    untraced = getattr(stage_module, runner)(_tiny(off_dir, trace=False), checkpoint_dir=off_dir / "stage")
    traced = getattr(stage_module, runner)(_tiny(on_dir, trace=True), checkpoint_dir=on_dir / "stage")

    assert _strip_runtime(traced) == _strip_runtime(untraced)
    assert not (off_dir / "stage" / TRACE_FILENAME).exists()
    trace = read_trace(on_dir / "stage" / TRACE_FILENAME)
    num_methods = len(traced["methods"])
    num_points = len(traced["ebno_db_range"])
    assert len(trace["contexts"]) == 3 * num_points
    assert len(trace["methods"]) == 3 * num_points * num_methods
    # Summing the per-terminal trace reproduces the aggregated bit errors of every point.
    for method, metrics in traced["methods"].items():
        for i, ebno in enumerate(traced["ebno_db_range"]):
            from_trace = sum(
                sum(v for v in r["bit_errors"] if v is not None)
                for r in trace["methods"]
                if r["method"] == method and r["ebno_db"] == ebno
            )
            assert from_trace == metrics["bit_errors"][i]


@pytest.mark.slow
def test_tracing_leaves_real_resource_manager_stage_unchanged(tmp_path):
    """Phase 0 gate on the real PHY: same seed, trace on or off, same results."""
    results = []
    for trace in (False, True):
        run_dir = tmp_path / ("on" if trace else "off")
        run_dir.mkdir()
        config = _tiny(run_dir, trace=trace, batches=2)
        set_all_seeds(1234)
        results.append(rm_stage.run_resource_manager_stage(config, checkpoint_dir=run_dir / "stage"))
    assert _strip_runtime(results[1]) == _strip_runtime(results[0])
    with gzip.open(tmp_path / "on" / "stage" / TRACE_FILENAME, "rt", encoding="utf-8") as handle:
        header = json.loads(handle.readline())
    assert header["meta"]["methods"] == ["static", "round_robin"]
