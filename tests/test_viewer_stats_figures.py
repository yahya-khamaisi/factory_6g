from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from factory6g.sim.trace import TraceRecorder
from factory6g.viewer.stats import (
    BatchSeries,
    batch_series,
    delivered_per_terminal,
    jain_index,
    mean_ci,
    paired_difference,
    point_summary,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _record(method: str, batch: int, bit_errors: list, block_errors: list, active=(True, True)) -> dict:
    return {
        "method": method, "batch": batch, "ebno_db": 6.0, "active": list(active), "power": [1.0, 1.0],
        "bit_errors": [e if on else None for e, on in zip(bit_errors, active)],
        "bits": [100 if on else None for on in active],
        "block_errors": [e if on else None for e, on in zip(block_errors, active)],
        "blocks": [10 if on else None for on in active],
    }


def test_mean_ci_matches_normal_interval():
    mean, lo, hi = mean_ci(np.array([1.0, 2.0, 3.0, 4.0]))
    assert mean == pytest.approx(2.5)
    half = 1.959963984540054 * np.std([1, 2, 3, 4], ddof=1) / 2.0
    assert (lo, hi) == pytest.approx((2.5 - half, 2.5 + half))
    assert np.isnan(mean_ci(np.array([1.0]))[1])


def test_batch_series_ignores_unscheduled_terminals():
    records = [_record("pf", 0, [5, 7], [1, 1], active=(True, False)), _record("pf", 1, [0, 3], [0, 1])]
    series = batch_series(records, "pf", 6.0)
    assert series.bit_errors.tolist() == [5, 3]
    assert series.bits.tolist() == [100, 200]
    summary = point_summary(series)
    assert summary.bit_errors == 8 and summary.bits == 300
    assert not summary.resolved  # fewer than 30 bit errors


def test_paired_difference_uses_shared_batches_only():
    a = BatchSeries(np.array([0, 1, 2]), np.zeros(3), np.ones(3), np.array([1.0, 2.0, 3.0]), np.full(3, 10.0))
    b = BatchSeries(np.array([1, 2, 3]), np.zeros(3), np.ones(3), np.array([1.0, 1.0, 1.0]), np.full(3, 10.0))
    mean, lo, hi, n = paired_difference(a, b)
    assert n == 2
    assert mean == pytest.approx(0.15)
    assert lo < mean < hi


def test_delivered_bits_and_jain():
    records = [_record("rr", 0, [10, 0], [1, 0], active=(True, False)), _record("rr", 1, [0, 0], [0, 0], active=(False, True))]
    delivered = delivered_per_terminal(records, "rr", 6.0)
    assert delivered.tolist() == [[90, 0], [0, 100]]
    assert jain_index(np.array([1.0, 1.0])) == pytest.approx(1.0)
    assert jain_index(np.array([1.0, 0.0])) == pytest.approx(0.5)


def _traced_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    meta = {"methods": ["static", "pf"], "ebno_db_range": [0.0, 6.0], "num_ut": 2, "seed": 42,
            "rt_channel_bank": "data/rt_channel_bank.npz", "large_scale_model": "hall", "channel_model_type": "rayleigh"}
    recorder = TraceRecorder(run_dir / "resource_managers", stage="resource_managers", meta=meta)
    rng = np.random.default_rng(0)
    for batch in range(4):
        for eb in (0.0, 6.0):
            recorder.record_context(batch, eb, {"positions": rng.uniform(-6, 6, (2, 3)), "los": np.array([True, False]), "gain_db": np.array([0.5, -6.0])})
            for method in meta["methods"]:
                bits = np.zeros((2, 2, 1, 16), dtype=np.int32)
                bits_hat = bits.copy()
                bits_hat[0, 0, 0, : int(rng.integers(0, 12))] = 1
                recorder.record_method(batch, eb, method, result={"bits": bits, "bits_hat": bits_hat},
                                       active_mask=[1, 1], per_ut_power=None, runtime_sec=0.0)
        recorder.flush()
    return run_dir


@pytest.mark.skipif(not (REPO_ROOT / "data" / "rt_channel_bank.npz").exists(), reason="needs the channel bank")
def test_figure_pack_writes_every_figure(tmp_path):
    pytest.importorskip("matplotlib")
    from factory6g.viewer.figures import make_figures

    written = make_figures(_traced_run(tmp_path), tmp_path / "figures", repo_root=REPO_ROOT)
    names = {p.name for p in written}
    for stem in ("fig_floor_radio_map", "fig_pathloss_vs_distance", "fig_ber_vs_ebno_resource_managers", "fig_throughput_cdf", "fig_fairness_vs_reliability"):
        assert f"{stem}.png" in names and f"{stem}.pdf" in names
    assert "figure_pack_summary.csv" in names


def test_recording_is_written(tmp_path):
    pytest.importorskip("rerun")
    from factory6g.viewer.recording import write_recording

    path = write_recording(_traced_run(tmp_path), tmp_path / "run.rrd", repo_root=REPO_ROOT)
    assert path.exists() and path.stat().st_size > 1000
