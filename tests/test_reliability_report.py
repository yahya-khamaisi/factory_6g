from __future__ import annotations

import numpy as np
import pytest

from factory6g.analysis.reliability import (
    build_report,
    clopper_pearson,
    holm,
    mcnemar_exact,
    paired_t_pvalue,
    probability_of_improvement,
    write_report,
)
from factory6g.sim.trace import TraceRecorder


def test_clopper_pearson_known_values():
    lo, hi = clopper_pearson(0, 10)
    assert lo == 0.0 and hi == pytest.approx(0.3085, abs=1e-4)
    lo, hi = clopper_pearson(5, 10)
    assert (lo, hi) == pytest.approx((0.1871, 0.8129), abs=1e-4)


def test_holm_adjustment():
    adjusted = holm([0.01, 0.04, 0.03, float("nan")])
    assert adjusted[0] == pytest.approx(0.03)
    assert adjusted[2] == pytest.approx(0.06)
    assert adjusted[1] == pytest.approx(0.06)  # monotone: never below the previous step
    assert np.isnan(adjusted[3])


def test_mcnemar_and_paired_t():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(0, 10) == pytest.approx(2 * 0.5 ** 10)
    assert paired_t_pvalue(np.array([0.1, 0.1, 0.1])) == 0.0
    assert paired_t_pvalue(np.array([0.1, -0.1, 0.1, -0.1])) == pytest.approx(1.0)


def test_probability_of_improvement():
    assert probability_of_improvement([1, 2], [3, 4]) == 1.0
    assert probability_of_improvement([1, 3], [3]) == pytest.approx(0.75)


def _run(tmp_path, seed: int, worse_errors: int):
    run_dir = tmp_path / f"run_seed{seed}"
    meta = {"methods": ["base", "better"], "ebno_db_range": [0.0, 6.0], "num_ut": 2, "seed": seed}
    rec = TraceRecorder(run_dir / "resource_managers", stage="resource_managers", meta=meta)
    rng = np.random.default_rng(seed)
    for batch in range(12):
        for eb in (0.0, 6.0):
            for method, n_err in (("base", worse_errors), ("better", 0)):
                bits = np.zeros((4, 2, 1, 32), dtype=np.int32)
                bits_hat = bits.copy()
                hits = rng.integers(0, 4, size=n_err)
                bits_hat[hits, 0, 0, 0] = 1  # codeword errors on terminal 0
                rec.record_method(batch, eb, method, result={"bits": bits, "bits_hat": bits_hat}, active_mask=[1, 1], per_ut_power=None, runtime_sec=0.0)
        rec.flush()
    return run_dir


def test_report_finds_the_better_method_and_summarises_seeds(tmp_path):
    runs = [_run(tmp_path, 42, 3), _run(tmp_path, 43, 2)]
    report = build_report(runs, baseline="base")
    stage = report["stages"]["resource_managers"]
    assert stage["seeds"] == [42, 43]
    rows = [r for r in stage["per_seed"][0]["rows"] if r["method"] == "better"]
    assert all(r["verdict"] == "fewer block errors" for r in rows)
    assert all(r["p_mcnemar"] < 0.05 and r["mcnemar_baseline_only"] > 0 and r["mcnemar_method_only"] == 0 for r in rows)
    seeds = [r for r in stage["across_seeds"] if r["method"] == "better"]
    assert all(r["prob_improvement"] == 1.0 and r["seeds_better"] == 2 for r in seeds)
    written = write_report(report, tmp_path / "out")
    assert {p.name for p in written} >= {"reliability_report.json", "reliability_report.md", "reliability_resource_managers_points.csv", "reliability_resource_managers_seeds.csv"}
