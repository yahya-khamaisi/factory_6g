from __future__ import annotations

import numpy as np
import pytest

from factory6g.analysis.service import energy_per_bit, reliability_within_budget, slot_seconds, survival, tail_bler


def test_reliability_within_budget_counts_late_delivery():
    delivered = np.array([[False], [True], [False], [False], [True]])
    assert reliability_within_budget(delivered, 1) == pytest.approx(2 / 5)
    # Arrivals 0..3 have a full 2-interval window: 0 -> yes (1), 1 -> yes, 2 -> no, 3 -> yes (4).
    assert reliability_within_budget(delivered, 2) == pytest.approx(3 / 4)


def test_survival_counts_runs_longer_than_the_survival_time():
    delivered = np.array([[True], [False], [False], [False], [True], [False]])
    out = survival(delivered, survival_intervals=1)
    assert out["violations"] == 1  # one run of three failures exceeds a one-interval survival time
    assert out["availability"] == pytest.approx(1 - 2 / 6)  # intervals 3 and 4 of the run are down


def test_energy_and_tail():
    rec = {"method": "m", "ebno_db": 0.0, "active": [True, False], "power": [0.5, 0.0],
           "bits": [1000, None], "bit_errors": [0, None], "blocks": [10, None], "block_errors": [1, None]}
    e = energy_per_bit([rec], "m", 0.0, batch_size=20, slot_seconds=5e-4, max_power_w=0.2)
    assert e == pytest.approx(0.5 * 0.2 * 5e-4 * 20 / 1000)
    t = tail_bler([rec, dict(rec, block_errors=[0, None])], "m", 0.0, 0.05)
    assert t["bler_violation_rate"] == pytest.approx(0.5)
    assert slot_seconds({"fft_size": 128, "subcarrier_spacing": 30e3}) == pytest.approx(14 * (1 + 20 / 128) / 30e3)


def _industry_trace():
    meta = {"methods": ["m"], "ebno_db_range": [0.0], "batch_size": 1,
            "industry": {"device_classes": ["a", "b"],
                         "class_specs": [{"label": "A", "latency_intervals": 1, "reliability": 0.5, "survival_intervals": 1},
                                         {"label": "B", "latency_intervals": 1, "reliability": 0.999999, "survival_intervals": 1}],
                         "events": [{"type": "bs_port_fault", "start": 10, "end": 12}]}}
    records = []
    for b in range(30):
        bad = 10 <= b <= 13
        records.append({"method": "m", "ebno_db": 0.0, "batch": b, "active": [True, True], "power": [1.0, 1.0],
                        "bits": [10, 10], "bit_errors": [5 if bad else 0, 0], "blocks": [1, 1], "block_errors": [1 if bad else 0, 0]})
    return {"header": {"stage": "resource_managers", "meta": meta}, "methods": records}


def test_class_verdicts_and_event_recovery():
    from factory6g.analysis.service import class_service_rows, event_recovery_rows

    rows = {r["device_class"]: r for r in class_service_rows(_industry_trace())}
    assert rows["a"]["verdict"] == "met"  # 26/30 delivered, target 0.5
    assert rows["b"]["verdict"] == "inconclusive"  # 30 clean packets cannot confirm six nines
    assert rows["b"]["packets_needed"] == 3000000
    ev = event_recovery_rows(_industry_trace())[0]
    assert ev["delivery_before"] == 1.0 and ev["delivery_during"] == 0.5
    assert ev["recovery_intervals_after_end"] == 1  # batch 13 still fails, window from 14 is clean
