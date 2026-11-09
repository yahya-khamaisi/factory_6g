from __future__ import annotations

import numpy as np

from factory6g.analysis.equivalence import compare, write
from factory6g.sim.trace import TraceRecorder


def _run(root, name: str, seed: int, error_rate: float):
    run_dir = root / name
    meta = {"methods": ["ls"], "ebno_db_range": [0.0, 30.0], "num_ut": 2, "seed": seed}
    rec = TraceRecorder(run_dir / "estimators", stage="estimators", meta=meta)
    rng = np.random.default_rng(seed)
    for batch in range(20):
        for eb, p in ((0.0, error_rate), (30.0, 0.0)):
            bits = np.zeros((4, 2, 1, 64), dtype=np.int32)
            bits_hat = (rng.random(bits.shape) < p).astype(np.int32)
            rec.record_method(batch, eb, "ls", result={"bits": bits, "bits_hat": bits_hat}, active_mask=[1, 1], per_ut_power=None, runtime_sec=0.0)
        rec.flush()
    return run_dir


def test_gate_accepts_the_same_ber_and_rejects_a_shifted_one(tmp_path):
    ref = [_run(tmp_path, f"tf{s}", s, 0.10) for s in range(42, 47)]
    same = compare(ref, [_run(tmp_path, "torch_same", 7, 0.10)], stage="estimators")
    assert same["passed"]
    rows = {r["ebno_db"]: r for r in same["rows"]}
    assert rows[0.0]["verdict"] == "equivalent"
    assert rows[30.0]["verdict"] == "not resolved"  # no errors on either side

    shifted = compare(ref, [_run(tmp_path, "torch_off", 7, 0.20)], stage="estimators")
    assert not shifted["passed"]
    assert {r["ebno_db"]: r for r in shifted["rows"]}[0.0]["verdict"] == "differs"
    paths = write(shifted, tmp_path / "out")
    assert "FAILED" in paths[1].read_text()
