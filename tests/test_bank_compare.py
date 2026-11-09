from __future__ import annotations

import json

import numpy as np
import pytest

from factory6g.analysis.bank_compare import compare_banks


def _bank(path, h, k=7.0, ds=5.0, los=True):
    n = h.shape[0]
    arrays = {}
    for split in ("train", "eval"):
        arrays.update({f"{split}_h": h, f"{split}_positions": np.arange(n * 3, dtype=np.float32).reshape(n, 3),
                       f"{split}_los": np.full(n, los), f"{split}_k_factor_db": np.full(n, k, np.float32),
                       f"{split}_rms_delay_spread_ns": np.full(n, ds, np.float32), f"{split}_num_paths": np.full(n, 200, np.int32)})
    np.savez(path, **arrays)
    (path.parent / (path.name + ".manifest.json")).write_text(json.dumps({"ray_tracing": {}}))


def test_identical_and_scaled_banks(tmp_path):
    rng = np.random.default_rng(0)
    h = (rng.standard_normal((5, 8, 16)) + 1j * rng.standard_normal((5, 8, 16))).astype(np.complex64)
    _bank(tmp_path / "a.npz", h)
    _bank(tmp_path / "b.npz", h * np.sqrt(2.0), k=8.0, ds=6.0, los=False)
    out = compare_banks(tmp_path / "a.npz", tmp_path / "b.npz")["splits"]["eval"]
    assert out["same_positions"]
    assert out["gain_change_db_p5_p50_p95"][1] == pytest.approx(10 * np.log10(2.0), abs=1e-4)
    assert out["cfr_correlation_p5_p50_p95"][1] == pytest.approx(1.0, abs=1e-6)
    assert out["k_change_db_p5_p50_p95"][1] == pytest.approx(1.0)
    assert out["ds_change_ns_p5_p50_p95"][1] == pytest.approx(1.0)
    assert out["los_flag_agreement"] == 0.0
