from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from factory6g.sim.trace import TraceRecorder
from factory6g.viewer.export import export_run

REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_stage(stage_dir: Path) -> None:
    meta = {"methods": ["pf"], "ebno_db_range": [0.0, 3.0], "num_ut": 2, "seed": 42,
            "rt_channel_bank": "data/rt_channel_bank.npz", "large_scale_model": "hall", "channel_model_type": "rayleigh"}
    recorder = TraceRecorder(stage_dir, stage="resource_managers", meta=meta)
    bits = np.zeros((1, 2, 1, 8), dtype=np.int32)
    for batch in range(2):
        for eb in (0.0, 3.0):
            recorder.record_context(batch, eb, {"positions": np.zeros((2, 3)), "los": np.array([True, False]), "gain_db": np.array([1.0, -4.0])})
            bits_hat = bits.copy()
            bits_hat[0, 1, 0, :2] = 1
            recorder.record_method(batch, eb, "pf", result={"bits": bits, "bits_hat": bits_hat}, active_mask=[1, 1], per_ut_power=None, runtime_sec=0.0)
        recorder.flush()
    (stage_dir / "stage_results_v2.json").write_text(json.dumps({"methods": {"pf": {"ber": [0.25, 0.25], "bit_errors": [4.0, 4.0]}}}))


def test_export_writes_data_and_viewer_assets(tmp_path):
    run_dir = tmp_path / "20261109_run"
    _write_stage(run_dir / "resource_managers")
    data_path = export_run(run_dir, run_dir / "viewer", repo_root=REPO_ROOT)

    data = json.loads(data_path.read_text())
    assert data["run"] == "20261109_run"
    stage = data["stages"][0]
    assert stage["id"] == "resource_managers"
    assert stage["num_batches"] == 2
    assert set(stage["contexts"]) == {"0|0", "0|3", "1|0", "1|3"}
    assert stage["records"][0]["bit_errors"] == [0, 2]
    assert stage["results"]["pf"]["bit_errors"] == [4.0, 4.0]
    for asset in ("index.html", "app.js", "app.css"):
        assert (run_dir / "viewer" / asset).exists()
    if (REPO_ROOT / "data" / "rt_channel_bank.npz.manifest.json").exists():
        assert len(data["hall"]["machines"]) == 5


def test_export_without_trace_explains_how_to_get_one(tmp_path):
    with pytest.raises(FileNotFoundError, match="trace"):
        export_run(tmp_path, tmp_path / "viewer")
