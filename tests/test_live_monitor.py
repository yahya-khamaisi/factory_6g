from __future__ import annotations

import gzip
import json
import time

import numpy as np

from factory6g.sim.trace import TRACE_FILENAME, TraceRecorder
from factory6g.viewer.live import LiveTrace, RunWatcher, _last_progress


def _batch(rec: TraceRecorder, batch: int, errors: int) -> None:
    bits = np.zeros((4, 2, 1, 32), dtype=np.int32)
    bits_hat = bits.copy()
    bits_hat[:errors, 0, 0, 0] = 1
    rec.record_method(batch, 6.0, "ls", result={"bits": bits, "bits_hat": bits_hat}, active_mask=[1, 1], per_ut_power=None, runtime_sec=0.0)
    rec.flush()


def test_reader_follows_appended_batches_and_skips_a_member_in_progress(tmp_path):
    rec = TraceRecorder(tmp_path / "estimators", stage="estimators", meta={"methods": ["ls"], "ebno_db_range": [6.0]})
    live = LiveTrace(tmp_path / "estimators" / TRACE_FILENAME)
    assert live.poll() and live.header["stage"] == "estimators"
    _batch(rec, 0, 2)
    _batch(rec, 1, 1)
    assert live.poll()
    cell = live.summary()["cells"][0]
    assert (cell["bit_errors"], cell["bits"], cell["batches"]) == (3, 512, 2)
    assert not live.poll()  # nothing new

    # Half of a new gzip member on disk: not read yet, and read once it completes.
    member = gzip.compress((json.dumps({"type": "method", "batch": 2, "ebno_db": 6.0, "method": "ls", "bit_errors": [4, None],
                                        "bits": [256, None], "block_errors": [1, None], "blocks": [4, None]}) + "\n").encode())
    path = tmp_path / "estimators" / TRACE_FILENAME
    with path.open("ab") as fh:
        fh.write(member[:10])
    assert not live.poll()
    with path.open("ab") as fh:
        fh.write(member[10:])
    assert live.poll()
    assert live.summary()["cells"][0]["bit_errors"] == 7
    assert live.convergence(6.0)["ls"] == [2 / 256, 3 / 512, 7 / 768]


def test_reader_restarts_when_a_resumed_run_rewrites_the_trace(tmp_path):
    rec = TraceRecorder(tmp_path / "s", stage="estimators", meta={"methods": ["ls"], "ebno_db_range": [6.0]})
    for b in range(3):
        _batch(rec, b, 1)
    live = LiveTrace(tmp_path / "s" / TRACE_FILENAME)
    live.poll()
    TraceRecorder(tmp_path / "s", stage="estimators", meta={}, resume_from_batch=1)  # keeps batch 0 only
    live.poll()
    assert live.summary()["cells"][0]["batches"] == 1


def test_progress_line_and_watcher_status(tmp_path):
    run = tmp_path / "results" / "20261109_120000_ls_rayleigh_qpsk_s"
    rec = TraceRecorder(run / "estimators", stage="estimators", meta={"methods": ["ls"], "ebno_db_range": [6.0]})
    _batch(rec, 0, 1)
    (run / "provenance.json").write_text(json.dumps({"seed": 42, "argv": ["run.py", "--channel", "rayleigh"],
                                                     "versions": {"torch": "2.15.0+cpu", "sionna": "2.2.0"}}))
    (run / "simulation.log").write_text("Estimator batch 1/200 (active points 11/11) [0s elapsed] ... Done (40.0s)\n"
                                        "Estimator batch 2/200 (active points 10/11) [40s elapsed] ...")
    p = _last_progress(run / "simulation.log")
    assert (p["batch"], p["total"], p["active_points"], p["seconds_per_batch"]) == (2, 200, 10, 40.0)
    assert p["eta_s"] == 40.0 * 199

    watcher = RunWatcher([tmp_path / "results"])
    watcher.poll()
    snap = watcher.snapshot()
    (r,) = snap["runs"]
    assert r["status"] == "running" and r["framework"]["key"] == "torch" and r["seed"] == 42
    assert r["stages"][0]["cells"][0]["bit_errors"] == 1
    assert watcher.convergence(r["id"], "estimators", 6.0)["series"]["ls"]

    (run / "estimators" / "stage_results_v2.json").write_text("{}")
    v = snap["version"]
    watcher.poll()
    assert watcher.snapshot()["runs"][0]["status"] == "finished" and watcher.snapshot()["version"] > v


def test_training_log_is_followed(tmp_path):
    logs = tmp_path / "results" / "training_logs"
    logs.mkdir(parents=True)
    lines = [json.dumps({"iteration": i, "mean_step_reward": 0.5, "rollout_ber": 0.01, "elapsed_s": 10.0 * i,
                         "lagrange": {"0.0": 1.0}}) for i in range(1, 4)]
    (logs / "ppo_constrained.log").write_text("starting\n" + "\n".join(lines) + "\n")
    watcher = RunWatcher([tmp_path / "results"], training_iterations=10)
    watcher.poll()
    (t,) = watcher.snapshot()["training"]
    assert t["iteration"] == 3 and t["seconds_per_iteration"] == 10.0 and t["eta_s"] == 70.0
    assert t["lagrange"] == {"0.0": 1.0} and t["status"] == "running"
    assert time.time() - t["age_s"] > 0


def test_scene_keeps_the_latest_drop_and_per_terminal_outcomes(tmp_path):
    run = tmp_path / "results" / "20261109_130000_static_rayleigh_qpsk_s"
    meta = {"methods": ["static"], "ebno_db_range": [6.0], "num_ut": 2, "rt_channel_bank": "data/rt_channel_bank.npz"}
    rec = TraceRecorder(run / "resource_managers", stage="resource_managers", meta=meta)
    for b in range(3):
        rec.record_context(b, 6.0, {"positions": [[b, 0.0, 1.0], [0.0, b, 1.0]], "los": [True, False], "gain_db": [1.0, -2.0]})
        bits = np.zeros((4, 2, 1, 32), dtype=np.int32)
        bits_hat = bits.copy()
        if b == 2:
            bits_hat[0, 1, 0, 0] = 1  # UT1 fails in the last batch
        rec.record_method(b, 6.0, "static", result={"bits": bits, "bits_hat": bits_hat}, active_mask=[1, 1], per_ut_power=[1.0, 0.5], runtime_sec=0.0)
        rec.flush()
    data = tmp_path / "data"
    data.mkdir()
    (data / "rt_channel_bank.npz.manifest.json").write_text(json.dumps({"hall": {"room_dimensions": [15, 15, 5], "machines": [], "bs_position": [0, 0, 4], "ut_height": 1.0}}))
    watcher = RunWatcher([tmp_path / "results"], data_roots=[data])
    watcher.poll()
    (r,) = watcher.snapshot()["runs"]
    sc = watcher.scene(r["id"], "resource_managers", 6.0, None)
    assert sc["method"] == "static" and sc["batch"] == 2
    assert sc["drop"]["positions"][0] == [2.0, 0.0, 1.0] and sc["drop"]["los"] == [True, False]
    assert len(sc["history"]) == 3
    assert sc["latest"]["block_errors"] == [0, 1] and sc["latest"]["power"] == [1.0, 0.5]
    assert sc["per_ut"] == {"scheduled": [3, 3], "blocks": [12, 12], "block_errors": [0, 1]}
    assert sc["hall"]["room_dimensions"] == [15, 15, 5]
