"""Collect a run's traces, hall geometry and results into one ``data.json``."""

from __future__ import annotations

import datetime as _dt
import json
import shutil
from pathlib import Path
from typing import Any

from factory6g.sim.trace import TRACE_FILENAME, read_trace

EXPORT_FORMAT_VERSION = 1
STATIC_DIR = Path(__file__).parent / "static"


def find_traces(run_dir: Path) -> list[Path]:
    return sorted(Path(run_dir).rglob(TRACE_FILENAME))


def load_hall(bank_path: str, *, search_roots: list[Path]) -> dict[str, Any] | None:
    """Hall geometry from the channel bank manifest (room, machines, base station)."""
    for root in search_roots:
        manifest = (root / bank_path).with_name(Path(bank_path).name + ".manifest.json")
        if manifest.exists():
            hall = json.loads(manifest.read_text(encoding="utf-8"))["hall"]
            return {
                "room_dimensions": hall["room_dimensions"],
                "machines": hall["machines"],
                "bs_position": hall["bs_position"],
                "ut_height": hall.get("ut_height", 1.0),
                "source": str(manifest.relative_to(root)) if manifest.is_relative_to(root) else str(manifest),
            }
    return None


def _stage_entry(trace_path: Path, run_dir: Path) -> dict[str, Any]:
    trace = read_trace(trace_path)
    stage_dir = trace_path.parent
    rel = stage_dir.relative_to(run_dir).as_posix()
    results_path = stage_dir / "stage_results_v2.json"
    point_results = None
    if results_path.exists():
        payload = json.loads(results_path.read_text(encoding="utf-8"))
        keep = ("ber", "ber_upper_confidence", "bit_errors", "total_bits", "block_errors", "total_blocks", "num_batches", "point_status", "jain_fairness")
        point_results = {
            method: {k: metrics.get(k) for k in keep}
            for method, metrics in payload.get("methods", {}).items()
        }
    contexts = {
        f"{c['batch']}|{c['ebno_db']:g}": {k: c[k] for k in ("positions", "los", "gain_db") if k in c}
        for c in trace["contexts"]
    }
    methods = [{k: v for k, v in r.items() if k != "type"} for r in trace["methods"]]
    meta = trace["header"]["meta"]
    return {
        "id": rel,
        "stage": trace["header"]["stage"],
        "meta": meta,
        "num_batches": 1 + max((r["batch"] for r in methods), default=-1),
        "contexts": contexts,
        "records": methods,
        "results": point_results,
    }


def export_run(
    run_dir: Path,
    out_dir: Path,
    *,
    repo_root: Path | None = None,
    radio_map: dict[str, Any] | None = None,
    scene: dict[str, Any] | None = None,
) -> Path:
    run_dir = Path(run_dir)
    out_dir = Path(out_dir)
    traces = find_traces(run_dir)
    if not traces:
        raise FileNotFoundError(
            f"No {TRACE_FILENAME} under {run_dir}. Re-run with \"simulation\": {{\"trace\": true}}."
        )
    stages = [_stage_entry(p, run_dir) for p in traces]
    roots = [r for r in (repo_root, Path.cwd()) if r is not None]
    hall = load_hall(stages[0]["meta"].get("rt_channel_bank", "data/rt_channel_bank.npz"), search_roots=roots)
    data = {
        "version": EXPORT_FORMAT_VERSION,
        "run": run_dir.name,
        "exported_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "hall": hall,
        "radio_map": radio_map,
        "scene": scene,
        "stages": stages,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "data.json").write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    for asset in STATIC_DIR.iterdir():
        if asset.is_file():
            shutil.copy2(asset, out_dir / asset.name)
    return out_dir / "data.json"
