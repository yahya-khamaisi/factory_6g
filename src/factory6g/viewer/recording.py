"""Rerun recording (.rrd) of a traced run, for scrubbing batches in the Rerun viewer.

One recording per run, on a ``batch`` timeline, for one stage and one Eb/N0
point:

* ``hall/...``: room outline, machines and base station (static);
* ``hall/radio_map``: path-gain map cells as coloured points, when exported;
* ``hall/terminals``: terminal positions of every batch, LOS and NLOS;
* ``methods/<m>/links``: links to the scheduled terminals, thicker at higher power;
* ``metrics/<m>/batch_bler`` and ``metrics/<m>/running_ber``: scalar series.

Open it with ``rerun results/<run>/recording.rrd`` (the ``rerun-sdk`` package
installs the viewer) or in the web viewer at https://rerun.io/viewer.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from factory6g.sim.trace import read_trace
from factory6g.viewer.export import find_traces

UT_COLORS = [(0, 114, 178), (213, 94, 0), (0, 158, 115), (204, 121, 167), (230, 159, 0), (86, 180, 233), (107, 107, 107), (136, 34, 85)]
METHOD_COLORS = [(35, 51, 105), (204, 102, 119), (17, 119, 51), (170, 68, 153), (63, 143, 191), (153, 153, 51), (136, 34, 85), (68, 170, 153)]


def _blues(t: np.ndarray) -> np.ndarray:
    lo, hi = np.array([247, 251, 255]), np.array([8, 48, 107])
    return (lo + (hi - lo) * np.clip(t, 0, 1)[:, None]).astype(np.uint8)


def write_recording(run_dir: Path, out_path: Path | None = None, *, ebno_db: float | None = None, repo_root: Path | None = None) -> Path:
    import rerun as rr

    run_dir = Path(run_dir)
    out_path = Path(out_path) if out_path else run_dir / "recording.rrd"
    traces = [read_trace(p) for p in find_traces(run_dir)]
    if not traces:
        raise FileNotFoundError(f"No trace under {run_dir}; run with simulation.trace on.")
    trace = next((t for t in traces if t["header"]["stage"] in ("resource_managers", "coupled")), traces[0])
    meta = trace["header"]["meta"]
    ebnos = meta["ebno_db_range"]
    ebno = float(ebno_db) if ebno_db is not None else float(ebnos[len(ebnos) // 2])
    root = Path(repo_root) if repo_root else Path.cwd()
    bank = root / meta.get("rt_channel_bank", "data/rt_channel_bank.npz")
    hall = json.loads(bank.with_name(bank.name + ".manifest.json").read_text(encoding="utf-8"))["hall"]

    rr.init("factory6g_run_viewer", spawn=False)
    rr.save(str(out_path))
    rr.log("info", rr.TextDocument(
        f"Run {run_dir.name}\nStage {trace['header']['stage']}, Eb/N0 {ebno:g} dB, seed {meta.get('seed')}\n"
        f"Methods: {', '.join(meta['methods'])}\nTimeline 'batch' steps through the Monte Carlo batches; "
        "each batch is one hall draw shared by every method."), static=True)

    L, W, H = hall["room_dimensions"]
    floor = [[-L / 2, -W / 2, 0], [L / 2, -W / 2, 0], [L / 2, W / 2, 0], [-L / 2, W / 2, 0], [-L / 2, -W / 2, 0]]
    rr.log("hall/room", rr.LineStrips3D([floor, [[x, y, H] for x, y, _ in floor]], colors=[(120, 120, 120)], radii=0.03), static=True)
    machines = hall["machines"]
    rr.log("hall/machines", rr.Boxes3D(
        centers=[[m["x"], m["y"], m["sz"] / 2] for m in machines],
        half_sizes=[[m["sx"] / 2, m["sy"] / 2, m["sz"] / 2] for m in machines],
        labels=[f"M{i + 1}" for i in range(len(machines))], colors=[(190, 194, 200)], fill_mode="solid"), static=True)
    bs = hall["bs_position"]
    rr.log("hall/base_station", rr.Points3D([bs], colors=[(31, 35, 40)], radii=0.25, labels=["BS"]), static=True)

    viewer_data = run_dir / "viewer" / "data.json"
    radio_map = json.loads(viewer_data.read_text(encoding="utf-8")).get("radio_map") if viewer_data.exists() else None
    if radio_map:
        grid = np.array([[np.nan if v is None else v for v in row] for row in radio_map["gain_db"]], dtype=float)
        cs = radio_map["cell_size_m"]
        ys, xs = np.nonzero(np.isfinite(grid))
        vals = grid[ys, xs]
        lo, hi = np.nanpercentile(grid, 5), np.nanpercentile(grid, 95)
        positions = np.stack([radio_map["x0"] + (xs + 0.5) * cs, radio_map["y0"] + (ys + 0.5) * cs, np.full(xs.shape, 0.02)], axis=1)
        rr.log("hall/radio_map", rr.Points3D(positions, colors=_blues((vals - lo) / (hi - lo)), radii=cs / 2), static=True)

    contexts = {c["batch"]: c for c in trace["contexts"] if float(c["ebno_db"]) == ebno}
    records = [r for r in trace["methods"] if float(r["ebno_db"]) == ebno]
    by_method: dict[str, dict[int, dict[str, Any]]] = {}
    for r in records:
        by_method.setdefault(r["method"], {})[r["batch"]] = r
    totals = {m: [0, 0] for m in meta["methods"]}
    for batch in sorted(contexts):
        rr.set_time("batch", sequence=batch + 1)
        ctx = contexts[batch]
        pos = ctx.get("positions")
        if pos:
            los = ctx.get("los", [True] * len(pos))
            rr.log("hall/terminals", rr.Points3D(
                pos, colors=[UT_COLORS[i % len(UT_COLORS)] for i in range(len(pos))],
                radii=[0.22 if on else 0.14 for on in los],
                labels=[f"UT{i} {'LOS' if on else 'NLOS'} {g:+.1f} dB" for i, (on, g) in enumerate(zip(los, ctx.get("gain_db", [0] * len(pos))))]))
        for k, method in enumerate(meta["methods"]):
            r = by_method.get(method, {}).get(batch)
            if r is None:
                continue
            if pos:
                strips = [[bs, pos[i]] for i, on in enumerate(r["active"]) if on]
                radii = [0.02 + 0.06 * float(r["power"][i] or 0) for i, on in enumerate(r["active"]) if on]
                rr.log(f"methods/{method}/links", rr.LineStrips3D(strips, radii=radii, colors=[METHOD_COLORS[k % len(METHOD_COLORS)]]))
            be = sum(v for v in r["bit_errors"] if v is not None)
            bits = sum(v for v in r["bits"] if v is not None)
            ble = sum(v for v in r["block_errors"] if v is not None)
            blocks = sum(v for v in r["blocks"] if v is not None)
            totals[method][0] += be
            totals[method][1] += bits
            rr.log(f"metrics/{method}/batch_bler", rr.Scalars(ble / blocks if blocks else 0.0))
            rr.log(f"metrics/{method}/running_ber", rr.Scalars(totals[method][0] / totals[method][1] if totals[method][1] else 0.0))
    rr.disconnect()
    return out_path
