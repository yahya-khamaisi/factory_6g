"""Live monitor: follow every simulation and training run while it writes (``viewer monitor``).

A traced stage appends one gzip member per batch to ``trace_v1.jsonl.gz``
(see ``factory6g.sim.trace``), so a reader can decompress the members that are
complete and leave a member still being written for the next poll. Each run is
watched incrementally from its last complete member; the monitor never writes
into a run folder.

Per run the snapshot holds:

* status (running, finished, stopped) from the stage files and file ages;
* progress, rate and ETA from the latest progress line of ``simulation.log``;
* the stack that produced it (TensorFlow / Sionna 1.2.1 or PyTorch / Sionna 2.2)
  from ``provenance.json``;
* per stage, method and Eb/N0: summed bit and block errors so far, from which
  the page draws BER curves that move as batches arrive;
* on request, the running BER after every batch (convergence) for one point.

PPO training logs (one JSON object per iteration) are followed the same way.
"""

from __future__ import annotations

import json
import re
import threading
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from factory6g.sim.trace import TRACE_FILENAME

STALL_SECONDS = 15 * 60  # no write for this long while unfinished: "stopped"
_PROGRESS = re.compile(r"(?:(\w+) )?batch (\d+)/(\d+) \(active points (\d+)/(\d+)\) \[([^\]]*) elapsed\]", re.IGNORECASE)
_DONE = re.compile(r"Done \(([\d.]+)s\)")


def _seconds(text: str) -> float:
    total = 0.0
    for value, unit in re.findall(r"([\d.]+)\s*([hms])", text):
        total += float(value) * {"h": 3600, "m": 60, "s": 1}[unit]
    return total


def _active_sum(values: list[Any]) -> int:
    return int(sum(v for v in values if v is not None))


@dataclass
class _Cell:
    bit_errors: int = 0
    bits: int = 0
    block_errors: int = 0
    blocks: int = 0
    batches: int = 0
    running_ber: list[float] = field(default_factory=list)  # cumulative BER after each batch


class LiveTrace:
    """Incremental reader of one stage trace; tolerates a member that is still being written."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.reset()

    def reset(self) -> None:
        self.offset = 0
        self.header: dict[str, Any] | None = None
        self.cells: dict[tuple[str, float], _Cell] = {}
        self.last_batch = -1
        self.records = 0
        self.version = 0
        # Hall state for the 2D/3D views: one large-scale drop per batch (shared by every
        # Eb/N0 point of the batch), the latest record per (method, Eb/N0), and per-terminal
        # outcome counts per (method, Eb/N0).
        self.drops: dict[int, dict[str, Any]] = {}
        self.latest: dict[tuple[str, float], dict[str, Any]] = {}
        self.per_ut: dict[tuple[str, float], dict[str, list[int]]] = {}

    def poll(self) -> bool:
        """Read members completed since the last poll; True when anything changed."""
        try:
            size = self.path.stat().st_size
        except FileNotFoundError:
            return False
        if size < self.offset:  # rewritten (a resumed run truncates its trace)
            self.reset()
        if size == self.offset:
            return False
        with self.path.open("rb") as fh:
            fh.seek(self.offset)
            data = fh.read()
        pos = 0
        changed = False
        while pos < len(data):
            d = zlib.decompressobj(wbits=31)
            try:
                text = d.decompress(data[pos:])
            except zlib.error:
                break
            if not d.eof:  # member still being written
                break
            used = len(data) - pos - len(d.unused_data)
            pos += used
            for line in text.decode("utf-8").splitlines():
                if line.strip():
                    self._add(json.loads(line))
                    changed = True
        self.offset += pos
        if changed:
            self.version += 1
        return changed

    def _add(self, r: dict[str, Any]) -> None:
        kind = r.get("type")
        if kind == "header":
            self.header = r
            return
        if kind == "context":
            if "positions" in r and int(r["batch"]) not in self.drops:
                self.drops[int(r["batch"])] = {k: r.get(k) for k in ("batch", "positions", "los", "gain_db", "industry")}
            return
        if kind != "method":
            return
        self.records += 1
        key = (str(r["method"]), float(r["ebno_db"]))
        # Records always carry "active"; a terminal without traffic has None counts, so fall back to that.
        active = r.get("active") or [v is not None for v in r["bit_errors"]]
        self.latest[key] = {**{k: r.get(k) for k in ("batch", "power", "block_errors", "blocks", "bit_errors", "bits")}, "active": active}
        ut = self.per_ut.setdefault(key, {"scheduled": [], "blocks": [], "block_errors": []})
        for i, on in enumerate(active):
            for name in ("scheduled", "blocks", "block_errors"):
                if len(ut[name]) <= i:
                    ut[name].append(0)
            if on:
                ut["scheduled"][i] += 1
                ut["blocks"][i] += int(r["blocks"][i] or 0)
                ut["block_errors"][i] += int(r["block_errors"][i] or 0)
        self.last_batch = max(self.last_batch, int(r["batch"]))
        c = self.cells.setdefault((str(r["method"]), float(r["ebno_db"])), _Cell())
        c.bit_errors += _active_sum(r["bit_errors"])
        c.bits += _active_sum(r["bits"])
        c.block_errors += _active_sum(r["block_errors"])
        c.blocks += _active_sum(r["blocks"])
        c.batches += 1
        c.running_ber.append(c.bit_errors / c.bits if c.bits else float("nan"))

    def summary(self) -> dict[str, Any]:
        meta = (self.header or {}).get("meta", {})
        return {
            "stage": (self.header or {}).get("stage"),
            "methods": list(meta.get("methods", [])),
            "ebno_db": [float(e) for e in meta.get("ebno_db_range", [])],
            "batches_traced": self.last_batch + 1,
            "cells": [{"method": m, "ebno_db": eb, "bit_errors": c.bit_errors, "bits": c.bits,
                       "block_errors": c.block_errors, "blocks": c.blocks, "batches": c.batches}
                      for (m, eb), c in sorted(self.cells.items(), key=lambda kv: (kv[0][0], kv[0][1]))],
        }

    def convergence(self, ebno_db: float, max_points: int = 200) -> dict[str, list[float | None]]:
        out = {}
        for (m, eb), c in self.cells.items():
            if eb != float(ebno_db):
                continue
            series = c.running_ber
            step = max(1, len(series) // max_points)
            out[m] = [None if v != v else v for v in series[::step]]
        return out


    def scene(self, ebno_db: float, method: str | None, max_drops: int = 400) -> dict[str, Any]:
        meta = (self.header or {}).get("meta", {})
        methods = list(meta.get("methods", []))
        method = method if method in methods else (methods[0] if methods else None)
        latest = self.latest.get((method, float(ebno_db))) if method else None
        batch = latest["batch"] if latest else (max(self.drops) if self.drops else None)
        drops = [self.drops[b] for b in sorted(self.drops)][-max_drops:]
        return {"method": method, "methods": methods, "ebno_db": float(ebno_db), "batch": batch,
                "num_ut": meta.get("num_ut"), "large_scale_model": meta.get("large_scale_model"),
                "drop": self.drops.get(batch) if batch is not None else None, "history": drops,
                "latest": latest, "per_ut": self.per_ut.get((method, float(ebno_db))) if method else None,
                "latest_all": {m: self.latest.get((m, float(ebno_db))) for m in methods}}


def load_hall(data_roots: Iterable[Path], bank: str) -> dict[str, Any] | None:
    """Room, machines and base station from the channel-bank manifest (as the run viewer uses)."""
    name = Path(bank).name + ".manifest.json"
    for root in data_roots:
        for cand in (Path(root) / name, Path(root) / Path(bank).with_name(name), Path(root) / "data" / name):
            if cand.exists():
                hall = json.loads(cand.read_text(encoding="utf-8"))["hall"]
                return {k: hall.get(k) for k in ("room_dimensions", "machines", "bs_position", "ut_height", "num_bs_ant", "carrier_frequency")}
    return None


def _framework(prov: dict[str, Any]) -> dict[str, str]:
    v = prov.get("versions", {})
    if v.get("torch"):
        return {"key": "torch", "label": f"PyTorch {v['torch'].split('+')[0]} · Sionna {v.get('sionna', '2.x')}"}
    if v.get("tensorflow"):
        sionna = v.get("sionna")
        return {"key": "tf", "label": f"TensorFlow {v['tensorflow']} · Sionna {sionna if sionna not in (None, 'unknown') else '1.2.1'}"}
    return {"key": "unknown", "label": "unrecorded stack"}


def _last_progress(log: Path) -> dict[str, Any] | None:
    try:
        with log.open("rb") as fh:
            fh.seek(max(0, log.stat().st_size - 16384))
            tail = fh.read().decode("utf-8", "replace")
    except FileNotFoundError:
        return None
    matches = list(_PROGRESS.finditer(tail))
    if not matches:
        return None
    m = matches[-1]
    done_times = [float(x) for x in _DONE.findall(tail)][-5:]  # recent per-batch wall times
    batch, total = int(m.group(2)), int(m.group(3))
    elapsed = _seconds(m.group(6))
    rate = (sum(done_times) / len(done_times)) if done_times else (elapsed / max(batch - 1, 1) if batch > 1 else None)
    return {"label": (m.group(1) or "").lower() or None, "batch": batch, "total": total,
            "active_points": int(m.group(4)), "points": int(m.group(5)), "elapsed_s": elapsed,
            "seconds_per_batch": rate, "eta_s": rate * (total - batch + 1) if rate else None}


class RunWatcher:
    """Finds runs under one or more results roots and keeps their live state."""

    def __init__(self, roots: Iterable[Path], *, since_hours: float = 24.0, training_iterations: int = 150,
                 data_roots: Iterable[Path] = ()) -> None:
        self.roots = [Path(r) for r in roots]
        self.data_roots = [Path(r) for r in data_roots]
        self._halls: dict[str, dict[str, Any] | None] = {}
        self.since_s = since_hours * 3600.0
        self.training_iterations = int(training_iterations)
        self.traces: dict[Path, LiveTrace] = {}
        self.training: dict[Path, dict[str, Any]] = {}
        self.lock = threading.Lock()
        self.version = 0
        self._snapshot: dict[str, Any] = {"runs": [], "training": [], "version": 0}

    # -------------------------------------------------------------- discovery
    def _run_dirs(self) -> list[tuple[str, Path]]:
        out = []
        for root in self.roots:
            if not root.exists():
                continue
            for trace in root.glob(f"**/{TRACE_FILENAME}"):
                run = trace.parent.parent
                if "viewer" in run.parts or "_superseded" in str(run):
                    continue
                out.append((root.name, run))
        return sorted(set(out))

    def _training_logs(self) -> list[tuple[str, Path]]:
        out = []
        for root in self.roots:
            for log in root.glob("**/training_logs/*.log"):
                out.append((root.name, log))
        return sorted(set(out))

    # -------------------------------------------------------------- polling
    def poll(self) -> bool:
        now = time.time()
        runs, changed = [], False
        for root_name, run in self._run_dirs():
            stage_dirs = sorted({p.parent for p in run.glob(f"*/{TRACE_FILENAME}")})
            mtimes = [p.stat().st_mtime for p in run.rglob("*") if p.is_file()]
            last_write = max(mtimes) if mtimes else 0.0
            finished_all = all((sd / "stage_results_v2.json").exists() for sd in stage_dirs)
            if finished_all and now - last_write > self.since_s:
                continue
            stages = []
            for sd in stage_dirs:
                tr = self.traces.setdefault(sd / TRACE_FILENAME, LiveTrace(sd / TRACE_FILENAME))
                changed |= tr.poll()
                stages.append({**tr.summary(), "dir": sd.name, "finished": (sd / "stage_results_v2.json").exists(),
                               "version": tr.version})
            if finished_all:
                status = "finished"
            elif now - last_write > STALL_SECONDS:
                status = "stopped"
            else:
                status = "running"
            prov = {}
            if (run / "provenance.json").exists():
                try:
                    prov = json.loads((run / "provenance.json").read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    prov = {}
            progress = _last_progress(run / "simulation.log") if status == "running" else None
            runs.append({
                "id": f"{root_name}/{run.relative_to(next(r for r in self.roots if r.name == root_name))}",
                "root": root_name, "name": run.name, "status": status, "last_write": last_write,
                "age_s": now - last_write, "seed": prov.get("seed"), "config": prov.get("config_path"),
                "argv": prov.get("argv", [])[1:], "framework": _framework(prov), "progress": progress, "stages": stages,
            })
        training = []
        for root_name, log in self._training_logs():
            st = log.stat()
            if now - st.st_mtime > self.since_s:
                continue
            cached = self.training.get(log)
            if cached is None or cached["size"] != st.st_size:
                rows = []
                for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
                    line = line.strip()
                    if line.startswith("{") and '"iteration"' in line:
                        try:
                            rows.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
                cached = {"size": st.st_size, "rows": rows}
                self.training[log] = cached
                changed = True
            rows = cached["rows"]
            last = rows[-1] if rows else {}
            rate = None
            if len(rows) >= 2:
                rate = (rows[-1]["elapsed_s"] - rows[max(0, len(rows) - 6)]["elapsed_s"]) / (len(rows) - max(0, len(rows) - 6) - 1 or 1)
            training.append({
                "id": f"{root_name}/{log.name}", "name": log.stem, "root": root_name,
                "status": "running" if now - st.st_mtime < STALL_SECONDS else "stopped",
                "age_s": now - st.st_mtime, "iteration": last.get("iteration", 0), "total": self.training_iterations,
                "seconds_per_iteration": rate,
                "eta_s": rate * (self.training_iterations - last.get("iteration", 0)) if rate else None,
                "reward": [r.get("mean_step_reward") for r in rows], "ber": [r.get("rollout_ber") for r in rows],
                "bler_by_ebno": last.get("rollout_bler_by_ebno"), "lagrange": last.get("lagrange"),
            })
        # Stable order (status, then newest start first) so cards do not jump as runs write.
        runs.sort(key=lambda r: r["name"], reverse=True)
        runs.sort(key=lambda r: {"running": 0, "stopped": 1, "finished": 2}[r["status"]])
        with self.lock:
            if changed or [r["status"] for r in runs] != [r["status"] for r in self._snapshot["runs"]] or \
                    len(runs) != len(self._snapshot["runs"]):
                self.version += 1
            self._snapshot = {"runs": runs, "training": training, "version": self.version, "polled_at": now}
        return changed

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return dict(self._snapshot)

    def _trace(self, run_id: str, stage_dir: str) -> LiveTrace:
        root_name, _, rel = run_id.partition("/")
        root = next((r for r in self.roots if r.name == root_name), None)
        if root is None:
            raise KeyError(run_id)
        tr = self.traces.get(root / rel / stage_dir / TRACE_FILENAME)
        if tr is None:
            raise KeyError(run_id)
        return tr

    def scene(self, run_id: str, stage_dir: str, ebno_db: float, method: str | None) -> dict[str, Any]:
        tr = self._trace(run_id, stage_dir)
        bank = (tr.header or {}).get("meta", {}).get("rt_channel_bank", "data/rt_channel_bank.npz")
        if bank not in self._halls:
            self._halls[bank] = load_hall(self.data_roots, bank)
        return {"run": run_id, "stage": stage_dir, "hall": self._halls[bank], **tr.scene(ebno_db, method)}

    def convergence(self, run_id: str, stage_dir: str, ebno_db: float) -> dict[str, Any]:
        root_name, _, rel = run_id.partition("/")
        root = next((r for r in self.roots if r.name == root_name), None)
        if root is None:
            raise KeyError(run_id)
        path = (root / rel / stage_dir / TRACE_FILENAME).resolve()
        if root.resolve() not in path.parents:
            raise KeyError(run_id)
        tr = self.traces.get(root / rel / stage_dir / TRACE_FILENAME)
        if tr is None:
            raise KeyError(run_id)
        return {"run": run_id, "stage": stage_dir, "ebno_db": float(ebno_db), "series": tr.convergence(ebno_db)}
