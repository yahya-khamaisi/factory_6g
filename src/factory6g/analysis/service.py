"""Service-level metrics from a stage trace (Phase 2): latency budget, survival time, energy, tails.

Each Monte Carlo batch is treated as one scheduling interval: every terminal
has one packet per interval, carried by the codewords it sends in that batch.
A packet is delivered when the terminal is scheduled and all its codewords
decode. With that mapping:

* **reliability within a latency budget of K intervals** (TR 38.824 style):
  the share of packets delivered within K intervals of arrival, counting the
  arrival interval;
* **survival time** (TS 22.104): the number of consecutive failed intervals a
  terminal tolerates before the service is unavailable; a *violation* is a run
  of failures longer than that, and *availability* is the share of intervals in
  which the terminal's service is up;
* **uplink energy per delivered bit**: allocated power times the maximum UE
  power times the slot duration, summed over the scheduled terminals, divided
  by the delivered bits;
* **tail BLER**: the 95th and 99th percentile of per-batch BLER and the share
  of batches above a BLER target.

The interval mapping is a modelling assumption. A batch holds independent
slots that the simulator processes in parallel, so latency here is counted in
scheduling decisions rather than milliseconds.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np


def packet_outcomes(records: Iterable[dict[str, Any]], method: str, ebno_db: float) -> tuple[np.ndarray, np.ndarray]:
    """Per batch and terminal: delivered (bool [B, U]) and the batch index order."""
    rows = sorted((r for r in records if r["method"] == method and float(r["ebno_db"]) == float(ebno_db)), key=lambda r: r["batch"])
    delivered = np.array([[bool(on) and (be or 0) == 0 for on, be in zip(r["active"], r["block_errors"])] for r in rows], dtype=bool)
    return delivered, np.array([r["batch"] for r in rows])


def reliability_within_budget(delivered: np.ndarray, budget: int) -> float:
    """Share of packets delivered within ``budget`` intervals; packets near the end without a full window are left out."""
    batches, _ = delivered.shape
    if batches < budget or budget < 1:
        return float("nan")
    windows = np.stack([delivered[k:batches - budget + 1 + k] for k in range(budget)], axis=0)
    return float(np.mean(np.any(windows, axis=0)))


def survival(delivered: np.ndarray, survival_intervals: int) -> dict[str, float]:
    """Violations (failure runs longer than the survival time) and availability, over all terminals."""
    violations = down = 0
    total = delivered.size
    for u in range(delivered.shape[1]):
        run = 0
        for ok in delivered[:, u]:
            run = 0 if ok else run + 1
            if run == survival_intervals + 1:
                violations += 1
            if run > survival_intervals:
                down += 1
    return {"violations": float(violations), "availability": 1.0 - down / total if total else float("nan")}


def energy_per_bit(records: Iterable[dict[str, Any]], method: str, ebno_db: float, *, batch_size: int,
                   slot_seconds: float, max_power_w: float) -> float:
    energy = delivered = 0.0
    for r in records:
        if r["method"] != method or float(r["ebno_db"]) != float(ebno_db):
            continue
        for on, p, bits, errs in zip(r["active"], r["power"], r["bits"], r["bit_errors"]):
            if on:
                energy += float(p or 0.0) * max_power_w * slot_seconds * batch_size
                delivered += float(bits) - float(errs)
    return energy / delivered if delivered > 0 else float("nan")


def tail_bler(records: Iterable[dict[str, Any]], method: str, ebno_db: float, target: float) -> dict[str, float]:
    per_batch = []
    for r in records:
        if r["method"] != method or float(r["ebno_db"]) != float(ebno_db):
            continue
        blocks = sum(v for v in r["blocks"] if v is not None)
        if blocks:
            per_batch.append(sum(v for v in r["block_errors"] if v is not None) / blocks)
    if not per_batch:
        return {"bler_p95": float("nan"), "bler_p99": float("nan"), "bler_violation_rate": float("nan")}
    v = np.asarray(per_batch)
    return {"bler_p95": float(np.percentile(v, 95)), "bler_p99": float(np.percentile(v, 99)), "bler_violation_rate": float(np.mean(v > target))}


def slot_seconds(meta: dict[str, Any], num_ofdm_symbols: int = 14, cyclic_prefix_length: int = 20) -> float:
    fft = float(meta.get("fft_size", 128))
    scs = float(meta.get("subcarrier_spacing", 30e3))
    return num_ofdm_symbols * (1.0 + cyclic_prefix_length / fft) / scs


def service_rows(trace: dict[str, Any], *, budgets: Sequence[int] = (1, 2, 4), survival_intervals: int = 1,
                 bler_target: float = 0.01, max_power_dbm: float = 23.0) -> list[dict[str, Any]]:
    meta, records = trace["header"]["meta"], trace["methods"]
    slot = slot_seconds(meta)
    p_max = 10 ** ((max_power_dbm - 30.0) / 10.0)
    rows = []
    for method in meta["methods"]:
        for eb in meta["ebno_db_range"]:
            delivered, _ = packet_outcomes(records, method, eb)
            if delivered.size == 0:
                continue
            row: dict[str, Any] = {"stage": trace["header"]["stage"], "method": method, "ebno_db": float(eb),
                                   "intervals": int(delivered.shape[0]), "packet_delivery": float(delivered.mean())}
            for k in budgets:
                row[f"reliability_within_{k}"] = reliability_within_budget(delivered, k)
            surv = survival(delivered, survival_intervals)
            row[f"survival_violations_s{survival_intervals}"] = surv["violations"]
            row[f"availability_s{survival_intervals}"] = surv["availability"]
            row["energy_nj_per_bit"] = 1e9 * energy_per_bit(records, method, eb, batch_size=int(meta.get("batch_size", 20)),
                                                             slot_seconds=slot, max_power_w=p_max)
            row.update(tail_bler(records, method, eb, bler_target))
            rows.append(row)
    return rows


# ---------------------------------------------------------------- Industry 5.0 classes and events
def class_service_rows(trace: dict[str, Any]) -> list[dict[str, Any]]:
    """Per device class: reliability within the class latency budget against its target, with a verdict."""
    from factory6g.analysis.reliability import clopper_pearson

    meta, records = trace["header"]["meta"], trace["methods"]
    industry = meta.get("industry")
    if not industry:
        return []
    classes = industry["device_classes"]
    specs = industry["class_specs"]
    rows = []
    for method in meta["methods"]:
        for eb in meta["ebno_db_range"]:
            delivered, _ = packet_outcomes(records, method, eb)
            if delivered.size == 0:
                continue
            for name in dict.fromkeys(classes):
                uts = [u for u, c in enumerate(classes) if c == name]
                spec = specs[uts[0]]
                budget, target, surv = int(spec["latency_intervals"]), float(spec["reliability"]), int(spec["survival_intervals"])
                batches = delivered.shape[0]
                if batches < budget:
                    continue
                ok = np.concatenate([np.any(np.stack([delivered[k:batches - budget + 1 + k, u] for k in range(budget)]), axis=0) for u in uts])
                n, successes = int(ok.size), int(ok.sum())
                lo, hi = clopper_pearson(successes, n)
                verdict = "met" if lo >= target else "missed" if hi < target else "inconclusive"
                s = survival(delivered[:, uts], surv)
                rows.append({"stage": trace["header"]["stage"], "method": method, "ebno_db": float(eb), "device_class": name,
                             "label": spec.get("label", name), "latency_intervals": budget, "target": target, "packets": n,
                             "reliability": successes / n if n else float("nan"), "reliability_lo": lo, "reliability_hi": hi,
                             "verdict": verdict, "survival_intervals": surv, "survival_violations": s["violations"],
                             "availability": s["availability"],
                             # Packets a zero-failure run needs before the target can be confirmed at 95 % (rule of three).
                             "packets_needed": int(np.ceil(3.0 / max(1e-12, 1.0 - target)))})
    return rows


def event_recovery_rows(trace: dict[str, Any], *, window: int = 3, level: float = 0.95) -> list[dict[str, Any]]:
    """Per scripted event: delivery before and during it, and intervals after it ends until delivery recovers."""
    meta, records = trace["header"]["meta"], trace["methods"]
    industry = meta.get("industry")
    if not industry or not industry.get("events"):
        return []
    rows = []
    for method in meta["methods"]:
        for eb in meta["ebno_db_range"]:
            delivered, batches = packet_outcomes(records, method, eb)
            if delivered.size == 0:
                continue
            ratio = delivered.mean(axis=1)
            for e in industry["events"]:
                start, end = int(e["start"]), int(e["end"])
                before = ratio[batches < start]
                during = ratio[(batches >= start) & (batches <= end)]
                after = ratio[batches > end]
                pre = float(before.mean()) if before.size else float("nan")
                recovery = None
                if np.isfinite(pre) and after.size >= window:
                    rolling = np.convolve(after, np.ones(window) / window, mode="valid")
                    hits = np.flatnonzero(rolling >= level * pre)
                    recovery = int(hits[0]) if hits.size else None
                rows.append({"stage": trace["header"]["stage"], "method": method, "ebno_db": float(eb), "event": e["type"],
                             "start": start, "end": end, "delivery_before": pre,
                             "delivery_during": float(during.mean()) if during.size else float("nan"),
                             "recovery_intervals_after_end": recovery})
    return rows
