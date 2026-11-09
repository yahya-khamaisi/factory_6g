"""Reliability report over one or more traced runs (Phase 1).

For every stage, method and Eb/N0 point of each run it reports:

* BER pooled over all bits, with a **batch-level** 95 % interval (spread of
  per-batch BER; a batch is one hall draw, so it is the unit of evidence);
* BLER with an exact Clopper-Pearson 95 % interval over codewords. Codewords
  of one batch share the hall draw, so this interval is optimistic; it is
  reported next to the batch-level one, not instead of it;
* evidence status by bit errors (the stage's rule) and by block errors;
* against a baseline method: the paired per-batch BLER difference with a
  normal and a bootstrap 95 % interval and a paired-t p-value, Holm-corrected
  over the Eb/N0 points of each comparison, and an exact McNemar test over the
  codewords both methods carried on the same terminal in the same batch.

Across runs that differ only in the seed it adds the seed-level picture: mean
and 95 % t-interval of per-seed BER, between-seed standard deviation, the
interquartile mean, the probability that a seed of the method beats a seed of
the baseline, and in how many seeds the method beat the baseline.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from factory6g.analysis.service import class_service_rows, event_recovery_rows, service_rows
from factory6g.analysis.stats import MIN_RESOLVED_BIT_ERRORS, batch_series, mean_ci
from factory6g.sim.trace import TRACE_FILENAME, read_trace, unpack_block_flags

BOOTSTRAP_RESAMPLES = 2000


# ---------------------------------------------------------------- primitives
def clopper_pearson(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    from scipy.stats import beta

    if n <= 0:
        return float("nan"), float("nan")
    alpha = 1.0 - confidence
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


def paired_t_pvalue(diff: np.ndarray) -> float:
    from scipy.stats import t

    d = np.asarray(diff, dtype=float)
    d = d[np.isfinite(d)]
    if d.size < 2:
        return float("nan")
    sd = d.std(ddof=1)
    if sd <= 1e-12 * max(1.0, float(np.abs(d).max())):  # all differences equal
        return 1.0 if d.mean() == 0 else 0.0
    stat = d.mean() / (sd / math.sqrt(d.size))
    return float(2 * t.sf(abs(stat), d.size - 1))


def bootstrap_ci(diff: np.ndarray, *, resamples: int = BOOTSTRAP_RESAMPLES, seed: int = 0) -> tuple[float, float]:
    d = np.asarray(diff, dtype=float)
    d = d[np.isfinite(d)]
    if d.size < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, d.size, size=(resamples, d.size))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def mcnemar_exact(method_only_errors: int, baseline_only_errors: int) -> float:
    """Two-sided exact McNemar p-value from the discordant codeword counts."""
    from scipy.stats import binomtest

    n = method_only_errors + baseline_only_errors
    if n == 0:
        return 1.0
    return float(binomtest(min(method_only_errors, baseline_only_errors), n, 0.5).pvalue)


def holm(pvalues: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (NaN entries are left out of the family)."""
    p = np.asarray(pvalues, dtype=float)
    out = np.full(p.shape, np.nan)
    idx = np.flatnonzero(np.isfinite(p))
    if idx.size == 0:
        return out.tolist()
    order = idx[np.argsort(p[idx])]
    m = idx.size
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        out[i] = running
    return out.tolist()


def interquartile_mean(values: Sequence[float]) -> float:
    from scipy.stats import trim_mean

    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    return float(trim_mean(v, 0.25)) if v.size else float("nan")


def probability_of_improvement(method: Sequence[float], baseline: Sequence[float]) -> float:
    """P(a seed of the method has lower BER than a seed of the baseline), ties count half."""
    a, b = np.asarray(method, dtype=float), np.asarray(baseline, dtype=float)
    if a.size == 0 or b.size == 0:
        return float("nan")
    less = (a[:, None] < b[None, :]).mean()
    ties = (a[:, None] == b[None, :]).mean()
    return float(less + 0.5 * ties)


# ---------------------------------------------------------------- per run
def _mcnemar_counts(records: list[dict[str, Any]], method: str, baseline: str, ebno: float) -> tuple[int, int, int]:
    """Discordant codeword counts over terminals both methods carried in the same batch."""
    def index(m: str) -> dict[int, dict[str, Any]]:
        return {r["batch"]: r for r in records if r["method"] == m and float(r["ebno_db"]) == ebno}

    a_by, b_by = index(method), index(baseline)
    only_a = only_b = shared = 0
    for batch, ra in a_by.items():
        rb = b_by.get(batch)
        if rb is None or "block_error_flags" not in ra:
            continue
        for u, (fa, fb) in enumerate(zip(ra["block_error_flags"], rb["block_error_flags"])):
            if fa is None or fb is None:
                continue
            n = int(ra["blocks"][u])
            ea, eb = unpack_block_flags(fa, n), unpack_block_flags(fb, n)
            only_a += int(np.sum(ea & ~eb))
            only_b += int(np.sum(eb & ~ea))
            shared += n
    return only_a, only_b, shared


def _min_error_batches(stage_dir: Path) -> int:
    results = stage_dir / "stage_results_v2.json"
    if results.exists():
        snap = json.loads(results.read_text(encoding="utf-8")).get("config_snapshot", {})
        return int(snap.get("monte_carlo", {}).get("min_error_batches", 0) or 0)
    return 0


def stage_report(trace_path: Path, *, baseline: str | None) -> dict[str, Any]:
    trace = read_trace(trace_path)
    meta, records, stage = trace["header"]["meta"], trace["methods"], trace["header"]["stage"]
    min_error_batches = _min_error_batches(trace_path.parent)
    methods = list(meta["methods"])
    base = baseline if baseline in methods else None
    rows: list[dict[str, Any]] = []
    for method in methods:
        for eb in meta["ebno_db_range"]:
            s = batch_series(records, method, eb)
            be, bits = int(s.bit_errors.sum()), int(s.bits.sum())
            ble, blocks = int(s.block_errors.sum()), int(s.blocks.sum())
            _, ber_lo, ber_hi = mean_ci(s.ber)
            bler_lo, bler_hi = clopper_pearson(ble, blocks)
            error_batches = int(np.sum(s.bit_errors > 0))
            enough_batches = error_batches >= min_error_batches
            row: dict[str, Any] = {
                "stage": stage, "method": method, "ebno_db": float(eb), "batches": int(s.batch.size),
                "bit_errors": be, "bits": bits, "block_errors": ble, "blocks": blocks, "error_batches": error_batches,
                "ber": be / bits if bits else float("nan"),
                "ber_batch_lo": max(0.0, ber_lo) if np.isfinite(ber_lo) else float("nan"), "ber_batch_hi": ber_hi,
                "bler": ble / blocks if blocks else float("nan"), "bler_cp_lo": bler_lo, "bler_cp_hi": bler_hi,
                "status_bits": "resolved" if be >= MIN_RESOLVED_BIT_ERRORS and enough_batches else "upper_bound_only",
                "status_blocks": "resolved" if ble >= MIN_RESOLVED_BIT_ERRORS and enough_batches else "upper_bound_only",
            }
            if base and method != base:
                sb = batch_series(records, base, eb)
                common, ia, ib = np.intersect1d(s.batch, sb.batch, return_indices=True)
                diff = s.bler[ia] - sb.bler[ib]
                mean, lo, hi = mean_ci(diff)
                blo, bhi = bootstrap_ci(diff, seed=int(eb * 10) + methods.index(method))
                only_m, only_b, shared = _mcnemar_counts(records, method, base, float(eb))
                row.update({
                    "baseline": base, "paired_batches": int(common.size),
                    "d_bler": mean, "d_bler_lo": lo, "d_bler_hi": hi, "d_bler_boot_lo": blo, "d_bler_boot_hi": bhi,
                    "p_paired_t": paired_t_pvalue(diff),
                    "mcnemar_method_only": only_m, "mcnemar_baseline_only": only_b, "mcnemar_shared_codewords": shared,
                    "p_mcnemar": mcnemar_exact(only_m, only_b) if shared else float("nan"),
                })
            rows.append(row)
    if base:
        for method in methods:
            if method == base:
                continue
            fam = [r for r in rows if r["method"] == method]
            for r, adj in zip(fam, holm([r["p_paired_t"] for r in fam])):
                r["p_holm"] = adj
                r["verdict"] = _verdict(r)
    return {"stage": stage, "seed": int(meta.get("seed", -1)), "methods": methods, "baseline": base,
            "ebno_db_range": [float(e) for e in meta["ebno_db_range"]], "rows": rows,
            "service": service_rows(trace), "classes": class_service_rows(trace), "events": event_recovery_rows(trace)}


def _verdict(row: dict[str, Any]) -> str:
    p = row.get("p_holm", float("nan"))
    if not np.isfinite(p):
        return "not testable"
    if p >= 0.05:
        return "no clear difference"
    return "fewer block errors" if row["d_bler"] < 0 else "more block errors"


# ---------------------------------------------------------------- across seeds
def seed_summary(stage_reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    base = stage_reports[0]["baseline"]
    per: dict[tuple[str, float], dict[int, float]] = defaultdict(dict)
    for rep in stage_reports:
        for r in rep["rows"]:
            per[(r["method"], r["ebno_db"])][rep["seed"]] = r["ber"]
    out = []
    from scipy.stats import t

    for (method, eb), by_seed in sorted(per.items(), key=lambda kv: (stage_reports[0]["methods"].index(kv[0][0]), kv[0][1])):
        seeds = sorted(by_seed)
        v = np.array([by_seed[s] for s in seeds], dtype=float)
        n = int(np.isfinite(v).sum())
        mean = float(np.nanmean(v)) if n else float("nan")
        sd = float(np.nanstd(v, ddof=1)) if n > 1 else float("nan")
        half = float(t.ppf(0.975, n - 1) * sd / math.sqrt(n)) if n > 1 else float("nan")
        row = {"method": method, "ebno_db": eb, "seeds": seeds, "ber_per_seed": v.tolist(), "ber_seed_mean": mean,
               "ber_seed_sd": sd, "ber_seed_lo": max(0.0, mean - half) if np.isfinite(half) else float("nan"),
               "ber_seed_hi": mean + half if np.isfinite(half) else float("nan"), "ber_iqm": interquartile_mean(v)}
        if base and method != base:
            b = per.get((base, eb), {})
            bv = np.array([b[s] for s in seeds if s in b], dtype=float)
            row["prob_improvement"] = probability_of_improvement(v, bv)
            row["seeds_better"] = int(sum(1 for s in seeds if s in b and by_seed[s] < b[s]))
            row["seeds_compared"] = int(sum(1 for s in seeds if s in b))
        out.append(row)
    return out


# ---------------------------------------------------------------- report
def find_stage_traces(run_dirs: Iterable[Path], stage: str | None) -> list[Path]:
    paths = []
    for run in run_dirs:
        for p in sorted(Path(run).rglob(TRACE_FILENAME)):
            if stage is None or read_trace(p)["header"]["stage"] == stage:
                paths.append(p)
    return paths


def build_report(run_dirs: Sequence[Path], *, baseline: str | None = None, stage: str | None = None) -> dict[str, Any]:
    traces = find_stage_traces(run_dirs, stage)
    if not traces:
        raise FileNotFoundError("No matching trace files; run with simulation.trace on.")
    reports = [stage_report(p, baseline=baseline) for p in traces]
    by_stage: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rep, path in zip(reports, traces):
        rep["trace"] = str(path)
        by_stage[rep["stage"]].append(rep)
    out = {"runs": [str(r) for r in run_dirs], "stages": {}}
    for name, reps in by_stage.items():
        seeds = sorted({r["seed"] for r in reps})
        out["stages"][name] = {
            "seeds": seeds,
            "per_seed": reps,
            "across_seeds": seed_summary(reps) if len(seeds) > 1 else [],
        }
    return out


def _fmt(x: Any) -> str:
    if isinstance(x, float):
        if not np.isfinite(x):
            return "–"
        return f"{x:.3e}" if (x != 0 and (abs(x) < 1e-2 or abs(x) >= 1e3)) else f"{x:.4g}"
    return str(x)


def write_report(report: dict[str, Any], out_dir: Path) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    path = out_dir / "reliability_report.json"
    path.write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")
    written.append(path)
    md = ["# Reliability report", "", f"Runs: {', '.join(Path(r).name for r in report['runs'])}", ""]
    for name, st in report["stages"].items():
        rows = [r for rep in st["per_seed"] for r in rep["rows"]]
        csv_path = out_dir / f"reliability_{name}_points.csv"
        keys = sorted({k for r in rows for k in r} | {"seed"})
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            for rep in st["per_seed"]:
                for r in rep["rows"]:
                    writer.writerow({**r, "seed": rep["seed"]})
        written.append(csv_path)
        service = [dict(r, seed=rep["seed"]) for rep in st["per_seed"] for r in rep.get("service", [])]
        if service:
            svc_path = out_dir / f"service_{name}.csv"
            with svc_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=sorted({k for r in service for k in r}))
                writer.writeheader()
                writer.writerows(service)
            written.append(svc_path)
        md += [f"## Stage `{name}`", "", f"Seeds: {', '.join(map(str, st['seeds']))}. Baseline: `{st['per_seed'][0]['baseline']}`.", ""]
        first = st["per_seed"][0]
        tested = [r for r in first["rows"] if "verdict" in r]
        if tested:
            md += [f"### Paired tests against the baseline, seed {first['seed']}", "",
                   "Per Eb/N0 point, Holm-corrected over the points of each method; the paired difference is per-batch BLER.", "",
                   "| Method | Fewer block errors | More block errors | No clear difference | Not testable |", "|:--|--:|--:|--:|--:|"]
            for m in first["methods"]:
                mr = [r for r in tested if r["method"] == m]
                if not mr:
                    continue
                c = lambda v: sum(1 for r in mr if r["verdict"] == v)  # noqa: E731
                md.append(f"| {m} | {c('fewer block errors')} | {c('more block errors')} | {c('no clear difference')} | {c('not testable')} |")
            md.append("")
        for key, title in (("classes", "industry_classes"), ("events", "industry_events")):
            extra = [dict(r, seed=rep["seed"]) for rep in st["per_seed"] for r in rep.get(key, [])]
            if extra:
                path_x = out_dir / f"{title}_{name}.csv"
                with path_x.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=sorted({k for r in extra for k in r}))
                    writer.writeheader()
                    writer.writerows(extra)
                written.append(path_x)
        classes_first = st["per_seed"][0].get("classes", [])
        if classes_first:
            md += [f"### Device classes, seed {first['seed']}", "",
                   "Reliability within the class latency budget against the class target. *met*: the lower 95 % bound clears the target; "
                   "*missed*: the upper bound is below it; *inconclusive*: the run has too few packets to decide (see *packets needed*).", "",
                   "| Method | Eb/N0 | Class | Budget | Target | Reliability | 95 % | Verdict | Packets | Packets needed | Survival violations |",
                   "|:--|--:|:--|--:|--:|--:|:--|:--|--:|--:|--:|"]
            for r in classes_first:
                md.append(f"| {r['method']} | {r['ebno_db']:g} | {r['label']} | {r['latency_intervals']} | {r['target']} | {_fmt(r['reliability'])} | "
                          f"{_fmt(r['reliability_lo'])} – {_fmt(r['reliability_hi'])} | {r['verdict']} | {r['packets']} | {r['packets_needed']} | {int(r['survival_violations'])} |")
            md.append("")
        events_first = st["per_seed"][0].get("events", [])
        if events_first:
            md += [f"### Disruptions, seed {first['seed']}", "", "Delivery is the share of terminals whose packet got through in an interval; recovery counts intervals after the event ends until a 3-interval mean is back at 95 % of the pre-event level.", "",
                   "| Method | Eb/N0 | Event | Batches | Delivery before | Delivery during | Recovery (intervals) |", "|:--|--:|:--|:--|--:|--:|--:|"]
            for r in events_first:
                rec = "not within the run" if r["recovery_intervals_after_end"] is None else r["recovery_intervals_after_end"]
                md.append(f"| {r['method']} | {r['ebno_db']:g} | {r['event']} | {r['start']}–{r['end']} | {_fmt(r['delivery_before'])} | {_fmt(r['delivery_during'])} | {rec} |")
            md.append("")
        first_service = st["per_seed"][0].get("service", [])
        if first_service:
            md += [f"### Service metrics, seed {first['seed']}", "",
                   "Each batch is one scheduling interval with one packet per terminal; latency budgets count intervals. "
                   "Availability uses a survival time of one interval.", "",
                   "| Method | Eb/N0 | Delivered | Within 1 | Within 2 | Within 4 | Availability | Survival violations | Energy (nJ/bit) | BLER p99 | Batches over 1 % BLER |",
                   "|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
            for r in first_service:
                md.append(f"| {r['method']} | {r['ebno_db']:g} | {_fmt(r['packet_delivery'])} | {_fmt(r['reliability_within_1'])} | {_fmt(r['reliability_within_2'])} | "
                          f"{_fmt(r['reliability_within_4'])} | {_fmt(r['availability_s1'])} | {int(r['survival_violations_s1'])} | {_fmt(r['energy_nj_per_bit'])} | "
                          f"{_fmt(r['bler_p99'])} | {_fmt(r['bler_violation_rate'])} |")
            md.append("")
        if st["across_seeds"]:
            seed_path = out_dir / f"reliability_{name}_seeds.csv"
            keys = sorted({k for r in st["across_seeds"] for k in r})
            with seed_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=keys)
                writer.writeheader()
                writer.writerows(st["across_seeds"])
            written.append(seed_path)
            md += ["### Across seeds", "", "BER mean over seeds with a 95 % t-interval; P(better) is the probability that a seed of the method has lower BER than a seed of the baseline.", "",
                   "| Method | Eb/N0 | Mean BER | 95 % over seeds | Seed SD | P(better) | Seeds better |", "|:--|--:|--:|:--|--:|--:|--:|"]
            for r in st["across_seeds"]:
                md.append(f"| {r['method']} | {r['ebno_db']:g} | {_fmt(r['ber_seed_mean'])} | {_fmt(r['ber_seed_lo'])} – {_fmt(r['ber_seed_hi'])} | {_fmt(r['ber_seed_sd'])} | "
                          f"{_fmt(r.get('prob_improvement', float('nan')))} | {r.get('seeds_better', '–')}/{r.get('seeds_compared', '–')} |")
            md.append("")
    md_path = out_dir / "reliability_report.md"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    written.append(md_path)
    return written
