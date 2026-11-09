"""Equivalence gate for the PyTorch port (Phase 4).

Compares a run on the PyTorch stack (Sionna 2.2) with the seed spread of the
TensorFlow stack (Sionna 1.2.1, Phase 1 seeds 42-46) on the same config.

The two stacks draw different random numbers, so the same seed does not give
the same batches; equality is not expected. The question is whether the port
moves BER by more than the run-to-run spread of the old stack. Per method and
Eb/N0:

* ``within_seed_range``: the PyTorch BER lies between the lowest and highest
  TensorFlow seed BER, widened by each side's own batch interval;
* ``z``: (BER_torch - mean_tf) / sqrt(sd_tf^2 + se_torch^2), where sd_tf is
  the across-seed standard deviation and se_torch the batch standard error of
  the PyTorch run;
* verdict ``equivalent`` when |z| <= 3 or within_seed_range; ``not resolved``
  when both sides have fewer than 30 bit errors in total; else ``differs``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from factory6g.analysis.reliability import find_stage_traces
from factory6g.analysis.stats import MIN_RESOLVED_BIT_ERRORS, batch_series, mean_ci
from factory6g.sim.trace import read_trace

Z_LIMIT = 3.0


def _points(trace_path: Path) -> dict[tuple[str, float], dict[str, float]]:
    trace = read_trace(trace_path)
    meta, records = trace["header"]["meta"], trace["methods"]
    out = {}
    for method in meta["methods"]:
        for eb in meta["ebno_db_range"]:
            s = batch_series(records, method, eb)
            be, bits = float(s.bit_errors.sum()), float(s.bits.sum())
            mean, lo, hi = mean_ci(s.ber)
            se = (hi - mean) / 1.959963984540054 if np.isfinite(hi) else float("nan")
            out[(method, float(eb))] = {"ber": be / bits if bits else float("nan"), "bit_errors": be,
                                        "lo": max(0.0, lo) if np.isfinite(lo) else float("nan"),
                                        "hi": hi, "se": se, "seed": int(meta.get("seed", -1))}
    return out


def compare(reference_runs: Sequence[Path], candidate_runs: Sequence[Path], *, stage: str) -> dict[str, Any]:
    ref = [_points(p) for p in find_stage_traces(reference_runs, stage)]
    cand = [_points(p) for p in find_stage_traces(candidate_runs, stage)]
    if not ref or not cand:
        raise FileNotFoundError(f"Need traces of stage '{stage}' on both sides (reference {len(ref)}, candidate {len(cand)}).")
    rows = []
    for key in sorted(cand[0], key=lambda k: (k[0], k[1])):
        if not all(key in r for r in ref):
            continue
        c = cand[0][key]
        tf_ber = np.array([r[key]["ber"] for r in ref])
        tf_lo = np.array([r[key]["lo"] for r in ref])
        tf_hi = np.array([r[key]["hi"] for r in ref])
        mean = float(np.nanmean(tf_ber))
        sd = float(np.nanstd(tf_ber, ddof=1)) if len(ref) > 1 else 0.0
        se = c["se"] if np.isfinite(c["se"]) else 0.0
        denom = math.sqrt(sd ** 2 + se ** 2)
        z = (c["ber"] - mean) / denom if denom > 0 else (0.0 if c["ber"] == mean else float("inf"))
        lo_band = float(np.nanmin(np.where(np.isfinite(tf_lo), tf_lo, tf_ber)))
        hi_band = float(np.nanmax(np.where(np.isfinite(tf_hi), tf_hi, tf_ber)))
        c_lo = c["lo"] if np.isfinite(c["lo"]) else c["ber"]
        c_hi = c["hi"] if np.isfinite(c["hi"]) else c["ber"]
        within = bool(c_hi >= lo_band and c_lo <= hi_band)
        resolved = c["bit_errors"] >= MIN_RESOLVED_BIT_ERRORS or float(np.mean([r[key]["bit_errors"] for r in ref])) >= MIN_RESOLVED_BIT_ERRORS
        verdict = "not resolved" if not resolved else ("equivalent" if within or abs(z) <= Z_LIMIT else "differs")
        rows.append({"method": key[0], "ebno_db": key[1], "ber_torch": c["ber"], "ber_torch_lo": c["lo"], "ber_torch_hi": c["hi"],
                     "ber_tf_mean": mean, "ber_tf_sd": sd, "ber_tf_min": float(np.nanmin(tf_ber)), "ber_tf_max": float(np.nanmax(tf_ber)),
                     "tf_seeds": [r[key]["seed"] for r in ref], "z": z, "within_seed_range": within, "verdict": verdict})
    counts = {v: sum(r["verdict"] == v for r in rows) for v in ("equivalent", "differs", "not resolved")}
    return {"stage": stage, "reference_runs": [str(p) for p in reference_runs], "candidate_runs": [str(p) for p in candidate_runs],
            "z_limit": Z_LIMIT, "counts": counts, "passed": counts["differs"] == 0, "rows": rows}


def write(result: dict[str, Any], out_dir: Path) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stage = result["stage"]
    js = out_dir / f"equivalence_{stage}.json"
    js.write_text(json.dumps(result, indent=2), encoding="utf-8")
    f = lambda x: "–" if x is None or not np.isfinite(x) else f"{x:.3g}"  # noqa: E731
    c = result["counts"]
    lines = [f"# PyTorch port against TensorFlow seed spread: {stage} stage", "",
             f"Reference (Sionna 1.2.1 / TensorFlow): {len(result['rows'][0]['tf_seeds']) if result['rows'] else 0} seeds. "
             f"Candidate: Sionna 2.2 / PyTorch. Gate {'passed' if result['passed'] else 'FAILED'}: "
             f"{c['equivalent']} equivalent, {c['differs']} differ, {c['not resolved']} not resolved (|z| limit {result['z_limit']:g}).", "",
             "| Method | Eb/N0 (dB) | BER PyTorch [95 % batch CI] | BER TF mean (sd) | TF seed range | z | Verdict |",
             "|:--|--:|:--|:--|:--|--:|:--|"]
    for r in result["rows"]:
        lines.append(f"| {r['method']} | {r['ebno_db']:g} | {f(r['ber_torch'])} [{f(r['ber_torch_lo'])}, {f(r['ber_torch_hi'])}] | "
                     f"{f(r['ber_tf_mean'])} ({f(r['ber_tf_sd'])}) | {f(r['ber_tf_min'])}–{f(r['ber_tf_max'])} | {f(r['z'])} | {r['verdict']} |")
    md = out_dir / f"equivalence_{stage}.md"
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return [js, md]


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="Compare a PyTorch-stack run with the TensorFlow seed spread.")
    p.add_argument("--reference", nargs="+", required=True, help="TensorFlow run folders (Phase 1 seeds).")
    p.add_argument("--candidate", nargs="+", required=True, help="PyTorch run folder(s).")
    p.add_argument("--stage", required=True, help="Stage name in the trace, e.g. estimators or resource_managers.")
    p.add_argument("--out", default="results/phase4/equivalence")
    a = p.parse_args(argv)
    res = compare([Path(x) for x in a.reference], [Path(x) for x in a.candidate], stage=a.stage)
    for path in write(res, Path(a.out)):
        print(f"Wrote {path}")
    print(json.dumps(res["counts"]))
    return 0 if res["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
