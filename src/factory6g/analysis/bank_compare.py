"""Compare two ray-traced channel banks position by position (Phase 3).

The v2 bank is traced with Sionna RT 2.x from the same layout and position
seeds as v1, so both hold the same terminal positions. Per position this
reports the change in link gain, Rician K-factor and RMS delay spread, whether
the LOS flag agrees, and the correlation of the two frequency responses
|<h1, h2>| / (||h1|| ||h2||) over ports and subcarriers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def compare_banks(v1_path: Path, v2_path: Path) -> dict[str, Any]:
    v1_path, v2_path = Path(v1_path), Path(v2_path)
    m1 = json.loads(v1_path.with_name(v1_path.name + ".manifest.json").read_text(encoding="utf-8"))
    m2 = json.loads(v2_path.with_name(v2_path.name + ".manifest.json").read_text(encoding="utf-8"))
    out: dict[str, Any] = {"v1": str(v1_path), "v2": str(v2_path), "v1_tracer": m1.get("tracer", "sionna-rt 1.2.1 (not recorded)"),
                           "v2_tracer": m2.get("tracer"), "ray_tracing_v1": m1.get("ray_tracing"), "ray_tracing_v2": m2.get("ray_tracing"), "splits": {}}
    with np.load(v1_path) as a, np.load(v2_path) as b:
        for split in ("train", "eval"):
            p1, p2 = a[f"{split}_positions"], b[f"{split}_positions"]
            same = p1.shape == p2.shape and np.allclose(p1, p2, atol=1e-5)
            if not same:
                out["splits"][split] = {"same_positions": False}
                continue
            h1, h2 = a[f"{split}_h"], b[f"{split}_h"]
            e1 = np.mean(np.abs(h1) ** 2, axis=(1, 2))
            e2 = np.mean(np.abs(h2) ** 2, axis=(1, 2))
            dgain = 10 * np.log10(e2 / e1)
            corr = np.abs(np.sum(np.conj(h1) * h2, axis=(1, 2))) / np.sqrt(np.sum(np.abs(h1) ** 2, axis=(1, 2)) * np.sum(np.abs(h2) ** 2, axis=(1, 2)))
            k1, k2 = a[f"{split}_k_factor_db"].astype(float), b[f"{split}_k_factor_db"].astype(float)
            finite = np.isfinite(k1) & np.isfinite(k2)
            ds1, ds2 = a[f"{split}_rms_delay_spread_ns"].astype(float), b[f"{split}_rms_delay_spread_ns"].astype(float)
            los1, los2 = a[f"{split}_los"].astype(bool), b[f"{split}_los"].astype(bool)
            q = lambda v: [float(np.percentile(v, x)) for x in (5, 50, 95)]  # noqa: E731
            out["splits"][split] = {
                "same_positions": True, "count": int(len(p1)),
                "gain_change_db_p5_p50_p95": q(dgain),
                "gain_change_abs_gt_1db_fraction": float(np.mean(np.abs(dgain) > 1.0)),
                "cfr_correlation_p5_p50_p95": q(corr),
                "k_change_db_p5_p50_p95": q(k2[finite] - k1[finite]),
                "ds_change_ns_p5_p50_p95": q(ds2 - ds1),
                "los_flag_agreement": float(np.mean(los1 == los2)),
                "los_fraction_v1": float(los1.mean()), "los_fraction_v2": float(los2.mean()),
                "num_paths_median_v1": float(np.median(a[f"{split}_num_paths"])), "num_paths_median_v2": float(np.median(b[f"{split}_num_paths"])),
            }
    return out


def write_comparison(result: dict[str, Any], out_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from factory6g.visualization.thesis_plot_style import INK, THESIS_FIGSIZE, apply_thesis_rcparams

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    path = out_dir / "bank_v1_vs_v2.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    written.append(path)
    f = lambda v: " / ".join(f"{x:.3g}" for x in v)  # noqa: E731
    lines = ["# Channel bank v1 (Sionna RT 1.2.1) against v2 (Sionna RT 2.x)", "",
             f"v1 tracer: {result['v1_tracer']}. v2 tracer: {result['v2_tracer']}.", "",
             "Same layout and position seeds; values are 5th / 50th / 95th percentiles over positions.", "",
             "| Split | Positions | Gain change (dB) | |Δgain| > 1 dB | CFR correlation | K change (dB) | Delay-spread change (ns) | LOS flags agree | Paths (v1 → v2) |",
             "|:--|--:|:--|--:|:--|:--|:--|--:|:--|"]
    for split, s in result["splits"].items():
        if not s.get("same_positions"):
            lines.append(f"| {split} | positions differ | | | | | | | |")
            continue
        lines.append(f"| {split} | {s['count']} | {f(s['gain_change_db_p5_p50_p95'])} | {100 * s['gain_change_abs_gt_1db_fraction']:.1f} % | "
                     f"{f(s['cfr_correlation_p5_p50_p95'])} | {f(s['k_change_db_p5_p50_p95'])} | {f(s['ds_change_ns_p5_p50_p95'])} | "
                     f"{100 * s['los_flag_agreement']:.1f} % | {s['num_paths_median_v1']:.0f} → {s['num_paths_median_v2']:.0f} |")
    md = out_dir / "bank_v1_vs_v2.md"
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    written.append(md)

    apply_thesis_rcparams(plt)
    with np.load(result["v1"]) as a, np.load(result["v2"]) as b:
        e1 = 10 * np.log10(np.mean(np.abs(a["eval_h"]) ** 2, axis=(1, 2)))
        e2 = 10 * np.log10(np.mean(np.abs(b["eval_h"]) ** 2, axis=(1, 2)))
        ds1, ds2 = a["eval_rms_delay_spread_ns"], b["eval_rms_delay_spread_ns"]
    fig, axes = plt.subplots(1, 2, figsize=(THESIS_FIGSIZE[0] * 1.6, THESIS_FIGSIZE[1]))
    axes[0].scatter(e1, e2, s=3, color=INK, alpha=0.4, linewidths=0)
    lo, hi = min(e1.min(), e2.min()), max(e1.max(), e2.max())
    axes[0].plot([lo, hi], [lo, hi], color="#cc6677", linewidth=1)
    axes[0].set_xlabel("Link gain, v1 (dB)"); axes[0].set_ylabel("Link gain, v2 (dB)")
    axes[1].scatter(ds1, ds2, s=3, color=INK, alpha=0.4, linewidths=0)
    m = max(ds1.max(), ds2.max())
    axes[1].plot([0, m], [0, m], color="#cc6677", linewidth=1)
    axes[1].set_xlabel("RMS delay spread, v1 (ns)"); axes[1].set_ylabel("RMS delay spread, v2 (ns)")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        p = out_dir / f"fig_bank_v1_vs_v2.{ext}"
        fig.savefig(p); written.append(p)
    plt.close(fig)
    return written


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Compare two channel banks position by position.")
    parser.add_argument("--v1", default="data/rt_channel_bank.npz")
    parser.add_argument("--v2", default="data/rt_channel_bank_v2.npz")
    parser.add_argument("--out", default="results/phase3/bank_compare")
    args = parser.parse_args(argv)
    for p in write_comparison(compare_banks(Path(args.v1), Path(args.v2)), Path(args.out)):
        print(f"Wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
