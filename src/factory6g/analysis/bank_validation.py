"""Validate the ray-traced channel bank against 3GPP TR 38.901 indoor-factory statistics (Phase 2).

Compares, per split and for LOS and NLOS positions, the bank's RMS delay
spread, Rician K-factor and path-loss exponent with the InF parameters:

* delay spread: lognormal with mean lg(DS) = log10(26 V/S + 14) - 9.35 (LOS)
  and log10(30 V/S + 32) - 9.44 (NLOS), sigma 0.15 and 0.19, where V and S are
  the hall's volume and surface area (TR 38.901 Table 7.5-6 Part-3);
* K-factor (LOS): normal in dB with mean 7 dB and sigma 8 dB (same table);
* path-loss exponent: 2.15 (InF-SL LOS) and 2.55 (InF-SL NLOS) from
  Table 7.4.1-1.

The 3GPP values are written from the specification tables as recalled for this
tool; check them against the edition you cite before quoting the comparison.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

INF = {
    "ds_los": lambda vs: (np.log10(26.0 * vs + 14.0) - 9.35, 0.15),
    "ds_nlos": lambda vs: (np.log10(30.0 * vs + 32.0) - 9.44, 0.19),
    "k_los_db": (7.0, 8.0),
    "ple_los": 2.15,
    "ple_nlos": 2.55,
}


def hall_volume_to_surface(room: list[float]) -> float:
    length, width, height = (float(v) for v in room)
    volume = length * width * height
    surface = 2.0 * (length * width + length * height + width * height)
    return volume / surface


def _ple(distance: np.ndarray, gain_db: np.ndarray) -> float:
    if distance.size < 3:
        return float("nan")
    slope, _ = np.polyfit(10.0 * np.log10(distance), gain_db, 1)
    return float(-slope)


def validate_bank(bank_path: Path) -> dict[str, Any]:
    bank_path = Path(bank_path)
    manifest = json.loads(bank_path.with_name(bank_path.name + ".manifest.json").read_text(encoding="utf-8"))
    hall = manifest["hall"]
    vs = hall_volume_to_surface(hall["room_dimensions"])
    bs = np.asarray(hall["bs_position"], dtype=float)
    out: dict[str, Any] = {"bank": str(bank_path), "volume_to_surface_m": vs, "splits": {}}
    with np.load(bank_path) as data:
        for split in ("train", "eval"):
            if f"{split}_positions" not in data:
                continue
            pos = data[f"{split}_positions"].astype(float)
            los = data[f"{split}_los"].astype(bool)
            ds = data[f"{split}_rms_delay_spread_ns"].astype(float)
            k = data[f"{split}_k_factor_db"].astype(float)
            gain_db = 10.0 * np.log10(np.mean(np.abs(data[f"{split}_h"]) ** 2, axis=(1, 2)))
            d = np.linalg.norm(pos - bs, axis=1)
            rows = {}
            for state, m in (("los", los), ("nlos", ~los)):
                if not m.any():
                    continue
                mu, sigma = INF[f"ds_{state}"](vs)
                lg = np.log10(np.maximum(ds[m], 1e-3) * 1e-9)
                row = {
                    "count": int(m.sum()),
                    "ds_median_ns": float(np.median(ds[m])),
                    "ds_lg_mean": float(lg.mean()), "ds_lg_std": float(lg.std()),
                    "inf_ds_median_ns": float(10 ** mu * 1e9), "inf_ds_lg_mean": float(mu), "inf_ds_lg_std": float(sigma),
                    "ple": _ple(d[m], gain_db[m]), "inf_ple": INF[f"ple_{state}"],
                }
                if state == "los":
                    row.update({"k_median_db": float(np.median(k[m])), "k_mean_db": float(k[m].mean()), "k_std_db": float(k[m].std()),
                                "inf_k_mean_db": INF["k_los_db"][0], "inf_k_std_db": INF["k_los_db"][1]})
                rows[state] = row
            out["splits"][split] = {"los_fraction": float(los.mean()), **rows}
    return out


def write_validation(result: dict[str, Any], out_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import norm

    from factory6g.visualization.thesis_plot_style import INK, MUTED, THESIS_FIGSIZE, apply_thesis_rcparams

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    path = out_dir / "bank_validation.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    written.append(path)

    lines = ["# Channel bank against TR 38.901 InF", "",
             f"Hall volume-to-surface ratio V/S = {result['volume_to_surface_m']:.2f} m. 3GPP values are from TR 38.901 Tables 7.5-6 and 7.4.1-1; verify against the cited edition.", "",
             "| Split | State | Positions | DS median (ns) | InF DS median (ns) | lg DS std | InF std | K mean (dB) | InF K mean | PLE | InF PLE |",
             "|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for split, s in result["splits"].items():
        for state in ("los", "nlos"):
            r = s.get(state)
            if not r:
                continue
            k = f"{r['k_mean_db']:.1f}" if "k_mean_db" in r else "–"
            ik = f"{r['inf_k_mean_db']:.0f}" if "inf_k_mean_db" in r else "–"
            lines.append(f"| {split} | {state.upper()} | {r['count']} | {r['ds_median_ns']:.1f} | {r['inf_ds_median_ns']:.1f} | {r['ds_lg_std']:.2f} | "
                         f"{r['inf_ds_lg_std']:.2f} | {k} | {ik} | {r['ple']:.2f} | {r['inf_ple']:.2f} |")
    md = out_dir / "bank_validation.md"
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    written.append(md)

    apply_thesis_rcparams(plt)
    bank = np.load(result["bank"])
    los = bank["eval_los"].astype(bool)
    fig, axes = plt.subplots(1, 2, figsize=(THESIS_FIGSIZE[0] * 1.6, THESIS_FIGSIZE[1]))
    ds = np.sort(bank["eval_rms_delay_spread_ns"][los].astype(float))
    axes[0].step(ds, np.arange(1, ds.size + 1) / ds.size, where="post", color=INK, label="Ray-traced bank, LOS")
    mu, sigma = INF["ds_los"](result["volume_to_surface_m"])
    grid = np.logspace(np.log10(max(ds.min() * 0.5, 0.5)), np.log10(ds.max() * 10), 200)
    axes[0].plot(grid, norm.cdf((np.log10(grid * 1e-9) - mu) / sigma), color=MUTED, linestyle="--", label="TR 38.901 InF, LOS")
    axes[0].set_xscale("log"); axes[0].set_xlabel("RMS delay spread (ns)"); axes[0].set_ylabel("CDF"); axes[0].legend(fontsize=9)
    k = np.sort(bank["eval_k_factor_db"][los].astype(float))
    axes[1].step(k, np.arange(1, k.size + 1) / k.size, where="post", color=INK, label="Ray-traced bank, LOS")
    kg = np.linspace(-15, 30, 200)
    axes[1].plot(kg, norm.cdf((kg - INF["k_los_db"][0]) / INF["k_los_db"][1]), color=MUTED, linestyle="--", label="TR 38.901 InF, LOS")
    axes[1].set_xlabel("Rician K-factor (dB)"); axes[1].legend(fontsize=9)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        p = out_dir / f"fig_bank_vs_inf.{ext}"
        fig.savefig(p); written.append(p)
    plt.close(fig)
    return written


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Validate the ray-traced channel bank against TR 38.901 InF.")
    parser.add_argument("--bank", default="data/rt_channel_bank.npz")
    parser.add_argument("--out", default="results/phase2/bank_validation")
    args = parser.parse_args(argv)
    for p in write_validation(validate_bank(Path(args.bank)), Path(args.out)):
        print(f"Wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
