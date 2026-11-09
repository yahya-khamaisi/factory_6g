"""Thesis figure pack drawn from a traced run, in the thesis plot style.

Five figures, each saved as PNG and PDF, plus a CSV with the numbers behind
them:

1. ``fig_floor_radio_map``: hall plan with the ray-traced path-gain map and
   every terminal draw of the run, LOS filled and NLOS hollow.
2. ``fig_pathloss_vs_distance``: traced link gain of the channel-bank positions
   against 3D distance, with the TR 38.901 InF-SL curves and a fitted exponent.
3. ``fig_ber_vs_ebno_<stage>``: BER per method with batch-level 95 % intervals;
   points with fewer than 30 bit errors drawn as upper bounds.
4. ``fig_throughput_cdf``: CDF of per-terminal delivered bits per batch, with
   the 5th percentile marked.
5. ``fig_fairness_vs_reliability``: Jain index of delivered bits against BER.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from factory6g.sim.trace import read_trace
from factory6g.viewer.export import find_traces
from factory6g.viewer.stats import batch_series, delivered_per_terminal, jain_index, point_summary
from factory6g.visualization.thesis_plot_style import (
    INK,
    MUTED,
    THESIS_FIGSIZE,
    THESIS_FIGSIZE_LEGEND,
    apply_thesis_rcparams,
    method_color,
    method_marker,
    order_method_names,
)


def _inf_sl_gain_db(d: np.ndarray, los: bool, fc_ghz: float) -> np.ndarray:
    d = np.maximum(d, 1.0)
    pl_los = 31.84 + 21.50 * np.log10(d) + 19.00 * np.log10(fc_ghz)
    if los:
        return -pl_los
    return -np.maximum(pl_los, 33.00 + 25.50 * np.log10(d) + 20.00 * np.log10(fc_ghz))


def _save(fig: Any, out_dir: Path, name: str, written: list[Path]) -> None:
    for ext in ("png", "pdf"):
        path = out_dir / f"{name}.{ext}"
        fig.savefig(path)
        written.append(path)


def _floor(plt: Any, stage: dict[str, Any], hall: dict[str, Any], radio_map: dict[str, Any] | None, out_dir: Path, written: list[Path]) -> None:
    from matplotlib.patches import Polygon, Rectangle

    L, W, _ = hall["room_dimensions"]
    fig, ax = plt.subplots(figsize=(THESIS_FIGSIZE[0], THESIS_FIGSIZE[0] * 0.82))
    if radio_map:
        grid = np.array([[np.nan if v is None else v for v in row] for row in radio_map["gain_db"]], dtype=float)
        cs = radio_map["cell_size_m"]
        extent = [radio_map["x0"], radio_map["x0"] + grid.shape[1] * cs, radio_map["y0"], radio_map["y0"] + grid.shape[0] * cs]
        img = ax.imshow(grid, origin="lower", extent=extent, cmap="Blues", vmin=np.nanpercentile(grid, 5), vmax=np.nanpercentile(grid, 95), zorder=0)
        bar = fig.colorbar(img, ax=ax, fraction=0.046, pad=0.03)
        bar.set_label(f"Path gain at {radio_map['height_m']:g} m (dB)")
    ax.add_patch(Rectangle((-L / 2, -W / 2), L, W, fill=False, edgecolor=INK, linewidth=1.0, zorder=2))
    for i, m in enumerate(hall["machines"]):
        ax.add_patch(Rectangle((m["x"] - m["sx"] / 2, m["y"] - m["sy"] / 2), m["sx"], m["sy"], facecolor="#f1f2f4", edgecolor="#7d828b", linewidth=0.8, zorder=3))
        ax.text(m["x"], m["y"], f"M{i + 1}", ha="center", va="center", fontsize=9, color=MUTED, zorder=4)
    pos, los = [], []
    for ctx in stage["contexts"]:
        if "positions" in ctx:
            pos.extend(p[:2] for p in ctx["positions"])
            los.extend(ctx["los"])
    if pos:
        pos_arr, los_arr = np.array(pos), np.array(los, dtype=bool)
        ax.scatter(pos_arr[los_arr, 0], pos_arr[los_arr, 1], s=9, color=INK, linewidths=0, label="Terminal draw, LOS", zorder=5)
        if (~los_arr).any():
            ax.scatter(pos_arr[~los_arr, 0], pos_arr[~los_arr, 1], s=14, facecolors="white", edgecolors=INK, linewidths=0.8, label="Terminal draw, NLOS", zorder=5)
    bs = hall["bs_position"]
    ax.add_patch(Polygon([[bs[0], bs[1] + 0.45], [bs[0] + 0.4, bs[1] - 0.3], [bs[0] - 0.4, bs[1] - 0.3]], closed=True, facecolor=INK, edgecolor="white", zorder=6))
    ax.set_xlim(-L / 2 - 0.3, L / 2 + 0.3)
    ax.set_ylim(-W / 2 - 0.3, W / 2 + 0.3)
    ax.set_aspect("equal")
    ax.grid(False)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend(loc="upper left", fontsize=10, markerscale=1.6)
    _save(fig, out_dir, "fig_floor_radio_map", written)
    plt.close(fig)


def _pathloss(plt: Any, bank_path: Path, hall: dict[str, Any], fc_hz: float, out_dir: Path, written: list[Path], rows: list[dict[str, Any]]) -> None:
    with np.load(bank_path) as data:
        pos = data["eval_positions"].astype(np.float64)
        los = data["eval_los"].astype(bool)
        energy = np.mean(np.abs(data["eval_h"]) ** 2, axis=(1, 2))
    d = np.linalg.norm(pos - np.asarray(hall["bs_position"], dtype=np.float64), axis=1)
    gain_db = 10.0 * np.log10(energy)
    fc_ghz = fc_hz / 1e9
    fig, ax = plt.subplots(figsize=THESIS_FIGSIZE)
    ax.scatter(d[los], gain_db[los], s=4, color=INK, alpha=0.35, linewidths=0, label="Ray traced, LOS")
    if (~los).any():
        ax.scatter(d[~los], gain_db[~los], s=7, facecolors="none", edgecolors="#cc6677", linewidths=0.6, label="Ray traced, NLOS")
    grid = np.linspace(max(d.min(), 1.0), d.max(), 100)
    ax.plot(grid, _inf_sl_gain_db(grid, True, fc_ghz), color="#233369", label="InF-SL LOS")
    ax.plot(grid, _inf_sl_gain_db(grid, False, fc_ghz), color="#233369", linestyle="--", label="InF-SL NLOS")
    slope, intercept = np.polyfit(10.0 * np.log10(d[los]), gain_db[los], 1)
    ax.plot(grid, intercept + slope * 10.0 * np.log10(grid), color="#cc6677", label=f"LOS fit, n = {-slope:.2f}")
    ax.set_xscale("log")
    from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter

    ticks = [t for t in (2, 3, 4, 5, 6, 8, 10, 12, 15) if grid[0] * 0.95 <= t <= grid[-1] * 1.05]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("3D distance to the base station (m)")
    ax.set_ylabel("Link gain (dB)")
    ax.legend(loc="lower left", fontsize=10)
    _save(fig, out_dir, "fig_pathloss_vs_distance", written)
    plt.close(fig)
    rows.append({"figure": "pathloss_vs_distance", "method": "ray_traced_los", "ebno_db": "", "metric": "path_loss_exponent", "value": round(-slope, 4)})
    rows.append({"figure": "pathloss_vs_distance", "method": "ray_traced", "ebno_db": "", "metric": "los_fraction", "value": round(float(los.mean()), 4)})


def _ber(plt: Any, stage: dict[str, Any], out_dir: Path, written: list[Path], rows: list[dict[str, Any]]) -> None:
    meta, records, name = stage["header"]["meta"], stage["methods"], stage["header"]["stage"]
    ebnos = meta["ebno_db_range"]
    fig, ax = plt.subplots(figsize=THESIS_FIGSIZE_LEGEND)
    floor_values = []
    bounds = []
    for method in order_method_names(meta["methods"], stage_hint=name):
        color, marker = method_color(method, stage_hint=name), method_marker(method, stage_hint=name)
        xs, ys, lo, hi = [], [], [], []
        for eb in ebnos:
            p = point_summary(batch_series(records, method, eb))
            rows.append({"figure": f"ber_vs_ebno_{name}", "method": method, "ebno_db": eb, "metric": "ber", "value": p.ber})
            rows.append({"figure": f"ber_vs_ebno_{name}", "method": method, "ebno_db": eb, "metric": "ber_batch_ci95", "value": f"{p.ber_lo:.3e}..{p.ber_hi:.3e}"})
            if p.resolved:
                xs.append(eb); ys.append(p.ber); lo.append(p.ber_lo if np.isfinite(p.ber_lo) else p.ber); hi.append(p.ber_hi if np.isfinite(p.ber_hi) else p.ber)
                floor_values.append(p.ber)
            else:
                bound = p.ber if p.bit_errors else p.zero_error_bound
                bounds.append((eb, bound, color, marker))
                floor_values.append(bound)
        if xs:
            ax.plot(xs, ys, color=color, marker=marker, label=method)
            for x, y, l, h in zip(xs, ys, lo, hi):
                ax.plot([x, x], [max(l, 1e-12), h], color=color, linewidth=1.0)
    bottom = min(floor_values) / 4 if floor_values else 1e-6
    for eb, bound, color, marker in bounds:
        ax.plot([eb], [bound], marker=marker, color=color, markerfacecolor="white", linestyle="none")
        ax.annotate("", xy=(eb, max(bound / 2.5, bottom * 1.2)), xytext=(eb, bound), arrowprops={"arrowstyle": "->", "color": color, "lw": 1.0})
    ax.set_yscale("log")
    ax.set_ylim(bottom=bottom)
    ax.set_xlabel(r"$E_b/N_0$ (dB)")
    ax.set_ylabel("BER")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=11)
    ax.text(0.0, -0.3, "Bars: batch-level 95 %, cut at the axis when they reach zero. Hollow ↓: fewer than 30 bit errors, upper bound.", transform=ax.transAxes, fontsize=9, color=MUTED)
    _save(fig, out_dir, f"fig_ber_vs_ebno_{name}", written)
    plt.close(fig)


def _throughput_and_fairness(plt: Any, stage: dict[str, Any], ebno: float, out_dir: Path, written: list[Path], rows: list[dict[str, Any]]) -> None:
    meta, records, name = stage["header"]["meta"], stage["methods"], stage["header"]["stage"]
    methods = order_method_names(meta["methods"], stage_hint=name)
    fig, ax = plt.subplots(figsize=THESIS_FIGSIZE_LEGEND)
    points = []
    for method in methods:
        delivered = delivered_per_terminal(records, method, ebno)
        if delivered.size == 0:
            continue
        scheduled = np.array([r["active"] for r in records if r["method"] == method and float(r["ebno_db"]) == float(ebno)], dtype=bool)
        values = np.sort(delivered[scheduled]) / 1e3
        share = scheduled.mean()
        ax.step(values, np.arange(1, values.size + 1) / values.size, where="post", color=method_color(method, stage_hint=name), label=f"{method} ({share:.0%} scheduled)")
        p5, p50 = np.percentile(values, 5), np.percentile(values, 50)
        jain = jain_index(delivered.sum(axis=0))
        summary = point_summary(batch_series(records, method, ebno))
        points.append((method, summary.ber if summary.bit_errors else summary.zero_error_bound, jain, summary.resolved))
        for metric, value in (("p5_kbit", p5), ("median_kbit", p50), ("jain", jain)):
            rows.append({"figure": "throughput_cdf", "method": method, "ebno_db": ebno, "metric": metric, "value": round(float(value), 4)})
    ax.axhline(0.05, color=MUTED, linestyle=":", linewidth=0.9)
    ax.text(ax.get_xlim()[0], 0.065, " 5th percentile", ha="left", fontsize=9, color=MUTED)
    ax.set_xlabel(f"Delivered bits per scheduled terminal per batch (kbit), {ebno:g} dB")
    ax.set_ylabel("CDF")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=11)
    _save(fig, out_dir, "fig_throughput_cdf", written)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=THESIS_FIGSIZE)
    for k, (method, ber, jain, resolved) in enumerate(points):
        if np.isfinite(ber) and ber > 0:
            color = method_color(method, stage_hint=name)
            ax.scatter([ber], [jain], s=48, marker=method_marker(method, stage_hint=name), zorder=3,
                       color=color if resolved else "white", edgecolors=color)
            # Stagger labels: several methods often sit at a Jain index close to 1.
            ax.annotate(method if resolved else f"{method} (≤ bound)", (ber, jain), textcoords="offset points",
                        xytext=(6, -14 - 13 * k), fontsize=10, arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.5})
    ax.set_xscale("log")
    ax.invert_xaxis()
    jains = [j for *_, j, _ in points if np.isfinite(j)]
    if jains:
        ax.set_ylim(min(0.5, min(jains) - 0.05), 1.02)
    ax.ticklabel_format(axis="y", useOffset=False)
    ax.set_xlabel(f"BER at {ebno:g} dB (better →)")
    ax.set_ylabel("Jain index of delivered bits")
    _save(fig, out_dir, "fig_fairness_vs_reliability", written)
    plt.close(fig)


def make_figures(run_dir: Path, out_dir: Path | None = None, *, ebno_db: float | None = None, repo_root: Path | None = None) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    apply_thesis_rcparams(plt)
    run_dir = Path(run_dir)
    out_dir = Path(out_dir) if out_dir else run_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    stages = [read_trace(p) for p in find_traces(run_dir)]
    if not stages:
        raise FileNotFoundError(f"No trace under {run_dir}; run with simulation.trace on.")
    root = Path(repo_root) if repo_root else Path.cwd()
    meta = stages[0]["header"]["meta"]
    bank = root / meta.get("rt_channel_bank", "data/rt_channel_bank.npz")
    manifest = json.loads(bank.with_name(bank.name + ".manifest.json").read_text(encoding="utf-8"))
    hall = manifest["hall"]
    viewer_data = run_dir / "viewer" / "data.json"
    radio_map = json.loads(viewer_data.read_text(encoding="utf-8")).get("radio_map") if viewer_data.exists() else None

    written: list[Path] = []
    rows: list[dict[str, Any]] = []
    rm_stage = next((s for s in stages if s["header"]["stage"] in ("resource_managers", "coupled")), stages[0])
    _floor(plt, rm_stage, hall, radio_map, out_dir, written)
    if bank.exists():
        _pathloss(plt, bank, hall, float(hall["carrier_frequency"]), out_dir, written, rows)
    for stage in stages:
        _ber(plt, stage, out_dir, written, rows)
    ebnos = rm_stage["header"]["meta"]["ebno_db_range"]
    ebno = float(ebno_db) if ebno_db is not None else float(ebnos[len(ebnos) // 2])
    _throughput_and_fairness(plt, rm_stage, ebno, out_dir, written, rows)

    summary = out_dir / "figure_pack_summary.csv"
    with summary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["figure", "method", "ebno_db", "metric", "value"])
        writer.writeheader()
        writer.writerows(rows)
    written.append(summary)
    return written
