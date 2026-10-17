"""Thesis summary plots and tables sourced from canonical stage_results_v2.json."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from factory6g.visualization.thesis_plot_style import (
    FIG_CALLOUT_PT,
    FIG_TITLE_PT,
    THESIS_DPI,
    apply_thesis_rcparams,
)

SUNWAY_GREY = "#64748b"
POINT_STATUS_UPPER_BOUND_ONLY = "upper_bound_only"

CANONICAL_SCHEDULERS: tuple[str, ...] = (
    "static",
    "round_robin",
    "max_throughput",
    "pf",
    "wmmse",
    "queue_aware",
    "drl",
    "reliability_drl",
)

RM_LABELS: dict[str, str] = {
    "static": "Static",
    "round_robin": "Round-robin",
    "max_throughput": "Max-throughput",
    "pf": "PF",
    "wmmse": "WMMSE",
    "queue_aware": "Queue-aware",
    "drl": "DRL",
    "reliability_drl": "Reliability-aware DRL",
}

CHANNEL_LABELS: dict[str, str] = {
    "rayleigh": "Rayleigh",
    "rician": "Rician",
    "tr38901": "TR\u00a038.901 UMi",
    "raytraced": "Ray-traced hall",
}

ALL_CHANNELS: tuple[str, ...] = ("rayleigh", "rician", "tr38901", "raytraced")


def available_channels(run_dir: Path, stage: str = "resource_managers") -> list[str]:
    """Channels of a multi-channel run that have a stage file, in canonical order."""
    return [ch for ch in ALL_CHANNELS if (run_dir / ch / stage / "stage_results_v2.json").exists()]


def load_stage_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _ebno_index(payload: dict[str, Any], ebno_db: float) -> int:
    return payload["ebno_db_range"].index(float(ebno_db))


def _effective_ber(metric_map: dict[str, list[Any]], index: int) -> tuple[float, str]:
    ber = float(metric_map["ber"][index])
    status = str(metric_map["point_status"][index])
    # An upper-bound-only point is reported at its 95% bound even when a few
    # errors were seen, so that a "<=" label never sits on a point estimate.
    if status != "upper_bound_only" and ber > 0:
        return ber, status
    upper = float(metric_map.get("ber_upper_confidence", [ber])[index])
    return upper, status


def plot_ch04_ber_heatmap(
    *,
    run_b_dir: Path,
    ebno_db: float,
    output_path: Path,
) -> None:
    """Scheduler × channel BER summary at one anchor operating point (Run~B)."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    apply_thesis_rcparams(plt)
    channels = available_channels(run_b_dir)
    schedulers = list(CANONICAL_SCHEDULERS)

    matrix = np.full((len(schedulers), len(channels)), np.nan)
    statuses = [["" for _ in channels] for _ in schedulers]

    for col, channel in enumerate(channels):
        payload = load_stage_json(run_b_dir / channel / "resource_managers/stage_results_v2.json")
        idx = _ebno_index(payload, ebno_db)
        for row, scheduler in enumerate(schedulers):
            metric_map = payload["methods"][scheduler]
            ber, status = _effective_ber(metric_map, idx)
            matrix[row, col] = max(ber, 1e-12)
            statuses[row][col] = status

    # Colour scale spans the data (at least one decade) so cells at one
    # operating point, which lie within a decade or two, stay distinguishable.
    lo, hi = float(np.nanmin(matrix)), float(np.nanmax(matrix))
    if hi / lo < 10.0:
        lo = hi / 10.0
    threshold = float(np.sqrt(lo * hi))

    fig, ax = plt.subplots(figsize=(3.0 + 1.45 * len(channels), 3.9))
    im = ax.imshow(matrix, cmap="Blues", norm=LogNorm(vmin=lo, vmax=hi), aspect="auto")

    ax.set_xticks(range(len(channels)))
    ax.set_xticklabels([CHANNEL_LABELS[ch] for ch in channels])
    ax.set_yticks(range(len(schedulers)))
    ax.set_yticklabels([RM_LABELS[s] for s in schedulers])
    ax.set_xlabel("Channel model (Run\u00a0B)")
    ax.set_ylabel("Resource manager")
    ax.grid(False)
    # No in-image title: the LaTeX caption states the operating point.

    for row in range(len(schedulers)):
        for col in range(len(channels)):
            value = matrix[row, col]
            is_upper = statuses[row][col] == POINT_STATUS_UPPER_BOUND_ONLY
            exponent = int(np.floor(np.log10(value)))
            mantissa = value / 10.0**exponent
            value_fmt = rf"{mantissa:.1f}{{\times}}10^{{{exponent}}}"
            if is_upper:
                text = rf"$\leq {value_fmt}$"
            else:
                text = rf"${value_fmt}$"
            ax.text(
                col,
                row,
                text,
                ha="center",
                va="center",
                fontsize=FIG_CALLOUT_PT - 2,
                color="black" if value < threshold else "white",
            )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("BER (colour scale)")
    # The caption explains the leq marking; no free-floating note in the image.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=THESIS_DPI, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"Wrote {output_path}")
