"""Combined and summary thesis figures from stage_results_v2.json (multi-panel, Pareto, inventory)."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from factory6g.visualization.thesis_plot_style import (
    method_label,
    EBNO_XLABEL,
    FIG_CALLOUT_PT,
    FIG_TITLE_PT,
    THESIS_DPI,
    THESIS_FIGSIZE,
    THESIS_FIGSIZE_LEGEND,
    apply_thesis_rcparams,
    method_color,
    method_marker,
    order_method_names,
    order_methods_dict,
    style_ebno_axis,
)
from factory6g.visualization.thesis_summary_figures import (
    CANONICAL_SCHEDULERS,
    CHANNEL_LABELS,
    RM_LABELS,
    _ebno_index,
    available_channels,
    _effective_ber,
    load_stage_json,
    plot_ch04_ber_heatmap,
)

SUNWAY_GREY = "#64748b"
POINT_STATUS_UPPER_BOUND_ONLY = "upper_bound_only"


def _coerce_float_array(values: list[Any]) -> np.ndarray:
    return np.asarray([float(v) for v in values], dtype=float)


def _point_status_array(metric_map: dict[str, list[Any]], *, fallback_len: int) -> np.ndarray:
    statuses = metric_map.get("point_status", [])
    if isinstance(statuses, list) and len(statuses) == fallback_len:
        return np.asarray([str(value) for value in statuses], dtype=object)
    return np.asarray(["resolved"] * fallback_len, dtype=object)


UPPER_BOUND_NOTE = "Open markers, dashed: 95% upper bound on BER (fewer than 30 bit errors)"


def _status_handles() -> tuple[list[Any], list[str]]:
    """Legend keys for the point-status rule, so no free-floating note is needed."""
    from matplotlib.lines import Line2D

    from factory6g.visualization.thesis_plot_style import MUTED

    resolved = Line2D([], [], color=MUTED, marker="o", linestyle="-", linewidth=1.4)
    bound = Line2D([], [], color=MUTED, marker="o", markerfacecolor="none", linestyle="--", linewidth=1.2)
    return [resolved, bound], ["Resolved (≥ 30 errors)", "95% upper bound"]


def _wrap_ylabel(text: str, max_chars: int = 26) -> str:
    """Two-line y-label when a long label would outgrow a short panel axis."""
    import textwrap

    return "\n".join(textwrap.wrap(text, max_chars)) if len(text) > max_chars else text


def _plain_text(text: str | None) -> str | None:
    """Replace LaTeX non-breaking tildes, which matplotlib prints literally."""
    return None if text is None else text.replace("~", "\u00a0")


def _span_one_decade(ax: Any) -> None:
    """Widen a log BER axis to whole decades so that major ticks are labelled."""
    lo, hi = ax.get_ylim()
    if lo <= 0 or hi <= 0:
        return
    ax.set_ylim(10 ** math.floor(math.log10(lo)), 10 ** math.ceil(math.log10(hi)))


def _place_legend(ax, n_entries: int, *, status_keys: bool = False) -> None:
    """Put the legend beside the axes rather than on top of the curves.

    With eight schedulers the default in-axes legend covers the right half of a
    BER sweep, including the x-tick labels. Placing it outside keeps every
    marker and both axes readable at print size.
    """
    handles, labels = ax.get_legend_handles_labels()
    if status_keys:
        extra_h, extra_l = _status_handles()
        # A blank spacer row separates method keys from the marker-style keys.
        from matplotlib.lines import Line2D

        handles += [Line2D([], [], linestyle="none")] + extra_h
        labels += [""] + extra_l
    ax.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        frameon=False,
        ncol=1 if n_entries <= 9 else 2,
        fontsize=9,
    )


def _draw_ber_series(ax: Any, x: np.ndarray, metric_map: dict[str, list[Any]], name: str, stage_hint: str | None) -> None:
    """Draw one method's BER sweep following the point-status rule.

    Resolved points (at least 30 bit errors) are solid markers joined only to
    neighbouring resolved points; upper-bound-only points are open markers at
    their 95% upper confidence bound, joined by dashed segments. Lines never
    bridge a point of the other status.
    """
    ber = _coerce_float_array(metric_map["ber"])
    upper = _coerce_float_array(metric_map.get("ber_upper_confidence", metric_map["ber"]))
    statuses = metric_map.get("point_status", [])
    resolved = np.asarray([str(s) != POINT_STATUS_UPPER_BOUND_ONLY for s in statuses], dtype=bool) if statuses else ber > 0
    marker = method_marker(name, stage_hint=stage_hint)
    color = method_color(name, stage_hint=stage_hint)
    label = method_label(name)
    y_res = np.where(resolved & (ber > 0), ber, np.nan)
    y_ub = np.where(~resolved, upper, np.nan)
    if np.any(np.isfinite(y_res)):
        ax.semilogy(x, y_res, marker=marker, linewidth=1.8, color=color, label=label)
        label = "_nolegend_"
    if np.any(np.isfinite(y_ub)):
        ax.semilogy(x, y_ub, marker=marker, markerfacecolor="none", linestyle="--", linewidth=1.3, color=color, label=label)


def _plot_ber_publication(
    *,
    plt: Any,
    methods: dict[str, dict[str, list[Any]]],
    ebno_range: list[float],
    title: str,
    output_path: Path,
    stage_hint: str | None = None,
) -> None:
    apply_thesis_rcparams(plt)
    fig, ax = plt.subplots(1, 1, figsize=THESIS_FIGSIZE_LEGEND)
    x = np.asarray(ebno_range, dtype=float)
    ordered = order_methods_dict(methods, stage_hint=stage_hint)
    for name, metric_map in ordered.items():
        if "ber" not in metric_map:
            continue
        _draw_ber_series(ax, x, metric_map, name, stage_hint)
    style_ebno_axis(ax, ylabel="BER")  # caption is the title
    has_upper_bound = any(
        str(st) == POINT_STATUS_UPPER_BOUND_ONLY
        for m in ordered.values() for st in m.get("point_status", [])
    )
    _place_legend(ax, len(ordered), status_keys=has_upper_bound)
    fig.tight_layout()
    fig.savefig(output_path, dpi=THESIS_DPI)
    plt.close(fig)


def _plot_metric_vs_ebno(
    *,
    plt: Any,
    methods: dict[str, dict[str, list[Any]]],
    ebno_range: list[float],
    metric: str,
    ylabel: str,
    title: str,
    output_path: Path,
    stage_hint: str | None = None,
) -> None:
    apply_thesis_rcparams(plt)
    fig, ax = plt.subplots(1, 1, figsize=THESIS_FIGSIZE_LEGEND)
    ordered = order_methods_dict(methods, stage_hint=stage_hint)
    for name, metric_map in ordered.items():
        if metric not in metric_map:
            continue
        values = _coerce_float_array(metric_map[metric])
        ax.plot(
            ebno_range,
            values,
            marker=method_marker(name, stage_hint=stage_hint),
            color=method_color(name, stage_hint=stage_hint),
            label=method_label(name),
            linewidth=1.8,
        )
    style_ebno_axis(ax, ylabel=ylabel)  # caption is the title
    _place_legend(ax, len(ordered))
    fig.tight_layout()
    fig.savefig(output_path, dpi=THESIS_DPI)
    plt.close(fig)


def _filter_methods(
    methods: dict[str, dict[str, list[Any]]],
    methods_filter: Sequence[str] | None,
) -> dict[str, dict[str, list[Any]]]:
    if not methods_filter:
        return methods
    allowed = set(methods_filter)
    return {name: data for name, data in methods.items() if name in allowed}


def _plot_ber_on_axis(
    ax: Any,
    *,
    methods: dict[str, dict[str, list[Any]]],
    ebno_range: list[float],
    stage_hint: str | None,
    legend: bool = False,
) -> None:
    x = np.asarray(ebno_range, dtype=float)
    ordered = order_methods_dict(methods, stage_hint=stage_hint)
    for name, metric_map in ordered.items():
        if "ber" not in metric_map:
            continue
        _draw_ber_series(ax, x, metric_map, name, stage_hint)
    style_ebno_axis(ax, ylabel="BER")
    if legend:
        ax.legend(fontsize=FIG_CALLOUT_PT)


def _plot_metric_on_axis(
    ax: Any,
    *,
    methods: dict[str, dict[str, list[Any]]],
    ebno_range: list[float],
    metric: str,
    ylabel: str,
    stage_hint: str | None,
    legend: bool = False,
) -> None:
    ordered = order_methods_dict(methods, stage_hint=stage_hint)
    for name, metric_map in ordered.items():
        if metric not in metric_map:
            continue
        values = _coerce_float_array(metric_map[metric])
        ax.plot(
            ebno_range,
            values,
            marker=method_marker(name, stage_hint=stage_hint),
            color=method_color(name, stage_hint=stage_hint),
            label=method_label(name),
            linewidth=1.8,
        )
    style_ebno_axis(ax, ylabel=ylabel)
    if legend:
        ax.legend(fontsize=FIG_CALLOUT_PT, loc="best")


def plot_multi_panel_ber(
    *,
    panels: list[dict[str, str]],
    output_path: Path,
    supertitle: str | None = None,
    methods_filter: Sequence[str] | None = None,
    ncols: int | None = None,
) -> None:
    import matplotlib.pyplot as plt

    apply_thesis_rcparams(plt)
    n = len(panels)
    ncols = min(n, ncols or n)
    nrows = math.ceil(n / ncols)
    # A shared BER axis lets the reader compare levels across panels directly.
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(2.9 * ncols, 3.9 * nrows), squeeze=False, sharey=True, layout="constrained"
    )
    axes_flat = axes.flatten()
    stage_hint = "estimators"
    has_upper_bound = False

    for i, (ax, panel) in enumerate(zip(axes_flat, panels)):
        payload = load_stage_json(Path(panel["source"]))
        methods = _filter_methods(payload["methods"], methods_filter)
        stage_hint = payload.get("stage", stage_hint)
        has_upper_bound |= any(
            str(st) == POINT_STATUS_UPPER_BOUND_ONLY
            for m in methods.values() for st in m.get("point_status", [])
        )
        _plot_ber_on_axis(
            ax,
            methods=methods,
            ebno_range=payload["ebno_db_range"],
            stage_hint=stage_hint,
        )
        ax.set_title(_plain_text(panel["title"]), fontsize=FIG_TITLE_PT)
        if i % ncols > 0:
            ax.set_ylabel("")
        if i < n - ncols:
            ax.set_xlabel("")
    _span_one_decade(axes_flat[0])

    handles, labels = axes_flat[0].get_legend_handles_labels()
    if has_upper_bound:
        extra_h, extra_l = _status_handles()
        handles, labels = handles + extra_h, labels + extra_l
    # The LaTeX caption names the campaign, so ``supertitle`` is not drawn.
    if len(labels) > 1:
        fig.legend(handles, labels, loc="outside upper center", ncol=min(len(labels), 4),
                   frameon=False, fontsize=FIG_CALLOUT_PT)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=THESIS_DPI, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"Wrote {output_path}")


def plot_multi_panel_metric(
    *,
    panels: list[dict[str, str]],
    metric: str,
    ylabel: str,
    output_path: Path,
    supertitle: str | None = None,
    methods_filter: Sequence[str] | None = None,
    ncols: int | None = None,
) -> None:
    import matplotlib.pyplot as plt

    apply_thesis_rcparams(plt)
    n = len(panels)
    ncols = min(n, ncols or n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(2.9 * ncols, 3.9 * nrows), squeeze=False, layout="constrained"
    )
    axes_flat = axes.flatten()
    stage_hint = None
    if panels:
        first = load_stage_json(Path(panels[0]["source"]))
        stage_hint = first.get("stage")

    for i, (ax, panel) in enumerate(zip(axes_flat, panels)):
        payload = load_stage_json(Path(panel["source"]))
        methods = _filter_methods(payload["methods"], methods_filter)
        stage_hint = payload.get("stage")
        _plot_metric_on_axis(
            ax,
            methods=methods,
            ebno_range=payload["ebno_db_range"],
            metric=metric,
            ylabel=_wrap_ylabel(ylabel),
            stage_hint=stage_hint,
        )
        ax.set_title(_plain_text(panel["title"]), fontsize=FIG_TITLE_PT)
        if i % ncols > 0:
            ax.set_ylabel("")
        if i < n - ncols:
            ax.set_xlabel("")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    if len(labels) > 1:
        fig.legend(handles, labels, loc="outside upper center", ncol=len(labels) if len(labels) <= 5 else 4,
                   frameon=False, fontsize=FIG_CALLOUT_PT)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=THESIS_DPI, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"Wrote {output_path}")


def plot_runtime_bar(
    *,
    payload: dict[str, Any],
    output_path: Path,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    apply_thesis_rcparams(plt)
    # Methods close operating points after different numbers of batches, so
    # total runtime is not comparable; plot the mean runtime per batch.
    stage_hint = payload.get("stage")
    methods = payload["methods"]
    names = order_method_names(list(methods.keys()), stage_hint=stage_hint)
    values = [
        sum(float(v) for v in methods[n]["runtime_sec"]) / sum(float(v) for v in methods[n]["num_batches"])
        for n in names
    ]
    colors = [method_color(n, stage_hint=stage_hint) for n in names]

    fig, ax = plt.subplots(figsize=THESIS_FIGSIZE)
    bars = ax.bar([method_label(n) for n in names], values, color=colors, width=0.55)
    for bar, value in zip(bars, values):
        ax.annotate(f"{value:.2f} s", (bar.get_x() + bar.get_width() / 2, value),
                    ha="center", va="bottom", xytext=(0, 3), textcoords="offset points")
    ax.set_ylabel("Runtime per batch (s)")
    ax.set_ylim(0, max(values) * 1.18)
    ax.grid(True, axis="y")
    ax.grid(False, axis="x")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=THESIS_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output_path}")


def plot_rm_pareto(
    *,
    run_b_dir: Path,
    ebno_db: float,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    apply_thesis_rcparams(plt)
    channels = available_channels(run_b_dir)
    channel_markers = {"rayleigh": "o", "rician": "s", "tr38901": "D", "raytraced": "^"}

    fig, ax = plt.subplots(figsize=THESIS_FIGSIZE_LEGEND)
    n_sched = len(CANONICAL_SCHEDULERS)
    for c, channel in enumerate(channels):
        payload = load_stage_json(run_b_dir / channel / "resource_managers/stage_results_v2.json")
        idx = _ebno_index(payload, ebno_db)
        for s_idx, scheduler in enumerate(CANONICAL_SCHEDULERS):
            metric_map = payload["methods"][scheduler]
            ber, status = _effective_ber(metric_map, idx)
            throughput = float(metric_map["throughput_bits_per_batch"][idx]) / 1e3
            # Throughput takes only two tiers; dodge markers horizontally so
            # coincident points stay visible.
            dodge = (s_idx - (n_sched - 1) / 2) * 1.1 + (c - (len(channels) - 1) / 2) * 0.3
            color = method_color(scheduler, stage_hint="resource_managers")
            bound = status == POINT_STATUS_UPPER_BOUND_ONLY
            ax.scatter(
                throughput + dodge,
                ber,
                facecolors="none" if bound else color,
                edgecolors=color if bound else "black",
                marker=channel_markers[channel],
                s=55,
                linewidths=1.2 if bound else 0.4,
                zorder=3,
            )

    ax.set_yscale("log")
    _span_one_decade(ax)
    ax.set_xlim(50, 135)
    ax.set_xlabel(r"Throughput ($10^{3}$ bits/batch)")
    ax.set_ylabel("BER")
    ax.grid(True, which="major")

    channel_handles = [
        Line2D(
            [0],
            [0],
            marker=channel_markers[ch],
            color="0.2",
            linestyle="None",
            markersize=7,
            label=CHANNEL_LABELS[ch],
        )
        for ch in channels
    ]
    scheduler_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color=method_color(sched, stage_hint="resource_managers"),
            linestyle="None",
            markersize=7,
            markeredgecolor="black",
            markeredgewidth=0.4,
            label=RM_LABELS[sched],
        )
        for sched in CANONICAL_SCHEDULERS
    ]
    # Compact single legend below axes (column-friendly).
    spacer = Line2D([0], [0], linestyle="None", label=" ")
    bound_handle = Line2D(
        [0], [0], marker="o", color="0.2", markerfacecolor="none", linestyle="None",
        markersize=7, label="Open: 95% upper bound",
    )
    all_handles = channel_handles + [spacer] + scheduler_handles + [spacer, bound_handle]
    ax.legend(
        handles=all_handles,
        fontsize=9,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        ncol=1,
        frameon=False,
        borderaxespad=0.0,
        handletextpad=0.35,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=THESIS_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output_path}")


def export_single_json_sweeps(
    *,
    json_path: Path,
    output_prefix: str,
    out_dir: Path,
) -> None:
    import matplotlib.pyplot as plt

    payload = load_stage_json(json_path)
    stage = payload["stage"]
    ebno_range = payload["ebno_db_range"]
    methods = payload["methods"]

    _plot_ber_publication(
        plt=plt,
        methods=methods,
        ebno_range=ebno_range,
        title="",
        output_path=out_dir / f"{output_prefix}_ber_vs_ebno.png",
        stage_hint=stage,
    )
    _plot_metric_vs_ebno(
        plt=plt,
        methods=methods,
        ebno_range=ebno_range,
        metric="throughput_bits_per_batch",
        ylabel="Throughput (bits/batch)",
        title="",
        output_path=out_dir / f"{output_prefix}_throughput_vs_ebno.png",
        stage_hint=stage,
    )


def export_single_json_ber(
    *,
    json_path: Path,
    output_path: Path,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    payload = load_stage_json(json_path)
    _plot_ber_publication(
        plt=plt,
        methods=payload["methods"],
        ebno_range=payload["ebno_db_range"],
        title=title,
        output_path=output_path,
        stage_hint=payload.get("stage"),
    )


def _format_ber_latex(ber: float, *, upper_bound: bool = False) -> str:
    if ber <= 0:
        body = "0"
    else:
        exponent = int(math.floor(math.log10(ber)))
        mantissa = ber / (10**exponent)
        body = f"{mantissa:.2f}\\times10^{{{exponent}}}"
    if upper_bound:
        return rf"$\leq {body}$"
    return f"${body}$"


def write_exploratory_anchor_table(
    *,
    rows: list[dict[str, str]],
    ebno_db: float,
    output_path: Path,
    note: str,
) -> None:
    lines = [
        "% Auto-generated from thesis/figures/manifest.json — do not edit by hand.",
        "\\begin{table}[htbp]",
        "  \\centering",
        "  \\caption{BER of exploratory single-method channel-estimation campaigns at "
        f"$E_b/N_0 = {ebno_db:.0f}$\\,dB on 3GPP TR~38.901 UMi (QPSK, $U = 4$, $B = 64$). "
        "Each row is a separate campaign, so the rows do not share channel draws (P1).}",
        "  \\label{tab:ch04-ce-exploratory}",
        "  \\small",
        "  \\begin{tabular}{lcccc}",
        "    \\toprule",
        "    \\tablehead \\textbf{Estimator} & \\textbf{BER} & \\textbf{Status} & \\textbf{Block errors} & \\textbf{Run ID} \\\\",
        "    \\midrule",
    ]
    for row in rows:
        payload = load_stage_json(Path(row["source"]))
        method = row["method"]
        metric_map = payload["methods"][method]
        idx = _ebno_index(payload, ebno_db)
        ber, status = _effective_ber(metric_map, idx)
        status_tex = "Resolved" if status != POINT_STATUS_UPPER_BOUND_ONLY else "Upper-bound only"
        ber_tex = _format_ber_latex(ber, upper_bound=(status == POINT_STATUS_UPPER_BOUND_ONLY))
        run_id = str(payload.get("run_id", Path(row["source"]).parts[1][:15])).replace("_", "\\_")
        blocks = f"{int(metric_map['block_errors'][idx]):,}".replace(",", "{,}")
        lines.append(f"    {row['label']} & {ber_tex} & {status_tex} & {blocks} & \\texttt{{{run_id}}} \\\\")
    lines.extend(
        [
            "    \\bottomrule",
            "  \\end{tabular}",
            "",
            "  \\vspace{0.35em}",
            "  \\footnotesize",
            f"  {note}",
            "\\end{table}",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {output_path}")


COUPLED_ESTIMATOR_LABELS: dict[str, str] = {
    "ls": "LS",
    "adaptive": "Adaptive",
    "neural": "Neural",
    "perfect": "Genie",
}


def plot_coupled_ber_heatmap(
    *,
    panels: list[dict[str, str]],
    ebno_db: float,
    output_path: Path,
    estimators: Sequence[str] = ("ls", "adaptive", "neural", "perfect"),
    schedulers: Sequence[str] = ("static", "max_throughput", "pf", "wmmse", "drl", "reliability_drl"),
    ncols: int = 2,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    """Run D: BER at one operating point, scheduler (rows) x estimator (columns), one panel per channel.

    The colour scale spans the data of the figure (floored at 1e-6) unless
    ``vmin``/``vmax`` are given, so that panels at one operating point, where
    all BERs lie within one or two decades, remain distinguishable.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    apply_thesis_rcparams(plt)
    n = len(panels)
    ncols = min(n, ncols)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.6 * ncols, 3.2 * nrows), squeeze=False, layout="constrained"
    )
    matrices: list[tuple[np.ndarray, np.ndarray]] = []
    for panel in panels:
        payload = load_stage_json(Path(panel["source"]))
        idx = _ebno_index(payload, ebno_db)
        matrix = np.full((len(schedulers), len(estimators)), np.nan)
        upper = np.zeros_like(matrix, dtype=bool)
        for r, sch in enumerate(schedulers):
            for c, est in enumerate(estimators):
                metric_map = payload["methods"][f"{est}+{sch}"]
                ber, status = _effective_ber(metric_map, idx)
                matrix[r, c] = max(ber, 1e-6)
                upper[r, c] = status == POINT_STATUS_UPPER_BOUND_ONLY
        matrices.append((matrix, upper))
    all_values = np.concatenate([m.ravel() for m, _ in matrices])
    lo = float(np.nanmin(all_values)) if vmin is None else vmin
    hi = float(np.nanmax(all_values)) if vmax is None else vmax
    if hi / lo < 10.0:  # keep at least one decade so the colour bar reads as a log scale
        lo = hi / 10.0
    norm = LogNorm(vmin=lo, vmax=hi)
    threshold = math.sqrt(lo * hi)
    cell_pt, tick_pt = 7.5, 8.5
    image = None
    for i, (ax, panel, (matrix, upper)) in enumerate(zip(axes.flatten(), panels, matrices)):
        image = ax.imshow(np.clip(matrix, lo, hi), cmap="Blues", norm=norm, aspect="auto")
        ax.set_title(_plain_text(panel["title"]), fontsize=FIG_TITLE_PT)
        ax.set_xticks(range(len(estimators)))
        ax.set_xticklabels(
            [COUPLED_ESTIMATOR_LABELS.get(e, e) for e in estimators] if i >= n - ncols else [""] * len(estimators),
            fontsize=tick_pt,
        )
        ax.set_yticks(range(len(schedulers)))
        ax.set_yticklabels(
            [RM_LABELS.get(sch, sch).replace("Reliability-aware DRL", "Rel.-aware DRL") for sch in schedulers]
            if i % ncols == 0 else [""] * len(schedulers),
            fontsize=tick_pt,
        )
        ax.grid(False)
        for r in range(len(schedulers)):
            for c in range(len(estimators)):
                value = matrix[r, c]
                exponent = int(np.floor(np.log10(value)))
                mantissa = round(value / 10.0**exponent, 1)
                if mantissa >= 10.0:
                    mantissa, exponent = 1.0, exponent + 1
                core = rf"{mantissa:.1f}{{\times}}10^{{{exponent}}}"
                text = rf"$\leq{core}$" if upper[r, c] else rf"${core}$"
                ax.text(c, r, text, ha="center", va="center", fontsize=cell_pt,
                        color="white" if value > threshold else "black")
    if image is not None:
        cbar = fig.colorbar(image, ax=axes.flatten().tolist(), fraction=0.04, pad=0.02)
        cbar.set_label("BER")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=THESIS_DPI, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"Wrote {output_path}")


def _metric_at(metric_map: dict[str, Any], key: str, idx: int) -> float:
    values = metric_map.get(key) or []
    return float(values[idx]) if idx < len(values) else math.nan


def write_ce_matrix_table(
    *,
    panels: list[dict[str, str]],
    ber_ebno_db: Sequence[float],
    nmse_ebno_db: Sequence[float],
    calibration_ebno_db: float,
    output_path: Path,
    caption: str,
    note: str,
) -> None:
    """Run C summary: BER, NMSE and variance calibration per channel and estimator.

    Channels are group rows rather than a column, which keeps the table within
    the text width. Runtime is left out on purpose: campaign runtimes were logged
    on a shared, loaded host, and the cost axis comes from a quiet timing pass.
    """
    n_ber, n_nmse = len(ber_ebno_db), len(nmse_ebno_db)
    n_cols = 1 + n_ber + n_nmse + 1
    ber_first, nmse_first = 2, 2 + n_ber
    lines = [
        "% Auto-generated from thesis/figures/manifest.json — do not edit by hand.",
        "\\begin{table}[htbp]",
        "  \\centering",
        f"  \\caption{{{caption}}}",
        "  \\label{tab:ch04-ce-matrix}",
        "  \\small",
        "  \\setlength{\\tabcolsep}{4pt}",
        "  \\begin{tabular}{l" + "r" * (n_cols - 1) + "}",
        "    \\toprule",
        "    \\tablehead \\textbf{Estimator}"
        + f" & \\multicolumn{{{n_ber}}}{{c}}{{\\textbf{{BER}}}}"
        + f" & \\multicolumn{{{n_nmse}}}{{c}}{{\\textbf{{NMSE (dB)}}}}"
        + " & $\\kappa_{\\mathrm{e}}$ \\\\",
        f"    \\cmidrule(lr){{{ber_first}-{nmse_first - 1}}}"
        f"\\cmidrule(lr){{{nmse_first}-{n_cols - 1}}}\\cmidrule(lr){{{n_cols}-{n_cols}}}",
        "    \\tablehead"
        + "".join(f" & {e:.0f}\\,dB" for e in ber_ebno_db)
        + "".join(f" & {e:.0f}\\,dB" for e in nmse_ebno_db)
        + f" & {calibration_ebno_db:.0f}\\,dB \\\\",
        "    \\midrule",
    ]
    for p_idx, panel in enumerate(panels):
        payload = load_stage_json(Path(panel["source"]))
        methods = order_methods_dict(payload["methods"], stage_hint="estimators")
        if p_idx > 0:
            lines.append("    \\midrule")
        lines.append(f"    \\tablegroup{{{n_cols}}}{{{panel['title']}}} \\\\")
        for method, metric_map in methods.items():
            cells = [method_label(method)]
            for e in ber_ebno_db:
                ber, status = _effective_ber(metric_map, _ebno_index(payload, e))
                cells.append(_format_ber_latex(ber, upper_bound=(status == POINT_STATUS_UPPER_BOUND_ONLY)))
            for e in nmse_ebno_db:
                val = _metric_at(metric_map, "nmse_db", _ebno_index(payload, e))
                cells.append("---" if math.isnan(val) else f"${val:.1f}$")
            kappa = _metric_at(metric_map, "err_var_calibration", _ebno_index(payload, calibration_ebno_db))
            cells.append("---" if math.isnan(kappa) else f"${kappa:.2f}$")
            lines.append("    " + " & ".join(cells) + " \\\\")
    lines.extend(["    \\bottomrule", "  \\end{tabular}", ""])
    if note:
        lines.append(f"  \\tablenote{{{note}}}")
    lines.append("\\end{table}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {output_path}")


def scan_results_inventory(results_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_dir in sorted(results_root.iterdir()):
        if not run_dir.is_dir() or not run_dir.name[0].isdigit():
            continue
        jsons = list(run_dir.rglob("stage_results_v2.json"))
        if not jsons:
            rows.append(
                {
                    "run_dir": run_dir.name,
                    "status": "incomplete",
                    "stage_jsons": 0,
                    "detail": "no stage_results_v2.json",
                }
            )
            continue
        for jpath in sorted(jsons):
            rel = str(jpath.relative_to(run_dir))
            try:
                payload = load_stage_json(jpath)
            except json.JSONDecodeError:
                rows.append({"run_dir": run_dir.name, "status": "bad_json", "json": rel})
                continue
            methods = list(payload.get("methods", {}).keys())
            rows.append(
                {
                    "run_dir": run_dir.name,
                    "status": "complete",
                    "json": rel,
                    "stage": payload.get("stage"),
                    "run_id": payload.get("run_id"),
                    "methods": ", ".join(methods),
                    "n_ebno": len(payload.get("ebno_db_range", [])),
                }
            )
    return rows


def _tex_escape(text: str) -> str:
    return text.replace("_", "\\_").replace("%", "\\%").replace("&", "\\&")


def _tex_path(text: str) -> str:
    """Raw path text for \\tabpath / \\seqsplit (underscores stay literal)."""
    return text.replace("%", "\\%").replace("&", "\\&").replace("#", "\\#")


def write_run_inventory_tables(
    *,
    inventory: list[dict[str, Any]],
    output_path: Path,
    cleanup: dict[str, Any],
    roles: dict[str, str] | None = None,
) -> None:
    """Write the Appendix B run inventory.

    The inventory is a ``longtable`` because it spans several pages; a floating
    ``table`` of this height is "too large for page" and spills off the bottom.
    ``roles`` maps a run directory (``results/<name>`` or bare ``<name>``) to a
    short role such as ``Run A (Rayleigh)`` or ``superseded``.
    """
    role_by_name: dict[str, str] = {}
    for key, value in (roles or {}).items():
        role_by_name[Path(key).name] = value
    ncols = 5
    lines = [
        "% Auto-generated from results/ scan — do not edit by hand.",
        "\\section{Local run directory inventory}",
        "\\label{app:run-inventory}",
        "",
        "Table~\\ref{tab:app-run-inventory} lists every timestamped directory under",
        "\\texttt{results/} at export time, one row per stage-result file. The",
        "\\emph{Role} column names the locked campaign a directory belongs to",
        "(Table~\\ref{tab:app-runs}); \\emph{superseded} marks earlier campaigns kept",
        "for traceability but not used in Chapter~\\ref{ch:results}, and",
        "\\emph{archive} marks the exploratory campaigns that predate the locked",
        "system model. Incomplete folders were removed after successful figure",
        "regeneration (Table~\\ref{tab:app-run-deletions}).",
        "",
        "\\begingroup",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\arrayrulecolor{tblrule}",
        "\\renewcommand{\\arraystretch}{1.3}",
        "\\rowcolors{1}{tblstripe}{}",
        "\\begin{longtable}{",
        "    >{\\raggedright\\arraybackslash}p{4.7cm}",
        "    >{\\raggedright\\arraybackslash}p{1.45cm}",
        "    >{\\raggedright\\arraybackslash}p{2.1cm}",
        "    >{\\raggedright\\arraybackslash}p{3.1cm}",
        "    >{\\raggedright\\arraybackslash}p{1.95cm}}",
        "  \\caption{Simulation run inventory (complete stage JSON artefacts).}",
        "  \\label{tab:app-run-inventory}\\\\",
        "  \\toprule",
        "  \\tablehead \\textbf{Run directory} & \\textbf{Stage} & \\textbf{Methods}"
        " & \\textbf{JSON path} & \\textbf{Role} \\\\",
        "  \\midrule",
        "\\endfirsthead",
        f"  \\multicolumn{{{ncols}}}{{l}}{{\\textit{{Table~\\thetable\\ (continued)}}}}\\\\",
        "  \\toprule",
        "  \\tablehead \\textbf{Run directory} & \\textbf{Stage} & \\textbf{Methods}"
        " & \\textbf{JSON path} & \\textbf{Role} \\\\",
        "  \\midrule",
        "\\endhead",
        "  \\bottomrule",
        f"  \\multicolumn{{{ncols}}}{{r}}{{\\textit{{continued on next page}}}}\\\\",
        "\\endfoot",
        "  \\bottomrule",
        "\\endlastfoot",
    ]
    for row in inventory:
        if row.get("status") != "complete":
            continue
        run_tex = _tex_path(row["run_dir"])
        methods_raw = row.get("methods", "")
        if len(methods_raw) > 42:
            methods_raw = methods_raw[:39] + "..."
        methods = _tex_path(methods_raw)
        json_path = _tex_path(row.get("json", ""))
        stage_tex = _tex_path(str(row.get("stage", "")))
        role = _tex_escape(role_by_name.get(row["run_dir"], "archive"))
        lines.append(
            f"    \\tabpath{{{run_tex}}} & \\tabpath{{{stage_tex}}} & "
            f"\\tabpath{{{methods}}} & \\tabpath{{{json_path}}} & {role} \\\\"
        )
    lines.extend(
        [
            "\\end{longtable}",
            "\\endgroup",
            "",
            "\\begin{table}[htbp]",
            "  \\centering",
            "  \\caption{Run directories deleted after export (incomplete or superseded).}",
            "  \\label{tab:app-run-deletions}",
            "  \\footnotesize",
            "  \\setlength{\\tabcolsep}{4pt}",
            "  \\begin{tabularx}{\\linewidth}{",
            "    >{\\raggedright\\arraybackslash}X",
            "    >{\\raggedright\\arraybackslash}X}",
            "    \\toprule",
            "    \\tablehead \\textbf{Directory} & \\textbf{Reason} \\\\",
            "    \\midrule",
        ]
    )
    for path in cleanup.get("incomplete", []):
        name = _tex_path(Path(path).name)
        lines.append(f"    \\tabpath{{{name}}} & Incomplete (no stage JSON) \\\\")
    for item in cleanup.get("redundant", []):
        name = _tex_path(Path(item["path"]).name)
        reason = _tex_escape(item["reason"].replace("Run~B", "Run\\textasciitilde{}B"))
        lines.append(f"    \\tabpath{{{name}}} & {reason} \\\\")
    lines.extend(["    \\bottomrule", "  \\end{tabularx}", "\\end{table}", ""])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {output_path}")


__all__ = [
    "export_single_json_ber",
    "export_single_json_sweeps",
    "plot_ch04_ber_heatmap",
    "plot_multi_panel_ber",
    "plot_multi_panel_metric",
    "plot_rm_pareto",
    "plot_runtime_bar",
    "scan_results_inventory",
    "write_exploratory_anchor_table",
    "write_run_inventory_tables",
]
