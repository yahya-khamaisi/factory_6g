from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from factory6g.sim.stages.common import (
    MIN_RESOLVED_BIT_ERRORS,
    POINT_STATUS_RESOLVED,
    POINT_STATUS_UPPER_BOUND_ONLY,
)
from factory6g.visualization.thesis_plot_style import (
    apply_thesis_rcparams,
    method_color,
    method_marker,
    order_methods_dict,
    style_ebno_axis,
    THESIS_FIGSIZE,
    THESIS_FIGSIZE_LEGEND,
    THESIS_DPI,
)


SCHEMA_VERSION = "2.0"


def write_stage_outputs(
    *,
    run_id: str,
    stage_name: str,
    stage_result: dict[str, Any],
    stage_dir: Path,
    config_snapshot: dict[str, Any],
    confidence_level: float,
    plot_results: bool,
) -> dict[str, str]:
    stage_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "stage": stage_name,
        "ebno_db_range": stage_result["ebno_db_range"],
        "confidence_level": confidence_level,
        "config_snapshot": config_snapshot,
        "methods": stage_result["methods"],
        "runtime_totals_sec": stage_result["runtime_totals_sec"],
    }

    json_path = stage_dir / "stage_results_v2.json"
    csv_path = stage_dir / "stage_results_v2.csv"
    _write_json(json_path, payload)
    _write_stage_csv(csv_path, payload)
    if plot_results:
        _write_stage_plots(stage_dir=stage_dir, payload=payload)
    return {
        "dir": str(stage_dir),
        "json": str(json_path),
        "csv": str(csv_path),
    }


def write_summary_outputs(
    *,
    run_id: str,
    run_dir: Path,
    stage_order: list[str],
    stage_paths: dict[str, dict[str, str]],
    stage_payloads: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    aggregate_means: dict[str, dict[str, dict[str, float]]] = {}
    runtime_totals_sec: dict[str, dict[str, float]] = {}
    for stage_name, payload in stage_payloads.items():
        stage_means: dict[str, dict[str, float]] = {}
        for method, metrics in payload["methods"].items():
            stage_means[method] = {
                metric: _safe_mean(values)
                for metric, values in metrics.items()
                if _is_numeric_series(values)
            }
        aggregate_means[stage_name] = stage_means
        runtime_totals_sec[stage_name] = {
            method: float(value)
            for method, value in payload.get("runtime_totals_sec", {}).items()
        }

    summary_payload = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "stage_order": stage_order,
        "stage_paths": stage_paths,
        "aggregate_means": aggregate_means,
        "runtime_totals_sec": runtime_totals_sec,
    }
    summary_json_path = run_dir / "summary_v2.json"
    summary_csv_path = run_dir / "summary_v2.csv"
    _write_json(summary_json_path, summary_payload)
    _write_summary_csv(summary_csv_path, summary_payload)
    return {
        "summary_payload": summary_payload,
        "summary_json": str(summary_json_path),
        "summary_csv": str(summary_csv_path),
    }


def _write_stage_csv(path: Path, payload: dict[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    ebno_values = payload["ebno_db_range"]
    stage = payload["stage"]
    methods = payload["methods"]
    for method_name, metric_map in methods.items():
        for metric_name, values in metric_map.items():
            if not isinstance(values, list):
                continue
            for index, value in enumerate(values):
                ebno = ebno_values[index] if index < len(ebno_values) else ""
                rows.append(
                    {
                        "stage": stage,
                        "method": method_name,
                        "metric": metric_name,
                        "ebno_db": ebno,
                        "value": value,
                    }
                )
        rows.append(
            {
                "stage": stage,
                "method": method_name,
                "metric": "runtime_total_sec",
                "ebno_db": "",
                "value": payload.get("runtime_totals_sec", {}).get(method_name, 0.0),
            }
        )

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["stage", "method", "metric", "ebno_db", "value"])
        writer.writeheader()
        writer.writerows(rows)


def _write_summary_csv(path: Path, payload: dict[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    for stage in payload["stage_order"]:
        stage_path = payload["stage_paths"][stage]
        rows.extend(
            [
                {"stage": stage, "method": "__stage__", "metric": "stage_dir", "value": stage_path["dir"]},
                {"stage": stage, "method": "__stage__", "metric": "stage_json", "value": stage_path["json"]},
                {"stage": stage, "method": "__stage__", "metric": "stage_csv", "value": stage_path["csv"]},
            ]
        )
        for method, metric_map in payload["aggregate_means"].get(stage, {}).items():
            for metric, value in metric_map.items():
                rows.append(
                    {
                        "stage": stage,
                        "method": method,
                        "metric": f"mean_{metric}",
                        "value": value,
                    }
                )
        for method, runtime in payload["runtime_totals_sec"].get(stage, {}).items():
            rows.append(
                {
                    "stage": stage,
                    "method": method,
                    "metric": "runtime_total_sec",
                    "value": runtime,
                }
            )

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["stage", "method", "metric", "value"])
        writer.writeheader()
        writer.writerows(rows)


def write_overview_plots(
    run_dir: Path,
    entries: list[tuple[str, str, dict[str, Any], list[float], dict[str, float]]],
    title_prefix: str = "Overview",
) -> None:
    """Generate overview plots combining all combos, grouped by stage.

    entries: list of (combo_label, stage_name, methods_dict, ebno_db_range, runtime_totals_sec)
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Group by stage_name
    from collections import defaultdict
    by_stage: dict[str, list[tuple[str, dict[str, Any], list[float], dict[str, float]]]] = defaultdict(list)
    for combo_label, stage_name, methods, ebno_range, runtime_totals in entries:
        by_stage[stage_name].append((combo_label, methods, ebno_range, runtime_totals))

    for stage_name, stage_entries in by_stage.items():
        stage_display = stage_name.replace("_", " ").title()
        overview_dir = run_dir / "overview" / stage_name
        overview_dir.mkdir(parents=True, exist_ok=True)

        # Build merged methods dict: key = "combo_label – method_name"
        merged_methods: dict[str, dict[str, Any]] = {}
        merged_runtime: dict[str, float] = {}
        for combo_label, methods, _ebno, runtime_totals in stage_entries:
            for method_name, metric_map in methods.items():
                label = f"{combo_label} – {method_name}" if combo_label else method_name
                merged_methods[label] = metric_map
                merged_runtime[label] = runtime_totals.get(method_name, 0.0)

        # Use the ebno_range from the first entry (all should be identical)
        ebno_range = stage_entries[0][2]
        title_base = f"All {title_prefix}"

        _plot_ber_publication(
            plt=plt,
            methods=merged_methods,
            ebno_range=ebno_range,
            title=f"{title_base}: BER vs Eb/No",
            output_path=overview_dir / "ber_vs_ebno.png",
            stage_hint=stage_name,
        )
        _plot_ber_raw(
            plt=plt,
            methods=merged_methods,
            ebno_range=ebno_range,
            title=f"{title_base}: Raw BER vs Eb/No",
            output_path=overview_dir / "ber_raw_vs_ebno.png",
            stage_hint=stage_name,
        )
        _plot_metric_vs_ebno(
            plt=plt,
            methods=merged_methods,
            ebno_range=ebno_range,
            metric="latency_model_ms",
            ylabel="Modelled latency (ms)",
            title=f"{title_base}: Latency vs Eb/No",
            output_path=overview_dir / "latency_vs_ebno.png",
            stage_hint=stage_name,
        )
        _plot_metric_vs_ebno(
            plt=plt,
            methods=merged_methods,
            ebno_range=ebno_range,
            metric="throughput_bits_per_batch",
            ylabel="Throughput (bits/batch)",
            title=f"{title_base}: Throughput vs Eb/No",
            output_path=overview_dir / "throughput_vs_ebno.png",
            stage_hint=stage_name,
        )
        _plot_metric_vs_ebno(
            plt=plt,
            methods=merged_methods,
            ebno_range=ebno_range,
            metric="tx_power_w",
            ylabel="Allocated transmit power (W)",
            title=f"{title_base}: Power vs Eb/No",
            output_path=overview_dir / "power_vs_ebno.png",
            stage_hint=stage_name,
        )
        _plot_metric_vs_ebno(
            plt=plt,
            methods=merged_methods,
            ebno_range=ebno_range,
            metric="jain_fairness",
            ylabel="Jain's fairness index",
            title=f"{title_base}: Fairness vs Eb/No",
            output_path=overview_dir / "fairness_vs_ebno.png",
            stage_hint=stage_name,
        )

        apply_thesis_rcparams(plt)
        names = list(merged_runtime.keys())
        values = [merged_runtime[n] for n in names]
        fig_rt, ax_rt = plt.subplots(1, 1, figsize=(max(9, len(names) * 0.8 + 2), 6))
        ax_rt.bar(names, values)
        ax_rt.set_ylabel("Runtime (sec)")
        ax_rt.set_title(f"{title_base}: Runtime by Method")
        ax_rt.grid(True, axis="y", alpha=0.3)
        plt.setp(ax_rt.get_xticklabels(), rotation=30, ha="right", fontsize=8)
        fig_rt.tight_layout()
        fig_rt.savefig(overview_dir / "runtime_by_method.png", dpi=THESIS_DPI)
        plt.close(fig_rt)

        print(f"[plot] Overview plots ({stage_name}): {overview_dir}")


def _write_stage_plots(*, stage_dir: Path, payload: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stage_key = payload["stage"]
    stage_name = stage_key.replace("_", " ").title()
    ebno_range = payload["ebno_db_range"]
    methods = payload["methods"]
    runtime_totals = payload.get("runtime_totals_sec", {})

    _plot_ber_publication(
        plt=plt,
        methods=methods,
        ebno_range=ebno_range,
        title=f"{stage_name}: BER vs Eb/No",
        output_path=stage_dir / "ber_vs_ebno.png",
        stage_hint=stage_key,
    )
    _plot_ber_raw(
        plt=plt,
        methods=methods,
        ebno_range=ebno_range,
        title=f"{stage_name}: Raw BER vs Eb/No",
        output_path=stage_dir / "ber_raw_vs_ebno.png",
        stage_hint=stage_key,
    )
    _plot_metric_vs_ebno(
        plt=plt,
        methods=methods,
        ebno_range=ebno_range,
        metric="latency_model_ms",
        ylabel="Modelled latency (ms)",
        title=f"{stage_name}: Latency vs Eb/No",
        output_path=stage_dir / "latency_vs_ebno.png",
        stage_hint=stage_key,
    )
    _plot_metric_vs_ebno(
        plt=plt,
        methods=methods,
        ebno_range=ebno_range,
        metric="throughput_bits_per_batch",
        ylabel="Throughput (bits/batch)",
        title=f"{stage_name}: Throughput vs Eb/No",
        output_path=stage_dir / "throughput_vs_ebno.png",
        stage_hint=stage_key,
    )
    _plot_metric_vs_ebno(
        plt=plt,
        methods=methods,
        ebno_range=ebno_range,
        metric="tx_power_w",
        ylabel="Allocated transmit power (W)",
        title=f"{stage_name}: Power vs Eb/No",
        output_path=stage_dir / "power_vs_ebno.png",
        stage_hint=stage_key,
    )
    _plot_metric_vs_ebno(
        plt=plt,
        methods=methods,
        ebno_range=ebno_range,
        metric="jain_fairness",
        ylabel="Jain's fairness index",
        title=f"{stage_name}: Fairness vs Eb/No",
        output_path=stage_dir / "fairness_vs_ebno.png",
        stage_hint=stage_key,
    )

    apply_thesis_rcparams(plt)
    fig_runtime, ax_runtime = plt.subplots(1, 1, figsize=THESIS_FIGSIZE)
    names = list(runtime_totals.keys())
    values = [runtime_totals[name] for name in names]
    ax_runtime.bar(names, values)
    ax_runtime.set_ylabel("Runtime (sec)")
    ax_runtime.set_title(f"{stage_name}: Runtime by Method")
    ax_runtime.grid(True, axis="y", alpha=0.3)
    fig_runtime.tight_layout()
    fig_runtime.savefig(stage_dir / "runtime_by_method.png", dpi=THESIS_DPI)
    plt.close(fig_runtime)


EXPERIMENTAL_METHODS = frozenset({"pso", "ista"})


def _plot_ber_publication(
    *,
    plt,
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
    # Experimental estimators stay out of the headline plot when canonical
    # methods are present; the raw plot still shows them.
    if any(name not in EXPERIMENTAL_METHODS for name in ordered):
        ordered = {k: v for k, v in ordered.items() if k not in EXPERIMENTAL_METHODS}
    for name, metric_map in ordered.items():
        if "ber" not in metric_map:
            continue

        ber = _coerce_float_array(metric_map["ber"])
        upper = _coerce_float_array(metric_map.get("ber_upper_confidence", metric_map["ber"]))
        statuses = _point_status_array(metric_map, fallback_len=ber.size)

        upper_mask = statuses == POINT_STATUS_UPPER_BOUND_ONLY
        valid_mask = ber > 0
        marker = method_marker(name, stage_hint=stage_hint)
        color = method_color(name, stage_hint=stage_hint)

        if np.any(valid_mask):
            ax.semilogy(
                x[valid_mask],
                ber[valid_mask],
                marker=marker,
                linewidth=1.8,
                color=color,
                label=name,
            )

        upper_show = upper_mask & ~valid_mask
        if np.any(upper_show):
            ax.semilogy(
                x[upper_show],
                upper[upper_show],
                marker=marker,
                markerfacecolor="none",
                linestyle="--",
                linewidth=1.5,
                color=color,
                label=name if not np.any(valid_mask) else "_nolegend_",
            )

    style_ebno_axis(ax, ylabel="BER", title=title)
    _place_legend(ax, len(ordered))
    # Place outside axes so the note does not collide with y-tick labels.
    fig.text(
        0.5,
        0.01,
        "Dashed/open markers: 95% BER upper bound (zero observed errors)",
        ha="center",
        fontsize=9,
        alpha=0.85,
    )
    fig.tight_layout(rect=(0.0, 0.04, 1.0, 1.0))
    fig.savefig(output_path, dpi=THESIS_DPI)
    plt.close(fig)


def _plot_ber_raw(
    *,
    plt,
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
        values = _coerce_float_array(metric_map["ber"])
        # A zero BER has no place on a log axis; leave a gap instead of clipping.
        values = np.where(values > 0, values, np.nan)
        ax.semilogy(
            x,
            values,
            marker=method_marker(name, stage_hint=stage_hint),
            color=method_color(name, stage_hint=stage_hint),
            label=name,
            linewidth=1.8,
        )

    style_ebno_axis(ax, ylabel="BER", title=title)
    _place_legend(ax, len(ordered))
    fig.tight_layout()
    fig.savefig(output_path, dpi=THESIS_DPI)
    plt.close(fig)


def _plot_metric_vs_ebno(
    *,
    plt,
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
            label=name,
            linewidth=1.8,
        )
    style_ebno_axis(ax, ylabel=ylabel, title=title)
    _place_legend(ax, len(ordered))
    fig.tight_layout()
    fig.savefig(output_path, dpi=THESIS_DPI)
    plt.close(fig)



def _point_status_array(metric_map: dict[str, list[Any]], *, fallback_len: int) -> np.ndarray:
    statuses = metric_map.get("point_status", [])
    if isinstance(statuses, list) and len(statuses) == fallback_len:
        return np.asarray([str(value) for value in statuses], dtype=object)
    bit_errors = metric_map.get("bit_errors", [])
    if isinstance(bit_errors, list) and len(bit_errors) == fallback_len:
        return np.asarray(
            [
                POINT_STATUS_UPPER_BOUND_ONLY
                if float(value) < float(MIN_RESOLVED_BIT_ERRORS)
                else POINT_STATUS_RESOLVED
                for value in bit_errors
            ],
            dtype=object,
        )
    return np.asarray([POINT_STATUS_RESOLVED] * fallback_len, dtype=object)


def _place_legend(ax, n_entries: int) -> None:
    """Put the legend beside the axes rather than on top of the curves.

    With eight schedulers the default in-axes legend covers the right half of a
    BER sweep, including the x-tick labels. Placing it outside keeps every
    marker and both axes readable at print size.
    """
    ncol = 1 if n_entries <= 9 else 2
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        frameon=False,
        ncol=ncol,
        fontsize=9,
    )


def _coerce_float_array(values: list[Any]) -> np.ndarray:
    return np.asarray(values, dtype=float)


def _is_numeric_series(values: Any) -> bool:
    if not isinstance(values, list) or not values:
        return False
    try:
        arr = np.asarray(values)
    except Exception:
        return False
    return bool(np.issubdtype(arr.dtype, np.number))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _safe_mean(values: list[Any]) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(np.nanmean(arr))
