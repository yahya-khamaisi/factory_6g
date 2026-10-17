#!/usr/bin/env python3
"""Export all thesis figures and appendix tables from thesis/figures/manifest.json."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]

import matplotlib

matplotlib.use("Agg")

from factory6g.visualization.thesis_extension_figures import (
    export_single_json_ber,
    export_single_json_sweeps,
    plot_ch04_ber_heatmap,
    plot_coupled_ber_heatmap,
    plot_multi_panel_ber,
    plot_multi_panel_metric,
    plot_rm_pareto,
    plot_runtime_bar,
    scan_results_inventory,
    write_ce_matrix_table,
    write_exploratory_anchor_table,
    write_run_inventory_tables,
)
from factory6g.visualization.thesis_summary_figures import load_stage_json


def _resolve(path: str | Path, root: Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else root / p


def export_manifest(manifest_path: Path, *, project_root: Path, skip_tables: bool = False) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    out_dir = _resolve(manifest.get("output_dir", "thesis/figures"), project_root)
    generated_dir = _resolve(manifest.get("generated_dir", "thesis/generated"), project_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    generated_dir.mkdir(parents=True, exist_ok=True)

    for entry in manifest.get("figures", []):
        entry_id = entry["id"]
        entry_type = entry["type"]

        if entry_type == "single_json_sweeps":
            source = _resolve(entry["sources"][0], project_root)
            if not source.exists():
                raise FileNotFoundError(f"[{entry_id}] missing {source}")
            export_single_json_sweeps(
                json_path=source,
                output_prefix=entry["output_prefix"],
                out_dir=out_dir,
            )
            continue

        if entry_type == "single_json_ber":
            source = _resolve(entry["sources"][0], project_root)
            output = out_dir / entry["output"]
            export_single_json_ber(
                json_path=source,
                output_path=output,
                title=entry.get("title", "BER vs Eb/No"),
            )
            continue

        if entry_type == "ber_heatmap":
            plot_ch04_ber_heatmap(
                run_b_dir=_resolve(entry["run_b_dir"], project_root),
                ebno_db=float(entry.get("ebno_db", 0.0)),
                output_path=out_dir / entry["output"],
            )
            continue

        if entry_type == "runtime_bar":
            source = _resolve(entry["sources"][0], project_root)
            payload = load_stage_json(source)
            plot_runtime_bar(
                payload=payload,
                output_path=out_dir / entry["output"],
                title=entry.get("title", "Runtime by method"),
            )
            continue

        if entry_type == "multi_panel_ber":
            panels = [
                {**panel, "source": str(_resolve(panel["source"], project_root))}
                for panel in entry["panels"]
            ]
            for panel in panels:
                if not Path(panel["source"]).exists():
                    raise FileNotFoundError(f"[{entry_id}] missing {panel['source']}")
            plot_multi_panel_ber(
                panels=panels,
                output_path=out_dir / entry["output"],
                supertitle=entry.get("supertitle"),
                methods_filter=entry.get("methods_filter"),
                ncols=entry.get("ncols"),
            )
            continue

        if entry_type == "multi_panel_metric":
            panels = [
                {**panel, "source": str(_resolve(panel["source"], project_root))}
                for panel in entry["panels"]
            ]
            for panel in panels:
                if not Path(panel["source"]).exists():
                    raise FileNotFoundError(f"[{entry_id}] missing {panel['source']}")
            plot_multi_panel_metric(
                panels=panels,
                metric=entry["metric"],
                ylabel=entry["ylabel"],
                output_path=out_dir / entry["output"],
                supertitle=entry.get("supertitle"),
                methods_filter=entry.get("methods_filter"),
                ncols=entry.get("ncols"),
            )
            continue

        if entry_type == "coupled_ber_heatmap":
            panels = [
                {**panel, "source": str(_resolve(panel["source"], project_root))}
                for panel in entry["panels"]
            ]
            for panel in panels:
                if not Path(panel["source"]).exists():
                    raise FileNotFoundError(f"[{entry_id}] missing {panel['source']}")
            plot_coupled_ber_heatmap(
                panels=panels,
                ebno_db=float(entry.get("ebno_db", 0.0)),
                output_path=out_dir / entry["output"],
            )
            continue

        if entry_type == "rm_pareto":
            plot_rm_pareto(
                run_b_dir=_resolve(entry["run_b_dir"], project_root),
                ebno_db=float(entry.get("ebno_db", 0.0)),
                output_path=out_dir / entry["output"],
            )
            continue

        raise ValueError(f"Unknown figure type '{entry_type}' for id={entry_id}")

    if skip_tables:
        return

    tables = manifest.get("tables", {})
    if "ch04_ce_exploratory_anchor" in tables:
        spec = tables["ch04_ce_exploratory_anchor"]
        write_exploratory_anchor_table(
            rows=spec["rows"],
            ebno_db=float(spec.get("ebno_db", 0.0)),
            output_path=_resolve(spec["output"], project_root),
            note=spec.get("note", ""),
        )

    if "ch04_ce_matrix" in tables:
        spec = tables["ch04_ce_matrix"]
        write_ce_matrix_table(
            panels=[
                {**panel, "source": str(_resolve(panel["source"], project_root))}
                for panel in spec["panels"]
            ],
            ber_ebno_db=spec["ber_ebno_db"],
            nmse_ebno_db=spec["nmse_ebno_db"],
            calibration_ebno_db=float(spec["calibration_ebno_db"]),
            output_path=_resolve(spec["output"], project_root),
            caption=spec["caption"],
            note=spec.get("note", ""),
        )

    inventory = scan_results_inventory(project_root / "results")
    write_run_inventory_tables(
        inventory=inventory,
        output_path=generated_dir / "appendix_b_inventory.tex",
        cleanup=manifest.get("cleanup", {}),
        roles=manifest.get("run_roles", {}),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export all thesis figures from figure_manifest.json")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=project_root / "thesis" / "figures" / "manifest.json",
        help="Path to figure manifest",
    )
    parser.add_argument("--skip-tables", action="store_true", help="Only export PNG figures")
    args = parser.parse_args()
    export_manifest(args.manifest.resolve(), project_root=project_root, skip_tables=args.skip_tables)
    print("Export complete.")


if __name__ == "__main__":
    main()
