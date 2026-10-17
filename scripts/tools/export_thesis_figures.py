#!/usr/bin/env python3
"""Export thesis-ready BER/metric plots from stage_results_v2.json.

Uses the shared thesis matplotlib style (serif, $E_b/N_0$ axis, canonical legend order).

Example (inside Docker):
    python scripts/tools/export_thesis_figures.py \\
        --json results/20260926_202358_neural_ls_max_throughput_rayleigh_qpsk_s/estimators/stage_results_v2.json \\
        --prefix ch04_estimators_rayleigh_qpsk
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]

import matplotlib

matplotlib.use("Agg")

from factory6g.sim.output import _plot_ber_publication, _plot_metric_vs_ebno


def main() -> None:
    parser = argparse.ArgumentParser(description="Export thesis figures from stage JSON")
    parser.add_argument("--json", type=Path, required=True, help="Path to stage_results_v2.json")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=project_root / "thesis" / "figures",
        help="Output directory (default: thesis/figures)",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="thesis",
        help="Filename prefix for exported PNGs",
    )
    parser.add_argument(
        "--no-title",
        action="store_true",
        help="Omit subplot titles (cleaner for LaTeX captions)",
    )
    args = parser.parse_args()

    payload = json.loads(args.json.read_text(encoding="utf-8"))
    stage = payload["stage"]
    ebno_range = payload["ebno_db_range"]
    methods = payload["methods"]
    stage_label = stage.replace("_", " ").title()
    title_suffix = "" if args.no_title else f"{stage_label}: "

    args.out_dir.mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt

    _plot_ber_publication(
        plt=plt,
        methods=methods,
        ebno_range=ebno_range,
        title=f"{title_suffix}BER vs Eb/No",
        output_path=args.out_dir / f"{args.prefix}_ber_vs_ebno.png",
        stage_hint=stage,
    )
    _plot_metric_vs_ebno(
        plt=plt,
        methods=methods,
        ebno_range=ebno_range,
        metric="throughput_bits_per_batch",
        ylabel="Throughput (bits/batch)",
        title=f"{title_suffix}Throughput vs Eb/No",
        output_path=args.out_dir / f"{args.prefix}_throughput_vs_ebno.png",
        stage_hint=stage,
    )

    print(f"Exported thesis figures to {args.out_dir} with prefix '{args.prefix}_'")


if __name__ == "__main__":
    main()
