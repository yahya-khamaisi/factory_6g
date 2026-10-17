#!/usr/bin/env python3
"""Export conceptual / motivation thesis figures (matplotlib).

Example (inside Docker):
    python scripts/tools/export_thesis_engagement_figures.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]

import matplotlib

matplotlib.use("Agg")

from factory6g.visualization.thesis_engagement_figures import (
    plot_ce_estimator_tradeoff,
    plot_drl_rm_loop,
    plot_industry50,
    plot_fading_mechanisms,
    plot_fading_scenarios,
    plot_lr_eval_coupling,
    plot_lr_taxonomy,
    plot_urllc_tradeoff,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export thesis engagement figures")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=project_root / "thesis" / "figures",
        help="Output directory (default: thesis/figures)",
    )
    args = parser.parse_args()

    out = args.out_dir
    # Chapter 1
    plot_urllc_tradeoff(output_path=out / "fig_urllc_tradeoff.png")
    # Chapter 2 literature-review figure set (all matplotlib since Aug 2026;
    # fig_lr_taxonomy and fig_lr_eval_coupling were draw.io sources before that).
    plot_lr_taxonomy(output_path=out / "fig_lr_taxonomy.png")
    plot_industry50(output_path=out / "fig_lr_industry50.png")
    plot_fading_scenarios(output_path=out / "fig_fading_scenarios.png")
    plot_fading_mechanisms(output_path=out / "fig_fading_mechanisms.png")
    plot_ce_estimator_tradeoff(output_path=out / "fig_ce_estimator_tradeoff.png")
    plot_drl_rm_loop(output_path=out / "fig_drl_rm_loop.png")
    plot_lr_eval_coupling(output_path=out / "fig_lr_eval_coupling.png")
    print(f"Engagement figures exported to {out}")


if __name__ == "__main__":
    main()
