#!/usr/bin/env python3
"""Export Chapter 3 matplotlib methodology figures (scaled factory plan view)."""

from __future__ import annotations

import argparse
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]

import matplotlib

matplotlib.use("Agg")

from factory6g.visualization.thesis_methodology_figures import render_all


def main() -> None:
    parser = argparse.ArgumentParser(description="Export thesis Ch3 methodology figures")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=project_root / "thesis" / "figures",
        help="Output directory for the rendered PNGs",
    )
    args = parser.parse_args()

    produced = render_all(args.out_dir)
    for name, path in sorted(produced.items()):
        print(f"Wrote {name}: {path}")


if __name__ == "__main__":
    main()
