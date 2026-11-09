"""Reliability report CLI.

    python -m factory6g.analysis report results/<run> [results/<run_seed43> ...] --baseline round_robin
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m factory6g.analysis", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    rep = sub.add_parser("report", help="Batch-level intervals, paired tests and seed summaries for traced runs.")
    rep.add_argument("runs", nargs="+", help="Run directories; several runs that differ only in the seed give the seed-level summary.")
    rep.add_argument("--baseline", help="Method every other method is tested against (e.g. round_robin, ls).")
    rep.add_argument("--stage", help="Only this stage (estimators, resource_managers, coupled).")
    rep.add_argument("--out", default=None, help="Output directory (default: <first run>/reliability).")
    args = parser.parse_args(argv)

    from factory6g.analysis.reliability import build_report, write_report

    runs = [Path(r) for r in args.runs]
    report = build_report(runs, baseline=args.baseline, stage=args.stage)
    for path in write_report(report, Path(args.out) if args.out else runs[0] / "reliability"):
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
