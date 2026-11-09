"""Run viewer CLI.

    python -m factory6g.viewer export results/<run>            # writes results/<run>/viewer/
    python -m factory6g.viewer export results/<run> --radio-map
    python -m factory6g.viewer serve results/<run>/viewer --port 8765
"""

from __future__ import annotations

import argparse
import functools
import http.server
import sys
from pathlib import Path


def _export(args: argparse.Namespace) -> int:
    from factory6g.viewer.export import export_run, find_traces

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out) if args.out else run_dir / "viewer"
    radio_map = None
    if args.radio_map:
        from factory6g.sim.trace import read_trace
        from factory6g.viewer.radio_map import compute_radio_map

        traces = find_traces(run_dir)
        if not traces:
            print(f"No trace files under {run_dir}.", file=sys.stderr)
            return 1
        bank = read_trace(traces[0])["header"]["meta"].get("rt_channel_bank", "data/rt_channel_bank.npz")
        manifest = Path(bank).with_name(Path(bank).name + ".manifest.json")
        print(f"Tracing the radio map from {manifest} ...", flush=True)
        radio_map = compute_radio_map(manifest, max_depth=args.max_depth)
    path = export_run(run_dir, out_dir, radio_map=radio_map)
    print(f"Wrote {path}")
    print(f"Serve it with: python -m factory6g.viewer serve {out_dir}")
    return 0


class _NoCacheHandler(http.server.SimpleHTTPRequestHandler):
    """Re-exporting a run rewrites the same file names, so never let the browser cache them."""

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


def _serve(args: argparse.Namespace) -> int:
    directory = Path(args.directory)
    if not (directory / "data.json").exists():
        print(f"{directory} has no data.json; run the export command first.", file=sys.stderr)
        return 1
    handler = functools.partial(_NoCacheHandler, directory=str(directory))
    with http.server.ThreadingHTTPServer((args.host, args.port), handler) as server:
        print(f"Viewer on http://localhost:{args.port}  (Ctrl+C to stop)", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m factory6g.viewer", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    exp = sub.add_parser("export", help="Export a traced run for the viewer.")
    exp.add_argument("run_dir")
    exp.add_argument("--out", help="Output directory (default: <run_dir>/viewer).")
    exp.add_argument("--radio-map", action="store_true", help="Trace a Sionna RT path-gain map of the hall.")
    exp.add_argument("--max-depth", type=int, default=3, help="Radio-map ray depth.")
    exp.set_defaults(func=_export)
    srv = sub.add_parser("serve", help="Serve an exported viewer directory.")
    srv.add_argument("directory")
    srv.add_argument("--host", default="0.0.0.0")
    srv.add_argument("--port", type=int, default=8765)
    srv.set_defaults(func=_serve)
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
