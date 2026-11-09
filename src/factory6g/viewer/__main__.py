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
    scene = None
    traces = find_traces(run_dir)
    if not traces:
        print(f"No trace files under {run_dir}.", file=sys.stderr)
        return 1
    if args.radio_map or args.scene:
        from factory6g.sim.trace import read_trace

        stage_traces = [read_trace(t) for t in traces]
        main_trace = next((t for t in stage_traces if t["header"]["stage"] in ("resource_managers", "coupled")), stage_traces[0])
        meta = main_trace["header"]["meta"]
        bank = meta.get("rt_channel_bank", "data/rt_channel_bank.npz")
        manifest = Path(bank).with_name(Path(bank).name + ".manifest.json")
        if args.scene:
            from factory6g.viewer.scene import compute_scene

            ebnos = meta["ebno_db_range"]
            ebno = float(args.scene_ebno) if args.scene_ebno is not None else float(ebnos[len(ebnos) // 2])
            batches = sorted({c["batch"] for c in main_trace["contexts"]})
            batch = int(args.scene_batch) if args.scene_batch is not None else batches[-1]
            ctx = next((c for c in main_trace["contexts"] if c["batch"] == batch and float(c["ebno_db"]) == ebno), None)
            if ctx is None or "positions" not in ctx:
                print(f"No terminal positions for batch {batch} at {ebno} dB in the trace.", file=sys.stderr)
                return 1
            print(f"Tracing the scene for batch {batch + 1} at {ebno:g} dB from {manifest} ...", flush=True)
            result = compute_scene(manifest, ctx["positions"], out_dir=out_dir, max_depth=args.max_depth,
                                   num_subcarriers=int(meta.get("fft_size", 128)),
                                   subcarrier_spacing=float(meta.get("subcarrier_spacing", 30e3)))
            radio_map = result.pop("radio_map")
            scene = {"stage": main_trace["header"]["stage"], "batch": batch, "ebno_db": ebno, **result}
        else:
            from factory6g.viewer.radio_map import compute_radio_map

            print(f"Tracing the radio map from {manifest} ...", flush=True)
            radio_map = compute_radio_map(manifest, max_depth=args.max_depth)
    path = export_run(run_dir, out_dir, radio_map=radio_map, scene=scene)
    print(f"Wrote {path}")
    print(f"Serve it with: python -m factory6g.viewer serve {out_dir}")
    return 0


def _figures(args: argparse.Namespace) -> int:
    from factory6g.viewer.figures import make_figures

    written = make_figures(Path(args.run_dir), Path(args.out) if args.out else None, ebno_db=args.ebno)
    for path in written:
        print(f"Wrote {path}")
    return 0


def _record(args: argparse.Namespace) -> int:
    from factory6g.viewer.recording import write_recording

    path = write_recording(Path(args.run_dir), Path(args.out) if args.out else None, ebno_db=args.ebno)
    print(f"Wrote {path}")
    print(f"Open it with: rerun {path}")
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
    exp.add_argument("--scene", action="store_true", help="Also trace per-link paths and render the hall for one batch (implies --radio-map).")
    exp.add_argument("--scene-batch", type=int, help="Batch index for --scene (default: last batch).")
    exp.add_argument("--scene-ebno", type=float, help="Eb/N0 point for --scene (default: middle point).")
    exp.add_argument("--max-depth", type=int, default=5, help="Ray depth for the map and paths (the channel bank used 5).")
    exp.set_defaults(func=_export)
    fig = sub.add_parser("figures", help="Draw the thesis figure pack from a traced run.")
    fig.add_argument("run_dir")
    fig.add_argument("--out", help="Output directory (default: <run_dir>/figures).")
    fig.add_argument("--ebno", type=float, help="Eb/N0 for the throughput and fairness figures (default: middle point).")
    fig.set_defaults(func=_figures)
    rec = sub.add_parser("record", help="Write a Rerun recording (.rrd) of a traced run.")
    rec.add_argument("run_dir")
    rec.add_argument("--out", help="Output file (default: <run_dir>/recording.rrd).")
    rec.add_argument("--ebno", type=float, help="Eb/N0 point to record (default: middle point).")
    rec.set_defaults(func=_record)
    srv = sub.add_parser("serve", help="Serve an exported viewer directory.")
    srv.add_argument("directory")
    srv.add_argument("--host", default="0.0.0.0")
    srv.add_argument("--port", type=int, default=8765)
    srv.set_defaults(func=_serve)
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
