"""Build the ray-traced channel bank for the canonical factory hall.

Traces terminal positions in the hall of Figures 3.15/3.16 with Sionna RT and
writes ``data/rt_channel_bank.npz`` plus a provenance manifest. Positions are
split spatially (checkerboard with a guard band) into ``train`` and ``eval``.

Usage (inside Docker):
    python scripts/tools/build_rt_channel_bank.py --train 8192 --eval 4096
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from factory6g.components import raytracing as rt


def _fill_split(rng, want: int, split_id: int, hall: rt.HallSpec, args) -> np.ndarray:
    kept: list[np.ndarray] = []
    have = 0
    while have < want:
        cand = rt.sample_terminal_positions(
            rng,
            4 * (want - have) + 64,
            room_dimensions=hall.room_dimensions,
            boundary_padding=args.boundary_padding,
            height=hall.ut_height,
            machines=hall.machines,
        )
        cand = cand[rt.checkerboard_split(cand, cell=args.cell, guard=args.guard) == split_id]
        kept.append(cand[: want - have])
        have += len(kept[-1])
    return np.concatenate(kept, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="config/config.json")
    parser.add_argument("--output", default="data/rt_channel_bank.npz")
    parser.add_argument("--train", type=int, default=8192)
    parser.add_argument("--eval", type=int, default=4096)
    parser.add_argument("--layout-seed", type=int, default=42, help="Machine layout seed (matches the thesis figures).")
    parser.add_argument("--position-seed", type=int, default=2026)
    parser.add_argument("--cell", type=float, default=1.5, help="Checkerboard cell size in metres.")
    parser.add_argument("--guard", type=float, default=0.25, help="Guard band inside each cell edge in metres.")
    parser.add_argument("--boundary-padding", type=float, default=None)
    parser.add_argument("--chunk", type=int, default=128)
    args = parser.parse_args()

    raw = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if args.boundary_padding is None:
        args.boundary_padding = float(raw["transceiver"].get("rx_boundary_padding", 1.0))
    rt_cfg = raw["ray_tracing"]
    system = raw["system"]

    import mitsuba as mi

    if mi.variant() is None:  # Sionna RT 2.x without a GPU: Dr.Jit's LLVM (CPU) backend
        mi.set_variant("llvm_ad_mono_polarized")
    hall = rt.HallSpec.from_config(raw, layout_seed=args.layout_seed)
    scene = rt.build_scene(hall)

    freqs = rt.subcarrier_frequencies(int(system["fft_size"]), float(system["subcarrier_spacing"]))
    rng = np.random.default_rng(args.position_seed)

    splits = {}
    stats = {}
    for name, want, split_id in (("train", args.train, 0), ("eval", args.eval, 1)):
        positions = _fill_split(rng, want, split_id, hall, args)
        t0 = time.time()
        result = rt.trace_positions(
            scene, positions, freqs,
            max_depth=int(rt_cfg["max_depth"]),
            samples_per_src=int(rt_cfg["samples_per_src"]),
            chunk=args.chunk,
            seed=args.position_seed + 1_000_000 * split_id,
        )
        finite_k = result.k_factor_db[np.isfinite(result.k_factor_db)]
        stats[name] = {
            "count": int(len(positions)),
            "trace_seconds": round(time.time() - t0, 1),
            "los_fraction": float(result.los.mean()),
            "k_factor_db_median": float(np.median(finite_k)) if finite_k.size else None,
            "k_factor_db_p10_p90": [float(np.percentile(finite_k, 10)), float(np.percentile(finite_k, 90))] if finite_k.size else None,
            "rms_delay_spread_ns_median": float(np.median(result.rms_delay_spread_ns)),
            "rms_delay_spread_ns_p90": float(np.percentile(result.rms_delay_spread_ns, 90)),
            "num_paths_median": float(np.median(result.num_paths)),
        }
        print(f"[{name}] {stats[name]}", flush=True)
        splits[name] = (positions, result)

    manifest = {
        "format_version": rt.BANK_FORMAT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),
        "hall": hall.to_dict(),
        "layout_seed": args.layout_seed,
        "position_seed": args.position_seed,
        "split": {"method": "checkerboard", "cell_m": args.cell, "guard_m": args.guard,
                  "min_train_eval_distance_m": 2 * args.guard},
        "ray_tracing": {"max_depth": int(rt_cfg["max_depth"]), "samples_per_src": int(rt_cfg["samples_per_src"]),
                        "los": True, "specular_reflection": True, "diffuse_reflection": False, "refraction": False},
        "fft_size": int(system["fft_size"]),
        "subcarrier_spacing": float(system["subcarrier_spacing"]),
        "array_shape": "[N, num_bs_ant, fft_size] uplink CFR, delays normalised, not power-normalised",
        "stats": stats,
        "tracer": rt.tracer_versions(),
    }
    rt.save_bank(Path(args.output), splits, manifest)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
