"""Quiet, sequential runtime measurement for the channel estimators.

Runtimes logged during the Monte Carlo campaigns are affected by whatever else
shares the CPU. This script is the single timing pass reported as the RQ1 cost
axis: run it with nothing else running. For every channel model it draws a fixed
set of shared batch contexts (P1) and times, per estimator, (a) channel
estimation alone and (b) the full receive chain of ``Model.run_batch``
(estimation, equalisation, demapping, decoding), after warm-up batches that
absorb graph tracing.

Usage (inside Docker, nothing else running):
    python scripts/tools/time_estimators.py --config config/config_run_a.json \\
        --output results/timing/estimator_runtime.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from factory6g.sim.rng import seed_all

from factory6g.models.model import Model
from factory6g.sim.config import load_config
from factory6g.sim.stages.common import resolve_kwargs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="config/config.json")
    parser.add_argument("--estimators", default="ls,dft,lmmse,adaptive,neural")
    parser.add_argument("--channels", default="rayleigh,rician,tr38901,raytraced")
    parser.add_argument("--ebno", default="0,10,20", help="Comma-separated Eb/N0 points (dB).")
    parser.add_argument("--batches", type=int, default=20, help="Timed batches per channel and Eb/N0 point.")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=None, help="Default: monte_carlo.batch_size.")
    parser.add_argument("--seed", type=int, default=None, help="Default: simulation.seed.")
    parser.add_argument("--output", default="results/timing/estimator_runtime.json")
    args = parser.parse_args()

    config = load_config(args.config)
    seed = args.seed if args.seed is not None else config.simulation.seed
    batch_size = args.batch_size or config.monte_carlo.batch_size
    estimators = [e.strip() for e in args.estimators.split(",") if e.strip()]
    channels = [c.strip() for c in args.channels.split(",") if c.strip()]
    ebno_points = [float(v) for v in args.ebno.split(",")]

    results: dict[str, dict] = {}
    for channel in channels:
        cfg = dict(config.system_runtime_config, channel_model_type=channel)
        models = {
            est: Model(cfg, estimator_type=est, estimator_kwargs=resolve_kwargs(config.estimators.kwargs, est))
            for est in estimators
        }
        context_model = models[estimators[0]]
        seed_all(seed)
        contexts = {
            eb: [context_model.prepare_batch_context(batch_size, eb, include_feedback=False)
                 for _ in range(args.warmup + args.batches)]
            for eb in ebno_points
        }
        results[channel] = {}
        for est, model in models.items():
            est_times: list[float] = []
            chain_times: list[float] = []
            for eb in ebno_points:
                for i, ctx in enumerate(contexts[eb]):
                    y = model.get_channel().apply_frequency_response(
                        model.get_transmitter().call(batch_size, bits=ctx.source_bits)[0], ctx.h_freq
                    ) + ctx.data_noise
                    t0 = time.perf_counter()
                    h_hat, _ = model.get_receiver().estimate_channel(y, ctx.noise_variance)
                    _ = h_hat.numpy() if hasattr(h_hat, "numpy") else h_hat
                    t1 = time.perf_counter()
                    model.run_batch(ctx, include_details=False)
                    t2 = time.perf_counter()
                    if i >= args.warmup:
                        est_times.append(t1 - t0)
                        chain_times.append(t2 - t1)
            est_ms = 1e3 * np.asarray(est_times) / batch_size
            chain_ms = 1e3 * np.asarray(chain_times) / batch_size
            results[channel][est] = {
                "estimation_ms_per_slot_median": float(np.median(est_ms)),
                "estimation_ms_per_slot_p90": float(np.percentile(est_ms, 90)),
                "chain_ms_per_slot_median": float(np.median(chain_ms)),
                "chain_ms_per_slot_p90": float(np.percentile(chain_ms, 90)),
                "timed_batches": len(est_times),
            }
            print(channel, est, results[channel][est], flush=True)

    record = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": args.config,
        "batch_size": batch_size,
        "ebno_db": ebno_points,
        "warmup_batches": args.warmup,
        "host": {"machine": platform.machine(), "cpu_count": os.cpu_count(),
                 "torch_threads": [torch.get_num_threads(), torch.get_num_interop_threads()]},
        "note": "Measured sequentially with no other workload; per-slot = per-batch time / batch size.",
        "results": results,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
