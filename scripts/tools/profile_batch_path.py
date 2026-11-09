#!/usr/bin/env python3
"""Time each step of the resource-manager stage's batch path on the real model.

Wraps the model, receiver, channel, transmitter and scheduler calls the stage
makes, runs a few warm-up batches, then reports where the wall-clock time of
one batch goes. Run inside the simulation container:

    docker run --rm -v "$PWD":/app -w /app --entrypoint python factory6g:latest \
        scripts/tools/profile_batch_path.py --config config/config.json --batches 10
"""

from __future__ import annotations

import argparse
import functools
import os
import time
from collections import defaultdict

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

TIMES: dict[str, float] = defaultdict(float)
CALLS: dict[str, int] = defaultdict(int)


def timed(label: str, fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            TIMES[label] += time.perf_counter() - start
            CALLS[label] += 1
    return wrapper


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config/config.json")
    parser.add_argument("--batches", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--ebno", type=float, default=6.0)
    parser.add_argument("--schedulers", default="static,round_robin,max_throughput,pf")
    args = parser.parse_args()

    import numpy as np

    from factory6g.models.model import Model
    from factory6g.models.resource_manager import create_resource_manager
    from factory6g.sim.config import load_config
    from factory6g.sim.stages import common
    from factory6g.sim.stages.common import extract_error_stats, per_ut_delivered_bits, resolve_kwargs, transmitted_ut_mask

    config = load_config(args.config)
    system = config.system_runtime_config
    num_ut = int(system.get("num_ut", 4))
    model = Model(config=system, estimator_type="adaptive",
                  estimator_kwargs=resolve_kwargs(config.estimators.kwargs, "adaptive"), perfect_csi=False)
    schedulers = [s.strip() for s in args.schedulers.split(",") if s.strip()]
    managers = {
        s: create_resource_manager(s, num_ut=num_ut, num_active=config.resource_managers.num_active_users,
                                   cnn_model_path=config.resource_managers.cnn_model_path,
                                   drl_model_path=config.resource_managers.drl_model_path,
                                   manager_kwargs=resolve_kwargs(config.resource_managers.kwargs, s))
        for s in schedulers
    }

    # Wrap the steps. Inner steps are counted inside their parents too, so the
    # report lists parents and children separately.
    model._channel.sample_frequency_response = timed("  channel draw (hall + small scale)", model._channel.sample_frequency_response)
    model._channel.apply_frequency_response = timed("  apply channel", model._channel.apply_frequency_response)
    model._channel.sample_noise = timed("  sample noise", model._channel.sample_noise)
    model._transmitter.call = timed("  transmitter (bits, LDPC, QAM, grid)", model._transmitter.call)
    model._receiver.estimate_channel = timed("  channel estimation", model._receiver.estimate_channel)
    model._receiver.equalize = timed("  LMMSE equalisation", model._receiver.equalize)
    model._receiver.demap = timed("  demapping", model._receiver.demap)
    model._receiver.decode = timed("  LDPC decoding", model._receiver.decode)
    prepare = timed("prepare_batch_context", model.prepare_batch_context)
    run = timed("run_batch", model.run_batch)
    errors = timed("error counting (NumPy)", extract_error_stats)
    delivered = timed("delivered bits (NumPy)", per_ut_delivered_bits)
    del common

    batch_size = config.monte_carlo.batch_size
    total = 0.0
    for b in range(args.warmup + args.batches):
        if b == args.warmup:
            TIMES.clear()
            CALLS.clear()
        start = time.perf_counter()
        context = prepare(batch_size=batch_size, ebno_db=args.ebno, include_feedback=True)
        for name, manager in managers.items():
            sched = timed(f"scheduler {name}", manager.get_runtime_directives)
            directives = sched(system, args.ebno, feedback=context.feedback if manager.needs_channel_feedback else None)
            result = run(context, directives=directives, include_details=True)
            mask = transmitted_ut_mask(directives, num_ut=num_ut)
            errors(result["bits"], result["bits_hat"], ut_mask=mask)
            delivered(result["bits"], result["bits_hat"], ut_mask=mask)
        if b >= args.warmup:
            total += time.perf_counter() - start

    n = args.batches
    print(f"\n{n} batches after {args.warmup} warm-up · batch size {batch_size} · {len(managers)} schedulers · "
          f"{args.ebno} dB · channel {system.get('channel_model_type')} · CPU")
    print(f"{'step':44s} {'ms / batch':>10s} {'share':>7s} {'calls/batch':>11s}")
    print(f"{'whole batch (one Eb/N0 point)':44s} {1000 * total / n:10.1f} {'100 %':>7s}")
    for label, t in sorted(TIMES.items(), key=lambda kv: -kv[1]):
        print(f"{label:44s} {1000 * t / n:10.1f} {100 * t / total:6.1f}% {CALLS[label] / n:11.1f}")
    del np
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
