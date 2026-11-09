"""Gate check for system model v2 (thesis/notes/SYSTEM_MODEL_V2.md) before retraining.

For each channel model it reports:
1. the spread of each terminal's effective SNR (Eb/N0 + large-scale gain re the
   hall average) at 0 and 20 dB, from many batch draws;
2. the LS BER waterfall with all four terminals active, to check that the
   Eb/N0 sweep still spans it;
3. whether an energy-based scheduler now separates from round-robin under genie
   CSI: the mean gain of the served terminals and the resulting BER.

Usage (inside Docker):
    python scripts/tools/smoke_system_model_v2.py --output results/smoke_v2/smoke.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from factory6g.sim.rng import seed_all

from factory6g.models.model import Model
from factory6g.models.resource_manager import MaxThroughputResourceManager, RoundRobinResourceManager
from factory6g.sim.config import load_config
from factory6g.sim.stages.common import extract_error_stats


def ber(model: Model, ctx, directives=None) -> tuple[int, int]:
    out = model.run_batch(ctx, directives=directives, include_details=False)
    mask = None if directives is None else directives.active_ut_mask
    stats = extract_error_stats(out["bits"], out["bits_hat"], ut_mask=mask)
    return int(stats["bit_errors"]), int(stats["total_bits"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="config/config_run_b.json")
    parser.add_argument("--channels", default="rayleigh,rician,tr38901,raytraced")
    parser.add_argument("--ebno", default="-10,-5,0,5,10,15,20,25,30")
    parser.add_argument("--batches", type=int, default=25)
    parser.add_argument("--gain-draws", type=int, default=2000)
    parser.add_argument("--sched-ebno", type=float, default=5.0)
    parser.add_argument("--output", default="results/smoke_v2/smoke.json")
    args = parser.parse_args()

    config = load_config(args.config)
    seed = config.simulation.seed
    batch = config.monte_carlo.batch_size
    report: dict[str, dict] = {}
    for ch in [c.strip() for c in args.channels.split(",") if c.strip()]:
        cfg = dict(config.system_runtime_config, channel_model_type=ch, large_scale_model="hall")
        rec: dict[str, object] = {}

        # 1. Effective-SNR spread of the terminals.
        seed_all(seed)
        ls_model = Model(cfg, estimator_type="ls")
        chan = ls_model.get_channel()
        gains = []
        for _ in range(args.gain_draws // 4):
            chan.sample_frequency_response(1)
            gains.append(chan.last_large_scale["gain_db"])
        g = np.concatenate(gains)
        pct = {p: float(np.percentile(g, p)) for p in (1, 10, 50, 90, 99)}
        rec["gain_db_percentiles"] = pct
        rec["effective_snr_db_at_0dB"] = pct
        rec["effective_snr_db_at_20dB"] = {p: v + 20.0 for p, v in pct.items()}
        rec["mean_linear_gain"] = float(np.mean(10 ** (g / 10)))

        # 2. LS waterfall, all terminals active.
        seed_all(seed)
        wf = {}
        for eb in [float(v) for v in args.ebno.split(",")]:
            e = n = 0
            for _ in range(args.batches):
                ctx = ls_model.prepare_batch_context(batch, eb, include_feedback=False)
                de, dn = ber(ls_model, ctx)
                e += de
                n += dn
            wf[f"{eb:g}"] = e / n
        rec["ls_ber_all_active"] = wf

        # 3. Genie-CSI scheduling: max-throughput vs round-robin on shared contexts.
        seed_all(seed)
        genie = Model(cfg, perfect_csi=True)
        mt, rr = MaxThroughputResourceManager(num_active=2), RoundRobinResourceManager(num_active=2)
        stats = {"max_throughput": [0, 0, []], "round_robin": [0, 0, []]}
        for _ in range(args.batches):
            ctx = genie.prepare_batch_context(batch, args.sched_ebno, include_feedback=True)
            gain_db = genie.get_channel().last_large_scale["gain_db"]
            for name, mgr in (("max_throughput", mt), ("round_robin", rr)):
                d = mgr.get_runtime_directives(cfg, args.sched_ebno, ctx.feedback)
                de, dn = ber(genie, ctx, d)
                stats[name][0] += de
                stats[name][1] += dn
                stats[name][2].append(float(np.mean(gain_db[np.asarray(d.active_ut_mask, bool)])))
        rec["genie_scheduling"] = {
            k: {"ber": v[0] / max(v[1], 1), "mean_served_gain_db": float(np.mean(v[2]))}
            for k, v in stats.items()
        }
        report[ch] = rec
        print(ch, json.dumps(rec), flush=True)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"config": args.config, "batch_size": batch, "results": report}, indent=2) + "\n",
                   encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
