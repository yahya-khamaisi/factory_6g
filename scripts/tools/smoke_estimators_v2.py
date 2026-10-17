"""Estimator-side gate check for system model v2 (thesis/notes/SYSTEM_MODEL_V2.md).

Run after scripts/tools/smoke_system_model_v2.py, before the dataset is
regenerated. For each channel model and operating point it reports:
1. the neural conditioning input (each terminal's effective Eb/N0 from its LS
   grid energy): percentiles and the share clipped at the ends of the range
   recorded in LS_ENERGY_CONDITIONING;
2. how well that LS-energy estimate tracks the true per-terminal gain;
3. the adaptive estimator's per-terminal branch mix;
4. LS and adaptive NMSE, as a sanity check of the estimators on v2 channels.

Usage (inside Docker):
    python scripts/tools/smoke_estimators_v2.py --output results/smoke_v2/smoke_estimators.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from factory6g.components.estimators.neural_estimator import (
    LS_ENERGY_CONDITIONING,
    effective_ebno_db_np,
)
from factory6g.models.model import Model
from factory6g.sim.config import load_config
from factory6g.sim.stages.common import channel_error_stats, nmse_db, resolve_kwargs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="config/config_run_b.json")
    parser.add_argument("--channels", default="rayleigh,rician,tr38901,raytraced")
    parser.add_argument("--ebno", default="0,10,20")
    parser.add_argument("--batches", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--output", default="results/smoke_v2/smoke_estimators.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    lo_db, hi_db = LS_ENERGY_CONDITIONING["range_db"]
    report: dict = {"config": args.config, "conditioning": LS_ENERGY_CONDITIONING, "channels": {}}
    for channel in args.channels.split(","):
        sc = dict(cfg.system_runtime_config)
        sc["channel_model_type"] = channel
        ls_model = Model(sc, estimator_type="ls")
        adaptive = Model(sc, estimator_type="adaptive", estimator_kwargs=resolve_kwargs(cfg.estimators.kwargs, "adaptive"))
        adaptive_est = adaptive._receiver._channel_estimator
        per_ebno = {}
        for ebno in [float(v) for v in args.ebno.split(",")]:
            cond_all, true_eff, branches = [], [], Counter()
            acc = {"ls": [0.0, 0.0], "adaptive": [0.0, 0.0]}
            for _ in range(args.batches):
                ctx = ls_model.prepare_batch_context(args.batch_size, ebno, include_feedback=False)
                gains = ls_model._channel.last_large_scale
                r_ls = ls_model.run_batch(ctx, include_details=True)
                cond = effective_ebno_db_np(r_ls["channel_hat"], r_ls["err_var_mean"], ebno)  # [B,1,M,U,1]
                cond_all.append(cond[0, 0, 0, :, 0])
                if gains is not None:
                    true_eff.append(ebno + np.asarray(gains["gain_db"], dtype=np.float64))
                r_ad = adaptive.run_batch(ctx, include_details=True)
                branches.update(adaptive_est.last_branch if isinstance(adaptive_est.last_branch, list)
                                else [adaptive_est.last_branch])
                for name, r in (("ls", r_ls), ("adaptive", r_ad)):
                    st = channel_error_stats(r["channel"], r["channel_hat"])
                    acc[name][0] += st["sq_error"]
                    acc[name][1] += st["channel_energy"]
            cond_arr = np.concatenate(cond_all)
            entry = {
                "cond_db_percentiles": dict(zip(("p1", "p10", "p50", "p90", "p99"),
                                                np.percentile(cond_arr, [1, 10, 50, 90, 99]).round(2).tolist())),
                "share_clipped_low": float(np.mean(cond_arr <= lo_db)),
                "share_clipped_high": float(np.mean(cond_arr >= hi_db)),
                "adaptive_branch_share": {k: v / sum(branches.values()) for k, v in sorted(branches.items())},
                "nmse_db": {k: nmse_db(*v) for k, v in acc.items()},
            }
            if true_eff:
                err = cond_arr - np.concatenate(true_eff)
                entry["cond_minus_true_db"] = {"mean": float(err.mean()), "std": float(err.std()),
                                               "p5": float(np.percentile(err, 5)), "p95": float(np.percentile(err, 95))}
            per_ebno[str(ebno)] = entry
            print(channel, ebno, json.dumps(entry))
        report["channels"][channel] = per_ebno
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
