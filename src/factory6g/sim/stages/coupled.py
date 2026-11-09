"""Coupled estimator x scheduler stage (Run D).

The resource-manager stage holds the estimator fixed (adaptive) and varies the
scheduler. This stage varies both: on every shared context, each estimator in
``config.estimators.enabled`` forms the scheduler feedback from the same probe
and then receives the data transmission, and each scheduler in
``config.resource_managers.enabled`` acts on that feedback. Method keys are
``"<estimator>+<scheduler>"``. The estimator ``perfect`` is a genie that hands
the true channel to both the scheduler and the equaliser.
"""

from __future__ import annotations

import dataclasses
import time
from pathlib import Path
from typing import Any

from factory6g.models.model import Model
from factory6g.models.resource_manager import create_resource_manager
from factory6g.sim.config import Factory6GConfig
from factory6g.sim.industry import make_scenario
from factory6g.sim.trace import make_trace_recorder, trace_meta

from .checkpoint import delete_checkpoint, load_checkpoint, reseed_for_resume, save_checkpoint
from .common import (
    append_point_metrics,
    classify_point_status,
    extract_error_stats,
    fmt_elapsed,
    initialize_stage_metrics,
    mc_stop_reason,
    per_ut_delivered_bits,
    resolve_kwargs,
    transmitted_ut_mask,
)
from .resource_managers import _new_rm_point_state

METHOD_SEPARATOR = "+"


def coupled_method_key(estimator: str, scheduler: str) -> str:
    return f"{estimator}{METHOD_SEPARATOR}{scheduler}"


def split_coupled_method_key(key: str) -> tuple[str, str]:
    estimator, _, scheduler = key.partition(METHOD_SEPARATOR)
    return estimator, scheduler


def run_coupled_stage(
    config: Factory6GConfig,
    *,
    checkpoint_dir: Path | None = None,
) -> dict[str, Any]:
    system_config = config.system_runtime_config
    estimators = list(config.estimators.enabled)
    schedulers = list(config.resource_managers.enabled)
    if not estimators or not schedulers:
        raise ValueError("The coupled stage needs at least one estimator and one scheduler.")
    ebno_db_range = [float(v) for v in config.monte_carlo.ebno_db_range]

    num_ut = int(system_config.get("num_ut", 8))
    if config.resource_managers.num_active_users > num_ut:
        raise ValueError("resource_managers.num_active_users must be <= system.num_ut.")

    methods = [coupled_method_key(e, s) for e in estimators for s in schedulers]
    aggregated = initialize_stage_metrics(methods)
    runtime_totals_sec = {method: 0.0 for method in methods}

    # The context generator only draws channels, noise and bits; its estimator is unused.
    context_model = Model(
        config=system_config,
        estimator_type="ls",
        estimator_kwargs=resolve_kwargs(config.estimators.kwargs, "ls"),
        perfect_csi=False,
    )
    models = {
        est: Model(
            config=system_config,
            estimator_type=est,
            estimator_kwargs=resolve_kwargs(config.estimators.kwargs, est),
            perfect_csi=(est == "perfect"),
        )
        for est in estimators
    }
    # One manager per (estimator, scheduler): stateful schedulers keep their own history.
    managers = {
        coupled_method_key(est, sch): create_resource_manager(
            sch,
            num_ut=num_ut,
            num_active=config.resource_managers.num_active_users,
            cnn_model_path=config.resource_managers.cnn_model_path,
            drl_model_path=config.resource_managers.drl_model_path,
            manager_kwargs=resolve_kwargs(config.resource_managers.kwargs, sch),
        )
        for est in estimators
        for sch in schedulers
    }
    point_state = {method: [_new_rm_point_state() for _ in ebno_db_range] for method in methods}

    start_batch_index = 0
    if checkpoint_dir is not None:
        ckpt = load_checkpoint(checkpoint_dir)
        if ckpt is not None:
            point_state = ckpt["point_state"]
            runtime_totals_sec = ckpt["runtime_totals_sec"]
            start_batch_index = int(ckpt["batch_index"]) + 1
            reseed_for_resume(config.simulation.seed, start_batch_index)
            print(
                f"[checkpoint] Resuming coupled stage from batch "
                f"{start_batch_index + 1}/{config.monte_carlo.max_batches}"
            )

    scenario = make_scenario(config)
    trace = make_trace_recorder(
        config.simulation.trace,
        checkpoint_dir,
        stage="coupled",
        meta=trace_meta(config, methods=methods, scenario=scenario),
        resume_from_batch=start_batch_index if start_batch_index > 0 else None,
    )

    total_points = len(methods) * len(ebno_db_range)
    stage_start = time.perf_counter()
    for batch_index in range(start_batch_index, config.monte_carlo.max_batches):
        remaining = sum(1 for m in methods for st in point_state[m] if not st["done"])
        if remaining == 0:
            break

        batch_start = time.time()
        print(
            f"Coupled batch {batch_index + 1}/{config.monte_carlo.max_batches} "
            f"(active points {remaining}/{total_points}) "
            f"[{fmt_elapsed(time.perf_counter() - stage_start)} elapsed] ... ",
            end="",
            flush=True,
        )

        for ebno_index, ebno_db in enumerate(ebno_db_range):
            if all(point_state[m][ebno_index]["done"] for m in methods):
                continue
            print(f"  snr={ebno_db:+.1f}dB ", end="", flush=True)
            # One draw per batch and point, shared by every estimator and scheduler (P1).
            industry_state = scenario.batch_state(batch_index) if scenario else None
            context_model.channel_hook = scenario.channel_hook(industry_state) if scenario else None
            base_context = context_model.prepare_batch_context(
                batch_size=config.monte_carlo.batch_size,
                ebno_db=ebno_db,
                include_feedback=False,
            )
            trace.record_context(batch_index, ebno_db, getattr(context_model, "last_large_scale", None),
                                 {"industry": industry_state} if industry_state else None)
            for est, model in models.items():
                open_methods = [
                    coupled_method_key(est, sch)
                    for sch in schedulers
                    if not point_state[coupled_method_key(est, sch)][ebno_index]["done"]
                ]
                if not open_methods:
                    continue
                fb_start = time.perf_counter()
                feedback = model.estimate_feedback(base_context)
                fb_elapsed = time.perf_counter() - fb_start
                context = dataclasses.replace(base_context, feedback=feedback)

                for method in open_methods:
                    manager = managers[method]
                    stats = point_state[method][ebno_index]
                    print(f"[{method}]", end="", flush=True)
                    method_start = time.perf_counter()
                    directives = manager.get_runtime_directives(
                        system_config,
                        ebno_db,
                        feedback=context.feedback if manager.needs_channel_feedback else None,
                    )
                    if scenario is not None:
                        directives = scenario.cap_power(directives)
                    result = model.run_batch(context, directives=directives, include_details=True)
                    # The feedback estimate is shared by the open schedulers of this estimator.
                    elapsed = time.perf_counter() - method_start + fb_elapsed / len(open_methods)
                    runtime_totals_sec[method] += elapsed
                    stats["runtime_sec"] += elapsed

                    active_ut_mask = transmitted_ut_mask(directives, num_ut=num_ut)
                    trace.record_method(
                        batch_index,
                        ebno_db,
                        method,
                        result=result,
                        active_mask=active_ut_mask,
                        per_ut_power=getattr(directives, "per_ut_power", None),
                        runtime_sec=elapsed,
                    )
                    batch_stats = extract_error_stats(result["bits"], result["bits_hat"], ut_mask=active_ut_mask)
                    stats["errors"] += batch_stats["bit_errors"]
                    if batch_stats["bit_errors"] > 0:
                        stats["error_batches"] = stats.get("error_batches", 0) + 1
                    stats["bits"] += batch_stats["total_bits"]
                    stats["block_errors"] += batch_stats["block_errors"]
                    stats["blocks"] += batch_stats["total_blocks"]
                    stats["throughput"] += max(0.0, batch_stats["total_bits"] - batch_stats["bit_errors"])
                    stats["latency_sec"] += float(result.get("runtime_latency_sec", 0.0))
                    stats["energy_joules"] += float(result.get("energy_joules", 0.0))
                    stats["latency_model_sec"] += float(result.get("latency_model_sec", 0.0))
                    stats["tx_power_w"] += float(result.get("tx_power_w", 0.0))
                    stats["tx_energy_joules"] += float(result.get("tx_energy_joules", 0.0))
                    delivered = per_ut_delivered_bits(result["bits"], result["bits_hat"], ut_mask=active_ut_mask)
                    if stats["per_ut_delivered_bits"] is None:
                        stats["per_ut_delivered_bits"] = [0] * len(delivered)
                    stats["per_ut_delivered_bits"] = [
                        int(a) + int(b) for a, b in zip(stats["per_ut_delivered_bits"], delivered)
                    ]
                    stats["num_batches"] += 1

                    stop_reason = mc_stop_reason(
                        num_batches=stats["num_batches"],
                        total_bits=stats["bits"],
                        total_block_errors=stats["block_errors"],
                        target_block_errors=config.monte_carlo.target_block_errors,
                        total_bit_errors=stats["errors"],
                        target_ber=config.monte_carlo.target_ber,
                        stop_policy=config.monte_carlo.stop_policy,
                        confidence_level=config.monte_carlo.confidence_level,
                        min_batches=config.monte_carlo.min_batches,
                        min_total_bits=config.monte_carlo.min_total_bits,
                        error_batches=stats.get("error_batches", 0),
                        min_error_batches=config.monte_carlo.min_error_batches,
                    )
                    if stop_reason is not None:
                        stats["done"] = True
                        stats["stop_reason"] = stop_reason
        print(f"Done ({time.time() - batch_start:.2f}s)")
        trace.flush()
        if checkpoint_dir is not None:
            save_checkpoint(
                checkpoint_dir,
                {
                    "batch_index": batch_index,
                    "point_state": point_state,
                    "runtime_totals_sec": runtime_totals_sec,
                },
            )

    if checkpoint_dir is not None:
        delete_checkpoint(checkpoint_dir)

    for method in methods:
        for stats in point_state[method]:
            safe_batches = max(1, int(stats["num_batches"]))
            avg_latency_sec = float(stats["latency_sec"]) / safe_batches
            avg_energy = float(stats["energy_joules"]) / safe_batches
            point = {
                "ber": (float(stats["errors"]) / float(stats["bits"])) if stats["bits"] > 0 else 0.0,
                "latency_ms": avg_latency_sec * 1000.0,
                "throughput_bits_per_batch": float(stats["throughput"]) / safe_batches,
                "energy_joules_per_batch": avg_energy,
                "avg_power_w": avg_energy / max(avg_latency_sec, 1e-12),
                "runtime_sec": float(stats["runtime_sec"]),
                "bit_errors": float(stats["errors"]),
                "total_bits": float(stats["bits"]),
                "block_errors": float(stats["block_errors"]),
                "total_blocks": float(stats["blocks"]),
                "num_batches": float(stats["num_batches"]),
                "stop_reason": str(stats["stop_reason"] or "max_batches"),
                "point_status": classify_point_status(
                    float(stats["errors"]),
                    stats.get("error_batches", 0),
                    config.monte_carlo.min_error_batches,
                ),
                "error_batches": int(stats.get("error_batches", 0)),
                "latency_model_ms": float(stats["latency_model_sec"]) / safe_batches * 1000.0,
                "tx_power_w": float(stats["tx_power_w"]) / safe_batches,
                "tx_energy_j_per_batch": float(stats["tx_energy_joules"]) / safe_batches,
                "per_ut_delivered_bits": list(stats["per_ut_delivered_bits"] or []),
            }
            append_point_metrics(
                aggregated[method],
                confidence_level=config.monte_carlo.confidence_level,
                point=point,
            )

    return {
        "stage": "coupled",
        "ebno_db_range": ebno_db_range,
        "methods": aggregated,
        "runtime_totals_sec": runtime_totals_sec,
    }
