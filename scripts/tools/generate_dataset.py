from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
import sys
from typing import Any

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from factory6g.sim.env import configure_env

configure_env(force_cpu=True, gpu_num=0)

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from factory6g.models.model import Model
from factory6g.models.resource_manager import ResourceDirectives, create_resource_manager
from factory6g.sim.config import load_config
from factory6g.sim.stages.common import ber_upper_confidence_bound, extract_error_stats


def _preprocess_channel_for_cnn(h_hat: tf.Tensor) -> np.ndarray:
    h_hat = tf.cast(h_hat, tf.complex64)
    power = tf.abs(h_hat) ** 2
    power = tf.reduce_mean(power, axis=1)
    power = tf.reduce_mean(power, axis=1)
    power = tf.reduce_mean(power, axis=2)
    channel_energy = tf.reduce_mean(power, axis=2)
    return channel_energy[0].numpy().astype(np.float32)


_VALID_CHANNELS = {"tr38901", "rayleigh", "rician", "raytraced", "awgn"}
_BER_FIRST_SOURCE_PRIORITY = {
    "max_throughput": 0,
    "wmmse": 1,
    "drl": 2,
    "queue_aware": 3,
    "pf": 4,
    "round_robin": 5,
    "static": 6,
    "default": 7,
}


def _source_priority(source: str) -> int:
    if source.startswith("random_"):
        return 50
    return _BER_FIRST_SOURCE_PRIORITY.get(source, 25)


def _sample_random_allocation(
    num_ut: int,
    *,
    active_count: int | None = None,
    min_power: float = 0.1,
    max_power: float = 1.0,
) -> tuple[list[int], list[float]]:
    if active_count is None:
        num_active = np.random.randint(1, num_ut + 1)
    else:
        num_active = max(1, min(int(active_count), num_ut))
    active_indices = np.random.choice(num_ut, num_active, replace=False)
    mask = np.zeros(num_ut, dtype=np.int32)
    mask[active_indices] = 1
    power = np.zeros(num_ut, dtype=np.float32)
    power_low = float(np.clip(min_power, 0.0, 1.0))
    power_high = float(np.clip(max_power, power_low, 1.0))
    power[active_indices] = np.random.uniform(power_low, power_high, size=num_active).astype(np.float32)
    return mask.tolist(), power.tolist()


@dataclass(frozen=True)
class CandidateEvaluation:
    source: str
    directives: ResourceDirectives
    utility: float
    avg_ber: float
    ber_upper_confidence: float
    throughput_eff: float
    throughput_bits: float
    latency_ms: float
    bit_errors: int
    total_bits: int


def _candidate_metrics(
    res: dict[str, Any],
    latency_weight: float,
    *,
    confidence_level: float = 0.95,
    ut_mask: list[int] | None = None,
) -> dict[str, float | int]:
    bits = res["bits"]
    bits_hat = res["bits_hat"]
    stats = extract_error_stats(bits, bits_hat, ut_mask=ut_mask)
    total_bits = int(stats["total_bits"])
    bit_errors = int(stats["bit_errors"])
    avg_ber = bit_errors / total_bits if total_bits > 0 else 1.0
    throughput_bits = max(0.0, float(total_bits - bit_errors))
    throughput_eff = throughput_bits / max(float(total_bits), 1.0)
    latency_ms = float(res["latency_sec"]) * 1e3
    utility = throughput_eff - latency_weight * latency_ms
    return {
        "utility": float(utility),
        "avg_ber": float(avg_ber),
        "ber_upper_confidence": float(
            ber_upper_confidence_bound(bit_errors, total_bits, confidence_level)
        ),
        "throughput_eff": float(throughput_eff),
        "throughput_bits": float(throughput_bits),
        "latency_ms": float(latency_ms),
        "bit_errors": int(bit_errors),
        "total_bits": int(total_bits),
    }


def _score_candidate(res: dict, latency_weight: float) -> tuple[float, float, float, float]:
    metrics = _candidate_metrics(res, latency_weight)
    return (
        float(metrics["utility"]),
        float(metrics["avg_ber"]),
        float(metrics["throughput_eff"]),
        float(metrics["latency_ms"]),
    )


def _candidate_sort_key(candidate: CandidateEvaluation, objective: str) -> tuple[float, ...]:
    if objective == "ber_first":
        return (
            candidate.avg_ber,
            candidate.ber_upper_confidence,
            -candidate.throughput_bits,
            float(_source_priority(candidate.source)),
            candidate.latency_ms,
        )
    return (
        -candidate.utility,
        candidate.avg_ber,
        candidate.ber_upper_confidence,
        candidate.latency_ms,
    )


def _normalize_directives(directives: ResourceDirectives, num_ut: int) -> ResourceDirectives:
    if directives.active_ut_mask is None:
        mask = [1] * num_ut
    else:
        mask = [1 if int(value) else 0 for value in list(directives.active_ut_mask)[:num_ut]]
        mask.extend([0] * max(0, num_ut - len(mask)))
    if not any(mask):
        mask[0] = 1

    if directives.per_ut_power is None:
        power = [1.0 if mask[idx] else 0.0 for idx in range(num_ut)]
    else:
        power = [float(value) for value in list(directives.per_ut_power)[:num_ut]]
        power.extend([0.0] * max(0, num_ut - len(power)))
        power = [max(0.0, min(1.0, power[idx])) if mask[idx] else 0.0 for idx in range(num_ut)]

    return ResourceDirectives(
        active_ut_mask=mask,
        per_ut_power=power,
        pilot_reuse_factor=directives.pilot_reuse_factor or 1,
    )


def _evaluate_candidate(
    model: Model,
    context,
    *,
    source: str,
    directives: ResourceDirectives,
    latency_weight: float,
    confidence_level: float,
    num_ut: int,
) -> CandidateEvaluation:
    normalized = _normalize_directives(directives, num_ut)
    result = model.run_batch(context, directives=normalized, include_details=True)
    metrics = _candidate_metrics(
        result,
        latency_weight,
        confidence_level=confidence_level,
        ut_mask=normalized.active_ut_mask,
    )
    return CandidateEvaluation(
        source=source,
        directives=normalized,
        utility=float(metrics["utility"]),
        avg_ber=float(metrics["avg_ber"]),
        ber_upper_confidence=float(metrics["ber_upper_confidence"]),
        throughput_eff=float(metrics["throughput_eff"]),
        throughput_bits=float(metrics["throughput_bits"]),
        latency_ms=float(metrics["latency_ms"]),
        bit_errors=int(metrics["bit_errors"]),
        total_bits=int(metrics["total_bits"]),
    )


def _parse_manager_names(raw: str | None, *, objective: str) -> list[str]:
    if raw is None:
        if objective != "ber_first":
            return []
        return ["static", "round_robin", "max_throughput", "pf", "wmmse", "queue_aware", "drl"]
    return [item.strip().lower() for item in raw.split(",") if item.strip()]


def _load_runtime_config(config_path: str, scenario: str, channel: str | None = None) -> dict:
    app_config = load_config(config_path)
    requested_scenario = scenario.lower()
    if requested_scenario != app_config.system.scenario:
        raise ValueError(
            f"Scenario override '{scenario}' is not supported. Locked 5G preset requires "
            f"scenario='{app_config.system.scenario}'."
        )
    runtime_config = app_config.system_runtime_config
    if channel is not None:
        channel_lower = channel.lower()
        if channel_lower not in _VALID_CHANNELS:
            raise ValueError(f"channel must be one of: {', '.join(sorted(_VALID_CHANNELS))}.")
        runtime_config["channel_model_type"] = channel_lower
    return runtime_config


def _parse_ebno_grid(raw: str | None) -> list[float] | None:
    if raw is None:
        return None
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("--ebno-grid must contain at least one value.")
    return values


def _sample_ebno(
    sample_index: int,
    *,
    min_ebno: float,
    max_ebno: float,
    ebno_grid: list[float] | None,
    ebno_jitter: float,
) -> float:
    if ebno_grid is None:
        return float(np.random.uniform(min_ebno, max_ebno))
    base = float(ebno_grid[sample_index % len(ebno_grid)])
    if ebno_jitter > 0.0:
        base += float(np.random.uniform(-ebno_jitter, ebno_jitter))
    return float(np.clip(base, min_ebno, max_ebno))


def generate_dataset(
    output_path: str,
    samples: int,
    batch_size: int,
    scenario: str,
    min_ebno: float,
    max_ebno: float,
    seed: int,
    tries: int = 16,
    latency_weight: float = 0.002,
    config_path: str = "config/config.json",
    channel: str | None = None,
    ebno_grid: list[float] | None = None,
    ebno_jitter: float = 0.0,
    objective: str = "utility",
    candidate_managers: str | None = None,
    confidence_level: float = 0.95,
    random_active_count: int | None = None,
    random_power_min: float = 0.1,
    random_power_max: float = 1.0,
    label_active_count: int | None = None,
):
    if batch_size != 1:
        raise ValueError("For supervised RM dataset generation, batch_size must be 1.")
    if objective not in {"utility", "ber_first"}:
        raise ValueError("objective must be one of: utility, ber_first.")

    tf.random.set_seed(seed)
    np.random.seed(seed)

    app_config = load_config(config_path)
    system_config = _load_runtime_config(config_path, scenario, channel=channel)
    model = Model(config=system_config, perfect_csi=False, estimator_type="ls")
    data_records: list[dict] = []
    num_ut = int(model.get_config().get("num_ut", 8))
    if objective == "ber_first" and label_active_count is None:
        label_active_count = app_config.resource_managers.num_active_users

    manager_names = _parse_manager_names(candidate_managers, objective=objective)
    managers = {
        name: create_resource_manager(
            name,
            num_ut=num_ut,
            num_active=app_config.resource_managers.num_active_users,
            cnn_model_path=app_config.resource_managers.cnn_model_path,
            drl_model_path=app_config.resource_managers.drl_model_path,
            manager_kwargs=app_config.resource_managers.kwargs.get(name, {}),
        )
        for name in manager_names
    }

    print(f"Generating {samples} channel realizations with {tries} candidate allocations each...")
    print(f"Scenario: {scenario}, channel: {system_config.get('channel_model_type')}")
    if ebno_grid is None:
        print(f"Eb/No sampling: uniform [{min_ebno}, {max_ebno}] dB")
    else:
        print(f"Eb/No sampling: benchmark grid {ebno_grid} dB, jitter={ebno_jitter}")
    print(f"Objective: {objective}")
    if label_active_count is not None:
        print(f"Oracle label active-count filter: {label_active_count}")
    if manager_names:
        print(f"Oracle manager candidates: {manager_names}")
    print("Feature extraction: runtime-aligned with CNNResourceManager.preprocess_channel()")

    for sample_index in tqdm(range(samples)):
        ebno = _sample_ebno(
            sample_index,
            min_ebno=min_ebno,
            max_ebno=max_ebno,
            ebno_grid=ebno_grid,
            ebno_jitter=ebno_jitter,
        )
        context = model.prepare_batch_context(batch_size=1, ebno_db=ebno, include_feedback=True)
        if context.feedback is None:
            raise RuntimeError("Expected precomputed feedback for dataset generation.")

        channel_energy_np = _preprocess_channel_for_cnn(context.feedback.h_hat)
        default_directives = model.default_directives()
        candidate_evaluations = [
            _evaluate_candidate(
                model,
                context,
                source="default",
                directives=default_directives,
                latency_weight=latency_weight,
                confidence_level=confidence_level,
                num_ut=num_ut,
            )
        ]

        for manager_name, manager in managers.items():
            feedback = context.feedback if manager.needs_channel_feedback else None
            directives = manager.get_runtime_directives(system_config, ebno, feedback=feedback)
            candidate_evaluations.append(
                _evaluate_candidate(
                    model,
                    context,
                    source=manager_name,
                    directives=directives,
                    latency_weight=latency_weight,
                    confidence_level=confidence_level,
                    num_ut=num_ut,
                )
            )

        random_candidates = max(0, tries - len(candidate_evaluations))
        for random_index in range(random_candidates):
            cand_mask, cand_power = _sample_random_allocation(
                num_ut,
                active_count=random_active_count,
                min_power=random_power_min,
                max_power=random_power_max,
            )
            directives = ResourceDirectives(
                active_ut_mask=cand_mask,
                per_ut_power=cand_power,
                pilot_reuse_factor=1,
            )
            candidate_evaluations.append(
                _evaluate_candidate(
                    model,
                    context,
                    source=f"random_{random_index + 1}",
                    directives=directives,
                    latency_weight=latency_weight,
                    confidence_level=confidence_level,
                    num_ut=num_ut,
                )
            )

        eligible_candidates = candidate_evaluations
        if label_active_count is not None:
            eligible_candidates = [
                candidate
                for candidate in candidate_evaluations
                if sum(candidate.directives.active_ut_mask or []) == int(label_active_count)
            ]
            if not eligible_candidates:
                eligible_candidates = candidate_evaluations

        best = min(
            eligible_candidates,
            key=lambda candidate: _candidate_sort_key(candidate, objective),
        )
        best_mask = list(best.directives.active_ut_mask or [])
        best_power = list(best.directives.per_ut_power or [])

        data_records.append(
            {
                "scenario": scenario,
                "channel_model_type": system_config.get("channel_model_type"),
                "ebno_db": ebno,
                "sample_index": int(sample_index),
                "channel_energy": channel_energy_np.tolist(),
                "active_ut_mask": best_mask,
                "per_ut_power": best_power,
                "oracle_utility": float(best.utility),
                "oracle_avg_ber": float(best.avg_ber),
                "oracle_ber_upper_confidence": float(best.ber_upper_confidence),
                "oracle_throughput_eff": float(best.throughput_eff),
                "oracle_throughput_bits": float(best.throughput_bits),
                "oracle_latency_ms": float(best.latency_ms),
                "oracle_candidates": int(len(candidate_evaluations)),
                "oracle_eligible_candidates": int(len(eligible_candidates)),
                "oracle_objective": objective,
                "oracle_source_manager": best.source,
                "oracle_source_priority": int(_source_priority(best.source)),
            }
        )

    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("pandas is required to write the RM training parquet dataset.") from exc

    df = pd.DataFrame(data_records)
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"Dataset saved to {output_path}")
    print(f"Shape: {df.shape}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate Sionna-aligned dataset for 6G resource management CNN"
    )
    parser.add_argument("--config", type=str, default="config/config.json", help="Path to the config JSON file")
    parser.add_argument("--output", type=str, default="data/dataset.parquet", help="Output Parquet file path")
    parser.add_argument("--samples", type=int, default=100, help="Number of samples to generate")
    parser.add_argument("--batch-size", type=int, default=1, help="Must be 1 for per-sample supervised dataset")
    parser.add_argument("--scenario", type=str, default="umi", help="Channel scenario")
    parser.add_argument("--min-ebno", type=float, default=0.0, help="Min Eb/No (dB)")
    parser.add_argument("--max-ebno", type=float, default=20.0, help="Max Eb/No (dB)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--tries", type=int, default=16, help="Allocation candidates evaluated per channel realization")
    parser.add_argument("--latency-weight", type=float, default=0.002, help="Latency penalty weight in oracle utility")
    parser.add_argument(
        "--channel",
        choices=sorted(_VALID_CHANNELS),
        default=None,
        help="Channel model used for synthetic Sionna samples. Defaults to config system.channel_model_type.",
    )
    parser.add_argument(
        "--ebno-grid",
        type=str,
        default=None,
        help="Comma-separated Eb/No grid to cycle through, e.g. 0,2,4,6,8,10,12,14,16,18,20.",
    )
    parser.add_argument(
        "--ebno-jitter",
        type=float,
        default=0.0,
        help="Uniform +/- jitter applied to --ebno-grid samples, clipped to min/max Eb/No.",
    )
    parser.add_argument(
        "--objective",
        choices=["utility", "ber_first"],
        default="utility",
        help="Oracle objective used to select the best candidate allocation.",
    )
    parser.add_argument(
        "--candidate-managers",
        type=str,
        default=None,
        help=(
            "Comma-separated manager candidates for ber_first objective. "
            "Defaults to static,round_robin,max_throughput,pf,wmmse,queue_aware,drl."
        ),
    )
    parser.add_argument(
        "--confidence-level",
        type=float,
        default=0.95,
        help="Confidence level used for BER upper-bound tie-breaking.",
    )
    parser.add_argument(
        "--random-active-count",
        type=int,
        default=None,
        help="Active UT count for random oracle candidates. Defaults to random 1..num_ut.",
    )
    parser.add_argument(
        "--random-power-min",
        type=float,
        default=0.1,
        help="Minimum active per-UT power for random oracle candidates.",
    )
    parser.add_argument(
        "--random-power-max",
        type=float,
        default=1.0,
        help="Maximum active per-UT power for random oracle candidates.",
    )
    parser.add_argument(
        "--label-active-count",
        type=int,
        default=None,
        help=(
            "Require selected oracle labels to have this active UT count. "
            "Defaults to resource_managers.num_active_users for ber_first."
        ),
    )
    args = parser.parse_args()

    generate_dataset(
        output_path=args.output,
        samples=args.samples,
        batch_size=args.batch_size,
        scenario=args.scenario,
        min_ebno=args.min_ebno,
        max_ebno=args.max_ebno,
        seed=args.seed,
        tries=args.tries,
        latency_weight=args.latency_weight,
        config_path=args.config,
        channel=args.channel,
        ebno_grid=_parse_ebno_grid(args.ebno_grid),
        ebno_jitter=args.ebno_jitter,
        objective=args.objective,
        candidate_managers=args.candidate_managers,
        confidence_level=args.confidence_level,
        random_active_count=args.random_active_count,
        random_power_min=args.random_power_min,
        random_power_max=args.random_power_max,
        label_active_count=args.label_active_count,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
