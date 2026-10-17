from __future__ import annotations

import math
import time
from statistics import NormalDist
from typing import TYPE_CHECKING, Any, Callable, Sequence

import numpy as np

from factory6g.models.resource_manager import ResourceDirectives

if TYPE_CHECKING:
    from factory6g.models.model import Model


MIN_RESOLVED_BIT_ERRORS = 30


def fmt_elapsed(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"
POINT_STATUS_RESOLVED = "resolved"
POINT_STATUS_UPPER_BOUND_ONLY = "upper_bound_only"


METRIC_KEYS = (
    "ber",
    "ber_upper_confidence",
    "latency_ms",
    "throughput_bits_per_batch",
    "energy_joules_per_batch",
    "avg_power_w",
    "runtime_sec",
    "bit_errors",
    "total_bits",
    "block_errors",
    "total_blocks",
    "num_batches",
    "stop_reason",
    "point_status",
    "latency_model_ms",
    "tx_power_w",
    "tx_energy_j_per_batch",
    "per_ut_delivered_bits",
    "jain_fairness",
    "nmse_db",
    "err_var_calibration",
    "error_batches",
)


def initialize_stage_metrics(methods: list[str]) -> dict[str, dict[str, list[Any]]]:
    return {method: {key: [] for key in METRIC_KEYS} for method in methods}


def classify_point_status(
    bit_errors: float,
    error_batches: int | None = None,
    min_error_batches: int = 0,
) -> str:
    """Resolved needs enough bit errors and, if required, errors from enough distinct batches."""
    if float(bit_errors) < float(MIN_RESOLVED_BIT_ERRORS):
        return POINT_STATUS_UPPER_BOUND_ONLY
    if error_batches is not None and int(error_batches) < int(min_error_batches):
        return POINT_STATUS_UPPER_BOUND_ONLY
    return POINT_STATUS_RESOLVED


def transmitted_ut_mask(
    directives: ResourceDirectives | None,
    num_ut: int,
) -> list[int]:
    base_active = np.ones(max(num_ut, 1), dtype=bool)
    power_active = np.ones(max(num_ut, 1), dtype=bool)

    if directives is not None and directives.active_ut_mask is not None:
        raw = np.asarray(directives.active_ut_mask, dtype=np.float32).reshape(-1)
        if raw.size < num_ut:
            raise ValueError(
                f"active_ut_mask length {raw.size} is smaller than num_ut={num_ut}."
            )
        base_active = raw[:num_ut] > 0.0
    if directives is not None and directives.per_ut_power is not None:
        raw = np.asarray(directives.per_ut_power, dtype=np.float32).reshape(-1)
        if raw.size < num_ut:
            raise ValueError(
                f"per_ut_power length {raw.size} is smaller than num_ut={num_ut}."
            )
        power_active = raw[:num_ut] > 0.0

    return (base_active & power_active).astype(np.int32).tolist()


def jain_fairness_index(per_ut_values: Sequence[float]) -> float:
    """Jain's index (sum x)^2 / (n * sum x^2); 1 is perfectly even, 1/n is one user only."""
    x = np.asarray(list(per_ut_values), dtype=np.float64)
    denom = x.size * float(np.sum(x * x))
    if x.size == 0 or denom <= 0.0:
        return float("nan")
    return float(np.sum(x) ** 2 / denom)


def per_ut_delivered_bits(
    bits: np.ndarray,
    bits_hat: np.ndarray,
    ut_mask: Sequence[int] | None = None,
) -> list[int]:
    """Correctly delivered information bits per terminal; masked terminals deliver none."""
    correct = np.equal(bits, bits_hat)
    axes = tuple(i for i in range(correct.ndim) if i != 1)
    delivered = correct.sum(axis=axes).astype(np.int64)
    if ut_mask is not None:
        delivered = delivered * (np.asarray(list(ut_mask), dtype=np.int64).reshape(-1) > 0)
    return delivered.tolist()


def channel_error_stats(channel: np.ndarray, channel_hat: np.ndarray) -> dict[str, float]:
    """Squared estimation error, channel energy and element count for one batch.

    Summed rather than averaged so that per-point NMSE (Eq. nmse) is the ratio
    of totals over all batches, not a mean of per-batch ratios.
    """
    h = np.asarray(channel)
    h_hat = np.asarray(channel_hat)
    if h.shape != h_hat.shape:
        raise ValueError(f"channel shape {h.shape} does not match estimate shape {h_hat.shape}.")
    return {
        "sq_error": float(np.sum(np.abs(h_hat - h) ** 2)),
        "channel_energy": float(np.sum(np.abs(h) ** 2)),
        "num_elements": float(h.size),
    }


def nmse_db(sq_error: float, channel_energy: float) -> float:
    if channel_energy <= 0.0:
        return float("nan")
    return float(10.0 * np.log10(max(sq_error, 1e-30) / channel_energy))


def extract_error_stats(
    bits: np.ndarray,
    bits_hat: np.ndarray,
    ut_mask: Sequence[int] | None = None,
) -> dict[str, int]:
    diff = np.not_equal(bits, bits_hat)
    if ut_mask is not None:
        if diff.ndim < 2:
            raise ValueError(
                f"Expected at least 2-D bit tensor with user axis at dim=1, got shape {diff.shape}."
            )
        mask = np.asarray(list(ut_mask), dtype=bool).reshape(-1)
        if mask.size != diff.shape[1]:
            raise ValueError(
                f"ut_mask length {mask.size} does not match user axis size {diff.shape[1]}."
            )
        diff = diff[:, mask, ...]

    block_error_mask = np.any(diff, axis=-1)
    return {
        "bit_errors": int(diff.sum()),
        "total_bits": int(diff.size),
        "block_errors": int(block_error_mask.sum()),
        "total_blocks": int(block_error_mask.size),
    }


def zero_error_upper_bound(total_bits: int, confidence_level: float) -> float:
    alpha = max(1e-12, 1.0 - confidence_level)
    return -math.log(alpha) / max(total_bits, 1)


def ber_upper_confidence_bound(bit_errors: int, total_bits: int, confidence_level: float) -> float:
    if total_bits <= 0:
        return 1.0
    if bit_errors <= 0:
        return zero_error_upper_bound(total_bits, confidence_level)
    p_hat = bit_errors / total_bits
    z = NormalDist().inv_cdf(max(1e-12, min(1.0 - 1e-12, confidence_level)))
    denom = 1.0 + (z * z / total_bits)
    center = p_hat + (z * z / (2.0 * total_bits))
    radius = z * math.sqrt(
        (p_hat * (1.0 - p_hat) / total_bits) + ((z * z) / (4.0 * total_bits * total_bits))
    )
    return min(1.0, (center + radius) / denom)


def mc_stop_reason(
    *,
    num_batches: int,
    total_bits: int,
    total_block_errors: int,
    target_block_errors: int | None,
    total_bit_errors: int,
    target_ber: float | None,
    stop_policy: str,
    confidence_level: float,
    min_batches: int,
    min_total_bits: int,
    error_batches: int | None = None,
    min_error_batches: int = 0,
) -> str | None:
    if num_batches < min_batches or total_bits < min_total_bits:
        return None
    if error_batches is not None and int(error_batches) < int(min_error_batches):
        # Not yet enough independent batches behind the errors: keep sampling.
        return None
    if stop_policy == "sweep":
        if target_block_errors is not None and total_block_errors >= target_block_errors:
            return "target_block_errors"
        return None
    if stop_policy != "threshold":
        raise ValueError(f"Unsupported monte carlo stop policy '{stop_policy}'.")
    if target_block_errors is None and target_ber is None:
        return "min_evidence"
    if target_block_errors is not None and total_block_errors >= target_block_errors:
        return "target_block_errors"
    if target_ber is not None:
        ber_upper = ber_upper_confidence_bound(total_bit_errors, total_bits, confidence_level)
        if ber_upper <= target_ber:
            return "target_ber"
    return None


def append_point_metrics(
    aggregate: dict[str, list[Any]],
    *,
    confidence_level: float,
    point: dict[str, Any],
) -> None:
    aggregate["ber"].append(float(point["ber"]))
    aggregate["ber_upper_confidence"].append(
        ber_upper_confidence_bound(
            int(point["bit_errors"]),
            int(point["total_bits"]),
            confidence_level,
        )
    )
    aggregate["latency_ms"].append(float(point["latency_ms"]))
    aggregate["throughput_bits_per_batch"].append(float(point["throughput_bits_per_batch"]))
    aggregate["energy_joules_per_batch"].append(float(point["energy_joules_per_batch"]))
    aggregate["avg_power_w"].append(float(point["avg_power_w"]))
    aggregate["runtime_sec"].append(float(point["runtime_sec"]))
    aggregate["bit_errors"].append(float(point["bit_errors"]))
    aggregate["total_bits"].append(float(point["total_bits"]))
    aggregate["block_errors"].append(float(point["block_errors"]))
    aggregate["total_blocks"].append(float(point["total_blocks"]))
    aggregate["num_batches"].append(float(point["num_batches"]))
    aggregate["latency_model_ms"].append(float(point.get("latency_model_ms", float("nan"))))
    aggregate["tx_power_w"].append(float(point.get("tx_power_w", float("nan"))))
    aggregate["tx_energy_j_per_batch"].append(float(point.get("tx_energy_j_per_batch", float("nan"))))
    delivered = [float(v) for v in point.get("per_ut_delivered_bits", [])]
    aggregate["per_ut_delivered_bits"].append(delivered)
    aggregate["jain_fairness"].append(jain_fairness_index(delivered) if delivered else float("nan"))
    aggregate["nmse_db"].append(float(point.get("nmse_db", float("nan"))))
    aggregate["err_var_calibration"].append(float(point.get("err_var_calibration", float("nan"))))
    aggregate["error_batches"].append(float(point.get("error_batches", float("nan"))))
    aggregate["stop_reason"].append(str(point.get("stop_reason", "unknown")))
    aggregate["point_status"].append(
        str(point.get("point_status", classify_point_status(float(point["bit_errors"]))))
    )


def run_monte_carlo_point(
    *,
    model: "Model",
    batch_size: int,
    ebno_db: float,
    min_batches: int,
    max_mc_batches: int,
    target_block_errors: int | None,
    target_ber: float | None,
    stop_policy: str,
    confidence_level: float,
    min_total_bits: int,
    include_feedback: bool,
    directives_fn: Callable[[Any], ResourceDirectives | None] | None = None,
) -> dict[str, Any]:
    total_errors = 0
    total_bits = 0
    total_block_errors = 0
    total_blocks = 0
    total_throughput = 0.0
    total_latency_sec = 0.0
    total_energy_joules = 0.0
    num_batches_run = 0

    runtime_start = time.perf_counter()
    final_stop_reason = "max_batches"
    for _ in range(max_mc_batches):
        context = model.prepare_batch_context(
            batch_size=batch_size,
            ebno_db=float(ebno_db),
            include_feedback=include_feedback,
        )
        directives = directives_fn(context) if directives_fn is not None else None
        res = model.run_batch(context, directives=directives, include_details=True)
        num_batches_run += 1

        batch_stats = extract_error_stats(res["bits"], res["bits_hat"])
        total_errors += batch_stats["bit_errors"]
        total_bits += batch_stats["total_bits"]
        total_block_errors += batch_stats["block_errors"]
        total_blocks += batch_stats["total_blocks"]
        total_throughput += max(0.0, batch_stats["total_bits"] - batch_stats["bit_errors"])
        total_latency_sec += float(res.get("runtime_latency_sec", 0.0))
        total_energy_joules += float(res.get("energy_joules", 0.0))

        stop_reason = mc_stop_reason(
            num_batches=num_batches_run,
            total_bits=total_bits,
            total_block_errors=total_block_errors,
            target_block_errors=target_block_errors,
            total_bit_errors=total_errors,
            target_ber=target_ber,
            stop_policy=stop_policy,
            confidence_level=confidence_level,
            min_batches=min_batches,
            min_total_bits=min_total_bits,
        )
        if stop_reason is not None:
            final_stop_reason = stop_reason
            break

    runtime_sec = time.perf_counter() - runtime_start
    safe_batches = max(num_batches_run, 1)
    avg_latency_sec = total_latency_sec / safe_batches
    avg_energy_joules = total_energy_joules / safe_batches
    avg_power_w = avg_energy_joules / max(avg_latency_sec, 1e-12)
    return {
        "ber": (total_errors / total_bits) if total_bits > 0 else 0.0,
        "latency_ms": avg_latency_sec * 1000.0,
        "throughput_bits_per_batch": total_throughput / safe_batches,
        "energy_joules_per_batch": avg_energy_joules,
        "avg_power_w": avg_power_w,
        "runtime_sec": runtime_sec,
        "bit_errors": float(total_errors),
        "total_bits": float(total_bits),
        "block_errors": float(total_block_errors),
        "total_blocks": float(total_blocks),
        "num_batches": float(num_batches_run),
        "stop_reason": final_stop_reason,
        "point_status": classify_point_status(float(total_errors)),
    }


def resolve_kwargs(kwargs_config: dict[str, dict[str, Any]], key: str) -> dict[str, Any]:
    direct = kwargs_config.get(key)
    if isinstance(direct, dict):
        return direct
    return {}
