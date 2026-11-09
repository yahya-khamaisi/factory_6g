"""Batch-level statistics from a stage trace (NumPy only).

Each batch is one hall draw, so errors inside a batch are correlated. These
helpers therefore treat the batch, not the bit, as the unit of evidence: the
interval on BER comes from the spread of per-batch BER, and two methods are
compared batch by batch, since every method saw the same channel and noise.
The viewer's JavaScript computes the same quantities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

__all__ = [
    "MIN_RESOLVED_BIT_ERRORS", "Z95", "BatchSeries", "PointSummary", "batch_series", "mean_ci",
    "point_summary", "paired_difference", "delivered_per_terminal", "jain_index",
]

MIN_RESOLVED_BIT_ERRORS = 30  # same rule as factory6g.sim.stages.common
Z95 = 1.959963984540054


def _active_sum(values: list[Any]) -> int:
    return int(sum(v for v in values if v is not None))


@dataclass(frozen=True)
class BatchSeries:
    batch: np.ndarray
    bit_errors: np.ndarray
    bits: np.ndarray
    block_errors: np.ndarray
    blocks: np.ndarray

    @property
    def ber(self) -> np.ndarray:
        return np.divide(self.bit_errors, self.bits, out=np.full(self.bits.shape, np.nan), where=self.bits > 0)

    @property
    def bler(self) -> np.ndarray:
        return np.divide(self.block_errors, self.blocks, out=np.full(self.blocks.shape, np.nan), where=self.blocks > 0)


def batch_series(records: Iterable[dict[str, Any]], method: str, ebno_db: float) -> BatchSeries:
    rows = sorted(
        (r for r in records if r["method"] == method and float(r["ebno_db"]) == float(ebno_db)),
        key=lambda r: r["batch"],
    )
    return BatchSeries(
        batch=np.array([r["batch"] for r in rows], dtype=np.int64),
        bit_errors=np.array([_active_sum(r["bit_errors"]) for r in rows], dtype=np.float64),
        bits=np.array([_active_sum(r["bits"]) for r in rows], dtype=np.float64),
        block_errors=np.array([_active_sum(r["block_errors"]) for r in rows], dtype=np.float64),
        blocks=np.array([_active_sum(r["blocks"]) for r in rows], dtype=np.float64),
    )


def mean_ci(values: np.ndarray) -> tuple[float, float, float]:
    """Mean and normal 95 % interval of per-batch values (NaN when n < 2)."""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(v.mean())
    if v.size < 2:
        return mean, float("nan"), float("nan")
    half = Z95 * float(v.std(ddof=1)) / np.sqrt(v.size)
    return mean, mean - half, mean + half


@dataclass(frozen=True)
class PointSummary:
    batches: int
    bit_errors: int
    bits: int
    block_errors: int
    blocks: int
    ber: float
    ber_lo: float
    ber_hi: float
    bler: float
    resolved: bool
    zero_error_bound: float


def point_summary(series: BatchSeries) -> PointSummary:
    be, bits = int(series.bit_errors.sum()), int(series.bits.sum())
    _, lo, hi = mean_ci(series.ber)
    return PointSummary(
        batches=int(series.batch.size),
        bit_errors=be,
        bits=bits,
        block_errors=int(series.block_errors.sum()),
        blocks=int(series.blocks.sum()),
        ber=be / bits if bits else float("nan"),
        ber_lo=max(0.0, lo) if np.isfinite(lo) else float("nan"),
        ber_hi=hi,
        bler=float(series.block_errors.sum() / series.blocks.sum()) if series.blocks.sum() else float("nan"),
        resolved=be >= MIN_RESOLVED_BIT_ERRORS,
        zero_error_bound=-np.log(0.05) / bits if bits else float("nan"),
    )


def paired_difference(a: BatchSeries, b: BatchSeries) -> tuple[float, float, float, int]:
    """Mean per-batch BLER difference a − b over shared batches, with a 95 % interval."""
    common, ia, ib = np.intersect1d(a.batch, b.batch, return_indices=True)
    mean, lo, hi = mean_ci(a.bler[ia] - b.bler[ib])
    return mean, lo, hi, int(common.size)


def delivered_per_terminal(records: Iterable[dict[str, Any]], method: str, ebno_db: float) -> np.ndarray:
    """Delivered bits per terminal per batch, shape [batches, terminals]; unscheduled terminals deliver 0."""
    rows = sorted(
        (r for r in records if r["method"] == method and float(r["ebno_db"]) == float(ebno_db)),
        key=lambda r: r["batch"],
    )
    return np.array([
        [(bits - errs) if on else 0 for on, bits, errs in zip(r["active"], r["bits"], r["bit_errors"])]
        for r in rows
    ], dtype=np.float64)


def jain_index(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    denom = x.size * float(np.sum(x * x))
    return float(np.sum(x) ** 2 / denom) if denom > 0 else float("nan")
