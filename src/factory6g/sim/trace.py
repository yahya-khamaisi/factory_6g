"""Per-batch trace of a stage run, for the run viewer and batch-level statistics.

The stage loops aggregate errors over every batch of an Eb/N0 point, which is
enough for a BER curve but not for batch-level intervals, paired tests between
methods, or replaying what each scheduler did. When ``simulation.trace`` is on,
each stage also writes ``trace_v1.jsonl.gz`` next to ``stage_results_v2.json``:

* one ``header`` line describing the stage;
* per batch and Eb/N0 point, one ``context`` line with the shared large-scale
  state (terminal positions, LOS flags, gains), when the channel model has one;
* per batch, Eb/N0 point and method, one ``method`` line with per-terminal
  activity, power, bit and block errors, plus receiver diagnostics.

The recorder only reads arrays the stage already holds. It draws no random
numbers and changes no state, so a traced run gives the same stage results as
an untraced one. Records are buffered per batch and written when the stage
saves its checkpoint, so a resumed run drops the records of the batch it
repeats instead of duplicating them.
"""

from __future__ import annotations

import gzip
import json
import math
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

TRACE_FORMAT_VERSION = 1
TRACE_FILENAME = "trace_v1.jsonl.gz"


def _round_list(values: Any, digits: int) -> list[Any]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    return [None if not math.isfinite(v) else round(float(v), digits) for v in arr]


def per_ut_error_stats(bits: np.ndarray, bits_hat: np.ndarray) -> dict[str, list[int]]:
    """Bit and block (codeword) errors per terminal; the terminal axis is dim 1."""
    diff = np.not_equal(np.asarray(bits), np.asarray(bits_hat))
    if diff.ndim < 2:
        raise ValueError(f"Expected a bit tensor with the terminal axis at dim 1, got shape {diff.shape}.")
    other_axes = tuple(i for i in range(diff.ndim) if i != 1)
    block_errors = np.any(diff, axis=-1)
    block_axes = tuple(i for i in range(block_errors.ndim) if i != 1)
    num_ut = diff.shape[1]
    return {
        "bit_errors": diff.sum(axis=other_axes).astype(np.int64).tolist(),
        "bits": [int(diff.size // num_ut)] * num_ut,
        "block_errors": block_errors.sum(axis=block_axes).astype(np.int64).tolist(),
        "blocks": [int(block_errors.size // num_ut)] * num_ut,
    }


class NullTraceRecorder:
    """Stand-in used when tracing is off; every call is a no-op."""

    enabled = False
    path: Path | None = None

    def record_context(self, *args: Any, **kwargs: Any) -> None:
        return None

    def record_method(self, *args: Any, **kwargs: Any) -> None:
        return None

    def flush(self) -> None:
        return None


class TraceRecorder:
    enabled = True

    def __init__(
        self,
        stage_dir: Path,
        *,
        stage: str,
        meta: dict[str, Any],
        resume_from_batch: int | None = None,
    ) -> None:
        self.path = Path(stage_dir) / TRACE_FILENAME
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._pending: list[str] = []
        if resume_from_batch is not None and self.path.exists():
            self._truncate_from(int(resume_from_batch))
        else:
            header = {"type": "header", "version": TRACE_FORMAT_VERSION, "stage": stage, "meta": meta}
            with gzip.open(self.path, "wt", encoding="utf-8") as handle:
                handle.write(json.dumps(header, separators=(",", ":")) + "\n")

    def _truncate_from(self, batch_index: int) -> None:
        kept = [
            line
            for line in _iter_lines(self.path)
            if json.loads(line).get("batch", -1) < batch_index
        ]
        with gzip.open(self.path, "wt", encoding="utf-8") as handle:
            for line in kept:
                handle.write(line + "\n")

    def _add(self, record: dict[str, Any]) -> None:
        self._pending.append(json.dumps(record, separators=(",", ":")))

    def record_context(self, batch_index: int, ebno_db: float, large_scale: dict[str, Any] | None) -> None:
        record: dict[str, Any] = {"type": "context", "batch": int(batch_index), "ebno_db": float(ebno_db)}
        if large_scale:
            record["positions"] = [_round_list(p, 3) for p in np.asarray(large_scale["positions"])]
            record["los"] = [bool(v) for v in np.asarray(large_scale["los"]).reshape(-1)]
            record["gain_db"] = _round_list(large_scale["gain_db"], 3)
        self._add(record)

    def record_method(
        self,
        batch_index: int,
        ebno_db: float,
        method: str,
        *,
        result: dict[str, Any],
        active_mask: Sequence[int],
        per_ut_power: Sequence[float] | None,
        runtime_sec: float,
    ) -> None:
        stats = per_ut_error_stats(result["bits"], result["bits_hat"])
        active = [int(v) > 0 for v in active_mask]
        # Inactive terminals carry no traffic, so their error counts are not evidence.
        masked = {key: [v if on else None for v, on in zip(vals, active)] for key, vals in stats.items()}
        if per_ut_power is None:
            power = [1.0 if on else 0.0 for on in active]
        else:
            power = _round_list(per_ut_power, 4)
        record: dict[str, Any] = {
            "type": "method",
            "batch": int(batch_index),
            "ebno_db": float(ebno_db),
            "method": str(method),
            "active": active,
            "power": power,
            **masked,
            "runtime_sec": round(float(runtime_sec), 6),
        }
        if "err_var_mean" in result:
            record["err_var_mean"] = _round_list([result["err_var_mean"]], 6)[0]
        if "decoder_iterations" in result:
            record["decoder_iterations_mean"] = round(float(np.mean(result["decoder_iterations"])), 3)
        self._add(record)

    def flush(self) -> None:
        if not self._pending:
            return
        # Appending a gzip member keeps earlier batches intact if the run stops.
        with gzip.open(self.path, "at", encoding="utf-8") as handle:
            for line in self._pending:
                handle.write(line + "\n")
        self._pending.clear()


def make_trace_recorder(
    enabled: bool,
    stage_dir: Path | None,
    *,
    stage: str,
    meta: dict[str, Any],
    resume_from_batch: int | None = None,
) -> TraceRecorder | NullTraceRecorder:
    if not enabled or stage_dir is None:
        return NullTraceRecorder()
    return TraceRecorder(stage_dir, stage=stage, meta=meta, resume_from_batch=resume_from_batch)


def trace_meta(config: Any, *, methods: Sequence[str]) -> dict[str, Any]:
    """Run facts the viewer needs to draw the hall and label the trace."""
    system = config.system_runtime_config
    return {
        "methods": list(methods),
        "ebno_db_range": [float(v) for v in config.monte_carlo.ebno_db_range],
        "num_ut": int(system.get("num_ut", 0)),
        "batch_size": int(config.monte_carlo.batch_size),
        "seed": int(config.simulation.seed),
        "channel_model_type": str(system.get("channel_model_type", "")),
        "large_scale_model": str(system.get("large_scale_model", "hall")),
        "rt_channel_bank": str(system.get("rt_channel_bank", "data/rt_channel_bank.npz")),
        "rt_bank_split": str(system.get("rt_bank_split", "eval")),
        "num_active_users": int(getattr(config.resource_managers, "num_active_users", 0)),
    }


def _iter_lines(path: Path) -> Iterator[str]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield line


def read_trace(path: Path) -> dict[str, Any]:
    """Load a trace file into its header, context records and method records."""
    header: dict[str, Any] | None = None
    contexts: list[dict[str, Any]] = []
    methods: list[dict[str, Any]] = []
    for line in _iter_lines(Path(path)):
        record = json.loads(line)
        kind = record.get("type")
        if kind == "header":
            header = record
        elif kind == "context":
            contexts.append(record)
        elif kind == "method":
            methods.append(record)
    if header is None:
        raise ValueError(f"Trace {path} has no header line.")
    return {"header": header, "contexts": contexts, "methods": methods}
