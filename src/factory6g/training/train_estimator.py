"""Training utilities for the NeuralChannelEstimator.

Usage (inside Docker via docker-compose):

    # Generate training data then train (default output paths):
    docker compose run train-estimator

    # Override paths:
    docker compose run train-estimator \
        --data data/channel_train.npz \
        --output models/neural_estimator.keras \
        --batches 500 --epochs 30 \
        --channel-types rayleigh rician tr38901

How it works:
    1. ``generate_dataset`` runs the simulation (LS estimator only) across
       multiple channel types using a curriculum schedule.  It collects
       triplets of (h_ls, h_residual, ebno_db) where h_residual = h_true - h_ls.
       Learning to predict the *correction* is much easier than predicting the
       full channel, because the corrections are small at high SNR and the
       network only needs to learn where LS is wrong.
    2. ``train`` builds a 3-input-channel Conv2D network (re, im, snr_map),
       trains it to predict h_residual with MSE loss, and saves the model.
    3. At inference time the network output (delta) is added back to h_ls.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import json
import logging
import os
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)

# Legacy SNR normalisation (batch Eb/No over 0-20 dB). Datasets that store a
# per-grid conditioning value use the rule recorded in their manifest instead;
# both are defined in neural_estimator.py so training and inference match.
EBNO_MIN_DB: float = 0.0
EBNO_MAX_DB: float = 20.0


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def generate_dataset(
    config: dict,
    output_path: str,
    num_batches: int = 200,
    ebno_db_range: tuple[float, float] = (0.0, 20.0),
    channel_types: Sequence[str] = ("tr38901",),
    seed: int | None = None,
    batch_size: int | None = None,
    rt_bank_split: str = "train",
) -> None:
    """Run simulation batches and save (h_ls, h_residual, ebno_db) to .npz.

    Curriculum schedule (when multiple channel_types are given):
      - First third of batches:  easiest channel only (channel_types[0])
      - Middle third:            first two channels equally
      - Final third:             all channels equally

    This lets the network learn basic denoising on Rayleigh before tackling
    the harder multipath of TR 38.901.

    Args:
        config: Flat system config dict (same format used by Model).
        output_path: Path to write the .npz dataset.
        num_batches: Total number of simulation batches to collect.
        ebno_db_range: (min, max) Eb/No in dB, sampled uniformly per batch.
        channel_types: Ordered list of channel model names for curriculum.
                       Typical: ["rayleigh", "rician", "tr38901"].
        seed: Seed already applied by the caller, recorded in the manifest so a
              dataset can be traced back to the run that produced it.
        rt_bank_split: bank split for ray-traced samples. Training data uses
              "train"; "eval" builds a held-out test set at unseen positions.
    """
    import torch

    torch.set_grad_enabled(False)
    from factory6g.models.model import Model

    import glob

    channel_types = list(channel_types)
    batch_size = int(batch_size if batch_size is not None else config.get("batch_size", 64))

    stale_chunks = sorted(glob.glob(f"{output_path}.chunk*.npz"))
    if stale_chunks:
        logger.warning(
            "%d chunk file(s) from an earlier run already exist and will be "
            "overwritten or orphaned: %s. Only the chunks listed in the manifest "
            "written by THIS run belong to the dataset; delete the rest.",
            len(stale_chunks),
            ", ".join(stale_chunks),
        )

    from factory6g.components.estimators.neural_estimator import (
        LS_ENERGY_CONDITIONING,
        effective_ebno_db_np,
    )

    h_ls_list: list[np.ndarray] = []
    h_residual_list: list[np.ndarray] = []
    ebno_list: list[float] = []
    cond_list: list[np.ndarray] = []
    channel_list: list[int] = []
    channel_batch_counts = {ch: 0 for ch in channel_types}

    # One Model instance per channel type (re-created only when type changes)
    model_cache: dict[str, Model] = {}

    def _get_model(ch_type: str) -> Model:
        if ch_type not in model_cache:
            cfg = copy.deepcopy(config)
            cfg["channel_model_type"] = ch_type
            # Training data never sees evaluation positions of the ray-traced bank;
            # rt_bank_split="eval" is only for building held-out test sets.
            cfg["rt_bank_split"] = rt_bank_split
            model_cache[ch_type] = Model(cfg, estimator_type="ls")
            logger.info("Created LS model for channel_model_type='%s'.", ch_type)
        return model_cache[ch_type]

    def _curriculum_type(batch_idx: int) -> str:
        """Return channel type according to curriculum stage."""
        n = len(channel_types)
        if n == 1:
            return channel_types[0]
        stage = int(batch_idx / num_batches * n)
        stage = min(stage, n - 1)
        pool = channel_types[: stage + 1]
        return pool[np.random.randint(len(pool))]

    def _flatten(arr: np.ndarray) -> np.ndarray:
        """[batch, rx, tx, streams, ofdm_syms, fft_size] → [N, ofdm_syms, fft_size]."""
        s = arr.shape
        return arr.reshape(-1, s[-2], s[-1])

    # Flush accumulated arrays to disk every FLUSH_EVERY batches to avoid OOM.
    FLUSH_EVERY = 10  # ~20k grids (~0.6 GB) per chunk keeps training memory bounded
    flush_files: list[str] = []
    flush_idx = 0

    def _flush(h_ls_buf, h_res_buf, ebno_buf, channel_buf, cond_buf):
        nonlocal flush_idx
        chunk_path = f"{output_path}.chunk{flush_idx}.npz"
        np.savez_compressed(
            chunk_path,
            h_ls=np.concatenate(h_ls_buf, axis=0),
            h_residual=np.concatenate(h_res_buf, axis=0),
            ebno_db=np.array(ebno_buf, dtype=np.float32),
            # Per-grid network conditioning input (effective Eb/No, dB).
            cond_db=np.concatenate(cond_buf, axis=0),
            channel_id=np.array(channel_buf, dtype=np.int8),
        )
        flush_files.append(chunk_path)
        flush_idx += 1

    logger.info(
        "Generating %d batches | channel types: %s | Eb/No: %.1f–%.1f dB",
        num_batches, channel_types, *ebno_db_range,
    )

    for i in range(num_batches):
        ch_type = _curriculum_type(i)
        ebno_db = float(np.random.uniform(*ebno_db_range))

        model = _get_model(ch_type)
        ctx = model.prepare_batch_context(batch_size, ebno_db, include_feedback=False)
        result = model.run_batch(ctx, include_details=True)

        h_true = result["channel"]      # complex64
        h_ls = result["channel_hat"]    # complex64 (LS estimate)

        h_ls_flat = _flatten(h_ls)
        h_residual_flat = _flatten(h_true) - h_ls_flat  # correction to learn

        h_ls_list.append(h_ls_flat)
        h_residual_list.append(h_residual_flat)
        ebno_list.extend([ebno_db] * h_ls_flat.shape[0])
        # Same rule the estimator applies at inference, computed before flattening
        # so that each grid gets its own terminal's estimate.
        cond_list.append(
            effective_ebno_db_np(h_ls, float(result["err_var_mean"]), ebno_db).reshape(-1)
        )
        channel_list.extend([channel_types.index(ch_type)] * h_ls_flat.shape[0])
        channel_batch_counts[ch_type] += 1

        if (i + 1) % FLUSH_EVERY == 0:
            logger.info("  %d / %d batches done — flushing chunk to disk", i + 1, num_batches)
            _flush(h_ls_list, h_residual_list, ebno_list, channel_list, cond_list)
            h_ls_list.clear()
            h_residual_list.clear()
            ebno_list.clear()
            channel_list.clear()
            cond_list.clear()

    # Flush any remaining batches
    if h_ls_list:
        _flush(h_ls_list, h_residual_list, ebno_list, channel_list, cond_list)

    # Skip the in-memory merge — write a manifest so train() can stream chunks.
    manifest_path = output_path + ".manifest.json"
    # Count total samples from chunk headers (no full load needed)
    total_samples = 0
    for p in flush_files:
        with np.load(p) as c:
            total_samples += c["h_ls"].shape[0]
    manifest = {
        "chunks": flush_files,
        "total_samples": total_samples,
        # Provenance: without these fields a dataset cannot be traced back to
        # the channel models it was drawn from, and a trained estimator cannot
        # be reported against its training distribution.
        "channel_types": channel_types,
        "channel_batch_counts": channel_batch_counts,
        "rt_bank_split": rt_bank_split,
        "num_generation_batches": int(num_batches),
        "batch_size": batch_size,
        "ebno_db_range": [float(ebno_db_range[0]), float(ebno_db_range[1])],
        "conditioning": LS_ENERGY_CONDITIONING,
        "seed": None if seed is None else int(seed),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(
        "Dataset manifest saved to '%s' (%d samples across %d chunk(s); "
        "channels=%s, seed=%s).",
        manifest_path, total_samples, len(flush_files), channel_types, seed,
    )


# ---------------------------------------------------------------------------
# Network definition
# ---------------------------------------------------------------------------

def build_model(num_ofdm_symbols: int, fft_size: int):
    """The residual Conv2D network of the neural estimator (PyTorch)."""
    from factory6g.components.estimators.neural_estimator import NeuralEstimatorNet

    return NeuralEstimatorNet(num_ofdm_symbols, fft_size)


def _predict(net, x: np.ndarray, batch_size: int) -> np.ndarray:
    import torch

    net.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            out.append(net(torch.as_tensor(x[start:start + batch_size])).numpy())
    return np.concatenate(out, axis=0) if out else np.zeros((0,) + x.shape[1:-1] + (2,), np.float32)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _split_of(index: int) -> str:
    """Interleaved 80/10/10 split by sample index, so every curriculum stage and
    every channel model contributes to all three splits."""
    r = index % 10
    if r == 8:
        return "val"
    if r == 9:
        return "test"
    return "train"


class _DatasetReader:
    """Streams a chunked estimator dataset and builds network inputs from it."""

    def __init__(self, dataset_path: str) -> None:
        self.manifest_path = dataset_path + ".manifest.json"
        if not os.path.exists(self.manifest_path):
            raise FileNotFoundError(
                f"'{self.manifest_path}' not found. Run train.py --step generate first."
            )
        with open(self.manifest_path) as f:
            self.manifest = json.load(f)
        self.chunk_paths = self.manifest["chunks"]
        self.channel_types = list(self.manifest.get("channel_types", []))
        with np.load(self.chunk_paths[0]) as first:
            _, self.num_ofdm_symbols, self.fft_size = first["h_ls"].shape
            self.has_channel_id = "channel_id" in first.files
            self.has_cond = "cond_db" in first.files
        from factory6g.components.estimators.neural_estimator import LEGACY_CONDITIONING
        conditioning = self.manifest.get("conditioning") if self.has_cond else None
        self.conditioning = dict(conditioning or LEGACY_CONDITIONING)
        self.cond_min_db, self.cond_max_db = (float(v) for v in self.conditioning["range_db"])
        self.ebno_range = tuple(float(v) for v in self.manifest.get("ebno_db_range", [0.0, 20.0]))
        self.counts = {"train": 0, "val": 0, "test": 0}
        for i in range(int(self.manifest["total_samples"])):
            self.counts[_split_of(i)] += 1

    @staticmethod
    def to_ri(arr: np.ndarray) -> np.ndarray:
        return np.stack([arr.real, arr.imag], axis=-1).astype(np.float32)

    def snr_map(self, cond_db: np.ndarray) -> np.ndarray:
        norm = np.clip((cond_db - self.cond_min_db) / (self.cond_max_db - self.cond_min_db), 0.0, 1.0)
        return np.tile(norm[:, None, None, None], (1, self.num_ofdm_symbols, self.fft_size, 1)).astype(np.float32)

    def inputs(self, h_ls: np.ndarray, cond_db: np.ndarray) -> np.ndarray:
        return np.concatenate([self.to_ri(h_ls), self.snr_map(cond_db)], axis=-1)

    def iter_split(self, split: str):
        """Yield (h_ls, h_residual, ebno_db, cond_db, channel_id) blocks of one split."""
        offset = 0
        for path in self.chunk_paths:
            data = np.load(path)
            n = data["h_ls"].shape[0]
            keep = np.array([_split_of(offset + i) == split for i in range(n)])
            offset += n
            if not keep.any():
                continue
            idx = np.flatnonzero(keep)
            h_ls_all = data["h_ls"]
            h_res_all = data["h_residual"]
            ebno_all = data["ebno_db"]
            cond_all = data["cond_db"] if self.has_cond else ebno_all
            ch_all = data["channel_id"] if self.has_channel_id else np.full(n, -1, np.int8)
            # Yield in slices so only one small block is converted at a time.
            for start in range(0, idx.size, 1024):
                sel = idx[start:start + 1024]
                yield h_ls_all[sel], h_res_all[sel], ebno_all[sel], cond_all[sel], ch_all[sel]
            del h_ls_all, h_res_all

    def channel_name(self, ch_id: int) -> str:
        if ch_id == -2:
            return "all"
        return self.channel_types[ch_id] if 0 <= ch_id < len(self.channel_types) else "unknown"


def _evaluate_on_test(net, reader: _DatasetReader, batch_size: int) -> dict:
    """Unweighted per-grid NMSE of LS and of the refined estimate on the test split.

    Returns three reports: ``test_nmse`` (four equal Eb/No bands over the
    generated range, the format of the thesis table), ``test_nmse_fine`` (5 dB
    Eb/No bands) and ``test_nmse_by_cond`` (bands of each grid's own effective
    SNR), so that an error floor cannot hide inside a wide band.
    """
    lo_db, hi_db = reader.ebno_range
    edges = np.linspace(lo_db, hi_db, 5)
    bands = [(float(edges[k]), float(edges[k + 1]) + (1e-4 if k == 3 else 0.0)) for k in range(4)]
    fine_edges = np.arange(lo_db, hi_db + 5.0, 5.0)
    cond_edges = np.array([-30.0, -10.0, 0.0, 10.0, 20.0, 30.0, 35.0, 45.0])
    acc: dict[tuple[int, int], np.ndarray] = {}
    acc_fine: dict[tuple[int, int], np.ndarray] = {}
    acc_cond: dict[tuple[int, int], np.ndarray] = {}

    def _band_of(value: float, band_edges: np.ndarray) -> int:
        k = int(np.searchsorted(band_edges, value, side="right")) - 1
        return min(max(k, 0), len(band_edges) - 2)

    for h_ls, h_res, ebno, cond, ch in reader.iter_split("test"):
        delta = _predict(net, reader.inputs(h_ls, cond), batch_size)
        delta_c = delta[..., 0] + 1j * delta[..., 1]
        err_ls = np.sum(np.abs(h_res) ** 2, axis=(1, 2))
        err_nn = np.sum(np.abs(h_res - delta_c) ** 2, axis=(1, 2))
        power = np.sum(np.abs(h_ls + h_res) ** 2, axis=(1, 2))
        for i in range(len(ebno)):
            terms = (err_ls[i], err_nn[i], power[i], 1)
            band = next(b for b, (lo, hi) in enumerate(bands) if lo <= ebno[i] < hi)
            for key in ((int(ch[i]), band), (int(ch[i]), -1), (-2, band), (-2, -1)):
                acc.setdefault(key, np.zeros(4))
                acc[key] += terms
            for table, b in ((acc_fine, _band_of(float(ebno[i]), fine_edges)),
                             (acc_cond, _band_of(float(cond[i]), cond_edges))):
                for key in ((int(ch[i]), b), (-2, b)):
                    table.setdefault(key, np.zeros(4))
                    table[key] += terms

    def _row(ch_id: int, band_key: str, band_value, e_ls, e_nn, pw, n) -> dict:
        return {
            "channel": reader.channel_name(ch_id),
            band_key: band_value,
            "num_grids": int(n),
            "nmse_ls_db": float(10 * np.log10(e_ls / pw)),
            "nmse_neural_db": float(10 * np.log10(e_nn / pw)),
        }

    test_report = [
        _row(ch_id, "ebno_band_db", "all" if band == -1 else [bands[band][0], min(bands[band][1], hi_db)], *v)
        for (ch_id, band), v in sorted(acc.items())
    ]
    test_fine = [_row(ch_id, "ebno_band_db", [float(fine_edges[b]), float(fine_edges[b + 1])], *v)
                 for (ch_id, b), v in sorted(acc_fine.items())]
    test_by_cond = [_row(ch_id, "cond_band_db", [float(cond_edges[b]), float(cond_edges[b + 1])], *v)
                    for (ch_id, b), v in sorted(acc_cond.items())]
    for row in test_report:
        logger.info("TEST %s", row)
    for row in test_by_cond:
        logger.info("TEST-COND %s", row)
    return {"test_nmse": test_report, "test_nmse_fine": test_fine, "test_nmse_by_cond": test_by_cond}


def train(
    dataset_path: str,
    model_output_path: str,
    epochs: int = 20,
    batch_size: int = 256,
    seed: int | None = None,
    loss_weighting: str = "none",
) -> dict:
    """Stream chunk files, train the residual network, evaluate it on a held-out
    test split, and save the model with a metadata record.

    Samples are assigned to train (80%), validation (10%) and test (10%) splits
    by index modulo 10. The test split is never used for fitting or early
    stopping; after training, LS and neural NMSE are reported on it per channel
    model and per Eb/No band, and written to ``<model_output_path>.metadata.json``.

    ``loss_weighting="noise"`` weights each grid's squared error by
    1/sigma_n^2, i.e. 10^(Eb/No/10) normalised to mean one over the generated
    range. The LS residual scales with the noise, so plain MSE is dominated by
    low-SNR grids and leaves the network an absolute error floor that exceeds
    the LS error at high SNR; noise weighting makes every operating point count
    in proportion to its own LS error.

    Returns the metadata dictionary.
    """
    if loss_weighting not in {"none", "noise"}:
        raise ValueError(f"loss_weighting must be 'none' or 'noise' (got {loss_weighting!r}).")
    import copy as _copy

    import torch

    torch.set_grad_enabled(True)
    if seed is not None:
        torch.manual_seed(int(seed))
    shuffle_rng = np.random.default_rng(seed)

    reader = _DatasetReader(dataset_path)
    logger.info("Network conditioning: %s", reader.conditioning)
    logger.info("Training on %s samples, grid %dx%d.", reader.counts, reader.num_ofdm_symbols, reader.fft_size)

    lo_gen, hi_gen = reader.ebno_range
    # Mean of 10^(x/10) for x uniform on the generated range, so weights average one.
    if hi_gen > lo_gen:
        weight_norm = (10 ** (hi_gen / 10) - 10 ** (lo_gen / 10)) / ((hi_gen - lo_gen) * np.log(10) / 10)
    else:
        weight_norm = 10 ** (lo_gen / 10)

    def _sample_weight(ebno_db: np.ndarray) -> np.ndarray:
        if loss_weighting == "none":
            return np.ones(len(ebno_db), dtype=np.float32)
        return (10.0 ** (np.asarray(ebno_db, dtype=np.float64) / 10.0) / weight_norm).astype(np.float32)

    def _batches(split: str, shuffle: bool):
        """Mini-batches of (x, y, w); training shuffles within a 4096-sample buffer, like the TensorFlow input pipeline."""
        buf_x, buf_y, buf_w = [], [], []

        def _flush(final: bool):
            nonlocal buf_x, buf_y, buf_w
            x, y, w = np.concatenate(buf_x), np.concatenate(buf_y), np.concatenate(buf_w)
            order = shuffle_rng.permutation(len(x)) if shuffle else np.arange(len(x))
            keep = len(x) if final else (len(x) // batch_size) * batch_size
            for start in range(0, keep, batch_size):
                sel = order[start:start + batch_size]
                yield x[sel], y[sel], w[sel]
            rest = order[keep:]
            buf_x, buf_y, buf_w = [x[rest]], [y[rest]], [w[rest]]

        for h_ls, h_res, ebno, cond, _ in reader.iter_split(split):
            buf_x.append(reader.inputs(h_ls, cond))
            buf_y.append(reader.to_ri(h_res))
            buf_w.append(_sample_weight(ebno))
            if sum(len(b) for b in buf_x) >= 4096:
                yield from _flush(final=False)
        if buf_x and sum(len(b) for b in buf_x):
            yield from _flush(final=True)

    def _loss(net, x, y, w):
        pred = net(torch.as_tensor(x))
        # Keras MSE with sample weights: mean over the real/imag axis, weight, mean over the rest.
        per_pixel = torch.mean((pred - torch.as_tensor(y)) ** 2, dim=-1)
        return torch.mean(per_pixel * torch.as_tensor(w)[:, None, None])

    net = build_model(reader.num_ofdm_symbols, reader.fft_size)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, eps=1e-7)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3, min_lr=1e-6)
    logger.info("Network: %s (%d parameters)", net, sum(p.numel() for p in net.parameters()))
    history: dict[str, list[float]] = {"loss": [], "val_loss": []}
    best_val, best_state, since_best = float("inf"), None, 0
    for epoch in range(epochs):
        net.train()
        losses = []
        for x, y, w in _batches("train", shuffle=True):
            loss = _loss(net, x, y, w)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        net.eval()
        with torch.no_grad():
            val = [(float(_loss(net, x, y, w)), len(x)) for x, y, w in _batches("val", shuffle=False)]
        val_loss = sum(v * n for v, n in val) / max(1, sum(n for _, n in val))
        history["loss"].append(float(np.mean(losses)) if losses else float("nan"))
        history["val_loss"].append(val_loss)
        logger.info("Epoch %d/%d - loss %.6g - val_loss %.6g - lr %.2g", epoch + 1, epochs, history["loss"][-1], val_loss,
                    opt.param_groups[0]["lr"])
        print(f"Epoch {epoch + 1}/{epochs} loss={history['loss'][-1]:.6g} val_loss={val_loss:.6g}", flush=True)
        sched.step(val_loss)
        if val_loss < best_val:
            best_val, best_state, since_best = val_loss, _copy.deepcopy(net.state_dict()), 0
        else:
            since_best += 1
            if since_best >= 7:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    torch.set_grad_enabled(False)
    from factory6g.components.estimators.neural_estimator import torch_weights_path

    weights_path = torch_weights_path(model_output_path)
    torch.save(net.state_dict(), weights_path)
    logger.info("Model saved to '%s'.", weights_path)

    class _History:
        pass

    history_obj = _History()
    history_obj.history = history
    history = history_obj

    metadata = {
        "model_path": model_output_path,
        "dataset_manifest": reader.manifest_path,
        "dataset": {k: reader.manifest.get(k) for k in (
            "total_samples", "channel_types", "channel_batch_counts", "num_generation_batches",
            "batch_size", "ebno_db_range", "seed", "created_at_utc")},
        "split": {"rule": "sample index mod 10: 0-7 train, 8 validation, 9 test", "counts": reader.counts},
        "training": {
            "framework": "pytorch", "weights": weights_path,
            "optimizer": "Adam", "initial_learning_rate": 1e-3, "batch_size": batch_size,
            "loss_weighting": loss_weighting,
            "max_epochs": epochs, "epochs_run": len(history.history.get("loss", [])),
            "best_val_loss": float(min(history.history.get("val_loss", [float("nan")]))),
            "seed": seed,
        },
        # Read back by NeuralChannelEstimator so inference uses the same input rule.
        "conditioning": reader.conditioning,
        # Unweighted per-grid NMSE: thesis bands, 5 dB Eb/No bands, effective-SNR bands.
        **_evaluate_on_test(net, reader, batch_size),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    with open(model_output_path + ".metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    return metadata


def evaluate(dataset_path: str, model_path: str, batch_size: int = 256) -> dict:
    """Re-score an existing model on the dataset's test split and refresh its metadata.

    Only the three NMSE reports are replaced; the training record is kept. Used
    for weights trained before the fine-band reports existed.
    """
    from factory6g.components.estimators.neural_estimator import load_neural_estimator

    reader = _DatasetReader(dataset_path)
    net = load_neural_estimator(model_path, reader.num_ofdm_symbols, reader.fft_size)
    meta_path = model_path + ".metadata.json"
    metadata = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            metadata = json.load(f)
    metadata.update(_evaluate_on_test(net, reader, batch_size))
    metadata["evaluated_at_utc"] = datetime.now(timezone.utc).isoformat()
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    return metadata
