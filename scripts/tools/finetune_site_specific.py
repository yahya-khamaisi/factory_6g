#!/usr/bin/env python3
"""Site-specific fine-tuning of the neural channel estimator (Phase 2).

1. Generate a held-out ray-traced test set at the bank's *evaluation* positions
   (unless it already exists).
2. Pre-train a *generic* estimator on the statistical channels of the training
   dataset only (Rayleigh, Rician, TR 38.901 UMi): no ray-traced samples.
3. Fine-tune copies of it on N ray-traced samples from the bank's *training*
   positions, for each N in --sizes.
4. Score NMSE on the held-out ray-traced positions for LS, the generic model,
   each fine-tuned model and the locked mixed-data model (v2), and on the
   statistical test split to check that fine-tuning does not forget.

The generic model scored on ray-traced positions is also the out-of-distribution
test: a model trained without the hall, evaluated in the hall.

Run inside the simulation image (data mounted at /app/data):
    python scripts/tools/finetune_site_specific.py --out results/phase2/finetune
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np

logger = logging.getLogger("finetune")


def _nmse_terms(net, reader, h_ls, h_res, cond, batch_size=256):
    if net is None:  # LS: the residual is the whole error
        err = np.sum(np.abs(h_res) ** 2, axis=(1, 2))
    else:
        delta = net.predict(reader.inputs(h_ls, cond), batch_size=batch_size, verbose=0)
        err = np.sum(np.abs(h_res - (delta[..., 0] + 1j * delta[..., 1])) ** 2, axis=(1, 2))
    power = np.sum(np.abs(h_ls + h_res) ** 2, axis=(1, 2))
    return err, power


def score(net, reader, blocks, bands):
    """Pooled NMSE (dB) overall and per Eb/N0 band over (h_ls, h_res, ebno, cond) blocks."""
    tot = {"all": [0.0, 0.0], **{f"{lo:g}-{hi:g}": [0.0, 0.0] for lo, hi in bands}}
    for h_ls, h_res, ebno, cond in blocks:
        err, power = _nmse_terms(net, reader, h_ls, h_res, cond)
        tot["all"][0] += err.sum(); tot["all"][1] += power.sum()
        for lo, hi in bands:
            m = (ebno >= lo) & (ebno < hi)
            tot[f"{lo:g}-{hi:g}"][0] += err[m].sum(); tot[f"{lo:g}-{hi:g}"][1] += power[m].sum()
    return {k: float(10 * np.log10(e / p)) if p > 0 else float("nan") for k, (e, p) in tot.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config/config.json")
    parser.add_argument("--dataset", default="data/channel_train_v4.npz")
    parser.add_argument("--reference", default="models/neural_estimator_v2.keras")
    parser.add_argument("--rt-test", default="results/phase2/datasets/channel_rt_eval.npz")
    parser.add_argument("--rt-test-batches", type=int, default=64)
    parser.add_argument("--out", default="results/phase2/finetune")
    parser.add_argument("--pretrain-epochs", type=int, default=8)
    parser.add_argument("--sizes", default="0,256,1024,4096,16384")
    parser.add_argument("--finetune-steps", type=int, default=400, help="Gradient steps per fine-tune (batch 64), with early stopping.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    import random

    import sionna.phy
    import tensorflow as tf

    random.seed(args.seed); np.random.seed(args.seed); tf.random.set_seed(args.seed); sionna.phy.config.seed = args.seed
    from factory6g.cli.train import _load_flat_config
    from factory6g.training.train_estimator import _DatasetReader, build_model, generate_dataset

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    flat = _load_flat_config(args.config)
    rt_test_path = Path(args.rt_test)
    if not Path(str(rt_test_path) + ".manifest.json").exists():
        rt_test_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Generating the held-out ray-traced test set at evaluation positions ...")
        generate_dataset(flat, str(rt_test_path), num_batches=args.rt_test_batches, ebno_db_range=(0.0, 30.0),
                         channel_types=["raytraced"], seed=args.seed + 1000, batch_size=64, rt_bank_split="eval")

    reader = _DatasetReader(args.dataset)
    rt_id = reader.channel_types.index("raytraced")
    stat_ids = [i for i, c in enumerate(reader.channel_types) if c != "raytraced"]
    bands = [(0, 10), (10, 20), (20, 31)]

    def blocks(split, ids):
        for h_ls, h_res, ebno, cond, ch in reader.iter_split(split):
            m = np.isin(ch, ids)
            if m.any():
                yield h_ls[m], h_res[m], ebno[m], cond[m]

    test_reader = _DatasetReader(str(rt_test_path))

    def rt_test_blocks():
        for split in ("train", "val", "test"):  # the held-out file is all test data
            for h_ls, h_res, ebno, cond, _ in test_reader.iter_split(split):
                yield h_ls, h_res, ebno, cond

    # ---- 1. generic model: statistical channels only
    generic_path = out / "generic_statistical.keras"
    if generic_path.exists():
        generic = tf.keras.models.load_model(generic_path, compile=False)
    else:
        sig = (tf.TensorSpec((reader.num_ofdm_symbols, reader.fft_size, 3), tf.float32),
               tf.TensorSpec((reader.num_ofdm_symbols, reader.fft_size, 2), tf.float32))

        def gen(split):
            def g():
                for h_ls, h_res, _, cond in blocks(split, stat_ids):
                    x, y = reader.inputs(h_ls, cond), reader.to_ri(h_res)
                    for i in range(len(x)):
                        yield x[i], y[i]
            return g

        n_train = sum(len(b[0]) for b in blocks("train", stat_ids))
        n_val = sum(len(b[0]) for b in blocks("val", stat_ids))
        ds = tf.data.Dataset.from_generator(gen("train"), output_signature=sig).shuffle(4096, seed=args.seed).batch(256).repeat().prefetch(2)
        dv = tf.data.Dataset.from_generator(gen("val"), output_signature=sig).batch(256).repeat().prefetch(2)
        generic = build_model(reader.num_ofdm_symbols, reader.fft_size)
        generic.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
        t0 = time.time()
        generic.fit(ds, epochs=args.pretrain_epochs, steps_per_epoch=n_train // 256, validation_data=dv, validation_steps=max(1, n_val // 256),
                    callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)], verbose=2)
        logger.info("Pre-trained the generic model on %d statistical samples in %.0f s.", n_train, time.time() - t0)
        generic.save(generic_path)

    # ---- 2. ray-traced fine-tuning data (training positions), shuffled once
    rt_x, rt_y = [], []
    for h_ls, h_res, _, cond in blocks("train", [rt_id]):
        rt_x.append(reader.inputs(h_ls, cond)); rt_y.append(reader.to_ri(h_res))
    rt_x, rt_y = np.concatenate(rt_x), np.concatenate(rt_y)
    order = np.random.default_rng(args.seed).permutation(len(rt_x))
    rt_x, rt_y = rt_x[order], rt_y[order]
    val_x, val_y = [], []
    for h_ls, h_res, _, cond in blocks("val", [rt_id]):
        val_x.append(reader.inputs(h_ls, cond)); val_y.append(reader.to_ri(h_res))
    val_x, val_y = np.concatenate(val_x), np.concatenate(val_y)
    logger.info("Ray-traced fine-tuning pool: %d training-position samples, %d for validation.", len(rt_x), len(val_x))

    results = {"rt_test": str(rt_test_path), "dataset": args.dataset, "seed": args.seed, "rows": []}
    stat_test = list(blocks("test", stat_ids))
    rt_test = list(rt_test_blocks())

    def record(name, net, n):
        row = {"model": name, "finetune_samples": n, "nmse_rt_eval_db": score(net, test_reader, rt_test, bands),
               "nmse_statistical_test_db": score(net, reader, stat_test, bands)}
        results["rows"].append(row)
        logger.info("%s: %s", name, json.dumps(row))

    record("ls", None, 0)
    if Path(args.reference).exists():
        record("v2_mixed_data", tf.keras.models.load_model(args.reference, compile=False), -1)
    for n in [int(v) for v in args.sizes.split(",")]:
        n = min(n, len(rt_x))
        net = tf.keras.models.clone_model(generic)
        net.set_weights(generic.get_weights())
        if n > 0:
            net.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss="mse")
            epochs = int(min(50, max(3, np.ceil(args.finetune_steps * 64 / n))))
            net.fit(rt_x[:n], rt_y[:n], batch_size=64, epochs=epochs, validation_data=(val_x, val_y), verbose=0,
                    callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)])
            net.save(out / f"finetuned_{n}.keras")
        record("generic" if n == 0 else f"finetuned_{n}", net, n)
    (out / "finetune_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info("Wrote %s", out / "finetune_results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
