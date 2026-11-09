"""The receiver's decode path must match calling the Sionna decoder directly."""

import torch

from factory6g.models.model import Model
from factory6g.sim.config import load_config
from factory6g.sim.rng import seed_all


def test_receiver_decoder_matches_direct_decoder():
    seed_all(7)
    cfg = load_config("config/config.json").system_runtime_config
    m = Model(cfg, estimator_type="ls")
    rx = m._receiver
    for ebno_db in (0.0, 2.0, 6.0):
        ctx = m.prepare_batch_context(batch_size=4, ebno_db=ebno_db, include_feedback=False)
        x_rg, _, _ = m._transmitter.call(4, bits=ctx.source_bits)
        y = m._channel.apply_frequency_response(x_rg, ctx.h_freq) + ctx.data_noise
        h_hat, err_var = rx.estimate_channel(y, ctx.noise_variance)
        x_hat, no_eff = rx.equalize(y, h_hat, err_var, ctx.noise_variance)
        llr = rx.demap(x_hat, no_eff)
        direct = rx._decoder(llr)
        decoded, num_iter = rx.decode(llr)
        assert torch.equal(direct, decoded)
        assert float(num_iter.mean()) == float(rx._num_iter)
