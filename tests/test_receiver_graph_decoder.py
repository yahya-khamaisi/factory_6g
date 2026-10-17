"""The graph-mode LDPC decoder must match eager decoding exactly."""

import tensorflow as tf

from factory6g.models.model import Model
from factory6g.sim.config import load_config


def test_graph_decoder_matches_eager_decoder():
    tf.random.set_seed(7)
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
        eager = rx._decoder(llr)
        graph = rx._decode_graph(llr)
        assert bool(tf.reduce_all(tf.equal(eager[0], graph[0])))
        assert bool(tf.reduce_all(tf.equal(eager[-1], graph[-1])))
