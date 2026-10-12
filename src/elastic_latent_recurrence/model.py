"""Tiny next-token model that uses the elastic latent recurrence block."""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from elastic_latent_recurrence.ops import (
    apply_experts,
    condition_embedding,
    empty_kv_cache,
    entry_z,
    exit_state_update,
    f_for_target_p1,
    first_loop_inputs,
    init_beta_params,
    init_entry_params,
    init_modulation_params,
    init_moe_params,
    last_loop_attention,
    linear,
    load_balance_aux,
    modulate_input,
    modulated_layer,
    rms_norm,
    route_tokens,
    sample_loop_counts,
    write_kv,
    zero_linear,
)

try:
    import flax.linen as nn

    HAS_FLAX = True
except ImportError:  # pragma: no cover
    nn = None
    HAS_FLAX = False

try:
    import optax

    HAS_OPTAX = True
except ImportError:  # pragma: no cover
    optax = None
    HAS_OPTAX = False


def _init_matrix(key, shape, scale, dtype):
    if HAS_FLAX:
        return nn.initializers.normal(stddev=scale)(key, shape, dtype)
    return jax.random.normal(key, shape, dtype) * scale


def _init_linear(key, in_dim, out_dim, dtype, scale=0.02):
    return {
        "W": _init_matrix(key, (out_dim, in_dim), scale, dtype),
        "b": jnp.zeros((out_dim,), dtype=dtype),
    }


@dataclass(frozen=True)
class ELRConfig:
    vocab_size: int = 32
    d_model: int = 32
    seq_len: int = 16
    max_loops: int = 4
    n_heads: int = 4
    moe_experts: int = 4
    rank: int = 4
    cond_dim: int = 32
    sin_dim: int = 32
    mlp_hidden: int = 32
    target_p1: float = 1.0 / 3.0

    @property
    def head_dim(self):
        return self.d_model // self.n_heads

    @property
    def forced_fraction(self):
        return f_for_target_p1(self.max_loops, self.target_p1)


def init_params(key, cfg: ELRConfig | None = None):
    cfg = cfg or ELRConfig()
    dtype = jnp.float32
    d = cfg.d_model
    keys = jax.random.split(key, 16)
    return {
        "embed": _init_matrix(keys[0], (cfg.vocab_size, d), 0.02, dtype),
        "input": {
            "fc1": _init_linear(keys[1], d, cfg.mlp_hidden, dtype),
            "fc2": _init_linear(keys[2], cfg.mlp_hidden, d, dtype),
        },
        "cond": {
            "fc1": _init_linear(keys[3], cfg.sin_dim * 2, cfg.cond_dim, dtype),
            "fc2": _init_linear(keys[4], cfg.cond_dim, cfg.cond_dim, dtype),
        },
        "beta": init_beta_params(d, dtype=dtype),
        "entry": init_entry_params(cfg.cond_dim, d, dtype=dtype),
        "g_exit": zero_linear(cfg.cond_dim, d, dtype=dtype),
        "attn": {
            "mod": init_modulation_params(keys[5], d, cfg.rank, cfg.cond_dim, dtype=dtype),
            "W_q": _init_linear(keys[6], d, d, dtype),
            "W_k": _init_linear(keys[7], d, d, dtype),
            "W_v": _init_linear(keys[8], d, d, dtype),
            "W_o": _init_linear(keys[9], d, d, dtype),
        },
        "moe": init_moe_params(keys[10], d, cfg.moe_experts, cfg.mlp_hidden, dtype=dtype),
        "output": {
            "mod": init_modulation_params(keys[11], d, cfg.rank, cfg.cond_dim, dtype=dtype),
            "fc1": _init_linear(keys[12], d, cfg.mlp_hidden, dtype),
            "fc2": _init_linear(keys[13], cfg.mlp_hidden, d, dtype),
        },
        "lm_head": _init_linear(keys[14], d, cfg.vocab_size, dtype, scale=0.02),
    }


def _mlp(x, params):
    h = jax.nn.gelu(linear(x, params["fc1"]))
    return linear(h, params["fc2"])


def _input_block(x, params):
    """Unconditioned input block. No loop embedding."""
    return x + _mlp(rms_norm(x), params)


def _condition(params, tau, kappa, sin_dim):
    return _mlp(condition_embedding(tau, kappa, sin_dim), params)


def _split_heads(x, n_heads):
    batch, width = x.shape
    head_dim = width // n_heads
    return x.reshape(batch, n_heads, head_dim)


def _attention_sublayer(h, c, cache_k, cache_v, t, params, n_heads):
    """Modulated attention. The current token's loop-i KV is written, then read."""
    x_hat, omega = modulate_input(h, c, params["mod"])
    q = _split_heads(linear(x_hat, params["W_q"]), n_heads)
    k = _split_heads(linear(x_hat, params["W_k"]), n_heads)
    v = _split_heads(linear(x_hat, params["W_v"]), n_heads)
    cache_k = write_kv(cache_k, t, k)
    cache_v = write_kv(cache_v, t, v)
    attn = last_loop_attention(q, cache_k, cache_v, t)
    merged = attn.reshape(h.shape[0], -1)
    delta = linear(merged, params["W_o"])
    return h + (1.0 + omega) * delta, cache_k, cache_v


def forward(params, tokens, loop_counts, cfg: ELRConfig | None = None):
    """Token-outer, loop-inner recurrence with last-loop KV and route replay.

    Earlier tokens finish every loop before the next token starts, so their cache
    slot holds final-loop KV. The current token overwrites its own slot each loop
    and attends to that loop-i KV plus earlier tokens' last write.
    """
    cfg = cfg or ELRConfig()
    dtype = jnp.float32
    k_max = cfg.max_loops
    n_heads = cfg.n_heads
    n_experts = cfg.moe_experts
    sin_dim = cfg.sin_dim

    x = params["embed"][tokens]
    h_in = _input_block(x, params["input"])
    batch, seq_len, width = h_in.shape
    s = jnp.zeros((batch, seq_len, width), dtype=dtype)
    cache_k, cache_v = empty_kv_cache(
        batch, seq_len, n_heads, cfg.head_dim, k_max, dtype=dtype
    )
    route_idx = jnp.zeros((batch, seq_len), dtype=jnp.int32)
    route_w = jnp.zeros((batch, seq_len), dtype=dtype)
    route_probs = jnp.zeros((batch, seq_len, n_experts), dtype=dtype)
    loop_counts_f = loop_counts.astype(dtype)
    kappa = loop_counts_f / dtype(k_max)

    for t in range(seq_len):
        h_t = h_in[:, t, :]
        for i in range(1, k_max + 1):
            active = loop_counts >= i
            tau = dtype(i) / loop_counts_f
            c = _condition(params["cond"], tau, kappa, sin_dim)
            prev_k = cache_k[:, t]
            prev_v = cache_v[:, t]
            if i == 1:
                s_tilde, z = first_loop_inputs(h_t)
                idx, weight, probs = route_tokens(params["moe"], z)
                route_idx = route_idx.at[:, t].set(idx)
                route_w = route_w.at[:, t].set(weight)
                route_probs = route_probs.at[:, t].set(probs)
                s_prev = jnp.zeros((batch, width), dtype=dtype)
            else:
                # Carried state, unmodulated. The gate reads this, not z.
                s_tilde = s[:, t, :]
                z = entry_z(s_tilde, h_t, c, params["entry"])
                idx = route_idx[:, t]
                weight = route_w[:, t]
                s_prev = s[:, t, :]
            h, cache_k, cache_v = _attention_sublayer(
                z, c, cache_k, cache_v, t, params["attn"], n_heads
            )
            cache_k = cache_k.at[:, t].set(
                jnp.where(active[:, None, None], cache_k[:, t], prev_k)
            )
            cache_v = cache_v.at[:, t].set(
                jnp.where(active[:, None, None], cache_v[:, t], prev_v)
            )
            # Experts run on this loop's z. Routes from loop 1 are replayed.
            moe_out = apply_experts(params["moe"], z, idx, weight)
            r = h + moe_out
            exit_params = {"beta": params["beta"], "g_exit": params["g_exit"]}
            s_new = exit_state_update(s_prev, s_tilde, r, c, i, exit_params)
            s = s.at[:, t, :].set(jnp.where(active[:, None], s_new, s[:, t, :]))

    tau_out = jnp.ones((batch,), dtype=dtype)
    c_out = _condition(params["cond"], tau_out, kappa, sin_dim)
    c_out = jnp.broadcast_to(c_out[:, None, :], (batch, seq_len, cfg.cond_dim))
    y = modulated_layer(
        s,
        c_out,
        lambda x_hat: _mlp(x_hat, params["output"]),
        params["output"]["mod"],
    )
    logits = linear(y, params["lm_head"])
    aux1 = load_balance_aux(
        route_probs.reshape(batch * seq_len, n_experts),
        route_idx.reshape(batch * seq_len),
        n_experts,
    )
    # One loop-1 aux, multiplied by the realized loop count (batch mean of k).
    aux = aux1 * jnp.mean(loop_counts_f)
    return logits, aux


def cross_entropy(logits, labels):
    if HAS_OPTAX:
        return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, labels))
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    nll = -jnp.take_along_axis(log_probs, labels[..., None], axis=-1)[..., 0]
    return jnp.mean(nll)


def loss_fn(params, tokens, loop_counts, cfg):
    logits, aux = forward(params, tokens, loop_counts, cfg)
    ce = cross_entropy(logits[:, :-1, :], tokens[:, 1:])
    return ce + aux, {"ce": ce, "aux": aux}


def _manual_adam_init(params):
    zeros = jax.tree.map(jnp.zeros_like, params)
    return {"m": zeros, "v": zeros, "t": jnp.array(0, dtype=jnp.int32)}


def _manual_adam_update(grads, state, lr, b1=0.9, b2=0.999, eps=1e-8):
    t = state["t"] + 1
    m = jax.tree.map(lambda m_, g: b1 * m_ + (1.0 - b1) * g, state["m"], grads)
    v = jax.tree.map(lambda v_, g: b2 * v_ + (1.0 - b2) * (g * g), state["v"], grads)
    mhat = jax.tree.map(lambda m_: m_ / (1.0 - b1**t), m)
    vhat = jax.tree.map(lambda v_: v_ / (1.0 - b2**t), v)
    updates = jax.tree.map(lambda m_, v_: m_ / (jnp.sqrt(v_) + eps), mhat, vhat)
    return updates, {"m": m, "v": v, "t": t}


def train_smoke(steps=30, batch_size=8, seed=0, lr=1e-3, cfg: ELRConfig | None = None):
    """30 CPU Adam steps on a fixed synthetic token batch. Not a paper benchmark."""
    cfg = cfg or ELRConfig()
    key = jax.random.PRNGKey(seed)
    key_init, key_tok, key_k = jax.random.split(key, 3)
    params = init_params(key_init, cfg)
    tokens = jax.random.randint(key_tok, (batch_size, cfg.seq_len), 0, cfg.vocab_size)
    loop_counts = sample_loop_counts(key_k, batch_size, cfg.max_loops, cfg.forced_fraction)
    optimizer_name = "optax" if HAS_OPTAX else "manual"
    if HAS_OPTAX:
        tx = optax.adam(lr)
        opt_state = tx.init(params)

        def _apply(params, opt_state, grads):
            updates, opt_state = tx.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state

    else:
        opt_state = _manual_adam_init(params)

        def _apply(params, opt_state, grads):
            updates, opt_state = _manual_adam_update(grads, opt_state, lr)
            params = jax.tree.map(lambda p, u: p - lr * u, params, updates)
            return params, opt_state

    @jax.jit
    def _step(params, opt_state, tokens, loop_counts):
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, tokens, loop_counts, cfg
        )
        params, opt_state = _apply(params, opt_state, grads)
        return params, opt_state, loss, aux

    loss0, _ = loss_fn(params, tokens, loop_counts, cfg)
    loss0 = jax.block_until_ready(loss0)
    for _ in range(steps):
        params, opt_state, _, _ = _step(params, opt_state, tokens, loop_counts)
    params = jax.block_until_ready(params)
    loss30, _ = loss_fn(params, tokens, loop_counts, cfg)
    loss30 = jax.block_until_ready(loss30)
    return {
        "loss_step_0": float(loss0),
        "loss_step_30": float(loss30),
        "optimizer": optimizer_name,
        "used_flax": HAS_FLAX,
    }


def main():
    result = train_smoke()
    print(
        f"loss_step_0={result['loss_step_0']:.8f} "
        f"loss_step_30={result['loss_step_30']:.8f} "
        f"optimizer={result['optimizer']} flax={result['used_flax']}"
    )


if __name__ == "__main__":
    main()
