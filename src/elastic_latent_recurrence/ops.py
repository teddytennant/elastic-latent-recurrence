"""Mechanisms from the elastic latent recurrence recipe (Section 3)."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp

# sigmoid(0) * softplus(0) = 0.5 * log(2)
BETA_ZERO = 0.5 * math.log(2.0)
SIN_BASE = 10000.0


def linear(x, params):
    """Affine map. params['W'] is (out, in), so W @ x for a column vector."""
    return jnp.einsum("...i,oi->...o", x, params["W"]) + params["b"]


def zero_linear(in_dim, out_dim, dtype=jnp.float32):
    return {
        "W": jnp.zeros((out_dim, in_dim), dtype=dtype),
        "b": jnp.zeros((out_dim,), dtype=dtype),
    }


def rms_norm(x, eps=1e-6):
    """Parameter-free RMSNorm. Affine shift and scale come from conditioning."""
    ms = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
    return x * jnp.reciprocal(jnp.sqrt(ms + eps))


def init_beta_params(d, dtype=jnp.float32):
    """W_beta = 0, b_beta = 0, gamma = 0. Shared across loops."""
    return {
        "W": jnp.zeros((d, d), dtype=dtype),
        "b": jnp.zeros((d,), dtype=dtype),
        "gamma": jnp.zeros((d,), dtype=dtype),
    }


def beta(s_tilde, params):
    """Per-token, per-channel adaptive step size.

    beta = sigmoid(W_beta @ s_tilde + b_beta) * softplus(gamma)
    W_beta is (d, d). The gate always reads unmodulated s_tilde, never z.
    """
    logits = jnp.einsum("...d,od->...o", s_tilde, params["W"]) + params["b"]
    return jax.nn.sigmoid(logits) * jax.nn.softplus(params["gamma"])


def adaptive_delta_step(s_prev, s_tilde, r, params):
    """Plain adaptive-delta update: s + beta * (r - s), with s_0 = 0 at the start."""
    return s_prev + beta(s_tilde, params) * (r - s_prev)


def p1_marginal(K, f):
    """Realized one-loop marginal after a fraction f is forced to k=1.

    p_1 = 1/K + (1 - 1/K) * f
    """
    K = float(K)
    f = float(f)
    return 1.0 / K + (1.0 - 1.0 / K) * f


def mean_loop_count(K, f):
    """Mean loop count. Forced examples contribute 1; the rest are uniform on 1..K.

    k_bar = f + (1 - f) * (K + 1) / 2
    """
    K = float(K)
    f = float(f)
    return f + (1.0 - f) * (K + 1.0) / 2.0


def f_for_target_p1(K, p1=1.0 / 3.0):
    """Solve p_1 = 1/K + (1 - 1/K) * f for the forced fraction f.

    f = (K * p1 - 1) / (K - 1)
    Requires p1 >= 1/K so that f is nonnegative.
    """
    K = float(K)
    p1 = float(p1)
    return (K * p1 - 1.0) / (K - 1.0)


def sample_loop_counts(key, batch_size, K, f):
    """Mixed depth. A Bernoulli(f) split is forced to k=1 (rank-override stand-in).

    The rest draw k uniformly from 1..K. Expected one-loop rate is p1_marginal.
    """
    key_f, key_u = jax.random.split(key)
    forced = jax.random.bernoulli(key_f, float(f), (batch_size,))
    uniform = jax.random.randint(key_u, (batch_size,), 1, int(K) + 1)
    return jnp.where(forced, jnp.int32(1), uniform).astype(jnp.int32)


def sinusoidal_embedding(t, dim, base=SIN_BASE):
    """Transformer-style sin/cos embedding, base 10000, interleaved."""
    if dim % 2 != 0:
        raise ValueError("sinusoidal dim must be even")
    t = jnp.asarray(t)
    dtype = t.dtype
    half = dim // 2
    inv_freq = jnp.exp(
        jnp.arange(half, dtype=dtype) * (-jnp.log(jnp.asarray(base, dtype=dtype)) / dim)
    )
    angles = t[..., None] * inv_freq
    # (..., half, 2) -> (..., dim) interleaves sin and cos.
    return jnp.stack([jnp.sin(angles), jnp.cos(angles)], axis=-1).reshape(t.shape + (dim,))


def progress_and_budget(loop_index, loop_count, max_loops):
    """tau = i/k, kappa = k/K."""
    tau = loop_index / loop_count
    kappa = loop_count / max_loops
    return tau, kappa


def output_progress_and_budget(loop_count, max_loops):
    """Output block uses (tau, kappa) = (1, k/K)."""
    kappa = loop_count / max_loops
    if jnp.isscalar(loop_count) or (
        isinstance(loop_count, (int, float)) and not hasattr(loop_count, "shape")
    ):
        return 1.0, kappa
    return jnp.ones_like(jnp.asarray(loop_count, dtype=jnp.float32)), kappa


def condition_embedding(tau, kappa, sin_dim):
    """concat(phi(tau), phi(kappa))."""
    return jnp.concatenate(
        [sinusoidal_embedding(tau, sin_dim), sinusoidal_embedding(kappa, sin_dim)],
        axis=-1,
    )


def init_modulation_params(key, d, rank, cond_dim, dtype=jnp.float32, scale=0.02):
    """AdaLN + low-rank heads are zero so modulation starts as the identity.

    A is (d, r), B is (r, d). g is r-dimensional and multiplies B @ x_tilde.
    """
    k_a, k_b = jax.random.split(key)
    return {
        "lambda": zero_linear(cond_dim, d, dtype=dtype),
        "mu": zero_linear(cond_dim, d, dtype=dtype),
        "omega": zero_linear(cond_dim, d, dtype=dtype),
        "g": zero_linear(cond_dim, rank, dtype=dtype),
        "A": jax.random.normal(k_a, (d, rank), dtype=dtype) * scale,
        "B": jax.random.normal(k_b, (rank, d), dtype=dtype) * scale,
    }


def modulate_input(x, c, params):
    """Return x_hat and omega(c) from the AdaLN + low-rank map."""
    lam = linear(c, params["lambda"])
    mu = linear(c, params["mu"])
    omega = linear(c, params["omega"])
    g = linear(c, params["g"])
    x_tilde = (1.0 + lam) * rms_norm(x) + mu
    bx = jnp.einsum("rd,...d->...r", params["B"], x_tilde)
    low = jnp.einsum("dr,...r->...d", params["A"], g * bx)
    x_hat = x_tilde + low
    return x_hat, omega


def modulated_layer(x, c, f, params):
    """Per-layer modulation of f.

    x_tilde = (1 + lambda(c)) * RMSNorm(x) + mu(c)
    x_hat   = x_tilde + A @ (g(c) * (B @ x_tilde))
    y       = x + (1 + omega(c)) * f(x_hat)
    """
    x_hat, omega = modulate_input(x, c, params)
    return x + (1.0 + omega) * f(x_hat)


def first_loop_inputs(h_in):
    """First loop matches the one-loop checkpoint: s_tilde_1 = z_1 = 2 * h_in."""
    s_tilde = 2.0 * h_in
    return s_tilde, s_tilde


def entry_z(s_tilde, h_in, c, params):
    """Entry map for loops i >= 2. Heads lambda_e, mu_e, eta are zero-initialized.

    z = (1 + lambda_e(c)) * RMSNorm(s_tilde) + mu_e(c) + eta(c) * h_in
    """
    lam = linear(c, params["lambda_e"])
    mu = linear(c, params["mu_e"])
    eta = linear(c, params["eta"])
    return (1.0 + lam) * rms_norm(s_tilde) + mu + eta * h_in


def init_entry_params(cond_dim, d, dtype=jnp.float32):
    return {
        "lambda_e": zero_linear(cond_dim, d, dtype=dtype),
        "mu_e": zero_linear(cond_dim, d, dtype=dtype),
        "eta": zero_linear(cond_dim, d, dtype=dtype),
    }


def init_exit_params(d, cond_dim, dtype=jnp.float32):
    return {
        "beta": init_beta_params(d, dtype=dtype),
        "g_exit": zero_linear(cond_dim, d, dtype=dtype),
    }


def exit_state_update(s_prev, s_tilde, r, c, loop_index, params):
    """Exit update. g_exit is a zero-initialized linear head of c.

    pi_1 = 1 and pi_{i>1} = 0, so zero-init g_exit reproduces adaptive-delta on
    loop 1 and leaves the state unchanged on later loops:
        s = s_prev + (pi + g_exit(c)) * beta(s_tilde) * (r - s_prev)
    beta is computed from unmodulated s_tilde.
    """
    b = beta(s_tilde, params["beta"])
    g = linear(c, params["g_exit"])
    pi = jnp.where(jnp.asarray(loop_index) == 1, jnp.asarray(1.0, dtype=b.dtype), jnp.asarray(0.0, dtype=b.dtype))
    return s_prev + (pi + g) * b * (r - s_prev)


def init_moe_params(key, d, n_experts, hidden, dtype=jnp.float32, scale=0.02):
    k_r, k_e1, k_e2 = jax.random.split(key, 3)
    return {
        "router": {
            "W": jax.random.normal(k_r, (n_experts, d), dtype=dtype) * scale,
            "b": jnp.zeros((n_experts,), dtype=dtype),
        },
        "W1": jax.random.normal(k_e1, (n_experts, hidden, d), dtype=dtype) * scale,
        "b1": jnp.zeros((n_experts, hidden), dtype=dtype),
        "W2": jax.random.normal(k_e2, (n_experts, d, hidden), dtype=dtype) * scale,
        "b2": jnp.zeros((n_experts, d), dtype=dtype),
    }


def route_tokens(params, z):
    """Top-1 router. Returns indices, routing weights, and softmax probabilities."""
    logits = jnp.einsum("...d,ed->...e", z, params["router"]["W"]) + params["router"]["b"]
    probs = jax.nn.softmax(logits, axis=-1)
    idx = jnp.argmax(probs, axis=-1)
    weight = jnp.take_along_axis(probs, idx[..., None], axis=-1)[..., 0]
    return idx.astype(jnp.int32), weight, probs


def apply_experts(params, z, idx, weight):
    """Run every expert on this loop's z, then mix with the (possibly replayed) route."""
    hidden = jax.nn.gelu(jnp.einsum("...d,ehd->...eh", z, params["W1"]) + params["b1"])
    outs = jnp.einsum("...eh,edh->...ed", hidden, params["W2"]) + params["b2"]
    chosen = jnp.take_along_axis(outs, idx[..., None, None], axis=-2)[..., 0, :]
    return chosen * weight[..., None]


def load_balance_aux(probs, idx, n_experts):
    """Switch-style load-balance loss: E * sum_i f_i * P_i, computed on one router pass."""
    flat_probs = probs.reshape(-1, n_experts)
    flat_idx = idx.reshape(-1)
    assigned = jax.nn.one_hot(flat_idx, n_experts, dtype=flat_probs.dtype)
    freq = jnp.mean(assigned, axis=0)
    mean_prob = jnp.mean(flat_probs, axis=0)
    return n_experts * jnp.sum(freq * mean_prob)


def run_moe_route_replay(params, z_by_loop, n_experts=None):
    """Router runs on loop 1 only. Later loops reuse indices and weights.

    Experts still run on that loop's z. Aux is the loop-1 load-balance loss
    multiplied by the realized loop count k.
    """
    if n_experts is None:
        n_experts = int(params["router"]["W"].shape[0])
    k = len(z_by_loop)
    idx, weight, probs = route_tokens(params, z_by_loop[0])
    aux1 = load_balance_aux(probs, idx, n_experts)
    outputs = []
    indices = []
    weights = []
    for z in z_by_loop:
        outputs.append(apply_experts(params, z, idx, weight))
        indices.append(idx)
        weights.append(weight)
    aux_total = aux1 * k
    return {
        "outputs": outputs,
        "indices": indices,
        "weights": weights,
        "aux_loop1": aux1,
        "aux_total": aux_total,
        "probs_loop1": probs,
    }


def empty_kv_cache(batch, seq, n_heads, head_dim, n_loops, dtype=jnp.float32):
    """One cache slot per token. n_loops does not appear in the shape.

    Memory stays the one-loop size at every k because each loop overwrites the slot.
    """
    del n_loops
    k = jnp.zeros((batch, seq, n_heads, head_dim), dtype=dtype)
    v = jnp.zeros((batch, seq, n_heads, head_dim), dtype=dtype)
    return k, v


def write_kv(cache, t, value):
    """Overwrite the single slot for token t. value is (batch, heads, head_dim)."""
    return cache.at[:, t].set(value)


def last_loop_attention(q, k_cache, v_cache, t):
    """Causal attention over the one-loop cache.

    q is (batch, heads, head_dim). Positions 0..t-1 hold each earlier token's
    last-written KV (its final loop, once that token has finished). Position t
    holds the current token's loop-i KV, which the caller writes before attending.
    """
    scale = q.shape[-1] ** -0.5
    logits = jnp.einsum("bhd,bthd->bht", q, k_cache) * scale
    positions = jnp.arange(k_cache.shape[1])
    mask = positions <= t
    neg = jnp.asarray(-1.0e30, dtype=logits.dtype)
    logits = jnp.where(mask[None, None, :], logits, neg)
    weights = jax.nn.softmax(logits, axis=-1)
    return jnp.einsum("bht,bthd->bhd", weights, v_cache)
