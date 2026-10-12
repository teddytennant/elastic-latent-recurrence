"""Contract tests for elastic latent recurrence."""

import math

import jax
import jax.numpy as jnp
import pytest

from elastic_latent_recurrence import (
    BETA_ZERO,
    adaptive_delta_step,
    beta,
    empty_kv_cache,
    entry_z,
    exit_state_update,
    f_for_target_p1,
    first_loop_inputs,
    init_beta_params,
    init_exit_params,
    init_modulation_params,
    init_moe_params,
    last_loop_attention,
    load_balance_aux,
    mean_loop_count,
    modulated_layer,
    output_progress_and_budget,
    p1_marginal,
    progress_and_budget,
    rms_norm,
    run_moe_route_replay,
    sample_loop_counts,
    sinusoidal_embedding,
    train_smoke,
    write_kv,
)
from elastic_latent_recurrence.ops import route_tokens


def _enable_x64():
    jax.config.update("jax_enable_x64", True)


def test_beta_zero_init_is_half_log_two():
    _enable_x64()
    d = 8
    params = init_beta_params(d, dtype=jnp.float64)
    s_tilde = jax.random.normal(jax.random.PRNGKey(0), (4, d), dtype=jnp.float64)
    got = beta(s_tilde, params)
    target = 0.34657359027997264
    assert math.isclose(BETA_ZERO, target, rel_tol=0.0, abs_tol=1e-15)
    assert jnp.allclose(got, target, atol=1e-12, rtol=0.0)
    assert jnp.allclose(got, 0.5 * jnp.log(jnp.float64(2.0)), atol=1e-12, rtol=0.0)
    assert got.shape == s_tilde.shape


def test_beta_is_left_multiply():
    s = jnp.array([1.0, 0.0, 0.0])
    weight = jnp.zeros((3, 3)).at[1, 0].set(4.0)
    params = {
        "W": weight,
        "b": jnp.array([0.1, -0.2, 0.3]),
        "gamma": jnp.array([0.0, 0.5, -0.5]),
    }
    got = beta(s, params)
    logits = weight @ s + params["b"]
    expected = jax.nn.sigmoid(logits) * jax.nn.softplus(params["gamma"])
    assert jnp.allclose(got, expected, atol=1e-6)


def test_mixed_depth_formulas_and_solver():
    for K, f in ((4, 0.0), (4, 1.0), (4, 0.2), (8, 0.5), (5, 1.0 / 9.0)):
        p1 = 1.0 / K + (1.0 - 1.0 / K) * f
        k_bar = f + (1.0 - f) * (K + 1) / 2.0
        assert p1_marginal(K, f) == pytest.approx(p1)
        assert mean_loop_count(K, f) == pytest.approx(k_bar)
    f_star = f_for_target_p1(4, 1.0 / 3.0)
    assert f_star == pytest.approx((4 * (1.0 / 3.0) - 1.0) / 3.0)
    assert p1_marginal(4, f_star) == pytest.approx(1.0 / 3.0)
    assert f_for_target_p1(6) == pytest.approx(f_for_target_p1(6, 1.0 / 3.0))
    assert mean_loop_count(4, 0.0) == pytest.approx(2.5)
    assert mean_loop_count(4, 1.0) == pytest.approx(1.0)


def test_sampler_empirical_one_loop_fraction():
    K = 4
    f = f_for_target_p1(K, 1.0 / 3.0)
    p1 = p1_marginal(K, f)
    counts = sample_loop_counts(jax.random.PRNGKey(0), 100_000, K, f)
    assert int(jnp.min(counts)) >= 1
    assert int(jnp.max(counts)) <= K
    empirical = float(jnp.mean(counts == 1))
    assert empirical == pytest.approx(p1, abs=0.01)


def test_progress_budget_and_output_pair():
    tau, kappa = progress_and_budget(2, 4, 8)
    assert tau == pytest.approx(0.5)
    assert kappa == pytest.approx(0.5)
    tau_out, kappa_out = output_progress_and_budget(3, 4)
    assert tau_out == pytest.approx(1.0)
    assert kappa_out == pytest.approx(0.75)


def test_sinusoidal_embedding_is_transformer_style():
    dim = 8
    t = jnp.asarray(1.25)
    emb = sinusoidal_embedding(t, dim, base=10000.0)
    half = dim // 2
    inv_freq = jnp.exp(jnp.arange(half, dtype=t.dtype) * (-jnp.log(jnp.asarray(10000.0)) / dim))
    angles = t * inv_freq
    expected = jnp.zeros((dim,), dtype=t.dtype)
    expected = expected.at[0::2].set(jnp.sin(angles))
    expected = expected.at[1::2].set(jnp.cos(angles))
    assert emb.shape == (dim,)
    assert jnp.allclose(emb, expected, atol=1e-6)


def test_zero_init_modulation_matches_unconditioned_rmsnorm_f():
    _enable_x64()
    d, rank, cond_dim, batch = 16, 4, 8, 5
    key = jax.random.PRNGKey(2)
    k_x, k_c, k_p, k_f = jax.random.split(key, 4)
    x = jax.random.normal(k_x, (batch, d), dtype=jnp.float64)
    c = jax.random.normal(k_c, (batch, cond_dim), dtype=jnp.float64)
    params = init_modulation_params(k_p, d, rank, cond_dim, dtype=jnp.float64, scale=0.3)
    # Heads stay zero; A and B are nonzero so the low-rank term must be gated off.
    assert float(jnp.max(jnp.abs(params["A"]))) > 0.0
    assert float(jnp.max(jnp.abs(params["g"]["W"]))) == 0.0
    weight = jax.random.normal(k_f, (d, d), dtype=jnp.float64)

    def f(h):
        return jnp.tanh(h @ weight.T)

    got = modulated_layer(x, c, f, params)
    expected = x + f(rms_norm(x))
    assert jnp.allclose(got, expected, atol=1e-6, rtol=0.0)


def test_entry_first_loop_and_zero_init_later_loops():
    _enable_x64()
    d, cond_dim = 8, 4
    h_in = jax.random.normal(jax.random.PRNGKey(3), (2, d), dtype=jnp.float64)
    s_tilde, z = first_loop_inputs(h_in)
    assert jnp.allclose(s_tilde, 2.0 * h_in)
    assert jnp.allclose(z, s_tilde)
    params = {
        "lambda_e": {"W": jnp.zeros((d, cond_dim), dtype=jnp.float64), "b": jnp.zeros((d,), dtype=jnp.float64)},
        "mu_e": {"W": jnp.zeros((d, cond_dim), dtype=jnp.float64), "b": jnp.zeros((d,), dtype=jnp.float64)},
        "eta": {"W": jnp.zeros((d, cond_dim), dtype=jnp.float64), "b": jnp.zeros((d,), dtype=jnp.float64)},
    }
    carried = jax.random.normal(jax.random.PRNGKey(4), (2, d), dtype=jnp.float64)
    c = jax.random.normal(jax.random.PRNGKey(5), (2, cond_dim), dtype=jnp.float64)
    z_later = entry_z(carried, h_in, c, params)
    assert jnp.allclose(z_later, rms_norm(carried), atol=1e-6, rtol=0.0)


def test_zero_g_exit_matches_adaptive_delta_then_freezes():
    _enable_x64()
    d, cond_dim, batch = 8, 4, 3
    key = jax.random.PRNGKey(6)
    keys = jax.random.split(key, 8)
    params = init_exit_params(d, cond_dim, dtype=jnp.float64)
    params["beta"] = {
        "W": jax.random.normal(keys[0], (d, d), dtype=jnp.float64) * 0.1,
        "b": jax.random.normal(keys[1], (d,), dtype=jnp.float64) * 0.1,
        "gamma": jax.random.normal(keys[2], (d,), dtype=jnp.float64) * 0.1,
    }
    s0 = jnp.zeros((batch, d), dtype=jnp.float64)
    s_tilde = jax.random.normal(keys[3], (batch, d), dtype=jnp.float64)
    r = jax.random.normal(keys[4], (batch, d), dtype=jnp.float64)
    c = jax.random.normal(keys[5], (batch, cond_dim), dtype=jnp.float64)
    s1 = exit_state_update(s0, s_tilde, r, c, 1, params)
    s1_plain = adaptive_delta_step(s0, s_tilde, r, params["beta"])
    assert jnp.allclose(s1, s1_plain, atol=1e-8, rtol=0.0)
    state = s1
    for i, k in enumerate(jax.random.split(keys[6], 3), start=2):
        k1, k2, k3 = jax.random.split(k, 3)
        s_tilde_i = jax.random.normal(k1, (batch, d), dtype=jnp.float64)
        r_i = jax.random.normal(k2, (batch, d), dtype=jnp.float64)
        c_i = jax.random.normal(k3, (batch, cond_dim), dtype=jnp.float64)
        nxt = exit_state_update(state, s_tilde_i, r_i, c_i, i, params)
        assert jnp.array_equal(nxt, state)
        state = nxt


def test_nonzero_g_exit_moves_later_loops():
    _enable_x64()
    d, cond_dim, batch = 6, 3, 2
    key = jax.random.PRNGKey(7)
    k_s, k_st, k_r, k_c, k_b, k_g = jax.random.split(key, 6)
    g_value = jax.random.normal(k_g, (d,), dtype=jnp.float64)
    params = init_exit_params(d, cond_dim, dtype=jnp.float64)
    params["beta"] = {
        "W": jax.random.normal(k_b, (d, d), dtype=jnp.float64) * 0.2,
        "b": jnp.zeros((d,), dtype=jnp.float64),
        "gamma": jax.random.normal(k_b, (d,), dtype=jnp.float64) * 0.2,
    }
    params["g_exit"] = {
        "W": jnp.zeros((d, cond_dim), dtype=jnp.float64),
        "b": g_value,
    }
    s_prev = jax.random.normal(k_s, (batch, d), dtype=jnp.float64)
    s_tilde = jax.random.normal(k_st, (batch, d), dtype=jnp.float64)
    r = jax.random.normal(k_r, (batch, d), dtype=jnp.float64)
    c = jax.random.normal(k_c, (batch, cond_dim), dtype=jnp.float64)
    s_new = exit_state_update(s_prev, s_tilde, r, c, 3, params)
    b = beta(s_tilde, params["beta"])
    expected = s_prev + g_value * b * (r - s_prev)
    assert jnp.allclose(s_new - s_prev, g_value * b * (r - s_prev), atol=1e-8, rtol=0.0)
    assert jnp.allclose(s_new, expected, atol=1e-8, rtol=0.0)
    assert float(jnp.max(jnp.abs(s_new - s_prev))) > 1e-6


def test_route_replay_keeps_loop1_indices_and_scales_aux():
    d, n_experts, hidden, n, k = 8, 4, 8, 16, 3
    params = init_moe_params(jax.random.PRNGKey(8), d, n_experts, hidden, dtype=jnp.float32, scale=0.2)
    eye = jnp.concatenate([jnp.eye(n_experts), jnp.zeros((n_experts, d - n_experts))], axis=1)
    params["router"] = {"W": eye, "b": jnp.zeros((n_experts,))}
    z1 = jnp.zeros((n, d)).at[:, 0].set(1.0)
    z2 = jnp.zeros((n, d)).at[:, 1].set(1.0)
    z3 = jnp.zeros((n, d)).at[:, 2].set(2.0)
    result = run_moe_route_replay(params, [z1, z2, z3], n_experts=n_experts)
    idx_fresh, _, _ = route_tokens(params, z2)
    assert not jnp.array_equal(idx_fresh, result["indices"][0])
    for idx, weight in zip(result["indices"], result["weights"]):
        assert jnp.array_equal(idx, result["indices"][0])
        assert jnp.allclose(weight, result["weights"][0])
    aux_ref = load_balance_aux(result["probs_loop1"], result["indices"][0], n_experts)
    assert jnp.allclose(result["aux_loop1"], aux_ref, atol=1e-6)
    assert jnp.allclose(result["aux_total"], k * aux_ref, atol=1e-6)
    assert not jnp.allclose(result["outputs"][0], result["outputs"][1], atol=1e-5)


def test_last_loop_kv_shape_and_attention():
    batch, seq, n_heads, head_dim, n_loops = 2, 5, 2, 4, 7
    cache_k, cache_v = empty_kv_cache(batch, seq, n_heads, head_dim, n_loops)
    one_loop_shape = (batch, seq, n_heads, head_dim)
    assert cache_k.shape == one_loop_shape
    assert cache_v.shape == one_loop_shape
    assert cache_k.ndim == 4
    assert n_loops not in cache_k.shape

    keys = jax.random.split(jax.random.PRNGKey(9), 7)
    shape = (batch, n_heads, head_dim)
    k_loop1 = jax.random.normal(keys[0], shape, dtype=jnp.float32)
    v_loop1 = jax.random.normal(keys[1], shape, dtype=jnp.float32)
    k_last = jax.random.normal(keys[2], shape, dtype=jnp.float32) + 3.0
    v_last = jax.random.normal(keys[3], shape, dtype=jnp.float32) - 2.0
    k_self = jax.random.normal(keys[4], shape, dtype=jnp.float32)
    v_self = jax.random.normal(keys[5], shape, dtype=jnp.float32)
    q = jax.random.normal(keys[6], shape, dtype=jnp.float32)

    # Previous token finishes loop 1, then its last loop overwrites the only slot.
    ck = write_kv(cache_k, 0, k_loop1)
    cv = write_kv(cache_v, 0, v_loop1)
    ck = write_kv(ck, 0, k_last)
    cv = write_kv(cv, 0, v_last)
    # Current token is on loop 2 and writes its own loop-2 KV before attending.
    ck = write_kv(ck, 1, k_self)
    cv = write_kv(cv, 1, v_self)
    assert ck.shape == one_loop_shape

    out = last_loop_attention(q, ck, cv, t=1)

    def reference(query, keys, values):
        key_mat = jnp.stack(keys, axis=1)
        value_mat = jnp.stack(values, axis=1)
        scale = head_dim ** -0.5
        logits = jnp.einsum("bhd,bshd->bhs", query, key_mat) * scale
        weights = jax.nn.softmax(logits, axis=-1)
        return jnp.einsum("bhs,bshd->bhd", weights, value_mat)

    expected = reference(q, [k_last, k_self], [v_last, v_self])
    wrong = reference(q, [k_loop1, k_self], [v_loop1, v_self])
    assert jnp.allclose(out, expected, atol=1e-5, rtol=1e-5)
    assert not jnp.allclose(out, wrong, atol=1e-3)


def test_smoke_train_losses_are_finite():
    result = train_smoke(steps=30, batch_size=4, seed=0)
    assert math.isfinite(result["loss_step_0"])
    assert math.isfinite(result["loss_step_30"])
    assert result["loss_step_0"] == result["loss_step_0"]
    assert result["loss_step_30"] == result["loss_step_30"]
