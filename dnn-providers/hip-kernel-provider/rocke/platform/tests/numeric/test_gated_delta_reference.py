# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Host-only tests for the float64 gated delta rule references in
``rocke.numeric.gated_delta``.

The gate functions are checked against their defining formulas, one token of
the rule against the rule written longhand with named einsum axes (which pins
the axis the decay scales), and the chunkwise tiles against the token-serial
recurrence: rebuilding every chunk's output and end state from the tiles must
reproduce the recurrence, which holds only if both are right.
"""

from __future__ import annotations

import re
import warnings

import numpy as np
import pytest
from rocke.helpers.activations import SOFTPLUS_THRESHOLD
from rocke.numeric.gated_delta import (
    CHUNKWISE_TILES,
    chunkwise_tiles,
    gated_delta_recurrence,
    gated_delta_step,
    gdn_log_decay,
    kda_log_decay,
    l2_normalize,
    sigmoid,
    softplus,
)

# DK != DV everywhere, so a state transposed by mistake fails on shape or value.
HEAD_K, HEAD_V = 6, 5


def test_softplus_switches_to_identity_at_the_emitter_threshold():
    below = np.array([-30.0, -1.0, 0.0, 3.0, SOFTPLUS_THRESHOLD])
    np.testing.assert_allclose(
        softplus(below), np.log1p(np.exp(below)), rtol=1e-15, equal_nan=False
    )
    above = np.array([SOFTPLUS_THRESHOLD + 1e-3, 50.0, 1e4])
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no overflow in the branch not taken
        np.testing.assert_array_equal(softplus(above), above)


def test_sigmoid_matches_its_definition_and_stays_finite():
    x = np.linspace(-30.0, 30.0, 61)
    np.testing.assert_allclose(
        sigmoid(x), 1.0 / (1.0 + np.exp(-x)), rtol=1e-14, equal_nan=False
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        extremes = sigmoid(np.array([-1e4, 1e4]))
    np.testing.assert_array_equal(extremes, [0.0, 1.0])


def test_l2_normalize_gives_unit_rows_and_leaves_zero_rows_zero():
    rows = np.array([[3.0, 4.0], [0.0, 0.0], [1e-3, 0.0]])
    np.testing.assert_allclose(
        l2_normalize(rows), [[0.6, 0.8], [0, 0], [1, 0]], equal_nan=False
    )


def test_gdn_log_decay_matches_its_formula():
    rng = np.random.default_rng(0)
    a, dt_bias, a_log = rng.normal(size=(3, 4)), rng.normal(size=4), rng.normal(size=4)
    x = a + dt_bias
    want = -np.exp(a_log) * np.log1p(np.exp(x))
    np.testing.assert_allclose(
        gdn_log_decay(a, dt_bias, a_log), want, rtol=1e-15, equal_nan=False
    )


def test_kda_log_decay_matches_its_formula_and_lies_in_the_bound():
    rng = np.random.default_rng(1)
    g, dt_bias = rng.normal(size=(3, 4, 8)), rng.normal(size=(4, 8))
    a_log = rng.normal(size=(4, 1))
    got = kda_log_decay(g, dt_bias, a_log, lower_bound=-5.0)
    want = -5.0 / (1.0 + np.exp(-np.exp(a_log) * (g + dt_bias)))
    np.testing.assert_allclose(got, want, rtol=1e-14, equal_nan=False)
    assert ((got > -5.0) & (got < 0.0)).all()


@pytest.mark.parametrize("lower_bound", [0.0, 2.0])
def test_kda_log_decay_refuses_a_bound_that_would_amplify(lower_bound):
    with pytest.raises(
        ValueError,
        match=re.escape(f"kda_log_decay failed - lower_bound={lower_bound!r}"),
    ):
        kda_log_decay(np.zeros(2), np.zeros(2), np.zeros(2), lower_bound)


def _one_token(rng, batch=(2, 3)):
    state = rng.normal(size=(*batch, HEAD_K, HEAD_V))
    q = rng.normal(size=(*batch, HEAD_K))
    k = rng.normal(size=(*batch, HEAD_K))
    v = rng.normal(size=(*batch, HEAD_V))
    log_decay = -rng.random(size=(*batch, HEAD_K))
    beta = rng.random(size=batch)
    return state, q, k, v, log_decay, beta


def test_step_matches_the_rule_written_longhand():
    """Named einsum axes: ``d`` is the K channel on the state and on the decay,
    so this pins the decay to the K axis (rows) of the state."""
    state, q, k, v, log_decay, beta = _one_token(np.random.default_rng(2))
    faded = np.einsum("bhde,bhd->bhde", state, np.exp(log_decay))
    update = beta[..., None] * (v - np.einsum("bhd,bhde->bhe", k, faded))
    after = faded + np.einsum("bhd,bhe->bhde", k, update)
    out = np.einsum("bhd,bhde->bhe", q, after)

    got_out, got_after = gated_delta_step(state, q, k, v, log_decay, beta)
    np.testing.assert_allclose(got_after, after, rtol=1e-13, equal_nan=False)
    np.testing.assert_allclose(got_out, out, rtol=1e-13, equal_nan=False)


def test_step_scalar_gate_equals_a_channel_constant_gate():
    state, q, k, v, log_decay, beta = _one_token(np.random.default_rng(3))
    scalar = log_decay[..., :1]
    constant = np.broadcast_to(scalar, log_decay.shape)
    np.testing.assert_array_equal(
        gated_delta_step(state, q, k, v, scalar, beta)[1],
        gated_delta_step(state, q, k, v, constant, beta)[1],
    )


def test_a_full_strength_write_stores_the_value_exactly():
    """beta = 1, no decay and a unit key: afterwards ``k^T S == v``, the delta
    rule's defining property (the write replaces what the key read)."""
    state, q, k, v, _, _ = _one_token(np.random.default_rng(4))
    k = l2_normalize(k)
    _, after = gated_delta_step(state, q, k, v, np.zeros_like(k), np.ones(k.shape[:-1]))
    np.testing.assert_allclose(
        np.einsum("bhd,bhde->bhe", k, after), v, atol=1e-13, equal_nan=False
    )


def test_step_refuses_a_transposed_state():
    state, q, k, v, log_decay, beta = _one_token(np.random.default_rng(5))
    with pytest.raises(
        ValueError,
        match=re.escape("gated_delta_step failed - state.shape=(2, 3, 5, 6)"),
    ):
        gated_delta_step(np.swapaxes(state, -1, -2), q, k, v, log_decay, beta)


def _sequence(rng, batch=2, heads=3, tokens=64):
    q = rng.normal(size=(batch, heads, tokens, HEAD_K))
    k = l2_normalize(rng.normal(size=(batch, heads, tokens, HEAD_K)))
    v = rng.normal(size=(batch, heads, tokens, HEAD_V))
    log_decay = -2.0 * rng.random(size=(batch, heads, tokens, HEAD_K))
    beta = rng.random(size=(batch, heads, tokens))
    initial = rng.normal(size=(batch, heads, HEAD_K, HEAD_V))
    return q, k, v, log_decay, beta, initial


def test_recurrence_is_the_step_applied_token_by_token():
    q, k, v, log_decay, beta, initial = _sequence(np.random.default_rng(6), tokens=3)
    scale = 0.5
    out, final = gated_delta_recurrence(
        q, k, v, log_decay, beta, scale=scale, initial_state=initial
    )
    state = initial
    for t in range(3):
        o_t, state = gated_delta_step(
            state,
            scale * q[:, :, t],
            k[:, :, t],
            v[:, :, t],
            log_decay[:, :, t],
            beta[:, :, t],
        )
        np.testing.assert_array_equal(out[:, :, t], o_t)
    np.testing.assert_array_equal(final, state)


def test_recurrence_starts_from_a_zero_state_by_default():
    q, k, v, log_decay, beta, _ = _sequence(np.random.default_rng(7), tokens=4)
    zero = np.zeros((*q.shape[:2], HEAD_K, HEAD_V))
    np.testing.assert_array_equal(
        gated_delta_recurrence(q, k, v, log_decay, beta, scale=1.0)[1],
        gated_delta_recurrence(q, k, v, log_decay, beta, scale=1.0, initial_state=zero)[
            1
        ],
    )


@pytest.mark.parametrize("chunk", [8, 16])
def test_chunkwise_tiles_reproduce_the_token_serial_recurrence(chunk):
    """Walk the chunks with the tiles alone -- ``U = a (V - gk S)``, output
    ``gq S + aqk U``, end state ``dec * S + kt U`` -- and compare every output
    row and the final state with the recurrence. Steep decays (down to -2 per
    token) make the pairwise exponents span many orders of magnitude."""
    q, k, v, log_decay, beta, initial = _sequence(np.random.default_rng(8))
    scale = HEAD_K**-0.5
    want_out, want_final = gated_delta_recurrence(
        q, k, v, log_decay, beta, scale=scale, initial_state=initial
    )

    batch, heads, tokens, _ = q.shape
    chunks = tokens // chunk

    def by_chunk(x):
        return x.reshape(batch * heads * chunks, chunk, *x.shape[3:])

    tiles = chunkwise_tiles(
        by_chunk(q),
        by_chunk(k),
        by_chunk(log_decay),
        by_chunk(beta),
        scale,
        chunks_per_block=5,
    )
    assert tuple(tiles) == CHUNKWISE_TILES
    values = by_chunk(v).reshape(batch * heads, chunks, chunk, HEAD_V)
    tiles = {
        name: tile.reshape(batch * heads, chunks, *tile.shape[1:])
        for name, tile in tiles.items()
    }
    state = initial.reshape(batch * heads, HEAD_K, HEAD_V)
    out = np.empty_like(values)
    for c in range(chunks):
        tile = {name: t[:, c] for name, t in tiles.items()}
        update = tile["a"] @ (values[:, c] - tile["gk"] @ state)
        out[:, c] = tile["gq"] @ state + tile["aqk"] @ update
        state = tile["dec"][:, :, None] * state + tile["kt"] @ update

    np.testing.assert_allclose(
        out.reshape(want_out.shape), want_out, rtol=0, atol=1e-11, equal_nan=False
    )
    np.testing.assert_allclose(
        state.reshape(want_final.shape), want_final, rtol=0, atol=1e-11, equal_nan=False
    )
