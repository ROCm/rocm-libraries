# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU-only contracts for the key-permutation relation: token maps, fixture,
derived bound, analyzer and subject scope.

The analyzer is exercised on outputs of ``_model_dense_run``, an FP32 model of
the dense kernel's single-pass online softmax. The model exists only in these
contracts. No GPU verdict uses it: the relation compares two subject outputs.
"""

from __future__ import annotations

import hashlib
import itertools
import math
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache

import numpy as np
import pytest
from rocke.numeric.bf16 import bf16_to_f32, f32_to_bf16

from ._analytical import _FORMAT, decode_bits, encode_bits
from ._dense_subject import (
    DENSE_SUBJECTS,
    PERMUTATION_SUBJECT_CASES,
    _verify_resolved_spec,
    resolve_dense_subject,
    softmax_scale,
)
from ._permutation import (
    _LATTICE_LIMIT,
    CANONICAL_SPAN,
    PERMUTATION_MUTATION_MAP,
    PERMUTATION_QUICK_CASES,
    PERMUTATION_QUICK_MAPS,
    PERMUTATION_SQUARE_BF16,
    PERMUTATION_SQUARE_FP16,
    PairingMutation,
    PermutationBound,
    PermutationCase,
    PermutationResult,
    TokenMap,
    analyze_permutation,
    format_permutation_result,
    mutated_token_maps,
    permutation_bound,
    permutation_case_bound,
    permutation_inputs,
    permute_tokens,
    score_spread_bound,
    token_map,
)
from .test_contracts import _fake_tuning_spec
from .test_permutation_dense import _unresponsive

_LOG2E = 1.4426950408889634
# KV tile of the shipped gfx942 dense default; the model also runs at half of it.
_MODEL_BLOCK_N = 64


def _case_id(case: PermutationCase) -> str:
    return case.case_id


# ---------------------------------------------------------------------------
# Token maps
# ---------------------------------------------------------------------------


def test_token_maps_pin_the_destination_to_source_convention() -> None:
    index = np.arange(37)
    assert np.array_equal(token_map("shift_one", 37), (index + 1) % 37)
    assert np.array_equal(token_map("reverse", 37), 36 - index)
    assert np.array_equal(token_map("cross_tile", 37), (index + CANONICAL_SPAN) % 37)
    # The 5-token tail span 32..36 moves to the front.
    assert np.array_equal(token_map("partial_final_tile", 37), (index + 32) % 37)
    assert token_map("partial_final_tile", 37)[:6].tolist() == [32, 33, 34, 35, 36, 0]
    assert token_map("adjacent_swap", 37).tolist() == [1, 0, *range(2, 37)]

    # Destination token k receives source token map[k], for every batch and head.
    tokens = np.arange(2 * 37 * 3 * 4, dtype=np.uint16).reshape(2, 37, 3, 4)
    moved = permute_tokens(tokens, token_map(TokenMap.SHIFT_ONE, 37))
    assert np.array_equal(moved[:, 0], tokens[:, 1])
    assert np.array_equal(moved[:, 36], tokens[:, 0])
    assert moved.flags.c_contiguous and moved.dtype == tokens.dtype


@pytest.mark.parametrize("seqlen_k", [2, 3, 17, 37, 64, 250, 256])
def test_every_applicable_token_map_is_a_nontrivial_permutation(seqlen_k: int) -> None:
    for kind in TokenMap:
        try:
            source = token_map(kind, seqlen_k)
        except ValueError:
            assert kind in {TokenMap.CROSS_TILE, TokenMap.PARTIAL_FINAL_TILE}
            continue
        assert np.array_equal(np.sort(source), np.arange(seqlen_k))
        assert not np.array_equal(source, np.arange(seqlen_k))


def test_token_maps_refuse_lengths_they_cannot_move() -> None:
    with pytest.raises(ValueError, match="at least two key tokens"):
        token_map("shift_one", 1)
    with pytest.raises(ValueError, match="more than 16 key tokens"):
        token_map("cross_tile", 16)
    with pytest.raises(ValueError, match="more than 16 key tokens"):
        token_map("partial_final_tile", 9)
    with pytest.raises(ValueError, match="partial final span"):
        token_map("partial_final_tile", 256)
    with pytest.raises(ValueError, match="unknown token map"):
        token_map("rotate", 8)
    with pytest.raises(ValueError, match="every key index exactly once"):
        permute_tokens(np.zeros((1, 4, 1, 2), dtype=np.uint16), np.array([0, 0, 1, 2]))
    with pytest.raises(ValueError, match=r"\[B, Sk, Hkv, D\]"):
        permute_tokens(np.zeros((4, 2), dtype=np.uint16), np.arange(2))


def test_pairing_mutations_always_separate_the_key_and_value_maps() -> None:
    identity = np.arange(256)
    reverse = token_map("reverse", 256)
    shift = token_map("shift_one", 256)

    key_map, value_map = mutated_token_maps("key_only", "reverse", 256)
    assert np.array_equal(key_map, reverse) and np.array_equal(value_map, identity)
    key_map, value_map = mutated_token_maps("value_only", "reverse", 256)
    assert np.array_equal(key_map, identity) and np.array_equal(value_map, reverse)
    key_map, value_map = mutated_token_maps("different_maps", "reverse", 256)
    assert np.array_equal(key_map, reverse) and np.array_equal(value_map, shift)
    key_map, value_map = mutated_token_maps("different_maps", "shift_one", 256)
    assert np.array_equal(key_map, shift) and np.array_equal(value_map, reverse)

    # With two tokens shift_one and reverse coincide, so V keeps the identity.
    key_map, value_map = mutated_token_maps("different_maps", "reverse", 2)
    assert key_map.tolist() == [1, 0] and value_map.tolist() == [0, 1]

    for seqlen_k in (2, 3, 37, 256):
        for mutation in PairingMutation:
            for kind in (TokenMap.SHIFT_ONE, TokenMap.REVERSE, TokenMap.ADJACENT_SWAP):
                key_map, value_map = mutated_token_maps(mutation, kind, seqlen_k)
                assert not np.array_equal(key_map, value_map)
    with pytest.raises(ValueError, match="unknown pairing mutation"):
        mutated_token_maps("drop_value", "reverse", 8)


def _commuting(pairings: np.ndarray, source: np.ndarray) -> np.ndarray:
    """Which rows of ``pairings`` commute with the map ``source``."""
    return np.all(pairings[:, source] == source[pairings], axis=1)


@pytest.mark.parametrize("seqlen_k", [6, 7, 8])
def test_quick_maps_leave_no_pairing_permutation_undetected(seqlen_k: int) -> None:
    """A kernel that pairs the score of key slot ``j`` with the value of slot
    ``sigma(j)``, for a fixed ``sigma``, satisfies the relation under a map for
    all inputs exactly when ``sigma`` commutes with it. The shifts and the
    reverse generate a dihedral group whose even-length centralizer still holds
    the half-sequence shift; the adjacent swap leaves only the identity. This is
    a statement in exact arithmetic about fixed slot maps: whether one fixture
    shows a given defect beyond the bound is a separate question."""
    pairings = np.array(list(itertools.permutations(range(seqlen_k))))
    dihedral = _commuting(pairings, token_map("shift_one", seqlen_k)) & _commuting(
        pairings, token_map("reverse", seqlen_k)
    )
    survivors = {tuple(row) for row in pairings[dihedral]}
    identity = tuple(range(seqlen_k))
    half_shift = tuple((np.arange(seqlen_k) + seqlen_k // 2) % seqlen_k)
    assert survivors == ({identity, half_shift} if seqlen_k % 2 == 0 else {identity})

    separated = dihedral & _commuting(pairings, token_map("adjacent_swap", seqlen_k))
    assert {tuple(row) for row in pairings[separated]} == {identity}


@pytest.mark.parametrize("seqlen_k", [3, 4, 5, 6])
def test_shift_and_swap_leave_no_pairing_map_undetected(seqlen_k: int) -> None:
    """The criterion does not need the pairing to be a bijection. Over all
    ``Sk**Sk`` maps, the ones commuting with ``shift_one`` are the ``Sk`` cyclic
    shifts, and the identity alone also commutes with ``adjacent_swap``."""
    pairings = np.array(list(itertools.product(range(seqlen_k), repeat=seqlen_k)))
    shifts = _commuting(pairings, token_map("shift_one", seqlen_k))
    index = np.arange(seqlen_k)
    assert {tuple(row) for row in pairings[shifts]} == {
        tuple((index + shift) % seqlen_k) for shift in range(seqlen_k)
    }
    separated = shifts & _commuting(pairings, token_map("adjacent_swap", seqlen_k))
    assert {tuple(row) for row in pairings[separated]} == {tuple(index)}


def test_two_tokens_are_outside_the_separation_argument() -> None:
    """With two tokens the three maps are one exchange, and exchanging the two
    values commutes with it: that pairing defect passes every map. A case must
    therefore have more than two tokens, and the lane requires more than 16."""
    exchange = token_map("shift_one", 2)
    assert np.array_equal(exchange, token_map("reverse", 2))
    assert np.array_equal(exchange, token_map("adjacent_swap", 2))
    pairings = np.array(list(itertools.product(range(2), repeat=2)))
    survivors = {tuple(row) for row in pairings[_commuting(pairings, exchange)]}
    assert survivors == {(0, 1), (1, 0)}
    with pytest.raises(ValueError, match="more than 16 key tokens"):
        PermutationCase(1, 4, 2, 2, 2, 8, "fp16").require_scope()


def test_quick_maps_hold_the_separating_set() -> None:
    assert len(set(PERMUTATION_QUICK_MAPS)) == len(PERMUTATION_QUICK_MAPS)
    assert {
        TokenMap.SHIFT_ONE,
        TokenMap.REVERSE,
        TokenMap.ADJACENT_SWAP,
    } <= set(PERMUTATION_QUICK_MAPS)
    for case in PERMUTATION_QUICK_CASES:
        for kind in PERMUTATION_QUICK_MAPS:
            token_map(kind, case.seqlen_k)
        # The mutations start from a map that moves every token.
        source = token_map(PERMUTATION_MUTATION_MAP, case.seqlen_k)
        assert not np.any(source == np.arange(case.seqlen_k))


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", PERMUTATION_QUICK_CASES, ids=_case_id)
def test_inputs_are_exact_in_both_dtypes_and_on_the_contraction_lattice(
    case: PermutationCase,
) -> None:
    q, k, v = permutation_inputs(case)
    assert q.shape == case.output_shape
    assert k.shape == v.shape == case.key_shape
    for values in (q, k, v):
        assert values.dtype == np.float32
        for dtype in ("fp16", "bf16"):
            bits = encode_bits(values, dtype)
            assert np.array_equal(decode_bits(bits, dtype), values.astype(np.float64))
    for values in (q, k):
        eighths = values.astype(np.float64) * 8.0
        assert np.array_equal(eighths, np.round(eighths))
        assert np.max(np.abs(eighths)) <= _LATTICE_LIMIT
    assert set(np.unique(v).tolist()) == {-1.0, 1.0}
    # Every product is a multiple of 1/64 and every partial sum of a contraction
    # stays below 2**24 of them, so FP32 accumulates the scores exactly.
    assert case.head_dim * _LATTICE_LIMIT**2 <= 1 << 24


@pytest.mark.parametrize("case", PERMUTATION_QUICK_CASES, ids=_case_id)
def test_inputs_are_nondegenerate(case: PermutationCase) -> None:
    """A broken pairing is visible only if the tokens differ: no zero query row,
    and no two equal key or value tokens within one batch element and KV head."""
    q, k, v = permutation_inputs(case)
    assert np.all(np.any(q != 0.0, axis=-1))
    for values in (k, v):
        for batch in range(case.batch):
            for head in range(case.nhead_k):
                rows = values[batch, :, head, :]
                assert len(np.unique(rows, axis=0)) == case.seqlen_k
    # Moderately concentrated attention: unit-scale Q and K, balanced V signs.
    for values in (q, k):
        assert abs(float(values.mean())) < 0.02
        assert 0.85 < float(values.std()) < 1.0
    assert abs(float(v.mean())) < 0.02


def test_inputs_do_not_depend_on_the_dtype_and_stay_pinned() -> None:
    fp16 = permutation_inputs(PERMUTATION_SQUARE_FP16)
    bf16 = permutation_inputs(PERMUTATION_SQUARE_BF16)
    digests = []
    for left, right in zip(fp16, bf16):
        assert np.array_equal(left, right)
        raw = encode_bits(left, "fp16").astype("<u2").tobytes()
        digests.append(hashlib.sha256(raw).hexdigest()[:16])
    assert digests == _INPUT_DIGESTS


# First 16 hex digits of the SHA-256 of the FP16 little-endian Q, K and V bits
# of the quick shape. The inputs are generated from an explicit integer hash; a
# change here changes what every launch of the lane reads.
_INPUT_DIGESTS = [
    "3c90c69242d0c3c6",
    "ff9062699a312ba1",
    "a83610710c22742d",
]


def test_case_scope_refuses_what_the_bound_does_not_cover() -> None:
    for case in PERMUTATION_QUICK_CASES:
        case.require_scope()
        assert case.mask == "none" and case.window == 0
    # Every quick map must exist; cross_tile needs more than 16 key tokens.
    for seqlen_k in (1, 2, CANONICAL_SPAN):
        with pytest.raises(ValueError, match="more than 16 key tokens"):
            PermutationCase(1, 4, seqlen_k, 2, 2, 8, "fp16").require_scope()
    PermutationCase(1, 4, CANONICAL_SPAN + 1, 2, 2, 8, "fp16").require_scope()
    with pytest.raises(ValueError, match="exact in FP32"):
        PermutationCase(1, 4, 32, 2, 2, 1 << 15, "fp16").require_scope()
    with pytest.raises(ValueError, match="multiple of nhead_k"):
        PermutationCase(1, 4, 4, 3, 2, 8, "fp16")
    with pytest.raises(ValueError, match="dtype must be one of"):
        PermutationCase(1, 4, 4, 2, 2, 8, "fp32")
    assert len({case.case_id for case in PERMUTATION_QUICK_CASES}) == len(
        PERMUTATION_QUICK_CASES
    )


# ---------------------------------------------------------------------------
# Bound
# ---------------------------------------------------------------------------


def _quick_bound(case: PermutationCase, block_n: int = _MODEL_BLOCK_N):
    q, k, v = permutation_inputs(case)
    return permutation_case_bound(
        case,
        q,
        k,
        v,
        key_tiles=-(-case.seqlen_k // block_n),
        scale=softmax_scale(case),
    )


@pytest.mark.parametrize(
    "case,residual,magnitude",
    [
        (PERMUTATION_SQUARE_FP16, 2.136950539635555e-3, 1.001068475269818),
        (PERMUTATION_SQUARE_BF16, 1.5794141335453805e-2, 1.007897070667727),
    ],
    ids=["fp16", "bf16"],
)
def test_bound_pinned_values_at_the_quick_shape(
    case: PermutationCase, residual: float, magnitude: float
) -> None:
    """Four 64-key tiles, sign-valued V. A change here is a change of the
    derivation, never an adjustment to a measurement."""
    bound = _quick_bound(case)
    assert bound.keys == 256 and bound.key_tiles == 4 and bound.value_max == 1.0
    assert bound.residual == pytest.approx(residual, rel=1e-9)
    assert bound.magnitude == pytest.approx(magnitude, rel=1e-12)


@pytest.mark.parametrize("case", PERMUTATION_QUICK_CASES, ids=_case_id)
def test_bound_is_the_probability_cast_plus_the_output_store(
    case: PermutationCase,
) -> None:
    bound = _quick_bound(case)
    half_ulp = 2.0 ** -(_FORMAT[case.dtype][0] + 1)
    # One half-ULP of relative weight error from the probability cast, and one
    # half-ULP of the binade [1, 2): the largest output is just above one.
    assert bound.cast_term == half_ulp
    assert half_ulp < bound.output_term < half_ulp * (1.0 + 2.0**-10)
    assert 0.0 < bound.fp32_term < 0.15 * bound.per_run
    assert bound.per_run == pytest.approx(
        bound.cast_term + bound.output_term + bound.fp32_term, rel=1e-12
    )
    assert bound.residual == math.nextafter(2.0 * bound.per_run, math.inf)
    assert 1.0 + half_ulp < bound.magnitude < 1.0 + 3.0 * half_ulp


def test_bound_output_term_follows_the_binade_of_the_largest_output() -> None:
    args = {"keys": 256, "key_tiles": 4, "score_spread": 32.0}
    unit = permutation_bound("fp16", value_max=1.0, **args)
    below = permutation_bound("fp16", value_max=0.96875, **args)
    assert unit.output_term == pytest.approx(2.0**-11, rel=1e-3)
    assert below.output_term == pytest.approx(2.0**-12, rel=1e-3)


def test_bound_grows_with_every_input_statistic() -> None:
    args = {"keys": 256, "key_tiles": 4, "value_max": 1.0, "score_spread": 32.0}
    reference = permutation_bound("fp16", **args).residual
    for name, value in (
        ("keys", 512),
        ("key_tiles", 8),
        ("value_max", 1.5),
        ("score_spread", 64.0),
    ):
        assert permutation_bound("fp16", **{**args, name: value}).residual > reference
    assert permutation_bound("bf16", **args).residual > reference


@pytest.mark.parametrize(
    "override,match",
    [
        ({"keys": 1, "key_tiles": 1}, "at least two keys"),
        ({"key_tiles": 0}, "key_tiles"),
        ({"key_tiles": 257}, "key_tiles"),
        ({"value_max": 0.0}, "value_max"),
        ({"value_max": 4096.0}, "value_max"),
        ({"score_spread": -1.0}, "score_spread"),
        ({"score_spread": math.inf}, "score_spread"),
        ({"keys": 1 << 23, "key_tiles": 1}, "too many keys"),
    ],
)
def test_bound_refuses_arguments_outside_its_derivation(override, match: str) -> None:
    args = {"keys": 256, "key_tiles": 4, "value_max": 1.0, "score_spread": 32.0}
    with pytest.raises(ValueError, match=match):
        permutation_bound("fp16", **{**args, **override})
    with pytest.raises(ValueError, match="dtype must be one of"):
        permutation_bound("fp32", **args)


def _head_scores(
    case: PermutationCase, q: np.ndarray, k: np.ndarray, *, head: int
) -> np.ndarray:
    """Log2-domain scores ``[Sq, Sk]`` of one query head of batch 0, in FP64."""
    kv_head = head // (case.nhead_q // case.nhead_k)
    products = (
        q[0, :, head, :].astype(np.float64) @ k[0, :, kv_head, :].astype(np.float64).T
    )
    return products * (softmax_scale(case) * _LOG2E)


def _ideal_weights(
    case: PermutationCase, q: np.ndarray, k: np.ndarray, *, head: int
) -> np.ndarray:
    """Normalized ideal weights ``[Sq, Sk]`` of one query head of batch 0:
    ``exp2(s - max)`` on the FP32 scores, evaluated in FP64."""
    kv_head = head // (case.nhead_q // case.nhead_k)
    products = (
        q[0, :, head, :].astype(np.float64) @ k[0, :, kv_head, :].astype(np.float64).T
    )
    scale = np.float32(np.float32(softmax_scale(case)) * np.float32(_LOG2E))
    scores = (products.astype(np.float32) * scale).astype(np.float64)
    weights = np.exp2(scores - scores.max(axis=1, keepdims=True))
    return weights / weights.sum(axis=1, keepdims=True)


def _exact_defect_residual(
    case: PermutationCase,
    inputs: tuple[np.ndarray, np.ndarray, np.ndarray],
    pairing: np.ndarray,
    source: np.ndarray,
) -> np.ndarray:
    """Exact-arithmetic residual ``[Sq, Hq, D]`` between the base and the
    permuted run of a kernel with the fixed pairing defect ``pairing``:
    ``sum_i P_i (V[pairing[i]] - V[conjugated[i]])``, where ``conjugated`` is
    ``source o pairing o source^-1``."""
    q, k, v = inputs
    conjugated = source[pairing[np.argsort(source)]]
    group = case.nhead_q // case.nhead_k
    residual = np.empty(case.output_shape[1:], dtype=np.float64)
    for head in range(case.nhead_q):
        values = v[0, :, head // group, :].astype(np.float64)
        residual[:, head, :] = _ideal_weights(case, q, k, head=head) @ (
            values[pairing] - values[conjugated]
        )
    return residual


@pytest.mark.parametrize("case", PERMUTATION_QUICK_CASES, ids=_case_id)
def test_score_spread_bound_covers_every_row_of_the_fixture(
    case: PermutationCase,
) -> None:
    q, k, _ = permutation_inputs(case)
    scale = softmax_scale(case)
    spread = score_spread_bound(q, k, scale)
    largest = max(
        float(np.max(np.ptp(_head_scores(case, q, k, head=head), axis=1)))
        for head in range(case.nhead_q)
    )
    assert largest < spread < 4.0 * largest

    ones = np.ones((1, 1, 1, 4), dtype=np.float32)
    assert score_spread_bound(ones, 2.0 * ones, 0.5) == pytest.approx(
        2.0 * 2.0 * 4.0 * 0.5 * _LOG2E, rel=1e-5
    )
    with pytest.raises(ValueError, match="scale must be positive"):
        score_spread_bound(ones, ones, 0.0)


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------


def _constant_output(case: PermutationCase, value: float) -> np.ndarray:
    return encode_bits(np.full(case.output_shape, value, dtype=np.float32), case.dtype)


def _with_value(bits: np.ndarray, index, value: float, dtype: str) -> np.ndarray:
    changed = bits.copy()
    changed[index] = encode_bits(np.array([value], dtype=np.float32), dtype)[0]
    return changed


@pytest.mark.parametrize("case", PERMUTATION_QUICK_CASES, ids=_case_id)
def test_analyzer_separates_residuals_at_the_bound(case: PermutationCase) -> None:
    bound = _quick_bound(case)
    base = _constant_output(case, 0.25)
    clean = analyze_permutation(case, bound, base, base.copy())
    assert not clean.violations
    assert clean.max_abs == 0.0 and clean.bound_ratio == 0.0
    # Identical outputs are what an ignored pairing mutation looks like.
    assert len(clean.unresponsive_heads) == case.batch * case.nhead_q

    # The largest power of two inside the bound passes; twice it fails.
    inside = 2.0 ** math.floor(math.log2(bound.residual))
    index = (0, 17, 5, 33)
    passing = analyze_permutation(
        case, bound, base, _with_value(base, index, 0.25 + inside, case.dtype)
    )
    assert not passing.violations
    assert passing.max_abs == inside and passing.worst_coordinate == index
    assert passing.signed_at_worst == -inside

    failing = analyze_permutation(
        case, bound, base, _with_value(base, index, 0.25 - 2.0 * inside, case.dtype)
    )
    assert failing.exceed_count == 1 and failing.worst_coordinate == index
    assert failing.signed_at_worst == 2.0 * inside
    assert any(item.startswith("residual=") for item in failing.violations)
    assert failing.head_exceed_counts[0][5] == 1
    assert (0, 5) not in failing.unresponsive_heads
    assert len(failing.unresponsive_heads) == case.batch * case.nhead_q - 1
    assert "residual=" in format_permutation_result(failing)


@pytest.mark.parametrize("case", PERMUTATION_QUICK_CASES, ids=_case_id)
def test_analyzer_rejects_invalid_and_oversized_outputs(case: PermutationCase) -> None:
    bound = _quick_bound(case)
    base = _constant_output(case, 0.25)
    unwritten = base.copy()
    unwritten[0, 3, 1, 2] = {"fp16": 0x7E00, "bf16": 0x7FC0}[case.dtype]
    unwritten[0, 4, 1, 2] = {"fp16": 0x7C00, "bf16": 0x7F80}[case.dtype]
    result = analyze_permutation(case, bound, base, unwritten)
    assert result.invalid_count == 2 and "invalid=2" in result.violations
    assert result.exceed_count == 0 and result.max_abs == 0.0

    # A blow-up both launches share cancels in the residual; the magnitude guard
    # sees it.
    large = _constant_output(case, 1.5)
    result = analyze_permutation(case, bound, large, large.copy())
    assert result.exceed_count == 0
    assert result.magnitude_count == 2 * large.size
    assert any(item.startswith("magnitude>") for item in result.violations)
    # One representable step above one is still inside the guard.
    top = _constant_output(case, 1.0 + 2.0 ** -_FORMAT[case.dtype][0])
    assert not analyze_permutation(case, bound, top, top.copy()).violations


def test_analyzer_validates_its_inputs() -> None:
    case = PERMUTATION_SQUARE_FP16
    bound = _quick_bound(case)
    base = _constant_output(case, 0.25)
    with pytest.raises(ValueError, match="raw uint16 bits"):
        analyze_permutation(case, bound, base.astype(np.float16), base)
    with pytest.raises(ValueError, match="other output shape"):
        analyze_permutation(case, bound, base, base[:, :-1])
    with pytest.raises(ValueError, match="bound dtype"):
        analyze_permutation(case, _quick_bound(PERMUTATION_SQUARE_BF16), base, base)


# ---------------------------------------------------------------------------
# FP32 model of the single-pass dense kernel
# ---------------------------------------------------------------------------


def _round_to(values: np.ndarray, dtype: str) -> np.ndarray:
    """Round FP32 values to the subject dtype (nearest-even), back in FP32."""
    if dtype == "fp16":
        return values.astype(np.float16).astype(np.float32)
    return bf16_to_f32(f32_to_bf16(values)).reshape(values.shape).astype(np.float32)


def _store(values: np.ndarray, dtype: str) -> np.ndarray:
    if dtype == "fp16":
        return values.astype(np.float16).view(np.uint16)
    return f32_to_bf16(values).reshape(values.shape)


def _model_dense_run(
    case: PermutationCase,
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    *,
    block_n: int = _MODEL_BLOCK_N,
    value_token_offset: int = 0,
    pairing: np.ndarray | None = None,
) -> np.ndarray:
    """Model one launch: per KV tile the running maximum, ``exp2(s - m)`` in FP32,
    the normalizer over uncast weights, the accumulator over weights cast to the
    subject dtype, both rescaled by ``exp2(m_old - m_new)``; then one reciprocal
    and one output conversion.

    ``pairing`` models a fixed pairing defect: the score of key slot ``j`` meets
    the value of slot ``pairing[j]``. ``value_token_offset`` is the cyclic case
    ``pairing[j] = j + offset``."""
    out = np.empty(case.output_shape, dtype=np.uint16)
    group = case.nhead_q // case.nhead_k
    qk_scale = np.float32(np.float32(softmax_scale(case)) * np.float32(_LOG2E))
    if pairing is None:
        pairing = (np.arange(case.seqlen_k) + value_token_offset) % case.seqlen_k
    for batch in range(case.batch):
        for head in range(case.nhead_q):
            keys = k[batch, :, head // group, :].astype(np.float64)
            values = v[batch, :, head // group, :][pairing]
            scores = (q[batch, :, head, :].astype(np.float64) @ keys.T).astype(
                np.float32
            ) * qk_scale
            running_max = np.full(case.seqlen_q, -1e30, dtype=np.float32)
            normalizer = np.zeros(case.seqlen_q, dtype=np.float32)
            accumulator = np.zeros((case.seqlen_q, case.head_dim), dtype=np.float32)
            for first in range(0, case.seqlen_k, block_n):
                tile = scores[:, first : first + block_n]
                new_max = np.maximum(running_max, tile.max(axis=1))
                rescale = np.exp2(running_max - new_max)
                weights = np.exp2(tile - new_max[:, None])
                normalizer = normalizer * rescale + weights.sum(
                    axis=1, dtype=np.float32
                )
                accumulator = accumulator * rescale[:, None] + (
                    _round_to(weights, case.dtype) @ values[first : first + block_n]
                )
                running_max = new_max
            reciprocal = np.float32(1.0) / normalizer
            out[batch, :, head, :] = _store(
                (accumulator * reciprocal[:, None]).astype(np.float32), case.dtype
            )
    return out


def _model_pair(
    case: PermutationCase,
    key_map: np.ndarray,
    value_map: np.ndarray,
    *,
    block_n: int = _MODEL_BLOCK_N,
    value_token_offset: int = 0,
    pairing: np.ndarray | None = None,
    inputs: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> PermutationResult:
    """Model a base launch and a transformed one and analyze the pair against
    the bound evaluated for the same inputs and tile."""
    q, k, v = permutation_inputs(case) if inputs is None else inputs
    bound = permutation_case_bound(
        case,
        q,
        k,
        v,
        key_tiles=-(-case.seqlen_k // block_n),
        scale=softmax_scale(case),
    )
    settings = {
        "block_n": block_n,
        "value_token_offset": value_token_offset,
        "pairing": pairing,
    }
    base = _model_dense_run(case, q, k, v, **settings)
    other = _model_dense_run(
        case, q, permute_tokens(k, key_map), permute_tokens(v, value_map), **settings
    )
    return analyze_permutation(case, bound, base, other)


@pytest.mark.parametrize("block_n", [_MODEL_BLOCK_N, _MODEL_BLOCK_N // 2])
@pytest.mark.parametrize("case", PERMUTATION_QUICK_CASES, ids=_case_id)
def test_modeled_joint_permutations_stay_inside_the_bound(
    case: PermutationCase, block_n: int
) -> None:
    for kind in PERMUTATION_QUICK_MAPS:
        source = token_map(kind, case.seqlen_k)
        result = _model_pair(case, source, source, block_n=block_n)
        assert not result.violations, (kind.value, format_permutation_result(result))
        assert result.bound_ratio < 1.0


def _late_maximum_inputs(
    case: PermutationCase, level: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Nonnegative Q and a last key that is ``level`` in every dimension, so that
    key holds every row's largest score. The base launch meets the maximum in
    its last tile and the reversed launch in its first."""
    q, k, v = permutation_inputs(case)
    k = k.copy()
    k[:, -1] = level
    return np.abs(q), k, v


def _sharp_inputs(
    case: PermutationCase,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Q doubled and clipped to the lattice range: a wider logit spread."""
    q, k, v = permutation_inputs(case)
    return np.clip(2.0 * q, -3.0, 3.0).astype(np.float32), k, v


# Inputs away from the fixture. At level 1 the last key holds about half of a
# row's probability; at level 1.5 nearly all of it, and a third of the other
# weights fall below the FP16 normal range.
_ADVERSARIAL_INPUTS = {
    "late-maximum": lambda case: _late_maximum_inputs(case, 1.0),
    "dominant-late-key": lambda case: _late_maximum_inputs(case, 1.5),
    "sharp": _sharp_inputs,
}


@pytest.mark.parametrize("profile", sorted(_ADVERSARIAL_INPUTS))
@pytest.mark.parametrize("case", PERMUTATION_QUICK_CASES, ids=_case_id)
def test_modeled_adversarial_rows_stay_inside_their_bound(
    case: PermutationCase, profile: str
) -> None:
    """The bound is evaluated for the adversarial inputs, not for the fixture:
    a wider score spread widens it only through the FP32 remainder."""
    inputs = _ADVERSARIAL_INPUTS[profile](case)
    for values in inputs:
        encode_bits(values, case.dtype)
    if profile != "sharp":
        scores = _head_scores(case, inputs[0], inputs[1], head=0)
        assert np.all(np.argmax(scores, axis=1) == case.seqlen_k - 1)
        weights = np.exp2(scores - scores.max(axis=1, keepdims=True))
        subnormal = float(np.mean(weights < 2.0**-14))
        assert (subnormal > 0.25) == (profile == "dominant-late-key")
    fixture = _quick_bound(case)
    for kind in PERMUTATION_QUICK_MAPS:
        source = token_map(kind, case.seqlen_k)
        result = _model_pair(case, source, source, inputs=inputs)
        assert not result.violations, (kind.value, format_permutation_result(result))
        assert fixture.residual <= result.bound < 1.1 * fixture.residual
    # The mutations stay visible on concentrated rows.
    key_map, value_map = mutated_token_maps(
        PairingMutation.KEY_ONLY, PERMUTATION_MUTATION_MAP, case.seqlen_k
    )
    assert not _model_pair(case, key_map, value_map, inputs=inputs).unresponsive_heads


def _single_slot_defect(case: PermutationCase) -> np.ndarray:
    """A fixed pairing defect in one slot: key slot 5 meets the value of slot
    200. Every other slot is paired correctly."""
    pairing = np.arange(case.seqlen_k)
    pairing[5] = 200
    return pairing


def _stored_residual(
    case: PermutationCase,
    inputs: tuple[np.ndarray, np.ndarray, np.ndarray],
    pairing: np.ndarray,
    source: np.ndarray,
) -> np.ndarray:
    """Stored base-minus-permuted residual ``[Sq, Hq, D]`` of the modeled kernel
    with the pairing defect."""
    q, k, v = inputs
    base = _model_dense_run(case, q, k, v, pairing=pairing)
    other = _model_dense_run(
        case, q, permute_tokens(k, source), permute_tokens(v, source), pairing=pairing
    )
    return (decode_bits(base, case.dtype) - decode_bits(other, case.dtype))[0]


@pytest.mark.parametrize("case", PERMUTATION_QUICK_CASES, ids=_case_id)
def test_fixed_pairing_defect_residual_follows_the_conjugated_map(
    case: PermutationCase,
) -> None:
    """What the relation measures on a faulty kernel is the difference of two
    faulty runs, not the distance to a correct output. A defect in slot ``a``
    moves key ``a`` in the base run and the key that sits in slot ``a`` in the
    permuted run; the two contributions add or cancel. The stored residual is
    within the bound of that exact residual, so a defect is certainly visible
    only when its exact residual exceeds twice the bound."""
    inputs = permutation_inputs(case)
    bound = _quick_bound(case)
    pairing = _single_slot_defect(case)
    values = inputs[2][0].astype(np.float64)
    for kind in PERMUTATION_QUICK_MAPS:
        source = token_map(kind, case.seqlen_k)
        conjugated = source[pairing[np.argsort(source)]]
        moved = np.flatnonzero(pairing != conjugated).tolist()
        exact = _exact_defect_residual(case, inputs, pairing, source)
        stored = _stored_residual(case, inputs, pairing, source)
        assert np.max(np.abs(stored - exact)) <= bound.residual
        result = _model_pair(case, source, source, pairing=pairing)
        if kind is TokenMap.ADJACENT_SWAP:
            # Slots 5 and 200 are fixed by this map: the defect commutes with it.
            assert moved == [] and not np.any(exact)
            assert not result.violations
            continue
        assert moved == sorted({5, int(source[5])})
        assert np.max(np.abs(exact)) > 2.0 * bound.residual
        assert result.violations, (kind.value, format_permutation_result(result))
        if kind is TokenMap.SHIFT_ONE:
            # Head 0: the base-run term alone, against the two-term residual.
            first = _ideal_weights(case, inputs[0], inputs[1], head=0)[:, 5:6] * (
                values[200, 0] - values[5, 0]
            )
            assert np.any(np.abs(exact[:, 0]) < np.abs(first))
            assert np.any(np.abs(exact[:, 0]) > np.abs(first))


def test_pairing_defect_on_low_weight_keys_stays_inside_the_bound() -> None:
    """The limit of the relation. On rows where one key holds nearly all the
    probability, the same single-slot defect moves the BF16 output by less than
    the bound under every map and passes; the FP16 bound is tight enough to
    reject it."""
    for case, visible in (
        (PERMUTATION_SQUARE_BF16, False),
        (PERMUTATION_SQUARE_FP16, True),
    ):
        inputs = _ADVERSARIAL_INPUTS["dominant-late-key"](case)
        pairing = _single_slot_defect(case)
        for kind in (TokenMap.SHIFT_ONE, TokenMap.REVERSE, TokenMap.CROSS_TILE):
            source = token_map(kind, case.seqlen_k)
            result = _model_pair(case, source, source, pairing=pairing, inputs=inputs)
            exact = np.max(
                np.abs(_exact_defect_residual(case, inputs, pairing, source))
            )
            assert exact > 0.0
            assert bool(result.violations) == visible, (case.dtype, kind.value)
            assert (exact > 2.0 * result.bound) == visible


def _argmax_value_outputs(
    case: PermutationCase, q: np.ndarray, k: np.ndarray, v: np.ndarray
) -> np.ndarray:
    """Outputs of a kernel that returns the value of each row's highest-scoring
    key, averaged over ties: a pairing that depends on the inputs and moves with
    them. Breaking ties by slot index instead would not move with them."""
    out = np.empty(case.output_shape, dtype=np.float32)
    group = case.nhead_q // case.nhead_k
    for head in range(case.nhead_q):
        scores = _head_scores(case, q, k, head=head)
        best = (scores == scores.max(axis=1, keepdims=True)).astype(np.float64)
        values = v[0, :, head // group, :].astype(np.float64)
        out[0, :, head, :] = (best @ values) / best.sum(axis=1, keepdims=True)
    return _store(out, case.dtype)


@pytest.mark.parametrize("case", PERMUTATION_QUICK_CASES, ids=_case_id)
def test_equivariant_pairing_defect_is_outside_the_relation(
    case: PermutationCase,
) -> None:
    """The separation argument covers fixed slot maps only. A wrong pairing that
    is chosen from the inputs and is carried along by every permutation, here
    "the value of the highest-scoring key", satisfies every joint permutation
    exactly and responds to every mutation: the lane passes it."""
    q, k, v = permutation_inputs(case)
    bound = _quick_bound(case)
    base = _argmax_value_outputs(case, q, k, v)
    for kind in PERMUTATION_QUICK_MAPS:
        source = token_map(kind, case.seqlen_k)
        other = _argmax_value_outputs(
            case, q, permute_tokens(k, source), permute_tokens(v, source)
        )
        result = analyze_permutation(case, bound, base, other)
        assert not result.violations and result.max_abs == 0.0
    for mutation in PairingMutation:
        key_map, value_map = mutated_token_maps(
            mutation, PERMUTATION_MUTATION_MAP, case.seqlen_k
        )
        other = _argmax_value_outputs(
            case, q, permute_tokens(k, key_map), permute_tokens(v, value_map)
        )
        assert not analyze_permutation(case, bound, base, other).unresponsive_heads


@pytest.mark.parametrize("mutation", list(PairingMutation))
@pytest.mark.parametrize("case", PERMUTATION_QUICK_CASES, ids=_case_id)
def test_modeled_pairing_mutations_reach_every_head(
    case: PermutationCase, mutation: PairingMutation
) -> None:
    key_map, value_map = mutated_token_maps(
        mutation, PERMUTATION_MUTATION_MAP, case.seqlen_k
    )
    result = _model_pair(case, key_map, value_map)
    assert result.violations
    assert not result.unresponsive_heads, format_permutation_result(result)
    # Far beyond the bound, not marginally: the verdict does not hinge on it.
    assert result.bound_ratio > 10.0


@pytest.mark.parametrize("case", PERMUTATION_QUICK_CASES, ids=_case_id)
def test_modeled_pairing_defects_need_the_whole_map_set(case: PermutationCase) -> None:
    """A value read one tile ahead commutes with every cyclic shift, and one read
    half a sequence ahead also commutes with the reverse. Each defect is caught,
    but not by the same map."""

    def failing_maps(offset: int) -> set[TokenMap]:
        failed = set()
        for kind in PERMUTATION_QUICK_MAPS:
            source = token_map(kind, case.seqlen_k)
            if _model_pair(case, source, source, value_token_offset=offset).violations:
                failed.add(kind)
        return failed

    assert failing_maps(_MODEL_BLOCK_N) == {TokenMap.REVERSE, TokenMap.ADJACENT_SWAP}
    assert failing_maps(case.seqlen_k // 2) == {TokenMap.ADJACENT_SWAP}


def _result(*, exceed: bool, heads: tuple[tuple[int, ...], ...]) -> PermutationResult:
    return PermutationResult(
        violations=("residual=0.5>0.002 at 9",) if exceed else (),
        invalid_count=0,
        magnitude_count=0,
        exceed_count=9 if exceed else 0,
        head_exceed_counts=heads,
        max_abs=0.5 if exceed else 0.0,
        worst_coordinate=(0, 0, 0, 0),
        signed_at_worst=0.5 if exceed else 0.0,
        mean_signed=0.0,
        rms=0.0,
        bound=0.002,
    )


def test_unresponsive_mutation_blames_failing_joint_permutations() -> None:
    """A mutation that stays inside the bound on some head is a kernel finding.
    When the joint permutations fail too, their violations are the diagnosis."""
    mutated = _result(exceed=True, heads=((9, 0),))
    assert mutated.unresponsive_heads == ((0, 1),)
    clean = {"reverse": _result(exceed=False, heads=((0, 0),))}
    message = _unresponsive("label", "key_only", mutated, clean)
    assert "b0:h1 did not respond to key_only" in message
    assert "ignores the key-value pairing" in message

    failing = {"reverse": _result(exceed=True, heads=((9, 9),))}
    message = _unresponsive("label", "key_only", mutated, failing)
    assert "joint permutations themselves fail" in message
    assert "reverse: violations=residual=0.5>0.002 at 9" in message


# ---------------------------------------------------------------------------
# Lowered arithmetic path of the unmasked specifications
# ---------------------------------------------------------------------------

_REDERIVE = "re-derive the bound"

# LLVM spells FP32 constants as the hex of the equivalent double.
_F32_LOG2E = "0x3FF7154760000000"
_F32_ONE = "0x3FF0000000000000"
_F32_ZERO = "0x0000000000000000"
_F32_INITIAL_MAX = "0xC6293E5940000000"  # -1e30

_MFMA_ATOM = {
    "fp16": "llvm.amdgcn.mfma.f32.32x32x8f16",
    "bf16": "llvm.amdgcn.mfma.f32.32x32x8bf16.1k",
}
_CAST_TYPE = {"fp16": "half", "bf16": "bfloat"}
_EXP2 = "llvm.amdgcn.exp2.f32"
_MAXNUM = "llvm.maxnum.f32"
_BPERMUTE = "llvm.amdgcn.ds.bpermute"
_FP_FLAGS = ("fast", "contract", "afn", "reassoc", "nnan", "ninf", "nsz", "arcp")
_FP_MODE_ATTRIBUTES = (
    "denormal-fp-math",
    "unsafe-fp-math",
    "no-nans-fp-math",
    "no-infs-fp-math",
    "approx-func-fp-math",
    "no-signed-zeros-fp-math",
)
_FP_TYPE = re.compile(r"\b(float|half|bfloat|double)\b")
_INSTRUCTION = re.compile(r"^\s*(%[-\w.]+) = (.+)$")
_VALUE = re.compile(r"%[-\w.]+")
_CALLEE = re.compile(r"@([-\w.]+)\(")
_DECLARE = re.compile(r"^declare (.+?) @([-\w.]+)\((.*)\)")
_NO_OP_BITCAST = re.compile(r"bitcast (\S+(?: x \S+>)?) %[-\w.]+ to (\S+(?: x \S+>)?)$")


@dataclass(frozen=True)
class _Instruction:
    name: str
    opcode: str
    text: str
    operands: tuple[str, ...]


def _parse(ir: str) -> dict[str, _Instruction]:
    """Map each SSA name to its defining instruction. ``opcode`` is the first
    word, or the callee for a call; ``operands`` are the SSA values it reads."""
    instructions: dict[str, _Instruction] = {}
    for line in ir.splitlines():
        match = _INSTRUCTION.match(line)
        if not match:
            continue
        name, text = match.groups()
        if re.match(r"(?:tail |musttail |notail )?call ", text):
            callee = _CALLEE.search(text)
            opcode = callee.group(1) if callee else "call"
            arguments = text[callee.end() :] if callee else text
        else:
            opcode = text.split()[0]
            arguments = text[len(opcode) :]
        instructions[name] = _Instruction(
            name, opcode, text, tuple(_VALUE.findall(arguments))
        )
    return instructions


def _classify_unmasked_path(ir: str, dtype: str, block_n: int) -> tuple[list, Counter]:
    """Classify every floating-point instruction of one lowered unmasked kernel
    against the path the bound is derived for, and return the violations and
    the per-step instruction counts.

    The path, per key tile and lane: scores ``fmul(acc, scale * LOG2E)`` with no
    mask; ``exp2(fsub(x, m_new))``; each probability feeds the normalizer chain
    uncast and the PV operand through one ``fptrunc``; one ``alpha`` rescales
    the normalizer and every accumulator element; the key-tile loop runs
    ``seqlen_kv / block_n`` times; the epilogue is one ``fdiv 1.0, l`` and
    ``fptrunc(fmul(o, rcp))``. No fast-math flag, ``!fpmath`` or floating-point
    mode attribute may appear."""
    violations: list[str] = []
    counts: Counter = Counter()
    instructions = _parse(ir)
    uses: dict[str, list[_Instruction]] = {}
    for instruction in instructions.values():
        for operand in instruction.operands:
            uses.setdefault(operand, []).append(instruction)

    def source(name: str) -> str:
        """Follow type-preserving ``bitcast`` copies."""
        while name in instructions:
            item = instructions[name]
            no_op = _NO_OP_BITCAST.match(item.text)
            if not (no_op and no_op.group(1) == no_op.group(2)):
                break
            name = item.operands[0]
        return name

    def opcode(name: str) -> str | None:
        item = instructions.get(source(name))
        return item.opcode if item else None

    def users(name: str) -> list[_Instruction]:
        found: list[_Instruction] = []
        for user in uses.get(name, []):
            if user.opcode == "bitcast" and source(user.name) == source(name):
                found.extend(users(user.name))
            else:
                found.append(user)
        return found

    for line in ir.splitlines():
        stripped = line.strip()
        match = re.search(r"= (?:f\w+|call|select|phi) ((?:\w+ )*)", stripped)
        if match and set(match.group(1).split()) & set(_FP_FLAGS):
            violations.append(f"fast-math flags: {stripped}")
        if "!fpmath" in stripped:
            violations.append(f"!fpmath metadata: {stripped}")
        if stripped.startswith("attributes #"):
            for attribute in _FP_MODE_ATTRIBUTES:
                if f'"{attribute}' in stripped:
                    violations.append(f"FP mode attribute {attribute}: {stripped}")
        declared = _DECLARE.match(stripped)
        if declared:
            returned, callee, arguments = declared.groups()
            modeled = callee in (_EXP2, _MAXNUM, _MFMA_ATOM[dtype])
            if not modeled and (
                _FP_TYPE.search(returned) or _FP_TYPE.search(arguments)
            ):
                violations.append(f"unmodeled floating-point intrinsic @{callee}")

    # Loop-carried state: the running maximum and the normalizer. No mask.
    running_max = normalizer = None
    for item in instructions.values():
        zeroed = item.text.count("zeroinitializer") == 2
        if item.opcode == "select" and _FP_TYPE.search(item.text) and not zeroed:
            violations.append(f"float select on an unmasked path: {item.text}")
        if item.opcode == "phi" and item.text.startswith("phi float"):
            initial = re.findall(r"\[ (0x[0-9A-F]+),", item.text)
            if initial == [_F32_INITIAL_MAX] and running_max is None:
                running_max = item.name
            elif initial == [_F32_ZERO] and normalizer is None:
                normalizer = item.name
            else:
                violations.append(f"unmodeled float phi: {item.text}")
    if running_max is None or normalizer is None:
        violations.append("no loop-carried running maximum or normalizer")

    # The key-tile loop visits every tile: kt in [0, seqlen_kv / block_n).
    compares = [item for item in instructions.values() if item.opcode == "icmp"]
    bound = None
    if len(compares) == 1:
        match = re.fullmatch(r"icmp slt i32 (%[-\w.]+), (%[-\w.]+)", compares[0].text)
        if match:
            counter, bound = instructions.get(match.group(1)), match.group(2)
            start = re.match(
                r"phi i32 \[ 0, %[-\w.]+ \], \[ (%[-\w.]+), ", counter.text
            )
            step = instructions.get(start.group(1)) if start else None
            if step is None or step.text != f"add nsw i32 {counter.name}, 1":
                violations.append(f"key-tile counter is not 0, 1, ...: {counter.text}")
    if bound is None or instructions[bound].text != f"sdiv i32 %seqlen_kv, {block_n}":
        violations.append(
            f"key-tile loop bound is not seqlen_kv / {block_n}: "
            f"{[item.text for item in compares]}"
        )

    scale = [
        item.name
        for item in instructions.values()
        if item.text
        in (f"fmul float %scale, {_F32_LOG2E}", f"fmul float {_F32_LOG2E}, %scale")
    ]
    if len(scale) != 1:
        violations.append(f"expected one fmul(scale, LOG2E), found {len(scale)}")
    reciprocal = [item for item in instructions.values() if item.opcode == "fdiv"]
    if (
        len(reciprocal) != 1
        or not reciprocal[0].text.startswith(f"fdiv float {_F32_ONE}, %")
        or source(reciprocal[0].operands[0]) != normalizer
    ):
        violations.append(
            f"expected one fdiv 1.0, l; found {[item.text for item in reciprocal]}"
        )
    rcp = reciprocal[0].name if reciprocal else None

    # exp2: a probability (normalizer fadd + one cast) or the rescale alpha.
    probabilities: set[str] = set()
    alphas: set[str] = set()
    for item in instructions.values():
        if item.opcode == "fsub" and {u.opcode for u in users(item.name)} != {_EXP2}:
            violations.append(f"fsub not consumed by exp2 only: {item.text}")
        if item.opcode != _EXP2:
            continue
        argument = instructions.get(source(item.operands[0]))
        if argument is None or argument.opcode != "fsub":
            violations.append(f"exp2 argument is not an fsub: {item.text}")
        elif opcode(argument.operands[1]) != _MAXNUM:
            violations.append(f"exp2 does not subtract the row maximum: {item.text}")
        kinds = Counter(user.opcode for user in users(item.name))
        if kinds == Counter({"fadd": 1, "fptrunc": 1}):
            probabilities.add(item.name)
        elif kinds and set(kinds) == {"fmul"}:
            alphas.add(item.name)
        else:
            violations.append(f"exp2 with unmodeled uses {dict(kinds)}: {item.text}")
    counts["exp2_probability"] = len(probabilities)
    counts["exp2_alpha"] = len(alphas)

    def element_of(name: str, producer: str) -> bool:
        item = instructions.get(source(name))
        return (
            item is not None
            and item.opcode == "extractelement"
            and opcode(item.operands[0]) == producer
        )

    for item in instructions.values():
        if item.opcode != "fmul" or item.name in scale:
            continue
        left, right = item.operands
        if right in scale and element_of(left, _MFMA_ATOM[dtype]):
            counts["score"] += 1
        elif source(right) in alphas and source(left) == normalizer:
            counts["rescale_normalizer"] += 1
        elif source(right) in alphas and element_of(left, "phi"):
            counts["rescale_accumulator"] += 1
        elif right == rcp and {u.opcode for u in users(item.name)} == {"fptrunc"}:
            counts["output_product"] += 1
        else:
            violations.append(f"unmodeled fmul: {item.text}")

    # Normalizer: a lane chain of uncast probabilities from +0, the partner
    # lane's sum, then l * alpha + tile sum.
    chain: dict[str, int] = {}
    shuffles: set[str] = set()
    for item in instructions.values():
        if item.opcode != "fadd":
            continue
        left, right = item.operands[0], item.operands[-1]
        rescaled = instructions.get(source(left))
        if source(right) in probabilities and (
            item.text.startswith(f"fadd float {_F32_ZERO}, ") or source(left) in chain
        ):
            chain[item.name] = chain.get(source(left), 0) + 1
        elif source(left) in chain and opcode(right) == "bitcast":
            shuffles.add(item.name)
            counts["lane_chain"] = chain[source(left)]
            if opcode(instructions[source(right)].operands[0]) != _BPERMUTE:
                violations.append(f"partner sum is not a lane shuffle: {item.text}")
        elif (
            rescaled is not None
            and rescaled.opcode == "fmul"
            and source(rescaled.operands[0]) == normalizer
            and source(right) in shuffles
        ):
            counts["combine"] += 1
            if normalizer not in {user.name for user in users(item.name)}:
                violations.append(f"combine is not the next normalizer: {item.text}")
        else:
            violations.append(f"unmodeled fadd: {item.text}")
    counts["shuffle"] = len(shuffles)

    cast_type = _CAST_TYPE[dtype]
    for item in instructions.values():
        if item.opcode == "fpext":
            violations.append(f"unmodeled fpext: {item.text}")
        if item.opcode != "fptrunc":
            continue
        if not item.text.endswith(f" to {cast_type}"):
            violations.append(f"cast to a type other than {cast_type}: {item.text}")
        operand = instructions.get(source(item.operands[0]))
        if operand is not None and operand.name in probabilities:
            counts["cast_probability"] += 1
        elif (
            operand is not None and operand.opcode == "fmul" and rcp in operand.operands
        ):
            counts["cast_output"] += 1
        else:
            violations.append(f"unmodeled cast: {item.text}")

    for item in instructions.values():
        if item.opcode == _MFMA_ATOM[dtype]:
            counts["mfma"] += 1
            if not item.text.endswith("i32 0, i32 0, i32 0)"):
                violations.append(f"MFMA with nonzero controls: {item.text}")
    return violations, counts


_INSERT = re.compile(
    r"insertelement <(\d+) x \w+> (undef|%[-\w.]+), \w+ (%[-\w.]+), i32 (\d+)$"
)
_EXTRACT = re.compile(r"extractelement <\d+ x \w+> (%[-\w.]+), i32 (\d+)$")
_LDS_FILL = re.compile(
    r"call void @llvm\.amdgcn\.raw\.ptr\.buffer\.load\.lds\("
    r"ptr addrspace\(8\) (%[-\w.]+), ptr addrspace\(3\) (%[-\w.]+),"
)
_LDS_ARRAY = re.compile(
    r"getelementptr inbounds \[1 x \[(\d+) x \[(\d+) x \w+\]\]\], "
    r"ptr addrspace\(3\) (@[-\w.]+|%[-\w.]+),"
)
_LDS_OFFSET = re.compile(
    r"getelementptr inbounds i8, ptr addrspace\(3\) @[-\w.]+, i32 (\d+)$"
)
_LDS_BASE = re.compile(r"ptrtoint ptr addrspace\(3\) (@[-\w.]+|%[-\w.]+) to i64$")
_BUFFER = "llvm.amdgcn.make.buffer.rsrc.p1"


def _trace_mfma_operands(ir: str, dtype: str) -> tuple[list, Counter]:
    """Trace every MFMA operand of one lowered unmasked kernel back to the
    tensor it is loaded from, and measure every chain of MFMA calls.

    The score chains start from a zero vector, multiply K tiles read from the K
    region of the LDS by Q read from ``q_ptr``, and end in the 16 scores of a
    lane. The value chains start from the accumulator rescaled by ``alpha``,
    multiply V read from the V region of the LDS by the cast probabilities,
    each of which enters every chain once, and end in the accumulator the loop
    carries. The K region is filled from ``k_ptr`` only and the V region from
    ``v_ptr`` only. Which token lands in which LDS row is index arithmetic and
    is not traced; the relation itself tests it on the device."""
    violations: list[str] = []
    counts: Counter = Counter()
    instructions = _parse(ir)
    uses: dict[str, list[_Instruction]] = {}
    for instruction in instructions.values():
        for operand in instruction.operands:
            uses.setdefault(operand, []).append(instruction)

    def through(name: str) -> str:
        """Follow every ``bitcast``, type-changing ones included."""
        while name in instructions and instructions[name].opcode == "bitcast":
            name = instructions[name].operands[0]
        return name

    def users(name: str) -> list[_Instruction]:
        found: list[_Instruction] = []
        for user in uses.get(name, []):
            found.extend(users(user.name) if user.opcode == "bitcast" else [user])
        return found

    def elements(name: str) -> list[str] | None:
        """The values at positions ``0..n-1`` of an ``insertelement`` chain."""
        found: dict[int, str] = {}
        width = None
        name = through(name)
        while name != "undef":
            item = instructions.get(name)
            match = _INSERT.match(item.text) if item else None
            if match is None:
                return None
            width, index = int(match.group(1)), int(match.group(4))
            if index in found:
                return None
            found[index] = through(match.group(3))
            name = match.group(2)
            if name != "undef":
                name = through(name)
        if width is None or sorted(found) != list(range(width)):
            return None
        return [found[index] for index in range(width)]

    def scalar(name: str) -> str:
        """Follow ``extractelement`` of an ``insertelement`` chain to the value."""
        name = through(name)
        while name in instructions:
            match = _EXTRACT.match(instructions[name].text)
            packed = elements(match.group(1)) if match else None
            if packed is None or int(match.group(2)) >= len(packed):
                break
            name = packed[int(match.group(2))]
        return name

    def lds_offset(base: str) -> int | None:
        """Byte offset of an LDS base inside the pool: the pool itself, or an
        ``i8`` offset from it."""
        if base.startswith("@"):
            return 0
        item = instructions.get(base)
        match = _LDS_OFFSET.match(item.text) if item else None
        return int(match.group(1)) if match else None

    def loaded_from(name: str) -> tuple[str, object]:
        """``("lds", (offset, bytes of the array))`` or ``("global", pointer)``
        for a ``load``; ``("other", text)`` otherwise."""
        item = instructions.get(through(name))
        if item is None or item.opcode != "load" or not item.operands:
            return ("other", name if item is None else item.text)
        address = instructions.get(item.operands[0])
        if address is None or address.opcode != "getelementptr":
            return ("other", item.text)
        if "ptr addrspace(3)" in item.text:
            array = _LDS_ARRAY.match(address.text)
            offset = lds_offset(array.group(3)) if array else None
            if offset is None:
                return ("other", address.text)
            return ("lds", (offset, 2 * int(array.group(1)) * int(array.group(2))))
        match = re.search(r"ptr addrspace\(1\) (%[-\w.]+),", address.text)
        return ("global", match.group(1)) if match else ("other", address.text)

    # Chains: an MFMA call continues the chain of its accumulator operand.
    atom = _MFMA_ATOM[dtype]
    calls = [item for item in instructions.values() if item.opcode == atom]
    successor: dict[str, list[str]] = {}
    starts: list[_Instruction] = []
    for call in calls:
        previous = through(call.operands[2])
        if previous in instructions and instructions[previous].opcode == atom:
            successor.setdefault(previous, []).append(call.name)
        else:
            starts.append(call)
    chains: list[list[_Instruction]] = []
    for start in starts:
        chain = [start]
        while len(successor.get(chain[-1].name, [])) == 1:
            chain.append(instructions[successor[chain[-1].name][0]])
        if successor.get(chain[-1].name):
            violations.append(f"MFMA result feeds two MFMA calls: {chain[-1].text}")
        chains.append(chain)
    if sum(len(chain) for chain in chains) != len(calls):
        violations.append("an MFMA call is in no chain that starts outside the MFMA")

    probabilities = {
        item.name
        for item in instructions.values()
        if item.opcode == "fptrunc"
        and through(item.operands[0]) in instructions
        and instructions[through(item.operands[0])].opcode == _EXP2
    }
    alphas = {
        item.name
        for item in instructions.values()
        if item.opcode == _EXP2
        and {user.opcode for user in users(item.name)} == {"fmul"}
    }
    k_region: set[tuple[int, int]] = set()
    v_region: set[tuple[int, int]] = set()
    packs_by_step: dict[int, set[tuple[str, ...]]] = {}
    for chain in chains:
        start = instructions.get(through(chain[0].operands[2]))
        if start is not None and start.text.count("zeroinitializer") == 2:
            # Score chain: K from the LDS times Q from q_ptr, from zero.
            counts["score_chains"] += 1
            counts[f"score_chain_length_{len(chain)}"] += 1
            for call in chain:
                kind, where = loaded_from(call.operands[0])
                if kind == "lds":
                    k_region.add(where)
                else:
                    violations.append(
                        f"score MFMA does not read K from the LDS: {where}"
                    )
                if loaded_from(call.operands[1]) != ("global", "%q_ptr"):
                    violations.append(
                        f"score MFMA does not read Q from q_ptr: {call.text}"
                    )
            ends = users(chain[-1].name)
            indices = sorted(
                int(match.group(2))
                for match in (_EXTRACT.match(user.text) for user in ends)
                if match
            )
            if len(ends) != 16 or indices != list(range(16)):
                violations.append(
                    f"score chain does not end in the 16 scores of a lane: {chain[-1].text}"
                )
            continue

        # Value chain: the rescaled accumulator, then V times cast probabilities.
        carried = elements(chain[0].operands[2])
        state = None
        if carried is not None and len(carried) == 16:
            sources = set()
            for index, name in enumerate(carried):
                product = instructions.get(name)
                element = (
                    _EXTRACT.match(instructions[through(product.operands[0])].text)
                    if product is not None
                    and product.opcode == "fmul"
                    and through(product.operands[0]) in instructions
                    else None
                )
                if (
                    element is None
                    or int(element.group(2)) != index
                    or through(product.operands[1]) not in alphas
                ):
                    sources.add(None)
                else:
                    sources.add(through(element.group(1)))
            if len(sources) == 1:
                state = sources.pop()
        loop_state = instructions.get(state) if state else None
        if loop_state is None or not loop_state.text.startswith("phi <16 x float>"):
            violations.append(
                f"value chain does not start from the rescaled accumulator: {chain[0].text}"
            )
        elif chain[-1].name not in {through(name) for name in loop_state.operands}:
            violations.append(
                f"value chain does not end in the accumulator it started from: {chain[-1].text}"
            )
        counts["value_chains"] += 1
        counts[f"value_chain_length_{len(chain)}"] += 1
        for step, call in enumerate(chain):
            rows = elements(call.operands[0])
            if rows is None or len(rows) != 4:
                violations.append(
                    f"value MFMA operand is not four V elements: {call.text}"
                )
            else:
                for row in rows:
                    kind, where = loaded_from(scalar(row))
                    if kind == "lds":
                        v_region.add(where)
                    else:
                        violations.append(
                            f"value MFMA does not read V from the LDS: {where}"
                        )
            pack = elements(call.operands[1])
            if pack is None or len(pack) != 4 or not set(pack) <= probabilities:
                violations.append(
                    f"value MFMA operand is not four cast probabilities: {call.text}"
                )
            else:
                packs_by_step.setdefault(step, set()).add(tuple(pack))

    # Every cast probability enters every value chain exactly once, at the
    # same step in each.
    packed: list[str] = []
    for step, packs in sorted(packs_by_step.items()):
        if len(packs) != 1:
            violations.append(
                f"value chains disagree on the probabilities of atom {step}"
            )
        packed.extend(next(iter(packs)))
    counts["probability_packs"] = len(packs_by_step)
    if sorted(packed) != sorted(probabilities):
        violations.append(
            f"the value chains use {len(set(packed))} of {len(probabilities)} cast "
            f"probabilities, {len(packed) - len(set(packed))} of them twice"
        )

    # The two LDS regions are disjoint: V starts where the K array ends.
    k_offsets = {offset for offset, _ in k_region}
    v_offsets = {offset for offset, _ in v_region}
    k_bytes = {size for _, size in k_region}
    if k_offsets != {0} or len(k_bytes) != 1 or v_offsets != k_bytes:
        violations.append(
            f"K is read at LDS offsets {sorted(k_offsets)} (array bytes "
            f"{sorted(k_bytes)}) and V at {sorted(v_offsets)}"
        )

    def fill_offset(pointer: str) -> int | None:
        """LDS region a fill writes: follow its address arithmetic to the
        ``ptrtoint`` of the region's base."""
        item = instructions.get(through(pointer))
        while item is not None:
            base = _LDS_BASE.match(item.text)
            if base:
                return lds_offset(base.group(1))
            item = instructions.get(item.operands[0]) if item.operands else None
        return None

    # LDS fills: the K region from k_ptr only, the V region from v_ptr only.
    for match in _LDS_FILL.finditer(ir):
        buffer = instructions.get(match.group(1))
        source = buffer.operands[0] if buffer and buffer.opcode == _BUFFER else None
        offset = fill_offset(match.group(2))
        if source == "%k_ptr" and offset in k_offsets:
            counts["k_fills"] += 1
        elif source == "%v_ptr" and offset in v_offsets and offset not in k_offsets:
            counts["v_fills"] += 1
        else:
            violations.append(
                f"LDS fill from {source} into the region at offset {offset}"
            )
    if not counts["k_fills"] or not counts["v_fills"]:
        violations.append("no LDS fill of the K or of the V region")
    return violations, counts


@cache
def _lowered_subject(arch: str, case: PermutationCase) -> tuple[str, object]:
    """Lower one case's dispatch-resolved spec to LLVM IR text on the CPU."""
    from rocke.core.lower_llvm import lower_kernel_to_llvm

    from kernels.gfx942.attention_dense import build_attention_dense

    assert arch == "gfx942", "the path check models the gfx942 dense body"
    spec = resolve_dense_subject(arch, case).tuning_spec.kernel_spec
    return lower_kernel_to_llvm(build_attention_dense(spec, arch=arch), arch=arch), spec


@pytest.mark.parametrize(
    "subject_case",
    PERMUTATION_SUBJECT_CASES,
    ids=lambda item: f"{item[0]}-{item[1].case_id}",
)
def test_lowered_unmasked_path_is_the_one_the_bound_models(
    subject_case: tuple[str, PermutationCase],
) -> None:
    """The bound is derived for one arithmetic path. Any floating-point
    instruction outside it fails here: the bound is re-derived, never widened."""
    arch, case = subject_case
    ir, spec = _lowered_subject(arch, case)
    violations, counts = _classify_unmasked_path(ir, case.dtype, spec.block_n)
    assert not violations, f"{_REDERIVE}: " + "; ".join(violations)

    # Two lane halves share a key tile; 16 accumulator elements per lane and
    # 32-wide output tile; eight products per MFMA call.
    lane_keys = spec.block_n // 2
    output_elements = 16 * (spec.head_size // 32)
    assert counts == {
        "exp2_probability": lane_keys,
        "exp2_alpha": 1,
        "score": lane_keys,
        "lane_chain": lane_keys,
        "shuffle": 1,
        "combine": 1,
        "rescale_normalizer": 1,
        "rescale_accumulator": output_elements,
        "cast_probability": lane_keys,
        "output_product": output_elements,
        "cast_output": output_elements,
        "mfma": (spec.block_n // 32) * (spec.head_size // 8)
        + (spec.head_size // 32) * (spec.block_n // 8),
    }


def _replace_once(old: str, new: str) -> Callable[[str], str]:
    def edit(ir: str) -> str:
        assert old in ir, f"edit target {old!r} not in the lowered IR"
        return ir.replace(old, new, 1)

    return edit


def _substitute_once(pattern: str, replacement: str) -> Callable[[str], str]:
    def edit(ir: str) -> str:
        edited, count = re.subn(pattern, replacement, ir, count=1)
        assert count == 1, f"edit pattern {pattern!r} not in the lowered IR"
        return edited

    return edit


@pytest.mark.parametrize(
    "edit,match",
    [
        (_replace_once("= fadd float", "= fadd fast float"), "fast-math flags"),
        (
            _substitute_once(r"(= fdiv float [^\n]+)", r"\1, !fpmath !0"),
            "!fpmath metadata",
        ),
        (
            _replace_once("sdiv i32 %seqlen_kv, 64", "sdiv i32 %seqlen_kv, 128"),
            "key-tile loop bound",
        ),
        (
            _replace_once(f"fdiv float {_F32_ONE},", "fdiv float 0x4000000000000000,"),
            "expected one fdiv 1.0",
        ),
        (_replace_once(" to half", " to bfloat"), "cast to a type other than half"),
        (
            # The normalizer sums the cast weight instead of the uncast one.
            _substitute_once(
                r"(%fadd\d+) = fadd float (%fadd\d+), (%exp2f\d+)\n"
                r"(\s+)(%cast\d+) = fptrunc float \3 to half\n",
                r"\5 = fptrunc float \3 to half\n"
                r"\4%recast = fpext half \5 to float\n"
                r"\4\1 = fadd float \2, %recast\n",
            ),
            "unmodeled fpext",
        ),
        (
            # A score is masked after the scale.
            _substitute_once(
                r"(%fsub\d+) = fsub float (%fmul\d+), (%fmax\d+)\n",
                rf"%masked = select i1 true, float \2, float {_F32_INITIAL_MAX}\n"
                r"  \1 = fsub float %masked, \3\n",
            ),
            "float select on an unmasked path",
        ),
    ],
    ids=[
        "fast-math",
        "fpmath",
        "half-the-key-tiles",
        "reciprocal-of-two",
        "probability-cast-type",
        "normalizer-sums-cast-weights",
        "masked-score",
    ],
)
def test_path_check_rejects_each_unmodeled_change(edit, match: str) -> None:
    ir, spec = _lowered_subject("gfx942", PERMUTATION_SQUARE_FP16)
    violations, _ = _classify_unmasked_path(edit(ir), "fp16", spec.block_n)
    assert any(match in item for item in violations), violations


@pytest.mark.parametrize(
    "subject_case",
    PERMUTATION_SUBJECT_CASES,
    ids=lambda item: f"{item[0]}-{item[1].case_id}",
)
def test_lowered_mfma_operands_reach_the_stored_tensors(
    subject_case: tuple[str, PermutationCase],
) -> None:
    """The bound counts ``seqlen_k`` products per output element in chains of
    eight-product atoms, and takes each product as one cast probability times
    one value. This pins the chains and where their operands come from."""
    arch, case = subject_case
    ir, spec = _lowered_subject(arch, case)
    violations, counts = _trace_mfma_operands(ir, case.dtype)
    assert not violations, f"{_REDERIVE}: " + "; ".join(violations)
    fills = {name: counts.pop(name) for name in ("k_fills", "v_fills")}
    assert all(fills.values()), fills
    assert counts == {
        # Two 32-key halves of a tile; head_size / 8 atoms of eight products.
        "score_chains": spec.block_n // 32,
        f"score_chain_length_{spec.head_size // 8}": spec.block_n // 32,
        # One chain per 32 output columns; block_n / 8 atoms of eight keys.
        "value_chains": spec.head_size // 32,
        f"value_chain_length_{spec.block_n // 8}": spec.head_size // 32,
        "probability_packs": spec.block_n // 8,
    }


@pytest.mark.parametrize(
    "edit,match",
    [
        (
            # The V region aliases the K region.
            lambda ir: re.sub(
                r"(getelementptr inbounds i8, ptr addrspace\(3\) @[-\w.]+, i32 )\d+",
                r"\g<1>0",
                ir,
            ),
            "K is read at LDS offsets",
        ),
        (
            # The V region is filled from K.
            _substitute_once(
                r"(make\.buffer\.rsrc\.p1\(ptr addrspace\(1\) )%v_ptr", r"\1%k_ptr"
            ),
            "LDS fill from %k_ptr",
        ),
        (
            # One probability enters an atom twice and its neighbour never.
            _substitute_once(
                r"(insertelement <4 x half> undef, half (%cast\d+), i32 0\n"
                r"\s+%[-\w.]+ = insertelement <4 x half> %[-\w.]+, half )%cast\d+"
                r"(, i32 1\n)",
                r"\1\2\3",
            ),
            "cast probabilities",
        ),
        (
            # A value chain skips its first atom.
            _substitute_once(
                r"(%acc\d+)( = call <16 x float> @llvm\.amdgcn\.mfma[^\n]*"
                r"<16 x float> (%vp\d+), i32 0, i32 0, i32 0\)\n(?:[^\n]*\n)*?"
                r"[^\n]*<16 x float> )\1(, i32 0, i32 0, i32 0\))",
                r"\1\2\3\4",
            ),
            "value chain",
        ),
        (
            # A score atom multiplies K by K.
            _substitute_once(
                r"(getelementptr inbounds half, ptr addrspace\(1\) )%q_ptr", r"\1%k_ptr"
            ),
            "does not read Q from q_ptr",
        ),
    ],
    ids=[
        "value-region-aliases-keys",
        "values-filled-from-keys",
        "probability-used-twice",
        "value-chain-skips-an-atom",
        "score-reads-keys-twice",
    ],
)
def test_operand_trace_rejects_each_misrouted_operand(edit, match: str) -> None:
    ir, _ = _lowered_subject("gfx942", PERMUTATION_SQUARE_FP16)
    violations, _ = _trace_mfma_operands(edit(ir), "fp16")
    assert any(match in item for item in violations), violations


# ---------------------------------------------------------------------------
# Subject scope
# ---------------------------------------------------------------------------


def _unmasked_tuning_spec(**overrides):
    return _fake_tuning_spec(**{"causal": False, "sliding_window": 0, **overrides})


def test_subject_accepts_only_an_unmasked_spec_for_an_unmasked_case() -> None:
    case = PERMUTATION_SQUARE_FP16
    _verify_resolved_spec("gfx942", case, _unmasked_tuning_spec())
    with pytest.raises(AssertionError, match="causal=True"):
        _verify_resolved_spec("gfx942", case, _unmasked_tuning_spec(causal=True))
    with pytest.raises(AssertionError, match="sliding_window=64"):
        _verify_resolved_spec("gfx942", case, _unmasked_tuning_spec(sliding_window=64))
    for name, value, match in (
        ("kv_segments", 2, "single-pass"),
        ("ragged", True, "ragged"),
        ("use_sinks", True, "sinks"),
        ("emit_lse", True, "LSE"),
        ("paged", True, "non-paged"),
    ):
        with pytest.raises(AssertionError, match=match):
            _verify_resolved_spec(
                "gfx942", case, _unmasked_tuning_spec(**{name: value})
            )


@pytest.mark.parametrize(
    "subject_case",
    PERMUTATION_SUBJECT_CASES,
    ids=lambda item: f"{item[0]}-{item[1].case_id}",
)
def test_subject_resolves_the_shipped_unmasked_spec(
    subject_case: tuple[str, PermutationCase],
) -> None:
    arch, case = subject_case
    subject = resolve_dense_subject(arch, case)
    assert (subject.arch, subject.case_id) == (arch, case.case_id)
    assert subject.candidate_name == DENSE_SUBJECTS[arch].candidate_name
    spec = subject.tuning_spec.kernel_spec
    assert spec.causal is False and spec.sliding_window == 0
    # The bound is evaluated at the tile count dispatch resolves, whatever it is.
    tiles = subject.key_tiles(case.seqlen_k)
    assert tiles == -(-case.seqlen_k // spec.block_n)
    q, k, v = permutation_inputs(case)
    bound = permutation_case_bound(
        case, q, k, v, key_tiles=tiles, scale=softmax_scale(case)
    )
    assert isinstance(bound, PermutationBound) and bound.key_tiles == tiles


def test_permutation_matrix_runs_every_quick_case_on_a_known_subject() -> None:
    assert len(set(PERMUTATION_SUBJECT_CASES)) == len(PERMUTATION_SUBJECT_CASES)
    assert {arch for arch, _ in PERMUTATION_SUBJECT_CASES} <= set(DENSE_SUBJECTS)
    on_gfx942 = [case for arch, case in PERMUTATION_SUBJECT_CASES if arch == "gfx942"]
    assert on_gfx942 == list(PERMUTATION_QUICK_CASES)
    assert {case.dtype for case in PERMUTATION_QUICK_CASES} == {"fp16", "bf16"}
