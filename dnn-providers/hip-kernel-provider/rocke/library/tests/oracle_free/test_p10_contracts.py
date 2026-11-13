# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU-only contracts for the coordinate-codeword placement probe (P10)."""

from __future__ import annotations

import ast
import ctypes
import importlib
import itertools
import math
from dataclasses import fields, replace
from fractions import Fraction
from functools import cache
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ._analytical import (
    _FORMAT,
    UNIFORM_FOCUSED_ROWS,
    UNIFORM_QUICK_CASES,
    UniformCase,
    _ordered_array,
    decode_bits,
    encode_bits,
    legal_interval,
    legal_keys,
)
from ._dense_subject import DENSE_SUBJECTS
from ._p10_analytical import (
    CANARY_BITS,
    LANE_BATCH,
    LANE_KEY,
    LANE_KV_HEAD,
    LANE_PARITY,
    MASKED_EXPONENTIALS,
    P10_COORDINATE_LANES,
    P10_GROUPED_CASES,
    P10_QUICK_CASES,
    P10_UNGROUPED_CASES,
    SENTINEL_BITS,
    GuardedOutput,
    GuardLayout,
    RowMoments,
    StoreSafety,
    TileSchedule,
    _codeword,
    codeword_inputs,
    codeword_target,
    codeword_target_bits,
    codeword_value,
    emulate_path_exact,
    exact_target_bits,
    guard_layout,
    masked_leading_tiles,
    require_p10_scope,
    row_codeword_collisions,
    row_moments,
    visited_tiles,
)
from ._p10_controls import (
    P10_BLIND_CONTROL_CASES,
    P10_EMITTED_CONTROL_CASES,
    P10_EMITTED_CONTROLS,
    P10_ROW_BOUND_CASES,
    P10_SUBJECT_CONTROLS,
    _mutant_builder,
    lowered_ir,
    private_build,
    row_bound_bypass_spec,
)
from ._p10_policy import (
    P10_KILL_MATRIX,
    AdmissibleBits,
    Channel,
    CodewordResult,
    DecodedSource,
    P10Mutation,
    Scope,
    _placement_mutant_bits,
    analyze_codeword,
    applicable_mutations,
    apply_mutation,
    codeword_decoder,
    format_codeword_result,
    in_scope,
    path_exact_admissible,
)
from ._p10_subject import (
    P10_ARCHES,
    P10_SUBJECT_CASES,
    _launch_guarded,
    _verify_p10_resolved_spec,
    p10_kernel_key,
    p10_schedule,
    resolve_p10_subject,
)
from .test_contracts import _SKIP_CALLS, _dotted_name
from .test_p10_dense import _judge_output, _not_rejected

# Literal, so a change of the uniform-probability lane's specifications does not
# silently change what the placement probe covers.
_QUICK_CASE_IDS = (
    "fp16-window64-b1-sq256-sk256-hq8-hk2-d64",
    "fp16-window64-b2-sq256-sk256-hq8-hk2-d64",
    "bf16-causal-b1-sq256-sk256-hq8-hk2-d64",
    "bf16-causal-b2-sq256-sk256-hq8-hk2-d64",
    "fp16-window64-b2-sq256-sk256-hq4-hk4-d64",
    "bf16-causal-b2-sq256-sk256-hq4-hk4-d64",
)


def _subject_case_id(item: tuple[str, UniformCase]) -> str:
    arch, case = item
    return f"{arch}-{case.case_id}"


def test_quick_matrix_pins_the_grouped_and_ungrouped_cases() -> None:
    assert tuple(case.case_id for case in P10_QUICK_CASES) == _QUICK_CASE_IDS
    assert {replace(case, batch=1) for case in P10_GROUPED_CASES} == set(
        UNIFORM_QUICK_CASES
    )
    # Batch placement needs batch 2, and output-head placement needs one query
    # head per KV head, on every dtype and mask of the lane.
    pairings = {(case.dtype, case.mask) for case in UNIFORM_QUICK_CASES}
    assert {
        (case.dtype, case.mask) for case in P10_GROUPED_CASES if case.batch == 2
    } == pairings
    assert {(case.dtype, case.mask) for case in P10_UNGROUPED_CASES} == pairings
    for case in P10_GROUPED_CASES:
        assert case.nhead_q > case.nhead_k
    for case in P10_UNGROUPED_CASES:
        assert case.nhead_q == case.nhead_k
        assert case.batch == 2


@pytest.mark.parametrize("case", P10_QUICK_CASES, ids=lambda case: case.case_id)
def test_scope_admits_the_quick_matrix(case: UniformCase) -> None:
    require_p10_scope(case)


@pytest.mark.parametrize(
    "case, message",
    [
        (
            UniformCase(3, 256, 256, 8, 2, 64, "bf16", "causal"),
            r"batch=3, expected at most 2",
        ),
        (
            UniformCase(1, 256, 256, 8, 8, 64, "bf16", "causal"),
            r"nhead_k=8, expected at most 4",
        ),
        (
            UniformCase(1, 512, 512, 8, 2, 64, "bf16", "causal"),
            r"seqlen_k=512, expected at most 256",
        ),
        (
            UniformCase(1, 256, 256, 8, 2, 128, "bf16", "causal"),
            r"head_dim=128, expected at most 64",
        ),
        (
            UniformCase(1, 256, 256, 2, 2, 2, "fp16", "causal"),
            r"head_dim=2, expected at least 4",
        ),
        # Row q of a window keeps keys (q - 64, q]; with 256 keys the first row
        # that keeps none is 319.
        (
            UniformCase(1, 384, 256, 8, 2, 64, "fp16", "window", 64),
            r"query row 319 has no legal key",
        ),
    ],
)
def test_scope_refuses_cases_outside_the_codeword(
    case: UniformCase, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        require_p10_scope(case)


def test_subject_matrix_runs_every_quick_case_on_each_derived_arch() -> None:
    assert set(P10_ARCHES) <= set(DENSE_SUBJECTS)
    assert P10_SUBJECT_CASES == tuple(
        (arch, case) for arch in P10_ARCHES for case in P10_QUICK_CASES
    )
    assert len(set(P10_SUBJECT_CASES)) == len(P10_SUBJECT_CASES)


def test_subject_refuses_an_arch_without_a_derivation() -> None:
    assert "gfx950" in DENSE_SUBJECTS
    with pytest.raises(ValueError, match=r"arch='gfx950', expected one of"):
        resolve_p10_subject("gfx950", P10_QUICK_CASES[0])


def test_subject_refuses_a_case_outside_the_codeword_before_dispatch() -> None:
    with pytest.raises(ValueError, match=r"batch=3, expected at most 2"):
        resolve_p10_subject(
            "gfx942", UniformCase(3, 256, 256, 8, 2, 64, "bf16", "causal")
        )


def _fake_tuning_spec(**overrides) -> SimpleNamespace:
    kernel_spec = SimpleNamespace(
        window_wave_skip=False, persistent=False, block_m=128, block_n=64
    )
    for name, value in overrides.items():
        setattr(kernel_spec, name, value)
    return SimpleNamespace(kernel_spec=kernel_spec)


def test_subject_refuses_the_wave_skipping_window_schedule() -> None:
    _verify_p10_resolved_spec(_fake_tuning_spec())
    with pytest.raises(AssertionError, match=r"window_wave_skip=True"):
        _verify_p10_resolved_spec(_fake_tuning_spec(window_wave_skip=True))


@pytest.mark.parametrize(
    "name,value", [("persistent", True), ("block_m", 64), ("block_n", 32)]
)
def test_tile_and_grid_knobs_stay_free(name: str, value) -> None:
    """The schedule is read from the resolved spec, so a retune inside the scope
    keeps running instead of failing a pinned geometry."""
    _verify_p10_resolved_spec(_fake_tuning_spec(**{name: value}))


@pytest.mark.parametrize("subject_case", P10_SUBJECT_CASES, ids=_subject_case_id)
def test_quick_cases_resolve_inside_the_scope(
    subject_case: tuple[str, UniformCase],
) -> None:
    arch, case = subject_case
    subject = resolve_p10_subject(arch, case)
    assert (subject.arch, subject.case_id) == (arch, case.case_id)
    assert subject.candidate_name == DENSE_SUBJECTS[arch].candidate_name
    assert not subject.tuning_spec.kernel_spec.window_wave_skip


@pytest.mark.parametrize("subject_case", P10_SUBJECT_CASES, ids=_subject_case_id)
def test_batch_and_head_counts_are_kernel_arguments(
    subject_case: tuple[str, UniformCase],
) -> None:
    """On the runtime-shape path one compiled kernel serves every batch and head
    count, so the grouped and ungrouped cases of a dtype differ only in kernel
    arguments. A baked-shape kernel must instead split its identity on them."""
    arch, case = subject_case
    spec = resolve_p10_subject(arch, case).tuning_spec.kernel_spec
    heads = (4, 4) if case.nhead_q != case.nhead_k else (8, 2)
    other = replace(
        spec,
        batch=3 - case.batch,
        num_query_heads=heads[0],
        num_kv_heads=heads[1],
    )
    same_kernel = p10_kernel_key(arch, other) == p10_kernel_key(arch, spec)
    assert same_kernel == bool(spec.runtime_shape)


# One case per dtype and mask: the legal-key moments and the key and parity lanes
# do not depend on the batch or the head counts.
_PAIRING_CASES = UNIFORM_QUICK_CASES
# The BF16 window specification of the earlier H4/4 matrix. Its key-lane magnitude
# reaches 64, where BF16 spacing is 1/2. It is never launched.
_BF16_WINDOW_FIXTURE = UniformCase(1, 256, 256, 4, 4, 64, "bf16", "window", 64)


def _bits_value(bits: int, dtype: str) -> Fraction:
    return Fraction(float(decode_bits(np.uint16(bits), dtype)))


def _representable(value: Fraction, dtype: str) -> bool:
    return _bits_value(exact_target_bits(value, dtype), dtype) == value


def _is_midpoint(value: Fraction, dtype: str) -> bool:
    """Whether a value lies exactly between two adjacent values of the dtype."""
    bits = exact_target_bits(value, dtype)
    nearest = _bits_value(bits, dtype)
    if nearest == value:
        return False
    # Raw bits are sign-magnitude, so the other neighbour is one step in magnitude.
    step = 1 if abs(nearest) < abs(value) else -1
    return abs(value - nearest) == abs(value - _bits_value(bits + step, dtype))


def _only(values: np.ndarray) -> Fraction:
    unique = np.unique(values)
    assert unique.size == 1, unique
    return Fraction(float(unique[0]))


def test_codeword_restates_the_profile_values() -> None:
    assert [codeword_value(batch, 0, 0, LANE_BATCH) for batch in range(2)] == [
        Fraction(81, 4),
        Fraction(41, 2),
    ]
    assert [codeword_value(0, head, 0, LANE_KV_HEAD) for head in range(4)] == [
        Fraction(105, 4),
        Fraction(53, 2),
        Fraction(107, 4),
        Fraction(27),
    ]
    assert [codeword_value(0, 0, key, LANE_KEY) for key in (0, 1, 255)] == [
        Fraction(0),
        Fraction(-1, 2),
        Fraction(-255, 2),
    ]
    assert [codeword_value(0, 0, key, LANE_PARITY) for key in range(4)] == [
        Fraction(5, 32),
        Fraction(7, 32),
        Fraction(5, 32),
        Fraction(7, 32),
    ]
    assert [codeword_value(0, 0, 0, dim) for dim in (4, 5, 63)] == [
        Fraction(5, 16),
        Fraction(3, 8),
        Fraction(4),
    ]


@pytest.mark.parametrize("case", P10_QUICK_CASES, ids=lambda case: case.case_id)
def test_inputs_are_zero_q_finite_k_and_the_exact_codeword(case: UniformCase) -> None:
    q, k, v = codeword_inputs(case)
    assert q.shape == case.output_shape
    assert not q.any() and not np.signbit(q).any()
    assert k.shape == case.key_shape
    assert np.isfinite(k).all() and np.unique(k).size > 1
    assert v.shape == case.key_shape and v.dtype == np.float32

    # Each lane depends on its own coordinate only and holds the exact value.
    for batch in range(case.batch):
        assert _only(v[batch, :, :, LANE_BATCH]) == codeword_value(
            batch, 0, 0, LANE_BATCH
        )
    for head in range(case.nhead_k):
        assert _only(v[:, :, head, LANE_KV_HEAD]) == codeword_value(
            0, head, 0, LANE_KV_HEAD
        )
    for key in range(case.seqlen_k):
        for lane in (LANE_KEY, LANE_PARITY):
            assert _only(v[:, key, :, lane]) == codeword_value(0, 0, key, lane)
    for dim in range(P10_COORDINATE_LANES, case.head_dim):
        assert _only(v[..., dim]) == codeword_value(0, 0, 0, dim)

    # Exact in both dtypes, and key 0 of the key lane is stored as -0.
    for dtype in ("fp16", "bf16"):
        bits = encode_bits(v, dtype)
        assert (bits[:, 0, :, LANE_KEY] == 0x8000).all()
    encode_bits(q, case.dtype)
    encode_bits(k, case.dtype)


def test_bf16_stops_encoding_the_key_lane_past_the_scope_bound() -> None:
    beyond = UniformCase(1, 512, 512, 2, 2, 8, "bf16", "causal")
    with pytest.raises(ValueError, match=r"seqlen_k=512, expected at most 256"):
        codeword_inputs(beyond)
    values = _codeword(beyond)
    with pytest.raises(ValueError, match=r"first -128\.5 at \(0, 257, 0, 2\)"):
        encode_bits(values, "bf16")
    encode_bits(values, "fp16")


@pytest.mark.parametrize("case", _PAIRING_CASES, ids=lambda case: case.case_id)
def test_row_moments_match_the_enumerated_legal_keys(case: UniformCase) -> None:
    for query in range(case.seqlen_q):
        keys = legal_keys(case, query)
        assert row_moments(case, query) == RowMoments(
            count=len(keys),
            key_sum=sum(keys),
            odd_count=sum(key % 2 for key in keys),
        )


def test_row_moments_refuse_a_row_without_a_key() -> None:
    case = UniformCase(1, 384, 256, 8, 2, 64, "fp16", "window", 64)
    assert row_moments(case, 318).count == 1
    with pytest.raises(ValueError, match=r"row_moments failed - query=319"):
        row_moments(case, 319)


@pytest.mark.parametrize("case", P10_QUICK_CASES, ids=lambda case: case.case_id)
def test_targets_are_codeword_means_over_the_legal_keys(case: UniformCase) -> None:
    group = case.nhead_q // case.nhead_k
    dims = (*range(P10_COORDINATE_LANES + 2), case.head_dim - 1)
    for batch, head, query in itertools.product(
        range(case.batch), range(case.nhead_q), UNIFORM_FOCUSED_ROWS
    ):
        keys = legal_keys(case, query)
        for dim in dims:
            total = sum(codeword_value(batch, head // group, key, dim) for key in keys)
            assert codeword_target(case, batch, query, head, dim) == total / len(keys)


@pytest.mark.parametrize("case", _PAIRING_CASES, ids=lambda case: case.case_id)
def test_moment_lane_targets_have_the_closed_forms(case: UniformCase) -> None:
    for query in range(case.seqlen_q):
        key_lane = codeword_target(case, 0, query, 0, LANE_KEY)
        parity = codeword_target(case, 0, query, 0, LANE_PARITY)
        if case.mask == "window" and query >= case.window - 1:
            assert key_lane == Fraction(-(2 * query - case.window + 1), 4)
            assert parity == Fraction(3, 16)
        else:
            assert key_lane == Fraction(-query, 4)
            assert parity == (
                Fraction(3, 16)
                if query % 2
                else Fraction(6 * query + 5, 32 * (query + 1))
            )


# Per dtype and mask: the largest N_q, the largest key-lane magnitude, and the rows
# whose parity-lane target is not a value of the dtype.
_PAIRING_FACTS = {
    ("bf16", "causal"): (256, Fraction(255, 4), tuple(range(2, 255, 2))),
    ("fp16", "window"): (64, Fraction(447, 4), tuple(range(2, 63, 2))),
}


@pytest.mark.parametrize("case", _PAIRING_CASES, ids=lambda case: case.case_id)
def test_pairing_maxima_and_unrepresentable_rows(case: UniformCase) -> None:
    max_count, max_key_lane, unrepresentable = _PAIRING_FACTS[(case.dtype, case.mask)]
    rows = range(case.seqlen_q)
    key_lane = [codeword_target(case, 0, query, 0, LANE_KEY) for query in rows]
    parity = [codeword_target(case, 0, query, 0, LANE_PARITY) for query in rows]
    assert max(row_moments(case, query).count for query in rows) == max_count
    assert max(abs(value) for value in key_lane) == max_key_lane
    assert all(_representable(value, case.dtype) for value in key_lane)
    assert (
        tuple(query for query in rows if not _representable(parity[query], case.dtype))
        == unrepresentable
    )
    # No target is a midpoint of the dtype, so no stored value hangs on a tie.
    assert not any(_is_midpoint(value, case.dtype) for value in key_lane + parity)


@pytest.mark.parametrize("case", P10_QUICK_CASES, ids=lambda case: case.case_id)
def test_target_bits_round_each_coordinate_and_keep_plus_zero(
    case: UniformCase,
) -> None:
    bits = codeword_target_bits(case)
    assert bits.shape == case.output_shape and bits.dtype == np.uint16

    def rounded(batch: int, query: int, head: int, dim: int) -> int:
        target = codeword_target(case, batch, query, head, dim)
        return exact_target_bits(target, case.dtype)

    dims = (*range(P10_COORDINATE_LANES + 2), case.head_dim - 1)
    for batch, query, head, dim in itertools.product(
        range(case.batch), UNIFORM_FOCUSED_ROWS, range(case.nhead_q), dims
    ):
        assert int(bits[batch, query, head, dim]) == rounded(batch, query, head, dim)
    for query in range(case.seqlen_q):
        for lane in (LANE_KEY, LANE_PARITY):
            assert int(bits[0, query, 0, lane]) == rounded(0, query, 0, lane)

    # Each lane varies with its own coordinate only.
    for lane in (LANE_KEY, LANE_PARITY):
        assert (bits[..., lane] == bits[:1, :, :1, lane]).all()
    assert (bits[..., LANE_BATCH] == bits[:, :1, :1, LANE_BATCH]).all()
    assert (bits[..., LANE_KV_HEAD] == bits[:1, :1, :, LANE_KV_HEAD]).all()
    dimension_lanes = bits[..., P10_COORDINATE_LANES:]
    assert (dimension_lanes == dimension_lanes[:1, :1, :1]).all()

    # The one zero target is the key lane of row 0, and it is +0. Every other
    # key-lane target is negative.
    zero = np.argwhere((bits & 0x7FFF) == 0)
    assert len(zero) == case.batch * case.nhead_q
    assert {(int(row[1]), int(row[3])) for row in zero} == {(0, LANE_KEY)}
    assert not (bits[:, 0, :, LANE_KEY] & 0x8000).any()
    assert (bits[:, 1:, :, LANE_KEY] & 0x8000).all()


@pytest.mark.parametrize("case", P10_QUICK_CASES, ids=lambda case: case.case_id)
def test_query_heads_of_one_kv_group_share_every_target(case: UniformCase) -> None:
    """Why the grouped cases cannot own output-head placement, and the ungrouped
    ones can: the stored values distinguish KV heads, not query heads."""
    bits = codeword_target_bits(case)
    group = case.nhead_q // case.nhead_k
    for head in range(case.nhead_q):
        assert (bits[:, :, head] == bits[:, :, head - head % group]).all()
    distinct = {bits[0, 0, head].tobytes() for head in range(case.nhead_q)}
    assert len(distinct) == case.nhead_k


@pytest.mark.parametrize("case", P10_QUICK_CASES, ids=lambda case: case.case_id)
def test_lane_classes_are_disjoint_and_nonzero_targets_are_normal(
    case: UniformCase,
) -> None:
    bits = codeword_target_bits(case)
    classes = [
        set(np.unique(bits[..., lane]).tolist()) for lane in range(P10_COORDINATE_LANES)
    ]
    classes.append(set(np.unique(bits[..., P10_COORDINATE_LANES:]).tolist()))
    for index, first in enumerate(classes):
        for second in classes[index + 1 :]:
            assert not first & second
    assert len(classes[LANE_BATCH]) == case.batch
    assert len(classes[LANE_KV_HEAD]) == case.nhead_k
    assert len(classes[-1]) == case.head_dim - P10_COORDINATE_LANES

    fraction_bits, exponent_bits, _ = _FORMAT[case.dtype]
    magnitude = bits & 0x7FFF
    exponent = magnitude >> fraction_bits
    normal = (exponent > 0) & (exponent < (1 << exponent_bits) - 1)
    assert normal[magnitude != 0].all()


@pytest.mark.parametrize("case", P10_QUICK_CASES, ids=lambda case: case.case_id)
def test_lane_value_ranges_match_the_derivation(case: UniformCase) -> None:
    rows = range(case.seqlen_q)
    batch_lane = {codeword_target(case, b, 0, 0, LANE_BATCH) for b in range(case.batch)}
    head_lane = {
        codeword_target(case, 0, 0, head, LANE_KV_HEAD) for head in range(case.nhead_q)
    }
    key_lane = [codeword_target(case, 0, query, 0, LANE_KEY) for query in rows]
    parity = [codeword_target(case, 0, query, 0, LANE_PARITY) for query in rows]
    dimension = [
        codeword_target(case, 0, 0, 0, dim)
        for dim in range(P10_COORDINATE_LANES, case.head_dim)
    ]
    assert batch_lane == {Fraction(81, 4), Fraction(41, 2)} or (
        case.batch == 1 and batch_lane == {Fraction(81, 4)}
    )
    kv_heads = [Fraction(105, 4), Fraction(53, 2), Fraction(107, 4), Fraction(27)]
    assert head_lane == set(kv_heads[: case.nhead_k])
    lowest = Fraction(-447, 4) if case.mask == "window" else Fraction(-255, 4)
    assert (min(key_lane), max(key_lane)) == (lowest, 0)
    assert (min(parity), max(parity)) == (Fraction(5, 32), Fraction(3, 16))
    assert (min(dimension), max(dimension)) == (Fraction(5, 16), Fraction(4))
    # The real ranges are disjoint and ordered, so a value stored in the wrong
    # lane is never one the right lane admits.
    assert max(key_lane) < min(parity) <= max(parity) < min(dimension)
    assert max(dimension) < min(batch_lane) <= max(batch_lane) < min(head_lane)
    constant = [*batch_lane, *head_lane, *dimension]
    assert all(_representable(value, case.dtype) for value in constant)


# Each lane's entries are multiples of 2**e.
_LANE_GRID_EXPONENT = {LANE_BATCH: -2, LANE_KV_HEAD: -2, LANE_KEY: -1, LANE_PARITY: -5}
_DIMENSION_GRID_EXPONENT = -4


def _lane_totals(case: UniformCase, dim: int) -> list[Fraction]:
    """Sum of one lane's magnitudes over all keys, per batch or KV head."""
    batches = range(case.batch) if dim == LANE_BATCH else (0,)
    heads = range(case.nhead_k) if dim == LANE_KV_HEAD else (0,)
    keys = range(case.seqlen_k)
    return [
        sum((abs(codeword_value(batch, head, key, dim)) for key in keys), Fraction(0))
        for batch in batches
        for head in heads
    ]


@pytest.mark.parametrize("case", P10_QUICK_CASES, ids=lambda case: case.case_id)
def test_lane_sums_stay_on_a_grid_within_exact_fp32_accumulation(
    case: UniformCase,
) -> None:
    """Every sum of codeword entries of one lane over any set of keys is exact in
    FP32: the entries are multiples of ``2**e`` and their magnitudes over all keys
    total at most ``2**(24 + e)``, so every partial sum, in every order, is an
    FP32 value. The sums in fact need at most sixteen significant bits."""
    for dim in range(case.head_dim):
        exponent = _LANE_GRID_EXPONENT.get(dim, _DIMENSION_GRID_EXPONENT)
        grid = Fraction(2) ** exponent
        for key in (0, 1, case.seqlen_k - 1):
            assert (codeword_value(1, 3, key, dim) / grid).denominator == 1
        for total in _lane_totals(case, dim):
            assert total <= Fraction(2) ** (24 + exponent)
            assert total / grid < 1 << 16


def test_largest_lane_sums_of_the_matrix() -> None:
    def largest(dims) -> Fraction:
        return max(
            total
            for case in P10_QUICK_CASES
            for dim in dims
            for total in _lane_totals(case, dim)
        )

    assert largest([LANE_BATCH]) == 5248
    assert largest([LANE_KV_HEAD]) == 6912
    assert largest([LANE_KEY]) == 16320
    assert largest([LANE_PARITY]) == 48
    assert largest(range(P10_COORDINATE_LANES, 64)) == 1024


@pytest.mark.parametrize("case", P10_QUICK_CASES, ids=lambda case: case.case_id)
def test_quick_cases_have_pairwise_distinct_row_codewords(case: UniformCase) -> None:
    assert row_codeword_collisions(case) == ()
    bits = codeword_target_bits(case)
    for batch, head in itertools.product(range(case.batch), range(case.nhead_q)):
        words = {
            bits[batch, query, head, :P10_COORDINATE_LANES].tobytes()
            for query in range(case.seqlen_q)
        }
        assert len(words) == case.seqlen_q
    # The key lane alone separates the rows: it is strictly decreasing.
    key_lane = [
        codeword_target(case, 0, query, 0, LANE_KEY) for query in range(case.seqlen_q)
    ]
    assert all(first > second for first, second in zip(key_lane, key_lane[1:]))


def test_bf16_window_fixture_loses_47_adjacent_row_pairs() -> None:
    case = _BF16_WINDOW_FIXTURE
    pairs = tuple((query, query + 1) for query in range(161, 254, 2))
    assert len(pairs) == 47
    assert row_codeword_collisions(case) == pairs
    with pytest.raises(
        ValueError,
        match=r"47 pair\(s\) of query rows store the same rounded codeword, "
        r"first rows 161 and 162",
    ):
        require_p10_scope(case)

    exact = [codeword_target(case, 0, query, 0, LANE_KEY) for query in range(256)]
    stored = [
        _bits_value(exact_target_bits(value, case.dtype), case.dtype) for value in exact
    ]
    assert stored[161] == stored[162] == -65
    # From row 160 on every key-lane target is a BF16 midpoint, stored 1/4 away
    # from the exact value: a correct kernel exceeds any absolute ceiling below that.
    ties = [query for query in range(256) if _is_midpoint(exact[query], case.dtype)]
    assert ties == list(range(160, 256))
    assert all(abs(stored[query] - exact[query]) == Fraction(1, 4) for query in ties)
    # The same shape in FP16 keeps every row: eleven significant bits hold 447/4.
    assert row_codeword_collisions(replace(case, dtype="fp16")) == ()


def test_reflected_key_lane_separates_the_colliding_bf16_window_rows() -> None:
    """The inequality a second codeword would rest on; no such codeword exists here.

    A lane holding ``-(S_k - 1 - k) / 2`` has the row mean
    ``-(S_k - 1 - T_q / N_q) / 2``. On the rows whose key lane rounds together it
    stays below magnitude 64, where BF16 spacing is at most 1/4: adjacent exact
    values are 1/2 apart and rounding moves each by at most 1/8, so the stored
    values stay at least 1/4 apart."""
    case = _BF16_WINDOW_FIXTURE

    def reflected(query: int) -> Fraction:
        moments = row_moments(case, query)
        return -(case.seqlen_k - 1 - Fraction(moments.key_sum, moments.count)) / 2

    def stored(value: Fraction) -> Fraction:
        return _bits_value(exact_target_bits(value, case.dtype), case.dtype)

    exact = [reflected(query) for query in range(160, case.seqlen_q)]
    magnitudes = [abs(value) for value in exact]
    assert (min(magnitudes), max(magnitudes)) == (Fraction(63, 4), Fraction(253, 4))
    assert all(
        after - before == Fraction(1, 2) for before, after in zip(exact, exact[1:])
    )
    rounded = [stored(value) for value in exact]
    assert all(abs(new - old) <= Fraction(1, 8) for old, new in zip(exact, rounded))
    assert all(
        after - before >= Fraction(1, 4) for before, after in zip(rounded, rounded[1:])
    )

    # The key lane separates the rows below 160, so the two lanes together give
    # every row of the fixture its own pair.
    pairs = {
        (
            exact_target_bits(codeword_target(case, 0, query, 0, LANE_KEY), case.dtype),
            exact_target_bits(reflected(query), case.dtype),
        )
        for query in range(case.seqlen_q)
    }
    assert len(pairs) == case.seqlen_q


# The tile geometry the derivation's tables are written for, and geometries a
# retune could select instead.
_DERIVED_SCHEDULE = TileSchedule(128, 64)
_OTHER_SCHEDULES = (TileSchedule(64, 64), TileSchedule(256, 64), TileSchedule(128, 32))
_MASKED_EXPONENTIAL_IDS = ("masked-exp-zero", "masked-exp-subnormal")


def _schedule_id(schedule: TileSchedule) -> str:
    return f"{schedule.block_m}x{schedule.block_n}"


@pytest.mark.parametrize(
    "block_m, block_n, message",
    [
        (0, 64, r"block_m=0, block_n=64, expected positive tile sizes"),
        (128, -1, r"block_m=128, block_n=-1, expected positive tile sizes"),
        (96, 64, r"block_m=96, expected a multiple of block_n=64"),
    ],
)
def test_tile_schedule_rejects_a_geometry_that_does_not_tile(
    block_m: int, block_n: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        TileSchedule(block_m, block_n)


def test_tile_helpers_require_whole_tiles_and_a_valid_block() -> None:
    window, causal = _PAIRING_CASES
    with pytest.raises(
        ValueError, match=r"seqlen_q=256, expected a multiple of block_m=192"
    ):
        visited_tiles(window, TileSchedule(192, 64), 0)
    with pytest.raises(
        ValueError, match=r"window=64, expected a multiple of block_n=128"
    ):
        emulate_path_exact(window, TileSchedule(128, 128))
    with pytest.raises(IndexError):
        visited_tiles(causal, _DERIVED_SCHEDULE, 2)
    with pytest.raises(ValueError, match=r"dtype='bf16', expected 'fp16'"):
        emulate_path_exact(causal, _DERIVED_SCHEDULE, single_rounded=True)


def test_visited_tiles_follow_the_kernel_loop_bounds() -> None:
    window, causal = _PAIRING_CASES

    def tiles(case: UniformCase) -> list[list[int]]:
        return [list(visited_tiles(case, _DERIVED_SCHEDULE, block)) for block in (0, 1)]

    assert tiles(causal) == [[0, 1], [0, 1, 2, 3]]
    assert tiles(window) == [[0, 1], [1, 2, 3]]


@pytest.mark.parametrize(
    "schedule", (_DERIVED_SCHEDULE, *_OTHER_SCHEDULES), ids=_schedule_id
)
@pytest.mark.parametrize("case", _PAIRING_CASES, ids=lambda case: case.case_id)
def test_visited_tiles_hold_every_legal_key(
    case: UniformCase, schedule: TileSchedule
) -> None:
    for query in range(case.seqlen_q):
        tiles = visited_tiles(case, schedule, query // schedule.block_m)
        first, last = legal_interval(case, query)
        assert tiles.start * schedule.block_n <= first
        assert last < tiles.stop * schedule.block_n


def test_fully_masked_leading_tiles_of_the_derived_schedule() -> None:
    window, causal = _PAIRING_CASES
    rows = range(256)
    assert not any(masked_leading_tiles(causal, _DERIVED_SCHEDULE, row) for row in rows)
    depth = {row: masked_leading_tiles(window, _DERIVED_SCHEDULE, row) for row in rows}
    assert {row for row in rows if depth[row] == 1} == {127, *range(191, 255)}
    assert {row for row in rows if depth[row] == 2} == {255}
    assert max(depth.values()) == 2
    # A 64-row query tile meets a masked leading tile on one row per block only.
    narrow = TileSchedule(64, 64)
    assert {
        row: masked_leading_tiles(window, narrow, row)
        for row in rows
        if masked_leading_tiles(window, narrow, row)
    } == {127: 1, 191: 1, 255: 1}


@pytest.mark.parametrize(
    "masked_exponential", MASKED_EXPONENTIALS, ids=_MASKED_EXPONENTIAL_IDS
)
@pytest.mark.parametrize("case", P10_QUICK_CASES, ids=lambda case: case.case_id)
def test_path_exact_emulation_stores_every_target(
    case: UniformCase, masked_exponential: float
) -> None:
    """With the exponential of zero equal to one and round to nearest, every
    coordinate stores the correctly rounded target, whichever value the masked
    exponential takes: what a masked leading tile leaves behind is absorbed."""
    stored = emulate_path_exact(
        case, _DERIVED_SCHEDULE, masked_exponential=masked_exponential
    )
    assert stored.shape == case.output_shape and stored.dtype == np.uint16
    assert np.array_equal(stored, codeword_target_bits(case))


@pytest.mark.parametrize(
    "masked_exponential", MASKED_EXPONENTIALS, ids=_MASKED_EXPONENTIAL_IDS
)
@pytest.mark.parametrize("schedule", _OTHER_SCHEDULES, ids=_schedule_id)
@pytest.mark.parametrize("case", _PAIRING_CASES, ids=lambda case: case.case_id)
def test_path_exact_emulation_does_not_depend_on_the_tile_geometry(
    case: UniformCase, schedule: TileSchedule, masked_exponential: float
) -> None:
    stored = emulate_path_exact(case, schedule, masked_exponential=masked_exponential)
    assert np.array_equal(stored, codeword_target_bits(case))


@pytest.mark.parametrize(
    "case",
    [case for case in P10_QUICK_CASES if case.dtype == "fp16"],
    ids=lambda case: case.case_id,
)
def test_fp16_single_rounded_store_keeps_every_target(case: UniformCase) -> None:
    for masked_exponential in MASKED_EXPONENTIALS:
        stored = emulate_path_exact(
            case,
            _DERIVED_SCHEDULE,
            masked_exponential=masked_exponential,
            single_rounded=True,
        )
        assert np.array_equal(stored, codeword_target_bits(case))


@pytest.mark.parametrize("subject_case", P10_SUBJECT_CASES, ids=_subject_case_id)
def test_emulation_on_the_dispatch_resolved_schedule_stores_every_target(
    subject_case: tuple[str, UniformCase],
) -> None:
    arch, case = subject_case
    schedule = p10_schedule(resolve_p10_subject(arch, case))
    stored = emulate_path_exact(case, schedule)
    assert np.array_equal(stored, codeword_target_bits(case))


def test_emulation_moves_when_masked_keys_keep_weight() -> None:
    """The emulation is not blind to the mask: a masked exponential that does not
    vanish turns each row's mean into a mean over every visited key."""
    window, causal = _PAIRING_CASES
    for case in (window, causal):
        stored = emulate_path_exact(case, _DERIVED_SCHEDULE, masked_exponential=1.0)
        differing = stored != codeword_target_bits(case)
        assert differing[..., LANE_KEY].any()
        # The lanes that do not depend on the key are means of a constant.
        assert not differing[..., LANE_BATCH].any()
        assert not differing[..., P10_COORDINATE_LANES:].any()


_UNIT_ROUNDOFF = Fraction(1, 1 << 24)


def _binade(value: Fraction) -> int:
    """``floor(log2(|value|))`` of a nonzero rational."""
    magnitude = abs(value)
    exponent = magnitude.numerator.bit_length() - magnitude.denominator.bit_length()
    return exponent if Fraction(2) ** exponent <= magnitude else exponent - 1


def _is_power_of_two(value: Fraction) -> bool:
    return abs(value) == Fraction(2) ** _binade(value)


def _cell_edges(value: Fraction, dtype: str) -> tuple[Fraction, Fraction]:
    """Magnitudes of the two rounding boundaries that enclose a nonzero value.

    A rounding boundary is the midpoint of two adjacent values of the dtype, not
    a value of the dtype. A value rounds to its target, with no tie to resolve,
    exactly when it lies strictly between the two boundaries around the target."""
    bits = exact_target_bits(value, dtype)
    center = abs(_bits_value(bits, dtype))
    below = abs(_bits_value(bits - 1, dtype))
    above = abs(_bits_value(bits + 1, dtype))
    return (below + center) / 2, (center + above) / 2


def _cell_margin(value: Fraction, dtype: str) -> Fraction:
    """Exact distance from a nonzero value to the nearest rounding boundary."""
    lower, upper = _cell_edges(value, dtype)
    return min(abs(value) - lower, upper - abs(value))


def _constant_lane_values() -> list[Fraction]:
    """Every value a batch, KV-head or dimension lane takes inside the scope."""
    return [
        *(codeword_value(batch, 0, 0, LANE_BATCH) for batch in range(2)),
        *(codeword_value(0, head, 0, LANE_KV_HEAD) for head in range(4)),
        *(codeword_value(0, 0, 0, dim) for dim in range(P10_COORDINATE_LANES, 64)),
    ]


@pytest.mark.parametrize("case", _PAIRING_CASES, ids=lambda case: case.case_id)
def test_every_target_keeps_its_rounding_cell_under_the_path_exact_error(
    case: UniformCase,
) -> None:
    """Why the emulation stores the targets. With an exact numerator and
    denominator only ``1 / N`` and ``o * (1 / N)`` round, so the value cast to the
    dtype is within ``(2u + u**2) |y|`` of the exact target ``y``. That is strictly
    inside the target's rounding cell: by a relative margin of ``2**-(p + 2)`` for
    a representable target, and for the other targets by a margin that follows
    from the odd denominator ``q + 1``."""
    fraction_bits = _FORMAT[case.dtype][0]
    error = 2 * _UNIT_ROUNDOFF + _UNIT_ROUNDOFF**2
    rows = range(case.seqlen_q)
    key_lane = [codeword_target(case, 0, query, 0, LANE_KEY) for query in rows[1:]]
    parity = [codeword_target(case, 0, query, 0, LANE_PARITY) for query in rows]
    margins = {
        value: _cell_margin(value, case.dtype)
        for value in (*key_lane, *parity, *_constant_lane_values())
    }
    assert all(margin > error * abs(value) for value, margin in margins.items())

    # Representable targets. The relative margin 2**-(p + 2) is a sufficient
    # bound, not the size of every cell: it is attained only at a power of two,
    # whose lower boundary belongs to the finer binade below.
    cell = Fraction(1, 1 << (fraction_bits + 2))
    representable = {
        value: margin / abs(value)
        for value, margin in margins.items()
        if _representable(value, case.dtype)
    }
    assert min(representable.values()) == cell
    assert {value for value, margin in representable.items() if margin == cell} == {
        value for value in representable if _is_power_of_two(value)
    }

    # The other targets are (6q + 5)/(32(q + 1)) on even rows. They lie in the
    # binade of 1/8, whose spacing is 2**-13 in FP16 and 2**-10 in BF16, and
    # q + 1 times a target is a multiple of that spacing.
    unrepresentable = [
        query for query in rows if not _representable(parity[query], case.dtype)
    ]
    assert max(unrepresentable) + 1 == {"fp16": 63, "bf16": 255}[case.dtype]
    quantum = Fraction(1, 1 << (3 + fraction_bits))
    assert quantum == Fraction(1, {"fp16": 1 << 13, "bf16": 1 << 10}[case.dtype])
    for query in unrepresentable:
        value, odd = parity[query], query + 1
        assert _binade(value) == -3
        assert odd % 2 == 1 and odd * value == Fraction(6 * query + 5, 32)
        assert (odd * value / quantum).denominator == 1
        # Both boundaries around the target are inside its binade, so the margin
        # is the full quantum / (2 (q + 1)). Half of that is what holds against a
        # boundary of the binade below as well.
        lower, upper = _cell_edges(value, case.dtype)
        assert Fraction(1, 8) < lower and upper < Fraction(1, 4)
        assert margins[value] >= quantum / (2 * odd)

    # The thinnest margin of the pairing, relative to the target, and its row.
    # It attains the bound above, and it is still dozens of times the error.
    relative = {
        query: margins[parity[query]] / parity[query] for query in unrepresentable
    }
    thinnest = min(relative, key=relative.get)
    row, margin = {"fp16": (56, 5.728e-6), "bf16": (64, 4.017e-5)}[case.dtype]
    assert thinnest == row
    assert margins[parity[thinnest]] == quantum / (2 * (thinnest + 1))
    assert float(relative[thinnest]) == pytest.approx(margin, rel=1e-3)
    assert relative[thinnest] > 48 * error


def test_masked_leading_tile_rows_keep_their_cell_under_the_residue_bound() -> None:
    """A fully masked leading tile leaves a subnormal residue in the numerator. An
    accumulation that only rounds faithfully need not absorb it: the 64 legal
    additions then move the sum by at most ``gamma_64(2**-23)`` relatively, plus
    the residue. Such a row has a full window of 64 keys, so the division is
    exact, and every target on it is a value of the dtype, about 32 times further
    from its rounding boundaries than that bound. This holds on every tile
    geometry, and the causal case has no such row."""
    window, causal = _PAIRING_CASES
    relative, residue = Fraction(1, (1 << 17) - 1), Fraction(1, 1 << 141)

    def keeps_cell(value: Fraction) -> bool:
        bound = relative * abs(value) + residue
        margin = _cell_margin(value, window.dtype)
        return _representable(value, window.dtype) and margin > 31 * bound

    assert all(keeps_cell(value) for value in _constant_lane_values())
    assert keeps_cell(Fraction(3, 16))
    rows = range(window.seqlen_q)
    for schedule in (_DERIVED_SCHEDULE, *_OTHER_SCHEDULES):
        assert not any(masked_leading_tiles(causal, schedule, row) for row in rows)
        masked = [row for row in rows if masked_leading_tiles(window, schedule, row)]
        assert masked
        for row in masked:
            assert row_moments(window, row).count == window.window == 64
            assert codeword_target(window, 0, row, 0, LANE_PARITY) == Fraction(3, 16)
            assert keeps_cell(codeword_target(window, 0, row, 0, LANE_KEY))


def test_guard_patterns_are_a_finite_canary_and_a_nan_sentinel() -> None:
    for dtype in ("fp16", "bf16"):
        assert np.isfinite(decode_bits(np.uint16(CANARY_BITS), dtype))
        assert np.isnan(decode_bits(np.uint16(SENTINEL_BITS), dtype))
    canary_fp16 = float(decode_bits(np.uint16(CANARY_BITS), "fp16"))
    canary_bf16 = float(decode_bits(np.uint16(CANARY_BITS), "bf16"))
    assert canary_fp16 == pytest.approx(-0.02205, rel=1e-3)
    assert canary_bf16 == pytest.approx(-2.86e-16, rel=1e-2)


@pytest.mark.parametrize("case", P10_QUICK_CASES, ids=lambda case: case.case_id)
def test_no_input_or_admissible_output_has_a_guard_pattern(case: UniformCase) -> None:
    """A changed canary is then always a stray write, and a sentinel is never a
    value a correct kernel stores. Admissible outputs are taken generously: each
    target and both of its neighbours, and both zero signs where the target is
    zero."""
    patterns = [CANARY_BITS, SENTINEL_BITS]
    for values in codeword_inputs(case):
        assert not np.isin(encode_bits(values, case.dtype), patterns).any()
    targets = codeword_target_bits(case).astype(np.int64)
    nonzero = targets[(targets & 0x7FFF) != 0]
    admissible = np.concatenate([nonzero - 1, nonzero, nonzero + 1, [0x0000, 0x8000]])
    assert not np.isin(admissible, patterns).any()


@pytest.mark.parametrize("case", P10_QUICK_CASES, ids=lambda case: case.case_id)
def test_guard_holds_one_query_tile_on_each_side(case: UniformCase) -> None:
    layout = guard_layout(case, _DERIVED_SCHEDULE)
    assert layout.guard == 128 * case.nhead_q * case.head_dim
    assert layout.guard == (65536 if case.nhead_q == 8 else 32768)
    assert layout.logical == case.batch * 256 * case.nhead_q * 64
    assert layout.total == layout.logical + 2 * layout.guard
    buffer = layout.new_buffer()
    assert buffer.dtype == np.uint16 and buffer.shape == (layout.total,)
    assert (buffer[: layout.guard] == CANARY_BITS).all()
    assert (buffer[layout.guard : -layout.guard] == SENTINEL_BITS).all()
    assert (buffer[-layout.guard :] == CANARY_BITS).all()
    # A retune to a larger query tile grows the guard with it.
    assert guard_layout(case, TileSchedule(256, 64)).guard == 2 * layout.guard


def test_guard_is_rounded_up_to_the_pointer_alignment() -> None:
    tiny = UniformCase(1, 4, 4, 1, 1, 5, "fp16", "causal")
    layout = guard_layout(tiny, TileSchedule(1, 1))
    assert (layout.guard, layout.logical) == (8, 20)


@pytest.mark.parametrize(
    "guard, logical, message",
    [
        (0, 16, r"guard=0, expected a positive multiple of 8 elements"),
        (12, 16, r"guard=12, expected a positive multiple of 8 elements"),
        (8, 0, r"logical=0, expected a positive element count"),
    ],
)
def test_guard_layout_rejects_bad_element_counts(
    guard: int, logical: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        GuardLayout(guard=guard, logical=logical)


def test_guarded_output_rejects_a_buffer_that_is_not_the_allocation() -> None:
    layout = GuardLayout(guard=8, logical=16)
    GuardedOutput(layout, layout.new_buffer())
    with pytest.raises(ValueError, match=r"expected uint16 and \(32,\)"):
        GuardedOutput(layout, layout.new_buffer()[:-1])
    with pytest.raises(ValueError, match=r"raw has dtype int32"):
        GuardedOutput(layout, layout.new_buffer().astype(np.int32))
    output = GuardedOutput(layout, layout.new_buffer())
    with pytest.raises(ValueError, match=r"has 131072 elements, expected 16"):
        output.bits(P10_QUICK_CASES[0])


class _FakeDevice:
    """A flat address space standing in for device memory, so the guarded launch
    is exercised on the CPU. An access outside every allocation fails."""

    def __init__(self) -> None:
        self.allocations: dict[int, bytearray] = {}
        self._next = 0x10000

    def allocate(self, nbytes: int) -> int:
        base = self._next
        self.allocations[base] = bytearray(nbytes)
        self._next = base + (nbytes // 16 + 16) * 16
        return base

    def view(self, address: int, nbytes: int) -> memoryview:
        for base, data in self.allocations.items():
            if base <= address and address + nbytes <= base + len(data):
                offset = address - base
                return memoryview(data)[offset : offset + nbytes]
        raise AssertionError(f"access outside every allocation at {address:#x}")

    def store(self, address: int, bits: np.ndarray) -> None:
        data = np.ascontiguousarray(bits, dtype=np.uint16).tobytes()
        self.view(address, len(data))[:] = data

    def backend(self) -> SimpleNamespace:
        device = self

        class Memory:
            def __init__(self, nbytes: int) -> None:
                self._nbytes = int(nbytes)
                self._ptr = device.allocate(self._nbytes)

            def ptr(self) -> int:
                return self._ptr

            def nbytes(self) -> int:
                return self._nbytes

        class Runtime:
            def memcpy_h2d(self, dst: int, src_buf, nbytes: int) -> None:
                device.view(dst, nbytes)[:] = bytes(src_buf)[:nbytes]

            def memcpy_d2h(self, dst_buf, src: int, nbytes: int) -> None:
                ctypes.memmove(dst_buf, bytes(device.view(src, nbytes)), nbytes)

            def sync(self) -> None:
                pass

        def as_u8_buffer(array: np.ndarray):
            return (ctypes.c_uint8 * int(array.nbytes)).from_buffer(array)

        return SimpleNamespace(
            DeviceMem=Memory, Runtime=Runtime, as_u8_buffer=as_u8_buffer
        )


def _fake_launch(case: UniformCase, kernel) -> tuple[GuardedOutput, _FakeDevice, dict]:
    """Run ``kernel(device, out_address)`` through the guarded launch on a fake
    device, with the case's real inputs and guard layout."""
    device = _FakeDevice()
    layout = guard_layout(case, _DERIVED_SCHEDULE)
    q, k, v = (encode_bits(values, case.dtype) for values in codeword_inputs(case))
    host = {"q": q, "k": k, "v": v, "out": layout.new_buffer()}
    seen: dict = {}

    def launch(*, q, k, v, out) -> None:
        seen.update(q=q, k=k, v=v, out=out)
        kernel(device, out)

    raw = _launch_guarded(launch, device.backend(), host, layout)
    return GuardedOutput(layout, raw), device, seen


def test_guarded_launch_hands_the_kernel_the_interior_of_the_allocation() -> None:
    case = P10_UNGROUPED_CASES[0]
    targets = codeword_target_bits(case)
    output, device, seen = _fake_launch(
        case, lambda device, out: device.store(out, targets)
    )
    layout = output.layout

    # The kernel was given the address of the logical tensor: one guard past the
    # start of the output allocation, in bytes, and still 16-byte aligned.
    (out_base,) = (
        base
        for base, data in device.allocations.items()
        if len(data) == 2 * layout.total
    )
    assert seen["out"] == out_base + 2 * layout.guard
    assert seen["out"] % 16 == 0
    # The inputs it was given hold exactly the encoded host bits.
    for name, values in zip("qkv", codeword_inputs(case)):
        expected = encode_bits(values, case.dtype).tobytes()
        assert bytes(device.view(seen[name].ptr(), len(expected))) == expected

    assert np.array_equal(output.bits(case), targets)
    assert output.safety(case.dtype) == StoreSafety(0, 0, 0, 0)
    assert output.safety(case.dtype).violations == ()


def test_store_safety_counts_a_write_before_the_tensor() -> None:
    case = P10_UNGROUPED_CASES[0]
    targets = codeword_target_bits(case)

    def kernel(device: _FakeDevice, out: int) -> None:
        device.store(out - 2 * 3, targets.ravel()[:3])
        device.store(out, targets)

    output, _, _ = _fake_launch(case, kernel)
    safety = output.safety(case.dtype)
    assert safety == StoreSafety(prefix=3, suffix=0, unwritten=0, nonfinite=0)
    assert safety.violations == ("guard elements changed before the tensor: 3",)
    assert np.array_equal(output.bits(case), targets)


@pytest.mark.parametrize("case", P10_GROUPED_CASES, ids=lambda case: case.case_id)
def test_store_safety_counts_one_extra_query_tile_after_the_tensor(
    case: UniformCase,
) -> None:
    """What a missing row bound looks like when the last query tile holds one
    logical row: every logical row is stored, and the other ``block_m - 1`` rows
    of the tile land after the tensor. The guard holds all of them."""
    targets = codeword_target_bits(case)
    row = case.nhead_q * case.head_dim
    overrun = (_DERIVED_SCHEDULE.block_m - 1) * row
    assert overrun == 65024

    def kernel(device: _FakeDevice, out: int) -> None:
        device.store(out, targets)
        device.store(out + 2 * targets.size, targets.ravel()[:overrun])

    output, _, _ = _fake_launch(case, kernel)
    assert output.layout.guard >= overrun
    assert output.safety(case.dtype) == StoreSafety(
        prefix=0, suffix=overrun, unwritten=0, nonfinite=0
    )


def test_store_safety_separates_missing_stores_from_non_finite_values() -> None:
    case = P10_UNGROUPED_CASES[0]
    assert case.dtype == "fp16"
    targets = codeword_target_bits(case)
    flat = targets.ravel().copy()
    dropped = slice(1024, 1028)  # one four-element store vector, never written
    flat[2000] = 0x7C00  # +inf
    flat[2001] = 0xFE00  # a NaN that is not the sentinel
    flat[2002] = SENTINEL_BITS  # a store of the sentinel bits looks like no store

    def kernel(device: _FakeDevice, out: int) -> None:
        device.store(out, flat[: dropped.start])
        device.store(out + 2 * dropped.stop, flat[dropped.stop :])

    output, _, _ = _fake_launch(case, kernel)
    safety = output.safety(case.dtype)
    assert safety == StoreSafety(prefix=0, suffix=0, unwritten=5, nonfinite=2)
    assert safety.violations == (
        "sentinel pattern, presumed missing store: 5",
        "computed non-finite values: 2",
    )
    stored = output.bits(case).ravel()
    assert (stored[dropped] == SENTINEL_BITS).all()


def test_a_write_past_the_guard_is_not_contained() -> None:
    """The guard bounds what the counts can show, not where a kernel can write:
    the fake device faults on a store beyond the allocation, a real one may not."""
    case = P10_UNGROUPED_CASES[0]
    targets = codeword_target_bits(case)

    def kernel(device: _FakeDevice, out: int) -> None:
        guard_bytes = 2 * guard_layout(case, _DERIVED_SCHEDULE).guard
        device.store(out + 2 * targets.size + guard_bytes, targets.ravel()[:1])

    with pytest.raises(AssertionError, match=r"access outside every allocation"):
        _fake_launch(case, kernel)


@cache
def _correct_output(case: UniformCase) -> GuardedOutput:
    """A guarded output holding the path-exact emulation, with untouched guards."""
    layout = guard_layout(case, _DERIVED_SCHEDULE)
    raw = layout.new_buffer()
    stored = emulate_path_exact(case, _DERIVED_SCHEDULE)
    raw[layout.guard : layout.guard + layout.logical] = stored.ravel()
    return GuardedOutput(layout, raw)


def _with_bits(case: UniformCase, bits: np.ndarray) -> GuardedOutput:
    """The correct guarded output of a case with its logical tensor replaced."""
    correct = _correct_output(case)
    raw = correct.raw.copy()
    layout = correct.layout
    raw[layout.guard : layout.guard + layout.logical] = bits.ravel()
    return GuardedOutput(layout, raw)


def _rounded(case: UniformCase, value: Fraction) -> int:
    return exact_target_bits(value, case.dtype)


def _judge(case: UniformCase, bits: np.ndarray) -> CodewordResult:
    """Analyze stored bits in clean guards against the path-exact set."""
    return analyze_codeword(case, _with_bits(case, bits), path_exact_admissible(case))


@pytest.mark.parametrize("case", P10_QUICK_CASES, ids=lambda case: case.case_id)
def test_path_exact_admissible_adds_only_the_other_zero_sign(
    case: UniformCase,
) -> None:
    targets = codeword_target_bits(case)
    admissible = path_exact_admissible(case)
    assert np.array_equal(admissible.first, targets)
    differs = admissible.first != admissible.second
    assert np.array_equal(differs, targets == 0)
    assert (admissible.second[differs] == 0x8000).all()
    assert int(np.count_nonzero(differs)) == case.batch * case.nhead_q


def test_admissible_bits_reject_arrays_that_are_not_two_bit_tables() -> None:
    bits = np.zeros((2, 3), dtype=np.uint16)
    AdmissibleBits(bits, bits)
    with pytest.raises(ValueError, match=r"dtypes float32 and uint16"):
        AdmissibleBits(bits.astype(np.float32), bits)
    with pytest.raises(ValueError, match=r"shapes \(2, 3\) and \(3, 2\)"):
        AdmissibleBits(bits, bits.T.copy())


@pytest.mark.parametrize("case", P10_QUICK_CASES, ids=lambda case: case.case_id)
def test_correct_output_passes_with_clean_guards(case: UniformCase) -> None:
    result = analyze_codeword(case, _correct_output(case), path_exact_admissible(case))
    assert result.violations == ()
    assert result.channels == frozenset()
    assert result.compared == case.batch * 256 * case.nhead_q * 64
    assert not any(result.outside.values())
    assert (result.near, result.misplaced, result.foreign) == (0, 0, 0)
    assert result.worst_ulp == 0 and result.negative_zero == 0
    assert result.failures == ()
    assert format_codeword_result(result).startswith("pass: ")


def test_analyzer_requires_the_admissible_bits_of_the_case() -> None:
    grouped, ungrouped = P10_GROUPED_CASES[0], P10_UNGROUPED_CASES[0]
    with pytest.raises(ValueError, match=r"admissible shape \(2, 256, 4, 64\)"):
        analyze_codeword(
            grouped, _correct_output(grouped), path_exact_admissible(ungrouped)
        )


def test_negative_zero_is_admissible_only_where_it_is_listed() -> None:
    case = P10_UNGROUPED_CASES[1]
    targets = codeword_target_bits(case)
    stored = targets.copy()
    stored[:, 0, :, LANE_KEY] = 0x8000
    zeros = case.batch * case.nhead_q

    listed = _judge(case, stored)
    assert listed.violations == ()
    assert listed.negative_zero == zeros

    # Compared as bit patterns: with only +0 listed, -0 fails although it is
    # numerically equal to the target.
    plus_only = AdmissibleBits(targets, targets)
    refused = analyze_codeword(case, _with_bits(case, stored), plus_only)
    assert refused.outside[Channel.ROW] == zeros
    assert (refused.near, refused.worst_ulp) == (zeros, 0)

    # Zero is admissible nowhere else.
    stored = targets.copy()
    stored[0, 5, 0, LANE_BATCH] = 0x0000
    elsewhere = _judge(case, stored)
    assert elsewhere.outside[Channel.BATCH] == 1
    assert elsewhere.failures[0].source == DecodedSource(Channel.ROW, 0)


@pytest.mark.parametrize(
    "dim, channel",
    [
        (LANE_BATCH, Channel.BATCH),
        (LANE_KV_HEAD, Channel.KV_HEAD),
        (LANE_KEY, Channel.ROW),
        (LANE_PARITY, Channel.ROW),
        (17, Channel.DIMENSION),
    ],
)
@pytest.mark.parametrize("case", _PAIRING_CASES, ids=lambda case: case.case_id)
def test_one_step_off_the_target_fails_as_near_in_its_lane_channel(
    case: UniformCase, dim: int, channel: Channel
) -> None:
    targets = codeword_target_bits(case)
    coordinate = (0, 9, 1, dim)
    stored = targets.copy()
    stored[coordinate] += 1
    result = _judge(case, stored)
    assert result.channels == frozenset({channel})
    assert result.outside[channel] == 1
    assert (result.near, result.misplaced, result.foreign) == (1, 0, 0)
    assert result.worst_ulp == 1
    (failure,) = result.failures
    assert failure.coordinate == coordinate
    assert (failure.stored, failure.target) == (
        int(targets[coordinate]) + 1,
        int(targets[coordinate]),
    )
    assert channel.value in format_codeword_result(result)


def test_a_value_of_another_coordinate_is_misplaced_and_decoded() -> None:
    case = P10_UNGROUPED_CASES[1]
    targets = codeword_target_bits(case)
    moves = [
        ((0, 5, 0, LANE_KEY), (0, 7, 0, LANE_KEY), DecodedSource(Channel.ROW, 7)),
        ((0, 5, 0, 12), (0, 5, 0, 9), DecodedSource(Channel.DIMENSION, 9)),
        (
            (0, 5, 0, LANE_KV_HEAD),
            (0, 5, 3, LANE_KV_HEAD),
            DecodedSource(Channel.KV_HEAD, 3),
        ),
        ((0, 5, 0, LANE_BATCH), (1, 5, 0, LANE_BATCH), DecodedSource(Channel.BATCH, 1)),
    ]
    for destination, origin, source in moves:
        stored = targets.copy()
        stored[destination] = targets[origin]
        result = _judge(case, stored)
        assert (result.near, result.misplaced, result.foreign) == (0, 1, 0)
        (failure,) = result.failures
        assert failure.coordinate == destination
        assert failure.source == source
        assert f"codeword of index {source.index}" in format_codeword_result(result)


def test_a_value_of_no_lane_is_foreign() -> None:
    case = P10_UNGROUPED_CASES[1]
    stored = codeword_target_bits(case).copy()
    stored[1, 200, 2, 30] = _rounded(case, Fraction(1000))
    result = _judge(case, stored)
    assert (result.near, result.misplaced, result.foreign) == (0, 0, 1)
    assert result.failures[0].source is None
    assert "no codeword" in format_codeword_result(result)


def test_decoder_names_each_lane_and_marks_shared_values() -> None:
    grouped = P10_GROUPED_CASES[3]
    assert (grouped.dtype, grouped.mask, grouped.batch) == ("bf16", "causal", 2)
    decoder = codeword_decoder(grouped)
    assert decoder[_rounded(grouped, Fraction(41, 2))] == DecodedSource(
        Channel.BATCH, 1
    )
    # The KV head, not the query head: 26.5 is what query heads 4-7 store.
    assert decoder[_rounded(grouped, Fraction(53, 2))] == DecodedSource(
        Channel.KV_HEAD, 1
    )
    assert _rounded(grouped, Fraction(27)) not in decoder
    assert decoder[_rounded(grouped, Fraction(-200, 4))] == DecodedSource(
        Channel.ROW, 200
    )
    assert decoder[_rounded(grouped, Fraction(17, 16))] == DecodedSource(
        Channel.DIMENSION, 16
    )
    # 5/32 is the parity target of row 0 only; 3/16 is shared by every odd row.
    assert decoder[_rounded(grouped, Fraction(5, 32))] == DecodedSource(Channel.ROW, 0)
    assert decoder[_rounded(grouped, Fraction(3, 16))] == DecodedSource(
        Channel.ROW, None
    )
    assert decoder[0x0000] == decoder[0x8000] == DecodedSource(Channel.ROW, 0)


def test_kill_matrix_has_a_row_for_every_mutation_and_a_case_for_every_scope() -> None:
    assert set(P10_KILL_MATRIX) == set(P10Mutation)
    for scope in Scope:
        assert any(in_scope(case, scope) for case in P10_QUICK_CASES), scope
    for case in P10_QUICK_CASES:
        applicable = applicable_mutations(case)
        assert set(applicable) == {
            mutation
            for mutation, row in P10_KILL_MATRIX.items()
            if in_scope(case, row.scope)
        }
        assert (P10Mutation.BATCH_SWAP in applicable) == (case.batch == 2)
        grouped = case.nhead_q > case.nhead_k
        assert (P10Mutation.GROUP_MAP_INTERLEAVED in applicable) == grouped
        assert (P10Mutation.OUT_HEAD_SWAP in applicable) == (not grouped)


_MUTATION_CELLS = [
    pytest.param(case, mutation, id=f"{case.case_id}-{mutation.value}")
    for case in P10_QUICK_CASES
    for mutation in applicable_mutations(case)
]


@pytest.mark.parametrize("case, mutation", _MUTATION_CELLS)
def test_every_owned_mutation_is_rejected_through_its_channel(
    case: UniformCase, mutation: P10Mutation
) -> None:
    mutant = apply_mutation(case, _DERIVED_SCHEDULE, mutation, _correct_output(case))
    result = analyze_codeword(case, mutant, path_exact_admissible(case))
    assert result.violations, mutation
    assert P10_KILL_MATRIX[mutation].channel in result.channels, (
        format_codeword_result(result)
    )


@pytest.mark.parametrize(
    "case, mutation",
    [
        (P10_GROUPED_CASES[1], P10Mutation.OUT_HEAD_SWAP),
        (P10_GROUPED_CASES[3], P10Mutation.OUT_HEAD_SWAP),
        (P10_UNGROUPED_CASES[0], P10Mutation.GROUP_MAP_INTERLEAVED),
        (P10_GROUPED_CASES[0], P10Mutation.BATCH_SWAP),
    ],
    ids=lambda value: getattr(value, "case_id", getattr(value, "value", value)),
)
def test_a_mutation_outside_its_scope_stores_the_correct_output(
    case: UniformCase, mutation: P10Mutation
) -> None:
    """Why the scopes exist. Query heads of one KV group store the same values,
    so swapping two of them changes nothing; with one query head per KV head the
    interleaved and the blocked group maps coincide; and one batch has nothing to
    swap. The mutation is not owned on such a case, and applying it is refused."""
    assert not in_scope(case, P10_KILL_MATRIX[mutation].scope)
    bits = codeword_target_bits(case)
    assert np.array_equal(_placement_mutant_bits(case, mutation, bits), bits)
    with pytest.raises(ValueError, match=rf"mutation='{mutation.value}' on case"):
        apply_mutation(case, _DERIVED_SCHEDULE, mutation, _correct_output(case))


@pytest.mark.parametrize("case", P10_QUICK_CASES, ids=lambda case: case.case_id)
def test_store_mutations_change_exactly_the_predicted_counts(
    case: UniformCase,
) -> None:
    correct = _correct_output(case)
    admissible = path_exact_admissible(case)

    def result(mutation: P10Mutation) -> CodewordResult:
        mutant = apply_mutation(case, _DERIVED_SCHEDULE, mutation, correct)
        return analyze_codeword(case, mutant, admissible)

    dropped = result(P10Mutation.DROPPED_STORE)
    assert dropped.safety == StoreSafety(prefix=0, suffix=0, unwritten=4, nonfinite=0)
    assert dropped.channels == frozenset({Channel.UNWRITTEN})

    bypass = result(P10Mutation.ROW_BOUND_BYPASS)
    overrun = 127 * case.nhead_q * case.head_dim
    assert bypass.safety == StoreSafety(
        prefix=0, suffix=overrun, unwritten=0, nonfinite=0
    )
    # Every logical row is stored correctly; only the guard shows the defect.
    assert bypass.channels == frozenset({Channel.SUFFIX})

    underrun = result(P10Mutation.BASE_UNDERRUN)
    assert (underrun.safety.prefix, underrun.safety.unwritten) == (4, 4)
    assert Channel.PREFIX in underrun.channels

    # One legal key and a denominator of N - 1: row 0 divides by zero.
    minus_one = result(P10Mutation.DENOMINATOR_MINUS_ONE)
    assert minus_one.safety.nonfinite == case.batch * case.nhead_q * case.head_dim
    assert Channel.NONFINITE in minus_one.channels


def _gap(
    case: UniformCase, mutation: P10Mutation, lanes: slice | int
) -> tuple[int, int]:
    """Smallest and largest target-format distance between the correct output
    and a mutant, over the outputs of the given lanes that the mutant changes."""
    correct = _correct_output(case)
    mutant = apply_mutation(case, _DERIVED_SCHEDULE, mutation, correct)
    before = _ordered_array(correct.bits(case)[..., lanes]).astype(np.int64)
    after = _ordered_array(mutant.bits(case)[..., lanes]).astype(np.int64)
    distance = np.abs(after - before)
    changed = distance[distance > 0]
    return int(changed.min()), int(changed.max())


@pytest.mark.parametrize("case", P10_QUICK_CASES, ids=lambda case: case.case_id)
def test_placement_gaps_match_the_derivation(case: UniformCase) -> None:
    """The thinnest margin of each placement channel, in target-format values."""
    fp16 = case.dtype == "fp16"
    step = 16 if fp16 else 2
    assert _gap(case, P10Mutation.KV_HEAD_ROTATE, LANE_KV_HEAD)[0] == step
    if case.batch == 2:
        assert _gap(case, P10Mutation.BATCH_SWAP, LANE_BATCH)[0] == step
    if case.nhead_q == case.nhead_k:
        assert _gap(case, P10Mutation.OUT_HEAD_SWAP, LANE_KV_HEAD)[0] == step
    dimension_lanes = slice(P10_COORDINATE_LANES, None)
    assert _gap(case, P10Mutation.DIM_ROTATE_IN_VECTOR, dimension_lanes)[0] == (
        32 if fp16 else 4
    )
    # FP16 window: adjacent rows are at least 1/2 apart where the spacing is
    # 1/16. BF16 causal: 1/4 apart, which is one BF16 spacing from row 128 on.
    assert _gap(case, P10Mutation.ADJACENT_ROW_SWAP, LANE_KEY)[0] == (8 if fp16 else 1)


@pytest.mark.parametrize(
    "case",
    [case for case in P10_QUICK_CASES if case.dtype == "bf16"],
    ids=lambda case: case.case_id,
)
def test_bf16_rows_past_128_are_one_step_apart_and_still_rejected(
    case: UniformCase,
) -> None:
    targets = codeword_target_bits(case)
    key_lane = _ordered_array(targets[0, :, 0, LANE_KEY]).astype(np.int64)
    steps = np.abs(np.diff(key_lane))
    assert (steps[128:] == 1).all()
    assert steps[:128].min() == 2

    # The parity lane does not help there: an even row's target is within half a
    # BF16 spacing of 3/16, the odd rows' value, so the two round together.
    assert targets[0, 200, 0, LANE_PARITY] == targets[0, 201, 0, LANE_PARITY]

    stored = targets.copy()
    stored[:, [200, 201]] = targets[:, [201, 200]]
    result = _judge(case, stored)
    assert result.channels == frozenset({Channel.ROW})
    # Only the key lane of the two rows tells them apart, by one step. It is
    # refused as firmly as any other gap, because the admissible set is a single
    # bit pattern.
    swapped = 2 * case.batch * case.nhead_q
    assert result.outside[Channel.ROW] == swapped
    assert (result.near, result.worst_ulp) == (swapped, 1)


# Each GPU module and the case matrices its tests may be parametrized over.
_GPU_LANES = {
    "test_p10_dense.py": {"P10_SUBJECT_CASES"},
    "test_p10_dense_controls.py": {
        "P10_EMITTED_CONTROL_CASES",
        "P10_BLIND_CONTROL_CASES",
        "P10_ROW_BOUND_CASES",
    },
}


def _gpu_lane_calls(filename: str) -> list[ast.Call]:
    path = Path(__file__).with_name(filename)
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return [node for node in ast.walk(tree) if isinstance(node, ast.Call)]


@pytest.mark.parametrize("filename", _GPU_LANES)
def test_gpu_lanes_parametrize_over_their_case_matrices(filename: str) -> None:
    argvalues: list[ast.AST] = []
    for node in _gpu_lane_calls(filename):
        if not _dotted_name(node.func).endswith("mark.parametrize"):
            continue
        keywords = {keyword.arg: keyword.value for keyword in node.keywords}
        argvalues.append(node.args[1] if len(node.args) > 1 else keywords["argvalues"])
    for value in argvalues:
        assert isinstance(value, ast.Name), ast.dump(value)
    assert {value.id for value in argvalues} == _GPU_LANES[filename]
    assert len(argvalues) == len(_GPU_LANES[filename])


@pytest.mark.parametrize("filename", _GPU_LANES)
def test_gpu_lanes_skip_only_through_the_require_aware_arch_gate(filename: str) -> None:
    """The shared architecture gate on the case's own architecture is the only
    permitted skip, so ``ROCKE_REQUIRE_<ARCH>=1`` fails a lane that lost its GPU,
    and an out-of-scope dispatch outcome on the architecture fails, never skips."""
    calls = _gpu_lane_calls(filename)
    escapes = [
        f"line {node.lineno}: {_dotted_name(node.func)}"
        for node in calls
        if _dotted_name(node.func).split(".")[-1] in _SKIP_CALLS
    ]
    assert not escapes
    gates = [
        node for node in calls if _dotted_name(node.func) == "current_device_arch_in"
    ]
    # One gate per parametrized test, each on the case's own architecture.
    assert len(gates) == len(_GPU_LANES[filename])
    for gate in gates:
        arches = gate.args[0]
        assert isinstance(arches, ast.Tuple), ast.dump(arches)
        assert [_dotted_name(item) for item in arches.elts] == ["arch"]


@pytest.mark.parametrize("case", P10_QUICK_CASES, ids=lambda case: case.case_id)
def test_gpu_lane_judgement_passes_the_emulated_output(case: UniformCase) -> None:
    control = _judge_output("label", case, _DERIVED_SCHEDULE, _correct_output(case))
    assert control.violations == ()
    assert control.compared == math.prod(case.output_shape)


def test_gpu_lane_judgement_fails_an_output_one_step_off_its_target() -> None:
    case = P10_UNGROUPED_CASES[1]
    stored = codeword_target_bits(case).copy()
    stored[1, 200, 2, LANE_BATCH] += 1
    with pytest.raises(AssertionError, match=r"label: batch lanes: 1 output\(s\)"):
        _judge_output("label", case, _DERIVED_SCHEDULE, _with_bits(case, stored))


def test_gpu_lane_judgement_blames_the_output_for_a_cancelled_mutation() -> None:
    """A kernel that swaps adjacent rows stores exactly what that mutation turns
    back into the correct output, so the mutation is not rejected. The message
    must then point at the output's own violations rather than at the analyzer."""
    case = P10_UNGROUPED_CASES[1]
    mutation = P10Mutation.ADJACENT_ROW_SWAP
    correct = _correct_output(case)
    buggy = apply_mutation(case, _DERIVED_SCHEDULE, mutation, correct)
    with pytest.raises(AssertionError) as raised:
        _judge_output("label", case, _DERIVED_SCHEDULE, buggy)
    blamed = str(raised.value)
    assert "adjacent_row_swap was not rejected through the row channel" in blamed
    assert "the kernel output itself fails the verdict" in blamed
    assert "analyzer failed" not in blamed

    admissible = path_exact_admissible(case)
    clean = analyze_codeword(case, correct, admissible)
    assert "analyzer failed to reject adjacent_row_swap" in _not_rejected(
        "label", mutation, clean, clean
    )


def test_subject_controls_cover_every_seam_of_the_kill_matrix_once() -> None:
    seams = [P10_KILL_MATRIX[mutation].seam for mutation in P10_SUBJECT_CONTROLS]
    assert len(set(seams)) == len(seams)
    assert set(seams) == {row.seam for row in P10_KILL_MATRIX.values()}
    # Every control is emitted but the row bound, which the host enforces.
    assert P10_EMITTED_CONTROLS == P10_SUBJECT_CONTROLS[:-1]
    assert P10_SUBJECT_CONTROLS[-1] is P10Mutation.ROW_BOUND_BYPASS


def test_control_matrices_run_each_control_where_it_can_be_separated() -> None:
    batch_two = [case for case in P10_QUICK_CASES if case.batch == 2]
    assert P10_EMITTED_CONTROL_CASES == tuple(
        (arch, case, mutation)
        for arch in P10_ARCHES
        for case in batch_two
        for mutation in P10_EMITTED_CONTROLS
        if mutation in applicable_mutations(case)
    )
    # Both dtypes, and both head configurations where the control is owned.
    for mutation in P10_EMITTED_CONTROLS:
        cases = [
            case for _, case, found in P10_EMITTED_CONTROL_CASES if found is mutation
        ]
        assert {case.dtype for case in cases} == {"fp16", "bf16"}
        grouped = {case.nhead_q > case.nhead_k for case in cases}
        owned_grouped = mutation is not P10Mutation.OUT_HEAD_SWAP
        assert grouped == ({True, False} if owned_grouped else {False})
    # The head swap that the grouped cases cannot see is still launched on them,
    # as a control that must pass.
    assert P10_BLIND_CONTROL_CASES == tuple(
        (arch, case, P10Mutation.OUT_HEAD_SWAP)
        for arch in P10_ARCHES
        for case in batch_two
        if case.nhead_q > case.nhead_k
    )
    assert P10_ROW_BOUND_CASES == tuple(
        (arch, case)
        for arch in P10_ARCHES
        for case in P10_GROUPED_CASES
        if case.batch == 1
    )
    assert len(P10_EMITTED_CONTROL_CASES) == 34
    assert len(P10_BLIND_CONTROL_CASES) == len(P10_ROW_BOUND_CASES) == 2


@pytest.mark.parametrize(
    "subject_case",
    [item for item in P10_SUBJECT_CASES if item[1].batch == 2],
    ids=_subject_case_id,
)
def test_emitted_controls_change_the_lowered_ir(
    subject_case: tuple[str, UniformCase],
) -> None:
    """A control must be a different kernel. The mechanism alone changes
    nothing, so what differs in a control's IR is its defect; and no two controls
    of a case lower to the same IR."""
    arch, case = subject_case
    shipped = lowered_ir(arch, case, private=False)
    assert lowered_ir(arch, case) == shipped
    controls = [
        mutation
        for found_arch, found_case, mutation in (
            *P10_EMITTED_CONTROL_CASES,
            *P10_BLIND_CONTROL_CASES,
        )
        if (found_arch, found_case) == (arch, case)
    ]
    assert len(controls) == len(P10_EMITTED_CONTROLS)
    texts = [lowered_ir(arch, case, mutation) for mutation in controls]
    assert shipped not in texts
    assert len(set(texts)) == len(texts)


def test_private_build_restores_the_runner_also_after_a_failure() -> None:
    arch, case = P10_SUBJECT_CASES[1]
    runner = importlib.import_module(DENSE_SUBJECTS[arch].runner_module)
    builder, cache = runner.IRBuilder, runner._DENSE_LAUNCHER_CACHE
    with pytest.raises(RuntimeError, match="stop"):
        with private_build(arch, case, P10Mutation.D_HALF_SWAP):
            assert issubclass(runner.IRBuilder, builder)
            assert runner.IRBuilder is not builder
            # A cache of its own: the shipped launchers are not served to it.
            assert runner._DENSE_LAUNCHER_CACHE == {}
            assert runner._DENSE_LAUNCHER_CACHE is not cache
            raise RuntimeError("stop")
    assert runner.IRBuilder is builder
    assert runner._DENSE_LAUNCHER_CACHE is cache


def test_emitted_control_refuses_what_it_cannot_keep_inside_the_tensor() -> None:
    case = P10_SUBJECT_CASES[1][1]
    with pytest.raises(ValueError, match=r"mutation='row_bound_bypass'"):
        _mutant_builder(object, P10Mutation.ROW_BOUND_BYPASS, case)
    # Flipping one index bit swaps neighbours only when the extent is a power of
    # two; otherwise it would address past the tensor.
    with pytest.raises(ValueError, match=r"seqlen_q=384, expected a power of two"):
        _mutant_builder(object, P10Mutation.BATCH_SWAP, replace(case, seqlen_q=384))


@pytest.mark.parametrize("control", P10_ROW_BOUND_CASES, ids=_subject_case_id)
def test_row_bound_control_is_the_shipped_kernel_on_a_refused_shape(
    control: tuple[str, UniformCase],
) -> None:
    """The row-bound control cannot be told from the shipped kernel by its IR:
    the query length is a kernel argument. Its identity is host-side instead."""
    arch, case = control
    runner = importlib.import_module(DENSE_SUBJECTS[arch].runner_module)
    shipped = resolve_p10_subject(arch, case).tuning_spec.kernel_spec
    mutant = row_bound_bypass_spec(arch, case)

    # The host refuses the shape twice: on construction and in the predicate.
    with pytest.raises(ValueError, match=r"seqlen_q must be a multiple of block_m"):
        replace(shipped, seqlen_q=mutant.seqlen_q)
    supported, reason = runner.supports_attention_dense(mutant, arch=arch)
    assert not supported and "seqlen_q" in reason

    # It is the shipped spec in every field but the query length, which is one
    # row past a whole query tile; the shipped spec itself is left alone.
    differing = [
        field.name
        for field in fields(shipped)
        if getattr(shipped, field.name) != getattr(mutant, field.name)
    ]
    assert differing == ["seqlen_q"]
    assert (shipped.seqlen_q, mutant.seqlen_q) == (256, shipped.block_m + 1)
    # So it compiles to the shipped kernel and is launched as it: same launcher
    # key, same IR, and kernel arguments that differ in the query length only.
    assert mutant.runtime_shape and not mutant.persistent
    assert p10_kernel_key(arch, mutant) == p10_kernel_key(arch, shipped)
    arguments = runner.attention_dense_runtime_args(mutant)
    shipped_arguments = runner.attention_dense_runtime_args(shipped)
    assert {
        name for name in arguments if arguments[name] != shipped_arguments[name]
    } == {"seqlen_q"}
    # The grid is sized with a ceiling: two query blocks per head.
    assert math.prod(runner.attention_dense_grid(mutant)) == 2 * case.nhead_q

    # What the launch must then show: every logical row stored, and the rest of
    # the second query tile after the tensor, inside the guard.
    ragged = replace(case, seqlen_q=mutant.seqlen_q)
    require_p10_scope(ragged)
    schedule = TileSchedule(int(shipped.block_m), int(shipped.block_n))
    layout = guard_layout(ragged, schedule)
    assert (layout.guard, layout.logical) == (65536, 129 * 8 * 64)
    raw = layout.new_buffer()
    raw[layout.guard : layout.guard + layout.logical] = codeword_target_bits(
        ragged
    ).ravel()
    stored = GuardedOutput(layout, raw)
    overrun = apply_mutation(ragged, schedule, P10Mutation.ROW_BOUND_BYPASS, stored)
    result = analyze_codeword(ragged, overrun, path_exact_admissible(ragged))
    assert result.safety == StoreSafety(0, 65024, 0, 0)
    assert result.channels == frozenset({Channel.SUFFIX})
