# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU-only contracts for the coordinate-codeword placement probe (P10)."""

from __future__ import annotations

import itertools
from dataclasses import replace
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest

from ._analytical import (
    _FORMAT,
    UNIFORM_FOCUSED_ROWS,
    UNIFORM_QUICK_CASES,
    UniformCase,
    decode_bits,
    encode_bits,
    legal_keys,
)
from ._dense_subject import DENSE_SUBJECTS
from ._p10_analytical import (
    LANE_BATCH,
    LANE_KEY,
    LANE_KV_HEAD,
    LANE_PARITY,
    P10_COORDINATE_LANES,
    P10_GROUPED_CASES,
    P10_QUICK_CASES,
    P10_UNGROUPED_CASES,
    RowMoments,
    _codeword,
    codeword_inputs,
    codeword_target,
    codeword_target_bits,
    codeword_value,
    exact_target_bits,
    require_p10_scope,
    row_codeword_collisions,
    row_moments,
)
from ._p10_subject import (
    P10_ARCHES,
    P10_SUBJECT_CASES,
    _verify_p10_resolved_spec,
    p10_kernel_key,
    resolve_p10_subject,
)

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
