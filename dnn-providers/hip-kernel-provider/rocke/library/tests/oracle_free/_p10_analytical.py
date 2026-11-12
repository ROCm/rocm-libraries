# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Fixture, scope and exact targets of the coordinate-codeword placement probe (P10).

The probe launches a dense kernel on zero Q, so the softmax row of query ``q`` is
uniform over its legal keys, and on a V whose value at ``(batch, KV head, key,
dimension)`` encodes that coordinate. Every stored output then names the
coordinate it belongs to, and a guarded output allocation shows whether anything
was written outside the logical tensor. There is intentionally no attention
implementation here: each expectation is a mean of codeword values over a
legal-key interval, formed in exact rational arithmetic.

This module holds what does not depend on dispatch: the cases, the bounds inside
which the codeword is exact, the codeword, and the targets.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
from functools import cache

import numpy as np

from ._analytical import (
    UNIFORM_QUICK_CASES,
    UniformCase,
    legal_interval,
    residue_count,
    target_bits,
    uniform_inputs,
)

# Lanes 0-3 carry the coordinate: the batch, the KV head, and two moments of the
# key index. Every later lane carries its own dimension index.
LANE_BATCH = 0
LANE_KV_HEAD = 1
LANE_KEY = 2
LANE_PARITY = 3
P10_COORDINATE_LANES = 4

# Inside these bounds every codeword entry is both an FP16 and a BF16 value, and
# the lanes take disjoint value ranges. BF16's eight significant bits are what
# stops the key lane: key 257 would store -128.5.
P10_MAX_BATCH = 2
P10_MAX_KV_HEADS = 4
P10_MAX_SEQLEN_K = 256
P10_MAX_HEAD_DIM = 64

_SIGN_BIT = 0x8000

# The uniform-probability lane's two specifications, at batch 1 and at batch 2.
# Batch placement is observable only at batch 2. These are grouped-query cases:
# with zero Q the query heads of one KV group store identical values, so they
# expose KV-head routing and the group map but not which head of a group an
# output was written to.
P10_GROUPED_CASES = tuple(
    replace(case, batch=batch) for case in UNIFORM_QUICK_CASES for batch in (1, 2)
)
# One query head per KV head makes every output head distinct. These cases own
# output-head placement.
P10_UNGROUPED_CASES = tuple(
    replace(
        case,
        batch=P10_MAX_BATCH,
        nhead_q=P10_MAX_KV_HEADS,
        nhead_k=P10_MAX_KV_HEADS,
    )
    for case in UNIFORM_QUICK_CASES
)
P10_QUICK_CASES = P10_GROUPED_CASES + P10_UNGROUPED_CASES


def codeword_value(batch: int, head: int, key: int, dim: int) -> Fraction:
    """Exact codeword entry of one ``(batch, KV head, key, dimension)`` coordinate.

    The key lane is linear in the key and the parity lane alternates with it, so
    a mean over a legal-key interval identifies the interval. The other lanes do
    not depend on the key, so a mean over any keys returns them unchanged."""
    if dim == LANE_BATCH:
        return 20 + Fraction(batch + 1, 4)
    if dim == LANE_KV_HEAD:
        return 26 + Fraction(head + 1, 4)
    if dim == LANE_KEY:
        return Fraction(-key, 2)
    if dim == LANE_PARITY:
        return Fraction(5, 32) + Fraction(key % 2, 16)
    return Fraction(dim + 1, 16)


def _codeword(case: UniformCase) -> np.ndarray:
    """FP32 codeword in the runner's ``[B, Sk, Hkv, D]`` layout."""
    batch = np.arange(case.batch, dtype=np.float32)[:, None, None]
    key = np.arange(case.seqlen_k, dtype=np.float32)[None, :, None]
    head = np.arange(case.nhead_k, dtype=np.float32)[None, None, :]
    values = np.empty(case.key_shape, dtype=np.float32)
    values[..., LANE_BATCH] = 20.0 + (batch + 1.0) / 4.0
    values[..., LANE_KV_HEAD] = 26.0 + (head + 1.0) / 4.0
    # Key 0 is stored as -0 (zero times -1/2). The expectations hold for that bit
    # pattern: the one output whose mean is zero admits both zero signs.
    values[..., LANE_KEY] = key * np.float32(-0.5)
    values[..., LANE_PARITY] = 5.0 / 32.0 + (key % 2.0) / 16.0
    for dim in range(P10_COORDINATE_LANES, case.head_dim):
        values[..., dim] = (dim + 1.0) / 16.0
    return values


def codeword_inputs(case: UniformCase) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build zero Q, the uniform lane's finite nonconstant K, and codeword V.

    The arrays are FP32 in the runner layout; ``encode_bits`` turns them into the
    subject's raw bits and refuses a value that would round. K must be finite:
    zero Q annihilates it only because ``0 * k`` is zero."""
    require_p10_scope(case)
    q, k, _ = uniform_inputs(case, 0)
    return q, k, _codeword(case)


@dataclass(frozen=True)
class RowMoments:
    """Legal-key moments of one query row: ``N_q``, ``T_q`` and ``O_q``."""

    count: int
    key_sum: int
    odd_count: int


def row_moments(case: UniformCase, query: int) -> RowMoments:
    """Count, index sum and odd-index count of the legal keys of one query."""
    first, last = legal_interval(case, query)
    count = last - first + 1
    if count <= 0:
        raise ValueError(
            f"row_moments failed - query={query} of case {case.case_id}, "
            "expected a row with a legal key"
        )
    return RowMoments(
        count=count,
        key_sum=(first + last) * count // 2,
        odd_count=residue_count(first, last, 1, 2),
    )


def codeword_target(
    case: UniformCase, batch: int, query: int, head: int, dim: int
) -> Fraction:
    """Exact output at ``[batch, query, head, dim]``, ``head`` a query head.

    It is the mean of the codeword over the legal keys of the row, read from the
    KV head that serves the query head."""
    if dim == LANE_KV_HEAD:
        return codeword_value(batch, head // (case.nhead_q // case.nhead_k), 0, dim)
    if dim not in (LANE_KEY, LANE_PARITY):
        return codeword_value(batch, 0, 0, dim)
    moments = row_moments(case, query)
    if dim == LANE_KEY:
        return Fraction(-moments.key_sum, 2 * moments.count)
    return Fraction(5, 32) + Fraction(moments.odd_count, 16 * moments.count)


def exact_target_bits(value: Fraction, dtype: str) -> int:
    """Round an exact rational to FP16 or BF16 bits, nearest with ties to even.

    An exact rational has no negative zero, so zero is ``+0`` and the sign bit is
    set only for a negative value."""
    magnitude = target_bits(abs(value.numerator), value.denominator, dtype)
    return magnitude | _SIGN_BIT if value < 0 else magnitude


def codeword_target_bits(case: UniformCase) -> np.ndarray:
    """Correctly rounded targets as raw bits in the ``[B, Sq, Hq, D]`` layout."""

    def rounded(batch: int, query: int, head: int, dim: int) -> int:
        target = codeword_target(case, batch, query, head, dim)
        return exact_target_bits(target, case.dtype)

    bits = np.empty(case.output_shape, dtype=np.uint16)
    for dim in range(P10_COORDINATE_LANES, case.head_dim):
        bits[..., dim] = rounded(0, 0, 0, dim)
    for batch in range(case.batch):
        bits[batch, :, :, LANE_BATCH] = rounded(batch, 0, 0, LANE_BATCH)
    for head in range(case.nhead_q):
        bits[:, :, head, LANE_KV_HEAD] = rounded(0, 0, head, LANE_KV_HEAD)
    for query in range(case.seqlen_q):
        for lane in (LANE_KEY, LANE_PARITY):
            bits[:, query, :, lane] = rounded(0, query, 0, lane)
    return bits


@cache
def row_codeword_collisions(case: UniformCase) -> tuple[tuple[int, int], ...]:
    """Pairs of query rows whose rounded codewords are equal, in row order.

    Within one batch and KV head only the key and parity lanes vary with the
    row, so two rows are indistinguishable exactly when both lanes round to the
    same bits. That is a property of the case, not of the codeword: BF16 spacing
    reaches 1/2 at magnitude 64, where adjacent key-lane targets 1/2 apart can
    round together."""
    rows: dict[tuple[int, ...], list[int]] = {}
    for query in range(case.seqlen_q):
        word = tuple(
            exact_target_bits(codeword_target(case, 0, query, 0, lane), case.dtype)
            for lane in (LANE_KEY, LANE_PARITY)
        )
        rows.setdefault(word, []).append(query)
    return tuple(
        sorted(
            (first, second)
            for group in rows.values()
            for index, first in enumerate(group)
            for second in group[index + 1 :]
        )
    )


def require_p10_scope(case: UniformCase) -> None:
    """Refuse a case on which the codeword expectations are not defined.

    Three things put a case outside the scope. Past the bounds above a codeword
    entry would round on encoding, so the kernel would not read the values the
    expectations assume. A query row without a legal key has no mean to expect;
    a window whose queries run past the keys produces such rows. And two rows
    whose rounded codewords coincide cannot be told apart, so a row swap between
    them would pass.
    """
    upper_bounds = (
        ("batch", case.batch, P10_MAX_BATCH),
        ("nhead_k", case.nhead_k, P10_MAX_KV_HEADS),
        ("seqlen_k", case.seqlen_k, P10_MAX_SEQLEN_K),
        ("head_dim", case.head_dim, P10_MAX_HEAD_DIM),
    )
    for name, value, limit in upper_bounds:
        if value > limit:
            raise ValueError(
                f"case {case.case_id} is outside the codeword scope - "
                f"{name}={value}, expected at most {limit}"
            )
    if case.head_dim < P10_COORDINATE_LANES:
        raise ValueError(
            f"case {case.case_id} is outside the codeword scope - "
            f"head_dim={case.head_dim}, expected at least {P10_COORDINATE_LANES} "
            "for the coordinate lanes"
        )
    for query in range(case.seqlen_q):
        first, last = legal_interval(case, query)
        if last < first:
            raise ValueError(
                f"case {case.case_id} is outside the codeword scope - "
                f"query row {query} has no legal key, expected every row to have one"
            )
    collisions = row_codeword_collisions(case)
    if collisions:
        first, second = collisions[0]
        raise ValueError(
            f"case {case.case_id} is outside the codeword scope - "
            f"{len(collisions)} pair(s) of query rows store the same rounded "
            f"codeword, first rows {first} and {second}, expected distinct rows"
        )
