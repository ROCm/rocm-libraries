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

import math
from dataclasses import dataclass, replace
from fractions import Fraction
from functools import cache

import numpy as np
from rocke.numeric.bf16 import bf16_to_f32, f32_to_bf16

from ._analytical import (
    _FORMAT,
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


@dataclass(frozen=True)
class TileSchedule:
    """Tile geometry of a dense kernel: query rows per work item, keys per KV tile.

    The kernel runs one KV loop per query block, so which tiles a row visits, and
    whether it meets a fully masked tile before its legal keys, follows from the
    geometry and not from the row alone."""

    block_m: int
    block_n: int

    def __post_init__(self) -> None:
        if self.block_m <= 0 or self.block_n <= 0:
            raise ValueError(
                f"TileSchedule failed - block_m={self.block_m}, "
                f"block_n={self.block_n}, expected positive tile sizes"
            )
        if self.block_m % self.block_n:
            raise ValueError(
                f"TileSchedule failed - block_m={self.block_m}, expected a "
                f"multiple of block_n={self.block_n}"
            )


def _require_whole_tiles(case: UniformCase, schedule: TileSchedule) -> None:
    whole = (
        ("seqlen_q", case.seqlen_q, "block_m", schedule.block_m),
        ("seqlen_k", case.seqlen_k, "block_n", schedule.block_n),
        ("window", case.window, "block_n", schedule.block_n),
    )
    for name, value, tile_name, tile in whole:
        if value % tile:
            raise ValueError(
                f"case {case.case_id} does not tile - {name}={value}, "
                f"expected a multiple of {tile_name}={tile}"
            )


def visited_tiles(case: UniformCase, schedule: TileSchedule, query_block: int) -> range:
    """KV tiles the kernel's loop visits for one query block.

    The upper bound is the block's causal diagonal, clamped to the key extent. A
    sliding window moves only the lower bound, to the tile holding the oldest key
    the block's first row can still see."""
    _require_whole_tiles(case, schedule)
    if not 0 <= query_block < case.seqlen_q // schedule.block_m:
        raise IndexError(query_block)
    tiles_per_block = schedule.block_m // schedule.block_n
    diagonal = query_block * tiles_per_block
    stop = min(diagonal + tiles_per_block, case.seqlen_k // schedule.block_n)
    start = max(diagonal - case.window // schedule.block_n, 0)
    return range(start if case.mask == "window" else 0, stop)


def masked_leading_tiles(case: UniformCase, schedule: TileSchedule, query: int) -> int:
    """Fully masked tiles one row visits before its first tile with a legal key.

    On such a tile the running maximum is the mask value, so every key of the
    tile gets weight one; the first legal tile then rescales that state away."""
    first, last = legal_interval(case, query)
    count = 0
    for tile in visited_tiles(case, schedule, query // schedule.block_m):
        tile_first = tile * schedule.block_n
        if first <= tile_first + schedule.block_n - 1 and tile_first <= last:
            return count
        count += 1
    raise ValueError(
        f"masked_leading_tiles failed - query={query} of case {case.case_id} "
        "visits no tile with a legal key"
    )


# What the base-2 exponential may return for the masked argument, about -1e30:
# zero or the smallest FP32 subnormal.
MASKED_EXPONENTIALS = (0.0, float(np.nextafter(np.float32(0), np.float32(1))))


def _through_dtype(values: np.ndarray, dtype: str) -> np.ndarray:
    """Round FP32 values to the subject dtype and widen them back to FP32."""
    if dtype == "fp16":
        return values.astype(np.float16).astype(np.float32)
    return bf16_to_f32(f32_to_bf16(values)).reshape(values.shape)


def emulate_path_exact(
    case: UniformCase,
    schedule: TileSchedule,
    *,
    masked_exponential: float = 0.0,
    single_rounded: bool = False,
) -> np.ndarray:
    """Stored bits of one realization of the kernel's arithmetic, ``[B, Sq, Hq, D]``.

    A diagnostic, not a verdict and not an attention implementation: there are no
    scores, only legal-key membership, because zero Q makes every legal score
    equal. It follows the kernel's online softmax tile by tile in FP32 with round
    to nearest: the exponential of zero is one, a fully masked leading tile is
    accumulated with weight one and rescaled by ``masked_exponential`` at the
    first legal tile, masked keys of a legal tile weigh ``masked_exponential`` in
    the denominator and their dtype cast in the numerator, and the output is
    ``cast(o * (1 / l))``.

    ``single_rounded`` models the FP16 epilogue an affected compiler selects: the
    exact product ``o * (1 / l)`` rounded once to FP16 instead of to FP32 first.
    """
    _require_whole_tiles(case, schedule)
    if single_rounded and case.dtype != "fp16":
        raise ValueError(
            "emulate_path_exact failed - single_rounded=True with "
            f"dtype={case.dtype!r}, expected 'fp16'"
        )
    one = np.float32(1.0)
    masked = np.float32(masked_exponential)
    # [B, Hkv, Sk, D], so a key index selects one row of V for every batch and head.
    values = np.moveaxis(_codeword(case), 1, 2)
    bounds = [legal_interval(case, query) for query in range(case.seqlen_q)]
    first = np.array([bound[0] for bound in bounds], dtype=np.int64)
    last = np.array([bound[1] for bound in bounds], dtype=np.int64)

    stored = np.empty(
        (case.batch, case.nhead_k, case.seqlen_q, case.head_dim), dtype=np.uint16
    )
    for block in range(case.seqlen_q // schedule.block_m):
        rows = slice(block * schedule.block_m, (block + 1) * schedule.block_m)
        denominator = np.zeros(schedule.block_m, dtype=np.float32)
        numerator = np.zeros(
            (case.batch, case.nhead_k, schedule.block_m, case.head_dim),
            dtype=np.float32,
        )
        max_is_legal = np.zeros(schedule.block_m, dtype=bool)
        for tile in visited_tiles(case, schedule, block):
            keys = np.arange(tile * schedule.block_n, (tile + 1) * schedule.block_n)
            legal = (first[rows, None] <= keys) & (keys <= last[rows, None])
            now_legal = max_is_legal | legal.any(axis=1)
            # The running maximum either stays (rescale by the exponential of
            # zero) or moves from the mask value to a legal score.
            alpha = np.where(now_legal & ~max_is_legal, masked, one)
            denominator = denominator * alpha
            numerator = numerator * alpha[None, None, :, None]
            weight = np.where(legal | ~now_legal[:, None], one, masked)
            operand = _through_dtype(weight, case.dtype)
            tile_sum = np.zeros(schedule.block_m, dtype=np.float32)
            for index, key in enumerate(keys):
                tile_sum = tile_sum + weight[:, index]
                term = operand[None, None, :, index, None] * values[:, :, key, None, :]
                numerator = numerator + term
            denominator = denominator + tile_sum
            max_is_legal = now_legal
        reciprocal = (one / denominator)[None, None, :, None]
        product = numerator * reciprocal
        if single_rounded:
            # Both factors are FP32, so the FP64 product is exact and the one
            # rounding below is the single rounding to FP16.
            exact = numerator.astype(np.float64) * reciprocal.astype(np.float64)
            bits = exact.astype(np.float16).view(np.uint16)
        elif case.dtype == "fp16":
            bits = product.astype(np.float16).view(np.uint16)
        else:
            bits = f32_to_bf16(product).reshape(product.shape)
        stored[:, :, rows] = bits

    kv_head = np.arange(case.nhead_q) // (case.nhead_q // case.nhead_k)
    return np.ascontiguousarray(np.moveaxis(stored[:, kv_head], 1, 2))


# Guard patterns. The canary is a finite value no output can equal, so a changed
# guard element is a write outside the logical tensor. The sentinel is a NaN, so
# a logical element the kernel never stores is told apart from every value it
# could store. A store of exactly the sentinel bits looks the same as no store.
CANARY_BITS = 0xA5A5
SENTINEL_BITS = 0xFFFF
# Eight 16-bit elements are the 16-byte alignment the subject adapter checks on
# every device pointer, the interior output pointer included.
_GUARD_ALIGNMENT = 8


@dataclass(frozen=True)
class GuardLayout:
    """Element counts of a guarded output: ``guard | logical | guard``."""

    guard: int
    logical: int

    def __post_init__(self) -> None:
        if self.logical <= 0:
            raise ValueError(
                f"GuardLayout failed - logical={self.logical}, expected a positive "
                "element count"
            )
        if self.guard <= 0 or self.guard % _GUARD_ALIGNMENT:
            raise ValueError(
                f"GuardLayout failed - guard={self.guard}, expected a positive "
                f"multiple of {_GUARD_ALIGNMENT} elements"
            )

    @property
    def total(self) -> int:
        """Elements of the whole allocation."""
        return 2 * self.guard + self.logical

    def new_buffer(self) -> np.ndarray:
        """A host image of the allocation before the launch, as raw bits."""
        buffer = np.full(self.total, CANARY_BITS, dtype=np.uint16)
        buffer[self.guard : self.guard + self.logical] = SENTINEL_BITS
        return buffer


def guard_layout(case: UniformCase, schedule: TileSchedule) -> GuardLayout:
    """Guards that hold one query tile of output on each side of the tensor.

    The kernel stores without a row bound, and its unit of work is a query tile,
    so the overrun a missing bound produces is up to one tile of rows. A guard of
    that size shows such an overrun in full. It is no bound on a write to an
    arbitrary bad address. The tile comes from the schedule the kernel resolved
    to, so a retune to a larger query tile grows the guard with it."""
    tile = schedule.block_m * case.nhead_q * case.head_dim
    guard = -(-tile // _GUARD_ALIGNMENT) * _GUARD_ALIGNMENT
    return GuardLayout(guard=guard, logical=math.prod(case.output_shape))


@dataclass(frozen=True)
class StoreSafety:
    """Exact element counts of the four ways a launch can leave the guarded buffer.

    The classes are disjoint: ``unwritten`` counts logical elements that still
    hold the sentinel bits, ``nonfinite`` the other logical elements that are a
    NaN or an infinity. Every count must be zero."""

    prefix: int
    suffix: int
    unwritten: int
    nonfinite: int

    @property
    def violations(self) -> tuple[str, ...]:
        """The nonzero counts, named."""
        counts = (
            ("guard elements changed before the tensor", self.prefix),
            ("guard elements changed after the tensor", self.suffix),
            ("sentinel pattern, presumed missing store", self.unwritten),
            ("computed non-finite values", self.nonfinite),
        )
        return tuple(f"{name}: {count}" for name, count in counts if count)


@dataclass(frozen=True, eq=False)
class GuardedOutput:
    """The whole guarded allocation read back after a launch, as raw bits."""

    layout: GuardLayout
    raw: np.ndarray

    def __post_init__(self) -> None:
        raw = np.asarray(self.raw)
        if raw.dtype != np.uint16 or raw.shape != (self.layout.total,):
            raise ValueError(
                f"GuardedOutput failed - raw has dtype {raw.dtype} and shape "
                f"{raw.shape}, expected uint16 and ({self.layout.total},)"
            )
        copy = np.ascontiguousarray(raw).copy()
        copy.setflags(write=False)
        object.__setattr__(self, "raw", copy)

    def bits(self, case: UniformCase) -> np.ndarray:
        """The logical tensor's raw bits in the ``[B, Sq, Hq, D]`` layout."""
        if math.prod(case.output_shape) != self.layout.logical:
            raise ValueError(
                f"GuardedOutput.bits failed - case {case.case_id} has "
                f"{math.prod(case.output_shape)} elements, expected "
                f"{self.layout.logical}"
            )
        start = self.layout.guard
        logical = self.raw[start : start + self.layout.logical]
        return logical.reshape(case.output_shape).copy()

    def safety(self, dtype: str) -> StoreSafety:
        """Count changed guard elements and unwritten or non-finite outputs."""
        fraction_bits, exponent_bits, _ = _FORMAT[dtype]
        start, stop = self.layout.guard, self.layout.guard + self.layout.logical
        logical = self.raw[start:stop]
        unwritten = logical == SENTINEL_BITS
        exponent_field = ((1 << exponent_bits) - 1) << fraction_bits
        nonfinite = ((logical & exponent_field) == exponent_field) & ~unwritten
        return StoreSafety(
            prefix=int(np.count_nonzero(self.raw[:start] != CANARY_BITS)),
            suffix=int(np.count_nonzero(self.raw[stop:] != CANARY_BITS)),
            unwritten=int(np.count_nonzero(unwritten)),
            nonfinite=int(np.count_nonzero(nonfinite)),
        )
