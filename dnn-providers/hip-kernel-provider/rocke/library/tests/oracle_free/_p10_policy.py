# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Verdict, diagnostics and owned mutations of the coordinate-codeword probe (P10).

The verdict has two parts and no tolerance. Every stored output must be one of
the bit patterns a correct kernel may store at that coordinate, and the four
store-safety counts of the guarded allocation must be zero. Everything else here
is a diagnostic: the distance of a failing output to its target, the coordinate
whose codeword it carries, and the sign of the one zero output.

The owned mutations are the placement, routing, numerical and store errors the
probe claims to detect. Each is a transform of a guarded output, applied to a
correct output to show that the analyzer rejects it through a named channel.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from functools import cache
from types import MappingProxyType

import numpy as np

from ._analytical import (
    _FORMAT,
    UniformCase,
    _ordered_array,
    legal_interval,
    residue_count,
)
from ._p10_analytical import (
    LANE_BATCH,
    LANE_KEY,
    LANE_KV_HEAD,
    LANE_PARITY,
    P10_COORDINATE_LANES,
    SENTINEL_BITS,
    GuardedOutput,
    StoreSafety,
    TileSchedule,
    codeword_target,
    codeword_target_bits,
    codeword_value,
    exact_target_bits,
)

_SIGN_BIT = 0x8000
_MAGNITUDE = 0x7FFF


class Channel(str, Enum):
    """Where a defect shows: a group of codeword lanes or a store-safety count."""

    BATCH = "batch"
    KV_HEAD = "kv_head"
    ROW = "row"
    DIMENSION = "dimension"
    PREFIX = "prefix"
    SUFFIX = "suffix"
    UNWRITTEN = "unwritten"
    NONFINITE = "nonfinite"


_LANE_CHANNELS = (Channel.BATCH, Channel.KV_HEAD, Channel.ROW, Channel.DIMENSION)


def _lane_channel_index(case: UniformCase) -> np.ndarray:
    """Per output dimension, the index in ``_LANE_CHANNELS`` of its lane's channel."""
    index = np.full(case.head_dim, _LANE_CHANNELS.index(Channel.DIMENSION))
    index[LANE_BATCH] = _LANE_CHANNELS.index(Channel.BATCH)
    index[LANE_KV_HEAD] = _LANE_CHANNELS.index(Channel.KV_HEAD)
    index[[LANE_KEY, LANE_PARITY]] = _LANE_CHANNELS.index(Channel.ROW)
    return index


@dataclass(frozen=True, eq=False)
class AdmissibleBits:
    """Per coordinate, the bit patterns a correct kernel may store: one or two.

    They are compared as bit patterns, never as numbers, so the two zero signs
    stay distinct: ``-0`` is admissible only where it is listed."""

    first: np.ndarray
    second: np.ndarray

    def __post_init__(self) -> None:
        first, second = np.asarray(self.first), np.asarray(self.second)
        if first.dtype != np.uint16 or second.dtype != np.uint16:
            raise ValueError(
                f"AdmissibleBits failed - dtypes {first.dtype} and {second.dtype}, "
                "expected uint16 raw bits"
            )
        if first.shape != second.shape:
            raise ValueError(
                f"AdmissibleBits failed - shapes {first.shape} and {second.shape}, "
                "expected equal shapes"
            )

    def contains(self, bits: np.ndarray) -> np.ndarray:
        """Whether each stored bit pattern is admissible at its coordinate."""
        return (bits == self.first) | (bits == self.second)


def path_exact_admissible(case: UniformCase) -> AdmissibleBits:
    """The correctly rounded target at every coordinate, and both zero signs at
    the one coordinate whose target is zero.

    This is the verdict's admissible set. With the exponential of zero equal to
    one, round-to-nearest FP32 arithmetic and an accumulation that is exact on
    the codeword's lane grids, the kernel stores exactly the target: the value
    before the output conversion is within about ``2**-23`` of it, relatively,
    and every target is further than that from the edge of its rounding cell.
    The zero output is a sum of signed zeros whose sign nothing fixes."""
    first = codeword_target_bits(case)
    second = first.copy()
    second[first == 0] = _SIGN_BIT
    return AdmissibleBits(first, second)


@dataclass(frozen=True)
class DecodedSource:
    """The coordinate component a stored value is the correct codeword of.

    ``index`` is a batch, a KV head, a query row or a dimension, by ``channel``;
    ``None`` when several rows share the value."""

    channel: Channel
    index: int | None


@cache
def codeword_decoder(case: UniformCase) -> Mapping[int, DecodedSource]:
    """Map each rounded codeword value of a case to the coordinate it names.

    The lanes take disjoint value ranges, so a stored value identifies its lane
    as well. A value that is correct somewhere else therefore says which index
    went wrong, and a value that is in no lane's range is not a placement error.
    The KV head is decoded, not the query head: the heads of one KV group store
    the same values."""
    table: dict[int, DecodedSource] = {}

    def add(value: Fraction, source: DecodedSource) -> None:
        bits = exact_target_bits(value, case.dtype)
        known = table.get(bits)
        if known is not None and known.channel != source.channel:
            raise ValueError(
                f"codeword_decoder failed - bits {bits:#06x} of case {case.case_id} "
                f"are in the {known.channel.value} and {source.channel.value} lanes"
            )
        ambiguous = known is not None and known.index != source.index
        table[bits] = DecodedSource(source.channel, None) if ambiguous else source

    group = case.nhead_q // case.nhead_k
    for batch in range(case.batch):
        add(
            codeword_target(case, batch, 0, 0, LANE_BATCH),
            DecodedSource(Channel.BATCH, batch),
        )
    for head in range(case.nhead_k):
        add(
            codeword_target(case, 0, 0, head * group, LANE_KV_HEAD),
            DecodedSource(Channel.KV_HEAD, head),
        )
    for query in range(case.seqlen_q):
        for lane in (LANE_KEY, LANE_PARITY):
            add(
                codeword_target(case, 0, query, 0, lane),
                DecodedSource(Channel.ROW, query),
            )
    for dim in range(P10_COORDINATE_LANES, case.head_dim):
        add(
            codeword_target(case, 0, 0, 0, dim),
            DecodedSource(Channel.DIMENSION, dim),
        )
    table[_SIGN_BIT] = table[0]
    return MappingProxyType(table)


@dataclass(frozen=True)
class CodewordFailure:
    """One stored output outside its admissible set."""

    coordinate: tuple[int, int, int, int]
    stored: int
    target: int
    ulp: int
    source: DecodedSource | None


@dataclass(frozen=True)
class CodewordResult:
    """Verdict counts and diagnostics of one guarded output.

    ``outside`` counts, per lane channel, the stored outputs that are not
    admissible; unwritten and non-finite outputs are counted by ``safety``
    instead. The three classes split ``outside``: ``near`` is within one
    target-format value of the target, ``misplaced`` is further away and is the
    codeword of another coordinate, ``foreign`` is further away and is no
    codeword at all."""

    compared: int
    outside: Mapping[Channel, int]
    near: int
    misplaced: int
    foreign: int
    worst_ulp: int
    negative_zero: int
    safety: StoreSafety
    failures: tuple[CodewordFailure, ...]

    @property
    def channels(self) -> frozenset[Channel]:
        """Every channel with a nonzero count."""
        counts = {
            **self.outside,
            Channel.PREFIX: self.safety.prefix,
            Channel.SUFFIX: self.safety.suffix,
            Channel.UNWRITTEN: self.safety.unwritten,
            Channel.NONFINITE: self.safety.nonfinite,
        }
        return frozenset(channel for channel, count in counts.items() if count)

    @property
    def violations(self) -> tuple[str, ...]:
        """Every hard failure of the verdict, named."""
        found = [
            f"{channel.value} lanes: {count} output(s) outside the admissible set"
            for channel, count in self.outside.items()
            if count
        ]
        found.extend(self.safety.violations)
        if self.compared == 0:
            found.append("no output was compared")
        return tuple(found)


def analyze_codeword(
    case: UniformCase,
    output: GuardedOutput,
    admissible: AdmissibleBits,
    *,
    max_failures: int = 8,
) -> CodewordResult:
    """Judge one guarded output against the admissible bits and the guards."""
    bits = output.bits(case)
    if admissible.first.shape != bits.shape:
        raise ValueError(
            f"analyze_codeword failed - admissible shape {admissible.first.shape}, "
            f"expected {bits.shape} of case {case.case_id}"
        )
    targets = codeword_target_bits(case)
    fraction_bits, exponent_bits, _ = _FORMAT[case.dtype]
    exponent_field = ((1 << exponent_bits) - 1) << fraction_bits
    # Unwritten and non-finite outputs belong to the store-safety counts.
    judged = (bits != SENTINEL_BITS) & ((bits & exponent_field) != exponent_field)
    outside = judged & ~admissible.contains(bits)

    distance = np.abs(
        _ordered_array(bits).astype(np.int64) - _ordered_array(targets).astype(np.int64)
    )
    decoder = codeword_decoder(case)
    known = np.isin(bits, np.fromiter(decoder, dtype=np.uint16))
    near = outside & (distance <= 1)
    far = outside & (distance > 1)

    lane_channel = _lane_channel_index(case)
    per_channel = {
        channel: int(np.count_nonzero(outside[..., lane_channel == index]))
        for index, channel in enumerate(_LANE_CHANNELS)
    }
    failures = tuple(
        CodewordFailure(
            coordinate=tuple(int(index) for index in coordinate),
            stored=int(bits[tuple(coordinate)]),
            target=int(targets[tuple(coordinate)]),
            ulp=int(distance[tuple(coordinate)]),
            source=decoder.get(int(bits[tuple(coordinate)])),
        )
        for coordinate in np.argwhere(outside)[:max_failures]
    )
    return CodewordResult(
        compared=int(np.count_nonzero(judged)),
        outside=MappingProxyType(per_channel),
        near=int(np.count_nonzero(near)),
        misplaced=int(np.count_nonzero(far & known)),
        foreign=int(np.count_nonzero(far & ~known)),
        worst_ulp=int(distance[outside].max()) if outside.any() else 0,
        negative_zero=int(np.count_nonzero((targets == 0) & (bits == _SIGN_BIT))),
        safety=output.safety(case.dtype),
        failures=failures,
    )


def format_codeword_result(result: CodewordResult) -> str:
    """One line for an assertion message: violations, classes, first failures."""
    if not result.violations:
        return (
            f"pass: {result.compared} outputs admissible, guards clean, "
            f"{result.negative_zero} zero output(s) stored as -0"
        )
    parts = [
        "; ".join(result.violations),
        f"near={result.near} misplaced={result.misplaced} foreign={result.foreign} "
        f"worst_ulp={result.worst_ulp}",
    ]
    for failure in result.failures:
        source = failure.source
        named = (
            "no codeword"
            if source is None
            else f"the {source.channel.value} codeword of index {source.index}"
        )
        parts.append(
            f"[b,q,h,d]={list(failure.coordinate)} stored {failure.stored:#06x} "
            f"target {failure.target:#06x} ({failure.ulp} ulp): {named}"
        )
    return " | ".join(parts)


class P10Mutation(str, Enum):
    """Defects the probe owns, each as a transform of a correct guarded output."""

    KV_HEAD_ROTATE = "kv_head_rotate"
    GROUP_MAP_INTERLEAVED = "group_map_interleaved"
    QUERY_HEAD_BROADCAST = "query_head_broadcast"
    OUT_HEAD_SWAP = "out_head_swap"
    BATCH_SWAP = "batch_swap"
    DIM_ROTATE_IN_VECTOR = "dim_rotate_in_vector"
    D_HALF_SWAP = "d_half_swap"
    ROW_SHIFT = "row_shift"
    ADJACENT_ROW_SWAP = "adjacent_row_swap"
    V_TOKEN_PLUS_ONE = "v_token_plus_one"
    DENOMINATOR_PLUS_ONE = "denominator_plus_one"
    DENOMINATOR_MINUS_ONE = "denominator_minus_one"
    DROPPED_STORE = "dropped_store"
    BASE_UNDERRUN = "base_underrun"
    ROW_BOUND_BYPASS = "row_bound_bypass"


class Scope(str, Enum):
    """Which cases can separate a mutation."""

    EVERY = "every case"
    BATCH_TWO = "batch 2"
    GROUPED = "more query heads than KV heads"
    UNGROUPED = "one query head per KV head"


@dataclass(frozen=True)
class KillRow:
    """How one owned mutation is rejected.

    ``seam`` names the place in the kernel the mutation stands for; mutations
    that share a seam share one subject-side build. ``channel`` is where the
    analyzer must reject it. ``boundary`` says what the transform does where the
    index it moves would leave the tensor."""

    seam: str
    channel: Channel
    scope: Scope
    boundary: str


# Output elements per global store of the dense epilogue, and the elements one
# lane half contributes to each group of eight.
_STORE_VECTOR = 4

P10_KILL_MATRIX: Mapping[P10Mutation, KillRow] = MappingProxyType(
    {
        P10Mutation.KV_HEAD_ROTATE: KillRow(
            "V-head routing", Channel.KV_HEAD, Scope.EVERY, "KV heads wrap"
        ),
        P10Mutation.GROUP_MAP_INTERLEAVED: KillRow(
            "V-head routing", Channel.KV_HEAD, Scope.GROUPED, "none"
        ),
        P10Mutation.QUERY_HEAD_BROADCAST: KillRow(
            "output-head base", Channel.KV_HEAD, Scope.EVERY, "none"
        ),
        P10Mutation.OUT_HEAD_SWAP: KillRow(
            "output-head base", Channel.KV_HEAD, Scope.UNGROUPED, "none"
        ),
        P10Mutation.BATCH_SWAP: KillRow(
            "batch base", Channel.BATCH, Scope.BATCH_TWO, "none"
        ),
        P10Mutation.DIM_ROTATE_IN_VECTOR: KillRow(
            "dimension addressing",
            Channel.DIMENSION,
            Scope.EVERY,
            "wraps inside each store vector",
        ),
        P10Mutation.D_HALF_SWAP: KillRow(
            "dimension addressing", Channel.DIMENSION, Scope.EVERY, "none"
        ),
        P10Mutation.ROW_SHIFT: KillRow(
            "query-row addressing",
            Channel.ROW,
            Scope.EVERY,
            "the last row takes row 0; the write it stands for leaves the tensor "
            "and is the row-bound bypass",
        ),
        P10Mutation.ADJACENT_ROW_SWAP: KillRow(
            "query-row addressing", Channel.ROW, Scope.EVERY, "none"
        ),
        P10Mutation.V_TOKEN_PLUS_ONE: KillRow(
            "V addressing",
            Channel.ROW,
            Scope.EVERY,
            "the key past the last reads zero, as a bounded buffer load does",
        ),
        P10Mutation.DENOMINATOR_PLUS_ONE: KillRow(
            "denominator", Channel.BATCH, Scope.EVERY, "none"
        ),
        P10Mutation.DENOMINATOR_MINUS_ONE: KillRow(
            "denominator",
            Channel.BATCH,
            Scope.EVERY,
            "a row with one legal key divides by zero and stores a non-finite value",
        ),
        P10Mutation.DROPPED_STORE: KillRow(
            "output store", Channel.UNWRITTEN, Scope.EVERY, "none"
        ),
        P10Mutation.BASE_UNDERRUN: KillRow(
            "output base",
            Channel.PREFIX,
            Scope.EVERY,
            "one store vector lands before the tensor and the last is never written",
        ),
        P10Mutation.ROW_BOUND_BYPASS: KillRow(
            "row bound",
            Channel.SUFFIX,
            Scope.EVERY,
            "the query tile after the last logical row is stored after the tensor",
        ),
    }
)


def in_scope(case: UniformCase, scope: Scope) -> bool:
    """Whether a case can separate a mutation of the given scope."""
    if scope is Scope.BATCH_TWO:
        return case.batch == 2
    if scope is Scope.GROUPED:
        return case.nhead_q > case.nhead_k
    if scope is Scope.UNGROUPED:
        return case.nhead_q == case.nhead_k
    return True


def applicable_mutations(case: UniformCase) -> tuple[P10Mutation, ...]:
    """The owned mutations one case separates, in kill-matrix order."""
    return tuple(
        mutation
        for mutation, row in P10_KILL_MATRIX.items()
        if in_scope(case, row.scope)
    )


def _nonfinite_bits(numerator: Fraction, dtype: str) -> int:
    """What ``numerator * (1 / 0)`` stores: an infinity, or a NaN for zero."""
    fraction_bits, exponent_bits, _ = _FORMAT[dtype]
    infinity = ((1 << exponent_bits) - 1) << fraction_bits
    if numerator == 0:
        return infinity | (1 << (fraction_bits - 1))
    return infinity | _SIGN_BIT if numerator < 0 else infinity


def _mutant_bits(value: Fraction, scale: Fraction | None, dtype: str) -> int:
    """Round ``value * scale``; a missing scale is a division by zero."""
    if scale is None:
        return _nonfinite_bits(value, dtype)
    return exact_target_bits(value * scale, dtype)


def _numeric_mutant_bits(case: UniformCase, mutation: P10Mutation) -> np.ndarray:
    """Stored bits of a kernel with one numerical defect, ``[B, Sq, Hq, D]``.

    ``V_TOKEN_PLUS_ONE`` reads key ``k + 1`` where it should read ``k``; the key
    past the last reads zero. The denominator mutations divide the correct
    numerator by ``N_q + 1`` or ``N_q - 1``."""
    bits = np.empty(case.output_shape, dtype=np.uint16)
    group = case.nhead_q // case.nhead_k
    for query in range(case.seqlen_q):
        first, last = legal_interval(case, query)
        count = last - first + 1
        if mutation is P10Mutation.V_TOKEN_PLUS_ONE:
            read_first, read_last = first + 1, min(last + 1, case.seqlen_k - 1)
            read = max(read_last - read_first + 1, 0)
            odd = residue_count(read_first, read_last, 1, 2)
            key_lane = Fraction(-(read_first + read_last) * read // 2, 2 * count)
            parity = (Fraction(5, 32) * read + Fraction(odd, 16)) / count
            moment_scale: Fraction | None = Fraction(1)
            constant_scale: Fraction | None = Fraction(read, count)
        else:
            delta = 1 if mutation is P10Mutation.DENOMINATOR_PLUS_ONE else -1
            key_lane = codeword_target(case, 0, query, 0, LANE_KEY)
            parity = codeword_target(case, 0, query, 0, LANE_PARITY)
            moment_scale = Fraction(count, count + delta) if count + delta else None
            constant_scale = moment_scale

        dtype = case.dtype
        bits[:, query, :, LANE_KEY] = _mutant_bits(key_lane, moment_scale, dtype)
        bits[:, query, :, LANE_PARITY] = _mutant_bits(parity, moment_scale, dtype)
        for batch in range(case.batch):
            value = codeword_value(batch, 0, 0, LANE_BATCH)
            bits[batch, query, :, LANE_BATCH] = _mutant_bits(
                value, constant_scale, dtype
            )
        for head in range(case.nhead_q):
            value = codeword_value(0, head // group, 0, LANE_KV_HEAD)
            bits[:, query, head, LANE_KV_HEAD] = _mutant_bits(
                value, constant_scale, dtype
            )
        for dim in range(P10_COORDINATE_LANES, case.head_dim):
            value = codeword_value(0, 0, 0, dim)
            bits[:, query, :, dim] = _mutant_bits(value, constant_scale, dtype)
    return bits


_NUMERIC_MUTATIONS = frozenset(
    {
        P10Mutation.V_TOKEN_PLUS_ONE,
        P10Mutation.DENOMINATOR_PLUS_ONE,
        P10Mutation.DENOMINATOR_MINUS_ONE,
    }
)


def _placement_mutant_bits(
    case: UniformCase, mutation: P10Mutation, bits: np.ndarray
) -> np.ndarray:
    """Move correct outputs to the coordinates a placement defect stores them at."""
    group = case.nhead_q // case.nhead_k
    heads = np.arange(case.nhead_q)
    if mutation is P10Mutation.KV_HEAD_ROTATE:
        return np.roll(bits, -group, axis=2)
    if mutation is P10Mutation.GROUP_MAP_INTERLEAVED:
        return bits[:, :, (heads % case.nhead_k) * group]
    if mutation is P10Mutation.QUERY_HEAD_BROADCAST:
        return bits[:, :, np.zeros_like(heads)]
    if mutation is P10Mutation.OUT_HEAD_SWAP:
        heads[[0, 1]] = [1, 0]
        return bits[:, :, heads]
    if mutation is P10Mutation.BATCH_SWAP:
        return bits[::-1]
    if case.head_dim % (2 * _STORE_VECTOR):
        raise ValueError(
            f"_placement_mutant_bits failed - head_dim={case.head_dim}, expected a "
            f"multiple of {2 * _STORE_VECTOR} for the store vectors"
        )
    if mutation is P10Mutation.DIM_ROTATE_IN_VECTOR:
        vectors = bits.reshape(*bits.shape[:-1], -1, _STORE_VECTOR)
        return np.roll(vectors, -1, axis=-1).reshape(bits.shape)
    if mutation is P10Mutation.D_HALF_SWAP:
        halves = bits.reshape(*bits.shape[:-1], -1, 2, _STORE_VECTOR)
        return halves[..., ::-1, :].reshape(bits.shape)
    if mutation is P10Mutation.ROW_SHIFT:
        return np.roll(bits, -1, axis=1)
    if mutation is P10Mutation.ADJACENT_ROW_SWAP:
        pairs = bits.reshape(bits.shape[0], -1, 2, *bits.shape[2:])
        return pairs[:, :, ::-1].reshape(bits.shape)
    raise ValueError(
        f"_placement_mutant_bits failed - mutation={mutation.value!r}, "
        "expected a placement mutation"
    )


def apply_mutation(
    case: UniformCase,
    schedule: TileSchedule,
    mutation: P10Mutation,
    output: GuardedOutput,
) -> GuardedOutput:
    """The guarded output a kernel with one owned defect would leave.

    Placement mutations move the outputs of ``output``, so they can be applied
    to a real kernel's output as well as to a synthetic one. Numerical mutations
    replace the outputs by those of the defective arithmetic. Store mutations
    change which elements of the allocation are written."""
    if not in_scope(case, P10_KILL_MATRIX[mutation].scope):
        raise ValueError(
            f"apply_mutation failed - mutation={mutation.value!r} on case "
            f"{case.case_id}, expected {P10_KILL_MATRIX[mutation].scope.value}"
        )
    layout = output.layout
    start, stop = layout.guard, layout.guard + layout.logical
    raw = output.raw.copy()
    if mutation in _NUMERIC_MUTATIONS:
        raw[start:stop] = _numeric_mutant_bits(case, mutation).ravel()
    elif mutation is P10Mutation.DROPPED_STORE:
        vector = start + layout.logical // 2 // _STORE_VECTOR * _STORE_VECTOR
        raw[vector : vector + _STORE_VECTOR] = SENTINEL_BITS
    elif mutation is P10Mutation.BASE_UNDERRUN:
        raw[start - _STORE_VECTOR : stop - _STORE_VECTOR] = output.raw[start:stop]
        raw[stop - _STORE_VECTOR : stop] = SENTINEL_BITS
    elif mutation is P10Mutation.ROW_BOUND_BYPASS:
        overrun = (schedule.block_m - 1) * case.nhead_q * case.head_dim
        if overrun > layout.guard:
            raise ValueError(
                f"apply_mutation failed - overrun={overrun} elements, expected at "
                f"most the guard of {layout.guard}"
            )
        raw[stop : stop + overrun] = output.raw[start : start + overrun]
    else:
        moved = _placement_mutant_bits(case, mutation, output.bits(case))
        raw[start:stop] = moved.ravel()
    return GuardedOutput(layout, raw)
