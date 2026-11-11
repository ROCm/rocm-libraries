# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Shared solver primitives for the transforms package -- the neutral core the observers (toolbox)
and the ``transform_fragment`` verb both sit on top of.

The register-reorder delta between two layouts is solved GENERALLY from the two encodings
(:func:`_classify_maps`); nothing here consults a hand-registered table of named transform pairs, so
there is no missing-pair case that could silently return wrong registers. ``as_forward_map`` is the
single normalizer that lets labels flow the same way whether they come from a
``WarpDistributionEncoding`` or from another stage. This module imports no IR.
"""

from __future__ import annotations

import itertools
from collections import Counter
from dataclasses import dataclass

from ..encoding import WarpDistributionEncoding
from ..register_mapper import RegisterMapper


def interleave_idx(
    gather: int, stride: int, count: int, length: int | None = None
) -> tuple[int, ...]:
    """Register-index permutation ``interleave_idx<gather, stride, count>`` from the reference
    layout tables. Within each ``count``-sized block the local index is transposed as an
    ``(stride, count//stride)`` grid. ``target[i]`` for ``i in range(length)`` (``length`` defaults
    to ``count`` and must be a multiple of ``count``). A NOP when ``stride in {1, count}``.

    Only ``gather == 1`` is implemented -- and the static-distribution recipe reaches every interleaved
    target (any atom, incl. 32x32) at ``gather == 1``. The grouped ``gather > 1`` form is unimplemented and
    is not this layer's path.
    """
    if gather != 1:
        raise NotImplementedError(
            f"interleave_idx gather>1 not supported -- gather={gather} "
            "(grouped form; the static-distribution recipe uses gather==1)"
        )
    if count <= 0 or stride <= 0 or count % stride != 0:
        raise ValueError(
            f"interleave_idx needs count % stride == 0 -- gather={gather}, stride={stride}, count={count}"
        )
    length = count if length is None else length
    if length % count != 0:
        raise ValueError(
            f"interleave_idx length must be a multiple of count -- length={length}, count={count}"
        )
    inner = count // stride
    perm = [0] * length
    for i in range(length):
        block = (i // count) * count
        local = i % count
        perm[i] = block + (local % inner) * stride + (local // inner)
    return tuple(perm)


def name_permutation(perm: tuple[int, ...]) -> str:
    """Recognise a register-permutation as a closed-form ``interleave_idx`` and name it EXACTLY (the concrete
    function + params + the per-register lambda), else fall back to the raw tuple. So a viz/log can state the
    ACTUAL reorder a ``transform_fragment`` emitted -- e.g. the CRC C-shuffle is ``interleave_idx(1, 16, 64)``,
    ``dst = (r%4)*16 + (r//4)`` -- instead of a vague 'in-register reorder'."""
    n = len(perm)
    if tuple(perm) == tuple(range(n)):
        return "identity (no reorder)"
    for stride in range(2, n):  # count == n (single block); gather==1
        if n % stride:
            continue
        if tuple(perm) == interleave_idx(1, stride, n):
            inner = n // stride
            return f"interleave_idx(1, {stride}, {n})   dst = (r%{inner})*{stride} + (r//{inner})"
    return f"reorder perm={perm}"


def _forward_map(
    enc: WarpDistributionEncoding,
) -> tuple[dict[tuple[int, int], tuple[int, ...]], RegisterMapper]:
    """{(lane, register) -> matrix coordinate} for every slot of the encoding."""
    m = RegisterMapper(enc)
    fmap = {
        (lane, reg): m.matrix_coordinates(lane, reg)
        for lane in range(m.num_lanes)
        for reg in range(m.num_vector_items)
    }
    return fmap, m


def k_distribution(
    enc: WarpDistributionEncoding, k_axis: int = 1
) -> tuple[tuple[int, ...], ...]:
    """Project the encoding onto its K axis: per lane, the tuple of K coordinates by register slot.

    ``k_axis`` is the contraction axis index (1 for both A=(M,K) and B=(N,K)). Whether two operands
    are K-aligned is :func:`~rocke.helpers.tiling.transforms.mma_pair_k_aligned`'s question, answered
    against the instruction's canonical layouts.
    """
    m = RegisterMapper(enc)
    return tuple(
        tuple(
            m.matrix_coordinates(lane, reg)[k_axis] for reg in range(m.num_vector_items)
        )
        for lane in range(m.num_lanes)
    )


def as_forward_map(x) -> dict[tuple[int, int], tuple[int, ...]]:
    """Normalize a layout input to a forward map ``{(lane, reg) -> coord}``. Accepts EITHER a
    ``WarpDistributionEncoding`` (labels generated from the distribution via ``RegisterMapper``) OR an
    already-populated forward map (labels sourced from ANOTHER STAGE). Lets the machinery flow labels the
    same way regardless of where they came from."""
    if isinstance(x, dict):
        return x
    rm = RegisterMapper(x)
    return {
        (l, r): tuple(rm.matrix_coordinates(l, r))
        for l in range(rm.num_lanes)
        for r in range(rm.num_vector_items)
    }


@dataclass(frozen=True)
class TransformPlan:
    """The classified delta between two layouts. ``tier`` is ``"reorder"`` or ``"cross_lane"``.
    For ``reorder``, ``permutation[src_reg] == dst_reg`` (lane-uniform). ``reason`` explains a
    ``cross_lane`` outcome."""

    tier: str
    permutation: tuple[int, ...] | None
    reason: str


def _classify_maps(
    smap: dict[tuple[int, int], tuple[int, ...]],
    tmap: dict[tuple[int, int], tuple[int, ...]],
) -> TransformPlan:
    """Classify the delta ``smap -> tmap`` between two forward maps ``{(lane,reg)->coord}`` (the IR-free
    core; works on labels from an encoding OR from another stage).

    - ``reorder``    -- every lane keeps the same element set AND the source->target register permutation
      is identical on every lane. Emittable as a single compile-time register permutation.
    - ``cross_lane`` -- some element changes lanes, or the on-lane permutation is not lane-uniform.

    Raises ``ValueError`` if the maps describe different fragment dimensions or different element sets.
    """
    s_lanes = {l for l, _ in smap}
    s_regs = {r for _, r in smap}
    t_lanes = {l for l, _ in tmap}
    t_regs = {r for _, r in tmap}
    if (len(s_lanes), len(s_regs)) != (len(t_lanes), len(t_regs)):
        raise ValueError(
            "cannot transform between fragments of different dimensions -- source is "
            f"{len(s_lanes)}x{len(s_regs)} (lanes x regs), target is {len(t_lanes)}x{len(t_regs)}"
        )
    if Counter(smap.values()) != Counter(tmap.values()):
        raise ValueError(
            "cannot transform between layouts that hold different elements -- source and target "
            "describe different tiles, or the same element appears a different number of times "
            "(check shapes / that they are the same logical tile)"
        )

    # A REPLICATED layout holds one element at several (lane,reg) slots (e.g. gfx11 WMMA duplicates
    # the operand across lane halves). This coordinate->slot classifier cannot pair source and target
    # copies -- report it rather than let one slot silently win (which mis-classifies the reorder).
    # Classify within a single replication copy instead.
    target_of: dict[tuple[int, ...], tuple[int, int]] = {}
    for (lane, reg), coord in tmap.items():
        if coord in target_of:
            raise ValueError(
                "cannot classify a transform for a REPLICATED layout -- element "
                f"{coord} occupies multiple (lane, reg) slots, so source and target copies cannot be "
                "paired by coordinate; classify within a single replication copy instead"
            )
        target_of[coord] = (lane, reg)

    per_lane_perm: dict[int, dict[int, int]] = {}
    for (lane, reg), coord in smap.items():
        dst_lane, dst_reg = target_of[coord]
        if dst_lane != lane:
            return TransformPlan(
                "cross_lane",
                None,
                f"element {coord} moves lane {lane}->{dst_lane}; needs cross-lane movement",
            )
        per_lane_perm.setdefault(lane, {})[reg] = dst_reg

    reference = per_lane_perm[min(per_lane_perm)]
    for lane, perm in per_lane_perm.items():
        if perm != reference:
            return TransformPlan(
                "cross_lane",
                None,
                f"register permutation on lane {lane} differs from lane 0 -- not lane-uniform, "
                "so not a single compile-time reorder",
            )
    permutation = tuple(reference[reg] for reg in range(len(s_regs)))
    return TransformPlan("reorder", permutation, "")


@dataclass(frozen=True)
class ReorderPlan:
    """A discovered IN-REGISTER reorder that bridges a COALESCED (memory-order) register frame to the
    REQUESTED (consumer-order) frame -- e.g. the ``v_perm_b32`` that turns a coalesced ``ds_read`` landing
    order into the MMA-operand order. Everything is DERIVED per case (never a stored/hardcoded interleave
    param): ``label`` is :func:`name_permutation`; ``tier`` is the cost-ladder rung
    (``tiling_interleaving_design.md`` sec 7a) -- a *dword*-aligned reorder is a register renumber, a
    *sub-dword* reorder is a real ``v_perm_b32`` repack; ``vperm_per_lane`` is the emitted-op estimate;
    ``cost`` is the one-line render string. A returned plan ALWAYS means a real reorder (identity ->
    ``reorder_between`` returns ``None``)."""

    tier: str  # "reorder (dword)" | "reorder (sub-dword, Nx)" | "cross_lane"
    permutation: tuple[int, ...] | None
    label: str  # name_permutation(perm), e.g. "interleave_idx(1, 8, 32) ..."
    vperm_per_lane: int
    cost: str


def _dword_aligned(perm: tuple[int, ...], pack: int) -> bool:
    """True iff ``perm`` moves whole ``pack``-sized dword blocks as units (register renumber); False iff it
    splits a dword (a sub-dword repack -- a real ``v_perm_b32``). ``pack`` = elements per 32-bit register.
    """
    if pack <= 1:
        return True
    for b in range(0, len(perm), pack):
        if perm[b] % pack != 0 or any(perm[b + j] != perm[b] + j for j in range(pack)):
            return False
    return True


def _axis_permutation(
    smap: dict[tuple[int, int], tuple[int, ...]],
    tmap: dict[tuple[int, int], tuple[int, ...]],
) -> tuple[int, ...] | None:
    """The fixed axis permutation ``pi`` with ``tmap[k] == tuple(smap[k][pi[i]] for i)`` for EVERY shared
    key, or ``None``. Identity ``pi`` = a pure rename (numeric coords unchanged); a non-identity ``pi`` = a
    transpose / axis swap. Both maps must share keys and coord rank."""
    if set(smap) != set(tmap):
        return None
    ndim = len(next(iter(smap.values())))
    for pi in itertools.permutations(range(ndim)):
        if all(tmap[k] == tuple(smap[k][i] for i in pi) for k in smap):
            return pi
    return None


_Layout = WarpDistributionEncoding | dict[tuple[int, int], tuple[int, ...]]

# One MMA operand position read against its canonical layout: (register, instruction K, K label).
_KPosition = tuple[int, int, int]


@dataclass(frozen=True)
class _KRelabelling:
    """How one MMA operand labels the instruction's K, read against the operand's canonical layout.

    The instruction multiplies the A value at canonical ``(M, K)`` with the B value at canonical
    ``(N, K)`` for every M and N of the wave tile that share the canonical (instruction) K. So an
    operand is pairable only if, at each instruction K, every free row (M for A, N for B) holds the
    same K label: ``labels`` maps instruction K to that label. ``conflict`` is ``None`` then; otherwise
    it names the first instruction K two free rows disagree on, and ``labels`` holds the first label
    seen at each instruction K. ``rows`` holds the operand's positions per ``(lane, canonical free
    row)``, in register order: the positions an in-register reorder may permute among themselves
    without moving a value to another row.
    """

    labels: dict[int, int]
    conflict: str | None
    rows: dict[tuple[int, int | tuple[int, ...]], tuple[_KPosition, ...]]

    def unpaired_row(
        self, labels: dict[int, int]
    ) -> tuple[int, int | tuple[int, ...]] | None:
        """The first ``(lane, free row)`` whose K labels are not, in any register order, the ones
        ``labels`` gives its instruction K; ``None`` when an in-register reorder within each row
        makes the operand hold ``labels``."""
        for row, positions in self.rows.items():
            held = sorted(label for _, _, label in positions)
            if held != sorted(labels[k] for _, k, _ in positions):
                return row
        return None

    def first_disagreement(self, labels: dict[int, int]) -> tuple[int, int, int, int]:
        """``(lane, register, instruction K, K label)`` of the first position whose K label is not
        the one ``labels`` gives its instruction K. The caller has found the operand not aligned
        with ``labels``, so such a position exists."""
        for (lane, _), positions in self.rows.items():
            for register, instruction_k, label in positions:
                if label != labels[instruction_k]:
                    return lane, register, instruction_k, label
        raise AssertionError(
            "first_disagreement: every position holds the K label `labels` gives its instruction "
            "K; call it only for an operand that disagrees with `labels`"
        )


def _k_relabelling(
    operand: _Layout, canon: _Layout, *, k_axis: int, role: str
) -> _KRelabelling | str:
    """The :class:`_KRelabelling` of ``operand`` against its canonical layout ``canon`` (each an
    encoding or a forward map); a message when the two do not hold the same ``(lane, register)``
    positions, since then ``canon`` is not this operand's canonical layout."""
    fwd, canon_fwd = as_forward_map(operand), as_forward_map(canon)
    if len(fwd) != len(canon_fwd):
        return (
            f"{role} holds {len(fwd)} (lane, register) positions but its canonical layout holds "
            f"{len(canon_fwd)} -- pass canonical_layouts(traits, subtiles) of this wave tile"
        )
    if set(fwd) != set(canon_fwd):
        return (
            f"{role} holds (lane, register) {min(set(fwd) - set(canon_fwd))}, which its canonical "
            "layout does not -- pass canonical_layouts(traits, subtiles) of this wave tile"
        )
    free_name = "M" if role == "A" else "N"
    labels: dict[int, int] = {}
    first_free: dict[int, int | tuple[int, ...]] = {}
    conflict: str | None = None
    rows: dict[tuple[int, int | tuple[int, ...]], list[_KPosition]] = {}
    for lane, register in sorted(canon_fwd):
        canonical = canon_fwd[(lane, register)]
        instruction_k = canonical[k_axis]
        label = fwd[(lane, register)][k_axis]
        free_coords = canonical[:k_axis] + canonical[k_axis + 1 :]
        free = free_coords[0] if len(free_coords) == 1 else free_coords
        rows.setdefault((lane, free), []).append((register, instruction_k, label))
        if instruction_k not in labels:
            labels[instruction_k], first_free[instruction_k] = label, free
        elif labels[instruction_k] != label and conflict is None:
            if free == first_free[instruction_k]:
                conflict = (
                    f"{role}'s replicas of {free_name} {free} disagree on K: instruction K "
                    f"{instruction_k} holds K {labels[instruction_k]} in one and K {label} in "
                    "another; every replica must hold the same values"
                )
            else:
                conflict = (
                    f"{role}'s K labels depend on its {free_name} row: instruction K "
                    f"{instruction_k} holds K {labels[instruction_k]} in {free_name} "
                    f"{first_free[instruction_k]} but K {label} in {free_name} {free}; the "
                    f"instruction multiplies every {free_name} row at one instruction K with the "
                    "same K of the other operand"
                )
    return _KRelabelling(
        labels, conflict, {row: tuple(found) for row, found in rows.items()}
    )


def _free_relabel(
    canon: WarpDistributionEncoding,
    supplied_fwd: dict[tuple[int, int], tuple[int, ...]],
    free_axis: int = 0,
    k_axis: int = 1,
) -> dict[int, int]:
    """Map each canonical free index -> the SUPPLIED layout's free label at that index's canonical K=0 slot.

    The machine treats a physical register as its CANONICAL ``(free, K)``; the supplied distribution's
    label sitting on that register is what actually FLOWS through the machine (docs/mma_is_machinery.md).
    Canonical input -> identity; a relabeled input -> the relabel.
    """
    cinv = RegisterMapper(
        canon
    ).inverse_map()  # (free, k) -> LaneRegister  (the canonical machine)
    rel: dict[int, int] = {}
    for coord, lr in cinv.items():
        if coord[k_axis] == 0:
            rel[coord[free_axis]] = supplied_fwd[(lr.lane, lr.register)][free_axis]
    return rel


@dataclass(frozen=True)
class Diagnostic:
    """A pure OBSERVATION about a layout/pair -- ``severity`` in ``{"ok", "warning", "error"}`` + a
    ``message``. Diagnostics never mutate a distribution; they only report what is true of the labels.
    """

    severity: str
    message: str
