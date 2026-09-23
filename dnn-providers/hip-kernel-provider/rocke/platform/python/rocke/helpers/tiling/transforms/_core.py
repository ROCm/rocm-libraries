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
from dataclasses import dataclass

from ..encoding import WarpDistributionEncoding
from ..register_mapper import RegisterMapper


def interleave_idx(gather: int, stride: int, count: int, length: int | None = None) -> tuple[int, ...]:
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
        raise ValueError(f"interleave_idx length must be a multiple of count -- length={length}, count={count}")
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
    for stride in range(2, n):                                  # count == n (single block); gather==1
        if n % stride:
            continue
        if tuple(perm) == interleave_idx(1, stride, n):
            inner = n // stride
            return f"interleave_idx(1, {stride}, {n})   dst = (r%{inner})*{stride} + (r//{inner})"
    return f"reorder perm={perm}"


def _forward_map(enc: WarpDistributionEncoding) -> tuple[dict[tuple[int, int], tuple[int, ...]], RegisterMapper]:
    """{(lane, register) -> matrix coordinate} for every slot of the encoding."""
    m = RegisterMapper(enc)
    fmap = {
        (lane, reg): m.matrix_coordinates(lane, reg)
        for lane in range(m.num_lanes)
        for reg in range(m.num_vector_items)
    }
    return fmap, m


def k_distribution(enc: WarpDistributionEncoding, k_axis: int = 1) -> tuple[tuple[int, ...], ...]:
    """Project the encoding onto its K axis: per lane, the tuple of K coordinates by register slot.

    ``k_axis`` is the contraction axis index (1 for both A=(M,K) and B=(N,K)). Two operands are
    K-aligned iff their ``k_distribution`` are equal position-for-position.
    """
    m = RegisterMapper(enc)
    return tuple(
        tuple(m.matrix_coordinates(lane, reg)[k_axis] for reg in range(m.num_vector_items))
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
    return {(l, r): tuple(rm.matrix_coordinates(l, r))
            for l in range(rm.num_lanes) for r in range(rm.num_vector_items)}


@dataclass(frozen=True)
class TransformPlan:
    """The classified delta between two layouts. ``tier`` is ``"reorder"`` or ``"cross_lane"``.
    For ``reorder``, ``permutation[src_reg] == dst_reg`` (lane-uniform). ``reason`` explains a
    ``cross_lane`` outcome."""

    tier: str
    permutation: tuple[int, ...] | None
    reason: str


def _classify_maps(smap: dict[tuple[int, int], tuple[int, ...]],
                   tmap: dict[tuple[int, int], tuple[int, ...]]) -> TransformPlan:
    """Classify the delta ``smap -> tmap`` between two forward maps ``{(lane,reg)->coord}`` (the IR-free
    core; works on labels from an encoding OR from another stage).

    - ``reorder``    -- every lane keeps the same element set AND the source->target register permutation
      is identical on every lane. Emittable as a single compile-time register permutation.
    - ``cross_lane`` -- some element changes lanes, or the on-lane permutation is not lane-uniform.

    Raises ``ValueError`` if the maps describe different fragment dimensions or different element sets.
    """
    s_lanes = {l for l, _ in smap}; s_regs = {r for _, r in smap}
    t_lanes = {l for l, _ in tmap}; t_regs = {r for _, r in tmap}
    if (len(s_lanes), len(s_regs)) != (len(t_lanes), len(t_regs)):
        raise ValueError(
            "cannot transform between fragments of different dimensions -- source is "
            f"{len(s_lanes)}x{len(s_regs)} (lanes x regs), target is {len(t_lanes)}x{len(t_regs)}"
        )
    if set(smap.values()) != set(tmap.values()):
        raise ValueError(
            "cannot transform between layouts that hold different elements -- source and target "
            "describe different tiles (check shapes / that they are the same logical tile)"
        )

    target_of: dict[tuple[int, ...], tuple[int, int]] = {}
    for (lane, reg), coord in tmap.items():
        target_of.setdefault(coord, (lane, reg))

    per_lane_perm: dict[int, dict[int, int]] = {}
    for (lane, reg), coord in smap.items():
        dst_lane, dst_reg = target_of[coord]
        if dst_lane != lane:
            return TransformPlan(
                "cross_lane", None,
                f"element {coord} moves lane {lane}->{dst_lane}; needs cross-lane movement",
            )
        per_lane_perm.setdefault(lane, {})[reg] = dst_reg

    reference = per_lane_perm[min(per_lane_perm)]
    for lane, perm in per_lane_perm.items():
        if perm != reference:
            return TransformPlan(
                "cross_lane", None,
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
    tier: str                                  # "reorder (dword)" | "reorder (sub-dword, Nx)" | "cross_lane"
    permutation: tuple[int, ...] | None
    label: str                                 # name_permutation(perm), e.g. "interleave_idx(1, 8, 32) ..."
    vperm_per_lane: int
    cost: str


def _dword_aligned(perm: tuple[int, ...], pack: int) -> bool:
    """True iff ``perm`` moves whole ``pack``-sized dword blocks as units (register renumber); False iff it
    splits a dword (a sub-dword repack -- a real ``v_perm_b32``). ``pack`` = elements per 32-bit register."""
    if pack <= 1:
        return True
    for b in range(0, len(perm), pack):
        if perm[b] % pack != 0 or any(perm[b + j] != perm[b] + j for j in range(pack)):
            return False
    return True


def _axis_permutation(smap: dict[tuple[int, int], tuple[int, ...]],
                      tmap: dict[tuple[int, int], tuple[int, ...]]) -> tuple[int, ...] | None:
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


def _atom_k_signature(
    per_lane_k: tuple[tuple[int, ...], ...], atoms: int, role: str
) -> tuple[tuple[tuple[int, ...], ...], str]:
    """Reduce a whole-fragment per-lane K-distribution to its PER-ATOM K signature.

    A wave-tile fragment tiles ``atoms`` free-dim atoms (M-atoms for A, N-atoms for B) along the
    register axis, each carrying the SAME K sequence (the free dim only permutes M/N, never K). So the
    per-lane K-list is that atom K-signature repeated ``atoms`` times. Chop it back to one atom's worth
    and verify the repeats are consistent. Returns ``(signature, reason)``; ``reason`` non-empty on a
    malformed (non-uniform) repeat, which means the fragment is not a clean atom tiling.
    """
    sig: list[tuple[int, ...]] = []
    for lane, kl in enumerate(per_lane_k):
        if atoms <= 0 or len(kl) % atoms:
            return (), (
                f"{role} fragment lane {lane} has {len(kl)} K-slots, not divisible by {atoms} "
                f"{role}-atoms -- not a clean atom tiling"
            )
        width = len(kl) // atoms
        chunks = [kl[i * width:(i + 1) * width] for i in range(atoms)]
        if any(c != chunks[0] for c in chunks):
            return (), (
                f"{role} fragment lane {lane} K-slots {kl} are not a uniform repeat across "
                f"{atoms} atoms -- K differs between atoms"
            )
        sig.append(chunks[0])
    return tuple(sig), ""


def _kdist_from_fwd(fwd: dict[tuple[int, int], tuple[int, ...]], k_axis: int = 1) -> tuple[tuple[int, ...], ...]:
    """Per-lane K sequence (by ascending register) from a forward map -- the ``k_distribution`` of a map."""
    lanes = sorted({l for l, _ in fwd})
    return tuple(tuple(fwd[(l, r)][k_axis] for r in sorted(rr for (ll, rr) in fwd if ll == l)) for l in lanes)


def _free_relabel(
    canon: WarpDistributionEncoding, supplied_fwd: dict[tuple[int, int], tuple[int, ...]],
    free_axis: int = 0, k_axis: int = 1,
) -> dict[int, int]:
    """Map each canonical free index -> the SUPPLIED layout's free label at that index's canonical K=0 slot.

    The machine treats a physical register as its CANONICAL ``(free, K)``; the supplied distribution's
    label sitting on that register is what actually FLOWS through the machine (docs/mma_is_machinery.md).
    Canonical input -> identity; a relabeled input -> the relabel.
    """
    cinv = RegisterMapper(canon).inverse_map()          # (free, k) -> LaneRegister  (the canonical machine)
    rel: dict[int, int] = {}
    for coord, lr in cinv.items():
        if coord[k_axis] == 0:
            rel[coord[free_axis]] = supplied_fwd[(lr.lane, lr.register)][free_axis]
    return rel


@dataclass(frozen=True)
class Diagnostic:
    """A pure OBSERVATION about a layout/pair -- ``severity`` in ``{"ok", "warning", "error"}`` + a
    ``message``. Diagnostics never mutate a distribution; they only report what is true of the labels."""

    severity: str
    message: str
