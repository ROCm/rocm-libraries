# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Fragment-transform solver + MMA-safety + the register-reorder verb that realises a plan.

Moving a fragment between two layouts (canonical <-> interleaved, on-lane register reorders) is a
DELTA between two :class:`~rocke.helpers.tiling.encoding.WarpDistributionEncoding` values. This
module solves that delta GENERALLY from the two encodings (it never consults a hand-registered
table of named transform pairs -- that is the rocWMMA failure mode where a missing pair silently
returns wrong registers). The closed-form :func:`interleave_idx` from the reference layout tables
is kept only as a fast path + a test oracle for the solver.

Two outcomes (plus a hard reject):
- ``reorder``  -- every lane keeps the same element set AND the source->target register permutation
  is identical on every lane. Emittable as a single compile-time register permutation.
- ``cross_lane`` -- some element changes lanes, or the on-lane permutation is not lane-uniform.
  Needs cross-lane data movement (DPP / ds_bpermute / LDS); deferred (the verb rejects it).
- reject (``ValueError``) -- the two descs describe different fragment dimensions or different
  element sets, so no transform between them exists.

MMA safety (:func:`validate_operands`): the MFMA/WMMA hardware multiply-accumulates by pairing
A-slot-s with B-slot-s and summing over K. The sum is order-independent, so the K-slot ordering is
FREE -- the ONLY validity constraint is that A and B share the SAME positional K-distribution (they
agree on which logical K sits in each paired slot). M/N register order is free, and K need NOT match
any "canonical" atom order: interleaved-A x interleaved-B is valid as long as their K-dists match.
Fragments reaching the check are register-reorders of the atom (same lane ownership by construction),
so a positional A-vs-B K-match is sufficient. It is what ``TileMma.__call__`` uses to reject a
mismatched operand pair with a constructive message.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .encoding import WarpDistributionEncoding
from .fragments import Fragment, TileDesc
from .register_mapper import RegisterMapper

__all__ = ["TransformPlan", "interleave_idx", "k_distribution", "classify_transform", "validate_operands",
           "derive_c_distribution", "Diagnostic", "diagnose_k_match", "as_forward_map",
           "operand_soundness", "mma_compatible", "mma_pair_compatible", "transform_fragment"]


def interleave_idx(gather: int, stride: int, count: int, length: int | None = None) -> tuple[int, ...]:
    """Register-index permutation ``interleave_idx<gather, stride, count>`` from the reference
    layout tables. Within each ``count``-sized block the local index is transposed as an
    ``(stride, count//stride)`` grid. ``target[i]`` for ``i in range(length)`` (``length`` defaults
    to ``count`` and must be a multiple of ``count``). A NOP when ``stride in {1, count}``.

    Only ``gather == 1`` (the gfx90a A/B case) is implemented; ``gather > 1`` (the 32x32-accumulator
    ``interleave<4,8,16>`` grouped form) is a deferred milestone -- it needs the acc-transform tables.
    """
    if gather != 1:
        raise NotImplementedError(
            f"interleave_idx gather>1 not supported yet -- gather={gather} "
            "(grouped form, e.g. 32x32-acc interleave<4,8,16>); deferred"
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
    core; works on labels from an encoding OR from another stage). See the module docstring for tiers.

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


def classify_transform(source, target) -> TransformPlan:
    """Solve the delta ``source -> target`` and classify it (see the module docstring). ``source``/``target``
    may each be a ``WarpDistributionEncoding`` OR a forward map ``{(lane,reg)->coord}`` (from another stage).

    Raises ``ValueError`` if the two describe different fragment dimensions or different element sets
    (no transform exists between them).
    """
    return _classify_maps(as_forward_map(source), as_forward_map(target))


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


def validate_operands(
    a_layout: WarpDistributionEncoding,
    b_layout: WarpDistributionEncoding,
    k_axis: int = 1,
    a_free_atoms: int = 1,
    b_free_atoms: int = 1,
) -> tuple[bool, str]:
    """MMA safety: A and B must share the SAME positional K-distribution PER ATOM.

    The MFMA/WMMA hardware pairs A-slot-s with B-slot-s and sums over K; the sum is
    order-independent, so the K-slot ordering is FREE -- the sole validity constraint is that A and B
    agree on which logical K sits in each paired slot. M/N register order is unconstrained, and K need
    NOT match any "canonical" atom order (interleaved-A x interleaved-B is valid iff their K-dists
    match). Fragments reaching here are register-reorders of the atom (same lane ownership by
    construction), so a positional A-vs-B K-match is sufficient for a correct contraction.

    ``a_free_atoms`` / ``b_free_atoms`` are the free-dim atom counts (M-atoms for A, N-atoms for B) the
    driver walks. A rectangular wave tile has ``a_free_atoms != b_free_atoms``, so the WHOLE-fragment
    K-lists differ in length (A repeats its atom-K ``m_sub`` times, B ``n_sub`` times) even though every
    issued atom pairs the SAME K. Comparison is therefore PER ATOM: reduce each operand to its atom-K
    signature (defaults of 1 make this the whole-fragment compare, unchanged for square/single tiles).

    Returns ``(ok, reason)`` with a constructive reason naming the first divergent lane.
    """
    a_sig, a_reason = _atom_k_signature(k_distribution(a_layout, k_axis), a_free_atoms, "A")
    if a_reason:
        return False, a_reason
    b_sig, b_reason = _atom_k_signature(k_distribution(b_layout, k_axis), b_free_atoms, "B")
    if b_reason:
        return False, b_reason
    if len(a_sig) != len(b_sig):
        return False, (
            f"A fragment spans {len(a_sig)} lanes but B spans {len(b_sig)} -- operands not "
            "MMA-compatible"
        )
    for lane, (ak, bk) in enumerate(zip(a_sig, b_sig)):
        if ak != bk:
            return False, (
                f"A/B fragments are not K-aligned: lane {lane} holds A-atom-K {ak} but B-atom-K {bk}. "
                "transform_fragment one operand to match the other's K-distribution first."
            )
    return True, "ok"


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


def derive_c_distribution(
    a_enc,
    b_enc,
    *,
    a_canon: WarpDistributionEncoding,
    b_canon: WarpDistributionEncoding,
    c_canon: WarpDistributionEncoding,
) -> dict[tuple[int, int], tuple[int, int]]:
    """Flow the SUPPLIED A/B logical labels through the FIXED canonical machine to label C.

    ``A = (M, K)``, ``B = (N, K)``, ``C = (M, N)``. ``a_enc``/``b_enc`` are the distributions you hand in --
    a ``WarpDistributionEncoding`` OR a pre-populated forward map ``{(lane,reg)->coord}`` from another stage.
    The canonical references ARE the machine (the atom's fixed physical coupling). For each physical C slot
    whose canonical identity is ``(Mc, Nc)``, its label is ``(the M that A holds where canonical-A holds row
    Mc, the N that B holds where canonical-B holds col Nc)`` -- labels from A and B flowing through the
    machine into C. Deterministic for ANY inputs; no compatibility judgement, no reordering.

    Returns ``derived_fwd``: ``{(lane, reg) -> (m, n)}`` for every physical C slot.
    """
    pi_m = _free_relabel(a_canon, as_forward_map(a_enc))
    pi_n = _free_relabel(b_canon, as_forward_map(b_enc))
    cm = RegisterMapper(c_canon)
    derived_fwd: dict[tuple[int, int], tuple[int, int]] = {}
    for lane in range(cm.num_lanes):
        for reg in range(cm.num_vector_items):
            mc, nc = cm.matrix_coordinates(lane, reg)[:2]
            derived_fwd[(lane, reg)] = (pi_m.get(mc, mc), pi_n.get(nc, nc))
    return derived_fwd


@dataclass(frozen=True)
class Diagnostic:
    """A pure OBSERVATION about a layout/pair -- ``severity`` in ``{"ok", "warning", "error"}`` + a
    ``message``. Diagnostics never mutate a distribution; they only report what is true of the labels."""

    severity: str
    message: str


def diagnose_k_match(a_enc, b_enc) -> Diagnostic:
    """DIAGNOSTIC (observer, NEVER a mutator): do A's and B's LABELS share a K-distribution, so the MMA is
    meaningful? Accepts an encoding OR a forward map for each. Judged on the labels at atom granularity
    (per-lane K over the common register prefix, which handles rectangular waves). Reports only -- it never
    reorders or falls back to canonical:

    - ``ok``      -- ``k_distribution(A) == k_distribution(B)`` position-for-position.
    - ``warning`` -- K order differs but each lane holds the SAME K set: reconcilable by an IN-REGISTER
      reorder (the transform is named, not performed).
    - ``error``   -- lanes hold DIFFERENT K sets: no in-register reorder reconciles them.
    """
    ka, kb = _kdist_from_fwd(as_forward_map(a_enc)), _kdist_from_fwd(as_forward_map(b_enc))
    if len(ka) != len(kb):
        return Diagnostic("error", f"A spans {len(ka)} lanes but B spans {len(kb)} -- not the same wave")
    n = min((len(ka[0]) if ka else 0), (len(kb[0]) if kb else 0))
    mism = [lane for lane in range(len(ka)) if ka[lane][:n] != kb[lane][:n]]
    if not mism:
        return Diagnostic("ok", "A.K == B.K (labels K-aligned; valid MMA)")
    lane = mism[0]
    if all(sorted(ka[l][:n]) == sorted(kb[l][:n]) for l in range(len(ka))):
        return Diagnostic("warning",
                          f"A.K != B.K positionally (lane {lane}: A {ka[lane][:n]} vs B {kb[lane][:n]}); "
                          "same K set per lane -> reconcilable by an in-register reorder")
    return Diagnostic("error",
                      f"A.K and B.K hold different K sets (lane {lane}: A {sorted(ka[lane][:n])} vs "
                      f"B {sorted(kb[lane][:n])}) -- no in-register reorder reconciles them")


def operand_soundness(layout, canon: WarpDistributionEncoding, *, free_axis: int = 0, k_axis: int = 1,
                      role: str = "operand") -> Diagnostic:
    """DIAGNOSTIC (observer, NEVER a mutator): is ONE operand's LOGICAL-LABEL layout a mathematically
    sound MMA operand? Judges the LABELS ONLY, against the FIXED machine (``canon``); it never checks the
    machine and never reorders. ``layout`` is the logical data -- a ``WarpDistributionEncoding`` OR a
    forward map ``{(lane,reg)->coord}`` from another stage.

    The machine couples physical positions; the positions feeding one output (canonical free-coord ``Mc``)
    are those the machine assigns to that row across K. Rule 2/3 on the labels sitting there:
    - every A label's M (B label's N) must be FIXED -- one free-label across the row's positions;
    - the K-labels must be WELL-FORMED -- the same multiset as the machine's contraction K-set for that row.
    ``ok`` iff both hold on every machine output-row; else ``error`` naming the first offending row.
    """
    sup = as_forward_map(layout)
    cm = RegisterMapper(canon)
    rows: dict[int, list[tuple[int, int]]] = {}
    for lane in range(cm.num_lanes):
        for reg in range(cm.num_vector_items):
            rows.setdefault(cm.matrix_coordinates(lane, reg)[free_axis], []).append((lane, reg))
    for cf in sorted(rows):
        cells = rows[cf]
        frees = {sup[c][free_axis] for c in cells}
        if len(frees) != 1:
            return Diagnostic("error",
                f"{role} not sound: machine output-row {cf} carries {len(frees)} free-labels "
                f"{sorted(frees)} -- M/N not fixed along the contraction (rule 2)")
        sup_k = sorted(sup[c][k_axis] for c in cells)
        can_k = sorted(cm.matrix_coordinates(l, r)[k_axis] for (l, r) in cells)
        if sup_k != can_k:
            return Diagnostic("error",
                f"{role} not sound: machine output-row {cf} K-labels {sup_k} != contraction set "
                f"{can_k} -- malformed/duplicated K (rule 2)")
    return Diagnostic("ok", f"{role} sound: fixed free-label + well-formed K on every machine output-row")


def mma_compatible(layout, canon: WarpDistributionEncoding, *, free_axis: int = 0, k_axis: int = 1,
                   role: str = "operand") -> Diagnostic:
    """yes/no: is this LOGICAL-LABEL layout MMA-compatible, and if not, can a transform MAKE-IT-SO?
    ``ok`` -- compatible (sound, :func:`operand_soundness`). Otherwise classify the fix toward a
    known-sound target (``canon``): ``warning`` -- an in-register ``reorder`` makes-it-so (no data
    movement); ``error`` -- needs ``cross_lane`` movement, or no transform reconciles it. Observer only.
    """
    snd = operand_soundness(layout, canon, free_axis=free_axis, k_axis=k_axis, role=role)
    if snd.severity == "ok":
        return Diagnostic("ok", f"{role} MMA-compatible ({snd.message})")
    try:
        plan = _classify_maps(as_forward_map(layout), as_forward_map(canon))
    except ValueError as e:
        return Diagnostic("error", f"{role} NOT MMA-compatible; no transform reconciles it -- {snd.message} [{e}]")
    if plan.tier == "reorder":
        return Diagnostic("warning",
            f"{role} NOT MMA-compatible, but an in-register reorder makes-it-so "
            f"(permutation {plan.permutation}) -- {snd.message}")
    return Diagnostic("error",
        f"{role} NOT MMA-compatible; needs cross-lane movement to make-it-so ({plan.reason})")


def mma_pair_compatible(a_enc, b_enc, *, a_canon: WarpDistributionEncoding,
                        b_canon: WarpDistributionEncoding, k_axis: int = 1) -> Diagnostic:
    """Full A x B check: BOTH operands sound (:func:`operand_soundness`) AND their K-dists match
    positionally (the relationship, :func:`diagnose_k_match`). Observer only. ``ok`` iff the pair is a
    valid, meaningful MMA; else the first failing operand's soundness error, or the K-match diagnostic."""
    for enc, canon, role in ((a_enc, a_canon, "A"), (b_enc, b_canon, "B")):
        d = operand_soundness(enc, canon, k_axis=k_axis, role=role)
        if d.severity != "ok":
            return d
    return diagnose_k_match(a_enc, b_enc)


def transform_fragment(b: Any, fragment: Fragment, target_desc: TileDesc) -> Fragment:
    """Retarget `fragment` to `target_desc`'s layout via the cheapest op sequence -- the verb that
    REALISES what :func:`classify_transform` plans. Lives here (not in ``emit``) so the memory verbs
    never import the solver; the IRBuilder ``b`` is duck-typed, so this module still takes NO IR
    import (the tests drive it with a plain-int builder, exactly like the address-map replay).

    The delta between the two layouts is solved from their encodings and classified: a `reorder`
    (same element set per lane, one lane-uniform register permutation) is emitted as a compile-time
    register shuffle (`vec_extract`/`vec_insert`, element-granular so it is correct for any dtype
    packing). A `cross_lane` delta (an element changes lanes) needs cross-lane movement (DPP /
    ds_bpermute / LDS) and raises `NotImplementedError` -- the reserved seam a future cross-lane
    realisation fills. Same element set required (else `ValueError`). Dtype is carried through unchanged.
    """
    plan = classify_transform(fragment.tile_desc.layout, target_desc.layout)
    if plan.tier != "reorder":
        raise NotImplementedError(
            f"cross-lane fragment transform is not supported yet -- {plan.reason}"
        )
    n = target_desc.register_count
    out = b.zero_vec(fragment.dtype, n)
    for src_reg, dst_reg in enumerate(plan.permutation):
        out = b.vec_insert(out, b.vec_extract(fragment.value, src_reg), dst_reg)
    return Fragment(target_desc, fragment.dtype, out)
