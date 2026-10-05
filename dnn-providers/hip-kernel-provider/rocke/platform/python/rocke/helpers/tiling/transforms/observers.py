# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Transform OBSERVERS -- the read-only, IR-free analysis toolbox (public tier).

Every function here is a pure observation over layout LABELS: it classifies an edge, checks MMA
safety, or derives the C label map. None mutates a distribution or emits IR. They are the seam that
makes the Plan/Pipeline glass-box (an author can ask "is this edge a reorder / reposition / cross-lane,
and is this A/B pair sound?" before building on it), so they sit on top of the neutral ``_core`` and
carry no "machinery" import.

MMA safety (:func:`validate_operands`, :func:`operand_soundness`, :func:`mma_pair_compatible`): the
MFMA/WMMA hardware multiply-accumulates by pairing A-slot-s with B-slot-s and summing over K. The sum
is order-independent, so the K-slot ordering is FREE -- the sole CROSS-OPERAND constraint is that A and
B share the SAME positional K-distribution. Per-operand soundness (one M per output on A, one N on B) is
the OTHER half of the sound MAC. Correctness SOT: ``docs/mma_is_machinery.md`` (the three-condition sound
MAC); edge-kind SOT: ``docs/label_flow_and_transforms.md``.
"""

from __future__ import annotations

from ..encoding import WarpDistributionEncoding
from ..register_mapper import RegisterMapper
from ._core import (
    Diagnostic,
    ReorderPlan,
    TransformPlan,
    _atom_k_signature,
    _axis_permutation,
    _classify_maps,
    _dword_aligned,
    _free_relabel,
    _kdist_from_fwd,
    as_forward_map,
    k_distribution,
    name_permutation,
)


def classify_transform(source, target) -> TransformPlan:
    """Solve the delta ``source -> target`` and classify it (``reorder`` / ``cross_lane``). ``source``/
    ``target`` may each be a ``WarpDistributionEncoding`` OR a forward map ``{(lane,reg)->coord}`` (from
    another stage).

    Raises ``ValueError`` if the two describe different fragment dimensions or different element sets
    (no transform exists between them).
    """
    return _classify_maps(as_forward_map(source), as_forward_map(target))


def reorder_between(coalesced_fwd, requested_fwd, *, pack) -> "ReorderPlan | None":
    """Detect + classify the IN-REGISTER reorder bridging a COALESCED (memory-order) register frame to the
    REQUESTED (consumer-order) frame. Both args are ``{(lane,reg)->coord}`` forward maps (or encodings)
    holding the **same per-lane data**; ``pack`` = elements per 32-bit register (f16=2, f32=1, f8=4 -- REQUIRED,
    no silent default: it decides dword vs sub-dword). Returns ``None`` when the two frames are already equal
    (no reorder -> no panel). GENERIC: nothing is hardcoded -- :func:`classify_transform` is ground truth and
    :func:`name_permutation` only LABELS the permutation it finds. Raises if the two frames do NOT hold the
    same per-lane data (that means the wrong pair was passed -- a within-lane reorder keeps each lane's element
    set; never silently reinterpret it as cross-lane)."""
    src = as_forward_map(coalesced_fwd)
    tgt = as_forward_map(requested_fwd)
    def _per_lane(m):
        d: dict[int, set] = {}
        for (l, _r), c in m.items():
            d.setdefault(l, set()).add(c)
        return d
    if _per_lane(src) != _per_lane(tgt):
        raise ValueError(
            "reorder_between: the two frames do not hold the same per-lane data -- the wrong pair was passed "
            "(a within-lane reorder keeps each lane's element set). Fix the inputs; do not force a reorder.")
    plan = classify_transform(src, tgt)
    if plan.tier == "cross_lane":                              # element changes lane -> NOT the two-panel model
        return ReorderPlan("cross_lane", None, "cross_lane (DPP / ds_bpermute)", 0,
                           "cross_lane: element changes lane -- needs ds_bpermute/DPP (last resort)")
    perm = plan.permutation
    if perm == tuple(range(len(perm))):
        return None                                            # identity -> no reorder
    label = name_permutation(perm)
    if _dword_aligned(perm, pack):
        return ReorderPlan("reorder (dword)", perm, label, 0,
                           f"reorder (dword-aligned): {label} -- register renumber, ~0 v_perm")
    ndword = -(-len(perm) // max(1, pack))                     # ceil(nregs / pack): dest dwords repacked
    return ReorderPlan(f"reorder (sub-dword, {pack}x)", perm, label, ndword,
                       f"reorder (sub-dword, {pack} elem/dword): {label}, ~{ndword} v_perm_b32/lane")


def describe_edge(src, tgt=None, *, src_dims=("d0", "d1"), tgt_dims=None, to_space=None, relabel=False):
    """Classify + describe ONE pipeline edge -> ``(kind, why)``. A datum's LABEL is its identity and flows
    INVARIANT across every space; it changes ONLY on an explicit ``relabel`` edge. Kinds:

    - ``identity``   -- ``src == tgt``; nothing changed.
    - ``reposition`` -- register -> memory space (``to_space`` set): label INVARIANT; only the datum's physical
      storage-axis alignment / address changes. NEVER a *label* transpose -- the memref's axis order is
      positional, not a relabel of the datum. Free -- absorbed into addressing; renders INTO the space.
    - ``reorder``    -- register->register lane-uniform register permutation (cost: register shuffle).
    - ``cross_lane`` -- register->register element changes lane / non-uniform (cost: cross-lane movement).
    - ``relabel``    -- EXPLICIT (``relabel=True``) axis-permutation + rename that *changes the label*. The ONE
      sanctioned label change: reinterpreting a FINISHED tile's axes when it is reused as a downstream input
      (e.g. a computed C ``(M,N)`` re-viewed as an input ``(M,K)``/``(N,K)``). NOT AB-swap -- that is a
      machine-input ROUTING (operand->opposite slot), labels INVARIANT, C DERIVES; not a relabel and not an
      edge kind. Raises unless ``src->tgt`` is a consistent axis permutation (``pi``).

    ``src``/``tgt`` are ``WarpDistributionEncoding`` or forward maps. ``src_dims``/``tgt_dims`` name the axes
    (used to phrase the ``why``). This is the single classifier every arrow label routes through, so a free
    edge is never a silent no-op (the ``why`` is mandatory on ``reposition``/``relabel``).
    SOT: ``docs/label_flow_and_transforms.md`` (storage != label; source-swap != relabel)."""
    sd, td = tuple(src_dims), tuple(tgt_dims) if tgt_dims is not None else tuple(src_dims)
    if relabel:
        if tgt is None:
            raise ValueError("an explicit relabel needs a target layout")
        pi = _axis_permutation(as_forward_map(src), as_forward_map(tgt))
        if pi is None:
            raise ValueError("declared relabel is not a consistent axis permutation/rename of the source")
        swapped = pi != tuple(range(len(pi)))
        how = "axes swapped" if swapped else "renamed"
        return "relabel", f"{sd}->{td} reinterpret ({how}); free (label CHANGES)"
    if to_space is not None:
        # A store/read is a REPOSITION: the datum's physical storage-axis alignment / address changes, its
        # LABEL never does. NEVER phrase this as a label transpose (`(M,K)->(K,M)`): the memref's axis order
        # is POSITIONAL, not a relabel of the datum. SOT: docs/label_flow_and_transforms.md.
        return "reposition", f"place into {to_space}; free (label invariant)"
    s, t = as_forward_map(src), as_forward_map(tgt)
    if s == t:
        return "identity", "no change"
    tier = classify_transform(s, t).tier
    why = {"reorder": "lane-uniform register permutation (register shuffle)",
           "cross_lane": "element changes lane (cross-lane: LDS / DPP)"}[tier]
    return tier, why


def validate_operands(
    a_layout: WarpDistributionEncoding,
    b_layout: WarpDistributionEncoding,
    k_axis: int = 1,
    a_free_atoms: int = 1,
    b_free_atoms: int = 1,
) -> tuple[bool, str]:
    """MMA safety: A and B must share the SAME positional K-distribution PER ATOM.

    This is the PAIRWISE half of the sound MAC (correctness SOT: ``docs/mma_is_machinery.md``). The
    MFMA/WMMA hardware pairs A-slot-s with B-slot-s and sums over K; the sum is order-independent, so the
    K-slot ordering is FREE -- the sole CROSS-OPERAND constraint is that A and B agree on which logical K
    sits in each paired slot. Per-operand soundness (M/N fixed per output) is the other half, checked by
    :func:`operand_soundness`. M/N register order is unconstrained, and K need NOT match any "canonical" atom
    order (interleaved-A x interleaved-B is valid iff their K-dists match), so a positional A-vs-B K-match is
    sufficient here for a correct contraction.

    ``a_free_atoms`` / ``b_free_atoms`` are the free-dim atom counts (M-atoms for A, N-atoms for B) the
    driver walks. A rectangular wave tile has ``a_free_atoms != b_free_atoms``, so the WHOLE-fragment
    K-lists differ in length even though every issued atom pairs the SAME K. Comparison is therefore PER
    ATOM: reduce each operand to its atom-K signature (defaults of 1 make this the whole-fragment compare,
    unchanged for square/single tiles).

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


def diagnose_k_match(a_enc, b_enc) -> Diagnostic:
    """DIAGNOSTIC (observer, NEVER a mutator): do A's and B's LABELS share a K-distribution, so the MMA is
    meaningful? Accepts an encoding OR a forward map for each. Judged on the labels at atom granularity
    (per-lane K over the common register prefix, which handles rectangular waves). Reports only -- it never
    reorders or falls back to canonical:

    - ``ok``      -- ``k_distribution(A) == k_distribution(B)`` position-for-position.
    - ``warning`` -- K order differs but each lane holds the SAME K set: reconcilable by an IN-REGISTER
      reorder (the transform is named, not performed).
    - ``error``   -- lanes hold DIFFERENT K sets: no in-register reorder reconciles them.

    Correctness SOT: ``docs/mma_is_machinery.md`` (this is the pairwise K-match, sound-MAC condition 3).
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

    Correctness SOT: ``docs/mma_is_machinery.md`` (this is per-operand soundness, sound-MAC conditions 1-2).
    """
    sup = as_forward_map(layout)
    cm = RegisterMapper(canon)
    n_slots = cm.num_lanes * cm.num_vector_items
    if len(sup) != n_slots:
        # Dimension pre-check: a custom fragment with the wrong (lanes x regs) count would otherwise
        # KeyError below (indexing sup at a canonical slot it does not have). Give a clean diagnostic.
        return Diagnostic("error",
            f"{role} not sound: layout has {len(sup)} (lane,reg) slots but the machine has {n_slots} "
            "-- dimension mismatch (wrong fragment for this atom)")
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
                f"{can_k} -- malformed/duplicated K (rule 3: well-formed K)")
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
    valid, meaningful MMA; else the first failing operand's soundness error, or the K-match diagnostic.
    The full sound MAC = per-operand soundness (conditions 1-2) + pairwise K-match (3); correctness SOT:
    ``docs/mma_is_machinery.md``."""
    for enc, canon, role in ((a_enc, a_canon, "A"), (b_enc, b_canon, "B")):
        d = operand_soundness(enc, canon, k_axis=k_axis, role=role)
        if d.severity != "ok":
            return d
    return diagnose_k_match(a_enc, b_enc)
