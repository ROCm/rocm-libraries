# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Transform OBSERVERS -- the read-only, IR-free analysis toolbox (public tier).

Every function here is a pure observation over layout LABELS: it classifies an edge, checks MMA
safety, or derives the C label map. None mutates a distribution or emits IR. They are the seam that
makes the Plan/Pipeline glass-box (an author can ask "is this edge a reorder / reposition / cross-lane,
and is this A/B pair sound?" before building on it), so they sit on top of the neutral ``_core`` and
carry no "machinery" import.

MMA safety (:func:`mma_pair_k_aligned`, :func:`mma_operand_layout_sound`, :func:`mma_pair_compatible`): the
MFMA/WMMA hardware multiply-accumulates by pairing A-slot-s with B-slot-s and summing over K. The sum
is order-independent, so the K-slot ordering is FREE -- the sole CROSS-OPERAND constraint is that A and
B share the SAME positional K-distribution. Per-operand soundness (one M per output on A, one N on B) is
the OTHER half of the sound MAC. :func:`mma_accumulator_flow_consistent` is the separate C store-coordinate
gate (passed C == machine-derived C), NOT a sound-MAC condition. Correctness SOT: ``docs/mma_is_machinery.md``
(the three-condition sound MAC); edge-kind SOT: ``docs/label_flow_and_transforms.md``.
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
            "(a within-lane reorder keeps each lane's element set). Fix the inputs; do not force a reorder."
        )
    plan = classify_transform(src, tgt)
    if plan.tier == "cross_lane":  # element changes lane -> NOT the two-panel model
        return ReorderPlan(
            "cross_lane",
            None,
            "cross_lane (DPP / ds_bpermute)",
            0,
            "cross_lane: element changes lane -- needs ds_bpermute/DPP (last resort)",
        )
    perm = plan.permutation
    if perm == tuple(range(len(perm))):
        return None  # identity -> no reorder
    label = name_permutation(perm)
    if _dword_aligned(perm, pack):
        return ReorderPlan(
            "reorder (dword)",
            perm,
            label,
            0,
            f"reorder (dword-aligned): {label} -- register renumber, ~0 v_perm",
        )
    ndword = -(-len(perm) // max(1, pack))  # ceil(nregs / pack): dest dwords repacked
    return ReorderPlan(
        f"reorder (sub-dword, {pack}x)",
        perm,
        label,
        ndword,
        f"reorder (sub-dword, {pack} elem/dword): {label}, ~{ndword} v_perm_b32/lane",
    )


def describe_edge(
    src, tgt=None, *, src_dims=("d0", "d1"), tgt_dims=None, to_space=None, relabel=False
):
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
    SOT: ``docs/label_flow_and_transforms.md`` (storage != label; source-swap != relabel).
    """
    sd, td = tuple(src_dims), (
        tuple(tgt_dims) if tgt_dims is not None else tuple(src_dims)
    )
    if relabel:
        if tgt is None:
            raise ValueError("an explicit relabel needs a target layout")
        pi = _axis_permutation(as_forward_map(src), as_forward_map(tgt))
        if pi is None:
            raise ValueError(
                "declared relabel is not a consistent axis permutation/rename of the source"
            )
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
    why = {
        "reorder": "lane-uniform register permutation (register shuffle)",
        "cross_lane": "element changes lane (cross-lane: LDS / DPP)",
    }[tier]
    return tier, why


def mma_pair_k_aligned(
    a,
    b,
    *,
    a_free_atoms: int = 1,
    b_free_atoms: int = 1,
    k_axis: int = 1,
) -> Diagnostic:
    """DIAGNOSTIC (observer): do A and B share the SAME positional K PER ATOM -- the PAIRWISE half of the
    sound MAC (correctness SOT: ``docs/mma_is_machinery.md``, condition 3)? Accepts an encoding OR a forward
    map ``{(lane,reg)->coord}`` for each.

    The MFMA/WMMA hardware pairs A-slot-s with B-slot-s and sums over K; the sum is order-independent, so the
    K ORDER is FREE -- the sole cross-operand constraint is that A and B agree on which logical K sits in
    each paired slot. (Per-operand soundness -- one M/N per output -- is the OTHER half,
    :func:`mma_operand_layout_sound`.) M/N register order is unconstrained, and K need NOT match any
    "canonical" atom order (interleaved-A x interleaved-B is valid iff their K-dists match).

    Comparison is PER ATOM: ``a_free_atoms``/``b_free_atoms`` are the free-dim atom counts (M-atoms for A,
    N-atoms for B) the driver walks. A rectangular wave tile (``a_free_atoms != b_free_atoms``) has
    whole-fragment K-lists of different lengths even though every issued atom pairs the SAME K, so each
    operand is reduced to its per-atom K signature first (defaults of 1 = whole-fragment compare, for a
    caller with no plan/atom-counts in hand). Three-tier:

    - ``ok``      -- atom K-signatures match position-for-position.
    - ``warning`` -- order differs but every lane holds the SAME K set per atom: reconcilable by an
      in-register reorder (named, not performed).
    - ``error``   -- a malformed atom tiling, a lane-count mismatch, or lanes holding different K sets.
    """
    a_sig, a_reason = _atom_k_signature(
        _kdist_from_fwd(as_forward_map(a), k_axis), a_free_atoms, "A"
    )
    if a_reason:
        return Diagnostic("error", a_reason)
    b_sig, b_reason = _atom_k_signature(
        _kdist_from_fwd(as_forward_map(b), k_axis), b_free_atoms, "B"
    )
    if b_reason:
        return Diagnostic("error", b_reason)
    if len(a_sig) != len(b_sig):
        return Diagnostic(
            "error",
            f"A spans {len(a_sig)} lanes but B spans {len(b_sig)} -- operands not MMA-compatible",
        )
    mism = [lane for lane, (ak, bk) in enumerate(zip(a_sig, b_sig)) if ak != bk]
    if not mism:
        return Diagnostic("ok", "A.K == B.K per atom (labels K-aligned; valid MMA)")
    lane = mism[0]
    if all(sorted(a_sig[l]) == sorted(b_sig[l]) for l in range(len(a_sig))):
        return Diagnostic(
            "warning",
            f"A.K != B.K positionally (lane {lane}: A {a_sig[lane]} vs B {b_sig[lane]}); same K set per "
            "atom -> reconcilable by an in-register reorder (transform_fragment one operand first)",
        )
    return Diagnostic(
        "error",
        f"A.K and B.K hold different K sets (lane {lane}: A {sorted(a_sig[lane])} vs "
        f"B {sorted(b_sig[lane])}) -- no in-register reorder reconciles them",
    )


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


def mma_accumulator_flow_consistent(
    c,
    a,
    b,
    *,
    a_canon: WarpDistributionEncoding,
    b_canon: WarpDistributionEncoding,
    c_canon: WarpDistributionEncoding,
) -> Diagnostic:
    """DIAGNOSTIC (observer): does the passed C accumulator carry the labels the machine ACTUALLY produces
    from the passed A/B operands? The store reads C's ``(lane,reg) -> (m,n)`` map to place each result; if
    that map disagrees with the machine's fall-out, the store writes the WRONG coordinates.
    This is the C store-coordinate gate, NOT a sound-MAC condition.

    ``c`` / ``a`` / ``b`` are encodings OR forward maps. Compares ``as_forward_map(c)`` to
    :func:`derive_c_distribution` (A's M and B's N flowed through the fixed canonical machine). Equality is
    EXACT, including register order -- the driver returns C machine-native (it never reorders the
    accumulator), so an AOS/transposed C is a real mis-store, not a free relabel. Do NOT relax to a
    label-SET compare: that reopens the AOS-C mis-slice with nothing behind it. Assumes A/B are already
    per-operand sound (:func:`mma_operand_layout_sound`) -- run that FIRST; ``derive_c_distribution`` is
    undefined on an unsound operand.
    """
    got = as_forward_map(c)
    want = derive_c_distribution(
        a, b, a_canon=a_canon, b_canon=b_canon, c_canon=c_canon
    )
    if len(got) != len(want):
        return Diagnostic(
            "error",
            f"C not consistent: accumulator has {len(got)} (lane,reg) slots but the machine produces "
            f"{len(want)} -- wrong accumulator for this wave tile",
        )
    for slot in sorted(want):
        gm = got.get(slot)
        if gm is None or tuple(gm[:2]) != tuple(want[slot]):
            return Diagnostic(
                "error",
                f"C not consistent at lane/reg {slot}: accumulator labels it "
                f"{None if gm is None else tuple(gm[:2])} but the machine derives {tuple(want[slot])} "
                "-- the store would write the wrong coordinates",
            )
    return Diagnostic(
        "ok", "C accumulator matches the machine-derived (m,n) on every slot"
    )


def mma_operand_layout_sound(
    layout,
    canon: WarpDistributionEncoding,
    *,
    free_axis: int = 0,
    k_axis: int = 1,
    role: str = "operand",
) -> Diagnostic:
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
        return Diagnostic(
            "error",
            f"{role} not sound: layout has {len(sup)} (lane,reg) slots but the machine has {n_slots} "
            "-- dimension mismatch (wrong fragment for this atom)",
        )
    rows: dict[int, list[tuple[int, int]]] = {}
    for lane in range(cm.num_lanes):
        for reg in range(cm.num_vector_items):
            rows.setdefault(cm.matrix_coordinates(lane, reg)[free_axis], []).append(
                (lane, reg)
            )
    for cf in sorted(rows):
        cells = rows[cf]
        frees = {sup[c][free_axis] for c in cells}
        if len(frees) != 1:
            return Diagnostic(
                "error",
                f"{role} not sound: machine output-row {cf} carries {len(frees)} free-labels "
                f"{sorted(frees)} -- M/N not fixed along the contraction (rule 2)",
            )
        sup_k = sorted(sup[c][k_axis] for c in cells)
        can_k = sorted(cm.matrix_coordinates(l, r)[k_axis] for (l, r) in cells)
        if sup_k != can_k:
            return Diagnostic(
                "error",
                f"{role} not sound: machine output-row {cf} K-labels {sup_k} != contraction set "
                f"{can_k} -- malformed/duplicated K (rule 3: well-formed K)",
            )
    return Diagnostic(
        "ok",
        f"{role} sound: fixed free-label + well-formed K on every machine output-row",
    )


def mma_operand_repair_hint(
    layout,
    canon: WarpDistributionEncoding,
    *,
    free_axis: int = 0,
    k_axis: int = 1,
    role: str = "operand",
) -> Diagnostic:
    """yes/no: is this LOGICAL-LABEL layout MMA-compatible, and if not, can a transform MAKE-IT-SO?
    ``ok`` -- compatible (sound, :func:`mma_operand_layout_sound`). Otherwise classify the fix toward a
    known-sound target (``canon``): ``warning`` -- an in-register ``reorder`` makes-it-so (no data
    movement); ``error`` -- needs ``cross_lane`` movement, or no transform reconciles it. Observer only.
    """
    snd = mma_operand_layout_sound(
        layout, canon, free_axis=free_axis, k_axis=k_axis, role=role
    )
    if snd.severity == "ok":
        return Diagnostic("ok", f"{role} MMA-compatible ({snd.message})")
    try:
        plan = _classify_maps(as_forward_map(layout), as_forward_map(canon))
    except ValueError as e:
        return Diagnostic(
            "error",
            f"{role} NOT MMA-compatible; no transform reconciles it -- {snd.message} [{e}]",
        )
    if plan.tier == "reorder":
        return Diagnostic(
            "warning",
            f"{role} NOT MMA-compatible, but an in-register reorder makes-it-so "
            f"(permutation {plan.permutation}) -- {snd.message}",
        )
    return Diagnostic(
        "error",
        f"{role} NOT MMA-compatible; needs cross-lane movement to make-it-so ({plan.reason})",
    )


def mma_pair_compatible(
    a_enc,
    b_enc,
    *,
    a_canon: WarpDistributionEncoding,
    b_canon: WarpDistributionEncoding,
    a_free_atoms: int = 1,
    b_free_atoms: int = 1,
    k_axis: int = 1,
) -> Diagnostic:
    """Full A x B check: BOTH operands sound (:func:`mma_operand_layout_sound`) AND their K-dists match
    PER ATOM (the relationship, :func:`mma_pair_k_aligned`). Observer only. ``ok`` iff the pair is a
    valid, meaningful MMA; else the first failing operand's soundness error, or the K-match diagnostic.
    The full sound MAC = per-operand soundness (conditions 1-2) + pairwise K-match (3); correctness SOT:
    ``docs/mma_is_machinery.md``.

    Pass ``a_free_atoms``/``b_free_atoms`` (the free-dim atom counts m_sub/n_sub) so the K-match is per
    atom -- **REQUIRED for a rectangular wave tile** (``m_sub != n_sub``), whose whole-wave K-lists differ
    in length. Defaults of 1 (whole-fragment) are only correct for a truly square/single-atom pair with no
    shape in hand; a caller that HAS the shape (the recorder op, a ``TileMma``) must pass the counts.
    """
    for enc, canon, role in ((a_enc, a_canon, "A"), (b_enc, b_canon, "B")):
        d = mma_operand_layout_sound(enc, canon, k_axis=k_axis, role=role)
        if d.severity != "ok":
            return d
    return mma_pair_k_aligned(
        a_enc,
        b_enc,
        a_free_atoms=a_free_atoms,
        b_free_atoms=b_free_atoms,
        k_axis=k_axis,
    )
