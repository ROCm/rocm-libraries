# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Lower LoopIR (layer 1) into GIR (layer 2)."""

from __future__ import annotations

from dataclasses import replace
from ..LoopModel import Space, Loop, Branch, Bind, Peel, Inst, Load, Cond
from ..LoopModel.emit import emit_mainloop
from ..LoopModel.schedule import peel_depths
from ..LoopModel.schedule import build_S
from ..LoopModel.traversal import lds_buffers
from ..LoopModel import traversal as _geometry
from .gir.passes import pipeline as default_pipeline, run_pipeline

from .gir.nodes import (Tile, Gen, Ref, Move, Mma, Mark, Pred, Bound, Trips, LoopBack,
                        covered_coords,
                        Goto, CondGoto,
                        Block, GenPhi, GenXfer, Program)
from ..LoopModel.traversal import free_names, presence, summation_names


# ===========================================================================
# small Expr helpers
def _gir_bound(e) -> Bound:
    """One side of a control-flow test, LoopIR `Expr` (or plain int) -> GIR `Bound`, by field."""
    if not hasattr(e, "var"):
        return Bound(const=int(e))
    if e.terms or e.carry or e.mod:
        raise NotImplementedError(
            f"control-flow predicate operand {e!r} uses mixed-radix terms / a modulus / a "
            f"chunk-carry; GIR's Bound holds `var + const` only.  A loop TEST with that shape is "
            f"not something the scaffold can branch on -- if one is ever needed, widen Bound "
            f"deliberately rather than rendering the predicate back to a string.")
    return Bound(var=e.var, const=e.add)


def _short_steps(els, M):
    """Split the short-loop arm into per-step ``(guard, nodes)`` pairs."""
    if not els:
        return []
    if len(els) != M:
        raise RuntimeError(
            f"short-loop arm has {len(els)} steps but the peel depth M is {M} -- the arm is the "
            f"`T < M` degenerate path, so it has exactly one step per peeled chunk")
    steps = []
    for t, step in enumerate(els):
        node, guard = step, None
        if isinstance(node, Cond):
            if node.kind != "short_step_validity" or node.els or len(node.then) != 1:
                raise RuntimeError(
                    f"short-loop step {t} is wrapped in an unexpected {node.kind!r} Cond -- the "
                    f"short path's only guard is per-step peel validity")
            guard, node = _gir_pred(node.pred), node.then[0]
        if not isinstance(node, Bind):
            raise RuntimeError(
                f"short-loop step {t} is a {type(node).__name__}, expected a Bind pinning the "
                f"chunk to a concrete index")
        steps.append((guard, [node]))
    return steps


def _trips_of(loop_trip, per_trip: int = 1) -> Trips:
    """Convert a LoopIR range bound into a GIR trip count."""
    if loop_trip.op != "<":
        raise NotImplementedError(
            f"steady loop range uses {loop_trip.op!r}; the trip count is read from a `<` upper "
            f"bound only.")
    b = _gir_bound(loop_trip.rhs)
    return Trips(var=b.var, sub=-b.const, div=per_trip)


def _gir_pred(p) -> Pred:
    """LoopIR structured `Pred` -> GIR structured `Pred`, FIELD BY FIELD (no string round-trip)."""
    lhs = p.lhs
    lb = _gir_bound(lhs)
    if not lb.var or lb.const:
        raise NotImplementedError(
            f"control-flow predicate lhs {lhs!r} is not a bare counter symbol -- GIR's terminator "
            f"tests `<counter> op <bound>`.")
    return Pred(lb.var, p.op, _gir_bound(p.rhs))


def _expr_raw(e, env) -> int:
    """Evaluate an expression without applying its modulus."""
    if e is None:
        return 0
    return e if isinstance(e, int) else replace(e, mod=0).eval(env)


def _deps_of(inst):
    """Return the LoopIR awaits as GIR dependency records."""
    return tuple((a.dep, a.counter, a.kind, a.scope) for a in getattr(inst, "awaits", ()))


def _op_of_token(theta, token: str):
    """The theta operand a Load token names, or None if theta has no such operand.

    A token IS an operand name -- a straight lookup, no parsing.
    region suffix (`'A0'` -> `'A'`) because `emit._copy_inst` baked the storage region into the
    name; the region is now carried where a position belongs, on the Load's `coord`/`part`, so
    there is nothing to strip."""
    return next((o for o in theta.operands if o.name == token), None)


# ===========================================================================
# stage flattening -- one concrete (kind, Load/Mma, env) stream per stage body
def _flatten(nodes, env, out, rel=None):
    """Mirror render_unrolled's traversal: unroll int-extent Loops; a symbolic-trip Loop (the
    steady summation-chunk loop) is walked once at the current env; bind Branch residues (pinned
    or fanned); descend Peel/Cond bodies; yield each Inst with its concrete env.  Appends
    `(inst, dict(env), rel)` to `out` in program order.
    """
    for nd in nodes:
        if isinstance(nd, Loop):
            if nd.outer:                                      # OUTER runtime loop: show one rep
                for body in nd.bodies:
                    _flatten(body, {**env, nd.axis: env.get(nd.axis, 0)}, out, rel)
            else:
                for lo, hi, body in nd.ranged_bodies():
                    end = hi if hi is not None else lo + 1
                    for v in range(lo, end):
                        _flatten(body, {**env, nd.axis: v}, out, rel)
        elif isinstance(nd, Branch):
            if nd.axis in env:
                _flatten(nd.arms.get(env[nd.axis] % nd.modulus, []), env, out, rel)
            else:
                for r, arm in nd.arms.items():                # pinned-residue: key IS the value
                    _flatten(arm, {**env, nd.axis: r}, out, rel)
        elif isinstance(nd, Peel):
            _flatten(nd.body, env, out, rel)
        elif isinstance(nd, Bind):
            # a peel-step chunk binding : bind the induction to its value. A CONCRETE
            # value (prologue `j-M`, short `t`) binds a real int.  A SYMBOLIC drain value `T-M+t`
            benv = nd.bound_env(env)
            brel = rel
            if nd.axis not in benv:                     # symbolic (drain): T-relative chunk offset
                benv, brel = dict(env), nd.value.add    # value = Expr(var="T", add=t-M) -> add = t-M
            _flatten(nd.body, benv, out, brel)
        elif isinstance(nd, Cond):
            # honor a RESOLVED guard in the current env: a drain NLL guard `Pred(k < n_s-dr)` (the
            # summation Loop has bound k) really DROPS the out-of-bounds read-ahead; an UNRESOLVED
            v = nd.pred.eval(env)
            if v is None or v:
                _flatten(nd.then, env, out, rel)
            else:
                _flatten(nd.els, env, out, rel)
        elif isinstance(nd, Inst):
            out.append((nd, dict(env), rel))


def _drain_steps(peel_body):
    """Split a drain Peel body into its per-step node lists. `build_ir` emits one
 `Bind(iter = T-M+t)` per drain step (the induction pinned to the model's `i = T-M+t`), so each
 top-level Bind is one step."""
    steps = []
    for nd in peel_body:
        if isinstance(nd, Bind):
            steps.append([nd])
        else:                                                 # defensive: attach stray nodes
            if steps:
                steps[-1].append(nd)
            else:
                steps.append([nd])
    return steps


# ===========================================================================
# Inst -> GIR node conversion
def _region_axes(theta):
    """Every storage-region axis in theta, whichever dimension it cuts."""
    return {axis for operand in theta.operands
            if operand.trajectory.shared is not None
            for axis in operand.trajectory.shared.regions.axes}


def _absent_at_region(coord, op, theta):
    """Does this fused member sit OUT of the unit's walk at this instance's region?

 A member owning fewer regions than the unit moves its whole tile once and rides region 0; on
 every later step it is absent, and the descriptor is nulled for the waves that carry it.
 Keyed on `Operand.split` -- the region COUNT -- never on `region_axes`, which an unsplit MX
 scale also owns (`region_axes=('M_split',)`, `split=1`).  Which DIMENSION the walk cuts is not
 the question: an unsplit member moves once whether the peer is split on a free axis or on K.
    """
    if op is None or max(1, getattr(op, "split", 1)) > 1:
        return False
    spanned = _region_axes(theta)
    return any(ax in spanned and v not in (None, 0) for ax, v in coord)


def _member_region_coord(coord, op, theta):
    """Re-key a FUSED instance's region coordinate onto THIS member's OWN region axis."""
    own = (op.trajectory.shared.regions.axes
           if op is not None and op.trajectory.shared is not None else ())
    all_region = _region_axes(theta)
    foreign = [(ax, v) for ax, v in coord if ax in all_region and ax not in own]
    if not foreign:
        return coord                              # already this member's own axis (or unsplit)
    if len(own) > 1:
        # The index alone does not say which of several axes it pairs on.  Refuse rather than
        # guess: picking one is how a region coordinate silently names the wrong half.
        raise NotImplementedError(
            f"fused member {getattr(op, 'name', op)!r} is split on {list(own)} but the instance "
            f"coord names {[a for a, _ in foreign]}: by-index pairing across several region axes "
            f"is undefined.")
    idx = foreign[0][1]
    out = []
    for ax, v in coord:
        if ax in all_region and ax not in own:
            out.extend((m, idx) for m in own)     # same index, this member's axis
        else:
            out.append((ax, v))
    return tuple(out)


def _slot_of(theta, op, coord, pl):
    """The `pl.slots` entry naming THIS member's generation, as `(label, Expr)`.

    A slot is keyed by the member's own group, so every caller asking "which slot applies here"
    answers it from `coord`.  Taking `slots[0]` instead hands every member the first one's
    generation, and a fused copy then writes each region at the wrong phase."""
    if not (pl and getattr(pl, "slots", None)):
        return None, None
    groups = op.fragment.groups() if op is not None else ("",)
    if op is not None and len(groups) > 1 and len(pl.slots) > 1:
        want = groups[_geometry.group_index(theta, op, coord)]
        for label, gexpr in pl.slots:
            if label == want:
                return label, gexpr
    return pl.slots[0]


def _reg_residence(theta, op, coord, pl, env):
    """`(group index, concrete slot, rotation width W)` for `op`'s register fragment AT `coord`."""
    label, gexpr = _slot_of(theta, op, coord, pl)
    if gexpr is None:
        return None, None, None
    groups = op.fragment.groups() if op is not None else ("",)
    grp_idx = list(groups).index(label) if label in groups else 0
    return grp_idx, gexpr.eval(env), (getattr(gexpr, "mod", 0) or 1)


def _concrete_coord(op_coord, env):
    """A LoopIR op coord ((axis, None|val|Expr), ...) -> hashable concrete ((axis, val), ...)."""
    out = []
    for ax, v in op_coord:
        if hasattr(v, "eval"):
            out.append((ax, v.eval(env)))
        elif ax in env:
            out.append((ax, env[ax]))
        elif v is not None:
            out.append((ax, v))
        else:
            out.append((ax, None))
    return tuple(out)


def _absorbed(theta, op) -> tuple:
    """`((axis, extent),...)` the operand's read instruction SPANS -- the absorbed axes."""
    if op is None:
        return ()
    hop = op.trajectory.fragment_fill
    if hop is None or hop.coverage is None:
        return ()
    spanned = _geometry.coverage_axes(theta, op, hop.coverage)      # full: q == N
    fold = _geometry.transfer_coverage(theta, op, hop)                     # partial: 1 < q < N
    return tuple((m.name, m.extent if m.name in spanned else fold[m.name])
                 for m in theta.inner_axes()
                 if m.name in spanned or m.name in fold)


def _read_is_leader(theta, op, coord) -> bool:
    """Is this read the LEADER of its carrier group?

    The fold is resolved HERE so GIR is born faithful: a follower's payload rides the leader's
    instruction, so a Move for it would be a movement nothing issues.  The rule is
    `coverage.plan_coverage`'s -- the smallest tile sharing a carrier emits -- over the flat free
    index `emit_plan._project` builds, so the two layers cannot disagree.
    """
    hop = op.trajectory.fragment_fill
    if hop is None or hop.coverage is None:
        return True
    inner = theta.inner_axes()
    free = set(_geometry.free_names(theta, op))
    at = coord if isinstance(coord, dict) else dict(coord)
    tile, stride, extent = 0, 1, 1
    for m in reversed([m for m in inner if m.name in free]):
        tile += int(at.get(m.name, 0) or 0) * stride
        stride *= max(1, m.extent)
        extent *= max(1, m.extent)
    carrier = hop.coverage.carrier_of(tile)
    lead = next((u for u in range(max(1, extent))
                 if hop.coverage.carrier_of(u) == carrier), tile)
    return tile == lead


def _storage_region(theta, op, coord):
    """Dense storage-region index selected by this operand coordinate."""
    if op is None or max(1, int(getattr(op, "split", 1) or 1)) == 1:
        return 0
    at = dict(coord)
    extents = {axis.name: axis.extent for axis in theta.inner_axes()}
    shared = op.trajectory.shared
    spans = shared.regions.span if shared is not None else {}
    region = 0
    for axis in (shared.regions.axes if shared is not None else ()):
        span = max(1, int(spans.get(axis, 1)))
        extent = max(1, int(extents.get(axis, 1)) // span)
        value = at.get(axis)
        if value is None:
            raise RuntimeError(
                f"{op.name}: shared access has no concrete {axis} coordinate, so it cannot "
                "select one of its independent LDS region generations")
        region = region * extent + int(value) // span
    if region >= max(1, int(op.split)):
        raise RuntimeError(
            f"{op.name}: coordinate {coord} selects LDS region {region}, "
            f"outside its {op.split} regions")
    return region


def _gen_of(gens, theta, op, coord):
    return gens.get((op.name, _storage_region(theta, op, coord))) if op is not None else None


def _convert_load(theta, inst, env, rel, gens):
    """A LoopIR Load -> a GIR Move. dst=SHARED => a copy (global->shared); dst=REGISTER => a read
 (shared/global->register).

 `rel` is the steady-relative summation-chunk offset (see `_flatten`): not-None => the ref
 joins the loop-carried generation timeline as `gen` + `gdelta` evaluated at that offset;
 None => the chunk is concrete, so the generation is absolute (`abs_gen`, fact)."""
    ld = inst.op
    pl = inst.placement
    chunk = theta.summation_chunk_name("iter")     # the level the generation timeline runs on
    if ld.dst == Space.SHARED:
        # --- COPY: one Ref per fused token; dst carries the LDS generation ---
        srcs, dsts = [], []
        base_coord = _concrete_coord(ld.coord, env)
        present = []
        for tok in ld.tokens:
            op = _op_of_token(theta, tok)
            if _absent_at_region(base_coord, op, theta):
                continue                              # not in THIS movement; see `unit` below
            present.append(tok)
            coord = _member_region_coord(base_coord, op, theta)
            # Per member: a fused copy moves several, and each lands in its own group's slot.
            slot_expr = _slot_of(theta, op, coord, pl)[1]
            src_tile = Tile(op.name if op else tok, "global", coord,
                            (("bytes", ld.size_bytes),))
            dst_tile = Tile(op.name if op else tok, "shared", coord,
                            (("bytes", ld.size_bytes),))
            srcs.append(Ref(src_tile, size_bytes=ld.size_bytes))
            if rel is not None:
                g = _gen_of(gens, theta, op, coord)
                dsts.append(Ref(dst_tile, gen=g,
                                gdelta=_expr_raw(slot_expr, {**env, chunk: rel}),
                                size_bytes=ld.size_bytes))
            else:
                dsts.append(Ref(dst_tile, abs_gen=(slot_expr.eval(env) if slot_expr else 0),
                                size_bytes=ld.size_bytes))
        # THE UNIT IS WHO ACTUALLY MOVES.  A later region carries only the split operand, so it
        # is a SOLO movement -- tagging it with the whole fuse group makes the absent member look
        # like a group participant that has to be nulled out of a load it is not in.
        return Move(srcs=tuple(srcs), dsts=tuple(dsts), deps=_deps_of(inst),
                    unit=tuple(present))
    else:
        # --- READ: shared(or global DTV)->register ---
        op = _op_of_token(theta, ld.tokens[0])
        coord = _concrete_coord(ld.coord, env)
        src_space = ld.src                                    # 'shared' (2-hop) or 'global' (DTV)
        src_tile = Tile(op.name if op else ld.tokens[0], src_space, coord,
                        (("regs", ld.size_regs),))
        dst_tile = Tile(op.name if op else ld.tokens[0], "register", coord,
                        (("regs", ld.size_regs),))
        # register residence: (group index, concrete slot, rotation width W) for THIS COORDINATE's
        # group. W is the slot Expr's modulus (a LoopIR fact GIR consumes, B3): mod>1 -> W
        grp_idx, slot_val, reg_ring = _reg_residence(theta, op, coord, pl, env)
        # source shared generation
        src_ref_kw = {}
        if src_space == Space.SHARED and pl is not None and pl.src_slot is not None:
            if rel is not None:
                # steady (rel=0) / drain (rel=t-M) alike: continue the one residue timeline, so the
                # read-pointer swap chain resolves across the peel boundary instead of re-pinning.
                src_ref_kw = dict(gen=_gen_of(gens, theta, op, coord),
                                  gdelta=_expr_raw(pl.src_slot, {**env, chunk: rel}))
            else:
                src_ref_kw = dict(abs_gen=pl.src_slot.eval(env))
        # The absorbed axes are the SOURCE's footprint too: one instruction reads every location
        # it spans, and the hazard set is built from the shared-space Ref.
        covers = _absorbed(theta, op)
        if covers and not _read_is_leader(theta, op, coord):
            return None
        src = Ref(src_tile, size_regs=ld.size_regs, covers=covers, **src_ref_kw)
        # ONE instruction, ONE completion, ONE def -- but it FILLS `Pi factor` locations.
        unit_index = _geometry.unit_tile_index(theta, op, coord)
        covered = Ref(dst_tile, unit_index=unit_index, covers=covers, size_regs=ld.size_regs)
        unit_indexes = tuple(_geometry.unit_tile_index(theta, op, dict(c))
                             for c in covered_coords(covered))
        dst = Ref(dst_tile, group=grp_idx, slot=slot_val, reg_ring=reg_ring,
                  unit_index=unit_index, unit_indexes=unit_indexes,
                  covers=covers, size_regs=ld.size_regs)
        logical_axes = {axis.name for axis in _geometry.presence_axes(theta, op)}
        labels = op.fragment.groups()
        label = labels[grp_idx] if grp_idx is not None and grp_idx < len(labels) else labels[0]
        repeats = False
        issue = (tuple((axis.name, env.get(axis.name, 0)) for axis in theta.inner_axes()
                       if axis.name not in logical_axes and axis.extent > 1)
                 if repeats else ())
        return Move(srcs=(src,), dsts=(dst,), deps=_deps_of(inst),
                    advance=int(getattr(ld, "advance", 0) or 0), issue=issue)


def _convert_mma(theta, inst, env, reads):
    """A LoopIR Mma -> a GIR Mma. srcs = one register Ref per read operand (+ scales);
 dst = the accumulator. Operand identity is on each Ref.tile.operand.
    """
    m = inst.op
    coord = _concrete_coord(m.coord, env)
    pls = inst.placement or {}                       # {operand: Placement} from the decoder
    srcs = []
    for op in reads:
        pl = pls.get(op.name)
        grp_idx, slot_val, reg_ring = _reg_residence(theta, op, coord, pl, env)
        size_regs = _geometry.frag_regs(theta, op)
        # THE SOURCE NAMES THE OPERAND'S OWN COORDINATE, NOT THE WMMA'S.
        #
        pres = set(presence(theta, op))
        own = tuple((ax, v) for ax, v in coord if ax in pres)
        t = Tile(op.name, "register", own, (("regs", size_regs),))
        labels = op.fragment.groups()
        label = labels[grp_idx] if grp_idx is not None and grp_idx < len(labels) else labels[0]
        repeats = False
        issue_axes = (tuple(axis.name for axis in theta.inner_axes()
                            if axis.name not in pres and axis.extent > 1)
                      if repeats else ())
        srcs.append(Ref(t, group=grp_idx, slot=slot_val, reg_ring=reg_ring,
                        unit_index=_geometry.unit_tile_index(theta, op, own),
                        issue_axes=issue_axes,
                        size_regs=size_regs))
    # scales (MX): each scale operand appears in `reads` already if it ends in a register;
    # m.scales are display strings, so operand identity comes from the reads list above.
    acc = Ref(Tile("acc", "register", coord, ()))
    return Mma(srcs=tuple(srcs), dsts=(acc,), coord=coord, block=m.block, deps=_deps_of(inst))


# ===========================================================================
# the lowering
def _loop_carried_gens(theta):
    """One loop-carried Gen per independent ``(operand, LDS region)`` storage ring."""
    gens = {}
    for op in theta.operands:
        if op.trajectory.shared is not None:
            for region in range(max(1, int(getattr(op, "split", 1) or 1))):
                gens[(op.name, region)] = Gen(id=len(gens),
                                              ring=max(1, lds_buffers(theta, op)))
    return gens


def _scan_regions(ir, M):
    """Split the LoopIR root into its four regions: prologue, steady, drain, short arm.

    The emitted skeleton wraps the whole pipeline in one peel-validity `Cond`, with the three
    pipelined regions in its `then` and the `T < M` arm in its `els`. At M == 0 there is no
    peel, so the regions sit at the top level and there is no short arm.
    """
    found = {"prologue": None, "steady": None, "drain": None, "guard": None, "short": ()}

    def scan(nodes):
        for nd in nodes:
            if isinstance(nd, Peel) and nd.kind == "prologue":
                found["prologue"] = nd.body
            elif isinstance(nd, Peel) and nd.kind == "drain":
                found["drain"] = nd
            elif isinstance(nd, Loop) and nd.outer:   # STRUCTURAL flag, never the level name
                found["steady"] = nd
            elif isinstance(nd, Cond):
                if nd.kind == "peel_validity":
                    found["guard"] = nd.pred          # the structured guard, carried not re-derived
                    found["short"] = tuple(nd.els)
                scan(nd.then)

    scan(ir)
    short_steps = _short_steps(found["short"], M)
    if short_steps and (found["prologue"] is None or found["steady"] is None):
        raise RuntimeError(
            f"the LoopIR carries a {len(short_steps)}-step `T < M` short arm, but this kernel has "
            f"{'no prologue block' if found['prologue'] is None else 'no steady loop'} -- there is "
            f"no peel-validity branch for it to hang off, so it could only be dropped unchecked "
            f".  Widen the lowering deliberately if this shape is real.")
    return found["prologue"], found["steady"], found["drain"], found["guard"], short_steps


def _program_meta(theta, mainloop, M, red_names, free_axes, mma_inputs, mma_scales,
                  short_steps):
    """Everything the GIR passes need to know about the theta this program came from.

    A read-only bag: the lowering fills it once and each analysis takes the few keys it
    needs. Separate from `lower_to_gir` because it translates theta; it does not build
    the CFG.
    """
    meta = {"buffer_depths": mainloop.depths, "peel_depth": M}
    meta["register_layout"] = _geometry.register_layout(theta, mainloop.depths)
    meta.update(_axis_facts(theta, red_names, free_axes, mma_inputs, mma_scales))
    meta.update(_region_facts(theta))
    meta.update(_movement_facts(theta))
    meta.update(_coverage_facts(theta))
    meta.update(_short_arm_facts(M, short_steps))
    return meta


def _axis_facts(theta, red_names, free_axes, mma_inputs, mma_scales):
    """Which axes exist, in what order, at what extent, and which two operands the wmma grids."""
    inner = theta.inner_axes()
    return {
        "summation_axes": red_names,
        "free_axes": free_axes,
        "mma_inputs": mma_inputs,
        "mma_scales": mma_scales,
        # loop order of the inner axes plus their extents, so emit_plan can mixed-radix-combine
        # a multi-axis role (M_split+M_inner) deterministically.
        "inner_axis_order": [axis.name for axis in inner],
        "axis_extents": {axis.name: axis.extent for axis in inner},
    }


def _region_facts(theta):
    """How each operand's tile splits into storage regions, and who completes independently."""
    return {
        # Address generation may be wave-relative, but the factorized coordinate still selects
        # one independent LDS region generation/token.
        "wave_relative_regions": {
            op.name for op in theta.operands
            if op.trajectory.shared is not None
            and op.trajectory.shared.regions.wave_relative},
        # False = every region of an operand shares one completion class, so a read awaits them
        # all; True = one class per region, so a read of region j leaves the others in flight.
        "per_region_completion": bool(theta.per_region_completion),
        "region_axes": {
            op.name: tuple(op.trajectory.shared.regions.axes)
            if op.trajectory.shared is not None else ()
            for op in theta.operands},
        # how many regions ITS OWN tile occupies -- distinct from `unit_regions`, which counts
        # what the whole movement emits.
        "operand_regions": {op.name: max(1, op.split) for op in theta.operands},
        # how many values of a SHARED region axis one of this operand's regions spans: the K walk
        # is common, so an operand that splits K coarser holds a region across several steps.
        "region_span": {
            op.name: dict(op.trajectory.shared.regions.span)
            if op.trajectory.shared is not None else {}
            for op in theta.operands},
    }


def _movement_facts(theta):
    """Which movements are fused, how many regions each emits, and which span several waves."""
    return {
        # a fused group is ONE completion event, so a read of any member awaits the whole group
        "fused_copy_groups": [tuple(g) for g in theta.fused_copy_groups],
        "unit_regions": {key: nreg for key, _members, nreg in theta.movement_units()},
        # PER MEMBER, so a shorter member can be told apart from the unit it rides in.  A member
        # whose count is below the unit's moves only on region 0 and is nulled for the rest.
        "unit_member_regions": {key: {m.name: max(1, m.split) for m in members}
                                for key, members, _nreg in theta.movement_units()},
        "agent_distributed": {op.name: theta.agent_distributed(op.name) for op in theta.operands},
    }


def _register_reads(theta):
    """Every operand's shared->register read hop, as (operand, hop) pairs."""
    return [(op, op.trajectory.fragment_fill) for op in theta.operands
            if op.trajectory.fragment_fill is not None]


def _coverage_facts(theta):
    """What one read instruction covers: its map, the axes it folds in part, the axes it spans."""
    reads = _register_reads(theta)
    inner = theta.inner_axes()
    return {
        "read_coverage": {op.name: h.coverage for op, h in reads if h.coverage is not None},
        # `{axis: factor}` for each axis the coverage folds only IN PART (1 < q < N), so the axis
        # stays in the loop nest at a reduced trip rather than disappearing from it.
        "read_fold": {op.name: dict(_geometry.transfer_coverage(theta, op, h))
                      for op, h in reads if _geometry.transfer_coverage(theta, op, h)},
        "coverage_axes": {op.name: tuple(
                              (m.name, m.extent) for m in inner
                              if m.name in _geometry.coverage_axes(theta, op, h.coverage))
                          for op, h in reads if h.coverage is not None},
    }


def _short_arm_facts(M, short_steps):
    """The `T < M` arm as lowered: real blocks now, which FoldShortPathPass may later fold away."""
    if not short_steps:
        return {"short_loop": None}
    return {"short_loop": {
        "steps": M,
        "guards": tuple(g for g, _ in short_steps),
        "verdict": "unfolded",
        "model_only": {
            "blocks": tuple("short%d" % i for i in range(len(short_steps))),
            "marks_dropped": 0,
            "reason": "the T < M arm is scaffold-owned (TensileLite's toPGR1 path); GIR keeps "
                      "it for the analyses but no backend emits it",
        }}}


def _add_prologue_blocks(prog, theta, gens, reads, pro_body, guard_pred, short_entry, steady_loop):
    """The prologue block: the peeled chunks that fill the pipeline before the steady loop."""
    if pro_body is not None:
        flat = []
        _flatten(pro_body, {}, flat)
        body = [Mark("phase_boundary", {"phase": "prologue"})]
        for inst, env, rel in flat:
            if isinstance(inst.op, Load):
                _mv = _convert_load(theta, inst, env, rel, gens)
                if _mv is not None:
                    body.append(_mv)
            else:
                body.append(_convert_mma(theta, inst, env, reads))
        steady_exists = steady_loop is not None
        if steady_exists and guard_pred is None:
            raise RuntimeError(
                "lowering: a prologue block exists but the LoopIR carries no peel-validity Cond -- "
                "the guard predicate must be CARRIED from the LoopIR, not re-derived here")
        term = (CondGoto(_gir_pred(guard_pred), "steady", short_entry)
                if steady_exists else Goto(short_entry))
        succs = (("steady", short_entry) if steady_exists else (short_entry,))
        # frame (Block.gen_rel/chunk_base): the prologue's Refs carry ABSOLUTE generations, and its
        # read-ahead fills the chunk the first steady trip consumes -- chunk-timeline slot 0.
        prog.add_block(Block(phase="prologue", preds=(), succs=succs, body=body, term=term,
                             gen_rel=None, chunk_base=0))


def _add_steady_blocks(prog, theta, gens, reads, steady_loop, pro_body, first_drain, loop_copies):
    """The steady loop's blocks, one per unrolled copy, and the phis on its header."""
    if steady_loop is not None:
        n_copies = max(1, int(loop_copies))
        labels = ["steady"] + [f"steady{i}" for i in range(1, n_copies)]
        for i, lab in enumerate(labels):
            flat = []
            _flatten([steady_loop], {}, flat, rel=i)      # copy i = chunk offset i on ONE timeline
            body = [Mark("phase_boundary", {"phase": lab})]
            for inst, env, rel in flat:
                if isinstance(inst.op, Load):
                    _mv = _convert_load(theta, inst, env, rel, gens)
                    if _mv is not None:
                        body.append(_mv)
                else:
                    body.append(_convert_mma(theta, inst, env, reads))
            last = (i == n_copies - 1)
            # phis on the HEADER only (the join of entry- and back-edge); xfers on the LAST copy
            # only (the back-edge transfer), advancing by the number of chunks a trip now consumes.
            phis = [GenPhi(g, entry_val=0) for g in gens.values()] if i == 0 else []
            xfers = [GenXfer(g, adv=n_copies, ring=g.ring) for g in gens.values()] if last else []
            if last:
                # the loop states its TRIP COUNT, not a comparison: the outer `Loop.trip`
                # range's bound IS the count.  `div` is the chunks a trip consumes, so a multi-block
                term = LoopBack(_trips_of(steady_loop.trip, per_trip=n_copies),
                                "steady", first_drain)
                succs = ("steady", first_drain)
            else:
                term, succs = Goto(labels[i + 1]), (labels[i + 1],)
            if i == 0:
                preds = (("prologue",) if pro_body is not None else ()) + (labels[-1],)
            else:
                preds = (labels[i - 1],)
            # frame: copy i's Refs were flattened at rel=i, and it occupies chunk-timeline slot i
            # of the trip -- rel == base here, so a steady requirement is just its gdelta.
            prog.add_block(Block(phase=lab, loop=(i == 0),
                                 preds=preds, succs=succs,
                                 phis=phis, xfers=xfers, body=body, term=term,
                                 gen_rel=i, chunk_base=i))


def _add_drain_blocks(prog, theta, gens, reads, drain_peel, M, drain_labels, first_drain, short_entry,
                      steady_loop, pro_body, loop_copies):
    """The drain blocks: one per chunk still in flight when the steady loop stops."""
    if drain_peel is not None and M > 0:
        steps = _drain_steps(drain_peel.body)
        for i, step_nodes in enumerate(steps):
            flat = []
            _flatten(step_nodes, {}, flat)
            body = [Mark("phase_boundary", {"phase": f"drain{i}"})]
            for inst, env, rel in flat:
                if isinstance(inst.op, Load):
                    _mv = _convert_load(theta, inst, env, rel, gens)
                    if _mv is not None:
                        body.append(_mv)
                else:
                    body.append(_convert_mma(theta, inst, env, reads))
            nxt = drain_labels[i + 1] if i + 1 < len(drain_labels) else "end"
            last_steady = (f"steady{loop_copies-1}" if (steady_loop is not None
                                                        and loop_copies > 1) else "steady")
            pro_pred = (("prologue",) if (i == 0 and pro_body is not None
                                          and short_entry == first_drain) else ())
            preds = (pro_pred + (last_steady,) if i == 0 else (f"drain{i-1}",))
            # frame: `_flatten` gave drain step i the offset `rel = i - M` (its Bind is
            # `iter = T - M + i`, so its gdeltas are relative to T).  Its position on the chunk
            n_cp = max(1, int(loop_copies)) if steady_loop is not None else 0
            prog.add_block(Block(phase=f"drain{i}", preds=preds, succs=(nxt,),
                                 body=body, term=Goto(nxt),
                                 gen_rel=i - M, chunk_base=n_cp + i))


def _add_short_blocks(prog, theta, gens, reads, short_steps, short_labels):
    """The `T < M` arm's blocks: the degenerate path where the pipeline never fills."""
    for i, (_guard, nodes) in enumerate(short_steps):
        flat = []
        _flatten(nodes, {}, flat)                       # concrete chunk -> rel=None -> abs_gen
        body = [Mark("phase_boundary", {"phase": short_labels[i]})]
        for inst, env, rel in flat:
            if isinstance(inst.op, Load):
                _mv = _convert_load(theta, inst, env, rel, gens)
                if _mv is not None:
                    body.append(_mv)
            else:
                body.append(_convert_mma(theta, inst, env, reads))
        nxt_guard = short_steps[i + 1][0] if i + 1 < len(short_steps) else None
        if nxt_guard is not None:
            term = CondGoto(nxt_guard, short_labels[i + 1], "end")
            succs = (short_labels[i + 1], "end")
        else:
            if i + 1 < len(short_steps):
                raise RuntimeError(
                    f"short-loop step {i+1} carries no validity guard -- every step past the first "
                    f"runs only when `T > t`, so an unguarded one would read a chunk that does "
                    f"not exist")
            term, succs = Goto("end"), ("end",)
        preds = ((short_labels[i - 1],) if i else ("prologue",))
        # frame: the short arm's chunks are concrete (0..M-1) and there is no steady trip, so its
        # Refs carry absolute generations (gen_rel=None) and it sits at the head of the chunk
        prog.add_block(Block(phase=short_labels[i], preds=preds, succs=succs, body=body, term=term,
                             gen_rel=None, chunk_base=i, model_only=True))

def lower_to_gir(theta, mainloop=None, loop_copies=1) -> Program:
    """Build the GIR `Program` from a theta point.  `mainloop` = a prior `emit_mainloop(theta)`
    result (built once by KernelWriter, R-ONCE); if omitted it is computed here.
    """
    if mainloop is None:
        mainloop = emit_mainloop(theta)
    ir = mainloop.ir
    _peel = peel_depths(theta, build_S(theta)[0])
    off, dr, M = _peel.copy_off(), _peel.requested_steps, _peel.chunk_depth

    # operands that end in a register fragment = the "reads" (their Mma sources).  The output/
    # accumulator operand has no mainloop path (hops=[]) -> not a read.
    reads = [op for op in theta.operands if op.trajectory.fragment_fill is not None]

    gens = _loop_carried_gens(theta)

    pro_body, steady_loop, drain_peel, guard_pred, short_steps = _scan_regions(ir, M)

    # PRESENCE-DERIVED addressing : L3 reads a leaf's free-tile /
    # summation index by PROJECTING the coord onto per-operand free axes / the summation axes.
    red_names = sorted(summation_names(theta))
    free_axes = {op.name: free_names(theta, op) for op in theta.operands}
    _reg_inputs = [op.name for op in reads if op.is_input]
    mma_inputs = _reg_inputs[:2]
    mma_scales = _reg_inputs[2:]

    # the entry names a block that ACTUALLY EXISTS.  A one-hop path (direct-to-register,
    #) stages nothing through shared, so there is no prefetch to peel and NO prologue block;
    prog = Program(entry=("prologue" if pro_body is not None else "steady"),
                   meta=_program_meta(theta, mainloop, M, red_names, free_axes,
                                      mma_inputs, mma_scales, short_steps))
    prog.meta["generation_regions"] = {
        gen.id: {"operand": operand, "region": region}
        for (operand, region), gen in gens.items()
    }

    drain_labels = [f"drain{i}" for i in range(M)] if (drain_peel and M > 0) else []
    first_drain = drain_labels[0] if drain_labels else "end"
    short_labels = [f"short{i}" for i in range(len(short_steps))]
    short_entry = short_labels[0] if short_labels else first_drain

    # --- prologue ------------------------------------------------------------------------
    _add_prologue_blocks(prog, theta, gens, reads, pro_body, guard_pred, short_entry, steady_loop)

    # --- steady --------------------------------------------------------------------------
    # ONE source loop, `n_copies` FALLTHROUGH-CHAINED blocks sharing ONE back-edge.
    _add_steady_blocks(prog, theta, gens, reads, steady_loop, pro_body, first_drain, loop_copies)

    # --- drain{0..M-1} -------------------------------------------------------------------
    _add_drain_blocks(prog, theta, gens, reads, drain_peel, M, drain_labels, first_drain, short_entry, steady_loop, pro_body, loop_copies)

    _add_short_blocks(prog, theta, gens, reads, short_steps, short_labels)

    return prog


def gir_text(prog) -> str:
    """Render an ALREADY-BUILT GIR Program as text for the `OutputLoopIR` dump."""
    if prog is None:
        return "# GIR unavailable: this kernel did not go through the GIR lowering path\n"
    try:  # deliberately deferred: this is an observability dump, so an import failure here
        from .gir.render import render_gir, gir_counts   # must degrade to a note, not a crash
    except Exception as e:
        return f"# GIR unavailable (import error): {e}\n"
    try:
        head = ("# GIR dump (observability; the finalized layer-2 program handed to L3)\n"
                f"# counts = {gir_counts(prog)}\n")
        return head + render_gir(prog) + "\n"
    except Exception as e:
        return f"# GIR render failed: {e}\n"


def build_gir(theta, mainloop=None, params=None, pipeline=None):
    """R-ONCE: build the theta model, lower to GIR, and run the pass pipeline EXACTLY ONCE, returning
    the finalized Program the phase forks emit from.
    """
    prog = lower_to_gir(theta, mainloop)
    if params:
        prog.params.update(params)
    run_pipeline(prog, passes=(pipeline if pipeline is not None else default_pipeline()))
    return prog
