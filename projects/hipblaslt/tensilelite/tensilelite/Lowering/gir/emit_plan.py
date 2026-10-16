# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
Emit plan -- the PURE half of GIR -> rocisa (L3).
"""

from __future__ import annotations
from dataclasses import dataclass, replace

from .nodes import Move, Mma, Mark
from .analyses.region_increment import region_of, walk_ref, _flat, _n_regions
from .nodes import copy_unit, covered_coords, first_shared_ref


@dataclass(frozen=True)
class EmitAction:
    """One thing L3 must emit: its `kind`, and the `at` payload that kind needs."""
    kind: str
    at: dict
    # Dense, deterministic identity in Program block/action order. ``source`` is the GIR node
    # that produced this action and is intentionally not serialized by ordinary emitters.
    action_id: int = -1
    source: object = None


def _project(coord, axes, inner_order, extents):
    """Mixed-radix combine of `coord` over `axes` -- an operand's free axes, or the summation
    axes -- into one flat index, in loop order."""
    present = [m for m in inner_order if m in axes]
    idx, stride = 0, 1
    for name in reversed(present):
        idx += coord.get(name, 0) * stride
        stride *= max(1, extents.get(name, 1))
    return idx


def _register_dst(refs):
    for r in refs:
        if r.tile.space == "register":
            return r
    return None


def _plan_read(prog, inst, dst_reg, red, free, order, ext, actions):
    """A shared(or global DTV)->register read: its tile index, summation index and region."""
    if dst_reg is not None:
        # a READ: shared(or global DTV) -> register.  free-tile index = project the read's coord
        # onto THIS operand's free axes; summation index = project onto the summation axes.
        operand = dst_reg.tile.operand
        coord = {ax: (int(v) if v is not None else 0) for ax, v in dst_reg.tile.coord}
        nregions = int((prog.meta.get("operand_regions", {}) or {}).get(operand, 1) or 1)
        rmodes = set((prog.meta.get("region_axes", {}) or {}).get(operand) or ()) \
            if nregions > 1 else set()
        freeset = set(free.get(operand, []))
        tile = _project(coord, freeset - rmodes, order, ext)
        region = _project(coord, rmodes, order, ext) if rmodes else 0
        # A wave-chosen free split is already present in the lane address. Every coordinate-relative
        # split, free OR reduction, needs its explicit packed-region displacement. Reducing this to
        # ``rmodes & red`` dropped M/N-split region 1 for transpose layouts.
        agent_relative = set(prog.meta.get("wave_relative_regions", ()) or ())
        address_region = 0 if operand in agent_relative else region
        tile_flat = _project(coord, freeset, order, ext)
        # AND `k` IS WITHIN-REGION TOO, for the same reason `tile` is.  When the split cuts the
        # REDUCTION axis the region already carries `K_split`, so leaving it in `k` would count
        k = _project(coord, set(red) - rmodes, order, ext)
        # AND THE ABSOLUTE REDUCTION INDEX, which `k` is not once a region carries part of it.
        k_flat = _project(coord, set(red), order, ext)
        reg_buf = dst_reg.slot if dst_reg.slot is not None else 0
        fills = tuple(
            (_project(cd, freeset - rmodes, order, ext),
             _project(cd, rmodes, order, ext) if rmodes else 0,
             _project(cd, freeset, order, ext),
             _project(cd, set(red) - rmodes, order, ext),
             _project(cd, set(red), order, ext))
            for cd in ({ax: (int(v) if v is not None else 0) for ax, v in c}
                       for c in covered_coords(dst_reg)))
        unit_indexes = tuple(dst_reg.unit_indexes or (dst_reg.unit_index,))
        actions.append(EmitAction("read", {
            "tc": operand, "tile": tile, "k": k, "k_flat": k_flat, "reg_buf": reg_buf,
            # `regions` is THIS OP-CLASS's count, not its movement's: under Phi the movement is keyed
            # `('A','B')` and `_n_regions(prog, ('A',))` would answer 1 for a split fused operand.
            "region": region, "address_region": address_region,
            "regions": nregions, "tile_flat": tile_flat, "fills": fills,
            "group": dst_reg.group, "size_regs": dst_reg.size_regs,
            "unit_index": dst_reg.unit_index, "unit_indexes": unit_indexes,
            "issue": tuple(getattr(inst, "issue", ())),
            # The LDS generation this read SOURCES; a DTV read has none and carries None.
            "gen": _ref_generation(prog, first_shared_ref(inst.srcs)),
            "advance": int(getattr(inst, "advance", 0) or 0)}))
        return


def _ref_generation(prog, ref):
    """The buffer a shared Ref names, for read and copy alike: a peel ref pins one, a loop-carried
    ref advances its phi by its own step.  `abs_gen` is None for every steady access."""
    if ref is None:
        return None
    gen = getattr(ref, "gen", None)
    if gen is None:
        return int(ref.abs_gen or 0)
    entry = 0
    for block in prog.blocks.values():
        for phi in block.phis:
            if phi.gen.id == gen.id and phi.entry_val is not None:
                entry = int(phi.entry_val)
                break
    return (entry + int(ref.gdelta)) % max(1, int(gen.ring))


def _plan_move(prog, inst, actions):
    red = prog.meta.get("summation_axes", [])
    free = prog.meta.get("free_axes", {})
    order = prog.meta.get("inner_axis_order", [])
    ext = prog.meta.get("axis_extents", {})
    dst_reg = _register_dst(inst.dsts)
    _plan_read(prog, inst, dst_reg, red, free, order, ext, actions)
    members, dst_refs = copy_unit(inst)
    if members is not None:
        gen = _ref_generation(prog, dst_refs[0])
        n_reg = _n_regions(prog, members)
        wref = walk_ref(prog, dst_refs)
        reg = region_of(prog, wref) if n_reg > 1 else ()
        actions.append(EmitAction("copy", {
            "unit": members, "gen": gen, "regions": n_reg,
            "region": _flat(reg, prog, wref.tile.operand) if reg else None}))


def _mma_scale_pairs(prog, inst, in0, in1, free):
    """Which scale operand rides with each matmul input, paired by their free axes."""
    scale_of = {}
    if in0 or in1:
        for r in inst.srcs:
            if r.tile.space != "register":
                continue
            op = r.tile.operand
            if op in (in0, in1):
                continue
            # A scale spanning its parent's region split has one fewer free axis than the
            # parent; pair on the axes they must share.
            # THIS scale's own region axes: the union over the kernel makes every operand's
            # distinguishing axis vanish, and then a scale matches both parents.
            _reg = set((prog.meta.get("region_axes", {}) or {}).get(op, ()) or ())
            _drop = _reg if int(
                (prog.meta.get("operand_regions", {}) or {}).get(op, 1)) <= 1 else set()
            _own = lambda n: set(free.get(n, [])) - _drop
            owner = [p for p in (in0, in1) if p and _own(op) == _own(p)]
            if len(owner) != 1:
                raise NotImplementedError(
                    f"wmma extra register source {op!r} has free axes {free.get(op)}, which "
                    f"matches {len(owner)} of the matmul inputs ({in0!r}: {free.get(in0)}, "
                    f"{in1!r}: {free.get(in1)}); a scale operand must follow exactly one parent.")
            scale_of[owner[0]] = op

    return scale_of


def _mma_sources(inst, in0, in1, idx0, idx1, scale_of, scale_idx=None):
    """The wmma's source refs, and the register buffer each input and scale reads from."""
    bufA = bufB = sizeA = sizeB = bufMXA = bufMXB = None
    unitA = unitB = unitMXA = unitMXB = None
    srcs = []
    seen = {}
    for r in inst.srcs:
        if r.tile.space != "register":
            continue
        op = r.tile.operand
        if op in (in0, in1):
            if op in seen and seen[op] != r.group:
                raise NotImplementedError(
                    f"wmma source operand {op!r} spans register groups {seen[op]} and {r.group}: "
                    f"the emit plan carries ONE slot per operand, so a multi-group operand has no "
                    f"faithful representation here ( multi-group MFMA addressing).")
            seen[op] = r.group
        if op == in0:
            bufA, sizeA, unitA = r.slot, r.size_regs, r.unit_index
            srcs.append((op, r.group, r.slot, r.unit_index, idx0))
        elif op == in1:
            bufB, sizeB, unitB = r.slot, r.size_regs, r.unit_index
            srcs.append((op, r.group, r.slot, r.unit_index, idx1))
        elif op == scale_of.get(in0):
            bufMXA, unitMXA = r.slot, r.unit_index
            srcs.append((op, r.group, r.slot, r.unit_index,
                         (scale_idx or {}).get(op, idx0)))
        elif op == scale_of.get(in1):
            bufMXB, unitMXB = r.slot, r.unit_index
            srcs.append((op, r.group, r.slot, r.unit_index,
                         (scale_idx or {}).get(op, idx1)))
    return (srcs, bufA, bufB, sizeA, sizeB, bufMXA, bufMXB,
            unitA, unitB, unitMXA, unitMXB)


def _plan_mma(prog, inst, actions):
    red = prog.meta.get("summation_axes", [])
    free = prog.meta.get("free_axes", {})
    order = prog.meta.get("inner_axis_order", [])
    ext = prog.meta.get("axis_extents", {})
    mma_inputs = prog.meta.get("mma_inputs", [])
    coord = {ax: (int(v) if v is not None else 0) for ax, v in inst.coord}
    # wmma grid: idx0/idx1 = the two matmul INPUTS' free-tile indices,
    # u = the summation index -- all the axes it varies over-projected, no m/n/k role literal.
    in0 = mma_inputs[0] if len(mma_inputs) > 0 else None
    in1 = mma_inputs[1] if len(mma_inputs) > 1 else None
    idx0 = _project(coord, set(free.get(in0, [])), order, ext) if in0 else 0
    idx1 = _project(coord, set(free.get(in1, [])), order, ext) if in1 else 0
    u = _project(coord, set(red), order, ext)
    # Per-operand register generation (buffer/slot) + size the wmma CONSUMES -- carried on the
    # src Refs by `loopir_to_gir._convert_mma` from LoopModel. The leaf
    scale_of = _mma_scale_pairs(prog, inst, in0, in1, free)
    # A scale spanning its parent's region split owns fewer tiles, so it is indexed over ITS OWN
    # free axes rather than the parent's flat index.
    scale_idx = {sc: _project(coord, set(free.get(sc, [])), order, ext)
                 for sc in scale_of.values() if sc}
    (srcs, bufA, bufB, sizeA, sizeB, bufMXA, bufMXB,
     unitA, unitB, unitMXA, unitMXB) = _mma_sources(
        inst, in0, in1, idx0, idx1, scale_of, scale_idx)
    groups = {op: group for op, group, _slot, _unit, _index in srcs}
    source_issues = {
        ref.tile.operand: tuple((axis, coord.get(axis, 0)) for axis in ref.issue_axes)
        for ref in inst.srcs if ref.tile.space == "register"
    }
    actions.append(EmitAction("wmma", {
        "idx0": idx0, "idx1": idx1, "u": u,
        "bufA": bufA, "bufB": bufB, "sizeA": sizeA, "sizeB": sizeB,
        "groupA": groups.get(in0), "groupB": groups.get(in1),
        "unitA": unitA, "unitB": unitB,
        # None on a kernel with no scales; the leaf switches on the kernel's own MXBlock, so these
        # are carried, not interpreted, here.
        "bufMXA": bufMXA, "bufMXB": bufMXB,
        "groupMXA": groups.get(scale_of.get(in0)), "groupMXB": groups.get(scale_of.get(in1)),
        "unitMXA": unitMXA, "unitMXB": unitMXB,
        "srcs": tuple(srcs), "source_issues": source_issues}))


def _plan_mark(inst, actions):
    if inst.kind == "swap":
        actions.append(EmitAction("swap", dict(inst.at)))
    elif inst.kind == "gr_increment":
        actions.append(EmitAction("gr_inc", dict(inst.at)))
    elif inst.kind == "region_increment":
        # ONE STEP of the TDM descriptor between storage regions.  Distinct from `gr_inc`.
        # that advances the movement to the next summation CHUNK, this moves within one chunk's
        actions.append(EmitAction("region_inc", dict(inst.at)))
    elif inst.kind == "descriptor_enable":
        actions.append(EmitAction("desc_enable", dict(inst.at)))
    elif inst.kind == "fence":
        # the PROC-SCOPED SELECTOR covering a set of cross-wave LDS hazards. L3 realizes
        # it; GIR only says where it stands and what it orders.
        actions.append(EmitAction("fence", dict(inst.at)))
    elif inst.kind == "gl2_prefetch":
        # the L2 warm-up pair, at the point `Gl2PrefetchRegions` chose.  Unlike every other
        # act here it names no operand and no movement -- the position IS the entire content.
        actions.append(EmitAction("gl2_prefetch", dict(inst.at)))
    # phase_boundary / gsu_guard: phase_boundary is a no-op for emit (label handled by fork);
    # gsu_guard is realized by the R4 scaffold-anchor path.
    elif inst.kind == "gsu_guard":
        actions.append(EmitAction("gsu_guard", dict(inst.at)))


def _plan_block_raw(prog, phase):
    blk = prog.block(phase)
    actions = []
    for inst in blk.body:
        begin = len(actions)
        if isinstance(inst, Mma):
            _plan_mma(prog, inst, actions)
        elif isinstance(inst, Move):
            _plan_move(prog, inst, actions)
        elif isinstance(inst, Mark):
            _plan_mark(inst, actions)
        for i in range(begin, len(actions)):
            actions[i] = replace(actions[i], source=inst)
    return actions


def plan_block(prog, phase):
    """Ordered [EmitAction] for ``phase`` with stable dense Program-wide action ids."""
    base = 0
    for label in prog.blocks:
        if label == phase:
            break
        base += len(_plan_block_raw(prog, label))
    return [replace(action, action_id=base + i)
            for i, action in enumerate(_plan_block_raw(prog, phase))]


def plan_program(prog):
    """{phase: [EmitAction]} for every block, in block order."""
    return {phase: plan_block(prog, phase) for phase in prog.blocks}
