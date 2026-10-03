# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Analyze TDM descriptor steps between storage regions."""

from __future__ import annotations

from ..nodes import Move, Mark
from ..nodes import copy_unit, descriptor_unit, Region, PendingMark, BLOCK_EXIT, EARLIEST
from ..analysis import Analysis


def region_of(prog, ref):
    """Return the region coordinate named by a shared reference."""
    rms = (prog.meta.get("region_axes", {}) or {}).get(ref.tile.operand) or ()
    # An MX scale follows its parent's M_split but has one storage region, so it
    # has no region coordinate of its own.
    if not rms or int((prog.meta.get("operand_regions", {}) or {}).get(ref.tile.operand, 1)) <= 1:
        return ()
    coord = dict(ref.tile.coord or ())
    return tuple(coord.get(rm) for rm in rms)


def walk_ref(prog, refs):
    """Return the reference belonging to the member with the most regions."""
    have = prog.meta.get("operand_regions", {}) or {}
    return max(refs, key=lambda r: int(have.get(r.tile.operand, 1)))


def _flat(region, prog, operand):
    """Flatten an operand's mixed-radix region coordinate."""
    rms = (prog.meta.get("region_axes", {}) or {}).get(operand) or ()
    ext = prog.meta.get("axis_extents", {}) or {}
    idx = 0
    for rm, v in zip(rms, region):
        if v is None:
            return None
        idx = idx * max(1, ext.get(rm, 1)) + int(v)
    return idx


#: The chunk advance and the region walk are both per DESCRIPTOR; `nodes.descriptor_unit` is the
#: one derivation, shared with `swap_regions`.
_canonical_unit = descriptor_unit


def _short_members(prog, unit):
    """Members of `unit` owning fewer regions than the unit walks -- they ride region 0."""
    per = (prog.meta.get("unit_member_regions", {}) or {}).get(tuple(unit), {})
    n = _n_regions(prog, unit)
    return tuple(m for m in unit if max(1, int(per.get(m, n))) < n)


def _n_regions(prog, unit):
    """How many storage regions this movement is actually split into.

    From `meta['unit_regions']`, the count theta derived -- NOT from `region_axes` being non-empty.
    An operand can own a region AXIS while being split into ONE region (the MX scale operands:
    `region_modes=('M_split',)`, `split=1`); its copy is a single movement and carries no region
    coordinate, so treating the axis as a walk asks it for a value it correctly does not have."""
    reg = prog.meta.get("unit_regions", {}) or {}
    if tuple(unit) in reg:
        return max(1, int(reg[tuple(unit)]))
    # A SOLO SUBSET walks its own operand's regions: the later regions of a split member are its
    # own movement, so the count comes from the member, not from a fuse-group key it is not.
    own = prog.meta.get("operand_regions", {}) or {}
    return max(1, max((int(own.get(m, 1)) for m in unit), default=1))


def _split_copies(prog, blk):
    """[(node, unit, descriptor, flat_region, ...)] for the region-split copy Moves of `blk`, in
 program order.  `descriptor` is the walk key; `unit` stays the members that actually move."""
    out = []
    for node in blk.body:
        if not isinstance(node, Move):
            continue
        members, refs = copy_unit(node)
        if members is None:
            continue
        desc = _canonical_unit(prog, members)
        if _n_regions(prog, desc) <= 1:
            continue                       # unsplit movement: no walk, no marks
        wref = walk_ref(prog, refs)
        reg = region_of(prog, wref)
        if not reg:
            continue
        flat = _flat(reg, prog, wref.tile.operand)
        if flat is None:
            continue                       # unpinned axis --, see `_flat`
        # READING refs[0] IS JUSTIFIED BY A PRECONDITION, SO CHECK THE PRECONDITION.
        #
        for _r in [r for r in refs if r is not wref]:
            _reg = region_of(prog, _r)
            _other = _flat(_reg, prog, _r.tile.operand) if _reg else None
            if _other is not None and _other != flat:
                raise RuntimeError(
                    "GIR: fused movement %s pairs region %d of %r with region %d of %r -- a Phi "
                    "group is paired BY INDEX, and the aliased descriptor advances "
                    "every member with ONE step.  Differing indices need one add per axis, which "
                    "L3 refuses."
                    % (members, flat, wref.tile.operand, _other, _r.tile.operand))
        out.append((node, members, desc, flat, reg,
                    tuple((prog.meta.get("region_axes", {}) or {}).get(
                        wref.tile.operand) or ())))
    return out


class RegionIncrementRegions(Analysis):
    """See module docstring. Pure; returns `[PendingMark]` of kind `region_increment`.

 `steps` on the Mark is SIGNED: positive walks forward to a later region, negative walks back.
 L3 turns each unit step into one `tdmRegionIncrementGir(back=steps < 0)` -- GIR owns WHERE and
 HOW MANY, the backend owns the byte quantities, which is the standing split."""

    def run(self, prog, am):
        pending = []
        for blk in prog.blocks.values():
            at, prev = {}, {}                # descriptor -> region it sits on / its last copy node
            vec, axes = {}, {}                # descriptor -> per-axis coord tuple / axis names
            last = {}                        # descriptor -> members of its last movement
            for node, unit, desc, reg, rtup, rms in _split_copies(prog, blk):
                last[desc] = unit
                cur = at.get(desc, 0)                   # the setup leaves it on region 0
                axes[desc] = rms
                curv = vec.get(desc, tuple(0 for _ in rtup))
                if reg != cur:
                    # TIGHTLY BOUNDED: `after` is this descriptor's PREVIOUS copy, `before` is this.
                    pending.append(PendingMark(
                        Mark("region_increment", {"unit": unit, "steps": reg - cur,
                                                  "from_region": cur, "to_region": reg,
                                                  "axis_steps": _axis_delta(rms, curv, rtup)}),
                        Region(blk.label, after=prev.get(desc), before=node, policy=EARLIEST)))
                    # leaving region 0 nulls the short members; wrapping back restores them
                    if cur == 0 or reg == 0:
                        for m in _short_members(prog, unit):
                            pending.append(PendingMark(
                                Mark("descriptor_enable",
                                     {"unit": unit, "member": m, "enabled": reg == 0}),
                                Region(blk.label, after=prev.get(desc), before=node,
                                       policy=EARLIEST)))
                at[desc], prev[desc], vec[desc] = reg, node, tuple(rtup)
            # CLOSE THE WALK.  The chunk stride `tdmIncrementGir` applies assumes the descriptor is
            # back at the chunk base, so anything left displaced at the block exit is walked back.
            for desc, reg in sorted(at.items(), key=str):
                if reg:
                    unit = last.get(desc, desc)
                    pending.append(PendingMark(
                        Mark("region_increment", {"unit": unit, "steps": -reg,
                                                  "from_region": reg, "to_region": 0,
                                                  "axis_steps": _axis_delta(
                                                      axes.get(desc, ()), vec.get(desc, ()),
                                                      tuple(0 for _ in vec.get(desc, ())))}),
                        Region(blk.label, after=prev.get(desc), before=BLOCK_EXIT,
                               policy=EARLIEST)))
                    for m in _short_members(prog, unit):
                        pending.append(PendingMark(
                            Mark("descriptor_enable",
                                 {"unit": unit, "member": m, "enabled": True}),
                            Region(blk.label, after=prev.get(desc), before=BLOCK_EXIT,
                                   policy=EARLIEST)))
        return pending


def _axis_delta(rms, frm, to):
    """`{axis: delta}` for the non-zero axes of one descriptor step."""
    return {m: int(b) - int(a) for m, a, b in zip(rms, frm, to) if int(b) != int(a)}


def walk_violations(prog):
    """[] or human-readable G-WALK violations.  Two properties, both replayed in PROGRAM ORDER:

      1. EVERY SPLIT COPY LOADS THE REGION THE DESCRIPTOR IS ON.  Walking the block, the running
         position must equal each copy's own region coordinate when that copy issues.
    """
    out = []
    for blk in prog.blocks.values():
        at = {}
        for node in blk.body:
            if isinstance(node, Mark) and node.kind == "region_increment":
                a = node.at if isinstance(node.at, dict) else {}
                u = _canonical_unit(prog, a.get("unit") or ())
                at[u] = at.get(u, 0) + int(a.get("steps", 0))
            elif isinstance(node, Move):
                members, refs = copy_unit(node)
                if members is None:
                    continue
                members = _canonical_unit(prog, members)
                if _n_regions(prog, members) <= 1:
                    continue
                reg = region_of(prog, refs[0])
                flat = _flat(reg, prog, refs[0].tile.operand) if reg else None
                if reg and flat is None:
                    out.append(
                        "%s: copy of %s names region %s with an UNPINNED axis, so its walk cannot "
                        "be derived and no region increment is emitted for it.  region_modes=%s "
                        "(a movement split on SEVERAL axes at once --; the per-axis step "
                        "vector is not implemented, so this shape is unsupported, not clean)."
                        % (blk.label, refs[0].tile.operand, reg,
                           (prog.meta.get("region_axes", {}) or {}).get(refs[0].tile.operand)))
                    continue
                if flat is None:
                    continue                        # unsplit movement: nothing to walk
                if at.get(members, 0) != flat:
                    out.append(
                        f"G-WALK: in {blk.label!r} a copy of {members!r} loads region {flat} but "
                        f"the descriptor is on region {at.get(members, 0)} -- every split copy must "
                        f"load the region the walk has reached")
        for unit, n in sorted(at.items(), key=str):
            if n:
                out.append(
                    f"G-WALK: block {blk.label!r} leaves the TDM descriptor for {unit!r} {n} "
                    f"region(s) from the chunk base -- the walk must close, because "
                    f"`tdmIncrementGir` applies the plain chunk stride and would compound the "
                    f"leftover every trip")
        out += _enable_violations(blk)
    return out


def _enable_violations(blk) -> list:
    """A descriptor nulled and never restored issues no load at all -- silently, so nothing
    downstream can catch it."""
    off = {}
    for node in blk.body:
        if isinstance(node, Mark) and node.kind == "descriptor_enable":
            a = node.at if isinstance(node.at, dict) else {}
            key = (a.get("unit"), a.get("member"))
            off[key] = not a.get("enabled")
    return [f"G-ENABLE: block {blk.label!r} leaves {member!r} of {unit!r} NULLED -- its "
            f"tensor_load_to_lds never issues and the operand is silently never loaded"
            for (unit, member), disabled in sorted(off.items(), key=str) if disabled]
