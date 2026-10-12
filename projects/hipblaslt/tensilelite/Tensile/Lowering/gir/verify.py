# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
verify_gir -- the GIR well-formedness verifier.
"""

from __future__ import annotations

from .nodes import Move, Mma, Mark, MARK_KINDS, terminator_targets
from .analysis import AnalysisManager
from .analyses import BackEdges
from .analyses.loop_shape import LoopShape, reduction_coverage_violations
from .analyses.barrier_uniformity import BarrierUniformity
from .analyses.frame_hazards import FrameHazards, _disjoint_storage
from .analyses.reg_band import RegBandAnalysis
from .analyses.lds_buffers import LdsBufferIds, shared_refs
from .analyses.region_increment import walk_violations


# Per-kind Mark `at` payload schema.
_MARK_SCHEMA = {
    "phase_boundary": ("phase",),
    "fence":          ("buffers", "scope", "kinds", "edges"),
    # A swap additionally names the pointer, but WHICH name depends on the hop: a read swap moves
    # an operand's LocalReadAddr (`operand`), a copy swap moves a Phi movement's descriptor
    "swap":           ("hop", "gen_from", "gen_to"),
    "gr_increment":   ("unit",),
    # A region_increment moves the TDM descriptor between STORAGE REGIONS of one split tile.
    # `steps` is SIGNED -- forward to a later region, negative to walk back -- and the walk must CLOSE
    "region_increment": ("unit", "steps", "from_region", "to_region"),
    # A member that does not move on this step: its descriptor is nulled (count=0) and restored.
    "descriptor_enable": ("unit", "member", "enabled"),
    "gl2_prefetch":   ("depth", "block"),
    "gsu_guard":      ("variant",),
    # The chunk a movement's address must hold HERE, so an advance serving a conditionally
    # executed successor is established on the path both arms take.
    "chunk_pin":      ("unit", "chunk"),
    # The buffer generation a movement's pointer must hold HERE, for the same reason.
    "buffer_pin":     ("hop", "unit", "gen"),
}


def _all_insts(prog):
    for blk in prog.blocks.values():
        for inst in blk.body:
            yield blk, inst


def _refs_of(inst):
    if isinstance(inst, (Move, Mma)):
        return tuple(inst.srcs) + tuple(inst.dsts)
    return ()


def _check_gen_ssa(prog):
    # Collect Gen defs (phis + xfers) and Gen uses (Refs).
    defs = set()
    for blk in prog.blocks.values():
        seen = set()
        for phi in blk.phis:
            if phi.gen.id in seen:
                raise RuntimeError(
                    f"G-GEN-SSA: block {blk.label!r} has several phis for Gen id={phi.gen.id}")
            seen.add(phi.gen.id)
            defs.add(phi.gen.id)
            if phi.incomings and {pred for pred, _value in phi.incomings} != set(blk.preds):
                raise RuntimeError(
                    f"G-GEN-SSA: phi for Gen id={phi.gen.id} in {blk.label!r} does not name "
                    "every predecessor")
    uses = set()
    for blk, inst in _all_insts(prog):
        for ref in _refs_of(inst):
            if getattr(ref, "gen", None) is not None:
                uses.add(ref.gen.id)
    # every Gen with a def must have a use (strict; a dropped consumer is the bug)
    for gid in defs:
        if gid not in uses:
            raise RuntimeError(
                f"G-GEN-SSA: Gen id={gid} is defined but never used (dropped consumer). "
                f"A scaffold-owned span must NOT mint a Gen def.")
    # every used Gen must be defined
    for gid in uses:
        if gid not in defs:
            raise RuntimeError(f"G-GEN-SSA: Gen id={gid} is used but has no phi def")


def _check_noiter(prog):
    def bad(coord):
        return any(m == "iter" for m, _ in coord)
    for blk, inst in _all_insts(prog):
        for ref in _refs_of(inst):
            if bad(ref.tile.coord):
                raise RuntimeError(f"G-NOITER: Tile coord carries 'iter' in {blk.label}")
        if isinstance(inst, Mma) and bad(inst.coord):
            raise RuntimeError(f"G-NOITER: Mma coord carries 'iter' in {blk.label}")


_PHYSICAL_ATTRS = ("vgpr", "addr", "byte_off", "reg_off", "vector_width", "color")


def _check_semantic(prog):
    # GIR nodes structurally have no physical field; guard against accidental attachment.
    for blk, inst in _all_insts(prog):
        for a in _PHYSICAL_ATTRS:
            if hasattr(inst, a):
                raise RuntimeError(f"G-SEMANTIC: node in {blk.label} carries physical attr '{a}'")


def _check_phase_counts(phases):
    """At most one prologue and one tail block."""
    n_pro = sum(1 for p in phases if p == "prologue")
    if n_pro > 1:
        raise RuntimeError(f"G-CFG: {n_pro} prologue blocks (expected <=1)")
    n_tail = sum(1 for p in phases if p == "tail")
    if n_tail > 1:
        raise RuntimeError(f"G-CFG: {n_tail} tail blocks (expected <=1)")
    # exactly M drain blocks, M taken FROM THE LOWERING (`meta['M']`, the peel depth theta derives),
    # never approximated as PGR-1.  This half of G-CFG was documented but unchecked; a lowering


def _check_drain_count(phases, steps):
    """The drain has exactly one block per chunk the lowering says ramps out."""
    if steps is not None:
        n_drain = sum(1 for p in phases if isinstance(p, str) and p.startswith("drain"))
        if n_drain != steps:
            raise RuntimeError(
                f"G-CFG: {n_drain} drain blocks but the lowering records {steps} drain step(s) "
                f"-- the drain must be exactly that many straight-line steps ")


def _check_short_arm_count(prog, phases, M):
    """The `T < M` arm has one block per peeled chunk, or none once it is folded."""
    sl = prog.meta.get("short_loop")
    n_short = sum(1 for p in phases if isinstance(p, str) and p.startswith("short"))
    if sl is not None and M is not None:
        if sl.get("steps") != M:
            raise RuntimeError(
                f"G-CFG: short_loop records {sl.get('steps')} steps but the peel depth M is {M} -- "
                f"the T<M path has exactly one step per peeled chunk")
        want = 0 if sl.get("verdict") == "folded" else M
        if n_short != want:
            raise RuntimeError(
                f"G-CFG: {n_short} short blocks but verdict {sl.get('verdict')!r} requires "
                f"{want} -- a folded arm keeps none, an unfolded one keeps all M")
    elif n_short:
        raise RuntimeError(
            f"G-CFG: {n_short} short blocks but no short_loop record -- the arm's verdict must be "
            f"carried, not inferred from the block names")
    # steady loops = blocks with a self/dominating back-edge.
    #


def _check_loop_headers(prog, am, expect_steady_loops):
    """Exactly one steady loop header, unless the caller expects more."""
    be = am.get(BackEdges(), prog)
    tail_labels = {lab for lab, blk in prog.blocks.items() if blk.phase == "tail"}
    loop_headers = {e.header for e in be} - tail_labels
    if expect_steady_loops == "one" and len(loop_headers) > 1:
        raise RuntimeError(
            f"G-CFG: {len(loop_headers)} summation-loop headers in single-tile scope (expected "
            f"exactly one; the tail loop is counted separately); persistent scope relaxes this "
            f"to >=1 ")


def _check_cfg(prog, am, expect_steady_loops="one"):
    phases = [blk.phase for blk in prog.blocks.values()]
    M = prog.meta.get("peel_depth")
    _check_phase_counts(phases)
    _check_drain_count(phases, prog.meta.get("drain_steps", M))
    _check_short_arm_count(prog, phases, M)
    _check_loop_headers(prog, am, expect_steady_loops)


def _check_walk(prog):
    """G-WALK: the TDM region walk CLOSES in every block.
 A region-split tile is loaded one region at a time and the descriptor steps between them. Those
 steps must net to zero per block, because `tdmIncrementGir` then applies the PLAIN chunk stride
 -- anything left over compounds every trip.
 """
    bad = walk_violations(prog)
    if bad:
        raise RuntimeError("; ".join(bad))


def _check_tokens(prog, am):
    """G-TOKEN: the LDS buffer numbering is a correct naming of shared storage.

    The id is the alias relation StinkyTofu's frame pipeline resolves each access against, so it
    is not decoration: every access must resolve, and provably disjoint storage must not collide.
    """
    tokens = am.get(LdsBufferIds(), prog)
    if tokens.unresolved:                                            # V2
        raise RuntimeError(
            f"G-TOKEN: {len(tokens.unresolved)} shared access(es) could not be named "
            f"{tokens.unresolved[:3]} -- an unnamed LDS access breaks StinkyTofu's all-or-none "
            f"token rule for its whole basic block, so this is not a partial result")
    for e in am.get(FrameHazards(), prog):                           # V1
        if not _disjoint_storage(e.producer, e.consumer):
            continue
        shared = (set(tokens.ids_for(e.producer.ref))
                  & set(tokens.ids_for(e.consumer.ref)))
        if shared:
            raise RuntimeError(
                f"G-TOKEN: {e.producer.storage} and {e.consumer.storage} are provably disjoint "
                f"storage but share token id(s) {sorted(shared)} -- a shared id orders them for no "
                f"reason (precision), and means the key no longer distinguishes what it claims")


def _check_emit(prog):
    counted = 0
    for rec in (prog.meta.get("short_loop") or {}).get("model_only", {}),:
        counted += rec.get("marks_dropped", 0) if rec else 0
    actual = sum(1 for blk in prog.blocks.values() if blk.model_only
                 for n in blk.body if isinstance(n, Mark) and n.kind != "phase_boundary")
    declared_blocks = {b.label for b in prog.blocks.values() if b.model_only}
    if declared_blocks and not (prog.meta.get("short_loop") or {}).get("model_only"):
        raise RuntimeError(
            f"G-EMIT: blocks {sorted(declared_blocks)} are model_only but nothing records WHY or "
            f"how much was discarded with them -- an unemitted block must be a stated decision")
    if actual != counted:
        raise RuntimeError(
            f"G-EMIT: {actual} state-changing Mark(s) sit in model-only blocks but {counted} were "
            f"recorded -- a pass added to a block nothing emits without declaring the loss.  Either "
            f"place them where they are emitted, or update the record so the drop stays visible")


def _check_barrier_uniformity(prog, am):
    """G-UNIFORM. Every cross-wave fence is reached by every wave of its scope."""
    viol = am.get(BarrierUniformity(), prog)
    if viol:
        raise RuntimeError(
            "G-UNIFORM: %d cross-wave fence(s) some wave of the scope can skip -- a rendezvous "
            "off one wave's path hangs the waves that do reach it, and the empty-ledger gate "
            "cannot see it: %s" % (len(viol), "; ".join(repr(v) for v in viol[:6])))


def _check_trip(prog, am):
    viol = reduction_coverage_violations(prog, am.get(LoopShape(), prog))
    if viol:
        raise RuntimeError("G-TRIP: " + "; ".join(viol))


def _check_term(prog):
    for lab, blk in prog.blocks.items():
        if blk.term is None:
            raise RuntimeError(f"G-TERM: block {lab} has no terminator")
        targets = set(terminator_targets(blk.term))
        for t in targets:
            if t not in prog.blocks and t != "end":
                raise RuntimeError(f"G-TERM: block {lab} targets missing block '{t}'")
        unknown_xfers = set((blk.edge_xfers or {})) - targets
        if unknown_xfers:
            raise RuntimeError(
                f"G-TERM: block {lab} has edge transfers for non-successors "
                f"{sorted(unknown_xfers)}")
    # declared succs must equal the terminator's raw targets ("end" sink included)
    for lab, blk in prog.blocks.items():
        if blk.succs:
            declared = set(blk.succs)
            actual = set(terminator_targets(blk.term))
            if declared != actual:
                raise RuntimeError(
                    f"G-TERM: block {lab} succs {sorted(declared)} disagree with terminator "
                    f"{sorted(actual)}")
    # declared preds must equal the CFG preds among REAL blocks (the "end" sink has no block)
    computed_preds = {lab: set() for lab in prog.blocks}
    for lab, blk in prog.blocks.items():
        for t in terminator_targets(blk.term):
            if t in prog.blocks:
                computed_preds[t].add(lab)
    for lab, blk in prog.blocks.items():
        declared = set(blk.preds)
        if declared and declared != computed_preds[lab]:
            raise RuntimeError(
                f"G-TERM: block {lab} preds {sorted(declared)} disagree with CFG "
                f"{sorted(computed_preds[lab])}")


def _check_operands(prog):
    # Every register-resident Mma src Tile must be produced by a Move (its read) somewhere.
    produced = set()   # (operand, space) a Move writes
    for blk, inst in _all_insts(prog):
        if isinstance(inst, Move):
            for d in inst.dsts:
                produced.add((d.tile.operand, d.tile.space))
    for blk, inst in _all_insts(prog):
        if isinstance(inst, Mma):
            for s in inst.srcs:
                key = (s.tile.operand, s.tile.space)
                if s.tile.space == "register" and key not in produced:
                    raise RuntimeError(
                        f"G-OPERANDS: Mma src {s.tile.operand}@{s.tile.space} in {blk.label} "
                        f"has no producing Move (dangling operand)")


def _check_marks(prog):
    for blk, inst in _all_insts(prog):
        if not isinstance(inst, Mark):
            continue
        if inst.kind not in MARK_KINDS:
            raise RuntimeError(f"G-MARK: unknown Mark kind '{inst.kind}' in {blk.label}")
        need = _MARK_SCHEMA[inst.kind]
        have = set((inst.at or {}).keys())
        missing = [k for k in need if k not in have]
        if missing:
            raise RuntimeError(
                f"G-MARK: Mark('{inst.kind}') in {blk.label} missing fields {missing}")
        if inst.kind == "swap":
            # The pointer name is hop-dependent (see _MARK_SCHEMA).  Requiring the RIGHT one, not
            # merely "some name", is what stops a copy swap from being labelled with a bare operand
            key = "operand" if inst.at.get("hop") == "read" else "unit"
            if key not in have:
                raise RuntimeError(
                    f"G-MARK: Mark('swap', hop={inst.at.get('hop')!r}) in {blk.label} must name "
                    f"its pointer with '{key}' (a read swap names an operand, a copy swap names "
                    f"the Phi movement's member tuple); got fields {sorted(have)}")


def verify_gir(prog, expect_steady_loops="one"):
    """Run every G-* check.  Raises RuntimeError on the first violation; returns True if the
    Program is well-formed.  `expect_steady_loops='ge1'` relaxes G-CFG for persistent scope."""
    am = AnalysisManager()
    _check_term(prog)
    _check_cfg(prog, am, expect_steady_loops)
    _check_walk(prog)
    _check_tokens(prog, am)
    _check_emit(prog)
    _check_barrier_uniformity(prog, am)
    _check_trip(prog, am)
    _check_gen_ssa(prog)
    _check_noiter(prog)
    _check_semantic(prog)
    _check_operands(prog)
    _check_marks(prog)
    check_register_slots(prog, am)
    check_rotation_waw(prog, am)
    check_block_scope_covered(prog, am)
    return True


def check_register_slots(prog, am):
    """G-SLOT. Every register destination carries a concrete rotation slot, and the width W each
    (operand, group) rotates through is consistent.

    W is a LoopIR fact GIR consumes, never one GIR decides; `RegBandAnalysis` raises if the refs
    of one group disagree about it.
    """
    am.get(RegBandAnalysis(), prog)
    layout = prog.meta.get("register_layout", {}) or {}
    for blk in prog.blocks.values():
        for inst in blk.body:
            refs = ((inst.dsts if isinstance(inst, Move) else inst.srcs)
                    if isinstance(inst, (Move, Mma)) else ())
            for ref in refs:
                if ref.tile.space != "register":
                    continue
                if ref.slot is None:
                    raise RuntimeError(
                        f"G-SLOT: register read of {ref.tile.operand} in {blk.phase} has no "
                        f"concrete slot (lowering bug)")
                op_layout = layout.get(ref.tile.operand)
                if not op_layout:
                    continue
                key = (int(ref.group), int(ref.slot))
                base = op_layout.get("slots", {}).get(key)
                if base is None:
                    raise RuntimeError(
                        f"G-SLOT: {ref.tile.operand} group/slot {key} in {blk.phase} has no "
                        f"compact register-layout entry")
                if ref.unit_index is None:
                    raise RuntimeError(
                        f"G-SLOT: {ref.tile.operand} group/slot {key} in {blk.phase} has no "
                        f"unit index")
                units = tuple(ref.unit_indexes or (ref.unit_index,))
                end = int(base) + (max(map(int, units)) + 1) * int(ref.size_regs)
                if end > int(op_layout.get("total", 0)):
                    raise RuntimeError(
                        f"G-SLOT: {ref.tile.operand} group/slot {key} unit {ref.unit_index} "
                        f"in {blk.phase} ends at VGPR offset {end}, beyond compact allocation "
                        f"{op_layout.get('total', 0)}")


def check_block_scope_covered(prog, am):
    """G-SCOPE. Every operand theta gives a block-scoped shared residency has a cross-wave hazard
    edge for the fence analysis to cover.

    The two derivations of "is this edge cross-wave" -- theta's wave-distributed movement, and
    `FrameHazards`' buffer overlap across waves -- must agree, or a cooperative fill and the reads
    of it are separated by nothing.
    """
    hz = am.get(FrameHazards(), prog)
    cross_operands = set()
    for h in hz.needing_fence():
        cross_operands.add(h.producer.operand)
        cross_operands.add(h.consumer.operand)
    claimed = {}
    for name, blk in prog.blocks.items():
        for i, inst in enumerate(blk.body):
            for dep, counter, kind, scope in (getattr(inst, "deps", ()) or ()):
                if scope != "block":
                    continue
                # `dep` is a ledger Endpoint; `.op` is the operand name, or the group tuple when
                # a Phi-fused movement names several operands at once.
                op = getattr(dep, "op", dep)
                for m in (op if isinstance(op, tuple) else (op,)):
                    claimed.setdefault(m, (name, i, kind))
    missing = {op: w for op, w in claimed.items() if op not in cross_operands}
    if missing:
        det = "; ".join(f"{op} (first at {b}[{i}] as {k})" for op, (b, i, k) in
                        sorted(missing.items())[:4])
        raise RuntimeError(
            f"G-SCOPE: theta declares a BLOCK-scoped shared residency for operand(s) "
            f"{sorted(missing)}, but `FrameHazards` finds no cross-wave shared edge for them, so "
            f"StinkyTofu's fence placement never had an edge to cover and nothing guarantees a "
            f"barrier between their cooperative fill and the reads of it.  {det}")


def check_rotation_waw(prog, am):
    """G-WAW. Every rotating buffer is read, and over every axis it is written.

    The ledger files a rotation WAR but no rotation WAW, which is sound only while the vacating
    read orders the two generations. A buffer nothing reads, or bytes no read covers, has no such
    read, so the WAW is unordered and must be filed instead.
    """
    tokens = am.get(LdsBufferIds(), prog)
    writes, reads, wcov, rcov = {}, {}, {}, {}
    for _blk, _inst, ref, is_write in shared_refs(prog):
        cov = frozenset(a for a, _f in (getattr(ref, "covers", ()) or ()))
        for t in tokens.ids_for(ref):
            tgt, ctgt = (writes, wcov) if is_write else (reads, rcov)
            tgt[t] = tgt.get(t, 0) + 1
            ctgt.setdefault(t, set()).update(cov)
    names = {i: b.render() for b, i in tokens.buffers.items()}
    # Generations ROTATE: the write of one is consumed by the read of that generation on a later
    # trip, and in a frame-relative block no single absolute generation shows both ends.  So the
    # thing that must have a reader is the ring, not each of its generations.
    ring_of = {i: (b.operand, b.region) for b, i in tokens.buffers.items()}
    ring_reads = {}
    for t, n in reads.items():
        ring_reads[ring_of.get(t)] = ring_reads.get(ring_of.get(t), 0) + n
    for t, n in sorted(writes.items()):
        if reads.get(t, 0) == 0 and ring_reads.get(ring_of.get(t), 0) == 0:
            raise RuntimeError(
                f"G-WAW: buffer {names.get(t, t)!r} is written {n}x and NEVER READ, so "
                f"generations t and t+S are ordered only by program-order-on-issue, which does "
                f"not order completions.  The rotation WAW row must be filed for it")
        missing = wcov.get(t, set()) - rcov.get(t, set())
        if missing:
            raise RuntimeError(
                f"G-WAW: buffer {names.get(t, t)!r} is written over axes {sorted(missing)} that "
                f"no read covers -- a pure write-after-write the WAR cannot see; the rotation "
                f"WAW row must be filed")
