# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
Semantic verification of the EMIT PLAN -- "does this schedule compute the right GEMM?" -- checked on
the plan alone, with no rocisa, no assembler and no GPU.

`verify_gir` checks that the IR is WELL-FORMED (blocks reachable, terminators sane, Refs resolved).
"""

from __future__ import annotations

from .analyses.cfg import successors
from .emit_plan import plan_block


def _wmma_inputs(prog):
    """`(in0, in1)` -- the operands of the two matmul GRID axes, from theta, no A/B literal."""
    mi = list(prog.meta.get("mma_inputs", []) or [])
    return (mi[0] if len(mi) > 0 else None, mi[1] if len(mi) > 1 else None)


def _reg_name(at):
    """The vgpr group a read WRITES / a wmma READS, keyed as `leaves` names it.
 `tile_flat`, not `tile`: the register file is not region-displaced, so the flat index
 is the register and the within-region index is only the address."""
    return (at["tc"], at.get("group"), at["reg_buf"], at.get("tile_flat", at["tile"]))


def _read_fills(at):
    """Every `(tile_flat, k_flat, residency-pass)` one read act delivers."""
    issue = tuple(at.get("issue") or ())
    f = at.get("fills")
    if not f:
        return ((at.get("tile_flat", at["tile"]), at.get("k_flat", at["k"]), issue),)
    return tuple((e[2], e[4], issue) for e in f)


def _expected(prog, at, which):
    """`(operand, tile_index, reduction_index)` the wmma's `which`-th input demands."""
    in0, in1 = _wmma_inputs(prog)
    op = in0 if which == 0 else in1
    return (op, at["idx0"] if which == 0 else at["idx1"], at["u"])


class _RegFile:
    """`name -> (source coordinates, used)` -- one abstract vgpr file."""

    def __init__(self, initial=None):
        self.slots = dict(initial or {})

    def snapshot(self):
        return {name: (sources, True) for name, (sources, _used) in self.slots.items()}

    def clone(self):
        return _RegFile(self.slots)


def _walk(prog, phase, regs, viol, acts=None):
    """Walk one block's plan, updating `regs` and appending violations."""
    for i, act in enumerate(plan_block(prog, phase) if acts is None else acts):
        at = act.at
        if act.kind == "read":
            fills = tuple(_read_fills(at))
            units = tuple(at.get("unit_indexes") or
                          (at.get("unit_index", at.get("tile_flat", at["tile"])),))
            if len(units) == len(fills):
                by_unit = {}
                for unit, fill in zip(units, fills):
                    by_unit.setdefault(unit, set()).add(fill)
                writes = tuple((unit, frozenset(sources))
                               for unit, sources in by_unit.items())
            else:
                writes = ((units[0], frozenset(fills)),)
            for unit, sources in writes:
                name = (at["tc"], at.get("group"), at["reg_buf"], unit)
                prev = regs.slots.get(name)
                if prev is not None and not prev[1] and prev[0] != sources:
                    viol.append(
                        "%s:%d OVERWRITE-BEFORE-USE  register %s held source(s) %s (never read by a "
                        "wmma) and is refilled with %s.  The rotation width cannot hold both "
                        "generations." % (phase, i, name, sorted(prev[0]), sorted(sources)))
                regs.slots[name] = (sources, False)
        elif act.kind == "wmma":
            # EVERY REGISTER SOURCE, READ OFF THE ACT (`srcs`) -- not reconstructed here.
            # `emit_plan._plan_mma` already DERIVES which operand each source is and which grid
            for op, grp, buf, unit_index, logical_index in at.get("srcs", ()):
                u = at["u"]
                need = (logical_index, u, tuple(at.get("source_issues", {}).get(op, ())))
                # find the live register of this operand whose slot matches the wmma's buffer
                cand = [n for n in regs.slots
                        if n[0] == op and n[1] == grp and n[2] == buf and n[3] == unit_index]
                if not cand:
                    viol.append(
                        "%s:%d USE-BEFORE-DEF  wmma(idx0=%s,idx1=%s,u=%s) reads %s buffer "
                        "X%s unit-index %s, which no read in scope has filled."
                        % (phase, i, at["idx0"], at["idx1"], at["u"], op, buf, unit_index))
                    continue
                for n in cand:
                    held, _used = regs.slots[n]
                    if need not in held:
                        viol.append(
                            "%s:%d WRONG SOURCE  wmma(idx0=%s,idx1=%s,u=%s) reads %s register %s, "
                            "which holds source(s) %s but the instruction needs "
                            "(tile=%s,k=%s)."
                            % (phase, i, at["idx0"], at["idx1"], at["u"], op, n,
                               sorted(held), logical_index, u))
                    regs.slots[n] = (held, True)


def prologue_blocks(prog, entry="prologue", loop="steady"):
    """The prologue REGION in program order: every block from `entry` that reaches `loop`.

    The prologue is not one block -- a peeled prefetch generation gets its own, guarded -- so the
    fill this walk models is spread over all of them.
    """
    if entry not in prog.blocks or loop not in prog.blocks:
        return [entry] if entry in prog.blocks else []
    succ = successors(prog)

    def reaches(src, want, seen=None):
        seen = set() if seen is None else seen
        for t in succ.get(src, ()):
            if t == want or (t not in seen and (seen.add(t) or reaches(t, want, seen))):
                return True
        return False

    # A later block of a multi-block loop BODY also reaches the header and is reachable from the
    # entry; what separates the peel is that the loop cannot reach back into it.
    return [blk.label for blk in prog.walk_rpo()
            if blk.label != loop and (blk.label == entry or reaches(entry, blk.label))
            and reaches(blk.label, loop) and not reaches(loop, blk.label)]


def loop_chain_blocks(prog, loop="steady"):
    """The loop body in program order: the header, then the blocks the back edge passes through."""
    succ = successors(prog)

    def reaches(src, want, seen=None):
        seen = set() if seen is None else seen
        for t in succ.get(src, ()):
            if t == want or (t not in seen and (seen.add(t) or reaches(t, want, seen))):
                return True
        return False

    chain, cur = [loop], loop
    while True:
        nxt = next((t for t in succ.get(cur, ())
                    if t not in chain and t in prog.blocks and reaches(t, loop)), None)
        if nxt is None:
            return chain
        chain.append(nxt)
        cur = nxt


def check_register_dataflow(prog, entry="prologue", loop="steady", plans=None) -> list:
    """USE-BEFORE-DEF / OVERWRITE-BEFORE-USE / WRONG-SOURCE over the pipelined shape."""
    viol, regs = [], _RegFile()
    blocks = list(prog.blocks)
    get = (lambda ph: None) if plans is None else (lambda ph: plans.get(ph))
    for lab in prologue_blocks(prog, entry, loop):
        _walk(prog, lab, regs, viol, get(lab))
    if loop not in blocks:
        return viol
    # THE LOOP BODY IS THE WHOLE CHAIN, not just the header.  Replaying only `steady` models
    # `steady -> steady`, so an unrolled copy's state is read back one copy out of phase -- which
    # for a ring that closes over several copies (HalfPLR's three halves) is every register.
    chain = loop_chain_blocks(prog, loop)
    for lab in chain:
        _walk(prog, lab, regs, viol, get(lab))
    n_after_first = len(viol)
    for lab in chain:
        _walk(prog, lab, regs, viol, get(lab))
    # A violation seen ONLY on the second pass is loop-carried; label it so the reader is not
    # hunting for it in a straight-line reading of the block.
    for j in range(n_after_first, len(viol)):
        viol[j] = "[loop-carried] " + viol[j]

    # The drain is executable code too. Walking only the rolled header let a drain read overwrite a
    # compact physical register before the header value's remaining consumers, even though every
    # steady-trip check passed. Explore each acyclic loop-exit path with an independent register
    # snapshot; joins may be visited more than once, which is harmless for a verifier.
    succ = successors(prog)
    work = [(label, regs.clone(), (chain[-1],))
            for label in succ.get(chain[-1], ()) if label not in chain]
    seen_exit_blocks = set()
    while work:
        label, state, path = work.pop()
        if label in path or label not in blocks or label in seen_exit_blocks:
            continue
        seen_exit_blocks.add(label)
        _walk(prog, label, state, viol, get(label))
        for target in succ.get(label, ()):
            if target != loop:
                work.append((target, state.clone(), path + (label,)))
    return viol


def check_region_coverage(prog, phase="steady", acts=None) -> list:
    """Every storage region of an operand is read, and no region it does not have.

    `operand_regions` is theta's own per-operand count.  Under-coverage means a region is written by
    a copy and never consumed; over-coverage means a read names a region that does not exist, which
    on the address side is a displacement into somebody else's bytes."""
    viol = []
    have = prog.meta.get("operand_regions", {}) or {}
    seen = {}
    for act in (plan_block(prog, phase) if acts is None else acts):
        if act.kind == "read":
            seen.setdefault(act.at["tc"], set()).add(act.at.get("region", 0))
    # AN WAVE-RELATIVE REGION IS COVERED BY THE AGENTS, NOT BY THE COORDINATE.
    agent_rel = set(prog.meta.get("wave_relative_regions", ()) or ())
    for op, regions in sorted(seen.items()):
        want = set(range(max(1, int(have.get(op, 1) or 1))))
        if regions != want and op in agent_rel and regions and not (regions - want):
            continue                     # under-coverage on an wave-chosen region: correct
        if regions != want:
            viol.append(
                "%s: operand %s has %d storage region(s) %s but its reads name %s%s"
                % (phase, op, len(want), sorted(want), sorted(regions),
                   "  (a region the operand does not have)" if regions - want else
                   "  (a region nothing reads)"))
    return viol


def check_source_coverage(prog, phase="steady", acts=None) -> list:
    """One steady trip reads each `(operand, tile, k, residency-pass)` EXACTLY ONCE."""
    viol = []
    in0, in1 = _wmma_inputs(prog)
    acts = plan_block(prog, phase) if acts is None else list(acts)
    want = {}
    for act in acts:
        if act.kind != "wmma":
            continue
        for which, op in ((0, in0), (1, in1)):
            if op is None:
                continue
            idx = act.at["idx0"] if which == 0 else act.at["idx1"]
            issue = tuple(act.at.get("source_issues", {}).get(op, ()))
            want.setdefault(op, set()).add((idx, act.at["u"], issue))
    got = {}
    for act in acts:
        if act.kind == "read":
            # EXACTLY ONCE is per COORDINATE, not per act.  A Phi-folded read is one act
            # delivering `q` source coordinates, so counting acts would report the `q-1`
            got.setdefault(act.at["tc"], []).extend(_read_fills(act.at))
    for op in sorted(want):
        seq = got.get(op, [])
        dups = sorted({c for c in seq if seq.count(c) > 1})
        missing = sorted(want[op] - set(seq))
        extra = sorted(set(seq) - want[op])
        if dups:
            viol.append("%s: operand %s reads source coordinate(s) %s MORE THAN ONCE per trip"
                        % (phase, op, dups))
        if missing:
            viol.append("%s: operand %s never reads source coordinate(s) %s that its wmma consume"
                        % (phase, op, missing))
        if extra:
            viol.append("%s: operand %s reads source coordinate(s) %s that no wmma consumes"
                        % (phase, op, extra))
    return viol


def check_address_keys(prog, packed, phase="steady", acts=None, force_within=False) -> list:
    """The ADDRESS-level check, which needs one geometry fact the plan does not carry.

    `check_source_coverage` proves the reads name distinct SOURCE coordinates.  That is not the
    same as distinct ADDRESSES.  The leaf builds an address from TWO coupled choices.
    
    """
    viol, keys = [], {}
    for act in (plan_block(prog, phase) if acts is None else acts):
        if act.kind != "read":
            continue
        at = act.at
        op = at["tc"]
        p = bool(packed(op))
        within = p or force_within
        key = (at.get("region", 0) if p else 0,
               at["tile"] if within else at.get("tile_flat", at["tile"]),
               at["k"] if within else at.get("k_flat", at["k"]))
        src = (at.get("tile_flat", at["tile"]), at.get("k_flat", at["k"]))
        prev = keys.setdefault((op, key), src)
        if prev != src:
            viol.append(
                "%s: operand %s reads sources %s and %s at the SAME address key %s -- one reads "
                "the other's bytes.  regions are %s, so the address is built from %s."
                % (phase, op, prev, src, key,
                   "PACKED" if p else "CONTIGUOUS",
                   "region + within-region coords" if p else "flat coords"))
    return viol


def check_refill_splits_consumers(prog, phase="steady", acts=None) -> list:
    """A HOISTED refill must not split consumers of the SAME logical value.

    A short physical ring legitimately reuses one register for two different logical tiles inside
    a body.  The refill belongs between those consumer sets; only an overlap of the logical values
    consumed on both sides means it split a generation.
    """
    order = {}
    for i, act in enumerate(plan_block(prog, phase) if acts is None else acts):
        at = act.at
        if act.kind == "read":
            units = tuple(at.get("unit_indexes") or
                          (at.get("unit_index", at.get("tile_flat", at["tile"])),))
            for unit in units:
                slot = (at["tc"], at.get("group"), at["reg_buf"], unit)
                order.setdefault(slot, []).append(
                    (i, "refill" if at.get("advance") else "fill", None))
        elif act.kind == "wmma":
            for op, grp, buf, unit_index, _logical_index in at.get("srcs", ()) or ():
                issue = tuple(at.get("source_issues", {}).get(op, ()))
                value = (_logical_index, at.get("u"), issue)
                order.setdefault((op, grp, buf, unit_index), []).append((i, "use", value))
    viol = []
    for slot, seq in sorted(order.items(), key=lambda kv: str(kv[0])):
        seq.sort()
        for index, (n, kind, _value) in enumerate(seq):
            if kind == "refill":
                before = {value for _m, k, value in seq[:index] if k == "use"}
                after = {value for _m, k, value in seq[index + 1:] if k == "use"}
                split = before & after
                if split:
                    viol.append(
                        "%s:%d REFILL-SPLITS-CONSUMERS  register %s is consumed, refilled by a "
                        "hoisted read (advance != 0), then consumes the same logical value(s) %s "
                        "AGAIN -- the second consumer reads the next generation.  Event order: %s"
                        % (phase, n, slot, sorted(split), [(m, k, value) for m, k, value in seq]))
                    break
    return viol


def _read_slot_pattern(prog, label) -> dict:
    """The register slots each operand's reads write in this block, IN ISSUE ORDER."""
    out = {}
    for act in plan_block(prog, label):
        if act.kind == "read":
            out.setdefault((act.at.get("tc"), act.at.get("group")), []).append(act.at["reg_buf"])
    return {key: tuple(seq) for key, seq in out.items()}


def check_drain_continues_rotation(prog, loop="steady") -> list:
    """A loop-exit block RESUMES the copy chain; it does not restart it.

    A drain built at copy 0 replays the slots `steady` already wrote and serves a wmma the
    previous generation.  11/11 of the PGR2 kernels that compute wrong answers, 0/66 that do not.
    """
    chain = loop_chain_blocks(prog, loop)
    if len(chain) < 2:
        return []                      # one copy: nothing rotates, nothing to resume
    want = _read_slot_pattern(prog, chain[1])
    viol = []
    for label in (lab for lab in successors(prog).get(chain[-1], ()) if lab not in chain):
        if label not in prog.blocks:
            continue
        got = _read_slot_pattern(prog, label)
        for key in sorted(set(want) & set(got), key=str):
            if got[key] != want[key]:
                wrong = {v: (got[key][v], want[key][v]) for v in sorted(set(want[key]) & set(got[key]), key=str)
                         if got[key][v] != want[key][v]}
                viol.append(
                    "%s DRAIN-RESTARTS-ROTATION  %s sends (tile,k,region)->slot %s, but resuming "
                    "the %d-copy chain after %s requires %s -- the drain replays a copy the loop "
                    "already ran."
                    % (label, key, {v: a for v, (a, _b) in wrong.items()}, len(chain), chain[0],
                       {v: b for v, (_a, b) in wrong.items()}))
    return viol


def _read_values(at) -> set:
    """Every (tile, k) a read leaves in its register, absorbed axes included."""
    fills = at.get("fills") or ()
    if fills:
        return {(fill[2], fill[4]) for fill in fills}
    return {(at.get("tile_flat", at["tile"]), at.get("k_flat", at["k"]))}


def check_unit_index_fits_its_buffer(prog) -> list:
    """A slot's registers must hold every unit indexed into it.

    The address is `slot_base + unit * frag`, so a unit span wider than the distance between two
    slot bases runs into the next slot -- and, at the last slot, past the allocation, where the
    fold lands it on slot 0.  Nothing downstream can tell that apart from a legitimate address.
    """
    layout = prog.meta.get("register_layout") or {}
    alloc = prog.meta.get("valu_alloc") or {}
    span, frag = {}, {}
    for act in (a for phase in prog.blocks for a in plan_block(prog, phase)):
        at = act.at
        if act.kind != "read" or at["tc"] not in layout:
            continue
        units = tuple(at.get("unit_indexes") or (at.get("unit_index", 0),))
        size = int(at.get("size_regs") or 0)
        if not size:
            continue
        frag[at["tc"]] = size
        span[at["tc"]] = max(span.get(at["tc"], 0), (max(units) + 1) * size)
    viol = []
    for operand, reach in sorted(span.items()):
        bases = sorted(layout[operand]["slots"].values())
        stride = min((b - a for a, b in zip(bases, bases[1:])), default=reach)
        per_buffer = int((alloc.get(operand) or {}).get("per_buffer", stride))
        total = int((alloc.get(operand) or {}).get("total", layout[operand]["total"]))
        if int(layout[operand]["total"]) > total:
            viol.append(
                "%s LAYOUT-EXCEEDS-ALLOCATION  the register layout uses %d registers per slot "
                "and %d total, but the allocation has %d per buffer and %d total."
                % (operand, stride, layout[operand]["total"], per_buffer, total))
        if reach > stride:
            viol.append(
                "%s UNIT-OVERRUNS-BUFFER  slots are %d registers apart but its units reach %d "
                "(%d units of %d) -- every slot above the first addresses into the next one."
                % (operand, stride, reach, reach // frag[operand], frag[operand]))
    return viol


def _walk_values(prog, phases, held, viol=None):
    """Thread register contents through `phases` in order, collecting stale reads into `viol`."""
    for phase in phases:
        for act in plan_block(prog, phase):
            at = act.at
            if act.kind == "read":
                values = _read_values(at)
                for unit in (at.get("unit_indexes") or (at.get("unit_index", 0),)):
                    held[(at["tc"], at.get("group"), at["reg_buf"], unit)] = values
            elif act.kind == "wmma" and viol is not None:
                for op, group, buf, unit, index in at.get("srcs", ()) or ():
                    have = held.get((op, group, buf, unit))
                    want = (index, at.get("u"))
                    if have is None or want in have:
                        continue
                    viol.append(
                        "%s WMMA-READS-STALE-VALUE  %s slot %s unit %s holds %s, but the wmma at "
                        "C[%s,%s] reads it as tile %s k %s -- a read overwrote the value before "
                        "its last consumer."
                        % (phase, op, buf, unit, sorted(have), at.get("idx0"), at.get("idx1"),
                           index, at.get("u")))
    return held


def check_wmma_reads_its_own_value(prog, loop="steady") -> list:
    """A wmma source must still hold the (tile, k) that wmma's coordinate names.

    The loop body is the whole copy chain, walked twice so the second pass sees what the previous
    iteration left: most of a body's consumers read values filled before it, and a ring too short
    for them is only visible across that edge.  `check_refill_splits_consumers` questions one
    block, and only reads carrying an `advance`, so neither case reaches it.
    """
    chain = loop_chain_blocks(prog, loop) or []
    viol = []
    if chain:
        _walk_values(prog, chain, _walk_values(prog, chain, {}))
        _walk_values(prog, chain, _walk_values(prog, chain, {}), viol)
    for phase in prog.blocks:
        if phase not in chain:
            _walk_values(prog, [phase], {}, viol)
    return viol


def check_plan(prog, plans=None) -> list:
    """Every semantic check, as one list of violations.  Empty means the plan computes the GEMM its
    theta describes -- up to the address geometry, which `lds_geometry` owns.

    `check_refill_splits_consumers` BELONGS HERE: it was written, left out of this list, and so
    never ran on the build path.  Two kernels reached hardware computing wrong answers while
    every check in this function reported clean -- the ring was big enough, the refill was just
    scheduled between two consumers of the value it overwrites.  Register DATAFLOW and register
    SCHEDULING are different questions and the first does not imply the second.
    """
    steady = None if plans is None else plans.get("steady")
    out = []
    out += check_region_coverage(prog, acts=steady)
    out += check_source_coverage(prog, acts=steady)
    out += check_register_dataflow(prog, plans=plans)
    out += check_refill_splits_consumers(prog, acts=steady)
    out += check_unit_index_fits_its_buffer(prog)
    out += check_wmma_reads_its_own_value(prog)
    out += check_drain_continues_rotation(prog)
    return out
