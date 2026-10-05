# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
render_gir -- a human-readable DEBUG VIEW of a GIR Program (not the IR itself).

This is a RENDERING, not a serialization -- no field here is authoritative, and nothing in the
compiler reads it back.
"""

from __future__ import annotations

from .nodes import Move, Mma, Mark, Goto, CondGoto, CondChain, LoopBack, Return
from ...LoopModel.ir import OBLIGATION_KINDS
from .analysis import AnalysisManager
from .analyses.frame_map import FrameMap


def _coord_str(coord):
    return ",".join(f"{ax}{v}" for ax, v in coord if v is not None)


def _gen_str(ref):
    """The buffer this Ref names: a loop-carried Gen IDENTITY, an absolute generation VALUE, or a
    register rotation slot.
    """
    parts = []
    if getattr(ref, "gen", None) is not None:
        d = f"+{ref.gdelta}" if ref.gdelta else ""
        parts.append(f"gen#{ref.gen.id}[v{d}]")           # SSA identity of the phi
    if getattr(ref, "abs_gen", None) is not None:
        parts.append(f"gen@{ref.abs_gen}")                # concrete generation value
    if ref.group is not None:
        ring = getattr(ref, "reg_ring", None)
        parts.append(f"reg(g{ref.group},s{ref.slot},u{ref.unit_index}"
                     + (f"/W{ring})" if ring else ")"))
    return " ".join(parts)


def _gen_name(prog, gen):
    fact = ((prog.meta or {}).get("generation_regions", {}) or {}).get(gen.id, {})
    suffix = (f":{fact.get('operand')}/r{fact.get('region')}"
              if fact else "")
    return f"gen{gen.id}{suffix}"


def _covers_str(ref):
    """The ABSORBED axes: inner axes this ONE reference spans."""
    cov = getattr(ref, "covers", ()) or ()
    return "+".join(f"{ax}x{ext}" for ax, ext in cov)


def _ref_str(ref):
    t = ref.tile
    pos = _coord_str(t.coord)
    tag = _gen_str(ref)
    inner = f"{t.operand}.{t.space}"
    if pos:
        inner += f"{{{pos}}}"
    if tag:
        inner += f"[{tag}]"
    cov = _covers_str(ref)
    if cov:
        inner += f"<spans {cov}>"
    return inner


def _dep_str(inst):
    """The ledger obligations this instruction discharges, WITH their hazard kind."""
    deps = getattr(inst, "deps", ()) or ()
    if not deps:
        return ""
    parts = []
    for dep, counter, kind, scope in deps:
        mark = "" if kind in OBLIGATION_KINDS else " !UNKNOWN-KIND"
        # `scope` is load-bearing: a "block" dep asserts this edge needs a cross-wave fence, and
        # `verify.check_block_scope_covered` fails the build when no hazard edge backs the claim.
        sc = "" if scope == "wave" else f"/{scope}"
        parts.append(f"{kind}:{dep}@{counter}{sc}{mark}")
    return "  deps[" + ", ".join(parts) + "]"


def _relation_endpoint(endpoint, show_frame=True):
    mode = f"abs={endpoint['gdelta']}" if endpoint["absolute"] \
        else f"gdelta={endpoint['gdelta']}"
    where = endpoint.get("at", f"{endpoint.get('block')}[{endpoint.get('pos')}]")
    frame = endpoint["frame"] or "-"
    token = endpoint["token"]
    if not show_frame and not endpoint["absolute"] and endpoint["space"] == "shared":
        frame = "uniform"
        token = f"(frame+{endpoint['gdelta']})%{endpoint['ring']}"
    return (f"{where}:{endpoint['operand']}.{endpoint['space']} "
            f"frame={frame} {mode}/R{endpoint['ring']}"
            f"=>tok{token} regions={endpoint['regions']}"
            f" advance={endpoint['advance']}")


def _relation_line(relation, show_frame=True):
    return (f"{relation['kind']} {relation.get('counter', 'fence')} rank="
            f"{relation.get('rank', '-')} gap={relation['gap']} scope={relation['scope']}  "
            f"{_relation_endpoint(relation['producer'], show_frame)} -> "
            f"{_relation_endpoint(relation['consumer'], show_frame)}")


def _relation_lines(relations, show_frame=True):
    def endpoint_key(endpoint):
        ignored = {"at", "block", "pos", "identity", "register"}
        if not show_frame:
            ignored |= {"frame", "token"}
        return tuple((key, repr(value)) for key, value in endpoint.items()
                     if key not in ignored)

    grouped = {}
    for relation in relations:
        key = (relation["kind"], relation.get("counter"), relation.get("rank"),
               relation["gap"], relation["scope"],
               endpoint_key(relation["producer"]), endpoint_key(relation["consumer"]))
        grouped.setdefault(key, [relation, 0])[1] += 1
    for relation, count in grouped.values():
        suffix = (f" uniform-equivalent x{count}" if not show_frame and count > 1
                  else f" x{count}" if count > 1 else "")
        yield _relation_line(relation, show_frame) + suffix


def _uniform_access_lines(prog):
    accesses = {}
    for block in prog.blocks.values():
        for node in block.body:
            for relation in getattr(node, "at", {}).get("relations", ()) or ():
                for endpoint in (relation["producer"], relation["consumer"]):
                    key = (endpoint["operand"], endpoint["space"], endpoint["regions"],
                           endpoint["ring"], endpoint["gdelta"], endpoint["absolute"],
                           endpoint["advance"])
                    accesses[key] = endpoint
    for endpoint in accesses.values():
        if endpoint["space"] == "register":
            relation = f"slot={endpoint['token']}/W{endpoint['ring']}"
        else:
            relation = (f"abs={endpoint['gdelta']} -> tok{endpoint['token']}"
                        if endpoint["absolute"] else
                        f"gdelta={endpoint['gdelta']} -> "
                        f"tok=(frame+{endpoint['gdelta']})%{endpoint['ring']}")
        yield (f"# frame access: {endpoint['operand']}.{endpoint['space']} "
               f"regions={endpoint['regions']} {relation} R={endpoint['ring']} "
               f"advance={endpoint['advance']}")


def _move_line(mv):
    kind = "copy" if any(d.tile.space == "shared" for d in mv.dsts) else "read"
    srcs = "+".join(_ref_str(r) for r in mv.srcs)
    dsts = "+".join(_ref_str(r) for r in mv.dsts)
    return f"{kind:5s} {srcs} -> {dsts}{_dep_str(mv)}"


def _mma_line(mm):
    srcs = ",".join(_ref_str(r) for r in mm.srcs)
    dst = _ref_str(mm.dsts[0]) if mm.dsts else "acc"
    return f"mma   {dst} += {srcs}   # {mm.block}{_dep_str(mm)}"


def _pred_str(p):
    """A terminator predicate WITH its scaffold hint.  `Pred.label` is attached downstream by
    ScaffoldMapPass and is the whole reason a consumer can route an arm to the right scaffold
    region -- a dump that omits it makes every terminator look label-free and unroutable, which is
    indistinguishable from ScaffoldMapPass not having run.
    """
    return f"{p.render()!r}" + (f" [{p.label}]" if p.label else "")


def _term_line(term):
    if isinstance(term, Goto):
        return f"term: Goto({term.target})"
    if isinstance(term, LoopBack):
        lbl = f" [{term.label}]" if term.label else ""
        return (f"term: LoopBack(trips={term.trips.render()} -> {term.body}, "
                f"exit {term.exit_target}){lbl}")
    if isinstance(term, CondGoto):
        return (f"term: CondGoto({_pred_str(term.pred)} -> {term.t_target}, "
                f"else {term.f_target})")
    if isinstance(term, Return):
        return "term: Return"
    if isinstance(term, CondChain):
        arms = ", ".join(f"{_pred_str(p)} -> {t}" for p, t in term.arms)
        return f"term: CondChain({arms}, else {term.default})"
    return "term: <none>"


def _short_loop_lines(prog):
    """The `T <= M` arm: lowered to real `short{i}` blocks and then FOLDED into the drain
 chain or KEPT, per `ShortPathFold` (-223). This header reports which happened.
 """
    sl = prog.meta.get("short_loop")
    if not sl:
        return []
    guards = ", ".join((g.render() if g is not None else "unguarded")
                       for g in sl.get("guards", ()))
    out = [f"# short-loop (T <= M) arm: {sl.get('steps')} chunk steps, guards [{guards}]"]
    # PRINT THE VERDICT, do not assert one: the arm is lowered to `short{i}` blocks and the fold
    # pass decides whether they survive.
    #", which stopped being true at-223 -- the arm IS lowered to `short{i}` blocks
    verdict = sl.get("verdict")
    if verdict == "folded":
        out.append("#   FOLDED into the drain chain: %s" % sl.get("reason", "?"))
    elif verdict == "kept":
        out.append("#   KEPT as its own `short{i}` blocks ( SPLIT): %s" % sl.get("reason", "?"))
    else:
        out.append("#   verdict=%r fold=%r: %s"
                   % (verdict, sl.get("fold"), sl.get("reason", "?")))
    # The obligations are the part a reader debugging this path most needs: they are what the coverage
    # does NOT discharge and hands to the scaffold.
    folded = verdict == "folded"
    for ob in sl.get("obligations", ()) or ():
        out.append("#   OBLIGATION ON THE SCAFFOLD%s: %s"
                   % ("" if folded else " (stated for the FOLDED path; this program is %s, so read "
                                       "it as background, not as this program's obligation)"
                      % (verdict or "?"), ob))
    if not folded:
        out.append("#   OBLIGATION ON THE SCAFFOLD (this program): route the kept arm's guard "
                   "chain -- G5/ (fold_short_path.py:33)")

    # THE EDGE, READ OFF THE CFG: the guard's false edge targets the drain chain only on
    # FOLD/VACUOUS, where `fold_short_path` retargets it.
    # targets the drain chain.  That holds only on FOLD/VACUOUS, where `fold_short_path` retargets
    entry = prog.blocks.get(prog.entry)
    f_target = getattr(getattr(entry, "term", None), "f_target", None)
    out.append("#   The prologue block sits OUTSIDE the peel-validity guard; the guard's false edge "
               "targets %s." % (f_target if f_target else "(no CondGoto on the entry block)"))
    return out


def _fused_copy_fact(m, prog):
    """Which movements are fused into one cooperative instruction."""
    out = []
    fuse = m.get("fused_copy_groups") or ()
    if any(len(g) > 1 for g in fuse):
        out.append("# fused copies: " + ", ".join(
            "+".join(group) for group in fuse if len(group) > 1))
    return out


def _region_fact(m, prog):
    """Which operands are region-split, and over which axes."""
    out = []
    regions = {k: v for k, v in (m.get("operand_regions") or {}).items() if int(v or 1) > 1}
    if regions:
        rmodes, agent_rel = m.get("region_axes") or {}, set(m.get("wave_relative_regions") or ())
        for op in sorted(regions):
            axes = "/".join(rmodes.get(op) or ()) or "(no axis)"
            who = (
                f"wave-relative read reaches {regions[op]} regions"
                if op in agent_rel else "coordinate-selected read")
            out.append(f"#   {op}: {regions[op]} storage regions on {axes} -- {who}")
        units = {k: v for k, v in (m.get("unit_regions") or {}).items() if int(v or 1) > 1}
        for u in sorted(units, key=str):
            out.append(f"#   movement {'+'.join(u)}: {units[u]} regions "
                       f"(the walk is per MOVEMENT, not per operand)")
    return out


def _distributed_fact(m, prog):
    """Which operands' copies are spread across waves."""
    out = []
    dist = [k for k, v in sorted((m.get("agent_distributed") or {}).items()) if v]
    if dist:
        out.append("# wave-distributed operands: " + ", ".join(dist)
                   + "   (their cross-wave obligations are block-scoped)")
    return out


def _coverage_fact(m, prog):
    """Which axes one instruction's coverage absorbs."""
    out = []
    qa = {k: v for k, v in (m.get("coverage_axes") or {}).items() if v}
    if qa:
        out.append("# instruction coverage spans: "
                   + "; ".join(f"{k}:{v}" for k, v in sorted(qa.items(), key=str)))
    return out


def _extent_fact(m, prog):
    """The inner loop axes and their extents."""
    out = []
    ext = m.get("axis_extents") or {}
    if ext:
        out.append("# axis extents: " + ", ".join(f"{k}={v}" for k, v in sorted(ext.items()))
                   + "   summation=" + ",".join(m.get("summation_axes") or ())
                   + "   mma inputs=" + ",".join(m.get("mma_inputs") or ()))
    return out


def _completion_fact(m, prog):
    """Whether each region completes on its own."""
    out = []
    if m.get("per_region_completion"):
        out.append("# per-region completion: enabled")
    return out


def _gl2_fact(m, prog):
    """The PrefetchGL2 preset, which is a target choice, not theta."""
    out = []
    gl2 = int((prog.params or {}).get("PrefetchGL2", 0) or 0)
    if gl2:
        out.append(f"# preset PrefetchGL2={gl2} (NOT theta): one `gl2_prefetch` Mark per steady block, "
                   "placed MIDPOINT -- a bandwidth choice, not a bound")
    return out


def _theta_facts_lines(prog):
    """The theta facts the blocks are DERIVED from, which the block listing cannot show.

    One emitter per fact; each returns its lines or nothing when the fact does not apply.
    """
    m = prog.meta
    out = []
    for fact in (_fused_copy_fact,
                 _region_fact,
                 _distributed_fact,
                 _coverage_fact,
                 _extent_fact,
                 _completion_fact,
                 _gl2_fact):
        out += fact(m, prog)
    return out


def render_gir(prog) -> str:
    L = [f"Program(entry={prog.entry}, peel_depth={prog.meta.get('peel_depth')}, "
         f"buffer_depths={prog.meta.get('buffer_depths')}, "
         f"order={prog.meta.get('inner_axis_order')})"]
    L += _theta_facts_lines(prog)
    L += _short_loop_lines(prog)
    has_relations = any(getattr(node, "at", {}).get("relations")
                        for block in prog.blocks.values() for node in block.body)
    show_frames = True
    if has_relations:
        frame_map = AnalysisManager().get(FrameMap(), prog)
        divergent = tuple(block for block in prog.blocks
                          if len(frame_map.entrances(block)) > 1)
        show_frames = bool(divergent)
        L.append("# frame token rule: relative token=(frame+gdelta)%ring; "
                 "absolute generations ignore frame")
        L.append("# frame topology: " + (
            "divergent entrances in " + ",".join(divergent)
            if divergent else "uniform rotation only; no divergent entrances"))
        if not divergent:
            L.extend(_uniform_access_lines(prog))
    for blk in prog.walk_rpo():
        head = f"\nBlock {blk.phase}"
        if blk.loop:
            head += "  loop=True"
        for _f in ("gen_rel", "chunk_base", "path_chunk_base"):
            _v = getattr(blk, _f, None)
            if _v is not None:
                head += f"  {_f}={_v}"
        if getattr(blk, "model_only", False):
            head += "  MODEL-ONLY (no backend emits this block; its ops are still counted below)"
        if blk.preds:
            head += f"  preds={tuple(blk.preds)}"
        if blk.succs:
            head += f"  succs={tuple(blk.succs)}"
        L.append(head + ":")
        for phi in blk.phis:
            incoming = (
                ", ".join(
                    "%s:%s" % (pred, "forward" if value is None else value)
                    for pred, value in phi.incomings
                )
                if phi.incomings
                else "entry:%s, back" % phi.entry_val
            )
            L.append(
                f"  phi:  {_gen_name(prog, phi.gen)} = "
                f"phi({incoming}) ring={phi.gen.ring}"
            )
        for inst in blk.body:
            if isinstance(inst, Mark):
                relations = tuple(inst.at.get("relations", ()) or ())
                at = {key: value for key, value in inst.at.items() if key != "relations"}
                if relations:
                    at["frame_relations"] = len(relations)
                    at["frame_ranks"] = tuple(sorted({
                        relation["rank"] for relation in relations
                        if relation.get("rank") is not None}))
                    if inst.kind == "fence":
                        tokens = at.pop("tokens", ())
                        order_tokens = at.pop("order_tokens", ())
                        if tokens:
                            at["legacy_tokens"] = tokens
                        if order_tokens:
                            at["legacy_order_tokens"] = order_tokens
                L.append(f"  Mark  {inst.kind} {at}")
                if show_frames:
                    L.extend("    frame: " + line
                             for line in _relation_lines(relations, show_frames))
            elif isinstance(inst, Move):
                L.append("  " + _move_line(inst))
            elif isinstance(inst, Mma):
                L.append("  " + _mma_line(inst))
        for xf in blk.xfers:
            L.append(f"  xfer: {_gen_name(prog, xf.gen)}' = (v + {xf.adv}) % {xf.ring}")
        L.append("  " + _term_line(blk.term))
    return "\n".join(L)


def gir_counts(prog) -> dict:
    """Per-block op counts {block: {'read':n,'copy':n,'mma':n}} for the op-parity assertion."""
    out = {}
    for blk in prog.blocks.values():
        c = {"read": 0, "copy": 0, "mma": 0,
             "model_only": bool(getattr(blk, "model_only", False))}
        for inst in blk.body:
            if isinstance(inst, Move):
                if any(d.tile.space == "shared" for d in inst.dsts):
                    c["copy"] += 1
                else:
                    c["read"] += 1
            elif isinstance(inst, Mma):
                c["mma"] += 1
        out[blk.phase] = c
    return out
