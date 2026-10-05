# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Rendering -- human-readable debug views of the IR (not the IR itself).
"""

from __future__ import annotations

from . import traversal as geometry
from .emit import emit_mainloop
from .traversal import (_region_count, frag_regs, free_tiles, lds_buffers, operand_buffer_regs,
                       readahead_level, summation_tiles, resident_free_tiles)
from .ir import Bind, Branch, Cond, Inst, Load, Loop, OBLIGATION_KINDS, Peel, Space, WAR_KINDS
from .schedule import Schedule, build_S, peel_depths as _pd, prefetch_refusals
from .traversal import reload_modes, reloads_whole_set, varying_axes
from .traversal import presence, summation_names


# ===========================================================================

_VERB = {(Space.GLOBAL, Space.SHARED): "tdm",  # bulk tile mover (global->shared)
         (Space.SHARED, Space.REGISTER): "ds_read",  # shared -> registers
         (Space.GLOBAL, Space.REGISTER): "vmem_ld"}  # direct global -> registers

_SPACE = {Space.SHARED: "shared", Space.REGISTER: "vgpr", Space.GLOBAL: "mem"}


def _loop_header(node) -> str:
    if node.outer:
        pred = node.trip.render() if hasattr(node.trip, "render") else str(node.trip)
        return f"for {node.axis} while ({pred})"
    return f"for {node.axis} in range({node.trip})"


def _subbody_header(node, lo, hi) -> str:
    end = hi if hi is not None else node.trip
    if hi is not None and hi - lo == 1:
        return f"{node.axis} = {lo}"
    return f"for {node.axis} in range({lo}, {end})"


_COND_NOTE = {
    "peel_validity":      "peel-validity: enter the software-pipelined body; the predicate beside this note is "
                          "the authority -- emit.py emits a STRICT `T > M`, so the arm runs at T >= M+1)",
    "first_touch":        "first-touch: issue this read once, on the enclosing invariant mode's first pass",
    "readahead_suppress": "read-ahead suppress: drop the substep that would read the out-of-bounds next chunk",
    "short_step_validity": "short-loop step validity: this peeled step has a real chunk only if T > t",
}


def _cond_note(cond) -> str:
    kind = getattr(cond, "kind", cond) or ""
    return _COND_NOTE.get(kind, kind or "guard")


def _counter_tag(counter):
    return counter[2:] if counter.startswith("C_") else counter


def _coord_of(operand, env):
    parts = []
    for axis, value in operand.coord:
        if hasattr(value, "eval"):  # Expr (read-ahead shifted reduction coord)
            parts.append(f"{axis}{value.eval(env)}" if env is not None else f"{axis}=({value.render()})")
        elif env is not None and axis in env:
            parts.append(f"{axis}{env[axis]}")
        elif value is not None:
            parts.append(f"{axis}{value}")
        else:
            parts.append(axis)
    return ",".join(parts)


def _range_value(value, env):
    if hasattr(value, "free_vars") and env is not None and value.free_vars() <= set(env):
        return str(value.eval(env))
    return value.render() if hasattr(value, "render") else str(value)


def _range_str(axis, start, count, env):
    lo = _range_value(start, env)
    count = max(1, int(count))
    if count == 1:
        return axis if lo == "0" else f"{axis}{lo}"
    if lo.lstrip("-").isdigit():
        return f"{axis}{lo}..{int(lo) + count - 1}"
    return f"{axis}[{lo}..{lo}+{count - 1}]"


def _copy_coverage_str(operand, env):
    by_token = dict(getattr(operand, "ranges", ()) or ())
    if not by_token:
        return ""
    return "+".join(
        f"{token}[" + ",".join(_range_str(axis, start, count, env)
                                for axis, start, count in by_token.get(token, ())) + "]"
        for token in operand.tokens)


def _shape_str(operand) -> str:
    """The movement coverage and the read-ahead advance, neither of which was printed anywhere."""
    bits = []
    q = getattr(operand, "coverage", None)
    if q is not None:
        bits.append(f"coverage(carrier={q.carrier.render()}, slot={q.slot.render()})")
    adv = int(getattr(operand, "advance", 0) or 0)
    if adv:
        bits.append(f"advance={adv}")  # prefetch: hoisted this many flat positions
    return ("  " + " ".join(bits)) if bits else ""


def _load_line(operand, placement=None, env=None) -> str:
    verb = _VERB.get((operand.src, operand.dst), f"{operand.src[0]}2{operand.dst[0]}")
    fused = "+".join(operand.tokens)
    dstname = _SPACE.get(operand.dst, operand.dst)
    tag = _counter_tag(operand.counter)
    buf = placement.render() if placement else f"{dstname}[?]"
    if operand.dst == Space.SHARED:
        size = f"{operand.size_bytes}B" if operand.size_bytes else ""
        pos = _coord_of(operand, env)
        who = _copy_coverage_str(operand, env) or (f"{fused}[{pos}]" if pos else fused)
        if operand.n_parts > 1:
            who += f" /{operand.n_parts}"
        return f"{verb:8s}{who:16s} -> {buf}  {size:>8s}   # {tag}{_shape_str(operand)}"
    pos = _coord_of(operand, env)
    rng = f"{operand.size_regs}reg" if operand.size_regs else ""
    return (f"{verb:8s}{fused}[{pos}]".ljust(26)
            + f" -> {buf} {rng}   # {tag}{_shape_str(operand)}")


def _mma_line(operand, placement=None, env=None) -> str:
    sc = f"  [*{','.join(operand.scales)}]" if operand.scales else ""
    pos = _coord_of(operand, env)
    src = ""
    if isinstance(placement, dict) and placement:
        parts = []
        for name, pl in placement.items():
            reg = pl.at(env) if (env is not None and hasattr(pl, "at")) else pl
            parts.append(f"{name}={_reg_slots(reg)}")
        src = "  <- " + ", ".join(parts)
    return (f"{'wmma':8s}{operand.acc or 'C'}[{pos}] += {operand.a or '?'}*{operand.b or '?'}{sc}{src}"
            f"   # {operand.block}")


def _reg_slots(placement) -> str:
    if len(placement.slots) == 1 and not placement.slots[0][0]:
        return f"reg[buf{placement.slots[0][1].render()}]"
    return "reg[" + ",".join(f"{group},buf{e.render()}" for group, e in placement.slots) + "]"


def _await_line(await_node) -> str:
    tag = _counter_tag(await_node.counter)
    cnt = "" if await_node.count < 0 else f" (<= {await_node.count})"
    kind = await_node.kind if await_node.kind in OBLIGATION_KINDS else f"{await_node.kind}!UNKNOWN-KIND"
    order = " [issue AFTER those readers]" if await_node.kind in WAR_KINDS else ""
    return (f"{'await':8s}{await_node.dep}  on {tag}@{await_node.scope}{cnt}  <{kind}>{order}"
            f"   # {await_node.note}")


# ===========================================================================

def render_ir_raw(ir) -> str:
    """One line per node = its raw repr, walking the rolled nest pre-order."""
    lines = ["# raw rolled theta-nest IR (dataclass repr, pre-order)"]
    def walk(nodes, depth):
        for node in nodes:
            if isinstance(node, Loop):
                _o = " outer" if node.outer else ""
                _t = node.trip.render() if hasattr(node.trip, "render") else node.trip
                lines.append("  " * depth + f"Loop({node.axis} trip={_t}{_o})")
                for bi, (lo, hi, body) in enumerate(node.ranged_bodies()):
                    if len(node.bodies) > 1:
                        lines.append("  " * depth + f"  body {bi} [{lo},{hi}):")
                    walk(body, depth + 1)
            elif isinstance(node, Branch):
                lines.append("  " * depth + f"Branch({node.axis} % {node.modulus})")
                for row, branch_arm in node.arms.items():
                    lines.append("  " * depth + f"  arm {row}:")
                    walk(branch_arm, depth + 2)
            elif isinstance(node, Peel):
                lines.append("  " * depth + f"Peel({node.kind} {node.axis} k={node.k})")
                walk(node.body, depth + 1)
            elif isinstance(node, Bind):
                lines.append("  " * depth + f"Bind({node.axis} = {node.value.render()})")
                walk(node.body, depth + 1)
            elif isinstance(node, Cond):
                _lab = f" -> {node.label}" if node.label else ""
                lines.append("  " * depth + f"Cond({node.pred.render()}{_lab})  # {_cond_note(node)}")
                lines.append("  " * depth + "  then:")
                walk(node.then, depth + 2)
                if node.els:
                    lines.append("  " * depth + "  else:")
                    walk(node.els, depth + 2)
            else:
                lines.append("  " * depth + repr(node))
    walk(ir, 0)
    return "\n".join(lines)


def render_stream(ir, raw=False, unroll=False) -> str:
    """The ONE IR view both entry points call."""
    if raw:
        return render_ir_raw(ir)
    if unroll:
        return render_unrolled(ir)
    return render_ir(ir)


# ===========================================================================

def _copy_only(nodes):
    return bool(nodes) and all(isinstance(node, Inst)
                               and isinstance(node.op, Load)
                               and node.op.dst == Space.SHARED for node in nodes)


def _render_loop(lines, node, indent, depth, env):
    if not node.outer and isinstance(node.trip, int) \
            and len(node.bodies) == 1 and _copy_only(node.bodies[0]):
        for value in range(int(node.trip)):
            lines.append(f"{indent}{node.axis} = {value}:  # copy sub-body")
            _render_nodes(lines, node.bodies[0], depth + 1, {**env, node.axis: value})
        return
    if len(node.bodies) == 1:
        lines.append(f"{indent}{_loop_header(node)}:")
        _render_nodes(lines, node.bodies[0], depth + 1, env)
        return
    for lo, hi, body in node.ranged_bodies():
        lines.append(f"{indent}{_subbody_header(node, lo, hi)}:  # sub-body")
        _render_nodes(lines, body, depth + 1, env)


def _render_cond(lines, node, indent, depth, env):
    note = f"  # {_cond_note(node)}" if node.kind or node.label else ""
    lines.append(f"{indent}if {node.pred.render()}:{note}")
    _render_nodes(lines, node.then, depth + 1, env)
    if node.els:
        lines.append(f"{indent}else:  # short-loop: T is below the peel depth, so the prologue "
                     f"meets the drain")
        _render_nodes(lines, node.els, depth + 1, env)


def _render_nodes(lines, nodes, depth, env=None):
    env = {} if env is None else env
    indent = "  " * (depth + 1)
    for node in nodes:
        if isinstance(node, Loop):
            _render_loop(lines, node, indent, depth, env)
        elif isinstance(node, Branch):
            lines.append(f"{indent}rotate buf = {node.axis} % {node.modulus}:")
            for row, arm in node.arms.items():
                lines.append(f"{indent}  buf{row}:")
                _render_nodes(lines, arm, depth + 2, env)
        elif isinstance(node, Peel):
            tag = "PROLOGUE" if node.kind == "prologue" else "DRAIN"
            lines.append(f"{indent}# --- {tag} (peel {node.k} of {node.axis}) ---")
            _render_nodes(lines, node.body, depth, env)
        elif isinstance(node, Bind):
            lines.append(f"{indent}{node.axis} = {node.value.render()}:"
                         f"  # peel-step chunk binding")
            _render_nodes(lines, node.body, depth + 1, env)
        elif isinstance(node, Cond):
            _render_cond(lines, node, indent, depth, env)
        elif isinstance(node, Inst):
            for await_node in node.awaits:
                lines.append(indent + _await_line(await_node))
            line_env = env if isinstance(node.op, Load) and node.op.dst == Space.SHARED else None
            lines.append(indent + _one_line(node.op, node.placement, line_env))


def render_ir(ir) -> str:
    lines = ["# rolled theta-nest IR (the stored IR; render_stream(unroll=True) for the "
             "interleaved view)"]
    _render_nodes(lines, ir, 0)
    return "\n".join(lines)



# ===========================================================================

def _unroll_loop(lines, node, indent, env, depth):
    if node.outer:
        lines.append(f"{indent}{_loop_header(node)}:  # one iter shown")
        for body in node.bodies:
            _unroll_nodes(lines, body, {**env, node.axis: 0}, depth + 1)
        return
    for lo, hi, body in node.ranged_bodies():
        for value in range(lo, hi if hi is not None else lo + 1):
            _unroll_nodes(lines, body, {**env, node.axis: value}, depth)


def _unroll_cond(lines, node, indent, env, depth):
    taken = node.pred.eval(env)
    if taken is not None:  # the env decides it, so only that arm is reached
        _unroll_nodes(lines, node.then if taken else node.els, env, depth)
        return
    note = f"  # {_cond_note(node)}" if node.kind or node.label else ""
    lines.append(f"{indent}if {node.pred.render()}:{note}")
    _unroll_nodes(lines, node.then, env, depth + 1)
    if node.els:
        lines.append(f"{indent}else:")
        _unroll_nodes(lines, node.els, env, depth + 1)


def _unroll_inst(lines, node, indent, env):
    if node.placement is not None and hasattr(node.placement, "at"):
        placement = node.placement.at(env, keep_unresolved=True)
    else:
        placement = node.placement  # wmma: {operand: Placement} dict, or None
    for await_node in node.awaits:
        lines.append(indent + _await_line(await_node))
    lines.append(indent + _one_line(node.op, placement, env))


def _unroll_nodes(lines, nodes, env, depth):
    indent = "  " * (depth + 1)
    for node in nodes:
        if isinstance(node, Loop):
            _unroll_loop(lines, node, indent, env, depth)
        elif isinstance(node, Branch):
            if node.axis in env:
                _unroll_nodes(lines, node.arms.get(env[node.axis] % node.modulus, []), env, depth)
            else:
                for row, arm in node.arms.items():
                    _unroll_nodes(lines, arm, {**env, node.axis: row}, depth)
        elif isinstance(node, Peel):
            tag = "PROLOGUE" if node.kind == "prologue" else "DRAIN"
            lines.append(f"{indent}# --- {tag} ---")
            _unroll_nodes(lines, node.body, env, depth)
        elif isinstance(node, Bind):
            lines.append(f"{indent}# {node.axis} = {node.value.render()}")
            _unroll_nodes(lines, node.body, node.bound_env(env), depth)
        elif isinstance(node, Cond):
            _unroll_cond(lines, node, indent, env, depth)
        elif isinstance(node, Inst):
            _unroll_inst(lines, node, indent, env)


def render_unrolled(ir) -> str:
    lines = ["# UNROLLED view (interleaved read;wmma per leaf; a rendering of the rolled IR"]
    _unroll_nodes(lines, ir, {}, 0)
    return "\n".join(lines)


def _one_line(operand, placement=None, env=None) -> str:
    if isinstance(operand, Load):
        return _load_line(operand, placement, env)
    return _mma_line(operand, placement, env)


# ===========================================================================

def _geometry_operand_table(lines, depths, requested_steps, theta):
    """Append one row per operand."""
    plans = Schedule(theta, depths, requested_steps)
    lines.append("  operand role   axes               free ktiles fragregs tileB reads "
                 "shared_depth register_depths vgpr read_mode")
    for operand in theta.operands:
        frag_regs_count = geometry.frag_regs(theta, operand)
        free_tiles_count = geometry.free_tiles(theta, operand)
        tileB = geometry.operand_tile_bytes(theta, operand)
        ds_hop = operand.trajectory.shared_read
        nds = geometry.plan_transfers(theta, operand, ds_hop) if (ds_hop and ds_hop.src == Space.SHARED) else 0
        vgpr = geometry.operand_footprint_regs(theta, operand, depths)
        varying = ",".join(presence(theta, operand)) or "-"
        shared_depth = depths.shared(operand.name, default=0)
        wregs = {group: depths.get(operand.name, group) for group in operand.fragment.groups()} if not operand.is_output else {}
        wstr = ",".join(f"{group}:{w}" for group, w in wregs.items()) or "-"
        shp = "-"
        if wregs:
            parts = []
            for group in wregs:
                try:
                    dg = plans.steps(operand, group, requested_steps)
                except Exception:
                    parts.append(f"{group}:?"); continue
                parts.append(f"{group}:{'prefetch' if dg else 'inplace'}({dg})")
            shp = ",".join(parts)
        lines.append(f"  {operand.name:7s} {operand.role:6s} {varying:18s} {free_tiles_count:4d} "
                     f"{geometry.k_tiles(theta):6d} {frag_regs_count:8d} {tileB:5d} {nds:5d} "
                     f"{shared_depth:12d} {wstr:15s} {vgpr:4d} {shp}")


def _geometry_ring_table(lines, depths, requested_steps, theta):
    """Append prefetch status and placement-owned depths."""
    refusals = prefetch_refusals(theta, depths, requested_steps)
    if refusals:
        lines.append(f"  prefetch_shortfall: {len(refusals)} group(s), requested={requested_steps}")
        for refusal in refusals:
            lines.append("    %s/%s requested=%d actual=%d reason=%s: %s"
                         % (refusal.operand, refusal.group, refusal.requested, refusal.derived,
                            refusal.reason, REFUSAL_TEXT[refusal.reason] % refusal.values))
    elif requested_steps:
        lines.append(f"  prefetch: requested depth {requested_steps} is satisfied")
    lines.append(f"  register_depths: {depths}")


def _geometry_unit_table(lines, theta, depths):
    """The ord/axis units and internal per-operand PLR conversion."""
    lines.append("  operand units: op  P(original/region/group)  U(rotation/group)  "
                 "PLR(request/cap/realized)")
    for operand in theta.operands:
        if operand.is_output or not operand.movements:
            continue
        lines.append("    %-6s %d/%d/%d                    %d/%d              %d/%d/%d"
                     % (operand.name,
                        geometry.original_prefetch_unit_tiles(theta, operand),
                        geometry.prefetch_unit_tiles(theta, operand),
                        geometry.group_prefetch_unit_tiles(theta, operand),
                        geometry.rotation_unit_tiles(theta, operand),
                        geometry.group_rotation_unit_tiles(theta, operand),
                        geometry.requested_read_ahead(theta, operand),
                        geometry.prefetch_capacity(theta, operand, depths),
                        geometry.requested_read_ahead(theta, operand, depths)))


def render_geometry(theta) -> str:
    depths, _ = build_S(theta)
    shape = geometry.mma_shape(theta)
    def _lvl(axis):
        tag = "OUTER" if axis.is_outer else f"x{axis.extent}"
        return f"{axis.name}({tag})"
    lines = ["# LoopModel geometry"]
    lines.append(f"  reg_bytes={theta.reg_bytes}  lanes={theta.lanes}")
    lines.append(f"  order={' . '.join(_lvl(m) for m in theta.levels())}")
    lines.append(f"  outer={[m.name for m in theta.outer_axes()]} "
                 f"inner={[m.name for m in theta.inner_axes()]}")
    _red = set(summation_names(theta))
    lines.append(f"  reduction_axes={[m.name for m in theta.levels() if m.name in _red]}")
    offsets = theta.movement_offsets()
    if offsets:
        offs = ", ".join(f"{operand}:{role}@{lvl}={depth}"
                         for (operand, role, lvl), depth in sorted(offsets.items()))
        lines.append(f"  offsets={offs}")
    _dr_req = _pd(theta, depths).requested_steps
    lines.append(f"  mma_grid={shape.m_tiles}x{shape.n_tiles}x{shape.k_tiles} "
                 f"count={shape.count}")
    _geometry_unit_table(lines, theta, depths)
    _geometry_operand_table(lines, depths, _dr_req, theta)
    _geometry_ring_table(lines, depths, _dr_req, theta)
    lines += _region_and_agent_lines(theta)
    return "\n".join(lines)


def _region_and_agent_lines(theta) -> list:
    """Render shared-region and wave facts."""
    out = []
    waves = theta.wave_count
    if waves > 1:
        out.append(f"  wave_count={waves}")
    for operand in theta.operands:
        shared = operand.trajectory.shared
        if shared is None:
            continue
        region_axes = shared.regions.axes
        free_split = shared.regions.free_split
        if not region_axes and free_split <= 1:
            continue
        selection = "wave-relative read" if shared.regions.wave_relative else "coordinate read"
        out.append(f"  {operand.name}: regions={shared.split} axes={region_axes or ()} "
                   f"free_split={free_split} {selection}")
    return out


# ===========================================================================

def render_schedule(schedule) -> str:
    """One row per (operand, register group): the whole policy, checkable by hand."""
    rows = ["operand  grp  bufs  axis          ahead  dist  reload_after      lds  refused",
            "-------  ---  ----  ------------  -----  ----  ---------------   ---  -------"]
    for operand, group in schedule.operands():
        plan = schedule.plan(operand, group)
        reason = plan.prefetch_refused.reason if plan.prefetch_refused else "-"
        where = ",".join("%s=%s" % kv for kv in sorted(plan.reload_after.items())) or "-"
        rows.append("%-7s  %-3s  %4d  %-12s  %5d  %4d  %-15s   %3d  %s"
                    % (operand.name, group, plan.register_buffers, plan.prefetch_axis or "-",
                       plan.prefetch_steps, plan.prefetch_distance, where,
                       plan.lds_buffers, reason))
    return "\n".join(rows)




#: The detail behind each refusal reason; the derivation records the numbers, this words them.
REFUSAL_TEXT = {
    "band-empty": "W<=R=%(rate)d leaves no room for depth %(depth)d "
                  "(W=%(buffers)d, fan=%(fan)d)",
    "capacity":   "live peak %(peak)d > W=%(buffers)d at depth %(depth)d",
    "assignment": "shift=%(distance)d clobbers a live name in group(s) %(groups)s "
                  "at W=%(buffers)d (fan=%(fan)d)",
    "unknown":    "depth %(depth)d refused by the walk for no recorded reason",
}


def render_operand_facts(theta) -> str:
    """What each operand is: the intrinsic half, beside `render_schedule`'s decisions."""
    rows = ["operand  varying axes                        reload axes                  "
            "prefetch  whole  tiles f/r/res  regions  regs f/buf  lds",
            "-------  ----------------------------------  ---------------------------  "
            "--------  -----  ------------  -------  ----------  ---"]
    for operand in theta.operands:
        if getattr(operand, "is_output", False) or not getattr(operand, "fragment", None):
            continue
        axis = readahead_level(theta, operand)
        rows.append("%-7s  %-34s  %-27s  %-8s  %-5s  %2d/%2d/%2d      %7d  %4d/%-5d  %3d"
                    % (operand.name,
                       ",".join(varying_axes(theta, operand)),
                       ",".join(reload_modes(theta, operand)),
                       (axis[0] if axis else "-"),
                       reloads_whole_set(theta, operand),
                       free_tiles(theta, operand),
                       summation_tiles(theta, operand),
                       resident_free_tiles(theta, operand),
                       _region_count(theta, operand),
                       frag_regs(theta, operand),
                       operand_buffer_regs(theta, operand),
                       lds_buffers(theta, operand)))
    return "\n".join(rows)


def grouping_note(share, policies, spent, why):
    """One line of the register-grouping search: what was tried, what it cost, why it failed."""
    head = "%d-way%s" % (share, "" if policies is None else "/" + "+".join(map(str, policies)))
    if why is not None and why[0] == "derive":
        return "%s: %s" % (head, why[1])
    body = "%s: %d regs" % (head, spent)
    if why is None:
        return body
    if why[0] == "ring":
        return "%s, %s ring carries %d of %d" % ((body,) + why[1:])
    return ("%s, %s reads %d chunk(s) ahead but the copy stages %d, needs %d"
            % ((body,) + why[1:]))


def render_theta(theta, mainloop=None) -> str:
    """Render model facts and LoopIR without affecting emission."""
    try:
        result = emit_mainloop(theta) if mainloop is None else mainloop
    except Exception as exc:
        return f"# LoopModel unavailable: {exc}\n"

    lines = [
        "# LoopModel",
        f"# obligations: {len(result.obligations)}; "
        f"discharged: {result.obligations_discharged}",
    ]
    for obligation in result.undischarged:
        lines.append(
            f"# undischarged: {obligation.kind} "
            f"{obligation.producer}->{obligation.consumer} on {obligation.counter}")
    lines.extend(("", render_geometry(theta), "", render_stream(result.ir)))
    return "\n".join(lines) + "\n"
