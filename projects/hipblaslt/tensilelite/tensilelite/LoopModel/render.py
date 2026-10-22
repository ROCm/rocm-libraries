# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Human-readable LoopIR and target-model diagnostics."""

from __future__ import annotations

from . import traversal as geometry
from .schedule import Schedule, build_S, prefetch_refusals
from .traversal import (
    _region_count,
    frag_regs,
    free_tiles,
    lds_buffers,
    operand_buffer_regs,
    presence,
    readahead_level,
    reload_modes,
    reloads_whole_set,
    resident_free_tiles,
    summation_names,
    summation_tiles,
    varying_axes,
)


def _geometry_unit_table(lines, theta, depths):
    lines.append(
        "  operand units: op  P(original/region/group)  "
        "U(rotation/group)  PLR(request/cap/realized)"
    )
    for operand in theta.operands:
        if operand.is_output or not operand.movements:
            continue
        lines.append(
            "    %-6s %d/%d/%d                    %d/%d              %d/%d/%d"
            % (
                operand.name,
                geometry.original_prefetch_unit_tiles(theta, operand),
                geometry.prefetch_unit_tiles(theta, operand),
                geometry.group_prefetch_unit_tiles(theta, operand),
                geometry.rotation_unit_tiles(theta, operand),
                geometry.group_rotation_unit_tiles(theta, operand),
                geometry.requested_read_ahead(theta, operand),
                geometry.prefetch_capacity(theta, operand, depths),
                geometry.requested_read_ahead(theta, operand, depths),
            )
        )


def _geometry_operand_table(lines, depths, requested_steps, theta, allocated=None):
    plans = Schedule(theta, depths)
    lines.append(
        "  operand role   axes               free ktiles fragregs tileB reads "
        "shared_depth register_depths vgpr read_mode"
    )
    for operand in theta.operands:
        fragment = operand.fragment
        fragment_registers = geometry.frag_regs(theta, operand)
        tile_bytes = geometry.operand_tile_bytes(theta, operand)
        read = operand.trajectory.shared_read
        reads = geometry.plan_transfers(theta, operand, read) if read else 0
        # The footprint is what theta's own ring would occupy.  When the writer
        # has sized this operand, that allocation is the register count the
        # kernel really has, and the one a reader of the dump needs.
        vgpr = (allocated or {}).get(operand.name)
        if vgpr is None:
            vgpr = geometry.operand_footprint_regs(theta, operand, depths)
        axes = ",".join(presence(theta, operand)) or "-"
        shared_depth = depths.shared(operand.name, default=0)
        register_depths = (
            {
                group: depths.get(operand.name, group)
                for group in fragment.groups()
            }
            if fragment is not None and not operand.is_output else {}
        )
        depth_text = ",".join(
            f"{group}:{depth}"
            for group, depth in register_depths.items()
        ) or "-"
        modes = []
        for group in register_depths:
            try:
                distance = plans.steps(operand, group, requested_steps)
                modes.append(
                    f"{group}:{'prefetch' if distance else 'inplace'}({distance})"
                )
            except Exception:
                modes.append(f"{group}:?")
        lines.append(
            f"  {operand.name:7s} {operand.role:6s} {axes:18s} "
            f"{geometry.free_tiles(theta, operand):4d} "
            f"{geometry.k_tiles(theta):6d} {fragment_registers:8d} "
            f"{tile_bytes:5d} {reads:5d} {shared_depth:12d} "
            f"{depth_text:15s} {vgpr:4d} {','.join(modes) or '-'}"
        )


REFUSAL_TEXT = {
    "band-empty": "W<=R=%(rate)d leaves no room for depth %(depth)d "
                  "(W=%(buffers)d, fan=%(fan)d)",
    "capacity": "live peak %(peak)d > W=%(buffers)d at depth %(depth)d",
    "assignment": "shift=%(distance)d clobbers a live name in group(s) "
                  "%(groups)s at W=%(buffers)d (fan=%(fan)d)",
    "unknown": "depth %(depth)d refused by the walk for no recorded reason",
}


def _geometry_ring_table(lines, depths, requested_steps, theta):
    refusals = prefetch_refusals(theta, depths, requested_steps)
    if refusals:
        lines.append(
            f"  prefetch_shortfall: {len(refusals)} group(s), "
            f"requested={requested_steps}"
        )
        for refusal in refusals:
            lines.append(
                "    %s/%s requested=%d actual=%d reason=%s: %s"
                % (
                    refusal.operand,
                    refusal.group,
                    refusal.requested,
                    refusal.derived,
                    refusal.reason,
                    REFUSAL_TEXT[refusal.reason] % refusal.values,
                )
            )
    elif requested_steps:
        lines.append(f"  prefetch: requested depth {requested_steps} is satisfied")
    lines.append(f"  register_depths: {depths}")


def _region_and_agent_lines(theta):
    lines = []
    if theta.wave_count > 1:
        lines.append(f"  wave_count={theta.wave_count}")
    for operand in theta.operands:
        shared = operand.trajectory.shared
        if shared is None:
            continue
        axes = shared.regions.axes
        split = shared.regions.free_split
        if not axes and split <= 1:
            continue
        selection = (
            "wave-relative read"
            if shared.regions.wave_relative
            else "coordinate read"
        )
        lines.append(
            f"  {operand.name}: regions={shared.split} axes={axes or ()} "
            f"free_split={split} {selection}"
        )
    return lines


def _valu_alloc_table(lines, valu):
    """The writer's PU * VA / VG statement, one row per operand.

    Theta's own footprint answers a different question -- what its ring would
    occupy -- so the terms the kernel was actually sized by are spelled out
    rather than left to be inferred from the product.
    """
    lines.append(
        "  valu_alloc (PU * VA / VG):  op   pu_tiles pu_regs  vg   va  buffer  total"
    )
    for operand, alloc in sorted(valu.items()):
        lines.append(
            f"  {'':26s}{operand:5s}{alloc.puTiles:9d}{alloc.puRegs:8d}"
            f"{alloc.vg:4d}{alloc.va:5d}{alloc.perBuffer:8d}{alloc.total:7d}"
        )


def render_geometry(theta, valu=None) -> str:
    """The derived sizing tables for `theta`.

    `valu` is the writer's Valu allocation, `{operand: ValuAlloc}`.  Without it
    the `vgpr` column falls back to theta's own footprint, which is what a
    standalone theta has and not what a kernel allocates.
    """
    depths, _minimum = build_S(theta)
    shape = geometry.mma_shape(theta)

    def level(axis):
        extent = axis.extent.constant_value
        return f"{axis.name}({'OUTER' if extent == 0 else f'x{extent}'})"

    lines = [
        "# LoopModel geometry",
        f"  reg_bytes={theta.reg_bytes}  lanes={theta.lanes}",
        f"  order={' . '.join(level(axis) for axis in theta.levels())}",
        f"  outer={[axis.name for axis in theta.outer_axes()]} "
        f"inner={[axis.name for axis in theta.inner_axes()]}",
    ]
    reduction = set(summation_names(theta))
    lines.append(
        f"  reduction_axes="
        f"{[axis.name for axis in theta.levels() if axis.name in reduction]}"
    )
    offsets = theta.movement_offsets()
    if offsets:
        rows = sorted(
            offsets.items(),
            key=lambda item: (
                item[0][0],
                item[0][1].__name__,
                item[0][2],
            ),
        )
        lines.append(
            "  offsets="
            + ", ".join(
                f"{operand}:{operation.__name__}@{axis}={depth}"
                for (operand, operation, axis), depth in rows
            )
        )
    requested = Schedule(theta, depths).read_ahead_steps
    lines.append(
        f"  mma_grid={shape.m_tiles}x{shape.n_tiles}x{shape.k_tiles} "
        f"count={shape.count}"
    )
    _geometry_unit_table(lines, theta, depths)
    allocated = {op: alloc.total for op, alloc in (valu or {}).items()}
    _geometry_operand_table(lines, depths, requested, theta, allocated)
    if valu:
        _valu_alloc_table(lines, valu)
    _geometry_ring_table(lines, depths, requested, theta)
    lines.extend(_region_and_agent_lines(theta))
    return "\n".join(lines)


def render_schedule(schedule) -> str:
    rows = [
        "operand  grp  bufs  axis          ahead  dist  reload_after      lds  refused",
        "-------  ---  ----  ------------  -----  ----  ---------------   ---  -------",
    ]
    for operand, group in schedule.operands():
        plan = schedule.plan(operand, group)
        reason = plan.prefetch_refused.reason if plan.prefetch_refused else "-"
        where = ",".join(
            "%s=%s" % item
            for item in sorted(plan.reload_after.items())
        ) or "-"
        rows.append(
            "%-7s  %-3s  %4d  %-12s  %5d  %4d  %-15s   %3d  %s"
            % (
                operand.name,
                group,
                plan.register_buffers,
                plan.prefetch_axis or "-",
                plan.prefetch_steps,
                plan.prefetch_distance,
                where,
                plan.lds_buffers,
                reason,
            )
        )
    return "\n".join(rows)


def render_operand_facts(theta) -> str:
    rows = [
        "operand  varying axes                        reload axes                  "
        "prefetch  whole  tiles f/r/res  regions  regs f/buf  lds",
        "-------  ----------------------------------  ---------------------------  "
        "--------  -----  ------------  -------  ----------  ---",
    ]
    for operand in theta.operands:
        if operand.is_output or operand.fragment is None:
            continue
        axis = readahead_level(theta)
        rows.append(
            "%-7s  %-34s  %-27s  %-8s  %-5s  %2d/%2d/%2d      "
            "%7d  %4d/%-5d  %3d"
            % (
                operand.name,
                ",".join(varying_axes(theta, operand)),
                ",".join(reload_modes(theta, operand)),
                axis[0] if axis else "-",
                reloads_whole_set(theta, operand),
                free_tiles(theta, operand),
                summation_tiles(theta, operand),
                resident_free_tiles(theta, operand),
                _region_count(theta, operand),
                frag_regs(theta, operand),
                operand_buffer_regs(theta, operand),
                lds_buffers(theta, operand),
            )
        )
    return "\n".join(rows)


__all__ = [
    "render_geometry",
    "render_operand_facts",
    "render_schedule",
]
