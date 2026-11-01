# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The persistent layer-loop driver (P6's R1-R5; D0.2, D0.7, DESIGN-LB WP-L4).

Path B (D0.1) launches ``layers=1`` kernels from a graph; Path A (``pass`` /
``round``, a default-off granularity with D0.7's switch rule) and the
large-batch ``pass`` (``DESIGN-LB.md`` WP-L4) run many layers in ONE launch.
:func:`build_persistent_kernel` is that launch: a runtime loop over layers
(never unrolled, D0.8) running the stage emitters of one layer kind, built on
the framework's :class:`..framework.LayerCtx`, with P6's rules:

* R1 (``args="const"``): every stage reloads its arguments with scalar loads
  from a read-only per-layer argument table (``arg_tab[layer][k]``, i64
  addresses); ``args="lds"``: wave 0 copies the layer's row into LDS once per
  layer and stages read it from there (P6: never wins; a knob).
* R2 (``fresh="all"``), R5 (``wgid="lds"``): as in the static framework build
  (:class:`..framework.LayerCtx`).
* R3 (``loop_carry``): the loop carries only the layer index; the step counter
  is re-read per layer (the launch tag of layer ``l`` is derived from it).
* R4 (``machine_licm="off"``): kernel attribute -> ``-disable-machine-licm``.

Parity: layer ``li`` uses scratch slot ``li & 1`` (a runtime slot:
``LayerCtx.region_addr`` handles ``Value`` slots); the table must pass
``validate(layers=2)`` (the steady state of any layer count). The previous
layer's tag is ``ctx.extra["prev_tag"]`` (what a stack's first stage polls
the previous layer's output with).

ABI: ``(i64 step_ptr, i32 layer, i32 nlayers, i64 arg_tab[, i64 err])``:
layer ids ``layer .. layer + nlayers - 1``; ``arg_tab`` holds ``nlayers`` rows
of ``len(args)`` i64 addresses (``scratch`` included: every row's must be the
same buffer).

Mirrored ops only (inline asm, LDS, ``scf``); the persist attributes keep the
build on the Python engine's lowering of kernel attributes.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from rocke.core.ir import I32, I64, IRBuilder, KernelDef, Value
from rocke.helpers import device_ops as dops
from rocke.helpers import mailbox as mbx
from rocke.helpers.lds_arena import LdsEntry
from rocke.helpers.task_schedule import TaskTable, for_my_tasks

from .. import stamps as stp  # noqa: F401  (stamps are refused below)
from ..framework import (
    ATOM_LAYER_ARGS,
    FrameworkKnobs,
    LayerCtx,
    Params,
    ScratchLayout,
    StageSpec,
    _arena,
    _barrier_points,
    _load_step,
    _persist_entry,
    _persist_finish,
    _persist_pin_lds,
    _RustScope,
)
from .knobs import D0_DEFAULTS, PersistKnobs

__all__ = ["TableParams", "build_persistent_kernel", "steady_barriers", "validate_round_plan"]


class TableParams(Params):
    """Kernel arguments of the current layer, loaded from the per-layer
    argument table (R1). ``args="const"``: one scalar load per argument per
    stage (cached within the stage, dropped at the next stage);
    ``args="lds"``: from the LDS copy of the layer's row."""

    def __init__(self, b: IRBuilder, names: Sequence[str], tab: Value, mode: str, lds_addr: Value | None):
        super().__init__(b, {})
        self._names = tuple(names)
        self._tab = tab
        self._mode = mode
        self._lds = lds_addr
        self._li: Value | None = None
        self._cache: dict[str, Value] = {}

    def names(self) -> tuple[str, ...]:
        return self._names

    def ptr(self, b: IRBuilder, name: str, pointee=I32) -> Value:
        return dops.ptr_from_i64(b, self.addr(name), pointee=pointee)

    def set_layer(self, li: Value) -> None:
        self._li = li
        self._cache = {}

    def new_stage(self) -> None:
        self._cache = {}

    def addr(self, name: str) -> Value:
        if name in self._cache:
            return self._cache[name]
        b = self._b
        k = self._names.index(name)
        if self._mode == "const":
            off = b.readfirstlane(
                b.mul(b.add(b.mul(self._li, b.const_i32(len(self._names))), b.const_i32(k)), b.const_i32(8))
            )
            v = b.inline_asm(
                "s_load_dwordx2 $0, $1, $2\ns_waitcnt lgkmcnt(0)", "=s,s,s", [self._tab, off], I64,
                result_name_hint=f"arg_{name}",
            )  # fmt: skip
        else:
            from .sync import lds_read

            # the layer's copy: two parity copies (see build_persistent_kernel)
            par = b.mul(b.land(self._li, b.const_i32(1)), b.const_i32(8 * len(self._names)))
            base = b.add(b.add(self._lds, par), b.const_i32(8 * k))
            lo = b.zext(lds_read(b, base), I64)
            hi = b.zext(lds_read(b, b.add(base, b.const_i32(4))), I64)
            v = b.lor(lo, b.shl(hi, b.const_i64(32)))
        self._cache[name] = v
        return v


def steady_barriers(specs: Sequence[StageSpec]) -> set[int]:
    """Stage indices before which a CTA barrier is needed in the loop body:
    the steady-state (second layer) set of ``framework._barrier_points``,
    which also covers the wrap from the last stage to the first."""
    return {i for li, i in _barrier_points(specs, 2) if li == 1}


def build_persistent_kernel(
    name: str,
    specs: Sequence[StageSpec],
    table: TaskTable,
    layout: ScratchLayout,
    *,
    rows: int = 1,
    seq: bool = False,
    group: int = 0,
    kind: str = "",
    knobs: FrameworkKnobs | None = None,
    threads: int = 512,
    args: Sequence[str] = ATOM_LAYER_ARGS,
    pinned_lds: Sequence[LdsEntry] = (),
    entry: Callable[[IRBuilder, LayerCtx, Params], None] | None = None,
) -> KernelDef:
    """A persistent kernel running ``nlayers`` layers of ``specs`` in a
    runtime loop (module docstring). ``knobs.persist`` defaults to
    :data:`.knobs.D0_DEFAULTS` (R1-R5 on)."""
    knobs = knobs or FrameworkKnobs(persist=D0_DEFAULTS)
    if knobs.persist is None:
        from dataclasses import replace as _rep

        knobs = _rep(knobs, persist=D0_DEFAULTS)
    ps: PersistKnobs = knobs.persist
    ps.check()
    if ps.args not in ("const", "lds"):
        raise ValueError("the loop driver takes its arguments from the per-layer table: args 'const' or 'lds'")
    if knobs.stamps:
        raise NotImplementedError("stage stamps in the loop driver are not built (use a static build)")
    if knobs.numa_flip_arg:
        raise NotImplementedError("numa_flip_arg reads a kernel argument; the loop driver's arguments are per layer")
    order = table.order
    if tuple(s.name for s in specs) != order:
        raise ValueError(f"specs {[s.name for s in specs]} != table order {order}")
    if layout.parity_slots != 2:
        raise ValueError("the loop driver double-buffers scratch: layout.parity_slots must be 2")
    table.validate(layers=2)
    if "scratch" not in args:
        raise ValueError("args must include 'scratch'")
    sfx = knobs.suffix()
    b = IRBuilder(name + "_loop" + (f"_{sfx}" if sfx else ""))
    b.kernel.attrs["max_workgroup_size"] = threads
    if knobs.waves_per_eu:
        b.kernel.attrs["waves_per_eu"] = knobs.waves_per_eu
    if ps.machine_licm == "off":
        b.kernel.attrs["machine_licm"] = "off"
    if knobs.edges is not None:
        from ..framework import _edge_names

        knobs.edges.validate(_edge_names(table, layout))
    step_ptr = b.param("step_ptr", I64)
    layer0 = b.readfirstlane(b.param("layer", I32))
    nlayers = b.readfirstlane(b.param("nlayers", I32))
    tab = b.param("arg_tab", I64)
    vals = {"arg_tab": tab}
    if knobs.err_arg:
        vals[knobs.err_arg] = b.param(knobs.err_arg, I64)
    nargs = len(args)
    lds_args = None
    pin_args = None
    if ps.args == "lds":
        # two parity copies of the layer's row: wave 0 writes layer li's copy
        # while slow waves may still read layer li - 1's (one barrier per layer
        # then suffices: no wave is two layers behind another)
        w = b.smem_alloc(I32, [4 * nargs], name_hint="args")
        lds_args = b.trunc(b.smem_addr_of(w), I32)
        pin_args = w
    p = TableParams(b, args, tab, ps.args, lds_args)
    arena = _arena(specs, rows, knobs, pinned_lds)
    lds = arena.emit(b)
    ctx = LayerCtx(
        b, table=table, scratch_layout=layout, knobs=knobs, lds=lds,
        scratch=None, rows=rows, seq=seq, group=group, kind=kind, stamps=None,
    )  # fmt: skip
    ctx.extra["arena"] = arena
    ctx.params = p
    ctx.step_ptr = step_ptr
    _persist_entry(b, ctx, knobs, vals)
    if entry is not None:
        raise NotImplementedError("entry hooks in the loop driver are not built")
    step0 = None if ps.loop_carry else _load_step(b, step_ptr, knobs)
    barriers = steady_barriers(specs) if knobs.phase_barriers else set()
    with b.scf_for(b.const_i32(0), nlayers, b.const_i32(1), iv_name="pl") as li:
        p.set_layer(li)
        if ps.args == "lds":
            # wave 0 copies this layer's argument row into LDS (2 words per arg)
            with b.scf_if(dops.is_wave0(b)):
                lane = b.lane_id()
                with b.scf_if(b.cmp_lt(lane, b.const_i32(2 * nargs))):
                    from .sync import lds_write

                    word = b.add(b.mul(li, b.const_i32(2 * nargs)), lane)
                    v = b.global_load_i32(dops.ptr_from_i64(b, tab), word)
                    par = b.mul(b.land(li, b.const_i32(1)), b.const_i32(8 * nargs))
                    lds_write(b, b.add(b.add(lds_args, par), b.mul(lane, b.const_i32(4))), v)
            b.s_barrier_bare(fence="compiler")
        # R3: everything but the index is re-derived per layer
        step = _load_step(b, step_ptr, knobs) if ps.loop_carry else step0
        ctx.li = 0
        ctx.slot = b.land(li, b.const_i32(1))
        ctx.layer = b.add(layer0, li)
        ctx.tag = mbx.launch_tag(b, step, ctx.layer)
        ctx.tags = [ctx.tag]
        ctx.extra["prev_tag"] = mbx.launch_tag(b, step, b.sub(ctx.layer, b.const_i32(1)))
        ctx.scratch = p.addr("scratch")
        for i, s in enumerate(specs):
            if i in barriers:
                ctx.barrier()
            p.new_stage()
            ctx._enter_stage(i, s.name)
            if s.prologue is not None:
                s.prologue(b, ctx, p)
            emit = s.emit

            def body(t: Value, emit=emit) -> None:
                emit(b, ctx, p, t)

            fresh = (lambda v: dops.fresh_s(b, v)) if knobs.fresh_ids else None
            with _RustScope(b, knobs, s.name):
                if s.per_cta:
                    if s.stage.tasks > table.grid:
                        raise ValueError(f"per_cta stage {s.name}: {s.stage.tasks} tasks > grid")
                    from rocke.helpers.task_schedule import task_index

                    body(task_index(b, table, s.name, ctx.bid, 0))
                else:
                    for_my_tasks(b, table, s.name, ctx.bid, body, fresh=fresh)
    if pin_args is not None:
        ctx.extra.setdefault("pinned_smem", []).append(pin_args)
    _persist_pin_lds(b, ctx)
    b.ret()
    _persist_finish(b, ctx, knobs)
    return b.kernel


def validate_round_plan(kinds: Sequence[str], *, parity: Sequence[str] | None = None, strict: bool = True) -> list[str]:
    """``TaskTable.validate_stack`` over a whole layer sequence of ATOM's
    schedules (``atom_layout.atom_stack``): the cross-layer write-after-read
    check of a persistent launch running ``kinds`` in order (a pass: the
    model's layer pattern; a round: the pass plus the drafter's layers).
    ``parity``: the double-buffered regions (default: every mailbox region,
    the loop driver's layout). Returns the problems (raises with ``strict``)."""
    from rocke.helpers.task_schedule import validate_stack

    from .. import atom_layout as AL

    par = tuple(AL.REGIONS) if parity is None else tuple(parity)
    return validate_stack(AL.atom_stack(list(kinds), parity=par), strict=strict)
