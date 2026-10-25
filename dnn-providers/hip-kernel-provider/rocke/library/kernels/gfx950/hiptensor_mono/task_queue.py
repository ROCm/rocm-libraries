# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Persistent heterogeneous task-queue monokernel for small / medium hipTensor ops.

EXPERIMENTAL: not reached by any dispatcher; build it explicitly.

One launch of ``P`` resident workgroups drains a device-side ticket queue that
holds every tile of every op of a plan (permutes, contractions as GEMM tiles,
elementwise, reductions). Each op runs its own *specialized* body -- the
stand-alone rocKE kernel, inlined (:func:`.inline.inline_kernel_body`) -- on
the tile its ticket names. Ops form a DAG: a ticket of op ``k`` waits until
every producer op has published all its tickets (per-op done counters), then
publishes its own ticket. Tickets are handed out in a topological order, so a
waiting workgroup only ever waits on tickets already held by running
workgroups: the dynamic queue needs no co-residency.

Ops that share one body (``body_key``; e.g. several contractions dispatched to
the same GEMM spec) are inlined ONCE: the op is decoded from the ticket and
its pointers / extents are runtime-selected scalars, so code size and LDS do
not grow with the number of ops.

Counters (one ``i32`` buffer, ``queue_counter_words(plan)`` words, zero before
the first launch): ``[0]`` ticket, ``[1]`` exit count, ``[2]`` an always-zero
slot, ``[3 + k]`` op ``k``'s published tickets, then one (ticket, arrive) pair
per dependency level. The last workgroup to exit resets every word, so the
kernel can be relaunched (or replayed in a graph) without a memset.

Cross-workgroup ordering: a producer publishes with an agent-scope release
fence, a barrier and one atomic add; a consumer spins on the counter, then
issues an agent-scope acquire fence.

Levers (:class:`QueueKnobs`, every one a knob):

* ``sched``: ``dynamic`` (atomic ticket queue, work stealing) | ``static``
  (ticket ``t = wg + i * P``; needs all ``P`` workgroups resident) |
  ``phase`` (static round-robin tickets per dependency level, one grid
  barrier between levels; needs all ``P`` workgroups resident).
* ``first_static``: dynamic queue whose first ticket is the workgroup id (one
  atomic fewer per workgroup; the queue then starts at ``P``).
* ``acquire_all_waves``: every wave issues the acquire fence after a
  dependency wait (else only the waiting thread's).
* ``slot_parity``: alternate the LDS ticket slot by iteration parity (one
  barrier per ticket instead of two).
* ``publish_terminal``: ops without consumers publish too (default: they
  publish nothing).
* ``acquire_once``: wait on / acquire a producer op once per workgroup.
* ``stamps``: an instrumented twin with per-ticket ``s_memrealtime`` stamps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from rocke.core.arch.target import ArchTarget
from rocke.core.ir import F16, I32, I64, IRBuilder, KernelDef, PtrType, Value

from .inline import inline_kernel_body

__all__ = [
    "QueueOp",
    "QueuePlan",
    "QueueKnobs",
    "TaskQueueSpec",
    "build_task_queue",
    "queue_counter_words",
    "kernel_lds_bytes",
    "CTR_TICKET",
    "CTR_EXIT",
    "CTR_ZERO",
    "CTR_OP0",
]

CTR_TICKET, CTR_EXIT, CTR_ZERO, CTR_OP0 = 0, 1, 2, 3


@dataclass
class QueueOp:
    """One op of the plan.

    ``kdef``: the stand-alone kernel (its ``max_workgroup_size`` must equal the
    plan's block size). ``grid``: its stand-alone launch grid ``(gx, gy)``
    (task-local tiles; ``block_id_x`` = tile % gx). ``bind``: kernel param ->
    ``("buf", tensor)`` | ``("i32", int)``. ``chunk``: stand-alone blocks per
    ticket. ``body_key``: ops with the same key share one inlined body (same
    ``kdef`` IR, different bindings).
    """

    name: str
    kind: str
    kdef: KernelDef
    grid: Tuple[int, int]
    bind: Mapping[str, Tuple[str, object]]
    deps: Sequence[str] = ()
    chunk: int = 1
    body_key: Optional[str] = None
    # output permute fused into the GEMM epilogue: (param, Mtot, tile_rows,
    # tile_cols, m_digits, n_digits). The body stores C[n, m] at flat index
    # n * Mtot + m; the rewrite stores it at sum(digit_i(m) * stride_i) +
    # sum(digit_j(n) * stride_j) (digits fastest first, (extent, stride)
    # pairs), through a per-ticket LDS table of the tile's m and n offsets.
    out_map: Optional[Tuple] = None

    @property
    def blocks(self) -> int:
        return self.grid[0] * self.grid[1]

    @property
    def tickets(self) -> int:
        return -(-self.blocks // self.chunk)


@dataclass
class QueuePlan:
    name: str
    ops: List[QueueOp]
    tensors: List[str]  # buffer names -> kernel params T0..Tn (f16 global)
    block_size: int
    tensor_dtypes: Dict[str, str] = field(default_factory=dict)

    def ranges(self) -> List[Tuple[int, int]]:
        out, s = [], 0
        for op in self.ops:
            out.append((s, s + op.tickets))
            s += op.tickets
        return out

    @property
    def total_tickets(self) -> int:
        return sum(op.tickets for op in self.ops)

    def validate(self) -> None:
        idx = {op.name: i for i, op in enumerate(self.ops)}
        if len(idx) != len(self.ops):
            raise ValueError("duplicate op names")
        for i, op in enumerate(self.ops):
            ws = int(op.kdef.attrs.get("max_workgroup_size", 256))
            if ws != self.block_size:
                raise ValueError(
                    f"{op.name}: body block size {ws} != plan {self.block_size}"
                )
            for d in op.deps:
                if d not in idx:
                    raise ValueError(f"{op.name}: unknown dep {d}")
                if idx[d] >= i:
                    raise ValueError(
                        f"{op.name}: dep {d} is not earlier (ticket order must be topological)"
                    )
            for p, (kind, v) in op.bind.items():
                if kind == "buf" and v not in self.tensors:
                    raise ValueError(f"{op.name}: unknown tensor {v}")
        # shared bodies: contiguous, identical body, same param set
        seen: Dict[str, int] = {}
        for i, op in enumerate(self.ops):
            if op.body_key is None:
                continue
            if op.body_key in seen and seen[op.body_key] != i - 1:
                raise ValueError(
                    f"{op.name}: ops sharing body {op.body_key} must be contiguous"
                )
            seen[op.body_key] = i


@dataclass(frozen=True)
class QueueKnobs:
    # dynamic | static | phase (phase: static round-robin tickets per
    # dependency level, one grid barrier between levels -- one release and one
    # acquire per workgroup per level instead of one per producer / consumer
    # ticket; needs all ``workgroups`` co-resident)
    sched: str = "dynamic"
    first_static: bool = False
    acquire_all_waves: bool = True
    slot_parity: bool = False
    workgroups: int = 256
    # ops without consumers publish nothing (no fence, no counter) unless set
    publish_terminal: bool = False
    # per-ticket timeline stamps (s_memrealtime, 100 MHz): i64 param ``stamp``
    # [ticket * 4 + {0: before the dependency wait, 1: after it, 2: after the
    # body, 3: workgroup id}] -- an instrumented twin, not a timed build
    stamps: bool = False
    # wait on / acquire a producer op once per workgroup (an LDS flag per op):
    # a completed op's data never changes again within the launch, so later
    # tickets of the same workgroup skip the spin and the cache invalidation
    acquire_once: bool = False

    def suffix(self) -> str:
        s = {"dynamic": "d", "static": "s", "phase": "ph"}[self.sched]
        s += "f" if self.first_static else ""
        s += "" if self.acquire_all_waves else "a1"
        s += "q" if self.slot_parity else ""
        s += "T" if self.publish_terminal else ""
        s += "S" if self.stamps else ""
        s += "O" if self.acquire_once else ""
        return f"{s}P{self.workgroups}"


def op_phases(plan: QueuePlan) -> List[int]:
    """Dependency level of every op (0 = no deps)."""
    idx = {op.name: i for i, op in enumerate(plan.ops)}
    ph: List[int] = []
    for op in plan.ops:
        ph.append(1 + max((ph[idx[d]] for d in op.deps), default=-1))
    return ph


def queue_counter_words(plan: QueuePlan) -> int:
    # ticket, exit, zero, one done counter per op, then (ticket, arrive) per phase
    return CTR_OP0 + len(plan.ops) + 2 * (max(op_phases(plan), default=0) + 1)


def _groups(plan: QueuePlan) -> List[List[int]]:
    """Consecutive op indices that share one inlined body."""
    out: List[List[int]] = []
    for i, op in enumerate(plan.ops):
        if (
            out
            and op.body_key is not None
            and plan.ops[out[-1][0]].body_key == op.body_key
        ):
            out[-1].append(i)
        else:
            out.append([i])
    return out


@dataclass(frozen=True)
class TaskQueueSpec:
    """One task-queue monokernel: the op plan (problem) and the knobs."""

    plan: QueuePlan
    knobs: QueueKnobs = field(default_factory=QueueKnobs)


# cross-workgroup memory scope of the dependency protocol's release / acquire
_SCOPE = "agent"


def build_task_queue(spec: TaskQueueSpec, *, arch: str = "gfx950") -> KernelDef:
    """KernelDef of the task-queue monokernel.

    Kernel params: ``T0 .. Tn`` (one per ``plan.tensors`` entry), ``ctr``
    (``queue_counter_words(plan)`` i32, zero before the first launch),
    [``stamp`` (i64, 4 per ticket) with ``knobs.stamps``]. Launch grid
    ``knobs.workgroups`` x ``plan.block_size`` threads.
    """
    if arch != "gfx950":
        raise ValueError(f"the task-queue monokernel is gfx950-only, got {arch}")
    kernel = _build_task_queue(spec.plan, spec.knobs)
    used = kernel_lds_bytes(kernel)
    cap = ArchTarget.from_gfx(arch).lds_capacity_bytes
    if used > cap:
        bodies = sorted(
            {op.body_key or op.name for op in spec.plan.ops if op.kind == "gemm"}
        )
        raise ValueError(
            f"task-queue kernel {kernel.name} needs {used} B of LDS, {arch} has {cap} B: every "
            f"distinct inlined body adds its LDS ({len(bodies)} GEMM bodies: {bodies}); share one body "
            "(fuse_out='all', gemm_policy='unify') or split the plan"
        )
    return kernel


def kernel_lds_bytes(kdef: KernelDef) -> int:
    """Bytes of every ``tile.smem_alloc`` of ``kdef`` (inlined bodies do not overlay)."""
    total = 0

    def walk(region) -> None:
        nonlocal total
        for op in region.ops:
            if op.name == "tile.smem_alloc":
                t = op.result.type
                n = 1
                for d in t.shape:
                    n *= int(d)
                total += n * _elem_bytes(t.elem)
            for r in op.regions:
                walk(r)

    walk(kdef.body)
    return total


def _elem_bytes(t) -> int:
    bits = getattr(t, "bits", None) or getattr(t, "bitwidth", None)
    if bits:
        return max(1, int(bits) // 8)
    return {
        "f16": 2,
        "bf16": 2,
        "i16": 2,
        "f32": 4,
        "i32": 4,
        "i64": 8,
        "f64": 8,
        "i8": 1,
    }[t.name]


def _build_task_queue(plan: QueuePlan, knobs: QueueKnobs) -> KernelDef:
    plan.validate()
    if knobs.sched not in ("dynamic", "static", "phase"):
        raise ValueError(f"sched {knobs.sched!r}")
    if knobs.sched in ("static", "phase") and (knobs.first_static or knobs.slot_parity):
        raise ValueError("first_static / slot_parity are dynamic-queue levers")
    P = int(knobs.workgroups)
    BS = plan.block_size
    ranges = plan.ranges()
    total = plan.total_tickets

    b = IRBuilder(f"rocke_htmono_{plan.name}_{knobs.suffix()}")
    b.kernel.attrs["max_workgroup_size"] = BS
    # union of the bodies' kernel attrs (refuse conflicting ones)
    for op in plan.ops:
        for k, v in op.kdef.attrs.items():
            if k == "max_workgroup_size":
                continue
            if k in b.kernel.attrs and b.kernel.attrs[k] != v:
                raise ValueError(
                    f"{op.name}: kernel attr {k}={v!r} conflicts with {b.kernel.attrs[k]!r}"
                )
            b.kernel.attrs[k] = v

    tparams: Dict[str, Value] = {}
    for i, t in enumerate(plan.tensors):
        dt = plan.tensor_dtypes.get(t, "f16")
        ty = {"f16": F16}.get(dt)
        if ty is None:
            from rocke.core.ir import dtype_to_ir_type

            ty = dtype_to_ir_type(dt)
        tparams[t] = b.param(f"T{i}", PtrType(ty, "global"), align=16)
    ctr = b.param("ctr", PtrType(I32, "global"), align=16)
    stamp_p = (
        b.param("stamp", PtrType(I64, "global"), align=16) if knobs.stamps else None
    )

    tid = b.thread_id_x()
    wg = b.block_id_x()
    c0, c1 = b.const_i32(0), b.const_i32(1)
    cP = b.const_i32(P)
    ctotal = b.const_i32(total)
    is_t0 = b.cmp_eq(tid, c0)
    slot = b.smem_alloc(I32, [4], name_hint="mono_slot")

    def fetch_atomic(slot_idx: Value, base: Optional[Value]) -> None:
        with b.scf_if(is_t0):
            t = b.global_atomic_add(ctr, b.const_i32(CTR_TICKET), c1)
            if base is not None:
                t = b.add(t, base)
            b.smem_store_vN(slot, [slot_idx], t, n=1)

    def read_slot(slot_idx: Value) -> Value:
        v = b.vec_extract(b.smem_load_vN(slot, slot_idx, dtype=I32, n=1), 0)
        return b.to_sgpr_u32(v)

    def exit_protocol() -> None:
        with b.scf_if(is_t0):
            e = b.global_atomic_add(ctr, b.const_i32(CTR_EXIT), c1)
            with b.scf_if(b.cmp_eq(e, b.const_i32(P - 1))):
                for w in range(queue_counter_words(plan)):
                    b.global_flag_store(ctr, b.const_i32(w), c0, ordering="monotonic")
        b.ret()

    nwords = queue_counter_words(plan)
    acq = None
    if knobs.acquire_once:
        acq = b.smem_alloc(I32, [nwords], name_hint="mono_acq")
        with b.scf_if(b.cmp_lt(tid, b.const_i32(nwords))):
            b.smem_store_vN(acq, [tid], c0, n=1)
    tbl = None
    if any(op.out_map is not None for op in plan.ops):
        tw = max(
            op.out_map[2] + op.out_map[3] for op in plan.ops if op.out_map is not None
        )
        tbl = b.smem_alloc(I32, [tw], name_hint="mono_otbl")

    def wait_deps(dep_slots: Sequence[Tuple[Value, Value]]) -> None:
        if not dep_slots:
            return
        if acq is not None:
            # thread 0 only: skip deps this workgroup has already acquired
            with b.scf_if(is_t0):
                newly = c0
                for cidx, tgt in dep_slots:
                    seen = b.vec_extract(b.smem_load_vN(acq, cidx, dtype=I32, n=1), 0)
                    with b.scf_if(b.cmp_eq(seen, c0)):
                        b.global_flag_wait_eq(ctr, cidx, tgt, ordering="monotonic")
                        b.smem_store_vN(acq, [cidx], c1, n=1)
                    newly = b.lor(newly, b.select(b.cmp_eq(seen, c0), c1, c0))
                with b.scf_if(b.cmp_ne(newly, c0)):
                    b.fence(scope=_SCOPE, ordering="acquire")
            b.sync()
            return
        with b.scf_if(is_t0):
            for cidx, tgt in dep_slots:
                b.global_flag_wait_eq(ctr, cidx, tgt, ordering="monotonic")
            b.fence(scope=_SCOPE, ordering="acquire")
        b.sync()
        if knobs.acquire_all_waves:
            b.fence(scope=_SCOPE, ordering="acquire")

    def publish(op_ctr: Value) -> None:
        b.fence(scope=_SCOPE, ordering="release")
        b.sync()
        with b.scf_if(is_t0):
            b.global_atomic_add(ctr, op_ctr, c1)

    consumed = {d for op in plan.ops for d in op.deps}

    def stamp(t: Value, slot_i: int) -> None:
        if stamp_p is None:
            return
        with b.scf_if(is_t0):
            if slot_i == 3:
                v = b.zext(wg, I64)
            else:
                v = b.inline_asm(
                    "s_memrealtime $0\n s_waitcnt lgkmcnt(0)", "=s", [], I64
                )
            b.global_store(
                stamp_p,
                b.add(b.mul(t, b.const_i32(4)), b.const_i32(slot_i)),
                v,
                align=8,
            )

    def run_ticket(t: Value) -> None:
        for gi, grp in enumerate(_groups(plan)):
            lo, hi = ranges[grp[0]][0], ranges[grp[-1]][1]
            inr = b.land(b.cmp_ge(t, b.const_i32(lo)), b.cmp_lt(t, b.const_i32(hi)))
            with b.scf_if(inr):
                stamp(t, 0)
                stamp(t, 3)
                _emit_group(
                    b,
                    plan,
                    grp,
                    ranges,
                    t,
                    tparams,
                    wait_deps,
                    publish,
                    gi,
                    consumed,
                    knobs.publish_terminal,
                    lambda i, t=t: stamp(t, i),
                    tbl,
                )

    if knobs.sched == "phase":
        ph = op_phases(plan)
        if ph != sorted(ph):
            raise ValueError("phase schedule needs the ops ordered by dependency level")
        nph = max(ph) + 1
        pbase = CTR_OP0 + len(plan.ops)
        for p in range(nph):
            ids = [i for i, x in enumerate(ph) if x == p]
            lo, hi = ranges[ids[0]][0], ranges[ids[-1]][1]
            trips = -(-(hi - lo) // P)
            with b.scf_for(c0, b.const_i32(trips), c1, iv_name=f"mono_ph{p}") as it:
                t = b.to_sgpr_u32(b.add(b.const_i32(lo), b.add(wg, b.mul(it, cP))))
                with b.scf_if(b.cmp_lt(t, b.const_i32(hi))):
                    for gi, grp in enumerate(_groups(plan)):
                        if ph[grp[0]] != p:
                            continue
                        glo, ghi = ranges[grp[0]][0], ranges[grp[-1]][1]
                        inr = b.land(
                            b.cmp_ge(t, b.const_i32(glo)), b.cmp_lt(t, b.const_i32(ghi))
                        )
                        with b.scf_if(inr):
                            stamp(t, 0)
                            stamp(t, 3)
                            _emit_group(
                                b,
                                plan,
                                grp,
                                ranges,
                                t,
                                tparams,
                                lambda deps: None,
                                lambda c: None,
                                gi,
                                set(),
                                False,
                                lambda i, t=t: stamp(t, i),
                                tbl,
                            )
                b.sync()
            if p + 1 < nph:
                arr = b.const_i32(pbase + 2 * p + 1)
                with b.scf_if(is_t0):
                    b.fence(scope=_SCOPE, ordering="release")
                    b.global_atomic_add(ctr, arr, c1)
                    b.global_flag_wait_eq(ctr, arr, cP, ordering="monotonic")
                    b.fence(scope=_SCOPE, ordering="acquire")
                b.sync()
        exit_protocol()
        return b.kernel

    loop_trips = total + 1 if knobs.sched == "dynamic" else -(-total // P)
    if knobs.sched == "dynamic" and knobs.first_static:
        # first ticket = wg, later tickets = P + atomic
        with b.scf_if(is_t0):
            b.smem_store_vN(slot, [c0], wg, n=1)
    elif knobs.sched == "dynamic":
        fetch_atomic(c0, None)
    b.sync()
    with b.scf_for(c0, b.const_i32(loop_trips), c1, iv_name="mono_it") as it:
        if knobs.sched == "static":
            t = b.to_sgpr_u32(b.add(wg, b.mul(it, cP)))
        else:
            cur = b.land(it, c1) if knobs.slot_parity else c0
            t = read_slot(cur)
            if not knobs.slot_parity:
                b.sync()
        with b.scf_if(b.cmp_ge(t, ctotal)):
            exit_protocol()
        if knobs.sched == "dynamic":
            nxt = b.xor(cur, c1) if knobs.slot_parity else c0
            base = cP if knobs.first_static else None
            run_ticket(t)
            fetch_atomic(nxt, base)
            b.sync()
        else:
            run_ticket(t)
            # the next ticket must not overwrite LDS (GEMM ring, epilogue
            # offset table) that slower waves of this ticket still read
            b.sync()
    if knobs.sched == "static":
        exit_protocol()
    return b.kernel


def _sel_chain(b: IRBuilder, idx: Value, vals: Sequence[Value]) -> Value:
    out = vals[0]
    for j in range(1, len(vals)):
        out = b.select(b.cmp_eq(idx, b.const_i32(j)), vals[j], out)
    return out


def _emit_group(
    b,
    plan,
    grp,
    ranges,
    t,
    tparams,
    wait_deps,
    publish,
    gi,
    consumed=None,
    publish_terminal=True,
    stamp=lambda i: None,
    tbl=None,
) -> None:
    ops = [plan.ops[i] for i in grp]
    op_index = {op.name: i for i, op in enumerate(plan.ops)}
    lead = ops[0]
    # which op of the group, and its first ticket
    if len(grp) == 1:
        j = b.const_i32(0)
        start = b.const_i32(ranges[grp[0]][0])
    else:
        j = b.const_i32(0)
        for q in range(1, len(grp)):
            j = b.add(j, b.zext(b.cmp_ge(t, b.const_i32(ranges[grp[q]][0])), I32))
        j = b.to_sgpr_u32(j)
        start = _sel_chain(b, j, [b.const_i32(ranges[i][0]) for i in grp])

    def per_op(f) -> Value:
        vals = [f(op) for op in ops]
        if len(vals) == 1:
            return vals[0]
        return _sel_chain(b, j, vals)

    local = b.sub(t, start)
    # dependency waits (pad to the group's max dep count with the zero slot)
    nd = max(len(op.deps) for op in ops)
    deps = []
    for d in range(nd):

        def _ci(op, d=d):
            if d < len(op.deps):
                return b.const_i32(3 + op_index[op.deps[d]])
            return b.const_i32(2)

        def _tg(op, d=d):
            if d < len(op.deps):
                return b.const_i32(plan.ops[op_index[op.deps[d]]].tickets)
            return b.const_i32(0)

        deps.append((per_op(_ci), per_op(_tg)))
    wait_deps(deps)
    stamp(1)

    # bindings
    params: Dict[str, Value] = {}
    for pname in lead.bind:
        kinds = {op.bind[pname][0] for op in ops}
        if len(kinds) != 1:
            raise ValueError(f"group {gi}: param {pname} bound to different kinds")
        kind = kinds.pop()
        if kind == "buf":
            params[pname] = per_op(lambda op: tparams[op.bind[pname][1]])
        elif kind == "i32":
            params[pname] = per_op(lambda op: b.const_i32(int(op.bind[pname][1])))
        else:
            raise ValueError(kind)
    gx = per_op(lambda op: b.const_i32(op.grid[0]))
    nblk = per_op(lambda op: b.const_i32(op.blocks))
    chunk = lead.chunk
    if any(op.chunk != chunk for op in ops):
        raise ValueError(f"group {gi}: ops must share chunk")

    omaps = [op.out_map for op in ops]
    if any(m is not None for m in omaps) and not all(m is not None for m in omaps):
        raise ValueError(f"group {gi}: out_map on some ops only")

    def body(blk: Value) -> None:
        bx = b.to_sgpr_u32(b.mod(blk, gx))
        by = b.to_sgpr_u32(b.div(blk, gx))
        rewrite = None
        if omaps[0] is not None:
            rewrite = {omaps[0][0]: _out_map_rewrite(b, ops, per_op, bx, by, tbl)}
        inline_kernel_body(
            b,
            lead.kdef,
            params=params,
            block_ids={"x": bx, "y": by, "z": b.const_i32(0)},
            prefix=f"g{gi}",
            store_rewrite=rewrite,
        )

    if chunk == 1:
        body(b.to_sgpr_u32(local))
    else:
        base = b.mul(local, b.const_i32(chunk))
        lim = b.smin(b.sub(nblk, base), b.const_i32(chunk))
        with b.scf_for(b.const_i32(0), lim, b.const_i32(1), iv_name=f"g{gi}_ch") as cj:
            body(b.to_sgpr_u32(b.add(base, cj)))
    stamp(2)
    pubs = (
        [op.name in consumed for op in ops]
        if consumed is not None
        else [True] * len(ops)
    )
    if publish_terminal or all(pubs):
        publish(per_op(lambda op: b.const_i32(3 + op_index[op.name])))
    elif any(pubs):
        # group mixes producer and terminal ops: publish only for the producers
        # (a wave-uniform branch on the op index)
        is_prod = per_op(lambda op: b.const_i32(1 if op.name in consumed else 0))
        with b.scf_if(b.cmp_ne(is_prod, b.const_i32(0))):
            publish(per_op(lambda op: b.const_i32(3 + op_index[op.name])))
    # else: terminal ops only -- nothing to publish (the loop's barrier after
    # the next ticket fetch orders the LDS slot)


def _magic(d: int) -> Tuple[int, int]:
    from rocke.helpers.magic_division import calculate_magic_numbers

    m, sh = calculate_magic_numbers(int(d))
    return (m - (1 << 32) if m >= (1 << 31) else m), sh


def _out_map_rewrite(b: IRBuilder, ops, per_op, bx: Value, by: Value, tbl: Value):
    """Build this tile's LDS offset tables (m offsets at [0, tc), n offsets at
    [tc, tc + tr)) and return the store-index rewrite for the GEMM epilogue."""
    from rocke.helpers.magic_division import do_magic_division_dynamic

    lead = ops[0].out_map
    _p, _M, tr, tc = lead[0], lead[1], lead[2], lead[3]
    for op in ops:
        if op.out_map[2:4] != (tr, tc) or op.out_map[0] != _p:
            raise ValueError(
                "ops sharing a GEMM body must share the out_map tile and param"
            )
    tid = b.thread_id_x()
    c0 = b.const_i32(0)
    m0 = b.to_sgpr_u32(b.mul(bx, b.const_i32(tc)))
    n0 = b.to_sgpr_u32(b.mul(by, b.const_i32(tr)))

    def digits_side(side: int, x: Value) -> Value:
        nd = max(len(op.out_map[4 + side]) for op in ops)
        h = c0
        for i in range(nd):

            def _dg(op, i=i):
                dg = op.out_map[4 + side]
                return dg[i] if i < len(dg) else (1, 0)

            ext = per_op(lambda op: b.const_i32(_dg(op)[0]))
            mul = per_op(lambda op: b.const_i32(_magic(_dg(op)[0])[0]))
            sh = per_op(lambda op: b.const_i32(_magic(_dg(op)[0])[1]))
            st = per_op(lambda op: b.const_i32(_dg(op)[1]))
            q = do_magic_division_dynamic(b, x, mul, sh)
            r = b.sub(x, b.mul(q, ext))
            h = b.add(h, b.mul(r, st))
            x = q
        return h

    # slower waves may still read the previous tile's table in their epilogue
    b.sync()
    with b.scf_if(b.cmp_lt(tid, b.const_i32(tc))):
        b.smem_store_vN(tbl, [tid], digits_side(0, b.add(m0, tid)), n=1)
    with b.scf_if(
        b.land(b.cmp_ge(tid, b.const_i32(tc)), b.cmp_lt(tid, b.const_i32(tc + tr)))
    ):
        nl = b.sub(tid, b.const_i32(tc))
        b.smem_store_vN(tbl, [tid], digits_side(1, b.add(n0, nl)), n=1)
    b.sync()
    Mt = per_op(lambda op: b.const_i32(op.out_map[1]))
    Mmul = per_op(lambda op: b.const_i32(_magic(op.out_map[1])[0]))
    Msh = per_op(lambda op: b.const_i32(_magic(op.out_map[1])[1]))
    base = b.to_sgpr_u32(b.add(b.mul(n0, Mt), m0))

    def rewrite(bb: IRBuilder, idx: Value) -> Value:
        local = bb.sub(idx, base)
        nl = do_magic_division_dynamic(bb, local, Mmul, Msh)
        ml = bb.sub(local, bb.mul(nl, Mt))
        hm = bb.vec_extract(bb.smem_load_vN(tbl, ml, dtype=I32, n=1), 0)
        hn = bb.vec_extract(
            bb.smem_load_vN(tbl, bb.add(nl, bb.const_i32(tc)), dtype=I32, n=1), 0
        )
        return bb.add(hm, hn)

    return rewrite
