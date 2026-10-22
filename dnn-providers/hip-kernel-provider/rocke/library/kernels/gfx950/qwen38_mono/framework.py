# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The rocKE persistent layer-kernel framework (Qwen3.8 roadmap T9).

A layer kernel is a resident grid (``grid`` CTAs x ``threads`` threads, one
CTA per CU) running a static :class:`rocke.helpers.task_schedule.TaskTable`:
every CTA walks the stages in order and runs the tasks it holds; stages hand
values over through tagged mailboxes (:mod:`rocke.helpers.mailbox`) in a
scratch buffer, and share one LDS arena with phase overlays
(:mod:`rocke.helpers.lds_arena`). This module composes *stage emitters* into
one ``KernelDef``.

Stage emitter interface (what a stage track writes; it plugs in unchanged):

``StageSpec(stage=Stage(...), emit=fn, lds=(LdsEntry, ...), prologue=None)``
where ``fn(b, ctx, p, t)`` emits ONE task ``t`` (uniform i32) of the stage:

* ``b``: the ``IRBuilder``;
* ``ctx``: :class:`LayerCtx`: ``tid``, ``lane``, ``wave`` (uniform, a fresh
  copy per stage), ``bid`` (uniform), ``tag`` (the launch tag of the layer
  being run; ``tags[li]`` every earlier layer's in a multi-layer launch),
  ``layer`` (runtime layer id, i32), ``li`` (static layer index
  inside the launch: 0 except in a multi-layer launch), ``slot`` (static
  parity slot), ``rows``, ``seq``, ``group``, ``kind``, ``knobs``
  (:class:`FrameworkKnobs`), ``lds`` (the :class:`LdsView`), ``scratch``
  (i64), and the mailbox helpers :meth:`LayerCtx.mb`, :meth:`LayerCtx.poll`,
  :meth:`LayerCtx.put`, :meth:`LayerCtx.put2`, :meth:`LayerCtx.region_addr`;
* ``p``: :class:`Params`: every kernel argument as a uniform i64 address
  (``p.addr(name)``) or a global pointer (``p.ptr(b, name)``);
* every thread of the CTA calls ``fn`` (the framework guards the call with a
  wave-uniform ``scf_if`` on "this CTA holds task ``t``"); a stage that needs
  only some waves branches on ``ctx.wave`` itself.

Rules an emitter follows (``rocke.helpers.mailbox`` module docstring and
roadmap section 3.4): issue weight loads before the poll; never ``b.sync()``
(its ``vmcnt`` drain kills in-flight prefetch): CTA barriers are
``s_barrier_bare(fence="compiler")`` / ``sync_lds_only``; polls go through
:meth:`LayerCtx.poll` (stage stamps) inside wave-uniform branches.

The same emitter serves the hybrid path: :func:`stage_kernel` wraps one
stage into a standalone ``KernelDef`` with the same ABI, which
``device_function.lower_device_function`` turns into a device function the
FlyDSL mono kernel links.

ABI (A/B-swappable with ATOM's ``build_layer`` launchers): params
``(i64 step_ptr, i32 layer, args...)`` with ``args`` defaulting to ATOM's 41
``LAYER_ARGS`` as i64 addresses, so kernarg offset of arg ``k`` is
``16 + 8 k`` (:func:`kernarg_offset`).

Everything here emits mirrored ops only (inline asm, ``readfirstlane``,
``scf_if``, LDS ops, ``s_barrier_bare``): a build that uses only mirrored
stage emitters keeps cpp parity.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace

from rocke.core.ir import I32, I64, IRBuilder, KernelDef, Value
from rocke.helpers import device_ops as dops
from rocke.helpers import mailbox as mbx
from rocke.helpers.lds_arena import (
    ONE_CTA_PER_CU_WORDS,
    LdsArena,
    LdsEntry,
    LdsView,
)
from rocke.helpers.task_schedule import Stage, TaskTable, for_my_tasks

from . import stamps as stp

__all__ = [
    "ATOM_LAYER_ARGS",
    "LDS_BARRIERS",
    "FrameworkKnobs",
    "LayerCtx",
    "LayerShape",
    "Params",
    "ScratchLayout",
    "StageSpec",
    "build_layer",
    "build_layer_kernel",
    "kernarg_offset",
    "register_stage",
    "registered_stages",
    "stage_kernel",
]

# ATOM ``qwen4_exp_mono/kernels/kernel.py`` LAYER_ARGS (contract-tested
# against the ATOM tree in library/tests/test_qwen38_layer_framework.py).
ATOM_LAYER_ARGS = (
    "scratch", "ple", "h_in", "hc_norm_w", "w_down", "s_down", "w_up", "s_up",
    "w_inj", "w_qkvz", "s_qkvz", "conv_w", "conv_in", "conv_out", "ba_w",
    "a_log", "dt_bias", "norm_w", "rec_in", "rec_out", "w_qsa", "s_qsa", "qsa",
    "w_out", "s_out", "m_norm_w", "m_w_down", "m_s_down", "m_w_up", "m_s_up",
    "m_w_inj", "r_w", "e_w_gu", "e_s_gu", "sh_w_gu", "sh_s_gu", "e_w_dn",
    "e_s_dn", "sh_w_dn", "sh_s_dn", "h_out",
)  # fmt: skip

LDS_BARRIERS = ("bare", "lds_only")


def kernarg_offset(name: str, args: Sequence[str] = ATOM_LAYER_ARGS) -> int:
    """Byte offset of argument ``name`` in the kernarg segment: after the i64
    step pointer (0) and the i32 layer (8, padded to 16)."""
    return 16 + 8 * tuple(args).index(name)


@dataclass(frozen=True)
class FrameworkKnobs:
    """Framework build knobs; every default is the shipped behaviour and the
    kernel name names only deviations (:meth:`suffix`).

    mailbox: :class:`rocke.helpers.mailbox.MailboxKnobs` of every hand-off.
    stamps: emit the per-CTA stage stamps (ATOM ``ATOM_Q38_STAMPS`` format).
    fresh_ids: each stage takes fresh SGPR copies of ``wave`` / ``bid``
      (ATOM ``_fresh``): index math is recomputed per stage instead of held.
      ``False`` is unsafe on hop-only stages: the toy chain at ``work=0``,
      depth >= 3 miscompiles (llvm22: the asm poll's SGPR base lands in VGPRs,
      an assembler error; llvm23 (ROCm 10 comgr): a null-address fault on
      gfx950). Refreshing either ``wave`` or ``bid`` alone avoids it.
    lds_barrier: barrier at an LDS phase change: ``bare``
      (``s_barrier_bare(fence="compiler")``) | ``lds_only`` (``sync_lds_only``:
      also waits ``lgkmcnt(0)``).
    head_sleep: ``s_sleep`` units (64 clocks) CTAs without a first-stage task
      sleep at entry (ATOM ``HEAD_SLEEP`` = 64 for builds starting at A1).
    lds_min_words: arena padding; the default keeps one CTA per CU.
    waves_per_eu: ``amdgpu-waves-per-eu`` = (n, n) (0: the compiler's
      default). 1 or 2 lets a 512-thread CTA use up to 512 / 256 VGPRs.
    kernarg_preload: mark the first N kernel parameters (``step_ptr``,
      ``layer``, then the arguments in order) ``inreg``: the gfx950
      backend preloads them into user SGPRs at dispatch, so the kernel does
      not wait on their ``s_load`` (LLVM preloads at most 16 SGPRs: step_ptr,
      layer + padding and six i64 arguments, N <= 8). Python engine only
      (``ir.PYTHON_ONLY_PARAM_ATTRS``; ``compile_for`` picks it).
    step_load: how the step counter is read at the kernel entry: ``vmem``
      (a global load; the launch tag waits ``vmcnt``) | ``smem`` (a scalar
      ``s_load_dword ... glc`` and ``lgkmcnt(0)`` in one asm: the scalar
      path, and the wait never covers vector (weight) loads).
    stamps_lds: where the stamps' LDS words live: ``arena`` (a pinned
      arena entry: the arena stays allocated to the kernel end) | ``own``
      (their own small allocation: the arena's live range ends at its last
      stage use, so later stages' own pools can overlay it).
    lds_touch: one LDS store at the kernel entry to the arena's last word,
      so a padded arena (``lds_min_words``) stays allocated even when no
      stage uses it (residency padding for builds whose stages allocate
      their own LDS, ``stage_adapt.resident_build``).
    step_first: issue the step-counter load before the ``entry`` hook's
      loads (default: after them). Loads retire in order, so with the entry
      first the launch tag's wait on the step is ``vmcnt(0)`` and also waits
      for every entry (weight) load; step first, it waits only for the step.
    phase_barriers: insert a CTA barrier at every LDS phase change (off:
      the stages barrier themselves before their first LDS write, as ATOM's
      do through ``payload_wait``).
    numa_flip_arg: with ``mailbox.numa_copies == 2``, the kernel argument
      holding the host-calibrated i32 flip of every region
      (``ScratchLayout.region_index`` order; see
      ``persistent_mailbox.harness.calibrate_iod_home``); empty = flip 0
      (uncalibrated).
    """

    mailbox: mbx.MailboxKnobs = field(default_factory=mbx.MailboxKnobs)
    stamps: bool = False
    fresh_ids: bool = True
    lds_barrier: str = "bare"
    head_sleep: int = 0
    lds_min_words: int = ONE_CTA_PER_CU_WORDS
    numa_flip_arg: str = ""
    phase_barriers: bool = True
    waves_per_eu: int = 0
    step_first: bool = False
    stamps_lds: str = "arena"
    step_load: str = "vmem"
    kernarg_preload: int = 0
    lds_touch: bool = False

    def __post_init__(self) -> None:
        if self.lds_barrier not in LDS_BARRIERS:
            raise ValueError(f"lds_barrier must be one of {LDS_BARRIERS}")
        if not 0 <= self.kernarg_preload <= 8:
            raise ValueError("kernarg_preload must be in 0..8 (16 user SGPRs)")
        if self.step_load not in ("vmem", "smem"):
            raise ValueError("step_load must be 'vmem' or 'smem'")
        if self.stamps_lds not in ("arena", "own"):
            raise ValueError("stamps_lds must be 'arena' or 'own'")
        if self.head_sleep < 0:
            raise ValueError("head_sleep must be >= 0")

    def suffix(self) -> str:
        d = FrameworkKnobs()
        parts = []
        s = self.mailbox.suffix()
        if s:
            parts.append(s)
        if self.stamps:
            parts.append("tl")
        if not self.fresh_ids:
            parts.append("nofresh")
        if self.lds_barrier != d.lds_barrier:
            parts.append(f"lb{self.lds_barrier}")
        if self.head_sleep != d.head_sleep:
            parts.append(f"hs{self.head_sleep}")
        if self.lds_min_words != d.lds_min_words:
            parts.append(f"lmin{self.lds_min_words}")
        if self.numa_flip_arg:
            parts.append("cal")
        if not self.phase_barriers:
            parts.append("nopb")
        if self.waves_per_eu:
            parts.append(f"wpe{self.waves_per_eu}")
        if self.step_first:
            parts.append("sf")
        if self.stamps and self.stamps_lds != "arena":
            parts.append(f"tl{self.stamps_lds}")
        if self.lds_touch:
            parts.append("lt")
        if self.kernarg_preload:
            parts.append(f"kp{self.kernarg_preload}")
        if self.step_load != "vmem":
            parts.append(f"st{self.step_load}")
        return "_".join(parts)


@dataclass(frozen=True)
class ScratchLayout:
    """Mailbox regions of the scratch buffer (offsets in bytes).

    ``regions``: ``{name: pairs}`` in layout order; each region starts on a
    256-byte boundary (ATOM ``layout.SC``). Request row ``r`` of every region
    sits ``r * row_bytes`` past row 0; ``rows`` rows are reserved. A
    multi-layer launch double-buffers ``parity`` regions: slot 1 sits
    ``rows * row_bytes`` past slot 0. Timeline stamps follow at
    :attr:`stamp_base`.

    ``numa_copies=2`` (with ``MailboxKnobs(numa_copies=2)``): every region is
    a 16 KiB block holding two copies 8 KiB apart (``mailbox.NUMA_COPY_BYTES``;
    at most 1024 pairs), homed at different IODs; puts write both, consumers
    poll the copy homed at their IOD (:meth:`LayerCtx.mb` ``local=True``).
    The scratch base must be 16 KiB aligned.
    """

    regions: tuple[tuple[str, int], ...]
    rows: int = 1
    parity_slots: int = 1
    tail_bytes: int = 0
    numa_copies: int = 1

    def __post_init__(self) -> None:
        if self.numa_copies not in (1, 2):
            raise ValueError("numa_copies must be 1 or 2")
        if self.numa_copies == 2:
            for name, pairs in self.regions:
                if pairs * mbx.PAIR_BYTES > mbx.NUMA_COPY_BYTES:
                    raise ValueError(f"{name}: {pairs} pairs exceed one NUMA copy")

    @staticmethod
    def of(regions: Mapping[str, int], **kw) -> ScratchLayout:
        return ScratchLayout(tuple(regions.items()), **kw)

    def _bytes(self, pairs: int) -> int:
        if self.numa_copies == 2:
            return 2 * mbx.NUMA_COPY_BYTES
        return (pairs * mbx.PAIR_BYTES + 255) // 256 * 256

    @property
    def offsets(self) -> dict[str, int]:
        out, off = {}, 0
        for name, pairs in self.regions:
            out[name] = off
            off += self._bytes(pairs)
        return out

    @property
    def row_bytes(self) -> int:
        return sum(self._bytes(p) for _, p in self.regions)

    def region_index(self, name: str, row: int = 0, slot: int = 0) -> int:
        """Index of (region, row, slot) in a per-region table (NUMA flips)."""
        n = len(self.regions)
        names = [r for r, _ in self.regions]
        return names.index(name) + n * (row + self.rows * slot)

    @property
    def table_entries(self) -> int:
        return len(self.regions) * self.rows * self.parity_slots

    def offset(self, name: str, row: int = 0, slot: int = 0) -> int:
        return (
            self.offsets[name]
            + row * self.row_bytes
            + slot * self.rows * self.row_bytes
        )

    @property
    def mailbox_bytes(self) -> int:
        return self.parity_slots * self.rows * self.row_bytes + self.tail_bytes

    @property
    def stamp_base(self) -> int:
        return self.mailbox_bytes

    def total_bytes(self, stamps: bool, grid: int = stp.GRID) -> int:
        return self.mailbox_bytes + (stp.region_bytes(grid) if stamps else 0)


class Params:
    """Kernel arguments as uniform i64 addresses; pointers on demand."""

    def __init__(self, b: IRBuilder, values: Mapping[str, Value]):
        self._b = b
        self._v = dict(values)

    def names(self) -> tuple[str, ...]:
        return tuple(self._v)

    def addr(self, name: str) -> Value:
        return self._v[name]

    def ptr(self, b: IRBuilder, name: str, pointee=I32) -> Value:
        """A ``ptr addrspace(1)`` to ``name`` (``device_ops.ptr_from_i64``)."""
        return dops.ptr_from_i64(b, self._v[name], pointee=pointee)


@dataclass(frozen=True)
class StageSpec:
    """A stage: its schedule entry, its emitter and its LDS entries.

    ``emit(b, ctx, p, t)``: one task (see the module docstring).
    ``lds``: entries of ``stage.lds_phase`` (an arena phase).
    ``prologue(b, ctx, p)``: optional, run by every CTA (all threads) right
    before the stage's task rounds, whether or not it holds a task (e.g.
    idle-CTA prefetch of a later stage's weights).
    ``per_cta``: the emitter is self-guarded: every CTA calls ``emit`` once,
    unguarded, with ``t`` = its round-0 task index (``>= stage.tasks``: no
    task), the stage-function ABI's ``task`` (``stage_adapt``: hybrid stage
    bodies that place themselves, e.g. the MoE stages). Needs
    ``stage.tasks <= grid``.
    """

    stage: Stage
    emit: Callable[[IRBuilder, LayerCtx, Params, Value], None]
    lds: tuple[LdsEntry, ...] = ()
    prologue: Callable[[IRBuilder, LayerCtx, Params], None] | None = None
    per_cta: bool = False

    @property
    def name(self) -> str:
        return self.stage.name


class LayerCtx:
    """What a stage emitter sees of the kernel (module docstring)."""

    def __init__(self, b, *, table, scratch_layout, knobs, lds, scratch, rows,
                 seq, group, kind, stamps):  # fmt: skip
        self.b = b
        self.table: TaskTable = table
        self.layout: ScratchLayout = scratch_layout
        self.knobs: FrameworkKnobs = knobs
        self.lds: LdsView = lds
        self.scratch: Value = scratch
        self.rows, self.seq, self.group, self.kind = rows, seq, group, kind
        self._stamps: stp.Stamps | None = stamps
        self.stage_i = -1
        self.stage_name = ""
        self.tid = b.thread_id_x()
        self.lane = b.lane_id()
        self._wave0 = dops.wave_id(b)
        self._bid0 = b.readfirstlane(b.block_id_x())
        self.wave = self._wave0
        self.bid = self._bid0
        self.tag: Value | None = None
        self.tags: list[Value] = []  # tag of every layer run so far (li order)
        self.params: Params | None = None
        self.step_ptr: Value | None = None  # the kernel's i64 step pointer
        self.layer: Value | None = None
        self.li = 0
        self.slot = 0
        self.extra: dict = {}

    # -- per stage
    def _enter_stage(self, i: int, name: str) -> None:
        self.stage_i, self.stage_name = i, name
        if self.knobs.fresh_ids:
            self.wave = dops.fresh_s(self.b, self._wave0)
            self.bid = dops.fresh_s(self.b, self._bid0)

    # -- mailboxes
    def region_addr(self, name: str, row: int = 0, slot: int | None = None) -> Value:
        """Uniform i64 address of region ``name`` (row ``row``, parity slot
        ``slot``, default the current layer's)."""
        s = self.slot if slot is None else slot
        off = self.layout.offset(name, row, s)
        return self.b.add(self.scratch, self.b.const_i64(off))

    def mb(
        self, name: str, row: int = 0, slot: int | None = None, *, local: bool = False
    ) -> mbx.Mailbox:
        """Mailbox over region ``name``. ``local=True`` (polls): with NUMA
        copies, the view of the copy homed at this CTA's IOD."""
        m = mbx.mailbox_from_i64(
            self.b, self.region_addr(name, row, slot), self.b.const_i32(0x7FFFFFF0),
            self.knobs.mailbox,
        )  # fmt: skip
        if local and self.knobs.mailbox.numa_copies == 2:
            m = mbx.local_copy(self.b, m, self._flip(name, row, slot))
        return m

    def _flip(self, name: str, row: int, slot: int | None) -> Value | int:
        arg = self.knobs.numa_flip_arg
        if not arg:
            return 0
        s = self.slot if slot is None else slot
        i = self.layout.region_index(name, row, s)
        ptr = self.params.ptr(self.b, arg)
        return self.b.readfirstlane(self.b.global_load_i32(ptr, self.b.const_i32(i)))

    def put(self, mb: mbx.Mailbox, i, v: Value, tag: Value | None = None) -> None:
        mbx.put(self.b, mb, i, v, self.tag if tag is None else tag, self.knobs.mailbox)

    def put2(self, mb: mbx.Mailbox, i, v0: Value, v1: Value, tag=None) -> None:
        mbx.put2(
            self.b, mb, i, v0, v1, self.tag if tag is None else tag, self.knobs.mailbox
        )

    def poll(self, specs, tag: Value | None = None) -> list[list[Value]]:
        """``mailbox.poll`` with this layer's tag; stamps the wave's ready
        point of the current stage (stamped builds)."""
        out = mbx.poll(self.b, specs, self.tag if tag is None else tag, self.knobs.mailbox)
        if self._stamps is not None and self.stage_i >= 0:
            self._stamps.ready(self.stage_i)
        return out

    def barrier(self) -> None:
        """A CTA barrier that does not drain ``vmcnt`` (LDS phase changes,
        cross-wave LDS exchange)."""
        if self.knobs.lds_barrier == "lds_only":
            self.b.sync_lds_only(fence="compiler")
        else:
            self.b.s_barrier_bare(fence="compiler")


def _arena(specs: Sequence[StageSpec], rows: int, knobs: FrameworkKnobs,
           extra_pinned: Sequence[LdsEntry]) -> LdsArena:  # fmt: skip
    phases: dict[str, list[LdsEntry]] = {}
    pinned: list[LdsEntry] = list(extra_pinned)
    for s in specs:
        if not s.lds:
            continue
        if s.stage.lds_phase is None:
            pinned += list(s.lds)
            continue
        ph = phases.setdefault(s.stage.lds_phase, [])
        have = {e.name for e in ph}
        ph += [e for e in s.lds if e.name not in have]
    if knobs.stamps and knobs.stamps_lds == "arena":
        pinned.append(stp.LDS_ENTRY)
    return LdsArena(phases, rows=rows, pinned=pinned, min_words=knobs.lds_min_words)


def build_layer_kernel(
    name: str,
    specs: Sequence[StageSpec],
    table: TaskTable,
    layout: ScratchLayout,
    *,
    rows: int = 1,
    seq: bool = False,
    group: int = 0,
    kind: str = "",
    knobs: FrameworkKnobs = FrameworkKnobs(),  # noqa: B008
    threads: int = 512,
    args: Sequence[str] = ATOM_LAYER_ARGS,
    layers: int = 1,
    parity: bool = True,
    pinned_lds: Sequence[LdsEntry] = (),
    entry: Callable[[IRBuilder, LayerCtx, Params], None] | None = None,
) -> KernelDef:
    """Compose ``specs`` (in ``table`` order) into one layer kernel.

    ``layers`` > 1: a multi-layer launch (``build_stack``): the stages run
    for layer ids ``layer .. layer + layers - 1`` in one launch (statically
    unrolled), each layer with its own tag; with ``parity`` layer ``li`` uses
    scratch slot ``li & 1`` (``layout.parity_slots`` must be 2). The table
    must pass ``validate(layers=layers)``.
    ``entry(b, ctx, p)``: optional code at the kernel entry, before the
    first stage (e.g. weight preloads issued before the step counter load).
    """
    order = table.order
    if tuple(s.name for s in specs) != order:
        raise ValueError(f"specs {[s.name for s in specs]} != table order {order}")
    table.validate(layers=layers)
    if layers > 1 and parity and layout.parity_slots != 2:
        raise ValueError("a parity multi-layer launch needs layout.parity_slots == 2")
    if knobs.stamps and len(specs) > stp.MAX_STAGES:
        raise ValueError(f"stamps support at most {stp.MAX_STAGES} stages per layer")
    sfx = knobs.suffix()
    b = IRBuilder(name + (f"_{sfx}" if sfx else ""))
    b.kernel.attrs["max_workgroup_size"] = threads
    if knobs.waves_per_eu:
        b.kernel.attrs["waves_per_eu"] = knobs.waves_per_eu
    kp = knobs.kernarg_preload

    def pre(i: int) -> dict:
        return {"inreg": True} if i < kp else {}

    step_ptr = b.param("step_ptr", I64, **pre(0))
    layer0 = b.readfirstlane(b.param("layer", I32, **pre(1)))
    vals = {a: b.param(a, I64, **pre(2 + k)) for k, a in enumerate(args)}
    p = Params(b, vals)
    arena = _arena(specs, rows, knobs, pinned_lds)
    tl = None
    if knobs.stamps and knobs.stamps_lds == "own":
        # allocated first: the pool packer places it at offset 0, so the
        # arena's slot stays at the pool's end, where later stages' own
        # pools can reuse and extend it once the arena is dead
        tl = LdsArena({}, pinned=[stp.LDS_ENTRY]).emit(b, name_hint="tl")
    lds = arena.emit(b)
    st = stp.Stamps(b, tl or lds, len(specs)) if knobs.stamps else None
    if knobs.lds_touch and arena.words:
        # one store at the entry keeps the padded arena allocated from here
        # on (an otherwise unused arena is dropped by the lowering)
        with b.scf_if(b.cmp_eq(b.thread_id_x(), b.const_i32(0))):
            b.smem_store_vN(lds.smem, [b.const_i32(arena.words - 1)], b.const_i32(0), 1)
    ctx = LayerCtx(
        b, table=table, scratch_layout=layout, knobs=knobs, lds=lds,
        scratch=vals["scratch"], rows=rows, seq=seq, group=group, kind=kind,
        stamps=st,
    )  # fmt: skip
    ctx.extra["arena"] = arena
    ctx.params = p
    ctx.step_ptr = step_ptr
    if (knobs.mailbox.numa_copies == 2) != (layout.numa_copies == 2):
        raise ValueError("mailbox numa_copies and ScratchLayout numa_copies must agree")
    if knobs.numa_flip_arg and knobs.numa_flip_arg not in vals:
        raise ValueError(f"numa_flip_arg {knobs.numa_flip_arg!r} is not a kernel argument")
    if st is not None:
        st.begin()
    if knobs.step_first:
        # step counter load first, then the entry's loads: the tag's wait on
        # the step is ``vmcnt(#entry loads)``, not ``vmcnt(0)``
        step = _load_step(b, step_ptr, knobs)
        if entry is not None:
            entry(b, ctx, p)
    else:
        if entry is not None:
            entry(b, ctx, p)
        step = _load_step(b, step_ptr, knobs)
    if knobs.head_sleep:
        first = specs[0].stage
        t0 = for_task_probe(b, table, first.name, ctx.bid)
        with b.scf_if(b.cmp_ge(t0, b.const_i32(first.tasks))):
            dops.s_sleep(b, knobs.head_sleep)
    barriers = _barrier_points(specs, layers) if knobs.phase_barriers else set()
    for li in range(layers):
        ctx.li = li
        ctx.slot = (li & 1) if (layers > 1 and parity) else 0
        ctx.layer = b.add(layer0, b.const_i32(li)) if li else layer0
        ctx.tag = mbx.launch_tag(b, step, ctx.layer)
        ctx.tags.append(ctx.tag)
        if li and st is not None:
            # the previous layer's flush reads every CTA's points from LDS in
            # waves 0..2; begin() zeroes them from thread 0 / each wave's lane
            # 0: without this barrier a wave racing ahead clears points another
            # wave has not flushed yet (stacked stamped builds only)
            b.sync_lds_only(fence="compiler")
            st.begin()
        for i, s in enumerate(specs):
            if st is not None:
                st.enter(i)
            if (li, i) in barriers:
                ctx.barrier()
            ctx._enter_stage(i, s.name)
            if s.prologue is not None:
                s.prologue(b, ctx, p)
            emit = s.emit

            def body(t: Value, emit=emit) -> None:
                emit(b, ctx, p, t)

            fresh = (lambda v: dops.fresh_s(b, v)) if knobs.fresh_ids else None
            if s.per_cta:
                if s.stage.tasks > table.grid:
                    raise ValueError(f"per_cta stage {s.name}: {s.stage.tasks} tasks > grid")
                from rocke.helpers.task_schedule import task_index

                t = task_index(b, table, s.name, ctx.bid, 0)
                body(t)  # bid is this stage's fresh copy already
            else:
                for_my_tasks(b, table, s.name, ctx.bid, body, fresh=fresh)
            if st is not None:
                st.leave(i)
        if st is not None:
            st.flush(ctx.scratch, layout.stamp_base, ctx.layer, ctx._bid0)
    b.ret()
    return b.kernel


def _load_step(b: IRBuilder, step_ptr: Value, knobs: FrameworkKnobs) -> Value:
    if knobs.step_load == "smem":
        # glc: the counter changes between graph replays (no stale K$ line)
        return b.inline_asm(
            "s_load_dword $0, $1, 0x0 glc\ns_waitcnt lgkmcnt(0)", "=s,s", [step_ptr], I32,
            result_name_hint="step",
        )  # fmt: skip
    return b.global_load_i32(dops.ptr_from_i64(b, step_ptr), b.const_i32(0))


def _barrier_points(specs: Sequence[StageSpec], layers: int) -> set[tuple[int, int]]:
    """``(layer index, stage index)`` before which a CTA barrier is needed: the
    stage's LDS phase differs from the last phase used before it, over the
    whole (unrolled) launch (``TaskTable.lds_barriers`` per layer, plus the
    layer boundaries of a multi-layer launch)."""
    out, last = set(), None
    for li in range(layers):
        for i, s in enumerate(specs):
            ph = s.stage.lds_phase
            if ph is None:
                continue
            if last is not None and ph != last:
                out.add((li, i))
            last = ph
    return out


def for_task_probe(b: IRBuilder, table: TaskTable, name: str, bid: Value) -> Value:
    """This CTA's round-0 task index of ``name`` (for head-of-kernel logic)."""
    from rocke.helpers.task_schedule import task_index

    return task_index(b, table, name, bid, 0)


def stage_kernel(
    spec: StageSpec,
    table: TaskTable,
    layout: ScratchLayout,
    **kw,
) -> KernelDef:
    """One stage as a standalone layer kernel with the same ABI: the hybrid
    path's unit (``device_function.lower_device_function`` lowers it as a
    device function) and the isolated-stage A/B build (the host fills the
    mailboxes the stage reads). The stage keeps its placement; its
    dependencies become inputs."""
    one = replace(spec.stage, deps=())
    sub = replace(table, stages=(one,), inputs=tuple(table.inputs) + one.reads)
    name = kw.pop("name", f"rk_stage_{spec.name}")
    return build_layer_kernel(name, [replace(spec, stage=one)], sub, layout, **kw)


# ------------------------------------------------------- per-kind assembly


@dataclass(frozen=True)
class LayerShape:
    """A layer build's request rows (ATOM ``build_layer``): ``rows`` (1..4),
    ``seq`` (MTP verify rows of one request), ``group`` (grouped verify)."""

    rows: int = 1
    seq: bool = False
    group: int = 0


# stage name -> factory(kind, shape, stage) -> StageSpec. ``stage`` is the
# schedule entry of the ATOM table (tasks, base, deps, reads, writes,
# lds_phase); a factory may change its task count or placement (knobs) and
# returns the StageSpec to run.
StageFactory = Callable[[str, LayerShape, Stage], StageSpec]
_FACTORIES: dict[str, StageFactory] = {}


def register_stage(name: str, factory: StageFactory) -> StageFactory:
    """Make ``factory`` the default emitter of stage ``name`` (a stage track
    registers its emitter here; ``build_layer(factories=...)`` overrides per
    build, e.g. for a knob sweep)."""
    _FACTORIES[name] = factory
    return factory


def registered_stages() -> tuple[str, ...]:
    return tuple(sorted(_FACTORIES))


def build_layer(
    kind: str,
    shape: LayerShape = LayerShape(),  # noqa: B008
    *,
    stages: Sequence[str] | None = None,
    factories: Mapping[str, StageFactory] | None = None,
    knobs: FrameworkKnobs = FrameworkKnobs(),  # noqa: B008
    placement: str = "atom",
    layout: ScratchLayout | None = None,
) -> KernelDef:
    """A rocKE layer kernel of ``kind`` (``delta`` | ``delta_ple`` | ``qsa``)
    on ATOM's schedule (``atom_layout.atom_table``) and scratch layout
    (``atom_compat``), running ``stages`` (default: the whole kind; a
    contiguous run of the order: the stages before it are the host's / the
    FlyDSL prefix's, their regions become inputs) with the registered
    emitters (``register_stage``) or ``factories``. Same ABI as ATOM's
    ``build_layer`` launchers."""
    from .atom_layout import ORDERS, atom_layout, atom_table

    order = tuple(ORDERS[kind])
    run = tuple(stages) if stages else order
    first = order.index(run[0])
    if order[first:first + len(run)] != run:
        raise ValueError(f"{run} is not a contiguous run of {kind}'s order {order}")
    full = atom_table(kind).place(placement)
    facts = dict(_FACTORIES)
    facts.update(factories or {})
    missing = [n for n in run if n not in facts]
    if missing:
        raise KeyError(
            f"no emitter registered for stages {missing} (registered: "
            f"{registered_stages()}); pass factories= or register_stage()"
        )
    def inside(deps):
        # dependencies on stages outside the run are satisfied by the host
        return tuple(d for d in deps if d.stage in run or d.layer)

    specs = []
    for n in run:
        st = full.stage(n)
        sp = facts[n](kind, shape, replace(st, deps=inside(st.deps)))
        specs.append(replace(sp, stage=replace(sp.stage, deps=inside(sp.stage.deps))))
    produced = {r for sp in specs for r in sp.stage.writes}
    inputs = tuple(full.inputs) + tuple(
        r for sp in specs for r in sp.stage.reads if r not in produced
    )
    table = replace(full, stages=tuple(sp.stage for sp in specs), inputs=inputs)
    name = f"rk_q38_{kind}_" + "_".join(run) + (f"_r{shape.rows}" if shape.rows > 1 else "")
    name += ("_s" if shape.seq else "") + (f"_g{shape.group}" if shape.group else "")
    return build_layer_kernel(
        name, specs, table, layout or atom_layout(), rows=shape.rows, seq=shape.seq,
        group=shape.group, kind=kind, knobs=knobs,
    )  # fmt: skip
