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
from rocke.helpers.compile import request_code_object_backend
from rocke.helpers.lds_arena import (
    ONE_CTA_PER_CU_WORDS,
    LdsArena,
    LdsEntry,
    LdsView,
)
from rocke.helpers.task_schedule import Stage, TaskTable, for_my_tasks

from . import irops as iro
from . import stamps as stp
from .persist.knobs import EdgeKnobs, EdgePlan, PersistKnobs, RustScope

__all__ = [
    "ATOM_LAYER_ARGS",
    "CODE_OBJECT_BACKENDS",
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
CODE_OBJECT_BACKENDS = ("", "comgr", "direct", "auto")


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

    The true-mono platform knobs (``persist`` package, WP1; every one
    ``None`` / empty by default: a build without them is byte-identical):

    persist: :class:`.persist.knobs.PersistKnobs`: P6's R2 (``fresh="all"``:
      fresh lane / thread ids per stage too), R4 (``machine_licm="off"``:
      the compile drops MachineLICM), R5 (``wgid="lds"``: the workgroup id
      through an LDS word written at entry). A static build (``layers`` >= 1,
      unrolled) takes its arguments from the kernarg segment
      (``args="kernarg"``); R1 / R3 (``args="const"`` / ``"lds"``, a loop that
      carries only its index) are the persistent loop driver's
      (:func:`.persist.driver.build_persistent_kernel`).
    edges: :class:`.persist.knobs.EdgePlan`: per-edge poll implementation,
      pause, NUMA polarity and deadline, applied by :meth:`LayerCtx.mb` /
      :meth:`LayerCtx.poll`; protocol / placement / queue values other than
      the defaults must be consumed by the stage emitter (``ctx.edge(...)``
      marks them consumed); an unconsumed one is refused at the end of the
      build (never silently ignored).
    err_arg: the kernel argument holding the i32 error buffer (``persist.sync``
      ``ERR_*`` words) that bounded polls write; needed by any deadline edge.
    rust_scope: :class:`.persist.knobs.RustScope`: per-stage lever sets of the
      direct code-object backend: every op of a scoped stage carries
      ``attrs["rocke.scope"]`` and the kernel ``attrs["direct_scopes"]``
      (invisible to the LLVM lowering: COMGR objects unchanged).
    code_object_backend: the compiler that turns the kernel into a code
      object: ``""`` (the process default: ``ROCKE_CODEOBJ_BACKEND``, else
      COMGR) | ``"comgr"`` | ``"direct"`` (the Rust rocKE IR compiler, no
      LLVM; a kernel it does not lower is refused and COMGR compiles it,
      recorded in ``KernelArtifact.direct_fallback``) | ``"auto"`` (D0.6:
      direct when every stage of the kernel is scoped by ``rust_scope`` to a
      family whose framework-hosted direct builds won on both rocke nodes
      (``persist.knobs.RUST_FRAMEWORK_DIRECT_FAMILIES``; none yet, so COMGR),
      else COMGR; :func:`.persist.knobs.rust_scope_backend`). It marks the built
      ``KernelDef`` (``rocke.helpers.compile.request_code_object_backend``)
      and changes no IR, so it is not in the kernel name; the hybrid path's
      device functions (LLVM text) ignore it.
    irops: emit the device ops as IR ops (``irops``): runtime pointers as
      ``int_to_ptr``, the XCD id, sleeps and clocks as ``IRBuilder.s_getreg``
      / ``s_sleep`` / ``memrealtime`` instead of ``device_ops`` inline asm
      (Python engine only; name suffix ``irops``). With ``mailbox.impl="ir"``
      a build has no inline asm left but the empty ``fresh`` pins, which the
      direct code-object compiler takes.
    stage_levers: Rust direct compiler only: lever scopes per stage,
      ``"<stage>=<levers>;..."`` (stage names of the build's specs, levers
      ``+`` / ``,`` joined ``ROCKE_DIRECT_LEVERS`` names, ``-name`` turning a
      kernel lever off): every op a stage emits (its barrier, prologue and
      tasks) carries the scope (``IRBuilder.lever_scope``), so one mono
      kernel can take e.g. ``nofence`` in its MoE stage and not in its head.
      COMGR builds ignore it; ``""``: byte-identical IR. Not in the kernel
      name (a code-generation choice, as ``code_object_backend``: the same
      symbol for every scope set).
    direct_levers: Rust direct compiler only: the kernel's own levers
      (``KernelDef.attrs["direct_levers"]``, joined to
      ``ROCKE_DIRECT_LEVERS`` for its compile; not IR, not in the name).
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
    persist: PersistKnobs | None = None
    edges: EdgePlan | None = None
    err_arg: str = ""
    rust_scope: RustScope | None = None
    code_object_backend: str = field(default="", repr=False)
    irops: bool = field(default=False, repr=False)
    stage_levers: str = field(default="", repr=False)
    direct_levers: str = field(default="", repr=False)

    def stage_lever_map(self) -> dict:
        """``stage_levers`` as stage name -> lever scope text."""
        out = {}
        for part in filter(None, (x.strip() for x in self.stage_levers.split(";"))):
            name, sep, levers = part.partition("=")
            if not sep or not name.strip():
                raise ValueError(f"stage_levers: {part!r} is not <stage>=<levers>")
            out[name.strip()] = levers.strip()
        return out

    def __post_init__(self) -> None:
        if self.persist is not None and not isinstance(self.persist, PersistKnobs):
            raise TypeError("persist must be a PersistKnobs")
        if self.edges is not None and not isinstance(self.edges, EdgePlan):
            raise TypeError("edges must be an EdgePlan")
        if self.rust_scope is not None and not isinstance(self.rust_scope, RustScope):
            raise TypeError("rust_scope must be a RustScope")
        self.stage_lever_map()
        if self.code_object_backend not in CODE_OBJECT_BACKENDS:
            raise ValueError(f"code_object_backend must be one of {CODE_OBJECT_BACKENDS}")
        if self.code_object_backend == "auto" and (self.rust_scope is None or not self.rust_scope.enabled):
            raise ValueError("code_object_backend='auto' picks the backend from rust_scope's families: set rust_scope")
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
        if self.persist is not None:
            parts.append("ps" + self.persist.suffix())
        if self.edges is not None and not self.edges.is_default:
            parts.append("eg" + self.edges.suffix())
        if self.irops:
            parts.append("irops")
        # code_object_backend changes no IR: provenance is the artifact's
        # ``code_object_backend`` / ``direct_fallback``, not the name.
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

    def __init__(self, b: IRBuilder, values: Mapping[str, Value], irops: bool = False):
        self._b = b
        self._v = dict(values)
        self.irops = irops

    def names(self) -> tuple[str, ...]:
        return tuple(self._v)

    def addr(self, name: str) -> Value:
        return self._v[name]

    def ptr(self, b: IRBuilder, name: str, pointee=I32) -> Value:
        """A ``ptr addrspace(1)`` to ``name`` (``device_ops.ptr_from_i64``,
        or ``int_to_ptr`` with ``irops``)."""
        return iro.ptr(b, self._v[name], self.irops, pointee=pointee)


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
        # platform (persist) state; unused unless the knobs are set
        self._wgid_lds: Value | None = None
        self._mb_region: dict[int, str] = {}
        self._edges_used: set[str] = set()
        self._persist_used: set[str] = {"inline", "args", "fresh", "machine_licm", "wgid", "loop_carry", "unroll"}
        self.err_addr: Value | None = None

    def persist_knob(self, name: str):
        """A :class:`.persist.knobs.PersistKnobs` value for a stage emitter
        (``arrivals``, ``stripes``, ``ticket``, ``deadline``, ``fatal_word``,
        ``kv_bulk``, ``scalar_reread``, ``xhold``): the emitter applies it;
        reading it marks it consumed (a non-default value no emitter read is
        refused at the end of the build, never silently ignored)."""
        ps = self.knobs.persist
        if ps is None:
            from .persist.knobs import PersistKnobs as _PK

            return getattr(_PK(), name)
        self._persist_used.add(name)
        return getattr(ps, name)

    # -- per stage
    def _enter_stage(self, i: int, name: str) -> None:
        self.stage_i, self.stage_name = i, name
        ps = self.knobs.persist
        if ps is not None and ps.wgid == "lds" and self._wgid_lds is not None:
            # R5: the workgroup id read back from its LDS word (the hardware
            # SGPR is dead after the entry)
            from .persist.sync import lds_read

            self._bid0 = lds_read(self.b, self._wgid_lds)
        if self.knobs.fresh_ids:
            self.wave = dops.fresh_s(self.b, self._wave0)
            self.bid = dops.fresh_s(self.b, self._bid0)
        if ps is not None and ps.fresh == "all":
            # R2: fresh lane / thread ids per stage (no id math held across stages)
            self.lane = dops.fresh_v(self.b, self.b.lane_id())
            self.tid = self.b.add(self.b.mul(self.wave, self.b.const_i32(64)), self.lane)

    # -- edges (persist.knobs.EdgePlan)
    def _producer_of(self, region: str) -> str:
        for st in self.table.stages:
            if region in st.writes:
                return st.name
        return "*"

    def edge(self, producer: str | None = None, consumer: str | None = None,
             region: str | None = None) -> EdgeKnobs:  # fmt: skip
        """The :class:`EdgeKnobs` of an edge (default: the current stage as
        the consumer of ``region``'s producer; a region-named entry wins).
        Marks the entry consumed: a non-default protocol / placement / queue
        is then the asking emitter's responsibility (unconsumed ones are
        refused at the end of the build)."""
        key, ek = self._lookup_edge(producer, consumer, region)
        if key is not None:
            self._edges_used.add(key)
        return ek

    def _lookup_edge(self, producer, consumer, region) -> tuple[str | None, EdgeKnobs]:
        plan = self.knobs.edges
        if plan is None:
            return None, EdgeKnobs()
        cons = consumer or self.stage_name
        prod = producer or (self._producer_of(region) if region else "*")
        keys = ([region] if region else []) + [f"{prod}>{cons}", f"*>{cons}", f"{prod}>*"]
        for key in keys:
            for k, v in plan.edges:
                if k == key:
                    return k, v
        return None, plan.default

    def _edge_for_region(self, region: str) -> EdgeKnobs | None:
        # the framework's own lookup (poll / pause / polarity / deadline):
        # it does not count as an emitter consuming the protocol / placement /
        # queue knobs
        if self.knobs.edges is None:
            return None
        return self._lookup_edge(None, None, region)[1]

    def _mbk(self, ek: EdgeKnobs | None) -> mbx.MailboxKnobs:
        k = self.knobs.mailbox
        if ek is None:
            return k
        if ek.mailbox_impl in ("ir", "irf"):
            from . import mailbox_ir  # noqa: F401  (registers "ir" / "irf")
        return replace(k, impl=ek.mailbox_impl, pause=ek.pause)

    # -- mailboxes
    def region_addr(self, name: str, row: int = 0, slot: int | None = None) -> Value:
        """Uniform i64 address of region ``name`` (row ``row``, parity slot
        ``slot``, default the current layer's)."""
        s = self.slot if slot is None else slot
        if isinstance(s, Value):
            # a runtime parity slot (the persistent loop driver: slot = li & 1)
            b = self.b
            off0 = b.const_i64(self.layout.offset(name, row, 0))
            step = self.layout.rows * self.layout.row_bytes
            dyn = b.mul(b.zext(s, I64), b.const_i64(step))
            return b.add(self.scratch, b.add(off0, dyn))
        off = self.layout.offset(name, row, s)
        return self.b.add(self.scratch, self.b.const_i64(off))

    def mb(
        self, name: str, row: int = 0, slot: int | None = None, *, local: bool = False
    ) -> mbx.Mailbox:
        """Mailbox over region ``name``. ``local=True`` (polls): with NUMA
        copies, the view of the copy homed at this CTA's IOD."""
        ek = self._edge_for_region(name)
        m = mbx.mailbox_from_i64(
            self.b, self.region_addr(name, row, slot), self.b.const_i32(0x7FFFFFF0),
            self._mbk(ek),
        )  # fmt: skip
        if local and self.knobs.mailbox.numa_copies == 2:
            flip = self._flip(name, row, slot)
            if ek is not None and ek.polarity == "remote":
                # the copy homed at the other IOD (P4's microbench polarity)
                flip = flip ^ 1 if isinstance(flip, int) else self.b.xor(flip, self.b.const_i32(1))
            if ek is None or ek.polarity != "none":
                xcc = (lambda b: iro.xcc_id(b, True)) if self.knobs.irops else None
                m = mbx.local_copy(self.b, m, flip, xcc)
        if ek is not None:
            self._mb_region[id(m)] = name
        return m

    def _flip(self, name: str, row: int, slot: int | None) -> Value | int:
        arg = self.knobs.numa_flip_arg
        if not arg:
            return 0
        s = self.slot if slot is None else slot
        ptr = self.params.ptr(self.b, arg)
        if isinstance(s, Value):
            b = self.b
            i0 = self.layout.region_index(name, row, 0)
            per = len(self.layout.regions) * self.layout.rows
            idx = b.add(b.const_i32(i0), b.mul(s, b.const_i32(per)))
            return b.readfirstlane(b.global_load_i32(ptr, idx))
        i = self.layout.region_index(name, row, s)
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
        tg = self.tag if tag is None else tag
        ek = None
        if self.knobs.edges is not None:
            regions = {self._mb_region.get(id(s[0])) for s in specs}
            regions.discard(None)
            eks = {self._edge_for_region(r) for r in regions}
            if len(eks) > 1:
                raise ValueError(f"one poll over regions {sorted(regions)} with different edge knobs")
            ek = eks.pop() if eks else None
        if ek is not None and ek.poll == "deadline":
            out = self._poll_deadline(specs, tg, ek)
        else:
            out = mbx.poll(self.b, specs, tg, self._mbk(ek))
        if self._stamps is not None and self.stage_i >= 0:
            self._stamps.ready(self.stage_i)
        return out

    def _poll_deadline(self, specs, tag: Value, ek: EdgeKnobs) -> list[list[Value]]:
        """Bounded asm polls (``persist.sync.poll_pair_deadline``), one pair
        per call; a timeout counts in ``err[ERR_TIMEOUT]`` and sets
        ``poll_timeout`` in the launch-fatal word (lane 0 of the polling
        wave); the stale payload is returned (the host voids the launch)."""
        from .persist import sync as PS

        if self.err_addr is None:
            raise ValueError("a deadline edge needs FrameworkKnobs.err_arg (the error buffer argument)")
        b = self.b
        maxit = b.const_i32(ek.deadline)
        out = []
        for mb, i, n in specs:
            vals = []
            for q in range(n):
                idx = mbx._i32(b, i)
                if q:
                    idx = b.add(idx, b.const_i32(q))
                voff = mbx.pair_offset(b, idx)
                if mb.copy_off is not None:
                    voff = b.add(voff, mb.copy_off)
                v, it = PS.poll_pair_deadline(b, voff, mb.addr, tag, maxit, ek.pause)
                with b.scf_if(b.land(b.cmp_ge(it, maxit), b.cmp_eq(b.lane_id(), b.const_i32(0)))):
                    dops.atomic_add_scoped(b, self.err_addr, b.const_i32(4 * PS.ERR_TIMEOUT), b.const_i32(1))
                    PS.raise_fatal(b, self.err_addr, "poll_timeout")
                vals.append(v)
            out.append(vals)
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
    stage_levers = knobs.stage_lever_map()
    if set(stage_levers) - set(order):
        raise ValueError(f"stage_levers names stages {sorted(set(stage_levers) - set(order))} not in {order}")
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
    ps = knobs.persist
    if ps is not None:
        ps.check()
        if ps.args != "kernarg":
            raise ValueError(
                f"persist.args={ps.args!r}: a static layer kernel takes its arguments from the kernarg "
                "segment (args='kernarg'); R1 ('const' / 'lds') is the persistent loop driver's "
                "(persist.driver.build_persistent_kernel)"
            )
        _apply_persist_attrs(b, ps)
    if knobs.edges is not None:
        knobs.edges.validate(_edge_names(table, layout))
    kp = knobs.kernarg_preload

    def pre(i: int) -> dict:
        return {"inreg": True} if i < kp else {}

    step_ptr = b.param("step_ptr", I64, **pre(0))
    layer0 = b.readfirstlane(b.param("layer", I32, **pre(1)))
    vals = {a: b.param(a, I64, **pre(2 + k)) for k, a in enumerate(args)}
    p = Params(b, vals, knobs.irops)
    arena = _arena(specs, rows, knobs, pinned_lds)
    tl = None
    if knobs.stamps and knobs.stamps_lds == "own":
        # allocated first: the pool packer places it at offset 0, so the
        # arena's slot stays at the pool's end, where later stages' own
        # pools can reuse and extend it once the arena is dead
        tl = LdsArena({}, pinned=[stp.LDS_ENTRY]).emit(b, name_hint="tl")
    lds = arena.emit(b)
    st = stp.Stamps(b, tl or lds, len(specs), knobs.irops) if knobs.stamps else None
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
    _persist_entry(b, ctx, knobs, vals)
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
            iro.sleep(b, knobs.head_sleep, knobs.irops)
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
            with _RustScope(b, knobs, s.name):
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
    _persist_pin_lds(b, ctx)
    b.ret()
    _persist_finish(b, ctx, knobs)
    return b.kernel


# ------------------------------------------- platform (persist) hooks, WP1


def _apply_persist_attrs(b: IRBuilder, ps: PersistKnobs) -> None:
    """Kernel attributes of the persist knobs (R4: ``machine_licm="off"``,
    which ``rocke.helpers.compile`` turns into ``-disable-machine-licm``)."""
    if ps.machine_licm == "off":
        b.kernel.attrs["machine_licm"] = "off"


def _edge_names(table: TaskTable, layout: ScratchLayout) -> set[str]:
    return {s.name for s in table.stages} | {r for r, _ in layout.regions}


def _persist_entry(b: IRBuilder, ctx: LayerCtx, knobs: FrameworkKnobs, vals: Mapping[str, Value]) -> None:
    """Entry code of the platform knobs: the error buffer address and (R5)
    the workgroup id's LDS word (written by thread 0, then a CTA barrier)."""
    if knobs.err_arg:
        if knobs.err_arg not in vals:
            raise ValueError(f"err_arg {knobs.err_arg!r} is not a kernel argument")
        ctx.err_addr = vals[knobs.err_arg]
    ps = knobs.persist
    if ps is None:
        return
    if ps.wgid == "lds":
        from .persist.sync import lds_write

        w = b.smem_alloc(I32, [1], name_hint="wgid")
        ctx.extra.setdefault("pinned_smem", []).append(w)
        addr = b.trunc(b.smem_addr_of(w), I32)
        with b.scf_if(dops.is_thread0(b)):
            lds_write(b, addr, b.block_id_x())
        b.s_barrier_bare(fence="compiler")
        ctx._wgid_lds = addr


def _persist_pin_lds(b: IRBuilder, ctx: LayerCtx) -> None:
    """Keep the platform's address-taken LDS words alive to the kernel end.

    They are read and written through inline asm on their integer address
    (``smem_addr_of``), which the lowering's LDS liveness does not see: its
    interval would end at the ``smem_addr_of`` and the pool packer could
    overlay another allocation on the word (seen on hardware: the loop
    driver's argument row overlaid the R5 workgroup-id word and the launch
    hung). One more ``smem_addr_of`` right before the return extends the
    interval over the whole kernel."""
    for w in ctx.extra.get("pinned_smem", ()):
        b.smem_addr_of(w)


def _persist_finish(b: IRBuilder, ctx: LayerCtx, knobs: FrameworkKnobs) -> None:
    """Refuse edge knobs nobody consumed (a protocol / placement / queue the
    framework does not apply itself and no emitter asked for), and record the
    Rust lever scopes on the kernel."""
    ps = knobs.persist
    if ps is not None:
        from dataclasses import fields as _fields

        d0 = PersistKnobs()
        unused = [f.name for f in _fields(ps)
                  if getattr(ps, f.name) != getattr(d0, f.name) and f.name not in ctx._persist_used]  # fmt: skip
        if unused:
            raise ValueError(
                f"persist knobs {unused} differ from the defaults but no stage emitter read them "
                "(ctx.persist_knob(...)); the framework does not apply them itself"
            )
    plan = knobs.edges
    if plan is not None:
        d = EdgeKnobs()
        for k, v in plan.edges:
            emitter_level = (v.protocol, v.placement, v.queue, v.island) != (d.protocol, d.placement, d.queue, d.island)
            if emitter_level and k not in ctx._edges_used:
                raise ValueError(
                    f"edge {k!r}: protocol / placement / queue knobs {v.suffix()!r} were not consumed by any "
                    "stage emitter (ctx.edge(...)); the framework does not apply them itself"
                )
    sc = knobs.rust_scope
    if sc is not None and sc.enabled:
        b.kernel.attrs["direct_scopes"] = ";".join(
            f"{k}:{'+'.join(v)}" for k, v in sorted(sc.levers_by_stage().items())
        )
    # The direct code-object backend (WP-R3): the request is a Python mark on the
    # KernelDef and ``direct_levers`` a kernel attribute the direct backend strips;
    # neither reaches the LLVM IR, the serialized IR of a COMGR build or the name.
    backend = knobs.code_object_backend
    if backend == "auto":
        from .persist.knobs import rust_scope_backend

        backend = rust_scope_backend(sc, tuple(s.name for s in ctx.table.stages))
    if backend:
        request_code_object_backend(b.kernel, backend)
    if knobs.direct_levers:
        b.kernel.attrs["direct_levers"] = knobs.direct_levers


class _RustScope:
    """The direct backend's lever scopes of one stage (its barrier, prologue and
    tasks): ``rust_scope`` tags every op the stage emits, constants included,
    with ``attrs["rocke.scope"] = <stage>`` (the direct backend turns the named
    set into the compiler's ``dlevers``); ``stage_levers`` puts an explicit
    ``dlevers`` scope on them (``IRBuilder.lever_scope``; it wins over the
    named set). Neither reaches the LLVM IR."""

    def __init__(self, b: IRBuilder, knobs: FrameworkKnobs, stage: str):
        sc = knobs.rust_scope
        self.on = sc is not None and sc.levers(stage) is not None
        self.levers = knobs.stage_lever_map().get(stage)
        self.b, self.stage, self._cm, self._ls = b, stage, None, None

    def __enter__(self):
        if self.on:
            from .persist.kv import marked

            self._cm = marked(self.b, self.stage, attr="rocke.scope", constants=True)
            self._cm.__enter__()
        if self.levers:
            self._ls = self.b.lever_scope(self.levers)
            self._ls.__enter__()
        return self

    def __exit__(self, *exc):
        if self._ls is not None:
            self._ls.__exit__(*exc)
        if self._cm is not None:
            return self._cm.__exit__(*exc)
        return False


def _load_step(b: IRBuilder, step_ptr: Value, knobs: FrameworkKnobs) -> Value:
    if knobs.step_load == "smem":
        # glc: the counter changes between graph replays (no stale K$ line)
        return b.inline_asm(
            "s_load_dword $0, $1, 0x0 glc\ns_waitcnt lgkmcnt(0)", "=s,s", [step_ptr], I32,
            result_name_hint="step",
        )  # fmt: skip
    return b.global_load_i32(iro.ptr(b, step_ptr, knobs.irops), b.const_i32(0))


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
