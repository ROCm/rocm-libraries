# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Stage-scheduler table for the attention backward tile step.

One q step of the backward body runs five GEMMs (G0 ``S = Q K^T``, G1
``dV += P^T dO``, G2 ``dP = dO V^T``, G3 ``dK += dS^T Q``, G4
``dQ = dS K``). :class:`StageTable` groups them into four stages and gives,
per wave and per lane, how many MFMA (or WMMA) instructions and how many
VMEM-read, VMEM-write, DS-read and DS-write instructions each stage issues.
The counts come from the tile, the warp grids and each GEMM's own atom, never
from G0's atom alone, and K = 32 steps on an arch without a native 16x16x32
atom count as two 16x16x16 instructions.

Count models (``StageTable.pipeline``): ``"sync"``, ``"ring"`` and
``"merged"`` are the built bodies (:meth:`StageTable.from_geometry` picks one
from the geometry); ``"ck"`` is the reference pipeline's placement, kept as
the design reference.

* ``"sync"`` (``global_path = "vgpr"``, ``ring_depth = 1``, no barrier
  merge): the counts of the synchronous q step of
  ``_attention_bwd_body.BwdTileStep.run_q_step`` under the fenced placement.
  Per stage:

  * A: G0 with its Q (and LDS K) fragment reads, the lse2 / Dsum loads
    (``stats_load = "at_use"``), the dO^T fragments of G1 read ahead into
    registers and, under ``stage_table_atomics``, the previous step's dQ
    atomics;
  * B: G2 + G1 with their operand reads, the dS (and P) tile writes, the LDS
    accumulator read-modify-write (``acc_in_lds``) and the Q^T fragments of
    G3 read ahead;
  * C: G3 with its operand reads and, when K^T comes from LDS, the K^T
    fragments of G4 read ahead;
  * D: G4 with its dS reads and the dQ atomics (none under
    ``stage_table_atomics``, whose atomics move into the next step's A).

  The Q / dO staging ahead of stage A (global loads, LDS writes) is not
  grouped. Each count is the number of IR memory / matrix ops of that kind
  the stage emits (the census of the step's own IR; a test compares the two
  on the built instances).

* ``"ring"`` (the Q / dO prefetch ring of ``_attention_bwd_ring``: the DMA
  path at any ``ring_depth``, the register path at ``ring_depth`` 2 / 3): the
  synchronous stages with the ring's own work. The step reads Q / dO and the
  lse2 / Dsum rows from its ring slot (the stats are DS reads: in A under a
  fenced sched, else in B). The register ring's prefetch (the global loads of
  the tile ``R - 1`` steps ahead: its Q / dO chunks and stats row) sits in
  stage A under a stage table, spread over G0's MFMAs (one group of VMEM
  reads before each), and ahead of a ``sched_barrier(0)`` before the step
  otherwise (:meth:`StageTable.prefetch_region`); its slot stores close the
  step after D (:attr:`StageCounts.fixed`, outside the groups: a branch
  guards the stats store). The DMA ring's copy is inline asm in per-wave
  branches at the step top (no IR memory op the groups could take; the copy
  is asynchronous, so it overlaps the whole step anyway): it is counted per
  wave (:meth:`StageTable.ring_prefetch`), and its ``s_waitcnt vmcnt`` (step
  top for ``R = 1``, step end otherwise) is the ring's wait model
  (:meth:`StageTable.ring_waits`).
* ``"merged"`` (``barrier_merge = "on"``): the 2-slot merged-barrier ring.
  Stats from the slot as above; the next step's global loads sit in stage A
  under a stage table with ``ring_issue = "early"`` (at the step top
  otherwise, or right before the merged barrier with ``"late"``), and their
  slot stores right before the merged barrier, ahead of D (outside the
  groups).
* ``"ck"`` (not built): the reference pipeline's placement, with the next
  tile's global reads in A, its LDS writes in B, the dS slices of ``kK4`` keys
  in C / D and the next A operands in C / D (``prefetch``) and the previous
  step's dQ atomics in A (``prev_atomics``).

The split-dQ main kernel (``dq_mode = "split"``: no G4, no dS tile, no dQ
atomics; its dQ comes from the Q-parallel dQ kernel) runs three stages: A
(G0), B (G2 + G1) and C (G3), C closing the step (no fence after it; the
ring's slot stores fall in C's region). Its bodies are the synchronous step
and the prefetch rings (the merged barrier and the counted waits need G4 and
its atomics); ``stage_table_atomics`` has no atomics to defer there and is
refused (:data:`SPLIT_DEFERRED`). Under the relabel route the hand-off writes
no dS tile, and no K^T fragment is read.

:class:`DqStageTable` is the stage table of that dQ kernel's key step (the
``dq_sched`` values of :data:`DQ_SCHEDS`).

``stage_table_atomics`` with a ring: the previous step's dQ atomics are
issued in stage A, after the step's prefetch issue. The DMA ring's wait
(:func:`._attention_bwd_ring.dma_wait_counts`) counts the VMEM operations a
wave issues after its part of a tile; they are the ``n_a`` deferred atomics
instead of the step's own (the same number, branch-free, zeros on the first
step), so the immediates are unchanged (:meth:`StageTable.vm_after_prefetch`
is checked against them). The register rings' waits are the compiler's.

Values of ``sched``: ``none`` emits nothing; ``iglp0`` / ``iglp1`` emit one
``iglp_opt`` after the step's staging barrier and nothing else (never with a
stage table); the fenced values (``FENCED_SCHEDS``) take the fenced placement
and put a ``sched_barrier(0)`` after every stage but the last (A, B and C; A
and B without G4); ``fences_only`` stops there,
``stage_table`` and ``stage_table_atomics`` also interleave each stage's work
with ``sched_group_barrier(mask, count, 0)`` groups (one MFMA per group, the
memory work spread between them; operand reads of the first MFMA and every
VMEM read lead, the dS / P writes of B follow G1's MFMAs, the dQ atomics of D
follow its last K step). ``setprio`` raises the wave priority around each
stage's ops (stage tables only).

:meth:`StageTable.ir_count_model` gives the IR op counts the emission
produces and :meth:`StageTable.hint_sequence` the per-step order of the hint
ops (the register ring's prefetch fence included), so a test can compare
model and IR.

Residency: with ``kv_residency = "lds"`` the K (G0) and V (G2) B fragments are
read from their resident LDS images (D contiguous, so one read per 16 bytes)
once per distinct (n tile, k step) of the wave in every q step. With
``acc_in_lds`` the fp32 dV (G1) and dK (G3) accumulators live in row-major
fp32 LDS images; each 16x16 output tile of the wave is read before its MMA
chain and written back after it, one ``b32`` access per accumulator element
(the lanes of an MMA C fragment hold distinct rows, so the elements are not
contiguous in a row-major image).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from rocke.core.arch import ArchTarget
from rocke.helpers.compile import COMGR_FRESH_PROCESS_ATTR
from rocke.helpers.schedule import DS_READ, DS_WRITE, MFMA, VMEM_READ, VMEM_WRITE
from rocke.helpers.spec import ceil_div

from ._attention_bwd_lds import default_xt_writer

__all__ = [
    "BUILT_PIPELINES",
    "COUNTED_LATE_DEFERRED",
    "DQ_FENCED_SCHEDS",
    "DQ_SCHEDS",
    "DQ_STAGES",
    "FENCED_SCHEDS",
    "GEMMS",
    "IGLP_SCHEDS",
    "PIPELINES",
    "SCHEDS",
    "SPLIT_DEFERRED",
    "SPLIT_STAGES",
    "STAGES",
    "STAGE_GEMMS",
    "STAGE_TABLE_SCHEDS",
    "DqStageTable",
    "GemmCounts",
    "StageCounts",
    "StageTable",
    "instruction_k",
    "interleave_groups",
    "native_atom_ks",
]

SCHEDS = ("none", "iglp0", "iglp1", "fences_only", "stage_table", "stage_table_atomics")
STAGE_TABLE_SCHEDS = ("stage_table", "stage_table_atomics")
# The values with the fenced placement and the stage fences.
FENCED_SCHEDS = ("fences_only",) + STAGE_TABLE_SCHEDS
IGLP_SCHEDS = ("iglp0", "iglp1")
# Count models: the synchronous body, the prefetch ring and the merged-barrier
# ring (built), and the reference pipeline's placement ("ck", not built).
PIPELINES = ("sync", "ring", "merged", "ck")
BUILT_PIPELINES = ("sync", "ring", "merged")
# The one schedule value the counted waits refuse, and why: their late issue
# puts the next step's loads after the deferred atomics of stage A, the
# reverse of the counted issue order (every load of the step before its dQ
# atomics, so a barrier's vmcnt retires the loads and leaves the atomics).
COUNTED_LATE_DEFERRED = (
    "sched='stage_table_atomics' with loop_sync='counted' needs "
    "counted_issue='early': the late issue would load the next step after the "
    "deferred dQ atomics of stage A (the counted waits keep every load of the "
    "step ahead of its atomics)"
)
STAGES = ("A", "B", "C", "D")
# The split-dQ main kernel's stages: no G4, so the step ends with C (G3).
SPLIT_STAGES = STAGES[:3]
# The one schedule value the split-dQ main kernel refuses, and why.
SPLIT_DEFERRED = (
    "sched='stage_table_atomics' needs dq_mode='atomic': the split-dQ main "
    "kernel issues no dQ atomics to defer (its stage table is "
    "sched='stage_table')"
)
# The split dQ kernel's schedule values (``dq_sched``): none, one iglp_opt per
# key step, the stage fences alone, the stage table (fences and groups) and
# the stage table with the wave priority raised around each stage.
DQ_SCHEDS = (
    "none",
    "iglp0",
    "iglp1",
    "fences_only",
    "stage_table",
    "stage_table_setprio",
)
DQ_FENCED_SCHEDS = ("fences_only", "stage_table", "stage_table_setprio")
# Its key step's stages: A (G0' S^T = K Q^T), B (G2' dP^T = V dO^T and the
# softmax gradient), C (G4' dQ += dS K).
DQ_STAGES = ("A", "B", "C")
GEMMS = ("g0", "g1", "g2", "g3", "g4")
STAGE_GEMMS: Mapping[str, tuple[str, ...]] = {
    "A": ("g0",),
    "B": ("g1", "g2"),
    "C": ("g3",),
    "D": ("g4",),
}
_TRANSPOSE_SOURCES = ("xt_lds", "tr_read", "lds_plain", "wmma_lds")
# G4 dS key slicings of the body (``_attention_bwd_body.G4_SLICINGS``, which
# imports this module and checks that the two agree).
_G4_SLICINGS = ("step", "slice", "slice_db", "slice_db_g3")
_DTYPES = {"fp16": "fp16", "f16": "fp16", "bf16": "bf16"}
_ELEM_BYTES = 2
# Memory classes in emission order around each MFMA group: VMEM reads go
# before the MFMA (they have the longest latency), the rest after it.
_PRE = (VMEM_READ,)
_POST = (DS_READ, DS_WRITE, VMEM_WRITE)


def _mma_ops(arch: str, dtype: str):
    tgt = ArchTarget.from_gfx(arch)
    dt = _DTYPES.get(dtype)
    if dt is None:
        raise ValueError(f"unsupported dtype {dtype!r}")
    family = "mma" if tgt.has_mfma else "wmma"
    ops = tgt.mma.enumerate(
        family=family, a_dtype=dt, b_dtype=dt, c_dtype="fp32", m=16, n=16
    )
    ops = [op for op in ops if op.wave_size == tgt.wave_size and op.a_frag_len > 0]
    if not ops:
        raise ValueError(f"no 16x16 {dt} MMA atom on {arch}")
    return sorted(ops, key=lambda op: op.k)


def native_atom_ks(arch: str, dtype: str = "fp16") -> tuple[int, ...]:
    """K extents of the native 16x16 MMA atoms of ``arch`` (core catalog)."""
    return tuple(op.k for op in _mma_ops(arch, dtype))


def instruction_k(arch: str, atom_k: int, dtype: str = "fp16") -> int:
    """K of the instruction that implements a 16x16x``atom_k`` step on ``arch``.

    A native atom is one instruction; a K = 32 step on an arch whose widest
    native atom is K = 16 is two 16x16x16 instructions.
    """
    ks = native_atom_ks(arch, dtype)
    if atom_k in ks:
        return int(atom_k)
    smaller = [k for k in ks if atom_k % k == 0]
    if not smaller:
        raise ValueError(f"no instruction can implement a K={atom_k} step on {arch}")
    return max(smaller)


def _atom_k(value: Any) -> int:
    if value is None:
        return 16
    if isinstance(value, int):
        return int(value)
    k = getattr(value, "k", None)
    if k is not None:
        return int(k)
    if isinstance(value, str):
        text = value.lower().replace("wmma", "")
        parts = text.split("x")
        if len(parts) == 3:
            return int(parts[2])
    raise ValueError(f"cannot read an atom K from {value!r}")


def _atom_from(atoms: Any, *names: str) -> Any:
    if atoms is None:
        return None
    for name in names:
        if isinstance(atoms, Mapping) and name in atoms:
            return atoms[name]
        if hasattr(atoms, name):
            return getattr(atoms, name)
    return None


def interleave_groups(
    mfma: int,
    mem: Sequence[tuple[int, int]],
    *,
    lead: Mapping[int, int] | None = None,
    spans: Mapping[int, int] | None = None,
) -> list[tuple[int, int]]:
    """Spread memory work evenly between ``mfma`` single-MFMA groups.

    Group ``i`` receives ``floor((i+1) c / m) - floor(i c / m)`` of a class
    with total ``c``; VMEM reads go before the MFMA, the other classes after.
    ``lead`` (``{mask: n}``) takes ``n`` of a class out of the spread and
    places them before the first MFMA (VMEM reads first, then the others in
    ``mem`` order). ``spans`` (``{mask: s}``) spreads a class over the last
    ``s`` MFMAs only (work that depends on the stage's late MFMAs). Adjacent
    groups of one class are merged. With no MFMA the memory groups are
    emitted as they are. The per-mask sums always equal the inputs.
    """
    lead = dict(lead or {})
    spans = dict(spans or {})
    raw: list[tuple[int, int]] = []
    if mfma <= 0:
        raw = [(m, c) for m, c in mem if c > 0]
    else:
        total = {m: c for m, c in mem}
        for m, n in lead.items():
            if not 0 <= n <= total.get(m, 0):
                raise ValueError(f"lead {n} of mask {m:#x} exceeds its count")
        for m, s in spans.items():
            if not 0 < s <= mfma:
                raise ValueError(f"span {s} of mask {m:#x} outside 1..{mfma}")
        order = [m for m, _ in mem]
        for m in sorted(order, key=lambda x: x not in _PRE):
            if lead.get(m, 0):
                raw.append((m, lead[m]))
        rest = [(m, c - lead.get(m, 0)) for m, c in mem]
        pre = [(m, c) for m, c in rest if m in _PRE and c > 0]
        post = [(m, c) for m, c in rest if m not in _PRE and c > 0]

        def share(m: int, c: int, i: int) -> int:
            span = spans.get(m, mfma)
            j = i - (mfma - span)
            if j < 0:
                return 0
            return (j + 1) * c // span - j * c // span

        for i in range(mfma):
            for m, c in pre:
                s = share(m, c, i)
                if s:
                    raw.append((m, s))
            raw.append((MFMA, 1))
            for m, c in post:
                s = share(m, c, i)
                if s:
                    raw.append((m, s))
    out: list[tuple[int, int]] = []
    for m, c in raw:
        if out and out[-1][0] == m:
            out[-1] = (m, out[-1][1] + c)
        else:
            out.append((m, c))
    return out


@dataclass(frozen=True)
class GemmCounts:
    """MMA instructions per wave per q step, per GEMM."""

    g0: int
    g1: int
    g2: int
    g3: int
    g4: int

    def as_dict(self) -> dict[str, int]:
        return {g: getattr(self, g) for g in GEMMS}

    @property
    def total(self) -> int:
        return self.g0 + self.g1 + self.g2 + self.g3 + self.g4


@dataclass(frozen=True)
class StageCounts:
    """Per-wave instruction counts of one stage (memory counts per lane).

    ``lead`` / ``spans`` shape the stage's groups (see
    :func:`interleave_groups`): ``((mask, n), ...)`` placed before the first
    MFMA, and ``((mask, s), ...)`` spread over the last ``s`` MFMAs only.
    ``fixed`` (``((mask, n), ...)``): memory ops of the ring that the step
    emits inside the stage's fenced region but outside its groups (slot
    stores behind a branch or before a barrier, loads issued at a barrier);
    the IR census of the region is the counts plus ``fixed``.
    """

    stage: str
    mfma: int
    vmem_read: int = 0
    vmem_write: int = 0
    ds_read: int = 0
    ds_write: int = 0
    lead: tuple[tuple[int, int], ...] = ()
    spans: tuple[tuple[int, int], ...] = ()
    fixed: tuple[tuple[int, int], ...] = ()

    def memory(self) -> tuple[tuple[int, int], ...]:
        return (
            (VMEM_READ, self.vmem_read),
            (DS_READ, self.ds_read),
            (DS_WRITE, self.ds_write),
            (VMEM_WRITE, self.vmem_write),
        )

    def by_mask(self) -> dict[int, int]:
        return {
            MFMA: self.mfma,
            VMEM_READ: self.vmem_read,
            VMEM_WRITE: self.vmem_write,
            DS_READ: self.ds_read,
            DS_WRITE: self.ds_write,
        }

    def census(self) -> dict[int, int]:
        """IR ops per class in the stage's fenced region: the grouped counts
        plus :attr:`fixed`."""
        out = self.by_mask()
        for m, n in self.fixed:
            out[m] += n
        return out

    def groups(self) -> list[tuple[int, int]]:
        """``(mask, count)`` sched-group sequence of this stage."""
        return interleave_groups(
            self.mfma, self.memory(), lead=dict(self.lead), spans=dict(self.spans)
        )


def _swizzle_pairs(value: str | None) -> dict[str, str]:
    """``{"q": "pad8", ...}`` of a canonical ``lds_swizzle`` text."""
    out = {}
    for item in (value or "").split(","):
        name, sep, val = item.strip().partition("=")
        if sep:
            out[name.strip()] = val.strip()
    return out


@dataclass(frozen=True)
class StageTable:
    """Backward stage table for one resolved tile geometry on one arch.

    Use :meth:`from_geometry` (any object with the ``BwdTileGeometry`` field
    names) or :meth:`from_params`. ``atom_g02`` / ``atom_g13`` / ``atom_g4``
    are the GEMM step K extents (16 or 32); the instruction K is resolved
    from the catalog of ``arch``. ``grid_g02`` / ``grid_g13`` / ``grid_g4`` are
    the warp grids of G0 / G2, G1 / G3 and G4 (``(1, 0)``, ``(0, 1)`` and
    ``(1, 0)`` mean the natural ``(1, W)``, ``(W, 1)`` and ``(1, W)``);
    ``stats_load``, ``dq_issue`` and ``lds_swizzle`` are the body knobs the
    synchronous model reads (where the stats load, how G4 issues, the LDS row
    pads that split vector reads).
    """

    arch: str
    head_size: int
    k_m0: int
    k_n0: int
    k_k4: int
    waves: int
    grid_g4: tuple[int, int] = (1, 0)
    atom_g02: int = 16
    atom_g13: int = 16
    atom_g4: int = 16
    dtype: str = "fp16"
    transpose_source: str = "xt_lds"
    kt_source: str = "reg"
    global_path: str = "vgpr"
    pt_route: str = "relabel"
    stage_vec: int = 8
    dq_mode: str = "atomic"
    sched: str = "stage_table"
    setprio: bool = False
    kv_residency: str = "reg"
    acc_in_lds: bool = False
    # G0 / G2 and G1 / G3 warp grids; (1, 0) / (0, 1) resolve to the natural
    # (1, W) / (W, 1). A G0 / G2 grid may hold a divisor of W (replicated).
    grid_g02: tuple[int, int] = (1, 0)
    grid_g13: tuple[int, int] = (0, 1)
    pipeline: str = "sync"
    stats_load: str = "at_use"
    dq_issue: str = "block"
    lds_swizzle: str | None = None
    # G4 dS key slicing (the body's ``g4_slicing``): the first kK4 dS slice is
    # read in stage C (under G3) only for "slice_db_g3"; otherwise every dS
    # slice is read in stage D. Totals per step are the same for every value.
    g4_slicing: str = "step"
    # The Q / dO staging of the built body: ring depth (with ``global_path``,
    # the prefetch ring), the merged-barrier ring and its issue point, and
    # head packing (the register ring's chunk distribution).
    ring_depth: int = 1
    barrier_merge: str = "off"
    ring_issue: str = "early"
    head_pack: bool = False
    # The counted waits of the synchronous body (``_attention_bwd_counted``):
    # the step stores a tile loaded one step earlier and issues the next
    # step's loads between two fences (after the staging barrier or before G4).
    loop_sync: str = "full"
    counted_issue: str = "early"

    def __post_init__(self) -> None:
        w = int(self.waves)
        for name, natural in (
            ("grid_g4", (1, w)),
            ("grid_g02", (1, w)),
            ("grid_g13", (w, 1)),
        ):
            if tuple(getattr(self, name)) in ((1, 0), (0, 1)):
                object.__setattr__(self, name, natural)
            else:
                object.__setattr__(self, name, tuple(getattr(self, name)))
        w02 = self.grid_g02[0] * self.grid_g02[1]
        if w02 < 1 or w % w02:
            raise ValueError(f"warp grid G0 {self.grid_g02} does not divide the waves")
        if self.grid_g13[0] * self.grid_g13[1] != w:
            raise ValueError(f"warp grid G1 {self.grid_g13} does not hold the waves")
        if self.sched in STAGE_TABLE_SCHEDS and w02 != w:
            raise ValueError("a stage table needs a G0 grid of every wave")
        g4r, g4c = self.grid_g4
        d, m0, n0, k4 = self.head_size, self.k_m0, self.k_n0, self.k_k4
        if self.sched not in SCHEDS:
            raise ValueError(f"unknown sched {self.sched!r}; expected one of {SCHEDS}")
        if self.setprio and self.sched not in STAGE_TABLE_SCHEDS:
            raise ValueError("setprio is legal only inside a stage table")
        if self.pipeline not in PIPELINES:
            raise ValueError(f"pipeline must be one of {PIPELINES}")
        self._check_staging()
        if self.stats_load not in ("at_use", "step_top"):
            raise ValueError("stats_load must be 'at_use' or 'step_top'")
        if self.dq_issue not in ("block", "fragment", "fragment_regs"):
            raise ValueError("dq_issue must be 'block', 'fragment' or 'fragment_regs'")
        g0r, g0c = self.grid_g02
        g1r, g1c = self.grid_g13
        if self.transpose_source not in _TRANSPOSE_SOURCES:
            raise ValueError(f"unknown transpose_source {self.transpose_source!r}")
        if self.kt_source not in ("reg", "lds"):
            raise ValueError("kt_source must be 'reg' or 'lds'")
        if self.kv_residency not in ("reg", "lds"):
            raise ValueError("kv_residency must be 'reg' or 'lds'")
        if self.kt_source == "reg" and self.kv_residency != "reg":
            raise ValueError("kt_source='reg' requires kv_residency='reg'")
        if not isinstance(self.acc_in_lds, bool):
            raise ValueError("acc_in_lds must be a bool")
        if self.global_path not in ("vgpr", "dma"):
            raise ValueError("global_path must be 'vgpr' or 'dma'")
        if self.pt_route not in ("relabel", "lds"):
            raise ValueError("pt_route must be 'relabel' or 'lds'")
        if self.dq_mode not in ("atomic", "split"):
            raise ValueError("dq_mode must be 'atomic' or 'split'")
        if self.stage_vec not in (1, 8):
            raise ValueError("stage_vec must be 8 or 1")
        if self.g4_slicing not in _G4_SLICINGS:
            raise ValueError(f"g4_slicing must be one of {_G4_SLICINGS}")
        if g4r * g4c != w:
            raise ValueError(f"warp grid G4 {self.grid_g4} does not hold {w} waves")
        # The count model splits kN0 over the W waves (the natural grids); a
        # body without a stage table (sched "none" etc.) also runs the
        # small-tile grids of _attention_bwd_tiles, where kN0 < 16 * W.
        counted = self.sched in STAGE_TABLE_SCHEDS
        checks = (
            (not counted or n0 % (16 * w) == 0, f"kN0 {n0} % (16 * waves {w}) != 0"),
            (m0 % 16 == 0, f"kM0 {m0} % 16 != 0"),
            (d % (16 * g4c) == 0, f"D {d} % (16 * g4c {g4c}) != 0"),
            (m0 % (16 * g4r) == 0, f"kM0 {m0} % (16 * g4r {g4r}) != 0"),
            (n0 % k4 == 0, f"kN0 {n0} % kK4 {k4} != 0"),
            (k4 % self.atom_g4 == 0, f"kK4 {k4} % atom_g4 K {self.atom_g4} != 0"),
            (d % self.atom_g02 == 0, f"D {d} % atom_g02 K {self.atom_g02} != 0"),
            (m0 % self.atom_g13 == 0, f"kM0 {m0} % atom_g13 K {self.atom_g13} != 0"),
            (self.atom_g13 != 32 or m0 >= 32, "atom_g13 K = 32 needs kM0 >= 32"),
            (m0 % (16 * g0r) == 0, f"kM0 {m0} % (16 * g0r {g0r}) != 0"),
            (n0 % (16 * g0c) == 0, f"kN0 {n0} % (16 * g0c {g0c}) != 0"),
            (n0 % (16 * g1r) == 0, f"kN0 {n0} % (16 * g1r {g1r}) != 0"),
            (d % (16 * g1c) == 0, f"D {d} % (16 * g1c {g1c}) != 0"),
        )
        for ok, why in checks:
            if not ok:
                raise ValueError(f"illegal backward stage-table geometry: {why}")
        for k in (self.atom_g02, self.atom_g13, self.atom_g4):
            instruction_k(self.arch, k, self.dtype)  # raises when not implementable

    def _check_staging(self) -> None:
        """The staging fields agree with the count model (``"ck"`` is the
        reference placement and reads none of them)."""
        if self.ring_depth not in (1, 2, 3):
            raise ValueError("ring_depth must be 1, 2 or 3")
        if self.barrier_merge not in ("off", "on"):
            raise ValueError("barrier_merge must be 'off' or 'on'")
        if self.ring_issue not in ("early", "late"):
            raise ValueError("ring_issue must be 'early' or 'late'")
        if self.loop_sync not in ("full", "lds_only", "counted"):
            raise ValueError("loop_sync must be 'full', 'lds_only' or 'counted'")
        if self.counted_issue not in ("early", "late"):
            raise ValueError("counted_issue must be 'early' or 'late'")
        if self.pipeline == "ck":
            return
        if self.dq_mode == "split":
            # no G4: no atomics to defer, and neither the merged barrier nor
            # the counted waits (both are placed against G4 and its atomics)
            if self.sched == "stage_table_atomics":
                raise ValueError(SPLIT_DEFERRED)
            if self.barrier_merge == "on" or self.counted:
                raise ValueError(
                    "the split-dQ main kernel runs on the synchronous body or a "
                    "prefetch ring (barrier_merge='off', loop_sync != 'counted')"
                )
        if self.counted:
            if self.pipeline != "sync":
                raise ValueError("loop_sync='counted' runs on the synchronous body")
            if self.sched == "stage_table_atomics" and self.counted_issue == "late":
                raise ValueError(COUNTED_LATE_DEFERRED)
        merged = self.barrier_merge == "on"
        staged = (self.global_path, self.ring_depth) != ("vgpr", 1)
        want = "merged" if merged else ("ring" if staged else "sync")
        if self.pipeline != want:
            raise ValueError(
                f"pipeline={self.pipeline!r} does not model global_path="
                f"{self.global_path!r}, ring_depth={self.ring_depth}, barrier_merge="
                f"{self.barrier_merge!r} (that body is {want!r})"
            )
        if merged and (self.ring_depth, self.global_path) != (2, "vgpr"):
            raise ValueError(
                "barrier_merge='on' needs ring_depth=2, global_path='vgpr'"
            )
        if self.pipeline == "ring" and self.global_path == "dma":
            if self.stage_vec != 8 or self.head_pack:
                raise ValueError("the DMA ring needs stage_vec = 8 and no head packing")
        if self.pipeline != "sync" and self.stats_load != "at_use":
            raise ValueError("a ring stages the stats: stats_load must be 'at_use'")

    # ----- construction ---------------------------------------------------

    @classmethod
    def from_params(cls, **kwargs: Any) -> StageTable:
        return cls(**kwargs)

    @classmethod
    def from_geometry(
        cls,
        geom: Any,
        arch: str,
        *,
        dtype: str = "fp16",
        stage_vec: int = 8,
        dq_mode: str = "atomic",
        acc_in_lds: bool = False,
    ) -> StageTable:
        """Read a ``BwdTileGeometry``-like object (duck-typed attributes).

        ``acc_in_lds`` is a family-spec field rather than a geometry field;
        an ``acc_in_lds`` attribute on ``geom`` takes precedence over the
        keyword. The count model is the built body's: ``"merged"`` with
        ``barrier_merge = "on"``, ``"ring"`` for the prefetch ring (the DMA
        path, or ``ring_depth >= 2``) and ``"sync"`` otherwise.
        """
        atoms = getattr(geom, "atoms", None)
        grid = getattr(geom, "grid_g4", None)
        g02 = getattr(geom, "grid_g0", None)
        g13 = getattr(geom, "grid_g1", None)
        depth = int(getattr(geom, "ring_depth", 1) or 1)
        path = str(getattr(geom, "global_path", "vgpr"))
        merge = str(getattr(geom, "barrier_merge", "off"))
        if merge == "on":
            pipeline = "merged"
        elif (path, depth) != ("vgpr", 1):
            pipeline = "ring"
        else:
            pipeline = "sync"
        return cls(
            arch=arch,
            head_size=int(geom.head_size),
            k_m0=int(geom.k_m0),
            k_n0=int(geom.k_n0),
            k_k4=int(geom.k_k4),
            waves=int(geom.waves),
            grid_g4=tuple(grid) if grid is not None else (1, 0),
            atom_g02=_atom_k(_atom_from(atoms, "g02", "g0", "atom_g02")),
            atom_g13=_atom_k(_atom_from(atoms, "g13", "g1", "atom_g13")),
            atom_g4=_atom_k(_atom_from(atoms, "g4", "atom_g4")),
            dtype=dtype,
            transpose_source=str(getattr(geom, "transpose_source", "xt_lds")),
            kt_source=str(getattr(geom, "kt_source", "reg")),
            global_path=path,
            pt_route=str(getattr(geom, "pt_route", "relabel")),
            stage_vec=int(getattr(geom, "stage_vec", stage_vec)),
            dq_mode=str(getattr(geom, "dq_mode", dq_mode)),
            sched=str(getattr(geom, "sched", "stage_table")),
            setprio=bool(getattr(geom, "setprio", False)),
            kv_residency=str(getattr(geom, "kv_residency", "reg")),
            acc_in_lds=bool(getattr(geom, "acc_in_lds", acc_in_lds)),
            grid_g02=tuple(g02) if g02 is not None else (1, 0),
            grid_g13=tuple(g13) if g13 is not None else (0, 1),
            pipeline=pipeline,
            stats_load=str(getattr(geom, "stats_load", "at_use")),
            dq_issue=str(getattr(geom, "dq_issue", "block")),
            lds_swizzle=getattr(geom, "lds_swizzle", None),
            g4_slicing=str(getattr(geom, "g4_slicing", "step")),
            ring_depth=depth,
            barrier_merge=merge,
            ring_issue=str(getattr(geom, "ring_issue", "early")),
            head_pack=bool(getattr(geom, "head_pack", False)),
            loop_sync=str(getattr(geom, "loop_sync", "full")),
            counted_issue=str(getattr(geom, "counted_issue", "early")),
        )

    # ----- arch facts -----------------------------------------------------

    @property
    def wave_size(self) -> int:
        return int(ArchTarget.from_gfx(self.arch).wave_size)

    def _op(self, k: int):
        for op in _mma_ops(self.arch, self.dtype):
            if op.k == k:
                return op
        raise ValueError(f"no native 16x16x{k} atom on {self.arch}")

    @property
    def instr_k(self) -> dict[str, int]:
        """Instruction K per GEMM after splitting non-native K = 32 steps."""
        k02 = instruction_k(self.arch, self.atom_g02, self.dtype)
        k13 = instruction_k(self.arch, self.atom_g13, self.dtype)
        k4 = instruction_k(self.arch, self.atom_g4, self.dtype)
        return {"g0": k02, "g1": k13, "g2": k02, "g3": k13, "g4": k4}

    def mma_op_id(self, gemm: str) -> str:
        return self._op(self.instr_k[gemm]).op_id

    def frag_len(self, gemm: str) -> int:
        return int(self._op(self.instr_k[gemm]).a_frag_len)

    @property
    def c_frag_len(self) -> int:
        return int(self._op(16).c_frag_len)

    # ----- MMA counts -----------------------------------------------------

    @property
    def mfma(self) -> GemmCounts:
        """MMA instructions per wave per q step, per GEMM, from the warp grids."""
        w, d, m0, n0 = self.waves, self.head_size, self.k_m0, self.k_n0
        g4r, g4c = self.grid_g4
        k = self.instr_k
        # every wave of a grid owns the same number of 16 x 16 tiles
        w02 = self.grid_g02[0] * self.grid_g02[1]
        g0 = (m0 // 16) * (n0 // 16) // w02 * (d // k["g0"])
        g1 = (n0 // 16) * (d // 16) // w * (m0 // k["g1"])
        g4 = 0
        if self.dq_mode == "atomic":
            g4 = (m0 // (16 * g4r)) * (d // (16 * g4c)) * (n0 // k["g4"])
        return GemmCounts(g0=g0, g1=g1, g2=g0, g3=g1, g4=g4)

    @property
    def stage_names(self) -> tuple[str, ...]:
        """The stages of the built step in order: A..D, or A..C for the
        split-dQ main kernel (no G4; C closes the step). The reference
        placement (``"ck"``) keeps its four stages (D empty without G4)."""
        if self.dq_mode == "split" and self.pipeline != "ck":
            return SPLIT_STAGES
        return STAGES

    @property
    def dq_atomics_per_lane(self) -> int:
        """fp32 dQ atomics per lane per q step: ``kM0 * D / (W * wave_size)``."""
        if self.dq_mode != "atomic":
            return 0
        return self.k_m0 * self.head_size // (self.waves * self.wave_size)

    # ----- memory model ---------------------------------------------------

    def _reads_per_frag(self, gemm: str, source: str) -> int:
        """LDS read instructions for one lane's operand fragment of ``gemm``.

        ``source``: "contig" (K contiguous in LDS: plain image for A operands,
        XT image, or the WMMA LDS transpose), "tr_read" (``ds_read_tr16_b64``,
        four elements per read) or "lds_plain" (one element per read).
        """
        n = self.frag_len(gemm)
        if source == "tr_read":
            return ceil_div(n, 4)
        if source == "lds_plain":
            return n
        return ceil_div(n * _ELEM_BYTES, 16)

    @property
    def _tsrc(self) -> str:
        if self.transpose_source in ("xt_lds", "wmma_lds"):
            return "contig"
        return self.transpose_source

    def _prefetch_loads(self) -> int:
        vec = 8 if self.global_path == "dma" else self.stage_vec
        threads = self.waves * self.wave_size
        tile = ceil_div(self.k_m0 * self.head_size, threads * vec)
        stats = ceil_div(self.k_m0, self.wave_size)
        return 2 * tile + 2 * stats

    def _next_tile_lds_writes(self) -> int:
        if self.global_path == "dma":
            return 0  # the DMA fills LDS; no ds_write
        threads = self.waves * self.wave_size
        tile = ceil_div(self.k_m0 * self.head_size, threads * self.stage_vec)
        n = 2 * tile
        if self.transpose_source == "xt_lds":
            w = default_xt_writer(self.k_m0, self.head_size, threads)
            n += 2 * w.stores_per_thread
        n += 2 * ceil_div(self.k_m0, threads)
        return n

    def _ds_tile_writes(self) -> int:
        tiles = (self.k_m0 // 16) * (self.k_n0 // (16 * self.waves))
        per_tile = 1 if self.wave_size == 64 else self.c_frag_len
        return tiles * per_tile

    def _kv_frag_reads(self, gemm: str) -> int:
        """LDS reads of the wave's resident K (G0) or V (G2) B fragments."""
        if self.kv_residency != "lds":
            return 0
        k = self.instr_k[gemm]
        frags = (self.k_n0 // (16 * self.waves)) * (self.head_size // k)
        return frags * self._reads_per_frag(gemm, "contig")

    def _acc_rmw(self) -> int:
        """DS reads (= DS writes) of one of dK / dV per wave per q step."""
        if not self.acc_in_lds:
            return 0
        tiles = (self.k_n0 // (16 * self.waves)) * (self.head_size // 16)
        return tiles * self.c_frag_len

    def stages(
        self, *, prefetch: bool = True, prev_atomics: bool = True
    ) -> tuple[StageCounts, ...]:
        """Per-stage counts of one q step under :attr:`pipeline`.

        ``prefetch`` (``"ck"`` model only): the step loads / stages the next q
        tile (false on the peeled last step). The built bodies peel no step:
        every step of a ring prefetches (past the loop's last trip the step
        itself), and the synchronous body has no next-tile work.
        ``prev_atomics``: a previous step's dQ atomics are pending; used only
        by ``stage_table_atomics`` (the built bodies always issue a batch in
        A: zeros on the first step).
        """
        if self.pipeline == "ck":
            return self._ck_stages(prefetch=prefetch, prev_atomics=prev_atomics)
        return self._sync_stages(prev_atomics=prev_atomics)

    # ----- synchronous model ----------------------------------------------

    def _align(self, buf: str) -> int:
        """Vector-read alignment (elements, at most 8) of LDS buffer ``buf``.

        Every buffer starts on a 128-byte line; a row pad of 8 bytes
        (``pad8``) leaves 4-element rows boundaries, wider pads and the packed
        layout keep 8.
        """
        sw = _swizzle_pairs(self.lds_swizzle).get(buf, "none")
        if sw.startswith("pad"):
            pitch = 16 + int(sw[3:]) // _ELEM_BYTES  # any multiple of 16 columns
            a = 8
            while a > 1 and pitch % a:
                a //= 2
            return a
        return 8

    def _contig_reads(self, gemm: str, buf: str) -> int:
        """LDS reads of one K-contiguous fragment of ``gemm`` from ``buf``."""
        return ceil_div(self.frag_len(gemm), min(8, self._align(buf)))

    def _transposed_reads(self, gemm: str) -> int:
        """LDS reads of one K-outer fragment of ``gemm`` (the transpose
        reader of the body: ``ds_read_tr16_b64`` per four elements, else one
        element per read)."""
        n = self.frag_len(gemm)
        if self.transpose_source == "tr_read":
            return n // 4
        if self.transpose_source == "xt_lds":
            return ceil_div(n * _ELEM_BYTES, 16)
        return n

    def _frags(self, gemm: str) -> tuple[int, int, int]:
        """``(frags_m, frags_n, k_steps)`` of one wave of ``gemm``."""
        d, m0, n0 = self.head_size, self.k_m0, self.k_n0
        k = self.instr_k
        if gemm in ("g0", "g2"):
            r, c = self.grid_g02
            return m0 // (16 * r), n0 // (16 * c), d // k["g0"]
        if gemm in ("g1", "g3"):
            r, c = self.grid_g13
            return n0 // (16 * r), d // (16 * c), m0 // k["g1"]
        r, c = self.grid_g4
        return m0 // (16 * r), d // (16 * c), n0 // k["g4"]

    def _g13_reads(self, b_hoisted: bool) -> tuple[int, int, int]:
        """``(DS reads, DS writes, lead reads)`` of G1 (or G3) in its stage.

        A operand: registers (relabel) or K-outer LDS reads (``pt_route =
        "lds"``); B operand (dO^T / Q^T): K-outer LDS reads unless read ahead
        (``b_hoisted``). Register accumulators issue K step by K step; LDS
        accumulators fragment by fragment, each with its own operand reads
        and a ``c_frag_len`` read and write of the accumulator image.
        """
        fm, fn, ks = self._frags("g1")
        t = self._transposed_reads("g1")
        a = t if self.pt_route == "lds" else 0
        bb = 0 if b_hoisted else t
        if self.acc_in_lds:
            cf = self.c_frag_len
            return fm * fn * (cf + ks * (a + bb)), fm * fn * cf, cf + a + bb
        return ks * (fm * a + fn * bb), 0, fm * a + fn * bb

    def _g4_reads(self, kt_hoisted: bool) -> tuple[int, int]:
        """``(DS reads, lead reads)`` of G4: dS (A, K-contiguous) and K^T (B,
        K-outer, when from LDS and not read ahead) by the issue granularity."""
        fm, fn, ks = self._frags("g4")
        a = self._contig_reads("g4", "ds")
        bb = 0
        if self.kt_source == "lds" and not kt_hoisted:
            bb = self._transposed_reads("g4")
        issue = "fragment" if self.acc_in_lds else self.dq_issue
        if issue == "block":
            return ks * (fm * a + fn * bb), fm * a + fn * bb
        if issue == "fragment":
            return fm * fn * ks * (a + bb), a + bb
        return fm * ks * a + fm * fn * ks * bb, fm * ks * a + bb

    def stats_loads(self) -> int:
        """lse2 + Dsum workspace loads of one q step (per lane): one of each
        per (G0 row fragment, C slot)."""
        fm, _fn, _ks = self._frags("g0")
        return 2 * fm * self.c_frag_len

    def _sync_stages(self, *, prev_atomics: bool = True) -> tuple[StageCounts, ...]:
        """Counts of the built q step (see the module docstring).

        Under a fenced sched the step loads the stats in A and reads the
        dO^T (A), Q^T (B) and LDS K^T (C) fragments one stage ahead; under
        ``none`` / ``iglp*`` (no groups) the counts follow the default
        placement (stats with the softmax in B, each operand where it is
        used). The rings (``"ring"``, ``"merged"``) read the stats from
        their slot (DS reads, spread rather than leading) and add their own
        prefetch and slot stores (:meth:`_ring_terms`).

        The split-dQ main kernel (no G4) returns A, B and C: the relabel
        hand-off writes no dS tile (the LDS route still writes P and dS for
        G1 / G3), no K^T is read ahead, and the ring's slot stores that close
        the step fall in C's region.
        """
        mf = self.mfma
        hoist = self.sched in FENCED_SCHEDS
        deferred = self.sched == "stage_table_atomics"
        slot_stats = self.pipeline in ("ring", "merged")
        k_buf = "k_res" if "lds" in (self.kv_residency, self.kt_source) else "k_stage"
        fm0, fn0, ks0 = self._frags("g0")
        q_r = fm0 * self._contig_reads("g0", "q")
        do_r = fm0 * self._contig_reads("g2", "do")
        k_r = v_r = 0
        if self.kv_residency == "lds":
            k_r = fn0 * self._contig_reads("g0", k_buf)
            v_r = fn0 * self._contig_reads("g2", "v_res")
        stats = self.stats_loads() if self.stats_load == "at_use" else 0
        if self.counted:
            stats = 0  # the counted step's stats are the staged registers
        # (VMEM reads of the workspace, or DS reads of the ring's stats slot)
        vm_stats, ds_stats = (0, stats) if slot_stats else (stats, 0)
        ring = self._ring_terms()
        fm1, fn1, ks1 = self._frags("g1")
        bt = fn1 * ks1 * self._transposed_reads("g1")  # dO^T or Q^T, all K steps
        cf = self.c_frag_len
        tile_w = fm0 * fn0 * cf  # one C-layout tile of G0 / G2, scalar stores
        # the LDS route writes P and dS (G1 / G3 read them back); the relabel
        # route writes the dS tile of G4 only
        ds_w = tile_w * (2 if self.pt_route == "lds" else (1 if mf.g4 else 0))
        g13_r, g13_w, g13_lead = self._g13_reads(b_hoisted=hoist)
        fm4, fn4, ks4 = self._frags("g4")
        kt_ahead = 0
        if hoist and self.kt_source == "lds" and mf.g4:
            kt_ahead = fn4 * ks4 * self._transposed_reads("g4")
        g4_r, g4_lead = self._g4_reads(kt_hoisted=hoist) if mf.g4 else (0, 0)
        atoms = self.dq_atomics_per_lane
        a_w = atoms if deferred and prev_atomics else 0
        a = StageCounts(
            "A",
            mf.g0,
            vmem_read=(vm_stats if hoist else 0) + ring["a_vmem_read"],
            vmem_write=a_w,
            ds_read=ks0 * (q_r + k_r)
            + (bt if hoist else 0)
            + (ds_stats if hoist else 0),
            lead=((VMEM_READ, vm_stats if hoist else 0), (DS_READ, q_r + k_r)),
            fixed=ring["a_fixed"],
        )
        b = StageCounts(
            "B",
            mf.g2 + mf.g1,
            vmem_read=0 if hoist else vm_stats,
            ds_read=ks0 * (do_r + v_r)
            + g13_r
            + (bt if hoist else 0)
            + (0 if hoist else ds_stats),
            ds_write=ds_w + g13_w,
            lead=((DS_READ, do_r + v_r),),
            spans=((DS_WRITE, mf.g1),) if mf.g1 else (),
        )
        c = StageCounts(
            "C",
            mf.g3,
            ds_read=g13_r + kt_ahead,
            ds_write=g13_w,
            lead=((DS_READ, g13_lead),),
        )
        if self.stage_names == SPLIT_STAGES:
            # no G4: C closes the step (the ring's slot stores follow G3)
            return a, b, replace(c, fixed=ring["d_fixed"])
        d = StageCounts(
            "D",
            mf.g4,
            vmem_write=0 if deferred else atoms,
            ds_read=g4_r,
            lead=((DS_READ, g4_lead),),
            spans=((VMEM_WRITE, fm4 * fn4),) if mf.g4 and not deferred else (),
            fixed=ring["d_fixed"],
        )
        return a, b, c, d

    # ----- the rings ------------------------------------------------------

    @property
    def _threads(self) -> int:
        return self.waves * self.wave_size

    def _chunk_loads(self) -> int:
        """Global load ops of one 8-element Q / dO chunk (``stage_vec`` 8:
        one 16-byte vector; 1: eight scalars)."""
        return 1 if self.stage_vec == 8 else 8

    def _chunk_stores(self, buf: str) -> int:
        """LDS store ops of one 8-element chunk into ``buf`` (a row pad that
        breaks the 16-byte alignment splits it)."""
        return ceil_div(8, min(8, self._align(buf)))

    def _ring_reg_chunks(self) -> tuple[int, bool]:
        """``(chunks, joint)``: 8-element Q / dO chunks one thread of the
        register ring moves per tile (``QRing.n_reg_chunks``), and whether
        they are one joint pass over Q and dO."""
        total = self.k_m0 * self.head_size // 8
        threads = self._threads
        joint = not self.head_pack and total < threads and total % self.wave_size == 0
        return (1 if joint else 2 * ceil_div(total, threads)), joint

    def ring_prefetch(self) -> dict[str, Any]:
        """The prefetch of one q step of a ring (``{}`` for ``"sync"``).

        ``issue``: ``"dma"`` (the copy into the slot at the step top),
        ``"loads"`` (the register ring's global loads), ``"merged"`` (the
        merged ring's global loads). ``vmem_read``: global load ops per
        thread (register paths); ``ds_write``: their slot stores;
        ``dma_per_wave``: ``(fewest, most)`` copy instructions of a wave
        (DMA). ``where``: ``"A"`` (in stage A, grouped), ``"top"`` (ahead
        of the step: a separate region behind the ring's ``sched_barrier(0)``
        for the register ring, the head of A's region for the merged ring)
        or ``"barrier"`` (the merged ring's ``ring_issue = "late"``: right
        before the merged barrier, in D's region).
        """
        if self.pipeline not in ("ring", "merged"):
            return {}
        table = self.sched in STAGE_TABLE_SCHEDS
        if self.pipeline == "ring" and self.global_path == "dma":
            from ._attention_bwd_ring import DmaBatch  # (imports the body)

            batch = DmaBatch(self.k_m0, self.head_size, self.waves, self.wave_size)
            per = [len(batch.of_wave(w)) for w in range(self.waves)]
            return {
                "issue": "dma",
                "where": "top",
                "vmem_read": 0,
                "ds_write": 0,
                "dma_per_wave": (min(per), max(per)),
                "dma_x4": 2 * batch.pieces,
                "dma_x1": 2,
            }
        stats_stores = 2  # one lse2 and one Dsum row per thread below kM0
        if self.pipeline == "ring":
            chunks, joint = self._ring_reg_chunks()
            if joint:
                stores = 1  # one wave-uniform select of the Q / dO view
            else:
                half = chunks // 2
                stores = half * (self._chunk_stores("q") + self._chunk_stores("do"))
            return {
                "issue": "loads",
                "where": "A" if table else "top",
                "vmem_read": chunks * self._chunk_loads() + 2,
                "ds_write": stores + stats_stores,
            }
        per_tensor = ceil_div(self.k_m0 * self.head_size // 8, self._threads)
        late = self.ring_issue == "late"
        return {
            "issue": "merged",
            "where": "barrier" if late else ("A" if table else "top"),
            "vmem_read": 2 * per_tensor * self._chunk_loads() + 2,
            "ds_write": per_tensor
            * (self._chunk_stores("q") + self._chunk_stores("do"))
            + stats_stores,
        }

    def _ring_terms(self) -> dict[str, Any]:
        """Where the ring's (or the counted step's) own memory ops fall in
        the stage regions."""
        out: dict[str, Any] = {"a_vmem_read": 0, "a_fixed": (), "d_fixed": ()}
        cnt = self.counted_terms()
        if cnt and cnt["issue"] == "late":
            # the staged tile's stores open A's region (no fence before A)
            out["a_fixed"] = ((DS_WRITE, cnt["stores"]),)
        pf = self.ring_prefetch()
        if not pf:
            return out
        where = pf["where"]
        if where == "A":
            out["a_vmem_read"] = pf["vmem_read"]
        elif pf["issue"] == "merged" and where == "top":
            out["a_fixed"] = ((VMEM_READ, pf["vmem_read"]),)
        d_fixed = []
        if pf["issue"] == "merged" and where == "barrier":
            d_fixed.append((VMEM_READ, pf["vmem_read"]))
        if pf["ds_write"]:
            d_fixed.append((DS_WRITE, pf["ds_write"]))
        out["d_fixed"] = tuple(d_fixed)
        return out

    def prefetch_region(self) -> dict[int, int] | None:
        """IR ops per class of the register ring's prefetch when it is issued
        ahead of the step behind its own ``sched_barrier(0)`` (no stage
        table), else ``None`` (no such region)."""
        pf = self.ring_prefetch()
        if pf.get("issue") == "loads" and pf["where"] == "top":
            return {VMEM_READ: pf["vmem_read"]}
        return None

    @property
    def prefetch_fence(self) -> bool:
        """Whether the step emits the register ring's prefetch fence."""
        return self.prefetch_region() is not None

    def ring_waits(self) -> tuple[int, ...]:
        """The explicit ``s_waitcnt vmcnt`` immediates of one q step: the
        DMA ring's wait at the step top (``R = 1``) or end (``R >= 2``);
        none for the register rings (the compiler waits for its loads)."""
        pf = self.ring_prefetch()
        if pf.get("issue") != "dma":
            return ()
        from ._attention_bwd_ring import dma_wait_counts

        w = dma_wait_counts(
            self.ring_depth, pf["dma_per_wave"][0], self.ring_atomics_per_lane
        )
        return (w.step_top,) if self.ring_depth == 1 else (w.step_end,)

    @property
    def ring_atomics_per_lane(self) -> int:
        """The dQ atomics per lane and q step the DMA wait model counts
        (``n_a`` of :func:`._attention_bwd_ring.dma_wait_counts`)."""
        return self.dq_atomics_per_lane

    def vm_after_prefetch(self, *, prev_atomics: bool = True) -> int:
        """VMEM ops a wave issues in the step after its DMA copy and before
        the step's wait: every VMEM op of the stages (the copy is the
        step's first VMEM op). The DMA wait immediates hold when this is at
        least :attr:`ring_atomics_per_lane`."""
        total = 0
        for st in self.stages(prev_atomics=prev_atomics):
            c = st.census()
            total += c[VMEM_READ] + c[VMEM_WRITE]
        return total

    # ----- the counted waits ----------------------------------------------

    @property
    def counted(self) -> bool:
        return self.loop_sync == "counted"

    def counted_terms(self) -> dict[str, Any]:
        """The counted step's own memory ops (``{}`` without the counted
        waits): ``issue`` (``counted_issue``), ``loads`` (the next step's
        global loads: its lse2 / Dsum rows and Q / dO chunks, between two
        ``sched_barrier(0)`` fences) and ``stores`` (the staged Q / dO tile
        written at the step top, before the staging barrier)."""
        if not self.counted:
            return {}
        chunks = ceil_div(self.k_m0 * self.head_size // 8, self._threads)
        return {
            "issue": self.counted_issue,
            "loads": self.stats_loads() + 2 * chunks * self._chunk_loads(),
            "stores": chunks * (self._chunk_stores("q") + self._chunk_stores("do")),
        }

    def counted_waits(self) -> tuple[int, ...]:
        """``vmcnt`` of the counted step's three barriers (staging, dS or
        P / dS, step end): the dQ atomics issued after the last global load
        (``_attention_bwd_counted.CountedWaits``). Early issue: the previous
        step ends ``n`` atomics after its loads; the dS barrier follows this
        step's loads and precedes its atomics in D (``0``), or follows the
        deferred atomics of stage A (``n``). Late issue: ``n`` at every
        barrier (the loads precede G4 and its atomics)."""
        if not self.counted:
            return ()
        n = self.dq_atomics_per_lane
        if self.counted_issue == "late":
            return (n, n, n)
        return (n, n if self.sched == "stage_table_atomics" else 0, n)

    def step_waits(self) -> tuple[int, ...]:
        """Every explicit ``s_waitcnt vmcnt`` of one q step in program order:
        the DMA ring's (:meth:`ring_waits`) or the counted barriers'
        (:meth:`counted_waits`)."""
        return self.ring_waits() + self.counted_waits()

    def hint_sequence(self) -> list[tuple]:
        """The hint ops of one q step in program order: ``("fence", 0)``
        (the register ring's prefetch fence, the counted step's pair around
        its next-step loads, the stage fences), ``("iglp", level)``,
        ``("prio", level)``, ``("group", mask, count)``."""
        out: list[tuple] = []
        cnt = self.counted_terms()
        if self.prefetch_fence:
            out.append(("fence", 0))
        if cnt.get("issue") == "early":
            out += [("fence", 0), ("fence", 0)]
        if self.sched in IGLP_SCHEDS:
            out.append(("iglp", int(self.sched[-1])))
        last = self.stage_names[-1]
        for st in self.stages():
            if st.stage == "D" and cnt.get("issue") == "late" and self.mfma.g4:
                out += [("fence", 0), ("fence", 0)]
            if self.setprio:
                out.append(("prio", 1))
            if self.sched in STAGE_TABLE_SCHEDS:
                out += [("group", m, c) for m, c in st.groups()]
            if self.setprio:
                out.append(("prio", 0))
            if self.sched in FENCED_SCHEDS and st.stage != last:
                out.append(("fence", 0))
        return out

    def regions(self) -> list[tuple[str, dict[int, int], dict[int, int] | None]]:
        """The ``sched_barrier(0)``-delimited regions of one q step under a
        fenced sched, in program order: ``(name, IR ops per class, leading
        reads or None)``. ``"prefetch"``: the register ring's loads behind
        its fence; ``"staging"`` / ``"issue"``: the counted step's staged
        stores and barrier, and its fenced next-step loads (``"barrier"``:
        the gap before a late issue); ``A`` .. ``D``: the stages (their
        census includes :attr:`StageCounts.fixed`). The synchronous step's
        staging (before its first barrier) is not modelled: its A region
        starts after that barrier."""
        if self.sched not in FENCED_SCHEDS:
            raise ValueError("regions exist under a fenced sched only")
        out: list[tuple[str, dict[int, int], dict[int, int] | None]] = []
        pre = self.prefetch_region()
        if pre is not None:
            out.append(("prefetch", pre, None))
        cnt = self.counted_terms()
        if cnt.get("issue") == "early":
            out.append(("staging", {DS_WRITE: cnt["stores"]}, None))
            out.append(("issue", {VMEM_READ: cnt["loads"]}, None))
        for st in self.stages():
            if st.stage == "D" and cnt.get("issue") == "late" and self.mfma.g4:
                out.append(("barrier", {}, None))
                out.append(("issue", {VMEM_READ: cnt["loads"]}, None))
            out.append((st.stage, st.census(), dict(st.lead)))
        return out

    def _ck_stages(
        self, *, prefetch: bool = True, prev_atomics: bool = True
    ) -> tuple[StageCounts, ...]:
        """Counts of the reference pipeline (next-tile work, kK4 dS slices)."""
        w, d, m0, n0, kk4 = self.waves, self.head_size, self.k_m0, self.k_n0, self.k_k4
        g4r, g4c = self.grid_g4
        k = self.instr_k
        mf = self.mfma
        t = self._tsrc
        g4_on = self.dq_mode == "atomic"

        # A operands of G0/G2 (next Q / dO), B operands of G1/G3 (dO^T / Q^T).
        bt_frags = (d // 16) * (m0 // k["g1"])
        bt_reads = bt_frags * self._reads_per_frag("g1", t)
        next_a_reads = (
            (m0 // 16) * (d // k["g0"]) * self._reads_per_frag("g0", "contig")
        )
        stats_reads = m0 // 16
        slice_reads = (
            (m0 // (16 * g4r)) * (kk4 // k["g4"]) * self._reads_per_frag("g4", "contig")
        )
        n_slices = n0 // kk4 if g4_on else 0

        a_vmem = self._prefetch_loads() if prefetch else 0
        a_vmem_w = (
            self.dq_atomics_per_lane
            if self.sched == "stage_table_atomics" and prev_atomics
            else 0
        )
        a_ds_r = bt_reads
        a_ds_w = 0
        b_ds_r = bt_reads
        b_ds_w = self._next_tile_lds_writes() if prefetch else 0
        if self.pt_route == "lds":
            a_ds_w += self._ds_tile_writes()  # P to LDS after G0
            pt_frags = (n0 // (16 * w)) * (m0 // k["g1"])
            b_ds_r += pt_frags * self._reads_per_frag("g1", "contig")
        c_ds_w = self._ds_tile_writes() if (g4_on or self.pt_route == "lds") else 0
        # dS slices read in stage C (slice 0 under G3) and the rest in D
        c_slices = 1 if n_slices and self.g4_slicing == "slice_db_g3" else 0
        c_ds_r = c_slices * slice_reads + (
            next_a_reads + stats_reads if prefetch else 0
        )
        if self.pt_route == "lds":
            c_ds_r += (
                (n0 // (16 * w))
                * (m0 // k["g3"])
                * self._reads_per_frag("g3", "contig")
            )
        # Resident K / V fragments and LDS accumulators (zero when absent).
        a_ds_r += self._kv_frag_reads("g0")
        b_ds_r += self._kv_frag_reads("g2")
        b_ds_r += self._acc_rmw()
        b_ds_w += self._acc_rmw()
        c_ds_r += self._acc_rmw()
        c_ds_w += self._acc_rmw()
        d_ds_r = (n_slices - c_slices) * slice_reads
        if g4_on and self.kt_source == "lds":
            kt_frags = (d // (16 * g4c)) * (n0 // k["g4"])
            d_ds_r += kt_frags * self._reads_per_frag("g4", t)
        if prefetch:
            d_ds_r += next_a_reads + stats_reads
        return (
            StageCounts(
                "A",
                mf.g0,
                vmem_read=a_vmem,
                vmem_write=a_vmem_w,
                ds_read=a_ds_r,
                ds_write=a_ds_w,
            ),
            StageCounts("B", mf.g1 + mf.g2, ds_read=b_ds_r, ds_write=b_ds_w),
            StageCounts("C", mf.g3, ds_read=c_ds_r, ds_write=c_ds_w),
            StageCounts("D", mf.g4, ds_read=d_ds_r),
        )

    def stage(self, name: str, **kw: bool) -> StageCounts:
        return self.stages(**kw)[self.stage_names.index(name)]

    def groups(
        self, name: str, *, prefetch: bool = True, prev_atomics: bool = True
    ) -> list[tuple[int, int]]:
        """``(mask, count)`` sched-group sequence of one stage (empty if none)."""
        if self.sched not in STAGE_TABLE_SCHEDS:
            return []
        return self.stage(name, prefetch=prefetch, prev_atomics=prev_atomics).groups()

    # ----- emission -------------------------------------------------------

    def emit_loop_head(self, b: Any) -> None:
        """Once per q step, at the head of its MMA work (the body places it
        right after the Q / dO staging barrier, so the hint lives in the
        region of the step's matrix ops): ``iglp_opt`` for the iglp modes.

        ``iglp_opt(1)`` also marks the kernel to compile in a fresh process
        (``COMGR_FRESH_PROCESS_ATTR``; the LLVM IR is unchanged): LLVM's
        strategy behind it keeps counts across the compiles of a process, so
        in-process its code would depend on what the process compiled before."""
        if self.sched == "iglp0":
            b.iglp_opt(0)
        elif self.sched == "iglp1":
            b.iglp_opt(1)
            b.kernel.attrs[COMGR_FRESH_PROCESS_ATTR] = True

    def emit_stage_begin(self, b: Any, name: str) -> None:
        """Before a stage's ops: raise the wave priority when ``setprio``."""
        if self.setprio:
            b.s_setprio(1)

    def emit_stage_end(
        self, b: Any, name: str, *, prefetch: bool = True, prev_atomics: bool = True
    ) -> None:
        """After a stage's ops: the interleave groups, priority reset, fence
        (none after the step's last stage: D, or C without G4)."""
        if name not in self.stage_names:
            raise ValueError(f"unknown stage {name!r} (stages {self.stage_names})")
        for mask, count in self.groups(
            name, prefetch=prefetch, prev_atomics=prev_atomics
        ):
            b.sched_group_barrier(mask, count, 0)
        if self.setprio:
            b.s_setprio(0)
        if name != self.stage_names[-1] and self.sched in FENCED_SCHEDS:
            b.sched_barrier(0)

    # ----- IR count model -------------------------------------------------

    def ir_count_model(
        self, *, prefetch: bool = True, prev_atomics: bool = True
    ) -> dict[str, Any]:
        """IR op counts one q step's emission produces, plus the MMA model.

        Keys: ``tile.sched_group_barrier``, ``tile.sched_barrier``,
        ``tile.s_setprio``, ``tile.iglp_opt`` (IR op counts of
        :meth:`emit_loop_head` + the stage begin/end calls of
        :attr:`stage_names`, plus the
        register ring's prefetch fence and the counted step's two fences),
        ``tile.s_waitcnt`` (the DMA ring's wait, the counted barriers'),
        ``group_counts`` (``{mask: summed count}`` over the groups),
        ``mfma`` (per GEMM), ``mfma_total``, ``stage_mfma`` (per stage),
        ``ring_waits`` (every wait immediate, :meth:`step_waits`) and
        ``prefetch`` (:meth:`ring_prefetch`).
        """
        groups = 0
        sums: dict[int, int] = {}
        names = self.stage_names
        for name in names:
            for mask, count in self.groups(
                name, prefetch=prefetch, prev_atomics=prev_atomics
            ):
                groups += 1
                sums[mask] = sums.get(mask, 0) + count
        fences = len(names) - 1 if self.sched in FENCED_SCHEDS else 0
        if self.pipeline != "ck":
            fences += int(self.prefetch_fence) + (2 if self.counted else 0)
        st = self.stages(prefetch=prefetch, prev_atomics=prev_atomics)
        waits = self.step_waits() if self.pipeline != "ck" else ()
        return {
            "tile.sched_group_barrier": groups,
            "tile.sched_barrier": fences,
            "tile.s_setprio": 2 * len(names) if self.setprio else 0,
            "tile.iglp_opt": 1 if self.sched in IGLP_SCHEDS else 0,
            "tile.s_waitcnt": len(waits),
            "group_counts": sums,
            "mfma": self.mfma.as_dict(),
            "mfma_total": self.mfma.total,
            "stage_mfma": {s.stage: s.mfma for s in st},
            "pipeline": self.pipeline,
            "ring_waits": waits,
            "prefetch": self.ring_prefetch() if self.pipeline != "ck" else {},
        }

    def describe(self) -> dict[str, Any]:
        """Plain-data summary (for sweep records and the ``--count`` output)."""
        return {
            "arch": self.arch,
            "pipeline": self.pipeline,
            "ring_depth": self.ring_depth,
            "global_path": self.global_path,
            "barrier_merge": self.barrier_merge,
            "sched": self.sched,
            "setprio": self.setprio,
            "kv_residency": self.kv_residency,
            "acc_in_lds": self.acc_in_lds,
            "g4_slicing": self.g4_slicing,
            "instr_k": self.instr_k,
            "mfma": self.mfma.as_dict(),
            "dq_atomics_per_lane": self.dq_atomics_per_lane,
            "stages": {
                s.stage: {
                    "mfma": s.mfma,
                    "vmem_read": s.vmem_read,
                    "vmem_write": s.vmem_write,
                    "ds_read": s.ds_read,
                    "ds_write": s.ds_write,
                }
                for s in self.stages()
            },
        }


# ---------------------------------------------------------------------------
# the split dQ kernel's key step
# ---------------------------------------------------------------------------

# Transpose readers of the dQ kernel's G4' K-outer B operand.
_DQ_TRANSPOSE_SOURCES = ("tr_read", "lds_plain")
_DQ_KV_PADS = (0, 8, 16, 32)


@dataclass(frozen=True)
class DqStageTable:
    """Stage table of one key step of the split dQ kernel
    (``_attention_bwd_dqbody.BwdDqStep.run_kv_step``), ``dq_sched``.

    The CTA holds ``kM0`` query rows (Q^T / dO^T resident in registers, the
    per-column stats formed once); a key step stages ``kN0`` keys of K and V
    in LDS and runs three stages, per wave (memory counts per lane):

    * A: G0' ``S^T = K Q^T`` (grid ``(1, W)``: every key, ``kM0 / W`` query
      columns per wave) with its K-inner K fragment reads and, with
      ``kv_prefetch``, the next key tile's global loads (its K and V chunks:
      they are issued at the step top, ahead of G0', spread over G0''s MFMAs
      by the groups);
    * B: G2' ``dP^T = V dO^T`` with its V reads, then the softmax gradient
      and the dS^T register relabel (VALU only);
    * C: G4' ``dQ += dS K`` (grid ``(W, 1)``) with its K-outer K reads
      (``ds_read_tr16_b64`` per four elements under ``tr_read``, one element
      per read under ``lds_plain``); C closes the step: the barrier that
      retires the step's LDS reads and, with ``kv_prefetch``, the next tile's
      LDS stores (a tail chunk's behind its guard) and their barrier fall in
      its region outside the groups (:attr:`StageCounts.fixed`).

    The plain step's K / V staging copy and its barrier precede A and are
    not modelled (A's region starts after that barrier). The placement does
    not change with the value: ``iglp0`` / ``iglp1`` put one ``iglp_opt``
    ahead of A; the fenced values (:data:`DQ_FENCED_SCHEDS`) put a
    ``sched_barrier(0)`` after A and B; the stage tables add per stage the
    ``sched_group_barrier(mask, count, 0)`` groups of :func:`interleave_groups`
    (one MFMA per group; the operand reads of the first K step lead, the
    other reads and the prefetch loads spread); ``stage_table_setprio``
    raises the wave priority around each stage (``s_setprio`` is a
    scheduling boundary, so A's raise precedes the prefetch loads). Every
    value computes the default's dQ bit for bit (same MMA chains, same
    operands).
    """

    arch: str
    head_size: int
    k_m0: int
    k_n0: int
    waves: int
    atom_g02: int = 16
    atom_g4: int = 16
    dtype: str = "fp16"
    transpose_source: str = "tr_read"
    kv_pad: int = 0
    kv_prefetch: bool = False
    stage_vec: int = 8
    sched: str = "stage_table"

    def __post_init__(self) -> None:
        if self.sched not in DQ_SCHEDS:
            raise ValueError(f"dq sched {self.sched!r} not in {DQ_SCHEDS}")
        if self.transpose_source not in _DQ_TRANSPOSE_SOURCES:
            raise ValueError(
                f"dQ transpose_source must be one of {_DQ_TRANSPOSE_SOURCES}"
            )
        if self.kv_pad not in _DQ_KV_PADS:
            raise ValueError(f"kv_pad must be one of {_DQ_KV_PADS}")
        if self.stage_vec not in (1, 8):
            raise ValueError("stage_vec must be 8 or 1")
        if not isinstance(self.kv_prefetch, bool):
            raise ValueError("kv_prefetch must be a bool")
        w, d, m0, n0 = self.waves, self.head_size, self.k_m0, self.k_n0
        checks = (
            (m0 % (16 * w) == 0, f"kM0 {m0} % (16 * waves {w}) != 0"),
            (n0 % 16 == 0, f"kN0 {n0} % 16 != 0"),
            (d % 16 == 0, f"D {d} % 16 != 0"),
            (d % self.atom_g02 == 0, f"D {d} % atom_g02 K {self.atom_g02} != 0"),
            (n0 % self.atom_g4 == 0, f"kN0 {n0} % atom_g4 K {self.atom_g4} != 0"),
        )
        for ok, why in checks:
            if not ok:
                raise ValueError(f"illegal dQ stage-table geometry: {why}")
        for k in (self.atom_g02, self.atom_g4):
            instruction_k(self.arch, k, self.dtype)  # raises when not implementable

    @classmethod
    def from_geometry(cls, geom: Any) -> DqStageTable:
        """Read a ``BwdDqGeometry``-like object (duck-typed attributes)."""
        return cls(
            arch=geom.arch,
            head_size=int(geom.head_size),
            k_m0=int(geom.k_m0),
            k_n0=int(geom.k_n0),
            waves=int(geom.waves),
            atom_g02=int(geom.atom_g02),
            atom_g4=int(geom.atom_g4),
            dtype=str(geom.dtype),
            transpose_source=str(geom.transpose_source),
            kv_pad=int(getattr(geom, "kv_pad", 0)),
            kv_prefetch=bool(getattr(geom, "kv_prefetch", False)),
            stage_vec=int(getattr(geom, "stage_vec", 8)),
            sched=str(getattr(geom, "sched", "none")),
        )

    # ----- arch facts -----------------------------------------------------

    @property
    def wave_size(self) -> int:
        return int(ArchTarget.from_gfx(self.arch).wave_size)

    @property
    def setprio(self) -> bool:
        return self.sched == "stage_table_setprio"

    @property
    def table(self) -> bool:
        """Whether the value emits ``sched_group_barrier`` groups."""
        return self.sched in ("stage_table", "stage_table_setprio")

    @property
    def fenced(self) -> bool:
        return self.sched in DQ_FENCED_SCHEDS

    @property
    def stage_names(self) -> tuple[str, ...]:
        return DQ_STAGES

    def _frag_len(self, atom_k: int) -> int:
        k = instruction_k(self.arch, atom_k, self.dtype)
        for op in _mma_ops(self.arch, self.dtype):
            if op.k == k:
                return int(op.a_frag_len)
        raise ValueError(f"no native 16x16x{k} atom on {self.arch}")

    @property
    def _kv_align(self) -> int:
        """Vector alignment (elements, at most 8) of the K / V rows in LDS:
        the row pitch ``D + kv_pad / 2`` on a 128-byte line."""
        pitch = self.head_size + self.kv_pad // _ELEM_BYTES
        a = 8
        while a > 1 and pitch % a:
            a //= 2
        return a

    # ----- counts ---------------------------------------------------------

    @property
    def mfma(self) -> dict[str, int]:
        """MMA instructions per wave per key step: ``g0`` (G0'), ``g2``
        (G2'), ``g4`` (G4')."""
        w, d, m0, n0 = self.waves, self.head_size, self.k_m0, self.k_n0
        k02 = instruction_k(self.arch, self.atom_g02, self.dtype)
        k4 = instruction_k(self.arch, self.atom_g4, self.dtype)
        g0 = (n0 // 16) * (m0 // (16 * w)) * (d // k02)
        g4 = (m0 // (16 * w)) * (d // 16) * (n0 // k4)
        return {"g0": g0, "g2": g0, "g4": g4}

    def _kv_reads(self) -> tuple[int, int]:
        """``(DS reads, reads of the first K step)`` of G0' (or G2'): one
        K-inner A fragment per 16 keys and K step."""
        k02 = instruction_k(self.arch, self.atom_g02, self.dtype)
        per = ceil_div(self._frag_len(self.atom_g02), min(8, self._kv_align))
        first = (self.k_n0 // 16) * per
        return (self.head_size // k02) * first, first

    def _kt_reads(self) -> tuple[int, int]:
        """``(DS reads, reads of the first K step)`` of G4': one K-outer B
        fragment per 16 head columns and K step."""
        k4 = instruction_k(self.arch, self.atom_g4, self.dtype)
        n = self._frag_len(self.atom_g4)
        per = n // 4 if self.transpose_source == "tr_read" else n
        first = (self.head_size // 16) * per
        return (self.k_n0 // k4) * first, first

    def _tile_chunks(self) -> int:
        """8-element chunks of one ``kN0 x D`` tile a thread moves."""
        threads = self.waves * self.wave_size
        return ceil_div(self.k_n0 * self.head_size // 8, threads)

    def prefetch(self) -> dict[str, int]:
        """The next key tile's traffic per thread (``{}`` without
        ``kv_prefetch``): ``vmem_read`` (K then V chunks; one 16-byte load,
        or eight scalars at ``stage_vec = 1``) and ``ds_write`` (their LDS
        stores; an 8-byte row pad splits each in two)."""
        if not self.kv_prefetch:
            return {}
        chunks = self._tile_chunks()
        loads = 1 if self.stage_vec == 8 else 8
        stores = ceil_div(8, min(8, self._kv_align))
        return {"vmem_read": 2 * chunks * loads, "ds_write": 2 * chunks * stores}

    def stages(self) -> tuple[StageCounts, ...]:
        """Per-stage counts of one key step (see the class docstring)."""
        mf = self.mfma
        kv_r, kv_lead = self._kv_reads()
        kt_r, kt_lead = self._kt_reads()
        pf = self.prefetch()
        a = StageCounts(
            "A",
            mf["g0"],
            vmem_read=pf.get("vmem_read", 0),
            ds_read=kv_r,
            lead=((DS_READ, kv_lead),),
        )
        b = StageCounts("B", mf["g2"], ds_read=kv_r, lead=((DS_READ, kv_lead),))
        c = StageCounts(
            "C",
            mf["g4"],
            ds_read=kt_r,
            lead=((DS_READ, kt_lead),),
            fixed=((DS_WRITE, pf["ds_write"]),) if pf else (),
        )
        return a, b, c

    def stage(self, name: str) -> StageCounts:
        return self.stages()[DQ_STAGES.index(name)]

    def groups(self, name: str) -> list[tuple[int, int]]:
        """``(mask, count)`` sched-group sequence of one stage (empty unless
        a stage table)."""
        if not self.table:
            return []
        return self.stage(name).groups()

    def hint_sequence(self) -> list[tuple]:
        """The hint ops of one key step in program order (as
        :meth:`StageTable.hint_sequence`)."""
        out: list[tuple] = []
        if self.sched in IGLP_SCHEDS:
            out.append(("iglp", int(self.sched[-1])))
        for name in DQ_STAGES:
            if self.setprio:
                out.append(("prio", 1))
            out += [("group", m, c) for m, c in self.groups(name)]
            if self.setprio:
                out.append(("prio", 0))
            if self.fenced and name != DQ_STAGES[-1]:
                out.append(("fence", 0))
        return out

    def regions(self) -> list[tuple[str, dict[int, int], dict[int, int]]]:
        """The ``sched_barrier(0)``-delimited regions of one key step under
        a fenced value: ``(stage, IR ops per class, leading reads)`` (the
        plain step's staging, before its barrier, is not part of A)."""
        if not self.fenced:
            raise ValueError("regions exist under a fenced dq sched only")
        return [(st.stage, st.census(), dict(st.lead)) for st in self.stages()]

    def ir_count_model(self) -> dict[str, Any]:
        """IR hint op counts of one key step, the group sums and the MMA
        counts."""
        groups = 0
        sums: dict[int, int] = {}
        for name in DQ_STAGES:
            for mask, count in self.groups(name):
                groups += 1
                sums[mask] = sums.get(mask, 0) + count
        return {
            "tile.sched_group_barrier": groups,
            "tile.sched_barrier": len(DQ_STAGES) - 1 if self.fenced else 0,
            "tile.s_setprio": 2 * len(DQ_STAGES) if self.setprio else 0,
            "tile.iglp_opt": 1 if self.sched in IGLP_SCHEDS else 0,
            "tile.s_waitcnt": 0,
            "group_counts": sums,
            "mfma": dict(self.mfma),
            "stage_mfma": {s.stage: s.mfma for s in self.stages()},
            "prefetch": self.prefetch(),
        }

    # ----- emission -------------------------------------------------------

    def emit_loop_head(self, b: Any) -> None:
        """Once per key step, ahead of stage A: ``iglp_opt`` for the iglp
        values. ``iglp_opt(1)`` also marks the kernel to compile in a fresh
        process (``COMGR_FRESH_PROCESS_ATTR``, as the main kernel's
        :meth:`StageTable.emit_loop_head`; the LLVM IR is unchanged)."""
        if self.sched in IGLP_SCHEDS:
            level = int(self.sched[-1])
            b.iglp_opt(level)
            if level == 1:
                b.kernel.attrs[COMGR_FRESH_PROCESS_ATTR] = True

    def emit_stage_begin(self, b: Any, name: str) -> None:
        if name not in DQ_STAGES:
            raise ValueError(f"unknown dQ stage {name!r}")
        if self.setprio:
            b.s_setprio(1)

    def emit_stage_end(self, b: Any, name: str) -> None:
        """After a stage's ops: its groups, the priority reset and (A, B)
        the fence."""
        if name not in DQ_STAGES:
            raise ValueError(f"unknown dQ stage {name!r}")
        for mask, count in self.groups(name):
            b.sched_group_barrier(mask, count, 0)
        if self.setprio:
            b.s_setprio(0)
        if self.fenced and name != DQ_STAGES[-1]:
            b.sched_barrier(0)
