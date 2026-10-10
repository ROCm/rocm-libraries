# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Layout-general SDPA backward (dQ, dK, dV): spec, legality, kernarg ABI.

This module owns the compile-time description of the layout-general backward
family: :class:`AttnBwdSpec` (every tile, residency, path, schedule and codegen
knob as a defaulted field; only ``head_size`` is required), the per-arch default
geometry policy that resolves ``None`` knobs, the static legality rules
(value sets, couplings, the LDS footprint of the phase planner, the two-budget
register estimate), kernel names salted by every non-default compile-time
field, and the ordered kernarg lists of the ``rocke.attn_bwd.v3`` ABI.

Runtime plan values (``g_split``, scale placement multipliers, load-balance
order, XCD remap, head-pack count, work list) are never spec fields; the host
plan in :mod:`kernels.common.attention_bwd_plan` chooses them. The one
compile-time switch among them is ``xcd_remap``, which only decides whether
the decode reads the runtime ``xcd_n`` / ``xcd_chunk`` (the plan sets it with
a non-zero ``xcd_n``), so a kernel without the remap emits no remap IR.

:func:`build_attn_bwd_main` emits the main kernel: the CTA decode, sequence
lengths (batched: ``S_max`` or the per-batch SEQ_LEN counts; THD: the int32 /
int64 ragged offsets, converted to tokens in 64 bits, with the workspace rows
of a sequence starting at its first token or, under the runtime ``ws_seg``,
at its slot ``b * S_max``), user-tensor and workspace addressing, the mask policy (no mask, or
the runtime band: top-left / bottom-right per sequence, two-sided windows),
the one-time K / V staging and the flattened q loop (``step_order``: q tiles
of the inverse band outer and query-head groups inner, or ``"head_outer"``:
head groups outer, q tiles inner) over the shared tile step of
:mod:`kernels.common._attention_bwd_body`, then the dK / dV epilogue. With
``edge_tiles`` the loop runs as edge / interior / edge loops over contiguous
q-tile ranges in the same order (under ``"head_outer"`` inside one loop over
the head groups); with ``head_pack`` a group of query heads
shares the M rows of a step (runtime ``pack_heads``). In direct mode a CTA
owns one kv head and sums the dK / dV contributions of its ``G = h_q / h_k``
query heads on chip (MHA, GQA, MQA; the ``G`` kernarg); every step rebases Q,
dO and the dQ workspace on its query head in 64 bits. In atomic mode
(``dkv_mode = "atomic"``: ``h_k != h_v``, or a runtime ``g_split > 1``) a
CTA owns ``G / g_split`` query heads of a unit of ``G = gcd(h_q / h_k, h_q /
h_v)`` heads, which share one K head and one V head, and adds its fp32 dK / dV
into the ``WS_DK`` / ``WS_DV`` workspace with atomics (two converts follow).
The CTA decode (:mod:`kernels.common._attention_bwd_worklist`) applies the
runtime load-balance order (natural, reverse, or interleave under the
``lb_interleave`` decode; one kv tile per CTA, so the
``pair`` kernarg must be 0 and the host plan refuses mirror pairing).
``stage_vec`` sets the global access width of the user tensors only: 8 moves
16-byte vectors (every B/H/S stride a multiple of 8 elements, bases 16-byte
aligned), 1 moves single elements, so any element stride and any
element-aligned base is served; LDS, matrix ops and workspace accesses are
the same for both, and ``stage_vec = 1`` instances default to the narrow
tiles of :func:`attn_bwd_default_geometry`. Knob values whose emission is
not built yet raise :class:`NotImplementedError`.

Direct-mode epilogue (``dkv_epilogue``): the dK / dV accumulators are scaled
and rounded to the I/O dtype once, then either transposed through an LDS
buffer and stored as rows of ``dkv_store_vec`` elements (``"lds_staged"``,
dK then dV through one buffer; ``"lds_staged_pair"``, both through two
buffers behind one barrier) or stored element by element straight from the
accumulator slots (``"direct"``, no epilogue buffer, no epilogue barrier).
Every variant writes the same values to the same elements. The atomic
epilogue guards each key row with a branch or, under ``dkv_atomic_guard =
"select"``, adds ``+0.0`` for a row outside the sequence or the workspace
(identical sums). ``lds_order`` moves the resident LDS region above the
phase-aliased one; the footprint and every access pattern inside a buffer
stay the same. :data:`INERT_KNOBS` names the values that change nothing
under a given spec.

Register model (CDNA wave64 and RDNA wave32, per lane, 4-byte registers, two
16-bit operands per register), used by :func:`register_estimates`:

* resident K, V (``kv_residency = "reg"``) and K^T (``kt_source = "reg"``):
  ``kN0*D / (2*wave*W)`` each;
* dK, dV fp32 accumulators: ``kN0*D / (wave*W)`` each (AGPRs on MFMA unless
  ``agpr_alloc = (0, 0)``; none in registers when ``acc_in_lds``). Under the
  attr route (an AGPR form without a split ``agpr_route``) the backend also
  keeps S / dP and the dQ partial in AGPRs (see :func:`_agpr_placement`);
* per-step operands (Q and dO as A and as B): ``2*kM0*D / (2*wave)``;
* register prefetch of the next Q / dO (``global_path = "vgpr"`` with
  ``ring_depth >= 2``, or the gfx942 one-deep prefetch under a stage table):
  ``kM0*D / (wave*W)``;
* S and dP fp32 accumulators ``2*kM0*kN0 / (wave*W)`` (AGPRs when
  ``s_in_agpr``) plus the P and dS operands ``kM0*kN0 / (wave*W)``;
* dQ partial ``kM0*D / (wave*W)``; dS operands for G4 ``2*kM0*k4a / (2*wave)``
  (``k4a`` = the G4 atom K: G4 is emitted one atom-K step at a time, so the
  live operand does not grow with kK4);
* an allowance of 24 for addresses, loop state, lse2 / Dsum and masks.
"""

from __future__ import annotations

import dataclasses
import hashlib
from dataclasses import dataclass, fields
from functools import cache
from types import MappingProxyType
from typing import Optional

from rocke.core.arch import ArchTarget
from rocke.core.codegen_policy import (
    SCHEDULER_STRATEGY_ATTR,
    normalize_scheduler_strategy,
)
from rocke.core.ir import F32, I32, I64, IRBuilder, PtrType

from rocke.helpers.attention_band import AttnRuntimeBounds

from kernels.common._attention_bwd_addr import (
    PackedRows,
    StridedTensorAddr,
    WorkspaceAddr,
    decode_batched,
    decode_thd,
)
from kernels.common._attention_bwd_band import (
    NoMaskKTail,
    RuntimeBand,
    dkv_store_rule,
)
from kernels.common._attention_bwd_body import (
    DEFAULT_STAGE_ORDER,
    STAGE_ORDERS,
    COPY_ORDERS,
    BwdLds,
    BwdTileGeometry,
    BwdTileStep,
    DkDvAcc,
    PackedStep,
    StatsView,
)
from kernels.common._attention_bwd_frag import ir_type
from kernels.common._attention_bwd_gemm import lane_and_wave
from kernels.common._attention_bwd_lds import (
    DKV_EPILOGUES,
    LDS_ORDERS,
    BwdLdsRequest,
    LdsPlan,
    plan_bwd_lds,
)
from kernels.common._attention_bwd_sinks import (
    AtomicWorkspaceSink,
    PackedAtomicWorkspaceSink,
)
from kernels.common._attention_bwd_worklist import XcdRemap, decode_cta, head_range

__all__ = [
    "ATTN_BWD_ABI",
    "BWD_ARCHS",
    "BWD_DTYPES",
    "BWD_HEAD_SIZES",
    "AttnBwdSpec",
    "BwdArchFacts",
    "attn_bwd_arch_facts",
    "attn_bwd_default_geometry",
    "attn_bwd_params",
    "NOT_YET_EFFECTIVE_KNOBS",
    "DECLARATION_ONLY",
    "DKV_ATOMIC_GUARDS",
    "DKV_EPILOGUES",
    "DKV_STORE_VECS",
    "INERT_KNOBS",
    "LDS_ORDERS",
    "AGPR_ROUTES",
    "ASM_CLUSTERS",
    "LOOP_SYNCS",
    "STEP_ORDERS",
    "agpr_copy_budget",
    "build_attn_bwd_main",
    "lds_footprint_bytes",
    "lds_plan",
    "lds_request",
    "register_estimates",
    "static_unit_mults",
    "validate_attn_bwd_spec",
]

ATTN_BWD_ABI = "rocke.attn_bwd.v3"

# Targets the family knows (all four have a hardware validation run).
BWD_ARCHS = ("gfx942", "gfx950", "gfx1151", "gfx1201")
BWD_DTYPES = ("fp16", "bf16")
BWD_HEAD_SIZES = (32, 64, 128)

SEQ_MODES = ("batched", "thd")
DKV_MODES = ("direct", "atomic")
STAGE_VECS = (8, 1)
MASK_CLASSES = ("band", "none")
DQ_MODES = ("atomic", "split")
WS_LAYOUTS = ("head_major", "token_major")
# ``static_scale``: the plan's scale placements (attention_bwd_plan
# SCALE_PLACEMENTS), or "runtime" (one kernel for every placement).
STATIC_SCALES = ("fold_ds", "dq_before_atomic", "convert", "runtime")
BLOCK_M_VALUES = (16, 32, 64)
BLOCK_N_VALUES = (16, 32, 64, 128, 192, 256)
BLOCK_K4_VALUES = (16, 32, 64)
ATOMS = ("16x16x16", "16x16x32", "wmma16x16x16")
PT_ROUTES = ("relabel", "lds")
RESIDENCIES = ("reg", "lds")
TRANSPOSE_SOURCES = ("xt_lds", "tr_read", "lds_plain", "wmma_lds")
GLOBAL_PATHS = ("vgpr", "dma")
SWIZZLES = ("xor", "pad8", "pad16", "pad32", "none")
SCHEDS = ("none", "iglp0", "iglp1", "fences_only", "stage_table", "stage_table_atomics")
# Elements per row store of the LDS-staged dK / dV epilogue (and per LDS read
# of the staging buffer or of the fp32 accumulator images).
DKV_STORE_VECS = (8, 4, 2, 1)
# Row guard of the atomic dK / dV epilogue: one branch per key row around its
# atomics ("branch"), or every atomic issued with +0.0 for a row outside the
# sequence or the workspace ("select", branch-free; identical sums).
DKV_ATOMIC_GUARDS = ("branch", "select")
STAGE_TABLE_SCHEDS = ("stage_table", "stage_table_atomics")
# The stage-table count model (``_attention_bwd_sched``) places G1 + G2 in
# stage B, G3 in C and G4 with the dS slices in D, i.e. the default order.
_STAGE_ORDER_SCHED_RULE = (
    "stage_order={!r} is not legal with sched={!r}: the stage-table count "
    "model assumes the default G1, G3, G4 order"
)
# ``None`` is the policy value (resolved by :func:`_policy_agpr_alloc`);
# ``"backend"`` asks for no reservation and leaves the accumulator placement to
# the backend (the emission of a policy result of ``None``) even where the
# policy would pick the VGPR form; ``(n, n)`` reserves n AGPRs.
AGPR_ALLOC_BACKEND = "backend"
AGPR_ALLOCS = (None, AGPR_ALLOC_BACKEND, (0, 0), (128, 128), (192, 192), (256, 256))
# Register plan of an AGPR reservation (n, n), n > 0. ``None`` (spelled "attr"
# too): the reservation attribute only, so the backend selects the AGPR form for
# every MFMA (dK / dV, S / dP and the dQ partial all in AGPRs; the CK plan).
# "vgpr_form" / "agpr_fence" / "asm_g13": only dK / dV in AGPRs (the split
# plan), by a comgr option, by fencing the loop-carried accumulators into AGPRs,
# or by issuing G1 / G3 as AGPR-form inline asm (with ``s_in_agpr``, G0 / G2 too).
AGPR_ROUTES = (None, "vgpr_form", "agpr_fence", "asm_g13")
SPLIT_AGPR_ROUTES = ("vgpr_form", "agpr_fence", "asm_g13")
AGPR_ROUTES_BUILT = (None, "vgpr_form", "agpr_fence", "asm_g13")
# Granularity of the inline-asm G1 / G3 MFMA blocks under ``asm_g13``: one asm
# block per atom-K step ("k_step") or one per GEMM call covering every K step
# ("gemm"). ``asm_tail_nop`` (None = AGPR_ASM_TAIL_NOP of the atom) is the
# s_nop count that ends each block; both are legal with ``asm_g13`` only.
ASM_CLUSTERS = ("k_step", "gemm")
ASM_TAIL_NOP_MAX = 15
# Issue order and thread distribution of the per-step Q / dO staging copies.
# "split": each tensor's copy in turn, every chunk's LDS store right after its
# global load. "loads_first": every Q and dO global load of the step before the
# first LDS store (tail chunks load a clamped chunk; only the store is
# predicated). "joint": one chunk space over the Q and dO tiles, so a CTA with
# more threads than one tile has chunks copies both in one pass (whole waves per
# tensor; legal only when kM0 * D / 8 < threads).
QDO_STAGES = COPY_ORDERS
# Where the per-row lse2 / Dsum workspace loads of a q step are issued: just
# before the softmax gradient ("at_use") or at the top of the step, ahead of the
# Q / dO staging and G0 / G2 ("step_top"), so their latency overlaps them.
STATS_LOADS = ("at_use", "step_top")
# LDS buffers staged by DMA that the transpose read consumes (Q^T and dO^T for
# the dV / dK GEMMs, K^T for dQ); their swizzle is fixed to the slab XOR.
TR_READ_DMA_BUFFERS = ("q", "do", "k")
# Barrier flavour of the in-loop barriers of a q step (they publish LDS writes
# or retire LDS reads): "full" drains VMEM and LDS (``s_waitcnt vmcnt(0)
# lgkmcnt(0)``, so the step's dQ atomics too); "lds_only" drains LDS only.
LOOP_SYNCS = ("full", "lds_only")
# Order of the flattened (q tile, head group) steps of a CTA: "qtile_outer"
# (head groups inner) or "head_outer" (q tiles inner). The dK / dV sums run in
# step order, so "head_outer" reorders the fp32 accumulation when a CTA has
# more than one head group and more than one q tile.
STEP_ORDERS = ("qtile_outer", "head_outer")
# An LDS-only drain is proved for the register staging path only: there the
# global loads that feed a ds_write are waited by data dependency in their own
# wave. A DMA path writes LDS from VMEM, so its barriers need the VMEM wait.
_DMA_LOOP_SYNC = "loop_sync='lds_only' requires global_path='vgpr'"

# The separate dQ kernel (Q-parallel recompute, no dQ workspace) is not built,
# so ``dq_mode = "split"`` is rejected until it is.
SPLIT_DQ_AVAILABLE = False

# G4 issue granularity: "block" computes the whole kM0 x D dQ partial, then
# its atomics; "fragment" computes one 16 x 16 fragment and issues its atomics
# before the next, reading its dS row from LDS again per fragment;
# "fragment_regs" reads dS into registers once, then issues per fragment.
# Bit-identical partials; "fragment" has the block IR when a wave owns a
# single G4 fragment; inert with acc_in_lds (always per fragment).
DQ_ISSUES = ("block", "fragment", "fragment_regs")
# Main-kernel pointers marked ``noalias``. False: none. "workspace": the
# plan-owned WS_* sub-buffers: disjoint from each other, and from every byte
# main accesses through a user tensor (prep writes each workspace byte main
# reads or adds to before main runs, so such an overlap would already clobber
# that input, and a dK / dV store into it would race the atomics). "all": every
# pointer; legal only with dkv_mode = "atomic", where main writes nothing but
# the workspace (in direct mode it stores dK / dV, and a request carries no
# pointers to prove them disjoint from Q / K / V / dO).
PTR_NOALIAS = (False, "workspace", "all")
_MAIN_WORKSPACE_PTRS = frozenset(("WS_LSE2", "WS_DSUM", "WS_DQ", "WS_DK", "WS_DV"))

# Knobs that are accepted, validated and salted into the kernel name but whose
# emission is not built yet: the main kernel's IR equals the default's apart
# from the name. They stay in the knob catalog (levers are retained) and a
# sweep must not count their values as distinct configurations.
NOT_YET_EFFECTIVE_KNOBS = MappingProxyType(
    {
        "block_k4": "the G4 dS key slicing is not built: G4 is emitted one "
        "atom-K step at a time over the whole kN0 dS tile; the value enters no "
        "estimate and is read only by the stage-table scheduler, which is not "
        "built",
    }
)

# Knobs that change no code under some specs but still declare a plan: under
# the attr route (``agpr_route = None``) the backend already puts S / dP in
# AGPRs, so ``s_in_agpr`` only selects the register estimate and the hot-loop
# AGPR-copy budget (:func:`agpr_copy_budget`); the code object equals the
# ``s_in_agpr = False`` one apart from the name. Predicate: True where inert.
DECLARATION_ONLY = MappingProxyType(
    {"s_in_agpr": lambda spec: getattr(spec, "agpr_route", None) is None}
)


def _atomic_epilogue_inert(spec, arch: str) -> bool:
    """Atomic dK / dV: the epilogue knob only sizes the LDS allocation.

    The atomic epilogue adds the accumulators into the workspace without LDS;
    ``"lds_staged"`` (the default) still reserves the epilogue buffer and
    ``"direct"`` drops it, which shrinks the allocation only where the
    epilogue phase is the largest phase. ``"lds_staged_pair"`` plans like the
    default in atomic mode.
    """
    if spec.dkv_mode != "atomic":
        return False
    default = dataclasses.replace(spec, dkv_epilogue="lds_staged")
    return lds_plan(spec, arch).total_bytes == lds_plan(default, arch).total_bytes


def _lds_order_inert(spec, arch: str) -> bool:
    """Without resident buffers both placement orders give the same offsets."""
    other = "phase_low" if spec.lds_order == "resident_low" else "resident_low"
    alt = dataclasses.replace(spec, lds_order=other)
    offsets = [
        tuple((b.name, b.slot, b.offset) for b in lds_plan(s, arch).buffers)
        for s in (spec, alt)
    ]
    return sorted(offsets[0]) == sorted(offsets[1])


# Knob values that are legal under a spec but change nothing there: the code
# object equals the one of the field's default apart from the kernel name, so
# a sweep must not count them as distinct configurations. Predicate
# ``(resolved spec, arch) -> bool``: True where the spec's own value of the
# knob is inert. Atomic dK / dV leave the CTA as fp32 atomics from the
# accumulators, so the store width does not exist there and the epilogue
# choice only sizes the allocation; direct dK / dV have no atomic row guard;
# without resident buffers both placement orders put every buffer at the
# same offset.
def _dq_issue_inert(spec, arch) -> bool:
    """``dq_issue`` emits the block IR: LDS accumulators always issue per
    fragment, and ``"fragment"`` over a single G4 fragment per wave is the
    block order (``"fragment_regs"`` reorders the dS reads; not inert)."""
    if spec.acc_in_lds:
        return True
    if spec.dq_issue != "fragment":
        return False
    g4 = BwdTileGeometry.from_spec(spec, arch).g4
    return g4.frags_m * g4.frags_n == 1


INERT_KNOBS = MappingProxyType(
    {
        "dkv_epilogue": _atomic_epilogue_inert,
        "dkv_store_vec": lambda spec, arch: spec.dkv_mode == "atomic",
        "dkv_atomic_guard": lambda spec, arch: spec.dkv_mode == "direct",
        "lds_order": _lds_order_inert,
        "dq_issue": _dq_issue_inert,
        # the slab hoist exists in THD direct mode only
        "dkv_slab_hoist": lambda spec, arch: not (
            spec.dkv_mode == "direct" and spec.seq_mode == "thd"
        ),
        # interior (unclamped) q steps exist with edge tiles only
        "interior_clamp": lambda spec, arch: not spec.edge_tiles,
    }
)

# Static register prune: an estimate may exceed its budget by up to this many
# registers and still be compiled (the operand-liveness term is conservative);
# the compiled resource notes then decide.
REGISTER_PRUNE_ALLOWANCE = 32
_REG_ALLOWANCE_FIXED = 24
# SIMDs per CDNA compute unit: a CTA of W waves puts ceil(W / 4) waves on each
# SIMD, which share its unified (arch VGPR + AGPR) register file.
_CDNA_SIMDS_PER_CU = 4

_ARCH_TRANSPOSE = {
    "mfma_tr": ("tr_read", "xt_lds", "lds_plain"),
    "mfma": ("xt_lds", "lds_plain"),
    "wmma": ("wmma_lds",),
}


# ---------------------------------------------------------------------------
# per-arch facts (from the arch catalog)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BwdArchFacts:
    """Backward-relevant facts of one gfx target, read from ``ArchTarget``."""

    arch: str
    wave_size: int
    matrix_path: str  # "mfma" | "wmma"
    lds_capacity_bytes: int
    arch_vgprs: int
    agprs: int
    has_tr_read: bool
    has_lds_dma: bool
    wide_k_atom: bool  # native 16x16x32 f16/bf16 atom

    @property
    def legal_waves(self) -> tuple[int, ...]:
        return (1, 2, 4, 8) if self.matrix_path == "mfma" else (1, 2, 4)

    @property
    def legal_atoms(self) -> tuple[str, ...]:
        if self.matrix_path == "wmma":
            return ("wmma16x16x16",)
        return ("16x16x16", "16x16x32") if self.wide_k_atom else ("16x16x16",)

    @property
    def legal_transpose_sources(self) -> tuple[str, ...]:
        if self.matrix_path == "wmma":
            return _ARCH_TRANSPOSE["wmma"]
        return _ARCH_TRANSPOSE["mfma_tr" if self.has_tr_read else "mfma"]

    @property
    def legal_global_paths(self) -> tuple[str, ...]:
        # LDS DMA wide enough for a 16-byte row segment exists only where the
        # transpose read exists too (gfx950); gfx942's one-dword DMA is not used.
        if self.matrix_path == "mfma" and self.has_lds_dma and self.has_tr_read:
            return ("vgpr", "dma")
        return ("vgpr",)

    @property
    def legal_ring_depths(self) -> tuple[int, ...]:
        if self.matrix_path == "wmma":
            return (1,)
        return (1, 2, 3) if self.has_tr_read else (1, 2)


@cache
def attn_bwd_arch_facts(arch: str) -> BwdArchFacts:
    """Facts for ``arch`` (a gfx id); raises ``ValueError`` for other targets."""
    if arch not in BWD_ARCHS:
        raise ValueError(
            f"attention backward does not target {arch!r}; known: {BWD_ARCHS}"
        )
    t = ArchTarget.from_gfx(arch)
    path = "mfma" if t.has_mfma else "wmma"
    agprs = int(t.limits.agprs)
    wide = False
    if path == "mfma":
        for dt in BWD_DTYPES:
            ops = t.mma.enumerate(
                family="mma", a_dtype=dt, b_dtype=dt, c_dtype="fp32", m=16, n=16
            )
            if any(op.k == 32 and op.wave_size == t.wave_size for op in ops):
                wide = True
    return BwdArchFacts(
        arch=arch,
        wave_size=int(t.wave_size),
        matrix_path=path,
        lds_capacity_bytes=int(t.lds_capacity_bytes),
        arch_vgprs=int(t.limits.vgprs) - agprs,
        agprs=agprs,
        has_tr_read=bool(t.memory.has_ds_read_tr),
        has_lds_dma=bool(t.memory.has_async_lds),
        wide_k_atom=wide,
    )


# ---------------------------------------------------------------------------
# spec
# ---------------------------------------------------------------------------


def _atom_k(atom: str) -> int:
    return int(atom.rsplit("x", 1)[1])


def _canonical_swizzle(value: str) -> str:
    """Normalise ``"k=xor,q=pad8"`` to sorted ``name=value`` pairs."""
    pairs = {}
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        name, sep, val = item.partition("=")
        name, val = name.strip(), val.strip()
        if not sep or not name.isidentifier():
            raise ValueError(f"lds_swizzle entry {item!r} is not 'buffer=value'")
        if val not in SWIZZLES:
            raise ValueError(
                f"lds_swizzle value {val!r} for {name!r} not in {SWIZZLES}"
            )
        if name in pairs:
            raise ValueError(f"lds_swizzle names buffer {name!r} twice")
        pairs[name] = val
    return ",".join(f"{k}={pairs[k]}" for k in sorted(pairs))


def _check_in(name, value, allowed):
    if value not in allowed:
        raise ValueError(f"{name}={value!r} not in {tuple(allowed)}")


def _canonical_asm_knobs(spec: "AttnBwdSpec") -> "AttnBwdSpec":
    """Spell the asm cluster knobs of a resolved spec by the code they emit.

    An ``asm_tail_nop`` equal to the G1 / G3 atom's ``AGPR_ASM_TAIL_NOP`` is the
    default (``None``), and ``asm_cluster = "gemm"`` over a single G1 / G3 K
    step (``block_m`` equal to the atom K) is the ``"k_step"`` cluster: the
    resolved spec, its kernel name and its IR are those of the default.
    """
    from kernels.common._attention_bwd_caps import AGPR_ASM_TAIL_NOP

    if spec.agpr_route != "asm_g13" or spec.atom_g13 is None:
        return spec
    atom_k = _atom_k(spec.atom_g13)
    updates = {}
    if spec.asm_tail_nop is not None and spec.asm_tail_nop == AGPR_ASM_TAIL_NOP.get(
        atom_k
    ):
        updates["asm_tail_nop"] = None
    if spec.asm_cluster == "gemm" and spec.block_m == atom_k:
        updates["asm_cluster"] = "k_step"
    return dataclasses.replace(spec, **updates) if updates else spec


@dataclass(frozen=True)
class AttnBwdSpec:
    """Compile-time spec of the layout-general backward main kernel.

    ``head_size`` is the only required field. A ``None`` knob is resolved by
    :func:`attn_bwd_default_geometry` for the target arch (see :meth:`resolved`).
    On MFMA targets ``agpr_alloc = None`` resolves to ``(0, 0)`` (VGPR-form
    MFMA) when the register estimate fits the arch-VGPR file, else stays
    ``None`` (backend choice); see :func:`_policy_agpr_alloc`. A one-wave
    ``head_size = 128`` tile with ``scheduler_strategy = None`` resolves to
    the register-minimising machine scheduler on every arch (see
    :data:`_D128_LOW_WAVE_SCHED`), and so does the CDNA narrow d128 tile
    (``stage_vec = 1``; see :data:`_NARROW_D128_SCHED`).
    """

    head_size: int
    dtype: str = "fp16"
    seq_mode: str = "batched"  # "batched" | "thd"
    dkv_mode: str = "direct"  # "direct" | "atomic"
    stage_vec: int = 8  # global access width of user tensors: 8 | 1
    mask_class: str = "band"  # "band" | "none"
    dq_mode: str = "atomic"  # "atomic" | "split"
    # Scale placement the kernel is compiled for: the multipliers that are 1.0
    # under it are not applied (see STATIC_SCALES); "runtime" applies all.
    static_scale: str = "fold_ds"
    waves: Optional[int] = None
    block_m: Optional[int] = None  # kM0
    block_n: Optional[int] = None  # kN0
    block_k4: Optional[int] = None  # kK4
    warp_grid_g4: Optional[tuple[int, int]] = None
    atom_g02: Optional[str] = None
    atom_g13: Optional[str] = None
    atom_g4: Optional[str] = None
    pt_route: Optional[str] = None
    kv_residency: Optional[str] = None
    kt_source: Optional[str] = None
    transpose_source: Optional[str] = None
    ring_depth: Optional[int] = None
    global_path: Optional[str] = None
    lds_swizzle: Optional[str] = None  # canonical "buf=value,..."; None = predictor
    sched: Optional[str] = None
    setprio: Optional[bool] = None
    waves_per_eu: Optional[int] = None
    edge_tiles: Optional[bool] = None
    head_pack: Optional[bool] = None
    acc_in_lds: Optional[bool] = None
    ws_layout: str = "head_major"
    # True: the CTA decode reads the runtime xcd_n / xcd_chunk kernargs and
    # applies the XCD remap (identity at xcd_n = 0). False emits no remap IR.
    xcd_remap: bool = False
    agpr_alloc: Optional[tuple[int, int] | str] = None  # see AGPR_ALLOCS
    s_in_agpr: Optional[bool] = None
    agpr_route: Optional[str] = None  # see AGPR_ROUTES; None = "attr"
    asm_tail_nop: Optional[int] = None  # 0..15; None = AGPR_ASM_TAIL_NOP
    asm_cluster: str = "k_step"  # see ASM_CLUSTERS
    scheduler_strategy: Optional[str] = None  # codegen scheduler strategy
    # Warp grids (rows, cols) of G0/G2 (S, dP) and G1/G3 (dV, dK). None resolves
    # to (1, W) / (W, 1); (2, W/2) is legal only with pt_route = "lds".
    warp_grid_g02: Optional[tuple[int, int]] = None
    warp_grid_g13: Optional[tuple[int, int]] = None
    # Issue order of G1 (dV), G3 (dK) and G4 (dQ and its atomics) inside a q
    # step; see STAGE_ORDERS. Every order gives bit-identical dK / dV / dQ
    # partials; non-default orders exclude the stage-table scheds.
    stage_order: str = DEFAULT_STAGE_ORDER
    loop_sync: str = "full"  # see LOOP_SYNCS
    step_order: str = "qtile_outer"  # see STEP_ORDERS
    # dK / dV epilogue (see DKV_EPILOGUES): "lds_staged" transposes the
    # register accumulators through one LDS buffer (dK, then dV), and stores
    # rows of dkv_store_vec elements; "lds_staged_pair" stages dK and dV in
    # two buffers behind one barrier; "direct" stores every accumulator
    # element from its register (no epilogue buffer). Values are identical.
    dkv_epilogue: str = "lds_staged"
    dkv_store_vec: int = 8  # see DKV_STORE_VECS
    dkv_atomic_guard: str = "branch"  # atomic dK / dV; see DKV_ATOMIC_GUARDS
    lds_order: str = "resident_low"  # see LDS_ORDERS
    qdo_stage: str = "split"  # see QDO_STAGES
    stats_load: str = "at_use"  # see STATS_LOADS
    # dQ-path levers; each default emits the shipped IR.
    dq_issue: str = "block"  # G4 to the dQ atomics: see DQ_ISSUES
    ptr_noalias: bool | str = False  # see PTR_NOALIAS
    q_rotate: bool = False  # start each q-tile range at (kv tile mod length)
    lb_interleave: bool = False  # the CTA decode also serves lb_order interleave
    # THD direct mode: form the dK / dV output slabs before the q loops and pin
    # each 64-bit pointer with an opaque SGPR copy, so two pointers (not the
    # i64 kv token base, both tensor pointers and their strides) stay live
    # across the loops (an SGPR spill fix at the SGPR ceiling). Same values;
    # inert in batched or atomic mode.
    dkv_slab_hoist: bool = False
    # edge_tiles: clamp the rows of the interior q steps too (Q / dO staging,
    # packed staging, dQ sink), as the edge steps do. An identity there (the
    # default drops it); kept because it changes the interior loop's register
    # allocation. Same values; inert without edge_tiles.
    interior_clamp: bool = False
    name: str = "rocke_attn_bwd"

    def __post_init__(self) -> None:
        """Arch-independent value sets and couplings (rules of the knob catalog)."""
        _check_in("head_size", self.head_size, BWD_HEAD_SIZES)
        _check_in("dtype", self.dtype, BWD_DTYPES)
        _check_in("seq_mode", self.seq_mode, SEQ_MODES)
        _check_in("dkv_mode", self.dkv_mode, DKV_MODES)
        _check_in("stage_vec", self.stage_vec, STAGE_VECS)
        _check_in("mask_class", self.mask_class, MASK_CLASSES)
        _check_in("dq_mode", self.dq_mode, DQ_MODES)
        _check_in("ws_layout", self.ws_layout, WS_LAYOUTS)
        _check_in("asm_cluster", self.asm_cluster, ASM_CLUSTERS)
        nop = self.asm_tail_nop
        if nop is not None and (
            isinstance(nop, bool)
            or not isinstance(nop, int)
            or not 0 <= nop <= ASM_TAIL_NOP_MAX
        ):
            raise ValueError(
                f"asm_tail_nop={nop!r} must be None or an int in "
                f"[0, {ASM_TAIL_NOP_MAX}]"
            )
        _check_in("stage_order", self.stage_order, STAGE_ORDERS)
        _check_in("loop_sync", self.loop_sync, LOOP_SYNCS)
        _check_in("step_order", self.step_order, STEP_ORDERS)
        _check_in("dkv_epilogue", self.dkv_epilogue, DKV_EPILOGUES)
        _check_in("dkv_store_vec", self.dkv_store_vec, DKV_STORE_VECS)
        _check_in("dkv_atomic_guard", self.dkv_atomic_guard, DKV_ATOMIC_GUARDS)
        _check_in("lds_order", self.lds_order, LDS_ORDERS)
        if (
            self.dkv_mode == "direct"
            and self.dkv_epilogue == "direct"
            and self.dkv_store_vec != 8
        ):
            raise ValueError(
                "dkv_store_vec applies to the LDS-staged epilogue; "
                "dkv_epilogue='direct' stores one element per accumulator slot"
            )
        _check_in("qdo_stage", self.qdo_stage, QDO_STAGES)
        _check_in("stats_load", self.stats_load, STATS_LOADS)
        _check_in("static_scale", self.static_scale, STATIC_SCALES)
        if self.dq_mode == "split" and not SPLIT_DQ_AVAILABLE:
            raise ValueError(
                "dq_mode='split' needs the split dQ kernel, which is not built"
            )
        opt = {
            "waves": (1, 2, 4, 8),
            "block_m": BLOCK_M_VALUES,
            "block_n": BLOCK_N_VALUES,
            "block_k4": BLOCK_K4_VALUES,
            "atom_g02": ATOMS,
            "atom_g13": ATOMS,
            "atom_g4": ATOMS,
            "pt_route": PT_ROUTES,
            "kv_residency": RESIDENCIES,
            "kt_source": RESIDENCIES,
            "transpose_source": TRANSPOSE_SOURCES,
            "ring_depth": (1, 2, 3),
            "global_path": GLOBAL_PATHS,
            "sched": SCHEDS,
            "waves_per_eu": (1, 2, 3),
        }
        for fname, allowed in opt.items():
            v = getattr(self, fname)
            if v is not None:
                _check_in(fname, v, allowed)
        for fname in ("setprio", "edge_tiles", "head_pack", "acc_in_lds", "s_in_agpr"):
            v = getattr(self, fname)
            if v is not None and not isinstance(v, bool):
                raise ValueError(f"{fname} must be a bool or None, got {v!r}")
        _check_in("dq_issue", self.dq_issue, DQ_ISSUES)
        for fname in ("q_rotate", "lb_interleave"):
            if not isinstance(getattr(self, fname), bool):
                raise ValueError(
                    f"{fname} must be a bool, got {getattr(self, fname)!r}"
                )
        if not (self.ptr_noalias is False or self.ptr_noalias in PTR_NOALIAS[1:]):
            raise ValueError(f"ptr_noalias={self.ptr_noalias!r} not in {PTR_NOALIAS}")
        if not isinstance(self.xcd_remap, bool):
            raise ValueError(f"xcd_remap must be a bool, got {self.xcd_remap!r}")
        for fname in ("dkv_slab_hoist", "interior_clamp"):
            if not isinstance(getattr(self, fname), bool):
                raise ValueError(
                    f"{fname} must be a bool, got {getattr(self, fname)!r}"
                )
        for gname in ("warp_grid_g4", "warp_grid_g02", "warp_grid_g13"):
            gv = getattr(self, gname)
            if gv is None:
                continue
            g = tuple(gv)
            if len(g) != 2 or any(not isinstance(x, int) or x < 1 for x in g):
                raise ValueError(f"{gname}={gv!r} must be (rows, cols)")
            object.__setattr__(self, gname, g)
        if self.agpr_alloc is not None:
            a = self.agpr_alloc
            a = a if isinstance(a, str) else tuple(a)
            _check_in("agpr_alloc", a, AGPR_ALLOCS)
            object.__setattr__(self, "agpr_alloc", a)
        if self.lds_swizzle is not None:
            object.__setattr__(
                self, "lds_swizzle", _canonical_swizzle(self.lds_swizzle)
            )
        if self.scheduler_strategy is not None:
            normalize_scheduler_strategy(self.scheduler_strategy)
        if not self.name or not self.name.replace("_", "").isalnum():
            raise ValueError(f"name={self.name!r} must be a non-empty identifier")
        # couplings that need no arch
        if self.kt_source == "reg" and self.kv_residency == "lds":
            raise ValueError("kt_source='reg' requires kv_residency='reg'")
        if self.global_path == "dma" and self.stage_vec != 8:
            raise ValueError("global_path='dma' requires stage_vec=8")
        if self.global_path == "dma" and self.transpose_source == "xt_lds":
            raise ValueError("global_path='dma' excludes transpose_source='xt_lds'")
        if self.global_path == "dma" and self.loop_sync != "full":
            raise ValueError(_DMA_LOOP_SYNC)
        if (
            self.setprio
            and self.sched is not None
            and self.sched not in STAGE_TABLE_SCHEDS
        ):
            raise ValueError("setprio=True is legal only with a stage-table sched")
        if self.stage_order != DEFAULT_STAGE_ORDER and self.sched in STAGE_TABLE_SCHEDS:
            raise ValueError(
                _STAGE_ORDER_SCHED_RULE.format(self.stage_order, self.sched)
            )
        if self.s_in_agpr and self.agpr_alloc == (0, 0):
            raise ValueError("s_in_agpr=True requires agpr_alloc != (0, 0)")
        if self.acc_in_lds and self.agpr_alloc not in (None, (0, 0)):
            raise ValueError("acc_in_lds=True implies agpr_alloc=(0, 0)")
        if self.agpr_route == "attr":
            object.__setattr__(self, "agpr_route", None)
        _check_in("agpr_route", self.agpr_route, AGPR_ROUTES)
        if self.agpr_route is not None:
            a = self.agpr_alloc
            if not (isinstance(a, tuple) and a[0] > 0):
                raise ValueError(
                    f"agpr_route={self.agpr_route!r} needs a reservation "
                    f"agpr_alloc=(n, n) with n > 0, got {a!r}"
                )
            if self.s_in_agpr and self.agpr_route != "asm_g13":
                raise ValueError(
                    f"s_in_agpr=True is legal with agpr_route None or 'asm_g13', "
                    f"not {self.agpr_route!r} (it keeps S / dP in arch VGPRs)"
                )
            if (
                self.agpr_route == "asm_g13"
                and self.sched is not None
                and (self.sched in STAGE_TABLE_SCHEDS or self.sched.startswith("iglp"))
            ):
                raise ValueError(
                    "agpr_route='asm_g13' excludes stage tables and iglp (the "
                    "scheduler cannot see the MFMAs inside an asm block)"
                )
        if self.agpr_route != "asm_g13":
            if self.asm_tail_nop is not None:
                raise ValueError("asm_tail_nop is legal with agpr_route='asm_g13' only")
            if self.asm_cluster != "k_step":
                raise ValueError(
                    f"asm_cluster={self.asm_cluster!r} is legal with "
                    "agpr_route='asm_g13' only"
                )
        if (
            self.atom_g13 == "16x16x32"
            and self.block_m is not None
            and self.block_m < 32
        ):
            raise ValueError("atom_g13='16x16x32' requires block_m >= 32")

    # -- derived -----------------------------------------------------------

    def resolved(self, arch: str) -> "AttnBwdSpec":
        """A copy with every ``None`` knob replaced by the arch policy value."""
        defaults = attn_bwd_default_geometry(
            arch, self.head_size, self.dtype, self.stage_vec, self.seq_mode
        )
        updates = {k: v for k, v in defaults.items() if getattr(self, k) is None}
        out = dataclasses.replace(self, **updates)
        # the G0/G2 and G1/G3 grids follow the resolved wave count
        grids = {}
        if out.warp_grid_g02 is None:
            grids["warp_grid_g02"] = (1, out.waves)
        if out.warp_grid_g13 is None:
            grids["warp_grid_g13"] = (out.waves, 1)
        if grids:
            out = dataclasses.replace(out, **grids)
        if out.agpr_alloc is None and attn_bwd_arch_facts(arch).matrix_path == "mfma":
            out = dataclasses.replace(out, agpr_alloc=_policy_agpr_alloc(out, arch))
        if out.scheduler_strategy is None and out.waves == 1 and out.head_size == 128:
            out = dataclasses.replace(out, scheduler_strategy=_D128_LOW_WAVE_SCHED)
        return _canonical_asm_knobs(out)

    def non_default_fields(self) -> tuple[tuple[str, object], ...]:
        """Compile-time fields that differ from their declared default."""
        out = []
        for f in fields(self):
            if f.name in ("head_size", "name") or f.default is dataclasses.MISSING:
                continue
            v = getattr(self, f.name)
            if v != f.default:
                out.append((f.name, v))
        return tuple(out)

    def kernel_name(self, stage: str = "main") -> str:
        """Kernel name: base key plus a salt over every non-default knob."""
        _check_in("stage", stage, ("main", "dq"))
        base = (
            f"{self.name}_{stage}_{self.dtype}_d{self.head_size}_"
            f"{self.seq_mode}_{self.dkv_mode}_{self.mask_class}_sv{self.stage_vec}"
        )
        salted = [
            (k, v)
            for k, v in self.non_default_fields()
            if k not in ("dtype", "seq_mode", "dkv_mode", "stage_vec", "mask_class")
        ]
        if not salted:
            return base
        digest = hashlib.sha1(
            repr(sorted(salted, key=lambda kv: kv[0])).encode()
        ).hexdigest()
        return f"{base}_{digest[:10]}"

    def aux_spec(
        self,
        stage: str = "prep",
        *,
        rows_per_cta: Optional[int] = None,
        convert_vec: int = 8,
    ):
        """The shared prep / convert spec matching this main spec.

        ``rows_per_cta`` (``None``: the aux default) and ``convert_vec`` (the
        convert only) are the aux kernels' own tuning values; they never change
        the main kernel. Imported lazily: the aux module is a separate file of
        the family.
        """
        _check_in("stage", stage, ("prep", "convert"))
        try:
            from kernels.common.attention_bwd_aux import AttnBwdAuxSpec
        except ImportError as exc:  # pragma: no cover - depends on the tree
            raise NotImplementedError(
                "AttnBwdAuxSpec is not available: kernels/common/attention_bwd_aux.py "
                "is missing from this tree"
            ) from exc
        if stage == "prep" and convert_vec != 8:
            raise ValueError("convert_vec applies to the convert stage only")
        return AttnBwdAuxSpec(
            head_size=self.head_size,
            dtype=self.dtype,
            seq_mode=self.seq_mode,
            stage_vec=self.stage_vec,
            ws_layout=self.ws_layout,
            rows_per_cta=rows_per_cta,
            convert_vec=convert_vec,
            name=self.name,
        )


# ---------------------------------------------------------------------------
# default geometry policy (start points; the tuned table overrides it)
# ---------------------------------------------------------------------------

# (kM0, kN0, kK4, W, warp_grid_g4, kv_residency, kt_source) per (arch class, d).
# The d64 / d128 start points run the 3.2 / 3.3 tiles on eight waves (gfx950
# d128: 16 x 128, as 192 keys do not split over eight waves). On four waves
# the per-wave state (resident K / V / K^T and the dK / dV accumulators) fits
# only with the accumulators in AGPRs, and the production backend then puts
# the S / dP / dQ accumulators in AGPRs as well, which the VALU reads inside
# the q loop (the hot-loop AGPR copy rule of the resource gate fails). Eight
# waves halve the per-wave state, so the VGPR-form MFMA holds all of it.
_CDNA_TILES = {
    ("gfx942", 32): (32, 128, 64, 4, (2, 2), "reg", "reg"),
    ("gfx942", 64): (32, 128, 32, 8, (2, 4), "reg", "reg"),
    ("gfx942", 128): (16, 128, 32, 8, (1, 8), "reg", "reg"),
    ("gfx950", 32): (32, 128, 64, 4, (2, 2), "reg", "reg"),
    ("gfx950", 64): (32, 256, 32, 8, (2, 4), "reg", "lds"),
    ("gfx950", 128): (16, 128, 32, 8, (1, 8), "reg", "lds"),
}
# stage_vec = 1 tiles: 16 x 64, except d32 where a four-wave G4 grid needs two
# warp rows (D / (16 * 4) < 1), hence kM0 = 32.
_NARROW_TILE = (16, 64, 32, 4, (1, 4), "reg", "reg")
_NARROW_TILE_D32 = (32, 64, 32, 4, (2, 2), "reg", "reg")
_DEBUG_TILE = (16, 16, 16, 1, (1, 1), "lds", "lds")
# Machine-scheduler strategy of the RDNA d128 start points (LDS accumulators,
# one or two waves) and of every one-wave d128 tile (the CDNA 16 x 16
# correctness tile): one wave holds all of D = 128 of a 16-row tile, the default
# scheduler fills the 256-VGPR file and the production backend then spills a
# few registers on some instances (atomic dK / dV with head packing).
_RDNA_D128_SCHED = "iterative-minreg"
_D128_LOW_WAVE_SCHED = _RDNA_D128_SCHED
# The CDNA narrow d128 tile (stage_vec = 1, 16 x 64, K / V / K^T in registers)
# sits at the 256-VGPR file under the default scheduler, and the production
# backend spills one register on gfx950 (atomic dK / dV under a band); the
# register-minimising scheduler keeps every narrow d128 instance spill-free.
_NARROW_D128_SCHED = _RDNA_D128_SCHED


def attn_bwd_default_geometry(
    arch: str,
    head_size: int,
    dtype: str = "fp16",
    stage_vec: int = 8,
    seq_mode: str = "batched",
) -> dict:
    """Per-arch start point for every ``None`` knob (no AGPR choice).

    The shipped correctness configuration uses the portable paths:
    ``global_path = "vgpr"``, ring depth 1, no stage table; on gfx942 the plain
    LDS transpose, on gfx950 the transpose read. RDNA uses the one-wave 16 x 16
    WMMA schedule with LDS accumulators (d128 with the register-minimising
    machine scheduler). ``seq_mode`` does not change the start point today; it
    is part of the signature so a tuned table may key on it.
    """
    facts = attn_bwd_arch_facts(arch)
    _check_in("head_size", head_size, BWD_HEAD_SIZES)
    _check_in("dtype", dtype, BWD_DTYPES)
    _check_in("stage_vec", stage_vec, STAGE_VECS)
    _check_in("seq_mode", seq_mode, SEQ_MODES)
    if facts.matrix_path == "wmma":
        km, kn, kk, w, g4, kv, kt = _DEBUG_TILE
        return {
            "waves": w,
            "block_m": km,
            "block_n": kn,
            "block_k4": kk,
            "warp_grid_g4": g4,
            "atom_g02": "wmma16x16x16",
            "atom_g13": "wmma16x16x16",
            "atom_g4": "wmma16x16x16",
            "pt_route": "lds",
            "kv_residency": kv,
            "kt_source": kt,
            "transpose_source": "wmma_lds",
            "ring_depth": 1,
            "global_path": "vgpr",
            "sched": "none",
            "setprio": False,
            "waves_per_eu": None,
            "edge_tiles": False,
            "head_pack": False,
            "acc_in_lds": True,
            "s_in_agpr": False,
            # d128 on one / two waves sits at the 256-VGPR file: the
            # register-minimising machine scheduler keeps it free of spills.
            "scheduler_strategy": _RDNA_D128_SCHED if head_size == 128 else None,
        }
    key = "gfx950" if facts.has_tr_read else "gfx942"
    if stage_vec == 8:
        tile = _CDNA_TILES[(key, head_size)]
    else:
        tile = _NARROW_TILE_D32 if head_size == 32 else _NARROW_TILE
    km, kn, kk, w, g4, kv, kt = tile
    if facts.has_tr_read and stage_vec == 8:
        transpose = "tr_read"
    else:
        transpose = "lds_plain"
    wide = facts.wide_k_atom and kk % 32 == 0 and head_size % 32 == 0
    return {
        "waves": w,
        "block_m": km,
        "block_n": kn,
        "block_k4": kk,
        "warp_grid_g4": g4,
        "atom_g02": "16x16x32" if wide else "16x16x16",
        "atom_g13": "16x16x16",
        "atom_g4": "16x16x32" if wide else "16x16x16",
        "pt_route": "relabel",
        "kv_residency": kv,
        "kt_source": kt,
        "transpose_source": transpose,
        "ring_depth": 1,
        "global_path": "vgpr",
        "sched": "none",
        "setprio": False,
        "waves_per_eu": 1,
        "edge_tiles": False,
        "head_pack": False,
        "acc_in_lds": False,
        "s_in_agpr": False,
        "scheduler_strategy": (
            _NARROW_D128_SCHED if stage_vec == 1 and head_size == 128 else None
        ),
    }


# ---------------------------------------------------------------------------
# static legality: LDS footprint and register estimates
# ---------------------------------------------------------------------------


def _policy_agpr_alloc(spec: AttnBwdSpec, arch: str) -> Optional[tuple[int, int]]:
    """``(0, 0)`` (VGPR-form MFMA) when the whole register estimate fits the
    arch-VGPR file (always with LDS accumulators), else ``None`` (backend).

    An explicit AGPR reservation makes the backend select every MFMA in AGPR
    form, including S / dP / dQ, whose values the VALU then reads with AGPR
    copies inside the q loop; the VGPR form has no such copies.
    """
    if spec.acc_in_lds:
        return (0, 0)
    vgpr, _ = _raw_estimates(spec, arch, agpr_form=False)
    return (0, 0) if vgpr <= attn_bwd_arch_facts(arch).arch_vgprs else None


def _require_resolved(spec: AttnBwdSpec) -> None:
    missing = [
        f.name
        for f in fields(spec)
        if getattr(spec, f.name) is None
        and f.name
        not in (
            "lds_swizzle",
            "agpr_alloc",
            "agpr_route",
            "waves_per_eu",
            "scheduler_strategy",
            "asm_tail_nop",
        )
    ]
    if missing:
        raise ValueError(f"spec is not resolved (call .resolved(arch)): {missing}")


def lds_request(spec: AttnBwdSpec, arch: str) -> BwdLdsRequest:
    """The LDS phase-planner request of a resolved spec on ``arch``."""
    _require_resolved(spec)
    return BwdLdsRequest(
        head_size=spec.head_size,
        k_m0=spec.block_m,
        k_n0=spec.block_n,
        waves=spec.waves,
        wave_size=attn_bwd_arch_facts(arch).wave_size,
        dtype=spec.dtype,
        kv_residency=spec.kv_residency,
        kt_source=spec.kt_source,
        transpose_source=spec.transpose_source,
        ring_depth=spec.ring_depth,
        global_path=spec.global_path,
        pt_route=spec.pt_route,
        acc_in_lds=bool(spec.acc_in_lds),
        dq_mode=spec.dq_mode,
        lds_swizzle=BwdLdsRequest.canonical_swizzle(spec.lds_swizzle),
        dkv_epilogue=_planned_epilogue(spec),
        lds_order=spec.lds_order,
    )


def _planned_epilogue(spec: AttnBwdSpec) -> str:
    """The epilogue the LDS plan provides for (see :data:`INERT_KNOBS`).

    Atomic dK / dV use no staging buffer: the pair is planned like the
    default (one reserved buffer), ``"direct"`` reserves none.
    """
    if spec.dkv_mode == "atomic" and spec.dkv_epilogue == "lds_staged_pair":
        return "lds_staged"
    return spec.dkv_epilogue


def lds_plan(spec: AttnBwdSpec, arch: str) -> LdsPlan:
    """The LDS phase plan of a resolved spec (not checked against capacity)."""
    return plan_bwd_lds(lds_request(spec, arch), arch=arch, strict=False)


def lds_footprint_bytes(spec: AttnBwdSpec, arch: str) -> int:
    """LDS bytes of a resolved spec: the phase planner's ``total_bytes``.

    The phase planner (:func:`plan_bwd_lds`) is the only LDS model of the
    family: resident buffers (LDS K / V, the K^T image or copy, LDS
    accumulators) plus the largest of the phase-aliased stage-0 K, stage-0 V,
    loop and epilogue phases, every buffer aligned to a 128-byte line.
    """
    return lds_plan(spec, arch).total_bytes


def _raw_estimates(spec: AttnBwdSpec, arch: str, *, agpr_form: bool) -> tuple[int, int]:
    facts = attn_bwd_arch_facts(arch)
    wave, w = facts.wave_size, spec.waves
    d, km, kn = spec.head_size, spec.block_m, spec.block_n
    half = 2 * wave * w  # 16-bit elements per register, spread over the CTA
    vgpr = _REG_ALLOWANCE_FIXED
    if spec.kv_residency == "reg":
        vgpr += 2 * (kn * d // half)
    if spec.kt_source == "reg":
        vgpr += kn * d // half
    vgpr += 2 * km * d // (2 * wave)
    prefetch = spec.global_path == "vgpr" and (
        spec.ring_depth >= 2 or (arch == "gfx942" and spec.sched in STAGE_TABLE_SCHEDS)
    )
    if prefetch:
        vgpr += km * d // (wave * w)
    vgpr += km * kn // (wave * w)
    vgpr += 2 * km * _atom_k(spec.atom_g4) // (2 * wave)
    acc, s_dp, dq = _accumulator_terms(spec, arch)
    in_agpr = _agpr_placement(spec, arch, agpr_form=agpr_form)
    agpr = 0
    for term, name in ((acc, "acc"), (s_dp, "s_dp"), (dq, "dq")):
        if name in in_agpr:
            agpr += term
        else:
            vgpr += term
    return vgpr, agpr


def _accumulator_terms(spec: AttnBwdSpec, arch: str) -> tuple[int, int, int]:
    """Per-lane fp32 registers of the dK / dV accumulators, of S and dP, and of
    the dQ partial."""
    wave, w = attn_bwd_arch_facts(arch).wave_size, spec.waves
    d, km, kn = spec.head_size, spec.block_m, spec.block_n
    acc = 0 if spec.acc_in_lds else 2 * (kn * d // (wave * w))
    return acc, 2 * km * kn // (wave * w), km * d // (wave * w)


def _agpr_placement(spec: AttnBwdSpec, arch: str, *, agpr_form: bool) -> frozenset:
    """Accumulators the emitted code keeps in AGPRs.

    VGPR form (``(0, 0)``, ``acc_in_lds``, WMMA): none. The attr route (an
    AGPR form without a split route: ``(n, n)``, ``"backend"``, or a policy
    ``None``): the backend selects the AGPR form for every MFMA, so dK / dV, S /
    dP and the dQ partial, whatever ``s_in_agpr`` says. Split routes: dK / dV,
    plus S / dP under ``asm_g13`` with ``s_in_agpr``.
    """
    if attn_bwd_arch_facts(arch).matrix_path != "mfma" or not agpr_form:
        return frozenset()
    route = getattr(spec, "agpr_route", None)
    if route is None:
        return frozenset({"acc", "s_dp", "dq"})
    if route == "asm_g13" and spec.s_in_agpr:
        return frozenset({"acc", "s_dp"})
    return frozenset({"acc"})


def agpr_copy_budget(spec: AttnBwdSpec, arch: str) -> dict:
    """Hot-loop AGPR copies per q step a resolved spec may issue (rule 6).

    ``{"reads": n, "writes": 0, "moves": 0}``. Reads are allowed only where the
    spec declares accumulators in AGPRs that the VALU reads every step: under
    the attr route with ``s_in_agpr``, S and dP (softmax gradient) and the dQ
    partial (the atomic sink); under ``asm_g13`` with ``s_in_agpr``, S and dP.
    Everything else (VGPR form, split routes, the attr route without the
    declaration) allows none: a read, write or move there is AGPR spill or
    copy traffic.
    """
    _require_resolved(spec)
    facts = attn_bwd_arch_facts(arch)
    if facts.matrix_path != "mfma" or spec.agpr_alloc == (0, 0) or not spec.s_in_agpr:
        return {"reads": 0, "writes": 0, "moves": 0}
    _, s_dp, dq = _accumulator_terms(spec, arch)
    agpr_form = spec.agpr_alloc != (0, 0)
    placed = _agpr_placement(spec, arch, agpr_form=agpr_form)
    reads = (s_dp if "s_dp" in placed else 0) + (dq if "dq" in placed else 0)
    return {"reads": reads, "writes": 0, "moves": 0}


def register_estimates(spec: AttnBwdSpec, arch: str) -> dict:
    """Two-budget register estimate of a resolved spec.

    Returns ``{"arch_vgpr": n, "agpr": n}``. On MFMA, unless ``agpr_alloc =
    (0, 0)`` (VGPR form) or ``acc_in_lds``, the accumulators the emitted code
    keeps in AGPRs count as AGPRs (see :func:`_agpr_placement`): all of them
    (dK / dV, S / dP, the dQ partial) under the attr route, dK / dV (and S / dP
    under ``asm_g13`` with ``s_in_agpr``) under a split route.
    """
    _require_resolved(spec)
    facts = attn_bwd_arch_facts(arch)
    agpr_form = facts.matrix_path == "mfma" and spec.agpr_alloc != (0, 0)
    vgpr, agpr = _raw_estimates(spec, arch, agpr_form=agpr_form)
    return {"arch_vgpr": vgpr, "agpr": agpr}


def validate_attn_bwd_spec(spec: AttnBwdSpec, arch: str) -> AttnBwdSpec:
    """Resolve ``spec`` for ``arch`` and apply every static legality rule.

    Returns the resolved spec; raises ``ValueError`` naming the first broken
    rule. Rules: catalog (waves, atoms, transpose read, DMA, ring depth),
    geometry divisibility and couplings, the LDS phase plan within the
    per-arch capacity, and the two register budgets (arch-VGPR and AGPR, their sum
    under ``waves_per_eu``) with the static prune allowance.
    """
    facts = attn_bwd_arch_facts(arch)
    s = spec.resolved(arch)
    w, km, kn, kk, d = s.waves, s.block_m, s.block_n, s.block_k4, s.head_size
    mfma = facts.matrix_path == "mfma"
    # catalog
    _check_in(f"waves on {arch}", w, facts.legal_waves)
    for fname in ("atom_g02", "atom_g13", "atom_g4"):
        _check_in(f"{fname} on {arch}", getattr(s, fname), facts.legal_atoms)
    _check_in(
        f"transpose_source on {arch}", s.transpose_source, facts.legal_transpose_sources
    )
    _check_in(f"global_path on {arch}", s.global_path, facts.legal_global_paths)
    _check_in(f"ring_depth on {arch}", s.ring_depth, facts.legal_ring_depths)
    # geometry
    if kn % (16 * w) != 0:
        raise ValueError(f"block_n={kn} must be a multiple of 16*waves={16 * w}")
    if w == 1 and (km, kn) != (16, 16):
        raise ValueError("waves=1 is legal only for the 16 x 16 debug / fallback tile")
    k4a = _atom_k(s.atom_g4)
    if kk % k4a != 0:
        raise ValueError(f"block_k4={kk} must be a multiple of the G4 atom K ({k4a})")
    if kn % kk != 0:
        raise ValueError(f"block_n={kn} must be a multiple of block_k4={kk}")
    if d % _atom_k(s.atom_g02) != 0:
        raise ValueError(f"head_size={d} must be a multiple of the G0/G2 atom K")
    if s.atom_g13 == "16x16x32" and km < 32:
        raise ValueError("atom_g13='16x16x32' requires block_m >= 32")
    g4r, g4c = s.warp_grid_g4
    if (g4r, g4c) not in ((1, w), (2, w // 2)) or g4r * g4c != w:
        raise ValueError(
            f"warp_grid_g4={s.warp_grid_g4} must be (1, {w}) or (2, {w // 2})"
        )
    if d % (16 * g4c) != 0 or km % (16 * g4r) != 0:
        raise ValueError(
            f"warp_grid_g4={s.warp_grid_g4} does not divide D={d} / block_m={km}"
        )
    if km % 16 != 0:
        raise ValueError(f"block_m={km} must be a multiple of 16")
    # G0/G2 tile kM0 x kN0, G1/G3 tile kN0 x D: fixed grids unless P^T goes via LDS
    for gname, natural, rows, cols in (
        ("warp_grid_g02", (1, w), km, kn),
        ("warp_grid_g13", (w, 1), kn, d),
    ):
        g = getattr(s, gname)
        if g == natural:
            continue
        alt = (2, w // 2)
        if w < 2 or g != alt:
            raise ValueError(
                f"{gname}={g} must be {natural} (or {alt} with pt_route='lds')"
            )
        if s.pt_route != "lds":
            raise ValueError(
                f"{gname}={g} needs pt_route='lds' (the relabel route fixes it)"
            )
        if rows % 32 != 0 or cols % (16 * (w // 2)) != 0:
            raise ValueError(f"{gname}={g} does not divide its {rows} x {cols} tile")
    # couplings
    if s.ptr_noalias == "all" and s.dkv_mode != "atomic":
        raise ValueError(
            "ptr_noalias='all' needs dkv_mode='atomic' (direct mode stores dK / dV, "
            "which the request cannot prove disjoint from the inputs)"
        )
    if s.kt_source == "reg" and s.kv_residency != "reg":
        raise ValueError("kt_source='reg' requires kv_residency='reg'")
    if not mfma and s.kv_residency == "reg":
        raise ValueError("kv_residency='reg' is not legal on WMMA targets")
    if not mfma and s.pt_route == "relabel":
        raise ValueError(
            "pt_route='relabel' is not legal on WMMA targets (C is not A^T)"
        )
    if s.global_path == "dma" and (s.stage_vec != 8 or s.transpose_source == "xt_lds"):
        raise ValueError("global_path='dma' needs stage_vec=8 and excludes xt_lds")
    if s.loop_sync != "full" and s.global_path != "vgpr":
        raise ValueError(_DMA_LOOP_SYNC)
    if s.global_path == "dma" and s.transpose_source == "tr_read" and s.lds_swizzle:
        for item in s.lds_swizzle.split(","):
            buf, _, val = item.partition("=")
            if buf in TR_READ_DMA_BUFFERS and val != "xor":
                raise ValueError(
                    f"lds_swizzle {buf}={val}: DMA buffers read by tr_read need 'xor'"
                )
    if s.acc_in_lds and s.dkv_mode == "direct" and s.dkv_epilogue != "lds_staged":
        raise ValueError(
            f"dkv_epilogue={s.dkv_epilogue!r} needs register accumulators; with "
            "acc_in_lds the epilogue reads the fp32 LDS images ('lds_staged')"
        )
    staged = (s.global_path, s.ring_depth) != ("vgpr", 1)
    if staged and (s.qdo_stage != "split" or s.stats_load != "at_use"):
        raise ValueError(
            "qdo_stage / stats_load reorder the synchronous VGPR staging; with "
            "global_path='dma' or ring_depth > 1 the ring owns the issue order"
        )
    if s.qdo_stage == "joint":
        chunks = km * d // 8
        threads = w * facts.wave_size
        if chunks >= threads or chunks % facts.wave_size:
            raise ValueError(
                f"qdo_stage='joint' needs block_m * D / 8 = {chunks} below the "
                f"{threads} threads and a multiple of the wave size (otherwise "
                "it is the 'split' distribution)"
            )
    if s.sched.startswith("iglp") and s.setprio:
        raise ValueError("setprio is legal only inside a stage table")
    if s.setprio and s.sched not in STAGE_TABLE_SCHEDS:
        raise ValueError("setprio is legal only inside a stage table")
    if s.stage_order != DEFAULT_STAGE_ORDER and s.sched in STAGE_TABLE_SCHEDS:
        raise ValueError(_STAGE_ORDER_SCHED_RULE.format(s.stage_order, s.sched))
    if not mfma:
        if not s.acc_in_lds:
            raise ValueError("acc_in_lds=False is not legal on WMMA targets")
        if s.agpr_alloc is not None or s.s_in_agpr or s.agpr_route is not None:
            raise ValueError(
                "agpr_alloc / s_in_agpr / agpr_route apply to MFMA targets only"
            )
        if s.waves_per_eu not in (None, 1, 2):
            raise ValueError("waves_per_eu on WMMA targets must be None, 1 or 2")
    else:
        if s.acc_in_lds and s.agpr_alloc != (0, 0):
            raise ValueError("acc_in_lds=True implies agpr_alloc=(0, 0)")
        if s.s_in_agpr and s.agpr_alloc == (0, 0):
            raise ValueError("s_in_agpr=True requires agpr_alloc != (0, 0)")
        if s.waves_per_eu == 3 and not (arch == "gfx942" and w <= 2):
            raise ValueError("waves_per_eu=3 is legal only on gfx942 with waves <= 2")
        if s.agpr_route == "asm_g13" and s.asm_tail_nop is not None:
            # the s_nop after each asm cluster must cover the ISA's MFMA result
            # wait states (G1 and G3 share the atom_g13 atom)
            from kernels.common._attention_bwd_caps import check_asm_tail_nop

            check_asm_tail_nop(arch, s.dtype, _atom_k(s.atom_g13), s.asm_tail_nop)
    # LDS: the phase planner decides
    plan = lds_plan(s, arch)
    if not plan.fits:
        raise ValueError(
            f"LDS plan {plan.total_bytes} B exceeds {plan.capacity_bytes} B on {arch}"
        )
    # registers
    est = register_estimates(s, arch)
    vg, ag = est["arch_vgpr"], est["agpr"]
    slack = REGISTER_PRUNE_ALLOWANCE
    if mfma:
        if vg > facts.arch_vgprs + slack:
            raise ValueError(
                f"arch-VGPR estimate {vg} exceeds {facts.arch_vgprs} (+{slack})"
            )
        if s.agpr_alloc == (0, 0):
            if vg > facts.arch_vgprs:
                raise ValueError(
                    f"agpr_alloc=(0, 0) needs arch-VGPR + AGPR estimate {vg} <= 256"
                )
        elif s.agpr_alloc not in (None, AGPR_ALLOC_BACKEND):
            n = s.agpr_alloc[0]
            # The reservation is a minimum the backend must honour inside the
            # per-wave share of the unified file; it must leave arch VGPRs (at
            # least the fixed allowance of the estimate), or codegen fails
            # (for example (256, 256) on an eight-wave tile: 256 per wave).
            per_simd = max(-(-w // _CDNA_SIMDS_PER_CU), s.waves_per_eu or 1)
            share = (facts.arch_vgprs + facts.agprs) // per_simd
            if share - n < _REG_ALLOWANCE_FIXED:
                raise ValueError(
                    f"agpr_alloc={s.agpr_alloc} leaves {share - n} of the {share} "
                    f"registers per wave ({per_simd} waves per SIMD at waves={w}) "
                    f"for arch VGPRs; at least {_REG_ALLOWANCE_FIXED} are needed"
                )
            if n < ag:
                raise ValueError(
                    f"agpr_alloc={s.agpr_alloc} below the AGPR estimate {ag}"
                )
            if ag > min(facts.agprs, n) + slack:
                raise ValueError(
                    f"AGPR estimate {ag} exceeds {min(facts.agprs, n)} (+{slack})"
                )
        elif ag > facts.agprs + slack:
            raise ValueError(f"AGPR estimate {ag} exceeds {facts.agprs} (+{slack})")
        wpe = s.waves_per_eu
        if wpe in (2, 3) and vg + ag > (facts.arch_vgprs + facts.agprs) // wpe:
            raise ValueError(
                f"waves_per_eu={wpe} needs arch-VGPR + AGPR estimate {vg + ag} "
                f"<= {(facts.arch_vgprs + facts.agprs) // wpe}"
            )
        # A CTA of more than four waves puts several of its own waves on each
        # SIMD, which share the unified file whatever agpr_alloc says.
        per_wg = -(-w // _CDNA_SIMDS_PER_CU)
        wg_share = (facts.arch_vgprs + facts.agprs) // per_wg
        if per_wg > 1 and vg + ag > wg_share:
            raise ValueError(
                f"workgroup share: waves={w} puts {per_wg} waves on each SIMD; "
                f"arch-VGPR + AGPR estimate {vg + ag} exceeds {wg_share} per wave"
            )
    else:
        if vg > facts.arch_vgprs + slack:
            raise ValueError(
                f"register estimate {vg} exceeds {facts.arch_vgprs} (+{slack})"
            )
        if s.waves_per_eu == 2 and vg > facts.arch_vgprs // 2:
            raise ValueError(
                "waves_per_eu=2 needs the register estimate <= half the file"
            )
    return s


# ---------------------------------------------------------------------------
# kernarg ABI rocke.attn_bwd.v3
# ---------------------------------------------------------------------------
#
# Order: pointers, i64 scalars, f32 scalars, i32 scalars. Every pointer is 8
# bytes, so the i64 block starts 8-byte aligned and the packed argument bytes
# need no padding. The batch and head strides are i64 (a tensor may span more
# than 2^31 elements between batches or heads; the kernels rebase in i64);
# the token strides stay i32 (the host predicate bounds S_max * stride_t).

_MAIN_PTRS = (
    "Q K V dO dK dV WS_LSE2 WS_DSUM WS_DQ WS_DK WS_DV SEQ_Q SEQ_KV OFF_Q OFF_KV WORKLIST"
).split()
_MAIN_I64 = "q_b q_h k_b k_h v_b v_h do_b do_h dk_b dk_h dv_b dv_h".split()
_MAIN_F32 = ("scale_log2", "ds_mult", "dk_mult", "dq_mult")
_MAIN_I32 = (
    "q_t k_t v_t do_t dk_t dv_t "
    "h_q h_k h_v G gk gv S_q_max S_kv_max has_len len_stride off64 q_mult q_div kv_mult kv_div "
    "left right bottom_right ws_rows_q ws_rows_kv ws_seg "
    "g_split n_kv_tiles lb_order pair xcd_n xcd_chunk pack_heads use_worklist"
).split()
_PREP_PTRS = (
    "O dO LSE SEQ_Q SEQ_KV OFF_Q OFF_KV WS_LSE2 WS_DSUM WS_DQ WS_DK WS_DV WORKLIST"
).split()
_PREP_I64 = "o_b o_h do_b do_h l_b l_h".split()
_PREP_I32 = (
    "o_t do_t l_t h_q h_k h_v S_q_max S_kv_max "
    "has_len len_stride off64 q_mult q_div kv_mult kv_div ws_rows_q ws_rows_kv ws_seg "
    "zero_kv zero_dq use_worklist n_batch wl_kn0"
).split()
_CONVERT_PTRS = ("SRC", "DST", "SEQ", "OFF")
_CONVERT_I64 = ("d_b", "d_h")
_CONVERT_F32 = ("mult",)
_CONVERT_I32 = (
    "H d_t S_max has_len len_stride off64 tok_mult tok_div ws_rows ws_seg"
).split()
_DQ_PTRS = "Q K V dO dQ WS_LSE2 WS_DSUM SEQ_Q SEQ_KV OFF_Q OFF_KV".split()
_DQ_I64 = "q_b q_h k_b k_h v_b v_h do_b do_h dq_b dq_h".split()
_DQ_F32 = ("scale_log2", "ds_mult", "dq_mult")
_DQ_I32 = (
    "q_t k_t v_t do_t dq_t "
    "h_q h_k h_v G S_q_max S_kv_max has_len len_stride off64 q_mult q_div kv_mult kv_div "
    "left right bottom_right ws_rows_q ws_seg xcd_n xcd_chunk"
).split()

_ABI_LISTS = {
    "main": (_MAIN_PTRS, _MAIN_I64, _MAIN_F32, _MAIN_I32),
    "prep": (_PREP_PTRS, _PREP_I64, (), _PREP_I32),
    "convert": (_CONVERT_PTRS, _CONVERT_I64, _CONVERT_F32, _CONVERT_I32),
    "dq": (_DQ_PTRS, _DQ_I64, _DQ_F32, _DQ_I32),
}

# struct format of each kernarg kind (little endian): pointer, i64, f32, i32.
KERNARG_FORMATS = ("Q", "q", "f", "i")


def static_unit_mults(static_scale: str, dkv_mode: str) -> frozenset[str]:
    """Main-kernel multipliers that are 1.0 under scale placement ``static_scale``.

    The plan passes exactly 1.0 for these kernargs, and ``x * 1.0 == x`` for
    every fp32 ``x`` (denormals are not flushed), so the kernel compiled for
    the placement does not apply them; the kernargs stay in the ABI.
    ``fold_ds`` scales dS (``ds_mult``); ``dq_before_atomic`` scales the dQ
    partial (``dq_mult``); ``convert`` scales in the converts; dK takes
    ``dk_mult`` in direct mode unless dS is scaled, and is converted in atomic
    mode. ``"runtime"`` applies every multiplier.
    """
    _check_in("static_scale", static_scale, STATIC_SCALES)
    if static_scale == "runtime":
        return frozenset()
    unit = {"ds_mult", "dk_mult", "dq_mult"}
    if static_scale == "fold_ds":
        unit.discard("ds_mult")
    elif dkv_mode == "direct":
        unit.discard("dk_mult")
    if static_scale == "dq_before_atomic":
        unit.discard("dq_mult")
    return frozenset(unit)


def attn_bwd_params(
    stage: str, spec: Optional[AttnBwdSpec] = None
) -> tuple[tuple[str, str], ...]:
    """Ordered ``(name, struct format)`` kernargs of ``stage`` under the v3 ABI.

    ``"Q"`` is a 64-bit global pointer, ``"q"`` an i64, ``"f"`` an f32, ``"i"``
    an i32. The lists do not depend on the spec (every compile-time knob keeps
    the ABI); ``spec`` is accepted so callers can pass the instance they launch.
    """
    _check_in("stage", stage, tuple(_ABI_LISTS))
    ptrs, i64s, f32s, i32s = _ABI_LISTS[stage]
    return (
        tuple((p, "Q") for p in ptrs)
        + tuple((n, "q") for n in i64s)
        + tuple((f, "f") for f in f32s)
        + tuple((i, "i") for i in i32s)
    )


# ---------------------------------------------------------------------------
# builder
# ---------------------------------------------------------------------------

_MAIN_READONLY = frozenset(
    "Q K V dO WS_LSE2 WS_DSUM SEQ_Q SEQ_KV OFF_Q OFF_KV WORKLIST".split()
)
# The lowering's flat-work-group-size attribute has a fixed minimum of 64.
_MIN_WG_ATTR = 64


def _not_built(spec: AttnBwdSpec) -> list[str]:
    """Resolved knob values whose emission is not part of the body yet."""
    out = []
    if spec.transpose_source == "xt_lds":
        out.append("transpose_source='xt_lds'")
    if spec.global_path != "vgpr":
        out.append(f"global_path={spec.global_path!r}")
    if spec.ring_depth != 1:
        out.append(f"ring_depth={spec.ring_depth}")
    if spec.sched != "none" or spec.setprio:
        out.append(f"sched={spec.sched!r}, setprio={spec.setprio}")
    if spec.lds_swizzle and any(
        item.endswith("=xor") for item in spec.lds_swizzle.split(",")
    ):
        out.append(f"lds_swizzle={spec.lds_swizzle!r}")
    if (
        spec.transpose_source == "tr_read"
        and spec.pt_route == "relabel"
        and spec.atom_g13 == "16x16x32"
    ):
        out.append("tr_read with a K-permuted G1/G3 B operand")
    if spec.agpr_route not in AGPR_ROUTES_BUILT:
        out.append(f"agpr_route={spec.agpr_route!r}")
    if spec.agpr_route == "asm_g13" and spec.s_in_agpr:
        out.append("agpr_route='asm_g13' with s_in_agpr (G0 / G2 as asm)")
    return out


def _main_noalias(spec: AttnBwdSpec) -> frozenset:
    """Main-kernel pointers marked ``noalias`` under ``spec.ptr_noalias``."""
    if spec.ptr_noalias is False:
        return frozenset()
    if spec.ptr_noalias == "workspace":
        return _MAIN_WORKSPACE_PTRS
    return frozenset(_MAIN_PTRS)


def _declare_main(b, io, noalias: frozenset = frozenset()) -> dict:
    args = {}
    for name, fmt in attn_bwd_params("main"):
        if fmt == "Q":
            if name in ("Q", "K", "V", "dO", "dK", "dV"):
                t = PtrType(io, "global")
            elif name.startswith("WS_"):
                t = PtrType(F32, "global")
            else:
                t = PtrType(I32, "global")
            attrs = {"readonly": True} if name in _MAIN_READONLY else {}
            if name in noalias:
                attrs["noalias"] = True
            args[name] = b.param(name, t, **attrs)
        elif fmt == "q":
            args[name] = b.param(name, I64)
        elif fmt == "f":
            args[name] = b.param(name, F32)
        else:
            args[name] = b.param(name, I32)
    return args


def _decode_sequences(b, s: AttnBwdSpec, a: dict, bz):
    """``(seq_q, seq_kv)`` of batch / THD sequence ``bz``.

    Batched: the lengths are ``S_max`` or the clamped SEQ_LEN counts (runtime
    ``has_len``). THD: ``OFF_Q`` / ``OFF_KV`` entries ``bz`` and ``bz + 1``
    (int32 or int64 by the runtime ``off64``) give the first token
    ``off * mult / div`` and the length (i64), clamped to ``[0, S_max]`` and
    ``min``-ed with the SEQ_LEN count under ``has_len``; the workspace rows of
    the sequence start at its first token (``ws_seg = 0``) or at its slot
    ``bz * S_max`` (``ws_seg != 0``).
    """
    return _decode_side(b, s, a, bz, "q"), _decode_side(b, s, a, bz, "kv")


def _decode_side(b, s: AttnBwdSpec, a: dict, bz, side: str):
    """The q-side or kv-side ``SeqInfo`` of batch / THD sequence ``bz``."""
    smax, off, seq = {
        "q": ("S_q_max", "OFF_Q", "SEQ_Q"),
        "kv": ("S_kv_max", "OFF_KV", "SEQ_KV"),
    }[side]
    if s.seq_mode == "thd":
        return decode_thd(
            b,
            bz,
            s_max=a[smax],
            off_ptr=a[off],
            off64=a["off64"],
            mult=a[f"{side}_mult"],
            div=a[f"{side}_div"],
            seq_ptr=a[seq],
            has_len=a["has_len"],
            len_stride=a["len_stride"],
            ws_seg=a["ws_seg"],
        )
    return decode_batched(
        b,
        bz,
        s_max=a[smax],
        seq_ptr=a[seq],
        has_len=a["has_len"],
        len_stride=a["len_stride"],
    )


def _emit_main(spec: AttnBwdSpec, arch: str, *, sentinel: bool = True):
    """Emit the main kernel of ``spec`` on ``arch``.

    ``sentinel=False`` drops the dead-row term ``lse2 < +inf`` from the dS
    select; it exists only for the negative-control tests and is not a spec
    field.
    """
    s = validate_attn_bwd_spec(spec, arch)
    missing = _not_built(s)
    if missing:
        raise NotImplementedError(
            "the layout-general attention backward main kernel is not built for "
            + ", ".join(missing)
        )
    facts = attn_bwd_arch_facts(arch)
    geom = BwdTileGeometry.from_spec(s, arch)
    plan = lds_plan(s, arch)
    io = ir_type(s.dtype)
    D, km, kn = s.head_size, s.block_m, s.block_n

    b = IRBuilder(s.kernel_name("main"))
    a = _declare_main(b, io, _main_noalias(s))
    # Multipliers that are 1.0 under the compiled placement are not applied.
    a.update(dict.fromkeys(static_unit_mults(s.static_scale, s.dkv_mode)))
    b.kernel.attrs["max_workgroup_size"] = max(geom.threads, _MIN_WG_ATTR)
    if facts.matrix_path == "mfma" and s.agpr_alloc not in (None, AGPR_ALLOC_BACKEND):
        b.kernel.attrs["agpr_alloc"] = tuple(s.agpr_alloc)
    if s.agpr_route in SPLIT_AGPR_ROUTES:
        # VGPR-form selection for every MFMA (helpers.compile adds the comgr
        # option); the reservation lets the allocator keep dK / dV in AGPRs
        b.kernel.attrs["mfma_vgpr_form"] = True
    if s.waves_per_eu is not None:
        b.kernel.attrs["waves_per_eu"] = int(s.waves_per_eu)
    if s.scheduler_strategy is not None:
        b.kernel.attrs[SCHEDULER_STRATEGY_ATTR] = normalize_scheduler_strategy(
            s.scheduler_strategy
        )

    tid, lane, wave = lane_and_wave(b, facts.wave_size)
    atomic = s.dkv_mode == "atomic"
    # CTA decode: kv tile fastest, then the head unit (and split), then b; the
    # load-balance order permutes the kv tiles of one (unit, b).
    split_arg = a["g_split"] if atomic else None
    xcd = None
    if s.xcd_remap:
        # grid y = units * g_split: h_k (direct) or h_q / G (atomic) units
        gy = b.mul(b.div(a["h_q"], a["G"]), a["g_split"]) if atomic else a["h_k"]
        xcd = XcdRemap(n=a["xcd_n"], chunk=a["xcd_chunk"], gy=gy)
    work = decode_cta(
        b, x=b.block_id_x(), y=b.block_id_y(), z=b.block_id_z(),
        n_kv_tiles=a["n_kv_tiles"], lb_order=a["lb_order"], g_split=split_arg,
        interleave=s.lb_interleave,
        xcd=xcd,
    )  # fmt: skip
    bz = work.batch
    seq_q, seq_kv = _decode_sequences(b, s, a, bz)
    # Direct mode: the unit is the kv head and owns its query heads
    # [unit * G, unit * G + G). Atomic mode: the unit's G / g_split heads of
    # this split share one K head and one V head.
    heads = head_range(
        b, unit=work.unit, split=work.split, group=a["G"], gk=a["gk"], gv=a["gv"],
        g_split=split_arg,
    )  # fmt: skip
    hk, hv, h0, n_h = heads.hk, heads.hv, heads.h0, heads.n_heads

    def tensor(name, prefix, seq):
        return StridedTensorAddr(
            b, a[name], dtype=io, seq=seq, stage_vec=s.stage_vec,
            strides=(a[f"{prefix}_b"], a[f"{prefix}_h"], a[f"{prefix}_t"]),
        )  # fmt: skip

    q_addr = tensor("Q", "q", seq_q)
    do_addr = tensor("dO", "do", seq_q)
    k_addr = tensor("K", "k", seq_kv)
    v_addr = tensor("V", "v", seq_kv)
    nh = a["h_q"] if s.ws_layout == "token_major" else None

    def ws(name, width):
        return WorkspaceAddr(
            b, a[name], ws_rows=a["ws_rows_q"], width=width, row0=seq_q.ws_row0,
            layout=s.ws_layout, n_heads=nh,
        )  # fmt: skip

    stats = StatsView(ws("WS_LSE2", 1), ws("WS_DSUM", 1))
    sink_cls = PackedAtomicWorkspaceSink if s.head_pack else AtomicWorkspaceSink
    sink = sink_cls(b, ws("WS_DQ", D), seq=seq_q, scale=a["dq_mult"])
    zero = b.const_i32(0)
    if s.mask_class == "band":
        # Per-sequence diagonal: 0 (top-left) or len_kv - len_q (bottom-right).
        off = b.select(
            b.cmp_ne(a["bottom_right"], zero),
            b.sub(seq_kv.length, seq_q.length),
            zero,
        )
        bounds = AttnRuntimeBounds(
            seq_q.length, seq_kv.length, off, a["left"], a["right"]
        )
        mask = RuntimeBand(b, bounds, k_m0=km, k_n0=kn)
    else:
        mask = NoMaskKTail(b, seq_q.length, seq_kv.length, k_m0=km, k_n0=kn)
    lds = BwdLds(b, plan, s.dtype)
    step = BwdTileStep(
        b, geom, mask=mask, stats=stats, dq_sink=sink, lds=lds, tid=tid, lane=lane,
        wave=wave, scale_log2=a["scale_log2"], ds_mult=a["ds_mult"],
        sentinel=sentinel,
    )  # fmt: skip

    # One kv tile per CTA (natural, reverse or interleave; the plan refuses mirror
    # pairing, whose second kv tile per CTA is not built).
    _kv_tile(
        b, s, a, step, mask, kt=work.first_kt, hk=hk, hv=hv, h0=h0, n_h=n_h,
        seq_q=seq_q, seq_kv=seq_kv, q_addr=q_addr, do_addr=do_addr, k_addr=k_addr,
        v_addr=v_addr, io=io, tid=tid,
    )  # fmt: skip
    b.ret()
    return b.kernel


def _kv_tile(
    b, s, a, step, mask, *, kt, hk, hv, h0, n_h, seq_q, seq_kv, q_addr, do_addr,
    k_addr, v_addr, io, tid,
) -> None:  # fmt: skip
    """Everything the CTA does for kv tile ``kt``: staging, q loops, epilogue."""
    geom = step.geom
    km, kn = s.block_m, s.block_n
    zero, one = b.const_i32(0), b.const_i32(1)
    k0 = b.mul(kt, b.const_i32(kn))
    live = b.cmp_lt(k0, seq_kv.length)  # CTA-uniform
    k_slab = k_addr.slab(hk, k0)
    v_slab = v_addr.slab(hv, k0)
    step.stage_kv(b, k_slab, v_slab, k0, live)

    if s.head_pack:
        # pack_heads = P > 0: P query heads share the M rows of the single q
        # tile (the plan guarantees P * len_q <= kM0, so n_qt <= 1); the loop
        # visits ceil(n_h / P) head groups. pack_heads = 0 is the unpacked loop.
        pk = a["pack_heads"]
        pack_on = b.cmp_ne(pk, zero)
        per = b.select(pack_on, pk, one)
        rows_per_head = b.select(pack_on, b.smax(seq_q.length, one), b.const_i32(km))
        n_grp = b.div(b.add(n_h, b.sub(per, one)), per)
    else:
        pack_on = per = rows_per_head = None
        n_grp = n_h

    # Inverse band (empty band, a tile past len_kv or len_q == 0: 0 tiles).
    qt_start, n_qt = mask.q_tile_range(b, k0)
    # Flattened order: q tiles outer, head groups inner, so every q-tile range
    # is a contiguous run of steps; the trip counts live in SGPRs. Under
    # step_order = "head_outer" one loop runs the q tiles of each head group in
    # turn (q tiles inner, wrapping at qt_end); with several q-tile ranges a
    # loop over the head groups runs the range loops of one group per trip.
    head_outer = s.step_order == "head_outer"
    qt_end = b.add(qt_start, n_qt)
    if s.edge_tiles:
        # Interior q tiles [i0, i1) skip the per-cell selects: the steps run
        # as edge [qt_start, i0), interior [i0, i1) and edge [i1, qt_end)
        # loops in the single loop's order, so the accumulation order (and
        # dK / dV bit for bit) is the same. A packed tile is an edge tile.
        i0, i1 = mask.interior_q_range(b, k0, qt_start, n_qt)
        if pack_on is not None:
            i1 = b.select(pack_on, i0, i1)
        ranges = ((qt_start, i0, True), (i0, i1, False), (i1, qt_end, True))
    else:
        ranges = ((qt_start, qt_end, True),)
    # head_outer with several q-tile ranges: a loop over the head groups (no
    # trip when the inverse band is empty) around one loop per range.
    grouped = head_outer and len(ranges) > 1
    if grouped:
        trips = [b.readfirstlane(b.sub(e, s0)) for s0, e, _ in ranges]
        n_outer = b.readfirstlane(b.select(b.cmp_lt(zero, n_qt), n_grp, zero))
    else:
        trips = [b.readfirstlane(b.mul(n_grp, b.sub(e, s0))) for s0, e, _ in ranges]
    dkv_slabs = None
    if s.dkv_slab_hoist and s.dkv_mode == "direct" and s.seq_mode == "thd":
        # THD: form the dK / dV slabs before the q loops, so two 64-bit
        # pointers (not the i64 kv token base, both tensor pointers and their
        # strides) stay live across them; the gfx950 THD kernels sit at the
        # SGPR ceiling and spill without it. The empty asm makes each pointer
        # opaque, else the backend sinks the address arithmetic past the loops.
        dkv_slabs = {}
        for w in ("dk", "dv"):
            sl = _dkv_slab(b, s, a, w, hk=hk, k0=k0, seq_kv=seq_kv, io=io)
            ptr = b.inline_asm(
                "", "=s,0", [sl.ptr], result_type=sl.ptr.type, sideeffect=False,
                result_name_hint="dkvptr",
            )  # fmt: skip
            dkv_slabs[w] = dataclasses.replace(sl, ptr=ptr)
    acc0 = step.zero_acc(b)  # after the exit test: accumulators start here
    fm, fn = geom.g1.frags_m, geom.g1.frags_n
    n_acc = len(acc0.flat())

    def carried(prefix, acc, qt0, *, grp=True):
        vals = acc.flat()
        half = len(vals) // 2
        names = [f"{prefix}dk{i}" for i in range(half)]
        names += [f"{prefix}dv{i}" for i in range(half)]
        extra = [] if qt0 is None else [qt0]
        names += [] if qt0 is None else [f"{prefix}qt"]
        if grp:
            names.append(f"{prefix}grp")
            extra.append(zero)
        return list(zip(names, vals + extra))

    def acc_of(cur):
        return DkDvAcc.from_flat(cur[:n_acc], fm, fn) if n_acc else acc0

    def head_of(gi):
        """``(h, first)``: the query head of head group ``gi`` (the first
        head of the group when packing, ``first`` its offset in the unit)."""
        if s.head_pack:
            first = b.mul(gi, per)
            return b.add(h0, first), first
        return b.add(h0, gi), None

    def q_steps(acc, qt0, trip, *, edge, name, end, wrap_at=None, group=None):
        """``trip`` steps from q tile ``qt0``.

        Default: from head group 0, head groups inner (wrapping at ``n_grp``).
        ``wrap_at = (lo, hi)``: from head group 0, q tiles inner over ``[lo,
        hi)``. ``group = (gi, h, first)``: the q tiles of that head group only
        (the caller's group loop has rebased the dQ sink on ``h``).

        ``q_rotate``: the range ``[qt0, end)`` starts at ``qt0 + kt mod len``
        and wraps to ``qt0`` past its end, so the kv-tile CTAs of one head
        walk its q tiles (and dQ rows) out of lockstep. A range keeps its q
        tiles and its edge flag; only the order of its steps changes (with
        ``wrap_at``, a head group ends when its q tiles come back to the
        rotated start).
        """
        start0 = qt0
        if s.q_rotate:
            qt0 = b.add(start0, b.mod(kt, b.smax(b.sub(end, start0), one)))
        loop = b.scf_for_iter(
            zero, trip, one, carried(f"bwd_{name}", acc, qt0, grp=group is None),
            iv_name=f"bwd_{name}it", elide_trailing_barrier=False,
        )  # fmt: skip
        with loop as (_it, cur):
            if group is not None:
                qt, gi = cur[n_acc], group[0]
                qt_next = b.add(qt, one)
                if s.q_rotate:
                    qt_next = b.select(b.cmp_ge(qt_next, end), start0, qt_next)
                nxt_idx = (qt_next,)
            elif wrap_at is None:
                qt, gi = cur[n_acc], cur[n_acc + 1]
                # next (q tile, head group): head groups inner
                nxt = b.add(gi, one)
                wrap = b.cmp_ge(nxt, n_grp)
                qt_next = b.add(qt, b.select(wrap, one, zero))
                if s.q_rotate:
                    qt_next = b.select(b.cmp_ge(qt_next, end), start0, qt_next)
                gi_next = b.select(wrap, zero, nxt)
                nxt_idx = (qt_next, gi_next)
            else:
                qt, gi = cur[n_acc], cur[n_acc + 1]
                # next (q tile, head group): q tiles inner, wrapping at hi
                lo, hi = wrap_at
                nxt = b.add(qt, one)
                wrap = b.cmp_ge(nxt, hi)
                qt_next = b.select(wrap, lo, nxt)
                if s.q_rotate:
                    # the group is done when its q tiles are back at the start
                    wrap = b.cmp_eq(qt_next, qt0)
                nxt_idx = (qt_next, b.add(gi, b.select(wrap, one, zero)))
            q0 = b.mul(qt, b.const_i32(km))
            h, first = head_of(gi) if group is None else group[1:]
            if s.head_pack:
                rows = PackedRows(
                    head0=h, n_heads=b.smin(per, b.sub(n_h, first)),
                    rows_per_head=rows_per_head, row0=q0,
                )  # fmt: skip
                packed = PackedStep(rows=rows, q_addr=q_addr, do_addr=do_addr)
                q_slab = do_slab = None
            else:
                packed = None
                # interior steps: every row is < len_q, so no row clamp
                # (unless interior_clamp keeps it)
                q_slab = q_addr.slab(h, q0, clamp=edge or s.interior_clamp)
                do_slab = do_addr.slab(h, q0, clamp=edge or s.interior_clamp)
            if group is None:
                step.sink.begin_head(b, h)
            out = step.run_q_step(
                b, acc_of(cur), q_slab, do_slab, qt, h, k0, edge=edge, packed=packed,
            )  # fmt: skip
            b.scf_yield(*out.flat(), *nxt_idx)
        res = list(loop.results)[:n_acc]
        return DkDvAcc.from_flat(res, fm, fn) if n_acc else acc0

    acc = acc0
    names = ("q", "qi_", "qe_")
    if not grouped:
        for (s0, e, edge), trip, name in zip(ranges, trips, names):
            wrap_at = (s0, e) if head_outer else None
            acc = q_steps(acc, s0, trip, edge=edge, name=name, end=e, wrap_at=wrap_at)
    else:
        outer = b.scf_for_iter(
            zero, n_outer, one, carried("bwd_g", acc0, None),
            iv_name="bwd_git", elide_trailing_barrier=False,
        )  # fmt: skip
        with outer as (_it, cur):
            gi = cur[n_acc]
            h, first = head_of(gi)
            step.sink.begin_head(b, h)  # dominates the range loops' steps
            inner = acc_of(cur)
            for (s0, e, edge), trip, name in zip(ranges, trips, names):
                inner = q_steps(
                    inner, s0, trip, edge=edge, name=name, end=e,
                    group=(gi, h, first),
                )  # fmt: skip
            b.scf_yield(*inner.flat(), b.add(gi, one))
        res = list(outer.results)[:n_acc]
        acc = DkDvAcc.from_flat(res, fm, fn) if n_acc else acc0
    acc = step.finish(b, acc)
    if s.dkv_mode == "atomic":
        _atomic_epilogue(
            b, s, a, step, acc, hk=hk, hv=hv, k0=k0, seq_kv=seq_kv, live=live
        )
    else:
        _direct_epilogue(
            b, s, a, step, acc, hk=hk, k0=k0, seq_kv=seq_kv, io=io, tid=tid,
            slabs=dkv_slabs,
        )  # fmt: skip


def _atomic_epilogue(b, s, a, step, acc, *, hk, hv, k0, seq_kv, live) -> None:
    """fp32 dK / dV of the key tile added into the ``WS_DK`` / ``WS_DV`` workspace.

    Inside ``scf_if(live)`` (a tile past ``len_kv`` adds nothing). Every
    accumulator element of the lane is one ``global_atomic_add_f32`` (agent
    scope) into kv head ``hk`` (dK, times ``dk_mult``) or ``hv`` (dV), with the
    workspace rebased in i64 per head and tile row. One branch per key row of
    the lane: a row is added only when it is inside the sequence (``k <
    len_kv``) and the workspace (``< ws_rows_kv``), and the atomics of a row
    share its row base. Batched rows ``[len_kv, S_kv_max)`` keep the zeros
    written by prep, and the dK / dV converts store them as zeros. With
    ``dkv_atomic_guard = "select"`` there is no row branch: a row outside the
    sequence or the workspace adds ``+0.0`` at its clamped workspace row,
    which leaves every sum bit for bit unchanged.
    """
    D = s.head_size
    tok_major = s.ws_layout == "token_major"
    select = s.dkv_atomic_guard == "select"
    with b.scf_if(live):
        for which, ptr, head, n_heads, mult in (
            ("dk", "WS_DK", hk, a["h_k"], a["dk_mult"]),
            ("dv", "WS_DV", hv, a["h_v"], None),
        ):
            slab = WorkspaceAddr(
                b, a[ptr], ws_rows=a["ws_rows_kv"], width=D, row0=seq_kv.ws_row0,
                layout=s.ws_layout, n_heads=n_heads if tok_major else None,
            ).slab(head, k0)  # fmt: skip
            for row, items in step.dkv_row_slots(b, which, acc):
                if mult is not None:
                    items = [(c, b.fmul(v, mult)) for c, v in items]
                in_seq = b.cmp_lt(b.add(k0, row), seq_kv.length)
                if select:
                    slab.atomic_add_row_select(row, items, in_seq)
                else:
                    slab.atomic_add_row(row, items, in_seq)


def _dkv_slab(b, s, a, which, *, hk, k0, seq_kv, io):
    """The ``which`` (``"dk"`` / ``"dv"``) output slab of kv head ``hk`` from
    row ``k0`` (rows unclamped)."""
    return StridedTensorAddr(
        b, a[{"dk": "dK", "dv": "dV"}[which]], dtype=io, seq=seq_kv,
        stage_vec=s.stage_vec,
        strides=(a[f"{which}_b"], a[f"{which}_h"], a[f"{which}_t"]),
    ).slab(hk, k0, clamp=False)  # fmt: skip


def _direct_epilogue(
    b, s, a, step, acc, *, hk, k0, seq_kv, io, tid, slabs=None
) -> None:
    """dK / dV rows of the key tile to the output dtype at their strides.

    Rows ``k < len_kv`` get the values (``dk_mult`` on dK); batched rows
    ``len_kv <= k < S_kv_max`` get exact zeros; rows at or past ``S_kv_max``
    are not touched. The stores are predicated per row (after the loop).
    ``dkv_epilogue`` picks the path (see :class:`AttnBwdSpec`); every path
    stores the same values to the same elements. ``slabs`` (``{"dk": ...,
    "dv": ...}``) are slabs formed before the q loops; ``None`` forms each
    one here.
    """
    outputs = (("dk", "dK", "dk", a["dk_mult"]), ("dv", "dV", "dv", None))

    def out_slab(ptr, prefix):
        if slabs is not None:
            return slabs[prefix]
        return _dkv_slab(b, s, a, prefix, hk=hk, k0=k0, seq_kv=seq_kv, io=io)

    if s.dkv_epilogue == "direct":
        # every accumulator slot stored from its register: no LDS, no barrier
        for which, ptr, prefix, mult in outputs:
            slab = out_slab(ptr, prefix)
            cells = step.dkv_cells(b, which, acc, mult)
            _store_dkv_cells(b, s, a, slab, cells, k0=k0, seq_kv=seq_kv, io=io)
        return
    vec = s.dkv_store_vec
    zero = b.zero_vec(io, vec)
    kw = dict(k0=k0, seq_kv=seq_kv, tid=tid, zero=zero)
    if s.dkv_epilogue == "lds_staged_pair":
        # dK in buffer 0 and dV in buffer 1, one barrier before the reads
        staged = []
        for slot, (which, ptr, prefix, mult) in enumerate(outputs):
            slab = out_slab(ptr, prefix)
            last = slot == len(outputs) - 1
            read = step.dkv_rows(b, which, acc, mult, vec=vec, slot=slot, sync=last)
            staged.append((slab, read))
        for slab, read in staged:
            _store_dkv_rows(b, s, a, step, slab, read, **kw)
        b.sync()  # reads of the epilogue buffers done before they are reused
        return
    for which, ptr, prefix, mult in outputs:
        slab = out_slab(ptr, prefix)
        read = step.dkv_rows(b, which, acc, mult, vec=vec)
        _store_dkv_rows(b, s, a, step, slab, read, **kw)
        b.sync()  # reads of the epilogue buffer done before it is reused


def _store_dkv_rows(b, s, a, step, slab, read, *, k0, seq_kv, tid, zero) -> None:
    """The ``kN0 x D`` tile from ``read`` to ``slab`` as rows of
    ``dkv_store_vec`` elements, chunks dealt to the CTA's threads."""
    D, kn, vec = s.head_size, s.block_n, s.dkv_store_vec
    threads = step.geom.threads
    per_row = D // vec
    total = kn * per_row
    for it in range(-(-total // threads)):
        chunk = tid if it == 0 else b.add(tid, b.const_i32(it * threads))
        r = b.div(chunk, b.const_i32(per_row))
        c = b.mul(b.mod(chunk, b.const_i32(per_row)), b.const_i32(vec))
        k = b.add(k0, r)
        store, row_live = dkv_store_rule(
            b, k, seq_kv.length, seq_mode=s.seq_mode, s_kv_max=a["S_kv_max"]
        )
        if (it + 1) * threads > total:
            store = b.land(store, b.cmp_lt(chunk, b.const_i32(total)))
        with b.scf_if(store):
            v = b.select(row_live, read(r, c), zero)
            slab.store_vec(k, c, v)


def _store_dkv_cells(b, s, a, slab, cells, *, k0, seq_kv, io) -> None:
    """Each lane's accumulator cells (:meth:`BwdTileStep.dkv_cells`) to
    ``slab``, one element store per cell, predicated per key row."""
    zero = b.vec_extract(b.zero_vec(io, 1), 0)
    for row, items in cells:
        k = b.add(k0, row)
        store, row_live = dkv_store_rule(
            b, k, seq_kv.length, seq_mode=s.seq_mode, s_kv_max=a["S_kv_max"]
        )
        with b.scf_if(store):
            for col, v in items:
                slab.store(k, col, b.select(row_live, v, zero))


def build_attn_bwd_main(spec: AttnBwdSpec, *, arch: str):
    """Build the layout-general backward main kernel (``rocke.attn_bwd.v3``).

    The spec is resolved and validated for ``arch`` first (an illegal
    geometry raises ``ValueError``); knob values whose emission is not built
    yet raise ``NotImplementedError`` before any IR is emitted.
    """
    return _emit_main(spec, arch)
