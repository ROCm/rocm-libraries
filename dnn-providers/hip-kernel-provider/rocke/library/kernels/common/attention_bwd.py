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
plan in :mod:`kernels.common.attention_bwd_plan` chooses them. The
compile-time switches among them only decide what the decode reads:
``xcd_remap`` the runtime ``xcd_n`` / ``xcd_chunk`` (the plan sets it with a
non-zero ``xcd_n``), ``lb_interleave`` the interleaved order, and
``worklist`` (THD) the ``WORKLIST`` items under the runtime ``use_worklist``
(the plan sets it with the work list); a kernel without a switch emits none of
its IR.

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
into the ``WS_DK`` / ``WS_DV`` workspace with atomics (two converts follow);
with ``dkv_sink = "slices"`` it stores them into its own workspace slice
instead (no atomics) and two reduce launches sum the slices in a fixed order.
The CTA decode (:mod:`kernels.common._attention_bwd_worklist`) applies the
runtime load-balance order (natural, reverse, or interleave under the
``lb_interleave`` decode): one kv tile per CTA, or, in a kernel built with
``lb_mirror`` and a runtime ``pair = 1`` (mirror pairing), kv tile ``t`` and
then kv tile ``n_kv_tiles - 1 - t`` in one CTA (the second tile behind a
barrier, the same code under loop names of its own), or, in a ``worklist``
kernel with ``use_worklist`` set, the (sequence, kv tile) of the THD work
list. ``loop_sync = "counted"`` (CDNA) loads the next q step's Q / dO tile and
stats into registers before the step's dQ atomics and ends every in-loop
barrier's wait at the atomics (:mod:`kernels.common._attention_bwd_counted`).
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
  ``kN0*D / (2*wave*W)`` each (K and V: ``W`` is the G0 / G2 grid's columns
  on the small-tile grids, :func:`._attention_bwd_tiles.kv_operand_split`);
* dK, dV fp32 accumulators: ``kN0*D / (wave*W)`` each (AGPRs on MFMA unless
  ``agpr_alloc = (0, 0)``; none in registers when ``acc_in_lds``). Under the
  attr route (an AGPR form without a split ``agpr_route``) the backend also
  keeps S / dP and the dQ partial in AGPRs (see :func:`_agpr_placement`);
* per-step operands (Q and dO as A and as B): ``2*kM0*D / (2*wave)``;
* register prefetch of the next Q / dO (``global_path = "vgpr"`` with
  ``ring_depth >= 2``; the synchronous stage tables prefetch nothing):
  ``kM0*D / (wave*W)`` per register set in flight (``ring_depth - 1`` sets,
  at least one); the DMA ring (``global_path = "dma"``) holds none;
* S and dP fp32 accumulators ``2*kM0*kN0 / (wave*W)`` (AGPRs when
  ``s_in_agpr``) plus the P and dS operands ``kM0*kN0 / (wave*W)`` (``W``:
  the G0 / G2 grid's waves, fewer on a replicated small-tile grid);
* dQ partial ``kM0*D / (wave*W)``; dS operands for G4 ``2*kM0*k4a / (2*wave)``
  under ``g4_slicing = "step"`` (``k4a`` = the G4 atom K: G4 is emitted one
  atom-K step at a time, so the live operand does not grow with kK4), and
  ``2*kM0*kK4 / (2*wave)`` under kK4 key slicing, plus the streamed K^T
  slices ``2*D*kK4 / (2*wave*g4c)`` when K^T is read from LDS; under
  ``dq_mode = "split"`` the main kernel has no G4, so neither term (nor K^T)
  is counted;
* an allowance of 24 for addresses, loop state, lse2 / Dsum and masks.

``dq_mode = "split"`` pairs this main kernel (dK / dV only, a ``NullSink``) with
the Q-parallel dQ kernel of :mod:`kernels.common.attention_bwd_dq`
(:meth:`AttnBwdSpec.dq_spec`); :data:`SPLIT_DQ_ARCHS` lists its targets.
"""

from __future__ import annotations

import dataclasses
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
from rocke.helpers.spec import non_default_fields, salted_kernel_name

from kernels.common._attention_bwd_addr import (
    PackedRows,
    StridedTensorAddr,
    WorkspaceAddr,
    chunk_coords,
    deal_chunks,
    decode_batched,
    decode_thd,
)
from kernels.common._attention_bwd_band import (
    NoMaskKTail,
    RuntimeBand,
    dkv_store_rule,
)
from kernels.common._attention_bwd_caps import BWD_BARRIER_FENCE
from kernels.common._attention_bwd_body import (
    DEFAULT_G4_SLICING,
    DEFAULT_STAGE_ORDER,
    G4_AFTER_G3_ORDERS,
    G4_SLICINGS,
    STAGE_ORDERS,
    COPY_ORDERS,
    BwdLds,
    BwdTileGeometry,
    BwdTileStep,
    DkDvAcc,
    PackedStep,
    QStepInputs,
    RingStep,
    StatsView,
)
from kernels.common._attention_bwd_counted import COUNTED_ISSUES, StagedStep
from kernels.common._attention_bwd_frag import ir_type
from kernels.common._attention_bwd_gemm import lane_and_wave
from kernels.common._attention_bwd_lds import (
    DKV_EPILOGUES,
    DMA_FILLED,
    LDS_ORDERS,
    BwdLdsRequest,
    LdsPlan,
    plan_bwd_lds,
)
from kernels.common._attention_bwd_refusals import resource_refusal
from kernels.common._attention_bwd_ring import QRing, PrefetchStep, PrefetchTile
from kernels.common._attention_bwd_sched import (
    FENCED_SCHEDS,
    SCHEDS,
    SPLIT_DEFERRED,
    STAGE_TABLE_SCHEDS,
)
from kernels.common._attention_bwd_sinks import (
    AtomicWorkspaceSink,
    NullSink,
    PackedAtomicWorkspaceSink,
)
from kernels.common._attention_bwd_tiles import (
    check_lds_route_grid,
    grid_waves,
    kv_operand_split,
    natural_grids,
)
from kernels.common._attention_bwd_valu import BF16_CASTS, BWD_EXP2S, STATS_VECS
from kernels.common._attention_bwd_worklist import (
    WorkList,
    XcdRemap,
    decode_cta,
    head_range,
)

__all__ = [
    "ATTN_BWD_ABI",
    "BWD_ARCHS",
    "BWD_DTYPES",
    "BWD_HEAD_SIZES",
    "AttnBwdSpec",
    "BF16_CASTS",
    "BWD_EXP2S",
    "BwdArchFacts",
    "attn_bwd_arch_facts",
    "attn_bwd_default_geometry",
    "attn_bwd_params",
    "NOT_YET_EFFECTIVE_KNOBS",
    "DECLARATION_ONLY",
    "DKV_ATOMIC_GUARDS",
    "DKV_EPILOGUES",
    "DKV_SINKS",
    "DKV_SINK_NOT_BUILT",
    "DKV_CELL_VECS",
    "NARROW_VECS",
    "WS_BOUNDS",
    "DKV_STORE_VECS",
    "G4_SLICINGS",
    "INERT_KNOBS",
    "LDS_ORDERS",
    "AGPR_ROUTES",
    "ASM_CLUSTERS",
    "BARRIER_MERGES",
    "COUNTED_ISSUES",
    "LOOP_SYNCS",
    "BWD_BARRIER_FENCE",
    "RING_ISSUES",
    "STATS_VECS",
    "STEP_ORDERS",
    "WAVES_PER_EU_VALUES",
    "SCHEDS",
    "STAGE_TABLE_SCHEDS",
    "FENCED_SCHEDS",
    "agpr_copy_budget",
    "build_attn_bwd_main",
    "check_counted_sync",
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
# Occupancy hints (waves per SIMD the backend allocates registers for; MFMA
# targets: the arch-VGPR + AGPR estimate must fit 512 / n; WMMA: None, 1, 2).
WAVES_PER_EU_VALUES = (1, 2, 3, 4)
ATOMS = ("16x16x16", "16x16x32", "wmma16x16x16")
PT_ROUTES = ("relabel", "lds")
RESIDENCIES = ("reg", "lds")
TRANSPOSE_SOURCES = ("xt_lds", "tr_read", "lds_plain", "wmma_lds")
GLOBAL_PATHS = ("vgpr", "dma")
SWIZZLES = ("xor", "pad8", "pad16", "pad32", "none")
# SCHEDS, STAGE_TABLE_SCHEDS and FENCED_SCHEDS come from _attention_bwd_sched
# (one source for the scheduler and the spec).
# Elements per row store of the LDS-staged dK / dV epilogue (and per LDS read
# of the staging buffer or of the fp32 accumulator images).
DKV_STORE_VECS = (8, 4, 2, 1)
# Row guard of the atomic dK / dV epilogue: one branch per key row around its
# atomics ("branch"), or every atomic issued with +0.0 for a row outside the
# sequence or the workspace ("select", branch-free; identical sums).
DKV_ATOMIC_GUARDS = ("branch", "select")
# Sink of the fp32 dK / dV partials in atomic mode (``dkv_mode = "atomic"``):
# "atomic" adds every accumulator element into the zero-filled WS_DK / WS_DV
# workspace with fp32 atomics (two converts follow); "slices" stores each
# CTA's partial with plain stores into its own workspace slice (no atomics,
# no zero fill) and the reduce kernel sums the slices of every kv head in a
# fixed order (deterministic); "last_cta" (per-tile arrival counters, the last
# CTA reduces) is registered but not built (DKV_SINK_NOT_BUILT). Inert in
# direct mode.
DKV_SINKS = ("atomic", "slices", "last_cta")
DKV_SINK_NOT_BUILT = MappingProxyType(
    {
        "last_cta": "the arrival-counter sink needs agent-scope release / "
        "acquire fences around a workgroup barrier that orders memory: the IR "
        "has no fence op, global_atomic_add carries no sync scope, and "
        "tile.sync lowers to s_waitcnt + s_barrier without workgroup fences, so "
        "the slice stores of the other waves are not ordered before the "
        "counter increment, nor the last CTA's slice loads after it",
    }
)
# Bounds of the main kernel's workspace accesses: address clamps and per-row
# branches ("clamp"), or a buffer descriptor whose range ends at the last
# valid row ("buffer": out-of-range loads read 0 before the same neutral
# select, out-of-range stores are dropped); head-major workspace only.
WS_BOUNDS = ("clamp", "buffer")
# Elements per store of the per-slot dK / dV epilogue (dkv_epilogue =
# "direct"): 1, or 4 after an in-quad 4 x 4 transpose (C layouts where a quad
# holds 4 consecutive columns of 4 consecutive rows).
DKV_CELL_VECS = (1, 4)
# User-tensor access width (elements) of a stage_vec = 1 instance; the plan
# narrows it to what the request's strides and base alignment allow.
NARROW_VECS = (1, 2, 4)
# The stage-table count model (``_attention_bwd_sched``) places G1 + G2 in
# stage B, G3 in C and G4 with the dS slices in D, i.e. the default order.
_STAGE_ORDER_SCHED_RULE = (
    "stage_order={!r} is not legal with sched={!r}: the stage-table count "
    "model assumes the default G1, G3, G4 order"
)
# The XT images are written from the published Q / dO tile of the step and
# published by the barrier before the first G1 / G3: one image per CTA, built
# on the synchronous body (no ring slots, no merged or counted barriers).
_XT_SYNC_BODY = (
    "transpose_source='xt_lds' needs the synchronous body: global_path='vgpr', "
    "ring_depth=1, barrier_merge='off' and loop_sync != 'counted' (the XT "
    "images are not planned per ring slot and their publishing barrier is the "
    "synchronous step's)"
)
# A fenced schedule reads dO^T one stage ahead, in stage A, before the barrier
# that publishes the Q^T / dO^T images.
_XT_FENCED_SCHED = (
    "transpose_source='xt_lds' with sched={!r}: the fenced read-ahead of dO^T "
    "would read the XT image before the barrier that publishes it"
)
# The row XOR view has no run-time slot shift: a ring-slotted row-major buffer
# (q / do of a register ring, ds of the merged barrier) cannot take 'xor'.
_RING_ROW_XOR = (
    "lds_swizzle {}=xor on a ring-slotted row-major buffer: the XOR view has "
    "no run-time slot shift (pad / none, or the DMA slab layout, instead)"
)
# Layout-lever combinations whose production-comgr compile spills arch VGPRs
# (scratch), refused by the validator. The line-XOR address of a row-major
# buffer is not an immediate offset of one base (the XOR does not commute with
# the fragment and K-step offsets), so every swizzled access of the q loop
# keeps its own address register; the one-wave gfx1151 d64 tile has no room
# for the XT writer and the image reads. Key: arch and the resolved
# (head_size, dtype, block_m, block_n, waves, seq_mode, mask_class,
# edge_tiles, head_pack, dkv_mode, stage_vec, transpose_source, lds_swizzle),
# see :func:`_layout_spill_key`. The IR suite's resource gate compiles every
# other layout-lever instance of these tiles spill-free.
_ROW_XOR_ALL = (
    "dkv_out=xor,do=xor,ds=xor,k_res=xor,k_stage=xor,pt=xor,q=xor,v_res=xor,"
    "v_stage=xor"
)
LAYOUT_SPILLING_COMBOS = frozenset(
    (
        ("gfx1151", 64, "fp16", 16, 16, 1, "batched", "band", False, False, "direct", 8, "xt_lds", None),
        ("gfx1151", 64, "fp16", 16, 16, 1, "batched", "band", True, False, "direct", 8, "xt_lds", None),
        ("gfx1151", 64, "fp16", 16, 16, 1, "batched", "band", False, False, "direct", 1, "xt_lds", None),
        ("gfx1151", 64, "fp16", 16, 16, 1, "thd", "band", False, False, "atomic", 8, "xt_lds", None),
        ("gfx1151", 64, "fp16", 16, 16, 1, "batched", "band", True, False, "direct", 8, "wmma_lds", _ROW_XOR_ALL),
        ("gfx1151", 64, "fp16", 16, 16, 1, "batched", "band", False, True, "direct", 8, "wmma_lds", _ROW_XOR_ALL),
        ("gfx1201", 64, "fp16", 16, 16, 1, "batched", "band", False, True, "direct", 8, "wmma_lds", _ROW_XOR_ALL),
        ("gfx1201", 128, "fp16", 16, 16, 1, "batched", "band", True, False, "direct", 8, "wmma_lds", _ROW_XOR_ALL),
        ("gfx1201", 128, "fp16", 16, 16, 1, "batched", "band", False, True, "direct", 8, "wmma_lds", _ROW_XOR_ALL),
        ("gfx1201", 128, "fp16", 16, 16, 1, "thd", "none", False, True, "atomic", 1, "wmma_lds", _ROW_XOR_ALL),
        ("gfx942", 128, "fp16", 16, 128, 8, "batched", "band", True, False, "direct", 8, "lds_plain", _ROW_XOR_ALL),
        ("gfx950", 64, "fp16", 32, 256, 8, "batched", "band", False, True, "direct", 8, "tr_read", _ROW_XOR_ALL),
    )
)  # fmt: skip


def _layout_spill_key(s, arch: str) -> tuple:
    """The :data:`LAYOUT_SPILLING_COMBOS` key of a resolved spec."""
    return (arch, s.head_size, s.dtype, s.block_m, s.block_n, s.waves, s.seq_mode,
            s.mask_class, s.edge_tiles, s.head_pack, s.dkv_mode, s.stage_vec,
            s.transpose_source, s.lds_swizzle)  # fmt: skip


# The G4 dS key slicing is built on the synchronous body with atomic dQ and no
# stage table (its slice reads and the moved dS barrier of slice_db_g3 are
# placed against the three barriers of that body); under xt_lds K^T comes from
# registers.
_G4_SLICING_SYNC_BODY = (
    "g4_slicing={!r} needs the synchronous body (global_path='vgpr', "
    "ring_depth=1, barrier_merge='off', loop_sync != 'counted'), dq_mode='atomic', "
    "sched='none' without setprio, and kt_source='reg' under xt_lds"
)
# A schedule value (and setprio) is built on the synchronous body, the Q / dO
# prefetch ring (register or DMA path), the merged-barrier ring and the counted
# waits: its stage table models each of them (``StageTable.pipeline``), with G4
# and its dQ atomics inside the step, and the split-dQ main kernel (no G4: the
# stages A..C, on the synchronous body and the prefetch rings, the only bodies
# split dQ runs on). ``stage_table_atomics`` defers the dQ atomics, which the
# split-dQ main kernel does not issue: refused there (``SPLIT_DEFERRED``).
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
# s_nop count that ends each block; both are legal with ``asm_g13`` only, as
# is ``asm_load_hoist`` (the second G1 / G3 GEMM's LDS operand reads issued
# before the first one's asm cluster).
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
# the dV / dK GEMMs); their swizzle is fixed to the slab XOR. K (and K^T) is
# staged once per CTA through registers on both paths.
TR_READ_DMA_BUFFERS = DMA_FILLED
# The DMA ring stages one head per q step (a wave-uniform source pointer per
# copy): head packing stays on the register path.
_DMA_HEAD_PACK = "global_path='dma' excludes head_pack=True"
# The prefetch ring's in-loop barriers are always LDS-only; the barrier-flavour
# knob applies to the synchronous staging only (and, where it is built, to the
# merged-barrier 2-slot ring, which owns the staging instead of the prefetch
# ring: the two are exclusive by value).
_RING_LOOP_SYNC = (
    "ring_depth > 1 requires loop_sync='full' (the ring owns its barriers)"
)


def _merged_ring(spec) -> bool:
    """The merged-barrier 2-slot ring stages the q steps (not the prefetch
    ring); ``barrier_merge`` is read defensively (absent: off)."""
    return getattr(spec, "barrier_merge", "off") == "on"


# Barrier flavour of the in-loop barriers of a q step (they publish LDS writes
# or retire LDS reads): "full" drains VMEM and LDS (``s_waitcnt vmcnt(0)
# lgkmcnt(0)``, so the step's dQ atomics too); "lds_only" drains LDS only;
# "counted" (CDNA) loads the next step's Q / dO tile and stats into registers
# before the step's dQ atomics (``counted_issue``) and waits ``vmcnt(N)
# lgkmcnt(0)`` with N the atomics issued after the wave's last global load, so
# every load has retired and only the atomics stay in flight
# (``_attention_bwd_counted``).
LOOP_SYNCS = ("full", "lds_only", "counted")
# Order of the flattened (q tile, head group) steps of a CTA: "qtile_outer"
# (head groups inner) or "head_outer" (q tiles inner). The dK / dV sums run in
# step order, so "head_outer" reorders the fp32 accumulation when a CTA has
# more than one head group and more than one q tile.
STEP_ORDERS = ("qtile_outer", "head_outer")
# Barriers per q step: "off" keeps the three in-loop barriers (after the Q / dO
# staging, before G4, at the step end); "on" (with ring_depth = 2) keeps only
# the pre-G4 one. Q / dO / lse2 / Dsum become a 2-slot LDS ring and dS gets 2
# slots: step i reads slot i % 2 and, before its barrier, writes the next step's
# tiles into the other slot, so one barrier publishes this step's dS and the
# next step's inputs (one more barrier per q loop and one after the loops).
# The next step's global loads are issued before the dQ atomics of this step,
# so no VMEM wait of a step waits on its own atomics. Relabel route on gfx942 /
# gfx950, G1 and G3 before G4 (the hazard proof is in the body).
BARRIER_MERGES = ("off", "on")
# Where barrier_merge issues the next step's global loads: at the top of the
# step ("early": latency hidden behind G0..G3, the tiles live in registers
# meanwhile) or right before their LDS stores ahead of the merged barrier
# ("late": fewer live registers, exposed latency).
RING_ISSUES = ("early", "late")
BARRIER_MERGE_ARCHS = ("gfx942", "gfx950")
BARRIER_MERGE_STAGE_ORDERS = ("g1g3g4", "g3g1g4")
# An LDS-only drain is proved for the register staging path only: there the
# global loads that feed a ds_write are waited by data dependency in their own
# wave. A DMA path writes LDS from VMEM, so its barriers need the VMEM wait.
_DMA_LOOP_SYNC = "loop_sync='lds_only' requires global_path='vgpr'"

# ``dq_mode = "split"``: the main kernel computes dK / dV only (no G4, no dS
# tile, no dQ atomics, no dQ workspace) and the separate Q-parallel dQ kernel
# (:mod:`kernels.common.attention_bwd_dq`) recomputes S and dP per kv tile and
# stores dQ once, without atomics. Built on the MFMA wave64 targets of
# SPLIT_DQ_ARCHS (its dS^T hand-off is the C-to-A register relabel, which the
# WMMA targets do not have); the validator refuses it elsewhere.
SPLIT_DQ_AVAILABLE = True
SPLIT_DQ_ARCHS = ("gfx942", "gfx950")

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
# sweep must not count their values as distinct configurations. Empty: every
# knob's emission is built (``block_k4`` is inert only under the default
# ``g4_slicing = "step"``, see INERT_KNOBS).
NOT_YET_EFFECTIVE_KNOBS = MappingProxyType({})

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
    default in atomic mode. With ``dkv_sink = "slices"`` the epilogue stages
    the fp32 partials through the buffer, so the knob picks the code, except
    with LDS accumulators (their fp32 images are read directly).
    """
    if spec.dkv_mode != "atomic":
        return False
    if spec.dkv_sink == "slices":
        return bool(spec.acc_in_lds)
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
# choice only sizes the allocation (with the slice sink both apply: the fp32
# partials are staged and stored as rows of dkv_store_vec floats, except
# from the per-slot "direct" epilogue); direct dK / dV have no atomic row
# guard and no sink, nor does the slice sink have a row guard; without
# resident buffers both placement orders put every buffer at the same offset;
# a swizzle entry that changes no planned buffer changes nothing.
def _slice_sink(spec) -> bool:
    return spec.dkv_mode == "atomic" and spec.dkv_sink == "slices"


def _scalar_ws_stats(spec) -> bool:
    """The step reads its lse2 / Dsum rows with the scalar workspace loads of
    :meth:`StatsView.rows` (the synchronous step, the register and merged
    rings' row loads): one stats row per load (``stats_vec = 1``), unpacked
    steps (a packed step rebases each row on its own head), not the counted
    waits (their staged loads, :meth:`StatsView.raw`) nor the DMA ring (its
    copies read the stats slab)."""
    return (
        spec.stats_vec == 1
        and not spec.head_pack
        and spec.loop_sync != "counted"
        and spec.global_path != "dma"
    )


def _ws_bounds_inert(spec, arch) -> bool:
    """The buffer bounds reach the scalar workspace stats loads and the slice
    sink's stores; elsewhere nothing changes."""
    return not (_scalar_ws_stats(spec) or _slice_sink(spec))


def _dkv_cell_vec_inert(spec, arch) -> bool:
    """The per-slot epilogue exists with register accumulators under
    ``dkv_epilogue = "direct"``, in direct mode or with the slice sink."""
    if spec.dkv_epilogue != "direct" or spec.acc_in_lds:
        return True
    return spec.dkv_mode == "atomic" and not _slice_sink(spec)


def _dkv_store_vec_inert(spec, arch) -> bool:
    if spec.dkv_mode != "atomic":
        return False
    if spec.dkv_sink != "slices":
        return True
    return spec.dkv_epilogue == "direct" and not spec.acc_in_lds


def _dq_issue_inert(spec, arch) -> bool:
    """``dq_issue`` emits the block IR: LDS accumulators always issue per
    fragment, and ``"fragment"`` over a single G4 fragment per wave is the
    block order (``"fragment_regs"`` reorders the dS reads; not inert). The
    split-dQ main kernel has no G4 at all."""
    if spec.acc_in_lds or spec.dq_mode == "split":
        return True
    if spec.dq_issue != "fragment":
        return False
    g4 = BwdTileGeometry.from_spec(spec, arch).g4
    return g4.frags_m * g4.frags_n == 1


def _split_dq_inert(spec, arch) -> bool:
    """G4-only knobs (its atom and warp grid, the K^T source) of a main kernel
    without G4 (``dq_mode = "split"``): no code, no LDS buffer, no register
    term depends on them there."""
    return spec.dq_mode == "split"


def _lds_swizzle_inert(spec, arch) -> bool:
    """The ``lds_swizzle`` entries change no buffer of the LDS plan: each names
    a buffer the plan does not have (an XT image without ``xt_lds``), or the
    layout the buffer takes without it (``none`` on a row-major buffer, the
    predictor's choice on an XT image). The kernel is a function of the plan."""
    if not spec.lds_swizzle:
        return True
    a = lds_plan(spec, arch)
    b = lds_plan(dataclasses.replace(spec, lds_swizzle=None), arch)
    fields = ("buffers", "total_bytes", "layouts")
    return all(getattr(a, f) == getattr(b, f) for f in fields)


INERT_KNOBS = MappingProxyType(
    {
        "dkv_epilogue": _atomic_epilogue_inert,
        "dkv_store_vec": _dkv_store_vec_inert,
        "dkv_atomic_guard": lambda spec, arch: (
            spec.dkv_mode == "direct" or spec.dkv_sink != "atomic"
        ),
        "dkv_sink": lambda spec, arch: spec.dkv_mode == "direct",
        # head-packed steps load each row's stats from its own head (scalar)
        # and stage Q / dO per row (no slab)
        "stats_vec": lambda spec, arch: bool(spec.head_pack),
        # ... and the grouped head_outer loops (edge tiles) keep their slabs
        "slab_carry": lambda spec, arch: bool(spec.head_pack)
        or (spec.step_order == "head_outer" and bool(spec.edge_tiles)),
        "ws_bounds": _ws_bounds_inert,
        "dkv_cell_vec": _dkv_cell_vec_inert,
        "lds_order": _lds_order_inert,
        "lds_swizzle": _lds_swizzle_inert,
        "dq_issue": _dq_issue_inert,
        # the slab hoist exists in THD direct mode only
        "dkv_slab_hoist": lambda spec, arch: not (
            spec.dkv_mode == "direct" and spec.seq_mode == "thd"
        ),
        # interior (unclamped) q steps exist with edge tiles only
        "interior_clamp": lambda spec, arch: not spec.edge_tiles,
        # the split-dQ main kernel has no G4 and reads no K^T
        **dict.fromkeys(("atom_g4", "warp_grid_g4", "kt_source"), _split_dq_inert),
        # kK4 sizes the G4 dS key slices; "step" emits G4 one atom-K step at
        # a time over the whole kN0 tile, so kK4 enters no code or estimate
        "block_k4": lambda spec, arch: spec.g4_slicing == DEFAULT_G4_SLICING,
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
    "wmma": ("wmma_lds", "xt_lds"),
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

    An ``asm_tail_nop`` equal to the asm atoms' ``AGPR_ASM_TAIL_NOP`` is the
    default (``None``), and ``asm_cluster = "gemm"`` where every asm GEMM has a
    single K step (G1 / G3: ``block_m`` equal to the atom K; with
    ``s_in_agpr`` also G0 / G2: ``head_size`` equal to their atom K) is the
    ``"k_step"`` cluster: the resolved spec, its kernel name and its IR are
    those of the default.
    """
    from kernels.common._attention_bwd_caps import AGPR_ASM_TAIL_NOP

    if spec.agpr_route != "asm_g13" or spec.atom_g13 is None:
        return spec
    # the asm GEMMs: G1 / G3 (K = kM0), and G0 / G2 (K = D) under s_in_agpr
    asm_ks = [(_atom_k(spec.atom_g13), spec.block_m)]
    if spec.s_in_agpr and spec.atom_g02 is not None:
        asm_ks.append((_atom_k(spec.atom_g02), spec.head_size))
    updates = {}
    shipped = {AGPR_ASM_TAIL_NOP.get(atom_k) for atom_k, _ in asm_ks}
    if spec.asm_tail_nop is not None and shipped == {spec.asm_tail_nop}:
        updates["asm_tail_nop"] = None
    if spec.asm_cluster == "gemm" and all(k == atom_k for atom_k, k in asm_ks):
        updates["asm_cluster"] = "k_step"
    return dataclasses.replace(spec, **updates) if updates else spec


def _canonical_g4_slicing(spec: "AttnBwdSpec") -> "AttnBwdSpec":
    """Spell ``g4_slicing`` of a resolved spec by the code it emits.

    ``"slice_db"`` over a single kK4 slice (``block_k4 == block_n``) is
    ``"slice"`` (nothing to read ahead), and ``"slice"`` with one atom-K step
    per slice (``block_k4`` equal to the G4 atom K) is ``"step"``: the same
    instruction order, so the resolved spec names the kernel it emits.
    ``"slice_db_g3"`` is kept (it reads slice 0 under G3).
    """
    v = spec.g4_slicing
    if v == DEFAULT_G4_SLICING or spec.block_k4 is None or spec.atom_g4 is None:
        return spec
    if v == "slice_db" and spec.block_k4 == spec.block_n:
        v = "slice"
    if v == "slice" and spec.block_k4 == _atom_k(spec.atom_g4):
        v = DEFAULT_G4_SLICING
    return dataclasses.replace(spec, g4_slicing=v) if v != spec.g4_slicing else spec


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
    # asm_g13: read the LDS operands of the G1 / G3 GEMM issued second (G3 in
    # the default stage order) before the first one's asm cluster, so their
    # LDS latency overlaps its MFMAs (an asm block is a scheduling barrier for
    # memory operations). Same MFMAs on the same values; more live registers.
    asm_load_hoist: bool = False
    scheduler_strategy: Optional[str] = None  # codegen scheduler strategy
    # Warp grids (rows, cols) of G0/G2 (S, dP) and G1/G3 (dV, dK). None resolves
    # to (1, W) / (W, 1); (2, W/2), the transposed grids and the replicated
    # G0/G2 grids of the small tiles (_attention_bwd_tiles) need pt_route="lds".
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
    dkv_sink: str = "atomic"  # atomic dK / dV; see DKV_SINKS
    lds_order: str = "resident_low"  # see LDS_ORDERS
    qdo_stage: str = "split"  # see QDO_STAGES
    stats_load: str = "at_use"  # see STATS_LOADS
    # dQ-path levers; each default emits the shipped IR.
    dq_issue: str = "block"  # G4 to the dQ atomics: see DQ_ISSUES
    # G4 dS key slicing in block_k4 (kK4) key slices: see G4_SLICINGS. The
    # default "step" is today's emission (kK4 inert); the others need MFMA
    # register accumulators and dq_issue = "block". Bit-identical partials.
    g4_slicing: str = DEFAULT_G4_SLICING
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
    barrier_merge: str = "off"  # see BARRIER_MERGES (needs ring_depth = 2)
    ring_issue: str = "early"  # see RING_ISSUES (barrier_merge = "on" only)
    # loop_sync = "counted": where the next q step's global loads issue, right
    # after the staging barrier ("early") or right before G4 ("late"); legal
    # with "counted" only. Same values either way.
    counted_issue: str = "early"
    # The CTA decode reads the runtime ``pair`` kernarg: under mirror pairing a
    # CTA runs kv tile t and then kv tile n_kv_tiles - 1 - t (the plan sets it
    # for lb_order = "mirror"; with pair = 0 it is the one-tile decode). Every
    # kv tile's dK / dV are those of the one-tile kernel.
    lb_mirror: bool = False
    # Softmax-gradient VALU levers (kernels.common._attention_bwd_valu):
    # bf16_cast: the P / dS rounding to bf16 ("rne_noguard": round to nearest
    # even without the NaN guard; bf16 only; a NaN input gives an unspecified
    # result); bwd_exp2: "fast" flushes P below 2^-126 to 0; stats_fma: P's
    # argument as one fma (one rounding fewer); stats_vec: lse2 / Dsum loaded
    # as <n x f32> runs of consecutive C rows (head-major workspace, MFMA C
    # maps; same values).
    bf16_cast: str = "generic"  # see BF16_CASTS
    bwd_exp2: str = "exact"  # see BWD_EXP2S
    stats_fma: bool = False
    stats_vec: int = 1  # see STATS_VECS
    # THD: the CTA decode also serves the work list (runtime use_worklist:
    # the sequence and kv tile of a CTA come from the WORKLIST items the prep
    # role writes, heaviest first); False emits no work-list IR. The plan
    # sets it with use_worklist.
    worklist: bool = False
    # Memory-access width / bounds levers (each default emits the shipped IR;
    # every value moves the same bytes, so outputs are bit for bit equal):
    # ws_bounds = "buffer" bounds the main kernel's scalar workspace stats
    # loads and the slice sink's stores by a buffer descriptor instead of
    # address clamps / per-row branches; dkv_cell_vec = 4 transposes each
    # 4 x 4 accumulator block inside a lane quad so the per-slot "direct" dK /
    # dV epilogue stores 4 consecutive columns per lane; narrow_vec
    # (stage_vec = 1 only) is the user-tensor access width in elements when
    # every stride and base allows it; slab_carry advances the Q / dO slab
    # offsets per q step by precomputed i64 deltas instead of recomputing them
    # from the head and row (the synchronous step; not in head-packed steps
    # nor in the grouped head_outer loops). See WS_BOUNDS, DKV_CELL_VECS,
    # NARROW_VECS.
    ws_bounds: str = "clamp"
    dkv_cell_vec: int = 1
    narrow_vec: int = 1
    slab_carry: bool = False
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
        _check_in("dkv_sink", self.dkv_sink, DKV_SINKS)
        _check_in("ws_bounds", self.ws_bounds, WS_BOUNDS)
        _check_in("dkv_cell_vec", self.dkv_cell_vec, DKV_CELL_VECS)
        _check_in("narrow_vec", self.narrow_vec, NARROW_VECS)
        if self.narrow_vec != 1 and self.stage_vec != 1:
            raise ValueError("narrow_vec applies to stage_vec = 1 instances only")
        if self.ws_bounds != "clamp" and self.ws_layout != "head_major":
            raise ValueError(
                "ws_bounds needs the head-major workspace (token-major rows of one "
                "head are H floats apart)"
            )
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
            "waves_per_eu": WAVES_PER_EU_VALUES,
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
        _check_in("g4_slicing", self.g4_slicing, G4_SLICINGS)
        if self.g4_slicing != DEFAULT_G4_SLICING:
            if self.dq_issue != "block":
                raise ValueError(
                    f"g4_slicing={self.g4_slicing!r} needs dq_issue='block' "
                    "(the per-fragment issue reads dS its own way)"
                )
            if self.acc_in_lds:
                raise ValueError(
                    f"g4_slicing={self.g4_slicing!r} needs register accumulators "
                    "(acc_in_lds feeds the dQ sink one fragment at a time)"
                )
            if (
                self.g4_slicing == "slice_db_g3"
                and self.stage_order not in G4_AFTER_G3_ORDERS
            ):
                raise ValueError(
                    f"g4_slicing='slice_db_g3' reads G4's first dS slice under "
                    f"G3: stage_order must be one of {G4_AFTER_G3_ORDERS}"
                )
        for fname in ("q_rotate", "lb_interleave"):
            if not isinstance(getattr(self, fname), bool):
                raise ValueError(
                    f"{fname} must be a bool, got {getattr(self, fname)!r}"
                )
        if not (self.ptr_noalias is False or self.ptr_noalias in PTR_NOALIAS[1:]):
            raise ValueError(f"ptr_noalias={self.ptr_noalias!r} not in {PTR_NOALIAS}")
        if not isinstance(self.xcd_remap, bool):
            raise ValueError(f"xcd_remap must be a bool, got {self.xcd_remap!r}")
        _check_in("counted_issue", self.counted_issue, COUNTED_ISSUES)
        if self.counted_issue != COUNTED_ISSUES[0] and self.loop_sync != "counted":
            raise ValueError("counted_issue applies to loop_sync='counted' only")
        _check_in("bf16_cast", self.bf16_cast, BF16_CASTS)
        _check_in("bwd_exp2", self.bwd_exp2, BWD_EXP2S)
        if isinstance(self.stats_vec, bool):
            raise ValueError(f"stats_vec must be an int, got {self.stats_vec!r}")
        _check_in("stats_vec", self.stats_vec, STATS_VECS)
        if self.bf16_cast != "generic" and self.dtype != "bf16":
            raise ValueError(
                f"bf16_cast={self.bf16_cast!r} rounds to bf16: it needs dtype='bf16'"
            )
        if self.stats_vec != 1 and self.ws_layout != "head_major":
            raise ValueError(
                f"stats_vec={self.stats_vec} needs ws_layout='head_major' (token-major "
                "stats rows are h_q floats apart)"
            )
        for fname in (
            "dkv_slab_hoist",
            "interior_clamp",
            "lb_mirror",
            "stats_fma",
            "worklist",
            "asm_load_hoist",
            "slab_carry",
        ):
            if not isinstance(getattr(self, fname), bool):
                raise ValueError(
                    f"{fname} must be a bool, got {getattr(self, fname)!r}"
                )
        _check_in("barrier_merge", self.barrier_merge, BARRIER_MERGES)
        _check_in("ring_issue", self.ring_issue, RING_ISSUES)
        if self.barrier_merge == "off" and self.ring_issue != RING_ISSUES[0]:
            raise ValueError("ring_issue applies to barrier_merge='on' only")
        if self.barrier_merge == "on" and self.ring_depth not in (None, 2):
            raise ValueError("barrier_merge='on' needs ring_depth=2 (or None)")
        if self.worklist and self.seq_mode != "thd":
            raise ValueError(
                "worklist=True needs seq_mode='thd': a batched CTA past len_kv "
                "still stores the zero rows of its tile, so every grid CTA runs"
            )
        if self.worklist and self.lb_mirror:
            raise ValueError(
                "worklist=True excludes lb_mirror: the work list fixes the CTA "
                "order (natural), mirror pairing reorders it"
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
        if self.global_path == "dma" and self.head_pack:
            raise ValueError(_DMA_HEAD_PACK)
        if self.ring_depth is not None and self.ring_depth > 1:
            if self.loop_sync != "full" and not _merged_ring(self):
                raise ValueError(_RING_LOOP_SYNC)
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
            if self.asm_load_hoist:
                raise ValueError(
                    "asm_load_hoist is legal with agpr_route='asm_g13' only"
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
        if self.barrier_merge == "on" and self.ring_depth is None:
            updates["ring_depth"] = 2  # the merge runs on the 2-slot ring
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
        return _canonical_g4_slicing(_canonical_asm_knobs(out))

    def non_default_fields(self) -> tuple[tuple[str, object], ...]:
        """Compile-time fields that differ from their declared default."""
        return non_default_fields(self, skip=("head_size", "name"))

    def kernel_name(self, stage: str = "main") -> str:
        """Kernel name: base key plus a salt over every non-default knob."""
        _check_in("stage", stage, ("main", "dq"))
        base = (
            f"{self.name}_{stage}_{self.dtype}_d{self.head_size}_"
            f"{self.seq_mode}_{self.dkv_mode}_{self.mask_class}_sv{self.stage_vec}"
        )
        in_base = ("dtype", "seq_mode", "dkv_mode", "stage_vec", "mask_class")
        return salted_kernel_name(
            base, [(k, v) for k, v in self.non_default_fields() if k not in in_base]
        )

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
        the main kernel. The prep spec carries ``worklist`` (the work-list
        role); the convert has no role of it. Imported lazily: the aux module
        is a separate file of the family.
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
            narrow_vec=self.narrow_vec,
            name=self.name,
            worklist=stage == "prep" and self.worklist,
        )

    def dq_spec(self, **knobs):
        """The split dQ kernel's spec matching this main spec.

        The problem fields (head size, dtype, sequence mode, ``stage_vec``,
        mask class, workspace layout, kernel-name stem) come from this spec;
        ``static_scale`` follows its scale placement (the dQ kernel applies
        ``ds_mult`` under ``"fold_ds"``, ``dq_mult`` under ``"convert"`` and
        ``"dq_before_atomic"``, which compile to the same dQ kernel, and both
        under ``"runtime"``); ``knobs`` are the dQ kernel's own tuning fields.
        Imported lazily: the dQ kernel is a separate file of the family.
        """
        from kernels.common.attention_bwd_dq import AttnBwdDqSpec, dq_static_scale

        return AttnBwdDqSpec(
            head_size=self.head_size,
            dtype=self.dtype,
            seq_mode=self.seq_mode,
            stage_vec=self.stage_vec,
            mask_class=self.mask_class,
            ws_layout=self.ws_layout,
            static_scale=dq_static_scale(self.static_scale),
            narrow_vec=self.narrow_vec,
            name=self.name,
            **knobs,
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
        ds_slots=2 if spec.barrier_merge == "on" else 1,
        atom_k_g13=_atom_k(spec.atom_g13),
        atom_k_g4=_atom_k(spec.atom_g4),
    )


def _planned_epilogue(spec: AttnBwdSpec) -> str:
    """The epilogue the LDS plan provides for (see :data:`INERT_KNOBS`).

    Atomic dK / dV use no staging buffer: the pair is planned like the
    default (one reserved buffer), ``"direct"`` reserves none. The slice
    sink stages its fp32 partials (one column half per buffer), so it gets
    the epilogue it names.
    """
    if (
        spec.dkv_mode == "atomic"
        and spec.dkv_sink != "slices"
        and spec.dkv_epilogue == "lds_staged_pair"
    ):
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
    g4 = spec.dq_mode == "atomic"  # split dQ: no G4, no K^T, no dQ partial
    # waves the G0 / G2 K / V operand and the S / dP tile are spread over
    # (W except on the small-tile grids; see _attention_bwd_tiles)
    w_kv, w02 = kv_operand_split(spec.warp_grid_g02, w), grid_waves(spec.warp_grid_g02)
    vgpr = _REG_ALLOWANCE_FIXED
    if spec.kv_residency == "reg":
        vgpr += 2 * (kn * d // (2 * wave * w_kv))
    if spec.kt_source == "reg" and g4:
        vgpr += kn * d // half
    vgpr += 2 * km * d // (2 * wave)
    # The register prefetch of the next tile exists with the ring only; the
    # synchronous stage tables read nothing ahead of the step, so a schedule
    # value leaves the estimate (and the resolved agpr_alloc) as it is.
    prefetch = spec.global_path == "vgpr" and spec.ring_depth >= 2
    if prefetch:
        vgpr += max(spec.ring_depth - 1, 1) * (km * d // (wave * w))
    if getattr(spec, "loop_sync", "full") == "counted":
        # the next step's Q / dO tile and its lse2 / Dsum rows carried in
        # registers across the step (16x16 C fragments: 16 rows per wave
        # hold 16 * 16 / wave slots per lane, each with lse2 and Dsum)
        vgpr += km * d // (wave * w) + 2 * km * 16 // wave
    vgpr += km * kn // (wave * w02)
    if g4:
        vgpr += _ds_slice_regs(spec, wave)
    acc, s_dp, dq = _accumulator_terms(spec, arch)
    in_agpr = _agpr_placement(spec, arch, agpr_form=agpr_form)
    agpr = 0
    for term, name in ((acc, "acc"), (s_dp, "s_dp"), (dq, "dq")):
        if name in in_agpr:
            agpr += term
        else:
            vgpr += term
    return vgpr, agpr


def _ds_slice_regs(spec: AttnBwdSpec, wave: int) -> int:
    """Per-lane registers of the live G4 operands.

    ``"step"``: one atom-K step of dS (``k4a`` keys), which the backend may
    hoist one step ahead: ``2*kM0*k4a / (2*wave)``. Key slicing holds a kK4
    slice (two under the double-buffered values; counted as two for
    ``"slice"`` too, since the backend may hoist): ``2*kM0*kK4 / (2*wave)``,
    plus with K^T from LDS the streamed K^T slices ``2*D*kK4 / (2*wave*g4c)``.
    """
    km = spec.block_m
    if spec.g4_slicing == DEFAULT_G4_SLICING:
        return 2 * km * _atom_k(spec.atom_g4) // (2 * wave)
    kk = spec.block_k4
    regs = 2 * km * kk // (2 * wave)
    if spec.kt_source == "lds":
        regs += 2 * spec.head_size * kk // (2 * wave * spec.warp_grid_g4[1])
    return regs


def _accumulator_terms(spec: AttnBwdSpec, arch: str) -> tuple[int, int, int]:
    """Per-lane fp32 registers of the dK / dV accumulators, of S and dP, and of
    the dQ partial (none under split dQ)."""
    wave, w = attn_bwd_arch_facts(arch).wave_size, spec.waves
    d, km, kn = spec.head_size, spec.block_m, spec.block_n
    acc = 0 if spec.acc_in_lds else 2 * (kn * d // (wave * w))
    w02 = grid_waves(spec.warp_grid_g02)  # a replicated grid: fewer than W
    dq = km * d // (wave * w) if spec.dq_mode == "atomic" else 0
    return acc, 2 * km * kn // (wave * w02), dq


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


def _barrier_merge_on(spec) -> bool:
    """Whether the spec merges the in-loop barriers (``barrier_merge``, a
    field of the merged-barrier ring; absent means off)."""
    return getattr(spec, "barrier_merge", "off") not in ("off", None, False)


def check_counted_sync(spec, arch: str) -> None:
    """Couplings of ``loop_sync = "counted"`` on a resolved spec.

    CDNA only; the register staging path without a ring (a schedule value is
    legal: its stage table models the counted step, the next-step loads keep
    their own fences; ``stage_table_atomics`` needs ``counted_issue =
    "early"``), with the default staging and stats issue order (counted owns
    the issue order);
    at most ``COUNTED_VMCNT_MAX`` dQ atomics per wave and q step (the vmcnt
    field); not with ``barrier_merge``, whose ring issues the next step's
    loads into a second LDS slot ahead of a single barrier before G4 (a
    counted wait there would be ``vmcnt(0)``, the full flavour); not in the
    grouped head-outer form (``step_order = "head_outer"`` with edge tiles),
    whose nested range loops with carried tiles spill registers on the
    production backend on both CDNA targets (the flat head-outer loop is
    legal); not with the split-dQ main kernel (no dQ atomics to keep in
    flight). Raises ``ValueError``.
    """
    if spec.loop_sync != "counted":
        return
    if _barrier_merge_on(spec):
        raise ValueError(
            "loop_sync='counted' excludes barrier_merge (the merged ring owns the "
            "next-step loads and its one barrier precedes the dQ atomics)"
        )
    if attn_bwd_arch_facts(arch).matrix_path != "mfma":
        raise ValueError(
            f"loop_sync='counted' is CDNA-only; {arch} counts no-return atomics in "
            "vscnt, not vmcnt"
        )
    if spec.step_order == "head_outer" and spec.edge_tiles:
        raise ValueError(
            "loop_sync='counted' excludes the grouped head-outer form "
            "(step_order='head_outer' with edge_tiles): its three range loops "
            "nested in the head-group loop, each with a carried next-step tile, "
            "exceed the register file"
        )
    if spec.dq_mode != "atomic":
        raise ValueError(
            "loop_sync='counted' needs dq_mode='atomic': its waits keep the "
            "step's dQ atomics in flight, and the split-dQ main kernel issues "
            "none"
        )
    # ring, stage table, staging order and the vmcnt field (BwdTileGeometry)
    BwdTileGeometry.from_spec(spec, arch)


def validate_attn_bwd_spec(
    spec: AttnBwdSpec, arch: str, *, main_kernel_refusals: bool = True
) -> AttnBwdSpec:
    """Resolve ``spec`` for ``arch`` and apply every static legality rule.

    Returns the resolved spec; raises ``ValueError`` naming the first broken
    rule. Rules: catalog (waves, atoms, transpose read, DMA, ring depth),
    geometry divisibility and couplings, the LDS phase plan within the
    per-arch capacity, and the two register budgets (arch-VGPR and AGPR, their sum
    under ``waves_per_eu``) with the static prune allowance; last, the
    recorded resource-gate failures of this family's main kernel
    (:data:`RESOURCE_REFUSED`). A shell that builds another kernel around the
    same body (the dense family) passes ``main_kernel_refusals=False``: those
    records measure the general main kernel's code object, not its own, which
    its own gate rows cover.
    """
    facts = attn_bwd_arch_facts(arch)
    s = spec.resolved(arch)
    w, km, kn, kk, d = s.waves, s.block_m, s.block_n, s.block_k4, s.head_size
    mfma = facts.matrix_path == "mfma"
    if s.barrier_merge == "on" and arch not in BARRIER_MERGE_ARCHS:
        raise ValueError(f"barrier_merge='on' is built for {BARRIER_MERGE_ARCHS}")
    # catalog
    _check_in(f"waves on {arch}", w, facts.legal_waves)
    for fname in ("atom_g02", "atom_g13", "atom_g4"):
        _check_in(f"{fname} on {arch}", getattr(s, fname), facts.legal_atoms)
    _check_in(
        f"transpose_source on {arch}", s.transpose_source, facts.legal_transpose_sources
    )
    _check_in(f"global_path on {arch}", s.global_path, facts.legal_global_paths)
    _check_in(f"ring_depth on {arch}", s.ring_depth, facts.legal_ring_depths)
    if s.dq_mode == "split" and arch not in SPLIT_DQ_ARCHS:
        raise ValueError(
            f"dq_mode='split' on {arch}: the split dQ kernel is built for "
            f"{SPLIT_DQ_ARCHS} only (its dS^T hand-off is the MFMA C-to-A relabel)"
        )
    # geometry (the natural grids put kN0 / W keys on each wave; the other
    # G0 / G2 and G1 / G3 grids are checked against their tiles below)
    natural = (s.warp_grid_g02, s.warp_grid_g13) == natural_grids(w)
    if (natural or s.sched in STAGE_TABLE_SCHEDS) and kn % (16 * w) != 0:
        raise ValueError(f"block_n={kn} must be a multiple of 16*waves={16 * w}")
    if s.sched in STAGE_TABLE_SCHEDS and grid_waves(s.warp_grid_g02) != w:
        raise ValueError(
            f"sched={s.sched!r}: the stage-table count model needs a G0/G2 grid of "
            f"every wave, not the replicated {s.warp_grid_g02}"
        )
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
    # G0/G2 tile kM0 x kN0, G1/G3 tile kN0 x D: fixed grids unless P^T goes via
    # LDS (see _attention_bwd_tiles: (2, W/2), the transposed grids and the
    # replicated G0/G2 grids of the small tiles)
    for gname, nat, rows, cols in (
        ("warp_grid_g02", (1, w), km, kn),
        ("warp_grid_g13", (w, 1), kn, d),
    ):
        g = getattr(s, gname)
        if g == nat:
            # a natural grid next to a small-tile grid: kN0 is not checked above
            if not natural and (rows % (16 * g[0]) or cols % (16 * g[1])):
                raise ValueError(
                    f"{gname}={g} does not divide its {rows} x {cols} tile"
                )
            continue
        check_lds_route_grid(gname, g, nat, w, rows, cols, s.pt_route)
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
    if s.loop_sync != "full" and s.ring_depth > 1 and not _merged_ring(s):
        raise ValueError(_RING_LOOP_SYNC)
    if s.global_path == "dma" and s.head_pack:
        raise ValueError(_DMA_HEAD_PACK)
    if s.global_path == "dma" and (km * s.head_size * 2) % 1024:
        raise ValueError(
            f"global_path='dma' copies 1 KiB pieces: block_m * head_size * 2 = "
            f"{km * s.head_size * 2} is not a multiple of 1024"
        )
    check_counted_sync(s, arch)
    if s.global_path == "dma" and s.transpose_source == "tr_read" and s.lds_swizzle:
        for item in s.lds_swizzle.split(","):
            buf, _, val = item.partition("=")
            if buf in TR_READ_DMA_BUFFERS and val != "xor":
                raise ValueError(
                    f"lds_swizzle {buf}={val}: DMA buffers read by tr_read need 'xor'"
                )
    if s.dkv_cell_vec != 1:
        from kernels.common._attention_bwd_frag import c_quad_transposable

        if not c_quad_transposable(BwdTileGeometry.from_spec(s, arch).g1.op):
            raise ValueError(
                f"dkv_cell_vec={s.dkv_cell_vec} needs a G1 / G3 C layout whose "
                f"lane quads hold 4 consecutive columns of 4 rows (not on {arch})"
            )
    if s.slab_carry:
        if s.q_rotate:
            raise ValueError(
                "slab_carry excludes q_rotate (a rotated range wraps to its start: "
                "no constant per-step delta)"
            )
        if s.global_path == "dma" or s.ring_depth > 1:
            raise ValueError(
                "slab_carry is built for the synchronous step: the prefetch ring "
                "forms the slabs of the steps ahead from their (q tile, head) "
                "index (not built with a ring)"
            )
        if s.loop_sync == "counted":
            raise ValueError(
                "slab_carry is built for the synchronous step: the counted waits "
                "load the next step from its index (not built with "
                "loop_sync='counted')"
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
    if s.barrier_merge == "on":
        # The hazard proof (body ``run_q_step``) covers exactly this route:
        # register staging into the 2-slot ring, P^T / dS^T by relabel (no P
        # LDS round trip), register dK / dV, G1 and G3 issued before G4.
        if s.ring_depth != 2 or s.global_path != "vgpr":
            raise ValueError(
                "barrier_merge='on' needs ring_depth=2, global_path='vgpr'"
            )
        if s.pt_route != "relabel" or s.acc_in_lds:
            raise ValueError(
                "barrier_merge='on' needs pt_route='relabel' and register dK / dV"
            )
        if s.stage_order not in BARRIER_MERGE_STAGE_ORDERS:
            raise ValueError(
                f"barrier_merge='on' needs G4 last (stage_order in "
                f"{BARRIER_MERGE_STAGE_ORDERS}): G1 / G3 read this step's Q / dO "
                "slot, which the next step overwrites after the merged barrier"
            )
        if s.dq_mode != "atomic":
            raise ValueError(
                "barrier_merge='on' needs dq_mode='atomic': the merged barrier "
                "and the next step's slot stores sit before G4, which the split "
                "dQ main kernel does not issue (its steps would have no barrier)"
            )
    if s.step_order == "head_outer" and s.edge_tiles and s.ring_depth > 1:
        raise ValueError(
            f"ring_depth={s.ring_depth} (the prefetch / DMA ring or the merged "
            "barrier) excludes the grouped head-outer form (step_order="
            "'head_outer' with edge_tiles): its three range loops nested in the "
            "head-group loop, each carrying the ring's next-step tile, spill "
            "SGPRs on the production backend on both CDNA targets (the flat "
            "head-outer loop and edge tiles alone take the ring)"
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
    if s.sched == "stage_table_atomics" and s.dq_mode != "atomic":
        raise ValueError(SPLIT_DEFERRED)
    if s.transpose_source == "xt_lds":
        if staged or s.barrier_merge != "off" or s.loop_sync == "counted":
            raise ValueError(_XT_SYNC_BODY)
        if s.sched in FENCED_SCHEDS:
            raise ValueError(_XT_FENCED_SCHED.format(s.sched))
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
        if s.agpr_route == "asm_g13" and s.asm_tail_nop is not None:
            # the s_nop after each asm cluster must cover the ISA's MFMA result
            # wait states (G1 and G3 share the atom_g13 atom; with s_in_agpr
            # G0 / G2 are asm clusters on the atom_g02 atom too)
            from kernels.common._attention_bwd_caps import check_asm_tail_nop

            check_asm_tail_nop(arch, s.dtype, _atom_k(s.atom_g13), s.asm_tail_nop)
            if s.s_in_agpr:
                check_asm_tail_nop(arch, s.dtype, _atom_k(s.atom_g02), s.asm_tail_nop)
    if spec.g4_slicing != DEFAULT_G4_SLICING or s.g4_slicing != DEFAULT_G4_SLICING:
        # checked on the requested value too: a value the resolved spec spells
        # as "step" (one atom-K step per slice) is still refused where key
        # slicing is not legal
        if not mfma or s.acc_in_lds:
            raise ValueError(
                f"g4_slicing={spec.g4_slicing!r} needs MFMA register "
                "accumulators (acc_in_lds=False); WMMA / LDS-accumulator G4 "
                "feeds the dQ sink one fragment at a time"
            )
        if s.dq_issue != "block":
            raise ValueError(f"g4_slicing={spec.g4_slicing!r} needs dq_issue='block'")
        if (
            staged
            or s.barrier_merge != "off"
            or s.loop_sync == "counted"
            or s.dq_mode != "atomic"
            or s.sched != "none"
            or s.setprio
            or (s.transpose_source == "xt_lds" and s.kt_source == "lds")
        ):
            raise ValueError(_G4_SLICING_SYNC_BODY.format(spec.g4_slicing))
    # LDS: the phase planner decides
    plan = lds_plan(s, arch)
    if not plan.fits:
        raise ValueError(
            f"LDS plan {plan.total_bytes} B exceeds {plan.capacity_bytes} B on {arch}"
        )
    slotted_xor = sorted(
        {buf.name for buf in plan.buffers if buf.slot and buf.layout == "rowmajor:xor"}
    )
    if slotted_xor:
        raise ValueError(_RING_ROW_XOR.format(",".join(slotted_xor)))
    if _layout_spill_key(s, arch) in LAYOUT_SPILLING_COMBOS:
        raise ValueError(
            f"transpose_source={s.transpose_source!r} / lds_swizzle="
            f"{s.lds_swizzle!r} on this {s.block_m} x {s.block_n} x {s.waves}-wave "
            f"d{s.head_size} tile spills arch VGPRs on {arch} "
            "(LAYOUT_SPILLING_COMBOS)"
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
        if (
            wpe is not None
            and wpe > 1
            and vg + ag > (facts.arch_vgprs + facts.agprs) // wpe
        ):
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
    if s.stats_vec != 1:
        from kernels.common._attention_bwd_valu import check_stats_vec_layout

        if not mfma:
            raise ValueError(f"stats_vec={s.stats_vec} is built for MFMA targets only")
        # the lane's C rows must form runs of stats_vec consecutive rows
        check_stats_vec_layout(BwdTileGeometry.from_spec(s, arch).g0.op, s.stats_vec)
    refused = resource_refusal(s, arch) if main_kernel_refusals else None
    if refused is not None:
        raise ValueError(
            f"resource gate: this resolved spec fails on {arch} ({refused}; "
            "RESOURCE_REFUSED)"
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
# The slice reduce (dkv_sink = "slices"): the convert's arguments plus the
# element stride between two slices (i64) and the slice count (i32).
_REDUCE_I64 = (*_CONVERT_I64, "s_stride")
_REDUCE_I32 = (*_CONVERT_I32, "n_slices")
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
    "reduce": (_CONVERT_PTRS, _REDUCE_I64, _CONVERT_F32, _REDUCE_I32),
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
    if spec.agpr_route not in AGPR_ROUTES_BUILT:
        out.append(f"agpr_route={spec.agpr_route!r}")
    if spec.dkv_mode == "atomic" and spec.dkv_sink in DKV_SINK_NOT_BUILT:
        out.append(f"dkv_sink={spec.dkv_sink!r}")
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


def _emit_main(
    spec: AttnBwdSpec,
    arch: str,
    *,
    sentinel: bool = True,
    main_kernel_refusals: bool = True,
):
    """Emit the main kernel of ``spec`` on ``arch``.

    ``sentinel=False`` drops the dead-row term ``lse2 < +inf`` from the dS
    select; it exists only for the negative-control tests and is not a spec
    field. ``main_kernel_refusals=False`` emits a spec whose compiled kernel
    is a recorded resource-gate failure (``RESOURCE_REFUSED``) for tests of
    the emitted IR's structure; every other rule still applies.
    """
    s = validate_attn_bwd_spec(spec, arch, main_kernel_refusals=main_kernel_refusals)
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

    b = IRBuilder(s.kernel_name("main"))
    # Every barrier of the kernel orders LDS accesses across waves (stage-0
    # buffers under the loop buffers, ping-pong tiles, epilogue reuse), so each
    # one carries the compiler memory barrier that keeps LLVM from moving an LDS
    # access across it. The workgroup fences ("workgroup") add nothing to that
    # and cost cache waits / invalidations on the work-group-processor targets.
    b.fence_barriers = BWD_BARRIER_FENCE
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
    cta = _cta_setup(
        b, s, a, geom, plan, io, tid=tid, lane=lane, wave=wave, sentinel=sentinel
    )
    # One kv tile per CTA (natural, reverse or interleave); under lb_mirror a
    # paired CTA (runtime pair = 1, two distinct tiles) runs its second kv tile
    # after the first: the same code on kv tile n_kv_tiles - 1 - t, behind a
    # barrier (pass 1 restages K / V and the LDS phases pass 0 read last).
    stash = None
    if s.lb_mirror and "stash" in _MIRROR_FRESH_ROOTS:
        # pass 1's integer roots parked in VGPR lanes before pass 0's loops
        stash = _MirrorStash(b, a, cta.ids, facts.wave_size)
    _kv_tile(b, s, a, cta.step, cta.mask, kt=cta.work.first_kt, **cta.tile_kw)
    if s.lb_mirror:
        with b.scf_if(b.cmp_eq(cta.work.n_pass, b.const_i32(2))):  # CTA-uniform
            b.sync()
            # Pass 1 recomputes every value from fresh copies of its roots
            # (thread id, block ids, integer kernargs): a value pass 0 formed
            # is not reused (it would stay live across pass 0's q loops). The
            # integer kernargs and block ids come back from the VGPR lanes they
            # were parked in, so their SGPRs need not stay live across pass 0
            # either; the LDS allocation is shared.
            fresh = _MIRROR_FRESH_ROOTS
            a1, ids1 = a, cta.ids
            if stash is not None:
                a1, ids1 = stash.restore(b)
            elif "kernargs" in fresh:
                a1 = {
                    k: (
                        _opaque(b, v, "s")
                        if v is not None and v.type in (I32, I64)
                        else v
                    )
                    for k, v in a.items()
                }  # fmt: skip
            if stash is None and "ids" in fresh:
                ids1 = tuple(_opaque(b, x, "s") for x in cta.ids)
            tid1 = _opaque(b, tid, "v") if "tid" in fresh else tid
            _, lane1, wave1 = lane_and_wave(b, facts.wave_size, tid=tid1)
            cta1 = _cta_setup(
                b, s, a1, geom, plan, io, tid=tid1, lane=lane1, wave=wave1,
                sentinel=sentinel, ids=ids1, lds=cta.lds,
            )  # fmt: skip
            _kv_tile(
                b, s, a1, cta1.step, cta1.mask, kt=cta1.work.second_kt, tag="m1_",
                **cta1.tile_kw,
            )  # fmt: skip
    b.ret()
    return b.kernel


# Roots the second kv tile of a paired CTA reads through a fresh copy, so the
# values it derives from them are its own (see _emit_main): "tid" (an opaque
# VGPR copy), "ids" / "kernargs" (opaque SGPR copies), or with "stash" the
# block ids and integer kernargs from VGPR lanes (_MirrorStash).
_MIRROR_FRESH_ROOTS = frozenset({"tid", "ids", "kernargs", "stash"})


class _MirrorStash:
    """The integer kernargs and block ids of a paired CTA, parked in VGPR lanes
    (``writelane``, one 32-bit word per lane, a 64-bit kernarg in two) before
    the first kv tile and read back (``readlane``) for the second, so the
    second tile's scalar roots occupy no SGPR across the first tile. Each
    stash register passes a volatile empty asm: it is formed where it is
    emitted and the read-back cannot be folded into the original value."""

    def __init__(self, b, a: dict, ids: tuple, wave_size: int) -> None:
        self.a, self.wave, self.n_ids = a, wave_size, len(ids)
        self.keys = [k for k, v in a.items() if v is not None and v.type in (I32, I64)]
        words = []
        for k in self.keys:
            v = a[k]
            if v.type == I32:
                words.append(v)
            else:
                words += [
                    b.trunc(v, I32),
                    b.trunc(b.lshr(v, b.const_i64(32)), I32),
                ]
        words += list(ids)
        self.regs = []
        for c in range(0, len(words), wave_size):
            reg = b.const_i32(0)
            for j, w in enumerate(words[c : c + wave_size]):
                reg = b.writelane(w, b.const_i32(j), reg)
            self.regs.append(
                b.inline_asm(
                    "", "=v,0", [reg], result_type=I32, sideeffect=True,
                    result_name_hint="stash",
                )  # fmt: skip
            )

    def restore(self, b) -> tuple[dict, tuple]:
        """``(kernargs, block ids)`` read back from the stash."""
        it = iter(range(len(self.regs) * self.wave))

        def word():
            i = next(it)
            return b.readlane(self.regs[i // self.wave], b.const_i32(i % self.wave))

        a1 = dict(self.a)
        for k in self.keys:
            if self.a[k].type == I32:
                a1[k] = word()
            else:
                lo, hi = word(), word()
                a1[k] = b.lor(b.zext(lo, I64), b.shl(b.zext(hi, I64), b.const_i64(32)))
        return a1, tuple(word() for _ in range(self.n_ids))


def _opaque(b, v, cls: str):
    """``v`` through an empty asm (register class ``"s"`` or ``"v"``): the
    same value, which the optimizer cannot relate to ``v``."""
    return b.inline_asm(
        "", f"={cls},0", [v], result_type=v.type, sideeffect=False,
        result_name_hint="m1",
    )  # fmt: skip


@dataclass(frozen=True, eq=False)
class _Cta:
    """The decoded work and per-CTA objects of one kv-tile pass."""

    work: object
    step: BwdTileStep
    mask: object
    tile_kw: dict
    ids: tuple
    lds: BwdLds


def _cta_setup(
    b, s, a, geom, plan, io, *, tid, lane, wave, sentinel, ids=None, lds=None
) -> _Cta:
    """The CTA decode, sequences, addressing, mask and tile step from the
    roots ``a`` (kernargs), ``tid`` / ``lane`` / ``wave`` and the block ids
    ``ids`` (``None``: read here); ``lds`` (``None``: allocated here)."""
    D, km, kn = s.head_size, s.block_m, s.block_n
    atomic = s.dkv_mode == "atomic"
    # CTA decode: kv tile fastest, then the head unit (and split), then b; the
    # load-balance order permutes the kv tiles of one (unit, b).
    split_arg = a["g_split"] if atomic else None
    xcd = None
    if s.xcd_remap:
        # grid y = units * g_split: h_k (direct) or h_q / G (atomic) units
        gy = b.mul(b.div(a["h_q"], a["G"]), a["g_split"]) if atomic else a["h_k"]
        xcd = XcdRemap(n=a["xcd_n"], chunk=a["xcd_chunk"], gy=gy)
    if ids is None:
        ids = (b.block_id_x(), b.block_id_y(), b.block_id_z())
    # THD work list: the sequence and kv tile come from the prep role's list
    # when use_worklist is set (no work-list IR without the switch).
    wl = WorkList(ptr=a["WORKLIST"], use=a["use_worklist"]) if s.worklist else None
    work = decode_cta(
        b, x=ids[0], y=ids[1], z=ids[2],
        n_kv_tiles=a["n_kv_tiles"], lb_order=a["lb_order"], g_split=split_arg,
        interleave=s.lb_interleave,
        xcd=xcd,
        **({"pair": a["pair"]} if s.lb_mirror else {}),
        worklist=wl,
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
            narrow_vec=s.narrow_vec,
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

    stats = StatsView(
        ws("WS_LSE2", 1), ws("WS_DSUM", 1),
        n_heads=a["h_q"] if s.stats_vec != 1 else None,
        bounds=s.ws_bounds, span=km,
    )  # fmt: skip
    if s.dq_mode == "split":
        # dQ comes from the Q-parallel dQ kernel: no G4, no dS tile, no WS_DQ
        # access (the kernarg stays in the ABI, unused).
        sink = NullSink()
    else:
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
    if lds is None:
        lds = BwdLds(b, plan, s.dtype)
    step = BwdTileStep(
        b, geom, mask=mask, stats=stats, dq_sink=sink, lds=lds, tid=tid, lane=lane,
        wave=wave, scale_log2=a["scale_log2"], ds_mult=a["ds_mult"],
        sentinel=sentinel,
    )  # fmt: skip
    ring = None
    staged = (s.global_path, s.ring_depth) != ("vgpr", 1)
    if staged and not _merged_ring(s):
        # Q / dO prefetch ring (``_attention_bwd_ring``): the q loops stage
        # through ring slots ahead of the step that reads them. (A merged-
        # barrier 2-slot ring, where built, owns the staging instead; the two
        # are exclusive by value.) The DMA wait counts the step's dQ atomics:
        # none when the main kernel emits no G4 (split dQ). Each kv-tile pass
        # of a paired CTA has its own ring on its own roots.
        n_atomics = km * D // (geom.waves * geom.wave_size)
        ring = QRing(
            b, geom, lds, tid=tid, lane=lane, wave=wave, stats=stats,
            n_atomics=n_atomics if s.dq_mode == "atomic" else 0,
        )  # fmt: skip
        ring.check_step(step.sched)  # the stage placement keeps the DMA waits
    tile_kw = dict(
        hk=hk, hv=hv, h0=h0, n_h=n_h, seq_q=seq_q, seq_kv=seq_kv, q_addr=q_addr,
        do_addr=do_addr, k_addr=k_addr, v_addr=v_addr, io=io, tid=tid, ring=ring,
    )  # fmt: skip
    return _Cta(work=work, step=step, mask=mask, tile_kw=tile_kw, ids=ids, lds=lds)


def _kv_tile(
    b, s, a, step, mask, *, kt, hk, hv, h0, n_h, seq_q, seq_kv, q_addr, do_addr,
    k_addr, v_addr, io, tid, epilogue=None, ring=None, tag="",
) -> None:  # fmt: skip
    """Everything the CTA does for kv tile ``kt``: staging, q loops, epilogue.

    ``k_addr`` / ``v_addr`` only need ``slab(head, k0)`` returning a row
    source with ``load_vec(row, col, n)`` (the paged caches of the unified
    family pass their own). ``epilogue(b, step, acc, k0=, live=)``, when
    given, replaces the strided dK / dV epilogue of ``s.dkv_mode``.

    ``ring`` (a ``QRing``, ``ring_depth > 1`` or ``global_path = "dma"``
    without the barrier merge): every q loop gets a ring prologue, each step
    a :class:`PrefetchStep` (the tile it prefetches: the step ``R - 1`` ahead
    in the loop's own order, or the step itself past the last trip) and the
    loop a drain. ``barrier_merge = "on"`` stages its own 2-slot ring.
    ``loop_sync = "counted"`` carries the next step's loads in registers.

    ``tag`` prefixes the loop and carried-value names (a second kv tile of a
    paired CTA needs names of its own).
    """
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
        # A policy that classified every tile at build time (the dense
        # family's) drops the runs it proved empty; runtime policies keep all.
        static = getattr(mask, "static_runs", None)
        if static is not None:
            ranges = tuple(r for r in ranges if static(r[2]))
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
    # stage_table_atomics: each step's dQ partial is carried to the next step,
    # whose stage A issues its atomics; zeros before the first step, and the
    # last partial is flushed after the loops.
    pend0 = None
    if s.sched == "stage_table_atomics":
        if s.head_pack:
            pend0 = step.zero_pending_dq(b, h0, one, rows_per_head)
        else:
            pend0 = step.zero_pending_dq(b, h0)

    def carried(prefix, acc, qt0, *, grp=True, pend=None):
        vals = acc.flat()
        half = len(vals) // 2
        names = [f"{prefix}dk{i}" for i in range(half)]
        names += [f"{prefix}dv{i}" for i in range(half)]
        extra = [] if qt0 is None else [qt0]
        names += [] if qt0 is None else [f"{prefix}qt"]
        if grp:
            names.append(f"{prefix}grp")
            extra.append(zero)
        if pend is not None:
            names += pend.names(prefix)
            extra += pend.flat()
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

    # slab_carry (the synchronous step; the validator refuses it with a ring or
    # the counted waits, whose steps form the slabs of the steps ahead from
    # their index): the Q / dO slab offsets are loop-carried i64 values
    # advanced by per-CTA deltas (one head, one q tile of kM0 rows), so a step
    # adds one i64 instead of forming head * s_h + q0 * s_t (two i64
    # multiplies per tensor). Every visited q tile has q0 <= len_q - 1, so the
    # clamp of the slab's base row is an identity and the addresses are the
    # default's. Not in head-packed steps (no slab) nor in the grouped
    # head_outer loops: their head is loop-invariant inside the range loops
    # already (one row product per step is left), and the carried offsets and
    # deltas there cost SGPRs the edge loops do not have.
    carry = s.slab_carry and not s.head_pack and not grouped
    if carry:
        head_d = [x.stride_bytes("h") for x in (q_addr, do_addr)]
        row_d = [b.mul(b.const_i64(km), x.stride_bytes("t")) for x in (q_addr, do_addr)]

    def q_steps(
        acc, qt0, trip, *, edge, name, end, wrap_at=None, group=None, pend=None
    ):
        """``trip`` steps from q tile ``qt0``; returns ``(acc, pend)``.

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

        ``pend`` (``stage_table_atomics``): the deferred dQ partial entering
        the loop; it is carried from step to step and the one leaving the
        loop is returned.
        """
        start0 = qt0
        if s.q_rotate:
            qt0 = b.add(start0, b.mod(kt, b.smax(b.sub(end, start0), one)))

        def advance(qt, gi, *, with_wrap=False):
            """The (q tile, head group) index after ``(qt, gi)``; with
            ``with_wrap`` also the predicate of the inner index wrapping
            (``(index, wrap)``, ungrouped loops only)."""
            if group is not None:
                qt_next = b.add(qt, one)
                if s.q_rotate:
                    qt_next = b.select(b.cmp_ge(qt_next, end), start0, qt_next)
                return (qt_next,)
            if wrap_at is None:
                # next (q tile, head group): head groups inner
                nxt = b.add(gi, one)
                wrap = b.cmp_ge(nxt, n_grp)
                qt_next = b.add(qt, b.select(wrap, one, zero))
                if s.q_rotate:
                    qt_next = b.select(b.cmp_ge(qt_next, end), start0, qt_next)
                gi_next = b.select(wrap, zero, nxt)
                return ((qt_next, gi_next), wrap) if with_wrap else (qt_next, gi_next)
            # next (q tile, head group): q tiles inner, wrapping at hi
            lo, hi = wrap_at
            nxt = b.add(qt, one)
            wrap = b.cmp_ge(nxt, hi)
            qt_next = b.select(wrap, lo, nxt)
            if s.q_rotate:
                # the group is done when its q tiles are back at the start
                wrap = b.cmp_eq(qt_next, qt0)
            idx = (qt_next, b.add(gi, b.select(wrap, one, zero)))
            return (idx, wrap) if with_wrap else idx

        def inputs(qt, gi, *, slabs=True):
            """Where step (q tile ``qt``, head group ``gi``) reads; ``slabs =
            False`` forms no Q / dO slab (a counted step stores staged
            values)."""
            q0 = b.mul(qt, b.const_i32(km))
            h, first = head_of(gi) if group is None else group[1:]
            if s.head_pack:
                rows = PackedRows(
                    head0=h, n_heads=b.smin(per, b.sub(n_h, first)),
                    rows_per_head=rows_per_head, row0=q0,
                )  # fmt: skip
                packed = PackedStep(rows=rows, q_addr=q_addr, do_addr=do_addr)
                return QStepInputs(q0=q0, h=h, q_slab=None, do_slab=None, packed=packed)
            if not slabs:
                return QStepInputs(q0=q0, h=h, q_slab=None, do_slab=None, packed=None)
            # interior steps: every row is < len_q, so no row clamp (unless
            # interior_clamp keeps it)
            clamp = edge or s.interior_clamp
            return QStepInputs(
                q0=q0, h=h, q_slab=q_addr.slab(h, q0, clamp=clamp),
                do_slab=do_addr.slab(h, q0, clamp=clamp), packed=None,
            )  # fmt: skip

        def ring_tile(qt, gi):
            t = inputs(qt, gi)
            return PrefetchTile(t.q0, t.h, t.q_slab, t.do_slab, t.packed)

        def ahead_of(qt, gi, k):
            """The index ``k`` steps after ``(qt, gi)`` (k >= 1)."""
            idx = (qt, gi)
            for _ in range(k):
                nxt = advance(*idx)
                idx = (nxt[0], gi if group is not None else nxt[1])
            return idx

        def step_loads(qt, gi):
            """``loop_sync = "counted"``: every global load of the q step at
            ``(qt, gi)`` into registers (:meth:`BwdTileStep.counted_loads`)."""
            t = inputs(qt, gi)
            return step.counted_loads(
                b, h=t.h, q0=t.q0, q_slab=t.q_slab, do_slab=t.do_slab,
                packed=t.packed, edge=edge,
            )  # fmt: skip

        gi0 = zero if group is None else group[0]
        merge = s.barrier_merge == "on"
        if merge:
            # ring prologue: the loop's first step into slot 0, then barrier P
            with b.scf_if(b.cmp_lt(zero, trip)):
                step.ring_prologue(b, inputs(qt0, gi0))
            step.ring_publish(b)
        carried_regs = []
        if ring is not None:
            tiles = [ring_tile(qt0, gi0)]
            if ring.depth == 3:
                # the second step of the loop, or the first again (one trip)
                aq, ag = ahead_of(qt0, gi0, 1)
                two = b.cmp_lt(one, trip)
                ag = ag if group is not None else b.select(two, ag, gi0)
                tiles.append(ring_tile(b.select(two, aq, qt0), ag))
            if ring.path == "dma" and ring.depth == 1:
                tiles = []
            carried_regs = ring.prologue(b, trip, tiles, name=f"bwd_{name}")
        n_idx = 1 if group is not None else 2
        iter_args = carried(f"bwd_{name}", acc, qt0, grp=group is None)
        iter_args += [(f"bwd_{name}r{i}", v) for i, v in enumerate(carried_regs)]
        if counted:
            # the first step's loads, guarded so an empty range loads nothing
            guard = b.scf_for_iter(
                zero, b.select(b.cmp_gt(trip, zero), one, zero), one,
                [(f"bwd_{name}pz{i}", z) for i, z in enumerate(step.counted_zero(b))],
                iv_name=f"bwd_{name}pzit",
            )  # fmt: skip
            with guard as (_g, _gcur):
                b.scf_yield(*step_loads(qt0, gi0).flat())
            # The first step's loads retire here, before the loop: the step
            # top consumes the carried values on every trip, and a load still
            # pending at the loop entry would make the backend's wait at the
            # top a vmcnt(0) on every trip (draining the previous step's dQ
            # atomics); inside the loop the counted barriers have retired the
            # carried loads before the back-edge.
            b.s_waitcnt(vmcnt=0)
            iter_args += [(f"bwd_{name}pf{i}", v) for i, v in enumerate(guard.results)]
        n_carry = 0
        if carry:
            # the first step's Q / dO slab offsets, then the per-CTA deltas
            # of a step: a head (heads inner), a q tile (q tiles inner) or, at
            # a wrap, the jump back
            h_first = head_of(gi0)[0]
            row_first = b.mul(qt0, b.const_i32(km))
            iter_args += [
                (f"bwd_{name}o{t}", addr.offset_bytes(h_first, row_first))
                for t, addr in (("q", q_addr), ("d", do_addr))
            ]
            n_carry = 2
            if wrap_at is None:
                back = b.sext(b.sub(n_grp, one), I64)
                wrap_d = [b.sub(row_d[t], b.mul(back, head_d[t])) for t in (0, 1)]
                step_d = head_d
            else:
                lo, hi = wrap_at
                back = b.sext(b.sub(b.sub(hi, lo), one), I64)
                wrap_d = [b.sub(head_d[t], b.mul(back, row_d[t])) for t in (0, 1)]
                step_d = row_d
        n_pend = 0 if pend is None else len(pend.flat())
        if pend is not None:
            iter_args += list(zip(pend.names(f"bwd_{name}"), pend.flat()))
        loop = b.scf_for_iter(
            zero, trip, one, iter_args,
            iv_name=f"bwd_{name}it", elide_trailing_barrier=False,
        )  # fmt: skip
        with loop as (_it, cur):
            qt = cur[n_acc]
            gi = group[0] if group is not None else cur[n_acc + 1]
            carry_yields = ()
            if carry:
                nxt_idx, wrap = advance(qt, gi, with_wrap=True)
            else:
                nxt_idx = advance(qt, gi)
            here = inputs(qt, gi, slabs=not (counted or carry))
            h = here.h
            # the deferred dQ partial is last, the carried slab offsets before it
            body_end = len(cur) - n_pend - n_carry
            if carry:
                offs = cur[body_end : body_end + n_carry]
                clamp = edge or s.interior_clamp
                here = QStepInputs(
                    q0=here.q0, h=h, q_slab=q_addr.slab_at(offs[0], here.q0, clamp=clamp),
                    do_slab=do_addr.slab_at(offs[1], here.q0, clamp=clamp), packed=None,
                )  # fmt: skip
                carry_yields = tuple(
                    b.add(o, b.select(wrap, wrap_d[t], step_d[t]))
                    for t, o in enumerate(offs)
                )
            if group is None and pend is None:
                # (the deferred sink rebases at each flush: its head is carried)
                step.sink.begin_head(b, h)
            if counted:
                staged = StagedStep.from_flat(
                    cur[n_acc + n_idx : body_end], step.counted_n_stats
                )

                def next_loads(it=_it, qt=qt, gi=gi, nxt_idx=nxt_idx):
                    # the next step's loads; the last trip reloads its own
                    last = b.cmp_ge(b.add(it, one), trip)
                    nq = b.select(last, qt, nxt_idx[0])
                    ngi = gi if group is not None else b.select(last, gi, nxt_idx[1])
                    return step_loads(nq, ngi)

                out, nxt_staged = step.run_q_step(
                    b, acc_of(cur), None, None, qt, h, k0, edge=edge,
                    packed=here.packed, staged=staged, next_loads=next_loads,
                    pending=None if pend is None else pend.from_flat(cur[body_end + n_carry :]),
                )  # fmt: skip
                pend_yields = ()
                if pend is not None:  # stage_table_atomics: (staged, pending)
                    nxt_staged, nxt_pend = nxt_staged
                    pend_yields = nxt_pend.flat()
                b.scf_yield(*out.flat(), *nxt_idx, *nxt_staged.flat(), *pend_yields)
            else:
                merged = None
                if merge:
                    # The next step of this loop (the step itself on the
                    # last trip, so every address stays in range) and the
                    # ring slot.
                    more = b.cmp_lt(b.add(_it, one), trip)
                    nqt = b.select(more, nxt_idx[0], qt)
                    ngi = gi if group is not None else b.select(more, nxt_idx[1], gi)
                    slot = b.mod(_it, b.const_i32(2))
                    merged = RingStep(slot=slot, nxt=inputs(nqt, ngi))
                pf = None
                if ring is not None:
                    tile = PrefetchTile(
                        here.q0, h, here.q_slab, here.do_slab, here.packed
                    )
                    ahead = tile
                    if ring.schedule["ahead"]:
                        k = ring.schedule["ahead"]
                        aq, ag = ahead_of(qt, gi, k)
                        more = b.cmp_lt(b.add(_it, b.const_i32(k)), trip)
                        ag = ag if group is not None else b.select(more, ag, gi)
                        ahead = ring_tile(b.select(more, aq, qt), ag)
                    pf = PrefetchStep(
                        ring, it=_it, here=tile, ahead=ahead,
                        carried=list(cur[n_acc + n_idx : body_end]),
                    )  # fmt: skip
                out = step.run_q_step(
                    b, acc_of(cur), here.q_slab, here.do_slab, qt, h, k0, edge=edge,
                    packed=here.packed, ring=merged, prefetch=pf,
                    pending=None if pend is None else pend.from_flat(cur[body_end + n_carry :]),
                )  # fmt: skip
                pf_yields = pf.yields if pf is not None else ()
                pend_yields = ()
                if pend is not None:
                    out, nxt_pend = out
                    pend_yields = nxt_pend.flat()
                b.scf_yield(
                    *out.flat(), *nxt_idx, *pf_yields, *carry_yields, *pend_yields
                )
        if ring is not None:
            ring.drain(b)
        results = list(loop.results)
        res = results[:n_acc]
        if pend is not None:
            pend = pend.from_flat(results[len(results) - n_pend :])
        return (DkDvAcc.from_flat(res, fm, fn) if n_acc else acc0), pend

    counted = s.loop_sync == "counted"

    acc, pend = acc0, pend0
    names = tuple(f"{tag}{n}" for n in ("q", "qi_", "qe_"))
    if not grouped:
        for (s0, e, edge), trip, name in zip(ranges, trips, names):
            wrap_at = (s0, e) if head_outer else None
            acc, pend = q_steps(
                acc, s0, trip, edge=edge, name=name, end=e, wrap_at=wrap_at,
                pend=pend,
            )  # fmt: skip
    else:
        outer = b.scf_for_iter(
            zero, n_outer, one, carried(f"bwd_{tag}g", acc0, None, pend=pend0),
            iv_name=f"bwd_{tag}git", elide_trailing_barrier=False,
        )  # fmt: skip
        with outer as (_it, cur):
            gi = cur[n_acc]
            h, first = head_of(gi)
            if pend0 is None:
                step.sink.begin_head(b, h)  # dominates the range loops' steps
                ipend = None
            else:
                ipend = pend0.from_flat(cur[len(cur) - len(pend0.flat()) :])
            inner = acc_of(cur)
            for (s0, e, edge), trip, name in zip(ranges, trips, names):
                inner, ipend = q_steps(
                    inner, s0, trip, edge=edge, name=name, end=e,
                    group=(gi, h, first), pend=ipend,
                )  # fmt: skip
            tail = [] if ipend is None else ipend.flat()
            b.scf_yield(*inner.flat(), b.add(gi, one), *tail)
        results = list(outer.results)
        res = results[:n_acc]
        acc = DkDvAcc.from_flat(res, fm, fn) if n_acc else acc0
        if pend0 is not None:
            pend = pend0.from_flat(results[len(results) - len(pend0.flat()) :])
    if pend is not None:
        step.flush_dq(b, pend)  # the last step's dQ atomics
    acc = step.finish(b, acc)
    if epilogue is not None:
        epilogue(b, step, acc, k0=k0, live=live)
    elif s.dkv_mode == "atomic" and s.dkv_sink == "slices":
        _slices_epilogue(
            b, s, a, step, acc, hk=hk, hv=hv, h0=h0, n_h=n_h, k0=k0, seq_kv=seq_kv,
            live=live, tid=tid,
        )  # fmt: skip
    elif s.dkv_mode == "atomic":
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


def _slice_slab(b, s, a, which, *, head, h0, n_h, k0, seq_kv):
    """Workspace slab of this CTA's fp32 ``which`` (``"dk"`` / ``"dv"``) slice
    of kv head ``head`` from tile row ``k0``.

    ``WS_DK`` holds ``gk / n_h`` slices of ``ws_rows_kv * h_k * D`` floats
    (``WS_DV``: ``gv / n_h`` of ``ws_rows_kv * h_v * D``), each laid out like
    the atomic workspace. The CTAs of one K head own its ``gk`` query heads
    ``n_h`` at a time, so this CTA's slice is ``(h0 mod gk) / n_h``: every
    slice row of a live tile is written by exactly one CTA.
    """
    D = s.head_size
    ptr, n_heads, g = {
        "dk": ("WS_DK", a["h_k"], a["gk"]),
        "dv": ("WS_DV", a["h_v"], a["gv"]),
    }[which]
    sl = b.div(b.mod(h0, g), n_h)
    elems = b.mul(
        b.mul(b.zext(a["ws_rows_kv"], I64), b.zext(n_heads, I64)), b.const_i64(D)
    )
    base = b.global_ptr_add(
        a[ptr], b.mul(b.mul(b.zext(sl, I64), elems), b.const_i64(4))
    )
    tok_major = s.ws_layout == "token_major"
    return WorkspaceAddr(
        b, base, ws_rows=a["ws_rows_kv"], width=D, row0=seq_kv.ws_row0,
        layout=s.ws_layout, n_heads=n_heads if tok_major else None,
    ).slab(head, k0)  # fmt: skip


def _slices_epilogue(
    b, s, a, step, acc, *, hk, hv, h0, n_h, k0, seq_kv, live, tid
) -> None:
    """fp32 dK / dV of the key tile stored into this CTA's workspace slices.

    Inside ``scf_if(live)`` (a tile past ``len_kv`` stores nothing; its rows
    are never read as values). No atomics: each slice row is written by one
    CTA, with plain stores of rows ``k < len_kv`` inside the workspace (the
    rows the atomic epilogue adds), dK times ``dk_mult`` as there; the
    reduce kernel sums the slices in a fixed order. ``dkv_epilogue`` picks the
    path: ``"lds_staged"`` stages each column half of the fp32 tile through
    the ``dkv_out`` buffer (two halves per tensor, each behind a barrier) and
    stores rows of ``dkv_store_vec`` floats; ``"lds_staged_pair"`` stages both
    halves in two buffers behind one barrier; ``"direct"`` stores every
    accumulator element from its register (no LDS), or with ``dkv_cell_vec =
    4`` one 16-byte row piece per fragment after an in-quad transpose. LDS
    accumulators are read from their fp32 images after one barrier, whatever
    the epilogue. ``ws_bounds = "buffer"`` stores through a buffer descriptor
    that ends at the tile's last valid row instead of per-row branches.
    """
    D = s.head_size
    hw = D // 2
    outputs = (("dk", hk, a["dk_mult"]), ("dv", hv, None))

    def target(which, head):
        # formed right before the tensor's stores: one slab (and descriptor)
        # live at a time
        slab = _slice_slab(
            b, s, a, which, head=head, h0=h0, n_h=n_h, k0=k0, seq_kv=seq_kv
        )
        return _SliceTarget(b, s, slab, k0=k0, seq_kv=seq_kv)

    with b.scf_if(live):
        if s.acc_in_lds:
            b.sync()  # the images' cells were last updated by other lanes
            for which, head, mult in outputs:
                view = step.dkv_f32_image(which)
                _store_slice_rows(
                    b, s, step, target(which, head), view, cols=D, col0=0,
                    mult=mult, tid=tid,
                )  # fmt: skip
            return
        if s.dkv_epilogue == "direct":
            for which, head, mult in outputs:
                dst = target(which, head)
                if s.dkv_cell_vec == 4:
                    for row, col0, vals in step.dkv_quad_rows(b, which, acc, mult):
                        dst.store_vec(row, col0, b.vec_pack(vals, F32))
                    continue
                for row, items in step.dkv_row_slots(b, which, acc):
                    if mult is not None:
                        items = [(c, b.fmul(v, mult)) for c, v in items]
                    dst.store_row(row, items)
            return
        pair = s.dkv_epilogue == "lds_staged_pair"
        for which, head, mult in outputs:
            dst = target(which, head)
            if pair:
                views = [
                    step.dkv_f32_part(b, which, acc, mult, half=h, slot=h)
                    for h in (0, 1)
                ]
                b.sync()
                for h, view in enumerate(views):
                    _store_slice_rows(
                        b, s, step, dst, view, cols=hw, col0=h * hw, tid=tid
                    )
                b.sync()  # reads done before the buffers are written again
                continue
            for h in (0, 1):
                view = step.dkv_f32_part(b, which, acc, mult, half=h)
                b.sync()
                _store_slice_rows(b, s, step, dst, view, cols=hw, col0=h * hw, tid=tid)
                b.sync()  # reads done before the buffer is written again


class _SliceTarget:
    """One slice slab of the tile and its bounds form.

    ``ws_bounds = "clamp"``: a store is predicated on its row being inside the
    workspace and the sequence (one branch per row or chunk). ``"buffer"``:
    every store goes through a buffer descriptor whose range ends at the
    tile's last valid row, ``min(len_kv - k0, rows_left, kN0)`` rows of ``D``
    floats (head-major rows are ``D`` floats apart; the validator requires the
    head-major workspace), so the stores of rows past it are dropped by the
    hardware and no branch is emitted. Either way exactly the rows ``k <
    len_kv`` inside the workspace are written, with the same values.
    """

    def __init__(self, b, s, slab, *, k0, seq_kv) -> None:
        self.b, self.slab = b, slab
        self.k0, self.length = k0, seq_kv.length
        self.buffer = s.ws_bounds == "buffer"
        self.D = s.head_size
        if self.buffer:
            if slab.row_stride != self.D:
                raise ValueError("buffer-bounded slices need the head-major workspace")
            n = b.smin(b.sub(seq_kv.length, k0), slab.rows_left)
            n = b.smax(b.smin(n, b.const_i32(s.block_n)), b.const_i32(0))
            self.rsrc = b.buffer_rsrc(slab.ptr, b.mul(n, b.const_i32(4 * self.D)))

    def ok(self, r):
        b = self.b
        return b.land(self.slab.row_ok(r), b.cmp_lt(b.add(self.k0, r), self.length))

    def _offset(self, r, c):
        """Byte offset of ``(r, c)`` from the slab base (head-major rows)."""
        b = self.b
        idx = b.mul(r, b.const_i32(self.D))
        if isinstance(c, int):
            if c:
                idx = b.add(idx, b.const_i32(c))
        else:
            idx = b.add(idx, c)
        return b.mul(idx, b.const_i32(4))

    def _buffer_store(self, r, c, v) -> None:
        b = self.b
        n = v.type.count
        zero = b.const_i32(0)
        off = self._offset(r, c)
        if n == 1:
            b.buffer_store_f32(self.rsrc, off, zero, b.vec_extract(v, 0))
            return
        for k in range(0, n, 4):
            part = v
            if n > 4:
                part = b.vec_pack([b.vec_extract(v, k + i) for i in range(4)], F32)
            o = off if k == 0 else b.add(off, b.const_i32(4 * k))
            b.buffer_store_vN_f32(self.rsrc, o, zero, part, min(n, 4))

    def store_vec(self, r, c, v) -> None:
        """``<n x f32>`` at row ``r``, column ``c`` of the slice tile."""
        b = self.b
        if self.buffer:
            self._buffer_store(r, c, v)
            return
        with b.scf_if(self.ok(r)):
            self.slab.store_vec_in_row(r, c, v)

    def store_row(self, r, items) -> None:
        """fp32 ``(column, value)`` pairs of row ``r``."""
        b = self.b
        if not self.buffer:
            self.slab.store_row(r, items, b.cmp_lt(b.add(self.k0, r), self.length))
            return
        zero = b.const_i32(0)
        for c, v in items:
            b.buffer_store_f32(self.rsrc, self._offset(r, c), zero, v)


def _store_slice_rows(b, s, step, dst, view, *, cols, col0, tid, mult=None) -> None:
    """The ``kN0 x cols`` fp32 ``view`` to slice columns ``col0 ..`` as rows
    of ``dkv_store_vec`` floats, chunks dealt to the CTA's threads and stored
    by ``dst`` (a :class:`_SliceTarget`) where the row is inside the sequence
    and the workspace. ``mult`` (when given) multiplies each value after the
    read. A chunk index past the tile (the last round of a tile that the
    threads do not divide) is skipped in both bounds forms."""
    kn, vec = s.block_n, s.dkv_store_vec
    if cols % vec:
        raise ValueError(f"dkv_store_vec={vec} does not divide {cols} columns")
    threads = step.geom.threads
    per_row = cols // vec
    total = kn * per_row
    for chunk, tail in deal_chunks(b, tid, total, threads):
        r, c = chunk_coords(b, chunk, per_row, vec)
        cc = c if col0 == 0 else b.add(c, b.const_i32(col0))

        def read(r=r, c=c):
            v = view.load(b, r, c, dtype=F32, n=vec)
            if mult is not None:
                v = b.vec_pack(
                    [b.fmul(mult, b.vec_extract(v, i)) for i in range(vec)], F32
                )
            return v

        if dst.buffer:
            if tail:
                with b.scf_if(b.cmp_lt(chunk, b.const_i32(total))):
                    dst.store_vec(r, cc, read())
            else:
                dst.store_vec(r, cc, read())
            continue
        ok = dst.ok(r)
        if tail:
            ok = b.land(ok, b.cmp_lt(chunk, b.const_i32(total)))
        with b.scf_if(ok):
            dst.slab.store_vec_in_row(r, cc, read())


def _dkv_slab(b, s, a, which, *, hk, k0, seq_kv, io):
    """The ``which`` (``"dk"`` / ``"dv"``) output slab of kv head ``hk`` from
    row ``k0`` (rows unclamped)."""
    return StridedTensorAddr(
        b, a[{"dk": "dK", "dv": "dV"}[which]], dtype=io, seq=seq_kv,
        stage_vec=s.stage_vec,
        strides=(a[f"{which}_b"], a[f"{which}_h"], a[f"{which}_t"]),
        narrow_vec=s.narrow_vec,
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
            if s.dkv_cell_vec == 4:
                # an in-quad transpose first: 4 consecutive columns per store
                quads = step.dkv_quad_rows(b, which, acc, mult)
                _store_dkv_quads(b, s, a, slab, quads, k0=k0, seq_kv=seq_kv, io=io)
                continue
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
    for chunk, tail in deal_chunks(b, tid, total, threads):
        r, c = chunk_coords(b, chunk, per_row, vec)
        k = b.add(k0, r)
        store, row_live = dkv_store_rule(
            b, k, seq_kv.length, seq_mode=s.seq_mode, s_kv_max=a["S_kv_max"]
        )
        if tail:
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


def _store_dkv_quads(b, s, a, slab, quads, *, k0, seq_kv, io) -> None:
    """Each lane's ``(row, col0, [4 fp32])`` pieces (:meth:`BwdTileStep.
    dkv_quad_rows`) rounded once to the I/O dtype and stored as 4 consecutive
    elements, predicated per row as :func:`_store_dkv_cells` (the same values
    to the same elements; one store per fragment instead of four)."""
    zero = b.zero_vec(io, 4)
    for row, col0, vals in quads:
        k = b.add(k0, row)
        store, row_live = dkv_store_rule(
            b, k, seq_kv.length, seq_mode=s.seq_mode, s_kv_max=a["S_kv_max"]
        )
        with b.scf_if(store):
            v = b.vec_cast_f32_to(b.vec_pack(vals, F32), io)
            slab.store_vec(k, col0, b.select(row_live, v, zero))


def build_attn_bwd_main(spec: AttnBwdSpec, *, arch: str):
    """Build the layout-general backward main kernel (``rocke.attn_bwd.v3``).

    The spec is resolved and validated for ``arch`` first (an illegal
    geometry raises ``ValueError``); knob values whose emission is not built
    yet raise ``NotImplementedError`` before any IR is emitted.
    """
    return _emit_main(spec, arch)
