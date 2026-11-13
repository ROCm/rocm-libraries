# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Split dQ kernel of the layout-general attention backward (``dq_mode = "split"``).

Under ``dq_mode = "split"`` the main kernel computes dK / dV only (a
``NullSink``: no G4, no dS tile, no dQ atomics, no dQ workspace, no dQ zero
fill in prep and no dQ convert) and this Q-parallel kernel produces dQ: one
CTA owns ``kM0`` query rows, keeps Q and dO in registers, recomputes ``S^T =
K Q^T`` and ``dP^T = V dO^T`` for every key tile its band reaches, forms dS
and accumulates ``dQ = dS K`` in fp32 registers, then stores dQ once in the
output dtype (no workspace, no atomics: deterministic). The body is
:mod:`kernels.common._attention_bwd_dqbody`.

* :class:`AttnBwdDqSpec`: compile-time spec; only ``head_size`` is required.
  The problem fields (dtype, sequence mode, ``stage_vec``, mask class,
  workspace layout, scale placement, name stem) follow the main spec
  (:meth:`AttnBwdSpec.dq_spec`); every tuning field is defaulted, ``None``
  ones resolve through :func:`attn_bwd_dq_default_geometry`.
* :func:`validate_attn_bwd_dq_spec`: value sets, geometry, the relabel and
  transpose-read proofs, the LDS plan and the register budgets.
* :func:`build_attn_bwd_dq` (``(spec, *, arch)``): the kernel, kernargs of
  ``attn_bwd_params("dq")`` of the ``rocke.attn_bwd.v3`` ABI.

Grid ``(ceil(S_q_max / R), h_q / H, B)`` with ``R`` rows per head: ``head_loop
= "single"`` (``H = 1``, ``R = kM0``) gives each CTA one query head;
``"group"`` packs the ``H = G`` query heads that share one K head and one V
head (the runtime ``G`` kernarg: ``gcd(h_q / h_k, h_q / h_v)``; the plan
requires ``G | kM0``) onto the ``kM0`` rows, ``R = kM0 / G`` rows each, so a
key tile staged once serves the whole group. With ``xcd_remap`` the CTA id
goes through the chunked XCD transform (runtime ``xcd_n`` / ``xcd_chunk``,
identity at ``xcd_n = 0``) per batch plane.

Scale placement (``static_scale``): ``"fold_ds"`` applies ``ds_mult`` (the
scale) to dS and not ``dq_mult``; ``"convert"`` (the main kernel's
``"convert"`` and ``"dq_before_atomic"``: there is no dQ convert, the plan
passes ``dq_mult = scale``) applies ``dq_mult`` to dQ and not ``ds_mult``;
``"runtime"`` applies both. A multiplier that is not applied is exactly 1.0.

``stats_prescale`` (default on: the kernel has no legacy IR): ``lse2`` comes
from prep already in base 2 (``lse * log2(e)``); ``-lse2`` and ``-ds_mult *
Dsum`` are formed once per CTA, so a cell costs ``exp2(fma(S, scale_log2,
-lse2))`` and ``P * fma(dP, ds_mult, -ds_mult * Dsum)`` instead of
``exp2(S * scale_log2 - lse2)`` and ``ds_mult * (P * (dP - Dsum))``
(another rounding of the same values; ``False`` keeps the main kernel's
form).

Register model (wave64, per lane, 4-byte registers, two 16-bit operands per
register), :func:`dq_register_estimates`, with ``C = kM0 / W`` query columns
per wave: resident Q^T and dO^T ``2 * C * D / (2 * wave)``; the dS^T A
operand ``kN0 * C / (2 * wave)``; one K step of the G0' A fragments ``kN0 *
k02 / (2 * wave)`` and of the G4' B fragments ``D * k4 / (2 * wave)``; the
per-column stats ``6 * C / 16``; with ``kv_prefetch`` the next K and V tile
``2 * kN0 * D / (2 * wave * W)``; an allowance of 24; the fp32 accumulators
``S^T`` and ``dP^T`` ``2 * kN0 * C / wave`` and ``dQ`` ``C * D / wave`` (AGPRs
unless ``agpr_alloc = (0, 0)``: the backend then selects the AGPR form for
every MFMA).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, fields
from types import MappingProxyType
from typing import Optional

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
)
from kernels.common._attention_bwd_band import NoMaskKTail, RuntimeBand
from kernels.common._attention_bwd_body import StatsView
from kernels.common._attention_bwd_dqbody import (
    DQ_TRANSPOSE_READERS,
    BwdDqGeometry,
    BwdDqStep,
    DqLdsViews,
    interior_kv_range,
    kv_tile_range,
)
from kernels.common._attention_bwd_frag import LdsView, ir_type
from kernels.common._attention_bwd_gemm import lane_and_wave
from kernels.common._attention_bwd_refusals import spec_digest
from kernels.common._attention_bwd_sched import DQ_SCHEDS
from kernels.common._attention_bwd_worklist import xcd_remap_id
from kernels.common.attention_bwd import (
    BWD_BARRIER_FENCE,
    BWD_DTYPES,
    BWD_HEAD_SIZES,
    MASK_CLASSES,
    SEQ_MODES,
    SPLIT_DQ_ARCHS,
    STAGE_VECS,
    WS_LAYOUTS,
    _decode_sequences,
    attn_bwd_arch_facts,
    attn_bwd_params,
)

__all__ = [
    "DQ_AGPR_ALLOCS",
    "DQ_RESOURCE_REFUSED",
    "DQ_BLOCK_M_VALUES",
    "DQ_BLOCK_N_VALUES",
    "DQ_HEAD_LOOPS",
    "DQ_KV_PADS",
    "DQ_PROBLEM_FIELDS",
    "DQ_SCHEDS",
    "DQ_SCHED_SPILLING",
    "DQ_STATIC_SCALES",
    "DQ_TRANSPOSE_SOURCES",
    "DQ_TUNING_FIELDS",
    "AttnBwdDqSpec",
    "DqLdsPlan",
    "attn_bwd_dq_default_geometry",
    "attn_bwd_dq_grid",
    "build_attn_bwd_dq",
    "dq_lds_plan",
    "dq_register_estimates",
    "dq_rows_per_head",
    "dq_static_scale",
    "dq_static_unit_mults",
    "validate_attn_bwd_dq_spec",
]

DQ_HEAD_LOOPS = ("single", "group")
DQ_BLOCK_M_VALUES = (16, 32, 64, 128)
DQ_BLOCK_N_VALUES = (16, 32, 64, 128, 256)
DQ_TRANSPOSE_SOURCES = tuple(DQ_TRANSPOSE_READERS)
# Lever combinations the production compiler spills to scratch (resource gate,
# both measured dtype / sequence / access variants): (arch, head_size, waves,
# block_m, block_n, kv_prefetch, head_loop, edge_tiles). Each lever alone and
# group + edge are clean there; the validator refuses exactly these.
DQ_SPILLING_COMBOS = frozenset({("gfx950", 128, 4, 64, 64, True, "group", True)})
# Resolved dQ specs the production code-object compiler cannot build: a comgr
# segfault in the "iterative-ilp" scheduler, found by compiling every
# problem shape (dtype x head size x layout x mask x stage_vec x {plain, edge
# tiles, group head loop, K / V prefetch}) at the default dQ tile and the
# 8-wave 128 x 32 tile; the dQ geometries that spill to scratch on the
# production compiler in the device-test cells (2-wave 32 x 128 tiles); and the
# dQ specs of custom-configuration device cells that crash the compiler or
# spill (split dQ knob variants, the THD LSE-stride cell); the 4-wave
# 64 x 128 tile at d128 on gfx950 and at d64 on gfx942 on the dQ schedule
# suite's gate rows (the default key step spills there); keyed
# like RESOURCE_REFUSED: (arch, digest of the non-default resolved fields) ->
# the failure and the row it was found on.
DQ_RESOURCE_REFUSED = MappingProxyType(
    {
        ("gfx942", "09183592ab80660c"): "compiler crash (comgr segfault): dqilp|bf16|d64|thd|band|sv8|edge|default",
        ("gfx942", "0d992a9a5a5dc2d6"): "compiler crash (comgr segfault): dqilp|fp16|d128|thd|band|sv8|prefetch|default",
        ("gfx942", "0e98e572f05d792f"): "compiler crash (comgr segfault): dqilp|fp16|d64|batched|band|sv1|group|default",
        ("gfx942", "0f3ec3b8fb1fbd4a"): "scratch+vgpr_spill: custom numeric::test_table_case[bwd_dtype_bf16_d64_mha-custom]",
        ("gfx942", "16274981bf893c51"): "compiler crash (comgr segfault): dqilp|bf16|d128|thd|none|sv1|prefetch|default",
        ("gfx942", "196783cd474a56d1"): "compiler crash (comgr segfault): dqilp|bf16|d64|batched|none|sv1|edge|default",
        ("gfx942", "1a072f3721025cf0"): "compiler crash (comgr segfault): dqilp|fp16|d128|thd|none|sv1|prefetch|default",
        ("gfx942", "1ccd261d8e265830"): "compiler crash (comgr segfault): dqilp|bf16|d128|thd|band|sv1|prefetch|default",
        ("gfx942", "1d857fb962707a81"): "scratch+vgpr_spill: custom numeric::test_head_packing[bwd_kern_pack_thd_sq1_g8_tl_bf16_d128-custom]",
        ("gfx942", "1da5bc3f56a13957"): "compiler crash (comgr segfault): dqilp|bf16|d64|batched|band|sv1|group|default",
        ("gfx942", "252c58c3aa1a5299"): "compiler crash (comgr segfault): dqilp|fp16|d128|batched|none|sv1|prefetch|default",
        ("gfx942", "25c4d51fc7f7c1b2"): "scratch+vgpr_spill: custom numeric::test_table_case[bwd_layout_bshd_fp16_causal_tl-custom]",
        ("gfx942", "2850b7e2e3a51f3a"): "compiler crash (comgr segfault): dqilp|bf16|d64|batched|band|sv8|plain|default",
        ("gfx942", "322e463a14985320"): "compiler crash (comgr segfault): dqilp|fp16|d64|batched|none|sv8|edge|default",
        ("gfx942", "3265515ecd91cd2a"): "compiler crash (comgr segfault): dqilp|fp16|d128|thd|none|sv8|prefetch|default",
        ("gfx942", "344d64165933427d"): "scratch+vgpr_spill+sgpr_spill: custom numeric::test_table_case[bwd_dtype_bf16_d128_causal_tl-custom]",
        ("gfx942", "36523b3bfc86b46f"): "compiler crash (comgr segfault): dqilp|bf16|d128|thd|none|sv8|prefetch|default",
        ("gfx942", "3fce661e7f9933ec"): "scratch+vgpr_spill: dq_sched twin gfx942 d64 fp16 batched band sv8 n128",
        ("gfx942", "43be8c1495bca6bc"): "compiler crash (comgr segfault): dqilp|fp16|d64|thd|none|sv8|edge|default",
        ("gfx942", "44f6d242e09a9caf"): "compiler crash (comgr segfault): dqilp|bf16|d128|batched|none|sv1|prefetch|default",
        ("gfx942", "4944e81a2fa35cf2"): "compiler crash (comgr segfault): dqilp|bf16|d64|batched|band|sv1|plain|default",
        ("gfx942", "4945908bcc0f9221"): "scratch+vgpr_spill: custom numeric::test_table_case[bwd_x_gqa4_ragged_br_win_gt-custom]",
        ("gfx942", "4e9933bf1a025ce7"): "compiler crash (comgr segfault): dqilp|bf16|d64|batched|none|sv8|edge|default",
        ("gfx942", "52724705ce34b8fa"): "compiler crash (comgr segfault): dqilp|fp16|d128|batched|band|sv1|prefetch|default",
        ("gfx942", "53f24bf7a967256a"): "compiler crash (comgr segfault): dqilp|bf16|d128|batched|none|sv8|prefetch|default",
        ("gfx942", "55f5fb6a1e9018e3"): "compiler crash (comgr segfault): dqilp|fp16|d64|thd|band|sv8|plain|default",
        ("gfx942", "6563b5f289bb355e"): "compiler crash (comgr segfault): dqilp|bf16|d64|thd|none|sv8|edge|default",
        ("gfx942", "664ae1c61086214c"): "compiler crash (comgr segfault): dqilp|fp16|d64|batched|band|sv1|edge|default",
        ("gfx942", "7128388aacea1e50"): "scratch+vgpr_spill: custom numeric::test_table_case[bwd_ragged_q_lt_kv_br-custom]",
        ("gfx942", "7405d79a1dbf0424"): "compiler crash (comgr segfault): dqilp|fp16|d64|thd|band|sv1|plain|default",
        ("gfx942", "752336c650b8fe78"): "compiler crash (comgr segfault): dqilp|fp16|d64|batched|band|sv8|edge|default",
        ("gfx942", "76a921f180573bdf"): "compiler crash (comgr segfault): dqilp|fp16|d64|batched|band|sv1|plain|default",
        ("gfx942", "8364bda7a13ff36f"): "compiler crash (comgr segfault): dqilp|bf16|d64|thd|band|sv1|group|default",
        ("gfx942", "8c25a2bffb051ebe"): "compiler crash (comgr segfault): dqilp|bf16|d64|batched|band|sv1|edge|default",
        ("gfx942", "8c60e74e03ebbe08"): "compiler crash (comgr segfault): dqilp|fp16|d64|batched|band|sv8|plain|default",
        ("gfx942", "8ce360e6853eaeca"): "scratch+vgpr_spill: custom numeric::test_table_case[bwd_headdim_fp16_d64_br_gqa-custom]",
        ("gfx942", "924ff98a01816d57"): "compiler crash (comgr segfault): dqilp|bf16|d64|batched|band|sv8|edge|default",
        ("gfx942", "948a9dd4918b3232"): "compiler crash (comgr segfault): dqilp|fp16|d64|thd|band|sv1|edge|default",
        ("gfx942", "9fac47fbe0871a7d"): "compiler crash (comgr segfault): dqilp|bf16|d128|batched|band|sv1|prefetch|default",
        ("gfx942", "a0b4d52649a29165"): "compiler crash (comgr segfault): dqilp|fp16|d128|batched|none|sv8|prefetch|default",
        ("gfx942", "a7596dcf902264d6"): "scratch+vgpr_spill: plain cell c16-gfx942::bwd_decode_ragged_kv_lens",
        ("gfx942", "b612fdf3332339d4"): "compiler crash (comgr segfault): dqilp|fp16|d64|thd|band|sv1|group|default",
        ("gfx942", "b622ec77d7cb8712"): "compiler crash (comgr segfault): dqilp|bf16|d64|thd|band|sv1|plain|default",
        ("gfx942", "bb0f0749f2373e3a"): "compiler crash (comgr segfault): dqilp|bf16|d64|thd|none|sv1|edge|default",
        ("gfx942", "bde97acba14ab020"): "compiler crash (comgr segfault): dqilp|fp16|d64|thd|band|sv8|edge|default",
        ("gfx942", "c6225e0b788f0a35"): "compiler crash (comgr segfault): dqilp|fp16|d128|thd|band|sv1|prefetch|default",
        ("gfx942", "c871fb6d0b36b5db"): "compiler crash (comgr segfault): dqilp|bf16|d64|thd|band|sv8|group|default",
        ("gfx942", "c9b270c9d8a13741"): "compiler crash (comgr segfault): dqilp|bf16|d64|batched|band|sv8|group|default",
        ("gfx942", "ca468cab59d5334f"): "compiler crash (comgr segfault): dqilp|fp16|d64|thd|band|sv8|group|default",
        ("gfx942", "d231a785ca9a7c10"): "scratch+vgpr_spill: custom numeric::test_table_case[bwd_dtype_fp16_d128_mha-custom]",
        ("gfx942", "d309ba59fca486d9"): "compiler crash (comgr segfault): dqilp|bf16|d64|thd|band|sv1|edge|default",
        ("gfx942", "d44134ed00ca60b2"): "scratch+vgpr_spill: dq_sched twin gfx942 d64 bf16 thd none sv1 n128",
        ("gfx942", "d52e65d33d65cfc0"): "compiler crash (comgr segfault): dqilp|fp16|d64|batched|none|sv1|edge|default",
        ("gfx942", "d69c90b5bf9ea89f"): "scratch+vgpr_spill: custom numeric::test_head_packing[bwd_kern_pack_sq3_g2_win_bf16_d64-custom]",
        ("gfx942", "d72209bd582b3585"): "compiler crash (comgr segfault): dqilp|fp16|d64|batched|band|sv8|group|default",
        ("gfx942", "da2bdce8a350578f"): "compiler crash (comgr segfault): dqilp|bf16|d128|thd|band|sv8|prefetch|default",
        ("gfx942", "e3af8a7b8a14a1c3"): "compiler crash (comgr segfault): dqilp|bf16|d64|thd|band|sv8|plain|default",
        ("gfx942", "e8d7613fdaaa3b72"): "compiler crash (comgr segfault): dqilp|fp16|d64|thd|none|sv1|edge|default",
        ("gfx942", "ee508c3472a04291"): "compiler crash (comgr segfault): dqilp|fp16|d128|batched|band|sv8|prefetch|default",
        ("gfx942", "f615b017888adc34"): "scratch+vgpr_spill+sgpr_spill: custom numeric::test_table_case[bwd_kern_base_bshd_bf16_d128-custom]",
        ("gfx942", "f68e146d25e7bfe3"): "compiler crash (comgr segfault): dqilp|bf16|d128|batched|band|sv8|prefetch|default",
        ("gfx942", "fc12a8d59e57cddc"): "scratch+vgpr_spill: custom numeric::test_table_case[bwd_kern_base_bshd_fp16_d64-custom]",
        ("gfx950", "01c42a03998508a6"): "compiler crash (comgr segfault): dqilp|bf16|d128|batched|none|sv1|prefetch|s3",
        ("gfx950", "03fbfad4d4c39188"): "compiler crash (comgr segfault): dqilp|bf16|d128|batched|none|sv8|prefetch|default",
        ("gfx950", "0afb4aa0d7b83ff9"): "compiler crash (comgr segfault): launch c26-gfx950::split_numeric::test_split_dq_knobs[bwd_seqlen_33x72_br-lds_plain-custom]",
        ("gfx950", "108dca04d2a354cd"): "compiler crash (comgr segfault): dqilp|bf16|d128|batched|none|sv1|group|s3",
        ("gfx950", "110eb12f69f6eb8f"): "compiler crash (comgr segfault): dqilp|bf16|d128|batched|none|sv8|group|s3",
        ("gfx950", "155777e0a307c68f"): "compiler crash (comgr segfault): dqilp|bf16|d128|thd|none|sv1|edge|s3",
        ("gfx950", "1e5d66f1cce2e109"): "compiler crash (comgr segfault): dqilp|bf16|d128|batched|band|sv8|plain|default",
        ("gfx950", "1f4524852843370b"): "compiler crash (comgr segfault): dqilp|bf16|d128|thd|band|sv1|edge|s3",
        ("gfx950", "22da194becf8f6f4"): "compiler crash (comgr segfault): launch c15-gfx950::split_numeric::test_split_dq_knobs[bwd_gqa_gqa4_causal_tl-atom4_32-custom]",
        ("gfx950", "263edf5dfe075ebc"): "compiler crash (comgr segfault): dqilp|bf16|d128|thd|band|sv8|plain|default",
        ("gfx950", "2ac01b45a2160dd0"): "compiler crash (comgr segfault): dqilp|bf16|d128|batched|band|sv8|edge|s3",
        ("gfx950", "2dc2a412891bcf11"): "compiler crash (comgr segfault): dqilp|bf16|d128|thd|none|sv8|plain|default",
        ("gfx950", "31b556a65f2351f8"): "compiler crash (comgr segfault): dqilp|bf16|d128|batched|none|sv1|prefetch|default",
        ("gfx950", "324dbb363eef745f"): "compiler crash (comgr segfault): dqilp|bf16|d128|thd|none|sv8|plain|s3",
        ("gfx950", "355a57946bdb3e42"): "compiler crash (comgr segfault): dqilp|bf16|d128|thd|none|sv8|group|s3",
        ("gfx950", "3756e4d342ff232b"): "compiler crash (comgr segfault): dqilp|bf16|d128|batched|none|sv8|plain|default",
        ("gfx950", "3c224e4807cd3076"): "compiler crash (comgr segfault): dqilp|bf16|d128|batched|band|sv8|group|s3",
        ("gfx950", "3d40ad974b2ca923"): "compiler crash (comgr segfault): dqilp|bf16|d128|batched|band|sv8|prefetch|s3",
        ("gfx950", "3d5d65e03855b806"): "compiler crash (comgr segfault): dqilp|bf16|d128|thd|band|sv8|edge|s3",
        ("gfx950", "3f6a51b9934181d8"): "compiler crash (comgr segfault): dqilp|bf16|d128|thd|none|sv8|prefetch|default",
        ("gfx950", "4024f37e0d526336"): "compiler crash (comgr segfault): dqilp|bf16|d128|thd|band|sv8|group|s3",
        ("gfx950", "407a0a0175b630b5"): "compiler crash (comgr segfault): dqilp|bf16|d128|thd|none|sv8|edge|s3",
        ("gfx950", "439c3e1c14f72917"): "compiler crash (comgr segfault): dqilp|bf16|d128|batched|band|sv1|group|default",
        ("gfx950", "44b7f3c61344fb8b"): "compiler crash (comgr segfault): dqilp|bf16|d128|batched|band|sv1|edge|s3",
        ("gfx950", "479789b4d9b5b54c"): "compiler crash (comgr segfault): dqilp|bf16|d128|batched|band|sv8|group|default",
        ("gfx950", "47a21eed1bb142bb"): "compiler crash (comgr segfault): dqilp|bf16|d64|batched|band|sv1|prefetch|default",
        ("gfx950", "483637ec672bbc6f"): "compiler crash (comgr segfault): launch c15-gfx950::split_numeric::test_split_dq_knobs[bwd_gqa_gqa4_causal_tl-lds_plain-custom]",
        ("gfx950", "49819c3f95922c6a"): "compiler crash (comgr segfault): dqilp|bf16|d128|batched|none|sv1|edge|s3",
        ("gfx950", "50fa8fd6a834226d"): "compiler crash (comgr segfault): dqilp|bf16|d128|thd|band|sv1|prefetch|default",
        ("gfx950", "51d91e996dd892d7"): "compiler crash (comgr segfault): dqilp|bf16|d64|thd|band|sv8|prefetch|default",
        ("gfx950", "520a50f0a3671b55"): "compiler crash (comgr segfault): dqilp|bf16|d128|thd|band|sv1|group|default",
        ("gfx950", "53284263f3f3123c"): "compiler crash (comgr segfault): dqilp|bf16|d128|batched|band|sv8|prefetch|default",
        ("gfx950", "54336df53190eb3a"): "scratch+vgpr_spill: custom numeric::test_table_case[bwd_layout_bshd_fp16_causal_tl-custom]",
        ("gfx950", "5ef8a0a5751f7959"): "compiler crash (comgr segfault): dqilp|bf16|d128|thd|none|sv8|prefetch|s3",
        ("gfx950", "6a5ec9b5f19b8e7a"): "compiler crash (comgr segfault): dqilp|bf16|d128|thd|none|sv8|edge|default",
        ("gfx950", "6ab4fc53c267d586"): "compiler crash (comgr segfault): dqilp|bf16|d128|thd|none|sv1|plain|s3",
        ("gfx950", "6c7a177bb9512da1"): "compiler crash (comgr segfault): dqilp|bf16|d128|batched|none|sv8|plain|s3",
        ("gfx950", "6e46e7301acfa041"): "compiler crash (comgr segfault): dqilp|bf16|d128|thd|band|sv1|edge|default",
        ("gfx950", "7721fb9a2562d8ad"): "compiler crash (comgr segfault): dqilp|bf16|d128|thd|none|sv1|prefetch|default",
        ("gfx950", "775fbceb64ec505a"): "compiler crash (comgr segfault): dqilp|bf16|d128|batched|none|sv1|plain|default",
        ("gfx950", "78d3673ac925bf47"): "compiler crash (comgr segfault): dqilp|bf16|d128|batched|none|sv8|edge|s3",
        ("gfx950", "7a04107076695c42"): "compiler crash (comgr segfault): dqilp|bf16|d128|batched|band|sv1|group|s3",
        ("gfx950", "7d9b60af19f30b95"): "compiler crash (comgr segfault): dqilp|bf16|d128|thd|band|sv8|edge|default",
        ("gfx950", "8551cb4dbdecfbcd"): "compiler crash (comgr segfault): dqilp|bf16|d128|thd|none|sv1|group|s3",
        ("gfx950", "856f1b2296e76d76"): "compiler crash (comgr segfault): dqilp|bf16|d64|thd|band|sv1|prefetch|default",
        ("gfx950", "864f90b79a6a4f7f"): "compiler crash (comgr segfault): launch c26-gfx950::split_numeric::test_split_dq_knobs[bwd_seqlen_33x72_br-atom4_32-custom]",
        ("gfx950", "876a3fd2e6603175"): "compiler crash (comgr segfault): dqilp|bf16|d128|batched|none|sv8|group|default",
        ("gfx950", "89cd1bee1bb678cc"): "compiler crash (comgr segfault): dqilp|bf16|d128|thd|none|sv1|prefetch|s3",
        ("gfx950", "95664a235eb32769"): "compiler crash (comgr segfault): dqilp|bf16|d128|thd|none|sv1|plain|default",
        ("gfx950", "95c6a47cad0ca5d9"): "compiler crash (comgr segfault): dqilp|bf16|d128|batched|band|sv1|prefetch|s3",
        ("gfx950", "97d3a5bc80e61eba"): "compiler crash (comgr segfault): dqilp|bf16|d128|thd|band|sv1|plain|s3",
        ("gfx950", "9ac1f0f405ad26fd"): "compiler crash (comgr segfault): dqilp|bf16|d128|thd|band|sv8|prefetch|s3",
        ("gfx950", "9f06ce65404fd5ea"): "compiler crash (comgr segfault): dqilp|bf16|d128|batched|none|sv8|edge|default",
        ("gfx950", "a7b37d64bfed7fc4"): "compiler crash (comgr segfault): dqilp|bf16|d128|batched|none|sv8|prefetch|s3",
        ("gfx950", "afcfa5450d077513"): "compiler crash (comgr segfault): dqilp|bf16|d128|thd|band|sv8|prefetch|default",
        ("gfx950", "b1ce95e1e9249633"): "compiler crash (comgr segfault): dqilp|bf16|d128|thd|none|sv8|group|default",
        ("gfx950", "b3a9c8b5f5c5cc59"): "scratch+vgpr_spill: custom numeric::test_table_case[bwd_dtype_bf16_d128_causal_tl-custom]",
        ("gfx950", "b66b8b6977353620"): "compiler crash (comgr segfault): dqilp|bf16|d128|batched|band|sv1|plain|default",
        ("gfx950", "b7af7a797c52c484"): "compiler crash (comgr segfault): launch c15-gfx950::split_numeric::test_split_dq_knobs[bwd_ragged_gqa4_br-atom4_32-custom]",
        ("gfx950", "bfc5bf6503095835"): "compiler crash (comgr segfault): dqilp|bf16|d128|thd|band|sv1|group|s3",
        ("gfx950", "c2207cca8e9aa5f8"): "compiler crash (comgr segfault): dqilp|bf16|d128|batched|band|sv8|plain|s3",
        ("gfx950", "c3a9f8ca70834612"): "compiler crash (comgr segfault): dqilp|bf16|d128|thd|band|sv8|group|default",
        ("gfx950", "cf56b0770814cab2"): "compiler crash (comgr segfault): dqilp|bf16|d128|batched|none|sv1|group|default",
        ("gfx950", "d174246c75cc60bf"): "compiler crash (comgr segfault): dqilp|bf16|d128|batched|band|sv1|prefetch|default",
        ("gfx950", "d2d8449c60b4b33c"): "scratch+vgpr_spill: custom numeric::test_table_case[bwd_dtype_fp16_d128_mha-custom]",
        ("gfx950", "d6ca19d4769a06e3"): "compiler crash (comgr segfault): launch c15-gfx950::split_numeric::test_split_dq_knobs[bwd_ragged_gqa4_br-lds_plain-custom]",
        ("gfx950", "d786664a42031043"): "compiler crash (comgr segfault): dqilp|bf16|d128|batched|band|sv1|edge|default",
        ("gfx950", "de0d1e5eadf511a5"): "compiler crash (comgr segfault): dqilp|bf16|d64|batched|band|sv8|prefetch|default",
        ("gfx950", "df18afee1cd6c8ee"): "compiler crash (comgr segfault): dqilp|bf16|d128|batched|none|sv1|plain|s3",
        ("gfx950", "e47b2fc6ec53f578"): "compiler crash (comgr segfault): dqilp|bf16|d128|thd|band|sv1|prefetch|s3",
        ("gfx950", "e49dd005b5c33098"): "scratch+vgpr_spill: plain cell c16-gfx950::bwd_decode_ragged_kv_lens",
        ("gfx950", "ed107b218a9db0fa"): "compiler crash (comgr segfault): dqilp|bf16|d128|thd|band|sv1|plain|default",
        ("gfx950", "ed9712be375b17af"): "compiler crash (comgr segfault): dqilp|bf16|d128|thd|none|sv1|edge|default",
        ("gfx950", "eeb2c0138cbba2a4"): "scratch+vgpr_spill: dq_sched twin gfx950 d128 bf16 thd none sv1 n128",
        ("gfx950", "ef82c632482c44d8"): "scratch+vgpr_spill: dq_sched twin gfx950 d128 fp16 batched band sv8 n128",
        ("gfx950", "f847c71670b84533"): "compiler crash (comgr segfault): dqilp|bf16|d128|batched|band|sv1|plain|s3",
        ("gfx950", "f9b435da0516d3c8"): "scratch+vgpr_spill: custom numeric::test_table_case[bwd_kern_base_bshd_bf16_d128-custom]",
        ("gfx950", "fa3170445cb28d65"): "scratch+vgpr_spill: custom numeric::test_head_packing[bwd_kern_pack_thd_sq1_g8_tl_bf16_d128-custom]",
        ("gfx950", "fc908e666773afcf"): "compiler crash (comgr segfault): dqilp|bf16|d128|thd|none|sv1|group|default",
        ("gfx950", "fdcd2b2f6ca5398a"): "compiler crash (comgr segfault): dqilp|bf16|d128|batched|none|sv1|edge|default",
        ("gfx950", "fe261dbb209bda2d"): "compiler crash (comgr segfault): dqilp|bf16|d128|thd|band|sv8|plain|s3",
        ("gfx950", "fe85d82b98c66a58"): "compiler crash (comgr segfault): dqilp|bf16|d128|batched|band|sv8|edge|default",
    }
)  # fmt: skip
# Schedule values (``sched``) the production compiler spills to scratch at a
# tile where the default key step does not (resource gate, both measured
# dtype / sequence / access variants): (arch, head_size, waves, block_m,
# block_n, kv_prefetch, head_loop, edge_tiles, kv_pad, transpose_source,
# sched). The validator refuses exactly these.
DQ_SCHED_SPILLING = frozenset(
    {("gfx942", 128, 4, 64, 64, True, "single", False, 0, "lds_plain", "iglp0")}
)
# Row pad (bytes) of the K and V tiles in LDS (a bank-spreading lever).
DQ_KV_PADS = (0, 8, 16, 32)
DQ_AGPR_ALLOC_BACKEND = "backend"
DQ_AGPR_ALLOCS = (None, DQ_AGPR_ALLOC_BACKEND, (0, 0), (128, 128))
DQ_STATIC_SCALES = ("fold_ds", "convert", "runtime")
_DQ_WAVES_PER_EU = (None, 1, 2)
# Fields of the problem (from the main spec), not tuning decisions.
DQ_PROBLEM_FIELDS = (
    "head_size",
    "dtype",
    "seq_mode",
    "stage_vec",
    "mask_class",
    "ws_layout",
    "static_scale",
    "narrow_vec",
    "name",
)
_REG_ALLOWANCE_FIXED = 24
REGISTER_PRUNE_ALLOWANCE = 32
_CDNA_SIMDS_PER_CU = 4
_BUFFER_ALIGN = 128
# The lowering's flat-work-group-size attribute has a fixed minimum of 64.
_MIN_WG_ATTR = 64


def _check_in(name, value, allowed):
    if value not in allowed:
        raise ValueError(f"{name}={value!r} not in {tuple(allowed)}")


def _atom_k(atom: str) -> int:
    return int(atom.rsplit("x", 1)[1])


def dq_static_scale(main_static_scale: str) -> str:
    """The dQ kernel's ``static_scale`` for a main kernel's scale placement.

    ``"fold_ds"`` scales dS; ``"convert"`` and ``"dq_before_atomic"`` both
    scale dQ by ``dq_mult`` (there is no dQ convert under split, so the plan
    folds the convert multiplier into ``dq_mult``): one dQ kernel serves both.
    """
    table = {
        "fold_ds": "fold_ds",
        "convert": "convert",
        "dq_before_atomic": "convert",
        "runtime": "runtime",
    }
    _check_in("static_scale", main_static_scale, tuple(table))
    return table[main_static_scale]


def dq_static_unit_mults(static_scale: str) -> frozenset[str]:
    """dQ-kernel multipliers that are exactly 1.0 (not applied) under
    ``static_scale``."""
    _check_in("static_scale", static_scale, DQ_STATIC_SCALES)
    return {
        "fold_ds": frozenset({"dq_mult"}),
        "convert": frozenset({"ds_mult"}),
        "runtime": frozenset(),
    }[static_scale]


@dataclass(frozen=True)
class AttnBwdDqSpec:
    """Compile-time spec of the split dQ kernel (``head_size`` is required).

    ``None`` knobs resolve per arch (:meth:`resolved`). Tuning fields:
    ``waves``, ``block_m`` (query rows per CTA, kM0), ``block_n`` (keys per
    step, kN0), ``atom_g02`` (G0' / G2'), ``atom_g4`` (G4'),
    ``transpose_source`` (the G4' K-outer read), ``head_loop``,
    ``stats_prescale``, ``edge_tiles``, ``kv_pad``, ``kv_prefetch`` (the next
    key tile's global loads issued into registers before a step's matrix
    work; same values), ``waves_per_eu``, ``agpr_alloc``,
    ``scheduler_strategy``, ``sched`` (the key step's schedule value,
    :data:`DQ_SCHEDS`: an ``iglp_opt``, the stage fences, the stage table
    with or without the wave priority; hints only, same values); ``xcd_remap``
    is set by the plan, and ``narrow_vec`` (the ``stage_vec = 1`` access
    width) follows the main spec.
    """

    head_size: int
    dtype: str = "fp16"
    seq_mode: str = "batched"
    stage_vec: int = 8
    mask_class: str = "band"
    ws_layout: str = "head_major"
    static_scale: str = "fold_ds"  # see DQ_STATIC_SCALES
    waves: Optional[int] = None
    block_m: Optional[int] = None
    block_n: Optional[int] = None
    atom_g02: Optional[str] = None
    atom_g4: Optional[str] = None
    transpose_source: Optional[str] = None
    head_loop: str = "single"  # see DQ_HEAD_LOOPS
    stats_prescale: bool = True
    edge_tiles: bool = False
    kv_pad: int = 0  # see DQ_KV_PADS
    kv_prefetch: bool = False
    waves_per_eu: Optional[int] = None
    agpr_alloc: Optional[tuple[int, int] | str] = None  # see DQ_AGPR_ALLOCS
    scheduler_strategy: Optional[str] = None
    # the key step's schedule value (DQ_SCHEDS; "none" emits no hint)
    sched: str = "none"
    # True: the CTA decode reads the runtime xcd_n / xcd_chunk kernargs.
    xcd_remap: bool = False
    # user-tensor access width (elements) under stage_vec = 1 (from the main
    # spec; the plan narrows it to what the request allows)
    narrow_vec: int = 1
    name: str = "rocke_attn_bwd"

    def __post_init__(self) -> None:
        _check_in("head_size", self.head_size, BWD_HEAD_SIZES)
        _check_in("dtype", self.dtype, BWD_DTYPES)
        _check_in("seq_mode", self.seq_mode, SEQ_MODES)
        _check_in("stage_vec", self.stage_vec, STAGE_VECS)
        _check_in("mask_class", self.mask_class, MASK_CLASSES)
        _check_in("ws_layout", self.ws_layout, WS_LAYOUTS)
        _check_in("static_scale", self.static_scale, DQ_STATIC_SCALES)
        _check_in("head_loop", self.head_loop, DQ_HEAD_LOOPS)
        _check_in("kv_pad", self.kv_pad, DQ_KV_PADS)
        _check_in("sched", self.sched, DQ_SCHEDS)
        _check_in("narrow_vec", self.narrow_vec, (1, 2, 4))
        if self.narrow_vec != 1 and self.stage_vec != 1:
            raise ValueError("narrow_vec applies to stage_vec = 1 only")
        opt = {
            "waves": (1, 2, 4, 8),
            "block_m": DQ_BLOCK_M_VALUES,
            "block_n": DQ_BLOCK_N_VALUES,
            "atom_g02": ("16x16x16", "16x16x32"),
            "atom_g4": ("16x16x16", "16x16x32"),
            "transpose_source": DQ_TRANSPOSE_SOURCES,
            "waves_per_eu": _DQ_WAVES_PER_EU,
        }
        for fname, allowed in opt.items():
            v = getattr(self, fname)
            if v is not None:
                _check_in(fname, v, allowed)
        for fname in ("stats_prescale", "edge_tiles", "kv_prefetch", "xcd_remap"):
            if not isinstance(getattr(self, fname), bool):
                raise ValueError(
                    f"{fname} must be a bool, got {getattr(self, fname)!r}"
                )
        if self.agpr_alloc is not None:
            a = self.agpr_alloc
            a = a if isinstance(a, str) else tuple(a)
            _check_in("agpr_alloc", a, DQ_AGPR_ALLOCS)
            object.__setattr__(self, "agpr_alloc", a)
        if self.scheduler_strategy is not None:
            normalize_scheduler_strategy(self.scheduler_strategy)
        if not self.name or not self.name.replace("_", "").isalnum():
            raise ValueError(f"name={self.name!r} must be a non-empty identifier")

    def resolved(self, arch: str) -> "AttnBwdDqSpec":
        """A copy with every ``None`` geometry knob set by the arch policy and
        ``agpr_alloc = None`` resolved: ``(0, 0)`` (VGPR-form MFMA) when the
        register estimate fits the arch-VGPR file, else ``None`` (backend)."""
        defaults = attn_bwd_dq_default_geometry(
            arch, self.head_size, self.dtype, self.stage_vec
        )
        updates = {k: v for k, v in defaults.items() if getattr(self, k) is None}
        out = dataclasses.replace(self, **updates)
        if out.agpr_alloc is None:
            vgpr, _ = _raw_estimates(out, arch, agpr_form=False)
            if vgpr <= attn_bwd_arch_facts(arch).arch_vgprs:
                out = dataclasses.replace(out, agpr_alloc=(0, 0))
        return out

    def non_default_fields(self) -> tuple[tuple[str, object], ...]:
        """Compile-time fields that differ from their declared default."""
        return non_default_fields(self, skip=("head_size", "name"))

    def kernel_name(self) -> str:
        """``{name}_dq_{dtype}_d{D}_{seq_mode}_{mask_class}_sv{stage_vec}``
        plus a salt over every other non-default field."""
        base = (
            f"{self.name}_dq_{self.dtype}_d{self.head_size}_{self.seq_mode}_"
            f"{self.mask_class}_sv{self.stage_vec}"
        )
        in_base = ("dtype", "seq_mode", "stage_vec", "mask_class")
        return salted_kernel_name(
            base, [(k, v) for k, v in self.non_default_fields() if k not in in_base]
        )


# Tuning fields (the plan's ``dq_<field>`` keys); ``xcd_remap`` is plan-owned.
DQ_TUNING_FIELDS = tuple(
    f.name
    for f in fields(AttnBwdDqSpec)
    if f.name not in DQ_PROBLEM_FIELDS and f.name != "xcd_remap"
)


# ---------------------------------------------------------------------------
# default geometry policy (start points; the tuned table overrides them)
# ---------------------------------------------------------------------------

# (kM0, kN0, W) per (arch class, d): four waves split 64 query rows, 16
# columns per wave, so S^T / dP^T, the resident Q^T / dO^T and the dQ
# accumulator fit the arch-VGPR file (VGPR-form MFMA); 64 keys per step keep
# K + V of one step at 32 KiB at most (two CTAs per CU on gfx942). At 128
# keys the production compiler holds far more registers than the tile needs:
# on gfx942 (whose K-outer G4' operand is read element by element from LDS)
# d64 with stage_vec 1 or kv_prefetch spills, and on gfx950 the edge / interior
# / edge key loops reach the SGPR ceiling (the grouped d64 kernel spills).
_DQ_TILES = {
    ("gfx942", 32): (64, 64, 4),
    ("gfx942", 64): (64, 64, 4),
    ("gfx942", 128): (64, 64, 4),
    ("gfx950", 32): (64, 64, 4),
    ("gfx950", 64): (64, 64, 4),
    ("gfx950", 128): (64, 64, 4),
}


def attn_bwd_dq_default_geometry(
    arch: str, head_size: int, dtype: str = "fp16", stage_vec: int = 8
) -> dict:
    """Per-arch start point of every ``None`` geometry knob of the dQ kernel.

    G0' / G2' use the native ``16x16x32`` atom where the arch has it and D
    allows it; G4' uses ``16x16x16`` (its relabel is the identity, so the
    transpose read serves the K-outer B operand); ``tr_read`` where the arch
    has the transpose read, else ``lds_plain``. ``stage_vec`` does not change
    the start point (the global width only); it is part of the signature so a
    tuned table may key on it.
    """
    if arch not in SPLIT_DQ_ARCHS:
        raise ValueError(
            f"the split dQ kernel is built for {SPLIT_DQ_ARCHS} only, not {arch!r}"
        )
    _check_in("head_size", head_size, BWD_HEAD_SIZES)
    _check_in("dtype", dtype, BWD_DTYPES)
    _check_in("stage_vec", stage_vec, STAGE_VECS)
    facts = attn_bwd_arch_facts(arch)
    key = "gfx950" if facts.has_tr_read else "gfx942"
    km, kn, w = _DQ_TILES[(key, head_size)]
    wide = facts.wide_k_atom and head_size % 32 == 0
    return {
        "waves": w,
        "block_m": km,
        "block_n": kn,
        "atom_g02": "16x16x32" if wide else "16x16x16",
        "atom_g4": "16x16x16",
        "transpose_source": "tr_read" if facts.has_tr_read else "lds_plain",
    }


def _require_resolved(spec: AttnBwdDqSpec) -> None:
    missing = [
        n
        for n in ("waves", "block_m", "block_n", "atom_g02", "atom_g4")
        + ("transpose_source",)
        if getattr(spec, n) is None
    ]
    if missing:
        raise ValueError(f"dQ spec is not resolved (call .resolved(arch)): {missing}")


# ---------------------------------------------------------------------------
# LDS plan
# ---------------------------------------------------------------------------


def _align(n: int, a: int = _BUFFER_ALIGN) -> int:
    return (int(n) + a - 1) // a * a


@dataclass(frozen=True)
class DqLdsPlan:
    """LDS of one dQ CTA: ``{name: (byte offset, bytes, pitch elems, rows)}``.

    Three phase-aliased groups from offset 0: stage 0 (``q``, ``do``: ``kM0 x
    D``), the key loop (``k``, ``v``: ``kN0 x (D + kv_pad / 2)``) and the
    epilogue (``out``: ``kM0 x D``), each buffer on a 128-byte line.
    """

    arch: str
    buffers: tuple[tuple[str, int, int, int, int], ...]
    total_bytes: int
    capacity_bytes: int

    @property
    def fits(self) -> bool:
        return self.total_bytes <= self.capacity_bytes

    def buffer(self, name: str) -> tuple[int, int, int, int]:
        for n, off, nbytes, pitch, rows in self.buffers:
            if n == name:
                return off, nbytes, pitch, rows
        raise KeyError(f"no dQ LDS buffer {name!r}")


def dq_lds_plan(spec: AttnBwdDqSpec, arch: str) -> DqLdsPlan:
    """The LDS plan of a resolved dQ spec (not checked against capacity)."""
    _require_resolved(spec)
    eb = 2
    d, km, kn = spec.head_size, spec.block_m, spec.block_n
    kv_pitch = d + spec.kv_pad // eb
    phases = (
        (("q", km, d), ("do", km, d)),
        (("k", kn, kv_pitch), ("v", kn, kv_pitch)),
        (("out", km, d),),
    )
    buffers, total = [], 0
    for phase in phases:
        off = 0
        for name, rows, pitch in phase:
            nbytes = rows * pitch * eb
            buffers.append((name, off, nbytes, pitch, rows))
            off = _align(off + nbytes)
        total = max(total, off)
    return DqLdsPlan(
        arch=arch,
        buffers=tuple(buffers),
        total_bytes=total,
        capacity_bytes=int(attn_bwd_arch_facts(arch).lds_capacity_bytes),
    )


# ---------------------------------------------------------------------------
# registers
# ---------------------------------------------------------------------------


def _raw_estimates(
    spec: AttnBwdDqSpec, arch: str, *, agpr_form: bool
) -> tuple[int, int]:
    wave = attn_bwd_arch_facts(arch).wave_size
    w, d, km, kn = spec.waves, spec.head_size, spec.block_m, spec.block_n
    cols = km // w
    k02, k4 = _atom_k(spec.atom_g02), _atom_k(spec.atom_g4)
    vgpr = _REG_ALLOWANCE_FIXED
    vgpr += 2 * cols * d // (2 * wave)  # resident Q^T, dO^T
    vgpr += kn * cols // (2 * wave)  # dS^T as the G4' A operand
    vgpr += kn * k02 // (2 * wave)  # one K step of the G0' / G2' A fragments
    vgpr += d * k4 // (2 * wave)  # one K step of the G4' B fragments
    vgpr += 6 * max(1, cols // 16)  # per-column lse2, Dsum, flags, limits
    if spec.kv_prefetch:
        vgpr += 2 * kn * d // (2 * wave * w)  # the next K and V tile in flight
    acc = 2 * kn * cols // wave + cols * d // wave  # S^T, dP^T, dQ (fp32)
    if agpr_form:
        return vgpr, acc
    return vgpr + acc, 0


def dq_register_estimates(spec: AttnBwdDqSpec, arch: str) -> dict:
    """``{"arch_vgpr": n, "agpr": n}`` of a resolved dQ spec (module docstring)."""
    _require_resolved(spec)
    agpr_form = spec.agpr_alloc != (0, 0)
    vgpr, agpr = _raw_estimates(spec, arch, agpr_form=agpr_form)
    return {"arch_vgpr": vgpr, "agpr": agpr}


# ---------------------------------------------------------------------------
# legality
# ---------------------------------------------------------------------------


def validate_attn_bwd_dq_spec(spec: AttnBwdDqSpec, arch: str) -> AttnBwdDqSpec:
    """Resolve ``spec`` for ``arch`` and apply every static legality rule.

    Returns the resolved spec; raises ``ValueError`` naming the first broken
    rule: the target (:data:`SPLIT_DQ_ARCHS`), the catalog (waves, atoms,
    transpose source), geometry divisibility, the relabel / transpose-read
    proofs of :class:`BwdDqGeometry`, the LDS plan within the per-arch
    capacity and the two register budgets.
    """
    if arch not in SPLIT_DQ_ARCHS:
        raise ValueError(
            f"the split dQ kernel is built for {SPLIT_DQ_ARCHS} only, not {arch!r} "
            "(its dS^T hand-off is the MFMA C-to-A relabel)"
        )
    facts = attn_bwd_arch_facts(arch)
    s = spec.resolved(arch)
    w, km, kn, d = s.waves, s.block_m, s.block_n, s.head_size
    _check_in(f"waves on {arch}", w, facts.legal_waves)
    for fname in ("atom_g02", "atom_g4"):
        _check_in(f"{fname} on {arch}", getattr(s, fname), facts.legal_atoms)
    _check_in(
        f"transpose_source on {arch}",
        s.transpose_source,
        tuple(t for t in facts.legal_transpose_sources if t in DQ_TRANSPOSE_SOURCES),
    )
    if km % (16 * w):
        raise ValueError(f"block_m={km} must be a multiple of 16*waves={16 * w}")
    if kn % _atom_k(s.atom_g4):
        raise ValueError(f"block_n={kn} must be a multiple of the G4' atom K")
    if d % _atom_k(s.atom_g02):
        raise ValueError(f"head_size={d} must be a multiple of the G0' atom K")
    BwdDqGeometry.from_spec(s, arch)  # relabel / transpose-read proofs
    plan = dq_lds_plan(s, arch)
    if not plan.fits:
        raise ValueError(
            f"LDS plan {plan.total_bytes} B exceeds {plan.capacity_bytes} B on {arch}"
        )
    est = dq_register_estimates(s, arch)
    vg, ag = est["arch_vgpr"], est["agpr"]
    slack = REGISTER_PRUNE_ALLOWANCE
    if vg > facts.arch_vgprs + slack:
        raise ValueError(
            f"arch-VGPR estimate {vg} exceeds {facts.arch_vgprs} (+{slack})"
        )
    if s.agpr_alloc == (0, 0):
        if vg > facts.arch_vgprs:
            raise ValueError(
                f"agpr_alloc=(0, 0) needs arch-VGPR + AGPR estimate {vg} <= "
                f"{facts.arch_vgprs}"
            )
    elif isinstance(s.agpr_alloc, tuple):
        n = s.agpr_alloc[0]
        per_simd = max(-(-w // _CDNA_SIMDS_PER_CU), s.waves_per_eu or 1)
        share = (facts.arch_vgprs + facts.agprs) // per_simd
        if share - n < _REG_ALLOWANCE_FIXED:
            raise ValueError(
                f"agpr_alloc={s.agpr_alloc} leaves {share - n} of the {share} "
                f"registers per wave for arch VGPRs"
            )
        if n < ag:
            raise ValueError(f"agpr_alloc={s.agpr_alloc} below the AGPR estimate {ag}")
    elif ag > facts.agprs + slack:
        raise ValueError(f"AGPR estimate {ag} exceeds {facts.agprs} (+{slack})")
    if s.waves_per_eu == 2 and vg + ag > (facts.arch_vgprs + facts.agprs) // 2:
        raise ValueError(
            f"waves_per_eu=2 needs arch-VGPR + AGPR estimate {vg + ag} <= "
            f"{(facts.arch_vgprs + facts.agprs) // 2}"
        )
    per_wg = -(-w // _CDNA_SIMDS_PER_CU)
    if per_wg > 1 and vg + ag > (facts.arch_vgprs + facts.agprs) // per_wg:
        raise ValueError(
            f"workgroup share: waves={w} puts {per_wg} waves on each SIMD; "
            f"arch-VGPR + AGPR estimate {vg + ag} exceeds "
            f"{(facts.arch_vgprs + facts.agprs) // per_wg} per wave"
        )
    combo = (arch, d, w, km, kn, s.kv_prefetch, s.head_loop, s.edge_tiles)
    if combo in DQ_SPILLING_COMBOS:
        raise ValueError(
            f"kv_prefetch with head_loop='group' and edge_tiles at d{d} "
            f"(waves={w}, block_m={km}, block_n={kn}) on {arch} spills VGPRs to "
            "scratch on the production compiler (not built)"
        )
    refused = DQ_RESOURCE_REFUSED.get((arch, spec_digest(s, arch)))
    if refused is not None:
        raise ValueError(
            f"resource gate: this resolved dQ spec fails on {arch} ({refused}; "
            "DQ_RESOURCE_REFUSED)"
        )
    key = combo + (s.kv_pad, s.transpose_source, s.sched)
    if key in DQ_SCHED_SPILLING:
        raise ValueError(
            f"sched={s.sched!r} at d{d} (waves={w}, block_m={km}, block_n={kn}, "
            f"kv_prefetch={s.kv_prefetch}) on {arch} spills VGPRs to scratch on the "
            "production compiler (DQ_SCHED_SPILLING; the default key step does not)"
        )
    return s


# ---------------------------------------------------------------------------
# launch geometry (host)
# ---------------------------------------------------------------------------


def dq_rows_per_head(spec: AttnBwdDqSpec, group: int) -> int:
    """Query rows per head of a CTA: ``kM0``, or ``kM0 / G`` when packing the
    ``G`` query heads of a K / V head pair (``group`` must divide ``kM0``)."""
    _require_resolved(spec)
    if spec.head_loop == "single":
        return spec.block_m
    if group < 1 or spec.block_m % group:
        raise ValueError(
            f"head_loop='group' needs the head group G={group} to divide "
            f"block_m={spec.block_m}"
        )
    return spec.block_m // group


def attn_bwd_dq_grid(
    spec: AttnBwdDqSpec, *, s_q_max: int, h_q: int, group: int, batch: int
) -> tuple[int, int, int]:
    """``(ceil(S_q_max / R), h_q / H, B)`` (see the module docstring)."""
    rows = dq_rows_per_head(spec, group)
    heads = h_q if spec.head_loop == "single" else h_q // group
    return (-(-max(s_q_max, 1) // rows), heads, batch)


# ---------------------------------------------------------------------------
# builder
# ---------------------------------------------------------------------------

_DQ_READONLY = frozenset("Q K V dO WS_LSE2 WS_DSUM SEQ_Q SEQ_KV OFF_Q OFF_KV".split())


def _declare_dq(b: IRBuilder, io) -> dict:
    args = {}
    for name, fmt in attn_bwd_params("dq"):
        if fmt == "Q":
            if name in ("Q", "K", "V", "dO", "dQ"):
                t = PtrType(io, "global")
            elif name.startswith("WS_"):
                t = PtrType(F32, "global")
            else:
                t = PtrType(I32, "global")
            attrs = {"readonly": True} if name in _DQ_READONLY else {}
            args[name] = b.param(name, t, **attrs)
        elif fmt == "q":
            args[name] = b.param(name, I64)
        elif fmt == "f":
            args[name] = b.param(name, F32)
        else:
            args[name] = b.param(name, I32)
    return args


def _lds_views(b: IRBuilder, plan: DqLdsPlan, io) -> DqLdsViews:
    smem = b.smem_alloc(io, (plan.total_bytes // 2,), "bwd_dq_lds")

    def view(name):
        off, _nbytes, pitch, rows = plan.buffer(name)
        return LdsView(smem, off // 2, pitch, rows)

    return DqLdsViews(
        q=view("q"), do=view("do"), k=view("k"), v=view("v"), out=view("out")
    )


def _emit_dq(spec: AttnBwdDqSpec, arch: str, *, sentinel: bool = True):
    """Emit the split dQ kernel of ``spec`` on ``arch``.

    ``sentinel=False`` drops the dead-row term ``lse2 < +inf`` from the dS
    select; it exists only for the negative-control tests and is not a spec
    field.
    """
    s = validate_attn_bwd_dq_spec(spec, arch)
    facts = attn_bwd_arch_facts(arch)
    geom = BwdDqGeometry.from_spec(s, arch)
    plan = dq_lds_plan(s, arch)
    io = ir_type(s.dtype)
    km, kn = s.block_m, s.block_n

    b = IRBuilder(s.kernel_name())
    # the key-loop barriers retire the K / V reads before the next tile is
    # staged over them: fenced as in the main kernel (BWD_BARRIER_FENCE)
    b.fence_barriers = BWD_BARRIER_FENCE
    a = _declare_dq(b, io)
    # Multipliers that are 1.0 under the compiled placement are not applied.
    a.update(dict.fromkeys(dq_static_unit_mults(s.static_scale)))
    b.kernel.attrs["max_workgroup_size"] = max(geom.threads, _MIN_WG_ATTR)
    if s.agpr_alloc not in (None, DQ_AGPR_ALLOC_BACKEND):
        b.kernel.attrs["agpr_alloc"] = tuple(s.agpr_alloc)
    if s.waves_per_eu is not None:
        b.kernel.attrs["waves_per_eu"] = int(s.waves_per_eu)
    if s.scheduler_strategy is not None:
        b.kernel.attrs[SCHEDULER_STRATEGY_ATTR] = normalize_scheduler_strategy(
            s.scheduler_strategy
        )

    tid, lane, wave = lane_and_wave(b, facts.wave_size)
    zero, one = b.const_i32(0), b.const_i32(1)
    group = s.head_loop == "group"
    G = a["G"] if group else None
    # rows of one head in the CTA (CTA-uniform)
    rph = b.div(b.const_i32(km), G) if group else b.const_i32(km)
    x, y, z = b.block_id_x(), b.block_id_y(), b.block_id_z()
    if s.xcd_remap:
        gx = b.div(b.add(a["S_q_max"], b.sub(rph, one)), rph)
        gx1 = b.select(b.cmp_gt(gx, zero), gx, one)
        gy = b.div(a["h_q"], G) if group else a["h_q"]
        plane = xcd_remap_id(
            b, b.add(x, b.mul(gx1, y)), num_wgs=b.mul(gx1, gy), n=a["xcd_n"],
            chunk=a["xcd_chunk"],
        )  # fmt: skip
        x, y = b.mod(plane, gx1), b.div(plane, gx1)
    seq_q, seq_kv = _decode_sequences(b, s, a, z)
    gk = b.div(a["h_q"], a["h_k"])
    gv = b.div(a["h_q"], a["h_v"])
    h = b.mul(y, G) if group else y  # the (first) query head of the CTA
    hk, hv = b.div(h, gk), b.div(h, gv)
    q0 = b.mul(x, rph)
    rows = PackedRows(head0=h, n_heads=G, rows_per_head=rph, row0=q0) if group else None

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
    dq_addr = tensor("dQ", "dq", seq_q)
    nh = a["h_q"] if s.ws_layout == "token_major" else None

    def ws(name):
        return WorkspaceAddr(
            b, a[name], ws_rows=a["ws_rows_q"], width=1, row0=seq_q.ws_row0,
            layout=s.ws_layout, n_heads=nh,
        )  # fmt: skip

    stats = StatsView(ws("WS_LSE2"), ws("WS_DSUM"))
    if s.mask_class == "band":
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
    step = BwdDqStep(
        b, geom, mask=mask, stats=stats, lds=_lds_views(b, plan, io), tid=tid,
        lane=lane, wave=wave, scale_log2=a["scale_log2"], ds_mult=a["ds_mult"],
        prescale=s.stats_prescale, sentinel=sentinel,
    )  # fmt: skip

    # Stage 0 (only a CTA with a row < len_q reads Q / dO).
    live = b.cmp_lt(q0, seq_q.length)  # CTA-uniform
    if group:
        q_slab = do_slab = None
    else:
        q_slab = q_addr.slab(h, q0)
        do_slab = do_addr.slab(h, q0)
    step.stage_q(
        b, live=live, h=h, q0=q0, q_slab=q_slab, do_slab=do_slab, q_addr=q_addr,
        do_addr=do_addr, rows=rows,
    )  # fmt: skip

    # Key tiles of the rows (0 tiles: no live row, an empty band, len_kv = 0).
    kt_start, n_kt = kv_tile_range(b, mask.band, q0, rph, k_n0=kn)
    kt_end = b.add(kt_start, n_kt)
    if s.edge_tiles:
        i0, i1 = interior_kv_range(
            b, mask.band, q0, rph, kt_start, n_kt, k_n0=kn, masked=mask.masked
        )
        ranges = ((kt_start, i0, True), (i0, i1, False), (i1, kt_end, True))
    else:
        ranges = ((kt_start, kt_end, True),)
    trips = [b.readfirstlane(b.sub(e, s0)) for s0, e, _ in ranges]

    def pinned(slab, hint):
        # The empty asm makes the slab pointer opaque, so one 64-bit pointer
        # stays live across the key loops instead of the tensor pointer, its
        # strides and the sequence / head offsets (the backend otherwise sinks
        # the address arithmetic to the uses; the edge / grouped kernels sit at
        # the SGPR ceiling and spill without it). Same values.
        ptr = b.inline_asm(
            "", "=s,0", [slab.ptr], result_type=slab.ptr.type, sideeffect=False,
            result_name_hint=hint,
        )  # fmt: skip
        return dataclasses.replace(slab, ptr=ptr)

    k_slab = pinned(k_addr.slab(hk), "kptr")
    v_slab = pinned(v_addr.slab(hv), "vptr")
    # dQ at head h, row 0 (rows are addressed as given; the store predicates)
    dq_slab = pinned(dq_addr.slab(h, clamp=False), "dqptr")
    head_step = None
    if group:
        # the head stride (i64 elements) in bytes; fp16 / bf16 elements
        head_step = b.mul(a["dq_h"], b.const_i64(2))
    if s.kv_prefetch:
        # the first tile of the range; every step prefetches the next one, so
        # the edge / interior / edge loops (contiguous ranges) chain through
        step.prologue(
            b, k_slab, v_slab, b.mul(kt_start, b.const_i32(kn)), b.cmp_lt(zero, n_kt)
        )
    dq = step.zero_dq(b)  # after the range: the accumulators start here
    fm, fn = geom.g4t.frags_m, geom.g4t.frags_n
    for (s0, _e, edge), trip, name in zip(ranges, trips, ("k", "ki_", "ke_")):
        flat = [v for row in dq for v in row]
        loop = b.scf_for_iter(
            zero, trip, one,
            [(f"dq_{name}{i}", v) for i, v in enumerate(flat)],
            iv_name=f"dq_{name}it", elide_trailing_barrier=False,
        )  # fmt: skip
        with loop as (it, cur):
            k0 = b.mul(b.add(s0, it), b.const_i32(kn))
            acc = [list(cur[r * fn : (r + 1) * fn]) for r in range(fm)]
            out = step.run_kv_step(
                b, acc, k_slab, v_slab, k0, edge=edge, prefetch=s.kv_prefetch
            )
            b.scf_yield(*[v for row in out for v in row])
        res = list(loop.results)
        dq = [res[r * fn : (r + 1) * fn] for r in range(fm)]

    step.store_dq(
        b, dq, mult=a["dq_mult"], dq_slab=dq_slab, head_step=head_step, h=h,
        q0=q0, rows=rows,
        len_q=seq_q.length, s_q_max=a["S_q_max"], seq_mode=s.seq_mode,
    )  # fmt: skip
    b.ret()
    return b.kernel


def build_attn_bwd_dq(spec: AttnBwdDqSpec, *, arch: str):
    """Build the split dQ kernel (``rocke.attn_bwd.v3``, ``attn_bwd_params("dq")``).

    The spec is resolved and validated for ``arch`` first (an illegal
    geometry raises ``ValueError`` before any IR is emitted).
    """
    return _emit_dq(spec, arch)
