#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/gfx950_attention_tiled_2d_emit.py -- Python reference emitter for
# the gfx950 WIDE-ATOM tiled-2D unified-attention kernel parity harness.
#
# Selects one of the sampled configs by argv[1], builds a
# UnifiedAttention2DTiledSpec, emits the kernel via
# build_unified_attention_2d_tiled(spec, arch="gfx950") and prints
# lower_kernel_to_llvm(kernel, arch="gfx950") to stdout so it can be
# byte-compared with the C emitter gfx950_attention_tiled_2d_emit.c.
# prove_parity_binding keeps a copy of _CONFIGS in
# platform/cpp/bindings/attention_tiled_parity_configs.py; update it in step.
#
# The config table is kept IN LOCKSTEP with the C make_spec() switch (same index
# -> same UnifiedAttention2DTiledSpec). This is the "edge / feature-flag" cluster:
# minimal dims, GQA ratios, and every feature-flag path (qq_bias, fp8 KV,
# register-PV, i64 KV addressing, the transposed 32x32 + grouped-KV2 softmax
# stack, early-V schedule, and the fast paged-KV descriptor).
from kernels.gfx950.attention_tiled_2d import (
    UnifiedAttention2DTiledSpec,
    build_unified_attention_2d_tiled,
)
from _emit_common import run_emit


_CONFIGS = {
    # --- idx0-4: minimal dims, block_size=16, head_size {64,128,256}, GQA ratios
    0: dict(
        head_size=64,
        block_size=16,
        num_query_heads=1,
        num_kv_heads=1,
        dtype="bf16",
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
    ),
    1: dict(
        head_size=128,
        block_size=16,
        num_query_heads=1,
        num_kv_heads=1,
        dtype="bf16",
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
    ),
    2: dict(
        head_size=256,
        block_size=16,
        num_query_heads=1,
        num_kv_heads=1,
        dtype="bf16",
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
    ),
    3: dict(
        head_size=64,
        block_size=16,
        num_query_heads=16,
        num_kv_heads=1,
        dtype="bf16",
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
    ),
    4: dict(
        head_size=64,
        block_size=16,
        num_query_heads=2,
        num_kv_heads=1,
        dtype="bf16",
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
    ),
    # --- idx5-14: baseline dtype / mask-feature / head-size / block-size variety
    5: dict(
        head_size=64,
        block_size=32,
        num_query_heads=32,
        num_kv_heads=32,
        dtype="bf16",
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
    ),
    6: dict(
        head_size=64,
        block_size=32,
        num_query_heads=32,
        num_kv_heads=32,
        dtype="fp16",
        use_sinks=True,
        sliding_window=2048,
        has_softcap=True,
    ),
    7: dict(
        head_size=64,
        block_size=32,
        num_query_heads=32,
        num_kv_heads=32,
        dtype="bf16",
        use_sinks=True,
        sliding_window=1,
        has_softcap=False,
    ),
    8: dict(
        head_size=128,
        block_size=32,
        num_query_heads=32,
        num_kv_heads=32,
        dtype="fp16",
        use_sinks=True,
        sliding_window=0,
        has_softcap=True,
    ),
    9: dict(
        head_size=256,
        block_size=32,
        num_query_heads=32,
        num_kv_heads=32,
        dtype="bf16",
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
    ),
    10: dict(
        head_size=64,
        block_size=64,
        num_query_heads=32,
        num_kv_heads=32,
        dtype="bf16",
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
    ),
    11: dict(
        head_size=64,
        block_size=32,
        num_query_heads=7,
        num_kv_heads=7,
        dtype="bf16",
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
    ),
    12: dict(
        head_size=64,
        block_size=32,
        num_query_heads=64,
        num_kv_heads=8,
        dtype="bf16",
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
    ),
    13: dict(
        head_size=64,
        block_size=32,
        num_query_heads=40,
        num_kv_heads=8,
        dtype="bf16",
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
    ),
    14: dict(
        head_size=64,
        block_size=32,
        num_query_heads=128,
        num_kv_heads=1,
        dtype="bf16",
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
    ),
    # --- idx15: QQ-bias feature flag
    15: dict(
        head_size=64,
        block_size=32,
        num_query_heads=32,
        num_kv_heads=32,
        dtype="bf16",
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
        use_qq_bias=True,
    ),
    # --- idx16,17: ALiBi / composite mask features
    16: dict(
        head_size=64,
        block_size=32,
        num_query_heads=32,
        num_kv_heads=32,
        dtype="bf16",
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
        use_alibi=True,
    ),
    17: dict(
        head_size=64,
        block_size=32,
        num_query_heads=32,
        num_kv_heads=32,
        dtype="fp16",
        use_sinks=True,
        sliding_window=512,
        has_softcap=True,
        use_alibi=True,
        use_qq_bias=True,
    ),
    # --- idx18: num_warps=8 (BLOCK_M=128), no tile_size
    18: dict(
        head_size=64,
        block_size=64,
        num_query_heads=64,
        num_kv_heads=8,
        dtype="bf16",
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
        num_warps=8,
    ),
    # --- idx19-26: num_warps / tile_size / waves_per_eu / num_seqs variety
    19: dict(
        head_size=64,
        block_size=32,
        num_query_heads=32,
        num_kv_heads=32,
        dtype="bf16",
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
        num_warps=2,
    ),
    20: dict(
        head_size=64,
        block_size=32,
        num_query_heads=64,
        num_kv_heads=8,
        dtype="bf16",
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
        num_warps=4,
    ),
    21: dict(
        head_size=64,
        block_size=32,
        num_query_heads=32,
        num_kv_heads=32,
        dtype="bf16",
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
        tile_size=64,
    ),
    22: dict(
        head_size=64,
        block_size=32,
        num_query_heads=32,
        num_kv_heads=32,
        dtype="bf16",
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
        tile_size=128,
    ),
    23: dict(
        head_size=128,
        block_size=32,
        num_query_heads=32,
        num_kv_heads=32,
        dtype="fp16",
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
        num_warps=2,
        tile_size=128,
    ),
    24: dict(
        head_size=64,
        block_size=32,
        num_query_heads=32,
        num_kv_heads=32,
        dtype="bf16",
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
        waves_per_eu=2,
    ),
    25: dict(
        head_size=64,
        block_size=32,
        num_query_heads=32,
        num_kv_heads=32,
        dtype="bf16",
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
        num_seqs=1,
    ),
    26: dict(
        head_size=64,
        block_size=32,
        num_query_heads=32,
        num_kv_heads=32,
        dtype="bf16",
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
        num_seqs=257,
    ),
    # --- idx27: fp8 KV cache with native fp8 PV MFMA
    27: dict(
        head_size=64,
        block_size=32,
        num_query_heads=32,
        num_kv_heads=32,
        dtype="bf16",
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
        kv_storage_dtype="fp8e4m3",
        use_fp8_mfma_pv=True,
    ),
    # --- idx28: 64-bit paged-KV addressing
    28: dict(
        head_size=64,
        block_size=32,
        num_query_heads=32,
        num_kv_heads=32,
        dtype="bf16",
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
        use_i64_kv_addr=True,
    ),
    # --- idx29: register-PV bf16 path
    29: dict(
        head_size=64,
        block_size=32,
        num_query_heads=32,
        num_kv_heads=32,
        dtype="bf16",
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
        use_register_pv=True,
    ),
    # --- idx30-32: fp8 KV (dequant), fp8 QK MFMA, mfma_32x32 base
    30: dict(
        head_size=64,
        block_size=32,
        num_query_heads=32,
        num_kv_heads=32,
        dtype="bf16",
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
        kv_storage_dtype="fp8e4m3",
    ),
    31: dict(
        head_size=64,
        block_size=32,
        num_query_heads=32,
        num_kv_heads=32,
        dtype="bf16",
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
        kv_storage_dtype="fp8e4m3",
        use_fp8_mfma_qk=True,
    ),
    32: dict(
        head_size=128,
        block_size=32,
        num_query_heads=32,
        num_kv_heads=32,
        dtype="fp16",
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
        use_mfma_32x32=True,
        block_m_per_warp=32,
        tile_size=64,
    ),
    # --- idx33-35: transposed 32x32 + scalar-state + invariant-hoist + mask-once
    #               + grouped-KV2 softmax stack (bf16, BLOCK_M=128 warp slice)
    33: dict(
        head_size=64,
        block_size=32,
        num_query_heads=32,
        num_kv_heads=32,
        dtype="bf16",
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
        use_mfma_32x32=True,
        use_transposed_qk_32x32=True,
        use_transposed_scalar_state=True,
        use_transposed_invariant_hoist=True,
        use_transposed_mask_once=True,
        use_grouped_kv2_softmax=True,
        block_m_per_warp=32,
        tile_size=64,
    ),
    34: dict(
        head_size=128,
        block_size=32,
        num_query_heads=32,
        num_kv_heads=32,
        dtype="bf16",
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
        use_mfma_32x32=True,
        use_transposed_qk_32x32=True,
        use_transposed_scalar_state=True,
        use_transposed_invariant_hoist=True,
        use_transposed_mask_once=True,
        use_grouped_kv2_softmax=True,
        block_m_per_warp=32,
        tile_size=64,
    ),
    35: dict(
        head_size=64,
        block_size=32,
        num_query_heads=64,
        num_kv_heads=8,
        dtype="bf16",
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
        use_mfma_32x32=True,
        use_transposed_qk_32x32=True,
        use_transposed_scalar_state=True,
        use_transposed_invariant_hoist=True,
        use_transposed_mask_once=True,
        use_grouped_kv2_softmax=True,
        num_warps=4,
        block_m_per_warp=32,
        tile_size=64,
    ),
    # --- idx36: early-V schedule
    36: dict(
        head_size=64,
        block_size=32,
        num_query_heads=32,
        num_kv_heads=32,
        dtype="bf16",
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
        use_early_v_schedule=True,
    ),
    # --- idx37: fast paged-KV descriptor (bf16 h64kv8 HD=64 BS=32 T=64 nw=4)
    37: dict(
        head_size=64,
        block_size=32,
        num_query_heads=64,
        num_kv_heads=8,
        dtype="bf16",
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
        num_warps=4,
        block_m_per_warp=16,
        tile_size=64,
        use_fast_paged_kv_desc=True,
    ),
    # --- idx38-41: transposed 32x32 + half-local-PV (hlpv) reschedule stack.
    #     hlpv requires use_transposed_qk_32x32; covers d64/d128 x fp16/bf16.
    38: dict(
        head_size=64,
        block_size=32,
        num_query_heads=32,
        num_kv_heads=32,
        dtype="bf16",
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
        use_mfma_32x32=True,
        use_transposed_qk_32x32=True,
        use_transposed_half_local_pv=True,
        block_m_per_warp=32,
        tile_size=64,
    ),
    39: dict(
        head_size=128,
        block_size=32,
        num_query_heads=32,
        num_kv_heads=32,
        dtype="bf16",
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
        use_mfma_32x32=True,
        use_transposed_qk_32x32=True,
        use_transposed_half_local_pv=True,
        block_m_per_warp=32,
        tile_size=64,
    ),
    40: dict(
        head_size=64,
        block_size=32,
        num_query_heads=32,
        num_kv_heads=32,
        dtype="fp16",
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
        use_mfma_32x32=True,
        use_transposed_qk_32x32=True,
        use_transposed_half_local_pv=True,
        block_m_per_warp=32,
        tile_size=64,
    ),
    41: dict(
        head_size=128,
        block_size=32,
        num_query_heads=32,
        num_kv_heads=32,
        dtype="fp16",
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
        use_mfma_32x32=True,
        use_transposed_qk_32x32=True,
        use_transposed_half_local_pv=True,
        block_m_per_warp=32,
        tile_size=64,
    ),
    # --- idx42: exact provider build_sdpa_tiled hlpv shape (combo multi_batch
    #     GQA-8 h64kv8 bf16: the shape the provider prefill path routes to). ---
    42: dict(
        head_size=64,
        block_size=32,
        num_query_heads=64,
        num_kv_heads=8,
        dtype="bf16",
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
        num_seqs=2,
        num_warps=4,
        use_mfma_32x32=True,
        use_transposed_qk_32x32=True,
        use_transposed_scalar_state=True,
        use_transposed_mask_once=True,
        use_transposed_half_local_pv=True,
        block_m_per_warp=32,
        tile_size=64,
    ),
    # --- idx43-49: gfx950 PRODUCTION combo knobs the selector
    #     (_tiled_spec_from_problem gfx950 path) sets but the gate never
    #     sampled: the FULL no-SW combo with use_transposed_mask_limit ON
    #     (idx33-35 left it OFF), use_v_double_buffer, use_early_v_schedule
    #     stacked on the mask-limit combo, and use_grouped_kv2_softmax on the
    #     main 2D builder (previously only in the fastkv_regp variant). ---
    # idx43: FULL no-SW combo incl. use_transposed_mask_limit + skip_legacy +
    #        hlpv + fast_paged_kv (the bf16 GQA-8 d64 production combo).
    43: dict(
        head_size=64,
        block_size=32,
        num_query_heads=64,
        num_kv_heads=8,
        dtype="bf16",
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
        num_warps=4,
        block_m_per_warp=32,
        tile_size=64,
        use_mfma_32x32=True,
        use_transposed_qk_32x32=True,
        use_transposed_scalar_state=True,
        use_transposed_invariant_hoist=True,
        use_transposed_mask_once=True,
        use_transposed_mask_limit=True,
        use_mfma32_skip_legacy_qreg=True,
        use_transposed_half_local_pv=True,
        use_fast_paged_kv_desc=True,
    ),
    # idx44: d128 full mask-limit combo (no fast_paged_kv; HD=128).
    44: dict(
        head_size=128,
        block_size=32,
        num_query_heads=32,
        num_kv_heads=32,
        dtype="bf16",
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
        num_warps=4,
        block_m_per_warp=32,
        tile_size=64,
        use_mfma_32x32=True,
        use_transposed_qk_32x32=True,
        use_transposed_scalar_state=True,
        use_transposed_invariant_hoist=True,
        use_transposed_mask_once=True,
        use_transposed_mask_limit=True,
        use_mfma32_skip_legacy_qreg=True,
        use_transposed_half_local_pv=True,
    ),
    # idx45: fp16 full mask-limit combo.
    45: dict(
        head_size=64,
        block_size=32,
        num_query_heads=32,
        num_kv_heads=32,
        dtype="fp16",
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
        num_warps=4,
        block_m_per_warp=32,
        tile_size=64,
        use_mfma_32x32=True,
        use_transposed_qk_32x32=True,
        use_transposed_scalar_state=True,
        use_transposed_invariant_hoist=True,
        use_transposed_mask_once=True,
        use_transposed_mask_limit=True,
        use_mfma32_skip_legacy_qreg=True,
        use_transposed_half_local_pv=True,
    ),
    # idx46: mask-limit combo with invariant-hoist + skip_legacy but WITHOUT
    #        half_local_pv (the plain-PV transposed combo the multi-batch
    #        selector picks when hlpv regresses), num_warps=2 BLOCK_M=64.
    46: dict(
        head_size=64,
        block_size=32,
        num_query_heads=32,
        num_kv_heads=32,
        dtype="bf16",
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
        num_warps=2,
        block_m_per_warp=32,
        tile_size=64,
        use_mfma_32x32=True,
        use_transposed_qk_32x32=True,
        use_transposed_scalar_state=True,
        use_transposed_invariant_hoist=True,
        use_transposed_mask_once=True,
        use_transposed_mask_limit=True,
        use_mfma32_skip_legacy_qreg=True,
    ),
    # idx47: use_early_v_schedule on the mask-limit combo (long-prefill
    #        single-batch combo V-prefetch winner; mutually exclusive with
    #        v_double_buffer).
    47: dict(
        head_size=64,
        block_size=32,
        num_query_heads=64,
        num_kv_heads=8,
        dtype="bf16",
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
        num_warps=4,
        block_m_per_warp=32,
        tile_size=64,
        use_mfma_32x32=True,
        use_transposed_qk_32x32=True,
        use_transposed_scalar_state=True,
        use_transposed_mask_once=True,
        use_transposed_mask_limit=True,
        use_transposed_half_local_pv=True,
        use_early_v_schedule=True,
    ),
    # idx48: use_grouped_kv2_softmax on the MAIN 2D builder (the bf16
    #        transposed R4 grouped-KV2 path; gate previously only sampled it
    #        in the fastkv_regp variant).
    48: dict(
        head_size=64,
        block_size=32,
        num_query_heads=64,
        num_kv_heads=8,
        dtype="bf16",
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
        num_warps=4,
        block_m_per_warp=32,
        tile_size=64,
        use_mfma_32x32=True,
        use_transposed_qk_32x32=True,
        use_transposed_scalar_state=True,
        use_grouped_kv2_softmax=True,
    ),
    # idx49: single-batch (num_seqs=1) full mask-limit combo with early_v
    #        schedule stacked (the _enable_single_batch_combo long-prefill
    #        production winner: mask-limit VALU + hlpv + early-V issue).
    49: dict(
        head_size=64,
        block_size=32,
        num_query_heads=64,
        num_kv_heads=8,
        dtype="bf16",
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
        num_seqs=1,
        num_warps=4,
        block_m_per_warp=32,
        tile_size=64,
        use_mfma_32x32=True,
        use_transposed_qk_32x32=True,
        use_transposed_scalar_state=True,
        use_transposed_invariant_hoist=True,
        use_transposed_mask_once=True,
        use_transposed_mask_limit=True,
        use_mfma32_skip_legacy_qreg=True,
        use_transposed_half_local_pv=True,
        use_early_v_schedule=True,
    ),
    # idx50: lever-3 sched_barrier (#51/#57) on the single-batch d128 combo
    # (num_warps=1, the cohort the dispatcher auto-enables it for). Exercises the
    # rocke_b_sched_barrier emit point in the C twin's transposed-QK path.
    50: dict(
        head_size=128,
        block_size=32,
        num_query_heads=64,
        num_kv_heads=8,
        dtype="bf16",
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
        num_seqs=1,
        num_warps=1,
        block_m_per_warp=32,
        tile_size=64,
        use_mfma_32x32=True,
        use_transposed_qk_32x32=True,
        use_transposed_scalar_state=True,
        use_transposed_invariant_hoist=True,
        use_transposed_mask_once=True,
        use_transposed_mask_limit=True,
        use_mfma32_skip_legacy_qreg=True,
        use_transposed_half_local_pv=True,
        use_sched_barrier=True,
    ),
    # idx51: #66 small-tile d128 2-WG/CU win path -- the full single-batch d128
    # combo with tile_size=32 (== block_size, one paged block per iter) at
    # num_warps=2. This is the geometry that halves K_lds/V_lds (48->24 KB) and
    # drops VGPR (229->173) -> 2 WG/CU at d128. It uses NO new code path (just a
    # smaller tile_size on the existing combo), so it MUST be C/Python identical.
    51: dict(
        head_size=128,
        block_size=32,
        num_query_heads=64,
        num_kv_heads=8,
        dtype="bf16",
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
        num_seqs=1,
        num_warps=2,
        block_m_per_warp=32,
        tile_size=32,
        use_mfma_32x32=True,
        use_transposed_qk_32x32=True,
        use_transposed_scalar_state=True,
        use_transposed_invariant_hoist=True,
        use_transposed_mask_once=True,
        use_transposed_mask_limit=True,
        use_mfma32_skip_legacy_qreg=True,
        use_transposed_half_local_pv=True,
    ),
    # idx52 (renumbered from 53; kept contiguous so the gate's shared
    # end-of-range detection doesn't stop early on a hole; the ring-3 config is
    # idx93): #69 K single-buffer at T=64 -- the d128 long-context 2-WG/CU
    # win. The full single-batch d128 combo with tile_size=64 (== 2*block_size)
    # AND use_k_single_buffer=True. K_lds collapses 2->1 slot (32->16 KB) so
    # total LDS is 32 KB -> 2 WG/CU at the LARGER tile (better S4096 amortisation
    # than the idx51 T=32 pick). The next-K prefetch is re-issued AFTER the
    # PV-wait s_barrier (all QK reads drained) so the single slot cannot
    # WAR-race. This IS ported to the gfx950 C twin (K_BUFS=1 + deferred K
    # issue), so it MUST be C/Python byte-identical.
    52: dict(
        head_size=128,
        block_size=32,
        num_query_heads=64,
        num_kv_heads=8,
        dtype="bf16",
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
        num_seqs=1,
        num_warps=2,
        block_m_per_warp=32,
        tile_size=64,
        use_k_single_buffer=True,
        use_mfma_32x32=True,
        use_transposed_qk_32x32=True,
        use_transposed_scalar_state=True,
        use_transposed_invariant_hoist=True,
        use_transposed_mask_once=True,
        use_transposed_mask_limit=True,
        use_mfma32_skip_legacy_qreg=True,
        use_transposed_half_local_pv=True,
    ),
    # idx53: the production nw=4 softmax<->MFMA interleave config. This is exactly
    # what the gfx950 single-batch d128 selector now routes to (see
    # _enable_softmax_mfma_interleave): block_size=64 -> tile_size=128 (2*BS) so
    # BLOCK_M=128 (nw4 * mw32) fits the single K slot (block_m <= tile_size). The
    # interleave emits one iglp_opt(1) after the iter-start K drain; the gfx950 C
    # twin ports it (kv_body_qk_softmax.cpp), so it MUST be C/Python byte-identical.
    53: dict(
        head_size=128,
        block_size=64,
        num_query_heads=64,
        num_kv_heads=8,
        dtype="bf16",
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
        num_seqs=1,
        num_warps=4,
        block_m_per_warp=32,
        tile_size=128,
        use_k_single_buffer=True,
        use_mfma_32x32=True,
        use_transposed_qk_32x32=True,
        use_transposed_scalar_state=True,
        use_transposed_invariant_hoist=True,
        use_transposed_mask_once=True,
        use_transposed_mask_limit=True,
        use_mfma32_skip_legacy_qreg=True,
        use_transposed_half_local_pv=True,
        use_softmax_mfma_interleave=True,
        softmax_interleave_mode=1,
    ),
    # fp16 + sinks combo (D64/b32/GQA-8): the shipping spec for the fp16-sink
    # combo path -- exactly what `_tiled_spec_from_problem` emits for the Gate-1
    # fp16+sinks problem, incl. use_transposed_mask_limit + skip_legacy_qreg.
    # fp16 cannot set use_fast_paged_kv_desc (bf16-only), so it stays off.
    54: dict(
        head_size=64,
        block_size=32,
        num_query_heads=64,
        num_kv_heads=8,
        dtype="fp16",
        use_sinks=True,
        sliding_window=0,
        has_softcap=False,
        num_seqs=2,
        num_warps=4,
        use_mfma_32x32=True,
        use_transposed_qk_32x32=True,
        use_transposed_scalar_state=True,
        use_transposed_mask_once=True,
        use_transposed_mask_limit=True,
        use_mfma32_skip_legacy_qreg=True,
        use_transposed_half_local_pv=True,
        block_m_per_warp=32,
        tile_size=64,
    ),
    # bf16 + sinks combo (D64/b32/GQA-8): the shipping spec for the bf16-sink
    # combo path. Same as idx54 but dtype=bf16, which additionally enables
    # use_fast_paged_kv_desc (bf16-only). `_enable_combo_2d` admits bf16+sinks
    # but no other emit case covers it.
    55: dict(
        head_size=64,
        block_size=32,
        num_query_heads=64,
        num_kv_heads=8,
        dtype="bf16",
        use_sinks=True,
        sliding_window=0,
        has_softcap=False,
        num_seqs=2,
        num_warps=4,
        use_mfma_32x32=True,
        use_transposed_qk_32x32=True,
        use_transposed_scalar_state=True,
        use_transposed_mask_once=True,
        use_transposed_mask_limit=True,
        use_mfma32_skip_legacy_qreg=True,
        use_transposed_half_local_pv=True,
        use_fast_paged_kv_desc=True,
        block_m_per_warp=32,
        tile_size=64,
    ),
    # --- idx56-57: FP8 K/V single-batch combo (sync-dequant loader feeding the
    #               transposed 32x32 stack; d128 T64 and d64 T64, nw=4)
    56: dict(
        head_size=128,
        block_size=64,
        num_query_heads=64,
        num_kv_heads=8,
        dtype="bf16",
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
        num_seqs=1,
        num_warps=4,
        waves_per_eu=2,
        kv_storage_dtype="fp8e4m3",
        use_mfma_32x32=True,
        use_transposed_qk_32x32=True,
        use_transposed_scalar_state=True,
        use_transposed_mask_once=True,
        use_transposed_mask_limit=True,
        use_mfma32_skip_legacy_qreg=True,
        use_transposed_half_local_pv=True,
        block_m_per_warp=32,
        tile_size=64,
    ),
    57: dict(
        head_size=64,
        block_size=64,
        num_query_heads=64,
        num_kv_heads=8,
        dtype="bf16",
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
        num_seqs=1,
        num_warps=4,
        waves_per_eu=2,
        kv_storage_dtype="fp8e4m3",
        use_mfma_32x32=True,
        use_transposed_qk_32x32=True,
        use_transposed_scalar_state=True,
        use_transposed_mask_once=True,
        use_transposed_mask_limit=True,
        use_mfma32_skip_legacy_qreg=True,
        use_transposed_half_local_pv=True,
        block_m_per_warp=32,
        tile_size=64,
    ),
    # --- idx58: head_size=192 MHA prefill selector spec
    58: dict(
        head_size=192,
        block_size=16,
        num_query_heads=96,
        num_kv_heads=96,
        dtype="bf16",
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
        num_seqs=1,
        num_warps=2,
        waves_per_eu=2,
        use_register_pv=True,
        tile_size=32,
    ),
    # --- idx59: head_size=192 single-batch combo prefill selector spec (T=64, softmax/MFMA interleave)
    59: dict(
        head_size=192,
        block_size=16,
        num_query_heads=96,
        num_kv_heads=96,
        dtype="bf16",
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
        num_seqs=1,
        num_warps=4,
        waves_per_eu=2,
        tile_size=64,
        block_m_per_warp=32,
        use_mfma_32x32=True,
        use_transposed_qk_32x32=True,
        use_transposed_scalar_state=True,
        use_transposed_mask_once=True,
        use_transposed_half_local_pv=True,
        use_mfma32_skip_legacy_qreg=True,
        use_transposed_mask_limit=True,
        use_softmax_mfma_interleave=True,
    ),
    # --- idx60: d128 single-batch sliding-window + sinks combo selector spec
    #     (T=64, nw=4, transposed 32x32 + scalar state + half-local PV + interleave)
    60: dict(
        head_size=128,
        block_size=64,
        num_query_heads=128,
        num_kv_heads=8,
        dtype="fp16",
        use_sinks=True,
        sliding_window=128,
        has_softcap=False,
        num_seqs=1,
        num_warps=4,
        waves_per_eu=2,
        tile_size=64,
        block_m_per_warp=32,
        use_mfma_32x32=True,
        use_transposed_qk_32x32=True,
        use_transposed_scalar_state=True,
        use_transposed_half_local_pv=True,
        use_softmax_mfma_interleave=True,
    ),
    # --- idx61: d128 single-batch sinks (no window) full-combo selector spec
    #     (T=2*block_size, K single buffer, softmax/MFMA interleave)
    61: dict(
        head_size=128,
        block_size=64,
        num_query_heads=128,
        num_kv_heads=8,
        dtype="bf16",
        use_sinks=True,
        sliding_window=0,
        has_softcap=False,
        num_seqs=1,
        num_warps=4,
        waves_per_eu=2,
        tile_size=128,
        block_m_per_warp=32,
        use_mfma_32x32=True,
        use_transposed_qk_32x32=True,
        use_transposed_scalar_state=True,
        use_transposed_mask_once=True,
        use_transposed_half_local_pv=True,
        use_mfma32_skip_legacy_qreg=True,
        use_transposed_mask_limit=True,
        use_k_single_buffer=True,
        use_softmax_mfma_interleave=True,
    ),  # --- idx62: d64 single-batch sliding-window + sinks combo selector spec (long:
    #     early V; the short-prefill V double buffer has no C twin yet)
    62: dict(
        head_size=64,
        block_size=64,
        num_query_heads=64,
        num_kv_heads=8,
        dtype="bf16",
        use_sinks=True,
        sliding_window=128,
        has_softcap=False,
        num_seqs=1,
        num_warps=4,
        waves_per_eu=2,
        tile_size=128,
        block_m_per_warp=32,
        use_mfma_32x32=True,
        use_transposed_qk_32x32=True,
        use_transposed_scalar_state=True,
        use_transposed_half_local_pv=True,
        use_early_v_schedule=True,
    ),
    # Longest-first query blocks (D256/b16/GQA-8, two sequences): the reversed
    # grid-y decode reads total_q from cu_q[num_seqs].
    63: dict(
        head_size=256,
        block_size=16,
        num_query_heads=16,
        num_kv_heads=2,
        dtype="bf16",
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
        num_seqs=2,
        use_qblock_reverse=True,
    ),
    # cfg68: bf16 D64/b16 GQA-4 multi-batch: the shipping spec, whose one K/V copy per
    # tile (32 tokens) spans two pages, so each wave's page comes from the wave.
    68: dict(
        head_size=64,
        block_size=16,
        num_query_heads=32,
        num_kv_heads=8,
        dtype="bf16",
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
        num_seqs=2,
        num_warps=4,
        waves_per_eu=2,
        tile_size=32,
        block_m_per_warp=32,
        use_mfma_32x32=True,
        use_transposed_qk_32x32=True,
        use_transposed_scalar_state=True,
        use_transposed_mask_once=True,
        use_transposed_half_local_pv=True,
        use_mfma32_skip_legacy_qreg=True,
        use_transposed_mask_limit=True,
    ),
    # cfg69/70: page copies over 16 and 18 pages per tile (page ids read in place).
    69: dict(
        head_size=64,
        block_size=16,
        num_query_heads=32,
        num_kv_heads=8,
        dtype="bf16",
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
        num_seqs=2,
        tile_size=256,
        block_m_per_warp=32,
        use_mfma_32x32=True,
        use_transposed_qk_32x32=True,
    ),
    70: dict(
        head_size=64,
        block_size=16,
        num_query_heads=32,
        num_kv_heads=8,
        dtype="bf16",
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
        num_seqs=2,
        tile_size=288,
        block_m_per_warp=32,
        use_mfma_32x32=True,
        use_transposed_qk_32x32=True,
    ),
    # cfg71: bf16 D192/b16 one warp, the spec dispatch picks for short D192
    # prefill. One copy is 512 halves, 2 2/3 rows, so it is not a whole number
    # of tokens and each lane's position in its page is split per copy.
    71: dict(
        head_size=192,
        block_size=16,
        num_query_heads=16,
        num_kv_heads=2,
        dtype="bf16",
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
        num_seqs=4,
        waves_per_eu=2,
        use_register_pv=True,
        tile_size=32,
    ),
    # idx137-110: additive bias. idx107: the 16x16 path with two M atoms per warp
    # (GQA).
    107: dict(
        head_size=128,
        block_size=16,
        num_query_heads=16,
        num_kv_heads=4,
        dtype="bf16",
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
        use_additive_bias=True,
        num_seqs=3,
        num_warps=2,
        block_m_per_warp=32,
        tile_size=32,
    ),
    # idx108: the 16x16 path composed with ALiBi, QQ bias, softcap, sinks and a
    # sliding window (the ALiBi slope replaces the hoisted query head).
    108: dict(
        head_size=64,
        block_size=32,
        num_query_heads=32,
        num_kv_heads=8,
        dtype="fp16",
        use_sinks=True,
        sliding_window=512,
        has_softcap=True,
        use_alibi=True,
        use_qq_bias=True,
        use_additive_bias=True,
    ),
    # idx109: the transposed 32x32 combo the selector builds for a biased
    # d64/b32/GQA-8 problem (mask-once and mask-limit off).
    109: dict(
        head_size=64,
        block_size=32,
        num_query_heads=64,
        num_kv_heads=8,
        dtype="bf16",
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
        use_additive_bias=True,
        num_seqs=2,
        num_warps=4,
        block_m_per_warp=32,
        tile_size=64,
        use_mfma_32x32=True,
        use_transposed_qk_32x32=True,
        use_transposed_scalar_state=True,
        use_transposed_half_local_pv=True,
        use_mfma32_skip_legacy_qreg=True,
        use_fast_paged_kv_desc=True,
    ),
    # idx110: the bare transposed 32x32 path with QQ bias.
    110: dict(
        head_size=128,
        block_size=32,
        num_query_heads=16,
        num_kv_heads=4,
        dtype="fp16",
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
        use_qq_bias=True,
        use_additive_bias=True,
        num_warps=2,
        block_m_per_warp=32,
        tile_size=64,
        use_mfma_32x32=True,
        use_transposed_qk_32x32=True,
    ),
    # idx141, 112: additive bias with paths that do not emit it; both engines
    # reject these (BOTH_REJECTED).
    111: dict(
        head_size=64,
        block_size=32,
        num_query_heads=64,
        num_kv_heads=8,
        dtype="bf16",
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
        use_additive_bias=True,
        num_warps=4,
        block_m_per_warp=32,
        tile_size=64,
        use_mfma_32x32=True,
        use_transposed_qk_32x32=True,
        use_transposed_scalar_state=True,
        use_transposed_mask_once=True,
    ),
    112: dict(
        head_size=64,
        block_size=32,
        num_query_heads=32,
        num_kv_heads=8,
        dtype="bf16",
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
        use_additive_bias=True,
        use_register_pv=True,
    ),
    # idx143-116: more additive-bias paths that do not emit it; both engines
    # reject these too (BOTH_REJECTED). 81/82 cover mask-once and
    # register-PV; these four cover the remaining rejections so losing any
    # one of them would go unnoticed (the non-transposed 32x32 path has no
    # bias site in either engine, so C++ would otherwise silently emit a
    # kernel that takes the bias params and never adds them; an
    # invariant-hoist spec would pass a NULL query head into
    # rocke_add_additive_bias).
    113: dict(
        head_size=64,
        block_size=32,
        num_query_heads=64,
        num_kv_heads=8,
        dtype="bf16",
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
        use_additive_bias=True,
        num_warps=4,
        block_m_per_warp=32,
        tile_size=64,
        use_mfma_32x32=True,
        use_transposed_qk_32x32=True,
        use_transposed_invariant_hoist=True,
    ),
    114: dict(
        head_size=64,
        block_size=32,
        num_query_heads=64,
        num_kv_heads=8,
        dtype="bf16",
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
        use_additive_bias=True,
        num_warps=4,
        block_m_per_warp=32,
        tile_size=64,
        use_mfma_32x32=True,
        use_transposed_qk_32x32=False,
    ),
    115: dict(
        head_size=64,
        block_size=32,
        num_query_heads=64,
        num_kv_heads=8,
        dtype="bf16",
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
        use_additive_bias=True,
        num_warps=4,
        block_m_per_warp=32,
        tile_size=64,
        use_mfma_32x32=True,
        use_transposed_qk_32x32=True,
        use_transposed_scalar_state=True,
        use_grouped_kv2_softmax=True,
    ),
    116: dict(
        head_size=128,
        block_size=32,
        num_query_heads=32,
        num_kv_heads=32,
        dtype="bf16",
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
        use_additive_bias=True,
        num_warps=4,
        block_m_per_warp=32,
        tile_size=64,
        use_mfma_32x32=True,
        use_transposed_qk_32x32=True,
        use_transposed_scalar_state=True,
        use_transposed_mask_once=True,
        use_transposed_mask_limit=True,
        use_mfma32_skip_legacy_qreg=True,
        use_transposed_half_local_pv=True,
    ),
}

# --- idx64-67: emit_lse (trailing lse_ptr + FP32 LSE store), one per epilogue
#     store site, each an existing config plus emit_lse=True.
#   idx64: transposed32 scalar-state (idx55 + LSE).
#   idx65: coalesced mfma32 Acc_lds epilogue (idx32 + LSE).
#   idx66: striped16 epilogue, GQA-16 (idx3 + LSE).
#   idx67: striped16 + ALiBi (idx5 + ALiBi + LSE): the LSE row index must use
#          the query head, not the hoisted ALiBi slope.
_CONFIGS[64] = {**_CONFIGS[55], "emit_lse": True}
_CONFIGS[65] = {**_CONFIGS[32], "emit_lse": True}
_CONFIGS[66] = {**_CONFIGS[3], "emit_lse": True}
_CONFIGS[67] = {**_CONFIGS[5], "use_alibi": True, "emit_lse": True}
# --- idx72-74: lse_layout="head_major" at each store site (idx64-66 + head-major).
# idx71 is cfg71 above (bf16 D192 one warp); these three sit after it.
_CONFIGS[72] = {**_CONFIGS[64], "lse_layout": "head_major"}
_CONFIGS[73] = {**_CONFIGS[65], "lse_layout": "head_major"}
_CONFIGS[74] = {**_CONFIGS[66], "lse_layout": "head_major"}
# idx117: additive bias + emit_lse on the idx107 path; the four bias
# params precede lse_ptr in both engines.
_CONFIGS[117] = {**_CONFIGS[107], "emit_lse": True}

# Configs both engines must reject, with the reason each one pins: run_diff
# requires the regex in both engines' stderr (BOTH_REJECTED alone accepts any
# reason, a typo included).
_EXPECTED_REJECTIONS = {
    111: r"use_additive_bias does not support use_transposed_mask_once",
    112: r"use_register_pv v1 does not support QQ bias or additive bias",
    113: r"use_additive_bias does not support use_transposed_invariant_hoist",
    114: r"use_additive_bias does not support the non-transposed 32x32 path",
    115: r"use_grouped_kv2_softmax v1 does not support .*additive bias",
    116: r"transposed softmax VALU opts do not support .*additive bias",
}

# --- idx75-76: fp8 K in LDS on each 32x32 QK path.
# fp8 KV cache kept as raw fp8 in K_lds (use_fp8_mfma_qk) on the
# non-transposed 32x32 QK path: K is read as fp8 and dequantized with
# k_scale before the bf16 MFMA, as on the transposed path.
_CONFIGS[75] = dict(
    head_size=64,
    block_size=32,
    num_query_heads=32,
    num_kv_heads=32,
    dtype="bf16",
    use_sinks=False,
    sliding_window=0,
    has_softcap=False,
    kv_storage_dtype="fp8e4m3",
    use_fp8_mfma_qk=True,
    use_mfma_32x32=True,
    block_m_per_warp=32,
    tile_size=64,
)
# The same fp8 K in LDS on the transposed 32x32 QK path, which reads K through
# the same shared reader.
_CONFIGS[76] = {**_CONFIGS[75], "use_transposed_qk_32x32": True}

# ALiBi on the transposed 32x32 path without mask-once: the slope is loaded
# inline per lane. idx77 is the multi-batch fp16 D128 spec
# _tiled_spec_from_problem selects, idx78 the bf16 combo spec (scalar state,
# still no mask-once). idx79 takes the hoisted slope (invariant hoist).
_CONFIGS[77] = dict(
    head_size=128,
    block_size=16,
    num_query_heads=8,
    num_kv_heads=8,
    dtype="fp16",
    use_sinks=False,
    sliding_window=0,
    has_softcap=False,
    use_alibi=True,
    num_seqs=4,
    num_warps=4,
    waves_per_eu=2,
    tile_size=32,
    block_m_per_warp=32,
    use_mfma_32x32=True,
    use_transposed_qk_32x32=True,
    use_transposed_half_local_pv=True,
)

_CONFIGS[78] = dict(
    head_size=64,
    block_size=32,
    num_query_heads=64,
    num_kv_heads=8,
    dtype="bf16",
    use_sinks=False,
    sliding_window=0,
    has_softcap=False,
    use_alibi=True,
    num_seqs=4,
    num_warps=4,
    waves_per_eu=4,
    tile_size=64,
    block_m_per_warp=32,
    use_mfma_32x32=True,
    use_transposed_qk_32x32=True,
    use_transposed_scalar_state=True,
    use_transposed_half_local_pv=True,
    use_mfma32_skip_legacy_qreg=True,
    use_fast_paged_kv_desc=True,
)

_CONFIGS[79] = dict(
    head_size=128,
    block_size=16,
    num_query_heads=8,
    num_kv_heads=8,
    dtype="fp16",
    use_sinks=False,
    sliding_window=0,
    has_softcap=False,
    use_alibi=True,
    num_seqs=4,
    num_warps=4,
    waves_per_eu=2,
    tile_size=32,
    block_m_per_warp=32,
    use_mfma_32x32=True,
    use_transposed_qk_32x32=True,
    use_transposed_half_local_pv=True,
    use_transposed_invariant_hoist=True,
    use_transposed_mask_once=True,
)

# V double-buffer (V[i+1] prefetched into the alternate V_lds slot). idx80 is
# the bf16 D64 GQA-8 single-batch prefill spec _tiled_spec_from_problem
# selects; idx81 the fp16 MHA one (tile 128) plus the staggered iter-start
# wait. idx84-84 cover the other QK/PV paths: narrow 16x16, non-transposed
# 32x32 with the staggered wait, and early-V + sched_barrier at D128.
_CONFIGS[80] = dict(
    head_size=64,
    block_size=32,
    num_query_heads=64,
    num_kv_heads=8,
    dtype="bf16",
    use_sinks=False,
    sliding_window=0,
    has_softcap=False,
    num_seqs=1,
    num_warps=4,
    waves_per_eu=2,
    use_v_double_buffer=True,
    tile_size=64,
    block_m_per_warp=32,
    use_mfma_32x32=True,
    use_transposed_qk_32x32=True,
    use_transposed_scalar_state=True,
    use_transposed_mask_once=True,
    use_transposed_half_local_pv=True,
    use_mfma32_skip_legacy_qreg=True,
    use_transposed_mask_limit=True,
    use_fast_paged_kv_desc=True,
)

_CONFIGS[81] = dict(
    head_size=64,
    block_size=32,
    num_query_heads=32,
    num_kv_heads=32,
    dtype="fp16",
    use_sinks=False,
    sliding_window=0,
    has_softcap=False,
    num_seqs=1,
    num_warps=4,
    waves_per_eu=2,
    use_v_double_buffer=True,
    use_staggered_iter_wait=True,
    tile_size=128,
    block_m_per_warp=32,
    use_mfma_32x32=True,
    use_transposed_qk_32x32=True,
    use_transposed_scalar_state=True,
    use_transposed_mask_once=True,
    use_transposed_half_local_pv=True,
    use_mfma32_skip_legacy_qreg=True,
    use_transposed_mask_limit=True,
)

_CONFIGS[82] = dict(
    head_size=64,
    block_size=32,
    num_query_heads=8,
    num_kv_heads=8,
    dtype="bf16",
    use_sinks=False,
    sliding_window=0,
    has_softcap=False,
    num_seqs=1,
    num_warps=2,
    use_v_double_buffer=True,
)

_CONFIGS[83] = dict(
    head_size=64,
    block_size=32,
    num_query_heads=8,
    num_kv_heads=8,
    dtype="bf16",
    use_sinks=False,
    sliding_window=0,
    has_softcap=False,
    num_seqs=1,
    num_warps=2,
    block_m_per_warp=32,
    use_mfma_32x32=True,
    use_v_double_buffer=True,
    use_staggered_iter_wait=True,
)

_CONFIGS[84] = dict(
    head_size=128,
    block_size=32,
    num_query_heads=8,
    num_kv_heads=8,
    dtype="fp16",
    use_sinks=False,
    sliding_window=0,
    has_softcap=False,
    num_seqs=1,
    num_warps=1,
    use_v_double_buffer=True,
    use_early_v_schedule=True,
    use_sched_barrier=True,
)

# D256 fast path (_d256_gfx950_spec_overrides): idx85 is the bf16 GQA-8 spec
# _tiled_spec_from_problem selects (direct-to-register Q, K-slab LDS pad,
# mask phase split, interleave mode 2), idx86 the same with the invariant
# hoist, idx89 the MHA block-16 shape. idx89-88 take the levers apart at
# D128: direct-register Q alone; K-slab pad + phase split + mode 2 (2 groups).
_CONFIGS[85] = dict(
    head_size=256,
    block_size=32,
    num_query_heads=64,
    num_kv_heads=8,
    dtype="bf16",
    use_sinks=False,
    sliding_window=0,
    has_softcap=False,
    num_seqs=2,
    num_warps=2,
    waves_per_eu=2,
    tile_size=64,
    block_m_per_warp=32,
    use_mfma_32x32=True,
    use_transposed_qk_32x32=True,
    use_kq_lds_pad=True,
    kq_lds_pad_halves=16,
    use_transposed_scalar_state=True,
    use_transposed_mask_once=True,
    use_transposed_half_local_pv=True,
    use_transposed_mask_limit=True,
    use_k_single_buffer=True,
    use_q_direct_reg=True,
    use_softmax_mfma_interleave=True,
    softmax_interleave_mode=2,
    use_mask_phase_split=True,
)

_CONFIGS[86] = dict(
    head_size=256,
    block_size=32,
    num_query_heads=64,
    num_kv_heads=8,
    dtype="bf16",
    use_sinks=False,
    sliding_window=0,
    has_softcap=False,
    num_seqs=2,
    num_warps=2,
    waves_per_eu=2,
    tile_size=64,
    block_m_per_warp=32,
    use_mfma_32x32=True,
    use_transposed_qk_32x32=True,
    use_kq_lds_pad=True,
    kq_lds_pad_halves=16,
    use_transposed_scalar_state=True,
    use_transposed_mask_once=True,
    use_transposed_half_local_pv=True,
    use_transposed_mask_limit=True,
    use_k_single_buffer=True,
    use_q_direct_reg=True,
    use_softmax_mfma_interleave=True,
    softmax_interleave_mode=2,
    use_mask_phase_split=True,
    use_transposed_invariant_hoist=True,
)

_CONFIGS[87] = dict(
    head_size=128,
    block_size=32,
    num_query_heads=64,
    num_kv_heads=8,
    dtype="bf16",
    use_sinks=False,
    sliding_window=0,
    has_softcap=False,
    num_seqs=2,
    num_warps=4,
    tile_size=64,
    block_m_per_warp=32,
    use_mfma_32x32=True,
    use_transposed_qk_32x32=True,
    use_transposed_scalar_state=True,
    use_transposed_mask_once=True,
    use_transposed_half_local_pv=True,
    use_transposed_mask_limit=True,
    use_q_direct_reg=True,
)

_CONFIGS[88] = dict(
    head_size=128,
    block_size=32,
    num_query_heads=64,
    num_kv_heads=8,
    dtype="fp16",
    use_sinks=False,
    sliding_window=0,
    has_softcap=False,
    num_seqs=1,
    num_warps=2,
    tile_size=64,
    block_m_per_warp=32,
    use_mfma_32x32=True,
    use_transposed_qk_32x32=True,
    use_transposed_scalar_state=True,
    use_transposed_mask_once=True,
    use_transposed_half_local_pv=True,
    use_transposed_mask_limit=True,
    use_k_single_buffer=True,
    use_kq_lds_pad=True,
    use_q_direct_reg=True,
    use_mask_phase_split=True,
    use_softmax_mfma_interleave=True,
    softmax_interleave_mode=2,
    softmax_interleave_groups=2,
)

_CONFIGS[89] = dict(
    head_size=256,
    block_size=16,
    num_query_heads=32,
    num_kv_heads=32,
    dtype="bf16",
    use_sinks=False,
    sliding_window=0,
    has_softcap=False,
    num_seqs=1,
    num_warps=2,
    waves_per_eu=2,
    tile_size=64,
    block_m_per_warp=32,
    use_mfma_32x32=True,
    use_transposed_qk_32x32=True,
    use_kq_lds_pad=True,
    kq_lds_pad_halves=16,
    use_transposed_scalar_state=True,
    use_transposed_mask_once=True,
    use_transposed_half_local_pv=True,
    use_transposed_mask_limit=True,
    use_k_single_buffer=True,
    use_q_direct_reg=True,
    use_softmax_mfma_interleave=True,
    softmax_interleave_mode=2,
    use_mask_phase_split=True,
)

# fp8-in-LDS K (native-fp8 MFMA QK loader) on the transposed 32x32 path: the
# transposed QK reads K through the shared dequantizing K reader.
_CONFIGS[90] = dict(
    head_size=64,
    block_size=32,
    num_query_heads=64,
    num_kv_heads=8,
    dtype="bf16",
    use_sinks=False,
    sliding_window=0,
    has_softcap=False,
    num_seqs=2,
    num_warps=4,
    tile_size=64,
    block_m_per_warp=32,
    use_mfma_32x32=True,
    use_transposed_qk_32x32=True,
    use_transposed_scalar_state=True,
    use_transposed_half_local_pv=True,
    kv_storage_dtype="fp8e4m3",
    use_fp8_mfma_qk=True,
)

# Q re-read (Q operand read from a dedicated Q_lds inside the QK loop) and
# the depth-3 deep K prefetch ring. idx91 re-reads Q on the transposed
# 32x32 combo, idx92 adds the grouped-KV2 second score tile, idx93 is the
# d128 small-tile ring with the two-phase mask peel, idx94 combines both.
_CONFIGS[91] = dict(
    head_size=128,
    block_size=32,
    num_query_heads=64,
    num_kv_heads=8,
    dtype="bf16",
    use_sinks=False,
    sliding_window=0,
    has_softcap=False,
    num_seqs=2,
    num_warps=4,
    tile_size=64,
    block_m_per_warp=32,
    use_mfma_32x32=True,
    use_transposed_qk_32x32=True,
    use_transposed_scalar_state=True,
    use_transposed_mask_once=True,
    use_transposed_half_local_pv=True,
    use_transposed_mask_limit=True,
    use_q_reread=True,
)

_CONFIGS[92] = dict(
    head_size=128,
    block_size=32,
    num_query_heads=32,
    num_kv_heads=32,
    dtype="bf16",
    use_sinks=False,
    sliding_window=0,
    has_softcap=False,
    use_mfma_32x32=True,
    use_transposed_qk_32x32=True,
    use_transposed_scalar_state=True,
    use_transposed_invariant_hoist=True,
    use_transposed_mask_once=True,
    use_grouped_kv2_softmax=True,
    block_m_per_warp=32,
    tile_size=64,
    use_q_reread=True,
)

_CONFIGS[93] = dict(
    head_size=128,
    block_size=32,
    num_query_heads=64,
    num_kv_heads=8,
    dtype="bf16",
    use_sinks=False,
    sliding_window=0,
    has_softcap=False,
    num_seqs=1,
    num_warps=4,
    tile_size=32,
    block_m_per_warp=32,
    use_mfma_32x32=True,
    use_transposed_qk_32x32=True,
    use_transposed_scalar_state=True,
    use_transposed_mask_once=True,
    use_transposed_half_local_pv=True,
    use_transposed_mask_limit=True,
    use_mask_phase_split=True,
    kv_ring_depth=3,
)

_CONFIGS[94] = dict(
    head_size=64,
    block_size=16,
    num_query_heads=16,
    num_kv_heads=16,
    dtype="fp16",
    use_sinks=True,
    sliding_window=128,
    has_softcap=True,
    num_seqs=2,
    num_warps=2,
    tile_size=32,
    block_m_per_warp=32,
    use_mfma_32x32=True,
    use_transposed_qk_32x32=True,
    use_transposed_scalar_state=True,
    use_transposed_half_local_pv=True,
    kv_ring_depth=3,
    use_q_reread=True,
)

# Selector spec for single-batch fp16 D128 sliding-window prefill (the
# 4-warp T=64 transposed-32x32 combo with softmax/MFMA interleave).
_CONFIGS[95] = dict(
    head_size=128,
    block_size=16,
    num_query_heads=32,
    num_kv_heads=8,
    dtype="fp16",
    use_sinks=False,
    sliding_window=256,
    has_softcap=False,
    num_seqs=1,
    num_warps=4,
    waves_per_eu=2,
    tile_size=64,
    block_m_per_warp=32,
    use_mfma_32x32=True,
    use_transposed_qk_32x32=True,
    use_transposed_half_local_pv=True,
    use_softmax_mfma_interleave=True,
)

# Config 50 with a non-zero sched_barrier mask (MFMA may cross the fence).
_CONFIGS[96] = dict(
    head_size=128,
    block_size=32,
    num_query_heads=64,
    num_kv_heads=8,
    dtype="bf16",
    use_sinks=False,
    sliding_window=0,
    has_softcap=False,
    num_seqs=1,
    num_warps=1,
    block_m_per_warp=32,
    tile_size=64,
    use_mfma_32x32=True,
    use_transposed_qk_32x32=True,
    use_transposed_scalar_state=True,
    use_transposed_invariant_hoist=True,
    use_transposed_mask_once=True,
    use_transposed_mask_limit=True,
    use_mfma32_skip_legacy_qreg=True,
    use_transposed_half_local_pv=True,
    use_sched_barrier=True,
    sched_barrier_mask=8,
)

# --- idx99-101: top-left / right-bound mask band.
_CONFIGS[97] = dict(
    head_size=64,
    block_size=16,
    num_query_heads=8,
    num_kv_heads=2,
    dtype="bf16",
    use_sinks=False,
    sliding_window=0,
    has_softcap=False,
    num_seqs=2,
    causal_top_left=True,
)

_CONFIGS[98] = dict(
    head_size=128,
    block_size=32,
    num_query_heads=16,
    num_kv_heads=4,
    dtype="fp16",
    use_sinks=False,
    sliding_window=64,
    has_softcap=False,
    num_warps=2,
    use_mfma_32x32=True,
    block_m_per_warp=32,
    tile_size=64,
    right_bound=5,
)

_CONFIGS[99] = dict(
    head_size=128,
    block_size=32,
    num_query_heads=32,
    num_kv_heads=8,
    dtype="bf16",
    use_sinks=True,
    sliding_window=0,
    has_softcap=False,
    num_seqs=2,
    num_warps=4,
    tile_size=64,
    block_m_per_warp=32,
    use_mfma_32x32=True,
    use_transposed_qk_32x32=True,
    use_transposed_scalar_state=True,
    use_transposed_half_local_pv=True,
    right_bound=-1,
)

_CONFIGS[100] = dict(
    head_size=128,
    block_size=32,
    num_query_heads=64,
    num_kv_heads=8,
    dtype="bf16",
    use_sinks=False,
    sliding_window=128,
    has_softcap=True,
    num_seqs=1,
    num_warps=1,
    block_m_per_warp=32,
    tile_size=64,
    use_mfma_32x32=True,
    use_transposed_qk_32x32=True,
    use_transposed_scalar_state=True,
    use_transposed_invariant_hoist=True,
    use_transposed_mask_once=True,
    use_transposed_half_local_pv=True,
    causal_top_left=True,
    right_bound=3,
)

_CONFIGS[101] = dict(
    head_size=64,
    block_size=32,
    num_query_heads=32,
    num_kv_heads=32,
    dtype="bf16",
    use_sinks=False,
    sliding_window=0,
    has_softcap=False,
    kv_storage_dtype="fp8e4m3",
    use_fp8_mfma_pv=True,
    causal_top_left=True,
)

# --- idx104-106: the band on the transposed mask-limit path. idx102 hoisted causal
# limit with rb4, idx103 unbounded, idx106-105 the D256 fast path (idx85, mask phase
# split) unbounded and top-left + rb7, idx106 the per-iteration causal limit with
# top-left + rb2.
_CONFIGS[102] = dict(_CONFIGS[44], right_bound=4)

_CONFIGS[103] = dict(_CONFIGS[46], right_bound=-1)

_CONFIGS[104] = dict(_CONFIGS[85], right_bound=-1)

_CONFIGS[105] = dict(_CONFIGS[85], causal_top_left=True, right_bound=7)

_CONFIGS[106] = dict(
    _CONFIGS[46],
    use_transposed_invariant_hoist=False,
    causal_top_left=True,
    right_bound=2,
)

# --- idx118-122: K/V page copies (i64 page base, pages from the wave, one
# 64-token page per tile).
# i64 K/V, two pages per tile, one page per copy (page copies, i64 page base).
_CONFIGS[118] = dict(
    head_size=128,
    block_size=16,
    num_query_heads=32,
    num_kv_heads=8,
    dtype="bf16",
    use_sinks=False,
    sliding_window=0,
    has_softcap=False,
    tile_size=32,
    use_i64_kv_addr=True,
)
# cfg71 with i64 K/V (mid-row copies, i64 page base).
_CONFIGS[119] = {**_CONFIGS[71], "use_i64_kv_addr": True}
# i64 K/V with each wave's page from the wave: whole-row copies spanning pages
# (cfg68) and mid-row copies (cfg59).
_CONFIGS[120] = {**_CONFIGS[68], "use_i64_kv_addr": True}
_CONFIGS[121] = {**_CONFIGS[59], "use_i64_kv_addr": True}
# cfg48 with one 64-token page per tile: the page base stays in soffset
# without prefetching (grouped KV2 carries no page ids).
_CONFIGS[122] = {**_CONFIGS[48], "block_size": 64}
# cfg85 with V double buffered (K double buffered too): the mask-free first
# phase still bounds V, since the body also loads the next tile's V.
_CONFIGS[123] = {
    **_CONFIGS[85],
    "use_k_single_buffer": False,
    "use_v_double_buffer": True,
}

# --- idx124-127: ALiBi at the slope sites idx16/17 miss: per-row hoist (124, GQA with 24 heads so the slope formula's
#     tail heads appear), and the transposed 32x32 per-iter (125, mask-once),
#     hoisted (126) and inline (127) per-lane slopes.
_CONFIGS[124] = dict(
    head_size=64,
    block_size=32,
    num_query_heads=24,
    num_kv_heads=6,
    dtype="bf16",
    use_sinks=False,
    sliding_window=0,
    has_softcap=False,
    use_alibi=True,
)
_TRANSPOSED_ALIBI = dict(
    head_size=64,
    block_size=32,
    num_query_heads=32,
    num_kv_heads=32,
    dtype="bf16",
    use_sinks=False,
    sliding_window=0,
    has_softcap=False,
    use_alibi=True,
    use_mfma_32x32=True,
    use_transposed_qk_32x32=True,
    block_m_per_warp=32,
    tile_size=64,
)
_CONFIGS[125] = {**_TRANSPOSED_ALIBI, "use_transposed_mask_once": True}
_CONFIGS[126] = {**_TRANSPOSED_ALIBI, "use_transposed_invariant_hoist": True}
_CONFIGS[127] = dict(_TRANSPOSED_ALIBI)

# --- idx128-142: the ALiBi arms (alibi_design). 128-131 each arm on the transposed
#     mask-limit softmax; default arms: 132 D256 mask-limit (fma_split), 133 D192
#     mask-limit (seed_tile), 134 grouped_kv2 and 135 register-P (fma_tile); 136-139
#     each arm on the 16x16 path (two atoms, four N-tiles); 140-142 the other arms
#     on grouped_kv2 (second-tile seed, two split groups).
_MLIM_ALIBI = {
    **_TRANSPOSED_ALIBI,
    "use_transposed_scalar_state": True,
    "use_transposed_mask_once": True,
    "use_transposed_half_local_pv": True,
    "use_mfma32_skip_legacy_qreg": True,
    "use_transposed_mask_limit": True,
}
_ALIBI_16X16 = dict(
    head_size=64,
    block_size=32,
    num_query_heads=24,
    num_kv_heads=6,
    dtype="bf16",
    use_sinks=False,
    sliding_window=0,
    has_softcap=False,
    use_alibi=True,
    block_m_per_warp=32,
    tile_size=64,
)
_CONFIGS[128] = {**_MLIM_ALIBI, "alibi_design": "seed_tile"}
_CONFIGS[129] = {**_MLIM_ALIBI, "alibi_design": "fma_tile"}
_CONFIGS[130] = {**_MLIM_ALIBI, "alibi_design": "seed_split"}
_CONFIGS[131] = {**_MLIM_ALIBI, "alibi_design": "fma_split"}
_CONFIGS[132] = {**_MLIM_ALIBI, "head_size": 256}
_CONFIGS[133] = {**_MLIM_ALIBI, "head_size": 192}
_CONFIGS[134] = {**_TRANSPOSED_ALIBI, "use_grouped_kv2_softmax": True}
_CONFIGS[135] = dict(
    head_size=64,
    block_size=32,
    num_query_heads=32,
    num_kv_heads=32,
    dtype="bf16",
    use_sinks=False,
    sliding_window=0,
    has_softcap=False,
    use_alibi=True,
    num_seqs=2,
    waves_per_eu=2,
    use_register_pv=True,
    tile_size=64,
)
_CONFIGS[136] = {**_ALIBI_16X16, "alibi_design": "seed_tile"}
_CONFIGS[137] = {**_ALIBI_16X16, "alibi_design": "fma_tile"}
_CONFIGS[138] = {**_ALIBI_16X16, "alibi_design": "seed_split"}
_CONFIGS[139] = {**_ALIBI_16X16, "alibi_design": "fma_split"}
_CONFIGS[140] = {**_CONFIGS[134], "alibi_design": "fma_split"}
_CONFIGS[141] = {**_CONFIGS[134], "alibi_design": "seed_tile"}
_CONFIGS[142] = {**_CONFIGS[134], "alibi_design": "seed_split"}


def _spec(idx: int) -> UnifiedAttention2DTiledSpec:
    if idx not in _CONFIGS:
        raise SystemExit(f"unknown config index {idx}")
    return UnifiedAttention2DTiledSpec(**_CONFIGS[idx])


def main() -> int:
    return run_emit(
        _spec,
        build_unified_attention_2d_tiled,
        usage="usage: gfx950_attention_tiled_2d_emit.py <config_index>\n",
    )


if __name__ == "__main__":
    raise SystemExit(main())
