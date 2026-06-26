/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/gfx950_attention_tiled_2d_emit.c -- C-side emitter for the gfx950
 * WIDE-ATOM tiled-2D unified-attention parity harness.
 *
 * Selects one of the sampled configs by argv[1], fills a
 * rocke_attention_tiled_2d_spec_t identically to the Python emitter
 * gfx950_attention_tiled_2d_emit.py, builds the kernel via
 * rocke_gfx950_build_unified_attention_2d_tiled_new(&b, &spec, "gfx950"), lowers it
 * with rocke_lower_kernel_to_llvm(kernel, AUTO, "gfx950", ...) and prints the .ll
 * to stdout for byte comparison.
 *
 * The config table is kept IN LOCKSTEP with the Python emitter's _CONFIGS dict
 * (same index -> same UnifiedAttention2DTiledSpec). This is the "edge /
 * feature-flag" cluster: minimal dims, GQA ratios, and every feature-flag path.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/instance_gfx950_attention_tiled_2d.h"
#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"
#include "rocke/verify.h"

/* Fill `s` for config index `idx`. Returns 0 on success, -1 on unknown idx. */
static int make_spec(int idx, rocke_attention_tiled_2d_spec_t* s)
{
    *s = rocke_attention_tiled_2d_spec_default();
    switch(idx)
    {
    /* --- idx0-4: minimal dims, block_size=16, head_size {64,128,256}, GQA --- */
    case 0:
        s->head_size = 64;
        s->block_size = 16;
        s->num_query_heads = 1;
        s->num_kv_heads = 1;
        s->dtype = "bf16";
        s->use_sinks = false;
        s->sliding_window = 0;
        s->has_softcap = false;
        break;
    case 1:
        s->head_size = 128;
        s->block_size = 16;
        s->num_query_heads = 1;
        s->num_kv_heads = 1;
        s->dtype = "bf16";
        s->use_sinks = false;
        s->sliding_window = 0;
        s->has_softcap = false;
        break;
    case 2:
        s->head_size = 256;
        s->block_size = 16;
        s->num_query_heads = 1;
        s->num_kv_heads = 1;
        s->dtype = "bf16";
        s->use_sinks = false;
        s->sliding_window = 0;
        s->has_softcap = false;
        break;
    case 3:
        s->head_size = 64;
        s->block_size = 16;
        s->num_query_heads = 16;
        s->num_kv_heads = 1;
        s->dtype = "bf16";
        s->use_sinks = false;
        s->sliding_window = 0;
        s->has_softcap = false;
        break;
    case 4:
        s->head_size = 64;
        s->block_size = 16;
        s->num_query_heads = 2;
        s->num_kv_heads = 1;
        s->dtype = "bf16";
        s->use_sinks = false;
        s->sliding_window = 0;
        s->has_softcap = false;
        break;

    /* --- idx5-14: baseline dtype / mask / head-size / block-size variety --- */
    case 5:
        s->head_size = 64;
        s->block_size = 32;
        s->num_query_heads = 32;
        s->num_kv_heads = 32;
        s->dtype = "bf16";
        s->use_sinks = false;
        s->sliding_window = 0;
        s->has_softcap = false;
        break;
    case 6:
        s->head_size = 64;
        s->block_size = 32;
        s->num_query_heads = 32;
        s->num_kv_heads = 32;
        s->dtype = "fp16";
        s->use_sinks = true;
        s->sliding_window = 2048;
        s->has_softcap = true;
        break;
    case 7:
        s->head_size = 64;
        s->block_size = 32;
        s->num_query_heads = 32;
        s->num_kv_heads = 32;
        s->dtype = "bf16";
        s->use_sinks = true;
        s->sliding_window = 1;
        s->has_softcap = false;
        break;
    case 8:
        s->head_size = 128;
        s->block_size = 32;
        s->num_query_heads = 32;
        s->num_kv_heads = 32;
        s->dtype = "fp16";
        s->use_sinks = true;
        s->sliding_window = 0;
        s->has_softcap = true;
        break;
    case 9:
        s->head_size = 256;
        s->block_size = 32;
        s->num_query_heads = 32;
        s->num_kv_heads = 32;
        s->dtype = "bf16";
        s->use_sinks = false;
        s->sliding_window = 0;
        s->has_softcap = false;
        break;
    case 10:
        s->head_size = 64;
        s->block_size = 64;
        s->num_query_heads = 32;
        s->num_kv_heads = 32;
        s->dtype = "bf16";
        s->use_sinks = false;
        s->sliding_window = 0;
        s->has_softcap = false;
        break;
    case 11:
        s->head_size = 64;
        s->block_size = 32;
        s->num_query_heads = 7;
        s->num_kv_heads = 7;
        s->dtype = "bf16";
        s->use_sinks = false;
        s->sliding_window = 0;
        s->has_softcap = false;
        break;
    case 12:
        s->head_size = 64;
        s->block_size = 32;
        s->num_query_heads = 64;
        s->num_kv_heads = 8;
        s->dtype = "bf16";
        s->use_sinks = false;
        s->sliding_window = 0;
        s->has_softcap = false;
        break;
    case 13:
        s->head_size = 64;
        s->block_size = 32;
        s->num_query_heads = 40;
        s->num_kv_heads = 8;
        s->dtype = "bf16";
        s->use_sinks = false;
        s->sliding_window = 0;
        s->has_softcap = false;
        break;
    case 14:
        s->head_size = 64;
        s->block_size = 32;
        s->num_query_heads = 128;
        s->num_kv_heads = 1;
        s->dtype = "bf16";
        s->use_sinks = false;
        s->sliding_window = 0;
        s->has_softcap = false;
        break;

    /* --- idx15: QQ-bias feature flag --- */
    case 15:
        s->head_size = 64;
        s->block_size = 32;
        s->num_query_heads = 32;
        s->num_kv_heads = 32;
        s->dtype = "bf16";
        s->use_sinks = false;
        s->sliding_window = 0;
        s->has_softcap = false;
        s->use_qq_bias = true;
        break;

    /* --- idx16,17: ALiBi / composite mask features --- */
    case 16:
        s->head_size = 64;
        s->block_size = 32;
        s->num_query_heads = 32;
        s->num_kv_heads = 32;
        s->dtype = "bf16";
        s->use_sinks = false;
        s->sliding_window = 0;
        s->has_softcap = false;
        s->use_alibi = true;
        break;
    case 17:
        s->head_size = 64;
        s->block_size = 32;
        s->num_query_heads = 32;
        s->num_kv_heads = 32;
        s->dtype = "fp16";
        s->use_sinks = true;
        s->sliding_window = 512;
        s->has_softcap = true;
        s->use_alibi = true;
        s->use_qq_bias = true;
        break;

    /* --- idx18: num_warps=8 (BLOCK_M=128), no tile_size --- */
    case 18:
        s->head_size = 64;
        s->block_size = 64;
        s->num_query_heads = 64;
        s->num_kv_heads = 8;
        s->dtype = "bf16";
        s->use_sinks = false;
        s->sliding_window = 0;
        s->has_softcap = false;
        s->num_warps = 8;
        break;

    /* --- idx19-26: num_warps / tile_size / waves_per_eu / num_seqs --- */
    case 19:
        s->head_size = 64;
        s->block_size = 32;
        s->num_query_heads = 32;
        s->num_kv_heads = 32;
        s->dtype = "bf16";
        s->use_sinks = false;
        s->sliding_window = 0;
        s->has_softcap = false;
        s->num_warps = 2;
        break;
    case 20:
        s->head_size = 64;
        s->block_size = 32;
        s->num_query_heads = 64;
        s->num_kv_heads = 8;
        s->dtype = "bf16";
        s->use_sinks = false;
        s->sliding_window = 0;
        s->has_softcap = false;
        s->num_warps = 4;
        break;
    case 21:
        s->head_size = 64;
        s->block_size = 32;
        s->num_query_heads = 32;
        s->num_kv_heads = 32;
        s->dtype = "bf16";
        s->use_sinks = false;
        s->sliding_window = 0;
        s->has_softcap = false;
        s->has_tile_size = true;
        s->tile_size = 64;
        break;
    case 22:
        s->head_size = 64;
        s->block_size = 32;
        s->num_query_heads = 32;
        s->num_kv_heads = 32;
        s->dtype = "bf16";
        s->use_sinks = false;
        s->sliding_window = 0;
        s->has_softcap = false;
        s->has_tile_size = true;
        s->tile_size = 128;
        break;
    case 23:
        s->head_size = 128;
        s->block_size = 32;
        s->num_query_heads = 32;
        s->num_kv_heads = 32;
        s->dtype = "fp16";
        s->use_sinks = false;
        s->sliding_window = 0;
        s->has_softcap = false;
        s->num_warps = 2;
        s->has_tile_size = true;
        s->tile_size = 128;
        break;
    case 24:
        s->head_size = 64;
        s->block_size = 32;
        s->num_query_heads = 32;
        s->num_kv_heads = 32;
        s->dtype = "bf16";
        s->use_sinks = false;
        s->sliding_window = 0;
        s->has_softcap = false;
        s->has_waves_per_eu = true;
        s->waves_per_eu = 2;
        break;
    case 25:
        s->head_size = 64;
        s->block_size = 32;
        s->num_query_heads = 32;
        s->num_kv_heads = 32;
        s->dtype = "bf16";
        s->use_sinks = false;
        s->sliding_window = 0;
        s->has_softcap = false;
        s->num_seqs = 1;
        break;
    case 26:
        s->head_size = 64;
        s->block_size = 32;
        s->num_query_heads = 32;
        s->num_kv_heads = 32;
        s->dtype = "bf16";
        s->use_sinks = false;
        s->sliding_window = 0;
        s->has_softcap = false;
        s->num_seqs = 257;
        break;

    /* --- idx27: fp8 KV cache with native fp8 PV MFMA --- */
    case 27:
        s->head_size = 64;
        s->block_size = 32;
        s->num_query_heads = 32;
        s->num_kv_heads = 32;
        s->dtype = "bf16";
        s->use_sinks = false;
        s->sliding_window = 0;
        s->has_softcap = false;
        s->kv_storage_dtype = "fp8e4m3";
        s->use_fp8_mfma_pv = true;
        break;

    /* --- idx28: 64-bit paged-KV addressing --- */
    case 28:
        s->head_size = 64;
        s->block_size = 32;
        s->num_query_heads = 32;
        s->num_kv_heads = 32;
        s->dtype = "bf16";
        s->use_sinks = false;
        s->sliding_window = 0;
        s->has_softcap = false;
        s->use_i64_kv_addr = true;
        break;

    /* --- idx29: register-PV bf16 path --- */
    case 29:
        s->head_size = 64;
        s->block_size = 32;
        s->num_query_heads = 32;
        s->num_kv_heads = 32;
        s->dtype = "bf16";
        s->use_sinks = false;
        s->sliding_window = 0;
        s->has_softcap = false;
        s->use_register_pv = true;
        break;

    /* --- idx30-32: fp8 KV (dequant), fp8 QK MFMA, mfma_32x32 base --- */
    case 30:
        s->head_size = 64;
        s->block_size = 32;
        s->num_query_heads = 32;
        s->num_kv_heads = 32;
        s->dtype = "bf16";
        s->use_sinks = false;
        s->sliding_window = 0;
        s->has_softcap = false;
        s->kv_storage_dtype = "fp8e4m3";
        break;
    case 31:
        s->head_size = 64;
        s->block_size = 32;
        s->num_query_heads = 32;
        s->num_kv_heads = 32;
        s->dtype = "bf16";
        s->use_sinks = false;
        s->sliding_window = 0;
        s->has_softcap = false;
        s->kv_storage_dtype = "fp8e4m3";
        s->use_fp8_mfma_qk = true;
        break;
    case 32:
        s->head_size = 128;
        s->block_size = 32;
        s->num_query_heads = 32;
        s->num_kv_heads = 32;
        s->dtype = "fp16";
        s->use_sinks = false;
        s->sliding_window = 0;
        s->has_softcap = false;
        s->use_mfma_32x32 = true;
        s->block_m_per_warp = 32;
        s->has_tile_size = true;
        s->tile_size = 64;
        break;

    /* --- idx33-35: transposed 32x32 + scalar-state + invariant-hoist +
     *     mask-once + grouped-KV2 softmax stack (bf16, 32-row warp slice) --- */
    case 33:
        s->head_size = 64;
        s->block_size = 32;
        s->num_query_heads = 32;
        s->num_kv_heads = 32;
        s->dtype = "bf16";
        s->use_sinks = false;
        s->sliding_window = 0;
        s->has_softcap = false;
        s->use_mfma_32x32 = true;
        s->use_transposed_qk_32x32 = true;
        s->use_transposed_scalar_state = true;
        s->use_transposed_invariant_hoist = true;
        s->use_transposed_mask_once = true;
        s->use_grouped_kv2_softmax = true;
        s->block_m_per_warp = 32;
        s->has_tile_size = true;
        s->tile_size = 64;
        break;
    case 34:
        s->head_size = 128;
        s->block_size = 32;
        s->num_query_heads = 32;
        s->num_kv_heads = 32;
        s->dtype = "bf16";
        s->use_sinks = false;
        s->sliding_window = 0;
        s->has_softcap = false;
        s->use_mfma_32x32 = true;
        s->use_transposed_qk_32x32 = true;
        s->use_transposed_scalar_state = true;
        s->use_transposed_invariant_hoist = true;
        s->use_transposed_mask_once = true;
        s->use_grouped_kv2_softmax = true;
        s->block_m_per_warp = 32;
        s->has_tile_size = true;
        s->tile_size = 64;
        break;
    case 35:
        s->head_size = 64;
        s->block_size = 32;
        s->num_query_heads = 64;
        s->num_kv_heads = 8;
        s->dtype = "bf16";
        s->use_sinks = false;
        s->sliding_window = 0;
        s->has_softcap = false;
        s->use_mfma_32x32 = true;
        s->use_transposed_qk_32x32 = true;
        s->use_transposed_scalar_state = true;
        s->use_transposed_invariant_hoist = true;
        s->use_transposed_mask_once = true;
        s->use_grouped_kv2_softmax = true;
        s->num_warps = 4;
        s->block_m_per_warp = 32;
        s->has_tile_size = true;
        s->tile_size = 64;
        break;

    /* --- idx36: early-V schedule --- */
    case 36:
        s->head_size = 64;
        s->block_size = 32;
        s->num_query_heads = 32;
        s->num_kv_heads = 32;
        s->dtype = "bf16";
        s->use_sinks = false;
        s->sliding_window = 0;
        s->has_softcap = false;
        s->use_early_v_schedule = true;
        break;

    /* --- idx37: fast paged-KV descriptor (bf16 h64kv8 HD=64 BS=32 T=64 nw=4) --- */
    case 37:
        s->head_size = 64;
        s->block_size = 32;
        s->num_query_heads = 64;
        s->num_kv_heads = 8;
        s->dtype = "bf16";
        s->use_sinks = false;
        s->sliding_window = 0;
        s->has_softcap = false;
        s->num_warps = 4;
        s->block_m_per_warp = 16;
        s->has_tile_size = true;
        s->tile_size = 64;
        s->use_fast_paged_kv_desc = true;
        break;

    /* --- idx38-41: transposed 32x32 + half-local-PV (hlpv) reschedule.
     *     hlpv requires use_transposed_qk_32x32; d64/d128 x fp16/bf16. --- */
    case 38:
        s->head_size = 64;
        s->block_size = 32;
        s->num_query_heads = 32;
        s->num_kv_heads = 32;
        s->dtype = "bf16";
        s->use_sinks = false;
        s->sliding_window = 0;
        s->has_softcap = false;
        s->use_mfma_32x32 = true;
        s->use_transposed_qk_32x32 = true;
        s->use_transposed_half_local_pv = true;
        s->block_m_per_warp = 32;
        s->has_tile_size = true;
        s->tile_size = 64;
        break;
    case 39:
        s->head_size = 128;
        s->block_size = 32;
        s->num_query_heads = 32;
        s->num_kv_heads = 32;
        s->dtype = "bf16";
        s->use_sinks = false;
        s->sliding_window = 0;
        s->has_softcap = false;
        s->use_mfma_32x32 = true;
        s->use_transposed_qk_32x32 = true;
        s->use_transposed_half_local_pv = true;
        s->block_m_per_warp = 32;
        s->has_tile_size = true;
        s->tile_size = 64;
        break;
    case 40:
        s->head_size = 64;
        s->block_size = 32;
        s->num_query_heads = 32;
        s->num_kv_heads = 32;
        s->dtype = "fp16";
        s->use_sinks = false;
        s->sliding_window = 0;
        s->has_softcap = false;
        s->use_mfma_32x32 = true;
        s->use_transposed_qk_32x32 = true;
        s->use_transposed_half_local_pv = true;
        s->block_m_per_warp = 32;
        s->has_tile_size = true;
        s->tile_size = 64;
        break;
    case 41:
        s->head_size = 128;
        s->block_size = 32;
        s->num_query_heads = 32;
        s->num_kv_heads = 32;
        s->dtype = "fp16";
        s->use_sinks = false;
        s->sliding_window = 0;
        s->has_softcap = false;
        s->use_mfma_32x32 = true;
        s->use_transposed_qk_32x32 = true;
        s->use_transposed_half_local_pv = true;
        s->block_m_per_warp = 32;
        s->has_tile_size = true;
        s->tile_size = 64;
        break;

    /* --- idx42: exact provider build_sdpa_tiled hlpv shape (combo multi_batch
     *     GQA-8 h64kv8 bf16: the shape the provider prefill path routes to). --- */
    case 42:
        s->head_size = 64;
        s->block_size = 32;
        s->num_query_heads = 64;
        s->num_kv_heads = 8;
        s->dtype = "bf16";
        s->use_sinks = false;
        s->sliding_window = 0;
        s->has_softcap = false;
        s->num_seqs = 2;
        s->num_warps = 4;
        s->use_mfma_32x32 = true;
        s->use_transposed_qk_32x32 = true;
        s->use_transposed_scalar_state = true;
        s->use_transposed_mask_once = true;
        s->use_transposed_half_local_pv = true;
        s->block_m_per_warp = 32;
        s->has_tile_size = true;
        s->tile_size = 64;
        break;

    /* idx43-49: gfx950 PRODUCTION combo knobs (mirrors the .py). */
    case 43:
        s->head_size = 64;
        s->block_size = 32;
        s->num_query_heads = 64;
        s->num_kv_heads = 8;
        s->dtype = "bf16";
        s->use_sinks = false;
        s->sliding_window = 0;
        s->has_softcap = false;
        s->num_warps = 4;
        s->block_m_per_warp = 32;
        s->has_tile_size = true;
        s->tile_size = 64;
        s->use_mfma_32x32 = true;
        s->use_transposed_qk_32x32 = true;
        s->use_transposed_scalar_state = true;
        s->use_transposed_invariant_hoist = true;
        s->use_transposed_mask_once = true;
        s->use_transposed_mask_limit = true;
        s->use_mfma32_skip_legacy_qreg = true;
        s->use_transposed_half_local_pv = true;
        s->use_fast_paged_kv_desc = true;
        break;
    case 44:
        s->head_size = 128;
        s->block_size = 32;
        s->num_query_heads = 32;
        s->num_kv_heads = 32;
        s->dtype = "bf16";
        s->use_sinks = false;
        s->sliding_window = 0;
        s->has_softcap = false;
        s->num_warps = 4;
        s->block_m_per_warp = 32;
        s->has_tile_size = true;
        s->tile_size = 64;
        s->use_mfma_32x32 = true;
        s->use_transposed_qk_32x32 = true;
        s->use_transposed_scalar_state = true;
        s->use_transposed_invariant_hoist = true;
        s->use_transposed_mask_once = true;
        s->use_transposed_mask_limit = true;
        s->use_mfma32_skip_legacy_qreg = true;
        s->use_transposed_half_local_pv = true;
        break;
    case 45:
        s->head_size = 64;
        s->block_size = 32;
        s->num_query_heads = 32;
        s->num_kv_heads = 32;
        s->dtype = "fp16";
        s->use_sinks = false;
        s->sliding_window = 0;
        s->has_softcap = false;
        s->num_warps = 4;
        s->block_m_per_warp = 32;
        s->has_tile_size = true;
        s->tile_size = 64;
        s->use_mfma_32x32 = true;
        s->use_transposed_qk_32x32 = true;
        s->use_transposed_scalar_state = true;
        s->use_transposed_invariant_hoist = true;
        s->use_transposed_mask_once = true;
        s->use_transposed_mask_limit = true;
        s->use_mfma32_skip_legacy_qreg = true;
        s->use_transposed_half_local_pv = true;
        break;
    case 46:
        s->head_size = 64;
        s->block_size = 32;
        s->num_query_heads = 32;
        s->num_kv_heads = 32;
        s->dtype = "bf16";
        s->use_sinks = false;
        s->sliding_window = 0;
        s->has_softcap = false;
        s->num_warps = 2;
        s->block_m_per_warp = 32;
        s->has_tile_size = true;
        s->tile_size = 64;
        s->use_mfma_32x32 = true;
        s->use_transposed_qk_32x32 = true;
        s->use_transposed_scalar_state = true;
        s->use_transposed_invariant_hoist = true;
        s->use_transposed_mask_once = true;
        s->use_transposed_mask_limit = true;
        s->use_mfma32_skip_legacy_qreg = true;
        break;
    case 47:
        s->head_size = 64;
        s->block_size = 32;
        s->num_query_heads = 64;
        s->num_kv_heads = 8;
        s->dtype = "bf16";
        s->use_sinks = false;
        s->sliding_window = 0;
        s->has_softcap = false;
        s->num_warps = 4;
        s->block_m_per_warp = 32;
        s->has_tile_size = true;
        s->tile_size = 64;
        s->use_mfma_32x32 = true;
        s->use_transposed_qk_32x32 = true;
        s->use_transposed_scalar_state = true;
        s->use_transposed_mask_once = true;
        s->use_transposed_mask_limit = true;
        s->use_transposed_half_local_pv = true;
        s->use_early_v_schedule = true;
        break;
    case 48:
        s->head_size = 64;
        s->block_size = 32;
        s->num_query_heads = 64;
        s->num_kv_heads = 8;
        s->dtype = "bf16";
        s->use_sinks = false;
        s->sliding_window = 0;
        s->has_softcap = false;
        s->num_warps = 4;
        s->block_m_per_warp = 32;
        s->has_tile_size = true;
        s->tile_size = 64;
        s->use_mfma_32x32 = true;
        s->use_transposed_qk_32x32 = true;
        s->use_transposed_scalar_state = true;
        s->use_grouped_kv2_softmax = true;
        break;
    case 49:
        s->head_size = 64;
        s->block_size = 32;
        s->num_query_heads = 64;
        s->num_kv_heads = 8;
        s->dtype = "bf16";
        s->use_sinks = false;
        s->sliding_window = 0;
        s->has_softcap = false;
        s->num_seqs = 1;
        s->num_warps = 4;
        s->block_m_per_warp = 32;
        s->has_tile_size = true;
        s->tile_size = 64;
        s->use_mfma_32x32 = true;
        s->use_transposed_qk_32x32 = true;
        s->use_transposed_scalar_state = true;
        s->use_transposed_invariant_hoist = true;
        s->use_transposed_mask_once = true;
        s->use_transposed_mask_limit = true;
        s->use_mfma32_skip_legacy_qreg = true;
        s->use_transposed_half_local_pv = true;
        s->use_early_v_schedule = true;
        break;
    /* idx50: lever-3 sched_barrier on the single-batch d128 combo (nw=1). */
    case 50:
        s->head_size = 128;
        s->block_size = 32;
        s->num_query_heads = 64;
        s->num_kv_heads = 8;
        s->dtype = "bf16";
        s->use_sinks = false;
        s->sliding_window = 0;
        s->has_softcap = false;
        s->num_seqs = 1;
        s->num_warps = 1;
        s->block_m_per_warp = 32;
        s->has_tile_size = true;
        s->tile_size = 64;
        s->use_mfma_32x32 = true;
        s->use_transposed_qk_32x32 = true;
        s->use_transposed_scalar_state = true;
        s->use_transposed_invariant_hoist = true;
        s->use_transposed_mask_once = true;
        s->use_transposed_mask_limit = true;
        s->use_mfma32_skip_legacy_qreg = true;
        s->use_transposed_half_local_pv = true;
        s->use_sched_barrier = true;
        break;
    /* idx51: #66 small-tile d128 2-WG/CU win path (tile_size=32 on the combo,
     * num_warps=2). Pure existing code path -> must be C/Python byte-identical. */
    case 51:
        s->head_size = 128;
        s->block_size = 32;
        s->num_query_heads = 64;
        s->num_kv_heads = 8;
        s->dtype = "bf16";
        s->use_sinks = false;
        s->sliding_window = 0;
        s->has_softcap = false;
        s->num_seqs = 1;
        s->num_warps = 2;
        s->block_m_per_warp = 32;
        s->has_tile_size = true;
        s->tile_size = 32;
        s->use_mfma_32x32 = true;
        s->use_transposed_qk_32x32 = true;
        s->use_transposed_scalar_state = true;
        s->use_transposed_invariant_hoist = true;
        s->use_transposed_mask_once = true;
        s->use_transposed_mask_limit = true;
        s->use_mfma32_skip_legacy_qreg = true;
        s->use_transposed_half_local_pv = true;
        break;
    /* idx52 REMOVED: it exercised the deep K prefetch ring (kv_ring_depth=3), a
     * Python-only experimental lever the production selector never sets and the
     * gfx950 C twin rejects by design. A config one engine cannot emit is
     * intrinsically asymmetric and does not belong in the byte-identity (emit-
     * comparison) gate. The C reject guard remains in the public entry glue;
     * ring-3 was superseded by use_softmax_mfma_interleave (not production). */
    /* idx52 (renumbered from 53 after the ring-3 config was removed; kept
     * contiguous so the gate does not stop early on a hole): #69 K single-buffer
     * at T=64 -- the d128 long-context 2-WG/CU win. tile_size=64 +
     * use_k_single_buffer (K_lds 2->1 slot, next-K deferred to after the PV-wait
     * barrier). Ported to the gfx950 C twin -> must be C/Python byte-identical. */
    case 52:
        s->head_size = 128;
        s->block_size = 32;
        s->num_query_heads = 64;
        s->num_kv_heads = 8;
        s->dtype = "bf16";
        s->use_sinks = false;
        s->sliding_window = 0;
        s->has_softcap = false;
        s->num_seqs = 1;
        s->num_warps = 2;
        s->block_m_per_warp = 32;
        s->has_tile_size = true;
        s->tile_size = 64;
        s->use_k_single_buffer = true;
        s->use_mfma_32x32 = true;
        s->use_transposed_qk_32x32 = true;
        s->use_transposed_scalar_state = true;
        s->use_transposed_invariant_hoist = true;
        s->use_transposed_mask_once = true;
        s->use_transposed_mask_limit = true;
        s->use_mfma32_skip_legacy_qreg = true;
        s->use_transposed_half_local_pv = true;
        break;

    default:
        return -1;
    }
    return 0;
}

int main(int argc, char** argv)
{
    if(argc < 2)
    {
        fprintf(stderr, "usage: %s <config_index>\n", argv[0]);
        return 2;
    }
    int idx = atoi(argv[1]);
    const char* mode = (argc > 2) ? argv[2] : "ll";

    rocke_attention_tiled_2d_spec_t s;
    if(make_spec(idx, &s) != 0)
    {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 1;
    }

    rocke_ir_builder_t b;
    rocke_kernel_def_t* kernel
        = rocke_gfx950_build_unified_attention_2d_tiled_new(&b, &s, "gfx950");
    if(!kernel)
    {
        fprintf(stderr, "build failed: err=%s\n", b.err);
        rocke_ir_builder_free(&b);
        return 1;
    }

    if(strcmp(mode, "ll") == 0)
    {
        char* llvm_text = NULL;
        rocke_status_t st
            = rocke_lower_kernel_to_llvm(kernel, ROCKE_LLVM_FLAVOR_AUTO, "gfx950", &llvm_text);
        if(st != ROCKE_OK || !llvm_text)
        {
            fprintf(stderr, "lower failed: status=%d\n", (int)st);
            rocke_ir_builder_free(&b);
            return 1;
        }
        fputs(llvm_text, stdout);
        free(llvm_text);
    }
    else if(strcmp(mode, "ir") == 0)
    {
        char* t = NULL;
        rocke_status_t st = rocke_ir_serialize(kernel, &t);
        if(st != ROCKE_OK || !t)
        {
            fprintf(stderr, "serialize failed: status=%d\n", (int)st);
            rocke_ir_builder_free(&b);
            return 1;
        }
        fputs(t, stdout);
        free(t);
    }
    else if(strcmp(mode, "verify") == 0)
    {
        rocke_diag_t* d = NULL;
        size_t n = 0;
        rocke_verify(kernel, &d, &n);
        for(size_t i = 0; i < n; i++)
        {
            char* s2 = rocke_diag_to_string(&d[i]);
            if(s2)
            {
                puts(s2);
                free(s2);
            }
        }
        rocke_diags_free(d, n);
    }
    else
    {
        fprintf(stderr, "unknown mode %s\n", mode);
        rocke_ir_builder_free(&b);
        return 2;
    }

    rocke_ir_builder_free(&b);
    return 0;
}
