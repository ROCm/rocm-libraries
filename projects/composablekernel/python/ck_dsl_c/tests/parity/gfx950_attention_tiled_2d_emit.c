/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/gfx950_attention_tiled_2d_emit.c -- C-side emitter for the gfx950
 * WIDE-ATOM tiled-2D unified-attention parity harness.
 *
 * Selects one of the sampled configs by argv[1], fills a
 * ckc_attention_tiled_2d_spec_t identically to the Python emitter
 * gfx950_attention_tiled_2d_emit.py, builds the kernel via
 * ckc_gfx950_build_unified_attention_2d_tiled_new(&b, &spec, "gfx950"), lowers it
 * with ckc_lower_kernel_to_llvm(kernel, AUTO, "gfx950", ...) and prints the .ll
 * to stdout for byte comparison.
 *
 * The config table is kept IN LOCKSTEP with the Python emitter's _CONFIGS dict
 * (same index -> same UnifiedAttention2DTiledSpec). This is the "edge /
 * feature-flag" cluster: minimal dims, GQA ratios, and every feature-flag path.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ckc/ir.h"
#include "ckc/lower_llvm.h"
#include "ckc/instance_gfx950_attention_tiled_2d.h"

/* Fill `s` for config index `idx`. Returns 0 on success, -1 on unknown idx. */
static int make_spec(int idx, ckc_attention_tiled_2d_spec_t *s) {
    *s = ckc_attention_tiled_2d_spec_default();
    switch (idx) {
    /* --- idx0-4: minimal dims, block_size=16, head_size {64,128,256}, GQA --- */
    case 0:
        s->head_size = 64;  s->block_size = 16;
        s->num_query_heads = 1; s->num_kv_heads = 1;
        s->dtype = "bf16"; s->use_sinks = false;
        s->sliding_window = 0; s->has_softcap = false;
        break;
    case 1:
        s->head_size = 128; s->block_size = 16;
        s->num_query_heads = 1; s->num_kv_heads = 1;
        s->dtype = "bf16"; s->use_sinks = false;
        s->sliding_window = 0; s->has_softcap = false;
        break;
    case 2:
        s->head_size = 256; s->block_size = 16;
        s->num_query_heads = 1; s->num_kv_heads = 1;
        s->dtype = "bf16"; s->use_sinks = false;
        s->sliding_window = 0; s->has_softcap = false;
        break;
    case 3:
        s->head_size = 64;  s->block_size = 16;
        s->num_query_heads = 16; s->num_kv_heads = 1;
        s->dtype = "bf16"; s->use_sinks = false;
        s->sliding_window = 0; s->has_softcap = false;
        break;
    case 4:
        s->head_size = 64;  s->block_size = 16;
        s->num_query_heads = 2; s->num_kv_heads = 1;
        s->dtype = "bf16"; s->use_sinks = false;
        s->sliding_window = 0; s->has_softcap = false;
        break;

    /* --- idx5-14: baseline dtype / mask / head-size / block-size variety --- */
    case 5:
        s->head_size = 64;  s->block_size = 32;
        s->num_query_heads = 32; s->num_kv_heads = 32;
        s->dtype = "bf16"; s->use_sinks = false;
        s->sliding_window = 0; s->has_softcap = false;
        break;
    case 6:
        s->head_size = 64;  s->block_size = 32;
        s->num_query_heads = 32; s->num_kv_heads = 32;
        s->dtype = "fp16"; s->use_sinks = true;
        s->sliding_window = 2048; s->has_softcap = true;
        break;
    case 7:
        s->head_size = 64;  s->block_size = 32;
        s->num_query_heads = 32; s->num_kv_heads = 32;
        s->dtype = "bf16"; s->use_sinks = true;
        s->sliding_window = 1; s->has_softcap = false;
        break;
    case 8:
        s->head_size = 128; s->block_size = 32;
        s->num_query_heads = 32; s->num_kv_heads = 32;
        s->dtype = "fp16"; s->use_sinks = true;
        s->sliding_window = 0; s->has_softcap = true;
        break;
    case 9:
        s->head_size = 256; s->block_size = 32;
        s->num_query_heads = 32; s->num_kv_heads = 32;
        s->dtype = "bf16"; s->use_sinks = false;
        s->sliding_window = 0; s->has_softcap = false;
        break;
    case 10:
        s->head_size = 64;  s->block_size = 64;
        s->num_query_heads = 32; s->num_kv_heads = 32;
        s->dtype = "bf16"; s->use_sinks = false;
        s->sliding_window = 0; s->has_softcap = false;
        break;
    case 11:
        s->head_size = 64;  s->block_size = 32;
        s->num_query_heads = 7; s->num_kv_heads = 7;
        s->dtype = "bf16"; s->use_sinks = false;
        s->sliding_window = 0; s->has_softcap = false;
        break;
    case 12:
        s->head_size = 64;  s->block_size = 32;
        s->num_query_heads = 64; s->num_kv_heads = 8;
        s->dtype = "bf16"; s->use_sinks = false;
        s->sliding_window = 0; s->has_softcap = false;
        break;
    case 13:
        s->head_size = 64;  s->block_size = 32;
        s->num_query_heads = 40; s->num_kv_heads = 8;
        s->dtype = "bf16"; s->use_sinks = false;
        s->sliding_window = 0; s->has_softcap = false;
        break;
    case 14:
        s->head_size = 64;  s->block_size = 32;
        s->num_query_heads = 128; s->num_kv_heads = 1;
        s->dtype = "bf16"; s->use_sinks = false;
        s->sliding_window = 0; s->has_softcap = false;
        break;

    /* --- idx15: QQ-bias feature flag --- */
    case 15:
        s->head_size = 64;  s->block_size = 32;
        s->num_query_heads = 32; s->num_kv_heads = 32;
        s->dtype = "bf16"; s->use_sinks = false;
        s->sliding_window = 0; s->has_softcap = false;
        s->use_qq_bias = true;
        break;

    /* --- idx16,17: ALiBi / composite mask features --- */
    case 16:
        s->head_size = 64;  s->block_size = 32;
        s->num_query_heads = 32; s->num_kv_heads = 32;
        s->dtype = "bf16"; s->use_sinks = false;
        s->sliding_window = 0; s->has_softcap = false;
        s->use_alibi = true;
        break;
    case 17:
        s->head_size = 64;  s->block_size = 32;
        s->num_query_heads = 32; s->num_kv_heads = 32;
        s->dtype = "fp16"; s->use_sinks = true;
        s->sliding_window = 512; s->has_softcap = true;
        s->use_alibi = true; s->use_qq_bias = true;
        break;

    /* --- idx18: num_warps=8 (BLOCK_M=128), no tile_size --- */
    case 18:
        s->head_size = 64;  s->block_size = 64;
        s->num_query_heads = 64; s->num_kv_heads = 8;
        s->dtype = "bf16"; s->use_sinks = false;
        s->sliding_window = 0; s->has_softcap = false;
        s->num_warps = 8;
        break;

    /* --- idx19-26: num_warps / tile_size / waves_per_eu / num_seqs --- */
    case 19:
        s->head_size = 64;  s->block_size = 32;
        s->num_query_heads = 32; s->num_kv_heads = 32;
        s->dtype = "bf16"; s->use_sinks = false;
        s->sliding_window = 0; s->has_softcap = false;
        s->num_warps = 2;
        break;
    case 20:
        s->head_size = 64;  s->block_size = 32;
        s->num_query_heads = 64; s->num_kv_heads = 8;
        s->dtype = "bf16"; s->use_sinks = false;
        s->sliding_window = 0; s->has_softcap = false;
        s->num_warps = 4;
        break;
    case 21:
        s->head_size = 64;  s->block_size = 32;
        s->num_query_heads = 32; s->num_kv_heads = 32;
        s->dtype = "bf16"; s->use_sinks = false;
        s->sliding_window = 0; s->has_softcap = false;
        s->has_tile_size = true; s->tile_size = 64;
        break;
    case 22:
        s->head_size = 64;  s->block_size = 32;
        s->num_query_heads = 32; s->num_kv_heads = 32;
        s->dtype = "bf16"; s->use_sinks = false;
        s->sliding_window = 0; s->has_softcap = false;
        s->has_tile_size = true; s->tile_size = 128;
        break;
    case 23:
        s->head_size = 128; s->block_size = 32;
        s->num_query_heads = 32; s->num_kv_heads = 32;
        s->dtype = "fp16"; s->use_sinks = false;
        s->sliding_window = 0; s->has_softcap = false;
        s->num_warps = 2;
        s->has_tile_size = true; s->tile_size = 128;
        break;
    case 24:
        s->head_size = 64;  s->block_size = 32;
        s->num_query_heads = 32; s->num_kv_heads = 32;
        s->dtype = "bf16"; s->use_sinks = false;
        s->sliding_window = 0; s->has_softcap = false;
        s->has_waves_per_eu = true; s->waves_per_eu = 2;
        break;
    case 25:
        s->head_size = 64;  s->block_size = 32;
        s->num_query_heads = 32; s->num_kv_heads = 32;
        s->dtype = "bf16"; s->use_sinks = false;
        s->sliding_window = 0; s->has_softcap = false;
        s->num_seqs = 1;
        break;
    case 26:
        s->head_size = 64;  s->block_size = 32;
        s->num_query_heads = 32; s->num_kv_heads = 32;
        s->dtype = "bf16"; s->use_sinks = false;
        s->sliding_window = 0; s->has_softcap = false;
        s->num_seqs = 257;
        break;

    /* --- idx27: fp8 KV cache with native fp8 PV MFMA --- */
    case 27:
        s->head_size = 64;  s->block_size = 32;
        s->num_query_heads = 32; s->num_kv_heads = 32;
        s->dtype = "bf16"; s->use_sinks = false;
        s->sliding_window = 0; s->has_softcap = false;
        s->kv_storage_dtype = "fp8e4m3";
        s->use_fp8_mfma_pv = true;
        break;

    /* --- idx28: 64-bit paged-KV addressing --- */
    case 28:
        s->head_size = 64;  s->block_size = 32;
        s->num_query_heads = 32; s->num_kv_heads = 32;
        s->dtype = "bf16"; s->use_sinks = false;
        s->sliding_window = 0; s->has_softcap = false;
        s->use_i64_kv_addr = true;
        break;

    /* --- idx29: register-PV bf16 path --- */
    case 29:
        s->head_size = 64;  s->block_size = 32;
        s->num_query_heads = 32; s->num_kv_heads = 32;
        s->dtype = "bf16"; s->use_sinks = false;
        s->sliding_window = 0; s->has_softcap = false;
        s->use_register_pv = true;
        break;

    /* --- idx30-32: fp8 KV (dequant), fp8 QK MFMA, mfma_32x32 base --- */
    case 30:
        s->head_size = 64;  s->block_size = 32;
        s->num_query_heads = 32; s->num_kv_heads = 32;
        s->dtype = "bf16"; s->use_sinks = false;
        s->sliding_window = 0; s->has_softcap = false;
        s->kv_storage_dtype = "fp8e4m3";
        break;
    case 31:
        s->head_size = 64;  s->block_size = 32;
        s->num_query_heads = 32; s->num_kv_heads = 32;
        s->dtype = "bf16"; s->use_sinks = false;
        s->sliding_window = 0; s->has_softcap = false;
        s->kv_storage_dtype = "fp8e4m3";
        s->use_fp8_mfma_qk = true;
        break;
    case 32:
        s->head_size = 128; s->block_size = 32;
        s->num_query_heads = 32; s->num_kv_heads = 32;
        s->dtype = "fp16"; s->use_sinks = false;
        s->sliding_window = 0; s->has_softcap = false;
        s->use_mfma_32x32 = true;
        s->block_m_per_warp = 32;
        s->has_tile_size = true; s->tile_size = 64;
        break;

    /* --- idx33-35: transposed 32x32 + scalar-state + invariant-hoist +
     *     mask-once + grouped-KV2 softmax stack (bf16, 32-row warp slice) --- */
    case 33:
        s->head_size = 64;  s->block_size = 32;
        s->num_query_heads = 32; s->num_kv_heads = 32;
        s->dtype = "bf16"; s->use_sinks = false;
        s->sliding_window = 0; s->has_softcap = false;
        s->use_mfma_32x32 = true;
        s->use_transposed_qk_32x32 = true;
        s->use_transposed_scalar_state = true;
        s->use_transposed_invariant_hoist = true;
        s->use_transposed_mask_once = true;
        s->use_grouped_kv2_softmax = true;
        s->block_m_per_warp = 32;
        s->has_tile_size = true; s->tile_size = 64;
        break;
    case 34:
        s->head_size = 128; s->block_size = 32;
        s->num_query_heads = 32; s->num_kv_heads = 32;
        s->dtype = "bf16"; s->use_sinks = false;
        s->sliding_window = 0; s->has_softcap = false;
        s->use_mfma_32x32 = true;
        s->use_transposed_qk_32x32 = true;
        s->use_transposed_scalar_state = true;
        s->use_transposed_invariant_hoist = true;
        s->use_transposed_mask_once = true;
        s->use_grouped_kv2_softmax = true;
        s->block_m_per_warp = 32;
        s->has_tile_size = true; s->tile_size = 64;
        break;
    case 35:
        s->head_size = 64;  s->block_size = 32;
        s->num_query_heads = 64; s->num_kv_heads = 8;
        s->dtype = "bf16"; s->use_sinks = false;
        s->sliding_window = 0; s->has_softcap = false;
        s->use_mfma_32x32 = true;
        s->use_transposed_qk_32x32 = true;
        s->use_transposed_scalar_state = true;
        s->use_transposed_invariant_hoist = true;
        s->use_transposed_mask_once = true;
        s->use_grouped_kv2_softmax = true;
        s->num_warps = 4;
        s->block_m_per_warp = 32;
        s->has_tile_size = true; s->tile_size = 64;
        break;

    /* --- idx36: early-V schedule --- */
    case 36:
        s->head_size = 64;  s->block_size = 32;
        s->num_query_heads = 32; s->num_kv_heads = 32;
        s->dtype = "bf16"; s->use_sinks = false;
        s->sliding_window = 0; s->has_softcap = false;
        s->use_early_v_schedule = true;
        break;

    /* --- idx37: fast paged-KV descriptor (bf16 h64kv8 HD=64 BS=32 T=64 nw=4) --- */
    case 37:
        s->head_size = 64;  s->block_size = 32;
        s->num_query_heads = 64; s->num_kv_heads = 8;
        s->dtype = "bf16"; s->use_sinks = false;
        s->sliding_window = 0; s->has_softcap = false;
        s->num_warps = 4;
        s->block_m_per_warp = 16;
        s->has_tile_size = true; s->tile_size = 64;
        s->use_fast_paged_kv_desc = true;
        break;

    default:
        return -1;
    }
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <config_index>\n", argv[0]);
        return 2;
    }
    int idx = atoi(argv[1]);

    ckc_attention_tiled_2d_spec_t s;
    if (make_spec(idx, &s) != 0) {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 1;
    }

    ckc_ir_builder_t b;
    ckc_kernel_def_t *kernel =
        ckc_gfx950_build_unified_attention_2d_tiled_new(&b, &s, "gfx950");
    if (!kernel) {
        fprintf(stderr, "build failed: err=%s\n", b.err);
        ckc_ir_builder_free(&b);
        return 1;
    }

    char *llvm_text = NULL;
    ckc_status_t st = ckc_lower_kernel_to_llvm(
        kernel, CKC_LLVM_FLAVOR_AUTO, "gfx950", &llvm_text);
    if (st != CKC_OK || !llvm_text) {
        fprintf(stderr, "lower failed: status=%d\n", (int)st);
        ckc_ir_builder_free(&b);
        return 1;
    }
    fputs(llvm_text, stdout);
    free(llvm_text);
    ckc_ir_builder_free(&b);
    return 0;
}
