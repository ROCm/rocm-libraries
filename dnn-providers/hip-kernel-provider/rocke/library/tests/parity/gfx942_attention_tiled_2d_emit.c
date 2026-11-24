/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/gfx942_attention_tiled_2d_emit.c -- C-side emitter for the
 * gfx942 narrow-atom tiled-2D unified-attention parity harness.
 *
 * Selects one of the sampled configs by argv[1], fills a
 * rocke_attention_tiled_2d_spec_t identically to the Python emitter
 * gfx942_attention_tiled_2d_emit.py, builds the kernel via
 * rocke_build_unified_attention_2d_tiled_scalar(&b, &spec, "gfx942"), lowers it
 * with rocke_lower_kernel_to_llvm(kernel, AUTO, "gfx942", ...) and prints the .ll
 * to stdout for byte comparison.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/instance_gfx942_attention_tiled_2d.h"
#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"
#include "rocke/verify.h"

static const char* const alibi_design_order[4]
    = {"seed_tile", "fma_tile", "seed_split", "fma_split"};

/* Fill `s` for config index `idx`. Returns 0 on success, -1 on unknown idx. */
static int make_spec(int idx, rocke_attention_tiled_2d_spec_t* s)
{
    *s = rocke_attention_tiled_2d_spec_default();
    switch(idx)
    {
    case 0:
        s->head_size = 64;
        s->block_size = 32;
        s->num_query_heads = 32;
        s->num_kv_heads = 32;
        s->dtype = "bf16";
        s->use_sinks = true;
        s->sliding_window = 2048;
        s->has_softcap = false;
        break;
    case 1:
        s->head_size = 128;
        s->block_size = 32;
        s->num_query_heads = 32;
        s->num_kv_heads = 32;
        s->dtype = "fp16";
        s->use_sinks = true;
        s->sliding_window = 0;
        s->has_softcap = true;
        break;
    case 2:
        s->head_size = 64;
        s->block_size = 128;
        s->num_query_heads = 16;
        s->num_kv_heads = 4;
        s->dtype = "bf16";
        s->use_sinks = false;
        s->sliding_window = 4096;
        s->has_softcap = false;
        break;
    case 3:
        s->head_size = 128;
        s->block_size = 64;
        s->num_query_heads = 32;
        s->num_kv_heads = 8;
        s->dtype = "fp16";
        s->use_sinks = true;
        s->sliding_window = 2048;
        s->has_softcap = true;
        break;
    case 4:
        s->head_size = 256;
        s->block_size = 32;
        s->num_query_heads = 32;
        s->num_kv_heads = 32;
        s->dtype = "bf16";
        s->use_sinks = false;
        s->sliding_window = 0;
        s->has_softcap = false;
        break;
    case 5: /* gfx942 bf16 D128 small-tile double-K (landed prefill lever). */
        s->head_size = 128;
        s->block_size = 32;
        s->num_query_heads = 8;
        s->num_kv_heads = 1;
        s->dtype = "bf16";
        s->use_sinks = false;
        s->sliding_window = 0;
        s->has_softcap = false;
        s->num_warps = 2;
        s->block_m_per_warp = 32;
        s->has_tile_size = true;
        s->tile_size = 32;
        s->use_mfma_32x32x8 = true;
        s->use_transposed_qk_32x32 = true;
        s->use_k_single_buffer = false;
        break;
    case 6: /* cfg5 + use_agpr_alloc_zero (ck83 Fix-A residency lever). */
        s->head_size = 128;
        s->block_size = 32;
        s->num_query_heads = 8;
        s->num_kv_heads = 1;
        s->dtype = "bf16";
        s->use_sinks = false;
        s->sliding_window = 0;
        s->has_softcap = false;
        s->num_warps = 2;
        s->block_m_per_warp = 32;
        s->has_tile_size = true;
        s->tile_size = 32;
        s->use_mfma_32x32x8 = true;
        s->use_transposed_qk_32x32 = true;
        s->use_k_single_buffer = false;
        s->use_agpr_alloc_zero = true;
        break;
    case 7: /* Selector spec for bf16 D128 decode with i64 K/V addressing. */
        s->head_size = 128;
        s->block_size = 16;
        s->num_query_heads = 32;
        s->num_kv_heads = 8;
        s->dtype = "bf16";
        s->use_sinks = false;
        s->sliding_window = 0;
        s->has_softcap = false;
        s->num_seqs = 256;
        s->has_waves_per_eu = true;
        s->waves_per_eu = 2;
        s->use_register_pv = true;
        s->has_tile_size = true;
        s->tile_size = 64;
        s->use_i64_kv_addr = true;
        break;
    case 8: /* i64 K/V, four warps, two pages per tile. */
        s->head_size = 128;
        s->block_size = 32;
        s->num_query_heads = 32;
        s->num_kv_heads = 8;
        s->dtype = "bf16";
        s->use_sinks = false;
        s->sliding_window = 0;
        s->has_softcap = false;
        s->num_seqs = 256;
        s->num_warps = 4;
        s->has_waves_per_eu = true;
        s->waves_per_eu = 2;
        s->use_register_pv = true;
        s->has_tile_size = true;
        s->tile_size = 64;
        s->use_i64_kv_addr = true;
        break;
    case 9: /* i64 K/V, one page per tile. */
        s->head_size = 128;
        s->block_size = 64;
        s->num_query_heads = 32;
        s->num_kv_heads = 8;
        s->dtype = "bf16";
        s->use_sinks = false;
        s->sliding_window = 0;
        s->has_softcap = false;
        s->num_seqs = 256;
        s->has_waves_per_eu = true;
        s->waves_per_eu = 2;
        s->use_register_pv = true;
        s->has_tile_size = true;
        s->tile_size = 64;
        s->use_i64_kv_addr = true;
        break;
    case 10: /* cfg7 with K copied by global_load_lds. */
        s->head_size = 128;
        s->block_size = 16;
        s->num_query_heads = 32;
        s->num_kv_heads = 8;
        s->dtype = "bf16";
        s->use_sinks = false;
        s->sliding_window = 0;
        s->has_softcap = false;
        s->num_seqs = 256;
        s->has_waves_per_eu = true;
        s->waves_per_eu = 2;
        s->use_register_pv = true;
        s->has_tile_size = true;
        s->tile_size = 64;
        s->use_i64_kv_addr = true;
        s->use_global_load_lds_k = true;
        break;
    /* head_size=192 MHA prefill selector spec */
    case 11:
        s->head_size = 192;
        s->block_size = 16;
        s->num_query_heads = 96;
        s->num_kv_heads = 96;
        s->dtype = "bf16";
        s->use_sinks = false;
        s->sliding_window = 0;
        s->has_softcap = false;
        s->num_seqs = 1;
        s->num_warps = 2;
        s->has_waves_per_eu = true;
        s->waves_per_eu = 2;
        s->use_register_pv = true;
        s->has_tile_size = true;
        s->tile_size = 32;
        break;
    /* head_size=192 MHA prefill selector spec (wide flash geometry) */
    case 12:
        s->head_size = 192;
        s->block_size = 16;
        s->num_query_heads = 96;
        s->num_kv_heads = 96;
        s->dtype = "bf16";
        s->use_sinks = false;
        s->sliding_window = 0;
        s->has_softcap = false;
        s->num_seqs = 1;
        s->num_warps = 2;
        s->has_waves_per_eu = true;
        s->waves_per_eu = 2;
        s->has_tile_size = true;
        s->tile_size = 64;
        s->block_m_per_warp = 32;
        s->use_mfma_32x32x8 = true;
        s->use_transposed_qk_32x32 = true;
        s->use_k_single_buffer = true;
        s->kv_cache_policy = "all";
        break;
    case 13: /* cfg5 + emit_lse (transposed-x8, 16 slots, slot 0 stored). */
        s->head_size = 128;
        s->block_size = 32;
        s->num_query_heads = 8;
        s->num_kv_heads = 1;
        s->dtype = "bf16";
        s->use_sinks = false;
        s->sliding_window = 0;
        s->has_softcap = false;
        s->num_warps = 2;
        s->block_m_per_warp = 32;
        s->has_tile_size = true;
        s->tile_size = 32;
        s->use_mfma_32x32x8 = true;
        s->use_transposed_qk_32x32 = true;
        s->use_k_single_buffer = false;
        s->emit_lse = true;
        break;
    case 14: /* emit_lse, transposed-x8 one-slot state, NQK=6. */
        s->head_size = 128;
        s->block_size = 32;
        s->num_query_heads = 12;
        s->num_kv_heads = 2;
        s->dtype = "bf16";
        s->use_sinks = false;
        s->sliding_window = 0;
        s->has_softcap = false;
        s->num_warps = 2;
        s->block_m_per_warp = 32;
        s->has_tile_size = true;
        s->tile_size = 32;
        s->use_mfma_32x32x8 = true;
        s->use_transposed_qk_32x32 = true;
        s->use_transposed_scalar_state = true;
        s->use_k_single_buffer = false;
        s->emit_lse = true;
        break;
    case 15: /* emit_lse, non-transposed x8 Acc_lds epilogue, NQK=6. */
        s->head_size = 128;
        s->block_size = 32;
        s->num_query_heads = 12;
        s->num_kv_heads = 2;
        s->dtype = "bf16";
        s->use_sinks = false;
        s->sliding_window = 0;
        s->has_softcap = false;
        s->num_warps = 2;
        s->block_m_per_warp = 32;
        s->has_tile_size = true;
        s->tile_size = 32;
        s->use_mfma_32x32x8 = true;
        s->emit_lse = true;
        break;
    case 16: /* emit_lse, striped16 epilogue, NQK=6. */
        s->head_size = 128;
        s->block_size = 32;
        s->num_query_heads = 12;
        s->num_kv_heads = 2;
        s->dtype = "bf16";
        s->use_sinks = false;
        s->sliding_window = 0;
        s->has_softcap = false;
        s->emit_lse = true;
        break;
    case 17: /* emit_lse, striped16, two atoms per warp, q-major grid. */
        s->head_size = 128;
        s->block_size = 32;
        s->num_query_heads = 32;
        s->num_kv_heads = 8;
        s->dtype = "bf16";
        s->use_sinks = false;
        s->sliding_window = 0;
        s->has_softcap = false;
        s->block_m_per_warp = 32;
        s->use_q_major_grid = true;
        s->emit_lse = true;
        break;
    case 18: /* cfvst on the bf16 D64 GQA 32/8 prefill geometry, i32 K/V. */
    case 19: /* cfg18 with i64 K/V addressing. */
        s->head_size = 64;
        s->block_size = 32;
        s->num_query_heads = 32;
        s->num_kv_heads = 8;
        s->dtype = "bf16";
        s->use_sinks = false;
        s->sliding_window = 0;
        s->has_softcap = false;
        s->num_warps = 4;
        s->block_m_per_warp = 32;
        s->has_tile_size = true;
        s->tile_size = 64;
        s->use_mfma_32x32x8 = true;
        s->use_transposed_qk_32x32 = true;
        s->use_conflict_free_v_store = true;
        s->use_i64_kv_addr = (idx == 19);
        break;
    case 20:
        /* Additive bias on gfx942; both engines reject it (BOTH_REJECTED). */
        s->head_size = 64;
        s->block_size = 32;
        s->num_query_heads = 32;
        s->num_kv_heads = 32;
        s->dtype = "bf16";
        s->use_sinks = true;
        s->sliding_window = 2048;
        s->has_softcap = false;
        s->use_additive_bias = true;
        break;
    case 21: /* ALiBi, per-row hoist, GQA with 24 heads. */
        s->head_size = 64;
        s->block_size = 32;
        s->num_query_heads = 24;
        s->num_kv_heads = 6;
        s->dtype = "fp16";
        s->use_sinks = false;
        s->sliding_window = 0;
        s->has_softcap = false;
        s->use_alibi = true;
        break;
    case 22: /* ALiBi, transposed x8 per-iter slope (mask-once). */
        s->head_size = 64;
        s->block_size = 32;
        s->num_query_heads = 32;
        s->num_kv_heads = 32;
        s->dtype = "fp16";
        s->use_sinks = false;
        s->sliding_window = 0;
        s->has_softcap = false;
        s->use_alibi = true;
        s->num_warps = 2;
        s->block_m_per_warp = 32;
        s->has_tile_size = true;
        s->tile_size = 64;
        s->use_mfma_32x32x8 = true;
        s->use_transposed_qk_32x32 = true;
        s->use_transposed_mask_once = true;
        break;
    /* cfg23-26: each ALiBi arm on the x8 k1buf mask-limit softmax (bf16 D128);
     * cfg27: x8 cfvst fp16 D128 default arm; cfg34: the k1buf default + LSE. */
    case 23:
    case 24:
    case 25:
    case 26:
    case 27:
    case 34:
        s->head_size = 128;
        s->block_size = 64;
        s->num_query_heads = 32;
        s->num_kv_heads = 8;
        s->dtype = idx == 27 ? "fp16" : "bf16";
        s->use_sinks = false;
        s->sliding_window = 0;
        s->has_softcap = false;
        s->use_alibi = true;
        s->num_seqs = 1;
        s->num_warps = idx == 27 ? 4 : 2;
        s->has_waves_per_eu = true;
        s->waves_per_eu = 2;
        s->has_tile_size = true;
        s->tile_size = 64;
        s->block_m_per_warp = 32;
        s->use_mfma_32x32x8 = true;
        s->use_transposed_qk_32x32 = true;
        s->use_transposed_scalar_state = true;
        s->use_transposed_invariant_hoist = true;
        s->use_transposed_mask_once = true;
        s->use_transposed_mask_limit = true;
        s->use_k_single_buffer = idx != 27;
        s->use_conflict_free_v_store = idx == 27;
        s->kv_cache_policy = "all";
        s->emit_lse = idx == 34;
        if(idx <= 26)
            s->alibi_design = alibi_design_order[idx - 23];
        break;
    /* cfg28: register-P D128 default arm; cfg30-33: each arm on the 16x16 path
     * (fp16 D128). */
    case 28:
    case 30:
    case 31:
    case 32:
    case 33:
        s->head_size = 128;
        s->block_size = 64;
        s->num_query_heads = 64;
        s->num_kv_heads = 8;
        s->dtype = idx == 28 ? "bf16" : "fp16";
        s->use_sinks = false;
        s->sliding_window = 0;
        s->has_softcap = false;
        s->use_alibi = true;
        s->num_seqs = 1;
        s->num_warps = 2;
        s->has_waves_per_eu = true;
        s->waves_per_eu = 2;
        s->has_tile_size = true;
        s->tile_size = 64;
        s->use_register_pv = idx == 28;
        if(idx >= 30)
            s->alibi_design = alibi_design_order[idx - 30];
        break;
    case 29: /* register-P D256 mw32, seed_tile. */
        s->head_size = 256;
        s->block_size = 16;
        s->num_query_heads = 16;
        s->num_kv_heads = 2;
        s->dtype = "bf16";
        s->use_sinks = false;
        s->sliding_window = 0;
        s->has_softcap = false;
        s->use_alibi = true;
        s->num_seqs = 1;
        s->has_waves_per_eu = true;
        s->waves_per_eu = 2;
        s->use_register_pv = true;
        s->has_tile_size = true;
        s->tile_size = 32;
        s->block_m_per_warp = 32;
        s->alibi_design = "seed_tile";
        break;
    /* cfg36-37: the x8 k1buf spec dispatch picks for bf16 D192 at block_size 64
     * (full-mask transposed softmax, inline per-lane slope): 36 the per-score
     * add, 37 fma_tile. */
    case 36:
    case 37:
        s->head_size = 192;
        s->block_size = 64;
        s->num_query_heads = 32;
        s->num_kv_heads = 8;
        s->dtype = "bf16";
        s->use_sinks = false;
        s->sliding_window = 0;
        s->has_softcap = false;
        s->use_alibi = true;
        s->num_seqs = 1;
        s->num_warps = 2;
        s->has_waves_per_eu = true;
        s->waves_per_eu = 2;
        s->has_tile_size = true;
        s->tile_size = 64;
        s->block_m_per_warp = 32;
        s->use_mfma_32x32x8 = true;
        s->use_transposed_qk_32x32 = true;
        s->use_k_single_buffer = true;
        s->kv_cache_policy = "all";
        if(idx == 37)
            s->alibi_design = "fma_tile";
        break;
    case 35: /* cfg22 (x8 mask-once, no mask limit) with seed_split. */
        make_spec(22, s);
        s->alibi_design = "seed_split";
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

    rocke_attention_tiled_2d_spec_t s;
    if(make_spec(idx, &s) != 0)
    {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 1;
    }

    rocke_ir_builder_t b;
    rocke_kernel_def_t* kernel
        = rocke_build_unified_attention_2d_tiled_scalar_new(&b, &s, "gfx942");
    if(!kernel)
    {
        fprintf(stderr, "build failed: err=%s\n", b.err);
        rocke_ir_builder_free(&b);
        return 1;
    }

    const char* mode = (argc > 2) ? argv[2] : "ll";

    if(strcmp(mode, "ll") == 0)
    {
        char* llvm_text = NULL;
        rocke_status_t st
            = rocke_lower_kernel_to_llvm(kernel, ROCKE_LLVM_FLAVOR_AUTO, "gfx942", &llvm_text);
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
            fprintf(stderr, "ir_serialize failed: status=%d\n", (int)st);
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
            char* s = rocke_diag_to_string(&d[i]);
            if(s)
            {
                puts(s);
                free(s);
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
