/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * STRESS variant of gfx942_attention_tiled_2d_emit.c -- wide adversarial config
 * set mirroring gfx942_attention_tiled_2d_stress_emit.py for deep parity.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ckc/ir.h"
#include "ckc/lower_llvm.h"
#include "ckc/instance_gfx942_attention_tiled_2d.h"

static int make_spec(int idx, ckc_attention_tiled_2d_spec_t *s) {
    *s = ckc_attention_tiled_2d_spec_default();
    switch (idx) {
    case 0:
        s->head_size=64; s->block_size=32; s->num_query_heads=32; s->num_kv_heads=32;
        s->dtype="bf16"; s->use_sinks=false; s->sliding_window=0; s->has_softcap=false;
        break;
    case 1:
        s->head_size=64; s->block_size=32; s->num_query_heads=32; s->num_kv_heads=32;
        s->dtype="fp16"; s->use_sinks=true; s->sliding_window=2048; s->has_softcap=true;
        break;
    case 2:
        s->head_size=64; s->block_size=32; s->num_query_heads=32; s->num_kv_heads=32;
        s->dtype="bf16"; s->use_sinks=true; s->sliding_window=1; s->has_softcap=false;
        break;
    case 3:
        s->head_size=64; s->block_size=32; s->num_query_heads=32; s->num_kv_heads=32;
        s->dtype="bf16"; s->use_sinks=false; s->sliding_window=131072; s->has_softcap=false;
        break;
    case 4:
        s->head_size=64; s->block_size=32; s->num_query_heads=32; s->num_kv_heads=32;
        s->dtype="fp16"; s->use_sinks=true; s->sliding_window=512; s->has_softcap=true;
        s->use_alibi=true; s->use_qq_bias=true;
        break;
    case 5:
        s->head_size=32; s->block_size=32; s->num_query_heads=32; s->num_kv_heads=32;
        s->dtype="bf16"; s->use_sinks=false; s->sliding_window=0; s->has_softcap=false;
        break;
    case 6:
        s->head_size=96; s->block_size=32; s->num_query_heads=32; s->num_kv_heads=32;
        s->dtype="bf16"; s->use_sinks=false; s->sliding_window=0; s->has_softcap=false;
        break;
    case 7:
        s->head_size=128; s->block_size=32; s->num_query_heads=32; s->num_kv_heads=32;
        s->dtype="fp16"; s->use_sinks=true; s->sliding_window=0; s->has_softcap=true;
        break;
    case 8:
        s->head_size=192; s->block_size=32; s->num_query_heads=32; s->num_kv_heads=32;
        s->dtype="bf16"; s->use_sinks=false; s->sliding_window=0; s->has_softcap=false;
        break;
    case 9:
        s->head_size=256; s->block_size=32; s->num_query_heads=32; s->num_kv_heads=32;
        s->dtype="bf16"; s->use_sinks=false; s->sliding_window=0; s->has_softcap=false;
        break;
    case 10:
        s->head_size=512; s->block_size=32; s->num_query_heads=32; s->num_kv_heads=32;
        s->dtype="bf16"; s->use_sinks=false; s->sliding_window=0; s->has_softcap=false;
        break;
    case 11:
        s->head_size=64; s->block_size=16; s->num_query_heads=32; s->num_kv_heads=32;
        s->dtype="bf16"; s->use_sinks=false; s->sliding_window=0; s->has_softcap=false;
        break;
    case 12:
        s->head_size=64; s->block_size=64; s->num_query_heads=32; s->num_kv_heads=32;
        s->dtype="bf16"; s->use_sinks=false; s->sliding_window=0; s->has_softcap=false;
        break;
    case 13:
        s->head_size=64; s->block_size=96; s->num_query_heads=32; s->num_kv_heads=32;
        s->dtype="bf16"; s->use_sinks=false; s->sliding_window=0; s->has_softcap=false;
        break;
    case 14:
        s->head_size=64; s->block_size=160; s->num_query_heads=32; s->num_kv_heads=32;
        s->dtype="bf16"; s->use_sinks=false; s->sliding_window=0; s->has_softcap=false;
        break;
    case 15:
        s->head_size=64; s->block_size=256; s->num_query_heads=32; s->num_kv_heads=32;
        s->dtype="bf16"; s->use_sinks=false; s->sliding_window=0; s->has_softcap=false;
        break;
    case 16:
        s->head_size=64; s->block_size=32; s->num_query_heads=1; s->num_kv_heads=1;
        s->dtype="bf16"; s->use_sinks=false; s->sliding_window=0; s->has_softcap=false;
        break;
    case 17:
        s->head_size=64; s->block_size=32; s->num_query_heads=7; s->num_kv_heads=7;
        s->dtype="bf16"; s->use_sinks=false; s->sliding_window=0; s->has_softcap=false;
        break;
    case 18:
        s->head_size=64; s->block_size=32; s->num_query_heads=13; s->num_kv_heads=13;
        s->dtype="bf16"; s->use_sinks=false; s->sliding_window=0; s->has_softcap=false;
        break;
    case 19:
        s->head_size=64; s->block_size=32; s->num_query_heads=64; s->num_kv_heads=8;
        s->dtype="bf16"; s->use_sinks=false; s->sliding_window=0; s->has_softcap=false;
        break;
    case 20:
        s->head_size=64; s->block_size=32; s->num_query_heads=40; s->num_kv_heads=8;
        s->dtype="bf16"; s->use_sinks=false; s->sliding_window=0; s->has_softcap=false;
        break;
    case 21:
        s->head_size=64; s->block_size=32; s->num_query_heads=32; s->num_kv_heads=5;
        s->dtype="bf16"; s->use_sinks=false; s->sliding_window=0; s->has_softcap=false;
        break;
    case 22:
        s->head_size=64; s->block_size=32; s->num_query_heads=128; s->num_kv_heads=1;
        s->dtype="bf16"; s->use_sinks=false; s->sliding_window=0; s->has_softcap=false;
        break;
    case 23:
        s->head_size=64; s->block_size=32; s->num_query_heads=32; s->num_kv_heads=32;
        s->dtype="bf16"; s->use_sinks=false; s->sliding_window=0; s->has_softcap=false;
        s->num_warps=2;
        break;
    case 24:
        s->head_size=64; s->block_size=32; s->num_query_heads=64; s->num_kv_heads=8;
        s->dtype="bf16"; s->use_sinks=false; s->sliding_window=0; s->has_softcap=false;
        s->num_warps=4;
        break;
    case 25:
        s->head_size=64; s->block_size=32; s->num_query_heads=32; s->num_kv_heads=32;
        s->dtype="bf16"; s->use_sinks=false; s->sliding_window=0; s->has_softcap=false;
        s->has_tile_size=true; s->tile_size=64;
        break;
    case 26:
        s->head_size=64; s->block_size=32; s->num_query_heads=32; s->num_kv_heads=32;
        s->dtype="bf16"; s->use_sinks=false; s->sliding_window=0; s->has_softcap=false;
        s->has_tile_size=true; s->tile_size=128;
        break;
    case 27:
        s->head_size=128; s->block_size=32; s->num_query_heads=32; s->num_kv_heads=32;
        s->dtype="fp16"; s->use_sinks=false; s->sliding_window=0; s->has_softcap=false;
        s->num_warps=2; s->has_tile_size=true; s->tile_size=128;
        break;
    case 28:
        s->head_size=64; s->block_size=32; s->num_query_heads=32; s->num_kv_heads=32;
        s->dtype="bf16"; s->use_sinks=false; s->sliding_window=0; s->has_softcap=false;
        s->kv_cache_policy="all";
        break;
    case 29:
        s->head_size=64; s->block_size=32; s->num_query_heads=32; s->num_kv_heads=32;
        s->dtype="bf16"; s->use_sinks=false; s->sliding_window=0; s->has_softcap=false;
        s->kv_cache_policy="global";
        break;
    case 30:
        s->head_size=64; s->block_size=32; s->num_query_heads=32; s->num_kv_heads=32;
        s->dtype="bf16"; s->use_sinks=false; s->sliding_window=0; s->has_softcap=false;
        s->kv_cache_policy="nt";
        break;
    case 31:
        s->head_size=64; s->block_size=32; s->num_query_heads=32; s->num_kv_heads=32;
        s->dtype="bf16"; s->use_sinks=false; s->sliding_window=0; s->has_softcap=false;
        s->has_waves_per_eu=true; s->waves_per_eu=2;
        break;
    case 32:
        s->head_size=64; s->block_size=32; s->num_query_heads=32; s->num_kv_heads=32;
        s->dtype="bf16"; s->use_sinks=false; s->sliding_window=0; s->has_softcap=false;
        s->num_seqs=1;
        break;
    case 33:
        s->head_size=64; s->block_size=32; s->num_query_heads=32; s->num_kv_heads=32;
        s->dtype="bf16"; s->use_sinks=false; s->sliding_window=0; s->has_softcap=false;
        s->num_seqs=257;
        break;
    case 34:
        s->head_size=64; s->block_size=32; s->num_query_heads=32; s->num_kv_heads=32;
        s->dtype="bf16"; s->use_sinks=false; s->sliding_window=0; s->has_softcap=false;
        s->num_seqs=1000000;
        break;
    case 35:
        s->head_size=128; s->block_size=32; s->num_query_heads=32; s->num_kv_heads=32;
        s->dtype="fp16"; s->use_sinks=false; s->sliding_window=0; s->has_softcap=false;
        s->use_mfma_32x32x8=true; s->block_m_per_warp=32; s->has_tile_size=true; s->tile_size=64;
        break;
    case 36:
        s->head_size=64; s->block_size=32; s->num_query_heads=32; s->num_kv_heads=32;
        s->dtype="fp16"; s->use_sinks=false; s->sliding_window=0; s->has_softcap=false;
        s->use_mfma_32x32x8=true; s->use_transposed_qk_32x32=true;
        s->block_m_per_warp=32; s->has_tile_size=true; s->tile_size=64;
        break;
    case 37:
        s->head_size=64; s->block_size=32; s->num_query_heads=32; s->num_kv_heads=32;
        s->dtype="bf16"; s->use_sinks=false; s->sliding_window=0; s->has_softcap=false;
        s->use_register_pv=true;
        break;
    case 38:
        s->head_size=128; s->block_size=128; s->num_query_heads=32; s->num_kv_heads=32;
        s->dtype="bf16"; s->use_sinks=true; s->sliding_window=4096; s->has_softcap=true;
        s->num_warps=8; s->has_tile_size=true; s->tile_size=128;
        break;
    case 39:
        s->head_size=384; s->block_size=128; s->num_query_heads=16; s->num_kv_heads=4;
        s->dtype="bf16"; s->use_sinks=false; s->sliding_window=0; s->has_softcap=false;
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
        ckc_build_unified_attention_2d_tiled_scalar_new(&b, &s, "gfx942");
    if (!kernel) {
        fprintf(stderr, "build failed: err=%s\n", b.err);
        ckc_ir_builder_free(&b);
        return 1;
    }

    char *llvm_text = NULL;
    ckc_status_t st = ckc_lower_kernel_to_llvm(
        kernel, CKC_LLVM_FLAVOR_AUTO, "gfx942", &llvm_text);
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
