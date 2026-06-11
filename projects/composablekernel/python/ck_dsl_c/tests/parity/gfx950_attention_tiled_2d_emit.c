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
 * The configs specify num_queries_per_kv (= num_query_heads // num_kv_heads);
 * each case sets num_query_heads / num_kv_heads to realise that ratio exactly as
 * the Python emitter does.
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
    case 0:
        /* head_size=128 block_size=64 nqpkv=8 bf16 sw=0 no-softcap nw=1 */
        s->head_size = 128; s->block_size = 64;
        s->num_query_heads = 8; s->num_kv_heads = 1;
        s->dtype = "bf16"; s->use_sinks = false;
        s->sliding_window = 0; s->has_softcap = false;
        s->num_warps = 1;
        break;
    case 1:
        /* head_size=64 block_size=32 nqpkv=8 fp16 sw=0 no-softcap nw=4
         * mfma32 + transposed_qk_32x32, tile_size=64 (block_m_per_warp=32) */
        s->head_size = 64; s->block_size = 32;
        s->num_query_heads = 64; s->num_kv_heads = 8;
        s->dtype = "fp16"; s->use_sinks = false;
        s->sliding_window = 0; s->has_softcap = false;
        s->num_warps = 4;
        s->block_m_per_warp = 32;
        s->use_mfma_32x32 = true;
        s->use_transposed_qk_32x32 = true;
        s->has_tile_size = true; s->tile_size = 64;
        break;
    case 2:
        /* head_size=128 block_size=32 nqpkv=4 bf16 sw=2048 softcap nw=2
         * mfma32, tile_size=64 (block_m_per_warp=32) */
        s->head_size = 128; s->block_size = 32;
        s->num_query_heads = 32; s->num_kv_heads = 8;
        s->dtype = "bf16"; s->use_sinks = false;
        s->sliding_window = 2048; s->has_softcap = true;
        s->num_warps = 2;
        s->block_m_per_warp = 32;
        s->use_mfma_32x32 = true;
        s->has_tile_size = true; s->tile_size = 64;
        break;
    case 3:
        /* head_size=64 block_size=64 nqpkv=1 bf16 kv=fp8e4m3 sw=0 nw=1
         * use_fp8_mfma_qk */
        s->head_size = 64; s->block_size = 64;
        s->num_query_heads = 8; s->num_kv_heads = 8;
        s->dtype = "bf16"; s->use_sinks = false;
        s->sliding_window = 0; s->has_softcap = false;
        s->num_warps = 1;
        s->kv_storage_dtype = "fp8e4m3";
        s->use_fp8_mfma_qk = true;
        break;
    case 4:
        /* head_size=256 block_size=64 nqpkv=16 fp16 sw=0 no-softcap nw=4
         * block_m_per_warp=32, tile_size=128 (no mfma32) */
        s->head_size = 256; s->block_size = 64;
        s->num_query_heads = 32; s->num_kv_heads = 2;
        s->dtype = "fp16"; s->use_sinks = false;
        s->sliding_window = 0; s->has_softcap = false;
        s->num_warps = 4;
        s->block_m_per_warp = 32;
        s->has_tile_size = true; s->tile_size = 128;
        break;
    case 5:
        /* head_size=128 block_size=16 nqpkv=2 bf16 sw=512 no-softcap nw=1
         * use_register_pv */
        s->head_size = 128; s->block_size = 16;
        s->num_query_heads = 16; s->num_kv_heads = 8;
        s->dtype = "bf16"; s->use_sinks = false;
        s->sliding_window = 512; s->has_softcap = false;
        s->num_warps = 1;
        s->use_register_pv = true;
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
