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
