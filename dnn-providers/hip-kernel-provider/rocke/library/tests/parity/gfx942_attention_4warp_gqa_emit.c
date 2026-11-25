/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/gfx942_attention_4warp_gqa_emit.c -- C-side emitter for the gfx942
 * 4-warp GQA paged-attention parity harness.
 *
 * Selects one config by argv[1], fills a rocke_attention_tiled_2d_spec_t
 * identically to gfx942_attention_4warp_gqa_emit.py, builds the kernel via
 * rocke_build_gfx942_4warp_gqa(&b, &spec, "gfx942"), lowers it with
 * rocke_lower_kernel_to_llvm(kernel, AUTO, "gfx942", ...) and prints the .ll to
 * stdout for byte comparison.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/instance_gfx942_attention_4warp_gqa.h"
#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"
#include "rocke/verify.h"

static void set_shape(rocke_attention_tiled_2d_spec_t* s,
                      int head_size,
                      int block_size,
                      int num_query_heads,
                      int num_kv_heads,
                      const char* dtype,
                      int sliding_window,
                      int num_seqs,
                      int tile_size)
{
    *s = rocke_attention_tiled_2d_spec_default();
    s->use_sinks = false;
    s->has_softcap = false;
    s->num_warps = 1;
    s->block_m_per_warp = 32;
    s->has_waves_per_eu = true;
    s->waves_per_eu = 2;
    s->head_size = head_size;
    s->block_size = block_size;
    s->num_query_heads = num_query_heads;
    s->num_kv_heads = num_kv_heads;
    s->dtype = dtype;
    s->sliding_window = sliding_window;
    s->num_seqs = num_seqs;
    s->has_tile_size = true;
    s->tile_size = tile_size;
}

/* Fill `s` for config index `idx`. Returns 0 on success, -1 on unknown idx. */
static int make_spec(int idx, rocke_attention_tiled_2d_spec_t* s)
{
    switch(idx)
    {
    case 0: /* D256 lean, the rocke_ir_parity_harness D256 pin. */
        set_shape(s, 256, 16, 16, 2, "bf16", 0, 1, 32);
        s->use_register_pv = true;
        break;
    case 1: /* D256 lean, bs32, multi-sequence search. */
        set_shape(s, 256, 32, 32, 8, "bf16", 0, 4, 32);
        break;
    case 2: /* D128 pipe + head-fold, the rocke_ir_parity_harness fold pin. */
        set_shape(s, 128, 16, 32, 8, "bf16", 4096, 1, 32);
        break;
    case 3:
        set_shape(s, 128, 32, 32, 8, "bf16", 1024, 4, 32);
        break;
    case 4: /* D128 pipe, no fold: fp16. */
        set_shape(s, 128, 32, 32, 8, "fp16", 4096, 1, 32);
        break;
    case 5: /* D128 pipe, no fold: 5:1 GQA. */
        set_shape(s, 128, 16, 40, 8, "bf16", 4096, 1, 32);
        break;
    case 6: /* D128 bs64 single-buffer body. */
        set_shape(s, 128, 64, 32, 8, "bf16", 4096, 1, 64);
        break;
    case 7:
        set_shape(s, 128, 64, 32, 8, "fp16", 2048, 2, 64);
        break;
    case 8: /* D128 pipe, causal, no waves_per_eu. */
        set_shape(s, 128, 32, 32, 8, "bf16", 0, 1, 32);
        s->has_waves_per_eu = false;
        s->waves_per_eu = 0;
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
    rocke_kernel_def_t* kernel = rocke_build_gfx942_4warp_gqa(&b, &s, "gfx942");
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
            char* str = rocke_diag_to_string(&d[i]);
            if(str)
            {
                puts(str);
                free(str);
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
