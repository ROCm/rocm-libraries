/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/gfx1151_wmma_fmha_fwd_emit.c -- C-side emitter for the gfx1151
 * WMMA FMHA forward parity harness. Selects one of 131 configurations
 * by argv[1] (0..130), builds it exactly as the
 * Python emitter gfx1151_wmma_fmha_fwd_emit.py does, and lowers to LLVM .ll
 * text at arch=gfx1151 (flavor AUTO) so the two outputs can be byte-compared.
 *
 * Build flow (mirrors the Python build_wmma_fmha_fwd path):
 *   (1) rocke_ir_builder_init(b, spec.kernel_name())
 *   (2) rocke_build_wmma_fmha_fwd(b, &spec, "gfx1151")  -> KernelDef
 *   (3) rocke_lower_kernel_to_llvm(kernel, AUTO, "gfx1151", &ll)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/instance_gfx1151_wmma_fmha_fwd.h"
#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"
#include "rocke/verify.h"

/* Fill `spec` for config index `idx`. Returns 0 on success, -1 if unknown. */
static int make_spec(int idx, rocke_wmma_fmha_fwd_spec_t* spec)
{
    *spec = rocke_wmma_fmha_fwd_spec_default();
    if(idx >= 127 && idx < 131)
    {
        /* head_size, sliding_window, window_right, bottom_right */
        static const int cases[4][4] = {
            {64, 0, 16, 0},
            {64, 128, 16, 0},
            {128, 64, 0, 1},
            {64, 0, 32, 1},
        };
        const int* c = cases[idx - 127];
        spec->head_size = c[0];
        spec->num_query_heads = 8;
        spec->num_kv_heads = 2;
        spec->mask_mode = ROCKE_FMHA_MASK_NONE;
        spec->sliding_window = c[1];
        spec->window_right = c[2];
        spec->causal_bottom_right = c[3] != 0;
        spec->query_tail = c[3] != 0;
        spec->kv_tail = c[3] != 0;
        spec->v_lds_stage = c[3] != 0;
        return 0;
    }
    if(idx >= 123 && idx < 127)
    {
        /* head_size, v_head_size, value_tile_size, causal, v_lds_stage */
        static const int cases[4][5] = {
            {128, 64, 0, 0, 0},
            {64, 128, 0, 1, 1},
            {192, 128, 0, 0, 1},
            {128, 256, 64, 0, 0},
        };
        const int* c = cases[idx - 123];
        spec->head_size = c[0];
        spec->num_query_heads = 8;
        spec->num_kv_heads = 2;
        spec->mask_mode = c[3] ? ROCKE_FMHA_MASK_CAUSAL : ROCKE_FMHA_MASK_NONE;
        spec->v_lds_stage = c[4] != 0;
        spec->v_head_size = c[1];
        spec->value_tile_size = c[2];
        return 0;
    }
    if(idx >= 119 && idx < 123)
    {
        int variant = idx - 119;
        bool bottom_right = variant % 2 != 0;
        spec->head_size = variant < 2 ? 64 : 128;
        spec->num_query_heads = 8;
        spec->num_kv_heads = 2;
        spec->mask_mode = ROCKE_FMHA_MASK_CAUSAL;
        spec->causal_bottom_right = bottom_right;
        spec->query_tail = bottom_right;
        spec->kv_tail = bottom_right;
        spec->v_lds_stage = bottom_right;
        spec->causal_tile_skip = true;
        return 0;
    }
    if(idx >= 99 && idx < 115)
    {
        int variant = idx - 99;
        static const int tiles[] = {16, 32, 64, 128};
        bool vlds = variant % 2 != 0;
        spec->head_size = 256;
        spec->num_query_heads = 8;
        spec->num_kv_heads = 2;
        spec->dtype = variant < 8 ? "fp16" : "bf16";
        spec->mask_mode = vlds ? ROCKE_FMHA_MASK_CAUSAL : ROCKE_FMHA_MASK_NONE;
        spec->causal_bottom_right = vlds;
        spec->query_tail = vlds;
        spec->kv_tail = vlds;
        spec->v_lds_stage = vlds;
        spec->value_tile_size = tiles[(variant % 8) / 2];
        return 0;
    }
    if(idx >= 115 && idx < 119)
    {
        int variant = idx - 115;
        bool paged = variant % 2 != 0;
        spec->head_size = 256;
        spec->num_query_heads = 8;
        spec->num_kv_heads = 2;
        spec->dtype = variant < 2 ? "fp16" : "bf16";
        spec->mask_mode = ROCKE_FMHA_MASK_CAUSAL;
        spec->causal_bottom_right = true;
        spec->query_tail = true;
        spec->kv_tail = true;
        spec->layout = paged ? "paged" : "ragged";
        spec->page_block_size = paged ? 32 : 0;
        spec->kv_dtype = "fp8e4m3";
        spec->value_tile_size = 128;
        spec->v_lds_stage = paged;
        return 0;
    }
    if(idx >= 94 && idx < 99)
    {
        static const char* strategies[] = {"max-ilp",
                                           "max-memory-clause",
                                           "iterative-ilp",
                                           "iterative-minreg",
                                           "iterative-maxocc"};
        spec->head_size = 64;
        spec->num_query_heads = 8;
        spec->num_kv_heads = 8;
        spec->mask_mode = ROCKE_FMHA_MASK_CAUSAL;
        spec->causal_bottom_right = true;
        spec->layout = "ragged";
        spec->query_tail = true;
        spec->kv_tail = true;
        spec->v_lds_stage = true;
        spec->sliding_window = 320;
        spec->scheduler_strategy = strategies[idx - 94];
        return 0;
    }
    if(idx >= 86 && idx < 94)
    {
        static const int bases[] = {74, 75, 76, 77, 82, 83, 84, 85};
        if(make_spec(bases[idx - 86], spec) != 0)
            return -1;
        spec->causal_bottom_right = true;
        return 0;
    }
    if(idx >= 70 && idx < 86)
    {
        int variant = idx - 70;
        spec->head_size = variant < 8 ? 64 : 128;
        spec->num_query_heads = 8;
        spec->num_kv_heads = 2;
        spec->mask_mode = variant % 8 < 4 ? ROCKE_FMHA_MASK_NONE : ROCKE_FMHA_MASK_CAUSAL;
        spec->transposed_qk = true;
        spec->block_n = variant % 4 < 2 ? 32 : 64;
        spec->num_waves = 1 + variant % 2;
        return 0;
    }
    if(idx >= 58 && idx < 70)
    {
        static const int bases[] = {0, 7, 14, 15, 22, 23, 24, 25, 44, 47, 51, 57};
        spec->kv_dtype = "fp8e4m3";
        idx = bases[idx - 58];
    }
    if(idx >= 42 && idx < 58)
    {
        static const int bases[] = {0, 7, 12, 15, 16, 39, 0, 7, 12, 15, 16, 17, 26, 33, 38, 39};
        static const int pages[] = {0, 0, 0, 0, 0, 0, 16, 64, 16, 32, 32, 64, 64, 32, 16, 64};
        spec->page_block_size = pages[idx - 42];
        spec->layout = spec->page_block_size ? "paged" : "ragged";
        idx = bases[idx - 42];
    }
    if(idx >= 26 && idx < 42)
    {
        static const int bases[] = {12, 13, 2, 8, 2, 8, 12, 13, 0, 6, 18, 19, 24, 25, 2, 8};
        const int feature = (idx - 26) / 2;
        spec->sliding_window = feature == 0 || feature == 5 ? 128 : feature == 6 ? 64 : 0;
        spec->use_softcap = feature == 1 || feature == 6 || feature == 7;
        spec->use_sinks = feature == 2 || feature == 5 || feature == 6;
        spec->use_alibi = feature == 3 || feature == 6 || feature == 7;
        spec->use_qq_bias = feature == 4 || feature == 6;
        idx = bases[idx - 26];
    }
    if(idx >= 18 && idx < 26)
    {
        static const int bases[] = {12, 13, 3, 9, 12, 13, 16, 17};
        spec->query_tail = idx < 20 || idx >= 22;
        spec->kv_tail = idx >= 20;
        idx = bases[idx - 18];
    }
    if(idx >= 12 && idx < 18)
    {
        static const int bases[] = {2, 8, 3, 9, 5, 11};
        spec->causal_bottom_right = true;
        idx = bases[idx - 12];
    }
    if(idx >= 6 && idx < 12)
    {
        spec->dtype = "bf16";
        idx -= 6;
    }

    switch(idx)
    {
    case 0: /* H64, HQ4, HK0 (MHA), NONE, v_lds=False */
        spec->head_size = 64;
        spec->num_query_heads = 4;
        spec->num_kv_heads = 0;
        spec->mask_mode = ROCKE_FMHA_MASK_NONE;
        spec->v_lds_stage = false;
        break;
    case 1: /* H128, HQ8, HK0 (MHA), NONE, v_lds=False */
        spec->head_size = 128;
        spec->num_query_heads = 8;
        spec->num_kv_heads = 0;
        spec->mask_mode = ROCKE_FMHA_MASK_NONE;
        spec->v_lds_stage = false;
        break;
    case 2: /* H64, HQ4, HK0 (MHA), CAUSAL, v_lds=False */
        spec->head_size = 64;
        spec->num_query_heads = 4;
        spec->num_kv_heads = 0;
        spec->mask_mode = ROCKE_FMHA_MASK_CAUSAL;
        spec->v_lds_stage = false;
        break;
    case 3: /* H256, HQ8, HK2 (GQA), NONE, v_lds=False */
        spec->head_size = 256;
        spec->num_query_heads = 8;
        spec->num_kv_heads = 2;
        spec->mask_mode = ROCKE_FMHA_MASK_NONE;
        spec->v_lds_stage = false;
        break;
    case 4: /* H128, HQ4, HK4, CAUSAL, v_lds=False */
        spec->head_size = 128;
        spec->num_query_heads = 4;
        spec->num_kv_heads = 4;
        spec->mask_mode = ROCKE_FMHA_MASK_CAUSAL;
        spec->v_lds_stage = false;
        break;
    case 5: /* H64, HQ6, HK0 (MHA), NONE, v_lds=True */
        spec->head_size = 64;
        spec->num_query_heads = 6;
        spec->num_kv_heads = 0;
        spec->mask_mode = ROCKE_FMHA_MASK_NONE;
        spec->v_lds_stage = true;
        break;
    default:
        return -1;
    }
    if(spec->causal_bottom_right)
        spec->mask_mode = ROCKE_FMHA_MASK_CAUSAL;
    return 0;
}

int main(int argc, char** argv)
{
    if(argc < 2)
    {
        fprintf(stderr, "usage: %s <config_index 0..130>\n", argv[0]);
        return 2;
    }
    int idx = atoi(argv[1]);
    const char* mode = (argc > 2) ? argv[2] : "ll";

    rocke_wmma_fmha_fwd_spec_t spec;
    if(make_spec(idx, &spec) != 0)
    {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 2;
    }

    const char* arch = "gfx1151";

    /* Validate the spec (mirrors is_valid_spec). */
    char reason[256];
    reason[0] = 0;
    if(!rocke_wmma_fmha_fwd_is_valid_spec(&spec, arch, reason, sizeof reason))
    {
        fprintf(stderr, "invalid spec: %s\n", reason);
        return 1;
    }

    /* (1) init builder with spec.kernel_name() */
    char name[256];
    if(rocke_wmma_fmha_fwd_kernel_name(&spec, name, sizeof name) != ROCKE_OK)
    {
        fprintf(stderr, "kernel_name failed\n");
        return 1;
    }

    rocke_ir_builder_t b;
    if(rocke_ir_builder_init(&b, name) != ROCKE_OK)
    {
        fprintf(stderr, "ir_builder_init failed\n");
        return 1;
    }

    /* (2) build */
    rocke_kernel_def_t* kernel = rocke_build_wmma_fmha_fwd(&b, &spec, arch);
    if(kernel == NULL)
    {
        const char* m = rocke_ir_builder_error(&b);
        fprintf(stderr, "build failed: %s\n", m ? m : "(no message)");
        rocke_ir_builder_free(&b);
        return 1;
    }

    int ret = 0;
    if(strcmp(mode, "ll") == 0)
    {
        /* (3) lower to .ll (arch gfx1151, flavor AUTO) */
        char* llvm_text = NULL;
        rocke_status_t st
            = rocke_lower_kernel_to_llvm(kernel, ROCKE_LLVM_FLAVOR_AUTO, arch, &llvm_text);
        if(st != ROCKE_OK || !llvm_text)
        {
            fprintf(stderr, "lower failed: status=%d\n", (int)st);
            ret = 1;
        }
        else
        {
            fputs(llvm_text, stdout);
            free(llvm_text);
        }
    }
    else if(strcmp(mode, "ir") == 0)
    {
        char* t = NULL;
        rocke_status_t st = rocke_ir_serialize(kernel, &t);
        if(st != ROCKE_OK || !t)
        {
            fprintf(stderr, "serialize failed: status=%d\n", (int)st);
            ret = 1;
        }
        else
        {
            fputs(t, stdout);
            free(t);
        }
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
        ret = 2;
    }
    rocke_ir_builder_free(&b);
    return ret;
}
