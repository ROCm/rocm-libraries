/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/conv_direct_dw_wgrad_wmma_emit.c -- C-side emitter for the
 * Toeplitz WMMA depthwise wgrad (direct_depthwise_wgrad_wmma). Selects one of
 * N sampled spec configs by argv[1] (the config index), builds the
 * rocke_direct_dw_wgrad_wmma_spec_t identically to the Python emitter
 * conv_direct_dw_wgrad_wmma_emit.py, builds the kernel via
 * rocke_build_direct_dw_wgrad_wmma_new and lowers via rocke_lower_kernel_to_llvm
 * (per-config arch, flavor AUTO), printing the .ll to stdout so the two
 * outputs can be byte-compared.
 *
 * Config index table: see the Python emitter (must stay in sync).
 */
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/instance_conv_direct_dw_wgrad_wmma.h"
#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"
#include "rocke/verify.h"

static rocke_direct_conv_problem_t make_problem(int KH,
                                                int KW,
                                                int PAD,
                                                const char* dtype,
                                                int pad_w,
                                                int G,
                                                int H,
                                                int W,
                                                int stride,
                                                int cpg,
                                                int dil)
{
    rocke_direct_conv_problem_t p = rocke_direct_conv_problem_default();
    p.N = 2;
    p.H = H;
    p.W = W;
    p.groups = G;
    p.cpg = cpg;
    p.kpg = cpg;
    p.KH = KH;
    p.KW = KW;
    p.PAD = PAD;
    p.pad_w = pad_w;
    p.stride = stride;
    p.dtype = dtype;
    p.dil_h = dil;
    p.dil_w = dil;
    return p;
}

/* _p(KH, KW, PAD, dtype, pad_w) on the default 64 groups, H 20, W 37. */
static rocke_direct_conv_problem_t p7(int KH, int KW, int PAD, const char* dtype, int pad_w)
{
    return make_problem(KH, KW, PAD, dtype, pad_w, 64, 20, 37, 1, 1, 1);
}

/* _p31(dtype): 1x31, column padding 15, H 6, W 80. */
static rocke_direct_conv_problem_t p31(const char* dtype)
{
    return make_problem(1, 31, 0, dtype, 15, 64, 6, 80, 1, 1, 1);
}

static void
    knobs(rocke_direct_dw_wgrad_wmma_spec_t* s, int waves, int tile_q, int rows, int tb, bool pf)
{
    s->waves = waves;
    s->tile_q = tile_q;
    s->rows = rows;
    s->target_blocks = tb;
    s->prefetch = pf;
}

/* Returns 0 for a known config, -1 if unknown. */
static int make_cfg(int idx, rocke_direct_dw_wgrad_wmma_spec_t* s, const char** arch)
{
    *s = rocke_direct_dw_wgrad_wmma_spec_default();
    *arch = "gfx1250";
    switch(idx)
    {
    case 0:
        s->problem = p7(7, 7, 3, "bf16", -1);
        return 0;
    case 1:
        s->problem = p7(7, 7, 3, "fp16", -1);
        knobs(s, 8, 16, 4, 512, true);
        return 0;
    case 2:
        s->problem = p7(7, 7, 0, "bf16", -1);
        knobs(s, 4, 8, 8, 512, false);
        return 0;
    case 3:
        s->problem = p7(7, 7, 0, "fp16", -1);
        knobs(s, 8, 32, 4, 1024, false);
        return 0;
    case 4:
        s->problem = p7(7, 7, 3, "bf16", -1);
        knobs(s, 8, 8, 4, 1024, false);
        return 0;
    case 5:
        s->problem = p7(7, 7, 3, "fp16", -1);
        knobs(s, 4, 16, 8, 512, false);
        return 0;
    case 6:
        s->problem = p7(7, 7, 0, "fp16", -1);
        knobs(s, 4, 16, 8, 1024, true);
        return 0;
    case 7:
        s->problem = p31("bf16");
        knobs(s, 2, 32, 2, 1024, true);
        return 0;
    case 8:
        s->problem = p31("fp16");
        knobs(s, 4, 8, 2, 512, false);
        return 0;
    case 9:
        s->problem = p31("bf16");
        knobs(s, 4, 16, 2, 512, true);
        return 0;
    case 10:
        s->problem = p31("fp16");
        knobs(s, 2, 16, 2, 1024, false);
        return 0;
    case 11:
        s->problem = p7(3, 3, 1, "fp16", -1);
        return 0;
    case 12:
        s->problem = p7(5, 5, 2, "bf16", -1);
        knobs(s, 1, 4, 2, 1024, false);
        return 0;
    case 13:
        s->problem = p7(1, 7, 0, "fp16", 3);
        s->waves = 2;
        s->tile_q = 12;
        s->rows = 6;
        return 0;
    case 14:
        s->problem = make_problem(3, 5, 1, "bf16", 2, 24, 20, 37, 1, 1, 1);
        s->ws_replicas = 1;
        return 0;
    case 15:
        s->problem = make_problem(7, 7, 3, "fp16", -1, 40, 13, 19, 1, 1, 1);
        s->waves = 16;
        s->tile_q = 4;
        s->rows = 2;
        return 0;
    case 16:
        s->problem = p7(2, 9, 0, "bf16", -1);
        return 0;
    case 17:
        s->problem = make_problem(7, 7, 3, "bf16", -1, 64, 20, 37, 2, 1, 1);
        return 0;
    case 18:
        s->problem = make_problem(7, 7, 3, "bf16", -1, 32, 20, 37, 1, 2, 1);
        return 0;
    case 19:
        s->problem = p7(7, 7, 3, "bf16", -1);
        s->rows = 3;
        return 0;
    case 20:
        s->problem = p7(7, 7, 3, "bf16", -1);
        s->tile_q = 6;
        return 0;
    case 21:
        s->problem = p7(7, 7, 3, "bf16", -1);
        *arch = "gfx950";
        return 0;
    case 22:
        s->problem = make_problem(7, 7, 6, "bf16", -1, 64, 20, 37, 1, 1, 2);
        return 0;
    case 23:
        s->problem = p7(7, 7, 3, "bf16", -2);
        return 0;
    case 24:
        s->problem = make_problem(7, 31, 3, "bf16", 15, 64, 40, 80, 1, 1, 1);
        s->rows = 16;
        s->tile_q = 64;
        return 0;
    case 25:
        s->problem = make_problem(7, 7, 3, "bf16", -1, 12, 20, 37, 1, 1, 1);
        return 0;
    case 26:
        s->problem = p7(7, 7, 3, "bf16", -1);
        s->wave_size = 64;
        return 0;
    case 27:
        s->problem = p7(7, 7, 3, "bf16", -1);
        s->waves = 16;
        s->rows = 16;
        s->tile_q = 64;
        return 0;
    default:
        return -1;
    }
}

int main(int argc, char** argv)
{
    if(argc < 2)
    {
        fprintf(stderr, "usage: %s <config_index> [ll|ir|verify]\n", argv[0]);
        return 2;
    }
    int idx = atoi(argv[1]);
    const char* mode = (argc > 2) ? argv[2] : "ll";

    rocke_direct_dw_wgrad_wmma_spec_t spec;
    const char* arch = "gfx1250";
    if(make_cfg(idx, &spec, &arch) < 0)
    {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 2;
    }

    rocke_ir_builder_t b;
    rocke_kernel_def_t* kernel = rocke_build_direct_dw_wgrad_wmma_new(&b, &spec, arch);
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
        char* llvm_text = NULL;
        rocke_status_t st
            = rocke_lower_kernel_to_llvm(kernel, ROCKE_LLVM_FLAVOR_AUTO, arch, &llvm_text);
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
        char* ir_text = NULL;
        rocke_status_t st = rocke_ir_serialize(kernel, &ir_text);
        if(st != ROCKE_OK || !ir_text)
        {
            fprintf(stderr, "serialize failed: status=%d\n", (int)st);
            rocke_ir_builder_free(&b);
            return 1;
        }
        fputs(ir_text, stdout);
        free(ir_text);
    }
    else if(strcmp(mode, "verify") == 0)
    {
        rocke_diag_t* diags = NULL;
        size_t ndiag = 0;
        rocke_verify(kernel, &diags, &ndiag);
        for(size_t i = 0; i < ndiag; i++)
        {
            char* s = rocke_diag_to_string(&diags[i]);
            if(s)
            {
                fputs(s, stdout);
                fputc('\n', stdout);
                free(s);
            }
        }
        rocke_diags_free(diags, ndiag);
    }
    else
    {
        fprintf(stderr, "unknown mode '%s' (expected ll|ir|verify)\n", mode);
        ret = 2;
    }

    rocke_ir_builder_free(&b);
    return ret;
}
