/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/conv_direct_fwd_tapouter_wmma_emit.c -- C-side emitter for the
 * tap-outer direct forward of large dilations (direct_fwd_tapouter_wmma).
 * Selects one of N sampled spec configs by argv[1] (the config index), builds
 * the rocke_fwd_tapouter_wmma_spec_t identically to the Python emitter
 * conv_direct_fwd_tapouter_wmma_emit.py, builds the kernel via
 * rocke_build_fwd_tapouter_wmma_new and lowers via rocke_lower_kernel_to_llvm
 * (per-config arch, flavor AUTO), printing the .ll to stdout so the two
 * outputs can be byte-compared.
 *
 * Config index table: see the Python emitter (must stay in sync).
 */
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/instance_conv_direct_fwd_tapouter_wmma.h"
#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"
#include "rocke/verify.h"

/* make_problem(dil_w, dil_h, PAD, dtype, groups, cpg, kpg, H, W, stride, pad_w):
 * the Python emitter's _p() (N 2, 3x3). */
static rocke_direct_conv_problem_t make_problem(
    int d, int dh, int pad, const char* dtype, int g, int c, int k, int H, int W, int s, int pad_w)
{
    rocke_direct_conv_problem_t p = rocke_direct_conv_problem_default();
    p.N = 2;
    p.H = H;
    p.W = W;
    p.groups = g;
    p.cpg = c;
    p.kpg = k;
    p.KH = 3;
    p.KW = 3;
    p.PAD = pad;
    p.pad_w = pad_w;
    p.stride = s;
    p.dtype = dtype;
    p.dil_h = dh;
    p.dil_w = d;
    return p;
}

static rocke_direct_conv_problem_t same(int d, const char* dtype)
{
    return make_problem(d, d, d, dtype, 1, 64, 96, 33, 33, 1, -1);
}

static void
    knobs(rocke_fwd_tapouter_wmma_spec_t* s, int wk, int wp, int tk, int tp, int tw, int ck, int st)
{
    s->waves_k = wk;
    s->waves_p = wp;
    s->tiles_k = tk;
    s->tiles_p = tp;
    s->tile_w = tw;
    s->ck = ck;
    s->stages = st;
}

/* Returns 0 for a known config, -1 if unknown. */
static int make_cfg(int idx, rocke_fwd_tapouter_wmma_spec_t* s, const char** arch)
{
    *s = rocke_fwd_tapouter_wmma_spec_default();
    *arch = "gfx1250";
    switch(idx)
    {
    case 0:
        s->problem = same(2, "bf16");
        knobs(s, 1, 4, 8, 2, 16, 64, 2);
        return 0;
    case 1:
        s->problem = same(6, "fp16");
        knobs(s, 2, 2, 4, 4, 8, 128, 2);
        return 0;
    case 2:
        s->problem = same(12, "bf16");
        knobs(s, 2, 2, 4, 4, 32, 128, 2);
        return 0;
    case 3:
        s->problem = make_problem(24, 24, 24, "fp16", 2, 32, 48, 33, 33, 1, -1);
        knobs(s, 4, 1, 4, 4, 16, 128, 2);
        return 0;
    case 4:
        s->problem = make_problem(36, 36, 36, "bf16", 1, 64, 96, 40, 44, 1, -1);
        knobs(s, 4, 2, 4, 4, 16, 64, 2);
        return 0;
    case 5:
        s->problem = same(4, "fp16");
        return 0;
    case 6:
        s->problem = same(18, "bf16");
        knobs(s, 2, 2, 4, 2, 16, 64, 2);
        s->prefetch = true;
        s->sched = "iterative-ilp";
        s->tdm_split = 2;
        return 0;
    case 7:
        s->problem = make_problem(3, 2, 3, "fp16", 1, 64, 96, 33, 33, 1, -1);
        return 0;
    case 8:
        s->problem = same(2, "bf16");
        *arch = "gfx950";
        return 0;
    case 9:
        s->problem = make_problem(2, 2, 2, "bf16", 1, 64, 96, 33, 33, 2, -1);
        return 0;
    case 10:
        s->problem = make_problem(2, 2, 2, "bf16", 1, 12, 96, 33, 33, 1, -1);
        return 0;
    case 11:
        s->problem = same(2, "bf16");
        s->tdm_split = 3;
        return 0;
    case 12:
        s->problem = make_problem(2, 2, 2, "bf16", 1, 64, 96, 33, 33, 1, 1);
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

    rocke_fwd_tapouter_wmma_spec_t spec;
    const char* arch = "gfx1250";
    if(make_cfg(idx, &spec, &arch) < 0)
    {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 2;
    }

    rocke_ir_builder_t b;
    rocke_kernel_def_t* kernel = rocke_build_fwd_tapouter_wmma_new(&b, &spec, arch);
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
