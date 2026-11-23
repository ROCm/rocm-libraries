/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/conv_direct_fwd_flex_mfma_emit.c -- C-side emitter for the
 * gfx950 flattened-pixel implicit-GEMM MFMA direct forward parity harness.
 * Selects one of N sampled spec configs by argv[1] (the config index), builds
 * the rocke_fwd_flex_mfma_spec_t identically to the Python emitter
 * conv_direct_fwd_flex_mfma_emit.py, builds the kernel via
 * rocke_build_fwd_flex_mfma_new and lowers via rocke_lower_kernel_to_llvm
 * (per-config arch, flavor AUTO), printing the .ll to stdout so the two outputs
 * can be byte-compared.
 *
 * Config index table: see the Python emitter (must stay in sync).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/instance_conv_direct_fwd_flex_mfma.h"
#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"
#include "rocke/verify.h"

/* The N=2, 20x24, C=512, K=96 forward problem with a 3x3 filter, padding 1,
 * stride 1 and no dilation; each config overrides what its row changes. */
static rocke_direct_conv_problem_t make_problem(const char* dtype)
{
    rocke_direct_conv_problem_t p = rocke_direct_conv_problem_default();
    p.N = 2;
    p.H = 20;
    p.W = 24;
    p.groups = 1;
    p.cpg = 512;
    p.kpg = 96;
    p.KH = 3;
    p.KW = 3;
    p.PAD = 1;
    p.pad_w = -1;
    p.stride = 1;
    p.dil_h = 1;
    p.dil_w = 1;
    p.dtype = dtype;
    return p;
}

/* Filter, padding (pad_w -1 = PAD), stride and dilation of a config row. */
static void set_geom(rocke_direct_conv_problem_t* p,
                     int KH,
                     int KW,
                     int PAD,
                     int pad_w,
                     int stride,
                     int dil_h,
                     int dil_w)
{
    p->KH = KH;
    p->KW = KW;
    p->PAD = PAD;
    p->pad_w = pad_w;
    p->stride = stride;
    p->dil_h = dil_h;
    p->dil_w = dil_w;
}

static int make_cfg(int idx, rocke_fwd_flex_mfma_spec_t* s, const char** arch)
{
    *s = rocke_fwd_flex_mfma_spec_default();
    *arch = "gfx950";
    switch(idx)
    {
    case 0: /* bf16, 3x3 s2 pad 1, bk 256, split_c 1 */
        s->problem = make_problem("bf16");
        s->problem.stride = 2;
        return 0;
    case 1: /* fp16, 3x3 s2 pad 1, bk 128, split_c 1 */
        s->problem = make_problem("fp16");
        s->problem.stride = 2;
        s->bk = 128;
        return 0;
    case 2: /* bf16, 1x1 s2, bk 256, split_c 2 */
        s->problem = make_problem("bf16");
        set_geom(&s->problem, 1, 1, 0, -1, 2, 1, 1);
        s->split_c = 2;
        return 0;
    case 3: /* fp16, 2x2 s2, bk 128, split_c 4 */
        s->problem = make_problem("fp16");
        set_geom(&s->problem, 2, 2, 0, -1, 2, 1, 1);
        s->bk = 128;
        s->split_c = 4;
        return 0;
    case 4: /* bf16, 3x3 dil 2 pad 2, bk 128, split_c 8 */
        s->problem = make_problem("bf16");
        set_geom(&s->problem, 3, 3, 2, -1, 1, 2, 2);
        s->bk = 128;
        s->split_c = 8;
        return 0;
    case 5: /* fp16, 3x3 dil 12 pad 12, bk 256, split_c 1 */
        s->problem = make_problem("fp16");
        set_geom(&s->problem, 3, 3, 12, -1, 1, 12, 12);
        return 0;
    case 6: /* bf16, 1x3 dil 1x3 pad 0x3, bk 256, split_c 2 */
        s->problem = make_problem("bf16");
        set_geom(&s->problem, 1, 3, 0, 3, 1, 1, 3);
        s->split_c = 2;
        return 0;
    case 7: /* fp16, 7x1 pad 3x0, bk 128, split_c 1 */
        s->problem = make_problem("fp16");
        set_geom(&s->problem, 7, 1, 3, 0, 1, 1, 1);
        s->bk = 128;
        return 0;
    case 8: /* bf16, 1x11 dil 1x3 pad 0x15, bk 128, split_c 2 */
        s->problem = make_problem("bf16");
        set_geom(&s->problem, 1, 11, 0, 15, 1, 1, 3);
        s->bk = 128;
        s->split_c = 2;
        return 0;
    case 9: /* fp16, 1x7 dil 1x3 pad 0x9, bk 256, split_c 4 */
        s->problem = make_problem("fp16");
        set_geom(&s->problem, 1, 7, 0, 9, 1, 1, 3);
        s->split_c = 4;
        return 0;
    case 10: /* bf16, 3x3 dil 24 pad 24, bk 256, split_c 8 */
        s->problem = make_problem("bf16");
        set_geom(&s->problem, 3, 3, 24, -1, 1, 24, 24);
        s->split_c = 8;
        return 0;
    case 11: /* rejected by both engines: bk 192 */
        s->problem = make_problem("bf16");
        s->bk = 192;
        return 0;
    case 12: /* rejected by both engines: gfx942 */
        s->problem = make_problem("bf16");
        *arch = "gfx942";
        return 0;
    case 13: /* rejected by both engines: C = 96 */
        s->problem = make_problem("bf16");
        s->problem.cpg = 96;
        return 0;
    case 14: /* rejected by both engines: K = 100 */
        s->problem = make_problem("bf16");
        s->problem.kpg = 100;
        return 0;
    case 15: /* rejected by both engines: split_c 16 > C / 64 */
        s->problem = make_problem("bf16");
        s->split_c = 16;
        return 0;
    case 16: /* rejected by both engines: 7x7 (more than 32 taps) */
        s->problem = make_problem("bf16");
        set_geom(&s->problem, 7, 7, 3, -1, 1, 1, 1);
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

    rocke_fwd_flex_mfma_spec_t spec;
    const char* arch = "gfx950";
    if(make_cfg(idx, &spec, &arch) != 0)
    {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 2;
    }

    rocke_ir_builder_t b;
    rocke_kernel_def_t* kernel = rocke_build_fwd_flex_mfma_new(&b, &spec, arch);
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
