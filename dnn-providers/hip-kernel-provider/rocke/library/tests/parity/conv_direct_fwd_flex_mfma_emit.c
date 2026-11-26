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
#include <stdbool.h>
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
    case 11: /* bf16, 3x3 s1 pad 1, bk 192 */
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
    case 16: /* rejected by both engines: 7x7 (the offset table's LDS) */
        s->problem = make_problem("bf16");
        set_geom(&s->problem, 7, 7, 3, -1, 1, 1, 1);
        return 0;
    case 17: /* fp16, bk 64 */
        s->problem = make_problem("fp16");
        s->bk = 64;
        return 0;
    case 18: /* bf16, s2, bk 192, split_c 2 */
        s->problem = make_problem("bf16");
        s->problem.stride = 2;
        s->bk = 192;
        s->split_c = 2;
        return 0;
    case 19: /* fp16, bx 128, bk 128 */
        s->problem = make_problem("fp16");
        s->bx = 128;
        s->bk = 128;
        return 0;
    case 20: /* bf16, bx 128, bk 256, split_c 2 */
        s->problem = make_problem("bf16");
        s->bx = 128;
        s->split_c = 2;
        return 0;
    case 21: /* fp16, cs 4 */
        s->problem = make_problem("fp16");
        s->bx = 128;
        s->bk = 128;
        s->cs = 4;
        return 0;
    case 22: /* bf16, cs 4, wt, xcd, split_c 2 */
        s->problem = make_problem("bf16");
        s->bx = 128;
        s->bk = 128;
        s->cs = 4;
        s->wt = true;
        s->xcd = 1;
        s->split_c = 2;
        return 0;
    case 23: /* fp16, cs 4, xcd, kmaj, s2 */
        s->problem = make_problem("fp16");
        s->problem.stride = 2;
        s->bx = 128;
        s->bk = 128;
        s->cs = 4;
        s->xcd = 1;
        s->kmaj = true;
        return 0;
    case 24: /* bf16, cs 2 */
        s->problem = make_problem("bf16");
        s->bx = 128;
        s->cs = 2;
        return 0;
    case 25: /* fp16, cs 2, wt, xcd, kmaj, split_c 2 */
        s->problem = make_problem("fp16");
        s->bx = 128;
        s->cs = 2;
        s->wt = true;
        s->xcd = 1;
        s->kmaj = true;
        s->split_c = 2;
        return 0;
    case 26: /* bf16, wt, bk 256 */
        s->problem = make_problem("bf16");
        s->wt = true;
        return 0;
    case 27: /* fp16, wt, bk 192, split_c 3 */
        s->problem = make_problem("fp16");
        s->bk = 192;
        s->wt = true;
        s->split_c = 3;
        return 0;
    case 28: /* bf16, wt, bx 128, bk 128 */
        s->problem = make_problem("bf16");
        s->bx = 128;
        s->bk = 128;
        s->wt = true;
        return 0;
    case 29: /* fp16, wt, xt8, c_tail (C=200), bk 256 */
        s->problem = make_problem("fp16");
        s->problem.cpg = 200;
        s->wt = true;
        s->xt8 = true;
        s->c_tail = true;
        return 0;
    case 30: /* bf16, wt, xt8, sw, xcd, bk 256 */
        s->problem = make_problem("bf16");
        s->wt = true;
        s->xt8 = true;
        s->sw = true;
        s->xcd = 1;
        return 0;
    case 31: /* fp16, xcd, bk 256, split_c 2 */
        s->problem = make_problem("fp16");
        s->xcd = 1;
        s->split_c = 2;
        return 0;
    case 32: /* bf16, xcd 2, sw, kmaj, bk 128 */
        s->problem = make_problem("bf16");
        s->bk = 128;
        s->xcd = 2;
        s->sw = true;
        s->kmaj = true;
        return 0;
    case 33: /* fp16, xcd 3, sw, bk 256 */
        s->problem = make_problem("fp16");
        s->xcd = 3;
        s->sw = true;
        return 0;
    case 34: /* bf16, xcd, kmaj, bk 256 */
        s->problem = make_problem("bf16");
        s->xcd = 1;
        s->kmaj = true;
        return 0;
    case 35: /* fp16, c_tail (C=200), bk 256 */
        s->problem = make_problem("fp16");
        s->problem.cpg = 200;
        s->c_tail = true;
        return 0;
    case 36: /* bf16, c_tail (C=200), sw, xcd, bk 128 */
        s->problem = make_problem("bf16");
        s->problem.cpg = 200;
        s->bk = 128;
        s->c_tail = true;
        s->sw = true;
        s->xcd = 1;
        return 0;
    case 37: /* fp16, flat (C=200), bk 128 */
        s->problem = make_problem("fp16");
        s->problem.cpg = 200;
        s->bk = 128;
        s->flat = true;
        return 0;
    case 38: /* bf16, flat (C=200), bk 192, split_c 4 */
        s->problem = make_problem("bf16");
        s->problem.cpg = 200;
        s->bk = 192;
        s->flat = true;
        s->split_c = 4;
        return 0;
    case 39: /* fp16, k_tail (K=27), bk 64 */
        s->problem = make_problem("fp16");
        s->problem.kpg = 27;
        s->bk = 64;
        s->k_tail = true;
        return 0;
    case 40: /* bf16, k_tail (K=36), bk 64, split_c 2 */
        s->problem = make_problem("bf16");
        s->problem.kpg = 36;
        s->bk = 64;
        s->k_tail = true;
        s->split_c = 2;
        return 0;
    case 41: /* fp16, sw, bk 128 */
        s->problem = make_problem("fp16");
        s->bk = 128;
        s->sw = true;
        return 0;
    case 42: /* bf16, sw, bk 256, split_c 2 */
        s->problem = make_problem("bf16");
        s->sw = true;
        s->split_c = 2;
        return 0;
    case 43: /* fp16, grp (4 groups of 64 / 64), bk 64 */
        s->problem = make_problem("fp16");
        s->problem.groups = 4;
        s->problem.cpg = 64;
        s->problem.kpg = 64;
        s->bk = 64;
        s->grp = true;
        return 0;
    case 44: /* bf16, grp (2 groups of 128 / 256), sw, wt, bx 128, bk 256 */
        s->problem = make_problem("bf16");
        s->problem.groups = 2;
        s->problem.cpg = 128;
        s->problem.kpg = 256;
        s->bx = 128;
        s->grp = true;
        s->sw = true;
        s->wt = true;
        return 0;
    case 45: /* fp16, grp (2 groups of 56 / 64), c_tail, bk 64 */
        s->problem = make_problem("fp16");
        s->problem.groups = 2;
        s->problem.cpg = 56;
        s->problem.kpg = 64;
        s->bk = 64;
        s->grp = true;
        s->c_tail = true;
        return 0;
    case 46: /* bf16, ds dgrad 3x3 s2, wt, bx 128, bk 128 */
        s->problem = make_problem("bf16");
        s->problem.stride = 2;
        s->problem.cpg = 256;
        s->problem.kpg = 128;
        s->bx = 128;
        s->bk = 128;
        s->wt = true;
        s->ds = true;
        return 0;
    case 47: /* fp16, ds dgrad 1x16 s8 pad 0x4 (H=1), h1, wt, bk 128 */
        s->problem = make_problem("fp16");
        set_geom(&s->problem, 1, 16, 0, 4, 8, 1, 1);
        s->problem.H = 1;
        s->problem.W = 64;
        s->problem.cpg = 128;
        s->problem.kpg = 128;
        s->bk = 128;
        s->wt = true;
        s->ds = true;
        s->h1 = true;
        return 0;
    case 48: /* bf16, ds dgrad 1x1 s2, zf, wt, bk 192 */
        s->problem = make_problem("bf16");
        set_geom(&s->problem, 1, 1, 0, -1, 2, 1, 1);
        s->problem.cpg = 256;
        s->problem.kpg = 128;
        s->bk = 192;
        s->wt = true;
        s->ds = true;
        s->zf = true;
        return 0;
    case 49: /* fp16, ds dgrad 1x1 s2, zf, zl 8, wt, bx 128, bk 256 */
        s->problem = make_problem("fp16");
        set_geom(&s->problem, 1, 1, 0, -1, 2, 1, 1);
        s->problem.cpg = 256;
        s->problem.kpg = 128;
        s->bx = 128;
        s->wt = true;
        s->ds = true;
        s->zf = true;
        s->zl = 8;
        return 0;
    case 50: /* bf16, ds dgrad 1x1 s2 (K=264), zf, zl 8, c_tail, wt, bx 128, bk 256 */
        s->problem = make_problem("bf16");
        set_geom(&s->problem, 1, 1, 0, -1, 2, 1, 1);
        s->problem.cpg = 256;
        s->problem.kpg = 264;
        s->bx = 128;
        s->wt = true;
        s->ds = true;
        s->zf = true;
        s->zl = 8;
        s->c_tail = true;
        return 0;
    case 51: /* fp16, s2, bk 128, split_c 6 */
        s->problem = make_problem("fp16");
        s->problem.stride = 2;
        s->bk = 128;
        s->split_c = 6;
        return 0;
    case 52: /* bf16, C=1024, bk 256, split_c 16 */
        s->problem = make_problem("bf16");
        s->problem.cpg = 1024;
        s->split_c = 16;
        return 0;
    case 53: /* fp16, ds dgrad 3x3 s2, bk 256, split_c 2 (the transposed filter) */
        s->problem = make_problem("fp16");
        s->problem.stride = 2;
        s->problem.cpg = 256;
        s->problem.kpg = 128;
        s->ds = true;
        s->split_c = 2;
        return 0;
    case 54: /* rejected by both engines: cs 2 on a 2x2 filter (4 taps) */
        s->problem = make_problem("bf16");
        set_geom(&s->problem, 2, 2, 0, -1, 1, 1, 1);
        s->bx = 128;
        s->cs = 2;
        return 0;
    case 55: /* rejected by both engines: zf without wt */
        s->problem = make_problem("bf16");
        set_geom(&s->problem, 1, 1, 0, -1, 2, 1, 1);
        s->ds = true;
        s->zf = true;
        return 0;
    case 56: /* rejected by both engines: xt8 with bx 128 */
        s->problem = make_problem("bf16");
        s->bx = 128;
        s->xt8 = true;
        return 0;
    case 57: /* rejected by both engines: grp on a dense problem */
        s->problem = make_problem("bf16");
        s->grp = true;
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
