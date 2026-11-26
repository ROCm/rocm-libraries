/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/conv_direct_fwd_rr_mfma_emit.c -- C-side emitter for the gfx950
 * rows-in-registers MFMA direct forward parity harness (the conv and its
 * weight-format helper). Selects one of N sampled spec configs by argv[1] (the
 * config index), builds the rocke_fwd_rr_mfma_spec_t (or
 * rocke_fwd_rr_wfmt_spec_t) identically to the Python emitter
 * conv_direct_fwd_rr_mfma_emit.py, builds the kernel via
 * rocke_build_fwd_rr_{mfma,wfmt}_new and lowers via rocke_lower_kernel_to_llvm
 * (per-config arch, flavor AUTO), printing the .ll to stdout so the two outputs
 * can be byte-compared.
 *
 * Config index table: see the Python emitter (must stay in sync).
 */
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/instance_conv_direct_fwd_rr_mfma.h"
#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"
#include "rocke/verify.h"

/* Problem overrides of a row (pad_w -1 = PAD). */
typedef struct
{
    int N, W, groups, C, K, KW, PAD, pad_w;
} prob_t;

/* Spec knobs of a row. */
typedef struct
{
    int kw16;
    bool odd_c, c64, unfold, k4w, c_tail, one_c, grp, xcd, chunk;
} knobs_t;

#define BASE_P                       \
    {                                \
        2, 40, 1, 128, 128, 3, 1, -1 \
    }
#define KN(kw16)                                                            \
    {                                                                       \
        kw16, false, false, false, false, false, false, false, false, false \
    }

typedef struct
{
    prob_t p;
    knobs_t k;
} conv_row_t;

/* Rows 0-20 (fp16); rows 21-41 are the same in bf16. */
static const conv_row_t k_conv[] = {
    /* 0  3x3 p1 */
    {BASE_P, KN(4)},
    /* 1  3x3 p0, K 192, kw16 3 */
    {{2, 40, 1, 128, 192, 3, 0, -1}, KN(3)},
    /* 2  3x3 p2, C 192, kw16 2, odd_c */
    {{2, 40, 1, 192, 128, 3, 2, -1},
     {2, true, false, false, false, false, false, false, false, false}},
    /* 3  3x3 p1, kw16 2, c64 */
    {BASE_P, {2, false, true, false, false, false, false, false, false, false}},
    /* 4  3x3 p2, C 192, kw16 2, c64, odd_c */
    {{2, 40, 1, 192, 128, 3, 2, -1},
     {2, true, true, false, false, false, false, false, false, false}},
    /* 5  3x3 p1, K 256, k4w */
    {{2, 40, 1, 128, 256, 3, 1, -1},
     {4, false, false, false, true, false, false, false, false, false}},
    /* 6  3x3 p0, C 72, K 256, k4w, c_tail */
    {{2, 40, 1, 72, 256, 3, 0, -1},
     {4, false, false, false, true, true, false, false, false, false}},
    /* 7  3x3 p0, C 64, one_c */
    {{2, 40, 1, 64, 128, 3, 0, -1},
     {4, false, false, false, false, false, true, false, false, false}},
    /* 8  3x3 p1, C 40, kw16 2, one_c, c_tail */
    {{2, 40, 1, 40, 128, 3, 1, -1},
     {2, false, false, false, false, true, true, false, false, false}},
    /* 9  3x3 p1, groups 2, grp */
    {{2, 40, 2, 128, 128, 3, 1, -1},
     {4, false, false, false, false, false, false, true, false, false}},
    /* 10 3x3 p1, groups 4, C 64, K 64, grp, kw16 2, one_c */
    {{2, 40, 4, 64, 64, 3, 1, -1},
     {2, false, false, false, false, false, true, true, false, false}},
    /* 11 3x3 p1, xcd */
    {BASE_P, {4, false, false, false, false, false, false, false, true, false}},
    /* 12 3x3 p1, W 20, K 256, k4w, chunk */
    {{2, 20, 1, 128, 256, 3, 1, -1},
     {4, false, false, false, true, false, false, false, false, true}},
    /* 13 3x1 p1x0 */
    {{2, 40, 1, 128, 128, 1, 1, 0}, KN(4)},
    /* 14 3x1 p1x0, W 1, N 40, unfold */
    {{40, 1, 1, 128, 128, 1, 1, 0},
     {4, false, false, true, false, false, false, false, false, false}},
    /* 15 3x1 p2x0, W 1, N 40, K 256, unfold, k4w, xcd */
    {{40, 1, 1, 128, 256, 1, 2, 0},
     {4, false, false, true, true, false, false, false, true, false}},
    /* 16 3x1 p1x0, W 1, N 40, C 64, unfold, kw16 2, one_c */
    {{40, 1, 1, 64, 128, 1, 1, 0},
     {2, false, false, true, false, false, true, false, false, false}},
    /* 17 3x1 p2x0, K 192, kw16 3 */
    {{2, 40, 1, 128, 192, 1, 2, 0}, KN(3)},
    /* 18 3x1 p0x0, C 192, kw16 2, odd_c */
    {{2, 40, 1, 192, 128, 1, 0, 0},
     {2, true, false, false, false, false, false, false, false, false}},
    /* 19 3x3 p2, kw16 2, c64, xcd */
    {{2, 40, 1, 128, 128, 3, 2, -1},
     {2, false, true, false, false, false, false, false, true, false}},
    /* 20 3x3 p1, C 64, K 256, k4w, one_c, xcd */
    {{2, 40, 1, 64, 256, 3, 1, -1},
     {4, false, false, false, true, false, true, false, true, false}},
};
#define N_CONV ((int)(sizeof k_conv / sizeof k_conv[0]))

/* (kw16, fw, nwk, dg) of rows 42-47 (fp16); 48-53 the same in bf16. */
static const int k_wfmt[][4] = {
    {4, 3, 2, 0},
    {3, 1, 2, 0},
    {2, 3, 2, 0},
    {4, 3, 4, 0},
    {2, 1, 4, 0},
    {4, 3, 2, 1},
};
#define N_WFMT ((int)(sizeof k_wfmt / sizeof k_wfmt[0]))

/* Rejected by both engines: rows 54-60. */
typedef struct
{
    conv_row_t row;
    const char* arch;
} reject_row_t;

static const reject_row_t k_reject[] = {
    /* 54 gfx942 */
    {{BASE_P, KN(4)}, "gfx942"},
    /* 55 C 96 */
    {{{2, 40, 1, 96, 128, 3, 1, -1}, KN(4)}, "gfx950"},
    /* 56 3x1 with c64 */
    {{{2, 40, 1, 128, 128, 1, 1, 0},
      {2, false, true, false, false, false, false, false, false, false}},
     "gfx950"},
    /* 57 chunk without k4w */
    {{BASE_P, {4, false, false, false, false, false, false, false, false, true}}, "gfx950"},
    /* 58 C 192 without odd_c */
    {{{2, 40, 1, 192, 128, 3, 1, -1}, KN(4)}, "gfx950"},
    /* 59 kw16 5 */
    {{BASE_P, KN(5)}, "gfx950"},
    /* 60 padding 3 */
    {{{2, 40, 1, 128, 128, 3, 3, -1}, KN(4)}, "gfx950"},
};
#define REJECT0 (2 * N_CONV + 2 * N_WFMT)
#define N_REJECT ((int)(sizeof k_reject / sizeof k_reject[0]))

static rocke_direct_conv_problem_t make_problem(const prob_t* r, const char* dtype)
{
    rocke_direct_conv_problem_t p = rocke_direct_conv_problem_default();
    p.N = r->N;
    p.H = 20;
    p.W = r->W;
    p.groups = r->groups;
    p.cpg = r->C;
    p.kpg = r->K;
    p.KH = 3;
    p.KW = r->KW;
    p.PAD = r->PAD;
    p.pad_w = r->pad_w;
    p.stride = 1;
    p.dil_h = 1;
    p.dil_w = 1;
    p.dtype = dtype;
    return p;
}

static rocke_fwd_rr_mfma_spec_t make_conv(const conv_row_t* r, const char* dtype)
{
    rocke_fwd_rr_mfma_spec_t s = rocke_fwd_rr_mfma_spec_default();
    s.problem = make_problem(&r->p, dtype);
    s.kw16 = r->k.kw16;
    s.odd_c = r->k.odd_c;
    s.c64 = r->k.c64;
    s.unfold = r->k.unfold;
    s.k4w = r->k.k4w;
    s.c_tail = r->k.c_tail;
    s.one_c = r->k.one_c;
    s.grp = r->k.grp;
    s.xcd = r->k.xcd;
    s.chunk = r->k.chunk;
    return s;
}

/* 0: conv spec in *conv, 1: helper spec in *wfmt, -1: unknown index. */
static int make_cfg(int idx,
                    rocke_fwd_rr_mfma_spec_t* conv,
                    rocke_fwd_rr_wfmt_spec_t* wfmt,
                    const char** arch)
{
    *arch = "gfx950";
    if(idx >= 0 && idx < 2 * N_CONV)
    {
        *conv = make_conv(&k_conv[idx % N_CONV], idx >= N_CONV ? "bf16" : "fp16");
        return 0;
    }
    if(idx >= 2 * N_CONV && idx < REJECT0)
    {
        const int j = idx - 2 * N_CONV;
        const int* w = k_wfmt[j % N_WFMT];
        prob_t pr = {2, 40, 1, 72, 192, w[1], 1, -1};
        *wfmt = rocke_fwd_rr_wfmt_spec_default();
        wfmt->problem = make_problem(&pr, j >= N_WFMT ? "bf16" : "fp16");
        wfmt->kw16 = w[0];
        wfmt->fw = w[1];
        wfmt->nwk = w[2];
        wfmt->dg = w[3] != 0;
        return 1;
    }
    if(idx >= REJECT0 && idx < REJECT0 + N_REJECT)
    {
        const reject_row_t* r = &k_reject[idx - REJECT0];
        *conv = make_conv(&r->row, "fp16");
        *arch = r->arch;
        return 0;
    }
    return -1;
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

    rocke_fwd_rr_mfma_spec_t conv;
    rocke_fwd_rr_wfmt_spec_t wfmt;
    const char* arch = "gfx950";
    int kind = make_cfg(idx, &conv, &wfmt, &arch);
    if(kind < 0)
    {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 2;
    }

    rocke_ir_builder_t b;
    rocke_kernel_def_t* kernel = kind == 0 ? rocke_build_fwd_rr_mfma_new(&b, &conv, arch)
                                           : rocke_build_fwd_rr_wfmt_new(&b, &wfmt, arch);
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
