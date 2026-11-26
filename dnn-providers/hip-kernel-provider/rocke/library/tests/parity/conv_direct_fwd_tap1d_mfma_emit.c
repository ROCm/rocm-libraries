/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/conv_direct_fwd_tap1d_mfma_emit.c -- C-side emitter for the
 * gfx950 1-D tap-reuse MFMA direct forward parity harness. Selects one of N
 * sampled spec configs by argv[1] (the config index), builds the
 * rocke_fwd_tap1d_mfma_spec_t identically to the Python emitter
 * conv_direct_fwd_tap1d_mfma_emit.py, builds the kernel via
 * rocke_build_fwd_tap1d_mfma_new and lowers via rocke_lower_kernel_to_llvm
 * (per-config arch, flavor AUTO), printing the .ll to stdout so the two outputs
 * can be byte-compared.
 *
 * Config index table: see the Python emitter (must stay in sync).
 */
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/instance_conv_direct_fwd_tap1d_mfma.h"
#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"
#include "rocke/verify.h"

/* (S, pad_w, dil, C, K, bk, nwv, ck, one) of rows 0-7 (fp16); rows 8-15 are
 * the same in bf16. */
static const int k_rows[][9] = {
    {3, 0, 1, 128, 256, 128, 8, 64, 0},
    {3, 1, 1, 192, 128, 128, 4, 64, 0},
    {7, 9, 3, 96, 128, 128, 4, 32, 0},
    {11, 15, 3, 128, 64, 64, 4, 64, 0},
    {11, 25, 5, 96, 128, 64, 4, 32, 0},
    {7, 0, 1, 64, 128, 64, 4, 64, 1},
    {3, 5, 5, 64, 96, 32, 4, 64, 1},
    {11, 5, 1, 32, 64, 32, 4, 32, 1},
};
#define N_ROWS ((int)(sizeof k_rows / sizeof k_rows[0]))

/* The N=2, 4x50, C=128, K=128, 1x3 problem, stride 1, no padding or dilation. */
static rocke_direct_conv_problem_t make_problem(const char* dtype)
{
    rocke_direct_conv_problem_t p = rocke_direct_conv_problem_default();
    p.N = 2;
    p.H = 4;
    p.W = 50;
    p.groups = 1;
    p.cpg = 128;
    p.kpg = 128;
    p.KH = 1;
    p.KW = 3;
    p.PAD = 0;
    p.pad_w = 0;
    p.stride = 1;
    p.dil_h = 1;
    p.dil_w = 1;
    p.dtype = dtype;
    return p;
}

static int make_cfg(int idx, rocke_fwd_tap1d_mfma_spec_t* s, const char** arch)
{
    *s = rocke_fwd_tap1d_mfma_spec_default();
    *arch = "gfx950";
    s->problem = make_problem("fp16");
    if(idx >= 0 && idx < 2 * N_ROWS)
    {
        const int* r = k_rows[idx % N_ROWS];
        s->problem = make_problem(idx >= N_ROWS ? "bf16" : "fp16");
        s->problem.KW = r[0];
        s->problem.pad_w = r[1];
        s->problem.dil_w = r[2];
        s->problem.cpg = r[3];
        s->problem.kpg = r[4];
        s->bk = r[5];
        s->nwv = r[6];
        s->ck = r[7];
        s->one = r[8] != 0;
        return 0;
    }
    switch(idx)
    {
    case 16: /* rejected by both engines: gfx942 */
        *arch = "gfx942";
        return 0;
    case 17: /* rejected by both engines: 3x3 filter */
        s->problem.KH = 3;
        return 0;
    case 18: /* rejected by both engines: bk 96 */
        s->problem.kpg = 192;
        s->bk = 96;
        return 0;
    case 19: /* rejected by both engines: one with C != ck */
        s->one = true;
        return 0;
    case 20: /* rejected by both engines: C 96 with ck 64 */
        s->problem.cpg = 96;
        return 0;
    case 21: /* rejected by both engines: width padding 3 > dilation * (S - 1) */
        s->problem.pad_w = 3;
        return 0;
    case 22: /* rejected by both engines: stride 2 */
        s->problem.stride = 2;
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

    rocke_fwd_tap1d_mfma_spec_t spec;
    const char* arch = "gfx950";
    if(make_cfg(idx, &spec, &arch) != 0)
    {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 2;
    }

    rocke_ir_builder_t b;
    rocke_kernel_def_t* kernel = rocke_build_fwd_tap1d_mfma_new(&b, &spec, arch);
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
