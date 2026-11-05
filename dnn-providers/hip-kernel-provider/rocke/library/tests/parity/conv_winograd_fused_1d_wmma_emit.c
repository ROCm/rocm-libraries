/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/conv_winograd_fused_1d_wmma_emit.c -- C-side emitter for the
 * fused 1-D Winograd forward parity harness. Selects one of N sampled spec
 * configs by argv[1] (the config index), builds the
 * rocke_winograd_fused_1d_spec_t identically to the Python emitter
 * conv_winograd_fused_1d_wmma_emit.py, builds the kernel via
 * rocke_build_winograd_fused_1d_new and lowers via rocke_lower_kernel_to_llvm
 * (per-config arch, flavor AUTO), printing the .ll to stdout so the two outputs
 * can be byte-compared.
 *
 * Config index table: see the Python emitter (must stay in sync).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/instance_conv_winograd_fused_1d_wmma.h"
#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"
#include "rocke/verify.h"

/* _p(C, K, X, Wi, N, pW (-1 => dW * (X // 2)), dW, Hi) */
static rocke_conv_problem_t make_problem(int C, int K, int X, int Wi, int N, int pW, int dW, int Hi)
{
    if(pW < 0)
        pW = dW * (X / 2);
    rocke_conv_problem_t p = rocke_conv_problem_make(N, Hi, Wi, C, K, 1, X, 1, 1, 0, pW, 1, dW);
    p.groups = 1;
    return p;
}

static void axis(rocke_winograd_fused_1d_spec_t* s, int m, int r, int seg)
{
    s->ax_m = m;
    s->ax_r = r;
    s->ax_seg = seg;
}

static void sel(rocke_winograd_fused_1d_spec_t* s)
{
    s->q = 2;
    s->fk = 4;
    s->waves = 4;
    s->xf = "wmma";
    s->xf_batch = 16;
    s->prefetch = true;
    s->m_f16 = true;
    s->merge = true;
}

static int make_cfg(int idx, rocke_winograd_fused_1d_spec_t* s, const char** arch)
{
    *s = rocke_winograd_fused_1d_spec_default();
    *arch = "gfx1250";
    switch(idx)
    {
    case 0:
        s->problem = make_problem(128, 128, 11, 600, 1, -1, 1, 1);
        axis(s, 4, 11, 3);
        sel(s);
        return 0;
    case 1:
        s->problem = make_problem(32, 64, 11, 300, 2, -1, 1, 1);
        axis(s, 4, 11, 3);
        s->q = 4;
        s->fk = 2;
        s->xf = "wmma";
        s->xf_batch = 8;
        s->prefetch = true;
        s->m_f16 = true;
        s->merge = true;
        return 0;
    case 2:
        s->problem = make_problem(128, 128, 7, 600, 1, -1, 1, 1);
        axis(s, 4, 7, 2);
        sel(s);
        return 0;
    case 3:
        s->problem = make_problem(64, 64, 3, 300, 1, -1, 2, 1);
        axis(s, 6, 3, 1);
        sel(s);
        return 0;
    case 4:
        s->problem = make_problem(64, 64, 11, 517, 1, -1, 3, 1);
        axis(s, 4, 11, 3);
        sel(s);
        return 0;
    case 5:
        s->problem = make_problem(128, 128, 11, 600, 1, -1, 1, 1);
        axis(s, 4, 11, 3);
        return 0;
    case 6:
        s->problem = make_problem(128, 128, 11, 600, 1, -1, 1, 1);
        axis(s, 4, 11, 3);
        s->prefetch = true;
        s->m_f16 = true;
        return 0;
    case 7:
        s->problem = make_problem(64, 64, 11, 600, 1, -1, 1, 1);
        axis(s, 4, 11, 3);
        s->q = 2;
        s->fk = 2;
        s->prefetch = true;
        return 0;
    case 8:
        s->problem = make_problem(128, 128, 11, 600, 1, -1, 1, 1);
        axis(s, 4, 11, 3);
        s->q = 2;
        s->fk = 2;
        s->prefetch = true;
        s->m_f16 = true;
        s->merge = true;
        return 0;
    case 9:
        s->problem = make_problem(128, 128, 11, 600, 1, -1, 1, 1);
        axis(s, 4, 11, 3);
        sel(s);
        s->xf_batch = 8;
        s->waves_per_eu = 2;
        return 0;
    case 10:
        s->problem = make_problem(64, 64, 7, 517, 1, 15, 5, 1);
        axis(s, 4, 7, 1);
        s->q = 2;
        s->fk = 4;
        s->prefetch = true;
        s->m_f16 = true;
        s->merge = true;
        s->waves = 4;
        return 0;
    case 11:
        s->problem = make_problem(64, 64, 11, 600, 1, -1, 1, 1);
        axis(s, 6, 11, 2);
        s->q = 2;
        s->fk = 2;
        return 0;
    case 12:
        s->problem = make_problem(64, 64, 3, 300, 1, -1, 1, 1);
        axis(s, 2, 3, 1);
        s->xf = "wmma";
        s->xf_batch = 4;
        s->prefetch = true;
        s->m_f16 = true;
        return 0;
    case 13:
        s->problem = make_problem(128, 128, 11, 600, 1, -1, 1, 2);
        axis(s, 4, 11, 3);
        return 0;
    case 14:
        s->problem = make_problem(64, 64, 11, 600, 1, -1, 1, 1);
        axis(s, 6, 11, 2);
        s->xf = "wmma";
        return 0;
    case 15:
        s->problem = make_problem(128, 128, 11, 600, 1, -1, 1, 1);
        axis(s, 4, 11, 3);
        sel(s);
        *arch = "gfx950";
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

    rocke_winograd_fused_1d_spec_t spec;
    const char* arch = "gfx1250";
    if(make_cfg(idx, &spec, &arch) != 0)
    {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 2;
    }

    rocke_ir_builder_t b;
    rocke_kernel_def_t* kernel = rocke_build_winograd_fused_1d_new(&b, &spec, arch);
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
