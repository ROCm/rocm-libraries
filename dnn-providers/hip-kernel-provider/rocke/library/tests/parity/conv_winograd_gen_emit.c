/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/conv_winograd_gen_emit.c -- C-side emitter for the generalized
 * Winograd forward transform kernels (conv_winograd_gen). Selects one config
 * by argv[1], builds the rocke_winograd_gen_spec_t identically to the Python
 * emitter conv_winograd_gen_emit.py, builds the stage via
 * rocke_build_winograd_gen_stage_new and prints the .ll (or ir / verify).
 *
 * Config index table: see the Python emitter (must stay in sync).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/instance_conv_winograd_gen.h"
#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"
#include "rocke/verify.h"

static rocke_wino_axis_t ax(int m, int r, int seg)
{
    rocke_wino_axis_t a;
    a.m = m;
    a.r = r;
    a.seg = seg;
    return a;
}

static rocke_conv_problem_t prob(int N, int Hi, int Wi, int C, int K, int Y, int X)
{
    rocke_conv_problem_t p = rocke_conv_problem_default(N, Hi, Wi, C, K, Y, X);
    p.groups = 1;
    return p;
}

static const char* const STAGES[3] = {"K1", "K2", "K4"};

/* Fill case i (see the Python _case); returns 0, or -1 past the last case. */
static int make_case(int i, rocke_winograd_gen_spec_t* s, const char** arch)
{
    *s = rocke_winograd_gen_spec_default();
    *arch = "gfx1250";
    switch(i)
    {
    case 0:
        s->problem = prob(2, 1, 200, 64, 64, 1, 11);
        s->problem.pW = 5;
        s->ax_x = ax(6, 11, 2);
        return 0;
    case 1:
        s->problem = prob(2, 1, 150, 64, 96, 1, 11);
        s->problem.pW = 5;
        s->ax_x = ax(4, 11, 3);
        s->io_dtype = "bf16";
        s->vec = 8;
        return 0;
    case 2:
        s->problem = prob(1, 1, 100, 32, 64, 1, 7);
        s->problem.pW = 6;
        s->problem.dW = 2;
        s->ax_x = ax(4, 7, 1);
        s->nt_stores = false;
        return 0;
    case 3:
        s->problem = prob(1, 1, 90, 128, 64, 1, 7);
        s->problem.pW = 3;
        s->ax_x = ax(4, 7, 2);
        s->block_size = 128;
        *arch = "gfx950";
        return 0;
    case 4:
        s->problem = prob(1, 2, 100, 128, 128, 1, 3);
        s->problem.pW = 1;
        s->ax_x = ax(6, 3, 1);
        return 0;
    case 5:
        s->problem = prob(1, 50, 4, 64, 64, 7, 1);
        s->problem.pH = 3;
        s->ax_y = ax(4, 7, 1);
        return 0;
    case 6:
        s->problem = prob(1, 20, 20, 64, 64, 5, 5);
        s->problem.pH = 2;
        s->problem.pW = 2;
        s->problem.groups = 2;
        s->ax_y = ax(2, 5, 1);
        s->ax_x = ax(2, 5, 1);
        return 0;
    case 7:
        s->problem = prob(1, 24, 22, 128, 64, 5, 5);
        s->problem.pH = 4;
        s->problem.pW = 4;
        s->problem.dH = 2;
        s->problem.dW = 2;
        s->problem.groups = 4;
        s->ax_y = ax(3, 5, 1);
        s->ax_x = ax(3, 5, 1);
        s->io_dtype = "bf16";
        *arch = "gfx950";
        return 0;
    case 8:
        s->problem = prob(1, 1, 120, 33, 33, 1, 11);
        s->problem.pW = 15;
        s->problem.dW = 3;
        s->ax_x = ax(6, 11, 2);
        return 0;
    case 9:
        s->problem = prob(3, 1, 40, 64, 32, 1, 3);
        s->problem.pW = 1;
        s->ax_x = ax(2, 3, 1);
        s->vec = 2;
        return 0;
    default:
        return -1;
    }
}

static int make_cfg(int idx, rocke_winograd_gen_spec_t* s, const char** stage, const char** arch)
{
    *s = rocke_winograd_gen_spec_default();
    *arch = "gfx1250";
    switch(idx)
    {
    case 30:
        s->problem = prob(1, 1, 64, 64, 64, 1, 3);
        s->problem.pW = 1;
        s->problem.sW = 2;
        s->ax_x = ax(2, 3, 1);
        *stage = "K1";
        return 0;
    case 31:
        s->problem = prob(1, 1, 64, 64, 64, 1, 11);
        s->problem.pW = 5;
        s->ax_x = ax(4, 11, 1);
        *stage = "K2";
        return 0;
    case 32:
        s->problem = prob(1, 20, 20, 64, 64, 5, 5);
        s->problem.pH = 2;
        s->problem.pW = 2;
        s->ax_y = ax(3, 5, 2);
        s->ax_x = ax(2, 5, 1);
        *stage = "K4";
        return 0;
    default:
        break;
    }
    if(idx < 0 || make_case(idx / 3, s, arch) != 0)
        return -1;
    *stage = STAGES[idx % 3];
    return 0;
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

    rocke_winograd_gen_spec_t spec;
    const char* stage = "K1";
    const char* arch = "gfx1250";
    if(make_cfg(idx, &spec, &stage, &arch) != 0)
    {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 2;
    }

    rocke_ir_builder_t b;
    rocke_kernel_def_t* kernel = rocke_build_winograd_gen_stage_new(&b, &spec, stage, arch);
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
