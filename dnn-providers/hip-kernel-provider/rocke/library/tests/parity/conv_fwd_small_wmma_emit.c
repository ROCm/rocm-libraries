/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/conv_fwd_small_wmma_emit.c -- C-side emitter for the wave32
 * WMMA forward of tiny channel counts (fwd_smallc_wmma and fwd_smallk_wmma).
 * Selects one of N sampled spec configs by argv[1] (the config index), builds
 * the rocke_fwd_small{c,k}_wmma_spec_t identically to the Python emitter
 * conv_fwd_small_wmma_emit.py, builds the kernel via
 * rocke_build_fwd_small{c,k}_wmma_new and lowers via
 * rocke_lower_kernel_to_llvm (per-config arch, flavor AUTO), printing the .ll
 * to stdout so the two outputs can be byte-compared.
 *
 * Config index table: see the Python emitter (must stay in sync).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/instance_conv_fwd_small_wmma.h"
#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"
#include "rocke/verify.h"

static rocke_direct_conv_problem_t
    make_problem(int K, int PAD, int S, int C, int Kout, const char* dtype, int H, int W)
{
    rocke_direct_conv_problem_t p = rocke_direct_conv_problem_default();
    p.N = 2;
    p.H = H;
    p.W = W;
    p.groups = 1;
    p.cpg = C;
    p.kpg = Kout;
    p.KH = K;
    p.KW = K;
    p.PAD = PAD;
    p.stride = S;
    p.dtype = dtype;
    return p;
}

/* Returns 0 for a small-C config, 1 for a small-K config, -1 if unknown. */
static int make_cfg(int idx,
                    rocke_fwd_smallc_wmma_spec_t* c,
                    rocke_fwd_smallk_wmma_spec_t* k,
                    const char** arch)
{
    *c = rocke_fwd_smallc_wmma_spec_default();
    *k = rocke_fwd_smallk_wmma_spec_default();
    *arch = "gfx1250";
    switch(idx)
    {
    case 0:
        c->problem = make_problem(3, 1, 1, 3, 64, "bf16", 32, 40);
        return 0;
    case 1:
        c->problem = make_problem(3, 1, 1, 4, 64, "fp16", 32, 40);
        c->waves_k = 2;
        c->waves_p = 2;
        c->tiles_k = 2;
        c->tiles_p = 4;
        c->out_h = 8;
        c->out_w = 16;
        c->tdm_store = true;
        return 0;
    case 2:
        c->problem = make_problem(3, 1, 2, 3, 64, "bf16", 32, 40);
        c->waves_k = 1;
        c->waves_p = 2;
        c->tiles_k = 4;
        c->tiles_p = 4;
        c->out_h = 8;
        c->out_w = 16;
        c->tiles_per_block = 4;
        c->tdm_store = true;
        return 0;
    case 3:
        c->problem = make_problem(7, 3, 2, 3, 64, "bf16", 32, 40);
        c->waves_k = 1;
        c->waves_p = 1;
        c->tiles_k = 4;
        c->tiles_p = 4;
        c->out_h = 2;
        c->out_w = 32;
        c->tiles_per_block = 2;
        c->sched = "iterative-ilp";
        return 0;
    case 4:
        c->problem = make_problem(4, 0, 4, 3, 64, "fp16", 32, 40);
        c->waves_k = 1;
        c->waves_p = 2;
        c->tiles_k = 8;
        c->tiles_p = 4;
        c->out_h = 4;
        c->out_w = 32;
        c->tiles_per_block = 4;
        c->tdm_store = true;
        return 0;
    case 5:
        c->problem = make_problem(14, 0, 14, 3, 64, "bf16", 224, 224);
        c->waves_k = 2;
        c->waves_p = 2;
        c->tiles_k = 2;
        c->tiles_p = 4;
        c->out_h = 4;
        c->out_w = 32;
        return 0;
    case 6:
        c->problem = make_problem(16, 0, 16, 3, 64, "fp16", 224, 224);
        c->waves_k = 2;
        c->waves_p = 2;
        c->tiles_k = 2;
        c->tiles_p = 2;
        c->out_h = 4;
        c->out_w = 16;
        c->sched = "max-ilp";
        return 0;
    case 7:
        c->problem = make_problem(3, 1, 1, 3, 64, "bf16", 32, 40);
        c->name = "rocke_fwd_smallc_wmma";
        c->tiles_per_block = 3;
        return 0;
    case 8:
        k->problem = make_problem(3, 1, 1, 128, 4, "bf16", 32, 40);
        k->waves_c = 1;
        k->waves_p = 4;
        k->tiles_c = 3;
        k->tiles_p = 3;
        k->out_h = 6;
        k->out_w = 16;
        k->ck = 64;
        k->prefetch = true;
        return 1;
    case 9:
        k->problem = make_problem(3, 1, 1, 128, 4, "fp16", 32, 40);
        k->waves_c = 3;
        k->waves_p = 2;
        k->tiles_c = 1;
        k->tiles_p = 5;
        k->out_h = 10;
        k->out_w = 10;
        k->ck = 64;
        k->stages = 3;
        k->prefetch = true;
        k->sched = "max-ilp";
        return 1;
    case 10:
        k->problem = make_problem(3, 1, 1, 64, 8, "bf16", 32, 40);
        k->tiles_c = 5;
        return 1;
    case 11:
        k->problem = make_problem(3, 0, 2, 64, 2, "fp16", 32, 40);
        k->waves_c = 1;
        k->waves_p = 2;
        k->tiles_c = 2;
        k->tiles_p = 3;
        k->out_h = 4;
        k->out_w = 4;
        k->stages = 4;
        return 1;
    case 12:
        k->problem = make_problem(1, 0, 1, 96, 1, "bf16", 32, 40);
        k->waves_c = 1;
        k->waves_p = 2;
        k->tiles_c = 1;
        k->tiles_p = 2;
        k->out_h = 8;
        k->out_w = 8;
        k->ck = 96;
        k->sched = "iterative-ilp";
        return 1;
    case 13:
        c->problem = make_problem(3, 1, 1, 3, 64, "bf16", 32, 40);
        *arch = "gfx950";
        return 0;
    case 14:
        k->problem = make_problem(3, 1, 1, 36, 4, "bf16", 32, 40);
        return 1;
    case 15:
        k->problem = make_problem(3, 1, 1, 64, 3, "bf16", 32, 40);
        k->waves_c = 1;
        k->waves_p = 3;
        k->tiles_c = 2;
        k->tiles_p = 3;
        k->out_h = 6;
        k->out_w = 16;
        k->ck = 32;
        k->prefetch = true;
        return 1;
    case 16:
        k->problem = make_problem(3, 1, 1, 64, 3, "fp16", 32, 40);
        k->waves_c = 1;
        k->waves_p = 4;
        k->tiles_c = 2;
        k->tiles_p = 4;
        k->out_h = 6;
        k->out_w = 30;
        k->ck = 64;
        k->prefetch = true;
        return 1;
    case 17:
        k->problem = make_problem(1, 0, 1, 32, 1, "bf16", 1, 300);
        k->problem.KW = 7;
        k->problem.pad_w = 3;
        k->waves_c = 1;
        k->waves_p = 2;
        k->tiles_c = 1;
        k->tiles_p = 2;
        k->out_h = 1;
        k->out_w = 58;
        k->ck = 32;
        k->prefetch = true;
        return 1;
    case 18:
        k->problem = make_problem(1, 0, 1, 32, 1, "fp16", 1, 300);
        k->problem.KW = 7;
        k->problem.pad_w = 3;
        k->waves_c = 1;
        k->waves_p = 2;
        k->tiles_c = 1;
        k->tiles_p = 4;
        k->out_h = 1;
        k->out_w = 122;
        k->ck = 32;
        return 1;
    case 19:
        k->problem = make_problem(3, 1, 1, 64, 5, "bf16", 32, 40);
        k->waves_c = 1;
        k->waves_p = 3;
        k->tiles_c = 3;
        k->tiles_p = 3;
        k->out_h = 6;
        k->out_w = 16;
        k->ck = 32;
        return 1;
    case 20:
        k->problem = make_problem(3, 1, 1, 64, 6, "fp16", 32, 40);
        k->waves_c = 1;
        k->waves_p = 3;
        k->tiles_c = 4;
        k->tiles_p = 3;
        k->out_h = 6;
        k->out_w = 16;
        k->ck = 32;
        return 1;
    case 21:
        k->problem = make_problem(1, 0, 1, 32, 1, "bf16", 1, 300);
        k->problem.KW = 7;
        k->problem.pad_w = 7;
        return 1;
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

    rocke_fwd_smallc_wmma_spec_t cspec;
    rocke_fwd_smallk_wmma_spec_t kspec;
    const char* arch = "gfx1250";
    int which = make_cfg(idx, &cspec, &kspec, &arch);
    if(which < 0)
    {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 2;
    }

    rocke_ir_builder_t b;
    rocke_kernel_def_t* kernel = (which == 0) ? rocke_build_fwd_smallc_wmma_new(&b, &cspec, arch)
                                              : rocke_build_fwd_smallk_wmma_new(&b, &kspec, arch);
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
