/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/conv_direct_dgrad_wmma_emit.c -- C-side emitter for the wave32
 * WMMA direct dgrad parity harness. Selects one of N sampled spec configs by
 * argv[1] (the config index), builds the rocke_direct_dgrad_wmma_spec_t
 * identically to the Python emitter conv_direct_dgrad_wmma_emit.py, builds the
 * kernel via rocke_build_direct_dgrad_wmma_new and lowers via
 * rocke_lower_kernel_to_llvm (per-config arch, flavor AUTO), printing the .ll
 * to stdout so the two outputs can be byte-compared.
 *
 * Config index table: see the Python emitter (must stay in sync).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/instance_conv_direct_dgrad_wmma.h"
#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"
#include "rocke/verify.h"

static rocke_direct_conv_problem_t make_problem(int KH, int PAD, const char* dtype, int stride)
{
    rocke_direct_conv_problem_t p = rocke_direct_conv_problem_default();
    p.N = 2;
    p.H = 20;
    p.W = 24;
    p.groups = 1;
    p.cpg = 64;
    p.kpg = 96;
    p.KH = KH;
    p.KW = KH;
    p.PAD = PAD;
    p.stride = stride;
    p.dtype = dtype;
    return p;
}

static int make_cfg(int idx, rocke_direct_dgrad_wmma_spec_t* s, const char** arch)
{
    *s = rocke_direct_dgrad_wmma_spec_default();
    *arch = "gfx1250";
    switch(idx)
    {
    case 0:
        s->problem = make_problem(3, 1, "bf16", 1);
        return 0;
    case 1:
        s->problem = make_problem(3, 1, "fp16", 1);
        s->waves_c = 2;
        s->waves_p = 4;
        s->tiles_c = 2;
        s->tiles_p = 2;
        s->tile_w = 8;
        s->tile_n = 2;
        s->stages = 3;
        s->prefetch = true;
        s->sched = "iterative-ilp";
        return 0;
    case 2:
        s->problem = make_problem(3, 0, "bf16", 1);
        s->waves_c = 1;
        s->waves_p = 8;
        s->tiles_c = 1;
        s->tiles_p = 2;
        s->tile_w = 4;
        s->tile_n = 4;
        return 0;
    case 3:
        s->problem = make_problem(5, 2, "bf16", 1);
        s->waves_c = 1;
        s->waves_p = 4;
        s->tiles_c = 4;
        s->tiles_p = 6;
        s->tile_w = 32;
        s->prefetch = true;
        s->sched = "max-ilp";
        return 0;
    case 4:
        s->problem = make_problem(3, 1, "fp16", 1);
        s->waves_c = 1;
        s->waves_p = 2;
        s->tiles_c = 1;
        s->tiles_p = 1;
        s->stages = 3;
        return 0;
    case 5:
        s->problem = make_problem(3, 1, "bf16", 1);
        s->tiles_c = 4;
        s->tiles_p = 8;
        s->tile_w = 64;
        s->prefetch = true;
        return 0;
    case 6:
        s->problem = make_problem(3, 1, "bf16", 1);
        *arch = "gfx950";
        return 0;
    case 7:
        s->problem = make_problem(3, 1, "bf16", 2);
        return 0;
    case 8:
        s->problem = make_problem(3, 1, "bf16", 1);
        s->waves_c = 1;
        s->waves_p = 4;
        s->tiles_c = 4;
        s->tiles_p = 4;
        s->ck = 64;
        s->prefetch = true;
        s->sched = "iterative-ilp";
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

    rocke_direct_dgrad_wmma_spec_t spec;
    const char* arch = "gfx1250";
    if(make_cfg(idx, &spec, &arch) != 0)
    {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 2;
    }

    rocke_ir_builder_t b;
    rocke_kernel_def_t* kernel = rocke_build_direct_dgrad_wmma_new(&b, &spec, arch);
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
