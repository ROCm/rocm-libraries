/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/conv_direct_fwd_wmma_emit.c -- C-side emitter for the wave32
 * WMMA direct forward parity harness. Selects one of N sampled spec configs by
 * argv[1] (the config index), builds the rocke_direct_fwd_wmma_spec_t
 * identically to the Python emitter conv_direct_fwd_wmma_emit.py, builds the
 * kernel via rocke_build_direct_fwd_wmma_new and lowers via
 * rocke_lower_kernel_to_llvm (per-config arch, flavor AUTO), printing the .ll
 * to stdout so the two outputs can be byte-compared.
 *
 * Config index table: see the Python emitter (must stay in sync).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/instance_conv_direct_fwd_wmma.h"
#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"
#include "rocke/verify.h"

/* make_problem(KH, PAD, stride, groups, cpg, kpg, dil_h, dil_w, dtype); the
 * Python emitter's _p() defaults are 3, 1, 1, 1, 64, 96, 1, 1, "bf16". */
static rocke_direct_conv_problem_t make_problem(int KH,
                                                int PAD,
                                                int stride,
                                                int groups,
                                                int cpg,
                                                int kpg,
                                                int dil_h,
                                                int dil_w,
                                                const char* dtype)
{
    rocke_direct_conv_problem_t p = rocke_direct_conv_problem_default();
    p.N = 2;
    p.H = 20;
    p.W = 24;
    p.groups = groups;
    p.cpg = cpg;
    p.kpg = kpg;
    p.KH = KH;
    p.KW = KH;
    p.PAD = PAD;
    p.stride = stride;
    p.dtype = dtype;
    p.dil_h = dil_h;
    p.dil_w = dil_w;
    return p;
}

static void knobs(
    rocke_direct_fwd_wmma_spec_t* s, int wk, int wp, int tk, int tp, int tw, int tn, int ck, int st)
{
    s->waves_k = wk;
    s->waves_p = wp;
    s->tiles_k = tk;
    s->tiles_p = tp;
    s->tile_w = tw;
    s->tile_n = tn;
    s->ck = ck;
    s->stages = st;
}

static void pf(rocke_direct_fwd_wmma_spec_t* s)
{
    s->prefetch = true;
    s->sched = "iterative-ilp";
}

static int make_cfg(int idx, rocke_direct_fwd_wmma_spec_t* s, const char** arch)
{
    *s = rocke_direct_fwd_wmma_spec_default();
    *arch = "gfx1250";
    switch(idx)
    {
    case 0:
        s->problem = make_problem(3, 1, 1, 1, 64, 96, 1, 1, "bf16");
        return 0;
    case 1:
        s->problem = make_problem(3, 1, 1, 1, 64, 96, 1, 1, "fp16");
        knobs(s, 2, 4, 2, 2, 8, 2, 32, 3);
        pf(s);
        return 0;
    case 2:
        s->problem = make_problem(3, 1, 1, 1, 64, 96, 1, 1, "bf16");
        knobs(s, 1, 4, 4, 6, 16, 2, 32, 2);
        s->split_barrier = true;
        s->wave_epilogue = true;
        pf(s);
        return 0;
    case 3:
        s->problem = make_problem(3, 1, 1, 1, 64, 96, 1, 1, "bf16");
        knobs(s, 1, 8, 6, 8, 32, 1, 32, 3);
        s->row_ring = true;
        return 0;
    case 4:
        s->problem = make_problem(3, 1, 1, 1, 64, 96, 1, 1, "fp16");
        knobs(s, 2, 4, 2, 4, 16, 1, 32, 3);
        s->tap_ring = true;
        return 0;
    case 5:
        s->problem = make_problem(3, 1, 2, 1, 64, 96, 1, 1, "bf16");
        knobs(s, 1, 4, 6, 2, 16, 1, 32, 2);
        pf(s);
        return 0;
    case 6:
        s->problem = make_problem(3, 1, 2, 1, 64, 96, 1, 1, "bf16");
        knobs(s, 2, 2, 1, 4, 16, 1, 64, 2);
        s->phase_split = true;
        pf(s);
        return 0;
    case 7:
        s->problem = make_problem(2, 0, 2, 1, 64, 96, 1, 1, "bf16");
        knobs(s, 1, 4, 2, 2, 16, 1, 64, 2);
        s->phase_split = true;
        return 0;
    case 8:
        s->problem = make_problem(3, 1, 1, 8, 4, 4, 1, 1, "bf16");
        knobs(s, 1, 4, 2, 4, 32, 1, 32, 2);
        s->pack_groups = 8;
        pf(s);
        return 0;
    case 9:
        s->problem = make_problem(3, 1, 2, 4, 8, 8, 1, 1, "fp16");
        knobs(s, 2, 4, 1, 2, 32, 2, 32, 2);
        s->pack_groups = 4;
        s->phase_split = true;
        pf(s);
        return 0;
    case 10:
        s->problem = make_problem(3, 2, 1, 1, 64, 96, 1, 2, "bf16");
        knobs(s, 1, 4, 2, 6, 16, 1, 64, 2);
        pf(s);
        return 0;
    case 11:
        s->problem = make_problem(3, 1, 2, 1, 64, 96, 3, 1, "bf16");
        knobs(s, 1, 4, 2, 2, 32, 2, 32, 2);
        pf(s);
        return 0;
    case 12:
        s->problem = make_problem(3, 1, 1, 1, 64, 96, 1, 1, "bf16");
        knobs(s, 2, 4, 4, 10, 16, 1, 32, 2);
        s->pingpong = true;
        return 0;
    case 13:
        s->problem = make_problem(3, 1, 1, 1, 64, 96, 1, 1, "fp16");
        knobs(s, 2, 4, 4, 6, 16, 1, 32, 2);
        s->pingpong = true;
        s->pp_group = 3;
        s->sched = "iterative-ilp";
        return 0;
    case 14:
        s->problem = make_problem(3, 1, 1, 1, 64, 96, 1, 1, "bf16");
        knobs(s, 1, 4, 4, 4, 16, 1, 32, 2);
        s->tap_reuse = true;
        pf(s);
        return 0;
    case 15:
        s->problem = make_problem(3, 1, 1, 1, 64, 96, 1, 1, "bf16");
        knobs(s, 1, 4, 2, 4, 32, 1, 32, 2);
        s->tap_reuse = true;
        s->tap_shift = true;
        return 0;
    case 16:
        s->problem = make_problem(3, 2, 1, 1, 64, 96, 1, 2, "bf16");
        knobs(s, 1, 4, 2, 4, 16, 1, 32, 2);
        s->tap_reuse = true;
        s->tap_shift = true;
        s->split_barrier = true;
        return 0;
    case 17:
        s->problem = make_problem(3, 1, 2, 1, 64, 96, 1, 1, "bf16");
        knobs(s, 1, 4, 2, 2, 16, 1, 32, 3);
        s->phase_split = true;
        s->row_ring = true;
        return 0;
    case 18:
        s->problem = make_problem(3, 1, 2, 1, 64, 96, 1, 1, "bf16");
        knobs(s, 2, 4, 2, 2, 16, 1, 32, 2);
        s->phase_split = true;
        s->pingpong = true;
        s->pp_group = 2;
        return 0;
    case 19:
        s->problem = make_problem(3, 1, 1, 1, 64, 96, 1, 1, "bf16");
        *arch = "gfx950";
        return 0;
    case 20:
        s->problem = make_problem(3, 1, 3, 1, 64, 96, 1, 1, "bf16");
        return 0;
    case 21:
        s->problem = make_problem(5, 2, 1, 1, 64, 96, 1, 1, "bf16");
        knobs(s, 1, 4, 2, 4, 16, 1, 32, 2);
        s->prefetch = true;
        s->sched = "max-ilp";
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

    rocke_direct_fwd_wmma_spec_t spec;
    const char* arch = "gfx1250";
    if(make_cfg(idx, &spec, &arch) != 0)
    {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 2;
    }

    rocke_ir_builder_t b;
    rocke_kernel_def_t* kernel = rocke_build_direct_fwd_wmma_new(&b, &spec, arch);
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
