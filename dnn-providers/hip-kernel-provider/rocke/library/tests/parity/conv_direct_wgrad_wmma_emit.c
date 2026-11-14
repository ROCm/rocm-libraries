/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/conv_direct_wgrad_wmma_emit.c -- C-side emitter for the wave32
 * WMMA dense direct wgrad parity harness. Selects one of N sampled spec
 * configs by argv[1] (the config index), builds the
 * rocke_direct_wgrad_wmma_spec_t identically to the Python emitter
 * conv_direct_wgrad_wmma_emit.py, builds the kernel via
 * rocke_build_direct_wgrad_wmma_new and lowers via rocke_lower_kernel_to_llvm
 * (per-config arch, flavor AUTO), printing the .ll to stdout so the two
 * outputs can be byte-compared.
 *
 * Config index table: see the Python emitter (must stay in sync).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/instance_conv_direct_wgrad_wmma.h"
#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"
#include "rocke/verify.h"

/* N=2 H=12 W=40, groups 1 unless set, C / K the channel totals. */
static rocke_direct_conv_problem_t make_problem(
    int KH, int KW, int PAD, const char* dtype, int stride, int groups, int C, int K, int dil)
{
    rocke_direct_conv_problem_t p = rocke_direct_conv_problem_default();
    p.N = 2;
    p.H = 12;
    p.W = 40;
    p.groups = groups;
    p.cpg = C / groups;
    p.kpg = K / groups;
    p.KH = KH;
    p.KW = KW;
    p.PAD = PAD;
    p.stride = stride;
    p.dtype = dtype;
    p.dil_h = dil;
    p.dil_w = dil;
    return p;
}

static rocke_direct_conv_problem_t base(int PAD, const char* dtype)
{
    return make_problem(3, 3, PAD, dtype, 1, 1, 64, 96, 1);
}

/* The shipped (AOT) knob set; waves and target_blocks vary per job. */
static void aot(rocke_direct_wgrad_wmma_spec_t* s, int wk, int wc, int tb)
{
    s->waves_k = wk;
    s->waves_c = wc;
    s->target_blocks = tb;
    s->tiles_k = 2;
    s->tiles_c = 2;
    s->stages = 4;
    s->rows = 2;
    s->prefetch = true;
    s->sched = true;
    s->pad = 16;
    s->xcd = 8;
    s->tdm = true;
}

static int make_cfg(int idx, rocke_direct_wgrad_wmma_spec_t* s, const char** arch)
{
    *s = rocke_direct_wgrad_wmma_spec_default();
    *arch = "gfx1250";
    switch(idx)
    {
    case 0: /* bf16, AOT, waves 4x2, target_blocks 256 */
        s->problem = base(1, "bf16");
        aot(s, 4, 2, 256);
        return 0;
    case 1: /* fp16, AOT, waves 2x4, target_blocks 512 */
        s->problem = base(1, "fp16");
        aot(s, 2, 4, 512);
        return 0;
    case 2: /* bf16, AOT, PAD 0, waves 2x4, target_blocks 256 */
        s->problem = base(0, "bf16");
        aot(s, 2, 4, 256);
        return 0;
    case 3: /* fp16, AOT, PAD 0, waves 4x2, target_blocks 512 */
        s->problem = base(0, "fp16");
        aot(s, 4, 2, 512);
        return 0;
    case 4: /* bf16, defaults */
        s->problem = base(1, "bf16");
        return 0;
    case 5: /* fp16, per-lane copies, prefetch (rows 1), q_steps 2, tiles 1x1, 2 stages */
        s->problem = base(1, "fp16");
        s->tiles_k = 1;
        s->tiles_c = 1;
        s->q_steps = 2;
        s->stages = 2;
        s->prefetch = true;
        return 0;
    case 6: /* bf16, per-lane copies, rows 2, prefetch, 3 stages, sched, xcd 8, pad 16 */
        s->problem = base(1, "bf16");
        s->rows = 2;
        s->stages = 3;
        s->prefetch = true;
        s->sched = true;
        s->xcd = 8;
        s->pad = 16;
        return 0;
    case 7: /* bf16, TDM, rows 1, no prefetch, waves 2x2, tiles 4x1 */
        s->problem = base(1, "bf16");
        s->waves_k = 2;
        s->waves_c = 2;
        s->tiles_k = 4;
        s->tiles_c = 1;
        s->tdm = true;
        return 0;
    case 8: /* fp16, TDM, PAD 0, q_steps 2, tiles 1x4, 2 stages */
        s->problem = base(0, "fp16");
        s->tiles_k = 1;
        s->tiles_c = 4;
        s->q_steps = 2;
        s->stages = 2;
        s->tdm = true;
        return 0;
    case 9: /* bf16, TDM, 16 waves, rows 4, prefetch, 3 stages, target_blocks 64 */
        s->problem = base(1, "bf16");
        s->waves_k = 4;
        s->waves_c = 4;
        s->rows = 4;
        s->stages = 3;
        s->prefetch = true;
        s->target_blocks = 64;
        s->tdm = true;
        return 0;
    case 10: /* fp16, per-lane copies, 2 waves (1x2), tiles 2x1, sched, target_blocks 1 */
        s->problem = base(1, "fp16");
        s->waves_k = 1;
        s->waves_c = 2;
        s->tiles_k = 2;
        s->tiles_c = 1;
        s->sched = true;
        s->target_blocks = 1;
        return 0;
    case 11: /* bf16, TDM, 2 waves (2x1), rows 3, prefetch, 3 stages */
        s->problem = base(1, "bf16");
        s->waves_k = 2;
        s->waves_c = 1;
        s->rows = 3;
        s->stages = 3;
        s->prefetch = true;
        s->tdm = true;
        return 0;
    case 12: /* fp16, 5x5 PAD 2, TDM, waves 2x2, tiles 1x1, prefetch */
        s->problem = make_problem(5, 5, 2, "fp16", 1, 1, 64, 96, 1);
        s->waves_k = 2;
        s->waves_c = 2;
        s->tiles_k = 1;
        s->tiles_c = 1;
        s->prefetch = true;
        s->tdm = true;
        return 0;
    case 13: /* bf16, 1x3 PAD 0 (no dY ring), per-lane copies, xcd 3 */
        s->problem = make_problem(1, 3, 0, "bf16", 1, 1, 64, 96, 1);
        s->xcd = 3;
        return 0;
    case 14: /* fp16, TDM, rows 2, no prefetch, 2 stages, sched, pad 8, xcd 0 */
        s->problem = base(1, "fp16");
        s->rows = 2;
        s->stages = 2;
        s->sched = true;
        s->tdm = true;
        return 0;
    case 15: /* rejected: gfx950 (AOT knobs) */
        s->problem = base(1, "bf16");
        aot(s, 4, 2, 256);
        *arch = "gfx950";
        return 0;
    case 16: /* rejected: stride 2 */
        s->problem = make_problem(3, 3, 1, "bf16", 2, 1, 64, 96, 1);
        return 0;
    case 17: /* rejected: groups 2 */
        s->problem = make_problem(3, 3, 1, "bf16", 1, 2, 64, 96, 1);
        return 0;
    case 18: /* rejected: tiles 4x4 (too many VGPRs) */
        s->problem = base(1, "bf16");
        s->tiles_k = 4;
        s->tiles_c = 4;
        return 0;
    case 19: /* rejected: TDM rows 3 on 4 waves */
        s->problem = base(1, "bf16");
        s->waves_k = 2;
        s->waves_c = 2;
        s->rows = 3;
        s->tdm = true;
        return 0;
    case 20: /* rejected: rows 2 with prefetch on 2 stages */
        s->problem = base(1, "bf16");
        s->rows = 2;
        s->stages = 2;
        s->prefetch = true;
        return 0;
    case 21: /* rejected: pad 4 */
        s->problem = base(1, "bf16");
        s->pad = 4;
        return 0;
    case 22: /* rejected: dilation 2x2 */
        s->problem = make_problem(3, 3, 2, "bf16", 1, 1, 64, 96, 2);
        return 0;
    case 23: /* rejected: PAD 3 on a 3x3 filter */
        s->problem = base(3, "bf16");
        return 0;
    case 24: /* rejected: C 36 */
        s->problem = make_problem(3, 3, 1, "bf16", 1, 1, 36, 96, 1);
        return 0;
    case 25: /* rejected: LDS past 300 KiB */
        s->problem = base(1, "bf16");
        s->waves_k = 4;
        s->waves_c = 4;
        s->rows = 4;
        s->stages = 8;
        return 0;
    case 26: /* rejected: PAD 1 with column padding 0 */
        s->problem = base(1, "bf16");
        s->problem.pad_w = 0;
        return 0;
    case 27: /* rejected: 3x1 filter with PAD 1 */
        s->problem = make_problem(3, 1, 1, "bf16", 1, 1, 64, 96, 1);
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

    rocke_direct_wgrad_wmma_spec_t spec;
    const char* arch = "gfx1250";
    if(make_cfg(idx, &spec, &arch) != 0)
    {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 2;
    }

    rocke_ir_builder_t b;
    rocke_kernel_def_t* kernel = rocke_build_direct_wgrad_wmma_new(&b, &spec, arch);
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
