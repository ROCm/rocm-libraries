/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/conv_direct_fwd_pp_mfma_emit.c -- C-side emitter for the gfx950
 * LDS-DMA ping-pong MFMA direct forward parity harness. Selects one of N
 * sampled spec configs by argv[1] (the config index), builds the
 * rocke_fwd_pp_mfma_spec_t identically to the Python emitter
 * conv_direct_fwd_pp_mfma_emit.py, builds the kernel via
 * rocke_build_fwd_pp_mfma_new and lowers via rocke_lower_kernel_to_llvm
 * (per-config arch, flavor AUTO), printing the .ll to stdout so the two outputs
 * can be byte-compared.
 *
 * Config index table: see the Python emitter (must stay in sync).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/instance_conv_direct_fwd_pp_mfma.h"
#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"
#include "rocke/verify.h"

/* The N=2, 20x24, C=cpg, K=96 "same"-padded stride-1 forward problem. */
static rocke_direct_conv_problem_t make_problem(int KH, const char* dtype, int cpg)
{
    rocke_direct_conv_problem_t p = rocke_direct_conv_problem_default();
    p.N = 2;
    p.H = 20;
    p.W = 24;
    p.groups = 1;
    p.cpg = cpg;
    p.kpg = 96;
    p.KH = KH;
    p.KW = KH;
    p.PAD = (KH - 1) / 2;
    p.stride = 1;
    p.dtype = dtype;
    return p;
}

static int make_cfg(int idx, rocke_fwd_pp_mfma_spec_t* s, const char** arch)
{
    *s = rocke_fwd_pp_mfma_spec_default();
    *arch = "gfx950";
    switch(idx)
    {
    case 0: /* bf16, 3x3, block_k 64 */
        s->problem = make_problem(3, "bf16", 72);
        s->block_k = 64;
        return 0;
    case 1: /* fp16, 3x3, block_k 64 */
        s->problem = make_problem(3, "fp16", 72);
        s->block_k = 64;
        return 0;
    case 2: /* bf16, 3x3, block_k 128 */
        s->problem = make_problem(3, "bf16", 72);
        return 0;
    case 3: /* fp16, 3x3, block_k 128 */
        s->problem = make_problem(3, "fp16", 72);
        return 0;
    case 4: /* bf16, 3x3, block_k 256 */
        s->problem = make_problem(3, "bf16", 72);
        s->block_k = 256;
        return 0;
    case 5: /* fp16, 3x3, block_k 256 */
        s->problem = make_problem(3, "fp16", 72);
        s->block_k = 256;
        return 0;
    case 6: /* bf16, 5x5, block_k 128 */
        s->problem = make_problem(5, "bf16", 72);
        return 0;
    case 7: /* fp16, 7x7, block_k 64 */
        s->problem = make_problem(7, "fp16", 72);
        s->block_k = 64;
        return 0;
    case 8: /* rejected by both engines: block_k 96 */
        s->problem = make_problem(3, "bf16", 72);
        s->block_k = 96;
        return 0;
    case 9: /* rejected by both engines: 5x5 at block_k 256 (LDS) */
        s->problem = make_problem(5, "bf16", 72);
        s->block_k = 256;
        return 0;
    case 10: /* rejected by both engines: gfx942 */
        s->problem = make_problem(3, "bf16", 72);
        *arch = "gfx942";
        return 0;
    case 11: /* rejected by both engines: stride 2 */
        s->problem = make_problem(3, "bf16", 72);
        s->problem.stride = 2;
        return 0;
    case 12: /* rejected by both engines: C = 12 */
        s->problem = make_problem(3, "bf16", 12);
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

    rocke_fwd_pp_mfma_spec_t spec;
    const char* arch = "gfx950";
    if(make_cfg(idx, &spec, &arch) != 0)
    {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 2;
    }

    rocke_ir_builder_t b;
    rocke_kernel_def_t* kernel = rocke_build_fwd_pp_mfma_new(&b, &spec, arch);
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
