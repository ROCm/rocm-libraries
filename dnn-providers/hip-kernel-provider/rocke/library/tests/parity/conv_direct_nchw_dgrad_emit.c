/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/conv_direct_nchw_dgrad_emit.c -- C-side emitter for the
 * native-NCHW direct dgrad convolution parity harness. Selects one of N
 * sampled spec configs by argv[1] (the config index), builds
 * rocke_direct_nchw_dgrad_spec_t identically to the Python emitter
 * conv_direct_nchw_dgrad_emit.py, builds the kernel via
 * rocke_build_direct_nchw_dgrad_new and lowers via rocke_lower_kernel_to_llvm
 * (gfx950, flavor AUTO), printing the .ll to stdout so the two outputs can be
 * byte-compared. Keep in lockstep with _CONFIGS in
 * conv_direct_nchw_dgrad_emit.py (same numbering, same coverage).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/instance_conv_direct_nchw_dgrad.h"
#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"
#include "rocke/verify.h"

static rocke_direct_conv_problem_t make_problem(int KH, int KW, int stride, int PAD,
                                                const char* dtype)
{
    rocke_direct_conv_problem_t p = rocke_direct_conv_problem_default();
    p.N = 2;
    p.H = 32;
    p.W = 32;
    p.groups = 1;
    p.cpg = 64;
    p.kpg = 64;
    p.KH = KH;
    p.KW = KW;
    p.PAD = PAD;
    p.stride = stride;
    p.dtype = dtype;
    return p;
}

/* Fill the config for index `idx`. Returns 0 on success, -1 if unknown. */
static int make_cfg(int idx, rocke_direct_nchw_dgrad_spec_t* spec)
{
    *spec = rocke_direct_nchw_dgrad_spec_default();
    switch(idx)
    {
    case 0: /* baseline fp16 3x3 s1 p1, default knobs */
        spec->problem = make_problem(3, 3, 1, 1, "fp16");
        break;
    case 1: /* bf16 3x3 s1 p1, default knobs */
        spec->problem = make_problem(3, 3, 1, 1, "bf16");
        break;
    case 2: /* fp16 3x3 s2 p1 */
        spec->problem = make_problem(3, 3, 2, 1, "fp16");
        break;
    case 3: /* fp16 5x5 s1 p2, default knobs (pad=2) */
        spec->problem = make_problem(5, 5, 1, 2, "fp16");
        break;
    case 4: /* fp16 5x5 s2 p2 */
        spec->problem = make_problem(5, 5, 2, 2, "fp16");
        break;
    case 5: /* fp16 3x3 s1 p0 (pad=0) */
        spec->problem = make_problem(3, 3, 1, 0, "fp16");
        break;
    case 6: /* fp16 3x3 s1 p1, lds_pad=0 */
        spec->problem = make_problem(3, 3, 1, 1, "fp16");
        spec->lds_pad = 0;
        break;
    case 7: /* fp16 3x3 s1 p1, stage_group=1 */
        spec->problem = make_problem(3, 3, 1, 1, "fp16");
        spec->stage_group = 1;
        break;
    case 8: /* fp16 3x3 s1 p1, stage_group=2 */
        spec->problem = make_problem(3, 3, 1, 1, "fp16");
        spec->stage_group = 2;
        break;
    case 9: /* fp16 3x3 s1 p1, stage_group=4 */
        spec->problem = make_problem(3, 3, 1, 1, "fp16");
        spec->stage_group = 4;
        break;
    case 10: /* fp16 3x3 s2 p1, stage_group=1 */
        spec->problem = make_problem(3, 3, 2, 1, "fp16");
        spec->stage_group = 1;
        break;
    case 11: /* bf16 5x5 s2 p2, stage_group=2 + lds_pad=0 combined */
        spec->problem = make_problem(5, 5, 2, 2, "bf16");
        spec->stage_group = 2;
        spec->lds_pad = 0;
        break;
    default:
        return -1;
    }
    return 0;
}

int main(int argc, char** argv)
{
    if(argc < 2)
    {
        fprintf(stderr, "usage: conv_direct_nchw_dgrad_emit <config_index> [ll|ir|verify]\n");
        return 2;
    }
    int idx = atoi(argv[1]);
    const char* mode = argc > 2 ? argv[2] : "ll";
    const char* arch = "gfx950";

    rocke_direct_nchw_dgrad_spec_t spec;
    if(make_cfg(idx, &spec) != 0)
    {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 2;
    }

    rocke_ir_builder_t b;
    rocke_kernel_def_t* kernel = rocke_build_direct_nchw_dgrad_new(&b, &spec, arch);
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
        char* t = NULL;
        rocke_status_t st = rocke_ir_serialize(kernel, &t);
        if(st != ROCKE_OK || !t)
        {
            fprintf(stderr, "ir_serialize failed: status=%d\n", (int)st);
            rocke_ir_builder_free(&b);
            return 1;
        }
        fputs(t, stdout);
        free(t);
    }
    else if(strcmp(mode, "verify") == 0)
    {
        rocke_diag_t* d = NULL;
        size_t n = 0;
        rocke_verify(kernel, &d, &n);
        for(size_t i = 0; i < n; i++)
        {
            char* s = rocke_diag_to_string(&d[i]);
            if(s)
            {
                puts(s);
                free(s);
            }
        }
        rocke_diags_free(d, n);
    }
    else
    {
        fprintf(stderr, "unknown mode %s\n", mode);
        rocke_ir_builder_free(&b);
        return 2;
    }
    rocke_ir_builder_free(&b);
    return ret;
}
