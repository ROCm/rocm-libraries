/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/conv_wgrad_workspace_reduce_emit.c -- C-side emitter for the
 * wgrad workspace-reduce (Stage 2) parity harness.  Selects one of N sampled
 * spec configs by argv[1] (the config index), builds the
 * rocke_wgrad_reduce_spec_t identically to the Python emitter
 * conv_wgrad_workspace_reduce_emit.py, builds the kernel via
 * rocke_build_wgrad_workspace_reduce_new and lowers via
 * rocke_lower_kernel_to_llvm (per-config arch, flavor AUTO), printing the .ll
 * to stdout so the two outputs can be byte-compared.
 *
 * Config index table (must stay in sync with the Python emitter):
 *   0  fp16 output, wg_M=64, wg_N=576 (3x3, C=64, K=64), gfx950
 *   1  bf16 output, wg_M=64, wg_N=576, gfx950
 *   2  fp32 output, wg_M=64, wg_N=576, gfx950
 *   3  fp16 output, wg_M=32, wg_N=72  (3x3, C=8, K=32), gfx950
 *   4  fp16 output, wg_M=64, wg_N=576, gfx942
 *   5  bf16 output, wg_M=32, wg_N=72, gfx950
 *   6  fp16 output, custom tile_m=8, tile_n=32, wg_M=64, wg_N=576, gfx950
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/instance_conv_wgrad_workspace_reduce.h"
#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"
#include "rocke/verify.h"

/* Fill the spec for config index `idx`.
 * Returns 0 on success, -1 if unknown.
 * wg_M = K, wg_N = Y * X * C.
 * Python uses Y=3, X=3 when wg_N % 9 == 0; otherwise Y=1, X=1, C=wg_N.
 * problem_short must survive for the lifetime of the caller; use a static
 * buffer per case. */
static int make_cfg(int idx, rocke_wgrad_reduce_spec_t* spec, const char** arch)
{
    *spec = rocke_wgrad_reduce_spec_default();
    *arch = "gfx950";

    switch(idx)
    {
    case 0:
        spec->wg_M = 64;
        spec->wg_N = 576;
        spec->dtype_d = "fp16";
        spec->problem_short = "N1H4W4C64_K64Y3X3";
        return 0;
    case 1:
        spec->wg_M = 64;
        spec->wg_N = 576;
        spec->dtype_d = "bf16";
        spec->problem_short = "N1H4W4C64_K64Y3X3";
        return 0;
    case 2:
        spec->wg_M = 64;
        spec->wg_N = 576;
        spec->dtype_d = "fp32";
        spec->problem_short = "N1H4W4C64_K64Y3X3";
        return 0;
    case 3:
        spec->wg_M = 32;
        spec->wg_N = 72;
        spec->dtype_d = "fp16";
        spec->problem_short = "N1H4W4C8_K32Y3X3";
        return 0;
    case 4:
        spec->wg_M = 64;
        spec->wg_N = 576;
        spec->dtype_d = "fp16";
        spec->problem_short = "N1H4W4C64_K64Y3X3";
        *arch = "gfx942";
        return 0;
    case 5:
        spec->wg_M = 32;
        spec->wg_N = 72;
        spec->dtype_d = "bf16";
        spec->problem_short = "N1H4W4C8_K32Y3X3";
        return 0;
    case 6:
        spec->wg_M = 64;
        spec->wg_N = 576;
        spec->dtype_d = "fp16";
        spec->tile_m = 8;
        spec->tile_n = 32;
        spec->problem_short = "N1H4W4C64_K64Y3X3";
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

    rocke_wgrad_reduce_spec_t spec;
    const char* arch = "gfx950";
    if(make_cfg(idx, &spec, &arch) != 0)
    {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 2;
    }

    rocke_ir_builder_t b;
    rocke_kernel_def_t* kernel = rocke_build_wgrad_workspace_reduce_new(&b, &spec, arch);
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
        char** msgs = NULL;
        int nmsg = 0;
        rocke_verify(kernel, &msgs, &nmsg);
        for(int i = 0; i < nmsg; i++)
        {
            fputs(msgs[i], stdout);
            fputc('\n', stdout);
            free(msgs[i]);
        }
        free(msgs);
    }
    else
    {
        fprintf(stderr, "unknown mode '%s' (expected ll|ir|verify)\n", mode);
        ret = 2;
    }

    rocke_ir_builder_free(&b);
    return ret;
}
