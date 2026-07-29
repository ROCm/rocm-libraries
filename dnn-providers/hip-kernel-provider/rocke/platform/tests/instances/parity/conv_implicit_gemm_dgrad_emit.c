/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/conv_implicit_gemm_dgrad_emit.c -- C-side emitter for the
 * implicit-GEMM backward-data convolution parity harness.  Selects one of N
 * sampled spec configs by argv[1] (the config index), builds
 * rocke_dgrad_conv_spec_t identically to the Python emitter
 * conv_implicit_gemm_dgrad_emit.py, builds the kernel via
 * rocke_build_implicit_gemm_conv_dgrad_new and lowers via
 * rocke_lower_kernel_to_llvm (per-config arch, flavor AUTO), printing the .ll
 * to stdout so the two outputs can be byte-compared.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/instance_conv_implicit_gemm_dgrad.h"
#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"
#include "rocke/verify.h"

/* Fill the config for index `idx`.  Returns 0 on success, -1 if unknown.
 * On success sets *spec and *arch. */
static int make_cfg(int idx, rocke_dgrad_conv_spec_t* spec, const char** arch)
{
    *spec = rocke_dgrad_conv_spec_default();
    spec->tile_m = 64;
    spec->tile_n = 64;
    spec->tile_k = 64;
    spec->warp_m = 2;
    spec->warp_n = 2;
    spec->warp_tile_m = 32;
    spec->warp_tile_n = 32;
    spec->warp_tile_k = 16;
    spec->pipeline = "mem";
    spec->epilogue = "default";

    switch(idx)
    {
    case 0:
        /* Baseline stride=1, default tile geometry, gfx950 */
        spec->problem = rocke_conv_problem_make(8, 56, 56, 64, 64, 3, 3, 1, 1, 1, 1, 1, 1);
        *arch = "gfx950";
        return 0;
    case 1:
        /* Larger tile, compv4 pipeline, gfx950 */
        spec->problem = rocke_conv_problem_make(8, 56, 56, 64, 64, 3, 3, 1, 1, 1, 1, 1, 1);
        spec->tile_m = 128;
        spec->tile_n = 128;
        spec->tile_k = 64;
        spec->pipeline = "compv4";
        *arch = "gfx950";
        return 0;
    case 2:
        /* cshuffle epilogue, gfx950 */
        spec->problem = rocke_conv_problem_make(16, 112, 112, 128, 128, 3, 3, 1, 1, 1, 1, 1, 1);
        spec->epilogue = "cshuffle";
        *arch = "gfx950";
        return 0;
    case 3:
        /* async_dma pipeline, gfx950 */
        spec->problem = rocke_conv_problem_make(8, 56, 56, 64, 64, 3, 3, 1, 1, 1, 1, 1, 1);
        spec->async_dma = true;
        *arch = "gfx950";
        return 0;
    case 4:
        /* 1x1 filter (no spatial reduction), gfx950 */
        spec->problem = rocke_conv_problem_default(8, 56, 56, 64, 64, 1, 1);
        *arch = "gfx950";
        return 0;
    case 5:
        /* stride=2, pad=1, 3x3 -- tilde path (2x2 = 4 sub-GEMMs), gfx950 */
        spec->problem = rocke_conv_problem_make(4, 28, 28, 64, 64, 3, 3, 2, 2, 1, 1, 1, 1);
        *arch = "gfx950";
        return 0;
    case 6:
        /* stride=2, pad=1, 4x4 -- JIRA ticket scope, gfx950 */
        spec->problem = rocke_conv_problem_make(4, 28, 28, 64, 64, 4, 4, 2, 2, 1, 1, 1, 1);
        *arch = "gfx950";
        return 0;
    case 7:
        /* split_k=2, stride=1, gfx950 */
        spec->problem = rocke_conv_problem_make(8, 56, 56, 64, 64, 3, 3, 1, 1, 1, 1, 1, 1);
        spec->split_k = 2;
        *arch = "gfx950";
        return 0;
    case 8:
        /* WMMA wave32, gfx1151 */
        spec->problem = rocke_conv_problem_make(8, 56, 56, 64, 64, 3, 3, 1, 1, 1, 1, 1, 1);
        spec->warp_tile_m = 16;
        spec->warp_tile_n = 16;
        spec->warp_tile_k = 16;
        spec->wave_size = 32;
        *arch = "gfx1151";
        return 0;
    case 9:
        /* WMMA wave32, gfx1201 */
        spec->problem = rocke_conv_problem_make(8, 56, 56, 64, 64, 3, 3, 1, 1, 1, 1, 1, 1);
        spec->warp_tile_m = 16;
        spec->warp_tile_n = 16;
        spec->warp_tile_k = 16;
        spec->wave_size = 32;
        *arch = "gfx1201";
        return 0;
    case 10:
        /* chiplet_swizzle, gfx950 */
        spec->problem = rocke_conv_problem_make(8, 56, 56, 64, 64, 3, 3, 1, 1, 1, 1, 1, 1);
        spec->tile_m = 128;
        spec->tile_n = 128;
        spec->tile_k = 64;
        spec->chiplet_swizzle = true;
        spec->chiplet_wgm = 8;
        spec->chiplet_num_xcds = 8;
        spec->chiplet_chunk_size = 64;
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
        fprintf(stderr, "usage: %s <config_index>\n", argv[0]);
        return 2;
    }
    int idx = atoi(argv[1]);
    const char* mode = (argc > 2) ? argv[2] : "ll";

    rocke_dgrad_conv_spec_t spec;
    const char* arch = "gfx950";
    if(make_cfg(idx, &spec, &arch) != 0)
    {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 2;
    }

    rocke_ir_builder_t b;
    rocke_kernel_def_t* kernel = rocke_build_implicit_gemm_conv_dgrad_new(&b, &spec, arch);
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
