/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/conv_implicit_gemm_wgrad_emit.c -- C-side emitter for the
 * implicit-GEMM backward-weight convolution parity harness.  Selects one of N
 * sampled spec configs by argv[1] (the config index), builds the
 * rocke_implicit_gemm_conv_wgrad_spec_t identically to the Python emitter
 * conv_implicit_gemm_wgrad_emit.py, builds the kernel via
 * rocke_build_implicit_gemm_conv_wgrad_new and lowers via
 * rocke_lower_kernel_to_llvm (per-config arch, flavor AUTO), printing the .ll
 * to stdout so the two outputs can be byte-compared.
 *
 * Config index table (must stay in sync with the Python emitter):
 *   0  N8H56W56C64_K64Y3X3, t64x64x64, w2x2, a32x32x16, mem/default,      gfx950
 *   1  N8H56W56C64_K64Y3X3, t64x64x64, w2x2, a32x32x16, mem/cshuffle,     gfx950
 *   2  N8H56W56C64_K64Y3X3, t64x64x64, w2x2, a32x32x16, mem/default,      gfx950, split_k=4 fp16
 *   3  N8H56W56C64_K64Y1X1, t64x64x64, w2x2, a32x32x16, mem/default,      gfx950
 *   4  N8H56W56C64_K64Y3X3, t128x128x64, w2x2, a32x32x16, compv4/default, gfx950
 *   5  N8H56W56C64_K64Y3X3, t64x64x64, w2x2, a16x16x16, mem/default,      gfx1151 (WMMA w32)
 *   6  N8H56W56C64_K64Y3X3, t64x64x64, w2x2, a16x16x16, mem/default,      gfx1201 (WMMA w32)
 *   7  N8H56W56C64_K64Y3X3, t64x64x64, w2x2, a32x32x16, mem/default,      gfx950, split_k=4 fp32
 *   8  3-D conv N4Di14H14W14C32_K32Z3Y3X3, t64x64x64, w2x2, a32x32x16, mem/default, gfx950
 *   9  N8H56W56C64_K64Y3X3, t64x64x64, w2x2, a32x32x16, mem/default,      gfx950, split_k=4 bf16
 *  10  N8H56W56C64_K64Y3X3, t64x64x64, w2x2, a32x32x16, mem/default,      gfx950, chiplet_swizzle
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/instance_conv_implicit_gemm_wgrad.h"
#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"
#include "rocke/verify.h"

/* Fill the config for index `idx`. Returns 0 on success, -1 if unknown.
 * On success sets *spec and *arch. */
static int make_cfg(int idx, rocke_implicit_gemm_conv_wgrad_spec_t* spec, const char** arch)
{
    *spec = rocke_implicit_gemm_conv_wgrad_spec_default();
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
        spec->problem = rocke_conv_problem_default(8, 56, 56, 64, 64, 3, 3);
        *arch = "gfx950";
        return 0;
    case 1:
        spec->problem = rocke_conv_problem_default(8, 56, 56, 64, 64, 3, 3);
        spec->epilogue = "cshuffle";
        *arch = "gfx950";
        return 0;
    case 2:
        /* split_k=4, dtype_d="fp16" (default). */
        spec->problem = rocke_conv_problem_default(8, 56, 56, 64, 64, 3, 3);
        spec->split_k = 4;
        *arch = "gfx950";
        return 0;
    case 3:
        spec->problem = rocke_conv_problem_default(8, 56, 56, 64, 64, 1, 1);
        *arch = "gfx950";
        return 0;
    case 4:
        spec->problem = rocke_conv_problem_default(8, 56, 56, 64, 64, 3, 3);
        spec->tile_m = 128;
        spec->tile_n = 128;
        spec->tile_k = 64;
        spec->pipeline = "compv4";
        *arch = "gfx950";
        return 0;
    case 5:
        /* WMMA wave32, gfx1151. */
        spec->problem = rocke_conv_problem_default(8, 56, 56, 64, 64, 3, 3);
        spec->warp_tile_m = 16;
        spec->warp_tile_n = 16;
        spec->warp_tile_k = 16;
        spec->wave_size = 32;
        *arch = "gfx1151";
        return 0;
    case 6:
        /* WMMA wave32, gfx1201. */
        spec->problem = rocke_conv_problem_default(8, 56, 56, 64, 64, 3, 3);
        spec->warp_tile_m = 16;
        spec->warp_tile_n = 16;
        spec->warp_tile_k = 16;
        spec->wave_size = 32;
        *arch = "gfx1201";
        return 0;
    case 7:
        /* split_k=4, dtype_d="fp32". */
        spec->problem = rocke_conv_problem_default(8, 56, 56, 64, 64, 3, 3);
        spec->dtype_d = "fp32";
        spec->split_k = 4;
        *arch = "gfx950";
        return 0;
    case 8:
        /* 3-D convolution: N4 Di14 Hi14 Wi14 C32 K32 Z3 Y3 X3, s=1, p=1, d=1. */
        spec->problem
            = rocke_conv_problem_make_3d(4, 14, 14, 14, 32, 32, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1);
        spec->warp_tile_m = 32;
        spec->warp_tile_n = 32;
        spec->warp_tile_k = 16;
        *arch = "gfx950";
        return 0;
    case 9:
        /* split_k=4, dtype_d="bf16" (packed bf16 atomic path). */
        spec->problem = rocke_conv_problem_default(8, 56, 56, 64, 64, 3, 3);
        spec->dtype_d = "bf16";
        spec->split_k = 4;
        *arch = "gfx950";
        return 0;
    case 10:
        /* chiplet_swizzle enabled. */
        spec->problem = rocke_conv_problem_default(8, 56, 56, 64, 64, 3, 3);
        spec->chiplet_swizzle = true;
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

    rocke_implicit_gemm_conv_wgrad_spec_t spec;
    const char* arch = "gfx950";
    if(make_cfg(idx, &spec, &arch) != 0)
    {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 2;
    }

    rocke_ir_builder_t b;
    rocke_kernel_def_t* kernel = rocke_build_implicit_gemm_conv_wgrad_new(&b, &spec, arch);
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
