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
    case 11:
        /* K-outer B tile + ds_read_tr16_b64 transpose read, 32x32x16 atom.
         * cshuffle deliberately: the validator rejects 16-bit dtype_d with the
         * default epilogue, and a rejected config lands as BOTH_REJECTED, which
         * the gate counts as a pass -- it would compare nothing. */
        spec->problem = rocke_conv_problem_make(8, 56, 56, 64, 64, 3, 3, 1, 1, 1, 1, 1, 1);
        spec->tile_m = 128;
        spec->tile_n = 128;
        spec->tile_k = 64;
        spec->warp_tile_m = 32;
        spec->warp_tile_n = 32;
        spec->warp_tile_k = 16;
        spec->epilogue = "cshuffle";
        spec->lds_k_outer = true;
        *arch = "gfx950";
        return 0;
    case 12:
        /* Same path on the 16x16x16 atom, where b_frag_len is 4 rather than 8. */
        spec->problem = rocke_conv_problem_make(8, 56, 56, 64, 64, 3, 3, 1, 1, 1, 1, 1, 1);
        spec->tile_m = 64;
        spec->tile_n = 64;
        spec->tile_k = 32;
        spec->warp_tile_m = 16;
        spec->warp_tile_n = 16;
        spec->warp_tile_k = 16;
        spec->epilogue = "cshuffle";
        spec->lds_k_outer = true;
        *arch = "gfx950";
        return 0;
    case 13:
        /* gfx1250 wave32 WMMA 16x16x32 K-outer. The shared transpose-read
         * helper takes its wave32 branch here and lowers to ds_load_tr16_b128
         * (8 per lane), so a 16-element fragment is two reads. Configs 11 and
         * 12 are both wave64. */
        spec->problem = rocke_conv_problem_make(8, 56, 56, 64, 64, 3, 3, 1, 1, 1, 1, 1, 1);
        spec->tile_m = 32;
        spec->tile_n = 32;
        spec->tile_k = 32;
        spec->warp_m = 1;
        spec->warp_n = 1;
        spec->warp_tile_m = 16;
        spec->warp_tile_n = 16;
        spec->warp_tile_k = 32;
        spec->wave_size = 32;
        spec->pipeline = "mem";
        spec->epilogue = "default";
        spec->lds_k_outer = true;
        *arch = "gfx1250";
        return 0;
    case 14:
        /* Rejected by both validators: explicit vector_size_a 8 on a 16x32 dY
         * tile gives 64 chunks for a 128-thread block. */
        spec->problem = rocke_conv_problem_make(8, 56, 56, 64, 64, 3, 3, 1, 1, 1, 1, 1, 1);
        spec->tile_m = 16;
        spec->tile_n = 32;
        spec->tile_k = 32;
        spec->warp_m = 1;
        spec->warp_n = 2;
        spec->warp_tile_m = 16;
        spec->warp_tile_n = 16;
        spec->warp_tile_k = 32;
        spec->has_vector_size_a = true;
        spec->vector_size_a = 8;
        *arch = "gfx950";
        return 0;
    case 15:
        /* uniform_k_decode, stride 1: the per-tile decode loop plus the plain
         * fallback loop, sharing one MFMA phase. */
        spec->problem = rocke_conv_problem_make(8, 56, 56, 64, 64, 3, 3, 1, 1, 1, 1, 1, 1);
        spec->uniform_k_decode = true;
        *arch = "gfx950";
        return 0;
    case 16:
        /* uniform_k_decode on the strided tilde path with the K-outer B tile
         * and the cshuffle epilogue. */
        spec->problem = rocke_conv_problem_make(4, 28, 28, 64, 64, 3, 3, 2, 2, 1, 1, 1, 1);
        spec->tile_m = 128;
        spec->tile_n = 128;
        spec->tile_k = 64;
        spec->pipeline = "compv3";
        spec->epilogue = "cshuffle";
        spec->lds_k_outer = true;
        spec->uniform_k_decode = true;
        *arch = "gfx950";
        return 0;
    case 17:
        /* uniform_k_decode with split-K: decode from a non-zero k_lo. */
        spec->problem = rocke_conv_problem_make(8, 56, 56, 64, 64, 3, 3, 1, 1, 1, 1, 1, 1);
        spec->tile_k = 32;
        spec->split_k = 2;
        spec->uniform_k_decode = true;
        *arch = "gfx950";
        return 0;
    case 18:
        /* uniform_k_decode on a pointwise problem. */
        spec->problem = rocke_conv_problem_default(8, 56, 56, 64, 64, 1, 1);
        spec->uniform_k_decode = true;
        *arch = "gfx950";
        return 0;
    case 19:
        /* uniform_k_decode on WMMA wave32 (gfx1201). */
        spec->problem = rocke_conv_problem_make(8, 56, 56, 64, 64, 3, 3, 1, 1, 1, 1, 1, 1);
        spec->warp_tile_m = 16;
        spec->warp_tile_n = 16;
        spec->warp_tile_k = 16;
        spec->wave_size = 32;
        spec->uniform_k_decode = true;
        *arch = "gfx1201";
        return 0;
    case 20:
        /* Rejected by both validators: wavelet has its own K loop. */
        spec->problem = rocke_conv_problem_make(8, 56, 56, 64, 64, 3, 3, 1, 1, 1, 1, 1, 1);
        spec->tile_k = 32;
        spec->warp_tile_m = 16;
        spec->warp_tile_n = 16;
        spec->warp_tile_k = 32;
        spec->wave_size = 32;
        spec->pipeline = "wavelet";
        spec->uniform_k_decode = true;
        *arch = "gfx1250";
        return 0;
    /* Configs 21-33: grouped convolution (groups > 1). The conv group rides
     * blockIdx.y; the reduction decodes by kpg, the dY/W channel is offset by
     * g*kpg, W's channel stride is cpg and dX's channel is offset by g*cpg.
     * Every epilogue leaf, the uniform_k_decode loop and the per-group vector
     * widths are covered, plus the two grouped rejections. All use fp16
     * operands: the C++ tile loader stages through f16-typed LDS stores, so a
     * bf16 operand differs from Python in that IR type alone, grouped or not. */
    case 21:
        /* Grouped stride 1, direct epilogue. */
        spec->problem = rocke_conv_problem_make(8, 56, 56, 64, 64, 3, 3, 1, 1, 1, 1, 1, 1);
        spec->problem.groups = 4;
        *arch = "gfx950";
        return 0;
    case 22:
        /* Grouped stride 1 with uniform_k_decode: the decode divides by kpg,
         * and both per-tile offsets carry the group's output-channel base. */
        spec->problem = rocke_conv_problem_make(8, 56, 56, 64, 64, 3, 3, 1, 1, 1, 1, 1, 1);
        spec->problem.groups = 4;
        spec->uniform_k_decode = true;
        *arch = "gfx950";
        return 0;
    case 23:
        /* Grouped cshuffle with cpg = kpg = 6: the load and store widths
         * follow the per-group channel run (2), not the dense C (8). */
        spec->problem = rocke_conv_problem_make(4, 28, 28, 24, 24, 3, 3, 1, 1, 1, 1, 1, 1);
        spec->problem.groups = 4;
        spec->epilogue = "cshuffle";
        *arch = "gfx950";
        return 0;
    case 24:
        /* Grouped uniform_k_decode with the K-outer B tile and cshuffle. */
        spec->problem = rocke_conv_problem_make(8, 28, 28, 128, 128, 3, 3, 1, 1, 1, 1, 1, 1);
        spec->problem.groups = 4;
        spec->tile_m = 128;
        spec->tile_n = 128;
        spec->epilogue = "cshuffle";
        spec->lds_k_outer = true;
        spec->uniform_k_decode = true;
        *arch = "gfx950";
        return 0;
    case 25:
        /* Grouped stride 2: the tilde path with the direct tilde epilogue. */
        spec->problem = rocke_conv_problem_make(4, 28, 28, 64, 64, 3, 3, 2, 2, 1, 1, 1, 1);
        spec->problem.groups = 4;
        *arch = "gfx950";
        return 0;
    case 26:
        /* Grouped stride 2 with uniform_k_decode and the tilde cshuffle
         * epilogue. cpg = 12, so the tilde store is 4 wide: it follows the
         * per-group channel run like the stride-1 cshuffle, not a fixed 8. */
        spec->problem = rocke_conv_problem_make(4, 28, 28, 24, 24, 3, 3, 2, 2, 1, 1, 1, 1);
        spec->problem.groups = 2;
        spec->tile_m = 128;
        spec->tile_n = 128;
        spec->pipeline = "compv3";
        spec->epilogue = "cshuffle";
        spec->uniform_k_decode = true;
        *arch = "gfx950";
        return 0;
    case 27:
        /* Grouped split-K with uniform_k_decode: the packed fp16 atomic
         * epilogue on the absolute dX channel. */
        spec->problem = rocke_conv_problem_make(8, 28, 28, 128, 128, 3, 3, 1, 1, 1, 1, 1, 1);
        spec->problem.groups = 2;
        spec->tile_k = 32;
        spec->split_k = 2;
        spec->uniform_k_decode = true;
        *arch = "gfx950";
        return 0;
    case 28:
        /* Grouped split-K on the strided tilde path (packed fp16 atomics). */
        spec->problem = rocke_conv_problem_make(4, 28, 28, 64, 64, 3, 3, 2, 2, 1, 1, 1, 1);
        spec->problem.groups = 4;
        spec->tile_k = 32;
        spec->split_k = 2;
        *arch = "gfx950";
        return 0;
    case 29:
        /* Grouped 1x1 with uniform_k_decode: a grouped problem never takes the
         * ungrouped pointwise fast path. */
        spec->problem = rocke_conv_problem_default(8, 56, 56, 64, 64, 1, 1);
        spec->problem.groups = 4;
        spec->uniform_k_decode = true;
        *arch = "gfx950";
        return 0;
    case 30:
        /* Grouped WMMA wave32 stride 1 with uniform_k_decode (the WMMA direct
         * epilogue). */
        spec->problem = rocke_conv_problem_make(8, 56, 56, 64, 64, 3, 3, 1, 1, 1, 1, 1, 1);
        spec->problem.groups = 4;
        spec->warp_tile_m = 16;
        spec->warp_tile_n = 16;
        spec->warp_tile_k = 16;
        spec->wave_size = 32;
        spec->uniform_k_decode = true;
        *arch = "gfx1201";
        return 0;
    case 31:
        /* Grouped WMMA wave32 stride 2 (the WMMA tilde direct epilogue). */
        spec->problem = rocke_conv_problem_make(4, 28, 28, 64, 64, 3, 3, 2, 2, 1, 1, 1, 1);
        spec->problem.groups = 4;
        spec->warp_tile_m = 16;
        spec->warp_tile_n = 16;
        spec->warp_tile_k = 16;
        spec->wave_size = 32;
        *arch = "gfx1201";
        return 0;
    case 32:
        /* Grouped depthwise (channels-per-group == 1): a one-column GEMM per
         * group, every operand scalar. cshuffle so the config is accepted
         * (the default epilogue rejects a 16-bit dX). */
        spec->problem = rocke_conv_problem_make(8, 56, 56, 64, 64, 3, 3, 1, 1, 1, 1, 1, 1);
        spec->problem.groups = 64;
        spec->epilogue = "cshuffle";
        *arch = "gfx950";
        return 0;
    case 33:
        /* Rejected by both validators: packed fp16 split-K atomics need an even
         * channels-per-group (C = 12 is even, cpg = 3 is not). */
        spec->problem = rocke_conv_problem_make(8, 56, 56, 12, 12, 3, 3, 1, 1, 1, 1, 1, 1);
        spec->problem.groups = 4;
        spec->tile_k = 32;
        spec->split_k = 2;
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
