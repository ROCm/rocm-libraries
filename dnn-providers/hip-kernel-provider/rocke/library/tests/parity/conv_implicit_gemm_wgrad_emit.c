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
 *  11  N8H56W56C64_K64Y3X3, t64x64x64, w2x2, a32x32x16, mem/cshuffle,     gfx950, lds_k_outer
 *  12  N8H56W56C64_K64Y3X3, t64x64x64, w2x2, a32x32x16, mem/default,      gfx950, lds_k_outer + async_dma, split_k=4 fp32
 *  13  N8H56W56C64_K64Y3X3, t64x64x16, w2x2, a16x16x16, mem/cshuffle,     gfx950, lds_k_outer (4 operand elems/lane)
 *  14  N8H56W56C64_K64Y3X3, t64x64x64, w2x2, a32x32x16, basic/default,    gfx950, split_k=4 fp32
 *  15  N8H56W56C64_K64Y3X3, t64x64x64, w2x2, a32x32x16, mem/default,      gfx950, split_k=4 two_stage fp16
 *  16  N8H56W56C64_K64Y3X3, t64x64x64, w2x2, a16x16x16, mem/default,      gfx942, split_k=4 two_stage fp16
 *  17  N8H56W56C64_K64Y3X3 pad1, t32x32x32, w1x1, a16x16x32, mem/default, gfx1250 (WMMA w32), lds_k_outer fp32 out
 *  18  N8H56W56C64_K64Y3X3, t64x64x64, w2x2, a32x32x16, mem/default,      gfx950, unroll_k, split_k=4 fp32
 *  19  N8H56W56C3_K64Y3X3,  t64x64x64, w2x2, a32x32x16, mem/default,      gfx950, split_k=2 two_stage fp16
 *  20  N10H8W8C64_K64Y3X3,  t64x64x64, w2x2, a32x32x16, mem/cshuffle,     gfx950, lds_k_outer + async_dma, split_k=2
 *  21  N8H56W56C64_K64Y3X3, a32x32x16, gfx950, streamk dp_sk linear fp16
 *  22  N8H56W56C64_K64Y3X3, a32x32x16, gfx950, streamk dp_sk tree fp16
 *  23  N8H56W56C64_K64Y3X3, a32x32x16, gfx950, streamk dp_sk atomic fp32
 *  24  N8H56W56C64_K64Y3X3, a32x32x16, gfx950, streamk dp_sk workspace fp16
 *  25  N8H56W56C64_K64Y3X3, a16x16x16, gfx942, streamk persistent linear fp16
 *  26  N4H28W28C96_K120Y3X3G3 pad1, a32x32x16, gfx950, streamk dp_sk linear fp16
 *  27  N4H28W28C64_K128Y3X3G2 pad1, a32x32x16, gfx950, streamk persistent tree bf16 dW
 *  28  N4H28W28C64_K128Y3X3G2 pad1, a32x32x16, gfx950, streamk persistent atomic fp32
 *  29  N8H56W56C64_K64Y3X3, a16x16x16, gfx942, streamk persistent workspace fp16
 *  30  N4H14W14C64_K96Y3X3G2 pad1, a32x32x16, gfx950, streamk dp_sk linear fp16, lds_k_outer
 *  (21-30 carry a CTA pool, but it is a launch parameter and leaves the IR as is.)
 *  31-48  prefetch_addr=true on a base config (pfa_base below); 45-47 pair it
 *         with async_dma / unroll_k / WMMA and must be rejected by both engines
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/instance_conv_implicit_gemm_wgrad.h"
#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"
#include "rocke/verify.h"

/* prefetch_addr configs 31-48: the base config each one flips the flag on.
 * Must match _PFA_BASES in the Python emitter. -1 = not a prefetch_addr index. */
static int pfa_base(int idx)
{
    static const int bases[] = {1, 11, 4, 13, 7, 11, 15, 19, 14, 3, 8, 16, 0, 10, 12, 18, 5, 4};
    if(idx < 31 || idx >= 31 + (int)(sizeof(bases) / sizeof(bases[0])))
        return -1;
    return bases[idx - 31];
}

static int make_base_cfg(int idx, rocke_implicit_gemm_conv_wgrad_spec_t* spec, const char** arch);

/* Stream-K fields of configs 21-29 (Python _STREAMK_CFGS). */
static int streamk_cfg(int idx, rocke_implicit_gemm_conv_wgrad_spec_t* spec, const char** arch)
{
    /* idx: dtype a/b, dtype d, atom edge, streamk, reduction, ctas, arch */
    static const struct
    {
        const char *dt_ab, *dt_d;
        int edge;
        const char *mode, *reduction;
        int ctas;
        const char* arch;
    } cfgs[9] = {
        {"fp16", "fp16", 32, "dp_sk", "linear", 16, "gfx950"},
        {"fp16", "fp16", 32, "dp_sk", "tree", 16, "gfx950"},
        {"fp16", "fp32", 32, "dp_sk", "atomic", 4, "gfx950"},
        {"fp16", "fp16", 32, "dp_sk", "workspace", 4, "gfx950"},
        {"fp16", "fp16", 16, "persistent", "linear", 4, "gfx942"},
        {"fp16", "fp16", 32, "dp_sk", "linear", 6, "gfx950"},
        {"fp16", "bf16", 32, "persistent", "tree", 5, "gfx950"},
        {"fp16", "fp32", 32, "persistent", "atomic", 3, "gfx950"},
        {"fp16", "fp16", 16, "persistent", "workspace", 6, "gfx942"},
    };
    const int i = idx - 21;
    spec->dtype_a = cfgs[i].dt_ab;
    spec->dtype_b = cfgs[i].dt_ab;
    spec->dtype_d = cfgs[i].dt_d;
    spec->warp_tile_m = cfgs[i].edge;
    spec->warp_tile_n = cfgs[i].edge;
    spec->streamk = cfgs[i].mode;
    spec->streamk_reduction = cfgs[i].reduction;
    spec->streamk_ctas = cfgs[i].ctas;
    *arch = cfgs[i].arch;
    return 0;
}

/* Fill the config for index `idx`. Returns 0 on success, -1 if unknown.
 * On success sets *spec and *arch. */
static int make_cfg(int idx, rocke_implicit_gemm_conv_wgrad_spec_t* spec, const char** arch)
{
    int base = pfa_base(idx);
    if(base < 0)
        return make_base_cfg(idx, spec, arch);
    if(make_base_cfg(base, spec, arch) != 0)
        return -1;
    spec->prefetch_addr = true;
    if(idx == 36)
        spec->split_k = 2; /* split-K packed fp16 atomic on the K-outer cshuffle base */
    if(idx == 43)
        spec->pipeline = "compv3";
    if(idx == 48)
    {
        /* C = K = 3 on the 128x128x64 base: both operands load scalar, so
         * more carried offsets than accumulators */
        spec->problem = rocke_conv_problem_default(8, 56, 56, 3, 3, 3, 3);
    }
    return 0;
}

static int make_base_cfg(int idx, rocke_implicit_gemm_conv_wgrad_spec_t* spec, const char** arch)
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
    case 11:
        /* K-outer LDS tile + gfx950 ds_read_b64_tr_b16 transpose reads. */
        spec->problem = rocke_conv_problem_default(8, 56, 56, 64, 64, 3, 3);
        spec->epilogue = "cshuffle";
        spec->lds_k_outer = true;
        *arch = "gfx950";
        return 0;
    case 12:
        /* K-outer + direct load (async_dma). */
        spec->problem = rocke_conv_problem_default(8, 56, 56, 64, 64, 3, 3);
        spec->dtype_d = "fp32";
        spec->lds_k_outer = true;
        spec->async_dma = true;
        spec->split_k = 4;
        *arch = "gfx950";
        return 0;
    case 13:
        /* K-outer with the 16x16x16 atom: 4 MFMA operand elements per lane
         * rather than 8, so the transpose-read k-stride between lane groups
         * is 4. Cases 11 and 12 both use 32x32x16 (8 per lane). */
        spec->problem = rocke_conv_problem_default(8, 56, 56, 64, 64, 3, 3);
        spec->tile_k = 16;
        spec->warp_tile_m = 16;
        spec->warp_tile_n = 16;
        spec->warp_tile_k = 16;
        spec->epilogue = "cshuffle";
        spec->lds_k_outer = true;
        *arch = "gfx950";
        return 0;
    case 14:
        /* pipeline="basic": unrolled global-read/compute overlap loop. Needs a
         * compile-time trip count, hence the fixed split_k. */
        spec->problem = rocke_conv_problem_default(8, 56, 56, 64, 64, 3, 3);
        spec->pipeline = "basic";
        spec->dtype_d = "fp32";
        spec->split_k = 4;
        *arch = "gfx950";
        return 0;
    case 15:
        /* Two-stage deterministic: workspace-store epilogue, split_k=4, fp16, gfx950. */
        spec->problem = rocke_conv_problem_default(8, 56, 56, 64, 64, 3, 3);
        spec->split_k = 4;
        spec->two_stage = true;
        *arch = "gfx950";
        return 0;
    case 16:
        /* Two-stage deterministic: workspace-store epilogue, split_k=4, fp16, gfx942
         * (16x16x16 MFMA only). */
        spec->problem = rocke_conv_problem_default(8, 56, 56, 64, 64, 3, 3);
        spec->warp_tile_m = 16;
        spec->warp_tile_n = 16;
        spec->warp_tile_k = 16;
        spec->split_k = 4;
        spec->two_stage = true;
        *arch = "gfx942";
        return 0;
    case 17:
        /* gfx1250 wave32 WMMA 16x16x32 K-outer. The transpose read lowers to
         * ds_load_tr16_b128 (8 per lane), so a 16-element fragment is two
         * reads. dtype_d=fp32 because WMMA wgrad supports only the 'default'
         * epilogue, which rejects 16-bit dW. */
        spec->problem = rocke_conv_problem_default(8, 56, 56, 64, 64, 3, 3);
        spec->problem.pH = 1;
        spec->problem.pW = 1;
        spec->dtype_a = "fp16";
        spec->dtype_b = "fp16";
        spec->dtype_d = "fp32";
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
    case 18:
        /* unroll_k under split-K: the double-buffered loop's odd-tail prefetch
         * is redirected to wg_K so it cannot read the next slice's tile. */
        spec->problem = rocke_conv_problem_default(8, 56, 56, 64, 64, 3, 3);
        spec->dtype_d = "fp32";
        spec->unroll_k = true;
        spec->split_k = 4;
        *arch = "gfx950";
        return 0;
    case 19:
        /* Split-K (degree chosen at launch) + two-stage scratch; odd wg_N = 27,
         * which the packed 16-bit atomic cannot address. */
        spec->problem = rocke_conv_problem_default(8, 56, 56, 3, 64, 3, 3);
        spec->split_k = 2;
        spec->two_stage = true;
        *arch = "gfx950";
        return 0;
    case 20:
        /* K-outer + async_dma + split-K=2, fp16 cshuffle atomic epilogue; 3 K
         * tiles per slice, so phase B's prefetch takes the zero-fill redirect. */
        spec->problem = rocke_conv_problem_default(10, 8, 8, 64, 64, 3, 3);
        spec->epilogue = "cshuffle";
        spec->lds_k_outer = true;
        spec->async_dma = true;
        spec->split_k = 2;
        *arch = "gfx950";
        return 0;
    case 21:
    case 22:
    case 23:
    case 24:
    case 25:
    case 29:
        spec->problem = rocke_conv_problem_default(8, 56, 56, 64, 64, 3, 3);
        return streamk_cfg(idx, spec, arch);
    case 26:
        spec->problem = rocke_conv_problem_default(4, 28, 28, 96, 120, 3, 3);
        spec->problem.pH = 1;
        spec->problem.pW = 1;
        spec->problem.groups = 3;
        return streamk_cfg(idx, spec, arch);
    case 27:
    case 28:
        spec->problem = rocke_conv_problem_default(4, 28, 28, 64, 128, 3, 3);
        spec->problem.pH = 1;
        spec->problem.pW = 1;
        spec->problem.groups = 2;
        return streamk_cfg(idx, spec, arch);
    case 30:
        spec->problem = rocke_conv_problem_default(4, 14, 14, 64, 96, 3, 3);
        spec->problem.pH = 1;
        spec->problem.pW = 1;
        spec->problem.groups = 2;
        streamk_cfg(26, spec, arch);
        spec->streamk_ctas = 7;
        spec->lds_k_outer = true;
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
