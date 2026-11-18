/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/conv_implicit_gemm_emit.c -- C-side emitter for the implicit-GEMM
 * convolution parity harness. Selects one of N sampled spec configs by argv[1]
 * (the config index), builds the rocke_implicit_gemm_conv_spec_t identically to
 * the Python emitter conv_implicit_gemm_emit.py, builds the kernel via
 * rocke_build_implicit_gemm_conv_new (initialized builder + spec + arch, NULL
 * overrides = stock conv body) and lowers via rocke_lower_kernel_to_llvm
 * (per-config arch, flavor AUTO), printing the .ll to stdout so the two outputs
 * can be byte-compared.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/instance_conv_implicit_gemm.h"
#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"
#include "rocke/verify.h"

/* Fill the config for index `idx`. Returns 0 on success, -1 if unknown.
 * On success sets *spec and *arch. */
static int make_cfg_raw(int idx, rocke_implicit_gemm_conv_spec_t* spec, const char** arch)
{
    *spec = rocke_implicit_gemm_conv_spec_default();
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
        spec->tile_m = 128;
        spec->tile_n = 128;
        spec->tile_k = 64;
        spec->pipeline = "compv4";
        *arch = "gfx950";
        return 0;
    case 2:
        spec->problem = rocke_conv_problem_default(16, 112, 112, 128, 128, 3, 3);
        spec->epilogue = "cshuffle";
        *arch = "gfx950";
        return 0;
    case 3:
        spec->problem = rocke_conv_problem_default(8, 56, 56, 64, 64, 3, 3);
        spec->async_dma = true;
        *arch = "gfx950";
        return 0;
    case 4:
        spec->problem = rocke_conv_problem_default(1, 224, 224, 3, 64, 7, 7);
        spec->tile_m = 128;
        spec->tile_n = 128;
        spec->tile_k = 64;
        *arch = "gfx950";
        return 0;
    case 5:
        spec->problem = rocke_conv_problem_default(8, 56, 56, 64, 64, 1, 1);
        *arch = "gfx950";
        return 0;
    case 6:
    case 7:
    case 8:
        /* WMMA wave32 RDNA targets: 16x16x16 / mem / default, w32. */
        spec->problem = rocke_conv_problem_default(8, 56, 56, 64, 64, 3, 3);
        spec->warp_tile_m = 16;
        spec->warp_tile_n = 16;
        spec->warp_tile_k = 16;
        spec->wave_size = 32;
        *arch = (idx == 6) ? "gfx1151" : (idx == 7) ? "gfx1201" : "gfx11-generic";
        return 0;
    case 9:
        /* chiplet_swizzle gfx950. */
        spec->problem = rocke_conv_problem_default(8, 56, 56, 64, 64, 3, 3);
        spec->tile_m = 128;
        spec->tile_n = 128;
        spec->tile_k = 64;
        spec->chiplet_swizzle = true;
        spec->chiplet_wgm = 8;
        spec->chiplet_num_xcds = 8;
        spec->chiplet_chunk_size = 64;
        *arch = "gfx950";
        return 0;
    case 10:
        /* cshuffle with cshuffle_no_alias=True (idx 2 shape). */
        spec->problem = rocke_conv_problem_default(16, 112, 112, 128, 128, 3, 3);
        spec->epilogue = "cshuffle";
        spec->cshuffle_no_alias = true;
        *arch = "gfx950";
        return 0;
    case 11:
        /* basic pipeline, default epilogue. */
        spec->problem = rocke_conv_problem_default(8, 56, 56, 64, 64, 3, 3);
        spec->pipeline = "basic";
        *arch = "gfx950";
        return 0;
    case 12:
        /* basic pipeline, cshuffle epilogue. */
        spec->problem = rocke_conv_problem_default(8, 56, 56, 64, 64, 3, 3);
        spec->pipeline = "basic";
        spec->epilogue = "cshuffle";
        *arch = "gfx950";
        return 0;
    case 13:
        /* wavelet pipeline, default epilogue (gfx1250 WMMA).
         * warp_tile_k=32: gfx1250 WMMA requires the 16x16x32 atom.
         * vector_size_c=1: K=64 auto-derives vec_c=8 which blocks default epilogue. */
        spec->problem = rocke_conv_problem_default(8, 56, 56, 64, 64, 3, 3);
        spec->tile_m = 32;
        spec->tile_n = 32;
        spec->tile_k = 32;
        spec->warp_m = 1;
        spec->warp_n = 1;
        spec->warp_tile_m = 16;
        spec->warp_tile_n = 16;
        spec->warp_tile_k = 32;
        spec->wave_size = 32;
        spec->pipeline = "wavelet";
        spec->epilogue = "default";
        spec->has_vector_size_c = true;
        spec->vector_size_c = 1;
        spec->num_load_waves = 2;
        *arch = "gfx1250";
        return 0;
    case 14:
        /* wavelet pipeline, cshuffle epilogue (gfx1250 WMMA).
         * warp_tile_k=32: gfx1250 WMMA requires the 16x16x32 atom. */
        spec->problem = rocke_conv_problem_default(8, 56, 56, 64, 64, 3, 3);
        spec->tile_m = 32;
        spec->tile_n = 32;
        spec->tile_k = 32;
        spec->warp_m = 1;
        spec->warp_n = 1;
        spec->warp_tile_m = 16;
        spec->warp_tile_n = 16;
        spec->warp_tile_k = 32;
        spec->wave_size = 32;
        spec->pipeline = "wavelet";
        spec->epilogue = "cshuffle";
        spec->num_load_waves = 2;
        *arch = "gfx1250";
        return 0;
    case 15:
        /* Grouped conv: groups=4, C=64->cpg=16, K=64->kpg=16. */
        spec->problem = rocke_conv_problem_make(2, 14, 14, 64, 64, 3, 3, 1, 1, 1, 1, 1, 1);
        spec->problem.groups = 4;
        *arch = "gfx950";
        return 0;
    case 16:
        /* Cardinality-grouped conv: groups=32, C=256->cpg=8, K=256->kpg=8. */
        spec->problem = rocke_conv_problem_make(2, 8, 8, 256, 256, 3, 3, 1, 1, 1, 1, 1, 1);
        spec->problem.groups = 32;
        *arch = "gfx950";
        return 0;
    case 17:
        /* Pointwise + cshuffle: the pointwise d_addr closure on the cshuffle
         * store (idx 5 covers it on the direct store). */
        spec->problem = rocke_conv_problem_default(8, 56, 56, 64, 64, 1, 1);
        spec->epilogue = "cshuffle";
        *arch = "gfx950";
        return 0;
    case 18:
        /* Rejected by both validators: explicit vec 8 on a 16x32 A tile gives
         * 64 chunks for a 128-thread block (rocke_conv_coalesced_load_ok). */
        spec->problem = rocke_conv_problem_default(8, 56, 56, 64, 64, 3, 3);
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
        spec->has_vector_size_b = true;
        spec->vector_size_b = 8;
        *arch = "gfx950";
        return 0;
    case 19:
        /* unroll_k double-buffered K loop + its "unroll" name tag; K_gemm is
         * 9 tiles of 64 (odd). */
        spec->problem = rocke_conv_problem_default(8, 56, 56, 64, 64, 3, 3);
        spec->epilogue = "cshuffle";
        spec->unroll_k = true;
        *arch = "gfx950";
        return 0;
    case 20:
        /* gfx942 (CDNA3) async_dma: same shape as config 3, but on an arch
         * whose buffer_load_lds moves only a dword (not CDNA4's b128) per
         * lane -- covers the arch-width cap in the C++ async-conv port,
         * which previously had parity coverage only on gfx950. warp_tile is
         * 16x16x16 (not config 3's 32x32x16): that atom is CDNA4-only,
         * gfx942's MFMA catalog tops out at 16x16x16 / 32x32x8. */
        spec->problem = rocke_conv_problem_default(8, 56, 56, 64, 64, 3, 3);
        spec->warp_tile_m = 16;
        spec->warp_tile_n = 16;
        spec->warp_tile_k = 16;
        spec->async_dma = true;
        *arch = "gfx942";
        return 0;
    /* --- depthwise + merged groups ---------------------------------------
     * Every config below is depthwise (C == K == groups, so cpg == kpg == 1),
     * which is the only shape group_merge admits. 21 is the unmerged control:
     * without it the merged configs would have nothing to differ *from*, and
     * the depthwise path itself carried no structural coverage at all.
     *
     * 22-26 span the axes that change emitted IR under merge: the shift/mask
     * width (log2 Gm) of the merged k decode, Gm == groups (one merged group,
     * the grouped decode still engaged), which epilogue consumes the merged
     * dims, and the unroll_k K loop over the in-kernel merged extent. Keep in
     * lockstep with the matching block in conv_implicit_gemm_emit.py. */
    case 21:
    case 22:
    case 23:
    case 24:
    case 25:
    case 26:
    {
        /* groups=64, C=64 -> cpg=1, K=64 -> kpg=1. M = 2*14*14 = 392. */
        spec->problem = rocke_conv_problem_make(2, 14, 14, 64, 64, 3, 3, 1, 1, 1, 1, 1, 1);
        spec->problem.groups = 64;
        static const int kGm[] = {1, 8, 32, 4, 64, 8}; /* idx 21..26 */
        spec->group_merge = kGm[idx - 21];
        /* 24 keeps the *direct* epilogue by pinning vector_size_c=1 -- merged
         * kpg would otherwise auto-derive vec_c > 1, which the validator turns
         * into a cshuffle requirement, so without the pin no merged config
         * would exercise rocke_conv_emit_direct_epilogue. */
        if(idx == 24)
        {
            spec->has_vector_size_c = true;
            spec->vector_size_c = 1;
        }
        else if(idx != 21)
        {
            spec->epilogue = "cshuffle";
        }
        spec->unroll_k = (idx == 26);
        *arch = "gfx950";
        return 0;
    }
    case 27:
    case 28:
    {
        /* The 1-wide cshuffle store (store_vec == 1): 27 is merged Gm=8 pinned
         * to vector_size_c=1, 28 the unmerged depthwise control whose open
         * store width derives from kpg == 1. Same depthwise problem as 21-26.
         * Keep in lockstep with conv_implicit_gemm_emit.py. */
        spec->problem = rocke_conv_problem_make(2, 14, 14, 64, 64, 3, 3, 1, 1, 1, 1, 1, 1);
        spec->problem.groups = 64;
        spec->epilogue = "cshuffle";
        if(idx == 27)
        {
            spec->group_merge = 8;
            spec->has_vector_size_c = true;
            spec->vector_size_c = 1;
        }
        *arch = "gfx950";
        return 0;
    }
    case 29:
    case 30:
    case 31:
    case 32:
    {
        /* Group merging on WMMA (wave32). Same depthwise problem as 21-28.
         * 29: gfx1250 mem/cshuffle, 30: gfx1250 wavelet/cshuffle, 31: gfx1250
         * direct epilogue (vector_size_c pinned to 1), 32: gfx1151 (gfx11
         * WMMA, 16x16x16 atom) mem/cshuffle. Keep in lockstep with
         * conv_implicit_gemm_emit.py. */
        static const int kGm[] = {8, 16, 4, 8}; /* idx 29..32 */
        spec->problem = rocke_conv_problem_make(2, 14, 14, 64, 64, 3, 3, 1, 1, 1, 1, 1, 1);
        spec->problem.groups = 64;
        spec->group_merge = kGm[idx - 29];
        spec->tile_m = 32;
        spec->tile_n = 32;
        spec->tile_k = 32;
        spec->warp_m = 1;
        spec->warp_n = 1;
        spec->warp_tile_m = 16;
        spec->warp_tile_n = 16;
        spec->warp_tile_k = (idx == 32) ? 16 : 32;
        spec->wave_size = 32;
        if(idx == 30)
        {
            spec->pipeline = "wavelet";
            spec->num_load_waves = 2;
        }
        if(idx == 31)
        {
            spec->has_vector_size_c = true;
            spec->vector_size_c = 1;
        }
        else
        {
            spec->epilogue = "cshuffle";
        }
        *arch = (idx == 32) ? "gfx1151" : "gfx1250";
        return 0;
    }
    case 33:
    case 34:
    case 35:
    {
        /* Grid-per-group WMMA (unmerged) on gfx1250 grouped problems.
         * 33: groups=4 (cpg=kpg=16) mem/cshuffle, 34:
         * depthwise (cpg=kpg=1) mem/default, 35: groups=4 wavelet/cshuffle.
         * Keep in lockstep with conv_implicit_gemm_emit.py. */
        spec->problem = rocke_conv_problem_make(2, 14, 14, 64, 64, 3, 3, 1, 1, 1, 1, 1, 1);
        spec->problem.groups = (idx == 34) ? 64 : 4;
        spec->tile_m = 32;
        spec->tile_n = 32;
        spec->tile_k = 32;
        spec->warp_m = 2;
        spec->warp_n = 1;
        spec->warp_tile_m = 16;
        spec->warp_tile_n = 16;
        spec->warp_tile_k = 32;
        spec->wave_size = 32;
        if(idx == 35)
        {
            spec->pipeline = "wavelet";
            spec->num_load_waves = 2;
        }
        if(idx != 34)
        {
            spec->epilogue = "cshuffle";
        }
        *arch = "gfx1250";
        return 0;
    }
    case 36:
    case 37:
    case 38:
        /* Grouped pointwise (1x1, stride 1, no pad): the flat fast path with
         * the group's channel slab on A, the within-group filter bound on B and
         * the group's columns in D's K-wide rows. groups=4, cpg=kpg=16. 36:
         * direct store, 37: cshuffle store, 38: async_dma loaders. Keep in
         * lockstep with conv_implicit_gemm_emit.py. */
        spec->problem = rocke_conv_problem_default(2, 14, 14, 64, 64, 1, 1);
        spec->problem.groups = 4;
        if(idx == 37)
            spec->epilogue = "cshuffle";
        if(idx == 38)
            spec->async_dma = true;
        *arch = "gfx950";
        return 0;
    case 39:
        /* Grouped pointwise on the WMMA direct store, which spells the
         * pointwise D offset out inline rather than through d_addr. */
        spec->problem = rocke_conv_problem_default(2, 14, 14, 64, 64, 1, 1);
        spec->problem.groups = 4;
        spec->tile_m = 32;
        spec->tile_n = 32;
        spec->tile_k = 32;
        spec->warp_m = 2;
        spec->warp_n = 1;
        spec->warp_tile_m = 16;
        spec->warp_tile_n = 16;
        spec->warp_tile_k = 32;
        spec->wave_size = 32;
        *arch = "gfx1250";
        return 0;
    default:
        return -1;
    }
}

/* Split-K forward (split_k > 1) and its single-launch cluster form
 * (split_k_cluster) on gfx1250 WMMA, plus one gfx950 MFMA split. Keep in
 * lockstep with _split_k_spec() in conv_implicit_gemm_emit.py.
 *   40: wavelet/cshuffle, 16x16x64 tile (small M, deep K)
 *   41: mem/cshuffle, 64x64 tile, 2x2 warps
 *   42: mem/default (vector_size_c=1), 32x32x32 tile
 *   43: grouped (groups=2) wavelet/cshuffle
 *   44: cluster S=4, mem/cshuffle, 16x16x64 tile
 *   45: cluster S=8, mem/cshuffle, 64x64 tile
 *   46: cluster S=15, mem/cshuffle, 16x16x64 tile
 *   47: cluster S=4, grouped (groups=2), 32x32 tile, 1x2 warps
 *   48: cluster S=12, mem/default, 32x32x32 tile
 *   49: split_k=2 on gfx950 (MFMA mem/cshuffle, 64x64 tile)
 *   50: split_k=2 with a bias acc_epilogue: rejected by both engines
 *   51: split_k=0: rejected by both engines
 *   52: cluster S=4 with K=kpg=10 (not a multiple of 4): rejected by both
 *   53: cluster S=4 on gfx1201 (no cluster launch): rejected by both engines */
static int make_cfg_split_k(int idx, rocke_implicit_gemm_conv_spec_t* spec, const char** arch)
{
    const rocke_conv_problem_t small_m
        = rocke_conv_problem_make(1, 7, 7, 256, 64, 3, 3, 1, 1, 1, 1, 1, 1);
    const rocke_conv_problem_t csh_64
        = rocke_conv_problem_make(2, 12, 12, 128, 96, 3, 3, 1, 1, 1, 1, 1, 1);
    const rocke_conv_problem_t dflt
        = rocke_conv_problem_make(1, 9, 9, 64, 40, 3, 3, 1, 1, 1, 1, 1, 1);
    rocke_conv_problem_t grouped
        = rocke_conv_problem_make(1, 10, 10, 128, 64, 3, 3, 1, 1, 1, 1, 1, 1);
    const rocke_conv_problem_t odd_k
        = rocke_conv_problem_make(1, 7, 7, 256, 10, 3, 3, 1, 1, 1, 1, 1, 1);
    grouped.groups = 2;

    *spec = rocke_implicit_gemm_conv_spec_default();
    spec->tile_m = 16;
    spec->tile_n = 16;
    spec->tile_k = 64;
    spec->warp_m = 1;
    spec->warp_n = 1;
    spec->warp_tile_m = 16;
    spec->warp_tile_n = 16;
    spec->warp_tile_k = 32;
    spec->wave_size = 32;
    spec->has_vector_size_a = true;
    spec->vector_size_a = 8;
    spec->has_vector_size_b = true;
    spec->vector_size_b = 8;
    spec->has_vector_size_c = true;
    spec->vector_size_c = 8;
    spec->pipeline = "wavelet";
    spec->epilogue = "cshuffle";
    spec->num_load_waves = 1;
    spec->split_k = 2;
    *arch = "gfx1250";

    switch(idx)
    {
    case 40:
        spec->problem = small_m;
        return 0;
    case 41:
    case 45:
        spec->problem = csh_64;
        spec->tile_m = 64;
        spec->tile_n = 64;
        spec->warp_m = 2;
        spec->warp_n = 2;
        spec->pipeline = "mem";
        spec->split_k_cluster = (idx == 45) ? 8 : 0;
        return 0;
    case 42:
    case 48:
        spec->problem = dflt;
        spec->tile_m = 32;
        spec->tile_n = 32;
        spec->tile_k = 32;
        spec->warp_m = 2;
        spec->warp_n = 1;
        spec->epilogue = "default";
        spec->vector_size_c = 1;
        spec->pipeline = "mem";
        spec->split_k_cluster = (idx == 48) ? 12 : 0;
        return 0;
    case 43:
        spec->problem = grouped;
        return 0;
    case 44:
    case 46:
        spec->problem = small_m;
        spec->pipeline = "mem";
        spec->split_k_cluster = (idx == 44) ? 4 : 15;
        return 0;
    case 47:
        spec->problem = grouped;
        spec->tile_m = 32;
        spec->tile_n = 32;
        spec->warp_m = 1;
        spec->warp_n = 2;
        spec->pipeline = "mem";
        spec->split_k_cluster = 4;
        return 0;
    case 49:
        spec->problem = csh_64;
        spec->tile_m = 64;
        spec->tile_n = 64;
        spec->warp_m = 2;
        spec->warp_n = 2;
        spec->warp_tile_m = 32;
        spec->warp_tile_n = 32;
        spec->warp_tile_k = 16;
        spec->wave_size = 64;
        spec->pipeline = "mem";
        *arch = "gfx950";
        return 0;
    case 50:
        spec->problem = small_m;
        spec->pipeline = "mem";
        spec->acc_epilogue.bias = 1.0;
        return 0;
    case 51:
        spec->problem = small_m;
        spec->split_k = 0;
        return 0;
    case 52:
        spec->problem = odd_k;
        spec->pipeline = "mem";
        spec->split_k_cluster = 4;
        spec->vector_size_c = 2;
        return 0;
    case 53:
        spec->problem = small_m;
        spec->pipeline = "mem";
        spec->split_k_cluster = 4;
        spec->warp_tile_k = 16;
        *arch = "gfx1201";
        return 0;
    default:
        return -1;
    }
}

/* TF32 compute (mma_dtype "tf32"): fp32 storage, each LDS fragment bitcast to
 * vec<tf32xN> for the arch's TF32 atom -- native XF32 on gfx942 (16x16x8,
 * 32x32x4), BF16x3 emulation on gfx950 (16x16x32, 32x32x16). 64x64x32 tile,
 * 2x2 warps. Keep in lockstep with _tf32_spec() in conv_implicit_gemm_emit.py
 * (which lists what each index covers). */
static int make_cfg_tf32(int idx, rocke_implicit_gemm_conv_spec_t* spec, const char** arch)
{
    /* idx: is_3d, depthwise, arch, warp tile m=n, k, pipeline, epilogue */
    static const struct
    {
        int is_3d;
        int depthwise;
        const char* arch;
        int m;
        int k;
        const char* pipeline;
        const char* epilogue;
    } cfgs[] = {
        {0, 0, "gfx950", 16, 32, "mem", "cshuffle"}, /* 54 */
        {0, 0, "gfx950", 32, 16, "compv4", "default"}, /* 55 */
        {0, 0, "gfx942", 16, 8, "compv3", "cshuffle"}, /* 56 */
        {0, 0, "gfx942", 32, 4, "compv4", "default"}, /* 57 */
        {1, 0, "gfx950", 16, 32, "compv3", "cshuffle"}, /* 58 */
        {1, 0, "gfx942", 16, 8, "mem", "default"}, /* 59 */
        {0, 0, "gfx950", 32, 16, "mem", "cshuffle"}, /* 60: async_dma */
        {0, 0, "gfx942", 32, 4, "compv4", "cshuffle"}, /* 61: unroll_k */
        {0, 0, "gfx950", 16, 32, "basic", "default"}, /* 62 */
        {0, 1, "gfx942", 16, 8, "mem", "cshuffle"}, /* 63: group_merge=4 */
        {0, 0, "gfx942", 16, 32, "mem", "cshuffle"}, /* 64: rejected */
        {0, 0, "gfx950", 16, 32, "mem", "cshuffle"}, /* 65: lds_k_pad=2, rejected */
    };
    const int i = idx - 54;
    *spec = rocke_implicit_gemm_conv_spec_default();
    if(cfgs[i].is_3d)
        spec->problem
            = rocke_conv_problem_make_3d(1, 4, 6, 6, 32, 32, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1);
    else if(cfgs[i].depthwise)
    {
        spec->problem = rocke_conv_problem_make(2, 14, 14, 64, 64, 3, 3, 1, 1, 1, 1, 1, 1);
        spec->problem.groups = 64;
        spec->group_merge = 4;
    }
    else
        spec->problem = rocke_conv_problem_make(2, 14, 14, 32, 64, 3, 3, 1, 1, 1, 1, 1, 1);
    spec->dtype_a = "fp32";
    spec->dtype_b = "fp32";
    spec->dtype_d = "fp32";
    spec->mma_dtype = "tf32";
    spec->tile_m = 64;
    spec->tile_n = 64;
    spec->tile_k = 32;
    spec->warp_m = 2;
    spec->warp_n = 2;
    spec->warp_tile_m = cfgs[i].m;
    spec->warp_tile_n = cfgs[i].m;
    spec->warp_tile_k = cfgs[i].k;
    spec->pipeline = cfgs[i].pipeline;
    spec->epilogue = cfgs[i].epilogue;
    spec->async_dma = (idx == 60);
    spec->unroll_k = (idx == 61);
    if(idx == 65)
    {
        spec->has_lds_k_pad = true;
        spec->lds_k_pad = 2;
    }
    *arch = cfgs[i].arch;
    return 0;
}

/* Split-K forward on gfx950 MFMA (wave64): the plain mem / compv3 / compv4 K
 * loops over one slice, the tile / warp-tile axes the AOT grid varies, a
 * 4-wide-vector variant, the direct epilogue, and the MFMA loops the split is
 * not wired for.
 * fp16 only: this engine's forward builder stages A/B as f16 whatever the
 * dtype, so bf16 is covered by the representative golden and the GPU numerics
 * instead. Keep in lockstep with _mfma_split_k_spec() in
 * conv_implicit_gemm_emit.py. The rocke_conv_implicit_gemm_spec_guards CTest
 * (platform/tests/instances/conv_implicit_gemm_spec_guards.cpp) compiles this
 * file and checks the C++ verdicts on configs 66..85, which the CI goldens
 * run with the Python engine only: a config added to that range needs a case
 * there too.
 *   66: mem/cshuffle, 32x32x64 tile, 2x2 warps, 16x16x32 warp tile
 *   67: compv3/cshuffle, 64x128x64 tile, 2x4 warps
 *   68: mem/cshuffle, grouped (groups=2), 64x64x64 tile, 4-wide vectors
 *   69: compv4/cshuffle, 128x64x64 tile, 4x2 warps
 *   70: compv3/default (vector_size_c=1), 32x32x32 tile, 2x1 warps
 *   71: basic pipeline: rejected by both engines (mem/compv3/compv4 only)
 *   72: mem with async_dma: rejected by both engines
 *   73: compv3/cshuffle, 128x128x64 tile, 4x2 warps */
static int make_cfg_split_k_mfma(int idx, rocke_implicit_gemm_conv_spec_t* spec, const char** arch)
{
    const rocke_conv_problem_t small_m
        = rocke_conv_problem_make(1, 7, 7, 256, 64, 3, 3, 1, 1, 1, 1, 1, 1);
    const rocke_conv_problem_t csh_64
        = rocke_conv_problem_make(2, 12, 12, 128, 96, 3, 3, 1, 1, 1, 1, 1, 1);
    const rocke_conv_problem_t dflt
        = rocke_conv_problem_make(1, 9, 9, 64, 40, 3, 3, 1, 1, 1, 1, 1, 1);
    rocke_conv_problem_t grouped
        = rocke_conv_problem_make(1, 10, 10, 128, 64, 3, 3, 1, 1, 1, 1, 1, 1);
    grouped.groups = 2;

    *spec = rocke_implicit_gemm_conv_spec_default();
    spec->problem = small_m;
    spec->tile_m = 32;
    spec->tile_n = 32;
    spec->tile_k = 64;
    spec->warp_m = 2;
    spec->warp_n = 2;
    spec->warp_tile_m = 16;
    spec->warp_tile_n = 16;
    spec->warp_tile_k = 32;
    spec->wave_size = 64;
    spec->has_vector_size_a = true;
    spec->vector_size_a = 8;
    spec->has_vector_size_b = true;
    spec->vector_size_b = 8;
    spec->has_vector_size_c = true;
    spec->vector_size_c = 8;
    spec->pipeline = "mem";
    spec->epilogue = "cshuffle";
    spec->split_k = 2;
    *arch = "gfx950";

    const bool wt32 = idx == 67 || idx == 68 || idx == 69 || idx == 73;
    if(wt32)
    {
        spec->warp_tile_m = 32;
        spec->warp_tile_n = 32;
        spec->warp_tile_k = 16;
    }

    switch(idx)
    {
    case 66:
        return 0;
    case 67:
        spec->problem = csh_64;
        spec->tile_m = 64;
        spec->tile_n = 128;
        spec->warp_n = 4;
        spec->pipeline = "compv3";
        return 0;
    case 68:
        spec->problem = grouped;
        spec->tile_m = 64;
        spec->tile_n = 64;
        spec->vector_size_a = 4;
        spec->vector_size_b = 4;
        spec->vector_size_c = 4;
        return 0;
    case 69:
        spec->tile_m = 128;
        spec->tile_n = 64;
        spec->warp_m = 4;
        spec->warp_n = 2;
        spec->pipeline = "compv4";
        return 0;
    case 70:
        spec->problem = dflt;
        spec->tile_k = 32;
        spec->warp_n = 1;
        spec->pipeline = "compv3";
        spec->epilogue = "default";
        spec->vector_size_c = 1;
        return 0;
    case 71:
        spec->pipeline = "basic";
        return 0;
    case 72:
        spec->async_dma = true;
        return 0;
    case 73:
        spec->tile_m = 128;
        spec->tile_n = 128;
        spec->warp_m = 4;
        spec->pipeline = "compv3";
        return 0;
    default:
        return -1;
    }
}

/* The gfx950 spec gate around the MFMA split: a missing (NULL) pipeline or
 * epilogue, split or not, and the LDS budget that charges compv4 one A/B
 * tile. Built on config 66 (mem/cshuffle, 32x32x64 tile, split_k=2). Keep in
 * lockstep with _mfma_gate_spec() in conv_implicit_gemm_emit.py.
 *   74: unsplit, pipeline NULL: rejected by both engines
 *   75: unsplit, epilogue NULL: rejected by both engines
 *   76: split, pipeline NULL: rejected by both engines
 *   77: split, epilogue NULL: rejected by both engines
 *   78: split compv4/cshuffle, 128x64x256 tile, 4x2 warps, lds_k_pad=0 (fits
 *       only because compv4 is charged one A/B tile)
 *   79: config 78 unsplit
 *   80: config 78 unsplit on compv3 with unroll_k (two A/B tiles): rejected
 *       by both engines on the LDS budget
 *   81: config 78 at 256x128x256: rejected by both engines on the LDS budget */
static int make_cfg_mfma_gate(int idx, rocke_implicit_gemm_conv_spec_t* spec, const char** arch)
{
    if(make_cfg_split_k_mfma(66, spec, arch) != 0)
        return -1;
    if(idx >= 78)
    {
        spec->tile_m = 128;
        spec->tile_n = 64;
        spec->tile_k = 256;
        spec->warp_m = 4;
        spec->warp_n = 2;
        spec->warp_tile_m = 32;
        spec->warp_tile_n = 32;
        spec->warp_tile_k = 16;
        spec->has_lds_k_pad = true;
        spec->lds_k_pad = 0;
        spec->pipeline = "compv4";
    }
    switch(idx)
    {
    case 74:
        spec->split_k = 1;
        spec->pipeline = NULL;
        return 0;
    case 75:
        spec->split_k = 1;
        spec->epilogue = NULL;
        return 0;
    case 76:
        spec->pipeline = NULL;
        return 0;
    case 77:
        spec->epilogue = NULL;
        return 0;
    case 78:
        return 0;
    case 79:
        spec->split_k = 1;
        return 0;
    case 80:
        spec->split_k = 1;
        spec->pipeline = "compv3";
        spec->unroll_k = true;
        return 0;
    case 81:
        spec->tile_m = 256;
        spec->tile_n = 128;
        return 0;
    default:
        return -1;
    }
}

/* fp32 output under the two-stage split: Stage 2 casts to fp16 / bf16 only,
 * on every family, and the cluster form folds the slabs with the same cast,
 * so both engines reject either. Keep in lockstep with
 * _tf32_split_k_spec() in conv_implicit_gemm_emit.py.
 *   82: config 54 (gfx950 TF32 16x16x32, mem/cshuffle) with split_k=2
 *   83: config 56 (gfx942 TF32 16x16x8, compv3/cshuffle) with split_k=2
 *   84: config 41 (gfx1250 WMMA, mem/cshuffle, split_k=2) in fp32 storage on
 *       the 16x16x4 fp32 atom, 4-wide vectors
 *   85: config 84 in the cluster form (split_k_cluster=4): rejected by both */
static int make_cfg_tf32_split_k(int idx, rocke_implicit_gemm_conv_spec_t* spec, const char** arch)
{
    if(idx == 84 || idx == 85)
    {
        if(make_cfg_split_k(41, spec, arch) != 0)
            return -1;
        spec->dtype_a = "fp32";
        spec->dtype_b = "fp32";
        spec->dtype_d = "fp32";
        spec->warp_tile_k = 4;
        spec->vector_size_a = 4;
        spec->vector_size_b = 4;
        spec->vector_size_c = 4;
        spec->split_k_cluster = (idx == 85) ? 4 : 0;
        return 0;
    }
    if(idx != 82 && idx != 83)
        return -1;
    if(make_cfg_tf32(idx == 82 ? 54 : 56, spec, arch) != 0)
        return -1;
    spec->split_k = 2;
    return 0;
}

/* The default epilogue stores one element per lane, so the validator rejects
 * it with the vec_c the spec would auto-derive from kpg (8 here). Configs that
 * leave vector_size_c unset get vector_size_c=1 -- otherwise they do not build
 * and the gate only compares two rejections. Mirrors _spec() in
 * conv_implicit_gemm_emit.py. */
static int make_cfg(int idx, rocke_implicit_gemm_conv_spec_t* spec, const char** arch)
{
    if(idx >= 40 && idx <= 53)
    {
        if(make_cfg_split_k(idx, spec, arch) != 0)
            return -1;
    }
    else if(idx >= 54 && idx <= 65)
    {
        if(make_cfg_tf32(idx, spec, arch) != 0)
            return -1;
    }
    else if(idx >= 66 && idx <= 73)
    {
        if(make_cfg_split_k_mfma(idx, spec, arch) != 0)
            return -1;
    }
    else if(idx >= 74 && idx <= 81)
    {
        if(make_cfg_mfma_gate(idx, spec, arch) != 0)
            return -1;
    }
    else if(idx >= 82 && idx <= 85)
    {
        if(make_cfg_tf32_split_k(idx, spec, arch) != 0)
            return -1;
    }
    else if(make_cfg_raw(idx, spec, arch) != 0)
        return -1;
    if(spec->epilogue != NULL && strcmp(spec->epilogue, "default") == 0 && !spec->has_vector_size_c)
    {
        spec->has_vector_size_c = true;
        spec->vector_size_c = 1;
    }
    return 0;
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

    rocke_implicit_gemm_conv_spec_t spec;
    const char* arch = "gfx950";
    if(make_cfg(idx, &spec, &arch) != 0)
    {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 2;
    }

    rocke_ir_builder_t b;
    rocke_kernel_def_t* kernel = rocke_build_implicit_gemm_conv_new(&b, &spec, arch, NULL);
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
