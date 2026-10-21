/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/conv_direct_grouped_emit.c -- C-side emitter for the direct
 * grouped convolution parity harness. Selects one of N sampled spec configs by
 * argv[1] (the config index), builds the rocke_direct_conv_16c_spec_t /
 * rocke_direct_conv_4c_spec_t / rocke_direct_conv_8c_spec_t /
 * rocke_direct_conv_32c_spec_t / rocke_direct_depthwise_spec_t /
 * rocke_direct_depthwise_col_spec_t / rocke_direct_conv_wgrad_spec_t (and,
 * from index 42, the non-grouped rocke_direct_conv_nongrouped_spec_t; from
 * index 57 the output-stationary rocke_direct_depthwise_tiled_spec_t; from
 * index 63 the dilated non-grouped spec; from index 66 specs both engines must
 * reject; from index 71 the runtime-channel wgrad spec; from index 75 the
 * parametric rocke_direct_conv_spec_t and the MFMA dgrad weight transforms,
 * baked and runtime-channel; from index 95 the split-C non-grouped spec and its
 * rocke_direct_conv_split_reduce_spec_t second pass; from index 103 the grouped /
 * group-merge non-grouped spec) identically to the Python emitter
 * conv_direct_grouped_emit.py, builds the
 * kernel via the matching rocke_build_direct_conv_*_new function and lowers via
 * rocke_lower_kernel_to_llvm (per-config arch, flavor AUTO) and prints the .ll
 * to stdout so the two outputs can be byte-compared.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/instance_conv_direct_grouped.h"
#include "rocke/instance_conv_direct_nongrouped.h"
#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"
#include "rocke/verify.h"

enum
{
    KIND_16C = 0,
    KIND_4C = 1,
    KIND_8C = 2,
    KIND_32C = 3,
    KIND_DW = 4,
    KIND_SPATIAL = 5,
    KIND_DGRAD = 6,
    KIND_DW_DGRAD = 7,
    KIND_WGRAD = 8,
    KIND_DWCOL = 9,
    KIND_NONGROUPED = 10,
    KIND_DW_TILED = 11,
    KIND_GENERIC = 12,
    KIND_TRANSPOSE = 13,
    KIND_REORG = 14,
    KIND_SPLIT_REDUCE = 15
};

#define NONGROUPED_CFG_BASE 42
/* Output-stationary depthwise configs follow the 15 non-grouped ones. */
#define TILED_CFG_BASE 57
/* Dilated non-grouped configs follow the 6 tiled ones. */
#define NONGROUPED_DIL_CFG_BASE 63
/* Specs both engines must reject follow the 3 dilated non-grouped ones. */
#define REJECT_CFG_BASE 66
/* Runtime-channel wgrad configs 71..74 (explicit cases in make_cfg) follow the
 * 5 must-reject ones; the parametric DirectConvSpec + dgrad weight-transform
 * configs follow them. */
#define GENERIC_CFG_BASE 75
/* Split-C non-grouped configs and the split-C reduction follow the 20 generic
 * ones. */
#define NONGROUPED_SPLIT_CFG_BASE 95
/* Grouped (grouped = true, group_merge) non-grouped configs follow the 8 split-C
 * ones. */
#define NONGROUPED_GROUPED_CFG_BASE 103

static rocke_direct_conv_problem_t
    mk_dw_problem(int N, int H, int W, int groups, int KH, int PAD, int stride, const char* dtype)
{
    rocke_direct_conv_problem_t p = rocke_direct_conv_problem_default();
    p.N = N;
    p.H = H;
    p.W = W;
    p.groups = groups;
    p.cpg = 1;
    p.kpg = 1;
    p.KH = KH;
    p.KW = KH;
    p.PAD = PAD;
    p.stride = stride;
    p.dtype = dtype;
    return p;
}

/* Problem helper mirroring the Python emitter's _gp(): N=2, H=20, W=24,
 * square KHxKH filter, stride 1. */
static rocke_direct_conv_problem_t
    mk_gp(int groups, int cpg, int kpg, int KH, int PAD, const char* dtype)
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
    p.stride = 1;
    p.dtype = dtype;
    return p;
}

/* Parametric DirectConvSpec + MFMA dgrad weight-transform configs, emitted as
 * GENERIC_CFG_BASE + idx (see _generic_spec in the Python emitter). Returns 0
 * on success, -1 if unknown. */
static int make_generic_cfg(int idx,
                            int* kind,
                            rocke_direct_conv_spec_t* sg,
                            rocke_direct_transpose_weights_dgrad_spec_t* st,
                            rocke_direct_reorganize_weights_spec_t* sr)
{
    *sg = rocke_direct_conv_spec_default();
    memset(st, 0, sizeof(*st));
    memset(sr, 0, sizeof(*sr));
    *kind = KIND_GENERIC;
    switch(idx)
    {
    case 0:
        sg->problem = mk_gp(8, 16, 16, 3, 1, "fp16");
        return 0;
    case 1:
        sg->problem = mk_gp(4, 24, 40, 3, 1, "bf16");
        sg->block_q = 32;
        sg->block_groups = 2;
        sg->double_buffer = false;
        sg->block_h = 8;
        return 0;
    case 2:
        sg->problem = mk_gp(1, 192, 48, 3, 1, "bf16");
        sg->block_groups = 1;
        sg->block_h = 8;
        sg->waves_k = 4;
        return 0;
    case 3:
        sg->problem = mk_gp(1, 192, 48, 3, 1, "bf16");
        sg->block_groups = 1;
        sg->block_h = 8;
        sg->waves_k = 2;
        sg->fold_k32 = true;
        return 0;
    case 4:
        sg->problem = mk_gp(2, 48, 20, 3, 1, "fp16");
        sg->block_q = 32;
        sg->block_groups = 1;
        sg->waves_q = 2;
        sg->runtime_k_loop = true;
        return 0;
    case 5:
        sg->problem = mk_gp(16, 32, 32, 3, 1, "fp16");
        sg->block_groups = 4;
        sg->persistent_grid = true;
        return 0;
    case 6:
        sg->problem = mk_gp(4, 16, 16, 5, 2, "fp16");
        sg->block_groups = 2;
        return 0;
    case 7:
        sg->problem = mk_gp(1, 24, 24, 3, 1, "bf16");
        sg->block_q = 32;
        sg->block_groups = 1;
        sg->block_h = 8;
        sg->runtime_channels = true;
        sg->k_atoms = 2;
        sg->m_tiles = 2;
        return 0;
    case 8:
        sg->problem = mk_gp(1, 192, 48, 3, 1, "bf16");
        sg->block_groups = 1;
        sg->block_h = 8;
        sg->waves_k = 2;
        sg->fold_k32 = true;
        sg->runtime_channels = true;
        sg->k_atoms = 6;
        sg->m_tiles = 3;
        return 0;
    case 9:
        sg->problem = mk_gp(1, 320, 4, 3, 1, "fp16");
        sg->block_groups = 1;
        sg->block_h = 8;
        sg->waves_k = 4;
        sg->runtime_channels = true;
        sg->k_atoms = 20;
        sg->m_tiles = 1;
        return 0;
    case 10:
        sg->problem = mk_gp(4, 40, 20, 3, 1, "fp16");
        sg->block_q = 32;
        sg->block_groups = 2;
        sg->waves_q = 2;
        sg->runtime_k_loop = true;
        sg->runtime_channels = true;
        sg->k_atoms = 3;
        sg->m_tiles = 2;
        return 0;
    case 11:
        sg->problem = mk_gp(8, 64, 64, 3, 1, "fp16");
        sg->block_groups = 2;
        sg->persistent_grid = true;
        sg->runtime_channels = true;
        sg->k_atoms = 4;
        sg->m_tiles = 4;
        return 0;
    case 12:
        sg->problem = mk_gp(2, 16, 16, 5, 2, "bf16");
        sg->block_groups = 1;
        sg->double_buffer = false;
        sg->runtime_channels = true;
        sg->k_atoms = 1;
        sg->m_tiles = 1;
        return 0;
    case 13:
        *kind = KIND_TRANSPOSE;
        st->problem = mk_gp(2, 24, 40, 3, 1, "fp16");
        return 0;
    case 14:
        *kind = KIND_TRANSPOSE;
        st->problem = mk_gp(1, 48, 192, 3, 1, "bf16");
        st->runtime_channels = true;
        return 0;
    case 15:
        *kind = KIND_REORG;
        sr->problem = mk_gp(2, 24, 40, 3, 1, "bf16");
        return 0;
    case 16:
        *kind = KIND_REORG;
        sr->problem = mk_gp(1, 48, 192, 3, 1, "fp16");
        sr->fold_k32 = true;
        return 0;
    case 17:
        *kind = KIND_REORG;
        sr->problem = mk_gp(1, 24, 40, 3, 1, "fp16");
        sr->runtime_channels = true;
        return 0;
    case 18:
        *kind = KIND_REORG;
        sr->problem = mk_gp(1, 4, 320, 3, 1, "bf16");
        sr->fold_k32 = true;
        sr->runtime_channels = true;
        return 0;
    case 19:
        *kind = KIND_TRANSPOSE;
        st->problem = mk_gp(1, 16, 16, 5, 2, "fp16");
        st->runtime_channels = true;
        return 0;
    default:
        return -1;
    }
}

/* DirectDepthwiseTiledSpec configs, emitted as TILED_CFG_BASE + idx (mirrors
 * Python _tiled_spec). Returns 0 on success, -1 past the last one. */
static int make_tiled_cfg(int idx, rocke_direct_depthwise_tiled_spec_t* s, const char** arch)
{
    *s = rocke_direct_depthwise_tiled_spec_default();
    *arch = "gfx950";
    switch(idx)
    {
    case 0:
        /* 31x31: the large-filter case the row-streaming kernel cannot compile. */
        s->problem = mk_dw_problem(1, 56, 56, 192, 31, 15, 1, "bf16");
        s->block_w = 8;
        s->block_h = 2;
        return 0;
    case 1:
        /* fp16 7x7, filter-row loop unrolled. */
        s->problem = mk_dw_problem(4, 14, 14, 128, 7, 3, 1, "fp16");
        s->block_w = 14;
        s->block_h = 7;
        s->unroll_rows = true;
        return 0;
    case 2:
        /* Stride 2: two phases with a window each; two waves; gfx942. */
        s->problem = mk_dw_problem(2, 28, 28, 256, 5, 2, 2, "bf16");
        s->block_w = 8;
        s->block_h = 4;
        s->block_waves = 2;
        *arch = "gfx942";
        return 0;
    case 3:
        /* block_h 1: no carried window. */
        s->problem = mk_dw_problem(2, 16, 16, 64, 3, 1, 1, "bf16");
        s->block_w = 16;
        s->block_h = 1;
        return 0;
    case 4:
        /* Non-"same" padding: outputs are indexed directly. */
        s->problem = mk_dw_problem(2, 16, 16, 64, 3, 0, 1, "fp16");
        s->block_w = 4;
        s->block_h = 2;
        return 0;
    case 5:
        /* Over the register budget: both engines reject. */
        s->problem = mk_dw_problem(1, 56, 56, 192, 31, 15, 1, "bf16");
        s->block_w = 16;
        s->block_h = 4;
        return 0;
    default:
        return -1;
    }
}

static rocke_direct_conv_problem_t mk_nongrouped_problem(
    int N, int H, int W, int C, int K, int KH, int KW, int PAD, int stride, const char* dtype)
{
    rocke_direct_conv_problem_t p = rocke_direct_conv_problem_default();
    p.N = N;
    p.H = H;
    p.W = W;
    p.groups = 1;
    p.cpg = C;
    p.kpg = K;
    p.KH = KH;
    p.KW = KW;
    p.PAD = PAD;
    p.stride = stride;
    p.dtype = dtype;
    return p;
}

/* Python _BASE: t8x32x64, ck32, 2x2 waves, iglp 0 (everything else default). */
static rocke_direct_conv_nongrouped_spec_t mk_nongrouped_spec(rocke_direct_conv_problem_t p)
{
    rocke_direct_conv_nongrouped_spec_t s = rocke_direct_conv_nongrouped_spec_default();
    s.problem = p;
    s.tile_h = 8;
    s.tile_w = 32;
    s.tile_k = 64;
    s.ck = 32;
    s.waves_m = 2;
    s.waves_n = 2;
    s.iglp = 0;
    return s;
}

/* Dilated non-grouped configs, emitted as NONGROUPED_DIL_CFG_BASE + idx (mirrors
 * Python _nongrouped_dil_spec). Returns 0 on success, -1 past the last one. */
static int
    make_nongrouped_dil_cfg(int idx, rocke_direct_conv_nongrouped_spec_t* s, const char** arch)
{
    rocke_direct_conv_problem_t p;
    *arch = "gfx950";
    switch(idx)
    {
    case 0:
        /* Row dilation 3, PAD 1: a taller halo, tap rows 3 input rows apart. */
        p = mk_nongrouped_problem(2, 16, 32, 64, 128, 3, 3, 1, 1, "bf16");
        p.dil_h = 3;
        *s = mk_nongrouped_spec(p);
        return 0;
    case 1:
        /* Column dilation 4, PAD 4: a wider halo, tap columns 4 apart. */
        p = mk_nongrouped_problem(1, 32, 32, 64, 64, 3, 3, 4, 1, "bf16");
        p.dil_w = 4;
        *s = mk_nongrouped_spec(p);
        return 0;
    case 2:
        /* fp16, both axes dilated, stride 2, double-buffered LDS. */
        p = mk_nongrouped_problem(1, 32, 64, 64, 64, 3, 3, 2, 2, "fp16");
        p.dil_h = 2;
        p.dil_w = 2;
        *s = mk_nongrouped_spec(p);
        s->ck = 16;
        s->double_buffer = true;
        return 0;
    default:
        return -1;
    }
}

/* Non-grouped (groups == 1) DirectNongroupedConvSpec configs, emitted as indices
 * NONGROUPED_CFG_BASE + idx. Returns 0 on success, -1 if unknown. */
static int make_nongrouped_cfg(int idx, rocke_direct_conv_nongrouped_spec_t* s, const char** arch)
{
    *arch = "gfx950";
    switch(idx)
    {
    case 0:
        *s = mk_nongrouped_spec(mk_nongrouped_problem(2, 16, 32, 64, 128, 3, 3, 1, 1, "bf16"));
        return 0;
    case 1:
        *s = mk_nongrouped_spec(mk_nongrouped_problem(2, 16, 32, 64, 128, 3, 3, 1, 1, "fp16"));
        return 0;
    case 2:
        *s = mk_nongrouped_spec(mk_nongrouped_problem(2, 16, 32, 64, 128, 3, 3, 1, 1, "bf16"));
        s->double_buffer = true;
        return 0;
    case 3:
        *s = mk_nongrouped_spec(mk_nongrouped_problem(2, 16, 32, 64, 128, 3, 3, 1, 1, "bf16"));
        s->chiplet_swizzle = false;
        return 0;
    case 4:
        *s = mk_nongrouped_spec(mk_nongrouped_problem(2, 16, 32, 64, 128, 3, 3, 1, 1, "bf16"));
        s->iglp = ROCKE_DCONV_NONGROUPED_IGLP_NONE;
        s->waves_per_eu = 3;
        return 0;
    case 5:
        *s = mk_nongrouped_spec(mk_nongrouped_problem(1, 32, 64, 64, 64, 3, 3, 1, 2, "bf16"));
        return 0;
    case 6:
        *s = mk_nongrouped_spec(mk_nongrouped_problem(1, 16, 32, 64, 64, 1, 1, 0, 1, "bf16"));
        return 0;
    case 7:
        *s = mk_nongrouped_spec(mk_nongrouped_problem(2, 20, 40, 64, 64, 3, 3, 1, 1, "bf16"));
        s->tile_w = 48;
        s->tile_k = 32;
        s->waves_m = 1;
        s->atom = "16x16x32";
        return 0;
    case 8:
        *s = mk_nongrouped_spec(mk_nongrouped_problem(1, 16, 32, 64, 64, 3, 3, 1, 1, "fp16"));
        s->tile_w = 16;
        s->tile_k = 32;
        s->ck = 16;
        s->waves_m = 1;
        s->atom = "16x16x16";
        return 0;
    case 9:
        *s = mk_nongrouped_spec(mk_nongrouped_problem(1, 16, 32, 64, 64, 3, 3, 1, 1, "fp16"));
        s->atom = "32x32x8";
        s->ck = 16;
        *arch = "gfx942";
        return 0;
    case 10:
        *s = mk_nongrouped_spec(mk_nongrouped_problem(1, 16, 48, 32, 64, 3, 3, 1, 1, "bf16"));
        s->tile_w = 48;
        s->tile_k = 32;
        s->ck = 16;
        s->waves_m = 1;
        s->waves_n = 4;
        s->atom = "16x16x16";
        *arch = "gfx942";
        return 0;
    case 11:
        *s = mk_nongrouped_spec(mk_nongrouped_problem(1, 16, 32, 64, 96, 5, 5, 2, 1, "bf16"));
        s->swizzle_wgm = 1;
        return 0;
    case 12:
        *s = mk_nongrouped_spec(mk_nongrouped_problem(1, 20, 64, 64, 64, 3, 3, 1, 1, "bf16"));
        s->tile_h = 16;
        s->tile_w = 64;
        s->waves_n = 4;
        return 0;
    case 13:
        *s = mk_nongrouped_spec(mk_nongrouped_problem(4, 64, 64, 640, 640, 3, 3, 1, 1, "bf16"));
        s->tile_h = 16;
        s->tile_w = 64;
        s->tile_k = 128;
        s->ck = 16;
        s->waves_m = 2;
        s->waves_n = 4;
        return 0;
    case 14:
        *s = mk_nongrouped_spec(mk_nongrouped_problem(1, 32, 64, 64, 64, 3, 3, 1, 2, "fp16"));
        s->ck = 16;
        s->double_buffer = true;
        s->iglp = 1;
        s->chiplet_swizzle = false;
        return 0;
    default:
        return -1;
    }
}

/* Split-C configs, emitted as NONGROUPED_SPLIT_CFG_BASE + idx (mirrors Python
 * _nongrouped_split_spec). Returns 0 on success, -1 past the last one. */
static int make_nongrouped_split_cfg(int idx,
                                     int* kind,
                                     rocke_direct_conv_nongrouped_spec_t* s,
                                     rocke_direct_conv_split_reduce_spec_t* sred,
                                     const char** arch)
{
    *arch = "gfx950";
    *kind = KIND_NONGROUPED;
    switch(idx)
    {
    case 0:
        *s = mk_nongrouped_spec(mk_nongrouped_problem(2, 14, 14, 256, 128, 3, 3, 1, 1, "bf16"));
        s->split_c = 2;
        return 0;
    case 1:
        *s = mk_nongrouped_spec(mk_nongrouped_problem(1, 24, 24, 128, 64, 3, 3, 1, 2, "fp16"));
        s->ck = 16;
        s->double_buffer = true;
        s->split_c = 4;
        return 0;
    case 2:
        *s = mk_nongrouped_spec(mk_nongrouped_problem(1, 7, 7, 512, 64, 3, 3, 1, 1, "bf16"));
        s->chiplet_swizzle = false;
        s->waves_per_eu = 3;
        s->split_c = 16;
        return 0;
    case 3:
        *s = mk_nongrouped_spec(mk_nongrouped_problem(1, 12, 12, 320, 64, 3, 3, 1, 1, "fp16"));
        s->tile_h = 12;
        s->tile_w = 16;
        s->waves_n = 4;
        s->atom = "16x16x32";
        s->split_c = 4;
        return 0;
    case 4:
        *s = mk_nongrouped_spec(mk_nongrouped_problem(1, 28, 28, 256, 64, 3, 3, 1, 1, "fp16"));
        s->tile_k = 32;
        s->waves_m = 1;
        s->atom = "32x32x8";
        s->ck = 16;
        s->double_buffer = true;
        s->chiplet_swizzle = false;
        s->split_c = 2;
        *arch = "gfx942";
        return 0;
    case 5:
        *sred = rocke_direct_conv_split_reduce_spec_default();
        sred->dtype = "fp16";
        sred->split_c = 2;
        *kind = KIND_SPLIT_REDUCE;
        return 0;
    case 6:
        *sred = rocke_direct_conv_split_reduce_spec_default();
        sred->dtype = "bf16";
        sred->split_c = 8;
        sred->threads = 128;
        *kind = KIND_SPLIT_REDUCE;
        return 0;
    case 7:
        /* Must reject: 16 splits of a C that has only 8 chunks of ck=32. */
        *s = mk_nongrouped_spec(mk_nongrouped_problem(1, 7, 7, 256, 64, 3, 3, 1, 1, "bf16"));
        s->split_c = 16;
        return 0;
    default:
        return -1;
    }
}

static rocke_direct_conv_problem_t mk_grouped_problem(
    int N, int H, int W, int groups, int cpg, int kpg, int stride, const char* dtype)
{
    rocke_direct_conv_problem_t p
        = mk_nongrouped_problem(N, H, W, cpg, kpg, 3, 3, 1, stride, dtype);
    p.groups = groups;
    return p;
}

/* Grouped configs, emitted as NONGROUPED_GROUPED_CFG_BASE + idx (mirrors Python
 * _nongrouped_grouped_spec); 7..10 are must-reject. Returns 0 on success, -1 past
 * the last one. */
static int
    make_nongrouped_grouped_cfg(int idx, rocke_direct_conv_nongrouped_spec_t* s, const char** arch)
{
    *arch = "gfx950";
    switch(idx)
    {
    case 0:
        /* group_merge 1, fp16, cpg 32: plain weight path, K_M masks. */
        *s = mk_nongrouped_spec(mk_grouped_problem(2, 14, 20, 4, 32, 32, 1, "fp16"));
        s->grouped = true;
        s->group_merge = 1;
        return 0;
    case 1:
        /* group_merge 8 of cpg 4, bf16, 16x16x32 (two pieces), ck 32. */
        *s = mk_nongrouped_spec(mk_grouped_problem(1, 16, 16, 16, 4, 4, 1, "bf16"));
        s->tile_k = 32;
        s->atom = "16x16x32";
        s->ck = 32;
        s->grouped = true;
        s->group_merge = 8;
        return 0;
    case 2:
        /* group_merge 4 of cpg 8, fp16, stride 2, 32x32x16, ck 16. */
        *s = mk_nongrouped_spec(mk_grouped_problem(1, 24, 24, 8, 8, 8, 2, "fp16"));
        s->tile_k = 32;
        s->waves_m = 1;
        s->atom = "32x32x16";
        s->ck = 16;
        s->grouped = true;
        s->group_merge = 4;
        return 0;
    case 3:
        /* group_merge 4 of cpg 12, bf16, 16x16x16 (one piece), ck 16. */
        *s = mk_nongrouped_spec(mk_grouped_problem(1, 16, 16, 8, 12, 12, 1, "bf16"));
        s->tile_k = 32;
        s->atom = "16x16x16";
        s->ck = 16;
        s->grouped = true;
        s->group_merge = 4;
        return 0;
    case 4:
        /* Grouped + split_c 2 + double-buffered LDS. */
        *s = mk_nongrouped_spec(mk_grouped_problem(1, 14, 14, 4, 64, 32, 1, "fp16"));
        s->double_buffer = true;
        s->split_c = 2;
        s->grouped = true;
        s->group_merge = 2;
        return 0;
    case 5:
        /* Grouped, no chiplet swizzle, waves_per_eu, bf16, group_merge 2. */
        *s = mk_nongrouped_spec(mk_grouped_problem(2, 12, 20, 6, 16, 16, 1, "bf16"));
        s->tile_k = 32;
        s->waves_m = 1;
        s->chiplet_swizzle = false;
        s->waves_per_eu = 3;
        s->grouped = true;
        s->group_merge = 2;
        return 0;
    case 6:
        /* gfx942 32x32x8, fp16, group_merge 4 of cpg 8, ck 16. */
        *s = mk_nongrouped_spec(mk_grouped_problem(1, 20, 20, 8, 8, 8, 1, "fp16"));
        s->tile_k = 32;
        s->waves_m = 1;
        s->atom = "32x32x8";
        s->ck = 16;
        s->grouped = true;
        s->group_merge = 4;
        *arch = "gfx942";
        return 0;
    case 7:
        /* Must reject: group_merge 3 does not divide 8 groups. */
        *s = mk_nongrouped_spec(mk_grouped_problem(1, 16, 16, 8, 8, 8, 1, "bf16"));
        s->tile_k = 32;
        s->grouped = true;
        s->group_merge = 3;
        return 0;
    case 8:
        /* Must reject: cpg 6 is not a whole number of 4-channel weight loads. */
        *s = mk_nongrouped_spec(mk_grouped_problem(1, 16, 16, 4, 6, 8, 1, "fp16"));
        s->tile_k = 32;
        s->waves_m = 1;
        s->atom = "32x32x8";
        s->ck = 8;
        s->grouped = true;
        s->group_merge = 4;
        return 0;
    case 9:
        /* Must reject: group_merge without grouped. */
        *s = mk_nongrouped_spec(mk_nongrouped_problem(1, 16, 16, 64, 64, 3, 3, 1, 1, "bf16"));
        s->group_merge = 2;
        return 0;
    case 10:
        /* Must reject: grouped with no groups (0 % group_merge is 0). */
        *s = mk_nongrouped_spec(mk_grouped_problem(1, 16, 16, 0, 8, 8, 1, "bf16"));
        s->tile_k = 32;
        s->grouped = true;
        s->group_merge = 1;
        return 0;
    default:
        return -1;
    }
}

/* Must-reject configs, emitted as REJECT_CFG_BASE + idx (mirrors Python
 * _reject_spec). Each pins a validator branch whose failure mode the engines
 * must share. Returns 0 on success, -1 past the last one. */
static int make_reject_cfg(int idx,
                           int* kind,
                           rocke_direct_conv_16c_spec_t* s16,
                           rocke_direct_depthwise_col_spec_t* sdwc,
                           rocke_direct_conv_nongrouped_spec_t* snongrouped,
                           rocke_direct_depthwise_tiled_spec_t* stiled,
                           const char** arch)
{
    rocke_direct_conv_problem_t p;
    *arch = "gfx950";
    switch(idx)
    {
    case 0:
        /* The column-streamed tap indexing has no dilation term. */
        p = mk_dw_problem(2, 8, 8, 128, 3, 1, 1, "fp16");
        p.dil_h = 2;
        *sdwc = rocke_direct_depthwise_col_spec_default();
        sdwc->problem = p;
        sdwc->block_w = 4;
        sdwc->block_waves = 2;
        sdwc->dtype = "fp16";
        *kind = KIND_DWCOL;
        return 0;
    case 1:
        /* Tiled: a filter larger than the padded input (Ho = Wo = -26). */
        *stiled = rocke_direct_depthwise_tiled_spec_default();
        stiled->problem = mk_dw_problem(1, 4, 4, 64, 31, 0, 1, "fp16");
        stiled->block_w = 8;
        stiled->block_h = 2;
        *kind = KIND_DW_TILED;
        return 0;
    case 2:
        /* Tiled: H + 2*PAD - KH = -1 at stride 2. Ho floors to 0; a truncating
         * division would make it 1. */
        *stiled = rocke_direct_depthwise_tiled_spec_default();
        stiled->problem = mk_dw_problem(1, 4, 16, 64, 5, 0, 2, "fp16");
        stiled->block_w = 4;
        stiled->block_h = 2;
        *kind = KIND_DW_TILED;
        return 0;
    case 3:
        /* Non-grouped: a zero dilation is refused, not read as 1. */
        p = mk_nongrouped_problem(2, 16, 32, 64, 128, 3, 3, 1, 1, "bf16");
        p.dil_h = 0;
        *snongrouped = mk_nongrouped_spec(p);
        *kind = KIND_NONGROUPED;
        return 0;
    case 4:
        /* 16c: any dilation other than 1, zero included, is refused. */
        p = rocke_direct_conv_problem_default();
        p.N = 2;
        p.H = 16;
        p.W = 16;
        p.groups = 16;
        p.cpg = 16;
        p.kpg = 16;
        p.dil_w = 0;
        *s16 = rocke_direct_conv_16c_spec_default();
        s16->problem = p;
        s16->block_groups = 4;
        *kind = KIND_16C;
        return 0;
    default:
        return -1;
    }
}

/* Fill the config for index `idx`. Returns 0 on success, -1 if unknown.
 * On success sets *kind, the matching spec struct, and *arch. */
static int make_cfg(int idx,
                    int* kind,
                    rocke_direct_conv_16c_spec_t* s16,
                    rocke_direct_conv_4c_spec_t* s4,
                    rocke_direct_conv_8c_spec_t* s8,
                    rocke_direct_conv_32c_spec_t* s32,
                    rocke_direct_depthwise_spec_t* sdw,
                    rocke_direct_depthwise_spatial_spec_t* ssp,
                    rocke_direct_depthwise_col_spec_t* sdwc,
                    rocke_direct_conv_dgrad_spec_t* sdgrad,
                    rocke_direct_depthwise_dgrad_spec_t* sdw_dgrad,
                    rocke_direct_conv_wgrad_spec_t* swg,
                    rocke_direct_conv_nongrouped_spec_t* snongrouped,
                    rocke_direct_depthwise_tiled_spec_t* stiled,
                    const char** arch)
{
    rocke_direct_conv_problem_t p = rocke_direct_conv_problem_default();
    p.KH = 3;
    p.KW = 3;
    p.PAD = 1;
    p.stride = 1;

    switch(idx)
    {
    case 0:
        p.N = 32;
        p.H = 200;
        p.W = 200;
        p.groups = 16;
        p.cpg = 16;
        p.kpg = 16;
        *s16 = rocke_direct_conv_16c_spec_default();
        s16->problem = p;
        s16->block_groups = 4;
        s16->fold_k32 = true;
        *kind = KIND_16C;
        *arch = "gfx950";
        return 0;
    case 1:
        p.N = 32;
        p.H = 200;
        p.W = 200;
        p.groups = 16;
        p.cpg = 16;
        p.kpg = 16;
        *s16 = rocke_direct_conv_16c_spec_default();
        s16->problem = p;
        s16->block_groups = 8;
        s16->fold_k32 = true;
        *kind = KIND_16C;
        *arch = "gfx950";
        return 0;
    case 2:
        p.N = 32;
        p.H = 200;
        p.W = 200;
        p.groups = 64;
        p.cpg = 4;
        p.kpg = 4;
        *s4 = rocke_direct_conv_4c_spec_default();
        s4->problem = p;
        s4->block_q = 4;
        s4->block_groups = 16;
        *kind = KIND_4C;
        *arch = "gfx950";
        return 0;
    case 3:
        p.N = 32;
        p.H = 200;
        p.W = 200;
        p.groups = 64;
        p.cpg = 4;
        p.kpg = 4;
        *s4 = rocke_direct_conv_4c_spec_default();
        s4->problem = p;
        s4->block_q = 8;
        s4->block_groups = 16;
        *kind = KIND_4C;
        *arch = "gfx950";
        return 0;
    case 4:
        p.N = 1;
        p.H = 8;
        p.W = 8;
        p.groups = 8;
        p.cpg = 16;
        p.kpg = 16;
        *s16 = rocke_direct_conv_16c_spec_default();
        s16->problem = p;
        s16->block_groups = 1;
        s16->fold_k32 = false;
        *kind = KIND_16C;
        *arch = "gfx942";
        return 0;
    case 5:
        p.N = 1;
        p.H = 8;
        p.W = 8;
        p.groups = 16;
        p.cpg = 4;
        p.kpg = 4;
        *s4 = rocke_direct_conv_4c_spec_default();
        s4->problem = p;
        s4->block_q = 4;
        s4->block_groups = 16;
        *kind = KIND_4C;
        *arch = "gfx950";
        return 0;
    case 6:
        p.N = 32;
        p.H = 200;
        p.W = 200;
        p.groups = 16;
        p.cpg = 8;
        p.kpg = 8;
        *s8 = rocke_direct_conv_8c_spec_default();
        s8->problem = p;
        s8->block_q = 16;
        s8->block_groups = 8;
        s8->double_buffer = true;
        *kind = KIND_8C;
        *arch = "gfx950";
        return 0;
    case 7:
        p.N = 32;
        p.H = 200;
        p.W = 200;
        p.groups = 8;
        p.cpg = 32;
        p.kpg = 32;
        *s32 = rocke_direct_conv_32c_spec_default();
        s32->problem = p;
        s32->block_q = 32;
        s32->block_groups = 4;
        s32->double_buffer = true;
        *kind = KIND_32C;
        *arch = "gfx950";
        return 0;
    case 8:
        /* groups must be divisible by block_ch = block_waves * wave_size (2 * 64 = 128) */
        p.N = 32;
        p.H = 200;
        p.W = 200;
        p.groups = 128;
        p.cpg = 1;
        p.kpg = 1;
        *sdw = rocke_direct_depthwise_spec_default();
        sdw->problem = p;
        sdw->block_w = 16;
        sdw->block_waves = 2;
        *kind = KIND_DW;
        *arch = "gfx950";
        return 0;
    case 9:
        /* depthwise stride=2: exercises Ho/Wo descriptors and stride-aware flush */
        p.N = 2;
        p.H = 14;
        p.W = 14;
        p.groups = 64;
        p.cpg = 1;
        p.kpg = 1;
        p.stride = 2;
        *sdw = rocke_direct_depthwise_spec_default();
        sdw->problem = p;
        sdw->block_w = 8;
        sdw->block_waves = 1;
        *kind = KIND_DW;
        *arch = "gfx950";
        return 0;
    case 10:
        /* spatial layout: groups=3 (non-power-of-two, exercises partial wave) */
        p.N = 2;
        p.H = 14;
        p.W = 14;
        p.groups = 3;
        p.cpg = 1;
        p.kpg = 1;
        *ssp = rocke_direct_depthwise_spatial_spec_default();
        ssp->problem = p;
        ssp->block_waves = 2;
        *kind = KIND_SPATIAL;
        *arch = "gfx950";
        return 0;
    case 11:
        /* spatial layout with stride=2: exercises Ho/Wo + spatial thread mapping */
        p.N = 2;
        p.H = 14;
        p.W = 14;
        p.groups = 3;
        p.cpg = 1;
        p.kpg = 1;
        p.stride = 2;
        *ssp = rocke_direct_depthwise_spatial_spec_default();
        ssp->problem = p;
        ssp->block_waves = 1;
        *kind = KIND_SPATIAL;
        *arch = "gfx950";
        return 0;
    case 12:
        /* dgrad: baseline grouped dgrad stride=1 */
        p.N = 2;
        p.H = 8;
        p.W = 8;
        p.groups = 8;
        p.cpg = 16;
        p.kpg = 16;
        *sdgrad = rocke_direct_conv_dgrad_spec_default();
        sdgrad->problem = p;
        sdgrad->block_q = 16;
        sdgrad->block_groups = 8;
        *kind = KIND_DGRAD;
        *arch = "gfx950";
        return 0;
    case 13:
        /* dgrad: larger groups / different block_groups */
        p.N = 2;
        p.H = 8;
        p.W = 8;
        p.groups = 8;
        p.cpg = 32;
        p.kpg = 32;
        *sdgrad = rocke_direct_conv_dgrad_spec_default();
        sdgrad->problem = p;
        sdgrad->block_q = 16;
        sdgrad->block_groups = 4;
        *kind = KIND_DGRAD;
        *arch = "gfx950";
        return 0;
    case 14:
        /* dgrad: gfx942 target */
        p.N = 1;
        p.H = 8;
        p.W = 8;
        p.groups = 8;
        p.cpg = 16;
        p.kpg = 16;
        *sdgrad = rocke_direct_conv_dgrad_spec_default();
        sdgrad->problem = p;
        sdgrad->block_q = 16;
        sdgrad->block_groups = 8;
        *kind = KIND_DGRAD;
        *arch = "gfx942";
        return 0;
    case 15:
        /* depthwise_dgrad: stride=1 */
        p.N = 2;
        p.H = 14;
        p.W = 14;
        p.groups = 64;
        p.cpg = 1;
        p.kpg = 1;
        *sdw_dgrad = rocke_direct_depthwise_dgrad_spec_default();
        sdw_dgrad->problem = p;
        sdw_dgrad->block_w = 8;
        sdw_dgrad->block_waves = 1;
        *kind = KIND_DW_DGRAD;
        *arch = "gfx950";
        return 0;
    case 16:
        /* depthwise_dgrad: stride=2 exercises divisibility checks */
        p.N = 2;
        p.H = 14;
        p.W = 14;
        p.groups = 64;
        p.cpg = 1;
        p.kpg = 1;
        p.stride = 2;
        *sdw_dgrad = rocke_direct_depthwise_dgrad_spec_default();
        sdw_dgrad->problem = p;
        sdw_dgrad->block_w = 8;
        sdw_dgrad->block_waves = 1;
        *kind = KIND_DW_DGRAD;
        *arch = "gfx950";
        return 0;
    case 17:
        /* 16c bf16: exercises bf16 I/O, bf16 load/store taps, mfma_f32_16x16x16_bf16 */
        p.N = 32;
        p.H = 200;
        p.W = 200;
        p.groups = 16;
        p.cpg = 16;
        p.kpg = 16;
        p.dtype = "bf16";
        *s16 = rocke_direct_conv_16c_spec_default();
        s16->problem = p;
        s16->block_groups = 8;
        s16->fold_k32 = false;
        *kind = KIND_16C;
        *arch = "gfx950";
        return 0;
    case 18:
        /* 8c bf16: exercises bf16 I/O, bf16 load/store taps, mfma_f32_16x16x16_bf16 */
        p.N = 32;
        p.H = 200;
        p.W = 200;
        p.groups = 16;
        p.cpg = 8;
        p.kpg = 8;
        p.dtype = "bf16";
        *s8 = rocke_direct_conv_8c_spec_default();
        s8->problem = p;
        s8->block_q = 16;
        s8->block_groups = 8;
        s8->double_buffer = true;
        *kind = KIND_8C;
        *arch = "gfx950";
        return 0;
    case 19:
        /* dgrad bf16: exercises bf16 I/O on the scalar-FMA grouped dgrad path */
        p.N = 2;
        p.H = 8;
        p.W = 8;
        p.groups = 8;
        p.cpg = 16;
        p.kpg = 16;
        p.dtype = "bf16";
        *sdgrad = rocke_direct_conv_dgrad_spec_default();
        sdgrad->problem = p;
        sdgrad->block_q = 16;
        sdgrad->block_groups = 8;
        *kind = KIND_DGRAD;
        *arch = "gfx950";
        return 0;
    case 20:
        /* 16c bf16 with fold_k32=True: pins the non-default fold_k32 path under bf16 */
        p.N = 32;
        p.H = 200;
        p.W = 200;
        p.groups = 16;
        p.cpg = 16;
        p.kpg = 16;
        p.dtype = "bf16";
        *s16 = rocke_direct_conv_16c_spec_default();
        s16->problem = p;
        s16->block_groups = 8;
        s16->fold_k32 = true;
        *kind = KIND_16C;
        *arch = "gfx950";
        return 0;
    case 21:
        /* 32c bf16: exercises bf16 I/O on the 32c MFMA path */
        p.N = 32;
        p.H = 200;
        p.W = 200;
        p.groups = 32;
        p.cpg = 32;
        p.kpg = 32;
        p.dtype = "bf16";
        *s32 = rocke_direct_conv_32c_spec_default();
        s32->problem = p;
        s32->block_groups = 8;
        *kind = KIND_32C;
        *arch = "gfx950";
        return 0;
    case 22:
        /* depthwise forward bf16: exercises bf16 I/O on the scalar-FMA depthwise path */
        p.N = 2;
        p.H = 14;
        p.W = 14;
        p.groups = 64;
        p.cpg = 1;
        p.kpg = 1;
        p.dtype = "bf16";
        *sdw = rocke_direct_depthwise_spec_default();
        sdw->problem = p;
        sdw->block_w = 8;
        sdw->block_waves = 1;
        *kind = KIND_DW;
        *arch = "gfx950";
        return 0;
    case 23:
        /* depthwise spatial bf16: exercises bf16 I/O on the small-group spatial path */
        p.N = 2;
        p.H = 14;
        p.W = 14;
        p.groups = 16;
        p.cpg = 1;
        p.kpg = 1;
        p.dtype = "bf16";
        *ssp = rocke_direct_depthwise_spatial_spec_default();
        ssp->problem = p;
        ssp->block_waves = 1;
        *kind = KIND_SPATIAL;
        *arch = "gfx950";
        return 0;
    case 24:
        /* depthwise dgrad bf16: exercises bf16 I/O on the scalar-FMA depthwise dgrad path */
        p.N = 2;
        p.H = 14;
        p.W = 14;
        p.groups = 64;
        p.cpg = 1;
        p.kpg = 1;
        p.dtype = "bf16";
        *sdw_dgrad = rocke_direct_depthwise_dgrad_spec_default();
        sdw_dgrad->problem = p;
        sdw_dgrad->block_w = 8;
        sdw_dgrad->block_waves = 1;
        *kind = KIND_DW_DGRAD;
        *arch = "gfx950";
        return 0;
    /* ---- wgrad (backward weights) ---- */
    case 25:
        /* Defaults: mfma_k=32 (VEC_CH=8, two ds_read_tr per fragment),
         * waves_k=waves_c=waves_q=1, ho_per_block=4. */
        p.N = 2;
        p.H = 8;
        p.W = 8;
        p.groups = 8;
        p.cpg = 16;
        p.kpg = 16;
        *swg = rocke_direct_conv_wgrad_spec_default();
        swg->problem = p;
        *kind = KIND_WGRAD;
        *arch = "gfx950";
        return 0;
    case 26:
        /* Narrow MFMA: mfma_k=16 (VEC_CH=4, one ds_read_tr per fragment). */
        p.N = 2;
        p.H = 8;
        p.W = 8;
        p.groups = 8;
        p.cpg = 16;
        p.kpg = 16;
        *swg = rocke_direct_conv_wgrad_spec_default();
        swg->problem = p;
        swg->mfma_k = 16;
        swg->ho_per_block = 2;
        *kind = KIND_WGRAD;
        *arch = "gfx950";
        return 0;
    case 27:
        /* Multi-wave: K/C/Q all split, so n_k_tiles = n_c_tiles = 2,
         * STRIP_GROUPS = 2 and the kernel name carries the _wq flag. */
        p.N = 4;
        p.H = 16;
        p.W = 16;
        p.groups = 4;
        p.cpg = 64;
        p.kpg = 64;
        *swg = rocke_direct_conv_wgrad_spec_default();
        swg->problem = p;
        swg->waves_k = 2;
        swg->waves_c = 2;
        swg->waves_q = 2;
        swg->ho_per_block = 3;
        *kind = KIND_WGRAD;
        *arch = "gfx950";
        return 0;
    case 28:
        /* gfx942 has no 16x16x32 f16 atom -> both engines reject (mfma_k=32). */
        p.N = 2;
        p.H = 8;
        p.W = 8;
        p.groups = 8;
        p.cpg = 16;
        p.kpg = 16;
        *swg = rocke_direct_conv_wgrad_spec_default();
        swg->problem = p;
        *kind = KIND_WGRAD;
        *arch = "gfx942";
        return 0;
    case 29:
        /* gfx942 with mfma_k=16 clears the atom gate and is rejected one check
         * later, on the missing ds_read_tr16_b64 the LDS staging needs. */
        p.N = 2;
        p.H = 8;
        p.W = 8;
        p.groups = 8;
        p.cpg = 16;
        p.kpg = 16;
        *swg = rocke_direct_conv_wgrad_spec_default();
        swg->problem = p;
        swg->mfma_k = 16;
        *kind = KIND_WGRAD;
        *arch = "gfx942";
        return 0;
    case 30:
        /* wgrad bf16: bf16 I/O and the bf16 MFMA atom, same LDS transpose
         * staging. Pins that only the atom and the element type move. */
        p.N = 2;
        p.H = 8;
        p.W = 8;
        p.groups = 8;
        p.cpg = 16;
        p.kpg = 16;
        p.dtype = "bf16";
        *swg = rocke_direct_conv_wgrad_spec_default();
        swg->problem = p;
        *kind = KIND_WGRAD;
        *arch = "gfx950";
        return 0;
    case 31:
        /* wgrad bf16 at mfma_k=16: the narrow atom under bf16, one ds_read_tr
         * per fragment. */
        p.N = 2;
        p.H = 8;
        p.W = 8;
        p.groups = 8;
        p.cpg = 16;
        p.kpg = 16;
        p.dtype = "bf16";
        *swg = rocke_direct_conv_wgrad_spec_default();
        swg->problem = p;
        swg->mfma_k = 16;
        swg->ho_per_block = 2;
        *kind = KIND_WGRAD;
        *arch = "gfx950";
        return 0;
    case 32:
        /* column-streamed depthwise, stride=1 fp16, default 16-row tile. Exact
         * channel and W tiles, but the guards bound against kernargs (AOT), so
         * they are emitted all the same. */
        p.N = 2;
        p.H = 8;
        p.W = 8;
        p.groups = 128;
        p.cpg = 1;
        p.kpg = 1;
        *sdwc = rocke_direct_depthwise_col_spec_default();
        sdwc->problem = p;
        sdwc->block_w = 4;
        sdwc->block_waves = 2;
        sdwc->dtype = "fp16";
        *kind = KIND_DWCOL;
        *arch = "gfx950";
        return 0;
    case 33:
        /* col stride=2 bf16 with partial channel and W tiles (groups=70 % 64,
         * Wo=5 % 4) and two-row tiles, so Ho=5 needs a partial last row tile. */
        p.N = 1;
        p.H = 9;
        p.W = 9;
        p.groups = 70;
        p.cpg = 1;
        p.kpg = 1;
        p.stride = 2;
        *sdwc = rocke_direct_depthwise_col_spec_default();
        sdwc->problem = p;
        sdwc->block_h = 2;
        sdwc->block_w = 4;
        sdwc->block_waves = 1;
        sdwc->dtype = "bf16";
        *kind = KIND_DWCOL;
        *arch = "gfx950";
        return 0;
    case 34:
        /* col stride=3: exercises the (y - r) % stride tap pruning at a stride
         * no other case reaches. */
        p.N = 1;
        p.H = 16;
        p.W = 16;
        p.groups = 64;
        p.cpg = 1;
        p.kpg = 1;
        p.stride = 3;
        *sdwc = rocke_direct_depthwise_col_spec_default();
        sdwc->problem = p;
        sdwc->block_w = 4;
        sdwc->block_waves = 1;
        sdwc->dtype = "bf16";
        *kind = KIND_DWCOL;
        *arch = "gfx950";
        return 0;
    case 35:
        /* col with a large filter (31x31): the regime the variant exists for --
         * KW rides the runtime loop so only KH weights are live. A 4-row tile
         * still unrolls (4-1)+31 input rows. */
        p.N = 1;
        p.H = 8;
        p.W = 8;
        p.groups = 64;
        p.cpg = 1;
        p.kpg = 1;
        p.KH = 31;
        p.KW = 31;
        p.PAD = 15;
        *sdwc = rocke_direct_depthwise_col_spec_default();
        sdwc->problem = p;
        sdwc->block_h = 4;
        sdwc->block_w = 4;
        sdwc->block_waves = 1;
        sdwc->dtype = "fp16";
        *kind = KIND_DWCOL;
        *arch = "gfx950";
        return 0;
    case 36:
        /* col 1x1 / PAD=0 degenerate with a non-power-of-two group count */
        p.N = 2;
        p.H = 6;
        p.W = 6;
        p.groups = 3;
        p.cpg = 1;
        p.kpg = 1;
        p.KH = 1;
        p.KW = 1;
        p.PAD = 0;
        *sdwc = rocke_direct_depthwise_col_spec_default();
        sdwc->problem = p;
        sdwc->block_w = 2;
        sdwc->block_waves = 1;
        sdwc->dtype = "fp16";
        *kind = KIND_DWCOL;
        *arch = "gfx950";
        return 0;
    case 37:
        /* col with KH != KW (5x3): separates the unrolled axis from the runtime one */
        p.N = 1;
        p.H = 8;
        p.W = 8;
        p.groups = 128;
        p.cpg = 1;
        p.kpg = 1;
        p.KH = 5;
        p.KW = 3;
        p.PAD = 2;
        *sdwc = rocke_direct_depthwise_col_spec_default();
        sdwc->problem = p;
        sdwc->block_w = 4;
        sdwc->block_waves = 2;
        sdwc->dtype = "bf16";
        *kind = KIND_DWCOL;
        *arch = "gfx950";
        return 0;
    case 38:
        /* col stride=2 with valid padding (PAD=0) and an odd 3-row tile */
        p.N = 1;
        p.H = 13;
        p.W = 13;
        p.groups = 64;
        p.cpg = 1;
        p.kpg = 1;
        p.PAD = 0;
        p.stride = 2;
        *sdwc = rocke_direct_depthwise_col_spec_default();
        sdwc->problem = p;
        sdwc->block_h = 3;
        sdwc->block_w = 6;
        sdwc->block_waves = 1;
        sdwc->dtype = "fp16";
        *kind = KIND_DWCOL;
        *arch = "gfx950";
        return 0;
    case 39:
        /* col with block_w=1 and a channel tail (groups=100 % 128) */
        p.N = 1;
        p.H = 10;
        p.W = 10;
        p.groups = 100;
        p.cpg = 1;
        p.kpg = 1;
        *sdwc = rocke_direct_depthwise_col_spec_default();
        sdwc->problem = p;
        sdwc->block_w = 1;
        sdwc->block_waves = 2;
        sdwc->dtype = "bf16";
        *kind = KIND_DWCOL;
        *arch = "gfx950";
        return 0;
    case 40:
        /* dwcol PAD-overhang: PAD=2 > (KH-1)/2=1 with stride=2 and two-row
         * tiles. Cross-verifies the n_iters = (block_h-1)*stride + KH formula. */
        p.N = 1;
        p.H = 10;
        p.W = 10;
        p.groups = 64;
        p.cpg = 1;
        p.kpg = 1;
        p.KH = 3;
        p.KW = 3;
        p.PAD = 2;
        p.stride = 2;
        *sdwc = rocke_direct_depthwise_col_spec_default();
        sdwc->problem = p;
        sdwc->block_h = 2;
        sdwc->block_w = 4;
        sdwc->block_waves = 1;
        sdwc->dtype = "fp16";
        *kind = KIND_DWCOL;
        *arch = "gfx950";
        return 0;
    case 41:
        /* depthwise 7x7 stride 2 bf16: PAD > 1 shifts the row stream's static
         * slots, and stride 2 gates its flushes on the row parity. */
        p.N = 2;
        p.H = 14;
        p.W = 14;
        p.groups = 128;
        p.cpg = 1;
        p.kpg = 1;
        p.KH = 7;
        p.KW = 7;
        p.PAD = 3;
        p.stride = 2;
        p.dtype = "bf16";
        *sdw = rocke_direct_depthwise_spec_default();
        sdw->problem = p;
        sdw->block_w = 8;
        sdw->block_waves = 2;
        *kind = KIND_DW;
        *arch = "gfx950";
        return 0;
    /* ---- wgrad, runtime channels (indices 71..74) ----
     * cpg/kpg come from p_total_c / p_total_k / p_groups: runtime k/c tile
     * counts in the grid decode, a dynamic dW descriptor and runtime masks. */
    case 71:
        /* Defaults otherwise: the baked config 25 with runtime channels. */
        p.N = 2;
        p.H = 8;
        p.W = 8;
        p.groups = 8;
        p.cpg = 16;
        p.kpg = 16;
        *swg = rocke_direct_conv_wgrad_spec_default();
        swg->problem = p;
        swg->runtime_channels = 1;
        *kind = KIND_WGRAD;
        *arch = "gfx950";
        return 0;
    case 72:
        /* bf16, K/C/Q split with a non-power-of-two block_c (waves_c=3). */
        p.N = 2;
        p.H = 8;
        p.W = 8;
        p.groups = 1;
        p.cpg = 48;
        p.kpg = 192;
        p.dtype = "bf16";
        *swg = rocke_direct_conv_wgrad_spec_default();
        swg->problem = p;
        swg->waves_k = 2;
        swg->waves_c = 3;
        swg->waves_q = 2;
        swg->ho_per_block = 3;
        swg->runtime_channels = 1;
        *kind = KIND_WGRAD;
        *arch = "gfx950";
        return 0;
    case 73:
        /* 1x1 filter, PAD 0, the narrow atom and four k-waves. */
        p.N = 2;
        p.H = 8;
        p.W = 8;
        p.groups = 4;
        p.cpg = 64;
        p.kpg = 64;
        p.KH = 1;
        p.KW = 1;
        p.PAD = 0;
        *swg = rocke_direct_conv_wgrad_spec_default();
        swg->problem = p;
        swg->waves_k = 4;
        swg->mfma_k = 16;
        swg->ho_per_block = 2;
        swg->runtime_channels = 1;
        *kind = KIND_WGRAD;
        *arch = "gfx950";
        return 0;
    case 74:
        /* 5x5 filter with valid padding, bf16, a ragged channel count. */
        p.N = 1;
        p.H = 12;
        p.W = 12;
        p.groups = 1;
        p.cpg = 24;
        p.kpg = 24;
        p.KH = 5;
        p.KW = 5;
        p.PAD = 0;
        p.dtype = "bf16";
        *swg = rocke_direct_conv_wgrad_spec_default();
        swg->problem = p;
        swg->waves_c = 2;
        swg->ho_per_block = 5;
        swg->runtime_channels = 1;
        *kind = KIND_WGRAD;
        *arch = "gfx950";
        return 0;
    default:
        if(idx >= REJECT_CFG_BASE)
            return make_reject_cfg(
                idx - REJECT_CFG_BASE, kind, s16, sdwc, snongrouped, stiled, arch);
        if(idx >= NONGROUPED_DIL_CFG_BASE)
        {
            if(make_nongrouped_dil_cfg(idx - NONGROUPED_DIL_CFG_BASE, snongrouped, arch) == 0)
            {
                *kind = KIND_NONGROUPED;
                return 0;
            }
            return -1;
        }
        if(idx >= TILED_CFG_BASE)
        {
            if(make_tiled_cfg(idx - TILED_CFG_BASE, stiled, arch) == 0)
            {
                *kind = KIND_DW_TILED;
                return 0;
            }
            return -1;
        }
        if(idx >= NONGROUPED_CFG_BASE
           && make_nongrouped_cfg(idx - NONGROUPED_CFG_BASE, snongrouped, arch) == 0)
        {
            *kind = KIND_NONGROUPED;
            return 0;
        }
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

    int kind = KIND_16C;
    rocke_direct_conv_16c_spec_t s16;
    rocke_direct_conv_4c_spec_t s4;
    rocke_direct_conv_8c_spec_t s8;
    rocke_direct_conv_32c_spec_t s32;
    rocke_direct_depthwise_spec_t sdw;
    rocke_direct_depthwise_spatial_spec_t ssp;
    rocke_direct_depthwise_col_spec_t sdwc;
    rocke_direct_conv_dgrad_spec_t sdgrad;
    rocke_direct_depthwise_dgrad_spec_t sdw_dgrad;
    rocke_direct_conv_wgrad_spec_t swg;
    rocke_direct_conv_nongrouped_spec_t snongrouped;
    rocke_direct_depthwise_tiled_spec_t stiled;
    rocke_direct_conv_spec_t sgen;
    rocke_direct_transpose_weights_dgrad_spec_t stp;
    rocke_direct_reorganize_weights_spec_t sreorg;
    rocke_direct_conv_split_reduce_spec_t sred;
    const char* arch = "gfx950";
    if(idx >= NONGROUPED_GROUPED_CFG_BASE)
    {
        kind = KIND_NONGROUPED;
        if(make_nongrouped_grouped_cfg(idx - NONGROUPED_GROUPED_CFG_BASE, &snongrouped, &arch) != 0)
        {
            fprintf(stderr, "unknown config index %d\n", idx);
            return 2;
        }
    }
    else if(idx >= NONGROUPED_SPLIT_CFG_BASE)
    {
        if(make_nongrouped_split_cfg(
               idx - NONGROUPED_SPLIT_CFG_BASE, &kind, &snongrouped, &sred, &arch)
           != 0)
        {
            fprintf(stderr, "unknown config index %d\n", idx);
            return 2;
        }
    }
    else if(idx >= GENERIC_CFG_BASE
            && make_generic_cfg(idx - GENERIC_CFG_BASE, &kind, &sgen, &stp, &sreorg) == 0)
    {
        /* selected above */
    }
    else if(make_cfg(idx,
                     &kind,
                     &s16,
                     &s4,
                     &s8,
                     &s32,
                     &sdw,
                     &ssp,
                     &sdwc,
                     &sdgrad,
                     &sdw_dgrad,
                     &swg,
                     &snongrouped,
                     &stiled,
                     &arch)
            != 0)
    {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 2;
    }

    rocke_ir_builder_t b;
    rocke_kernel_def_t* kernel = NULL;
    if(kind == KIND_16C)
        kernel = rocke_build_direct_conv_16c_new(&b, &s16, arch);
    else if(kind == KIND_4C)
        kernel = rocke_build_direct_conv_4c_new(&b, &s4, arch);
    else if(kind == KIND_8C)
        kernel = rocke_build_direct_conv_8c_new(&b, &s8, arch);
    else if(kind == KIND_32C)
        kernel = rocke_build_direct_conv_32c_new(&b, &s32, arch);
    else if(kind == KIND_SPATIAL)
        kernel = rocke_build_direct_depthwise_spatial_new(&b, &ssp, arch);
    else if(kind == KIND_DWCOL)
        kernel = rocke_build_direct_depthwise_col_new(&b, &sdwc, arch);
    else if(kind == KIND_DGRAD)
        kernel = rocke_build_direct_conv_dgrad_new(&b, &sdgrad, arch);
    else if(kind == KIND_DW_DGRAD)
        kernel = rocke_build_direct_depthwise_dgrad_new(&b, &sdw_dgrad, arch);
    else if(kind == KIND_WGRAD)
        kernel = rocke_build_direct_conv_wgrad_new(&b, &swg, arch);
    else if(kind == KIND_NONGROUPED)
        kernel = rocke_build_direct_conv_nongrouped_new(&b, &snongrouped, arch);
    else if(kind == KIND_GENERIC)
        kernel = rocke_build_direct_conv_new(&b, &sgen, arch);
    else if(kind == KIND_TRANSPOSE)
        kernel = rocke_build_direct_transpose_weights_dgrad_new(&b, &stp, arch);
    else if(kind == KIND_REORG)
        kernel = rocke_build_direct_reorganize_weights_new(&b, &sreorg, arch);
    else if(kind == KIND_DW_TILED)
        kernel = rocke_build_direct_depthwise_tiled_new(&b, &stiled, arch);
    else if(kind == KIND_SPLIT_REDUCE)
        kernel = rocke_build_direct_conv_split_reduce_new(&b, &sred, arch);
    else
        kernel = rocke_build_direct_depthwise_new(&b, &sdw, arch);
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
