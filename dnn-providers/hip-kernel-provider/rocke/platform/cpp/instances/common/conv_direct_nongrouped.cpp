// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * conv_direct_nongrouped.cpp -- C++ engine port of build_direct_conv_nongrouped
 * (library/kernels/common/conv_direct_nongrouped.py): the non-grouped (groups == 1)
 * NHWC direct convolution with an LDS-staged, halo-inclusive activation tile,
 * plus its grouped (group-merge) mode.
 *
 * BYTE-IDENTITY CONTRACT. Every IR op (including each arith.constant) is
 * emitted in exactly the order the Python builder emits it. Python evaluates
 * call arguments left to right; C++ leaves argument evaluation order
 * unspecified, so a nested call such as
 *     b.add(b.mul(x, b.const_i32(S)), b.const_i32(PAD))
 * is written here as one statement per op, in Python order. Never fold two
 * emitting calls into one argument list in this file.
 *
 * Layout of this TU:
 *   - spec defaults / derived geometry / kernel_name / validate / is_valid
 *     (host-side, no IR)
 *   - the builder, split into the same phases as the Python function body
 *     (prologue, grid decode, staging metadata, read bases, channel loop,
 *     epilogue) sharing one file-local context struct
 *   - the _new / lower_to_llvm convenience entries
 *   - the split-C reduction (DirectSplitReduceSpec): spec, builder, entries
 */
#include "rocke/instance_conv_direct_nongrouped.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <string>
#include <vector>

#include "rocke/error_boundary.hpp" /* ckc::guard_builder boundary shim */
#include "rocke/helper_rocke.core.arch.h" /* rocke_archtarget_from_gfx, has_shape */
#include "rocke/helper_rocke.helpers.grid.h" /* rocke_chiplet_aware_super_tile_dynamic */
#include "rocke/helper_rocke.helpers.io.h" /* rocke_b_io_ir_type */
#include "rocke/helper_rocke.helpers.spec.h" /* rocke_kernel_name_join */
#include "rocke/instance_conv_direct_grouped_internal.h" /* rocke_dconv_emit_params */
#include "rocke/lower_llvm.h"

/* Global load width for the activation staging path, in halves (dwordx4). */
#define ROCKE_DCONV_NONGROUPED_X_LOAD_VEC 8

/* Canonical i32 byte offset for every masked-out buffer access (staging loads
 * and epilogue stores); see the Python `_OOB_BASE` comment. Loop-invariant
 * predication is folded into the base offset, and a buffer access past
 * num_records is dropped (a load returns zero). */
#define ROCKE_DCONV_NONGROUPED_OOB_BASE 0x7F000000

/* Output elements per thread of the split-C reduction (Python _REDUCE_VEC): one
 * dwordx4 store of 16-bit outputs, read as two dwordx4 fp32 loads per slab. */
#define ROCKE_DCONV_SPLIT_REDUCE_VEC 8

/* ===================================================================== *
 *  Atom table  (Python _ATOMS: name -> (tile, K, frag))
 * ===================================================================== */
typedef rocke_value_t* (*rocke_dconv_nongrouped_mfma_fn)(rocke_ir_builder_t*,
                                                         rocke_value_t*,
                                                         rocke_value_t*,
                                                         rocke_value_t*);

typedef struct rocke_dconv_nongrouped_atom
{
    const char* name;
    int tile; /* MFMA M == N                         */
    int k; /* MFMA K                              */
    int frag; /* halves per lane in an A/B fragment  */
    rocke_dconv_nongrouped_mfma_fn f16;
    rocke_dconv_nongrouped_mfma_fn bf16;
} rocke_dconv_nongrouped_atom_t;

static const rocke_dconv_nongrouped_atom_t k_nongrouped_atoms[] = {
    {"32x32x16", 32, 16, 8, rocke_b_mfma_f32_32x32x16_f16, rocke_b_mfma_f32_32x32x16_bf16},
    {"32x32x8", 32, 8, 4, rocke_b_mfma_f32_32x32x8_f16, rocke_b_mfma_f32_32x32x8_bf16},
    {"16x16x32", 16, 32, 8, rocke_b_mfma_f32_16x16x32_f16, rocke_b_mfma_f32_16x16x32_bf16},
    {"16x16x16", 16, 16, 4, rocke_b_mfma_f32_16x16x16_f16, rocke_b_mfma_f32_16x16x16_bf16},
};

static const rocke_dconv_nongrouped_atom_t* nongrouped_atom(const char* name)
{
    size_t i;
    if(name == NULL)
    {
        return NULL;
    }
    for(i = 0; i < sizeof(k_nongrouped_atoms) / sizeof(k_nongrouped_atoms[0]); ++i)
    {
        if(strcmp(k_nongrouped_atoms[i].name, name) == 0)
        {
            return &k_nongrouped_atoms[i];
        }
    }
    return NULL;
}

static const char* nongrouped_dtype(const rocke_direct_conv_nongrouped_spec_t* spec)
{
    return spec->problem.dtype ? spec->problem.dtype : "fp16";
}

static bool nongrouped_is_bf16(const rocke_direct_conv_nongrouped_spec_t* spec)
{
    return strcmp(nongrouped_dtype(spec), "bf16") == 0;
}

static int nongrouped_ho(const rocke_direct_conv_problem_t* p)
{
    return (p->H + 2 * p->PAD - (p->KH - 1) * p->dil_h - 1) / p->stride + 1;
}

static int nongrouped_wo(const rocke_direct_conv_problem_t* p)
{
    return (p->W + 2 * p->PAD - (p->KW - 1) * p->dil_w - 1) / p->stride + 1;
}

static void nongrouped_set_reason(char* reason, size_t reason_cap, const char* msg)
{
    if(reason && reason_cap > 0)
    {
        snprintf(reason, reason_cap, "%s", msg);
    }
}

/* ===================================================================== *
 *  Spec defaults + derived geometry
 * ===================================================================== */

rocke_direct_conv_nongrouped_spec_t rocke_direct_conv_nongrouped_spec_default(void)
{
    rocke_direct_conv_nongrouped_spec_t spec;
    memset(&spec, 0, sizeof(spec));
    spec.problem = rocke_direct_conv_problem_default();
    spec.problem.groups = 1;
    spec.name = "direct_conv_nongrouped";
    spec.tile_h = 16;
    spec.tile_w = 32;
    spec.tile_k = 128;
    spec.ck = 16;
    spec.waves_m = 2;
    spec.waves_n = 4;
    spec.atom = "32x32x16";
    spec.wave_size = 64;
    spec.lds_pad = 8;
    spec.chiplet_swizzle = true;
    spec.swizzle_wgm = 8;
    spec.chiplet_chunk = 64;
    spec.num_xcds = 8;
    spec.double_buffer = false;
    spec.iglp = ROCKE_DCONV_NONGROUPED_IGLP_NONE;
    spec.waves_per_eu = ROCKE_DCONV_NONGROUPED_WAVES_PER_EU_NONE;
    spec.split_c = 1;
    spec.grouped = false;
    spec.group_merge = 1;
    return spec;
}

int rocke_direct_conv_nongrouped_threads_per_block(const rocke_direct_conv_nongrouped_spec_t* spec)
{
    return spec->waves_m * spec->waves_n * spec->wave_size;
}

long rocke_direct_conv_nongrouped_lds_bytes(const rocke_direct_conv_nongrouped_spec_t* spec)
{
    const rocke_dconv_nongrouped_atom_t* at = nongrouped_atom(spec->atom);
    const rocke_direct_conv_problem_t* p = &spec->problem;
    long lds_in_h, lds_in_w, x_halves, w_halves, one;
    if(at == NULL)
    {
        return 0;
    }
    lds_in_h = (long)(spec->tile_h - 1) * p->stride + (long)(p->KH - 1) * p->dil_h + 1;
    lds_in_w = (long)(spec->tile_w - 1) * p->stride + (long)(p->KW - 1) * p->dil_w + 1;
    x_halves = lds_in_h * lds_in_w * (spec->ck + spec->lds_pad);
    w_halves = (long)spec->tile_k * p->KH * p->KW * spec->ck;
    /* + one scratch fragment per array (see the builder). */
    one = x_halves + ROCKE_DCONV_NONGROUPED_X_LOAD_VEC + w_halves + at->frag;
    return 2 * one * (spec->double_buffer ? 2 : 1);
}

int rocke_direct_conv_nongrouped_acc_vgprs(const rocke_direct_conv_nongrouped_spec_t* spec)
{
    const rocke_dconv_nongrouped_atom_t* at = nongrouped_atom(spec->atom);
    int acc_per_lane, m_tiles_per_wave, n_tiles_per_wave;
    if(at == NULL || spec->waves_m <= 0 || spec->waves_n <= 0 || spec->wave_size <= 0)
    {
        return 0;
    }
    acc_per_lane = at->tile * at->tile / spec->wave_size;
    m_tiles_per_wave = (spec->tile_k / at->tile) / spec->waves_m;
    n_tiles_per_wave = (spec->tile_h / spec->waves_n) * (spec->tile_w / at->tile);
    return m_tiles_per_wave * n_tiles_per_wave * acc_per_lane;
}

/* merged_c / merged_k: the input channels one workgroup reduces over and the
 * output channels its channel tiles span (cpg / kpg x group_merge when grouped). */
static int nongrouped_merged_c(const rocke_direct_conv_nongrouped_spec_t* spec)
{
    return spec->problem.cpg * (spec->grouped ? spec->group_merge : 1);
}

static int nongrouped_merged_k(const rocke_direct_conv_nongrouped_spec_t* spec)
{
    return spec->problem.kpg * (spec->grouped ? spec->group_merge : 1);
}

/* w_vec: channels per weight staging load. A merged block's fragment can
 * straddle two groups when cpg is 4, so it is loaded (and masked) in halves of
 * 4 channels. Returns 0 for an unknown atom. */
static int nongrouped_w_vec(const rocke_direct_conv_nongrouped_spec_t* spec)
{
    const rocke_dconv_nongrouped_atom_t* at = nongrouped_atom(spec->atom);
    if(spec->grouped && spec->group_merge > 1)
    {
        return 4;
    }
    return at != NULL ? at->frag : 0;
}

/* Merged groups per image (the n_mg grid factor); 1 for the dense kernel. */
static int nongrouped_n_mg(const rocke_direct_conv_nongrouped_spec_t* spec)
{
    if(!spec->grouped || spec->group_merge < 1)
    {
        return 1;
    }
    return spec->problem.groups / spec->group_merge;
}

static void nongrouped_tile_counts(const rocke_direct_conv_nongrouped_spec_t* spec,
                                   int* n_wt,
                                   int* n_ht,
                                   int* n_kt)
{
    const rocke_direct_conv_problem_t* p = &spec->problem;
    *n_wt = (nongrouped_wo(p) + spec->tile_w - 1) / spec->tile_w;
    *n_ht = (nongrouped_ho(p) + spec->tile_h - 1) / spec->tile_h;
    *n_kt = (nongrouped_merged_k(spec) + spec->tile_k - 1) / spec->tile_k;
}

void rocke_direct_conv_nongrouped_grid(const rocke_direct_conv_nongrouped_spec_t* spec, int grid[3])
{
    int n_wt, n_ht, n_kt;
    nongrouped_tile_counts(spec, &n_wt, &n_ht, &n_kt);
    grid[0] = n_wt * n_ht * spec->problem.N * nongrouped_n_mg(spec) * n_kt * spec->split_c;
    grid[1] = 1;
    grid[2] = 1;
}

long long
    rocke_direct_conv_nongrouped_workspace_bytes(const rocke_direct_conv_nongrouped_spec_t* spec)
{
    const rocke_direct_conv_problem_t* p = &spec->problem;
    if(spec->split_c == 1)
    {
        return 0;
    }
    return 4LL * spec->split_c * p->N * nongrouped_ho(p) * nongrouped_wo(p)
           * ((long long)p->groups * p->kpg);
}

/* ===================================================================== *
 *  kernel_name()
 * ===================================================================== */
rocke_status_t rocke_direct_conv_nongrouped_kernel_name(
    const rocke_direct_conv_nongrouped_spec_t* spec, char* out, size_t out_cap)
{
    char prob_short[128];
    char t_buf[64];
    char ck_buf[24];
    char w_buf[32];
    char a_buf[48];
    char g_buf[24];
    char iglp_buf[24];
    char we_buf[24];
    char sc_buf[24];
    char grp_buf[24];
    char d_buf[32];
    const char* parts[13];

    if(spec == NULL || out == NULL || out_cap == 0)
    {
        return ROCKE_ERR_VALUE;
    }
    if(rocke_direct_conv_problem_short(&spec->problem, prob_short, sizeof(prob_short)) != ROCKE_OK)
    {
        return ROCKE_ERR_VALUE;
    }
    snprintf(t_buf, sizeof(t_buf), "t%dx%dx%d", spec->tile_h, spec->tile_w, spec->tile_k);
    snprintf(ck_buf, sizeof(ck_buf), "ck%d", spec->ck);
    snprintf(w_buf, sizeof(w_buf), "w%dx%d", spec->waves_m, spec->waves_n);
    snprintf(a_buf, sizeof(a_buf), "a%s", spec->atom ? spec->atom : "");
    if(spec->chiplet_swizzle)
    {
        snprintf(g_buf, sizeof(g_buf), "g%d", spec->swizzle_wgm);
    }
    else
    {
        snprintf(g_buf, sizeof(g_buf), "gnone");
    }
    iglp_buf[0] = '\0';
    if(spec->iglp != ROCKE_DCONV_NONGROUPED_IGLP_NONE)
    {
        snprintf(iglp_buf, sizeof(iglp_buf), "iglp%d", spec->iglp);
    }
    we_buf[0] = '\0';
    if(spec->waves_per_eu != ROCKE_DCONV_NONGROUPED_WAVES_PER_EU_NONE)
    {
        snprintf(we_buf, sizeof(we_buf), "we%d", spec->waves_per_eu);
    }
    sc_buf[0] = '\0';
    if(spec->split_c > 1)
    {
        snprintf(sc_buf, sizeof(sc_buf), "sc%d", spec->split_c);
    }
    grp_buf[0] = '\0';
    if(spec->grouped)
    {
        snprintf(grp_buf, sizeof(grp_buf), "grp%d", spec->group_merge);
    }
    /* f"d{dil_h}x{dil_w}" only for a dilated problem: undilated names are unchanged. */
    d_buf[0] = '\0';
    if(spec->problem.dil_h != 1 || spec->problem.dil_w != 1)
    {
        snprintf(d_buf, sizeof(d_buf), "d%dx%d", spec->problem.dil_h, spec->problem.dil_w);
    }
    parts[0] = prob_short;
    parts[1] = d_buf;
    parts[2] = t_buf;
    parts[3] = ck_buf;
    parts[4] = w_buf;
    parts[5] = a_buf;
    parts[6] = g_buf;
    parts[7] = spec->double_buffer ? "db" : "";
    parts[8] = iglp_buf;
    parts[9] = we_buf;
    parts[10] = sc_buf;
    parts[11] = grp_buf;
    parts[12] = nongrouped_is_bf16(spec) ? "bf16" : "";
    return rocke_kernel_name_join(spec->name, parts, 13, NULL, NULL, 0, out, out_cap, NULL);
}

/* ===================================================================== *
 *  validate()  (Python DirectNongroupedConvSpec.validate, same check order)
 * ===================================================================== */
rocke_status_t rocke_direct_conv_nongrouped_validate(
    const rocke_direct_conv_nongrouped_spec_t* spec, char* reason, size_t reason_cap)
{
    const rocke_direct_conv_problem_t* p;
    const rocke_dconv_nongrouped_atom_t* at;
    char msg[ROCKE_ERR_MSG_CAP];
    int t, merged_c, merged_k;

    if(spec == NULL)
    {
        nongrouped_set_reason(reason, reason_cap, "null spec");
        return ROCKE_ERR_VALUE;
    }
    p = &spec->problem;

#define NONGROUPED_REJECT(...)                          \
    do                                                  \
    {                                                   \
        snprintf(msg, sizeof(msg), __VA_ARGS__);        \
        nongrouped_set_reason(reason, reason_cap, msg); \
        return ROCKE_ERR_VALUE;                         \
    } while(0)

    if(strcmp(nongrouped_dtype(spec), "fp16") != 0 && strcmp(nongrouped_dtype(spec), "bf16") != 0)
    {
        NONGROUPED_REJECT("DirectNongroupedConvSpec: unsupported dtype '%s'",
                          nongrouped_dtype(spec));
    }
    if(!spec->grouped && p->groups != 1)
    {
        NONGROUPED_REJECT("DirectNongroupedConvSpec is the groups==1 family (got groups=%d); "
                          "use conv_direct_grouped for grouped shapes",
                          p->groups);
    }
    if(spec->grouped && p->groups < 1)
    {
        NONGROUPED_REJECT("grouped=True needs groups >= 1 (got groups=%d)", p->groups);
    }
    /* Python's `%` floors; a nonzero remainder is nonzero in either convention. */
    if(spec->grouped && (spec->group_merge < 1 || p->groups % spec->group_merge != 0))
    {
        NONGROUPED_REJECT(
            "group_merge %d must be positive and divide groups %d", spec->group_merge, p->groups);
    }
    if(!spec->grouped && spec->group_merge != 1)
    {
        NONGROUPED_REJECT("group_merge %d needs grouped=True", spec->group_merge);
    }
    at = nongrouped_atom(spec->atom);
    if(at == NULL)
    {
        NONGROUPED_REJECT("unknown atom '%s'; expected one of ['32x32x16', '32x32x8', '16x16x32', "
                          "'16x16x16']",
                          spec->atom ? spec->atom : "None");
    }
    /* Checked before any derived geometry: those divide by these. A positive C
     * is also what lets the kernel assume its channel loop runs at least once. */
    if(spec->waves_m <= 0 || spec->waves_n <= 0 || spec->wave_size <= 0 || spec->ck <= 0
       || spec->tile_h <= 0 || spec->tile_w <= 0 || spec->tile_k <= 0 || p->stride <= 0
       || p->cpg <= 0 || p->kpg <= 0)
    {
        NONGROUPED_REJECT("DirectNongroupedConvSpec: tile, wave, stride and channel parameters "
                          "must be positive");
    }
    if(p->dil_h < 1 || p->dil_w < 1)
    {
        NONGROUPED_REJECT(
            "DirectNongroupedConvSpec: dilation must be positive (got %dx%d)", p->dil_h, p->dil_w);
    }
    t = at->tile;
    if(spec->tile_w % t != 0)
    {
        NONGROUPED_REJECT("tile_w must be a multiple of %d (got %d)", t, spec->tile_w);
    }
    if(spec->tile_k % t != 0)
    {
        NONGROUPED_REJECT("tile_k must be a multiple of %d (got %d)", t, spec->tile_k);
    }
    if(spec->tile_h % spec->waves_n != 0)
    {
        NONGROUPED_REJECT("tile_h %d not divisible by waves_n %d", spec->tile_h, spec->waves_n);
    }
    if((spec->tile_k / t) % spec->waves_m != 0)
    {
        NONGROUPED_REJECT(
            "tile_k/%d = %d not divisible by waves_m %d", t, spec->tile_k / t, spec->waves_m);
    }
    if(spec->ck % at->k != 0)
    {
        NONGROUPED_REJECT("ck %d not divisible by atom K %d", spec->ck, at->k);
    }
    if(spec->ck % ROCKE_DCONV_NONGROUPED_X_LOAD_VEC != 0)
    {
        NONGROUPED_REJECT("ck %d not divisible by %d", spec->ck, ROCKE_DCONV_NONGROUPED_X_LOAD_VEC);
    }
    merged_c = nongrouped_merged_c(spec);
    merged_k = nongrouped_merged_k(spec);
    if(merged_c % spec->ck != 0)
    {
        NONGROUPED_REJECT("C %d not divisible by ck %d", merged_c, spec->ck);
    }
    if(merged_k % t != 0)
    {
        NONGROUPED_REJECT("K %d must be a multiple of %d (got %d)", merged_k, t, merged_k);
    }
    if(merged_c % ROCKE_DCONV_NONGROUPED_X_LOAD_VEC != 0)
    {
        NONGROUPED_REJECT(
            "C %d must be a multiple of %d", merged_c, ROCKE_DCONV_NONGROUPED_X_LOAD_VEC);
    }
    if(spec->grouped && p->cpg % nongrouped_w_vec(spec) != 0)
    {
        NONGROUPED_REJECT("cpg %d must be a multiple of the %d-channel weight load",
                          p->cpg,
                          nongrouped_w_vec(spec));
    }
    if(rocke_direct_conv_nongrouped_threads_per_block(spec) > 1024)
    {
        NONGROUPED_REJECT("threads_per_block %d > 1024",
                          rocke_direct_conv_nongrouped_threads_per_block(spec));
    }
    if(spec->lds_pad < 0 || spec->lds_pad % 8 != 0)
    {
        /* A negative pad shrinks the pixel stride below the ck halves each
         * pixel stages, so neighbouring pixels overlap in LDS. The vec8 LDS
         * store/load is emitted with align 16, so the pixel stride
         * (ck + lds_pad halves) must be a multiple of 8 halves. */
        NONGROUPED_REJECT(
            "lds_pad must be a non-negative multiple of 8 to keep ds_read_b128 aligned (got %d)",
            spec->lds_pad);
    }
    if(spec->swizzle_wgm < 1)
    {
        NONGROUPED_REJECT("swizzle_wgm must be >= 1 (got %d)", spec->swizzle_wgm);
    }
    /* -1 / 0 are this port's "None" sentinels; anything below them has no
     * Python spelling, and Python rejects the sentinel values themselves. */
    if(spec->iglp < ROCKE_DCONV_NONGROUPED_IGLP_NONE)
    {
        NONGROUPED_REJECT("iglp must be None or >= 0 (got %d)", spec->iglp);
    }
    if(spec->waves_per_eu < ROCKE_DCONV_NONGROUPED_WAVES_PER_EU_NONE)
    {
        NONGROUPED_REJECT("waves_per_eu must be None or >= 1 (got %d)", spec->waves_per_eu);
    }
    {
        /* The kernel decodes its tile from a flat 1-D block id and computes the
         * workgroup count in i32; past gridDim.x's 2**31 - 1 limit both the
         * launch and that product would overflow. (The D-size bound below
         * implies this one; it is checked on its own so the reason names the
         * real limit.) */
        int n_wt, n_ht, n_kt;
        long long grid_x;
        nongrouped_tile_counts(spec, &n_wt, &n_ht, &n_kt);
        grid_x = (long long)n_wt * n_ht * p->N * nongrouped_n_mg(spec) * n_kt * spec->split_c;
        if(grid_x > INT32_MAX)
        {
            NONGROUPED_REJECT("flattened grid of %lld workgroups exceeds %d: grow the tile or "
                              "split the batch",
                              grid_x,
                              INT32_MAX);
        }
    }
    {
        /* Masked staging loads and epilogue stores go to OOB_BASE and rely on
         * it being past num_records; a larger tensor would hand the loads real
         * data and take the stores. */
        long long total_c = (long long)p->groups * p->cpg;
        long long total_k = (long long)p->groups * p->kpg;
        long long a_bytes = 2LL * p->N * p->H * p->W * total_c;
        long long b_bytes = 2LL * total_k * p->KH * p->KW * p->cpg;
        long long d_bytes = 2LL * p->N * nongrouped_ho(p) * nongrouped_wo(p) * total_k;
        long long biggest = a_bytes > b_bytes ? a_bytes : b_bytes;
        biggest = d_bytes > biggest ? d_bytes : biggest;
        if(biggest > ROCKE_DCONV_NONGROUPED_OOB_BASE)
        {
            NONGROUPED_REJECT("tensors must fit below the %#x-byte masked-access offset (A %lld B, "
                              "B %lld B, D %lld B)",
                              ROCKE_DCONV_NONGROUPED_OOB_BASE,
                              a_bytes,
                              b_bytes,
                              d_bytes);
        }
        /* The last prefetch runs one chunk (two when double-buffered) past C. */
        if((long long)ROCKE_DCONV_NONGROUPED_OOB_BASE + 2LL * (total_c + spec->ck) > INT32_MAX)
        {
            NONGROUPED_REJECT("C %d too large: the masked-load offset plus the channel offset "
                              "would overflow i32",
                              p->cpg);
        }
    }
    if(spec->split_c < 1)
    {
        NONGROUPED_REJECT("split_c must be >= 1 (got %d)", spec->split_c);
    }
    if(merged_c / spec->ck < spec->split_c)
    {
        /* Every split owns at least one whole chunk: the channel loop assumes
         * it runs at least once. */
        NONGROUPED_REJECT("C %d has %d chunks of ck=%d, fewer than split_c=%d",
                          merged_c,
                          merged_c / spec->ck,
                          spec->ck,
                          spec->split_c);
    }
    if(rocke_direct_conv_nongrouped_workspace_bytes(spec) > ROCKE_DCONV_NONGROUPED_OOB_BASE)
    {
        NONGROUPED_REJECT(
            "split_c=%d workspace of %lld B exceeds the %#x-byte masked-access offset",
            spec->split_c,
            rocke_direct_conv_nongrouped_workspace_bytes(spec),
            ROCKE_DCONV_NONGROUPED_OOB_BASE);
    }
    if(rocke_direct_conv_nongrouped_acc_vgprs(spec) > 256)
    {
        /* Past the 256-entry accumulator file the backend scheduler does not
         * just slow down: a 2048-accumulator tile hung the compiler. */
        NONGROUPED_REJECT("accumulator tile needs %d registers per lane (max 256): shrink "
                          "tile_k/tile_h/tile_w or add waves",
                          rocke_direct_conv_nongrouped_acc_vgprs(spec));
    }
#undef NONGROUPED_REJECT

    nongrouped_set_reason(reason, reason_cap, "ok");
    return ROCKE_OK;
}

/* ===================================================================== *
 *  is_valid_nongrouped_spec(spec, arch)
 * ===================================================================== */
bool rocke_direct_conv_nongrouped_is_valid_spec(const rocke_direct_conv_nongrouped_spec_t* spec,
                                                const char* arch,
                                                char* reason,
                                                size_t reason_cap)
{
    const rocke_archtarget_t* target;
    const rocke_direct_conv_problem_t* p;
    const rocke_dconv_nongrouped_atom_t* at;
    const char* ab;
    char msg[ROCKE_ERR_MSG_CAP];

    if(spec == NULL)
    {
        nongrouped_set_reason(reason, reason_cap, "null spec");
        return false;
    }
    if(arch == NULL)
    {
        arch = "gfx950";
    }
    target = rocke_archtarget_from_gfx(arch);
    if(target == NULL)
    {
        /* Same text as Python's str(KeyError) from ArchTarget.from_gfx. */
        rocke_set_unknown_arch_reason(reason, reason_cap, arch);
        return false;
    }
    if(rocke_direct_conv_nongrouped_validate(spec, reason, reason_cap) != ROCKE_OK)
    {
        return false;
    }
    p = &spec->problem;
    if(p->stride != 1 && p->stride != 2)
    {
        snprintf(msg, sizeof(msg), "stride %d is not supported (expected 1 or 2)", p->stride);
        nongrouped_set_reason(reason, reason_cap, msg);
        return false;
    }
    at = nongrouped_atom(spec->atom);
    ab = nongrouped_is_bf16(spec) ? "bf16" : "f16";
    if(!rocke_mma_catalog_has_shape(
           rocke_archtarget_mma(target), "mma", ab, ab, "fp32", at->tile, at->tile, at->k))
    {
        snprintf(msg, sizeof(msg), "missing mfma_f32_%s_%s on %s", at->name, ab, arch);
        nongrouped_set_reason(reason, reason_cap, msg);
        return false;
    }
    if(spec->wave_size != target->wave_size)
    {
        snprintf(msg,
                 sizeof(msg),
                 "wave_size %d != %s wave %d",
                 spec->wave_size,
                 arch,
                 target->wave_size);
        nongrouped_set_reason(reason, reason_cap, msg);
        return false;
    }
    if(rocke_direct_conv_nongrouped_threads_per_block(spec)
       > rocke_archtarget_max_threads_per_block(target))
    {
        snprintf(msg,
                 sizeof(msg),
                 "threads_per_block %d exceeds arch limit",
                 rocke_direct_conv_nongrouped_threads_per_block(spec));
        nongrouped_set_reason(reason, reason_cap, msg);
        return false;
    }
    if(!rocke_archtarget_fits_lds(target, rocke_direct_conv_nongrouped_lds_bytes(spec)))
    {
        snprintf(msg,
                 sizeof(msg),
                 "LDS %ld B exceeds %d B",
                 rocke_direct_conv_nongrouped_lds_bytes(spec),
                 target->lds_capacity_bytes);
        nongrouped_set_reason(reason, reason_cap, msg);
        return false;
    }
    nongrouped_set_reason(reason, reason_cap, "ok");
    return true;
}

/* ===================================================================== *
 *  Builder context  (the locals the Python closures share)
 * ===================================================================== */
namespace
{

/* One W_VEC-channel piece of a merged (group_merge > 1) weight fragment. */
struct nongrouped_w_piece
{
    rocke_value_t* elems; /* element offset of the piece's first channel at chunk 0 */
    rocke_value_t* c_rel; /* its channel relative to k_out's group span             */
};

struct nongrouped_stage_meta
{
    rocke_value_t* base; /* global byte offset base (chunk offset added per iter) */
    rocke_value_t* lds_idx; /* LDS half index of this thread's slot                 */
    /* Merged-group weights only (base unused): the slot's k_out < K_M mask and
     * the per-piece offsets, masked per chunk in issue_stage_loads. */
    rocke_value_t* valid;
    std::vector<nongrouped_w_piece> pieces;
};

struct nongrouped_ctx
{
    rocke_ir_builder_t* b;
    const rocke_direct_conv_nongrouped_spec_t* spec;
    rocke_direct_conv_problem_t p;
    const rocke_type_t* io_type;
    bool bf16;
    rocke_dconv_nongrouped_mfma_fn mfma;

    /* Build-time geometry (Python all-caps locals). */
    int KH, KW, S, PAD;
    int DH, DW; /* filter dilation */
    int TH, TW, TK, CK, THREADS, WAVE, FRAG, AK, AT, ACC, QUADS;
    int NCB, M_TILES, KATOMS, ROWS_W, MT_W, NT_W, CSTRIDE, LDS_IN_H, LDS_IN_W;
    int X_CV, X_VECS, X_PASSES, W_SLOTS, W_PASSES;
    int x_stage, w_stage, x_dump, w_dump;
    bool DB;
    int SPLIT; /* spec->split_c */
    bool GROUPED; /* spec->grouped */
    int GM; /* spec->group_merge */
    int W_VEC; /* channels per weight staging load (Python spec.w_vec) */

    /* The AOT kernarg block. C / K are p_total_c / p_total_k. Dense: C_M, K_M
     * and C_W alias them. Grouped: C_M / K_M are the merged block's channel
     * counts a workgroup reduces / produces and C_W (= cpg) the weight row. */
    rocke_dconv_params_t params;
    rocke_value_t *C, *K;
    rocke_value_t *cpg_v, *kpg_v, *C_M, *K_M, *C_W; /* cpg_v / kpg_v NULL when dense */

    /* SSA values. */
    rocke_value_t *c0, *c1, *c_half, *oob_base;
    rocke_value_t *a_rsrc, *b_rsrc, *d_rsrc;
    rocke_value_t *X_smem, *W_smem;
    rocke_value_t *tid, *lane, *wave_id, *wave_m, *wave_n, *lane_lo, *lane_hi;
    rocke_value_t *k_tile, *cell, *n_img, *out_h0, *out_w0, *k_base, *in_h0, *in_w0;
    rocke_value_t* split; /* this workgroup's split; NULL when SPLIT == 1 */
    /* Channel offsets of the block's merged group in X (and D) and K; NULL
     * when dense. */
    rocke_value_t *a_ch_base, *k_glob_base;
    rocke_value_t *x_read_base, *w_read_base;

    std::vector<nongrouped_stage_meta> x_meta;
    std::vector<nongrouped_stage_meta> w_meta;
};

rocke_value_t* nongrouped_c(nongrouped_ctx& c, int64_t v)
{
    return rocke_b_const_i32(c.b, v);
}

rocke_value_t*
    nongrouped_buf_load(nongrouped_ctx& c, rocke_value_t* rsrc, rocke_value_t* voff, int dwords)
{
    if(c.bf16)
    {
        return rocke_b_buffer_load_vN_bf16(c.b, rsrc, voff, c.c0, dwords);
    }
    return rocke_b_buffer_load_vN_f16(c.b, rsrc, voff, c.c0, dwords);
}

/* ---- prologue: attrs, params, constants, rsrcs, LDS, thread decode ---- */
bool nongrouped_prologue(nongrouped_ctx& c, const char* arch)
{
    rocke_ir_builder_t* b = c.b;
    const rocke_direct_conv_nongrouped_spec_t* spec = c.spec;
    const rocke_dconv_nongrouped_atom_t* at;
    char reason[ROCKE_ERR_MSG_CAP];

    if(!rocke_direct_conv_nongrouped_is_valid_spec(spec, arch, reason, sizeof reason))
    {
        if(b->status == ROCKE_OK)
        {
            b->status = ROCKE_ERR_VALUE;
            std::string msg
                = std::string("invalid DirectNongroupedConvSpec for ") + arch + ": " + reason;
            snprintf(b->err, sizeof b->err, "%.*s", (int)(sizeof b->err - 1), msg.c_str());
        }
        return false;
    }

    c.p = spec->problem;
    c.io_type = rocke_b_io_ir_type(b, nongrouped_dtype(spec));
    if(c.io_type == NULL)
    {
        return false;
    }
    c.bf16 = nongrouped_is_bf16(spec);
    at = nongrouped_atom(spec->atom);
    c.mfma = c.bf16 ? at->bf16 : at->f16;

    c.KH = c.p.KH;
    c.KW = c.p.KW;
    c.S = c.p.stride;
    c.PAD = c.p.PAD;
    c.DH = c.p.dil_h;
    c.DW = c.p.dil_w;

    c.TH = spec->tile_h;
    c.TW = spec->tile_w;
    c.TK = spec->tile_k;
    c.CK = spec->ck;
    c.THREADS = rocke_direct_conv_nongrouped_threads_per_block(spec);
    c.WAVE = spec->wave_size;
    c.FRAG = at->frag;
    c.AK = at->k;
    c.AT = at->tile;
    c.ACC = c.AT * c.AT / c.WAVE;
    c.QUADS = c.ACC / 4;
    c.NCB = c.TW / c.AT;
    c.M_TILES = c.TK / c.AT;
    c.KATOMS = c.CK / c.AK;
    c.ROWS_W = c.TH / spec->waves_n;
    c.MT_W = c.M_TILES / spec->waves_m;
    c.NT_W = c.ROWS_W * c.NCB;
    c.CSTRIDE = c.CK + spec->lds_pad;
    c.LDS_IN_H = (c.TH - 1) * c.S + (c.KH - 1) * c.DH + 1;
    c.LDS_IN_W = (c.TW - 1) * c.S + (c.KW - 1) * c.DW + 1;

    c.X_CV = c.CK / ROCKE_DCONV_NONGROUPED_X_LOAD_VEC;
    c.X_VECS = c.LDS_IN_H * c.LDS_IN_W * c.X_CV;
    c.X_PASSES = (c.X_VECS + c.THREADS - 1) / c.THREADS;
    c.W_SLOTS = c.KH * c.KW * c.M_TILES * c.KATOMS * c.WAVE;
    c.W_PASSES = (c.W_SLOTS + c.THREADS - 1) / c.THREADS;
    c.SPLIT = spec->split_c;
    c.GROUPED = spec->grouped;
    c.GM = spec->group_merge;
    c.W_VEC = (c.GROUPED && c.GM > 1) ? 4 : c.FRAG;

    rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", c.THREADS);
    if(spec->waves_per_eu != ROCKE_DCONV_NONGROUPED_WAVES_PER_EU_NONE)
    {
        rocke_attr_set_int(b, &b->kernel->attrs, "waves_per_eu", spec->waves_per_eu);
    }

    rocke_dconv_emit_params(b, &c.params, "fwd", c.io_type);
    if(!rocke_ir_builder_ok(b))
    {
        return false;
    }
    c.C = c.params.p_total_c;
    c.K = c.params.p_total_k;
    if(c.GROUPED)
    {
        /* Per-group channel counts, and the merged block's: C_W is the weight
         * row length, C_M / K_M the channels a workgroup reduces / produces. */
        rocke_value_t* k;
        c.cpg_v = rocke_b_div(b, c.C, c.params.p_groups);
        c.kpg_v = rocke_b_div(b, c.K, c.params.p_groups);
        c.C_M = c.cpg_v;
        if(c.GM > 1)
        {
            k = nongrouped_c(c, c.GM);
            c.C_M = rocke_b_mul(b, c.cpg_v, k);
        }
        c.K_M = c.kpg_v;
        if(c.GM > 1)
        {
            k = nongrouped_c(c, c.GM);
            c.K_M = rocke_b_mul(b, c.kpg_v, k);
        }
        c.C_W = c.cpg_v;
    }
    else
    {
        c.cpg_v = NULL;
        c.kpg_v = NULL;
        c.C_M = c.C;
        c.K_M = c.K;
        c.C_W = c.C;
    }

    c.c0 = nongrouped_c(c, 0);
    c.c1 = nongrouped_c(c, 1);
    c.c_half = nongrouped_c(c, 2);
    /* Shared by the masked staging loads and the masked epilogue stores. */
    c.oob_base = nongrouped_c(c, ROCKE_DCONV_NONGROUPED_OOB_BASE);

    c.a_rsrc = rocke_b_buffer_rsrc(b, c.params.A, c.params.A_bytes);
    c.b_rsrc = rocke_b_buffer_rsrc(b, c.params.Bp, c.params.B_bytes);
    c.d_rsrc = rocke_b_buffer_rsrc(b, c.params.D, c.params.D_bytes);

    /* The staging loops are sized in whole thread-passes, so the last pass can
     * own slots past the end of the tile; those write into a scratch tail. */
    {
        long x_halves = (long)c.LDS_IN_H * c.LDS_IN_W * c.CSTRIDE;
        long w_halves = (long)c.TK * c.KH * c.KW * c.CK;
        int nbuf;
        int shape[2];
        c.DB = spec->double_buffer;
        c.x_stage = (int)(x_halves + ROCKE_DCONV_NONGROUPED_X_LOAD_VEC);
        c.w_stage = (int)(w_halves + c.FRAG);
        c.x_dump = (int)x_halves;
        c.w_dump = (int)w_halves;
        nbuf = c.DB ? 2 : 1;
        shape[0] = 1;
        shape[1] = c.x_stage * nbuf;
        c.X_smem = rocke_b_smem_alloc(b, c.io_type, shape, 2, "lds_x");
        shape[1] = c.w_stage * nbuf;
        c.W_smem = rocke_b_smem_alloc(b, c.io_type, shape, 2, "lds_w");
    }

    /* Thread / wave decomposition. */
    {
        rocke_value_t* k;
        c.tid = rocke_b_thread_id_x(b);
        k = nongrouped_c(c, c.WAVE);
        c.lane = rocke_b_mod(b, c.tid, k);
        k = nongrouped_c(c, c.WAVE);
        c.wave_id = rocke_b_div(b, c.tid, k);
        k = nongrouped_c(c, spec->waves_n);
        c.wave_m = rocke_b_div(b, c.wave_id, k);
        k = nongrouped_c(c, spec->waves_n);
        c.wave_n = rocke_b_mod(b, c.wave_id, k);
        k = nongrouped_c(c, c.AT);
        c.lane_lo = rocke_b_mod(b, c.lane, k);
        k = nongrouped_c(c, c.AT);
        c.lane_hi = rocke_b_div(b, c.lane, k);
    }
    return rocke_ir_builder_ok(b);
}

/* ---- grid decode: flat 1-D grid of (spatial cell x channel tile) ----
 * The tile sizes are build-time; the tile counts follow the runtime extents. */
void nongrouped_grid_decode(nongrouped_ctx& c)
{
    rocke_ir_builder_t* b = c.b;
    const rocke_direct_conv_nongrouped_spec_t* spec = c.spec;
    rocke_value_t *wgid, *k, *t, *n_wt, *n_ht, *n_kt, *n_hw, *n_cells, *n_cols, *st, *w_tile,
        *h_tile, *n_mg = NULL;

    k = nongrouped_c(c, c.TW - 1);
    t = rocke_b_add(b, c.params.p_Wo, k);
    k = nongrouped_c(c, c.TW);
    n_wt = rocke_b_div(b, t, k);
    k = nongrouped_c(c, c.TH - 1);
    t = rocke_b_add(b, c.params.p_Ho, k);
    k = nongrouped_c(c, c.TH);
    n_ht = rocke_b_div(b, t, k);
    k = nongrouped_c(c, c.TK - 1);
    t = rocke_b_add(b, c.K_M, k);
    k = nongrouped_c(c, c.TK);
    n_kt = rocke_b_div(b, t, k);
    n_hw = rocke_b_mul(b, n_wt, n_ht);
    n_cells = rocke_b_mul(b, n_hw, c.params.p_N);
    if(c.GROUPED)
    {
        /* Merged group innermost on the cell axis: (n * n_hw + st) * n_mg + mg.
         * The groups of one pixel tile share every cache line of its input
         * pixels, so they run back to back and each line is fetched once. */
        n_mg = c.params.p_groups;
        if(c.GM > 1)
        {
            k = nongrouped_c(c, c.GM);
            n_mg = rocke_b_div(b, c.params.p_groups, k);
        }
        n_cells = rocke_b_mul(b, n_cells, n_mg);
    }
    /* A split kernel's cell axis is split-major: split * n_cells + cell, so the
     * swizzle treats the splits as more spatial cells of the same grid. */
    n_cols = n_cells;
    if(c.SPLIT > 1)
    {
        k = nongrouped_c(c, c.SPLIT);
        n_cols = rocke_b_mul(b, n_cells, k);
    }

    wgid = rocke_b_block_id_x(b);
    if(spec->chiplet_swizzle)
    {
        /* A swizzle_wgm above the runtime channel-tile count needs no clamp:
         * the swizzle sizes its last group as min(wgm, tiles left). */
        rocke_super_tile_swizzle_result_t sw = rocke_chiplet_aware_super_tile_dynamic(
            b, wgid, n_kt, n_cols, spec->swizzle_wgm, spec->num_xcds, spec->chiplet_chunk);
        c.k_tile = sw.row;
        c.cell = sw.col;
    }
    else
    {
        c.k_tile = rocke_b_mod(b, wgid, n_kt);
        c.cell = rocke_b_div(b, wgid, n_kt);
    }
    c.split = NULL;
    if(c.SPLIT > 1)
    {
        c.split = rocke_b_div(b, c.cell, n_cells);
        c.cell = rocke_b_mod(b, c.cell, n_cells);
    }

    c.a_ch_base = NULL;
    c.k_glob_base = NULL;
    if(c.GROUPED)
    {
        rocke_value_t* mg = rocke_b_mod(b, c.cell, n_mg);
        c.cell = rocke_b_div(b, c.cell, n_mg);
        /* Channel offsets of the block's merged group in X (and D) and K. */
        c.a_ch_base = rocke_b_mul(b, mg, c.C_M);
        c.k_glob_base = rocke_b_mul(b, mg, c.K_M);
    }
    c.n_img = rocke_b_div(b, c.cell, n_hw);
    st = rocke_b_mod(b, c.cell, n_hw);
    w_tile = rocke_b_mod(b, st, n_wt);
    h_tile = rocke_b_div(b, st, n_wt);

    /* Output-tile origins. */
    k = nongrouped_c(c, c.TH);
    c.out_h0 = rocke_b_mul(b, h_tile, k);
    k = nongrouped_c(c, c.TW);
    c.out_w0 = rocke_b_mul(b, w_tile, k);
    k = nongrouped_c(c, c.TK);
    c.k_base = rocke_b_mul(b, c.k_tile, k);

    /* Input origin of the staged (halo-inclusive) activation tile. */
    k = nongrouped_c(c, c.S);
    t = rocke_b_mul(b, c.out_h0, k);
    k = nongrouped_c(c, c.PAD);
    c.in_h0 = rocke_b_sub(b, t, k);
    k = nongrouped_c(c, c.S);
    t = rocke_b_mul(b, c.out_w0, k);
    k = nongrouped_c(c, c.PAD);
    c.in_w0 = rocke_b_sub(b, t, k);
}

/* ---- _x_pass_meta(): per-pass (base, lds_idx) for the activation tile ---- */
void nongrouped_x_pass_meta(nongrouped_ctx& c)
{
    rocke_ir_builder_t* b = c.b;
    rocke_value_t *k, *img_base;
    int j;

    img_base = rocke_b_mul(b, c.n_img, c.params.p_A_stride_n);
    if(c.GROUPED)
    {
        img_base = rocke_b_add(b, img_base, c.a_ch_base);
    }
    for(j = 0; j < c.X_PASSES; ++j)
    {
        rocke_value_t *v, *cv, *pos, *ih_l, *iw_l, *ih, *iw;
        rocke_value_t *ge, *lt, *ok_h, *ok_w, *valid, *in_tile = NULL;
        rocke_value_t *t0, *t1, *t2, *elems, *lds_idx, *base;

        k = nongrouped_c(c, (int64_t)j * c.THREADS);
        v = rocke_b_add(b, c.tid, k);
        k = nongrouped_c(c, c.X_CV);
        cv = rocke_b_mod(b, v, k);
        k = nongrouped_c(c, c.X_CV);
        pos = rocke_b_div(b, v, k);
        k = nongrouped_c(c, c.LDS_IN_W);
        ih_l = rocke_b_div(b, pos, k);
        k = nongrouped_c(c, c.LDS_IN_W);
        iw_l = rocke_b_mod(b, pos, k);
        ih = rocke_b_add(b, c.in_h0, ih_l);
        iw = rocke_b_add(b, c.in_w0, iw_l);

        ge = rocke_b_cmp_ge(b, ih, c.c0);
        lt = rocke_b_cmp_lt(b, ih, c.params.p_Hi);
        ok_h = rocke_b_land(b, ge, lt);
        ge = rocke_b_cmp_ge(b, iw, c.c0);
        lt = rocke_b_cmp_lt(b, iw, c.params.p_Wi);
        ok_w = rocke_b_land(b, ge, lt);
        valid = rocke_b_land(b, ok_h, ok_w);
        if(c.X_VECS % c.THREADS != 0)
        {
            k = nongrouped_c(c, c.X_VECS);
            in_tile = rocke_b_cmp_lt(b, v, k);
            valid = rocke_b_land(b, valid, in_tile);
        }

        /* base = n*stride_n + ih*stride_hi + iw*stride_wi + cv*VEC; the chunk
         * offset is added per iteration. */
        t0 = rocke_b_mul(b, ih, c.params.p_A_stride_hi);
        t1 = rocke_b_mul(b, iw, c.params.p_A_stride_wi);
        t0 = rocke_b_add(b, t0, t1);
        t1 = rocke_b_add(b, img_base, t0);
        k = nongrouped_c(c, ROCKE_DCONV_NONGROUPED_X_LOAD_VEC);
        t2 = rocke_b_mul(b, cv, k);
        elems = rocke_b_add(b, t1, t2);

        k = nongrouped_c(c, c.CSTRIDE);
        t0 = rocke_b_mul(b, pos, k);
        k = nongrouped_c(c, ROCKE_DCONV_NONGROUPED_X_LOAD_VEC);
        t1 = rocke_b_mul(b, cv, k);
        lds_idx = rocke_b_add(b, t0, t1);
        if(in_tile != NULL)
        {
            k = nongrouped_c(c, c.x_dump);
            lds_idx = rocke_b_select(b, in_tile, lds_idx, k);
        }
        t0 = rocke_b_mul(b, elems, c.c_half);
        base = rocke_b_select(b, valid, t0, c.oob_base);
        c.x_meta.push_back({base, lds_idx, NULL, {}});
    }
}

/* ---- _w_pass_meta(): per-pass (base, lds_idx) for the fragment-order W ---- */
void nongrouped_w_pass_meta(nongrouped_ctx& c)
{
    rocke_ir_builder_t* b = c.b;
    int j;

    for(j = 0; j < c.W_PASSES; ++j)
    {
        rocke_value_t *k, *g, *g_lane, *rest, *katom, *rest2, *m_tile, *tap, *r, *s;
        rocke_value_t *t0, *t1, *k_out, *c_in_chunk, *valid, *in_tile = NULL;
        rocke_value_t *krs_h, *krs_hr, *krs_w, *krs, *elems, *lds_idx, *base;

        k = nongrouped_c(c, (int64_t)j * c.THREADS);
        g = rocke_b_add(b, c.tid, k);
        k = nongrouped_c(c, c.WAVE);
        g_lane = rocke_b_mod(b, g, k);
        k = nongrouped_c(c, c.WAVE);
        rest = rocke_b_div(b, g, k);
        k = nongrouped_c(c, c.KATOMS);
        katom = rocke_b_mod(b, rest, k);
        k = nongrouped_c(c, c.KATOMS);
        rest2 = rocke_b_div(b, rest, k);
        k = nongrouped_c(c, c.M_TILES);
        m_tile = rocke_b_mod(b, rest2, k);
        k = nongrouped_c(c, c.M_TILES);
        tap = rocke_b_div(b, rest2, k);
        k = nongrouped_c(c, c.KW);
        r = rocke_b_div(b, tap, k);
        k = nongrouped_c(c, c.KW);
        s = rocke_b_mod(b, tap, k);

        k = nongrouped_c(c, c.AT);
        t0 = rocke_b_mul(b, m_tile, k);
        k = nongrouped_c(c, c.AT);
        t1 = rocke_b_mod(b, g_lane, k);
        t0 = rocke_b_add(b, t0, t1);
        k_out = rocke_b_add(b, c.k_base, t0);

        k = nongrouped_c(c, c.AK);
        t0 = rocke_b_mul(b, katom, k);
        k = nongrouped_c(c, c.AT);
        t1 = rocke_b_div(b, g_lane, k);
        k = nongrouped_c(c, c.FRAG);
        t1 = rocke_b_mul(b, t1, k);
        c_in_chunk = rocke_b_add(b, t0, t1);

        valid = rocke_b_cmp_lt(b, k_out, c.K_M);
        if(c.W_SLOTS % c.THREADS != 0)
        {
            k = nongrouped_c(c, c.W_SLOTS);
            in_tile = rocke_b_cmp_lt(b, g, k);
            valid = rocke_b_land(b, valid, in_tile);
        }
        if(c.GROUPED && c.GM > 1)
        {
            /* Block-diagonal weights: k_out's group owns merged channels
             * [grp_k*cpg, (grp_k+1)*cpg). Each W_VEC-channel piece of the
             * fragment is loaded from its own offset relative to that span,
             * and masked per chunk (see issue_stage_loads). */
            nongrouped_stage_meta m;
            rocke_value_t *grp_k, *k_glob, *c_rel0;
            int h;
            grp_k = rocke_b_div(b, k_out, c.kpg_v);
            k_glob = rocke_b_add(b, k_out, c.k_glob_base);
            k = nongrouped_c(c, c.KH);
            krs_h = rocke_b_mul(b, k_glob, k);
            krs_hr = rocke_b_add(b, krs_h, r);
            k = nongrouped_c(c, c.KW);
            krs_w = rocke_b_mul(b, krs_hr, k);
            krs = rocke_b_add(b, krs_w, s);
            t0 = rocke_b_mul(b, grp_k, c.cpg_v);
            c_rel0 = rocke_b_sub(b, c_in_chunk, t0);
            for(h = 0; h < c.FRAG / c.W_VEC; ++h)
            {
                nongrouped_w_piece pc;
                k = nongrouped_c(c, (int64_t)h * c.W_VEC);
                pc.c_rel = rocke_b_add(b, c_rel0, k);
                t0 = rocke_b_mul(b, krs, c.C_W);
                pc.elems = rocke_b_add(b, t0, pc.c_rel);
                m.pieces.push_back(pc);
            }
            k = nongrouped_c(c, c.FRAG);
            lds_idx = rocke_b_mul(b, g, k);
            if(in_tile != NULL)
            {
                k = nongrouped_c(c, c.w_dump);
                lds_idx = rocke_b_select(b, in_tile, lds_idx, k);
            }
            m.base = NULL;
            m.lds_idx = lds_idx;
            m.valid = valid;
            c.w_meta.push_back(m);
            continue;
        }
        if(c.GROUPED)
        {
            k_out = rocke_b_add(b, k_out, c.k_glob_base);
        }

        /* base = ((k_out*KH + r)*KW + s)*C + c_in_chunk (KRSC) */
        k = nongrouped_c(c, c.KH);
        krs_h = rocke_b_mul(b, k_out, k);
        krs_hr = rocke_b_add(b, krs_h, r);
        k = nongrouped_c(c, c.KW);
        krs_w = rocke_b_mul(b, krs_hr, k);
        krs = rocke_b_add(b, krs_w, s);
        t0 = rocke_b_mul(b, krs, c.C_W);
        elems = rocke_b_add(b, t0, c_in_chunk);
        k = nongrouped_c(c, c.FRAG);
        lds_idx = rocke_b_mul(b, g, k);
        if(in_tile != NULL)
        {
            k = nongrouped_c(c, c.w_dump);
            lds_idx = rocke_b_select(b, in_tile, lds_idx, k);
        }
        t0 = rocke_b_mul(b, elems, c.c_half);
        base = rocke_b_select(b, valid, t0, c.oob_base);
        c.w_meta.push_back({base, lds_idx, NULL, {}});
    }
}

/* ---- issue_stage_loads(c_off): global loads for one channel chunk ---- */
void nongrouped_issue_stage_loads(nongrouped_ctx& c,
                                  rocke_value_t* c_off,
                                  std::vector<rocke_value_t*>& xs,
                                  std::vector<rocke_value_t*>& ws)
{
    xs.clear();
    ws.clear();
    for(const nongrouped_stage_meta& m : c.x_meta)
    {
        rocke_value_t* voff = rocke_b_add(c.b, m.base, c_off);
        xs.push_back(nongrouped_buf_load(c, c.a_rsrc, voff, ROCKE_DCONV_NONGROUPED_X_LOAD_VEC / 2));
    }
    if(c.GROUPED && c.GM > 1)
    {
        /* Merged channel of the chunk's first element; a piece belongs to
         * k_out's group iff its relative channel lands in [0, cpg). */
        rocke_ir_builder_t* b = c.b;
        rocke_value_t* c_el = rocke_b_div(b, c_off, c.c_half);
        for(const nongrouped_stage_meta& m : c.w_meta)
        {
            rocke_value_t* v = NULL;
            for(const nongrouped_w_piece& pc : m.pieces)
            {
                rocke_value_t *rel, *ge, *lt, *ok, *t, *off, *part;
                rel = rocke_b_add(b, pc.c_rel, c_el);
                ge = rocke_b_cmp_ge(b, rel, c.c0);
                lt = rocke_b_cmp_lt(b, rel, c.cpg_v);
                ok = rocke_b_land(b, ge, lt);
                ok = rocke_b_land(b, m.valid, ok);
                t = rocke_b_add(b, pc.elems, c_el);
                t = rocke_b_mul(b, t, c.c_half);
                off = rocke_b_select(b, ok, t, c.oob_base);
                part = nongrouped_buf_load(c, c.b_rsrc, off, c.W_VEC / 2);
                v = (v == NULL) ? part : rocke_b_vec_concat(b, v, part);
            }
            ws.push_back(v);
        }
        return;
    }
    for(const nongrouped_stage_meta& m : c.w_meta)
    {
        rocke_value_t* voff = rocke_b_add(c.b, m.base, c_off);
        ws.push_back(nongrouped_buf_load(c, c.b_rsrc, voff, c.FRAG / 2));
    }
}

/* ---- commit_stage(xs, ws, x_buf, w_buf): publish one chunk into LDS ---- */
void nongrouped_commit_stage(nongrouped_ctx& c,
                             const std::vector<rocke_value_t*>& xs,
                             const std::vector<rocke_value_t*>& ws,
                             rocke_value_t* x_buf,
                             rocke_value_t* w_buf)
{
    size_t i;
    for(i = 0; i < c.x_meta.size(); ++i)
    {
        rocke_value_t* idx = c.x_meta[i].lds_idx;
        rocke_value_t* ind[2];
        if(x_buf != NULL)
        {
            idx = rocke_b_add(c.b, idx, x_buf);
        }
        ind[0] = c.c0;
        ind[1] = idx;
        rocke_b_smem_store_vN(c.b, c.X_smem, ind, 2, xs[i], ROCKE_DCONV_NONGROUPED_X_LOAD_VEC);
    }
    for(i = 0; i < c.w_meta.size(); ++i)
    {
        rocke_value_t* idx = c.w_meta[i].lds_idx;
        rocke_value_t* ind[2];
        if(w_buf != NULL)
        {
            idx = rocke_b_add(c.b, idx, w_buf);
        }
        ind[0] = c.c0;
        ind[1] = idx;
        rocke_b_smem_store_vN(c.b, c.W_smem, ind, 2, ws[i], c.FRAG);
    }
}

/* ---- per-lane LDS read bases ---- */
void nongrouped_read_bases(nongrouped_ctx& c)
{
    rocke_ir_builder_t* b = c.b;
    rocke_value_t *k, *t0, *t1, *x_wave_term, *x_lane_term;

    /* X: idx = wave row term + lane term + compile-time term. */
    k = nongrouped_c(c, (int64_t)c.ROWS_W * c.S);
    t0 = rocke_b_mul(b, c.wave_n, k);
    k = nongrouped_c(c, (int64_t)c.LDS_IN_W * c.CSTRIDE);
    x_wave_term = rocke_b_mul(b, t0, k);
    k = nongrouped_c(c, (int64_t)c.S * c.CSTRIDE);
    t0 = rocke_b_mul(b, c.lane_lo, k);
    k = nongrouped_c(c, c.FRAG);
    t1 = rocke_b_mul(b, c.lane_hi, k);
    x_lane_term = rocke_b_add(b, t0, t1);
    c.x_read_base = rocke_b_add(b, x_wave_term, x_lane_term);

    /* W: idx = wave_m block + lane*FRAG + compile-time term. */
    k = nongrouped_c(c, (int64_t)c.MT_W * c.KATOMS * c.WAVE * c.FRAG);
    t0 = rocke_b_mul(b, c.wave_m, k);
    k = nongrouped_c(c, c.FRAG);
    t1 = rocke_b_mul(b, c.lane, k);
    c.w_read_base = rocke_b_add(b, t0, t1);
}

rocke_value_t*
    nongrouped_read_a_frag(nongrouped_ctx& c, rocke_value_t* base, int tap, int mt, int katom)
{
    int64_t off = ((int64_t)(tap * c.M_TILES + mt) * c.KATOMS + katom) * c.WAVE * c.FRAG;
    rocke_value_t* k = nongrouped_c(c, off);
    rocke_value_t* ind[2];
    ind[0] = c.c0;
    ind[1] = rocke_b_add(c.b, base, k);
    return rocke_b_smem_load_vN(c.b, c.W_smem, ind, 2, c.io_type, c.FRAG);
}

/* One activation fragment, addressed by *input* row inside the wave window, so
 * the KH taps of a column share ``(ROWS_W - 1) * stride + (KH - 1) * dil_h + 1``
 * fragments. Filter column s sits s * dil_w columns in. */
rocke_value_t* nongrouped_read_b_frag(
    nongrouped_ctx& c, rocke_value_t* base, int in_row, int cb, int s, int katom)
{
    int64_t pos = (int64_t)in_row * c.LDS_IN_W + (int64_t)cb * c.AT * c.S + (int64_t)s * c.DW;
    int64_t off = pos * c.CSTRIDE + (int64_t)katom * c.AK;
    rocke_value_t* k = nongrouped_c(c, off);
    rocke_value_t* ind[2];
    ind[0] = c.c0;
    ind[1] = rocke_b_add(c.b, base, k);
    return rocke_b_smem_load_vN(c.b, c.X_smem, ind, 2, c.io_type, c.FRAG);
}

/* ---- emit_mfmas(): one channel chunk of MFMAs against the staged tiles ---- */
void nongrouped_emit_mfmas(nongrouped_ctx& c,
                           std::vector<rocke_value_t*>& accs,
                           rocke_value_t* x_base,
                           rocke_value_t* w_base)
{
    const int n_in_rows = (c.ROWS_W - 1) * c.S + (c.KH - 1) * c.DH + 1;
    std::vector<rocke_value_t*> b_frags((size_t)n_in_rows * c.NCB);
    std::vector<rocke_value_t*> a_frags((size_t)c.MT_W);
    int katom, s_c, r_c, ir, cb, mt, row;

    /* Filter-column outermost: every activation fragment loaded for one
     * (s, k_atom) is consumed by all KH tap rows and all MT_W channel tiles. */
    for(katom = 0; katom < c.KATOMS; ++katom)
    {
        for(s_c = 0; s_c < c.KW; ++s_c)
        {
            for(ir = 0; ir < n_in_rows; ++ir)
            {
                for(cb = 0; cb < c.NCB; ++cb)
                {
                    b_frags[(size_t)ir * c.NCB + cb]
                        = nongrouped_read_b_frag(c, x_base, ir, cb, s_c, katom);
                }
            }
            for(r_c = 0; r_c < c.KH; ++r_c)
            {
                int tap = r_c * c.KW + s_c;
                for(mt = 0; mt < c.MT_W; ++mt)
                {
                    a_frags[(size_t)mt] = nongrouped_read_a_frag(c, w_base, tap, mt, katom);
                }
                for(row = 0; row < c.ROWS_W; ++row)
                {
                    int bf_row = row * c.S + r_c * c.DH;
                    for(cb = 0; cb < c.NCB; ++cb)
                    {
                        int nt = row * c.NCB + cb;
                        for(mt = 0; mt < c.MT_W; ++mt)
                        {
                            size_t idx = (size_t)mt * c.NT_W + nt;
                            accs[idx] = c.mfma(c.b,
                                               a_frags[(size_t)mt],
                                               b_frags[(size_t)bf_row * c.NCB + cb],
                                               accs[idx]);
                        }
                    }
                }
            }
        }
    }
}

/* ---- the runtime channel loop; returns the accumulator results ---- */
std::vector<rocke_value_t*> nongrouped_channel_loop(nongrouped_ctx& c)
{
    rocke_ir_builder_t* b = c.b;
    const int n_acc = c.MT_W * c.NT_W;
    rocke_value_t* zero_acc;
    rocke_value_t* c_ck_bytes;
    rocke_value_t* hi;
    rocke_value_t *chunk_lo = NULL, *chunk_hi = NULL, *c_start = NULL;
    std::vector<rocke_value_t*> xs0, ws0;
    std::vector<rocke_iter_arg_t> iter_args;
    std::vector<std::string> names;
    std::vector<rocke_value_t*> out;
    rocke_for_t loop;
    int i;

    zero_acc = rocke_b_zero_vec_f32(b, c.ACC);

    c_ck_bytes = nongrouped_c(c, (int64_t)c.CK * 2);
    if(c.SPLIT > 1)
    {
        /* Split `split` owns chunks [split*T/S, (split+1)*T/S) of the T chunks
         * of C; validate() keeps T >= S, so every split owns one. */
        rocke_value_t *k, *t, *t_chunks, *c_split;
        k = nongrouped_c(c, c.CK);
        t_chunks = rocke_b_div(b, c.C_M, k);
        c_split = nongrouped_c(c, c.SPLIT);
        t = rocke_b_mul(b, c.split, t_chunks);
        chunk_lo = rocke_b_div(b, t, c_split);
        t = rocke_b_add(b, c.split, c.c1);
        t = rocke_b_mul(b, t, t_chunks);
        chunk_hi = rocke_b_div(b, t, c_split);
        c_start = rocke_b_mul(b, chunk_lo, c_ck_bytes);
        nongrouped_issue_stage_loads(c, c_start, xs0, ws0);
    }
    else
    {
        nongrouped_issue_stage_loads(c, c.c0, xs0, ws0);
    }
    if(c.DB)
    {
        /* Prologue: publish chunk 0 into buffer 0 and have chunk 1 in flight. */
        rocke_value_t* nxt = c_ck_bytes;
        nongrouped_commit_stage(c, xs0, ws0, NULL, NULL);
        if(c.SPLIT > 1)
        {
            nxt = rocke_b_add(b, c_start, c_ck_bytes);
        }
        nongrouped_issue_stage_loads(c, nxt, xs0, ws0);
    }

    for(i = 0; i < n_acc; ++i)
    {
        names.push_back("acc" + std::to_string(i));
    }
    for(i = 0; i < (int)xs0.size(); ++i)
    {
        names.push_back("xs" + std::to_string(i));
    }
    for(i = 0; i < (int)ws0.size(); ++i)
    {
        names.push_back("ws" + std::to_string(i));
    }
    for(i = 0; i < n_acc; ++i)
    {
        iter_args.push_back({names[(size_t)i].c_str(), zero_acc});
    }
    for(i = 0; i < (int)xs0.size(); ++i)
    {
        iter_args.push_back({names[(size_t)(n_acc + i)].c_str(), xs0[(size_t)i]});
    }
    for(i = 0; i < (int)ws0.size(); ++i)
    {
        iter_args.push_back({names[(size_t)(n_acc + (int)xs0.size() + i)].c_str(), ws0[(size_t)i]});
    }

    /* Runtime trip count, clamped to one chunk. validate() guarantees C is a
     * positive whole number of chunks, so the clamp never changes it; it lets
     * LLVM prove the loop runs, without which it keeps a zero-trip guard and
     * schedules the prefetch loads late. */
    if(c.SPLIT > 1)
    {
        hi = rocke_b_sub(b, chunk_hi, chunk_lo);
    }
    else
    {
        hi = nongrouped_c(c, c.CK);
        hi = rocke_b_div(b, c.C_M, hi);
    }
    hi = rocke_b_smax(b, hi, c.c1);
    loop = rocke_b_scf_for_iter(
        b, c.c0, hi, c.c1, iter_args.data(), (int)iter_args.size(), "c_iter", false, false);
    if(loop.op == NULL || loop.iter_vars == NULL)
    {
        return out;
    }

    rocke_b_region_enter(b, loop.body);
    {
        std::vector<rocke_value_t*> accs(loop.iter_vars, loop.iter_vars + n_acc);
        std::vector<rocke_value_t*> xs(loop.iter_vars + n_acc, loop.iter_vars + n_acc + c.X_PASSES);
        std::vector<rocke_value_t*> ws(loop.iter_vars + n_acc + c.X_PASSES,
                                       loop.iter_vars + loop.num_iter_vars);
        std::vector<rocke_value_t*> xs_n, ws_n, yields;
        rocke_value_t *k, *t, *c_off;

        if(c.spec->iglp != ROCKE_DCONV_NONGROUPED_IGLP_NONE)
        {
            rocke_b_iglp_opt(b, c.spec->iglp);
        }

        if(c.DB)
        {
            /* Parity of the induction variable picks the live buffer; every
             * fragment offset stays a compile-time constant off these bases. */
            rocke_value_t *par, *cur_x, *cur_w, *nxt_x, *nxt_w, *xb, *wb;
            k = nongrouped_c(c, 2);
            par = rocke_b_mod(b, loop.iv, k);
            k = nongrouped_c(c, c.x_stage);
            cur_x = rocke_b_mul(b, par, k);
            k = nongrouped_c(c, c.w_stage);
            cur_w = rocke_b_mul(b, par, k);
            k = nongrouped_c(c, c.x_stage);
            nxt_x = rocke_b_sub(b, k, cur_x);
            k = nongrouped_c(c, c.w_stage);
            nxt_w = rocke_b_sub(b, k, cur_w);

            rocke_b_sync(b);
            /* Stage chunk i+1 into the idle buffer while chunk i feeds MFMAs. */
            nongrouped_commit_stage(c, xs, ws, nxt_x, nxt_w);
            k = nongrouped_c(c, 2);
            t = rocke_b_add(b, loop.iv, k);
            c_off = rocke_b_mul(b, t, c_ck_bytes);
            if(c.SPLIT > 1)
            {
                c_off = rocke_b_add(b, c_off, c_start);
            }
            nongrouped_issue_stage_loads(c, c_off, xs_n, ws_n);
            xb = rocke_b_add(b, c.x_read_base, cur_x);
            wb = rocke_b_add(b, c.w_read_base, cur_w);
            nongrouped_emit_mfmas(c, accs, xb, wb);
        }
        else
        {
            /* Publish the chunk prefetched during the previous iteration. */
            rocke_b_sync(b);
            nongrouped_commit_stage(c, xs, ws, NULL, NULL);
            rocke_b_sync(b);
            t = rocke_b_add(b, loop.iv, c.c1);
            c_off = rocke_b_mul(b, t, c_ck_bytes);
            if(c.SPLIT > 1)
            {
                c_off = rocke_b_add(b, c_off, c_start);
            }
            nongrouped_issue_stage_loads(c, c_off, xs_n, ws_n);
            nongrouped_emit_mfmas(c, accs, c.x_read_base, c.w_read_base);
        }

        yields.insert(yields.end(), accs.begin(), accs.end());
        yields.insert(yields.end(), xs_n.begin(), xs_n.end());
        yields.insert(yields.end(), ws_n.begin(), ws_n.end());
        rocke_b_scf_yield(b, yields.data(), (int)yields.size());
    }
    rocke_b_region_leave(b);

    for(i = 0; i < n_acc && i < loop.op->num_results; ++i)
    {
        out.push_back(loop.op->results[i]);
    }
    return out;
}

/* ---- epilogue: square-atom accumulator -> packed dwordx2 quad stores ---- */
void nongrouped_epilogue(nongrouped_ctx& c, const std::vector<rocke_value_t*>& accs_out)
{
    rocke_ir_builder_t* b = c.b;
    rocke_value_t *k, *t0, *t1, *k_lane_base, *row_base, *img_out_base, *c_f32 = NULL;
    const rocke_dconv_params_t* prm = &c.params;
    const bool split = c.SPLIT > 1;
    int mt, row, cb, q, jj;

    /* Slot i -> row = (i//4)*(AT//4) + lane_hi*4 + (i%4), col = lane_lo.
     * Slots 4q..4q+3 are four consecutive output channels. */
    k = nongrouped_c(c, (int64_t)c.MT_W * c.AT);
    t0 = rocke_b_mul(b, c.wave_m, k);
    t0 = rocke_b_add(b, c.k_base, t0);
    k = nongrouped_c(c, 4);
    t1 = rocke_b_mul(b, c.lane_hi, k);
    k_lane_base = rocke_b_add(b, t0, t1);
    k = nongrouped_c(c, c.ROWS_W);
    t0 = rocke_b_mul(b, c.wave_n, k);
    row_base = rocke_b_add(b, c.out_h0, t0);
    if(split)
    {
        /* D is the fp32 workspace: slab `split` of a packed
         * [split_c, N, Ho, Wo, K] array, whatever the output's strides. */
        t0 = rocke_b_mul(b, c.split, prm->p_N);
        img_out_base = rocke_b_add(b, t0, c.n_img);
        c_f32 = nongrouped_c(c, 4);
    }
    else
    {
        img_out_base = rocke_b_mul(b, c.n_img, prm->p_D_stride_n);
    }

    for(mt = 0; mt < c.MT_W; ++mt)
    {
        rocke_value_t* k_mt;
        k = nongrouped_c(c, (int64_t)mt * c.AT);
        k_mt = rocke_b_add(b, k_lane_base, k);
        for(row = 0; row < c.ROWS_W; ++row)
        {
            rocke_value_t *out_h, *h_ok, *row_off;
            k = nongrouped_c(c, row);
            out_h = rocke_b_add(b, row_base, k);
            h_ok = rocke_b_cmp_lt(b, out_h, prm->p_Ho);
            if(split)
            {
                t0 = rocke_b_mul(b, img_out_base, prm->p_Ho);
                t0 = rocke_b_add(b, t0, out_h);
                row_off = rocke_b_mul(b, t0, prm->p_Wo);
            }
            else
            {
                t0 = rocke_b_mul(b, out_h, prm->p_D_stride_ho);
                row_off = rocke_b_add(b, img_out_base, t0);
            }
            for(cb = 0; cb < c.NCB; ++cb)
            {
                int nt = row * c.NCB + cb;
                rocke_value_t* acc = accs_out[(size_t)mt * c.NT_W + nt];
                rocke_value_t *out_w, *hw_ok, *base_off;

                k = nongrouped_c(c, (int64_t)cb * c.AT);
                t0 = rocke_b_add(b, c.out_w0, k);
                out_w = rocke_b_add(b, t0, c.lane_lo);
                t0 = rocke_b_cmp_lt(b, out_w, prm->p_Wo);
                hw_ok = rocke_b_land(b, h_ok, t0);
                if(split)
                {
                    t0 = rocke_b_add(b, row_off, out_w);
                    base_off = rocke_b_mul(b, t0, c.K);
                }
                else
                {
                    t0 = rocke_b_mul(b, out_w, prm->p_D_stride_wo);
                    base_off = rocke_b_add(b, row_off, t0);
                }
                for(q = 0; q < c.QUADS; ++q)
                {
                    rocke_value_t *k_out, *valid, *d_off, *safe, *quad, *narrow;
                    rocke_value_t* comps[4];

                    k = nongrouped_c(c, (int64_t)q * (c.AT / 4));
                    k_out = rocke_b_add(b, k_mt, k);
                    t0 = rocke_b_cmp_lt(b, k_out, c.K_M);
                    valid = rocke_b_land(b, hw_ok, t0);
                    /* Output channel index = local (bounded by the merged K)
                     * + the merged group's base. */
                    if(c.GROUPED)
                    {
                        k_out = rocke_b_add(b, k_out, c.k_glob_base);
                    }
                    if(split)
                    {
                        rocke_value_t* ws_off;
                        t0 = rocke_b_add(b, base_off, k_out);
                        ws_off = rocke_b_mul(b, t0, c_f32);
                        safe = rocke_b_select(b, valid, ws_off, c.oob_base);
                        for(jj = 0; jj < 4; ++jj)
                        {
                            comps[jj] = rocke_b_vec_extract(b, acc, 4 * q + jj);
                        }
                        quad = rocke_b_vec_pack(b, comps, 4, rocke_f32());
                        rocke_b_buffer_store_vN_f32(b, c.d_rsrc, safe, c.c0, quad, 4);
                        continue;
                    }
                    t0 = rocke_b_add(b, base_off, k_out);
                    d_off = rocke_b_mul(b, t0, c.c_half);
                    /* Same sentinel as the staging loads: validate() keeps D at
                     * or below OOB_BASE, so the store lies past num_records. */
                    safe = rocke_b_select(b, valid, d_off, c.oob_base);
                    for(jj = 0; jj < 4; ++jj)
                    {
                        comps[jj] = rocke_b_vec_extract(b, acc, 4 * q + jj);
                    }
                    quad = rocke_b_vec_pack(b, comps, 4, rocke_f32());
                    if(c.bf16)
                    {
                        narrow = rocke_b_vec_trunc_f32_to_bf16(b, quad);
                        rocke_b_buffer_store_vN_bf16(b, c.d_rsrc, safe, c.c0, narrow, 2);
                    }
                    else
                    {
                        narrow = rocke_b_vec_trunc_f32_to_f16(b, quad);
                        rocke_b_buffer_store_vN_f16(b, c.d_rsrc, safe, c.c0, narrow, 2);
                    }
                }
            }
        }
    }
}

void nongrouped_set_err(char* err, size_t err_cap, const char* msg)
{
    if(err != NULL && err_cap > 0)
    {
        snprintf(err, err_cap, "%s", msg ? msg : "");
    }
}

} // namespace

/* ===================================================================== *
 *  build_direct_conv_nongrouped(spec, arch)
 * ===================================================================== */
rocke_kernel_def_t* rocke_build_direct_conv_nongrouped(
    rocke_ir_builder_t* b, const rocke_direct_conv_nongrouped_spec_t* spec, const char* arch)
{
    nongrouped_ctx c = {};
    std::vector<rocke_value_t*> accs_out;

    if(b == NULL || spec == NULL)
    {
        return NULL;
    }
    if(arch == NULL)
    {
        arch = "gfx950";
    }
    c.b = b;
    c.spec = spec;

    if(!nongrouped_prologue(c, arch))
    {
        return NULL;
    }
    nongrouped_grid_decode(c);
    nongrouped_x_pass_meta(c);
    nongrouped_w_pass_meta(c);
    nongrouped_read_bases(c);
    accs_out = nongrouped_channel_loop(c);
    if((int)accs_out.size() != c.MT_W * c.NT_W || !rocke_ir_builder_ok(b))
    {
        return NULL;
    }
    nongrouped_epilogue(c, accs_out);
    return rocke_ir_builder_ok(b) ? b->kernel : NULL;
}

rocke_kernel_def_t* rocke_build_direct_conv_nongrouped_new(
    rocke_ir_builder_t* b, const rocke_direct_conv_nongrouped_spec_t* spec, const char* arch)
{
    if(b != NULL)
    {
        /* Zero first, so every early return below leaves a builder the caller
         * can safely query and rocke_ir_builder_free(). */
        memset(b, 0, sizeof(*b));
    }
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        char name[512];
        if(b == NULL)
        {
            return NULL;
        }
        if(spec == NULL)
        {
            rocke_i_set_err_msg(b, ROCKE_ERR_VALUE, "build_direct_conv_nongrouped: null spec");
            return NULL;
        }
        if(rocke_direct_conv_nongrouped_kernel_name(spec, name, sizeof(name)) != ROCKE_OK)
        {
            rocke_i_set_err_msg(
                b, ROCKE_ERR_VALUE, "build_direct_conv_nongrouped: kernel name does not fit");
            return NULL;
        }
        if(rocke_ir_builder_init(b, name) != ROCKE_OK)
        {
            return NULL;
        }
        return rocke_build_direct_conv_nongrouped(b, spec, arch);
    });
}

rocke_status_t
    rocke_direct_conv_nongrouped_lower_to_llvm(const rocke_direct_conv_nongrouped_spec_t* spec,
                                               const char* arch,
                                               rocke_llvm_flavor_t flavor,
                                               char** out_ll,
                                               char* err,
                                               size_t err_cap)
{
    rocke_ir_builder_t b = {};
    rocke_kernel_def_t* kernel;
    rocke_status_t st;

    if(out_ll != NULL)
    {
        *out_ll = NULL;
    }
    if(spec == NULL || out_ll == NULL)
    {
        nongrouped_set_err(err, err_cap, "lower_to_llvm: null spec/out");
        return ROCKE_ERR_VALUE;
    }
    if(arch == NULL)
    {
        arch = "gfx950";
    }
    kernel = rocke_build_direct_conv_nongrouped_new(&b, spec, arch);
    if(kernel == NULL)
    {
        const char* m = rocke_ir_builder_error(&b);
        st = rocke_ir_builder_status(&b);
        nongrouped_set_err(
            err, err_cap, (m != NULL && m[0] != '\0') ? m : "build_direct_conv_nongrouped failed");
        rocke_ir_builder_free(&b);
        return (st == ROCKE_OK) ? ROCKE_ERR_VALUE : st;
    }
    st = rocke_lower_kernel_to_llvm_ex(kernel, flavor, arch, out_ll, err, err_cap);
    rocke_ir_builder_free(&b);
    return st;
}

/* ===================================================================== *
 *  Split-C reduction  (Python DirectSplitReduceSpec /
 *  build_direct_conv_split_reduce)
 * ===================================================================== */

static const char* split_reduce_dtype(const rocke_direct_conv_split_reduce_spec_t* spec)
{
    return spec->dtype ? spec->dtype : "fp16";
}

static bool split_reduce_is_bf16(const rocke_direct_conv_split_reduce_spec_t* spec)
{
    return strcmp(split_reduce_dtype(spec), "bf16") == 0;
}

rocke_direct_conv_split_reduce_spec_t rocke_direct_conv_split_reduce_spec_default(void)
{
    rocke_direct_conv_split_reduce_spec_t spec;
    memset(&spec, 0, sizeof(spec));
    spec.dtype = "fp16";
    spec.split_c = 2;
    spec.threads = 256;
    spec.name = "direct_conv_split_reduce";
    return spec;
}

rocke_status_t rocke_direct_conv_split_reduce_kernel_name(
    const rocke_direct_conv_split_reduce_spec_t* spec, char* out, size_t out_cap)
{
    char sc_buf[24];
    char b_buf[24];
    const char* parts[3];

    if(spec == NULL || out == NULL || out_cap == 0)
    {
        return ROCKE_ERR_VALUE;
    }
    snprintf(sc_buf, sizeof(sc_buf), "sc%d", spec->split_c);
    snprintf(b_buf, sizeof(b_buf), "b%d", spec->threads);
    parts[0] = sc_buf;
    parts[1] = b_buf;
    parts[2] = split_reduce_is_bf16(spec) ? "bf16" : "";
    return rocke_kernel_name_join(spec->name, parts, 3, NULL, NULL, 0, out, out_cap, NULL);
}

rocke_status_t rocke_direct_conv_split_reduce_validate(
    const rocke_direct_conv_split_reduce_spec_t* spec, char* reason, size_t reason_cap)
{
    char msg[ROCKE_ERR_MSG_CAP];

    if(spec == NULL)
    {
        nongrouped_set_reason(reason, reason_cap, "null spec");
        return ROCKE_ERR_VALUE;
    }
    if(strcmp(split_reduce_dtype(spec), "fp16") != 0
       && strcmp(split_reduce_dtype(spec), "bf16") != 0)
    {
        snprintf(msg,
                 sizeof(msg),
                 "DirectSplitReduceSpec: unsupported dtype '%s'",
                 split_reduce_dtype(spec));
        nongrouped_set_reason(reason, reason_cap, msg);
        return ROCKE_ERR_VALUE;
    }
    if(spec->split_c < 2)
    {
        snprintf(msg, sizeof(msg), "split_c must be >= 2 (got %d)", spec->split_c);
        nongrouped_set_reason(reason, reason_cap, msg);
        return ROCKE_ERR_VALUE;
    }
    if(spec->threads <= 0 || spec->threads % 64 != 0 || spec->threads > 1024)
    {
        snprintf(msg,
                 sizeof(msg),
                 "threads must be a positive multiple of 64 up to 1024 (got %d)",
                 spec->threads);
        nongrouped_set_reason(reason, reason_cap, msg);
        return ROCKE_ERR_VALUE;
    }
    nongrouped_set_reason(reason, reason_cap, "ok");
    return ROCKE_OK;
}

void rocke_direct_conv_split_reduce_grid(const rocke_direct_conv_split_reduce_spec_t* spec,
                                         const rocke_direct_conv_problem_t* problem,
                                         int grid[3])
{
    long long per_block = (long long)spec->threads * ROCKE_DCONV_SPLIT_REDUCE_VEC;
    long long total = (long long)problem->N * nongrouped_ho(problem) * nongrouped_wo(problem)
                      * ((long long)problem->groups * problem->kpg);
    grid[0] = (int)((total + per_block - 1) / per_block);
    grid[1] = 1;
    grid[2] = 1;
}

rocke_kernel_def_t* rocke_build_direct_conv_split_reduce(
    rocke_ir_builder_t* b, const rocke_direct_conv_split_reduce_spec_t* spec, const char* arch)
{
    char reason[ROCKE_ERR_MSG_CAP];
    const rocke_type_t* io_type;
    rocke_dconv_params_t params;
    rocke_value_t *K, *c0, *oob_base, *ws_rsrc, *d_rsrc, *k, *t, *gid, *elem, *hw, *total, *valid;
    rocke_value_t *slab_bytes, *elem_bytes, *ws_off, *ws_off_hi, *sum_lo = NULL, *sum_hi = NULL;
    rocke_value_t *pix, *kk, *n, *rem, *ho, *wo, *t0, *t1, *d_elems, *d_off, *out;
    bool bf16;
    int s;

    /* The kernel uses no arch-specific instruction; arch is accepted for the
     * builder-table signature. */
    (void)arch;
    if(b == NULL || spec == NULL)
    {
        return NULL;
    }
    if(rocke_direct_conv_split_reduce_validate(spec, reason, sizeof reason) != ROCKE_OK)
    {
        if(b->status == ROCKE_OK)
        {
            b->status = ROCKE_ERR_VALUE;
            snprintf(b->err, sizeof b->err, "%.*s", (int)(sizeof b->err - 1), reason);
        }
        return NULL;
    }
    io_type = rocke_b_io_ir_type(b, split_reduce_dtype(spec));
    if(io_type == NULL)
    {
        return NULL;
    }
    bf16 = split_reduce_is_bf16(spec);

    rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", spec->threads);
    rocke_dconv_emit_params(b, &params, "fwd", io_type);
    if(!rocke_ir_builder_ok(b))
    {
        return NULL;
    }
    K = params.p_total_k;

    c0 = rocke_b_const_i32(b, 0);
    oob_base = rocke_b_const_i32(b, ROCKE_DCONV_NONGROUPED_OOB_BASE);
    ws_rsrc = rocke_b_buffer_rsrc(b, params.A, params.A_bytes);
    d_rsrc = rocke_b_buffer_rsrc(b, params.D, params.D_bytes);

    t = rocke_b_block_id_x(b);
    k = rocke_b_const_i32(b, spec->threads);
    t = rocke_b_mul(b, t, k);
    t0 = rocke_b_thread_id_x(b);
    gid = rocke_b_add(b, t, t0);
    k = rocke_b_const_i32(b, ROCKE_DCONV_SPLIT_REDUCE_VEC);
    elem = rocke_b_mul(b, gid, k);
    hw = rocke_b_mul(b, params.p_Ho, params.p_Wo);
    t = rocke_b_mul(b, params.p_N, hw);
    total = rocke_b_mul(b, t, K);
    valid = rocke_b_cmp_lt(b, elem, total);

    /* The fp32 slabs are packed; elem indexes one of them directly. */
    k = rocke_b_const_i32(b, 4);
    slab_bytes = rocke_b_mul(b, total, k);
    k = rocke_b_const_i32(b, 4);
    elem_bytes = rocke_b_mul(b, elem, k);
    ws_off = rocke_b_select(b, valid, elem_bytes, oob_base);
    k = rocke_b_const_i32(b, 16);
    t = rocke_b_add(b, elem_bytes, k);
    ws_off_hi = rocke_b_select(b, valid, t, oob_base);
    /* The slab offset goes in soffset, summed by the hardware after the
     * select: added to OOB_BASE in IR it would pass INT32_MAX once a slab
     * nears 16 MiB. All loads first, then the adds: the slabs' loads are
     * independent, so issuing them back to back overlaps their latencies. */
    {
        std::vector<rocke_value_t*> los, his;
        for(s = 0; s < spec->split_c; ++s)
        {
            rocke_value_t* soff = c0;
            if(s != 0)
            {
                k = rocke_b_const_i32(b, s);
                soff = rocke_b_mul(b, slab_bytes, k);
            }
            los.push_back(rocke_b_buffer_load_vN(b, ws_rsrc, ws_off, soff, rocke_f32(), 4));
            his.push_back(rocke_b_buffer_load_vN(b, ws_rsrc, ws_off_hi, soff, rocke_f32(), 4));
        }
        sum_lo = los[0];
        sum_hi = his[0];
        for(s = 1; s < spec->split_c; ++s)
        {
            sum_lo = rocke_b_fadd(b, sum_lo, los[(size_t)s]);
            sum_hi = rocke_b_fadd(b, sum_hi, his[(size_t)s]);
        }
    }

    /* Packed element -> (n, ho, wo, k) -> strided output offset. */
    pix = rocke_b_div(b, elem, K);
    kk = rocke_b_mod(b, elem, K);
    n = rocke_b_div(b, pix, hw);
    rem = rocke_b_mod(b, pix, hw);
    ho = rocke_b_div(b, rem, params.p_Wo);
    wo = rocke_b_mod(b, rem, params.p_Wo);
    t0 = rocke_b_mul(b, n, params.p_D_stride_n);
    t1 = rocke_b_mul(b, ho, params.p_D_stride_ho);
    t0 = rocke_b_add(b, t0, t1);
    t1 = rocke_b_mul(b, wo, params.p_D_stride_wo);
    t1 = rocke_b_add(b, t1, kk);
    d_elems = rocke_b_add(b, t0, t1);
    k = rocke_b_const_i32(b, 2);
    t = rocke_b_mul(b, d_elems, k);
    d_off = rocke_b_select(b, valid, t, oob_base);
    t = rocke_b_vec_concat(b, sum_lo, sum_hi);
    if(bf16)
    {
        out = rocke_b_vec_trunc_f32_to_bf16(b, t);
        rocke_b_buffer_store_vN_bf16(b, d_rsrc, d_off, c0, out, 4);
    }
    else
    {
        out = rocke_b_vec_trunc_f32_to_f16(b, t);
        rocke_b_buffer_store_vN_f16(b, d_rsrc, d_off, c0, out, 4);
    }
    return rocke_ir_builder_ok(b) ? b->kernel : NULL;
}

rocke_kernel_def_t* rocke_build_direct_conv_split_reduce_new(
    rocke_ir_builder_t* b, const rocke_direct_conv_split_reduce_spec_t* spec, const char* arch)
{
    if(b != NULL)
    {
        /* Zero first, so every early return below leaves a builder the caller
         * can safely query and rocke_ir_builder_free(). */
        memset(b, 0, sizeof(*b));
    }
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        char name[512];
        if(b == NULL)
        {
            return NULL;
        }
        if(spec == NULL)
        {
            rocke_i_set_err_msg(b, ROCKE_ERR_VALUE, "build_direct_conv_split_reduce: null spec");
            return NULL;
        }
        if(rocke_direct_conv_split_reduce_kernel_name(spec, name, sizeof(name)) != ROCKE_OK)
        {
            rocke_i_set_err_msg(
                b, ROCKE_ERR_VALUE, "build_direct_conv_split_reduce: kernel name does not fit");
            return NULL;
        }
        if(rocke_ir_builder_init(b, name) != ROCKE_OK)
        {
            return NULL;
        }
        return rocke_build_direct_conv_split_reduce(b, spec, arch);
    });
}

rocke_status_t
    rocke_direct_conv_split_reduce_lower_to_llvm(const rocke_direct_conv_split_reduce_spec_t* spec,
                                                 const char* arch,
                                                 rocke_llvm_flavor_t flavor,
                                                 char** out_ll,
                                                 char* err,
                                                 size_t err_cap)
{
    rocke_ir_builder_t b = {};
    rocke_kernel_def_t* kernel;
    rocke_status_t st;

    if(out_ll != NULL)
    {
        *out_ll = NULL;
    }
    if(spec == NULL || out_ll == NULL)
    {
        nongrouped_set_err(err, err_cap, "lower_to_llvm: null spec/out");
        return ROCKE_ERR_VALUE;
    }
    if(arch == NULL)
    {
        arch = "gfx950";
    }
    kernel = rocke_build_direct_conv_split_reduce_new(&b, spec, arch);
    if(kernel == NULL)
    {
        const char* m = rocke_ir_builder_error(&b);
        st = rocke_ir_builder_status(&b);
        nongrouped_set_err(err,
                           err_cap,
                           (m != NULL && m[0] != '\0') ? m
                                                       : "build_direct_conv_split_reduce failed");
        rocke_ir_builder_free(&b);
        return (st == ROCKE_OK) ? ROCKE_ERR_VALUE : st;
    }
    st = rocke_lower_kernel_to_llvm_ex(kernel, flavor, arch, out_ll, err, err_cap);
    rocke_ir_builder_free(&b);
    return st;
}
