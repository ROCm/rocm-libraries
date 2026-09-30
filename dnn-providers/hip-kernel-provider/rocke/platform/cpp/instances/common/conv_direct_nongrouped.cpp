// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * conv_direct_nongrouped.cpp -- C++ engine port of build_direct_conv_nongrouped
 * (library/kernels/common/conv_direct_nongrouped.py): the non-grouped (groups == 1)
 * NHWC direct convolution with an LDS-staged, halo-inclusive activation tile.
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
#include "rocke/helper_rocke.helpers.grid.h" /* rocke_chiplet_aware_super_tile */
#include "rocke/helper_rocke.helpers.io.h" /* rocke_b_io_ir_type */
#include "rocke/helper_rocke.helpers.spec.h" /* rocke_kernel_name_join */
#include "rocke/lower_llvm.h"

/* Global load width for the activation staging path, in halves (dwordx4). */
#define ROCKE_DCONV_NONGROUPED_X_LOAD_VEC 8

/* Canonical i32 byte offset for a masked-out staging load; see the Python
 * `_OOB_BASE` comment. Loop-invariant predication is folded into the base
 * offset, and a buffer load past num_records returns zero. */
#define ROCKE_DCONV_NONGROUPED_OOB_BASE 0x7F000000

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
    return (p->H + 2 * p->PAD - p->KH) / p->stride + 1;
}

static int nongrouped_wo(const rocke_direct_conv_problem_t* p)
{
    return (p->W + 2 * p->PAD - p->KW) / p->stride + 1;
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
    lds_in_h = (long)(spec->tile_h - 1) * p->stride + p->KH;
    lds_in_w = (long)(spec->tile_w - 1) * p->stride + p->KW;
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

static void nongrouped_tile_counts(const rocke_direct_conv_nongrouped_spec_t* spec,
                                   int* n_wt,
                                   int* n_ht,
                                   int* n_kt)
{
    const rocke_direct_conv_problem_t* p = &spec->problem;
    *n_wt = (nongrouped_wo(p) + spec->tile_w - 1) / spec->tile_w;
    *n_ht = (nongrouped_ho(p) + spec->tile_h - 1) / spec->tile_h;
    *n_kt = (p->kpg + spec->tile_k - 1) / spec->tile_k;
}

void rocke_direct_conv_nongrouped_grid(const rocke_direct_conv_nongrouped_spec_t* spec, int grid[3])
{
    int n_wt, n_ht, n_kt;
    nongrouped_tile_counts(spec, &n_wt, &n_ht, &n_kt);
    grid[0] = n_wt * n_ht * spec->problem.N * n_kt;
    grid[1] = 1;
    grid[2] = 1;
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
    const char* parts[10];

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
    parts[0] = prob_short;
    parts[1] = t_buf;
    parts[2] = ck_buf;
    parts[3] = w_buf;
    parts[4] = a_buf;
    parts[5] = g_buf;
    parts[6] = spec->double_buffer ? "db" : "";
    parts[7] = iglp_buf;
    parts[8] = we_buf;
    parts[9] = nongrouped_is_bf16(spec) ? "bf16" : "";
    return rocke_kernel_name_join(spec->name, parts, 10, NULL, NULL, 0, out, out_cap, NULL);
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
    int t;

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
    if(p->groups != 1)
    {
        NONGROUPED_REJECT("DirectNongroupedConvSpec is the groups==1 family (got groups=%d); "
                          "use conv_direct_grouped for grouped shapes",
                          p->groups);
    }
    at = nongrouped_atom(spec->atom);
    if(at == NULL)
    {
        NONGROUPED_REJECT("unknown atom '%s'; expected one of ['32x32x16', '32x32x8', '16x16x32', "
                          "'16x16x16']",
                          spec->atom ? spec->atom : "None");
    }
    /* Python raises ZeroDivisionError on these; reject them cleanly instead. */
    if(spec->waves_m <= 0 || spec->waves_n <= 0 || spec->wave_size <= 0 || spec->ck <= 0
       || spec->tile_h <= 0 || spec->tile_w <= 0 || spec->tile_k <= 0 || p->stride <= 0)
    {
        NONGROUPED_REJECT(
            "DirectNongroupedConvSpec: tile, wave and stride parameters must be positive");
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
    if(p->cpg % spec->ck != 0)
    {
        NONGROUPED_REJECT("C %d not divisible by ck %d", p->cpg, spec->ck);
    }
    if(p->kpg % t != 0)
    {
        NONGROUPED_REJECT("K %d must be a multiple of %d (got %d)", p->kpg, t, p->kpg);
    }
    if(p->cpg % ROCKE_DCONV_NONGROUPED_X_LOAD_VEC != 0)
    {
        NONGROUPED_REJECT(
            "C %d must be a multiple of %d", p->cpg, ROCKE_DCONV_NONGROUPED_X_LOAD_VEC);
    }
    if(rocke_direct_conv_nongrouped_threads_per_block(spec) > 1024)
    {
        NONGROUPED_REJECT("threads_per_block %d > 1024",
                          rocke_direct_conv_nongrouped_threads_per_block(spec));
    }
    if(spec->lds_pad % 2 != 0)
    {
        NONGROUPED_REJECT("lds_pad must be even to keep ds_read_b128 aligned");
    }
    if(spec->swizzle_wgm < 1)
    {
        NONGROUPED_REJECT("swizzle_wgm must be >= 1 (got %d)", spec->swizzle_wgm);
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
        snprintf(msg, sizeof(msg), "unknown gfx target '%s'", arch);
        nongrouped_set_reason(reason, reason_cap, msg);
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

struct nongrouped_stage_meta
{
    rocke_value_t* base; /* global byte offset base (chunk offset added per iter) */
    rocke_value_t* lds_idx; /* LDS half index of this thread's slot                 */
};

struct nongrouped_ctx
{
    rocke_ir_builder_t* b;
    const rocke_direct_conv_nongrouped_spec_t* spec;
    rocke_direct_conv_problem_t p;
    const rocke_type_t* io_type;
    bool bf16;
    rocke_dconv_nongrouped_mfma_fn mfma;

    /* Geometry (Python all-caps locals). */
    int C, K, KH, KW, S, PAD, Ho, Wo;
    int TH, TW, TK, CK, THREADS, WAVE, FRAG, AK, AT, ACC, QUADS;
    int NCB, M_TILES, KATOMS, ROWS_W, MT_W, NT_W, CSTRIDE, LDS_IN_H, LDS_IN_W;
    int X_CV, X_VECS, X_PASSES, W_SLOTS, W_PASSES, N_CHUNKS;
    int x_stage, w_stage, x_dump, w_dump;
    bool DB;

    /* SSA values. */
    rocke_value_t *A, *Bp, *D, *A_bytes, *B_bytes, *D_bytes;
    rocke_value_t *c0, *c1, *c_half, *oob, *oob_base;
    rocke_value_t *a_rsrc, *b_rsrc, *d_rsrc;
    rocke_value_t *X_smem, *W_smem;
    rocke_value_t *tid, *lane, *wave_id, *wave_m, *wave_n, *lane_lo, *lane_hi;
    rocke_value_t *k_tile, *cell, *n_img, *out_h0, *out_w0, *k_base, *in_h0, *in_w0;
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

    c.C = c.p.cpg;
    c.K = c.p.kpg;
    c.KH = c.p.KH;
    c.KW = c.p.KW;
    c.S = c.p.stride;
    c.PAD = c.p.PAD;
    c.Ho = nongrouped_ho(&c.p);
    c.Wo = nongrouped_wo(&c.p);

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
    c.LDS_IN_H = (c.TH - 1) * c.S + c.KH;
    c.LDS_IN_W = (c.TW - 1) * c.S + c.KW;

    c.X_CV = c.CK / ROCKE_DCONV_NONGROUPED_X_LOAD_VEC;
    c.X_VECS = c.LDS_IN_H * c.LDS_IN_W * c.X_CV;
    c.X_PASSES = (c.X_VECS + c.THREADS - 1) / c.THREADS;
    c.W_SLOTS = c.KH * c.KW * c.M_TILES * c.KATOMS * c.WAVE;
    c.W_PASSES = (c.W_SLOTS + c.THREADS - 1) / c.THREADS;
    c.N_CHUNKS = c.C / c.CK;

    rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", c.THREADS);
    if(spec->waves_per_eu != ROCKE_DCONV_NONGROUPED_WAVES_PER_EU_NONE)
    {
        rocke_attr_set_int(b, &b->kernel->attrs, "waves_per_eu", spec->waves_per_eu);
    }

    {
        const rocke_type_t* io_ptr = rocke_ptr_type(b, c.io_type, "global");
        rocke_param_opts_t ro = {};
        rocke_param_opts_t wo = {};
        rocke_param_opts_t none = {};

        ro.noalias = true;
        ro.noalias_set = true;
        ro.readonly = true;
        ro.readonly_set = true;
        ro.align = 16;
        ro.align_set = true;
        c.A = rocke_b_param(b, "A", io_ptr, &ro);
        c.Bp = rocke_b_param(b, "B", io_ptr, &ro);

        wo.noalias = true;
        wo.noalias_set = true;
        wo.writeonly = true;
        wo.writeonly_set = true;
        wo.align = 16;
        wo.align_set = true;
        c.D = rocke_b_param(b, "D", io_ptr, &wo);

        c.A_bytes = rocke_b_param(b, "A_bytes", rocke_i32(), &none);
        c.B_bytes = rocke_b_param(b, "B_bytes", rocke_i32(), &none);
        c.D_bytes = rocke_b_param(b, "D_bytes", rocke_i32(), &none);
    }

    c.c0 = nongrouped_c(c, 0);
    c.c1 = nongrouped_c(c, 1);
    c.c_half = nongrouped_c(c, 2);
    c.oob = nongrouped_c(c, ((int64_t)1 << 31) - 1);
    c.oob_base = nongrouped_c(c, ROCKE_DCONV_NONGROUPED_OOB_BASE);

    c.a_rsrc = rocke_b_buffer_rsrc(b, c.A, c.A_bytes);
    c.b_rsrc = rocke_b_buffer_rsrc(b, c.Bp, c.B_bytes);
    c.d_rsrc = rocke_b_buffer_rsrc(b, c.D, c.D_bytes);

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

/* ---- grid decode: flat 1-D grid of (spatial cell x channel tile) ---- */
void nongrouped_grid_decode(nongrouped_ctx& c)
{
    rocke_ir_builder_t* b = c.b;
    const rocke_direct_conv_nongrouped_spec_t* spec = c.spec;
    int n_wt, n_ht, n_kt, n_cells;
    rocke_value_t *wgid, *k, *st, *w_tile, *h_tile, *t;

    nongrouped_tile_counts(spec, &n_wt, &n_ht, &n_kt);
    n_cells = n_wt * n_ht * c.p.N;

    wgid = rocke_b_block_id_x(b);
    if(spec->chiplet_swizzle)
    {
        /* Python: chiplet_transform_chunked(num_wgs=n_cells*n_kt) followed by
         * super_tile_swizzle(num_pid_m=n_kt, num_pid_n=n_cells) -- exactly the
         * compile-time chiplet_aware_super_tile composition. */
        rocke_super_tile_swizzle_result_t sw
            = rocke_chiplet_aware_super_tile(b,
                                             wgid,
                                             n_kt,
                                             n_cells,
                                             spec->swizzle_wgm < n_kt ? spec->swizzle_wgm : n_kt,
                                             spec->num_xcds,
                                             spec->chiplet_chunk);
        c.k_tile = sw.row;
        c.cell = sw.col;
    }
    else
    {
        k = nongrouped_c(c, n_kt);
        c.k_tile = rocke_b_mod(b, wgid, k);
        k = nongrouped_c(c, n_kt);
        c.cell = rocke_b_div(b, wgid, k);
    }

    k = nongrouped_c(c, n_wt * n_ht);
    c.n_img = rocke_b_div(b, c.cell, k);
    k = nongrouped_c(c, n_wt * n_ht);
    st = rocke_b_mod(b, c.cell, k);
    k = nongrouped_c(c, n_wt);
    w_tile = rocke_b_mod(b, st, k);
    k = nongrouped_c(c, n_wt);
    h_tile = rocke_b_div(b, st, k);

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

    k = nongrouped_c(c, (int64_t)c.p.H * c.p.W * c.C);
    img_base = rocke_b_mul(b, c.n_img, k);
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
        k = nongrouped_c(c, c.p.H);
        lt = rocke_b_cmp_lt(b, ih, k);
        ok_h = rocke_b_land(b, ge, lt);
        ge = rocke_b_cmp_ge(b, iw, c.c0);
        k = nongrouped_c(c, c.p.W);
        lt = rocke_b_cmp_lt(b, iw, k);
        ok_w = rocke_b_land(b, ge, lt);
        valid = rocke_b_land(b, ok_h, ok_w);
        if(c.X_VECS % c.THREADS != 0)
        {
            k = nongrouped_c(c, c.X_VECS);
            in_tile = rocke_b_cmp_lt(b, v, k);
            valid = rocke_b_land(b, valid, in_tile);
        }

        /* base = ((n*H + ih)*W + iw)*C + cv*VEC; chunk offset added per iter. */
        k = nongrouped_c(c, c.p.W);
        t0 = rocke_b_mul(b, ih, k);
        t0 = rocke_b_add(b, t0, iw);
        k = nongrouped_c(c, c.C);
        t0 = rocke_b_mul(b, t0, k);
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
        c.x_meta.push_back({base, lds_idx});
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

        k = nongrouped_c(c, c.K);
        valid = rocke_b_cmp_lt(b, k_out, k);
        if(c.W_SLOTS % c.THREADS != 0)
        {
            k = nongrouped_c(c, c.W_SLOTS);
            in_tile = rocke_b_cmp_lt(b, g, k);
            valid = rocke_b_land(b, valid, in_tile);
        }

        /* base = ((k_out*KH + r)*KW + s)*C + c_in_chunk */
        k = nongrouped_c(c, c.KH);
        krs_h = rocke_b_mul(b, k_out, k);
        krs_hr = rocke_b_add(b, krs_h, r);
        k = nongrouped_c(c, c.KW);
        krs_w = rocke_b_mul(b, krs_hr, k);
        krs = rocke_b_add(b, krs_w, s);
        k = nongrouped_c(c, c.C);
        t0 = rocke_b_mul(b, krs, k);
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
        c.w_meta.push_back({base, lds_idx});
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
 * the KH taps of a column share ``(ROWS_W - 1) * stride + KH`` fragments. */
rocke_value_t* nongrouped_read_b_frag(
    nongrouped_ctx& c, rocke_value_t* base, int in_row, int cb, int s, int katom)
{
    int64_t pos = (int64_t)in_row * c.LDS_IN_W + (int64_t)cb * c.AT * c.S + s;
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
    const int n_in_rows = (c.ROWS_W - 1) * c.S + c.KH;
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
                    int bf_row = row * c.S + r_c;
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
    std::vector<rocke_value_t*> xs0, ws0;
    std::vector<rocke_iter_arg_t> iter_args;
    std::vector<std::string> names;
    std::vector<rocke_value_t*> out;
    rocke_for_t loop;
    int i;

    zero_acc = rocke_b_zero_vec_f32(b, c.ACC);

    c_ck_bytes = nongrouped_c(c, (int64_t)c.CK * 2);
    nongrouped_issue_stage_loads(c, c.c0, xs0, ws0);
    if(c.DB)
    {
        /* Prologue: publish chunk 0 into buffer 0 and have chunk 1 in flight. */
        nongrouped_commit_stage(c, xs0, ws0, NULL, NULL);
        nongrouped_issue_stage_loads(c, c_ck_bytes, xs0, ws0);
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

    hi = nongrouped_c(c, c.N_CHUNKS);
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
    rocke_value_t *k, *t0, *t1, *k_lane_base, *row_base, *img_out_base;
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
    k = nongrouped_c(c, (int64_t)c.Ho * c.Wo * c.K);
    img_out_base = rocke_b_mul(b, c.n_img, k);

    for(mt = 0; mt < c.MT_W; ++mt)
    {
        rocke_value_t* k_mt;
        k = nongrouped_c(c, (int64_t)mt * c.AT);
        k_mt = rocke_b_add(b, k_lane_base, k);
        for(row = 0; row < c.ROWS_W; ++row)
        {
            rocke_value_t *out_h, *h_ok;
            k = nongrouped_c(c, row);
            out_h = rocke_b_add(b, row_base, k);
            k = nongrouped_c(c, c.Ho);
            h_ok = rocke_b_cmp_lt(b, out_h, k);
            for(cb = 0; cb < c.NCB; ++cb)
            {
                int nt = row * c.NCB + cb;
                rocke_value_t* acc = accs_out[(size_t)mt * c.NT_W + nt];
                rocke_value_t *out_w, *hw_ok, *pix, *base_off;

                k = nongrouped_c(c, (int64_t)cb * c.AT);
                t0 = rocke_b_add(b, c.out_w0, k);
                out_w = rocke_b_add(b, t0, c.lane_lo);
                k = nongrouped_c(c, c.Wo);
                t0 = rocke_b_cmp_lt(b, out_w, k);
                hw_ok = rocke_b_land(b, h_ok, t0);
                k = nongrouped_c(c, c.Wo);
                t0 = rocke_b_mul(b, out_h, k);
                pix = rocke_b_add(b, t0, out_w);
                k = nongrouped_c(c, c.K);
                t0 = rocke_b_mul(b, pix, k);
                base_off = rocke_b_add(b, img_out_base, t0);
                for(q = 0; q < c.QUADS; ++q)
                {
                    rocke_value_t *k_out, *valid, *d_off, *safe, *quad, *narrow;
                    rocke_value_t* comps[4];

                    k = nongrouped_c(c, (int64_t)q * (c.AT / 4));
                    k_out = rocke_b_add(b, k_mt, k);
                    k = nongrouped_c(c, c.K);
                    t0 = rocke_b_cmp_lt(b, k_out, k);
                    valid = rocke_b_land(b, hw_ok, t0);
                    t0 = rocke_b_add(b, base_off, k_out);
                    d_off = rocke_b_mul(b, t0, c.c_half);
                    safe = rocke_b_select(b, valid, d_off, c.oob);
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
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        char name[512];
        if(b == NULL || spec == NULL)
        {
            return NULL;
        }
        if(rocke_direct_conv_nongrouped_kernel_name(spec, name, sizeof(name)) != ROCKE_OK)
        {
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
    rocke_ir_builder_t b;
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
