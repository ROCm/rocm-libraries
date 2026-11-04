// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * conv_direct_grouped_build_depthwise_dgrad_win.cpp -- C++ port of the
 * windowed depthwise dgrad kernel (library/kernels/common/conv_direct_grouped.py):
 *
 *   Python                                          C++ (this file)
 *   DirectDepthwiseDgradWindowedSpec             -> rocke_direct_depthwise_dgrad_win_spec_t
 *   _depthwise_dgrad_win_check / is_valid_...    -> rocke_direct_depthwise_dgrad_win_is_valid_spec()
 *   kernel_name()                                -> rocke_direct_depthwise_dgrad_win_kernel_name()
 *   build_direct_depthwise_dgrad_windowed(...)   -> rocke_build_direct_depthwise_dgrad_win_new()
 *   _build_dw_dgrad_win_mfma(...)                -> dw_win_build_mfma()
 *
 * Stride 1, cpg = kpg = 1. The kernel takes the direct dgrad kernarg block
 * (rocke_dconv_emit_params(direction "dgrad")): N, H, W, Ho, Wo, groups and the
 * dY/dX strides are runtime; KH, KW and PAD are baked capabilities, the tile
 * and the forms (dot2, mfma) are knobs.
 *
 * Every builder call with side effects is issued as its own statement in
 * Python's left-to-right evaluation order, so the IR numbering matches the
 * Python engine byte for byte.
 */
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <string>
#include <vector>

#include "rocke/arch_target.h" /* rocke_set_unknown_arch_reason */
#include "rocke/error_boundary.hpp" /* ckc::guard_builder boundary shim */
#include "rocke/helper_rocke.core.arch.h" /* rocke_archtarget_from_gfx */
#include "rocke/helper_rocke.helpers.io.h"
#include "rocke/instance_conv_direct_grouped.h"
#include "rocke/instance_conv_direct_grouped_internal.h"
#include "rocke/ir.h"
#include "rocke/ir_internal.h" /* rocke_i_set_err */

typedef rocke_value_t V;

/* ===================================================================== *
 *  Spec helpers
 * ===================================================================== */
rocke_direct_depthwise_dgrad_win_spec_t rocke_direct_depthwise_dgrad_win_spec_default(void)
{
    rocke_direct_depthwise_dgrad_win_spec_t s;
    memset(&s, 0, sizeof s);
    s.problem = rocke_direct_conv_problem_default();
    s.name = "direct_depthwise_dgrad_win";
    s.block_w = 8;
    s.block_waves = 1;
    s.ch_per_lane = 1;
    s.block_h = 8;
    s.dot2 = false;
    s.mfma = false;
    s.w_fold = 1;
    s.prefetch_rows = 1;
    s.wave_size = 64;
    return s;
}

int rocke_direct_depthwise_dgrad_win_threads_per_block(
    const rocke_direct_depthwise_dgrad_win_spec_t* spec)
{
    return spec->block_waves * spec->wave_size;
}

int rocke_direct_depthwise_dgrad_win_block_ch(const rocke_direct_depthwise_dgrad_win_spec_t* spec)
{
    if(spec->mfma)
        return spec->block_waves * ROCKE_DW_DGRAD_MFMA_CH;
    return rocke_direct_depthwise_dgrad_win_threads_per_block(spec) * spec->ch_per_lane;
}

int rocke_direct_depthwise_dgrad_win_tile_w(const rocke_direct_depthwise_dgrad_win_spec_t* spec)
{
    return spec->mfma ? 32 / spec->w_fold : spec->block_w;
}

static int dw_win_passes(int KW)
{
    return (KW + 4) / 4;
}

static int dw_win_cdiv(int a, int b)
{
    return (a + b - 1) / b;
}

long long rocke_direct_depthwise_dgrad_win_unrolled_fmas(
    const rocke_direct_depthwise_dgrad_win_spec_t* spec)
{
    const rocke_direct_conv_problem_t* p = &spec->problem;
    return (long long)spec->block_h * p->KH * p->KW * spec->block_w * spec->ch_per_lane;
}

long long rocke_direct_depthwise_dgrad_win_unrolled_mfmas(
    const rocke_direct_depthwise_dgrad_win_spec_t* spec)
{
    const rocke_direct_conv_problem_t* p = &spec->problem;
    return (long long)spec->block_h * p->KH * dw_win_passes(p->KW);
}

static int dw_win_mfma_in_passes(const rocke_direct_depthwise_dgrad_win_spec_t* spec)
{
    const int win
        = rocke_direct_depthwise_dgrad_win_tile_w(spec) + 4 * dw_win_passes(spec->problem.KW) - 2;
    const int per_px = rocke_direct_depthwise_dgrad_win_block_ch(spec) / ROCKE_DW_DGRAD_MFMA_CH;
    return dw_win_cdiv(spec->w_fold * win * per_px,
                       rocke_direct_depthwise_dgrad_win_threads_per_block(spec));
}

int rocke_direct_depthwise_dgrad_win_mfma_frag_vgprs(
    const rocke_direct_depthwise_dgrad_win_spec_t* spec)
{
    const rocke_direct_conv_problem_t* p = &spec->problem;
    const int kwp = dw_win_passes(p->KW);
    const int rows = spec->prefetch_rows * dw_win_mfma_in_passes(spec);
    return 4 * (p->KH * kwp + p->KH + kwp + rows);
}

int rocke_direct_depthwise_dgrad_win_mfma_lds_bytes(
    const rocke_direct_depthwise_dgrad_win_spec_t* spec)
{
    const int block_ch = rocke_direct_depthwise_dgrad_win_block_ch(spec);
    const int per_px = block_ch / ROCKE_DW_DGRAD_MFMA_CH;
    const int threads = rocke_direct_depthwise_dgrad_win_threads_per_block(spec);
    const int in_px = dw_win_cdiv(dw_win_mfma_in_passes(spec) * threads, per_px);
    const int out_px = spec->w_fold * rocke_direct_depthwise_dgrad_win_tile_w(spec);
    return 2 * (in_px + out_px) * (block_ch + ROCKE_DW_DGRAD_MFMA_CH) * 2;
}

static bool dw_win_is_bf16(const rocke_direct_conv_problem_t* p)
{
    return p->dtype != NULL && strcmp(p->dtype, "bf16") == 0;
}

rocke_status_t rocke_direct_depthwise_dgrad_win_kernel_name(
    const rocke_direct_depthwise_dgrad_win_spec_t* spec, char* out, size_t out_cap)
{
    const rocke_direct_conv_problem_t* p;
    const char* name;
    int n;
    if(spec == NULL || out == NULL || out_cap == 0)
        return ROCKE_ERR_VALUE;
    p = &spec->problem;
    name = spec->name != NULL ? spec->name : "direct_depthwise_dgrad_win";
    if(spec->mfma)
        n = snprintf(out,
                     out_cap,
                     "%s_r%ds%dp%d_wv%d_bh%d_f%d_pf%d_mfma%s",
                     name,
                     p->KH,
                     p->KW,
                     p->PAD,
                     spec->block_waves,
                     spec->block_h,
                     spec->w_fold,
                     spec->prefetch_rows,
                     dw_win_is_bf16(p) ? "_bf16" : "");
    else
        n = snprintf(out,
                     out_cap,
                     "%s_r%ds%dp%d_bw%d_wv%d_cpl%d_bh%d%s%s",
                     name,
                     p->KH,
                     p->KW,
                     p->PAD,
                     spec->block_w,
                     spec->block_waves,
                     spec->ch_per_lane,
                     spec->block_h,
                     spec->dot2 ? "_dot2" : "",
                     dw_win_is_bf16(p) ? "_bf16" : "");
    if(n < 0 || (size_t)n >= out_cap)
        return ROCKE_ERR_VALUE;
    for(char* c = out; *c; ++c)
        if(*c == '/')
            *c = '_';
    return ROCKE_OK;
}

/* _depthwise_dgrad_win_check: true when the spec is buildable for its problem. */
static bool
    dw_win_check(const rocke_direct_depthwise_dgrad_win_spec_t* spec, char* reason, size_t cap)
{
    const rocke_direct_conv_problem_t* p = &spec->problem;
    const char* dtype = p->dtype != NULL ? p->dtype : "fp16";
#define DW_WIN_FAIL(...)                        \
    do                                          \
    {                                           \
        if(reason != NULL && cap > 0)           \
            snprintf(reason, cap, __VA_ARGS__); \
        return false;                           \
    } while(0)
    if(strcmp(dtype, "fp16") != 0 && strcmp(dtype, "bf16") != 0)
        DW_WIN_FAIL("unsupported dtype '%s'; expected fp16 or bf16", dtype);
    if(p->cpg != 1 || p->kpg != 1)
        DW_WIN_FAIL("requires cpg=kpg=1 (got %d, %d)", p->cpg, p->kpg);
    if(p->stride != 1)
        DW_WIN_FAIL("requires stride=1 (got %d)", p->stride);
    if(p->dil_h != 1 || p->dil_w != 1)
        DW_WIN_FAIL("this direct conv kernel has no dilation (got %dx%d); only the non-grouped "
                    "forward kernel computes a dilated convolution",
                    p->dil_h,
                    p->dil_w);
    const int Ho = p->H + 2 * p->PAD - p->KH + 1;
    const int Wo = p->W + 2 * p->PAD - p->KW + 1;
    if(p->PAD < 0 || Ho <= 0 || Wo <= 0)
        DW_WIN_FAIL("degenerate geometry (PAD=%d, Ho=%d, Wo=%d)", p->PAD, Ho, Wo);
    if(spec->block_w < 1)
        DW_WIN_FAIL("block_w must be >= 1 (got %d)", spec->block_w);
    if(spec->block_h < 1)
        DW_WIN_FAIL("block_h must be >= 1 (got %d); the image height is a kernarg, so a block "
                    "covers a tile of rows",
                    spec->block_h);
    const int threads = rocke_direct_depthwise_dgrad_win_threads_per_block(spec);
    if(spec->block_waves < 1 || threads > 1024)
        DW_WIN_FAIL("block_waves must give 1..1024 threads (got %d)", spec->block_waves);
    const int cpl = spec->ch_per_lane;
    if(cpl != 1 && cpl != 2 && cpl != 4 && cpl != 8)
        DW_WIN_FAIL("ch_per_lane must be 1, 2, 4 or 8 (got %d)", cpl);
    if(p->groups % cpl != 0)
        DW_WIN_FAIL("groups=%d is not a multiple of ch_per_lane=%d", p->groups, cpl);
    if(spec->dot2 && cpl != 1)
        DW_WIN_FAIL("dot2 requires ch_per_lane=1 (got %d)", cpl);
    if(spec->prefetch_rows < 1)
        DW_WIN_FAIL("prefetch_rows must be >= 1 (got %d)", spec->prefetch_rows);
    if(spec->mfma)
    {
        if(spec->dot2 || cpl != 1)
            DW_WIN_FAIL("mfma excludes dot2 and ch_per_lane > 1");
        if(spec->w_fold != 1 && spec->w_fold != 2 && spec->w_fold != 4)
            DW_WIN_FAIL("w_fold must be 1, 2 or 4 (got %d)", spec->w_fold);
        if(spec->prefetch_rows > ROCKE_DW_DGRAD_MFMA_MAX_PREFETCH_ROWS)
            DW_WIN_FAIL("prefetch_rows must be <= %d (got %d)",
                        ROCKE_DW_DGRAD_MFMA_MAX_PREFETCH_ROWS,
                        spec->prefetch_rows);
        if(spec->block_waves > ROCKE_DW_DGRAD_MFMA_MAX_WAVES)
            DW_WIN_FAIL("mfma needs block_waves <= %d (got %d)",
                        ROCKE_DW_DGRAD_MFMA_MAX_WAVES,
                        spec->block_waves);
        if(p->groups % ROCKE_DW_DGRAD_MFMA_CH != 0)
            DW_WIN_FAIL("mfma needs groups %% %d == 0 (got %d)", ROCKE_DW_DGRAD_MFMA_CH, p->groups);
        const int frag = rocke_direct_depthwise_dgrad_win_mfma_frag_vgprs(spec);
        if(frag > ROCKE_DW_DGRAD_MFMA_MAX_FRAG_VGPRS)
            DW_WIN_FAIL("mfma fragments need %d VGPRs (> %d); lower prefetch_rows or the filter",
                        frag,
                        ROCKE_DW_DGRAD_MFMA_MAX_FRAG_VGPRS);
    }
    else if(spec->w_fold != 1 || spec->prefetch_rows != 1)
        DW_WIN_FAIL("w_fold and prefetch_rows require mfma (got w_fold=%d, prefetch_rows=%d)",
                    spec->w_fold,
                    spec->prefetch_rows);
    {
        const long long dy = (long long)p->N * Ho * Wo * p->groups * 2;
        const long long dx = (long long)p->N * p->H * p->W * p->groups * 2;
        const long long mx = dy > dx ? dy : dx;
        if(mx > ROCKE_DW_DGRAD_WIN_MAX_TENSOR_BYTES)
            DW_WIN_FAIL("dY/dX of %lld bytes exceed the %lld-byte sentinel addressing range",
                        mx,
                        (long long)ROCKE_DW_DGRAD_WIN_MAX_TENSOR_BYTES);
    }
    if(spec->mfma)
    {
        const long long u = rocke_direct_depthwise_dgrad_win_unrolled_mfmas(spec);
        if(u > ROCKE_DW_DGRAD_MFMA_MAX_UNROLL)
            DW_WIN_FAIL("unrolled body too large (%lld MFMAs > %d); lower block_h",
                        u,
                        ROCKE_DW_DGRAD_MFMA_MAX_UNROLL);
        const int lds = rocke_direct_depthwise_dgrad_win_mfma_lds_bytes(spec);
        if(lds > ROCKE_DW_DGRAD_MFMA_LDS_BUDGET)
            DW_WIN_FAIL("mfma stages %d LDS bytes per workgroup (budget %d); lower block_waves",
                        lds,
                        ROCKE_DW_DGRAD_MFMA_LDS_BUDGET);
    }
    else
    {
        const long long u = rocke_direct_depthwise_dgrad_win_unrolled_fmas(spec);
        if(u > ROCKE_DW_DGRAD_WIN_MAX_UNROLL)
            DW_WIN_FAIL("unrolled body too large (%lld FMAs > %d); lower block_w or block_h",
                        u,
                        ROCKE_DW_DGRAD_WIN_MAX_UNROLL);
    }
#undef DW_WIN_FAIL
    return true;
}

bool rocke_direct_depthwise_dgrad_win_is_valid_spec(
    const rocke_direct_depthwise_dgrad_win_spec_t* spec,
    const char* arch,
    char* reason,
    size_t reason_cap)
{
    if(spec == NULL)
    {
        if(reason != NULL && reason_cap > 0)
            snprintf(reason, reason_cap, "null spec");
        return false;
    }
    if(arch == NULL)
        arch = "gfx950";
    if(rocke_archtarget_from_gfx(arch) == NULL)
    {
        rocke_set_unknown_arch_reason(reason, reason_cap, arch);
        return false;
    }
    if(spec->dot2 && strcmp(arch, "gfx950") != 0)
    {
        if(reason != NULL && reason_cap > 0)
            snprintf(reason, reason_cap, "dot2 needs one of ('gfx950',) (got %s)", arch);
        return false;
    }
    if(spec->mfma && strcmp(arch, "gfx950") != 0)
    {
        if(reason != NULL && reason_cap > 0)
            snprintf(reason, reason_cap, "mfma needs one of ('gfx950',) (got %s)", arch);
        return false;
    }
    return dw_win_check(spec, reason, reason_cap);
}

/* ===================================================================== *
 *  Shared emission helpers
 * ===================================================================== */

/* _dw_win_byte_strides: (dy_img, dy_row, dy_px, dx_img, dx_row, dx_px). */
static void dw_win_byte_strides(rocke_ir_builder_t* b, const rocke_dconv_params_t* prm, V* out[6])
{
    V* c2 = rocke_b_const_i32(b, 2);
    out[0] = rocke_b_mul(b, prm->p_A_stride_n, c2); /* p_dY_stride_n  */
    out[1] = rocke_b_mul(b, prm->p_A_stride_hi, c2); /* p_dY_stride_ho */
    out[2] = rocke_b_mul(b, prm->p_A_stride_wi, c2); /* p_dY_stride_wo */
    out[3] = rocke_b_mul(b, prm->p_D_stride_n, c2); /* p_dX_stride_n  */
    out[4] = rocke_b_mul(b, prm->p_D_stride_ho, c2); /* p_dX_stride_hi */
    out[5] = rocke_b_mul(b, prm->p_D_stride_wo, c2); /* p_dX_stride_wi */
}

/* _dw_win_tile: (image_tile, h0) from block z and h_tiles = ceil(p_Hi / rows). */
static void dw_win_tile(rocke_ir_builder_t* b, V* p_Hi, int rows, V** n_tile, V** h0)
{
    V* bz = rocke_b_block_id_z(b);
    V* c_rows = rocke_b_const_i32(b, rows);
    V* c_rm1 = rocke_b_const_i32(b, rows - 1);
    V* sum = rocke_b_add(b, p_Hi, c_rm1);
    V* h_tiles = rocke_b_div(b, sum, c_rows);
    *n_tile = rocke_b_div(b, bz, h_tiles);
    V* m = rocke_b_mul(b, *n_tile, h_tiles);
    V* d = rocke_b_sub(b, bz, m);
    *h0 = rocke_b_mul(b, d, c_rows);
}

static V* dw_win_load_half(rocke_ir_builder_t* b, bool bf16, V* rsrc, V* off, V* c0)
{
    return bf16 ? rocke_b_buffer_load_bf16(b, rsrc, off, c0)
                : rocke_b_buffer_load_f16(b, rsrc, off, c0);
}

static V* dw_win_load_vN(rocke_ir_builder_t* b, bool bf16, V* rsrc, V* off, V* c0, int dwords)
{
    return bf16 ? rocke_b_buffer_load_vN_bf16(b, rsrc, off, c0, dwords)
                : rocke_b_buffer_load_vN_f16(b, rsrc, off, c0, dwords);
}

static void dw_win_store_vN(rocke_ir_builder_t* b, bool bf16, V* rsrc, V* off, V* c0, V* v, int dw)
{
    if(bf16)
        rocke_b_buffer_store_vN_bf16(b, rsrc, off, c0, v, dw);
    else
        rocke_b_buffer_store_vN_f16(b, rsrc, off, c0, v, dw);
}

static V* dw_win_trunc_vec(rocke_ir_builder_t* b, bool bf16, V* v)
{
    return bf16 ? rocke_b_vec_trunc_f32_to_bf16(b, v) : rocke_b_vec_trunc_f32_to_f16(b, v);
}

/* A one-trip (or zero-trip) loop guarding `body` on the block-uniform `ok`,
 * carrying `accs` (updated in place to the loop results). */
template <typename Body>
static void dw_win_guarded(rocke_ir_builder_t* b,
                           V* ok,
                           V* c0,
                           V* c1,
                           std::vector<V*>& accs,
                           const char* prefix,
                           int y,
                           bool elide_trailing_barrier,
                           Body body)
{
    V* trip = rocke_b_select(b, ok, c1, c0);
    std::vector<std::string> names(accs.size());
    std::vector<rocke_iter_arg_t> iargs(accs.size());
    for(size_t i = 0; i < accs.size(); ++i)
    {
        names[i] = std::string(prefix) + "_y" + std::to_string(y) + "_" + std::to_string(i);
        iargs[i].name = names[i].c_str();
        iargs[i].init = accs[i];
    }
    const std::string iv = std::string(prefix) + "_y" + std::to_string(y);
    rocke_for_t loop = rocke_b_scf_for_iter(b,
                                            c0,
                                            trip,
                                            c1,
                                            iargs.data(),
                                            (int)iargs.size(),
                                            iv.c_str(),
                                            /*unroll=*/false,
                                            elide_trailing_barrier);
    if(!rocke_ir_builder_ok(b))
        return;
    rocke_b_region_enter(b, loop.body);
    std::vector<V*> carried(loop.iter_vars, loop.iter_vars + loop.num_iter_vars);
    std::vector<V*> out = body(carried);
    rocke_b_scf_yield(b, out.data(), (int)out.size());
    rocke_b_region_leave(b);
    for(size_t i = 0; i < accs.size(); ++i)
        accs[i] = loop.op->results[i];
}

/* ===================================================================== *
 *  VALU / dot2 form
 * ===================================================================== */
static rocke_kernel_def_t* dw_win_build_valu(rocke_ir_builder_t* b,
                                             const rocke_direct_depthwise_dgrad_win_spec_t* spec)
{
    const rocke_direct_conv_problem_t* p = &spec->problem;
    const bool bf16 = dw_win_is_bf16(p);
    const int BW = spec->block_w;
    const int CPL = spec->ch_per_lane;
    const int KH = p->KH, KW = p->KW, PAD = p->PAD;
    const int ROWS = spec->block_h;
    const int WIN = BW + KW - 1;
    const int N_PAIRS = (KW + 1) / 2;
    const bool DOT2 = spec->dot2;

    rocke_attr_set_int(b,
                       &b->kernel->attrs,
                       "max_workgroup_size",
                       rocke_direct_depthwise_dgrad_win_threads_per_block(spec));
    const rocke_type_t* io_type = rocke_b_io_ir_type(b, p->dtype ? p->dtype : "fp16");
    rocke_dconv_params_t prm;
    rocke_dconv_emit_params(b, &prm, "dgrad", io_type);
    if(!rocke_ir_builder_ok(b))
        return NULL;
    V* p_Hi = prm.p_Hi;
    V* p_Wi = prm.p_Wi;
    V* p_Ho = prm.p_Ho;
    V* p_Wo = prm.p_Wo;

    V* c0 = rocke_b_const_i32(b, 0);
    V* oob_lane = rocke_b_const_i32(b, ROCKE_DW_DGRAD_WIN_OOB_LANE);
    V* oob_uniform = rocke_b_const_i32(b, ROCKE_DW_DGRAD_WIN_OOB_UNIFORM);
    V* zero_f32 = rocke_b_const_f32(b, 0.0);
    V* zero_acc = CPL > 1 ? rocke_b_vector_splat(b, zero_f32, CPL) : zero_f32;

    V* tid = rocke_b_thread_id_x(b);
    V* bx = rocke_b_block_id_x(b);
    V* by = rocke_b_block_id_y(b);
    V* n = NULL;
    V* h0 = NULL;
    dw_win_tile(b, p_Hi, ROWS, &n, &h0);
    V* c_bw = rocke_b_const_i32(b, BW);
    V* wi0 = rocke_b_mul(b, bx, c_bw);
    V* lane_ch = tid;
    if(CPL > 1)
    {
        V* c_cpl = rocke_b_const_i32(b, CPL);
        lane_ch = rocke_b_mul(b, tid, c_cpl);
    }
    V* c_bch = rocke_b_const_i32(b, rocke_direct_depthwise_dgrad_win_block_ch(spec));
    V* by_ch = rocke_b_mul(b, by, c_bch);
    V* ch = rocke_b_add(b, by_ch, lane_ch);
    V* ch_ok = rocke_b_cmp_lt(b, ch, prm.p_groups);
    V* c2 = rocke_b_const_i32(b, 2);
    V* ch_bytes = rocke_b_mul(b, ch, c2);
    V* c_taps = rocke_b_const_i32(b, KH * KW * 2);
    V* ch_taps = rocke_b_mul(b, ch, c_taps);
    V* w_lane = rocke_b_select(b, ch_ok, ch_taps, oob_lane);
    V* bs[6];
    dw_win_byte_strides(b, &prm, bs);
    V *dy_img = bs[0], *dy_row_bytes = bs[1], *dy_px = bs[2];
    V *dx_img = bs[3], *dx_row_bytes = bs[4], *dx_px = bs[5];

    V* a_rsrc = rocke_b_buffer_rsrc(b, prm.A, prm.A_bytes);
    V* b_rsrc = rocke_b_buffer_rsrc(b, prm.Bp, prm.B_bytes);
    V* d_rsrc = rocke_b_buffer_rsrc(b, prm.D, prm.D_bytes);

    /* weights[r][k|s] */
    std::vector<std::vector<V*>> weights(KH);
    if(DOT2)
    {
        std::vector<std::vector<V*>> raw_w(KH, std::vector<V*>(KW));
        for(int r = 0; r < KH; ++r)
            for(int s = 0; s < KW; ++s)
            {
                V* c = rocke_b_const_i32(b, (r * KW + s) * 2);
                V* off = rocke_b_add(b, w_lane, c);
                raw_w[r][s] = dw_win_load_half(b, bf16, b_rsrc, off, c0);
            }
        V* zero_half = NULL;
        if(KW % 2)
            zero_half = bf16 ? rocke_b_trunc_f32_to_bf16(b, zero_f32)
                             : rocke_b_trunc_f32_to_f16(b, zero_f32);
        for(int r = 0; r < KH; ++r)
            for(int k = 0; k < N_PAIRS; ++k)
            {
                V* comps[2] = {raw_w[r][2 * k], 2 * k + 1 < KW ? raw_w[r][2 * k + 1] : zero_half};
                weights[r].push_back(rocke_b_vec_pack(b, comps, 2, io_type));
            }
    }
    else
    {
        for(int r = 0; r < KH; ++r)
            for(int s = 0; s < KW; ++s)
            {
                std::vector<V*> comps(CPL);
                for(int i = 0; i < CPL; ++i)
                {
                    V* c = rocke_b_const_i32(b, ((i * KH + r) * KW + s) * 2);
                    V* off = rocke_b_add(b, w_lane, c);
                    V* h = dw_win_load_half(b, bf16, b_rsrc, off, c0);
                    comps[i] = rocke_b_cast_to_f32(b, h);
                }
                weights[r].push_back(CPL > 1 ? rocke_b_vec_pack(b, comps.data(), CPL, rocke_f32())
                                             : comps[0]);
            }
    }

    std::vector<V*> win_lane(WIN);
    for(int t = 0; t < WIN; ++t)
    {
        const int rel = PAD - (KW - 1) + t;
        V* c = rocke_b_const_i32(b, rel);
        V* wo = rocke_b_add(b, wi0, c);
        V* ok = rocke_b_cmp_lt(b, wo, p_Wo);
        if(rel < 0)
        {
            V* ge = rocke_b_cmp_ge(b, wo, c0);
            ok = rocke_b_land(b, ge, ok);
        }
        ok = rocke_b_land(b, ch_ok, ok);
        V* m = rocke_b_mul(b, wo, dy_px);
        V* a = rocke_b_add(b, ch_bytes, m);
        win_lane[t] = rocke_b_select(b, ok, a, oob_lane);
    }
    std::vector<V*> out_lane(BW);
    for(int j = 0; j < BW; ++j)
    {
        V* c = rocke_b_const_i32(b, j);
        V* wi = rocke_b_add(b, wi0, c);
        V* lt = rocke_b_cmp_lt(b, wi, p_Wi);
        V* ok = rocke_b_land(b, ch_ok, lt);
        V* m = rocke_b_mul(b, wi, dx_px);
        V* a = rocke_b_add(b, ch_bytes, m);
        out_lane[j] = rocke_b_select(b, ok, a, oob_lane);
    }
    V* dy_n = rocke_b_mul(b, n, dy_img);
    V* dx_n = rocke_b_mul(b, n, dx_img);

    const int n_rows = ROWS + KH - 1;
    std::vector<V*> row_ok(n_rows, NULL);

    auto load_window = [&](int y) -> std::vector<V*> {
        const int rel_ho = y + PAD - (KH - 1);
        V* c = rocke_b_const_i32(b, rel_ho);
        V* ho = rocke_b_add(b, h0, c);
        V* ok = rocke_b_cmp_lt(b, ho, p_Ho);
        if(rel_ho < 0)
        {
            V* ge = rocke_b_cmp_ge(b, ho, c0);
            ok = rocke_b_land(b, ge, ok);
        }
        row_ok[y] = ok;
        V* m = rocke_b_mul(b, ho, dy_row_bytes);
        V* part = rocke_b_add(b, dy_n, m);
        V* dy_row = rocke_b_select(b, ok, part, oob_uniform);
        std::vector<V*> raw(WIN);
        for(int t = 0; t < WIN; ++t)
        {
            V* off = rocke_b_add(b, win_lane[t], dy_row);
            raw[t] = CPL == 1 ? dw_win_load_half(b, bf16, a_rsrc, off, c0)
                              : dw_win_load_vN(b, bf16, a_rsrc, off, c0, CPL / 2);
        }
        rocke_b_sched_barrier(b, 0);
        return raw;
    };

    auto widen = [&](V* v) -> V* {
        if(CPL == 1)
            return rocke_b_cast_to_f32(b, v);
        std::vector<V*> comps(CPL);
        for(int i = 0; i < CPL; ++i)
        {
            V* e = rocke_b_vec_extract(b, v, i);
            comps[i] = rocke_b_cast_to_f32(b, e);
        }
        return rocke_b_vec_pack(b, comps.data(), CPL, rocke_f32());
    };

    auto taps = [&](const std::vector<int>& rs,
                    const std::vector<V*>& raw,
                    const std::vector<V*>& accs) -> std::vector<V*> {
        std::vector<V*> out = accs;
        if(DOT2)
        {
            const int t_lo = KW % 2 == 0 ? 0 : (KW > 1 ? 2 : WIN);
            std::vector<V*> pair(WIN, NULL);
            for(int t = t_lo; t < WIN; ++t)
            {
                V* comps[2] = {raw[t], t > 0 ? raw[t - 1] : raw[t]};
                pair[t] = rocke_b_vec_pack(b, comps, 2, io_type);
            }
            std::vector<V*> tail;
            if(KW % 2)
            {
                const rocke_type_t* v2 = rocke_vector_type(b, io_type, 2);
                for(int j = 0; j < BW; ++j)
                {
                    V* bits = rocke_b_bitcast(b, raw[j], rocke_i16());
                    V* z = rocke_b_zext(b, bits, rocke_i32());
                    tail.push_back(rocke_b_vec_bitcast(b, z, v2));
                }
            }
            std::vector<int> k_order;
            if(KW % 2)
                k_order.push_back(N_PAIRS - 1);
            for(int k = 0; k < (KW % 2 ? N_PAIRS - 1 : N_PAIRS); ++k)
                k_order.push_back(k);
            for(size_t i = 0; i < rs.size(); ++i)
            {
                const int r = rs[i];
                for(int k : k_order)
                    for(int j = 0; j < BW; ++j)
                    {
                        V* x = (2 * k + 1 == KW) ? tail[j] : pair[j + KW - 1 - 2 * k];
                        out[i * BW + j] = rocke_b_fdot2(b, weights[r][k], x, out[i * BW + j]);
                    }
            }
        }
        else
        {
            std::vector<V*> win(WIN);
            for(int t = 0; t < WIN; ++t)
                win[t] = widen(raw[t]);
            for(size_t i = 0; i < rs.size(); ++i)
            {
                const int r = rs[i];
                for(int s = 0; s < KW; ++s)
                    for(int j = 0; j < BW; ++j)
                    {
                        V* x = win[j + KW - 1 - s];
                        V* a = out[i * BW + j];
                        out[i * BW + j] = CPL > 1 ? rocke_b_vector_fma(b, weights[r][s], x, a)
                                                  : rocke_b_fma(b, weights[r][s], x, a);
                    }
            }
        }
        return out;
    };

    V* c1 = rocke_b_const_i32(b, 1);
    std::vector<V*> pending = load_window(0);
    std::vector<std::vector<V*>> acc(KH, std::vector<V*>(BW, zero_acc));
    for(int y = 0; y < n_rows; ++y)
    {
        std::vector<int> rs, slots;
        for(int r = 0; r < KH; ++r)
        {
            const int hl = y + r - (KH - 1);
            if(hl >= 0 && hl < ROWS)
            {
                rs.push_back(r);
                slots.push_back(hl % KH);
            }
        }
        std::vector<V*> raw = pending;
        if(y + 1 < n_rows)
            pending = load_window(y + 1);
        std::vector<V*> accs;
        for(int sl : slots)
            for(int j = 0; j < BW; ++j)
                accs.push_back(acc[sl][j]);
        const int rel_ho = y + PAD - (KH - 1);
        if(rel_ho < 0 || rel_ho >= ROWS)
            dw_win_guarded(
                b, row_ok[y], c0, c1, accs, "dww", y, true, [&](const std::vector<V*>& carried) {
                    return taps(rs, raw, carried);
                });
        else
            accs = taps(rs, raw, accs);
        if(!rocke_ir_builder_ok(b))
            return NULL;
        for(size_t i = 0; i < slots.size(); ++i)
            for(int j = 0; j < BW; ++j)
                acc[slots[i]][j] = accs[i * BW + j];

        const int hi_local = y - (KH - 1);
        if(hi_local < 0 || hi_local >= ROWS)
            continue;
        const int slot = hi_local % KH;
        V* c = rocke_b_const_i32(b, hi_local);
        V* hi = rocke_b_add(b, h0, c);
        V* m = rocke_b_mul(b, hi, dx_row_bytes);
        V* part = rocke_b_add(b, dx_n, m);
        V* lt = rocke_b_cmp_lt(b, hi, p_Hi);
        V* dx_row = rocke_b_select(b, lt, part, oob_uniform);
        for(int j = 0; j < BW; ++j)
        {
            V* off = rocke_b_add(b, out_lane[j], dx_row);
            if(CPL == 1)
            {
                if(bf16)
                {
                    V* h = rocke_b_trunc_f32_to_bf16(b, acc[slot][j]);
                    rocke_b_buffer_store_bf16(b, d_rsrc, off, c0, h);
                }
                else
                {
                    V* h = rocke_b_trunc_f32_to_f16(b, acc[slot][j]);
                    rocke_b_buffer_store_f16(b, d_rsrc, off, c0, h);
                }
            }
            else
            {
                V* h = dw_win_trunc_vec(b, bf16, acc[slot][j]);
                dw_win_store_vN(b, bf16, d_rsrc, off, c0, h, CPL / 2);
            }
            acc[slot][j] = zero_acc;
        }
    }
    return b->kernel;
}

/* ===================================================================== *
 *  Toeplitz MFMA form
 * ===================================================================== */
typedef struct dw_win_plan
{
    V* px;
    V* ch8;
    V* g_off;
} dw_win_plan;

static rocke_kernel_def_t* dw_win_build_mfma(rocke_ir_builder_t* b,
                                             const rocke_direct_depthwise_dgrad_win_spec_t* spec)
{
    const rocke_direct_conv_problem_t* p = &spec->problem;
    const bool bf16 = dw_win_is_bf16(p);
    const int KH = p->KH, KW = p->KW, PAD = p->PAD;
    const int F = spec->w_fold;
    const int SUB = 16 / F;
    const int TW = rocke_direct_depthwise_dgrad_win_tile_w(spec);
    const int KWP = dw_win_passes(KW);
    const int U = TW + 4 * KWP - 2;
    const int ROWS = spec->block_h;
    const int THREADS = rocke_direct_depthwise_dgrad_win_threads_per_block(spec);
    const int BLOCK_CH = rocke_direct_depthwise_dgrad_win_block_ch(spec);
    const int PER_PX = BLOCK_CH / ROCKE_DW_DGRAD_MFMA_CH;
    const int CH_PAD = BLOCK_CH + ROCKE_DW_DGRAD_MFMA_CH;
    const int PF = spec->prefetch_rows;

    rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", THREADS);
    const rocke_type_t* io_type = rocke_b_io_ir_type(b, p->dtype ? p->dtype : "fp16");
    rocke_dconv_params_t prm;
    rocke_dconv_emit_params(b, &prm, "dgrad", io_type);
    if(!rocke_ir_builder_ok(b))
        return NULL;
    V* p_N = prm.p_N;
    V* p_Hi = prm.p_Hi;
    V* p_Wi = prm.p_Wi;
    V* p_Ho = prm.p_Ho;
    V* p_Wo = prm.p_Wo;
    V* p_G = prm.p_groups;

    V* c0 = rocke_b_const_i32(b, 0);
    V* oob_lane = rocke_b_const_i32(b, ROCKE_DW_DGRAD_WIN_OOB_LANE);
    V* oob_uniform = rocke_b_const_i32(b, ROCKE_DW_DGRAD_WIN_OOB_UNIFORM);
    V* zero_acc = rocke_b_zero_vec_f32(b, 4);
    V* zero_f32 = rocke_b_const_f32(b, 0.0);
    V* zero_half
        = bf16 ? rocke_b_trunc_f32_to_bf16(b, zero_f32) : rocke_b_trunc_f32_to_f16(b, zero_f32);

    V* tid = rocke_b_thread_id_x(b);
    V* bx = rocke_b_block_id_x(b);
    V* by = rocke_b_block_id_y(b);
    V* n_tile = NULL;
    V* h0 = NULL;
    dw_win_tile(b, p_Hi, ROWS, &n_tile, &h0);
    V* c64 = rocke_b_const_i32(b, 64);
    V* lane = rocke_b_mod(b, tid, c64);
    V* wave = rocke_b_div(b, tid, c64);
    V* c_tw = rocke_b_const_i32(b, TW);
    V* x0 = rocke_b_mul(b, bx, c_tw);
    V* c_f = rocke_b_const_i32(b, F);
    V* img0 = rocke_b_mul(b, n_tile, c_f);
    V* c_bch = rocke_b_const_i32(b, BLOCK_CH);
    V* ch_blk = rocke_b_mul(b, by, c_bch);
    V* c_mch = rocke_b_const_i32(b, ROCKE_DW_DGRAD_MFMA_CH);
    V* wave_ch = rocke_b_mul(b, wave, c_mch);
    V* c_wave = rocke_b_add(b, ch_blk, wave_ch);
    V* ch_ok = rocke_b_cmp_lt(b, c_wave, p_G);
    V* c16 = rocke_b_const_i32(b, 16);
    V* c8 = rocke_b_const_i32(b, 8);
    V* col16 = rocke_b_mod(b, lane, c16);
    V* t = rocke_b_div(b, lane, c16);
    V* k = rocke_b_mod(b, col16, c8);
    V* q = rocke_b_div(b, col16, c8);
    V* bs[6];
    dw_win_byte_strides(b, &prm, bs);
    V *dy_img = bs[0], *dy_row_bytes = bs[1], *dy_px = bs[2];
    V *dx_img = bs[3], *dx_row_bytes = bs[4], *dx_px = bs[5];

    V* a_rsrc = rocke_b_buffer_rsrc(b, prm.A, prm.A_bytes);
    V* b_rsrc = rocke_b_buffer_rsrc(b, prm.Bp, prm.B_bytes);
    V* d_rsrc = rocke_b_buffer_rsrc(b, prm.D, prm.D_bytes);

    V* ck = rocke_b_add(b, c_wave, k);
    V* c_taps = rocke_b_const_i32(b, KH * KW * 2);
    V* w_base = rocke_b_mul(b, ck, c_taps);
    V* c_KW = rocke_b_const_i32(b, KW);
    V* is_k[ROCKE_DW_DGRAD_MFMA_CH];
    for(int j = 0; j < ROCKE_DW_DGRAD_MFMA_CH; ++j)
    {
        V* c = rocke_b_const_i32(b, j);
        is_k[j] = rocke_b_cmp_eq(b, k, c);
    }
    std::vector<std::vector<V*>> weights(KH, std::vector<V*>(KWP));
    for(int r = 0; r < KH; ++r)
        for(int ph = 0; ph < KWP; ++ph)
        {
            V* c = rocke_b_const_i32(b, 4 * ph);
            V* tp = rocke_b_add(b, t, c);
            V* s = rocke_b_sub(b, tp, q);
            V* ge = rocke_b_cmp_ge(b, s, c0);
            V* lt = rocke_b_cmp_lt(b, s, c_KW);
            V* in = rocke_b_land(b, ge, lt);
            V* ok = rocke_b_land(b, ch_ok, in);
            V* cr = rocke_b_const_i32(b, r * KW + KW - 1);
            V* tap = rocke_b_sub(b, cr, s);
            V* c2 = rocke_b_const_i32(b, 2);
            V* tb = rocke_b_mul(b, tap, c2);
            V* a = rocke_b_add(b, w_base, tb);
            V* off = rocke_b_select(b, ok, a, oob_lane);
            V* w = dw_win_load_half(b, bf16, b_rsrc, off, c0);
            V* comps[ROCKE_DW_DGRAD_MFMA_CH];
            for(int j = 0; j < ROCKE_DW_DGRAD_MFMA_CH; ++j)
                comps[j] = rocke_b_select(b, is_k[j], w, zero_half);
            weights[r][ph] = rocke_b_vec_pack(b, comps, ROCKE_DW_DGRAD_MFMA_CH, io_type);
        }

    V* c_sub = rocke_b_const_i32(b, SUB);
    V* f = rocke_b_div(b, col16, c_sub);
    V* cm = rocke_b_mod(b, col16, c_sub);
    V* c2a = rocke_b_const_i32(b, 2);
    V* pair2 = rocke_b_mul(b, cm, c2a);
    V* c_per_px = rocke_b_const_i32(b, PER_PX);

    const int o_shape[3] = {2, F * TW, CH_PAD};
    V* o_smem = rocke_b_smem_alloc(b, io_type, o_shape, 3, "dx_stage");
    V* c_tw2 = rocke_b_const_i32(b, TW);
    V* fo = rocke_b_mul(b, f, c_tw2);
    V* c2b = rocke_b_const_i32(b, 2);
    V* th = rocke_b_div(b, t, c2b);
    V* pt = rocke_b_add(b, pair2, th);
    V* o_col = rocke_b_add(b, fo, pt);
    V* c2c = rocke_b_const_i32(b, 2);
    V* tm = rocke_b_mod(b, t, c2c);
    V* c4 = rocke_b_const_i32(b, 4);
    V* tm4 = rocke_b_mul(b, tm, c4);
    V* o_ch = rocke_b_add(b, wave_ch, tm4);
    const int n_out = F * TW * PER_PX;
    V* c_TW = rocke_b_const_i32(b, TW);
    std::vector<dw_win_plan> drain;
    for(int i = 0; i < dw_win_cdiv(n_out, THREADS); ++i)
    {
        V* ci = rocke_b_const_i32(b, i * THREADS);
        V* idx = rocke_b_add(b, tid, ci);
        V* px = rocke_b_div(b, idx, c_per_px);
        V* im = rocke_b_mod(b, idx, c_per_px);
        V* ch8 = rocke_b_mul(b, im, c8);
        V* pd = rocke_b_div(b, px, c_TW);
        V* img_o = rocke_b_add(b, img0, pd);
        V* pm = rocke_b_mod(b, px, c_TW);
        V* wi_o = rocke_b_add(b, x0, pm);
        V* ch_o = rocke_b_add(b, ch_blk, ch8);
        V* cn = rocke_b_const_i32(b, n_out);
        V* l1 = rocke_b_cmp_lt(b, idx, cn);
        V* l2 = rocke_b_cmp_lt(b, img_o, p_N);
        V* a1 = rocke_b_land(b, l1, l2);
        V* l3 = rocke_b_cmp_lt(b, wi_o, p_Wi);
        V* l4 = rocke_b_cmp_lt(b, ch_o, p_G);
        V* a2 = rocke_b_land(b, l3, l4);
        V* ok = rocke_b_land(b, a1, a2);
        V* m1 = rocke_b_mul(b, img_o, dx_img);
        V* m2 = rocke_b_mul(b, wi_o, dx_px);
        V* c2 = rocke_b_const_i32(b, 2);
        V* m3 = rocke_b_mul(b, ch_o, c2);
        V* s1 = rocke_b_add(b, m2, m3);
        V* g_off = rocke_b_add(b, m1, s1);
        dw_win_plan e = {px, ch8, rocke_b_select(b, ok, g_off, oob_lane)};
        drain.push_back(e);
    }

    const int n_in = F * U * PER_PX;
    const int in_passes = dw_win_cdiv(n_in, THREADS);
    const int in_px = dw_win_cdiv(in_passes * THREADS, PER_PX);
    const int i_shape[3] = {2, in_px, CH_PAD};
    V* i_smem = rocke_b_smem_alloc(b, io_type, i_shape, 3, "dy_stage");
    V* c_U = rocke_b_const_i32(b, U);
    std::vector<dw_win_plan> in_plan;
    for(int i = 0; i < in_passes; ++i)
    {
        V* ci = rocke_b_const_i32(b, i * THREADS);
        V* idx = rocke_b_add(b, tid, ci);
        V* px = rocke_b_div(b, idx, c_per_px);
        V* im = rocke_b_mod(b, idx, c_per_px);
        V* ch8 = rocke_b_mul(b, im, c8);
        V* pd = rocke_b_div(b, px, c_U);
        V* img_i = rocke_b_add(b, img0, pd);
        V* pm = rocke_b_mod(b, px, c_U);
        V* cp = rocke_b_const_i32(b, PAD - (KW - 1));
        V* pmp = rocke_b_add(b, pm, cp);
        V* wo = rocke_b_add(b, x0, pmp);
        V* ch_i = rocke_b_add(b, ch_blk, ch8);
        V* cn = rocke_b_const_i32(b, n_in);
        V* l1 = rocke_b_cmp_lt(b, idx, cn);
        V* l2 = rocke_b_cmp_lt(b, img_i, p_N);
        V* a1 = rocke_b_land(b, l1, l2);
        V* g1 = rocke_b_cmp_ge(b, wo, c0);
        V* l3 = rocke_b_cmp_lt(b, wo, p_Wo);
        V* a2 = rocke_b_land(b, g1, l3);
        V* l4 = rocke_b_cmp_lt(b, ch_i, p_G);
        V* a3 = rocke_b_land(b, a2, l4);
        V* ok = rocke_b_land(b, a1, a3);
        V* m1 = rocke_b_mul(b, img_i, dy_img);
        V* m2 = rocke_b_mul(b, wo, dy_px);
        V* c2 = rocke_b_const_i32(b, 2);
        V* m3 = rocke_b_mul(b, ch_i, c2);
        V* s1 = rocke_b_add(b, m2, m3);
        V* g_off = rocke_b_add(b, m1, s1);
        dw_win_plan e = {px, ch8, rocke_b_select(b, ok, g_off, oob_lane)};
        in_plan.push_back(e);
    }
    V* f_u = rocke_b_mul(b, f, c_U);
    std::vector<V*> b_px(KWP);
    for(int ph = 0; ph < KWP; ++ph)
    {
        V* c = rocke_b_const_i32(b, 4 * ph);
        V* tc = rocke_b_add(b, t, c);
        V* pt2 = rocke_b_add(b, pair2, tc);
        b_px[ph] = rocke_b_add(b, f_u, pt2);
    }

    const int n_rows = ROWS + KH - 1;
    std::vector<V*> row_ok(n_rows, NULL);

    auto load_row = [&](int y) -> std::vector<V*> {
        const int rel_ho = y + PAD - (KH - 1);
        V* c = rocke_b_const_i32(b, rel_ho);
        V* ho = rocke_b_add(b, h0, c);
        V* ok = rocke_b_cmp_lt(b, ho, p_Ho);
        if(rel_ho < 0)
        {
            V* ge = rocke_b_cmp_ge(b, ho, c0);
            ok = rocke_b_land(b, ge, ok);
        }
        row_ok[y] = ok;
        V* m = rocke_b_mul(b, ho, dy_row_bytes);
        V* dy_row = rocke_b_select(b, ok, m, oob_uniform);
        std::vector<V*> vals;
        for(const dw_win_plan& e : in_plan)
        {
            V* off = rocke_b_add(b, e.g_off, dy_row);
            vals.push_back(dw_win_load_vN(b, bf16, a_rsrc, off, c0, 4));
        }
        rocke_b_sched_barrier(b, 0);
        return vals;
    };

    auto dx_row_part = [&](int hi_local) -> V* {
        V* c = rocke_b_const_i32(b, hi_local);
        V* hi = rocke_b_add(b, h0, c);
        V* part = rocke_b_mul(b, hi, dx_row_bytes);
        V* lt = rocke_b_cmp_lt(b, hi, p_Hi);
        return rocke_b_select(b, lt, part, oob_uniform);
    };

    auto drain_out = [&](V* buf, V* dx_row) {
        for(const dw_win_plan& e : drain)
        {
            V* idx[3] = {buf, e.px, e.ch8};
            V* v = rocke_b_smem_load_vN(b, o_smem, idx, 3, io_type, 8);
            V* off = rocke_b_add(b, e.g_off, dx_row);
            dw_win_store_vN(b, bf16, d_rsrc, off, c0, v, 4);
        }
    };

    std::vector<std::vector<V*>> pending(n_rows);
    for(int y = 0; y < (PF < n_rows ? PF : n_rows); ++y)
        pending[y] = load_row(y);

    V* c1 = rocke_b_const_i32(b, 1);
    bool have_out = false;
    V* pend_val = NULL;
    V* pend_row = NULL;
    std::vector<V*> acc(KH, zero_acc);
    for(int y = 0; y < n_rows; ++y)
    {
        std::vector<int> rs, slots;
        for(int r = 0; r < KH; ++r)
        {
            const int hl = y + r - (KH - 1);
            if(hl >= 0 && hl < ROWS)
            {
                rs.push_back(r);
                slots.push_back(hl % KH);
            }
        }
        std::vector<V*> raw = pending[y];
        V* buf = rocke_b_const_i32(b, y % 2);
        const int rel_ho = y + PAD - (KH - 1);
        const bool halo = rel_ho < 0 || rel_ho >= ROWS;

        auto stage_row = [&]() {
            for(size_t i = 0; i < in_plan.size(); ++i)
            {
                V* idx[3] = {buf, in_plan[i].px, in_plan[i].ch8};
                rocke_b_smem_store_vN(b, i_smem, idx, 3, raw[i], 8);
            }
        };
        auto multiply = [&](const std::vector<V*>& accs) -> std::vector<V*> {
            std::vector<V*> xs(KWP);
            for(int ph = 0; ph < KWP; ++ph)
            {
                V* idx[3] = {buf, b_px[ph], wave_ch};
                xs[ph] = rocke_b_smem_load_vN(b, i_smem, idx, 3, io_type, 8);
            }
            std::vector<V*> out = accs;
            for(int ph = 0; ph < KWP; ++ph)
                for(size_t i = 0; i < rs.size(); ++i)
                    out[i] = bf16 ? rocke_b_mfma_f32_16x16x32_bf16(
                                        b, weights[rs[i]][ph], xs[ph], out[i])
                                  : rocke_b_mfma_f32_16x16x32_f16(
                                        b, weights[rs[i]][ph], xs[ph], out[i]);
            return out;
        };

        if(halo)
        {
            rocke_if_t g = rocke_b_scf_if(b, row_ok[y]);
            rocke_b_region_enter(b, g.then_region);
            stage_row();
            rocke_b_region_leave(b);
        }
        else
            stage_row();
        if(have_out)
        {
            V* idx[3] = {buf, o_col, o_ch};
            rocke_b_smem_store_vN(b, o_smem, idx, 3, pend_val, 4);
        }
        rocke_b_sync_lds_only(b);
        if(have_out)
        {
            drain_out(buf, pend_row);
            have_out = false;
        }
        if(y + PF < n_rows)
            pending[y + PF] = load_row(y + PF);
        std::vector<V*> accs;
        for(int sl : slots)
            accs.push_back(acc[sl]);
        if(halo)
            dw_win_guarded(b, row_ok[y], c0, c1, accs, "dwm", y, true, multiply);
        else
            accs = multiply(accs);
        if(!rocke_ir_builder_ok(b))
            return NULL;
        for(size_t i = 0; i < slots.size(); ++i)
            acc[slots[i]] = accs[i];
        const int hi_local = y - (KH - 1);
        if(hi_local < 0 || hi_local >= ROWS)
            continue;
        const int slot = hi_local % KH;
        pend_val = dw_win_trunc_vec(b, bf16, acc[slot]);
        pend_row = dx_row_part(hi_local);
        have_out = true;
        acc[slot] = zero_acc;
    }
    if(have_out)
    {
        V* buf = rocke_b_const_i32(b, n_rows % 2);
        V* idx[3] = {buf, o_col, o_ch};
        rocke_b_smem_store_vN(b, o_smem, idx, 3, pend_val, 4);
        rocke_b_sync_lds_only(b);
        drain_out(buf, pend_row);
    }
    return b->kernel;
}

/* ===================================================================== *
 *  Entry
 * ===================================================================== */
rocke_kernel_def_t* rocke_build_direct_depthwise_dgrad_win(
    rocke_ir_builder_t* b, const rocke_direct_depthwise_dgrad_win_spec_t* spec, const char* arch)
{
    (void)arch;
    if(spec->mfma)
        return dw_win_build_mfma(b, spec);
    return dw_win_build_valu(b, spec);
}

rocke_kernel_def_t* rocke_build_direct_depthwise_dgrad_win_new(
    rocke_ir_builder_t* b, const rocke_direct_depthwise_dgrad_win_spec_t* spec, const char* arch)
{
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        char name[256];
        char why[512];
        if(b == NULL || spec == NULL)
            return NULL;
        /* Validated before the Python builds its IRBuilder. */
        if(!rocke_direct_depthwise_dgrad_win_is_valid_spec(spec, arch, why, sizeof why))
        {
            rocke_ir_builder_init(b, "direct_depthwise_dgrad_win_invalid");
            rocke_i_set_err(b,
                            ROCKE_ERR_VALUE,
                            "invalid DirectDepthwiseDgradWindowedSpec for %s: %s",
                            arch ? arch : "gfx950",
                            why);
            return NULL;
        }
        if(rocke_direct_depthwise_dgrad_win_kernel_name(spec, name, sizeof name) != ROCKE_OK)
            return NULL;
        if(rocke_ir_builder_init(b, name) != ROCKE_OK)
            return NULL;
        return rocke_build_direct_depthwise_dgrad_win(b, spec, arch);
    });
}
