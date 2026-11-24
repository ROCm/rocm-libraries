// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * conv_direct_dw_fwd_wmma.cpp -- C++ port of the Toeplitz WMMA depthwise
 * forward (library/kernels/common/conv_direct_dw_fwd_wmma.py):
 *
 *   Python                                    C++ (this file)
 *   DwFwdWmmaSpec                          -> rocke_direct_dw_fwd_wmma_spec_t
 *   dw_fwd_wmma_reason / is_valid_...      -> rocke_direct_dw_fwd_wmma_is_valid_spec()
 *   kernel_name()                          -> rocke_direct_dw_fwd_wmma_kernel_name()
 *   build_dw_fwd_wmma(spec, arch)          -> rocke_build_direct_dw_fwd_wmma_new()
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

#include "rocke/error_boundary.hpp" /* ckc::guard_builder boundary shim */
#include "rocke/helper_rocke.helpers.io.h"
#include "rocke/helper_rocke.helpers.spec.h" /* rocke_kernel_name_join, rocke_ceil_div, ... */
#include "rocke/instance_conv_direct_dw_fwd_wmma.h"
#include "rocke/instance_conv_direct_grouped.h"
#include "rocke/instance_conv_direct_grouped_internal.h"
#include "rocke/instance_conv_fwd_wmma_gate_internal.h" /* ROCKE_FWD_WMMA_*, rocke_wmma_ab_dtype */
#include "rocke/ir.h"
#include "rocke/ir_internal.h" /* rocke_i_set_err */

typedef rocke_value_t V;

/* ===================================================================== *
 *  Spec helpers
 * ===================================================================== */
rocke_direct_dw_fwd_wmma_spec_t rocke_direct_dw_fwd_wmma_spec_default(void)
{
    rocke_direct_dw_fwd_wmma_spec_t s;
    memset(&s, 0, sizeof s);
    s.problem = rocke_direct_conv_problem_default();
    s.name = "rocke_direct_dw_fwd_wmma";
    s.block_waves = 4;
    s.block_h = 8;
    s.w_fold = 1;
    s.prefetch_rows = 2;
    return s;
}

int rocke_direct_dw_fwd_wmma_threads_per_block(const rocke_direct_dw_fwd_wmma_spec_t* spec)
{
    return spec->block_waves * ROCKE_DW_FWD_WMMA_WAVE;
}

int rocke_direct_dw_fwd_wmma_block_ch(const rocke_direct_dw_fwd_wmma_spec_t* spec)
{
    return spec->block_waves * ROCKE_DW_DGRAD_MFMA_CH;
}

int rocke_direct_dw_fwd_wmma_tile_w(const rocke_direct_dw_fwd_wmma_spec_t* spec)
{
    return rocke_dw_mfma_tile_w(spec->w_fold);
}

int rocke_direct_dw_fwd_wmma_in_passes(const rocke_direct_dw_fwd_wmma_spec_t* spec)
{
    return rocke_dw_mfma_in_passes(rocke_direct_dw_fwd_wmma_tile_w(spec),
                                   spec->problem.KW,
                                   spec->w_fold,
                                   rocke_direct_dw_fwd_wmma_block_ch(spec),
                                   rocke_direct_dw_fwd_wmma_threads_per_block(spec));
}

int rocke_direct_dw_fwd_wmma_frag_vgprs(const rocke_direct_dw_fwd_wmma_spec_t* spec)
{
    const rocke_direct_conv_problem_t* p = &spec->problem;
    const int kwp = rocke_dw_mfma_passes(p->KW);
    return 8 * (p->KH * kwp + p->KH + kwp)
           + 4 * spec->prefetch_rows * rocke_direct_dw_fwd_wmma_in_passes(spec);
}

long rocke_direct_dw_fwd_wmma_lds_bytes(const rocke_direct_dw_fwd_wmma_spec_t* spec)
{
    return rocke_dw_mfma_lds_bytes(rocke_direct_dw_fwd_wmma_in_passes(spec),
                                   rocke_direct_dw_fwd_wmma_threads_per_block(spec),
                                   rocke_direct_dw_fwd_wmma_block_ch(spec),
                                   spec->w_fold,
                                   rocke_direct_dw_fwd_wmma_tile_w(spec));
}

int rocke_direct_dw_fwd_wmma_unrolled_wmmas(const rocke_direct_dw_fwd_wmma_spec_t* spec)
{
    return spec->block_h * spec->problem.KH * rocke_dw_mfma_passes(spec->problem.KW);
}

void rocke_direct_dw_fwd_wmma_grid(const rocke_direct_dw_fwd_wmma_spec_t* spec, int grid[3])
{
    const rocke_direct_conv_problem_t* p = &spec->problem;
    grid[0] = (int)rocke_ceil_div(rocke_direct_conv_problem_Wo(p),
                                  rocke_direct_dw_fwd_wmma_tile_w(spec));
    grid[1] = (int)rocke_ceil_div(p->groups, rocke_direct_dw_fwd_wmma_block_ch(spec));
    grid[2] = (int)rocke_ceil_div(p->N, spec->w_fold)
              * (int)rocke_ceil_div(rocke_direct_conv_problem_Ho(p), spec->block_h);
}

rocke_status_t rocke_direct_dw_fwd_wmma_kernel_name(const rocke_direct_dw_fwd_wmma_spec_t* spec,
                                                    char* out,
                                                    size_t out_cap)
{
    if(spec == NULL)
        return ROCKE_ERR_VALUE;
    const rocke_direct_conv_problem_t* p = &spec->problem;
    char rsp[64], wv[32], bh[32], f[32], pf[32];
    snprintf(rsp, sizeof rsp, "r%ds%dp%d", p->KH, p->KW, p->PAD);
    snprintf(wv, sizeof wv, "wv%d", spec->block_waves);
    snprintf(bh, sizeof bh, "bh%d", spec->block_h);
    snprintf(f, sizeof f, "f%d", spec->w_fold);
    snprintf(pf, sizeof pf, "pf%d", spec->prefetch_rows);
    const char* parts[] = {rsp, wv, bh, f, pf};
    const char* flag_names[] = {"bf16"};
    int flag_on[] = {rocke_direct_conv_problem_is_bf16(p) ? 1 : 0};
    return rocke_kernel_name_join(spec->name ? spec->name : "rocke_direct_dw_fwd_wmma",
                                  parts,
                                  sizeof(parts) / sizeof(parts[0]),
                                  flag_names,
                                  flag_on,
                                  1,
                                  out,
                                  out_cap,
                                  NULL);
}

/* dw_fwd_wmma_reason(spec): the checks in the Python order and wording. */
static bool dfw_check(const rocke_direct_dw_fwd_wmma_spec_t* spec, char* reason, size_t cap)
{
    const rocke_direct_conv_problem_t* p = &spec->problem;
    if(rocke_direct_problem_dims_reason(p, reason, cap))
        return false;
    const char* dtype = rocke_direct_conv_problem_dtype(p);
    if(strcmp(dtype, "fp16") != 0 && strcmp(dtype, "bf16") != 0)
        ROCKE_SPEC_FAIL(reason, cap, "unsupported dtype '%s'; expected fp16 or bf16", dtype);
    if(p->cpg != 1 || p->kpg != 1)
        ROCKE_SPEC_FAIL(reason, cap, "requires cpg=kpg=1 (got %d, %d)", p->cpg, p->kpg);
    if(p->stride != 1)
        ROCKE_SPEC_FAIL(reason, cap, "requires stride=1 (got %d)", p->stride);
    if(rocke_direct_dilation_reason(p, reason, cap)
       || rocke_direct_conv_one_padding_reason(p, "", reason, cap))
        return false;
    {
        const int kmax = p->KH > p->KW ? p->KH : p->KW;
        const long long Ho = rocke_direct_conv_problem_Ho(p), Wo = rocke_direct_conv_problem_Wo(p);
        if(!(0 <= p->PAD && p->PAD < kmax) || Ho <= 0 || Wo <= 0)
            ROCKE_SPEC_FAIL(
                reason, cap, "degenerate geometry (PAD=%d, Ho=%lld, Wo=%lld)", p->PAD, Ho, Wo);
    }
    if(spec->block_h < 1)
        ROCKE_SPEC_FAIL(reason, cap, "block_h must be >= 1 (got %d)", spec->block_h);
    if(spec->block_waves < 1 || spec->block_waves > ROCKE_DW_FWD_WMMA_MAX_WAVES)
        ROCKE_SPEC_FAIL(reason,
                        cap,
                        "block_waves must be 1..%d (got %d)",
                        ROCKE_DW_FWD_WMMA_MAX_WAVES,
                        spec->block_waves);
    if(!rocke_dw_mfma_w_fold_ok(spec->w_fold))
        ROCKE_SPEC_FAIL(reason, cap, "w_fold must be one of (1, 2, 4) (got %d)", spec->w_fold);
    if(spec->prefetch_rows < 1 || spec->prefetch_rows > ROCKE_DW_DGRAD_MFMA_MAX_PREFETCH_ROWS)
        ROCKE_SPEC_FAIL(reason,
                        cap,
                        "prefetch_rows must be 1..%d (got %d)",
                        ROCKE_DW_DGRAD_MFMA_MAX_PREFETCH_ROWS,
                        spec->prefetch_rows);
    if(p->groups % ROCKE_DW_DGRAD_MFMA_CH != 0)
        ROCKE_SPEC_FAIL(
            reason, cap, "groups %% %d != 0 (got %d)", ROCKE_DW_DGRAD_MFMA_CH, p->groups);
    {
        const int frag = rocke_direct_dw_fwd_wmma_frag_vgprs(spec);
        if(spec->block_waves > ROCKE_DW_FWD_WMMA_WIDE_WAVES
           && frag > ROCKE_DW_FWD_WMMA_WIDE_FRAG_VGPRS)
            ROCKE_SPEC_FAIL(reason,
                            cap,
                            "%d waves need fragments <= %d VGPRs (got %d)",
                            spec->block_waves,
                            ROCKE_DW_FWD_WMMA_WIDE_FRAG_VGPRS,
                            frag);
        if(frag > ROCKE_DW_FWD_WMMA_MAX_FRAG_VGPRS)
            ROCKE_SPEC_FAIL(reason,
                            cap,
                            "fragments need %d VGPRs (> %d); lower prefetch_rows or the filter",
                            frag,
                            ROCKE_DW_FWD_WMMA_MAX_FRAG_VGPRS);
    }
    {
        const int u = rocke_direct_dw_fwd_wmma_unrolled_wmmas(spec);
        if(u > ROCKE_DW_FWD_WMMA_MAX_UNROLL)
            ROCKE_SPEC_FAIL(reason,
                            cap,
                            "unrolled body too large (%d WMMAs > %d); lower block_h",
                            u,
                            ROCKE_DW_FWD_WMMA_MAX_UNROLL);
    }
    {
        const long lds = rocke_direct_dw_fwd_wmma_lds_bytes(spec);
        if(lds > ROCKE_DW_DGRAD_MFMA_LDS_BUDGET)
            ROCKE_SPEC_FAIL(reason,
                            cap,
                            "stages %ld LDS bytes per workgroup (budget %d); lower block_waves",
                            lds,
                            ROCKE_DW_DGRAD_MFMA_LDS_BUDGET);
    }
    {
        const long long x = (long long)p->N * p->H * p->W * p->groups * 2;
        const long long y = (long long)p->N * rocke_direct_conv_problem_Ho(p)
                            * rocke_direct_conv_problem_Wo(p) * p->groups * 2;
        const long long mx = x > y ? x : y;
        if(mx > ROCKE_DW_DGRAD_WIN_MAX_TENSOR_BYTES)
            ROCKE_SPEC_FAIL(reason,
                            cap,
                            "X/Y of %lld bytes exceed the %lld-byte sentinel addressing range",
                            mx,
                            (long long)ROCKE_DW_DGRAD_WIN_MAX_TENSOR_BYTES);
    }
    return true;
}

bool rocke_direct_dw_fwd_wmma_is_valid_spec(const rocke_direct_dw_fwd_wmma_spec_t* spec,
                                            const char* arch,
                                            char* reason,
                                            size_t reason_cap)
{
    if(spec == NULL)
    {
        rocke_spec_set_reason(reason, reason_cap, "null spec");
        return false;
    }
    if(arch == NULL)
        arch = "gfx1250";
    if(strcmp(arch, "gfx1250") != 0)
    {
        if(reason != NULL && reason_cap > 0)
            snprintf(reason, reason_cap, "needs one of ('gfx1250',) (got %s)", arch);
        return false;
    }
    if(!dfw_check(spec, reason, reason_cap))
        return false;
    rocke_spec_set_reason(reason, reason_cap, "ok");
    return true;
}

/* ===================================================================== *
 *  build_dw_fwd_wmma
 * ===================================================================== */
rocke_kernel_def_t* rocke_build_direct_dw_fwd_wmma(rocke_ir_builder_t* b,
                                                   const rocke_direct_dw_fwd_wmma_spec_t* spec,
                                                   const char* arch)
{
    const rocke_direct_conv_problem_t* p = &spec->problem;
    const bool bf16 = rocke_direct_conv_problem_is_bf16(p);
    const int KH = p->KH, KW = p->KW, PAD = p->PAD;
    const int F = spec->w_fold;
    const int SUB = 16 / F;
    const int TW = rocke_direct_dw_fwd_wmma_tile_w(spec);
    const int KWP = rocke_dw_mfma_passes(KW);
    const int U = TW + 4 * KWP - 2;
    const int ROWS = spec->block_h;
    const int THREADS = rocke_direct_dw_fwd_wmma_threads_per_block(spec);
    const int BLOCK_CH = rocke_direct_dw_fwd_wmma_block_ch(spec);
    const int CH = ROCKE_DW_DGRAD_MFMA_CH;
    const int PER_PX = BLOCK_CH / CH;
    const int CH_PAD = BLOCK_CH + CH;

    const char* wmma_op = rocke_wmma_fwd_op_id(
        b, rocke_direct_conv_problem_dtype(p), arch ? arch : "gfx1250", "build_dw_fwd_wmma");
    if(wmma_op == NULL)
        return NULL;

    rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", THREADS);
    const rocke_type_t* io_type = rocke_b_io_ir_type(b, rocke_direct_conv_problem_dtype(p));
    rocke_dconv_params_t prm;
    rocke_dconv_emit_params(b, &prm, "fwd", io_type);
    if(!rocke_ir_builder_ok(b))
        return NULL;
    V* p_N = prm.p_N;
    V* p_Hi = prm.p_Hi;
    V* p_Wi = prm.p_Wi;
    V* p_Ho = prm.p_Ho;
    V* p_Wo = prm.p_Wo;
    V* p_G = prm.p_groups;
    V* c2 = rocke_b_const_i32(b, 2);
    V* bs[6];
    rocke_dw_win_byte_strides(b, &prm, bs, c2);
    V *x_img = bs[0], *x_row = bs[1], *x_px = bs[2];
    V *y_img = bs[3], *y_row = bs[4], *y_px = bs[5];

    V* c0 = rocke_b_const_i32(b, 0);
    V* oob_lane = rocke_b_const_i32(b, ROCKE_DW_DGRAD_WIN_OOB_LANE);
    V* oob_uniform = rocke_b_const_i32(b, ROCKE_DW_DGRAD_WIN_OOB_UNIFORM);
    V* zero_acc = rocke_b_zero_vec_f32(b, 8);
    V* zero_f32 = rocke_b_const_f32(b, 0.0);
    V* zero_half = rocke_dconv_trunc_f32_elem(b, bf16, zero_f32);

    V* tid = rocke_b_thread_id_x(b);
    V* bx = rocke_b_block_id_x(b);
    V* by = rocke_b_block_id_y(b);
    V* n_tile;
    V* h0;
    rocke_dw_win_tile(b, p_Ho, ROWS, &n_tile, &h0);
    V* c_wave_size = rocke_b_const_i32(b, ROCKE_DW_FWD_WMMA_WAVE);
    V* lane = rocke_b_mod(b, tid, c_wave_size);
    V* wave = rocke_b_div(b, tid, c_wave_size);
    V* c_tw0 = rocke_b_const_i32(b, TW);
    V* x0 = rocke_b_mul(b, bx, c_tw0);
    V* c_f = rocke_b_const_i32(b, F);
    V* img0 = rocke_b_mul(b, n_tile, c_f);
    V* c_bch = rocke_b_const_i32(b, BLOCK_CH);
    V* ch_blk = rocke_b_mul(b, by, c_bch);
    V* c_ch = rocke_b_const_i32(b, CH);
    V* wave_ch = rocke_b_mul(b, wave, c_ch);
    V* c_wave = rocke_b_add(b, ch_blk, wave_ch);
    V* ch_ok = rocke_b_cmp_lt(b, c_wave, p_G);
    V* c16 = rocke_b_const_i32(b, 16);
    V* c8 = rocke_b_const_i32(b, 8);
    V* col16 = rocke_b_mod(b, lane, c16);
    V* t = rocke_b_div(b, lane, c16);
    V* k = rocke_b_mod(b, col16, c8);
    V* q = rocke_b_div(b, col16, c8);

    V* a_rsrc = rocke_b_buffer_rsrc(b, prm.A, prm.A_bytes);
    V* b_rsrc = rocke_b_buffer_rsrc(b, prm.Bp, prm.B_bytes);
    V* d_rsrc = rocke_b_buffer_rsrc(b, prm.D, prm.D_bytes);

    /* One-hot A fragments weights[r][ph]: W[k, r, s] at slot k. */
    V* ck = rocke_b_add(b, c_wave, k);
    V* c_taps = rocke_b_const_i32(b, KH * KW * 2);
    V* w_base = rocke_b_mul(b, ck, c_taps);
    V* c_KW = rocke_b_const_i32(b, KW);
    V* is_k[ROCKE_DW_DGRAD_MFMA_CH];
    for(int j = 0; j < CH; ++j)
    {
        V* c = rocke_b_const_i32(b, j);
        is_k[j] = rocke_b_cmp_eq(b, k, c);
    }
    std::vector<std::vector<V*>> weights(KH, std::vector<V*>(KWP));
    for(int r = 0; r < KH; ++r)
        for(int ph = 0; ph < KWP; ++ph)
        {
            V* comps[2 * ROCKE_DW_DGRAD_MFMA_CH];
            for(int j = 0; j < 2; ++j)
            {
                V* t2 = rocke_b_mul(b, t, c2);
                V* cj = rocke_b_const_i32(b, 4 * ph + j);
                V* tj = rocke_b_add(b, t2, cj);
                V* s = rocke_b_sub(b, tj, q);
                V* ge = rocke_b_cmp_ge(b, s, c0);
                V* lt = rocke_b_cmp_lt(b, s, c_KW);
                V* in = rocke_b_land(b, ge, lt);
                V* ok = rocke_b_land(b, ch_ok, in);
                V* cr = rocke_b_const_i32(b, r * KW);
                V* tap = rocke_b_add(b, cr, s);
                V* tb = rocke_b_mul(b, tap, c2);
                V* a = rocke_b_add(b, w_base, tb);
                V* off = rocke_b_select(b, ok, a, oob_lane);
                V* w = rocke_dconv_buf_load_half(b, bf16, b_rsrc, off, c0);
                for(int i = 0; i < CH; ++i)
                    comps[j * CH + i] = rocke_b_select(b, is_k[i], w, zero_half);
            }
            weights[r][ph] = rocke_b_vec_pack(b, comps, 2 * CH, io_type);
        }

    /* Lane parts shared by the B reads and the D staging. */
    V* c_sub = rocke_b_const_i32(b, SUB);
    V* f = rocke_b_div(b, col16, c_sub);
    V* cm = rocke_b_mod(b, col16, c_sub);
    V* pair2 = rocke_b_mul(b, cm, c2);
    V* c_per_px = rocke_b_const_i32(b, PER_PX);

    /* Y staging [buffer, image column, channel]. */
    const int o_shape[3] = {2, F * TW, CH_PAD};
    V* o_smem = rocke_b_smem_alloc(b, io_type, o_shape, 3, "y_stage");
    V* o_col;
    {
        V* c = rocke_b_const_i32(b, TW);
        V* fo = rocke_b_mul(b, f, c);
        V* pt = rocke_b_add(b, pair2, t);
        o_col = rocke_b_add(b, fo, pt);
    }
    const int n_out = F * TW * PER_PX;
    V* c_TW = rocke_b_const_i32(b, TW);
    std::vector<rocke_dw_win_plan_t> drain;
    for(int i = 0; i < (int)rocke_ceil_div(n_out, THREADS); ++i)
    {
        V* ci = rocke_b_const_i32(b, i * THREADS);
        V* idx = rocke_b_add(b, tid, ci);
        V* px = rocke_b_div(b, idx, c_per_px);
        V* im = rocke_b_mod(b, idx, c_per_px);
        V* ch8 = rocke_b_mul(b, im, c8);
        V* pd = rocke_b_div(b, px, c_TW);
        V* img_o = rocke_b_add(b, img0, pd);
        V* pm = rocke_b_mod(b, px, c_TW);
        V* wo = rocke_b_add(b, x0, pm);
        V* ch_o = rocke_b_add(b, ch_blk, ch8);
        V* cn = rocke_b_const_i32(b, n_out);
        V* l1 = rocke_b_cmp_lt(b, idx, cn);
        V* l2 = rocke_b_cmp_lt(b, img_o, p_N);
        V* a1 = rocke_b_land(b, l1, l2);
        V* l3 = rocke_b_cmp_lt(b, wo, p_Wo);
        V* l4 = rocke_b_cmp_lt(b, ch_o, p_G);
        V* a2 = rocke_b_land(b, l3, l4);
        V* ok = rocke_b_land(b, a1, a2);
        V* m1 = rocke_b_mul(b, img_o, y_img);
        V* m2 = rocke_b_mul(b, wo, y_px);
        V* m3 = rocke_b_mul(b, ch_o, c2);
        V* s1 = rocke_b_add(b, m2, m3);
        V* g_off = rocke_b_add(b, m1, s1);
        rocke_dw_win_plan_t e = {px, ch8, rocke_b_select(b, ok, g_off, oob_lane)};
        drain.push_back(e);
    }

    /* X staging [buffer, window column, channel]. */
    const int n_in = F * U * PER_PX;
    const int in_passes = (int)rocke_ceil_div(n_in, THREADS);
    const int in_px = (int)rocke_ceil_div((long long)in_passes * THREADS, PER_PX);
    const int i_shape[3] = {2, in_px, CH_PAD};
    V* i_smem = rocke_b_smem_alloc(b, io_type, i_shape, 3, "x_stage");
    V* c_U = rocke_b_const_i32(b, U);
    std::vector<rocke_dw_win_plan_t> in_plan;
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
        V* cp = rocke_b_const_i32(b, PAD);
        V* pmp = rocke_b_sub(b, pm, cp);
        V* wi = rocke_b_add(b, x0, pmp);
        V* ch_i = rocke_b_add(b, ch_blk, ch8);
        V* cn = rocke_b_const_i32(b, n_in);
        V* l1 = rocke_b_cmp_lt(b, idx, cn);
        V* l2 = rocke_b_cmp_lt(b, img_i, p_N);
        V* a1 = rocke_b_land(b, l1, l2);
        V* g1 = rocke_b_cmp_ge(b, wi, c0);
        V* l3 = rocke_b_cmp_lt(b, wi, p_Wi);
        V* a2 = rocke_b_land(b, g1, l3);
        V* l4 = rocke_b_cmp_lt(b, ch_i, p_G);
        V* a3 = rocke_b_land(b, a2, l4);
        V* ok = rocke_b_land(b, a1, a3);
        V* m1 = rocke_b_mul(b, img_i, x_img);
        V* m2 = rocke_b_mul(b, wi, x_px);
        V* m3 = rocke_b_mul(b, ch_i, c2);
        V* s1 = rocke_b_add(b, m2, m3);
        V* g_off = rocke_b_add(b, m1, s1);
        rocke_dw_win_plan_t e = {px, ch8, rocke_b_select(b, ok, g_off, oob_lane)};
        in_plan.push_back(e);
    }
    V* f_u = rocke_b_mul(b, f, c_U);
    V* t2 = rocke_b_mul(b, t, c2);
    std::vector<std::vector<V*>> b_px(KWP, std::vector<V*>(2));
    for(int ph = 0; ph < KWP; ++ph)
        for(int j = 0; j < 2; ++j)
        {
            V* c = rocke_b_const_i32(b, 4 * ph + j);
            V* tc = rocke_b_add(b, t2, c);
            V* pt2 = rocke_b_add(b, pair2, tc);
            b_px[ph][j] = rocke_b_add(b, f_u, pt2);
        }

    const int n_rows = ROWS + KH - 1;
    std::vector<V*> row_ok(n_rows, NULL);

    auto load_row = [&](int y) -> std::vector<V*> {
        const int rel = y - PAD;
        V* c = rocke_b_const_i32(b, rel);
        V* hi = rocke_b_add(b, h0, c);
        V* ok = rocke_b_cmp_lt(b, hi, p_Hi);
        if(rel < 0)
        {
            V* ge = rocke_b_cmp_ge(b, hi, c0);
            ok = rocke_b_land(b, ge, ok);
        }
        row_ok[y] = ok;
        V* m = rocke_b_mul(b, hi, x_row);
        V* x_r = rocke_b_select(b, ok, m, oob_uniform);
        std::vector<V*> vals;
        for(const rocke_dw_win_plan_t& e : in_plan)
        {
            V* off = rocke_b_add(b, e.g_off, x_r);
            vals.push_back(rocke_dconv_buf_load_vN(b, bf16, a_rsrc, off, c0, 4));
        }
        rocke_b_sched_barrier(b, 0);
        return vals;
    };

    auto y_row_part = [&](int o_local) -> V* {
        V* c = rocke_b_const_i32(b, o_local);
        V* ho = rocke_b_add(b, h0, c);
        V* lt = rocke_b_cmp_lt(b, ho, p_Ho);
        V* part = rocke_b_mul(b, ho, y_row);
        return rocke_b_select(b, lt, part, oob_uniform);
    };

    auto drain_out = [&](V* buf, V* y_r) {
        for(const rocke_dw_win_plan_t& e : drain)
        {
            V* idx[3] = {buf, e.px, e.ch8};
            V* v = rocke_b_smem_load_vN(b, o_smem, idx, 3, io_type, 8);
            V* off = rocke_b_add(b, e.g_off, y_r);
            rocke_dconv_buf_store_vN(b, bf16, d_rsrc, off, c0, v, 4);
        }
    };

    const int PF = spec->prefetch_rows;
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
            if(y - r >= 0 && y - r < ROWS)
            {
                rs.push_back(r);
                slots.push_back((y - r) % KH);
            }
        std::vector<V*> raw = pending[y];
        V* buf = rocke_b_const_i32(b, y % 2);
        const int rel = y - PAD;
        const bool halo = rel < 0 || rel >= ROWS;

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
                V* i0[3] = {buf, b_px[ph][0], wave_ch};
                V* lo = rocke_b_smem_load_vN(b, i_smem, i0, 3, io_type, 8);
                V* i1[3] = {buf, b_px[ph][1], wave_ch};
                V* hi = rocke_b_smem_load_vN(b, i_smem, i1, 3, io_type, 8);
                xs[ph] = rocke_b_vec_concat(b, lo, hi);
            }
            std::vector<V*> out = accs;
            for(int ph = 0; ph < KWP; ++ph)
                for(size_t i = 0; i < rs.size(); ++i)
                    out[i] = rocke_b_mma(b, wmma_op, weights[rs[i]][ph], xs[ph], out[i], NULL, 0);
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
            V* idx[3] = {buf, o_col, wave_ch};
            rocke_b_smem_store_vN(b, o_smem, idx, 3, pend_val, 8);
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
            rocke_dw_win_guarded(b, row_ok[y], c0, c1, accs, "dfw", y, true, multiply);
        else
            accs = multiply(accs);
        if(!rocke_ir_builder_ok(b))
            return NULL;
        for(size_t i = 0; i < slots.size(); ++i)
            acc[slots[i]] = accs[i];
        const int o_local = y - (KH - 1);
        if(o_local < 0 || o_local >= ROWS)
            continue;
        const int slot = o_local % KH;
        pend_val = rocke_dconv_trunc_f32(b, bf16, acc[slot]);
        pend_row = y_row_part(o_local);
        have_out = true;
        acc[slot] = zero_acc;
    }
    if(have_out)
    {
        V* buf = rocke_b_const_i32(b, n_rows % 2);
        V* idx[3] = {buf, o_col, wave_ch};
        rocke_b_smem_store_vN(b, o_smem, idx, 3, pend_val, 8);
        rocke_b_sync_lds_only(b);
        drain_out(buf, pend_row);
    }
    return b->kernel;
}

rocke_kernel_def_t* rocke_build_direct_dw_fwd_wmma_new(rocke_ir_builder_t* b,
                                                       const rocke_direct_dw_fwd_wmma_spec_t* spec,
                                                       const char* arch)
{
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        char name[256];
        char why[512];
        if(b == NULL || spec == NULL)
            return NULL;
        if(arch == NULL)
            arch = "gfx1250";
        /* Validated before the Python builds its IRBuilder. */
        if(!rocke_direct_dw_fwd_wmma_is_valid_spec(spec, arch, why, sizeof why))
        {
            rocke_ir_builder_init(b, "direct_dw_fwd_wmma_invalid");
            rocke_i_set_err(b, ROCKE_ERR_VALUE, "invalid DwFwdWmmaSpec for %s: %s", arch, why);
            return NULL;
        }
        if(rocke_direct_dw_fwd_wmma_kernel_name(spec, name, sizeof name) != ROCKE_OK)
            return NULL;
        if(rocke_ir_builder_init(b, name) != ROCKE_OK)
            return NULL;
        return rocke_build_direct_dw_fwd_wmma(b, spec, arch);
    });
}

rocke_status_t rocke_direct_dw_fwd_wmma_lower_to_llvm(const rocke_direct_dw_fwd_wmma_spec_t* spec,
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
        *out_ll = NULL;
    if(spec == NULL || out_ll == NULL)
    {
        rocke_spec_set_reason(err, err_cap, "lower_to_llvm: null spec/out");
        return ROCKE_ERR_VALUE;
    }
    if(arch == NULL)
        arch = "gfx1250";
    kernel = rocke_build_direct_dw_fwd_wmma_new(&b, spec, arch);
    if(kernel == NULL)
    {
        const char* m = rocke_ir_builder_error(&b);
        st = rocke_ir_builder_status(&b);
        rocke_spec_set_reason(
            err, err_cap, (m != NULL && m[0] != '\0') ? m : "build_dw_fwd_wmma failed");
        rocke_ir_builder_free(&b);
        return (st == ROCKE_OK) ? ROCKE_ERR_VALUE : st;
    }
    st = rocke_lower_kernel_to_llvm_ex(kernel, flavor, arch, out_ll, err, err_cap);
    rocke_ir_builder_free(&b);
    return st;
}
