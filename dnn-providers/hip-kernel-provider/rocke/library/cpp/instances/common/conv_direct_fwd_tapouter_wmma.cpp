// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * conv_direct_fwd_tapouter_wmma.cpp -- C++ port of the tap-outer direct
 * forward for large dilations (library/kernels/common/conv_direct_fwd_tapouter_wmma.py):
 *
 *   Python                                        C++ (this file)
 *   DirectFwdTapOuterSpec                      -> rocke_fwd_tapouter_wmma_spec_t
 *   tapouter_reason / is_valid_tapouter_spec   -> rocke_fwd_tapouter_wmma_is_valid_spec()
 *   kernel_name()                              -> rocke_fwd_tapouter_wmma_kernel_name()
 *   build_direct_fwd_tapouter_wmma(spec, arch) -> rocke_build_fwd_tapouter_wmma_new()
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

#include <algorithm>
#include <string>
#include <vector>

#include "rocke/error_boundary.hpp" /* ckc::guard_builder boundary shim */
#include "rocke/helper_rocke.core.arch.h" /* rocke_archtarget_fits_lds, max_threads_per_block */
#include "rocke/helper_rocke.helpers.io.h" /* rocke_b_io_ir_type */
#include "rocke/helper_rocke.helpers.spec.h" /* rocke_kernel_name_join, rocke_ceil_div, ... */
#include "rocke/instance_conv_direct_fwd_tapouter_wmma.h"
#include "rocke/instance_conv_direct_grouped.h"
#include "rocke/instance_conv_direct_grouped_internal.h" /* rocke_dconv_emit_params */
#include "rocke/instance_conv_fwd_wmma_gate_internal.h" /* rocke_wmma_fwd_gate */
#include "rocke/instance_conv_fwd_wmma_tdm_internal.h" /* rocke_fwd_tdm_descriptor_3d */
#include "rocke/ir.h"
#include "rocke/ir_internal.h" /* rocke_i_set_err */

typedef rocke_value_t* V;

namespace
{

int to_x_pitch(const rocke_fwd_tapouter_wmma_spec_t* s)
{
    return s->ck + ROCKE_FWD_WMMA_CH_PAD;
}

int to_epi_pitch(const rocke_fwd_tapouter_wmma_spec_t* s)
{
    return rocke_fwd_tapouter_wmma_block_k(s) + ROCKE_FWD_WMMA_CH_PAD;
}

long to_win_elems(const rocke_fwd_tapouter_wmma_spec_t* s)
{
    return ((long)rocke_fwd_tapouter_wmma_block_p(s) + s->tile_w) * to_x_pitch(s);
}

long to_stage_elems(const rocke_fwd_tapouter_wmma_spec_t* s)
{
    return to_win_elems(s) + (long)rocke_fwd_tapouter_wmma_block_k(s) * to_x_pitch(s);
}

long long to_grid0(const rocke_fwd_tapouter_wmma_spec_t* s)
{
    const rocke_direct_conv_problem_t* p = &s->problem;
    return (long long)p->N * p->groups
           * rocke_ceil_div(rocke_direct_conv_problem_Ho(p), rocke_fwd_tapouter_wmma_tile_h(s))
           * rocke_ceil_div(rocke_direct_conv_problem_Wo(p), s->tile_w)
           * rocke_ceil_div(p->kpg, rocke_fwd_tapouter_wmma_block_k(s));
}

} // namespace

/* ===================================================================== *
 *  Spec helpers
 * ===================================================================== */
rocke_fwd_tapouter_wmma_spec_t rocke_fwd_tapouter_wmma_spec_default(void)
{
    rocke_fwd_tapouter_wmma_spec_t s;
    memset(&s, 0, sizeof s);
    s.problem = rocke_direct_conv_problem_default();
    s.name = "direct_fwd_tapouter_wmma";
    s.waves_k = 2;
    s.waves_p = 4;
    s.tiles_k = 4;
    s.tiles_p = 2;
    s.tile_w = 16;
    s.ck = 64;
    s.stages = 3;
    s.prefetch = false;
    s.sched = NULL;
    s.wave_size = 32;
    s.tdm_split = 1;
    return s;
}

int rocke_fwd_tapouter_wmma_block_k(const rocke_fwd_tapouter_wmma_spec_t* spec)
{
    return spec->waves_k * spec->tiles_k * 16;
}

int rocke_fwd_tapouter_wmma_block_p(const rocke_fwd_tapouter_wmma_spec_t* spec)
{
    return spec->waves_p * spec->tiles_p * 16;
}

int rocke_fwd_tapouter_wmma_tile_h(const rocke_fwd_tapouter_wmma_spec_t* spec)
{
    return spec->tile_w > 0
               ? (int)rocke_floor_div(rocke_fwd_tapouter_wmma_block_p(spec), spec->tile_w)
               : 0;
}

int rocke_fwd_tapouter_wmma_threads_per_block(const rocke_fwd_tapouter_wmma_spec_t* spec)
{
    return spec->waves_k * spec->waves_p * spec->wave_size;
}

long rocke_fwd_tapouter_wmma_lds_bytes(const rocke_fwd_tapouter_wmma_spec_t* spec)
{
    const long ring = (long)spec->stages * to_stage_elems(spec);
    const long epi = (long)rocke_fwd_tapouter_wmma_block_p(spec) * to_epi_pitch(spec);
    return 2 * std::max(ring, epi);
}

void rocke_fwd_tapouter_wmma_grid(const rocke_fwd_tapouter_wmma_spec_t* spec, int grid[3])
{
    const long long g = to_grid0(spec);
    grid[0] = (int)(g > INT32_MAX ? INT32_MAX : g);
    grid[1] = 1;
    grid[2] = 1;
}

double rocke_fwd_tapouter_wmma_tile_efficiency(const rocke_fwd_tapouter_wmma_spec_t* spec)
{
    const rocke_direct_conv_problem_t* p = &spec->problem;
    const int th = rocke_fwd_tapouter_wmma_tile_h(spec);
    const int bk = rocke_fwd_tapouter_wmma_block_k(spec);
    const double t0 = (double)rocke_align_up(rocke_direct_conv_problem_Ho(p), th);
    const double t1 = (double)rocke_align_up(rocke_direct_conv_problem_Wo(p), spec->tile_w);
    const double t2 = (double)rocke_align_up(p->kpg, bk);
    return (double)(rocke_direct_conv_problem_Ho(p) * rocke_direct_conv_problem_Wo(p) * p->kpg)
           / (t0 * t1 * t2);
}

rocke_status_t rocke_fwd_tapouter_wmma_kernel_name(const rocke_fwd_tapouter_wmma_spec_t* spec,
                                                   char* out,
                                                   size_t out_cap)
{
    if(spec == NULL)
        return ROCKE_ERR_VALUE;
    const rocke_direct_conv_problem_t* p = &spec->problem;
    char f[32], pad[32], dil[48], wk[32], wp[32], tk[32], tp[32], tw[32], ck[32], st[32], ts[32];
    std::string sched;
    snprintf(f, sizeof f, "f%dx%d", p->KH, p->KW);
    snprintf(pad, sizeof pad, "p%d", p->PAD);
    snprintf(dil, sizeof dil, "dil%dx%d", p->dil_h, p->dil_w);
    snprintf(wk, sizeof wk, "wk%d", spec->waves_k);
    snprintf(wp, sizeof wp, "wp%d", spec->waves_p);
    snprintf(tk, sizeof tk, "tk%d", spec->tiles_k);
    snprintf(tp, sizeof tp, "tp%d", spec->tiles_p);
    snprintf(tw, sizeof tw, "tw%d", spec->tile_w);
    snprintf(ck, sizeof ck, "ck%d", spec->ck);
    snprintf(st, sizeof st, "st%d", spec->stages);
    if(spec->sched != NULL && spec->sched[0] != '\0')
        sched = std::string("s") + spec->sched;
    if(spec->tdm_split > 1)
        snprintf(ts, sizeof ts, "ts%d", spec->tdm_split);
    else
        ts[0] = '\0';
    const char* parts[]
        = {f, pad, dil, wk, wp, tk, tp, tw, ck, st, spec->prefetch ? "pf" : "", sched.c_str(), ts};
    const char* flag_names[] = {"bf16"};
    int flag_on[] = {rocke_direct_conv_problem_is_bf16(p) ? 1 : 0};
    return rocke_kernel_name_join(spec->name ? spec->name : "direct_fwd_tapouter_wmma",
                                  parts,
                                  sizeof(parts) / sizeof(parts[0]),
                                  flag_names,
                                  flag_on,
                                  1,
                                  out,
                                  out_cap,
                                  NULL);
}

bool rocke_fwd_tapouter_wmma_is_valid_spec(const rocke_fwd_tapouter_wmma_spec_t* spec,
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
    const rocke_direct_conv_problem_t* p = &spec->problem;
    const rocke_archtarget_t* target = NULL;
    if(!rocke_wmma_fwd_gate(p,
                            arch,
                            "direct_fwd_tapouter_wmma",
                            spec->wave_size,
                            false,
                            &target,
                            reason,
                            reason_cap))
        return false;
    if(p->stride != 1)
        ROCKE_SPEC_FAIL(
            reason, reason_cap, "stride %d: the tap windows are dense (stride 1 only)", p->stride);
    if(p->dil_h < 1 || p->dil_w < 1)
        ROCKE_SPEC_FAIL(reason, reason_cap, "dilation %dx%d < 1", p->dil_h, p->dil_w);
    if(p->PAD < 0)
        ROCKE_SPEC_FAIL(reason, reason_cap, "PAD %d < 0", p->PAD);
    const long long Ho = rocke_direct_conv_problem_Ho(p), Wo = rocke_direct_conv_problem_Wo(p);
    if(Ho < 1 || Wo < 1)
        ROCKE_SPEC_FAIL(reason, reason_cap, "empty output %lldx%lld", Ho, Wo);
    if(p->cpg % 8 != 0 || p->kpg % 8 != 0)
        ROCKE_SPEC_FAIL(reason,
                        reason_cap,
                        "cpg %d and kpg %d must be multiples of 8 (16-byte rows)",
                        p->cpg,
                        p->kpg);
    if(spec->ck % ROCKE_FWD_WMMA_K != 0 || spec->ck <= 0)
        ROCKE_SPEC_FAIL(reason,
                        reason_cap,
                        "ck %d must be a positive multiple of %d",
                        spec->ck,
                        ROCKE_FWD_WMMA_K);
    if(std::min(std::min(spec->waves_k, spec->waves_p), std::min(spec->tiles_k, spec->tiles_p)) < 1)
        ROCKE_SPEC_FAIL(reason, reason_cap, "waves and tiles must be >= 1");
    if(spec->waves_k * spec->waves_p < 2)
        ROCKE_SPEC_FAIL(
            reason, reason_cap, "needs at least two waves (one TDM issuer per operand)");
    if(spec->stages < 2)
        ROCKE_SPEC_FAIL(reason, reason_cap, "stages must be >= 2");
    const int ts = spec->tdm_split;
    if(ts < 1 || 2 * ts > spec->waves_k * spec->waves_p)
        ROCKE_SPEC_FAIL(reason, reason_cap, "tdm_split %d needs 2 * %d waves", ts, ts);
    if(rocke_fwd_tapouter_wmma_tile_h(spec) % ts != 0
       || rocke_fwd_tapouter_wmma_block_k(spec) % ts != 0)
        ROCKE_SPEC_FAIL(reason,
                        reason_cap,
                        "tdm_split %d must divide tile_h %d and block_k",
                        ts,
                        rocke_fwd_tapouter_wmma_tile_h(spec));
    /* A slab in the top padding issues no transfer, so a wave's tensor count
     * no longer marks its steps: only a full wait is safe. */
    if(ts > 1 && spec->stages > 2)
        ROCKE_SPEC_FAIL(
            reason, reason_cap, "tdm_split %d needs 2 stages (a skipped slab breaks the wait)", ts);
    if(spec->tile_w < 1 || rocke_fwd_tapouter_wmma_block_p(spec) % spec->tile_w != 0)
        ROCKE_SPEC_FAIL(reason,
                        reason_cap,
                        "tile_w %d must divide %d",
                        spec->tile_w,
                        rocke_fwd_tapouter_wmma_block_p(spec));
    if(rocke_fwd_tapouter_wmma_threads_per_block(spec)
       > rocke_archtarget_max_threads_per_block(target))
        ROCKE_SPEC_FAIL(reason,
                        reason_cap,
                        "%d threads per block",
                        rocke_fwd_tapouter_wmma_threads_per_block(spec));
    if(std::max(std::max(to_x_pitch(spec), to_epi_pitch(spec)),
                std::max(spec->tile_w, rocke_fwd_tapouter_wmma_block_k(spec)))
       > 0xFFFF)
        ROCKE_SPEC_FAIL(reason, reason_cap, "tile too wide for a TDM descriptor");
    if(!rocke_archtarget_fits_lds(target, rocke_fwd_tapouter_wmma_lds_bytes(spec)))
        ROCKE_SPEC_FAIL(reason,
                        reason_cap,
                        "LDS %ld B > %d",
                        rocke_fwd_tapouter_wmma_lds_bytes(spec),
                        target->lds_capacity_bytes);
    {
        const int vgprs = 8 * spec->tiles_k * spec->tiles_p
                          + 8 * (spec->tiles_k + spec->tiles_p) * (spec->prefetch ? 2 : 1);
        if(vgprs > ROCKE_FWD_TAPOUTER_VGPR_BUDGET)
            ROCKE_SPEC_FAIL(reason, reason_cap, "~%d VGPRs of accumulators and fragments", vgprs);
    }
    if(!rocke_wmma_fwd_limits_ok(p, spec->sched, to_grid0(spec), reason, reason_cap))
        return false;
    rocke_spec_set_reason(reason, reason_cap, "ok");
    return true;
}

/* ===================================================================== *
 *  build_direct_fwd_tapouter_wmma
 * ===================================================================== */
namespace
{

V ci(rocke_ir_builder_t* b, int64_t v)
{
    return rocke_b_const_i32(b, v);
}

template <typename Body>
void to_if(rocke_ir_builder_t* b, V cond, Body body)
{
    rocke_if_t g = rocke_b_scf_if(b, cond);
    rocke_b_region_enter(b, g.then_region);
    body();
    rocke_b_region_leave(b);
}

} // namespace

rocke_kernel_def_t* rocke_build_fwd_tapouter_wmma(rocke_ir_builder_t* b,
                                                  const rocke_fwd_tapouter_wmma_spec_t* spec,
                                                  const char* arch)
{
    const rocke_direct_conv_problem_t* p = &spec->problem;
    const int KH = p->KH, KW = p->KW;
    const int DH = p->dil_h, DW = p->dil_w;
    const int PAD = p->PAD;
    const char* wmma_op = rocke_wmma_fwd_op_id(
        b, rocke_direct_conv_problem_dtype(p), arch, "build_direct_fwd_tapouter_wmma");
    if(wmma_op == NULL)
        return NULL;

    const int WAVE = spec->wave_size;
    const int WK = spec->waves_k;
    const int TM = spec->tiles_k, TN = spec->tiles_p;
    const int THREADS = rocke_fwd_tapouter_wmma_threads_per_block(spec);
    const int BK = rocke_fwd_tapouter_wmma_block_k(spec);
    const int TH = rocke_fwd_tapouter_wmma_tile_h(spec), TW = spec->tile_w;
    const int CK = spec->ck;
    const int NS = spec->stages;
    const int XP = to_x_pitch(spec), EPITCH = to_epi_pitch(spec);
    const long WIN = to_win_elems(spec), STAGE = to_stage_elems(spec);
    const int TAPS = KH * KW;
    const int SPLIT = spec->tdm_split;
    const int XR = TH / SPLIT, WR = BK / SPLIT;

    const rocke_type_t* io_type = rocke_b_io_ir_type(b, rocke_direct_conv_problem_dtype(p));
    rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", THREADS);
    if(spec->sched != NULL && spec->sched[0] != '\0')
        rocke_attr_set_str(b, &b->kernel->attrs, "scheduler_strategy", spec->sched);
    rocke_dconv_params_t prm;
    rocke_dconv_emit_params(b, &prm, "fwd", io_type);
    if(!rocke_ir_builder_ok(b))
        return NULL;
    V Xp = prm.A, Wp = prm.Bp, Yp = prm.D;

    V c0 = ci(b, 0);
    V c1 = ci(b, 1);
    V c2 = ci(b, 2);
    V c8 = ci(b, 8);
    V c16 = ci(b, 16);
    V c_groups = prm.p_groups;
    V c_cpg = rocke_b_div(b, prm.p_total_c, c_groups);
    V c_kpg = rocke_b_div(b, prm.p_total_k, c_groups);
    V c_H = prm.p_Hi, c_W = prm.p_Wi;
    V c_Ho = prm.p_Ho, c_Wo = prm.p_Wo;
    V xs_n = prm.p_A_stride_n, xs_h = prm.p_A_stride_hi, xs_w = prm.p_A_stride_wi;
    V ys_n = prm.p_D_stride_n, ys_h = prm.p_D_stride_ho, ys_w = prm.p_D_stride_wo;

    V tid = rocke_b_thread_id_x(b);
    V c_wave = ci(b, WAVE);
    V wave_id = rocke_b_div(b, tid, c_wave);
    V lane = rocke_b_mod(b, tid, c_wave);
    V l16 = rocke_b_mod(b, lane, c16);
    V lhalf = rocke_b_div(b, lane, c16);
    V wave_k = rocke_b_mod(b, wave_id, ci(b, WK));
    V wave_p = rocke_b_div(b, wave_id, ci(b, WK));

    /* ---- tile of this block: bx = (((n * G + g) * n_ht + ht) * n_wt + wt) * n_kt + kt. */
    auto ceil_tiles = [&](V ext, int tile) -> V {
        V x = rocke_b_add(b, ext, ci(b, tile - 1));
        return rocke_b_div(b, x, ci(b, tile));
    };
    V n_kt = ceil_tiles(c_kpg, BK);
    V n_ht = ceil_tiles(c_Ho, TH);
    V n_wt = ceil_tiles(c_Wo, TW);
    V t = rocke_b_block_id_x(b);
    V kt = rocke_b_mod(b, t, n_kt);
    t = rocke_b_div(b, t, n_kt);
    V wt = rocke_b_mod(b, t, n_wt);
    t = rocke_b_div(b, t, n_wt);
    V ht = rocke_b_mod(b, t, n_ht);
    t = rocke_b_div(b, t, n_ht);
    V g = rocke_b_mod(b, t, c_groups);
    V n_img = rocke_b_div(b, t, c_groups);
    V k_base = rocke_b_mul(b, kt, ci(b, BK));
    V h_tile = rocke_b_mul(b, ht, ci(b, TH));
    V w_tile = rocke_b_mul(b, wt, ci(b, TW));
    V k_grp = rocke_b_mul(b, g, c_kpg);
    V c_grp = rocke_b_mul(b, g, c_cpg);
    V w_row = rocke_b_mul(b, c_cpg, ci(b, TAPS));
    V c_CK = ci(b, CK);
    V n_chunks;
    {
        V x = rocke_b_add(b, c_cpg, ci(b, CK - 1));
        n_chunks = rocke_b_div(b, x, c_CK);
    }
    V k_ext;
    {
        V x = rocke_b_sub(b, c_kpg, k_base);
        V y = rocke_b_smin(b, x, ci(b, BK));
        k_ext = rocke_b_smax(b, y, c0);
    }

    /* ---- taps whose window meets the image. */
    V oh0 = rocke_b_sub(b, h_tile, ci(b, PAD));
    V ow0 = rocke_b_sub(b, w_tile, ci(b, PAD));
    V th_eff;
    {
        V x = rocke_b_sub(b, c_Ho, h_tile);
        th_eff = rocke_b_smin(b, x, ci(b, TH));
    }
    V tw_eff;
    {
        V x = rocke_b_sub(b, c_Wo, w_tile);
        tw_eff = rocke_b_smin(b, x, ci(b, TW));
    }
    auto tap_range = [&](V o0, V ext, V lim, int nk, int dil, V* first, V* count) {
        V before = c0;
        V after = c0;
        for(int r = 0; r < nk; ++r)
        {
            V lo = r ? rocke_b_add(b, o0, ci(b, (int64_t)r * dil)) : o0;
            {
                V x = rocke_b_add(b, lo, ext);
                V le = rocke_b_cmp_le(b, x, c0);
                V s = rocke_b_select(b, le, c1, c0);
                before = rocke_b_add(b, before, s);
            }
            {
                V ge = rocke_b_cmp_ge(b, lo, lim);
                V s = rocke_b_select(b, ge, c1, c0);
                after = rocke_b_add(b, after, s);
            }
        }
        *first = before;
        V cn = ci(b, nk);
        V ba = rocke_b_add(b, before, after);
        V d = rocke_b_sub(b, cn, ba);
        *count = rocke_b_smax(b, d, c0);
    };
    V r_lo, n_r, s_lo, n_s;
    tap_range(oh0, th_eff, c_H, KH, DH, &r_lo, &n_r);
    tap_range(ow0, tw_eff, c_W, KW, DW, &s_lo, &n_s);
    V n_taps = rocke_b_mul(b, n_r, n_s);
    V n_steps = rocke_b_mul(b, n_chunks, n_taps);
    V n_taps1 = rocke_b_smax(b, n_taps, c1);
    V n_s1 = rocke_b_smax(b, n_s, c1);

    const int lds_shape[2] = {1, (int)(rocke_fwd_tapouter_wmma_lds_bytes(spec) / 2)};
    V lds = rocke_b_smem_alloc(b, io_type, lds_shape, 2, "ft_lds");
    V lds_addr;
    {
        V a = rocke_b_smem_addr_of(b, lds);
        lds_addr = rocke_b_trunc(b, a, rocke_i32());
    }
    V x_img;
    {
        V x = rocke_b_mul(b, n_img, xs_n);
        x_img = rocke_b_add(b, x, c_grp);
    }
    V w_blk;
    {
        V x = rocke_b_add(b, k_grp, k_base);
        w_blk = rocke_b_mul(b, x, w_row);
    }
    const long max_z = ((long)(TH - 1) * TW + TW - 1) * XP;
    V zero_v = rocke_b_zero_vec(b, io_type, 8);

    auto issue = [&](V u) {
        V chunk = rocke_b_div(b, u, n_taps1);
        V ti;
        {
            V x = rocke_b_mul(b, chunk, n_taps1);
            ti = rocke_b_sub(b, u, x);
        }
        V rr = rocke_b_div(b, ti, n_s1);
        V r = rocke_b_add(b, r_lo, rr);
        V s;
        {
            V x = rocke_b_mul(b, rr, n_s1);
            V y = rocke_b_sub(b, ti, x);
            s = rocke_b_add(b, s_lo, y);
        }
        V k0 = rocke_b_mul(b, chunk, c_CK);
        V c_ext;
        {
            V x = rocke_b_sub(b, c_cpg, k0);
            V y = rocke_b_smin(b, x, c_CK);
            c_ext = rocke_b_smax(b, y, c0);
        }
        V slot;
        {
            V x = rocke_b_mod(b, u, ci(b, NS));
            slot = rocke_b_mul(b, x, ci(b, STAGE));
        }
        V ih0;
        {
            V x = rocke_b_mul(b, r, ci(b, DH));
            ih0 = rocke_b_add(b, oh0, x);
        }
        V iw0;
        {
            V x = rocke_b_mul(b, s, ci(b, DW));
            iw0 = rocke_b_add(b, ow0, x);
        }
        V hs;
        {
            V x = rocke_b_sub(b, c0, ih0);
            V y = rocke_b_smax(b, x, c0);
            hs = rocke_b_smin(b, y, ci(b, TH));
        }
        V ws;
        {
            V x = rocke_b_sub(b, c0, iw0);
            V y = rocke_b_smax(b, x, c0);
            ws = rocke_b_smin(b, y, ci(b, TW));
        }
        V h_start = rocke_b_smax(b, ih0, c0);
        V w_start = rocke_b_smax(b, iw0, c0);
        V h_ext;
        {
            V x = rocke_b_sub(b, c_H, h_start);
            h_ext = rocke_b_smax(b, x, c0);
        }
        (void)h_ext;
        V w_ext;
        {
            V x = rocke_b_sub(b, c_W, w_start);
            V y = rocke_b_sub(b, ci(b, TW), ws);
            V z = rocke_b_smin(b, x, y);
            w_ext = rocke_b_smax(b, z, c0);
        }
        V z_elems;
        {
            V x = rocke_b_mul(b, hs, ci(b, TW));
            V y = rocke_b_add(b, x, ws);
            z_elems = rocke_b_mul(b, y, ci(b, XP));
        }
        if(max_z)
        {
            V z_vecs = rocke_b_div(b, z_elems, c8);
            V any = rocke_b_cmp_lt(b, c0, z_vecs);
            to_if(b, any, [&]() {
                for(long j = 0; j < (max_z + 7) / 8; j += THREADS)
                {
                    V v = rocke_b_add(b, tid, ci(b, j));
                    V lt = rocke_b_cmp_lt(b, v, z_vecs);
                    to_if(b, lt, [&]() {
                        V m = rocke_b_mul(b, v, c8);
                        V off = rocke_b_add(b, slot, m);
                        V idx[2] = {c0, off};
                        rocke_b_smem_store_vN(b, lds, idx, 2, zero_v, 8);
                    });
                }
            });
        }
        V x_row0;
        {
            V x = rocke_b_add(b, x_img, k0);
            V y = rocke_b_mul(b, w_start, xs_w);
            x_row0 = rocke_b_add(b, x, y);
        }
        V tap;
        {
            V x = rocke_b_mul(b, r, ci(b, KW));
            tap = rocke_b_add(b, x, s);
        }
        V w_off0;
        {
            V x = rocke_b_mul(b, tap, c_cpg);
            V y = rocke_b_add(b, x, k0);
            w_off0 = rocke_b_add(b, w_blk, y);
        }
        for(int j = 0; j < SPLIT; ++j)
        {
            const int a = j * XR;
            V start = rocke_b_smax(b, ci(b, a), hs);
            V rows = rocke_b_sub(b, ci(b, a + XR), start);
            V ok;
            {
                V x = rocke_b_cmp_eq(b, wave_id, ci(b, 2 * j));
                V y = rocke_b_cmp_lt(b, c0, rows);
                ok = rocke_b_land(b, x, y);
            }
            to_if(b, ok, [&]() {
                V ih = rocke_b_add(b, ih0, start);
                V src;
                {
                    V x = rocke_b_mul(b, ih, xs_h);
                    V y = rocke_b_add(b, x_row0, x);
                    V z = rocke_b_mul(b, y, c2);
                    src = rocke_b_global_ptr_add(b, Xp, z);
                }
                if(SPLIT > 1)
                {
                    V lt = rocke_b_cmp_lt(b, ih, c_H);
                    V base = rocke_b_global_ptr_add(b, Xp, c0);
                    src = rocke_b_select(b, lt, src, base);
                }
                V cell;
                {
                    V x = rocke_b_mul(b, start, ci(b, TW));
                    cell = rocke_b_add(b, x, ws);
                }
                V dst;
                {
                    V x = rocke_b_mul(b, cell, ci(b, XP));
                    V y = rocke_b_add(b, slot, x);
                    V z = rocke_b_mul(b, y, c2);
                    dst = rocke_b_add(b, lds_addr, z);
                }
                V d2;
                {
                    V x = rocke_b_sub(b, c_H, ih);
                    d2 = rocke_b_smax(b, x, c0);
                }
                V d[5];
                rocke_fwd_tdm_descriptor_3d(
                    b, src, dst, c_ext, w_ext, d2, xs_w, xs_h, XP, TW, rows, d);
                rocke_b_tensor_load_to_lds(b, d[0], d[1], d[2], d[3], d[4], 0);
            });
            V is_w = rocke_b_cmp_eq(b, wave_id, ci(b, 2 * j + 1));
            to_if(b, is_w, [&]() {
                V off;
                {
                    V x = rocke_b_mul(b, ci(b, (int64_t)j * WR), w_row);
                    off = rocke_b_add(b, w_off0, x);
                }
                V src;
                {
                    V x = rocke_b_mul(b, off, c2);
                    src = rocke_b_global_ptr_add(b, Wp, x);
                }
                V dst;
                {
                    V x = rocke_b_add(b, slot, ci(b, WIN + (long)j * WR * XP));
                    V y = rocke_b_mul(b, x, c2);
                    dst = rocke_b_add(b, lds_addr, y);
                }
                V k_ext_j;
                {
                    V x = rocke_b_sub(b, k_ext, ci(b, (int64_t)j * WR));
                    V y = rocke_b_smin(b, x, ci(b, WR));
                    k_ext_j = rocke_b_smax(b, y, c0);
                }
                V d[5];
                rocke_fwd_tdm_descriptor_3d(
                    b, src, dst, c_ext, k_ext_j, c1, w_row, w_row, XP, WR, c1, d);
                rocke_b_tensor_load_to_lds(b, d[0], d[1], d[2], d[3], d[4], 0);
            });
        }
    };

    /* ---- per-lane LDS read offsets (within a slot). */
    auto tile_row = [&](V wave, int tiles, int t, V l) -> V {
        V x = rocke_b_mul(b, wave, ci(b, tiles));
        V y = rocke_b_add(b, x, ci(b, t));
        V z = rocke_b_mul(b, y, c16);
        return rocke_b_add(b, z, l);
    };
    std::vector<V> px_off;
    for(int tn = 0; tn < TN; ++tn)
    {
        V px = tile_row(wave_p, TN, tn, l16);
        V x = rocke_b_mul(b, px, ci(b, XP));
        V y = rocke_b_mul(b, lhalf, c16);
        px_off.push_back(rocke_b_add(b, x, y));
    }
    std::vector<V> wei_off;
    for(int m = 0; m < TM; ++m)
    {
        V row = tile_row(wave_k, TM, m, l16);
        V x = rocke_b_mul(b, row, ci(b, XP));
        V y = rocke_b_mul(b, lhalf, c16);
        V z = rocke_b_add(b, y, ci(b, WIN));
        wei_off.push_back(rocke_b_add(b, x, z));
    }

    auto read16 = [&](V base, V off, int const_off) -> V {
        return rocke_fwd_lds_frag16(b, lds, base, off, const_off, c0, c8, io_type);
    };
    auto frags = [&](V base, int ks, std::vector<V>* af, std::vector<V>* bf) {
        af->clear();
        bf->clear();
        for(int m = 0; m < TM; ++m)
            af->push_back(read16(base, wei_off[m], ks * ROCKE_FWD_WMMA_K));
        for(int tn = 0; tn < TN; ++tn)
            bf->push_back(read16(base, px_off[tn], ks * ROCKE_FWD_WMMA_K));
    };

    const int KSTEPS = CK / ROCKE_FWD_WMMA_K;

    auto step = [&](V u, std::vector<V> accs) -> std::vector<V> {
        rocke_fwd_tdm_ring_wait(b, u, NS, n_steps);
        rocke_b_sync_lds_only(b);
        V nxt = rocke_b_add(b, u, ci(b, NS - 1));
        V lt = rocke_b_cmp_lt(b, nxt, n_steps);
        to_if(b, lt, [&]() { issue(nxt); });
        V base;
        {
            V x = rocke_b_mod(b, u, ci(b, NS));
            base = rocke_b_mul(b, x, ci(b, STAGE));
        }
        rocke_b_s_setprio(b, 1);
        std::vector<V> naf, nbf;
        if(spec->prefetch)
            frags(base, 0, &naf, &nbf);
        for(int ks = 0; ks < KSTEPS; ++ks)
        {
            std::vector<V> af, bf;
            if(spec->prefetch)
            {
                af = naf;
                bf = nbf;
                if(ks + 1 < KSTEPS)
                    frags(base, ks + 1, &naf, &nbf);
            }
            else
                frags(base, ks, &af, &bf);
            for(int m = 0; m < TM; ++m)
                for(int tn = 0; tn < TN; ++tn)
                {
                    const int i = m * TN + tn;
                    accs[i] = rocke_b_mma(b, wmma_op, af[m], bf[tn], accs[i], NULL, 0);
                }
        }
        rocke_b_s_setprio(b, 0);
        return accs;
    };

    V k_out = rocke_b_add(b, k_grp, k_base);

    auto epilogue = [&](const std::vector<V>& accs) {
        rocke_b_sync_lds_only(b);
        for(int m = 0; m < TM; ++m)
        {
            V col;
            {
                V x = rocke_b_mul(b, wave_k, ci(b, TM));
                V y = rocke_b_add(b, x, ci(b, m));
                V z = rocke_b_mul(b, y, c16);
                V u = rocke_b_mul(b, lhalf, c8);
                col = rocke_b_add(b, z, u);
            }
            for(int tn = 0; tn < TN; ++tn)
            {
                V px = tile_row(wave_p, TN, tn, l16);
                V val = rocke_b_vec_cast_f32_to(b, accs[m * TN + tn], io_type);
                V x = rocke_b_mul(b, px, ci(b, EPITCH));
                V y = rocke_b_add(b, x, col);
                V idx[2] = {c0, y};
                rocke_b_smem_store_vN(b, lds, idx, 2, val, 8);
            }
        }
        rocke_b_sync_lds_only(b);
        V w0 = rocke_b_cmp_eq(b, wave_id, c0);
        to_if(b, w0, [&]() {
            V off;
            {
                V m1 = rocke_b_mul(b, n_img, ys_n);
                V m2 = rocke_b_mul(b, h_tile, ys_h);
                V a1 = rocke_b_add(b, m1, m2);
                V m3 = rocke_b_mul(b, w_tile, ys_w);
                V a2 = rocke_b_add(b, m3, k_out);
                off = rocke_b_add(b, a1, a2);
            }
            V dst;
            {
                V x = rocke_b_mul(b, off, c2);
                dst = rocke_b_global_ptr_add(b, Yp, x);
            }
            V d1;
            {
                V x = rocke_b_sub(b, c_Wo, w_tile);
                d1 = rocke_b_smax(b, x, c0);
            }
            V d2;
            {
                V x = rocke_b_sub(b, c_Ho, h_tile);
                d2 = rocke_b_smax(b, x, c0);
            }
            V tile2 = ci(b, TH);
            V d[5];
            rocke_fwd_tdm_descriptor_3d(
                b, dst, lds_addr, k_ext, d1, d2, ys_w, ys_h, EPITCH, TW, tile2, d);
            rocke_b_tensor_store_from_lds(b, d[0], d[1], d[2], d[3], d[4], 0);
            rocke_b_s_wait_tensorcnt(b, 0);
        });
    };

    /* ---- main loop over (chunk, tap) steps. */
    for(int st = 0; st < NS - 1; ++st)
    {
        V cst = ci(b, st);
        V lt = rocke_b_cmp_lt(b, cst, n_steps);
        to_if(b, lt, [&]() { issue(ci(b, st)); });
    }
    V zero_acc = rocke_b_zero_vec_f32(b, 8);
    std::vector<std::string> names(TM * TN);
    std::vector<rocke_iter_arg_t> iargs(TM * TN);
    for(int i = 0; i < TM * TN; ++i)
    {
        names[i] = "acc" + std::to_string(i);
        iargs[i].name = names[i].c_str();
        iargs[i].init = zero_acc;
    }
    rocke_for_t loop = rocke_b_scf_for_iter(
        b, c0, n_steps, c1, iargs.data(), (int)iargs.size(), "u", false, true);
    if(!rocke_ir_builder_ok(b))
        return NULL;
    rocke_b_region_enter(b, loop.body);
    {
        std::vector<V> carried(loop.iter_vars, loop.iter_vars + loop.num_iter_vars);
        std::vector<V> out = step(loop.iv, carried);
        rocke_b_scf_yield(b, out.data(), (int)out.size());
    }
    rocke_b_region_leave(b);
    if(!rocke_ir_builder_ok(b))
        return NULL;
    std::vector<V> results(loop.op->results, loop.op->results + TM * TN);
    epilogue(results);
    if(!rocke_ir_builder_ok(b))
        return NULL;
    return b->kernel;
}

rocke_kernel_def_t* rocke_build_fwd_tapouter_wmma_new(rocke_ir_builder_t* b,
                                                      const rocke_fwd_tapouter_wmma_spec_t* spec,
                                                      const char* arch)
{
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        char name[512];
        char why[512];
        if(b == NULL || spec == NULL)
            return NULL;
        if(arch == NULL)
            arch = "gfx1250";
        if(!rocke_fwd_tapouter_wmma_is_valid_spec(spec, arch, why, sizeof why))
        {
            rocke_ir_builder_init(b, "direct_fwd_tapouter_wmma_invalid");
            rocke_i_set_err(
                b, ROCKE_ERR_VALUE, "invalid DirectFwdTapOuterSpec for %s: %s", arch, why);
            return NULL;
        }
        if(rocke_fwd_tapouter_wmma_kernel_name(spec, name, sizeof name) != ROCKE_OK)
            return NULL;
        if(rocke_ir_builder_init(b, name) != ROCKE_OK)
            return NULL;
        return rocke_build_fwd_tapouter_wmma(b, spec, arch);
    });
}

rocke_status_t rocke_fwd_tapouter_wmma_lower_to_llvm(const rocke_fwd_tapouter_wmma_spec_t* spec,
                                                     const char* arch,
                                                     rocke_llvm_flavor_t flavor,
                                                     char** out_ll,
                                                     char* err,
                                                     size_t err_cap)
{
    rocke_ir_builder_t b = {};
    if(out_ll != NULL)
        *out_ll = NULL;
    if(spec == NULL || out_ll == NULL)
    {
        rocke_spec_set_reason(err, err_cap, "lower_to_llvm: null spec/out");
        return ROCKE_ERR_VALUE;
    }
    if(arch == NULL)
        arch = "gfx1250";
    rocke_kernel_def_t* kernel = rocke_build_fwd_tapouter_wmma_new(&b, spec, arch);
    if(kernel == NULL)
    {
        const char* m = rocke_ir_builder_error(&b);
        rocke_status_t st = rocke_ir_builder_status(&b);
        rocke_spec_set_reason(
            err,
            err_cap,
            (m != NULL && m[0] != '\0') ? m : "build_direct_fwd_tapouter_wmma failed");
        rocke_ir_builder_free(&b);
        return (st == ROCKE_OK) ? ROCKE_ERR_VALUE : st;
    }
    rocke_status_t st = rocke_lower_kernel_to_llvm_ex(kernel, flavor, arch, out_ll, err, err_cap);
    rocke_ir_builder_free(&b);
    return st;
}
