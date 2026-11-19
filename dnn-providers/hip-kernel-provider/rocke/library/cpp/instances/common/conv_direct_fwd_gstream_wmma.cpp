// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * conv_direct_fwd_gstream_wmma.cpp -- C++ port of the small-group forward that
 * walks row bands (library/kernels/common/conv_direct_fwd_gstream_wmma.py):
 *
 *   Python                                    C++ (this file)
 *   FwdGstreamWmmaSpec                     -> rocke_fwd_gstream_wmma_spec_t
 *   fwd_gstream_reason / is_valid_...      -> rocke_fwd_gstream_wmma_is_valid_spec()
 *   kernel_name()                          -> rocke_fwd_gstream_wmma_kernel_name()
 *   build_fwd_gstream_wmma(spec, arch)     -> rocke_build_fwd_gstream_wmma_new()
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
#include "rocke/instance_conv_direct_fwd_gstream_wmma.h"
#include "rocke/instance_conv_direct_grouped.h"
#include "rocke/instance_conv_direct_grouped_internal.h" /* rocke_dconv_emit_params */
#include "rocke/instance_conv_fwd_wmma_gate_internal.h" /* rocke_wmma_fwd_gate */
#include "rocke/instance_conv_fwd_wmma_tdm_internal.h" /* rocke_fwd_tdm_descriptor_3d */
#include "rocke/ir.h"
#include "rocke/ir_internal.h" /* rocke_i_set_err */

typedef rocke_value_t* V;

namespace
{

bool gs_span16(const rocke_fwd_gstream_wmma_spec_t* s)
{
    return s->problem.cpg <= 16;
}

bool gs_share_b(const rocke_fwd_gstream_wmma_spec_t* s)
{
    const int kpad = rocke_fwd_gstream_wmma_kpad(s);
    return !gs_span16(s) && s->tiles_k > 0 && kpad % (s->tiles_k * 16) == 0;
}

int gs_block_p(const rocke_fwd_gstream_wmma_spec_t* s)
{
    return s->waves_p * s->tiles_p * 16;
}

int gs_halo_h(const rocke_fwd_gstream_wmma_spec_t* s)
{
    const rocke_direct_conv_problem_t* p = &s->problem;
    return rocke_wmma_fwd_halo_extent(p->stride, rocke_fwd_gstream_wmma_tile_h(s), p->KH, p->dil_h);
}

int gs_halo_w(const rocke_fwd_gstream_wmma_spec_t* s)
{
    const rocke_direct_conv_problem_t* p = &s->problem;
    return rocke_wmma_fwd_halo_extent(p->stride, s->tile_w, p->KW, p->dil_w);
}

int gs_phases_w(const rocke_fwd_gstream_wmma_spec_t* s)
{
    return s->phase_split ? s->problem.stride : 1;
}

int gs_halo_cols(const rocke_fwd_gstream_wmma_spec_t* s)
{
    return rocke_wmma_fwd_halo_cols(&s->problem, s->tile_w, s->phase_split);
}

int gs_lead_cols(const rocke_fwd_gstream_wmma_spec_t* s)
{
    return rocke_wmma_fwd_lead_cols(s->problem.PAD, s->problem.stride, s->phase_split);
}

int gs_x_pitch(const rocke_fwd_gstream_wmma_spec_t* s)
{
    return rocke_fwd_gstream_wmma_block_c(s) + ROCKE_FWD_WMMA_CH_PAD;
}

long gs_phase_elems(const rocke_fwd_gstream_wmma_spec_t* s)
{
    const long xp = gs_x_pitch(s);
    return (long)gs_halo_h(s) * gs_halo_cols(s) * xp + (long)gs_lead_cols(s) * xp;
}

long gs_slot_elems(const rocke_fwd_gstream_wmma_spec_t* s)
{
    if(s->phase_split)
        return (long)gs_phases_w(s) * gs_phase_elems(s);
    return ((long)gs_halo_h(s) * gs_halo_w(s) + s->problem.PAD) * gs_x_pitch(s);
}

struct gs_kstep
{
    int t0, t1, cc;
};

/* k_steps(): (tap of lanes 0-15, tap of lanes 16-31, 32-channel chunk). */
std::vector<gs_kstep> gs_k_steps(const rocke_fwd_gstream_wmma_spec_t* s)
{
    const int T = s->problem.KH * s->problem.KW;
    std::vector<gs_kstep> out;
    if(gs_span16(s))
    {
        for(int j = 0; j < T; j += 2)
            out.push_back({j, j + 1 < T ? j + 1 : -1, 0});
        return out;
    }
    const int nc = (int)rocke_ceil_div(s->problem.cpg, ROCKE_FWD_WMMA_K);
    for(int t = 0; t < T; ++t)
        for(int c = 0; c < nc; ++c)
            out.push_back({t, t, c});
    return out;
}

long long gs_n_bands(const rocke_fwd_gstream_wmma_spec_t* s)
{
    return rocke_ceil_div(rocke_direct_conv_problem_Ho(&s->problem),
                          rocke_fwd_gstream_wmma_tile_h(s));
}

long long gs_grid0(const rocke_fwd_gstream_wmma_spec_t* s)
{
    const rocke_direct_conv_problem_t* p = &s->problem;
    const long long n_kb
        = (long long)p->groups * rocke_fwd_gstream_wmma_kpad(s) / rocke_fwd_gstream_wmma_block_k(s);
    const long long n_wt = rocke_ceil_div(rocke_direct_conv_problem_Wo(p), s->tile_w);
    const long long walkers = rocke_ceil_div((long long)p->N * gs_n_bands(s), s->walk);
    return n_kb * n_wt * walkers;
}

} // namespace

/* ===================================================================== *
 *  Spec helpers
 * ===================================================================== */
rocke_fwd_gstream_wmma_spec_t rocke_fwd_gstream_wmma_spec_default(void)
{
    rocke_fwd_gstream_wmma_spec_t s;
    memset(&s, 0, sizeof s);
    s.problem = rocke_direct_conv_problem_default();
    s.name = "direct_fwd_gstream_wmma";
    s.waves_k = 1;
    s.waves_p = 4;
    s.tiles_k = 1;
    s.tiles_p = 2;
    s.tile_w = 16;
    s.walk = 8;
    s.stages = 2;
    s.phase_split = false;
    s.prefetch = true;
    s.pf_depth = 1;
    s.sched = NULL;
    s.wave_size = 32;
    return s;
}

int rocke_fwd_gstream_wmma_kpad(const rocke_fwd_gstream_wmma_spec_t* spec)
{
    const int kpg = spec->problem.kpg;
    return gs_span16(spec) ? kpg : (int)rocke_align_up(kpg, 16);
}

int rocke_fwd_gstream_wmma_block_k(const rocke_fwd_gstream_wmma_spec_t* spec)
{
    return spec->waves_k * spec->tiles_k * 16;
}

int rocke_fwd_gstream_wmma_block_c(const rocke_fwd_gstream_wmma_spec_t* spec)
{
    const int kpad = rocke_fwd_gstream_wmma_kpad(spec);
    if(kpad <= 0)
        return 0; /* cpg == kpg < 1 is rejected */
    const int bk = rocke_fwd_gstream_wmma_block_k(spec);
    return (int)rocke_floor_div((long long)std::max(bk, kpad) * spec->problem.cpg, kpad);
}

int rocke_fwd_gstream_wmma_tile_h(const rocke_fwd_gstream_wmma_spec_t* spec)
{
    return spec->tile_w > 0 ? (int)rocke_floor_div(gs_block_p(spec), spec->tile_w) : 0;
}

int rocke_fwd_gstream_wmma_threads_per_block(const rocke_fwd_gstream_wmma_spec_t* spec)
{
    return spec->waves_k * spec->waves_p * spec->wave_size;
}

long rocke_fwd_gstream_wmma_lds_bytes(const rocke_fwd_gstream_wmma_spec_t* spec)
{
    return 2L * spec->stages * gs_slot_elems(spec);
}

void rocke_fwd_gstream_wmma_grid(const rocke_fwd_gstream_wmma_spec_t* spec, int grid[3])
{
    const long long g = gs_grid0(spec);
    grid[0] = (int)(g > INT32_MAX ? INT32_MAX : g);
    grid[1] = 1;
    grid[2] = 1;
}

double rocke_fwd_gstream_wmma_tile_efficiency(const rocke_fwd_gstream_wmma_spec_t* spec)
{
    const rocke_direct_conv_problem_t* p = &spec->problem;
    const long long hp = gs_n_bands(spec) * rocke_fwd_gstream_wmma_tile_h(spec);
    const long long wp = rocke_align_up(rocke_direct_conv_problem_Wo(p), spec->tile_w);
    return (double)(rocke_direct_conv_problem_Ho(p) * rocke_direct_conv_problem_Wo(p))
           / (double)(hp * wp);
}

rocke_status_t rocke_fwd_gstream_wmma_kernel_name(const rocke_fwd_gstream_wmma_spec_t* spec,
                                                  char* out,
                                                  size_t out_cap)
{
    if(spec == NULL)
        return ROCKE_ERR_VALUE;
    const rocke_direct_conv_problem_t* p = &spec->problem;
    char f[32], pad[32], str[32], ck[48], wk[32], wp[32], tk[32], tp[32], tw[32], wa[32], st[32],
        pd[32];
    std::string sched;
    snprintf(f, sizeof f, "f%dx%d", p->KH, p->KW);
    snprintf(pad, sizeof pad, "p%d", p->PAD);
    if(p->stride > 1)
        snprintf(str, sizeof str, "str%d", p->stride);
    else
        str[0] = '\0';
    snprintf(ck, sizeof ck, "c%dk%d", p->cpg, p->kpg);
    snprintf(wk, sizeof wk, "wk%d", spec->waves_k);
    snprintf(wp, sizeof wp, "wp%d", spec->waves_p);
    snprintf(tk, sizeof tk, "tk%d", spec->tiles_k);
    snprintf(tp, sizeof tp, "tp%d", spec->tiles_p);
    snprintf(tw, sizeof tw, "tw%d", spec->tile_w);
    snprintf(wa, sizeof wa, "wa%d", spec->walk);
    snprintf(st, sizeof st, "st%d", spec->stages);
    if(spec->prefetch && spec->pf_depth > 1)
        snprintf(pd, sizeof pd, "pd%d", spec->pf_depth);
    else
        pd[0] = '\0';
    if(spec->sched != NULL && spec->sched[0] != '\0')
        sched = std::string("s") + spec->sched;
    const char* parts[] = {f,
                           pad,
                           str,
                           ck,
                           wk,
                           wp,
                           tk,
                           tp,
                           tw,
                           wa,
                           st,
                           spec->phase_split ? "ps" : "",
                           spec->prefetch ? "pf" : "",
                           pd,
                           sched.c_str()};
    const char* flag_names[] = {"bf16"};
    int flag_on[] = {rocke_direct_conv_problem_is_bf16(p) ? 1 : 0};
    return rocke_kernel_name_join(spec->name ? spec->name : "direct_fwd_gstream_wmma",
                                  parts,
                                  sizeof(parts) / sizeof(parts[0]),
                                  flag_names,
                                  flag_on,
                                  1,
                                  out,
                                  out_cap,
                                  NULL);
}

bool rocke_fwd_gstream_wmma_is_valid_spec(const rocke_fwd_gstream_wmma_spec_t* spec,
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
    if(!rocke_wmma_fwd_gate(
           p, arch, "direct_fwd_gstream_wmma", spec->wave_size, false, &target, reason, reason_cap))
        return false;
    if(p->stride != 1 && p->stride != 2)
        ROCKE_SPEC_FAIL(reason, reason_cap, "stride %d not in (1, 2)", p->stride);
    if(p->dil_h != 1 || p->dil_w != 1)
        ROCKE_SPEC_FAIL(reason, reason_cap, "dilation is not served");
    if(p->KH != p->KW)
        ROCKE_SPEC_FAIL(reason, reason_cap, "filter %dx%d is not square", p->KH, p->KW);
    if(p->PAD < 0 || p->PAD > p->KH - 1)
        ROCKE_SPEC_FAIL(reason, reason_cap, "PAD %d outside 0..K-1", p->PAD);
    const long long Ho = rocke_direct_conv_problem_Ho(p), Wo = rocke_direct_conv_problem_Wo(p);
    if(Ho < 1 || Wo < 1)
        ROCKE_SPEC_FAIL(reason, reason_cap, "empty output %lldx%lld", Ho, Wo);
    if(p->cpg != p->kpg)
        ROCKE_SPEC_FAIL(reason, reason_cap, "cpg %d != kpg %d", p->cpg, p->kpg);
    if(p->cpg != 8 && p->cpg != 16 && (p->cpg < ROCKE_FWD_WMMA_K || p->cpg % 8 != 0))
        ROCKE_SPEC_FAIL(reason, reason_cap, "cpg %d not 8, 16 or a multiple of 8 from 32", p->cpg);
    if(p->cpg > 16 && p->cpg % ROCKE_FWD_WMMA_K != 0
       && rocke_align_up(p->cpg, ROCKE_FWD_WMMA_K) - p->cpg > ROCKE_FWD_WMMA_CH_PAD)
        ROCKE_SPEC_FAIL(
            reason, reason_cap, "cpg %d: its last k-step reads past the LDS row skew", p->cpg);
    if(std::min(std::min(spec->waves_k, spec->waves_p), std::min(spec->tiles_k, spec->tiles_p)) < 1)
        ROCKE_SPEC_FAIL(reason, reason_cap, "waves and tiles must be >= 1");
    const int kpad = rocke_fwd_gstream_wmma_kpad(spec);
    const int bk = rocke_fwd_gstream_wmma_block_k(spec);
    if(bk % kpad != 0 && kpad % bk != 0)
        ROCKE_SPEC_FAIL(
            reason, reason_cap, "block_k %d is neither whole groups nor part of one", bk);
    if(((long long)p->groups * kpad) % bk != 0)
        ROCKE_SPEC_FAIL(reason,
                        reason_cap,
                        "K %lld not a multiple of block_k %d",
                        rocke_direct_conv_problem_total_k(p),
                        bk);
    if(spec->pf_depth < 1 || (spec->pf_depth > 1 && !spec->prefetch))
        ROCKE_SPEC_FAIL(
            reason, reason_cap, "pf_depth %d: >= 1, and > 1 only with prefetch", spec->pf_depth);
    if(spec->stages < 2 || spec->walk < 1)
        ROCKE_SPEC_FAIL(reason, reason_cap, "stages must be >= 2 and walk >= 1");
    const int tw = spec->tile_w;
    if(tw < 1 || gs_block_p(spec) % tw != 0)
        ROCKE_SPEC_FAIL(reason,
                        reason_cap,
                        "tile_w %d must divide the %d pixels of a band",
                        tw,
                        gs_block_p(spec));
    if(spec->phase_split)
    {
        if(p->stride != 2)
            ROCKE_SPEC_FAIL(reason, reason_cap, "phase_split serves stride 2 only");
        if(spec->waves_k * spec->waves_p < 2)
            ROCKE_SPEC_FAIL(reason, reason_cap, "phase_split needs two waves (one transfer each)");
    }
    if(rocke_fwd_gstream_wmma_threads_per_block(spec)
       > rocke_archtarget_max_threads_per_block(target))
        ROCKE_SPEC_FAIL(reason,
                        reason_cap,
                        "%d threads per block",
                        rocke_fwd_gstream_wmma_threads_per_block(spec));
    if(gs_x_pitch(spec) > 0xFFFF || gs_halo_w(spec) > 0xFFFF || gs_halo_h(spec) > 0xFFFF)
        ROCKE_SPEC_FAIL(reason, reason_cap, "tile too wide for a TDM descriptor");
    if(!rocke_archtarget_fits_lds(target, rocke_fwd_gstream_wmma_lds_bytes(spec)))
        ROCKE_SPEC_FAIL(reason,
                        reason_cap,
                        "LDS %ld B > %d",
                        rocke_fwd_gstream_wmma_lds_bytes(spec),
                        target->lds_capacity_bytes);
    {
        const int TM = spec->tiles_k, TN = spec->tiles_p;
        const int nks = (int)gs_k_steps(spec).size();
        const int b_sets = gs_share_b(spec) ? TN : TM * TN;
        const int vgprs
            = 8 * TM * TN + 8 * TM * nks + 8 * b_sets * (spec->prefetch ? 1 + spec->pf_depth : 1);
        if(vgprs > ROCKE_FWD_GSTREAM_VGPR_BUDGET)
            ROCKE_SPEC_FAIL(
                reason, reason_cap, "~%d VGPRs of weights, accumulators and fragments", vgprs);
    }
    if(!rocke_wmma_fwd_limits_ok(p, spec->sched, gs_grid0(spec), reason, reason_cap))
        return false;
    rocke_spec_set_reason(reason, reason_cap, "ok");
    return true;
}

/* ===================================================================== *
 *  build_fwd_gstream_wmma
 * ===================================================================== */
namespace
{

V ci(rocke_ir_builder_t* b, int64_t v)
{
    return rocke_b_const_i32(b, v);
}

template <typename Body>
void gs_if(rocke_ir_builder_t* b, V cond, Body body)
{
    rocke_if_t g = rocke_b_scf_if(b, cond);
    rocke_b_region_enter(b, g.then_region);
    body();
    rocke_b_region_leave(b);
}

struct gs_sub_col
{
    V sw, w_start, w_ext;
};

} // namespace

rocke_kernel_def_t* rocke_build_fwd_gstream_wmma(rocke_ir_builder_t* b,
                                                 const rocke_fwd_gstream_wmma_spec_t* spec,
                                                 const char* arch)
{
    const rocke_direct_conv_problem_t* p = &spec->problem;
    const bool bf16 = rocke_direct_conv_problem_is_bf16(p);
    const int KW = p->KW;
    const int S = p->stride, PAD = p->PAD;
    const int CPG = p->cpg;
    const char* wmma_op = rocke_wmma_fwd_op_id(
        b, rocke_direct_conv_problem_dtype(p), arch, "build_fwd_gstream_wmma");
    if(wmma_op == NULL)
        return NULL;

    const int WAVE = spec->wave_size;
    const int WK = spec->waves_k;
    const int TM = spec->tiles_k, TN = spec->tiles_p;
    const int THREADS = rocke_fwd_gstream_wmma_threads_per_block(spec);
    const int BK = rocke_fwd_gstream_wmma_block_k(spec);
    const int CH = rocke_fwd_gstream_wmma_block_c(spec);
    const int TH = rocke_fwd_gstream_wmma_tile_h(spec), TW = spec->tile_w;
    const int HALO_H = gs_halo_h(spec), HALO_W = gs_halo_w(spec);
    const int NS = spec->stages;
    const int XP = gs_x_pitch(spec);
    const long SLOT = gs_slot_elems(spec);
    const int TAPS = p->KH * p->KW;
    const bool PS = spec->phase_split;
    const int NPW = gs_phases_w(spec);
    const int HC = gs_halo_cols(spec);
    const long PSTR = PS ? gs_phase_elems(spec) : 0;
    const bool SPAN16 = gs_span16(spec);
    const bool SHARE_B = gs_share_b(spec);
    const std::vector<gs_kstep> KSTEPS = gs_k_steps(spec);
    const int KPAD = rocke_fwd_gstream_wmma_kpad(spec);
    const bool PADG = KPAD != p->kpg || (CPG % ROCKE_FWD_WMMA_K != 0 && !SPAN16);

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
    V c_N = prm.p_N;
    V c_H = prm.p_Hi, c_W = prm.p_Wi;
    V c_Ho = prm.p_Ho, c_Wo = prm.p_Wo;
    V c_K = prm.p_total_k;
    V xs_n = prm.p_A_stride_n, xs_h = prm.p_A_stride_hi, xs_w = prm.p_A_stride_wi;
    V ys_n = prm.p_D_stride_n, ys_h = prm.p_D_stride_ho, ys_w = prm.p_D_stride_wo;

    V tid = rocke_b_thread_id_x(b);
    V wave_id = rocke_b_div(b, tid, ci(b, WAVE));
    V lane = rocke_b_mod(b, tid, ci(b, WAVE));
    V l16 = rocke_b_mod(b, lane, c16);
    V lhalf = rocke_b_div(b, lane, c16);
    V wave_k = rocke_b_mod(b, wave_id, ci(b, WK));
    V wave_p = rocke_b_div(b, wave_id, ci(b, WK));

    /* ---- block: bx = (walker * n_wt + wt) * n_kb + kb. */
    V n_kb;
    if(KPAD != p->kpg)
    {
        V x = rocke_b_div(b, c_K, ci(b, p->kpg));
        V y = rocke_b_mul(b, x, ci(b, KPAD));
        n_kb = rocke_b_div(b, y, ci(b, BK));
    }
    else
        n_kb = rocke_b_div(b, c_K, ci(b, BK));
    V n_wt;
    {
        V x = rocke_b_add(b, c_Wo, ci(b, TW - 1));
        n_wt = rocke_b_div(b, x, ci(b, TW));
    }
    V n_bands;
    {
        V x = rocke_b_add(b, c_Ho, ci(b, TH - 1));
        n_bands = rocke_b_div(b, x, ci(b, TH));
    }
    V t = rocke_b_block_id_x(b);
    V kb = rocke_b_mod(b, t, n_kb);
    t = rocke_b_div(b, t, n_kb);
    V wt = rocke_b_mod(b, t, n_wt);
    V walker = rocke_b_div(b, t, n_wt);
    V k_base = rocke_b_mul(b, kb, ci(b, BK));
    V c_base;
    {
        V x = rocke_b_div(b, k_base, ci(b, KPAD));
        c_base = rocke_b_mul(b, x, ci(b, CPG));
    }
    V w_tile = rocke_b_mul(b, wt, ci(b, TW));
    V task0 = rocke_b_mul(b, walker, ci(b, spec->walk));
    V n_tasks = rocke_b_mul(b, c_N, n_bands);
    V n_it;
    {
        V w = ci(b, spec->walk);
        V x = rocke_b_sub(b, n_tasks, task0);
        n_it = rocke_b_smin(b, w, x);
    }

    const int lds_shape[2] = {1, (int)(rocke_fwd_gstream_wmma_lds_bytes(spec) / 2)};
    V lds = rocke_b_smem_alloc(b, io_type, lds_shape, 2, "gs_lds");
    V lds_addr;
    {
        V a = rocke_b_smem_addr_of(b, lds);
        lds_addr = rocke_b_trunc(b, a, rocke_i32());
    }

    /* ---- per (phase) sub-halo column geometry: fixed for the whole walk. */
    V c_S = ci(b, S);
    V w0;
    {
        V x = S == 1 ? w_tile : rocke_b_mul(b, w_tile, c_S);
        w0 = rocke_b_sub(b, x, ci(b, PAD));
    }
    std::vector<gs_sub_col> sub_cols;
    for(int a = 0; a < NPW; ++a)
    {
        V wa = a ? rocke_b_add(b, w0, ci(b, a)) : w0;
        V sw, w_start, w_ext;
        if(PS)
        {
            V x = rocke_b_sub(b, ci(b, S - 1), wa);
            V y = rocke_b_div(b, x, c_S);
            sw = rocke_b_smax(b, y, c0);
            sw = rocke_b_smin(b, sw, ci(b, HC));
            V m = rocke_b_mul(b, sw, c_S);
            V s = rocke_b_add(b, wa, m);
            w_start = rocke_b_smax(b, s, c0);
            V d = rocke_b_sub(b, c_W, w_start);
            V e = rocke_b_add(b, d, ci(b, S - 1));
            w_ext = rocke_b_div(b, e, c_S);
        }
        else
        {
            V x = rocke_b_sub(b, c0, wa);
            V y = rocke_b_smax(b, x, c0);
            sw = rocke_b_smin(b, y, ci(b, HC));
            V s = rocke_b_add(b, wa, sw);
            w_start = rocke_b_smax(b, s, c0);
            w_ext = rocke_b_sub(b, c_W, w_start);
        }
        {
            V h = rocke_b_sub(b, ci(b, HC), sw);
            V m = rocke_b_smin(b, w_ext, h);
            w_ext = rocke_b_smax(b, m, c0);
        }
        sub_cols.push_back({sw, w_start, w_ext});
    }
    V xs_col = PS ? rocke_b_mul(b, xs_w, c_S) : xs_w;
    const long max_z
        = ((long)std::min(PAD, HALO_H) * HC + std::min(gs_lead_cols(spec), HC)) * (long)XP;
    V zero_v = rocke_b_zero_vec(b, io_type, 8);

    auto issue = [&](V task, V slot) {
        V n = rocke_b_div(b, task, n_bands);
        V band = rocke_b_mod(b, task, n_bands);
        V h0;
        {
            V x = rocke_b_mul(b, band, ci(b, TH * S));
            h0 = rocke_b_sub(b, x, ci(b, PAD));
        }
        V hs;
        {
            V x = rocke_b_sub(b, c0, h0);
            V y = rocke_b_smax(b, x, c0);
            hs = rocke_b_smin(b, y, ci(b, HALO_H));
        }
        V h_start;
        {
            V x = rocke_b_add(b, h0, hs);
            h_start = rocke_b_smax(b, x, c0);
        }
        V h_ext;
        {
            V x = rocke_b_sub(b, c_H, h_start);
            h_ext = rocke_b_smax(b, x, c0);
        }
        V tile2 = rocke_b_sub(b, ci(b, HALO_H), hs);
        V slot_off = rocke_b_mul(b, slot, ci(b, SLOT));
        for(int a = 0; a < NPW; ++a)
        {
            const gs_sub_col& sc = sub_cols[a];
            V cond = rocke_b_cmp_eq(b, wave_id, ci(b, a));
            gs_if(b, cond, [&]() {
                V z_elems;
                {
                    V x = rocke_b_mul(b, hs, ci(b, HC));
                    V y = rocke_b_add(b, x, sc.sw);
                    z_elems = rocke_b_mul(b, y, ci(b, XP));
                }
                V region = rocke_b_add(b, slot_off, ci(b, a * PSTR));
                if(max_z)
                {
                    V z_vecs = rocke_b_div(b, z_elems, c8);
                    for(long j = 0; j < (max_z + 7) / 8; j += WAVE)
                    {
                        V v = rocke_b_add(b, lane, ci(b, j));
                        V lt = rocke_b_cmp_lt(b, v, z_vecs);
                        gs_if(b, lt, [&]() {
                            V m = rocke_b_mul(b, v, c8);
                            V off = rocke_b_add(b, region, m);
                            V idx[2] = {c0, off};
                            rocke_b_smem_store_vN(b, lds, idx, 2, zero_v, 8);
                        });
                    }
                }
                V base;
                {
                    V m1 = rocke_b_mul(b, n, xs_n);
                    V m2 = rocke_b_mul(b, h_start, xs_h);
                    V a1 = rocke_b_add(b, m1, m2);
                    V m3 = rocke_b_mul(b, sc.w_start, xs_w);
                    V a2 = rocke_b_add(b, m3, c_base);
                    base = rocke_b_add(b, a1, a2);
                }
                V src;
                {
                    V m = rocke_b_mul(b, base, c2);
                    src = rocke_b_global_ptr_add(b, Xp, m);
                }
                V dst;
                {
                    V x = rocke_b_add(b, region, z_elems);
                    V y = rocke_b_mul(b, x, c2);
                    dst = rocke_b_add(b, lds_addr, y);
                }
                V d0 = ci(b, CH);
                V d[5];
                rocke_fwd_tdm_descriptor_3d(
                    b, src, dst, d0, sc.w_ext, h_ext, xs_col, xs_h, XP, HC, tile2, d);
                rocke_b_tensor_load_to_lds(b, d[0], d[1], d[2], d[3], d[4], 0);
            });
        }
    };

    /* ---- A fragments (weights) in registers for the whole walk. */
    std::vector<V> ks_tap;
    for(const gs_kstep& k : KSTEPS)
    {
        if(k.t0 == k.t1)
            ks_tap.push_back(ci(b, k.t0));
        else
        {
            const int tb = k.t1 >= 0 ? k.t1 : k.t0;
            V eq = rocke_b_cmp_eq(b, lhalf, c0);
            V x = ci(b, k.t0);
            V y = ci(b, tb);
            ks_tap.push_back(rocke_b_select(b, eq, x, y));
        }
    }
    V zero8 = rocke_b_zero_vec(b, io_type, 8);
    std::vector<std::vector<V>> afrag;
    for(int m = 0; m < TM; ++m)
    {
        V row;
        {
            V x = rocke_b_mul(b, wave_k, ci(b, TM));
            V y = rocke_b_add(b, x, ci(b, m));
            V z = rocke_b_mul(b, y, c16);
            V u = rocke_b_add(b, k_base, z);
            row = rocke_b_add(b, u, l16);
        }
        V row_ok = NULL;
        if(KPAD != p->kpg)
        {
            V r_in = rocke_b_mod(b, row, ci(b, KPAD));
            row_ok = rocke_b_cmp_lt(b, r_in, ci(b, p->kpg));
            V x = rocke_b_div(b, row, ci(b, KPAD));
            V y = rocke_b_mul(b, x, ci(b, p->kpg));
            V z = rocke_b_add(b, y, r_in);
            row = rocke_b_select(b, row_ok, z, c0);
        }
        V row_base = rocke_b_mul(b, row, ci(b, TAPS));
        std::vector<V> frags;
        for(size_t ks = 0; ks < KSTEPS.size(); ++ks)
        {
            const gs_kstep& k = KSTEPS[ks];
            V gidx;
            {
                V x = rocke_b_add(b, row_base, ks_tap[ks]);
                gidx = rocke_b_mul(b, x, ci(b, CPG));
            }
            V lo, hi;
            if(SPAN16 && CPG == 8)
            {
                V w8 = rocke_b_global_load_vN(b, Wp, gidx, io_type, 8, 0);
                V d = rocke_b_div(b, l16, c8);
                V first = rocke_b_cmp_eq(b, d, c0);
                lo = rocke_b_select(b, first, w8, zero8);
                hi = rocke_b_select(b, first, zero8, w8);
            }
            else
            {
                if(!SPAN16)
                {
                    V x = ci(b, k.cc * ROCKE_FWD_WMMA_K);
                    V y = rocke_b_mul(b, lhalf, c16);
                    V z = rocke_b_add(b, x, y);
                    gidx = rocke_b_add(b, gidx, z);
                }
                if(PADG)
                {
                    V ch;
                    {
                        V x = ci(b, k.cc * ROCKE_FWD_WMMA_K);
                        V y = rocke_b_mul(b, lhalf, c16);
                        ch = rocke_b_add(b, x, y);
                    }
                    V ok_lo = rocke_b_cmp_lt(b, ch, ci(b, CPG));
                    V ok_hi;
                    {
                        V x = rocke_b_add(b, ch, c8);
                        ok_hi = rocke_b_cmp_lt(b, x, ci(b, CPG));
                    }
                    if(row_ok != NULL)
                    {
                        ok_lo = rocke_b_land(b, ok_lo, row_ok);
                        ok_hi = rocke_b_land(b, ok_hi, row_ok);
                    }
                    {
                        V s = rocke_b_select(b, ok_lo, gidx, c0);
                        lo = rocke_b_global_load_vN(b, Wp, s, io_type, 8, 0);
                    }
                    {
                        V x = rocke_b_add(b, gidx, c8);
                        V s = rocke_b_select(b, ok_hi, x, c0);
                        hi = rocke_b_global_load_vN(b, Wp, s, io_type, 8, 0);
                    }
                    lo = rocke_b_select(b, ok_lo, lo, zero8);
                    hi = rocke_b_select(b, ok_hi, hi, zero8);
                }
                else
                {
                    lo = rocke_b_global_load_vN(b, Wp, gidx, io_type, 8, 0);
                    V x = rocke_b_add(b, gidx, c8);
                    hi = rocke_b_global_load_vN(b, Wp, x, io_type, 8, 0);
                }
            }
            if(k.t1 < 0)
            {
                V keep = rocke_b_cmp_eq(b, lhalf, c0);
                lo = rocke_b_select(b, keep, lo, zero8);
                hi = rocke_b_select(b, keep, hi, zero8);
            }
            frags.push_back(rocke_b_vec_concat(b, lo, hi));
        }
        afrag.push_back(frags);
    }

    /* ---- per-lane B (X) read offsets. */
    std::vector<V> px_off;
    for(int tn = 0; tn < TN; ++tn)
    {
        V flat;
        {
            V x = rocke_b_mul(b, wave_p, ci(b, TN));
            V y = rocke_b_add(b, x, ci(b, tn));
            V z = rocke_b_mul(b, y, c16);
            flat = rocke_b_add(b, z, l16);
        }
        V th = rocke_b_div(b, flat, ci(b, TW));
        V tw = rocke_b_mod(b, flat, ci(b, TW));
        if(S > 1)
        {
            th = rocke_b_mul(b, th, c_S);
            if(!PS)
                tw = rocke_b_mul(b, tw, c_S);
        }
        V off;
        {
            V x = rocke_b_mul(b, th, ci(b, HC));
            V y = rocke_b_add(b, x, tw);
            off = rocke_b_mul(b, y, ci(b, XP));
        }
        if(!SPAN16)
        {
            V x = rocke_b_mul(b, lhalf, c16);
            off = rocke_b_add(b, off, x);
        }
        px_off.push_back(off);
    }

    auto tap_off = [&](int tap) -> long {
        const int r = tap / KW, s = tap % KW;
        if(PS)
        {
            const int a = s % S, j = s / S;
            return a * PSTR + (long)(r * HC + j) * XP;
        }
        return (long)(r * HALO_W + s) * XP;
    };
    struct ks_off_t
    {
        long stat;
        V var;
    };
    std::vector<ks_off_t> ks_off;
    for(const gs_kstep& k : KSTEPS)
    {
        const int tb = k.t1 >= 0 ? k.t1 : k.t0;
        if(tap_off(k.t0) == tap_off(tb))
            ks_off.push_back({tap_off(k.t0) + k.cc * ROCKE_FWD_WMMA_K, NULL});
        else
        {
            V eq = rocke_b_cmp_eq(b, lhalf, c0);
            V d = ci(b, tap_off(tb) - tap_off(k.t0));
            ks_off.push_back({tap_off(k.t0), rocke_b_select(b, eq, c0, d)});
        }
    }
    std::vector<V> m_glob;
    for(int m = 0; m < TM; ++m)
    {
        V x = rocke_b_mul(b, wave_k, ci(b, TM));
        m_glob.push_back(rocke_b_add(b, x, ci(b, m)));
    }
    std::vector<V> m_choff;
    if(BK > KPAD)
    {
        for(int m = 0; m < TM; ++m)
        {
            V x = rocke_b_mul(b, m_glob[m], c16);
            V y = rocke_b_div(b, x, ci(b, KPAD));
            m_choff.push_back(rocke_b_mul(b, y, ci(b, CPG)));
        }
    }
    else
        m_choff.assign(TM, c0);

    auto bfrags = [&](V slot_base, size_t ks) -> std::vector<V> {
        const ks_off_t& ko = ks_off[ks];
        std::vector<V> out;
        std::vector<std::pair<int, int>> sets;
        if(SHARE_B)
            for(int tn = 0; tn < TN; ++tn)
                sets.push_back({0, tn});
        else
            for(int m = 0; m < TM; ++m)
                for(int tn = 0; tn < TN; ++tn)
                    sets.push_back({m, tn});
        for(const auto& mt : sets)
        {
            V addr = rocke_b_add(b, slot_base, px_off[mt.second]);
            if(ko.var != NULL)
                addr = rocke_b_add(b, addr, ko.var);
            if(BK > KPAD)
                addr = rocke_b_add(b, addr, m_choff[mt.first]);
            V s = ci(b, ko.stat);
            V a = rocke_b_add(b, addr, s);
            out.push_back(rocke_fwd_lds_frag16(b, lds, a, NULL, 0, c0, c8, io_type));
        }
        return out;
    };

    /* ---- epilogue: 16 bytes per lane, masked through the buffer bounds. */
    V y_rsrc = rocke_b_buffer_rsrc(b, Yp, prm.D_bytes);
    std::vector<std::pair<V, V>> ep_px;
    for(int tn = 0; tn < TN; ++tn)
    {
        V flat;
        {
            V x = rocke_b_mul(b, wave_p, ci(b, TN));
            V y = rocke_b_add(b, x, ci(b, tn));
            V z = rocke_b_mul(b, y, c16);
            flat = rocke_b_add(b, z, l16);
        }
        V th = rocke_b_div(b, flat, ci(b, TW));
        V tw = rocke_b_mod(b, flat, ci(b, TW));
        ep_px.push_back({th, tw});
    }
    std::vector<V> ep_ch;
    for(int m = 0; m < TM; ++m)
    {
        V x = rocke_b_mul(b, m_glob[m], c16);
        V y = rocke_b_add(b, k_base, x);
        V z = rocke_b_mul(b, lhalf, c8);
        ep_ch.push_back(rocke_b_add(b, y, z));
    }
    std::vector<V> ep_ok(TM, (V)NULL);
    if(KPAD != p->kpg)
        for(int m = 0; m < TM; ++m)
        {
            V v = ep_ch[m];
            V r_in = rocke_b_mod(b, v, ci(b, KPAD));
            ep_ok[m] = rocke_b_cmp_lt(b, r_in, ci(b, p->kpg));
            V x = rocke_b_div(b, v, ci(b, KPAD));
            V y = rocke_b_mul(b, x, ci(b, p->kpg));
            ep_ch[m] = rocke_b_add(b, y, r_in);
        }

    auto epilogue = [&](V task, const std::vector<V>& accs) {
        V n = rocke_b_div(b, task, n_bands);
        V h_tile;
        {
            V x = rocke_b_mod(b, task, n_bands);
            h_tile = rocke_b_mul(b, x, ci(b, TH));
        }
        V nbase = rocke_b_mul(b, n, ys_n);
        for(int tn = 0; tn < TN; ++tn)
        {
            V h = rocke_b_add(b, h_tile, ep_px[tn].first);
            V w = rocke_b_add(b, w_tile, ep_px[tn].second);
            V ok;
            {
                V x = rocke_b_cmp_lt(b, h, c_Ho);
                V y = rocke_b_cmp_lt(b, w, c_Wo);
                ok = rocke_b_land(b, x, y);
            }
            V pix;
            {
                V x = rocke_b_mul(b, h, ys_h);
                V y = rocke_b_mul(b, w, ys_w);
                V z = rocke_b_add(b, x, y);
                pix = rocke_b_add(b, nbase, z);
            }
            for(int m = 0; m < TM; ++m)
            {
                V off;
                {
                    V x = rocke_b_add(b, pix, ep_ch[m]);
                    off = rocke_b_mul(b, x, c2);
                }
                V okm = ep_ok[m] == NULL ? ok : rocke_b_land(b, ok, ep_ok[m]);
                V voff = rocke_b_select(b, okm, off, prm.D_bytes);
                V val = rocke_b_vec_cast_f32_to(b, accs[m * TN + tn], io_type);
                rocke_dconv_buf_store_vN(b, bf16, y_rsrc, voff, c0, val, 4);
            }
        }
    };

    /* ---- the walk. */
    for(int st = 0; st < NS - 1; ++st)
    {
        V cst = ci(b, st);
        V lt = rocke_b_cmp_lt(b, cst, n_it);
        gs_if(b, lt, [&]() {
            V x = rocke_b_add(b, task0, ci(b, st));
            V s = ci(b, st);
            issue(x, s);
        });
    }
    V zero_acc = rocke_b_zero_vec_f32(b, 8);
    rocke_for_t loop = rocke_b_scf_for(b, c0, n_it, c1, "it");
    if(!rocke_ir_builder_ok(b))
        return NULL;
    rocke_b_region_enter(b, loop.body);
    {
        V it = loop.iv;
        rocke_fwd_tdm_ring_wait(b, it, NS, n_it);
        rocke_b_sync_lds_only(b);
        V nxt = rocke_b_add(b, it, ci(b, NS - 1));
        V lt = rocke_b_cmp_lt(b, nxt, n_it);
        gs_if(b, lt, [&]() {
            V x = rocke_b_add(b, task0, nxt);
            V s = rocke_b_mod(b, nxt, ci(b, NS));
            issue(x, s);
        });
        V slot_base;
        {
            V x = rocke_b_mod(b, it, ci(b, NS));
            slot_base = rocke_b_mul(b, x, ci(b, SLOT));
        }
        std::vector<V> accs(TM * TN, zero_acc);
        rocke_b_s_setprio(b, 1);
        const size_t NKS = KSTEPS.size();
        const size_t PD = spec->prefetch ? (size_t)spec->pf_depth : 0;
        std::vector<std::vector<V>> queue;
        for(size_t j = 0; j < std::min(PD, NKS); ++j)
            queue.push_back(bfrags(slot_base, j));
        for(size_t ks = 0; ks < NKS; ++ks)
        {
            std::vector<V> bf;
            if(PD)
            {
                bf = queue.front();
                queue.erase(queue.begin());
                if(ks + PD < NKS)
                    queue.push_back(bfrags(slot_base, ks + PD));
            }
            else
                bf = bfrags(slot_base, ks);
            for(int m = 0; m < TM; ++m)
                for(int tn = 0; tn < TN; ++tn)
                {
                    const int i = m * TN + tn;
                    const int bi = SHARE_B ? tn : i;
                    accs[i] = rocke_b_mma(b, wmma_op, afrag[m][ks], bf[bi], accs[i], NULL, 0);
                }
        }
        rocke_b_s_setprio(b, 0);
        V task = rocke_b_add(b, task0, it);
        epilogue(task, accs);
    }
    rocke_b_region_leave(b);
    if(!rocke_ir_builder_ok(b))
        return NULL;
    return b->kernel;
}

rocke_kernel_def_t* rocke_build_fwd_gstream_wmma_new(rocke_ir_builder_t* b,
                                                     const rocke_fwd_gstream_wmma_spec_t* spec,
                                                     const char* arch)
{
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        char name[512];
        char why[512];
        if(b == NULL || spec == NULL)
            return NULL;
        if(arch == NULL)
            arch = "gfx1250";
        if(!rocke_fwd_gstream_wmma_is_valid_spec(spec, arch, why, sizeof why))
        {
            rocke_ir_builder_init(b, "direct_fwd_gstream_wmma_invalid");
            rocke_i_set_err(b, ROCKE_ERR_VALUE, "invalid FwdGstreamWmmaSpec for %s: %s", arch, why);
            return NULL;
        }
        if(rocke_fwd_gstream_wmma_kernel_name(spec, name, sizeof name) != ROCKE_OK)
            return NULL;
        if(rocke_ir_builder_init(b, name) != ROCKE_OK)
            return NULL;
        return rocke_build_fwd_gstream_wmma(b, spec, arch);
    });
}

rocke_status_t rocke_fwd_gstream_wmma_lower_to_llvm(const rocke_fwd_gstream_wmma_spec_t* spec,
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
    rocke_kernel_def_t* kernel = rocke_build_fwd_gstream_wmma_new(&b, spec, arch);
    if(kernel == NULL)
    {
        const char* m = rocke_ir_builder_error(&b);
        rocke_status_t st = rocke_ir_builder_status(&b);
        rocke_spec_set_reason(
            err, err_cap, (m != NULL && m[0] != '\0') ? m : "build_fwd_gstream_wmma failed");
        rocke_ir_builder_free(&b);
        return (st == ROCKE_OK) ? ROCKE_ERR_VALUE : st;
    }
    rocke_status_t st = rocke_lower_kernel_to_llvm_ex(kernel, flavor, arch, out_ll, err, err_cap);
    rocke_ir_builder_free(&b);
    return st;
}
