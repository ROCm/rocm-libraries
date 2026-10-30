// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * conv_dgrad_patch_wmma.cpp -- C++ port of build_dgrad_patch_wmma
 * (library/kernels/common/conv_dgrad_patch_wmma.py): the wave32 WMMA dgrad of a
 * patch embedding for gfx1250. See the header and the Python module for the
 * algorithm.
 *
 * Every helper emits in PYTHON EVALUATION ORDER (arguments left-to-right,
 * innermost call first) so the SSA sequence -- and therefore the .ll bytes --
 * match exactly. Nested builder calls are hoisted into ordered locals: C++
 * leaves the evaluation order of call arguments unspecified.
 */
#include "rocke/instance_conv_dgrad_patch_wmma.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <string>
#include <vector>

#include "rocke/error_boundary.hpp" /* ckc::guard_builder boundary shim */
#include "rocke/helper_rocke.core.arch.h" /* rocke_archtarget_from_gfx, op_for_shape */
#include "rocke/helper_rocke.helpers.io.h" /* rocke_b_io_ir_type */
#include "rocke/helper_rocke.helpers.spec.h" /* rocke_kernel_name_join */
#include "rocke/instance_conv_dgrad_stem_wmma_internal.hpp"
#include "rocke/instance_conv_direct_grouped_internal.h" /* rocke_dconv_emit_params */
#include "rocke/ir_internal.h" /* rocke_i_set_err_msg */
#include "rocke/lower_llvm.h"

using rocke_dgstem::ceil_div;
using rocke_dgstem::ci;
using rocke_dgstem::floordiv;
using rocke_dgstem::kChPad;
using rocke_dgstem::kVgprBudget;
using rocke_dgstem::kWmmaK;
using rocke_dgstem::set_reason;
using rocke_dgstem::V;

namespace
{

const char* dpw_dtype(const rocke_dgrad_patch_wmma_spec_t* spec)
{
    return spec->problem.dtype ? spec->problem.dtype : "fp16";
}

bool dpw_is_bf16(const rocke_dgrad_patch_wmma_spec_t* spec)
{
    return strcmp(dpw_dtype(spec), "bf16") == 0;
}

/* Spec @property helpers the builder shares. */
int dpw_P(const rocke_dgrad_patch_wmma_spec_t* s)
{
    return s->problem.KH;
}
int dpw_run(const rocke_dgrad_patch_wmma_spec_t* s)
{
    return dpw_P(s) * s->problem.cpg;
}
int dpw_x_pitch(const rocke_dgrad_patch_wmma_spec_t* s)
{
    return s->ck + kChPad;
}
int dpw_w_pitch(const rocke_dgrad_patch_wmma_spec_t* s)
{
    return rocke_dgrad_patch_wmma_block_c(s) + kChPad;
}
int dpw_x_elems(const rocke_dgrad_patch_wmma_spec_t* s)
{
    return rocke_dgrad_patch_wmma_block_p(s) * dpw_x_pitch(s);
}
int dpw_wei_elems(const rocke_dgrad_patch_wmma_spec_t* s)
{
    return s->ck * dpw_w_pitch(s);
}

/* DirectConvProblem.Ho / Wo (strided, dilated). */
long long dpw_Ho(const rocke_direct_conv_problem_t* p)
{
    return floordiv((long long)p->H + 2 * p->PAD - (long long)(p->KH - 1) * p->dil_h - 1, p->stride)
           + 1;
}
long long dpw_Wo(const rocke_direct_conv_problem_t* p)
{
    return floordiv((long long)p->W + 2 * p->PAD - (long long)(p->KW - 1) * p->dil_w - 1, p->stride)
           + 1;
}

} // namespace

/* ===================================================================== *
 *  Spec defaults + derived geometry
 * ===================================================================== */

rocke_dgrad_patch_wmma_spec_t rocke_dgrad_patch_wmma_spec_default(void)
{
    rocke_dgrad_patch_wmma_spec_t s;
    memset(&s, 0, sizeof(s));
    s.problem = rocke_direct_conv_problem_default();
    s.name = "dgrad_patch_wmma";
    s.waves_c = 4;
    s.waves_p = 2;
    s.tiles_c = 3;
    s.tiles_p = 4;
    s.tile_w = 16;
    s.r_blk = 4;
    s.ck = 32;
    s.stages = 2;
    s.prefetch = false;
    s.sched = NULL;
    s.wave_size = 32;
    return s;
}

bool rocke_is_patch_dgrad_problem(const rocke_direct_conv_problem_t* p)
{
    return p->KH == p->KW && p->KH > 1 && p->stride == p->KH && p->PAD == 0 && p->groups == 1
           && p->dil_h == 1 && p->dil_w == 1;
}

int rocke_dgrad_patch_wmma_block_c(const rocke_dgrad_patch_wmma_spec_t* spec)
{
    return spec->waves_c * spec->tiles_c * 16;
}

int rocke_dgrad_patch_wmma_block_p(const rocke_dgrad_patch_wmma_spec_t* spec)
{
    return spec->waves_p * spec->tiles_p * 16;
}

int rocke_dgrad_patch_wmma_block_cols(const rocke_dgrad_patch_wmma_spec_t* spec)
{
    return spec->r_blk * dpw_run(spec);
}

int rocke_dgrad_patch_wmma_tile_h(const rocke_dgrad_patch_wmma_spec_t* spec)
{
    return spec->tile_w > 0 ? (int)floordiv(rocke_dgrad_patch_wmma_block_p(spec), spec->tile_w) : 0;
}

int rocke_dgrad_patch_wmma_threads_per_block(const rocke_dgrad_patch_wmma_spec_t* spec)
{
    return spec->waves_c * spec->waves_p * spec->wave_size;
}

long rocke_dgrad_patch_wmma_lds_bytes(const rocke_dgrad_patch_wmma_spec_t* spec)
{
    long ring = (long)spec->stages * ((long)dpw_x_elems(spec) + dpw_wei_elems(spec));
    long epi = (long)rocke_dgrad_patch_wmma_block_p(spec) * rocke_dgrad_patch_wmma_block_cols(spec);
    return 2 * (ring > epi ? ring : epi);
}

void rocke_dgrad_patch_wmma_grid(const rocke_dgrad_patch_wmma_spec_t* spec, int grid[3])
{
    const rocke_direct_conv_problem_t* p = &spec->problem;
    long long Ho = dpw_Ho(p);
    long long Wo = dpw_Wo(p);
    long long main_blocks = (long long)p->N * ceil_div(Ho, rocke_dgrad_patch_wmma_tile_h(spec))
                            * ceil_div(Wo, spec->tile_w) * floordiv(dpw_P(spec), spec->r_blk);
    bool border = Ho * dpw_P(spec) < p->H || Wo * dpw_P(spec) < p->W;
    grid[0] = (int)(main_blocks + (border ? p->N : 0));
    grid[1] = 1;
    grid[2] = 1;
}

rocke_status_t rocke_dgrad_patch_wmma_kernel_name(const rocke_dgrad_patch_wmma_spec_t* spec,
                                                  char* out,
                                                  size_t out_cap)
{
    char pp[32], cc[32], wc[32], wp[32], tc[32], tp[32], tw[32], rb[32], ck[32], st[32];
    std::string sched;
    snprintf(pp, sizeof pp, "p%d", dpw_P(spec));
    snprintf(cc, sizeof cc, "c%d", spec->problem.cpg);
    snprintf(wc, sizeof wc, "wc%d", spec->waves_c);
    snprintf(wp, sizeof wp, "wp%d", spec->waves_p);
    snprintf(tc, sizeof tc, "tc%d", spec->tiles_c);
    snprintf(tp, sizeof tp, "tp%d", spec->tiles_p);
    snprintf(tw, sizeof tw, "tw%d", spec->tile_w);
    snprintf(rb, sizeof rb, "rb%d", spec->r_blk);
    snprintf(ck, sizeof ck, "ck%d", spec->ck);
    snprintf(st, sizeof st, "st%d", spec->stages);
    if(spec->sched != NULL && spec->sched[0] != '\0')
        sched = std::string("s") + spec->sched;
    const char* parts[]
        = {pp, cc, wc, wp, tc, tp, tw, rb, ck, st, spec->prefetch ? "pf" : "", sched.c_str()};
    const char* flag_names[] = {"bf16"};
    int flag_on[] = {dpw_is_bf16(spec) ? 1 : 0};
    return rocke_kernel_name_join(spec->name ? spec->name : "dgrad_patch_wmma",
                                  parts,
                                  sizeof(parts) / sizeof(parts[0]),
                                  flag_names,
                                  flag_on,
                                  1,
                                  out,
                                  out_cap,
                                  NULL);
}

/* ===================================================================== *
 *  dgrad_patch_reason / is_valid_dgrad_patch_spec
 * ===================================================================== */

bool rocke_dgrad_patch_wmma_is_valid_spec(const rocke_dgrad_patch_wmma_spec_t* spec,
                                          const char* arch,
                                          char* reason,
                                          size_t reason_cap)
{
    char msg[512];
    const rocke_archtarget_t* target;
    const rocke_direct_conv_problem_t* p;
    const char* ab;
    int block_c, block_p, block_cols, pad_cols, vgprs;

    if(spec == NULL)
    {
        set_reason(reason, reason_cap, "null spec");
        return false;
    }
    if(arch == NULL)
        arch = "gfx1250";
    target = rocke_archtarget_from_gfx(arch);
    if(target == NULL)
    {
        rocke_set_unknown_arch_reason(reason, reason_cap, arch);
        return false;
    }
    p = &spec->problem;
    if(target->wave_size != 32 || spec->wave_size != 32)
    {
        set_reason(reason, reason_cap, "dgrad_patch_wmma is a wave32 kernel");
        return false;
    }
    if(!target->memory.has_tdm)
    {
        snprintf(msg, sizeof msg, "no TDM on %s", arch);
        set_reason(reason, reason_cap, msg);
        return false;
    }
    if(strcmp(dpw_dtype(spec), "fp16") != 0 && strcmp(dpw_dtype(spec), "bf16") != 0)
    {
        snprintf(msg, sizeof msg, "unsupported dtype '%s'", dpw_dtype(spec));
        set_reason(reason, reason_cap, msg);
        return false;
    }
    ab = dpw_is_bf16(spec) ? "bf16" : "f16";
    if(!rocke_mma_catalog_has_shape(
           rocke_archtarget_mma(target), "wmma", ab, ab, "fp32", 16, 16, kWmmaK))
    {
        snprintf(msg, sizeof msg, "missing wmma_f32_16x16x32_%s on %s", ab, arch);
        set_reason(reason, reason_cap, msg);
        return false;
    }
    if(!rocke_is_patch_dgrad_problem(p))
    {
        set_reason(reason,
                   reason_cap,
                   "not a patch embedding (stride == square filter, pad 0, ungrouped)");
        return false;
    }
    if(p->kpg % 8 != 0)
    {
        snprintf(msg, sizeof msg, "K %d must be a multiple of 8 (16-byte dY rows)", p->kpg);
        set_reason(reason, reason_cap, msg);
        return false;
    }
    if(spec->ck % kWmmaK != 0 || spec->ck <= 0)
    {
        snprintf(msg, sizeof msg, "ck %d must be a positive multiple of %d", spec->ck, kWmmaK);
        set_reason(reason, reason_cap, msg);
        return false;
    }
    if(spec->waves_c < 1 || spec->waves_p < 1 || spec->tiles_c < 1 || spec->tiles_p < 1
       || spec->r_blk < 1)
    {
        set_reason(reason, reason_cap, "waves, tiles and r_blk must be >= 1");
        return false;
    }
    if(dpw_P(spec) % spec->r_blk != 0)
    {
        snprintf(msg, sizeof msg, "r_blk %d must divide P %d", spec->r_blk, dpw_P(spec));
        set_reason(reason, reason_cap, msg);
        return false;
    }
    block_c = rocke_dgrad_patch_wmma_block_c(spec);
    block_p = rocke_dgrad_patch_wmma_block_p(spec);
    block_cols = rocke_dgrad_patch_wmma_block_cols(spec);
    pad_cols = block_c - block_cols;
    if(pad_cols < 0 || pad_cols > (15 > block_c / 8 ? 15 : block_c / 8))
    {
        snprintf(msg,
                 sizeof msg,
                 "block_c %d must cover r_blk*P*C = %d with at most an eighth of padding",
                 block_c,
                 block_cols);
        set_reason(reason, reason_cap, msg);
        return false;
    }
    if(spec->stages < 2)
    {
        set_reason(reason, reason_cap, "stages must be >= 2");
        return false;
    }
    if(spec->tile_w < 1 || block_p % spec->tile_w != 0)
    {
        snprintf(msg, sizeof msg, "tile_w %d must divide %d", spec->tile_w, block_p);
        set_reason(reason, reason_cap, msg);
        return false;
    }
    if(spec->waves_c * spec->waves_p < 2)
    {
        set_reason(reason, reason_cap, "needs at least two waves");
        return false;
    }
    if(rocke_dgrad_patch_wmma_threads_per_block(spec)
       > rocke_archtarget_max_threads_per_block(target))
    {
        snprintf(msg,
                 sizeof msg,
                 "%d threads per block",
                 rocke_dgrad_patch_wmma_threads_per_block(spec));
        set_reason(reason, reason_cap, msg);
        return false;
    }
    if((long long)spec->tile_w * dpw_run(spec) > 0xFFFF || dpw_w_pitch(spec) > 0xFFFF)
    {
        set_reason(reason, reason_cap, "tile too wide for a TDM descriptor");
        return false;
    }
    if(!rocke_archtarget_fits_lds(target, rocke_dgrad_patch_wmma_lds_bytes(spec)))
    {
        snprintf(msg,
                 sizeof msg,
                 "LDS %ld B > %d",
                 rocke_dgrad_patch_wmma_lds_bytes(spec),
                 target->lds_capacity_bytes);
        set_reason(reason, reason_cap, msg);
        return false;
    }
    vgprs = 8 * spec->tiles_c * spec->tiles_p
            + 8 * (spec->tiles_c + spec->tiles_p) * (spec->prefetch ? 2 : 1);
    if(vgprs > kVgprBudget)
    {
        snprintf(msg, sizeof msg, "~%d VGPRs of accumulators and fragments", vgprs);
        set_reason(reason, reason_cap, msg);
        return false;
    }
    if(spec->sched != NULL && !rocke_scheduler_strategy_is_valid(spec->sched))
    {
        snprintf(msg, sizeof msg, "unsupported scheduler_strategy '%s'", spec->sched);
        set_reason(reason, reason_cap, msg);
        return false;
    }
    /* Global offsets are i32 bytes. */
    {
        long long total_k = (long long)p->groups * p->kpg;
        long long total_c = (long long)p->groups * p->cpg;
        long long bytes[3] = {2 * p->N * dpw_Ho(p) * dpw_Wo(p) * total_k,
                              2 * total_k * p->KH * p->KW * p->cpg,
                              2 * p->N * (long long)p->H * p->W * total_c};
        long long mx = bytes[0];
        if(bytes[1] > mx)
            mx = bytes[1];
        if(bytes[2] > mx)
            mx = bytes[2];
        if(mx > INT32_MAX)
        {
            snprintf(msg, sizeof msg, "tensors of %lld B exceed the i32 byte offsets", mx);
            set_reason(reason, reason_cap, msg);
            return false;
        }
    }
    set_reason(reason, reason_cap, "ok");
    return true;
}

/* ===================================================================== *
 *  Builder
 * ===================================================================== */
namespace
{

struct dpw_ctx
{
    rocke_ir_builder_t* b;
    const rocke_dgrad_patch_wmma_spec_t* spec;
    rocke_dconv_params_t params;
    const rocke_type_t* io_type;
    const char* wmma_op_id;
    int P, C, RB, WAVE, WC, THREADS, TH, TW;
    V tid, wave_id, wave_c, wave_p, lane, l16, lhalf;
    V n_ht, n_wt, bid, lds, lds_addr;
};

/* _main(...): the GEMM blocks. */
bool dpw_main(dpw_ctx& c)
{
    rocke_ir_builder_t* b = c.b;
    const rocke_dgrad_patch_wmma_spec_t* spec = c.spec;
    const rocke_dconv_params_t* params = &c.params;
    const int P = c.P;
    const int RUN = dpw_run(spec);
    const int NG = P * RUN;
    const int RB = c.RB;
    const int BCR = rocke_dgrad_patch_wmma_block_cols(spec);
    const int TM = spec->tiles_c, TN = spec->tiles_p;
    const int TH = c.TH, TW = c.TW;
    const int CK = spec->ck, NS = spec->stages;
    const int XP = dpw_x_pitch(spec), WPITCH = dpw_w_pitch(spec);
    const int XE = dpw_x_elems(spec), WEI = dpw_wei_elems(spec);
    V c0 = ci(b, 0);
    V c1 = ci(b, 1);
    V c2 = ci(b, 2);
    V c8 = ci(b, 8);
    V c16 = ci(b, 16);
    V c_K = params->p_total_k;
    V c_Ho = params->p_Ho;
    V c_Wo = params->p_Wo;

    /* bid = ((n * n_ht + ht) * n_wt + wt) * n_rt + rt. */
    const int n_rt = P / RB;
    V t = c.bid;
    V rt, wt, ht, n_img;
    {
        V x = ci(b, n_rt);
        rt = rocke_b_mod(b, t, x);
        V y = ci(b, n_rt);
        t = rocke_b_div(b, t, y);
    }
    wt = rocke_b_mod(b, t, c.n_wt);
    t = rocke_b_div(b, t, c.n_wt);
    ht = rocke_b_mod(b, t, c.n_ht);
    n_img = rocke_b_div(b, t, c.n_ht);
    V h_tile, w_tile, r0, c_CK, n_chunks;
    {
        V x = ci(b, TH);
        h_tile = rocke_b_mul(b, ht, x);
    }
    {
        V x = ci(b, TW);
        w_tile = rocke_b_mul(b, wt, x);
    }
    {
        V x = ci(b, RB);
        r0 = rocke_b_mul(b, rt, x);
    }
    c_CK = ci(b, CK);
    {
        V x = ci(b, CK - 1);
        V y = rocke_b_add(b, c_K, x);
        n_chunks = rocke_b_div(b, y, c_CK);
    }

    const int WEI0 = NS * XE;
    V wei_addr, dy_base, dy_w_ext, dy_h_ext, w_base, c_NG;
    {
        V x = ci(b, (int64_t)WEI0 * 2);
        wei_addr = rocke_b_add(b, c.lds_addr, x);
    }
    {
        V x = rocke_b_mul(b, n_img, params->p_A_stride_n);
        V y = rocke_b_mul(b, h_tile, params->p_A_stride_hi);
        V z = rocke_b_add(b, x, y);
        V u = rocke_b_mul(b, w_tile, params->p_A_stride_wi);
        dy_base = rocke_b_add(b, z, u);
    }
    {
        V x = rocke_b_sub(b, c_Wo, w_tile);
        V y = ci(b, TW);
        V z = rocke_b_smin(b, x, y);
        dy_w_ext = rocke_b_smax(b, z, c0);
    }
    {
        V x = rocke_b_sub(b, c_Ho, h_tile);
        dy_h_ext = rocke_b_smax(b, x, c0);
    }
    {
        V x = ci(b, RUN);
        w_base = rocke_b_mul(b, r0, x);
    }
    c_NG = ci(b, NG);

    rocke_dgstem::Gemm g;
    g.b = b;
    g.io_type = c.io_type;
    g.wmma_op_id = c.wmma_op_id;
    g.prefetch = spec->prefetch;
    g.TM = TM;
    g.TN = TN;
    g.CK = CK;
    g.NS = NS;
    g.XP = XP;
    g.WPITCH = WPITCH;
    g.XE = XE;
    g.WEI = WEI;
    g.WEI0 = WEI0;
    g.c0 = c0;
    g.c1 = c1;
    g.c8 = c8;
    g.c16 = c16;
    g.lane = c.lane;
    g.l16 = c.l16;
    g.lhalf = c.lhalf;
    g.wave_c = c.wave_c;
    g.wave_p = c.wave_p;
    g.lds = c.lds;
    g.n_chunks = n_chunks;
    g.issue = [&](V chunk) {
        V stage, k0, k_ext;
        {
            V x = ci(b, NS);
            stage = rocke_b_mod(b, chunk, x);
        }
        k0 = rocke_b_mul(b, chunk, c_CK);
        {
            V x = rocke_b_sub(b, c_K, k0);
            V y = rocke_b_smin(b, x, c_CK);
            k_ext = rocke_b_smax(b, y, c0);
        }
        {
            V cond = rocke_b_cmp_eq(b, c.wave_id, c0);
            rocke_if_t gate = rocke_b_scf_if(b, cond);
            rocke_b_region_enter(b, gate.then_region);
            V src, dst, d[5];
            {
                V x = rocke_b_add(b, dy_base, k0);
                V y = rocke_b_mul(b, x, c2);
                src = rocke_b_global_ptr_add(b, params->A, y);
            }
            {
                V x = ci(b, (int64_t)XE * 2);
                V y = rocke_b_mul(b, stage, x);
                dst = rocke_b_add(b, c.lds_addr, y);
            }
            V th = ci(b, TH);
            rocke_dgstem::tdm_descriptor_3d(b,
                                            src,
                                            dst,
                                            k_ext,
                                            dy_w_ext,
                                            dy_h_ext,
                                            params->p_A_stride_wi,
                                            params->p_A_stride_hi,
                                            XP,
                                            TW,
                                            th,
                                            d);
            rocke_b_tensor_load_to_lds(b, d[0], d[1], d[2], d[3], d[4], 0);
            rocke_b_region_leave(b);
        }
        {
            V cond = rocke_b_cmp_eq(b, c.wave_id, c1);
            rocke_if_t gate = rocke_b_scf_if(b, cond);
            rocke_b_region_enter(b, gate.then_region);
            V src, dst, d[5];
            {
                V x = rocke_b_mul(b, k0, c_NG);
                V y = rocke_b_add(b, w_base, x);
                V z = rocke_b_mul(b, y, c2);
                src = rocke_b_global_ptr_add(b, params->Bp, z);
            }
            {
                V x = ci(b, (int64_t)WEI * 2);
                V y = rocke_b_mul(b, stage, x);
                dst = rocke_b_add(b, wei_addr, y);
            }
            V bcr = ci(b, BCR);
            V s2 = rocke_b_mul(b, c_NG, c_CK);
            rocke_dgstem::tdm_descriptor_3d(
                b, src, dst, bcr, k_ext, c1, c_NG, s2, WPITCH, CK, c1, d);
            rocke_b_tensor_load_to_lds(b, d[0], d[1], d[2], d[3], d[4], 0);
            rocke_b_region_leave(b);
        }
    };

    rocke_dgstem::operand_offsets(g);
    std::vector<V> accs = rocke_dgstem::k_loop(g);
    if(accs.empty())
        return false;

    /* ---- epilogue: stage as [i][r][j][s, c] and store with one rank-3 TDM. */
    rocke_b_sync_lds_only(b);
    const bool vec_ok = RUN % 8 == 0;
    for(int m = 0; m < TM; ++m)
    {
        V col;
        {
            V x = ci(b, TM);
            V y = rocke_b_mul(b, c.wave_c, x);
            V z = ci(b, m);
            V u = rocke_b_add(b, y, z);
            V v = rocke_b_mul(b, u, c16);
            V w = rocke_b_mul(b, c.lhalf, c8);
            col = rocke_b_add(b, v, w);
        }
        for(int tn = 0; tn < TN; ++tn)
        {
            V px, pi, pj;
            {
                V x = ci(b, TN);
                V y = rocke_b_mul(b, c.wave_p, x);
                V z = ci(b, tn);
                V u = rocke_b_add(b, y, z);
                V v = rocke_b_mul(b, u, c16);
                px = rocke_b_add(b, v, c.l16);
            }
            {
                V x = ci(b, TW);
                pi = rocke_b_div(b, px, x);
                V y = ci(b, TW);
                pj = rocke_b_mod(b, px, y);
            }
            V val = rocke_b_vec_cast_f32_to(b, accs[m * TN + tn], c.io_type);
            auto addr = [&](V n_col) -> V {
                V x = ci(b, RUN);
                V rl = rocke_b_div(b, n_col, x);
                V y = ci(b, RUN);
                V q = rocke_b_mod(b, n_col, y);
                V a0 = ci(b, RB);
                V a1 = rocke_b_mul(b, pi, a0);
                V a2 = rocke_b_add(b, a1, rl);
                V a3 = ci(b, TW);
                V a4 = rocke_b_mul(b, a2, a3);
                V a5 = rocke_b_add(b, a4, pj);
                V a6 = ci(b, RUN);
                V a7 = rocke_b_mul(b, a5, a6);
                return rocke_b_add(b, a7, q);
            };
            if(vec_ok)
            {
                V x = ci(b, BCR);
                V cond = rocke_b_cmp_lt(b, col, x);
                rocke_if_t gate = rocke_b_scf_if(b, cond);
                rocke_b_region_enter(b, gate.then_region);
                V a = addr(col);
                V idx[2] = {c0, a};
                rocke_b_smem_store_vN(b, c.lds, idx, 2, val, 8);
                rocke_b_region_leave(b);
            }
            else
            {
                for(int e = 0; e < 8; ++e)
                {
                    V ce = ci(b, e);
                    V n_col = rocke_b_add(b, col, ce);
                    V x = ci(b, BCR);
                    V cond = rocke_b_cmp_lt(b, n_col, x);
                    rocke_if_t gate = rocke_b_scf_if(b, cond);
                    rocke_b_region_enter(b, gate.then_region);
                    V a = addr(n_col);
                    V idx[2] = {c0, a};
                    V el = rocke_b_vec_extract(b, val, e);
                    rocke_b_smem_store_vN(b, c.lds, idx, 2, el, 1);
                    rocke_b_region_leave(b);
                }
            }
        }
    }
    rocke_b_sync_lds_only(b);
    {
        V cond = rocke_b_cmp_eq(b, c.wave_id, c0);
        rocke_if_t gate = rocke_b_scf_if(b, cond);
        rocke_b_region_enter(b, gate.then_region);
        V row = params->p_D_stride_ho; /* W*C */
        V off, dst, d[5];
        {
            V x = rocke_b_mul(b, n_img, params->p_D_stride_n);
            V y = ci(b, P);
            V z = rocke_b_mul(b, h_tile, y);
            V u = rocke_b_add(b, z, r0);
            V v = rocke_b_mul(b, u, row);
            V w = rocke_b_add(b, x, v);
            V y2 = ci(b, P);
            V z2 = rocke_b_mul(b, w_tile, y2);
            V u2 = rocke_b_mul(b, z2, params->p_D_stride_wo);
            off = rocke_b_add(b, w, u2);
        }
        {
            V x = rocke_b_mul(b, off, c2);
            dst = rocke_b_global_ptr_add(b, params->D, x);
        }
        V run = ci(b, RUN);
        V d0 = rocke_b_mul(b, dy_w_ext, run);
        V d1 = ci(b, RB);
        V pp = ci(b, P);
        V s2 = rocke_b_mul(b, row, pp);
        V th = ci(b, TH);
        rocke_dgstem::tdm_descriptor_3d(
            b, dst, c.lds_addr, d0, d1, dy_h_ext, row, s2, TW * RUN, RB, th, d);
        rocke_b_tensor_store_from_lds(b, d[0], d[1], d[2], d[3], d[4], 0);
        rocke_b_s_wait_tensorcnt(b, 0);
        rocke_b_region_leave(b);
    }
    return rocke_ir_builder_ok(b);
}

/* The extra workgroups: zero the dX rows and columns no patch covers. */
void dpw_zero_border(dpw_ctx& c, V n_main, V c0, V c1)
{
    rocke_ir_builder_t* b = c.b;
    const rocke_dconv_params_t* params = &c.params;
    const int THREADS = c.THREADS;
    V c_H = params->p_Hi;
    V c_Ho = params->p_Ho;
    V c_Wo = params->p_Wo;
    V n_img = rocke_b_sub(b, c.bid, n_main);
    V img = rocke_b_mul(b, n_img, params->p_D_stride_n);
    V row = params->p_D_stride_ho; /* W*C */
    V hc, wc, zero;
    {
        V x = ci(b, c.P);
        hc = rocke_b_mul(b, c_Ho, x);
    }
    {
        V x = ci(b, c.P);
        V y = rocke_b_mul(b, c_Wo, x);
        V z = ci(b, c.C);
        wc = rocke_b_mul(b, y, z);
    }
    {
        V zv = rocke_b_zero_vec(b, c.io_type, 2);
        zero = rocke_b_vec_extract(b, zv, 0);
    }
    /* Bottom rows: one contiguous run. */
    V n_bot, bot0;
    {
        V x = rocke_b_sub(b, c_H, hc);
        n_bot = rocke_b_mul(b, x, row);
    }
    {
        V x = rocke_b_mul(b, hc, row);
        bot0 = rocke_b_add(b, img, x);
    }
    {
        V x = ci(b, THREADS - 1);
        V y = rocke_b_add(b, n_bot, x);
        V z = ci(b, THREADS);
        V hi = rocke_b_div(b, y, z);
        rocke_for_t loop = rocke_b_scf_for(b, c0, hi, c1, "zb");
        rocke_b_region_enter(b, loop.body);
        {
            V u = ci(b, THREADS);
            V v = rocke_b_mul(b, loop.iv, u);
            V e = rocke_b_add(b, v, c.tid);
            V cond = rocke_b_cmp_lt(b, e, n_bot);
            rocke_if_t gate = rocke_b_scf_if(b, cond);
            rocke_b_region_enter(b, gate.then_region);
            V a = rocke_b_add(b, bot0, e);
            rocke_b_global_store(b, params->D, a, zero, 2);
            rocke_b_region_leave(b);
        }
        rocke_b_region_leave(b);
    }
    /* Right columns of the covered rows. */
    V rw = rocke_b_sub(b, row, wc);
    V n_right = rocke_b_mul(b, hc, rw);
    {
        V cond = rocke_b_cmp_gt(b, rw, c0);
        rocke_if_t gate = rocke_b_scf_if(b, cond);
        rocke_b_region_enter(b, gate.then_region);
        V x = ci(b, THREADS - 1);
        V y = rocke_b_add(b, n_right, x);
        V z = ci(b, THREADS);
        V hi = rocke_b_div(b, y, z);
        rocke_for_t loop = rocke_b_scf_for(b, c0, hi, c1, "zr");
        rocke_b_region_enter(b, loop.body);
        {
            V u = ci(b, THREADS);
            V v = rocke_b_mul(b, loop.iv, u);
            V e = rocke_b_add(b, v, c.tid);
            V cond2 = rocke_b_cmp_lt(b, e, n_right);
            rocke_if_t gate2 = rocke_b_scf_if(b, cond2);
            rocke_b_region_enter(b, gate2.then_region);
            V rr = rocke_b_div(b, e, rw);
            V cc = rocke_b_mod(b, e, rw);
            V o1 = rocke_b_mul(b, rr, row);
            V o2 = rocke_b_add(b, wc, cc);
            V o3 = rocke_b_add(b, o1, o2);
            V off = rocke_b_add(b, img, o3);
            rocke_b_global_store(b, params->D, off, zero, 2);
            rocke_b_region_leave(b);
        }
        rocke_b_region_leave(b);
        rocke_b_region_leave(b);
    }
}

} // namespace

rocke_kernel_def_t* rocke_build_dgrad_patch_wmma(rocke_ir_builder_t* b,
                                                 const rocke_dgrad_patch_wmma_spec_t* spec,
                                                 const char* arch)
{
    char reason[512];
    dpw_ctx c;

    if(b == NULL || spec == NULL)
        return NULL;
    if(arch == NULL)
        arch = "gfx1250";
    if(!rocke_dgrad_patch_wmma_is_valid_spec(spec, arch, reason, sizeof reason))
    {
        std::string msg = std::string("invalid DgradPatchWmmaSpec for ") + arch + ": " + reason;
        rocke_i_set_err_msg(b, ROCKE_ERR_VALUE, msg.c_str());
        return NULL;
    }
    c.b = b;
    c.spec = spec;
    c.P = dpw_P(spec);
    c.C = spec->problem.cpg;
    c.RB = spec->r_blk;
    c.io_type = rocke_b_io_ir_type(b, dpw_dtype(spec));
    if(c.io_type == NULL)
        return NULL;
    {
        const char* ab = dpw_is_bf16(spec) ? "bf16" : "f16";
        const rocke_archtarget_t* target = rocke_archtarget_from_gfx(arch);
        const rocke_mmaop_t* op
            = rocke_archtarget_op_for_shape(target, "wmma", ab, ab, "fp32", 16, 16, kWmmaK, NULL);
        if(op == NULL || op->op_id == NULL)
        {
            rocke_i_set_err_msg(b, ROCKE_ERR_VALUE, "build_dgrad_patch_wmma: no WMMA atom");
            return NULL;
        }
        c.wmma_op_id = op->op_id;
    }
    c.WAVE = spec->wave_size;
    c.WC = spec->waves_c;
    c.THREADS = rocke_dgrad_patch_wmma_threads_per_block(spec);
    c.TH = rocke_dgrad_patch_wmma_tile_h(spec);
    c.TW = spec->tile_w;

    rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", c.THREADS);
    if(spec->sched != NULL)
        rocke_attr_set_str(b, &b->kernel->attrs, "scheduler_strategy", spec->sched);
    rocke_dconv_emit_params(b, &c.params, "dgrad", c.io_type);
    const rocke_dconv_params_t* params = &c.params;

    V c0 = ci(b, 0);
    V c1 = ci(b, 1);
    V c16 = ci(b, 16);
    V c_N = params->p_N;
    V c_Ho = params->p_Ho;
    V c_Wo = params->p_Wo;

    c.tid = rocke_b_thread_id_x(b);
    {
        V c_wave = ci(b, c.WAVE);
        c.wave_id = rocke_b_div(b, c.tid, c_wave);
        c.lane = rocke_b_mod(b, c.tid, c_wave);
    }
    c.l16 = rocke_b_mod(b, c.lane, c16);
    c.lhalf = rocke_b_div(b, c.lane, c16);
    {
        V x = ci(b, c.WC);
        c.wave_c = rocke_b_mod(b, c.wave_id, x);
        V y = ci(b, c.WC);
        c.wave_p = rocke_b_div(b, c.wave_id, y);
    }

    const int n_rt = c.P / c.RB;
    {
        V x = ci(b, c.TH - 1);
        V y = rocke_b_add(b, c_Ho, x);
        V z = ci(b, c.TH);
        c.n_ht = rocke_b_div(b, y, z);
    }
    {
        V x = ci(b, c.TW - 1);
        V y = rocke_b_add(b, c_Wo, x);
        V z = ci(b, c.TW);
        c.n_wt = rocke_b_div(b, y, z);
    }
    V n_main;
    {
        V x = rocke_b_mul(b, c_N, c.n_ht);
        V y = ci(b, n_rt);
        V z = rocke_b_mul(b, c.n_wt, y);
        n_main = rocke_b_mul(b, x, z);
    }
    c.bid = rocke_b_block_id_x(b);
    {
        int shape[2] = {1, (int)(rocke_dgrad_patch_wmma_lds_bytes(spec) / 2)};
        c.lds = rocke_b_smem_alloc(b, c.io_type, shape, 2, "pt_lds");
    }
    {
        V a = rocke_b_smem_addr_of(b, c.lds);
        c.lds_addr = rocke_b_trunc(b, a, rocke_i32());
    }

    V cond = rocke_b_cmp_lt(b, c.bid, n_main);
    rocke_if_else_t ie = rocke_b_scf_if_else(b, cond);
    rocke_b_region_enter(b, ie.then_region);
    bool ok = dpw_main(c);
    rocke_b_region_leave(b);
    if(!ok)
        return NULL;
    rocke_b_region_enter(b, ie.else_region);
    dpw_zero_border(c, n_main, c0, c1);
    rocke_b_region_leave(b);
    if(!rocke_ir_builder_ok(b))
        return NULL;
    return b->kernel;
}

rocke_kernel_def_t* rocke_build_dgrad_patch_wmma_new(rocke_ir_builder_t* b,
                                                     const rocke_dgrad_patch_wmma_spec_t* spec,
                                                     const char* arch)
{
    if(b != NULL)
    {
        memset(b, 0, sizeof(*b));
    }
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        char name[512];
        if(b == NULL)
            return NULL;
        if(spec == NULL)
        {
            rocke_i_set_err_msg(b, ROCKE_ERR_VALUE, "build_dgrad_patch_wmma: null spec");
            return NULL;
        }
        if(rocke_dgrad_patch_wmma_kernel_name(spec, name, sizeof(name)) != ROCKE_OK)
        {
            rocke_i_set_err_msg(
                b, ROCKE_ERR_VALUE, "build_dgrad_patch_wmma: kernel name does not fit");
            return NULL;
        }
        if(rocke_ir_builder_init(b, name) != ROCKE_OK)
            return NULL;
        return rocke_build_dgrad_patch_wmma(b, spec, arch);
    });
}

rocke_status_t rocke_dgrad_patch_wmma_lower_to_llvm(const rocke_dgrad_patch_wmma_spec_t* spec,
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
        set_reason(err, err_cap, "lower_to_llvm: null spec/out");
        return ROCKE_ERR_VALUE;
    }
    if(arch == NULL)
        arch = "gfx1250";
    kernel = rocke_build_dgrad_patch_wmma_new(&b, spec, arch);
    if(kernel == NULL)
    {
        const char* m = rocke_ir_builder_error(&b);
        st = rocke_ir_builder_status(&b);
        set_reason(err, err_cap, (m != NULL && m[0] != '\0') ? m : "build_dgrad_patch_wmma failed");
        rocke_ir_builder_free(&b);
        return (st == ROCKE_OK) ? ROCKE_ERR_VALUE : st;
    }
    st = rocke_lower_kernel_to_llvm_ex(kernel, flavor, arch, out_ll, err, err_cap);
    rocke_ir_builder_free(&b);
    return st;
}
