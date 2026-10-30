// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * conv_direct_dgrad_wmma.cpp -- C++ port of build_direct_dgrad_wmma
 * (library/kernels/common/conv_direct_dgrad_wmma.py): the wave32 WMMA direct
 * dgrad for gfx1250 (stride 1 or 2, per-axis dilation, packed small groups,
 * split barrier, per-wave epilogue). See the header and the Python module for
 * the algorithm.
 *
 * Every helper emits in PYTHON EVALUATION ORDER (arguments left-to-right,
 * innermost call first) so the SSA sequence -- and therefore the .ll bytes --
 * match exactly. Nested builder calls are hoisted into ordered locals: C++
 * leaves the evaluation order of call arguments unspecified.
 */
#include "rocke/instance_conv_direct_dgrad_wmma.h"

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
#include "rocke/instance_conv_direct_grouped_internal.h" /* rocke_dconv_emit_params */
#include "rocke/ir_internal.h" /* rocke_i_set_err_msg */
#include "rocke/lower_llvm.h"

/* conv_direct_dgrad_wmma.WMMA_K / CH_PAD / DGRAD_WMMA_VGPR_BUDGET. */
#define DGW_WMMA_K 32
#define DGW_CH_PAD 8
#define DGW_VGPR_BUDGET 448

namespace
{

int dgw_ceil_div(int a, int b)
{
    return (a + b - 1) / b;
}

/* pack_groups serves cpg / kpg in (2, 4, 8, 16). */
bool dgw_packable(int c)
{
    return c == 2 || c == 4 || c == 8 || c == 16;
}

void dgw_set_reason(char* reason, size_t reason_cap, const char* msg)
{
    if(reason && reason_cap > 0)
    {
        snprintf(reason, reason_cap, "%s", msg);
    }
}

const char* dgw_dtype(const rocke_direct_dgrad_wmma_spec_t* spec)
{
    return spec->problem.dtype ? spec->problem.dtype : "fp16";
}

bool dgw_is_bf16(const rocke_direct_dgrad_wmma_spec_t* spec)
{
    return strcmp(dgw_dtype(spec), "bf16") == 0;
}

/* Python floor division / modulo (the C operators truncate toward zero). */
long long dgw_floordiv(long long a, long long b)
{
    long long q = a / b;
    return (a % b != 0 && ((a < 0) != (b < 0))) ? q - 1 : q;
}

/* _axis_plan(K, PAD, dil, S): output row S * i + a (phase a) collects filter
 * row r from dY row i + delta(r), delta(r) = (a + PAD - r * dil) / S, for the
 * one phase a that makes it exact. Fills the phase and the halo offset
 * (delta - dmin) of every tap, the halo rows past the tile (dmax - dmin) and
 * the halo origin shift (-dmin). */
struct dgw_plan
{
    std::vector<int> phase;
    std::vector<int> delta;
    int ext = 0;
    int shift = 0;
};

dgw_plan dgw_axis_plan(int K, int PAD, int dil, int S)
{
    dgw_plan pl;
    std::vector<long long> d;
    for(int r = 0; r < K; ++r)
    {
        for(int a = 0; a < S; ++a)
        {
            long long x = (long long)a + PAD - (long long)r * dil;
            if(x % S == 0)
            {
                pl.phase.push_back(a);
                d.push_back(dgw_floordiv(x, S));
                break;
            }
        }
    }
    if(d.empty())
        return pl; /* K < 1 or S < 1: rejected by the validator */
    long long dmin = d[0], dmax = d[0];
    for(long long v : d)
    {
        if(v < dmin)
            dmin = v;
        if(v > dmax)
            dmax = v;
    }
    for(long long v : d)
        pl.delta.push_back((int)(v - dmin));
    pl.ext = (int)(dmax - dmin);
    pl.shift = (int)(-dmin);
    return pl;
}

dgw_plan dgw_plan_h(const rocke_direct_dgrad_wmma_spec_t* s)
{
    return dgw_axis_plan(s->problem.KH, s->problem.PAD, s->problem.dil_h, s->problem.stride);
}
dgw_plan dgw_plan_w(const rocke_direct_dgrad_wmma_spec_t* s)
{
    return dgw_axis_plan(s->problem.KW, s->problem.PAD, s->problem.dil_w, s->problem.stride);
}

/* Spec @property helpers the builder shares. */
int dgw_phases(const rocke_direct_dgrad_wmma_spec_t* s)
{
    return s->problem.stride * s->problem.stride;
}
int dgw_cpg_eff(const rocke_direct_dgrad_wmma_spec_t* s)
{
    return s->pack_groups * s->problem.cpg;
}
int dgw_kpg_eff(const rocke_direct_dgrad_wmma_spec_t* s)
{
    return s->pack_groups * s->problem.kpg;
}
int dgw_groups_eff(const rocke_direct_dgrad_wmma_spec_t* s)
{
    return s->pack_groups != 0 ? s->problem.groups / s->pack_groups : 0;
}
int dgw_halo_h(const rocke_direct_dgrad_wmma_spec_t* s)
{
    return rocke_direct_dgrad_wmma_tile_h(s) + dgw_plan_h(s).ext;
}
int dgw_halo_w(const rocke_direct_dgrad_wmma_spec_t* s)
{
    return s->tile_w + dgw_plan_w(s).ext;
}
int dgw_x_pitch(const rocke_direct_dgrad_wmma_spec_t* s)
{
    return s->ck + DGW_CH_PAD;
}
int dgw_w_pitch(const rocke_direct_dgrad_wmma_spec_t* s)
{
    return rocke_direct_dgrad_wmma_block_c(s) + DGW_CH_PAD;
}
int dgw_image_elems(const rocke_direct_dgrad_wmma_spec_t* s)
{
    return dgw_halo_h(s) * dgw_halo_w(s) * dgw_x_pitch(s);
}
int dgw_halo_elems(const rocke_direct_dgrad_wmma_spec_t* s)
{
    return s->tile_n * dgw_image_elems(s);
}
int dgw_wei_elems(const rocke_direct_dgrad_wmma_spec_t* s)
{
    return s->problem.KH * s->problem.KW * s->ck * dgw_w_pitch(s);
}
int dgw_halo_slack(const rocke_direct_dgrad_wmma_spec_t* s)
{
    /* A left-edge block's last row wraps up to PADD_W columns past the ring. */
    return dgw_plan_w(s).shift * dgw_x_pitch(s);
}

} // namespace

/* ===================================================================== *
 *  Spec defaults + derived geometry
 * ===================================================================== */

rocke_direct_dgrad_wmma_spec_t rocke_direct_dgrad_wmma_spec_default(void)
{
    rocke_direct_dgrad_wmma_spec_t s;
    memset(&s, 0, sizeof(s));
    s.problem = rocke_direct_conv_problem_default();
    s.name = "direct_dgrad_wmma";
    s.waves_c = 2;
    s.waves_p = 4;
    s.tiles_c = 4;
    s.tiles_p = 4;
    s.tile_w = 16;
    s.tile_n = 1;
    s.ck = 32;
    s.stages = 2;
    s.prefetch = false;
    s.sched = NULL;
    s.wave_size = 32;
    s.pack_groups = 1;
    s.split_barrier = false;
    s.wave_epilogue = false;
    return s;
}

int rocke_direct_dgrad_wmma_block_c(const rocke_direct_dgrad_wmma_spec_t* spec)
{
    return spec->waves_c * spec->tiles_c * 16;
}

int rocke_direct_dgrad_wmma_block_p(const rocke_direct_dgrad_wmma_spec_t* spec)
{
    return spec->waves_p * spec->tiles_p * 16;
}

int rocke_direct_dgrad_wmma_tile_h(const rocke_direct_dgrad_wmma_spec_t* spec)
{
    int d = spec->tile_w * spec->tile_n;
    return d > 0 ? rocke_direct_dgrad_wmma_block_p(spec) / d : 0;
}

int rocke_direct_dgrad_wmma_threads_per_block(const rocke_direct_dgrad_wmma_spec_t* spec)
{
    return spec->waves_c * spec->waves_p * spec->wave_size;
}

long rocke_direct_dgrad_wmma_lds_bytes(const rocke_direct_dgrad_wmma_spec_t* spec)
{
    long ring = (long)spec->stages
                * ((long)dgw_halo_elems(spec) + dgw_halo_slack(spec) + dgw_wei_elems(spec));
    long epi = (long)dgw_phases(spec) * rocke_direct_dgrad_wmma_block_p(spec) * dgw_w_pitch(spec);
    if(spec->wave_epilogue)
    {
        long n_waves = (long)spec->waves_c * spec->waves_p;
        epi = n_waves * spec->tiles_p * 16 * (spec->tiles_c * 16 + DGW_CH_PAD);
    }
    return 2 * (ring > epi ? ring : epi);
}

void rocke_direct_dgrad_wmma_grid(const rocke_direct_dgrad_wmma_spec_t* spec, int grid[3])
{
    const rocke_direct_conv_problem_t* p = &spec->problem;
    int S = p->stride;
    int n_ct = dgw_ceil_div(dgw_cpg_eff(spec), rocke_direct_dgrad_wmma_block_c(spec));
    int n_ht = dgw_ceil_div(dgw_ceil_div(p->H, S), rocke_direct_dgrad_wmma_tile_h(spec));
    int n_wt = dgw_ceil_div(dgw_ceil_div(p->W, S), spec->tile_w);
    grid[0] = dgw_ceil_div(p->N, spec->tile_n) * dgw_groups_eff(spec) * n_ht * n_wt * n_ct;
    grid[1] = 1;
    grid[2] = 1;
}

rocke_status_t rocke_direct_dgrad_wmma_kernel_name(const rocke_direct_dgrad_wmma_spec_t* spec,
                                                   char* out,
                                                   size_t out_cap)
{
    char f[32], pad[32], wc[32], wp[32], tc[32], tp[32], tw[32], tn[32], ck[32], st[32];
    char str[32], dil[48], pg[64];
    std::string sched;
    const rocke_direct_conv_problem_t* p = &spec->problem;
    snprintf(f, sizeof f, "f%dx%d", p->KH, p->KW);
    snprintf(pad, sizeof pad, "p%d", p->PAD);
    snprintf(wc, sizeof wc, "wc%d", spec->waves_c);
    snprintf(wp, sizeof wp, "wp%d", spec->waves_p);
    snprintf(tc, sizeof tc, "tc%d", spec->tiles_c);
    snprintf(tp, sizeof tp, "tp%d", spec->tiles_p);
    snprintf(tw, sizeof tw, "tw%d", spec->tile_w);
    if(spec->tile_n > 1)
        snprintf(tn, sizeof tn, "tn%d", spec->tile_n);
    else
        tn[0] = '\0';
    snprintf(ck, sizeof ck, "ck%d", spec->ck);
    snprintf(st, sizeof st, "st%d", spec->stages);
    if(spec->sched != NULL && spec->sched[0] != '\0')
        sched = std::string("s") + spec->sched;
    if(p->stride > 1)
        snprintf(str, sizeof str, "str%d", p->stride);
    else
        str[0] = '\0';
    if(p->dil_h != 1 || p->dil_w != 1)
        snprintf(dil, sizeof dil, "dil%dx%d", p->dil_h, p->dil_w);
    else
        dil[0] = '\0';
    if(spec->pack_groups > 1)
        snprintf(pg, sizeof pg, "pg%dc%dk%d", spec->pack_groups, p->cpg, p->kpg);
    else
        pg[0] = '\0';
    const char* parts[] = {f,
                           pad,
                           wc,
                           wp,
                           tc,
                           tp,
                           tw,
                           tn,
                           ck,
                           st,
                           spec->prefetch ? "pf" : "",
                           sched.c_str(),
                           str,
                           dil,
                           pg,
                           spec->split_barrier ? "sb" : "",
                           spec->wave_epilogue ? "we" : ""};
    const char* flag_names[] = {"bf16"};
    int flag_on[] = {dgw_is_bf16(spec) ? 1 : 0};
    return rocke_kernel_name_join(spec->name ? spec->name : "direct_dgrad_wmma",
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
 *  dgrad_wmma_reason / is_valid_dgrad_wmma_spec
 * ===================================================================== */

bool rocke_direct_dgrad_wmma_is_valid_spec(const rocke_direct_dgrad_wmma_spec_t* spec,
                                           const char* arch,
                                           char* reason,
                                           size_t reason_cap)
{
    char msg[512];
    const rocke_archtarget_t* target;
    const rocke_direct_conv_problem_t* p;
    const char* ab;
    int tw, block_p, vgprs, G;

    if(spec == NULL)
    {
        dgw_set_reason(reason, reason_cap, "null spec");
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
        dgw_set_reason(reason, reason_cap, "direct_dgrad_wmma is a wave32 kernel");
        return false;
    }
    if(!target->memory.has_tdm)
    {
        snprintf(msg, sizeof msg, "no TDM on %s", arch);
        dgw_set_reason(reason, reason_cap, msg);
        return false;
    }
    ab = dgw_is_bf16(spec) ? "bf16" : "f16";
    if(strcmp(dgw_dtype(spec), "fp16") != 0 && strcmp(dgw_dtype(spec), "bf16") != 0)
    {
        snprintf(msg, sizeof msg, "unsupported dtype '%s'", dgw_dtype(spec));
        dgw_set_reason(reason, reason_cap, msg);
        return false;
    }
    if(!rocke_mma_catalog_has_shape(
           rocke_archtarget_mma(target), "wmma", ab, ab, "fp32", 16, 16, DGW_WMMA_K))
    {
        snprintf(msg, sizeof msg, "missing wmma_f32_16x16x32_%s on %s", ab, arch);
        dgw_set_reason(reason, reason_cap, msg);
        return false;
    }
    if(p->stride != 1 && p->stride != 2)
    {
        snprintf(msg, sizeof msg, "stride %d not in (1, 2)", p->stride);
        dgw_set_reason(reason, reason_cap, msg);
        return false;
    }
    if(p->stride > 1 && (p->dil_h != 1 || p->dil_w != 1))
    {
        dgw_set_reason(reason, reason_cap, "a strided dilated dgrad is not supported");
        return false;
    }
    if(p->dil_h < 1 || p->dil_w < 1)
    {
        snprintf(msg, sizeof msg, "dilation %dx%d < 1", p->dil_h, p->dil_w);
        dgw_set_reason(reason, reason_cap, msg);
        return false;
    }
    if(p->KH < 1 || p->KW < 1)
    {
        snprintf(msg, sizeof msg, "filter %dx%d is not positive", p->KH, p->KW);
        dgw_set_reason(reason, reason_cap, msg);
        return false;
    }
    if(p->PAD < 0 || dgw_plan_h(spec).shift < 0 || dgw_plan_w(spec).shift < 0)
    {
        snprintf(msg, sizeof msg, "PAD %d outside 0..(K-1)*dilation", p->PAD);
        dgw_set_reason(reason, reason_cap, msg);
        return false;
    }
    if(rocke_direct_conv_one_padding_reason(p, "", reason, reason_cap))
        return false;
    G = spec->pack_groups;
    if(G < 1)
    {
        snprintf(msg, sizeof msg, "pack_groups %d < 1", G);
        dgw_set_reason(reason, reason_cap, msg);
        return false;
    }
    if(G > 1)
    {
        if(p->groups % G != 0)
        {
            snprintf(msg, sizeof msg, "groups %d not a multiple of pack_groups %d", p->groups, G);
            dgw_set_reason(reason, reason_cap, msg);
            return false;
        }
        if(!dgw_packable(p->cpg) || !dgw_packable(p->kpg))
        {
            snprintf(
                msg, sizeof msg, "packed cpg %d / kpg %d not in (2, 4, 8, 16)", p->cpg, p->kpg);
            dgw_set_reason(reason, reason_cap, msg);
            return false;
        }
        if(dgw_kpg_eff(spec) > spec->ck)
        {
            snprintf(msg,
                     sizeof msg,
                     "packed kpg %d > ck %d (one dY chunk)",
                     dgw_kpg_eff(spec),
                     spec->ck);
            dgw_set_reason(reason, reason_cap, msg);
            return false;
        }
        if(dgw_cpg_eff(spec) > rocke_direct_dgrad_wmma_block_c(spec))
        {
            snprintf(msg,
                     sizeof msg,
                     "packed cpg %d > block_c %d",
                     dgw_cpg_eff(spec),
                     rocke_direct_dgrad_wmma_block_c(spec));
            dgw_set_reason(reason, reason_cap, msg);
            return false;
        }
    }
    if(dgw_cpg_eff(spec) % 8 != 0 || dgw_kpg_eff(spec) % 8 != 0)
    {
        snprintf(msg,
                 sizeof msg,
                 "cpg %d and kpg %d must be multiples of 8 (16-byte rows)",
                 dgw_cpg_eff(spec),
                 dgw_kpg_eff(spec));
        dgw_set_reason(reason, reason_cap, msg);
        return false;
    }
    if(spec->ck % DGW_WMMA_K != 0 || spec->ck <= 0)
    {
        snprintf(msg, sizeof msg, "ck %d must be a positive multiple of %d", spec->ck, DGW_WMMA_K);
        dgw_set_reason(reason, reason_cap, msg);
        return false;
    }
    if(spec->waves_c < 1 || spec->waves_p < 1 || spec->tiles_c < 1 || spec->tiles_p < 1)
    {
        dgw_set_reason(reason, reason_cap, "waves and tiles must be >= 1");
        return false;
    }
    if(spec->stages < 2)
    {
        dgw_set_reason(reason, reason_cap, "stages must be >= 2");
        return false;
    }
    tw = spec->tile_w;
    block_p = rocke_direct_dgrad_wmma_block_p(spec);
    if(tw < 1 || spec->tile_n < 1)
    {
        snprintf(msg, sizeof msg, "tile_w %d and tile_n %d must be >= 1", tw, spec->tile_n);
        dgw_set_reason(reason, reason_cap, msg);
        return false;
    }
    if(block_p % (tw * spec->tile_n) != 0)
    {
        snprintf(
            msg, sizeof msg, "tile_w %d x tile_n %d must divide %d", tw, spec->tile_n, block_p);
        dgw_set_reason(reason, reason_cap, msg);
        return false;
    }
    if(spec->tile_n > 1 && (rocke_direct_dgrad_wmma_tile_h(spec) * tw) % 16 != 0)
    {
        dgw_set_reason(reason, reason_cap, "an image tile must hold whole 16-pixel tiles");
        return false;
    }
    if(spec->tile_n > spec->waves_c * spec->waves_p - 1)
    {
        dgw_set_reason(reason,
                       reason_cap,
                       "tile_n needs one more wave than images (one dY transfer per image)");
        return false;
    }
    if((spec->split_barrier || spec->wave_epilogue) && (p->stride > 1 || G > 1))
    {
        dgw_set_reason(
            reason, reason_cap, "split_barrier / wave_epilogue serve stride-1 unpacked tiles only");
        return false;
    }
    if(spec->wave_epilogue)
    {
        int wpx = spec->tiles_p * 16;
        if(wpx % tw != 0 || (rocke_direct_dgrad_wmma_tile_h(spec) * tw) % wpx != 0)
        {
            dgw_set_reason(
                reason, reason_cap, "wave_epilogue needs whole rows of one image per wave");
            return false;
        }
    }
    if(rocke_direct_dgrad_wmma_threads_per_block(spec)
       > rocke_archtarget_max_threads_per_block(target))
    {
        snprintf(msg,
                 sizeof msg,
                 "%d threads per block",
                 rocke_direct_dgrad_wmma_threads_per_block(spec));
        dgw_set_reason(reason, reason_cap, msg);
        return false;
    }
    if(spec->waves_c * spec->waves_p < 2)
    {
        dgw_set_reason(reason, reason_cap, "needs at least two waves (one TDM issuer per operand)");
        return false;
    }
    if(dgw_x_pitch(spec) > 0xFFFF || dgw_w_pitch(spec) > 0xFFFF || dgw_halo_w(spec) > 0xFFFF
       || dgw_halo_h(spec) > 0xFFFF)
    {
        dgw_set_reason(reason, reason_cap, "tile too wide for a TDM descriptor");
        return false;
    }
    if(!rocke_archtarget_fits_lds(target, rocke_direct_dgrad_wmma_lds_bytes(spec)))
    {
        snprintf(msg,
                 sizeof msg,
                 "LDS %ld B > %d",
                 rocke_direct_dgrad_wmma_lds_bytes(spec),
                 target->lds_capacity_bytes);
        dgw_set_reason(reason, reason_cap, msg);
        return false;
    }
    vgprs = 8 * dgw_phases(spec) * spec->tiles_c * spec->tiles_p
            + 8 * (spec->tiles_c + spec->tiles_p) * (spec->prefetch ? 2 : 1);
    if(vgprs > DGW_VGPR_BUDGET)
    {
        snprintf(msg, sizeof msg, "~%d VGPRs of accumulators and fragments", vgprs);
        dgw_set_reason(reason, reason_cap, msg);
        return false;
    }
    if(spec->sched != NULL && !rocke_scheduler_strategy_is_valid(spec->sched))
    {
        snprintf(msg, sizeof msg, "unsupported scheduler_strategy '%s'", spec->sched);
        dgw_set_reason(reason, reason_cap, msg);
        return false;
    }
    /* Global offsets are i32 bytes and the tile index is a flat block id. */
    {
        /* DirectConvProblem.Ho / .Wo: strided, dilated output extents. */
        long long Ho
            = dgw_floordiv((long long)p->H + 2 * p->PAD - (long long)(p->KH - 1) * p->dil_h - 1,
                           p->stride)
              + 1;
        long long Wo
            = dgw_floordiv((long long)p->W + 2 * p->PAD - (long long)(p->KW - 1) * p->dil_w - 1,
                           p->stride)
              + 1;
        long long total_k = (long long)p->groups * p->kpg;
        long long total_c = (long long)p->groups * p->cpg;
        long long bytes[3] = {2 * p->N * Ho * Wo * total_k,
                              2 * total_k * p->KH * p->KW * p->cpg,
                              2 * p->N * (long long)p->H * p->W * total_c};
        long long mx = bytes[0];
        long long tiles
            = (long long)dgw_ceil_div(p->N, spec->tile_n) * dgw_groups_eff(spec)
              * dgw_ceil_div(dgw_ceil_div(p->H, p->stride), rocke_direct_dgrad_wmma_tile_h(spec))
              * dgw_ceil_div(dgw_ceil_div(p->W, p->stride), spec->tile_w)
              * dgw_ceil_div(dgw_cpg_eff(spec), rocke_direct_dgrad_wmma_block_c(spec));
        if(bytes[1] > mx)
            mx = bytes[1];
        if(bytes[2] > mx)
            mx = bytes[2];
        if(mx > INT32_MAX)
        {
            snprintf(msg, sizeof msg, "tensors of %lld B exceed the i32 byte offsets", mx);
            dgw_set_reason(reason, reason_cap, msg);
            return false;
        }
        if(tiles > INT32_MAX)
        {
            snprintf(msg, sizeof msg, "%lld workgroups exceed the i32 block id", tiles);
            dgw_set_reason(reason, reason_cap, msg);
            return false;
        }
    }
    dgw_set_reason(reason, reason_cap, "ok");
    return true;
}

/* ===================================================================== *
 *  Builder
 * ===================================================================== */
namespace
{

struct dgw_ctx
{
    rocke_ir_builder_t* b;
    const rocke_direct_dgrad_wmma_spec_t* spec;
    rocke_dconv_params_t params;
    const rocke_type_t* io_type;
    const char* wmma_op_id;

    int KH, KW, PADD_H, PADD_W, S, G, NPH;
    bool SB;
    dgw_plan plan_h, plan_w;
    int WAVE, WC, WP, TM, TN, THREADS, BC, TH, TW, HALO_H, HALO_W, CK, NS;
    int XP, WPITCH, HALO, WEI, NB, IMG, HALO_STAGE, TAPS, WEI0;

    rocke_value_t *c0, *c1, *c2, *c8, *c16;
    rocke_value_t *c_groups, *c_cpg, *c_kpg, *c_N, *c_H, *c_W, *c_Ho, *c_Wo;
    rocke_value_t *tid, *wave_id, *lane, *l16, *lhalf, *wave_c, *wave_p;
    rocke_value_t *n_base, *c_base, *h_tile, *w_tile, *k_grp, *c_grp, *w_row, *c_CK, *n_chunks;
    rocke_value_t *lds, *halo_addr, *wei_addr;
    rocke_value_t *z_elems, *dy_w_ext, *dy_h_ext, *dy_tile2, *dy_base, *c_ext, *w_base;

    std::vector<rocke_value_t*> px_off;
    std::vector<rocke_value_t*> wei_off;
};

rocke_value_t* ci(dgw_ctx& c, int64_t v)
{
    return rocke_b_const_i32(c.b, v);
}

/* _tdm_descriptor_3d(b, global_ptr, lds_addr, dims, strides, tile0, tile1, tile2). */
void tdm_descriptor_3d(dgw_ctx& c,
                       rocke_value_t* global_ptr,
                       rocke_value_t* lds_addr,
                       rocke_value_t* d0,
                       rocke_value_t* d1,
                       rocke_value_t* d2,
                       rocke_value_t* s1,
                       rocke_value_t* s2,
                       int tile0,
                       int tile1,
                       rocke_value_t* tile2,
                       rocke_value_t* out[5])
{
    rocke_ir_builder_t* b = c.b;
    rocke_value_t* c16 = ci(c, 16);
    rocke_value_t* gaddr = rocke_b_global_ptr_to_i64(b, global_ptr);
    rocke_value_t* c32 = rocke_b_const_i64(b, 32);
    rocke_value_t* sh = rocke_b_lshr(b, gaddr, c32);
    rocke_value_t* g_hi = rocke_b_trunc(b, sh, rocke_i32());
    rocke_value_t* g0[4];
    rocke_value_t* g1[8];
    rocke_value_t* g2[4];
    g0[0] = ci(c, 1);
    g0[1] = lds_addr;
    g0[2] = rocke_b_trunc(b, gaddr, rocke_i32());
    {
        rocke_value_t* m = ci(c, 0x1FFFFFF);
        rocke_value_t* lo = rocke_b_land(b, g_hi, m);
        rocke_value_t* ty = ci(c, -((int64_t)1 << 31));
        g0[3] = rocke_b_lor(b, lo, ty);
    }
    g1[0] = ci(c, 1 << 16);
    g1[1] = rocke_b_shl(b, d0, c16);
    {
        rocke_value_t* x = rocke_b_lshr(b, d0, c16);
        rocke_value_t* y = rocke_b_shl(b, d1, c16);
        g1[2] = rocke_b_lor(b, x, y);
    }
    {
        rocke_value_t* x = rocke_b_lshr(b, d1, c16);
        rocke_value_t* y = ci(c, (int64_t)tile0 << 16);
        g1[3] = rocke_b_lor(b, x, y);
    }
    {
        rocke_value_t* x = ci(c, tile1);
        rocke_value_t* y = rocke_b_shl(b, tile2, c16);
        g1[4] = rocke_b_lor(b, x, y);
    }
    g1[5] = s1;
    g1[6] = rocke_b_shl(b, s2, c16);
    g1[7] = rocke_b_lshr(b, s2, c16);
    g2[0] = d2;
    g2[1] = ci(c, 0);
    g2[2] = ci(c, 0);
    g2[3] = ci(c, 0);
    {
        rocke_value_t* r[8];
        int i;
        for(i = 0; i < 4; ++i)
            r[i] = rocke_b_readfirstlane(b, g0[i]);
        out[0] = rocke_b_vec_pack(b, r, 4, rocke_i32());
        for(i = 0; i < 8; ++i)
            r[i] = rocke_b_readfirstlane(b, g1[i]);
        out[1] = rocke_b_vec_pack(b, r, 8, rocke_i32());
        for(i = 0; i < 4; ++i)
            r[i] = rocke_b_readfirstlane(b, g2[i]);
        out[2] = rocke_b_vec_pack(b, r, 4, rocke_i32());
    }
    out[3] = rocke_b_zero_vec(b, rocke_i32(), 4);
    out[4] = rocke_b_zero_vec(b, rocke_i32(), 8);
}

/* _issue(chunk): TDM chunk `chunk` into ring slot chunk % NS. */
void issue(dgw_ctx& c, rocke_value_t* chunk)
{
    rocke_ir_builder_t* b = c.b;
    const rocke_dconv_params_t* params = &c.params;
    rocke_value_t* stage = rocke_b_mod(b, chunk, ci(c, c.NS));
    rocke_value_t* k0 = rocke_b_mul(b, chunk, c.c_CK);
    rocke_value_t* k_ext;
    rocke_value_t* stage_off;
    {
        rocke_value_t* x = rocke_b_sub(b, c.c_kpg, k0);
        rocke_value_t* y = rocke_b_smin(b, x, c.c_CK);
        k_ext = rocke_b_smax(b, y, c.c0);
    }
    {
        rocke_value_t* hs = ci(c, c.HALO_STAGE);
        stage_off = rocke_b_mul(b, stage, hs);
    }
    for(int im = 0; im < c.NB; ++im)
    {
        rocke_value_t* cim = ci(c, im);
        rocke_value_t* cond = rocke_b_cmp_eq(b, c.wave_id, cim);
        rocke_if_t gate = rocke_b_scf_if(b, cond);
        rocke_b_region_enter(b, gate.then_region);
        {
            rocke_value_t* img_off;
            rocke_value_t* img;
            rocke_value_t* src;
            rocke_value_t* dst;
            rocke_value_t* h_ext;
            rocke_value_t* d[5];
            {
                rocke_value_t* o = ci(c, (int64_t)im * c.IMG);
                img_off = rocke_b_add(b, stage_off, o);
            }
            {
                rocke_value_t* x = rocke_b_add(b, c.dy_base, k0);
                rocke_value_t* y = ci(c, im);
                rocke_value_t* z = rocke_b_mul(b, y, params->p_A_stride_n);
                img = rocke_b_add(b, x, z);
            }
            {
                rocke_value_t* bytes = rocke_b_mul(b, img, c.c2);
                src = rocke_b_global_ptr_add(b, params->A, bytes);
            }
            {
                rocke_value_t* x = rocke_b_add(b, img_off, c.z_elems);
                rocke_value_t* y = rocke_b_mul(b, x, c.c2);
                dst = rocke_b_add(b, c.halo_addr, y);
            }
            h_ext = c.dy_h_ext;
            if(c.NB > 1)
            {
                rocke_value_t* x = ci(c, im);
                rocke_value_t* y = rocke_b_add(b, c.n_base, x);
                rocke_value_t* n_ok = rocke_b_cmp_lt(b, y, c.c_N);
                h_ext = rocke_b_select(b, n_ok, c.dy_h_ext, c.c0);
                rocke_value_t* base = rocke_b_global_ptr_add(b, params->A, c.c0);
                src = rocke_b_select(b, n_ok, src, base);
            }
            tdm_descriptor_3d(c,
                              src,
                              dst,
                              k_ext,
                              c.dy_w_ext,
                              h_ext,
                              params->p_A_stride_wi,
                              params->p_A_stride_hi,
                              c.XP,
                              c.HALO_W,
                              c.dy_tile2,
                              d);
            rocke_b_tensor_load_to_lds(b, d[0], d[1], d[2], d[3], d[4], 0);
        }
        rocke_b_region_leave(b);
    }
    if(c.G > 1)
        return; /* the packed weights are staged once, before the loop */
    {
        rocke_value_t* cnb = ci(c, c.NB);
        rocke_value_t* cond = rocke_b_cmp_eq(b, c.wave_id, cnb);
        rocke_if_t gate = rocke_b_scf_if(b, cond);
        rocke_b_region_enter(b, gate.then_region);
        {
            rocke_value_t* src;
            rocke_value_t* dst;
            rocke_value_t* d[5];
            {
                rocke_value_t* x = rocke_b_mul(b, k0, c.w_row);
                rocke_value_t* y = rocke_b_add(b, c.w_base, x);
                rocke_value_t* z = rocke_b_mul(b, y, c.c2);
                src = rocke_b_global_ptr_add(b, params->Bp, z);
            }
            {
                rocke_value_t* x = ci(c, (int64_t)c.WEI * 2);
                rocke_value_t* y = rocke_b_mul(b, stage, x);
                dst = rocke_b_add(b, c.wei_addr, y);
            }
            rocke_value_t* taps0 = ci(c, c.TAPS);
            rocke_value_t* taps1 = ci(c, c.TAPS);
            tdm_descriptor_3d(
                c, src, dst, c.c_ext, k_ext, taps0, c.w_row, c.c_cpg, c.WPITCH, c.CK, taps1, d);
            rocke_b_tensor_load_to_lds(b, d[0], d[1], d[2], d[3], d[4], 0);
        }
        rocke_b_region_leave(b);
    }
}

/* _read_b(stage_base, tn, tap_off): the dY (B) fragment of pixel tile tn. */
rocke_value_t* read_b(dgw_ctx& c, rocke_value_t* stage_base, int tn, int tap_off)
{
    rocke_ir_builder_t* b = c.b;
    rocke_value_t* t = ci(c, tap_off);
    rocke_value_t* x = rocke_b_add(b, c.px_off[tn], t);
    rocke_value_t* addr = rocke_b_add(b, stage_base, x);
    rocke_value_t* idx0[2] = {c.c0, addr};
    rocke_value_t* lo = rocke_b_smem_load_vN(b, c.lds, idx0, 2, c.io_type, 8);
    rocke_value_t* addr8 = rocke_b_add(b, addr, c.c8);
    rocke_value_t* idx1[2] = {c.c0, addr8};
    rocke_value_t* hi = rocke_b_smem_load_vN(b, c.lds, idx1, 2, c.io_type, 8);
    return rocke_b_vec_concat(b, lo, hi);
}

/* _read_a(stage_base, m, row_off): the W^T (A) fragment of channel tile m. */
rocke_value_t* read_a(dgw_ctx& c, rocke_value_t* stage_base, int m, int row_off)
{
    rocke_ir_builder_t* b = c.b;
    rocke_value_t* t = ci(c, row_off);
    rocke_value_t* x = rocke_b_add(b, c.wei_off[m], t);
    rocke_value_t* addr = rocke_b_add(b, stage_base, x);
    rocke_value_t* idx0[2] = {c.c0, addr};
    rocke_value_t* lo = rocke_b_ds_read_tr16_b128(b, c.lds, idx0, 2, c.io_type);
    rocke_value_t* o = ci(c, 8 * c.WPITCH);
    rocke_value_t* addr8 = rocke_b_add(b, addr, o);
    rocke_value_t* idx1[2] = {c.c0, addr8};
    rocke_value_t* hi = rocke_b_ds_read_tr16_b128(b, c.lds, idx1, 2, c.io_type);
    return rocke_b_vec_concat(b, lo, hi);
}

/* One step per (k-step, halo shift): the taps sharing a shift (strided
 * phases) share its B fragments. Stride 1 has one tap per shift. */
struct dgw_step
{
    int ks, dh, dw;
    std::vector<int> taps; /* r * KW + s */
    std::vector<int> phs; /* a_h * S + a_w */
};

std::vector<dgw_step> make_steps(const dgw_ctx& c)
{
    std::vector<dgw_step> steps;
    for(int ks = 0; ks < c.CK / DGW_WMMA_K; ++ks)
    {
        /* Python dict: shifts in first-seen order. */
        size_t first = steps.size();
        for(int r = 0; r < c.KH; ++r)
            for(int s = 0; s < c.KW; ++s)
            {
                int dh = c.plan_h.delta[r], dw = c.plan_w.delta[s];
                int ph = c.plan_h.phase[r] * c.S + c.plan_w.phase[s];
                size_t k = first;
                while(k < steps.size() && !(steps[k].dh == dh && steps[k].dw == dw))
                    ++k;
                if(k == steps.size())
                {
                    dgw_step st;
                    st.ks = ks;
                    st.dh = dh;
                    st.dw = dw;
                    steps.push_back(st);
                }
                steps[k].taps.push_back(r * c.KW + s);
                steps[k].phs.push_back(ph);
            }
    }
    return steps;
}

/* _frags(step): TN B fragments, then TM A fragments per tap of the step. */
void frags(dgw_ctx& c,
           rocke_value_t* halo_base,
           rocke_value_t* wei_base,
           const dgw_step& st,
           std::vector<std::vector<rocke_value_t*>>& af,
           std::vector<rocke_value_t*>& bf)
{
    int tap_off = (st.dh * c.HALO_W + st.dw) * c.XP + st.ks * DGW_WMMA_K;
    bf.assign(c.TN, NULL);
    af.assign(st.taps.size(), std::vector<rocke_value_t*>());
    for(int tn = 0; tn < c.TN; ++tn)
        bf[tn] = read_b(c, halo_base, tn, tap_off);
    for(size_t t = 0; t < st.taps.size(); ++t)
    {
        af[t].assign(c.TM, NULL);
        for(int m = 0; m < c.TM; ++m)
            af[t][m] = read_a(c, wei_base, m, (st.taps[t] * c.CK + st.ks * DGW_WMMA_K) * c.WPITCH);
    }
}

/* _wait_landed(kc): wait until this wave's transfer of chunk kc has landed. */
void wait_landed(dgw_ctx& c, rocke_value_t* kc)
{
    rocke_ir_builder_t* b = c.b;
    if(c.NS == 2)
    {
        rocke_b_s_wait_tensorcnt(b, 0);
    }
    else
    {
        rocke_value_t* x = ci(c, c.NS - 2);
        rocke_value_t* y = rocke_b_add(b, kc, x);
        rocke_value_t* more = rocke_b_cmp_lt(b, y, c.n_chunks);
        rocke_if_else_t ie = rocke_b_scf_if_else(b, more);
        rocke_b_region_enter(b, ie.then_region);
        rocke_b_s_wait_tensorcnt(b, c.NS - 2);
        rocke_b_region_leave(b);
        rocke_b_region_enter(b, ie.else_region);
        rocke_b_s_wait_tensorcnt(b, 0);
        rocke_b_region_leave(b);
    }
}

/* _signal_next(kc): split barrier, first half -- this wave's reads of chunk
 * kc are done and its transfer of chunk kc + 1 has landed. */
void signal_next(dgw_ctx& c, rocke_value_t* kc)
{
    rocke_ir_builder_t* b = c.b;
    rocke_b_s_waitcnt(b, -1, 0, -1);
    rocke_value_t* nx = rocke_b_add(b, kc, c.c1);
    wait_landed(c, nx);
    rocke_b_s_barrier_signal(b, 0xFFFFFFFFu);
}

/* _chunk(kc, accs): publish chunk kc, refill the ring, multiply the chunk. */
void chunk(dgw_ctx& c, rocke_value_t* kc, std::vector<rocke_value_t*>& accs)
{
    rocke_ir_builder_t* b = c.b;
    if(c.SB)
    {
        rocke_b_s_barrier_wait(b, 0xFFFF);
    }
    else
    {
        wait_landed(c, kc);
        rocke_b_sync_lds_only(b);
    }
    {
        rocke_value_t* x = ci(c, c.NS - 1);
        rocke_value_t* nxt = rocke_b_add(b, kc, x);
        rocke_value_t* cond = rocke_b_cmp_lt(b, nxt, c.n_chunks);
        rocke_if_t gate = rocke_b_scf_if(b, cond);
        rocke_b_region_enter(b, gate.then_region);
        issue(c, nxt);
        rocke_b_region_leave(b);
    }
    rocke_value_t* stage;
    rocke_value_t* halo_base;
    rocke_value_t* wei_base;
    {
        rocke_value_t* x = ci(c, c.NS);
        stage = rocke_b_mod(b, kc, x);
    }
    {
        rocke_value_t* x = ci(c, c.HALO_STAGE);
        halo_base = rocke_b_mul(b, stage, x);
    }
    {
        rocke_value_t* x = ci(c, c.WEI);
        wei_base = rocke_b_mul(b, stage, x);
    }
    std::vector<dgw_step> steps = make_steps(c);

    rocke_b_s_setprio(b, 1);
    std::vector<std::vector<rocke_value_t*>> af, naf;
    std::vector<rocke_value_t*> bf, nbf;
    if(c.spec->prefetch)
        frags(c, halo_base, wei_base, steps[0], naf, nbf);
    for(size_t j = 0; j < steps.size(); ++j)
    {
        if(c.spec->prefetch)
        {
            af = naf;
            bf = nbf;
            if(j + 1 < steps.size())
                frags(c, halo_base, wei_base, steps[j + 1], naf, nbf);
        }
        else
        {
            frags(c, halo_base, wei_base, steps[j], af, bf);
        }
        if(c.SB && j == steps.size() - 1)
            signal_next(c, kc);
        for(size_t t = 0; t < steps[j].taps.size(); ++t)
        {
            int ph = steps[j].phs[t];
            for(int m = 0; m < c.TM; ++m)
                for(int tn = 0; tn < c.TN; ++tn)
                {
                    int i = (ph * c.TM + m) * c.TN + tn;
                    accs[i] = rocke_b_mma(b, c.wmma_op_id, af[t][m], bf[tn], accs[i], NULL, 0);
                }
        }
    }
    rocke_b_s_setprio(b, 0);
}

/* _epilogue(accs): stage the dX tile in LDS, one TDM store per image. With
 * stride S the phases interleave into the dense S*TH x S*TW tile. */
void epilogue(dgw_ctx& c, rocke_value_t* const* accs)
{
    rocke_ir_builder_t* b = c.b;
    const rocke_dconv_params_t* params = &c.params;
    const int S = c.S;
    rocke_b_sync_lds_only(b);
    for(int ph = 0; ph < c.NPH; ++ph)
    {
        const int a_h = ph / S, a_w = ph % S;
        for(int m = 0; m < c.TM; ++m)
        {
            rocke_value_t* col;
            {
                rocke_value_t* x = ci(c, c.TM);
                rocke_value_t* y = rocke_b_mul(b, c.wave_c, x);
                rocke_value_t* z = ci(c, m);
                rocke_value_t* u = rocke_b_add(b, y, z);
                rocke_value_t* v = rocke_b_mul(b, u, c.c16);
                rocke_value_t* w = rocke_b_mul(b, c.lhalf, c.c8);
                col = rocke_b_add(b, v, w);
            }
            for(int tn = 0; tn < c.TN; ++tn)
            {
                rocke_value_t* px;
                {
                    rocke_value_t* x = ci(c, c.TN);
                    rocke_value_t* y = rocke_b_mul(b, c.wave_p, x);
                    rocke_value_t* z = ci(c, tn);
                    rocke_value_t* u = rocke_b_add(b, y, z);
                    rocke_value_t* v = rocke_b_mul(b, u, c.c16);
                    px = rocke_b_add(b, v, c.l16);
                }
                if(S > 1)
                {
                    /* Phase-grid pixel (ti, tj) -> dX tile pixel
                     * (S * ti + a_h, S * tj + a_w). */
                    rocke_value_t* f = px;
                    rocke_value_t* nb = NULL;
                    if(c.NB > 1)
                    {
                        rocke_value_t* x = ci(c, c.TH * c.TW);
                        nb = rocke_b_div(b, f, x);
                        rocke_value_t* y = ci(c, c.TH * c.TW);
                        f = rocke_b_mod(b, f, y);
                    }
                    rocke_value_t* ti;
                    rocke_value_t* tj;
                    {
                        rocke_value_t* x = ci(c, c.TW);
                        ti = rocke_b_div(b, f, x);
                        rocke_value_t* y = ci(c, c.TW);
                        tj = rocke_b_mod(b, f, y);
                    }
                    rocke_value_t* row;
                    {
                        rocke_value_t* x = ci(c, S);
                        rocke_value_t* y = rocke_b_mul(b, ti, x);
                        rocke_value_t* z = ci(c, a_h);
                        rocke_value_t* u = rocke_b_add(b, y, z);
                        rocke_value_t* v = ci(c, S * c.TW);
                        row = rocke_b_mul(b, u, v);
                    }
                    rocke_value_t* colx;
                    {
                        rocke_value_t* x = ci(c, S);
                        rocke_value_t* y = rocke_b_mul(b, tj, x);
                        rocke_value_t* z = ci(c, a_w);
                        colx = rocke_b_add(b, y, z);
                    }
                    px = rocke_b_add(b, row, colx);
                    if(c.NB > 1)
                    {
                        rocke_value_t* x = ci(c, S * S * c.TH * c.TW);
                        rocke_value_t* y = rocke_b_mul(b, nb, x);
                        px = rocke_b_add(b, px, y);
                    }
                }
                rocke_value_t* val
                    = rocke_b_vec_cast_f32_to(b, accs[(ph * c.TM + m) * c.TN + tn], c.io_type);
                rocke_value_t* x = ci(c, c.WPITCH);
                rocke_value_t* y = rocke_b_mul(b, px, x);
                rocke_value_t* z = rocke_b_add(b, y, col);
                rocke_value_t* idx[2] = {c.c0, z};
                rocke_b_smem_store_vN(b, c.lds, idx, 2, val, 8);
            }
        }
    }
    rocke_b_sync_lds_only(b);
    rocke_value_t* hx_tile = c.h_tile;
    rocke_value_t* wx_tile = c.w_tile;
    if(S != 1)
    {
        rocke_value_t* x = ci(c, S);
        hx_tile = rocke_b_mul(b, c.h_tile, x);
        rocke_value_t* y = ci(c, S);
        wx_tile = rocke_b_mul(b, c.w_tile, y);
    }
    for(int im = 0; im < c.NB; ++im)
    {
        rocke_value_t* cim = ci(c, im);
        rocke_value_t* cond = rocke_b_cmp_eq(b, c.wave_id, cim);
        rocke_if_t gate = rocke_b_scf_if(b, cond);
        rocke_b_region_enter(b, gate.then_region);
        {
            rocke_value_t* n_img;
            rocke_value_t* off;
            rocke_value_t* h_ext;
            rocke_value_t* dst;
            rocke_value_t* d[5];
            {
                rocke_value_t* x = ci(c, im);
                n_img = rocke_b_add(b, c.n_base, x);
            }
            {
                rocke_value_t* x = rocke_b_mul(b, n_img, params->p_D_stride_n);
                rocke_value_t* y = rocke_b_mul(b, hx_tile, params->p_D_stride_ho);
                rocke_value_t* z = rocke_b_add(b, x, y);
                rocke_value_t* u = rocke_b_mul(b, wx_tile, params->p_D_stride_wo);
                rocke_value_t* v = rocke_b_add(b, c.c_grp, c.c_base);
                rocke_value_t* w = rocke_b_add(b, u, v);
                off = rocke_b_add(b, z, w);
            }
            {
                rocke_value_t* x = rocke_b_sub(b, c.c_H, hx_tile);
                h_ext = rocke_b_smax(b, x, c.c0);
            }
            {
                rocke_value_t* x = rocke_b_mul(b, off, c.c2);
                dst = rocke_b_global_ptr_add(b, params->D, x);
            }
            if(c.NB > 1)
            {
                rocke_value_t* n_ok = rocke_b_cmp_lt(b, n_img, c.c_N);
                h_ext = rocke_b_select(b, n_ok, h_ext, c.c0);
                rocke_value_t* base = rocke_b_global_ptr_add(b, params->D, c.c0);
                dst = rocke_b_select(b, n_ok, dst, base);
            }
            rocke_value_t* lds_src;
            {
                rocke_value_t* x = ci(c, (int64_t)im * c.NPH * c.TH * c.TW * c.WPITCH * 2);
                lds_src = rocke_b_add(b, c.halo_addr, x);
            }
            rocke_value_t* w_ext;
            {
                rocke_value_t* x = rocke_b_sub(b, c.c_W, wx_tile);
                w_ext = rocke_b_smax(b, x, c.c0);
            }
            rocke_value_t* th = ci(c, S * c.TH);
            tdm_descriptor_3d(c,
                              dst,
                              lds_src,
                              c.c_ext,
                              w_ext,
                              h_ext,
                              params->p_D_stride_wo,
                              params->p_D_stride_ho,
                              c.WPITCH,
                              S * c.TW,
                              th,
                              d);
            rocke_b_tensor_store_from_lds(b, d[0], d[1], d[2], d[3], d[4], 0);
            rocke_b_s_wait_tensorcnt(b, 0);
        }
        rocke_b_region_leave(b);
    }
}

/* _epilogue_wave(accs): each wave stages its own tiles in a private LDS
 * region and stores them with its own TDM transfer (wave_epilogue). */
void epilogue_wave(dgw_ctx& c, rocke_value_t* const* accs)
{
    rocke_ir_builder_t* b = c.b;
    const rocke_dconv_params_t* params = &c.params;
    if(!c.SB)
        rocke_b_sync_lds_only(b); /* every wave is done with the ring */
    const int WPX = c.TN * 16;
    const int EP = c.TM * 16 + DGW_CH_PAD;
    const int ROWS = WPX / c.TW;
    rocke_value_t* base;
    {
        rocke_value_t* x = ci(c, (int64_t)WPX * EP);
        base = rocke_b_mul(b, c.wave_id, x);
    }
    for(int m = 0; m < c.TM; ++m)
    {
        rocke_value_t* col;
        {
            rocke_value_t* x = ci(c, m * 16);
            rocke_value_t* y = rocke_b_mul(b, c.lhalf, c.c8);
            col = rocke_b_add(b, x, y);
        }
        for(int tn = 0; tn < c.TN; ++tn)
        {
            rocke_value_t* px;
            {
                rocke_value_t* x = ci(c, tn * 16);
                px = rocke_b_add(b, x, c.l16);
            }
            rocke_value_t* val = rocke_b_vec_cast_f32_to(b, accs[m * c.TN + tn], c.io_type);
            rocke_value_t* x = ci(c, EP);
            rocke_value_t* y = rocke_b_mul(b, px, x);
            rocke_value_t* z = rocke_b_add(b, y, col);
            rocke_value_t* u = rocke_b_add(b, base, z);
            rocke_value_t* idx[2] = {c.c0, u};
            rocke_b_smem_store_vN(b, c.lds, idx, 2, val, 8);
        }
    }
    rocke_b_s_waitcnt(b, -1, 0, -1);
    rocke_value_t *px0, *img, *row0, *n_img, *h_w, *c_w, *off, *c_ext_w, *h_ext, *dst;
    {
        rocke_value_t* x = ci(c, WPX);
        px0 = rocke_b_mul(b, c.wave_p, x);
    }
    {
        rocke_value_t* x = ci(c, c.TH * c.TW);
        img = rocke_b_div(b, px0, x);
    }
    {
        rocke_value_t* x = ci(c, c.TH * c.TW);
        rocke_value_t* y = rocke_b_mod(b, px0, x);
        rocke_value_t* z = ci(c, c.TW);
        row0 = rocke_b_div(b, y, z);
    }
    n_img = rocke_b_add(b, c.n_base, img);
    h_w = rocke_b_add(b, c.h_tile, row0);
    {
        rocke_value_t* x = ci(c, c.TM * 16);
        rocke_value_t* y = rocke_b_mul(b, c.wave_c, x);
        c_w = rocke_b_add(b, c.c_base, y);
    }
    {
        rocke_value_t* x = rocke_b_mul(b, n_img, params->p_D_stride_n);
        rocke_value_t* y = rocke_b_mul(b, h_w, params->p_D_stride_ho);
        rocke_value_t* z = rocke_b_add(b, x, y);
        rocke_value_t* u = rocke_b_mul(b, c.w_tile, params->p_D_stride_wo);
        rocke_value_t* v = rocke_b_add(b, c.c_grp, c_w);
        rocke_value_t* w = rocke_b_add(b, u, v);
        off = rocke_b_add(b, z, w);
    }
    {
        rocke_value_t* x = rocke_b_sub(b, c.c_cpg, c_w);
        rocke_value_t* y = ci(c, c.TM * 16);
        rocke_value_t* z = rocke_b_smin(b, x, y);
        c_ext_w = rocke_b_smax(b, z, c.c0);
    }
    {
        rocke_value_t* x = rocke_b_sub(b, c.c_H, h_w);
        h_ext = rocke_b_smax(b, x, c.c0);
    }
    {
        rocke_value_t* x = rocke_b_mul(b, off, c.c2);
        dst = rocke_b_global_ptr_add(b, params->D, x);
    }
    if(c.NB > 1)
    {
        rocke_value_t* n_ok = rocke_b_cmp_lt(b, n_img, c.c_N);
        h_ext = rocke_b_select(b, n_ok, h_ext, c.c0);
        rocke_value_t* zb = rocke_b_global_ptr_add(b, params->D, c.c0);
        dst = rocke_b_select(b, n_ok, dst, zb);
    }
    rocke_value_t* lds_src;
    {
        rocke_value_t* x = rocke_b_mul(b, base, c.c2);
        lds_src = rocke_b_add(b, c.halo_addr, x);
    }
    rocke_value_t* w_ext;
    {
        rocke_value_t* x = rocke_b_sub(b, c.c_W, c.w_tile);
        w_ext = rocke_b_smax(b, x, c.c0);
    }
    rocke_value_t* rows = ci(c, ROWS);
    rocke_value_t* d[5];
    tdm_descriptor_3d(c,
                      dst,
                      lds_src,
                      c_ext_w,
                      w_ext,
                      h_ext,
                      params->p_D_stride_wo,
                      params->p_D_stride_ho,
                      EP,
                      c.TW,
                      rows,
                      d);
    rocke_b_tensor_store_from_lds(b, d[0], d[1], d[2], d[3], d[4], 0);
    rocke_b_s_wait_tensorcnt(b, 0);
}

/* Stage the super-group's block-diagonal weights into slot 0 once: zero the
 * slot, then each lane copies one CPG-wide row of one group (pack_groups). */
void stage_packed_weights(dgw_ctx& c)
{
    rocke_ir_builder_t* b = c.b;
    const rocke_dconv_params_t* params = &c.params;
    const int CPG0 = c.spec->problem.cpg, KPG0 = c.spec->problem.kpg;
    rocke_value_t* zero_v = rocke_b_zero_vec(b, c.io_type, 8);
    const int nvec = c.WEI / 8;
    for(int j = 0; j < nvec; j += c.THREADS)
    {
        rocke_value_t* cj = ci(c, j);
        rocke_value_t* v = rocke_b_add(b, c.tid, cj);
        rocke_value_t* cn = ci(c, nvec);
        rocke_value_t* cond = rocke_b_cmp_lt(b, v, cn);
        rocke_if_t gate = rocke_b_scf_if(b, cond);
        rocke_b_region_enter(b, gate.then_region);
        {
            rocke_value_t* x = ci(c, 8);
            rocke_value_t* y = rocke_b_mul(b, v, x);
            rocke_value_t* z = ci(c, c.WEI0);
            rocke_value_t* u = rocke_b_add(b, y, z);
            rocke_value_t* idx[2] = {c.c0, u};
            rocke_b_smem_store_vN(b, c.lds, idx, 2, zero_v, 8);
        }
        rocke_b_region_leave(b);
    }
    rocke_b_sync_lds_only(b);
    const int VEC = CPG0 < 8 ? CPG0 : 8;
    const int PARTS = CPG0 / VEC;
    const int KQ = c.G * KPG0;
    const int total = c.TAPS * KQ * PARTS;
    for(int j = 0; j < total; j += c.THREADS)
    {
        rocke_value_t* cj = ci(c, j);
        rocke_value_t* v = rocke_b_add(b, c.tid, cj);
        rocke_value_t* ct = ci(c, total);
        rocke_value_t* cond = rocke_b_cmp_lt(b, v, ct);
        rocke_if_t gate = rocke_b_scf_if(b, cond);
        rocke_b_region_enter(b, gate.then_region);
        {
            rocke_value_t *part, *rw, *tap, *kq, *gi, *pv, *gidx, *val, *loff;
            {
                rocke_value_t* x = ci(c, PARTS);
                part = rocke_b_mod(b, v, x);
                rocke_value_t* y = ci(c, PARTS);
                rw = rocke_b_div(b, v, y);
            }
            {
                rocke_value_t* x = ci(c, KQ);
                tap = rocke_b_div(b, rw, x);
                rocke_value_t* y = ci(c, KQ);
                kq = rocke_b_mod(b, rw, y);
            }
            {
                rocke_value_t* x = ci(c, KPG0);
                gi = rocke_b_div(b, kq, x);
            }
            {
                rocke_value_t* x = ci(c, VEC);
                pv = rocke_b_mul(b, part, x);
            }
            {
                rocke_value_t* x = rocke_b_add(b, c.k_grp, kq);
                rocke_value_t* y = ci(c, (int64_t)c.TAPS * CPG0);
                rocke_value_t* z = rocke_b_mul(b, x, y);
                rocke_value_t* u = ci(c, CPG0);
                rocke_value_t* w = rocke_b_mul(b, tap, u);
                rocke_value_t* q = rocke_b_add(b, w, pv);
                gidx = rocke_b_add(b, z, q);
            }
            val = rocke_b_global_load_vN(b, params->Bp, gidx, c.io_type, VEC, 0);
            {
                rocke_value_t* x = ci(c, c.CK);
                rocke_value_t* y = rocke_b_mul(b, tap, x);
                rocke_value_t* z = rocke_b_add(b, y, kq);
                rocke_value_t* u = ci(c, c.WPITCH);
                rocke_value_t* w = rocke_b_mul(b, z, u);
                rocke_value_t* q = ci(c, CPG0);
                rocke_value_t* r = rocke_b_mul(b, gi, q);
                rocke_value_t* t = rocke_b_add(b, r, pv);
                loff = rocke_b_add(b, w, t);
            }
            rocke_value_t* x = ci(c, c.WEI0);
            rocke_value_t* y = rocke_b_add(b, loff, x);
            rocke_value_t* idx[2] = {c.c0, y};
            rocke_b_smem_store_vN(b, c.lds, idx, 2, val, VEC);
        }
        rocke_b_region_leave(b);
    }
}

} // namespace

rocke_kernel_def_t* rocke_build_direct_dgrad_wmma(rocke_ir_builder_t* b,
                                                  const rocke_direct_dgrad_wmma_spec_t* spec,
                                                  const char* arch)
{
    char reason[512];
    dgw_ctx c;

    if(b == NULL || spec == NULL)
        return NULL;
    if(arch == NULL)
        arch = "gfx1250";
    if(!rocke_direct_dgrad_wmma_is_valid_spec(spec, arch, reason, sizeof reason))
    {
        std::string msg = std::string("invalid DirectDgradWmmaSpec for ") + arch + ": " + reason;
        rocke_i_set_err_msg(b, ROCKE_ERR_VALUE, msg.c_str());
        return NULL;
    }
    const rocke_direct_conv_problem_t* p = &spec->problem;
    c.b = b;
    c.spec = spec;
    c.KH = p->KH;
    c.KW = p->KW;
    c.S = p->stride;
    c.G = spec->pack_groups;
    c.NPH = dgw_phases(spec);
    c.SB = spec->split_barrier;
    c.plan_h = dgw_plan_h(spec);
    c.plan_w = dgw_plan_w(spec);
    c.PADD_H = c.plan_h.shift; /* padding of the equivalent forward pass */
    c.PADD_W = c.plan_w.shift;
    c.io_type = rocke_b_io_ir_type(b, dgw_dtype(spec));
    if(c.io_type == NULL)
        return NULL;
    {
        const char* ab = dgw_is_bf16(spec) ? "bf16" : "f16";
        const rocke_archtarget_t* target = rocke_archtarget_from_gfx(arch);
        const rocke_mmaop_t* op = rocke_archtarget_op_for_shape(
            target, "wmma", ab, ab, "fp32", 16, 16, DGW_WMMA_K, NULL);
        if(op == NULL || op->op_id == NULL)
        {
            rocke_i_set_err_msg(b, ROCKE_ERR_VALUE, "build_direct_dgrad_wmma: no WMMA atom");
            return NULL;
        }
        c.wmma_op_id = op->op_id;
    }
    c.WAVE = spec->wave_size;
    c.WC = spec->waves_c;
    c.WP = spec->waves_p;
    c.TM = spec->tiles_c;
    c.TN = spec->tiles_p;
    c.THREADS = rocke_direct_dgrad_wmma_threads_per_block(spec);
    c.BC = rocke_direct_dgrad_wmma_block_c(spec);
    c.TH = rocke_direct_dgrad_wmma_tile_h(spec);
    c.TW = spec->tile_w;
    c.HALO_H = dgw_halo_h(spec);
    c.HALO_W = dgw_halo_w(spec);
    c.CK = spec->ck;
    c.NS = spec->stages;
    c.XP = dgw_x_pitch(spec);
    c.WPITCH = dgw_w_pitch(spec);
    c.HALO = dgw_halo_elems(spec);
    c.WEI = dgw_wei_elems(spec);
    c.NB = spec->tile_n;
    c.IMG = dgw_image_elems(spec);
    c.HALO_STAGE = c.HALO + dgw_halo_slack(spec);
    c.TAPS = c.KH * c.KW;

    rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", c.THREADS);
    if(spec->sched != NULL)
        rocke_attr_set_str(b, &b->kernel->attrs, "scheduler_strategy", spec->sched);
    rocke_dconv_emit_params(b, &c.params, "dgrad", c.io_type);
    const rocke_dconv_params_t* params = &c.params;

    c.c0 = ci(c, 0);
    c.c1 = ci(c, 1);
    c.c2 = ci(c, 2);
    if(c.G > 1)
    {
        /* Super-groups of G groups; the packed channel counts are baked. */
        rocke_value_t* x = ci(c, c.G);
        c.c_groups = rocke_b_div(b, params->p_groups, x);
        c.c_cpg = ci(c, dgw_cpg_eff(spec));
        c.c_kpg = ci(c, dgw_kpg_eff(spec));
    }
    else
    {
        c.c_groups = params->p_groups;
        c.c_cpg = rocke_b_div(b, params->p_total_c, c.c_groups);
        c.c_kpg = rocke_b_div(b, params->p_total_k, c.c_groups);
    }
    c.c_N = params->p_N;
    c.c_H = params->p_Hi;
    c.c_W = params->p_Wi;
    c.c_Ho = params->p_Ho;
    c.c_Wo = params->p_Wo;

    c.tid = rocke_b_thread_id_x(b);
    {
        rocke_value_t* c_wave = ci(c, c.WAVE);
        c.wave_id = rocke_b_div(b, c.tid, c_wave);
        c.lane = rocke_b_mod(b, c.tid, c_wave);
    }
    c.c16 = ci(c, 16);
    c.l16 = rocke_b_mod(b, c.lane, c.c16);
    c.lhalf = rocke_b_div(b, c.lane, c.c16);
    {
        rocke_value_t* x = ci(c, c.WC);
        c.wave_c = rocke_b_mod(b, c.wave_id, x);
        rocke_value_t* y = ci(c, c.WC);
        c.wave_p = rocke_b_div(b, c.wave_id, y);
    }

    /* ---- tile of this block. */
    rocke_value_t *n_ct, *n_ht, *n_wt, *c_Hq, *c_Wq;
    {
        rocke_value_t* x = ci(c, c.BC - 1);
        rocke_value_t* y = rocke_b_add(b, c.c_cpg, x);
        rocke_value_t* z = ci(c, c.BC);
        n_ct = rocke_b_div(b, y, z);
    }
    if(c.S > 1)
    {
        /* Tiles index the phase grid: output row S * i + a, i < ceil(H / S). */
        rocke_value_t* x = ci(c, c.S - 1);
        rocke_value_t* y = rocke_b_add(b, c.c_H, x);
        rocke_value_t* z = ci(c, c.S);
        c_Hq = rocke_b_div(b, y, z);
        rocke_value_t* u = ci(c, c.S - 1);
        rocke_value_t* v = rocke_b_add(b, c.c_W, u);
        rocke_value_t* w = ci(c, c.S);
        c_Wq = rocke_b_div(b, v, w);
    }
    else
    {
        c_Hq = c.c_H;
        c_Wq = c.c_W;
    }
    {
        rocke_value_t* x = ci(c, c.TH - 1);
        rocke_value_t* y = rocke_b_add(b, c_Hq, x);
        rocke_value_t* z = ci(c, c.TH);
        n_ht = rocke_b_div(b, y, z);
    }
    {
        rocke_value_t* x = ci(c, c.TW - 1);
        rocke_value_t* y = rocke_b_add(b, c_Wq, x);
        rocke_value_t* z = ci(c, c.TW);
        n_wt = rocke_b_div(b, y, z);
    }
    rocke_value_t* t = rocke_b_block_id_x(b);
    rocke_value_t* ct = rocke_b_mod(b, t, n_ct);
    t = rocke_b_div(b, t, n_ct);
    rocke_value_t* wt = rocke_b_mod(b, t, n_wt);
    t = rocke_b_div(b, t, n_wt);
    rocke_value_t* ht = rocke_b_mod(b, t, n_ht);
    t = rocke_b_div(b, t, n_ht);
    rocke_value_t* g = rocke_b_mod(b, t, c.c_groups);
    {
        rocke_value_t* x = rocke_b_div(b, t, c.c_groups);
        rocke_value_t* y = ci(c, c.NB);
        c.n_base = rocke_b_mul(b, x, y);
    }
    {
        rocke_value_t* x = ci(c, c.BC);
        c.c_base = rocke_b_mul(b, ct, x);
    }
    {
        rocke_value_t* x = ci(c, c.TH);
        c.h_tile = rocke_b_mul(b, ht, x);
    }
    {
        rocke_value_t* x = ci(c, c.TW);
        c.w_tile = rocke_b_mul(b, wt, x);
    }
    c.k_grp = rocke_b_mul(b, g, c.c_kpg);
    c.c_grp = rocke_b_mul(b, g, c.c_cpg);
    {
        rocke_value_t* x = ci(c, c.TAPS);
        c.w_row = rocke_b_mul(b, c.c_cpg, x);
    }
    c.c_CK = ci(c, c.CK);
    {
        rocke_value_t* x = ci(c, c.CK - 1);
        rocke_value_t* y = rocke_b_add(b, c.c_kpg, x);
        c.n_chunks = rocke_b_div(b, y, c.c_CK);
    }

    /* ---- one LDS block: the halo ring, then the weight ring. */
    c.WEI0 = c.NS * c.HALO_STAGE;
    {
        int shape[2] = {1, (int)(rocke_direct_dgrad_wmma_lds_bytes(spec) / 2)};
        c.lds = rocke_b_smem_alloc(b, c.io_type, shape, 2, "dg_lds");
    }
    {
        rocke_value_t* a = rocke_b_smem_addr_of(b, c.lds);
        c.halo_addr = rocke_b_trunc(b, a, rocke_i32());
        rocke_value_t* x = ci(c, (int64_t)c.WEI0 * 2);
        c.wei_addr = rocke_b_add(b, c.halo_addr, x);
    }

    /* ---- halo origin and edge shifts. */
    rocke_value_t *h0, *w0, *hs, *sw, *h_start, *w_start;
    {
        rocke_value_t* x = ci(c, c.PADD_H);
        h0 = rocke_b_sub(b, c.h_tile, x);
    }
    {
        rocke_value_t* x = ci(c, c.PADD_W);
        w0 = rocke_b_sub(b, c.w_tile, x);
    }
    {
        rocke_value_t* x = rocke_b_sub(b, c.c0, h0);
        hs = rocke_b_smax(b, x, c.c0);
    }
    {
        rocke_value_t* x = rocke_b_sub(b, c.c0, w0);
        sw = rocke_b_smax(b, x, c.c0);
    }
    h_start = rocke_b_add(b, h0, hs);
    w_start = rocke_b_add(b, w0, sw);
    {
        rocke_value_t* x = ci(c, c.HALO_W);
        rocke_value_t* y = rocke_b_mul(b, hs, x);
        rocke_value_t* z = rocke_b_add(b, y, sw);
        rocke_value_t* u = ci(c, c.XP);
        c.z_elems = rocke_b_mul(b, z, u);
    }
    {
        rocke_value_t* x = rocke_b_sub(b, c.c_Wo, w_start);
        rocke_value_t* y = ci(c, c.HALO_W);
        rocke_value_t* z = rocke_b_sub(b, y, sw);
        rocke_value_t* u = rocke_b_smin(b, x, z);
        c.dy_w_ext = rocke_b_smax(b, u, c.c0);
    }
    {
        rocke_value_t* x = rocke_b_sub(b, c.c_Ho, h_start);
        c.dy_h_ext = rocke_b_smax(b, x, c.c0);
    }
    {
        rocke_value_t* x = ci(c, c.HALO_H);
        c.dy_tile2 = rocke_b_sub(b, x, hs);
    }
    {
        rocke_value_t* x = rocke_b_mul(b, c.n_base, params->p_A_stride_n);
        rocke_value_t* y = rocke_b_mul(b, h_start, params->p_A_stride_hi);
        rocke_value_t* z = rocke_b_add(b, x, y);
        rocke_value_t* u = rocke_b_mul(b, w_start, params->p_A_stride_wi);
        rocke_value_t* v = rocke_b_add(b, u, c.k_grp);
        c.dy_base = rocke_b_add(b, z, v);
    }
    {
        rocke_value_t* x = rocke_b_sub(b, c.c_cpg, c.c_base);
        rocke_value_t* y = ci(c, c.BC);
        rocke_value_t* z = rocke_b_smin(b, x, y);
        c.c_ext = rocke_b_smax(b, z, c.c0);
    }
    {
        rocke_value_t* x = rocke_b_mul(b, c.k_grp, c.w_row);
        c.w_base = rocke_b_add(b, x, c.c_base);
    }

    /* ---- zero the halo cells no transfer writes (top / left image edge). */
    {
        int max_z = (c.PADD_H * c.HALO_W + c.PADD_W) * c.XP;
        if(max_z)
        {
            rocke_value_t* zero_v = rocke_b_zero_vec(b, c.io_type, 8);
            rocke_value_t* c8z = ci(c, 8);
            rocke_value_t* z_vecs = rocke_b_div(b, c.z_elems, c8z);
            for(int j = 0; j < dgw_ceil_div(max_z, 8); j += c.THREADS)
            {
                rocke_value_t* cj = ci(c, j);
                rocke_value_t* v = rocke_b_add(b, c.tid, cj);
                rocke_value_t* cond = rocke_b_cmp_lt(b, v, z_vecs);
                rocke_if_t gate = rocke_b_scf_if(b, cond);
                rocke_b_region_enter(b, gate.then_region);
                {
                    rocke_value_t* x = ci(c, 8);
                    rocke_value_t* flat = rocke_b_mul(b, v, x);
                    for(int st = 0; st < c.NS; ++st)
                        for(int im = 0; im < c.NB; ++im)
                        {
                            rocke_value_t* o
                                = ci(c, (int64_t)st * c.HALO_STAGE + (int64_t)im * c.IMG);
                            rocke_value_t* a = rocke_b_add(b, flat, o);
                            rocke_value_t* idx[2] = {c.c0, a};
                            rocke_b_smem_store_vN(b, c.lds, idx, 2, zero_v, 8);
                        }
                }
                rocke_b_region_leave(b);
            }
        }
    }

    /* ---- per-lane LDS read offsets. */
    c.px_off.assign(c.TN, NULL);
    for(int tn = 0; tn < c.TN; ++tn)
    {
        rocke_value_t* flat;
        rocke_value_t* nb;
        {
            rocke_value_t* x = ci(c, c.TN);
            rocke_value_t* y = rocke_b_mul(b, c.wave_p, x);
            rocke_value_t* z = ci(c, tn);
            rocke_value_t* u = rocke_b_add(b, y, z);
            rocke_value_t* v = rocke_b_mul(b, u, c.c16);
            flat = rocke_b_add(b, v, c.l16);
        }
        if(c.NB > 1)
        {
            rocke_value_t* x = ci(c, c.TH * c.TW);
            nb = rocke_b_div(b, flat, x);
            rocke_value_t* y = ci(c, c.TH * c.TW);
            flat = rocke_b_mod(b, flat, y);
        }
        else
        {
            nb = c.c0;
        }
        rocke_value_t* th;
        rocke_value_t* tw;
        {
            rocke_value_t* x = ci(c, c.TW);
            th = rocke_b_div(b, flat, x);
            rocke_value_t* y = ci(c, c.TW);
            tw = rocke_b_mod(b, flat, y);
        }
        rocke_value_t* off;
        {
            rocke_value_t* x = ci(c, c.HALO_W);
            rocke_value_t* y = rocke_b_mul(b, th, x);
            rocke_value_t* z = rocke_b_add(b, y, tw);
            rocke_value_t* u = ci(c, c.XP);
            rocke_value_t* v = rocke_b_mul(b, z, u);
            rocke_value_t* w = rocke_b_mul(b, c.lhalf, c.c16);
            off = rocke_b_add(b, v, w);
        }
        if(c.NB > 1)
        {
            rocke_value_t* x = ci(c, c.IMG);
            rocke_value_t* y = rocke_b_mul(b, nb, x);
            off = rocke_b_add(b, off, y);
        }
        c.px_off[tn] = off;
    }
    c.c8 = ci(c, 8);
    rocke_value_t *tr_col, *tr_row;
    {
        rocke_value_t* x = rocke_b_div(b, c.l16, c.c8);
        tr_col = rocke_b_mul(b, x, c.c8);
    }
    {
        rocke_value_t* x = rocke_b_mul(b, c.lhalf, c.c16);
        rocke_value_t* y = rocke_b_mod(b, c.lane, c.c8);
        tr_row = rocke_b_add(b, x, y);
    }
    c.wei_off.assign(c.TM, NULL);
    for(int m = 0; m < c.TM; ++m)
    {
        rocke_value_t* col;
        {
            rocke_value_t* x = ci(c, c.TM);
            rocke_value_t* y = rocke_b_mul(b, c.wave_c, x);
            rocke_value_t* z = ci(c, m);
            rocke_value_t* u = rocke_b_add(b, y, z);
            col = rocke_b_mul(b, u, c.c16);
        }
        rocke_value_t* x = ci(c, c.WPITCH);
        rocke_value_t* y = rocke_b_mul(b, tr_row, x);
        rocke_value_t* z = ci(c, c.WEI0);
        rocke_value_t* u = rocke_b_add(b, tr_col, z);
        rocke_value_t* v = rocke_b_add(b, col, u);
        c.wei_off[m] = rocke_b_add(b, y, v);
    }

    if(c.G > 1)
        stage_packed_weights(c);

    /* ---- main loop over dY channel chunks. */
    for(int st = 0; st < c.NS - 1; ++st)
    {
        rocke_value_t* x = ci(c, st);
        rocke_value_t* cond = rocke_b_cmp_lt(b, x, c.n_chunks);
        rocke_if_t gate = rocke_b_scf_if(b, cond);
        rocke_b_region_enter(b, gate.then_region);
        issue(c, ci(c, st));
        rocke_b_region_leave(b);
    }
    if(c.SB)
    {
        rocke_b_s_waitcnt(b, -1, 0, -1); /* the edge zeroing */
        wait_landed(c, c.c0);
        rocke_b_s_barrier_signal(b, 0xFFFFFFFFu);
    }
    int n_acc = c.NPH * c.TM * c.TN;
    rocke_value_t* zero_acc = rocke_b_zero_vec_f32(b, 8);
    std::vector<std::string> names(n_acc);
    std::vector<rocke_iter_arg_t> iargs(n_acc);
    for(int i = 0; i < n_acc; ++i)
    {
        names[i] = "acc" + std::to_string(i);
        iargs[i].name = names[i].c_str();
        iargs[i].init = zero_acc;
    }
    rocke_for_t loop
        = rocke_b_scf_for_iter(b, c.c0, c.n_chunks, c.c1, iargs.data(), n_acc, "kc", false, true);
    if(!rocke_ir_builder_ok(b))
        return NULL;
    rocke_b_region_enter(b, loop.body);
    {
        std::vector<rocke_value_t*> accs(loop.iter_vars, loop.iter_vars + n_acc);
        chunk(c, loop.iv, accs);
        rocke_b_scf_yield(b, accs.data(), n_acc);
    }
    rocke_b_region_leave(b);
    if(c.SB)
        rocke_b_s_barrier_wait(b, 0xFFFF);
    if(spec->wave_epilogue)
        epilogue_wave(c, loop.op->results);
    else
        epilogue(c, loop.op->results);
    if(!rocke_ir_builder_ok(b))
        return NULL;
    return b->kernel;
}

rocke_kernel_def_t* rocke_build_direct_dgrad_wmma_new(rocke_ir_builder_t* b,
                                                      const rocke_direct_dgrad_wmma_spec_t* spec,
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
            rocke_i_set_err_msg(b, ROCKE_ERR_VALUE, "build_direct_dgrad_wmma: null spec");
            return NULL;
        }
        if(rocke_direct_dgrad_wmma_kernel_name(spec, name, sizeof(name)) != ROCKE_OK)
        {
            rocke_i_set_err_msg(
                b, ROCKE_ERR_VALUE, "build_direct_dgrad_wmma: kernel name does not fit");
            return NULL;
        }
        if(rocke_ir_builder_init(b, name) != ROCKE_OK)
            return NULL;
        return rocke_build_direct_dgrad_wmma(b, spec, arch);
    });
}

rocke_status_t rocke_direct_dgrad_wmma_lower_to_llvm(const rocke_direct_dgrad_wmma_spec_t* spec,
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
        dgw_set_reason(err, err_cap, "lower_to_llvm: null spec/out");
        return ROCKE_ERR_VALUE;
    }
    if(arch == NULL)
        arch = "gfx1250";
    kernel = rocke_build_direct_dgrad_wmma_new(&b, spec, arch);
    if(kernel == NULL)
    {
        const char* m = rocke_ir_builder_error(&b);
        st = rocke_ir_builder_status(&b);
        dgw_set_reason(
            err, err_cap, (m != NULL && m[0] != '\0') ? m : "build_direct_dgrad_wmma failed");
        rocke_ir_builder_free(&b);
        return (st == ROCKE_OK) ? ROCKE_ERR_VALUE : st;
    }
    st = rocke_lower_kernel_to_llvm_ex(kernel, flavor, arch, out_ll, err, err_cap);
    rocke_ir_builder_free(&b);
    return st;
}
