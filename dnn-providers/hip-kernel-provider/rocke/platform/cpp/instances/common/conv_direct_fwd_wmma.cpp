// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * conv_direct_fwd_wmma.cpp -- C++ port of build_direct_fwd_wmma
 * (library/kernels/common/conv_direct_fwd_wmma.py): the wave32 WMMA direct
 * forward convolution for gfx1250. See the header and the Python module for
 * the algorithm and every knob.
 *
 * Every helper emits in PYTHON EVALUATION ORDER (arguments left-to-right,
 * innermost call first) so the SSA sequence -- and therefore the .ll bytes --
 * match exactly. Nested builder calls are hoisted into ordered locals: C++
 * leaves the evaluation order of call arguments unspecified.
 */
#include "rocke/instance_conv_direct_fwd_wmma.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <functional>
#include <map>
#include <string>
#include <tuple>
#include <vector>

#include "rocke/error_boundary.hpp" /* ckc::guard_builder boundary shim */
#include "rocke/helper_rocke.core.arch.h" /* rocke_archtarget_from_gfx, op_for_shape */
#include "rocke/helper_rocke.helpers.io.h" /* rocke_b_io_ir_type */
#include "rocke/helper_rocke.helpers.spec.h" /* rocke_kernel_name_join */
#include "rocke/instance_conv_direct_grouped_internal.h" /* rocke_dconv_emit_params */
#include "rocke/instance_conv_fwd_wmma_gate_internal.h" /* rocke_wmma_fwd_gate */
#include "rocke/ir_internal.h" /* rocke_i_set_err_msg */
#include "rocke/lower_llvm.h"

/* conv_direct_fwd_wmma.WMMA_K (conv_fwd_wmma_gate's) / CH_PAD /
 * FWD_WMMA_VGPR_BUDGET. */
#define FWW_WMMA_K ROCKE_FWD_WMMA_K
#define FWW_CH_PAD 8
#define FWW_VGPR_BUDGET 512

namespace
{

using V = rocke_value_t*;

int fww_ceil_div(int a, int b)
{
    return (a + b - 1) / b;
}

/* Python floor division (the output extents of a too-small input). */
long long fww_floordiv(long long a, long long b)
{
    long long q = a / b;
    if((a % b != 0) && ((a < 0) != (b < 0)))
        --q;
    return q;
}

void fww_set_reason(char* reason, size_t reason_cap, const char* msg)
{
    if(reason && reason_cap > 0)
    {
        snprintf(reason, reason_cap, "%s", msg);
    }
}

const char* fww_dtype(const rocke_direct_fwd_wmma_spec_t* spec)
{
    return spec->problem.dtype ? spec->problem.dtype : "fp16";
}

bool fww_is_bf16(const rocke_direct_fwd_wmma_spec_t* spec)
{
    return strcmp(fww_dtype(spec), "bf16") == 0;
}

/* DirectConvProblem.Ho / Wo (strided, dilated). */
long long fww_Ho(const rocke_direct_conv_problem_t* p)
{
    return fww_floordiv((long long)p->H + 2 * p->PAD - (long long)(p->KH - 1) * p->dil_h - 1,
                        p->stride)
           + 1;
}
long long fww_Wo(const rocke_direct_conv_problem_t* p)
{
    return fww_floordiv((long long)p->W + 2 * p->PAD - (long long)(p->KW - 1) * p->dil_w - 1,
                        p->stride)
           + 1;
}

/* Spec @property helpers the builder shares. */
int fww_ring_taps(const rocke_direct_fwd_wmma_spec_t* s)
{
    if(s->tap_ring)
        return 1;
    return s->row_ring ? s->problem.KW : s->problem.KH * s->problem.KW;
}
int fww_phases_w(const rocke_direct_fwd_wmma_spec_t* s)
{
    return s->phase_split ? s->problem.stride : 1;
}
int fww_halo_h(const rocke_direct_fwd_wmma_spec_t* s)
{
    const rocke_direct_conv_problem_t* p = &s->problem;
    return p->stride * (rocke_direct_fwd_wmma_tile_h(s) - 1) + (p->KH - 1) * p->dil_h + 1;
}
int fww_halo_w(const rocke_direct_fwd_wmma_spec_t* s)
{
    const rocke_direct_conv_problem_t* p = &s->problem;
    return p->stride * (s->tile_w - 1) + (p->KW - 1) * p->dil_w + 1;
}
int fww_halo_cols(const rocke_direct_fwd_wmma_spec_t* s)
{
    const rocke_direct_conv_problem_t* p = &s->problem;
    if(s->phase_split)
        return s->tile_w + (int)fww_floordiv((long long)(p->KW - 1) * p->dil_w, p->stride);
    return fww_halo_w(s);
}
int fww_lead_cols(const rocke_direct_fwd_wmma_spec_t* s)
{
    const rocke_direct_conv_problem_t* p = &s->problem;
    return s->phase_split ? fww_ceil_div(p->PAD, p->stride) : p->PAD;
}
int fww_x_pitch(const rocke_direct_fwd_wmma_spec_t* s)
{
    return s->ck + FWW_CH_PAD;
}
int fww_epi_pitch(const rocke_direct_fwd_wmma_spec_t* s)
{
    return rocke_direct_fwd_wmma_block_k(s) + FWW_CH_PAD;
}
int fww_phase_elems(const rocke_direct_fwd_wmma_spec_t* s)
{
    return fww_halo_h(s) * fww_halo_cols(s) * fww_x_pitch(s) + fww_lead_cols(s) * fww_x_pitch(s);
}
int fww_cpg_eff(const rocke_direct_fwd_wmma_spec_t* s)
{
    return s->pack_groups * s->problem.cpg;
}
int fww_kpg_eff(const rocke_direct_fwd_wmma_spec_t* s)
{
    return s->pack_groups * s->problem.kpg;
}
int fww_groups_eff(const rocke_direct_fwd_wmma_spec_t* s)
{
    /* Python floor division; pack_groups >= 1 for every spec that builds. */
    return s->pack_groups > 0 ? (int)fww_floordiv(s->problem.groups, s->pack_groups) : 0;
}
int fww_image_elems(const rocke_direct_fwd_wmma_spec_t* s)
{
    if(s->phase_split)
        return fww_phases_w(s) * fww_phase_elems(s);
    return fww_halo_h(s) * fww_halo_w(s) * fww_x_pitch(s);
}
int fww_halo_elems(const rocke_direct_fwd_wmma_spec_t* s)
{
    return s->tile_n * fww_image_elems(s);
}
int fww_wei_elems(const rocke_direct_fwd_wmma_spec_t* s)
{
    return fww_ring_taps(s) * rocke_direct_fwd_wmma_block_k(s) * fww_x_pitch(s);
}
int fww_halo_slack(const rocke_direct_fwd_wmma_spec_t* s)
{
    return s->phase_split ? 0 : s->problem.PAD * fww_x_pitch(s);
}

} // namespace

/* ===================================================================== *
 *  Spec defaults + derived geometry
 * ===================================================================== */

rocke_direct_fwd_wmma_spec_t rocke_direct_fwd_wmma_spec_default(void)
{
    rocke_direct_fwd_wmma_spec_t s;
    memset(&s, 0, sizeof(s));
    s.problem = rocke_direct_conv_problem_default();
    s.name = "direct_fwd_wmma";
    s.waves_k = 2;
    s.waves_p = 4;
    s.tiles_k = 4;
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
    s.phase_split = false;
    s.row_ring = false;
    s.tap_ring = false;
    s.pingpong = false;
    s.pp_group = 1;
    s.tap_reuse = false;
    s.tap_shift = false;
    return s;
}

int rocke_direct_fwd_wmma_block_k(const rocke_direct_fwd_wmma_spec_t* spec)
{
    return spec->waves_k * spec->tiles_k * 16;
}

int rocke_direct_fwd_wmma_block_p(const rocke_direct_fwd_wmma_spec_t* spec)
{
    return spec->waves_p * spec->tiles_p * 16;
}

int rocke_direct_fwd_wmma_tile_h(const rocke_direct_fwd_wmma_spec_t* spec)
{
    int d = spec->tile_w * spec->tile_n;
    return d > 0 ? (int)fww_floordiv(rocke_direct_fwd_wmma_block_p(spec), d) : 0;
}

int rocke_direct_fwd_wmma_threads_per_block(const rocke_direct_fwd_wmma_spec_t* spec)
{
    return spec->waves_k * spec->waves_p * spec->wave_size;
}

long rocke_direct_fwd_wmma_lds_bytes(const rocke_direct_fwd_wmma_spec_t* spec)
{
    long ring;
    long halo = (long)fww_halo_elems(spec) + fww_halo_slack(spec);
    if(spec->row_ring || spec->tap_ring)
        ring = 2 * halo + (long)spec->stages * fww_wei_elems(spec);
    else
        ring = (long)spec->stages * (halo + fww_wei_elems(spec));
    long epi = (long)rocke_direct_fwd_wmma_block_p(spec) * fww_epi_pitch(spec);
    if(spec->wave_epilogue)
    {
        long n_waves = (long)spec->waves_k * spec->waves_p;
        epi = n_waves * spec->tiles_p * 16 * (spec->tiles_k * 16 + FWW_CH_PAD);
    }
    return 2 * (ring > epi ? ring : epi);
}

void rocke_direct_fwd_wmma_grid(const rocke_direct_fwd_wmma_spec_t* spec, int grid[3])
{
    const rocke_direct_conv_problem_t* p = &spec->problem;
    long long n_kt = fww_ceil_div(fww_kpg_eff(spec), rocke_direct_fwd_wmma_block_k(spec));
    long long n_ht = fww_ceil_div((int)fww_Ho(p), rocke_direct_fwd_wmma_tile_h(spec));
    long long n_wt = fww_ceil_div((int)fww_Wo(p), spec->tile_w);
    long long g
        = (long long)fww_ceil_div(p->N, spec->tile_n) * fww_groups_eff(spec) * n_ht * n_wt * n_kt;
    grid[0] = (int)g;
    grid[1] = 1;
    grid[2] = 1;
}

rocke_status_t rocke_direct_fwd_wmma_kernel_name(const rocke_direct_fwd_wmma_spec_t* spec,
                                                 char* out,
                                                 size_t out_cap)
{
    char f[32], pad[32], wk[32], wp[32], tk[32], tp[32], tw[32], tn[32], ck[32], st[32];
    char str[32], dil[48], pg[64], ppg[32];
    std::string sched;
    const rocke_direct_conv_problem_t* p = &spec->problem;
    snprintf(f, sizeof f, "f%dx%d", p->KH, p->KW);
    snprintf(pad, sizeof pad, "p%d", p->PAD);
    snprintf(wk, sizeof wk, "wk%d", spec->waves_k);
    snprintf(wp, sizeof wp, "wp%d", spec->waves_p);
    snprintf(tk, sizeof tk, "tk%d", spec->tiles_k);
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
    if(spec->pp_group > 1)
        snprintf(ppg, sizeof ppg, "pg%d", spec->pp_group);
    else
        ppg[0] = '\0';
    const char* parts[] = {f,
                           pad,
                           wk,
                           wp,
                           tk,
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
                           spec->wave_epilogue ? "we" : "",
                           spec->phase_split ? "ps" : "",
                           spec->row_ring ? "rr" : "",
                           spec->tap_ring ? "tr" : "",
                           spec->pingpong ? "pp" : "",
                           ppg,
                           spec->tap_reuse ? "tu" : "",
                           spec->tap_shift ? "tsh" : ""};
    const char* flag_names[] = {"bf16"};
    int flag_on[] = {fww_is_bf16(spec) ? 1 : 0};
    return rocke_kernel_name_join(spec->name ? spec->name : "direct_fwd_wmma",
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
 *  fwd_wmma_reason / is_valid_fwd_wmma_spec
 * ===================================================================== */

#define FWW_FAIL(...)                            \
    do                                           \
    {                                            \
        snprintf(msg, sizeof msg, __VA_ARGS__);  \
        fww_set_reason(reason, reason_cap, msg); \
        return false;                            \
    } while(0)

bool rocke_direct_fwd_wmma_is_valid_spec(const rocke_direct_fwd_wmma_spec_t* spec,
                                         const char* arch,
                                         char* reason,
                                         size_t reason_cap)
{
    char msg[512];
    const rocke_archtarget_t* target;
    const rocke_direct_conv_problem_t* p;

    if(spec == NULL)
    {
        fww_set_reason(reason, reason_cap, "null spec");
        return false;
    }
    if(arch == NULL)
        arch = "gfx1250";
    p = &spec->problem;
    if(!rocke_wmma_fwd_gate(
           p, arch, "direct_fwd_wmma", spec->wave_size, &target, reason, reason_cap))
        return false;
    if(p->stride != 1 && p->stride != 2)
        FWW_FAIL("stride %d not in (1, 2)", p->stride);
    if(p->dil_h < 1 || p->dil_w < 1)
        FWW_FAIL("dilation %dx%d < 1", p->dil_h, p->dil_w);
    if(p->KH != p->KW)
        FWW_FAIL("filter %dx%d is not square", p->KH, p->KW);
    {
        int dmax = p->dil_h > p->dil_w ? p->dil_h : p->dil_w;
        if(p->PAD < 0 || p->PAD > (p->KH - 1) * dmax)
            FWW_FAIL("PAD %d outside 0..(K-1)*dilation", p->PAD);
    }
    long long Ho = fww_Ho(p), Wo = fww_Wo(p);
    if(Ho < 1 || Wo < 1)
        FWW_FAIL("empty output %lldx%lld", Ho, Wo);
    int G = spec->pack_groups;
    if(G < 1)
        FWW_FAIL("pack_groups %d < 1", G);
    if(G > 1)
    {
        bool cok = p->cpg == 2 || p->cpg == 4 || p->cpg == 8 || p->cpg == 16;
        bool kok = p->kpg == 2 || p->kpg == 4 || p->kpg == 8 || p->kpg == 16;
        if(p->groups % G)
            FWW_FAIL("groups %d not a multiple of pack_groups %d", p->groups, G);
        if(!cok || !kok)
            FWW_FAIL("packed cpg %d / kpg %d not in (2, 4, 8, 16)", p->cpg, p->kpg);
        if(fww_cpg_eff(spec) > spec->ck)
            FWW_FAIL("packed cpg %d > ck %d (one X chunk)", fww_cpg_eff(spec), spec->ck);
        if(fww_kpg_eff(spec) > rocke_direct_fwd_wmma_block_k(spec))
            FWW_FAIL("packed kpg %d > block_k %d",
                     fww_kpg_eff(spec),
                     rocke_direct_fwd_wmma_block_k(spec));
    }
    if(fww_cpg_eff(spec) % 8 || fww_kpg_eff(spec) % 8)
        FWW_FAIL("cpg %d and kpg %d must be multiples of 8 (16-byte rows)",
                 fww_cpg_eff(spec),
                 fww_kpg_eff(spec));
    if(spec->ck % FWW_WMMA_K || spec->ck <= 0)
        FWW_FAIL("ck %d must be a positive multiple of %d", spec->ck, FWW_WMMA_K);
    if(spec->waves_k < 1 || spec->waves_p < 1 || spec->tiles_k < 1 || spec->tiles_p < 1)
        FWW_FAIL("waves and tiles must be >= 1");
    if(spec->stages < 2)
        FWW_FAIL("stages must be >= 2");
    int tw = spec->tile_w;
    int block_p = rocke_direct_fwd_wmma_block_p(spec);
    int nw = spec->waves_k * spec->waves_p;
    if(tw < 1 || spec->tile_n < 1)
        FWW_FAIL("tile_w %d and tile_n %d must be >= 1", tw, spec->tile_n);
    if(block_p % (tw * spec->tile_n))
        FWW_FAIL("tile_w %d x tile_n %d must divide %d", tw, spec->tile_n, block_p);
    int TH = rocke_direct_fwd_wmma_tile_h(spec);
    if(spec->tile_n > 1 && (TH * tw) % 16)
        FWW_FAIL("an image tile must hold whole 16-pixel tiles");
    if(spec->tile_n > nw - 1)
        FWW_FAIL("tile_n needs one more wave than images (one X transfer per image)");
    if(spec->pingpong)
    {
        int issuers = spec->phase_split ? 2 * spec->tile_n + 1 : spec->tile_n + 1;
        if(spec->row_ring || spec->tap_ring || spec->split_barrier || spec->prefetch)
            FWW_FAIL("pingpong replaces the ring variants, split_barrier and prefetch");
        if(nw % 2 || issuers > nw / 2)
            FWW_FAIL("pingpong needs an even wave count with every issuer in rank 0");
        if(spec->wave_epilogue)
            FWW_FAIL("pingpong uses the block epilogue");
    }
    if(spec->pp_group < 1 || (spec->pp_group > 1 && !spec->pingpong))
        FWW_FAIL("pp_group %d: >= 1, and > 1 only with pingpong", spec->pp_group);
    if((spec->row_ring || spec->tap_ring) && (G > 1 || spec->split_barrier))
        FWW_FAIL("row_ring / tap_ring serve unpacked tiles without split_barrier");
    if(spec->row_ring && spec->tap_ring)
        FWW_FAIL("row_ring and tap_ring are exclusive");
    if(spec->tap_reuse)
    {
        if(spec->row_ring || spec->tap_ring)
            FWW_FAIL("tap_reuse needs all filter rows of a column in one weight slot");
        if(spec->pingpong)
            FWW_FAIL("tap_reuse is not combined with pingpong");
        if(p->stride != 1 || spec->tile_n != 1 || spec->phase_split)
            FWW_FAIL("tap_reuse serves stride 1, one image per block");
        if(tw % 16 || (spec->tiles_p * 16) % tw)
            FWW_FAIL("tap_reuse needs whole 16-pixel row pieces and whole rows per wave");
    }
    if(spec->tap_shift)
    {
        if(!spec->tap_reuse)
            FWW_FAIL("tap_shift builds on tap_reuse");
        if(p->dil_w > 15)
            FWW_FAIL("tap_shift rotates within a 16-lane row");
    }
    if(spec->phase_split)
    {
        if(p->stride != 2)
            FWW_FAIL("phase_split serves stride 2 only");
        if(spec->tile_n * 2 + 1 > nw)
            FWW_FAIL("phase_split needs 2 * tile_n + 1 waves (one transfer each)");
    }
    if(spec->wave_epilogue)
    {
        int wpx = spec->tiles_p * 16;
        if(wpx % tw || (TH * tw) % wpx)
            FWW_FAIL("wave_epilogue needs whole rows of one image per wave");
    }
    if(rocke_direct_fwd_wmma_threads_per_block(spec)
       > rocke_archtarget_max_threads_per_block(target))
        FWW_FAIL("%d threads per block", rocke_direct_fwd_wmma_threads_per_block(spec));
    if(nw < 2)
        FWW_FAIL("needs at least two waves (one TDM issuer per operand)");
    if(fww_x_pitch(spec) > 0xFFFF || fww_epi_pitch(spec) > 0xFFFF || fww_halo_w(spec) > 0xFFFF
       || fww_halo_h(spec) > 0xFFFF)
        FWW_FAIL("tile too wide for a TDM descriptor");
    if(!rocke_archtarget_fits_lds(target, rocke_direct_fwd_wmma_lds_bytes(spec)))
        FWW_FAIL(
            "LDS %ld B > %d", rocke_direct_fwd_wmma_lds_bytes(spec), target->lds_capacity_bytes);
    {
        int vgprs = 8 * spec->tiles_k * spec->tiles_p
                    + 8 * (spec->tiles_k + spec->tiles_p) * (spec->prefetch ? 2 : spec->pp_group);
        if(spec->tap_reuse)
            vgprs += 8 * ((p->KH - 1) * p->dil_h * (tw / 16));
        if(vgprs > FWW_VGPR_BUDGET)
            FWW_FAIL("~%d VGPRs of accumulators and fragments", vgprs);
    }
    if(spec->sched != NULL && !rocke_scheduler_strategy_is_valid(spec->sched))
        FWW_FAIL("unsupported scheduler_strategy '%s'", spec->sched);
    /* Global offsets are i32 bytes and the tile index is a flat block id. */
    {
        long long total_k = (long long)p->groups * p->kpg;
        long long total_c = (long long)p->groups * p->cpg;
        long long bytes[3] = {2LL * p->N * p->H * p->W * total_c,
                              2LL * total_k * p->KH * p->KW * p->cpg,
                              2LL * p->N * Ho * Wo * total_k};
        long long mx = bytes[0];
        if(bytes[1] > mx)
            mx = bytes[1];
        if(bytes[2] > mx)
            mx = bytes[2];
        if(mx > INT32_MAX)
            FWW_FAIL("tensors of %lld B exceed the i32 byte offsets", mx);
        long long tiles = (long long)fww_ceil_div(p->N, spec->tile_n) * fww_groups_eff(spec)
                          * fww_ceil_div((int)Ho, TH) * fww_ceil_div((int)Wo, spec->tile_w)
                          * fww_ceil_div(fww_kpg_eff(spec), rocke_direct_fwd_wmma_block_k(spec));
        if(tiles > INT32_MAX)
            FWW_FAIL("%lld workgroups exceed the i32 block id", tiles);
    }
    fww_set_reason(reason, reason_cap, "ok");
    return true;
}

#undef FWW_FAIL

/* ===================================================================== *
 *  Builder
 * ===================================================================== */
namespace
{

struct fww_step
{
    int ks, r, s;
};

struct fww_sub_halo
{
    V base, z_elems, w_ext;
};

struct fww_ctx
{
    rocke_ir_builder_t* b;
    const rocke_direct_fwd_wmma_spec_t* spec;
    rocke_dconv_params_t params;
    const rocke_type_t* io_type;
    const char* wmma_op_id;

    int KH, KW, S, DH, DW, PAD, G;
    int WAVE, WK, WP, TM, TN, THREADS, BK, TH, TW, HALO_H, HALO_W, CK, NS;
    int XP, EPITCH, HALO, WEI, NB, IMG, HALO_STAGE, TAPS;
    bool PS;
    int NPW, HC, PSTR, NHS, WTAPS, NG, WEI0;
    bool RR, TU, TSH, SB, PP;
    int TWF;
    int64_t SEL_LO, SEL_HI;

    V c0, c1, c2, c8, c16;
    V c_groups, c_cpg, c_kpg, c_cpg0, c_N, c_H, c_W, c_Ho, c_Wo;
    V xs_n, xs_h, xs_w, ys_n, ys_h, ys_w;
    V tid, wave_id, lane, l16, lhalf, wave_k, wave_p;
    V n_base, k_base, h_tile, w_tile, k_grp, c_grp, w_row, c_CK, n_chunks;
    V lds, halo_addr, wei_addr;
    V x_h_ext, x_tile2, xs_col, k_ext, w_base;
    V px_base, is_edge, edge_off;
    V rank, n_steps, k_out;
    std::vector<fww_sub_halo> sub_halos;
    std::vector<V> px_off;
    std::vector<V> wei_off;
    std::vector<fww_step> steps;
};

V ci(fww_ctx& c, int64_t v)
{
    return rocke_b_const_i32(c.b, v);
}
V add(fww_ctx& c, V x, V y)
{
    return rocke_b_add(c.b, x, y);
}
V sub(fww_ctx& c, V x, V y)
{
    return rocke_b_sub(c.b, x, y);
}
V mul(fww_ctx& c, V x, V y)
{
    return rocke_b_mul(c.b, x, y);
}
V divv(fww_ctx& c, V x, V y)
{
    return rocke_b_div(c.b, x, y);
}
V modv(fww_ctx& c, V x, V y)
{
    return rocke_b_mod(c.b, x, y);
}
V smaxv(fww_ctx& c, V x, V y)
{
    return rocke_b_smax(c.b, x, y);
}
V sminv(fww_ctx& c, V x, V y)
{
    return rocke_b_smin(c.b, x, y);
}
/* x OP const k: the const is emitted first (Python evaluates the argument). */
V addk(fww_ctx& c, V x, int64_t k)
{
    V t = ci(c, k);
    return add(c, x, t);
}
V mulk(fww_ctx& c, V x, int64_t k)
{
    V t = ci(c, k);
    return mul(c, x, t);
}
V divk(fww_ctx& c, V x, int64_t k)
{
    V t = ci(c, k);
    return divv(c, x, t);
}
V modk(fww_ctx& c, V x, int64_t k)
{
    V t = ci(c, k);
    return modv(c, x, t);
}

/* with b.scf_if(cond): body() */
void if_then(fww_ctx& c, V cond, const std::function<void()>& body)
{
    rocke_if_t gate = rocke_b_scf_if(c.b, cond);
    rocke_b_region_enter(c.b, gate.then_region);
    body();
    rocke_b_region_leave(c.b);
}

/* _tdm_descriptor_3d(b, global_ptr, lds_addr, dims, strides, tile0, tile1, tile2). */
void tdm_descriptor_3d(fww_ctx& c,
                       V global_ptr,
                       V lds_addr,
                       V d0,
                       V d1,
                       V d2,
                       V s1,
                       V s2,
                       int tile0,
                       int tile1,
                       V tile2,
                       V out[5])
{
    rocke_ir_builder_t* b = c.b;
    V c16 = ci(c, 16);
    V gaddr = rocke_b_global_ptr_to_i64(b, global_ptr);
    V c32 = rocke_b_const_i64(b, 32);
    V sh = rocke_b_lshr(b, gaddr, c32);
    V g_hi = rocke_b_trunc(b, sh, rocke_i32());
    V g0[4];
    V g1[8];
    V g2[4];
    g0[0] = ci(c, 1);
    g0[1] = lds_addr;
    g0[2] = rocke_b_trunc(b, gaddr, rocke_i32());
    {
        V m = ci(c, 0x1FFFFFF);
        V lo = rocke_b_land(b, g_hi, m);
        V ty = ci(c, -((int64_t)1 << 31));
        g0[3] = rocke_b_lor(b, lo, ty);
    }
    g1[0] = ci(c, 1 << 16);
    g1[1] = rocke_b_shl(b, d0, c16);
    {
        V x = rocke_b_lshr(b, d0, c16);
        V y = rocke_b_shl(b, d1, c16);
        g1[2] = rocke_b_lor(b, x, y);
    }
    {
        V x = rocke_b_lshr(b, d1, c16);
        V y = ci(c, (int64_t)tile0 << 16);
        g1[3] = rocke_b_lor(b, x, y);
    }
    {
        V x = ci(c, tile1);
        V y = rocke_b_shl(b, tile2, c16);
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
        V r[8];
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

/* _chunk_ext(chunk) -> (k0, c_ext). */
void chunk_ext(fww_ctx& c, V chunk, V* k0, V* c_ext)
{
    *k0 = mul(c, chunk, c.c_CK);
    V x = sub(c, c.c_cpg, *k0);
    V y = sminv(c, x, c.c_CK);
    *c_ext = smaxv(c, y, c.c0);
}

/* _issue_halo(chunk, stage). */
void issue_halo(fww_ctx& c, V chunk, V stage)
{
    rocke_ir_builder_t* b = c.b;
    const rocke_dconv_params_t* params = &c.params;
    V k0, c_ext;
    chunk_ext(c, chunk, &k0, &c_ext);
    V stage_off = mulk(c, stage, c.HALO_STAGE);
    for(int a = 0; a < (int)c.sub_halos.size(); ++a)
    {
        const fww_sub_halo& sh = c.sub_halos[a];
        for(int im = 0; im < c.NB; ++im)
        {
            int issuer = a == 0 ? im : c.NB + 1 + im;
            V x = ci(c, issuer);
            V e = rocke_b_cmp_eq(b, c.wave_id, x);
            V l = rocke_b_cmp_lt(b, c.c0, c.x_tile2);
            V issue = rocke_b_land(b, e, l);
            if_then(c, issue, [&]() {
                V img_off = addk(c, stage_off, (int64_t)im * c.IMG + (int64_t)a * c.PSTR);
                V img;
                {
                    V u = add(c, sh.base, k0);
                    V w = ci(c, im);
                    V z = mul(c, w, c.xs_n);
                    img = add(c, u, z);
                }
                V src;
                {
                    V bytes = mul(c, img, c.c2);
                    src = rocke_b_global_ptr_add(b, params->A, bytes);
                }
                V dst;
                {
                    V u = add(c, img_off, sh.z_elems);
                    V w = mul(c, u, c.c2);
                    dst = add(c, c.halo_addr, w);
                }
                V h_ext = c.x_h_ext;
                if(c.NB > 1)
                {
                    V u = addk(c, c.n_base, im);
                    V n_ok = rocke_b_cmp_lt(b, u, c.c_N);
                    h_ext = rocke_b_select(b, n_ok, c.x_h_ext, c.c0);
                    V base = rocke_b_global_ptr_add(b, params->A, c.c0);
                    src = rocke_b_select(b, n_ok, src, base);
                }
                V d[5];
                tdm_descriptor_3d(c,
                                  src,
                                  dst,
                                  c_ext,
                                  sh.w_ext,
                                  h_ext,
                                  c.xs_col,
                                  c.xs_h,
                                  c.XP,
                                  c.HC,
                                  c.x_tile2,
                                  d);
                rocke_b_tensor_load_to_lds(b, d[0], d[1], d[2], d[3], d[4], 0);
            });
        }
    }
}

/* _issue_wei(chunk, stage, grp=None). */
void issue_wei(fww_ctx& c, V chunk, V stage, V grp)
{
    rocke_ir_builder_t* b = c.b;
    const rocke_dconv_params_t* params = &c.params;
    V k0, c_ext;
    chunk_ext(c, chunk, &k0, &c_ext);
    V x = ci(c, c.NB);
    V cond = rocke_b_cmp_eq(b, c.wave_id, x);
    if_then(c, cond, [&]() {
        V off = add(c, c.w_base, k0);
        if(grp != NULL)
        {
            V t = mulk(c, c.c_cpg0, c.WTAPS);
            V u = mul(c, grp, t);
            off = add(c, off, u);
        }
        V src;
        {
            V bytes = mul(c, off, c.c2);
            src = rocke_b_global_ptr_add(b, params->Bp, bytes);
        }
        V dst;
        {
            V t = mulk(c, stage, (int64_t)c.WEI * 2);
            dst = add(c, c.wei_addr, t);
        }
        V taps0 = ci(c, c.WTAPS);
        V taps1 = ci(c, c.WTAPS);
        V d[5];
        tdm_descriptor_3d(
            c, src, dst, c_ext, c.k_ext, taps0, c.w_row, c.c_cpg0, c.XP, c.BK, taps1, d);
        rocke_b_tensor_load_to_lds(b, d[0], d[1], d[2], d[3], d[4], 0);
    });
}

/* _issue(chunk): TDM chunk `chunk` into ring slot chunk % NS. */
void issue(fww_ctx& c, V chunk)
{
    V stage = modk(c, chunk, c.NS);
    issue_halo(c, chunk, stage);
    if(c.G == 1)
        issue_wei(c, chunk, stage, NULL);
}

/* _read16(stage_base, off, const_off). */
V read16(fww_ctx& c, V stage_base, V off, int64_t const_off)
{
    rocke_ir_builder_t* b = c.b;
    V t = addk(c, off, const_off);
    V addr = add(c, stage_base, t);
    V idx0[2] = {c.c0, addr};
    V lo = rocke_b_smem_load_vN(b, c.lds, idx0, 2, c.io_type, 8);
    V addr8 = add(c, addr, c.c8);
    V idx1[2] = {c.c0, addr8};
    V hi = rocke_b_smem_load_vN(b, c.lds, idx1, 2, c.io_type, 8);
    return rocke_b_vec_concat(b, lo, hi);
}

int64_t tap_off0(fww_ctx& c, int ks, int s)
{
    return (int64_t)s * c.DW * c.XP + (int64_t)ks * FWW_WMMA_K;
}

/* _shift(frag, stage_base, off): the fragment one filter column to the right. */
V shift(fww_ctx& c, V frag, V stage_base, int64_t off)
{
    rocke_ir_builder_t* b = c.b;
    V words = rocke_b_vec_bitcast(b, frag, rocke_vector_type(b, rocke_i32(), 8));
    V lo_sel = ci(c, c.SEL_LO);
    V hi_sel = ci(c, c.SEL_HI);
    V moved[8];
    for(int i = 0; i < 8; ++i)
    {
        V w = rocke_b_vec_extract(b, words, i);
        moved[i] = rocke_b_permlane16(b, w, w, lo_sel, hi_sel, false, false);
    }
    V packed = rocke_b_vec_pack(b, moved, 8, rocke_i32());
    V out = rocke_b_vec_bitcast(b, packed, rocke_vector_type(b, c.io_type, 16));
    V a = add(c, c.px_base, c.edge_off);
    V edge = read16(c, stage_base, a, off);
    return rocke_b_select(b, c.is_edge, edge, out);
}

/* _wait_landed(kc). */
void wait_landed(fww_ctx& c, V kc)
{
    rocke_ir_builder_t* b = c.b;
    if(c.NS == 2)
    {
        rocke_b_s_wait_tensorcnt(b, 0);
        return;
    }
    V y = addk(c, kc, c.NS - 2);
    V more = rocke_b_cmp_lt(b, y, c.n_chunks);
    rocke_if_else_t ie = rocke_b_scf_if_else(b, more);
    rocke_b_region_enter(b, ie.then_region);
    rocke_b_s_wait_tensorcnt(b, c.NS - 2);
    rocke_b_region_leave(b);
    rocke_b_region_enter(b, ie.else_region);
    rocke_b_s_wait_tensorcnt(b, 0);
    rocke_b_region_leave(b);
}

/* _signal_next(kc). */
void signal_next(fww_ctx& c, V kc)
{
    rocke_b_s_waitcnt(c.b, -1, 0, -1);
    V n = add(c, kc, c.c1);
    wait_landed(c, n);
    rocke_b_s_barrier_signal(c.b, 0xFFFFFFFFu);
}

typedef std::tuple<int, int, int64_t> fww_key;

/* _frags(st) inside _multiply: TN B fragments, then TM A fragments. */
void frags(fww_ctx& c,
           std::map<fww_key, V>& cache,
           V halo_base,
           V wei_base,
           const fww_step& st,
           std::vector<V>& af,
           std::vector<V>& bf)
{
    int ks = st.ks, r = st.r, s = st.s;
    int64_t tap_off;
    if(c.PS)
    {
        int a = (s * c.DW) % c.S, j = (s * c.DW) / c.S;
        tap_off = (int64_t)a * c.PSTR + ((int64_t)r * c.DH * c.HC + j) * c.XP
                  + (int64_t)ks * FWW_WMMA_K;
    }
    else
    {
        tap_off
            = ((int64_t)r * c.DH * c.HALO_W + (int64_t)s * c.DW) * c.XP + (int64_t)ks * FWW_WMMA_K;
    }
    bf.assign(c.TN, NULL);
    af.assign(c.TM, NULL);
    if(c.TU)
    {
        for(int tn = 0; tn < c.TN; ++tn)
        {
            int64_t k2 = ((int64_t)(tn / c.TWF) + (int64_t)r * c.DH) * c.HALO_W * c.XP
                         + (int64_t)(tn % c.TWF) * 16 * c.XP;
            fww_key key(ks, s, k2);
            auto it = cache.find(key);
            if(it == cache.end())
            {
                V v;
                auto pit = cache.end();
                if(c.TSH && s > 0)
                    pit = cache.find(fww_key(ks, s - 1, k2));
                if(pit != cache.end())
                    v = shift(c, pit->second, halo_base, k2 + tap_off0(c, ks, s));
                else
                    v = read16(c, halo_base, c.px_base, k2 + tap_off0(c, ks, s));
                cache[key] = v;
                bf[tn] = v;
            }
            else
            {
                bf[tn] = it->second;
            }
        }
    }
    else
    {
        for(int tn = 0; tn < c.TN; ++tn)
            bf[tn] = read16(c, halo_base, c.px_off[tn], tap_off);
    }
    int tap = (r * c.KW + s) % c.WTAPS;
    int64_t w_off = (int64_t)tap * c.BK * c.XP + (int64_t)ks * FWW_WMMA_K;
    for(int m = 0; m < c.TM; ++m)
        af[m] = read16(c, wei_base, c.wei_off[m], w_off);
}

/* _multiply(st_list, halo_base, wei_base, accs, signal). */
void multiply(fww_ctx& c,
              std::vector<fww_step> st_list,
              V halo_base,
              V wei_base,
              std::vector<V>& accs,
              V signal_kc)
{
    rocke_ir_builder_t* b = c.b;
    std::map<fww_key, V> cache;
    if(c.TU)
    {
        std::stable_sort(st_list.begin(), st_list.end(), [](const fww_step& x, const fww_step& y) {
            return std::make_tuple(x.ks, x.s, x.r) < std::make_tuple(y.ks, y.s, y.r);
        });
    }
    rocke_b_s_setprio(b, 1);
    std::vector<V> af, bf, naf, nbf;
    bool pf = c.spec->prefetch;
    if(pf)
        frags(c, cache, halo_base, wei_base, st_list[0], naf, nbf);
    for(size_t j = 0; j < st_list.size(); ++j)
    {
        if(pf)
        {
            af = naf;
            bf = nbf;
            if(j + 1 < st_list.size())
                frags(c, cache, halo_base, wei_base, st_list[j + 1], naf, nbf);
        }
        else
        {
            frags(c, cache, halo_base, wei_base, st_list[j], af, bf);
        }
        if(signal_kc != NULL && j == st_list.size() - 1)
            signal_next(c, signal_kc);
        for(int m = 0; m < c.TM; ++m)
            for(int tn = 0; tn < c.TN; ++tn)
            {
                int i = m * c.TN + tn;
                accs[i] = rocke_b_mma(b, c.wmma_op_id, af[m], bf[tn], accs[i], NULL, 0);
            }
    }
    rocke_b_s_setprio(b, 0);
}

/* _chunk(kc, accs): publish chunk kc, refill the ring, multiply the chunk. */
void chunk(fww_ctx& c, V kc, std::vector<V>& accs)
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
    V nxt = addk(c, kc, c.NS - 1);
    V cond = rocke_b_cmp_lt(b, nxt, c.n_chunks);
    if_then(c, cond, [&]() { issue(c, nxt); });
    V stage = modk(c, kc, c.NS);
    V halo_base = mulk(c, stage, c.HALO_STAGE);
    V wei_base = c.G == 1 ? mulk(c, stage, c.WEI) : c.c0;
    multiply(c, c.steps, halo_base, wei_base, accs, c.SB ? kc : NULL);
}

void pp_bar(fww_ctx& c)
{
    rocke_b_s_barrier_bare_fence(c.b, "compiler");
    rocke_b_sched_barrier(c.b, 0);
}

/* _chunk_pp(kc, accs). */
void chunk_pp(fww_ctx& c, V kc, std::vector<V>& accs)
{
    rocke_ir_builder_t* b = c.b;
    V nxt = addk(c, kc, c.NS - 1);
    V cond = rocke_b_cmp_lt(b, nxt, c.n_chunks);
    if_then(c, cond, [&]() { issue(c, nxt); });
    V stage = modk(c, kc, c.NS);
    V halo_base = mulk(c, stage, c.HALO_STAGE);
    V wei_base = c.G == 1 ? mulk(c, stage, c.WEI) : c.c0;
    int PG = c.spec->pp_group;
    int n_steps = (int)c.steps.size();
    int n_groups = fww_ceil_div(n_steps, PG);
    for(int j = 0; j < n_groups; ++j)
    {
        std::vector<std::vector<V>> fa, fb;
        for(int i = j * PG; i < std::min(n_steps, (j + 1) * PG); ++i)
        {
            const fww_step& st = c.steps[i];
            int64_t tap_off;
            if(c.PS)
            {
                int a = (st.s * c.DW) % c.S, jj = (st.s * c.DW) / c.S;
                tap_off = (int64_t)a * c.PSTR + ((int64_t)st.r * c.DH * c.HC + jj) * c.XP
                          + (int64_t)st.ks * FWW_WMMA_K;
            }
            else
            {
                tap_off = ((int64_t)st.r * c.DH * c.HALO_W + (int64_t)st.s * c.DW) * c.XP
                          + (int64_t)st.ks * FWW_WMMA_K;
            }
            std::vector<V> bf(c.TN), af(c.TM);
            for(int tn = 0; tn < c.TN; ++tn)
                bf[tn] = read16(c, halo_base, c.px_off[tn], tap_off);
            int tap = (st.r * c.KW + st.s) % c.WTAPS;
            int64_t w_off = (int64_t)tap * c.BK * c.XP + (int64_t)st.ks * FWW_WMMA_K;
            for(int m = 0; m < c.TM; ++m)
                af[m] = read16(c, wei_base, c.wei_off[m], w_off);
            fa.push_back(af);
            fb.push_back(bf);
        }
        bool last = j == n_groups - 1;
        if(last)
            rocke_b_s_waitcnt(b, -1, 0, -1);
        pp_bar(c);
        rocke_b_s_setprio(b, 1);
        for(size_t f = 0; f < fa.size(); ++f)
            for(int m = 0; m < c.TM; ++m)
                for(int tn = 0; tn < c.TN; ++tn)
                {
                    int i = m * c.TN + tn;
                    accs[i] = rocke_b_mma(b, c.wmma_op_id, fa[f][m], fb[f][tn], accs[i], NULL, 0);
                }
        rocke_b_s_setprio(b, 0);
        if(last)
        {
            V n = add(c, kc, c.c1);
            wait_landed(c, n);
        }
        pp_bar(c);
    }
}

/* _chunk_rr(kc, accs): row_ring / tap_ring. */
void chunk_rr(fww_ctx& c, V kc, std::vector<V>& accs)
{
    rocke_ir_builder_t* b = c.b;
    V hslot = modv(c, kc, c.c2);
    V halo_base = mulk(c, hslot, c.HALO_STAGE);
    for(int r = 0; r < c.NG; ++r)
    {
        V u;
        {
            V x = mulk(c, kc, c.NG);
            u = addk(c, x, r);
        }
        {
            V x = ci(c, c.NB);
            V cond = rocke_b_cmp_eq(b, c.wave_id, x);
            if_then(c, cond, [&]() {
                if(c.NS == 2)
                {
                    rocke_b_s_wait_tensorcnt(b, 0);
                }
                else
                {
                    V y = addk(c, u, c.NS - 2);
                    V more = rocke_b_cmp_lt(b, y, c.n_steps);
                    rocke_if_else_t ie = rocke_b_scf_if_else(b, more);
                    rocke_b_region_enter(b, ie.then_region);
                    rocke_b_s_wait_tensorcnt(b, c.NS - 2);
                    rocke_b_region_leave(b);
                    rocke_b_region_enter(b, ie.else_region);
                    rocke_b_s_wait_tensorcnt(b, 0);
                    rocke_b_region_leave(b);
                }
            });
        }
        if(r == 0)
        {
            V x = ci(c, c.NB);
            V cond = rocke_b_cmp_ne(b, c.wave_id, x);
            if_then(c, cond, [&]() { rocke_b_s_wait_tensorcnt(b, 0); });
        }
        rocke_b_sync_lds_only(b);
        {
            V nxt = addk(c, u, c.NS - 1);
            V cond = rocke_b_cmp_lt(b, nxt, c.n_steps);
            if_then(c, cond, [&]() {
                V nc = divk(c, nxt, c.NG);
                V slot = modk(c, nxt, c.NS);
                V grp = modk(c, nxt, c.NG);
                issue_wei(c, nc, slot, grp);
            });
        }
        if(r == 0)
        {
            V nc = add(c, kc, c.c1);
            V cond = rocke_b_cmp_lt(b, nc, c.n_chunks);
            if_then(c, cond, [&]() {
                V slot = modv(c, nc, c.c2);
                issue_halo(c, nc, slot);
            });
        }
        V wslot = modk(c, u, c.NS);
        std::vector<fww_step> row_steps;
        for(int ks = 0; ks < c.CK / FWW_WMMA_K; ++ks)
            for(int t = r * c.WTAPS; t < (r + 1) * c.WTAPS; ++t)
                row_steps.push_back(fww_step{ks, t / c.KW, t % c.KW});
        V wb = mulk(c, wslot, c.WEI);
        multiply(c, row_steps, halo_base, wb, accs, NULL);
    }
}

/* _epilogue(accs): stage the Y tile in LDS, one TDM store per image. */
void epilogue(fww_ctx& c, V const* accs)
{
    rocke_ir_builder_t* b = c.b;
    const rocke_dconv_params_t* params = &c.params;
    rocke_b_sync_lds_only(b);
    for(int m = 0; m < c.TM; ++m)
    {
        V col;
        {
            V y = mulk(c, c.wave_k, c.TM);
            V u = addk(c, y, m);
            V v = mul(c, u, c.c16);
            V w = mul(c, c.lhalf, c.c8);
            col = add(c, v, w);
        }
        for(int tn = 0; tn < c.TN; ++tn)
        {
            V px;
            {
                V y = mulk(c, c.wave_p, c.TN);
                V u = addk(c, y, tn);
                V v = mul(c, u, c.c16);
                px = add(c, v, c.l16);
            }
            V val = rocke_b_vec_cast_f32_to(b, accs[m * c.TN + tn], c.io_type);
            V y = mulk(c, px, c.EPITCH);
            V z = add(c, y, col);
            V idx[2] = {c.c0, z};
            rocke_b_smem_store_vN(b, c.lds, idx, 2, val, 8);
        }
    }
    rocke_b_sync_lds_only(b);
    for(int im = 0; im < c.NB; ++im)
    {
        V x0 = ci(c, im);
        V cond = rocke_b_cmp_eq(b, c.wave_id, x0);
        if_then(c, cond, [&]() {
            V n_img = addk(c, c.n_base, im);
            V off;
            {
                V x = mul(c, n_img, c.ys_n);
                V y = mul(c, c.h_tile, c.ys_h);
                V z = add(c, x, y);
                V u = mul(c, c.w_tile, c.ys_w);
                V w = add(c, u, c.k_out);
                off = add(c, z, w);
            }
            V h_ext;
            {
                V x = sub(c, c.c_Ho, c.h_tile);
                h_ext = smaxv(c, x, c.c0);
            }
            V dst;
            {
                V x = mul(c, off, c.c2);
                dst = rocke_b_global_ptr_add(b, params->D, x);
            }
            if(c.NB > 1)
            {
                V n_ok = rocke_b_cmp_lt(b, n_img, c.c_N);
                h_ext = rocke_b_select(b, n_ok, h_ext, c.c0);
                V base = rocke_b_global_ptr_add(b, params->D, c.c0);
                dst = rocke_b_select(b, n_ok, dst, base);
            }
            V lds_src = addk(c, c.halo_addr, (int64_t)im * c.TH * c.TW * c.EPITCH * 2);
            V w_ext;
            {
                V x = sub(c, c.c_Wo, c.w_tile);
                w_ext = smaxv(c, x, c.c0);
            }
            V th = ci(c, c.TH);
            V d[5];
            tdm_descriptor_3d(
                c, dst, lds_src, c.k_ext, w_ext, h_ext, c.ys_w, c.ys_h, c.EPITCH, c.TW, th, d);
            rocke_b_tensor_store_from_lds(b, d[0], d[1], d[2], d[3], d[4], 0);
            rocke_b_s_wait_tensorcnt(b, 0);
        });
    }
}

/* _epilogue_wave(accs): per-wave staging and TDM store. */
void epilogue_wave(fww_ctx& c, V const* accs)
{
    rocke_ir_builder_t* b = c.b;
    const rocke_dconv_params_t* params = &c.params;
    if(!c.SB)
        rocke_b_sync_lds_only(b);
    int WPX = c.TN * 16;
    int EP = c.TM * 16 + FWW_CH_PAD;
    int ROWS = WPX / c.TW;
    V base = mulk(c, c.wave_id, (int64_t)WPX * EP);
    for(int m = 0; m < c.TM; ++m)
    {
        V col;
        {
            V x = ci(c, m * 16);
            V y = mul(c, c.lhalf, c.c8);
            col = add(c, x, y);
        }
        for(int tn = 0; tn < c.TN; ++tn)
        {
            V px;
            {
                V x = ci(c, tn * 16);
                px = add(c, x, c.l16);
            }
            V val = rocke_b_vec_cast_f32_to(b, accs[m * c.TN + tn], c.io_type);
            V y = mulk(c, px, EP);
            V z = add(c, y, col);
            V w = add(c, base, z);
            V idx[2] = {c.c0, w};
            rocke_b_smem_store_vN(b, c.lds, idx, 2, val, 8);
        }
    }
    rocke_b_s_waitcnt(b, -1, 0, -1);
    V px0 = mulk(c, c.wave_p, WPX);
    V img = divk(c, px0, c.TH * c.TW);
    V row0;
    {
        V x = modk(c, px0, c.TH * c.TW);
        row0 = divk(c, x, c.TW);
    }
    V n_img = add(c, c.n_base, img);
    V h_w = add(c, c.h_tile, row0);
    V k_w;
    {
        V x = mulk(c, c.wave_k, c.TM * 16);
        k_w = add(c, c.k_base, x);
    }
    V off;
    {
        V x = mul(c, n_img, c.ys_n);
        V y = mul(c, h_w, c.ys_h);
        V z = add(c, x, y);
        V u = mul(c, c.w_tile, c.ys_w);
        V v = add(c, c.k_grp, k_w);
        V w = add(c, u, v);
        off = add(c, z, w);
    }
    V k_ext_w;
    {
        V x = sub(c, c.c_kpg, k_w);
        V y = ci(c, c.TM * 16);
        V z = sminv(c, x, y);
        k_ext_w = smaxv(c, z, c.c0);
    }
    V h_ext;
    {
        V x = sub(c, c.c_Ho, h_w);
        h_ext = smaxv(c, x, c.c0);
    }
    V dst;
    {
        V x = mul(c, off, c.c2);
        dst = rocke_b_global_ptr_add(b, params->D, x);
    }
    if(c.NB > 1)
    {
        V n_ok = rocke_b_cmp_lt(b, n_img, c.c_N);
        h_ext = rocke_b_select(b, n_ok, h_ext, c.c0);
        V g = rocke_b_global_ptr_add(b, params->D, c.c0);
        dst = rocke_b_select(b, n_ok, dst, g);
    }
    V lds_src;
    {
        V x = mul(c, base, c.c2);
        lds_src = add(c, c.halo_addr, x);
    }
    V w_ext;
    {
        V x = sub(c, c.c_Wo, c.w_tile);
        w_ext = smaxv(c, x, c.c0);
    }
    V rows = ci(c, ROWS);
    V d[5];
    tdm_descriptor_3d(c, dst, lds_src, k_ext_w, w_ext, h_ext, c.ys_w, c.ys_h, EP, c.TW, rows, d);
    rocke_b_tensor_store_from_lds(b, d[0], d[1], d[2], d[3], d[4], 0);
    rocke_b_s_wait_tensorcnt(b, 0);
}

/* pack_groups > 1: stage the super-group's block-diagonal weights once. */
void stage_packed_weights(fww_ctx& c)
{
    rocke_ir_builder_t* b = c.b;
    const rocke_dconv_params_t* params = &c.params;
    const rocke_direct_conv_problem_t* p = &c.spec->problem;
    int CPG0 = p->cpg, KPG0 = p->kpg;
    V zero_v = rocke_b_zero_vec(b, c.io_type, 8);
    int nvec = c.WEI / 8;
    for(int j = 0; j < nvec; j += c.THREADS)
    {
        V v = addk(c, c.tid, j);
        V lim = ci(c, nvec);
        V cond = rocke_b_cmp_lt(b, v, lim);
        if_then(c, cond, [&]() {
            V x = mulk(c, v, 8);
            V y = addk(c, x, c.WEI0);
            V idx[2] = {c.c0, y};
            rocke_b_smem_store_vN(b, c.lds, idx, 2, zero_v, 8);
        });
    }
    rocke_b_sync_lds_only(b);
    int VEC = std::min(CPG0, 8);
    int PARTS = CPG0 / VEC;
    int KQ = c.G * KPG0;
    int total = c.TAPS * KQ * PARTS;
    for(int j = 0; j < total; j += c.THREADS)
    {
        V v = addk(c, c.tid, j);
        V lim = ci(c, total);
        V cond = rocke_b_cmp_lt(b, v, lim);
        if_then(c, cond, [&]() {
            V part = modk(c, v, PARTS);
            V rw = divk(c, v, PARTS);
            V tap = divk(c, rw, KQ);
            V kq = modk(c, rw, KQ);
            V gi = divk(c, kq, KPG0);
            V pv = mulk(c, part, VEC);
            V gidx;
            {
                V x = add(c, c.k_grp, kq);
                V y = mulk(c, x, (int64_t)c.TAPS * CPG0);
                V z = mulk(c, tap, CPG0);
                V u = add(c, z, pv);
                gidx = add(c, y, u);
            }
            V val = rocke_b_global_load_vN(b, params->Bp, gidx, c.io_type, VEC, 0);
            V loff;
            {
                V x = mulk(c, tap, c.BK);
                V y = add(c, x, kq);
                V z = mulk(c, y, c.XP);
                V u = mulk(c, gi, CPG0);
                V w = add(c, u, pv);
                loff = add(c, z, w);
            }
            V a = addk(c, loff, c.WEI0);
            V idx[2] = {c.c0, a};
            rocke_b_smem_store_vN(b, c.lds, idx, 2, val, VEC);
        });
    }
}

} // namespace

rocke_kernel_def_t* rocke_build_direct_fwd_wmma(rocke_ir_builder_t* b,
                                                const rocke_direct_fwd_wmma_spec_t* spec,
                                                const char* arch)
{
    char reason[512];
    fww_ctx c;

    if(b == NULL || spec == NULL)
        return NULL;
    if(arch == NULL)
        arch = "gfx1250";
    if(!rocke_direct_fwd_wmma_is_valid_spec(spec, arch, reason, sizeof reason))
    {
        std::string msg = std::string("invalid DirectFwdWmmaSpec for ") + arch + ": " + reason;
        rocke_i_set_err_msg(b, ROCKE_ERR_VALUE, msg.c_str());
        return NULL;
    }
    const rocke_direct_conv_problem_t* p = &spec->problem;
    c.b = b;
    c.spec = spec;
    c.KH = p->KH;
    c.KW = p->KW;
    c.S = p->stride;
    c.DH = p->dil_h;
    c.DW = p->dil_w;
    c.PAD = p->PAD;
    c.G = spec->pack_groups;
    c.io_type = rocke_b_io_ir_type(b, fww_dtype(spec));
    if(c.io_type == NULL)
        return NULL;
    {
        const char* ab = rocke_wmma_ab_dtype(fww_dtype(spec));
        const rocke_archtarget_t* target = rocke_archtarget_from_gfx(arch);
        const rocke_mmaop_t* op = rocke_archtarget_op_for_shape(
            target, "wmma", ab, ab, "fp32", ROCKE_FWD_WMMA_M, ROCKE_FWD_WMMA_N, FWW_WMMA_K, NULL);
        if(op == NULL || op->op_id == NULL)
        {
            rocke_i_set_err_msg(b, ROCKE_ERR_VALUE, "build_direct_fwd_wmma: no WMMA atom");
            return NULL;
        }
        c.wmma_op_id = op->op_id;
    }
    c.WAVE = spec->wave_size;
    c.WK = spec->waves_k;
    c.WP = spec->waves_p;
    c.TM = spec->tiles_k;
    c.TN = spec->tiles_p;
    c.THREADS = rocke_direct_fwd_wmma_threads_per_block(spec);
    c.BK = rocke_direct_fwd_wmma_block_k(spec);
    c.TH = rocke_direct_fwd_wmma_tile_h(spec);
    c.TW = spec->tile_w;
    c.HALO_H = fww_halo_h(spec);
    c.HALO_W = fww_halo_w(spec);
    c.CK = spec->ck;
    c.NS = spec->stages;
    c.XP = fww_x_pitch(spec);
    c.EPITCH = fww_epi_pitch(spec);
    c.HALO = fww_halo_elems(spec);
    c.WEI = fww_wei_elems(spec);
    c.NB = spec->tile_n;
    c.IMG = fww_image_elems(spec);
    c.HALO_STAGE = c.HALO + fww_halo_slack(spec);
    c.TAPS = c.KH * c.KW;
    c.PS = spec->phase_split;
    c.NPW = fww_phases_w(spec);
    c.HC = fww_halo_cols(spec);
    c.PSTR = c.PS ? fww_phase_elems(spec) : 0;
    c.RR = spec->row_ring || spec->tap_ring;
    c.NHS = c.RR ? 2 : c.NS;
    c.WTAPS = fww_ring_taps(spec);
    c.NG = c.TAPS / c.WTAPS;
    c.WEI0 = c.NHS * c.HALO_STAGE;
    c.TU = spec->tap_reuse;
    c.TSH = spec->tap_shift;
    c.TWF = c.TW / 16;
    c.SB = spec->split_barrier;
    c.PP = spec->pingpong;
    c.px_base = c.is_edge = c.edge_off = c.rank = NULL;

    rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", c.THREADS);
    if(spec->sched != NULL && spec->sched[0] != '\0')
        rocke_attr_set_str(b, &b->kernel->attrs, "scheduler_strategy", spec->sched);
    rocke_dconv_emit_params(b, &c.params, "fwd", c.io_type);
    const rocke_dconv_params_t* params = &c.params;

    c.c0 = ci(c, 0);
    c.c1 = ci(c, 1);
    c.c2 = ci(c, 2);
    if(c.G > 1)
    {
        c.c_groups = divk(c, params->p_groups, c.G);
        c.c_cpg = ci(c, fww_cpg_eff(spec));
        c.c_kpg = ci(c, fww_kpg_eff(spec));
        c.c_cpg0 = ci(c, p->cpg);
    }
    else
    {
        c.c_groups = params->p_groups;
        c.c_cpg = divv(c, params->p_total_c, c.c_groups);
        c.c_kpg = divv(c, params->p_total_k, c.c_groups);
        c.c_cpg0 = c.c_cpg;
    }
    c.c_N = params->p_N;
    c.c_H = params->p_Hi;
    c.c_W = params->p_Wi;
    c.c_Ho = params->p_Ho;
    c.c_Wo = params->p_Wo;
    c.xs_n = params->p_A_stride_n;
    c.xs_h = params->p_A_stride_hi;
    c.xs_w = params->p_A_stride_wi;
    c.ys_n = params->p_D_stride_n;
    c.ys_h = params->p_D_stride_ho;
    c.ys_w = params->p_D_stride_wo;

    c.tid = rocke_b_thread_id_x(b);
    {
        V c_wave = ci(c, c.WAVE);
        c.wave_id = divv(c, c.tid, c_wave);
        c.lane = modv(c, c.tid, c_wave);
    }
    c.c16 = ci(c, 16);
    c.c8 = ci(c, 8);
    c.l16 = modv(c, c.lane, c.c16);
    c.lhalf = divv(c, c.lane, c.c16);
    c.wave_k = modk(c, c.wave_id, c.WK);
    c.wave_p = divk(c, c.wave_id, c.WK);

    /* ---- tile of this block. */
    V n_kt, n_ht, n_wt;
    {
        V y = addk(c, c.c_kpg, c.BK - 1);
        n_kt = divk(c, y, c.BK);
    }
    {
        V y = addk(c, c.c_Ho, c.TH - 1);
        n_ht = divk(c, y, c.TH);
    }
    {
        V y = addk(c, c.c_Wo, c.TW - 1);
        n_wt = divk(c, y, c.TW);
    }
    V t = rocke_b_block_id_x(b);
    V kt = modv(c, t, n_kt);
    t = divv(c, t, n_kt);
    V wt = modv(c, t, n_wt);
    t = divv(c, t, n_wt);
    V ht = modv(c, t, n_ht);
    t = divv(c, t, n_ht);
    V g = modv(c, t, c.c_groups);
    {
        V x = divv(c, t, c.c_groups);
        c.n_base = mulk(c, x, c.NB);
    }
    c.k_base = mulk(c, kt, c.BK);
    c.h_tile = mulk(c, ht, c.TH);
    c.w_tile = mulk(c, wt, c.TW);
    c.k_grp = mul(c, g, c.c_kpg);
    c.c_grp = mul(c, g, c.c_cpg);
    c.w_row = mulk(c, c.c_cpg0, c.TAPS);
    c.c_CK = ci(c, c.CK);
    {
        V y = addk(c, c.c_cpg, c.CK - 1);
        c.n_chunks = divv(c, y, c.c_CK);
    }

    /* ---- one LDS block: the halo ring, then the weight ring. */
    {
        int shape[2] = {1, (int)(rocke_direct_fwd_wmma_lds_bytes(spec) / 2)};
        c.lds = rocke_b_smem_alloc(b, c.io_type, shape, 2, "fw_lds");
    }
    {
        V a = rocke_b_smem_addr_of(b, c.lds);
        c.halo_addr = rocke_b_trunc(b, a, rocke_i32());
        c.wei_addr = addk(c, c.halo_addr, (int64_t)c.WEI0 * 2);
    }

    /* ---- halo origin and the edge shifts. */
    V h0, w0, hs, h_start;
    {
        V x = c.S == 1 ? c.h_tile : mulk(c, c.h_tile, c.S);
        V y = ci(c, c.PAD);
        h0 = sub(c, x, y);
    }
    {
        V x = c.S == 1 ? c.w_tile : mulk(c, c.w_tile, c.S);
        V y = ci(c, c.PAD);
        w0 = sub(c, x, y);
    }
    {
        V x = sub(c, c.c0, h0);
        V y = smaxv(c, x, c.c0);
        V z = ci(c, c.HALO_H);
        hs = sminv(c, y, z);
    }
    {
        V x = add(c, h0, hs);
        h_start = smaxv(c, x, c.c0);
    }
    {
        V x = sub(c, c.c_H, h_start);
        c.x_h_ext = smaxv(c, x, c.c0);
    }
    {
        V x = ci(c, c.HALO_H);
        c.x_tile2 = sub(c, x, hs);
    }
    V c_S = ci(c, c.S);
    for(int a = 0; a < c.NPW; ++a)
    {
        V wa = a ? addk(c, w0, a) : w0;
        V sw, w_start, w_ext;
        if(c.PS)
        {
            {
                V x = ci(c, c.S - 1);
                V y = sub(c, x, wa);
                V z = divv(c, y, c_S);
                sw = smaxv(c, z, c.c0);
            }
            {
                V x = ci(c, c.HC);
                sw = sminv(c, sw, x);
            }
            {
                V x = mul(c, sw, c_S);
                V y = add(c, wa, x);
                w_start = smaxv(c, y, c.c0);
            }
            {
                V x = sub(c, c.c_W, w_start);
                V y = addk(c, x, c.S - 1);
                w_ext = divv(c, y, c_S);
            }
        }
        else
        {
            {
                V x = sub(c, c.c0, wa);
                V y = smaxv(c, x, c.c0);
                V z = ci(c, c.HC);
                sw = sminv(c, y, z);
            }
            {
                V x = add(c, wa, sw);
                w_start = smaxv(c, x, c.c0);
            }
            w_ext = sub(c, c.c_W, w_start);
        }
        {
            V x = ci(c, c.HC);
            V y = sub(c, x, sw);
            V z = sminv(c, w_ext, y);
            w_ext = smaxv(c, z, c.c0);
        }
        V z_elems;
        {
            V x = mulk(c, hs, c.HC);
            V y = add(c, x, sw);
            z_elems = mulk(c, y, c.XP);
        }
        V base;
        {
            V x = mul(c, c.n_base, c.xs_n);
            V y = mul(c, h_start, c.xs_h);
            V z = add(c, x, y);
            V u = mul(c, w_start, c.xs_w);
            V w = add(c, u, c.c_grp);
            base = add(c, z, w);
        }
        c.sub_halos.push_back(fww_sub_halo{base, z_elems, w_ext});
    }
    c.xs_col = c.PS ? mul(c, c.xs_w, c_S) : c.xs_w;
    {
        V x = sub(c, c.c_kpg, c.k_base);
        V y = ci(c, c.BK);
        V z = sminv(c, x, y);
        c.k_ext = smaxv(c, z, c.c0);
    }
    {
        V x = add(c, c.k_grp, c.k_base);
        c.w_base = mul(c, x, c.w_row);
    }

    /* ---- zero the halo cells no transfer writes (top / left image edge). */
    {
        int lead = fww_lead_cols(spec);
        int max_z = (std::min(c.PAD, c.HALO_H) * c.HC + std::min(lead, c.HC)) * c.XP;
        if(max_z)
        {
            V zero_v = rocke_b_zero_vec(b, c.io_type, 8);
            for(int a = 0; a < (int)c.sub_halos.size(); ++a)
            {
                V z_vecs = divk(c, c.sub_halos[a].z_elems, 8);
                for(int j = 0; j < fww_ceil_div(max_z, 8); j += c.THREADS)
                {
                    V v = addk(c, c.tid, j);
                    V cond = rocke_b_cmp_lt(b, v, z_vecs);
                    if_then(c, cond, [&]() {
                        V flat = mulk(c, v, 8);
                        for(int st = 0; st < c.NHS; ++st)
                            for(int im = 0; im < c.NB; ++im)
                            {
                                int64_t off = (int64_t)st * c.HALO_STAGE + (int64_t)im * c.IMG
                                              + (int64_t)a * c.PSTR;
                                V x = addk(c, flat, off);
                                V idx[2] = {c.c0, x};
                                rocke_b_smem_store_vN(b, c.lds, idx, 2, zero_v, 8);
                            }
                    });
                }
            }
        }
    }

    /* ---- per-lane LDS read offsets. */
    c.px_off.assign(c.TN, NULL);
    for(int tn = 0; tn < c.TN; ++tn)
    {
        V flat, nb;
        {
            V y = mulk(c, c.wave_p, c.TN);
            V u = addk(c, y, tn);
            V v = mul(c, u, c.c16);
            flat = add(c, v, c.l16);
        }
        if(c.NB > 1)
        {
            nb = divk(c, flat, c.TH * c.TW);
            flat = modk(c, flat, c.TH * c.TW);
        }
        else
        {
            nb = c.c0;
        }
        V th = divk(c, flat, c.TW);
        V tw = modk(c, flat, c.TW);
        if(c.S > 1)
        {
            th = mulk(c, th, c.S);
            if(!c.PS)
                tw = mulk(c, tw, c.S);
        }
        V off;
        {
            V y = mulk(c, th, c.HC);
            V z = add(c, y, tw);
            V v = mulk(c, z, c.XP);
            V w = mul(c, c.lhalf, c.c16);
            off = add(c, v, w);
        }
        if(c.NB > 1)
        {
            V y = mulk(c, nb, c.IMG);
            off = add(c, off, y);
        }
        c.px_off[tn] = off;
    }
    c.wei_off.assign(c.TM, NULL);
    for(int m = 0; m < c.TM; ++m)
    {
        V row;
        {
            V y = mulk(c, c.wave_k, c.TM);
            V u = addk(c, y, m);
            V v = mul(c, u, c.c16);
            row = add(c, v, c.l16);
        }
        V x = mulk(c, row, c.XP);
        V y = mul(c, c.lhalf, c.c16);
        V z = addk(c, y, c.WEI0);
        c.wei_off[m] = add(c, x, z);
    }

    if(c.TU)
    {
        c.px_base = c.px_off[0];
        if(c.TSH)
        {
            V c_edge = ci(c, 16 - c.DW);
            c.is_edge = rocke_b_cmp_ge(b, c.l16, c_edge);
            V x = smaxv(c, c.l16, c_edge);
            V y = sub(c, x, c.l16);
            c.edge_off = mulk(c, y, c.XP);
        }
    }
    {
        /* v_permlane16 selectors: lane l <- lane l + DW inside each 16-lane row. */
        int src[16];
        for(int i = 0; i < 16; ++i)
            src[i] = std::min(i + c.DW, 15);
        uint64_t lo = 0, hi = 0;
        for(int i = 0; i < 8; ++i)
        {
            lo += (uint64_t)src[i] << (4 * i);
            hi += (uint64_t)src[8 + i] << (4 * i);
        }
        c.SEL_LO = (lo & 0x80000000ull) ? (int64_t)lo - ((int64_t)1 << 32) : (int64_t)lo;
        c.SEL_HI = (hi & 0x80000000ull) ? (int64_t)hi - ((int64_t)1 << 32) : (int64_t)hi;
    }

    for(int ks = 0; ks < c.CK / FWW_WMMA_K; ++ks)
        for(int r = 0; r < c.KH; ++r)
            for(int s = 0; s < c.KW; ++s)
                c.steps.push_back(fww_step{ks, r, s});

    if(c.PP)
        c.rank = divk(c, c.wave_id, (c.WK * c.WP) / 2);
    c.n_steps = mulk(c, c.n_chunks, c.NG);
    c.k_out = add(c, c.k_grp, c.k_base);

    if(c.G > 1)
        stage_packed_weights(c);

    /* ---- main loop over input channel chunks. */
    if(c.RR)
    {
        issue_halo(c, c.c0, c.c0);
        for(int st = 0; st < c.NS - 1; ++st)
        {
            V x = ci(c, st);
            V cond = rocke_b_cmp_lt(b, x, c.n_steps);
            if_then(c, cond, [&]() {
                V ch = ci(c, st / c.NG);
                V sl = ci(c, st);
                V gr = ci(c, st % c.NG);
                issue_wei(c, ch, sl, gr);
            });
        }
    }
    else
    {
        for(int st = 0; st < c.NS - 1; ++st)
        {
            V x = ci(c, st);
            V cond = rocke_b_cmp_lt(b, x, c.n_chunks);
            if_then(c, cond, [&]() { issue(c, ci(c, st)); });
        }
    }
    if(c.SB)
    {
        rocke_b_s_waitcnt(b, -1, 0, -1);
        wait_landed(c, c.c0);
        rocke_b_s_barrier_signal(b, 0xFFFFFFFFu);
    }
    if(c.PP)
    {
        wait_landed(c, c.c0);
        rocke_b_sync_lds_only(b);
        V cond = rocke_b_cmp_eq(b, c.rank, c.c1);
        if_then(c, cond, [&]() { pp_bar(c); });
    }
    int n_acc = c.TM * c.TN;
    V zero_acc = rocke_b_zero_vec_f32(b, 8);
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
        std::vector<V> accs(loop.iter_vars, loop.iter_vars + n_acc);
        if(c.RR)
            chunk_rr(c, loop.iv, accs);
        else if(c.PP)
            chunk_pp(c, loop.iv, accs);
        else
            chunk(c, loop.iv, accs);
        rocke_b_scf_yield(b, accs.data(), n_acc);
    }
    rocke_b_region_leave(b);
    if(c.SB)
        rocke_b_s_barrier_wait(b, 0xFFFF);
    if(c.PP)
    {
        V cond = rocke_b_cmp_eq(b, c.rank, c.c0);
        if_then(c, cond, [&]() { pp_bar(c); });
    }
    if(spec->wave_epilogue)
        epilogue_wave(c, loop.op->results);
    else
        epilogue(c, loop.op->results);
    if(!rocke_ir_builder_ok(b))
        return NULL;
    return b->kernel;
}

rocke_kernel_def_t* rocke_build_direct_fwd_wmma_new(rocke_ir_builder_t* b,
                                                    const rocke_direct_fwd_wmma_spec_t* spec,
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
            rocke_i_set_err_msg(b, ROCKE_ERR_VALUE, "build_direct_fwd_wmma: null spec");
            return NULL;
        }
        if(rocke_direct_fwd_wmma_kernel_name(spec, name, sizeof(name)) != ROCKE_OK)
        {
            rocke_i_set_err_msg(
                b, ROCKE_ERR_VALUE, "build_direct_fwd_wmma: kernel name does not fit");
            return NULL;
        }
        if(rocke_ir_builder_init(b, name) != ROCKE_OK)
            return NULL;
        return rocke_build_direct_fwd_wmma(b, spec, arch);
    });
}

rocke_status_t rocke_direct_fwd_wmma_lower_to_llvm(const rocke_direct_fwd_wmma_spec_t* spec,
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
        fww_set_reason(err, err_cap, "lower_to_llvm: null spec/out");
        return ROCKE_ERR_VALUE;
    }
    if(arch == NULL)
        arch = "gfx1250";
    kernel = rocke_build_direct_fwd_wmma_new(&b, spec, arch);
    if(kernel == NULL)
    {
        const char* m = rocke_ir_builder_error(&b);
        st = rocke_ir_builder_status(&b);
        fww_set_reason(
            err, err_cap, (m != NULL && m[0] != '\0') ? m : "build_direct_fwd_wmma failed");
        rocke_ir_builder_free(&b);
        return (st == ROCKE_OK) ? ROCKE_ERR_VALUE : st;
    }
    st = rocke_lower_kernel_to_llvm_ex(kernel, flavor, arch, out_ll, err, err_cap);
    rocke_ir_builder_free(&b);
    return st;
}
