// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * conv_direct_dgrad_wmma.cpp -- C++ port of build_direct_dgrad_wmma
 * (library/kernels/common/conv_direct_dgrad_wmma.py): the wave32 WMMA direct
 * stride-1 dgrad for gfx1250. See the header and the Python module for the
 * algorithm.
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

/* Spec @property helpers the builder shares. */
int dgw_halo_h(const rocke_direct_dgrad_wmma_spec_t* s)
{
    return rocke_direct_dgrad_wmma_tile_h(s) + s->problem.KH - 1;
}
int dgw_halo_w(const rocke_direct_dgrad_wmma_spec_t* s)
{
    return s->tile_w + s->problem.KW - 1;
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
    return (s->problem.KW - 1 - s->problem.PAD) * dgw_x_pitch(s);
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
    long epi = (long)rocke_direct_dgrad_wmma_block_p(spec) * dgw_w_pitch(spec);
    return 2 * (ring > epi ? ring : epi);
}

void rocke_direct_dgrad_wmma_grid(const rocke_direct_dgrad_wmma_spec_t* spec, int grid[3])
{
    const rocke_direct_conv_problem_t* p = &spec->problem;
    int n_ct = dgw_ceil_div(p->cpg, rocke_direct_dgrad_wmma_block_c(spec));
    int n_ht = dgw_ceil_div(p->H, rocke_direct_dgrad_wmma_tile_h(spec));
    int n_wt = dgw_ceil_div(p->W, spec->tile_w);
    grid[0] = dgw_ceil_div(p->N, spec->tile_n) * p->groups * n_ht * n_wt * n_ct;
    grid[1] = 1;
    grid[2] = 1;
}

rocke_status_t rocke_direct_dgrad_wmma_kernel_name(const rocke_direct_dgrad_wmma_spec_t* spec,
                                                   char* out,
                                                   size_t out_cap)
{
    char f[32], pad[32], wc[32], wp[32], tc[32], tp[32], tw[32], tn[32], ck[32], st[32];
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
    const char* parts[]
        = {f, pad, wc, wp, tc, tp, tw, tn, ck, st, spec->prefetch ? "pf" : "", sched.c_str()};
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
    int tw, block_p, vgprs;

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
    if(p->stride != 1)
    {
        snprintf(msg, sizeof msg, "stride %d != 1", p->stride);
        dgw_set_reason(reason, reason_cap, msg);
        return false;
    }
    if(p->dil_h != 1 || p->dil_w != 1)
    {
        dgw_set_reason(reason, reason_cap, "dilation is not supported");
        return false;
    }
    if(!(0 <= p->PAD && p->PAD <= p->KH - 1 && p->PAD <= p->KW - 1))
    {
        snprintf(msg, sizeof msg, "PAD %d outside 0..K-1", p->PAD);
        dgw_set_reason(reason, reason_cap, msg);
        return false;
    }
    if(p->cpg % 8 != 0 || p->kpg % 8 != 0)
    {
        snprintf(msg,
                 sizeof msg,
                 "cpg %d and kpg %d must be multiples of 8 (16-byte rows)",
                 p->cpg,
                 p->kpg);
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
    if(!(tw == 1 || tw == 2 || tw == 4 || tw == 8 || tw % 16 == 0)
       || block_p % (tw * spec->tile_n) != 0)
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
    vgprs = 8 * spec->tiles_c * spec->tiles_p
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
        long long Ho = (long long)p->H + 2 * p->PAD - p->KH + 1;
        long long Wo = (long long)p->W + 2 * p->PAD - p->KW + 1;
        long long total_k = (long long)p->groups * p->kpg;
        long long total_c = (long long)p->groups * p->cpg;
        long long bytes[3] = {2 * p->N * Ho * Wo * total_k,
                              2 * total_k * p->KH * p->KW * p->cpg,
                              2 * p->N * (long long)p->H * p->W * total_c};
        long long mx = bytes[0];
        long long tiles = (long long)dgw_ceil_div(p->N, spec->tile_n) * p->groups
                          * dgw_ceil_div(p->H, rocke_direct_dgrad_wmma_tile_h(spec))
                          * dgw_ceil_div(p->W, spec->tile_w)
                          * dgw_ceil_div(p->cpg, rocke_direct_dgrad_wmma_block_c(spec));
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

    int KH, KW, PADD_H, PADD_W;
    int WAVE, WC, WP, TM, TN, THREADS, BC, TH, TW, HALO_H, HALO_W, CK, NS;
    int XP, WPITCH, HALO, WEI, NB, IMG, HALO_STAGE, TAPS, WEI0;

    rocke_value_t *c0, *c1, *c2, *c8, *c16;
    rocke_value_t *c_cpg, *c_kpg, *c_N, *c_H, *c_W, *c_Ho, *c_Wo;
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

struct dgw_step
{
    int ks, r, s;
};

/* _frags(step): TN B fragments, then TM A fragments. */
void frags(dgw_ctx& c,
           rocke_value_t* halo_base,
           rocke_value_t* wei_base,
           const dgw_step& st,
           std::vector<rocke_value_t*>& af,
           std::vector<rocke_value_t*>& bf)
{
    int tap = st.r * c.KW + st.s;
    int tap_off = ((c.KH - 1 - st.r) * c.HALO_W + (c.KW - 1 - st.s)) * c.XP + st.ks * DGW_WMMA_K;
    int row_off = (tap * c.CK + st.ks * DGW_WMMA_K) * c.WPITCH;
    bf.assign(c.TN, NULL);
    af.assign(c.TM, NULL);
    for(int tn = 0; tn < c.TN; ++tn)
        bf[tn] = read_b(c, halo_base, tn, tap_off);
    for(int m = 0; m < c.TM; ++m)
        af[m] = read_a(c, wei_base, m, row_off);
}

/* _chunk(kc, accs): publish chunk kc, refill the ring, multiply the chunk. */
void chunk(dgw_ctx& c, rocke_value_t* kc, std::vector<rocke_value_t*>& accs)
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
    rocke_b_sync_lds_only(b);
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
    std::vector<dgw_step> steps;
    for(int ks = 0; ks < c.CK / DGW_WMMA_K; ++ks)
        for(int r = 0; r < c.KH; ++r)
            for(int s = 0; s < c.KW; ++s)
                steps.push_back(dgw_step{ks, r, s});

    rocke_b_s_setprio(b, 1);
    std::vector<rocke_value_t*> af, bf, naf, nbf;
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
        for(int m = 0; m < c.TM; ++m)
            for(int tn = 0; tn < c.TN; ++tn)
            {
                int i = m * c.TN + tn;
                accs[i] = rocke_b_mma(b, c.wmma_op_id, af[m], bf[tn], accs[i], NULL, 0);
            }
    }
    rocke_b_s_setprio(b, 0);
}

/* _epilogue(accs): stage the dX tile in LDS, one TDM store per image. */
void epilogue(dgw_ctx& c, rocke_value_t* const* accs)
{
    rocke_ir_builder_t* b = c.b;
    const rocke_dconv_params_t* params = &c.params;
    rocke_b_sync_lds_only(b);
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
            rocke_value_t* val = rocke_b_vec_cast_f32_to(b, accs[m * c.TN + tn], c.io_type);
            rocke_value_t* x = ci(c, c.WPITCH);
            rocke_value_t* y = rocke_b_mul(b, px, x);
            rocke_value_t* z = rocke_b_add(b, y, col);
            rocke_value_t* idx[2] = {c.c0, z};
            rocke_b_smem_store_vN(b, c.lds, idx, 2, val, 8);
        }
    }
    rocke_b_sync_lds_only(b);
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
                rocke_value_t* y = rocke_b_mul(b, c.h_tile, params->p_D_stride_ho);
                rocke_value_t* z = rocke_b_add(b, x, y);
                rocke_value_t* u = rocke_b_mul(b, c.w_tile, params->p_D_stride_wo);
                rocke_value_t* v = rocke_b_add(b, c.c_grp, c.c_base);
                rocke_value_t* w = rocke_b_add(b, u, v);
                off = rocke_b_add(b, z, w);
            }
            {
                rocke_value_t* x = rocke_b_sub(b, c.c_H, c.h_tile);
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
                rocke_value_t* x = ci(c, (int64_t)im * c.TH * c.TW * c.WPITCH * 2);
                lds_src = rocke_b_add(b, c.halo_addr, x);
            }
            rocke_value_t* w_ext;
            {
                rocke_value_t* x = rocke_b_sub(b, c.c_W, c.w_tile);
                w_ext = rocke_b_smax(b, x, c.c0);
            }
            rocke_value_t* th = ci(c, c.TH);
            tdm_descriptor_3d(c,
                              dst,
                              lds_src,
                              c.c_ext,
                              w_ext,
                              h_ext,
                              params->p_D_stride_wo,
                              params->p_D_stride_ho,
                              c.WPITCH,
                              c.TW,
                              th,
                              d);
            rocke_b_tensor_store_from_lds(b, d[0], d[1], d[2], d[3], d[4], 0);
            rocke_b_s_wait_tensorcnt(b, 0);
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
    c.PADD_H = c.KH - 1 - p->PAD;
    c.PADD_W = c.KW - 1 - p->PAD;
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
    c.c_cpg = rocke_b_div(b, params->p_total_c, params->p_groups);
    c.c_kpg = rocke_b_div(b, params->p_total_k, params->p_groups);
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
    rocke_value_t *n_ct, *n_ht, *n_wt;
    {
        rocke_value_t* x = ci(c, c.BC - 1);
        rocke_value_t* y = rocke_b_add(b, c.c_cpg, x);
        rocke_value_t* z = ci(c, c.BC);
        n_ct = rocke_b_div(b, y, z);
    }
    {
        rocke_value_t* x = ci(c, c.TH - 1);
        rocke_value_t* y = rocke_b_add(b, c.c_H, x);
        rocke_value_t* z = ci(c, c.TH);
        n_ht = rocke_b_div(b, y, z);
    }
    {
        rocke_value_t* x = ci(c, c.TW - 1);
        rocke_value_t* y = rocke_b_add(b, c.c_W, x);
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
    rocke_value_t* g = rocke_b_mod(b, t, params->p_groups);
    {
        rocke_value_t* x = rocke_b_div(b, t, params->p_groups);
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
    int n_acc = c.TM * c.TN;
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
