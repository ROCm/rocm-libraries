// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * conv_dgrad_smallc_wmma.cpp -- C++ port of build_dgrad_smallc_wmma
 * (library/kernels/common/conv_dgrad_smallc_wmma.py): the wave32 WMMA dgrad of
 * a small-C stem (GEMM over the flattened filter, then col2im in LDS) for
 * gfx1250. See the header and the Python module for the algorithm.
 *
 * Every helper emits in PYTHON EVALUATION ORDER (arguments left-to-right,
 * innermost call first) so the SSA sequence -- and therefore the .ll bytes --
 * match exactly. Nested builder calls are hoisted into ordered locals: C++
 * leaves the evaluation order of call arguments unspecified.
 */
#include "rocke/instance_conv_dgrad_smallc_wmma.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <string>
#include <utility>
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

/* conv_dgrad_smallc_wmma.SMALLC_MAX_C. */
#define DSW_MAX_C 4

namespace
{

const char* dsw_dtype(const rocke_dgrad_smallc_wmma_spec_t* spec)
{
    return spec->problem.dtype ? spec->problem.dtype : "fp16";
}

bool dsw_is_bf16(const rocke_dgrad_smallc_wmma_spec_t* spec)
{
    return strcmp(dsw_dtype(spec), "bf16") == 0;
}

/* Spec @property helpers the builder shares. */
int dsw_S(const rocke_dgrad_smallc_wmma_spec_t* s)
{
    return s->problem.stride;
}
int dsw_ncols(const rocke_dgrad_smallc_wmma_spec_t* s)
{
    return s->problem.KH * s->problem.KW * s->problem.cpg;
}
int dsw_off_h(const rocke_dgrad_smallc_wmma_spec_t* s)
{
    return (int)floordiv(s->problem.KH - 1 - s->problem.PAD, dsw_S(s));
}
int dsw_off_w(const rocke_dgrad_smallc_wmma_spec_t* s)
{
    return (int)floordiv(s->problem.KW - 1 - s->problem.PAD, dsw_S(s));
}
int dsw_x_pitch(const rocke_dgrad_smallc_wmma_spec_t* s)
{
    return s->ck + kChPad;
}
int dsw_w_pitch(const rocke_dgrad_smallc_wmma_spec_t* s)
{
    return rocke_dgrad_smallc_wmma_block_c(s) + kChPad;
}
int dsw_z_pitch(const rocke_dgrad_smallc_wmma_spec_t* s)
{
    return rocke_dgrad_smallc_wmma_block_c(s) + 8;
}
int dsw_x_elems(const rocke_dgrad_smallc_wmma_spec_t* s)
{
    return (rocke_dgrad_smallc_wmma_block_p(s) + dsw_off_w(s) + 1) * dsw_x_pitch(s);
}
int dsw_wei_elems(const rocke_dgrad_smallc_wmma_spec_t* s)
{
    return s->ck * dsw_w_pitch(s);
}

/* DirectConvProblem.Ho / Wo (strided, dilated). */
long long dsw_Ho(const rocke_direct_conv_problem_t* p)
{
    return floordiv((long long)p->H + 2 * p->PAD - (long long)(p->KH - 1) * p->dil_h - 1, p->stride)
           + 1;
}
long long dsw_Wo(const rocke_direct_conv_problem_t* p)
{
    return floordiv((long long)p->W + 2 * p->PAD - (long long)(p->KW - 1) * p->dil_w - 1, p->stride)
           + 1;
}

} // namespace

/* ===================================================================== *
 *  Spec defaults + derived geometry
 * ===================================================================== */

rocke_dgrad_smallc_wmma_spec_t rocke_dgrad_smallc_wmma_spec_default(void)
{
    rocke_dgrad_smallc_wmma_spec_t s;
    memset(&s, 0, sizeof(s));
    s.problem = rocke_direct_conv_problem_default();
    s.name = "dgrad_smallc_wmma";
    s.waves_c = 1;
    s.waves_p = 4;
    s.tiles_c = 2;
    s.tiles_p = 5;
    s.out_h = 32;
    s.out_w = 32;
    s.ck = 32;
    s.stages = 2;
    s.prefetch = false;
    s.sched = NULL;
    s.wave_size = 32;
    return s;
}

int rocke_dgrad_smallc_wmma_block_c(const rocke_dgrad_smallc_wmma_spec_t* spec)
{
    return spec->waves_c * spec->tiles_c * 16;
}

int rocke_dgrad_smallc_wmma_block_p(const rocke_dgrad_smallc_wmma_spec_t* spec)
{
    return spec->waves_p * spec->tiles_p * 16;
}

int rocke_dgrad_smallc_wmma_dy_h(const rocke_dgrad_smallc_wmma_spec_t* spec)
{
    return (int)floordiv(spec->out_h - 1 + spec->problem.PAD, dsw_S(spec)) + dsw_off_h(spec) + 1;
}

int rocke_dgrad_smallc_wmma_dy_w(const rocke_dgrad_smallc_wmma_spec_t* spec)
{
    return (int)floordiv(spec->out_w - 1 + spec->problem.PAD, dsw_S(spec)) + dsw_off_w(spec) + 1;
}

int rocke_dgrad_smallc_wmma_threads_per_block(const rocke_dgrad_smallc_wmma_spec_t* spec)
{
    return spec->waves_c * spec->waves_p * spec->wave_size;
}

long rocke_dgrad_smallc_wmma_lds_bytes(const rocke_dgrad_smallc_wmma_spec_t* spec)
{
    long ring = 2L * spec->stages * ((long)dsw_x_elems(spec) + dsw_wei_elems(spec));
    long z = 4L * rocke_dgrad_smallc_wmma_block_p(spec) * dsw_z_pitch(spec);
    long out = 2L * spec->out_h * spec->out_w * spec->problem.cpg;
    return ring > z + out ? ring : z + out;
}

void rocke_dgrad_smallc_wmma_grid(const rocke_dgrad_smallc_wmma_spec_t* spec, int grid[3])
{
    const rocke_direct_conv_problem_t* p = &spec->problem;
    grid[0] = (int)((long long)p->N * ceil_div(p->H, spec->out_h) * ceil_div(p->W, spec->out_w));
    grid[1] = 1;
    grid[2] = 1;
}

rocke_status_t rocke_dgrad_smallc_wmma_kernel_name(const rocke_dgrad_smallc_wmma_spec_t* spec,
                                                   char* out,
                                                   size_t out_cap)
{
    char f[32], sv[32], pad[32], cc[32], wc[32], wp[32], tc[32], tp[32], oh[32], ow[32], ck[32],
        st[32];
    std::string sched;
    const rocke_direct_conv_problem_t* p = &spec->problem;
    snprintf(f, sizeof f, "f%dx%d", p->KH, p->KW);
    snprintf(sv, sizeof sv, "s%d", p->stride);
    snprintf(pad, sizeof pad, "p%d", p->PAD);
    snprintf(cc, sizeof cc, "c%d", p->cpg);
    snprintf(wc, sizeof wc, "wc%d", spec->waves_c);
    snprintf(wp, sizeof wp, "wp%d", spec->waves_p);
    snprintf(tc, sizeof tc, "tc%d", spec->tiles_c);
    snprintf(tp, sizeof tp, "tp%d", spec->tiles_p);
    snprintf(oh, sizeof oh, "oh%d", spec->out_h);
    snprintf(ow, sizeof ow, "ow%d", spec->out_w);
    snprintf(ck, sizeof ck, "ck%d", spec->ck);
    snprintf(st, sizeof st, "st%d", spec->stages);
    if(spec->sched != NULL && spec->sched[0] != '\0')
        sched = std::string("s") + spec->sched;
    const char* parts[] = {
        f, sv, pad, cc, wc, wp, tc, tp, oh, ow, ck, st, spec->prefetch ? "pf" : "", sched.c_str()};
    const char* flag_names[] = {"bf16"};
    int flag_on[] = {dsw_is_bf16(spec) ? 1 : 0};
    return rocke_kernel_name_join(spec->name ? spec->name : "dgrad_smallc_wmma",
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
 *  dgrad_smallc_reason / is_valid_dgrad_smallc_spec
 * ===================================================================== */

bool rocke_dgrad_smallc_wmma_is_valid_spec(const rocke_dgrad_smallc_wmma_spec_t* spec,
                                           const char* arch,
                                           char* reason,
                                           size_t reason_cap)
{
    char msg[512];
    const rocke_archtarget_t* target;
    const rocke_direct_conv_problem_t* p;
    const char* ab;
    int S, block_c, block_p, ncols, vgprs;

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
        set_reason(reason, reason_cap, "dgrad_smallc_wmma is a wave32 kernel");
        return false;
    }
    if(!target->memory.has_tdm)
    {
        snprintf(msg, sizeof msg, "no TDM on %s", arch);
        set_reason(reason, reason_cap, msg);
        return false;
    }
    if(strcmp(dsw_dtype(spec), "fp16") != 0 && strcmp(dsw_dtype(spec), "bf16") != 0)
    {
        snprintf(msg, sizeof msg, "unsupported dtype '%s'", dsw_dtype(spec));
        set_reason(reason, reason_cap, msg);
        return false;
    }
    ab = dsw_is_bf16(spec) ? "bf16" : "f16";
    if(!rocke_mma_catalog_has_shape(
           rocke_archtarget_mma(target), "wmma", ab, ab, "fp32", 16, 16, kWmmaK))
    {
        snprintf(msg, sizeof msg, "missing wmma_f32_16x16x32_%s on %s", ab, arch);
        set_reason(reason, reason_cap, msg);
        return false;
    }
    if(p->groups != 1 || p->dil_h != 1 || p->dil_w != 1)
    {
        set_reason(reason, reason_cap, "ungrouped, undilated convolutions only");
        return false;
    }
    if(p->cpg > DSW_MAX_C)
    {
        snprintf(msg, sizeof msg, "C %d > %d", p->cpg, DSW_MAX_C);
        set_reason(reason, reason_cap, msg);
        return false;
    }
    if(p->stride == p->KH && p->PAD == 0)
    {
        set_reason(reason, reason_cap, "patch embedding: use dgrad_patch_wmma");
        return false;
    }
    if(!(0 <= p->PAD && p->PAD <= (p->KH < p->KW ? p->KH : p->KW) - 1))
    {
        snprintf(msg, sizeof msg, "PAD %d outside 0..K-1", p->PAD);
        set_reason(reason, reason_cap, msg);
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
    if(spec->waves_c < 1 || spec->waves_p < 1 || spec->tiles_c < 1 || spec->tiles_p < 1)
    {
        set_reason(reason, reason_cap, "waves and tiles must be >= 1");
        return false;
    }
    S = dsw_S(spec);
    if(spec->out_h % S != 0 || spec->out_w % S != 0 || spec->out_h < 1 || spec->out_w < 1)
    {
        snprintf(msg,
                 sizeof msg,
                 "out tile %dx%d must be multiples of S %d",
                 spec->out_h,
                 spec->out_w,
                 S);
        set_reason(reason, reason_cap, msg);
        return false;
    }
    block_c = rocke_dgrad_smallc_wmma_block_c(spec);
    block_p = rocke_dgrad_smallc_wmma_block_p(spec);
    ncols = dsw_ncols(spec);
    if(block_c < ncols || block_c - ncols >= 16)
    {
        snprintf(msg, sizeof msg, "block_c %d must be R*S*C = %d rounded up to 16", block_c, ncols);
        set_reason(reason, reason_cap, msg);
        return false;
    }
    if((long long)rocke_dgrad_smallc_wmma_dy_h(spec) * rocke_dgrad_smallc_wmma_dy_w(spec) > block_p)
    {
        snprintf(msg,
                 sizeof msg,
                 "dY window %dx%d > %d GEMM pixels",
                 rocke_dgrad_smallc_wmma_dy_h(spec),
                 rocke_dgrad_smallc_wmma_dy_w(spec),
                 block_p);
        set_reason(reason, reason_cap, msg);
        return false;
    }
    if(spec->stages < 2)
    {
        set_reason(reason, reason_cap, "stages must be >= 2");
        return false;
    }
    if(spec->waves_c * spec->waves_p < 2)
    {
        set_reason(reason, reason_cap, "needs at least two waves");
        return false;
    }
    if(rocke_dgrad_smallc_wmma_threads_per_block(spec)
       > rocke_archtarget_max_threads_per_block(target))
    {
        snprintf(msg,
                 sizeof msg,
                 "%d threads per block",
                 rocke_dgrad_smallc_wmma_threads_per_block(spec));
        set_reason(reason, reason_cap, msg);
        return false;
    }
    if(!rocke_archtarget_fits_lds(target, rocke_dgrad_smallc_wmma_lds_bytes(spec)))
    {
        snprintf(msg,
                 sizeof msg,
                 "LDS %ld B > %d",
                 rocke_dgrad_smallc_wmma_lds_bytes(spec),
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
        long long bytes[3] = {2 * p->N * dsw_Ho(p) * dsw_Wo(p) * total_k,
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

rocke_kernel_def_t* rocke_build_dgrad_smallc_wmma(rocke_ir_builder_t* b,
                                                  const rocke_dgrad_smallc_wmma_spec_t* spec,
                                                  const char* arch)
{
    char reason[512];

    if(b == NULL || spec == NULL)
        return NULL;
    if(arch == NULL)
        arch = "gfx1250";
    if(!rocke_dgrad_smallc_wmma_is_valid_spec(spec, arch, reason, sizeof reason))
    {
        std::string msg = std::string("invalid DgradSmallcWmmaSpec for ") + arch + ": " + reason;
        rocke_i_set_err_msg(b, ROCKE_ERR_VALUE, msg.c_str());
        return NULL;
    }
    const rocke_direct_conv_problem_t* p = &spec->problem;
    const int KH = p->KH, KW = p->KW, S = p->stride, PAD = p->PAD, C = p->cpg;
    const int NCOL = dsw_ncols(spec);
    const rocke_type_t* io_type = rocke_b_io_ir_type(b, dsw_dtype(spec));
    if(io_type == NULL)
        return NULL;
    const char* wmma_op_id;
    {
        const char* ab = dsw_is_bf16(spec) ? "bf16" : "f16";
        const rocke_archtarget_t* target = rocke_archtarget_from_gfx(arch);
        const rocke_mmaop_t* op
            = rocke_archtarget_op_for_shape(target, "wmma", ab, ab, "fp32", 16, 16, kWmmaK, NULL);
        if(op == NULL || op->op_id == NULL)
        {
            rocke_i_set_err_msg(b, ROCKE_ERR_VALUE, "build_dgrad_smallc_wmma: no WMMA atom");
            return NULL;
        }
        wmma_op_id = op->op_id;
    }

    const int WAVE = spec->wave_size;
    const int WC = spec->waves_c;
    const int TM = spec->tiles_c, TN = spec->tiles_p;
    const int THREADS = rocke_dgrad_smallc_wmma_threads_per_block(spec);
    const int BP = rocke_dgrad_smallc_wmma_block_p(spec);
    const int OH = spec->out_h, OW = spec->out_w;
    const int DH = rocke_dgrad_smallc_wmma_dy_h(spec), DW = rocke_dgrad_smallc_wmma_dy_w(spec);
    const int OFFH = dsw_off_h(spec), OFFW = dsw_off_w(spec);
    const int CK = spec->ck, NS = spec->stages;
    const int XP = dsw_x_pitch(spec), WPITCH = dsw_w_pitch(spec), ZP = dsw_z_pitch(spec);
    const int XE = dsw_x_elems(spec), WEI = dsw_wei_elems(spec);

    rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", THREADS);
    if(spec->sched != NULL)
        rocke_attr_set_str(b, &b->kernel->attrs, "scheduler_strategy", spec->sched);
    rocke_dconv_params_t params_storage;
    rocke_dconv_emit_params(b, &params_storage, "dgrad", io_type);
    const rocke_dconv_params_t* params = &params_storage;

    V c0 = ci(b, 0);
    V c1 = ci(b, 1);
    V c2 = ci(b, 2);
    V c8 = ci(b, 8);
    V c16 = ci(b, 16);
    V c_K = params->p_total_k;
    V c_H = params->p_Hi;
    V c_W = params->p_Wi;
    V c_Ho = params->p_Ho;
    V c_Wo = params->p_Wo;

    V tid = rocke_b_thread_id_x(b);
    V wave_id, lane, l16, lhalf, wave_c, wave_p;
    {
        V c_wave = ci(b, WAVE);
        wave_id = rocke_b_div(b, tid, c_wave);
        lane = rocke_b_mod(b, tid, c_wave);
    }
    l16 = rocke_b_mod(b, lane, c16);
    lhalf = rocke_b_div(b, lane, c16);
    {
        V x = ci(b, WC);
        wave_c = rocke_b_mod(b, wave_id, x);
        V y = ci(b, WC);
        wave_p = rocke_b_div(b, wave_id, y);
    }

    V n_ht, n_wt;
    {
        V x = ci(b, OH - 1);
        V y = rocke_b_add(b, c_H, x);
        V z = ci(b, OH);
        n_ht = rocke_b_div(b, y, z);
    }
    {
        V x = ci(b, OW - 1);
        V y = rocke_b_add(b, c_W, x);
        V z = ci(b, OW);
        n_wt = rocke_b_div(b, y, z);
    }
    V t = rocke_b_block_id_x(b);
    V wt = rocke_b_mod(b, t, n_wt);
    t = rocke_b_div(b, t, n_wt);
    V ht = rocke_b_mod(b, t, n_ht);
    V n_img = rocke_b_div(b, t, n_ht);
    V h0, w0, c_CK, n_chunks;
    {
        V x = ci(b, OH);
        h0 = rocke_b_mul(b, ht, x);
    }
    {
        V x = ci(b, OW);
        w0 = rocke_b_mul(b, wt, x);
    }
    c_CK = ci(b, CK);
    {
        V x = ci(b, CK - 1);
        V y = rocke_b_add(b, c_K, x);
        n_chunks = rocke_b_div(b, y, c_CK);
    }

    /* The dY window starts at (h0/S - OFFH, w0/S - OFFW); a negative start is
     * clipped and the transfer lands shifted (the cells before it are zeroed). */
    V i_lo, j_lo, hs, sw, i_start, j_start, z_elems, dy_w_ext, dy_h_ext, dy_tile2, dy_base;
    {
        V x = ci(b, S);
        V y = rocke_b_div(b, h0, x);
        V z = ci(b, OFFH);
        i_lo = rocke_b_sub(b, y, z);
    }
    {
        V x = ci(b, S);
        V y = rocke_b_div(b, w0, x);
        V z = ci(b, OFFW);
        j_lo = rocke_b_sub(b, y, z);
    }
    {
        V x = rocke_b_sub(b, c0, i_lo);
        hs = rocke_b_smax(b, x, c0);
    }
    {
        V x = rocke_b_sub(b, c0, j_lo);
        sw = rocke_b_smax(b, x, c0);
    }
    i_start = rocke_b_add(b, i_lo, hs);
    j_start = rocke_b_add(b, j_lo, sw);
    {
        V x = ci(b, DW);
        V y = rocke_b_mul(b, hs, x);
        V z = rocke_b_add(b, y, sw);
        V u = ci(b, XP);
        z_elems = rocke_b_mul(b, z, u);
    }
    {
        V x = rocke_b_sub(b, c_Wo, j_start);
        V y = ci(b, DW);
        V z = rocke_b_sub(b, y, sw);
        V u = rocke_b_smin(b, x, z);
        dy_w_ext = rocke_b_smax(b, u, c0);
    }
    {
        V x = rocke_b_sub(b, c_Ho, i_start);
        dy_h_ext = rocke_b_smax(b, x, c0);
    }
    {
        V x = ci(b, DH);
        dy_tile2 = rocke_b_sub(b, x, hs);
    }
    {
        V x = rocke_b_mul(b, n_img, params->p_A_stride_n);
        V y = rocke_b_mul(b, i_start, params->p_A_stride_hi);
        V z = rocke_b_add(b, x, y);
        V u = rocke_b_mul(b, j_start, params->p_A_stride_wi);
        dy_base = rocke_b_add(b, z, u);
    }
    V c_NC = ci(b, NCOL);

    V lds, lds_addr, wei_addr;
    {
        int shape[2] = {1, (int)(rocke_dgrad_smallc_wmma_lds_bytes(spec) / 2)};
        lds = rocke_b_smem_alloc(b, io_type, shape, 2, "sc_lds");
    }
    {
        V a = rocke_b_smem_addr_of(b, lds);
        lds_addr = rocke_b_trunc(b, a, rocke_i32());
    }
    const int WEI0 = NS * XE;
    {
        V x = ci(b, (int64_t)WEI0 * 2);
        wei_addr = rocke_b_add(b, lds_addr, x);
    }

    {
        const int max_z = (OFFH * DW + OFFW) * XP;
        if(max_z)
        {
            V zero_v = rocke_b_zero_vec(b, io_type, 8);
            V z_vecs = rocke_b_div(b, z_elems, c8);
            for(int j = 0; j < (int)ceil_div(max_z, 8); j += THREADS)
            {
                V cj = ci(b, j);
                V v = rocke_b_add(b, tid, cj);
                V cond = rocke_b_cmp_lt(b, v, z_vecs);
                rocke_if_t gate = rocke_b_scf_if(b, cond);
                rocke_b_region_enter(b, gate.then_region);
                V flat = rocke_b_mul(b, v, c8);
                for(int st = 0; st < NS; ++st)
                {
                    V o = ci(b, (int64_t)st * XE);
                    V a = rocke_b_add(b, flat, o);
                    V idx[2] = {c0, a};
                    rocke_b_smem_store_vN(b, lds, idx, 2, zero_v, 8);
                }
                rocke_b_region_leave(b);
            }
        }
    }

    rocke_dgstem::Gemm g;
    g.b = b;
    g.io_type = io_type;
    g.wmma_op_id = wmma_op_id;
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
    g.lane = lane;
    g.l16 = l16;
    g.lhalf = lhalf;
    g.wave_c = wave_c;
    g.wave_p = wave_p;
    g.lds = lds;
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
            V cond = rocke_b_cmp_eq(b, wave_id, c0);
            rocke_if_t gate = rocke_b_scf_if(b, cond);
            rocke_b_region_enter(b, gate.then_region);
            V src, dst, d[5];
            {
                V x = rocke_b_add(b, dy_base, k0);
                V y = rocke_b_mul(b, x, c2);
                src = rocke_b_global_ptr_add(b, params->A, y);
            }
            {
                V x = ci(b, XE);
                V y = rocke_b_mul(b, stage, x);
                V z = rocke_b_add(b, y, z_elems);
                V u = rocke_b_mul(b, z, c2);
                dst = rocke_b_add(b, lds_addr, u);
            }
            rocke_dgstem::tdm_descriptor_3d(b,
                                            src,
                                            dst,
                                            k_ext,
                                            dy_w_ext,
                                            dy_h_ext,
                                            params->p_A_stride_wi,
                                            params->p_A_stride_hi,
                                            XP,
                                            DW,
                                            dy_tile2,
                                            d);
            rocke_b_tensor_load_to_lds(b, d[0], d[1], d[2], d[3], d[4], 0);
            rocke_b_region_leave(b);
        }
        {
            V cond = rocke_b_cmp_eq(b, wave_id, c1);
            rocke_if_t gate = rocke_b_scf_if(b, cond);
            rocke_b_region_enter(b, gate.then_region);
            V src, dst, d[5];
            {
                V x = rocke_b_mul(b, k0, c_NC);
                V y = rocke_b_mul(b, x, c2);
                src = rocke_b_global_ptr_add(b, params->Bp, y);
            }
            {
                V x = ci(b, (int64_t)WEI * 2);
                V y = rocke_b_mul(b, stage, x);
                dst = rocke_b_add(b, wei_addr, y);
            }
            V s2 = rocke_b_mul(b, c_NC, c_CK);
            rocke_dgstem::tdm_descriptor_3d(
                b, src, dst, c_NC, k_ext, c1, c_NC, s2, WPITCH, CK, c1, d);
            rocke_b_tensor_load_to_lds(b, d[0], d[1], d[2], d[3], d[4], 0);
            rocke_b_region_leave(b);
        }
    };

    rocke_dgstem::operand_offsets(g);
    std::vector<V> accs = rocke_dgstem::k_loop(g);
    if(accs.empty())
        return NULL;

    /* ---- epilogue 1: Z (fp32) to LDS as [pixel][col]. */
    rocke_b_sync_lds_only(b);
    for(int m = 0; m < TM; ++m)
    {
        V col;
        {
            V x = ci(b, TM);
            V y = rocke_b_mul(b, wave_c, x);
            V z = ci(b, m);
            V u = rocke_b_add(b, y, z);
            V v = rocke_b_mul(b, u, c16);
            V w = rocke_b_mul(b, lhalf, c8);
            col = rocke_b_add(b, v, w);
        }
        for(int tn = 0; tn < TN; ++tn)
        {
            V px;
            {
                V x = ci(b, TN);
                V y = rocke_b_mul(b, wave_p, x);
                V z = ci(b, tn);
                V u = rocke_b_add(b, y, z);
                V v = rocke_b_mul(b, u, c16);
                px = rocke_b_add(b, v, l16);
            }
            /* f32 element i of the shared 2-byte allocation sits at index 2*i. */
            V x = ci(b, ZP);
            V y = rocke_b_mul(b, px, x);
            V zi = rocke_b_add(b, y, col);
            V a = rocke_b_mul(b, zi, c2);
            V idx[2] = {c0, a};
            rocke_b_smem_store_vN(b, lds, idx, 2, accs[m * TN + tn], 8);
        }
    }
    rocke_b_sync_lds_only(b);

    /* ---- epilogue 2: overlap-add, phase by phase, into the dX tile. */
    const int OUT0 = BP * ZP * 2; /* dX tile after Z, in 2-byte elements */
    for(int ph = 0; ph < S; ++ph)
    {
        for(int pw = 0; pw < S; ++pw)
        {
            std::vector<std::pair<int, int>> taps;
            for(int r = 0; r < KH; ++r)
                for(int s = 0; s < KW; ++s)
                    if(floordiv(ph + PAD - r, S) * S == ph + PAD - r
                       && floordiv(pw + PAD - s, S) * S == pw + PAD - s)
                        taps.emplace_back(r, s);
            const int na = OH / S, nb = OW / S;
            const int E = na * nb; /* output pixels of this phase; a thread does all C */
            for(int it = 0; it < (int)ceil_div(E, THREADS); ++it)
            {
                V cit = ci(b, (int64_t)it * THREADS);
                V e = rocke_b_add(b, tid, cit);
                V cE = ci(b, E);
                V cond = rocke_b_cmp_lt(b, e, cE);
                rocke_if_t gate = rocke_b_scf_if(b, cond);
                rocke_b_region_enter(b, gate.then_region);
                V bcol, arow, base;
                {
                    V x = ci(b, nb);
                    bcol = rocke_b_mod(b, e, x);
                    V y = ci(b, nb);
                    arow = rocke_b_div(b, e, y);
                }
                {
                    V x = ci(b, DW);
                    V y = rocke_b_mul(b, arow, x);
                    V z = rocke_b_add(b, y, bcol);
                    V u = ci(b, ZP);
                    base = rocke_b_mul(b, z, u);
                }
                V zero_f = rocke_b_const_f32(b, 0.0);
                std::vector<V> acc(C, zero_f);
                for(const auto& tap : taps)
                {
                    const int r = tap.first, s = tap.second;
                    const int dh = (int)floordiv(ph + PAD - r, S) + OFFH;
                    const int dw = (int)floordiv(pw + PAD - s, S) + OFFW;
                    const int off = (dh * DW + dw) * ZP + (r * KW + s) * C;
                    V co = ci(b, off);
                    V idx0 = rocke_b_add(b, base, co);
                    V a = rocke_b_mul(b, idx0, c2);
                    V idx[2] = {c0, a};
                    V z = rocke_b_smem_load_vN(b, lds, idx, 2, rocke_f32(), C);
                    for(int i = 0; i < C; ++i)
                    {
                        V el = rocke_b_vec_extract(b, z, i);
                        acc[i] = rocke_b_fadd(b, acc[i], el);
                    }
                }
                V oh, ow, oidx;
                {
                    V x = ci(b, S);
                    V y = rocke_b_mul(b, arow, x);
                    V z = ci(b, ph);
                    oh = rocke_b_add(b, y, z);
                }
                {
                    V x = ci(b, S);
                    V y = rocke_b_mul(b, bcol, x);
                    V z = ci(b, pw);
                    ow = rocke_b_add(b, y, z);
                }
                {
                    V x = ci(b, OW);
                    V y = rocke_b_mul(b, oh, x);
                    V z = rocke_b_add(b, y, ow);
                    V u = ci(b, C);
                    V v = rocke_b_mul(b, z, u);
                    V w = ci(b, OUT0);
                    oidx = rocke_b_add(b, v, w);
                }
                for(int i = 0; i < C; ++i)
                {
                    V x = ci(b, i);
                    V a = rocke_b_add(b, oidx, x);
                    V idx[2] = {c0, a};
                    V h = dsw_is_bf16(spec) ? rocke_b_trunc_f32_to_bf16(b, acc[i])
                                            : rocke_b_trunc_f32_to_f16(b, acc[i]);
                    rocke_b_smem_store_vN(b, lds, idx, 2, h, 1);
                }
                rocke_b_region_leave(b);
            }
        }
    }
    rocke_b_sync_lds_only(b);
    {
        V cond = rocke_b_cmp_eq(b, wave_id, c0);
        rocke_if_t gate = rocke_b_scf_if(b, cond);
        rocke_b_region_enter(b, gate.then_region);
        V off, dst, d[5];
        {
            V x = rocke_b_mul(b, n_img, params->p_D_stride_n);
            V y = rocke_b_mul(b, h0, params->p_D_stride_ho);
            V z = rocke_b_add(b, x, y);
            V u = rocke_b_mul(b, w0, params->p_D_stride_wo);
            off = rocke_b_add(b, z, u);
        }
        {
            V x = rocke_b_mul(b, off, c2);
            dst = rocke_b_global_ptr_add(b, params->D, x);
        }
        V lds_src, d0, d1, s2;
        {
            V x = ci(b, (int64_t)OUT0 * 2);
            lds_src = rocke_b_add(b, lds_addr, x);
        }
        {
            V x = rocke_b_sub(b, c_W, w0);
            V y = rocke_b_smax(b, x, c0);
            V z = ci(b, C);
            d0 = rocke_b_mul(b, y, z);
        }
        {
            V x = rocke_b_sub(b, c_H, h0);
            d1 = rocke_b_smax(b, x, c0);
        }
        s2 = rocke_b_mul(b, params->p_D_stride_ho, c_H);
        rocke_dgstem::tdm_descriptor_3d(
            b, dst, lds_src, d0, d1, c1, params->p_D_stride_ho, s2, OW * C, OH, c1, d);
        rocke_b_tensor_store_from_lds(b, d[0], d[1], d[2], d[3], d[4], 0);
        rocke_b_s_wait_tensorcnt(b, 0);
        rocke_b_region_leave(b);
    }
    if(!rocke_ir_builder_ok(b))
        return NULL;
    return b->kernel;
}

rocke_kernel_def_t* rocke_build_dgrad_smallc_wmma_new(rocke_ir_builder_t* b,
                                                      const rocke_dgrad_smallc_wmma_spec_t* spec,
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
            rocke_i_set_err_msg(b, ROCKE_ERR_VALUE, "build_dgrad_smallc_wmma: null spec");
            return NULL;
        }
        if(rocke_dgrad_smallc_wmma_kernel_name(spec, name, sizeof(name)) != ROCKE_OK)
        {
            rocke_i_set_err_msg(
                b, ROCKE_ERR_VALUE, "build_dgrad_smallc_wmma: kernel name does not fit");
            return NULL;
        }
        if(rocke_ir_builder_init(b, name) != ROCKE_OK)
            return NULL;
        return rocke_build_dgrad_smallc_wmma(b, spec, arch);
    });
}

rocke_status_t rocke_dgrad_smallc_wmma_lower_to_llvm(const rocke_dgrad_smallc_wmma_spec_t* spec,
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
    kernel = rocke_build_dgrad_smallc_wmma_new(&b, spec, arch);
    if(kernel == NULL)
    {
        const char* m = rocke_ir_builder_error(&b);
        st = rocke_ir_builder_status(&b);
        set_reason(
            err, err_cap, (m != NULL && m[0] != '\0') ? m : "build_dgrad_smallc_wmma failed");
        rocke_ir_builder_free(&b);
        return (st == ROCKE_OK) ? ROCKE_ERR_VALUE : st;
    }
    st = rocke_lower_kernel_to_llvm_ex(kernel, flavor, arch, out_ll, err, err_cap);
    rocke_ir_builder_free(&b);
    return st;
}
