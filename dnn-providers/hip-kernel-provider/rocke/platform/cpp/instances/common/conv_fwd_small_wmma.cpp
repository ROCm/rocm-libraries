// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * conv_fwd_small_wmma.cpp -- C++ port of build_fwd_smallc_wmma and
 * build_fwd_smallk_wmma (library/kernels/common/conv_fwd_small_wmma.py): the
 * wave32 WMMA forward convolutions of tiny channel counts for gfx1250. See
 * the header and the Python module for the algorithms.
 *
 * Every helper emits in PYTHON EVALUATION ORDER (arguments left-to-right,
 * innermost call first) so the SSA sequence -- and therefore the .ll bytes --
 * match exactly. Nested builder calls are hoisted into ordered locals: C++
 * leaves the evaluation order of call arguments unspecified.
 */
#include "rocke/instance_conv_fwd_small_wmma.h"

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

/* conv_fwd_small_wmma.WMMA_K / CH_PAD / SMALL_WMMA_VGPR_BUDGET /
 * SMALLC_MAX_C / SMALLK_MAX_K. */
#define FSW_WMMA_K 32
#define FSW_CH_PAD 8
#define FSW_VGPR_BUDGET 448
#define FSW_SMALLC_MAX_C 4
#define FSW_SMALLK_MAX_K 8

namespace
{

/* Python floor division / ceil division / round-up on ints. */
long long fsw_floordiv(long long a, long long b)
{
    long long q = a / b;
    if((a % b != 0) && ((a < 0) != (b < 0)))
        --q;
    return q;
}

int fsw_ceil_div(long long a, long long b)
{
    return (int)-fsw_floordiv(-a, b);
}

int fsw_round_up(long long a, long long b)
{
    return (int)(fsw_ceil_div(a, b) * b);
}

void fsw_set_reason(char* reason, size_t reason_cap, const char* msg)
{
    if(reason && reason_cap > 0)
    {
        snprintf(reason, reason_cap, "%s", msg);
    }
}

const char* fsw_dtype(const rocke_direct_conv_problem_t* p)
{
    return p->dtype ? p->dtype : "fp16";
}

bool fsw_is_bf16(const rocke_direct_conv_problem_t* p)
{
    return strcmp(fsw_dtype(p), "bf16") == 0;
}

long long fsw_ho(const rocke_direct_conv_problem_t* p)
{
    return fsw_floordiv((long long)p->H + 2 * p->PAD - (long long)(p->KH - 1) * p->dil_h - 1,
                        p->stride)
           + 1;
}

long long fsw_wo(const rocke_direct_conv_problem_t* p)
{
    return fsw_floordiv((long long)p->W + 2 * p->PAD - (long long)(p->KW - 1) * p->dil_w - 1,
                        p->stride)
           + 1;
}

/* _common_reason(p, arch, name). On failure writes reason and returns false. */
bool fsw_common_reason(const rocke_direct_conv_problem_t* p,
                       const char* arch,
                       const char* name,
                       char* reason,
                       size_t reason_cap)
{
    char msg[512];
    const rocke_archtarget_t* target = rocke_archtarget_from_gfx(arch);
    if(target == NULL)
    {
        rocke_set_unknown_arch_reason(reason, reason_cap, arch);
        return false;
    }
    if(target->wave_size != 32)
    {
        snprintf(msg, sizeof msg, "%s is a wave32 kernel", name);
        fsw_set_reason(reason, reason_cap, msg);
        return false;
    }
    if(!target->memory.has_tdm)
    {
        snprintf(msg, sizeof msg, "no TDM on %s", arch);
        fsw_set_reason(reason, reason_cap, msg);
        return false;
    }
    if(strcmp(fsw_dtype(p), "fp16") != 0 && strcmp(fsw_dtype(p), "bf16") != 0)
    {
        snprintf(msg, sizeof msg, "unsupported dtype '%s'", fsw_dtype(p));
        fsw_set_reason(reason, reason_cap, msg);
        return false;
    }
    const char* ab = fsw_is_bf16(p) ? "bf16" : "f16";
    if(!rocke_mma_catalog_has_shape(
           rocke_archtarget_mma(target), "wmma", ab, ab, "fp32", 16, 16, FSW_WMMA_K))
    {
        snprintf(msg, sizeof msg, "missing wmma_f32_16x16x32_%s on %s", ab, arch);
        fsw_set_reason(reason, reason_cap, msg);
        return false;
    }
    if(p->groups != 1 || p->dil_h != 1 || p->dil_w != 1)
    {
        fsw_set_reason(reason, reason_cap, "ungrouped, undilated convolutions only");
        return false;
    }
    if(p->KH != p->KW)
    {
        fsw_set_reason(reason, reason_cap, "square filters only");
        return false;
    }
    if(!(0 <= p->PAD && p->PAD <= p->KH - 1))
    {
        snprintf(msg, sizeof msg, "PAD %d outside 0..K-1", p->PAD);
        fsw_set_reason(reason, reason_cap, msg);
        return false;
    }
    {
        long long total_c = (long long)p->groups * p->cpg;
        long long total_k = (long long)p->groups * p->kpg;
        long long bytes[3] = {2LL * p->N * p->H * p->W * total_c,
                              2LL * total_k * p->KH * p->KW * p->cpg,
                              2LL * p->N * fsw_ho(p) * fsw_wo(p) * total_k};
        long long mx = bytes[0];
        if(bytes[1] > mx)
            mx = bytes[1];
        if(bytes[2] > mx)
            mx = bytes[2];
        if(mx > INT32_MAX)
        {
            snprintf(msg, sizeof msg, "tensors of %lld B exceed the i32 byte offsets", mx);
            fsw_set_reason(reason, reason_cap, msg);
            return false;
        }
    }
    return true;
}

/* ---- FwdSmallcWmmaSpec @property helpers the builder shares. */
int sc_run(const rocke_fwd_smallc_wmma_spec_t* s)
{
    return fsw_ceil_div((long long)s->problem.KW * s->problem.cpg, 2);
}
int sc_steps(const rocke_fwd_smallc_wmma_spec_t* s)
{
    return fsw_ceil_div((long long)s->problem.KH * sc_run(s), 16);
}
int sc_win_h(const rocke_fwd_smallc_wmma_spec_t* s)
{
    return (s->out_h - 1) * s->problem.stride + s->problem.KH;
}
int sc_win_w(const rocke_fwd_smallc_wmma_spec_t* s)
{
    return (s->out_w - 1) * s->problem.stride + s->problem.KW + 1;
}
int sc_row_pitch(const rocke_fwd_smallc_wmma_spec_t* s)
{
    return fsw_round_up((long long)sc_win_w(s) * s->problem.cpg, 8) + 8;
}
long sc_buf_elems(const rocke_fwd_smallc_wmma_spec_t* s)
{
    return (long)(sc_win_h(s) + s->problem.PAD + 1) * sc_row_pitch(s);
}
int sc_nbuf(const rocke_fwd_smallc_wmma_spec_t* s)
{
    return s->tiles_per_block > 1 ? 2 : 1;
}
long sc_out_elems(const rocke_fwd_smallc_wmma_spec_t* s)
{
    return s->tdm_store ? (long)rocke_fwd_smallc_wmma_block_p(s) * rocke_fwd_smallc_wmma_block_k(s)
                        : 0;
}

/* ---- FwdSmallkWmmaSpec @property helpers. */
int sk_ncols(const rocke_fwd_smallk_wmma_spec_t* s)
{
    return s->problem.KH * s->problem.KW * s->problem.kpg;
}
int sk_win_h(const rocke_fwd_smallk_wmma_spec_t* s)
{
    return (s->out_h - 1) * s->problem.stride + s->problem.KH;
}
int sk_win_w(const rocke_fwd_smallk_wmma_spec_t* s)
{
    return (s->out_w - 1) * s->problem.stride + s->problem.KW;
}
int sk_x_pitch(const rocke_fwd_smallk_wmma_spec_t* s)
{
    return s->ck + FSW_CH_PAD;
}
int sk_w_pitch(const rocke_fwd_smallk_wmma_spec_t* s)
{
    return s->ck + FSW_CH_PAD;
}
int sk_z_pitch(const rocke_fwd_smallk_wmma_spec_t* s)
{
    return rocke_fwd_smallk_wmma_block_c(s) + 8;
}
long sk_x_elems(const rocke_fwd_smallk_wmma_spec_t* s)
{
    return (long)(rocke_fwd_smallk_wmma_block_p(s) + s->problem.PAD + 1) * sk_x_pitch(s);
}
long sk_wei_elems(const rocke_fwd_smallk_wmma_spec_t* s)
{
    return (long)rocke_fwd_smallk_wmma_block_c(s) * sk_w_pitch(s);
}

} // namespace

/* ===================================================================== *
 *  FwdSmallcWmmaSpec: defaults + derived geometry
 * ===================================================================== */

rocke_fwd_smallc_wmma_spec_t rocke_fwd_smallc_wmma_spec_default(void)
{
    rocke_fwd_smallc_wmma_spec_t s;
    memset(&s, 0, sizeof(s));
    s.problem = rocke_direct_conv_problem_default();
    s.name = "fwd_smallc_wmma";
    s.waves_k = 1;
    s.waves_p = 4;
    s.tiles_k = 2;
    s.tiles_p = 4;
    s.out_h = 8;
    s.out_w = 32;
    s.tiles_per_block = 1;
    s.tdm_store = false;
    s.sched = NULL;
    s.wave_size = 32;
    return s;
}

int rocke_fwd_smallc_wmma_block_k(const rocke_fwd_smallc_wmma_spec_t* spec)
{
    return spec->waves_k * spec->tiles_k * 16;
}

int rocke_fwd_smallc_wmma_block_p(const rocke_fwd_smallc_wmma_spec_t* spec)
{
    return spec->waves_p * spec->tiles_p * 16;
}

int rocke_fwd_smallc_wmma_threads_per_block(const rocke_fwd_smallc_wmma_spec_t* spec)
{
    return spec->waves_k * spec->waves_p * spec->wave_size;
}

long rocke_fwd_smallc_wmma_lds_bytes(const rocke_fwd_smallc_wmma_spec_t* spec)
{
    return 2 * ((long)sc_nbuf(spec) * sc_buf_elems(spec) + sc_out_elems(spec));
}

void rocke_fwd_smallc_wmma_grid(const rocke_fwd_smallc_wmma_spec_t* spec, int grid[3])
{
    const rocke_direct_conv_problem_t* p = &spec->problem;
    long long n_tiles = (long long)p->N * fsw_ceil_div(fsw_ho(p), spec->out_h)
                        * fsw_ceil_div(fsw_wo(p), spec->out_w);
    grid[0] = fsw_ceil_div(n_tiles, spec->tiles_per_block);
    grid[1] = fsw_ceil_div((long long)p->groups * p->kpg, rocke_fwd_smallc_wmma_block_k(spec));
    grid[2] = 1;
}

rocke_status_t rocke_fwd_smallc_wmma_kernel_name(const rocke_fwd_smallc_wmma_spec_t* spec,
                                                 char* out,
                                                 size_t out_cap)
{
    char f[32], st[32], pad[32], cc[32], wk[32], wp[32], tk[32], tp[32], oh[32], ow[32], tb[32];
    std::string sched;
    const rocke_direct_conv_problem_t* p = &spec->problem;
    snprintf(f, sizeof f, "f%dx%d", p->KH, p->KW);
    snprintf(st, sizeof st, "s%d", p->stride);
    snprintf(pad, sizeof pad, "p%d", p->PAD);
    snprintf(cc, sizeof cc, "c%d", p->cpg);
    snprintf(wk, sizeof wk, "wk%d", spec->waves_k);
    snprintf(wp, sizeof wp, "wp%d", spec->waves_p);
    snprintf(tk, sizeof tk, "tk%d", spec->tiles_k);
    snprintf(tp, sizeof tp, "tp%d", spec->tiles_p);
    snprintf(oh, sizeof oh, "oh%d", spec->out_h);
    snprintf(ow, sizeof ow, "ow%d", spec->out_w);
    snprintf(tb, sizeof tb, "tb%d", spec->tiles_per_block);
    if(spec->sched != NULL && spec->sched[0] != '\0')
        sched = std::string("s") + spec->sched;
    const char* parts[]
        = {f, st, pad, cc, wk, wp, tk, tp, oh, ow, tb, spec->tdm_store ? "ts" : "", sched.c_str()};
    const char* flag_names[] = {"bf16"};
    int flag_on[] = {fsw_is_bf16(p) ? 1 : 0};
    return rocke_kernel_name_join(spec->name ? spec->name : "fwd_smallc_wmma",
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
 *  fwd_smallc_reason / is_valid_fwd_smallc_spec
 * ===================================================================== */

bool rocke_fwd_smallc_wmma_is_valid_spec(const rocke_fwd_smallc_wmma_spec_t* spec,
                                         const char* arch,
                                         char* reason,
                                         size_t reason_cap)
{
    char msg[512];
    const rocke_archtarget_t* target;
    const rocke_direct_conv_problem_t* p;
    int vgprs;

    if(spec == NULL)
    {
        fsw_set_reason(reason, reason_cap, "null spec");
        return false;
    }
    if(arch == NULL)
        arch = "gfx1250";
    p = &spec->problem;
    if(!fsw_common_reason(p, arch, "fwd_smallc_wmma", reason, reason_cap))
        return false;
    target = rocke_archtarget_from_gfx(arch);
    if(spec->wave_size != 32)
    {
        fsw_set_reason(reason, reason_cap, "fwd_smallc_wmma is a wave32 kernel");
        return false;
    }
    if(p->cpg > FSW_SMALLC_MAX_C)
    {
        snprintf(msg, sizeof msg, "C %d > %d", p->cpg, FSW_SMALLC_MAX_C);
        fsw_set_reason(reason, reason_cap, msg);
        return false;
    }
    if(p->kpg % 8 != 0)
    {
        snprintf(msg, sizeof msg, "K %d must be a multiple of 8 (16-byte output rows)", p->kpg);
        fsw_set_reason(reason, reason_cap, msg);
        return false;
    }
    if(spec->waves_k < 1 || spec->waves_p < 1 || spec->tiles_k < 1 || spec->tiles_p < 1)
    {
        fsw_set_reason(reason, reason_cap, "waves and tiles must be >= 1");
        return false;
    }
    if(spec->tiles_per_block < 1)
    {
        fsw_set_reason(reason, reason_cap, "tiles_per_block must be >= 1");
        return false;
    }
    if(spec->out_h * spec->out_w != rocke_fwd_smallc_wmma_block_p(spec))
    {
        snprintf(msg,
                 sizeof msg,
                 "out tile %dx%d must hold exactly the %d GEMM pixels",
                 spec->out_h,
                 spec->out_w,
                 rocke_fwd_smallc_wmma_block_p(spec));
        fsw_set_reason(reason, reason_cap, msg);
        return false;
    }
    if(sc_row_pitch(spec) > 0xFFFF || sc_win_h(spec) > 0xFFFF)
    {
        fsw_set_reason(reason, reason_cap, "window too large for one TDM tile");
        return false;
    }
    if(rocke_fwd_smallc_wmma_threads_per_block(spec)
       > rocke_archtarget_max_threads_per_block(target))
    {
        snprintf(
            msg, sizeof msg, "%d threads per block", rocke_fwd_smallc_wmma_threads_per_block(spec));
        fsw_set_reason(reason, reason_cap, msg);
        return false;
    }
    if(!rocke_archtarget_fits_lds(target, rocke_fwd_smallc_wmma_lds_bytes(spec)))
    {
        snprintf(msg,
                 sizeof msg,
                 "LDS %ld B > %d",
                 rocke_fwd_smallc_wmma_lds_bytes(spec),
                 target->lds_capacity_bytes);
        fsw_set_reason(reason, reason_cap, msg);
        return false;
    }
    vgprs = 8 * spec->tiles_k * (spec->tiles_p + sc_steps(spec)) + 8 * spec->tiles_p;
    if(vgprs > FSW_VGPR_BUDGET)
    {
        snprintf(msg, sizeof msg, "~%d VGPRs of accumulators and fragments", vgprs);
        fsw_set_reason(reason, reason_cap, msg);
        return false;
    }
    if(spec->sched != NULL && !rocke_scheduler_strategy_is_valid(spec->sched))
    {
        snprintf(msg, sizeof msg, "unsupported scheduler_strategy '%s'", spec->sched);
        fsw_set_reason(reason, reason_cap, msg);
        return false;
    }
    fsw_set_reason(reason, reason_cap, "ok");
    return true;
}

/* ===================================================================== *
 *  FwdSmallkWmmaSpec: defaults + derived geometry
 * ===================================================================== */

rocke_fwd_smallk_wmma_spec_t rocke_fwd_smallk_wmma_spec_default(void)
{
    rocke_fwd_smallk_wmma_spec_t s;
    memset(&s, 0, sizeof(s));
    s.problem = rocke_direct_conv_problem_default();
    s.name = "fwd_smallk_wmma";
    s.waves_c = 1;
    s.waves_p = 4;
    s.tiles_c = 3;
    s.tiles_p = 6;
    s.out_h = 16;
    s.out_w = 16;
    s.ck = 32;
    s.stages = 2;
    s.prefetch = false;
    s.sched = NULL;
    s.wave_size = 32;
    return s;
}

int rocke_fwd_smallk_wmma_block_c(const rocke_fwd_smallk_wmma_spec_t* spec)
{
    return spec->waves_c * spec->tiles_c * 16;
}

int rocke_fwd_smallk_wmma_block_p(const rocke_fwd_smallk_wmma_spec_t* spec)
{
    return spec->waves_p * spec->tiles_p * 16;
}

int rocke_fwd_smallk_wmma_threads_per_block(const rocke_fwd_smallk_wmma_spec_t* spec)
{
    return spec->waves_c * spec->waves_p * spec->wave_size;
}

long rocke_fwd_smallk_wmma_lds_bytes(const rocke_fwd_smallk_wmma_spec_t* spec)
{
    long ring = 2L * spec->stages * (sk_x_elems(spec) + sk_wei_elems(spec));
    long z = 4L * rocke_fwd_smallk_wmma_block_p(spec) * sk_z_pitch(spec);
    return ring > z ? ring : z;
}

void rocke_fwd_smallk_wmma_grid(const rocke_fwd_smallk_wmma_spec_t* spec, int grid[3])
{
    const rocke_direct_conv_problem_t* p = &spec->problem;
    grid[0] = (int)((long long)p->N * fsw_ceil_div(fsw_ho(p), spec->out_h)
                    * fsw_ceil_div(fsw_wo(p), spec->out_w));
    grid[1] = 1;
    grid[2] = 1;
}

rocke_status_t rocke_fwd_smallk_wmma_kernel_name(const rocke_fwd_smallk_wmma_spec_t* spec,
                                                 char* out,
                                                 size_t out_cap)
{
    char f[32], st[32], pad[32], kk[32], wc[32], wp[32], tc[32], tp[32], oh[32], ow[32], ck[32],
        sg[32];
    std::string sched;
    const rocke_direct_conv_problem_t* p = &spec->problem;
    snprintf(f, sizeof f, "f%dx%d", p->KH, p->KW);
    snprintf(st, sizeof st, "s%d", p->stride);
    snprintf(pad, sizeof pad, "p%d", p->PAD);
    snprintf(kk, sizeof kk, "k%d", p->kpg);
    snprintf(wc, sizeof wc, "wc%d", spec->waves_c);
    snprintf(wp, sizeof wp, "wp%d", spec->waves_p);
    snprintf(tc, sizeof tc, "tc%d", spec->tiles_c);
    snprintf(tp, sizeof tp, "tp%d", spec->tiles_p);
    snprintf(oh, sizeof oh, "oh%d", spec->out_h);
    snprintf(ow, sizeof ow, "ow%d", spec->out_w);
    snprintf(ck, sizeof ck, "ck%d", spec->ck);
    snprintf(sg, sizeof sg, "st%d", spec->stages);
    if(spec->sched != NULL && spec->sched[0] != '\0')
        sched = std::string("s") + spec->sched;
    const char* parts[] = {
        f, st, pad, kk, wc, wp, tc, tp, oh, ow, ck, sg, spec->prefetch ? "pf" : "", sched.c_str()};
    const char* flag_names[] = {"bf16"};
    int flag_on[] = {fsw_is_bf16(p) ? 1 : 0};
    return rocke_kernel_name_join(spec->name ? spec->name : "fwd_smallk_wmma",
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
 *  fwd_smallk_reason / is_valid_fwd_smallk_spec
 * ===================================================================== */

bool rocke_fwd_smallk_wmma_is_valid_spec(const rocke_fwd_smallk_wmma_spec_t* spec,
                                         const char* arch,
                                         char* reason,
                                         size_t reason_cap)
{
    char msg[512];
    const rocke_archtarget_t* target;
    const rocke_direct_conv_problem_t* p;
    int vgprs, block_c, ncols;

    if(spec == NULL)
    {
        fsw_set_reason(reason, reason_cap, "null spec");
        return false;
    }
    if(arch == NULL)
        arch = "gfx1250";
    p = &spec->problem;
    if(!fsw_common_reason(p, arch, "fwd_smallk_wmma", reason, reason_cap))
        return false;
    target = rocke_archtarget_from_gfx(arch);
    if(spec->wave_size != 32)
    {
        fsw_set_reason(reason, reason_cap, "fwd_smallk_wmma is a wave32 kernel");
        return false;
    }
    if(p->kpg > FSW_SMALLK_MAX_K)
    {
        snprintf(msg, sizeof msg, "K %d > %d", p->kpg, FSW_SMALLK_MAX_K);
        fsw_set_reason(reason, reason_cap, msg);
        return false;
    }
    if(!(p->kpg == 1 || p->kpg == 2 || p->kpg == 4 || p->kpg == 8))
    {
        snprintf(msg, sizeof msg, "K %d must be 1, 2, 4 or 8 (vector Z reads)", p->kpg);
        fsw_set_reason(reason, reason_cap, msg);
        return false;
    }
    if(p->cpg % 8 != 0)
    {
        snprintf(msg, sizeof msg, "C %d must be a multiple of 8 (16-byte input rows)", p->cpg);
        fsw_set_reason(reason, reason_cap, msg);
        return false;
    }
    if(spec->ck % FSW_WMMA_K != 0 || spec->ck <= 0)
    {
        snprintf(msg, sizeof msg, "ck %d must be a positive multiple of %d", spec->ck, FSW_WMMA_K);
        fsw_set_reason(reason, reason_cap, msg);
        return false;
    }
    if(spec->waves_c < 1 || spec->waves_p < 1 || spec->tiles_c < 1 || spec->tiles_p < 1)
    {
        fsw_set_reason(reason, reason_cap, "waves and tiles must be >= 1");
        return false;
    }
    block_c = rocke_fwd_smallk_wmma_block_c(spec);
    ncols = sk_ncols(spec);
    if(block_c < ncols || block_c - ncols >= 16)
    {
        snprintf(msg, sizeof msg, "block_c %d must be R*S*K = %d rounded up to 16", block_c, ncols);
        fsw_set_reason(reason, reason_cap, msg);
        return false;
    }
    if((long long)sk_win_h(spec) * sk_win_w(spec) > rocke_fwd_smallk_wmma_block_p(spec))
    {
        snprintf(msg,
                 sizeof msg,
                 "input window %dx%d > %d GEMM pixels",
                 sk_win_h(spec),
                 sk_win_w(spec),
                 rocke_fwd_smallk_wmma_block_p(spec));
        fsw_set_reason(reason, reason_cap, msg);
        return false;
    }
    if(spec->stages < 2)
    {
        fsw_set_reason(reason, reason_cap, "stages must be >= 2");
        return false;
    }
    if(spec->waves_c * spec->waves_p < 2)
    {
        fsw_set_reason(reason, reason_cap, "needs at least two waves");
        return false;
    }
    if(rocke_fwd_smallk_wmma_threads_per_block(spec)
       > rocke_archtarget_max_threads_per_block(target))
    {
        snprintf(
            msg, sizeof msg, "%d threads per block", rocke_fwd_smallk_wmma_threads_per_block(spec));
        fsw_set_reason(reason, reason_cap, msg);
        return false;
    }
    if(!rocke_archtarget_fits_lds(target, rocke_fwd_smallk_wmma_lds_bytes(spec)))
    {
        snprintf(msg,
                 sizeof msg,
                 "LDS %ld B > %d",
                 rocke_fwd_smallk_wmma_lds_bytes(spec),
                 target->lds_capacity_bytes);
        fsw_set_reason(reason, reason_cap, msg);
        return false;
    }
    vgprs = 8 * spec->tiles_c * spec->tiles_p
            + 8 * (spec->tiles_c + spec->tiles_p) * (spec->prefetch ? 2 : 1);
    if(vgprs > FSW_VGPR_BUDGET)
    {
        snprintf(msg, sizeof msg, "~%d VGPRs of accumulators and fragments", vgprs);
        fsw_set_reason(reason, reason_cap, msg);
        return false;
    }
    if(spec->sched != NULL && !rocke_scheduler_strategy_is_valid(spec->sched))
    {
        snprintf(msg, sizeof msg, "unsupported scheduler_strategy '%s'", spec->sched);
        fsw_set_reason(reason, reason_cap, msg);
        return false;
    }
    fsw_set_reason(reason, reason_cap, "ok");
    return true;
}

/* ===================================================================== *
 *  Shared builder pieces
 * ===================================================================== */
namespace
{

rocke_value_t* ci(rocke_ir_builder_t* b, int64_t v)
{
    return rocke_b_const_i32(b, v);
}

/* _tdm_descriptor_3d(b, global_ptr, lds_addr, dims, strides, tile0, tile1,
 * tile2). */
void tdm_descriptor_3d(rocke_ir_builder_t* b,
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
    rocke_value_t* c16 = ci(b, 16);
    rocke_value_t* gaddr = rocke_b_global_ptr_to_i64(b, global_ptr);
    rocke_value_t* c32 = rocke_b_const_i64(b, 32);
    rocke_value_t* sh = rocke_b_lshr(b, gaddr, c32);
    rocke_value_t* g_hi = rocke_b_trunc(b, sh, rocke_i32());
    rocke_value_t* g0[4];
    rocke_value_t* g1[8];
    rocke_value_t* g2[4];
    g0[0] = ci(b, 1);
    g0[1] = lds_addr;
    g0[2] = rocke_b_trunc(b, gaddr, rocke_i32());
    {
        rocke_value_t* m = ci(b, 0x1FFFFFF);
        rocke_value_t* lo = rocke_b_land(b, g_hi, m);
        rocke_value_t* ty = ci(b, -((int64_t)1 << 31));
        g0[3] = rocke_b_lor(b, lo, ty);
    }
    g1[0] = ci(b, 1 << 16);
    g1[1] = rocke_b_shl(b, d0, c16);
    {
        rocke_value_t* x = rocke_b_lshr(b, d0, c16);
        rocke_value_t* y = rocke_b_shl(b, d1, c16);
        g1[2] = rocke_b_lor(b, x, y);
    }
    {
        rocke_value_t* x = rocke_b_lshr(b, d1, c16);
        rocke_value_t* y = ci(b, (int64_t)tile0 << 16);
        g1[3] = rocke_b_lor(b, x, y);
    }
    {
        rocke_value_t* x = ci(b, tile1);
        rocke_value_t* y = rocke_b_shl(b, tile2, c16);
        g1[4] = rocke_b_lor(b, x, y);
    }
    g1[5] = s1;
    g1[6] = rocke_b_shl(b, s2, c16);
    g1[7] = rocke_b_lshr(b, s2, c16);
    g2[0] = d2;
    g2[1] = ci(b, 0);
    g2[2] = ci(b, 0);
    g2[3] = ci(b, 0);
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

/* (wave * T + t) * 16 + extra, in Python order: mul(add(mul(wave, T), t), 16) +
 * extra. */
rocke_value_t* tile_lane(rocke_ir_builder_t* b,
                         rocke_value_t* wave,
                         int T,
                         int t,
                         rocke_value_t* c16,
                         rocke_value_t* extra)
{
    rocke_value_t* x = ci(b, T);
    rocke_value_t* y = rocke_b_mul(b, wave, x);
    rocke_value_t* z = ci(b, t);
    rocke_value_t* u = rocke_b_add(b, y, z);
    rocke_value_t* v = rocke_b_mul(b, u, c16);
    return rocke_b_add(b, v, extra);
}

/* div(add(x, T - 1), T): the ceil-div tile count of a runtime extent. */
rocke_value_t* n_tiles_of(rocke_ir_builder_t* b, rocke_value_t* x, int T)
{
    rocke_value_t* m = ci(b, T - 1);
    rocke_value_t* a = rocke_b_add(b, x, m);
    rocke_value_t* t = ci(b, T);
    return rocke_b_div(b, a, t);
}

/* add(add(mul(n, sn), mul(h, sh)), mul(w, sw)). */
rocke_value_t* nhw_off(rocke_ir_builder_t* b,
                       rocke_value_t* n,
                       rocke_value_t* sn,
                       rocke_value_t* h,
                       rocke_value_t* shh,
                       rocke_value_t* w,
                       rocke_value_t* sw)
{
    rocke_value_t* x = rocke_b_mul(b, n, sn);
    rocke_value_t* y = rocke_b_mul(b, h, shh);
    rocke_value_t* z = rocke_b_add(b, x, y);
    rocke_value_t* u = rocke_b_mul(b, w, sw);
    return rocke_b_add(b, z, u);
}

void scf_if_enter(rocke_ir_builder_t* b, rocke_value_t* cond)
{
    rocke_if_t gate = rocke_b_scf_if(b, cond);
    rocke_b_region_enter(b, gate.then_region);
}

const char* wmma_op_id(rocke_ir_builder_t* b,
                       const rocke_direct_conv_problem_t* p,
                       const char* arch,
                       const char* who)
{
    const char* ab = fsw_is_bf16(p) ? "bf16" : "f16";
    const rocke_archtarget_t* target = rocke_archtarget_from_gfx(arch);
    const rocke_mmaop_t* op
        = rocke_archtarget_op_for_shape(target, "wmma", ab, ab, "fp32", 16, 16, FSW_WMMA_K, NULL);
    if(op == NULL || op->op_id == NULL)
    {
        std::string msg = std::string(who) + ": no WMMA atom";
        rocke_i_set_err_msg(b, ROCKE_ERR_VALUE, msg.c_str());
        return NULL;
    }
    return op->op_id;
}

rocke_value_t* cvt_out(rocke_ir_builder_t* b, bool bf16, rocke_value_t* v)
{
    return bf16 ? rocke_b_vec_trunc_f32_to_bf16(b, v) : rocke_b_vec_trunc_f32_to_f16(b, v);
}

/* ===================================================================== *
 *  build_fwd_smallc_wmma
 * ===================================================================== */

struct sc_ctx
{
    rocke_ir_builder_t* b;
    const rocke_fwd_smallc_wmma_spec_t* spec;
    rocke_dconv_params_t params;
    const rocke_type_t* io_type;
    const char* wmma_op;
    bool bf16;

    int KH, KW, S, PAD, C, RUN, E0, STEPS;
    int WK, TM, TN, THREADS, BK, TOH, TOW, WIN_H, WP, NB, T;
    long BUF, OUT0;

    rocke_value_t *c0, *c1, *c2, *c8, *c16;
    rocke_value_t *c_N, *c_K, *c_H, *c_W, *c_Ho, *c_Wo, *c_C;
    rocke_value_t *tid, *wave_id, *lane, *l16, *lhalf, *wave_k, *wave_p;
    rocke_value_t *n_ht, *n_wt, *n_tiles, *k_blk, *lds, *lds_addr, *zero_e;

    std::vector<std::vector<rocke_value_t*>> a_frags;
    std::vector<rocke_value_t*> lane_ty, lane_tx, lane_off;
};

struct sc_geom
{
    rocke_value_t *n, *oh0, *ow0, *h_start, *w_start, *hs, *sw, *land, *o0;
};

sc_geom sc_tile_geom(sc_ctx& c, rocke_value_t* tile)
{
    rocke_ir_builder_t* b = c.b;
    sc_geom g;
    rocke_value_t* tw = rocke_b_mod(b, tile, c.n_wt);
    rocke_value_t* rest = rocke_b_div(b, tile, c.n_wt);
    rocke_value_t* th = rocke_b_mod(b, rest, c.n_ht);
    g.n = rocke_b_div(b, rest, c.n_ht);
    {
        rocke_value_t* x = ci(b, c.TOH);
        g.oh0 = rocke_b_mul(b, th, x);
    }
    {
        rocke_value_t* x = ci(b, c.TOW);
        g.ow0 = rocke_b_mul(b, tw, x);
    }
    {
        rocke_value_t* x = ci(b, c.S);
        rocke_value_t* y = rocke_b_mul(b, g.oh0, x);
        rocke_value_t* z = ci(b, c.PAD);
        g.h_start = rocke_b_sub(b, y, z);
    }
    {
        rocke_value_t* x = ci(b, c.S);
        rocke_value_t* y = rocke_b_mul(b, g.ow0, x);
        rocke_value_t* z = ci(b, c.PAD);
        g.w_start = rocke_b_sub(b, y, z);
    }
    {
        rocke_value_t* x = rocke_b_sub(b, c.c0, g.h_start);
        g.hs = rocke_b_smax(b, x, c.c0);
    }
    {
        rocke_value_t* x = rocke_b_sub(b, c.c0, g.w_start);
        g.sw = rocke_b_smax(b, x, c.c0);
    }
    rocke_value_t* swc = rocke_b_mul(b, g.sw, c.c_C);
    {
        rocke_value_t* x = ci(b, 7);
        rocke_value_t* y = rocke_b_add(b, swc, x);
        rocke_value_t* z = rocke_b_div(b, y, c.c8);
        g.land = rocke_b_mul(b, z, c.c8);
    }
    g.o0 = rocke_b_sub(b, g.land, swc);
    return g;
}

/* _stage(tile, buf): zero the padding prefix, then TDM the window into buf. */
void sc_stage(sc_ctx& c, rocke_value_t* tile, int buf)
{
    rocke_ir_builder_t* b = c.b;
    const rocke_dconv_params_t* params = &c.params;
    sc_geom g = sc_tile_geom(c, tile);
    long base = (long)buf * c.BUF;
    rocke_value_t* prefix;
    {
        rocke_value_t* x = ci(b, c.WP);
        rocke_value_t* y = rocke_b_mul(b, g.hs, x);
        prefix = rocke_b_add(b, y, g.land);
    }
    rocke_value_t* zero_v = rocke_b_zero_vec(b, c.io_type, 8);
    int max_prefix = c.PAD * c.WP + fsw_round_up((long long)c.PAD * c.C, 8);
    if(max_prefix)
    {
        rocke_value_t* nvec = rocke_b_div(b, prefix, c.c8);
        for(int j = 0; j < fsw_ceil_div(max_prefix, 8); j += c.THREADS)
        {
            rocke_value_t* cj = ci(b, j);
            rocke_value_t* v = rocke_b_add(b, c.tid, cj);
            rocke_value_t* cond = rocke_b_cmp_lt(b, v, nvec);
            scf_if_enter(b, cond);
            {
                rocke_value_t* x = rocke_b_mul(b, v, c.c8);
                rocke_value_t* y = ci(b, base);
                rocke_value_t* z = rocke_b_add(b, x, y);
                rocke_value_t* idx[2] = {c.c0, z};
                rocke_b_smem_store_vN(b, c.lds, idx, 2, zero_v, 8);
            }
            rocke_b_region_leave(b);
        }
    }
    rocke_value_t* hr0 = rocke_b_add(b, g.h_start, g.hs);
    rocke_value_t* wc0 = rocke_b_add(b, g.w_start, g.sw);
    {
        rocke_value_t* cond = rocke_b_cmp_eq(b, c.wave_id, c.c0);
        scf_if_enter(b, cond);
        rocke_value_t* off = nhw_off(
            b, g.n, params->p_A_stride_n, hr0, params->p_A_stride_hi, wc0, params->p_A_stride_wi);
        rocke_value_t* src;
        {
            rocke_value_t* x = rocke_b_mul(b, off, c.c2);
            src = rocke_b_global_ptr_add(b, params->A, x);
        }
        rocke_value_t* dst;
        {
            rocke_value_t* x = ci(b, base);
            rocke_value_t* y = rocke_b_add(b, prefix, x);
            rocke_value_t* z = rocke_b_mul(b, y, c.c2);
            dst = rocke_b_add(b, c.lds_addr, z);
        }
        rocke_value_t* d0;
        {
            rocke_value_t* x = rocke_b_sub(b, c.c_W, wc0);
            rocke_value_t* y = rocke_b_mul(b, x, c.c_C);
            rocke_value_t* z = ci(b, c.WP);
            rocke_value_t* u = rocke_b_sub(b, z, g.land);
            rocke_value_t* v = rocke_b_smin(b, y, u);
            d0 = rocke_b_smax(b, v, c.c0);
        }
        rocke_value_t* d1;
        {
            rocke_value_t* x = rocke_b_sub(b, c.c_H, hr0);
            rocke_value_t* y = ci(b, c.WIN_H);
            rocke_value_t* z = rocke_b_sub(b, y, g.hs);
            rocke_value_t* u = rocke_b_smin(b, x, z);
            d1 = rocke_b_smax(b, u, c.c0);
        }
        rocke_value_t* d[5];
        tdm_descriptor_3d(b,
                          src,
                          dst,
                          d0,
                          d1,
                          c.c1,
                          params->p_A_stride_hi,
                          params->p_A_stride_n,
                          c.WP,
                          c.WIN_H,
                          c.c1,
                          d);
        rocke_b_tensor_load_to_lds(b, d[0], d[1], d[2], d[3], d[4], 0);
        rocke_b_region_leave(b);
    }
}

void sc_compute(sc_ctx& c, rocke_value_t* tile, int buf)
{
    rocke_ir_builder_t* b = c.b;
    const rocke_dconv_params_t* params = &c.params;
    sc_geom g = sc_tile_geom(c, tile);
    rocke_value_t* base;
    {
        rocke_value_t* x = ci(b, (int64_t)buf * c.BUF);
        base = rocke_b_add(b, g.o0, x);
    }
    std::vector<rocke_value_t*> accs(c.TM * c.TN);
    for(int i = 0; i < c.TM * c.TN; ++i)
        accs[i] = rocke_b_zero_vec_f32(b, 8);
    for(int tn = 0; tn < c.TN; ++tn)
    {
        rocke_value_t* lb = rocke_b_add(b, base, c.lane_off[tn]);
        for(int t = 0; t < c.STEPS; ++t)
        {
            rocke_value_t* elems[16];
            for(int i = 0; i < 16; ++i)
            {
                int gs = t * 16 + i;
                if(gs >= c.KH * c.E0)
                {
                    elems[i] = c.zero_e;
                    continue;
                }
                int r = gs / c.E0;
                int e = gs % c.E0;
                if(e >= c.RUN)
                {
                    elems[i] = c.zero_e;
                    continue;
                }
                rocke_value_t* x = ci(b, (int64_t)r * c.WP + e);
                rocke_value_t* y = rocke_b_add(b, lb, x);
                rocke_value_t* idx[2] = {c.c0, y};
                rocke_value_t* v = rocke_b_smem_load_vN(b, c.lds, idx, 2, c.io_type, 1);
                elems[i] = rocke_b_vec_extract(b, v, 0);
            }
            rocke_value_t* bf = rocke_b_vec_pack(b, elems, 16, c.io_type);
            for(int m = 0; m < c.TM; ++m)
            {
                int i = m * c.TN + tn;
                accs[i] = rocke_b_mma(b, c.wmma_op, c.a_frags[m][t], bf, accs[i], NULL, 0);
            }
        }
    }
    if(c.spec->tdm_store)
    {
        for(int tn = 0; tn < c.TN; ++tn)
        {
            rocke_value_t* q = tile_lane(b, c.wave_p, c.TN, tn, c.c16, c.l16);
            for(int m = 0; m < c.TM; ++m)
            {
                rocke_value_t* kl;
                {
                    rocke_value_t* x = ci(b, c.TM);
                    rocke_value_t* y = rocke_b_mul(b, c.wave_k, x);
                    rocke_value_t* z = ci(b, m);
                    rocke_value_t* u = rocke_b_add(b, y, z);
                    rocke_value_t* v = rocke_b_mul(b, u, c.c16);
                    rocke_value_t* w = rocke_b_mul(b, c.lhalf, c.c8);
                    kl = rocke_b_add(b, v, w);
                }
                rocke_value_t* oi;
                {
                    rocke_value_t* x = ci(b, c.BK);
                    rocke_value_t* y = rocke_b_mul(b, q, x);
                    rocke_value_t* z = rocke_b_add(b, y, kl);
                    rocke_value_t* u = ci(b, c.OUT0);
                    oi = rocke_b_add(b, z, u);
                }
                rocke_value_t* idx[2] = {c.c0, oi};
                rocke_value_t* val = cvt_out(b, c.bf16, accs[m * c.TN + tn]);
                rocke_b_smem_store_vN(b, c.lds, idx, 2, val, 8);
            }
        }
        rocke_b_sync_lds_only(b);
        {
            rocke_value_t* cond = rocke_b_cmp_eq(b, c.wave_id, c.c0);
            scf_if_enter(b, cond);
            rocke_value_t* off;
            {
                rocke_value_t* x = nhw_off(b,
                                           g.n,
                                           params->p_D_stride_n,
                                           g.oh0,
                                           params->p_D_stride_ho,
                                           g.ow0,
                                           params->p_D_stride_wo);
                off = rocke_b_add(b, x, c.k_blk);
            }
            rocke_value_t* gptr;
            {
                rocke_value_t* x = rocke_b_mul(b, off, c.c2);
                gptr = rocke_b_global_ptr_add(b, params->D, x);
            }
            rocke_value_t* laddr;
            {
                rocke_value_t* x = ci(b, c.OUT0 * 2);
                laddr = rocke_b_add(b, c.lds_addr, x);
            }
            rocke_value_t *d0, *d1, *d2;
            {
                rocke_value_t* x = rocke_b_sub(b, c.c_K, c.k_blk);
                rocke_value_t* y = ci(b, c.BK);
                rocke_value_t* z = rocke_b_smin(b, x, y);
                d0 = rocke_b_smax(b, z, c.c0);
            }
            {
                rocke_value_t* x = rocke_b_sub(b, c.c_Wo, g.ow0);
                d1 = rocke_b_smax(b, x, c.c0);
            }
            {
                rocke_value_t* x = rocke_b_sub(b, c.c_Ho, g.oh0);
                d2 = rocke_b_smax(b, x, c.c0);
            }
            rocke_value_t* tile2 = ci(b, c.TOH);
            rocke_value_t* d[5];
            tdm_descriptor_3d(b,
                              gptr,
                              laddr,
                              d0,
                              d1,
                              d2,
                              params->p_D_stride_wo,
                              params->p_D_stride_ho,
                              c.BK,
                              c.TOW,
                              tile2,
                              d);
            rocke_b_tensor_store_from_lds(b, d[0], d[1], d[2], d[3], d[4], 0);
            rocke_b_region_leave(b);
        }
        return;
    }
    /* 8 consecutive output channels per lane, 16 B stores. */
    for(int tn = 0; tn < c.TN; ++tn)
    {
        rocke_value_t* oh = rocke_b_add(b, g.oh0, c.lane_ty[tn]);
        rocke_value_t* ow = rocke_b_add(b, g.ow0, c.lane_tx[tn]);
        rocke_value_t* px_ok;
        {
            rocke_value_t* x = rocke_b_cmp_lt(b, oh, c.c_Ho);
            rocke_value_t* y = rocke_b_cmp_lt(b, ow, c.c_Wo);
            px_ok = rocke_b_land(b, x, y);
        }
        rocke_value_t* px_off = nhw_off(
            b, g.n, params->p_D_stride_n, oh, params->p_D_stride_ho, ow, params->p_D_stride_wo);
        for(int m = 0; m < c.TM; ++m)
        {
            rocke_value_t* k;
            {
                rocke_value_t* x = ci(b, c.TM);
                rocke_value_t* y = rocke_b_mul(b, c.wave_k, x);
                rocke_value_t* z = ci(b, m);
                rocke_value_t* u = rocke_b_add(b, y, z);
                rocke_value_t* v = rocke_b_mul(b, u, c.c16);
                rocke_value_t* w = rocke_b_mul(b, c.lhalf, c.c8);
                rocke_value_t* s = rocke_b_add(b, v, w);
                k = rocke_b_add(b, c.k_blk, s);
            }
            rocke_value_t* cond;
            {
                rocke_value_t* x = rocke_b_cmp_lt(b, k, c.c_K);
                cond = rocke_b_land(b, px_ok, x);
            }
            scf_if_enter(b, cond);
            {
                rocke_value_t* idx = rocke_b_add(b, px_off, k);
                rocke_value_t* val = cvt_out(b, c.bf16, accs[m * c.TN + tn]);
                rocke_b_global_store_vN(b, params->D, idx, val, 8, 16);
            }
            rocke_b_region_leave(b);
        }
    }
}

rocke_kernel_def_t*
    build_smallc(rocke_ir_builder_t* b, const rocke_fwd_smallc_wmma_spec_t* spec, const char* arch)
{
    char reason[512];
    sc_ctx c;
    if(!rocke_fwd_smallc_wmma_is_valid_spec(spec, arch, reason, sizeof reason))
    {
        std::string msg = std::string("invalid FwdSmallcWmmaSpec for ") + arch + ": " + reason;
        rocke_i_set_err_msg(b, ROCKE_ERR_VALUE, msg.c_str());
        return NULL;
    }
    const rocke_direct_conv_problem_t* p = &spec->problem;
    c.b = b;
    c.spec = spec;
    c.KH = p->KH;
    c.KW = p->KW;
    c.S = p->stride;
    c.PAD = p->PAD;
    c.C = p->cpg;
    int NC = c.KH * c.KW * c.C;
    c.RUN = c.KW * c.C;
    c.E0 = sc_run(spec);
    c.STEPS = sc_steps(spec);
    c.bf16 = fsw_is_bf16(p);
    c.io_type = rocke_b_io_ir_type(b, fsw_dtype(p));
    if(c.io_type == NULL)
        return NULL;
    c.wmma_op = wmma_op_id(b, p, arch, "build_fwd_smallc_wmma");
    if(c.wmma_op == NULL)
        return NULL;
    int WAVE = spec->wave_size;
    c.WK = spec->waves_k;
    c.TM = spec->tiles_k;
    c.TN = spec->tiles_p;
    c.THREADS = rocke_fwd_smallc_wmma_threads_per_block(spec);
    c.BK = rocke_fwd_smallc_wmma_block_k(spec);
    c.TOH = spec->out_h;
    c.TOW = spec->out_w;
    c.WIN_H = sc_win_h(spec);
    c.WP = sc_row_pitch(spec);
    c.BUF = sc_buf_elems(spec);
    c.NB = sc_nbuf(spec);
    c.T = spec->tiles_per_block;
    c.OUT0 = (long)c.NB * c.BUF;

    rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", c.THREADS);
    if(spec->sched != NULL)
        rocke_attr_set_str(b, &b->kernel->attrs, "scheduler_strategy", spec->sched);
    rocke_dconv_emit_params(b, &c.params, "fwd", c.io_type);
    const rocke_dconv_params_t* params = &c.params;

    c.c0 = ci(b, 0);
    c.c1 = ci(b, 1);
    c.c2 = ci(b, 2);
    c.c8 = ci(b, 8);
    c.c16 = ci(b, 16);
    c.c_N = params->p_N;
    c.c_K = params->p_total_k;
    c.c_H = params->p_Hi;
    c.c_W = params->p_Wi;
    c.c_Ho = params->p_Ho;
    c.c_Wo = params->p_Wo;
    c.c_C = ci(b, c.C);

    c.tid = rocke_b_thread_id_x(b);
    {
        rocke_value_t* c_wave = ci(b, WAVE);
        c.wave_id = rocke_b_div(b, c.tid, c_wave);
        c.lane = rocke_b_mod(b, c.tid, c_wave);
    }
    c.l16 = rocke_b_mod(b, c.lane, c.c16);
    c.lhalf = rocke_b_div(b, c.lane, c.c16);
    {
        rocke_value_t* x = ci(b, c.WK);
        c.wave_k = rocke_b_mod(b, c.wave_id, x);
        rocke_value_t* y = ci(b, c.WK);
        c.wave_p = rocke_b_div(b, c.wave_id, y);
    }

    c.n_ht = n_tiles_of(b, c.c_Ho, c.TOH);
    c.n_wt = n_tiles_of(b, c.c_Wo, c.TOW);
    {
        rocke_value_t* x = rocke_b_mul(b, c.c_N, c.n_ht);
        c.n_tiles = rocke_b_mul(b, x, c.n_wt);
    }
    {
        rocke_value_t* by = rocke_b_block_id_y(b);
        rocke_value_t* x = ci(b, c.BK);
        c.k_blk = rocke_b_mul(b, by, x);
    }

    {
        int shape[2] = {1, (int)(rocke_fwd_smallc_wmma_lds_bytes(spec) / 2)};
        c.lds = rocke_b_smem_alloc(b, c.io_type, shape, 2, "sc_lds");
    }
    {
        rocke_value_t* a = rocke_b_smem_addr_of(b, c.lds);
        c.lds_addr = rocke_b_trunc(b, a, rocke_i32());
    }

    /* ---- weights: A fragments of every (m tile, step), resident. */
    {
        rocke_value_t* z2 = rocke_b_zero_vec(b, c.io_type, 2);
        c.zero_e = rocke_b_vec_extract(b, z2, 0);
    }
    c.a_frags.assign(c.TM, std::vector<rocke_value_t*>());
    for(int m = 0; m < c.TM; ++m)
    {
        rocke_value_t* k;
        {
            rocke_value_t* inner = tile_lane(b, c.wave_k, c.TM, m, c.c16, c.l16);
            k = rocke_b_add(b, c.k_blk, inner);
        }
        rocke_value_t* k_ok = rocke_b_cmp_lt(b, k, c.c_K);
        rocke_value_t* k_row;
        {
            rocke_value_t* x = ci(b, NC);
            k_row = rocke_b_mul(b, k, x);
        }
        for(int t = 0; t < c.STEPS; ++t)
        {
            rocke_value_t* elems[16];
            for(int i = 0; i < 16; ++i)
            {
                int gg = t * 16 + i;
                if(gg >= c.KH * c.E0)
                {
                    elems[i] = c.zero_e;
                    continue;
                }
                int r = gg / c.E0;
                int e = gg % c.E0;
                bool ok0 = e < c.RUN;
                bool ok1 = e + c.E0 < c.RUN;
                if(!ok0 && !ok1)
                {
                    elems[i] = c.zero_e;
                    continue;
                }
                rocke_value_t* j;
                {
                    rocke_value_t* x = ci(b, (int64_t)r * c.RUN + e);
                    rocke_value_t* y = ci(b, c.E0);
                    rocke_value_t* z = rocke_b_mul(b, c.lhalf, y);
                    j = rocke_b_add(b, x, z);
                }
                rocke_value_t* ok = k_ok;
                if(!ok1)
                {
                    rocke_value_t* x = rocke_b_cmp_eq(b, c.lhalf, c.c0);
                    ok = rocke_b_land(b, ok, x);
                }
                rocke_value_t* idx = rocke_b_add(b, k_row, j);
                elems[i]
                    = rocke_b_masked_global_load(b, params->Bp, idx, ok, c.zero_e, c.io_type, 2);
            }
            c.a_frags[m].push_back(rocke_b_vec_pack(b, elems, 16, c.io_type));
        }
    }

    /* ---- per-lane pixel geometry (tile independent). */
    c.lane_ty.assign(c.TN, NULL);
    c.lane_tx.assign(c.TN, NULL);
    for(int tn = 0; tn < c.TN; ++tn)
    {
        rocke_value_t* q = tile_lane(b, c.wave_p, c.TN, tn, c.c16, c.l16);
        rocke_value_t* x = ci(b, c.TOW);
        c.lane_ty[tn] = rocke_b_div(b, q, x);
        rocke_value_t* y = ci(b, c.TOW);
        c.lane_tx[tn] = rocke_b_mod(b, q, y);
    }
    c.lane_off.assign(c.TN, NULL);
    for(int tn = 0; tn < c.TN; ++tn)
    {
        rocke_value_t* x = ci(b, c.S);
        rocke_value_t* y = rocke_b_mul(b, c.lane_ty[tn], x);
        rocke_value_t* z = ci(b, c.WP);
        rocke_value_t* u = rocke_b_mul(b, y, z);
        rocke_value_t* v = ci(b, (int64_t)c.S * c.C);
        rocke_value_t* w = rocke_b_mul(b, c.lane_tx[tn], v);
        rocke_value_t* s = rocke_b_add(b, u, w);
        rocke_value_t* e0 = ci(b, c.E0);
        rocke_value_t* h = rocke_b_mul(b, c.lhalf, e0);
        c.lane_off[tn] = rocke_b_add(b, s, h);
    }

    rocke_value_t* first;
    {
        rocke_value_t* bx = rocke_b_block_id_x(b);
        rocke_value_t* x = ci(b, c.T);
        first = rocke_b_mul(b, bx, x);
    }
    if(c.T == 1)
    {
        sc_stage(c, first, 0);
        rocke_b_s_wait_tensorcnt(b, 0);
        rocke_b_sync_lds_only(b);
        sc_compute(c, first, 0);
        if(spec->tdm_store)
        {
            rocke_value_t* cond = rocke_b_cmp_eq(b, c.wave_id, c.c0);
            scf_if_enter(b, cond);
            rocke_b_s_wait_tensorcnt(b, 0);
            rocke_b_region_leave(b);
        }
        if(!rocke_ir_builder_ok(b))
            return NULL;
        return b->kernel;
    }

    {
        rocke_value_t* cond = rocke_b_cmp_lt(b, first, c.n_tiles);
        scf_if_enter(b, cond);
        sc_stage(c, first, 0);
        rocke_b_region_leave(b);
    }
    for(int t = 0; t < c.T; ++t)
    {
        rocke_value_t* tile;
        {
            rocke_value_t* x = ci(b, t);
            tile = rocke_b_add(b, first, x);
        }
        rocke_b_s_wait_tensorcnt(b, 0);
        rocke_b_sync_lds_only(b);
        if(t + 1 < c.T)
        {
            rocke_value_t* nxt = rocke_b_add(b, tile, c.c1);
            rocke_value_t* cond = rocke_b_cmp_lt(b, nxt, c.n_tiles);
            scf_if_enter(b, cond);
            sc_stage(c, nxt, (t + 1) % c.NB);
            rocke_b_region_leave(b);
        }
        {
            rocke_value_t* cond = rocke_b_cmp_lt(b, tile, c.n_tiles);
            scf_if_enter(b, cond);
            sc_compute(c, tile, t % c.NB);
            rocke_b_region_leave(b);
        }
    }
    if(spec->tdm_store)
    {
        rocke_value_t* cond = rocke_b_cmp_eq(b, c.wave_id, c.c0);
        scf_if_enter(b, cond);
        rocke_b_s_wait_tensorcnt(b, 0);
        rocke_b_region_leave(b);
    }
    if(!rocke_ir_builder_ok(b))
        return NULL;
    return b->kernel;
}

/* ===================================================================== *
 *  build_fwd_smallk_wmma
 * ===================================================================== */

struct sk_ctx
{
    rocke_ir_builder_t* b;
    const rocke_fwd_smallk_wmma_spec_t* spec;
    rocke_dconv_params_t params;
    const rocke_type_t* io_type;
    const char* wmma_op;

    int K, RS, TM, TN, DW, CK, NS, XP, WPITCH;
    long XE, WEI;

    rocke_value_t *c0, *c1, *c2, *c8, *c16, *c_C, *c_CK, *n_chunks;
    rocke_value_t *wave_id, *lds, *lds_addr, *wei_addr;
    rocke_value_t *x_base, *z_elems, *x_w_ext, *x_h_ext, *x_tile2, *c_RSC;
    std::vector<rocke_value_t*> px_off, wei_off;
};

void sk_issue(sk_ctx& c, rocke_value_t* chunk)
{
    rocke_ir_builder_t* b = c.b;
    const rocke_dconv_params_t* params = &c.params;
    rocke_value_t* stage;
    {
        rocke_value_t* x = ci(b, c.NS);
        stage = rocke_b_mod(b, chunk, x);
    }
    rocke_value_t* cc0 = rocke_b_mul(b, chunk, c.c_CK);
    rocke_value_t* c_ext;
    {
        rocke_value_t* x = rocke_b_sub(b, c.c_C, cc0);
        rocke_value_t* y = rocke_b_smin(b, x, c.c_CK);
        c_ext = rocke_b_smax(b, y, c.c0);
    }
    {
        rocke_value_t* cond = rocke_b_cmp_eq(b, c.wave_id, c.c0);
        scf_if_enter(b, cond);
        rocke_value_t* src;
        {
            rocke_value_t* x = rocke_b_add(b, c.x_base, cc0);
            rocke_value_t* y = rocke_b_mul(b, x, c.c2);
            src = rocke_b_global_ptr_add(b, params->A, y);
        }
        rocke_value_t* dst;
        {
            rocke_value_t* x = ci(b, c.XE);
            rocke_value_t* y = rocke_b_mul(b, stage, x);
            rocke_value_t* z = rocke_b_add(b, y, c.z_elems);
            rocke_value_t* u = rocke_b_mul(b, z, c.c2);
            dst = rocke_b_add(b, c.lds_addr, u);
        }
        rocke_value_t* d[5];
        tdm_descriptor_3d(b,
                          src,
                          dst,
                          c_ext,
                          c.x_w_ext,
                          c.x_h_ext,
                          params->p_A_stride_wi,
                          params->p_A_stride_hi,
                          c.XP,
                          c.DW,
                          c.x_tile2,
                          d);
        rocke_b_tensor_load_to_lds(b, d[0], d[1], d[2], d[3], d[4], 0);
        rocke_b_region_leave(b);
    }
    {
        rocke_value_t* cond = rocke_b_cmp_eq(b, c.wave_id, c.c1);
        scf_if_enter(b, cond);
        rocke_value_t* src;
        {
            rocke_value_t* x = rocke_b_mul(b, cc0, c.c2);
            src = rocke_b_global_ptr_add(b, params->Bp, x);
        }
        rocke_value_t* dst;
        {
            rocke_value_t* x = ci(b, c.WEI * 2);
            rocke_value_t* y = rocke_b_mul(b, stage, x);
            dst = rocke_b_add(b, c.wei_addr, y);
        }
        rocke_value_t* dk = ci(b, c.K);
        rocke_value_t* drs = ci(b, c.RS);
        rocke_value_t* t2 = ci(b, c.RS);
        rocke_value_t* d[5];
        tdm_descriptor_3d(b, src, dst, c_ext, dk, drs, c.c_RSC, c.c_C, c.WPITCH, c.K, t2, d);
        rocke_b_tensor_load_to_lds(b, d[0], d[1], d[2], d[3], d[4], 0);
        rocke_b_region_leave(b);
    }
}

rocke_value_t* sk_read(sk_ctx& c, rocke_value_t* base, rocke_value_t* off, int k_off)
{
    rocke_ir_builder_t* b = c.b;
    rocke_value_t* addr;
    {
        rocke_value_t* x = ci(b, k_off);
        rocke_value_t* y = rocke_b_add(b, off, x);
        addr = rocke_b_add(b, base, y);
    }
    rocke_value_t* idx0[2] = {c.c0, addr};
    rocke_value_t* lo = rocke_b_smem_load_vN(b, c.lds, idx0, 2, c.io_type, 8);
    rocke_value_t* a8 = rocke_b_add(b, addr, c.c8);
    rocke_value_t* idx1[2] = {c.c0, a8};
    rocke_value_t* hi = rocke_b_smem_load_vN(b, c.lds, idx1, 2, c.io_type, 8);
    return rocke_b_vec_concat(b, lo, hi);
}

void sk_frags(sk_ctx& c,
              rocke_value_t* xb,
              rocke_value_t* wb,
              int ks,
              std::vector<rocke_value_t*>& af,
              std::vector<rocke_value_t*>& bf)
{
    bf.assign(c.TN, NULL);
    af.assign(c.TM, NULL);
    for(int tn = 0; tn < c.TN; ++tn)
        bf[tn] = sk_read(c, xb, c.px_off[tn], ks * FSW_WMMA_K);
    for(int m = 0; m < c.TM; ++m)
        af[m] = sk_read(c, wb, c.wei_off[m], ks * FSW_WMMA_K);
}

void sk_chunk(sk_ctx& c, rocke_value_t* kc, std::vector<rocke_value_t*>& accs)
{
    rocke_ir_builder_t* b = c.b;
    if(c.NS == 2)
    {
        rocke_b_s_wait_tensorcnt(b, 0);
    }
    else
    {
        rocke_value_t* x = ci(b, c.NS - 2);
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
        rocke_value_t* x = ci(b, c.NS - 1);
        rocke_value_t* nxt = rocke_b_add(b, kc, x);
        rocke_value_t* cond = rocke_b_cmp_lt(b, nxt, c.n_chunks);
        scf_if_enter(b, cond);
        sk_issue(c, nxt);
        rocke_b_region_leave(b);
    }
    rocke_value_t *stage, *xb, *wb;
    {
        rocke_value_t* x = ci(b, c.NS);
        stage = rocke_b_mod(b, kc, x);
    }
    {
        rocke_value_t* x = ci(b, c.XE);
        xb = rocke_b_mul(b, stage, x);
    }
    {
        rocke_value_t* x = ci(b, c.WEI);
        wb = rocke_b_mul(b, stage, x);
    }
    int nsteps = c.CK / FSW_WMMA_K;
    rocke_b_s_setprio(b, 1);
    std::vector<rocke_value_t*> af, bf, naf, nbf;
    if(c.spec->prefetch)
        sk_frags(c, xb, wb, 0, naf, nbf);
    for(int j = 0; j < nsteps; ++j)
    {
        if(c.spec->prefetch)
        {
            af = naf;
            bf = nbf;
            if(j + 1 < nsteps)
                sk_frags(c, xb, wb, j + 1, naf, nbf);
        }
        else
        {
            sk_frags(c, xb, wb, j, af, bf);
        }
        for(int m = 0; m < c.TM; ++m)
            for(int tn = 0; tn < c.TN; ++tn)
            {
                int i = m * c.TN + tn;
                accs[i] = rocke_b_mma(b, c.wmma_op, af[m], bf[tn], accs[i], NULL, 0);
            }
    }
    rocke_b_s_setprio(b, 0);
}

rocke_kernel_def_t*
    build_smallk(rocke_ir_builder_t* b, const rocke_fwd_smallk_wmma_spec_t* spec, const char* arch)
{
    char reason[512];
    sk_ctx c;
    if(!rocke_fwd_smallk_wmma_is_valid_spec(spec, arch, reason, sizeof reason))
    {
        std::string msg = std::string("invalid FwdSmallkWmmaSpec for ") + arch + ": " + reason;
        rocke_i_set_err_msg(b, ROCKE_ERR_VALUE, msg.c_str());
        return NULL;
    }
    const rocke_direct_conv_problem_t* p = &spec->problem;
    int KH = p->KH, KW = p->KW, S = p->stride, PAD = p->PAD;
    c.b = b;
    c.spec = spec;
    c.K = p->kpg;
    c.RS = KH * KW;
    bool bf16 = fsw_is_bf16(p);
    c.io_type = rocke_b_io_ir_type(b, fsw_dtype(p));
    if(c.io_type == NULL)
        return NULL;
    c.wmma_op = wmma_op_id(b, p, arch, "build_fwd_smallk_wmma");
    if(c.wmma_op == NULL)
        return NULL;
    int WAVE = spec->wave_size;
    int WC = spec->waves_c;
    c.TM = spec->tiles_c;
    c.TN = spec->tiles_p;
    int THREADS = rocke_fwd_smallk_wmma_threads_per_block(spec);
    int OH = spec->out_h, OW = spec->out_w;
    int DH = sk_win_h(spec);
    c.DW = sk_win_w(spec);
    c.CK = spec->ck;
    c.NS = spec->stages;
    c.XP = sk_x_pitch(spec);
    c.WPITCH = sk_w_pitch(spec);
    int ZP = sk_z_pitch(spec);
    c.XE = sk_x_elems(spec);
    c.WEI = sk_wei_elems(spec);

    rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", THREADS);
    if(spec->sched != NULL)
        rocke_attr_set_str(b, &b->kernel->attrs, "scheduler_strategy", spec->sched);
    rocke_dconv_emit_params(b, &c.params, "fwd", c.io_type);
    const rocke_dconv_params_t* params = &c.params;

    c.c0 = ci(b, 0);
    c.c1 = ci(b, 1);
    c.c2 = ci(b, 2);
    c.c8 = ci(b, 8);
    c.c16 = ci(b, 16);
    c.c_C = params->p_total_c;
    rocke_value_t* c_H = params->p_Hi;
    rocke_value_t* c_W = params->p_Wi;
    rocke_value_t* c_Ho = params->p_Ho;
    rocke_value_t* c_Wo = params->p_Wo;

    rocke_value_t* tid = rocke_b_thread_id_x(b);
    rocke_value_t *lane, *l16, *lhalf, *wave_c, *wave_p;
    {
        rocke_value_t* c_wave = ci(b, WAVE);
        c.wave_id = rocke_b_div(b, tid, c_wave);
        lane = rocke_b_mod(b, tid, c_wave);
    }
    l16 = rocke_b_mod(b, lane, c.c16);
    lhalf = rocke_b_div(b, lane, c.c16);
    {
        rocke_value_t* x = ci(b, WC);
        wave_c = rocke_b_mod(b, c.wave_id, x);
        rocke_value_t* y = ci(b, WC);
        wave_p = rocke_b_div(b, c.wave_id, y);
    }

    rocke_value_t* n_ht = n_tiles_of(b, c_Ho, OH);
    rocke_value_t* n_wt = n_tiles_of(b, c_Wo, OW);
    rocke_value_t* t = rocke_b_block_id_x(b);
    rocke_value_t* wt = rocke_b_mod(b, t, n_wt);
    t = rocke_b_div(b, t, n_wt);
    rocke_value_t* ht = rocke_b_mod(b, t, n_ht);
    rocke_value_t* n_img = rocke_b_div(b, t, n_ht);
    rocke_value_t *oh0, *ow0;
    {
        rocke_value_t* x = ci(b, OH);
        oh0 = rocke_b_mul(b, ht, x);
    }
    {
        rocke_value_t* x = ci(b, OW);
        ow0 = rocke_b_mul(b, wt, x);
    }
    c.c_CK = ci(b, c.CK);
    {
        rocke_value_t* x = ci(b, c.CK - 1);
        rocke_value_t* y = rocke_b_add(b, c.c_C, x);
        c.n_chunks = rocke_b_div(b, y, c.c_CK);
    }

    rocke_value_t *i_lo, *j_lo, *hs, *sw, *i_start, *j_start;
    {
        rocke_value_t* x = ci(b, S);
        rocke_value_t* y = rocke_b_mul(b, oh0, x);
        rocke_value_t* z = ci(b, PAD);
        i_lo = rocke_b_sub(b, y, z);
    }
    {
        rocke_value_t* x = ci(b, S);
        rocke_value_t* y = rocke_b_mul(b, ow0, x);
        rocke_value_t* z = ci(b, PAD);
        j_lo = rocke_b_sub(b, y, z);
    }
    {
        rocke_value_t* x = rocke_b_sub(b, c.c0, i_lo);
        hs = rocke_b_smax(b, x, c.c0);
    }
    {
        rocke_value_t* x = rocke_b_sub(b, c.c0, j_lo);
        sw = rocke_b_smax(b, x, c.c0);
    }
    i_start = rocke_b_add(b, i_lo, hs);
    j_start = rocke_b_add(b, j_lo, sw);
    {
        rocke_value_t* x = ci(b, c.DW);
        rocke_value_t* y = rocke_b_mul(b, hs, x);
        rocke_value_t* z = rocke_b_add(b, y, sw);
        rocke_value_t* u = ci(b, c.XP);
        c.z_elems = rocke_b_mul(b, z, u);
    }
    {
        rocke_value_t* x = rocke_b_sub(b, c_W, j_start);
        rocke_value_t* y = ci(b, c.DW);
        rocke_value_t* z = rocke_b_sub(b, y, sw);
        rocke_value_t* u = rocke_b_smin(b, x, z);
        c.x_w_ext = rocke_b_smax(b, u, c.c0);
    }
    {
        rocke_value_t* x = rocke_b_sub(b, c_H, i_start);
        c.x_h_ext = rocke_b_smax(b, x, c.c0);
    }
    {
        rocke_value_t* x = ci(b, DH);
        c.x_tile2 = rocke_b_sub(b, x, hs);
    }
    c.x_base = nhw_off(b,
                       n_img,
                       params->p_A_stride_n,
                       i_start,
                       params->p_A_stride_hi,
                       j_start,
                       params->p_A_stride_wi);
    {
        rocke_value_t* x = ci(b, c.RS);
        c.c_RSC = rocke_b_mul(b, x, c.c_C);
    }

    {
        int shape[2] = {1, (int)(rocke_fwd_smallk_wmma_lds_bytes(spec) / 2)};
        c.lds = rocke_b_smem_alloc(b, c.io_type, shape, 2, "sk_lds");
    }
    {
        rocke_value_t* a = rocke_b_smem_addr_of(b, c.lds);
        c.lds_addr = rocke_b_trunc(b, a, rocke_i32());
    }
    long WEI0 = (long)c.NS * c.XE;
    {
        rocke_value_t* x = ci(b, WEI0 * 2);
        c.wei_addr = rocke_b_add(b, c.lds_addr, x);
    }

    {
        int max_z = (PAD * c.DW + PAD) * c.XP;
        if(max_z)
        {
            rocke_value_t* zero_v = rocke_b_zero_vec(b, c.io_type, 8);
            rocke_value_t* z_vecs = rocke_b_div(b, c.z_elems, c.c8);
            for(int j = 0; j < fsw_ceil_div(max_z, 8); j += THREADS)
            {
                rocke_value_t* cj = ci(b, j);
                rocke_value_t* v = rocke_b_add(b, tid, cj);
                rocke_value_t* cond = rocke_b_cmp_lt(b, v, z_vecs);
                scf_if_enter(b, cond);
                {
                    rocke_value_t* flat = rocke_b_mul(b, v, c.c8);
                    for(int st = 0; st < c.NS; ++st)
                    {
                        rocke_value_t* o = ci(b, (int64_t)st * c.XE);
                        rocke_value_t* a = rocke_b_add(b, flat, o);
                        rocke_value_t* idx[2] = {c.c0, a};
                        rocke_b_smem_store_vN(b, c.lds, idx, 2, zero_v, 8);
                    }
                }
                rocke_b_region_leave(b);
            }
        }
    }

    c.px_off.assign(c.TN, NULL);
    for(int tn = 0; tn < c.TN; ++tn)
    {
        rocke_value_t* flat = tile_lane(b, wave_p, c.TN, tn, c.c16, l16);
        rocke_value_t* x = ci(b, c.XP);
        rocke_value_t* y = rocke_b_mul(b, flat, x);
        rocke_value_t* z = rocke_b_mul(b, lhalf, c.c16);
        c.px_off[tn] = rocke_b_add(b, y, z);
    }
    c.wei_off.assign(c.TM, NULL);
    for(int m = 0; m < c.TM; ++m)
    {
        rocke_value_t* row = tile_lane(b, wave_c, c.TM, m, c.c16, l16);
        rocke_value_t* x = ci(b, c.WPITCH);
        rocke_value_t* y = rocke_b_mul(b, row, x);
        rocke_value_t* z = rocke_b_mul(b, lhalf, c.c16);
        rocke_value_t* u = rocke_b_add(b, y, z);
        rocke_value_t* v = ci(b, WEI0);
        c.wei_off[m] = rocke_b_add(b, u, v);
    }

    for(int st = 0; st < c.NS - 1; ++st)
    {
        rocke_value_t* x = ci(b, st);
        rocke_value_t* cond = rocke_b_cmp_lt(b, x, c.n_chunks);
        scf_if_enter(b, cond);
        sk_issue(c, ci(b, st));
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
        sk_chunk(c, loop.iv, accs);
        rocke_b_scf_yield(b, accs.data(), n_acc);
    }
    rocke_b_region_leave(b);
    rocke_value_t* const* accs = loop.op->results;

    /* ---- epilogue 1: Z (fp32) to LDS as [window pixel][(r, s, k)]. */
    rocke_b_sync_lds_only(b);
    for(int m = 0; m < c.TM; ++m)
    {
        rocke_value_t* col;
        {
            rocke_value_t* x = ci(b, c.TM);
            rocke_value_t* y = rocke_b_mul(b, wave_c, x);
            rocke_value_t* z = ci(b, m);
            rocke_value_t* u = rocke_b_add(b, y, z);
            rocke_value_t* v = rocke_b_mul(b, u, c.c16);
            rocke_value_t* w = rocke_b_mul(b, lhalf, c.c8);
            col = rocke_b_add(b, v, w);
        }
        for(int tn = 0; tn < c.TN; ++tn)
        {
            rocke_value_t* px = tile_lane(b, wave_p, c.TN, tn, c.c16, l16);
            rocke_value_t* x = ci(b, ZP);
            rocke_value_t* y = rocke_b_mul(b, px, x);
            rocke_value_t* zi = rocke_b_add(b, y, col);
            rocke_value_t* z2 = rocke_b_mul(b, zi, c.c2);
            rocke_value_t* idx[2] = {c.c0, z2};
            rocke_b_smem_store_vN(b, c.lds, idx, 2, accs[m * c.TN + tn], 8);
        }
    }
    rocke_b_sync_lds_only(b);

    /* ---- epilogue 2: overlap-add, one output pixel (all K) per thread. */
    int E = OH * OW;
    for(int it = 0; it < fsw_ceil_div(E, THREADS); ++it)
    {
        rocke_value_t* e;
        {
            rocke_value_t* x = ci(b, (int64_t)it * THREADS);
            e = rocke_b_add(b, tid, x);
        }
        rocke_value_t *ox, *oy, *oh, *ow, *ok;
        {
            rocke_value_t* x = ci(b, OW);
            ox = rocke_b_mod(b, e, x);
            rocke_value_t* y = ci(b, OW);
            oy = rocke_b_div(b, e, y);
        }
        oh = rocke_b_add(b, oh0, oy);
        ow = rocke_b_add(b, ow0, ox);
        {
            rocke_value_t* x = ci(b, E);
            rocke_value_t* y = rocke_b_cmp_lt(b, e, x);
            rocke_value_t* z = rocke_b_cmp_lt(b, oh, c_Ho);
            rocke_value_t* u = rocke_b_cmp_lt(b, ow, c_Wo);
            rocke_value_t* v = rocke_b_land(b, z, u);
            ok = rocke_b_land(b, y, v);
        }
        scf_if_enter(b, ok);
        {
            rocke_value_t* base;
            {
                rocke_value_t* x = ci(b, S);
                rocke_value_t* y = rocke_b_mul(b, oy, x);
                rocke_value_t* z = ci(b, c.DW);
                rocke_value_t* u = rocke_b_mul(b, y, z);
                rocke_value_t* v = ci(b, S);
                rocke_value_t* w = rocke_b_mul(b, ox, v);
                rocke_value_t* s = rocke_b_add(b, u, w);
                rocke_value_t* zp = ci(b, ZP);
                base = rocke_b_mul(b, s, zp);
            }
            rocke_value_t* acc = NULL;
            for(int r = 0; r < KH; ++r)
                for(int s = 0; s < KW; ++s)
                {
                    int64_t off = (int64_t)(r * c.DW + s) * ZP + (int64_t)(r * KW + s) * c.K;
                    rocke_value_t* x = ci(b, off);
                    rocke_value_t* y = rocke_b_add(b, base, x);
                    rocke_value_t* z = rocke_b_mul(b, y, c.c2);
                    rocke_value_t* idx[2] = {c.c0, z};
                    rocke_value_t* zv = rocke_b_smem_load_vN(b, c.lds, idx, 2, rocke_f32(), c.K);
                    acc = (acc == NULL) ? zv : rocke_b_vector_add(b, acc, zv);
                }
            rocke_value_t* out = nhw_off(b,
                                         n_img,
                                         params->p_D_stride_n,
                                         oh,
                                         params->p_D_stride_ho,
                                         ow,
                                         params->p_D_stride_wo);
            rocke_value_t* val = cvt_out(b, bf16, acc);
            rocke_b_global_store_vN(b, params->D, out, val, c.K, 2 * c.K);
        }
        rocke_b_region_leave(b);
    }
    if(!rocke_ir_builder_ok(b))
        return NULL;
    return b->kernel;
}

} // namespace

/* ===================================================================== *
 *  Public build / lower entry points
 * ===================================================================== */

rocke_kernel_def_t* rocke_build_fwd_smallc_wmma(rocke_ir_builder_t* b,
                                                const rocke_fwd_smallc_wmma_spec_t* spec,
                                                const char* arch)
{
    if(b == NULL || spec == NULL)
        return NULL;
    return build_smallc(b, spec, arch ? arch : "gfx1250");
}

rocke_kernel_def_t* rocke_build_fwd_smallk_wmma(rocke_ir_builder_t* b,
                                                const rocke_fwd_smallk_wmma_spec_t* spec,
                                                const char* arch)
{
    if(b == NULL || spec == NULL)
        return NULL;
    return build_smallk(b, spec, arch ? arch : "gfx1250");
}

rocke_kernel_def_t* rocke_build_fwd_smallc_wmma_new(rocke_ir_builder_t* b,
                                                    const rocke_fwd_smallc_wmma_spec_t* spec,
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
            rocke_i_set_err_msg(b, ROCKE_ERR_VALUE, "build_fwd_smallc_wmma: null spec");
            return NULL;
        }
        if(rocke_fwd_smallc_wmma_kernel_name(spec, name, sizeof(name)) != ROCKE_OK)
        {
            rocke_i_set_err_msg(
                b, ROCKE_ERR_VALUE, "build_fwd_smallc_wmma: kernel name does not fit");
            return NULL;
        }
        if(rocke_ir_builder_init(b, name) != ROCKE_OK)
            return NULL;
        return rocke_build_fwd_smallc_wmma(b, spec, arch);
    });
}

rocke_kernel_def_t* rocke_build_fwd_smallk_wmma_new(rocke_ir_builder_t* b,
                                                    const rocke_fwd_smallk_wmma_spec_t* spec,
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
            rocke_i_set_err_msg(b, ROCKE_ERR_VALUE, "build_fwd_smallk_wmma: null spec");
            return NULL;
        }
        if(rocke_fwd_smallk_wmma_kernel_name(spec, name, sizeof(name)) != ROCKE_OK)
        {
            rocke_i_set_err_msg(
                b, ROCKE_ERR_VALUE, "build_fwd_smallk_wmma: kernel name does not fit");
            return NULL;
        }
        if(rocke_ir_builder_init(b, name) != ROCKE_OK)
            return NULL;
        return rocke_build_fwd_smallk_wmma(b, spec, arch);
    });
}

namespace
{

template <typename Spec, typename BuildNew>
rocke_status_t lower_common(const Spec* spec,
                            const char* arch,
                            rocke_llvm_flavor_t flavor,
                            char** out_ll,
                            char* err,
                            size_t err_cap,
                            BuildNew build_new,
                            const char* who)
{
    rocke_ir_builder_t b = {};
    rocke_kernel_def_t* kernel;
    rocke_status_t st;

    if(out_ll != NULL)
        *out_ll = NULL;
    if(spec == NULL || out_ll == NULL)
    {
        fsw_set_reason(err, err_cap, "lower_to_llvm: null spec/out");
        return ROCKE_ERR_VALUE;
    }
    if(arch == NULL)
        arch = "gfx1250";
    kernel = build_new(&b, spec, arch);
    if(kernel == NULL)
    {
        const char* m = rocke_ir_builder_error(&b);
        st = rocke_ir_builder_status(&b);
        fsw_set_reason(err, err_cap, (m != NULL && m[0] != '\0') ? m : who);
        rocke_ir_builder_free(&b);
        return (st == ROCKE_OK) ? ROCKE_ERR_VALUE : st;
    }
    st = rocke_lower_kernel_to_llvm_ex(kernel, flavor, arch, out_ll, err, err_cap);
    rocke_ir_builder_free(&b);
    return st;
}

} // namespace

rocke_status_t rocke_fwd_smallc_wmma_lower_to_llvm(const rocke_fwd_smallc_wmma_spec_t* spec,
                                                   const char* arch,
                                                   rocke_llvm_flavor_t flavor,
                                                   char** out_ll,
                                                   char* err,
                                                   size_t err_cap)
{
    return lower_common(spec,
                        arch,
                        flavor,
                        out_ll,
                        err,
                        err_cap,
                        rocke_build_fwd_smallc_wmma_new,
                        "build_fwd_smallc_wmma failed");
}

rocke_status_t rocke_fwd_smallk_wmma_lower_to_llvm(const rocke_fwd_smallk_wmma_spec_t* spec,
                                                   const char* arch,
                                                   rocke_llvm_flavor_t flavor,
                                                   char** out_ll,
                                                   char* err,
                                                   size_t err_cap)
{
    return lower_common(spec,
                        arch,
                        flavor,
                        out_ll,
                        err,
                        err_cap,
                        rocke_build_fwd_smallk_wmma_new,
                        "build_fwd_smallk_wmma failed");
}
