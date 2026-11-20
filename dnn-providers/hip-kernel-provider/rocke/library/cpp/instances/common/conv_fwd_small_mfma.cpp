// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * conv_fwd_small_mfma.cpp -- C++ port of build_fwd_smallc_mfma and
 * build_fwd_smallk_mfma (library/kernels/common/conv_fwd_small_mfma.py): the
 * wave64 MFMA forward convolutions of tiny channel counts for gfx950. See the
 * header and the Python module for the algorithms.
 *
 * Every helper emits in PYTHON EVALUATION ORDER (arguments left-to-right,
 * innermost call first) so the SSA sequence -- and therefore the .ll bytes --
 * match exactly. Nested builder calls are hoisted into ordered locals: C++
 * leaves the evaluation order of call arguments unspecified.
 */
#include "rocke/instance_conv_fwd_small_mfma.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <string>
#include <vector>

#include "rocke/arch_target.h" /* rocke_set_unknown_arch_reason, rocke_mma_catalog_has_shape */
#include "rocke/error_boundary.hpp" /* ckc::guard_builder boundary shim */
#include "rocke/helper_rocke.core.arch.h" /* rocke_archtarget_fits_lds, max_threads_per_block */
#include "rocke/helper_rocke.helpers.io.h" /* rocke_b_io_ir_type */
#include "rocke/helper_rocke.helpers.spec.h" /* rocke_kernel_name_join, rocke_ceil_div, ... */
#include "rocke/instance_conv_direct_grouped_internal.h" /* rocke_dconv_emit_params */
#include "rocke/instance_conv_fwd_small_wmma_internal.h" /* rocke_fwd_small_common_shape_ok */
#include "rocke/ir_internal.h" /* rocke_i_set_err_msg */
#include "rocke/lower_llvm.h"

/* conv_fwd_small_mfma.SMALLC_MFMA_MAX_C / SMALLC_MFMA_VGPR_BUDGET / _MFMA_K /
 * SMALLK_MFMA_MAX_K / SMALLK_MFMA_VGPR_BUDGET. */
#define FSM_SMALLC_MAX_C 4
#define FSM_SMALLC_VGPR_BUDGET 400
#define FSM_MFMA_K 32
#define FSM_SMALLK_MAX_K 8
#define FSM_SMALLK_VGPR_BUDGET 400
/* Python _OOB_OFFSET: the byte offset of a masked-out buffer load, 2**31 --
 * at or past the end of every buffer (tensors are below 2 GiB, and the small-C
 * X buffer rounded up to whole dwords is at most 2**31 bytes). */
#define FSM_OOB_OFFSET (-2147483647LL - 1)

namespace
{

/* _mfma_common_reason(p, arch, name, wave_size, square): a wave64 target with
 * the 16x16x32 MFMA of the dtype, every extent at least 1, the padding, then
 * the shapes the WMMA forms serve too. On success stores the resolved target
 * in `*target_out` and returns true; otherwise writes reason and returns
 * false. */
bool fsm_common_reason(const rocke_direct_conv_problem_t* p,
                       const char* arch,
                       const char* name,
                       int wave_size,
                       bool square,
                       const rocke_archtarget_t** target_out,
                       char* reason,
                       size_t reason_cap)
{
    char msg[512];
    const rocke_archtarget_t* target = rocke_archtarget_from_gfx(arch);
    *target_out = target;
    if(target == NULL)
    {
        rocke_set_unknown_arch_reason(reason, reason_cap, arch);
        return false;
    }
    if(target->wave_size != 64 || wave_size != 64)
    {
        snprintf(msg, sizeof msg, "%s is a wave64 MFMA kernel", name);
        rocke_spec_set_reason(reason, reason_cap, msg);
        return false;
    }
    const char* dtype = rocke_direct_conv_problem_dtype(p);
    if(strcmp(dtype, "fp16") != 0 && strcmp(dtype, "bf16") != 0)
    {
        snprintf(msg, sizeof msg, "unsupported dtype '%s'", dtype);
        rocke_spec_set_reason(reason, reason_cap, msg);
        return false;
    }
    const char* ab = strcmp(dtype, "bf16") == 0 ? "bf16" : "f16";
    if(!rocke_mma_catalog_has_shape(
           rocke_archtarget_mma(target), "mma", ab, ab, "fp32", 16, 16, FSM_MFMA_K))
    {
        snprintf(msg, sizeof msg, "missing mfma_f32_16x16x32_%s on %s", ab, arch);
        rocke_spec_set_reason(reason, reason_cap, msg);
        return false;
    }
    if(rocke_direct_problem_dims_reason(p, reason, reason_cap))
        return false;
    if(square ? rocke_direct_conv_one_padding_reason(p, "", reason, reason_cap)
              : rocke_direct_conv_pad_w_reason(p, "", reason, reason_cap))
        return false;
    return rocke_fwd_small_common_shape_ok(p, square, reason, reason_cap);
}

/* ---- FwdSmallcMfmaSpec @property helpers the builder shares. */
int sc_nc(const rocke_fwd_smallc_mfma_spec_t* s)
{
    return s->problem.KH * s->problem.KW * s->problem.cpg;
}
int sc_steps(const rocke_fwd_smallc_mfma_spec_t* s)
{
    return (int)rocke_ceil_div(sc_nc(s), FSM_MFMA_K);
}
int sc_win_h(const rocke_fwd_smallc_mfma_spec_t* s)
{
    return (s->out_h - 1) * s->problem.stride + s->problem.KH;
}
int sc_win_w(const rocke_fwd_smallc_mfma_spec_t* s)
{
    return (s->out_w - 1) * s->problem.stride + s->problem.KW;
}
int sc_row_pitch(const rocke_fwd_smallc_mfma_spec_t* s)
{
    return sc_win_w(s) * s->problem.cpg;
}
int sc_lds_pitch(const rocke_fwd_smallc_mfma_spec_t* s)
{
    return sc_row_pitch(s) + 16;
}
int sc_row_vecs(const rocke_fwd_smallc_mfma_spec_t* s)
{
    return (int)rocke_ceil_div(sc_row_pitch(s) + 7, 8);
}
long sc_win_elems(const rocke_fwd_smallc_mfma_spec_t* s)
{
    return (long)sc_win_h(s) * sc_lds_pitch(s);
}

/* ---- FwdSmallkMfmaSpec @property helpers. */
int sk_ncols(const rocke_fwd_smallk_mfma_spec_t* s)
{
    return s->problem.KH * s->problem.KW * s->problem.kpg;
}
int sk_n_tiles(const rocke_fwd_smallk_mfma_spec_t* s)
{
    return (int)rocke_ceil_div(sk_ncols(s), 16);
}
int sk_win_h(const rocke_fwd_smallk_mfma_spec_t* s)
{
    return (s->out_h - 1) * s->problem.stride + s->problem.KH;
}
int sk_win_w(const rocke_fwd_smallk_mfma_spec_t* s)
{
    return (s->out_w - 1) * s->problem.stride + s->problem.KW;
}
int sk_m_tiles(const rocke_fwd_smallk_mfma_spec_t* s)
{
    long long pix_tiles = rocke_ceil_div((long long)sk_win_h(s) * sk_win_w(s), 16);
    /* max: a rejected waves < 1 still reports its LDS size. */
    return (int)rocke_ceil_div(pix_tiles, s->waves > 1 ? s->waves : 1);
}
int sk_z_pitch(const rocke_fwd_smallk_mfma_spec_t* s)
{
    return sk_n_tiles(s) * 16 + 4;
}

} // namespace

/* ===================================================================== *
 *  FwdSmallcMfmaSpec: defaults + derived geometry
 * ===================================================================== */

rocke_fwd_smallc_mfma_spec_t rocke_fwd_smallc_mfma_spec_default(void)
{
    rocke_fwd_smallc_mfma_spec_t s;
    memset(&s, 0, sizeof(s));
    s.problem = rocke_direct_conv_problem_default();
    s.name = "fwd_smallc_mfma";
    s.waves_k = 1;
    s.waves_p = 4;
    s.tiles_k = 4;
    s.tiles_p = 2;
    s.out_h = 4;
    s.out_w = 32;
    s.tiles_per_block = 1;
    s.prefetch = false;
    s.wave_size = 64;
    return s;
}

int rocke_fwd_smallc_mfma_block_k(const rocke_fwd_smallc_mfma_spec_t* spec)
{
    return spec->waves_k * spec->tiles_k * 16;
}

int rocke_fwd_smallc_mfma_block_p(const rocke_fwd_smallc_mfma_spec_t* spec)
{
    return spec->waves_p * spec->tiles_p * 16;
}

int rocke_fwd_smallc_mfma_threads_per_block(const rocke_fwd_smallc_mfma_spec_t* spec)
{
    return spec->waves_k * spec->waves_p * spec->wave_size;
}

long rocke_fwd_smallc_mfma_lds_bytes(const rocke_fwd_smallc_mfma_spec_t* spec)
{
    return 2 * sc_win_elems(spec);
}

void rocke_fwd_smallc_mfma_grid(const rocke_fwd_smallc_mfma_spec_t* spec, int grid[3])
{
    const rocke_direct_conv_problem_t* p = &spec->problem;
    long long n_tiles = (long long)p->N
                        * rocke_ceil_div(rocke_direct_conv_problem_Ho(p), spec->out_h)
                        * rocke_ceil_div(rocke_direct_conv_problem_Wo(p), spec->out_w);
    grid[0] = (int)rocke_ceil_div(n_tiles, spec->tiles_per_block);
    grid[1] = (int)rocke_ceil_div(rocke_direct_conv_problem_total_k(p),
                                  rocke_fwd_smallc_mfma_block_k(spec));
    grid[2] = 1;
}

rocke_status_t rocke_fwd_smallc_mfma_kernel_name(const rocke_fwd_smallc_mfma_spec_t* spec,
                                                 char* out,
                                                 size_t out_cap)
{
    char f[32], st[32], pad[32], cc[32], wk[32], wp[32], tk[32], tp[32], oh[32], ow[32], tb[32];
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
    const char* parts[] = {f, st, pad, cc, wk, wp, tk, tp, oh, ow, tb, spec->prefetch ? "pf" : ""};
    const char* flag_names[] = {"bf16"};
    int flag_on[] = {rocke_direct_conv_problem_is_bf16(p) ? 1 : 0};
    return rocke_kernel_name_join(spec->name ? spec->name : "fwd_smallc_mfma",
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
 *  fwd_smallc_mfma_reason / is_valid_fwd_smallc_mfma_spec
 * ===================================================================== */

bool rocke_fwd_smallc_mfma_is_valid_spec(const rocke_fwd_smallc_mfma_spec_t* spec,
                                         const char* arch,
                                         char* reason,
                                         size_t reason_cap)
{
    char msg[512];
    const rocke_archtarget_t* target;
    const rocke_direct_conv_problem_t* p;
    int vgprs, steps;

    if(spec == NULL)
    {
        rocke_spec_set_reason(reason, reason_cap, "null spec");
        return false;
    }
    if(arch == NULL)
        arch = "gfx950";
    p = &spec->problem;
    if(!fsm_common_reason(
           p, arch, "fwd_smallc_mfma", spec->wave_size, true, &target, reason, reason_cap))
        return false;
    if(p->cpg > FSM_SMALLC_MAX_C)
    {
        snprintf(msg, sizeof msg, "C %d > %d", p->cpg, FSM_SMALLC_MAX_C);
        rocke_spec_set_reason(reason, reason_cap, msg);
        return false;
    }
    if(p->kpg % 4 != 0)
    {
        snprintf(msg, sizeof msg, "K %d must be a multiple of 4 (8-byte output writes)", p->kpg);
        rocke_spec_set_reason(reason, reason_cap, msg);
        return false;
    }
    if(spec->waves_k < 1 || spec->waves_p < 1 || spec->tiles_k < 1 || spec->tiles_p < 1)
    {
        rocke_spec_set_reason(reason, reason_cap, "waves and tiles must be >= 1");
        return false;
    }
    if(spec->tiles_per_block < 1)
    {
        rocke_spec_set_reason(reason, reason_cap, "tiles_per_block must be >= 1");
        return false;
    }
    if(spec->prefetch && spec->tiles_per_block < 2)
    {
        rocke_spec_set_reason(
            reason, reason_cap, "prefetch needs tiles_per_block >= 2 (a next tile to prefetch)");
        return false;
    }
    if(spec->out_h < 1 || spec->out_w < 1)
    {
        snprintf(msg, sizeof msg, "out tile %dx%d must be at least 1x1", spec->out_h, spec->out_w);
        rocke_spec_set_reason(reason, reason_cap, msg);
        return false;
    }
    if(spec->out_h * spec->out_w != rocke_fwd_smallc_mfma_block_p(spec))
    {
        snprintf(msg,
                 sizeof msg,
                 "out tile %dx%d must hold exactly the %d GEMM pixels",
                 spec->out_h,
                 spec->out_w,
                 rocke_fwd_smallc_mfma_block_p(spec));
        rocke_spec_set_reason(reason, reason_cap, msg);
        return false;
    }
    if(rocke_fwd_smallc_mfma_threads_per_block(spec)
       > rocke_archtarget_max_threads_per_block(target))
    {
        snprintf(
            msg, sizeof msg, "%d threads per block", rocke_fwd_smallc_mfma_threads_per_block(spec));
        rocke_spec_set_reason(reason, reason_cap, msg);
        return false;
    }
    if(!rocke_archtarget_fits_lds(target, rocke_fwd_smallc_mfma_lds_bytes(spec)))
    {
        snprintf(msg,
                 sizeof msg,
                 "LDS %ld B > %d",
                 rocke_fwd_smallc_mfma_lds_bytes(spec),
                 target->lds_capacity_bytes);
        rocke_spec_set_reason(reason, reason_cap, msg);
        return false;
    }
    steps = sc_steps(spec);
    vgprs = 4 * spec->tiles_k * spec->tiles_p + 4 * spec->tiles_k * steps + 8 * steps + 4 * steps;
    if(vgprs > FSM_SMALLC_VGPR_BUDGET)
    {
        snprintf(msg, sizeof msg, "~%d VGPRs of accumulators, weights and offsets", vgprs);
        rocke_spec_set_reason(reason, reason_cap, msg);
        return false;
    }
    rocke_spec_set_reason(reason, reason_cap, "ok");
    return true;
}

/* ===================================================================== *
 *  FwdSmallkMfmaSpec: defaults + derived geometry
 * ===================================================================== */

rocke_fwd_smallk_mfma_spec_t rocke_fwd_smallk_mfma_spec_default(void)
{
    rocke_fwd_smallk_mfma_spec_t s;
    memset(&s, 0, sizeof(s));
    s.problem = rocke_direct_conv_problem_default();
    s.name = "fwd_smallk_mfma";
    s.waves = 4;
    s.out_h = 16;
    s.out_w = 16;
    s.wave_size = 64;
    return s;
}

int rocke_fwd_smallk_mfma_threads_per_block(const rocke_fwd_smallk_mfma_spec_t* spec)
{
    return spec->waves * spec->wave_size;
}

long rocke_fwd_smallk_mfma_lds_bytes(const rocke_fwd_smallk_mfma_spec_t* spec)
{
    return 4L * spec->waves * sk_m_tiles(spec) * 16 * sk_z_pitch(spec);
}

void rocke_fwd_smallk_mfma_grid(const rocke_fwd_smallk_mfma_spec_t* spec, int grid[3])
{
    const rocke_direct_conv_problem_t* p = &spec->problem;
    grid[0] = (int)((long long)p->N * rocke_ceil_div(rocke_direct_conv_problem_Ho(p), spec->out_h)
                    * rocke_ceil_div(rocke_direct_conv_problem_Wo(p), spec->out_w));
    grid[1] = 1;
    grid[2] = 1;
}

rocke_status_t rocke_fwd_smallk_mfma_kernel_name(const rocke_fwd_smallk_mfma_spec_t* spec,
                                                 char* out,
                                                 size_t out_cap)
{
    char f[32], st[32], pad[32], padw[32], kk[32], w[32], oh[32], ow[32];
    const rocke_direct_conv_problem_t* p = &spec->problem;
    const int pad_w = rocke_direct_conv_problem_pad_w(p);
    snprintf(f, sizeof f, "f%dx%d", p->KH, p->KW);
    snprintf(st, sizeof st, "s%d", p->stride);
    snprintf(pad, sizeof pad, "p%d", p->PAD);
    if(pad_w != p->PAD)
        snprintf(padw, sizeof padw, "pw%d", pad_w);
    else
        padw[0] = '\0';
    snprintf(kk, sizeof kk, "k%d", p->kpg);
    snprintf(w, sizeof w, "w%d", spec->waves);
    snprintf(oh, sizeof oh, "oh%d", spec->out_h);
    snprintf(ow, sizeof ow, "ow%d", spec->out_w);
    const char* parts[] = {f, st, pad, padw, kk, w, oh, ow};
    const char* flag_names[] = {"bf16"};
    int flag_on[] = {rocke_direct_conv_problem_is_bf16(p) ? 1 : 0};
    return rocke_kernel_name_join(spec->name ? spec->name : "fwd_smallk_mfma",
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
 *  fwd_smallk_mfma_reason / is_valid_fwd_smallk_mfma_spec
 * ===================================================================== */

bool rocke_fwd_smallk_mfma_is_valid_spec(const rocke_fwd_smallk_mfma_spec_t* spec,
                                         const char* arch,
                                         char* reason,
                                         size_t reason_cap)
{
    char msg[512];
    const rocke_archtarget_t* target;
    const rocke_direct_conv_problem_t* p;
    int vgprs, mt, nt;

    if(spec == NULL)
    {
        rocke_spec_set_reason(reason, reason_cap, "null spec");
        return false;
    }
    if(arch == NULL)
        arch = "gfx950";
    p = &spec->problem;
    if(!fsm_common_reason(
           p, arch, "fwd_smallk_mfma", spec->wave_size, false, &target, reason, reason_cap))
        return false;
    if(p->kpg > FSM_SMALLK_MAX_K)
    {
        snprintf(msg, sizeof msg, "K %d > %d", p->kpg, FSM_SMALLK_MAX_K);
        rocke_spec_set_reason(reason, reason_cap, msg);
        return false;
    }
    if(p->cpg % 8 != 0)
    {
        snprintf(msg, sizeof msg, "C %d must be a multiple of 8 (16-byte input fragments)", p->cpg);
        rocke_spec_set_reason(reason, reason_cap, msg);
        return false;
    }
    if(spec->waves < 1 || spec->out_h < 1 || spec->out_w < 1)
    {
        rocke_spec_set_reason(reason, reason_cap, "waves and the output tile must be >= 1");
        return false;
    }
    if(rocke_fwd_smallk_mfma_threads_per_block(spec)
       > rocke_archtarget_max_threads_per_block(target))
    {
        snprintf(
            msg, sizeof msg, "%d threads per block", rocke_fwd_smallk_mfma_threads_per_block(spec));
        rocke_spec_set_reason(reason, reason_cap, msg);
        return false;
    }
    if(!rocke_archtarget_fits_lds(target, rocke_fwd_smallk_mfma_lds_bytes(spec)))
    {
        snprintf(msg,
                 sizeof msg,
                 "LDS %ld B > %d",
                 rocke_fwd_smallk_mfma_lds_bytes(spec),
                 target->lds_capacity_bytes);
        rocke_spec_set_reason(reason, reason_cap, msg);
        return false;
    }
    mt = sk_m_tiles(spec);
    nt = sk_n_tiles(spec);
    vgprs = 4 * mt * nt + 4 * (mt + nt);
    if(vgprs > FSM_SMALLK_VGPR_BUDGET)
    {
        snprintf(msg, sizeof msg, "~%d VGPRs of accumulators and fragments", vgprs);
        rocke_spec_set_reason(reason, reason_cap, msg);
        return false;
    }
    rocke_spec_set_reason(reason, reason_cap, "ok");
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

/* add(mul(add(mul(wave, T), t), 16), extra), in Python order. */
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

/* mul(add(mul(wave, T), t), 16): tile_lane without the lane term, which the
 * caller evaluates after this (the order of the Python expression). */
rocke_value_t*
    tile_row(rocke_ir_builder_t* b, rocke_value_t* wave, int T, int t, rocke_value_t* c16)
{
    rocke_value_t* x = ci(b, T);
    rocke_value_t* y = rocke_b_mul(b, wave, x);
    rocke_value_t* z = ci(b, t);
    rocke_value_t* u = rocke_b_add(b, y, z);
    return rocke_b_mul(b, u, c16);
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

/* _mfma(b, dtype, "16x16x32", a, b, acc). */
rocke_value_t* mfma32(
    rocke_ir_builder_t* b, bool bf16, rocke_value_t* a, rocke_value_t* bv, rocke_value_t* acc)
{
    return bf16 ? rocke_b_mfma_f32_16x16x32_bf16(b, a, bv, acc)
                : rocke_b_mfma_f32_16x16x32_f16(b, a, bv, acc);
}

/* buffer_load_vN_{bf16,f16}(rsrc, voff, c0, 4). */
rocke_value_t* load16(
    rocke_ir_builder_t* b, bool bf16, rocke_value_t* rsrc, rocke_value_t* voff, rocke_value_t* c0)
{
    return bf16 ? rocke_b_buffer_load_vN_bf16(b, rsrc, voff, c0, 4)
                : rocke_b_buffer_load_vN_f16(b, rsrc, voff, c0, 4);
}

/* ===================================================================== *
 *  build_fwd_smallc_mfma
 * ===================================================================== */

struct sc_ctx
{
    rocke_ir_builder_t* b;
    rocke_dconv_params_t params;
    const rocke_type_t* io_type;
    bool bf16;

    int S, PAD, C, NC, STEPS;
    int WK, TM, TN, THREADS, BK, TOH, TOW, WIN_H, WP, WPS, NV;

    rocke_value_t *c0, *c4, *c8, *c16, *c_K, *c_H, *c_W, *c_Ho, *c_Wo, *c_C;
    rocke_value_t *tid, *lgrp, *wave_k, *n_ht, *n_wt, *k_blk, *lds, *zero_e;
    rocke_value_t *a_rsrc, *c_sent;

    std::vector<std::vector<rocke_value_t*>> a_frags, win_off;
    std::vector<rocke_value_t*> lane_ty, lane_tx, lane_base;
};

struct sc_geom
{
    rocke_value_t *n_img, *oh0, *ow0, *h_start, *w_start;
};

struct sc_load
{
    rocke_value_t *vec, *e0, *row, *in_win;
};

/* _geom(tile). */
sc_geom sc_tile_geom(sc_ctx& c, rocke_value_t* tile)
{
    rocke_ir_builder_t* b = c.b;
    sc_geom g;
    rocke_value_t* tw = rocke_b_mod(b, tile, c.n_wt);
    rocke_value_t* rest = rocke_b_div(b, tile, c.n_wt);
    rocke_value_t* th = rocke_b_mod(b, rest, c.n_ht);
    g.n_img = rocke_b_div(b, rest, c.n_ht);
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
    return g;
}

/* _load_window(g): the window's 16-byte loads. */
std::vector<sc_load> sc_load_window(sc_ctx& c, const sc_geom& g)
{
    rocke_ir_builder_t* b = c.b;
    const rocke_dconv_params_t* params = &c.params;
    rocke_value_t* img_base = rocke_b_mul(b, g.n_img, params->p_A_stride_n);
    rocke_value_t* wcC = rocke_b_mul(b, g.w_start, c.c_C);
    std::vector<sc_load> out;
    int n_it = (int)rocke_ceil_div((long long)c.WIN_H * c.NV, c.THREADS);
    for(int it = 0; it < n_it; ++it)
    {
        sc_load ld;
        rocke_value_t* v_lin;
        {
            rocke_value_t* x = ci(b, (int64_t)it * c.THREADS);
            v_lin = rocke_b_add(b, c.tid, x);
        }
        {
            rocke_value_t* x = ci(b, (int64_t)c.WIN_H * c.NV);
            ld.in_win = rocke_b_cmp_lt(b, v_lin, x);
        }
        {
            rocke_value_t* x = ci(b, c.NV);
            ld.row = rocke_b_div(b, v_lin, x);
        }
        rocke_value_t* v;
        {
            rocke_value_t* x = ci(b, c.NV);
            v = rocke_b_mod(b, v_lin, x);
        }
        rocke_value_t* hr = rocke_b_add(b, g.h_start, ld.row);
        rocke_value_t* row_ok;
        {
            rocke_value_t* x = rocke_b_cmp_ge(b, hr, c.c0);
            rocke_value_t* y = rocke_b_cmp_lt(b, hr, c.c_H);
            rocke_value_t* z = rocke_b_land(b, x, y);
            row_ok = rocke_b_land(b, ld.in_win, z);
        }
        rocke_value_t* gs;
        {
            rocke_value_t* x = rocke_b_mul(b, hr, params->p_A_stride_hi);
            rocke_value_t* y = rocke_b_add(b, img_base, x);
            gs = rocke_b_add(b, y, wcC);
        }
        rocke_value_t* land;
        {
            rocke_value_t* x = ci(b, 7);
            land = rocke_b_land(b, gs, x);
        }
        rocke_value_t* g0;
        {
            rocke_value_t* x = rocke_b_sub(b, gs, land);
            rocke_value_t* y = rocke_b_mul(b, v, c.c8);
            g0 = rocke_b_add(b, x, y);
        }
        rocke_value_t* ok;
        {
            rocke_value_t* x = rocke_b_cmp_ge(b, g0, c.c0);
            ok = rocke_b_land(b, row_ok, x);
        }
        rocke_value_t* voff;
        {
            rocke_value_t* x = ci(b, 2);
            rocke_value_t* y = rocke_b_mul(b, g0, x);
            voff = rocke_b_select(b, ok, y, c.c_sent);
        }
        ld.vec = load16(b, c.bf16, c.a_rsrc, voff, c.c0);
        {
            rocke_value_t* x = rocke_b_mul(b, v, c.c8);
            ld.e0 = rocke_b_sub(b, x, land);
        }
        out.push_back(ld);
    }
    return out;
}

/* _commit_window(g, loads): mask each element to the image, store to LDS. */
void sc_commit_window(sc_ctx& c, const sc_geom& g, const std::vector<sc_load>& loads)
{
    rocke_ir_builder_t* b = c.b;
    for(const sc_load& ld : loads)
    {
        rocke_value_t* lds_row;
        {
            rocke_value_t* x = ci(b, c.WPS);
            rocke_value_t* y = rocke_b_mul(b, ld.row, x);
            lds_row = rocke_b_add(b, y, c.c8);
        }
        scf_if_enter(b, ld.in_win);
        for(int i = 0; i < 8; ++i)
        {
            rocke_value_t* e;
            {
                rocke_value_t* x = ci(b, i);
                e = rocke_b_add(b, ld.e0, x);
            }
            rocke_value_t* e_pos = rocke_b_smax(b, e, c.c0);
            rocke_value_t* col;
            {
                rocke_value_t* x = rocke_b_div(b, e_pos, c.c_C);
                col = rocke_b_add(b, g.w_start, x);
            }
            rocke_value_t* valid;
            {
                rocke_value_t* x = rocke_b_cmp_ge(b, e, c.c0);
                rocke_value_t* y = ci(b, c.WP);
                rocke_value_t* z = rocke_b_cmp_lt(b, e, y);
                rocke_value_t* u = rocke_b_land(b, x, z);
                rocke_value_t* v = rocke_b_cmp_ge(b, col, c.c0);
                rocke_value_t* w = rocke_b_cmp_lt(b, col, c.c_W);
                rocke_value_t* t = rocke_b_land(b, v, w);
                valid = rocke_b_land(b, u, t);
            }
            rocke_value_t* val;
            {
                rocke_value_t* x = rocke_b_vec_extract(b, ld.vec, i);
                val = rocke_b_select(b, valid, x, c.zero_e);
            }
            /* Elements past the row's right slack land on its last slack element. */
            rocke_value_t* at;
            {
                rocke_value_t* x = ci(b, (int64_t)c.WP + 7);
                rocke_value_t* y = rocke_b_smin(b, e, x);
                at = rocke_b_add(b, lds_row, y);
            }
            rocke_value_t* idx[2] = {c.c0, at};
            rocke_b_smem_store_vN(b, c.lds, idx, 2, val, 1);
        }
        rocke_b_region_leave(b);
    }
}

/* _compute_store(g). */
void sc_compute_store(sc_ctx& c, const sc_geom& g)
{
    rocke_ir_builder_t* b = c.b;
    const rocke_dconv_params_t* params = &c.params;
    std::vector<rocke_value_t*> accs(c.TM * c.TN);
    for(int i = 0; i < c.TM * c.TN; ++i)
        accs[i] = rocke_b_zero_vec_f32(b, 4);
    for(int tn = 0; tn < c.TN; ++tn)
    {
        for(int t = 0; t < c.STEPS; ++t)
        {
            rocke_value_t* elems[8];
            for(int i = 0; i < 8; ++i)
            {
                rocke_value_t* at = rocke_b_add(b, c.lane_base[tn], c.win_off[t][i]);
                rocke_value_t* idx[2] = {c.c0, at};
                rocke_value_t* lv = rocke_b_smem_load_vN(b, c.lds, idx, 2, c.io_type, 1);
                elems[i] = rocke_b_vec_extract(b, lv, 0);
            }
            rocke_value_t* bf = rocke_b_vec_pack(b, elems, 8, c.io_type);
            for(int m = 0; m < c.TM; ++m)
            {
                int i = m * c.TN + tn;
                accs[i] = mfma32(b, c.bf16, c.a_frags[m][t], bf, accs[i]);
            }
        }
    }

    /* ---- epilogue: 4 consecutive output channels per lane, 8-byte writes. */
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
            b, g.n_img, params->p_D_stride_n, oh, params->p_D_stride_ho, ow, params->p_D_stride_wo);
        for(int m = 0; m < c.TM; ++m)
        {
            rocke_value_t* k;
            {
                rocke_value_t* x = tile_row(b, c.wave_k, c.TM, m, c.c16);
                rocke_value_t* y = rocke_b_mul(b, c.lgrp, c.c4);
                rocke_value_t* z = rocke_b_add(b, x, y);
                k = rocke_b_add(b, c.k_blk, z);
            }
            rocke_value_t* cond;
            {
                rocke_value_t* x = rocke_b_cmp_lt(b, k, c.c_K);
                cond = rocke_b_land(b, px_ok, x);
            }
            scf_if_enter(b, cond);
            {
                rocke_value_t* off = rocke_b_add(b, px_off, k);
                rocke_value_t* val = rocke_dconv_trunc_f32(b, c.bf16, accs[m * c.TN + tn]);
                rocke_b_global_store_vN(b, params->D, off, val, 4, 8);
            }
            rocke_b_region_leave(b);
        }
    }
}

rocke_kernel_def_t*
    build_smallc(rocke_ir_builder_t* b, const rocke_fwd_smallc_mfma_spec_t* spec, const char* arch)
{
    char reason[512];
    sc_ctx c;
    if(!rocke_fwd_smallc_mfma_is_valid_spec(spec, arch, reason, sizeof reason))
    {
        std::string msg = std::string("invalid FwdSmallcMfmaSpec for ") + arch + ": " + reason;
        rocke_i_set_err_msg(b, ROCKE_ERR_VALUE, msg.c_str());
        return NULL;
    }
    const rocke_direct_conv_problem_t* p = &spec->problem;
    c.b = b;
    const int KW = p->KW;
    c.S = p->stride;
    c.PAD = p->PAD;
    c.C = p->cpg;
    c.NC = sc_nc(spec);
    c.STEPS = sc_steps(spec);
    c.bf16 = rocke_direct_conv_problem_is_bf16(p);
    c.io_type = rocke_b_io_ir_type(b, rocke_direct_conv_problem_dtype(p));
    if(c.io_type == NULL)
        return NULL;
    const int WAVE = spec->wave_size;
    c.WK = spec->waves_k;
    c.TM = spec->tiles_k;
    c.TN = spec->tiles_p;
    c.THREADS = rocke_fwd_smallc_mfma_threads_per_block(spec);
    c.BK = rocke_fwd_smallc_mfma_block_k(spec);
    c.TOH = spec->out_h;
    c.TOW = spec->out_w;
    c.WIN_H = sc_win_h(spec);
    c.WP = sc_row_pitch(spec);
    c.WPS = sc_lds_pitch(spec);
    c.NV = sc_row_vecs(spec);
    const long WIN = sc_win_elems(spec);
    const int T = spec->tiles_per_block;

    rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", c.THREADS);
    rocke_dconv_emit_params(b, &c.params, "fwd", c.io_type);
    const rocke_dconv_params_t* params = &c.params;

    c.c0 = ci(b, 0);
    c.c4 = ci(b, 4);
    c.c8 = ci(b, 8);
    c.c16 = ci(b, 16);
    c.c_K = params->p_total_k;
    c.c_H = params->p_Hi;
    c.c_W = params->p_Wi;
    c.c_Ho = params->p_Ho;
    c.c_Wo = params->p_Wo;

    c.tid = rocke_b_thread_id_x(b);
    rocke_value_t *wave_id, *lane, *l16, *wave_p;
    {
        rocke_value_t* c_wave = ci(b, WAVE);
        wave_id = rocke_b_div(b, c.tid, c_wave);
        lane = rocke_b_mod(b, c.tid, c_wave);
    }
    l16 = rocke_b_mod(b, lane, c.c16);
    c.lgrp = rocke_b_div(b, lane, c.c16);
    {
        rocke_value_t* x = ci(b, c.WK);
        c.wave_k = rocke_b_mod(b, wave_id, x);
        rocke_value_t* y = ci(b, c.WK);
        wave_p = rocke_b_div(b, wave_id, y);
    }

    c.n_ht = n_tiles_of(b, c.c_Ho, c.TOH);
    c.n_wt = n_tiles_of(b, c.c_Wo, c.TOW);
    rocke_value_t* n_tiles;
    {
        rocke_value_t* x = rocke_b_mul(b, params->p_N, c.n_ht);
        n_tiles = rocke_b_mul(b, x, c.n_wt);
    }
    {
        rocke_value_t* by = rocke_b_block_id_y(b);
        rocke_value_t* x = ci(b, c.BK);
        c.k_blk = rocke_b_mul(b, by, x);
    }

    {
        int shape[2] = {1, (int)WIN};
        c.lds = rocke_b_smem_alloc(b, c.io_type, shape, 2, "scm_lds");
    }

    /* ---- the lane's reduction indices: j = t*32 + lgrp*8 + i. */
    rocke_value_t* lane_j0 = rocke_b_mul(b, c.lgrp, c.c8);
    {
        rocke_value_t* z2 = rocke_b_zero_vec(b, c.io_type, 2);
        c.zero_e = rocke_b_vec_extract(b, z2, 0);
    }

    /* ---- weights: A fragments of every (m tile, step), resident. */
    c.a_frags.assign(c.TM, std::vector<rocke_value_t*>());
    for(int m = 0; m < c.TM; ++m)
    {
        rocke_value_t* k;
        {
            rocke_value_t* inner = tile_lane(b, c.wave_k, c.TM, m, c.c16, l16);
            k = rocke_b_add(b, c.k_blk, inner);
        }
        rocke_value_t* k_ok = rocke_b_cmp_lt(b, k, c.c_K);
        rocke_value_t* k_row;
        {
            rocke_value_t* x = ci(b, c.NC);
            k_row = rocke_b_mul(b, k, x);
        }
        for(int t = 0; t < c.STEPS; ++t)
        {
            rocke_value_t* elems[8];
            for(int i = 0; i < 8; ++i)
            {
                rocke_value_t* j;
                {
                    rocke_value_t* x = ci(b, (int64_t)t * FSM_MFMA_K + i);
                    j = rocke_b_add(b, lane_j0, x);
                }
                rocke_value_t* ok;
                {
                    rocke_value_t* x = ci(b, c.NC);
                    rocke_value_t* y = rocke_b_cmp_lt(b, j, x);
                    ok = rocke_b_land(b, k_ok, y);
                }
                rocke_value_t* idx = rocke_b_add(b, k_row, j);
                elems[i]
                    = rocke_b_masked_global_load(b, params->Bp, idx, ok, c.zero_e, c.io_type, 2);
            }
            c.a_frags[m].push_back(rocke_b_vec_pack(b, elems, 8, c.io_type));
        }
    }

    /* ---- the window offset of each fragment element, relative to the
     * pixel's (tap 0, channel 0) element; past-the-filter indices read 0. */
    rocke_value_t* c_kwc = ci(b, (int64_t)KW * c.C);
    c.c_C = ci(b, c.C);
    c.win_off.assign(c.STEPS, std::vector<rocke_value_t*>());
    for(int t = 0; t < c.STEPS; ++t)
    {
        for(int i = 0; i < 8; ++i)
        {
            rocke_value_t* j;
            {
                rocke_value_t* x = ci(b, (int64_t)t * FSM_MFMA_K + i);
                j = rocke_b_add(b, lane_j0, x);
            }
            rocke_value_t* r = rocke_b_div(b, j, c_kwc);
            rocke_value_t* sc = rocke_b_mod(b, j, c_kwc);
            rocke_value_t* off;
            {
                rocke_value_t* x = ci(b, c.WPS);
                rocke_value_t* y = rocke_b_mul(b, r, x);
                off = rocke_b_add(b, y, sc);
            }
            rocke_value_t* x = ci(b, c.NC);
            rocke_value_t* y = rocke_b_cmp_lt(b, j, x);
            c.win_off[t].push_back(rocke_b_select(b, y, off, c.c0));
        }
    }

    /* ---- per-lane pixel geometry (tile independent). */
    c.lane_ty.assign(c.TN, NULL);
    c.lane_tx.assign(c.TN, NULL);
    c.lane_base.assign(c.TN, NULL);
    for(int tn = 0; tn < c.TN; ++tn)
    {
        rocke_value_t* q = tile_lane(b, wave_p, c.TN, tn, c.c16, l16);
        {
            rocke_value_t* x = ci(b, c.TOW);
            c.lane_ty[tn] = rocke_b_div(b, q, x);
            rocke_value_t* y = ci(b, c.TOW);
            c.lane_tx[tn] = rocke_b_mod(b, q, y);
        }
        {
            rocke_value_t* x = ci(b, c.S);
            rocke_value_t* y = rocke_b_mul(b, c.lane_ty[tn], x);
            rocke_value_t* z = ci(b, c.WPS);
            rocke_value_t* u = rocke_b_mul(b, y, z);
            rocke_value_t* v = ci(b, (int64_t)c.S * c.C);
            rocke_value_t* w = rocke_b_mul(b, c.lane_tx[tn], v);
            rocke_value_t* s = rocke_b_add(b, u, w);
            c.lane_base[tn] = rocke_b_add(b, s, c.c8);
        }
    }

    /* X rounded up to whole dwords (the bounds are checked per dword): the
     * last element of an odd-length X is read, the halfword past it masked. */
    {
        rocke_value_t* x = ci(b, 3);
        rocke_value_t* y = rocke_b_add_nuw(b, params->A_bytes, x);
        rocke_value_t* z = ci(b, -4);
        rocke_value_t* nbytes = rocke_b_land(b, y, z);
        c.a_rsrc = rocke_b_buffer_rsrc(b, params->A, nbytes);
    }
    c.c_sent = ci(b, FSM_OOB_OFFSET);

    rocke_value_t* first;
    {
        rocke_value_t* bx = rocke_b_block_id_x(b);
        rocke_value_t* x = ci(b, T);
        first = rocke_b_mul(b, bx, x);
    }
    if(spec->prefetch)
    {
        /* Tile t+1's loads are issued before tile t's compute and stores. */
        std::vector<rocke_value_t*> tiles(T);
        for(int t = 0; t < T; ++t)
        {
            if(t)
            {
                rocke_value_t* x = ci(b, t);
                tiles[t] = rocke_b_add(b, first, x);
            }
            else
                tiles[t] = first;
        }
        std::vector<sc_geom> geoms;
        for(int t = 0; t < T; ++t)
            geoms.push_back(sc_tile_geom(c, tiles[t]));
        std::vector<sc_load> loads = sc_load_window(c, geoms[0]);
        for(int t = 0; t < T; ++t)
        {
            if(t)
                rocke_b_sync_lds_only(b);
            sc_commit_window(c, geoms[t], loads);
            rocke_b_sync_lds_only(b);
            if(t + 1 < T)
                loads = sc_load_window(c, geoms[t + 1]);
            rocke_value_t* cond = rocke_b_cmp_lt(b, tiles[t], n_tiles);
            scf_if_enter(b, cond);
            sc_compute_store(c, geoms[t]);
            rocke_b_region_leave(b);
        }
    }
    else
    {
        for(int t = 0; t < T; ++t)
        {
            rocke_value_t* tile = first;
            if(t)
            {
                rocke_value_t* x = ci(b, t);
                tile = rocke_b_add(b, first, x);
                /* The previous tile's gathers before the restage. */
                rocke_b_sync_lds_only(b);
            }
            rocke_value_t* cond = rocke_b_cmp_lt(b, tile, n_tiles);
            scf_if_enter(b, cond);
            {
                sc_geom g = sc_tile_geom(c, tile);
                std::vector<sc_load> loads = sc_load_window(c, g);
                sc_commit_window(c, g, loads);
                rocke_b_sync_lds_only(b);
                sc_compute_store(c, g);
            }
            rocke_b_region_leave(b);
        }
    }
    if(!rocke_ir_builder_ok(b))
        return NULL;
    return b->kernel;
}

/* ===================================================================== *
 *  build_fwd_smallk_mfma
 * ===================================================================== */

/* _load(rsrc, (ok, base), c_off): one 16-byte fragment of 8 channels. */
rocke_value_t* sk_load(rocke_ir_builder_t* b,
                       bool bf16,
                       rocke_value_t* rsrc,
                       rocke_value_t* ok,
                       rocke_value_t* base,
                       rocke_value_t* c_off,
                       rocke_value_t* c_lane,
                       rocke_value_t* c_C,
                       rocke_value_t* c2,
                       rocke_value_t* c_sent,
                       rocke_value_t* c0)
{
    rocke_value_t* cc = rocke_b_add(b, c_off, c_lane);
    rocke_value_t* valid;
    {
        rocke_value_t* x = rocke_b_cmp_lt(b, cc, c_C);
        valid = rocke_b_land(b, ok, x);
    }
    rocke_value_t* voff;
    {
        rocke_value_t* x = rocke_b_mul(b, cc, c2);
        rocke_value_t* y = rocke_b_add(b, base, x);
        voff = rocke_b_select(b, valid, y, c_sent);
    }
    return load16(b, bf16, rsrc, voff, c0);
}

rocke_kernel_def_t*
    build_smallk(rocke_ir_builder_t* b, const rocke_fwd_smallk_mfma_spec_t* spec, const char* arch)
{
    char reason[512];
    if(!rocke_fwd_smallk_mfma_is_valid_spec(spec, arch, reason, sizeof reason))
    {
        std::string msg = std::string("invalid FwdSmallkMfmaSpec for ") + arch + ": " + reason;
        rocke_i_set_err_msg(b, ROCKE_ERR_VALUE, msg.c_str());
        return NULL;
    }
    const rocke_direct_conv_problem_t* p = &spec->problem;
    const int KH = p->KH, KW = p->KW, S = p->stride, PAD = p->PAD;
    const int PAD_W = rocke_direct_conv_problem_pad_w(p), K = p->kpg;
    const int RS = KH * KW;
    const int NCOL = sk_ncols(spec);
    const int NT = sk_n_tiles(spec);
    const int MT = sk_m_tiles(spec);
    const bool bf16 = rocke_direct_conv_problem_is_bf16(p);
    const rocke_type_t* io_type = rocke_b_io_ir_type(b, rocke_direct_conv_problem_dtype(p));
    if(io_type == NULL)
        return NULL;

    const int WAVE = spec->wave_size;
    const int THREADS = rocke_fwd_smallk_mfma_threads_per_block(spec);
    const int OH = spec->out_h, OW = spec->out_w;
    const int DW = sk_win_w(spec);
    const int NPIX = sk_win_h(spec) * DW;
    const int ZP = sk_z_pitch(spec);

    rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", THREADS);
    rocke_dconv_params_t pr;
    rocke_dconv_emit_params(b, &pr, "fwd", io_type);
    const rocke_dconv_params_t* params = &pr;

    rocke_value_t* c0 = ci(b, 0);
    rocke_value_t* c2 = ci(b, 2);
    rocke_value_t* c8 = ci(b, 8);
    rocke_value_t* c16 = ci(b, 16);
    rocke_value_t* c_C = params->p_total_c;
    rocke_value_t* c_H = params->p_Hi;
    rocke_value_t* c_W = params->p_Wi;
    rocke_value_t* c_Ho = params->p_Ho;
    rocke_value_t* c_Wo = params->p_Wo;
    rocke_value_t* c_sent = ci(b, FSM_OOB_OFFSET);

    rocke_value_t* tid = rocke_b_thread_id_x(b);
    rocke_value_t *wave_id, *lane;
    {
        rocke_value_t* c_wave = ci(b, WAVE);
        wave_id = rocke_b_div(b, tid, c_wave);
        lane = rocke_b_mod(b, tid, c_wave);
    }
    rocke_value_t* l16 = rocke_b_mod(b, lane, c16);
    rocke_value_t* lgrp = rocke_b_div(b, lane, c16);
    rocke_value_t* c_lane = rocke_b_mul(b, lgrp, c8);

    rocke_value_t* n_ht = n_tiles_of(b, c_Ho, OH);
    rocke_value_t* n_wt = n_tiles_of(b, c_Wo, OW);
    rocke_value_t* tile = rocke_b_block_id_x(b);
    rocke_value_t* tw = rocke_b_mod(b, tile, n_wt);
    rocke_value_t* rest = rocke_b_div(b, tile, n_wt);
    rocke_value_t* th = rocke_b_mod(b, rest, n_ht);
    rocke_value_t* n_img = rocke_b_div(b, rest, n_ht);
    rocke_value_t *oh0, *ow0, *h_start, *w_start;
    {
        rocke_value_t* x = ci(b, OH);
        oh0 = rocke_b_mul(b, th, x);
        rocke_value_t* y = ci(b, OW);
        ow0 = rocke_b_mul(b, tw, y);
    }
    {
        rocke_value_t* x = ci(b, S);
        rocke_value_t* y = rocke_b_mul(b, oh0, x);
        rocke_value_t* z = ci(b, PAD);
        h_start = rocke_b_sub(b, y, z);
    }
    {
        rocke_value_t* x = ci(b, S);
        rocke_value_t* y = rocke_b_mul(b, ow0, x);
        rocke_value_t* z = ci(b, PAD_W);
        w_start = rocke_b_sub(b, y, z);
    }

    rocke_value_t* a_rsrc = rocke_b_buffer_rsrc(b, params->A, params->A_bytes);
    rocke_value_t* b_rsrc = rocke_b_buffer_rsrc(b, params->Bp, params->B_bytes);

    /* ---- A rows: the window pixel of each of this wave's pixel tiles. */
    rocke_value_t* img_off = rocke_b_mul(b, n_img, params->p_A_stride_n);
    std::vector<rocke_value_t*> a_ok(MT), a_base(MT);
    for(int mt = 0; mt < MT; ++mt)
    {
        rocke_value_t* q = tile_lane(b, wave_id, MT, mt, c16, l16);
        rocke_value_t* hr;
        {
            rocke_value_t* x = ci(b, DW);
            rocke_value_t* y = rocke_b_div(b, q, x);
            hr = rocke_b_add(b, h_start, y);
        }
        rocke_value_t* wc;
        {
            rocke_value_t* x = ci(b, DW);
            rocke_value_t* y = rocke_b_mod(b, q, x);
            wc = rocke_b_add(b, w_start, y);
        }
        {
            rocke_value_t* x = ci(b, NPIX);
            rocke_value_t* y = rocke_b_cmp_lt(b, q, x);
            rocke_value_t* z = rocke_b_cmp_ge(b, hr, c0);
            rocke_value_t* u = rocke_b_cmp_lt(b, hr, c_H);
            rocke_value_t* v = rocke_b_land(b, z, u);
            rocke_value_t* w = rocke_b_cmp_ge(b, wc, c0);
            rocke_value_t* s = rocke_b_cmp_lt(b, wc, c_W);
            rocke_value_t* t = rocke_b_land(b, w, s);
            rocke_value_t* r = rocke_b_land(b, v, t);
            a_ok[mt] = rocke_b_land(b, y, r);
        }
        {
            rocke_value_t* x = rocke_b_mul(b, hr, params->p_A_stride_hi);
            rocke_value_t* y = rocke_b_mul(b, wc, params->p_A_stride_wi);
            rocke_value_t* z = rocke_b_add(b, x, y);
            rocke_value_t* off = rocke_b_add(b, img_off, z);
            a_base[mt] = rocke_b_mul(b, off, c2);
        }
    }
    /* ---- B columns: GEMM column n = (r*KW + s)*K + k reads W[k, r, s, :]. */
    std::vector<rocke_value_t*> b_ok(NT), b_base(NT);
    for(int nt = 0; nt < NT; ++nt)
    {
        rocke_value_t* n;
        {
            rocke_value_t* x = ci(b, (int64_t)nt * 16);
            n = rocke_b_add(b, x, l16);
        }
        rocke_value_t *rs, *kk;
        {
            rocke_value_t* x = ci(b, K);
            rs = rocke_b_div(b, n, x);
            rocke_value_t* y = ci(b, K);
            kk = rocke_b_mod(b, n, y);
        }
        {
            rocke_value_t* x = ci(b, NCOL);
            b_ok[nt] = rocke_b_cmp_lt(b, n, x);
        }
        {
            rocke_value_t* x = ci(b, RS);
            rocke_value_t* y = rocke_b_mul(b, kk, x);
            rocke_value_t* z = rocke_b_add(b, y, rs);
            rocke_value_t* off = rocke_b_mul(b, z, c_C);
            b_base[nt] = rocke_b_mul(b, off, c2);
        }
    }

    /* ---- Z = X * W^T over C, 32 channels per step. */
    const int n_acc = MT * NT;
    std::vector<rocke_value_t*> init(n_acc);
    for(int i = 0; i < n_acc; ++i)
        init[i] = rocke_b_zero_vec_f32(b, 4);
    std::vector<std::string> names(n_acc);
    std::vector<rocke_iter_arg_t> iargs(n_acc);
    for(int i = 0; i < n_acc; ++i)
    {
        names[i] = "zacc" + std::to_string(i);
        iargs[i].name = names[i].c_str();
        iargs[i].init = init[i];
    }
    rocke_value_t* c32 = ci(b, 32);
    rocke_for_t loop
        = rocke_b_scf_for_iter(b, c0, c_C, c32, iargs.data(), n_acc, "cs", false, true);
    if(!rocke_ir_builder_ok(b))
        return NULL;
    rocke_b_region_enter(b, loop.body);
    {
        std::vector<rocke_value_t*> bf(NT), af(MT);
        for(int nt = 0; nt < NT; ++nt)
            bf[nt] = sk_load(
                b, bf16, b_rsrc, b_ok[nt], b_base[nt], loop.iv, c_lane, c_C, c2, c_sent, c0);
        for(int mt = 0; mt < MT; ++mt)
            af[mt] = sk_load(
                b, bf16, a_rsrc, a_ok[mt], a_base[mt], loop.iv, c_lane, c_C, c2, c_sent, c0);
        std::vector<rocke_value_t*> nw(loop.iter_vars, loop.iter_vars + n_acc);
        for(int mt = 0; mt < MT; ++mt)
            for(int nt = 0; nt < NT; ++nt)
            {
                int i = mt * NT + nt;
                nw[i] = mfma32(b, bf16, af[mt], bf[nt], nw[i]);
            }
        rocke_b_scf_yield(b, nw.data(), n_acc);
    }
    rocke_b_region_leave(b);
    rocke_value_t* const* accs = loop.op->results;

    /* ---- Z to LDS (fp32): lane holds column l16 and pixel rows lgrp*4 .. +3. */
    rocke_value_t* zl;
    {
        int shape[2] = {1, (int)(rocke_fwd_smallk_mfma_lds_bytes(spec) / 4)};
        zl = rocke_b_smem_alloc(b, rocke_f32(), shape, 2, "z_lds");
    }
    for(int mt = 0; mt < MT; ++mt)
    {
        rocke_value_t* q0;
        {
            rocke_value_t* x = tile_row(b, wave_id, MT, mt, c16);
            rocke_value_t* y = ci(b, 4);
            rocke_value_t* z = rocke_b_mul(b, lgrp, y);
            q0 = rocke_b_add(b, x, z);
        }
        for(int nt = 0; nt < NT; ++nt)
        {
            rocke_value_t* col;
            {
                rocke_value_t* x = ci(b, (int64_t)nt * 16);
                col = rocke_b_add(b, x, l16);
            }
            rocke_value_t* acc = accs[mt * NT + nt];
            for(int i = 0; i < 4; ++i)
            {
                rocke_value_t* zi;
                {
                    rocke_value_t* x = ci(b, i);
                    rocke_value_t* y = rocke_b_add(b, q0, x);
                    rocke_value_t* z = ci(b, ZP);
                    rocke_value_t* u = rocke_b_mul(b, y, z);
                    zi = rocke_b_add(b, u, col);
                }
                rocke_value_t* idx[2] = {c0, zi};
                rocke_value_t* v = rocke_b_vec_extract(b, acc, i);
                rocke_b_smem_store_vN(b, zl, idx, 2, v, 1);
            }
        }
    }
    rocke_b_sync_lds_only(b);

    /* ---- overlap-add: one output pixel (all K channels) per thread and pass. */
    const int n_it = (int)rocke_ceil_div((long long)OH * OW, THREADS);
    for(int it = 0; it < n_it; ++it)
    {
        rocke_value_t* px;
        {
            rocke_value_t* x = ci(b, (int64_t)it * THREADS);
            px = rocke_b_add(b, tid, x);
        }
        /* The last pass's threads past the tile read the tile's last pixel, so
         * their Z reads stay in the LDS copy; the store below skips them. */
        rocke_value_t* pxr = px;
        if((long long)(it + 1) * THREADS > (long long)OH * OW)
        {
            rocke_value_t* x = ci(b, (int64_t)OH * OW - 1);
            pxr = rocke_b_smin(b, px, x);
        }
        rocke_value_t *oy, *ox;
        {
            rocke_value_t* x = ci(b, OW);
            oy = rocke_b_div(b, pxr, x);
            rocke_value_t* y = ci(b, OW);
            ox = rocke_b_mod(b, pxr, y);
        }
        rocke_value_t* oh = rocke_b_add(b, oh0, oy);
        rocke_value_t* ow = rocke_b_add(b, ow0, ox);
        rocke_value_t* ok;
        {
            rocke_value_t* x = ci(b, (int64_t)OH * OW);
            rocke_value_t* y = rocke_b_cmp_lt(b, px, x);
            rocke_value_t* z = rocke_b_cmp_lt(b, oh, c_Ho);
            rocke_value_t* u = rocke_b_cmp_lt(b, ow, c_Wo);
            rocke_value_t* v = rocke_b_land(b, z, u);
            ok = rocke_b_land(b, y, v);
        }
        rocke_value_t* q0;
        {
            rocke_value_t* x = ci(b, S);
            rocke_value_t* y = rocke_b_mul(b, oy, x);
            rocke_value_t* z = ci(b, DW);
            rocke_value_t* u = rocke_b_mul(b, y, z);
            rocke_value_t* v = ci(b, S);
            rocke_value_t* w = rocke_b_mul(b, ox, v);
            q0 = rocke_b_add(b, u, w);
        }
        std::vector<rocke_value_t*> sums(K);
        for(int k = 0; k < K; ++k)
        {
            rocke_value_t* acc = NULL;
            for(int r = 0; r < KH; ++r)
                for(int s = 0; s < KW; ++s)
                {
                    rocke_value_t* zi;
                    {
                        rocke_value_t* x = ci(b, (int64_t)r * DW + s);
                        rocke_value_t* y = rocke_b_add(b, q0, x);
                        rocke_value_t* z = ci(b, ZP);
                        rocke_value_t* u = rocke_b_mul(b, y, z);
                        rocke_value_t* v = ci(b, (int64_t)(r * KW + s) * K + k);
                        zi = rocke_b_add(b, u, v);
                    }
                    rocke_value_t* idx[2] = {c0, zi};
                    rocke_value_t* lv = rocke_b_smem_load_vN(b, zl, idx, 2, rocke_f32(), 1);
                    rocke_value_t* v = rocke_b_vec_extract(b, lv, 0);
                    acc = (acc == NULL) ? v : rocke_b_fadd(b, acc, v);
                }
            sums[k] = acc;
        }
        rocke_value_t* y_off = nhw_off(
            b, n_img, params->p_D_stride_n, oh, params->p_D_stride_ho, ow, params->p_D_stride_wo);
        scf_if_enter(b, ok);
        if(K == 2 || K == 4 || K == 8)
        {
            rocke_value_t* packed = rocke_b_vec_pack(b, sums.data(), K, rocke_f32());
            rocke_value_t* val = rocke_dconv_trunc_f32(b, bf16, packed);
            rocke_b_global_store_vN(b, params->D, y_off, val, K, 2 * K);
        }
        else
        {
            for(int k = 0; k < K; ++k)
            {
                rocke_value_t* x = ci(b, k);
                rocke_value_t* o = rocke_b_add(b, y_off, x);
                rocke_value_t* val = rocke_dconv_trunc_f32_elem(b, bf16, sums[k]);
                rocke_b_global_store(b, params->D, o, val, 2);
            }
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

rocke_kernel_def_t* rocke_build_fwd_smallc_mfma(rocke_ir_builder_t* b,
                                                const rocke_fwd_smallc_mfma_spec_t* spec,
                                                const char* arch)
{
    if(b == NULL || spec == NULL)
        return NULL;
    return build_smallc(b, spec, arch ? arch : "gfx950");
}

rocke_kernel_def_t* rocke_build_fwd_smallk_mfma(rocke_ir_builder_t* b,
                                                const rocke_fwd_smallk_mfma_spec_t* spec,
                                                const char* arch)
{
    if(b == NULL || spec == NULL)
        return NULL;
    return build_smallk(b, spec, arch ? arch : "gfx950");
}

rocke_kernel_def_t* rocke_build_fwd_smallc_mfma_new(rocke_ir_builder_t* b,
                                                    const rocke_fwd_smallc_mfma_spec_t* spec,
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
            rocke_i_set_err_msg(b, ROCKE_ERR_VALUE, "build_fwd_smallc_mfma: null spec");
            return NULL;
        }
        if(rocke_fwd_smallc_mfma_kernel_name(spec, name, sizeof(name)) != ROCKE_OK)
        {
            rocke_i_set_err_msg(
                b, ROCKE_ERR_VALUE, "build_fwd_smallc_mfma: kernel name does not fit");
            return NULL;
        }
        if(rocke_ir_builder_init(b, name) != ROCKE_OK)
            return NULL;
        return rocke_build_fwd_smallc_mfma(b, spec, arch);
    });
}

rocke_kernel_def_t* rocke_build_fwd_smallk_mfma_new(rocke_ir_builder_t* b,
                                                    const rocke_fwd_smallk_mfma_spec_t* spec,
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
            rocke_i_set_err_msg(b, ROCKE_ERR_VALUE, "build_fwd_smallk_mfma: null spec");
            return NULL;
        }
        if(rocke_fwd_smallk_mfma_kernel_name(spec, name, sizeof(name)) != ROCKE_OK)
        {
            rocke_i_set_err_msg(
                b, ROCKE_ERR_VALUE, "build_fwd_smallk_mfma: kernel name does not fit");
            return NULL;
        }
        if(rocke_ir_builder_init(b, name) != ROCKE_OK)
            return NULL;
        return rocke_build_fwd_smallk_mfma(b, spec, arch);
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
        rocke_spec_set_reason(err, err_cap, "lower_to_llvm: null spec/out");
        return ROCKE_ERR_VALUE;
    }
    if(arch == NULL)
        arch = "gfx950";
    kernel = build_new(&b, spec, arch);
    if(kernel == NULL)
    {
        const char* m = rocke_ir_builder_error(&b);
        st = rocke_ir_builder_status(&b);
        rocke_spec_set_reason(err, err_cap, (m != NULL && m[0] != '\0') ? m : who);
        rocke_ir_builder_free(&b);
        return (st == ROCKE_OK) ? ROCKE_ERR_VALUE : st;
    }
    st = rocke_lower_kernel_to_llvm_ex(kernel, flavor, arch, out_ll, err, err_cap);
    rocke_ir_builder_free(&b);
    return st;
}

} // namespace

rocke_status_t rocke_fwd_smallc_mfma_lower_to_llvm(const rocke_fwd_smallc_mfma_spec_t* spec,
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
                        rocke_build_fwd_smallc_mfma_new,
                        "build_fwd_smallc_mfma failed");
}

rocke_status_t rocke_fwd_smallk_mfma_lower_to_llvm(const rocke_fwd_smallk_mfma_spec_t* spec,
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
                        rocke_build_fwd_smallk_mfma_new,
                        "build_fwd_smallk_mfma failed");
}
