// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * conv_direct_dw_wgrad.cpp -- C++ engine port of
 * library/kernels/common/conv_direct_dw_wgrad.py: the partial-slot fold
 * (build_dw_wgrad_fold), the small-C depthwise wgrad with lanes along output
 * columns (build_dw_wgrad_cols) and the depthwise wgrad with LDS row rings
 * (build_dw_wgrad_lds).
 *
 * Every IR op (including each arith.constant) is emitted in the order the
 * Python builder emits it; see the contract in
 * rocke/instance_conv_direct_wgrad_internal.hpp.
 */
#include "rocke/instance_conv_direct_dw_wgrad.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <map>
#include <string>
#include <vector>

#include "rocke/arch_target.h" /* rocke_set_unknown_arch_reason */
#include "rocke/helper_rocke.core.arch.h" /* rocke_archtarget_from_gfx */
#include "rocke/helper_rocke.helpers.spec.h" /* rocke_kernel_name_join */
#include "rocke/instance_conv_direct_grouped_internal.h" /* rocke_dconv_emit_params */
#include "rocke/instance_conv_direct_wgrad_internal.hpp"

using namespace rocke_dwg;

namespace
{

constexpr int kFoldSlices = 16;
constexpr int kFoldCols = 64;
/* Partition loads each slice keeps in flight. */
constexpr int kFoldUnroll = 4;

const char* bf16_flag[1] = {"bf16"};

} // namespace

/* ===================================================================== *
 *  Fold: dW[c, tap] = sum_part ws[part][tap][c]
 * ===================================================================== */

extern "C" rocke_dw_wgrad_fold_spec_t rocke_dw_wgrad_fold_spec_default(void)
{
    rocke_dw_wgrad_fold_spec_t s;
    memset(&s, 0, sizeof(s));
    s.name = "dw_wgrad_fold";
    return s;
}

extern "C" rocke_status_t
    rocke_dw_wgrad_fold_kernel_name(const rocke_dw_wgrad_fold_spec_t* s, char* out, size_t out_cap)
{
    if(s == NULL || out == NULL || out_cap == 0)
    {
        return ROCKE_ERR_VALUE;
    }
    char t_buf[24];
    snprintf(t_buf, sizeof t_buf, "t%d", s->taps);
    const char* parts[1] = {t_buf};
    return rocke_kernel_name_join(s->name, parts, 1, NULL, NULL, 0, out, out_cap, NULL);
}

extern "C" void rocke_dw_wgrad_fold_launch_geometry(const rocke_dw_wgrad_fold_spec_t* s,
                                                    int channels,
                                                    int grid[3],
                                                    int block[3])
{
    grid[0] = cdiv(channels * s->taps, kFoldCols);
    grid[1] = 1;
    grid[2] = 1;
    block[0] = kFoldSlices * kWave;
    block[1] = 1;
    block[2] = 1;
}

namespace
{

/* Each block owns 64 outputs; 16 waves split the partition axis, meet in LDS. */
rocke_kernel_def_t* build_fold(rocke_ir_builder_t* b, const rocke_dw_wgrad_fold_spec_t* spec)
{
    Em e{b};
    rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", kFoldSlices * kWave);
    rocke_param_opts_t ro;
    memset(&ro, 0, sizeof(ro));
    ro.noalias = ro.noalias_set = true;
    ro.readonly = ro.readonly_set = true;
    ro.align = 16;
    ro.align_set = true;
    rocke_param_opts_t wo;
    memset(&wo, 0, sizeof(wo));
    wo.noalias = wo.noalias_set = true;
    wo.writeonly = wo.writeonly_set = true;
    wo.align = 4;
    wo.align_set = true;
    V ws = rocke_b_param(b, "A", rocke_ptr_type(b, rocke_f32(), "global"), &ro);
    V dw = rocke_b_param(b, "D", rocke_ptr_type(b, rocke_f32(), "global"), &wo);
    rocke_b_param(b, "A_bytes", rocke_i32(), NULL);
    rocke_b_param(b, "D_bytes", rocke_i32(), NULL);
    V parts = rocke_b_param(b, "parts", rocke_i32(), NULL);
    V chans = rocke_b_param(b, "channels", rocke_i32(), NULL);
    if(!rocke_ir_builder_ok(b))
    {
        return NULL;
    }

    V tid = rocke_b_thread_id_x(b);
    V lane = e.modi(tid, kWave);
    V slc = e.divi(tid, kWave);
    V n_out = e.muli(chans, spec->taps);
    V bx = rocke_b_block_id_x(b);
    V o = e.add(e.muli(bx, kFoldCols), lane);
    V o_ok = e.lt(o, n_out);
    V o_safe = e.select(o_ok, o, e.ci(0));
    V zero = rocke_b_const_f32(b, 0.0);
    const int U = kFoldUnroll;
    /* Partition p = slc + SLICES * (U * it + j). */
    V n_it = e.cdivi(parts, kFoldSlices * U);
    V lo = e.ci(0);
    V step = e.ci(1);
    std::vector<std::string> names;
    for(int j = 0; j < U; ++j)
    {
        names.push_back("s" + std::to_string(j));
    }
    std::vector<V> res = e.for_iter(
        lo, n_it, step, names, std::vector<V>(U, zero), "it", [&](V it, std::vector<V> acc) {
            for(int j = 0; j < U; ++j)
            {
                V pidx = e.addi(e.muli(it, U), j);
                pidx = e.add(slc, e.muli(pidx, kFoldSlices));
                V ok = e.lt(pidx, parts);
                V sel = e.select(ok, pidx, e.ci(0));
                V idx = e.add(e.mul(sel, n_out), o_safe);
                V v = rocke_b_global_load_f32(b, ws, idx, 4);
                acc[j] = rocke_b_fadd(b, acc[j], e.select(ok, v, zero));
            }
            return acc;
        });
    if((int)res.size() != U)
    {
        return NULL;
    }
    V tot = res[0];
    for(int j = 1; j < U; ++j)
    {
        tot = rocke_b_fadd(b, tot, res[j]);
    }
    V red = e.smem_alloc_1d(rocke_f32(), kFoldSlices * kWave, "fold");
    e.smem_store_f32(red, e.ci(0), tid, tot, 1);
    rocke_b_sync(b);
    V cond = e.land(e.eqi(slc, 0), o_ok);
    e.if_(cond, [&]() {
        V t = NULL;
        for(int q = 0; q < kFoldSlices; ++q)
        {
            V c0 = e.ci(0);
            V v = e.smem_load_f32(red, c0, e.addi(lane, q * kWave), 1);
            v = rocke_b_vec_extract(b, v, 0);
            t = (t == NULL) ? v : rocke_b_fadd(b, t, v);
        }
        V tap = e.div(o, chans);
        V c = e.mod(o, chans);
        V off = e.add(e.muli(c, spec->taps), tap);
        rocke_b_global_store(b, dw, off, t, 4);
    });
    return rocke_ir_builder_ok(b) ? b->kernel : NULL;
}

} // namespace

extern "C" rocke_kernel_def_t* rocke_build_dw_wgrad_fold(rocke_ir_builder_t* b,
                                                         const rocke_dw_wgrad_fold_spec_t* spec,
                                                         const char* arch)
{
    (void)arch;
    if(b == NULL || spec == NULL)
    {
        return NULL;
    }
    return build_fold(b, spec);
}

extern "C" rocke_kernel_def_t* rocke_build_dw_wgrad_fold_new(rocke_ir_builder_t* b,
                                                             const rocke_dw_wgrad_fold_spec_t* spec,
                                                             const char* arch)
{
    return build_new(
        b,
        spec != NULL,
        "build_dw_wgrad_fold",
        [&](char* out, size_t cap) { return rocke_dw_wgrad_fold_kernel_name(spec, out, cap); },
        [&]() { return rocke_build_dw_wgrad_fold(b, spec, arch); });
}

extern "C" rocke_status_t rocke_dw_wgrad_fold_lower_to_llvm(const rocke_dw_wgrad_fold_spec_t* spec,
                                                            const char* arch,
                                                            rocke_llvm_flavor_t flavor,
                                                            char** out_ll,
                                                            char* err,
                                                            size_t err_cap)
{
    if(arch == NULL)
    {
        arch = "gfx950";
    }
    return lower_to_llvm(
        spec != NULL,
        arch,
        flavor,
        out_ll,
        err,
        err_cap,
        "build_dw_wgrad_fold failed",
        [&](rocke_ir_builder_t* b) { return rocke_build_dw_wgrad_fold_new(b, spec, arch); });
}

/* ===================================================================== *
 *  Small-C depthwise wgrad: lanes along output columns
 *
 *  A lane owns filter row r (block x) and a strip of `strip` output columns,
 *  keeps KW x C accumulators, and per output row loads the dY strip and the
 *  X row segment it touches as whole 16-byte vectors at a build-time phase.
 *  Block y is a partition over (image, output row, column group) units.
 * ===================================================================== */

extern "C" rocke_dw_wgrad_cols_spec_t rocke_dw_wgrad_cols_spec_default(void)
{
    rocke_dw_wgrad_cols_spec_t s;
    memset(&s, 0, sizeof(s));
    s.problem = rocke_direct_conv_problem_default();
    s.name = "dw_wgrad_cols";
    s.strip = 8;
    s.threads = 64;
    s.target_blocks = 2048;
    return s;
}

static int cols_n_units(const rocke_dw_wgrad_cols_spec_t* s)
{
    const rocke_direct_conv_problem_t* p = &s->problem;
    return cdiv(p->N * problem_ho(p) * (problem_wo(p) / s->strip), s->threads);
}

extern "C" int rocke_dw_wgrad_cols_n_partitions(const rocke_dw_wgrad_cols_spec_t* s)
{
    int a = cols_n_units(s);
    int c = cdiv(s->target_blocks, s->problem.KH);
    return a < c ? a : c;
}

extern "C" long rocke_dw_wgrad_cols_workspace_bytes(const rocke_dw_wgrad_cols_spec_t* s)
{
    const rocke_direct_conv_problem_t* p = &s->problem;
    return (long)rocke_dw_wgrad_cols_n_partitions(s) * p->KH * p->KW * p->groups * 4;
}

extern "C" void rocke_dw_wgrad_cols_launch_geometry(const rocke_dw_wgrad_cols_spec_t* s,
                                                    int grid[3],
                                                    int block[3])
{
    grid[0] = s->problem.KH;
    grid[1] = rocke_dw_wgrad_cols_n_partitions(s);
    grid[2] = 1;
    block[0] = s->threads;
    block[1] = 1;
    block[2] = 1;
}

extern "C" long rocke_dw_wgrad_cols_lds_bytes(const rocke_dw_wgrad_cols_spec_t* s)
{
    /* The block reduction buffer: KW x C fp32 accumulators per thread. */
    return (long)s->threads * s->problem.KW * s->problem.groups * 4;
}

extern "C" rocke_status_t
    rocke_dw_wgrad_cols_kernel_name(const rocke_dw_wgrad_cols_spec_t* s, char* out, size_t out_cap)
{
    if(s == NULL || out == NULL || out_cap == 0)
    {
        return ROCKE_ERR_VALUE;
    }
    const rocke_direct_conv_problem_t* p = &s->problem;
    char f_buf[64], st_buf[24], t_buf[24], tb_buf[24];
    snprintf(f_buf, sizeof f_buf, "f%dx%dp%ds%dc%d", p->KH, p->KW, p->PAD, p->stride, p->groups);
    snprintf(st_buf, sizeof st_buf, "st%d", s->strip);
    snprintf(t_buf, sizeof t_buf, "t%d", s->threads);
    snprintf(tb_buf, sizeof tb_buf, "tb%d", s->target_blocks);
    const char* parts[4] = {f_buf, st_buf, t_buf, tb_buf};
    int on[1] = {problem_is_bf16(p) ? 1 : 0};
    return rocke_kernel_name_join(s->name, parts, 4, bf16_flag, on, 1, out, out_cap, NULL);
}

extern "C" bool rocke_dw_wgrad_cols_is_valid_spec(const rocke_dw_wgrad_cols_spec_t* s,
                                                  const char* arch,
                                                  char* reason,
                                                  size_t reason_cap)
{
    (void)arch;
    char msg[ROCKE_ERR_MSG_CAP];
    if(s == NULL)
    {
        set_reason(reason, reason_cap, "null spec");
        return false;
    }
    const rocke_direct_conv_problem_t* p = &s->problem;
    if(dilation_reason(p, msg, sizeof msg))
    {
        set_reason(reason, reason_cap, msg);
        return false;
    }
    const char* why = NULL;
    int C = p->groups;
    if(p->cpg != 1 || p->kpg != 1)
    {
        why = "depthwise only";
    }
    else if(!problem_dtype_ok(p))
    {
        why = "fp16/bf16 only";
    }
    else if(p->groups > 8)
    {
        why = "small-C kernel: C <= 8";
    }
    else if(p->KW * p->groups > 96)
    {
        why = "too many accumulators";
    }
    else if(s->threads < kWave || s->threads % kWave || s->threads > 1024)
    {
        why = "threads must be a multiple of 64";
    }
    /* Whole 16-byte vectors at a build-time phase: the strip and the rows must
     * keep every vector start on an 8-element boundary. */
    else if((s->strip * C) % 8 || (s->strip * p->stride * C) % 8)
    {
        why = "strip * C (and strip * stride * C) must be multiples of 8";
    }
    else if((p->W * C) % 8 || (problem_wo(p) * C) % 8 || s->strip <= 0 || problem_wo(p) % s->strip)
    {
        why = "W*C and Wo*C must be multiples of 8 and Wo of the strip";
    }
    else if(s->target_blocks < 1)
    {
        why = "target_blocks >= 1";
    }
    else if(rocke_dw_wgrad_cols_lds_bytes(s) > 160 * 1024)
    {
        ROCKE_ERR_SNPRINTF(
            msg, sizeof msg, "LDS %ld B > 160 KiB", rocke_dw_wgrad_cols_lds_bytes(s));
        why = msg;
    }
    if(why != NULL)
    {
        set_reason(reason, reason_cap, why);
        return false;
    }
    set_reason(reason, reason_cap, "ok");
    return true;
}

namespace
{

rocke_kernel_def_t*
    build_cols(rocke_ir_builder_t* b, const rocke_dw_wgrad_cols_spec_t* spec, const char* arch)
{
    char reason[ROCKE_ERR_MSG_CAP];
    if(!rocke_dw_wgrad_cols_is_valid_spec(spec, arch, reason, sizeof reason))
    {
        fail_invalid(b, "invalid DwWgradColsSpec", reason);
        return NULL;
    }
    Em e{b};
    const rocke_direct_conv_problem_t& p = spec->problem;
    const int KH = p.KH, KW = p.KW, PAD = p.PAD, S = p.stride, C = p.groups;
    const int WS = spec->strip;
    const int T = spec->threads;
    const bool bf16 = problem_is_bf16(&p);
    const rocke_type_t* io_type = bf16 ? rocke_bf16() : rocke_f16();
    rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", T);
    rocke_dconv_params_t P;
    rocke_dconv_emit_params(b, &P, "wgrad", io_type);
    if(!rocke_ir_builder_ok(b))
    {
        return NULL;
    }

    V c0 = e.ci(0);
    V c1 = e.ci(1);
    V c2 = e.ci(2);
    V oob = e.ci(((int64_t)1 << 31) - 1);
    V tid = rocke_b_thread_id_x(b);
    V r_tap = rocke_b_block_id_x(b); /* filter row */
    V part = rocke_b_block_id_y(b);
    V strips_per_row = e.divi(P.p_Wo, WS);
    V n_strips = e.mul(e.mul(P.p_N, P.p_Ho), strips_per_row);
    V n_units = e.cdivi(n_strips, T);
    V n_part = e.smin(n_units, e.ci(cdiv(spec->target_blocks, KH)));
    V u_lo = e.div(e.mul(part, n_units), n_part);
    V u_hi = e.div(e.mul(e.add(part, c1), n_units), n_part);

    V a_rsrc = rocke_b_buffer_rsrc(b, P.A, P.A_bytes); /* dY */
    V b_rsrc = rocke_b_buffer_rsrc(b, P.Bp, P.B_bytes); /* X */
    const int X_COLS = (WS - 1) * S + KW;
    /* Window element e sits at vector element e + PH: the window starts PAD
     * columns left of an 8-aligned strip. */
    const int PH = pymod(-PAD * C, 8);
    const int n_dy_vec = WS * C / 8;
    const int n_x_vec = cdiv(PH + X_COLS * C, 8);

    auto load8 = [&](V rsrc, V elem_off, V ok) {
        V voff = e.select(ok, e.mul(elem_off, c2), oob);
        return bf16 ? rocke_b_buffer_load_vN_bf16(b, rsrc, voff, c0, 4)
                    : rocke_b_buffer_load_vN_f16(b, rsrc, voff, c0, 4);
    };
    auto elem = [&](const std::vector<V>& vecs, int el) {
        return rocke_b_cast_to_f32(b, rocke_b_vec_extract(b, vecs[el / 8], el % 8));
    };

    V zero = rocke_b_const_f32(b, 0.0);
    const int n_acc = KW * C;
    std::vector<V> accs = e.for_iter(
        u_lo,
        u_hi,
        c1,
        iter_names("acc", n_acc),
        std::vector<V>(n_acc, zero),
        "unit",
        [&](V unit, std::vector<V> acc) {
            V strip = e.add(e.muli(unit, T), tid);
            V live = e.lt(strip, n_strips);
            V cs = e.mod(strip, strips_per_row);
            V rest = e.div(strip, strips_per_row);
            V ho = e.mod(rest, P.p_Ho);
            V n_img = e.div(rest, P.p_Ho);
            V wo0 = e.muli(cs, WS);
            V hi = e.muli(ho, S);
            hi = e.add(hi, e.subi(r_tap, PAD));
            V hi_ok = e.ge(hi, c0);
            hi_ok = e.land(hi_ok, e.lt(hi, P.p_Hi));
            hi_ok = e.land(live, hi_ok);
            V wi0 = e.subi(e.muli(wo0, S), PAD);
            V dy_base = e.mul(n_img, P.p_A_stride_n);
            dy_base = e.add(dy_base, e.mul(ho, P.p_A_stride_hi));
            dy_base = e.add(dy_base, e.mul(wo0, P.p_A_stride_wi));
            V x_base = e.mul(n_img, P.p_B_stride_n);
            x_base = e.add(x_base, e.mul(hi, P.p_B_stride_hi));
            x_base = e.add(x_base, e.mul(wi0, P.p_B_stride_wi));
            std::vector<V> dy_vecs, x_vecs;
            for(int i = 0; i < n_dy_vec; ++i)
            {
                dy_vecs.push_back(load8(a_rsrc, e.addi(dy_base, 8 * i), live));
            }
            for(int i = 0; i < n_x_vec; ++i)
            {
                x_vecs.push_back(load8(b_rsrc, e.addi(x_base, 8 * i - PH), hi_ok));
            }
            std::vector<std::vector<V>> dy(WS);
            for(int j = 0; j < WS; ++j)
            {
                for(int c = 0; c < C; ++c)
                {
                    dy[j].push_back(elem(dy_vecs, j * C + c));
                }
            }
            std::vector<std::vector<V>> xs(X_COLS);
            for(int col = 0; col < X_COLS; ++col)
            {
                V wi = e.addi(wi0, col);
                V okc = e.ge(wi, c0);
                okc = e.land(okc, e.lt(wi, P.p_Wi));
                for(int c = 0; c < C; ++c)
                {
                    xs[col].push_back(e.select(okc, elem(x_vecs, PH + col * C + c), zero));
                }
            }
            for(int j = 0; j < WS; ++j)
            {
                for(int s = 0; s < KW; ++s)
                {
                    for(int c = 0; c < C; ++c)
                    {
                        int i = s * C + c;
                        acc[i] = rocke_b_fma(b, dy[j][c], xs[j * S + s][c], acc[i]);
                    }
                }
            }
            return acc;
        });
    if((int)accs.size() != n_acc)
    {
        return NULL;
    }

    /* Block reduction: every lane holds this block's r row; sum over lanes. */
    V red = e.smem_alloc_1d(rocke_f32(), T * n_acc, "red");
    for(int i = 0; i < n_acc; ++i)
    {
        V ki = e.ci(i);
        V idx = e.add(e.muli(ki, T), tid);
        e.smem_store_f32(red, c0, idx, accs[i], 1);
    }
    rocke_b_sync(b);
    /* Thread tid sums outputs tid, tid + T, ... (KW * C may exceed T). */
    for(int k = 0; k < cdiv(n_acc, T); ++k)
    {
        V o = k == 0 ? tid : e.addi(tid, k * T);
        e.if_(e.lti(o, n_acc), [&]() {
            V tot = NULL;
            for(int q = 0; q < T; ++q)
            {
                V idx = e.addi(e.muli(o, T), q);
                V v = rocke_b_vec_extract(b, e.smem_load_f32(red, c0, idx, 1), 0);
                tot = (tot == NULL) ? v : rocke_b_fadd(b, tot, v);
            }
            /* ws[part][tap][c], tap = r*KW + s; output o is (s, c) = divmod(o, C). */
            V s_i = e.divi(o, C);
            V c_i = e.modi(o, C);
            V tap = e.add(e.muli(r_tap, KW), s_i);
            V off = e.add(e.muli(part, KH * KW), tap);
            off = e.add(e.muli(off, C), c_i);
            rocke_b_global_store(b, P.D, off, tot, 4);
        });
    }
    return rocke_ir_builder_ok(b) ? b->kernel : NULL;
}

} // namespace

extern "C" rocke_kernel_def_t* rocke_build_dw_wgrad_cols(rocke_ir_builder_t* b,
                                                         const rocke_dw_wgrad_cols_spec_t* spec,
                                                         const char* arch)
{
    if(b == NULL || spec == NULL)
    {
        return NULL;
    }
    return build_cols(b, spec, arch ? arch : "gfx950");
}

extern "C" rocke_kernel_def_t* rocke_build_dw_wgrad_cols_new(rocke_ir_builder_t* b,
                                                             const rocke_dw_wgrad_cols_spec_t* spec,
                                                             const char* arch)
{
    return build_new(
        b,
        spec != NULL,
        "build_dw_wgrad_cols",
        [&](char* out, size_t cap) { return rocke_dw_wgrad_cols_kernel_name(spec, out, cap); },
        [&]() { return rocke_build_dw_wgrad_cols(b, spec, arch); });
}

extern "C" rocke_status_t rocke_dw_wgrad_cols_lower_to_llvm(const rocke_dw_wgrad_cols_spec_t* spec,
                                                            const char* arch,
                                                            rocke_llvm_flavor_t flavor,
                                                            char** out_ll,
                                                            char* err,
                                                            size_t err_cap)
{
    if(arch == NULL)
    {
        arch = "gfx950";
    }
    return lower_to_llvm(
        spec != NULL,
        arch,
        flavor,
        out_ll,
        err,
        err_cap,
        "build_dw_wgrad_cols failed",
        [&](rocke_ir_builder_t* b) { return rocke_build_dw_wgrad_cols_new(b, spec, arch); });
}

/* ===================================================================== *
 *  Depthwise wgrad, LDS row rings
 *
 *  The block owns a channel tile and a column tile and streams output rows:
 *  per step it DMAs S new X rows and one dY row into LDS rings (prefetch steps
 *  ahead), and every lane reads its window out of LDS. Ring slots fold to
 *  constants by unrolling U = prefetch + ceil(KH / S) steps: the X ring has
 *  S*U row slots and the dY ring U.
 * ===================================================================== */

extern "C" rocke_dw_wgrad_lds_spec_t rocke_dw_wgrad_lds_spec_default(void)
{
    rocke_dw_wgrad_lds_spec_t s;
    memset(&s, 0, sizeof(s));
    s.problem = rocke_direct_conv_problem_default();
    s.name = "dw_wgrad_lds";
    s.lanes_c = 8;
    s.lanes_w = 32;
    s.cols_per_lane = 1;
    s.prefetch = 2;
    s.rows_per_seg = 32;
    s.target_blocks = 1024;
    return s;
}

namespace
{

int lds_block_c(const rocke_dw_wgrad_lds_spec_t* s)
{
    return 8 * s->lanes_c;
}

int lds_block_wo(const rocke_dw_wgrad_lds_spec_t* s)
{
    return s->lanes_w * s->cols_per_lane;
}

int lds_unroll(const rocke_dw_wgrad_lds_spec_t* s)
{
    return s->prefetch + cdiv(s->problem.KH, s->problem.stride);
}

int lds_x_cols(const rocke_dw_wgrad_lds_spec_t* s)
{
    return (lds_block_wo(s) - 1) * s->problem.stride + s->problem.KW;
}

int lds_rps(const rocke_dw_wgrad_lds_spec_t* s)
{
    return cdiv(s->rows_per_seg, lds_unroll(s)) * lds_unroll(s);
}

int lds_n_channel_tiles(const rocke_dw_wgrad_lds_spec_t* s)
{
    return cdiv(s->problem.groups, lds_block_c(s));
}

/* 16-byte DMA instructions per wave for one LDS row of `cols` pixels. */
int lds_dma_instrs(const rocke_dw_wgrad_lds_spec_t* s, int cols)
{
    int n_waves = s->lanes_c * s->lanes_w / kWave;
    return cdiv(cdiv(cols * (lds_block_c(s) / 8), kWave), n_waves);
}

/* Elements of one X / dY ring slot: whole DMA instructions of every wave. */
int lds_x_row_elems(const rocke_dw_wgrad_lds_spec_t* s)
{
    return s->lanes_c * s->lanes_w * lds_dma_instrs(s, lds_x_cols(s)) * 8;
}

int lds_dy_row_elems(const rocke_dw_wgrad_lds_spec_t* s)
{
    return s->lanes_c * s->lanes_w * lds_dma_instrs(s, lds_block_wo(s)) * 8;
}

} // namespace

extern "C" int rocke_dw_wgrad_lds_threads(const rocke_dw_wgrad_lds_spec_t* s)
{
    return s->lanes_c * s->lanes_w;
}

extern "C" long rocke_dw_wgrad_lds_lds_bytes(const rocke_dw_wgrad_lds_spec_t* s)
{
    /* The X and dY rings as the builder allocates them, plus the reduction. */
    long U = lds_unroll(s);
    long x = (long)s->problem.stride * U * lds_x_row_elems(s) * 2;
    long dy = U * lds_dy_row_elems(s) * 2;
    long red = (long)s->lanes_w * lds_block_c(s) * 4;
    return x + dy + red;
}

extern "C" int rocke_dw_wgrad_lds_n_partitions(const rocke_dw_wgrad_lds_spec_t* s)
{
    const rocke_direct_conv_problem_t* p = &s->problem;
    int n_units = p->N * cdiv(problem_wo(p), lds_block_wo(s)) * cdiv(problem_ho(p), lds_rps(s));
    int c = cdiv(s->target_blocks, lds_n_channel_tiles(s));
    return n_units < c ? n_units : c;
}

extern "C" long rocke_dw_wgrad_lds_workspace_bytes(const rocke_dw_wgrad_lds_spec_t* s)
{
    const rocke_direct_conv_problem_t* p = &s->problem;
    return (long)rocke_dw_wgrad_lds_n_partitions(s) * p->KH * p->KW * p->groups * 4;
}

extern "C" void rocke_dw_wgrad_lds_launch_geometry(const rocke_dw_wgrad_lds_spec_t* s,
                                                   int grid[3],
                                                   int block[3])
{
    grid[0] = lds_n_channel_tiles(s);
    grid[1] = rocke_dw_wgrad_lds_n_partitions(s);
    grid[2] = 1;
    block[0] = rocke_dw_wgrad_lds_threads(s);
    block[1] = 1;
    block[2] = 1;
}

extern "C" rocke_status_t
    rocke_dw_wgrad_lds_kernel_name(const rocke_dw_wgrad_lds_spec_t* s, char* out, size_t out_cap)
{
    if(s == NULL || out == NULL || out_cap == 0)
    {
        return ROCKE_ERR_VALUE;
    }
    const rocke_direct_conv_problem_t* p = &s->problem;
    char f_buf[64], lc_buf[24], lw_buf[24], cpl_buf[24], pf_buf[24], rps_buf[24], tb_buf[24];
    snprintf(f_buf, sizeof f_buf, "f%dx%dp%ds%d", p->KH, p->KW, p->PAD, p->stride);
    snprintf(lc_buf, sizeof lc_buf, "lc%d", s->lanes_c);
    snprintf(lw_buf, sizeof lw_buf, "lw%d", s->lanes_w);
    snprintf(cpl_buf, sizeof cpl_buf, "cpl%d", s->cols_per_lane);
    snprintf(pf_buf, sizeof pf_buf, "pf%d", s->prefetch);
    snprintf(rps_buf, sizeof rps_buf, "rps%d", s->rows_per_seg);
    snprintf(tb_buf, sizeof tb_buf, "tb%d", s->target_blocks);
    const char* parts[7] = {f_buf, lc_buf, lw_buf, cpl_buf, pf_buf, rps_buf, tb_buf};
    int on[1] = {problem_is_bf16(p) ? 1 : 0};
    return rocke_kernel_name_join(s->name, parts, 7, bf16_flag, on, 1, out, out_cap, NULL);
}

extern "C" bool rocke_dw_wgrad_lds_is_valid_spec(const rocke_dw_wgrad_lds_spec_t* s,
                                                 const char* arch,
                                                 char* reason,
                                                 size_t reason_cap)
{
    char msg[ROCKE_ERR_MSG_CAP];
    if(s == NULL)
    {
        set_reason(reason, reason_cap, "null spec");
        return false;
    }
    if(arch == NULL)
    {
        arch = "gfx950";
    }
    const rocke_archtarget_t* target = rocke_archtarget_from_gfx(arch);
    if(target == NULL)
    {
        rocke_set_unknown_arch_reason(reason, reason_cap, arch);
        return false;
    }
    const rocke_direct_conv_problem_t* p = &s->problem;
    if(dilation_reason(p, msg, sizeof msg))
    {
        set_reason(reason, reason_cap, msg);
        return false;
    }
    int threads = rocke_dw_wgrad_lds_threads(s);
    const char* why = NULL;
    if(target->wave_size != 64 || rocke_archtarget_async_lds_max_dwords(target) < 4)
    {
        ROCKE_ERR_SNPRINTF(
            msg, sizeof msg, "needs wave64 + 16-byte buffer_load_lds (absent on %s)", arch);
        why = msg;
    }
    else if(p->cpg != 1 || p->kpg != 1)
    {
        why = "depthwise only";
    }
    else if(!problem_dtype_ok(p))
    {
        why = "fp16/bf16 only";
    }
    else if(p->groups % 8)
    {
        why = "C must be a multiple of 8";
    }
    else if(threads < kWave || threads % kWave || threads > 1024)
    {
        why = "threads must be a multiple of 64, at most 1024";
    }
    else if(p->KH * p->KW * 8 > 128)
    {
        why = "too many accumulators";
    }
    else if(s->prefetch < 1)
    {
        why = "prefetch >= 1";
    }
    else if(rocke_dw_wgrad_lds_lds_bytes(s) > 160 * 1024)
    {
        ROCKE_ERR_SNPRINTF(msg, sizeof msg, "LDS %ld B > 160 KiB", rocke_dw_wgrad_lds_lds_bytes(s));
        why = msg;
    }
    else if(s->rows_per_seg < 1)
    {
        why = "rows_per_seg >= 1";
    }
    else if(s->target_blocks < 1)
    {
        why = "target_blocks >= 1";
    }
    if(why != NULL)
    {
        set_reason(reason, reason_cap, why);
        return false;
    }
    set_reason(reason, reason_cap, "ok");
    return true;
}

namespace
{

struct LdsLane
{
    V col, ch, ok;
};

struct LdsOff
{
    V off, ok;
};

rocke_kernel_def_t*
    build_lds(rocke_ir_builder_t* b, const rocke_dw_wgrad_lds_spec_t* spec, const char* arch)
{
    char reason[ROCKE_ERR_MSG_CAP];
    if(!rocke_dw_wgrad_lds_is_valid_spec(spec, arch, reason, sizeof reason))
    {
        fail_invalid(b, "invalid DwWgradLdsSpec", reason);
        return NULL;
    }
    Em e{b};
    const rocke_direct_conv_problem_t& p = spec->problem;
    const int KH = p.KH, KW = p.KW, PAD = p.PAD, S = p.stride;
    const int TAPS = KH * KW;
    const int LC = spec->lanes_c, LW = spec->lanes_w, WL = spec->cols_per_lane;
    const int BC = lds_block_c(spec), BWO = lds_block_wo(spec);
    const int CH = BC / 8; /* 16-byte chunks per pixel */
    const int XC = lds_x_cols(spec);
    const int T = rocke_dw_wgrad_lds_threads(spec);
    const int D = spec->prefetch;
    const int U = lds_unroll(spec);
    const int NX = S * U, NY = U;
    const int RPS = lds_rps(spec);
    const int PRE
        = KH - S > 0 ? KH - S : 0; /* X rows a segment needs before its first step's own */
    const bool bf16 = problem_is_bf16(&p);
    const rocke_type_t* io_type = bf16 ? rocke_bf16() : rocke_f16();

    rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", T);
    rocke_attr_set_bool(b, &b->kernel->attrs, "lds_alias_scopes", true);
    rocke_dconv_params_t P;
    rocke_dconv_emit_params(b, &P, "wgrad", io_type);
    if(!rocke_ir_builder_ok(b))
    {
        return NULL;
    }

    V c0 = e.ci(0);
    V c1 = e.ci(1);
    V c2 = e.ci(2);
    V oob = e.ci(((int64_t)1 << 31) - 1);
    V p_C = P.p_groups;
    V tid = rocke_b_thread_id_x(b);
    V lane = e.modi(tid, kWave);
    V wave = rocke_b_readfirstlane(b, e.divi(tid, kWave));
    V lc = e.modi(tid, LC);
    V lw = e.divi(tid, LC);
    V ctile = rocke_b_block_id_x(b);
    V part = rocke_b_block_id_y(b);
    V c_blk0 = e.muli(ctile, BC);
    V n_ct = e.cdivi(P.p_Wo, BWO);
    V n_hb = e.cdivi(P.p_Ho, RPS);
    V n_units = e.mul(e.mul(P.p_N, n_ct), n_hb);
    V n_tiles = e.cdivi(p_C, BC);
    V tb = e.ci(spec->target_blocks);
    V np = e.add(tb, e.sub(n_tiles, c1));
    np = e.div(np, n_tiles);
    V n_part = e.smin(n_units, np);

    V a_rsrc = rocke_b_buffer_rsrc(b, P.A, P.A_bytes); /* dY */
    V b_rsrc = rocke_b_buffer_rsrc(b, P.Bp, P.B_bytes); /* X */

    /* ---- LDS rings: one allocation per slot so the waitcnt pass can tell them apart */
    const int IX = lds_dma_instrs(spec, XC); /* DMA instrs per wave per X row */
    const int IY = lds_dma_instrs(spec, BWO);
    const int X_ELEMS = lds_x_row_elems(spec);
    const int Y_ELEMS = lds_dy_row_elems(spec);
    std::vector<V> x_bufs, y_bufs, x_addr, y_addr;
    for(int i = 0; i < NX; ++i)
    {
        std::string h = "xr" + std::to_string(i);
        x_bufs.push_back(e.smem_alloc_1d(io_type, X_ELEMS, h.c_str()));
    }
    for(int i = 0; i < NY; ++i)
    {
        std::string h = "yr" + std::to_string(i);
        y_bufs.push_back(e.smem_alloc_1d(io_type, Y_ELEMS, h.c_str()));
    }
    for(V s : x_bufs)
    {
        x_addr.push_back(rocke_b_smem_addr_of(b, s));
    }
    for(V s : y_bufs)
    {
        y_addr.push_back(rocke_b_smem_addr_of(b, s));
    }

    /* DMA lane geometry: slot L = (wave * I + i) * 64 + lane -> (col, chunk). */
    auto dma_lanes = [&](int n_instr, int cols) {
        std::vector<LdsLane> out;
        for(int i = 0; i < n_instr; ++i)
        {
            V slot = e.muli(wave, n_instr);
            slot = e.addi(slot, i);
            slot = e.muli(slot, kWave);
            slot = e.add(slot, lane);
            LdsLane l;
            l.col = e.divi(slot, CH);
            l.ch = e.muli(e.modi(slot, CH), 8);
            l.ok = e.lti(l.col, cols);
            out.push_back(l);
        }
        return out;
    };
    std::vector<LdsLane> x_lanes = dma_lanes(IX, XC);
    std::vector<LdsLane> y_lanes = dma_lanes(IY, BWO);

    auto dma = [&](V rsrc, V addr, int n_instr, int i, V voff) {
        V o = e.muli(wave, n_instr);
        o = e.addi(o, i);
        o = e.muli(o, kWave * 16);
        V lds = rocke_b_smem_ptr_add(b, addr, e.zext64(o));
        rocke_b_async_buffer_load_lds(b, rsrc, lds, voff, c0, 4, 0);
    };

    /* Fragment read element indices (lane: chunk lc of a column). */
    auto col_index = [&](V col) {
        V a = e.muli(col, BC);
        return e.add(a, e.muli(lc, 8));
    };
    auto cvt8 = [&](V v) {
        std::vector<V> out;
        for(int el = 0; el < 8; ++el)
        {
            out.push_back(rocke_b_cast_to_f32(b, rocke_b_vec_extract(b, v, el)));
        }
        return out;
    };

    V zero = rocke_b_const_f32(b, 0.0);
    const int n_acc = TAPS * 8;
    const int PER_STEP = S * IX + IY;

    std::vector<V> accs = e.for_iter(
        part,
        n_units,
        n_part,
        iter_names("acc", n_acc),
        std::vector<V>(n_acc, zero),
        "unit",
        [&](V unit, std::vector<V> carried) {
            V hb = e.mod(unit, n_hb);
            V rest = e.div(unit, n_hb);
            V ct = e.mod(rest, n_ct);
            V n_img = e.div(rest, n_ct);
            V h0 = e.muli(hb, RPS);
            V q0 = e.muli(ct, BWO);
            V xi0 = e.subi(e.muli(q0, S), PAD); /* first X column of the tile */
            V x_row0 = e.subi(e.muli(h0, S), PAD); /* X row of relative index 0 */
            V k_rps = e.ci(RPS);
            V n_steps = e.smin(k_rps, e.sub(P.p_Ho, h0));

            /* Per-lane DMA offsets of this unit (rows added per issue). */
            std::vector<LdsOff> x_offs, y_offs;
            for(const LdsLane& l : x_lanes)
            {
                V wi = e.add(xi0, l.col);
                V a = e.ge(wi, c0);
                a = e.land(a, e.lt(wi, P.p_Wi));
                V okc = e.land(l.ok, a);
                V cc = e.add(c_blk0, l.ch);
                okc = e.land(okc, e.lt(cc, p_C));
                V off = e.mul(n_img, P.p_B_stride_n);
                off = e.add(off, e.mul(wi, P.p_B_stride_wi));
                off = e.add(off, cc);
                x_offs.push_back({off, okc});
            }
            for(const LdsLane& l : y_lanes)
            {
                V wo = e.add(q0, l.col);
                V cc = e.add(c_blk0, l.ch);
                V a = e.lt(wo, P.p_Wo);
                a = e.land(a, e.lt(cc, p_C));
                V oky = e.land(l.ok, a);
                V off = e.mul(n_img, P.p_A_stride_n);
                off = e.add(off, e.mul(wo, P.p_A_stride_wi));
                off = e.add(off, cc);
                y_offs.push_back({off, oky});
            }

            auto issue_x = [&](V xr, int slot) {
                V hi = e.add(x_row0, xr);
                /* Rows of later steps than the segment's own stay masked. */
                V a = e.ge(hi, c0);
                a = e.land(a, e.lt(hi, P.p_Hi));
                V lim = e.addi(e.muli(n_steps, S), PRE);
                V ok_r = e.land(a, e.lt(xr, lim));
                for(int i = 0; i < (int)x_offs.size(); ++i)
                {
                    V okk = e.land(x_offs[i].ok, ok_r);
                    V row = e.mul(hi, P.p_B_stride_hi);
                    V voff = e.mul(e.add(x_offs[i].off, row), c2);
                    voff = e.select(okk, voff, oob);
                    dma(b_rsrc, x_addr[slot], IX, i, voff);
                }
            };
            auto issue_y = [&](V t, int slot) {
                V ho = e.add(h0, t);
                V ok_r = e.lt(t, n_steps);
                for(int i = 0; i < (int)y_offs.size(); ++i)
                {
                    V okk = e.land(y_offs[i].ok, ok_r);
                    V row = e.mul(ho, P.p_A_stride_hi);
                    V voff = e.mul(e.add(y_offs[i].off, row), c2);
                    voff = e.select(okk, voff, oob);
                    dma(a_rsrc, y_addr[slot], IY, i, voff);
                }
            };
            /* DMA the S X rows that step t adds and dY row t (tm = t mod U). */
            auto issue_step = [&](V t, int tm) {
                for(int j = 0; j < S; ++j)
                {
                    V xr = e.addi(e.muli(t, S), PRE + j);
                    issue_x(xr, (S * tm + PRE + j) % NX);
                }
                issue_y(t, tm % NY);
            };

            /* Prologue: the PRE leading X rows, then steps 0..D-1. */
            for(int j = 0; j < PRE; ++j)
            {
                issue_x(e.ci(j), j % NX);
            }
            for(int t = 0; t < D; ++t)
            {
                issue_step(e.ci(t), t);
            }

            auto compute = [&](int tm, std::vector<V>& la) {
                std::vector<std::vector<V>> dyv;
                for(int jw = 0; jw < WL; ++jw)
                {
                    V col = e.addi(e.muli(lw, WL), jw);
                    dyv.push_back(
                        cvt8(e.smem_load(y_bufs[tm % NY], c0, col_index(col), io_type, 8)));
                }
                for(int r = 0; r < KH; ++r)
                {
                    int slot = (S * tm + r) % NX;
                    std::map<int, std::vector<V>> win;
                    for(int jw = 0; jw < WL; ++jw)
                    {
                        for(int s = 0; s < KW; ++s)
                        {
                            int xc = jw * S + s;
                            if(win.find(xc) == win.end())
                            {
                                V col = e.addi(e.muli(lw, WL * S), xc);
                                win[xc] = cvt8(
                                    e.smem_load(x_bufs[slot], c0, col_index(col), io_type, 8));
                            }
                            const std::vector<V>& xv = win[xc];
                            for(int el = 0; el < 8; ++el)
                            {
                                int i = (r * KW + s) * 8 + el;
                                la[i] = rocke_b_fma(b, dyv[jw][el], xv[el], la[i]);
                            }
                        }
                    }
                }
            };

            V trips = e.cdivi(n_steps, U);
            trips = e.smax(trips, c1);
            std::vector<V> accs_out
                = e.for_iter(c0,
                             trips,
                             c1,
                             iter_names("a", carried.size()),
                             carried,
                             "rb",
                             [&](V rb, std::vector<V> la) {
                                 for(int u = 0; u < U; ++u)
                                 {
                                     V t = e.addi(e.muli(rb, U), u);
                                     rocke_b_s_waitcnt(b, (D - 1) * PER_STEP, 0, -1);
                                     rocke_b_s_barrier_bare(b);
                                     issue_step(e.addi(t, D), (u + D) % U);
                                     compute(u, la);
                                 }
                                 return la;
                             });
            rocke_b_s_waitcnt(b, 0, 0, -1);
            rocke_b_s_barrier_bare(b);
            return accs_out;
        });
    if((int)accs.size() != n_acc)
    {
        return NULL;
    }

    /* ---- reduce over the LW column lanes, store ws[part][tap][c] ---- */
    V red = e.smem_alloc_1d(rocke_f32(), LW * BC, "red");
    for(int t = 0; t < TAPS; ++t)
    {
        V base = e.muli(lw, BC);
        base = e.add(base, e.muli(lc, 8));
        for(int e0 = 0; e0 < 8; e0 += 4)
        {
            V comps[4];
            for(int el = 0; el < 4; ++el)
            {
                comps[el] = accs[t * 8 + e0 + el];
            }
            V vec = rocke_b_vec_pack(b, comps, 4, rocke_f32());
            e.smem_store_f32(red, c0, e.addi(base, e0), vec, 4);
        }
        rocke_b_sync(b);
        for(int k = 0; k < cdiv(BC, T); ++k)
        {
            V o = e.addi(tid, k * T);
            V o_ok = e.lti(o, BC);
            V tot = NULL;
            for(int q = 0; q < LW; ++q)
            {
                V kq = e.ci(q * BC);
                V idx = e.select(o_ok, e.add(kq, o), c0);
                V v = rocke_b_vec_extract(b, e.smem_load_f32(red, c0, idx, 1), 0);
                tot = (tot == NULL) ? v : rocke_b_fadd(b, tot, v);
            }
            V c_glb = e.add(c_blk0, o);
            V okw = e.land(o_ok, e.lt(c_glb, p_C));
            V off = e.addi(e.muli(part, TAPS), t);
            off = e.add(e.mul(off, p_C), c_glb);
            e.if_(okw, [&]() { rocke_b_global_store(b, P.D, off, tot, 4); });
        }
        rocke_b_sync(b);
    }
    return rocke_ir_builder_ok(b) ? b->kernel : NULL;
}

} // namespace

extern "C" rocke_kernel_def_t* rocke_build_dw_wgrad_lds(rocke_ir_builder_t* b,
                                                        const rocke_dw_wgrad_lds_spec_t* spec,
                                                        const char* arch)
{
    if(b == NULL || spec == NULL)
    {
        return NULL;
    }
    return build_lds(b, spec, arch ? arch : "gfx950");
}

extern "C" rocke_kernel_def_t* rocke_build_dw_wgrad_lds_new(rocke_ir_builder_t* b,
                                                            const rocke_dw_wgrad_lds_spec_t* spec,
                                                            const char* arch)
{
    return build_new(
        b,
        spec != NULL,
        "build_dw_wgrad_lds",
        [&](char* out, size_t cap) { return rocke_dw_wgrad_lds_kernel_name(spec, out, cap); },
        [&]() { return rocke_build_dw_wgrad_lds(b, spec, arch); });
}

extern "C" rocke_status_t rocke_dw_wgrad_lds_lower_to_llvm(const rocke_dw_wgrad_lds_spec_t* spec,
                                                           const char* arch,
                                                           rocke_llvm_flavor_t flavor,
                                                           char** out_ll,
                                                           char* err,
                                                           size_t err_cap)
{
    if(arch == NULL)
    {
        arch = "gfx950";
    }
    return lower_to_llvm(
        spec != NULL,
        arch,
        flavor,
        out_ll,
        err,
        err_cap,
        "build_dw_wgrad_lds failed",
        [&](rocke_ir_builder_t* b) { return rocke_build_dw_wgrad_lds_new(b, spec, arch); });
}
