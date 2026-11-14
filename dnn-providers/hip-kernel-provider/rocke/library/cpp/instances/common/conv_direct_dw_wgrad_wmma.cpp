// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * conv_direct_dw_wgrad_wmma.cpp -- C++ port of the Toeplitz WMMA depthwise
 * wgrad (library/kernels/common/conv_direct_dw_wgrad_wmma.py):
 *
 *   Python                                    C++ (this file)
 *   DwWgradWmmaSpec                        -> rocke_direct_dw_wgrad_wmma_spec_t
 *   dw_wgrad_wmma_reason                   -> rocke_direct_dw_wgrad_wmma_reason()
 *   is_valid_dw_wgrad_wmma_spec            -> rocke_direct_dw_wgrad_wmma_is_valid_spec()
 *   kernel_name()                          -> rocke_direct_dw_wgrad_wmma_kernel_name()
 *   build_dw_wgrad_wmma(spec, arch)        -> rocke_build_direct_dw_wgrad_wmma_new()
 *
 * Every IR op (including each arith.constant) is emitted in the order the
 * Python builder emits it; see the contract in
 * rocke/instance_conv_direct_wgrad_internal.hpp (a call nests at most one
 * emitting argument).
 */
#include "rocke/instance_conv_direct_dw_wgrad_wmma.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <string>
#include <vector>

#include "rocke/arch_target.h" /* rocke_set_unknown_arch_reason, rocke_mma_catalog_has_shape */
#include "rocke/helper_rocke.core.arch.h" /* rocke_archtarget_from_gfx, max_threads_per_block */
#include "rocke/helper_rocke.helpers.io.h" /* rocke_b_io_ir_type */
#include "rocke/helper_rocke.helpers.spec.h" /* rocke_kernel_name_join, rocke_ceil_div */
#include "rocke/instance_conv_direct_grouped_internal.h" /* rocke_dconv_emit_params, buf_load_vN */
#include "rocke/instance_conv_direct_wgrad_internal.hpp" /* Em, fail_invalid, build_new */
#include "rocke/instance_conv_fwd_wmma_gate_internal.h" /* rocke_wmma_fwd_op_id, ROCKE_FWD_WMMA_K */

using namespace rocke_dwg;

namespace
{

constexpr int kCH = ROCKE_DW_WGRAD_WMMA_CH;

int n_rb(const rocke_direct_dw_wgrad_wmma_spec_t* s)
{
    return (int)rocke_floor_div(s->problem.KH, 2) + 1;
}

int n_s(const rocke_direct_dw_wgrad_wmma_spec_t* s)
{
    return (int)rocke_ceil_div(s->problem.KW, 8);
}

int x_rows(const rocke_direct_dw_wgrad_wmma_spec_t* s)
{
    return s->rows + 2 * (int)rocke_floor_div(s->problem.KH, 2);
}

int x_cols(const rocke_direct_dw_wgrad_wmma_spec_t* s)
{
    return s->tile_q + 8 * n_s(s) - 1;
}

int ch_pitch(const rocke_direct_dw_wgrad_wmma_spec_t* s)
{
    return rocke_direct_dw_wgrad_wmma_block_ch(s) + 8; /* 16-byte skew per pixel */
}

long long grid_x(const rocke_direct_dw_wgrad_wmma_spec_t* s)
{
    const rocke_direct_conv_problem_t* p = &s->problem;
    return rocke_ceil_div(p->groups, rocke_direct_dw_wgrad_wmma_block_ch(s))
           * rocke_ceil_div(rocke_direct_conv_problem_Wo(p), s->tile_q);
}

long long items(const rocke_direct_dw_wgrad_wmma_spec_t* s)
{
    const rocke_direct_conv_problem_t* p = &s->problem;
    return (long long)p->N * rocke_ceil_div(rocke_direct_conv_problem_Ho(p), s->rows);
}

} // namespace

/* ===================================================================== *
 *  Spec helpers
 * ===================================================================== */
extern "C" rocke_direct_dw_wgrad_wmma_spec_t rocke_direct_dw_wgrad_wmma_spec_default(void)
{
    rocke_direct_dw_wgrad_wmma_spec_t s;
    memset(&s, 0, sizeof s);
    s.problem = rocke_direct_conv_problem_default();
    s.name = "direct_depthwise_wgrad_wmma";
    s.wave_size = 32;
    s.waves = 4;
    s.tile_q = 32;
    s.rows = 8;
    s.target_blocks = 1024;
    s.ws_replicas = 64;
    s.prefetch = true;
    return s;
}

extern "C" int
    rocke_direct_dw_wgrad_wmma_threads_per_block(const rocke_direct_dw_wgrad_wmma_spec_t* spec)
{
    return spec->waves * spec->wave_size;
}

extern "C" int rocke_direct_dw_wgrad_wmma_block_ch(const rocke_direct_dw_wgrad_wmma_spec_t* spec)
{
    return kCH * spec->waves;
}

extern "C" int rocke_direct_dw_wgrad_wmma_tiles(const rocke_direct_dw_wgrad_wmma_spec_t* spec)
{
    return n_rb(spec) * n_s(spec);
}

extern "C" long rocke_direct_dw_wgrad_wmma_lds_bytes(const rocke_direct_dw_wgrad_wmma_spec_t* spec)
{
    return ((long)x_rows(spec) * x_cols(spec) + (long)spec->rows * spec->tile_q) * ch_pitch(spec)
           * 2;
}

extern "C" int rocke_direct_dw_wgrad_wmma_n_split(const rocke_direct_dw_wgrad_wmma_spec_t* spec)
{
    long long n = rocke_ceil_div(spec->target_blocks, grid_x(spec));
    long long it = items(spec);
    if(it < n)
        n = it;
    return (int)(n < 1 ? 1 : n);
}

extern "C" void rocke_direct_dw_wgrad_wmma_grid(const rocke_direct_dw_wgrad_wmma_spec_t* spec,
                                                int grid[3])
{
    grid[0] = (int)grid_x(spec);
    grid[1] = rocke_direct_dw_wgrad_wmma_n_split(spec);
    grid[2] = 1;
}

extern "C" rocke_status_t rocke_direct_dw_wgrad_wmma_kernel_name(
    const rocke_direct_dw_wgrad_wmma_spec_t* spec, char* out, size_t out_cap)
{
    if(spec == NULL || out == NULL || out_cap == 0)
        return ROCKE_ERR_VALUE;
    const rocke_direct_conv_problem_t* p = &spec->problem;
    char shrt[128], f[64], wv[24], q[24], r[24], tb[24], wsr[24];
    if(rocke_direct_conv_problem_short(p, shrt, sizeof shrt) != ROCKE_OK)
        return ROCKE_ERR_VALUE;
    snprintf(f, sizeof f, "f%dx%dp%dx%d", p->KH, p->KW, p->PAD, rocke_direct_conv_problem_pad_w(p));
    snprintf(wv, sizeof wv, "wv%d", spec->waves);
    snprintf(q, sizeof q, "q%d", spec->tile_q);
    snprintf(r, sizeof r, "r%d", spec->rows);
    snprintf(tb, sizeof tb, "tb%d", spec->target_blocks);
    snprintf(wsr, sizeof wsr, "wsr%d", spec->ws_replicas);
    const char* parts[] = {shrt, f, wv, q, r, tb};
    const char* flag_names[] = {wsr, "nopf", "bf16"};
    int flag_on[]
        = {spec->ws_replicas > 1 ? 1 : 0, spec->prefetch ? 0 : 1, problem_is_bf16(p) ? 1 : 0};
    return rocke_kernel_name_join(spec->name ? spec->name : "direct_depthwise_wgrad_wmma",
                                  parts,
                                  sizeof(parts) / sizeof(parts[0]),
                                  flag_names,
                                  flag_on,
                                  3,
                                  out,
                                  out_cap,
                                  NULL);
}

/* dw_wgrad_wmma_reason(spec): the checks in the Python order and wording. */
extern "C" bool rocke_direct_dw_wgrad_wmma_reason(const rocke_direct_dw_wgrad_wmma_spec_t* spec,
                                                  char* reason,
                                                  size_t cap)
{
    const rocke_direct_conv_problem_t* p = &spec->problem;
    if(!problem_dtype_ok(p))
    {
        char rep[64];
        if(p->dtype == NULL)
            snprintf(rep, sizeof rep, "None");
        else
            rocke_py_str_repr(p->dtype, rep, sizeof rep);
        rocke_spec_set_reasonf(reason, cap, "unsupported dtype %s; expected fp16 or bf16", rep);
        return true;
    }
    if(p->cpg != 1 || p->kpg != 1)
    {
        rocke_spec_set_reasonf(reason, cap, "requires cpg=kpg=1 (got %d, %d)", p->cpg, p->kpg);
        return true;
    }
    if(p->stride != 1)
    {
        rocke_spec_set_reasonf(reason, cap, "requires stride 1 (got %d)", p->stride);
        return true;
    }
    if(p->groups % kCH != 0)
    {
        rocke_spec_set_reasonf(
            reason, cap, "channels %d must be a multiple of 8 (16-byte NHWC reads)", p->groups);
        return true;
    }
    if(p->KH < 1 || p->KW < 1)
    {
        rocke_spec_set_reasonf(reason, cap, "filter %dx%d is not positive", p->KH, p->KW);
        return true;
    }
    if(rocke_direct_conv_pad_w_reason(p, "", reason, cap))
        return true;
    if(p->PAD < 0 || rocke_direct_conv_problem_pad_w(p) < 0 || rocke_direct_conv_problem_Ho(p) < 1
       || rocke_direct_conv_problem_Wo(p) < 1)
    {
        rocke_spec_set_reason(reason, cap, "degenerate geometry");
        return true;
    }
    if(spec->wave_size != 32)
    {
        rocke_spec_set_reason(reason, cap, "the WMMA 16x16x32 lane maps are wave32");
        return true;
    }
    if(spec->waves < 1 || spec->waves > 16)
    {
        rocke_spec_set_reasonf(reason, cap, "waves must be 1..16 (got %d)", spec->waves);
        return true;
    }
    if(spec->tile_q < 4 || spec->tile_q % 4 != 0)
    {
        rocke_spec_set_reasonf(
            reason, cap, "tile_q must be a positive multiple of 4 (got %d)", spec->tile_q);
        return true;
    }
    if(spec->rows < 2 || spec->rows % 2 != 0)
    {
        rocke_spec_set_reasonf(
            reason, cap, "rows must be a positive even number (got %d)", spec->rows);
        return true;
    }
    if(spec->target_blocks < 1 || spec->ws_replicas < 1)
    {
        rocke_spec_set_reason(reason, cap, "target_blocks and ws_replicas must be >= 1");
        return true;
    }
    const long long steps
        = (long long)(spec->rows / 2) * spec->tile_q / 4 * rocke_direct_dw_wgrad_wmma_tiles(spec);
    if(steps > ROCKE_DW_WGRAD_WMMA_MAX_STEPS)
    {
        rocke_spec_set_reasonf(reason,
                               cap,
                               "%lld unrolled WMMAs per item exceed %d (smaller rows / tile_q)",
                               steps,
                               ROCKE_DW_WGRAD_WMMA_MAX_STEPS);
        return true;
    }
    return false;
}

extern "C" bool
    rocke_direct_dw_wgrad_wmma_is_valid_spec(const rocke_direct_dw_wgrad_wmma_spec_t* spec,
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
    const rocke_archtarget_t* target = rocke_archtarget_from_gfx(arch);
    if(target == NULL)
    {
        rocke_set_unknown_arch_reason(reason, reason_cap, arch);
        return false;
    }
    const rocke_direct_conv_problem_t* p = &spec->problem;
    if(rocke_direct_dilation_reason(p, reason, reason_cap))
        return false;
    if(rocke_direct_dw_wgrad_wmma_reason(spec, reason, reason_cap))
        return false;
    if(target->wave_size != 32)
    {
        rocke_spec_set_reasonf(
            reason, reason_cap, "wave32 kernel (got %s, wave%d)", arch, target->wave_size);
        return false;
    }
    const char* ab = problem_is_bf16(p) ? "bf16" : "f16";
    if(!rocke_mma_catalog_has_shape(
           rocke_archtarget_mma(target), "wmma", ab, ab, "fp32", 16, 16, ROCKE_FWD_WMMA_K))
    {
        rocke_spec_set_reasonf(reason, reason_cap, "missing wmma 16x16x32 %s on %s", ab, arch);
        return false;
    }
    const int threads = rocke_direct_dw_wgrad_wmma_threads_per_block(spec);
    if(threads > rocke_archtarget_max_threads_per_block(target))
    {
        rocke_spec_set_reasonf(
            reason, reason_cap, "%d threads > hardware cap on %s", threads, arch);
        return false;
    }
    const long lds = rocke_direct_dw_wgrad_wmma_lds_bytes(spec);
    if(lds > target->lds_capacity_bytes)
    {
        rocke_spec_set_reasonf(
            reason, reason_cap, "LDS %ld B > %d on %s", lds, target->lds_capacity_bytes, arch);
        return false;
    }
    rocke_spec_set_reason(reason, reason_cap, "ok");
    return true;
}

/* ===================================================================== *
 *  build_dw_wgrad_wmma
 * ===================================================================== */
namespace
{

/* One staging slot of plan(): a thread's 16-byte chunk of a staged pixel. */
struct Slot
{
    V live, pix, ch8, rr, cc;
};

/* `loop = b.scf_for_iter(lo, hi, step, iter_args, iv_name=...)` with the
 * Python default elide_trailing_barrier=True; body(iv, carried) returns the
 * yielded values; returns loop.results. */
template <class Fn>
std::vector<V> for_iter_elide(rocke_ir_builder_t* b,
                              V lo,
                              V hi,
                              V step,
                              const std::vector<std::string>& names,
                              const std::vector<V>& inits,
                              const char* iv_name,
                              Fn&& body)
{
    std::vector<rocke_iter_arg_t> args(names.size());
    for(size_t i = 0; i < names.size(); ++i)
    {
        args[i].name = names[i].c_str();
        args[i].init = inits[i];
    }
    rocke_for_t f = rocke_b_scf_for_iter(
        b, lo, hi, step, args.data(), (int)args.size(), iv_name, false, true);
    if(f.op == NULL || f.body == NULL)
        return std::vector<V>();
    std::vector<V> carried(f.iter_vars, f.iter_vars + f.num_iter_vars);
    rocke_b_region_enter(b, f.body);
    std::vector<V> out = body(f.iv, carried);
    rocke_b_scf_yield(b, out.data(), (int)out.size());
    rocke_b_region_leave(b);
    return std::vector<V>(f.op->results, f.op->results + f.op->num_results);
}

} // namespace

extern "C" rocke_kernel_def_t* rocke_build_direct_dw_wgrad_wmma(
    rocke_ir_builder_t* b, const rocke_direct_dw_wgrad_wmma_spec_t* spec, const char* arch)
{
    Em e{b};
    const rocke_direct_conv_problem_t* p = &spec->problem;
    const bool bf16 = problem_is_bf16(p);
    const int KH = p->KH, KW = p->KW;
    const int PH = p->PAD, PW = rocke_direct_conv_problem_pad_w(p);
    const int CB = rocke_direct_dw_wgrad_wmma_block_ch(spec);
    const int CP = ch_pitch(spec);
    const int CPX = CB / 8; /* 16-byte chunks per pixel */
    const int THREADS = rocke_direct_dw_wgrad_wmma_threads_per_block(spec);
    const int TQ = spec->tile_q, RT = spec->rows;
    const int XR = x_rows(spec), XC = x_cols(spec);
    const int NRB = n_rb(spec), NS = n_s(spec);
    const int TILES = NRB * NS;
    const int KK = KH * KW;

    const char* wmma_op = rocke_wmma_fwd_op_id(
        b, rocke_direct_conv_problem_dtype(p), arch ? arch : "gfx1250", "build_dw_wgrad_wmma");
    if(wmma_op == NULL)
        return NULL;

    rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", THREADS);
    const rocke_type_t* io_type = rocke_b_io_ir_type(b, rocke_direct_conv_problem_dtype(p));
    rocke_dconv_params_t P;
    rocke_dconv_emit_params(b, &P, "wgrad", io_type);
    if(!rocke_ir_builder_ok(b))
        return NULL;
    V dy_rsrc = rocke_b_buffer_rsrc(b, P.A, P.A_bytes);
    V x_rsrc = rocke_b_buffer_rsrc(b, P.Bp, P.B_bytes);
    V d_rsrc = rocke_b_buffer_rsrc(b, P.D, P.D_bytes);
    V p_N = P.p_N, p_H = P.p_Hi, p_W = P.p_Wi;
    V p_Ho = P.p_Ho, p_Wo = P.p_Wo, p_C = P.p_groups;

    V c0 = e.ci(0);
    V c1 = e.ci(1);
    V c2 = e.ci(2);
    V c8 = e.ci(8);
    V c16 = e.ci(16);
    V oob = e.ci((1 << 30) - 1); /* past every in-range byte offset */

    V tid = rocke_b_thread_id_x(b);
    V c32 = e.ci(32);
    V wave = rocke_b_readfirstlane(b, e.div(tid, c32));
    V lane = e.mod(tid, c32);
    V kh = e.div(lane, c16); /* K half: pixels 2 kh, 2 kh + 1; D row octet t */
    V col16 = e.mod(lane, c16); /* A row m / B, D column n */
    V a_t = e.div(col16, c8); /* A row m = 8 t + c' */
    V a_c = e.mod(col16, c8);
    V n_j = e.div(col16, c8); /* B / D column n = 8 j + s */
    V n_sv = e.mod(col16, c8);

    /* ---- block coordinates ---- */
    V c_TQ = e.ci(TQ);
    V c_RT = e.ci(RT);
    V c_CB = e.ci(CB);
    V n_q = e.div(e.addi(p_Wo, TQ - 1), c_TQ);
    V n_cg = e.div(e.addi(p_C, CB - 1), c_CB);
    V gx = e.mul(n_cg, n_q);
    V n_yt = e.div(e.addi(p_Ho, RT - 1), c_RT);
    V n_items = e.mul(p_N, n_yt);
    /* n_split = clamp(ceil(target / grid_x), 1, items): the host grid's formula. */
    V c_tb = e.ci(spec->target_blocks);
    V n_split = e.div(e.add(c_tb, e.sub(gx, c1)), gx);
    n_split = e.select(e.lt(n_items, n_split), n_items, n_split);
    n_split = e.select(e.lt(n_split, c1), c1, n_split);
    V bx = rocke_b_block_id_x(b);
    V split = rocke_b_block_id_y(b);
    V q_blk = e.mod(bx, n_q);
    V c_base = e.mul(e.div(bx, n_q), c_CB);
    V w0 = e.mul(q_blk, c_TQ);
    V wave_ch = e.mul(wave, c8);

    V sx_n = P.p_B_stride_n, sx_h = P.p_B_stride_hi, sx_w = P.p_B_stride_wi;
    V sy_n = P.p_A_stride_n, sy_h = P.p_A_stride_hi, sy_w = P.p_A_stride_wi;

    const int xs_shape[2] = {XR * XC, CP};
    V xs = rocke_b_smem_alloc(b, io_type, xs_shape, 2, "dww_x");
    const int ys_shape[2] = {RT * TQ, CP};
    V ys = rocke_b_smem_alloc(b, io_type, ys_shape, 2, "dww_dy");

    /* ---- staging plan (thread-static parts) ---- */
    auto plan = [&](int n_pix, int ncols) {
        std::vector<Slot> out;
        const int total = n_pix * CPX;
        const int passes = (int)rocke_ceil_div(total, THREADS);
        for(int i = 0; i < passes; ++i)
        {
            Slot s;
            V idx = e.add(tid, e.ci((int64_t)i * THREADS));
            s.live = e.lt(idx, e.ci(total));
            s.pix = e.div(idx, e.ci(CPX));
            s.ch8 = e.mul(e.modi(idx, CPX), c8);
            s.rr = e.div(s.pix, e.ci(ncols));
            s.cc = e.mod(s.pix, e.ci(ncols));
            out.push_back(s);
        }
        return out;
    };
    std::vector<Slot> x_plan = plan(XR * XC, XC);
    std::vector<Slot> y_plan = plan(RT * TQ, TQ);

    auto fetch = [&](V item, std::vector<V>& xv, std::vector<V>& yv) {
        V n_img = e.div(item, n_yt);
        V y0 = e.mul(e.sub(item, e.mul(n_img, n_yt)), c_RT);
        xv.clear();
        yv.clear();
        for(const Slot& s : x_plan)
        {
            V h = e.add(y0, e.subi(s.rr, PH));
            V w = e.add(w0, e.subi(s.cc, PW));
            V c = e.add(c_base, s.ch8);
            V hge = e.ge(h, c0);
            V hlt = e.lt(h, p_H);
            V ok = e.land(s.live, e.land(hge, hlt));
            V wge = e.ge(w, c0);
            V wlt = e.lt(w, p_W);
            ok = e.land(ok, e.land(wge, wlt));
            ok = e.land(ok, e.lt(c, p_C));
            V m1 = e.mul(n_img, sx_n);
            V m2 = e.mul(h, sx_h);
            V a1 = e.add(m1, m2);
            V m3 = e.mul(w, sx_w);
            V a2 = e.add(m3, c);
            V el = e.add(a1, a2);
            V off = e.select(ok, e.mul(el, c2), oob);
            xv.push_back(rocke_dconv_buf_load_vN(b, bf16, x_rsrc, off, c0, 4));
        }
        for(const Slot& s : y_plan)
        {
            V h = e.add(y0, s.rr);
            V w = e.add(w0, s.cc);
            V c = e.add(c_base, s.ch8);
            V hlt = e.lt(h, p_Ho);
            V wlt = e.lt(w, p_Wo);
            V ok = e.land(s.live, e.land(hlt, wlt));
            ok = e.land(ok, e.lt(c, p_C));
            V m1 = e.mul(n_img, sy_n);
            V m2 = e.mul(h, sy_h);
            V a1 = e.add(m1, m2);
            V m3 = e.mul(w, sy_w);
            V a2 = e.add(m3, c);
            V el = e.add(a1, a2);
            V off = e.select(ok, e.mul(el, c2), oob);
            yv.push_back(rocke_dconv_buf_load_vN(b, bf16, dy_rsrc, off, c0, 4));
        }
    };

    auto stage = [&](const std::vector<V>& xv, const std::vector<V>& yv) {
        for(size_t i = 0; i < x_plan.size(); ++i)
            e.if_(x_plan[i].live, [&]() {
                V idx[2] = {x_plan[i].pix, x_plan[i].ch8};
                rocke_b_smem_store_vN(b, xs, idx, 2, xv[i], 8);
            });
        for(size_t i = 0; i < y_plan.size(); ++i)
            e.if_(y_plan[i].live, [&]() {
                V idx[2] = {y_plan[i].pix, y_plan[i].ch8};
                rocke_b_smem_store_vN(b, ys, idx, 2, yv[i], 8);
            });
    };

    /* ---- per-lane constants for the fragments ---- */
    /* One-hot A: dY of channel c' sits in dword c' / 2, half c' % 2. */
    V is_d[4];
    for(int d = 0; d < 4; ++d)
    {
        V q = e.div(a_c, c2);
        is_d[d] = e.eq(q, e.ci(d));
    }
    V a_shift;
    {
        V m = e.mod(a_c, c2);
        a_shift = e.mul(m, e.ci(16));
    }
    const rocke_type_t* frag_t = rocke_vector_type(b, io_type, 16);
    V a_ch = e.add(wave_ch, a_c);
    V kh2 = e.mul(kh, c2);
    /* A pixel index (row y + t, column p0 + 2 kh + jj): lane part. */
    V a_pix[2];
    for(int jj = 0; jj < 2; ++jj)
    {
        V m = e.mul(a_t, c_TQ);
        a_pix[jj] = e.add(m, e.addi(kh2, jj));
    }
    /* B pixel index (row y + rb + j, column p0 + 2 kh + jj + s0 + s): lane part. */
    V c_XC = e.ci(XC);
    V b_pix[2];
    for(int jj = 0; jj < 2; ++jj)
    {
        V m = e.mul(n_j, c_XC);
        V k = e.addi(kh2, jj);
        b_pix[jj] = e.add(m, e.add(k, n_sv));
    }

    /* One staged item: per 4-column chunk, the A fragments of every dY row
     * pair first, then every X fragment once fed to each (y, rb) pair that
     * shares it. */
    auto compute = [&](const std::vector<V>& accs) {
        std::vector<V> out = accs;
        for(int qc = 0; qc < TQ / 4; ++qc)
        {
            const int p0 = 4 * qc;
            std::vector<V> a_frags;
            for(int yp = 0; yp < RT / 2; ++yp)
            {
                std::vector<V> words;
                for(int jj = 0; jj < 2; ++jj)
                {
                    V idx[2] = {e.addi(a_pix[jj], 2 * yp * TQ + p0), a_ch};
                    V v = rocke_b_smem_load_vN(b, ys, idx, 2, io_type, 1);
                    V x = rocke_b_vec_extract(b, v, 0);
                    x = rocke_b_bitcast(b, x, rocke_i16());
                    x = rocke_b_zext(b, x, rocke_i32());
                    V w = rocke_b_shl(b, x, a_shift);
                    for(int d = 0; d < 4; ++d)
                        words.push_back(e.select(is_d[d], w, c0));
                }
                V packed = rocke_b_vec_pack(b, words.data(), (int)words.size(), rocke_i32());
                a_frags.push_back(rocke_b_vec_bitcast(b, packed, frag_t));
            }
            for(int rbb = 0; rbb < RT - 2 + 2 * (NRB - 1) + 1; rbb += 2)
            {
                std::vector<std::pair<int, int>> pairs;
                for(int yp = 0; yp < RT / 2; ++yp)
                    for(int rbi = 0; rbi < NRB; ++rbi)
                        if(2 * yp + 2 * rbi == rbb)
                            pairs.push_back({yp, rbi});
                for(int si = 0; si < NS; ++si)
                {
                    const int base = rbb * XC + p0 + 8 * si;
                    V halves[2];
                    for(int jj = 0; jj < 2; ++jj)
                    {
                        V idx[2] = {e.addi(b_pix[jj], base), wave_ch};
                        halves[jj] = rocke_b_smem_load_vN(b, xs, idx, 2, io_type, 8);
                    }
                    V b_frag = rocke_b_vec_concat(b, halves[0], halves[1]);
                    for(const auto& pr : pairs)
                    {
                        const int t = pr.second * NS + si;
                        out[t]
                            = rocke_b_mma(b, wmma_op, a_frags[pr.first], b_frag, out[t], NULL, 0);
                    }
                }
            }
        }
        return out;
    };

    /* ---- item loop: the next item's global loads ride in registers while
     * the current one computes (prologue fetch, then store -> fetch next ->
     * compute). ---- */
    V zero_acc = rocke_b_zero_vec_f32(b, 8);
    std::vector<std::string> names = iter_names("dww_acc", TILES);
    std::vector<V> inits(TILES, zero_acc);
    std::vector<V> results;
    if(spec->prefetch)
    {
        std::vector<V> xv0, yv0;
        fetch(split, xv0, yv0);
        const size_t n_x = xv0.size();
        std::vector<std::string> xn = iter_names("dww_xv", xv0.size());
        std::vector<std::string> yn = iter_names("dww_yv", yv0.size());
        names.insert(names.end(), xn.begin(), xn.end());
        names.insert(names.end(), yn.begin(), yn.end());
        inits.insert(inits.end(), xv0.begin(), xv0.end());
        inits.insert(inits.end(), yv0.begin(), yv0.end());
        results = for_iter_elide(
            b, split, n_items, n_split, names, inits, "dww_item", [&](V item, std::vector<V> c) {
                std::vector<V> accs_in(c.begin(), c.begin() + TILES);
                std::vector<V> xv(c.begin() + TILES, c.begin() + TILES + n_x);
                std::vector<V> yv(c.begin() + TILES + n_x, c.end());
                rocke_b_sync_lds_only(b); /* the previous item's LDS reads are done */
                stage(xv, yv);
                rocke_b_sync_lds_only(b);
                V nxt = e.add(item, n_split);
                nxt = e.select(e.lt(nxt, n_items), nxt, item); /* past the end: refetch */
                std::vector<V> xnv, ynv;
                fetch(nxt, xnv, ynv);
                std::vector<V> out = compute(accs_in);
                out.insert(out.end(), xnv.begin(), xnv.end());
                out.insert(out.end(), ynv.begin(), ynv.end());
                return out;
            });
    }
    else
    {
        results = for_iter_elide(
            b, split, n_items, n_split, names, inits, "dww_item", [&](V item, std::vector<V> c) {
                std::vector<V> xv, yv;
                fetch(item, xv, yv);
                rocke_b_sync_lds_only(b);
                stage(xv, yv);
                rocke_b_sync_lds_only(b);
                return compute(c);
            });
    }
    if(!rocke_ir_builder_ok(b) || (int)results.size() < TILES)
        return NULL;

    /* ---- epilogue: atomics into the replica slab ---- */
    V lin = e.add(e.mul(split, gx), bx);
    V rep = e.modi(lin, spec->ws_replicas);
    V rep_c = e.mul(rep, p_C);
    V c4 = e.ci(4);
    V c_KH = e.ci(KH);
    V c_KW = e.ci(KW);
    V c_KK = e.ci(KK);
    for(int rbi = 0; rbi < NRB; ++rbi)
        for(int si = 0; si < NS; ++si)
        {
            /* Row of D lane (kh, i): m = 8 kh + i -> t = kh, c' = i. */
            V r = e.sub(e.add(e.ci(2 * rbi), n_j), kh);
            V s = e.add(e.ci(8 * si), n_sv);
            V rge = e.ge(r, c0);
            V rlt = e.lt(r, c_KH);
            V r_ok = e.land(rge, rlt);
            V tap_ok = e.land(r_ok, e.lt(s, c_KW));
            V tap = e.add(e.mul(r, c_KW), s);
            V acc = results[rbi * NS + si];
            for(int i = 0; i < kCH; ++i)
            {
                V c = e.addi(e.add(c_base, wave_ch), i);
                V ok = e.land(tap_ok, e.lt(c, p_C));
                V byte = e.mul(e.add(e.mul(e.add(rep_c, c), c_KK), tap), c4);
                e.if_(ok, [&]() {
                    V v = rocke_b_vec_extract(b, acc, i);
                    rocke_b_buffer_atomic_fadd_f32(
                        b, d_rsrc, byte, c0, v, ROCKE_CPOL_ATOMIC_CASCADE_DEV);
                });
            }
        }
    return b->kernel;
}

extern "C" rocke_kernel_def_t* rocke_build_direct_dw_wgrad_wmma_new(
    rocke_ir_builder_t* b, const rocke_direct_dw_wgrad_wmma_spec_t* spec, const char* arch)
{
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        char name[512];
        char why[ROCKE_ERR_MSG_CAP];
        if(b == NULL || spec == NULL)
            return NULL;
        if(arch == NULL)
            arch = "gfx1250";
        /* is_valid_dw_wgrad_wmma_spec (which runs the arch-independent
         * rules too), before the Python builds its IRBuilder. */
        if(!rocke_direct_dw_wgrad_wmma_is_valid_spec(spec, arch, why, sizeof why))
        {
            rocke_ir_builder_init(b, "direct_dw_wgrad_wmma_invalid");
            rocke_i_set_err(b, ROCKE_ERR_VALUE, "invalid DwWgradWmmaSpec for %s: %s", arch, why);
            return NULL;
        }
        if(rocke_direct_dw_wgrad_wmma_kernel_name(spec, name, sizeof name) != ROCKE_OK)
            return NULL;
        if(rocke_ir_builder_init(b, name) != ROCKE_OK)
            return NULL;
        return rocke_build_direct_dw_wgrad_wmma(b, spec, arch);
    });
}

extern "C" rocke_status_t
    rocke_direct_dw_wgrad_wmma_lower_to_llvm(const rocke_direct_dw_wgrad_wmma_spec_t* spec,
                                             const char* arch,
                                             rocke_llvm_flavor_t flavor,
                                             char** out_ll,
                                             char* err,
                                             size_t err_cap)
{
    if(arch == NULL)
        arch = "gfx1250";
    return lower_to_llvm(spec != NULL,
                         arch,
                         flavor,
                         out_ll,
                         err,
                         err_cap,
                         "build_dw_wgrad_wmma failed",
                         [&](rocke_ir_builder_t* bb) {
                             return rocke_build_direct_dw_wgrad_wmma_new(bb, spec, arch);
                         });
}
