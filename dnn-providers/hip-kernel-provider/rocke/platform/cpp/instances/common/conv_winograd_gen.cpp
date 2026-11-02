// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * conv_winograd_gen.cpp -- C++ port of the generalized Winograd forward
 * transform kernels (library/kernels/common/conv_winograd_gen.py): K1
 * build_filter_transform, K2 build_input_transform, K4
 * build_output_transform. See the header and the Python module for the
 * algorithm and the layouts.
 *
 * Every helper emits in PYTHON EVALUATION ORDER (arguments left-to-right,
 * innermost call first) so the SSA sequence -- and therefore the .ll bytes --
 * match exactly. Nested builder calls are hoisted into ordered locals: C++
 * leaves the evaluation order of call arguments unspecified.
 */
#include "rocke/instance_conv_winograd_gen.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <string>
#include <utility>
#include <vector>

#include "rocke/error_boundary.hpp" /* ckc::guard_builder boundary shim */
#include "rocke/instance_conv_winograd_internal.h"
#include "rocke/ir_internal.h" /* rocke_i_set_err_msg */

namespace
{

using rocke_wino::ToomCook;
using Vals = std::vector<rocke_value_t*>;

/* conv_winograd_fwd._MAX_BUFFER_BYTES */
constexpr long long kMaxBufferBytes = 0x7FFF0000LL;

void set_reason(char* reason, size_t cap, const std::string& msg)
{
    if(reason && cap > 0)
        snprintf(reason, cap, "%s", msg.c_str());
}

long long floordiv(long long a, long long b)
{
    long long q = a / b;
    if((a % b != 0) && ((a < 0) != (b < 0)))
        --q;
    return q;
}

long long cdiv(long long a, long long b)
{
    return -floordiv(-a, b);
}

const char* io_dt(const rocke_winograd_gen_spec_t* s)
{
    return s->io_dtype ? s->io_dtype : "fp16";
}

const char* op_dt(const rocke_winograd_gen_spec_t* s)
{
    return s->operand_dtype ? s->operand_dtype : "fp16";
}

/* WinoAxis.r1 / .n */
int ax_r1(const rocke_wino_axis_t& a)
{
    return a.seg > 1 ? a.m : a.r;
}

int ax_n(const rocke_wino_axis_t& a)
{
    return a.m + ax_r1(a) - 1;
}

std::string ax_label(const rocke_wino_axis_t& a)
{
    if(a.seg > 1)
        return std::to_string(a.seg) + "xF" + std::to_string(a.m) + "_" + std::to_string(a.m);
    return "F" + std::to_string(a.m) + "_" + std::to_string(a.r);
}

const ToomCook* ax_tc(const rocke_wino_axis_t& a)
{
    return rocke_wino::toom_cook(a.m, ax_r1(a));
}

/* WinoAxis.check(): "" when the axis is valid. */
std::string ax_check(const rocke_wino_axis_t& a)
{
    if(a.m < 1 || a.r < 1 || a.seg < 1)
    {
        return "bad axis WinoAxis(m=" + std::to_string(a.m) + ", r=" + std::to_string(a.r)
               + ", seg=" + std::to_string(a.seg) + ", pts=None)";
    }
    if(a.seg > 1 && a.seg * a.m < a.r)
    {
        return std::to_string(a.seg) + " segments of " + std::to_string(a.m)
               + " taps < r=" + std::to_string(a.r);
    }
    if(a.seg > 1 && (a.seg - 1) * a.m >= a.r)
        return "an all-zero segment";
    if(ax_tc(a) == nullptr)
    {
        return "no default points for F(" + std::to_string(a.m) + "," + std::to_string(ax_r1(a))
               + ")";
    }
    return "";
}

int ho(const rocke_winograd_gen_spec_t* s)
{
    return rocke_conv_problem_ho(&s->problem);
}

int wo(const rocke_winograd_gen_spec_t* s)
{
    return rocke_conv_problem_wo(&s->problem);
}

int cpg(const rocke_winograd_gen_spec_t* s)
{
    return s->problem.C / s->problem.groups;
}

int kpg(const rocke_winograd_gen_spec_t* s)
{
    return s->problem.K / s->problem.groups;
}

int vec_c(const rocke_winograd_gen_spec_t* s)
{
    return rocke_wino::pow2_vec(s->problem.C, s->vec);
}

int vec_k(const rocke_winograd_gen_spec_t* s)
{
    return rocke_wino::pow2_vec(s->problem.K, s->vec);
}

int vec_cg(const rocke_winograd_gen_spec_t* s)
{
    return rocke_wino::pow2_vec(cpg(s), s->vec);
}

bool is_io(const char* d)
{
    return d && (strcmp(d, "fp16") == 0 || strcmp(d, "bf16") == 0);
}

/* WinogradGenSpec.short() */
std::string spec_short(const rocke_winograd_gen_spec_t* s)
{
    const rocke_conv_problem_t& p = s->problem;
    std::string g = p.groups > 1 ? "G" + std::to_string(p.groups) : "";
    std::string prob = "N" + std::to_string(p.N) + "H" + std::to_string(p.Hi) + "W"
                       + std::to_string(p.Wi) + "C" + std::to_string(p.C) + "_K"
                       + std::to_string(p.K) + "Y" + std::to_string(p.Y) + "X" + std::to_string(p.X)
                       + g;
    std::string out = prob + "_" + ax_label(s->ax_y) + "_" + ax_label(s->ax_x) + "_" + io_dt(s)
                      + "_v" + std::to_string(s->vec) + "_b" + std::to_string(s->block_size);
    if(!s->nt_stores)
        out += "_rt";
    return out;
}

const char* stage_word(const char* stage)
{
    if(stage == nullptr)
        return nullptr;
    if(strcmp(stage, "K1") == 0)
        return "filter";
    if(strcmp(stage, "K2") == 0)
        return "input";
    if(strcmp(stage, "K4") == 0)
        return "output";
    return nullptr;
}

/* ------------------------------------------------------------------ IR */

struct Ctx
{
    rocke_ir_builder_t* b;
    const rocke_winograd_gen_spec_t* s;
};

rocke_value_t* ci(Ctx& c, int64_t v)
{
    return rocke_b_const_i32(c.b, v);
}

/* add(x, const(v)) */
rocke_value_t* addc(Ctx& c, rocke_value_t* x, int64_t v)
{
    rocke_value_t* k = ci(c, v);
    return rocke_b_add(c.b, x, k);
}

rocke_value_t* mulc(Ctx& c, rocke_value_t* x, int64_t v)
{
    rocke_value_t* k = ci(c, v);
    return rocke_b_mul(c.b, x, k);
}

rocke_value_t* divc(Ctx& c, rocke_value_t* x, int64_t v)
{
    rocke_value_t* k = ci(c, v);
    return rocke_b_div(c.b, x, k);
}

rocke_value_t* modc(Ctx& c, rocke_value_t* x, int64_t v)
{
    rocke_value_t* k = ci(c, v);
    return rocke_b_mod(c.b, x, k);
}

rocke_value_t* subc(Ctx& c, rocke_value_t* x, int64_t v)
{
    rocke_value_t* k = ci(c, v);
    return rocke_b_sub(c.b, x, k);
}

rocke_value_t* cmp_ltc(Ctx& c, rocke_value_t* x, int64_t v)
{
    rocke_value_t* k = ci(c, v);
    return rocke_b_cmp_lt(c.b, x, k);
}

rocke_value_t* param_ptr(Ctx& c, const char* name, const char* dtype, bool readonly)
{
    rocke_param_opts_t o;
    memset(&o, 0, sizeof(o));
    o.noalias = true;
    o.noalias_set = true;
    if(readonly)
    {
        o.readonly = true;
        o.readonly_set = true;
    }
    else
    {
        o.writeonly = true;
        o.writeonly_set = true;
    }
    const rocke_type_t* pt = rocke_ptr_type(c.b, rocke_wino::ir_dtype(dtype), "global");
    return rocke_b_param(c.b, name, pt, &o);
}

/* b.buffer_rsrc(ptr, b.param(<bytes>, I32)) */
rocke_value_t* rsrc_of(Ctx& c, rocke_value_t* ptr, const char* bytes_name)
{
    rocke_value_t* nb = rocke_b_param(c.b, bytes_name, rocke_i32(), NULL);
    return rocke_b_buffer_rsrc(c.b, ptr, nb);
}

struct TileCoords
{
    rocke_value_t *ok, *cv, *n, *a, *bs, *ti, *tj, *t;
};

/* _flat_tile(b, spec, num_vecs, tw_tiles) */
TileCoords flat_tile(Ctx& c, int num_vecs, int tw_tiles)
{
    rocke_ir_builder_t* b = c.b;
    const rocke_conv_problem_t& p = c.s->problem;
    int th = rocke_winograd_gen_tiles_h(c.s);
    TileCoords tc;
    rocke_value_t* bx = rocke_b_block_id_x(b);
    rocke_value_t* m = mulc(c, bx, c.s->block_size);
    rocke_value_t* tid = rocke_b_thread_id_x(b);
    rocke_value_t* gid = rocke_b_add(b, m, tid);
    long long total = rocke_winograd_gen_num_tiles(c.s) / rocke_winograd_gen_tiles_w_in(c.s)
                      * tw_tiles * num_vecs;
    tc.ok = cmp_ltc(c, gid, total);
    tc.cv = num_vecs > 1 ? modc(c, gid, num_vecs) : ci(c, 0);
    rocke_value_t* r = num_vecs > 1 ? divc(c, gid, num_vecs) : gid;
    tc.tj = modc(c, r, tw_tiles);
    r = divc(c, r, tw_tiles);
    tc.ti = th > 1 ? modc(c, r, th) : ci(c, 0);
    r = th > 1 ? divc(c, r, th) : r;
    tc.bs = p.dW > 1 ? modc(c, r, p.dW) : ci(c, 0);
    r = p.dW > 1 ? divc(c, r, p.dW) : r;
    tc.a = p.dH > 1 ? modc(c, r, p.dH) : ci(c, 0);
    tc.n = p.dH > 1 ? divc(c, r, p.dH) : r;
    rocke_value_t* x = mulc(c, tc.n, p.dH);
    x = rocke_b_add(b, x, tc.a);
    x = mulc(c, x, p.dW);
    x = rocke_b_add(b, x, tc.bs);
    x = mulc(c, x, th);
    rocke_value_t* row = rocke_b_add(b, x, tc.ti);
    rocke_value_t* rt = mulc(c, row, rocke_winograd_gen_tiles_w_in(c.s));
    tc.t = rocke_b_add(b, rt, tc.tj);
    return tc;
}

rocke_value_t* lin(Ctx& c, const std::vector<std::pair<double, rocke_value_t*>>& terms)
{
    return rocke_wino::lincomb_f(c.b, terms);
}

/* K1: w[k][ky][kx][c] -> U[G'][s][kg][c] */
void build_filter(Ctx& c)
{
    rocke_ir_builder_t* b = c.b;
    const rocke_winograd_gen_spec_t* s = c.s;
    const rocke_conv_problem_t& p = s->problem;
    int vec = vec_cg(s);
    int cg = cpg(s), kg = kpg(s), g = p.groups;
    int cgv = cg / vec;
    long long items = (long long)p.K * cgv;
    int S = s->ax_x.seg;
    const ToomCook* ty = ax_tc(s->ax_y);
    const ToomCook* tx = ax_tc(s->ax_x);
    int ny = ax_n(s->ax_y), nx = ax_n(s->ax_x);
    int r1x = ax_r1(s->ax_x);

    rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", s->block_size);
    rocke_value_t* w = param_ptr(c, "w", io_dt(s), true);
    rocke_value_t* u = param_ptr(c, "u", op_dt(s), false);
    rocke_value_t* w_rsrc = rsrc_of(c, w, "w_bytes");
    rocke_value_t* u_rsrc = rsrc_of(c, u, "u_bytes");
    rocke_value_t* bx = rocke_b_block_id_x(b);
    rocke_value_t* m0 = mulc(c, bx, s->block_size);
    rocke_value_t* tid = rocke_b_thread_id_x(b);
    rocke_value_t* gid = rocke_b_add(b, m0, tid);
    rocke_value_t* s_idx = rocke_b_block_id_y(b);
    rocke_value_t* ok = cmp_ltc(c, gid, items);
    rocke_value_t* k = divc(c, gid, cgv);
    rocke_value_t* cc = modc(c, gid, cgv);
    cc = mulc(c, cc, vec);
    rocke_value_t* wb = mulc(c, k, (int64_t)p.Y * p.X * cg);
    wb = rocke_b_add(b, wb, cc);
    wb = mulc(c, wb, 2);
    rocke_value_t* wbase = rocke_wino::oob_sel(b, ok, wb);
    rocke_value_t* kx0 = S > 1 ? mulc(c, s_idx, r1x) : ci(c, 0);
    std::vector<std::vector<Vals>> taps(p.Y);
    for(int ky = 0; ky < p.Y; ++ky)
    {
        for(int q = 0; q < r1x; ++q)
        {
            rocke_value_t* kx = addc(c, kx0, q);
            rocke_value_t* okx = cmp_ltc(c, kx, p.X);
            rocke_value_t* k0 = ci(c, (int64_t)ky * p.X);
            rocke_value_t* off = rocke_b_add(b, k0, kx);
            off = mulc(c, off, (int64_t)cg * 2);
            off = rocke_b_add(b, wbase, off);
            off = rocke_wino::oob_sel(b, okx, off);
            rocke_value_t* raw = rocke_wino::load_raw(b, w_rsrc, off, io_dt(s), vec);
            taps[ky].push_back(rocke_wino::raw_to_f32(b, raw, vec));
        }
    }
    /* tmp[ky][jx][ch] = sum_q Gx[jx][q] g[ky][q] */
    std::vector<std::vector<Vals>> tmp(p.Y, std::vector<Vals>(nx));
    for(int ky = 0; ky < p.Y; ++ky)
        for(int jx = 0; jx < nx; ++jx)
            for(int ch = 0; ch < vec; ++ch)
            {
                std::vector<std::pair<double, rocke_value_t*>> t;
                for(int q = 0; q < r1x; ++q)
                    t.emplace_back(tx->G(jx, q).value(), taps[ky][q][ch]);
                tmp[ky][jx].push_back(lin(c, t));
            }
    rocke_value_t* gi = g > 1 ? divc(c, k, kg) : ci(c, 0);
    rocke_value_t* kk = g > 1 ? modc(c, k, kg) : k;
    for(int jy = 0; jy < ny; ++jy)
    {
        for(int jx = 0; jx < nx; ++jx)
        {
            int bin = jy * nx + jx;
            Vals vals;
            for(int ch = 0; ch < vec; ++ch)
            {
                std::vector<std::pair<double, rocke_value_t*>> t;
                for(int ky = 0; ky < p.Y; ++ky)
                    t.emplace_back(ty->G(jy, ky).value(), tmp[ky][jx][ch]);
                vals.push_back(lin(c, t));
            }
            rocke_value_t* bg = ci(c, (int64_t)bin * g);
            rocke_value_t* gp = rocke_b_add(b, bg, gi);
            rocke_value_t* row = mulc(c, gp, S);
            row = rocke_b_add(b, row, s_idx);
            row = mulc(c, row, kg);
            row = rocke_b_add(b, row, kk);
            rocke_value_t* off = mulc(c, row, cg);
            off = rocke_b_add(b, off, cc);
            off = mulc(c, off, 2);
            off = rocke_wino::oob_sel(b, ok, off);
            rocke_value_t* packed = rocke_wino::pack_vec_f32(b, vals, op_dt(s));
            rocke_wino::store_packed(b, u_rsrc, off, packed, op_dt(s), 0);
        }
    }
}

/* K2: V[t][bin][c] = (B_y^T d B_x)[bin] */
void build_input(Ctx& c)
{
    rocke_ir_builder_t* b = c.b;
    const rocke_winograd_gen_spec_t* s = c.s;
    const rocke_conv_problem_t& p = s->problem;
    int vec = vec_c(s);
    const ToomCook* ty = ax_tc(s->ax_y);
    const ToomCook* tx = ax_tc(s->ax_x);
    int ny = ax_n(s->ax_y), nx = ax_n(s->ax_x);
    int my = s->ax_y.m, mx = s->ax_x.m;
    int bins = ny * nx;

    rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", s->block_size);
    rocke_value_t* x = param_ptr(c, "x", io_dt(s), true);
    rocke_value_t* v = param_ptr(c, "v", op_dt(s), false);
    rocke_value_t* x_rsrc = rsrc_of(c, x, "x_bytes");
    rocke_value_t* v_rsrc = rsrc_of(c, v, "v_bytes");
    TileCoords tc = flat_tile(c, p.C / vec, rocke_winograd_gen_tiles_w_in(s));
    rocke_value_t* cc = mulc(c, tc.cv, vec);
    rocke_value_t* img = mulc(c, tc.n, p.Hi);
    rocke_value_t* h = mulc(c, tc.ti, (int64_t)p.dH * my);
    h = rocke_b_add(b, tc.a, h);
    rocke_value_t* hi0 = subc(c, h, p.pH);
    Vals rows;
    for(int uu = 0; uu < ny; ++uu)
    {
        rocke_value_t* hi = addc(c, hi0, (int64_t)uu * p.dH);
        rocke_value_t* ge = rocke_b_cmp_ge(b, hi, ci(c, 0));
        rocke_value_t* lt = cmp_ltc(c, hi, p.Hi);
        rocke_value_t* in = rocke_b_land(b, ge, lt);
        rocke_value_t* okr = rocke_b_land(b, tc.ok, in);
        rocke_value_t* off = rocke_b_add(b, img, hi);
        off = mulc(c, off, p.Wi);
        off = mulc(c, off, p.C);
        off = rocke_b_add(b, off, cc);
        off = mulc(c, off, 2);
        rows.push_back(rocke_wino::oob_sel(b, okr, off));
    }
    rocke_value_t* w = mulc(c, tc.tj, (int64_t)p.dW * mx);
    w = rocke_b_add(b, tc.bs, w);
    rocke_value_t* wi0 = subc(c, w, p.pW);
    Vals cols;
    for(int vv = 0; vv < nx; ++vv)
    {
        rocke_value_t* wi = addc(c, wi0, (int64_t)vv * p.dW);
        rocke_value_t* ge = rocke_b_cmp_ge(b, wi, ci(c, 0));
        rocke_value_t* lt = cmp_ltc(c, wi, p.Wi);
        rocke_value_t* okc = rocke_b_land(b, ge, lt);
        rocke_value_t* off = mulc(c, wi, (int64_t)p.C * 2);
        cols.push_back(rocke_wino::oob_sel(b, okc, off));
    }
    std::vector<std::vector<Vals>> data(ny);
    for(int uu = 0; uu < ny; ++uu)
        for(int vv = 0; vv < nx; ++vv)
        {
            rocke_value_t* off = rocke_b_add(b, rows[uu], cols[vv]);
            rocke_value_t* raw = rocke_wino::load_raw(b, x_rsrc, off, io_dt(s), vec);
            data[uu].push_back(rocke_wino::raw_to_f32(b, raw, vec));
        }
    rocke_b_sched_barrier(b, 0);
    std::vector<std::vector<Vals>> txv(ny, std::vector<Vals>(nx));
    for(int uu = 0; uu < ny; ++uu)
        for(int jx = 0; jx < nx; ++jx)
            for(int ch = 0; ch < vec; ++ch)
            {
                std::vector<std::pair<double, rocke_value_t*>> t;
                for(int vv = 0; vv < nx; ++vv)
                    t.emplace_back(tx->BT(jx, vv).value(), data[uu][vv][ch]);
                txv[uu][jx].push_back(lin(c, t));
            }
    rocke_value_t* vb = mulc(c, tc.t, (int64_t)bins * p.C);
    vb = rocke_b_add(b, vb, cc);
    vb = mulc(c, vb, 2);
    rocke_value_t* vbase = rocke_wino::oob_sel(b, tc.ok, vb);
    std::vector<std::pair<rocke_value_t*, rocke_value_t*>> pending;
    for(int jy = 0; jy < ny; ++jy)
        for(int jx = 0; jx < nx; ++jx)
        {
            Vals vals;
            for(int ch = 0; ch < vec; ++ch)
            {
                std::vector<std::pair<double, rocke_value_t*>> t;
                for(int uu = 0; uu < ny; ++uu)
                    t.emplace_back(ty->BT(jy, uu).value(), txv[uu][jx][ch]);
                vals.push_back(lin(c, t));
            }
            rocke_value_t* off = addc(c, vbase, (int64_t)(jy * nx + jx) * p.C * 2);
            rocke_value_t* packed = rocke_wino::pack_vec_f32(b, vals, op_dt(s));
            pending.emplace_back(off, packed);
        }
    rocke_b_sched_barrier(b, 0);
    int cpol = s->nt_stores ? rocke_wino::kCpolNt : 0;
    for(auto& pr : pending)
        rocke_wino::store_packed(b, v_rsrc, pr.first, pr.second, op_dt(s), cpol);
}

/* K4: y = A_y^T M A_x per tile (halo tiles skipped), cropped. */
void build_output(Ctx& c)
{
    rocke_ir_builder_t* b = c.b;
    const rocke_winograd_gen_spec_t* s = c.s;
    const rocke_conv_problem_t& p = s->problem;
    int vec = vec_k(s);
    const ToomCook* ty = ax_tc(s->ax_y);
    const ToomCook* tx = ax_tc(s->ax_x);
    int ny = ax_n(s->ax_y), nx = ax_n(s->ax_x);
    int my = s->ax_y.m, mx = s->ax_x.m;
    int bins = ny * nx;
    int Ho = ho(s), Wo = wo(s);

    rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", s->block_size);
    rocke_value_t* mm = param_ptr(c, "mm", op_dt(s), true);
    rocke_value_t* y = param_ptr(c, "y", io_dt(s), false);
    rocke_value_t* m_rsrc = rsrc_of(c, mm, "m_bytes");
    rocke_value_t* y_rsrc = rsrc_of(c, y, "y_bytes");
    TileCoords tc = flat_tile(c, p.K / vec, rocke_winograd_gen_tiles_w(s));
    rocke_value_t* k = mulc(c, tc.cv, vec);
    rocke_value_t* mb = mulc(c, tc.t, (int64_t)bins * p.K);
    mb = rocke_b_add(b, mb, k);
    mb = mulc(c, mb, 2);
    rocke_value_t* mbase = rocke_wino::oob_sel(b, tc.ok, mb);
    std::vector<Vals> raw(ny);
    for(int uu = 0; uu < ny; ++uu)
        for(int vv = 0; vv < nx; ++vv)
        {
            rocke_value_t* off = addc(c, mbase, (int64_t)(uu * nx + vv) * p.K * 2);
            raw[uu].push_back(rocke_wino::load_raw(b, m_rsrc, off, op_dt(s), vec));
        }
    rocke_value_t* h = mulc(c, tc.ti, (int64_t)p.dH * my);
    rocke_value_t* ho0 = rocke_b_add(b, tc.a, h);
    rocke_value_t* w = mulc(c, tc.tj, (int64_t)p.dW * mx);
    rocke_value_t* wo0 = rocke_b_add(b, tc.bs, w);
    rocke_value_t* img = mulc(c, tc.n, Ho);
    Vals yrow, ycol;
    for(int i = 0; i < my; ++i)
    {
        rocke_value_t* hv = addc(c, ho0, (int64_t)i * p.dH);
        rocke_value_t* lt = cmp_ltc(c, hv, Ho);
        rocke_value_t* okr = rocke_b_land(b, tc.ok, lt);
        rocke_value_t* off = rocke_b_add(b, img, hv);
        off = mulc(c, off, Wo);
        off = mulc(c, off, p.K);
        off = rocke_b_add(b, off, k);
        off = mulc(c, off, 2);
        yrow.push_back(rocke_wino::oob_sel(b, okr, off));
    }
    for(int j = 0; j < mx; ++j)
    {
        rocke_value_t* wv = addc(c, wo0, (int64_t)j * p.dW);
        rocke_value_t* lt = cmp_ltc(c, wv, Wo);
        rocke_value_t* off = mulc(c, wv, (int64_t)p.K * 2);
        ycol.push_back(rocke_wino::oob_sel(b, lt, off));
    }
    rocke_b_sched_barrier(b, 0);
    std::vector<std::vector<Vals>> data(ny);
    for(int uu = 0; uu < ny; ++uu)
        for(int vv = 0; vv < nx; ++vv)
            data[uu].push_back(rocke_wino::raw_to_f32(b, raw[uu][vv], vec));
    std::vector<std::vector<Vals>> txv(ny, std::vector<Vals>(mx));
    for(int uu = 0; uu < ny; ++uu)
        for(int j = 0; j < mx; ++j)
            for(int ch = 0; ch < vec; ++ch)
            {
                std::vector<std::pair<double, rocke_value_t*>> t;
                for(int vv = 0; vv < nx; ++vv)
                    t.emplace_back(tx->AT(j, vv).value(), data[uu][vv][ch]);
                txv[uu][j].push_back(lin(c, t));
            }
    std::vector<std::pair<rocke_value_t*, rocke_value_t*>> pending;
    for(int i = 0; i < my; ++i)
        for(int j = 0; j < mx; ++j)
        {
            Vals vals;
            for(int ch = 0; ch < vec; ++ch)
            {
                std::vector<std::pair<double, rocke_value_t*>> t;
                for(int uu = 0; uu < ny; ++uu)
                    t.emplace_back(ty->AT(i, uu).value(), txv[uu][j][ch]);
                vals.push_back(lin(c, t));
            }
            rocke_value_t* off = rocke_b_add(b, yrow[i], ycol[j]);
            rocke_value_t* packed = rocke_wino::pack_vec_f32(b, vals, io_dt(s));
            pending.emplace_back(off, packed);
        }
    rocke_b_sched_barrier(b, 0);
    int cpol = s->nt_stores ? rocke_wino::kCpolNt : 0;
    for(auto& pr : pending)
        rocke_wino::store_packed(b, y_rsrc, pr.first, pr.second, io_dt(s), cpol);
}

} // namespace

/* ===================================================================== *
 *  Spec defaults, geometry, validity
 * ===================================================================== */

rocke_winograd_gen_spec_t rocke_winograd_gen_spec_default(void)
{
    rocke_winograd_gen_spec_t s;
    memset(&s, 0, sizeof(s));
    s.problem.groups = 1;
    s.problem.sH = s.problem.sW = 1;
    s.problem.dH = s.problem.dW = 1;
    s.ax_y = {1, 1, 1};
    s.ax_x = {1, 1, 1};
    s.io_dtype = "fp16";
    s.operand_dtype = "fp16";
    s.vec = 4;
    s.block_size = 256;
    s.nt_stores = true;
    s.name = "conv_winograd_gen";
    return s;
}

int rocke_winograd_gen_bins(const rocke_winograd_gen_spec_t* spec)
{
    return ax_n(spec->ax_y) * ax_n(spec->ax_x);
}

int rocke_winograd_gen_tiles_h(const rocke_winograd_gen_spec_t* spec)
{
    return (int)cdiv(cdiv(ho(spec), spec->problem.dH), spec->ax_y.m);
}

int rocke_winograd_gen_tiles_w(const rocke_winograd_gen_spec_t* spec)
{
    return (int)cdiv(cdiv(wo(spec), spec->problem.dW), spec->ax_x.m);
}

int rocke_winograd_gen_tiles_w_in(const rocke_winograd_gen_spec_t* spec)
{
    return rocke_winograd_gen_tiles_w(spec) + spec->ax_x.seg - 1;
}

long long rocke_winograd_gen_num_tiles(const rocke_winograd_gen_spec_t* spec)
{
    const rocke_conv_problem_t& p = spec->problem;
    return (long long)p.N * p.dH * p.dW * rocke_winograd_gen_tiles_h(spec)
           * rocke_winograd_gen_tiles_w_in(spec);
}

bool rocke_winograd_gen_is_valid_spec(const rocke_winograd_gen_spec_t* spec,
                                      char* reason,
                                      size_t reason_cap)
{
    if(spec == NULL)
    {
        set_reason(reason, reason_cap, "null spec");
        return false;
    }
    const rocke_conv_problem_t& p = spec->problem;
    if(p.groups < 1 || p.C % p.groups || p.K % p.groups)
    {
        set_reason(reason, reason_cap, "bad ConvProblem groups");
        return false;
    }
    if(p.is_3d)
    {
        set_reason(reason, reason_cap, "2-D convolutions only");
        return false;
    }
    if(p.sH != 1 || p.sW != 1)
    {
        set_reason(reason, reason_cap, "stride 1 only");
        return false;
    }
    for(const rocke_wino_axis_t* a : {&spec->ax_y, &spec->ax_x})
    {
        std::string why = ax_check(*a);
        if(!why.empty())
        {
            set_reason(reason, reason_cap, why);
            return false;
        }
    }
    if(spec->ax_y.r != p.Y || spec->ax_x.r != p.X)
    {
        set_reason(reason, reason_cap, "axis taps must match the filter");
        return false;
    }
    if(spec->ax_y.seg != 1)
    {
        set_reason(reason, reason_cap, "segments are supported along W only");
        return false;
    }
    if(!is_io(spec->io_dtype) || !is_io(spec->operand_dtype))
    {
        set_reason(reason, reason_cap, "dtype must be fp16 / bf16");
        return false;
    }
    if(!(spec->vec == 1 || spec->vec == 2 || spec->vec == 4 || spec->vec == 8))
    {
        set_reason(reason, reason_cap, "vec must be 1/2/4/8");
        return false;
    }
    if(spec->block_size < 32 || (spec->block_size & (spec->block_size - 1)))
    {
        set_reason(reason, reason_cap, "block_size must be a power of two >= 32");
        return false;
    }
    if(ho(spec) <= 0 || wo(spec) <= 0)
    {
        set_reason(reason, reason_cap, "empty output");
        return false;
    }
    {
        long long t = rocke_winograd_gen_num_tiles(spec);
        long long bins = rocke_winograd_gen_bins(spec);
        long long S = spec->ax_x.seg;
        long long cg = cpg(spec);
        const char* names[] = {"x", "w", "y", "U", "V", "M", "e"};
        long long bytes[] = {(long long)p.N * p.Hi * p.Wi * p.C * 2,
                             (long long)p.K * p.Y * p.X * cg * 2,
                             (long long)p.N * ho(spec) * wo(spec) * p.K * 2,
                             bins * S * p.K * cg * 2,
                             (t + S - 1) * bins * p.C * 2,
                             t * bins * p.K * 2,
                             4};
        for(int i = 0; i < 7; ++i)
        {
            if(bytes[i] > kMaxBufferBytes)
            {
                set_reason(reason,
                           reason_cap,
                           std::string(names[i]) + " needs " + std::to_string(bytes[i])
                               + " bytes (> 32-bit offsets)");
                return false;
            }
        }
    }
    set_reason(reason, reason_cap, "ok");
    return true;
}

rocke_status_t rocke_winograd_gen_stage_grid(const rocke_winograd_gen_spec_t* spec,
                                             const char* stage,
                                             int grid[3])
{
    const char* w = stage_word(stage);
    if(spec == NULL || w == NULL)
        return ROCKE_ERR_VALUE;
    const rocke_conv_problem_t& p = spec->problem;
    long long items;
    int gy = 1;
    if(strcmp(stage, "K1") == 0)
    {
        items = (long long)p.K * (cpg(spec) / vec_cg(spec));
        gy = spec->ax_x.seg;
    }
    else if(strcmp(stage, "K2") == 0)
        items = rocke_winograd_gen_num_tiles(spec) * (p.C / vec_c(spec));
    else
        items = rocke_winograd_gen_num_tiles(spec) / rocke_winograd_gen_tiles_w_in(spec)
                * rocke_winograd_gen_tiles_w(spec) * (p.K / vec_k(spec));
    long long gx = cdiv(items, spec->block_size);
    if(gx <= 0)
        return ROCKE_ERR_VALUE;
    grid[0] = (int)gx;
    grid[1] = gy;
    grid[2] = 1;
    return ROCKE_OK;
}

rocke_status_t rocke_winograd_gen_kernel_name(const rocke_winograd_gen_spec_t* spec,
                                              const char* stage,
                                              char* out,
                                              size_t out_cap)
{
    const char* w = stage_word(stage);
    if(spec == NULL || w == NULL || out == NULL)
        return ROCKE_ERR_VALUE;
    std::string name = std::string(spec->name ? spec->name : "conv_winograd_gen") + "_" + w + "_"
                       + spec_short(spec);
    for(char& ch : name)
        if(ch == '/')
            ch = '_';
    if(name.size() + 1 > out_cap)
        return ROCKE_ERR_VALUE;
    memcpy(out, name.c_str(), name.size() + 1);
    return ROCKE_OK;
}

/* ===================================================================== *
 *  Builders
 * ===================================================================== */

rocke_kernel_def_t* rocke_build_winograd_gen_stage(rocke_ir_builder_t* b,
                                                   const rocke_winograd_gen_spec_t* spec,
                                                   const char* stage,
                                                   const char* arch)
{
    char reason[512];
    (void)arch; /* the transforms do not depend on the target (as in Python) */
    if(b == NULL || spec == NULL)
        return NULL;
    if(stage_word(stage) == NULL)
    {
        rocke_i_set_err_msg(b, ROCKE_ERR_VALUE, "conv_winograd_gen: stage must be K1, K2 or K4");
        return NULL;
    }
    if(!rocke_winograd_gen_is_valid_spec(spec, reason, sizeof reason))
    {
        std::string msg = std::string("invalid WinogradGenSpec: ") + reason;
        rocke_i_set_err_msg(b, ROCKE_ERR_VALUE, msg.c_str());
        return NULL;
    }
    Ctx c{b, spec};
    if(strcmp(stage, "K1") == 0)
        build_filter(c);
    else if(strcmp(stage, "K2") == 0)
        build_input(c);
    else
        build_output(c);
    if(!rocke_ir_builder_ok(b))
        return NULL;
    return b->kernel;
}

rocke_kernel_def_t* rocke_build_winograd_gen_stage_new(rocke_ir_builder_t* b,
                                                       const rocke_winograd_gen_spec_t* spec,
                                                       const char* stage,
                                                       const char* arch)
{
    if(b != NULL)
        memset(b, 0, sizeof(*b));
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        char name[1024];
        if(b == NULL)
            return NULL;
        if(spec == NULL)
        {
            rocke_i_set_err_msg(b, ROCKE_ERR_VALUE, "conv_winograd_gen: null spec");
            return NULL;
        }
        if(rocke_winograd_gen_kernel_name(spec, stage, name, sizeof name) != ROCKE_OK)
        {
            rocke_ir_builder_init(b, "conv_winograd_gen");
            rocke_i_set_err_msg(b, ROCKE_ERR_VALUE, "conv_winograd_gen: bad stage or name");
            return NULL;
        }
        if(rocke_ir_builder_init(b, name) != ROCKE_OK)
            return NULL;
        return rocke_build_winograd_gen_stage(b, spec, stage, arch);
    });
}

rocke_status_t rocke_winograd_gen_lower_to_llvm(const rocke_winograd_gen_spec_t* spec,
                                                const char* stage,
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
    if(spec == NULL || out_ll == NULL || arch == NULL)
    {
        set_reason(err, err_cap, "lower_to_llvm: null spec/out/arch");
        return ROCKE_ERR_VALUE;
    }
    kernel = rocke_build_winograd_gen_stage_new(&b, spec, stage, arch);
    if(kernel == NULL)
    {
        const char* m = rocke_ir_builder_error(&b);
        st = rocke_ir_builder_status(&b);
        set_reason(err, err_cap, (m != NULL && m[0] != '\0') ? m : "conv_winograd_gen failed");
        rocke_ir_builder_free(&b);
        return (st == ROCKE_OK) ? ROCKE_ERR_VALUE : st;
    }
    st = rocke_lower_kernel_to_llvm_ex(kernel, flavor, arch, out_ll, err, err_cap);
    rocke_ir_builder_free(&b);
    return st;
}
