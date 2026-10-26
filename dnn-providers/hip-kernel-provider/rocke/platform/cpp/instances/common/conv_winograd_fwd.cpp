// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * conv_winograd_fwd.cpp -- C++ port of the Winograd F(mh x mw, 3 x 3) forward
 * transform kernels (library/kernels/common/conv_winograd_fwd.py): K1 filter
 * transform, K2a per-pixel max, K2 input transform (register and LDS-region
 * paths, fused or split prescale), K4 output transform, and the merged launch
 * of independent stages. See the header and the Python module.
 *
 * Every emitter runs in PYTHON EVALUATION ORDER (arguments left to right,
 * innermost call first): nested builder calls are hoisted into ordered
 * locals, since C++ leaves the evaluation order of call arguments
 * unspecified. The merged kernel re-emits the stage bodies through a context
 * that substitutes the stage's block / thread ids and shares parameters of
 * the same name (the Python _SubBuilder).
 */
#include "rocke/instance_conv_winograd_fwd.h"

#include <ctype.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <string>
#include <vector>

#include "rocke/error_boundary.hpp" /* ckc::guard_builder */
#include "rocke/helper_rocke.helpers.spec.h" /* rocke_kernel_name_join */
#include "rocke/instance_conv_winograd_internal.h"
#include "rocke/ir_internal.h" /* rocke_i_set_err_msg */
#include "rocke/lower_llvm.h"

namespace
{

using rocke_wino::kCpolNt;
using rocke_wino::kOob;
using Spec = rocke_winograd_fwd_spec_t;
using V = rocke_value_t*;
using VV = std::vector<rocke_value_t*>;

constexpr int kFilterBlock = 128;
constexpr int kPrescaleBlock = 256;
constexpr int kPrescaleLog2Limit = 14;
constexpr int kPrescaleMaxExp = 100;
constexpr long long kMaxBufferBytes = 0x7FFF0000LL;

long long cdiv(long long a, long long b)
{
    return (a + b - 1) / b;
}

/* Python floor division. */
long long fdiv(long long a, long long b)
{
    long long q = a / b;
    if((a % b != 0) && ((a < 0) != (b < 0)))
        --q;
    return q;
}

void set_reason(char* reason, size_t cap, const std::string& msg)
{
    if(reason && cap > 0)
        snprintf(reason, cap, "%s", msg.c_str());
}

/* ------------------------------------------------------------ geometry */

int cpg(const Spec* s)
{
    return s->problem.C / s->problem.groups;
}
int Ho(const Spec* s)
{
    const rocke_conv_problem_t& p = s->problem;
    return (int)(fdiv((long long)p.Hi + 2 * p.pH - (long long)p.dH * (p.Y - 1) - 1, p.sH) + 1);
}
int Wo(const Spec* s)
{
    const rocke_conv_problem_t& p = s->problem;
    return (int)(fdiv((long long)p.Wi + 2 * p.pW - (long long)p.dW * (p.X - 1) - 1, p.sW) + 1);
}
int mh(const Spec* s)
{
    return s->m;
}
int mw(const Spec* s)
{
    return s->m_w ? s->m_w : s->m;
}
int nh(const Spec* s)
{
    return mh(s) + 2;
}
int nw(const Spec* s)
{
    return mw(s) + 2;
}
int n2(const Spec* s)
{
    return nh(s) * nw(s);
}
int tiles_h(const Spec* s)
{
    return (int)cdiv(cdiv(Ho(s), s->problem.dH), mh(s));
}
int tiles_w(const Spec* s)
{
    return (int)cdiv(cdiv(Wo(s), s->problem.dW), mw(s));
}
long long num_tiles(const Spec* s)
{
    const rocke_conv_problem_t& p = s->problem;
    return (long long)p.N * p.dH * p.dW * tiles_h(s) * tiles_w(s);
}
int vec_c(const Spec* s)
{
    return rocke_wino::pow2_vec(s->problem.C, s->vec);
}
int vec_k(const Spec* s)
{
    return rocke_wino::pow2_vec(s->problem.K, s->vec);
}
int vec_cg(const Spec* s)
{
    return rocke_wino::pow2_vec(cpg(s), s->filter_vec ? s->filter_vec : s->vec);
}
bool input_lds(const Spec* s)
{
    return s->input_block_tiles[0] > 0;
}
bool prescale_split(const Spec* s)
{
    if(strcmp(s->prescale_mode ? s->prescale_mode : "fused", "split") == 0 || input_lds(s))
        return true;
    return rocke_wino::pow2_ceil(s->problem.C / vec_c(s)) > 1024;
}
double beta_of(int m)
{
    return m == 2 ? 4.0 : (m == 4 ? 100.0 : 225.0);
}
double prescale_beta(const Spec* s)
{
    return sqrt(beta_of(mh(s)) * beta_of(mw(s)));
}
int prescale_limit_log2(const Spec* s)
{
    int headroom = (int)ceil(log2((double)cpg(s)) / 2);
    return kPrescaleLog2Limit - headroom;
}
const char* io_dt(const Spec* s)
{
    return s->io_dtype ? s->io_dtype : "fp16";
}
const char* op_dt(const Spec* s)
{
    return s->operand_dtype ? s->operand_dtype : "fp16";
}

/* buffer_bytes() in its key order. */
void buffer_bytes(const Spec* s, long long out[7])
{
    const rocke_conv_problem_t& p = s->problem;
    long long t = num_tiles(s);
    out[0] = (long long)p.N * p.Hi * p.Wi * p.C * 2;
    out[1] = (long long)p.K * 9 * cpg(s) * 2;
    out[2] = (long long)p.N * Ho(s) * Wo(s) * p.K * 2;
    out[3] = (long long)n2(s) * p.K * cpg(s) * 2;
    out[4] = t * n2(s) * p.C * 2;
    out[5] = t * n2(s) * p.K * 2;
    out[6] = t * 4 + ((s->prescale && prescale_split(s)) ? (long long)p.N * p.Hi * p.Wi * 4 : 0);
}
const char* const kBufNames[7] = {"x", "w", "y", "U", "V", "M", "e"};

std::string problem_short(const rocke_conv_problem_t& p)
{
    char buf[160];
    char g[32] = "";
    if(p.groups > 1)
        snprintf(g, sizeof g, "G%d", p.groups);
    snprintf(buf, sizeof buf, "N%dH%dW%dC%d_K%dY%dX%d%s", p.N, p.Hi, p.Wi, p.C, p.K, p.Y, p.X, g);
    return buf;
}

std::string spec_short(const Spec* s)
{
    std::vector<std::string> tags;
    char buf[64];
    tags.push_back(problem_short(s->problem));
    if(mw(s) == mh(s))
        snprintf(buf, sizeof buf, "f%d", s->m);
    else
        snprintf(buf, sizeof buf, "f%dx%d", mh(s), mw(s));
    tags.push_back(buf);
    tags.push_back(io_dt(s));
    if(strcmp(op_dt(s), io_dt(s)) != 0)
        tags.push_back(std::string("op") + op_dt(s));
    tags.push_back("v" + std::to_string(s->vec));
    tags.push_back("b" + std::to_string(s->block_size));
    if(s->tiles_per_thread > 1)
        tags.push_back("r" + std::to_string(s->tiles_per_thread));
    if(input_lds(s))
        tags.push_back("lds" + std::to_string(s->input_block_tiles[0]) + "x"
                       + std::to_string(s->input_block_tiles[1]));
    if(s->wg_tile_rows > 1)
        tags.push_back("wr" + std::to_string(s->wg_tile_rows));
    if(!s->nt_stores)
        tags.push_back("rt");
    if(input_lds(s) && s->lds_lanes != 32)
        tags.push_back("ll" + std::to_string(s->lds_lanes));
    if(s->prescale)
        tags.push_back(prescale_split(s) ? "pss" : "ps");
    if(s->filter_rows != 1)
        tags.push_back("fr" + std::to_string(s->filter_rows));
    if(s->filter_vec)
        tags.push_back("fv" + std::to_string(s->filter_vec));
    std::string out;
    for(size_t i = 0; i < tags.size(); ++i)
        out += (i ? "_" : "") + tags[i];
    return out;
}

/* ------------------------------------------------------------ launch maps */

struct RunMap
{
    int tpc, runs_per_wg, cx, num_vecs, block, tiles, runs_per_row, num_runs, rows, wg_rows;
    int wg_runs() const
    {
        return runs_per_wg / wg_rows;
    }
    int run_blocks() const
    {
        return (int)cdiv(runs_per_row, wg_runs());
    }
};

bool run_map(
    const Spec* s, int num_vecs, int tiles, bool all_c, int wg_rows, RunMap* rm, std::string* err)
{
    int block = s->block_size;
    int tpc = rocke_wino::pow2_ceil(num_vecs);
    if(all_c)
    {
        if(tpc > 1024)
        {
            *err = "fused prescale needs C / vec <= 1024";
            return false;
        }
        block = std::max(block, tpc);
    }
    else
        tpc = std::min(tpc, block);
    int rpr = (int)cdiv(tiles_w(s), tiles);
    int rows = (int)(num_tiles(s) / tiles_w(s));
    int rpw = block / tpc;
    wg_rows = std::max(1, std::min(wg_rows, rpw));
    while(rpw % wg_rows)
        wg_rows -= 1;
    rm->tpc = tpc;
    rm->runs_per_wg = rpw;
    rm->cx = (int)cdiv(num_vecs, tpc);
    rm->num_vecs = num_vecs;
    rm->block = block;
    rm->tiles = tiles;
    rm->runs_per_row = rpr;
    rm->num_runs = rows * rpr;
    rm->rows = rows;
    rm->wg_rows = wg_rows;
    return true;
}

bool run_map_grid(const RunMap& rm, int grid[3], std::string* err)
{
    long long gx;
    if(rm.wg_rows > 1)
        gx = cdiv(rm.rows, rm.wg_rows) * rm.run_blocks() * rm.cx;
    else
        gx = cdiv(rm.num_runs, rm.runs_per_wg) * rm.cx;
    if(gx <= 0)
    {
        *err = "empty grid";
        return false;
    }
    grid[0] = (int)gx;
    grid[1] = 1;
    grid[2] = 1;
    return true;
}

bool input_run_map(const Spec* s, RunMap* rm, std::string* err)
{
    return run_map(s,
                   s->problem.C / vec_c(s),
                   s->tiles_per_thread,
                   s->prescale && !prescale_split(s),
                   s->wg_tile_rows,
                   rm,
                   err);
}

bool output_run_map(const Spec* s, RunMap* rm, std::string* err)
{
    return run_map(s, s->problem.K / vec_k(s), 1, false, 1, rm, err);
}

struct LdsMap
{
    int bh, bw, tpc, cc, cx, lv, rh, rw, nbh, nbw, block, loads, gx, tpt;
};

bool lds_map(const Spec* s, LdsMap* lm, std::string* err)
{
    const rocke_conv_problem_t& p = s->problem;
    int bh = s->input_block_tiles[0], bw = s->input_block_tiles[1];
    int vec = vec_c(s);
    int nv = p.C / vec;
    int tpc = std::min(rocke_wino::pow2_ceil(nv), s->lds_lanes);
    int cc = tpc * vec;
    int cx = (int)cdiv(nv, tpc);
    int lv = std::min(rocke_wino::pow2_vec(p.C, 8), cc);
    int rh = mh(s) * bh + nh(s) - mh(s);
    int rw = mw(s) * bw + nw(s) - mw(s);
    int nbh = (int)cdiv(tiles_h(s), bh);
    int nbw = (int)cdiv(tiles_w(s), bw);
    int tpt = s->tiles_per_thread;
    if((bh * bw) % tpt)
    {
        *err = "tiles_per_thread must divide the LDS block's tile count";
        return false;
    }
    int block = bh * bw * tpc / tpt;
    if(block > 1024)
    {
        *err = "input_block_tiles too large for one workgroup";
        return false;
    }
    long long pieces = (long long)rh * rw * (cc / lv);
    long long gx = (long long)p.N * p.dH * p.dW * nbh * nbw * cx;
    if(gx <= 0)
    {
        *err = "empty grid";
        return false;
    }
    *lm = LdsMap{
        bh, bw, tpc, cc, cx, lv, rh, rw, nbh, nbw, block, (int)cdiv(pieces, block), (int)gx, tpt};
    return true;
}

void prescale_lanes(const Spec* s, int* tpc, int* loads)
{
    int nv = s->problem.C / vec_c(s);
    *tpc = std::min(rocke_wino::pow2_ceil(nv), 32);
    *loads = (int)cdiv(nv, *tpc);
}

bool stage_launch(
    const Spec* s, rocke_winograd_fwd_stage_t st, int grid[3], int* block, std::string* err)
{
    const rocke_conv_problem_t& p = s->problem;
    switch(st)
    {
    case ROCKE_WINOGRAD_FWD_K1:
    {
        long long items = (long long)p.K * (cpg(s) / vec_cg(s));
        grid[0] = (int)cdiv(items, kFilterBlock);
        grid[1] = nh(s) / s->filter_rows;
        grid[2] = 1;
        *block = kFilterBlock;
        return true;
    }
    case ROCKE_WINOGRAD_FWD_K2A:
    {
        int tpc, loads;
        prescale_lanes(s, &tpc, &loads);
        long long gx = cdiv((long long)p.N * p.Hi * p.Wi, kPrescaleBlock / tpc);
        if(gx <= 0)
        {
            *err = "empty grid";
            return false;
        }
        grid[0] = (int)gx;
        grid[1] = 1;
        grid[2] = 1;
        *block = kPrescaleBlock;
        return true;
    }
    case ROCKE_WINOGRAD_FWD_K2:
    {
        if(input_lds(s))
        {
            LdsMap lm;
            if(!lds_map(s, &lm, err))
                return false;
            grid[0] = lm.gx;
            grid[1] = 1;
            grid[2] = 1;
            *block = lm.block;
            return true;
        }
        RunMap rm;
        if(!input_run_map(s, &rm, err) || !run_map_grid(rm, grid, err))
            return false;
        *block = rm.block;
        return true;
    }
    case ROCKE_WINOGRAD_FWD_K4:
    {
        RunMap rm;
        if(!output_run_map(s, &rm, err) || !run_map_grid(rm, grid, err))
            return false;
        *block = rm.block;
        return true;
    }
    }
    *err = "unknown stage";
    return false;
}

/* ------------------------------------------------------------ emission ctx */

struct E
{
    rocke_ir_builder_t* b;
    const Spec* s;
    bool sub; /* inside a merged kernel */
    V bid[3];
    V tid;
};

V ci(E& e, int64_t v)
{
    return rocke_b_const_i32(e.b, v);
}
V cf(E& e, double v)
{
    return rocke_b_const_f32(e.b, v);
}
V thread_x(E& e)
{
    return e.sub ? e.tid : rocke_b_thread_id_x(e.b);
}
V block_x(E& e)
{
    return e.sub ? e.bid[0] : rocke_b_block_id_x(e.b);
}
V block_y(E& e)
{
    return e.sub ? e.bid[1] : rocke_b_block_id_y(e.b);
}

/* b.param(name, t, noalias=..., readonly=..., writeonly=...); a merged kernel
 * shares a parameter of the same name. */
V param(E& e, const char* name, const rocke_type_t* t, int flags)
{
    rocke_ir_builder_t* b = e.b;
    if(e.sub)
    {
        for(int i = 0; i < b->num_param_lookup; i++)
        {
            if(b->param_names[i] && strcmp(b->param_names[i], name) == 0)
                return b->param_values[i];
        }
    }
    if(flags == 0)
        return rocke_b_param(b, name, t, NULL);
    rocke_param_opts_t o;
    memset(&o, 0, sizeof o);
    if(flags & 1)
    {
        o.noalias = true;
        o.noalias_set = true;
    }
    if(flags & 2)
    {
        o.readonly = true;
        o.readonly_set = true;
    }
    if(flags & 4)
    {
        o.writeonly = true;
        o.writeonly_set = true;
    }
    return rocke_b_param(b, name, t, &o);
}
constexpr int NOALIAS = 1, RO = 2, WO = 4;

const rocke_type_t* gptr(E& e, const rocke_type_t* elem)
{
    return rocke_ptr_type(e.b, elem, "global");
}

void set_mwgs(E& e, int v)
{
    rocke_attr_set_int(e.b, &rocke_ir_builder_kernel(e.b)->attrs, "max_workgroup_size", v);
}

/* _IROps */
V o_add(E& e, V x, V y)
{
    return rocke_b_fadd(e.b, x, y);
}
V o_sub(E& e, V x, V y)
{
    return rocke_b_fsub(e.b, x, y);
}
V o_fma(E& e, double c, V x, V y)
{
    V k = cf(e, c);
    return rocke_b_fma(e.b, k, x, y);
}

/* _matvec_1d over rows x cols of `mat` (row stride `cols`). */
VV matvec_1d(E& e, const rocke_wino::Rational* mat, int rows, int cols, const VV& d)
{
    VV out;
    for(int i = 0; i < rows; ++i)
    {
        std::vector<std::pair<double, V>> unit, rest;
        for(int k = 0; k < cols; ++k)
        {
            const rocke_wino::Rational& r = mat[i * cols + k];
            if(r.is_zero())
                continue;
            double c = r.value();
            if(fabs(c) == 1.0)
                unit.push_back({c, d[k]});
            else
                rest.push_back({c, d[k]});
        }
        V acc = nullptr;
        for(auto& t : unit)
        {
            if(acc == nullptr)
            {
                if(t.first > 0)
                    acc = t.second;
                else
                {
                    V z = o_sub(e, t.second, t.second);
                    acc = o_sub(e, z, t.second);
                }
            }
            else
                acc = t.first > 0 ? o_add(e, acc, t.second) : o_sub(e, acc, t.second);
        }
        for(auto& t : rest)
        {
            if(acc != nullptr)
                acc = o_fma(e, t.first, t.second, acc);
            else
            {
                V z = o_sub(e, t.second, t.second);
                acc = o_fma(e, t.first, t.second, z);
            }
        }
        out.push_back(acc != nullptr ? acc : o_sub(e, d[0], d[0]));
    }
    return out;
}

VV bt_1d(E& e, int m, const VV& d)
{
    if(m == 2)
    {
        V a = o_sub(e, d[2], d[0]);
        V b1 = o_add(e, d[1], d[2]);
        V c = o_sub(e, d[2], d[1]);
        V dd = o_sub(e, d[3], d[1]);
        return {a, b1, c, dd};
    }
    if(m == 4)
    {
        V s12 = o_add(e, d[1], d[2]);
        V s34 = o_add(e, d[3], d[4]);
        V d12 = o_sub(e, d[1], d[2]);
        V d43 = o_sub(e, d[4], d[3]);
        V d31 = o_sub(e, d[3], d[1]);
        V d42 = o_sub(e, d[4], d[2]);
        VV out;
        V t = o_fma(e, -5.0, d[2], d[4]);
        out.push_back(o_fma(e, 4.0, d[0], t));
        out.push_back(o_fma(e, -4.0, s12, s34));
        out.push_back(o_fma(e, 4.0, d12, d43));
        out.push_back(o_fma(e, 2.0, d31, d42));
        out.push_back(o_fma(e, -2.0, d31, d42));
        t = o_fma(e, -5.0, d[3], d[5]);
        out.push_back(o_fma(e, 4.0, d[1], t));
        return out;
    }
    const rocke_wino::ToomCook* tc = rocke_wino::toom_cook(m, 3);
    return matvec_1d(e, tc->bt, tc->n, tc->n, d);
}

VV at_1d(E& e, int m, const VV& d)
{
    if(m == 2)
    {
        V t = o_add(e, d[0], d[1]);
        V a = o_add(e, t, d[2]);
        V u = o_sub(e, d[1], d[2]);
        V c = o_add(e, u, d[3]);
        return {a, c};
    }
    if(m == 4)
    {
        V s12 = o_add(e, d[1], d[2]);
        V d12 = o_sub(e, d[1], d[2]);
        V s34 = o_add(e, d[3], d[4]);
        V d34 = o_sub(e, d[3], d[4]);
        VV out;
        V t = o_add(e, d[0], s12);
        out.push_back(o_add(e, t, s34));
        out.push_back(o_fma(e, 2.0, d34, d12));
        out.push_back(o_fma(e, 4.0, s34, s12));
        t = o_fma(e, 8.0, d34, d12);
        out.push_back(o_add(e, t, d[5]));
        return out;
    }
    const rocke_wino::ToomCook* tc = rocke_wino::toom_cook(m, 3);
    return matvec_1d(e, tc->at, tc->m, tc->n, d);
}

V oob_sel(E& e, V ok, V off)
{
    return rocke_wino::oob_sel(e.b, ok, off);
}
V load_raw(E& e, V rsrc, V off, const char* dt, int n)
{
    return rocke_wino::load_raw(e.b, rsrc, off, dt, n);
}

V pow2_f32(E& e, V x)
{
    rocke_ir_builder_t* b = e.b;
    V c127 = ci(e, 127);
    V a = rocke_b_add(b, x, c127);
    V c23 = ci(e, 23);
    V bits = rocke_b_shl(b, a, c23);
    return rocke_b_bitcast(b, bits, rocke_f32());
}

V abs_bits_vec(E& e, V raw, int n)
{
    rocke_ir_builder_t* b = e.b;
    if(n == 1)
    {
        V bc = rocke_b_bitcast(b, raw, rocke_i16());
        V z = rocke_b_zext(b, bc, rocke_i32());
        V m = ci(e, 0x7FFF);
        return rocke_b_land(b, z, m);
    }
    V bits = rocke_b_vec_bitcast(b, raw, rocke_vector_type(b, rocke_i16(), n));
    V c = ci(e, 0x7FFF);
    V t = rocke_b_trunc(b, c, rocke_i16());
    V mask = rocke_b_vector_splat(b, t, n);
    return rocke_b_vector_and(b, bits, mask);
}

V max_bits(E& e, V x, V y, int n)
{
    return n == 1 ? rocke_b_smax(e.b, x, y) : rocke_b_vector_smax(e.b, x, y);
}

V hmax_bits(E& e, V v, int n)
{
    if(n == 1)
        return v;
    V acc = nullptr;
    for(int i = 0; i < n; ++i)
    {
        V x = rocke_b_vec_extract(e.b, v, i);
        V z = rocke_b_zext(e.b, x, rocke_i32());
        acc = acc == nullptr ? z : rocke_b_smax(e.b, acc, z);
    }
    return acc;
}

V bits_to_f32(E& e, V bits, const char* dt)
{
    rocke_ir_builder_t* b = e.b;
    if(strcmp(dt, "bf16") == 0)
    {
        V c16 = ci(e, 16);
        V sh = rocke_b_shl(b, bits, c16);
        return rocke_b_bitcast(b, sh, rocke_f32());
    }
    V t = rocke_b_trunc(b, bits, rocke_i16());
    V h = rocke_b_bitcast(b, t, rocke_f16());
    return rocke_b_cast_to_f32(b, h);
}

V prescale_exp(E& e, V amax)
{
    rocke_ir_builder_t* b = e.b;
    const Spec* s = e.s;
    V k = cf(e, prescale_beta(s) / ldexp(1.0, prescale_limit_log2(s)));
    V sv = rocke_b_fmul(b, amax, k);
    V bits = rocke_b_bitcast(b, sv, rocke_i32());
    V c23 = ci(e, 23);
    V sh = rocke_b_lshr(b, bits, c23);
    V cff = ci(e, 0xFF);
    V an = rocke_b_land(b, sh, cff);
    V c127 = ci(e, 127);
    V expo = rocke_b_sub(b, an, c127);
    V cm = ci(e, 0x7FFFFF);
    V frac = rocke_b_land(b, bits, cm);
    V c0 = ci(e, 0);
    V ne = rocke_b_cmp_ne(b, frac, c0);
    V c1 = ci(e, 1);
    V c0b = ci(e, 0);
    V sel = rocke_b_select(b, ne, c1, c0b);
    V e_val = rocke_b_add(b, expo, sel);
    V z = ci(e, 0);
    V mx = rocke_b_smax(b, e_val, z);
    V cmax = ci(e, kPrescaleMaxExp);
    return rocke_b_smin(b, mx, cmax);
}

/* _wave_max_bits */
V wave_max_bits(E& e, V v, int lanes)
{
    int s = 1;
    while(s < lanes)
    {
        V sh = rocke_b_warp_shuffle_xor(e.b, v, s);
        v = rocke_b_smax(e.b, v, sh);
        s *= 2;
    }
    return v;
}

VV group_max_bits(E& e, VV vals, int tpc, int block, const char* name)
{
    rocke_ir_builder_t* b = e.b;
    for(auto& v : vals)
        v = wave_max_bits(e, v, std::min(tpc, 32));
    if(tpc <= 32)
        return vals;
    int rows = block / 32;
    int per = tpc / 32;
    int nv = (int)vals.size();
    int shape[1] = {rows * nv};
    V red = rocke_b_smem_alloc_f32(b, shape, 1, name);
    V tid = thread_x(e);
    V c32 = ci(e, 32);
    V row_id = rocke_b_div(b, tid, c32);
    {
        V c32b = ci(e, 32);
        V md = rocke_b_mod(b, tid, c32b);
        V c0 = ci(e, 0);
        V cond = rocke_b_cmp_eq(b, md, c0);
        rocke_if_t g = rocke_b_scf_if(b, cond);
        rocke_b_region_enter(b, g.then_region);
        for(int i = 0; i < nv; ++i)
        {
            V cnv = ci(e, nv);
            V mu = rocke_b_mul(b, row_id, cnv);
            V cii = ci(e, i);
            V idx = rocke_b_add(b, mu, cii);
            V bc = rocke_b_bitcast(b, vals[i], rocke_f32());
            V pk = rocke_b_vec_pack(b, &bc, 1, rocke_f32());
            rocke_b_smem_store_vN_f32(b, red, &idx, 1, pk, 1);
        }
        rocke_b_region_leave(b);
    }
    rocke_b_sync(b);
    V ctpc = ci(e, tpc);
    V dv = rocke_b_div(b, tid, ctpc);
    V cpn = ci(e, per * nv);
    V first = rocke_b_mul(b, dv, cpn);
    VV out;
    for(int i = 0; i < nv; ++i)
    {
        V acc = nullptr;
        for(int r = 0; r < per; ++r)
        {
            V c = ci(e, r * nv + i);
            V idx = rocke_b_add(b, first, c);
            V ld = rocke_b_smem_load_vN_f32(b, red, &idx, 1, 1);
            V ex = rocke_b_vec_extract(b, ld, 0);
            V x = rocke_b_bitcast(b, ex, rocke_i32());
            acc = acc == nullptr ? x : rocke_b_smax(b, acc, x);
        }
        out.push_back(acc);
    }
    return out;
}

/* _RunCoords */
struct RunCoords
{
    V ok, cv, c_ok, n, a, bsub, ti, tj0, t0;
};

RunCoords run_coords(E& e, const RunMap& rm)
{
    rocke_ir_builder_t* b = e.b;
    const Spec* s = e.s;
    const rocke_conv_problem_t& p = s->problem;
    RunCoords rc;
    V tid = thread_x(e);
    V bid = block_x(e);
    V lc, lr;
    if(rm.tpc > 1)
    {
        V c = ci(e, rm.tpc);
        lc = rocke_b_mod(b, tid, c);
        V c2 = ci(e, rm.tpc);
        lr = rocke_b_div(b, tid, c2);
    }
    else
    {
        lc = ci(e, 0);
        lr = tid;
    }
    V rb, cv;
    if(rm.cx > 1)
    {
        V c = ci(e, rm.cx);
        V cxi = rocke_b_mod(b, bid, c);
        V c2 = ci(e, rm.cx);
        rb = rocke_b_div(b, bid, c2);
        V c3 = ci(e, rm.tpc);
        V mu = rocke_b_mul(b, cxi, c3);
        cv = rocke_b_add(b, mu, lc);
    }
    else
    {
        rb = bid;
        cv = lc;
    }
    {
        V c = ci(e, rm.num_vecs);
        rc.c_ok = rocke_b_cmp_lt(b, cv, c);
    }
    V row, tj0, ok;
    if(rm.wg_rows > 1)
    {
        int wr = rm.wg_runs();
        V c = ci(e, rm.run_blocks());
        V rblk = rocke_b_mod(b, rb, c);
        V c2 = ci(e, rm.run_blocks());
        V hblk = rocke_b_div(b, rb, c2);
        V dr, dj;
        if(wr > 1)
        {
            V k = ci(e, wr);
            dr = rocke_b_div(b, lr, k);
        }
        else
            dr = lr;
        if(wr > 1)
        {
            V k = ci(e, wr);
            dj = rocke_b_mod(b, lr, k);
        }
        else
            dj = ci(e, 0);
        V k1 = ci(e, rm.wg_rows);
        V m1 = rocke_b_mul(b, hblk, k1);
        row = rocke_b_add(b, m1, dr);
        V k2 = ci(e, wr);
        V m2 = rocke_b_mul(b, rblk, k2);
        V jr = rocke_b_add(b, m2, dj);
        V k3 = ci(e, rm.rows);
        V l1 = rocke_b_cmp_lt(b, row, k3);
        V k4 = ci(e, rm.runs_per_row);
        V l2 = rocke_b_cmp_lt(b, jr, k4);
        ok = rocke_b_land(b, l1, l2);
        if(rm.tiles > 1)
        {
            V k = ci(e, rm.tiles);
            tj0 = rocke_b_mul(b, jr, k);
        }
        else
            tj0 = jr;
    }
    else if(rm.runs_per_row > 1)
    {
        V k = ci(e, rm.runs_per_wg);
        V mu = rocke_b_mul(b, rb, k);
        V rho = rocke_b_add(b, mu, lr);
        V k2 = ci(e, rm.num_runs);
        ok = rocke_b_cmp_lt(b, rho, k2);
        V k3 = ci(e, rm.runs_per_row);
        row = rocke_b_div(b, rho, k3);
        V k4 = ci(e, rm.runs_per_row);
        V jr = rocke_b_mod(b, rho, k4);
        if(rm.tiles > 1)
        {
            V k5 = ci(e, rm.tiles);
            tj0 = rocke_b_mul(b, jr, k5);
        }
        else
            tj0 = jr;
    }
    else
    {
        V k = ci(e, rm.runs_per_wg);
        V mu = rocke_b_mul(b, rb, k);
        V rho = rocke_b_add(b, mu, lr);
        V k2 = ci(e, rm.num_runs);
        ok = rocke_b_cmp_lt(b, rho, k2);
        row = rho;
        tj0 = ci(e, 0);
    }
    {
        V k = ci(e, tiles_w(s));
        V mu = rocke_b_mul(b, row, k);
        rc.t0 = rocke_b_add(b, mu, tj0);
    }
    V r = row;
    if(tiles_h(s) > 1)
    {
        V k = ci(e, tiles_h(s));
        rc.ti = rocke_b_mod(b, r, k);
        V k2 = ci(e, tiles_h(s));
        r = rocke_b_div(b, r, k2);
    }
    else
        rc.ti = ci(e, 0);
    if(p.dW > 1)
    {
        V k = ci(e, p.dW);
        rc.bsub = rocke_b_mod(b, r, k);
        V k2 = ci(e, p.dW);
        r = rocke_b_div(b, r, k2);
    }
    else
        rc.bsub = ci(e, 0);
    if(p.dH > 1)
    {
        V k = ci(e, p.dH);
        rc.a = rocke_b_mod(b, r, k);
        V k2 = ci(e, p.dH);
        r = rocke_b_div(b, r, k2);
    }
    else
        rc.a = ci(e, 0);
    rc.ok = ok;
    rc.cv = cv;
    rc.n = r;
    rc.tj0 = tj0;
    return rc;
}

/* ------------------------------------------------------------ K1 */

/* filter_matrix_int(m): integer numerators of G and their denominator. */
int64_t filter_matrix_int(int m, std::vector<int64_t>& gi)
{
    const rocke_wino::ToomCook* tc = rocke_wino::toom_cook(m, 3);
    int64_t den = 1;
    for(int i = 0; i < tc->n * 3; ++i)
    {
        int64_t d = tc->g[i].den;
        int64_t a = den, bb = d;
        while(bb)
        {
            int64_t t = a % bb;
            a = bb;
            bb = t;
        }
        den = den * d / a;
    }
    gi.resize(tc->n * 3);
    for(int i = 0; i < tc->n * 3; ++i)
        gi[i] = tc->g[i].num * (den / tc->g[i].den);
    return den;
}

void emit_filter(E& e)
{
    rocke_ir_builder_t* b = e.b;
    const Spec* s = e.s;
    const rocke_conv_problem_t& p = s->problem;
    int vec = vec_cg(s);
    int cg = cpg(s);
    int cgv = cg / vec;
    long long items = (long long)p.K * cgv;
    int n = nh(s), nwv = nw(s);
    std::vector<int64_t> gint, gint_w;
    int64_t den = filter_matrix_int(mh(s), gint);
    int64_t den_w = filter_matrix_int(mw(s), gint_w);
    double scale = 1.0 / (double)(den * den_w);

    set_mwgs(e, kFilterBlock);
    V w = param(e, "w", gptr(e, rocke_wino::ir_dtype(io_dt(s))), NOALIAS | RO);
    V u = param(e, "u", gptr(e, rocke_wino::ir_dtype(op_dt(s))), NOALIAS | WO);
    V w_bytes = param(e, "w_bytes", rocke_i32(), 0);
    V u_bytes = param(e, "u_bytes", rocke_i32(), 0);
    V w_rsrc = rocke_b_buffer_rsrc(b, w, w_bytes);
    V u_rsrc = rocke_b_buffer_rsrc(b, u, u_bytes);

    V gid;
    {
        V bx = block_x(e);
        V k = ci(e, kFilterBlock);
        V mu = rocke_b_mul(b, bx, k);
        V tx = thread_x(e);
        gid = rocke_b_add(b, mu, tx);
    }
    V row_y = block_y(e);
    V ok;
    {
        V k = ci(e, items);
        ok = rocke_b_cmp_lt(b, gid, k);
    }
    V kk, c;
    {
        V k1 = ci(e, cgv);
        kk = rocke_b_div(b, gid, k1);
        V k2 = ci(e, cgv);
        V md = rocke_b_mod(b, gid, k2);
        V k3 = ci(e, vec);
        c = rocke_b_mul(b, md, k3);
    }
    V wbase;
    {
        V k1 = ci(e, 9 * cg);
        V mu = rocke_b_mul(b, kk, k1);
        V ad = rocke_b_add(b, mu, c);
        V k2 = ci(e, 2);
        V m2 = rocke_b_mul(b, ad, k2);
        wbase = oob_sel(e, ok, m2);
    }
    std::vector<std::vector<VV>> taps(3);
    for(int ky = 0; ky < 3; ++ky)
    {
        for(int kx = 0; kx < 3; ++kx)
        {
            V k = ci(e, (int64_t)(ky * 3 + kx) * cg * 2);
            V off = rocke_b_add(b, wbase, k);
            V raw = load_raw(e, w_rsrc, off, io_dt(s), vec);
            taps[ky].push_back(rocke_wino::raw_to_f32(b, raw, vec));
        }
    }
    V sc = cf(e, scale);
    int nr = s->filter_rows;
    for(int rr = 0; rr < nr; ++rr)
    {
        V row_i;
        VV coef;
        if(nr == n)
        {
            row_i = ci(e, rr);
            for(int ky = 0; ky < 3; ++ky)
                coef.push_back(cf(e, (double)gint[rr * 3 + ky]));
        }
        else
        {
            V k1 = ci(e, nr);
            V mu = rocke_b_mul(b, row_y, k1);
            V k2 = ci(e, rr);
            row_i = rocke_b_add(b, mu, k2);
            for(int ky = 0; ky < 3; ++ky)
            {
                V cval = cf(e, (double)gint[0 * 3 + ky]);
                for(int i = 1; i < n; ++i)
                {
                    V k = ci(e, i);
                    V eq = rocke_b_cmp_eq(b, row_i, k);
                    V cv2 = cf(e, (double)gint[i * 3 + ky]);
                    cval = rocke_b_select(b, eq, cv2, cval);
                }
                coef.push_back(cval);
            }
        }
        std::vector<VV> tmp(3);
        for(int kx = 0; kx < 3; ++kx)
        {
            for(int ch = 0; ch < vec; ++ch)
            {
                V f0 = rocke_b_fmul(b, coef[0], taps[0][kx][ch]);
                V f1 = rocke_b_fma(b, coef[1], taps[1][kx][ch], f0);
                tmp[kx].push_back(rocke_b_fma(b, coef[2], taps[2][kx][ch], f1));
            }
        }
        V ubase;
        {
            V k1 = ci(e, (int64_t)nwv * p.K);
            V m1 = rocke_b_mul(b, row_i, k1);
            V a1 = rocke_b_add(b, m1, kk);
            V k2 = ci(e, cg);
            V m2 = rocke_b_mul(b, a1, k2);
            V a2 = rocke_b_add(b, m2, c);
            V k3 = ci(e, 2);
            V m3 = rocke_b_mul(b, a2, k3);
            ubase = oob_sel(e, ok, m3);
        }
        for(int j = 0; j < nwv; ++j)
        {
            VV vals;
            for(int ch = 0; ch < vec; ++ch)
            {
                std::vector<std::pair<int64_t, V>> terms;
                for(int kx = 0; kx < 3; ++kx)
                    terms.push_back({gint_w[j * 3 + kx], tmp[kx][ch]});
                V lc = rocke_wino::lincomb(b, terms);
                vals.push_back(rocke_b_fmul(b, lc, sc));
            }
            V k = ci(e, (int64_t)j * p.K * cg * 2);
            V off = rocke_b_add(b, ubase, k);
            rocke_wino::store_vec_f32(b, u_rsrc, off, vals, op_dt(s));
        }
    }
}

/* ------------------------------------------------------------ K2 */

void input_tiles_transform(E& e,
                           const std::vector<VV>& raw,
                           int R,
                           const VV& tile_ok,
                           const VV& scales,
                           V vbase0,
                           V v_rsrc,
                           int vec)
{
    rocke_ir_builder_t* b = e.b;
    const Spec* s = e.s;
    const rocke_conv_problem_t& p = s->problem;
    int n = nh(s), m = mh(s), nwv = nw(s), mwv = mw(s);
    long long vstride = (long long)n2(s) * p.C * 2;
    std::vector<bool> have(raw.empty() ? 0 : raw[0].size(), false);
    std::vector<std::vector<VV>> colt(have.size()); /* [vv][ch][i] */
    std::vector<std::pair<V, V>> pending;
    rocke_b_sched_barrier(b, 0);
    auto col_transform = [&](int vv) -> const std::vector<VV>& {
        if(!have[vv])
        {
            std::vector<VV> data;
            for(int uu = 0; uu < n; ++uu)
                data.push_back(rocke_wino::raw_to_f32(b, raw[uu][vv], vec));
            for(int ch = 0; ch < vec; ++ch)
            {
                VV col;
                for(int uu = 0; uu < n; ++uu)
                    col.push_back(data[uu][ch]);
                colt[vv].push_back(bt_1d(e, m, col));
            }
            have[vv] = true;
        }
        return colt[vv];
    };
    for(int r = 0; r < R; ++r)
    {
        std::vector<const std::vector<VV>*> cols;
        for(int vv = 0; vv < nwv; ++vv)
            cols.push_back(&col_transform(mwv * r + vv));
        V vbase;
        {
            V k = ci(e, (int64_t)r * vstride);
            V ad = rocke_b_add(b, vbase0, k);
            vbase = oob_sel(e, tile_ok[r], ad);
        }
        for(int i = 0; i < n; ++i)
        {
            std::vector<VV> outs;
            for(int ch = 0; ch < vec; ++ch)
            {
                VV row;
                for(int vv = 0; vv < nwv; ++vv)
                    row.push_back((*cols[vv])[ch][i]);
                outs.push_back(bt_1d(e, mwv, row));
            }
            for(int j = 0; j < nwv; ++j)
            {
                VV vals;
                for(int ch = 0; ch < vec; ++ch)
                    vals.push_back(outs[ch][j]);
                if(scales[r] != nullptr)
                    for(auto& x : vals)
                        x = rocke_b_fmul(b, x, scales[r]);
                V k = ci(e, (int64_t)(i * nwv + j) * p.C * 2);
                V off = rocke_b_add(b, vbase, k);
                V pk = rocke_wino::pack_vec_f32(b, vals, op_dt(s));
                pending.push_back({off, pk});
            }
        }
    }
    rocke_b_sched_barrier(b, 0);
    int cpol = s->nt_stores ? kCpolNt : 0;
    for(auto& pr : pending)
        rocke_wino::store_packed(b, v_rsrc, pr.first, pr.second, op_dt(s), cpol);
}

void input_params(E& e, V* x_rsrc, V* v_rsrc, V* e_rsrc)
{
    rocke_ir_builder_t* b = e.b;
    const Spec* s = e.s;
    V x = param(e, "x", gptr(e, rocke_wino::ir_dtype(io_dt(s))), NOALIAS | RO);
    V v = param(e, "v", gptr(e, rocke_wino::ir_dtype(op_dt(s))), NOALIAS | WO);
    V ep = param(e, "e", gptr(e, rocke_i32()), NOALIAS);
    V xb = param(e, "x_bytes", rocke_i32(), 0);
    V vb = param(e, "v_bytes", rocke_i32(), 0);
    V eb = param(e, "e_bytes", rocke_i32(), 0);
    *x_rsrc = rocke_b_buffer_rsrc(b, x, xb);
    *v_rsrc = rocke_b_buffer_rsrc(b, v, vb);
    *e_rsrc = s->prescale ? rocke_b_buffer_rsrc(b, ep, eb) : nullptr;
}

bool emit_input_lds(E& e, std::string* err);

bool emit_input(E& e, std::string* err)
{
    const Spec* s = e.s;
    if(input_lds(s))
        return emit_input_lds(e, err);
    rocke_ir_builder_t* b = e.b;
    const rocke_conv_problem_t& p = s->problem;
    int n = nh(s), m = mh(s), nwv = nw(s), mwv = mw(s);
    int vec = vec_c(s);
    RunMap rm;
    if(!input_run_map(s, &rm, err))
        return false;
    int R = rm.tiles;
    int ncols = mwv * R + (nwv - mwv);
    long long T = num_tiles(s);
    bool ps = s->prescale;
    bool split = ps && prescale_split(s);

    set_mwgs(e, rm.block);
    V x_rsrc, v_rsrc, e_rsrc;
    input_params(e, &x_rsrc, &v_rsrc, &e_rsrc);

    RunCoords rc = run_coords(e, rm);
    V c, h0, w0, img_row0;
    {
        V k = ci(e, vec);
        c = rocke_b_mul(b, rc.cv, k);
        V k1 = ci(e, p.pH);
        V sb = rocke_b_sub(b, rc.a, k1);
        V k2 = ci(e, (int64_t)p.dH * m);
        V mu = rocke_b_mul(b, rc.ti, k2);
        h0 = rocke_b_add(b, sb, mu);
        V k3 = ci(e, p.pW);
        V sb2 = rocke_b_sub(b, rc.bsub, k3);
        V k4 = ci(e, (int64_t)p.dW * mwv);
        V mu2 = rocke_b_mul(b, rc.tj0, k4);
        w0 = rocke_b_add(b, sb2, mu2);
        V k5 = ci(e, p.Hi);
        img_row0 = rocke_b_mul(b, rc.n, k5);
    }
    VV row_ok, row_pix, col_ok, col_w;
    for(int uu = 0; uu < n; ++uu)
    {
        V k = ci(e, (int64_t)uu * p.dH);
        V hi = rocke_b_add(b, h0, k);
        V k2 = ci(e, p.Hi);
        V lt = rocke_b_cmp_ult(b, hi, k2);
        row_ok.push_back(rocke_b_land(b, rc.ok, lt));
        V ad = rocke_b_add(b, img_row0, hi);
        V k3 = ci(e, p.Wi);
        row_pix.push_back(rocke_b_mul(b, ad, k3));
    }
    for(int vv = 0; vv < ncols; ++vv)
    {
        V k = ci(e, (int64_t)vv * p.dW);
        V wi = rocke_b_add(b, w0, k);
        V k2 = ci(e, p.Wi);
        col_ok.push_back(rocke_b_cmp_ult(b, wi, k2));
        col_w.push_back(wi);
    }
    VV xrow, xcol;
    for(int uu = 0; uu < n; ++uu)
    {
        V l = rocke_b_land(b, row_ok[uu], rc.c_ok);
        V k = ci(e, p.C);
        V mu = rocke_b_mul(b, row_pix[uu], k);
        V ad = rocke_b_add(b, mu, c);
        V k2 = ci(e, 2);
        V m2 = rocke_b_mul(b, ad, k2);
        xrow.push_back(oob_sel(e, l, m2));
    }
    for(int vv = 0; vv < ncols; ++vv)
    {
        V k = ci(e, (int64_t)p.C * 2);
        V mu = rocke_b_mul(b, col_w[vv], k);
        xcol.push_back(oob_sel(e, col_ok[vv], mu));
    }
    std::vector<VV> raw(n);
    for(int uu = 0; uu < n; ++uu)
        for(int vv = 0; vv < ncols; ++vv)
        {
            V ad = rocke_b_add(b, xrow[uu], xcol[vv]);
            raw[uu].push_back(load_raw(e, x_rsrc, ad, io_dt(s), vec));
        }
    VV tile_ok;
    for(int r = 0; r < R; ++r)
    {
        if(tiles_w(s) % R)
        {
            V l1 = rocke_b_land(b, rc.ok, rc.c_ok);
            V k = ci(e, r);
            V ad = rocke_b_add(b, rc.tj0, k);
            V k2 = ci(e, tiles_w(s));
            V lt = rocke_b_cmp_lt(b, ad, k2);
            tile_ok.push_back(rocke_b_land(b, l1, lt));
        }
        else
            tile_ok.push_back(rocke_b_land(b, rc.ok, rc.c_ok));
    }
    VV scales(R, nullptr);
    if(ps)
    {
        VV tmax;
        if(split)
        {
            VV prow, pcol, colmax;
            for(int uu = 0; uu < n; ++uu)
            {
                V k = ci(e, T);
                V ad = rocke_b_add(b, row_pix[uu], k);
                V k2 = ci(e, 4);
                V mu = rocke_b_mul(b, ad, k2);
                prow.push_back(oob_sel(e, row_ok[uu], mu));
            }
            for(int vv = 0; vv < ncols; ++vv)
            {
                V k = ci(e, 4);
                V mu = rocke_b_mul(b, col_w[vv], k);
                pcol.push_back(oob_sel(e, col_ok[vv], mu));
            }
            for(int vv = 0; vv < ncols; ++vv)
            {
                V acc = nullptr;
                for(int uu = 0; uu < n; ++uu)
                {
                    V ad = rocke_b_add(b, prow[uu], pcol[vv]);
                    V z = ci(e, 0);
                    V pm = rocke_b_buffer_load(b, e_rsrc, ad, z, rocke_i32());
                    acc = acc == nullptr ? pm : rocke_b_smax(b, acc, pm);
                }
                colmax.push_back(acc);
            }
            for(int r = 0; r < R; ++r)
            {
                V acc = colmax[mwv * r];
                for(int vv = 1; vv < nwv; ++vv)
                    acc = rocke_b_smax(b, acc, colmax[mwv * r + vv]);
                tmax.push_back(acc);
            }
        }
        else
        {
            VV colmax;
            for(int vv = 0; vv < ncols; ++vv)
            {
                V acc = nullptr;
                for(int uu = 0; uu < n; ++uu)
                {
                    V a_ = abs_bits_vec(e, raw[uu][vv], vec);
                    acc = acc == nullptr ? a_ : max_bits(e, acc, a_, vec);
                }
                colmax.push_back(acc);
            }
            for(int r = 0; r < R; ++r)
            {
                V acc = colmax[mwv * r];
                for(int vv = 1; vv < nwv; ++vv)
                    acc = max_bits(e, acc, colmax[mwv * r + vv], vec);
                tmax.push_back(hmax_bits(e, acc, vec));
            }
            tmax = group_max_bits(e, tmax, rm.tpc, rm.block, "wino_amax");
        }
        V lane0;
        {
            V z = ci(e, 0);
            lane0 = rocke_b_cmp_eq(b, rc.cv, z);
        }
        for(int r = 0; r < R; ++r)
        {
            V f = bits_to_f32(e, tmax[r], io_dt(s));
            V e_r = prescale_exp(e, f);
            V k = ci(e, r);
            V t_r = rocke_b_add(b, rc.t0, k);
            V ok_e = rocke_b_land(b, lane0, tile_ok[r]);
            V k4 = ci(e, 4);
            V mu = rocke_b_mul(b, t_r, k4);
            V off = oob_sel(e, ok_e, mu);
            V z = ci(e, 0);
            V bc = rocke_b_bitcast(b, e_r, rocke_f32());
            rocke_b_buffer_store_f32(b, e_rsrc, off, z, bc);
            V z2 = ci(e, 0);
            V ng = rocke_b_sub(b, z2, e_r);
            scales[r] = pow2_f32(e, ng);
        }
    }
    V vbase0;
    {
        V k = ci(e, (int64_t)n2(s) * p.C * 2);
        V mu = rocke_b_mul(b, rc.t0, k);
        V k2 = ci(e, 2);
        V mu2 = rocke_b_mul(b, c, k2);
        vbase0 = rocke_b_add(b, mu, mu2);
    }
    input_tiles_transform(e, raw, R, tile_ok, scales, vbase0, v_rsrc, vec);
    return true;
}

bool emit_input_lds(E& e, std::string* err)
{
    rocke_ir_builder_t* b = e.b;
    const Spec* s = e.s;
    const rocke_conv_problem_t& p = s->problem;
    int n = nh(s), m = mh(s), nwv = nw(s), mwv = mw(s);
    int vec = vec_c(s);
    LdsMap lm;
    if(!lds_map(s, &lm, err))
        return false;
    long long T = num_tiles(s);
    bool ps = s->prescale;
    const rocke_type_t* dt = rocke_wino::ir_dtype(io_dt(s));

    set_mwgs(e, lm.block);
    V x_rsrc, v_rsrc, e_rsrc;
    input_params(e, &x_rsrc, &v_rsrc, &e_rsrc);

    V tid = thread_x(e);
    V bid = block_x(e);
    V cxi, r;
    if(lm.cx > 1)
    {
        V k = ci(e, lm.cx);
        cxi = rocke_b_mod(b, bid, k);
    }
    else
        cxi = ci(e, 0);
    if(lm.cx > 1)
    {
        V k = ci(e, lm.cx);
        r = rocke_b_div(b, bid, k);
    }
    else
        r = bid;
    V bwi, bhi, sub, bsub, a, img;
    {
        V k = ci(e, lm.nbw);
        bwi = rocke_b_mod(b, r, k);
        V k2 = ci(e, lm.nbw);
        r = rocke_b_div(b, r, k2);
        V k3 = ci(e, lm.nbh);
        bhi = rocke_b_mod(b, r, k3);
        V k4 = ci(e, lm.nbh);
        sub = rocke_b_div(b, r, k4);
    }
    if(p.dW > 1)
    {
        V k = ci(e, p.dW);
        bsub = rocke_b_mod(b, sub, k);
    }
    else
        bsub = ci(e, 0);
    if(p.dW > 1)
    {
        V k = ci(e, p.dW);
        r = rocke_b_div(b, sub, k);
    }
    else
        r = sub;
    if(p.dH > 1)
    {
        V k = ci(e, p.dH);
        a = rocke_b_mod(b, r, k);
    }
    else
        a = ci(e, 0);
    if(p.dH > 1)
    {
        V k = ci(e, p.dH);
        img = rocke_b_div(b, r, k);
    }
    else
        img = r;
    V ti0, tj0, h0, w0, img_row0, c0;
    {
        V k = ci(e, lm.bh);
        ti0 = rocke_b_mul(b, bhi, k);
        V k2 = ci(e, lm.bw);
        tj0 = rocke_b_mul(b, bwi, k2);
        V k3 = ci(e, p.pH);
        V sb = rocke_b_sub(b, a, k3);
        V k4 = ci(e, (int64_t)p.dH * m);
        V mu = rocke_b_mul(b, ti0, k4);
        h0 = rocke_b_add(b, sb, mu);
        V k5 = ci(e, p.pW);
        V sb2 = rocke_b_sub(b, bsub, k5);
        V k6 = ci(e, (int64_t)p.dW * mwv);
        V mu2 = rocke_b_mul(b, tj0, k6);
        w0 = rocke_b_add(b, sb2, mu2);
        V k7 = ci(e, p.Hi);
        img_row0 = rocke_b_mul(b, img, k7);
        V k8 = ci(e, lm.cc);
        c0 = rocke_b_mul(b, cxi, k8);
    }
    int ppx = lm.cc / lm.lv;
    int npx = lm.rh * lm.rw;
    int shape[1] = {lm.loads * lm.block * lm.lv};
    V lds = rocke_b_smem_alloc(b, dt, shape, 1, "wino_xin");
    for(int k = 0; k < lm.loads; ++k)
    {
        V kq = ci(e, (int64_t)k * lm.block);
        V q = rocke_b_add(b, tid, kq);
        V px, pc;
        if(ppx > 1)
        {
            V kk = ci(e, ppx);
            px = rocke_b_div(b, q, kk);
        }
        else
            px = q;
        if(ppx > 1)
        {
            V kk = ci(e, ppx);
            pc = rocke_b_mod(b, q, kk);
        }
        else
            pc = ci(e, 0);
        V k1 = ci(e, lm.rw);
        V rr = rocke_b_div(b, px, k1);
        V k2 = ci(e, lm.rw);
        V cc = rocke_b_mod(b, px, k2);
        V k3 = ci(e, p.dH);
        V m1 = rocke_b_mul(b, rr, k3);
        V hi = rocke_b_add(b, h0, m1);
        V k4 = ci(e, p.dW);
        V m2 = rocke_b_mul(b, cc, k4);
        V wi = rocke_b_add(b, w0, m2);
        V k5 = ci(e, lm.lv);
        V m3 = rocke_b_mul(b, pc, k5);
        V ch = rocke_b_add(b, c0, m3);
        V k6 = ci(e, p.Hi);
        V u1 = rocke_b_cmp_ult(b, hi, k6);
        V k7 = ci(e, p.Wi);
        V u2 = rocke_b_cmp_ult(b, wi, k7);
        V ok = rocke_b_land(b, u1, u2);
        if((long long)(k + 1) * lm.block > (long long)npx * ppx)
        {
            V kk = ci(e, npx);
            V lt = rocke_b_cmp_lt(b, px, kk);
            ok = rocke_b_land(b, ok, lt);
        }
        if(lm.cx * lm.cc > p.C)
        {
            V kk = ci(e, p.C);
            V lt = rocke_b_cmp_lt(b, ch, kk);
            ok = rocke_b_land(b, ok, lt);
        }
        V ad = rocke_b_add(b, img_row0, hi);
        V k8 = ci(e, p.Wi);
        V mu = rocke_b_mul(b, ad, k8);
        V pix = rocke_b_add(b, mu, wi);
        V k9 = ci(e, p.C);
        V mu2 = rocke_b_mul(b, pix, k9);
        V ad2 = rocke_b_add(b, mu2, ch);
        V k10 = ci(e, 2);
        V mu3 = rocke_b_mul(b, ad2, k10);
        V off = oob_sel(e, ok, mu3);
        V val = load_raw(e, x_rsrc, off, io_dt(s), lm.lv);
        V k11 = ci(e, lm.lv);
        V idx = rocke_b_mul(b, q, k11);
        rocke_b_smem_store_vN(b, lds, &idx, 1, val, lm.lv);
    }
    V pm_lds = nullptr;
    if(ps)
    {
        int nb = (int)cdiv(npx, lm.block);
        int sh2[1] = {nb * lm.block};
        pm_lds = rocke_b_smem_alloc(b, rocke_i32(), sh2, 1, "wino_pmax");
        for(int k = 0; k < nb; ++k)
        {
            V kq = ci(e, (int64_t)k * lm.block);
            V px = rocke_b_add(b, tid, kq);
            V k1 = ci(e, lm.rw);
            V rr = rocke_b_div(b, px, k1);
            V k2 = ci(e, lm.rw);
            V cc = rocke_b_mod(b, px, k2);
            V k3 = ci(e, p.dH);
            V m1 = rocke_b_mul(b, rr, k3);
            V hi = rocke_b_add(b, h0, m1);
            V k4 = ci(e, p.dW);
            V m2 = rocke_b_mul(b, cc, k4);
            V wi = rocke_b_add(b, w0, m2);
            V k6 = ci(e, p.Hi);
            V u1 = rocke_b_cmp_ult(b, hi, k6);
            V k7 = ci(e, p.Wi);
            V u2 = rocke_b_cmp_ult(b, wi, k7);
            V ok = rocke_b_land(b, u1, u2);
            if((long long)(k + 1) * lm.block > npx)
            {
                V kk = ci(e, npx);
                V lt = rocke_b_cmp_lt(b, px, kk);
                ok = rocke_b_land(b, ok, lt);
            }
            V ad = rocke_b_add(b, img_row0, hi);
            V k8 = ci(e, p.Wi);
            V mu = rocke_b_mul(b, ad, k8);
            V pix = rocke_b_add(b, mu, wi);
            V kT = ci(e, T);
            V ad2 = rocke_b_add(b, pix, kT);
            V k10 = ci(e, 4);
            V mu3 = rocke_b_mul(b, ad2, k10);
            V off = oob_sel(e, ok, mu3);
            V z = ci(e, 0);
            V ld = rocke_b_buffer_load(b, e_rsrc, off, z, rocke_i32());
            rocke_b_smem_store_vN(b, pm_lds, &px, 1, ld, 1);
        }
    }
    rocke_b_sync(b);

    V cl, tb0, cv, c_ok;
    if(lm.tpc > 1)
    {
        V k = ci(e, lm.tpc);
        cl = rocke_b_mod(b, tid, k);
    }
    else
        cl = ci(e, 0);
    if(lm.tpc > 1)
    {
        V k = ci(e, lm.tpc);
        tb0 = rocke_b_div(b, tid, k);
    }
    else
        tb0 = tid;
    {
        V k = ci(e, lm.tpc);
        V mu = rocke_b_mul(b, cxi, k);
        cv = rocke_b_add(b, mu, cl);
        V k2 = ci(e, p.C / vec);
        c_ok = rocke_b_cmp_lt(b, cv, k2);
    }
    for(int rep = 0; rep < lm.tpt; ++rep)
    {
        V tb;
        if(rep)
        {
            V k = ci(e, (int64_t)rep * (lm.block / lm.tpc));
            tb = rocke_b_add(b, tb0, k);
        }
        else
            tb = tb0;
        V bi, bj;
        if(lm.bw > 1)
        {
            V k = ci(e, lm.bw);
            bi = rocke_b_div(b, tb, k);
        }
        else
            bi = tb;
        if(lm.bw > 1)
        {
            V k = ci(e, lm.bw);
            bj = rocke_b_mod(b, tb, k);
        }
        else
            bj = ci(e, 0);
        V ti = rocke_b_add(b, ti0, bi);
        V tj = rocke_b_add(b, tj0, bj);
        V tile_ok;
        {
            V k = ci(e, tiles_h(s));
            V l1 = rocke_b_cmp_lt(b, ti, k);
            V k2 = ci(e, tiles_w(s));
            V l2 = rocke_b_cmp_lt(b, tj, k2);
            V l3 = rocke_b_land(b, l1, l2);
            tile_ok = rocke_b_land(b, l3, c_ok);
        }
        V t;
        {
            V k = ci(e, tiles_h(s));
            V mu = rocke_b_mul(b, sub, k);
            V ad = rocke_b_add(b, mu, ti);
            V k2 = ci(e, tiles_w(s));
            V mu2 = rocke_b_mul(b, ad, k2);
            t = rocke_b_add(b, mu2, tj);
        }
        V pbase;
        {
            V k = ci(e, (int64_t)m * lm.rw);
            V mu = rocke_b_mul(b, bi, k);
            V k2 = ci(e, mwv);
            V mu2 = rocke_b_mul(b, bj, k2);
            pbase = rocke_b_add(b, mu, mu2);
        }
        std::vector<VV> raw(n);
        for(int uu = 0; uu < n; ++uu)
            for(int vv = 0; vv < nwv; ++vv)
            {
                V k = ci(e, (int64_t)uu * lm.rw + vv);
                V ad = rocke_b_add(b, pbase, k);
                V k2 = ci(e, lm.cc);
                V mu = rocke_b_mul(b, ad, k2);
                V k3 = ci(e, vec);
                V mu2 = rocke_b_mul(b, cl, k3);
                V idx = rocke_b_add(b, mu, mu2);
                V ld = rocke_b_smem_load_vN(b, lds, &idx, 1, dt, vec);
                raw[uu].push_back(vec == 1 ? rocke_b_vec_extract(b, ld, 0) : ld);
            }
        VV scales(1, nullptr);
        if(ps)
        {
            V acc = nullptr;
            for(int uu = 0; uu < n; ++uu)
                for(int vv = 0; vv < nwv; ++vv)
                {
                    V k = ci(e, (int64_t)uu * lm.rw + vv);
                    V ad = rocke_b_add(b, pbase, k);
                    V ld = rocke_b_smem_load_vN(b, pm_lds, &ad, 1, rocke_i32(), 1);
                    V pm = rocke_b_vec_extract(b, ld, 0);
                    acc = acc == nullptr ? pm : rocke_b_smax(b, acc, pm);
                }
            V f = bits_to_f32(e, acc, io_dt(s));
            V e_t = prescale_exp(e, f);
            V z = ci(e, 0);
            V eq = rocke_b_cmp_eq(b, cv, z);
            V ok_e = rocke_b_land(b, tile_ok, eq);
            V k4 = ci(e, 4);
            V mu = rocke_b_mul(b, t, k4);
            V off = oob_sel(e, ok_e, mu);
            V z2 = ci(e, 0);
            V bc = rocke_b_bitcast(b, e_t, rocke_f32());
            rocke_b_buffer_store_f32(b, e_rsrc, off, z2, bc);
            V z3 = ci(e, 0);
            V ng = rocke_b_sub(b, z3, e_t);
            scales[0] = pow2_f32(e, ng);
        }
        V vbase0;
        {
            V k = ci(e, (int64_t)n2(s) * p.C * 2);
            V mu = rocke_b_mul(b, t, k);
            V k2 = ci(e, vec);
            V mu2 = rocke_b_mul(b, cv, k2);
            V k3 = ci(e, 2);
            V mu3 = rocke_b_mul(b, mu2, k3);
            vbase0 = rocke_b_add(b, mu, mu3);
        }
        VV tok(1, tile_ok);
        input_tiles_transform(e, raw, 1, tok, scales, vbase0, v_rsrc, vec);
    }
    return true;
}

/* ------------------------------------------------------------ K2a */

void emit_prescale(E& e)
{
    rocke_ir_builder_t* b = e.b;
    const Spec* s = e.s;
    const rocke_conv_problem_t& p = s->problem;
    int vec = vec_c(s);
    int nv = p.C / vec;
    int tpc, loads;
    prescale_lanes(s, &tpc, &loads);
    long long npix = (long long)p.N * p.Hi * p.Wi;
    long long T = num_tiles(s);

    set_mwgs(e, kPrescaleBlock);
    V x = param(e, "x", gptr(e, rocke_wino::ir_dtype(io_dt(s))), NOALIAS | RO);
    V ep = param(e, "e", gptr(e, rocke_i32()), NOALIAS | WO);
    V xb = param(e, "x_bytes", rocke_i32(), 0);
    V eb = param(e, "e_bytes", rocke_i32(), 0);
    V x_rsrc = rocke_b_buffer_rsrc(b, x, xb);
    V e_rsrc = rocke_b_buffer_rsrc(b, ep, eb);
    V tid = thread_x(e);
    V lc, lp;
    if(tpc > 1)
    {
        V k = ci(e, tpc);
        lc = rocke_b_mod(b, tid, k);
    }
    else
        lc = ci(e, 0);
    if(tpc > 1)
    {
        V k = ci(e, tpc);
        lp = rocke_b_div(b, tid, k);
    }
    else
        lp = tid;
    V pix, pok, base;
    {
        V bx = block_x(e);
        V k = ci(e, kPrescaleBlock / tpc);
        V mu = rocke_b_mul(b, bx, k);
        pix = rocke_b_add(b, mu, lp);
        V k2 = ci(e, npix);
        pok = rocke_b_cmp_lt(b, pix, k2);
        V k3 = ci(e, p.C);
        V m1 = rocke_b_mul(b, pix, k3);
        V k4 = ci(e, vec);
        V m2 = rocke_b_mul(b, lc, k4);
        V ad = rocke_b_add(b, m1, m2);
        V k5 = ci(e, 2);
        base = rocke_b_mul(b, ad, k5);
    }
    V acc = nullptr;
    for(int q = 0; q < loads; ++q)
    {
        V ok = pok;
        if((q + 1) * tpc > nv)
        {
            V k = ci(e, q * tpc);
            V ad = rocke_b_add(b, lc, k);
            V k2 = ci(e, nv);
            V lt = rocke_b_cmp_lt(b, ad, k2);
            ok = rocke_b_land(b, ok, lt);
        }
        V k = ci(e, (int64_t)q * tpc * vec * 2);
        V ad = rocke_b_add(b, base, k);
        V off = oob_sel(e, ok, ad);
        V raw = load_raw(e, x_rsrc, off, io_dt(s), vec);
        V a_ = abs_bits_vec(e, raw, vec);
        acc = acc == nullptr ? a_ : max_bits(e, acc, a_, vec);
    }
    V hm = hmax_bits(e, acc, vec);
    V mx = wave_max_bits(e, hm, tpc);
    V z = ci(e, 0);
    V eq = rocke_b_cmp_eq(b, lc, z);
    V ok_s = rocke_b_land(b, pok, eq);
    V kT = ci(e, T);
    V ad = rocke_b_add(b, pix, kT);
    V k4 = ci(e, 4);
    V mu = rocke_b_mul(b, ad, k4);
    V off = oob_sel(e, ok_s, mu);
    V z2 = ci(e, 0);
    V bc = rocke_b_bitcast(b, mx, rocke_f32());
    rocke_b_buffer_store_f32(b, e_rsrc, off, z2, bc);
}

/* ------------------------------------------------------------ K4 */

bool emit_output(E& e, std::string* err)
{
    rocke_ir_builder_t* b = e.b;
    const Spec* s = e.s;
    const rocke_conv_problem_t& p = s->problem;
    int n = nh(s), m = mh(s), nwv = nw(s), mwv = mw(s);
    int vec = vec_k(s);
    RunMap rm;
    if(!output_run_map(s, &rm, err))
        return false;
    int ho_ = Ho(s), wo_ = Wo(s);

    set_mwgs(e, rm.block);
    V mm = param(e, "mm", gptr(e, rocke_wino::ir_dtype(op_dt(s))), NOALIAS | RO);
    V ep = param(e, "e", gptr(e, rocke_i32()), NOALIAS | RO);
    V y = param(e, "y", gptr(e, rocke_wino::ir_dtype(io_dt(s))), NOALIAS | WO);
    V mb = param(e, "m_bytes", rocke_i32(), 0);
    V yb = param(e, "y_bytes", rocke_i32(), 0);
    V m_rsrc = rocke_b_buffer_rsrc(b, mm, mb);
    V y_rsrc = rocke_b_buffer_rsrc(b, y, yb);

    RunCoords rc = run_coords(e, rm);
    V t = rc.t0;
    V ok = rocke_b_land(b, rc.ok, rc.c_ok);
    V k;
    {
        V kv = ci(e, vec);
        k = rocke_b_mul(b, rc.cv, kv);
    }
    V mbase;
    {
        V k1 = ci(e, (int64_t)n2(s) * p.K);
        V mu = rocke_b_mul(b, t, k1);
        V ad = rocke_b_add(b, mu, k);
        V k2 = ci(e, 2);
        V mu2 = rocke_b_mul(b, ad, k2);
        mbase = oob_sel(e, ok, mu2);
    }
    std::vector<VV> raw(n);
    for(int uu = 0; uu < n; ++uu)
        for(int vv = 0; vv < nwv; ++vv)
        {
            V kk = ci(e, (int64_t)(uu * nwv + vv) * p.K * 2);
            V ad = rocke_b_add(b, mbase, kk);
            raw[uu].push_back(load_raw(e, m_rsrc, ad, op_dt(s), vec));
        }
    V scale = nullptr;
    if(s->prescale)
    {
        V z = ci(e, 0);
        V t_safe = rocke_b_select(b, ok, t, z);
        V ld = rocke_b_global_load_i32(b, ep, t_safe, 4);
        scale = pow2_f32(e, ld);
    }
    V ho0, wo0, img_row0;
    {
        V k1 = ci(e, (int64_t)p.dH * m);
        V mu = rocke_b_mul(b, rc.ti, k1);
        ho0 = rocke_b_add(b, rc.a, mu);
        V k2 = ci(e, (int64_t)p.dW * mwv);
        V mu2 = rocke_b_mul(b, rc.tj0, k2);
        wo0 = rocke_b_add(b, rc.bsub, mu2);
        V k3 = ci(e, ho_);
        img_row0 = rocke_b_mul(b, rc.n, k3);
    }
    VV yrow, ycol;
    for(int i = 0; i < m; ++i)
    {
        V k1 = ci(e, (int64_t)i * p.dH);
        V ho = rocke_b_add(b, ho0, k1);
        V k2 = ci(e, ho_);
        V lt = rocke_b_cmp_lt(b, ho, k2);
        V okr = rocke_b_land(b, ok, lt);
        V ad = rocke_b_add(b, img_row0, ho);
        V k3 = ci(e, wo_);
        V mu = rocke_b_mul(b, ad, k3);
        V k4 = ci(e, p.K);
        V mu2 = rocke_b_mul(b, mu, k4);
        V ad2 = rocke_b_add(b, mu2, k);
        V k5 = ci(e, 2);
        V off = rocke_b_mul(b, ad2, k5);
        yrow.push_back(oob_sel(e, okr, off));
    }
    for(int j = 0; j < mwv; ++j)
    {
        V k1 = ci(e, (int64_t)j * p.dW);
        V wo = rocke_b_add(b, wo0, k1);
        V k2 = ci(e, wo_);
        V lt = rocke_b_cmp_lt(b, wo, k2);
        V k3 = ci(e, (int64_t)p.K * 2);
        V mu = rocke_b_mul(b, wo, k3);
        ycol.push_back(oob_sel(e, lt, mu));
    }
    rocke_b_sched_barrier(b, 0);
    std::vector<std::vector<VV>> colt; /* [vv][ch][i] */
    for(int vv = 0; vv < nwv; ++vv)
    {
        std::vector<VV> data;
        for(int uu = 0; uu < n; ++uu)
            data.push_back(rocke_wino::raw_to_f32(b, raw[uu][vv], vec));
        std::vector<VV> per;
        for(int ch = 0; ch < vec; ++ch)
        {
            VV col;
            for(int uu = 0; uu < n; ++uu)
                col.push_back(data[uu][ch]);
            per.push_back(at_1d(e, m, col));
        }
        colt.push_back(per);
    }
    std::vector<std::pair<V, V>> pending;
    for(int i = 0; i < m; ++i)
    {
        std::vector<VV> outs;
        for(int ch = 0; ch < vec; ++ch)
        {
            VV row;
            for(int vv = 0; vv < nwv; ++vv)
                row.push_back(colt[vv][ch][i]);
            outs.push_back(at_1d(e, mwv, row));
        }
        for(int j = 0; j < mwv; ++j)
        {
            VV vals;
            for(int ch = 0; ch < vec; ++ch)
                vals.push_back(outs[ch][j]);
            if(scale != nullptr)
                for(auto& x : vals)
                    x = rocke_b_fmul(b, x, scale);
            V off = rocke_b_add(b, yrow[i], ycol[j]);
            V pk = rocke_wino::pack_vec_f32(b, vals, io_dt(s));
            pending.push_back({off, pk});
        }
    }
    rocke_b_sched_barrier(b, 0);
    int cpol = s->nt_stores ? kCpolNt : 0;
    for(auto& pr : pending)
        rocke_wino::store_packed(b, y_rsrc, pr.first, pr.second, io_dt(s), cpol);
    return true;
}

bool emit_stage(E& e, rocke_winograd_fwd_stage_t st, std::string* err)
{
    switch(st)
    {
    case ROCKE_WINOGRAD_FWD_K1:
        emit_filter(e);
        return true;
    case ROCKE_WINOGRAD_FWD_K2A:
        emit_prescale(e);
        return true;
    case ROCKE_WINOGRAD_FWD_K2:
        return emit_input(e, err);
    case ROCKE_WINOGRAD_FWD_K4:
        return emit_output(e, err);
    }
    *err = "unknown stage";
    return false;
}

const char* stage_tag(rocke_winograd_fwd_stage_t st)
{
    switch(st)
    {
    case ROCKE_WINOGRAD_FWD_K1:
        return "filter";
    case ROCKE_WINOGRAD_FWD_K2A:
        return "pmax";
    case ROCKE_WINOGRAD_FWD_K2:
        return "input";
    case ROCKE_WINOGRAD_FWD_K4:
        return "output";
    }
    return "";
}

/* ------------------------------------------------------------ merged */

struct Part
{
    rocke_winograd_fwd_stage_t st;
    int grid[3];
    int block;
    int start, nflat, pack;
};

bool merge_layout(const Spec* s,
                  const rocke_winograd_fwd_stage_t* names,
                  int n_names,
                  std::vector<Part>* parts,
                  int* block_out,
                  std::string* err)
{
    /* transform_stages(spec): every present stage's launch is computed. */
    const rocke_winograd_fwd_stage_t order[4] = {ROCKE_WINOGRAD_FWD_K1,
                                                 ROCKE_WINOGRAD_FWD_K2A,
                                                 ROCKE_WINOGRAD_FWD_K2,
                                                 ROCKE_WINOGRAD_FWD_K4};
    int grids[4][3] = {};
    int blocks[4] = {};
    bool present[4] = {};
    for(int i = 0; i < 4; ++i)
    {
        present[i] = rocke_winograd_fwd_stage_present(s, order[i]);
        if(present[i] && !stage_launch(s, order[i], grids[i], &blocks[i], err))
            return false;
    }
    std::string missing;
    for(int i = 0; i < n_names; ++i)
    {
        if(names[i] < 0 || names[i] > 3 || !present[names[i]])
            missing += std::string(missing.empty() ? "" : ", ") + "'"
                       + rocke_winograd_fwd_stage_name(names[i]) + "'";
    }
    if(!missing.empty())
    {
        *err = "no stage(s) [" + missing + "] in this pipeline";
        return false;
    }
    std::vector<int> fixed;
    for(int i = 0; i < n_names; ++i)
        if(names[i] != ROCKE_WINOGRAD_FWD_K1
           && std::find(fixed.begin(), fixed.end(), blocks[names[i]]) == fixed.end())
            fixed.push_back(blocks[names[i]]);
    if(fixed.size() > 1)
    {
        *err = "stages with barriers need one block size";
        return false;
    }
    int block = 0;
    if(!fixed.empty())
        block = fixed[0];
    else
        for(int i = 0; i < n_names; ++i)
            block = std::max(block, blocks[names[i]]);
    int start = 0;
    parts->clear();
    for(int i = 0; i < n_names; ++i)
    {
        Part pt;
        pt.st = names[i];
        memcpy(pt.grid, grids[names[i]], sizeof pt.grid);
        pt.block = blocks[names[i]];
        long long sub = (long long)pt.grid[0] * pt.grid[1] * pt.grid[2];
        int pack;
        long long nflat;
        if(pt.block == block)
        {
            pack = 1;
            nflat = sub;
        }
        else if(pt.st != ROCKE_WINOGRAD_FWD_K1)
        {
            *err = "block mismatch in a merged stage";
            return false;
        }
        else if(pt.block < block)
        {
            if(block % pt.block)
            {
                *err = "block does not divide the merged block";
                return false;
            }
            pack = block / pt.block;
            nflat = cdiv(sub, pack);
        }
        else
        {
            if(pt.block % block)
            {
                *err = "merged block does not divide the stage block";
                return false;
            }
            pack = -(pt.block / block);
            nflat = sub * -pack;
        }
        pt.start = start;
        pt.nflat = (int)nflat;
        pt.pack = pack;
        start += (int)nflat;
        parts->push_back(pt);
    }
    *block_out = block;
    return true;
}

struct Merged
{
    rocke_ir_builder_t* b;
    const Spec* s;
    std::vector<Part> parts;
    int block;
    V flat, tid;
    std::string err;
    bool ok = true;
};

void merged_emit(Merged& M, int i)
{
    rocke_ir_builder_t* b = M.b;
    const Part& st = M.parts[i];
    E e{b, M.s, false, {nullptr, nullptr, nullptr}, nullptr};
    V local;
    {
        V k = rocke_b_const_i32(b, st.start);
        local = rocke_b_sub(b, M.flat, k);
    }
    V sub, stid;
    if(st.pack > 1)
    {
        V k = rocke_b_const_i32(b, st.pack);
        V mu = rocke_b_mul(b, local, k);
        V k2 = rocke_b_const_i32(b, st.block);
        V dv = rocke_b_div(b, M.tid, k2);
        sub = rocke_b_add(b, mu, dv);
        V k3 = rocke_b_const_i32(b, st.block);
        stid = rocke_b_mod(b, M.tid, k3);
    }
    else if(st.pack < 0)
    {
        V k = rocke_b_const_i32(b, -st.pack);
        sub = rocke_b_div(b, local, k);
        V k2 = rocke_b_const_i32(b, -st.pack);
        V md = rocke_b_mod(b, local, k2);
        V k3 = rocke_b_const_i32(b, M.block);
        V mu = rocke_b_mul(b, md, k3);
        stid = rocke_b_add(b, mu, M.tid);
    }
    else
    {
        sub = local;
        stid = M.tid;
    }
    int gx = st.grid[0], gy = st.grid[1], gz = st.grid[2];
    V bx, r, by, bz;
    if(gx > 1)
    {
        V k = rocke_b_const_i32(b, gx);
        bx = rocke_b_mod(b, sub, k);
    }
    else
        bx = rocke_b_const_i32(b, 0);
    if(gx > 1)
    {
        V k = rocke_b_const_i32(b, gx);
        r = rocke_b_div(b, sub, k);
    }
    else
        r = sub;
    if(gy > 1)
    {
        V k = rocke_b_const_i32(b, gy);
        by = rocke_b_mod(b, r, k);
    }
    else
        by = rocke_b_const_i32(b, 0);
    if(gy > 1)
    {
        V k = rocke_b_const_i32(b, gy);
        bz = rocke_b_div(b, r, k);
    }
    else
        bz = r;
    e.sub = true;
    e.bid[0] = bx;
    e.bid[1] = by;
    e.bid[2] = bz;
    e.tid = stid;
    long long total = (long long)gx * gy * gz;
    if(st.pack > 1 && (long long)st.nflat * st.pack != total)
    {
        V k = rocke_b_const_i32(b, total);
        V lt = rocke_b_cmp_lt(b, sub, k);
        rocke_if_t g = rocke_b_scf_if(b, lt);
        rocke_b_region_enter(b, g.then_region);
        if(!emit_stage(e, st.st, &M.err))
            M.ok = false;
        rocke_b_region_leave(b);
    }
    else if(!emit_stage(e, st.st, &M.err))
        M.ok = false;
}

void merged_chain(Merged& M, int i)
{
    rocke_ir_builder_t* b = M.b;
    if(i == (int)M.parts.size() - 1)
    {
        merged_emit(M, i);
        return;
    }
    const Part& st = M.parts[i];
    V k = rocke_b_const_i32(b, st.start + st.nflat);
    V lt = rocke_b_cmp_lt(b, M.flat, k);
    rocke_if_else_t ie = rocke_b_scf_if_else(b, lt);
    rocke_b_region_enter(b, ie.then_region);
    merged_emit(M, i);
    rocke_b_region_leave(b);
    rocke_b_region_enter(b, ie.else_region);
    merged_chain(M, i + 1);
    rocke_b_region_leave(b);
}

std::string merged_tag(const rocke_winograd_fwd_stage_t* names, int n_names)
{
    std::string tag = "merged_";
    for(int i = 0; i < n_names; ++i)
    {
        std::string nm = rocke_winograd_fwd_stage_name(names[i]);
        for(auto& ch : nm)
            ch = (char)tolower((unsigned char)ch);
        tag += (i ? "_" : "") + nm;
    }
    return tag;
}

} // namespace

/* ===================================================================== *
 *  Public API
 * ===================================================================== */

rocke_winograd_fwd_spec_t rocke_winograd_fwd_spec_default(void)
{
    rocke_winograd_fwd_spec_t s;
    memset(&s, 0, sizeof(s));
    s.problem.groups = 1;
    s.problem.sH = s.problem.sW = 1;
    s.problem.dH = s.problem.dW = 1;
    s.m = 4;
    s.io_dtype = "fp16";
    s.operand_dtype = "fp16";
    s.vec = 8;
    s.block_size = 256;
    s.prescale = false;
    s.tiles_per_thread = 1;
    s.prescale_mode = "fused";
    s.input_block_tiles[0] = 0;
    s.input_block_tiles[1] = 0;
    s.wg_tile_rows = 1;
    s.nt_stores = true;
    s.lds_lanes = 32;
    s.filter_rows = 1;
    s.filter_vec = 0;
    s.m_w = 0;
    s.name = "conv_winograd_fwd";
    return s;
}

bool rocke_winograd_fwd_stage_from_name(const char* name, rocke_winograd_fwd_stage_t* out)
{
    static const char* const names[4] = {"K1", "K2a", "K2", "K4"};
    for(int i = 0; i < 4; ++i)
        if(name && strcmp(name, names[i]) == 0)
        {
            *out = (rocke_winograd_fwd_stage_t)i;
            return true;
        }
    return false;
}

const char* rocke_winograd_fwd_stage_name(rocke_winograd_fwd_stage_t stage)
{
    switch(stage)
    {
    case ROCKE_WINOGRAD_FWD_K1:
        return "K1";
    case ROCKE_WINOGRAD_FWD_K2A:
        return "K2a";
    case ROCKE_WINOGRAD_FWD_K2:
        return "K2";
    case ROCKE_WINOGRAD_FWD_K4:
        return "K4";
    }
    return "?";
}

bool rocke_winograd_fwd_is_valid_spec(const rocke_winograd_fwd_spec_t* s,
                                      char* reason,
                                      size_t reason_cap)
{
    char msg[256];
    if(s == NULL)
    {
        set_reason(reason, reason_cap, "null spec");
        return false;
    }
    const rocke_conv_problem_t& p = s->problem;
    if(p.groups < 1)
    {
        set_reason(reason, reason_cap, "groups must be >= 1, got " + std::to_string(p.groups));
        return false;
    }
    if(p.C % p.groups || p.K % p.groups)
    {
        set_reason(reason, reason_cap, "C and K must be divisible by groups");
        return false;
    }
    auto tile_ok = [](int m) { return m == 2 || m == 4 || m == 6; };
    if(!tile_ok(s->m))
    {
        snprintf(msg, sizeof msg, "m must be one of (2, 4, 6) (got %d)", s->m);
        set_reason(reason, reason_cap, msg);
        return false;
    }
    if(s->m_w && !tile_ok(s->m_w))
    {
        set_reason(reason, reason_cap, "m_w must be 0 or one of (2, 4, 6)");
        return false;
    }
    if(p.is_3d)
    {
        set_reason(reason, reason_cap, "Winograd fwd supports 2-D convolutions only");
        return false;
    }
    if(p.Y != 3 || p.X != 3)
    {
        snprintf(msg, sizeof msg, "Winograd F(m,3) needs a 3x3 filter (got %dx%d)", p.Y, p.X);
        set_reason(reason, reason_cap, msg);
        return false;
    }
    if(p.sH != 1 || p.sW != 1)
    {
        set_reason(reason, reason_cap, "Winograd fwd needs stride 1");
        return false;
    }
    auto dt_ok = [](const char* d) { return d && (!strcmp(d, "fp16") || !strcmp(d, "bf16")); };
    if(!dt_ok(s->io_dtype))
    {
        set_reason(reason, reason_cap, "io_dtype must be one of ('fp16', 'bf16')");
        return false;
    }
    if(!dt_ok(s->operand_dtype))
    {
        set_reason(reason, reason_cap, "operand_dtype must be one of ('fp16', 'bf16')");
        return false;
    }
    if(!strcmp(s->operand_dtype, "bf16")
       && (mh(s) != 2 || mw(s) != 2 || strcmp(s->io_dtype, "bf16") != 0))
    {
        set_reason(
            reason, reason_cap, "bf16 operands are only accurate enough for F(2,3) with bf16 I/O");
        return false;
    }
    if(!(s->vec == 1 || s->vec == 2 || s->vec == 4 || s->vec == 8))
    {
        snprintf(msg, sizeof msg, "vec must be 1/2/4/8 (got %d)", s->vec);
        set_reason(reason, reason_cap, msg);
        return false;
    }
    if(s->block_size < 32 || (s->block_size & (s->block_size - 1)))
    {
        set_reason(reason, reason_cap, "block_size must be a power of two >= 32");
        return false;
    }
    if(s->block_size > 1024)
    {
        set_reason(reason, reason_cap, "block_size must be <= 1024");
        return false;
    }
    if(Ho(s) <= 0 || Wo(s) <= 0)
    {
        set_reason(reason, reason_cap, "empty output");
        return false;
    }
    if(!(s->filter_vec == 0 || s->filter_vec == 1 || s->filter_vec == 2 || s->filter_vec == 4
         || s->filter_vec == 8))
    {
        snprintf(msg, sizeof msg, "filter_vec must be 0/1/2/4/8 (got %d)", s->filter_vec);
        set_reason(reason, reason_cap, msg);
        return false;
    }
    if(s->filter_rows < 1 || nh(s) % s->filter_rows)
    {
        snprintf(msg, sizeof msg, "filter_rows must divide n = %d", nh(s));
        set_reason(reason, reason_cap, msg);
        return false;
    }
    if(s->tiles_per_thread < 1)
    {
        set_reason(reason, reason_cap, "tiles_per_thread must be >= 1");
        return false;
    }
    const char* pm = s->prescale_mode ? s->prescale_mode : "";
    if(strcmp(pm, "fused") != 0 && strcmp(pm, "split") != 0)
    {
        set_reason(reason, reason_cap, "prescale_mode must be 'fused' or 'split'");
        return false;
    }
    int bh = s->input_block_tiles[0], bw = s->input_block_tiles[1];
    if((bh > 0) != (bw > 0) || bh < 0 || bw < 0)
    {
        set_reason(reason, reason_cap, "input_block_tiles must be (0, 0) or two positive ints");
        return false;
    }
    long long bytes[7];
    buffer_bytes(s, bytes);
    for(int i = 0; i < 7; ++i)
        if(bytes[i] > kMaxBufferBytes)
        {
            snprintf(msg,
                     sizeof msg,
                     "%s needs %lld bytes; the transform kernels use 32-bit buffer offsets "
                     "(< 2 GiB)",
                     kBufNames[i],
                     bytes[i]);
            set_reason(reason, reason_cap, msg);
            return false;
        }
    set_reason(reason, reason_cap, "ok");
    return true;
}

bool rocke_winograd_fwd_stage_present(const rocke_winograd_fwd_spec_t* spec,
                                      rocke_winograd_fwd_stage_t stage)
{
    if(stage == ROCKE_WINOGRAD_FWD_K2A)
        return spec->prescale && prescale_split(spec);
    return stage >= ROCKE_WINOGRAD_FWD_K1 && stage <= ROCKE_WINOGRAD_FWD_K4;
}

bool rocke_winograd_fwd_stage_launch(const rocke_winograd_fwd_spec_t* spec,
                                     rocke_winograd_fwd_stage_t stage,
                                     int grid[3],
                                     int* block,
                                     char* reason,
                                     size_t reason_cap)
{
    std::string err;
    if(!rocke_winograd_fwd_is_valid_spec(spec, reason, reason_cap))
        return false;
    if(!stage_launch(spec, stage, grid, block, &err))
    {
        set_reason(reason, reason_cap, err);
        return false;
    }
    set_reason(reason, reason_cap, "ok");
    return true;
}

rocke_status_t rocke_winograd_fwd_kernel_name(const rocke_winograd_fwd_spec_t* spec,
                                              const char* tag,
                                              char* out,
                                              size_t out_cap)
{
    std::string sh = spec_short(spec);
    const char* parts[] = {tag, sh.c_str()};
    return rocke_kernel_name_join(
        spec->name ? spec->name : "conv_winograd_fwd", parts, 2, NULL, NULL, 0, out, out_cap, NULL);
}

rocke_kernel_def_t* rocke_build_winograd_fwd_stage(rocke_ir_builder_t* b,
                                                   const rocke_winograd_fwd_spec_t* spec,
                                                   rocke_winograd_fwd_stage_t stage,
                                                   const char* arch)
{
    (void)arch;
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        char reason[512];
        std::string err;
        if(!rocke_winograd_fwd_is_valid_spec(spec, reason, sizeof reason))
        {
            rocke_i_set_err_msg(b, ROCKE_ERR_VALUE, reason);
            return NULL;
        }
        E e{b, spec, false, {nullptr, nullptr, nullptr}, nullptr};
        if(!emit_stage(e, stage, &err))
        {
            rocke_i_set_err_msg(b, ROCKE_ERR_VALUE, err.c_str());
            return NULL;
        }
        if(!rocke_ir_builder_ok(b))
            return NULL;
        return rocke_ir_builder_kernel(b);
    });
}

rocke_kernel_def_t* rocke_build_winograd_fwd_stage_new(rocke_ir_builder_t* b,
                                                       const rocke_winograd_fwd_spec_t* spec,
                                                       rocke_winograd_fwd_stage_t stage,
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
            rocke_i_set_err_msg(b, ROCKE_ERR_VALUE, "build_winograd_fwd: null spec");
            return NULL;
        }
        if(rocke_winograd_fwd_kernel_name(spec, stage_tag(stage), name, sizeof name) != ROCKE_OK)
        {
            rocke_i_set_err_msg(b, ROCKE_ERR_VALUE, "build_winograd_fwd: kernel name too long");
            return NULL;
        }
        if(rocke_ir_builder_init(b, name) != ROCKE_OK)
            return NULL;
        return rocke_build_winograd_fwd_stage(b, spec, stage, arch);
    });
}

rocke_kernel_def_t* rocke_build_winograd_merged(rocke_ir_builder_t* b,
                                                const rocke_winograd_fwd_spec_t* spec,
                                                const rocke_winograd_fwd_stage_t* names,
                                                int n_names,
                                                const char* arch)
{
    (void)arch;
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        char reason[512];
        if(!rocke_winograd_fwd_is_valid_spec(spec, reason, sizeof reason))
        {
            rocke_i_set_err_msg(b, ROCKE_ERR_VALUE, reason);
            return NULL;
        }
        if(names == NULL || n_names < 1)
        {
            rocke_i_set_err_msg(b, ROCKE_ERR_VALUE, "build_winograd_merged: no stages");
            return NULL;
        }
        Merged M;
        M.b = b;
        M.s = spec;
        if(!merge_layout(spec, names, n_names, &M.parts, &M.block, &M.err))
        {
            rocke_i_set_err_msg(b, ROCKE_ERR_VALUE, M.err.c_str());
            return NULL;
        }
        M.flat = rocke_b_block_id_x(b);
        M.tid = rocke_b_thread_id_x(b);
        merged_chain(M, 0);
        if(!M.ok)
        {
            rocke_i_set_err_msg(b, ROCKE_ERR_VALUE, M.err.c_str());
            return NULL;
        }
        rocke_attr_set_int(b, &rocke_ir_builder_kernel(b)->attrs, "max_workgroup_size", M.block);
        if(!rocke_ir_builder_ok(b))
            return NULL;
        return rocke_ir_builder_kernel(b);
    });
}

rocke_kernel_def_t* rocke_build_winograd_merged_new(rocke_ir_builder_t* b,
                                                    const rocke_winograd_fwd_spec_t* spec,
                                                    const rocke_winograd_fwd_stage_t* names,
                                                    int n_names,
                                                    const char* arch)
{
    if(b != NULL)
        memset(b, 0, sizeof(*b));
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        char name[1024];
        if(b == NULL)
            return NULL;
        if(spec == NULL || names == NULL || n_names < 1)
        {
            rocke_i_set_err_msg(b, ROCKE_ERR_VALUE, "build_winograd_merged: null spec / names");
            return NULL;
        }
        std::string tag = merged_tag(names, n_names);
        if(rocke_winograd_fwd_kernel_name(spec, tag.c_str(), name, sizeof name) != ROCKE_OK)
        {
            rocke_i_set_err_msg(b, ROCKE_ERR_VALUE, "build_winograd_merged: kernel name too long");
            return NULL;
        }
        if(rocke_ir_builder_init(b, name) != ROCKE_OK)
            return NULL;
        return rocke_build_winograd_merged(b, spec, names, n_names, arch);
    });
}

rocke_status_t rocke_winograd_fwd_lower_to_llvm(const rocke_winograd_fwd_spec_t* spec,
                                                rocke_winograd_fwd_stage_t stage,
                                                const rocke_winograd_fwd_stage_t* names,
                                                int n_names,
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
    kernel = n_names > 0 ? rocke_build_winograd_merged_new(&b, spec, names, n_names, arch)
                         : rocke_build_winograd_fwd_stage_new(&b, spec, stage, arch);
    if(kernel == NULL)
    {
        const char* m = rocke_ir_builder_error(&b);
        st = rocke_ir_builder_status(&b);
        set_reason(err, err_cap, (m != NULL && m[0] != '\0') ? m : "build_winograd_fwd failed");
        rocke_ir_builder_free(&b);
        return (st == ROCKE_OK) ? ROCKE_ERR_VALUE : st;
    }
    st = rocke_lower_kernel_to_llvm_ex(kernel, flavor, arch, out_ll, err, err_cap);
    rocke_ir_builder_free(&b);
    return st;
}
