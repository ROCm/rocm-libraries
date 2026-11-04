// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * conv_winograd_fwd_rt.cpp -- C++ port of the shape-generic Winograd
 * F(mh x mw, 3 x 3) forward transform kernels
 * (library/kernels/common/conv_winograd_fwd_rt.py): K1 filter transform, K2a
 * per-pixel max, K2 input transform (register path with a fused or split
 * prescale, LDS-region path) and K4 output transform, every problem extent a
 * kernel argument. See the header and the Python module.
 *
 * Every emitter runs in PYTHON EVALUATION ORDER (arguments left to right,
 * innermost call first): nested builder calls are hoisted into ordered
 * locals, since C++ leaves the evaluation order of call arguments
 * unspecified.
 */
#include "rocke/instance_conv_winograd_fwd_rt.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <string>
#include <vector>

#include "rocke/error_boundary.hpp" /* ckc::guard_builder */
#include "rocke/instance_conv_winograd_internal.h"
#include "rocke/ir_internal.h" /* rocke_i_set_err_msg */
#include "rocke/lower_llvm.h"

namespace
{

using rocke_wino::kCpolNt;
using Spec = rocke_winograd_rt_spec_t;
using V = rocke_value_t*;
using VV = std::vector<rocke_value_t*>;

constexpr int kFilterBlock = 128;
constexpr int kPrescaleBlock = 256;
constexpr int kPrescaleMaxExp = 100;
constexpr int NOALIAS = 1, RO = 2, WO = 4;

void set_reason(char* reason, size_t cap, const std::string& msg)
{
    if(reason != NULL && cap > 0)
        snprintf(reason, cap, "%s", msg.c_str());
}

bool is_io(const char* d)
{
    return d != NULL && (strcmp(d, "fp16") == 0 || strcmp(d, "bf16") == 0);
}

bool is_tile(int m)
{
    return m == 2 || m == 4 || m == 6;
}

bool is_vec(int v)
{
    return v == 1 || v == 2 || v == 4 || v == 8;
}

/* _check_common(mh, mw, io, op, vec) */
bool check_common(const Spec* s, std::string* err)
{
    if(!is_tile(s->mh) || !is_tile(s->mw))
    {
        *err = "tile must be in (2, 4, 6) (got " + std::to_string(s->mh) + "x"
               + std::to_string(s->mw) + ")";
        return false;
    }
    if(!is_io(s->io_dtype) || !is_io(s->operand_dtype))
    {
        *err = "dtypes must be fp16 / bf16";
        return false;
    }
    if(strcmp(s->operand_dtype, "bf16") == 0
       && (s->mh != 2 || s->mw != 2 || strcmp(s->io_dtype, "bf16") != 0))
    {
        *err = "bf16 operands only for F(2,3) with bf16 I/O";
        return false;
    }
    if(!is_vec(s->vec))
    {
        *err = "vec must be 1/2/4/8 (got " + std::to_string(s->vec) + ")";
        return false;
    }
    return true;
}

bool check_block(const Spec* s, std::string* err)
{
    if(s->block_size < 32 || s->block_size > 1024 || (s->block_size & (s->block_size - 1)))
    {
        *err = "block_size must be a power of two in [32, 1024]";
        return false;
    }
    return true;
}

bool input_lds(const Spec* s)
{
    return s->lds[0] > 0;
}

int lds_block(const Spec* s)
{
    return s->lds[0] * s->lds[1] * s->lds_lanes / s->tiles_per_thread;
}

bool validate(const Spec* s, std::string* err)
{
    switch(s->stage)
    {
    case ROCKE_WINOGRAD_RT_K1:
        if(!check_common(s, err))
            return false;
        if(s->filter_rows < 1 || (s->mh + 2) % s->filter_rows)
        {
            *err = "filter_rows must divide n";
            return false;
        }
        return true;
    case ROCKE_WINOGRAD_RT_K2A:
        if(!is_io(s->io_dtype) || !is_vec(s->vec))
        {
            *err = "bad K2a spec";
            return false;
        }
        return true;
    case ROCKE_WINOGRAD_RT_K2:
        if(!check_common(s, err) || !check_block(s, err))
            return false;
        if(s->fused && (!s->prescale || input_lds(s)))
        {
            *err = "fused needs the prescale and the register path";
            return false;
        }
        if(s->tiles_per_thread < 1)
        {
            *err = "tiles_per_thread must be >= 1";
            return false;
        }
        if(input_lds(s))
        {
            if(s->lds[1] < 1)
            {
                *err = "lds must be (0, 0) or two positive extents";
                return false;
            }
            if(s->lds_lanes < 1 || (s->lds_lanes & (s->lds_lanes - 1)))
            {
                *err = "lds_lanes must be a power of two >= 1";
                return false;
            }
            if((s->lds[0] * s->lds[1]) % s->tiles_per_thread)
            {
                *err = "tiles_per_thread must divide bh * bw";
                return false;
            }
            if(lds_block(s) > 1024)
            {
                *err = "LDS region too large for one workgroup";
                return false;
            }
        }
        return true;
    case ROCKE_WINOGRAD_RT_K4:
        return check_common(s, err) && check_block(s, err);
    }
    *err = "unknown stage";
    return false;
}

std::string tile_tag(const Spec* s)
{
    if(s->mh == s->mw)
        return "f" + std::to_string(s->mh);
    return "f" + std::to_string(s->mh) + "x" + std::to_string(s->mw);
}

std::string join(const std::vector<std::string>& parts)
{
    std::string out;
    for(const std::string& p : parts)
    {
        if(p.empty())
            continue;
        if(!out.empty())
            out += "_";
        out += p;
    }
    return out;
}

std::string kernel_name(const Spec* s)
{
    std::string v = "v" + std::to_string(s->vec);
    std::string op = std::string("op") + s->operand_dtype;
    switch(s->stage)
    {
    case ROCKE_WINOGRAD_RT_K1:
        return join({"wino_rt_filter",
                     tile_tag(s),
                     s->io_dtype,
                     op,
                     v,
                     "fr" + std::to_string(s->filter_rows)});
    case ROCKE_WINOGRAD_RT_K2A:
        return join({"wino_rt_pmax", s->io_dtype, v});
    case ROCKE_WINOGRAD_RT_K2:
    {
        std::vector<std::string> tags = {"wino_rt_input", tile_tag(s), s->io_dtype, op, v};
        if(input_lds(s))
        {
            tags.push_back("lds" + std::to_string(s->lds[0]) + "x" + std::to_string(s->lds[1]));
            tags.push_back("ll" + std::to_string(s->lds_lanes));
            tags.push_back("t" + std::to_string(s->tiles_per_thread));
        }
        else
        {
            tags.push_back("b" + std::to_string(s->block_size));
            tags.push_back("r" + std::to_string(s->tiles_per_thread));
        }
        if(s->prescale)
            tags.push_back(s->fused ? "psf" : "ps");
        if(!s->nt_stores)
            tags.push_back("rt");
        return join(tags);
    }
    case ROCKE_WINOGRAD_RT_K4:
    {
        std::vector<std::string> tags = {
            "wino_rt_output", tile_tag(s), s->io_dtype, op, v, "b" + std::to_string(s->block_size)};
        if(s->prescale)
            tags.push_back("ps");
        if(!s->nt_stores)
            tags.push_back("rt");
        return join(tags);
    }
    }
    return "";
}

/* ------------------------------------------------------------ IR helpers */

struct E
{
    rocke_ir_builder_t* b;
    const Spec* s;
};

V ci(E& e, int64_t v)
{
    return rocke_b_const_i32(e.b, v);
}

/* b.param(name, t, noalias=..., readonly=..., writeonly=...) */
V param(E& e, const char* name, const rocke_type_t* t, int flags)
{
    if(flags == 0)
        return rocke_b_param(e.b, name, t, NULL);
    rocke_param_opts_t o;
    memset(&o, 0, sizeof o);
    if(flags & NOALIAS)
    {
        o.noalias = true;
        o.noalias_set = true;
    }
    if(flags & RO)
    {
        o.readonly = true;
        o.readonly_set = true;
    }
    if(flags & WO)
    {
        o.writeonly = true;
        o.writeonly_set = true;
    }
    return rocke_b_param(e.b, name, t, &o);
}

V p32(E& e, const char* name)
{
    return param(e, name, rocke_i32(), 0);
}

const rocke_type_t* gptr(E& e, const rocke_type_t* elem)
{
    return rocke_ptr_type(e.b, elem, "global");
}

void set_mwgs(E& e, int v)
{
    rocke_attr_set_int(e.b, &rocke_ir_builder_kernel(e.b)->attrs, "max_workgroup_size", v);
}

V oob(E& e, V ok, V off)
{
    return rocke_wino::oob_sel(e.b, ok, off);
}

/* _mul_c: v * k for a small constant k (0 and 1 folded). */
V mul_c(E& e, V v, int64_t k)
{
    if(k == 0)
        return ci(e, 0);
    if(k == 1)
        return v;
    V c = ci(e, k);
    return rocke_b_mul(e.b, v, c);
}

/* _Div: a runtime divisor (kernargs <name>, <name>_mg, <name>_sh). */
struct Div
{
    V d, mg, sh;
};

Div div_params(E& e, const char* name)
{
    Div dv;
    std::string n = name;
    dv.d = p32(e, n.c_str());
    dv.mg = p32(e, (n + "_mg").c_str());
    dv.sh = p32(e, (n + "_sh").c_str());
    return dv;
}

V div_q(E& e, const Div& dv, V n)
{
    rocke_ir_builder_t* b = e.b;
    V hi = rocke_b_umul_hi_i32(b, n, dv.mg);
    V q = rocke_b_lshr(b, hi, dv.sh);
    V z = ci(e, 0);
    V is1 = rocke_b_cmp_eq(b, dv.mg, z);
    return rocke_b_select(b, is1, n, q);
}

/* divmod(n) -> (n / d, n % d) */
void div_mod(E& e, const Div& dv, V n, V* q, V* r)
{
    *q = div_q(e, dv, n);
    V m = rocke_b_mul(e.b, *q, dv.d);
    *r = rocke_b_sub(e.b, n, m);
}

/* _prescale_exp_rt: clamp(ceil(log2(amax * ps_scale)), 0, 100). */
V prescale_exp_rt(E& e, V amax, V ps_scale)
{
    rocke_ir_builder_t* b = e.b;
    V sv = rocke_b_fmul(b, amax, ps_scale);
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

/* ------------------------------------------------------------ K1 */

void emit_filter(E& e)
{
    rocke_ir_builder_t* b = e.b;
    const Spec* s = e.s;
    int vec = s->vec;
    int n = s->mh + 2, nwv = s->mw + 2;
    std::vector<int64_t> gint, gint_w;
    int64_t den = rocke_wino::filter_matrix_int(s->mh, gint);
    int64_t den_w = rocke_wino::filter_matrix_int(s->mw, gint_w);
    double scale = 1.0 / (double)(den * den_w);

    set_mwgs(e, kFilterBlock);
    V w = param(e, "w", gptr(e, rocke_wino::ir_dtype(s->io_dtype)), NOALIAS | RO);
    V u = param(e, "u", gptr(e, rocke_wino::ir_dtype(s->operand_dtype)), NOALIAS | WO);
    V w_bytes = p32(e, "w_bytes");
    V u_bytes = p32(e, "u_bytes");
    V K = p32(e, "K");
    V cg = p32(e, "cg");
    V items = p32(e, "items");
    Div cgv = div_params(e, "cgv");
    V w_rsrc = rocke_b_buffer_rsrc(b, w, w_bytes);
    V u_rsrc = rocke_b_buffer_rsrc(b, u, u_bytes);

    V gid;
    {
        V bx = rocke_b_block_id_x(b);
        V k = ci(e, kFilterBlock);
        V mu = rocke_b_mul(b, bx, k);
        V tx = rocke_b_thread_id_x(b);
        gid = rocke_b_add(b, mu, tx);
    }
    V row_y = rocke_b_block_id_y(b);
    V ok = rocke_b_cmp_lt(b, gid, items);
    V kk, cm;
    div_mod(e, cgv, gid, &kk, &cm);
    V c = mul_c(e, cm, vec);
    V cg2;
    {
        V k2 = ci(e, 2);
        cg2 = rocke_b_mul(b, cg, k2);
    }
    V wbase;
    {
        V k9 = ci(e, 9);
        V cg9 = rocke_b_mul(b, cg, k9);
        V m1 = rocke_b_mul(b, kk, cg9);
        V a1 = rocke_b_add(b, m1, c);
        V k2 = ci(e, 2);
        V m2 = rocke_b_mul(b, a1, k2);
        wbase = oob(e, ok, m2);
    }
    std::vector<std::vector<VV>> taps(3);
    V off = wbase;
    for(int ky = 0; ky < 3; ++ky)
    {
        for(int kx = 0; kx < 3; ++kx)
        {
            if(ky || kx)
                off = rocke_b_add(b, off, cg2);
            V raw = rocke_wino::load_raw(b, w_rsrc, off, s->io_dtype, vec);
            taps[ky].push_back(rocke_wino::raw_to_f32(b, raw, vec));
        }
    }
    V sc = rocke_b_const_f32(b, scale);
    int nr = s->filter_rows;
    V kcg2 = rocke_b_mul(b, K, cg2);
    for(int rr = 0; rr < nr; ++rr)
    {
        V row_i;
        VV coef;
        if(nr == n)
        {
            row_i = ci(e, rr);
            for(int ky = 0; ky < 3; ++ky)
                coef.push_back(rocke_b_const_f32(b, (double)gint[rr * 3 + ky]));
        }
        else
        {
            V m1 = mul_c(e, row_y, nr);
            V k = ci(e, rr);
            row_i = rocke_b_add(b, m1, k);
            for(int ky = 0; ky < 3; ++ky)
            {
                V cval = rocke_b_const_f32(b, (double)gint[ky]);
                for(int i = 1; i < n; ++i)
                {
                    V ki = ci(e, i);
                    V eq = rocke_b_cmp_eq(b, row_i, ki);
                    V cv2 = rocke_b_const_f32(b, (double)gint[i * 3 + ky]);
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
            V rn = mul_c(e, row_i, nwv);
            V m1 = rocke_b_mul(b, rn, kcg2);
            V m2 = rocke_b_mul(b, kk, cg);
            V a2 = rocke_b_add(b, m2, c);
            V k2 = ci(e, 2);
            V m3 = rocke_b_mul(b, a2, k2);
            V a3 = rocke_b_add(b, m1, m3);
            ubase = oob(e, ok, a3);
        }
        V uoff = ubase;
        for(int j = 0; j < nwv; ++j)
        {
            if(j)
                uoff = rocke_b_add(b, uoff, kcg2);
            VV vals;
            for(int ch = 0; ch < vec; ++ch)
            {
                std::vector<std::pair<int64_t, V>> terms;
                for(int kx = 0; kx < 3; ++kx)
                    terms.push_back({gint_w[j * 3 + kx], tmp[kx][ch]});
                V lc = rocke_wino::lincomb(b, terms);
                vals.push_back(rocke_b_fmul(b, lc, sc));
            }
            rocke_wino::store_vec_f32(b, u_rsrc, uoff, vals, s->operand_dtype);
        }
    }
}

/* ------------------------------------------------------------ K2a */

void emit_prescale(E& e)
{
    rocke_ir_builder_t* b = e.b;
    const Spec* s = e.s;
    int vec = s->vec;
    set_mwgs(e, kPrescaleBlock);
    V x = param(e, "x", gptr(e, rocke_wino::ir_dtype(s->io_dtype)), NOALIAS | RO);
    V e_ptr = param(e, "e", gptr(e, rocke_i32()), NOALIAS | WO);
    V x_bytes = p32(e, "x_bytes");
    V e_bytes = p32(e, "e_bytes");
    V C = p32(e, "C");
    V nvec = p32(e, "nvec");
    V tpc_lg = p32(e, "tpc_lg");
    V loads = p32(e, "loads");
    V npix = p32(e, "npix");
    V T = p32(e, "T");
    V x_rsrc = rocke_b_buffer_rsrc(b, x, x_bytes);
    V e_rsrc = rocke_b_buffer_rsrc(b, e_ptr, e_bytes);

    V tid = rocke_b_thread_id_x(b);
    V tpc;
    {
        V c1 = ci(e, 1);
        tpc = rocke_b_shl(b, c1, tpc_lg);
    }
    V lc;
    {
        V c1 = ci(e, 1);
        V t1 = rocke_b_sub(b, tpc, c1);
        lc = rocke_b_land(b, tid, t1);
    }
    V lp = rocke_b_lshr(b, tid, tpc_lg);
    V ppw;
    {
        V cb = ci(e, kPrescaleBlock);
        ppw = rocke_b_lshr(b, cb, tpc_lg);
    }
    V pix;
    {
        V bx = rocke_b_block_id_x(b);
        V mu = rocke_b_mul(b, bx, ppw);
        pix = rocke_b_add(b, mu, lp);
    }
    V pok = rocke_b_cmp_lt(b, pix, npix);
    V base;
    {
        V m1 = rocke_b_mul(b, pix, C);
        V m2 = mul_c(e, lc, vec);
        V a1 = rocke_b_add(b, m1, m2);
        V k2 = ci(e, 2);
        base = rocke_b_mul(b, a1, k2);
    }
    V step_b;
    {
        V k = ci(e, vec * 2);
        step_b = rocke_b_mul(b, tpc, k);
    }
    V zero = ci(e, 0);
    V one = ci(e, 1);
    rocke_iter_arg_t ia[1];
    ia[0].name = "pm_acc";
    ia[0].init = zero;
    rocke_for_t loop = rocke_b_scf_for_iter(b, zero, loads, one, ia, 1, "pm_q", false, true);
    rocke_b_region_enter(b, loop.body);
    {
        V q = loop.iv;
        V acc = loop.iter_vars[0];
        V mq = rocke_b_mul(b, q, tpc);
        V idx = rocke_b_add(b, lc, mq);
        V lt = rocke_b_cmp_lt(b, idx, nvec);
        V ok = rocke_b_land(b, pok, lt);
        V ms = rocke_b_mul(b, q, step_b);
        V ad = rocke_b_add(b, base, ms);
        V off = oob(e, ok, ad);
        V raw = rocke_wino::load_raw(b, x_rsrc, off, s->io_dtype, vec);
        V ab = rocke_wino::abs_bits_vec(b, raw, vec);
        V a_ = rocke_wino::hmax_bits(b, ab, vec);
        V nx = rocke_b_smax(b, acc, a_);
        rocke_b_scf_yield(b, &nx, 1);
    }
    rocke_b_region_leave(b);
    V mx = loop.op->results[0];
    for(int sft = 1; sft < 32; sft *= 2)
    {
        V shv = rocke_b_warp_shuffle_xor(b, mx, sft);
        V ks = ci(e, sft);
        V lt = rocke_b_cmp_lt(b, ks, tpc);
        V m = rocke_b_smax(b, mx, shv);
        mx = rocke_b_select(b, lt, m, mx);
    }
    V ok_s;
    {
        V c0 = ci(e, 0);
        V eq = rocke_b_cmp_eq(b, lc, c0);
        ok_s = rocke_b_land(b, pok, eq);
    }
    V a1 = rocke_b_add(b, pix, T);
    V k4 = ci(e, 4);
    V m4 = rocke_b_mul(b, a1, k4);
    V off = oob(e, ok_s, m4);
    V z = ci(e, 0);
    V bc = rocke_b_bitcast(b, mx, rocke_f32());
    rocke_b_buffer_store_f32(b, e_rsrc, off, z, bc);
}

/* ------------------------------------------------------------ run coords */

struct Coords
{
    V ok, cv, c_ok, n, a, bsub, ti, tj0, t0, tiles_w, tpc;
};

/* _run_coords(b, tiles): the kernargs of the run map and the thread's run. */
Coords run_coords(E& e, int tiles)
{
    rocke_ir_builder_t* b = e.b;
    Coords rc;
    V nvec = p32(e, "nvec");
    V tpc_lg = p32(e, "tpc_lg");
    V rpw = p32(e, "rpw");
    V num_runs = p32(e, "num_runs");
    V tiles_w = p32(e, "tiles_w");
    Div cx = div_params(e, "cx");
    Div rpr = div_params(e, "rpr");
    Div th = div_params(e, "th");
    Div ddw = div_params(e, "ddw");
    Div ddh = div_params(e, "ddh");
    V tid = rocke_b_thread_id_x(b);
    V bid = rocke_b_block_id_x(b);
    V tpc;
    {
        V c1 = ci(e, 1);
        tpc = rocke_b_shl(b, c1, tpc_lg);
    }
    V lc;
    {
        V c1 = ci(e, 1);
        V t1 = rocke_b_sub(b, tpc, c1);
        lc = rocke_b_land(b, tid, t1);
    }
    V lr = rocke_b_lshr(b, tid, tpc_lg);
    V rb, cxi;
    div_mod(e, cx, bid, &rb, &cxi);
    {
        V mu = rocke_b_mul(b, cxi, tpc);
        rc.cv = rocke_b_add(b, mu, lc);
    }
    rc.c_ok = rocke_b_cmp_lt(b, rc.cv, nvec);
    V rho;
    {
        V mu = rocke_b_mul(b, rb, rpw);
        rho = rocke_b_add(b, mu, lr);
    }
    rc.ok = rocke_b_cmp_lt(b, rho, num_runs);
    V row, jr;
    div_mod(e, rpr, rho, &row, &jr);
    rc.tj0 = mul_c(e, jr, tiles);
    {
        V mu = rocke_b_mul(b, row, tiles_w);
        rc.t0 = rocke_b_add(b, mu, rc.tj0);
    }
    V r;
    div_mod(e, th, row, &r, &rc.ti);
    div_mod(e, ddw, r, &r, &rc.bsub);
    div_mod(e, ddh, r, &r, &rc.a);
    rc.n = r;
    rc.tiles_w = tiles_w;
    rc.tpc = tpc;
    return rc;
}

/* _group_max_rt: max of each i32 over the tpc (runtime) threads of a run. */
VV group_max_rt(E& e, const VV& vals, V tpc, int block)
{
    rocke_ir_builder_t* b = e.b;
    VV out;
    for(V v : vals)
    {
        for(int sft = 1; sft < 32; sft *= 2)
        {
            V shv = rocke_b_warp_shuffle_xor(b, v, sft);
            V ks = ci(e, sft);
            V lt = rocke_b_cmp_lt(b, ks, tpc);
            V m = rocke_b_smax(b, v, shv);
            v = rocke_b_select(b, lt, m, v);
        }
        out.push_back(v);
    }
    int rows = block / 32;
    if(rows <= 1)
        return out;
    int nv = (int)out.size();
    int shape[1] = {rows * nv};
    V red = rocke_b_smem_alloc_f32(b, shape, 1, "wino_rt_amax");
    V tid = rocke_b_thread_id_x(b);
    V row_id;
    {
        V c5 = ci(e, 5);
        row_id = rocke_b_lshr(b, tid, c5);
    }
    {
        V c31 = ci(e, 31);
        V la = rocke_b_land(b, tid, c31);
        V c0 = ci(e, 0);
        V cond = rocke_b_cmp_eq(b, la, c0);
        rocke_if_t g = rocke_b_scf_if(b, cond);
        rocke_b_region_enter(b, g.then_region);
        for(int i = 0; i < nv; ++i)
        {
            V m = mul_c(e, row_id, nv);
            V k = ci(e, i);
            V idx = rocke_b_add(b, m, k);
            V bc = rocke_b_bitcast(b, out[i], rocke_f32());
            V pk = rocke_b_vec_pack(b, &bc, 1, rocke_f32());
            rocke_b_smem_store_vN_f32(b, red, &idx, 1, pk, 1);
        }
        rocke_b_region_leave(b);
    }
    rocke_b_sync(b);
    V per;
    {
        V c5 = ci(e, 5);
        per = rocke_b_lshr(b, tpc, c5);
    }
    V big;
    {
        V c32 = ci(e, 32);
        big = rocke_b_cmp_lt(b, c32, tpc);
    }
    V first;
    {
        V c1 = ci(e, 1);
        V sp = rocke_b_select(b, big, per, c1);
        V dv = rocke_b_div(b, row_id, sp);
        V s1 = rocke_b_select(b, big, dv, row_id);
        V c1b = ci(e, 1);
        V sp2 = rocke_b_select(b, big, per, c1b);
        first = rocke_b_mul(b, s1, sp2);
    }
    VV res;
    for(int i = 0; i < nv; ++i)
    {
        V acc = out[i];
        for(int r = 0; r < rows; ++r)
        {
            V kr = ci(e, r);
            V fr = rocke_b_add(b, first, kr);
            V m = mul_c(e, fr, nv);
            V ki = ci(e, i);
            V idx = rocke_b_add(b, m, ki);
            V kr2 = ci(e, r);
            V lt = rocke_b_cmp_lt(b, kr2, per);
            V okr = rocke_b_land(b, big, lt);
            V z = ci(e, 0);
            idx = rocke_b_select(b, okr, idx, z);
            V ld = rocke_b_smem_load_vN_f32(b, red, &idx, 1, 1);
            V ex = rocke_b_vec_extract(b, ld, 0);
            V xv = rocke_b_bitcast(b, ex, rocke_i32());
            V mm = rocke_b_smax(b, acc, xv);
            acc = rocke_b_select(b, okr, mm, acc);
        }
        res.push_back(acc);
    }
    return res;
}

/* ------------------------------------------------------------ K2 */

const char* const kInputGeom[]
    = {"C", "Hi", "Wi", "pH", "pW", "dH", "dW", "dHm", "dWm", "T", "ps_bits"};
enum
{
    G_C,
    G_HI,
    G_WI,
    G_PH,
    G_PW,
    G_DH,
    G_DW,
    G_DHM,
    G_DWM,
    G_T,
    G_PS,
    G_N
};

/* _input_tiles_transform: V = B^T d B of R tiles adjacent along W. */
void input_tiles_transform(E& e,
                           const std::vector<VV>& raw,
                           int R,
                           const VV& tile_ok,
                           const VV& scales,
                           V vbase0,
                           V v_rsrc,
                           V cb2)
{
    rocke_ir_builder_t* b = e.b;
    const Spec* s = e.s;
    int n = s->mh + 2, m = s->mh;
    int nwv = s->mw + 2, mw = s->mw;
    int vec = s->vec;
    V vstride = mul_c(e, cb2, n * nwv);
    std::vector<std::vector<VV>> colt(raw.empty() ? 0 : raw[0].size());
    std::vector<bool> have(colt.size(), false);
    std::vector<std::pair<V, V>> pending;
    rocke_b_sched_barrier(b, 0);

    auto col_transform = [&](int vv) -> const std::vector<VV>& {
        if(!have[vv])
        {
            std::vector<VV> data;
            for(int uu = 0; uu < n; ++uu)
                data.push_back(rocke_wino::raw_to_f32(b, raw[uu][vv], vec));
            std::vector<VV> per_ch;
            for(int ch = 0; ch < vec; ++ch)
            {
                VV col;
                for(int uu = 0; uu < n; ++uu)
                    col.push_back(data[uu][ch]);
                per_ch.push_back(rocke_wino::bt_1d(b, m, col));
            }
            colt[vv] = per_ch;
            have[vv] = true;
        }
        return colt[vv];
    };

    for(int r = 0; r < R; ++r)
    {
        std::vector<std::vector<VV>> cols;
        for(int vv = 0; vv < nwv; ++vv)
            cols.push_back(col_transform(mw * r + vv));
        V base = vbase0;
        if(r != 0)
        {
            V ms = mul_c(e, vstride, r);
            base = rocke_b_add(b, vbase0, ms);
        }
        V off = oob(e, tile_ok[r], base);
        bool first = true;
        for(int i = 0; i < n; ++i)
        {
            std::vector<VV> outs;
            for(int ch = 0; ch < vec; ++ch)
            {
                VV row;
                for(int vv = 0; vv < nwv; ++vv)
                    row.push_back(cols[vv][ch][i]);
                outs.push_back(rocke_wino::bt_1d(b, mw, row));
            }
            for(int j = 0; j < nwv; ++j)
            {
                if(!first)
                    off = rocke_b_add(b, off, cb2);
                first = false;
                VV vals;
                for(int ch = 0; ch < vec; ++ch)
                    vals.push_back(outs[ch][j]);
                if(scales[r] != nullptr)
                {
                    for(auto& x_ : vals)
                        x_ = rocke_b_fmul(b, x_, scales[r]);
                }
                pending.push_back({off, rocke_wino::pack_vec_f32(b, vals, s->operand_dtype)});
            }
        }
    }
    rocke_b_sched_barrier(b, 0);
    int cpol = s->nt_stores ? kCpolNt : 0;
    for(auto& pr : pending)
        rocke_wino::store_packed(b, v_rsrc, pr.first, pr.second, s->operand_dtype, cpol);
}

/* K2's leading kernargs: the buffers, their sizes, the geometry. */
struct InputParams
{
    V x, v, e_ptr, x_bytes, v_bytes, e_bytes;
    V g[G_N];
};

InputParams input_params(E& e)
{
    const Spec* s = e.s;
    InputParams ip;
    ip.x = param(e, "x", gptr(e, rocke_wino::ir_dtype(s->io_dtype)), NOALIAS | RO);
    ip.v = param(e, "v", gptr(e, rocke_wino::ir_dtype(s->operand_dtype)), NOALIAS | WO);
    ip.e_ptr = param(e, "e", gptr(e, rocke_i32()), NOALIAS);
    ip.x_bytes = p32(e, "x_bytes");
    ip.v_bytes = p32(e, "v_bytes");
    ip.e_bytes = p32(e, "e_bytes");
    for(int i = 0; i < G_N; ++i)
        ip.g[i] = p32(e, kInputGeom[i]);
    return ip;
}

/* The prescale's tile store: e[t] (lane 0 of a tile) and the 2**-e scale. */
V store_e(E& e, V e_rsrc, V lane0, V tok, V t, V e_t)
{
    rocke_ir_builder_t* b = e.b;
    V la = rocke_b_land(b, lane0, tok);
    V k4 = ci(e, 4);
    V m4 = rocke_b_mul(b, t, k4);
    V off = oob(e, la, m4);
    V z = ci(e, 0);
    V bc = rocke_b_bitcast(b, e_t, rocke_f32());
    rocke_b_buffer_store_f32(b, e_rsrc, off, z, bc);
    V z2 = ci(e, 0);
    V ne = rocke_b_sub(b, z2, e_t);
    return rocke_wino::pow2_f32(b, ne);
}

bool emit_input_lds(E& e, std::string* err);

bool emit_input(E& e, std::string* err)
{
    const Spec* s = e.s;
    if(input_lds(s))
        return emit_input_lds(e, err);
    rocke_ir_builder_t* b = e.b;
    int n = s->mh + 2;
    int nwv = s->mw + 2, mw = s->mw;
    int vec = s->vec;
    int R = s->tiles_per_thread;
    int ncols = mw * R + (nwv - mw);
    bool ps = s->prescale;

    set_mwgs(e, s->block_size);
    InputParams ip = input_params(e);
    V* g = ip.g;
    V x_rsrc = rocke_b_buffer_rsrc(b, ip.x, ip.x_bytes);
    V v_rsrc = rocke_b_buffer_rsrc(b, ip.v, ip.v_bytes);
    V e_rsrc = ps ? rocke_b_buffer_rsrc(b, ip.e_ptr, ip.e_bytes) : nullptr;
    Coords rc = run_coords(e, R);

    V C = g[G_C], Hi = g[G_HI], Wi = g[G_WI];
    V cb2;
    {
        V k2 = ci(e, 2);
        cb2 = rocke_b_mul(b, C, k2);
    }
    V c = mul_c(e, rc.cv, vec);
    V h0;
    {
        V s1 = rocke_b_sub(b, rc.a, g[G_PH]);
        V m1 = rocke_b_mul(b, rc.ti, g[G_DHM]);
        h0 = rocke_b_add(b, s1, m1);
    }
    V w0;
    {
        V s1 = rocke_b_sub(b, rc.bsub, g[G_PW]);
        V m1 = rocke_b_mul(b, rc.tj0, g[G_DWM]);
        w0 = rocke_b_add(b, s1, m1);
    }
    V img_row0 = rocke_b_mul(b, rc.n, Hi);
    VV row_ok, row_pix;
    V hi = h0;
    for(int uu = 0; uu < n; ++uu)
    {
        if(uu)
            hi = rocke_b_add(b, hi, g[G_DH]);
        V lt = rocke_b_cmp_ult(b, hi, Hi);
        row_ok.push_back(rocke_b_land(b, rc.ok, lt));
        V a1 = rocke_b_add(b, img_row0, hi);
        row_pix.push_back(rocke_b_mul(b, a1, Wi));
    }
    VV col_ok, col_w;
    V wi = w0;
    for(int vv = 0; vv < ncols; ++vv)
    {
        if(vv)
            wi = rocke_b_add(b, wi, g[G_DW]);
        col_ok.push_back(rocke_b_cmp_ult(b, wi, Wi));
        col_w.push_back(wi);
    }
    V c2 = mul_c(e, c, 2);
    VV xrow, xcol;
    for(int uu = 0; uu < n; ++uu)
    {
        V la = rocke_b_land(b, row_ok[uu], rc.c_ok);
        V m1 = rocke_b_mul(b, row_pix[uu], cb2);
        V a1 = rocke_b_add(b, m1, c2);
        xrow.push_back(oob(e, la, a1));
    }
    for(int vv = 0; vv < ncols; ++vv)
    {
        V m1 = rocke_b_mul(b, col_w[vv], cb2);
        xcol.push_back(oob(e, col_ok[vv], m1));
    }
    std::vector<VV> raw(n);
    for(int uu = 0; uu < n; ++uu)
    {
        for(int vv = 0; vv < ncols; ++vv)
        {
            V ad = rocke_b_add(b, xrow[uu], xcol[vv]);
            raw[uu].push_back(rocke_wino::load_raw(b, x_rsrc, ad, s->io_dtype, vec));
        }
    }
    V base_ok = rocke_b_land(b, rc.ok, rc.c_ok);
    VV tile_ok;
    for(int r = 0; r < R; ++r)
    {
        if(r == 0)
        {
            tile_ok.push_back(base_ok);
            continue;
        }
        V kr = ci(e, r);
        V a1 = rocke_b_add(b, rc.tj0, kr);
        V lt = rocke_b_cmp_lt(b, a1, rc.tiles_w);
        tile_ok.push_back(rocke_b_land(b, base_ok, lt));
    }

    VV scales(R, nullptr);
    if(ps && s->fused)
    {
        VV colmax;
        for(int vv = 0; vv < ncols; ++vv)
        {
            V acc = nullptr;
            for(int uu = 0; uu < n; ++uu)
            {
                V a_ = rocke_wino::abs_bits_vec(b, raw[uu][vv], vec);
                acc = acc == nullptr ? a_ : rocke_wino::max_bits(b, acc, a_, vec);
            }
            colmax.push_back(acc);
        }
        VV tmax;
        for(int r = 0; r < R; ++r)
        {
            V acc = colmax[mw * r];
            for(int vv = 1; vv < nwv; ++vv)
                acc = rocke_wino::max_bits(b, acc, colmax[mw * r + vv], vec);
            tmax.push_back(rocke_wino::hmax_bits(b, acc, vec));
        }
        tmax = group_max_rt(e, tmax, rc.tpc, s->block_size);
        V ps_scale = rocke_b_bitcast(b, g[G_PS], rocke_f32());
        V lane0;
        {
            V c0 = ci(e, 0);
            lane0 = rocke_b_cmp_eq(b, rc.cv, c0);
        }
        for(int r = 0; r < R; ++r)
        {
            V f = rocke_wino::bits_to_f32(b, tmax[r], s->io_dtype);
            V e_r = prescale_exp_rt(e, f, ps_scale);
            V t_r = rc.t0;
            if(r)
            {
                V kr = ci(e, r);
                t_r = rocke_b_add(b, rc.t0, kr);
            }
            scales[r] = store_e(e, e_rsrc, lane0, tile_ok[r], t_r, e_r);
        }
    }
    else if(ps)
    {
        V T = g[G_T];
        VV prow, pcol;
        for(int uu = 0; uu < n; ++uu)
        {
            V a1 = rocke_b_add(b, row_pix[uu], T);
            V k4 = ci(e, 4);
            V m1 = rocke_b_mul(b, a1, k4);
            prow.push_back(oob(e, row_ok[uu], m1));
        }
        for(int vv = 0; vv < ncols; ++vv)
        {
            V k4 = ci(e, 4);
            V m1 = rocke_b_mul(b, col_w[vv], k4);
            pcol.push_back(oob(e, col_ok[vv], m1));
        }
        VV colmax;
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
        V ps_scale = rocke_b_bitcast(b, g[G_PS], rocke_f32());
        V lane0;
        {
            V c0 = ci(e, 0);
            lane0 = rocke_b_cmp_eq(b, rc.cv, c0);
        }
        for(int r = 0; r < R; ++r)
        {
            V acc = colmax[mw * r];
            for(int vv = 1; vv < nwv; ++vv)
                acc = rocke_b_smax(b, acc, colmax[mw * r + vv]);
            V f = rocke_wino::bits_to_f32(b, acc, s->io_dtype);
            V e_r = prescale_exp_rt(e, f, ps_scale);
            V t_r = rc.t0;
            if(r)
            {
                V kr = ci(e, r);
                t_r = rocke_b_add(b, rc.t0, kr);
            }
            scales[r] = store_e(e, e_rsrc, lane0, tile_ok[r], t_r, e_r);
        }
    }

    V vbase0;
    {
        V st = mul_c(e, cb2, n * nwv);
        V m1 = rocke_b_mul(b, rc.t0, st);
        vbase0 = rocke_b_add(b, m1, c2);
    }
    input_tiles_transform(e, raw, R, tile_ok, scales, vbase0, v_rsrc, cb2);
    (void)err;
    return true;
}

bool emit_input_lds(E& e, std::string* err)
{
    rocke_ir_builder_t* b = e.b;
    const Spec* s = e.s;
    int n = s->mh + 2, m = s->mh;
    int nwv = s->mw + 2, mw = s->mw;
    int vec = s->vec;
    int bh = s->lds[0], bw = s->lds[1];
    int tpc = s->lds_lanes;
    int cc = tpc * vec;
    int lv = vec;
    int rh = m * bh + n - m;
    int rw = mw * bw + nwv - mw;
    int block = lds_block(s);
    int tpt = s->tiles_per_thread;
    int ppx = cc / lv;
    int npx = rh * rw;
    int loads = (npx * ppx + block - 1) / block;
    bool ps = s->prescale;
    const rocke_type_t* dt = rocke_wino::ir_dtype(s->io_dtype);

    set_mwgs(e, block);
    InputParams ip = input_params(e);
    V* g = ip.g;
    V nvec = p32(e, "nvec");
    V tiles_h = p32(e, "tiles_h");
    V tiles_w = p32(e, "tiles_w");
    Div cx = div_params(e, "cx");
    Div nbw = div_params(e, "nbw");
    Div nbh = div_params(e, "nbh");
    Div ddw = div_params(e, "ddw");
    Div ddh = div_params(e, "ddh");
    V x_rsrc = rocke_b_buffer_rsrc(b, ip.x, ip.x_bytes);
    V v_rsrc = rocke_b_buffer_rsrc(b, ip.v, ip.v_bytes);
    V e_rsrc = ps ? rocke_b_buffer_rsrc(b, ip.e_ptr, ip.e_bytes) : nullptr;
    V C = g[G_C], Hi = g[G_HI], Wi = g[G_WI];
    V cb2;
    {
        V k2 = ci(e, 2);
        cb2 = rocke_b_mul(b, C, k2);
    }

    V tid = rocke_b_thread_id_x(b);
    V bid = rocke_b_block_id_x(b);
    V r, cxi, bwi, sub, bhi, bsub, img, a;
    div_mod(e, cx, bid, &r, &cxi);
    div_mod(e, nbw, r, &r, &bwi);
    div_mod(e, nbh, r, &sub, &bhi);
    div_mod(e, ddw, sub, &r, &bsub);
    div_mod(e, ddh, r, &img, &a);
    V ti0 = mul_c(e, bhi, bh);
    V tj0 = mul_c(e, bwi, bw);
    V h0;
    {
        V s1 = rocke_b_sub(b, a, g[G_PH]);
        V m1 = rocke_b_mul(b, ti0, g[G_DHM]);
        h0 = rocke_b_add(b, s1, m1);
    }
    V w0;
    {
        V s1 = rocke_b_sub(b, bsub, g[G_PW]);
        V m1 = rocke_b_mul(b, tj0, g[G_DWM]);
        w0 = rocke_b_add(b, s1, m1);
    }
    V img_row0 = rocke_b_mul(b, img, Hi);
    V c0 = mul_c(e, cxi, cc);

    int shape[1] = {loads * block * lv};
    V lds = rocke_b_smem_alloc(b, dt, shape, 1, "wino_xin");
    int ppx_lg = 0;
    while((1 << (ppx_lg + 1)) <= ppx)
        ++ppx_lg;
    for(int k = 0; k < loads; ++k)
    {
        V q = tid;
        if(k)
        {
            V kb = ci(e, (int64_t)k * block);
            q = rocke_b_add(b, tid, kb);
        }
        V px, pc;
        if(ppx > 1)
        {
            V kl = ci(e, ppx_lg);
            px = rocke_b_lshr(b, q, kl);
            V km = ci(e, ppx - 1);
            pc = rocke_b_land(b, q, km);
        }
        else
        {
            px = q;
            pc = ci(e, 0);
        }
        V krw = ci(e, rw);
        V rr = rocke_b_div(b, px, krw);
        V krw2 = ci(e, rw);
        V mr = rocke_b_mul(b, rr, krw2);
        V cc_ = rocke_b_sub(b, px, mr);
        V mh_ = rocke_b_mul(b, rr, g[G_DH]);
        V hi = rocke_b_add(b, h0, mh_);
        V mw_ = rocke_b_mul(b, cc_, g[G_DW]);
        V wi = rocke_b_add(b, w0, mw_);
        V mc = mul_c(e, pc, lv);
        V ch = rocke_b_add(b, c0, mc);
        V l1 = rocke_b_cmp_ult(b, hi, Hi);
        V l2 = rocke_b_cmp_ult(b, wi, Wi);
        V ok = rocke_b_land(b, l1, l2);
        if((k + 1) * block > npx * ppx)
        {
            V kn = ci(e, npx);
            V lt = rocke_b_cmp_lt(b, px, kn);
            ok = rocke_b_land(b, ok, lt);
        }
        {
            V lt = rocke_b_cmp_lt(b, ch, C);
            ok = rocke_b_land(b, ok, lt);
        }
        V a1 = rocke_b_add(b, img_row0, hi);
        V m1 = rocke_b_mul(b, a1, Wi);
        V pix = rocke_b_add(b, m1, wi);
        V m2 = rocke_b_mul(b, pix, cb2);
        V m3 = mul_c(e, ch, 2);
        V a2 = rocke_b_add(b, m2, m3);
        V off = oob(e, ok, a2);
        V val = rocke_wino::load_raw(b, x_rsrc, off, s->io_dtype, lv);
        V idx = mul_c(e, q, lv);
        rocke_b_smem_store_vN(b, lds, &idx, 1, val, lv);
    }
    V pm_lds = nullptr;
    if(ps)
    {
        V T = g[G_T];
        int npl = (npx + block - 1) / block;
        int sh2[1] = {npl * block};
        pm_lds = rocke_b_smem_alloc(b, rocke_i32(), sh2, 1, "wino_pmax");
        for(int k = 0; k < npl; ++k)
        {
            V px = tid;
            if(k)
            {
                V kb = ci(e, (int64_t)k * block);
                px = rocke_b_add(b, tid, kb);
            }
            V krw = ci(e, rw);
            V rr = rocke_b_div(b, px, krw);
            V krw2 = ci(e, rw);
            V mr = rocke_b_mul(b, rr, krw2);
            V cc_ = rocke_b_sub(b, px, mr);
            V mh_ = rocke_b_mul(b, rr, g[G_DH]);
            V hi = rocke_b_add(b, h0, mh_);
            V mw_ = rocke_b_mul(b, cc_, g[G_DW]);
            V wi = rocke_b_add(b, w0, mw_);
            V l1 = rocke_b_cmp_ult(b, hi, Hi);
            V l2 = rocke_b_cmp_ult(b, wi, Wi);
            V ok = rocke_b_land(b, l1, l2);
            if((k + 1) * block > npx)
            {
                V kn = ci(e, npx);
                V lt = rocke_b_cmp_lt(b, px, kn);
                ok = rocke_b_land(b, ok, lt);
            }
            V a1 = rocke_b_add(b, img_row0, hi);
            V m1 = rocke_b_mul(b, a1, Wi);
            V pix = rocke_b_add(b, m1, wi);
            V a2 = rocke_b_add(b, pix, T);
            V k4 = ci(e, 4);
            V m2 = rocke_b_mul(b, a2, k4);
            V off = oob(e, ok, m2);
            V z = ci(e, 0);
            V ld = rocke_b_buffer_load(b, e_rsrc, off, z, rocke_i32());
            rocke_b_smem_store_vN(b, pm_lds, &px, 1, ld, 1);
        }
    }
    rocke_b_sync(b);

    int tpc_lg = 0;
    while((1 << (tpc_lg + 1)) <= tpc)
        ++tpc_lg;
    V cl, tb0;
    if(tpc > 1)
    {
        V km = ci(e, tpc - 1);
        cl = rocke_b_land(b, tid, km);
        V kl = ci(e, tpc_lg);
        tb0 = rocke_b_lshr(b, tid, kl);
    }
    else
    {
        cl = ci(e, 0);
        tb0 = tid;
    }
    V cv;
    {
        V m1 = mul_c(e, cxi, tpc);
        cv = rocke_b_add(b, m1, cl);
    }
    V c_ok = rocke_b_cmp_lt(b, cv, nvec);
    V ps_scale = ps ? rocke_b_bitcast(b, g[G_PS], rocke_f32()) : nullptr;
    for(int rep = 0; rep < tpt; ++rep)
    {
        V tb = tb0;
        if(rep)
        {
            V k = ci(e, (int64_t)rep * (block / tpc));
            tb = rocke_b_add(b, tb0, k);
        }
        V bi, bj;
        if(bw > 1)
        {
            V kbw = ci(e, bw);
            bi = rocke_b_div(b, tb, kbw);
            V mb = mul_c(e, bi, bw);
            bj = rocke_b_sub(b, tb, mb);
        }
        else
        {
            bi = tb;
            bj = ci(e, 0);
        }
        V ti = rocke_b_add(b, ti0, bi);
        V tj = rocke_b_add(b, tj0, bj);
        V tile_ok;
        {
            V l1 = rocke_b_cmp_lt(b, ti, tiles_h);
            V l2 = rocke_b_cmp_lt(b, tj, tiles_w);
            V la = rocke_b_land(b, l1, l2);
            tile_ok = rocke_b_land(b, la, c_ok);
        }
        V t;
        {
            V m1 = rocke_b_mul(b, sub, tiles_h);
            V a1 = rocke_b_add(b, m1, ti);
            V m2 = rocke_b_mul(b, a1, tiles_w);
            t = rocke_b_add(b, m2, tj);
        }
        V pbase;
        {
            V m1 = mul_c(e, bi, m * rw);
            V m2 = mul_c(e, bj, mw);
            pbase = rocke_b_add(b, m1, m2);
        }
        std::vector<VV> raw(n);
        for(int uu = 0; uu < n; ++uu)
        {
            for(int vv = 0; vv < nwv; ++vv)
            {
                V k = ci(e, uu * rw + vv);
                V a1 = rocke_b_add(b, pbase, k);
                V m1 = mul_c(e, a1, cc);
                V m2 = mul_c(e, cl, vec);
                V idx = rocke_b_add(b, m1, m2);
                V ld = rocke_b_smem_load_vN(b, lds, &idx, 1, dt, vec);
                raw[uu].push_back(vec == 1 ? rocke_b_vec_extract(b, ld, 0) : ld);
            }
        }
        VV scales(1, nullptr);
        if(ps)
        {
            V acc = nullptr;
            for(int uu = 0; uu < n; ++uu)
            {
                for(int vv = 0; vv < nwv; ++vv)
                {
                    V k = ci(e, uu * rw + vv);
                    V ad = rocke_b_add(b, pbase, k);
                    V ld = rocke_b_smem_load_vN(b, pm_lds, &ad, 1, rocke_i32(), 1);
                    V pm = rocke_b_vec_extract(b, ld, 0);
                    acc = acc == nullptr ? pm : rocke_b_smax(b, acc, pm);
                }
            }
            V f = rocke_wino::bits_to_f32(b, acc, s->io_dtype);
            V e_t = prescale_exp_rt(e, f, ps_scale);
            V ok_e;
            {
                V c0b = ci(e, 0);
                V eq = rocke_b_cmp_eq(b, cv, c0b);
                ok_e = rocke_b_land(b, tile_ok, eq);
            }
            V k4 = ci(e, 4);
            V m4 = rocke_b_mul(b, t, k4);
            V off = oob(e, ok_e, m4);
            V z = ci(e, 0);
            V bc = rocke_b_bitcast(b, e_t, rocke_f32());
            rocke_b_buffer_store_f32(b, e_rsrc, off, z, bc);
            V z2 = ci(e, 0);
            V ne = rocke_b_sub(b, z2, e_t);
            scales[0] = rocke_wino::pow2_f32(b, ne);
        }
        V vbase0;
        {
            V st = mul_c(e, cb2, n * nwv);
            V m1 = rocke_b_mul(b, t, st);
            V m2 = mul_c(e, cv, vec * 2);
            vbase0 = rocke_b_add(b, m1, m2);
        }
        VV oks(1, tile_ok);
        input_tiles_transform(e, raw, 1, oks, scales, vbase0, v_rsrc, cb2);
    }
    (void)err;
    return true;
}

/* ------------------------------------------------------------ K4 */

const char* const kOutputGeom[] = {"K", "Ho", "Wo", "dH", "dW", "dHm", "dWm"};
enum
{
    O_K,
    O_HO,
    O_WO,
    O_DH,
    O_DW,
    O_DHM,
    O_DWM,
    O_N
};

void emit_output(E& e)
{
    rocke_ir_builder_t* b = e.b;
    const Spec* s = e.s;
    int n = s->mh + 2, m = s->mh;
    int nwv = s->mw + 2, mw = s->mw;
    int vec = s->vec;

    set_mwgs(e, s->block_size);
    V mm = param(e, "mm", gptr(e, rocke_wino::ir_dtype(s->operand_dtype)), NOALIAS | RO);
    V e_ptr = param(e, "e", gptr(e, rocke_i32()), NOALIAS | RO);
    V y = param(e, "y", gptr(e, rocke_wino::ir_dtype(s->io_dtype)), NOALIAS | WO);
    V m_bytes = p32(e, "m_bytes");
    V y_bytes = p32(e, "y_bytes");
    V g[O_N];
    for(int i = 0; i < O_N; ++i)
        g[i] = p32(e, kOutputGeom[i]);
    V m_rsrc = rocke_b_buffer_rsrc(b, mm, m_bytes);
    V y_rsrc = rocke_b_buffer_rsrc(b, y, y_bytes);
    Coords rc = run_coords(e, 1);

    V K = g[O_K], Ho = g[O_HO], Wo = g[O_WO];
    V kb2;
    {
        V k2 = ci(e, 2);
        kb2 = rocke_b_mul(b, K, k2);
    }
    V t = rc.t0;
    V ok = rocke_b_land(b, rc.ok, rc.c_ok);
    V k = mul_c(e, rc.cv, vec);
    V k2v = mul_c(e, k, 2);
    V mbase;
    {
        V st = mul_c(e, kb2, n * nwv);
        V m1 = rocke_b_mul(b, t, st);
        V a1 = rocke_b_add(b, m1, k2v);
        mbase = oob(e, ok, a1);
    }
    std::vector<VV> raw(n);
    V off = mbase;
    for(int uu = 0; uu < n; ++uu)
    {
        for(int vv = 0; vv < nwv; ++vv)
        {
            if(uu || vv)
                off = rocke_b_add(b, off, kb2);
            raw[uu].push_back(rocke_wino::load_raw(b, m_rsrc, off, s->operand_dtype, vec));
        }
    }
    V scale = nullptr;
    if(s->prescale)
    {
        V z = ci(e, 0);
        V t_safe = rocke_b_select(b, ok, t, z);
        V ld = rocke_b_global_load_i32(b, e_ptr, t_safe, 4);
        scale = rocke_wino::pow2_f32(b, ld);
    }
    V ho0;
    {
        V m1 = rocke_b_mul(b, rc.ti, g[O_DHM]);
        ho0 = rocke_b_add(b, rc.a, m1);
    }
    V wo0;
    {
        V m1 = rocke_b_mul(b, rc.tj0, g[O_DWM]);
        wo0 = rocke_b_add(b, rc.bsub, m1);
    }
    V img_row0 = rocke_b_mul(b, rc.n, Ho);
    VV yrow;
    V ho = ho0;
    for(int i = 0; i < m; ++i)
    {
        if(i)
            ho = rocke_b_add(b, ho, g[O_DH]);
        V lt = rocke_b_cmp_lt(b, ho, Ho);
        V okr = rocke_b_land(b, ok, lt);
        V a1 = rocke_b_add(b, img_row0, ho);
        V m1 = rocke_b_mul(b, a1, Wo);
        V m2 = rocke_b_mul(b, m1, kb2);
        V a2 = rocke_b_add(b, m2, k2v);
        yrow.push_back(oob(e, okr, a2));
    }
    VV ycol;
    V wo = wo0;
    for(int j = 0; j < mw; ++j)
    {
        if(j)
            wo = rocke_b_add(b, wo, g[O_DW]);
        V lt = rocke_b_cmp_lt(b, wo, Wo);
        V m1 = rocke_b_mul(b, wo, kb2);
        ycol.push_back(oob(e, lt, m1));
    }

    rocke_b_sched_barrier(b, 0);
    std::vector<std::vector<VV>> colt;
    for(int vv = 0; vv < nwv; ++vv)
    {
        std::vector<VV> data;
        for(int uu = 0; uu < n; ++uu)
            data.push_back(rocke_wino::raw_to_f32(b, raw[uu][vv], vec));
        std::vector<VV> per_ch;
        for(int ch = 0; ch < vec; ++ch)
        {
            VV col;
            for(int uu = 0; uu < n; ++uu)
                col.push_back(data[uu][ch]);
            per_ch.push_back(rocke_wino::at_1d(b, m, col));
        }
        colt.push_back(per_ch);
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
            outs.push_back(rocke_wino::at_1d(b, mw, row));
        }
        for(int j = 0; j < mw; ++j)
        {
            VV vals;
            for(int ch = 0; ch < vec; ++ch)
                vals.push_back(outs[ch][j]);
            if(scale != nullptr)
            {
                for(auto& x_ : vals)
                    x_ = rocke_b_fmul(b, x_, scale);
            }
            V ad = rocke_b_add(b, yrow[i], ycol[j]);
            pending.push_back({ad, rocke_wino::pack_vec_f32(b, vals, s->io_dtype)});
        }
    }
    rocke_b_sched_barrier(b, 0);
    int cpol = s->nt_stores ? kCpolNt : 0;
    for(auto& pr : pending)
        rocke_wino::store_packed(b, y_rsrc, pr.first, pr.second, s->io_dtype, cpol);
}

bool emit(E& e, std::string* err)
{
    switch(e.s->stage)
    {
    case ROCKE_WINOGRAD_RT_K1:
        emit_filter(e);
        return true;
    case ROCKE_WINOGRAD_RT_K2A:
        emit_prescale(e);
        return true;
    case ROCKE_WINOGRAD_RT_K2:
        return emit_input(e, err);
    case ROCKE_WINOGRAD_RT_K4:
        emit_output(e);
        return true;
    }
    *err = "unknown stage";
    return false;
}

} // namespace

extern "C" {

rocke_winograd_rt_spec_t rocke_winograd_rt_spec_default(rocke_winograd_rt_stage_t stage)
{
    rocke_winograd_rt_spec_t s;
    memset(&s, 0, sizeof s);
    s.stage = stage;
    s.mh = 4;
    s.mw = 4;
    s.io_dtype = "fp16";
    s.operand_dtype = "fp16";
    s.vec = 8;
    s.filter_rows = 1;
    s.block_size = 256;
    s.prescale = false;
    s.tiles_per_thread = 1;
    s.lds[0] = 0;
    s.lds[1] = 0;
    s.lds_lanes = 16;
    s.nt_stores = true;
    s.fused = false;
    return s;
}

bool rocke_winograd_rt_is_valid_spec(const rocke_winograd_rt_spec_t* spec,
                                     char* reason,
                                     size_t reason_cap)
{
    std::string err;
    if(spec == NULL)
    {
        set_reason(reason, reason_cap, "null spec");
        return false;
    }
    if(!validate(spec, &err))
    {
        set_reason(reason, reason_cap, err);
        return false;
    }
    set_reason(reason, reason_cap, "");
    return true;
}

rocke_status_t
    rocke_winograd_rt_kernel_name(const rocke_winograd_rt_spec_t* spec, char* out, size_t out_cap)
{
    if(spec == NULL || out == NULL || out_cap == 0)
        return ROCKE_ERR_VALUE;
    std::string err;
    if(!validate(spec, &err))
        return ROCKE_ERR_VALUE;
    std::string name = kernel_name(spec);
    if(name.size() + 1 > out_cap)
        return ROCKE_ERR_VALUE;
    memcpy(out, name.c_str(), name.size() + 1);
    return ROCKE_OK;
}

bool rocke_winograd_rt_magic_div(int64_t d, uint32_t* magic, int* shift)
{
    if(d < 1 || d >= ((int64_t)1 << 31) || magic == NULL || shift == NULL)
        return false;
    if(d == 1)
    {
        *magic = 0;
        *shift = 0;
        return true;
    }
    int lg = 0;
    while(((int64_t)1 << lg) < d)
        ++lg;
    /* ceil(2**(31 + lg) / d) < 2**32 for d >= 2 (see magic_div). */
    unsigned __int128 num = (unsigned __int128)1 << (31 + lg);
    unsigned __int128 q = (num + (unsigned __int128)(d - 1)) / (unsigned __int128)d;
    *magic = (uint32_t)q;
    *shift = lg - 1;
    return true;
}

rocke_kernel_def_t* rocke_build_winograd_rt(rocke_ir_builder_t* b,
                                            const rocke_winograd_rt_spec_t* spec,
                                            const char* arch)
{
    (void)arch;
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        std::string err;
        if(spec == NULL)
        {
            rocke_i_set_err_msg(b, ROCKE_ERR_VALUE, "build_winograd_rt: null spec");
            return NULL;
        }
        if(!validate(spec, &err))
        {
            rocke_i_set_err_msg(b, ROCKE_ERR_VALUE, err.c_str());
            return NULL;
        }
        E e{b, spec};
        if(!emit(e, &err))
        {
            rocke_i_set_err_msg(b, ROCKE_ERR_VALUE, err.c_str());
            return NULL;
        }
        if(rocke_ir_builder_status(b) != ROCKE_OK)
            return NULL;
        return rocke_ir_builder_kernel(b);
    });
}

rocke_kernel_def_t* rocke_build_winograd_rt_new(rocke_ir_builder_t* b,
                                                const rocke_winograd_rt_spec_t* spec,
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
            rocke_i_set_err_msg(b, ROCKE_ERR_VALUE, "build_winograd_rt: null spec");
            return NULL;
        }
        std::string err;
        if(!validate(spec, &err))
        {
            rocke_i_set_err_msg(b, ROCKE_ERR_VALUE, err.c_str());
            return NULL;
        }
        if(rocke_winograd_rt_kernel_name(spec, name, sizeof name) != ROCKE_OK)
        {
            rocke_i_set_err_msg(b, ROCKE_ERR_VALUE, "build_winograd_rt: kernel name too long");
            return NULL;
        }
        if(rocke_ir_builder_init(b, name) != ROCKE_OK)
            return NULL;
        return rocke_build_winograd_rt(b, spec, arch);
    });
}

rocke_status_t rocke_winograd_rt_lower_to_llvm(const rocke_winograd_rt_spec_t* spec,
                                               const char* arch,
                                               rocke_llvm_flavor_t flavor,
                                               char** out_ll,
                                               char* err,
                                               size_t err_cap)
{
    rocke_ir_builder_t b = {};
    if(out_ll != NULL)
        *out_ll = NULL;
    if(spec == NULL || out_ll == NULL)
    {
        set_reason(err, err_cap, "lower_to_llvm: null spec/out");
        return ROCKE_ERR_VALUE;
    }
    if(arch == NULL)
        arch = "gfx1250";
    rocke_kernel_def_t* kernel = rocke_build_winograd_rt_new(&b, spec, arch);
    if(kernel == NULL)
    {
        const char* m = rocke_ir_builder_error(&b);
        rocke_status_t st = rocke_ir_builder_status(&b);
        set_reason(err, err_cap, (m != NULL && m[0] != '\0') ? m : "build_winograd_rt failed");
        rocke_ir_builder_free(&b);
        return (st == ROCKE_OK) ? ROCKE_ERR_VALUE : st;
    }
    rocke_status_t st = rocke_lower_kernel_to_llvm_ex(kernel, flavor, arch, out_ll, err, err_cap);
    rocke_ir_builder_free(&b);
    return st;
}

} /* extern "C" */
