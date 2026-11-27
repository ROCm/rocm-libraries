// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * conv_direct_fwd_flex_mfma.cpp -- C++ engine port of
 * library/kernels/common/conv_direct_fwd_flex_mfma.py: the gfx950
 * flattened-pixel implicit-GEMM MFMA direct forward (build_fwd_flex_mfma):
 * the spec, its validation, the shared helpers and the main (cs 1) builder.
 * The cs 4 / cs 2 builders are in conv_direct_fwd_flex_mfma_cs4.cpp / _cs2.cpp.
 *
 * Every IR op (including each arith.constant) is emitted in the order the
 * Python builder emits it; see the contract in
 * rocke/instance_conv_direct_wgrad_internal.hpp.
 */
#include "rocke/instance_conv_direct_fwd_flex_mfma.h"

#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <algorithm>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include "rocke/arch_target.h" /* rocke_arch_target_from_gfx, rocke_mma_catalog_has_shape */
#include "rocke/helper_rocke.core.arch.h" /* rocke_archtarget_{async_lds_max_dwords,mma} */
#include "rocke/helper_rocke.helpers.spec.h" /* rocke_kernel_name_join, rocke_ceil_div */
#include "rocke/instance_conv_direct_fwd_flex_mfma_internal.hpp"
#include "rocke/instance_conv_direct_grouped_internal.h" /* rocke_dconv_emit_params */
#include "rocke/intexpr.h" /* rocke_intexpr_checked_mul */

using namespace rocke_dwg;

namespace rocke_ffx
{

/* ===================================================================== *
 *  Spec geometry
 * ===================================================================== */

std::pair<int, int> ds_strides(const rocke_fwd_flex_mfma_spec_t* s)
{
    const rocke_direct_conv_problem_t* p = &s->problem;
    return std::make_pair(s->h1 ? 1 : p->stride, p->stride);
}

std::vector<Phase> ds_phases_axis(const rocke_fwd_flex_mfma_spec_t* s, int axis)
{
    const rocke_direct_conv_problem_t* p = &s->problem;
    const std::pair<int, int> st2 = ds_strides(s);
    const int n = axis == 0 ? p->KH : p->KW;
    const int pad = axis == 0 ? p->PAD : rocke_direct_conv_problem_pad_w(p);
    const int st = axis == 0 ? st2.first : st2.second;
    std::vector<Phase> ax;
    for(int a = 0; a < st; ++a)
    {
        const int r0 = pymod(a + pad, st);
        const int t = pyfloordiv(n - r0 + st - 1, st);
        const int q = pyfloordiv(a + pad - r0, st);
        ax.push_back(Phase{t, t - 1 - q, r0 + st * (t - 1)});
    }
    return ax;
}

std::vector<std::pair<int, int>> ds_order(const rocke_fwd_flex_mfma_spec_t* s)
{
    const std::pair<int, int> st = ds_strides(s);
    const std::vector<Phase> ph = ds_phases_axis(s, 0), pw = ds_phases_axis(s, 1);
    std::vector<std::pair<int, int>> out;
    for(int a = 0; a < st.first; ++a)
    {
        for(int c = 0; c < st.second; ++c)
        {
            out.push_back(std::make_pair(a, c));
        }
    }
    std::stable_sort(out.begin(), out.end(), [&](const auto& x, const auto& y) {
        return -ph[x.first].t * pw[x.second].t < -ph[y.first].t * pw[y.second].t;
    });
    if(s->zf)
    {
        std::vector<std::pair<int, int>> keep;
        for(const auto& ac : out)
        {
            if(ph[ac.first].t * pw[ac.second].t > 0)
            {
                keep.push_back(ac);
            }
        }
        out = keep;
    }
    return out;
}

std::vector<std::pair<int, int>> ds_zero_deltas(const rocke_fwd_flex_mfma_spec_t* s)
{
    const std::pair<int, int> st = ds_strides(s);
    const std::vector<std::pair<int, int>> ord = ds_order(s);
    std::vector<std::pair<int, int>> out;
    if(ord.size() != 1)
    {
        return out;
    }
    const int a0 = ord[0].first, b0 = ord[0].second;
    for(int a = 0; a < st.first; ++a)
    {
        for(int c = 0; c < st.second; ++c)
        {
            if(a != a0 || c != b0)
            {
                out.push_back(std::make_pair(a - a0, c - b0));
            }
        }
    }
    return out;
}

std::pair<int, int> tap_grid(const rocke_fwd_flex_mfma_spec_t* s)
{
    const rocke_direct_conv_problem_t* p = &s->problem;
    if(s->ds)
    {
        int tr = 0, ts = 0;
        for(const Phase& f : ds_phases_axis(s, 0))
        {
            tr = std::max(tr, f.t);
        }
        for(const Phase& f : ds_phases_axis(s, 1))
        {
            ts = std::max(ts, f.t);
        }
        return std::make_pair(tr, ts);
    }
    return std::make_pair(p->KH, p->KW);
}

int tab_taps(const rocke_fwd_flex_mfma_spec_t* s)
{
    const std::pair<int, int> g = tap_grid(s);
    return cdiv(g.first * g.second, 8) * 8;
}

int nxi(const rocke_fwd_flex_mfma_spec_t* s)
{
    return s->cs == 4 ? 4 : s->bx / 64;
}

long long tab_bytes(const rocke_fwd_flex_mfma_spec_t* s)
{
    return (long long)tab_taps(s) * 8 * 8 * nxi(s) * 4;
}

int wt_half(const rocke_fwd_flex_mfma_spec_t* s)
{
    return s->bk == 192 ? 96 : 128;
}

long long w_elems(const rocke_fwd_flex_mfma_spec_t* s)
{
    if(s->wt)
    {
        return (long long)(s->bk / wt_half(s)) * kCK * 128;
    }
    return (long long)s->bk * kCK;
}

bool tab_db(const rocke_fwd_flex_mfma_spec_t* s, const char* arch)
{
    if(s->cs != 1 || s->ds || (s->wt && s->c_tail))
    {
        return false;
    }
    const rocke_arch_target_t* target = rocke_arch_target_from_gfx(arch ? arch : "gfx950");
    if(target == NULL)
    {
        return false;
    }
    const long long stage = ((long long)s->bx * kCK + w_elems(s)) * 2;
    return rocke_arch_fits_lds(target, (long)(2 * stage + 2 * tab_bytes(s)));
}

/* ===================================================================== *
 *  IR helpers
 * ===================================================================== */

std::vector<Div> divs(rocke_ir_builder_t* b, V lane, const std::vector<V>& ds)
{
    V c0 = rocke_b_const_i32(b, 0);
    V c1 = rocke_b_const_i32(b, 1);
    V d = c1;
    for(int i = (int)ds.size() - 1; i >= 0; --i)
    {
        V k = rocke_b_const_i32(b, i);
        V eq = rocke_b_cmp_eq(b, lane, k);
        d = rocke_b_select(b, eq, ds[i], d);
    }
    V dm1 = rocke_b_sub(b, d, c1);
    V fe = rocke_b_bitcast(b, rocke_b_sitofp_f32(b, dm1), rocke_i32());
    V k23 = rocke_b_const_i32(b, 23);
    V e = rocke_b_lshr(b, fe, k23);
    V k126 = rocke_b_const_i32(b, 126);
    V l = rocke_b_smax(b, rocke_b_sub(b, e, k126), c0);
    V a = rocke_b_sub(b, rocke_b_shl(b, c1, l), d);
    V rcp = rocke_b_rcp_fast(b, rocke_b_sitofp_f32(b, d));
    V k64k = rocke_b_const_f32(b, 65536.0);
    V rd = rocke_b_fmul(b, rcp, k64k);

    /* (floor(n 2^16 / d), remainder) for 0 <= n < d */
    auto half = [&](V n, V* rem) {
        V est = rocke_b_fmul(b, rocke_b_sitofp_f32(b, n), rd);
        V k23f = rocke_b_const_f32(b, 8388608.0);
        V qb = rocke_b_bitcast(b, rocke_b_fadd(b, est, k23f), rocke_i32());
        V kmag = rocke_b_const_i32(b, 0x4B000000);
        V q = rocke_b_sub(b, qb, kmag);
        V n64 = rocke_b_zext(b, n, rocke_i64());
        V k16 = rocke_b_const_i64(b, 16);
        V nsh = rocke_b_shl(b, n64, k16);
        V q64 = rocke_b_zext(b, q, rocke_i64());
        V d64 = rocke_b_zext(b, d, rocke_i64());
        V qd = rocke_b_mul(b, q64, d64);
        V r = rocke_b_trunc(b, rocke_b_sub(b, nsh, qd), rocke_i32());
        V neg = rocke_b_cmp_lt(b, r, c0);
        V q1 = rocke_b_sub(b, q, rocke_b_select(b, neg, c1, c0));
        *rem = rocke_b_add(b, r, rocke_b_select(b, neg, d, c0));
        return q1;
    };
    V r = NULL, r2 = NULL;
    V qh = half(a, &r);
    V ql = half(r, &r2);
    V k16 = rocke_b_const_i32(b, 16);
    V hi = rocke_b_shl(b, qh, k16);
    V m = rocke_b_add(b, hi, rocke_b_add(b, ql, c1));
    std::vector<Div> out;
    for(size_t i = 0; i < ds.size(); ++i)
    {
        V ki = rocke_b_const_i32(b, (int64_t)i);
        V mi = rocke_b_readlane(b, m, ki);
        V kl = rocke_b_const_i32(b, (int64_t)i);
        V li = rocke_b_readlane(b, l, kl);
        out.push_back(Div{ds[i], mi, li});
    }
    return out;
}

static void asm_void(rocke_ir_builder_t* b, const char* tmpl, const char* cons, V* ops, int n)
{
    rocke_inline_asm_opts_t opts = {};
    opts.sideeffect = true;
    opts.sideeffect_set = true;
    rocke_b_inline_asm(b, tmpl, cons, ops, n, NULL, 0, &opts);
}

void dma_lds(rocke_ir_builder_t* b, V rsrc, V dst, V voff, V soff)
{
    V so = rocke_b_readfirstlane(b, soff);
    V t = rocke_b_trunc(b, dst, rocke_i32());
    V ops[4] = {voff, rsrc, so, rocke_b_readfirstlane(b, t)};
    asm_void(b,
             "s_mov_b32 m0, $3\n\tbuffer_load_dwordx4 $0, $1, $2 offen lds",
             "v,s,s,s,~{m0},~{memory}",
             ops,
             4);
}

void dma_lds_at(rocke_ir_builder_t* b, V rsrc, V base, int imm, V voff, V soff)
{
    char tmpl[128];
    snprintf(
        tmpl, sizeof tmpl, "s_add_u32 m0, $3, %d\n\tbuffer_load_dwordx4 $0, $1, $2 offen lds", imm);
    V ops[4] = {voff, rsrc, rocke_b_readfirstlane(b, soff), base};
    asm_void(b, tmpl, "v,s,s,s,~{m0},~{scc},~{memory}", ops, 4);
}

V opaque(rocke_ir_builder_t* b, V v, const char* cons)
{
    rocke_inline_asm_opts_t opts = {};
    opts.sideeffect = true;
    opts.sideeffect_set = true;
    const rocke_type_t* i32_t = rocke_i32();
    rocke_op_t* op = rocke_b_inline_asm(b, "", cons, &v, 1, &i32_t, 1, &opts);
    return op ? op->results[0] : NULL;
}

} // namespace rocke_ffx

using namespace rocke_ffx;

namespace
{

/* a * b, saturated at INT64_MAX (Python's ints do not wrap). */
long long mul_sat(long long a, long long b)
{
    int64_t out;
    return rocke_intexpr_checked_mul(a, b, &out) ? out : INT64_MAX;
}

long long ceil_ll(long long a, long long b)
{
    return (a + b - 1) / b;
}

std::string fmt(const char* f, ...)
{
    char buf[ROCKE_ERR_MSG_CAP];
    va_list ap;
    va_start(ap, f);
    vsnprintf(buf, sizeof buf, f, ap);
    va_end(ap);
    return std::string(buf);
}

/* spec.workspace_bytes in saturating 64-bit arithmetic. */
long long ws_bytes(const rocke_fwd_flex_mfma_spec_t* s)
{
    const rocke_direct_conv_problem_t* p = &s->problem;
    if(s->split_c == 1)
    {
        return 0;
    }
    if(s->ds)
    {
        return mul_sat(mul_sat(mul_sat(mul_sat(4LL * s->split_c, p->N), p->H), p->W),
                       rocke_direct_conv_problem_total_c(p));
    }
    return mul_sat(
        mul_sat(mul_sat(mul_sat(4LL * s->split_c, p->N), rocke_direct_conv_problem_Ho(p)),
                rocke_direct_conv_problem_Wo(p)),
        rocke_direct_conv_problem_total_k(p));
}

const long long kI32Max = kOob;

/* fwd_flex_mfma_reason: the refusal text, or "" when the spec is valid. */
std::string flex_reason(const rocke_fwd_flex_mfma_spec_t* spec0, const char* arch)
{
    rocke_fwd_flex_mfma_spec_t spec = *spec0;
    const rocke_fwd_flex_mfma_spec_t* s = &spec;
    rocke_direct_conv_problem_t p = spec.problem;
    char msg[ROCKE_ERR_MSG_CAP];
    const rocke_arch_target_t* target = rocke_arch_target_from_gfx(arch);
    if(target == NULL)
    {
        return fmt("unknown arch %s", arch);
    }
    if(s->ds)
    {
        if(s->cs == 4)
        {
            return "ds: no cs 4 (its kernel has no dgrad phases)";
        }
        if(p.stride < 2 || p.dil_h != 1 || p.dil_w != 1 || p.groups != 1 || p.pad_w < -1)
        {
            return "ds: a dense strided dgrad without dilation";
        }
        if(s->h1 && (p.H != 1 || p.KH != 1 || p.PAD != 0))
        {
            return "h1: one dX row, a 1 x S filter and no row padding";
        }
        const std::vector<Phase> ph = ds_phases_axis(s, 0), pw = ds_phases_axis(s, 1);
        if(s->zf)
        {
            if(s->split_c != 1 || s->h1 || !s->wt)
            {
                return "zf: split_c 1 and wt only, no h1";
            }
            if(ds_order(s).size() != 1 || (long long)ds_zero_deltas(s).size() + kZfBits > 31)
            {
                return "zf: one output phase with taps, the others none";
            }
            if(s->zl && (s->zl < 0 || s->cs != 1 || s->xcd || s->k_tail))
            {
                return "zl: zf with cs 1, no xcd / k_tail";
            }
            if(mul_sat(mul_sat(p.N, p.H), p.W) >= (1LL << kZfBits))
            {
                return fmt("zf: dX pixels exceed 2^%d", kZfBits);
            }
            for(const auto* ax : {&ph, &pw})
            {
                for(const Phase& f : *ax)
                {
                    if(f.t > 0 && f.pa < -64)
                    {
                        return "ds: an input offset out of range";
                    }
                }
            }
        }
        else
        {
            for(const auto* ax : {&ph, &pw})
            {
                for(const Phase& f : *ax)
                {
                    if(f.t < 1 || f.pa < -64)
                    {
                        return "ds: an output phase without taps";
                    }
                }
            }
        }
        if(rocke_direct_conv_problem_Ho(&p) < 1 || rocke_direct_conv_problem_Wo(&p) < 1)
        {
            return "empty dY";
        }
        const long long tc = rocke_direct_conv_problem_total_c(&p);
        const long long tk = rocke_direct_conv_problem_total_k(&p);
        if(mul_sat(mul_sat(mul_sat(mul_sat(2, p.N), p.H), p.W), tc) > kI32Max)
        {
            return fmt("dX exceeds %lld B, the i32 buffer range", kI32Max);
        }
        if(mul_sat(mul_sat(mul_sat(mul_sat(2, tk), p.KH), p.KW), tc) > kI32Max)
        {
            return fmt("weights exceed %lld B, the i32 buffer range", kI32Max);
        }
        if(ws_bytes(s) > kI32Max)
        {
            return fmt("workspace exceeds %lld B, the i32 byte-count kernarg", kI32Max);
        }
        const std::pair<int, int> tg = tap_grid(s);
        if(tg.first * tg.second > 32)
        {
            return "ds: more than 32 taps in a phase";
        }
        const long long lds
            = 2 * ((long long)s->bx * kCK + w_elems(s)) * 2 + tab_bytes(s) + 3LL * s->bx * 4;
        if(!rocke_arch_fits_lds(target, (long)lds))
        {
            return fmt("LDS %lld B too large", lds);
        }
        /* the forward over dY the kernel runs (taps checked above) */
        rocke_direct_conv_problem_t q = rocke_direct_conv_problem_default();
        q.N = p.N;
        q.H = (int)rocke_direct_conv_problem_Ho(&p);
        q.W = (int)rocke_direct_conv_problem_Wo(&p);
        q.groups = 1;
        q.cpg = p.kpg;
        q.kpg = p.cpg;
        q.KH = 1;
        q.KW = 1;
        q.PAD = 0;
        q.stride = 1;
        q.dtype = p.dtype;
        p = q;
        spec.problem = q;
        spec.ds = false;
    }
    if(target->wave_size != 64 || s->wave_size != 64)
    {
        return "fwd_flex_mfma is a wave64 MFMA kernel";
    }
    if(rocke_archtarget_async_lds_max_dwords(target) < 4)
    {
        return fmt("fwd_flex_mfma needs the 16-byte LDS DMA (absent on %s)", arch);
    }
    if((s->bx != 128 && s->bx != 256) || (s->bx == 128 && s->bk == 64))
    {
        return "bx must be 256, or 128 with bk 128 / 256";
    }
    if(s->cs != 1 && s->cs != 2 && s->cs != 4)
    {
        return "cs must be 1, 2 or 4";
    }
    const long long tc = rocke_direct_conv_problem_total_c(&p);
    const long long tk = rocke_direct_conv_problem_total_k(&p);
    if(s->cs == 2)
    {
        if(s->bx != 128 || s->bk != 256 || s->c_tail || s->ds || s->flat)
        {
            return "cs 2: bx 128 and bk 256 only, no c_tail, ds or flat";
        }
        if((p.KH * p.KW) % 3)
        {
            return "cs 2: the tap count must be a multiple of 3 (a tap's stage is fixed)";
        }
        if(tc % kCK)
        {
            return fmt("cs 2: C must be a multiple of %d", kCK);
        }
    }
    if(s->cs == 4)
    {
        if(s->bx != 128 || s->bk != 128 || s->c_tail)
        {
            return "cs 4: bx 128 and bk 128 only, no c_tail";
        }
        if(tc % kCK || tc < 2LL * kCK * s->split_c)
        {
            return fmt("cs 4: C must be a multiple of %d and >= %lld (two chunks per piece)",
                       kCK,
                       2LL * kCK * s->split_c);
        }
    }
    if(s->wt && ((s->bk != 128 && s->bk != 192 && s->bk != 256) || (s->c_tail && s->cs == 4)))
    {
        return "wt: bk 128 / 192 / 256 only, no c_tail with cs 4";
    }
    const long long stage = ((long long)s->bx * kCK + w_elems(s)) * 2 * (s->cs == 4 ? 2 : 1);
    const long long lds = (s->cs == 2 ? 3 : 2) * stage + tab_bytes(s);
    if(!rocke_arch_fits_lds(target, (long)lds))
    {
        return fmt("LDS %lld B > %d", lds, (int)target->lds_capacity_bytes);
    }
    if(!problem_dtype_ok(&p))
    {
        return "fp16/bf16 only";
    }
    const char* ab = problem_is_bf16(&p) ? "bf16" : "f16";
    if(!rocke_mma_catalog_has_shape(
           rocke_archtarget_mma(target), "mma", ab, ab, "fp32", 16, 16, 32))
    {
        return fmt("missing mfma_f32_16x16x32_%s on %s", ab, arch);
    }
    if(rocke_direct_problem_dims_reason(&p, msg, sizeof msg))
    {
        return msg;
    }
    if(s->xcd < 0 || s->xcd > 3)
    {
        return "xcd: off, on, 2 (contiguous XCD runs) or 3 (no rotation)";
    }
    if((s->xcd == 2 || s->xcd == 3) && (!s->sw || s->ds || s->zl || s->cs != 1))
    {
        return "xcd 2 / 3: sw, cs 1, no ds / zl";
    }
    if(s->kmaj && (!s->xcd || s->ds || s->grp))
    {
        return "kmaj: needs xcd, no ds, no grp";
    }
    if(s->sw && (s->cs != 1 || s->ds || s->flat || p.KH * p.KW < 3))
    {
        return "sw: cs 1 only, no ds or flat, at least 3 taps";
    }
    if(s->grp)
    {
        if(p.groups < 2)
        {
            return "grp: grouped problems only";
        }
        if(s->ds || s->flat || s->k_tail || s->xt8 || s->cs != 1 || s->split_c != 1)
        {
            return "grp: cs 1 and split_c 1, no ds, flat, k_tail or xt8";
        }
    }
    else if(p.groups != 1)
    {
        return "groups must be 1";
    }
    /* the reduction / output channels of one group (all of them when dense) */
    const long long CC = s->grp ? p.cpg : tc;
    const long long KK = s->grp ? p.kpg : tk;
    if(p.stride < 1 || p.dil_h < 1 || p.dil_w < 1)
    {
        return fmt("stride and dilation must be >= 1 (got s%d d%dx%d)", p.stride, p.dil_h, p.dil_w);
    }
    if(rocke_direct_conv_pad_w_reason(&p, "", msg, sizeof msg))
    {
        return msg;
    }
    if(s->bk != 64 && s->bk != 128 && s->bk != 192 && s->bk != 256)
    {
        return "bk must be 64, 128, 192 or 256";
    }
    if(s->bk == 192 && s->bx != 256)
    {
        return "bk 192 needs bx 256";
    }
    if(s->xt8 && (s->bx != 256 || (s->bk != 128 && s->bk != 256) || s->cs != 1))
    {
        return "xt8: bx 256, bk 128 / 256, cs 1 only";
    }
    if(s->flat)
    {
        if(s->c_tail || s->wt || s->ds || s->cs != 1)
        {
            return "flat: no c_tail, wt, ds or cs";
        }
        if(tc % 8 || tc % kCK == 0)
        {
            return fmt("flat: C %lld must be a multiple of 8 and not of %d", tc, kCK);
        }
        if(s->split_c > ceil_ll((long long)p.KH * p.KW * tc, kCK))
        {
            return "flat: split_c exceeds the reduction chunks";
        }
    }
    else if(s->c_tail)
    {
        if(CC % 8 || CC % kCK == 0)
        {
            return fmt("c_tail: C %lld must be a multiple of 8 and not of %d", CC, kCK);
        }
    }
    else if(CC % kCK)
    {
        return fmt("C %lld must be a multiple of %d", CC, kCK);
    }
    if(s->k_tail)
    {
        if(KK % 8 == 0)
        {
            return fmt("k_tail: K %lld must not be a multiple of 8", KK);
        }
        if(s->cs != 1 || s->wt || s->ds)
        {
            return "k_tail: no cs, wt or ds";
        }
    }
    else if(KK % 8)
    {
        return fmt("K %lld must be a multiple of 8 (16-byte output writes)", KK);
    }
    if(s->split_c < 1 || (!s->flat && s->split_c > ceil_ll(CC, kCK)))
    {
        return fmt("split_c %d must be in [1, C / %d]", s->split_c, kCK);
    }
    if((long long)p.KH * p.KW > 32)
    {
        return "more than 32 taps (the DMA offset table's LDS)";
    }
    if(rocke_direct_conv_problem_Ho(&p) < 1 || rocke_direct_conv_problem_Wo(&p) < 1)
    {
        return "empty output";
    }
    /* every buffer offset is an i32 over the whole tensor (or workspace slab) */
    const long long nx
        = mul_sat(mul_sat(p.N, rocke_direct_conv_problem_Ho(&p)), rocke_direct_conv_problem_Wo(&p));
    const struct
    {
        const char* what;
        long long bytes;
    } tensors[] = {
        {"input", mul_sat(mul_sat(mul_sat(mul_sat(2, p.N), p.H), p.W), tc)},
        {"output", mul_sat(mul_sat(2, nx), tk)},
        {"weights", mul_sat(mul_sat(mul_sat(mul_sat(2, tk), p.KH), p.KW), tc)},
        {"workspace slab", s->split_c > 1 ? mul_sat(mul_sat(4, nx), tk) : 0},
    };
    for(const auto& t : tensors)
    {
        if(t.bytes > kI32Max)
        {
            return fmt("%s exceeds %lld B, the i32 buffer range", t.what, kI32Max);
        }
    }
    /* the whole workspace's size is an i32 kernarg of the conv and the fold */
    if(ws_bytes(s) > kI32Max)
    {
        return fmt("workspace exceeds %lld B, the i32 byte-count kernarg", kI32Max);
    }
    return "";
}

} // namespace

extern "C" rocke_fwd_flex_mfma_spec_t rocke_fwd_flex_mfma_spec_default(void)
{
    rocke_fwd_flex_mfma_spec_t s;
    memset(&s, 0, sizeof(s));
    s.problem = rocke_direct_conv_problem_default();
    s.bk = 256;
    s.split_c = 1;
    s.bx = 256;
    s.cs = 1;
    s.name = "fwd_flex_mfma";
    s.wave_size = 64;
    return s;
}

extern "C" int rocke_fwd_flex_mfma_kt(const rocke_fwd_flex_mfma_spec_t* s)
{
    if(s->xt8)
    {
        return s->bk / 64;
    }
    const int bx = s->bx / 128;
    const int d = bx > 0 ? 16 * (4 / bx) : 0;
    return d > 0 ? s->bk / d : 0;
}

extern "C" int rocke_fwd_flex_mfma_threads_per_block(const rocke_fwd_flex_mfma_spec_t* s)
{
    return 8 * s->wave_size;
}

extern "C" long long rocke_fwd_flex_mfma_items(const rocke_fwd_flex_mfma_spec_t* s)
{
    const rocke_direct_conv_problem_t* p = &s->problem;
    if(s->bk <= 0 || s->bx <= 0)
    {
        return 0;
    }
    if(s->ds)
    {
        const std::pair<int, int> st = ds_strides(s);
        const long long nxp = (long long)p->N * ceil_ll(p->H, st.first) * ceil_ll(p->W, st.second);
        return (long long)ds_order(s).size() * ceil_ll(nxp, s->bx)
               * ceil_ll(rocke_direct_conv_problem_total_c(p), s->bk) * s->split_c;
    }
    const long long nx
        = (long long)p->N * rocke_direct_conv_problem_Ho(p) * rocke_direct_conv_problem_Wo(p);
    const long long tiles = ceil_ll(nx, s->bx) * p->groups * ceil_ll(p->kpg, s->bk);
    return tiles * s->split_c;
}

extern "C" void rocke_fwd_flex_mfma_grid(const rocke_fwd_flex_mfma_spec_t* s, int grid[3])
{
    long long items = rocke_fwd_flex_mfma_items(s);
    if(s->zl)
    {
        const rocke_direct_conv_problem_t* p = &s->problem;
        const std::pair<int, int> st = ds_strides(s);
        const long long nxp = (long long)p->N * ceil_ll(p->H, st.first) * ceil_ll(p->W, st.second);
        items += ceil_ll(nxp, kZlPx);
    }
    grid[0] = (int)(items < kPersist ? items : kPersist);
    grid[1] = 1;
    grid[2] = 1;
}

extern "C" long long rocke_fwd_flex_mfma_workspace_bytes(const rocke_fwd_flex_mfma_spec_t* s)
{
    return ws_bytes(s);
}

extern "C" rocke_status_t
    rocke_fwd_flex_mfma_kernel_name(const rocke_fwd_flex_mfma_spec_t* s, char* out, size_t out_cap)
{
    if(s == NULL || out == NULL || out_cap == 0)
    {
        return ROCKE_ERR_VALUE;
    }
    const rocke_direct_conv_problem_t* p = &s->problem;
    char f_buf[48], p_buf[48], s_buf[24], d_buf[48], k_buf[24], sc_buf[24], x_buf[24], zl_buf[24];
    snprintf(f_buf, sizeof f_buf, "f%dx%d", p->KH, p->KW);
    snprintf(p_buf, sizeof p_buf, "p%dx%d", p->PAD, rocke_direct_conv_problem_pad_w(p));
    snprintf(s_buf, sizeof s_buf, "s%d", p->stride);
    snprintf(d_buf, sizeof d_buf, "d%dx%d", p->dil_h, p->dil_w);
    snprintf(k_buf, sizeof k_buf, "k%d", s->bk);
    snprintf(sc_buf, sizeof sc_buf, "sc%d", s->split_c);
    x_buf[0] = '\0';
    zl_buf[0] = '\0';
    if(s->bx != 256)
    {
        snprintf(x_buf, sizeof x_buf, "x%d", s->bx);
    }
    if(s->zl)
    {
        snprintf(zl_buf, sizeof zl_buf, "zl%d", s->zl);
    }
    const char* parts[8] = {f_buf, p_buf, s_buf, d_buf, k_buf, sc_buf, x_buf, zl_buf};
    static const char* const flag_names[17] = {"bf16",
                                               "xcd",
                                               "xc2",
                                               "xr0",
                                               "km",
                                               "ct",
                                               "cs4",
                                               "cs2",
                                               "wt",
                                               "ds",
                                               "h1",
                                               "zf",
                                               "kx",
                                               "fr",
                                               "x8",
                                               "gr",
                                               "sw"};
    const int on[17] = {problem_is_bf16(p) ? 1 : 0,
                        s->xcd ? 1 : 0,
                        s->xcd == 2,
                        s->xcd == 3,
                        s->kmaj,
                        s->c_tail,
                        s->cs == 4,
                        s->cs == 2,
                        s->wt,
                        s->ds,
                        s->h1,
                        s->zf,
                        s->k_tail,
                        s->flat,
                        s->xt8,
                        s->grp,
                        s->sw};
    return rocke_kernel_name_join(s->name, parts, 8, flag_names, on, 17, out, out_cap, NULL);
}

extern "C" bool rocke_fwd_flex_mfma_is_valid_spec(const rocke_fwd_flex_mfma_spec_t* s,
                                                  const char* arch,
                                                  char* reason,
                                                  size_t reason_cap)
{
    if(s == NULL)
    {
        rocke_spec_set_reason(reason, reason_cap, "null spec");
        return false;
    }
    const std::string why = flex_reason(s, arch ? arch : "gfx950");
    if(!why.empty())
    {
        rocke_spec_set_reason(reason, reason_cap, why.c_str());
        return false;
    }
    rocke_spec_set_reason(reason, reason_cap, "ok");
    return true;
}

/* ===================================================================== *
 *  The main (cs 1) builder
 * ===================================================================== */

namespace
{

using Dict = std::map<std::string, V>;

/* One DMA of a step: operand kind ("x" input, "w" weights, "w4" the weights
 * of wave + 4), instruction, rsrc, voffset, soffset. */
struct DmaOp
{
    const char* kind;
    int i;
    V rsrc, voff, soff;
};

rocke_kernel_def_t*
    build_main(rocke_ir_builder_t* b, const rocke_fwd_flex_mfma_spec_t* spec, const char* arch)
{
    Em e{b};
    const rocke_direct_conv_problem_t& p = spec->problem;
    const bool bf16 = problem_is_bf16(&p);
    const rocke_type_t* io_type = bf16 ? rocke_bf16() : rocke_f16();
    const rocke_type_t* i32_t = rocke_i32();
    const int BK = spec->bk, KT = rocke_fwd_flex_mfma_kt(spec), SP = spec->split_c;
    const int BX = spec->bx;
    const int XT = spec->xt8 ? 8 : kXT; /* 16-pixel MFMA tiles per wave */
    const int WKN = BK / (KT * 16); /* waves of a group along K */
    int NXI = BX / 64, NWI = BK / 64; /* DMA instructions per wave per step */
    const int R = p.KH, S = p.KW;
    const int SH = p.stride, PH = p.PAD, PW = rocke_direct_conv_problem_pad_w(&p);
    const int DH = p.dil_h, DW = p.dil_w;
    const long long IN_E = (long long)BX * kCK, W_E = w_elems(spec); /* elements per stage */
    const bool WT = spec->wt;
    const bool SW = spec->sw; /* positions-inner sweep */
    const int NT = R * S; /* sw: the taps of a chunk */
    int HK = 0;
    if(WT)
    {
        /* wt: NH halves of HK outputs, each [64 reduction rows][128] */
        HK = wt_half(spec);
        NWI = 2 * (BK / HK);
    }

    rocke_attr_set_int(
        b, &b->kernel->attrs, "max_workgroup_size", rocke_fwd_flex_mfma_threads_per_block(spec));
    rocke_attr_set_bool(b, &b->kernel->attrs, "lds_alias_scopes", true);
    rocke_attr_set_int(b, &b->kernel->attrs, "waves_per_eu", 2);
    const bool DS = spec->ds;
    rocke_dconv_params_t P;
    rocke_dconv_emit_params(b, &P, DS ? "dgrad" : "fwd", io_type);
    if(!rocke_ir_builder_ok(b))
    {
        return NULL;
    }

    auto ci = [&](int64_t v) { return e.ci(v); };
    auto dv = [&](V a, int64_t n) { return e.divi(a, n); };
    auto md = [&](V a, int64_t n) { return e.modi(a, n); };

    V c0 = ci(0);
    V c1 = ci(1);
    V c2 = ci(2);
    V oob = ci(kOob);
    V p_N = P.p_N, p_Hi = P.p_Hi, p_Wi = P.p_Wi;
    V p_Ho = P.p_Ho, p_Wo = P.p_Wo;
    V C = P.p_total_c, K = P.p_total_k;
    const std::pair<int, int> tg = tap_grid(spec);
    const int TR = tg.first, TS = tg.second;
    int SDH = 1, SDW = 1;
    std::vector<Phase> PHS_H, PHS_W;
    std::vector<std::pair<int, int>> PORD;
    if(DS)
    {
        /* the forward over dY: input dY [N, Ho, Wo, K], output dX [N, H, W, C] */
        std::swap(p_Hi, p_Ho);
        std::swap(p_Wi, p_Wo);
        std::swap(C, K);
        const std::pair<int, int> st = ds_strides(spec);
        SDH = st.first;
        SDW = st.second;
        PHS_H = ds_phases_axis(spec, 0);
        PHS_W = ds_phases_axis(spec, 1);
        PORD = ds_order(spec);
    }
    /* input / output pixel strides (the totals); grp: C, K become one group's */
    V XS = C, YS = K;
    const bool GRP = spec->grp;
    V NG = NULL;
    if(GRP)
    {
        NG = P.p_groups;
        C = e.div(C, NG);
        K = e.div(K, NG);
    }

    V tid = rocke_b_thread_id_x(b);
    V lane = md(tid, kWave);
    V wave = rocke_b_readfirstlane(b, dv(tid, kWave));
    V g = dv(wave, 4);
    V wq = md(wave, 4);
    V wk = md(wq, WKN);
    V wx = dv(wq, WKN);

    /* the lane-derived LDS bases (input granule, input / weight fragment rows,
     * table row), formed per group in run */
    V h_gran = NULL, h_tab = NULL;
    V h_xin[2] = {NULL, NULL}, h_kw[2] = {NULL, NULL};
    auto lane_bases = [&]() {
        V h_r16 = md(lane, 16);
        V h_q = dv(lane, 16);
        V h_l8 = dv(lane, 8);
        h_gran = e.muli(e.bxor(md(lane, 8), h_l8), 16);
        for(int h = 0; h < 2; ++h)
        {
            V hk = ci(h * 4);
            V hq = e.add(hk, h_q);
            V sl = e.muli(e.bxor(hq, md(h_r16, 8)), 8);
            V xg = e.muli(g, BX / 2);
            V xrow = e.add(e.add(xg, e.muli(wx, XT * 16)), h_r16);
            V krow = e.add(e.muli(wk, KT * 16), h_r16);
            h_xin[h] = e.add(e.muli(xrow, kCK), sl);
            h_kw[h] = e.add(e.muli(krow, kCK), sl);
        }
        h_tab = e.muli(e.add(e.muli(wave, 8), h_l8), NXI);
    };

    /* lane-derived addressing is recomputed where it is used instead of being
     * hoisted into long-lived VGPRs */
    auto opaque_lane = [&]() { return opaque(b, lane, "=v,0"); };

    V Hq = NULL, Wq = NULL, hwq = NULL, NXP = NULL, TPP = NULL, NX = NULL, nxt = NULL, hw = NULL;
    if(DS)
    {
        /* phase-major pixels: phase (a, b) holds dX[n, SDH i + a, SDW j + b]
         * for i < Hq, j < Wq, padded to TPP whole tiles */
        Hq = e.cdivi(p_Ho, SDH);
        Wq = e.cdivi(p_Wo, SDW);
        hwq = e.mul(Hq, Wq);
        NXP = e.mul(p_N, hwq);
        TPP = e.cdivi(NXP, BX);
        nxt = e.muli(TPP, (int64_t)PORD.size());
        NX = e.mul(p_N, e.mul(p_Ho, p_Wo)); /* dX pixels (split-C slab rows) */
    }
    else
    {
        hw = e.mul(p_Ho, p_Wo);
        NX = e.mul(p_N, hw);
        nxt = e.cdivi(NX, BX);
    }

    auto phase_sel = [&](V idx, const std::vector<int>& vals) {
        V out = ci(vals[0]);
        for(size_t i = 1; i < vals.size(); ++i)
        {
            V eq = e.eqi(idx, (int64_t)i);
            V v = ci(vals[i]);
            out = e.select(eq, v, out);
        }
        return out;
    };
    std::vector<int> ord_a, ord_b;
    for(const auto& ab : PORD)
    {
        ord_a.push_back(ab.first);
        ord_b.push_back(ab.second);
    }
    auto phase_ab = [&](V ph, V* pa, V* pb) {
        *pa = phase_sel(ph, ord_a);
        *pb = phase_sel(ph, ord_b);
    };
    auto fields = [](const std::vector<Phase>& ax, int which) {
        std::vector<int> out;
        for(const Phase& f : ax)
        {
            out.push_back(which == 0 ? f.t : which == 1 ? f.pa : f.f0);
        }
        return out;
    };
    struct Pix
    {
        V ok, n, i, j, a, b;
    };
    /* (valid, n, i, j, phase row a, phase column b) of the tile's pixel xi */
    auto ds_pix = [&](V tx, V xi) {
        Pix r;
        V ph = e.div(tx, TPP);
        V t0 = e.sub(tx, e.mul(ph, TPP));
        V xl = e.add(e.muli(t0, BX), xi);
        r.n = e.div(xl, hwq);
        V rem = e.sub(xl, e.mul(r.n, hwq));
        r.i = e.div(rem, Wq);
        r.j = e.sub(rem, e.mul(r.i, Wq));
        r.ok = e.lt(xl, NXP);
        phase_ab(ph, &r.a, &r.b);
        return r;
    };

    V nkt = e.cdivi(K, BK);
    V nktg = NULL;
    if(GRP)
    {
        /* grp: K tile = group * (K tiles per group) + tile */
        nktg = nkt;
        nkt = e.mul(nktg, NG);
    }
    V TI = e.muli(e.mul(nxt, nkt), SP);
    V G = e.smini(TI, kPersist);
    /* ds forms its pixel indices in ds_pix (no hw) */
    std::vector<V> dl;
    if(!DS)
    {
        dl.push_back(hw);
    }
    dl.push_back(p_Wo);
    dl.push_back(nkt);
    dl.push_back(G);
    if(spec->flat)
    {
        dl.push_back(dv(C, 8));
    }
    if(GRP)
    {
        dl.push_back(nktg);
    }
    std::vector<Div> fd = divs(b, lane, dl);
    size_t fi = 0;
    Div f_hw = {}, f_cg = {}, f_nktg = {};
    if(!DS)
    {
        f_hw = fd[fi++];
    }
    const Div f_wo = fd[fi++];
    const Div f_nkt = fd[fi++];
    const Div f_g = fd[fi++];
    if(spec->flat)
    {
        f_cg = fd[fi++];
    }
    if(GRP)
    {
        f_nktg = fd[fi++];
    }

    /* grp: (group, its K tile) of the item's K tile */
    auto grp_tk = [&](V tk, V* gi, V* tkg) {
        *gi = f_nktg.div(b, tk);
        *tkg = e.sub(tk, e.mul(*gi, nktg));
    };

    const bool FLAT = spec->flat;
    V CG = NULL, RSG = NULL, nch = NULL;
    if(FLAT)
    {
        CG = dv(C, 8); /* granules per tap */
        RSG = e.mul(ci((int64_t)R * S), CG); /* granules of a reduction row */
        nch = dv(e.addi(RSG, 7), 8);
    }
    else if(spec->c_tail)
    {
        nch = e.cdivi(C, kCK);
    }
    else
    {
        nch = dv(C, kCK);
    }
    V RSC = e.mul(ci((int64_t)R * S), C);
    V RSK = e.mul(ci((int64_t)R * S), K); /* wt: a reduction row of the KRSC filter */
    V blk = rocke_b_block_id_x(b);
    const bool XR = spec->xcd == 2;
    /* xcd 3: the xcd walk without the per-round rotation */
    const int rot = spec->xcd == 3 ? 0 : kRot;
    V run0 = NULL, wpx = NULL, rounds = NULL;
    if(XR)
    {
        /* xcd 2: XCD x (block % 8) owns a contiguous run of the tiles */
        V is8 = e.eq(md(G, 8), c0);
        V x8 = e.select(is8, md(blk, 8), c0);
        V w8 = e.select(is8, dv(blk, 8), blk);
        wpx = e.select(is8, dv(G, 8), G);
        V q8 = e.select(is8, dv(TI, 8), TI);
        V k8 = ci(8);
        V r8 = e.sub(TI, e.mul(q8, e.select(is8, k8, c1)));
        V xq = e.mul(x8, q8);
        run0 = e.add(e.add(xq, e.smin(x8, r8)), w8);
        V rlen = e.add(q8, e.select(e.lt(x8, r8), c1, c0));
        V rw = e.sub(rlen, w8);
        rounds = e.div(e.add(rw, e.sub(wpx, c1)), wpx);
    }
    else if(spec->xcd)
    {
        /* 8 XCDs; only when G is a multiple of 8 (else keep the dealt order) */
        V g8 = dv(G, 8);
        V xm = e.mul(md(blk, 8), g8);
        V xr = e.add(xm, dv(blk, 8));
        blk = e.select(e.eq(md(G, 8), c0), xr, blk);
    }
    /* the rounds from the grid size's magic (no signed division) */
    V full = f_g.div(b, TI);
    V part = e.sub(TI, e.mul(full, G));
    /* ds: a block takes slot b in even rounds and G - 1 - b in odd ones */
    auto slot = [&](V i) {
        V odd = e.eq(e.mod(i, c2), c1);
        V s = e.sub(e.sub(G, c1), blk);
        return e.select(odd, s, blk);
    };
    if(!XR)
    {
        V last_ok;
        if(DS)
        {
            last_ok = e.lt(slot(full), part);
        }
        else
        {
            V r = e.add(blk, e.muli(full, rot));
            last_ok = e.lt(f_g.mod(b, r), part);
        }
        rounds = e.add(full, e.select(last_ok, c1, c0));
    }

    auto tile_of = [&](V i) {
        if(DS)
        {
            V m = e.mul(i, G);
            return e.add(m, slot(i));
        }
        if(XR)
        {
            return e.add(run0, e.mul(i, wpx));
        }
        V m = e.mul(i, G);
        V r = e.add(blk, e.muli(i, rot));
        return e.add(m, f_g.mod(b, r));
    };

    /* ---- item -> (tile, piece, valid tap ranges) ---- */
    auto chunks = [&](V pc, Dict& out) {
        out["ch0"] = dv(e.mul(pc, nch), SP);
        out["ch1"] = dv(e.mul(e.add(pc, c1), nch), SP);
    };
    auto item_info = [&](V t) {
        Dict out;
        V tl = dv(t, SP);
        V pc = md(t, SP);
        V tx, tk;
        if(spec->kmaj)
        {
            tk = e.div(tl, nxt);
            tx = e.sub(tl, e.mul(tk, nxt));
        }
        else
        {
            tx = f_nkt.div(b, tl);
            tk = e.sub(tl, e.mul(tx, nkt));
        }
        out["pc"] = pc;
        out["tx"] = tx;
        out["tk"] = tk;
        if(DS)
        {
            V ph = e.div(tx, TPP);
            V pa, pb;
            phase_ab(ph, &pa, &pb);
            V fa = phase_sel(pa, fields(PHS_H, 2));
            V fam = e.muli(fa, S);
            out["fb"] = e.add(fam, phase_sel(pb, fields(PHS_W, 2)));
            out["r0"] = c0;
            out["s0"] = c0;
            chunks(pc, out);
            out["r1"] = phase_sel(pa, fields(PHS_H, 0));
            out["s1"] = phase_sel(pb, fields(PHS_W, 0));
            return out;
        }
        if(SW)
        {
            /* every tap runs (padding taps load zeros) */
            out["r0"] = c0;
            out["r1"] = ci(R);
            out["s0"] = c0;
            out["s1"] = ci(S);
            chunks(pc, out);
            if(GRP)
            {
                grp_tk(tk, &out["gi"], &out["tk"]);
            }
            return out;
        }
        V x0 = e.muli(tx, BX);
        V x1 = e.sub(e.smin(e.addi(x0, BX), NX), c1);
        V n0 = f_hw.div(b, x0);
        V n1 = f_hw.div(b, x1);
        V same = e.eq(n0, n1);
        V q0 = e.sub(x0, e.mul(n0, hw));
        V q1 = e.sub(x1, e.mul(n0, hw));
        V ho0 = e.select(same, f_wo.div(b, q0), c0);
        V ho1d = f_wo.div(b, q1);
        V ho1 = e.select(same, ho1d, e.sub(p_Ho, c1));
        V row = e.land(same, e.eq(ho0, ho1));
        V wo0 = e.select(row, e.sub(q0, e.mul(ho0, p_Wo)), c0);
        V wo1s = e.sub(q1, e.mul(ho1, p_Wo));
        V wo1 = e.select(row, wo1s, e.sub(p_Wo, c1));
        /* valid taps are an interval: a prefix of the rows reads only the top
         * padding and a suffix only the bottom */
        V lo_h = e.subi(e.muli(ho1, SH), PH);
        V hi_h = e.subi(e.muli(ho0, SH), PH);
        V lo_w = e.subi(e.muli(wo1, SH), PW);
        V hi_w = e.subi(e.muli(wo0, SH), PW);
        /* the counts in closed form: #{r < n: lo + r d < 0} =
         * min(ceil(max(-lo, 0) / d), n), #{r < n: hi + r d < E} =
         * min(ceil(max(E - hi, 0) / d), n) */
        auto cnt = [&](V v, int d, int n) {
            v = e.smax(v, c0);
            if(d > 1)
            {
                v = e.cdivi(v, d);
            }
            return e.smini(v, n);
        };
        V r0 = cnt(e.sub(c0, lo_h), DH, R);
        V r1 = cnt(e.sub(p_Hi, hi_h), DH, R);
        V s0 = cnt(e.sub(c0, lo_w), DW, S);
        V s1 = cnt(e.sub(p_Wi, hi_w), DW, S);
        if(FLAT)
        {
            /* one cursor tap; the chunks walk the whole reduction */
            r0 = c0;
            r1 = c1;
            s0 = c0;
            s1 = c1;
        }
        /* a tile reading only padding still runs one (zero) step */
        V er = e.ge(r0, r1);
        V empty = e.lor(er, e.ge(s0, s1));
        out["r0"] = e.select(empty, c0, r0);
        out["r1"] = e.select(empty, c1, r1);
        out["s0"] = e.select(empty, c0, s0);
        out["s1"] = e.select(empty, c1, s1);
        chunks(pc, out);
        if(GRP)
        {
            grp_tk(tk, &out["gi"], &out["tk"]);
        }
        return out;
    };

    /* ---- DMA cursor ---- */
    std::vector<std::string> names;
    if(SW)
    {
        names = {"i", "t", "ch", "ch1"};
    }
    else
    {
        names = {"i", "t", "ch", "ch1", "r", "s", "r0", "r1", "s0", "s1"};
    }
    if(DS)
    {
        names.push_back("fb");
    }
    if(WT)
    {
        names.push_back("wk0");
    }
    names.push_back("wo");
    if(WT && NWI >= 4)
    {
        names.push_back("wo1");
    }
    /* Input DMA offsets: a per-wave LDS table [tap][wave][pixel of 8][instr]
     * built at each tile change. */
    const int RSP = tab_taps(spec);
    /* wt: the weight stages first, at LDS offsets below 64 KiB, so their bases
     * fold into the transposing reads' offsets */
    V w_s[2] = {NULL, NULL};
    if(WT && SW)
    {
        /* sw: both stages in one allocation (the stage is a scalar offset) */
        w_s[0] = w_s[1] = e.smem_alloc_1d(rocke_i8(), (int)(2 * W_E * 2), "flex_w");
    }
    else if(WT)
    {
        for(int i = 0; i < 2; ++i)
        {
            std::string h = "flex_w" + std::to_string(i);
            w_s[i] = e.smem_alloc_1d(rocke_i8(), (int)(W_E * 2), h.c_str());
        }
    }
    const bool TB2 = tab_db(spec, arch);
    const int TABN = RSP * 64 * NXI; /* elements of one table buffer */
    V tab_s = e.smem_alloc_1d(i32_t, (TB2 ? 2 : 1) * TABN, "flex_tab");
    /* TB2: the group of the code being built and its round (set by run) */
    bool t_g1 = false;
    V t_rnd = NULL;
    V opix_s = NULL;
    if(DS)
    {
        /* ds: the dX pixel (or -1) of each tile pixel, formed with the offset
         * table and read by the epilogue; 3 buffers by item */
        opix_s = e.smem_alloc_1d(i32_t, 3 * BX, "flex_opix");
    }
    const std::vector<std::pair<int, int>> ZD
        = (DS && spec->zf) ? ds_zero_deltas(spec) : std::vector<std::pair<int, int>>();

    auto tab_index = [&](V tap, V l8, V wv, V buf) {
        V t8 = e.muli(tap, 8);
        V a = e.add(t8, wv);
        V a8 = e.muli(a, 8);
        V x = e.muli(e.add(a8, l8), NXI);
        if(buf == NULL)
        {
            return x;
        }
        return e.add(x, e.muli(buf, TABN));
    };

    /* Wave wv's (default: this wave's) table rows for the tile, into buffer
     * buf (TB2): lane (l8, c) writes taps c, c + 8, ... of its NXI pixels. */
    auto build_tab = [&](Dict& info, V ln, V l8, V wv, V buf) {
        if(wv == NULL)
        {
            wv = wave;
        }
        V c = md(ln, 8);
        struct TPix
        {
            V xok, hb, wb, base;
        };
        std::vector<TPix> pix;
        for(int i = 0; i < NXI; ++i)
        {
            V xa = e.muli(info["tx"], BX);
            V wn = e.muli(wv, NXI);
            V wi = e.addi(wn, i);
            V x = e.add(xa, e.add(e.muli(wi, 8), l8));
            if(DS)
            {
                V xn = e.muli(wave, NXI);
                V xii = e.addi(xn, i);
                V xi = e.add(e.muli(xii, 8), l8);
                Pix q = ds_pix(info["tx"], xi);
                V hb = e.sub(q.i, phase_sel(q.a, fields(PHS_H, 1)));
                V wb = e.sub(q.j, phase_sel(q.b, fields(PHS_W, 1)));
                V nh = e.mul(q.n, p_Hi);
                V base = e.add(e.mul(e.add(nh, hb), p_Wi), wb);
                pix.push_back(TPix{q.ok, hb, wb, base});
                V oh = e.add(e.muli(q.i, SDH), q.a);
                V ow = e.add(e.muli(q.j, SDW), q.b);
                V ohk = e.lt(oh, p_Ho);
                V ook = e.land(q.ok, e.land(ohk, e.lt(ow, p_Wo)));
                V no = e.mul(q.n, p_Ho);
                V op = e.add(e.mul(e.add(no, oh), p_Wo), ow);
                for(size_t d = 0; d < ZD.size(); ++d)
                {
                    V zh = e.lt(e.addi(oh, ZD[d].first), p_Ho);
                    V zok = e.land(zh, e.lt(e.addi(ow, ZD[d].second), p_Wo));
                    V bit = ci((int64_t)1 << (kZfBits + d));
                    op = e.add(op, e.select(zok, bit, c0));
                }
                V r3 = e.muli(md(info["rnd"], 3), BX);
                V idx = e.add(r3, xi);
                V m1 = ci(-1);
                V val = e.select(ook, op, m1);
                V ix[2] = {c0, idx};
                rocke_b_smem_store_vN(b, opix_s, ix, 2, val, 1);
                continue;
            }
            V n = f_hw.div(b, x);
            V rem = e.sub(x, e.mul(n, hw));
            V ho = f_wo.div(b, rem);
            V wo = e.sub(rem, e.mul(ho, p_Wo));
            V hb = e.subi(e.muli(ho, SH), PH);
            V wb = e.subi(e.muli(wo, SH), PW);
            V nh = e.mul(n, p_Hi);
            V base = e.add(e.mul(e.add(nh, hb), p_Wi), wb);
            pix.push_back(TPix{e.lt(x, NX), hb, wb, base});
        }
        V gc = GRP ? e.mul(info["gi"], C) : NULL;
        for(int kk = 0; kk < RSP / 8; ++kk)
        {
            V t = e.addi(c, 8 * kk);
            V r = dv(t, TS);
            V s_ = e.sub(t, e.muli(r, TS));
            V dh = e.muli(r, DH);
            V dw = e.muli(s_, DW);
            V tok = e.lti(t, (int64_t)TR * TS);
            V toff = e.add(e.mul(dh, p_Wi), dw);
            std::vector<V> ents;
            for(const TPix& q : pix)
            {
                V a = e.land(tok, q.xok);
                V hok = rocke_b_cmp_ult(b, e.add(q.hb, dh), p_Hi);
                V wok = rocke_b_cmp_ult(b, e.add(q.wb, dw), p_Wi);
                V ok = e.land(a, e.land(hok, wok));
                V fo = e.mul(e.add(q.base, toff), XS);
                if(GRP)
                {
                    fo = e.add(fo, gc);
                }
                ents.push_back(e.select(ok, e.mul(fo, c2), oob));
            }
            V val = NXI == 1 ? ents[0] : rocke_b_vec_pack(b, ents.data(), (int)ents.size(), i32_t);
            V ix[2] = {c0, tab_index(t, l8, wv, buf)};
            rocke_b_smem_store_vN(b, tab_s, ix, 2, val, NXI);
        }
    };

    /* TB2, group 1: the rows of this wave and of its group-0 partner (wave -
     * 4) for item k, into buffer k & 1 */
    auto build_tab_all = [&](Dict& info, V k) {
        V ln = opaque_lane();
        V l8 = dv(ln, 8);
        V buf = e.land(k, c1);
        build_tab(info, ln, l8, wave, buf);
        ln = opaque_lane();
        V l8b = dv(ln, 8);
        build_tab(info, ln, l8b, e.subi(wave, 4), buf);
    };

    /* The cursor's lane parts (and the item's offset-table rows). */
    auto lanes_of = [&](Dict& info, bool first) {
        Dict out;
        V ln = opaque_lane();
        V l8 = dv(ln, 8);
        V c8 = e.bxor(md(ln, 8), l8);
        if(first || DS)
        {
            build_tab(info, ln, l8, NULL, TB2 ? c0 : NULL);
        }
        else if(TB2)
        {
            if(t_g1)
            {
                /* built here unless built after the epilogue of item k - 2 */
                V k = info["rnd"], rnd_ = t_rnd;
                V k1 = e.eq(k, c1);
                e.if_(e.land(k1, e.lt(k, rounds)), [&]() { build_tab(info, ln, l8, NULL, c1); });
                V kr = e.lt(k, rounds);
                V need = e.land(kr, e.ge(k, e.add(rnd_, c2)));
                e.if_(need, [&]() { build_tab_all(info, k); });
            }
        }
        else
        {
            /* no table for the item past the block's last */
            e.if_(e.lt(info["rnd"], rounds), [&]() { build_tab(info, ln, l8, NULL, NULL); });
        }
        /* weight rows: wave w DMAs LDS rows (w * NWI + i) * 8 + l / 8; row 4g
         * + j of a 16-row tile holds channel (g & 1) * 8 + (g >> 1) * 4 + j.
         * Tail rows (k >= K) fall past B_bytes: zero fill, never stored. */
        V m = l8;
        V gq = dv(m, 4);
        V g8 = e.muli(gq, 8);
        V chan = e.add(g8, md(m, 4));
        V k = NULL;
        if(NWI == 1)
        {
            /* bk 64: one weight DMA per wave */
            V w2 = e.muli(dv(wave, 2), 16);
            V wbase = e.add(w2, e.muli(md(wave, 2), 4));
            k = e.add(e.add(e.muli(info["tk"], BK), wbase), chan);
        }
        else if(WT)
        {
            k = NULL;
        }
        else if(NWI == 3)
        {
            /* bk 192: the instruction's channel offset carries the wave too */
            k = e.add(e.muli(info["tk"], BK), chan);
        }
        else
        {
            V kt0 = e.muli(info["tk"], BK);
            k = e.add(e.add(kt0, e.muli(wave, NWI * 8)), chan);
        }
        if(GRP && k != NULL)
        {
            /* the group's filter rows */
            k = e.add(k, e.mul(info["gi"], K));
        }
        if(WT)
        {
            V w16 = dv(ln, 16);
            V s16 = md(ln, 16);
            V cbx = e.muli(e.bxor(dv(s16, 2), w16), 16);
            V cb = e.add(cbx, e.muli(md(s16, 2), 8));
            auto wlane = [&](V qb) {
                V col = e.bxor(cb, e.muli(qb, 64));
                V wr = e.mul(w16, RSK);
                V v = e.mul(e.add(wr, col), c2);
                if(HK == 128)
                {
                    return v;
                }
                return e.select(e.lti(col, HK), v, oob);
            };
            if(NWI >= 4)
            {
                out["wo"] = wlane(c0);
                out["wo1"] = wlane(c1);
            }
            else
            {
                /* q = 2 wave + i: qb = wave & 1 */
                out["wo"] = wlane(md(wave, 2));
            }
            /* the scalar part of the tile: its first output, the wave's half
             * and rows (made scalar) */
            V q0 = e.muli(wave, NWI);
            V a = e.muli(info["tk"], BK);
            V hh = e.muli(dv(q0, 16), HK);
            V a2 = e.add(a, hh);
            V rr = e.mul(e.muli(md(q0, 16), 4), RSK);
            V wkk = e.add(a2, rr);
            if(GRP)
            {
                /* the group's reduction rows of the KRSC filter */
                wkk = e.add(wkk, e.mul(e.mul(info["gi"], C), RSK));
            }
            out["wk0"] = rocke_b_readfirstlane(b, e.mul(wkk, c2));
        }
        else
        {
            V kr = e.mul(k, RSC);
            out["wo"] = e.mul(e.add(kr, e.muli(c8, 8)), c2);
        }
        return out;
    };

    auto cursor_at = [&](V i, bool first) {
        V t = tile_of(i);
        Dict info = item_info(t);
        info["rnd"] = i;
        Dict cur;
        cur["i"] = i;
        cur["t"] = t;
        cur["ch"] = info["ch0"];
        cur["ch1"] = info["ch1"];
        cur["r"] = info["r0"];
        cur["s"] = info["s0"];
        cur["r0"] = info["r0"];
        cur["r1"] = info["r1"];
        cur["s0"] = info["s0"];
        cur["s1"] = info["s1"];
        if(DS)
        {
            cur["fb"] = info["fb"];
        }
        Dict ln = lanes_of(info, first);
        for(auto& kv : ln)
        {
            cur[kv.first] = kv.second;
        }
        return cur;
    };

    const std::vector<const rocke_type_t*> cur_types(names.size(), i32_t);
    auto of_names = [&](Dict& d) {
        std::vector<V> out;
        for(const std::string& k : names)
        {
            out.push_back(d[k]);
        }
        return out;
    };
    auto zip_names = [&](const std::vector<V>& v, size_t off) {
        Dict d;
        for(size_t k = 0; k < names.size() && off + k < v.size(); ++k)
        {
            d[names[k]] = v[off + k];
        }
        return d;
    };
    std::function<Dict(Dict&)> cur_adv = [&](Dict& cur) {
        V s = e.add(cur["s"], c1);
        V ws = e.ge(s, cur["s1"]);
        s = e.select(ws, cur["s0"], s);
        V r = e.add(cur["r"], e.select(ws, c1, c0));
        V wr = e.ge(r, cur["r1"]);
        r = e.select(wr, cur["r0"], r);
        V ch = e.add(cur["ch"], e.select(wr, c1, c0));
        V wc = e.ge(ch, cur["ch1"]);
        Dict keep = cur;
        keep["s"] = s;
        keep["r"] = r;
        keep["ch"] = ch;
        std::vector<V> res = if_else(
            b,
            wc,
            cur_types,
            [&]() {
                Dict nx = cursor_at(e.add(cur["i"], c1), false);
                return of_names(nx);
            },
            [&]() { return of_names(keep); });
        return zip_names(res, 0);
    };

    /* ---- LDS ---- */
    V in_s[2];
    if(SW)
    {
        /* one allocation per operand, stage st at st * (stage size) */
        in_s[0] = in_s[1] = e.smem_alloc_1d(io_type, (int)(2 * IN_E), "flex_in");
        if(!WT)
        {
            w_s[0] = w_s[1] = e.smem_alloc_1d(io_type, (int)(2 * W_E), "flex_w");
        }
    }
    else
    {
        for(int i = 0; i < 2; ++i)
        {
            std::string h = "flex_in" + std::to_string(i);
            in_s[i] = e.smem_alloc_1d(io_type, (int)IN_E, h.c_str());
        }
        if(!WT)
        {
            for(int i = 0; i < 2; ++i)
            {
                std::string h = "flex_w" + std::to_string(i);
                w_s[i] = e.smem_alloc_1d(io_type, (int)W_E, h.c_str());
            }
        }
    }
    V in_a[2], w_a[2];
    for(int i = 0; i < 2; ++i)
    {
        in_a[i] = rocke_b_smem_addr_of(b, in_s[i]);
    }
    for(int i = 0; i < 2; ++i)
    {
        w_a[i] = rocke_b_smem_addr_of(b, w_s[i]);
    }

    /* A stage: static (st >= 0) or a scalar parity (sw). */
    struct St
    {
        int s;
        V v;
    };
    auto sv = [&](const St& st) { return st.v != NULL ? st.v : ci(st.s); };
    auto s_other = [&](const St& st) {
        if(st.v == NULL)
        {
            return St{1 - st.s, NULL};
        }
        return St{-1, e.bxor(st.v, c1)};
    };

    auto dma_issue = [&](const std::vector<DmaOp>& ops, const St& st) {
        for(const DmaOp& op : ops)
        {
            const bool x = strcmp(op.kind, "x") == 0;
            V* base = x ? in_a : w_a;
            const int n = x ? NXI : NWI;
            V wv = strcmp(op.kind, "w4") == 0 ? e.addi(wave, 4) : wave;
            if(SW)
            {
                /* the wave's first slot of the stage as one SGPR, the
                 * instruction's slot the m0 add's immediate */
                const long long sb = x ? IN_E * 2 : W_E * 2;
                V wo = e.muli(wv, n * 1024);
                V off = e.add(wo, e.muli(sv(st), sb));
                V pa = rocke_b_smem_ptr_add(b, base[0], e.zext64(off));
                V sbase = rocke_b_readfirstlane(b, rocke_b_trunc(b, pa, i32_t));
                dma_lds_at(b, op.rsrc, sbase, op.i * 1024, op.voff, op.soff);
                continue;
            }
            V wn = e.muli(wv, n);
            V off = e.muli(e.addi(wn, op.i), 1024);
            V dst = rocke_b_smem_ptr_add(b, base[st.s], e.zext64(off));
            dma_lds(b, op.rsrc, dst, op.voff, op.soff);
        }
    };

    /* (operand, instruction, rsrc, voffset, soffset) of the cursor's DMAs
     * (parts: "x" input, "w" weights; sw: stap the static tap, ct whether the
     * chunk may be the channel tail's) */
    auto dma_prep = [&](Dict& cur, const char* parts, int stap, bool ct) {
        std::vector<DmaOp> ops;
        const bool CT = spec->c_tail && ct;
        V live = e.lt(cur["t"], TI);
        V xr = rocke_b_buffer_rsrc(b, P.A, e.select(live, P.A_bytes, c0));
        V tap;
        if(stap >= 0)
        {
            tap = ci(stap);
        }
        else
        {
            tap = e.add(e.muli(cur["r"], TS), cur["s"]);
        }
        V g_ok = NULL, fl_gran = NULL;
        if(CT)
        {
            /* this lane's granule lies past C in the last chunk: zeros */
            V tl_ln = opaque_lane();
            V m8 = md(tl_ln, 8);
            V tl_c8 = e.bxor(m8, dv(tl_ln, 8));
            V cc = e.muli(cur["ch"], kCK);
            V gg = e.add(cc, e.muli(md(tl_c8, 8), 8));
            g_ok = e.lt(gg, C);
        }
        if(FLAT)
        {
            /* this lane's granule of the step: tap and channel granule */
            V fl_ln = opaque_lane();
            V c8 = e.muli(cur["ch"], 8);
            V m8 = md(fl_ln, 8);
            V fl_g = e.add(c8, e.bxor(m8, dv(fl_ln, 8)));
            g_ok = e.lt(fl_g, RSG);
            tap = e.select(g_ok, f_cg.div(b, fl_g), c0);
            fl_gran = e.muli(e.sub(fl_g, e.mul(tap, CG)), 16);
        }
        if(strchr(parts, 'x') != NULL)
        {
            V ta;
            if(stap >= 0)
            {
                ta = e.addi(h_tab, (int64_t)stap * 64 * NXI);
            }
            else
            {
                ta = e.add(e.muli(tap, 64 * NXI), h_tab);
            }
            if(TB2)
            {
                ta = e.add(ta, e.muli(e.land(cur["i"], c1), TABN));
            }
            V ent = e.smem_load(tab_s, c0, ta, i32_t, NXI);
            V soff = FLAT ? c0 : e.muli(cur["ch"], kCK * 2);
            for(int i = 0; i < NXI; ++i)
            {
                V en = NXI == 1 ? ent : rocke_b_vec_extract(b, ent, i);
                V voff = e.add(en, FLAT ? fl_gran : h_gran);
                if(CT || FLAT)
                {
                    voff = e.select(g_ok, voff, oob);
                }
                ops.push_back(DmaOp{"x", i, xr, voff, soff});
            }
        }
        if(strchr(parts, 'w') == NULL)
        {
            return ops;
        }
        V wr = rocke_b_buffer_rsrc(b, P.Bp, e.select(live, P.B_bytes, c0));
        if(WT)
        {
            /* piece q = wave * NWI + i: rows 4 (q % 16) + l // 16 of the
             * chunk, outputs k0 + HK (q // 16) + column of the flipped tap */
            V tapf;
            if(DS)
            {
                V rr = e.muli(cur["r"], SDH * S);
                V a = e.add(rr, e.muli(cur["s"], SDW));
                tapf = e.sub(cur["fb"], a);
            }
            else if(stap >= 0)
            {
                tapf = ci((int64_t)R * S - 1 - stap);
            }
            else
            {
                tapf = e.sub(ci((int64_t)R * S - 1), tap);
            }
            V cc = e.muli(cur["ch"], kCK);
            V wn = e.muli(wave, NWI);
            V row0 = e.add(cc, e.muli(md(wn, 16), 4));
            V ck = e.muli(cur["ch"], kCK);
            V ckr = e.mul(ck, RSK);
            V sb = e.add(e.mul(e.add(ckr, e.mul(tapf, K)), c2), cur["wk0"]);
            V ct_l16 = NULL;
            if(CT)
            {
                /* rows past C (the last chunk) read zeros */
                ct_l16 = dv(opaque_lane(), 16);
            }
            for(int i = 0; i < NWI; ++i)
            {
                V voff = (NWI >= 4 && ((i >> 1) & 1)) ? cur["wo1"] : cur["wo"];
                if(CT)
                {
                    V rw = e.addi(row0, 4 * i);
                    V rok = e.lt(e.add(rw, ct_l16), C);
                    voff = e.select(rok, voff, oob);
                }
                V soff = i ? e.add(sb, e.muli(RSK, 8 * i)) : sb;
                ops.push_back(DmaOp{"w", i, wr, voff, soff});
            }
            return ops;
        }
        if(DS)
        {
            /* the transposed filter (C/K swapped, flipped): position R*S-1-f */
            V k = ci((int64_t)R * S - 1);
            V rr = e.muli(cur["r"], SDH * S);
            V a = e.add(rr, e.muli(cur["s"], SDW));
            tap = e.sub(k, e.sub(cur["fb"], a));
        }
        V woff;
        if(FLAT)
        {
            /* the KRSC row read linearly */
            woff = e.muli(cur["ch"], kCK * 2);
        }
        else
        {
            V tc = e.mul(tap, C);
            woff = e.mul(e.add(tc, e.muli(cur["ch"], kCK)), c2);
        }
        for(int i = 0; i < NWI; ++i)
        {
            V wsh = ci(((i & 1) * 4 + (i >> 1) * 16) * 2);
            if(NWI == 3)
            {
                V w3 = e.muli(wave, 3);
                V ii = e.addi(w3, i);
                V a = e.muli(dv(ii, 2), 16);
                wsh = e.mul(e.add(a, e.muli(md(ii, 2), 4)), c2);
            }
            V vo = (CT || FLAT) ? e.select(g_ok, cur["wo"], oob) : cur["wo"];
            V so = e.add(woff, e.mul(RSC, wsh));
            ops.push_back(DmaOp{"w", i, wr, vo, so});
        }
        return ops;
    };

    /* ---- fragments ---- */
    V wt_l = NULL; /* wt: the transposing reads' lane part (set below) */
    /* A fragment (16 outputs x 32 reduction rows): two transposing reads. */
    auto wt_frag = [&](int st, int h, int kt, V wl) {
        V base = kt ? e.bxor(wl, ci(kt * 32)) : wl;
        V lo = e.ds_read_tr(w_s[st], c0, e.addi(base, h * 32 * 256), io_type);
        V hi = e.ds_read_tr(w_s[st], c0, e.addi(base, (h * 32 + 4) * 256), io_type);
        return rocke_b_vec_concat(b, lo, hi);
    };

    /* The step's A / B fragments from stage st; hook(i) is called before the
     * i-th LDS read. sw: st is a scalar parity, the stage's element offset is
     * added to the lane bases. */
    auto frags = [&](const St& st0,
                     const std::function<void(int)>& hook,
                     std::vector<std::vector<V>>& a,
                     std::vector<std::vector<V>>& bb) {
        a.assign(2, std::vector<V>());
        bb.assign(2, std::vector<V>());
        int nrd = 0;
        int st = st0.s;
        V xin[2] = {h_xin[0], h_xin[1]}, kw[2] = {h_kw[0], h_kw[1]};
        V wo_ = NULL;
        if(SW)
        {
            V xo = e.muli(sv(st0), IN_E);
            wo_ = e.muli(sv(st0), WT ? W_E * 2 : W_E);
            for(int h = 0; h < 2; ++h)
            {
                xin[h] = e.add(h_xin[h], xo);
            }
            if(!WT)
            {
                for(int h = 0; h < 2; ++h)
                {
                    kw[h] = e.add(h_kw[h], wo_);
                }
            }
            st = 0;
        }
        auto tick = [&]() {
            if(hook)
            {
                hook(nrd);
            }
            ++nrd;
        };
        V tl = NULL;
        for(int h = 0; h < 2; ++h)
        {
            for(int xt = 0; xt < XT; ++xt)
            {
                tick();
                V idx = e.addi(xin[h], (int64_t)xt * 16 * kCK);
                bb[h].push_back(e.smem_load(in_s[st], c0, idx, io_type, 8));
            }
            if(WT)
            {
                if(h == 0)
                {
                    /* KT 8: the XORs stay in the step (the 8 addresses kept
                     * in registers spill) */
                    tl = wt_l;
                    if(KT >= 8)
                    {
                        tl = opaque(b, wt_l, "=v,0");
                    }
                    if(wo_ != NULL)
                    {
                        /* (the stage offset is above the XORed bits) */
                        tl = e.add(tl, wo_);
                    }
                }
                for(int kt = 0; kt < KT; ++kt)
                {
                    a[h].push_back(wt_frag(st, h, kt, tl));
                }
                continue;
            }
            for(int kt = 0; kt < KT; ++kt)
            {
                tick();
                V idx = e.addi(kw[h], (int64_t)kt * 16 * kCK);
                a[h].push_back(e.smem_load(w_s[st], c0, idx, io_type, 8));
            }
        }
    };

    if(WT)
    {
        /* lane (tb, ti, tj) of a transposing read; a fragment's byte address
         * is one XOR of a lane constant */
        V kt0 = e.muli(wk, KT);
        const int GPH = HK / 16;
        V wt_wv = rocke_b_readfirstlane(b, e.muli(md(kt0, GPH), 32));
        V wt_hv = rocke_b_readfirstlane(b, e.muli(dv(kt0, GPH), kCK * 256));
        V ln = lane;
        auto bits = [&](int sh, int mask, int to) {
            V v = sh ? e.lshri(ln, sh) : ln;
            v = e.land(v, ci(mask));
            return to ? e.shli(v, to) : v;
        };
        V b1 = bits(4, 3, 11);
        V b2 = bits(2, 3, 8);
        V o1 = e.lor(b1, b2);
        V b3 = bits(2, 3, 5);
        V b4 = bits(4, 1, 7);
        V o2 = e.lor(o1, e.lor(b3, b4));
        V b5 = bits(0, 1, 4);
        V b6 = bits(1, 1, 3);
        V lp = e.lor(o2, e.lor(b5, b6));
        wt_l = e.add(e.bxor(lp, wt_wv), wt_hv);
    }

    V zero = NULL; /* set below, before any step */

    /* The step's MFMAs; inter callables are spread evenly between them from
     * MFMA start on (pinned by scheduling barriers). NULL accumulators: an
     * item's first step (C = 0). */
    auto mfmas = [&](std::vector<V> accs,
                     const std::vector<std::vector<V>>& a,
                     const std::vector<std::vector<V>>& bb,
                     const std::vector<std::function<void()>>& inter,
                     int start) {
        const int n = 2 * XT * KT;
        const int every = inter.empty() ? 0 : (n - start) / (int)inter.size();
        int j = 0;
        for(int h = 0; h < 2; ++h)
        {
            for(int xt = 0; xt < XT; ++xt)
            {
                for(int kt = 0; kt < KT; ++kt)
                {
                    const int q = j - start;
                    if(every && q >= 0 && q % every == 0 && q / every < (int)inter.size())
                    {
                        rocke_b_sched_barrier(b, 0);
                        inter[q / every]();
                        rocke_b_sched_barrier(b, 0);
                    }
                    const int i = xt * KT + kt;
                    V c_in = accs[i] == NULL ? zero : accs[i];
                    accs[i] = e.mfma16x16x32(bf16, a[h][kt], bb[h][xt], c_in);
                    ++j;
                }
            }
        }
        return accs;
    };

    auto bar = [&](int prio) {
        rocke_b_s_setprio(b, prio);
        rocke_b_s_barrier_bare(b);
        rocke_b_sched_barrier(b, 0);
    };

    /* ---- epilogue ---- */
    const rocke_type_t* i32x2 = rocke_vector_type(b, i32_t, 2);
    const rocke_type_t* i32x4 = rocke_vector_type(b, i32_t, 4);
    const rocke_type_t* half8 = rocke_vector_type(b, io_type, 8);
    V d_rsrc = rocke_b_buffer_rsrc(b, P.D, P.D_bytes);

    auto epilogue = [&](const std::vector<V>& accs, V t, V rnd) {
        V tl = dv(t, SP);
        V tx, tk;
        if(spec->kmaj)
        {
            tk = e.div(tl, nxt);
            tx = e.sub(tl, e.mul(tk, nxt));
        }
        else
        {
            tx = f_nkt.div(b, tl);
            tk = e.sub(tl, e.mul(tx, nkt));
        }
        V gk = NULL;
        if(GRP)
        {
            V gi;
            grp_tk(tk, &gi, &tk);
            gk = e.mul(gi, K); /* the group's first output channel */
        }
        V ln = opaque_lane();
        md(ln, 16);
        dv(ln, 16);
        /* lane 4 p + j stores pixel p's j-th 16 bytes of the store's 64: one
         * ds_bpermute per dword pair */
        V l16 = dv(ln, 4);
        V q = md(ln, 4);
        V sq = q;
        if(SP != 1)
        {
            V a = e.mul(md(q, 2), c2);
            sq = e.add(a, dv(q, 2));
        }
        V src = e.muli(e.add(e.muli(sq, 16), l16), 4);

        auto permute = [&](const std::vector<V>& words) {
            std::vector<V> out;
            for(int hlf = 0; hlf < 2; ++hlf)
            {
                V uv[2] = {words[2 * hlf], words[2 * hlf + 1]};
                V pk = rocke_b_vec_pack(b, uv, 2, i32_t);
                V pr = rocke_b_ds_bpermute_b64(b, src, rocke_b_vec_bitcast(b, pk, rocke_i64()));
                pr = rocke_b_vec_bitcast(b, pr, i32x2);
                for(int j = 0; j < 2; ++j)
                {
                    out.push_back(rocke_b_vec_extract(b, pr, j));
                }
            }
            return out;
        };

        V xa = e.muli(tx, BX);
        V xg = e.add(xa, e.muli(g, BX / 2));
        V xw = e.add(e.muli(wx, XT * 16), l16);
        V xbase = e.add(xg, xw);
        V ka = e.muli(tk, BK);
        V kbase = e.add(ka, e.muli(wk, KT * 16));

        V zfl = NULL;
        /* (valid, output pixel) of tile pixel x: x itself, or for ds the dX
         * pixel of its phase */
        auto opix = [&](V x, V* xok, V* xp) {
            if(!DS)
            {
                *xok = e.lt(x, NX);
                *xp = x;
                return;
            }
            V r3 = e.muli(md(rnd, 3), BX);
            V idx = e.add(r3, e.sub(x, e.muli(tx, BX)));
            V en = e.smem_load(opix_s, c0, idx, i32_t, 1);
            if(en != NULL && en->type != NULL && en->type->kind == ROCKE_TYPE_VECTOR)
            {
                en = rocke_b_vec_extract(b, en, 0);
            }
            if(spec->zf)
            {
                zfl = en;
                *xok = e.ge(en, c0);
                *xp = e.land(en, ci((1LL << kZfBits) - 1));
                return;
            }
            *xok = e.ge(en, c0);
            *xp = en;
        };

        if(SP == 1)
        {
            V qa = e.muli(dv(q, 2), 16);
            V kq = e.add(qa, e.muli(md(q, 2), 8)); /* = 8 q */
            std::vector<std::vector<V>> packed(XT * (KT / 2));
            for(int xt = 0; xt < XT; ++xt)
            {
                for(int cc = 0; cc < KT / 2; ++cc)
                {
                    V da = rocke_b_vec_bitcast(
                        b, rocke_dconv_trunc_f32(b, bf16, accs[xt * KT + 2 * cc]), i32x2);
                    V db = rocke_b_vec_bitcast(
                        b, rocke_dconv_trunc_f32(b, bf16, accs[xt * KT + 2 * cc + 1]), i32x2);
                    V lo0, hi0, lo1, hi1;
                    V a0 = rocke_b_vec_extract(b, da, 0);
                    rocke_b_permlane32_swap(b, a0, rocke_b_vec_extract(b, db, 0), &lo0, &hi0);
                    V a1 = rocke_b_vec_extract(b, da, 1);
                    rocke_b_permlane32_swap(b, a1, rocke_b_vec_extract(b, db, 1), &lo1, &hi1);
                    packed[xt * (KT / 2) + cc] = {lo0, lo1, hi0, hi1};
                }
            }
            /* every accumulator narrowed first: the permutes then take
             * registers from the freed half */
            rocke_b_sched_barrier(b, 0);
            V klim = e.sub(K, kbase); /* channels left from the wave's first */
            for(int xt = 0; xt < XT; ++xt)
            {
                V xok, xp;
                opix(e.addi(xbase, xt * 16), &xok, &xp);
                V b0 = NULL;
                if(!spec->k_tail)
                {
                    /* the store offsets as one select per 16-pixel tile, the
                     * channel step as the store's immediate offset */
                    V po = e.mul(xp, YS);
                    if(GRP)
                    {
                        po = e.add(po, gk);
                    }
                    V pk = e.add(po, e.add(kbase, kq));
                    b0 = e.select(xok, e.mul(pk, c2), oob);
                }
                std::vector<V> zb;
                if(spec->zf && !spec->zl)
                {
                    /* the tapless neighbours' pixels: zeros */
                    for(size_t d = 0; d < ZD.size(); ++d)
                    {
                        V bit = ci((int64_t)1 << (kZfBits + d));
                        V nz = rocke_b_cmp_ne(b, e.land(zfl, bit), c0);
                        V zok = e.land(xok, nz);
                        V da = ci(ZD[d].first);
                        V dm = e.mul(da, p_Wo);
                        V dp = e.addi(dm, ZD[d].second);
                        V dk = e.mul(e.mul(dp, K), c2);
                        zb.push_back(e.select(zok, e.add(b0, dk), oob));
                    }
                }
                for(int cc = 0; cc < KT / 2; ++cc)
                {
                    std::vector<V> words = permute(packed[xt * (KT / 2) + cc]);
                    V val = rocke_b_vec_bitcast(
                        b, rocke_b_vec_pack(b, words.data(), 4, i32_t), half8);
                    V k = e.add(e.addi(kbase, cc * 32), kq);
                    if(!zb.empty())
                    {
                        V kok = e.lt(kq, e.subi(klim, cc * 32));
                        V zs[4] = {c0, c0, c0, c0};
                        V zv = rocke_b_vec_bitcast(b, rocke_b_vec_pack(b, zs, 4, i32_t), half8);
                        for(V z : zb)
                        {
                            V off = e.addi(e.select(kok, z, oob), cc * 64);
                            rocke_dconv_buf_store_vN(b, bf16, d_rsrc, off, c0, zv, 4);
                        }
                    }
                    if(spec->k_tail)
                    {
                        /* 16-bit stores, each masked by K */
                        V xk = e.mul(xp, K);
                        V e0 = e.mul(e.add(xk, k), c2);
                        for(int j = 0; j < 8; ++j)
                        {
                            V kj = e.lt(e.addi(k, j), K);
                            V ok = e.land(xok, kj);
                            V off = e.select(ok, e.addi(e0, 2 * j), oob);
                            V v = rocke_b_vec_extract(b, val, j);
                            if(bf16)
                            {
                                rocke_b_buffer_store_bf16(b, d_rsrc, off, c0, v);
                            }
                            else
                            {
                                rocke_b_buffer_store_f16(b, d_rsrc, off, c0, v);
                            }
                        }
                        continue;
                    }
                    V kok = e.lt(kq, e.subi(klim, cc * 32));
                    V off = e.addi(e.select(kok, b0, oob), cc * 64);
                    rocke_dconv_buf_store_vN(b, bf16, d_rsrc, off, c0, val, 4);
                }
            }
        }
        else
        {
            V pc = md(t, SP);
            V slab = e.muli(e.mul(NX, K), 4);
            V pc64 = e.zext64(pc);
            V base = rocke_b_global_ptr_offset(b, P.D, e.mul(pc64, e.zext64(slab)));
            V s_rsrc = rocke_b_buffer_rsrc(b, base, slab);
            V kq = e.muli(q, 4);
            for(int xt = 0; xt < XT; ++xt)
            {
                V xok, xp;
                opix(e.addi(xbase, xt * 16), &xok, &xp);
                for(int kt = 0; kt < KT; ++kt)
                {
                    V k = e.add(e.addi(kbase, kt * 16), kq);
                    V ok = e.land(xok, e.lt(k, K));
                    V xk = e.mul(xp, K);
                    V off = e.select(ok, e.muli(e.add(xk, k), 4), oob);
                    V v = accs[xt * KT + kt];
                    const rocke_type_t* vt = v->type;
                    V w = rocke_b_vec_bitcast(b, v, i32x4);
                    std::vector<V> ws;
                    for(int j = 0; j < 4; ++j)
                    {
                        ws.push_back(rocke_b_vec_extract(b, w, j));
                    }
                    std::vector<V> pw = permute(ws);
                    v = rocke_b_vec_bitcast(b, rocke_b_vec_pack(b, pw.data(), 4, i32_t), vt);
                    if(spec->k_tail)
                    {
                        /* fp32 element stores, each masked by K */
                        V xk2 = e.mul(xp, K);
                        V e0 = e.muli(e.add(xk2, k), 4);
                        for(int j = 0; j < 4; ++j)
                        {
                            V kj = e.lt(e.addi(k, j), K);
                            V okj = e.land(xok, kj);
                            V so = e.select(okj, e.addi(e0, 4 * j), oob);
                            rocke_b_buffer_store_f32(
                                b, s_rsrc, so, c0, rocke_b_vec_extract(b, v, j));
                        }
                    }
                    else
                    {
                        rocke_b_buffer_store_vN_f32(b, s_rsrc, off, c0, v, 4);
                    }
                    rocke_b_sched_barrier(b, 0);
                }
            }
        }
    };

    /* ---- the two groups ---- */
    zero = rocke_b_zero_vec_f32(b, 4);
    const int nacc = XT * KT;
    const rocke_type_t* vec4 = zero->type;

    /* group 1's DMAs out of its MFMA phase. S6, bk 256 plain: group 0 issues
     * all eight waves' weight rows, group 1 only its input rows; S5, the other
     * kernels except wt: group 1's input rows of step + 1 at the start of its
     * memory phase, its weight rows over the second half of its MFMA phase */
    const bool S6 = BK == 256 && BX == 256 && !WT && !DS && !FLAT;
    const bool S5 = !S6 && !WT;

    /* sw: the cursor's advance in the step of tap tau */
    auto adv_sw = [&](Dict& cur, int tau, bool last) {
        if(tau != NT - 2)
        {
            return cur;
        }
        if(!last)
        {
            Dict out = cur;
            out["ch"] = e.add(cur["ch"], c1);
            return out;
        }
        Dict nx = cursor_at(e.add(cur["i"], c1), false);
        Dict out;
        for(const std::string& k : names)
        {
            out[k] = nx[k];
        }
        return out;
    };

    /* One step reading stage st; sw = (tap, last chunk) when tau >= 0. */
    auto step = [&](std::vector<V> accs, Dict& cur, const St& st, bool g1, int tau, bool last) {
        auto prep = [&](Dict& c, const char* parts, int d) {
            if(tau < 0)
            {
                return dma_prep(c, parts, -1, true);
            }
            return dma_prep(c, parts, (tau + d) % NT, last || tau + d >= NT);
        };
        auto adv = [&](Dict& c) {
            if(tau < 0)
            {
                return cur_adv(c);
            }
            return adv_sw(c, tau, last);
        };
        const St ost = s_other(st);
        std::vector<std::vector<V>> a, bb;
        if(!g1)
        {
            /* the DMAs of step j + 1 between the fragment reads (one every 3
             * reads, S6: 2), the cursor advance after them */
            std::vector<DmaOp> ops = prep(cur, "xw", 1);
            if(S6)
            {
                /* group 1's weight rows too (wave + 4: channel + 128) */
                const size_t n0 = ops.size();
                for(size_t q = 0; q < n0; ++q)
                {
                    if(strcmp(ops[q].kind, "w") == 0)
                    {
                        V m = e.muli(RSC, 256);
                        ops.push_back(
                            DmaOp{"w4", ops[q].i, ops[q].rsrc, ops[q].voff, e.add(ops[q].soff, m)});
                    }
                }
            }
            /* the first DMA before the advance */
            dma_issue(std::vector<DmaOp>(ops.begin(), ops.begin() + 1), ost);
            Dict nc = adv(cur);
            cur = nc;
            size_t nxt_op = 1;
            const int every = S6 ? 2 : 3;
            std::function<void(int)> hook = [&](int i) {
                if(i % every == 0 && nxt_op < ops.size())
                {
                    dma_issue(std::vector<DmaOp>(1, ops[nxt_op]), ost);
                    ++nxt_op;
                }
            };
            frags(st, hook, a, bb);
            for(size_t q = nxt_op; q < ops.size(); ++q)
            {
                dma_issue(std::vector<DmaOp>(1, ops[q]), ost);
            }
            rocke_b_s_waitcnt(b, -1, 0, -1);
            rocke_b_sched_barrier(b, 0);
            bar(1);
            accs = mfmas(accs, a, bb, {}, 0);
            rocke_b_sched_barrier(b, 0);
            rocke_b_s_waitcnt(b, 0, -1, -1);
            bar(0);
            return accs;
        }
        /* group 1: the barrier that ends its MFMA phase opens its next memory
         * phase */
        bar(0);
        if(S6)
        {
            /* only this group's input rows (step + 1, the other stage) */
            dma_issue(prep(cur, "x", 1), ost);
            Dict nc = adv(cur);
            cur = nc;
            rocke_b_sched_barrier(b, 0);
            frags(st, nullptr, a, bb);
            rocke_b_s_waitcnt(b, -1, 0, -1);
            rocke_b_sched_barrier(b, 0);
            bar(1);
            /* the inputs land before the barrier that opens the next memory
             * phase: waited in the MFMA phase's last quarter */
            const int n_ = 2 * XT * KT;
            std::vector<std::function<void()>> inter = {[&]() { rocke_b_s_waitcnt(b, 0, -1, -1); }};
            accs = mfmas(accs, a, bb, inter, n_ - n_ / 4);
            rocke_b_sched_barrier(b, 0);
            return accs;
        }
        if(S5)
        {
            /* the input DMAs of step + 1 into the other stage at the start of
             * the memory phase; the weight DMAs of step + 2 in the MFMA phase */
            dma_issue(prep(cur, "x", 1), ost);
            Dict nc = adv(cur);
            cur = nc;
            rocke_b_sched_barrier(b, 0);
            frags(st, nullptr, a, bb);
            std::vector<DmaOp> wops = prep(cur, "w", 2);
            rocke_b_s_waitcnt(b, NXI, 0, -1);
            rocke_b_sched_barrier(b, 0);
            bar(1);
            std::vector<std::function<void()>> inter = {[&]() { rocke_b_s_waitcnt(b, 0, -1, -1); }};
            for(const DmaOp& op : wops)
            {
                inter.push_back([&, op]() { dma_issue(std::vector<DmaOp>(1, op), st); });
            }
            accs = mfmas(accs, a, bb, inter, (2 * XT * KT) / 2);
            rocke_b_sched_barrier(b, 0);
            return accs;
        }
        Dict nc = adv(cur);
        cur = nc;
        rocke_b_sched_barrier(b, 0);
        frags(st, nullptr, a, bb);
        std::vector<DmaOp> ops = prep(cur, "xw", 2);
        rocke_b_s_waitcnt(b, 0, 0, -1);
        rocke_b_sched_barrier(b, 0);
        bar(1);
        std::vector<std::function<void()>> inter;
        for(const DmaOp& op : ops)
        {
            inter.push_back([&, op]() { dma_issue(std::vector<DmaOp>(1, op), st); });
        }
        accs = mfmas(accs, a, bb, inter, 0);
        rocke_b_sched_barrier(b, 0);
        return accs;
    };

    /* the round loop carries the item's step count */
    const bool STEPS = !WT && !spec->c_tail && !SW;
    /* zero-first: an item's first step peeled, its MFMAs on C = 0 */
    const bool ZF = BK <= 128 && !SW;

    std::vector<const rocke_type_t*> res_types(nacc, vec4);
    res_types.insert(res_types.end(), names.size() + 1, i32_t);

    /* sw: the NT steps of one chunk, stage parity p0 at its first */
    auto chunk_sw = [&](std::vector<V>& accs, Dict& cur, V p0, bool g1, bool last) {
        for(int tau = 0; tau < NT; ++tau)
        {
            V sp = tau % 2 == 0 ? p0 : e.bxor(p0, c1);
            /* an opaque copy per step */
            sp = opaque(b, sp, "=s,0");
            accs = step(accs, cur, St{-1, sp}, g1, tau, last);
        }
        return NT % 2 ? e.bxor(p0, c1) : p0;
    };

    /* sw: an item's plain chunks (a loop), then its last chunk */
    auto item_sw = [&](Dict& cur, V st, V rnd, bool g1, std::vector<V>& accs_out, V* st_out) {
        V nci;
        if(SP == 1)
        {
            nci = nch;
        }
        else
        {
            V pc = md(tile_of(rnd), SP);
            V hi = dv(e.mul(e.add(pc, c1), nch), SP);
            nci = e.sub(hi, dv(e.mul(pc, nch), SP));
        }
        std::vector<V> accs(nacc, zero);
        std::vector<std::string> it_names = iter_names("sa", nacc);
        std::vector<V> inits = accs;
        for(const std::string& k : names)
        {
            it_names.push_back("sc_" + k);
            inits.push_back(cur[k]);
        }
        it_names.push_back("ssp");
        inits.push_back(st);
        std::vector<V> res = e.for_iter(
            c0, e.sub(nci, c1), c1, it_names, inits, "chunk", [&](V, std::vector<V> lc) {
                std::vector<V> la(lc.begin(), lc.begin() + nacc);
                Dict lcur = zip_names(lc, nacc);
                V lp = chunk_sw(la, lcur, lc.back(), g1, false);
                std::vector<V> out = la;
                for(const std::string& k : names)
                {
                    out.push_back(lcur[k]);
                }
                out.push_back(lp);
                return out;
            });
        if(res.size() != it_names.size())
        {
            return;
        }
        accs_out.assign(res.begin(), res.begin() + nacc);
        cur = zip_names(res, nacc);
        *st_out = chunk_sw(accs_out, cur, res.back(), g1, true);
    };

    /* All steps of an item that starts on stage p: pairs, then an odd tail
     * (ZF: the first step peeled, its MFMAs on C = 0). */
    auto item_steps
        = [&](std::vector<V> accs, Dict cur, V steps, int pp, bool g1, const std::string& tag) {
              if(ZF)
              {
                  accs = step(std::vector<V>(nacc, (V)NULL), cur, St{pp, NULL}, g1, -1, false);
                  steps = e.sub(steps, c1);
                  pp = 1 - pp;
              }
              std::vector<std::string> it_names;
              std::vector<V> inits;
              for(int i = 0; i < nacc; ++i)
              {
                  it_names.push_back(tag + "a" + std::to_string(i));
                  inits.push_back(accs[i]);
              }
              for(const std::string& k : names)
              {
                  it_names.push_back(tag + "c_" + k);
                  inits.push_back(cur[k]);
              }
              std::string iv = tag + "pair";
              std::vector<V> res = e.for_iter(
                  c0, e.div(steps, c2), c1, it_names, inits, iv.c_str(), [&](V, std::vector<V> lc) {
                      std::vector<V> la(lc.begin(), lc.begin() + nacc);
                      Dict lcur = zip_names(lc, nacc);
                      la = step(la, lcur, St{pp, NULL}, g1, -1, false);
                      la = step(la, lcur, St{1 - pp, NULL}, g1, -1, false);
                      for(const std::string& k : names)
                      {
                          la.push_back(lcur[k]);
                      }
                      return la;
                  });
              if(res.size() != it_names.size())
              {
                  return std::vector<V>();
              }
              std::vector<V> ra(res.begin(), res.begin() + nacc);
              Dict rc = zip_names(res, nacc);
              V odd = e.eq(e.mod(steps, c2), c1);
              return if_else(
                  b,
                  odd,
                  res_types,
                  [&]() {
                      Dict tcur = rc;
                      std::vector<V> ta = step(ra, tcur, St{pp, NULL}, g1, -1, false);
                      for(const std::string& k : names)
                      {
                          ta.push_back(tcur[k]);
                      }
                      ta.push_back(ci(1 - pp));
                      return ta;
                  },
                  [&]() {
                      std::vector<V> out = ra;
                      for(const std::string& k : names)
                      {
                          out.push_back(rc[k]);
                      }
                      out.push_back(ci(pp));
                      return out;
                  });
          };

    auto run = [&](bool g1) {
        t_g1 = g1;
        t_rnd = c0;
        /* per group, from an opaque lane (not formed above the group branch
         * and held across it) */
        V lane_ = lane;
        lane = opaque_lane();
        lane_bases();
        lane = lane_;
        Dict cur = cursor_at(c0, true);
        if(SW)
        {
            dma_issue(dma_prep(cur, "xw", 0, true), St{0, NULL});
        }
        else
        {
            dma_issue(dma_prep(cur, "xw", -1, true), St{0, NULL});
        }
        if(TB2 && !g1)
        {
            /* group 0's rows of item 1 while it waits for group 1 at the
             * first barrier */
            e.if_(e.lt(c1, rounds), [&]() {
                V ln1 = opaque_lane();
                Dict inf = item_info(tile_of(c1));
                V l8 = dv(ln1, 8);
                build_tab(inf, ln1, l8, NULL, c1);
            });
        }
        if(!SW)
        {
            cur = cur_adv(cur);
        }
        const int st1 = SW ? 1 : -1;
        if(g1 && !S6)
        {
            /* group 1's DMAs of step 1 before the first barrier */
            if(S5)
            {
                dma_issue(dma_prep(cur, "w", st1, true), St{1, NULL});
            }
            else
            {
                dma_issue(dma_prep(cur, "xw", st1, true), St{1, NULL});
            }
            const int n1 = S5 ? NWI : NXI + NWI;
            if(DS || TB2)
            {
                rocke_b_s_waitcnt(b, n1, 0, -1);
            }
            else
            {
                rocke_b_s_waitcnt(b, n1, -1, -1);
            }
            rocke_b_s_barrier_bare(b);
        }
        else
        {
            if(DS || TB2)
            {
                rocke_b_s_waitcnt(b, 0, 0, -1);
            }
            else
            {
                rocke_b_s_waitcnt(b, 0, -1, -1);
            }
            rocke_b_s_barrier_bare(b);
        }
        rocke_b_sched_barrier(b, 0);

        auto steps_of = [&](V i) {
            Dict info = item_info(tile_of(i));
            V dr = e.sub(info["r1"], info["r0"]);
            V nrs = e.mul(dr, e.sub(info["s1"], info["s0"]));
            return e.mul(nrs, e.sub(info["ch1"], info["ch0"]));
        };

        std::vector<std::string> o_names;
        std::vector<V> o_inits;
        for(const std::string& k : names)
        {
            o_names.push_back("o_" + k);
            o_inits.push_back(cur[k]);
        }
        o_names.push_back("o_st");
        o_inits.push_back(c0);
        if(STEPS)
        {
            o_names.push_back("o_steps");
            o_inits.push_back(steps_of(c0));
        }
        e.for_iter(c0, rounds, c1, o_names, o_inits, "rnd", [&](V rnd, std::vector<V> oc) {
            Dict lc = zip_names(oc, 0);
            V st = oc[names.size()];
            V t = tile_of(rnd);
            V steps = NULL;
            if(SW)
            {
                steps = NULL;
            }
            else if(STEPS)
            {
                steps = oc[names.size() + 1];
            }
            else
            {
                steps = steps_of(rnd);
            }
            std::vector<V> accs(nacc, zero);
            t_rnd = rnd;
            if(SW)
            {
                std::vector<V> ra;
                V nst = NULL;
                item_sw(lc, st, rnd, g1, ra, &nst);
                accs = ra;
                st = nst;
            }
            else
            {
                std::vector<V> res = if_else(
                    b,
                    e.eq(st, c0),
                    res_types,
                    [&]() { return item_steps(accs, lc, steps, 0, g1, "p0"); },
                    [&]() { return item_steps(accs, lc, steps, 1, g1, "p1"); });
                if(res.size() != res_types.size())
                {
                    return std::vector<V>();
                }
                accs.assign(res.begin(), res.begin() + nacc);
                lc = zip_names(res, nacc);
                st = res.back();
            }
            /* the next item's step count before the epilogue */
            V nxt_steps = STEPS ? steps_of(e.add(rnd, c1)) : NULL;
            epilogue(accs, t, rnd);
            rocke_b_sched_barrier(b, 0);
            if(TB2 && g1)
            {
                /* item rnd + 2's rows while this group idles, unless the
                 * cursor already crossed into it */
                V k2 = e.add(rnd, c2);
                V r1 = e.add(rnd, c1);
                V cr = e.eq(lc["i"], r1);
                V cond = e.land(cr, e.lt(k2, rounds));
                e.if_(cond, [&]() {
                    Dict inf = item_info(tile_of(k2));
                    build_tab_all(inf, k2);
                    rocke_b_s_waitcnt(b, -1, 0, -1);
                });
                rocke_b_sched_barrier(b, 0);
            }
            std::vector<V> out = of_names(lc);
            out.push_back(st);
            if(STEPS)
            {
                out.push_back(nxt_steps);
            }
            return out;
        });
        rocke_b_s_setprio(b, 0);
        rocke_b_s_barrier_bare(b); /* balance the barrier count */
    };

    auto groups
        = [&]() { if_else_void(b, e.eq(g, c0), [&]() { run(false); }, [&]() { run(true); }); };

    if(!spec->zl)
    {
        groups();
        return rocke_ir_builder_ok(b) ? b->kernel : NULL;
    }

    /* ---- zl: light zero units after the items ---- */
    /* blocks past G (the items alone fill fewer CUs) run no item */
    e.if_(e.lt(blk, G), [&]() { groups(); });
    V ZU = e.cdivi(NXP, kZlPx);
    V big = e.gei(TI, kPersist);
    V GL = e.smini(e.add(TI, ZU), kPersist);
    V xtra = e.sub(GL, G); /* 0 when big */
    /* light blocks (one item fewer) */
    V pg = rocke_b_cmp_gt(b, part, c0);
    V uneven = e.select(big, pg, rocke_b_cmp_gt(b, xtra, c0));
    V lb = e.select(uneven, e.sub(G, part), G);
    V L = e.select(big, lb, e.select(uneven, xtra, G));
    V odd = e.eq(e.mod(full, c2), c1);
    V o_big = e.select(odd, blk, e.mod(e.add(blk, L), G));
    V bg = e.ge(blk, G);
    V bs = e.sub(blk, G);
    V o_sm = e.select(bg, bs, e.add(blk, xtra));
    V o = e.select(big, o_big, o_sm);
    V NB = e.select(big, G, GL);
    V E = e.select(uneven, ci(spec->zl), c0);
    V few = rocke_b_cmp_le(b, ZU, e.mul(L, E)); /* the light blocks' slack takes them all */
    V RZ = e.sub(ZU, e.mul(L, E));

    auto first_unit = [&](V o_) {
        V ol = e.smin(o_, L);
        V a = e.div(e.mul(ol, ZU), L);
        V oe = e.mul(ol, E);
        V bb = e.add(oe, e.div(e.mul(o_, RZ), NB));
        return e.select(few, a, bb);
    };
    V u0 = first_unit(o);
    V u1 = first_unit(e.add(o, c1));
    const int a0 = PORD[0].first, b0 = PORD[0].second;
    V ln = opaque_lane();
    V pp = dv(ln, 8);
    V jg = md(ln, 8);
    V nkc = e.cdivi(K, 64);
    V zs[4] = {c0, c0, c0, c0};
    V zv = rocke_b_vec_bitcast(b, rocke_b_vec_pack(b, zs, 4, i32_t), half8);
    rocke_for_t fu = rocke_b_scf_for(b, u0, u1, c1, "zu");
    if(fu.body == NULL)
    {
        return NULL;
    }
    rocke_b_region_enter(b, fu.body);
    {
        V u = fu.iv;
        V uz = e.muli(u, kZlPx);
        V xl = e.add(uz, e.add(e.muli(wave, 8), pp));
        V ok = e.lt(xl, NXP);
        V n = e.div(xl, hwq);
        V rem = e.sub(xl, e.mul(n, hwq));
        V qi = e.div(rem, Wq);
        V qj = e.sub(rem, e.mul(qi, Wq));
        V oh = e.addi(e.muli(qi, SDH), a0);
        V ow = e.addi(e.muli(qj, SDW), b0);
        V no = e.mul(n, p_Ho);
        V op = e.add(e.mul(e.add(no, oh), p_Wo), ow);
        std::vector<std::pair<V, V>> zb;
        for(const auto& d : ZD)
        {
            V zh = e.lt(e.addi(oh, d.first), p_Ho);
            V zw = e.lt(e.addi(ow, d.second), p_Wo);
            V zok = e.land(ok, e.land(zh, zw));
            V da = ci(d.first);
            V dm = e.mul(da, p_Wo);
            V zp = e.add(op, e.addi(dm, d.second));
            V zk = e.mul(e.mul(zp, K), c2);
            V zo = e.add(zk, e.muli(jg, 16));
            zb.push_back(std::make_pair(zok, zo));
        }
        rocke_for_t fk = rocke_b_scf_for(b, c0, nkc, c1, "zk");
        if(fk.body != NULL)
        {
            rocke_b_region_enter(b, fk.body);
            V kc = fk.iv;
            V k64 = e.muli(kc, 64);
            V kok = e.lt(e.add(k64, e.muli(jg, 8)), K);
            V ko = e.muli(kc, 128);
            for(const auto& z : zb)
            {
                V zk = e.land(z.first, kok);
                V off = e.select(zk, e.add(z.second, ko), oob);
                rocke_dconv_buf_store_vN(b, bf16, d_rsrc, off, c0, zv, 4);
            }
            rocke_b_region_leave(b);
        }
    }
    rocke_b_region_leave(b);
    return rocke_ir_builder_ok(b) ? b->kernel : NULL;
}

} // namespace

extern "C" rocke_kernel_def_t* rocke_build_fwd_flex_mfma(rocke_ir_builder_t* b,
                                                         const rocke_fwd_flex_mfma_spec_t* spec,
                                                         const char* arch)
{
    if(b == NULL || spec == NULL)
    {
        return NULL;
    }
    if(arch == NULL)
    {
        arch = "gfx950";
    }
    char reason[ROCKE_ERR_MSG_CAP];
    if(!rocke_fwd_flex_mfma_is_valid_spec(spec, arch, reason, sizeof reason))
    {
        std::string prefix = std::string("invalid FwdFlexMfmaSpec for ") + arch;
        fail_invalid(b, prefix.c_str(), reason);
        return NULL;
    }
    if(spec->cs == 4)
    {
        return build_cs4(b, spec, arch);
    }
    if(spec->cs == 2)
    {
        return build_cs2(b, spec, arch);
    }
    return build_main(b, spec, arch);
}

extern "C" rocke_kernel_def_t* rocke_build_fwd_flex_mfma_new(rocke_ir_builder_t* b,
                                                             const rocke_fwd_flex_mfma_spec_t* spec,
                                                             const char* arch)
{
    return build_new(
        b,
        spec != NULL,
        "build_fwd_flex_mfma",
        [&](char* out, size_t cap) { return rocke_fwd_flex_mfma_kernel_name(spec, out, cap); },
        [&]() { return rocke_build_fwd_flex_mfma(b, spec, arch); });
}

extern "C" rocke_status_t rocke_fwd_flex_mfma_lower_to_llvm(const rocke_fwd_flex_mfma_spec_t* spec,
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
        "build_fwd_flex_mfma failed",
        [&](rocke_ir_builder_t* b) { return rocke_build_fwd_flex_mfma_new(b, spec, arch); });
}
