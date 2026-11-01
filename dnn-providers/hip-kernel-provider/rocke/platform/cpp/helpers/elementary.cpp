// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * elementary.cpp -- C++ mirror of rocke/helpers/elementary.py (the IR side:
 * IRMath, _RecordingMath, the programs, family_constants, opaque_zero, emit).
 * See rocke/elementary.h.
 *
 * Order of builder calls: Python evaluates call arguments left to right, and
 * an IRMath op emits its constants (f / i) right before the op. The programs
 * below pass every argument list as a braced list (A2 / A3), whose elements
 * C++ evaluates left to right, so a nested expression makes its builder calls
 * in exactly the order of the Python expression it mirrors. Python evaluates
 * every argument even when the result is not used (sel with a build-time
 * predicate): so do these programs.
 */
#include "rocke/elementary.h"

#include "rocke/error_boundary.hpp"
#include "rocke/ir_internal.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#include <limits>
#include <map>
#include <mutex>
#include <string>

namespace rocke
{
namespace elementary
{
namespace
{

/* ------------------------------------------------------------- constants */

constexpr double TWO_OVER_PI = 0x1.45f306dc9c883p-1;
constexpr double PIO2_0 = 0x1.921fb54442d18p+0;
constexpr double PIO2_1 = 0x1.1a62633145c07p-54;
constexpr double PIO2_2 = -0x1.f1976b7ed8fbcp-110;
constexpr double PI_H = 0x1.921fb54442d18p+1;
constexpr double PI_L = 0x1.1a62633145c07p-53;
constexpr double LN2_42 = 0x1.62e42fefa38p-1;
constexpr double LN2_42_1 = 0x1.ef35793c7673p-45;
constexpr double LN2_42_2 = 0x1.f97b57a079a19p-103;
constexpr double LOG2E = 0x1.71547652b82fep+0;
constexpr double SQRT2 = 0x1.6a09e667f3bcdp+0;
constexpr double ATAN_HALF[2] = {0x1.dac670561bb4fp-2, 0x1.a2b7f222f65e2p-56};
constexpr double ATAN_ONE[2] = {0x1.921fb54442d18p-1, 0x1.1a62633145c07p-55};
constexpr double ATAN_3HALF[2] = {0x1.f730bd281f69bp-1, 0x1.007887af0cbbdp-56};
constexpr double ATAN_INF[2] = {PIO2_0, PIO2_1};

constexpr double TRIG_LARGE = 0x1p30;
constexpr double M52 = 1.5 * 0x1p52;
constexpr int64_t SIGN = std::numeric_limits<int64_t>::min(); /* -(1 << 63) */
constexpr int64_t ABS = std::numeric_limits<int64_t>::max(); /* (1 << 63) - 1 */
constexpr int64_t EXACT = (int64_t)1 << 53;
constexpr int64_t MANT_MASK = (int64_t)0x000FFFFFFFFFFFFF;
constexpr int64_t ONE_BITS = (int64_t)0x3FF0000000000000;
constexpr double INF = std::numeric_limits<double>::infinity();
constexpr double NAN_ = std::numeric_limits<double>::quiet_NaN();

const std::vector<double> SIN_S = {
    -0x1.5555555555555p-3,
    0x1.1111111111110p-7,
    -0x1.a01a01a019929p-13,
    0x1.71de3a545f037p-19,
    -0x1.ae6454104d5d8p-26,
    0x1.61217d41bb89bp-33,
    -0x1.ab16db5288252p-41,
};
const std::vector<double> COS_C = {
    0x1.5555555555555p-5,
    -0x1.6c16c16c16c16p-10,
    0x1.a01a01a019d06p-16,
    -0x1.27e4fb77125e8p-22,
    0x1.1eed8deadabefp-29,
    -0x1.9394b8fa4ece7p-37,
    0x1.ab77c5650a425p-45,
};
const std::vector<double> ATAN_A = {
    -0x1.5555555555555p-2,
    0x1.99999999998cap-3,
    -0x1.24924924858e8p-3,
    0x1.c71c71bd52eeap-4,
    -0x1.745d1552e8d71p-4,
    0x1.3b1376694c06dp-4,
    -0x1.110ca60ef6421p-4,
    0x1.e17131ed68ef0p-5,
    -0x1.ab5e0ae0eca25p-5,
    0x1.701255ab35430p-5,
    -0x1.0f87a102a636fp-5,
    0x1.e4a235b9c4a59p-7,
};
const std::vector<double> ASIN_R = {
    0x1.5555555555555p-3,
    0x1.333333333338dp-4,
    0x1.6db6db6dab864p-5,
    0x1.f1c71c7abbd6fp-6,
    0x1.6e8ba110ffef0p-6,
    0x1.1c4efee7125cap-6,
    0x1.c9906663d6d13p-7,
    0x1.7b05bf281d434p-7,
    0x1.3b2fc027779e6p-7,
    0x1.31f2160e1e4f6p-7,
    0x1.86bc5d25f4c92p-9,
    0x1.42e3180325070p-6,
    -0x1.3ec09802d3b38p-6,
    0x1.e752255431ce2p-6,
};
const std::vector<double> SINH_H = {
    0x1.5555555555555p-3,
    0x1.1111111111111p-7,
    0x1.a01a01a01a001p-13,
    0x1.71de3a556f8e6p-19,
    0x1.ae64567936642p-26,
    0x1.612467c3f11f8p-33,
    0x1.ae7782967fd2ep-41,
    0x1.99eeaddf33ce1p-49,
};
const std::vector<double> EXP_P = {
    0x1.5555555555555p-3,
    0x1.5555555555556p-5,
    0x1.1111111111110p-7,
    0x1.6c16c16c162dap-10,
    0x1.a01a01a01b3d5p-13,
    0x1.a01a01a6d56aap-16,
    0x1.71de3a5007d0dp-19,
    0x1.27e4db6d6a969p-22,
    0x1.ae645f12243e9p-26,
    0x1.1f72f2050aa81p-29,
    0x1.616b75e800303p-33,
};
const std::vector<double> LOG_G = {
    0x1.5555555555555p-1,
    0x1.9999999999a38p-2,
    0x1.2492492476b5cp-2,
    0x1.c71c72017e746p-3,
    0x1.745cf8f61afcep-3,
    0x1.3b1c38386f2d9p-3,
    0x1.0fbe4dac856a8p-3,
    0x1.0c0665b09f2b8p-3,
};

const char* const k_functions[] = {"sin",
                                   "cos",
                                   "tan",
                                   "sinh",
                                   "cosh",
                                   "asin",
                                   "acos",
                                   "atan",
                                   "asinh",
                                   "acosh",
                                   "atanh",
                                   "floor",
                                   "ceil"};

struct family_t
{
    const char* name;
    int n;
    const char* members[3];
};

const family_t k_families[] = {
    {"trig", 3, {"sin", "cos", "tan"}},
    {"hyp", 2, {"sinh", "cosh", nullptr}},
    {"asin_acos", 2, {"asin", "acos", nullptr}},
    {"atan", 1, {"atan", nullptr, nullptr}},
    {"ainvhyp", 3, {"asinh", "acosh", "atanh"}},
    {"round", 2, {"floor", "ceil", nullptr}},
};

const family_t* find_family(const char* fam)
{
    for(const family_t& f : k_families)
        if(fam != nullptr && strcmp(f.name, fam) == 0)
            return &f;
    return nullptr;
}

int64_t bits_of(double x)
{
    int64_t n;
    memcpy(&n, &x, sizeof n);
    return n;
}

bool is_i64(const MV& v)
{
    return v.k == MV::VAL && v.v != nullptr && rocke_type_eq(v.v->type, rocke_i64());
}

bool is_num(const MV& v)
{
    /* Python isinstance(v, (float, int)): bool is an int */
    return v.k != MV::VAL;
}

double as_float(const MV& v)
{
    return v.k == MV::FLT ? v.f : v.k == MV::INT ? (double)v.i : (v.bv ? 1.0 : 0.0);
}

} // namespace

/* =============================================================== Math */

rocke_value_t* Math::val(const MV& v)
{
    if(v.k == MV::VAL)
        return v.v;
    rocke_i_set_err(b_, ROCKE_ERR_VALUE, "elementary: a Python number where an IR value is needed");
    return nullptr;
}

MV Math::f(const MV& v)
{
    if(!is_num(v))
        return v;
    double x = as_float(v);
    if(recording)
    {
        int64_t key = bits_of(x);
        if(uses.find(key) == uses.end())
            seen.push_back(x);
        uses[key] += 1;
    }
    if(anchor_ == nullptr)
        return rocke_b_const_f64(b_, x);
    int64_t key = bits_of(x);
    auto it = pool_.find(key);
    if(it != pool_.end())
        return it->second;
    /* built at its use (a value defined in one branch must not be used in another) */
    rocke_value_t* k = k64(key);
    rocke_value_t* o = rocke_b_lor(b_, k, anchor_);
    return rocke_b_bitcast(b_, o, rocke_f64());
}

void Math::materialize(const std::vector<double>& constants)
{
    for(double c : constants)
    {
        int64_t key = bits_of(c);
        if(pool_.find(key) == pool_.end())
        {
            rocke_value_t* k = k64(key);
            rocke_value_t* o = rocke_b_lor(b_, k, anchor_);
            pool_[key] = rocke_b_bitcast(b_, o, rocke_f64());
        }
    }
}

rocke_value_t* Math::k64(int64_t v)
{
    if(-EXACT <= v && v <= EXACT)
        return rocke_b_const_i64(b_, v);
    uint64_t u = (uint64_t)v;
    int64_t hi = (int64_t)(u >> 32);
    int64_t lo = (int64_t)(u & 0xFFFFFFFFu);
    auto half = [&](int64_t h) {
        rocke_value_t* c
            = rocke_b_const_i32(b_, h >= ((int64_t)1 << 31) ? h - ((int64_t)1 << 32) : h);
        return rocke_b_zext(b_, c, rocke_i64());
    };
    rocke_value_t* h = half(hi);
    rocke_value_t* s = rocke_b_const_i64(b_, 32);
    rocke_value_t* sh = rocke_b_shl(b_, h, s);
    rocke_value_t* l = half(lo);
    return rocke_b_lor(b_, sh, l);
}

MV Math::i(const MV& v)
{
    if(v.k == MV::INT)
        return k64(v.i);
    if(v.k == MV::BOOL)
        return k64(v.bv ? 1 : 0);
    return v;
}

MV Math::add(const A2& x)
{
    MV a = f(x.a);
    MV c = f(x.c);
    return rocke_b_fadd(b_, val(a), val(c));
}

MV Math::sub(const A2& x)
{
    MV a = f(x.a);
    MV c = f(x.c);
    return rocke_b_fsub(b_, val(a), val(c));
}

MV Math::mul(const A2& x)
{
    MV a = f(x.a);
    MV c = f(x.c);
    return rocke_b_fmul(b_, val(a), val(c));
}

MV Math::div(const A2& x)
{
    MV a = f(x.a);
    MV c = f(x.c);
    return rocke_b_fdiv(b_, val(a), val(c));
}

MV Math::fma(const A3& x)
{
    MV a = f(x.a);
    MV c = f(x.c);
    MV d = f(x.d);
    return rocke_b_fma(b_, val(a), val(c), val(d));
}

MV Math::sqrt(const MV& a)
{
    MV v = f(a);
    return rocke_b_sqrt(b_, val(v));
}

MV Math::neg(const MV& a)
{
    MV v = f(a);
    return rocke_b_fneg(b_, val(v));
}

MV Math::abs(const MV& a)
{
    MV v = f(a);
    return rocke_b_fabs(b_, val(v));
}

MV Math::cmp(const char* pred, const A2& x)
{
    MV a = f(x.a);
    MV c = f(x.c);
    return rocke_b_fcmp(b_, pred, val(a), val(c));
}

MV Math::lt(const A2& x)
{
    return cmp("olt", x);
}
MV Math::le(const A2& x)
{
    return cmp("ole", x);
}
MV Math::gt(const A2& x)
{
    return cmp("ogt", x);
}
MV Math::ge(const A2& x)
{
    return cmp("oge", x);
}
MV Math::eq(const A2& x)
{
    return cmp("oeq", x);
}
MV Math::isnan(const MV& a)
{
    return cmp("uno", {a, a});
}

MV Math::band(const A2& x)
{
    if(x.a.is_bool())
        return x.a.bv ? x.c : MV(false);
    if(x.c.is_bool())
        return x.c.bv ? x.a : MV(false);
    return rocke_b_land(b_, val(x.a), val(x.c));
}

MV Math::bor(const A2& x)
{
    if(x.a.is_bool())
        return x.a.bv ? MV(true) : x.c;
    if(x.c.is_bool())
        return x.c.bv ? MV(true) : x.a;
    return rocke_b_lor(b_, val(x.a), val(x.c));
}

MV Math::bnot(const MV& p)
{
    if(p.is_bool())
        return MV(!p.bv);
    return rocke_b_lnot(b_, val(p));
}

MV Math::sel(const A3& x)
{
    const MV& p = x.a;
    const MV& a = x.c;
    const MV& c = x.d;
    if(p.is_bool())
        return p.bv ? a : c;
    if(is_num(a) && is_num(c))
    {
        /* Python isinstance(a, int) and isinstance(c, int) (bool is an int) */
        if(a.k != MV::FLT && c.k != MV::FLT)
        {
            MV ia = i(a);
            MV ic = i(c);
            return rocke_b_select(b_, val(p), val(ia), val(ic));
        }
        MV fa = f(a);
        MV fc = f(c);
        return rocke_b_select(b_, val(p), val(fa), val(fc));
    }
    if(is_i64(a) || is_i64(c))
    {
        MV ia = i(a);
        MV ic = i(c);
        return rocke_b_select(b_, val(p), val(ia), val(ic));
    }
    MV fa = f(a);
    MV fc = f(c);
    return rocke_b_select(b_, val(p), val(fa), val(fc));
}

MV Math::bits(const MV& a)
{
    MV v = f(a);
    return rocke_b_bitcast(b_, val(v), rocke_i64());
}

MV Math::from_bits(const MV& n)
{
    MV v = i(n);
    return rocke_b_bitcast(b_, val(v), rocke_f64());
}

MV Math::iand(const A2& x)
{
    MV n = i(x.a);
    MV m = i(x.c);
    return rocke_b_land(b_, val(n), val(m));
}

MV Math::ior(const A2& x)
{
    MV n = i(x.a);
    MV m = i(x.c);
    return rocke_b_lor(b_, val(n), val(m));
}

MV Math::ixor(const A2& x)
{
    MV n = i(x.a);
    MV m = i(x.c);
    return rocke_b_xor(b_, val(n), val(m));
}

MV Math::ishl(const MV& n, int64_t s)
{
    MV v = i(n);
    rocke_value_t* c = rocke_b_const_i64(b_, s);
    return rocke_b_shl(b_, val(v), c);
}

MV Math::ilshr(const MV& n, int64_t s)
{
    MV v = i(n);
    rocke_value_t* c = rocke_b_const_i64(b_, s);
    return rocke_b_lshr(b_, val(v), c);
}

MV Math::iadd(const A2& x)
{
    MV n = i(x.a);
    MV m = i(x.c);
    return rocke_b_add(b_, val(n), val(m));
}

MV Math::isub(const A2& x)
{
    MV n = i(x.a);
    MV m = i(x.c);
    return rocke_b_sub(b_, val(n), val(m));
}

MV Math::ieq(const A2& x)
{
    MV n = i(x.a);
    MV m = i(x.c);
    return rocke_b_cmp_eq(b_, val(n), val(m));
}

MV Math::trig_preop(const MV& x, int seg)
{
    MV v = f(x);
    rocke_value_t* ops[2];
    ops[0] = val(v);
    ops[1] = rocke_b_const_i32(b_, seg);
    const rocke_type_t* rt[1] = {rocke_f64()};
    rocke_inline_asm_opts_t o;
    memset(&o, 0, sizeof o);
    o.sideeffect = false;
    o.sideeffect_set = true;
    rocke_op_t* op = rocke_b_inline_asm_named(
        b_, "v_trig_preop_f64 $0, $1, $2", "=v,v,v", ops, 2, rt, 1, &o, "preop");
    return op != nullptr ? op->results[0] : nullptr;
}

std::vector<rocke_value_t*> Math::when_inputs(const std::vector<MV>& vals)
{
    /* Python: [self.f(v) if isinstance(v, float) else self.i(v) for v in vals] */
    std::vector<rocke_value_t*> out;
    for(const MV& v : vals)
    {
        MV c = v.k == MV::FLT ? f(v) : i(v);
        out.push_back(val(c));
    }
    return out;
}

rocke_for_t Math::when_open(const MV& cond, const std::vector<rocke_value_t*>& init)
{
    rocke_value_t* one = rocke_b_const_i32(b_, 1);
    rocke_value_t* zero = rocke_b_const_i32(b_, 0);
    rocke_value_t* trips = rocke_b_select(b_, val(cond), one, zero);
    const char* fresh = rocke_b_fresh(b_, "el");
    std::string tag = fresh != nullptr ? std::string(fresh + 1) : std::string("el");
    std::vector<std::string> names;
    for(size_t j = 0; j < init.size(); ++j)
        names.push_back(tag + "_" + std::to_string(j));
    std::vector<rocke_iter_arg_t> args(init.size());
    for(size_t j = 0; j < init.size(); ++j)
    {
        args[j].name = names[j].c_str();
        args[j].init = init[j];
    }
    std::string iv = tag + "_i";
    rocke_value_t* lo = rocke_b_const_i32(b_, 0);
    rocke_value_t* st = rocke_b_const_i32(b_, 1);
    rocke_for_t loop = rocke_b_scf_for_iter(
        b_, lo, trips, st, args.data(), (int)args.size(), iv.c_str(), false, true);
    rocke_b_region_enter(b_, loop.body);
    return loop;
}

std::vector<MV> Math::when_close(const rocke_for_t& loop, const std::vector<MV>& out)
{
    std::vector<rocke_value_t*> ys;
    for(const MV& v : out)
        ys.push_back(val(v));
    rocke_b_scf_yield(b_, ys.data(), (int)ys.size());
    rocke_b_region_leave(b_);
    std::vector<MV> res;
    if(loop.op != nullptr)
        for(int j = 0; j < loop.op->num_results; ++j)
            res.push_back(MV(loop.op->results[j]));
    return res;
}

/* ================================================= error-free pieces */

namespace
{

using P2 = std::pair<MV, MV>;

P2 two_sum(Math& m, const MV& a, const MV& c)
{
    MV s = m.add({a, c});
    MV bb = m.sub({s, a});
    MV lo = m.add({m.sub({a, m.sub({s, bb})}), m.sub({c, bb})});
    return {s, lo};
}

/* |a| >= |c| (or a zero) */
P2 fast_two_sum(Math& m, const MV& a, const MV& c)
{
    MV s = m.add({a, c});
    MV lo = m.sub({c, m.sub({s, a})});
    return {s, lo};
}

P2 two_prod(Math& m, const MV& a, const MV& c)
{
    MV p = m.mul({a, c});
    MV e = m.fma({a, c, m.neg(p)});
    return {p, e};
}

MV horner(Math& m, const std::vector<double>& coefs, const MV& z)
{
    MV acc = m.f(coefs.back());
    for(int k = (int)coefs.size() - 2; k >= 0; --k)
        acc = m.fma({acc, z, coefs[k]});
    return acc;
}

/* the integer nearest v (ties to even), |v| < 2^51 */
MV rint_small(Math& m, const MV& v)
{
    return m.sub({m.add({v, M52}), M52});
}

/* an integer-valued f64 |k| < 2^51 as an i64 */
MV int_bits(Math& m, const MV& k)
{
    return m.isub({m.bits(m.add({k, M52})), bits_of(M52)});
}

/* an i64 |n| < 2^51 as an f64 */
MV int_to_f64(Math& m, const MV& n)
{
    return m.sub({m.from_bits(m.iadd({n, bits_of(M52)})), M52});
}

/* 2^n for an i64 n in [-1022, 1023] */
MV pow2(Math& m, const MV& n)
{
    return m.from_bits(m.ishl(m.iadd({n, 1023}), 52));
}

MV copysign(Math& m, const MV& v, const MV& x)
{
    return m.from_bits(m.ior({m.iand({m.bits(v), ABS}), m.iand({m.bits(x), SIGN})}));
}

/* floor(v) for v >= 0 (v itself from 2^52) */
MV floor_nonneg(Math& m, const MV& v)
{
    MV t = m.sub({m.add({v, 0x1p52}), 0x1p52});
    t = m.sub({t, m.sel({m.gt({t, v}), 1.0, 0.0})});
    return m.sel({m.lt({v, 0x1p52}), t, v});
}

/* x + x z Q(z) as a dd for z = zh + zl */
P2 series_dd(Math& m, const MV& x, const MV& zh, const MV& zl, const std::vector<double>& coefs)
{
    P2 c = two_prod(m, x, zh);
    MV ch = c.first;
    MV cl = m.fma({x, zl, c.second});
    P2 t = two_prod(m, ch, coefs[0]);
    MV th = t.first;
    MV rest = m.fma({cl, coefs[0], t.second});
    if(coefs.size() > 1)
    {
        std::vector<double> tail(coefs.begin() + 1, coefs.end());
        MV q1 = horner(m, tail, zh);
        rest = m.fma({m.mul({ch, zh}), q1, rest});
    }
    P2 s = two_sum(m, x, th);
    return fast_two_sum(m, s.first, m.add({s.second, rest}));
}

/* the sign bit of x (true for -0 and negative NaNs) */
MV is_negative(Math& m, const MV& x)
{
    return m.bnot(m.ieq({m.iand({m.bits(x), SIGN}), 0}));
}

/* ------------------------------------------------------------ floor / ceil */

MV floor_ceil(Math& m, const MV& x, const MV& is_ceil)
{
    MV a = m.abs(x);
    MV t = m.sub({m.add({a, 0x1p52}), 0x1p52});
    MV fl = m.sub({t, m.sel({m.gt({t, a}), 1.0, 0.0})});
    MV ce = m.add({t, m.sel({m.lt({t, a}), 1.0, 0.0})});
    MV negative = is_negative(m, x);
    MV up = m.sel({is_ceil, m.bnot(negative), negative});
    MV r = copysign(m, m.sel({up, ce, fl}), x);
    return m.sel({m.lt({a, 0x1p52}), r, x});
}

/* ---------------------------------------------------------- sin / cos / tan */

struct Red
{
    MV rh, rl, q;
};

Red reduce_medium(Math& m, const MV& a)
{
    MV k = rint_small(m, m.mul({a, TWO_OVER_PI}));
    MV xt = m.fma({m.neg(k), PIO2_0, a});
    P2 p = two_prod(m, k, PIO2_1);
    P2 se = two_sum(m, xt, m.neg(p.first));
    MV t = m.fma({m.neg(k), PIO2_2, m.sub({se.second, p.second})});
    P2 r = two_sum(m, se.first, t);
    MV q = m.iand({int_bits(m, k), 3});
    return {r.first, r.second, q};
}

Red reduce_large(Math& m, const MV& a)
{
    MV xs = m.mul({a, m.sel({m.ge({a, 0x1p945}), 0x1p-128, 1.0})});
    std::vector<P2> parts;
    for(int seg = 0; seg < 4; ++seg)
    {
        MV p = m.trig_preop(a, seg);
        parts.push_back(two_prod(m, xs, p));
    }
    MV h0 = parts[0].first, l0 = parts[0].second;
    MV h1 = parts[1].first, l1 = parts[1].second;
    MV h2 = parts[2].first, l2 = parts[2].second;
    MV h3 = parts[3].first, l3 = parts[3].second;
    h0 = m.sub({h0, m.mul({4.0, floor_nonneg(m, m.mul({h0, 0.25}))})});
    P2 s1 = two_sum(m, h0, l0);
    P2 s2 = two_sum(m, s1.first, h1);
    MV k = rint_small(m, s2.first);
    MV f1 = m.sub({s2.first, k});
    P2 u1 = two_sum(m, f1, s2.second);
    P2 u2 = two_sum(m, u1.first, s1.second);
    P2 u3 = two_sum(m, u2.first, l1);
    P2 u4 = two_sum(m, u3.first, h2);
    MV lo = m.add({m.add({m.add({l3, h3}), l2}),
                   m.add({m.add({u4.second, u3.second}), m.add({u2.second, u1.second})})});
    P2 fr = two_sum(m, u4.first, lo);
    MV fh = fr.first, fl = fr.second;
    MV rh = m.mul({fh, PIO2_0});
    MV rl = m.fma({fh, PIO2_0, m.neg(rh)});
    rl = m.add({rl, m.fma({fl, PIO2_0, m.mul({fh, PIO2_1})})});
    P2 r = fast_two_sum(m, rh, rl);
    MV q = m.iand({int_bits(m, k), 3});
    return {r.first, r.second, q};
}

Red reduce_pio2(Math& m, const MV& a)
{
    Red r = reduce_medium(m, a);
    MV large = m.band({m.ge({a, TRIG_LARGE}), m.lt({a, INF})});
    std::vector<MV> v = m.when(large, {r.rh, r.rl, r.q}, [&](const std::vector<MV>&) {
        Red g = reduce_large(m, a);
        return std::vector<MV>{g.rh, g.rl, g.q};
    });
    return {v[0], v[1], v[2]};
}

std::pair<P2, P2> sin_cos_kernel(Math& m, const MV& rh, const MV& rl)
{
    P2 z = two_prod(m, rh, rh);
    MV zh = z.first, zl = z.second;
    P2 s = series_dd(m, rh, zh, zl, SIN_S);
    MV sh = s.first;
    MV sl = m.add({s.second, m.mul({rl, m.fma({-0.5, zh, 1.0})})});
    P2 s2 = fast_two_sum(m, sh, sl);
    MV hz = m.mul({0.5, zh});
    MV w = m.sub({1.0, hz});
    MV tail = m.sub({m.sub({1.0, w}), hz});
    MV c4 = m.mul({m.mul({zh, zh}), horner(m, COS_C, zh)});
    MV rest = m.add({m.sub({tail, m.fma({0.5, zl, m.mul({rh, rl})})}), c4});
    P2 c = fast_two_sum(m, w, rest);
    return {s2, c};
}

MV dd_div(Math& m, const MV& nh, const MV& nl, const MV& dh, const MV& dl)
{
    MV q = m.div({nh, dh});
    MV r = m.fma({m.neg(q), dh, nh});
    r = m.sub({m.add({r, nl}), m.mul({q, dl})});
    return m.add({q, m.div({r, dh})});
}

MV trig(Math& m, const MV& x, const MV& is_sin, const MV& is_cos, const MV& is_tan)
{
    MV a = m.abs(x);
    Red red = reduce_pio2(m, a);
    std::pair<P2, P2> sc = sin_cos_kernel(m, red.rh, red.rl);
    MV sh = sc.first.first, sl = sc.first.second;
    MV ch = sc.second.first, cl = sc.second.second;
    MV q = red.q;
    MV odd = m.ieq({m.iand({q, 1}), 1});
    MV out;
    bool have = false;
    if(!is_sin.is_false())
    {
        MV v = m.sel({odd, m.add({ch, cl}), m.add({sh, sl})});
        v = m.sel({m.ieq({m.iand({q, 2}), 2}), m.neg(v), v});
        out = v;
        have = true;
    }
    if(!is_cos.is_false())
    {
        MV v = m.sel({odd, m.add({sh, sl}), m.add({ch, cl})});
        v = m.sel({m.ieq({m.iand({m.iadd({q, 1}), 2}), 2}), m.neg(v), v});
        out = have ? m.sel({is_cos, v, out}) : v;
        have = true;
    }
    if(!is_tan.is_false())
    {
        auto tan_q = [&](const std::vector<MV>&) {
            MV n_h = m.sel({odd, ch, sh});
            MV n_l = m.sel({odd, cl, sl});
            MV d_h = m.sel({odd, sh, ch});
            MV d_l = m.sel({odd, sl, cl});
            MV qq = dd_div(m, n_h, n_l, d_h, d_l);
            return std::vector<MV>{m.sel({odd, m.neg(qq), qq})};
        };
        MV t = m.when(is_tan, {MV(0.0)}, tan_q)[0];
        out = have ? m.sel({is_tan, t, out}) : t;
        have = true;
    }
    MV odd_fn = m.bor({is_sin, is_tan});
    out = m.sel({odd_fn, m.from_bits(m.ixor({m.bits(out), m.iand({m.bits(x), SIGN})})), out});
    return m.sel({m.lt({a, INF}), out, m.sub({x, x})});
}

/* ------------------------------------------------------------ sinh / cosh */

struct ExpCore
{
    MV n, mh, ml;
};

ExpCore exp_core(Math& m, const MV& a)
{
    MV nf = rint_small(m, m.mul({a, LOG2E}));
    MV r0 = m.fma({m.neg(nf), LN2_42, a});
    P2 p = two_prod(m, nf, LN2_42_1);
    P2 re = two_sum(m, r0, m.neg(p.first));
    MV rh = re.first;
    MV rl = m.fma({m.neg(nf), LN2_42_2, m.sub({re.second, p.second})});
    P2 qq = two_prod(m, rh, rh);
    MV c3 = m.mul({m.mul({rh, qq.first}), horner(m, EXP_P, rh)});
    P2 ab1 = two_sum(m, 1.0, rh);
    P2 ab2 = two_sum(m, ab1.first, m.mul({0.5, qq.first}));
    MV ml = m.add({m.add({ab1.second, ab2.second}),
                   m.add({m.fma({0.5, qq.second, c3}), m.fma({rl, rh, rl})})});
    P2 mm = fast_two_sum(m, ab2.first, ml);
    MV n = int_bits(m, nf);
    return {n, mm.first, mm.second};
}

MV hyp(Math& m, const MV& x, const MV& is_sinh, const MV& is_cosh)
{
    MV a = m.abs(x);
    MV aa = m.sel({m.lt({a, 711.0}), a, 711.0});
    ExpCore e = exp_core(m, aa);
    MV n = e.n, mh = e.mh, ml = e.ml;
    MV i = m.isub({n, 1});
    MV i1 = m.ilshr(m.iadd({i, 1024}), 1);
    i1 = m.isub({i1, 512});
    MV s1 = pow2(m, i1);
    MV s2 = pow2(m, m.isub({i, i1}));
    MV hh = m.mul({m.mul({mh, s1}), s2});
    MV hl = m.mul({m.mul({ml, s1}), s2});
    MV gh = m.div({1.0, mh});
    MV gl = m.mul({gh, m.sub({m.fma({m.neg(gh), mh, 1.0}), m.mul({gh, ml})})});
    MV j = m.isub({m.isub({0, n}), 1});
    MV j1 = m.isub({m.ilshr(m.iadd({j, 2048}), 1), 1024});
    MV t1 = pow2(m, j1);
    MV t2 = pow2(m, m.isub({j, j1}));
    gh = m.mul({m.mul({gh, t1}), t2});
    gl = m.mul({m.mul({gl, t1}), t2});
    MV out;
    bool have = false;
    if(!is_sinh.is_false())
    {
        P2 uv = two_sum(m, hh, m.neg(gh));
        MV big = m.add({uv.first, m.add({uv.second, m.sub({hl, gl})})});
        auto series = [&](const std::vector<MV>&) {
            P2 z = two_prod(m, a, a);
            P2 s = series_dd(m, a, z.first, z.second, SINH_H);
            return std::vector<MV>{m.add({s.first, s.second})};
        };
        MV lt1 = m.lt({a, 1.0});
        MV small = m.when(m.band({is_sinh, lt1}), {MV(0.0)}, series)[0];
        MV v = m.sel({lt1, small, big});
        v = m.sel({m.eq({hh, INF}), INF, v});
        out = copysign(m, v, x);
        have = true;
    }
    if(!is_cosh.is_false())
    {
        P2 uv = two_sum(m, hh, gh);
        MV c = m.add({uv.first, m.add({uv.second, m.add({hl, gl})})});
        c = m.sel({m.eq({hh, INF}), INF, c});
        out = have ? m.sel({is_cosh, c, out}) : c;
        have = true;
    }
    return m.sel({m.isnan(x), x, out});
}

/* ----------------------------------------------------- asin / acos / atan */

MV asin_acos(Math& m, const MV& x, const MV& is_asin, const MV& is_acos)
{
    MV a = m.abs(x);
    MV negative = is_negative(m, x);
    MV small = m.le({a, 0.5});

    auto near_zero = [&](const std::vector<MV>&) {
        P2 z = two_prod(m, a, a);
        P2 s = series_dd(m, a, z.first, z.second, ASIN_R);
        return std::vector<MV>{s.first, s.second};
    };
    auto near_one = [&](const std::vector<MV>&) {
        MV z = m.mul({m.sub({1.0, a}), 0.5});
        MV s = m.sqrt(z);
        MV s_l = m.sel({m.gt({s, 0.0}), m.div({m.fma({m.neg(s), s, z}), m.add({s, s})}), 0.0});
        P2 w = series_dd(m, s, z, 0.0, ASIN_R);
        MV wl = m.add({w.second, m.mul({s_l, m.fma({z, ASIN_R[0], 1.0})})});
        P2 w2 = fast_two_sum(m, w.first, wl);
        MV r0 = m.mul({2.0, w2.first});
        MV r1 = m.mul({2.0, w2.second});
        return std::vector<MV>{r0, r1};
    };
    std::vector<MV> pp = m.when(small, {MV(0.0), MV(0.0)}, near_zero);
    MV ph = pp[0], pl = pp[1];
    std::vector<MV> ww = m.when(m.gt({a, 0.5}), {MV(0.0), MV(0.0)}, near_one);
    MV wh2 = ww[0], wl2 = ww[1];
    MV out;
    bool have = false;
    if(!is_asin.is_false())
    {
        P2 uv = two_sum(m, PIO2_0, m.neg(wh2));
        MV big = m.add({uv.first, m.add({uv.second, m.sub({PIO2_1, wl2})})});
        MV v = m.sel({small, m.add({ph, pl}), big});
        out = copysign(m, v, x);
        have = true;
    }
    if(!is_acos.is_false())
    {
        MV sph = m.sel({negative, m.neg(ph), ph});
        MV spl = m.sel({negative, m.neg(pl), pl});
        P2 uv = two_sum(m, PIO2_0, m.neg(sph));
        MV c_small = m.add({uv.first, m.add({uv.second, m.sub({PIO2_1, spl})})});
        P2 uv2 = two_sum(m, PI_H, m.neg(wh2));
        MV c_neg = m.add({uv2.first, m.add({uv2.second, m.sub({PI_L, wl2})})});
        MV c_pos = m.add({wh2, wl2});
        MV c = m.sel({small, c_small, m.sel({negative, c_neg, c_pos})});
        out = have ? m.sel({is_acos, c, out}) : c;
        have = true;
    }
    return m.sel({m.le({a, 1.0}), out, m.sub({m.add({x, INF}), INF})});
}

MV atan(Math& m, const MV& x)
{
    MV a = m.abs(x);
    MV b1 = m.gt({a, 7.0 / 16});
    MV b2 = m.gt({a, 11.0 / 16});
    MV b3 = m.gt({a, 19.0 / 16});
    MV b4 = m.gt({a, 39.0 / 16});
    auto pick = [&](const MV& v0, const MV& v1, const MV& v2, const MV& v3, const MV& v4) {
        return m.sel({b4, v4, m.sel({b3, v3, m.sel({b2, v2, m.sel({b1, v1, v0})})})});
    };
    MV nA = pick(1.0, 2.0, 1.0, 1.0, 0.0);
    MV nB = pick(0.0, -1.0, -1.0, -1.5, -1.0);
    MV dA = pick(0.0, 1.0, 1.0, 1.5, 1.0);
    MV dB = pick(1.0, 2.0, 1.0, 1.0, 0.0);
    MV num = m.fma({nA, a, nB});
    P2 p = two_prod(m, dA, a);
    P2 d = two_sum(m, p.first, dB);
    MV dh = d.first;
    MV dl = m.add({d.second, p.second});
    MV th = m.div({num, dh});
    MV tl = m.div({m.sub({m.fma({m.neg(th), dh, num}), m.mul({th, dl})}), dh});
    P2 z = two_prod(m, th, th);
    P2 s = series_dd(m, th, z.first, z.second, ATAN_A);
    MV sl = m.add({s.second, m.mul({tl, m.sub({1.0, z.first})})});
    MV ch = pick(0.0, ATAN_HALF[0], ATAN_ONE[0], ATAN_3HALF[0], ATAN_INF[0]);
    MV cl = pick(0.0, ATAN_HALF[1], ATAN_ONE[1], ATAN_3HALF[1], ATAN_INF[1]);
    P2 uv = two_sum(m, ch, s.first);
    MV r = m.add({uv.first, m.add({uv.second, m.add({sl, cl})})});
    r = m.sel({m.eq({a, INF}), PIO2_0, r});
    r = copysign(m, r, x);
    return m.sel({m.isnan(x), x, r});
}

/* --------------------------------------------------- asinh / acosh / atanh */

P2 log_dd(Math& m, const MV& yh, const MV& yl)
{
    MV bits = m.bits(yh);
    MV e = m.isub({m.ilshr(bits, 52), 1023});
    MV mm = m.from_bits(m.ior({m.iand({bits, MANT_MASK}), ONE_BITS}));
    MV up = m.gt({mm, SQRT2});
    mm = m.sel({up, m.mul({mm, 0.5}), mm});
    e = m.iadd({e, m.sel({up, 1, 0})});
    MV ef = int_to_f64(m, e);
    MV f = m.sub({mm, 1.0});
    MV fl = m.mul({yl, m.div({mm, yh})});
    MV s = m.div({f, m.add({2.0, f})});
    MV z = m.mul({s, s});
    MV rz = m.mul({z, horner(m, LOG_G, z)});
    P2 q = two_prod(m, f, f);
    MV hf = m.mul({0.5, q.first});
    MV hfl = m.mul({0.5, q.second});
    MV corr = m.mul({s, m.add({hf, rz})});
    P2 l1 = two_sum(m, f, m.neg(hf));
    MV tail = m.add({m.sub({l1.second, hfl}), m.add({corr, m.div({fl, mm})})});
    P2 t = two_sum(m, m.mul({ef, LN2_42}), l1.first);
    MV tl = m.add({t.second, m.fma({ef, LN2_42_1, tail})});
    return fast_two_sum(m, t.first, tl);
}

P2 sqrt_dd(Math& m, const MV& th, const MV& tl)
{
    MV sh = m.sqrt(th);
    MV sl = m.sel(
        {m.gt({sh, 0.0}), m.div({m.add({m.fma({m.neg(sh), sh, th}), tl}), m.add({sh, sh})}), 0.0});
    return {sh, sl};
}

MV ainvhyp(Math& m, const MV& x, const MV& is_asinh, const MV& is_acosh, const MV& is_atanh)
{
    MV a = m.abs(x);
    MV big = m.gt({a, 0x1p28});
    MV v = m.sel({is_acosh, x, a});

    auto root_arg = [&](const std::vector<MV>&) {
        P2 p = two_prod(m, v, v);
        P2 t = two_sum(m, p.first, m.sel({is_acosh, -1.0, 1.0}));
        P2 sq = sqrt_dd(m, t.first, m.add({t.second, p.second}));
        P2 y = two_sum(m, v, sq.first);
        MV ylo = m.add({y.second, sq.second});
        return std::vector<MV>{y.first, ylo};
    };
    auto ratio_arg = [&](const std::vector<MV>&) {
        P2 d = two_sum(m, 1.0, m.neg(a));
        MV two_a = m.add({a, a});
        MV uh = m.div({two_a, d.first});
        MV ul
            = m.div({m.sub({m.fma({m.neg(uh), d.first, two_a}), m.mul({uh, d.second})}), d.first});
        P2 z = two_sum(m, 1.0, uh);
        MV zlo = m.add({z.second, ul});
        return std::vector<MV>{z.first, zlo};
    };
    std::vector<MV> y = m.when(m.bnot(is_atanh), {MV(0.0), MV(0.0)}, root_arg);
    MV yh = y[0], yl = y[1];
    if(!is_atanh.is_false())
    {
        std::vector<MV> z = m.when(is_atanh, {MV(0.0), MV(0.0)}, ratio_arg);
        yh = m.sel({is_atanh, z[0], yh});
        yl = m.sel({is_atanh, z[1], yl});
    }
    MV use_big = m.band({big, m.bnot(is_atanh)});
    yh = m.sel({use_big, v, yh});
    yl = m.sel({use_big, 0.0, yl});
    P2 l = log_dd(m, yh, yl);
    P2 uw = two_sum(m, l.first, LN2_42);
    MV bh = uw.first;
    MV bl = m.add({uw.second, m.add({l.second, LN2_42_1})});
    MV rh = m.sel({use_big, bh, l.first});
    MV rl = m.sel({use_big, bl, l.second});
    MV r = m.add({rh, rl});
    if(!is_atanh.is_false())
        r = m.sel({is_atanh, m.mul({0.5, r}), r});
    MV out = r;
    if(!is_asinh.is_false())
    {
        MV s = m.sel({m.lt({a, 0x1p-28}), a, r});
        s = m.sel({m.eq({a, INF}), INF, s});
        s = copysign(m, s, x);
        out = is_asinh.is_true() ? s : m.sel({is_asinh, s, out});
    }
    if(!is_acosh.is_false())
    {
        MV c = m.sel({m.eq({x, 1.0}), 0.0, r});
        c = m.sel({m.eq({x, INF}), INF, c});
        c = m.sel({m.ge({x, 1.0}), c, NAN_});
        out = is_acosh.is_true() ? c : m.sel({is_acosh, c, out});
    }
    if(!is_atanh.is_false())
    {
        MV t = m.sel({m.lt({a, 0x1p-28}), a, r});
        t = m.sel({m.eq({a, 1.0}), INF, t});
        t = m.sel({m.le({a, 1.0}), t, NAN_});
        t = copysign(m, t, x);
        out = is_atanh.is_true() ? t : m.sel({is_atanh, t, out});
    }
    return m.sel({m.isnan(x), x, out});
}

MV pred_of(const std::vector<Pred>& preds, const char* name)
{
    for(const Pred& p : preds)
        if(strcmp(p.name, name) == 0)
            return p.p;
    return MV(false);
}

} // namespace

MV family(Math& m, const char* fam, const MV& x, const std::vector<Pred>& preds)
{
    const family_t* F = find_family(fam);
    if(F == nullptr)
    {
        rocke_i_set_err(m.builder(), ROCKE_ERR_VALUE, "elementary: unknown family");
        return MV();
    }
    MV p[3];
    for(int k = 0; k < F->n; ++k)
        p[k] = pred_of(preds, F->members[k]);
    if(strcmp(fam, "trig") == 0)
        return trig(m, x, p[0], p[1], p[2]);
    if(strcmp(fam, "hyp") == 0)
        return hyp(m, x, p[0], p[1]);
    if(strcmp(fam, "asin_acos") == 0)
        return asin_acos(m, x, p[0], p[1]);
    if(strcmp(fam, "atan") == 0)
        return atan(m, x);
    if(strcmp(fam, "ainvhyp") == 0)
        return ainvhyp(m, x, p[0], p[1], p[2]);
    /* round */
    return floor_ceil(m, x, p[1]);
}

std::vector<double> family_constants(const char* fam, const std::vector<Pred>& preds)
{
    static std::mutex mu;
    static std::map<std::string, std::vector<double>> cache;
    const family_t* F = find_family(fam);
    if(F == nullptr)
        return {};
    /* the key: the family and, per member present in preds, its kind */
    std::string key = fam;
    std::vector<Pred> kinds;
    for(int k = 0; k < F->n; ++k)
        for(const Pred& p : preds)
            if(strcmp(p.name, F->members[k]) == 0)
            {
                key += std::string("|") + p.name + (p.p.is_bool() ? (p.p.bv ? "=T" : "=F") : "=v");
                kinds.push_back(p);
                break;
            }
    {
        std::lock_guard<std::mutex> lock(mu);
        auto it = cache.find(key);
        if(it != cache.end())
            return it->second;
    }
    rocke_ir_builder_t sb;
    std::vector<double> out;
    if(rocke_ir_builder_init(&sb, "elementary_constants") == ROCKE_OK)
    {
        rocke_value_t* x = rocke_b_param(&sb, "x", rocke_f64(), nullptr);
        std::vector<Pred> p2;
        for(const Pred& p : kinds)
        {
            if(p.p.is_bool())
                p2.push_back({p.name, p.p});
            else
            {
                std::string pn = std::string("p_") + p.name;
                p2.push_back({p.name, MV(rocke_b_param(&sb, pn.c_str(), rocke_i1(), nullptr))});
            }
        }
        Math rec(&sb);
        rec.recording = true;
        family(rec, fam, MV(x), p2);
        /* constants used once stay at their use (the polynomial coefficients) */
        for(double c : rec.seen)
            if(rec.uses[bits_of(c)] >= 2)
                out.push_back(c);
        rocke_ir_builder_free(&sb);
    }
    std::lock_guard<std::mutex> lock(mu);
    cache[key] = out;
    return out;
}

} // namespace elementary
} // namespace rocke

/* ================================================================ C API */

using rocke::elementary::Math;
using rocke::elementary::MV;
using rocke::elementary::Pred;

extern "C" {

static rocke_value_t* opaque_zero_impl(rocke_ir_builder_t* b)
{
    rocke_inline_asm_opts_t o;
    memset(&o, 0, sizeof o);
    o.sideeffect = true;
    o.sideeffect_set = true;
    const rocke_type_t* rt[1] = {rocke_i64()};
    rocke_op_t* op
        = rocke_b_inline_asm_named(b, "s_mov_b64 $0, 0", "=s", nullptr, 0, rt, 1, &o, "ozero");
    return op != nullptr ? op->results[0] : nullptr;
}

int rocke_elementary_num_functions(void)
{
    return (int)(sizeof rocke::elementary::k_functions / sizeof rocke::elementary::k_functions[0]);
}

const char* rocke_elementary_function(int i)
{
    return i >= 0 && i < rocke_elementary_num_functions() ? rocke::elementary::k_functions[i]
                                                          : nullptr;
}

int rocke_elementary_num_families(void)
{
    return (int)(sizeof rocke::elementary::k_families / sizeof rocke::elementary::k_families[0]);
}

const char* rocke_elementary_family(int i)
{
    return i >= 0 && i < rocke_elementary_num_families() ? rocke::elementary::k_families[i].name
                                                         : nullptr;
}

int rocke_elementary_family_members(const char* fam, const char* const** out)
{
    const rocke::elementary::family_t* F = rocke::elementary::find_family(fam);
    if(F == nullptr)
        return 0;
    if(out != nullptr)
        *out = F->members;
    return F->n;
}

const char* rocke_elementary_family_of(const char* name)
{
    if(name == nullptr)
        return nullptr;
    for(const auto& F : rocke::elementary::k_families)
        for(int k = 0; k < F.n; ++k)
            if(strcmp(F.members[k], name) == 0)
                return F.name;
    return nullptr;
}

static rocke_value_t*
    emit_impl(rocke_ir_builder_t* b, const char* name, rocke_value_t* x, bool anchored)
{
    if(b == nullptr || x == nullptr)
        return nullptr;
    const char* fam = rocke_elementary_family_of(name);
    if(fam == nullptr)
    {
        char msg[96];
        snprintf(msg, sizeof msg, "unknown elementary function '%s'", name ? name : "(null)");
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "%s", msg);
        return nullptr;
    }
    rocke_value_t* anchor = anchored ? rocke_elementary_opaque_zero(b) : nullptr;
    Math m(b, anchor);
    std::vector<Pred> preds = {{name, MV(true)}};
    if(anchored)
        m.materialize(rocke::elementary::family_constants(fam, preds));
    if(rocke_type_eq(x->type, rocke_f64()))
        return rocke::elementary::family(m, fam, MV(x), preds).v;
    if(!rocke_type_eq(x->type, rocke_f32()))
    {
        char msg[96];
        snprintf(msg, sizeof msg, "elementary.%s takes f32 or f64", name);
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "%s", msg);
        return nullptr;
    }
    rocke_value_t* x64 = rocke_b_cvt_f32_to_f64(b, x);
    MV r = rocke::elementary::family(m, fam, MV(x64), preds);
    return rocke_b_cvt_f64_to_f32(b, r.v);
}

rocke_value_t* rocke_elementary_opaque_zero(rocke_ir_builder_t* b)
{
    return ckc::guard_builder(b, [&]() -> rocke_value_t* { return opaque_zero_impl(b); });
}

rocke_value_t*
    rocke_elementary_emit(rocke_ir_builder_t* b, const char* name, rocke_value_t* x, bool anchored)
{
    return ckc::guard_builder(b,
                              [&]() -> rocke_value_t* { return emit_impl(b, name, x, anchored); });
}

} /* extern "C" */
