// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * hiptensor_aot_common.cpp -- C++ mirror of
 * library/kernels/common/hiptensor_aot/common.py (Emit, ptr, the dtype and
 * operator tables). See rocke/instance_hiptensor_aot_internal.h.
 */
#include "rocke/instance_hiptensor_aot_internal.h"

#include "rocke/elementary.h"
#include "rocke/ir_internal.h"

#include <string.h>

namespace rocke
{
namespace htaot
{

bool streq(const char* a, const char* b)
{
    return a != nullptr && b != nullptr && strcmp(a, b) == 0;
}

bool present(const char* s)
{
    return s != nullptr && s[0] != '\0';
}

bool dtype_of(const char* dt, const rocke_type_t** ty, int* nbytes)
{
    const rocke_type_t* t = nullptr;
    int n = 0;
    if(streq(dt, "f16"))
        t = rocke_f16(), n = 2;
    else if(streq(dt, "bf16"))
        t = rocke_bf16(), n = 2;
    else if(streq(dt, "f32"))
        t = rocke_f32(), n = 4;
    else if(streq(dt, "f64"))
        t = rocke_f64(), n = 8;
    else
        return false;
    if(ty != nullptr)
        *ty = t;
    if(nbytes != nullptr)
        *nbytes = n;
    return true;
}

bool is_dtype(const char* dt)
{
    return dtype_of(dt, nullptr, nullptr);
}

V* param(rocke_ir_builder_t* b, const std::string& name, const rocke_type_t* t)
{
    return rocke_b_param(b, name.c_str(), t, nullptr);
}

V* ptr(rocke_ir_builder_t* b, const char* name, const char* dt, bool readonly)
{
    const rocke_type_t* ty = nullptr;
    int nb = 0;
    if(!dtype_of(dt, &ty, &nb))
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "hiptensor_aot: unknown dtype");
        return nullptr;
    }
    rocke_param_opts_t o;
    memset(&o, 0, sizeof o);
    o.align = nb;
    o.align_set = true;
    if(readonly)
    {
        o.readonly = true;
        o.readonly_set = true;
    }
    return rocke_b_param(b, name, rocke_ptr_type(b, ty, "global"), &o);
}

rocke_kernel_def_t* fail(rocke_ir_builder_t* b, const std::string& msg)
{
    rocke_i_set_err(b, ROCKE_ERR_VALUE, "%s", msg.c_str());
    return nullptr;
}

/* =================================================================== Emit */

Emit::Emit(rocke_ir_builder_t* b_, bool idx64_, const char* cdt, bool math_)
    : b(b_)
    , math(math_)
    , idx64(idx64_)
    , f64(streq(cdt, "f64"))
    , cty(f64 ? rocke_f64() : rocke_f32())
{
}

V* Emit::c32(int64_t v)
{
    return rocke_b_const_i32(b, v);
}

V* Emit::cf(double v)
{
    return f64 ? rocke_b_const_f64(b, v) : rocke_b_const_f32(b, v);
}

V* Emit::cf_f32(double v)
{
    return rocke_b_const_f32(b, v);
}

V* Emit::zero_off()
{
    return idx64 ? rocke_b_const_i64(b, 0) : rocke_b_const_i32(b, 0);
}

V* Emit::stride_param(const char* name)
{
    return rocke_b_param(b, name, idx64 ? rocke_i64() : rocke_i32(), nullptr);
}

V* Emit::scalar_param(const char* name)
{
    return rocke_b_param(b, name, cty, nullptr);
}

V* Emit::udiv(V* x, V* m, V* s)
{
    V* hi = rocke_b_umul_hi_i32(b, x, m);
    V* q = rocke_b_lshr(b, hi, s);
    V* z = c32(0);
    V* is1 = rocke_b_cmp_eq(b, m, z);
    return rocke_b_select(b, is1, x, q);
}

std::pair<V*, V*> Emit::divmod(V* x, V* d, V* m, V* s)
{
    V* q = udiv(x, m, s);
    V* qd = rocke_b_mul(b, q, d);
    return {q, rocke_b_sub(b, x, qd)};
}

V* Emit::off(V* coord, V* stride)
{
    if(idx64)
    {
        V* c = rocke_b_sext(b, coord, rocke_i64());
        return rocke_b_mul(b, c, stride);
    }
    return rocke_b_mul(b, coord, stride);
}

std::pair<V*, V*> Emit::addr(V* p, V* o, int nbytes)
{
    if(idx64)
    {
        V* nb = rocke_b_const_i64(b, nbytes);
        V* bo = rocke_b_mul(b, o, nb);
        V* pp = rocke_b_global_ptr_add(b, p, bo);
        return {pp, rocke_b_const_i32(b, 0)};
    }
    return {p, o};
}

std::vector<V*> Emit::load(V* p, V* o, const char* dt, bool cplx)
{
    const rocke_type_t* ty = nullptr;
    int nb = 0;
    dtype_of(dt, &ty, &nb);
    if(cplx)
    {
        std::pair<V*, V*> pi = addr(p, o, 2 * nb);
        V* idx = pi.second;
        if(!idx64)
            idx = rocke_b_add(b, idx, idx);
        V* v = rocke_b_global_load_vN(b, pi.first, idx, ty, 2, 2 * nb);
        V* x0 = rocke_b_vec_extract(b, v, 0);
        V* r0 = to_c(x0, dt);
        V* x1 = rocke_b_vec_extract(b, v, 1);
        V* r1 = to_c(x1, dt);
        return {r0, r1};
    }
    std::pair<V*, V*> pi = addr(p, o, nb);
    V* v = rocke_b_global_load(b, pi.first, pi.second, ty, nb);
    return {to_c(v, dt)};
}

void Emit::store(V* p, V* o, const std::vector<V*>& vals, const char* dt)
{
    const rocke_type_t* ty = nullptr;
    int nb = 0;
    dtype_of(dt, &ty, &nb);
    std::vector<V*> outs;
    for(V* v : vals)
        outs.push_back(from_c(v, dt));
    if(outs.size() == 2)
    {
        std::pair<V*, V*> pi = addr(p, o, 2 * nb);
        V* idx = pi.second;
        if(!idx64)
            idx = rocke_b_add(b, idx, idx);
        V* vec = rocke_b_vec_pack(b, outs.data(), 2, ty);
        rocke_b_global_store_vN(b, pi.first, idx, vec, 2, 2 * nb);
        return;
    }
    std::pair<V*, V*> pi = addr(p, o, nb);
    rocke_b_global_store(b, pi.first, pi.second, outs[0], nb);
}

V* Emit::to_c(V* v, const char* dt)
{
    if(f64)
    {
        if(streq(dt, "f64"))
            return v;
        V* v32 = streq(dt, "f32") ? v : rocke_b_cast_to_f32(b, v);
        return rocke_b_cvt_f32_to_f64(b, v32);
    }
    if(streq(dt, "f64"))
        return rocke_b_cvt_f64_to_f32(b, v);
    return streq(dt, "f32") ? v : rocke_b_cast_to_f32(b, v);
}

V* Emit::from_c(V* v, const char* dt)
{
    const rocke_type_t* ty = nullptr;
    dtype_of(dt, &ty, nullptr);
    if(f64)
    {
        if(streq(dt, "f64"))
            return v;
        if(streq(dt, "f32"))
            return rocke_b_cvt_f64_to_f32(b, v);
        V* odd = f64_to_f32_odd(v);
        return narrow(odd, ty);
    }
    if(streq(dt, "f64"))
        return rocke_b_cvt_f32_to_f64(b, v);
    return streq(dt, "f32") ? v : narrow(v, ty);
}

V* Emit::narrow(V* v, const rocke_type_t* ty)
{
    if(!rocke_type_eq(ty, rocke_f16()))
        return rocke_b_cast_f32_to(b, v, ty);
    V* bits = rocke_b_bitcast(b, v, rocke_i32());
    V* sm = c32(-((int64_t)1 << 31));
    V* sb = rocke_b_land(b, bits, sm);
    V* signed_zero = rocke_b_bitcast(b, sb, rocke_f32());
    V* z = cf_f32(0.0);
    V* is_zero = rocke_b_fcmp(b, "oeq", v, z);
    V* s = rocke_b_select(b, is_zero, signed_zero, v);
    return rocke_b_cast_f32_to(b, s, ty);
}

V* Emit::f64_to_f32_odd(V* v)
{
    V* r = rocke_b_cvt_f64_to_f32(b, v);
    V* back = rocke_b_cvt_f32_to_f64(b, r);
    V* inexact = rocke_b_fcmp(b, "one", back, v);
    V* ab = rocke_b_fabs(b, back);
    V* av = rocke_b_fabs(b, v);
    V* away = rocke_b_fcmp(b, "ogt", ab, av);
    V* bits = rocke_b_bitcast(b, r, rocke_i32());
    V* both = rocke_b_land(b, inexact, away);
    V* one = c32(1);
    V* dec = rocke_b_sub(b, bits, one);
    bits = rocke_b_select(b, both, dec, bits);
    V* one2 = c32(1);
    V* odd = rocke_b_lor(b, bits, one2);
    bits = rocke_b_select(b, inexact, odd, bits);
    return rocke_b_bitcast(b, bits, rocke_f32());
}

rocke_for_t Emit::when_open(V* cond, const std::vector<V*>& vals)
{
    V* one = c32(1);
    V* zero = c32(0);
    V* trips = rocke_b_select(b, cond, one, zero);
    int k = next_uid();
    std::vector<std::string> names;
    for(size_t i = 0; i < vals.size(); ++i)
        names.push_back("w" + std::to_string(k) + "_" + std::to_string(i));
    std::vector<rocke_iter_arg_t> args(vals.size());
    for(size_t i = 0; i < vals.size(); ++i)
    {
        args[i].name = names[i].c_str();
        args[i].init = vals[i];
    }
    std::string iv = "wi" + std::to_string(k);
    V* lo = c32(0);
    V* st = c32(1);
    rocke_for_t loop = rocke_b_scf_for_iter(
        b, lo, trips, st, args.data(), (int)args.size(), iv.c_str(), false, true);
    rocke_b_region_enter(b, loop.body);
    return loop;
}

std::vector<V*> Emit::when_close(const rocke_for_t& loop, const std::vector<V*>& out)
{
    rocke_b_scf_yield(b, out.data(), (int)out.size());
    rocke_b_region_leave(b);
    if(loop.op == nullptr)
        return std::vector<V*>(out.size(), nullptr);
    return std::vector<V*>(loop.op->results, loop.op->results + loop.op->num_results);
}

V* Emit::unary(V* v, V* code)
{
    return unary_list({v}, code)[0];
}

std::vector<V*> Emit::unary_list(const std::vector<V*>& vs, V* code)
{
    return unary_list(vs, code, math);
}

std::vector<V*> Emit::unary_list(const std::vector<V*>& vs, V* code, bool use_math)
{
    V* one = cf(1.0);
    auto table = [&](const std::vector<V*>& xs) {
        std::vector<V*> r = xs;
        static const int ops[]
            = {OP_SQRT, OP_RELU, OP_RCP, OP_SIGMOID, OP_TANH, OP_EXP, OP_LOG, OP_ABS, OP_NEG};
        for(int op : ops)
        {
            V* k = c32(op);
            V* hit = rocke_b_cmp_eq(b, code, k);
            r = when(hit, r, [&](const std::vector<V*>& c) {
                std::vector<V*> o;
                for(V* y : c)
                {
                    switch(op)
                    {
                    case OP_SQRT:
                        o.push_back(rocke_b_sqrt(b, y));
                        break;
                    case OP_RELU:
                    {
                        V* z = cf(0.0);
                        o.push_back(rocke_b_fmax(b, y, z));
                        break;
                    }
                    case OP_RCP:
                        o.push_back(rocke_b_fdiv(b, one, y));
                        break;
                    case OP_SIGMOID:
                        o.push_back(sigmoid(y));
                        break;
                    case OP_TANH:
                        o.push_back(rocke_b_tanh(b, y));
                        break;
                    case OP_EXP:
                        o.push_back(rocke_b_exp(b, y));
                        break;
                    case OP_LOG:
                        o.push_back(rocke_b_log(b, y));
                        break;
                    case OP_ABS:
                        o.push_back(rocke_b_fabs(b, y));
                        break;
                    default:
                        o.push_back(rocke_b_fneg(b, y));
                        break;
                    }
                }
                return o;
            });
        }
        if(use_math)
            r = math_ops(r, code);
        return r;
    };
    V* k1 = c32(OP_IDENTITY);
    V* n1 = rocke_b_cmp_ne(b, code, k1);
    V* k9 = c32(OP_CONJ);
    V* n9 = rocke_b_cmp_ne(b, code, k9);
    V* not_id = rocke_b_land(b, n1, n9);
    return when(not_id, vs, table);
}

std::vector<V*> Emit::unary_inputs(const std::vector<V*>& vs, const std::vector<V*>& codes)
{
    std::vector<V*> out;
    for(size_t i = 0; i < vs.size() && i < codes.size(); ++i)
        out.push_back(unary_list({vs[i]}, codes[i])[0]);
    return out;
}

V* Emit::sigmoid(V* y)
{
    V* one = cf(1.0);
    V* ay = rocke_b_fabs(b, y);
    V* ny = rocke_b_fneg(b, ay);
    V* t = rocke_b_exp(b, ny);
    V* den = rocke_b_fadd(b, one, t);
    V* z = cf(0.0);
    V* pos = rocke_b_fcmp(b, "oge", y, z);
    V* num = rocke_b_select(b, pos, one, t);
    return rocke_b_fdiv(b, num, den);
}

namespace
{

/* common.MATH_OPS in code order: (operator, elementary function name) */
const std::pair<int, const char*> k_math_ops[] = {
    {OP_SIN, "sin"},
    {OP_COS, "cos"},
    {OP_TAN, "tan"},
    {OP_SINH, "sinh"},
    {OP_COSH, "cosh"},
    {OP_ASIN, "asin"},
    {OP_ACOS, "acos"},
    {OP_ATAN, "atan"},
    {OP_ASINH, "asinh"},
    {OP_ACOSH, "acosh"},
    {OP_ATANH, "atanh"},
    {OP_CEIL, "ceil"},
    {OP_FLOOR, "floor"},
};

} // namespace

std::vector<V*> Emit::math_ops(const std::vector<V*>& xs, V* code)
{
    std::vector<V*> r = xs;
    for(int f = 0; f < rocke_elementary_num_families(); ++f)
    {
        const char* fam = rocke_elementary_family(f);
        const char* const* mem = nullptr;
        int nm = rocke_elementary_family_members(fam, &mem);
        std::vector<std::pair<const char*, V*>> preds;
        for(const auto& op : k_math_ops)
        {
            bool in = false;
            for(int k = 0; k < nm; ++k)
                in = in || strcmp(mem[k], op.second) == 0;
            if(!in)
                continue;
            V* k = c32(op.first);
            preds.push_back({op.second, rocke_b_cmp_eq(b, code, k)});
        }
        V* hit = nullptr;
        for(const auto& p : preds)
            hit = hit == nullptr ? p.second : rocke_b_lor(b, hit, p.second);
        r = when(hit, r, [&](const std::vector<V*>& c) { return math_loop(fam, c, preds); });
    }
    return r;
}

std::vector<V*> Emit::math_loop(const char* fam,
                                const std::vector<V*>& xs,
                                const std::vector<std::pair<const char*, V*>>& preds)
{
    if(xs.size() == 1)
        return {math_family(fam, xs[0], preds)};
    int k = next_uid();
    rocke_inline_asm_opts_t o;
    memset(&o, 0, sizeof o);
    o.sideeffect = true;
    o.sideeffect_set = true;
    const rocke_type_t* rt[1] = {rocke_i32()};
    rocke_op_t* aop
        = rocke_b_inline_asm_named(b, "s_mov_b32 $0, 0", "=s", nullptr, 0, rt, 1, &o, "ozero");
    V* opaque = aop != nullptr ? aop->results[0] : nullptr;
    std::vector<std::string> names;
    for(size_t i = 0; i < xs.size(); ++i)
        names.push_back("m" + std::to_string(k) + "_" + std::to_string(i));
    std::vector<rocke_iter_arg_t> args(xs.size());
    for(size_t i = 0; i < xs.size(); ++i)
    {
        args[i].name = names[i].c_str();
        args[i].init = xs[i];
    }
    std::string iv = "mi" + std::to_string(k);
    V* lo = c32(0);
    V* n = c32((int64_t)xs.size());
    V* hi = rocke_b_add(b, n, opaque);
    V* st = c32(1);
    rocke_for_t loop = rocke_b_scf_for_iter(
        b, lo, hi, st, args.data(), (int)args.size(), iv.c_str(), false, true);
    rocke_b_region_enter(b, loop.body);
    {
        std::vector<V*> cur(loop.iter_vars, loop.iter_vars + loop.num_iter_vars);
        V* it = loop.iv;
        V* y = cur[0];
        for(size_t j = 1; j < cur.size(); ++j)
        {
            V* cj = c32((int64_t)j);
            V* is = rocke_b_cmp_eq(b, it, cj);
            y = rocke_b_select(b, is, cur[j], y);
        }
        V* r = math_family(fam, y, preds);
        std::vector<V*> ys;
        for(size_t j = 0; j < cur.size(); ++j)
        {
            V* cj = c32((int64_t)j);
            V* is = rocke_b_cmp_eq(b, it, cj);
            ys.push_back(rocke_b_select(b, is, r, cur[j]));
        }
        rocke_b_scf_yield(b, ys.data(), (int)ys.size());
    }
    rocke_b_region_leave(b);
    if(loop.op == nullptr)
        return std::vector<V*>(xs.size(), nullptr);
    return std::vector<V*>(loop.op->results, loop.op->results + loop.op->num_results);
}

V* Emit::math_family(const char* fam, V* y, const std::vector<std::pair<const char*, V*>>& preds)
{
    using rocke::elementary::Math;
    using rocke::elementary::MV;
    using rocke::elementary::Pred;
    std::vector<Pred> ps;
    for(const auto& p : preds)
        ps.push_back({p.first, MV(p.second)});
    /* constants anchored in the branch: not hoisted out of the kernel's loops */
    V* anchor = rocke_elementary_opaque_zero(b);
    Math m(b, anchor);
    m.materialize(rocke::elementary::family_constants(fam, ps));
    if(f64)
        return rocke::elementary::family(m, fam, MV(y), ps).v;
    V* y64 = rocke_b_cvt_f32_to_f64(b, y);
    V* r = rocke::elementary::family(m, fam, MV(y64), ps).v;
    return rocke_b_cvt_f64_to_f32(b, r);
}

V* Emit::binary(V* a, V* c, V* code, bool nan_max)
{
    V* mx = rocke_b_fmax(b, a, c);
    V* mn = rocke_b_fmin(b, a, c);
    if(nan_max)
    {
        V* uno = rocke_b_fcmp(b, "uno", a, c);
        V* nan = rocke_b_fadd(b, a, c);
        mx = rocke_b_select(b, uno, nan, mx);
        mn = rocke_b_select(b, uno, nan, mn);
    }
    V* r = rocke_b_fadd(b, a, c);
    V* k5 = c32(OP_MUL);
    V* is_mul = rocke_b_cmp_eq(b, code, k5);
    V* prod = rocke_b_fmul(b, a, c);
    r = rocke_b_select(b, is_mul, prod, r);
    V* k6 = c32(OP_MAX);
    V* is_max = rocke_b_cmp_eq(b, code, k6);
    r = rocke_b_select(b, is_max, mx, r);
    V* k7 = c32(OP_MIN);
    V* is_min = rocke_b_cmp_eq(b, code, k7);
    r = rocke_b_select(b, is_min, mn, r);
    return r;
}

std::pair<V*, V*> Emit::cmul(V* xr, V* xi, V* yr, V* yi)
{
    V* a = rocke_b_fmul(b, xr, yr);
    V* c = rocke_b_fmul(b, xi, yi);
    V* re = rocke_b_fsub(b, a, c);
    V* d = rocke_b_fmul(b, xr, yi);
    V* e = rocke_b_fmul(b, xi, yr);
    V* im = rocke_b_fadd(b, d, e);
    return {re, im};
}

} // namespace htaot
} // namespace rocke
