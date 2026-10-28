// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * hiptensor_aot_ew.cpp -- C++ mirror of
 * library/kernels/common/hiptensor_aot/ew.py (EwSpec, build_ew): the
 * shape-generic strided elementwise kernel of hipTensor's kernel pack.
 */
#include "rocke/instance_hiptensor_aot_internal.h"

#include "rocke/error_boundary.hpp"
#include "rocke/ir_internal.h"

#include <stdio.h>
#include <string.h>

#include <map>
#include <string>
#include <vector>

using namespace rocke::htaot;

namespace
{

const char* scal_of(const std::string& n)
{
    return n == "A" ? "alpha" : n == "B" ? "beta" : "gamma";
}

std::vector<std::string> tensors(int arity)
{
    if(arity == 1)
        return {"A", "D"};
    if(arity == 2)
        return {"A", "C", "D"};
    return {"A", "B", "C", "D"};
}

const char* cdt_of(const rocke_htaot_ew_spec_t* s)
{
    const char* dts[4] = {s->a, s->b, s->c, s->d};
    for(const char* x : dts)
        if(streq(x, "f64"))
            return "f64";
    return "f32";
}

bool set_reason(char* reason, size_t cap, const char* msg)
{
    if(reason != nullptr && cap > 0)
        snprintf(reason, cap, "%s", msg);
    return false;
}

} // namespace

extern "C" {

rocke_htaot_ew_spec_t rocke_htaot_ew_spec_default(void)
{
    rocke_htaot_ew_spec_t s;
    memset(&s, 0, sizeof s);
    s.arity = 1;
    s.a = "f32";
    s.d = "f32";
    s.block = 256;
    return s;
}

bool rocke_htaot_ew_check(const rocke_htaot_ew_spec_t* s, char* reason, size_t cap)
{
    if(s == nullptr)
        return set_reason(reason, cap, "null spec");
    if(s->arity < 1 || s->arity > 3)
        return set_reason(reason, cap, "arity must be 1, 2 or 3");
    if(s->arity >= 2 && !present(s->c))
        return set_reason(reason, cap, "arity >= 2 needs a C dtype");
    if(s->arity == 3 && !present(s->b))
        return set_reason(reason, cap, "arity 3 needs a B dtype");
    if(s->cplx && s->arity == 3)
        return set_reason(reason, cap, "complex kernels have arity 1 or 2");
    if(s->cplx)
    {
        const char* xs[3] = {s->a, s->c, s->d};
        for(const char* x : xs)
            if(present(x) && !streq(x, "f32") && !streq(x, "f64"))
                return set_reason(reason, cap, "complex kernels move f32 or f64 component pairs");
    }
    if(s->block % 64)
        return set_reason(reason, cap, "block must be a multiple of 64");
    if(s->math && s->cplx)
        return set_reason(reason, cap, "complex kernels have no math operators");
    /* the builder's DTYPES lookup (Python KeyError) */
    const char* all[4] = {s->a, s->b, s->c, s->d};
    for(int i = 0; i < 4; ++i)
        if(present(all[i]) && !is_dtype(all[i]))
            return set_reason(reason, cap, "unknown dtype");
    if(!present(s->a) || !present(s->d))
        return set_reason(reason, cap, "A and D dtypes are required");
    if(reason != nullptr && cap > 0)
        reason[0] = '\0';
    return true;
}

rocke_status_t rocke_htaot_ew_kernel_name(const rocke_htaot_ew_spec_t* s, char* out, size_t cap)
{
    if(s == nullptr || out == nullptr)
        return ROCKE_ERR_VALUE;
    std::string dts;
    const char* all[4] = {s->a, s->b, s->c, s->d};
    for(const char* x : all)
        if(present(x))
            dts += (dts.empty() ? "" : "_") + std::string(x);
    std::string n = "hiptensor_aot_ew" + std::to_string(s->arity) + "_" + (s->cplx ? "c" : "r")
                    + "_" + dts + "_" + (s->idx64 ? "i64" : "i32") + "_b" + std::to_string(s->block)
                    + (s->math ? "_m" : "");
    if(n.size() + 1 > cap)
        return ROCKE_ERR_VALUE;
    memcpy(out, n.c_str(), n.size() + 1);
    return ROCKE_OK;
}

static rocke_kernel_def_t* ew_impl(rocke_ir_builder_t* b, const rocke_htaot_ew_spec_t* spec)
{
    char reason[160];
    char kname[160];
    if(b == nullptr)
        return nullptr;
    bool valid = rocke_htaot_ew_check(spec, reason, sizeof reason);
    if(spec == nullptr || rocke_htaot_ew_kernel_name(spec, kname, sizeof kname) != ROCKE_OK)
        snprintf(kname, sizeof kname, "hiptensor_aot_ew");
    if(rocke_ir_builder_init(b, kname) != ROCKE_OK)
        return nullptr;
    if(!valid)
        return fail(b, reason);
    rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", spec->block);
    Emit e(b, spec->idx64, cdt_of(spec), spec->math);
    std::vector<std::string> names = tensors(spec->arity);
    std::vector<std::string> ins(names.begin(), names.end() - 1);
    std::map<std::string, const char*> dts
        = {{"A", spec->a}, {"B", spec->b}, {"C", spec->c}, {"D", spec->d}};
    std::map<std::string, V*> P;
    for(const auto& n : names)
        P[n] = ptr(b, n.c_str(), dts[n], n != "D");
    std::map<std::string, std::vector<V*>> S;
    for(const auto& n : ins)
    {
        if(spec->cplx)
        {
            V* re = e.scalar_param((std::string(scal_of(n)) + "_re").c_str());
            V* im = e.scalar_param((std::string(scal_of(n)) + "_im").c_str());
            S[n] = {re, im};
        }
        else
            S[n] = {e.scalar_param(scal_of(n))};
    }
    std::map<std::string, V*> OPS;
    for(const auto& n : ins)
        OPS[n] = param(b, "op" + n, rocke_i32());
    if(spec->arity >= 2)
        OPS["AB"] = param(b, "opAB", rocke_i32());
    if(spec->arity == 3)
        OPS["ABC"] = param(b, "opABC", rocke_i32());
    V* n_in = param(b, "n_in", rocke_i32());
    V* n_out = param(b, "n_out", rocke_i32());
    V* ystep = param(b, "ystep", rocke_i32());
    std::vector<V*> ext;
    for(int i = 0; i < MAX_RANK; ++i)
        ext.push_back(param(b, "e" + std::to_string(i), rocke_i32()));
    std::map<int, std::pair<V*, V*>> mg;
    for(int i : {0, 1, 3, 4, 5, 6})
    {
        V* m = param(b, "m" + std::to_string(i), rocke_i32());
        V* s = param(b, "s" + std::to_string(i), rocke_i32());
        mg[i] = {m, s};
    }
    std::map<std::string, std::vector<V*>> ST;
    for(const auto& n : names)
        for(int i = 0; i < MAX_RANK; ++i)
            ST[n].push_back(e.stride_param(("s" + n + std::to_string(i)).c_str()));
    std::vector<V*> lim;
    if(spec->arity == 1)
        for(int i = 0; i < MAX_RANK; ++i)
            lim.push_back(param(b, "lim" + std::to_string(i), rocke_i32()));

    V* tid = rocke_b_thread_id_x(b);
    V* bx = rocke_b_block_id_x(b);
    V* cb = e.c32(spec->block);
    V* bxb = rocke_b_mul(b, bx, cb);
    V* t = rocke_b_add(b, bxb, tid);
    V* t_ok = rocke_b_cmp_lt(b, t, n_in);
    V* z0 = e.c32(0);
    V* ts = rocke_b_select(b, t_ok, t, z0);
    std::pair<V*, V*> d0 = e.divmod(ts, ext[0], mg[0].first, mg[0].second);
    V* t1 = d0.first;
    V* c0 = d0.second;
    std::pair<V*, V*> d1 = e.divmod(t1, ext[1], mg[1].first, mg[1].second);
    V* c2 = d1.first;
    V* c1 = d1.second;
    std::map<std::string, V*> ioff;
    for(const auto& n : names)
    {
        V* o = e.off(c0, ST[n][0]);
        V* o1 = e.off(c1, ST[n][1]);
        o = rocke_b_add(b, o, o1);
        V* o2 = e.off(c2, ST[n][2]);
        ioff[n] = rocke_b_add(b, o, o2);
    }
    V* in_lim = nullptr;
    if(!lim.empty())
    {
        V* l0 = rocke_b_cmp_lt(b, c0, lim[0]);
        V* l1 = rocke_b_cmp_lt(b, c1, lim[1]);
        V* l01 = rocke_b_land(b, l0, l1);
        V* l2 = rocke_b_cmp_lt(b, c2, lim[2]);
        in_lim = rocke_b_land(b, l01, l2);
    }
    V* zero = e.cf(0.0);
    std::map<std::string, V*> nz;
    for(const auto& n : ins)
    {
        V* f0 = rocke_b_fcmp(b, "one", S[n][0], zero);
        V* f1 = rocke_b_fcmp(b, "one", S[n].back(), zero);
        nz[n] = rocke_b_lor(b, f0, f1);
    }

    V* by = rocke_b_block_id_y(b);
    rocke_for_t rows = rocke_b_scf_for(b, by, n_out, ystep, "row");
    rocke_b_region_enter(b, rows.body);
    {
        V* r = rows.iv;
        std::vector<V*> coords;
        for(int i : {3, 4, 5, 6})
        {
            std::pair<V*, V*> qr = e.divmod(r, ext[i], mg[i].first, mg[i].second);
            r = qr.first;
            coords.push_back(qr.second);
        }
        coords.push_back(r);
        std::map<std::string, V*> roff;
        for(const auto& n : names)
        {
            V* o = ioff[n];
            for(size_t j = 0; j < coords.size(); ++j)
            {
                V* oj = e.off(coords[j], ST[n][3 + j]);
                o = rocke_b_add(b, o, oj);
            }
            roff[n] = o;
        }
        V* okv = t_ok;
        V* ok_load = nullptr;
        if(!lim.empty())
        {
            V* rl = in_lim;
            for(size_t j = 0; j < coords.size(); ++j)
            {
                V* lj = rocke_b_cmp_lt(b, coords[j], lim[3 + j]);
                rl = rocke_b_land(b, rl, lj);
            }
            ok_load = rl;
        }
        scf_if(b, okv, [&]() {
            std::map<std::string, std::vector<V*>> vals;
            int ncomp = spec->cplx ? 2 : 1;
            if(spec->math)
            {
                /* _math_inputs: every input loaded, then the operators of all of them */
                std::vector<V*> conds, vs, codes;
                for(const auto& n : ins)
                {
                    V* cond = nz[n];
                    if(!lim.empty())
                        cond = rocke_b_land(b, cond, ok_load);
                    std::vector<V*> v = e.when(cond, {zero}, [&](const std::vector<V*>&) {
                        return e.load(P[n], roff[n], dts[n], false);
                    });
                    conds.push_back(cond);
                    vs.push_back(v[0]);
                }
                for(const auto& n : ins)
                    codes.push_back(OPS[n]);
                std::vector<V*> xs = e.unary_inputs(vs, codes);
                for(size_t i = 0; i < ins.size(); ++i)
                {
                    V* p = rocke_b_fmul(b, S[ins[i]][0], xs[i]);
                    vals[ins[i]] = {rocke_b_select(b, conds[i], p, zero)};
                }
            }
            if(!spec->math)
                for(const auto& n : ins)
                {
                    std::vector<V*> zeros(ncomp, zero);
                    V* cond = nz[n];
                    if(!lim.empty())
                        cond = rocke_b_land(b, cond, ok_load);
                    std::vector<V*> v = e.when(cond, zeros, [&](const std::vector<V*>&) {
                        return e.load(P[n], roff[n], dts[n], spec->cplx);
                    });
                    if(spec->cplx)
                    {
                        V* re = v[0];
                        V* im = v[1];
                        V* k9 = e.c32(OP_CONJ);
                        V* isc = rocke_b_cmp_eq(b, OPS[n], k9);
                        V* nim = rocke_b_fneg(b, im);
                        im = rocke_b_select(b, isc, nim, im);
                        std::pair<V*, V*> m = e.cmul(S[n][0], S[n][1], re, im);
                        V* s0 = rocke_b_select(b, cond, m.first, zero);
                        V* s1 = rocke_b_select(b, cond, m.second, zero);
                        vals[n] = {s0, s1};
                    }
                    else
                    {
                        V* x = e.unary(v[0], OPS[n]);
                        V* p = rocke_b_fmul(b, S[n][0], x);
                        vals[n] = {rocke_b_select(b, cond, p, zero)};
                    }
                }
            std::vector<V*> out;
            if(spec->cplx)
            {
                out = vals["A"];
                if(spec->arity == 2)
                {
                    V* o0 = rocke_b_fadd(b, out[0], vals["C"][0]);
                    V* o1 = rocke_b_fadd(b, out[1], vals["C"][1]);
                    out = {o0, o1};
                }
            }
            else if(spec->arity == 1)
                out = vals["A"];
            else if(spec->arity == 2)
                out = {e.binary(vals["A"][0], vals["C"][0], OPS["AB"])};
            else
            {
                V* ab = e.binary(vals["A"][0], vals["B"][0], OPS["AB"]);
                out = {e.binary(ab, vals["C"][0], OPS["ABC"])};
            }
            e.store(P["D"], roff["D"], out, spec->d);
        });
    }
    rocke_b_region_leave(b);
    if(!rocke_ir_builder_ok(b))
        return nullptr;
    return rocke_ir_builder_kernel(b);
}

rocke_kernel_def_t* rocke_build_htaot_ew(rocke_ir_builder_t* b, const rocke_htaot_ew_spec_t* spec)
{
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* { return ew_impl(b, spec); });
}

} /* extern "C" */
