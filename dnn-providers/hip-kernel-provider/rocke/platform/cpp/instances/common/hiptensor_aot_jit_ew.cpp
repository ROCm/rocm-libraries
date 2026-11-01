// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * hiptensor_aot_jit_ew.cpp -- C++ mirror of
 * library/kernels/common/hiptensor_aot/jit_ew.py (EwJitSpec, build_ew_jit,
 * ewv_kernel_name, jit_vec): the problem-specialized form of the elementwise
 * kernel (geometry as constants) and the pack's vector class kernels (the same
 * geometry as kernel arguments).
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

/* jit_ew._runs (common.max_vec) */
std::vector<int> runs(int n, int nbytes)
{
    int mx = max_vec(nbytes);
    if(mx <= 0 || n <= 0)
        return {}; /* no vector width in the catalog (the Python max_vec raises) */
    int w = n < mx ? n : mx;
    return std::vector<int>(n / w, w);
}

std::string jit_name(const rocke_htaot_ew_jit_spec_t* s)
{
    char base[160];
    if(rocke_htaot_ew_kernel_name(&s->base, base, sizeof base) != ROCKE_OK)
        return "";
    std::string n = base;
    const std::string from = "hiptensor_aot_ew";
    size_t at = n.find(from);
    if(at != std::string::npos)
        n.replace(at, from.size(), "hiptensor_jit_ew");
    n += "_v" + std::to_string(s->vec);
    std::string u;
    for(int i = 0; i < s->n_unit; ++i)
        u += s->unit[i] ? "u" : "g";
    if(!u.empty())
        n += "_" + u;
    return n;
}

std::string ewv_name(const rocke_htaot_ew_jit_spec_t* s)
{
    std::string n = jit_name(s);
    const std::string from = "hiptensor_jit_ew";
    size_t at = n.find(from);
    if(at != std::string::npos)
        n.replace(at, from.size(), "hiptensor_aot_ewv");
    return n;
}

struct Axis
{
    std::string name;
    bool i64;
};

std::vector<Axis> axes(const rocke_htaot_ew_jit_spec_t* s)
{
    std::vector<std::string> names = tensors(s->base.arity);
    std::vector<Axis> out;
    for(size_t i = 0; i + 1 < names.size(); ++i)
        out.push_back({"op" + names[i], false});
    if(s->base.arity >= 2)
        out.push_back({"opAB", false});
    if(s->base.arity == 3)
        out.push_back({"opABC", false});
    out.push_back({"n_in", false});
    out.push_back({"n_out", false});
    out.push_back({"ystep", false});
    for(int i = 0; i < MAX_RANK; ++i)
        out.push_back({"e" + std::to_string(i), false});
    for(int i : {0, 1, 3, 4, 5, 6})
    {
        out.push_back({"m" + std::to_string(i), false});
        out.push_back({"s" + std::to_string(i), false});
    }
    for(const auto& n : names)
        for(int i = 0; i < MAX_RANK; ++i)
            out.push_back({"s" + n + std::to_string(i), s->base.idx64});
    if(s->base.arity == 1)
        for(int i = 0; i < MAX_RANK; ++i)
            out.push_back({"lim" + std::to_string(i), false});
    return out;
}

} // namespace

extern "C" {

int rocke_htaot_ew_jit_vec(const char* d)
{
    int nb = 0;
    if(!dtype_of(d, nullptr, &nb))
        return 0;
    return max_vec(nb);
}

bool rocke_htaot_ew_jit_check(const rocke_htaot_ew_jit_spec_t* s, char* reason, size_t cap)
{
    if(s == nullptr)
        return set_reason(reason, cap, "null spec");
    if(!rocke_htaot_ew_check(&s->base, reason, cap))
        return false;
    if(s->vec != 1 && s->vec != 2 && s->vec != 4 && s->vec != 8)
        return set_reason(reason, cap, "vec must be 1, 2, 4 or 8");
    if(s->vec > 1 && s->base.cplx)
        return set_reason(reason, cap, "complex kernels have vec 1");
    if(s->vec > 1 && s->base.math)
        for(int i = 0; i < s->n_unit; ++i)
            if(!s->unit[i])
                return set_reason(
                    reason, cap, "math kernels have vector classes only with every input a vector");
    int n_in = s->base.arity;
    if(s->vec == 1 && s->n_unit > 0)
        return set_reason(reason, cap, "unit flags need vec > 1");
    if(s->vec > 1 && s->n_unit != n_in)
    {
        char msg[64];
        snprintf(msg, sizeof msg, "unit needs %d flags", n_in);
        return set_reason(reason, cap, msg);
    }
    if(reason != nullptr && cap > 0)
        reason[0] = '\0';
    return true;
}

rocke_status_t
    rocke_htaot_ew_jit_kernel_name(const rocke_htaot_ew_jit_spec_t* s, char* out, size_t cap)
{
    if(s == nullptr || out == nullptr)
        return ROCKE_ERR_VALUE;
    std::string n = jit_name(s);
    if(n.empty() || n.size() + 1 > cap)
        return ROCKE_ERR_VALUE;
    memcpy(out, n.c_str(), n.size() + 1);
    return ROCKE_OK;
}

rocke_status_t
    rocke_htaot_ewv_kernel_name(const rocke_htaot_ew_jit_spec_t* s, char* out, size_t cap)
{
    if(s == nullptr || out == nullptr)
        return ROCKE_ERR_VALUE;
    std::string n = ewv_name(s);
    if(n.empty() || n.size() + 1 > cap)
        return ROCKE_ERR_VALUE;
    memcpy(out, n.c_str(), n.size() + 1);
    return ROCKE_OK;
}

int rocke_htaot_ew_jit_axes(const rocke_htaot_ew_jit_spec_t* s,
                            char names[][16],
                            bool* is_i64,
                            int cap)
{
    if(s == nullptr)
        return 0;
    std::vector<Axis> ax = axes(s);
    int n = 0;
    for(const Axis& a : ax)
    {
        if(n < cap)
        {
            if(names != nullptr)
                snprintf(names[n], 16, "%s", a.name.c_str());
            if(is_i64 != nullptr)
                is_i64[n] = a.i64;
        }
        ++n;
    }
    return n;
}

static rocke_kernel_def_t* ew_jit_impl(rocke_ir_builder_t* b,
                                       const rocke_htaot_ew_jit_spec_t* spec,
                                       const int64_t* geom,
                                       int n_geom)
{
    if(b == nullptr || spec == nullptr)
        return nullptr;
    char reason[160];
    bool valid = rocke_htaot_ew_jit_check(spec, reason, sizeof reason);
    std::string kname = geom != nullptr ? jit_name(spec) : ewv_name(spec);
    if(rocke_ir_builder_init(b, kname.empty() ? "hiptensor_jit_ew" : kname.c_str()) != ROCKE_OK)
        return nullptr;
    if(!valid)
        return fail(b, reason);
    std::vector<Axis> ax = axes(spec);
    if(geom != nullptr && n_geom != (int)ax.size())
        return fail(b, "geometry needs one value per axis");
    const rocke_htaot_ew_spec_t* base = &spec->base;
    const int vec = spec->vec;
    rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", base->block);
    Emit e(b, base->idx64, cdt_of(base), base->math);
    std::vector<std::string> names = tensors(base->arity);
    std::vector<std::string> ins(names.begin(), names.end() - 1);
    std::map<std::string, const char*> dts
        = {{"A", base->a}, {"B", base->b}, {"C", base->c}, {"D", base->d}};
    std::map<std::string, bool> unit;
    if(vec > 1)
        for(size_t i = 0; i < ins.size() && (int)i < spec->n_unit; ++i)
            unit[ins[i]] = spec->unit[i];
    std::map<std::string, V*> P;
    for(const auto& n : names)
        P[n] = ptr(b, n.c_str(), dts[n], n != "D");
    std::map<std::string, std::vector<V*>> S;
    for(const auto& n : ins)
    {
        if(base->cplx)
        {
            V* re = e.scalar_param((std::string(scal_of(n)) + "_re").c_str());
            V* im = e.scalar_param((std::string(scal_of(n)) + "_im").c_str());
            S[n] = {re, im};
        }
        else
            S[n] = {e.scalar_param(scal_of(n))};
    }
    std::map<std::string, V*> pv;
    std::map<std::string, int64_t> gv;
    if(geom == nullptr)
    {
        for(const Axis& a : ax)
            pv[a.name] = param(b, a.name, a.i64 ? rocke_i64() : rocke_i32());
    }
    else
        for(size_t i = 0; i < ax.size(); ++i)
            gv[ax[i].name] = geom[i];

    auto k32 = [&](const std::string& name) -> V* {
        if(geom == nullptr)
            return pv[name];
        return rocke_b_const_i32(b, gv[name]);
    };
    auto koff = [&](int64_t v) -> V* {
        return base->idx64 ? rocke_b_const_i64(b, v) : rocke_b_const_i32(b, v);
    };
    auto kst = [&](const std::string& name) -> V* {
        if(geom == nullptr)
            return pv[name];
        return base->idx64 ? rocke_b_const_i64(b, gv[name]) : rocke_b_const_i32(b, gv[name]);
    };

    std::map<std::string, V*> OPS;
    for(const auto& n : ins)
        OPS[n] = k32("op" + n);
    if(base->arity >= 2)
        OPS["AB"] = k32("opAB");
    if(base->arity == 3)
        OPS["ABC"] = k32("opABC");
    V* n_in = k32("n_in");
    V* n_out = k32("n_out");
    V* ystep = k32("ystep");
    std::vector<V*> ext;
    for(int i = 0; i < MAX_RANK; ++i)
        ext.push_back(k32("e" + std::to_string(i)));
    std::map<int, std::pair<V*, V*>> mg;
    for(int i : {0, 1, 3, 4, 5, 6})
    {
        V* m = k32("m" + std::to_string(i));
        V* s = k32("s" + std::to_string(i));
        mg[i] = {m, s};
    }
    std::map<std::string, std::vector<V*>> ST;
    for(const auto& n : names)
        for(int i = 0; i < MAX_RANK; ++i)
            ST[n].push_back(kst("s" + n + std::to_string(i)));
    std::vector<V*> lim;
    if(base->arity == 1)
        for(int i = 0; i < MAX_RANK; ++i)
            lim.push_back(k32("lim" + std::to_string(i)));

    V* tid = rocke_b_thread_id_x(b);
    V* bx = rocke_b_block_id_x(b);
    V* cb = e.c32(base->block);
    V* bxb = rocke_b_mul(b, bx, cb);
    V* t = rocke_b_add(b, bxb, tid);
    V* t_ok = rocke_b_cmp_lt(b, t, n_in);
    V* z0 = e.c32(0);
    V* ts = rocke_b_select(b, t_ok, t, z0);
    std::pair<V*, V*> d0 = e.divmod(ts, ext[0], mg[0].first, mg[0].second);
    V* t1 = d0.first;
    V* c0 = d0.second;
    if(vec > 1)
    {
        V* cv = e.c32(vec);
        c0 = rocke_b_mul(b, c0, cv);
    }
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

    auto load_vec = [&](const std::string& n, V* off) {
        const char* dt = dts[n];
        const rocke_type_t* ty = nullptr;
        int nb = 0;
        dtype_of(dt, &ty, &nb);
        std::vector<V*> out;
        if(unit[n])
        {
            int done = 0;
            for(int w : runs(vec, nb))
            {
                V* o = off;
                if(done != 0)
                {
                    V* k = koff(done);
                    o = rocke_b_add(b, off, k);
                }
                std::pair<V*, V*> pi = e.addr(P[n], o, nb);
                V* v = rocke_b_global_load_vN(b, pi.first, pi.second, ty, w, w * nb);
                for(int j = 0; j < w; ++j)
                {
                    V* x = rocke_b_vec_extract(b, v, j);
                    out.push_back(e.to_c(x, dt));
                }
                done += w;
            }
            return out;
        }
        for(int k = 0; k < vec; ++k)
        {
            V* o = off;
            if(k != 0)
            {
                V* ck = e.c32(k);
                V* ko = e.off(ck, ST[n][0]);
                o = rocke_b_add(b, off, ko);
            }
            std::vector<V*> l = e.load(P[n], o, dt);
            out.insert(out.end(), l.begin(), l.end());
        }
        return out;
    };
    auto store_vec = [&](V* off, const std::vector<V*>& vals) {
        const char* dt = base->d;
        const rocke_type_t* ty = nullptr;
        int nb = 0;
        dtype_of(dt, &ty, &nb);
        std::vector<V*> outs;
        for(V* v : vals)
            outs.push_back(e.from_c(v, dt));
        int done = 0;
        for(int w : runs(vec, nb))
        {
            V* o = off;
            if(done != 0)
            {
                V* k = koff(done);
                o = rocke_b_add(b, off, k);
            }
            std::pair<V*, V*> pi = e.addr(P["D"], o, nb);
            V* pk = rocke_b_vec_pack(b, outs.data() + done, w, ty);
            rocke_b_global_store_vN(b, pi.first, pi.second, pk, w, w * nb);
            done += w;
        }
    };

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
            int ncomp = base->cplx ? 2 : vec;
            bool math1 = base->math && vec == 1;
            if(math1)
            {
                std::vector<V*> conds, vs, codes;
                for(const auto& n : ins)
                {
                    V* cond = nz[n];
                    if(ok_load != nullptr)
                        cond = rocke_b_land(b, cond, ok_load);
                    std::vector<V*> v = e.when(cond, {zero}, [&](const std::vector<V*>&) {
                        return e.load(P[n], roff[n], dts[n]);
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
            if(!math1)
                for(const auto& n : ins)
                {
                    std::vector<V*> zeros(ncomp, zero);
                    V* cond = nz[n];
                    if(ok_load != nullptr)
                        cond = rocke_b_land(b, cond, ok_load);
                    std::vector<V*> v;
                    if(vec > 1)
                        v = e.when(cond, zeros, [&](const std::vector<V*>&) {
                            return load_vec(n, roff[n]);
                        });
                    else
                        v = e.when(cond, zeros, [&](const std::vector<V*>&) {
                            return e.load(P[n], roff[n], dts[n], base->cplx);
                        });
                    if(base->cplx)
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
                    else if(vec == 1)
                    {
                        V* x = e.unary(v[0], OPS[n]);
                        V* p = rocke_b_fmul(b, S[n][0], x);
                        vals[n] = {rocke_b_select(b, cond, p, zero)};
                    }
                    else
                    {
                        std::vector<V*> xs = e.unary_list(v, OPS[n]);
                        std::vector<V*> o;
                        for(V* x : xs)
                        {
                            V* p = rocke_b_fmul(b, S[n][0], x);
                            o.push_back(rocke_b_select(b, cond, p, zero));
                        }
                        vals[n] = o;
                    }
                }
            if(base->cplx)
            {
                std::vector<V*> out = vals["A"];
                if(base->arity == 2)
                {
                    V* o0 = rocke_b_fadd(b, out[0], vals["C"][0]);
                    V* o1 = rocke_b_fadd(b, out[1], vals["C"][1]);
                    out = {o0, o1};
                }
                e.store(P["D"], roff["D"], out, base->d);
            }
            else
            {
                std::vector<V*> out;
                for(int k = 0; k < vec; ++k)
                {
                    if(base->arity == 1)
                        out.push_back(vals["A"][k]);
                    else if(base->arity == 2)
                        out.push_back(e.binary(vals["A"][k], vals["C"][k], OPS["AB"]));
                    else
                    {
                        V* ab = e.binary(vals["A"][k], vals["B"][k], OPS["AB"]);
                        out.push_back(e.binary(ab, vals["C"][k], OPS["ABC"]));
                    }
                }
                if(vec > 1)
                    store_vec(roff["D"], out);
                else
                    e.store(P["D"], roff["D"], out, base->d);
            }
        });
    }
    rocke_b_region_leave(b);
    if(!rocke_ir_builder_ok(b))
        return nullptr;
    return rocke_ir_builder_kernel(b);
}

rocke_kernel_def_t* rocke_build_htaot_ew_jit(rocke_ir_builder_t* b,
                                             const rocke_htaot_ew_jit_spec_t* spec,
                                             const int64_t* geom,
                                             int n_geom)
{
    return ckc::guard_builder(
        b, [&]() -> rocke_kernel_def_t* { return ew_jit_impl(b, spec, geom, n_geom); });
}

} /* extern "C" */
