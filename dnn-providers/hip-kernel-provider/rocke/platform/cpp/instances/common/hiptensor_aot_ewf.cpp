// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * hiptensor_aot_ewf.cpp -- C++ mirror of library/kernels/common/hiptensor_aot/ewf.py
 * (EwfSpec, build_ewf): the flat elementwise kernel of tensors that are all contiguous with the
 * same layout. Every builder call follows the Python builder's order (the hiptensor_aot_ewf
 * parity family keeps the two byte-identical).
 */
#include "rocke/instance_hiptensor_aot_internal.h"

#include "rocke/error_boundary.hpp"
#include "rocke/ir_internal.h"

#include <stdio.h>
#include <string.h>

#include <map>
#include <string>
#include <tuple>
#include <vector>

using namespace rocke::htaot;

namespace
{

bool set_reason(char* reason, size_t cap, const char* msg)
{
    if(reason != nullptr && cap > 0)
        snprintf(reason, cap, "%s", msg);
    return false;
}

/* EwfSpec.base() */
rocke_htaot_ew_spec_t base_of(const rocke_htaot_ewf_spec_t* s)
{
    rocke_htaot_ew_spec_t e = rocke_htaot_ew_spec_default();
    e.arity = s->arity;
    e.a = s->a;
    e.d = s->d;
    e.b = s->b;
    e.c = s->c;
    e.block = s->block;
    return e;
}

const char* cdt_of(const rocke_htaot_ewf_spec_t* s)
{
    const char* dts[4] = {s->a, s->b, s->c, s->d};
    for(const char* x : dts)
        if(streq(x, "f64"))
            return "f64";
    return "f32";
}

/* ewf._runs (common.max_vec) */
std::vector<int> runs(int n, int nbytes)
{
    int mx = max_vec(nbytes);
    if(mx <= 0 || n <= 0)
        return {}; /* no vector width in the catalog (the Python max_vec raises) */
    int w = n < mx ? n : mx;
    return std::vector<int>(n / w, w);
}

std::string ewf_name(const rocke_htaot_ewf_spec_t* s)
{
    rocke_htaot_ew_spec_t base = base_of(s);
    char buf[160];
    if(rocke_htaot_ew_kernel_name(&base, buf, sizeof buf) != ROCKE_OK)
        return "";
    std::string n = buf;
    const std::string from = "hiptensor_aot_ew";
    size_t at = n.find(from);
    if(at != std::string::npos)
        n.replace(at, from.size(), "hiptensor_aot_ewf");
    at = n.find("_i32_");
    if(at != std::string::npos)
        n.replace(at, 5, "_");
    n += "_v" + std::to_string(rocke_htaot_ewf_vec(s)) + "_x"
         + std::to_string(ROCKE_HTAOT_EWF_UNROLL);
    if(s->nt)
        n += "_nt";
    return n;
}

} // namespace

extern "C" {

rocke_htaot_ewf_spec_t rocke_htaot_ewf_spec_default(void)
{
    rocke_htaot_ewf_spec_t s{};
    s.arity = 1;
    s.a = "f32";
    s.d = "f32";
    s.b = nullptr;
    s.c = nullptr;
    s.block = 256;
    s.nt = false;
    return s;
}

int rocke_htaot_ewf_vec(const rocke_htaot_ewf_spec_t* s)
{
    int nb = 0;
    if(s == nullptr || !dtype_of(s->d, nullptr, &nb))
        return 0;
    return max_vec(nb);
}

bool rocke_htaot_ewf_check(const rocke_htaot_ewf_spec_t* s, char* reason, size_t cap)
{
    if(s == nullptr)
        return set_reason(reason, cap, "null spec");
    rocke_htaot_ew_spec_t base = base_of(s);
    if(!rocke_htaot_ew_check(&base, reason, cap))
        return false;
    if(s->block != 64 && s->block != 128 && s->block != 256 && s->block != 512 && s->block != 1024)
        return set_reason(reason, cap, "block must be 64, 128, 256, 512 or 1024");
    if(reason != nullptr && cap > 0)
        reason[0] = '\0';
    return true;
}

rocke_status_t rocke_htaot_ewf_kernel_name(const rocke_htaot_ewf_spec_t* s, char* out, size_t cap)
{
    if(s == nullptr || out == nullptr)
        return ROCKE_ERR_VALUE;
    std::string n = ewf_name(s);
    if(n.empty() || n.size() + 1 > cap)
        return ROCKE_ERR_VALUE;
    memcpy(out, n.c_str(), n.size() + 1);
    return ROCKE_OK;
}

static rocke_kernel_def_t* ewf_impl(rocke_ir_builder_t* b, const rocke_htaot_ewf_spec_t* spec)
{
    if(b == nullptr || spec == nullptr)
        return nullptr;
    char reason[160];
    bool valid = rocke_htaot_ewf_check(spec, reason, sizeof reason);
    std::string kname = valid ? ewf_name(spec) : std::string();
    if(rocke_ir_builder_init(b, kname.empty() ? "hiptensor_aot_ewf" : kname.c_str()) != ROCKE_OK)
        return nullptr;
    if(!valid)
        return fail(b, reason);
    const int vec = rocke_htaot_ewf_vec(spec);
    const int U = ROCKE_HTAOT_EWF_UNROLL;
    rocke_mem_opts_t mo = ROCKE_MEM_OPTS_INIT;
    mo.temporal_hint = spec->nt ? ROCKE_TEMPORAL_STREAMING : ROCKE_TEMPORAL_DEFAULT;
    rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", spec->block);
    Emit e(b, false, cdt_of(spec), false);
    std::vector<std::string> names;
    if(spec->arity == 1)
        names = {"A", "D"};
    else if(spec->arity == 2)
        names = {"A", "C", "D"};
    else
        names = {"A", "B", "C", "D"};
    std::vector<std::string> ins(names.begin(), names.end() - 1);
    std::map<std::string, const char*> dts
        = {{"A", spec->a}, {"B", spec->b}, {"C", spec->c}, {"D", spec->d}};
    std::map<std::string, V*> P;
    for(const auto& n : names)
        P[n] = ptr(b, n.c_str(), dts[n], n != "D");
    std::map<std::string, V*> S;
    for(const auto& n : ins)
        S[n] = e.scalar_param(n == "A" ? "alpha" : n == "B" ? "beta" : "gamma");
    std::map<std::string, V*> OPS;
    for(const auto& n : ins)
        OPS[n] = param(b, "op" + n, rocke_i32());
    if(spec->arity >= 2)
        OPS["AB"] = param(b, "opAB", rocke_i32());
    if(spec->arity == 3)
        OPS["ABC"] = param(b, "opABC", rocke_i32());
    V* n_vec = param(b, "n", rocke_i32());

    V* tid = rocke_b_thread_id_x(b);
    V* bx = rocke_b_block_id_x(b);
    V* cbu = e.c32(int64_t(spec->block) * U);
    V* bxu = rocke_b_mul(b, bx, cbu);
    V* t0 = rocke_b_add(b, bxu, tid);
    std::vector<std::pair<V*, V*>> lanes; // (in range, element offset)
    for(int u = 0; u < U; ++u)
    {
        V* t = t0;
        if(u != 0)
        {
            V* cu = e.c32(int64_t(u) * spec->block);
            t = rocke_b_add(b, t0, cu);
        }
        V* ok = rocke_b_cmp_lt(b, t, n_vec);
        V* z0 = e.c32(0);
        V* ts = rocke_b_select(b, ok, t, z0);
        V* cv = e.c32(vec);
        lanes.push_back({ok, rocke_b_mul(b, ts, cv)});
    }

    V* zero = e.cf(0.0);
    std::map<std::string, V*> nz;
    for(const auto& n : ins)
        nz[n] = rocke_b_fcmp(b, "one", S[n], zero);
    V* allnz = nz[ins[0]];
    for(size_t i = 1; i < ins.size(); ++i)
        allnz = rocke_b_land(b, allnz, nz[ins[i]]);

    auto load_vec = [&](const std::string& n, V* off) {
        const char* dt = dts[n];
        const rocke_type_t* ty = nullptr;
        int nb = 0;
        dtype_of(dt, &ty, &nb);
        std::vector<V*> out;
        int done = 0;
        for(int w : runs(vec, nb))
        {
            V* o = off;
            if(done != 0)
            {
                V* k = e.c32(done);
                o = rocke_b_add(b, off, k);
            }
            std::pair<V*, V*> pi = e.addr(P[n], o, nb);
            V* v = rocke_b_global_load_vN_ex(b, pi.first, pi.second, ty, w, w * nb, &mo);
            for(int j = 0; j < w; ++j)
            {
                V* x = rocke_b_vec_extract(b, v, j);
                out.push_back(e.to_c(x, dt));
            }
            done += w;
        }
        return out;
    };
    auto store_vec = [&](V* off, const std::vector<V*>& vals) {
        const char* dt = spec->d;
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
                V* k = e.c32(done);
                o = rocke_b_add(b, off, k);
            }
            std::pair<V*, V*> pi = e.addr(P["D"], o, nb);
            V* pk = rocke_b_vec_pack(b, outs.data() + done, w, ty);
            rocke_b_global_store_vN_ex(b, pi.first, pi.second, pk, w, w * nb, &mo);
            done += w;
        }
    };
    auto body = [&](bool every_scale) {
        std::vector<std::tuple<V*, V*, std::vector<V*>>> results;
        for(const auto& lane : lanes)
        {
            V* off = lane.second;
            std::map<std::string, std::vector<V*>> vals;
            for(const auto& n : ins)
            {
                if(every_scale)
                {
                    std::vector<V*> v = load_vec(n, off);
                    std::vector<V*> xs = e.unary_list(v, OPS[n]);
                    std::vector<V*> o;
                    for(V* x : xs)
                        o.push_back(rocke_b_fmul(b, S[n], x));
                    vals[n] = o;
                }
                else
                {
                    std::vector<V*> zeros(vec, zero);
                    std::vector<V*> v = e.when(
                        nz[n], zeros, [&](const std::vector<V*>&) { return load_vec(n, off); });
                    std::vector<V*> xs = e.unary_list(v, OPS[n]);
                    std::vector<V*> o;
                    for(V* x : xs)
                    {
                        V* p = rocke_b_fmul(b, S[n], x);
                        o.push_back(rocke_b_select(b, nz[n], p, zero));
                    }
                    vals[n] = o;
                }
            }
            std::vector<V*> out;
            for(int k = 0; k < vec; ++k)
            {
                if(spec->arity == 1)
                    out.push_back(vals["A"][k]);
                else if(spec->arity == 2)
                    out.push_back(e.binary(vals["A"][k], vals["C"][k], OPS["AB"]));
                else
                {
                    V* ab = e.binary(vals["A"][k], vals["B"][k], OPS["AB"]);
                    out.push_back(e.binary(ab, vals["C"][k], OPS["ABC"]));
                }
            }
            results.emplace_back(lane.first, off, out);
        }
        for(auto& r : results)
            scf_if(b, std::get<0>(r), [&]() { store_vec(std::get<1>(r), std::get<2>(r)); });
    };
    // every scale non-zero: straight-line loads; otherwise skip the inputs of zero scale
    scf_if(b, allnz, [&]() { body(true); });
    V* notall = rocke_b_lnot(b, allnz);
    scf_if(b, notall, [&]() { body(false); });
    if(!rocke_ir_builder_ok(b))
        return nullptr;
    return rocke_ir_builder_kernel(b);
}

rocke_kernel_def_t* rocke_build_htaot_ewf(rocke_ir_builder_t* b, const rocke_htaot_ewf_spec_t* s)
{
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* { return ewf_impl(b, s); });
}

} /* extern "C" */
