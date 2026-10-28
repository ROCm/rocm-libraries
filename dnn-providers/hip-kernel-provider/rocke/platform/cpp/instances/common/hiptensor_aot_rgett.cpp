// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * hiptensor_aot_rgett.cpp -- C++ mirror of
 * library/kernels/common/hiptensor_aot/rgett.py (RgettSpec, geometry_names,
 * RtGettHooks, build_rgett, rgett_classes): the shape-generic ring GETT, the
 * tensor GETT ring body with its index maps as kernel arguments (or, for
 * hipTensor's JIT, as constants).
 */
#include "rocke/instance_hiptensor_aot_internal.h"
#include "rocke/instance_tensor_gett.h"

#include "rocke/error_boundary.hpp"
#include "rocke/ir_internal.h"

#include <stdio.h>
#include <string.h>

#include <algorithm>
#include <map>
#include <string>
#include <vector>

using namespace rocke::htaot;
using rocke::gett::Digits;

namespace
{

using V = rocke_value_t;

constexpr int SLOTS_ROW = ROCKE_HTAOT_RGETT_SLOTS_ROW;
constexpr int SLOTS_KL = ROCKE_HTAOT_RGETT_SLOTS_KL;
constexpr int SLOTS_KT = ROCKE_HTAOT_RGETT_SLOTS_KT;

/* SPACES: (prefix, stride names, slots) of every digit space */
struct Space
{
    const char* prefix;
    std::vector<std::string> strides;
    int n;
};

const std::vector<Space>& spaces()
{
    static const std::vector<Space> s = {
        {"ra", {"S"}, SLOTS_ROW},
        {"rb", {"S"}, SLOTS_ROW},
        {"dr", {"S"}, SLOTS_ROW},
        {"dc", {"S"}, SLOTS_ROW},
        {"kl", {"A", "B"}, SLOTS_KL},
        {"kt", {"A", "B"}, SLOTS_KT},
    };
    return s;
}

const std::vector<std::string>& geometry_names()
{
    static const std::vector<std::string> names = [] {
        std::vector<std::string> out;
        for(const auto& sp : spaces())
        {
            for(int i = 0; i < sp.n; ++i)
            {
                std::string p = sp.prefix;
                std::string I = std::to_string(i);
                out.push_back(p + "e" + I);
                if(i < sp.n - 1)
                {
                    out.push_back(p + "m" + I);
                    out.push_back(p + "s" + I);
                }
                for(const auto& st : sp.strides)
                    out.push_back(p + st + I);
            }
        }
        out.push_back("kps");
        out.push_back("kpl");
        return out;
    }();
    return names;
}

V* ci(rocke_ir_builder_t* b, int64_t v)
{
    return rocke_b_const_i32(b, v);
}

bool set_reason(char* reason, size_t cap, const std::string& msg)
{
    if(reason != nullptr && cap > 0)
        snprintf(reason, cap, "%s", msg.c_str());
    return false;
}

std::string kernel_name_of(const rocke_htaot_rgett_spec_t* s)
{
    std::string n = std::string("hiptensor_aot_rgett_") + (s->dtype ? s->dtype : "") + "_"
                    + (s->bilinear ? "bil" : "scl") + "_a" + std::to_string(s->vec_a) + "_b"
                    + std::to_string(s->vec_b) + "_d" + std::to_string(s->vec_d)
                    + (s->epi_pair ? "p" : "") + "_t" + std::to_string(s->tile_k) + "s"
                    + std::to_string(s->dtl_stages) + "w" + std::to_string(s->chiplet_wgm) + "c"
                    + std::to_string(s->chiplet_chunk_size);
    if(s->wide_d)
        n += "_wd";
    if(s->c_prefetch)
        n += "_cp" + std::to_string(s->c_prefetch);
    if(s->slots)
        n += "_n" + std::to_string(s->slots);
    return n;
}

rocke_status_t copy_out(const std::string& n, char* out, size_t cap)
{
    if(out == nullptr || n.size() + 1 > cap)
        return ROCKE_ERR_VALUE;
    memcpy(out, n.c_str(), n.size() + 1);
    return ROCKE_OK;
}

/* RtGettHooks: the ring GETT hooks with the index maps of the class geometry
 * (kernel arguments, or the constants `geom`). */
struct RtGettHooks : rocke_gett_hooks
{
    RtGettHooks(const rocke::gett::Plan& plan,
                const rocke::gett::Knobs& knobs,
                const rocke_gemm_universal_spec_t& spec,
                const char* dtype,
                const int64_t* geom,
                bool wide,
                int slots)
        : rocke_gett_hooks(plan, knobs, spec, dtype)
        , wide_(wide)
        , slots_(slots)
        , geom_(geom)
    {
        const auto& names = geometry_names();
        for(size_t i = 0; i < names.size(); ++i)
            index_[names[i]] = (int)i;
    }

    bool wide_;
    int slots_;
    const int64_t* geom_;
    std::map<std::string, int> index_;
    std::map<std::string, V*> g_;
    V* dimM_ = nullptr;
    V* dimN_ = nullptr;

    void add_params(rocke_ir_builder_t* b) override
    {
        rocke_gett_hooks::add_params(b);
        dimM_ = rocke_b_get_param(b, "M");
        dimN_ = rocke_b_get_param(b, "N");
        if(geom_ == nullptr)
            for(const auto& name : geometry_names())
                g_[name] = rocke_b_param(b, name.c_str(), rocke_i32(), nullptr);
    }

    V* v(rocke_ir_builder_t* b, const std::string& name)
    {
        if(geom_ == nullptr)
            return g_.at(name);
        return ci(b, geom_[index_.at(name)]);
    }

    /* (offset per stride name, digit values) of index x over a digit space */
    std::map<std::string, V*> space(rocke_ir_builder_t* b,
                                    V* x,
                                    const std::string& prefix,
                                    const std::vector<std::string>& strides,
                                    std::vector<V*>* digits_out)
    {
        int n = 0;
        for(const auto& sp : spaces())
            if(prefix == sp.prefix)
                n = sp.n;
        if(slots_)
            n = std::min(n, slots_);
        std::map<std::string, V*> offs;
        for(const auto& st : strides)
            offs[st] = nullptr;
        V* rest = x;
        for(int i = 0; i < n; ++i)
        {
            std::string I = std::to_string(i);
            V* e = v(b, prefix + "e" + I);
            V* d;
            V* q = nullptr;
            if(i == n - 1)
                d = rest;
            else if(geom_ == nullptr)
            {
                /* magic-number division (common.Emit.udiv) */
                V* m = v(b, prefix + "m" + I);
                V* s = v(b, prefix + "s" + I);
                V* hi = rocke_b_umul_hi_i32(b, rest, m);
                q = rocke_b_lshr(b, hi, s);
                V* c0 = ci(b, 0);
                V* z = rocke_b_cmp_eq(b, m, c0);
                q = rocke_b_select(b, z, rest, q);
                V* qe = rocke_b_mul(b, q, e);
                d = rocke_b_sub(b, rest, qe);
            }
            else
            {
                q = rocke_b_div(b, rest, e);
                d = rocke_b_mod(b, rest, e);
            }
            if(digits_out != nullptr)
                digits_out->push_back(d);
            for(const auto& st : strides)
            {
                V* sv = v(b, prefix + st + I);
                V* t = rocke_b_mul(b, d, sv);
                offs[st] = offs[st] == nullptr ? t : rocke_b_add(b, offs[st], t);
            }
            rest = q;
        }
        return offs;
    }

    V* digits(rocke_ir_builder_t* b, V* x, const Digits& dg) override
    {
        if(dg.rt)
            return space(b, x, dg.prefix, {dg.stride}, nullptr).at(dg.stride);
        return rocke_gett_hooks::digits(b, x, dg);
    }

    bool lane_split(char) override
    {
        return false;
    }

    V* klocal_expr(rocke_ir_builder_t* b, char nm, V* k) override
    {
        std::string s(1, nm);
        return space(b, k, "kl", {s}, nullptr).at(s);
    }

    V* kpad_ok(rocke_ir_builder_t* b, const std::vector<V*>& digits)
    {
        V* kps = v(b, "kps");
        V* kpl = v(b, "kpl");
        V* ok = nullptr;
        for(size_t i = 0; i < digits.size(); ++i)
        {
            V* ciw = ci(b, (int64_t)i);
            V* ne = rocke_b_cmp_ne(b, kps, ciw);
            V* lt = rocke_b_cmp_lt(b, digits[i], kpl);
            V* t = rocke_b_lor(b, ne, lt);
            ok = ok == nullptr ? t : rocke_b_land(b, ok, t);
        }
        return ok;
    }

    V* k_valid(rocke_ir_builder_t* b, V* k) override
    {
        std::vector<V*> dg;
        space(b, k, "kl", {}, &dg);
        return kpad_ok(b, dg);
    }

    std::vector<V*> tile_voffs(rocke_ir_builder_t* b, char nm, V* mn0) override
    {
        const auto& op = plan.op(nm);
        V* rows = nm == 'A' ? dimM_ : dimN_;
        int np = passes(nm);
        std::map<std::vector<int64_t>, int> groups;
        std::vector<int> rep;
        for(int pp = 0; pp < np; ++pp)
        {
            std::vector<int64_t> key;
            for(int t = 0; t < bs; ++t)
                key.push_back(lane_geom(nm, pp, t).second);
            if(groups.find(key) == groups.end())
                groups[key] = pp;
            rep.push_back(groups[key]);
        }
        std::map<int, std::pair<V*, V*>> rowoff;
        std::vector<V*> out;
        std::string ns(1, nm);
        for(int pp = 0; pp < np; ++pp)
        {
            std::pair<V*, V*> kr = lane_expr(b, nm, pp);
            V* k_v = kr.first;
            V* r_v = kr.second;
            int gp = rep[pp];
            if(rowoff.find(gp) == rowoff.end())
            {
                if(gp != pp)
                    r_v = lane_expr(b, nm, gp).second;
                V* row = rocke_b_add(b, mn0, r_v);
                V* ok = rocke_b_cmp_lt(b, row, rows);
                V* c0 = ci(b, 0);
                V* sel = rocke_b_select(b, ok, row, c0);
                V* ro = digits(b, sel, op.row_digits);
                rowoff[gp] = {ro, ok};
            }
            V* ro = rowoff[gp].first;
            V* ok = rowoff[gp].second;
            std::vector<V*> dg;
            std::map<std::string, V*> offs = space(b, k_v, "kl", {ns}, &dg);
            V* kp = kpad_ok(b, dg);
            ok = rocke_b_land(b, ok, kp);
            V* sum = rocke_b_add(b, ro, offs.at(ns));
            V* c2 = ci(b, 2);
            V* vv = rocke_b_mul(b, sum, c2);
            V* oob = ci(b, rocke::gett::OOB);
            out.push_back(rocke_b_select(b, ok, vv, oob));
        }
        return out;
    }

    V* ktile_off_bytes(rocke_ir_builder_t* b, char nm, V* k_ld) override
    {
        V* cT = ci(b, T);
        V* kt = rocke_b_div(b, k_ld, cT);
        std::string s(1, nm);
        V* off = space(b, kt, "kt", {s}, nullptr).at(s);
        V* c2 = ci(b, 2);
        V* x = rocke_b_mul(b, off, c2);
        return rocke_b_to_sgpr_u32(b, x);
    }

    int64_t d_bytes() const override
    {
        return wide_ ? rocke::gett::MAX_D_BYTES : rocke::gett::NARROW_D_BYTES;
    }

    bool wide_d() const override
    {
        return wide_;
    }

    DPart d_part(rocke_ir_builder_t* b, V* idx, const Digits& dg, int64_t, V* big) override
    {
        V* lim = &dg == &plan.d_row_digits ? dimM_ : dimN_;
        DPart out;
        V* e = digits(b, idx, dg);
        if(wide_)
        {
            out.e = e;
            out.ok = rocke_b_cmp_lt(b, idx, lim);
            return out;
        }
        V* c2 = ci(b, 2);
        V* off = rocke_b_mul(b, e, c2);
        V* lt = rocke_b_cmp_lt(b, idx, lim);
        out.off = rocke_b_select(b, lt, off, big);
        return out;
    }
};

Digits rt(const char* prefix)
{
    Digits d;
    d.rt = true;
    d.prefix = prefix;
    d.stride = "S";
    return d;
}

/* the class's stand-in plan: its structure with runtime digit spaces */
rocke::gett::Plan stand_in_plan(const rocke_htaot_rgett_spec_t* s)
{
    rocke::gett::Plan p;
    p.ops[0].name = "A";
    p.ops[0].rows = 1;
    p.ops[0].row_digits = rt("ra");
    p.ops[0].vec = s->vec_a;
    p.ops[0].nbytes = 0;
    p.ops[1].name = "B";
    p.ops[1].rows = 1;
    p.ops[1].row_digits = rt("rb");
    p.ops[1].vec = s->vec_b;
    p.ops[1].nbytes = 0;
    p.swapped = false;
    p.M = p.N = p.K = 1;
    p.tile_k = s->tile_k;
    p.d_row_digits = rt("dr");
    p.d_col_digits = rt("dc");
    p.vec_d = s->vec_d;
    p.vec_d8 = s->epi_pair;
    p.has_k_pad = false;
    p.kc = false;
    p.vec_dc8 = false;
    return p;
}

rocke_kernel_def_t* rgett_impl(rocke_ir_builder_t* b,
                               const rocke_htaot_rgett_spec_t* spec,
                               const char* arch,
                               const int64_t* geom)
{
    char reason[256] = {0};
    std::string err;
    if(!rocke_htaot_rgett_check(spec, reason, sizeof reason))
        err = reason;
    else if(arch == nullptr || strcmp(arch, "gfx950") != 0)
        err = "the ring GETT body needs ('gfx950',)";
    std::string name = spec != nullptr ? kernel_name_of(spec) : std::string("hiptensor_aot_rgett");
    if(geom != nullptr)
    {
        size_t at = name.find("hiptensor_aot_rgett");
        if(at != std::string::npos)
            name.replace(at, strlen("hiptensor_aot_rgett"), "hiptensor_jit_rgett");
    }
    if(!err.empty())
    {
        if(rocke_ir_builder_init(b, name.c_str()) != ROCKE_OK)
            return nullptr;
        return fail(b, err.c_str());
    }
    std::string dt = streq(spec->dtype, "f16") ? "fp16" : "bf16";
    rocke_gemm_universal_spec_t body = rocke::gett::body_spec(dt.c_str(), spec->tile_k);
    body.trait.dtl_stages = spec->dtl_stages;
    body.trait.chiplet_wgm = spec->chiplet_wgm;
    body.trait.chiplet_chunk_size = spec->chiplet_chunk_size;
    body.block_size = 0;
    rocke_gemm_universal_spec_finalize(&body);
    body.name = name.c_str();
    rocke::gett::Knobs knobs;
    knobs.bilinear = spec->bilinear;
    knobs.epi_pair = spec->epi_pair;
    knobs.c_prefetch = spec->c_prefetch;
    RtGettHooks hooks(
        stand_in_plan(spec), knobs, body, spec->dtype, geom, spec->wide_d, spec->slots);
    if(!hooks.ok())
    {
        if(rocke_ir_builder_init(b, name.c_str()) != ROCKE_OK)
            return nullptr;
        return fail(b, hooks.error().c_str());
    }
    body._gett = &hooks;
    return rocke_build_universal_gemm_new(b, &body, arch);
}

} // namespace

extern "C" {

rocke_htaot_rgett_spec_t rocke_htaot_rgett_spec_default(void)
{
    rocke_htaot_rgett_spec_t s;
    memset(&s, 0, sizeof s);
    s.dtype = "f16";
    s.bilinear = false;
    s.vec_a = 8;
    s.vec_b = 8;
    s.vec_d = 4;
    s.epi_pair = false;
    s.wide_d = false;
    s.c_prefetch = 0;
    s.slots = 0;
    s.tile_k = 32;
    s.dtl_stages = 3;
    s.chiplet_wgm = 8;
    s.chiplet_chunk_size = 64;
    return s;
}

bool rocke_htaot_rgett_check(const rocke_htaot_rgett_spec_t* s, char* reason, size_t cap)
{
    if(s == nullptr)
        return set_reason(reason, cap, "null spec");
    if(!streq(s->dtype, "f16") && !streq(s->dtype, "bf16"))
        return set_reason(reason, cap, "dtype must be f16 or bf16");
    if(!(s->vec_a == 8 || s->vec_a == 2) || !(s->vec_b == 8 || s->vec_b == 2))
        return set_reason(reason, cap, "vec_a / vec_b must be 8 or 2");
    if(!(s->vec_d == 1 || s->vec_d == 2 || s->vec_d == 4))
        return set_reason(reason, cap, "vec_d must be 1, 2 or 4");
    if(s->epi_pair && s->vec_d != 4)
        return set_reason(reason, cap, "epi_pair needs vec_d 4");
    if(!(s->tile_k == 32 || s->tile_k == 64))
        return set_reason(reason, cap, "tile_k must be 32 or 64");
    if(s->c_prefetch && !s->bilinear)
        return set_reason(reason, cap, "c_prefetch needs bilinear");
    if(s->slots
       && !(1 <= s->slots && s->slots <= std::min(SLOTS_KL, std::min(SLOTS_KT, SLOTS_ROW))))
        return set_reason(reason, cap, "slots must be 0 or 1 .. the smallest digit space");
    if(reason != nullptr && cap > 0)
        reason[0] = '\0';
    return true;
}

rocke_status_t
    rocke_htaot_rgett_kernel_name(const rocke_htaot_rgett_spec_t* s, char* out, size_t cap)
{
    if(s == nullptr)
        return ROCKE_ERR_VALUE;
    return copy_out(kernel_name_of(s), out, cap);
}

rocke_status_t rocke_htaot_rgett_key(const rocke_htaot_rgett_spec_t* s, char* out, size_t cap)
{
    if(s == nullptr)
        return ROCKE_ERR_VALUE;
    std::string n = kernel_name_of(s);
    n.replace(0, strlen("hiptensor_aot_rgett"), "hiptensor_jit_rgett");
    return copy_out("rgett:" + n, out, cap);
}

int rocke_htaot_rgett_geometry_count(void)
{
    return (int)geometry_names().size();
}

const char* rocke_htaot_rgett_geometry_name(int i)
{
    const auto& n = geometry_names();
    return i >= 0 && i < (int)n.size() ? n[(size_t)i].c_str() : nullptr;
}

int rocke_htaot_rgett_classes(const char* dtype, rocke_htaot_rgett_spec_t* out, int cap)
{
    static const char* const both[] = {"f16", "bf16"};
    static const int buckets[] = {2, 4, 0}; /* SLOT_BUCKETS */
    static const struct
    {
        int vd;
        bool pair;
    } dws[] = {{4, true}, {4, false}, {2, false}, {1, false}};
    int nd = dtype == nullptr ? 2 : 1;
    int count = 0;
    for(int di = 0; di < nd; ++di)
    {
        const char* dt = dtype == nullptr ? both[di] : (streq(dtype, "bf16") ? "bf16" : "f16");
        for(int bil = 0; bil < 2; ++bil)
        {
            int cpf = bil ? ROCKE_HTAOT_RGETT_BILINEAR_C_PREFETCH : 0;
            for(int va : {8, 2})
                for(int vb : {8, 2})
                    for(const auto& dw : dws)
                        for(int n : buckets)
                        {
                            if(bil && dw.pair && n == 0)
                                continue;
                            int nw = va == 8 && vb == 8 ? 2 : 1;
                            for(int w = 0; w < nw; ++w)
                            {
                                if(out != nullptr && count < cap)
                                {
                                    rocke_htaot_rgett_spec_t s = rocke_htaot_rgett_spec_default();
                                    s.dtype = dt;
                                    s.bilinear = bil != 0;
                                    s.vec_a = va;
                                    s.vec_b = vb;
                                    s.vec_d = dw.vd;
                                    s.epi_pair = dw.pair;
                                    s.wide_d = w != 0;
                                    s.c_prefetch = cpf;
                                    s.slots = n;
                                    out[count] = s;
                                }
                                ++count;
                            }
                        }
        }
    }
    return count;
}

rocke_kernel_def_t* rocke_build_htaot_rgett(rocke_ir_builder_t* b,
                                            const rocke_htaot_rgett_spec_t* s,
                                            const char* arch,
                                            const int64_t* geom)
{
    return ckc::guard_builder(
        b, [&]() -> rocke_kernel_def_t* { return rgett_impl(b, s, arch, geom); });
}

} /* extern "C" */
