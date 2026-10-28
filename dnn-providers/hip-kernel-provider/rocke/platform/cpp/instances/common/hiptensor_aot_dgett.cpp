// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * hiptensor_aot_dgett.cpp -- C++ mirror of
 * library/kernels/common/hiptensor_aot/dgett.py (DgettSpec, dgett_atom,
 * dgett_config, build_dgett): the shape-generic direct GETT (a contraction in
 * one launch, no workspace).
 */
#include "rocke/instance_hiptensor_aot_internal.h"

#include "rocke/arch_target.h"
#include "rocke/error.hpp"
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

constexpr int SLOTS_MN = ROCKE_HTAOT_DGETT_SLOTS_MN;
constexpr int SLOTS_KT = ROCKE_HTAOT_DGETT_SLOTS_KT;
constexpr int SLOTS_KL = ROCKE_HTAOT_DGETT_SLOTS_KL;
constexpr int SLOTS_L = ROCKE_HTAOT_DGETT_SLOTS_L;
constexpr int THREADS = ROCKE_HTAOT_DGETT_THREADS;
constexpr int SENTINEL = -(1 << 30) * 2; /* -(1 << 31) */
constexpr int RND_F16 = 1;
constexpr int RND_BF16 = 2;

int dt_code(const char* dt)
{
    return streq(dt, "f16") ? 0 : streq(dt, "bf16") ? 1 : streq(dt, "f32") ? 2 : 3;
}

const rocke_type_t* comp_type(const char* comp)
{
    const rocke_type_t* t = nullptr;
    dtype_of(comp, &t, nullptr);
    return t;
}

/* DgettSpec.load_types() */
std::vector<const char*> load_types(const rocke_htaot_dgett_spec_t* s)
{
    if(s->cplx)
        return streq(s->comp, "f32") ? std::vector<const char*>{"f32"}
                                     : std::vector<const char*>{"f32", "f64"};
    if(streq(s->comp, "f64"))
        return {"f16", "bf16", "f32", "f64"};
    return {"f16", "bf16", "f32"};
}

/* _E.compute */
void compute(Emit& e, const char* cdt)
{
    e.f64 = streq(cdt, "f64");
    e.cty = e.f64 ? rocke_f64() : rocke_f32();
}

/* _zero_of */
V* zero_of(rocke_ir_builder_t* b, const rocke_type_t* ty)
{
    if(rocke_type_eq(ty, rocke_f64()))
        return rocke_b_const_f64(b, 0.0);
    V* z = rocke_b_const_f32(b, 0.0);
    return rocke_type_eq(ty, rocke_f32()) ? z : rocke_b_cast_f32_to(b, z, ty);
}

struct Space
{
    std::vector<V*> ext;
    std::vector<std::pair<V*, V*>> mg;
    std::map<char, std::vector<V*>> st;
};

/* _space: the arguments of a digit space */
Space space(rocke_ir_builder_t* b,
            const std::string& prefix,
            int n,
            const char* tensors,
            const rocke_type_t* st_type)
{
    Space sp;
    for(const char* t = tensors; *t; ++t)
        sp.st[*t] = {};
    for(int i = 0; i < n; ++i)
    {
        sp.ext.push_back(param(b, prefix + "e" + std::to_string(i), rocke_i32()));
        if(i < n - 1)
        {
            V* m = param(b, prefix + "m" + std::to_string(i), rocke_i32());
            V* s = param(b, prefix + "s" + std::to_string(i), rocke_i32());
            sp.mg.push_back({m, s});
        }
        for(const char* t = tensors; *t; ++t)
            sp.st[*t].push_back(param(b, prefix + std::string(1, *t) + std::to_string(i), st_type));
    }
    return sp;
}

/* _chain: offsets of coordinate x of a digit space (one per stride list) and the digits */
std::pair<std::vector<V*>, std::vector<V*>> chain(Emit& e,
                                                  V* x,
                                                  const std::vector<V*>& ext,
                                                  const std::vector<std::pair<V*, V*>>& mg,
                                                  const std::vector<std::vector<V*>>& strides)
{
    rocke_ir_builder_t* b = e.b;
    std::vector<V*> offs(strides.size(), nullptr);
    std::vector<V*> digits;
    V* rest = x;
    int n = (int)ext.size();
    for(int i = 0; i < n; ++i)
    {
        V* q = nullptr;
        V* d = nullptr;
        if(i < n - 1)
        {
            std::pair<V*, V*> qd = e.divmod(rest, ext[i], mg[i].first, mg[i].second);
            q = qd.first;
            d = qd.second;
        }
        else
            d = rest;
        digits.push_back(d);
        for(size_t j = 0; j < strides.size(); ++j)
        {
            V* t = e.off(d, strides[j][i]);
            offs[j] = offs[j] == nullptr ? t : rocke_b_add(b, offs[j], t);
        }
        rest = q;
    }
    return {offs, digits};
}

/* _to: a loaded value of type dt in the operator type (exact) */
V* to_op(Emit& e, V* x, const char* dt, const char* optype)
{
    rocke_ir_builder_t* b = e.b;
    if(streq(optype, "f64"))
    {
        if(streq(dt, "f64"))
            return x;
        V* x32 = streq(dt, "f32") ? x : rocke_b_cast_to_f32(b, x);
        return rocke_b_cvt_f32_to_f64(b, x32);
    }
    if(streq(dt, "f32"))
        return x;
    return rocke_b_cast_to_f32(b, x);
}

/* _to_comp: an operator-type value as the MFMA operand type (one rounding) */
V* to_comp(Emit& e, V* x, const rocke_htaot_dgett_spec_t* s, V* rnd)
{
    rocke_ir_builder_t* b = e.b;
    if(streq(s->comp, "f64"))
    {
        if(s->cplx)
            return x;
        compute(e, "f64");
        const std::pair<int, const rocke_type_t*> rs[2]
            = {{RND_F16, rocke_f16()}, {RND_BF16, rocke_bf16()}};
        for(const auto& r : rs)
        {
            V* k = rocke_b_const_i32(b, r.first);
            V* is = rocke_b_cmp_eq(b, rnd, k);
            const rocke_type_t* ty = r.second;
            x = e.when(is, {x}, [&](const std::vector<V*>& c) {
                V* odd = e.f64_to_f32_odd(c[0]);
                V* n = e.narrow(odd, ty);
                V* w = rocke_b_cast_to_f32(b, n);
                return std::vector<V*>{rocke_b_cvt_f32_to_f64(b, w)};
            })[0];
        }
        return x;
    }
    if(streq(s->comp, "f32"))
        return x;
    return e.narrow(x, comp_type(s->comp));
}

/* _from64 */
V* from64(Emit& e, V* x, const char* dt)
{
    compute(e, "f64");
    return e.from_c(x, dt);
}

/* _buffer_elem */
std::vector<V*>
    buffer_elem(rocke_ir_builder_t* b, V* rsrc, V* voff, V* soff, const char* dt, int ncomp)
{
    if(streq(dt, "f64"))
    {
        V* raw = rocke_b_buffer_load_vN(b, rsrc, voff, soff, rocke_i32(), 2 * ncomp);
        V* v = rocke_b_vec_bitcast(b, raw, rocke_vector_type(b, rocke_f64(), ncomp));
        std::vector<V*> out;
        for(int i = 0; i < ncomp; ++i)
            out.push_back(rocke_b_vec_extract(b, v, i));
        return out;
    }
    const rocke_type_t* ty = nullptr;
    dtype_of(dt, &ty, nullptr);
    if(ncomp == 2)
    {
        V* v = rocke_b_buffer_load_vN(b, rsrc, voff, soff, ty, 2);
        V* x0 = rocke_b_vec_extract(b, v, 0);
        V* x1 = rocke_b_vec_extract(b, v, 1);
        return {x0, x1};
    }
    return {rocke_b_buffer_load(b, rsrc, voff, soff, ty)};
}

/* _buffer_store */
void buffer_store(
    rocke_ir_builder_t* b, V* rsrc, V* voff, const char* dt, const std::vector<V*>& comps)
{
    V* c0 = rocke_b_const_i32(b, 0);
    int n = (int)comps.size();
    if(streq(dt, "f64"))
    {
        V* pk = rocke_b_vec_pack(b, comps.data(), n, rocke_f64());
        V* v = rocke_b_vec_bitcast(b, pk, rocke_vector_type(b, rocke_f32(), 2 * n));
        rocke_b_buffer_store_vN_f32(b, rsrc, voff, c0, v, 2 * n);
        return;
    }
    if(streq(dt, "f32"))
    {
        if(n == 1)
            rocke_b_buffer_store_f32(b, rsrc, voff, c0, comps[0]);
        else
        {
            V* pk = rocke_b_vec_pack(b, comps.data(), n, rocke_f32());
            rocke_b_buffer_store_vN_f32(b, rsrc, voff, c0, pk, 2);
        }
        return;
    }
    if(streq(dt, "f16"))
        rocke_b_buffer_store_f16(b, rsrc, voff, c0, comps[0]);
    else
        rocke_b_buffer_store_bf16(b, rsrc, voff, c0, comps[0]);
}

/* _unary_looped */
std::vector<V*> unary_looped(Emit& e, const std::vector<V*>& vs, V* code)
{
    rocke_ir_builder_t* b = e.b;
    int n = (int)vs.size();
    if(n == 1)
        return e.unary_list(vs, code);
    V* k1 = e.c32(OP_IDENTITY);
    V* n1 = rocke_b_cmp_ne(b, code, k1);
    V* k9 = e.c32(OP_CONJ);
    V* n9 = rocke_b_cmp_ne(b, code, k9);
    V* not_id = rocke_b_land(b, n1, n9);
    auto body = [&](const std::vector<V*>& cur) {
        int k = e.next_uid();
        rocke_inline_asm_opts_t o;
        memset(&o, 0, sizeof o);
        o.sideeffect = true;
        o.sideeffect_set = true;
        const rocke_type_t* rt[1] = {rocke_i32()};
        rocke_op_t* aop
            = rocke_b_inline_asm_named(b, "s_mov_b32 $0, 0", "=s", nullptr, 0, rt, 1, &o, "ozero");
        V* opaque = aop != nullptr ? aop->results[0] : nullptr;
        std::vector<std::string> names;
        for(int i = 0; i < n; ++i)
            names.push_back("u" + std::to_string(k) + "_" + std::to_string(i));
        std::vector<rocke_iter_arg_t> args(n);
        for(int i = 0; i < n; ++i)
        {
            args[i].name = names[i].c_str();
            args[i].init = cur[i];
        }
        std::string iv = "ui" + std::to_string(k);
        V* lo = e.c32(0);
        V* cn = e.c32(n);
        V* hi = rocke_b_add(b, cn, opaque);
        V* st = e.c32(1);
        rocke_for_t loop
            = rocke_b_scf_for_iter(b, lo, hi, st, args.data(), n, iv.c_str(), false, true);
        rocke_b_region_enter(b, loop.body);
        {
            std::vector<V*> vals(loop.iter_vars, loop.iter_vars + loop.num_iter_vars);
            V* it = loop.iv;
            V* x = vals[0];
            for(int j = 1; j < n; ++j)
            {
                V* cj = e.c32(j);
                V* is = rocke_b_cmp_eq(b, it, cj);
                x = rocke_b_select(b, is, vals[j], x);
            }
            V* y = e.unary_list({x}, code)[0];
            std::vector<V*> ys;
            for(int j = 0; j < n; ++j)
            {
                V* cj = e.c32(j);
                V* is = rocke_b_cmp_eq(b, it, cj);
                ys.push_back(rocke_b_select(b, is, y, vals[j]));
            }
            rocke_b_scf_yield(b, ys.data(), n);
        }
        rocke_b_region_leave(b);
        if(loop.op == nullptr)
            return std::vector<V*>(n, nullptr);
        return std::vector<V*>(loop.op->results, loop.op->results + loop.op->num_results);
    };
    return e.when(not_id, vs, body);
}

bool set_reason(char* reason, size_t cap, const std::string& msg)
{
    if(reason != nullptr && cap > 0)
        snprintf(reason, cap, "%s", msg.c_str());
    return false;
}

} // namespace

extern "C" {

rocke_htaot_dgett_spec_t rocke_htaot_dgett_spec_default(void)
{
    rocke_htaot_dgett_spec_t s;
    memset(&s, 0, sizeof s);
    s.comp = "f32";
    s.a_fast = "k";
    s.b_fast = "k";
    return s;
}

const rocke_mma_op_t* rocke_htaot_dgett_atom(const rocke_htaot_dgett_spec_t* s, const char* arch)
{
    if(s == nullptr || arch == nullptr)
        return nullptr;
    const rocke_arch_target_t* target = rocke_arch_target_from_gfx(arch);
    if(target == nullptr)
        return nullptr;
    const char* dt = streq(s->comp, "f16")    ? "fp16"
                     : streq(s->comp, "bf16") ? "bf16"
                     : streq(s->comp, "f32")  ? "fp32"
                                              : "fp64";
    const char* cdt = streq(s->comp, "f64") ? "fp64" : "fp32";
    std::vector<int> ks = (streq(s->comp, "f16") || streq(s->comp, "bf16"))
                              ? std::vector<int>{32, 16}
                              : std::vector<int>{4};
    for(int k : ks)
    {
        const rocke_mma_op_t* op = nullptr;
        try
        {
            op = rocke_mma_catalog_op_for_shape(&target->mma, "mma", dt, dt, cdt, 16, 16, k, NULL);
        }
        catch(const ckc::Error&)
        {
            /* Python: except (KeyError, ValueError): op = None */
            op = nullptr;
        }
        if(op != nullptr && op->wave_size == 64)
            return op;
    }
    return nullptr;
}

bool rocke_htaot_dgett_check(const rocke_htaot_dgett_spec_t* s,
                             const char* arch,
                             char* reason,
                             size_t cap)
{
    if(s == nullptr)
        return set_reason(reason, cap, "null spec");
    if(!streq(s->comp, "f16") && !streq(s->comp, "bf16") && !streq(s->comp, "f32")
       && !streq(s->comp, "f64"))
        return set_reason(reason,
                          cap,
                          std::string("comp '") + (s->comp ? s->comp : "")
                              + "' not in ('f16', 'bf16', 'f32', 'f64')");
    if(!(streq(s->a_fast, "k") || streq(s->a_fast, "r"))
       || !(streq(s->b_fast, "k") || streq(s->b_fast, "r")))
        return set_reason(reason, cap, "a_fast / b_fast must be 'k' or 'r'");
    if(s->cplx && !streq(s->comp, "f32") && !streq(s->comp, "f64"))
        return set_reason(reason, cap, "complex kernels compute in f32 or f64");
    if(s->cplx && s->math)
        return set_reason(reason, cap, "complex kernels have no math operators");
    if(rocke_htaot_dgett_atom(s, arch) == nullptr)
        return set_reason(
            reason, cap, std::string(arch ? arch : "") + " has no MFMA atom for comp " + s->comp);
    if(reason != nullptr && cap > 0)
        reason[0] = '\0';
    return true;
}

rocke_htaot_dgett_config_t rocke_htaot_dgett_config(const rocke_htaot_dgett_spec_t* s)
{
    if(s->cplx)
    {
        if(streq(s->comp, "f32"))
            return {64, 64, 16, 32, 32, 1, 1};
        return {64, 64, 8, 32, 32, 1, 1};
    }
    if(streq(s->comp, "f16") || streq(s->comp, "bf16"))
        return {128, 128, 32, 64, 64, 4, 8};
    if(streq(s->comp, "f32"))
        return {128, 128, 16, 64, 64, 2, 2};
    return {64, 64, 16, 32, 32, 1, 1};
}

rocke_status_t
    rocke_htaot_dgett_kernel_name(const rocke_htaot_dgett_spec_t* s, char* out, size_t cap)
{
    if(s == nullptr || out == nullptr)
        return ROCKE_ERR_VALUE;
    std::string n = std::string("hiptensor_aot_dgett_") + (s->cplx ? "c" : "r") + "_"
                    + (s->comp ? s->comp : "") + "_a" + (s->a_fast ? s->a_fast : "") + "_b"
                    + (s->b_fast ? s->b_fast : "") + "_" + (s->idx64 ? "i64" : "i32")
                    + (s->math ? "_m" : "");
    if(n.size() + 1 > cap)
        return ROCKE_ERR_VALUE;
    memcpy(out, n.c_str(), n.size() + 1);
    return ROCKE_OK;
}

static rocke_kernel_def_t*
    dgett_impl(rocke_ir_builder_t* b, const rocke_htaot_dgett_spec_t* spec, const char* arch)
{
    if(b == nullptr || spec == nullptr)
        return nullptr;
    char kname[160];
    char reason[200];
    bool valid = rocke_htaot_dgett_check(spec, arch, reason, sizeof reason);
    if(rocke_htaot_dgett_kernel_name(spec, kname, sizeof kname) != ROCKE_OK)
        snprintf(kname, sizeof kname, "hiptensor_aot_dgett");
    if(rocke_ir_builder_init(b, kname) != ROCKE_OK)
        return nullptr;
    if(!valid)
        return fail(b, reason);
    const rocke_htaot_dgett_config_t cfg = rocke_htaot_dgett_config(spec);
    const rocke_mma_op_t* atom = rocke_htaot_dgett_atom(spec, arch);
    const bool cx = spec->cplx;
    const int ncomp = cx ? 2 : 1;
    const int tk = cfg.tk, tm = cfg.tm, tn = cfg.tn;
    const int am_n = cfg.wave_m / atom->m, an_n = cfg.wave_n / atom->n;
    const rocke_type_t* ctype = comp_type(spec->comp);
    const bool acc_f64 = streq(spec->comp, "f64");
    const rocke_type_t* st_type = spec->idx64 ? rocke_i64() : rocke_i32();

    rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", THREADS);
    Emit e(b, spec->idx64, "f64", spec->math);

    const rocke_type_t* gptr = rocke_ptr_type(b, rocke_i8(), "global");
    std::map<char, V*> P;
    {
        rocke_param_opts_t ro;
        memset(&ro, 0, sizeof ro);
        ro.readonly = true;
        ro.readonly_set = true;
        ro.align = 2;
        ro.align_set = true;
        rocke_param_opts_t rw;
        memset(&rw, 0, sizeof rw);
        rw.align = 2;
        rw.align_set = true;
        P['A'] = rocke_b_param(b, "A", gptr, &ro);
        P['B'] = rocke_b_param(b, "B", gptr, &ro);
        P['C'] = rocke_b_param(b, "C", gptr, &rw);
        P['D'] = rocke_b_param(b, "D", gptr, &rw);
    }
    std::vector<V*> alpha, beta;
    if(cx)
    {
        alpha.push_back(param(b, "alpha_re", rocke_f64()));
        alpha.push_back(param(b, "alpha_im", rocke_f64()));
        beta.push_back(param(b, "beta_re", rocke_f64()));
        beta.push_back(param(b, "beta_im", rocke_f64()));
    }
    else
    {
        alpha.push_back(param(b, "alpha", rocke_f64()));
        beta.push_back(param(b, "beta", rocke_f64()));
    }
    std::map<std::string, V*> arg;
    for(const char* n : {"dtA", "dtB", "dtC", "dtD", "opA", "opB", "opC", "rnd"})
        arg[n] = param(b, n, rocke_i32());
    for(const char* n : {"M", "N", "nk", "tiles_n", "tn_m", "tn_s"})
        arg[n] = param(b, n, rocke_i32());
    for(const char* n : {"a_bytes", "b_bytes", "c_bytes", "d_bytes"})
        arg[n] = param(b, n, rocke_i32());
    Space m_sp = space(b, "m", SLOTS_MN, "ACD", st_type);
    Space n_sp = space(b, "n", SLOTS_MN, "BCD", st_type);
    Space kl_sp = space(b, "kl", SLOTS_KL, "AB", st_type);
    Space kt_sp = space(b, "kt", SLOTS_KT, "AB", st_type);
    Space l_sp = space(b, "l", SLOTS_L, "ABCD", st_type);
    V* kps = param(b, "kps", rocke_i32());
    V* kpl = param(b, "kpl", rocke_i32());
    V* krl = param(b, "krl", rocke_i32());

    V* c0 = rocke_b_const_i32(b, 0);
    V* tid = rocke_b_thread_id_x(b);
    V* k64 = rocke_b_const_i32(b, 64);
    V* lane = rocke_b_mod(b, tid, k64);
    V* k64b = rocke_b_const_i32(b, 64);
    V* wd = rocke_b_div(b, tid, k64b);
    V* wave = rocke_b_readfirstlane(b, wd);
    V* k2 = rocke_b_const_i32(b, 2);
    V* wm = rocke_b_mod(b, wave, k2);
    V* k2b = rocke_b_const_i32(b, 2);
    V* wn = rocke_b_div(b, wave, k2b);
    V* bid = rocke_b_block_id_x(b);
    V* tile_m = e.udiv(bid, arg["tn_m"], arg["tn_s"]);
    V* tmn = rocke_b_mul(b, tile_m, arg["tiles_n"]);
    V* tile_n = rocke_b_sub(b, bid, tmn);
    V* ktm = rocke_b_const_i32(b, tm);
    V* m0 = rocke_b_mul(b, tile_m, ktm);
    V* ktn = rocke_b_const_i32(b, tn);
    V* n0 = rocke_b_mul(b, tile_n, ktn);

    /* batch: every tensor's base moved by its batch offset (64-bit) */
    V* bz = rocke_b_block_id_z(b);
    std::vector<V*> loffs
        = chain(e, bz, l_sp.ext, l_sp.mg, {l_sp.st['A'], l_sp.st['B'], l_sp.st['C'], l_sp.st['D']})
              .first;
    std::map<char, V*> base;
    {
        const char tn4[4] = {'A', 'B', 'C', 'D'};
        for(int i = 0; i < 4; ++i)
            base[tn4[i]] = spec->idx64 ? loffs[i] : rocke_b_sext(b, loffs[i], rocke_i64());
    }

    /* operand loaders */
    std::map<char, V*> lds;
    {
        const std::pair<char, int> ops[2] = {{'A', tm}, {'B', tn}};
        for(const auto& o : ops)
        {
            int shape[2] = {2 * ncomp * o.second, tk + cfg.pad};
            std::string hint = std::string(1, (char)(o.first - 'A' + 'a')) + "_lds";
            lds[o.first] = rocke_b_smem_alloc(b, ctype, shape, 2, hint.c_str());
        }
    }

    struct LaneMap
    {
        std::vector<V*> rs, ks;
        std::vector<std::pair<int, int>> elems;
    };
    auto lane_map = [&](const char* fast, int rows) {
        LaneMap l;
        if(streq(fast, "k"))
        {
            int tpr = tk / cfg.vk;
            int rpp = THREADS / tpr;
            int passes = rows / rpp;
            V* kt = rocke_b_const_i32(b, tpr);
            V* r0 = rocke_b_div(b, tid, kt);
            V* kt2 = rocke_b_const_i32(b, tpr);
            V* md = rocke_b_mod(b, tid, kt2);
            V* kv = rocke_b_const_i32(b, cfg.vk);
            V* k0 = rocke_b_mul(b, md, kv);
            for(int p = 0; p < passes; ++p)
            {
                if(p)
                {
                    V* c = rocke_b_const_i32(b, p * rpp);
                    l.rs.push_back(rocke_b_add(b, r0, c));
                }
                else
                    l.rs.push_back(r0);
            }
            for(int i = 0; i < cfg.vk; ++i)
            {
                if(i)
                {
                    V* c = rocke_b_const_i32(b, i);
                    l.ks.push_back(rocke_b_add(b, k0, c));
                }
                else
                    l.ks.push_back(k0);
            }
            for(int p = 0; p < passes; ++p)
                for(int i = 0; i < cfg.vk; ++i)
                    l.elems.push_back({p, i});
            return l;
        }
        int kpp = THREADS / rows;
        int passes = tk / kpp;
        V* kr = rocke_b_const_i32(b, rows);
        V* r0 = rocke_b_mod(b, tid, kr);
        V* kr2 = rocke_b_const_i32(b, rows);
        V* k0 = rocke_b_div(b, tid, kr2);
        for(int p = 0; p < passes; ++p)
        {
            if(p)
            {
                V* c = rocke_b_const_i32(b, p * kpp);
                l.ks.push_back(rocke_b_add(b, k0, c));
            }
            else
                l.ks.push_back(k0);
        }
        l.rs.push_back(r0);
        for(int p = 0; p < passes; ++p)
            l.elems.push_back({0, p});
        return l;
    };

    const char* optype = streq(spec->comp, "f64") ? "f64" : "f32";

    struct Operand
    {
        char nm;
        int rows;
        V* dt;
        V* op;
        V* bytes;
        std::vector<std::pair<int, int>> elems;
        std::vector<V*> ks;
        const char* fast;
        std::vector<V*> row_ok, row_off, rloc;
        std::vector<V*> k_off, k_ok;
        V* ptr;
    };
    auto make_operand = [&](char nm,
                            const char* fast,
                            int rows,
                            V* row0,
                            V* lim,
                            const Space& sp,
                            const std::vector<V*>& strides) {
        Operand o;
        o.nm = nm;
        o.rows = rows;
        o.dt = arg[std::string("dt") + nm];
        o.op = arg[std::string("op") + nm];
        o.bytes = arg[nm == 'A' ? "a_bytes" : "b_bytes"];
        LaneMap l = lane_map(fast, rows);
        o.elems = l.elems;
        o.ks = l.ks;
        o.fast = fast;
        for(V* r : l.rs)
        {
            V* g = rocke_b_add(b, row0, r);
            V* ok = rocke_b_cmp_lt(b, g, lim);
            V* gs = rocke_b_select(b, ok, g, c0);
            V* off = chain(e, gs, sp.ext, sp.mg, {strides}).first[0];
            o.row_ok.push_back(ok);
            o.row_off.push_back(off);
        }
        o.rloc = l.rs;
        for(V* k : l.ks)
        {
            auto od = chain(e, k, kl_sp.ext, kl_sp.mg, {kl_sp.st.at(nm)});
            V* ok = nullptr;
            for(size_t i = 0; i < od.second.size(); ++i)
            {
                V* ci = rocke_b_const_i32(b, (int64_t)i);
                V* ne = rocke_b_cmp_ne(b, kps, ci);
                V* lt = rocke_b_cmp_lt(b, od.second[i], kpl);
                V* t = rocke_b_lor(b, ne, lt);
                ok = ok == nullptr ? t : rocke_b_land(b, ok, t);
            }
            o.k_off.push_back(od.first[0]);
            o.k_ok.push_back(ok);
        }
        o.ptr = P[nm];
        return o;
    };

    /* Operand._raw */
    auto raw = [&](Operand& o,
                   const char* dt,
                   V* koff,
                   const std::vector<V*>& off,
                   const std::vector<V*>& ok) {
        const rocke_type_t* ty = nullptr;
        int nb = 0;
        dtype_of(dt, &ty, &nb);
        int es = nb * ncomp;
        std::vector<V*> out;
        V* rsrc = nullptr;
        V* soff = nullptr;
        if(!spec->idx64)
        {
            V* ce = rocke_b_const_i64(b, es);
            V* bo = rocke_b_mul(b, base[o.nm], ce);
            V* pp = rocke_b_global_ptr_add(b, o.ptr, bo);
            rsrc = rocke_b_buffer_rsrc(b, pp, o.bytes);
            V* ce2 = rocke_b_const_i32(b, es);
            soff = rocke_b_mul(b, koff, ce2);
        }
        for(size_t j = 0; j < off.size(); ++j)
        {
            std::vector<V*> comps;
            if(spec->idx64)
            {
                V* t1 = rocke_b_add(b, off[j], koff);
                V* tot = rocke_b_add(b, t1, base[o.nm]);
                V* z64 = rocke_b_const_i64(b, 0);
                tot = rocke_b_select(b, ok[j], tot, z64);
                V* ce = rocke_b_const_i64(b, es);
                V* bo = rocke_b_mul(b, tot, ce);
                V* p = rocke_b_global_ptr_add(b, o.ptr, bo);
                if(cx)
                {
                    V* rv = rocke_b_global_load_vN(b, p, c0, ty, 2, nb);
                    V* x0 = rocke_b_vec_extract(b, rv, 0);
                    V* x1 = rocke_b_vec_extract(b, rv, 1);
                    comps = {x0, x1};
                }
                else
                    comps = {rocke_b_global_load(b, p, c0, ty, nb)};
                for(V*& x : comps)
                {
                    V* z = zero_of(b, ty);
                    x = rocke_b_select(b, ok[j], x, z);
                }
            }
            else
            {
                V* ce = rocke_b_const_i32(b, es);
                V* bo = rocke_b_mul(b, off[j], ce);
                V* sn = rocke_b_const_i32(b, SENTINEL);
                V* vo = rocke_b_select(b, ok[j], bo, sn);
                comps = buffer_elem(b, rsrc, vo, soff, dt, ncomp);
            }
            for(V* x : comps)
                out.push_back(to_op(e, x, dt, optype));
        }
        return out;
    };

    /* Operand.load */
    struct Loaded
    {
        std::vector<std::vector<V*>> vals;
        std::vector<V*> ok;
    };
    auto load = [&](Operand& o, V* koff, V* kmax) {
        std::vector<V*> ok, off;
        for(const auto& el : o.elems)
        {
            V* a = rocke_b_land(b, o.row_ok[el.first], o.k_ok[el.second]);
            V* lt = rocke_b_cmp_lt(b, o.ks[el.second], kmax);
            ok.push_back(rocke_b_land(b, a, lt));
            off.push_back(rocke_b_add(b, o.row_off[el.first], o.k_off[el.second]));
        }
        size_t n = o.elems.size();
        compute(e, optype);
        V* zero = e.cf(0.0);
        std::vector<V*> vals(n * ncomp, zero);
        for(const char* dt : load_types(spec))
        {
            V* kc = rocke_b_const_i32(b, dt_code(dt));
            V* cond = rocke_b_cmp_eq(b, o.dt, kc);
            vals = e.when(
                cond, vals, [&](const std::vector<V*>&) { return raw(o, dt, koff, off, ok); });
        }
        Loaded l;
        for(size_t i = 0; i < n; ++i)
            l.vals.push_back(
                std::vector<V*>(vals.begin() + i * ncomp, vals.begin() + (i + 1) * ncomp));
        l.ok = ok;
        return l;
    };

    /* Operand.finish */
    auto finish = [&](Operand& o, const Loaded& l) {
        compute(e, optype);
        V* zero = e.cf(0.0);
        std::vector<std::vector<V*>> out;
        if(cx)
        {
            V* k9 = rocke_b_const_i32(b, OP_CONJ);
            V* conj = rocke_b_cmp_eq(b, o.op, k9);
            for(size_t i = 0; i < l.vals.size(); ++i)
            {
                V* re = l.vals[i][0];
                V* im = l.vals[i][1];
                V* nim = rocke_b_fneg(b, im);
                im = rocke_b_select(b, conj, nim, im);
                V* s0 = rocke_b_select(b, l.ok[i], re, zero);
                V* s1 = rocke_b_select(b, l.ok[i], im, zero);
                out.push_back({s0, s1});
            }
        }
        else
        {
            std::vector<V*> flat;
            for(const auto& x : l.vals)
                flat.push_back(x[0]);
            flat = e.unary_list(flat, o.op);
            for(size_t i = 0; i < flat.size(); ++i)
                out.push_back({rocke_b_select(b, l.ok[i], flat[i], zero)});
        }
        for(auto& c : out)
            for(V*& x : c)
                x = to_comp(e, x, spec, arg["rnd"]);
        return out;
    };

    /* Operand.store */
    auto store = [&](Operand& o, const std::vector<std::vector<V*>>& vals, V* stage) {
        int rows = o.rows;
        V* kr = rocke_b_const_i32(b, rows);
        V* row_base = rocke_b_mul(b, stage, kr);
        for(int c = 0; c < ncomp; ++c)
        {
            V* plane = row_base;
            if(c)
            {
                V* kc = rocke_b_const_i32(b, c * 2 * rows);
                plane = rocke_b_add(b, row_base, kc);
            }
            if(streq(o.fast, "k") && cfg.vk > 1)
            {
                for(size_t ri = 0; ri < o.rloc.size(); ++ri)
                {
                    std::vector<V*> chunk;
                    for(int i = 0; i < cfg.vk; ++i)
                        chunk.push_back(vals[ri * cfg.vk + i][c]);
                    V* r = rocke_b_add(b, plane, o.rloc[ri]);
                    V* pk = rocke_b_vec_pack(b, chunk.data(), cfg.vk, ctype);
                    V* idx[2] = {r, o.ks[0]};
                    rocke_b_smem_store_vN(b, lds[o.nm], idx, 2, pk, cfg.vk);
                }
            }
            else
            {
                for(size_t j = 0; j < o.elems.size(); ++j)
                {
                    V* r = rocke_b_add(b, plane, o.rloc[o.elems[j].first]);
                    V* idx[2] = {r, o.ks[o.elems[j].second]};
                    rocke_b_smem_store_vN(b, lds[o.nm], idx, 2, vals[j][c], 1);
                }
            }
        }
    };

    Operand opa = make_operand('A', spec->a_fast, tm, m0, arg["M"], m_sp, m_sp.st['A']);
    Operand opb = make_operand('B', spec->b_fast, tn, n0, arg["N"], n_sp, n_sp.st['B']);

    auto ktile = [&](V* kt) {
        auto od = chain(e, kt, kt_sp.ext, kt_sp.mg, {kt_sp.st['A'], kt_sp.st['B']});
        V* ktk = rocke_b_const_i32(b, tk);
        V* m = rocke_b_mul(b, od.second[0], ktk);
        V* kmax = rocke_b_sub(b, krl, m);
        return std::make_tuple(od.first[0], od.first[1], kmax);
    };
    auto load_tile = [&](V* kt, V* live) {
        auto t = ktile(kt);
        V* kmax = rocke_b_select(b, live, std::get<2>(t), c0);
        Loaded la = load(opa, std::get<0>(t), kmax);
        Loaded lb = load(opb, std::get<1>(t), kmax);
        return std::make_pair(la, lb);
    };
    auto store_tile = [&](const std::pair<Loaded, Loaded>& regs, V* stage) {
        std::vector<std::vector<V*>> fa = finish(opa, regs.first);
        store(opa, fa, stage);
        std::vector<std::vector<V*>> fb = finish(opb, regs.second);
        store(opb, fb, stage);
    };

    /* fragment coordinates (atom lane maps) */
    V *a_row = nullptr, *a_k = nullptr, *b_k = nullptr, *b_col = nullptr;
    rocke_layout_map_coord(atom->a_layout, b, lane, 0, &a_row, &a_k);
    rocke_layout_map_coord(atom->b_layout, b, lane, 0, &b_k, &b_col);
    V* kwm = rocke_b_const_i32(b, cfg.wave_m);
    V* wmm = rocke_b_mul(b, wm, kwm);
    V* a_row0 = rocke_b_add(b, wmm, a_row);
    V* kwn = rocke_b_const_i32(b, cfg.wave_n);
    V* wnn = rocke_b_mul(b, wn, kwn);
    V* b_row0 = rocke_b_add(b, wnn, b_col);
    const int kpl_a = atom->a_frag_len;
    const int kpl_b = atom->b_frag_len;

    auto frag = [&](char nm, V* stage, V* row, V* k, int comp_i, int n) {
        int rows = nm == 'A' ? tm : tn;
        V* kr = rocke_b_const_i32(b, rows);
        V* sr = rocke_b_mul(b, stage, kr);
        V* a1 = rocke_b_add(b, sr, row);
        V* kc = rocke_b_const_i32(b, comp_i * 2 * rows);
        V* r = rocke_b_add(b, a1, kc);
        V* idx[2] = {r, k};
        V* v = rocke_b_smem_load_vN(b, lds[nm], idx, 2, ctype, n);
        return n > 1 ? v : rocke_b_vec_extract(b, v, 0);
    };

    const rocke_type_t* acc_ty = acc_f64 ? rocke_f64() : rocke_f32();
    V* zc = acc_f64 ? rocke_b_const_f64(b, 0.0) : rocke_b_const_f32(b, 0.0);
    std::vector<V*> zcs(atom->c_frag_len, zc);
    V* zero_acc = rocke_b_vec_pack(b, zcs.data(), atom->c_frag_len, acc_ty);
    const int n_acc = am_n * an_n * ncomp;

    auto mfma_tile = [&](std::vector<V*> accs, V* stage) {
        for(int ks = 0; ks < tk / atom->k; ++ks)
        {
            V* kk_a = a_k;
            if(ks)
            {
                V* c = rocke_b_const_i32(b, ks * atom->k);
                kk_a = rocke_b_add(b, a_k, c);
            }
            V* kk_b = b_k;
            if(ks)
            {
                V* c = rocke_b_const_i32(b, ks * atom->k);
                kk_b = rocke_b_add(b, b_k, c);
            }
            std::vector<std::vector<V*>> fa, fb;
            for(int am = 0; am < am_n; ++am)
            {
                V* row = a_row0;
                if(am)
                {
                    V* c = rocke_b_const_i32(b, am * atom->m);
                    row = rocke_b_add(b, a_row0, c);
                }
                std::vector<V*> f;
                for(int c = 0; c < ncomp; ++c)
                    f.push_back(frag('A', stage, row, kk_a, c, kpl_a));
                fa.push_back(f);
            }
            for(int an = 0; an < an_n; ++an)
            {
                V* row = b_row0;
                if(an)
                {
                    V* c = rocke_b_const_i32(b, an * atom->n);
                    row = rocke_b_add(b, b_row0, c);
                }
                std::vector<V*> f;
                for(int c = 0; c < ncomp; ++c)
                    f.push_back(frag('B', stage, row, kk_b, c, kpl_b));
                fb.push_back(f);
            }
            std::vector<V*> neg_ai;
            if(cx)
                for(const auto& x : fa)
                    neg_ai.push_back(rocke_b_fneg(b, x[1]));
            for(int am = 0; am < am_n; ++am)
                for(int an = 0; an < an_n; ++an)
                {
                    int i = am * an_n + an;
                    if(!cx)
                    {
                        accs[i]
                            = rocke_b_mma(b, atom->op_id, fa[am][0], fb[an][0], accs[i], NULL, 0);
                        continue;
                    }
                    V* ar = fa[am][0];
                    V* ai = fa[am][1];
                    V* br = fb[an][0];
                    V* bi = fb[an][1];
                    V* r_ = accs[2 * i];
                    V* i_ = accs[2 * i + 1];
                    r_ = rocke_b_mma(b, atom->op_id, ar, br, r_, NULL, 0);
                    r_ = rocke_b_mma(b, atom->op_id, neg_ai[am], bi, r_, NULL, 0);
                    i_ = rocke_b_mma(b, atom->op_id, ar, bi, i_, NULL, 0);
                    i_ = rocke_b_mma(b, atom->op_id, ai, br, i_, NULL, 0);
                    accs[2 * i] = r_;
                    accs[2 * i + 1] = i_;
                }
        }
        return accs;
    };

    /* K loop */
    V* nk = arg["nk"];
    {
        std::vector<std::vector<V*>> zeros;
        for(size_t i = 0; i < opa.elems.size(); ++i)
        {
            V* z = zero_of(b, ctype);
            zeros.push_back(std::vector<V*>(ncomp, z));
        }
        V* one = rocke_b_const_i32(b, 1);
        store(opa, zeros, one);
        std::vector<std::vector<V*>> zb;
        for(size_t i = 0; i < opb.elems.size(); ++i)
        {
            V* z = zero_of(b, ctype);
            zb.push_back(std::vector<V*>(ncomp, z));
        }
        V* one2 = rocke_b_const_i32(b, 1);
        store(opb, zb, one2);
    }
    rocke_b_sync(b);
    std::vector<std::string> acc_names;
    for(int i = 0; i < n_acc; ++i)
        acc_names.push_back("dg_acc" + std::to_string(i));
    std::vector<rocke_iter_arg_t> acc_args(n_acc);
    for(int i = 0; i < n_acc; ++i)
    {
        acc_args[i].name = acc_names[i].c_str();
        acc_args[i].init = zero_acc;
    }
    V* k1 = rocke_b_const_i32(b, 1);
    V* hi = rocke_b_add(b, nk, k1);
    V* st1 = rocke_b_const_i32(b, 1);
    rocke_for_t loop
        = rocke_b_scf_for_iter(b, c0, hi, st1, acc_args.data(), n_acc, "dg_kt", false, false);
    rocke_b_region_enter(b, loop.body);
    {
        V* kt = loop.iv;
        std::vector<V*> accs(loop.iter_vars, loop.iter_vars + loop.num_iter_vars);
        V* kone = rocke_b_const_i32(b, 1);
        V* stage = rocke_b_land(b, kt, kone);
        V* live = rocke_b_cmp_lt(b, kt, nk);
        std::pair<Loaded, Loaded> nxt = load_tile(kt, live);
        V* kone2 = rocke_b_const_i32(b, 1);
        V* other = rocke_b_xor(b, stage, kone2);
        accs = mfma_tile(accs, other);
        store_tile(nxt, stage);
        rocke_b_sync(b);
        rocke_b_scf_yield(b, accs.data(), (int)accs.size());
    }
    rocke_b_region_leave(b);
    std::vector<V*> accs;
    if(loop.op != nullptr)
        accs.assign(loop.op->results, loop.op->results + loop.op->num_results);
    else
        accs.assign(n_acc, nullptr);

    /* epilogue (f64) */
    compute(e, "f64");
    V* zero = rocke_b_const_f64(b, 0.0);
    std::vector<V*> c_rows;
    V* col_l = nullptr;
    for(int slot = 0; slot < atom->c_frag_len; ++slot)
    {
        V *r = nullptr, *c = nullptr;
        rocke_layout_map_coord(atom->c_layout, b, lane, slot, &r, &c);
        c_rows.push_back(r);
        if(col_l == nullptr)
            col_l = c;
    }
    V* kwm2 = rocke_b_const_i32(b, cfg.wave_m);
    V* wmm2 = rocke_b_mul(b, wm, kwm2);
    V* mw = rocke_b_add(b, m0, wmm2);
    V* kwn2 = rocke_b_const_i32(b, cfg.wave_n);
    V* wnn2 = rocke_b_mul(b, wn, kwn2);
    V* nw = rocke_b_add(b, n0, wnn2);
    struct Col
    {
        V* ok;
        V* dn;
        V* cn;
    };
    std::vector<Col> cols;
    for(int an = 0; an < an_n; ++an)
    {
        V* inner = col_l;
        if(an)
        {
            V* c = rocke_b_const_i32(b, an * atom->n);
            inner = rocke_b_add(b, col_l, c);
        }
        V* n = rocke_b_add(b, nw, inner);
        V* ok = rocke_b_cmp_lt(b, n, arg["N"]);
        V* ns = rocke_b_select(b, ok, n, c0);
        std::vector<V*> dc = chain(e, ns, n_sp.ext, n_sp.mg, {n_sp.st['D'], n_sp.st['C']}).first;
        cols.push_back({ok, dc[0], dc[1]});
    }
    V* a_zero = rocke_b_fcmp(b, "oeq", alpha[0], zero);
    V* ai_zero = nullptr;
    V* bi_zero = nullptr;
    if(cx)
    {
        ai_zero = rocke_b_fcmp(b, "oeq", alpha[1], zero);
        a_zero = rocke_b_land(b, a_zero, ai_zero);
    }
    V* bz0 = rocke_b_fcmp(b, "oeq", beta[0], zero);
    V* use_c = rocke_b_lnot(b, bz0);
    if(cx)
    {
        bi_zero = rocke_b_fcmp(b, "oeq", beta[1], zero);
        V* nb = rocke_b_lnot(b, bi_zero);
        use_c = rocke_b_lor(b, use_c, nb);
    }
    std::vector<const char*> out_types
        = cx ? load_types(spec) : std::vector<const char*>{"f16", "bf16", "f32", "f64"};

    auto scaled = [&](const std::vector<V*>& s, const std::vector<V*>& vals, V* s_im_zero) {
        if(!cx)
            return std::vector<V*>{rocke_b_fmul(b, s[0], vals[0])};
        std::pair<V*, V*> p = e.cmul(s[0], s[1], vals[0], vals[1]);
        V* m0r = rocke_b_fmul(b, s[0], vals[0]);
        V* r0 = rocke_b_select(b, s_im_zero, m0r, p.first);
        V* m0i = rocke_b_fmul(b, s[0], vals[1]);
        V* r1 = rocke_b_select(b, s_im_zero, m0i, p.second);
        return std::vector<V*>{r0, r1};
    };
    auto out_rsrc = [&](char nm, const char* span, int es) {
        V* ce = rocke_b_const_i64(b, es);
        V* bo = rocke_b_mul(b, base[nm], ce);
        V* pp = rocke_b_global_ptr_add(b, P[nm], bo);
        return rocke_b_buffer_rsrc(b, pp, arg[span]);
    };
    struct Elem
    {
        V* ok;
        V* doff;
        V* coff;
        std::vector<V*> vals;
    };
    auto load_c = [&](const std::vector<Elem>& elems) {
        auto one = [&](const char* dt) {
            const rocke_type_t* ty = nullptr;
            int nb = 0;
            dtype_of(dt, &ty, &nb);
            int es = nb * ncomp;
            V* rsrc = spec->idx64 ? nullptr : out_rsrc('C', "c_bytes", es);
            std::vector<V*> out;
            for(const Elem& el : elems)
            {
                std::vector<V*> comps;
                if(spec->idx64)
                {
                    V* ad = rocke_b_add(b, el.coff, base['C']);
                    V* z64 = rocke_b_const_i64(b, 0);
                    V* tot = rocke_b_select(b, el.ok, ad, z64);
                    V* ce = rocke_b_const_i64(b, es);
                    V* bo = rocke_b_mul(b, tot, ce);
                    V* p = rocke_b_global_ptr_add(b, P['C'], bo);
                    if(cx)
                    {
                        V* rv = rocke_b_global_load_vN(b, p, c0, ty, 2, nb);
                        V* x0 = rocke_b_vec_extract(b, rv, 0);
                        V* x1 = rocke_b_vec_extract(b, rv, 1);
                        comps = {x0, x1};
                    }
                    else
                        comps = {rocke_b_global_load(b, p, c0, ty, nb)};
                    for(V*& x : comps)
                    {
                        V* z = zero_of(b, ty);
                        x = rocke_b_select(b, el.ok, x, z);
                    }
                }
                else
                {
                    V* ce = rocke_b_const_i32(b, es);
                    V* bo = rocke_b_mul(b, el.coff, ce);
                    V* sn = rocke_b_const_i32(b, SENTINEL);
                    V* vo = rocke_b_select(b, el.ok, bo, sn);
                    comps = buffer_elem(b, rsrc, vo, c0, dt, ncomp);
                }
                for(V* x : comps)
                    out.push_back(to_op(e, x, dt, "f64"));
            }
            return out;
        };
        auto branches = [&](const std::vector<V*>&) {
            std::vector<V*> vals(elems.size() * ncomp, zero);
            for(const char* dt : out_types)
            {
                V* kc = rocke_b_const_i32(b, dt_code(dt));
                V* cond = rocke_b_cmp_eq(b, arg["dtC"], kc);
                vals = e.when(cond, vals, [&](const std::vector<V*>&) { return one(dt); });
            }
            return vals;
        };
        return e.when(use_c, std::vector<V*>(elems.size() * ncomp, zero), branches);
    };
    auto store_d = [&](const std::vector<Elem>& elems, const std::vector<std::vector<V*>>& outs) {
        for(const char* dt : out_types)
        {
            V* kc = rocke_b_const_i32(b, dt_code(dt));
            V* is = rocke_b_cmp_eq(b, arg["dtD"], kc);
            scf_if(b, is, [&]() {
                const rocke_type_t* ty = nullptr;
                int nb = 0;
                dtype_of(dt, &ty, &nb);
                int es = nb * ncomp;
                V* rsrc = spec->idx64 ? nullptr : out_rsrc('D', "d_bytes", es);
                for(size_t i = 0; i < elems.size(); ++i)
                {
                    const Elem& el = elems[i];
                    std::vector<V*> comps;
                    for(V* x : outs[i])
                        comps.push_back(from64(e, x, dt));
                    if(spec->idx64)
                    {
                        scf_if(b, el.ok, [&]() {
                            V* ad = rocke_b_add(b, el.doff, base['D']);
                            V* ce = rocke_b_const_i64(b, es);
                            V* bo = rocke_b_mul(b, ad, ce);
                            V* p = rocke_b_global_ptr_add(b, P['D'], bo);
                            if(cx)
                            {
                                V* pk = rocke_b_vec_pack(b, comps.data(), 2, ty);
                                rocke_b_global_store_vN(b, p, c0, pk, 2, nb);
                            }
                            else
                                rocke_b_global_store(b, p, c0, comps[0], nb);
                        });
                    }
                    else
                    {
                        V* ce = rocke_b_const_i32(b, es);
                        V* bo = rocke_b_mul(b, el.doff, ce);
                        V* sn = rocke_b_const_i32(b, SENTINEL);
                        V* vo = rocke_b_select(b, el.ok, bo, sn);
                        buffer_store(b, rsrc, vo, dt, comps);
                    }
                }
            });
        }
    };

    for(int am = 0; am < am_n; ++am)
    {
        std::vector<Elem> elems;
        for(int slot = 0; slot < atom->c_frag_len; ++slot)
        {
            V* r = c_rows[slot];
            V* inner = r;
            if(am)
            {
                V* c = rocke_b_const_i32(b, am * atom->m);
                inner = rocke_b_add(b, r, c);
            }
            V* m = rocke_b_add(b, mw, inner);
            V* mok = rocke_b_cmp_lt(b, m, arg["M"]);
            V* ms = rocke_b_select(b, mok, m, c0);
            std::vector<V*> dc
                = chain(e, ms, m_sp.ext, m_sp.mg, {m_sp.st['D'], m_sp.st['C']}).first;
            for(int an = 0; an < an_n; ++an)
            {
                int i = am * an_n + an;
                std::vector<V*> vals;
                for(int c = 0; c < ncomp; ++c)
                    vals.push_back(rocke_b_vec_extract(b, accs[ncomp * i + c], slot));
                if(!acc_f64)
                    for(V*& v : vals)
                        v = rocke_b_cvt_f32_to_f64(b, v);
                V* okmn = rocke_b_land(b, mok, cols[an].ok);
                V* doff = rocke_b_add(b, dc[0], cols[an].dn);
                V* coff = rocke_b_add(b, dc[1], cols[an].cn);
                elems.push_back({okmn, doff, coff, vals});
            }
        }
        std::vector<V*> cvals = load_c(elems);
        std::vector<std::vector<V*>> cs;
        if(cx)
        {
            V* k9 = rocke_b_const_i32(b, OP_CONJ);
            V* conj_c = rocke_b_cmp_eq(b, arg["opC"], k9);
            for(size_t i = 0; i < elems.size(); ++i)
            {
                V* re = cvals[2 * i];
                V* im = cvals[2 * i + 1];
                V* nim = rocke_b_fneg(b, im);
                cs.push_back({re, rocke_b_select(b, conj_c, nim, im)});
            }
        }
        else
            for(V* v : unary_looped(e, cvals, arg["opC"]))
                cs.push_back({v});
        std::vector<std::vector<V*>> outs;
        for(size_t i = 0; i < elems.size(); ++i)
        {
            std::vector<V*> sa = scaled(alpha, elems[i].vals, ai_zero);
            std::vector<V*> t;
            for(V* x : sa)
                t.push_back(rocke_b_select(b, a_zero, zero, x));
            std::vector<V*> q = scaled(beta, cs[i], bi_zero);
            std::vector<V*> o;
            for(size_t j = 0; j < t.size(); ++j)
            {
                V* s = rocke_b_fadd(b, t[j], q[j]);
                o.push_back(rocke_b_select(b, use_c, s, t[j]));
            }
            outs.push_back(o);
        }
        store_d(elems, outs);
    }
    rocke_b_ret(b);
    if(!rocke_ir_builder_ok(b))
        return nullptr;
    return rocke_ir_builder_kernel(b);
}

rocke_kernel_def_t* rocke_build_htaot_dgett(rocke_ir_builder_t* b,
                                            const rocke_htaot_dgett_spec_t* spec,
                                            const char* arch)
{
    return ckc::guard_builder(b,
                              [&]() -> rocke_kernel_def_t* { return dgett_impl(b, spec, arch); });
}

} /* extern "C" */
