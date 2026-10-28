// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * hiptensor_aot_reduce.cpp -- C++ mirror of
 * library/kernels/common/hiptensor_aot/reduce.py (ReduceSpec, build_reduce:
 * the row / col reduction kernels and the split finalize).
 */
#include "rocke/instance_hiptensor_aot_internal.h"

#include "rocke/error_boundary.hpp"
#include "rocke/ir_internal.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#include <limits>
#include <map>
#include <string>
#include <vector>

using namespace rocke::htaot;

namespace
{

constexpr int WAVE = 64;

const char* cdt_of(const rocke_htaot_reduce_spec_t* s)
{
    return streq(s->a, "f64") || streq(s->d, "f64") ? "f64" : "f32";
}

/* reduce._identity */
V* identity(Emit& e, V* op)
{
    rocke_ir_builder_t* b = e.b;
    const double inf = std::numeric_limits<double>::infinity();
    V* r = e.cf(0.0);
    V* k5 = e.c32(OP_MUL);
    V* m = rocke_b_cmp_eq(b, op, k5);
    V* one = e.cf(1.0);
    r = rocke_b_select(b, m, one, r);
    V* k6 = e.c32(OP_MAX);
    V* mx = rocke_b_cmp_eq(b, op, k6);
    V* ninf = e.cf(-inf);
    r = rocke_b_select(b, mx, ninf, r);
    V* k7 = e.c32(OP_MIN);
    V* mn = rocke_b_cmp_eq(b, op, k7);
    V* pinf = e.cf(inf);
    r = rocke_b_select(b, mn, pinf, r);
    return r;
}

/* reduce._Geo */
struct Geo
{
    std::vector<V*> ke;
    std::vector<std::pair<V*, V*>> km;
    std::vector<V*> kA, kC, kD;
    std::vector<V*> re;
    std::vector<std::pair<V*, V*>> rm;
    std::vector<V*> rA;

    Geo(rocke_ir_builder_t* b, Emit& e, bool with_red)
    {
        for(int i = 0; i < MAX_RANK; ++i)
            ke.push_back(param(b, "ke" + std::to_string(i), rocke_i32()));
        for(int i = 0; i < MAX_RANK - 1; ++i)
        {
            V* m = param(b, "km" + std::to_string(i), rocke_i32());
            V* s = param(b, "ks" + std::to_string(i), rocke_i32());
            km.push_back({m, s});
        }
        if(with_red)
            for(int i = 0; i < MAX_RANK; ++i)
                kA.push_back(e.stride_param(("kA" + std::to_string(i)).c_str()));
        for(int i = 0; i < MAX_RANK; ++i)
            kC.push_back(e.stride_param(("kC" + std::to_string(i)).c_str()));
        for(int i = 0; i < MAX_RANK; ++i)
            kD.push_back(e.stride_param(("kD" + std::to_string(i)).c_str()));
        if(with_red)
        {
            for(int i = 0; i < MAX_RANK; ++i)
                re.push_back(param(b, "re" + std::to_string(i), rocke_i32()));
            for(int i = 0; i < MAX_RANK - 1; ++i)
            {
                V* m = param(b, "rm" + std::to_string(i), rocke_i32());
                V* s = param(b, "rs" + std::to_string(i), rocke_i32());
                rm.push_back({m, s});
            }
            for(int i = 0; i < MAX_RANK; ++i)
                rA.push_back(e.stride_param(("rA" + std::to_string(i)).c_str()));
        }
    }

    /* coordinates of flat index x over dims first..7 (innermost first) */
    static std::vector<V*> decode(Emit& e,
                                  V* x,
                                  const std::vector<V*>& ext,
                                  const std::vector<std::pair<V*, V*>>& mg,
                                  int first = 0)
    {
        std::vector<V*> out;
        V* r = x;
        for(int i = first; i < MAX_RANK - 1; ++i)
        {
            std::pair<V*, V*> qr = e.divmod(r, ext[i], mg[i].first, mg[i].second);
            r = qr.first;
            out.push_back(qr.second);
        }
        out.push_back(r);
        return out;
    }

    static V*
        dot(Emit& e, const std::vector<V*>& coords, const std::vector<V*>& strides, int first = 0)
    {
        V* o = nullptr;
        for(size_t j = 0; j < coords.size(); ++j)
        {
            V* t = e.off(coords[j], strides[first + j]);
            o = o == nullptr ? t : rocke_b_add(e.b, o, t);
        }
        return o;
    }
};

/* reduce._epilogue: alpha * acc + beta * opC(C) */
V* epilogue(Emit& e, V* acc, V* alpha, V* beta, V* opC, V* C, V* coff, const char* d)
{
    rocke_ir_builder_t* b = e.b;
    V* zero = e.cf(0.0);
    V* anz = rocke_b_fcmp(b, "one", alpha, zero);
    V* aa = rocke_b_fmul(b, alpha, acc);
    V* v = rocke_b_select(b, anz, aa, zero);
    V* bnz = rocke_b_fcmp(b, "one", beta, zero);
    std::vector<V*> cv = e.when(bnz, {zero}, [&](const std::vector<V*>&) {
        V* c = e.load(C, coff, d)[0];
        return std::vector<V*>{e.unary(c, opC)};
    });
    V* bnz2 = rocke_b_fcmp(b, "one", beta, zero);
    V* bc = rocke_b_fmul(b, beta, cv[0]);
    V* sum = rocke_b_fadd(b, v, bc);
    return rocke_b_select(b, bnz2, sum, v);
}

std::map<std::string, V*>
    common_params(rocke_ir_builder_t* b, Emit& e, const rocke_htaot_reduce_spec_t* s, bool with_a)
{
    std::map<std::string, V*> P;
    if(with_a)
        P["A"] = ptr(b, "A", s->a, true);
    P["C"] = ptr(b, "C", s->d, true);
    P["D"] = ptr(b, "D", s->d, false);
    rocke_param_opts_t o;
    memset(&o, 0, sizeof o);
    o.align = 8;
    o.align_set = true;
    P["W"] = rocke_b_param(b, "W", rocke_ptr_type(b, e.cty, "global"), &o);
    P["alpha"] = e.scalar_param("alpha");
    P["beta"] = e.scalar_param("beta");
    if(with_a)
        P["opA"] = param(b, "opA", rocke_i32());
    P["opC"] = param(b, "opC", rocke_i32());
    P["opR"] = param(b, "opR", rocke_i32());
    P["n_out"] = param(b, "n_out", rocke_i32());
    P["split"] = param(b, "split", rocke_i32());
    return P;
}

/* reduce._finish: the epilogue result (split == 1) or the partial (split > 1) */
void finish(Emit& e,
            std::map<std::string, V*>& P,
            Geo& geo,
            V* acc,
            V* o,
            const std::vector<V*>& kc,
            V* sidx,
            const rocke_htaot_reduce_spec_t* s)
{
    rocke_ir_builder_t* b = e.b;
    V* k1 = e.c32(1);
    V* one = rocke_b_cmp_eq(b, P["split"], k1);
    scf_if(b, one, [&]() {
        V* doff = Geo::dot(e, kc, geo.kD);
        V* coff = Geo::dot(e, kc, geo.kC);
        V* v = epilogue(e, acc, P["alpha"], P["beta"], P["opC"], P["C"], coff, s->d);
        e.store(P["D"], doff, {v}, s->d);
    });
    V* split_many = rocke_b_lnot(b, one);
    scf_if(b, split_many, [&]() {
        V* sn = rocke_b_mul(b, sidx, P["n_out"]);
        V* widx = rocke_b_add(b, sn, o);
        rocke_b_global_store(b, P["W"], widx, acc, e.f64 ? 8 : 4);
    });
}

V* red_outer_count(rocke_ir_builder_t* b, Geo& geo)
{
    V* n = geo.re[1];
    for(int i = 2; i < MAX_RANK; ++i)
        n = rocke_b_mul(b, n, geo.re[i]);
    return n;
}

void row(rocke_ir_builder_t* b,
         Emit& e,
         const rocke_htaot_reduce_spec_t* s,
         std::map<std::string, V*>& P,
         Geo& geo,
         V* nchunk,
         V* jchunk,
         V* jm,
         V* js)
{
    V* tid = rocke_b_thread_id_x(b);
    V* k63 = e.c32(WAVE - 1);
    V* lane = rocke_b_land(b, tid, k63);
    V* k6 = e.c32(6);
    V* sh = rocke_b_lshr(b, tid, k6);
    V* wave = rocke_b_readfirstlane(b, sh);
    V* bx = rocke_b_block_id_x(b);
    V* kw = e.c32(s->block / WAVE);
    V* bxw = rocke_b_mul(b, bx, kw);
    V* o = rocke_b_add(b, bxw, wave);
    V* sidx = rocke_b_block_id_y(b);
    V* zero = e.cf(0.0);
    V* alpha_nz = rocke_b_fcmp(b, "one", P["alpha"], zero);
    V* in = rocke_b_cmp_lt(b, o, P["n_out"]);
    scf_if(b, in, [&]() {
        std::vector<V*> kc = Geo::decode(e, o, geo.ke, geo.km);
        V* abase = Geo::dot(e, kc, geo.kA);
        V* ident = identity(e, P["opR"]);
        V* n_ro = red_outer_count(b, geo);
        V* jtot = rocke_b_mul(b, nchunk, n_ro);
        V* j0 = rocke_b_mul(b, sidx, jchunk);
        V* j0c = rocke_b_add(b, j0, jchunk);
        V* j1 = rocke_b_smin(b, j0c, jtot);
        V* j0t = rocke_b_smin(b, j0, jtot);
        std::pair<V*, V*> rc = e.divmod(j0t, nchunk, jm, js);
        V* ro0 = rc.first;
        V* ch0 = rc.second;

        auto red = [&](const std::vector<V*>&) {
            std::vector<V*> co0 = Geo::decode(e, ro0, geo.re, geo.rm, 1);
            V* rbase0 = Geo::dot(e, co0, geo.rA, 1);
            V* step = e.c32(1);
            rocke_iter_arg_t args[4]
                = {{"racc", ident}, {"rch", ch0}, {"rro", ro0}, {"rbase", rbase0}};
            rocke_for_t loop = rocke_b_scf_for_iter(b, j0, j1, step, args, 4, "rj", false, true);
            rocke_b_region_enter(b, loop.body);
            {
                V* acc = loop.iter_vars[0];
                V* ch = loop.iter_vars[1];
                V* ro = loop.iter_vars[2];
                V* rbase = loop.iter_vars[3];
                V* kw64 = e.c32(WAVE);
                V* chw = rocke_b_mul(b, ch, kw64);
                V* i = rocke_b_add(b, chw, lane);
                V* okv = rocke_b_cmp_lt(b, i, geo.re[0]);
                V* z = e.c32(0);
                V* isafe = rocke_b_select(b, okv, i, z);
                V* ab = rocke_b_add(b, abase, rbase);
                V* io = e.off(isafe, geo.rA[0]);
                V* off = rocke_b_add(b, ab, io);
                V* ld = e.load(P["A"], off, s->a)[0];
                V* x = e.unary(ld, P["opA"]);
                x = rocke_b_select(b, okv, x, ident);
                acc = e.binary(acc, x, P["opR"]);
                V* k1 = e.c32(1);
                ch = rocke_b_add(b, ch, k1);
                V* wrap = rocke_b_cmp_eq(b, ch, nchunk);
                V* k1b = e.c32(1);
                V* ro1 = rocke_b_add(b, ro, k1b);
                V* ro2 = rocke_b_select(b, wrap, ro1, ro);
                V* z2 = e.c32(0);
                ch = rocke_b_select(b, wrap, z2, ch);
                V* lt = rocke_b_cmp_lt(b, ro2, n_ro);
                V* reb = rocke_b_land(b, wrap, lt);
                rbase = e.when(reb, {rbase}, [&](const std::vector<V*>&) {
                    std::vector<V*> co = Geo::decode(e, ro2, geo.re, geo.rm, 1);
                    return std::vector<V*>{Geo::dot(e, co, geo.rA, 1)};
                })[0];
                V* ys[4] = {acc, ch, ro2, rbase};
                rocke_b_scf_yield(b, ys, 4);
            }
            rocke_b_region_leave(b);
            V* acc = loop.op != nullptr ? loop.op->results[0] : nullptr;
            for(int m : {32, 16, 8, 4, 2, 1})
            {
                V* sx = rocke_b_warp_shuffle_xor(b, acc, m);
                acc = e.binary(acc, sx, P["opR"]);
            }
            return std::vector<V*>{acc};
        };
        V* acc = e.when(alpha_nz, {ident}, red)[0];
        V* z = e.c32(0);
        V* l0 = rocke_b_cmp_eq(b, lane, z);
        scf_if(b, l0, [&]() { finish(e, P, geo, acc, o, kc, sidx, s); });
    });
}

void col(rocke_ir_builder_t* b,
         Emit& e,
         const rocke_htaot_reduce_spec_t* s,
         std::map<std::string, V*>& P,
         Geo& geo,
         V* n_red,
         V* rchunk,
         V* ystep,
         V* n_outer)
{
    V* tid = rocke_b_thread_id_x(b);
    V* bx = rocke_b_block_id_x(b);
    V* kb = e.c32(s->block);
    V* bxb = rocke_b_mul(b, bx, kb);
    V* k0 = rocke_b_add(b, bxb, tid);
    V* k_ok = rocke_b_cmp_lt(b, k0, geo.ke[0]);
    V* z = e.c32(0);
    V* k0s = rocke_b_select(b, k_ok, k0, z);
    V* sidx = rocke_b_block_id_z(b);
    V* zero = e.cf(0.0);
    V* alpha_nz = rocke_b_fcmp(b, "one", P["alpha"], zero);
    V* ident = identity(e, P["opR"]);
    V* r0 = rocke_b_mul(b, sidx, rchunk);
    V* r0c = rocke_b_add(b, r0, rchunk);
    V* r1 = rocke_b_smin(b, r0c, n_red);
    const int U = 4;
    V* by = rocke_b_block_id_y(b);
    rocke_for_t rows = rocke_b_scf_for(b, by, n_outer, ystep, "krow");
    rocke_b_region_enter(b, rows.body);
    {
        V* ko = rows.iv;
        std::vector<V*> kc = {k0s};
        std::vector<V*> dk = Geo::decode(e, ko, geo.ke, geo.km, 1);
        kc.insert(kc.end(), dk.begin(), dk.end());
        V* abase = Geo::dot(e, kc, geo.kA);

        auto red = [&](const std::vector<V*>&) {
            V* d = rocke_b_sub(b, r1, r0);
            V* ku = e.c32(U - 1);
            V* dd = rocke_b_add(b, d, ku);
            V* kU = e.c32(U);
            V* n_it = rocke_b_div(b, dd, kU);
            V* z0 = e.c32(0);
            n_it = rocke_b_smax(b, n_it, z0);
            V* lo = e.c32(0);
            V* st = e.c32(1);
            std::vector<std::string> names;
            for(int u = 0; u < U; ++u)
                names.push_back("cacc" + std::to_string(u));
            rocke_iter_arg_t args[U];
            for(int u = 0; u < U; ++u)
            {
                args[u].name = names[u].c_str();
                args[u].init = ident;
            }
            rocke_for_t loop = rocke_b_scf_for_iter(b, lo, n_it, st, args, U, "cj", false, true);
            rocke_b_region_enter(b, loop.body);
            {
                V* j = loop.iv;
                std::vector<V*> cur(loop.iter_vars, loop.iter_vars + loop.num_iter_vars);
                V* kU2 = e.c32(U);
                V* jU = rocke_b_mul(b, j, kU2);
                V* base = rocke_b_add(b, r0, jU);
                std::vector<V*> nxt;
                if(s->math)
                {
                    std::vector<V*> oks, xs;
                    for(int u = 0; u < U; ++u)
                    {
                        V* cu = e.c32(u);
                        V* r = rocke_b_add(b, base, cu);
                        V* okv = rocke_b_cmp_lt(b, r, r1);
                        V* zz = e.c32(0);
                        V* rs = rocke_b_select(b, okv, r, zz);
                        std::vector<V*> co = Geo::decode(e, rs, geo.re, geo.rm);
                        V* dt = Geo::dot(e, co, geo.rA);
                        V* off = rocke_b_add(b, abase, dt);
                        oks.push_back(okv);
                        xs.push_back(e.load(P["A"], off, s->a)[0]);
                    }
                    xs = e.unary_list(xs, P["opA"]);
                    for(int u = 0; u < U; ++u)
                    {
                        V* x = rocke_b_select(b, oks[u], xs[u], ident);
                        nxt.push_back(e.binary(cur[u], x, P["opR"]));
                    }
                }
                else
                {
                    for(int u = 0; u < U; ++u)
                    {
                        V* cu = e.c32(u);
                        V* r = rocke_b_add(b, base, cu);
                        V* okv = rocke_b_cmp_lt(b, r, r1);
                        V* zz = e.c32(0);
                        V* rs = rocke_b_select(b, okv, r, zz);
                        std::vector<V*> co = Geo::decode(e, rs, geo.re, geo.rm);
                        V* dt = Geo::dot(e, co, geo.rA);
                        V* off = rocke_b_add(b, abase, dt);
                        V* ld = e.load(P["A"], off, s->a)[0];
                        V* x = e.unary(ld, P["opA"]);
                        x = rocke_b_select(b, okv, x, ident);
                        nxt.push_back(e.binary(cur[u], x, P["opR"]));
                    }
                }
                rocke_b_scf_yield(b, nxt.data(), (int)nxt.size());
            }
            rocke_b_region_leave(b);
            std::vector<V*> r(U, nullptr);
            if(loop.op != nullptr)
                r.assign(loop.op->results, loop.op->results + U);
            V* a01 = e.binary(r[0], r[1], P["opR"]);
            V* a23 = e.binary(r[2], r[3], P["opR"]);
            return std::vector<V*>{e.binary(a01, a23, P["opR"])};
        };
        V* acc = e.when(alpha_nz, {ident}, red)[0];
        scf_if(b, k_ok, [&]() {
            V* kk = rocke_b_mul(b, ko, geo.ke[0]);
            V* o = rocke_b_add(b, k0, kk);
            finish(e, P, geo, acc, o, kc, sidx, s);
        });
    }
    rocke_b_region_leave(b);
}

rocke_kernel_def_t* finalize(rocke_ir_builder_t* b, const rocke_htaot_reduce_spec_t* s)
{
    rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", s->block);
    Emit e(b, s->idx64, cdt_of(s), s->math);
    std::map<std::string, V*> P = common_params(b, e, s, false);
    Geo geo(b, e, false);
    V* bx = rocke_b_block_id_x(b);
    V* kb = e.c32(s->block);
    V* bxb = rocke_b_mul(b, bx, kb);
    V* tx = rocke_b_thread_id_x(b);
    V* o = rocke_b_add(b, bxb, tx);
    V* zero = e.cf(0.0);
    (void)zero;
    V* in = rocke_b_cmp_lt(b, o, P["n_out"]);
    scf_if(b, in, [&]() {
        V* ident = identity(e, P["opR"]);
        V* lo = e.c32(0);
        V* st = e.c32(1);
        rocke_iter_arg_t args[1] = {{"facc", ident}};
        rocke_for_t loop = rocke_b_scf_for_iter(b, lo, P["split"], st, args, 1, "fs", false, true);
        rocke_b_region_enter(b, loop.body);
        {
            V* sn = rocke_b_mul(b, loop.iv, P["n_out"]);
            V* wi = rocke_b_add(b, sn, o);
            V* x = rocke_b_global_load(b, P["W"], wi, e.cty, e.f64 ? 8 : 4);
            V* y = e.binary(loop.iter_vars[0], x, P["opR"]);
            rocke_b_scf_yield(b, &y, 1);
        }
        rocke_b_region_leave(b);
        V* acc = loop.op != nullptr ? loop.op->results[0] : nullptr;
        std::vector<V*> kc = Geo::decode(e, o, geo.ke, geo.km);
        V* doff = Geo::dot(e, kc, geo.kD);
        V* coff = Geo::dot(e, kc, geo.kC);
        V* v = epilogue(e, acc, P["alpha"], P["beta"], P["opC"], P["C"], coff, s->d);
        e.store(P["D"], doff, {v}, s->d);
    });
    if(!rocke_ir_builder_ok(b))
        return nullptr;
    return rocke_ir_builder_kernel(b);
}

} // namespace

extern "C" {

rocke_htaot_reduce_spec_t rocke_htaot_reduce_spec_default(void)
{
    rocke_htaot_reduce_spec_t s;
    memset(&s, 0, sizeof s);
    s.mode = "row";
    s.a = "f32";
    s.d = "f32";
    s.block = 256;
    return s;
}

rocke_status_t
    rocke_htaot_reduce_kernel_name(const rocke_htaot_reduce_spec_t* s, char* out, size_t cap)
{
    if(s == nullptr || out == nullptr)
        return ROCKE_ERR_VALUE;
    std::string n = std::string("hiptensor_aot_reduce_") + (s->mode ? s->mode : "") + "_"
                    + (s->a ? s->a : "") + "_" + (s->d ? s->d : "") + "_"
                    + (s->idx64 ? "i64" : "i32") + "_b" + std::to_string(s->block)
                    + (s->math ? "_m" : "");
    if(n.size() + 1 > cap)
        return ROCKE_ERR_VALUE;
    memcpy(out, n.c_str(), n.size() + 1);
    return ROCKE_OK;
}

static rocke_kernel_def_t* reduce_impl(rocke_ir_builder_t* b, const rocke_htaot_reduce_spec_t* s)
{
    char kname[160];
    if(b == nullptr || s == nullptr)
        return nullptr;
    if(rocke_htaot_reduce_kernel_name(s, kname, sizeof kname) != ROCKE_OK)
        snprintf(kname, sizeof kname, "hiptensor_aot_reduce");
    if(rocke_ir_builder_init(b, kname) != ROCKE_OK)
        return nullptr;
    if(!is_dtype(s->d) || (!streq(s->mode, "final") && !is_dtype(s->a)))
        return fail(b, "unknown dtype");
    if(streq(s->mode, "final"))
        return finalize(b, s);
    if(s->block % WAVE)
        return fail(b, "block must be a multiple of 64");
    if(!streq(s->mode, "row") && !streq(s->mode, "col"))
        return fail(b, s->mode != nullptr ? s->mode : "mode");
    rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", s->block);
    Emit e(b, s->idx64, cdt_of(s), s->math);
    std::map<std::string, V*> P = common_params(b, e, s, true);
    Geo geo(b, e, true);
    if(streq(s->mode, "row"))
    {
        V* nchunk = param(b, "nchunk", rocke_i32());
        V* jchunk = param(b, "jchunk", rocke_i32());
        V* jm = param(b, "jm", rocke_i32());
        V* js = param(b, "js", rocke_i32());
        row(b, e, s, P, geo, nchunk, jchunk, jm, js);
    }
    else
    {
        V* n_red = param(b, "n_red", rocke_i32());
        V* rchunk = param(b, "rchunk", rocke_i32());
        V* ystep = param(b, "ystep", rocke_i32());
        V* n_outer = param(b, "n_outer", rocke_i32());
        col(b, e, s, P, geo, n_red, rchunk, ystep, n_outer);
    }
    if(!rocke_ir_builder_ok(b))
        return nullptr;
    return rocke_ir_builder_kernel(b);
}

rocke_kernel_def_t* rocke_build_htaot_reduce(rocke_ir_builder_t* b,
                                             const rocke_htaot_reduce_spec_t* s)
{
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* { return reduce_impl(b, s); });
}

} /* extern "C" */
