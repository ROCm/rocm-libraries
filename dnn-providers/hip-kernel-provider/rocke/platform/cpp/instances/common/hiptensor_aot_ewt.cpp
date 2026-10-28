// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * hiptensor_aot_ewt.cpp -- C++ mirror of
 * library/kernels/common/hiptensor_aot/ewt.py (EwtSpec, build_ewt): the
 * shape-generic tiled elementwise kernel for transposing layouts.
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

constexpr int SLOTS_B = ROCKE_HTAOT_EWT_SLOTS_B;
constexpr int THREADS = ROCKE_HTAOT_EWT_THREADS;
constexpr int TILE_O = ROCKE_HTAOT_EWT_TILE_O;
constexpr int KIND_G = 0, KIND_V = 1, KIND_T = 2;

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

std::map<std::string, const char*> dtypes(const rocke_htaot_ewt_spec_t* s)
{
    return {{"A", s->a}, {"B", s->b}, {"C", s->c}, {"D", s->d}};
}

int nbytes(const char* dt)
{
    int nb = 0;
    dtype_of(dt, nullptr, &nb);
    return nb;
}

/* ewt._zero */
V* zero_of(rocke_ir_builder_t* b, const rocke_type_t* ty)
{
    if(rocke_type_eq(ty, rocke_f64()))
        return rocke_b_const_f64(b, 0.0);
    V* z = rocke_b_const_f32(b, 0.0);
    return rocke_type_eq(ty, rocke_f32()) ? z : rocke_b_cast_f32_to(b, z, ty);
}

} // namespace

extern "C" {

rocke_status_t rocke_htaot_ewt_kernel_name(const rocke_htaot_ewt_spec_t* s, char* out, size_t cap)
{
    if(s == nullptr || out == nullptr)
        return ROCKE_ERR_VALUE;
    std::string dts;
    const char* all[4] = {s->a, s->b, s->c, s->d};
    for(const char* x : all)
        if(present(x))
            dts += (dts.empty() ? "" : "_") + std::string(x);
    std::string n
        = "hiptensor_aot_ewt" + std::to_string(s->arity) + "_" + dts + (s->fold ? "_f" : "");
    if(n.size() + 1 > cap)
        return ROCKE_ERR_VALUE;
    memcpy(out, n.c_str(), n.size() + 1);
    return ROCKE_OK;
}

int rocke_htaot_ewt_tile_t(const rocke_htaot_ewt_spec_t* s)
{
    std::vector<std::string> names = tensors(s->arity);
    std::map<std::string, const char*> dts = dtypes(s);
    for(size_t i = 0; i + 1 < names.size(); ++i)
        if(nbytes(dts[names[i]]) == 8)
            return 32;
    return 64;
}

static rocke_kernel_def_t* ewt_impl(rocke_ir_builder_t* b, const rocke_htaot_ewt_spec_t* spec)
{
    if(b == nullptr || spec == nullptr)
        return nullptr;
    char kname[160];
    if(rocke_htaot_ewt_kernel_name(spec, kname, sizeof kname) != ROCKE_OK)
        snprintf(kname, sizeof kname, "hiptensor_aot_ewt");
    if(rocke_ir_builder_init(b, kname) != ROCKE_OK)
        return nullptr;
    /* EwtSpec.check() */
    if(spec->arity < 1 || spec->arity > 3)
        return fail(b, "arity must be 1, 2 or 3");
    if(spec->arity >= 2 && !present(spec->c))
        return fail(b, "arity >= 2 needs a C dtype");
    if(spec->arity == 3 && !present(spec->b))
        return fail(b, "arity 3 needs a B dtype");
    std::vector<std::string> names = tensors(spec->arity);
    std::map<std::string, const char*> dts = dtypes(spec);
    for(const auto& n : names)
        if(!is_dtype(dts[n]))
            return fail(b, n + ": dtype");
    std::vector<std::string> ins(names.begin(), names.end() - 1);
    const char* cdt = "f32";
    for(const auto& n : names)
        if(streq(dts[n], "f64"))
            cdt = "f64";

    rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", THREADS);
    Emit e(b, false, cdt, false);
    const int TT = rocke_htaot_ewt_tile_t(spec);
    const int TO = TILE_O;
    const int vo = 16 / nbytes(spec->d);
    auto vt_of = [&](const std::string& n) { return 16 / nbytes(dts[n]); };
    std::map<std::string, V*> P;
    for(const auto& n : names)
        P[n] = ptr(b, n.c_str(), dts[n], n != "D");
    std::map<std::string, V*> S;
    for(const auto& n : ins)
        S[n] = e.scalar_param(scal_of(n));
    std::map<std::string, V*> OPS;
    for(const auto& n : ins)
        OPS[n] = param(b, "op" + n, rocke_i32());
    if(spec->arity >= 2)
        OPS["AB"] = param(b, "opAB", rocke_i32());
    if(spec->arity == 3)
        OPS["ABC"] = param(b, "opABC", rocke_i32());
    std::map<std::string, V*> KIND;
    for(const auto& n : ins)
        KIND[n] = param(b, "k" + n, rocke_i32());
    V* eo = param(b, "eo", rocke_i32());
    V* et = param(b, "et", rocke_i32());
    V* tiles_o = param(b, "tiles_o", rocke_i32());
    V* to_m = param(b, "to_m", rocke_i32());
    V* to_s = param(b, "to_s", rocke_i32());
    V* nbv = param(b, "nb", rocke_i32());
    V* ystep = param(b, "ystep", rocke_i32());
    std::map<std::string, V*> so, st;
    for(const auto& n : names)
        so[n] = param(b, "s" + n + "o", rocke_i32());
    for(const auto& n : names)
        st[n] = param(b, "s" + n + "t", rocke_i32());
    std::vector<V*> bext;
    std::vector<std::pair<V*, V*>> bmg;
    std::map<std::string, std::vector<V*>> bst;
    for(int i = 0; i < SLOTS_B; ++i)
    {
        bext.push_back(param(b, "be" + std::to_string(i), rocke_i32()));
        if(i < SLOTS_B - 1)
        {
            V* m = param(b, "bm" + std::to_string(i), rocke_i32());
            V* s = param(b, "bs" + std::to_string(i), rocke_i32());
            bmg.push_back({m, s});
        }
        for(const auto& n : names)
            bst[n].push_back(param(b, "s" + n + std::to_string(i), rocke_i32()));
    }
    V* fo = nullptr;
    V* ft = nullptr;
    V* eo_m = nullptr;
    V* eo_s = nullptr;
    V* et_m = nullptr;
    V* et_s = nullptr;
    V* nbt = nullptr;
    if(spec->fold)
    {
        fo = param(b, "fo", rocke_i32());
        ft = param(b, "ft", rocke_i32());
        eo_m = param(b, "eo_m", rocke_i32());
        eo_s = param(b, "eo_s", rocke_i32());
        et_m = param(b, "et_m", rocke_i32());
        et_s = param(b, "et_s", rocke_i32());
        nbt = param(b, "nbt", rocke_i32());
    }

    V* c0 = rocke_b_const_i32(b, 0);
    V* tid = rocke_b_thread_id_x(b);
    V* bid = rocke_b_block_id_x(b);
    V* tile_t_idx = e.udiv(bid, to_m, to_s);
    V* tto = rocke_b_mul(b, tile_t_idx, tiles_o);
    V* tile_o_idx = rocke_b_sub(b, bid, tto);
    V* kTO = e.c32(TO);
    V* o0 = rocke_b_mul(b, tile_o_idx, kTO);
    V* kTT = e.c32(TT);
    V* t0 = rocke_b_mul(b, tile_t_idx, kTT);
    V* zero = e.cf(0.0);
    std::map<std::string, V*> nz;
    for(const auto& n : ins)
        nz[n] = rocke_b_fcmp(b, "one", S[n], zero);

    std::map<std::string, V*> lds;
    for(const auto& n : ins)
    {
        const rocke_type_t* ty = nullptr;
        int nb_ = 0;
        dtype_of(dts[n], &ty, &nb_);
        int shape[2] = {TO, TT + 16 / nb_};
        std::string hint = "t" + n;
        lds[n] = rocke_b_smem_alloc(b, ty, shape, 2, hint.c_str());
    }

    struct P1
    {
        int vt;
        V* tl;
        std::vector<V*> ors;
    };
    auto p1_lanes = [&](const std::string& n) {
        P1 r;
        r.vt = vt_of(n);
        int tpr = TT / r.vt;
        int rpp = THREADS / tpr;
        V* ktpr = e.c32(tpr);
        V* md = rocke_b_mod(b, tid, ktpr);
        V* kvt = e.c32(r.vt);
        r.tl = rocke_b_mul(b, md, kvt);
        V* ktpr2 = e.c32(tpr);
        V* r0 = rocke_b_div(b, tid, ktpr2);
        for(int p = 0; p < TO / rpp; ++p)
        {
            if(p)
            {
                V* k = e.c32(p * rpp);
                r.ors.push_back(rocke_b_add(b, r0, k));
            }
            else
                r.ors.push_back(r0);
        }
        return r;
    };

    const int cpr = TO / vo;
    const int tpp = THREADS / cpr;
    V* kcpr = e.c32(cpr);
    V* md = rocke_b_mod(b, tid, kcpr);
    V* kvo = e.c32(vo);
    V* p2_o = rocke_b_mul(b, md, kvo);
    V* kcpr2 = e.c32(cpr);
    V* p2_t0 = rocke_b_div(b, tid, kcpr2);
    std::vector<V*> p2_ts;
    for(int q = 0; q < TT / tpp; ++q)
    {
        if(q)
        {
            V* k = e.c32(q * tpp);
            p2_ts.push_back(rocke_b_add(b, p2_t0, k));
        }
        else
            p2_ts.push_back(p2_t0);
    }
    V* go = rocke_b_add(b, o0, p2_o);
    V* o_ok = rocke_b_cmp_lt(b, go, eo);
    /* folded: the batch offsets of the tile's planes (column per tensor; D's -1:
     * past the batches) */
    V* btab = nullptr;
    V* nplanes = nullptr;
    const int ncol = (int)names.size();
    if(spec->fold)
    {
        int shape[2] = {THREADS, ncol};
        btab = rocke_b_smem_alloc(b, rocke_i32(), shape, 2, "btab");
        nplanes = rocke_b_mul(b, fo, ft);
    }
    struct Plane
    {
        V* pl;
        V* oo;
        V* tt;
        V* inside;
    };
    /* (plane index, in-plane o, in-plane t, in a plane) of tile position (o_l, t_l) */
    auto plane_of = [&](V* o_l, V* t_l) {
        std::pair<V*, V*> jo = e.divmod(o_l, eo, eo_m, eo_s);
        std::pair<V*, V*> jt = e.divmod(t_l, et, et_m, et_s);
        V* lo = rocke_b_cmp_lt(b, jo.first, fo);
        V* lt = rocke_b_cmp_lt(b, jt.first, ft);
        Plane p;
        p.inside = rocke_b_land(b, lo, lt);
        /* a position outside every plane (eo or et does not divide the tile)
         * reads table row 0 instead of a row past the table (its value is
         * dropped by `inside`) */
        V* m = rocke_b_mul(b, jt.first, fo);
        V* a = rocke_b_add(b, jo.first, m);
        V* z = e.c32(0);
        p.pl = rocke_b_select(b, p.inside, a, z);
        p.oo = jo.second;
        p.tt = jt.second;
        return p;
    };
    auto tab = [&](V* plane, int k) {
        V* ck = e.c32(k);
        V* idx[2] = {plane, ck};
        V* v = rocke_b_smem_load_vN(b, btab, idx, 2, rocke_i32(), 1);
        return rocke_b_vec_extract(b, v, 0);
    };
    auto col = [&](const std::string& n) {
        int k = 0;
        while(names[(size_t)k] != n)
            ++k;
        return k;
    };

    V* by = rocke_b_block_id_y(b);
    rocke_for_t rows = rocke_b_scf_for(b, by, nbv, ystep, "bt");
    rocke_b_region_enter(b, rows.body);
    {
        std::map<std::string, V*> boff;
        V* r = rows.iv;
        if(spec->fold)
        {
            V* m = rocke_b_mul(b, rows.iv, nplanes);
            r = rocke_b_add(b, m, tid);
        }
        for(int i = 0; i < SLOTS_B; ++i)
        {
            V* d;
            if(i < SLOTS_B - 1)
            {
                std::pair<V*, V*> qr = e.divmod(r, bext[i], bmg[i].first, bmg[i].second);
                r = qr.first;
                d = qr.second;
            }
            else
                d = r;
            for(const auto& n : names)
            {
                V* t_ = rocke_b_mul(b, d, bst[n][i]);
                boff[n] = boff.count(n) == 0 ? t_ : rocke_b_add(b, boff[n], t_);
            }
        }
        rocke_b_sync(b);
        if(spec->fold)
        {
            V* lt = rocke_b_cmp_lt(b, tid, nplanes);
            scf_if(b, lt, [&]() {
                V* m = rocke_b_mul(b, rows.iv, nplanes);
                V* a = rocke_b_add(b, m, tid);
                V* live_b = rocke_b_cmp_lt(b, a, nbt);
                for(int k = 0; k < ncol; ++k)
                {
                    const std::string& n = names[(size_t)k];
                    V* v = boff[n];
                    if(n == "D")
                    {
                        V* neg = e.c32(-1);
                        v = rocke_b_select(b, live_b, v, neg);
                    }
                    V* ck = e.c32(k);
                    V* idx[2] = {tid, ck};
                    rocke_b_smem_store_vN(b, btab, idx, 2, v, 1);
                }
            });
            rocke_b_sync(b);
        }
        /* phase 1: the transposed inputs into LDS */
        for(const auto& n : ins)
        {
            V* kT = e.c32(KIND_T);
            V* isT = rocke_b_cmp_eq(b, KIND[n], kT);
            V* cond = rocke_b_land(b, isT, nz[n]);
            scf_if(b, cond, [&]() {
                P1 l = p1_lanes(n);
                const rocke_type_t* ty = nullptr;
                dtype_of(dts[n], &ty, nullptr);
                V* gt = rocke_b_add(b, t0, l.tl);
                V* t_ok = rocke_b_cmp_lt(b, gt, et);
                for(V* orow : l.ors)
                {
                    V* okv;
                    V* off;
                    if(spec->fold)
                    {
                        Plane pp = plane_of(orow, l.tl);
                        V* db = tab(pp.pl, ncol - 1);
                        V* ge = rocke_b_cmp_ge(b, db, c0);
                        okv = rocke_b_land(b, pp.inside, ge);
                        V* tb = tab(pp.pl, col(n));
                        V* m1 = rocke_b_mul(b, pp.oo, so[n]);
                        V* a1 = rocke_b_add(b, tb, m1);
                        V* m2 = rocke_b_mul(b, pp.tt, st[n]);
                        off = rocke_b_add(b, a1, m2);
                        off = rocke_b_select(b, okv, off, c0);
                    }
                    else
                    {
                        V* g_o = rocke_b_add(b, o0, orow);
                        V* lo = rocke_b_cmp_lt(b, g_o, eo);
                        okv = rocke_b_land(b, t_ok, lo);
                        V* m1 = rocke_b_mul(b, g_o, so[n]);
                        V* a1 = rocke_b_add(b, boff[n], m1);
                        V* m2 = rocke_b_mul(b, gt, st[n]);
                        off = rocke_b_add(b, a1, m2);
                        off = rocke_b_select(b, okv, off, c0);
                    }
                    V* v = rocke_b_global_load_vN(b, P[n], off, ty, l.vt, 16);
                    V* z = zero_of(b, ty);
                    std::vector<V*> zs(l.vt, z);
                    V* zv = rocke_b_vec_pack(b, zs.data(), l.vt, ty);
                    v = rocke_b_select(b, okv, v, zv);
                    V* idx[2] = {orow, l.tl};
                    rocke_b_smem_store_vN(b, lds[n], idx, 2, v, l.vt);
                }
            });
        }
        rocke_b_sync(b);
        /* phase 2: per lane vo consecutive o of each of its t rows */
        for(V* t_local : p2_ts)
        {
            Plane pp{};
            V* dbase = nullptr;
            V* gt = nullptr;
            V* okv;
            if(spec->fold)
            {
                pp = plane_of(p2_o, t_local);
                dbase = tab(pp.pl, ncol - 1);
                V* ge = rocke_b_cmp_ge(b, dbase, c0);
                okv = rocke_b_land(b, pp.inside, ge);
            }
            else
            {
                gt = rocke_b_add(b, t0, t_local);
                V* lt = rocke_b_cmp_lt(b, gt, et);
                okv = rocke_b_land(b, o_ok, lt);
            }
            std::map<std::string, std::vector<V*>> vals;
            for(const auto& n : ins)
            {
                const rocke_type_t* ty = nullptr;
                int nb_ = 0;
                dtype_of(dts[n], &ty, &nb_);
                V* base;
                if(spec->fold)
                {
                    V* tb = tab(pp.pl, col(n));
                    V* m1 = rocke_b_mul(b, pp.oo, so[n]);
                    V* a1 = rocke_b_add(b, tb, m1);
                    V* m2 = rocke_b_mul(b, pp.tt, st[n]);
                    base = rocke_b_add(b, a1, m2);
                }
                else
                {
                    V* m1 = rocke_b_mul(b, go, so[n]);
                    V* a1 = rocke_b_add(b, boff[n], m1);
                    V* m2 = rocke_b_mul(b, gt, st[n]);
                    base = rocke_b_add(b, a1, m2);
                }
                base = rocke_b_select(b, okv, base, c0);

                auto from_lds = [&](const std::vector<V*>&) {
                    std::vector<V*> out;
                    for(int j = 0; j < vo; ++j)
                    {
                        V* row = p2_o;
                        if(j)
                        {
                            V* kj = e.c32(j);
                            row = rocke_b_add(b, p2_o, kj);
                        }
                        V* idx[2] = {row, t_local};
                        V* x = rocke_b_smem_load_vN(b, lds[n], idx, 2, ty, 1);
                        V* x0 = rocke_b_vec_extract(b, x, 0);
                        out.push_back(e.to_c(x0, dts[n]));
                    }
                    return out;
                };
                auto vec_load = [&](const std::vector<V*>&) {
                    std::vector<V*> out;
                    int done = 0;
                    int w = vo < 16 / nb_ ? vo : 16 / nb_;
                    while(done < vo)
                    {
                        V* o_ = base;
                        if(done != 0)
                        {
                            V* kd = e.c32(done);
                            o_ = rocke_b_add(b, base, kd);
                        }
                        V* v = rocke_b_global_load_vN(b, P[n], o_, ty, w, w * nb_);
                        for(int k = 0; k < w; ++k)
                        {
                            V* x = rocke_b_vec_extract(b, v, k);
                            out.push_back(e.to_c(x, dts[n]));
                        }
                        done += w;
                    }
                    return out;
                };
                auto gather = [&](const std::vector<V*>&) {
                    std::vector<V*> out;
                    for(int j = 0; j < vo; ++j)
                    {
                        V* o_ = base;
                        if(j != 0)
                        {
                            V* kj = e.c32(j);
                            V* js = rocke_b_mul(b, kj, so[n]);
                            o_ = rocke_b_add(b, base, js);
                        }
                        std::vector<V*> l = e.load(P[n], o_, dts[n]);
                        out.insert(out.end(), l.begin(), l.end());
                    }
                    return out;
                };
                V* live = rocke_b_land(b, nz[n], okv);
                std::vector<V*> v(vo, zero);
                V* kT = e.c32(KIND_T);
                V* isT = rocke_b_cmp_eq(b, KIND[n], kT);
                V* cT = rocke_b_land(b, live, isT);
                v = e.when(cT, v, from_lds);
                V* kV = e.c32(KIND_V);
                V* isV = rocke_b_cmp_eq(b, KIND[n], kV);
                V* cV = rocke_b_land(b, live, isV);
                v = e.when(cV, v, vec_load);
                V* kG = e.c32(KIND_G);
                V* isG = rocke_b_cmp_eq(b, KIND[n], kG);
                V* cG = rocke_b_land(b, live, isG);
                v = e.when(cG, v, gather);
                std::vector<V*> xs = e.unary_list(v, OPS[n]);
                std::vector<V*> o;
                for(V* x : xs)
                {
                    V* p = rocke_b_fmul(b, S[n], x);
                    o.push_back(rocke_b_select(b, nz[n], p, zero));
                }
                vals[n] = o;
            }
            std::vector<V*> out;
            for(int k = 0; k < vo; ++k)
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
            scf_if(b, okv, [&]() {
                const rocke_type_t* dty = nullptr;
                dtype_of(spec->d, &dty, nullptr);
                V* doff;
                if(spec->fold)
                {
                    V* m1 = rocke_b_mul(b, pp.oo, so["D"]);
                    V* a1 = rocke_b_add(b, dbase, m1);
                    V* m2 = rocke_b_mul(b, pp.tt, st["D"]);
                    doff = rocke_b_add(b, a1, m2);
                }
                else
                {
                    V* m1 = rocke_b_mul(b, go, so["D"]);
                    V* a1 = rocke_b_add(b, boff["D"], m1);
                    V* m2 = rocke_b_mul(b, gt, st["D"]);
                    doff = rocke_b_add(b, a1, m2);
                }
                std::vector<V*> outs;
                for(V* x : out)
                    outs.push_back(e.from_c(x, spec->d));
                V* pk = rocke_b_vec_pack(b, outs.data(), (int)outs.size(), dty);
                rocke_b_global_store_vN(b, P["D"], doff, pk, vo, 16);
            });
        }
    }
    rocke_b_region_leave(b);
    if(!rocke_ir_builder_ok(b))
        return nullptr;
    return rocke_ir_builder_kernel(b);
}

rocke_kernel_def_t* rocke_build_htaot_ewt(rocke_ir_builder_t* b, const rocke_htaot_ewt_spec_t* spec)
{
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* { return ewt_impl(b, spec); });
}

} /* extern "C" */
