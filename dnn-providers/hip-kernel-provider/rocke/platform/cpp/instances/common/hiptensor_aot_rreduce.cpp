// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * hiptensor_aot_rreduce.cpp -- C++ mirror of
 * library/kernels/common/hiptensor_aot/rreduce.py (RreduceSpec / build_rreduce: the tuned
 * reduction classes with the geometry as kernel arguments; RreduceFinalSpec /
 * build_rreduce_final: their split finalize). Every builder call is made in Python's order
 * (parity family hiptensor_aot_rreduce).
 */
#include "rocke/instance_hiptensor_aot_internal.h"

#include "rocke/error_boundary.hpp"
#include "rocke/helper_rocke.helpers.transforms.h"
#include "rocke/ir_internal.h"

#include <functional>
#include <limits>
#include <map>
#include <stdio.h>
#include <string.h>
#include <string>
#include <vector>

using namespace rocke::htaot;

namespace
{

bool pow2(int x)
{
    return x > 0 && (x & (x - 1)) == 0;
}

int log2i(int x)
{
    int n = 0;
    while(x > 1)
    {
        x >>= 1;
        ++n;
    }
    return n;
}

const char* cdt_of(const char* a, const char* d)
{
    return streq(a, "f64") || streq(d, "f64") ? "f64" : "f32";
}

bool is_op(const char* op)
{
    return streq(op, "add") || streq(op, "mul") || streq(op, "max") || streq(op, "min");
}

/* LOAD_VECS */
bool load_vec_ok(const char* a, int v)
{
    if(streq(a, "f16") || streq(a, "bf16"))
        return v == 1 || v == 2 || v == 4 || v == 8;
    if(streq(a, "f32"))
        return v == 1 || v == 2 || v == 4;
    if(streq(a, "f64"))
        return v == 1 || v == 2;
    return false;
}

bool slot_count_ok(int n)
{
    return n == 1 || n == 2 || n == 3 || n == 4 || n == 8;
}

/* RreduceSpec.check */
std::string spec_check(const rocke_htaot_rreduce_spec_t* s)
{
    if(!streq(s->mode, "row") && !streq(s->mode, "col"))
        return std::string("mode '") + (s->mode ? s->mode : "") + "'";
    if(!is_dtype(s->a) || !is_dtype(s->d))
        return "unknown dtype";
    if(!is_op(s->op))
        return std::string("op '") + (s->op ? s->op : "") + "'";
    if(!slot_count_ok(s->nr))
        return "nr " + std::to_string(s->nr);
    if(s->wave != 32 && s->wave != 64)
        return "wave " + std::to_string(s->wave);
    if(!pow2(s->block) || !(s->wave <= s->block && s->block <= 1024))
        return "block " + std::to_string(s->block);
    if(!pow2(s->lanes) || s->lanes > s->block)
        return "lanes " + std::to_string(s->lanes);
    if(!load_vec_ok(s->a, s->vec))
        return "vec " + std::to_string(s->vec);
    if(s->unroll != 1 && s->unroll != 2 && s->unroll != 4 && s->unroll != 8)
        return "unroll " + std::to_string(s->unroll);
    if(s->rows && (!streq(s->mode, "col") || !pow2(s->rows) || s->lanes * s->rows > s->block))
        return "rows " + std::to_string(s->rows);
    return "";
}

std::string final_check(const rocke_htaot_rreduce_final_spec_t* f)
{
    if(!is_dtype(f->a) || !is_dtype(f->d))
        return "unknown dtype";
    if(!is_op(f->op))
        return std::string("op '") + (f->op ? f->op : "") + "'";
    if(f->wave != 32 && f->wave != 64)
        return "wave " + std::to_string(f->wave);
    if(!pow2(f->block) || !(f->wave <= f->block && f->block <= 1024))
        return "block " + std::to_string(f->block);
    if(!pow2(f->lanes) || f->lanes > f->wave || f->block % f->lanes)
        return "lanes " + std::to_string(f->lanes);
    return "";
}

std::string spec_name(const rocke_htaot_rreduce_spec_t* s)
{
    std::string rows = s->rows ? "_w" + std::to_string(s->rows) : "";
    return std::string("hiptensor_aot_rreduce_") + s->mode + "_" + s->op + "_" + s->a + "_" + s->d
           + "_r" + std::to_string(s->nr) + "_b" + std::to_string(s->block) + "_v"
           + std::to_string(s->vec) + "_l" + std::to_string(s->lanes) + rows + "_u"
           + std::to_string(s->unroll) + (s->nt ? "_nt" : "") + (s->una ? "_una" : "")
           + (s->idx64 ? "_i64" : "_i32")
           + (s->wave != 64 ? "_wave" + std::to_string(s->wave) : "");
}

std::string final_name(const rocke_htaot_rreduce_final_spec_t* f)
{
    return std::string("hiptensor_aot_rreduce_final_") + f->op + "_" + f->a + "_" + f->d + "_l"
           + std::to_string(f->lanes) + "_b" + std::to_string(f->block)
           + (f->idx64 ? "_i64" : "_i32")
           + (f->wave != 64 ? "_wave" + std::to_string(f->wave) : "");
}

/* ------------------------------------------------------------------ helpers */

/* rreduce._RedEmit: Emit with the offset helpers of these kernels */
struct RedEmit : Emit
{
    RedEmit(rocke_ir_builder_t* b_, bool idx64_, const char* cdt)
        : Emit(b_, idx64_, cdt, false)
    {
    }
    V* widen(V* v)
    {
        return idx64 ? rocke_b_sext(b, v, rocke_i64()) : v;
    }
    V* one_stride()
    {
        return idx64 ? rocke_b_const_i64(b, 1) : c32(1);
    }
};

/* _identity */
V* identity(RedEmit& e, const char* op)
{
    if(streq(op, "add"))
        return e.cf(0.0);
    if(streq(op, "mul"))
        return e.cf(1.0);
    const double inf = std::numeric_limits<double>::infinity();
    return e.cf(streq(op, "max") ? -inf : inf);
}

/* _combine: MAX / MIN propagate NaN */
V* combine(RedEmit& e, const char* op, V* a, V* c)
{
    rocke_ir_builder_t* b = e.b;
    if(streq(op, "add"))
        return rocke_b_fadd(b, a, c);
    if(streq(op, "mul"))
        return rocke_b_fmul(b, a, c);
    V* m = streq(op, "max") ? rocke_b_fmax(b, a, c) : rocke_b_fmin(b, a, c);
    V* uno = rocke_b_fcmp(b, "uno", a, c);
    V* nan = rocke_b_fadd(b, a, c);
    return rocke_b_select(b, uno, nan, m);
}

/* _loop_combine */
V* loop_combine(RedEmit& e, const char* op, V* a, V* c)
{
    rocke_ir_builder_t* b = e.b;
    if(streq(op, "add"))
        return rocke_b_fadd(b, a, c);
    if(streq(op, "mul"))
        return rocke_b_fmul(b, a, c);
    return streq(op, "max") ? rocke_b_fmax(b, a, c) : rocke_b_fmin(b, a, c);
}

/* _tree */
V* tree(RedEmit& e, const char* op, std::vector<V*> vals, bool loop)
{
    while(vals.size() > 1)
    {
        std::vector<V*> nxt;
        for(size_t i = 0; i + 1 < vals.size(); i += 2)
        {
            if(loop)
                nxt.push_back(loop_combine(e, op, vals[i], vals[i + 1]));
            else
                nxt.push_back(combine(e, op, vals[i], vals[i + 1]));
        }
        if(vals.size() % 2)
            nxt.push_back(vals.back());
        vals = nxt;
    }
    return vals[0];
}

/* _nan_any */
V* nan_any(RedEmit& e, const std::vector<V*>& vals)
{
    rocke_ir_builder_t* b = e.b;
    std::vector<V*> flags;
    for(size_t i = 0; i + 1 < vals.size(); i += 2)
        flags.push_back(rocke_b_fcmp(b, "uno", vals[i], vals[i + 1]));
    if(vals.size() % 2)
        flags.push_back(rocke_b_fcmp(b, "uno", vals.back(), vals.back()));
    V* f = flags[0];
    for(size_t i = 1; i < flags.size(); ++i)
        f = rocke_b_lor(b, f, flags[i]);
    return f;
}

/* _divmod: magic division without the d == 1 select */
std::pair<V*, V*> divmod_nosel(RedEmit& e, V* x, V* d, V* m, V* s)
{
    rocke_ir_builder_t* b = e.b;
    V* q = rocke_do_magic_division_u31_dynamic(b, x, m, s, false);
    V* qd = rocke_b_mul(b, q, d);
    V* r = rocke_b_sub(b, x, qd);
    return {q, r};
}

/* _Slots */
struct Slots
{
    int n = 0;
    std::vector<V*> ext;
    std::vector<std::pair<V*, V*>> mag;
    std::map<std::string, std::vector<V*>> strides;

    Slots(RedEmit& e, const char* prefix, int n_, const std::vector<const char*>& tensors)
        : n(n_)
    {
        rocke_ir_builder_t* b = e.b;
        std::string p(prefix);
        for(int i = 0; i < n; ++i)
            ext.push_back(param(b, p + "e" + std::to_string(i), rocke_i32()));
        for(int i = 0; i < n - 1; ++i)
        {
            V* m = param(b, p + "m" + std::to_string(i), rocke_i32());
            V* s = param(b, p + "s" + std::to_string(i), rocke_i32());
            mag.push_back({m, s});
        }
        for(const char* t : tensors)
        {
            std::vector<V*> v;
            for(int i = 0; i < n; ++i)
                v.push_back(e.stride_param((p + t + std::to_string(i)).c_str()));
            strides[t] = v;
        }
    }

    std::vector<V*> coords(RedEmit& e, V* x) const
    {
        std::vector<V*> out;
        V* r = x;
        for(int i = 0; i < n - 1; ++i)
        {
            std::pair<V*, V*> qc = divmod_nosel(e, r, ext[i], mag[i].first, mag[i].second);
            out.push_back(qc.second);
            r = qc.first;
        }
        out.push_back(r);
        return out;
    }

    V* offset(RedEmit& e, const std::vector<V*>& coords, const char* t) const
    {
        const std::vector<V*>& st = strides.at(t);
        V* o = nullptr;
        for(size_t i = 0; i < coords.size(); ++i)
        {
            V* v = e.off(coords[i], st[i]);
            o = o == nullptr ? v : rocke_b_add(e.b, o, v);
        }
        return o;
    }
};

/* _load_vec */
std::vector<V*> load_vec(RedEmit& e, V* p0, V* off, const char* dt, int vec, bool nt)
{
    rocke_ir_builder_t* b = e.b;
    const rocke_type_t* ty = nullptr;
    int nb = 0;
    dtype_of(dt, &ty, &nb);
    V* p = p0;
    V* idx = off;
    if(e.idx64)
    {
        V* k = rocke_b_const_i64(b, nb);
        V* byte = rocke_b_mul(b, off, k);
        p = rocke_b_global_ptr_add(b, p0, byte);
        idx = e.c32(0);
    }
    std::vector<V*> raw;
    if(vec == 1)
    {
        raw.push_back(rocke_b_global_load(b, p, idx, ty, nb));
    }
    else
    {
        rocke_mem_opts_t o = ROCKE_MEM_OPTS_INIT;
        o.temporal_hint = nt ? ROCKE_TEMPORAL_STREAMING : ROCKE_TEMPORAL_DEFAULT;
        V* v = rocke_b_global_load_vN_ex(b, p, idx, ty, vec, nb * vec, &o);
        for(int i = 0; i < vec; ++i)
            raw.push_back(rocke_b_vec_extract(b, v, i));
    }
    std::vector<V*> out;
    for(V* x : raw)
        out.push_back(e.to_c(x, dt));
    return out;
}

/* _store_w */
void store_w(RedEmit& e, V* W, V* off, const std::vector<V*>& vals)
{
    rocke_ir_builder_t* b = e.b;
    int nb = e.f64 ? 8 : 4;
    V* p = W;
    V* idx = off;
    if(e.idx64)
    {
        V* k = rocke_b_const_i64(b, nb);
        V* byte = rocke_b_mul(b, off, k);
        p = rocke_b_global_ptr_add(b, W, byte);
        idx = e.c32(0);
    }
    if(vals.size() == 1)
    {
        rocke_b_global_store(b, p, idx, vals[0], nb);
        return;
    }
    V* v = rocke_b_vec_pack(b, vals.data(), (int)vals.size(), e.cty);
    rocke_b_global_store_vN(b, p, idx, v, (int)vals.size(), nb * (int)vals.size());
}

/* _load_w */
V* load_w(RedEmit& e, V* W, V* off)
{
    rocke_ir_builder_t* b = e.b;
    int nb = e.f64 ? 8 : 4;
    V* p = W;
    V* idx = off;
    if(e.idx64)
    {
        V* k = rocke_b_const_i64(b, nb);
        V* byte = rocke_b_mul(b, off, k);
        p = rocke_b_global_ptr_add(b, W, byte);
        idx = e.c32(0);
    }
    return rocke_b_global_load(b, p, idx, e.cty, nb);
}

/* _store_d_vec */
void store_d_vec(RedEmit& e, V* D, V* off, const std::vector<V*>& vals, const char* dt)
{
    rocke_ir_builder_t* b = e.b;
    const rocke_type_t* ty = nullptr;
    int nb = 0;
    dtype_of(dt, &ty, &nb);
    std::vector<V*> outs;
    for(V* v : vals)
        outs.push_back(e.from_c(v, dt));
    V* p = D;
    V* idx = off;
    if(e.idx64)
    {
        V* k = rocke_b_const_i64(b, nb);
        V* byte = rocke_b_mul(b, off, k);
        p = rocke_b_global_ptr_add(b, D, byte);
        idx = e.c32(0);
    }
    V* v = rocke_b_vec_pack(b, outs.data(), (int)outs.size(), ty);
    rocke_b_global_store_vN(b, p, idx, v, (int)outs.size(), nb * (int)outs.size());
}

typedef std::map<std::string, V*> Params;

/* _epilogue */
V* epilogue(RedEmit& e, Params& P, V* acc, V* coff, const char* d)
{
    rocke_ir_builder_t* b = e.b;
    V* zero = e.cf(0.0);
    V* anz = rocke_b_fcmp(b, "one", P["alpha"], zero);
    V* aa = rocke_b_fmul(b, P["alpha"], acc);
    V* v = rocke_b_select(b, anz, aa, zero);
    V* bnz = rocke_b_fcmp(b, "one", P["beta"], zero);
    std::vector<V*> cv = e.when(bnz, {zero}, [&](const std::vector<V*>&) {
        V* c = e.load(P["C"], coff, d)[0];
        return std::vector<V*>{e.unary(c, P["opC"])};
    });
    V* bc = rocke_b_fmul(b, P["beta"], cv[0]);
    V* s = rocke_b_fadd(b, v, bc);
    return rocke_b_select(b, bnz, s, v);
}

V* ptr_param(rocke_ir_builder_t* b, const char* name, const rocke_type_t* t, bool readonly)
{
    rocke_param_opts_t o;
    memset(&o, 0, sizeof o);
    if(readonly)
    {
        o.readonly = true;
        o.readonly_set = true;
    }
    o.align = 8;
    o.align_set = true;
    return rocke_b_param(b, name, rocke_ptr_type(b, t, "global"), &o);
}

/* _common_params */
Params common_params(RedEmit& e, const rocke_htaot_rreduce_spec_t* s, bool with_a)
{
    rocke_ir_builder_t* b = e.b;
    Params P;
    if(with_a)
        P["A"] = ptr(b, "A", s->a, true);
    P["C"] = ptr(b, "C", s->d, true);
    P["D"] = ptr(b, "D", s->d, false);
    P["W"] = ptr_param(b, "W", e.cty, false);
    P["alpha"] = e.scalar_param("alpha");
    P["beta"] = e.scalar_param("beta");
    if(with_a && s->una)
        P["opA"] = param(b, "opA", rocke_i32());
    P["opC"] = param(b, "opC", rocke_i32());
    P["n_out"] = param(b, "n_out", rocke_i32());
    return P;
}

struct LoopArgs
{
    V* n_idx;
    V* per;
    V* n_it;
    V* split;
};

/* _loop_params */
LoopArgs loop_params(rocke_ir_builder_t* b)
{
    LoopArgs a;
    a.n_idx = param(b, "n_idx", rocke_i32());
    a.per = param(b, "per", rocke_i32());
    a.n_it = param(b, "n_it", rocke_i32());
    a.split = param(b, "split", rocke_i32());
    return a;
}

typedef std::function<std::vector<V*>(V*)> LoadAt;
typedef std::function<std::vector<V*>(const std::vector<V*>&)> ReduceVals;

/* _red_loop */
std::vector<V*> red_loop(RedEmit& e,
                         const rocke_htaot_rreduce_spec_t* s,
                         const LoopArgs& la,
                         V* lane,
                         int step,
                         V* sy,
                         const LoadAt& load_at,
                         int n_vals,
                         const ReduceVals& reduce_vals,
                         V* opa)
{
    rocke_ir_builder_t* b = e.b;
    const int U = s->unroll;
    const char* op = s->op;
    const bool nan_mode = streq(op, "max") || streq(op, "min");
    V* sp = rocke_b_mul(b, sy, la.per);
    V* start = rocke_b_add(b, sp, lane);
    V* end = rocke_b_add(b, sp, la.per);
    V* hi = rocke_b_smin(b, end, la.n_idx);
    V* ident = identity(e, op);
    std::vector<V*> init(U * n_vals, ident);
    if(nan_mode)
    {
        V* z = e.cf(0.0);
        for(int i = 0; i < n_vals; ++i)
            init.push_back(z);
    }
    int k = e.next_uid();
    std::vector<std::string> names;
    for(size_t i = 0; i < init.size(); ++i)
        names.push_back("racc" + std::to_string(k) + "_" + std::to_string(i));
    std::vector<rocke_iter_arg_t> args(init.size());
    for(size_t i = 0; i < init.size(); ++i)
    {
        args[i].name = names[i].c_str();
        args[i].init = init[i];
    }
    std::string iv = "rit" + std::to_string(k);
    V* lo = e.c32(0);
    V* st = e.c32(1);
    rocke_for_t loop = rocke_b_scf_for_iter(
        b, lo, la.n_it, st, args.data(), (int)args.size(), iv.c_str(), false, true);
    rocke_b_region_enter(b, loop.body);
    {
        V* it = loop.iv;
        std::vector<V*> cur(loop.iter_vars, loop.iter_vars + loop.num_iter_vars);
        V* stride = e.c32(step * U);
        V* its = rocke_b_mul(b, it, stride);
        V* base = rocke_b_add(b, start, its);
        std::vector<std::vector<V*>> loads;
        std::vector<V*> valids;
        for(int u = 0; u < U; ++u)
        {
            V* idx = base;
            if(u)
            {
                V* cu = e.c32(u * step);
                idx = rocke_b_add(b, base, cu);
            }
            V* valid = rocke_b_cmp_lt(b, idx, hi);
            V* z = e.c32(0);
            V* safe = rocke_b_select(b, valid, idx, z);
            loads.push_back(load_at(safe));
            valids.push_back(valid);
        }
        if(opa != nullptr)
        {
            std::vector<V*> flat;
            for(auto& vs : loads)
                flat.insert(flat.end(), vs.begin(), vs.end());
            flat = e.unary_list(flat, opa);
            size_t n = loads[0].size();
            for(int u = 0; u < U; ++u)
                loads[u].assign(flat.begin() + u * n, flat.begin() + (u + 1) * n);
        }
        std::vector<V*> nxt;
        for(int u = 0; u < U; ++u)
        {
            std::vector<V*> ins = reduce_vals(loads[u]);
            for(int i = 0; i < n_vals; ++i)
            {
                V* x = rocke_b_select(b, valids[u], ins[i], ident);
                nxt.push_back(loop_combine(e, op, cur[u * n_vals + i], x));
            }
        }
        if(nan_mode)
        {
            for(int i = 0; i < n_vals; ++i)
            {
                V* f = nullptr;
                for(int u = 0; u < U; ++u)
                {
                    std::vector<V*> vals = n_vals == 1 ? loads[u] : std::vector<V*>{loads[u][i]};
                    V* g = nan_any(e, vals);
                    g = rocke_b_land(b, g, valids[u]);
                    f = f == nullptr ? g : rocke_b_lor(b, f, g);
                }
                V* one = e.cf(1.0);
                nxt.push_back(rocke_b_select(b, f, one, cur[U * n_vals + i]));
            }
        }
        rocke_b_scf_yield(b, nxt.data(), (int)nxt.size());
    }
    rocke_b_region_leave(b);
    std::vector<V*> res(args.size(), nullptr);
    if(loop.op != nullptr)
        res.assign(loop.op->results, loop.op->results + loop.op->num_results);
    std::vector<V*> accs;
    for(int i = 0; i < n_vals; ++i)
    {
        std::vector<V*> parts;
        for(int u = 0; u < U; ++u)
            parts.push_back(res[u * n_vals + i]);
        V* acc = tree(e, op, parts, true);
        if(nan_mode)
        {
            V* z = e.cf(0.0);
            V* flag = rocke_b_fcmp(b, "one", res[U * n_vals + i], z);
            V* inf = e.cf(std::numeric_limits<double>::infinity());
            V* z2 = e.cf(0.0);
            V* nan = rocke_b_fmul(b, inf, z2);
            acc = rocke_b_select(b, flag, nan, acc);
        }
        accs.push_back(acc);
    }
    return accs;
}

V* opa_of(Params& P)
{
    auto it = P.find("opA");
    return it == P.end() ? nullptr : it->second;
}

/* ------------------------------------------------------------------ row */

void row(RedEmit& e, const rocke_htaot_rreduce_spec_t* s, Params& P)
{
    rocke_ir_builder_t* b = e.b;
    Slots kept(e, "k", MAX_RANK, {"A", "D"});
    Slots red(e, "r", s->nr, {"A"});
    LoopArgs la = loop_params(b);
    V* split = la.split;
    const int L = s->lanes;
    const int PB = s->block / L;
    V* tid = rocke_b_thread_id_x(b);
    V* bx = rocke_b_block_id_x(b);
    V* sy = rocke_b_block_id_y(b);
    V* lane;
    V* g;
    if(L > 1)
    {
        V* km = e.c32(L - 1);
        lane = rocke_b_land(b, tid, km);
        V* ks = e.c32(log2i(L));
        g = rocke_b_lshr(b, tid, ks);
    }
    else
    {
        lane = e.c32(0);
        g = tid;
    }
    V* o;
    if(PB > 1)
    {
        V* kp = e.c32(PB);
        V* bp = rocke_b_mul(b, bx, kp);
        o = rocke_b_add(b, bp, g);
    }
    else
    {
        o = bx;
    }
    V* o_ok = rocke_b_cmp_lt(b, o, P["n_out"]);
    V* k1 = e.c32(1);
    V* last = rocke_b_sub(b, P["n_out"], k1);
    V* o_ld = rocke_b_select(b, o_ok, o, last);
    std::vector<V*> kc = kept.coords(e, o_ld);
    V* a_base = kept.offset(e, kc, "A");
    V* d_off = kept.offset(e, kc, "D");

    LoadAt load_at = [&](V* idx) {
        std::vector<V*> rc = red.coords(e, idx);
        V* ro = red.offset(e, rc, "A");
        V* off = rocke_b_add(b, a_base, ro);
        return load_vec(e, P["A"], off, s->a, s->vec, s->nt);
    };
    ReduceVals reduce_vals
        = [&](const std::vector<V*>& vals) { return std::vector<V*>{tree(e, s->op, vals, true)}; };
    V* z = e.cf(0.0);
    V* anz = rocke_b_fcmp(b, "one", P["alpha"], z);
    V* ident = identity(e, s->op);
    V* acc = e.when(anz, {ident}, [&](const std::vector<V*>&) {
        return red_loop(e, s, la, lane, L, sy, load_at, 1, reduce_vals, opa_of(P));
    })[0];
    const int in_wave = L < s->wave ? L : s->wave;
    for(int i = 0; i < log2i(in_wave); ++i)
    {
        V* sh = rocke_b_warp_shuffle_xor(b, acc, 1 << i);
        acc = combine(e, s->op, acc, sh);
    }
    if(L > s->wave)
    {
        const int nw = L / s->wave;
        int shape[1] = {s->block / s->wave};
        V* lds = rocke_b_smem_alloc(b, e.cty, shape, 1, "rred_lds");
        V* kw = e.c32(log2i(s->wave));
        V* wid = rocke_b_lshr(b, tid, kw);
        V* kwm = e.c32(s->wave - 1);
        V* lane_w = rocke_b_land(b, tid, kwm);
        V* z0 = e.c32(0);
        V* w0 = rocke_b_cmp_eq(b, lane_w, z0);
        scf_if(b, w0, [&]() {
            V* idx[1] = {wid};
            rocke_b_smem_store_vN(b, lds, idx, 1, acc, 1);
        });
        rocke_b_sync(b);
        V* km = e.c32(~(nw - 1) & 0x7FFFFFFF);
        V* first = rocke_b_land(b, wid, km);
        std::vector<V*> parts;
        for(int i = 0; i < nw; ++i)
        {
            V* ki = e.c32(i);
            V* slot = rocke_b_add(b, first, ki);
            V* idx[1] = {slot};
            V* v = rocke_b_smem_load_vN(b, lds, idx, 1, e.cty, 1);
            parts.push_back(rocke_b_vec_extract(b, v, 0));
        }
        acc = tree(e, s->op, parts, false);
    }
    V* cond = o_ok;
    if(L > 1)
    {
        V* z0 = e.c32(0);
        V* lead = rocke_b_cmp_eq(b, lane, z0);
        cond = rocke_b_land(b, lead, o_ok);
    }
    scf_if(b, cond, [&]() {
        V* k1b = e.c32(1);
        V* one = rocke_b_cmp_eq(b, split, k1b);
        scf_if(b, one, [&]() {
            V* v = epilogue(e, P, acc, d_off, s->d);
            e.store(P["D"], d_off, {v}, s->d);
        });
        V* nonone = rocke_b_lnot(b, one);
        scf_if(b, nonone, [&]() {
            V* wn = e.widen(P["n_out"]);
            V* so = e.off(sy, wn);
            V* ow = e.widen(o);
            V* woff = rocke_b_add(b, so, ow);
            store_w(e, P["W"], woff, {acc});
        });
    });
}

/* ------------------------------------------------------------------ col */

void col(RedEmit& e, const rocke_htaot_rreduce_spec_t* s, Params& P)
{
    rocke_ir_builder_t* b = e.b;
    V* m = param(b, "m", rocke_i32());
    V* ka = e.stride_param("ka");
    V* tiles = param(b, "tiles", rocke_i32());
    V* tm = param(b, "tm", rocke_i32());
    V* ts = param(b, "ts", rocke_i32());
    V* n_items = param(b, "n_items", rocke_i32());
    Slots grp(e, "g", MAX_RANK, {"D"});
    Slots rest(e, "x", MAX_RANK, {"A", "D"});
    Slots red(e, "r", s->nr, {"A"});
    LoopArgs la = loop_params(b);
    V* split = la.split;
    const int KT = s->lanes;
    const int RT = s->rows ? s->rows : s->block / s->lanes;
    const int OB = s->block / (s->lanes * RT);
    const int vec = s->vec;
    const int tile = KT * vec;
    V* tid = rocke_b_thread_id_x(b);
    V* bx = rocke_b_block_id_x(b);
    V* sy = rocke_b_block_id_y(b);
    V* tk;
    V* tr_all;
    if(KT > 1)
    {
        V* km = e.c32(KT - 1);
        tk = rocke_b_land(b, tid, km);
        V* ks = e.c32(log2i(KT));
        tr_all = rocke_b_lshr(b, tid, ks);
    }
    else
    {
        tk = e.c32(0);
        tr_all = tid;
    }
    V* item_ok = nullptr;
    V* tr;
    V* to = nullptr;
    V* item;
    if(OB > 1)
    {
        if(RT > 1)
        {
            V* km = e.c32(RT - 1);
            tr = rocke_b_land(b, tr_all, km);
            V* ks = e.c32(log2i(RT));
            to = rocke_b_lshr(b, tr_all, ks);
        }
        else
        {
            tr = e.c32(0);
            to = tr_all;
        }
        V* kob = e.c32(OB);
        V* bo = rocke_b_mul(b, bx, kob);
        item = rocke_b_add(b, bo, to);
        item_ok = rocke_b_cmp_lt(b, item, n_items);
        V* k1 = e.c32(1);
        V* last = rocke_b_sub(b, n_items, k1);
        item = rocke_b_select(b, item_ok, item, last);
    }
    else
    {
        tr = tr_all;
        item = bx;
    }
    std::pair<V*, V*> ok_ = e.divmod(item, tiles, tm, ts);
    V* orest = ok_.first;
    V* kt = ok_.second;
    V* ktile = e.c32(tile);
    V* kt0 = rocke_b_mul(b, kt, ktile);
    V* k0;
    if(vec > 1)
    {
        V* kv = e.c32(vec);
        V* tv = rocke_b_mul(b, tk, kv);
        k0 = rocke_b_add(b, kt0, tv);
    }
    else
    {
        k0 = rocke_b_add(b, kt0, tk);
    }
    V* k_ok = rocke_b_cmp_lt(b, k0, m);
    V* z0 = e.c32(0);
    V* k_ld = rocke_b_select(b, k_ok, k0, z0);
    std::vector<V*> xc = rest.coords(e, orest);
    V* a_rest = rest.offset(e, xc, "A");
    V* d_rest = rest.offset(e, xc, "D");
    V* ak = e.off(k_ld, ka);
    V* a_base = rocke_b_add(b, a_rest, ak);
    std::vector<V*> gc = grp.coords(e, k_ld);
    V* d_grp = grp.offset(e, gc, "D");

    LoadAt load_at = [&](V* idx) {
        std::vector<V*> rc = red.coords(e, idx);
        V* ro = red.offset(e, rc, "A");
        V* off = rocke_b_add(b, a_base, ro);
        return load_vec(e, P["A"], off, s->a, vec, s->nt);
    };
    ReduceVals reduce_vals = [&](const std::vector<V*>& vals) { return vals; };
    V* zf = e.cf(0.0);
    V* anz = rocke_b_fcmp(b, "one", P["alpha"], zf);
    V* ident = identity(e, s->op);
    std::vector<V*> acc = e.when(anz, std::vector<V*>(vec, ident), [&](const std::vector<V*>&) {
        return red_loop(e, s, la, tr, RT, sy, load_at, vec, reduce_vals, opa_of(P));
    });
    const int span = KT * RT < s->wave ? KT * RT : s->wave;
    if(KT < span)
    {
        for(int i = log2i(KT); i < log2i(span); ++i)
        {
            std::vector<V*> nxt;
            for(V* v : acc)
            {
                V* sh = rocke_b_warp_shuffle_xor(b, v, 1 << i);
                nxt.push_back(combine(e, s->op, v, sh));
            }
            acc = nxt;
        }
    }
    const int rows_per_wave = s->wave / KT > 1 ? s->wave / KT : 1;
    const int n_lead = RT / rows_per_wave > 1 ? RT / rows_per_wave : 1;
    if(n_lead > 1)
    {
        int shape[1] = {OB * n_lead * KT * vec};
        V* lds = rocke_b_smem_alloc(b, e.cty, shape, 1, "rred_col_lds");
        V* lead_row;
        V* is_lead = nullptr;
        if(rows_per_wave > 1)
        {
            V* ks = e.c32(log2i(rows_per_wave));
            lead_row = rocke_b_lshr(b, tr, ks);
            V* km = e.c32(rows_per_wave - 1);
            V* tm2 = rocke_b_land(b, tr, km);
            V* zz = e.c32(0);
            is_lead = rocke_b_cmp_eq(b, tm2, zz);
        }
        else
        {
            lead_row = tr;
        }
        V* row_slot;
        if(OB > 1)
        {
            V* kn = e.c32(n_lead);
            V* tn = rocke_b_mul(b, to, kn);
            row_slot = rocke_b_add(b, tn, lead_row);
        }
        else
        {
            row_slot = lead_row;
        }
        V* kkt = e.c32(KT);
        V* rk = rocke_b_mul(b, row_slot, kkt);
        V* rkt = rocke_b_add(b, rk, tk);
        V* kv = e.c32(vec);
        V* slot = rocke_b_mul(b, rkt, kv);
        auto wr = [&]() {
            for(int i = 0; i < vec; ++i)
            {
                V* ki = e.c32(i);
                V* si = rocke_b_add(b, slot, ki);
                V* idx[1] = {si};
                rocke_b_smem_store_vN(b, lds, idx, 1, acc[i], 1);
            }
        };
        if(is_lead == nullptr)
            wr();
        else
            scf_if(b, is_lead, wr);
        rocke_b_sync(b);
        V* kv2 = e.c32(vec);
        V* base = rocke_b_mul(b, tk, kv2);
        if(OB > 1)
        {
            V* kb = e.c32(n_lead * KT * vec);
            V* tb = rocke_b_mul(b, to, kb);
            base = rocke_b_add(b, base, tb);
        }
        std::vector<V*> nw;
        for(int i = 0; i < vec; ++i)
        {
            std::vector<V*> parts;
            for(int r = 0; r < n_lead; ++r)
            {
                V* ki = e.c32(r * KT * vec + i);
                V* si = rocke_b_add(b, base, ki);
                V* idx[1] = {si};
                V* v = rocke_b_smem_load_vN(b, lds, idx, 1, e.cty, 1);
                parts.push_back(rocke_b_vec_extract(b, v, 0));
            }
            nw.push_back(tree(e, s->op, parts, false));
        }
        acc = nw;
    }
    V* cond = k_ok;
    if(RT > 1)
    {
        V* zz = e.c32(0);
        V* lead = rocke_b_cmp_eq(b, tr, zz);
        cond = rocke_b_land(b, lead, cond);
    }
    if(item_ok != nullptr)
        cond = rocke_b_land(b, cond, item_ok);
    V* d_off = rocke_b_add(b, d_rest, d_grp);
    scf_if(b, cond, [&]() {
        V* k1 = e.c32(1);
        V* one = rocke_b_cmp_eq(b, split, k1);
        scf_if(b, one, [&]() {
            V* kd = grp.strides.at("D")[0];
            std::vector<V*> outs, doffs;
            for(int i = 0; i < vec; ++i)
            {
                V* doff = d_off;
                if(i)
                {
                    V* ki = e.c32(i);
                    V* di = e.off(ki, kd);
                    doff = rocke_b_add(b, d_off, di);
                }
                outs.push_back(epilogue(e, P, acc[i], doff, s->d));
                doffs.push_back(doff);
            }
            if(vec == 1)
            {
                e.store(P["D"], d_off, {outs[0]}, s->d);
            }
            else
            {
                V* os = e.one_stride();
                V* unit = rocke_b_cmp_eq(b, kd, os);
                scf_if(b, unit, [&]() { store_d_vec(e, P["D"], d_off, outs, s->d); });
                V* nunit = rocke_b_lnot(b, unit);
                scf_if(b, nunit, [&]() {
                    for(int i = 0; i < vec; ++i)
                        e.store(P["D"], doffs[i], {outs[i]}, s->d);
                });
            }
        });
        V* nonone = rocke_b_lnot(b, one);
        scf_if(b, nonone, [&]() {
            V* om = rocke_b_mul(b, orest, m);
            V* o_flat = rocke_b_add(b, om, k0);
            V* wn = e.widen(P["n_out"]);
            V* so = e.off(sy, wn);
            V* ow = e.widen(o_flat);
            V* woff = rocke_b_add(b, so, ow);
            store_w(e, P["W"], woff, acc);
        });
    });
}

rocke_kernel_def_t* rreduce_impl(rocke_ir_builder_t* b, const rocke_htaot_rreduce_spec_t* s)
{
    if(b == nullptr || s == nullptr)
        return nullptr;
    std::string why = spec_check(s);
    std::string name = why.empty() ? spec_name(s) : std::string("hiptensor_aot_rreduce");
    if(rocke_ir_builder_init(b, name.c_str()) != ROCKE_OK)
        return nullptr;
    if(!why.empty())
        return fail(b, why);
    rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", s->block);
    RedEmit e(b, s->idx64, cdt_of(s->a, s->d));
    Params P = common_params(e, s, true);
    if(streq(s->mode, "row"))
        row(e, s, P);
    else
        col(e, s, P);
    if(!rocke_ir_builder_ok(b))
        return nullptr;
    return rocke_ir_builder_kernel(b);
}

rocke_kernel_def_t* final_impl(rocke_ir_builder_t* b, const rocke_htaot_rreduce_final_spec_t* f)
{
    if(b == nullptr || f == nullptr)
        return nullptr;
    std::string why = final_check(f);
    std::string name = why.empty() ? final_name(f) : std::string("hiptensor_aot_rreduce_final");
    if(rocke_ir_builder_init(b, name.c_str()) != ROCKE_OK)
        return nullptr;
    if(!why.empty())
        return fail(b, why);
    rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", f->block);
    RedEmit e(b, f->idx64, cdt_of(f->a, f->d));
    Params P;
    P["W"] = ptr_param(b, "W", e.cty, true);
    P["C"] = ptr(b, "C", f->d, true);
    P["D"] = ptr(b, "D", f->d, false);
    P["alpha"] = e.scalar_param("alpha");
    P["beta"] = e.scalar_param("beta");
    P["opC"] = param(b, "opC", rocke_i32());
    P["n_out"] = param(b, "n_out", rocke_i32());
    V* split = param(b, "split", rocke_i32());
    Slots kept(e, "k", MAX_RANK, {"D"});
    const int L = f->lanes;
    V* tid = rocke_b_thread_id_x(b);
    V* bx = rocke_b_block_id_x(b);
    V* o;
    V* lane;
    if(L > 1)
    {
        V* kb = e.c32(f->block / L);
        V* bo = rocke_b_mul(b, bx, kb);
        V* ks = e.c32(log2i(L));
        V* g = rocke_b_lshr(b, tid, ks);
        o = rocke_b_add(b, bo, g);
        V* km = e.c32(L - 1);
        lane = rocke_b_land(b, tid, km);
    }
    else
    {
        V* kb = e.c32(f->block);
        V* bo = rocke_b_mul(b, bx, kb);
        o = rocke_b_add(b, bo, tid);
        lane = e.c32(0);
    }
    V* ok = rocke_b_cmp_lt(b, o, P["n_out"]);
    V* k1 = e.c32(1);
    V* last = rocke_b_sub(b, P["n_out"], k1);
    V* o_ld = rocke_b_select(b, ok, o, last);
    V* ident = identity(e, f->op);
    V* trips;
    if(L > 1)
    {
        V* kl = e.c32(L - 1);
        V* sl = rocke_b_add(b, split, kl);
        V* ks = e.c32(log2i(L));
        trips = rocke_b_lshr(b, sl, ks);
    }
    else
    {
        trips = split;
    }
    int k = e.next_uid();
    std::string an = "facc" + std::to_string(k);
    std::string iv = "fit" + std::to_string(k);
    rocke_iter_arg_t args[1] = {{an.c_str(), ident}};
    V* lo = e.c32(0);
    V* st = e.c32(1);
    rocke_for_t loop = rocke_b_scf_for_iter(b, lo, trips, st, args, 1, iv.c_str(), false, true);
    rocke_b_region_enter(b, loop.body);
    {
        V* kl = e.c32(L);
        V* jl = rocke_b_mul(b, loop.iv, kl);
        V* si = rocke_b_add(b, jl, lane);
        V* valid = rocke_b_cmp_lt(b, si, split);
        V* z = e.c32(0);
        V* safe = rocke_b_select(b, valid, si, z);
        V* wn = e.widen(P["n_out"]);
        V* so = e.off(safe, wn);
        V* ow = e.widen(o_ld);
        V* woff = rocke_b_add(b, so, ow);
        V* v = load_w(e, P["W"], woff);
        V* x = rocke_b_select(b, valid, v, ident);
        V* y = combine(e, f->op, loop.iter_vars[0], x);
        rocke_b_scf_yield(b, &y, 1);
    }
    rocke_b_region_leave(b);
    V* acc = loop.op != nullptr ? loop.op->results[0] : nullptr;
    for(int i = 0; i < log2i(L); ++i)
    {
        V* sh = rocke_b_warp_shuffle_xor(b, acc, 1 << i);
        acc = combine(e, f->op, acc, sh);
    }
    std::vector<V*> kc = kept.coords(e, o_ld);
    V* d_off = kept.offset(e, kc, "D");
    V* cond = ok;
    if(L > 1)
    {
        V* z = e.c32(0);
        V* lead = rocke_b_cmp_eq(b, lane, z);
        cond = rocke_b_land(b, ok, lead);
    }
    scf_if(b, cond, [&]() {
        V* v = epilogue(e, P, acc, d_off, f->d);
        e.store(P["D"], d_off, {v}, f->d);
    });
    if(!rocke_ir_builder_ok(b))
        return nullptr;
    return rocke_ir_builder_kernel(b);
}

bool copy_out(const std::string& n, char* out, size_t cap)
{
    if(out == nullptr || n.size() + 1 > cap)
        return false;
    memcpy(out, n.c_str(), n.size() + 1);
    return true;
}

} // namespace

extern "C" {

rocke_htaot_rreduce_spec_t rocke_htaot_rreduce_spec_default(void)
{
    rocke_htaot_rreduce_spec_t s;
    memset(&s, 0, sizeof s);
    s.mode = "row";
    s.a = "f32";
    s.d = "f32";
    s.op = "add";
    s.nr = 1;
    s.block = 256;
    s.vec = 1;
    s.lanes = 64;
    s.unroll = 1;
    s.rows = 0;
    s.wave = 64;
    return s;
}

bool rocke_htaot_rreduce_check(const rocke_htaot_rreduce_spec_t* s, char* reason, size_t cap)
{
    if(s == nullptr)
        return false;
    std::string why = spec_check(s);
    if(reason != nullptr && cap > 0)
        snprintf(reason, cap, "%s", why.c_str());
    return why.empty();
}

rocke_status_t
    rocke_htaot_rreduce_kernel_name(const rocke_htaot_rreduce_spec_t* s, char* out, size_t cap)
{
    if(s == nullptr || !spec_check(s).empty())
        return ROCKE_ERR_VALUE;
    return copy_out(spec_name(s), out, cap) ? ROCKE_OK : ROCKE_ERR_VALUE;
}

rocke_kernel_def_t* rocke_build_htaot_rreduce(rocke_ir_builder_t* b,
                                              const rocke_htaot_rreduce_spec_t* s)
{
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* { return rreduce_impl(b, s); });
}

rocke_htaot_rreduce_final_spec_t rocke_htaot_rreduce_final_spec_default(void)
{
    rocke_htaot_rreduce_final_spec_t f;
    memset(&f, 0, sizeof f);
    f.a = "f32";
    f.d = "f32";
    f.op = "add";
    f.lanes = 1;
    f.block = 256;
    f.wave = 64;
    return f;
}

bool rocke_htaot_rreduce_final_check(const rocke_htaot_rreduce_final_spec_t* f,
                                     char* reason,
                                     size_t cap)
{
    if(f == nullptr)
        return false;
    std::string why = final_check(f);
    if(reason != nullptr && cap > 0)
        snprintf(reason, cap, "%s", why.c_str());
    return why.empty();
}

rocke_status_t rocke_htaot_rreduce_final_kernel_name(const rocke_htaot_rreduce_final_spec_t* f,
                                                     char* out,
                                                     size_t cap)
{
    if(f == nullptr || !final_check(f).empty())
        return ROCKE_ERR_VALUE;
    return copy_out(final_name(f), out, cap) ? ROCKE_OK : ROCKE_ERR_VALUE;
}

rocke_kernel_def_t* rocke_build_htaot_rreduce_final(rocke_ir_builder_t* b,
                                                    const rocke_htaot_rreduce_final_spec_t* f)
{
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* { return final_impl(b, f); });
}

} /* extern "C" */
