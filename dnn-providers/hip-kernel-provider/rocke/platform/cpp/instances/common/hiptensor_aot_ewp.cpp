// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * hiptensor_aot_ewp.cpp -- C++ mirror of library/kernels/common/hiptensor_aot/ewp.py
 * (EwpSpec, build_ewp): the tuned tiled elementwise kernel for transposing
 * layouts. Every builder call follows the Python builder's order (the hiptensor_aot_ewp parity
 * family keeps the two byte-identical).
 */
#include "rocke/instance_hiptensor_aot_internal.h"

#include "rocke/error_boundary.hpp"
#include "rocke/ir_internal.h"

#include <stdio.h>
#include <string.h>

#include <functional>
#include <map>
#include <string>
#include <utility>
#include <vector>

using namespace rocke::htaot;

namespace
{

constexpr int SLOTS_B = ROCKE_HTAOT_EWP_SLOTS_B;

bool set_reason(char* reason, size_t cap, const std::string& msg)
{
    if(reason != nullptr && cap > 0)
        snprintf(reason, cap, "%s", msg.c_str());
    return false;
}

std::vector<std::string> tensors(int arity)
{
    if(arity == 1)
        return {"A", "D"};
    if(arity == 2)
        return {"A", "C", "D"};
    return {"A", "B", "C", "D"};
}

std::map<std::string, const char*> dtypes(const rocke_htaot_ewp_spec_t* s)
{
    return {{"A", s->a}, {"B", s->b}, {"C", s->c}, {"D", s->d}};
}

int nbytes(const char* dt)
{
    int nb = 0;
    dtype_of(dt, nullptr, &nb);
    return nb;
}

bool pow2(int x)
{
    return x > 0 && (x & (x - 1)) == 0;
}

int log2i(int x)
{
    int r = 0;
    while((1 << (r + 1)) <= x)
        ++r;
    return r;
}

bool mode_a(const rocke_htaot_ewp_spec_t* s)
{
    return streq(s->mode, "a");
}

int vec_of(const rocke_htaot_ewp_spec_t* s)
{
    int nb = nbytes(s->d);
    return nb > 0 ? s->vb / nb : 0;
}

int vect_of(const rocke_htaot_ewp_spec_t* s)
{
    int nb = nbytes(s->d);
    return nb > 0 ? (s->tb ? s->tb : s->vb) / nb : 0;
}

int64_t p1_units(const rocke_htaot_ewp_spec_t* s)
{
    if(mode_a(s))
        return int64_t(s->to) * s->tt * s->tw;
    int v = vec_of(s);
    return int64_t(s->to / v) * (s->tt / vect_of(s));
}

int64_t p2_units(const rocke_htaot_ewp_spec_t* s)
{
    if(mode_a(s))
        return int64_t(s->to) * s->tt * s->tw;
    return int64_t(s->to / vec_of(s)) * s->tt;
}

int64_t tile_bytes(const rocke_htaot_ewp_spec_t* s)
{
    if(mode_a(s))
        return int64_t(s->to) * s->tt * s->tw * s->vb;
    return int64_t(s->to) * s->tt * nbytes(s->d);
}

int swlo_of(const rocke_htaot_ewp_spec_t* s)
{
    return mode_a(s) ? log2i(s->tw) : 0;
}

const char* cdt_of(const rocke_htaot_ewp_spec_t* s)
{
    const char* dts[4] = {s->a, s->b, s->c, s->d};
    for(const char* x : dts)
        if(streq(x, "f64"))
            return "f64";
    return "f32";
}

std::string ewp_name(const rocke_htaot_ewp_spec_t* s)
{
    std::string dts;
    const char* all[4] = {s->a, s->b, s->c, s->d};
    for(const char* x : all)
        if(present(x))
            dts += (dts.empty() ? "" : "_") + std::string(x);
    std::string n = "hiptensor_aot_ewp" + std::to_string(s->arity) + "_" + dts + "_"
                    + std::string(s->kinds) + "_" + std::string(s->mode) + "_o"
                    + std::to_string(s->to) + "_t" + std::to_string(s->tt);
    if(mode_a(s))
        n += "_w" + std::to_string(s->tw);
    n += "_b" + std::to_string(s->block);
    if(s->nt)
        n += "_nt" + std::to_string(s->nt);
    if(s->swz > 1)
        n += "_z" + std::to_string(s->swz) + "s" + std::to_string(s->swsh);
    if(!streq(s->walk, "o"))
        n += "_wt";
    if(s->tpc > 1)
        n += "_x" + std::to_string(s->tpc);
    if(s->ilp)
        n += "_g" + std::to_string(s->ilp);
    if(s->vb != 16)
        n += "_v" + std::to_string(s->vb);
    if(s->rag)
        n += "_r";
    if(s->tb)
        n += "_u" + std::to_string(s->tb);
    return n;
}

} // namespace

extern "C" {

rocke_htaot_ewp_spec_t rocke_htaot_ewp_spec_default(void)
{
    rocke_htaot_ewp_spec_t s{};
    s.arity = 1;
    s.a = "f32";
    s.d = "f32";
    s.b = nullptr;
    s.c = nullptr;
    s.kinds = "T";
    s.mode = "t";
    s.to = 64;
    s.tt = 64;
    s.tw = 1;
    s.block = 256;
    s.nt = 0;
    s.swz = 0;
    s.swsh = 0;
    s.walk = "o";
    s.tpc = 1;
    s.ilp = 0;
    s.vb = 16;
    s.rag = 0;
    s.tb = 0;
    return s;
}

int rocke_htaot_ewp_vec(const rocke_htaot_ewp_spec_t* s)
{
    return s == nullptr ? 0 : vec_of(s);
}

int rocke_htaot_ewp_vect(const rocke_htaot_ewp_spec_t* s)
{
    return s == nullptr ? 0 : vect_of(s);
}

int rocke_htaot_ewp_chunks(const rocke_htaot_ewp_spec_t* s)
{
    return s == nullptr || s->vb <= 0 ? 0 : (int)(tile_bytes(s) / s->vb);
}

bool rocke_htaot_ewp_check(const rocke_htaot_ewp_spec_t* s, char* reason, size_t cap)
{
    if(s == nullptr)
        return set_reason(reason, cap, "null spec");
    if(s->arity < 1 || s->arity > 3)
        return set_reason(reason, cap, "arity must be 1, 2 or 3");
    if(s->arity >= 2 && !present(s->c))
        return set_reason(reason, cap, "arity >= 2 needs a C dtype");
    if(s->arity == 3 && !present(s->b))
        return set_reason(reason, cap, "arity 3 needs a B dtype");
    std::vector<std::string> names = tensors(s->arity);
    std::map<std::string, const char*> dts = dtypes(s);
    if(!is_dtype(s->d))
        return set_reason(reason, cap, "D: dtype");
    for(const auto& n : names)
    {
        if(!is_dtype(dts[n]))
            return set_reason(reason, cap, n + ": dtype");
        if(nbytes(dts[n]) != nbytes(s->d))
            return set_reason(reason, cap, "every tensor must have the same element size");
    }
    if(s->kinds == nullptr || (int)strlen(s->kinds) != s->arity)
        return set_reason(reason, cap, "kinds: one of T / V per input");
    bool anyT = false;
    for(int i = 0; i < s->arity; ++i)
    {
        if(s->kinds[i] != 'T' && s->kinds[i] != 'V')
            return set_reason(reason, cap, "kinds: one of T / V per input");
        anyT = anyT || s->kinds[i] == 'T';
    }
    if(!anyT)
        return set_reason(reason, cap, "at least one input read through LDS (kind T)");
    if(!streq(s->mode, "t") && !streq(s->mode, "a"))
        return set_reason(reason, cap, "mode must be t or a");
    if((s->vb != 2 && s->vb != 4 && s->vb != 8 && s->vb != 16) || s->vb < nbytes(s->d))
        return set_reason(
            reason, cap, "vb must be 2, 4, 8 or 16 bytes and at least the element size");
    if(s->rag != 0 && s->rag != 1)
        return set_reason(reason, cap, "rag must be 0 or 1");
    if(s->tb
       && ((s->tb != 2 && s->tb != 4 && s->tb != 8 && s->tb != 16) || s->tb < nbytes(s->d)
           || !streq(s->mode, "t")))
        return set_reason(
            reason, cap, "tb: 2, 4, 8 or 16 bytes, at least the element size, mode t only");
    if(!pow2(s->to) || !pow2(s->tt) || !pow2(s->tw) || !pow2(s->block))
        return set_reason(reason, cap, "tile extents and block must be powers of two");
    const int v = vec_of(s);
    if(!mode_a(s) && (s->to % v || s->tt % vect_of(s)))
        return set_reason(reason, cap, "mode t: the tile extents must be multiples of the vector");
    if(!mode_a(s) && s->tw != 1)
        return set_reason(reason, cap, "mode t: tw must be 1");
    if(s->block < 64 || s->block > 1024)
        return set_reason(reason, cap, "block must be 64 .. 1024");
    if(p1_units(s) < 1 || p2_units(s) < 1)
        return set_reason(reason, cap, "empty tile");
    int64_t held = (p1_units(s) + s->block - 1) / s->block * (mode_a(s) ? 1 : v);
    if(held > ROCKE_HTAOT_EWP_MAX_VECTORS)
        return set_reason(reason,
                          cap,
                          std::to_string(held) + " vectors per thread in phase 1 > "
                              + std::to_string(ROCKE_HTAOT_EWP_MAX_VECTORS));
    int nT = 0;
    for(int i = 0; i < s->arity; ++i)
        nT += s->kinds[i] == 'T';
    if(nT * tile_bytes(s) > ROCKE_HTAOT_EWP_LDS_BYTES)
        return set_reason(reason, cap, "the LDS tiles exceed the LDS budget");
    if(s->nt < 0 || s->nt > 3)
        return set_reason(reason, cap, "nt must be 0 .. 3");
    if(v == 1 && (s->nt & 1))
        return set_reason(
            reason, cap, "one-element vectors: scalar loads have no streaming hint (nt bit 0)");
    if(s->swz != 0 && s->swz != 1)
    {
        int64_t chunks = tile_bytes(s) / s->vb;
        if(!pow2(s->swz) || (int64_t(1) << s->swsh) < (int64_t(s->swz) << swlo_of(s))
           || chunks % (int64_t(1) << s->swsh))
            return set_reason(reason, cap, "the swizzle is not a bijection of the tile's chunks");
    }
    if(!streq(s->walk, "o") && !streq(s->walk, "t"))
        return set_reason(reason, cap, "walk must be o or t");
    if(s->tpc != 1 && s->tpc != 2 && s->tpc != 4 && s->tpc != 8)
        return set_reason(reason, cap, "tpc must be 1, 2, 4 or 8");
    if(s->ilp != 0 && s->ilp != 1 && s->ilp != 2 && s->ilp != 4 && s->ilp != 8)
        return set_reason(reason, cap, "ilp must be 0, 1, 2, 4 or 8");
    if(reason != nullptr && cap > 0)
        reason[0] = '\0';
    return true;
}

rocke_status_t rocke_htaot_ewp_kernel_name(const rocke_htaot_ewp_spec_t* s, char* out, size_t cap)
{
    if(s == nullptr || out == nullptr || s->kinds == nullptr || s->mode == nullptr
       || s->walk == nullptr)
        return ROCKE_ERR_VALUE;
    std::string n = ewp_name(s);
    if(n.size() + 1 > cap)
        return ROCKE_ERR_VALUE;
    memcpy(out, n.c_str(), n.size() + 1);
    return ROCKE_OK;
}

/* ewp._binary_fixed: Emit::binary of a known operator */
static V* binary_fixed(rocke_ir_builder_t* b, V* a, V* c, int op)
{
    if(op == OP_ADD)
        return rocke_b_fadd(b, a, c);
    if(op == OP_MUL)
        return rocke_b_fmul(b, a, c);
    V* r = op == OP_MAX ? rocke_b_fmax(b, a, c) : rocke_b_fmin(b, a, c);
    V* uno = rocke_b_fcmp(b, "uno", a, c);
    V* nan = rocke_b_fadd(b, a, c);
    return rocke_b_select(b, uno, nan, r);
}

static rocke_kernel_def_t* ewp_impl(rocke_ir_builder_t* b, const rocke_htaot_ewp_spec_t* spec)
{
    if(b == nullptr || spec == nullptr)
        return nullptr;
    char reason[200];
    bool valid = rocke_htaot_ewp_check(spec, reason, sizeof reason);
    std::string kname = valid ? ewp_name(spec) : std::string();
    if(rocke_ir_builder_init(b, kname.empty() ? "hiptensor_aot_ewp" : kname.c_str()) != ROCKE_OK)
        return nullptr;
    if(!valid)
        return fail(b, reason);
    rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", spec->block);
    Emit e(b, false, cdt_of(spec), false);
    std::vector<std::string> names = tensors(spec->arity);
    std::vector<std::string> ins(names.begin(), names.end() - 1);
    std::map<std::string, const char*> dts = dtypes(spec);
    const int VW = vec_of(spec);
    const int VT = vect_of(spec);
    const int BS = spec->block;
    const int TO = spec->to, TT = spec->tt, TW = spec->tw;
    const bool ma = mode_a(spec);
    rocke_mem_opts_t mo_ld = ROCKE_MEM_OPTS_INIT;
    mo_ld.temporal_hint = (spec->nt & 1) ? ROCKE_TEMPORAL_STREAMING : ROCKE_TEMPORAL_DEFAULT;
    rocke_mem_opts_t mo_st = ROCKE_MEM_OPTS_INIT;
    mo_st.temporal_hint = (spec->nt & 2) ? ROCKE_TEMPORAL_STREAMING : ROCKE_TEMPORAL_DEFAULT;
    auto kind = [&](const std::string& n) {
        for(size_t i = 0; i < ins.size(); ++i)
            if(ins[i] == n)
                return spec->kinds[i];
        return 'V';
    };
    std::vector<std::string> tins;
    for(const auto& n : ins)
        if(kind(n) == 'T')
            tins.push_back(n);

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
    V* eo = param(b, "eo", rocke_i32());
    V* om = param(b, "om", rocke_i32());
    V* os_ = param(b, "os", rocke_i32());
    V* et = param(b, "et", rocke_i32());
    V* tm = param(b, "tm", rocke_i32());
    V* ts = param(b, "ts", rocke_i32());
    V* nw = param(b, "nw", rocke_i32());
    V* wm = param(b, "wm", rocke_i32());
    V* ws = param(b, "ws", rocke_i32());
    V* nf = param(b, "nf", rocke_i32());
    V* fm = param(b, "fm", rocke_i32());
    V* fs = param(b, "fs", rocke_i32());
    V* ntl = param(b, "ntl", rocke_i32());
    V* nbv = param(b, "nb", rocke_i32());
    V* ystep = param(b, "ystep", rocke_i32());
    V* lo = nullptr;
    V* lt = nullptr;
    V* lw = nullptr;
    if(spec->rag)
    {
        lo = param(b, "lo", rocke_i32());
        lt = param(b, "lt", rocke_i32());
        if(ma)
            lw = param(b, "lw", rocke_i32());
    }
    std::map<std::string, std::pair<V*, V*>> SO, ST;
    for(const auto& n : names)
    {
        V* a0 = param(b, "s" + n + "o0", rocke_i32());
        V* a1 = param(b, "s" + n + "o1", rocke_i32());
        SO[n] = {a0, a1};
    }
    for(const auto& n : names)
    {
        V* a0 = param(b, "s" + n + "t0", rocke_i32());
        V* a1 = param(b, "s" + n + "t1", rocke_i32());
        ST[n] = {a0, a1};
    }
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

    V* tid = rocke_b_thread_id_x(b);
    V* bid = rocke_b_block_id_x(b);
    V* first = bid;
    if(spec->tpc > 1)
    {
        V* k = e.c32(spec->tpc);
        first = rocke_b_mul(b, bid, k);
    }
    V* zero = e.cf(0.0);
    std::map<std::string, V*> nz;
    for(const auto& n : ins)
        nz[n] = rocke_b_fcmp(b, "one", S[n], zero);
    V* allnz = nz[ins[0]];
    for(size_t i = 1; i < ins.size(); ++i)
        allnz = rocke_b_land(b, allnz, nz[ins[i]]);
    /* the fast path: every scale non-zero and every unary operator the identity (CONJ of a
     * real; hipTensor folds NEG into the scale) */
    V* idop = nullptr;
    for(const auto& n : ins)
    {
        V* kid = e.c32(OP_IDENTITY);
        V* isid = rocke_b_cmp_eq(b, OPS[n], kid);
        V* kcj = e.c32(OP_CONJ);
        V* iscj = rocke_b_cmp_eq(b, OPS[n], kcj);
        V* t_ = rocke_b_lor(b, isid, iscj);
        idop = idop == nullptr ? t_ : rocke_b_land(b, idop, t_);
    }
    V* fastp = rocke_b_land(b, allnz, idop);
    /* ragged class: the last position a lane may touch along each run (a lane past the end
     * takes the last vector of the run: the same data as the lane that owns it) */
    V* last_w = nullptr;
    V* last_o1 = nullptr;
    V* last_t1 = nullptr;
    V* last_ov = nullptr;
    V* last_tv = nullptr;
    if(spec->rag)
    {
        if(ma)
        {
            V* k1 = e.c32(VW);
            last_w = rocke_b_sub(b, lw, k1);
            V* k2 = e.c32(1);
            last_o1 = rocke_b_sub(b, lo, k2);
            V* k3 = e.c32(1);
            last_t1 = rocke_b_sub(b, lt, k3);
        }
        else
        {
            V* k1 = e.c32(VW);
            last_ov = rocke_b_sub(b, lo, k1);
            V* k2 = e.c32(VT);
            last_tv = rocke_b_sub(b, lt, k2);
            V* k3 = e.c32(1);
            last_t1 = rocke_b_sub(b, lt, k3);
        }
    }
    /* a vector of w elements of tensor n (<w x ty>, default vb bytes; w = 1: one scalar load,
     * no streaming hint) */
    auto gload = [&](const std::string& n, V* off, int w) {
        const rocke_type_t* ty = nullptr;
        int nb = 0;
        dtype_of(dts[n], &ty, &nb);
        const int bytes = w * nb;
        if(w == 1)
        {
            V* x = rocke_b_global_load(b, P[n], off, ty, bytes);
            V* xs[1] = {x};
            return rocke_b_vec_pack(b, xs, 1, ty);
        }
        return rocke_b_global_load_vN_ex(b, P[n], off, ty, w, bytes, &mo_ld);
    };

    auto run_off = [&](const std::string& n, V* pos, V* ext, V* m, V* s, std::pair<V*, V*> st) {
        std::pair<V*, V*> qr = e.divmod(pos, ext, m, s);
        V* a = rocke_b_mul(b, qr.second, st.first);
        V* c = rocke_b_mul(b, qr.first, st.second);
        (void)n;
        return rocke_b_add(b, a, c);
    };
    auto ooff = [&](const std::string& n, V* pos) { return run_off(n, pos, eo, om, os_, SO[n]); };
    auto toff = [&](const std::string& n, V* pos) { return run_off(n, pos, et, tm, ts, ST[n]); };

    std::map<std::string, V*> lds;
    for(const auto& n : tins)
    {
        const rocke_type_t* ty = nullptr;
        dtype_of(dts[n], &ty, nullptr);
        int shape[1] = {(int)(tile_bytes(spec) / spec->vb) * VW};
        std::string hint = "t" + n;
        lds[n] = rocke_b_smem_alloc(b, ty, shape, 1, hint.c_str());
    }
    const int swlo = swlo_of(spec);
    auto swz = [&](V* chunk) {
        if(spec->swz <= 1)
            return chunk;
        V* ks = e.c32(spec->swsh);
        V* sh = rocke_b_lshr(b, chunk, ks);
        V* km = e.c32(spec->swz - 1);
        V* key = rocke_b_land(b, sh, km);
        if(swlo)
        {
            V* kl = e.c32(swlo);
            key = rocke_b_shl(b, key, kl);
        }
        return rocke_b_xor(b, chunk, key);
    };
    auto units = [&](int64_t total) {
        std::vector<V*> out;
        for(int64_t it = 0; it < (total + BS - 1) / BS; ++it)
        {
            V* u = tid;
            if(it != 0)
            {
                V* k = e.c32(it * BS);
                u = rocke_b_add(b, tid, k);
            }
            if((it + 1) * BS > total)
            {
                V* k = e.c32(total - 1);
                u = rocke_b_smin(b, u, k);
            }
            out.push_back(u);
        }
        return out;
    };
    auto pow2_div = [&](V* x, int d) {
        if(d == 1)
            return x;
        V* k = e.c32(log2i(d));
        return rocke_b_lshr(b, x, k);
    };
    auto pow2_mod = [&](V* x, int d) {
        if(d == 1)
            return e.c32(0);
        V* k = e.c32(d - 1);
        return rocke_b_land(b, x, k);
    };
    auto vmul = [&](V* x, int64_t c) {
        V* k = e.c32(c);
        return rocke_b_mul(b, x, k);
    };
    auto lds_index = [&](V* ch) {
        V* sw = swz(ch);
        return vmul(sw, VW);
    };
    auto lstore = [&](const std::string& n, V* ch, V* v) {
        V* idx[1] = {lds_index(ch)};
        if(VW == 1)
        {
            V* x = rocke_b_vec_extract(b, v, 0);
            rocke_b_smem_store_vN(b, lds[n], idx, 1, x, 1);
        }
        else
            rocke_b_smem_store_vN(b, lds[n], idx, 1, v, VW);
    };

    struct Unit
    {
        V* ch;
        V* go;
        V* gt;
        V* wv; /* mode a only */
    };

    auto one_tile = [&](V* tile_id, std::map<std::string, V*>& boff) {
        std::pair<V*, V*> q1 = e.divmod(tile_id, nw, wm, ws);
        V* tile_w = q1.second;
        std::pair<V*, V*> q2 = e.divmod(q1.first, nf, fm, fs);
        V* slow = q2.first;
        V* fast = q2.second;
        V* tile_o = streq(spec->walk, "o") ? fast : slow;
        V* tile_t = streq(spec->walk, "o") ? slow : fast;
        V* o0 = vmul(tile_o, TO);
        V* t0 = vmul(tile_t, TT);
        V* w0 = vmul(tile_w, int64_t(TW) * VW);
        rocke_b_sync(b);

        /* ---------------- phase 1 */
        for(const auto& n : tins)
        {
            const rocke_type_t* ty = nullptr;
            dtype_of(dts[n], &ty, nullptr);
            scf_if(b, nz[n], [&]() {
                struct Ld
                {
                    V* ol;
                    V* tl;
                    V* c;
                    std::vector<V*> rws;
                };
                std::vector<Ld> loaded;
                for(V* u : units(p1_units(spec)))
                {
                    if(ma)
                    {
                        V* c = pow2_mod(u, TW);
                        V* rest = pow2_div(u, TW);
                        V* tl = pow2_mod(rest, TT);
                        V* ol = pow2_div(rest, TT);
                        V* off;
                        if(spec->rag)
                        {
                            V* cv = vmul(c, VW);
                            V* wc = rocke_b_add(b, w0, cv);
                            V* pw = rocke_b_smin(b, wc, last_w);
                            V* po0 = rocke_b_add(b, o0, ol);
                            V* po = rocke_b_smin(b, po0, last_o1);
                            V* pt0 = rocke_b_add(b, t0, tl);
                            V* pt = rocke_b_smin(b, pt0, last_t1);
                            V* dw = rocke_b_sub(b, pw, w0);
                            c = pow2_div(dw, VW);
                            ol = rocke_b_sub(b, po, o0);
                            tl = rocke_b_sub(b, pt, t0);
                            V* a1 = rocke_b_add(b, boff[n], pw);
                            V* oo = ooff(n, po);
                            V* to_ = toff(n, pt);
                            V* a2 = rocke_b_add(b, oo, to_);
                            off = rocke_b_add(b, a1, a2);
                        }
                        else
                        {
                            V* cv = vmul(c, VW);
                            V* wc = rocke_b_add(b, w0, cv);
                            V* a1 = rocke_b_add(b, boff[n], wc);
                            V* po = rocke_b_add(b, o0, ol);
                            V* oo = ooff(n, po);
                            V* pt = rocke_b_add(b, t0, tl);
                            V* to_ = toff(n, pt);
                            V* a2 = rocke_b_add(b, oo, to_);
                            off = rocke_b_add(b, a1, a2);
                        }
                        V* v = gload(n, off, VW);
                        loaded.push_back({ol, tl, c, {v}});
                    }
                    else
                    {
                        V* ub_t = pow2_mod(u, TT / VT);
                        V* ub_o = pow2_div(u, TT / VT);
                        V* ol = vmul(ub_o, VW);
                        V* tl = vmul(ub_t, VT);
                        V* base;
                        if(spec->rag)
                        {
                            V* po0 = rocke_b_add(b, o0, ol);
                            V* po = rocke_b_smin(b, po0, last_ov);
                            V* pt0 = rocke_b_add(b, t0, tl);
                            V* pt = rocke_b_smin(b, pt0, last_tv);
                            V* dob = rocke_b_sub(b, po, o0);
                            ub_o = pow2_div(dob, VW);
                            V* dtb = rocke_b_sub(b, pt, t0);
                            ub_t = pow2_div(dtb, VT);
                            V* oo = ooff(n, po);
                            V* to_ = toff(n, pt);
                            V* a2 = rocke_b_add(b, oo, to_);
                            base = rocke_b_add(b, boff[n], a2);
                        }
                        else
                        {
                            V* po = rocke_b_add(b, o0, ol);
                            V* oo = ooff(n, po);
                            V* pt = rocke_b_add(b, t0, tl);
                            V* to_ = toff(n, pt);
                            V* a2 = rocke_b_add(b, oo, to_);
                            base = rocke_b_add(b, boff[n], a2);
                        }
                        std::vector<V*> rws;
                        for(int j = 0; j < VW; ++j)
                        {
                            V* o_ = base;
                            if(j != 0)
                            {
                                V* kj = e.c32(j);
                                V* m = rocke_b_mul(b, kj, SO[n].first);
                                o_ = rocke_b_add(b, base, m);
                            }
                            rws.push_back(gload(n, o_, VT));
                        }
                        loaded.push_back({ub_o, ub_t, nullptr, rws});
                    }
                }
                for(auto& l : loaded)
                {
                    if(ma)
                    {
                        V* m1 = vmul(l.tl, TO);
                        V* a1 = rocke_b_add(b, m1, l.ol);
                        V* m2 = vmul(a1, TW);
                        V* ch = rocke_b_add(b, m2, l.c);
                        lstore(n, ch, l.rws[0]);
                        continue;
                    }
                    V* ub_o = l.ol;
                    V* ub_t = l.tl;
                    for(int i = 0; i < VT; ++i)
                    {
                        std::vector<V*> cols;
                        for(int j = 0; j < VW; ++j)
                            cols.push_back(rocke_b_vec_extract(b, l.rws[(size_t)j], i));
                        V* vv = rocke_b_vec_pack(b, cols.data(), VW, ty);
                        V* trow = vmul(ub_t, VT);
                        if(i)
                        {
                            V* ki = e.c32(i);
                            trow = rocke_b_add(b, trow, ki);
                        }
                        V* m = vmul(trow, TO / VW);
                        V* ch = rocke_b_add(b, m, ub_o);
                        lstore(n, ch, vv);
                    }
                }
            });
        }
        rocke_b_sync(b);

        /* ---------------- phase 2 */
        std::vector<Unit> p2;
        for(V* u : units(p2_units(spec)))
        {
            Unit un{};
            V* ol;
            V* tl;
            if(ma)
            {
                V* c = pow2_mod(u, TW);
                V* rest = pow2_div(u, TW);
                ol = pow2_mod(rest, TO);
                tl = pow2_div(rest, TO);
                if(spec->rag)
                {
                    V* cv = vmul(c, VW);
                    V* wc = rocke_b_add(b, w0, cv);
                    un.wv = rocke_b_smin(b, wc, last_w);
                    V* go0 = rocke_b_add(b, o0, ol);
                    un.go = rocke_b_smin(b, go0, last_o1);
                    V* gt0 = rocke_b_add(b, t0, tl);
                    un.gt = rocke_b_smin(b, gt0, last_t1);
                    V* dw = rocke_b_sub(b, un.wv, w0);
                    c = pow2_div(dw, VW);
                    ol = rocke_b_sub(b, un.go, o0);
                    tl = rocke_b_sub(b, un.gt, t0);
                }
                V* m1 = vmul(tl, TO);
                V* a1 = rocke_b_add(b, m1, ol);
                V* m2 = vmul(a1, TW);
                un.ch = rocke_b_add(b, m2, c);
                if(!spec->rag)
                {
                    V* cv = vmul(c, VW);
                    un.wv = rocke_b_add(b, w0, cv);
                }
            }
            else
            {
                V* uo = pow2_mod(u, TO / VW);
                tl = pow2_div(u, TO / VW);
                ol = vmul(uo, VW);
                if(spec->rag)
                {
                    V* go0 = rocke_b_add(b, o0, ol);
                    un.go = rocke_b_smin(b, go0, last_ov);
                    V* gt0 = rocke_b_add(b, t0, tl);
                    un.gt = rocke_b_smin(b, gt0, last_t1);
                    V* dob = rocke_b_sub(b, un.go, o0);
                    uo = pow2_div(dob, VW);
                    tl = rocke_b_sub(b, un.gt, t0);
                }
                V* m = vmul(tl, TO / VW);
                un.ch = rocke_b_add(b, m, uo);
                un.wv = nullptr;
            }
            if(!spec->rag)
            {
                un.go = rocke_b_add(b, o0, ol);
                un.gt = rocke_b_add(b, t0, tl);
            }
            p2.push_back(un);
        }
        auto offs = [&](const std::string& n, const Unit& un) {
            V* oo = ooff(n, un.go);
            V* to_ = toff(n, un.gt);
            V* a = rocke_b_add(b, oo, to_);
            V* o_ = rocke_b_add(b, boff[n], a);
            if(un.wv == nullptr)
                return o_;
            return rocke_b_add(b, o_, un.wv);
        };
        auto raw = [&](const std::string& n, const Unit& un) {
            const rocke_type_t* ty = nullptr;
            dtype_of(dts[n], &ty, nullptr);
            if(kind(n) == 'T')
            {
                V* idx[1] = {lds_index(un.ch)};
                return rocke_b_smem_load_vN(b, lds[n], idx, 1, ty, VW);
            }
            V* o_ = offs(n, un);
            return gload(n, o_, VW);
        };
        /* body(every, op): op >= 0: the fast path (no operator table; op the binary
         * operator, 0 for a permutation) */
        auto body = [&](bool every, int op) {
            size_t g = spec->ilp ? (size_t)spec->ilp : p2.size();
            const rocke_type_t* dty = nullptr;
            dtype_of(spec->d, &dty, nullptr);
            for(size_t g0 = 0; g0 < p2.size(); g0 += g)
            {
                std::vector<Unit> grp(p2.begin() + g0,
                                      p2.begin() + (g0 + g < p2.size() ? g0 + g : p2.size()));
                std::map<std::string, std::vector<V*>> vals;
                std::map<std::string, std::vector<V*>> loads;
                if(every)
                    for(const auto& n : ins)
                        for(const auto& un : grp)
                            loads[n].push_back(raw(n, un));
                for(const auto& n : ins)
                {
                    if(every)
                    {
                        std::vector<V*> xs;
                        for(V* v : loads[n])
                            for(int k = 0; k < VW; ++k)
                            {
                                V* x = rocke_b_vec_extract(b, v, k);
                                xs.push_back(e.to_c(x, dts[n]));
                            }
                        std::vector<V*> o;
                        if(op >= 0)
                        {
                            for(V* x : xs)
                                o.push_back(rocke_b_fmul(b, S[n], x));
                        }
                        else
                        {
                            std::vector<V*> us = e.unary_list(xs, OPS[n]);
                            for(V* x : us)
                                o.push_back(rocke_b_fmul(b, S[n], x));
                        }
                        vals[n] = o;
                    }
                    else
                    {
                        std::vector<V*> zs(VW * grp.size(), zero);
                        std::vector<V*> xs = e.when(nz[n], zs, [&](const std::vector<V*>&) {
                            std::vector<V*> vs;
                            for(const auto& un : grp)
                                vs.push_back(raw(n, un));
                            std::vector<V*> r;
                            for(V* v : vs)
                                for(int k = 0; k < VW; ++k)
                                {
                                    V* x = rocke_b_vec_extract(b, v, k);
                                    r.push_back(e.to_c(x, dts[n]));
                                }
                            return r;
                        });
                        std::vector<V*> us = e.unary_list(xs, OPS[n]);
                        std::vector<V*> o;
                        for(V* x : us)
                        {
                            V* p = rocke_b_fmul(b, S[n], x);
                            o.push_back(rocke_b_select(b, nz[n], p, zero));
                        }
                        vals[n] = o;
                    }
                }
                for(size_t q = 0; q < grp.size(); ++q)
                {
                    std::vector<V*> out;
                    for(size_t k = q * VW; k < q * VW + VW; ++k)
                    {
                        if(spec->arity == 1)
                            out.push_back(vals["A"][k]);
                        else if(op >= 0)
                        {
                            V* x = vals["A"][k];
                            if(spec->arity == 3)
                                x = binary_fixed(b, x, vals["B"][k], op);
                            out.push_back(binary_fixed(b, x, vals["C"][k], op));
                        }
                        else if(spec->arity == 2)
                            out.push_back(e.binary(vals["A"][k], vals["C"][k], OPS["AB"]));
                        else
                        {
                            V* ab = e.binary(vals["A"][k], vals["B"][k], OPS["AB"]);
                            out.push_back(e.binary(ab, vals["C"][k], OPS["ABC"]));
                        }
                    }
                    std::vector<V*> outs;
                    for(V* x : out)
                        outs.push_back(e.from_c(x, spec->d));
                    V* doff = offs("D", grp[q]);
                    V* pk = rocke_b_vec_pack(b, outs.data(), (int)outs.size(), dty);
                    rocke_b_global_store_vN_ex(b, P["D"], doff, pk, VW, spec->vb, &mo_st);
                }
            }
        };
        /* the fast path per binary operator (trinary: the same operator twice); then every
         * scale non-zero (straight-line loads, operator table); otherwise skip the inputs of
         * zero scale */
        std::vector<std::pair<V*, int>> conds;
        if(spec->arity == 1)
            conds.push_back({fastp, 0});
        else
            for(int bop : {OP_ADD, OP_MUL, OP_MAX, OP_MIN})
            {
                V* kb = e.c32(bop);
                V* eq = rocke_b_cmp_eq(b, OPS["AB"], kb);
                V* cnd = rocke_b_land(b, fastp, eq);
                if(spec->arity == 3)
                {
                    V* kb3 = e.c32(bop);
                    V* eq3 = rocke_b_cmp_eq(b, OPS["ABC"], kb3);
                    cnd = rocke_b_land(b, cnd, eq3);
                }
                conds.push_back({cnd, bop});
            }
        V* covered = nullptr;
        for(auto& c : conds)
        {
            int bop = c.second;
            scf_if(b, c.first, [&]() { body(true, bop); });
            covered = covered == nullptr ? c.first : rocke_b_lor(b, covered, c.first);
        }
        V* notcov = rocke_b_lnot(b, covered);
        V* gen = rocke_b_land(b, allnz, notcov);
        scf_if(b, gen, [&]() { body(true, -1); });
        V* notall = rocke_b_lnot(b, allnz);
        scf_if(b, notall, [&]() { body(false, -1); });
    };

    V* by = rocke_b_block_id_y(b);
    rocke_for_t rows = rocke_b_scf_for(b, by, nbv, ystep, "bt");
    rocke_b_region_enter(b, rows.body);
    {
        std::map<std::string, V*> boff;
        V* r = rows.iv;
        for(int i = 0; i < SLOTS_B; ++i)
        {
            V* dg;
            if(i < SLOTS_B - 1)
            {
                std::pair<V*, V*> qr
                    = e.divmod(r, bext[(size_t)i], bmg[(size_t)i].first, bmg[(size_t)i].second);
                r = qr.first;
                dg = qr.second;
            }
            else
                dg = r;
            for(const auto& n : names)
            {
                V* t_ = rocke_b_mul(b, dg, bst[n][(size_t)i]);
                boff[n] = boff.count(n) == 0 ? t_ : rocke_b_add(b, boff[n], t_);
            }
        }
        for(int k = 0; k < spec->tpc; ++k)
        {
            V* tile_id = first;
            if(k != 0)
            {
                V* kk = e.c32(k);
                tile_id = rocke_b_add(b, first, kk);
            }
            if(spec->tpc > 1)
            {
                V* lt = rocke_b_cmp_lt(b, tile_id, ntl);
                scf_if(b, lt, [&]() { one_tile(tile_id, boff); });
            }
            else
                one_tile(tile_id, boff);
        }
    }
    rocke_b_region_leave(b);
    if(!rocke_ir_builder_ok(b))
        return nullptr;
    return rocke_ir_builder_kernel(b);
}

rocke_kernel_def_t* rocke_build_htaot_ewp(rocke_ir_builder_t* b, const rocke_htaot_ewp_spec_t* s)
{
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* { return ewp_impl(b, s); });
}

/* the route table (hiptensor_aot_ewp_table.inc, written by gen_ewp_table.py from
 * ewp_host.pack_classes / routes) */
int rocke_htaot_ewp_pack_classes(const char* arch, rocke_htaot_ewp_spec_t* out, int cap)
{
    if(arch == nullptr)
        return -1;
    int n = 0;
#define ROCKE_EWP_CLASS(ARCH,                                           \
                        ARITY,                                          \
                        A,                                              \
                        B,                                              \
                        C,                                              \
                        D,                                              \
                        KINDS,                                          \
                        MODE,                                           \
                        TO,                                             \
                        TT,                                             \
                        TW,                                             \
                        BLOCK,                                          \
                        NT,                                             \
                        SWZ,                                            \
                        SWSH,                                           \
                        WALK,                                           \
                        TPC,                                            \
                        ILP,                                            \
                        VB,                                             \
                        RAG,                                            \
                        TB)                                             \
    if(strcmp(arch, ARCH) == 0)                                         \
    {                                                                   \
        if(out != nullptr && n < cap)                                   \
        {                                                               \
            rocke_htaot_ewp_spec_t s_ = rocke_htaot_ewp_spec_default(); \
            s_.arity = ARITY;                                           \
            s_.a = A;                                                   \
            s_.b = B;                                                   \
            s_.c = C;                                                   \
            s_.d = D;                                                   \
            s_.kinds = KINDS;                                           \
            s_.mode = MODE;                                             \
            s_.to = TO;                                                 \
            s_.tt = TT;                                                 \
            s_.tw = TW;                                                 \
            s_.block = BLOCK;                                           \
            s_.nt = NT;                                                 \
            s_.swz = SWZ;                                               \
            s_.swsh = SWSH;                                             \
            s_.walk = WALK;                                             \
            s_.tpc = TPC;                                               \
            s_.ilp = ILP;                                               \
            s_.vb = VB;                                                 \
            s_.rag = RAG;                                               \
            s_.tb = TB;                                                 \
            out[n] = s_;                                                \
        }                                                               \
        ++n;                                                            \
    }
#include "hiptensor_aot_ewp_table.inc"
#undef ROCKE_EWP_CLASS
    return n;
}

int rocke_htaot_ewp_route_count(const char* arch)
{
    if(arch == nullptr)
        return -1;
    int n = 0;
#define ROCKE_EWP_ROUTE(ARCH, NAME, VALUE) \
    if(strcmp(arch, ARCH) == 0)            \
        ++n;
#include "hiptensor_aot_ewp_table.inc"
#undef ROCKE_EWP_ROUTE
    return n;
}

bool rocke_htaot_ewp_route(const char* arch, int i, const char** name, const char** value)
{
    if(arch == nullptr || name == nullptr || value == nullptr || i < 0)
        return false;
    int n = 0;
#define ROCKE_EWP_ROUTE(ARCH, NAME, VALUE) \
    if(strcmp(arch, ARCH) == 0)            \
    {                                      \
        if(n == i)                         \
        {                                  \
            *name = NAME;                  \
            *value = VALUE;                \
            return true;                   \
        }                                  \
        ++n;                               \
    }
#include "hiptensor_aot_ewp_table.inc"
#undef ROCKE_EWP_ROUTE
    (void)n;
    return false;
}

} /* extern "C" */
