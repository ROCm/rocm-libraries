// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * gemm_prologue.cpp -- C++ mirror of instances/common/gemm_prologue.py: the
 * Prologue value type (defaults, check, kmul / parts / products, suffix) and
 * the PrologueEmitter the universal GEMM builder calls at each fragment load
 * and MFMA site (see rocke/instance_gemm_prologue.h).
 *
 * Every builder call is its own statement, in the order the Python emitter
 * makes it: each call consumes an SSA id.
 */
#include "rocke/instance_gemm_prologue.h"

#include "rocke/instance_gemm_internal.h"
#include "rocke/ir_internal.h"

#include <stdio.h>
#include <string.h>

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace
{

bool eq(const char* a, const char* b)
{
    return a != nullptr && b != nullptr && strcmp(a, b) == 0;
}

bool empty(const char* s)
{
    return s == nullptr || s[0] == '\0';
}

/* Python RECIPES: (parts, products listed big first). */
struct recipe_t
{
    const char* name;
    int parts;
    int n_products;
    int products[9][2];
};

const recipe_t k_recipes[] = {
    {"cvt", 1, 1, {{0, 0}}},
    {"x3", 2, 3, {{0, 0}, {0, 1}, {1, 0}}},
    {"x4", 2, 4, {{0, 0}, {0, 1}, {1, 0}, {1, 1}}},
    {"x6", 3, 6, {{0, 0}, {0, 1}, {1, 0}, {0, 2}, {1, 1}, {2, 0}}},
    {"x9", 3, 9, {{0, 0}, {0, 1}, {1, 0}, {0, 2}, {1, 1}, {2, 0}, {1, 2}, {2, 1}, {2, 2}}},
};

const recipe_t* find_recipe(const char* name)
{
    for(const recipe_t& r : k_recipes)
        if(eq(r.name, name))
            return &r;
    return nullptr;
}

/* Python _BF16_SAT: the largest f32 whose RNE bf16 rounding is finite. */
constexpr double k_bf16_sat = 3.3895313892515355e38;

} // namespace

extern "C" {

rocke_gemm_prologue_t rocke_gemm_prologue_default(void)
{
    rocke_gemm_prologue_t p;
    memset(&p, 0, sizeof p);
    p.src = "same";
    p.recipe = "cvt";
    p.unary_a = "";
    p.unary_b = "";
    p.unary_impl = "f32";
    p.order = "small_first";
    p.split_at = "load";
    p.stage = "lds";
    p.fast_exp = true;
    p.epi = "";
    return p;
}

bool rocke_gemm_prologue_check(const rocke_gemm_prologue_t* p, char* reason, size_t reason_cap)
{
    auto fail = [&](const char* fmt, const char* arg) {
        if(reason != nullptr && reason_cap > 0)
            snprintf(reason, reason_cap, fmt, arg != nullptr ? arg : "(null)");
        return false;
    };
    if(p == nullptr)
        return fail("%s", "null prologue");
    /* Python Prologue.check() */
    if(!eq(p->src, "same") && !eq(p->src, "f32"))
        return fail("prologue src '%s'", p->src);
    if(find_recipe(p->recipe) == nullptr)
        return fail("prologue recipe '%s'", p->recipe);
    if(!eq(p->src, "f32") && !eq(p->recipe, "cvt"))
        return fail("%s", "split recipes need src='f32'");
    if(!eq(p->unary_impl, "f32") && !eq(p->unary_impl, "bits"))
        return fail("prologue unary_impl '%s'", p->unary_impl);
    if(!eq(p->order, "small_first") && !eq(p->order, "big_first"))
        return fail("prologue order '%s'", p->order);
    if(!eq(p->split_at, "load") && !eq(p->split_at, "use"))
        return fail("prologue split_at '%s'", p->split_at);
    if(!eq(p->stage, "lds") && !eq(p->stage, "vgpr"))
        return fail("prologue stage '%s'", p->stage);
    if(eq(p->stage, "vgpr") && (!eq(p->src, "f32") || !eq(p->recipe, "cvt")))
        return fail("%s", "stage='vgpr' is built for src='f32', recipe='cvt'");
    if((p->conj_a || p->conj_b) && !p->cplx_b)
        return fail("%s", "conj_a / conj_b need cplx_b");
    if(p->cplx_b && (!eq(p->src, "f32") || !eq(p->stage, "lds") || !eq(p->split_at, "use")))
        return fail("%s", "cplx_b needs src='f32', stage='lds', split_at='use'");
    if(p->epi != nullptr && !empty(p->epi) && !eq(p->epi, "alpha") && !eq(p->epi, "bilinear"))
        return fail("prologue epi '%s'", p->epi);
    if(!empty(p->epi) && !p->out_f32)
        return fail("%s", "prologue epi (alpha / bilinear) needs out_f32");
    /* the mirror's scope: refuse what the C++ emitter does not build */
    if(!empty(p->unary_a) || !empty(p->unary_b))
        return fail("%s",
                    "prologue unary_a / unary_b: implemented by the Python engine only "
                    "(no C++ mirror yet)");
    if(eq(p->stage, "vgpr"))
        return fail("%s",
                    "prologue stage='vgpr': implemented by the Python engine only "
                    "(no C++ mirror yet)");
    if(!empty(p->epi))
        return fail("%s",
                    "prologue epi (alpha / bilinear): implemented by the Python engine only "
                    "(no C++ mirror yet)");
    if(reason != nullptr && reason_cap > 0)
        snprintf(reason, reason_cap, "ok");
    return true;
}

int rocke_gemm_prologue_kmul(const rocke_gemm_prologue_t* p)
{
    return (eq(p->src, "f32") && eq(p->stage, "lds")) ? 2 : 1;
}

int rocke_gemm_prologue_parts(const rocke_gemm_prologue_t* p)
{
    const recipe_t* r = find_recipe(p->recipe);
    return r != nullptr ? r->parts : 1;
}

int rocke_gemm_prologue_suffix(const rocke_gemm_prologue_t* p, char* out, size_t out_cap)
{
    std::string s = "_pro";
    bool ua = !empty(p->unary_a), ub = !empty(p->unary_b);
    if(eq(p->src, "f32"))
        s += std::string("_f32") + p->recipe;
    if(ua)
        s += std::string("_ua") + p->unary_a;
    if(ub)
        s += std::string("_ub") + p->unary_b;
    if((ua || ub) && !eq(p->unary_impl, "f32"))
        s += std::string("_ui") + p->unary_impl;
    if(p->out_f32)
        s += "_of32";
    if(p->cap)
        s += "_cap";
    if(!eq(p->order, "small_first"))
        s += "_bf";
    if(!eq(p->split_at, "load"))
        s += "_su";
    if(p->psub)
        s += "_pk";
    if(!eq(p->stage, "lds"))
        s += "_vg";
    if(!p->fast_exp && (eq(p->unary_a, "exp") || eq(p->unary_b, "exp")))
        s += "_gexp";
    if(p->pin_v)
        s += "_pv";
    if(p->cplx_b)
        s += std::string("_cx") + (p->conj_a ? "a" : "") + (p->conj_b ? "b" : "");
    if(!empty(p->epi))
        s += std::string("_e") + (eq(p->epi, "alpha") ? "a" : "ab");
    if(p->agpr_out)
        s += "_ao";
    if(s.size() + 1 > out_cap)
        return -1;
    memcpy(out, s.c_str(), s.size() + 1);
    return 0;
}

} /* extern "C" */

/* ------------------------------------------------------------------------ *
 * PrologueEmitter
 * ------------------------------------------------------------------------ */

struct rocke_gemm_pro_emitter
{
    rocke_ir_builder_t* b;
    const rocke_gemm_prologue_t* pro;
    const rocke_type_t* compute;
    int n; /* frag_len */
    /* Python _memo (id(frag) -> parts; a key with no parts is "tagged, not
     * converted yet") and _which (id(raw) -> operand) */
    std::unordered_map<rocke_value_t*, std::vector<rocke_value_t*>> memo;
    std::unordered_map<rocke_value_t*, char> which;
    /* cplx_b: B' row parity from ni (row-interleaved bodies) or a per-lane i1 */
    bool cplx_ni = false;
    rocke_value_t* cplx_lane_odd = nullptr;

    /* Python _to_f32_vec */
    rocke_value_t* to_f32_vec(rocke_value_t* v)
    {
        int cnt = v->type->count;
        if(!rocke_type_eq(v->type->elem, rocke_bf16()) || cnt % 2)
        {
            std::vector<rocke_value_t*> xs;
            for(int i = 0; i < cnt; ++i)
            {
                rocke_value_t* e = rocke_b_vec_extract(b, v, i);
                xs.push_back(rocke_b_cast_to_f32(b, e));
            }
            return rocke_b_vec_pack(b, xs.data(), (int)xs.size(), rocke_f32());
        }
        rocke_value_t* w = rocke_b_vec_bitcast(b, v, rocke_vector_type(b, rocke_i32(), cnt / 2));
        rocke_value_t* c16 = rocke_b_const_i32(b, 16);
        rocke_value_t* chi = rocke_b_const_i32(b, -65536); /* 0xFFFF0000 */
        std::vector<rocke_value_t*> outs;
        for(int i = 0; i < cnt / 2; ++i)
        {
            rocke_value_t* d = rocke_b_vec_extract(b, w, i);
            rocke_value_t* sh = rocke_b_shl(b, d, c16);
            outs.push_back(rocke_b_bitcast(b, sh, rocke_f32()));
            rocke_value_t* lo = rocke_b_land(b, d, chi);
            outs.push_back(rocke_b_bitcast(b, lo, rocke_f32()));
        }
        return rocke_b_vec_pack(b, outs.data(), (int)outs.size(), rocke_f32());
    }

    /* Python _slice */
    rocke_value_t* slice(rocke_value_t* v, int lo, int cnt)
    {
        std::vector<rocke_value_t*> xs;
        for(int i = 0; i < cnt; ++i)
            xs.push_back(rocke_b_vec_extract(b, v, lo + i));
        return rocke_b_vec_pack(b, xs.data(), cnt, v->type->elem);
    }

    /* Python _sub */
    rocke_value_t* sub(rocke_value_t* x, rocke_value_t* y)
    {
        int cnt = x->type->count;
        if(!pro->psub || cnt % 2)
            return rocke_b_fsub(b, x, y);
        const rocke_type_t* v2 = rocke_vector_type(b, rocke_f32(), 2);
        std::vector<rocke_value_t*> outs;
        for(int i = 0; i < cnt; i += 2)
        {
            rocke_value_t* xp[2];
            xp[0] = rocke_b_vec_extract(b, x, i);
            xp[1] = rocke_b_vec_extract(b, x, i + 1);
            rocke_value_t* xa = rocke_b_vec_pack(b, xp, 2, rocke_f32());
            rocke_value_t* yp[2];
            yp[0] = rocke_b_vec_extract(b, y, i);
            yp[1] = rocke_b_vec_extract(b, y, i + 1);
            rocke_value_t* ya = rocke_b_vec_pack(b, yp, 2, rocke_f32());
            rocke_value_t* ops[2] = {xa, ya};
            rocke_inline_asm_opts_t opts;
            memset(&opts, 0, sizeof opts);
            opts.sideeffect = false;
            opts.sideeffect_set = true;
            rocke_op_t* op = rocke_b_inline_asm(b,
                                                "v_pk_add_f32 $0, $1, $2 neg_lo:[0,1] neg_hi:[0,1]",
                                                "=v,v,v",
                                                ops,
                                                2,
                                                &v2,
                                                1,
                                                &opts);
            if(op == nullptr)
                return nullptr;
            rocke_value_t* d = op->results[0];
            outs.push_back(rocke_b_vec_extract(b, d, 0));
            outs.push_back(rocke_b_vec_extract(b, d, 1));
        }
        return rocke_b_vec_pack(b, outs.data(), (int)outs.size(), rocke_f32());
    }

    /* Python _split */
    std::vector<rocke_value_t*> split(rocke_value_t* x)
    {
        int P = rocke_gemm_prologue_parts(pro);
        rocke_value_t* hi_src = x;
        if(pro->cap && rocke_type_eq(compute, rocke_bf16()))
        {
            int cnt = x->type->count;
            rocke_value_t* sat = rocke_b_const_f32(b, k_bf16_sat);
            rocke_value_t* nsat = rocke_b_const_f32(b, -k_bf16_sat);
            std::vector<rocke_value_t*> xs;
            for(int i = 0; i < cnt; ++i)
            {
                rocke_value_t* e = rocke_b_vec_extract(b, x, i);
                rocke_value_t* mn = rocke_b_fmin(b, e, sat);
                xs.push_back(rocke_b_fmax(b, mn, nsat));
            }
            hi_src = rocke_b_vec_pack(b, xs.data(), cnt, rocke_f32());
        }
        std::vector<rocke_value_t*> parts;
        parts.push_back(rocke_b_vec_cast_f32_to(b, hi_src, compute));
        rocke_value_t* r = x;
        for(int p = 1; p < P; ++p)
        {
            rocke_value_t* w = to_f32_vec(parts.back());
            r = sub(r, w);
            parts.push_back(rocke_b_vec_cast_f32_to(b, r, compute));
        }
        return parts;
    }

    /* Python _pin */
    rocke_value_t* pin(rocke_value_t* v)
    {
        if(!pro->pin_v)
            return v;
        const rocke_type_t* ty = v->type;
        rocke_inline_asm_opts_t opts;
        memset(&opts, 0, sizeof opts);
        opts.sideeffect = false;
        opts.sideeffect_set = true;
        rocke_op_t* op = rocke_b_inline_asm(b, "", "=v,0", &v, 1, &ty, 1, &opts);
        return op != nullptr ? op->results[0] : nullptr;
    }

    /* Python _cplx_fix; par_int < 0 means the per-lane i1 `par_v` */
    rocke_value_t* cplx_fix(rocke_value_t* x, int par_int, rocke_value_t* par_v)
    {
        int sa = pro->conj_a ? -1 : 1;
        int sb = pro->conj_b ? -1 : 1;
        auto sg = [&](rocke_value_t* v, int s) { return s < 0 ? rocke_b_fneg(b, v) : v; };
        std::vector<rocke_value_t*> outs;
        for(int i = 0; i < n; i += 2)
        {
            rocke_value_t* x0 = rocke_b_vec_extract(b, x, i);
            rocke_value_t* x1 = rocke_b_vec_extract(b, x, i + 1);
            if(par_int >= 0)
            {
                if(par_int)
                {
                    rocke_value_t* o0 = sg(x1, sb);
                    rocke_value_t* o1 = sg(x0, sa);
                    outs.push_back(o0);
                    outs.push_back(o1);
                }
                else
                {
                    outs.push_back(x0);
                    outs.push_back(sg(x1, -sa * sb));
                }
                continue;
            }
            rocke_value_t* t0 = sg(x1, sb);
            rocke_value_t* s0 = rocke_b_select(b, par_v, t0, x0);
            rocke_value_t* t1 = sg(x0, sa);
            rocke_value_t* t2 = sg(x1, -sa * sb);
            rocke_value_t* s1 = rocke_b_select(b, par_v, t1, t2);
            outs.push_back(s0);
            outs.push_back(s1);
        }
        return rocke_b_vec_pack(b, outs.data(), (int)outs.size(), rocke_f32());
    }

    /* Python _convert_raw0 (no unary op: refused by check) */
    std::vector<rocke_value_t*> convert_raw0(rocke_value_t* raw, int par_int, rocke_value_t* par_v)
    {
        if(eq(pro->src, "f32"))
        {
            rocke_value_t* x = rocke_b_vec_bitcast(b, raw, rocke_vector_type(b, rocke_f32(), n));
            if(par_int >= 0 || par_v != nullptr)
                x = cplx_fix(x, par_int, par_v);
            return split(x);
        }
        return {raw};
    }

    /* Python _convert_raw */
    std::vector<rocke_value_t*> convert_raw(rocke_value_t* raw, int par_int, rocke_value_t* par_v)
    {
        std::vector<rocke_value_t*> parts = convert_raw0(raw, par_int, par_v);
        for(auto& p : parts)
            p = pin(p);
        return parts;
    }

    /* Python after_load */
    rocke_value_t* after_load(rocke_value_t* raw, char which_op)
    {
        if(eq(pro->split_at, "use"))
        {
            if(!eq(pro->src, "f32"))
                return raw;
            memo[raw]; /* tagged: converted at its first MFMA use */
            which[raw] = which_op;
            return raw;
        }
        std::vector<rocke_value_t*> parts = convert_raw(raw, -1, nullptr);
        rocke_value_t* v = parts[0];
        for(size_t i = 1; i < parts.size(); ++i)
            v = rocke_b_vec_concat(b, v, parts[i]);
        return v;
    }

    /* Python _parts_of; ni < 0 means None */
    std::vector<rocke_value_t*> parts_of(rocke_value_t* frag, char which_op, int ni)
    {
        auto hit = memo.find(frag);
        if(hit != memo.end() && !hit->second.empty())
            return hit->second;
        int P = rocke_gemm_prologue_parts(pro);
        std::vector<rocke_value_t*> parts;
        if(eq(pro->split_at, "use") && eq(pro->src, "f32"))
        {
            int par_int = -1;
            rocke_value_t* par_v = nullptr;
            if(pro->cplx_b && which_op == 'B')
            {
                if(cplx_ni)
                {
                    if(ni < 0)
                    {
                        rocke_i_set_err(b,
                                        ROCKE_ERR_VALUE,
                                        "cplx_b (row-interleaved body): MFMA site without ni");
                        return {frag};
                    }
                    par_int = ni & 1;
                }
                else
                {
                    if(cplx_lane_odd == nullptr)
                    {
                        rocke_i_set_err(b, ROCKE_ERR_VALUE, "cplx_b: lane parity not set");
                        return {frag};
                    }
                    par_v = cplx_lane_odd;
                }
            }
            parts = convert_raw(frag, par_int, par_v);
        }
        else if(P == 1)
        {
            parts.push_back(frag);
        }
        else
        {
            for(int p = 0; p < P; ++p)
                parts.push_back(slice(frag, p * n, n));
        }
        memo[frag] = parts;
        return parts;
    }
};

extern "C" {

rocke_gemm_pro_emitter_t* rocke_gemm_pro_emitter_new(rocke_ir_builder_t* b,
                                                     const rocke_gemm_prologue_t* pro,
                                                     const rocke_type_t* compute,
                                                     int frag_len)
{
    rocke_gemm_pro_emitter_t* pe = new rocke_gemm_pro_emitter_t();
    pe->b = b;
    pe->pro = pro;
    pe->compute = compute;
    pe->n = frag_len;
    return pe;
}

void rocke_gemm_pro_emitter_free(rocke_gemm_pro_emitter_t* pe)
{
    delete pe;
}

int rocke_gemm_pro_raw_len(const rocke_gemm_pro_emitter_t* pe)
{
    return pe->n * rocke_gemm_prologue_kmul(pe->pro);
}

void rocke_gemm_pro_set_cplx_ni(rocke_gemm_pro_emitter_t* pe)
{
    pe->cplx_ni = true;
}

void rocke_gemm_pro_set_cplx_lane(rocke_gemm_pro_emitter_t* pe, rocke_value_t* lane_odd)
{
    pe->cplx_ni = false;
    pe->cplx_lane_odd = lane_odd;
}

rocke_value_t*
    rocke_gemm_pro_after_load(rocke_gemm_pro_emitter_t* pe, rocke_value_t* raw, char which)
{
    return pe->after_load(raw, which);
}

rocke_value_t* rocke_gemm_pro_mma(rocke_gemm_pro_emitter_t* pe,
                                  const rocke_gemm_universal_spec_t* spec,
                                  const rocke_mmaop_t* op,
                                  rocke_value_t* a_frag,
                                  rocke_value_t* b_frag,
                                  rocke_value_t* acc,
                                  bool swap,
                                  int ni)
{
    rocke_ir_builder_t* b = pe->b;
    std::vector<rocke_value_t*> pa = pe->parts_of(a_frag, 'A', -1);
    std::vector<rocke_value_t*> pb = pe->parts_of(b_frag, 'B', ni);
    const recipe_t* r = find_recipe(pe->pro->recipe);
    if(r == nullptr || !rocke_ir_builder_ok(b))
        return acc;
    /* Python Prologue.products(): small_first = stable sort on -(i + j) */
    std::vector<std::pair<int, int>> prods;
    for(int k = 0; k < r->n_products; ++k)
        prods.emplace_back(r->products[k][0], r->products[k][1]);
    if(eq(pe->pro->order, "small_first"))
    {
        std::vector<std::pair<int, int>> sorted;
        for(int level = 4; level >= 0; --level)
            for(auto const& ij : prods)
                if(ij.first + ij.second == level)
                    sorted.push_back(ij);
        prods = sorted;
    }
    for(auto const& ij : prods)
    {
        if(swap)
            acc = rocke_gemm_emit_mma(b, op, pb[ij.second], pa[ij.first], acc);
        else
            acc = rocke_gemm_emit_mma(b, op, pa[ij.first], pb[ij.second], acc);
        acc = rocke_gemm_pin_acc_agpr(b, spec, acc);
    }
    return acc;
}

} /* extern "C" */

/* ------------------------------------------------------------------------ *
 * Builder hooks shared by the GEMM bodies (Python build_universal_gemm
 * closures _frag_ld, _brow, _mma_ab and the module helper _acc_f32_out).
 * ------------------------------------------------------------------------ */

extern "C" {

rocke_value_t* rocke_gemm_frag_ld(rocke_gemm_build_ctx_t* ctx,
                                  rocke_value_t* src,
                                  rocke_value_t* row,
                                  rocke_value_t* col_base,
                                  rocke_value_t* key,
                                  int n,
                                  char which,
                                  bool swz)
{
    rocke_ir_builder_t* b = ctx->b;
    if(ctx->pe == nullptr)
    {
        rocke_value_t* col = swz ? rocke_gemm_swz_col(ctx, col_base, key) : col_base;
        return rocke_gemm_emit_smem_load(b, src, row, col, n, ctx->storage_dtype);
    }
    int n_raw = rocke_gemm_pro_raw_len(ctx->pe);
    rocke_value_t* raw = nullptr;
    for(int off = 0; off < n_raw; off += 8)
    {
        rocke_value_t* col = col_base;
        if(off != 0)
        {
            rocke_value_t* c = rocke_b_const_i32(b, off);
            col = rocke_b_add(b, col_base, c);
        }
        if(swz)
            col = rocke_gemm_swz_col(ctx, col, key);
        rocke_value_t* ch = rocke_gemm_emit_smem_load(
            b, src, row, col, n_raw < 8 ? n_raw : 8, ctx->storage_dtype);
        raw = raw == nullptr ? ch : rocke_b_vec_concat(b, raw, ch);
    }
    return rocke_gemm_pro_after_load(ctx->pe, raw, which);
}

rocke_value_t* rocke_gemm_brow(rocke_gemm_build_ctx_t* ctx, rocke_value_t* r)
{
    if(!ctx->cplx)
        return r;
    rocke_value_t* c2 = rocke_b_const_i32(ctx->b, 2);
    return rocke_b_div(ctx->b, r, c2);
}

rocke_value_t* rocke_gemm_mma_ab(rocke_gemm_build_ctx_t* ctx,
                                 rocke_value_t* a_frag,
                                 rocke_value_t* b_frag,
                                 rocke_value_t* acc,
                                 int ni)
{
    rocke_ir_builder_t* b = ctx->b;
    if(ctx->pe != nullptr)
        return rocke_gemm_pro_mma(
            ctx->pe, ctx->spec, ctx->op, a_frag, b_frag, acc, ctx->swap_ab, ni);
    rocke_value_t* d = ctx->swap_ab ? rocke_gemm_emit_mma(b, ctx->op, b_frag, a_frag, acc)
                                    : rocke_gemm_emit_mma(b, ctx->op, a_frag, b_frag, acc);
    return rocke_gemm_pin_acc_agpr(b, ctx->spec, d);
}

rocke_value_t* rocke_gemm_acc_f32_out(rocke_ir_builder_t* b,
                                      const rocke_gemm_universal_spec_t* spec,
                                      rocke_value_t* acc)
{
    const rocke_gemm_prologue_t* p = spec->_prologue;
    if(spec->trait.acc_agpr && p != nullptr && !p->agpr_out && acc != nullptr)
    {
        const rocke_type_t* ty = acc->type;
        rocke_inline_asm_opts_t opts;
        memset(&opts, 0, sizeof opts);
        opts.sideeffect = true;
        opts.sideeffect_set = true;
        rocke_op_t* op = rocke_b_inline_asm(b, "", "=v,0", &acc, 1, &ty, 1, &opts);
        return op != nullptr ? op->results[0] : nullptr;
    }
    return rocke_gemm_acc_agpr_out(b, spec, acc);
}

bool rocke_gemm_out_f32(const rocke_gemm_universal_spec_t* spec)
{
    return spec->_prologue != nullptr && spec->_prologue->out_f32;
}

} /* extern "C" */
