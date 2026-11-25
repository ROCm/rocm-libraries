// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "attention_dense_internal.hpp"

#include <cassert>
#include <cstdint>
#include <string_view>
#include <type_traits>
#include <variant>

namespace rocke_dense_gfx950
{
namespace
{
// A shape key names its optional kernarg directly. A null field does not imply
// a baked extent: Nqb and Work can also be derived from other runtime values.
using Extent = V DecodeRuntime::*;
constexpr Extent Batch = &DecodeRuntime::batch, SeqQ = &DecodeRuntime::seqlen_q,
                 SeqKV = &DecodeRuntime::seqlen_kv, Heads = &DecodeRuntime::num_query_heads,
                 KvHeads = &DecodeRuntime::num_kv_heads, Gqa = &DecodeRuntime::gqa,
                 Nqb = &DecodeRuntime::nqb, Work = &DecodeRuntime::w;
using Flag = std::variant<bool, V>;

struct DecodeShape
{
    Builder* b;
    const rocke_attention_dense_spec_t& s;
    const DecodeRuntime& rt;
    bool fastdiv;

    V c(int64_t n) const
    {
        return rocke_b_const_i32(b, n);
    }

    int64_t baked(Extent e) const
    {
        if(e == Batch)
            return s.batch;
        if(e == SeqQ)
            return s.seqlen_q;
        if(e == SeqKV)
            return s.seqlen_kv;
        if(e == Heads)
            return s.num_query_heads;
        if(e == KvHeads)
            return s.num_kv_heads;
        if(e == Gqa)
            return rocke_attention_dense_spec_num_queries_per_kv(&s);
        const int64_t nqb = (int64_t{s.seqlen_q} + s.block_m - 1) / s.block_m;
        if(e == Nqb)
            return nqb;
        assert(e == Work);
        return nqb * s.num_query_heads * s.batch;
    }

    bool runtime(Extent e) const
    {
        if(rt.*e != nullptr)
            return true;
        if(e == Nqb)
            return runtime(SeqQ);
        return e == Work && (runtime(Nqb) || runtime(Heads) || runtime(Batch));
    }

    V val(Extent e, int64_t scale = 1) const
    {
        if(!runtime(e))
            return c(baked(e) / scale);
        V v = rt.*e;
        if(v == nullptr)
        {
            if(e == Nqb)
                v = DENSE_OP(
                    div, DENSE_OP(add, rt.seqlen_q, c(int64_t{s.block_m} - 1)), c(s.block_m));
            else
                v = DENSE_OP(mul, DENSE_OP(mul, val(Nqb), val(Heads)), val(Batch));
        }
        return scale == 1 ? v : DENSE_OP(div, v, c(scale));
    }

    const FastDivPair& pair(Extent e) const
    {
        if(e == Batch)
            return rt.batch_div;
        if(e == Heads)
            return rt.heads_div;
        if(e == Gqa)
            return rt.gqa_div;
        assert(e == Nqb);
        return rt.nqb_div;
    }

    V div(V n, Extent e, int64_t scale = 1) const
    {
        if(!fastdiv || !runtime(e))
            return DENSE_OP(div, n, val(e, scale));
        if(scale != 1)
            n = DENSE_OP(mul, n, c(scale));
        const auto& [magic, shift] = pair(e);
        return DENSE_OP(lshr, DENSE_OP(add, DENSE_OP(umul_hi_i32, n, magic), n), shift);
    }

    V mod(V n, Extent e, int64_t scale = 1) const
    {
        if(!fastdiv || !runtime(e))
            return DENSE_OP(mod, n, val(e, scale));
        return DENSE_OP(sub, n, DENSE_OP(mul, div(n, e, scale), val(e, scale)));
    }

    Flag balance() const
    {
        if(!s.causal)
            return false;
        if(s.sliding_window <= 0)
            return true;
        if(!runtime(SeqKV))
            return s.sliding_window >= s.seqlen_kv;
        return DENSE_OP(cmp_ge, c(s.sliding_window), rt.seqlen_kv);
    }

    Flag pair_fold(int64_t np) const
    {
        Flag bal = balance();
        const bool* fixed = std::get_if<bool>(&bal);
        if(fixed && !*fixed)
            return false;
        // Python's static case tests `bal is True`, not the value of an SSA flag.
        if(!runtime(Nqb) && !runtime(Work))
            return fixed != nullptr && baked(Nqb) % 2 == 0 && baked(Work) > np;
        V even = DENSE_OP(cmp_eq, DENSE_OP(land, val(Nqb), c(1)), c(0));
        V flag = DENSE_OP(land, even, DENSE_OP(cmp_lt, c(np), val(Work)));
        return fixed ? flag : DENSE_OP(land, flag, std::get<V>(bal));
    }

    template <class Then, class Other>
    auto branch(Flag flag, Then then, Other other) const
    {
        if(auto fixed = std::get_if<bool>(&flag))
            return *fixed ? then() : other();
        V cond = std::get<V>(flag);
        auto t = then();
        auto o = other();
        if constexpr(std::is_same_v<decltype(t), V>)
            return DENSE_OP(select, cond, t, o);
        else
        {
            // Python selects tuple components in tuple order, even equal ones.
            for(size_t i = 0; i < t.size(); ++i)
                t[i] = DENSE_OP(select, cond, t[i], o[i]);
            return t;
        }
    }
};

V reverse(const DecodeShape& d, V blk)
{
    auto* b = d.b;
    return DENSE_OP(sub, DENSE_OP(sub, d.val(Nqb), d.c(1)), blk);
}

V fold(const DecodeShape& d, V blk)
{
    auto* b = d.b;
    V nqb = d.val(Nqb);
    V half = DENSE_OP(lshr, nqb, d.c(1));
    V hi = DENSE_OP(sub, DENSE_OP(add, DENSE_OP(sub, nqb, d.c(1)), half), blk);
    return DENSE_OP(select, DENSE_OP(cmp_lt, blk, half), blk, hi);
}

std::pair<V, V> pair_unit(const DecodeShape& d, V wi)
{
    auto* b = d.b;
    const int64_t np = d.s.num_persistent;
    V w = d.val(Work);
    V rounds_end = DENSE_OP(mul, DENSE_OP(div, w, d.c(2 * np)), d.c(2 * np));
    V re_half = DENSE_OP(lshr, rounds_end, d.c(1));
    V paired_tail = DENSE_OP(sub, DENSE_OP(sub, w, rounds_end), d.c(np));
    V pt = DENSE_OP(smax, paired_tail, d.c(0));
    V j = DENSE_OP(div, wi, d.c(np));
    V unit = DENSE_OP(add,
                      DENSE_OP(sub, wi, DENSE_OP(mul, j, d.c(np))),
                      DENSE_OP(mul, DENSE_OP(div, j, d.c(2)), d.c(np)));
    V half = DENSE_OP(mod, j, d.c(2));

    // In the last round only CTAs below paired_tail get both halves.
    V r = DENSE_OP(sub, wi, rounds_end);
    V s = DENSE_OP(sub, r, pt);
    V unit_t = DENSE_OP(add, DENSE_OP(add, re_half, pt), DENSE_OP(div, s, d.c(2)));
    V half_t = DENSE_OP(mod, s, d.c(2));
    V lo = DENSE_OP(cmp_lt, r, paired_tail);
    V hi = DENSE_OP(cmp_lt, d.c(np - 1), r);
    unit_t
        = DENSE_OP(select,
                   lo,
                   DENSE_OP(add, re_half, r),
                   DENSE_OP(select, hi, DENSE_OP(add, DENSE_OP(sub, re_half, d.c(np)), r), unit_t));
    half_t = DENSE_OP(select, lo, d.c(0), DENSE_OP(select, hi, d.c(1), half_t));
    V in_rounds = DENSE_OP(cmp_lt, wi, rounds_end);
    return {DENSE_OP(select, in_rounds, unit, unit_t), DENSE_OP(select, in_rounds, half, half_t)};
}

V pair_block(const DecodeShape& d, V p, V half)
{
    auto* b = d.b;
    V first = DENSE_OP(cmp_lt, half, d.c(1));
    V last = DENSE_OP(sub, d.val(Nqb), d.c(1));
    return DENSE_OP(select, first, p, DENSE_OP(sub, last, p));
}

Decoded qb_major(const DecodeShape& d, V wi)
{
    auto* b = d.b;
    V bt = d.mod(wi, Batch);
    V rem = d.div(wi, Batch);
    V hq = d.mod(rem, Heads);
    V qb0 = d.div(rem, Heads);
    V qb;
    if(d.s.interleave && d.s.causal)
    {
        V odd = DENSE_OP(cmp_eq, DENSE_OP(mod, hq, d.c(2)), d.c(1));
        V last = DENSE_OP(sub, d.val(Nqb), d.c(1));
        qb = DENSE_OP(select, odd, DENSE_OP(sub, last, qb0), qb0);
    }
    else
        qb = d.branch(d.balance(), [&] { return fold(d, qb0); }, [&] { return qb0; });
    return {qb, hq, bt, nullptr};
}

Decoded hkv_major(const DecodeShape& d, V wi)
{
    auto* b = d.b;
    auto digits = [&](V idx, int scale) {
        V bt = d.mod(idx, Batch);
        V rem = d.div(idx, Batch);
        V hql = d.mod(rem, Gqa);
        V r2 = d.div(rem, Gqa);
        V blk = d.mod(r2, Nqb, scale);
        V hkv = d.div(r2, Nqb, scale);
        V hq = DENSE_OP(add, DENSE_OP(mul, hkv, d.val(Gqa)), hql);
        return std::array{bt, hq, blk, hkv};
    };
    auto paired = [&] {
        auto [unit, half] = pair_unit(d, wi);
        auto [bt, hq, p, hkv] = digits(unit, 2);
        return std::array{pair_block(d, p, half), hq, bt, hkv};
    };
    auto folded = [&] {
        auto [bt, hq, blk, hkv] = digits(wi, 1);
        return std::array{fold(d, blk), hq, bt, hkv};
    };
    // Unlike bt_hkv_minor, the non-paired path always folds, even if noncausal.
    auto [qb, hq, bt, hkv] = d.branch(d.pair_fold(d.s.num_persistent), paired, folded);
    return {qb, hq, bt, hkv};
}

Decoded bt_hkv_minor(const DecodeShape& d, V wi)
{
    auto* b = d.b;
    V bt = d.mod(wi, Batch);
    V rem = d.div(wi, Batch);
    V t = d.mod(rem, Heads);
    V blk = d.div(rem, Heads);
    V hql = d.div(DENSE_OP(mul, t, d.val(Gqa)), Heads);
    V hkv = DENSE_OP(sub, t, DENSE_OP(mul, hql, d.val(KvHeads)));
    V hq = DENSE_OP(add, DENSE_OP(mul, hkv, d.val(Gqa)), hql);
    V qb = d.branch(d.balance(), [&] { return fold(d, blk); }, [&] { return blk; });
    return {qb, hq, bt, hkv};
}

Decoded hq_minor_swz(const DecodeShape& d, V wi)
{
    auto* b = d.b;
    const int64_t m = d.s.chiplet_num_xcds;
    auto paired = [&] {
        auto [unit, half] = pair_unit(d, wi);
        V a = DENSE_OP(mod, unit, d.c(m));
        V rest = DENSE_OP(div, unit, d.c(m));
        V p = d.mod(rest, Nqb, 2);
        V hb = d.div(rest, Nqb, 2);
        return std::array{a, hb, pair_block(d, p, half)};
    };
    auto plain = [&] {
        V a = DENSE_OP(mod, wi, d.c(m));
        V rest = DENSE_OP(div, wi, d.c(m));
        V qb = d.mod(rest, Nqb);
        V hb = d.div(rest, Nqb);
        return std::array{a, hb, qb};
    };
    auto [a, hb, qb] = d.branch(d.pair_fold(d.s.num_persistent), paired, plain);
    V hq = DENSE_OP(add, DENSE_OP(mul, a, d.val(Heads, m)), d.mod(hb, Heads, m));
    V bt = d.div(hb, Heads, m);
    return {qb, hq, bt, nullptr};
}

// Both baked GQA orders first split cta into (lane, batch, kv head, qb pair).
// In the two-phase order the whole GQA group (rather than two CTAs) is a unit.
Decoded gqa_pair(const DecodeShape& d, V wi, bool two_phase)
{
    auto* b = d.b;
    assert(!d.runtime(Nqb));
    const int64_t nqb = d.baked(Nqb), gqa = d.baked(Gqa);
    const int64_t lanes = two_phase ? gqa : 2;
    V cta = DENSE_OP(mod, wi, d.c(d.s.num_persistent));
    V phase = DENSE_OP(div, wi, d.c(d.s.num_persistent));
    V lane = DENSE_OP(mod, cta, d.c(lanes));
    V rem = DENSE_OP(div, cta, d.c(lanes));
    V bt = DENSE_OP(mod, rem, d.c(d.s.batch));
    rem = DENSE_OP(div, rem, d.c(d.s.batch));
    V hkv = DENSE_OP(mod, rem, d.c(d.s.num_kv_heads));
    V qb_pair = DENSE_OP(div, rem, d.c(d.s.num_kv_heads));
    V hq, high;
    if(two_phase)
    {
        hq = DENSE_OP(add, DENSE_OP(mul, hkv, d.c(gqa)), lane);
        high = DENSE_OP(cmp_ne, phase, d.c(0));
    }
    else
    {
        high = DENSE_OP(cmp_ge, phase, d.c(gqa / 2));
        V phase_half = DENSE_OP(mod, phase, d.c(gqa / 2));
        V hql = DENSE_OP(add, DENSE_OP(mul, lane, d.c(gqa / 2)), phase_half);
        hq = DENSE_OP(add, DENSE_OP(mul, hkv, d.c(gqa)), hql);
    }
    V qb = DENSE_OP(select, high, DENSE_OP(sub, d.c(nqb - 1), qb_pair), qb_pair);
    return {qb, hq, bt, hkv};
}
} // namespace

Decoded decode_grid(Builder* b, const Spec& s, const DecodeRuntime& rt, V bx, V by, V bz)
{
    std::string_view order
        = rocke_attention_dense_spec_resolved_nonpersist_decode(&s.base, s.right_bound != 0);
    if(order == "qb_minor")
        return {bx, by, bz, nullptr};
    DecodeShape d{b, s.base, rt, false};
    V blk, hq, bt;
    if(order == "bt_hkv_minor")
    {
        hq = DENSE_OP(add, DENSE_OP(mul, d.mod(by, KvHeads), d.val(Gqa)), d.div(by, KvHeads));
        blk = bz;
        bt = bx;
    }
    else
    {
        assert(order == "hq_minor_swz");
        const int64_t m = s.base.chiplet_num_xcds;
        hq = DENSE_OP(add, DENSE_OP(mul, DENSE_OP(mod, bx, d.c(m)), d.val(Heads, m)), by);
        blk = DENSE_OP(div, bx, d.c(m));
        bt = bz;
    }
    V qb = d.branch(d.balance(), [&] { return reverse(d, blk); }, [&] { return blk; });
    return {qb, hq, bt, nullptr}; // Grid caller emits its own hq / gqa.
}

Decoded decode_persistent(Builder* b, const Spec& s, const DecodeRuntime& rt, V wi)
{
    std::string_view order
        = rocke_attention_dense_spec_resolved_persist_decode(&s.base, s.right_bound != 0);
    DecodeShape d{b, s.base, rt, true};
    if(order == "qb_major")
        return qb_major(d, wi);
    if(order == "hkv_major")
        return hkv_major(d, wi);
    if(order == "bt_hkv_minor")
        return bt_hkv_minor(d, wi);
    if(order == "hq_minor_swz")
        return hq_minor_swz(d, wi);
    assert(order == "gqa_pair" || order == "gqa_pair_2phase");
    return gqa_pair(d, wi, order == "gqa_pair_2phase");
}

V decode_hkv(Builder* b, const Spec& s, const DecodeRuntime& rt, V hq)
{
    return DecodeShape{b, s.base, rt, true}.div(hq, Gqa);
}
} // namespace rocke_dense_gfx950
