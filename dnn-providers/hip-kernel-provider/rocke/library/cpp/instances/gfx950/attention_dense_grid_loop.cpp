// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "attention_dense_grid.hpp"
#include <algorithm>
#include <cstdio>

namespace rocke_dense_gfx950::grid
{
namespace
{
// Only the active prefix is carried: the ABI of each phase is m,l,o...,pk....
constexpr int MaxCarry = 2 + std::tuple_size_v<Acc> + std::tuple_size_v<Packs>;
struct LoopArgs
{
    rocke_iter_arg_t args[MaxCarry];
    char names[MaxCarry][32];
    int size = 0;

    template <class... T>
    void append(V init, const char* fmt, T... operands)
    {
        std::snprintf(names[size], sizeof(names[size]), fmt, operands...);
        args[size] = {names[size], init};
        ++size;
    }
};
} // namespace

void Grid::run_loops(V& m, V& l, Acc& o_acc, Packs& p_prev)
{
    auto c = [&](int64_t i) { return rocke_b_const_i32(b, i); };
    int rs_counter = 0;
    auto phase = [&](V lo,
                     V hi,
                     const char* iv,
                     const char* suffix,
                     bool lower,
                     bool upper,
                     bool kbound) {
        LoopArgs args;
        args.append(m, "m%s", suffix);
        args.append(l, "l%s", suffix);
        for(int dt = 0; dt < D_TILES; ++dt)
            args.append(o_acc[dt], "o%d%s", dt, suffix);
        for(int kk = 0; kk < KK_STEPS; ++kk)
            args.append(p_prev[kk], "pk%d%s", kk, suffix);
        auto loop = rocke_b_scf_for_iter(b, lo, hi, c(1), args.args, args.size, iv, false, true);
        if(!loop.op)
            return;
        rocke_b_region_enter(b, loop.body);
        if(IGLP != -1)
            rocke_b_iglp_opt(b, IGLP);
        V j = loop.iv;
        V m_i = loop.iter_vars[0];
        V l_i = loop.iter_vars[1];
        std::copy_n(loop.iter_vars + 2, D_TILES, o_acc.begin());
        std::copy_n(loop.iter_vars + 2 + D_TILES, KK_STEPS, p_prev.begin());
        V kbuf = DENSE_OP(mod, j, c(NBUF));
        V vbuf_prev = DENSE_OP(mod, DENSE_OP(add, j, c(NBUF - 1)), c(NBUF));
        V pbuf = DENSE_OP(mod, DENSE_OP(add, j, c(1)), c(NBUF));

        rocke_b_s_waitcnt(b, LOOP_VMCNT, -1, -1);
        rocke_b_s_barrier_bare(b);
        Scores s = do_qk(kbuf);
        if(lower || upper)
            do_mask(s, j, lower, upper);
        if(kbound)
            do_kbound_mask(s, j);
        auto [m_new, alpha, skip] = softmax_max(s, m_i);
        if(PV_FENCE)
            rocke_b_sched_barrier(b, PV_FENCE_MASK);
        if(PV_PRIORITY)
            rocke_b_s_setprio(b, PV_PRIORITY);
        auto [p_vals, l_tile] = pv_fused_exp(o_acc, p_prev, vbuf_prev, s, m_new);
        if(PV_PRIORITY)
            rocke_b_s_setprio(b, 0);

        V l_new;
        if(LAZY_RESCALE)
        {
            // Zero trips preserve the post-PV accumulator and old denominator.
            int tg = ++rs_counter;
            V trips = DENSE_OP(select, skip, c(0), c(1));
            LoopArgs rs_args;
            for(int dt = 0; dt < D_TILES; ++dt)
                rs_args.append(o_acc[dt], "ro%d_rs%d", dt, tg);
            rs_args.append(l_i, "rl_rs%d", tg);
            char rs_iv[16];
            std::snprintf(rs_iv, sizeof(rs_iv), "rs%d", tg);
            auto rs = DENSE_OP(
                scf_for_iter, c(0), trips, c(1), rs_args.args, rs_args.size, rs_iv, false, true);
            if(!rs.op)
            {
                rocke_b_region_leave(b);
                return;
            }
            rocke_b_region_enter(b, rs.body);
            std::copy_n(rs.iter_vars, D_TILES, o_acc.begin());
            rescale_o(o_acc, alpha);
            std::array<V, std::tuple_size_v<Acc> + 1> scaled;
            std::copy_n(o_acc.begin(), D_TILES, scaled.begin());
            scaled[D_TILES] = DENSE_OP(fmul, rs.iter_vars[D_TILES], alpha);
            rocke_b_scf_yield(b, scaled.data(), D_TILES + 1);
            rocke_b_region_leave(b);
            std::copy_n(rs.op->results, D_TILES, o_acc.begin());
            l_new = DENSE_OP(fadd, rs.op->results[D_TILES], l_tile);
        }
        else
        {
            l_new = DENSE_OP(fadd, DENSE_OP(fmul, l_i, alpha), l_tile);
            rescale_o(o_acc, alpha);
        }
        Packs p_packs = relayout_p(p_vals);
        rocke_b_s_barrier_bare(b);
        load_tile(pbuf, DENSE_OP(add, j, c(1)));
        std::array<V, MaxCarry> yielded;
        yielded[0] = m_new;
        yielded[1] = l_new;
        std::copy_n(o_acc.begin(), D_TILES, yielded.begin() + 2);
        std::copy_n(p_packs.begin(), KK_STEPS, yielded.begin() + 2 + D_TILES);
        rocke_b_scf_yield(b, yielded.data(), args.size);
        rocke_b_region_leave(b);

        m = loop.op->results[0];
        l = loop.op->results[1];
        std::copy_n(loop.op->results + 2, D_TILES, o_acc.begin());
        std::copy_n(loop.op->results + 2 + D_TILES, KK_STEPS, p_prev.begin());
    };

    auto clamp = [&](V x, V lo, V hi) {
        x = DENSE_OP(select, DENSE_OP(cmp_lt, x, lo), lo, x);
        return DENSE_OP(select, DENSE_OP(cmp_lt, x, hi), x, hi);
    };
    if(W > 0)
    {
        V diag_start = DENSE_OP(mul, qb, c(n_per));
        V a = DENSE_OP(add, start_tile, c(1));
        V left_end = DENSE_OP(add, diag_start, c(n_per - Wt));
        if(causal && DIAG_TILES)
            diag_start = DENSE_OP(add, diag_start, c(DIAG_TILES));
        V mid_lo = clamp(left_end, a, n_upper);
        V mid_hi = causal ? clamp(diag_start, mid_lo, n_upper) : n_upper;
        phase(a, mid_lo, "swl", "", true, causal, false);
        phase(mid_lo, mid_hi, "swm", "_m", false, false, false);
        if(causal)
            phase(mid_hi, n_upper, "swr", "_r", true, true, false);
    }
    else if(causal && spec.causal_diag_split)
    {
        V diag_start = DENSE_OP(mul, qb, c(n_per));
        if(DIAG_TILES)
            diag_start = DENSE_OP(add, diag_start, c(DIAG_TILES));
        V body_upper
            = KVS > 1
                  ? clamp(diag_start, seg_lo1, seg_hi)
                  : DENSE_OP(select, DENSE_OP(cmp_lt, diag_start, n_upper), diag_start, n_upper);
        phase(KVS > 1 ? seg_lo1 : c(1), body_upper, "nb", "", false, false, RAG_KBOUND);
        V tail_lo = KVS > 1
                        ? body_upper
                        : DENSE_OP(select, DENSE_OP(cmp_lt, diag_start, c(1)), c(1), diag_start);
        phase(tail_lo, KVS > 1 ? seg_hi : n_upper, "nt", "_t", false, true, RAG_KBOUND);
    }
    else
        phase(KVS > 1 ? seg_lo1 : c(1),
              KVS > 1 ? seg_hi : n_upper,
              "nkt",
              "",
              false,
              causal,
              RAG_KBOUND);
}
} // namespace rocke_dense_gfx950::grid
