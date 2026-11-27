// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "attention_dense_grid.hpp"
#include "rocke/helper_helper_rocke.instances.gfx942.attention_tiled_2d.h"
#include "rocke/helper_rocke.helpers.attention.h"
#include "rocke/helper_rocke.helpers.mfma_attention.h"
#include "rocke/helper_rocke.helpers.schedule.h"
#include <algorithm>
#include <cstring>

namespace rocke_dense_gfx950::grid
{
namespace
{
template <class F>
void pv_steps(const Spec& spec, int d_tiles, int kk_steps, F step)
{
    if(std::strcmp(rocke_gfx950_attention_dense_resolved_pv_loop_order(&spec), "k_major") == 0)
        for(int k = 0; k < kk_steps; ++k)
            for(int d = 0; d < d_tiles; ++d)
                step(d, k);
    else
        for(int d = 0; d < d_tiles; ++d)
            for(int k = 0; k < kk_steps; ++k)
                step(d, k);
}
}

Scores Grid::do_qk(V kbuf)
{
    Scores scores{};
    for(int n = 0; n < N_SUB; ++n)
    {
        V acc = DENSE_OP(zero_vec_f32, 16);
        V row = K_GROUP == 1 ? DENSE_OP(add, DENSE_OP(const_i32, n * 32), lane_m)
                             : DENSE_OP(add, DENSE_OP(const_i32, n * (32 / K_GROUP)), k_lane_grp);
        for(int ks = 0; ks < K_STEPS; ++ks)
        {
            V col = DENSE_OP(add, DENSE_OP(const_i32, ks * 16), d_base);
            if(K_GROUP > 1)
                col = DENSE_OP(add, k_sub_col, col);
            V idx[] = {kbuf, row, col};
            V k = DENSE_OP(smem_load_vN, K_lds, idx, 3, dtype, 8);
            acc = rocke_mfma_attn_mfma_32x32x16_for_dtype(b, dtype, k, q_packs[ks], acc);
        }
        for(int i = 0; i < 16; ++i)
            scores[n][i] = DENSE_OP(vec_extract, acc, i);
    }
    return scores;
}

void Grid::do_mask(Scores& scores, V tile_idx, bool lower, bool upper)
{
    if(!causal && !lower)
        return;
    V tile = DENSE_OP(mul, tile_idx, DENSE_OP(const_i32, BN));
    V q = DENSE_OP(add, q_tok0, rocke__mfma_32x32_c_col(b, lane, 0));
    if(DIAG_OFF)
        q = DENSE_OP(add, q, DENSE_OP(const_i32, DIAG_OFF));
    V lo = lower ? DENSE_OP(sub, q, DENSE_OP(const_i32, W)) : nullptr;
    V hi = RIGHT ? DENSE_OP(add, q, DENSE_OP(const_i32, RIGHT)) : q;
    for(int n = 0; n < N_SUB; ++n)
    {
        V base = DENSE_OP(add, tile, DENSE_OP(const_i32, n * 32));
        for(int i = 0; i < 16; ++i)
        {
            V k = DENSE_OP(add, base, rocke__mfma_32x32_c_row(b, lane, i));
            V& s = scores[n][i];
            if(upper && causal)
                s = DENSE_OP(select, DENSE_OP(cmp_le, k, hi), s, neg_inf);
            if(lower)
                s = DENSE_OP(select, DENSE_OP(cmp_gt, k, lo), s, neg_inf);
        }
    }
}

void Grid::do_kbound_mask(Scores& scores, V tile_idx)
{
    V tile = DENSE_OP(mul, tile_idx, DENSE_OP(const_i32, BN));
    for(int n = 0; n < N_SUB; ++n)
    {
        V base = DENSE_OP(add, tile, DENSE_OP(const_i32, n * 32));
        for(int i = 0; i < 16; ++i)
        {
            V k = DENSE_OP(add, base, rocke__mfma_32x32_c_row(b, lane, i));
            V keep = DENSE_OP(cmp_lt, k, DENSE_OP(const_i32, Skv));
            scores[n][i] = DENSE_OP(select, keep, scores[n][i], neg_inf);
        }
    }
}

std::tuple<V, V, V> Grid::softmax_max(const Scores& scores, V m_i)
{
    V local = neg_inf;
    for(int n = 0; n < N_SUB; ++n)
        for(V s : scores[n])
            local = DENSE_OP(fmax, local, s);
    V raw = DENSE_OP(fmax, local, DENSE_OP(warp_shuffle_xor, local, 32));
    V tile = DENSE_OP(fmul, raw, qk_scale);
    V within = nullptr, skip = nullptr;
    if(LAZY_RESCALE)
    {
        within
            = DENSE_OP(fcmp, "ole", DENSE_OP(fsub, tile, m_i), DENSE_OP(const_f32, LAZY_THRESHOLD));
        V below = DENSE_OP(select, within, DENSE_OP(const_i32, 1), DENSE_OP(const_i32, 0));
        skip = DENSE_OP(cmp_ne, DENSE_OP(wave_all, below), DENSE_OP(const_i32, 0));
    }
    V m = DENSE_OP(fmax, m_i, tile);
    if(LAZY_RESCALE)
        m = DENSE_OP(select, within, m_i, m);
    V alpha = exp2(DENSE_OP(fsub, m_i, m));
    if(LAZY_RESCALE)
        alpha = DENSE_OP(select, within, DENSE_OP(const_f32, 1.0), alpha);
    return {m, alpha, skip};
}

Packs Grid::relayout_p(const Scores& p)
{
    Packs packs{};
    for(int k = 0; k < KK_STEPS; ++k)
    {
        V elems[8];
        for(int j = 0; j < 8; ++j)
        {
            int key = k * 16 + (j / 4) * 8 + j % 4;
            int row = key % 32;
            elems[j] = DENSE_OP(cast_f32_to, p[key / 32][(row / 8) * 4 + row % 4], dtype);
        }
        packs[k] = DENSE_OP(vec_pack, elems, 8, dtype);
    }
    return packs;
}

void Grid::do_pv(Acc& out, const Packs& p, V vbuf)
{
    pv_steps(spec, D_TILES, KK_STEPS, [&](int d, int k) {
        V v = rocke_pv32_v_load_paired(b, V_lds, vbuf, d, k, lane_h, lane_m, dtype);
        out[d] = rocke_mfma_attn_mfma_32x32x16_for_dtype(b, dtype, v, p[k], out[d]);
    });
}

void Grid::rescale_o(Acc& out, V alpha)
{
    for(int d = 0; d < D_TILES; ++d)
    {
        V elems[16];
        for(int i = 0; i < 16; ++i)
            elems[i] = DENSE_OP(fmul, DENSE_OP(vec_extract, out[d], i), alpha);
        out[d] = DENSE_OP(vec_pack, elems, 16, rocke_f32());
    }
}

std::pair<Scores, V>
    Grid::pv_fused_exp(Acc& out, const Packs& packs, V vbuf, const Scores& scores, V m_new)
{
    V neg_m = DENSE_OP(fneg, m_new);
    Scores p{};
    int slot = 0;
    auto emit = [&] {
        int n = slot / 16, i = slot % 16;
        p[n][i] = exp2(DENSE_OP(fma, scores[n][i], qk_scale, neg_m));
        ++slot;
    };
    int per = rocke_gfx950_attention_dense_resolved_exp_per_pv_step(&spec);
    bool group = rocke_gfx950_attention_dense_resolved_pv_sched_group_template(&spec);
    pv_steps(spec, D_TILES, KK_STEPS, [&](int d, int k) {
        V v = rocke_pv32_v_load_paired(b, V_lds, vbuf, d, k, lane_h, lane_m, dtype);
        out[d] = rocke_mfma_attn_mfma_32x32x16_for_dtype(b, dtype, v, packs[k], out[d]);
        int count = std::min(per, N_SUB * 16 - slot);
        for(int j = 0; j < count; ++j)
            emit();
        if(group)
        {
            DENSE_OP(sched_group_barrier, ROCKE_SCHED_DS_READ, spec.pv_sched_group_ds_read, 0);
            DENSE_OP(sched_group_barrier, ROCKE_SCHED_MFMA, 1, 0);
            DENSE_OP(sched_group_barrier, ROCKE_SCHED_VALU, std::max(1, count), 0);
            DENSE_OP(sched_group_barrier, ROCKE_SCHED_TRANS, std::max(1, count), 0);
        }
    });
    while(slot < N_SUB * 16)
        emit();
    V local = DENSE_OP(const_f32, 0.0);
    for(int n = 0; n < N_SUB; ++n)
        for(V prob : p[n])
            local = DENSE_OP(fadd, local, prob);
    return {p, DENSE_OP(fadd, local, DENSE_OP(warp_shuffle_xor, local, 32))};
}
} // namespace rocke_dense_gfx950::grid
