// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#ifndef ROCKE_GFX950_ATTENTION_DENSE_GRID_HPP
#define ROCKE_GFX950_ATTENTION_DENSE_GRID_HPP

#include "attention_dense_internal.hpp"

namespace rocke_dense_gfx950::grid
{
// Only N_SUB, D_TILES and KK_STEPS entries respectively are live. Admission
// bounds block_n by block_m <= 256 and head_size by 128.
using Scores = std::array<std::array<V, 16>, 8>;
using Acc = std::array<V, 4>;
using Packs = std::array<V, 16>;

struct Grid
{
    Builder* b;
    const Spec& spec;

    Grid(Builder* builder, const Spec& s)
        : b(builder)
        , spec(s)
    {
    }

    // Host geometry and choices made once per kernel, never SSA constants.
    int B, Sq, Skv, Hq, Hkv, D, BLOCK_M, WAVES, BN, NBUF, PAD, W, Wt;
    int RIGHT, DIAG_OFF, KVS, D_TILES, K_STEPS, N_SUB, KK_STEPS, ROWS_PER_INSTR;
    int n_per, n_ktiles, K_GROUP, LOOP_VMCNT, PV_PRIORITY, PV_FENCE_MASK, IGLP;
    int64_t DIAG_TILES;
    bool runtime_shape, causal, RAGGED, varlen, KV_FP8, WINDOW_EMPTY, EMPTY_ROWS;
    bool RAG_KBOUND, use_sinks, PV_FENCE, LAZY_RESCALE;
    double LAZY_THRESHOLD;
    Type dtype, kv_dtype;

    // ABI and lane/decode results, shared by the DMA, math and phase emitters.
    V q, k, v, o, scale, k_scale, v_scale, batch_p, seqlen_q_p, seqlen_kv_p;
    V sinks = nullptr, cu_q = nullptr, cu_kv = nullptr;
    V block_tables = nullptr, kv_lens = nullptr, bt_stride = nullptr, lse = nullptr;
    V hq_p = nullptr, hkv_p = nullptr, gqa_p = nullptr, stride_q_p = nullptr, stride_k_p = nullptr;
    V tid, wave, lane, lane_m, lane_h, d_base, qk_scale, neg_inf, rcp_ln2, one_f;
    V qb, hq, bt, hkv, seg, q_tok0, q_base, k_base;
    V q_seq0, kv_seq1, seqlen_q_b, seqlen_kv_b;
    std::array<V, 3> workspace;

    // Staged K/V, lane-local Q fragments, and loop bounds.
    V K_lds, V_lds, k_lane_grp, k_sub_col, k_rsrc, v_rsrc;
    V zero_soff, K_lds_addr, V_lds_addr, pg_seq_base, pg_n_pages;
    std::array<V, 8> q_packs;
    int K_BYTES_PER_BUF, K_GROUP_BYTES, V_BYTES_PER_BUF, V_GROUP_BYTES, ROWS_PER_WAVE;
    V start_tile, n_upper, seg_lo1, seg_hi;

    void configure();
    void declare_params();
    void decode();
    void prepare_loads();
    void load_tile(V buf_val, V tile_idx);
    Scores do_qk(V kbuf);
    void do_mask(Scores&, V tile_idx, bool lower = false, bool upper = true);
    void do_kbound_mask(Scores&, V tile_idx);
    std::tuple<V, V, V> softmax_max(const Scores&, V m);
    Packs relayout_p(const Scores&);
    void do_pv(Acc&, const Packs&, V vbuf);
    void rescale_o(Acc&, V alpha);
    std::pair<Scores, V> pv_fused_exp(Acc&, const Packs&, V vbuf, const Scores&, V m);
    void run_loops(V& m, V& l, Acc& o_acc, Packs& p_prev);
    void pipeline();
    void epilogue(V m, V l, Acc& o_acc, const Packs& p_prev);

    V stride_q()
    {
        return runtime_shape ? stride_q_p : rocke_b_const_i32(b, int64_t(Hq) * D);
    }
    V stride_k()
    {
        return runtime_shape ? stride_k_p : rocke_b_const_i32(b, int64_t(Hkv) * D);
    }
    V gqa_value()
    {
        return runtime_shape ? gqa_p : rocke_b_const_i32(b, Hq / Hkv);
    }
    V hq_count()
    {
        return runtime_shape ? hq_p : rocke_b_const_i32(b, Hq);
    }
    V exp2(V a)
    {
        return spec.use_exp2_fast ? rocke_b_exp2_fast(b, a) : rocke_b_exp2(b, a);
    }
};
} // namespace rocke_dense_gfx950::grid
#endif
