// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "attention_dense_grid.hpp"
#include "rocke/helper_helper_rocke.instances.gfx942.attention_tiled_2d.h"
#include "rocke/instance_lse_store_internal.h"
#include <cstring>

namespace rocke_dense_gfx950::grid
{

void Grid::configure()
{
    const auto& base = spec.base;
    B = base.batch;
    Sq = base.seqlen_q;
    Skv = base.seqlen_kv;
    Hq = base.num_query_heads;
    Hkv = base.num_kv_heads;
    D = base.head_size;
    dtype = std::strcmp(base.dtype, "bf16") == 0 ? rocke_bf16() : rocke_f16();
    causal = base.causal;
    BLOCK_M = base.block_m;
    WAVES = BLOCK_M / 32;
    BN = base.block_n;
    NBUF = spec.lds_num_buffers;
    PAD = spec.lds_k_row_pad;
    W = base.sliding_window;
    Wt = W / BN;
    DIAG_OFF = base.causal_bottom_right ? Skv - Sq : 0;
    RIGHT = spec.right_bound;
    DIAG_TILES = rocke_floor_div(int64_t(DIAG_OFF) + RIGHT, BN);
    varlen = base.varlen;
    WINDOW_EMPTY = W > 0 && !varlen && !spec.emit_lse;
    EMPTY_ROWS = WINDOW_EMPTY && Sq > int64_t(Skv) + W - 1;
    RAGGED = base.ragged;
    KV_FP8 = rocke_gfx950_attention_dense_kv_elem_bytes(&spec) == 1;
    kv_dtype = KV_FP8 ? rocke_fp8e4m3() : dtype;
    use_sinks = base.use_sinks;
    KVS = spec.kv_segments;
    runtime_shape = rocke_gfx950_attention_dense_runtime_shape(&spec);
    LAZY_RESCALE = base.lazy_rescale;
    LAZY_THRESHOLD = spec.lazy_rescale_threshold;
    PV_PRIORITY = spec.pv_priority;
    PV_FENCE = rocke_gfx950_attention_dense_resolved_pv_sched_fence(&spec);
    PV_FENCE_MASK = spec.pv_sched_fence_mask;
    IGLP = rocke_gfx950_attention_dense_resolved_iglp_mode(&spec);
    K_STEPS = D / 16;
    D_TILES = D / 32;
    N_SUB = BN / 32;
    KK_STEPS = BN / 16;
    ROWS_PER_INSTR = 128 / D;
    n_per = BLOCK_M / BN;
    n_ktiles = RAGGED ? static_cast<int>((int64_t(Skv) + BN - 1) / BN) : Skv / BN;
    RAG_KBOUND = RAGGED && Skv % BN && (!causal || int64_t(Sq) + RIGHT > Skv);
}

void Grid::declare_params()
{
    rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", WAVES * 64);
    rocke_attr_set_int(b, &b->kernel->attrs, "waves_per_eu", spec.base.waves_per_eu);
    q = pointer_param(b, "q_ptr", dtype, false, 16);
    k = pointer_param(b, "k_ptr", kv_dtype, false, 16);
    v = pointer_param(b, "v_ptr", kv_dtype, false, 16);
    o = pointer_param(b, "o_ptr", dtype, true, 16);
    scale = rocke_b_param(b, "scale", rocke_f32(), nullptr);
    if(KV_FP8)
    {
        k_scale = rocke_b_param(b, "k_scale", rocke_f32(), nullptr);
        v_scale = rocke_b_param(b, "v_scale", rocke_f32(), nullptr);
    }
    batch_p = rocke_b_param(b, "batch", rocke_i32(), nullptr);
    seqlen_q_p = rocke_b_param(b, "seqlen_q", rocke_i32(), nullptr);
    seqlen_kv_p = rocke_b_param(b, "seqlen_kv", rocke_i32(), nullptr);
    if(runtime_shape)
    {
        hq_p = rocke_b_param(b, "num_query_heads", rocke_i32(), nullptr);
        hkv_p = rocke_b_param(b, "num_kv_heads", rocke_i32(), nullptr);
        gqa_p = DENSE_OP(div, hq_p, hkv_p);
        stride_q_p = DENSE_OP(mul, hq_p, DENSE_OP(const_i32, D));
        stride_k_p = DENSE_OP(mul, hkv_p, DENSE_OP(const_i32, D));
    }
    if(use_sinks)
        sinks = pointer_param(b, "sink_ptr", dtype, false, 16);
    if(varlen)
    {
        cu_q = pointer_param(b, "cu_seqlens_q", rocke_i32(), false, 4);
        cu_kv = pointer_param(b, "cu_seqlens_kv", rocke_i32(), false, 4);
    }
    if(spec.base.paged)
    {
        block_tables = pointer_param(b, "block_tables", rocke_i32(), false, 4);
        kv_lens = pointer_param(b, "kv_lens", rocke_i32(), false, 4);
        bt_stride = rocke_b_param(b, "block_table_stride", rocke_i32(), nullptr);
    }
    if(spec.emit_lse)
        lse = pointer_param(b, "lse_ptr", rocke_f32(), true, 4);
    if(KVS > 1)
        workspace = segment_params(b);
}

void Grid::decode()
{
    qk_scale = DENSE_OP(fmul, scale, DENSE_OP(const_f32, log2e));
    tid = DENSE_OP(thread_id_x);
    wave = DENSE_OP(div, tid, DENSE_OP(const_i32, 64));
    lane = DENSE_OP(mod, tid, DENSE_OP(const_i32, 64));
    lane_m = DENSE_OP(mod, lane, DENSE_OP(const_i32, 32));
    lane_h = DENSE_OP(div, lane, DENSE_OP(const_i32, 32));
    d_base = DENSE_OP(mul, lane_h, DENSE_OP(const_i32, 8));
    neg_inf = DENSE_OP(const_f32, -0x1p99);
    if(use_sinks)
    {
        rcp_ln2 = DENSE_OP(const_f32, log2e);
        one_f = DENSE_OP(const_f32, 1.0);
    }

    DecodeRuntime rt;
    rt.seqlen_q = seqlen_q_p;
    if(runtime_shape)
    {
        rt.batch = batch_p;
        rt.seqlen_kv = seqlen_kv_p;
        rt.num_query_heads = hq_p;
        rt.num_kv_heads = hkv_p;
        rt.gqa = gqa_p;
    }
    V bx = DENSE_OP(block_id_x);
    if(KVS > 1)
    {
        seg = DENSE_OP(mod, bx, DENSE_OP(const_i32, KVS));
        bx = DENSE_OP(div, bx, DENSE_OP(const_i32, KVS));
    }
    V by = DENSE_OP(block_id_y);
    V bz = DENSE_OP(block_id_z);
    const Decoded ids = decode_grid(b, spec, rt, bx, by, bz);
    qb = ids.qb;
    hq = ids.hq;
    bt = ids.bt;
    // The reference recomputes hkv even for decodes that already emitted it.
    hkv = DENSE_OP(div, hq, gqa_value());
    q_tok0 = DENSE_OP(add,
                      DENSE_OP(mul, qb, DENSE_OP(const_i32, BLOCK_M)),
                      DENSE_OP(mul, wave, DENSE_OP(const_i32, 32)));
    if(varlen)
    {
        q_seq0 = DENSE_OP(global_load_i32, cu_q, bt, 4);
        V q_seq1 = DENSE_OP(global_load_i32, cu_q, DENSE_OP(add, bt, DENSE_OP(const_i32, 1)), 4);
        seqlen_q_b = DENSE_OP(sub, q_seq1, q_seq0);
        V kv_seq0 = DENSE_OP(global_load_i32, cu_kv, bt, 4);
        kv_seq1 = DENSE_OP(global_load_i32, cu_kv, DENSE_OP(add, bt, DENSE_OP(const_i32, 1)), 4);
        seqlen_kv_b = DENSE_OP(sub, kv_seq1, kv_seq0);
        auto guard = DENSE_OP(
            scf_if, DENSE_OP(cmp_ge, DENSE_OP(mul, qb, DENSE_OP(const_i32, BLOCK_M)), seqlen_q_b));
        rocke_b_region_enter(b, guard.then_region);
        rocke_b_ret(b);
        rocke_b_region_leave(b);
        q_base = DENSE_OP(
            add, DENSE_OP(mul, q_seq0, stride_q()), DENSE_OP(mul, hq, DENSE_OP(const_i32, D)));
        k_base = DENSE_OP(
            add, DENSE_OP(mul, kv_seq0, stride_k()), DENSE_OP(mul, hkv, DENSE_OP(const_i32, D)));
    }
    else
    {
        q_base = DENSE_OP(add,
                          DENSE_OP(mul, DENSE_OP(mul, bt, seqlen_q_p), stride_q()),
                          DENSE_OP(mul, hq, DENSE_OP(const_i32, D)));
        k_base = DENSE_OP(add,
                          DENSE_OP(mul, DENSE_OP(mul, bt, seqlen_kv_p), stride_k()),
                          DENSE_OP(mul, hkv, DENSE_OP(const_i32, D)));
    }
}

void Grid::pipeline()
{
    V tiles = varlen          ? DENSE_OP(div, seqlen_kv_b, DENSE_OP(const_i32, BN))
              : runtime_shape ? DENSE_OP(div, seqlen_kv_p, DENSE_OP(const_i32, BN))
                              : DENSE_OP(const_i32, n_ktiles);
    n_upper = tiles;
    if(causal)
    {
        n_upper = DENSE_OP(
            add,
            DENSE_OP(mul, qb, DENSE_OP(const_i32, n_per)),
            DENSE_OP(const_i32, rocke_floor_div(int64_t(BLOCK_M) - 1 + DIAG_OFF + RIGHT, BN) + 1));
        n_upper = DENSE_OP(select, DENSE_OP(cmp_lt, n_upper, tiles), n_upper, tiles);
    }
    if(W > 0)
    {
        V diag = DENSE_OP(mul, qb, DENSE_OP(const_i32, n_per));
        V lo = DENSE_OP(sub, diag, DENSE_OP(const_i32, Wt));
        start_tile = DENSE_OP(
            select, DENSE_OP(cmp_gt, lo, DENSE_OP(const_i32, 0)), lo, DENSE_OP(const_i32, 0));
        if(runtime_shape)
        {
            V last = DENSE_OP(sub, tiles, DENSE_OP(const_i32, 1));
            start_tile = DENSE_OP(select, DENSE_OP(cmp_lt, start_tile, last), start_tile, last);
        }
        else if(EMPTY_ROWS)
        {
            start_tile = DENSE_OP(select,
                                  DENSE_OP(cmp_lt, start_tile, DENSE_OP(const_i32, n_ktiles - 1)),
                                  start_tile,
                                  DENSE_OP(const_i32, n_ktiles - 1));
        }
    }
    else if(KVS > 1)
    {
        std::tie(start_tile, seg_hi)
            = segment_tile_range(b, seg, DENSE_OP(const_i32, 0), n_upper, KVS);
        seg_lo1 = DENSE_OP(add, start_tile, DENSE_OP(const_i32, 1));
    }
    else
        start_tile = DENSE_OP(const_i32, 0);

    V buf = DENSE_OP(mod, start_tile, DENSE_OP(const_i32, NBUF));
    V buf1 = DENSE_OP(
        mod, DENSE_OP(add, start_tile, DENSE_OP(const_i32, 1)), DENSE_OP(const_i32, NBUF));
    load_tile(buf, start_tile);
    load_tile(buf1, DENSE_OP(add, start_tile, DENSE_OP(const_i32, 1)));
    DENSE_OP(s_waitcnt, 0, -1, -1);
    DENSE_OP(s_barrier_bare);

    Scores s0 = do_qk(buf);
    do_mask(s0, start_tile, W > 0, W > 0 ? causal : true);
    if(RAG_KBOUND)
        do_kbound_mask(s0, start_tile);
    V m_init = neg_inf;
    if(use_sinks)
        m_init = DENSE_OP(
            fmul, DENSE_OP(cast_to_f32, DENSE_OP(global_load, sinks, hq, dtype, 2)), rcp_ln2);
    auto [m, alpha, skip] = softmax_max(s0, m_init);
    V neg_m = DENSE_OP(fneg, m);
    Scores p0{};
    for(int n = 0; n < N_SUB; ++n)
        for(int i = 0; i < 16; ++i)
            p0[n][i] = exp2(DENSE_OP(fma, s0[n][i], qk_scale, neg_m));
    V local = DENSE_OP(const_f32, 0.0);
    for(int n = 0; n < N_SUB; ++n)
        for(V p : p0[n])
            local = DENSE_OP(fadd, local, p);
    V l = DENSE_OP(fadd, local, DENSE_OP(warp_shuffle_xor, local, 32));
    if(use_sinks)
        l = DENSE_OP(fadd, l, DENSE_OP(fmul, one_f, alpha));
    Acc o_acc{};
    for(int d = 0; d < D_TILES; ++d)
        o_acc[d] = DENSE_OP(zero_vec_f32, 16);
    Packs packs = relayout_p(p0);
    run_loops(m, l, o_acc, packs);
    epilogue(m, l, o_acc, packs);
}

void Grid::epilogue(V m, V l, Acc& o_acc, const Packs& p_prev)
{
    // Drain the last prefetched V, also for an empty segment (whose output
    // statistics are replaced below). No pending work survives this store.
    DENSE_OP(s_waitcnt, 0, -1, -1);
    DENSE_OP(s_barrier_bare);
    V seg_nonempty = nullptr;
    V last_end = n_upper;
    if(KVS > 1)
    {
        seg_nonempty = DENSE_OP(cmp_lt, start_tile, seg_hi);
        last_end = DENSE_OP(select, seg_nonempty, seg_hi, seg_lo1);
    }
    V last_buf = DENSE_OP(
        mod, DENSE_OP(add, last_end, DENSE_OP(const_i32, NBUF - 1)), DENSE_OP(const_i32, NBUF));
    do_pv(o_acc, p_prev, last_buf);
    std::span<V const> output(o_acc.data(), D_TILES);
    if(KVS > 1)
    {
        V qtok = DENSE_OP(add, q_tok0, rocke__mfma_32x32_c_col(b, lane, 0));
        V token = DENSE_OP(add, DENSE_OP(mul, bt, seqlen_q_p), qtok);
        V heads = hq_count();
        V max_out = DENSE_OP(select, seg_nonempty, m, neg_inf);
        V denom = DENSE_OP(select, seg_nonempty, l, DENSE_OP(const_f32, 0.0));
        emit_segment_store(
            b, workspace, token, hq, heads, seg, KVS, D, output, max_out, denom, lane_h);
        DENSE_OP(ret);
        return;
    }
    V denom = l;
    if(spec.emit_lse)
    {
        V positive = DENSE_OP(fcmp, "ogt", l, DENSE_OP(const_f32, 0.0));
        denom = DENSE_OP(select, positive, l, DENSE_OP(const_f32, 1.0));
    }
    V rcp_l = DENSE_OP(rcp, denom);
    if(EMPTY_ROWS && !runtime_shape)
        rcp_l = DENSE_OP(select,
                         DENSE_OP(fcmp, "ogt", m, DENSE_OP(const_f32, -0x1p33)),
                         rcp_l,
                         DENSE_OP(const_f32, 0.0));
    V base = DENSE_OP(add,
                      DENSE_OP(mul, varlen ? q_seq0 : DENSE_OP(mul, bt, seqlen_q_p), stride_q()),
                      DENSE_OP(mul, hq, DENSE_OP(const_i32, D)));
    V qtok = DENSE_OP(add, q_tok0, rocke__mfma_32x32_c_col(b, lane, 0));
    V row = DENSE_OP(add, base, DENSE_OP(mul, qtok, stride_q()));
    V d_half = DENSE_OP(mul, lane_h, DENSE_OP(const_i32, 4));
    V has_key = nullptr;
    if(spec.emit_lse || (WINDOW_EMPTY && runtime_shape))
        has_key = rocke_dense_row_has_key(
            b, qtok, varlen ? seqlen_kv_b : seqlen_kv_p, causal, DIAG_OFF, W, RIGHT);
    V exists = nullptr;
    if(spec.emit_lse)
    {
        V index
            = varlen
                  ? DENSE_OP(add, DENSE_OP(mul, DENSE_OP(add, q_seq0, qtok), hq_count()), hq)
                  : DENSE_OP(
                        add,
                        DENSE_OP(mul, DENSE_OP(add, DENSE_OP(mul, bt, hq_count()), hq), seqlen_q_p),
                        qtok);
        exists = DENSE_OP(cmp_lt, qtok, varlen ? seqlen_q_b : seqlen_q_p);
        dense_lse_output(
            b, lse, index, m, l, exists, has_key, lane_h, use_sinks ? sinks : nullptr, hq, dtype);
    }
    rocke_region_t* guard = nullptr;
    if(RAGGED)
        guard = DENSE_OP(scf_if, DENSE_OP(cmp_lt, qtok, DENSE_OP(const_i32, Sq))).then_region;
    if(spec.emit_lse && varlen)
        guard = DENSE_OP(scf_if, exists).then_region;
    if(guard)
        rocke_b_region_enter(b, guard);
    emit_o_store(b, o, row, d_half, output, rcp_l, dtype, spec.o_store_width, has_key);
    if(guard)
        rocke_b_region_leave(b);
    DENSE_OP(ret);
}
} // namespace rocke_dense_gfx950::grid

void rocke_dense_gfx950::build_grid(Builder* b, const Spec& s)
{
    grid::Grid state{b, s};
    state.configure();
    state.declare_params();
    state.decode();
    state.prepare_loads();
    state.pipeline();
}
