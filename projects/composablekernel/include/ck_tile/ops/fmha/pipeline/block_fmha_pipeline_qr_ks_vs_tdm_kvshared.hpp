// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm_pingpong.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm_kvshared_policy.hpp"

namespace ck_tile {

// qr_tdm_pingpong for K and V that are one tensor (the caller passes the same K and V
// views): each K/V tile is staged once, by one TDM transfer into one of two LDS buffers,
// and serves both gemms, the QK gemm reading it row-wise and the PV gemm with transposed
// reads. Trip i reads buffer (i & 1) while the TDM fills the other with tile i + 1, issued
// at the top of the trip, so the next tile's transfer runs under the whole of this one.
// The V view is not read.
template <typename Problem_, typename Policy_ = BlockFmhaPipelineQRKSVSTdmKVSharedPolicy<>>
struct BlockFmhaPipelineQRKSVSTdmKVShared : BlockFmhaPipelineQRKSVSTdmPingPong<Problem_, Policy_>
{
    using PingPong = BlockFmhaPipelineQRKSVSTdmPingPong<Problem_, Policy_>;
    using Base     = typename PingPong::Base;

    using typename PingPong::BlockFmhaShape;
    using typename PingPong::FmhaMask;
    using typename PingPong::KDataType;
    using typename PingPong::Policy;
    using typename PingPong::Problem;
    using typename PingPong::QDataType;
    using typename PingPong::SaccDataType;
    using typename PingPong::SMPLComputeDataType;
    using typename PingPong::VDataType;

    using PingPong::BiasEnum;
    using PingPong::I0;
    using PingPong::I1;
    using PingPong::kHasLogitsSoftCap;
    using PingPong::kHasSink;
    using PingPong::kHasUnevenSplits;
    using PingPong::kK0;
    using PingPong::kK1;
    using PingPong::kM0;
    using PingPong::kN0;
    using PingPong::kN1;
    using PingPong::kNWarp;
    using PingPong::kPadSeqLenK;
    using PingPong::kQKHeaddim;
    using PingPong::kRowSumOnMatrixUnit;
    using PingPong::kSeparateLseSum;
    using PingPong::kStoreLSE;
    using PingPong::kSubQKHeaddim;

    using PingPong::ReduceRowSync;

    static constexpr const char* name = "qr_tdm_kvshared";

    static_assert(kN1 == kSubQKHeaddim && std::is_same_v<KDataType, VDataType>,
                  "qr_tdm_kvshared: K and V of one head dim and data type");

    template <typename QDramBlockWindowTmp,
              typename KDramBlockWindowTmp,
              typename VDramBlockWindowTmp,
              typename BiasDramBlockWindowTmp,
              typename LSEaccDramBlockWindowTmp,
              typename PositionEncoding>
    CK_TILE_HOST_DEVICE auto operator()(const QDramBlockWindowTmp& q_dram_block_window_tmp,
                                        const KDramBlockWindowTmp& k_dram_block_window_tmp,
                                        const VDramBlockWindowTmp& /* v: the K view */,
                                        const BiasDramBlockWindowTmp& bias_dram_block_window_tmp,
                                        LSEaccDramBlockWindowTmp& lse_acc_dram_window_tmp,
                                        FmhaMask mask,
                                        PositionEncoding position_encoding,
                                        float scale_s,
                                        void* smem_arena,
                                        float sink_v) const
    {
        using Layout = typename Policy::template LdsArenaLayout<Problem>;

        static_assert(
            std::is_same_v<QDataType, remove_cvref_t<typename QDramBlockWindowTmp::DataType>> &&
                std::is_same_v<KDataType, remove_cvref_t<typename KDramBlockWindowTmp::DataType>>,
            "wrong!");
        static_assert(kM0 == QDramBlockWindowTmp{}.get_window_lengths()[I0] &&
                          kSubQKHeaddim == QDramBlockWindowTmp{}.get_window_lengths()[I1] &&
                          kN0 == KDramBlockWindowTmp{}.get_window_lengths()[I0] &&
                          kM0 == BiasDramBlockWindowTmp{}.get_window_lengths()[I0] &&
                          kN0 == BiasDramBlockWindowTmp{}.get_window_lengths()[I1],
                      "wrong!");

        constexpr auto gemm_0 = Policy::template GetQKBlockGemm<Problem>();
        constexpr auto gemm_1 = Policy::template GetPVBlockGemm<Problem>();

        using SaccBlockTileType = decltype(gemm_0.MakeCBlockTile());
        auto s_acc              = SaccBlockTileType{};

        const auto f_max = [](auto e0, auto e1) { return max(e0, e1); };

        using OaccBlockTileType = decltype(gemm_1.MakeCBlockTile());
        auto o_acc              = OaccBlockTileType{};

        using SBlockTileType  = decltype(cast_tile<SMPLComputeDataType>(o_acc));
        using MLBlockTileType = decltype(block_tile_reduce<SMPLComputeDataType>(
            SBlockTileType{}, sequence<1>{}, f_max, SMPLComputeDataType{0}));

        // m, l and o_acc as in qr_tdm_pingpong, with its lazy rebasing to m_base
        auto m                      = MLBlockTileType{};
        auto l                      = MLBlockTileType{};
        [[maybe_unused]] auto l_lse = MLBlockTileType{};
        auto m_base                 = MLBlockTileType{};

        clear_tile(o_acc);
        PingPong::InitRowMaxSum(m, l, sink_v, scale_s);
        m_base = m;
        if constexpr(kSeparateLseSum)
            l_lse = l;

        const auto q_origin = q_dram_block_window_tmp.get_window_origin();

        const auto tile_range_result      = PingPong::GetKTileRange(mask, q_origin);
        const auto sink_seq_end           = tile_range_result.get(number<0>{});
        const auto logical_seqlen_k_start = tile_range_result.get(number<1>{});
        const auto logical_seqlen_k_end   = tile_range_result.get(number<2>{});

        const auto num_sink_loop = integer_divide_ceil(sink_seq_end, kN0);

        if constexpr(FmhaMask::IsMasking || kPadSeqLenK || kHasUnevenSplits)
        {
            const index_t logical_num_total_loop =
                integer_divide_ceil(logical_seqlen_k_end - logical_seqlen_k_start, kN0) +
                num_sink_loop;
            if(logical_num_total_loop <= 0)
            {
                if constexpr(kStoreLSE)
                    PingPong::StoreLseOfEmptyRows(lse_acc_dram_window_tmp, m, sink_v, scale_s);
                return o_acc;
            }
        }

        TDMConfig tdm_config_q = PingPong::template MakeTdmConfig<
            typename Policy::template LdsPaddingConfigQ<Problem>>();
        TDMConfig tdm_config_kv = PingPong::template MakeTdmConfig<
            typename Policy::template LdsPaddingConfigKV<Problem>>();

        // Q tile in LDS
        auto q_dram_window = make_tile_window(
            q_dram_block_window_tmp, Policy::template MakeQDramTileDistribution<Problem>());
        auto q_lds_view = make_tensor_view<address_space_enum::lds>(
            reinterpret_cast<QDataType*>(static_cast<char*>(smem_arena) + Layout::kQOffset),
            Policy::template MakeQLdsBlockDescriptor<Problem>());
        auto q_lds_store_window = make_tile_window(
            q_lds_view, Policy::template MakeQLdsBlockDescriptor<Problem>().get_lengths(), {0, 0});
        auto q_lds_read_window =
            make_tile_window(q_lds_view,
                             Policy::template MakeQLdsBlockDescriptor<Problem>().get_lengths(),
                             {0, 0},
                             Policy::template MakeQRegTileDistribution<Problem>());

        load_tile_tdm(tdm_config_q, q_lds_store_window, q_dram_window);

        // For sink: kv_load_start is 0 when there are sink tokens, otherwise seqlen_k_start.
        const auto kv_load_start =
            (sink_seq_end == 0 && logical_seqlen_k_start > 0) ? logical_seqlen_k_start : 0;
        const index_t physical_seqlen_k_start = logical_seqlen_k_start;
        const index_t physical_seqlen_k_end   = logical_seqlen_k_end;

        const auto bias_origin = bias_dram_block_window_tmp.get_window_origin();
        auto bias_dram_window =
            make_tile_window(bias_dram_block_window_tmp.get_bottom_tensor_view(),
                             bias_dram_block_window_tmp.get_window_lengths(),
                             {bias_origin.at(number<0>{}), kv_load_start},
                             gemm_0.MakeCBlockTile().get_tile_distribution());

        // K/V tiles, whole, in kNumKVBuffers buffers: tile t goes to buffer t % kNumKVBuffers
        constexpr index_t kNumKVBuffers = Policy::kNumKVBuffers;
        auto kv_dram_window =
            make_tile_window(k_dram_block_window_tmp,
                             {kv_load_start, 0},
                             Policy::template MakeKDramTileDistribution<Problem, true>());

        KDataType* const kv_smem_buf0 = reinterpret_cast<KDataType*>(smem_arena);
        const auto kv_smem_buf        = [&](index_t i_buf) {
            return reinterpret_cast<KDataType*>(static_cast<char*>(smem_arena) +
                                                i_buf * Layout::kKVStride);
        };

        // One descriptor for every buffer and both gemms; the buffers only swap the
        // windows' base pointer.
        auto kv_lds_view = make_tensor_view<address_space_enum::lds>(
            kv_smem_buf0, Policy::template MakeKVLdsBlockDescriptor<Problem>());
        auto kv_lds_write_window =
            make_tile_window(kv_lds_view,
                             Policy::template MakeKVLdsBlockDescriptor<Problem>().get_lengths(),
                             {0, 0});
        auto k_lds_read_window =
            make_tile_window(kv_lds_view,
                             make_tuple(number<kN0>{}, number<kK0>{}),
                             {0, 0},
                             Policy::template MakeKRegTileDistribution<Problem>());
        auto v_lds_read_window =
            make_tile_window(kv_lds_view,
                             make_tuple(number<kK1>{}, number<kN1>{}),
                             {0, 0},
                             Policy::template MakeVRegTileDistribution<Problem>());

        // Tile t into buffer i_buf: the window moves from tile t - 1 (the sink->normal jump
        // at the boundary between the sink tiles and the others). Past the end the TDM
        // zero-fills, and nothing reads it.
        const auto load_next_kv_tile = [&](index_t t, index_t i_buf) {
            if constexpr(kHasSink)
            {
                if(t == num_sink_loop)
                    move_tile_window(kv_dram_window, {physical_seqlen_k_start - sink_seq_end, 0});
            }
            move_tile_window(kv_dram_window, {kN0, 0});
            kv_lds_write_window.set_bottom_tensor_view_data_ptr(kv_smem_buf(i_buf));
            load_tile_tdm(tdm_config_kv, kv_lds_write_window, kv_dram_window);
        };

        // tiles 0 .. kNumKVBuffers - 2, issued while Q is still in flight
        load_tile_tdm(tdm_config_kv, kv_lds_write_window, kv_dram_window);
        static_for<1, kNumKVBuffers - 1, 1>{}([&](auto t) { load_next_kv_tile(t, t); });

        // S tile in LDS, only used with the gemm_0 warps on N
        auto s_lds = make_tensor_view<address_space_enum::lds>(
            reinterpret_cast<SaccDataType*>(static_cast<char*>(smem_arena) + Layout::kSOffset),
            Policy::template MakeSLdsBlockDescriptor<Problem>());
        [[maybe_unused]] auto s_write_lds_window = make_tile_window(
            s_lds, Policy::template MakeSLdsBlockDescriptor<Problem>().get_lengths(), {0, 0});
        [[maybe_unused]] auto s_read_lds_window =
            make_tile_window(s_lds,
                             Policy::template MakeSLdsBlockDescriptor<Problem>().get_lengths(),
                             {0, 0},
                             Policy::template MakeSRegTileDistribution<Problem>());

        // retire Q only; the K/V tiles, issued after it, stay in flight
        s_wait_tensorcnt_barrier<kNumKVBuffers - 1>();
        auto q_tile = load_tile(q_lds_read_window);

        const index_t num_total_loop =
            integer_divide_ceil(physical_seqlen_k_end - physical_seqlen_k_start, kN0) +
            num_sink_loop;

        index_t i_total_loops      = 0;
        constexpr index_t k0_loops = kQKHeaddim / kK0;
        constexpr index_t k1_loops = kN0 / kK1;

        static_assert(1 <= k0_loops);
        static_assert(1 <= k1_loops);

        // buffer of tile i_total_loops, and of tile i_total_loops + kNumKVBuffers - 1
        index_t i_read_buf  = 0;
        index_t i_write_buf = kNumKVBuffers - 1;
        do
        {
            KDataType* const kv_read_ptr = kv_smem_buf(i_read_buf);

            // One barrier for two hazards: tile i has landed (tiles i + 1 .. stay in
            // flight), and every read of the buffer tile i + kNumKVBuffers - 1 goes to
            // (the last trip's gemms, or on the first trip Q's) is done.
            s_wait_tensorcnt_barrier<kNumKVBuffers - 2, 0>();

            // the next tile, issued on the last trips too
            load_next_kv_tile(i_total_loops + kNumKVBuffers - 1, i_write_buf);
            i_read_buf  = i_read_buf + 1 == kNumKVBuffers ? 0 : i_read_buf + 1;
            i_write_buf = i_write_buf + 1 == kNumKVBuffers ? 0 : i_write_buf + 1;

            // STAGE 1, QK gemm on the K/V tile read row-wise
            clear_tile(s_acc);
            k_lds_read_window.set_bottom_tensor_view_data_ptr(kv_read_ptr);
            auto k_tile = load_tile(k_lds_read_window);

            if constexpr(1 < k0_loops)
            {
                static_for<0, k0_loops - 1, 1>{}([&](auto i_k0) {
                    move_tile_window(k_lds_read_window, {0, kK0});
                    auto k_tile_switch = load_tile(k_lds_read_window);

                    gemm_0(s_acc,
                           get_slice_tile(q_tile,
                                          sequence<0, i_k0 * kK0>{},
                                          sequence<kM0, (i_k0 + 1) * kK0>{}),
                           k_tile);

                    k_tile = k_tile_switch;
                });
                move_tile_window(k_lds_read_window, {0, -kK0 * (k0_loops - 1)});
            }

            gemm_0(s_acc,
                   get_slice_tile(q_tile,
                                  sequence<0, (k0_loops - 1) * kK0>{},
                                  sequence<kM0, k0_loops * kK0>{}),
                   k_tile);

            // two K reads per WMMA (see qr_tdm_pingpong)
            static_for<0, k0_loops, 1>{}([&](auto) {
                __builtin_amdgcn_sched_group_barrier(0x100, 2, 0); // DS read
                __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // WMMA
            });

            const index_t kv_tile_start =
                (num_sink_loop > i_total_loops)
                    ? kv_load_start + i_total_loops * kN0
                    : physical_seqlen_k_start + (i_total_loops - num_sink_loop) * kN0;

            // STAGE 2: scale_s, add bias, mask
            PingPong::ApplyScaleAndBias(
                s_acc, bias_dram_window, q_origin, kv_tile_start, scale_s, position_encoding);
            PingPong::MaskScores(s_acc,
                                 mask,
                                 q_origin,
                                 kv_tile_start,
                                 i_total_loops == num_total_loop - 1,
                                 physical_seqlen_k_end);
            if constexpr(kHasSink)
            {
                if(i_total_loops == num_sink_loop - 1)
                    move_tile_window(bias_dram_window, {0, physical_seqlen_k_start - sink_seq_end});
            }
            move_tile_window(bias_dram_window, {0, kN0});

            // S{j}. With the gemm_0 warps on N each warp holds a column slice, so S goes
            // through LDS into the row-complete layout gemm_1's A operand takes.
            const auto s_new = [&]() {
                if constexpr(kNWarp > 1)
                {
                    store_tile(s_write_lds_window, cast_tile<SMPLComputeDataType>(s_acc));
                    block_sync_lds();
                    return load_tile(s_read_lds_window);
                }
                else
                {
                    return cast_tile<SMPLComputeDataType>(s_acc);
                }
            }();

            auto m_local = block_tile_reduce<SMPLComputeDataType>(
                s_new, sequence<1>{}, f_max, -numeric<SMPLComputeDataType>::infinity());
            ReduceRowSync(m_local, f_max);

            tile_elementwise_inout([](auto& e0, auto e1) { e0 = max(e0, e1); }, m, m_local);

            auto p_compute =
                make_static_distributed_tensor<SMPLComputeDataType>(s_new.get_tile_distribution());

            static const auto get_validated_m = [](SMPLComputeDataType raw_m) {
                if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS ||
                             FmhaMask::IsMasking)
                {
                    return raw_m == -numeric<SMPLComputeDataType>::infinity()
                               ? type_convert<SMPLComputeDataType>(0.f)
                               : raw_m;
                }
                else
                {
                    return raw_m;
                }
            };

            const auto get_row_exp = [&](SMPLComputeDataType raw_m) {
                if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS ||
                             BiasEnum == BlockAttentionBiasEnum::ALIBI || kHasLogitsSoftCap)
                    return get_validated_m(raw_m);
                else
                    return scale_s * get_validated_m(raw_m);
            };

            constexpr auto o_spans = decltype(o_acc)::get_distributed_spans();

            // lazy rebasing, as in qr_tdm_pingpong
            constexpr auto kRebaseThreshold = SMPLComputeDataType{8};
            const auto row_gap              = [&](auto i_idx) {
                if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS ||
                             BiasEnum == BlockAttentionBiasEnum::ALIBI || kHasLogitsSoftCap)
                    return m[i_idx] - m_base[i_idx];
                else
                    return scale_s * (m[i_idx] - m_base[i_idx]);
            };
            bool need_rebase = false;
            sweep_tile_span(o_spans[I0], [&](auto idx0) {
                need_rebase |= row_gap(make_tuple(idx0)) > kRebaseThreshold;
            });
            if(need_rebase)
            {
                sweep_tile_span(o_spans[I0], [&](auto idx0) {
                    constexpr auto i_idx = make_tuple(idx0);
                    const auto gap       = row_gap(i_idx);
                    if(gap > 0)
                    {
                        const auto tmp = exp2(-gap);
                        l(i_idx) *= tmp;
                        if constexpr(kSeparateLseSum)
                            l_lse(i_idx) *= tmp;
                        m_base(i_idx) = m[i_idx];
                        sweep_tile_span(o_spans[I1],
                                        [&](auto idx1) { o_acc(make_tuple(idx0, idx1)) *= tmp; });
                    }
                });
            }

            constexpr auto p_spans = decltype(p_compute)::get_distributed_spans();
            sweep_tile_span(p_spans[I0], [&](auto idx0) {
                constexpr auto i_idx = make_tuple(idx0);
                auto row_max         = get_row_exp(m_base[i_idx]);
                sweep_tile_span(p_spans[I1], [&](auto idx1) {
                    constexpr auto i_j_idx = make_tuple(idx0, idx1);
                    if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS ||
                                 BiasEnum == BlockAttentionBiasEnum::ALIBI || kHasLogitsSoftCap)
                        p_compute(i_j_idx) = exp2(s_new[i_j_idx] - row_max);
                    else
                        p_compute(i_j_idx) = exp2(scale_s * s_new[i_j_idx] - row_max);
                });
            });

            static_assert(BlockFmhaShape::Gemm1BlockWarps::at(I1) == 1 ||
                          remove_cvref_t<decltype(gemm_1)>::MIterPerWarp == 1);
            [[maybe_unused]] int32_t p_scale = 0;
            auto p_tile = Base::template MakePForGemm1<decltype(gemm_1)>(p_compute, p_scale);

            if constexpr(kRowSumOnMatrixUnit)
                PingPong::AccumulateRowSumOnMatrixUnit(l, p_tile);
            if constexpr(!kRowSumOnMatrixUnit || kSeparateLseSum)
            {
                const auto rowsum_p = PingPong::BlockRowSum(p_compute);
                sweep_tile_span(o_spans[I0], [&](auto idx0) {
                    constexpr auto i_idx = make_tuple(idx0);
                    if constexpr(kSeparateLseSum)
                        l_lse(i_idx) += rowsum_p[i_idx];
                    else
                        l(i_idx) += rowsum_p[i_idx];
                });
            }

            // STAGE 3, PV gemm on the same K/V tile, read transposed. It has been in LDS
            // since the loop-head wait.
            v_lds_read_window.set_bottom_tensor_view_data_ptr(kv_read_ptr);
            auto v_tile = load_tile_transpose(v_lds_read_window);

            if constexpr(1 < k1_loops)
            {
                static_for<0, k1_loops - 1, 1>{}([&](auto i_k1) {
                    gemm_1(o_acc,
                           get_slice_tile(p_tile,
                                          sequence<0, i_k1 * kK1>{},
                                          sequence<kM0, (i_k1 + 1) * kK1>{}),
                           v_tile);

                    move_tile_window(v_lds_read_window, {kK1, 0});
                    v_tile = load_tile_transpose(v_lds_read_window);
                });
                move_tile_window(v_lds_read_window, {-kK1 * (k1_loops - 1), 0});
            }

            gemm_1(o_acc,
                   get_slice_tile(p_tile,
                                  sequence<0, (k1_loops - 1) * kK1>{},
                                  sequence<kM0, k1_loops * kK1>{}),
                   v_tile);

            using Gemm1T = remove_cvref_t<decltype(gemm_1)>;
            static_for<0, Gemm1T::MIterPerWarp * Gemm1T::NIterPerWarp * k1_loops, 1>{}([&](auto) {
                __builtin_amdgcn_sched_group_barrier(0x100, 2, 0); // DS read
                __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // WMMA
            });

        } while(++i_total_loops < num_total_loop);

        // retire the tile issued past the last one consumed
        s_wait_tensorcnt<0>();

        if constexpr(kStoreLSE)
        {
            auto m_lse = m_base;
            if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS ||
                         FmhaMask::IsMasking)
            {
                tile_elementwise_inout(
                    [](auto& x) {
                        if(x == -numeric<SMPLComputeDataType>::infinity())
                            x = type_convert<SMPLComputeDataType>(0.f);
                    },
                    m_lse);
            }
            if constexpr(kSeparateLseSum)
                PingPong::StoreLse(lse_acc_dram_window_tmp, m_lse, l_lse, scale_s);
            else
                PingPong::StoreLse(lse_acc_dram_window_tmp, m_lse, l, scale_s);
        }

        PingPong::NormalizeO(o_acc, l);
        return o_acc;
    }

    // The kernel's other LDS argument order (see FmhaFwdKernel).
    template <typename QDramBlockWindowTmp,
              typename KDramBlockWindowTmp,
              typename VDramBlockWindowTmp,
              typename BiasDramBlockWindowTmp,
              typename LSEaccDramBlockWindowTmp,
              typename PositionEncoding>
    CK_TILE_HOST_DEVICE auto operator()(const QDramBlockWindowTmp& q_dram_block_window_tmp,
                                        const KDramBlockWindowTmp& k_dram_block_window_tmp,
                                        const VDramBlockWindowTmp& v_dram_block_window_tmp,
                                        const BiasDramBlockWindowTmp& bias_dram_block_window_tmp,
                                        LSEaccDramBlockWindowTmp& lse_acc_dram_window_tmp,
                                        FmhaMask mask,
                                        PositionEncoding position_encoding,
                                        float scale_s,
                                        float sink_v,
                                        void* smem_arena) const
    {
        return (*this)(q_dram_block_window_tmp,
                       k_dram_block_window_tmp,
                       v_dram_block_window_tmp,
                       bias_dram_block_window_tmp,
                       lse_acc_dram_window_tmp,
                       mask,
                       position_encoding,
                       scale_s,
                       smem_arena,
                       sink_v);
    }
};

} // namespace ck_tile
