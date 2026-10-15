// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_bwd_pipeline_default_policy.hpp"

namespace ck_tile {

template <typename Problem, typename Policy = BlockFmhaBwdPipelineDefaultPolicy>
struct BlockFmhaBwdConvertQGrad
{
    using AccDataType   = remove_cvref_t<typename Problem::AccDataType>;
    using QGradDataType = remove_cvref_t<typename Problem::QGradDataType>;

    static constexpr index_t kM0 = Problem::kM0;

    static constexpr index_t kBlockPerCu = Problem::kBlockPerCu;
    static constexpr index_t kBlockSize  = Problem::kBlockSize;
    static constexpr index_t kQKHeaddim  = Problem::kQKHeaddim;

    static constexpr bool kIsGroupMode     = Problem::kIsGroupMode;
    static constexpr bool kPadSeqLenQ      = Problem::kPadSeqLenQ;
    static constexpr bool kPadHeadDimQ     = Problem::kPadHeadDimQ;
    static constexpr bool kIsDeterministic = Problem::kIsDeterministic;

    static constexpr index_t kAlignmentQGradAcc =
        kPadHeadDimQ ? 1 : Policy::template GetAlignmentPostQGradAcc<Problem>();
    static constexpr index_t kAlignmentQGrad =
        kPadHeadDimQ ? 1 : Policy::template GetAlignmentPostQGrad<Problem>();

    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize() { return 0; }

    // Convert only
    template <typename QGradAccDramBlockWindowTmp, typename QGradDramBlockWindowTmp>
    CK_TILE_HOST_DEVICE void
    operator()(const QGradAccDramBlockWindowTmp& dq_acc_dram_block_window_tmp,
               QGradDramBlockWindowTmp& dq_dram_block_window_tmp) const
    {
        static_assert(
            std::is_same_v<AccDataType,
                           remove_cvref_t<typename QGradAccDramBlockWindowTmp::DataType>> &&
                std::is_same_v<QGradDataType,
                               remove_cvref_t<typename QGradDramBlockWindowTmp::DataType>>,
            "wrong!");

        static_assert(kM0 == QGradDramBlockWindowTmp{}.get_window_lengths()[number<0>{}], "wrong!");

        auto dq_acc_dram_window =
            make_tile_window(dq_acc_dram_block_window_tmp.get_bottom_tensor_view(),
                             dq_acc_dram_block_window_tmp.get_window_lengths(),
                             dq_acc_dram_block_window_tmp.get_window_origin(),
                             Policy::template MakePostQGradDramTileDistribution<Problem>());

        auto dq_acc   = load_tile(dq_acc_dram_window);
        const auto dq = cast_tile<QGradDataType>(dq_acc);

        store_tile(dq_dram_block_window_tmp, dq);
    }

    // Reduce + Convert
    template <typename QGradAccDramBlockWindowTmp, typename QGradDramBlockWindowTmp>
    CK_TILE_HOST_DEVICE void
    operator()(const QGradAccDramBlockWindowTmp& dq_acc_dram_block_window_tmp,
               QGradDramBlockWindowTmp& dq_dram_block_window_tmp,
               index_t nsplits) const
    {
        static_assert(
            std::is_same_v<AccDataType,
                           remove_cvref_t<typename QGradAccDramBlockWindowTmp::DataType>> &&
                std::is_same_v<QGradDataType,
                               remove_cvref_t<typename QGradDramBlockWindowTmp::DataType>>,
            "wrong!");

        static_assert(kM0 == QGradDramBlockWindowTmp{}.get_window_lengths()[number<0>{}], "wrong!");

        auto dq_acc_dram_window =
            make_tile_window(dq_acc_dram_block_window_tmp.get_bottom_tensor_view(),
                             dq_acc_dram_block_window_tmp.get_window_lengths(),
                             dq_acc_dram_block_window_tmp.get_window_origin(),
                             Policy::template MakePostQGradAccDramTileDistribution<Problem>());

        auto dq_acc = decltype(load_tile(dq_acc_dram_window)){};
        clear_tile(dq_acc);

        constexpr auto dq_acc_spans = decltype(dq_acc)::get_distributed_spans();

        auto add_into = [&](auto& dst, auto& src) {
            sweep_tile_span(dq_acc_spans[number<0>{}], [&](auto idx0) {
                sweep_tile_span(dq_acc_spans[number<1>{}], [&](auto idx1) {
                    sweep_tile_span(dq_acc_spans[number<2>{}], [&](auto idx2) {
                        constexpr auto n_i_j_idx = make_tuple(idx0, idx1, idx2);
                        dst(n_i_j_idx) += src(n_i_j_idx);
                    });
                });
            });
        };

#if CK_TILE_FMHA_BWD_DET_FAST
        // Two independent accumulators, so two split loads are in flight instead of one.
        // The single-outstanding-load loop below runs this kernel at ~3.1 TB/s while a
        // one-split pass reaches ~4.3 TB/s: it is latency-bound, not bandwidth-bound.
        //
        // NOTE: this sums slots as (s0+s2+..) + (s1+s3+..) rather than strictly left to
        // right. The order is still fixed and schedule-independent, so the kernel stays
        // deterministic -- but the low bits differ from the single-accumulator version.
        auto acc1       = dq_acc; // both already cleared
        index_t i_split = 0;
        for(; i_split + 2 <= nsplits; i_split += 2)
        {
            auto b0 = load_tile(dq_acc_dram_window);
            move_tile_window(dq_acc_dram_window, {1, 0, 0});
            auto b1 = load_tile(dq_acc_dram_window);
            move_tile_window(dq_acc_dram_window, {1, 0, 0});
            add_into(dq_acc, b0);
            add_into(acc1, b1);
        }
        if(i_split < nsplits) // odd tail
        {
            auto b = load_tile(dq_acc_dram_window);
            move_tile_window(dq_acc_dram_window, {1, 0, 0});
            add_into(dq_acc, b);
        }
        add_into(dq_acc, acc1);
#else
        index_t i_total_loops = 0;
        auto dq_acc_buf       = load_tile(dq_acc_dram_window);
        move_tile_window(dq_acc_dram_window, {1, 0, 0});

        do
        {
            add_into(dq_acc, dq_acc_buf);

            dq_acc_buf = load_tile(dq_acc_dram_window);
            move_tile_window(dq_acc_dram_window, {1, 0, 0});

            i_total_loops += 1;
        } while(i_total_loops < (nsplits - 1));

        add_into(dq_acc, dq_acc_buf);
#endif

        // declare dq
        constexpr auto dq_converted_dstr =
            Policy::template MakePostQGradAccDramTileDistribution<Problem>();
        auto dq_converted = make_static_distributed_tensor<QGradDataType>(dq_converted_dstr);

        sweep_tile_span(dq_acc_spans[number<0>{}], [&](auto idx0) {
            sweep_tile_span(dq_acc_spans[number<1>{}], [&](auto idx1) {
                sweep_tile_span(dq_acc_spans[number<2>{}], [&](auto idx2) {
                    constexpr auto n_i_j_idx = make_tuple(idx0, idx1, idx2);
                    dq_converted(n_i_j_idx)  = type_convert<QGradDataType>(dq_acc[n_i_j_idx]);
                });
            });
        });

        constexpr auto dq_dstr = Policy::template MakePostQGradDramTileDistribution<Problem>();
        auto dq                = make_static_distributed_tensor<QGradDataType>(dq_dstr);
        dq.get_thread_buffer() = dq_converted.get_thread_buffer();

        store_tile(dq_dram_block_window_tmp, dq);
    }
};

} // namespace ck_tile
