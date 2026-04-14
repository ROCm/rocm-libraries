// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"
#include "ck_tile/ops/warp_decode/pipeline/warp_decode_problem.hpp"
#include "ck_tile/ops/warp_decode/pipeline/warp_decode_policy.hpp"

namespace ck_tile {

template <typename Problem_, typename Policy_>
struct WarpDecodeDownReduceKernel
{
    using Problem = remove_cvref_t<Problem_>;
    using Policy  = remove_cvref_t<Policy_>;

    using IntermediateDataType = typename Problem::IntermediateDataType;
    using WDataType            = typename Problem::WDataType;
    using ComputeDataType      = typename Problem::ComputeDataType;
    using YDataType            = typename Problem::YDataType;

    static constexpr index_t kBlockSize = Problem::kBlockSize;

    struct Kargs
    {
        const void* p_intermediate; // [B, TOP_K, INTER]
        const void* p_w_down;       // [E, HIDDEN, INTER]
        const void* p_w_down_scale;
        const int32_t* p_router_ids;// [B, TOP_K]
        const float* p_router_wts;  // [B, TOP_K]
        void* p_y;                  // [B, HIDDEN]

        index_t b;
        index_t hidden;
        index_t inter;
        index_t top_k;
        index_t e;

        index_t stride_intermediate;
        index_t stride_w_down;
        index_t stride_y;
    };

    CK_TILE_HOST static constexpr auto GridSize(const Kargs& hargs)
    {
        return dim3(hargs.b * hargs.hidden);
    }

    CK_TILE_HOST static constexpr auto BlockSize()
    {
        return Problem::kBlockSize;
    }

    CK_TILE_DEVICE static ComputeDataType wavefront_reduce_sum(ComputeDataType val)
    {
        constexpr index_t num_stages = integer_log2_floor(get_warp_size());
        static_for<0, num_stages, 1>{}([&](auto istage) {
            index_t offset = 1 << istage.value;
            index_t src_lane = get_lane_id() ^ offset;
            ComputeDataType remote_val = warp_shuffle(val, src_lane);
            val += remote_val;
        });
        return val;
    }

    template <typename ScaleLayout, typename ScaleDataType>
    CK_TILE_DEVICE static ComputeDataType load_block2d_scale(const void* p_scale, index_t row_idx, index_t k_idx, index_t max_k)
    {
        if constexpr(ScaleLayoutTraits<ScaleLayout>::is_block2d)
        {
            if (!p_scale) return type_convert<ComputeDataType>(1.0f);
            constexpr index_t Block_N = ScaleLayoutTraits<ScaleLayout>::block_n;
            constexpr index_t Block_K = ScaleLayoutTraits<ScaleLayout>::block_k;
            const ScaleDataType* ptr = static_cast<const ScaleDataType*>(p_scale);
            index_t r = row_idx / Block_N;
            index_t c = k_idx / Block_K;
            return type_convert<ComputeDataType>(ptr[r * (max_k / Block_K) + c]);
        }
        else
        {
            return type_convert<ComputeDataType>(1.0f);
        }
    }

    CK_TILE_DEVICE void operator()(Kargs kargs) const
    {
        const index_t block_id = get_block_id();
        const index_t out_j = block_id % kargs.hidden;
        const index_t token_b = block_id / kargs.hidden;

        if(token_b >= kargs.b)
        {
            return;
        }

        const auto intermediate_m_n = make_naive_tensor_view<address_space_enum::global>(
            static_cast<const IntermediateDataType*>(kargs.p_intermediate),
            make_tuple(kargs.b * kargs.top_k, kargs.inter),
            make_tuple(kargs.stride_intermediate, 1),
            number<1>{},
            number<1>{});

        const auto w_down_m_n = make_naive_tensor_view<address_space_enum::global>(
            static_cast<const WDataType*>(kargs.p_w_down),
            make_tuple(kargs.e * kargs.hidden, kargs.inter),
            make_tuple(kargs.stride_w_down, 1),
            number<1>{},
            number<1>{});

        ComputeDataType acc = 0;

        ComputeDataType w_down_scale_val = type_convert<ComputeDataType>(1.0f);
        if constexpr(std::is_same_v<typename Problem::WScaleLayout, WarpDecodeScaleLayout::PerTensor>)
        {
            if (kargs.p_w_down_scale)
                w_down_scale_val = type_convert<ComputeDataType>(*static_cast<const typename Problem::WScaleDataType*>(kargs.p_w_down_scale));
        }

        for(index_t k = 0; k < kargs.top_k; ++k)
        {
            const int32_t e = kargs.p_router_ids[token_b * kargs.top_k + k];
            const float w   = kargs.p_router_wts[token_b * kargs.top_k + k];

            if constexpr(std::is_same_v<typename Problem::WScaleLayout, WarpDecodeScaleLayout::PerToken>)
            {
                if (kargs.p_w_down_scale)
                    w_down_scale_val = type_convert<ComputeDataType>(static_cast<const typename Problem::WScaleDataType*>(kargs.p_w_down_scale)[e * kargs.hidden + out_j]);
            }

            auto intermediate_window = make_tile_window(
                intermediate_m_n, 
                make_tuple(number<1>{}, number<get_warp_size()>{}), 
                {token_b * kargs.top_k + k, 0}, 
                Policy::template MakeTileDistribution<Problem>());

            auto w_down_window = make_tile_window(
                w_down_m_n, 
                make_tuple(number<1>{}, number<get_warp_size()>{}), 
                {e * kargs.hidden + out_j, 0}, 
                Policy::template MakeTileDistribution<Problem>());

            index_t num_iterations = kargs.inter / get_warp_size();

            for(index_t i = 0; i < num_iterations; ++i)
            {
                auto inter_tile   = load_tile(intermediate_window);
                auto w_down_tile  = load_tile(w_down_window);

                index_t k_idx = i * get_warp_size() + get_lane_id();

                ComputeDataType ds = w_down_scale_val;
                if constexpr(ScaleLayoutTraits<typename Problem::WScaleLayout>::is_block2d)
                {
                    ds = load_block2d_scale<typename Problem::WScaleLayout, typename Problem::WScaleDataType>(kargs.p_w_down_scale, e * kargs.hidden + out_j, k_idx, kargs.inter);
                }

                constexpr auto spans = decltype(inter_tile)::get_distributed_spans();
                sweep_tile_span(spans[number<0>{}], [&](auto idx0) {
                    sweep_tile_span(spans[number<1>{}], [&](auto idx1) {
                        constexpr auto idx = make_tuple(idx0, idx1);
                        auto act_val = type_convert<ComputeDataType>(inter_tile[idx]);
                        auto d_val   = type_convert<ComputeDataType>(w_down_tile[idx]);

                        acc += w * act_val * (d_val * ds);
                    });
                });

                move_tile_window(intermediate_window, {0, get_warp_size()});
                move_tile_window(w_down_window, {0, get_warp_size()});
            }
        }

        ComputeDataType result = wavefront_reduce_sum(acc);

        if(get_lane_id() == 0)
        {
            auto y_m_n = make_naive_tensor_view<address_space_enum::global>(
                static_cast<YDataType*>(kargs.p_y),
                make_tuple(kargs.b, kargs.hidden),
                make_tuple(kargs.stride_y, 1),
                number<1>{},
                number<1>{});

            auto y_window = make_tile_window(
                y_m_n, 
                make_tuple(number<1>{}, number<1>{}), 
                {token_b, out_j}, 
                Policy::template MakeOutputScalarDistribution<Problem>());

            auto result_tile = make_static_distributed_tensor<YDataType>(
                Policy::template MakeOutputScalarDistribution<Problem>());
            result_tile.get_thread_buffer()[number<0>{}] = type_convert<YDataType>(result);

            store_tile(y_window, result_tile);
        }
    }
};

} // namespace ck_tile
