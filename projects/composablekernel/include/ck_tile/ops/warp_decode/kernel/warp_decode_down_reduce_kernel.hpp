// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"
#include "ck_tile/ops/warp_decode/pipeline/warp_decode_problem.hpp"
#include "ck_tile/ops/warp_decode/pipeline/warp_decode_policy.hpp"

namespace ck_tile {

// Host-side arguments
struct WarpDecodeDownReduceHostArgs
{
    const void* p_intermediate; // [B, TOP_K, INTER]
    const void* p_w_down;       // [E, HIDDEN, INTER]
    const int32_t* p_router_ids;// [B, TOP_K]
    const float* p_router_wts;  // [B, TOP_K]
    void* p_y;                  // [B, HIDDEN]

    index_t b;
    index_t hidden;
    index_t inter;
    index_t top_k;
    index_t e;
};

template <typename Problem_, typename Policy_>
struct WarpDecodeDownReduceKernel
{
    using Problem = remove_cvref_t<Problem_>;
    using Policy  = remove_cvref_t<Policy_>;

    using IntermediateDataType = typename Problem::IntermediateDataType;
    using WDataType            = typename Problem::WDataType;
    using ComputeDataType      = typename Problem::ComputeDataType;
    using YDataType            = typename Problem::YDataType;

    using Kargs = WarpDecodeDownReduceHostArgs;

    CK_TILE_HOST static constexpr Kargs MakeKargs(const WarpDecodeDownReduceHostArgs& hargs)
    {
        return hargs;
    }

    CK_TILE_HOST static constexpr auto GridSize(const WarpDecodeDownReduceHostArgs& hargs)
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
            make_tuple(kargs.inter, 1),
            number<1>{},
            number<1>{});

        const auto w_down_m_n = make_naive_tensor_view<address_space_enum::global>(
            static_cast<const WDataType*>(kargs.p_w_down),
            make_tuple(kargs.e * kargs.hidden, kargs.inter),
            make_tuple(kargs.inter, 1),
            number<1>{},
            number<1>{});

        ComputeDataType acc = 0;

        for(index_t k = 0; k < kargs.top_k; ++k)
        {
            const int32_t e = kargs.p_router_ids[token_b * kargs.top_k + k];
            const float w   = kargs.p_router_wts[token_b * kargs.top_k + k];

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

                constexpr auto spans = decltype(inter_tile)::get_distributed_spans();
                sweep_tile_span(spans[number<0>{}], [&](auto idx0) {
                    sweep_tile_span(spans[number<1>{}], [&](auto idx1) {
                        constexpr auto idx = make_tuple(idx0, idx1);
                        auto act_val = type_convert<ComputeDataType>(inter_tile[idx]);
                        auto d_val   = type_convert<ComputeDataType>(w_down_tile[idx]);

                        acc += w * act_val * d_val;
                    });
                });

                move_tile_window(intermediate_window, {0, get_warp_size()});
                move_tile_window(w_down_window, {0, get_warp_size()});
            }
        }

        ComputeDataType result = wavefront_reduce_sum(acc);

        if(get_lane_id() == 0)
        {
            auto* y_ptr = static_cast<YDataType*>(kargs.p_y);
            index_t y_idx = token_b * kargs.hidden + out_j;
            y_ptr[y_idx] = type_convert<YDataType>(result);
        }
    }
};

} // namespace ck_tile
