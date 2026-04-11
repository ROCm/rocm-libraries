// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"
#include "ck_tile/ops/warp_decode/pipeline/warp_decode_problem.hpp"
#include "ck_tile/ops/warp_decode/pipeline/warp_decode_policy.hpp"

namespace ck_tile {

// Host-side arguments
struct WarpDecodeGateUpHostArgs
{
    const void* p_x;            // [B, HIDDEN]
    const void* p_w_gate;       // [E, INTER, HIDDEN]
    const void* p_w_up;         // [E, INTER, HIDDEN]
    const int32_t* p_router_ids;// [B, TOP_K]
    void* p_intermediate;       // [B, TOP_K, INTER]

    index_t b;
    index_t hidden;
    index_t inter;
    index_t top_k;
    index_t e;
};

template <typename Problem_, typename Policy_>
struct WarpDecodeGateUpKernel
{
    using Problem = remove_cvref_t<Problem_>;
    using Policy  = remove_cvref_t<Policy_>;

    using XDataType            = typename Problem::XDataType;
    using WDataType            = typename Problem::WDataType;
    using ComputeDataType      = typename Problem::ComputeDataType;
    using IntermediateDataType = typename Problem::IntermediateDataType;

    using Kargs = WarpDecodeGateUpHostArgs;

    CK_TILE_HOST static constexpr Kargs MakeKargs(const WarpDecodeGateUpHostArgs& hargs)
    {
        return hargs;
    }

    CK_TILE_HOST static constexpr auto GridSize(const WarpDecodeGateUpHostArgs& hargs)
    {
        // 1 wavefront per output neuron
        return dim3(hargs.b * hargs.top_k * hargs.inter);
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
        const index_t neuron_j = block_id % kargs.inter;
        const index_t block_div_inter = block_id / kargs.inter;
        const index_t expert_k = block_div_inter % kargs.top_k;
        const index_t token_b  = block_div_inter / kargs.top_k;

        if(token_b >= kargs.b)
        {
            return;
        }

        const int32_t e = kargs.p_router_ids[token_b * kargs.top_k + expert_k];

        // Create tensor views
        const auto x_m_n = make_naive_tensor_view<address_space_enum::global>(
            static_cast<const XDataType*>(kargs.p_x),
            make_tuple(kargs.b, kargs.hidden),
            make_tuple(kargs.hidden, 1),
            number<1>{},
            number<1>{});

        const auto w_gate_m_n = make_naive_tensor_view<address_space_enum::global>(
            static_cast<const WDataType*>(kargs.p_w_gate),
            make_tuple(kargs.e * kargs.inter, kargs.hidden),
            make_tuple(kargs.hidden, 1),
            number<1>{},
            number<1>{});

        const auto w_up_m_n = make_naive_tensor_view<address_space_enum::global>(
            static_cast<const WDataType*>(kargs.p_w_up),
            make_tuple(kargs.e * kargs.inter, kargs.hidden),
            make_tuple(kargs.hidden, 1),
            number<1>{},
            number<1>{});

        // Initialize windows
        auto x_window = make_tile_window(
            x_m_n, 
            make_tuple(number<1>{}, number<get_warp_size()>{}), 
            {token_b, 0}, 
            Policy::template MakeTileDistribution<Problem>());

        auto w_gate_window = make_tile_window(
            w_gate_m_n, 
            make_tuple(number<1>{}, number<get_warp_size()>{}), 
            {e * kargs.inter + neuron_j, 0}, 
            Policy::template MakeTileDistribution<Problem>());

        auto w_up_window = make_tile_window(
            w_up_m_n, 
            make_tuple(number<1>{}, number<get_warp_size()>{}), 
            {e * kargs.inter + neuron_j, 0}, 
            Policy::template MakeTileDistribution<Problem>());

        index_t num_iterations = kargs.hidden / get_warp_size();

        ComputeDataType gate_acc = 0;
        ComputeDataType up_acc = 0;

        for(index_t i = 0; i < num_iterations; ++i)
        {
            auto x_tile      = load_tile(x_window);
            auto w_gate_tile = load_tile(w_gate_window);
            auto w_up_tile   = load_tile(w_up_window);

            // Inline dequantize & accumulate
            constexpr auto spans = decltype(x_tile)::get_distributed_spans();
            sweep_tile_span(spans[number<0>{}], [&](auto idx0) {
                sweep_tile_span(spans[number<1>{}], [&](auto idx1) {
                    constexpr auto idx = make_tuple(idx0, idx1);
                    auto x_val = type_convert<ComputeDataType>(x_tile[idx]);
                    auto g_val = type_convert<ComputeDataType>(w_gate_tile[idx]);
                    auto u_val = type_convert<ComputeDataType>(w_up_tile[idx]);

                    gate_acc += x_val * g_val;
                    up_acc   += x_val * u_val;
                });
            });

            move_tile_window(x_window, {0, get_warp_size()});
            move_tile_window(w_gate_window, {0, get_warp_size()});
            move_tile_window(w_up_window, {0, get_warp_size()});
        }

        gate_acc = wavefront_reduce_sum(gate_acc);
        up_acc   = wavefront_reduce_sum(up_acc);

        if(get_lane_id() == 0)
        {
            // Apply SwiGLU: SiLU(gate) * up
            ComputeDataType silu_gate = gate_acc / (type_convert<ComputeDataType>(1.0f) + math::exp(-gate_acc));
            ComputeDataType result = silu_gate * up_acc;

            auto* out_ptr = static_cast<IntermediateDataType*>(kargs.p_intermediate);
            index_t out_idx = token_b * (kargs.top_k * kargs.inter) + expert_k * kargs.inter + neuron_j;
            out_ptr[out_idx] = type_convert<IntermediateDataType>(result);
        }
    }
};

} // namespace ck_tile
