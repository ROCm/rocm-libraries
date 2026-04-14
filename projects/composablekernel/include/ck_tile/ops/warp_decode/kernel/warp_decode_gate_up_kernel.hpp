// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"
#include "ck_tile/ops/warp_decode/pipeline/warp_decode_problem.hpp"
#include "ck_tile/ops/warp_decode/pipeline/warp_decode_policy.hpp"

namespace ck_tile {

template <typename Problem_, typename Policy_>
struct WarpDecodeGateUpKernel
{
    using Problem = remove_cvref_t<Problem_>;
    using Policy  = remove_cvref_t<Policy_>;

    using XDataType            = typename Problem::XDataType;
    using WDataType            = typename Problem::WDataType;
    using ComputeDataType      = typename Problem::ComputeDataType;
    using IntermediateDataType = typename Problem::IntermediateDataType;

    static constexpr index_t kBlockSize = Problem::kBlockSize;

    struct Kargs
    {
        const void* p_x;            // [B, HIDDEN]
        const void* p_x_scale;
        const void* p_w_gate;       // [E, INTER, HIDDEN]
        const void* p_w_gate_scale;
        const void* p_w_up;         // [E, INTER, HIDDEN]
        const void* p_w_up_scale;
        const int32_t* p_router_ids;// [B, TOP_K]
        void* p_intermediate;       // [B, TOP_K, INTER]

        index_t b;
        index_t hidden;
        index_t inter;
        index_t top_k;
        index_t e;

        index_t stride_x;
        index_t stride_w_gate;
        index_t stride_w_up;
        index_t stride_intermediate;
    };

    CK_TILE_HOST static constexpr auto GridSize(const Kargs& hargs)
    {
        // 1 wavefront per output neuron
        return dim3(hargs.b * hargs.top_k * hargs.inter);
    }

    CK_TILE_HOST static constexpr auto BlockSize()
    {
        return Problem::kBlockSize;
    }

    CK_TILE_DEVICE static ComputeDataType unpack_fp4_nibble(uint8_t raw, index_t nibble_idx)
    {
        constexpr float lut[16] = {
            0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f,
            -0.f, -0.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};
        uint8_t nib = nibble_idx ? (raw >> 4) : (raw & 0x0F);
        return static_cast<ComputeDataType>(lut[nib]);
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
        const index_t neuron_j = block_id % kargs.inter;
        const index_t block_div_inter = block_id / kargs.inter;
        const index_t expert_k = block_div_inter % kargs.top_k;
        const index_t token_b  = block_div_inter / kargs.top_k;

        if(token_b >= kargs.b)
        {
            return;
        }

        const int32_t e = kargs.p_router_ids[token_b * kargs.top_k + expert_k];

        // Scale initialization (shared by both paths)
        ComputeDataType x_scale_val = type_convert<ComputeDataType>(1.0f);
        if constexpr(std::is_same_v<typename Problem::XScaleLayout, WarpDecodeScaleLayout::PerTensor>)
        {
            if (kargs.p_x_scale)
                x_scale_val = type_convert<ComputeDataType>(*static_cast<const typename Problem::XScaleDataType*>(kargs.p_x_scale));
        }
        else if constexpr(std::is_same_v<typename Problem::XScaleLayout, WarpDecodeScaleLayout::PerToken>)
        {
            if (kargs.p_x_scale)
                x_scale_val = type_convert<ComputeDataType>(static_cast<const typename Problem::XScaleDataType*>(kargs.p_x_scale)[token_b]);
        }

        ComputeDataType w_gate_scale_val = type_convert<ComputeDataType>(1.0f);
        if constexpr(std::is_same_v<typename Problem::WScaleLayout, WarpDecodeScaleLayout::PerTensor>)
        {
            if (kargs.p_w_gate_scale)
                w_gate_scale_val = type_convert<ComputeDataType>(*static_cast<const typename Problem::WScaleDataType*>(kargs.p_w_gate_scale));
        }
        else if constexpr(std::is_same_v<typename Problem::WScaleLayout, WarpDecodeScaleLayout::PerToken>)
        {
            if (kargs.p_w_gate_scale)
                w_gate_scale_val = type_convert<ComputeDataType>(static_cast<const typename Problem::WScaleDataType*>(kargs.p_w_gate_scale)[e * kargs.inter + neuron_j]);
        }

        ComputeDataType w_up_scale_val = type_convert<ComputeDataType>(1.0f);
        if constexpr(std::is_same_v<typename Problem::WScaleLayout, WarpDecodeScaleLayout::PerTensor>)
        {
            if (kargs.p_w_up_scale)
                w_up_scale_val = type_convert<ComputeDataType>(*static_cast<const typename Problem::WScaleDataType*>(kargs.p_w_up_scale));
        }
        else if constexpr(std::is_same_v<typename Problem::WScaleLayout, WarpDecodeScaleLayout::PerToken>)
        {
            if (kargs.p_w_up_scale)
                w_up_scale_val = type_convert<ComputeDataType>(static_cast<const typename Problem::WScaleDataType*>(kargs.p_w_up_scale)[e * kargs.inter + neuron_j]);
        }

        constexpr bool is_packed_w = std::is_same_v<WDataType, pk_fp4_t>;
        const index_t w_row = e * kargs.inter + neuron_j;

        ComputeDataType gate_acc = 0;
        ComputeDataType up_acc = 0;

        {
            // V2 activation: 2 scalar values per thread, [1, 2*WARP_SIZE] tile.
            const auto x_m_n = make_naive_tensor_view<address_space_enum::global>(
                static_cast<const XDataType*>(kargs.p_x),
                make_tuple(kargs.b, kargs.hidden),
                make_tuple(kargs.stride_x, 1),
                number<1>{},
                number<1>{});

            auto x_window = make_tile_window(
                x_m_n,
                make_tuple(number<1>{}, number<get_warp_size() * 2>{}),
                {token_b, 0},
                Policy::template MakeTileDistributionV2<Problem>());

            if constexpr(is_packed_w)
            {
                static_assert(sizeof(WDataType) == 1, "pk_fp4_t must be exactly 1 byte");

                // pk_fp4_t weights: raw pointer access (tile windows can't address sub-byte).
                // stride_w_gate is in pk_fp4_t units (HIDDEN/2 bytes per row).
                const auto* g_ptr = static_cast<const WDataType*>(kargs.p_w_gate) + w_row * kargs.stride_w_gate;
                const auto* u_ptr = static_cast<const WDataType*>(kargs.p_w_up)   + w_row * kargs.stride_w_up;

                const index_t num_iterations = kargs.hidden / (get_warp_size() * 2);

                for(index_t i = 0; i < num_iterations; ++i)
                {
                    auto x_tile = load_tile(x_window);

                    const index_t byte_idx = i * get_warp_size() + get_lane_id();
                    const index_t k_base = byte_idx * 2;

                    uint8_t g_raw = g_ptr[byte_idx].data;
                    uint8_t u_raw = u_ptr[byte_idx].data;

                    ComputeDataType g0 = unpack_fp4_nibble(g_raw, 0);
                    ComputeDataType g1 = unpack_fp4_nibble(g_raw, 1);
                    ComputeDataType u0 = unpack_fp4_nibble(u_raw, 0);
                    ComputeDataType u1 = unpack_fp4_nibble(u_raw, 1);

                    ComputeDataType xs = x_scale_val;
                    if constexpr(ScaleLayoutTraits<typename Problem::XScaleLayout>::is_block2d)
                        xs = load_block2d_scale<typename Problem::XScaleLayout, typename Problem::XScaleDataType>(kargs.p_x_scale, token_b, k_base, kargs.hidden);

                    ComputeDataType gs = w_gate_scale_val;
                    if constexpr(ScaleLayoutTraits<typename Problem::WScaleLayout>::is_block2d)
                        gs = load_block2d_scale<typename Problem::WScaleLayout, typename Problem::WScaleDataType>(kargs.p_w_gate_scale, w_row, k_base, kargs.hidden);

                    ComputeDataType us = w_up_scale_val;
                    if constexpr(ScaleLayoutTraits<typename Problem::WScaleLayout>::is_block2d)
                        us = load_block2d_scale<typename Problem::WScaleLayout, typename Problem::WScaleDataType>(kargs.p_w_up_scale, w_row, k_base, kargs.hidden);

                    index_t sub = 0;
                    constexpr auto x_spans = decltype(x_tile)::get_distributed_spans();
                    sweep_tile_span(x_spans[number<1>{}], [&](auto xidx1) {
                        constexpr auto xidx = make_tuple(make_tuple(), xidx1);
                        auto x_val = type_convert<ComputeDataType>(x_tile[xidx]);
                        ComputeDataType gw = (sub == 0) ? g0 : g1;
                        ComputeDataType uw = (sub == 0) ? u0 : u1;
                        gate_acc += (x_val * xs) * (gw * gs);
                        up_acc   += (x_val * xs) * (uw * us);
                        sub++;
                    });

                    move_tile_window(x_window, {0, get_warp_size() * 2});
                }
            }
            else
            {
                // Non-packed weights: V2 tile windows work normally.
                const auto w_gate_m_n = make_naive_tensor_view<address_space_enum::global>(
                    static_cast<const WDataType*>(kargs.p_w_gate),
                    make_tuple(kargs.e * kargs.inter, kargs.hidden),
                    make_tuple(kargs.stride_w_gate, 1),
                    number<1>{},
                    number<1>{});

                const auto w_up_m_n = make_naive_tensor_view<address_space_enum::global>(
                    static_cast<const WDataType*>(kargs.p_w_up),
                    make_tuple(kargs.e * kargs.inter, kargs.hidden),
                    make_tuple(kargs.stride_w_up, 1),
                    number<1>{},
                    number<1>{});

                auto w_gate_window = make_tile_window(
                    w_gate_m_n,
                    make_tuple(number<1>{}, number<get_warp_size() * 2>{}),
                    {w_row, 0},
                    Policy::template MakeTileDistributionV2<Problem>());

                auto w_up_window = make_tile_window(
                    w_up_m_n,
                    make_tuple(number<1>{}, number<get_warp_size() * 2>{}),
                    {w_row, 0},
                    Policy::template MakeTileDistributionV2<Problem>());

                const index_t num_iterations = kargs.hidden / (get_warp_size() * 2);

                for(index_t i = 0; i < num_iterations; ++i)
                {
                    auto x_tile      = load_tile(x_window);
                    auto w_gate_tile = load_tile(w_gate_window);
                    auto w_up_tile   = load_tile(w_up_window);

                    const index_t k_base = i * get_warp_size() * 2 + get_lane_id() * 2;

                    ComputeDataType xs = x_scale_val;
                    if constexpr(ScaleLayoutTraits<typename Problem::XScaleLayout>::is_block2d)
                        xs = load_block2d_scale<typename Problem::XScaleLayout, typename Problem::XScaleDataType>(kargs.p_x_scale, token_b, k_base, kargs.hidden);

                    ComputeDataType gs = w_gate_scale_val;
                    if constexpr(ScaleLayoutTraits<typename Problem::WScaleLayout>::is_block2d)
                        gs = load_block2d_scale<typename Problem::WScaleLayout, typename Problem::WScaleDataType>(kargs.p_w_gate_scale, w_row, k_base, kargs.hidden);

                    ComputeDataType us = w_up_scale_val;
                    if constexpr(ScaleLayoutTraits<typename Problem::WScaleLayout>::is_block2d)
                        us = load_block2d_scale<typename Problem::WScaleLayout, typename Problem::WScaleDataType>(kargs.p_w_up_scale, w_row, k_base, kargs.hidden);

                    constexpr auto spans = decltype(x_tile)::get_distributed_spans();
                    sweep_tile_span(spans[number<1>{}], [&](auto idx1) {
                        constexpr auto idx = make_tuple(make_tuple(), idx1);
                        auto x_val = type_convert<ComputeDataType>(x_tile[idx]);
                        auto g_val = type_convert<ComputeDataType>(w_gate_tile[idx]);
                        auto u_val = type_convert<ComputeDataType>(w_up_tile[idx]);

                        gate_acc += (x_val * xs) * (g_val * gs);
                        up_acc   += (x_val * xs) * (u_val * us);
                    });

                    move_tile_window(x_window, {0, get_warp_size() * 2});
                    move_tile_window(w_gate_window, {0, get_warp_size() * 2});
                    move_tile_window(w_up_window, {0, get_warp_size() * 2});
                }
            }
        }

        gate_acc = wavefront_reduce_sum(gate_acc);
        up_acc   = wavefront_reduce_sum(up_acc);

        if(get_lane_id() == 0)
        {
            typename Problem::Activation activation_func;
            ComputeDataType silu_gate;
            activation_func(silu_gate, gate_acc);
            ComputeDataType result = silu_gate * up_acc;

            static_cast<IntermediateDataType*>(kargs.p_intermediate)[(token_b * kargs.top_k + expert_k) * kargs.stride_intermediate + neuron_j] = type_convert<IntermediateDataType>(result);
        }
    }
};

} // namespace ck_tile
