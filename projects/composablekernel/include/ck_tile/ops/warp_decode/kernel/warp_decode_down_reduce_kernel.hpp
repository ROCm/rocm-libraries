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

    CK_TILE_HOST static bool IsSupportedArgument(const Kargs& kargs)
    {
        constexpr index_t kVector = Problem::kVector;
        constexpr index_t kTileN  = get_warp_size() * kVector;

        using WScaleLayout = typename Problem::WScaleLayout;

        const auto fail = [](const char* msg) {
            if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
            {
                CK_TILE_ERROR(msg);
            }
            return false;
        };

        if(kargs.p_intermediate == nullptr || kargs.p_w_down == nullptr ||
           kargs.p_router_ids == nullptr || kargs.p_router_wts == nullptr || kargs.p_y == nullptr)
        {
            return fail("WarpDecodeDownReduceKernel requires non-null tensor pointers.");
        }

        if(kargs.b <= 0 || kargs.hidden <= 0 || kargs.inter <= 0 || kargs.top_k <= 0 || kargs.e <= 0)
        {
            return fail("WarpDecodeDownReduceKernel requires positive tensor dimensions.");
        }

        if(kargs.stride_intermediate < kargs.inter || kargs.stride_w_down < kargs.inter ||
           kargs.stride_y < kargs.hidden)
        {
            return fail("WarpDecodeDownReduceKernel received an invalid row stride.");
        }

        if(kargs.inter % kTileN != 0)
        {
            return fail("WarpDecodeDownReduceKernel requires inter to be divisible by warp_size * kVector.");
        }

        if constexpr(ScaleLayoutTraits<WScaleLayout>::is_block2d)
        {
            constexpr index_t Block_N = ScaleLayoutTraits<WScaleLayout>::block_n;
            constexpr index_t Block_K = ScaleLayoutTraits<WScaleLayout>::block_k;

            if(kargs.inter % Block_K != 0 || (kargs.e * kargs.hidden) % Block_N != 0)
            {
                return fail("WarpDecodeDownReduceKernel weight Block2D scales require divisible inter and E*hidden dimensions.");
            }
        }

        return true;
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
        constexpr bool is_packed_w   = std::is_same_v<WDataType, pk_fp4_t>;
        constexpr index_t kVector    = Problem::kVector;
        constexpr index_t kTileN     = get_warp_size() * kVector;

        using WScaleLayout   = typename Problem::WScaleLayout;
        using WScaleDataType = typename Problem::WScaleDataType;

        const index_t block_id = get_block_id();
        const index_t out_j    = block_id % kargs.hidden;
        const index_t token_b  = block_id / kargs.hidden;

        if(token_b >= kargs.b)
            return;

        ComputeDataType acc = 0;

        // Loop-invariant per-tensor scale
        ComputeDataType w_down_scale_val = type_convert<ComputeDataType>(1.0f);
        if constexpr(std::is_same_v<WScaleLayout, WarpDecodeScaleLayout::PerTensor>)
        {
            if(kargs.p_w_down_scale)
                w_down_scale_val = type_convert<ComputeDataType>(
                    *static_cast<const WScaleDataType*>(kargs.p_w_down_scale));
        }

        // Intermediate view (always non-packed: float/bf16)
        const auto intermediate_view = make_naive_tensor_view<address_space_enum::global>(
            static_cast<const IntermediateDataType*>(kargs.p_intermediate),
            make_tuple(kargs.b * kargs.top_k, kargs.inter),
            make_tuple(kargs.stride_intermediate, 1),
            number<kVector>{}, number<1>{});

        // Weight view (kVector as guaranteed vector size for correct pk_fp4_t handling)
        const auto w_down_view = make_naive_tensor_view<address_space_enum::global>(
            static_cast<const WDataType*>(kargs.p_w_down),
            make_tuple(kargs.e * kargs.hidden, kargs.inter),
            make_tuple(kargs.stride_w_down, 1),
            number<kVector>{}, number<1>{});

        const index_t num_iterations = kargs.inter / kTileN;

        for(index_t k = 0; k < kargs.top_k; ++k)
        {
            const int32_t e = kargs.p_router_ids[token_b * kargs.top_k + k];
            const float w   = kargs.p_router_wts[token_b * kargs.top_k + k];

            if constexpr(std::is_same_v<WScaleLayout, WarpDecodeScaleLayout::PerToken>)
            {
                if(kargs.p_w_down_scale)
                    w_down_scale_val = type_convert<ComputeDataType>(
                        static_cast<const WScaleDataType*>(kargs.p_w_down_scale)[e * kargs.hidden + out_j]);
            }

            const index_t w_row = e * kargs.hidden + out_j;

            auto intermediate_window = make_tile_window(
                intermediate_view,
                make_tuple(number<1>{}, number<kTileN>{}),
                {token_b * kargs.top_k + k, 0},
                Policy::template MakeTileDistribution<Problem>());

            auto w_down_window = make_tile_window(
                w_down_view,
                make_tuple(number<1>{}, number<kTileN>{}),
                {w_row, 0},
                Policy::template MakeTileDistribution<Problem>());

            for(index_t i = 0; i < num_iterations; ++i)
            {
                auto inter_tile  = load_tile(intermediate_window);
                auto w_down_tile = load_tile(w_down_window);

                const index_t k_base = i * kTileN + get_lane_id() * kVector;

                ComputeDataType ds = w_down_scale_val;
                if constexpr(ScaleLayoutTraits<WScaleLayout>::is_block2d)
                    ds = load_block2d_scale<WScaleLayout, WScaleDataType>(
                        kargs.p_w_down_scale, w_row, k_base, kargs.inter);

                index_t sub = 0;
                constexpr auto spans = decltype(inter_tile)::get_distributed_spans();
                sweep_tile_span(spans[number<1>{}], [&](auto idx1) {
                    constexpr auto idx = make_tuple(make_tuple(), idx1);
                    auto act_val = type_convert<ComputeDataType>(inter_tile[idx]);

                    ComputeDataType d_val;
                    if constexpr(is_packed_w)
                    {
                        d_val = unpack_fp4_nibble(
                            static_cast<uint8_t>(w_down_tile[idx]), sub);
                        sub ^= 1;
                    }
                    else
                    {
                        d_val = type_convert<ComputeDataType>(w_down_tile[idx]);
                    }

                    acc += w * act_val * (d_val * ds);
                });

                move_tile_window(intermediate_window, {0, kTileN});
                move_tile_window(w_down_window, {0, kTileN});
            }
        }

        ComputeDataType result = wavefront_reduce_sum(acc);

        if(get_lane_id() == 0)
        {
            static_cast<YDataType*>(kargs.p_y)[token_b * kargs.stride_y + out_j] =
                type_convert<YDataType>(result);
        }
    }
};

} // namespace ck_tile
