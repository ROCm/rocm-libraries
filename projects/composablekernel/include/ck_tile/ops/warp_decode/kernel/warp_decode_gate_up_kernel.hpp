// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"
#include "ck_tile/ops/warp_decode/kernel/warp_decode_numeric.hpp"
#include "ck_tile/ops/warp_decode/pipeline/warp_decode_problem.hpp"
#include "ck_tile/ops/warp_decode/pipeline/warp_decode_policy.hpp"

namespace ck_tile {

template <typename Problem_, typename Policy_>
struct WarpDecodeGateUpKernel : public WarpDecodeNumeric
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
        constexpr index_t kNPerWarp = Problem::kNPerWarp;
        return dim3(hargs.b * hargs.top_k * integer_divide_ceil(hargs.inter, kNPerWarp));
    }

    CK_TILE_HOST static constexpr auto BlockSize()
    {
        return Problem::kBlockSize;
    }

    CK_TILE_HOST static bool IsSupportedArgument(const Kargs& kargs)
    {
        constexpr index_t kVector = Problem::kVector;
        constexpr index_t kTileN  = get_warp_size() * kVector;

        using XScaleLayout = typename Problem::XScaleLayout;
        using WScaleLayout = typename Problem::WScaleLayout;

        const auto fail = [](const char* msg) {
            if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
            {
                CK_TILE_ERROR(msg);
            }
            return false;
        };

        if(kargs.p_x == nullptr || kargs.p_w_gate == nullptr || kargs.p_w_up == nullptr ||
           kargs.p_router_ids == nullptr || kargs.p_intermediate == nullptr)
        {
            return fail("WarpDecodeGateUpKernel requires non-null tensor pointers.");
        }

        if(kargs.b <= 0 || kargs.hidden <= 0 || kargs.inter <= 0 || kargs.top_k <= 0 || kargs.e <= 0)
        {
            return fail("WarpDecodeGateUpKernel requires positive tensor dimensions.");
        }

        // Packed FP4 stores two values per pk_fp4_t, so a weight row of `hidden`
        // fp4 values spans hidden/2 bytes (mirrors WarpDecodeDownReduceKernel).
        const index_t min_w_stride =
            std::is_same_v<WDataType, pk_fp4_t> ? (kargs.hidden / 2) : kargs.hidden;
        if(kargs.stride_x < kargs.hidden || kargs.stride_w_gate < min_w_stride ||
           kargs.stride_w_up < min_w_stride || kargs.stride_intermediate < kargs.inter)
        {
            return fail("WarpDecodeGateUpKernel received an invalid row stride.");
        }

        if(kargs.hidden % kTileN != 0)
        {
            return fail("WarpDecodeGateUpKernel requires hidden to be divisible by warp_size * kVector.");
        }
        if constexpr(Problem::kNPerWarp != 1)
        {
            if constexpr(Problem::kNPerWarp != 2 || !Problem::kUseDot2)
            {
                return fail("WarpDecodeGateUpKernel NPerWarp prototype requires NPerWarp=2 and dot2.");
            }
            if(kargs.inter % Problem::kNPerWarp != 0)
            {
                return fail("WarpDecodeGateUpKernel NPerWarp prototype requires inter to be divisible by NPerWarp.");
            }
        }

        if constexpr(Problem::kUseDot2)
        {
            if constexpr(std::is_same_v<WDataType, pk_fp4_t> || !std::is_same_v<ComputeDataType, float> ||
                         kVector % 2 != 0)
            {
                return fail("WarpDecodeGateUpKernel dot2 path requires unpacked weights, FP32 accumulation, and even kVector.");
            }
        }
        if constexpr(Problem::kUsePackedFp32)
        {
            if constexpr(!std::is_same_v<WDataType, fp8_t> ||
                         !(std::is_same_v<XDataType, fp8_t> || std::is_same_v<XDataType, bf16_t>) ||
                         !std::is_same_v<ComputeDataType, float> || kVector % 2 != 0)
            {
                return fail("WarpDecodeGateUpKernel packed-FP32 path requires FP8 weights, FP8/BF16 activations, FP32 accumulation, and even kVector.");
            }
        }

        if constexpr(ScaleLayoutTraits<XScaleLayout>::is_block2d)
        {
            constexpr index_t Block_N = ScaleLayoutTraits<XScaleLayout>::block_n;
            constexpr index_t Block_K = ScaleLayoutTraits<XScaleLayout>::block_k;

            if(kargs.b % Block_N != 0 || kargs.hidden % Block_K != 0)
            {
                return fail("WarpDecodeGateUpKernel x Block2D scales require divisible B and hidden dimensions.");
            }
        }

        if constexpr(ScaleLayoutTraits<WScaleLayout>::is_block2d)
        {
            constexpr index_t Block_N = ScaleLayoutTraits<WScaleLayout>::block_n;
            constexpr index_t Block_K = ScaleLayoutTraits<WScaleLayout>::block_k;

            if(kargs.hidden % Block_K != 0 || (kargs.e * kargs.inter) % Block_N != 0)
            {
                return fail("WarpDecodeGateUpKernel weight Block2D scales require divisible hidden and E*inter dimensions.");
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

        using XScaleLayout = typename Problem::XScaleLayout;
        using WScaleLayout = typename Problem::WScaleLayout;
        using XScaleDataType = typename Problem::XScaleDataType;
        using WScaleDataType = typename Problem::WScaleDataType;

        if constexpr(Problem::kNPerWarp == 2)
        {
            static_assert(Problem::kWarpsPerBlock == 1,
                          "WarpDecodeGateUpKernel NPerWarp=2 currently expects one warp per block.");
            static_assert(Problem::kUseDot2,
                          "WarpDecodeGateUpKernel NPerWarp=2 currently expects the dot2 path.");
            static_assert(!is_packed_w,
                          "WarpDecodeGateUpKernel NPerWarp=2 does not support packed FP4 weights.");

            constexpr index_t kNPerWarp = Problem::kNPerWarp;
            const index_t block_id = get_block_id();
            const index_t inter_block = block_id % integer_divide_ceil(kargs.inter, kNPerWarp);
            const index_t block_div_inter = block_id / integer_divide_ceil(kargs.inter, kNPerWarp);
            const index_t expert_k = block_div_inter % kargs.top_k;
            const index_t token_b  = block_div_inter / kargs.top_k;
            const index_t neuron_j0 = inter_block * kNPerWarp;
            const index_t neuron_j1 = neuron_j0 + 1;

            if(token_b >= kargs.b || neuron_j1 >= kargs.inter)
                return;

            const int32_t e = kargs.p_router_ids[token_b * kargs.top_k + expert_k];
            const index_t w_row0 = e * kargs.inter + neuron_j0;
            const index_t w_row1 = w_row0 + 1;
            constexpr index_t kMaxScaleBlocks = 128;

            __shared__ ComputeDataType x_scale_lds[kMaxScaleBlocks];
            __shared__ ComputeDataType w_gate_scale_lds[kNPerWarp][kMaxScaleBlocks];
            __shared__ ComputeDataType w_up_scale_lds[kNPerWarp][kMaxScaleBlocks];

            ComputeDataType x_scale_val = type_convert<ComputeDataType>(1.0f);
            if constexpr(std::is_same_v<XScaleLayout, WarpDecodeScaleLayout::PerTensor>)
            {
                if(kargs.p_x_scale)
                    x_scale_val = type_convert<ComputeDataType>(
                        *static_cast<const XScaleDataType*>(kargs.p_x_scale));
            }
            else if constexpr(std::is_same_v<XScaleLayout, WarpDecodeScaleLayout::PerToken>)
            {
                if(kargs.p_x_scale)
                    x_scale_val = type_convert<ComputeDataType>(
                        static_cast<const XScaleDataType*>(kargs.p_x_scale)[token_b]);
            }

            ComputeDataType w_gate_scale_val0 = type_convert<ComputeDataType>(1.0f);
            ComputeDataType w_gate_scale_val1 = type_convert<ComputeDataType>(1.0f);
            ComputeDataType w_up_scale_val0   = type_convert<ComputeDataType>(1.0f);
            ComputeDataType w_up_scale_val1   = type_convert<ComputeDataType>(1.0f);
            if constexpr(std::is_same_v<WScaleLayout, WarpDecodeScaleLayout::PerTensor>)
            {
                if(kargs.p_w_gate_scale)
                {
                    w_gate_scale_val0 = type_convert<ComputeDataType>(
                        *static_cast<const WScaleDataType*>(kargs.p_w_gate_scale));
                    w_gate_scale_val1 = w_gate_scale_val0;
                }
                if(kargs.p_w_up_scale)
                {
                    w_up_scale_val0 = type_convert<ComputeDataType>(
                        *static_cast<const WScaleDataType*>(kargs.p_w_up_scale));
                    w_up_scale_val1 = w_up_scale_val0;
                }
            }
            else if constexpr(std::is_same_v<WScaleLayout, WarpDecodeScaleLayout::PerToken>)
            {
                if(kargs.p_w_gate_scale)
                {
                    const auto* ptr = static_cast<const WScaleDataType*>(kargs.p_w_gate_scale);
                    w_gate_scale_val0 = type_convert<ComputeDataType>(ptr[w_row0]);
                    w_gate_scale_val1 = type_convert<ComputeDataType>(ptr[w_row1]);
                }
                if(kargs.p_w_up_scale)
                {
                    const auto* ptr = static_cast<const WScaleDataType*>(kargs.p_w_up_scale);
                    w_up_scale_val0 = type_convert<ComputeDataType>(ptr[w_row0]);
                    w_up_scale_val1 = type_convert<ComputeDataType>(ptr[w_row1]);
                }
            }

            bool use_x_scale_lds = false;
            if constexpr(ScaleLayoutTraits<XScaleLayout>::is_block2d)
            {
                constexpr index_t Block_N = ScaleLayoutTraits<XScaleLayout>::block_n;
                constexpr index_t Block_K = ScaleLayoutTraits<XScaleLayout>::block_k;
                const index_t num_scale_blocks = kargs.hidden / Block_K;
                use_x_scale_lds = kargs.p_x_scale != nullptr && num_scale_blocks <= kMaxScaleBlocks;
                if(use_x_scale_lds)
                {
                    const auto* ptr = static_cast<const XScaleDataType*>(kargs.p_x_scale);
                    const index_t scale_row = token_b / Block_N;
                    for(index_t c = get_thread_id(); c < num_scale_blocks; c += kBlockSize)
                    {
                        x_scale_lds[c] = type_convert<ComputeDataType>(
                            ptr[scale_row * num_scale_blocks + c]);
                    }
                }
            }

            bool use_w_scale_lds = false;
            if constexpr(ScaleLayoutTraits<WScaleLayout>::is_block2d)
            {
                constexpr index_t Block_N = ScaleLayoutTraits<WScaleLayout>::block_n;
                constexpr index_t Block_K = ScaleLayoutTraits<WScaleLayout>::block_k;
                const index_t num_scale_blocks = kargs.hidden / Block_K;
                use_w_scale_lds = kargs.p_w_gate_scale != nullptr && kargs.p_w_up_scale != nullptr &&
                                  num_scale_blocks <= kMaxScaleBlocks;
                if(use_w_scale_lds)
                {
                    const auto* gate_ptr = static_cast<const WScaleDataType*>(kargs.p_w_gate_scale);
                    const auto* up_ptr   = static_cast<const WScaleDataType*>(kargs.p_w_up_scale);
                    const index_t scale_row0 = w_row0 / Block_N;
                    const index_t scale_row1 = w_row1 / Block_N;
                    for(index_t c = get_thread_id(); c < num_scale_blocks; c += kBlockSize)
                    {
                        w_gate_scale_lds[0][c] = type_convert<ComputeDataType>(
                            gate_ptr[scale_row0 * num_scale_blocks + c]);
                        w_up_scale_lds[0][c] = type_convert<ComputeDataType>(
                            up_ptr[scale_row0 * num_scale_blocks + c]);
                        w_gate_scale_lds[1][c] = type_convert<ComputeDataType>(
                            gate_ptr[scale_row1 * num_scale_blocks + c]);
                        w_up_scale_lds[1][c] = type_convert<ComputeDataType>(
                            up_ptr[scale_row1 * num_scale_blocks + c]);
                    }
                }
            }

            if(use_x_scale_lds || use_w_scale_lds)
            {
                block_sync_lds();
            }

            const auto x_view = make_naive_tensor_view<address_space_enum::global>(
                static_cast<const XDataType*>(kargs.p_x),
                make_tuple(kargs.b, kargs.hidden),
                make_tuple(kargs.stride_x, 1),
                number<kVector>{}, number<1>{});
            const auto w_gate_view = make_naive_tensor_view<address_space_enum::global>(
                static_cast<const WDataType*>(kargs.p_w_gate),
                make_tuple(kargs.e * kargs.inter, kargs.hidden),
                make_tuple(kargs.stride_w_gate, 1),
                number<kVector>{}, number<1>{});
            const auto w_up_view = make_naive_tensor_view<address_space_enum::global>(
                static_cast<const WDataType*>(kargs.p_w_up),
                make_tuple(kargs.e * kargs.inter, kargs.hidden),
                make_tuple(kargs.stride_w_up, 1),
                number<kVector>{}, number<1>{});

            auto x_window = make_tile_window(
                x_view,
                make_tuple(number<1>{}, number<kTileN>{}),
                {token_b, 0},
                Policy::template MakeXBroadcastTileDistribution<Problem>());
            auto w_gate_window0 = make_tile_window(
                w_gate_view,
                make_tuple(number<1>{}, number<kTileN>{}),
                {w_row0, 0},
                Policy::template MakeOutputTileDistribution<Problem>());
            auto w_gate_window1 = make_tile_window(
                w_gate_view,
                make_tuple(number<1>{}, number<kTileN>{}),
                {w_row1, 0},
                Policy::template MakeOutputTileDistribution<Problem>());
            auto w_up_window0 = make_tile_window(
                w_up_view,
                make_tuple(number<1>{}, number<kTileN>{}),
                {w_row0, 0},
                Policy::template MakeOutputTileDistribution<Problem>());
            auto w_up_window1 = make_tile_window(
                w_up_view,
                make_tuple(number<1>{}, number<kTileN>{}),
                {w_row1, 0},
                Policy::template MakeOutputTileDistribution<Problem>());

            ComputeDataType gate_acc0 = 0;
            ComputeDataType up_acc0   = 0;
            ComputeDataType gate_acc1 = 0;
            ComputeDataType up_acc1   = 0;
            const index_t num_iterations = kargs.hidden / kTileN;

            for(index_t i = 0; i < num_iterations; ++i)
            {
                auto x_tile       = load_tile(x_window);
                auto w_gate_tile0 = load_tile(w_gate_window0);
                auto w_gate_tile1 = load_tile(w_gate_window1);
                auto w_up_tile0   = load_tile(w_up_window0);
                auto w_up_tile1   = load_tile(w_up_window1);

                const index_t k_base = i * kTileN + get_lane_id() * kVector;

                ComputeDataType xs = x_scale_val;
                if constexpr(ScaleLayoutTraits<XScaleLayout>::is_block2d)
                {
                    constexpr index_t Block_K = ScaleLayoutTraits<XScaleLayout>::block_k;
                    xs = use_x_scale_lds ? x_scale_lds[k_base / Block_K]
                                          : load_block2d_scale<XScaleLayout, XScaleDataType>(
                                                kargs.p_x_scale, token_b, k_base, kargs.hidden);
                }

                ComputeDataType gs0 = w_gate_scale_val0;
                ComputeDataType gs1 = w_gate_scale_val1;
                ComputeDataType us0 = w_up_scale_val0;
                ComputeDataType us1 = w_up_scale_val1;
                if constexpr(ScaleLayoutTraits<WScaleLayout>::is_block2d)
                {
                    constexpr index_t Block_K = ScaleLayoutTraits<WScaleLayout>::block_k;
                    gs0 = use_w_scale_lds ? w_gate_scale_lds[0][k_base / Block_K]
                                          : load_block2d_scale<WScaleLayout, WScaleDataType>(
                                                kargs.p_w_gate_scale, w_row0, k_base, kargs.hidden);
                    gs1 = use_w_scale_lds ? w_gate_scale_lds[1][k_base / Block_K]
                                          : load_block2d_scale<WScaleLayout, WScaleDataType>(
                                                kargs.p_w_gate_scale, w_row1, k_base, kargs.hidden);
                    us0 = use_w_scale_lds ? w_up_scale_lds[0][k_base / Block_K]
                                          : load_block2d_scale<WScaleLayout, WScaleDataType>(
                                                kargs.p_w_up_scale, w_row0, k_base, kargs.hidden);
                    us1 = use_w_scale_lds ? w_up_scale_lds[1][k_base / Block_K]
                                          : load_block2d_scale<WScaleLayout, WScaleDataType>(
                                                kargs.p_w_up_scale, w_row1, k_base, kargs.hidden);
                }

                ComputeDataType gate_dot0 = 0;
                ComputeDataType up_dot0   = 0;
                ComputeDataType gate_dot1 = 0;
                ComputeDataType up_dot1   = 0;
                if constexpr(std::is_same_v<XDataType, bf16_t>)
                {
                    static_for<0, kVector / 2, 1>{}([&](auto ipair) {
                        const uint32_t x_pair =
                            x_tile.get_thread_buffer().template get_as<uint32_t>(ipair);
                        constexpr index_t w_word = ipair.value / 2;
                        constexpr index_t w_sel  = ipair.value % 2;
                        const uint32_t g_pair0 = fp8x2_to_bf16x2<w_sel>(
                            w_gate_tile0.get_thread_buffer().template get_as<uint32_t>(
                                number<w_word>{}));
                        const uint32_t u_pair0 = fp8x2_to_bf16x2<w_sel>(
                            w_up_tile0.get_thread_buffer().template get_as<uint32_t>(
                                number<w_word>{}));
                        const uint32_t g_pair1 = fp8x2_to_bf16x2<w_sel>(
                            w_gate_tile1.get_thread_buffer().template get_as<uint32_t>(
                                number<w_word>{}));
                        const uint32_t u_pair1 = fp8x2_to_bf16x2<w_sel>(
                            w_up_tile1.get_thread_buffer().template get_as<uint32_t>(
                                number<w_word>{}));
                        gate_dot0 = dot2_bf16_packed_add(gate_dot0, x_pair, g_pair0);
                        up_dot0   = dot2_bf16_packed_add(up_dot0, x_pair, u_pair0);
                        gate_dot1 = dot2_bf16_packed_add(gate_dot1, x_pair, g_pair1);
                        up_dot1   = dot2_bf16_packed_add(up_dot1, x_pair, u_pair1);
                    });
                }
                else
                {
                    static_for<0, kVector / 2, 1>{}([&](auto ipair) {
                        constexpr index_t word = ipair.value / 2;
                        constexpr index_t sel  = ipair.value % 2;
                        const uint32_t x_pair = fp8x2_to_bf16x2<sel>(
                            x_tile.get_thread_buffer().template get_as<uint32_t>(number<word>{}));
                        const uint32_t g_pair0 = fp8x2_to_bf16x2<sel>(
                            w_gate_tile0.get_thread_buffer().template get_as<uint32_t>(
                                number<word>{}));
                        const uint32_t u_pair0 = fp8x2_to_bf16x2<sel>(
                            w_up_tile0.get_thread_buffer().template get_as<uint32_t>(
                                number<word>{}));
                        const uint32_t g_pair1 = fp8x2_to_bf16x2<sel>(
                            w_gate_tile1.get_thread_buffer().template get_as<uint32_t>(
                                number<word>{}));
                        const uint32_t u_pair1 = fp8x2_to_bf16x2<sel>(
                            w_up_tile1.get_thread_buffer().template get_as<uint32_t>(
                                number<word>{}));
                        gate_dot0 = dot2_bf16_packed_add(gate_dot0, x_pair, g_pair0);
                        up_dot0   = dot2_bf16_packed_add(up_dot0, x_pair, u_pair0);
                        gate_dot1 = dot2_bf16_packed_add(gate_dot1, x_pair, g_pair1);
                        up_dot1   = dot2_bf16_packed_add(up_dot1, x_pair, u_pair1);
                    });
                }

                gate_acc0 += gate_dot0 * (xs * gs0);
                up_acc0   += up_dot0 * (xs * us0);
                gate_acc1 += gate_dot1 * (xs * gs1);
                up_acc1   += up_dot1 * (xs * us1);

                move_tile_window(x_window, {0, kTileN});
                move_tile_window(w_gate_window0, {0, kTileN});
                move_tile_window(w_gate_window1, {0, kTileN});
                move_tile_window(w_up_window0, {0, kTileN});
                move_tile_window(w_up_window1, {0, kTileN});
            }

            gate_acc0 = wavefront_reduce_sum(gate_acc0);
            up_acc0   = wavefront_reduce_sum(up_acc0);
            gate_acc1 = wavefront_reduce_sum(gate_acc1);
            up_acc1   = wavefront_reduce_sum(up_acc1);

            if(get_lane_id() == 0)
            {
                typename Problem::Activation activation_func;
                ComputeDataType silu_gate0;
                ComputeDataType silu_gate1;
                activation_func(silu_gate0, gate_acc0);
                activation_func(silu_gate1, gate_acc1);
                auto* out = static_cast<IntermediateDataType*>(kargs.p_intermediate) +
                            (token_b * kargs.top_k + expert_k) * kargs.stride_intermediate;
                out[neuron_j0] = type_convert<IntermediateDataType>(silu_gate0 * up_acc0);
                out[neuron_j1] = type_convert<IntermediateDataType>(silu_gate1 * up_acc1);
            }
            return;
        }

        const index_t block_id = get_block_id();
        const index_t neuron_j = block_id % kargs.inter;
        const index_t block_div_inter = block_id / kargs.inter;
        const index_t expert_k = block_div_inter % kargs.top_k;
        const index_t token_b  = block_div_inter / kargs.top_k;

        if(token_b >= kargs.b)
            return;

        const int32_t e = kargs.p_router_ids[token_b * kargs.top_k + expert_k];
        const index_t w_row = e * kargs.inter + neuron_j;
        constexpr index_t kMaxScaleBlocks = 128;

        __shared__ ComputeDataType x_scale_lds[kMaxScaleBlocks];
        __shared__ ComputeDataType w_gate_scale_lds[kMaxScaleBlocks];
        __shared__ ComputeDataType w_up_scale_lds[kMaxScaleBlocks];

        // --- Loop-invariant scale values ---
        ComputeDataType x_scale_val = type_convert<ComputeDataType>(1.0f);
        if constexpr(std::is_same_v<XScaleLayout, WarpDecodeScaleLayout::PerTensor>)
        {
            if(kargs.p_x_scale)
                x_scale_val = type_convert<ComputeDataType>(
                    *static_cast<const XScaleDataType*>(kargs.p_x_scale));
        }
        else if constexpr(std::is_same_v<XScaleLayout, WarpDecodeScaleLayout::PerToken>)
        {
            if(kargs.p_x_scale)
                x_scale_val = type_convert<ComputeDataType>(
                    static_cast<const XScaleDataType*>(kargs.p_x_scale)[token_b]);
        }

        ComputeDataType w_gate_scale_val = type_convert<ComputeDataType>(1.0f);
        if constexpr(std::is_same_v<WScaleLayout, WarpDecodeScaleLayout::PerTensor>)
        {
            if(kargs.p_w_gate_scale)
                w_gate_scale_val = type_convert<ComputeDataType>(
                    *static_cast<const WScaleDataType*>(kargs.p_w_gate_scale));
        }
        else if constexpr(std::is_same_v<WScaleLayout, WarpDecodeScaleLayout::PerToken>)
        {
            if(kargs.p_w_gate_scale)
                w_gate_scale_val = type_convert<ComputeDataType>(
                    static_cast<const WScaleDataType*>(kargs.p_w_gate_scale)[e * kargs.inter + neuron_j]);
        }

        ComputeDataType w_up_scale_val = type_convert<ComputeDataType>(1.0f);
        if constexpr(std::is_same_v<WScaleLayout, WarpDecodeScaleLayout::PerTensor>)
        {
            if(kargs.p_w_up_scale)
                w_up_scale_val = type_convert<ComputeDataType>(
                    *static_cast<const WScaleDataType*>(kargs.p_w_up_scale));
        }
        else if constexpr(std::is_same_v<WScaleLayout, WarpDecodeScaleLayout::PerToken>)
        {
            if(kargs.p_w_up_scale)
                w_up_scale_val = type_convert<ComputeDataType>(
                    static_cast<const WScaleDataType*>(kargs.p_w_up_scale)[e * kargs.inter + neuron_j]);
        }

        bool use_x_scale_lds = false;
        if constexpr(ScaleLayoutTraits<XScaleLayout>::is_block2d)
        {
            constexpr index_t Block_N = ScaleLayoutTraits<XScaleLayout>::block_n;
            constexpr index_t Block_K = ScaleLayoutTraits<XScaleLayout>::block_k;
            const index_t num_scale_blocks = kargs.hidden / Block_K;
            use_x_scale_lds = kargs.p_x_scale != nullptr && num_scale_blocks <= kMaxScaleBlocks;
            if(use_x_scale_lds)
            {
                const auto* ptr = static_cast<const XScaleDataType*>(kargs.p_x_scale);
                const index_t scale_row = token_b / Block_N;
                for(index_t c = get_thread_id(); c < num_scale_blocks; c += kBlockSize)
                {
                    x_scale_lds[c] = type_convert<ComputeDataType>(
                        ptr[scale_row * num_scale_blocks + c]);
                }
            }
        }

        bool use_w_scale_lds = false;
        if constexpr(ScaleLayoutTraits<WScaleLayout>::is_block2d)
        {
            constexpr index_t Block_N = ScaleLayoutTraits<WScaleLayout>::block_n;
            constexpr index_t Block_K = ScaleLayoutTraits<WScaleLayout>::block_k;
            const index_t num_scale_blocks = kargs.hidden / Block_K;
            use_w_scale_lds = kargs.p_w_gate_scale != nullptr && kargs.p_w_up_scale != nullptr &&
                              num_scale_blocks <= kMaxScaleBlocks;
            if(use_w_scale_lds)
            {
                const auto* gate_ptr = static_cast<const WScaleDataType*>(kargs.p_w_gate_scale);
                const auto* up_ptr   = static_cast<const WScaleDataType*>(kargs.p_w_up_scale);
                const index_t scale_row = w_row / Block_N;
                for(index_t c = get_thread_id(); c < num_scale_blocks; c += kBlockSize)
                {
                    w_gate_scale_lds[c] = type_convert<ComputeDataType>(
                        gate_ptr[scale_row * num_scale_blocks + c]);
                    w_up_scale_lds[c] = type_convert<ComputeDataType>(
                        up_ptr[scale_row * num_scale_blocks + c]);
                }
            }
        }

        if(use_x_scale_lds || use_w_scale_lds)
        {
            block_sync_lds();
        }

        // --- Tensor views (kVector as guaranteed vector size for correct pk_fp4_t handling) ---
        const auto x_view = make_naive_tensor_view<address_space_enum::global>(
            static_cast<const XDataType*>(kargs.p_x),
            make_tuple(kargs.b, kargs.hidden),
            make_tuple(kargs.stride_x, 1),
            number<kVector>{}, number<1>{});

        const auto w_gate_view = make_naive_tensor_view<address_space_enum::global>(
            static_cast<const WDataType*>(kargs.p_w_gate),
            make_tuple(kargs.e * kargs.inter, kargs.hidden),
            make_tuple(kargs.stride_w_gate, 1),
            number<kVector>{}, number<1>{});

        const auto w_up_view = make_naive_tensor_view<address_space_enum::global>(
            static_cast<const WDataType*>(kargs.p_w_up),
            make_tuple(kargs.e * kargs.inter, kargs.hidden),
            make_tuple(kargs.stride_w_up, 1),
            number<kVector>{}, number<1>{});

        auto x_window = make_tile_window(
            x_view,
            make_tuple(number<1>{}, number<kTileN>{}),
            {token_b, 0},
            Policy::template MakeXBroadcastTileDistribution<Problem>());

        auto w_gate_window = make_tile_window(
            w_gate_view,
            make_tuple(number<1>{}, number<kTileN>{}),
            {w_row, 0},
            Policy::template MakeOutputTileDistribution<Problem>());

        auto w_up_window = make_tile_window(
            w_up_view,
            make_tuple(number<1>{}, number<kTileN>{}),
            {w_row, 0},
            Policy::template MakeOutputTileDistribution<Problem>());

        // --- Main accumulation loop ---
        ComputeDataType gate_acc = 0;
        ComputeDataType up_acc   = 0;

        const index_t num_iterations = kargs.hidden / kTileN;

        for(index_t i = 0; i < num_iterations; ++i)
        {
            auto x_tile      = load_tile(x_window);
            auto w_gate_tile = load_tile(w_gate_window);
            auto w_up_tile   = load_tile(w_up_window);

            const index_t k_base = i * kTileN + get_lane_id() * kVector;

            ComputeDataType xs = x_scale_val;
            if constexpr(ScaleLayoutTraits<XScaleLayout>::is_block2d)
            {
                constexpr index_t Block_K = ScaleLayoutTraits<XScaleLayout>::block_k;
                xs = use_x_scale_lds ? x_scale_lds[k_base / Block_K]
                                      : load_block2d_scale<XScaleLayout, XScaleDataType>(
                                            kargs.p_x_scale, token_b, k_base, kargs.hidden);
            }

            ComputeDataType gs = w_gate_scale_val;
            if constexpr(ScaleLayoutTraits<WScaleLayout>::is_block2d)
            {
                constexpr index_t Block_K = ScaleLayoutTraits<WScaleLayout>::block_k;
                gs = use_w_scale_lds ? w_gate_scale_lds[k_base / Block_K]
                                      : load_block2d_scale<WScaleLayout, WScaleDataType>(
                                            kargs.p_w_gate_scale, w_row, k_base, kargs.hidden);
            }

            ComputeDataType us = w_up_scale_val;
            if constexpr(ScaleLayoutTraits<WScaleLayout>::is_block2d)
            {
                constexpr index_t Block_K = ScaleLayoutTraits<WScaleLayout>::block_k;
                us = use_w_scale_lds ? w_up_scale_lds[k_base / Block_K]
                                      : load_block2d_scale<WScaleLayout, WScaleDataType>(
                                            kargs.p_w_up_scale, w_row, k_base, kargs.hidden);
            }

            index_t sub = 0;
            constexpr auto spans = decltype(x_tile)::get_distributed_spans();
            if constexpr(Problem::kUseDot2 && !is_packed_w)
            {
                static_assert(std::is_same_v<ComputeDataType, float>,
                              "WarpDecode gate_up dot2 path expects FP32 accumulation.");
                static_assert(kVector % 2 == 0,
                              "WarpDecode gate_up dot2 path requires an even vector length.");

                ComputeDataType gate_dot = 0;
                ComputeDataType up_dot   = 0;
                if constexpr(std::is_same_v<XDataType, bf16_t>)
                {
                    static_for<0, kVector / 2, 1>{}([&](auto ipair) {
                        const uint32_t x_pair =
                            x_tile.get_thread_buffer().template get_as<uint32_t>(ipair);
                        constexpr index_t w_word = ipair.value / 2;
                        constexpr index_t w_sel  = ipair.value % 2;
                        const uint32_t g_pair = fp8x2_to_bf16x2<w_sel>(
                            w_gate_tile.get_thread_buffer().template get_as<uint32_t>(
                                number<w_word>{}));
                        const uint32_t u_pair = fp8x2_to_bf16x2<w_sel>(
                            w_up_tile.get_thread_buffer().template get_as<uint32_t>(
                                number<w_word>{}));
                        gate_dot = dot2_bf16_packed_add(gate_dot, x_pair, g_pair);
                        up_dot   = dot2_bf16_packed_add(up_dot, x_pair, u_pair);
                    });
                }
                else
                {
                    static_for<0, kVector / 2, 1>{}([&](auto ipair) {
                        constexpr index_t word = ipair.value / 2;
                        constexpr index_t sel  = ipair.value % 2;
                        const uint32_t x_pair = fp8x2_to_bf16x2<sel>(
                            x_tile.get_thread_buffer().template get_as<uint32_t>(number<word>{}));
                        const uint32_t g_pair = fp8x2_to_bf16x2<sel>(
                            w_gate_tile.get_thread_buffer().template get_as<uint32_t>(
                                number<word>{}));
                        const uint32_t u_pair = fp8x2_to_bf16x2<sel>(
                            w_up_tile.get_thread_buffer().template get_as<uint32_t>(
                                number<word>{}));
                        gate_dot = dot2_bf16_packed_add(gate_dot, x_pair, g_pair);
                        up_dot   = dot2_bf16_packed_add(up_dot, x_pair, u_pair);
                    });
                }
                gate_acc += gate_dot * (xs * gs);
                up_acc   += up_dot * (xs * us);
            }
            else if constexpr(Problem::kUsePackedFp32 && !is_packed_w)
            {
                static_assert(std::is_same_v<ComputeDataType, float>,
                              "WarpDecode gate_up packed-FP32 path expects FP32 accumulation.");
                static_assert(std::is_same_v<WDataType, fp8_t>,
                              "WarpDecode gate_up packed-FP32 path currently expects FP8 weights.");
                static_assert(kVector % 2 == 0,
                              "WarpDecode gate_up packed-FP32 path requires an even vector length.");

                fp32x2_t gate_dot{0.0f, 0.0f};
                fp32x2_t up_dot{0.0f, 0.0f};
                static_for<0, kVector / 2, 1>{}([&](auto ipair) {
                    constexpr index_t word = ipair.value / 2;
                    constexpr index_t sel  = ipair.value % 2;

                    fp32x2_t x_pair;
                    if constexpr(std::is_same_v<XDataType, bf16_t>)
                    {
                        x_pair = bf16x2_to_f32x2(
                            x_tile.get_thread_buffer().template get_as<uint32_t>(ipair));
                    }
                    else
                    {
                        static_assert(std::is_same_v<XDataType, fp8_t>,
                                      "WarpDecode gate_up packed-FP32 path expects FP8 or BF16 activations.");
                        x_pair = fp8x2_to_f32x2<sel>(
                            x_tile.get_thread_buffer().template get_as<uint32_t>(number<word>{}));
                    }

                    const fp32x2_t g_pair = fp8x2_to_f32x2<sel>(
                        w_gate_tile.get_thread_buffer().template get_as<uint32_t>(number<word>{}));
                    const fp32x2_t u_pair = fp8x2_to_f32x2<sel>(
                        w_up_tile.get_thread_buffer().template get_as<uint32_t>(number<word>{}));

                    gate_dot = pk_fma_f32(gate_dot, x_pair, g_pair);
                    up_dot   = pk_fma_f32(up_dot, x_pair, u_pair);
                });
                gate_acc += horizontal_add(gate_dot) * (xs * gs);
                up_acc   += horizontal_add(up_dot) * (xs * us);
            }
            else
            {
                sweep_tile_span(spans[number<1>{}], [&](auto idx1) {
                    constexpr auto idx = make_tuple(make_tuple(), idx1);
                    auto x_val = type_convert<ComputeDataType>(x_tile[idx]);

                    ComputeDataType g_val, u_val;
                    if constexpr(is_packed_w)
                    {
                        g_val = unpack_fp4_nibble(
                            static_cast<uint8_t>(w_gate_tile[idx]), sub);
                        u_val = unpack_fp4_nibble(
                            static_cast<uint8_t>(w_up_tile[idx]), sub);
                        sub ^= 1;
                    }
                    else
                    {
                        g_val = type_convert<ComputeDataType>(w_gate_tile[idx]);
                        u_val = type_convert<ComputeDataType>(w_up_tile[idx]);
                    }

                    gate_acc += (x_val * xs) * (g_val * gs);
                    up_acc   += (x_val * xs) * (u_val * us);
                });
            }

            move_tile_window(x_window, {0, kTileN});
            move_tile_window(w_gate_window, {0, kTileN});
            move_tile_window(w_up_window, {0, kTileN});
        }

        gate_acc = wavefront_reduce_sum(gate_acc);
        up_acc   = wavefront_reduce_sum(up_acc);

        if(get_lane_id() == 0)
        {
            typename Problem::Activation activation_func;
            ComputeDataType silu_gate;
            activation_func(silu_gate, gate_acc);
            ComputeDataType result = silu_gate * up_acc;

            static_cast<IntermediateDataType*>(kargs.p_intermediate)
                [(token_b * kargs.top_k + expert_k) * kargs.stride_intermediate + neuron_j] =
                    type_convert<IntermediateDataType>(result);
        }
    }
};

template <typename Problem_, typename Policy_>
struct WarpDecodeGateUpLdsXKernel : public WarpDecodeGateUpKernel<Problem_, Policy_>
{
    using Base    = WarpDecodeGateUpKernel<Problem_, Policy_>;
    using Problem = remove_cvref_t<Problem_>;
    using Policy  = remove_cvref_t<Policy_>;

    using XDataType            = typename Problem::XDataType;
    using WDataType            = typename Problem::WDataType;
    using ComputeDataType      = typename Problem::ComputeDataType;
    using IntermediateDataType = typename Problem::IntermediateDataType;
    using XScaleDataType       = typename Problem::XScaleDataType;
    using WScaleDataType       = typename Problem::WScaleDataType;

    using Kargs = typename Base::Kargs;

    static constexpr index_t kWarpsPerBlock = Problem::kWarpsPerBlock;
    static constexpr index_t kBlockSize     = Problem::kBlockSize;
    static constexpr index_t kMaxHidden     = 8192;
    static constexpr index_t kMaxScaleBlocks = 128;

    CK_TILE_HOST static constexpr auto GridSize(const Kargs& hargs)
    {
        constexpr index_t kColsPerCta = kWarpsPerBlock * Problem::kNPerWarp;
        return dim3(hargs.b * hargs.top_k * integer_divide_ceil(hargs.inter, kColsPerCta));
    }

    CK_TILE_HOST static constexpr auto BlockSize()
    {
        return kBlockSize;
    }

    CK_TILE_HOST static bool IsSupportedArgument(const Kargs& kargs)
    {
        if(!Base::IsSupportedArgument(kargs))
        {
            return false;
        }
        constexpr index_t kColsPerCta = kWarpsPerBlock * Problem::kNPerWarp;
        if(kargs.hidden > kMaxHidden || kargs.inter % kColsPerCta != 0)
        {
            return false;
        }
        if constexpr(Problem::kNPerWarp != 1)
        {
            if constexpr(Problem::kNPerWarp != 2 || !Problem::kUseDot2 ||
                         std::is_same_v<typename Problem::WDataType, pk_fp4_t>)
            {
                return false;
            }
        }
        return true;
    }

    CK_TILE_DEVICE void operator()(Kargs kargs) const
    {
        constexpr bool is_packed_w = std::is_same_v<WDataType, pk_fp4_t>;
        constexpr index_t kVector  = Problem::kVector;
        constexpr index_t kTileN   = get_warp_size() * kVector;

        using XScaleLayout = typename Problem::XScaleLayout;
        using WScaleLayout = typename Problem::WScaleLayout;
        using Activation   = typename Problem::Activation;

        if constexpr(Problem::kNPerWarp == 2)
        {
            static_assert(Problem::kUseDot2,
                          "WarpDecodeGateUpLdsXKernel NPerWarp=2 expects dot2.");
            static_assert(!is_packed_w,
                          "WarpDecodeGateUpLdsXKernel NPerWarp=2 does not support packed FP4 weights.");

            constexpr index_t kNPerWarp  = Problem::kNPerWarp;
            constexpr index_t kColsPerCta = kWarpsPerBlock * kNPerWarp;

            const index_t block_id = get_block_id();
            const index_t inter_block = block_id % integer_divide_ceil(kargs.inter, kColsPerCta);
            const index_t block_div_inter = block_id / integer_divide_ceil(kargs.inter, kColsPerCta);
            const index_t expert_k = block_div_inter % kargs.top_k;
            const index_t token_b  = block_div_inter / kargs.top_k;
            const index_t neuron_j0 = inter_block * kColsPerCta + get_warp_id() * kNPerWarp;
            const index_t neuron_j1 = neuron_j0 + 1;

            if(token_b >= kargs.b || neuron_j1 >= kargs.inter)
                return;

            const int32_t e = kargs.p_router_ids[token_b * kargs.top_k + expert_k];
            const index_t w_row0 = e * kargs.inter + neuron_j0;
            const index_t w_row1 = w_row0 + 1;

            constexpr index_t kXCopyVector =
                (sizeof(XDataType) * kVector / numeric_traits<XDataType>::PackedSize <= 16)
                    ? kVector
                    : (16 * numeric_traits<XDataType>::PackedSize / sizeof(XDataType));
            constexpr index_t kCopyTileN = kBlockSize * kXCopyVector;

            alignas(16) __shared__ XDataType x_lds[2][kCopyTileN];
            __shared__ ComputeDataType x_scale_lds[kMaxScaleBlocks];
            __shared__ ComputeDataType w_gate_scale_lds[kWarpsPerBlock][kNPerWarp][kMaxScaleBlocks];
            __shared__ ComputeDataType w_up_scale_lds[kWarpsPerBlock][kNPerWarp][kMaxScaleBlocks];

            const auto x_copy_base_view = make_naive_tensor_view<address_space_enum::global>(
                static_cast<const XDataType*>(kargs.p_x),
                make_tuple(kargs.b, kargs.hidden),
                make_tuple(kargs.stride_x, 1),
                number<kXCopyVector>{}, number<1>{});
            const auto x_copy_view = pad_tensor_view(
                x_copy_base_view,
                make_tuple(number<1>{}, number<kCopyTileN>{}),
                sequence<false, true>{});

            const auto prefetch_x_chunk = [&](index_t stage, index_t copy_offset) {
                const index_t valid_count = min(kCopyTileN, kargs.hidden - copy_offset);
                if(valid_count < kCopyTileN)
                {
                    for(index_t idx = get_thread_id(); idx < kCopyTileN; idx += kBlockSize)
                    {
                        if(idx >= valid_count)
                        {
                            x_lds[stage][idx] = XDataType{};
                        }
                    }
                    block_sync_lds();
                }

                auto x_lds_view = make_naive_tensor_view<address_space_enum::lds>(
                    &x_lds[stage][0],
                    make_tuple(number<1>{}, number<kCopyTileN>{}),
                    make_tuple(number<kCopyTileN>{}, number<1>{}),
                    number<kXCopyVector>{}, number<1>{});
                auto x_copy_window = make_tile_window(
                    x_copy_view,
                    make_tuple(number<1>{}, number<kCopyTileN>{}),
                    {token_b, copy_offset},
                    Policy::template MakeBlockCopyTileDistribution<Problem, kXCopyVector>());
                auto x_lds_window = make_tile_window(
                    x_lds_view,
                    make_tuple(number<1>{}, number<kCopyTileN>{}),
                    {0, 0},
                    Policy::template MakeBlockCopyTileDistribution<Problem, kXCopyVector>());
                async_load_tile(x_lds_window, x_copy_window);
            };

            prefetch_x_chunk(0, 0);

            bool use_x_scale_lds = false;
            if constexpr(ScaleLayoutTraits<XScaleLayout>::is_block2d)
            {
                constexpr index_t Block_N = ScaleLayoutTraits<XScaleLayout>::block_n;
                constexpr index_t Block_K = ScaleLayoutTraits<XScaleLayout>::block_k;
                const index_t num_scale_blocks = kargs.hidden / Block_K;
                use_x_scale_lds = kargs.p_x_scale != nullptr && num_scale_blocks <= kMaxScaleBlocks;
                if(use_x_scale_lds)
                {
                    const auto* ptr = static_cast<const XScaleDataType*>(kargs.p_x_scale);
                    const index_t scale_row = token_b / Block_N;
                    for(index_t c = get_thread_id(); c < num_scale_blocks; c += kBlockSize)
                    {
                        x_scale_lds[c] = type_convert<ComputeDataType>(
                            ptr[scale_row * num_scale_blocks + c]);
                    }
                }
            }

            bool use_w_scale_lds = false;
            if constexpr(ScaleLayoutTraits<WScaleLayout>::is_block2d)
            {
                constexpr index_t Block_N = ScaleLayoutTraits<WScaleLayout>::block_n;
                constexpr index_t Block_K = ScaleLayoutTraits<WScaleLayout>::block_k;
                const index_t num_scale_blocks = kargs.hidden / Block_K;
                use_w_scale_lds = kargs.p_w_gate_scale != nullptr && kargs.p_w_up_scale != nullptr &&
                                  num_scale_blocks <= kMaxScaleBlocks;
                if(use_w_scale_lds)
                {
                    const auto* gate_ptr = static_cast<const WScaleDataType*>(kargs.p_w_gate_scale);
                    const auto* up_ptr   = static_cast<const WScaleDataType*>(kargs.p_w_up_scale);
                    const index_t scale_row0 = w_row0 / Block_N;
                    const index_t scale_row1 = w_row1 / Block_N;
                    for(index_t c = get_lane_id(); c < num_scale_blocks; c += get_warp_size())
                    {
                        w_gate_scale_lds[get_warp_id()][0][c] = type_convert<ComputeDataType>(
                            gate_ptr[scale_row0 * num_scale_blocks + c]);
                        w_up_scale_lds[get_warp_id()][0][c] = type_convert<ComputeDataType>(
                            up_ptr[scale_row0 * num_scale_blocks + c]);
                        w_gate_scale_lds[get_warp_id()][1][c] = type_convert<ComputeDataType>(
                            gate_ptr[scale_row1 * num_scale_blocks + c]);
                        w_up_scale_lds[get_warp_id()][1][c] = type_convert<ComputeDataType>(
                            up_ptr[scale_row1 * num_scale_blocks + c]);
                    }
                }
            }

            block_sync_lds_direct_load();

            ComputeDataType x_scale_val = type_convert<ComputeDataType>(1.0f);
            if constexpr(std::is_same_v<XScaleLayout, WarpDecodeScaleLayout::PerTensor>)
            {
                if(kargs.p_x_scale)
                    x_scale_val = type_convert<ComputeDataType>(
                        *static_cast<const XScaleDataType*>(kargs.p_x_scale));
            }
            else if constexpr(std::is_same_v<XScaleLayout, WarpDecodeScaleLayout::PerToken>)
            {
                if(kargs.p_x_scale)
                    x_scale_val = type_convert<ComputeDataType>(
                        static_cast<const XScaleDataType*>(kargs.p_x_scale)[token_b]);
            }

            ComputeDataType w_gate_scale_val0 = type_convert<ComputeDataType>(1.0f);
            ComputeDataType w_gate_scale_val1 = type_convert<ComputeDataType>(1.0f);
            ComputeDataType w_up_scale_val0   = type_convert<ComputeDataType>(1.0f);
            ComputeDataType w_up_scale_val1   = type_convert<ComputeDataType>(1.0f);
            if constexpr(std::is_same_v<WScaleLayout, WarpDecodeScaleLayout::PerTensor>)
            {
                if(kargs.p_w_gate_scale)
                {
                    w_gate_scale_val0 = type_convert<ComputeDataType>(
                        *static_cast<const WScaleDataType*>(kargs.p_w_gate_scale));
                    w_gate_scale_val1 = w_gate_scale_val0;
                }
                if(kargs.p_w_up_scale)
                {
                    w_up_scale_val0 = type_convert<ComputeDataType>(
                        *static_cast<const WScaleDataType*>(kargs.p_w_up_scale));
                    w_up_scale_val1 = w_up_scale_val0;
                }
            }
            else if constexpr(std::is_same_v<WScaleLayout, WarpDecodeScaleLayout::PerToken>)
            {
                if(kargs.p_w_gate_scale)
                {
                    const auto* ptr = static_cast<const WScaleDataType*>(kargs.p_w_gate_scale);
                    w_gate_scale_val0 = type_convert<ComputeDataType>(ptr[w_row0]);
                    w_gate_scale_val1 = type_convert<ComputeDataType>(ptr[w_row1]);
                }
                if(kargs.p_w_up_scale)
                {
                    const auto* ptr = static_cast<const WScaleDataType*>(kargs.p_w_up_scale);
                    w_up_scale_val0 = type_convert<ComputeDataType>(ptr[w_row0]);
                    w_up_scale_val1 = type_convert<ComputeDataType>(ptr[w_row1]);
                }
            }

            const auto w_gate_view = make_naive_tensor_view<address_space_enum::global>(
                static_cast<const WDataType*>(kargs.p_w_gate),
                make_tuple(kargs.e * kargs.inter, kargs.hidden),
                make_tuple(kargs.stride_w_gate, 1),
                number<kVector>{}, number<1>{});
            const auto w_up_view = make_naive_tensor_view<address_space_enum::global>(
                static_cast<const WDataType*>(kargs.p_w_up),
                make_tuple(kargs.e * kargs.inter, kargs.hidden),
                make_tuple(kargs.stride_w_up, 1),
                number<kVector>{}, number<1>{});

            auto w_gate_window0 = make_tile_window(
                w_gate_view,
                make_tuple(number<1>{}, number<kTileN>{}),
                {w_row0, 0},
                Policy::template MakeXBroadcastTileDistribution<Problem>());
            auto w_gate_window1 = make_tile_window(
                w_gate_view,
                make_tuple(number<1>{}, number<kTileN>{}),
                {w_row1, 0},
                Policy::template MakeXBroadcastTileDistribution<Problem>());
            auto w_up_window0 = make_tile_window(
                w_up_view,
                make_tuple(number<1>{}, number<kTileN>{}),
                {w_row0, 0},
                Policy::template MakeXBroadcastTileDistribution<Problem>());
            auto w_up_window1 = make_tile_window(
                w_up_view,
                make_tuple(number<1>{}, number<kTileN>{}),
                {w_row1, 0},
                Policy::template MakeXBroadcastTileDistribution<Problem>());

            ComputeDataType gate_acc0 = 0;
            ComputeDataType up_acc0   = 0;
            ComputeDataType gate_acc1 = 0;
            ComputeDataType up_acc1   = 0;
            const index_t num_iterations = kargs.hidden / kTileN;
            const index_t num_copy_chunks = integer_divide_ceil(kargs.hidden, kCopyTileN);

            for(index_t copy_chunk = 0; copy_chunk < num_copy_chunks; ++copy_chunk)
            {
                const index_t chunk_base = copy_chunk * kCopyTileN;
                const index_t next_chunk = copy_chunk + 1;
                if(next_chunk < num_copy_chunks)
                {
                    prefetch_x_chunk(next_chunk & 1, next_chunk * kCopyTileN);
                }

                const index_t chunk_iter_begin = chunk_base / kTileN;
                const index_t chunk_iter_end =
                    min(num_iterations, (chunk_base + kCopyTileN) / kTileN);
                auto x_lds_compute_view = make_naive_tensor_view<address_space_enum::lds>(
                    &x_lds[copy_chunk & 1][0],
                    make_tuple(number<1>{}, number<kCopyTileN>{}),
                    make_tuple(number<kCopyTileN>{}, number<1>{}),
                    number<kVector>{}, number<1>{});
                auto x_lds_window = make_tile_window(
                    x_lds_compute_view,
                    make_tuple(number<1>{}, number<kTileN>{}),
                    {0, chunk_iter_begin * kTileN - chunk_base},
                    Policy::template MakeXBroadcastTileDistribution<Problem>());

                for(index_t i = chunk_iter_begin; i < chunk_iter_end; ++i)
                {
                    auto x_tile       = load_tile(x_lds_window);
                    auto w_gate_tile0 = load_tile(w_gate_window0);
                    auto w_gate_tile1 = load_tile(w_gate_window1);
                    auto w_up_tile0   = load_tile(w_up_window0);
                    auto w_up_tile1   = load_tile(w_up_window1);
                    const index_t k_base = i * kTileN + get_lane_id() * kVector;

                    ComputeDataType xs = x_scale_val;
                    if constexpr(ScaleLayoutTraits<XScaleLayout>::is_block2d)
                    {
                        constexpr index_t Block_K = ScaleLayoutTraits<XScaleLayout>::block_k;
                        xs = use_x_scale_lds ? x_scale_lds[k_base / Block_K]
                                              : Base::template load_block2d_scale<XScaleLayout, XScaleDataType>(
                                                    kargs.p_x_scale, token_b, k_base, kargs.hidden);
                    }

                    ComputeDataType gs0 = w_gate_scale_val0;
                    ComputeDataType gs1 = w_gate_scale_val1;
                    ComputeDataType us0 = w_up_scale_val0;
                    ComputeDataType us1 = w_up_scale_val1;
                    if constexpr(ScaleLayoutTraits<WScaleLayout>::is_block2d)
                    {
                        constexpr index_t Block_K = ScaleLayoutTraits<WScaleLayout>::block_k;
                        gs0 = use_w_scale_lds ? w_gate_scale_lds[get_warp_id()][0][k_base / Block_K]
                                              : Base::template load_block2d_scale<WScaleLayout, WScaleDataType>(
                                                    kargs.p_w_gate_scale, w_row0, k_base, kargs.hidden);
                        gs1 = use_w_scale_lds ? w_gate_scale_lds[get_warp_id()][1][k_base / Block_K]
                                              : Base::template load_block2d_scale<WScaleLayout, WScaleDataType>(
                                                    kargs.p_w_gate_scale, w_row1, k_base, kargs.hidden);
                        us0 = use_w_scale_lds ? w_up_scale_lds[get_warp_id()][0][k_base / Block_K]
                                              : Base::template load_block2d_scale<WScaleLayout, WScaleDataType>(
                                                    kargs.p_w_up_scale, w_row0, k_base, kargs.hidden);
                        us1 = use_w_scale_lds ? w_up_scale_lds[get_warp_id()][1][k_base / Block_K]
                                              : Base::template load_block2d_scale<WScaleLayout, WScaleDataType>(
                                                    kargs.p_w_up_scale, w_row1, k_base, kargs.hidden);
                    }

                    ComputeDataType gate_dot0 = 0;
                    ComputeDataType up_dot0   = 0;
                    ComputeDataType gate_dot1 = 0;
                    ComputeDataType up_dot1   = 0;
                    if constexpr(std::is_same_v<XDataType, bf16_t>)
                    {
                        static_for<0, kVector / 2, 1>{}([&](auto ipair) {
                            const uint32_t x_pair =
                                x_tile.get_thread_buffer().template get_as<uint32_t>(ipair);
                            constexpr index_t w_word = ipair.value / 2;
                            constexpr index_t w_sel  = ipair.value % 2;
                            const uint32_t g_pair0 = Base::template fp8x2_to_bf16x2<w_sel>(
                                w_gate_tile0.get_thread_buffer().template get_as<uint32_t>(
                                    number<w_word>{}));
                            const uint32_t u_pair0 = Base::template fp8x2_to_bf16x2<w_sel>(
                                w_up_tile0.get_thread_buffer().template get_as<uint32_t>(
                                    number<w_word>{}));
                            const uint32_t g_pair1 = Base::template fp8x2_to_bf16x2<w_sel>(
                                w_gate_tile1.get_thread_buffer().template get_as<uint32_t>(
                                    number<w_word>{}));
                            const uint32_t u_pair1 = Base::template fp8x2_to_bf16x2<w_sel>(
                                w_up_tile1.get_thread_buffer().template get_as<uint32_t>(
                                    number<w_word>{}));
                            gate_dot0 = Base::dot2_bf16_packed_add(gate_dot0, x_pair, g_pair0);
                            up_dot0   = Base::dot2_bf16_packed_add(up_dot0, x_pair, u_pair0);
                            gate_dot1 = Base::dot2_bf16_packed_add(gate_dot1, x_pair, g_pair1);
                            up_dot1   = Base::dot2_bf16_packed_add(up_dot1, x_pair, u_pair1);
                        });
                    }
                    else
                    {
                        static_for<0, kVector / 2, 1>{}([&](auto ipair) {
                            constexpr index_t word = ipair.value / 2;
                            constexpr index_t sel  = ipair.value % 2;
                            const uint32_t x_pair = Base::template fp8x2_to_bf16x2<sel>(
                                x_tile.get_thread_buffer().template get_as<uint32_t>(
                                    number<word>{}));
                            const uint32_t g_pair0 = Base::template fp8x2_to_bf16x2<sel>(
                                w_gate_tile0.get_thread_buffer().template get_as<uint32_t>(
                                    number<word>{}));
                            const uint32_t u_pair0 = Base::template fp8x2_to_bf16x2<sel>(
                                w_up_tile0.get_thread_buffer().template get_as<uint32_t>(
                                    number<word>{}));
                            const uint32_t g_pair1 = Base::template fp8x2_to_bf16x2<sel>(
                                w_gate_tile1.get_thread_buffer().template get_as<uint32_t>(
                                    number<word>{}));
                            const uint32_t u_pair1 = Base::template fp8x2_to_bf16x2<sel>(
                                w_up_tile1.get_thread_buffer().template get_as<uint32_t>(
                                    number<word>{}));
                            gate_dot0 = Base::dot2_bf16_packed_add(gate_dot0, x_pair, g_pair0);
                            up_dot0   = Base::dot2_bf16_packed_add(up_dot0, x_pair, u_pair0);
                            gate_dot1 = Base::dot2_bf16_packed_add(gate_dot1, x_pair, g_pair1);
                            up_dot1   = Base::dot2_bf16_packed_add(up_dot1, x_pair, u_pair1);
                        });
                    }

                    gate_acc0 += gate_dot0 * (xs * gs0);
                    up_acc0   += up_dot0 * (xs * us0);
                    gate_acc1 += gate_dot1 * (xs * gs1);
                    up_acc1   += up_dot1 * (xs * us1);

                    move_tile_window(x_lds_window, {0, kTileN});
                    move_tile_window(w_gate_window0, {0, kTileN});
                    move_tile_window(w_gate_window1, {0, kTileN});
                    move_tile_window(w_up_window0, {0, kTileN});
                    move_tile_window(w_up_window1, {0, kTileN});
                }

                if(next_chunk < num_copy_chunks)
                {
                    block_sync_lds_direct_load();
                }
            }

            gate_acc0 = Base::wavefront_reduce_sum(gate_acc0);
            up_acc0   = Base::wavefront_reduce_sum(up_acc0);
            gate_acc1 = Base::wavefront_reduce_sum(gate_acc1);
            up_acc1   = Base::wavefront_reduce_sum(up_acc1);

            if(get_lane_id() == 0)
            {
                Activation act;
                ComputeDataType out0;
                ComputeDataType out1;
                act(out0, gate_acc0);
                act(out1, gate_acc1);
                auto* out = static_cast<IntermediateDataType*>(kargs.p_intermediate) +
                            (token_b * kargs.top_k + expert_k) * kargs.stride_intermediate;
                out[neuron_j0] = type_convert<IntermediateDataType>(out0 * up_acc0);
                out[neuron_j1] = type_convert<IntermediateDataType>(out1 * up_acc1);
            }
            return;
        }

        const index_t block_id = get_block_id();
        const index_t inter_block = block_id % integer_divide_ceil(kargs.inter, kWarpsPerBlock);
        const index_t block_div_inter = block_id / integer_divide_ceil(kargs.inter, kWarpsPerBlock);
        const index_t expert_k = block_div_inter % kargs.top_k;
        const index_t token_b  = block_div_inter / kargs.top_k;
        const index_t neuron_j = inter_block * kWarpsPerBlock + get_warp_id();
        const index_t neuron_block_base = inter_block * kWarpsPerBlock;

        if(token_b >= kargs.b || neuron_j >= kargs.inter)
            return;

        const int32_t e = kargs.p_router_ids[token_b * kargs.top_k + expert_k];
        const index_t w_row = e * kargs.inter + neuron_j;
        const index_t w_row_base = e * kargs.inter + neuron_block_base;

        constexpr index_t kXCopyVector =
            (sizeof(XDataType) * kVector / numeric_traits<XDataType>::PackedSize <= 16)
                ? kVector
                : (16 * numeric_traits<XDataType>::PackedSize / sizeof(XDataType));
        constexpr index_t kCopyTileN = kBlockSize * kXCopyVector;

        alignas(16) __shared__ XDataType x_lds[2][kCopyTileN];
        __shared__ ComputeDataType x_scale_lds[kMaxScaleBlocks];
        __shared__ ComputeDataType w_gate_scale_lds[kWarpsPerBlock][kMaxScaleBlocks];
        __shared__ ComputeDataType w_up_scale_lds[kWarpsPerBlock][kMaxScaleBlocks];

        const auto x_copy_base_view = make_naive_tensor_view<address_space_enum::global>(
            static_cast<const XDataType*>(kargs.p_x),
            make_tuple(kargs.b, kargs.hidden),
            make_tuple(kargs.stride_x, 1),
            number<kXCopyVector>{}, number<1>{});
        const auto x_copy_view = pad_tensor_view(
            x_copy_base_view,
            make_tuple(number<1>{}, number<kCopyTileN>{}),
            sequence<false, true>{});

        const auto prefetch_x_chunk = [&](index_t stage, index_t copy_offset) {
            const index_t valid_count = min(kCopyTileN, kargs.hidden - copy_offset);
            if(valid_count < kCopyTileN)
            {
                for(index_t idx = get_thread_id(); idx < kCopyTileN; idx += kBlockSize)
                {
                    if(idx >= valid_count)
                    {
                        x_lds[stage][idx] = XDataType{};
                    }
                }
                block_sync_lds();
            }

            auto x_lds_view = make_naive_tensor_view<address_space_enum::lds>(
                &x_lds[stage][0],
                make_tuple(number<1>{}, number<kCopyTileN>{}),
                make_tuple(number<kCopyTileN>{}, number<1>{}),
                number<kXCopyVector>{}, number<1>{});
            auto x_copy_window = make_tile_window(
                x_copy_view,
                make_tuple(number<1>{}, number<kCopyTileN>{}),
                {token_b, copy_offset},
                Policy::template MakeBlockCopyTileDistribution<Problem, kXCopyVector>());
            auto x_lds_window = make_tile_window(
                x_lds_view,
                make_tuple(number<1>{}, number<kCopyTileN>{}),
                {0, 0},
                Policy::template MakeBlockCopyTileDistribution<Problem, kXCopyVector>());
            async_load_tile(x_lds_window, x_copy_window);
        };

        prefetch_x_chunk(0, 0);

        bool use_x_scale_lds = false;
        if constexpr(ScaleLayoutTraits<XScaleLayout>::is_block2d)
        {
            constexpr index_t Block_N = ScaleLayoutTraits<XScaleLayout>::block_n;
            constexpr index_t Block_K = ScaleLayoutTraits<XScaleLayout>::block_k;
            const index_t num_scale_blocks = kargs.hidden / Block_K;
            use_x_scale_lds = kargs.p_x_scale != nullptr && num_scale_blocks <= kMaxScaleBlocks;
            if(use_x_scale_lds)
            {
                const auto* ptr = static_cast<const XScaleDataType*>(kargs.p_x_scale);
                const index_t scale_row = token_b / Block_N;
                for(index_t c = get_thread_id(); c < num_scale_blocks; c += kBlockSize)
                {
                    x_scale_lds[c] = type_convert<ComputeDataType>(
                        ptr[scale_row * num_scale_blocks + c]);
                }
            }
        }

        bool use_w_scale_lds = false;
        if constexpr(ScaleLayoutTraits<WScaleLayout>::is_block2d)
        {
            constexpr index_t Block_N = ScaleLayoutTraits<WScaleLayout>::block_n;
            constexpr index_t Block_K = ScaleLayoutTraits<WScaleLayout>::block_k;
            const index_t num_scale_blocks = kargs.hidden / Block_K;
            use_w_scale_lds = kargs.p_w_gate_scale != nullptr && kargs.p_w_up_scale != nullptr &&
                              num_scale_blocks <= kMaxScaleBlocks;
            if(use_w_scale_lds)
            {
                const auto* gate_ptr = static_cast<const WScaleDataType*>(kargs.p_w_gate_scale);
                const auto* up_ptr   = static_cast<const WScaleDataType*>(kargs.p_w_up_scale);
                const index_t scale_row = w_row / Block_N;
                for(index_t c = get_lane_id(); c < num_scale_blocks; c += get_warp_size())
                {
                    w_gate_scale_lds[get_warp_id()][c] = type_convert<ComputeDataType>(
                        gate_ptr[scale_row * num_scale_blocks + c]);
                    w_up_scale_lds[get_warp_id()][c] = type_convert<ComputeDataType>(
                        up_ptr[scale_row * num_scale_blocks + c]);
                }
            }
        }

        block_sync_lds_direct_load();

        ComputeDataType x_scale_val = type_convert<ComputeDataType>(1.0f);
        if constexpr(std::is_same_v<XScaleLayout, WarpDecodeScaleLayout::PerTensor>)
        {
            if(kargs.p_x_scale)
                x_scale_val = type_convert<ComputeDataType>(
                    *static_cast<const XScaleDataType*>(kargs.p_x_scale));
        }
        else if constexpr(std::is_same_v<XScaleLayout, WarpDecodeScaleLayout::PerToken>)
        {
            if(kargs.p_x_scale)
                x_scale_val = type_convert<ComputeDataType>(
                    static_cast<const XScaleDataType*>(kargs.p_x_scale)[token_b]);
        }

        ComputeDataType w_gate_scale_val = type_convert<ComputeDataType>(1.0f);
        ComputeDataType w_up_scale_val   = type_convert<ComputeDataType>(1.0f);
        if constexpr(std::is_same_v<WScaleLayout, WarpDecodeScaleLayout::PerTensor>)
        {
            if(kargs.p_w_gate_scale)
                w_gate_scale_val = type_convert<ComputeDataType>(
                    *static_cast<const WScaleDataType*>(kargs.p_w_gate_scale));
            if(kargs.p_w_up_scale)
                w_up_scale_val = type_convert<ComputeDataType>(
                    *static_cast<const WScaleDataType*>(kargs.p_w_up_scale));
        }

        const auto w_gate_view = make_naive_tensor_view<address_space_enum::global>(
            static_cast<const WDataType*>(kargs.p_w_gate),
            make_tuple(kargs.e * kargs.inter, kargs.hidden),
            make_tuple(kargs.stride_w_gate, 1),
            number<kVector>{}, number<1>{});
        const auto w_up_view = make_naive_tensor_view<address_space_enum::global>(
            static_cast<const WDataType*>(kargs.p_w_up),
            make_tuple(kargs.e * kargs.inter, kargs.hidden),
            make_tuple(kargs.stride_w_up, 1),
            number<kVector>{}, number<1>{});

        auto w_gate_window = make_tile_window(
            w_gate_view,
            make_tuple(number<kWarpsPerBlock>{}, number<kTileN>{}),
            {w_row_base, 0},
            Policy::template MakeOutputTileDistribution<Problem>());
        auto w_up_window = make_tile_window(
            w_up_view,
            make_tuple(number<kWarpsPerBlock>{}, number<kTileN>{}),
            {w_row_base, 0},
            Policy::template MakeOutputTileDistribution<Problem>());

        ComputeDataType gate_acc = 0;
        ComputeDataType up_acc   = 0;
        const index_t num_iterations = kargs.hidden / kTileN;
        const index_t num_copy_chunks = integer_divide_ceil(kargs.hidden, kCopyTileN);

        for(index_t copy_chunk = 0; copy_chunk < num_copy_chunks; ++copy_chunk)
        {
            const index_t chunk_base = copy_chunk * kCopyTileN;
            const index_t next_chunk = copy_chunk + 1;
            if(next_chunk < num_copy_chunks)
            {
                prefetch_x_chunk(next_chunk & 1, next_chunk * kCopyTileN);
            }

            const index_t chunk_iter_begin = chunk_base / kTileN;
            const index_t chunk_iter_end =
                min(num_iterations, (chunk_base + kCopyTileN) / kTileN);
            auto x_lds_compute_view = make_naive_tensor_view<address_space_enum::lds>(
                &x_lds[copy_chunk & 1][0],
                make_tuple(number<1>{}, number<kCopyTileN>{}),
                make_tuple(number<kCopyTileN>{}, number<1>{}),
                number<kVector>{}, number<1>{});
            auto x_lds_window = make_tile_window(
                x_lds_compute_view,
                make_tuple(number<1>{}, number<kTileN>{}),
                {0, chunk_iter_begin * kTileN - chunk_base},
                Policy::template MakeXBroadcastTileDistribution<Problem>());

            for(index_t i = chunk_iter_begin; i < chunk_iter_end; ++i)
            {
                auto x_tile      = load_tile(x_lds_window);
                auto w_gate_tile = load_tile(w_gate_window);
                auto w_up_tile   = load_tile(w_up_window);
                const index_t k_base = i * kTileN + get_lane_id() * kVector;

                ComputeDataType xs = x_scale_val;
                if constexpr(ScaleLayoutTraits<XScaleLayout>::is_block2d)
                {
                    constexpr index_t Block_K = ScaleLayoutTraits<XScaleLayout>::block_k;
                    xs = use_x_scale_lds ? x_scale_lds[k_base / Block_K]
                                          : Base::template load_block2d_scale<XScaleLayout, XScaleDataType>(
                                                kargs.p_x_scale, token_b, k_base, kargs.hidden);
                }

                ComputeDataType gs = w_gate_scale_val;
                ComputeDataType us = w_up_scale_val;
                if constexpr(ScaleLayoutTraits<WScaleLayout>::is_block2d)
                {
                    constexpr index_t Block_K = ScaleLayoutTraits<WScaleLayout>::block_k;
                    gs = use_w_scale_lds ? w_gate_scale_lds[get_warp_id()][k_base / Block_K]
                                          : Base::template load_block2d_scale<WScaleLayout, WScaleDataType>(
                                                kargs.p_w_gate_scale, w_row, k_base, kargs.hidden);
                    us = use_w_scale_lds ? w_up_scale_lds[get_warp_id()][k_base / Block_K]
                                          : Base::template load_block2d_scale<WScaleLayout, WScaleDataType>(
                                                kargs.p_w_up_scale, w_row, k_base, kargs.hidden);
                }

                index_t sub = 0;
                constexpr auto spans = decltype(w_gate_tile)::get_distributed_spans();
                if constexpr(Problem::kUseDot2 && !is_packed_w)
                {
                    static_assert(std::is_same_v<ComputeDataType, float>,
                                  "WarpDecode gate_up LDS dot2 path expects FP32 accumulation.");
                    static_assert(kVector % 2 == 0,
                                  "WarpDecode gate_up LDS dot2 path requires an even vector length.");

                    ComputeDataType gate_dot = 0;
                    ComputeDataType up_dot   = 0;
                    if constexpr(std::is_same_v<XDataType, bf16_t>)
                    {
                        static_for<0, kVector / 2, 1>{}([&](auto ipair) {
                            const uint32_t x_pair =
                                x_tile.get_thread_buffer().template get_as<uint32_t>(ipair);
                            constexpr index_t w_word = ipair.value / 2;
                            constexpr index_t w_sel  = ipair.value % 2;
                            const uint32_t g_pair = Base::template fp8x2_to_bf16x2<w_sel>(
                                w_gate_tile.get_thread_buffer().template get_as<uint32_t>(
                                    number<w_word>{}));
                            const uint32_t u_pair = Base::template fp8x2_to_bf16x2<w_sel>(
                                w_up_tile.get_thread_buffer().template get_as<uint32_t>(
                                    number<w_word>{}));
                            gate_dot = Base::dot2_bf16_packed_add(gate_dot, x_pair, g_pair);
                            up_dot   = Base::dot2_bf16_packed_add(up_dot, x_pair, u_pair);
                        });
                    }
                    else
                    {
                        static_for<0, kVector / 2, 1>{}([&](auto ipair) {
                            constexpr index_t word = ipair.value / 2;
                            constexpr index_t sel  = ipair.value % 2;
                            const uint32_t x_pair = Base::template fp8x2_to_bf16x2<sel>(
                                x_tile.get_thread_buffer().template get_as<uint32_t>(
                                    number<word>{}));
                            const uint32_t g_pair = Base::template fp8x2_to_bf16x2<sel>(
                                w_gate_tile.get_thread_buffer().template get_as<uint32_t>(
                                    number<word>{}));
                            const uint32_t u_pair = Base::template fp8x2_to_bf16x2<sel>(
                                w_up_tile.get_thread_buffer().template get_as<uint32_t>(
                                    number<word>{}));
                            gate_dot = Base::dot2_bf16_packed_add(gate_dot, x_pair, g_pair);
                            up_dot   = Base::dot2_bf16_packed_add(up_dot, x_pair, u_pair);
                        });
                    }
                    gate_acc += gate_dot * (xs * gs);
                    up_acc   += up_dot * (xs * us);
                }
                else
                {
                    sweep_tile_span(spans[number<1>{}], [&](auto idx1) {
                        constexpr auto idx = make_tuple(make_tuple(), idx1);
                        auto x_val = type_convert<ComputeDataType>(x_tile[idx]);

                        ComputeDataType g_val, u_val;
                        if constexpr(is_packed_w)
                        {
                            g_val = Base::unpack_fp4_nibble(static_cast<uint8_t>(w_gate_tile[idx]), sub);
                            u_val = Base::unpack_fp4_nibble(static_cast<uint8_t>(w_up_tile[idx]), sub);
                            sub ^= 1;
                        }
                        else
                        {
                            g_val = type_convert<ComputeDataType>(w_gate_tile[idx]);
                            u_val = type_convert<ComputeDataType>(w_up_tile[idx]);
                            ++sub;
                        }

                        gate_acc += (x_val * xs) * (g_val * gs);
                        up_acc   += (x_val * xs) * (u_val * us);
                    });
                }

                move_tile_window(x_lds_window, {0, kTileN});
                move_tile_window(w_gate_window, {0, kTileN});
                move_tile_window(w_up_window, {0, kTileN});
            }

            if(next_chunk < num_copy_chunks)
            {
                block_sync_lds_direct_load();
            }
        }

        gate_acc = Base::wavefront_reduce_sum(gate_acc);
        up_acc   = Base::wavefront_reduce_sum(up_acc);

        if(get_lane_id() == 0)
        {
            Activation act;
            ComputeDataType out;
            act(out, gate_acc);
            out = out * up_acc;
            static_cast<IntermediateDataType*>(kargs.p_intermediate)
                [(token_b * kargs.top_k + expert_k) * kargs.stride_intermediate + neuron_j] =
                    type_convert<IntermediateDataType>(out);
        }
    }
};

} // namespace ck_tile
