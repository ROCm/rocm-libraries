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

        if constexpr(Problem::kUseDot2)
        {
            if constexpr(std::is_same_v<WDataType, pk_fp4_t> || !std::is_same_v<ComputeDataType, float> ||
                         kVector % 2 != 0)
            {
                return fail("WarpDecodeDownReduceKernel dot2 path requires unpacked weights, FP32 accumulation, and even kVector.");
            }
        }
        if constexpr(Problem::kUsePackedFp32)
        {
            if constexpr(!std::is_same_v<IntermediateDataType, bf16_t> ||
                         !std::is_same_v<WDataType, fp8_t> ||
                         !std::is_same_v<ComputeDataType, float> || kVector % 2 != 0)
            {
                return fail("WarpDecodeDownReduceKernel packed-FP32 path requires BF16 intermediate, FP8 weights, FP32 accumulation, and even kVector.");
            }
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

    CK_TILE_DEVICE static uint32_t pack_bf16_pair(bf16_t lo, bf16_t hi)
    {
        const uint32_t lo_bits = static_cast<uint32_t>(bit_cast<bf16_raw_t>(lo));
        const uint32_t hi_bits = static_cast<uint32_t>(bit_cast<bf16_raw_t>(hi));
        return lo_bits | (hi_bits << 16);
    }

    template <typename T>
    CK_TILE_DEVICE static bf16_t as_bf16_dot_operand(T x)
    {
        if constexpr(std::is_same_v<remove_cvref_t<T>, bf16_t>)
        {
            return x;
        }
        else
        {
            return type_convert<bf16_t>(type_convert<float>(x));
        }
    }

    CK_TILE_DEVICE static ComputeDataType dot2_bf16_scaled_add(
        ComputeDataType acc, bf16_t a0, bf16_t a1, bf16_t b0, bf16_t b1, ComputeDataType scale)
    {
        float dot = 0.0f;
        const uint32_t a = pack_bf16_pair(a0, a1);
        const uint32_t b = pack_bf16_pair(b0, b1);
        asm volatile("v_dot2_f32_bf16 %0, %1, %2, %0" : "+v"(dot) : "v"(a), "v"(b));
        return acc + type_convert<ComputeDataType>(dot) * scale;
    }

    CK_TILE_DEVICE static ComputeDataType dot2_bf16_add(
        ComputeDataType acc, bf16_t a0, bf16_t a1, bf16_t b0, bf16_t b1)
    {
        float dot = type_convert<float>(acc);
        const uint32_t a = pack_bf16_pair(a0, a1);
        const uint32_t b = pack_bf16_pair(b0, b1);
        asm volatile("v_dot2_f32_bf16 %0, %1, %2, %0" : "+v"(dot) : "v"(a), "v"(b));
        return type_convert<ComputeDataType>(dot);
    }

    CK_TILE_DEVICE static ComputeDataType dot2_bf16_packed_lhs_add(
        ComputeDataType acc, uint32_t a, bf16_t b0, bf16_t b1)
    {
        float dot = type_convert<float>(acc);
        const uint32_t b = pack_bf16_pair(b0, b1);
        asm volatile("v_dot2_f32_bf16 %0, %1, %2, %0" : "+v"(dot) : "v"(a), "v"(b));
        return type_convert<ComputeDataType>(dot);
    }

    CK_TILE_DEVICE static ComputeDataType dot2_bf16_packed_add(
        ComputeDataType acc, uint32_t a, uint32_t b)
    {
        float dot = type_convert<float>(acc);
        asm volatile("v_dot2_f32_bf16 %0, %1, %2, %0" : "+v"(dot) : "v"(a), "v"(b));
        return type_convert<ComputeDataType>(dot);
    }

    template <index_t PairInWord>
    CK_TILE_DEVICE static uint32_t fp8x2_to_bf16x2(uint32_t fp8x4)
    {
        static_assert(PairInWord == 0 || PairInWord == 1);
#if defined(__gfx950__)
        union
        {
            bf16x2_t vec;
            uint32_t raw;
        } out;
        out.vec = __builtin_amdgcn_cvt_scalef32_pk_bf16_fp8(
            fp8x4, type_convert<float>(1.0f), PairInWord);
        return out.raw;
#else
        (void)fp8x4;
        return 0;
#endif
    }

    template <index_t PairInWord>
    CK_TILE_DEVICE static fp32x2_t fp8x2_to_f32x2(uint32_t fp8x4)
    {
        static_assert(PairInWord == 0 || PairInWord == 1);
#if defined(__gfx950__)
        return __builtin_amdgcn_cvt_pk_f32_fp8(fp8x4, PairInWord);
#else
        (void)fp8x4;
        return fp32x2_t{0.0f, 0.0f};
#endif
    }

    CK_TILE_DEVICE static fp32x2_t bf16x2_to_f32x2(uint32_t bf16x2)
    {
        const uint32_t lo = (bf16x2 & 0x0000ffffu) << 16;
        const uint32_t hi = bf16x2 & 0xffff0000u;
        return fp32x2_t{bit_cast<float>(lo), bit_cast<float>(hi)};
    }

    CK_TILE_DEVICE static fp32x2_t pk_fma_f32(fp32x2_t acc, fp32x2_t a, fp32x2_t b)
    {
#if defined(__gfx950__)
        fp32x2_t out;
        asm volatile("v_pk_fma_f32 %[out], %[a], %[b], %[acc]"
                     : [out] "=v"(out)
                     : [a] "v"(a), [b] "v"(b), [acc] "v"(acc));
        return out;
#else
        return acc + a * b;
#endif
    }

    CK_TILE_DEVICE static ComputeDataType horizontal_add(fp32x2_t v)
    {
        return type_convert<ComputeDataType>(v[0] + v[1]);
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
                if constexpr(Problem::kUseDot2 && !is_packed_w)
                {
                    static_assert(std::is_same_v<ComputeDataType, float>,
                                  "WarpDecode down_reduce dot2 path expects FP32 accumulation.");
                    static_assert(kVector % 2 == 0,
                                  "WarpDecode down_reduce dot2 path requires an even vector length.");

                    ComputeDataType dot = 0;
                    if constexpr(std::is_same_v<IntermediateDataType, bf16_t>)
                    {
                        static_for<0, kVector / 2, 1>{}([&](auto ipair) {
                            const uint32_t act_pair =
                                inter_tile.get_thread_buffer().template get_as<uint32_t>(ipair);
                            constexpr index_t w_word = ipair.value / 2;
                            constexpr index_t w_sel  = ipair.value % 2;
                            const uint32_t d_pair = fp8x2_to_bf16x2<w_sel>(
                                w_down_tile.get_thread_buffer().template get_as<uint32_t>(
                                    number<w_word>{}));
                            dot = dot2_bf16_packed_add(dot, act_pair, d_pair);
                        });
                    }
                    else
                    {
                        bf16_t act_bf0, d_bf0;
                        sweep_tile_span(spans[number<1>{}], [&](auto idx1) {
                            constexpr auto idx = make_tuple(make_tuple(), idx1);
                            const auto act_bf = as_bf16_dot_operand(inter_tile[idx]);
                            const auto d_bf   = as_bf16_dot_operand(w_down_tile[idx]);

                            if((sub & 1) == 0)
                            {
                                act_bf0 = act_bf;
                                d_bf0   = d_bf;
                            }
                            else
                            {
                                dot = dot2_bf16_add(dot, act_bf0, act_bf, d_bf0, d_bf);
                            }
                            ++sub;
                        });
                    }
                    acc += dot * (type_convert<ComputeDataType>(w) * ds);
                }
                else if constexpr(Problem::kUsePackedFp32 && !is_packed_w)
                {
                    static_assert(std::is_same_v<ComputeDataType, float>,
                                  "WarpDecode down_reduce packed-FP32 path expects FP32 accumulation.");
                    static_assert(std::is_same_v<IntermediateDataType, bf16_t>,
                                  "WarpDecode down_reduce packed-FP32 path currently expects BF16 intermediate.");
                    static_assert(std::is_same_v<WDataType, fp8_t>,
                                  "WarpDecode down_reduce packed-FP32 path currently expects FP8 weights.");
                    static_assert(kVector % 2 == 0,
                                  "WarpDecode down_reduce packed-FP32 path requires an even vector length.");

                    fp32x2_t dot{0.0f, 0.0f};
                    static_for<0, kVector / 2, 1>{}([&](auto ipair) {
                        const fp32x2_t act_pair = bf16x2_to_f32x2(
                            inter_tile.get_thread_buffer().template get_as<uint32_t>(ipair));
                        constexpr index_t w_word = ipair.value / 2;
                        constexpr index_t w_sel  = ipair.value % 2;
                        const fp32x2_t d_pair = fp8x2_to_f32x2<w_sel>(
                            w_down_tile.get_thread_buffer().template get_as<uint32_t>(
                                number<w_word>{}));
                        dot = pk_fma_f32(dot, act_pair, d_pair);
                    });
                    acc += horizontal_add(dot) * (type_convert<ComputeDataType>(w) * ds);
                }
                else
                {
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
                }

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

template <typename Problem_, typename Policy_, index_t kWarpsPerBlock_ = 4>
struct WarpDecodeDownReduceLdsInterKernel : public WarpDecodeDownReduceKernel<Problem_, Policy_>
{
    using Base    = WarpDecodeDownReduceKernel<Problem_, Policy_>;
    using Problem = remove_cvref_t<Problem_>;
    using Policy  = remove_cvref_t<Policy_>;

    using IntermediateDataType = typename Problem::IntermediateDataType;
    using WDataType            = typename Problem::WDataType;
    using ComputeDataType      = typename Problem::ComputeDataType;
    using YDataType            = typename Problem::YDataType;
    using WScaleDataType       = typename Problem::WScaleDataType;

    using Kargs = typename Base::Kargs;

    static constexpr index_t kWarpsPerBlock = kWarpsPerBlock_;
    static constexpr index_t kBlockSize     = kWarpsPerBlock * get_warp_size();
    static constexpr index_t kMaxInter      = 4096;

    CK_TILE_HOST static constexpr auto GridSize(const Kargs& hargs)
    {
        return dim3(hargs.b * integer_divide_ceil(hargs.hidden, kWarpsPerBlock));
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
        if(kargs.inter > kMaxInter || kargs.hidden % kWarpsPerBlock != 0)
        {
            return false;
        }
        return true;
    }

    template <index_t CopyVector>
    CK_TILE_DEVICE static constexpr auto MakeBlockCopyTileDistribution()
    {
        return make_static_tile_distribution(
            tile_distribution_encoding<
                sequence<>,
                tuple<
                    sequence<1>,
                    sequence<kWarpsPerBlock, get_warp_size(), CopyVector>
                >,
                tuple<sequence<2>, sequence<2>>,
                tuple<sequence<0>, sequence<1>>,
                sequence<2>,
                sequence<2>
            >{});
    }

    template <typename ProblemT>
    CK_TILE_DEVICE static constexpr auto MakePerWarpTileDistribution()
    {
        constexpr index_t V = ProblemT::kVector;
        return make_static_tile_distribution(
            tile_distribution_encoding<
                sequence<>,
                tuple<
                    sequence<1>,
                    sequence<get_warp_size(), V>
                >,
                tuple<sequence<2>>,
                tuple<sequence<0>>,
                sequence<2>,
                sequence<1>
            >{});
    }

    CK_TILE_DEVICE void operator()(Kargs kargs) const
    {
        constexpr bool is_packed_w = std::is_same_v<WDataType, pk_fp4_t>;
        constexpr index_t kVector  = Problem::kVector;
        constexpr index_t kTileN   = get_warp_size() * kVector;

        using WScaleLayout = typename Problem::WScaleLayout;

        const index_t block_id = get_block_id();
        const index_t hidden_block = block_id % integer_divide_ceil(kargs.hidden, kWarpsPerBlock);
        const index_t token_b = block_id / integer_divide_ceil(kargs.hidden, kWarpsPerBlock);
        const index_t out_j = hidden_block * kWarpsPerBlock + get_warp_id();

        if(token_b >= kargs.b || out_j >= kargs.hidden)
            return;

        alignas(16) __shared__ IntermediateDataType inter_lds[kMaxInter];

        ComputeDataType acc = 0;
        ComputeDataType w_down_scale_val = type_convert<ComputeDataType>(1.0f);
        if constexpr(std::is_same_v<WScaleLayout, WarpDecodeScaleLayout::PerTensor>)
        {
            if(kargs.p_w_down_scale)
                w_down_scale_val = type_convert<ComputeDataType>(
                    *static_cast<const WScaleDataType*>(kargs.p_w_down_scale));
        }

        const auto w_down_view = make_naive_tensor_view<address_space_enum::global>(
            static_cast<const WDataType*>(kargs.p_w_down),
            make_tuple(kargs.e * kargs.hidden, kargs.inter),
            make_tuple(kargs.stride_w_down, 1),
            number<kVector>{}, number<1>{});

        const index_t num_iterations = kargs.inter / kTileN;

        for(index_t k = 0; k < kargs.top_k; ++k)
        {
            constexpr index_t kInterCopyVector =
                (sizeof(IntermediateDataType) * kVector /
                     numeric_traits<IntermediateDataType>::PackedSize <=
                 16)
                    ? kVector
                    : (16 * numeric_traits<IntermediateDataType>::PackedSize /
                       sizeof(IntermediateDataType));
            constexpr index_t kCopyTileN = kBlockSize * kInterCopyVector;
            const auto inter_copy_base_view = make_naive_tensor_view<address_space_enum::global>(
                static_cast<const IntermediateDataType*>(kargs.p_intermediate),
                make_tuple(kargs.b * kargs.top_k, kargs.inter),
                make_tuple(kargs.stride_intermediate, 1),
                number<kInterCopyVector>{}, number<1>{});
            const auto inter_copy_view = pad_tensor_view(
                inter_copy_base_view,
                make_tuple(number<1>{}, number<kCopyTileN>{}),
                sequence<false, true>{});
            auto inter_lds_view = make_naive_tensor_view<address_space_enum::lds>(
                inter_lds,
                make_tuple(number<1>{}, number<kMaxInter>{}),
                make_tuple(number<kMaxInter>{}, number<1>{}),
                number<kInterCopyVector>{}, number<1>{});

            for(index_t copy_offset = 0; copy_offset < kargs.inter; copy_offset += kCopyTileN)
            {
                const index_t valid_count = min(kCopyTileN, kargs.inter - copy_offset);
                if(valid_count < kCopyTileN)
                {
                    for(index_t idx = get_thread_id(); idx < kCopyTileN; idx += kBlockSize)
                    {
                        if(idx >= valid_count && copy_offset + idx < kMaxInter)
                        {
                            inter_lds[copy_offset + idx] = IntermediateDataType{};
                        }
                    }
                    block_sync_lds();
                }

                auto inter_copy_window = make_tile_window(
                    inter_copy_view,
                    make_tuple(number<1>{}, number<kCopyTileN>{}),
                    {token_b * kargs.top_k + k, copy_offset},
                    MakeBlockCopyTileDistribution<kInterCopyVector>());
                auto inter_lds_window = make_tile_window(
                    inter_lds_view,
                    make_tuple(number<1>{}, number<kCopyTileN>{}),
                    {0, copy_offset},
                    MakeBlockCopyTileDistribution<kInterCopyVector>());
                async_load_tile(inter_lds_window, inter_copy_window);
            }
            block_sync_lds_direct_load();

            const int32_t e = kargs.p_router_ids[token_b * kargs.top_k + k];
            const float w   = kargs.p_router_wts[token_b * kargs.top_k + k];
            const index_t w_row = e * kargs.hidden + out_j;

            if constexpr(std::is_same_v<WScaleLayout, WarpDecodeScaleLayout::PerToken>)
            {
                if(kargs.p_w_down_scale)
                    w_down_scale_val = type_convert<ComputeDataType>(
                        static_cast<const WScaleDataType*>(kargs.p_w_down_scale)[w_row]);
            }

            auto w_down_window = make_tile_window(
                w_down_view,
                make_tuple(number<1>{}, number<kTileN>{}),
                {w_row, 0},
                MakePerWarpTileDistribution<Problem>());

            for(index_t i = 0; i < num_iterations; ++i)
            {
                auto w_down_tile = load_tile(w_down_window);
                const index_t k_base = i * kTileN + get_lane_id() * kVector;

                ComputeDataType ds = w_down_scale_val;
                if constexpr(ScaleLayoutTraits<WScaleLayout>::is_block2d)
                    ds = Base::template load_block2d_scale<WScaleLayout, WScaleDataType>(
                        kargs.p_w_down_scale, w_row, k_base, kargs.inter);

                index_t sub = 0;
                constexpr auto spans = decltype(w_down_tile)::get_distributed_spans();
                if constexpr(Problem::kUseDot2 && !is_packed_w)
                {
                    static_assert(std::is_same_v<ComputeDataType, float>,
                                  "WarpDecode down_reduce LDS dot2 path expects FP32 accumulation.");
                    static_assert(kVector % 2 == 0,
                                  "WarpDecode down_reduce LDS dot2 path requires an even vector length.");

                    ComputeDataType dot = 0;
                    static_for<0, kVector / 2, 1>{}([&](auto ipair) {
                        constexpr index_t s0     = 2 * ipair.value;
                        constexpr index_t s1     = s0 + 1;
                        constexpr index_t w_word = ipair.value / 2;
                        constexpr index_t w_sel  = ipair.value % 2;
                        const uint32_t act_pair = Base::pack_bf16_pair(
                            Base::as_bf16_dot_operand(inter_lds[k_base + s0]),
                            Base::as_bf16_dot_operand(inter_lds[k_base + s1]));
                        const uint32_t d_pair = Base::template fp8x2_to_bf16x2<w_sel>(
                            w_down_tile.get_thread_buffer().template get_as<uint32_t>(
                                number<w_word>{}));
                        dot = Base::dot2_bf16_packed_add(dot, act_pair, d_pair);
                    });
                    acc += dot * (type_convert<ComputeDataType>(w) * ds);
                }
                else
                {
                    sweep_tile_span(spans[number<1>{}], [&](auto idx1) {
                        constexpr auto idx = make_tuple(make_tuple(), idx1);
                        auto act_val = type_convert<ComputeDataType>(inter_lds[k_base + sub]);

                        ComputeDataType d_val;
                        if constexpr(is_packed_w)
                        {
                            d_val = Base::unpack_fp4_nibble(static_cast<uint8_t>(w_down_tile[idx]), sub);
                            sub ^= 1;
                        }
                        else
                        {
                            d_val = type_convert<ComputeDataType>(w_down_tile[idx]);
                            ++sub;
                        }

                        acc += w * act_val * (d_val * ds);
                    });
                }

                move_tile_window(w_down_window, {0, kTileN});
            }

            block_sync_lds();
        }

        ComputeDataType result = Base::wavefront_reduce_sum(acc);

        if(get_lane_id() == 0)
        {
            static_cast<YDataType*>(kargs.p_y)[token_b * kargs.stride_y + out_j] =
                type_convert<YDataType>(result);
        }
    }
};

} // namespace ck_tile
