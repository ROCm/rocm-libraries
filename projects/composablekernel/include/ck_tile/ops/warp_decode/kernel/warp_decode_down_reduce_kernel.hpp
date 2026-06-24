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
struct WarpDecodeDownReduceKernel : public WarpDecodeNumeric
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
        return dim3(hargs.b * integer_divide_ceil(hargs.hidden, Problem::kHPerWarp));
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

        // Packed FP4 stores two values per pk_fp4_t, so a weight row spans
        // inter/2 storage elements rather than inter.
        const index_t min_w_stride =
            std::is_same_v<WDataType, pk_fp4_t> ? (kargs.inter / 2) : kargs.inter;
        if(kargs.stride_intermediate < kargs.inter || kargs.stride_w_down < min_w_stride ||
           kargs.stride_y < kargs.hidden)
        {
            return fail("WarpDecodeDownReduceKernel received an invalid row stride.");
        }

        if constexpr(Problem::kLanesPerOutput < get_warp_size())
        {
            if constexpr(Problem::kWarpsPerBlock != 1 || !Problem::kUseDot2 ||
                         Problem::kVector != 16 ||
                         Problem::kHPerWarp * Problem::kLanesPerOutput != get_warp_size())
            {
                return fail("WarpDecodeDownReduceKernel short-INTER path requires one warp, dot2, kVector=16, and full wave coverage.");
            }
            if(kargs.inter != Problem::kLanesPerOutput * Problem::kVector)
            {
                return fail("WarpDecodeDownReduceKernel short-INTER path requires inter == LanesPerOutput * kVector.");
            }
            if(kargs.hidden % Problem::kHPerWarp != 0)
            {
                return fail("WarpDecodeDownReduceKernel short-INTER path requires hidden to be divisible by HPerWarp.");
            }
        }
        else if(kargs.inter % kTileN != 0)
        {
            return fail("WarpDecodeDownReduceKernel requires inter to be divisible by warp_size * kVector.");
        }
        if constexpr(Problem::kHPerWarp != 1 && Problem::kLanesPerOutput == get_warp_size())
        {
            if constexpr(Problem::kHPerWarp != 2 || Problem::kWarpsPerBlock != 1 ||
                         !Problem::kUseDot2)
            {
                return fail("WarpDecodeDownReduceKernel HPerWarp prototype requires HPerWarp=2, one warp per block, and dot2.");
            }
            if(kargs.hidden % Problem::kHPerWarp != 0)
            {
                return fail("WarpDecodeDownReduceKernel HPerWarp prototype requires hidden to be divisible by HPerWarp.");
            }
        }

        if constexpr(Problem::kUseDot2)
        {
            if constexpr(std::is_same_v<WDataType, pk_fp4_t>)
            {
                if constexpr(!std::is_same_v<ComputeDataType, float> ||
                             !std::is_same_v<IntermediateDataType, bf16_t> || kVector % 8 != 0 ||
                             (Problem::kHPerWarp != 1 && Problem::kHPerWarp != 2) ||
                             Problem::kLanesPerOutput != get_warp_size())
                {
                    return fail("WarpDecodeDownReduceKernel FP4 dot2 path requires BF16 intermediate, "
                                "FP32 accumulation, kVector divisible by 8, and the full-wave "
                                "one- or two-output layout.");
                }
            }
            else if constexpr(!std::is_same_v<ComputeDataType, float> || kVector % 2 != 0)
            {
                return fail("WarpDecodeDownReduceKernel dot2 path requires FP32 accumulation and even kVector.");
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

    template <index_t LanesPerOutput>
    CK_TILE_DEVICE static ComputeDataType subgroup_reduce_sum(ComputeDataType val)
    {
        static_assert(LanesPerOutput > 0, "LanesPerOutput must be positive.");
        static_assert((LanesPerOutput & (LanesPerOutput - 1)) == 0,
                      "LanesPerOutput must be a power of two.");
        constexpr index_t num_stages = integer_log2_floor(LanesPerOutput);
        constexpr index_t groups_per_wave = get_warp_size() / LanesPerOutput;
        const index_t group_id = get_lane_id() / LanesPerOutput;
        const index_t lane_in_group = get_lane_id() % LanesPerOutput;
        static_for<0, num_stages, 1>{}([&](auto istage) {
            index_t offset = 1 << istage.value;
            index_t src_lane = group_id * LanesPerOutput + (lane_in_group ^ offset);
            ComputeDataType remote_val = warp_shuffle(val, src_lane);
            val += remote_val;
        });
        static_cast<void>(groups_per_wave);
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

        // Fast packed-FP4 path: full wave, one output per wave, raw 128-bit
        // packed loads + cvt_scalef32_pk_bf16_fp4 + dot2. Each lane owns kVector
        // contiguous FP4 along INTER (kVector/8 u32 words, 4 BF16 pairs each).
        // The (MX) block scale is constant over the lane's chunk for
        // kVector in {8,16,32} with block_k=32, so it is applied after the dot.
        if constexpr(is_packed_w && Problem::kUseDot2 && Problem::kHPerWarp == 1)
        {
            static_assert(std::is_same_v<IntermediateDataType, bf16_t>,
                          "WarpDecodeDownReduceKernel FP4 dot2 path expects BF16 intermediate.");
            static_assert(std::is_same_v<ComputeDataType, float>,
                          "WarpDecodeDownReduceKernel FP4 dot2 path expects FP32 accumulation.");
            static_assert(kVector % 8 == 0,
                          "WarpDecodeDownReduceKernel FP4 dot2 path needs kVector divisible by 8.");
            static_assert(Problem::kWarpsPerBlock == 1,
                          "WarpDecodeDownReduceKernel FP4 dot2 path expects one warp per block.");
            static_assert(Problem::kHPerWarp == 1 && Problem::kLanesPerOutput == get_warp_size(),
                          "WarpDecodeDownReduceKernel FP4 dot2 path is full-wave, one output per wave.");

            constexpr index_t kWordsPerLane = kVector / 8; // u32 words (8 FP4 each)

            const index_t block_id = get_block_id();
            const index_t out_j    = block_id % kargs.hidden;
            const index_t token_b  = block_id / kargs.hidden;

            if(token_b >= kargs.b)
                return;

            const index_t num_iterations = kargs.inter / kTileN;

            ComputeDataType acc = 0;
            ComputeDataType w_down_scale_val = type_convert<ComputeDataType>(1.0f);
            if constexpr(std::is_same_v<WScaleLayout, WarpDecodeScaleLayout::PerTensor>)
            {
                if(kargs.p_w_down_scale)
                    w_down_scale_val = type_convert<ComputeDataType>(
                        *static_cast<const WScaleDataType*>(kargs.p_w_down_scale));
            }

            for(index_t k = 0; k < kargs.top_k; ++k)
            {
                const int32_t e     = kargs.p_router_ids[token_b * kargs.top_k + k];
                const float w       = kargs.p_router_wts[token_b * kargs.top_k + k];
                const index_t w_row = e * kargs.hidden + out_j;

                if constexpr(std::is_same_v<WScaleLayout, WarpDecodeScaleLayout::PerToken>)
                {
                    if(kargs.p_w_down_scale)
                        w_down_scale_val = type_convert<ComputeDataType>(
                            static_cast<const WScaleDataType*>(kargs.p_w_down_scale)[w_row]);
                }

                const auto* w_base = static_cast<const WDataType*>(kargs.p_w_down) +
                                     static_cast<index_t>(w_row) * kargs.stride_w_down;
                const auto* x_base = static_cast<const IntermediateDataType*>(kargs.p_intermediate) +
                                     static_cast<index_t>(token_b * kargs.top_k + k) *
                                         kargs.stride_intermediate;

                for(index_t i = 0; i < num_iterations; ++i)
                {
                    const index_t k_elem = i * kTileN + get_lane_id() * kVector;

                    ComputeDataType ds = w_down_scale_val;
                    if constexpr(ScaleLayoutTraits<WScaleLayout>::is_block2d)
                        ds = load_block2d_scale<WScaleLayout, WScaleDataType>(
                            kargs.p_w_down_scale, w_row, k_elem, kargs.inter);

                    // pk_fp4_t holds two FP4 per byte; the lane's chunk starts at
                    // k_elem/2 bytes and spans kWordsPerLane 32-bit words. Use
                    // memcpy to avoid a strict-aliasing violation on the load.
                    const uint8_t* w_bytes =
                        reinterpret_cast<const uint8_t*>(w_base) + (k_elem >> 1);
                    uint32_t w_words[kWordsPerLane];
                    __builtin_memcpy(w_words, w_bytes, kWordsPerLane * sizeof(uint32_t));

                    // Four independent FP32 accumulators + s_nop-free v_dot2 keep
                    // the dot2 pipeline busy without serializing. dot2 also halves
                    // the conversion work vs pk_fma: only fp4->bf16 is needed (the
                    // activation stays bf16), so the kernel stays bandwidth-bound
                    // and realizes FP4's halved weight read.
                    float dot0 = 0.0f, dot1 = 0.0f, dot2v = 0.0f, dot3 = 0.0f;
                    static_for<0, kWordsPerLane, 1>{}([&](auto iw) {
                        const uint32_t ww   = w_words[iw.value];
                        constexpr index_t b = iw.value * 8;
                        const uint32_t a0 =
                            pack_bf16_pair(x_base[k_elem + b + 0], x_base[k_elem + b + 1]);
                        const uint32_t a1 =
                            pack_bf16_pair(x_base[k_elem + b + 2], x_base[k_elem + b + 3]);
                        const uint32_t a2 =
                            pack_bf16_pair(x_base[k_elem + b + 4], x_base[k_elem + b + 5]);
                        const uint32_t a3 =
                            pack_bf16_pair(x_base[k_elem + b + 6], x_base[k_elem + b + 7]);
                        dot0 = dot2_bf16_packed_raw_nonop(dot0, a0, fp4x2_to_bf16x2<0>(ww));
                        dot1 = dot2_bf16_packed_raw_nonop(dot1, a1, fp4x2_to_bf16x2<1>(ww));
                        dot2v = dot2_bf16_packed_raw_nonop(dot2v, a2, fp4x2_to_bf16x2<2>(ww));
                        dot3 = dot2_bf16_packed_raw_nonop(dot3, a3, fp4x2_to_bf16x2<3>(ww));
                    });
                    dot2_drain4(dot0, dot1, dot2v, dot3);
                    const ComputeDataType dot = (dot0 + dot1) + (dot2v + dot3);
                    acc += dot * (type_convert<ComputeDataType>(w) * ds);
                }
            }

            ComputeDataType result = wavefront_reduce_sum(acc);

            if(get_lane_id() == 0)
            {
                static_cast<YDataType*>(kargs.p_y)[token_b * kargs.stride_y + out_j] =
                    type_convert<YDataType>(result);
            }
            return;
        }

        // Fast packed-FP4 path, two hidden outputs per wave (FP4 analogue of
        // down_h2_d2). The 1-output FP4 path only reaches ~half of HBM peak
        // because each wave keeps too few weight loads in flight; owning two
        // adjacent output rows doubles the outstanding packed-FP4 loads per
        // wave and lets the shared BF16 activation row be loaded once for both.
        if constexpr(is_packed_w && Problem::kUseDot2 && Problem::kHPerWarp == 2)
        {
            static_assert(std::is_same_v<IntermediateDataType, bf16_t>,
                          "WarpDecodeDownReduceKernel FP4 H2 path expects BF16 intermediate.");
            static_assert(std::is_same_v<ComputeDataType, float>,
                          "WarpDecodeDownReduceKernel FP4 H2 path expects FP32 accumulation.");
            static_assert(kVector % 8 == 0,
                          "WarpDecodeDownReduceKernel FP4 H2 path needs kVector divisible by 8.");
            static_assert(Problem::kWarpsPerBlock == 1,
                          "WarpDecodeDownReduceKernel FP4 H2 path expects one warp per block.");
            static_assert(Problem::kLanesPerOutput == get_warp_size(),
                          "WarpDecodeDownReduceKernel FP4 H2 path is full-wave.");

            constexpr index_t kWordsPerLane = kVector / 8;

            const index_t block_id     = get_block_id();
            const index_t hidden_block = block_id % integer_divide_ceil(kargs.hidden, 2);
            const index_t token_b      = block_id / integer_divide_ceil(kargs.hidden, 2);
            const index_t out_j0       = hidden_block * 2;
            const index_t out_j1       = out_j0 + 1;

            if(token_b >= kargs.b || out_j1 >= kargs.hidden)
                return;

            const index_t num_iterations = kargs.inter / kTileN;

            ComputeDataType acc0            = 0;
            ComputeDataType acc1            = 0;
            ComputeDataType w_down_scale_val = type_convert<ComputeDataType>(1.0f);
            if constexpr(std::is_same_v<WScaleLayout, WarpDecodeScaleLayout::PerTensor>)
            {
                if(kargs.p_w_down_scale)
                    w_down_scale_val = type_convert<ComputeDataType>(
                        *static_cast<const WScaleDataType*>(kargs.p_w_down_scale));
            }

            for(index_t k = 0; k < kargs.top_k; ++k)
            {
                const int32_t e      = kargs.p_router_ids[token_b * kargs.top_k + k];
                const float w        = kargs.p_router_wts[token_b * kargs.top_k + k];
                const index_t w_row0 = e * kargs.hidden + out_j0;
                const index_t w_row1 = w_row0 + 1;

                ComputeDataType ws0 = w_down_scale_val;
                ComputeDataType ws1 = w_down_scale_val;
                if constexpr(std::is_same_v<WScaleLayout, WarpDecodeScaleLayout::PerToken>)
                {
                    if(kargs.p_w_down_scale)
                    {
                        const auto* ptr =
                            static_cast<const WScaleDataType*>(kargs.p_w_down_scale);
                        ws0 = type_convert<ComputeDataType>(ptr[w_row0]);
                        ws1 = type_convert<ComputeDataType>(ptr[w_row1]);
                    }
                }

                const auto* w_base0 = static_cast<const WDataType*>(kargs.p_w_down) +
                                      static_cast<index_t>(w_row0) * kargs.stride_w_down;
                const auto* w_base1 = static_cast<const WDataType*>(kargs.p_w_down) +
                                      static_cast<index_t>(w_row1) * kargs.stride_w_down;
                const auto* x_base =
                    static_cast<const IntermediateDataType*>(kargs.p_intermediate) +
                    static_cast<index_t>(token_b * kargs.top_k + k) * kargs.stride_intermediate;

                for(index_t i = 0; i < num_iterations; ++i)
                {
                    const index_t k_elem = i * kTileN + get_lane_id() * kVector;

                    ComputeDataType ds0 = ws0;
                    ComputeDataType ds1 = ws1;
                    if constexpr(ScaleLayoutTraits<WScaleLayout>::is_block2d)
                    {
                        ds0 = load_block2d_scale<WScaleLayout, WScaleDataType>(
                            kargs.p_w_down_scale, w_row0, k_elem, kargs.inter);
                        ds1 = load_block2d_scale<WScaleLayout, WScaleDataType>(
                            kargs.p_w_down_scale, w_row1, k_elem, kargs.inter);
                    }

                    // memcpy avoids a strict-aliasing violation on the packed loads.
                    const uint8_t* w_bytes0 =
                        reinterpret_cast<const uint8_t*>(w_base0) + (k_elem >> 1);
                    const uint8_t* w_bytes1 =
                        reinterpret_cast<const uint8_t*>(w_base1) + (k_elem >> 1);
                    uint32_t w_words0[kWordsPerLane];
                    uint32_t w_words1[kWordsPerLane];
                    __builtin_memcpy(w_words0, w_bytes0, kWordsPerLane * sizeof(uint32_t));
                    __builtin_memcpy(w_words1, w_bytes1, kWordsPerLane * sizeof(uint32_t));

                    // Eight independent FP32 accumulators (4 per output row) keep
                    // both dot2 chains s_nop-free; one drain per row covers the
                    // write->read hazard before the accumulators are summed.
                    float d0r0 = 0.0f, d1r0 = 0.0f, d2r0 = 0.0f, d3r0 = 0.0f;
                    float d0r1 = 0.0f, d1r1 = 0.0f, d2r1 = 0.0f, d3r1 = 0.0f;
                    static_for<0, kWordsPerLane, 1>{}([&](auto iw) {
                        const uint32_t w0   = w_words0[iw.value];
                        const uint32_t w1   = w_words1[iw.value];
                        constexpr index_t b = iw.value * 8;
                        const uint32_t a0 =
                            pack_bf16_pair(x_base[k_elem + b + 0], x_base[k_elem + b + 1]);
                        const uint32_t a1 =
                            pack_bf16_pair(x_base[k_elem + b + 2], x_base[k_elem + b + 3]);
                        const uint32_t a2 =
                            pack_bf16_pair(x_base[k_elem + b + 4], x_base[k_elem + b + 5]);
                        const uint32_t a3 =
                            pack_bf16_pair(x_base[k_elem + b + 6], x_base[k_elem + b + 7]);
                        d0r0  = dot2_bf16_packed_raw_nonop(d0r0, a0, fp4x2_to_bf16x2<0>(w0));
                        d1r0  = dot2_bf16_packed_raw_nonop(d1r0, a1, fp4x2_to_bf16x2<1>(w0));
                        d2r0  = dot2_bf16_packed_raw_nonop(d2r0, a2, fp4x2_to_bf16x2<2>(w0));
                        d3r0  = dot2_bf16_packed_raw_nonop(d3r0, a3, fp4x2_to_bf16x2<3>(w0));
                        d0r1  = dot2_bf16_packed_raw_nonop(d0r1, a0, fp4x2_to_bf16x2<0>(w1));
                        d1r1  = dot2_bf16_packed_raw_nonop(d1r1, a1, fp4x2_to_bf16x2<1>(w1));
                        d2r1  = dot2_bf16_packed_raw_nonop(d2r1, a2, fp4x2_to_bf16x2<2>(w1));
                        d3r1  = dot2_bf16_packed_raw_nonop(d3r1, a3, fp4x2_to_bf16x2<3>(w1));
                    });
                    dot2_drain4(d0r0, d1r0, d2r0, d3r0);
                    dot2_drain4(d0r1, d1r1, d2r1, d3r1);
                    acc0 += ((d0r0 + d1r0) + (d2r0 + d3r0)) *
                            (type_convert<ComputeDataType>(w) * ds0);
                    acc1 += ((d0r1 + d1r1) + (d2r1 + d3r1)) *
                            (type_convert<ComputeDataType>(w) * ds1);
                }
            }

            ComputeDataType result0 = wavefront_reduce_sum(acc0);
            ComputeDataType result1 = wavefront_reduce_sum(acc1);

            if(get_lane_id() == 0)
            {
                auto* y = static_cast<YDataType*>(kargs.p_y) + token_b * kargs.stride_y;
                y[out_j0] = type_convert<YDataType>(result0);
                y[out_j1] = type_convert<YDataType>(result1);
            }
            return;
        }

        if constexpr(Problem::kLanesPerOutput < get_warp_size())
        {
            static_assert(Problem::kWarpsPerBlock == 1,
                          "WarpDecodeDownReduceKernel short-INTER path expects one warp per block.");
            static_assert(Problem::kUseDot2,
                          "WarpDecodeDownReduceKernel short-INTER path expects dot2.");
            static_assert(!is_packed_w,
                          "WarpDecodeDownReduceKernel short-INTER path does not support packed FP4 weights.");
            static_assert(std::is_same_v<IntermediateDataType, bf16_t>,
                          "WarpDecodeDownReduceKernel short-INTER path expects BF16 intermediate.");
            static_assert(std::is_same_v<WDataType, fp8_t>,
                          "WarpDecodeDownReduceKernel short-INTER path expects FP8 weights.");
            static_assert(std::is_same_v<ComputeDataType, float>,
                          "WarpDecodeDownReduceKernel short-INTER path expects FP32 accumulation.");
            static_assert(Problem::kVector == 16,
                          "WarpDecodeDownReduceKernel short-INTER path preserves 128-bit loads.");
            static_assert(Problem::kHPerWarp * Problem::kLanesPerOutput == get_warp_size(),
                          "WarpDecodeDownReduceKernel short-INTER path expects HPerWarp * lanes = wave size.");

            constexpr index_t kLanesPerOutput = Problem::kLanesPerOutput;
            constexpr index_t kOutputsPerWave = Problem::kHPerWarp;
            constexpr index_t kShortInter = kLanesPerOutput * kVector;

            const index_t block_id = get_block_id();
            const index_t hidden_block = block_id % integer_divide_ceil(kargs.hidden, kOutputsPerWave);
            const index_t token_b = block_id / integer_divide_ceil(kargs.hidden, kOutputsPerWave);
            const index_t group_id = get_lane_id() / kLanesPerOutput;
            const index_t lane_in_group = get_lane_id() - group_id * kLanesPerOutput;
            const index_t out_j = hidden_block * kOutputsPerWave + group_id;

            if(token_b >= kargs.b || out_j >= kargs.hidden)
                return;
            if(kargs.inter != kShortInter)
                return;

            ComputeDataType acc = 0;
            ComputeDataType w_down_scale_val = type_convert<ComputeDataType>(1.0f);
            if constexpr(std::is_same_v<WScaleLayout, WarpDecodeScaleLayout::PerTensor>)
            {
                if(kargs.p_w_down_scale)
                    w_down_scale_val = type_convert<ComputeDataType>(
                        *static_cast<const WScaleDataType*>(kargs.p_w_down_scale));
            }

            for(index_t k = 0; k < kargs.top_k; ++k)
            {
                const int32_t e = kargs.p_router_ids[token_b * kargs.top_k + k];
                const float w   = kargs.p_router_wts[token_b * kargs.top_k + k];
                const index_t w_row = e * kargs.hidden + out_j;

                if constexpr(std::is_same_v<WScaleLayout, WarpDecodeScaleLayout::PerToken>)
                {
                    if(kargs.p_w_down_scale)
                        w_down_scale_val = type_convert<ComputeDataType>(
                            static_cast<const WScaleDataType*>(kargs.p_w_down_scale)[w_row]);
                }

                const auto* inter_ptr = static_cast<const IntermediateDataType*>(kargs.p_intermediate) +
                                        (token_b * kargs.top_k + k) * kargs.stride_intermediate +
                                        lane_in_group * kVector;
                const auto* w_ptr = static_cast<const WDataType*>(kargs.p_w_down) +
                                    w_row * kargs.stride_w_down + lane_in_group * kVector;

                ComputeDataType ds = w_down_scale_val;
                if constexpr(ScaleLayoutTraits<WScaleLayout>::is_block2d)
                {
                    ds = load_block2d_scale<WScaleLayout, WScaleDataType>(
                        kargs.p_w_down_scale,
                        w_row,
                        lane_in_group * kVector,
                        kargs.inter);
                }

                ComputeDataType dot = 0;
                static_for<0, kVector / 2, 1>{}([&](auto ipair) {
                    constexpr index_t elem0 = 2 * ipair.value;
                    constexpr index_t elem1 = elem0 + 1;
                    constexpr index_t w_word = ipair.value / 2;
                    constexpr index_t w_sel  = ipair.value % 2;
                    const uint32_t act_pair = pack_bf16_pair(inter_ptr[elem0], inter_ptr[elem1]);
                    const uint32_t d_word =
                        reinterpret_cast<const uint32_t*>(w_ptr)[w_word];
                    const uint32_t d_pair = fp8x2_to_bf16x2<w_sel>(d_word);
                    dot = dot2_bf16_packed_add(dot, act_pair, d_pair);
                });
                acc += dot * (type_convert<ComputeDataType>(w) * ds);
            }

            ComputeDataType result = subgroup_reduce_sum<kLanesPerOutput>(acc);

            if(lane_in_group == 0)
            {
                static_cast<YDataType*>(kargs.p_y)[token_b * kargs.stride_y + out_j] =
                    type_convert<YDataType>(result);
            }
            return;
        }

        if constexpr(Problem::kHPerWarp == 2 && !is_packed_w)
        {
            static_assert(Problem::kWarpsPerBlock == 1,
                          "WarpDecodeDownReduceKernel HPerWarp=2 expects one warp per block.");
            static_assert(Problem::kUseDot2,
                          "WarpDecodeDownReduceKernel HPerWarp=2 expects the dot2 path.");

            constexpr index_t kHPerWarp = Problem::kHPerWarp;
            const index_t block_id = get_block_id();
            const index_t hidden_block = block_id % integer_divide_ceil(kargs.hidden, kHPerWarp);
            const index_t token_b = block_id / integer_divide_ceil(kargs.hidden, kHPerWarp);
            const index_t out_j0 = hidden_block * kHPerWarp;
            const index_t out_j1 = out_j0 + 1;

            if(token_b >= kargs.b || out_j1 >= kargs.hidden)
                return;

            ComputeDataType acc0 = 0;
            ComputeDataType acc1 = 0;

            ComputeDataType w_down_scale_val0 = type_convert<ComputeDataType>(1.0f);
            ComputeDataType w_down_scale_val1 = type_convert<ComputeDataType>(1.0f);
            if constexpr(std::is_same_v<WScaleLayout, WarpDecodeScaleLayout::PerTensor>)
            {
                if(kargs.p_w_down_scale)
                {
                    w_down_scale_val0 = type_convert<ComputeDataType>(
                        *static_cast<const WScaleDataType*>(kargs.p_w_down_scale));
                    w_down_scale_val1 = w_down_scale_val0;
                }
            }

            const auto intermediate_view = make_naive_tensor_view<address_space_enum::global>(
                static_cast<const IntermediateDataType*>(kargs.p_intermediate),
                make_tuple(kargs.b * kargs.top_k, kargs.inter),
                make_tuple(kargs.stride_intermediate, 1),
                number<kVector>{}, number<1>{});
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
                const index_t w_row0 = e * kargs.hidden + out_j0;
                const index_t w_row1 = w_row0 + 1;

                if constexpr(std::is_same_v<WScaleLayout, WarpDecodeScaleLayout::PerToken>)
                {
                    if(kargs.p_w_down_scale)
                    {
                        const auto* ptr = static_cast<const WScaleDataType*>(kargs.p_w_down_scale);
                        w_down_scale_val0 = type_convert<ComputeDataType>(ptr[w_row0]);
                        w_down_scale_val1 = type_convert<ComputeDataType>(ptr[w_row1]);
                    }
                }

                auto intermediate_window = make_tile_window(
                    intermediate_view,
                    make_tuple(number<1>{}, number<kTileN>{}),
                    {token_b * kargs.top_k + k, 0},
                    Policy::template MakeXBroadcastTileDistribution<Problem>());
                auto w_down_window0 = make_tile_window(
                    w_down_view,
                    make_tuple(number<1>{}, number<kTileN>{}),
                    {w_row0, 0},
                    Policy::template MakeOutputTileDistribution<Problem>());
                auto w_down_window1 = make_tile_window(
                    w_down_view,
                    make_tuple(number<1>{}, number<kTileN>{}),
                    {w_row1, 0},
                    Policy::template MakeOutputTileDistribution<Problem>());

                for(index_t i = 0; i < num_iterations; ++i)
                {
                    auto inter_tile   = load_tile(intermediate_window);
                    auto w_down_tile0 = load_tile(w_down_window0);
                    auto w_down_tile1 = load_tile(w_down_window1);

                    const index_t k_base = i * kTileN + get_lane_id() * kVector;

                    ComputeDataType ds0 = w_down_scale_val0;
                    ComputeDataType ds1 = w_down_scale_val1;
                    if constexpr(ScaleLayoutTraits<WScaleLayout>::is_block2d)
                    {
                        ds0 = load_block2d_scale<WScaleLayout, WScaleDataType>(
                            kargs.p_w_down_scale, w_row0, k_base, kargs.inter);
                        ds1 = load_block2d_scale<WScaleLayout, WScaleDataType>(
                            kargs.p_w_down_scale, w_row1, k_base, kargs.inter);
                    }

                    ComputeDataType dot0 = 0;
                    ComputeDataType dot1 = 0;
                    static_for<0, kVector / 2, 1>{}([&](auto ipair) {
                        const uint32_t act_pair =
                            inter_tile.get_thread_buffer().template get_as<uint32_t>(ipair);
                        constexpr index_t w_word = ipair.value / 2;
                        constexpr index_t w_sel  = ipair.value % 2;
                        const uint32_t d_pair0 = fp8x2_to_bf16x2<w_sel>(
                            w_down_tile0.get_thread_buffer().template get_as<uint32_t>(
                                number<w_word>{}));
                        const uint32_t d_pair1 = fp8x2_to_bf16x2<w_sel>(
                            w_down_tile1.get_thread_buffer().template get_as<uint32_t>(
                                number<w_word>{}));
                        dot0 = dot2_bf16_packed_add(dot0, act_pair, d_pair0);
                        dot1 = dot2_bf16_packed_add(dot1, act_pair, d_pair1);
                    });
                    acc0 += dot0 * (type_convert<ComputeDataType>(w) * ds0);
                    acc1 += dot1 * (type_convert<ComputeDataType>(w) * ds1);

                    move_tile_window(intermediate_window, {0, kTileN});
                    move_tile_window(w_down_window0, {0, kTileN});
                    move_tile_window(w_down_window1, {0, kTileN});
                }
            }

            ComputeDataType result0 = wavefront_reduce_sum(acc0);
            ComputeDataType result1 = wavefront_reduce_sum(acc1);

            if(get_lane_id() == 0)
            {
                auto* y = static_cast<YDataType*>(kargs.p_y) + token_b * kargs.stride_y;
                y[out_j0] = type_convert<YDataType>(result0);
                y[out_j1] = type_convert<YDataType>(result1);
            }
            return;
        }

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
                Policy::template MakeXBroadcastTileDistribution<Problem>());

            auto w_down_window = make_tile_window(
                w_down_view,
                make_tuple(number<1>{}, number<kTileN>{}),
                {w_row, 0},
                Policy::template MakeOutputTileDistribution<Problem>());

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

template <typename Problem_, typename Policy_>
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

    static constexpr index_t kWarpsPerBlock = Problem::kWarpsPerBlock;
    static constexpr index_t kBlockSize     = Problem::kBlockSize;
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
        const index_t out_block_base = hidden_block * kWarpsPerBlock;

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
                    Policy::template MakeBlockCopyTileDistribution<Problem, kInterCopyVector>());
                auto inter_lds_window = make_tile_window(
                    inter_lds_view,
                    make_tuple(number<1>{}, number<kCopyTileN>{}),
                    {0, copy_offset},
                    Policy::template MakeBlockCopyTileDistribution<Problem, kInterCopyVector>());
                async_load_tile(inter_lds_window, inter_copy_window);
            }
            block_sync_lds_direct_load();

            const int32_t e = kargs.p_router_ids[token_b * kargs.top_k + k];
            const float w   = kargs.p_router_wts[token_b * kargs.top_k + k];
            const index_t w_row = e * kargs.hidden + out_j;
            const index_t w_row_base = e * kargs.hidden + out_block_base;

            if constexpr(std::is_same_v<WScaleLayout, WarpDecodeScaleLayout::PerToken>)
            {
                if(kargs.p_w_down_scale)
                    w_down_scale_val = type_convert<ComputeDataType>(
                        static_cast<const WScaleDataType*>(kargs.p_w_down_scale)[w_row]);
            }

            auto w_down_window = make_tile_window(
                w_down_view,
                make_tuple(number<kWarpsPerBlock>{}, number<kTileN>{}),
                {w_row_base, 0},
                Policy::template MakeOutputTileDistribution<Problem>());

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
                        const uint32_t act_pair =
                            Base::pack_bf16_pair(inter_lds[k_base + s0], inter_lds[k_base + s1]);
                        const uint32_t d_pair = Base::template fp8x2_to_bf16x2<w_sel>(
                            w_down_tile.get_thread_buffer().template get_as<uint32_t>(
                                number<w_word>{}));
                        dot = Base::dot2_bf16_packed_add(dot, act_pair, d_pair);
                    });
                    acc += dot * (type_convert<ComputeDataType>(w) * ds);
                }
                else if constexpr(Problem::kUsePackedFp32 && !is_packed_w)
                {
                    static_assert(std::is_same_v<ComputeDataType, float>,
                                  "WarpDecode down_reduce LDS packed-FP32 path expects FP32 accumulation.");
                    static_assert(std::is_same_v<IntermediateDataType, bf16_t>,
                                  "WarpDecode down_reduce LDS packed-FP32 path currently expects BF16 intermediate.");
                    static_assert(std::is_same_v<WDataType, fp8_t>,
                                  "WarpDecode down_reduce LDS packed-FP32 path currently expects FP8 weights.");
                    static_assert(kVector % 2 == 0,
                                  "WarpDecode down_reduce LDS packed-FP32 path requires an even vector length.");

                    fp32x2_t dot{0.0f, 0.0f};
                    static_for<0, kVector / 2, 1>{}([&](auto ipair) {
                        constexpr index_t s0     = 2 * ipair.value;
                        constexpr index_t s1     = s0 + 1;
                        constexpr index_t w_word = ipair.value / 2;
                        constexpr index_t w_sel  = ipair.value % 2;
                        const fp32x2_t act_pair = Base::bf16x2_to_f32x2(Base::pack_bf16_pair(
                            inter_lds[k_base + s0], inter_lds[k_base + s1]));
                        const fp32x2_t d_pair = Base::template fp8x2_to_f32x2<w_sel>(
                            w_down_tile.get_thread_buffer().template get_as<uint32_t>(
                                number<w_word>{}));
                        dot = Base::pk_fma_f32(dot, act_pair, d_pair);
                    });
                    acc += Base::horizontal_add(dot) * (type_convert<ComputeDataType>(w) * ds);
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
