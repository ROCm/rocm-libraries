// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"
#include "ck_tile/ops/gemm_decode/kernel/gemm_decode_numeric.hpp"
#include "ck_tile/ops/gemm_decode/pipeline/gemm_decode_problem.hpp"
#include "ck_tile/ops/gemm_decode/pipeline/gemm_decode_policy.hpp"

namespace ck_tile {

// Warp-per-scalar dense GEMM kernel: each wavefront computes one scalar of
// C = A * B^T (SmallM orientation) or, when kNPerWarp > 1, kNPerWarp adjacent
// columns of one row of C. Adapted from the gate/up form in
// ops/warp_decode/kernel/warp_decode_gate_up_kernel.hpp with expert routing,
// gate/up duality, SwiGLU, and the Block2D scale path removed.
//
// Supported scale subconfigurations (selected at compile time via
// (XScaleLayout, WScaleLayout)):
//   - (void, void)               unscaled BF16 or FP16              (P0)
//   - (PerTensor, PerTensor)     per-tensor FP8 (FP32 scale scalars) (P0b)
//
// Other constraints:
//   - kOutputAxis = SmallM
//   - kMPerWarp = kNPerWarp = 1
//   - kWarpsPerBlock = 1
//   - kBPreshuffle = false                (P4 hook reserved)
//   - kHasBias is honoured: when true, the [N] bias vector is added in
//     the epilogue (k_id = 0 only when split-K is active, see
//     gemm_decode_universal_bias_epilogue.png).
//   - AtomicAdd split-K epilogue when k_batch > 1
template <typename Problem_, typename Policy_ = GemmDecodePolicy>
struct GemmDecodeUniversalKernel : public GemmDecodeNumeric
{
    using Problem = remove_cvref_t<Problem_>;
    using Policy  = remove_cvref_t<Policy_>;

    using ADataType       = typename Problem::ADataType;
    using BDataType       = typename Problem::BDataType;
    using ComputeDataType = typename Problem::ComputeDataType;
    using CDataType       = typename Problem::CDataType;
    using XScaleDataType  = typename Problem::XScaleDataType;
    using WScaleDataType  = typename Problem::WScaleDataType;
    using XScaleLayout    = typename Problem::XScaleLayout;
    using WScaleLayout    = typename Problem::WScaleLayout;

    static constexpr index_t kBlockSize    = Problem::kBlockSize;
    static constexpr index_t kVector       = Problem::kVector;
    static constexpr index_t kMPerWarp     = Problem::kMPerWarp;
    static constexpr index_t kNPerWarp     = Problem::kNPerWarp;
    static constexpr GemmDecodeOutputAxis kOutputAxis = Problem::kOutputAxis;
    static constexpr bool    kHasBias      = Problem::kHasBias;

    static constexpr bool kIsUnscaled  = GemmDecodeScaleLayoutTraits<XScaleLayout>::is_unscaled &&
                                         GemmDecodeScaleLayoutTraits<WScaleLayout>::is_unscaled;
    static constexpr bool kIsPerTensor = GemmDecodeScaleLayoutTraits<XScaleLayout>::is_per_tensor &&
                                         GemmDecodeScaleLayoutTraits<WScaleLayout>::is_per_tensor;

    static_assert(kOutputAxis == GemmDecodeOutputAxis::SmallM,
                  "GemmDecodeUniversalKernel P0 supports only SmallM orientation.");
    static_assert(kMPerWarp == 1 && kNPerWarp == 1,
                  "GemmDecodeUniversalKernel P0 supports only kMPerWarp = kNPerWarp = 1.");
    static_assert(Problem::kWarpsPerBlock == 1,
                  "GemmDecodeUniversalKernel P0 expects exactly one warp per block.");
    static_assert(kIsUnscaled || kIsPerTensor,
                  "GemmDecodeUniversalKernel only supports (unscaled, unscaled) and "
                  "(PerTensor, PerTensor) scale layouts; blockscale uses the dedicated "
                  "GemmDecodeBlockscaleKernel.");
    static_assert(!Problem::kBPreshuffle,
                  "GemmDecodeUniversalKernel: preshuffled-B path lands in P4.");
    static_assert(!kHasBias,
                  "GemmDecodeUniversalKernel: bias epilogue lands in the next commit.");

    struct Kargs
    {
        const void* p_a;       // [M, K]
        const void* p_b;       // [N, K]
        void*       p_c;       // [M, N]

        const void* p_x_scale; // unused in P0
        const void* p_w_scale; // unused in P0
        const void* p_bias;    // unused in P0

        index_t M;
        index_t N;
        index_t K;

        index_t stride_a; // row stride of A in elements
        index_t stride_b; // row stride of B in elements
        index_t stride_c; // row stride of C in elements

        // 1 disables split-K; > 1 enables AtomicAdd epilogue. Caller must
        // zero-init p_c when k_batch > 1 so partials accumulate from zero.
        index_t k_batch;
    };

    CK_TILE_HOST static Kargs MakeKernelArgs(const void* p_a,
                                             const void* p_b,
                                             void*       p_c,
                                             index_t     M,
                                             index_t     N,
                                             index_t     K,
                                             index_t     stride_a,
                                             index_t     stride_b,
                                             index_t     stride_c,
                                             index_t     k_batch = 1)
    {
        return Kargs{p_a,
                     p_b,
                     p_c,
                     /*p_x_scale=*/nullptr,
                     /*p_w_scale=*/nullptr,
                     /*p_bias=*/nullptr,
                     M,
                     N,
                     K,
                     stride_a,
                     stride_b,
                     stride_c,
                     k_batch};
    }

    // Overload for PerTensor scaled subconfig. p_x_scale / p_w_scale are
    // FP32 scalars (one per tensor) and p_bias is an optional [N] vector.
    CK_TILE_HOST static Kargs MakeKernelArgs(const void* p_a,
                                             const void* p_b,
                                             void*       p_c,
                                             const void* p_x_scale,
                                             const void* p_w_scale,
                                             const void* p_bias,
                                             index_t     M,
                                             index_t     N,
                                             index_t     K,
                                             index_t     stride_a,
                                             index_t     stride_b,
                                             index_t     stride_c,
                                             index_t     k_batch = 1)
    {
        return Kargs{p_a,       p_b,       p_c,      p_x_scale, p_w_scale, p_bias,
                     M,         N,         K,        stride_a,  stride_b,  stride_c,
                     k_batch};
    }

    CK_TILE_HOST static constexpr auto GridSize(const Kargs& hargs)
    {
        return dim3(static_cast<uint32_t>(hargs.M),
                    static_cast<uint32_t>(integer_divide_ceil(hargs.N, kNPerWarp)),
                    static_cast<uint32_t>(hargs.k_batch));
    }

    CK_TILE_HOST static constexpr auto BlockSize() { return kBlockSize; }

    CK_TILE_HOST static bool IsSupportedArgument(const Kargs& kargs)
    {
        constexpr index_t kTileN = get_warp_size() * kVector;

        const auto fail = [](const char* msg) {
            if(EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
            {
                CK_TILE_ERROR(msg);
            }
            return false;
        };

        if(kargs.p_a == nullptr || kargs.p_b == nullptr || kargs.p_c == nullptr)
        {
            return fail("GemmDecodeUniversalKernel requires non-null A/B/C pointers.");
        }
        if(kargs.M <= 0 || kargs.N <= 0 || kargs.K <= 0)
        {
            return fail("GemmDecodeUniversalKernel requires positive M, N, K.");
        }
        if(kargs.k_batch <= 0)
        {
            return fail("GemmDecodeUniversalKernel requires k_batch >= 1.");
        }
        if(kargs.stride_a < kargs.K || kargs.stride_b < kargs.K || kargs.stride_c < kargs.N)
        {
            return fail("GemmDecodeUniversalKernel received an invalid row stride.");
        }
        if(kargs.K % kTileN != 0)
        {
            return fail("GemmDecodeUniversalKernel requires K divisible by warp_size * kVector.");
        }
        if(kargs.k_batch > (kargs.K / kTileN))
        {
            // Each shard must own at least one full kTileN iteration so every
            // K element is covered by exactly one shard.
            return fail("GemmDecodeUniversalKernel requires k_batch <= K / (warp_size * kVector).");
        }
        if(kargs.N % kNPerWarp != 0)
        {
            return fail("GemmDecodeUniversalKernel requires N divisible by kNPerWarp.");
        }
        if(kargs.M % kMPerWarp != 0)
        {
            return fail("GemmDecodeUniversalKernel requires M divisible by kMPerWarp.");
        }
        if(kargs.k_batch > 1 && (kargs.N % 2 != 0))
        {
            // The scalar atomic-add helper widens to a 32-bit pair; an odd N
            // would cause the last column's pair to extend one element past
            // the buffer.
            return fail("GemmDecodeUniversalKernel AtomicAdd split-K requires N % 2 == 0.");
        }
        if constexpr(kIsPerTensor)
        {
            if(kargs.p_x_scale == nullptr || kargs.p_w_scale == nullptr)
            {
                return fail("GemmDecodeUniversalKernel PerTensor requires non-null scale "
                            "pointers.");
            }
            // The dot2 K-loop body packs FP8x4 -> two BF16x2 pairs, so each
            // lane's K slice must contain a multiple of 4 FP8 elements.
            if((kVector % 4) != 0)
            {
                return fail("GemmDecodeUniversalKernel PerTensor FP8 path requires "
                            "kVector divisible by 4.");
            }
        }
        return true;
    }

    CK_TILE_DEVICE void operator()(Kargs kargs) const
    {
        constexpr index_t kTileN = get_warp_size() * kVector;

        const index_t m       = static_cast<index_t>(blockIdx.x);
        const index_t n_base  = static_cast<index_t>(blockIdx.y) * kNPerWarp;
        const index_t k_id    = static_cast<index_t>(blockIdx.z);
        const index_t k_batch = kargs.k_batch;

        if(m >= kargs.M || n_base >= kargs.N)
            return;

        // Distribute kTileN-sized iterations across k_batch shards. When
        // num_iter % k_batch != 0 the leading `extra` shards each get one
        // additional iteration so all K elements are covered exactly once
        // without requiring K % (kTileN * k_batch) == 0.
        const index_t num_iter_total = kargs.K / kTileN;
        const index_t base_iter      = num_iter_total / k_batch;
        const index_t extra_iter     = num_iter_total - base_iter * k_batch;
        const index_t iter_start     = k_id * base_iter + (k_id < extra_iter ? k_id : extra_iter);
        const index_t my_iter        = base_iter + (k_id < extra_iter ? 1 : 0);
        const index_t k_offset       = iter_start * kTileN;
        const index_t num_iter       = my_iter;

        // Loop-invariant scale broadcast: PerTensor reads two FP32 scalars.
        ComputeDataType x_scale_val = type_convert<ComputeDataType>(1.0f);
        ComputeDataType w_scale_val = type_convert<ComputeDataType>(1.0f);
        if constexpr(kIsPerTensor)
        {
            x_scale_val = type_convert<ComputeDataType>(
                *static_cast<const XScaleDataType*>(kargs.p_x_scale));
            w_scale_val = type_convert<ComputeDataType>(
                *static_cast<const WScaleDataType*>(kargs.p_w_scale));
        }

        const auto a_view = make_naive_tensor_view<address_space_enum::global>(
            static_cast<const ADataType*>(kargs.p_a),
            make_tuple(kargs.M, kargs.K),
            make_tuple(kargs.stride_a, 1),
            number<kVector>{},
            number<1>{});
        const auto b_view = make_naive_tensor_view<address_space_enum::global>(
            static_cast<const BDataType*>(kargs.p_b),
            make_tuple(kargs.N, kargs.K),
            make_tuple(kargs.stride_b, 1),
            number<kVector>{},
            number<1>{});

        auto a_window = make_tile_window(
            a_view,
            make_tuple(number<1>{}, number<kTileN>{}),
            {m, k_offset},
            Policy::template MakeOutputTileDistribution<Problem>());
        auto b_window = make_tile_window(
            b_view,
            make_tuple(number<1>{}, number<kTileN>{}),
            {n_base, k_offset},
            Policy::template MakeOutputTileDistribution<Problem>());

        ComputeDataType acc = type_convert<ComputeDataType>(0.0f);

        for(index_t i = 0; i < num_iter; ++i)
        {
            auto a_tile = load_tile(a_window);
            auto b_tile = load_tile(b_window);

            if constexpr(Problem::kUseDot2)
            {
                static_assert(std::is_same_v<ComputeDataType, float>,
                              "GemmDecodeUniversalKernel dot2 path expects FP32 "
                              "accumulation.");
                static_assert(kVector % 2 == 0,
                              "GemmDecodeUniversalKernel dot2 path requires kVector "
                              "divisible by 2.");

                // Sweep kVector/2 BF16x2 pairs. For BF16/FP16 inputs, each pair is
                // already one uint32_t in the tile thread buffer; for FP8 inputs we
                // have one uint32_t per 4 FP8s (kVector/4 words) and split each
                // word into two BF16x2 pairs via fp8x2_to_bf16x2.
                static_for<0, kVector / 2, 1>{}([&](auto ipair) {
                    uint32_t a_pair;
                    uint32_t b_pair;
                    if constexpr(std::is_same_v<ADataType, fp8_t>)
                    {
                        constexpr index_t word = ipair.value / 2;
                        constexpr index_t sel  = ipair.value % 2;
                        a_pair = fp8x2_to_bf16x2<sel>(
                            a_tile.get_thread_buffer().template get_as<uint32_t>(
                                number<word>{}));
                        b_pair = fp8x2_to_bf16x2<sel>(
                            b_tile.get_thread_buffer().template get_as<uint32_t>(
                                number<word>{}));
                    }
                    else
                    {
                        // BF16/FP16: each pair is one uint32_t holding two halfs.
                        a_pair = a_tile.get_thread_buffer().template get_as<uint32_t>(ipair);
                        b_pair = b_tile.get_thread_buffer().template get_as<uint32_t>(ipair);
                    }
                    acc = dot2_bf16_packed_add(acc, a_pair, b_pair);
                });
            }
            else
            {
                constexpr auto spans = decltype(a_tile)::get_distributed_spans();
                sweep_tile_span(spans[number<1>{}], [&](auto idx1) {
                    constexpr auto idx = make_tuple(make_tuple(), idx1);
                    const auto a_val = type_convert<ComputeDataType>(a_tile[idx]);
                    const auto b_val = type_convert<ComputeDataType>(b_tile[idx]);
                    acc += a_val * b_val;
                });
            }

            move_tile_window(a_window, {0, kTileN});
            move_tile_window(b_window, {0, kTileN});
        }

        acc = wavefront_reduce_sum(acc);

        if(get_lane_id() == 0)
        {
            const index_t n = n_base;
            // PerTensor: fold the two scalar scales into the reduced acc.
            if constexpr(kIsPerTensor)
            {
                acc = acc * x_scale_val * w_scale_val;
            }

            auto* p_c            = static_cast<CDataType*>(kargs.p_c);
            const auto out_value = type_convert<CDataType>(acc);
            if(k_batch == 1)
            {
                p_c[m * kargs.stride_c + n] = out_value;
            }
            else
            {
                gemm_decode_atomic_add(p_c + m * kargs.stride_c + n, out_value);
            }
        }
    }
};

} // namespace ck_tile
