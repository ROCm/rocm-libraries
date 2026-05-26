// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"
#include "ck_tile/ops/gemm_decode/kernel/gemm_decode_numeric.hpp"
#include "ck_tile/ops/gemm_decode/pipeline/gemm_decode_problem.hpp"
#include "ck_tile/ops/gemm_decode/pipeline/gemm_decode_policy.hpp"

namespace ck_tile {

// Warp-per-scalar dense GEMM kernel for FP8 with Block2D X / Block2D W
// scale layouts (DeepSeek-V3 / a8w8 blockscale convention). Compared to
// GemmDecodeUniversalKernel:
//
//   - Inner K-loop reads two `Block_K`-aligned scalar scales per outer
//     iteration (one for X, one for W) and folds them into the dot
//     accumulator at *each* iteration boundary.
//   - X scale tensor: row-major [M, K / Block_K_x], one scalar per
//     (m, k_block_x).
//   - W scale tensor: row-major [N / Block_N_w, K / Block_K_w], one
//     scalar per (w_row_block, k_block_w).
//   - Both X and W scales are FP32. Block sizes are inherited from
//     `Problem::XScaleLayout` / `Problem::WScaleLayout`, which must be
//     `GemmDecodeScaleLayout::Block2D<.,.>`. The DeepSeek-V3 convention
//     is X = Block2D<1, 128> ("PerToken on M, blocked on K") and
//     W = Block2D<128, 128>.
//
// This commit ships only the global-only path (every K-loop iteration
// reads the two scale scalars from HBM). The next commit adds an
// LDS scale-broadcast prologue (WD-OPT-18) that stages the workgroup's
// scales into LDS once and replaces the inner-loop loads with LDS reads.
//
// FUTURE: when `kBPreshuffle = true` lands in P4, B (and W scales)
// should be laid out as `[N / YTILE, K, YTILE]` and `[BQN / YTILE,
// BQK, YTILE]` respectively, interleaving YTILE columns of B at the
// innermost dim so a single `kVector`-wide load fills YTILE register
// rows and a single scalar fetch broadcasts the W scale across them.
// We assert `!kBPreshuffle` in this kernel so the eventual implementer
// can drop the new layout in alongside `kNPerWarp > 1` register reuse.
template <typename Problem_, typename Policy_ = GemmDecodePolicy>
struct GemmDecodeBlockscaleKernel : public GemmDecodeNumeric
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

    using XScaleTraits = GemmDecodeScaleLayoutTraits<XScaleLayout>;
    using WScaleTraits = GemmDecodeScaleLayoutTraits<WScaleLayout>;

    static constexpr index_t kBlockSize = Problem::kBlockSize;
    static constexpr index_t kVector    = Problem::kVector;
    static constexpr index_t kMPerWarp  = Problem::kMPerWarp;
    static constexpr index_t kNPerWarp  = Problem::kNPerWarp;
    static constexpr GemmDecodeOutputAxis kOutputAxis = Problem::kOutputAxis;
    static constexpr bool    kHasBias   = Problem::kHasBias;

    static_assert(kOutputAxis == GemmDecodeOutputAxis::SmallM,
                  "GemmDecodeBlockscaleKernel currently supports only SmallM orientation.");
    static_assert(kMPerWarp == 1 && kNPerWarp == 1,
                  "GemmDecodeBlockscaleKernel currently supports only kMPerWarp = kNPerWarp = 1.");
    static_assert(Problem::kWarpsPerBlock == 1,
                  "GemmDecodeBlockscaleKernel currently expects exactly one warp per block.");
    static_assert(XScaleTraits::is_block2d && WScaleTraits::is_block2d,
                  "GemmDecodeBlockscaleKernel requires Block2D X and Block2D W scale layouts.");
    static_assert(!Problem::kBPreshuffle,
                  "GemmDecodeBlockscaleKernel: preshuffled-B path lands in P4.");

    // Per-block scale dimensions, surfaced for the caller.
    static constexpr index_t kXScaleBlockN = XScaleTraits::block_n;
    static constexpr index_t kXScaleBlockK = XScaleTraits::block_k;
    static constexpr index_t kWScaleBlockN = WScaleTraits::block_n;
    static constexpr index_t kWScaleBlockK = WScaleTraits::block_k;

    static_assert(kXScaleBlockK == kWScaleBlockK,
                  "GemmDecodeBlockscaleKernel currently expects matching Block_K for X and W "
                  "scales (the K-loop reads one scalar pair per outer iteration).");

    static constexpr index_t kBlockK = kXScaleBlockK;

    struct Kargs
    {
        const void* p_a;       // [M, K]
        const void* p_b;       // [N, K]
        void*       p_c;       // [M, N]

        const void* p_x_scale; // [M, K / kXScaleBlockK]   row-major, FP32
        const void* p_w_scale; // [N / kWScaleBlockN, K / kWScaleBlockK]   row-major, FP32
        const void* p_bias;    // [N], CDataType, optional (kHasBias)

        index_t M;
        index_t N;
        index_t K;

        index_t stride_a; // row stride of A in elements
        index_t stride_b; // row stride of B in elements
        index_t stride_c; // row stride of C in elements

        index_t k_batch;
    };

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
            return fail("GemmDecodeBlockscaleKernel requires non-null A/B/C pointers.");
        }
        if(kargs.p_x_scale == nullptr || kargs.p_w_scale == nullptr)
        {
            return fail("GemmDecodeBlockscaleKernel requires non-null scale pointers.");
        }
        if(kargs.M <= 0 || kargs.N <= 0 || kargs.K <= 0)
        {
            return fail("GemmDecodeBlockscaleKernel requires positive M, N, K.");
        }
        if(kargs.k_batch <= 0)
        {
            return fail("GemmDecodeBlockscaleKernel requires k_batch >= 1.");
        }
        if(kargs.stride_a < kargs.K || kargs.stride_b < kargs.K || kargs.stride_c < kargs.N)
        {
            return fail("GemmDecodeBlockscaleKernel received an invalid row stride.");
        }
        if(kargs.K % kTileN != 0)
        {
            return fail("GemmDecodeBlockscaleKernel requires K divisible by warp_size * kVector.");
        }
        if(kargs.K % kBlockK != 0)
        {
            return fail("GemmDecodeBlockscaleKernel requires K divisible by the scale block_k.");
        }
        // Each outer K-loop iteration loads kTileN K elements; for the
        // per-iter scalar scale fetch to stay valid we need kTileN to land
        // on a kBlockK boundary so all lanes of the warp share the same
        // scale block.
        if(kTileN % kBlockK != 0)
        {
            return fail("GemmDecodeBlockscaleKernel requires (warp_size * kVector) "
                        "divisible by the scale block_k.");
        }
        if(kargs.k_batch > (kargs.K / kTileN))
        {
            return fail("GemmDecodeBlockscaleKernel requires k_batch <= K / (warp_size * kVector).");
        }
        if(kargs.N % kNPerWarp != 0)
        {
            return fail("GemmDecodeBlockscaleKernel requires N divisible by kNPerWarp.");
        }
        if(kargs.M % kMPerWarp != 0)
        {
            return fail("GemmDecodeBlockscaleKernel requires M divisible by kMPerWarp.");
        }
        if(kargs.k_batch > 1 && (kargs.N % 2 != 0))
        {
            return fail("GemmDecodeBlockscaleKernel AtomicAdd split-K requires N % 2 == 0.");
        }
        if(kargs.N % kWScaleBlockN != 0)
        {
            return fail("GemmDecodeBlockscaleKernel requires N divisible by W-scale block_n.");
        }
        if(kargs.M % kXScaleBlockN != 0)
        {
            return fail("GemmDecodeBlockscaleKernel requires M divisible by X-scale block_n.");
        }
        if(kVector % 4 != 0)
        {
            return fail("GemmDecodeBlockscaleKernel FP8 dot2 path requires kVector divisible by 4.");
        }
        if constexpr(kHasBias)
        {
            if(kargs.p_bias == nullptr)
            {
                return fail("GemmDecodeBlockscaleKernel kHasBias requires a non-null bias pointer.");
            }
        }
        return true;
    }

    // Global-only Block2D scale read. Mirrors warp_decode_gate_up_kernel's
    // load_block2d_scale (warp_decode_gate_up_kernel.hpp:174-191) and is the
    // fallback when LDS staging is disabled or out of capacity (P1 commit 2).
    template <typename ScaleLayout, typename ScaleDataType>
    CK_TILE_DEVICE static ComputeDataType
    load_block2d_scale(const void* p_scale, index_t row_idx, index_t k_idx, index_t max_k)
    {
        if constexpr(GemmDecodeScaleLayoutTraits<ScaleLayout>::is_block2d)
        {
            if(p_scale == nullptr)
                return type_convert<ComputeDataType>(1.0f);
            constexpr index_t Block_N = GemmDecodeScaleLayoutTraits<ScaleLayout>::block_n;
            constexpr index_t Block_K = GemmDecodeScaleLayoutTraits<ScaleLayout>::block_k;
            const ScaleDataType* ptr = static_cast<const ScaleDataType*>(p_scale);
            const index_t r          = row_idx / Block_N;
            const index_t c          = k_idx / Block_K;
            return type_convert<ComputeDataType>(ptr[r * (max_k / Block_K) + c]);
        }
        else
        {
            return type_convert<ComputeDataType>(1.0f);
        }
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

        const index_t num_iter_total = kargs.K / kTileN;
        const index_t base_iter      = num_iter_total / k_batch;
        const index_t extra_iter     = num_iter_total - base_iter * k_batch;
        const index_t iter_start     = k_id * base_iter + (k_id < extra_iter ? k_id : extra_iter);
        const index_t my_iter        = base_iter + (k_id < extra_iter ? 1 : 0);
        const index_t k_offset       = iter_start * kTileN;
        const index_t num_iter       = my_iter;

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

        // Per-lane K position model. The MakeOutputTileDistribution lays out
        // `warp_size * kVector` K elements per outer iter as
        // `lane_id * kVector + [0, kVector)`, so each lane sits inside a
        // single Block_K stride: lane 0..(Block_K/kVector - 1) own scale
        // sub-block 0 of the warp's tile, the next group own sub-block 1,
        // etc. We compute the lane's sub-block once and read its (xs, ws)
        // from HBM at every outer iteration. The next commit replaces the
        // per-iter HBM read with an LDS broadcast.
        const index_t lane_id   = get_lane_id();
        const index_t k_in_tile = lane_id * kVector;

        for(index_t i = 0; i < num_iter; ++i)
        {
            auto a_tile = load_tile(a_window);
            auto b_tile = load_tile(b_window);

            const index_t k_base       = k_offset + i * kTileN;
            const index_t k_lane_block = k_base + k_in_tile; // any K offset within the
                                                             // lane's sub-block works
            const ComputeDataType xs = load_block2d_scale<XScaleLayout, XScaleDataType>(
                kargs.p_x_scale, m, k_lane_block, kargs.K);
            const ComputeDataType ws = load_block2d_scale<WScaleLayout, WScaleDataType>(
                kargs.p_w_scale, n_base, k_lane_block, kargs.K);

            ComputeDataType iter_dot = type_convert<ComputeDataType>(0.0f);

            // Walk this lane's kVector K elements as kVector/2 BF16x2 pairs,
            // exactly as the universal kernel does.
            static_for<0, kVector / 2, 1>{}([&](auto ipair) {
                if constexpr(std::is_same_v<ADataType, fp8_t>)
                {
                    constexpr index_t word = ipair.value / 2;
                    constexpr index_t sel  = ipair.value % 2;
                    const uint32_t a_pair  = fp8x2_to_bf16x2<sel>(
                        a_tile.get_thread_buffer().template get_as<uint32_t>(
                            number<word>{}));
                    const uint32_t b_pair  = fp8x2_to_bf16x2<sel>(
                        b_tile.get_thread_buffer().template get_as<uint32_t>(
                            number<word>{}));
                    iter_dot = dot2_bf16_packed_add(iter_dot, a_pair, b_pair);
                }
                else
                {
                    const uint32_t a_pair =
                        a_tile.get_thread_buffer().template get_as<uint32_t>(ipair);
                    const uint32_t b_pair =
                        b_tile.get_thread_buffer().template get_as<uint32_t>(ipair);
                    iter_dot = dot2_bf16_packed_add(iter_dot, a_pair, b_pair);
                }
            });

            acc += iter_dot * xs * ws;

            move_tile_window(a_window, {0, kTileN});
            move_tile_window(b_window, {0, kTileN});
        }

        acc = wavefront_reduce_sum(acc);

        if(get_lane_id() == 0)
        {
            const index_t n = n_base;

            if constexpr(kHasBias)
            {
                if(k_id == 0)
                {
                    const auto* p_bias  = static_cast<const CDataType*>(kargs.p_bias);
                    const auto bias_val = type_convert<ComputeDataType>(p_bias[n]);
                    acc += bias_val;
                }
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
