// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"
#include "ck_tile/ops/gemm_decode/kernel/gemm_decode_chiplet_swizzle.hpp"
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
//   - kMPerWarp >= 1, kNPerWarp >= 1: each warp computes a kMPerWarp x
//     kNPerWarp output tile. kMPerWarp > 1 reuses each B row across the M
//     rows held in registers (B-reuse), the dual of the kNPerWarp A-reuse.
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
    static_assert(kMPerWarp >= 1,
                  "GemmDecodeUniversalKernel requires kMPerWarp >= 1.");
    static_assert(kNPerWarp >= 1,
                  "GemmDecodeUniversalKernel requires kNPerWarp >= 1.");
    static_assert(Problem::kWarpsPerBlock == 1,
                  "GemmDecodeUniversalKernel P0 expects exactly one warp per block.");
    static_assert(kIsUnscaled || kIsPerTensor,
                  "GemmDecodeUniversalKernel only supports (unscaled, unscaled) and "
                  "(PerTensor, PerTensor) scale layouts; blockscale uses the dedicated "
                  "GemmDecodeBlockscaleKernel.");
    static_assert(!Problem::kBPreshuffle,
                  "GemmDecodeUniversalKernel: preshuffled-B path lands in P4.");

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
        return dim3(static_cast<uint32_t>(integer_divide_ceil(hargs.M, kMPerWarp)),
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
        // M need not be divisible by kMPerWarp: the kernel launches
        // ceil(M / kMPerWarp) row-blocks, clamps the tail block's A-row loads
        // in-bounds, and masks those rows in the epilogue, so any runtime M is
        // valid.
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
        if constexpr(kHasBias)
        {
            if(kargs.p_bias == nullptr)
            {
                return fail("GemmDecodeUniversalKernel kHasBias requires a non-null bias "
                            "pointer.");
            }
        }
        return true;
    }

    CK_TILE_DEVICE void operator()(Kargs kargs) const
    {
        constexpr index_t kTileN = get_warp_size() * kVector;

        // (m_block, n_block) recovery. With the chiplet-swizzle path enabled,
        // the hardware (blockIdx.x, blockIdx.y) pair is treated as a flat
        // wgid, remapped through the XCD-aware permutation, and then
        // unflattened so consecutive logical wgids land on the same XCD's
        // L2 slice. k_id stays on blockIdx.z and is *not* part of the
        // remap: each split-K shard has its own contiguous (m_block, n_block)
        // sweep and can independently benefit from the chunked layout.
        index_t m_block;
        index_t n_block;
        if constexpr(Problem::kChipletSwizzle)
        {
            const index_t num_m_blocks = static_cast<index_t>(gridDim.x);
            const index_t num_n_blocks = static_cast<index_t>(gridDim.y);
            const index_t hw_wgid =
                static_cast<index_t>(blockIdx.y) * num_m_blocks +
                static_cast<index_t>(blockIdx.x);
            const index_t logical_wgid =
                GemmDecodeChipletSwizzle::remap_wgid(hw_wgid,
                                                    num_m_blocks * num_n_blocks,
                                                    Problem::kChipletNumXcds,
                                                    Problem::kChipletChunkSize);
            m_block = logical_wgid % num_m_blocks;
            n_block = logical_wgid / num_m_blocks;
        }
        else
        {
            m_block = static_cast<index_t>(blockIdx.x);
            n_block = static_cast<index_t>(blockIdx.y);
        }
        const index_t m_base  = m_block * kMPerWarp;
        const index_t n_base  = n_block * kNPerWarp;
        const index_t k_id    = static_cast<index_t>(blockIdx.z);
        const index_t k_batch = kargs.k_batch;

        if(m_base >= kargs.M || n_base >= kargs.N)
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

        // kMPerWarp activation-row windows. Each warp owns a
        // kMPerWarp x kNPerWarp output tile: A[m_base+jm, :] is loaded once per
        // K-iteration and reused across the kNPerWarp B rows (the A-reuse from
        // A1), and -- the key win over the old per-row grid -- each
        // B[n_base+jn, :] vector is loaded once and reused across the kMPerWarp
        // A rows held in registers (B-reuse). This is the dual of the kNPerWarp
        // A-reuse and the structural trick behind FlyDSL/MFMA's flat-in-M
        // curve (they reuse a 16-row A-fragment across N; we reuse B across
        // kMPerWarp rows in VGPRs), so B traffic drops from ~M*N*K to ~N*K and
        // the kernel stops scaling with M.
        //
        // Tail block: when m_base + jm >= M the row is clamped to M-1 so the
        // global load stays in-bounds; that row's result is dropped by the
        // masked epilogue.
        auto a_windows = generate_tuple(
            [&](auto jm) {
                index_t a_row = m_base + jm.value;
                if(a_row >= kargs.M)
                    a_row = kargs.M - 1;
                return make_tile_window(
                    a_view,
                    make_tuple(number<1>{}, number<kTileN>{}),
                    {a_row, k_offset},
                    Policy::template MakeOutputTileDistribution<Problem>());
            },
            number<kMPerWarp>{});

        // kMPerWarp x kNPerWarp output accumulators, row-major in
        // (jm * kNPerWarp + jn).
        array<ComputeDataType, kMPerWarp * kNPerWarp> acc;
        static_for<0, kMPerWarp * kNPerWarp, 1>{}([&](auto idx) {
            acc(idx) = type_convert<ComputeDataType>(0.0f);
        });

        // One persistent B window per N output, all moved together each iter.
        auto b_windows = generate_tuple(
            [&](auto jn) {
                return make_tile_window(
                    b_view,
                    make_tuple(number<1>{}, number<kTileN>{}),
                    {n_base + jn.value, k_offset},
                    Policy::template MakeOutputTileDistribution<Problem>());
            },
            number<kNPerWarp>{});

        for(index_t i = 0; i < num_iter; ++i)
        {
            if constexpr(Problem::kUseDot2)
            {
                static_assert(std::is_same_v<ComputeDataType, float>,
                              "GemmDecodeUniversalKernel dot2 path expects FP32 "
                              "accumulation.");
                static_assert(kVector % 2 == 0,
                              "GemmDecodeUniversalKernel dot2 path requires kVector "
                              "divisible by 2.");

                // Dequantize each A row into kVector/2 BF16x2 register pairs
                // once; a_pairs[jm*(kVector/2)+ipair] is reused across the
                // kNPerWarp B rows below.
                array<uint32_t, kMPerWarp*(kVector / 2)> a_pairs;
                static_for<0, kMPerWarp, 1>{}([&](auto jm) {
                    auto a_tile = load_tile(a_windows[jm]);
                    static_for<0, kVector / 2, 1>{}([&](auto ipair) {
                        constexpr index_t slot = jm.value * (kVector / 2) + ipair.value;
                        if constexpr(std::is_same_v<ADataType, fp8_t>)
                        {
                            constexpr index_t word = ipair.value / 2;
                            constexpr index_t sel  = ipair.value % 2;
                            a_pairs(number<slot>{}) = fp8x2_to_bf16x2<sel>(
                                a_tile.get_thread_buffer().template get_as<uint32_t>(
                                    number<word>{}));
                        }
                        else
                        {
                            a_pairs(number<slot>{}) =
                                a_tile.get_thread_buffer().template get_as<uint32_t>(ipair);
                        }
                    });
                });

                // Each B row is loaded and dequantized once, then reused across
                // all kMPerWarp accumulators -- the B-reuse win.
                static_for<0, kNPerWarp, 1>{}([&](auto jn) {
                    auto b_tile = load_tile(b_windows[jn]);
                    array<uint32_t, kVector / 2> b_pairs;
                    static_for<0, kVector / 2, 1>{}([&](auto ipair) {
                        if constexpr(std::is_same_v<BDataType, fp8_t>)
                        {
                            constexpr index_t word = ipair.value / 2;
                            constexpr index_t sel  = ipair.value % 2;
                            b_pairs(ipair) = fp8x2_to_bf16x2<sel>(
                                b_tile.get_thread_buffer().template get_as<uint32_t>(
                                    number<word>{}));
                        }
                        else
                        {
                            b_pairs(ipair) =
                                b_tile.get_thread_buffer().template get_as<uint32_t>(ipair);
                        }
                    });
                    static_for<0, kMPerWarp, 1>{}([&](auto jm) {
                        constexpr index_t acc_idx = jm.value * kNPerWarp + jn.value;
                        static_for<0, kVector / 2, 1>{}([&](auto ipair) {
                            constexpr index_t slot = jm.value * (kVector / 2) + ipair.value;
                            acc(number<acc_idx>{}) = dot2_bf16_packed_add(
                                acc[number<acc_idx>{}], a_pairs[number<slot>{}],
                                b_pairs[ipair]);
                        });
                    });
                });
            }
            else
            {
                // Plain path: load all kMPerWarp A rows once, then reuse each B
                // row across them.
                auto a_tiles = generate_tuple(
                    [&](auto jm) { return load_tile(a_windows[jm]); },
                    number<kMPerWarp>{});
                static_for<0, kNPerWarp, 1>{}([&](auto jn) {
                    auto b_tile = load_tile(b_windows[jn]);
                    static_for<0, kMPerWarp, 1>{}([&](auto jm) {
                        constexpr index_t acc_idx = jm.value * kNPerWarp + jn.value;
                        auto a_tile          = a_tiles[jm];
                        constexpr auto spans = decltype(a_tile)::get_distributed_spans();
                        sweep_tile_span(spans[number<1>{}], [&](auto idx1) {
                            constexpr auto idx = make_tuple(make_tuple(), idx1);
                            const auto a_val = type_convert<ComputeDataType>(a_tile[idx]);
                            const auto b_val = type_convert<ComputeDataType>(b_tile[idx]);
                            acc(number<acc_idx>{}) += a_val * b_val;
                        });
                    });
                });
            }

            static_for<0, kMPerWarp, 1>{}([&](auto jm) {
                move_tile_window(a_windows[jm], {0, kTileN});
            });
            static_for<0, kNPerWarp, 1>{}([&](auto jn) {
                move_tile_window(b_windows[jn], {0, kTileN});
            });
        }

        static_for<0, kMPerWarp * kNPerWarp, 1>{}([&](auto idx) {
            acc(idx) = wavefront_reduce_sum(acc[idx]);
        });

        if(get_lane_id() == 0)
        {
            auto* p_c = static_cast<CDataType*>(kargs.p_c);

            static_for<0, kMPerWarp, 1>{}([&](auto jm) {
                const index_t m_row = m_base + jm.value;
                // Tail block: rows >= M were clamped on load and are dropped
                // here. The grid is ceil(N / kNPerWarp) and N % kNPerWarp == 0
                // is enforced, so every n below is in range.
                if(m_row >= kargs.M)
                    return;

                static_for<0, kNPerWarp, 1>{}([&](auto jn) {
                    const index_t n           = n_base + jn.value;
                    constexpr index_t acc_idx = jm.value * kNPerWarp + jn.value;
                    ComputeDataType out_acc   = acc[number<acc_idx>{}];

                    // PerTensor: fold the two scalar scales into the reduced acc.
                    if constexpr(kIsPerTensor)
                    {
                        out_acc = out_acc * x_scale_val * w_scale_val;
                    }

                    // Bias: add bias[n] to the first split-K shard only so the
                    // atomicAdd partials sum to (bias + sum_k a*b). Mirrors
                    // wvSplitK*'s in-kernel bias add. Bias dtype follows CDataType.
                    if constexpr(kHasBias)
                    {
                        if(k_id == 0)
                        {
                            const auto* p_bias =
                                static_cast<const CDataType*>(kargs.p_bias);
                            const auto bias_val =
                                type_convert<ComputeDataType>(p_bias[n]);
                            out_acc += bias_val;
                        }
                    }

                    const auto out_value = type_convert<CDataType>(out_acc);
                    if(k_batch == 1)
                    {
                        p_c[m_row * kargs.stride_c + n] = out_value;
                    }
                    else
                    {
                        gemm_decode_atomic_add(p_c + m_row * kargs.stride_c + n,
                                               out_value);
                    }
                });
            });
        }
    }
};

} // namespace ck_tile
