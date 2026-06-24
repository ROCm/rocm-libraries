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
//   - kWarpsPerBlock >= 1: kWarpsPerBlock > 1 packs that many independent
//     warps per workgroup (each owning one output column) to raise
//     wavefronts/CU on the small M=1 grid (§15.F occupancy probe); wired for
//     mp=np=1, k_batch=1 only.
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
    // 2D modular-broadcast bias (wvSplitK* Bx/By). Only meaningful with bias.
    static constexpr bool    kBias2D       = Problem::kBias2D;
    // Stage the shared A row in LDS (wvSplitK* A-in-LDS). Multi-warp only.
    static constexpr bool    kStageAInLds  = Problem::kStageAInLds;

    // Non-temporal coherence for the streamed B loads (wvSplitK* cache-bypass).
    // DEVICE_NT1 sets the gfx94x/gfx950 non-temporal bit at device-cache scope;
    // on other archs the enum maps it to the legacy glc hint. coherence_default
    // keeps B cacheable. Applied only to the B view (A is reused / LDS-staged).
    static constexpr amd_buffer_coherence_enum kBCoherence =
        Problem::kStreamB ? amd_buffer_coherence_enum::DEVICE_NT1
                          : amd_buffer_coherence_enum::coherence_default;

    // Persistent fat-WG launch (wvSplitK* "1 WG/CU"). When set, the launcher
    // caps the grid at the CU count and each workgroup grid-strides over the
    // logical (m_block, n_block, k_id) tile space (see operator()).
    static constexpr bool    kPersistent   = Problem::kPersistent;

    // Max K (in A elements) the A-in-LDS staging path will hold per workgroup.
    // 8192 covers the decode K range (e.g. 7168) at 8 KB (FP8) / 16 KB (BF16)
    // of LDS, well under the 64 KB/CU budget. IsSupportedArgument rejects a
    // larger runtime K when kStageAInLds is on -- the analog of wvSplitKQ's
    // "A fits in LDS" check that selects its _sml_ kernel.
    static constexpr index_t kLdsStageMaxK = 8192;

    static constexpr bool kIsUnscaled  = GemmDecodeScaleLayoutTraits<XScaleLayout>::is_unscaled &&
                                         GemmDecodeScaleLayoutTraits<WScaleLayout>::is_unscaled;
    static constexpr bool kIsPerTensor = GemmDecodeScaleLayoutTraits<XScaleLayout>::is_per_tensor &&
                                         GemmDecodeScaleLayoutTraits<WScaleLayout>::is_per_tensor;
    // Per-token (wvSplitKQ-style activation quant): X carries one FP32 scale
    // per token (per output row m, an [M] vector); W stays per-tensor (one
    // scalar). The token scale is folded in the epilogue as a per-row factor
    // x_scale[m] * w_scale, so the K-loop is identical to PerTensor -- only the
    // X-scale load moves from a loop-invariant scalar to a per-row gather.
    static constexpr bool kIsPerToken  = GemmDecodeScaleLayoutTraits<XScaleLayout>::is_per_token &&
                                         GemmDecodeScaleLayoutTraits<WScaleLayout>::is_per_tensor;
    // Any scaled subconfig shares the per-tensor W scalar load + epilogue fold.
    static constexpr bool kIsScaled    = kIsPerTensor || kIsPerToken;

    static_assert(kOutputAxis == GemmDecodeOutputAxis::SmallM,
                  "GemmDecodeUniversalKernel P0 supports only SmallM orientation.");
    static_assert(kMPerWarp >= 1,
                  "GemmDecodeUniversalKernel requires kMPerWarp >= 1.");
    static_assert(kNPerWarp >= 1,
                  "GemmDecodeUniversalKernel requires kNPerWarp >= 1.");
    static_assert(Problem::kWarpsPerBlock >= 1,
                  "GemmDecodeUniversalKernel requires kWarpsPerBlock >= 1.");
    // Multi-warp (kWarpsPerBlock>1) is the §15.F occupancy probe ("B1-lite"):
    // it packs kWarpsPerBlock independent warps per workgroup, each owning one
    // output column, to lift wavefronts/CU on the small M=1 grid. It is wired
    // only for the mp=np=1 autotuned M=1 winner; register-tiled mp/np stay on
    // the single-warp path.
    static_assert(Problem::kWarpsPerBlock == 1 || (kMPerWarp == 1 && kNPerWarp == 1),
                  "GemmDecodeUniversalKernel multi-warp path requires kMPerWarp == "
                  "kNPerWarp == 1.");
    static_assert(kIsUnscaled || kIsPerTensor || kIsPerToken,
                  "GemmDecodeUniversalKernel only supports (unscaled, unscaled), "
                  "(PerTensor, PerTensor), and (PerToken, PerTensor) scale layouts; "
                  "blockscale uses the dedicated GemmDecodeBlockscaleKernel.");
    static_assert(!Problem::kBPreshuffle,
                  "GemmDecodeUniversalKernel: preshuffled-B path lands in P4.");
    static_assert(!kBias2D || kHasBias,
                  "GemmDecodeUniversalKernel kBias2D requires kHasBias.");
    // A-in-LDS staging is implemented in (and only benefits) the multi-warp
    // path, where warps share the activation row. The multi-warp path is
    // already restricted to mp == np == 1, so this keeps the contract tight.
    static_assert(!kStageAInLds || Problem::kWarpsPerBlock > 1,
                  "GemmDecodeUniversalKernel kStageAInLds requires kWarpsPerBlock > 1 "
                  "(multi-warp path; A reuse across warps is what it stages).");

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

        // 2D modular-broadcast bias extents (wvSplitK* Bx/By). Only read when
        // Problem::kBias2D is true: the bias is indexed
        //   bias[(feat % bias_x) + (tok % bias_y) * bias_x].
        // For the flat 1D bias (kBias2D == false) these are ignored; setting
        // bias_y = 1, bias_x = N reproduces the 1D result.
        index_t bias_x;
        index_t bias_y;
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
                     k_batch,
                     /*bias_x=*/0,
                     /*bias_y=*/1};
    }

    // Overload for the scaled subconfigs. p_w_scale is always a per-tensor FP32
    // scalar; p_x_scale is a per-tensor FP32 scalar (PerTensor) or an [M] FP32
    // activation-scale vector indexed by token/row (PerToken). p_bias is an
    // optional [N] vector. Same entry point for both -- the layout is selected
    // at compile time by the Problem's (XScaleLayout, WScaleLayout).
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
                                             index_t     k_batch = 1,
                                             // 2D-bias extents; ignored unless
                                             // Problem::kBias2D. bias_y = 1,
                                             // bias_x = N reproduces 1D bias.
                                             index_t     bias_x = 0,
                                             index_t     bias_y = 1)
    {
        return Kargs{p_a,       p_b,       p_c,      p_x_scale, p_w_scale, p_bias,
                     M,         N,         K,        stride_a,  stride_b,  stride_c,
                     k_batch,   bias_x,    bias_y};
    }

    CK_TILE_HOST static constexpr auto GridSize(const Kargs& hargs)
    {
        // Multi-warp packs kWarpsPerBlock independent warps per workgroup, each
        // owning one N column, so the N grid shrinks by that factor. The
        // single-warp default (kWarpsPerBlock==1) leaves this unchanged.
        return dim3(static_cast<uint32_t>(integer_divide_ceil(hargs.M, kMPerWarp)),
                    static_cast<uint32_t>(
                        integer_divide_ceil(hargs.N, kNPerWarp * Problem::kWarpsPerBlock)),
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
        if constexpr(kStageAInLds)
        {
            // The staged A row must fit the LDS budget (see kLdsStageMaxK).
            if(kargs.K > kLdsStageMaxK)
            {
                return fail("GemmDecodeUniversalKernel kStageAInLds requires K <= "
                            "kLdsStageMaxK (the A row must fit in LDS).");
            }
        }
        if constexpr(Problem::kWarpsPerBlock > 1)
        {
            // Each workgroup owns kWarpsPerBlock consecutive columns; require an
            // exact tiling so no warp's B-row read runs past N (the block-level
            // tile distribution loads all kWarpsPerBlock rows together, so a
            // partial last group cannot be masked per-warp on load).
            if(kargs.N % (kNPerWarp * Problem::kWarpsPerBlock) != 0)
            {
                return fail("GemmDecodeUniversalKernel multi-warp path requires N "
                            "divisible by kNPerWarp * kWarpsPerBlock.");
            }
            // The probe keeps the K loop full-length (the point is more warps,
            // not shorter waves); split-K is intentionally out of scope.
            if(kargs.k_batch != 1)
            {
                return fail("GemmDecodeUniversalKernel multi-warp path requires "
                            "k_batch == 1.");
            }
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
        if constexpr(kIsScaled)
        {
            if(kargs.p_x_scale == nullptr || kargs.p_w_scale == nullptr)
            {
                return fail("GemmDecodeUniversalKernel scaled path requires non-null scale "
                            "pointers.");
            }
            // The dot2 K-loop body packs FP8x4 -> two BF16x2 pairs, so each
            // lane's K slice must contain a multiple of 4 FP8 elements.
            if((kVector % 4) != 0)
            {
                return fail("GemmDecodeUniversalKernel scaled FP8 path requires "
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
            if constexpr(kBias2D)
            {
                // The modular bias index needs strictly-positive extents (the
                // % operands). bias_y == 1 with bias_x == N reproduces the 1D
                // bias; any larger period tiles/broadcasts.
                if(kargs.bias_x <= 0 || kargs.bias_y <= 0)
                {
                    return fail("GemmDecodeUniversalKernel kBias2D requires bias_x >= 1 and "
                                "bias_y >= 1.");
                }
            }
        }
        return true;
    }

    // Resolve the bias element index for output element (feat, tok), where
    // feat is the output-feature index (the column n in SmallM) and tok is the
    // token index (the row m in SmallM). The flat 1D bias is just bias[feat]
    // (per-feature, broadcast across tokens); the 2D modular-broadcast bias
    // tiles over both axes with periods bias_x (feature) and bias_y (token),
    // matching wvSplitK*'s BIAS[(feat % Bx) + (tok % By) * Bx].
    CK_TILE_DEVICE static index_t BiasIndex(index_t feat, index_t tok, const Kargs& kargs)
    {
        if constexpr(kBias2D)
        {
            return (feat % kargs.bias_x) + (tok % kargs.bias_y) * kargs.bias_x;
        }
        else
        {
            (void)tok;
            (void)kargs;
            return feat;
        }
    }

    // X-side scale for output row m (the token index in SmallM). PerTensor
    // broadcasts the single loop-invariant scalar passed in `x_scale_pertensor`;
    // PerToken gathers p_x_scale[m] (the [M] activation-scale vector). Unscaled
    // returns the identity. Keeps the two epilogue store sites uniform.
    CK_TILE_DEVICE static ComputeDataType
    XScaleForRow(const Kargs& kargs, index_t m, ComputeDataType x_scale_pertensor)
    {
        if constexpr(kIsPerToken)
        {
            return type_convert<ComputeDataType>(
                static_cast<const XScaleDataType*>(kargs.p_x_scale)[m]);
        }
        else
        {
            (void)kargs;
            (void)m;
            return x_scale_pertensor;
        }
    }

    CK_TILE_DEVICE void operator()(Kargs kargs) const
    {
        if constexpr(!kPersistent)
        {
            // One workgroup per logical tile: the hardware block / grid ids map
            // straight onto (m_block, n_block, k_id) and the logical grid is
            // gridDim itself -- bit-identical to the pre-persistent kernel.
            RunTile(kargs,
                    static_cast<index_t>(blockIdx.x),
                    static_cast<index_t>(blockIdx.y),
                    static_cast<index_t>(blockIdx.z),
                    static_cast<index_t>(gridDim.x),
                    static_cast<index_t>(gridDim.y));
        }
        else
        {
            // Persistent fat-WG (wvSplitK* "1 WG/CU"): the launcher capped the
            // grid at the CU count, so each workgroup grid-strides over the full
            // logical tile space. The logical extents mirror GridSize() exactly
            // (so the decoded (blk_x, blk_y, blk_z) reproduce the per-tile
            // launch's blockIdx), and every work item is visited once, so the
            // result is identical to the per-tile launch. All warps of a WG share
            // the loop index (derived from the uniform block/grid ids), so any
            // block_sync_lds() in RunTile is reached by the whole workgroup.
            const index_t logical_grid_m = integer_divide_ceil(kargs.M, kMPerWarp);
            const index_t logical_grid_n =
                integer_divide_ceil(kargs.N, kNPerWarp * Problem::kWarpsPerBlock);
            const index_t plane    = logical_grid_m * logical_grid_n;
            const index_t num_work = plane * kargs.k_batch;
            const index_t stride   = get_grid_size();

            for(index_t w = get_block_id(); w < num_work; w += stride)
            {
                const index_t blk_z = w / plane;
                const index_t rem   = w - blk_z * plane;
                const index_t blk_y = rem / logical_grid_m;
                const index_t blk_x = rem - blk_y * logical_grid_m;
                RunTile(kargs, blk_x, blk_y, blk_z, logical_grid_m, logical_grid_n);
            }
        }
    }

    // Compute one logical tile (blk_x, blk_y, blk_z) of the output. blk_* are
    // the (m_block, n_block, k_id) indices -- the hardware block ids for the
    // per-tile launch, or the decoded grid-stride work index for the persistent
    // launch. logical_grid_m / logical_grid_n are the logical (non-persistent)
    // grid extents, used by the chiplet swizzle to remap a flat wgid.
    CK_TILE_DEVICE void RunTile(const Kargs& kargs,
                                index_t      blk_x,
                                index_t      blk_y,
                                index_t      blk_z,
                                index_t      logical_grid_m,
                                index_t      logical_grid_n) const
    {
        constexpr index_t kTileN = get_warp_size() * kVector;

        // ---- Multi-warp occupancy path (design doc §15.F probe, "B1-lite") ----
        // Pack kWarpsPerBlock independent warps into one workgroup so the small
        // M=1 grid schedules ~kWarpsPerBlock x more wavefronts per CU. The
        // single-warp WG (64 threads) is capped near 10-28% occupancy by the
        // workgroups-per-CU limit -- not by registers (M=1 uses ~32 VGPR) -- so
        // adding warps/WG, without shortening the K loop the way split-K does,
        // is the lever §15.F flagged as the one untested M=1 occupancy probe.
        // Warp w in n-group blk_y owns column n = blk_y*WPB + w of
        // row m = blk_x; A is the shared activation row (warp-replicated
        // broadcast distribution), B is one row per warp (output distribution,
        // P0 = warp_id). Restricted to mp=np=1, k_batch=1 by the asserts /
        // IsSupportedArgument above.
        if constexpr(Problem::kWarpsPerBlock > 1)
        {
            const index_t warp_id = get_warp_id();
            const index_t m       = blk_x;
            const index_t n       = blk_y * Problem::kWarpsPerBlock + warp_id;

            if(m >= kargs.M || n >= kargs.N)
                return;

            // W is per-tensor for every scaled subconfig (one scalar); the X
            // scale is a per-tensor scalar (PerTensor) or gathered per-row in
            // the epilogue (PerToken), so only the W scalar is preloaded here.
            ComputeDataType x_scale_val = type_convert<ComputeDataType>(1.0f);
            ComputeDataType w_scale_val = type_convert<ComputeDataType>(1.0f);
            if constexpr(kIsScaled)
            {
                w_scale_val = type_convert<ComputeDataType>(
                    *static_cast<const WScaleDataType*>(kargs.p_w_scale));
                if constexpr(kIsPerTensor)
                    x_scale_val = type_convert<ComputeDataType>(
                        *static_cast<const XScaleDataType*>(kargs.p_x_scale));
            }

            const auto b_view = make_naive_tensor_view<address_space_enum::global,
                                                        memory_operation_enum::set,
                                                        kBCoherence>(
                static_cast<const BDataType*>(kargs.p_b),
                make_tuple(kargs.N, kargs.K),
                make_tuple(kargs.stride_b, 1),
                number<kVector>{},
                number<1>{});

            // B: kWarpsPerBlock consecutive rows, one per warp (P0 = warp_id).
            // Shared by both the global and A-in-LDS paths below.
            auto b_window = make_tile_window(
                b_view,
                make_tuple(number<Problem::kWarpsPerBlock>{}, number<kTileN>{}),
                {blk_y * Problem::kWarpsPerBlock, 0},
                Policy::template MakeOutputTileDistribution<Problem>());

            ComputeDataType acc      = type_convert<ComputeDataType>(0.0f);
            const index_t   num_iter = kargs.K / kTileN;

            // dot2 / plain accumulate of one (A-tile, B-tile) pair into acc.
            auto accumulate = [&](auto& a_tile, auto& b_tile) {
                if constexpr(Problem::kUseDot2)
                {
                    static_for<0, kVector / 2, 1>{}([&](auto ipair) {
                        uint32_t a_pair;
                        uint32_t b_pair;
                        if constexpr(std::is_same_v<ADataType, fp8_t>)
                        {
                            constexpr index_t word = ipair.value / 2;
                            constexpr index_t sel  = ipair.value % 2;
                            a_pair                 = fp8x2_to_bf16x2<sel>(
                                a_tile.get_thread_buffer().template get_as<uint32_t>(
                                    number<word>{}));
                            b_pair = fp8x2_to_bf16x2<sel>(
                                b_tile.get_thread_buffer().template get_as<uint32_t>(
                                    number<word>{}));
                        }
                        else
                        {
                            a_pair = a_tile.get_thread_buffer().template get_as<uint32_t>(ipair);
                            b_pair = b_tile.get_thread_buffer().template get_as<uint32_t>(ipair);
                        }
                        acc = dot2_bf16_packed_add(acc, a_pair, b_pair);
                    });
                }
                else
                {
                    static_for<0, kVector, 1>{}([&](auto iv) {
                        const auto a_val = type_convert<ComputeDataType>(
                            a_tile.get_thread_buffer().template get_as<ADataType>(iv));
                        const auto b_val = type_convert<ComputeDataType>(
                            b_tile.get_thread_buffer().template get_as<BDataType>(iv));
                        acc += a_val * b_val;
                    });
                }
            };

            // Double-buffered prefetch over an A window (global or LDS-staged)
            // and the shared B window: issue iter (i+1)'s loads before consuming
            // iter i, so the next fetch's HBM latency overlaps the current
            // tile's dot2. The single-warp M=1 kernel runs at MLP=1
            // (synchronous load->wait->compute) and is latency-bound at low
            // occupancy (§15.F rocprof: VALU 12%, MemUnitStalled ~0, HBM ~45%
            // of ideal), so raising memory-level parallelism is the lever that
            // occupancy alone (kWarpsPerBlock) does not reach.
            auto run_kloop = [&](auto& a_window) {
                auto a_cur = load_tile(a_window);
                auto b_cur = load_tile(b_window);
                for(index_t i = 0; i < num_iter; ++i)
                {
                    move_tile_window(a_window, {0, kTileN});
                    move_tile_window(b_window, {0, kTileN});
                    if(i + 1 < num_iter)
                    {
                        auto a_next = load_tile(a_window);
                        auto b_next = load_tile(b_window);
                        accumulate(a_cur, b_cur);
                        a_cur = a_next;
                        b_cur = b_next;
                    }
                    else
                    {
                        accumulate(a_cur, b_cur);
                    }
                }
            };

            if constexpr(kStageAInLds)
            {
                // wvSplitK* A-in-LDS: under the broadcast distribution every
                // warp would re-read the same A row from global each K-iter.
                // Stage row m into LDS once (all WG threads cooperate, scalar
                // strided copy), then stream it from LDS. Bounded by
                // kLdsStageMaxK (IsSupportedArgument rejects a larger K) -- the
                // analog of wvSplitKQ's "A fits in LDS" _sml_ launch condition.
                __shared__ ADataType a_smem[kLdsStageMaxK];
                const auto* a_global = static_cast<const ADataType*>(kargs.p_a) +
                                       static_cast<index_t>(m) * kargs.stride_a;
                if constexpr(kPersistent)
                {
                    // Persistent reuse: a prior grid-stride iteration's warps may
                    // still be streaming this WG's a_smem when we loop back to
                    // restage. Fence the WAR hazard before overwriting it. (The
                    // whole workgroup runs the same tile sequence, so the barrier
                    // is uniform.)
                    block_sync_lds();
                }
                for(index_t c = static_cast<index_t>(threadIdx.x); c < kargs.K;
                    c += static_cast<index_t>(kBlockSize))
                {
                    a_smem[c] = a_global[c];
                }
                block_sync_lds();

                const auto a_lds_view = make_naive_tensor_view<address_space_enum::lds>(
                    a_smem,
                    make_tuple(static_cast<index_t>(1), kargs.K),
                    make_tuple(kargs.K, static_cast<index_t>(1)),
                    number<kVector>{},
                    number<1>{});
                auto a_window = make_tile_window(
                    a_lds_view,
                    make_tuple(number<1>{}, number<kTileN>{}),
                    {0, 0},
                    Policy::template MakeXBroadcastTileDistribution<Problem>());
                run_kloop(a_window);
            }
            else
            {
                // A: the single shared row m, read from global and broadcast
                // (replicated) to every warp.
                const auto a_view = make_naive_tensor_view<address_space_enum::global>(
                    static_cast<const ADataType*>(kargs.p_a),
                    make_tuple(kargs.M, kargs.K),
                    make_tuple(kargs.stride_a, 1),
                    number<kVector>{},
                    number<1>{});
                auto a_window = make_tile_window(
                    a_view,
                    make_tuple(number<1>{}, number<kTileN>{}),
                    {m, 0},
                    Policy::template MakeXBroadcastTileDistribution<Problem>());
                run_kloop(a_window);
            }

            acc = wavefront_reduce_sum(acc);
            if(get_lane_id() == 0)
            {
                if constexpr(kIsScaled)
                    acc = acc * XScaleForRow(kargs, m, x_scale_val) * w_scale_val;
                if constexpr(kHasBias)
                {
                    const auto* p_bias = static_cast<const CDataType*>(kargs.p_bias);
                    acc += type_convert<ComputeDataType>(p_bias[BiasIndex(n, m, kargs)]);
                }
                auto* p_c                       = static_cast<CDataType*>(kargs.p_c);
                p_c[m * kargs.stride_c + n] = type_convert<CDataType>(acc);
            }
            return;
        }

        // (m_block, n_block) recovery. With the chiplet-swizzle path enabled,
        // the (blk_x, blk_y) tile pair is treated as a flat wgid, remapped
        // through the XCD-aware permutation, and then unflattened so
        // consecutive logical wgids land on the same XCD's L2 slice. The
        // logical grid extents (not gridDim, which is the CU count under the
        // persistent launch) drive the unflatten. k_id = blk_z is *not* part of the
        // remap: each split-K shard has its own contiguous (m_block, n_block)
        // sweep and can independently benefit from the chunked layout.
        index_t m_block;
        index_t n_block;
        if constexpr(Problem::kChipletSwizzle)
        {
            const index_t num_m_blocks = logical_grid_m;
            const index_t num_n_blocks = logical_grid_n;
            const index_t hw_wgid      = blk_y * num_m_blocks + blk_x;
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
            m_block = blk_x;
            n_block = blk_y;
        }
        const index_t m_base  = m_block * kMPerWarp;
        const index_t n_base  = n_block * kNPerWarp;
        const index_t k_id    = blk_z;
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

        // Loop-invariant scale broadcast. W is per-tensor (one scalar) for every
        // scaled subconfig; PerTensor also reads a single X scalar, while
        // PerToken gathers X per output row (x_scale[m]) in the epilogue below.
        ComputeDataType x_scale_val = type_convert<ComputeDataType>(1.0f);
        ComputeDataType w_scale_val = type_convert<ComputeDataType>(1.0f);
        if constexpr(kIsScaled)
        {
            w_scale_val = type_convert<ComputeDataType>(
                *static_cast<const WScaleDataType*>(kargs.p_w_scale));
            if constexpr(kIsPerTensor)
                x_scale_val = type_convert<ComputeDataType>(
                    *static_cast<const XScaleDataType*>(kargs.p_x_scale));
        }

        const auto a_view = make_naive_tensor_view<address_space_enum::global>(
            static_cast<const ADataType*>(kargs.p_a),
            make_tuple(kargs.M, kargs.K),
            make_tuple(kargs.stride_a, 1),
            number<kVector>{},
            number<1>{});
        const auto b_view = make_naive_tensor_view<address_space_enum::global,
                                                    memory_operation_enum::set,
                                                    kBCoherence>(
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

                    // Fold the scales into the reduced acc: PerTensor uses the
                    // two preloaded scalars; PerToken uses x_scale[m_row] (this
                    // row's token scale) * the per-tensor w_scale.
                    if constexpr(kIsScaled)
                    {
                        out_acc = out_acc * XScaleForRow(kargs, m_row, x_scale_val) * w_scale_val;
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
                            const auto bias_val = type_convert<ComputeDataType>(
                                p_bias[BiasIndex(n, m_row, kargs)]);
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
