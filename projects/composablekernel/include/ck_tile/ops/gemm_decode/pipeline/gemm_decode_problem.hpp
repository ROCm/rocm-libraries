// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

// Output orientation. SmallM is the autoregressive decode case
// (M = 1..16, N large). SmallN is the wvSplitK / vLLM case (M large, N
// small). P0 only implements SmallM; SmallN lands in P3.
enum struct GemmDecodeOutputAxis
{
    SmallM,
    SmallN
};

// Scale layout tags. Mirrors WarpDecodeScaleLayout::{PerTensor, PerToken,
// Block2D<>} from ops/warp_decode/. `void` denotes the unscaled subconfig
// (BF16/FP16 tested in P0). PerTensor / PerToken / Block2D are template
// surface for P0b/P1; their scale-loading code is not active in P0.
struct GemmDecodeScaleLayout
{
    struct PerTensor
    {
    };
    struct PerToken
    {
    };
    template <index_t Block_N, index_t Block_K>
    struct Block2D
    {
    };
};

template <typename Layout>
struct GemmDecodeScaleLayoutTraits
{
    static constexpr bool is_block2d   = false;
    static constexpr bool is_unscaled  = false;
    static constexpr bool is_per_token = false;
    static constexpr bool is_per_tensor = false;
};

template <>
struct GemmDecodeScaleLayoutTraits<void>
{
    static constexpr bool is_block2d    = false;
    static constexpr bool is_unscaled   = true;
    static constexpr bool is_per_token  = false;
    static constexpr bool is_per_tensor = false;
};

template <>
struct GemmDecodeScaleLayoutTraits<GemmDecodeScaleLayout::PerTensor>
{
    static constexpr bool is_block2d    = false;
    static constexpr bool is_unscaled   = false;
    static constexpr bool is_per_token  = false;
    static constexpr bool is_per_tensor = true;
};

template <>
struct GemmDecodeScaleLayoutTraits<GemmDecodeScaleLayout::PerToken>
{
    static constexpr bool is_block2d    = false;
    static constexpr bool is_unscaled   = false;
    static constexpr bool is_per_token  = true;
    static constexpr bool is_per_tensor = false;
};

template <index_t Block_N, index_t Block_K>
struct GemmDecodeScaleLayoutTraits<GemmDecodeScaleLayout::Block2D<Block_N, Block_K>>
{
    static constexpr bool is_block2d    = true;
    static constexpr bool is_unscaled   = false;
    static constexpr bool is_per_token  = false;
    static constexpr bool is_per_tensor = false;
    static constexpr index_t block_n    = Block_N;
    static constexpr index_t block_k    = Block_K;
};

// Compile-time problem traits for the warp-per-scalar dense GEMM kernel.
//
// A: [M, K] activation, B: [N, K] weight (each row of B is one output column
// of B^T), C: [M, N]. Layouts are implicit row-major in P0; transposed
// activations are out of scope. The (XScaleLayout, WScaleLayout) pair selects
// the scale subconfig at compile time:
//   - (void, void)               : unscaled BF16/FP16   <- P0
//   - (PerTensor, PerTensor)     : per-tensor FP8       <- P0b
//   - (PerToken,  PerTensor)     : per-token A scale    <- P0b
//   - (Block2D<>, Block2D<>)     : DeepSeek blockscale  <- P1
template <typename ADataType_,
          typename BDataType_,
          typename ComputeDataType_,
          typename CDataType_,
          typename XScaleDataType_                = float,
          typename WScaleDataType_                = float,
          typename XScaleLayout_                  = void,
          typename WScaleLayout_                  = void,
          index_t kVector_                        = 8,
          bool kUseDot2_                          = false,
          bool kUsePackedFp32_                    = false,
          index_t kMPerWarp_                      = 1,
          index_t kNPerWarp_                      = 1,
          GemmDecodeOutputAxis kOutputAxis_       = GemmDecodeOutputAxis::SmallM,
          bool kHasBias_                          = false,
          index_t kWarpsPerBlock_                 = 1,
          // Reserved for the P4 wvSplitK-style row-interleaved B layout
          // ([N / YTILE, K, YTILE]) with kNPerWarp > 1 register cross-N reuse.
          // P0/P0b/P1 keep row-major B; both kernels static_assert this off.
          bool kBPreshuffle_                      = false,
          // XCD-aware workgroup remap (MI300/MI355). When enabled, the
          // kernel takes the HW (m, n_block) wgid, runs it through
          // `GemmDecodeChipletSwizzle::remap_wgid`, and unflattens the
          // result so that `kChipletChunkSize` consecutive logical wgids
          // land on the same XCD. Defaults are off (no swizzle) and the
          // gfx950 / MI355X chiplet count.
          bool    kChipletSwizzle_                = false,
          index_t kChipletNumXcds_                = 8,
          index_t kChipletChunkSize_              = 8,
          // 2D modular-broadcast bias (wvSplitK* Bx/By indirection). When
          // true, the bias epilogue indexes
          //   bias[(feat % bias_x) + (tok % bias_y) * bias_x]
          // using the runtime bias_x / bias_y extents carried in Kargs, where
          // feat is the output-feature index and tok is the token index.
          // When false (default) the bias is the flat 1D vector indexed by
          // feat, so the common path stays branch-free. Requires kHasBias.
          bool    kBias2D_                        = false,
          // Stage the shared activation row in LDS (wvSplitK* A-in-LDS /
          // WD-OPT-21). Only meaningful in the multi-warp path, where every
          // warp would otherwise re-read the same A row from global each
          // K-iteration; with this flag the workgroup loads row m into LDS
          // once and all warps stream it from LDS. Requires kWarpsPerBlock > 1
          // and kMPerWarp == kNPerWarp == 1.
          bool    kStageAInLds_                   = false,
          // Mark the B (weight) global loads non-temporal (wvSplitK* streams B
          // with cache-bypassing loads). B is the dominant ~N*K traffic and is
          // read once per element with no temporal reuse, so a non-temporal
          // hint keeps it from evicting the reused A / scales from cache. Pure
          // performance hint -- correctness is unchanged.
          bool    kStreamB_                       = false,
          // Persistent fat-WG launch (wvSplitK* "1 WG/CU" geometry). When true
          // the launcher caps the grid at the CU count and each workgroup
          // grid-strides over the logical (m_block, n_block, k_id) tile space
          // instead of one workgroup per tile. Pairs with kWarpsPerBlock > 1 so
          // a single fat workgroup owns a CU and sweeps many output columns,
          // amortizing launch/setup and keeping A resident -- the launch-shape
          // half of wvSplitKQ's recipe. Pure scheduling change; each tile is
          // still computed exactly once, so results are unchanged.
          bool    kPersistent_                    = false>
struct GemmDecodeProblem
{
    using ADataType        = remove_cvref_t<ADataType_>;
    using BDataType        = remove_cvref_t<BDataType_>;
    using ComputeDataType  = remove_cvref_t<ComputeDataType_>;
    using CDataType        = remove_cvref_t<CDataType_>;
    using XScaleDataType   = remove_cvref_t<XScaleDataType_>;
    using WScaleDataType   = remove_cvref_t<WScaleDataType_>;
    using XScaleLayout     = XScaleLayout_;
    using WScaleLayout     = WScaleLayout_;

    static constexpr index_t kVector             = kVector_;
    static constexpr bool    kUseDot2            = kUseDot2_;
    static constexpr bool    kUsePackedFp32      = kUsePackedFp32_;
    static constexpr index_t kMPerWarp           = kMPerWarp_;
    static constexpr index_t kNPerWarp           = kNPerWarp_;
    static constexpr GemmDecodeOutputAxis kOutputAxis = kOutputAxis_;
    static constexpr bool    kHasBias            = kHasBias_;
    static constexpr bool    kBias2D             = kBias2D_;
    static constexpr bool    kStageAInLds        = kStageAInLds_;
    static constexpr bool    kStreamB            = kStreamB_;
    static constexpr bool    kPersistent         = kPersistent_;
    static constexpr bool    kBPreshuffle        = kBPreshuffle_;

    static constexpr bool    kChipletSwizzle     = kChipletSwizzle_;
    static constexpr index_t kChipletNumXcds     = kChipletNumXcds_;
    static constexpr index_t kChipletChunkSize   = kChipletChunkSize_;

    static constexpr index_t kWarpsPerBlock = kWarpsPerBlock_;
    static constexpr index_t kBlockSize     = kWarpsPerBlock * get_warp_size();
};

} // namespace ck_tile
