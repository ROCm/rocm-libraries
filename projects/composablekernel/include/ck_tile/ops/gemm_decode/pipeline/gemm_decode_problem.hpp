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
          bool kBPreshuffle_                      = false>
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
    static constexpr bool    kBPreshuffle        = kBPreshuffle_;

    static constexpr index_t kWarpsPerBlock = kWarpsPerBlock_;
    static constexpr index_t kBlockSize     = kWarpsPerBlock * get_warp_size();
};

} // namespace ck_tile
