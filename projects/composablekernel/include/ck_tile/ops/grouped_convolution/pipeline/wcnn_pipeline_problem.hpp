// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

/// @brief Block tile dimensions for WCNN forward convolution.
///
/// @tparam HPerBlock_  Height (H) elements per thread block tile
/// @tparam WPerBlock_  Width (W) elements per thread block tile
/// @tparam CPerBlock_  Input channels (C) per block tile (loop step along C dimension)
/// @tparam KPerBlock_  Output channels (K) per block tile
template <index_t HPerBlock_, index_t WPerBlock_, index_t CPerBlock_, index_t KPerBlock_>
struct WcnnBlockTile
{
    static constexpr index_t HPerBlock = HPerBlock_;
    static constexpr index_t WPerBlock = WPerBlock_;
    static constexpr index_t CPerBlock = CPerBlock_;
    static constexpr index_t KPerBlock = KPerBlock_;
};

/// @brief Warp-level tile dimensions for a single WCNN instruction.
///
/// @tparam HPerWcnn_  Height (H) elements per single WCNN warp instruction
/// @tparam WPerWcnn_  Width (W) elements per single WCNN warp instruction
template <index_t HPerWcnn_, index_t WPerWcnn_>
struct WcnnWarpTile
{
    static constexpr index_t HPerWcnn = HPerWcnn_;
    static constexpr index_t WPerWcnn = WPerWcnn_;
};

/// @brief Number of warps tiling each dimension within a block.
///
/// @tparam WarpsInH_  Number of warps tiling the H dimension
/// @tparam WarpsInW_  Number of warps tiling the W dimension
/// @tparam WarpsInK_  Number of warps tiling the K dimension
template <index_t WarpsInH_, index_t WarpsInW_, index_t WarpsInK_>
struct WcnnWarpCount
{
    static constexpr index_t WarpsInH = WarpsInH_;
    static constexpr index_t WarpsInW = WarpsInW_;
    static constexpr index_t WarpsInK = WarpsInK_;
};

/// @brief Composed block shape for WCNN forward convolution.
///
/// Combines block tile, warp tile, and warp count into a single shape descriptor.
/// BlockSize is derived as WarpsInH * WarpsInW * WarpsInK * 32.
///
/// @tparam BlockTile_  Block tile dimensions (WcnnBlockTile)
/// @tparam WarpTile_   Warp tile dimensions (WcnnWarpTile)
/// @tparam WarpCount_  Warp count per dimension (WcnnWarpCount)
template <typename BlockTile_, typename WarpTile_, typename WarpCount_>
struct BlockWcnnFwdShape
{
    static constexpr index_t HPerBlock = BlockTile_::HPerBlock;
    static constexpr index_t WPerBlock = BlockTile_::WPerBlock;
    static constexpr index_t CPerBlock = BlockTile_::CPerBlock;
    static constexpr index_t KPerBlock = BlockTile_::KPerBlock;
    static constexpr index_t HPerWcnn  = WarpTile_::HPerWcnn;
    static constexpr index_t WPerWcnn  = WarpTile_::WPerWcnn;
    static constexpr index_t WarpsInH  = WarpCount_::WarpsInH;
    static constexpr index_t WarpsInW  = WarpCount_::WarpsInW;
    static constexpr index_t WarpsInK  = WarpCount_::WarpsInK;
    static constexpr index_t NumWarps  = WarpsInH * WarpsInW * WarpsInK;

    static_assert(HPerBlock % (HPerWcnn * WarpsInH) == 0,
                  "HPerBlock must be divisible by HPerWcnn * WarpsInH");
    static_assert(WPerBlock % (WPerWcnn * WarpsInW) == 0,
                  "WPerBlock must be divisible by WPerWcnn * WarpsInW");
    static_assert(KPerBlock % WarpsInK == 0, "KPerBlock must be divisible by WarpsInK");

    // these three variables is defined to make compatible with TilePartitioner
    static constexpr index_t kM = HPerBlock;
    static constexpr index_t kN = WPerBlock;
    static constexpr index_t kK = CPerBlock;
};

/// @brief Problem definition for the WCNN forward pipeline.
///
/// Captures data types, tile sizes, and warp conv configuration for the pipeline.
///
/// @tparam ConvTraits_      Convolution traits (vector sizes, etc.)
/// @tparam ADataType_       A data type (fp16_t, bf16_t) - forward: image/input tensor
/// @tparam BDataType_       B data type (fp16_t, bf16_t) - forward: weight tensor
/// @tparam AccDataType_     Accumulator data type (float)
/// @tparam OutDataType_     Output data type (fp16_t, bf16_t)
/// @tparam BlockWcnnShape_  Block WCNN shape type (e.g. BlockWcnnFwdShape)
/// @tparam FilterY_         Filter height
/// @tparam FilterX_         Filter width
/// @tparam DilationY_       Dilation along height dimension (default 1)
/// @tparam DilationX_       Dilation along width dimension (default 1)
template <typename ConvTraits_,
          typename ADataType_,
          typename BDataType_,
          typename AccDataType_,
          typename OutDataType_,
          typename BlockWcnnShape_,
          index_t FilterY_,
          index_t FilterX_,
          index_t DilationY_ = 1,
          index_t DilationX_ = 1>
struct WcnnFwdPipelineProblem
{
    using ConvTraits     = remove_cvref_t<ConvTraits_>;
    using ADataType      = remove_cvref_t<ADataType_>;
    using BDataType      = remove_cvref_t<BDataType_>;
    using AccDataType    = remove_cvref_t<AccDataType_>;
    using OutDataType    = remove_cvref_t<OutDataType_>;
    using BlockWcnnShape = remove_cvref_t<BlockWcnnShape_>;

    static constexpr index_t FilterY   = FilterY_;
    static constexpr index_t FilterX   = FilterX_;
    static constexpr index_t DilationY = DilationY_;
    static constexpr index_t DilationX = DilationX_;

    static constexpr bool FixedVectorSize =
        (ConvTraits::VectorSizeA > 0) && (ConvTraits::VectorSizeB > 0);
    static constexpr index_t VectorSizeA = ConvTraits::VectorSizeA;
    static constexpr index_t VectorSizeB = ConvTraits::VectorSizeB;

    static constexpr index_t BlockSize = BlockWcnnShape::NumWarps * get_warp_size();
};

} // namespace ck_tile
