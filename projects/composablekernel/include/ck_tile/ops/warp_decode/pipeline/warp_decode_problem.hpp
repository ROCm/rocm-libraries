// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/elementwise/unary_element_wise_operation.hpp"

namespace ck_tile {

struct WarpDecodeScaleLayout {
    struct PerTensor {};
    struct PerToken {}; // For activations, e.g., per token scale
    template <index_t Block_N, index_t Block_K>
    struct Block2D {}; // For MXFP8/MXFP4 weights, e.g. 128x128
};

template <typename Layout>
struct ScaleLayoutTraits;

template <>
struct ScaleLayoutTraits<WarpDecodeScaleLayout::PerTensor> {
    static constexpr bool is_block2d = false;
};

template <>
struct ScaleLayoutTraits<WarpDecodeScaleLayout::PerToken> {
    static constexpr bool is_block2d = false;
};

template <index_t Block_N, index_t Block_K>
struct ScaleLayoutTraits<WarpDecodeScaleLayout::Block2D<Block_N, Block_K>> {
    static constexpr bool is_block2d = true;
    static constexpr index_t block_n = Block_N;
    static constexpr index_t block_k = Block_K;
};

template <typename XDataType_,
          typename WDataType_,
          typename ComputeDataType_,
          typename IntermediateDataType_,
          typename XScaleDataType_ = float,
          typename WScaleDataType_ = float,
          typename XScaleLayout_ = WarpDecodeScaleLayout::PerTensor,
          typename WScaleLayout_ = WarpDecodeScaleLayout::PerTensor,
          typename Activation_ = ck_tile::element_wise::Silu,
          index_t kVector_ = 1>
struct WarpDecodeGateUpProblem
{
    using XDataType            = remove_cvref_t<XDataType_>;
    using WDataType            = remove_cvref_t<WDataType_>;
    using ComputeDataType      = remove_cvref_t<ComputeDataType_>;
    using IntermediateDataType = remove_cvref_t<IntermediateDataType_>;
    using XScaleDataType       = remove_cvref_t<XScaleDataType_>;
    using WScaleDataType       = remove_cvref_t<WScaleDataType_>;
    using XScaleLayout         = remove_cvref_t<XScaleLayout_>;
    using WScaleLayout         = remove_cvref_t<WScaleLayout_>;
    using Activation           = remove_cvref_t<Activation_>;

    static constexpr index_t kBlockSize = get_warp_size();
    static constexpr index_t kVector    = kVector_;
};

template <typename IntermediateDataType_,
          typename WDataType_,
          typename ComputeDataType_,
          typename YDataType_,
          typename WScaleDataType_ = float,
          typename WScaleLayout_ = WarpDecodeScaleLayout::PerTensor,
          index_t kVector_ = 1>
struct WarpDecodeDownReduceProblem
{
    using IntermediateDataType = remove_cvref_t<IntermediateDataType_>;
    using WDataType            = remove_cvref_t<WDataType_>;
    using ComputeDataType      = remove_cvref_t<ComputeDataType_>;
    using YDataType            = remove_cvref_t<YDataType_>;
    using WScaleDataType       = remove_cvref_t<WScaleDataType_>;
    using WScaleLayout         = remove_cvref_t<WScaleLayout_>;

    static constexpr index_t kBlockSize = get_warp_size();
    static constexpr index_t kVector    = kVector_;
};

} // namespace ck_tile
