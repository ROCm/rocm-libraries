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

template <typename XDataType_,
          typename WDataType_,
          typename ComputeDataType_,
          typename IntermediateDataType_,
          typename XScaleLayout_ = WarpDecodeScaleLayout::PerTensor,
          typename WScaleLayout_ = WarpDecodeScaleLayout::PerTensor,
          typename Activation_ = ck_tile::element_wise::Silu>
struct WarpDecodeGateUpProblem
{
    using XDataType            = remove_cvref_t<XDataType_>;
    using WDataType            = remove_cvref_t<WDataType_>;
    using ComputeDataType      = remove_cvref_t<ComputeDataType_>;
    using IntermediateDataType = remove_cvref_t<IntermediateDataType_>;
    using XScaleLayout         = remove_cvref_t<XScaleLayout_>;
    using WScaleLayout         = remove_cvref_t<WScaleLayout_>;
    using Activation           = remove_cvref_t<Activation_>;

    static constexpr index_t kBlockSize = get_warp_size();
};

template <typename IntermediateDataType_,
          typename WDataType_,
          typename ComputeDataType_,
          typename YDataType_,
          typename WScaleLayout_ = WarpDecodeScaleLayout::PerTensor>
struct WarpDecodeDownReduceProblem
{
    using IntermediateDataType = remove_cvref_t<IntermediateDataType_>;
    using WDataType            = remove_cvref_t<WDataType_>;
    using ComputeDataType      = remove_cvref_t<ComputeDataType_>;
    using YDataType            = remove_cvref_t<YDataType_>;
    using WScaleLayout         = remove_cvref_t<WScaleLayout_>;

    static constexpr index_t kBlockSize = get_warp_size();
};

} // namespace ck_tile
