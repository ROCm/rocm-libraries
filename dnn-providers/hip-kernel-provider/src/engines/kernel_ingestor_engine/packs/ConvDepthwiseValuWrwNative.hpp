// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>

#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>

#include "engines/kernel_ingestor_engine/descriptors/conv/depthwise_valu_wrw/kernels/ConvDepthwiseValuWrwArgs.h"
#include "engines/kernel_ingestor_engine/packs/ConvDepthwiseValuProblem.hpp"
#include "engines/kernel_ingestor_engine/packs/ConvNaiveProblem.hpp"

namespace hip_kernel_provider::kernel_ingestor_engine::conv_depthwise_valu_wrw
{

/// A depthwise backward weights as the VALU kernels run it.
struct WrwProblem
{
    conv_naive::ConvNaiveProblem base;
    conv_depthwise_valu::DepthwiseLayout layout
        = conv_depthwise_valu::DepthwiseLayout::CHANNELS_LAST;
    /// 2 for 1-D and 2-D problems (1-D runs with height 1), 3 for 3-D.
    int ndims = 2;
    /// True convolution: dw is written reflected.
    bool flipFilter = false;
    int64_t xUid = 0;
    int64_t dyUid = 0;
    int64_t dwUid = 0;
    /// The kernels' argument block, pointers left null and the partitions unset until
    /// partitionFor.
    ConvDepthwiseValuWrwArgs args{};
};

/**
 * @brief The depthwise backward weights @p context's graph describes, or nullopt when the VALU
 *        kernels do not serve it.
 *
 * Everything parseConvNaiveProblem gates, plus: depthwise (k == c == groups); x and dy packed in
 * the same layout, channels-last or channels-first, and dw packed; every extent, padding,
 * stride, dilation, the reduction rows and the (channel, tap) count within 32 bits.
 */
std::optional<WrwProblem> parseWrwProblem(const hipdnn_plugin_sdk::ingestor::MatchContext& context);

/// Splits @p problem's reduction rows into partitions enough to fill @p computeUnits.
void partitionFor(WrwProblem& problem, int64_t computeUnits);

/// The partial sums' workspace, once partitioned.
size_t workspaceBytesFor(const WrwProblem& problem);

/// The two launches: the partial kernel's grid, the fold kernel's, both of `block` threads.
struct WrwLaunch
{
    int64_t block = 0;
    int64_t partialGrid = 0;
    int64_t foldGrid = 0;
};

/// The launches for a partitioned @p problem, or nullopt when a grid exceeds 32 bits.
std::optional<WrwLaunch> launchFor(const WrwProblem& problem);

} // namespace hip_kernel_provider::kernel_ingestor_engine::conv_depthwise_valu_wrw
