// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>

#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>

#include "engines/kernel_ingestor_engine/descriptors/conv/group/kernels/ConvGroupArgs.h"
#include "engines/kernel_ingestor_engine/packs/ConvIgemmProblem.hpp"

namespace hip_kernel_provider::kernel_ingestor_engine::conv_group
{

/// A small-group convolution as the K4 kernels run it: the activations staged channels-last
/// exactly as the implicit-GEMM engine stages them (its parse, its stagings), the filter expanded
/// per 16-channel block into the workspace after them.
struct GroupProblem
{
    conv_igemm::IgemmProblem igemm;
    /// The kernel's arguments; pointers are filled at launch.
    K4Args args{};
    /// The filter expansion; pointers are filled at launch.
    K4ExpandArgs expand{};
    int filterR = 0, filterS = 0;
    int stride = 1;
    size_t expandOffset = 0; ///< the expanded filter's place in the workspace
    size_t expandBytes = 0;
    size_t workspaceBytes = 0; ///< staged activations and the expanded filter
};

/// The convolution @p context's graph describes in @p direction, or nullopt when the K4 kernels
/// do not serve it. Everything parseIgemmProblem gates, plus: 2-D (no depth), groups of cg
/// channels each way with 2 <= cg <= 16 and cg dividing 16 (not depthwise: those are the
/// depthwise engines'), channels a multiple of 16, unit dilation, equal strides on both axes
/// (backward data: unit stride), and a grid within the launch limits for the smallest tile.
std::optional<GroupProblem>
    parseGroupProblem(const hipdnn_plugin_sdk::ingestor::MatchContext& context,
                      conv_naive::ConvDirection direction);

/// Workgroups of a TH x (16 wp) output tile over cb channel blocks: images x row tiles x column
/// tiles x channel tiles, or nullopt past the launch limit.
std::optional<unsigned int> groupBlocks(const GroupProblem& problem, int th, int wp, int cb);

/// LDS bytes of the input halo a config stages (the kernel's static array).
long long groupLdsBytes(int elementBytes, int r, int s, int stride, int th, int wp, int cb);

} // namespace hip_kernel_provider::kernel_ingestor_engine::conv_group
