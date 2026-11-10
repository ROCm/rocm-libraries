// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>

#include "engines/kernel_ingestor_engine/descriptors/conv/group/kernels/ConvGroupArgs.h"
#include "engines/kernel_ingestor_engine/packs/ConvIgemmProblem.hpp"

namespace hip_kernel_provider::kernel_ingestor_engine::conv_group
{

/// One launch of the K4 kernel and its filter expansion: forward and unit-stride backward data
/// run one; strided backward data one per stride phase that has outputs.
struct GroupPhase
{
    K4Args args{}; ///< pointers are filled at launch
    K4ExpandArgs expand{}; ///< pointers are filled at launch
    size_t expandOffset = 0; ///< this phase's expanded filter in the workspace
};

/// A small-group convolution as the K4 kernels run it: the activations staged channels-last
/// exactly as the implicit-GEMM engine stages them (its parse, its stagings), each phase's
/// expanded filter in the workspace after them.
struct GroupProblem
{
    conv_igemm::IgemmProblem igemm;
    std::vector<GroupPhase> phases;
    int filterT = 1, filterR = 0, filterS = 0; ///< the graph's filter extent
    int stride = 1; ///< the graph's row and column stride
    /// The kernel's compile-time filter extent and gather stride: forward the graph's;
    /// backward data the taps a stride phase reaches (ceil(F / stride)) at unit stride.
    int kernelT = 1, kernelR = 0, kernelS = 0, kernelStride = 1;
    size_t workspaceBytes = 0; ///< staged activations and the expanded filters
};

/// The convolution @p context's graph describes in @p direction, or nullopt when the K4 kernels
/// do not serve it. Everything parseIgemmProblem gates, plus: groups of cg channels each way with
/// 2 <= cg <= 16 and cg dividing 16 (not depthwise: those are the depthwise engines'), channels a
/// multiple of 16, unit dilation, equal row and column strides (backward data: row, column and
/// depth strides 1 or 2), and grids within the launch limits for the smallest tile. 2-D or 3-D.
std::optional<GroupProblem>
    parseGroupProblem(const hipdnn_plugin_sdk::ingestor::MatchContext& context,
                      conv_naive::ConvDirection direction);

/// Workgroups of a TH x (16 wp) output tile over cb channel blocks for one launch: images x
/// output planes x row tiles x column tiles x channel tiles, or nullopt past the launch limit.
std::optional<unsigned int> phaseBlocks(const K4Args& args, int th, int wp, int cb);

/// LDS bytes of the input halo a config stages (the kernel's static array).
long long groupLdsBytes(int elementBytes, int r, int s, int stride, int th, int wp, int cb);

} // namespace hip_kernel_provider::kernel_ingestor_engine::conv_group
