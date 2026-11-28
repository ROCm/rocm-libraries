// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstdint>
#include <string_view>
#include <vector>

#include <hip/hip_runtime_api.h>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/SavedDispatch.hpp>

#include "compilation/IRunnableKernel.hpp"
#include "compilation/KernelCompileOptions.hpp"
#include "engines/kernel_ingestor_engine/packs/ConvRdnaCommon.hpp"
#include "engines/kernel_ingestor_engine/packs/ConvRdnaFusedEpilogue.hpp"

/**
 * @file ConvRdnaDw2d.hpp
 * @brief The depthwise 2D kernels as the RDNA engines use them: the specialized kernel
 *        (descriptors/hip/conv_dw2d_rdna/kernels/ConvFwdDepthwise2d.cpp) and, for every other
 *        problem, the generic kernel of the dw3d bundle with depth 1. Which problems they serve,
 *        which variant a problem takes and how it walks the output, how a variant is compiled
 *        and launched, and the problem half of a saved launch. `hip_conv_fwd_dw2d_rdna` runs them
 *        as they are, `hip_conv_fwd_bias_dw2d_rdna` with their bias epilogue (HAS_BIAS in the
 *        specialized kernel, DW_HAS_BIAS in the generic one) and
 *        `hip_conv_fwd_bias_act_dw2d_rdna` with the bias and activation epilogues (HAS_ACT,
 *        DW_HAS_ACT on top).
 */
namespace hip_kernel_provider::kernel_ingestor_engine::conv_rdna::dw2d
{

/// How a depthwise 2D variant walks the output; the grid depends on it. Dtype, layout, the
/// specialized filter, padding, stride and dilation are compiled into the variant and only
/// matter for picking it.
struct Geometry
{
    /// The runtime-parameter kernel (filter, padding, stride and dilation as launch arguments).
    bool generic = false;
    /// The specialized kernel's dilation-grid walk (LANE_ON_DILATION_GRID).
    bool laneOnDilationGrid = false;
    /// Output columns per lane of the specialized kernel (OUT_PER_LANE).
    int outPerLane = 2;
};

/// Threads per workgroup of both kernels.
inline constexpr unsigned int BLOCK_SIZE = 256;

/// The variant @p problem takes: the specialized one when it is NCHW with K == C and its
/// filter, padding, stride and dilation are among the built specializations, the generic one
/// otherwise.
Geometry geometryOf(const ConvProblem& problem);

int64_t gridXBlocks(const ConvProblem& problem, const Geometry& geometry);

/// Whether the kernels serve @p problem: a 2-D (or 1-D) depthwise problem (one input channel
/// per group, K a multiple of C) whose grid fits the launch limits.
bool serves(const ConvProblem& problem);

/// Whether @p kernel is the variant geometryOf picks for @p problem: the specialization of its
/// filter, stride and dilation, or the generic variant; dtype and layout always match. One
/// candidate per problem.
bool isVariantOf(const hipdnn_plugin_sdk::ingestor::KernelDefinition& kernel,
                 const ConvProblem& problem);

/// The defines a variant is built with, for a kernel compiled at run time.
void addCompileOptions(compilation::KernelCompileOptions& options,
                       const ConvProblem& problem,
                       const Geometry& geometry,
                       bool hasBias = false,
                       bool hasActivation = false);

/// The argument list of the specialized or the generic kernel: in, wei, out, [bias,] then the
/// ints [, then the activation, its three float parameters and has_bias].
const std::vector<hipdnn_plugin_sdk::ingestor::KernelArgument>&
    kernelSignature(bool generic, bool hasBias = false, bool hasActivation = false);

/// Launches the kernel over the grid of @p problem and @p geometry (set by the caller).
/// Without @p activation, @p hasBias is set exactly when the kernel was built with its bias
/// epilogue, and @p bias is then passed after `y` whatever its value. @p activation is given
/// exactly when the kernel was built with its activation epilogue, which this engine family only
/// builds together with the bias one: @p bias is then always passed, and @p hasBias false
/// (has_bias 0) launches without a bias (@p bias may then be null).
void launch(const compilation::IRunnableKernel& kernel,
            hipStream_t stream,
            const void* x,
            const void* w,
            void* y,
            const ConvProblem& problem,
            const Geometry& geometry,
            const void* bias = nullptr,
            bool hasBias = false,
            const Activation* activation = nullptr);

/// The problem and geometry values of a saved launch: `n`, `c`, `k`, per spatial axis (`_h`,
/// `_w`) `in_`, `filter_`, `out_`, `pad_`, `stride_`, `dilation_`, and the geometry `generic`,
/// `lane_on_dilation_grid`, `out_per_lane`.
void addProblemLaunchValues(hipdnn_plugin_sdk::ingestor::MetadataValues& values,
                            const ConvProblem& problem,
                            const Geometry& geometry);

/// The names addProblemLaunchValues writes.
std::vector<std::string_view> problemLaunchValueNames();

/// Reads the values addProblemLaunchValues wrote. Every extent, stride and dilation must be in
/// [1, INT_MAX], every padding in [0, INT_MAX], K a multiple of C, the geometry one the bundles
/// build, and the grid within the launch limits.
///
/// @throws HipdnnPluginException with a load refusal when a value is missing, of another type or
///         out of range, or when the values do not form a launchable problem.
void readProblemLaunchValues(const hipdnn_plugin_sdk::ingestor::SavedLaunchInputs& inputs,
                             ConvProblem& problem,
                             Geometry& geometry);

} // namespace hip_kernel_provider::kernel_ingestor_engine::conv_rdna::dw2d

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
