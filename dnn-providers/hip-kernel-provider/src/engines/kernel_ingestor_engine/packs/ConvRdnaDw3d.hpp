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
 * @file ConvRdnaDw3d.hpp
 * @brief The depthwise 3D kernels (descriptors/hip/conv_dw3d_rdna/kernels/ConvFwdDepthwise3d.cpp)
 *        as the RDNA engines use them: which problems they serve, which variant a problem takes
 *        and how it tiles the output, how a variant is compiled and launched, and the problem
 *        half of a saved launch. `hip_conv_fwd_dw3d_rdna` runs them as they are,
 *        `hip_conv_fwd_bias_dw3d_rdna` with their bias epilogue (DW_HAS_BIAS) and
 *        `hip_conv_fwd_bias_act_dw3d_rdna` with the bias and activation epilogues (DW_HAS_BIAS,
 *        DW_HAS_ACT).
 */
namespace hip_kernel_provider::kernel_ingestor_engine::conv_rdna::dw3d
{

/// How a depthwise 3D variant tiles the output; the grid depends on it. Dtype, layout, the
/// specialized filter, H / W stride and dilation are compiled into the variant and only matter
/// for picking it.
struct Geometry
{
    /// The runtime-parameter kernel (filter, stride and dilation as launch arguments).
    bool generic = false;
    /// Output depths and columns per thread (the specialized kernel; the generic one is 1 x 4).
    int tileD = 1;
    int tileW = 4;
    /// NCDHW grid with one (n, k) output volume per blockIdx.y and 64-thread blocks.
    bool uniformK = false;
};

/// The variant @p problem takes: the specialized one when its filter, H / W stride and dilation
/// are among the built specializations and its grid fits, the generic one otherwise.
Geometry geometryOf(const ConvProblem& problem);

unsigned int blockSize(const Geometry& geometry);
int64_t gridXBlocks(const ConvProblem& problem, const Geometry& geometry);
int64_t gridY(const ConvProblem& problem, const Geometry& geometry);

/// Whether the kernels serve @p problem: a 3-D depthwise problem (one input channel per group,
/// K a multiple of C) whose grid fits the launch limits.
bool serves(const ConvProblem& problem);

/// Whether @p kernel is the variant geometryOf picks for @p problem: the specialization of its
/// filter, H / W stride and tile, or the generic variant; dtype and layout always match. One
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
/// Without @p activation, @p hasBias is set exactly when the kernel was built with DW_HAS_BIAS,
/// and @p bias is then passed after `y` whatever its value. @p activation is given exactly when
/// the kernel was built with DW_HAS_ACT, which this engine family only builds together with
/// DW_HAS_BIAS: @p bias is then always passed, and @p hasBias false (has_bias 0) launches without
/// a bias (@p bias may then be null).
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

/// The problem and geometry values of a saved launch: `n`, `c`, `k`, per spatial axis (`_d`,
/// `_h`, `_w`) `in_`, `filter_`, `out_`, `pad_`, `stride_`, `dilation_`, and the geometry
/// `generic`, `tile_d`, `tile_w`, `uniform_k`.
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

} // namespace hip_kernel_provider::kernel_ingestor_engine::conv_rdna::dw3d

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
