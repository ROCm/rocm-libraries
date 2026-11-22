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

/**
 * @file ConvRdnaFmaIgemm.hpp
 * @brief The FMA implicit-GEMM kernel (descriptors/hip/conv_rdna/kernels/ConvFwdFmaIgemm.cpp)
 *        as the RDNA forward convolution engines use it: which problems it serves, which
 *        variant a problem takes, how it is compiled and launched, and the problem half of a
 *        saved launch. `hip_conv_fwd_fma_igemm_rdna` runs it as is,
 *        `hip_conv_fwd_bias_fma_igemm_rdna` with its bias epilogue (CONV_HAS_BIAS) and
 *        `hip_conv_fwd_bias_act_fma_igemm_rdna` with the bias and activation epilogues
 *        (CONV_HAS_BIAS, CONV_HAS_ACT).
 */
namespace hip_kernel_provider::kernel_ingestor_engine::conv_rdna::fma_igemm
{

/// The KMD `kernel_family` value of every variant.
inline constexpr std::string_view FAMILY = "fma_igemm";

/// Threads per workgroup; must match kernels/ConvFwdFmaIgemm.cpp.
inline constexpr unsigned int BLOCK_SIZE = 256;

/// The activations of the CONV_HAS_ACT epilogue, numbered as the kernel's ACT_* values.
enum class ActivationMode : int32_t
{
    RELU = 0,
    SIGMOID = 1,
    TANH = 2,
    ELU = 3,
    GELU = 4,
    GELU_APPROX_TANH = 5,
    SWISH = 6,
    SOFTPLUS = 7,
    IDENTITY = 8,
};

/// An activation and its parameters as the kernel takes them: RELU reads p0 as the lower clip,
/// p1 as the upper clip and p2 as the slope below the lower clip; ELU reads p0 as alpha;
/// SWISH and SOFTPLUS read p0 as beta. The others read none.
struct Activation
{
    ActivationMode mode = ActivationMode::IDENTITY;
    float p0 = 0.0F;
    float p1 = 0.0F;
    float p2 = 0.0F;
};

/// The output channels per workgroup the variant for @p problem has: the narrowest tile that
/// holds one group's channels.
int64_t tileN(const ConvProblem& problem);

int64_t gridXBlocks(const ConvProblem& problem);
int64_t gridY(const ConvProblem& problem);

/// Whether the kernel serves @p problem: groups wide enough to fill a channel tile (depthwise
/// and narrower groups belong to a depthwise kernel) and a grid within the launch limits.
bool serves(const ConvProblem& problem);

/// Whether @p kernel is the variant of @p problem: family, dtype, layout and channel tile. Any
/// other would return wrong numbers: the grid is sized for that tile, so a kernel built for
/// another one leaves output channels unwritten or writes into the next group's.
bool isVariantOf(const hipdnn_plugin_sdk::ingestor::KernelDefinition& kernel,
                 const ConvProblem& problem);

/// The argument list of the kernel: in, wei, out, [bias,] then 22 ints [, then the activation,
/// its three float parameters and has_bias].
const std::vector<hipdnn_plugin_sdk::ingestor::KernelArgument>&
    kernelSignature(bool hasBias, bool hasActivation = false);

/// The defines a variant is built with, for a kernel compiled at run time.
void addCompileOptions(compilation::KernelCompileOptions& options,
                       const ConvProblem& problem,
                       bool hasBias,
                       bool hasActivation = false);

/// Launches the kernel over the grid of @p problem (set by the caller). @p bias is ignored
/// unless the kernel was built with CONV_HAS_BIAS; @p activation is given exactly when it was
/// built with CONV_HAS_ACT, which this engine family only builds together with the bias. With
/// an activation, @p hasBias false launches without a bias (@p bias may then be null).
void launch(const compilation::IRunnableKernel& kernel,
            hipStream_t stream,
            const void* x,
            const void* w,
            void* y,
            const void* bias,
            bool hasBias,
            const ConvProblem& problem,
            const Activation* activation = nullptr);

/// The problem values of a saved launch: `n`, `c`, `k`, `groups`, and per spatial axis (`_d`,
/// `_h`, `_w`; a 2-D plan saves depth 1) `in_`, `filter_`, `out_`, `pad_`, `stride_`,
/// `dilation_`.
void addProblemLaunchValues(hipdnn_plugin_sdk::ingestor::MetadataValues& values,
                            const ConvProblem& problem);

/// The names addProblemLaunchValues writes.
std::vector<std::string_view> problemLaunchValueNames();

/// Reads the problem values of a saved launch. Every extent, stride and dilation must be in
/// [1, INT_MAX], every padding in [0, INT_MAX], C and K must split into the groups, and the
/// grid must fit the launch limits. Dtype and layout are not saved: they are compiled into the
/// saved kernel.
///
/// @throws HipdnnPluginException with a load refusal when a value is missing, of another type
///         or out of range, or when the values do not form a launchable problem.
ConvProblem readProblemLaunchValues(const hipdnn_plugin_sdk::ingestor::SavedLaunchInputs& inputs);

} // namespace hip_kernel_provider::kernel_ingestor_engine::conv_rdna::fma_igemm

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
