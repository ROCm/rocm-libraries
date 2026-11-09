// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

#include <hipdnn_plugin_sdk/ingestor/DeviceProperties.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>

#include "engines/kernel_ingestor_engine/descriptors/conv/winograd/kernels/ConvWinogradArgs.h"
#include "engines/kernel_ingestor_engine/packs/ConvNaiveProblem.hpp"

/**
 * @file ConvWinogradProblem.hpp
 * @brief The graph-to-kernel half of the Winograd convolution engines (CONV_WINOGRAD_FWD,
 *        CONV_WINOGRAD_BWD): which graphs they accept, how a catalog config runs one, and the
 *        workspace that takes. Kept apart from the dispatch code so the rules can be tested
 *        without a device.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::conv_winograd
{

/// How an activation pair (x and y) or a filter is laid out: packed channels-first (NCHW, or
/// KCRS for the filter), packed channels-last (NHWC, KRSC), or anything else.
enum class Layout
{
    NCHW,
    NHWC,
    OTHER,
};

/// A 2-D, 3x3, stride-1 or stride-2, dilation-1 convolution in the kernels' forward form (see
/// ConvWinogradArgs.h; stride 2 as its phase convolution, `mode` 1 or 2). Pointers and tile counts are filled per config and launch.
struct WinogradProblem
{
    conv_naive::ConvNaiveProblem base;
    conv_naive::ConvDirection direction = conv_naive::ConvDirection::FORWARD;
    K5Args args{};
    int64_t xUid = 0;
    int64_t wUid = 0;
    int64_t yUid = 0;
    int elementBytes = 0;
    /// x and y together; OTHER unless both are packed the same way.
    Layout activations = Layout::OTHER;
    /// The filter as the graph holds it.
    Layout filter = Layout::OTHER;
};

/**
 * @brief The convolution @p context's graph describes in @p direction, or nullopt when the
 *        Winograd kernels do not serve it.
 *
 * Everything parseConvNaiveProblem gates, plus: two spatial axes, a 3x3 filter, stride 1 or 2
 * (the same on both axes), dilation 1, and every extent, channel count and padding within 32 bits. Any padding,
 * asymmetric included (the output extent the graph declares is the one computed), both
 * convolution modes, groups, fp32 / fp16 / bf16, and any operand layout (staged when the
 * variant does not read it).
 */
std::optional<WinogradProblem>
    parseWinogradProblem(const hipdnn_plugin_sdk::ingestor::MatchContext& context,
                         conv_naive::ConvDirection direction);

enum class Variant
{
    MULTIPASS,
    FUSED,
    WMMA,
};

/// A catalog config's compile-time choices, read from its metadata.
struct Config
{
    Variant variant = Variant::MULTIPASS;
    int m = 2; ///< output tile: F(m,3), or F(3,2) (m = 3) on stride 2's phase sub-filters
    Layout layout = Layout::NCHW; ///< what the kernels read; the fused variants read NCHW
    std::string arch; ///< "any", or the WMMA family: "gfx11", "gfx12"
    bool llcPlan = false; ///< multipass: slice the batch to fit the last-level cache
    int gemmBm = 0, gemmBn = 0, gemmBk = 0; ///< multipass GEMM tile
    int bk = 0, bt = 0, ccv = 0, tm = 0, tn = 0; ///< fused VALU block
    int waves = 0; ///< fused WMMA waves per workgroup
};

/// The config @p kernel describes, or nullopt for metadata the pack does not know.
std::optional<Config> configOf(const hipdnn_plugin_sdk::ingestor::KernelDefinition& kernel);

/// Whether the device @p gcnArchName (suffix intact) runs @p config's instructions.
bool configRunsOn(const Config& config, std::string_view gcnArchName);

/// A strided copy of one operand into (read) or out of (written) the workspace.
struct Staging
{
    bool needed = false;
    K5CopyArgs copy{};
    size_t workspaceOffset = 0;
    size_t bytes = 0;
};

/// How a config runs a problem: what it stages, the batch slices, and the workspace.
struct Plan
{
    /// The problem with this config's tile counts for one full slice.
    K5Args args{};
    Staging in; ///< forward-form input (x, or dy)
    Staging out; ///< forward-form output (y, or dx)
    Staging filter;
    size_t uOffset = 0, uBytes = 0;
    size_t vOffset = 0, vBytes = 0;
    size_t mOffset = 0, mBytes = 0;
    int sliceImages = 1; ///< images per launch
    int slices = 1;
    size_t workspaceBytes = 0;
    /// Launch grids for one slice: transforms and GEMM (multipass), or the persistent kernel.
    unsigned int filterBlocks[3] = {1, 1, 1};
    unsigned int inputBlocks[3] = {1, 1, 1};
    unsigned int gemmBlocks[3] = {1, 1, 1};
    unsigned int outputBlocks[3] = {1, 1, 1};
    unsigned int mainBlocks = 1;
    unsigned int mainThreads = 256;
};

/// Workspace limit the batch slicing keeps the multipass buffers under (the engine's own
/// default; HIP_KERNEL_PROVIDER_CONV_WINOGRAD_WORKSPACE_MB overrides).
size_t workspaceLimit();

/// The last-level (MALL) cache of an RDNA device, or 0 where the LLC-fit plan does not apply.
/// HIP_KERNEL_PROVIDER_CONV_WINOGRAD_LLC_BYTES overrides it on any device (tests).
size_t llcBytes(std::string_view gcnArchName);

/// The plan @p config runs @p problem with on @p device, or nullopt when the config does not
/// serve it (layout, launch limits, buffer limits, or an LLC plan that would not slice).
std::optional<Plan> planFor(const WinogradProblem& problem,
                            const Config& config,
                            const hipdnn_plugin_sdk::ingestor::DeviceProperties& device);

} // namespace hip_kernel_provider::kernel_ingestor_engine::conv_winograd

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
