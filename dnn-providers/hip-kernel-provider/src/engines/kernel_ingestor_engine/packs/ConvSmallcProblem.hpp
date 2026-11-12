// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>

#include "engines/kernel_ingestor_engine/descriptors/conv/smallc/kernels/ConvSmallcArgs.h"
#include "engines/kernel_ingestor_engine/packs/ConvNaiveProblem.hpp"

/**
 * @file ConvSmallcProblem.hpp
 * @brief The graph and config half of the small-input-channel convolution engines
 *        (CONV_SMALLC_FWD, and CONV_SMALLC_BWD for stride-1 backward data with few
 *        output-gradient channels): which graphs they take, what a catalog config declares, and
 *        how a config runs a problem. Kept apart from the dispatch code so the rules can be
 *        tested without a device.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::conv_smallc
{

/// A convolution the engine can run, as the forward convolution its kernel computes: groups 1,
/// 2-D or 3-D, fp32/fp16/bf16, any layout (element strides), one kernel input image within
/// 32-bit byte offsets.
///
/// Backward data (unit stride on every axis) is the forward convolution over dy that writes dx:
/// kernel input channels K, output channels C, input extents dy's, output extents dx's, padding
/// (filter - 1) dilation - pre-padding per axis (negative where the graph pads past the filter),
/// and the filter transposed (k <-> c) and reflected by the weight pre-pass.
struct SmallcProblem
{
    conv_naive::ConvNaiveProblem base;
    bool backward = false; ///< backward data
    int elementBytes = 0;
    SmallcArgs args{}; ///< extents, padding and strides; pointers, tile and kpad by the plan
};

/// The problem @p context's graph describes in @p direction (FORWARD or BACKWARD_DATA), or
/// nullopt.
std::optional<SmallcProblem>
    parseSmallcProblem(const hipdnn_plugin_sdk::ingestor::MatchContext& context,
                       conv_naive::ConvDirection direction = conv_naive::ConvDirection::FORWARD);

/// A catalog config: compile-time filter geometry and tile, read from its metadata.
struct Config
{
    std::string atom; ///< conv_igemm 16x16 atom name, or "fma" for the FMA kernel
    bool fma = false;
    std::string_view kernelType; ///< K1Atom* type
    std::string archs; ///< ship list
    int wave = 0;
    int ka = 0; ///< the atom's reduction step
    int cmax = 0;
    int kd = 0, r = 0, s = 0;
    int sd = 0, sh = 0, sw = 0;
    int dd = 0, dh = 0, dw = 0;
    int fm = 0, fn = 0, wm = 0, wq = 0, wn = 0;
    int ep = 0; ///< epilogue: 0 through LDS slots, 1 from registers (MMA, channels-fastest output)

    /// Taps per filter row in the weight layout: s rounded up to a power of two.
    int sp() const;
    /// Formatted weight row length: cmax kd r sp rounded up to whole MMA steps.
    int kpl() const;
    /// Output channels per block: 16 fn wn (MMA), 16 wn (FMA).
    int blockColumns() const
    {
        return fma ? wn * 16 : wn * fn * 16;
    }
    int blockThreads() const
    {
        return fma ? 256 : wm * wq * wn * wave;
    }
};

/// The config @p kernel describes, or nullopt for metadata the pack does not know.
std::optional<Config> configOf(const hipdnn_plugin_sdk::ingestor::KernelDefinition& kernel);

/// How a config runs a problem: the tile counts, the launch grid and the formatted weights.
struct Plan
{
    SmallcArgs args{};
    unsigned int grid[2] = {1, 1};
    long long weightElems = 0;
    size_t weightBytes = 0;
};

/**
 * @brief The plan @p config runs @p problem with, or nullopt when the config does not serve it:
 *        its channel bound, filter, stride or dilation is not the graph's, its register epilogue
 *        (ep 1) meets an output whose channels are not adjacent, or the launch exceeds the grid
 *        or the 32-bit offsets.
 */
std::optional<Plan> planFor(const SmallcProblem& problem, const Config& config);

} // namespace hip_kernel_provider::kernel_ingestor_engine::conv_smallc

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
