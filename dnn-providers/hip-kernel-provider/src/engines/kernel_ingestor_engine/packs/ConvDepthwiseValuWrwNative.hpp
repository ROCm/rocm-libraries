// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

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
    /// The depth multiplier: output channels per input channel (k / c).
    int64_t mult = 1;
    /// The kernels' argument block, pointers left null and the partitions unset until
    /// partitionFor.
    ConvDepthwiseValuWrwArgs args{};
};

/**
 * @brief The depthwise backward weights @p context's graph describes, or nullopt when the VALU
 *        kernels do not serve it.
 *
 * Everything parseConvNaiveProblem gates, plus: depthwise (c == groups, k a whole multiple of it:
 * mult = k / c output channels per input channel, output channel k reading input channel
 * k / mult); x and dy packed in
 * the same layout, channels-last or channels-first, and dw packed; every extent, padding,
 * stride, dilation, the reduction rows and the (channel, tap) count within 32 bits.
 */
std::optional<WrwProblem> parseWrwProblem(const hipdnn_plugin_sdk::ingestor::MatchContext& context);

/// A kernel config, from its metadata. Variants: "floor" (any problem), "tile" (VALU register
/// tile of a compile-time kd x kh x kw filter), "hankel" (matrix core, 4x4x4 MFMA: q output
/// columns a tile, g input rows a step, nwg 16-channel groups a wave, bpc blocks per dispatch
/// unit the partitions aim for). The hankel fields are 0 for the VALU variants.
struct WrwConfig
{
    std::string variant;
    int64_t kd = 0;
    int64_t kh = 0;
    int64_t kw = 0;
    int64_t q = 0;
    int64_t g = 0;
    int64_t nwg = 0;
    int64_t bpc = 0;
    /// The depth multiplier the kernel is compiled for (CONV_DW_MULT); 1 for hankel.
    int64_t mult = 1;
};

/// The LDS a hankel config's partial kernel declares (ConvDepthwiseHankelWrw.h's arithmetic).
int64_t hankelLdsBytes(const WrwConfig& config);

/**
 * @brief Whether the hankel config @p config serves @p problem on @p gcnArchName: 4x4x4 MFMA
 *        (gfx908 for fp16; gfx90a, gfx942, gfx950 for fp16 and bf16); fp16 or bf16; mult 1;
 *        channels-last, 1-D or 2-D; the config's filter; unit stride and dilation; channels a
 *        whole number of 8-channel vectors; q a multiple of 4 from 4 to 64, g 1 to 4, nwg 1, 2
 *        or 4, bpc 1, 2, 4 or 8; the LDS within the device's (160 KiB on gfx950, else 64 KiB).
 */
bool hankelAdmits(const WrwProblem& problem, const WrwConfig& config, std::string_view gcnArchName);

/// Splits @p problem's reduction rows into partitions enough to fill @p computeUnits, counted in
/// dispatch units (device_units: CUs on gfx9, WGPs on gfx10+), at @p config's blocks per unit
/// (4 for the VALU variants) and channels per block.
void partitionFor(WrwProblem& problem, int64_t computeUnits, const WrwConfig& config);

/// The partial sums' workspace, once partitioned.
size_t workspaceBytesFor(const WrwProblem& problem);

/// The two launches: the partial kernel's grid, the fold kernel's, both of `block` threads.
struct WrwLaunch
{
    int64_t block = 0;
    int64_t partialGrid = 0;
    int64_t foldGrid = 0;
};

/// The launches for @p problem partitioned for @p config, or nullopt when a grid exceeds 32
/// bits.
std::optional<WrwLaunch> launchFor(const WrwProblem& problem, const WrwConfig& config);

} // namespace hip_kernel_provider::kernel_ingestor_engine::conv_depthwise_valu_wrw
