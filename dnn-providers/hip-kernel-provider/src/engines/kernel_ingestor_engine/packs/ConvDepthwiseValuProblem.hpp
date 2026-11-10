// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstdint>
#include <optional>
#include <string>

#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>

#include "engines/kernel_ingestor_engine/descriptors/conv/depthwise_valu/kernels/ConvDepthwiseValuArgs.h"
#include "engines/kernel_ingestor_engine/packs/ConvNaiveProblem.hpp"

namespace hip_kernel_provider::kernel_ingestor_engine::conv_depthwise_valu
{

/// The tensor layout the VALU depthwise kernels address: channels innermost or channels
/// outside the spatial axes, every tensor packed.
enum class DepthwiseLayout
{
    CHANNELS_LAST,
    CHANNELS_FIRST,
};

/// One kernel config's variant (KMD `variant`), in the order of the kernel's CONV_DW_VARIANT.
enum class DepthwiseVariant
{
    FLOOR = 0,
    MICROTILE = 1,
    FUSED = 2,
    LDS = 3,
    /// Backward data as a gather over the taps: any stride (v5_gather_bwd).
    GATHER = 4,
    /// Backward data by stride phase blocks (v6_phase_block_*): 2-D, the tile's filter and
    /// stride, unit dilation.
    PHASE = 5,
};

/**
 * @brief A depthwise convolution as the VALU kernels run it: always a forward convolution of
 *        `in` into `out`.
 *
 * Forward reads x into y. Backward data is the forward convolution of dy into dx with the
 * padding mirrored, pre' = dilation * (filter - 1) - pre, and the filter reflected (true
 * convolution reflects the filter once more, so the two cancel) for unit stride; at any stride
 * it is also a gather over the taps (gatherArgs), the only form a strided one runs in.
 */
struct DepthwiseValuProblem
{
    conv_naive::ConvNaiveProblem base;
    DepthwiseLayout layout = DepthwiseLayout::CHANNELS_LAST;
    /// 2 for 1-D and 2-D problems (1-D runs with height 1), 3 for 3-D.
    int ndims = 2;
    /// The run convolution has unit stride and dilation on every axis: what the halo and LDS
    /// variants require besides a filter of their compile-time size.
    bool unitStrideAndDilation = false;
    /// The filter is read reflected (CONV_DW_FLIP).
    bool flipFilter = false;
    int64_t inUid = 0;
    int64_t filterUid = 0;
    int64_t outUid = 0;
    /// The kernel's argument block, pointers left null.
    ConvDepthwiseValuArgs args{};
    /// Backward data only: the gather variant's argument block, in = dy and out = dx with the
    /// forward convolution's padding, stride and dilation.
    ConvDepthwiseValuArgs gatherArgs{};
    /// Backward data with a stride: only the gather and phase variants serve it (args is unused).
    bool gatherOnly = false;
    /// True convolution: the gather variant reads the filter reflected.
    bool convolutionMode = false;
};

/**
 * @brief The depthwise convolution @p context's graph describes in @p direction, or nullopt when
 *        the VALU kernels do not serve it.
 *
 * Everything parseConvNaiveProblem gates, plus: forward or backward data; depthwise (k == c ==
 * groups); x and y packed in the same layout, channels-last or channels-first, and w packed;
 * every extent and padding within 32 bits.
 */
std::optional<DepthwiseValuProblem>
    parseDepthwiseValuProblem(const hipdnn_plugin_sdk::ingestor::MatchContext& context,
                              conv_naive::ConvDirection direction);

/// The layout x and y of @p args are both packed in, with w packed as (c, 1, spatial...), or
/// nullopt when there is none.
std::optional<DepthwiseLayout> depthwiseLayoutOf(const ConvNaiveArgs& args);

/// The KMD `layout` spelling of @p layout: "NHWC" or "NCHW", for every rank.
std::string layoutName(DepthwiseLayout layout);

/// The variant a KMD `variant` string names, or nullopt.
std::optional<DepthwiseVariant> variantNamed(const std::string& name);

/// A kernel config's tile, from its KMD metadata.
struct DepthwiseTile
{
    DepthwiseVariant variant = DepthwiseVariant::FLOOR;
    int64_t kd = 0;
    int64_t kh = 0;
    int64_t kw = 0;
    int64_t wstrip = 0;
    int64_t th = 0;
    int64_t tw = 0;
    int64_t bd = 0;
    int64_t bh = 0;
    int64_t bw = 0;
    int64_t bk = 0;
    int64_t rd = 0;
    int64_t rh = 0;
    int64_t rw = 0;
    /// Phase: the stride (height and width) the tile bakes, and channels per thread
    /// (channels-last; 0 channels-first).
    int64_t stride = 0;
    int64_t vec = 0;
};

/// The launch of @p tile on @p problem: threads per block and blocks.
struct DepthwiseLaunch
{
    int64_t block = 0;
    int64_t grid = 0;
};

/// The launch @p tile needs for @p problem, or nullopt when it exceeds a 32-bit grid. The cores
/// are not grid-stride loops, so the grid covers the whole problem.
std::optional<DepthwiseLaunch> launchFor(const DepthwiseValuProblem& problem,
                                         const DepthwiseTile& tile);

} // namespace hip_kernel_provider::kernel_ingestor_engine::conv_depthwise_valu
