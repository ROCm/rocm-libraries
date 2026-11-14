// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>

#include "engines/kernel_ingestor_engine/descriptors/conv/group_valu_wrw/kernels/ConvGroupValuWrwArgs.h"
#include "engines/kernel_ingestor_engine/packs/ConvNaiveProblem.hpp"

/**
 * @file ConvGroupValuWrwProblem.hpp
 * @brief The graph and config half of CONV_GROUP_VALU_WRW (backward weights on vector ALU, the
 *        design of MIOpen's ConvAsmBwdWrW3x3 in HIP, ConvGroupValuWrw.cpp): which graphs it
 *        takes, what a catalog config declares and how a config runs a problem (lanes a row,
 *        load widths, splits, workspace, grid). Kept apart from the dispatch code so the rules
 *        can be tested without a device.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::conv_group_valu_wrw
{

/// A backward weights the engine serves: one conv node, 1-D, 2-D or 3-D (a tile per depth tap),
/// fp32 / fp16 / bf16 tensors of one element type, compute type unset or fp32, any groups, any
/// padding, stride and dilation (a config fixes the width's), both conv modes, extents and
/// channel counts within 32 bits. x and dy either both channels-first with the width packed (`nchw`) or both
/// channels-last with the channels packed (`nhwc`); dw any strides.
struct GvwProblem
{
    conv_naive::ConvNaiveProblem base;
    int n = 0;
    int groups = 0;
    int cpg = 0; ///< input channels a group
    int kpg = 0; ///< output channels a group
    int hi = 0, wi = 0; ///< x extents (1-D: hi = 1)
    int ho = 0, wo = 0; ///< dy extents
    int r = 0, s = 0; ///< filter extents
    int ph = 0, pw = 0; ///< pre-padding
    int sh = 1, sw = 1;
    int dh = 1, dw = 1;
    int di = 1; ///< x depth (below 3-D: 1)
    int dOut = 1; ///< dy depth
    int t = 1; ///< filter depth taps
    int pd = 0; ///< depth pre-padding
    int sd = 1; ///< depth stride
    int dd = 1; ///< depth dilation
    long long xsd = 0; ///< depth element strides of x, dy and dw (below 3-D: 0)
    long long ysd = 0;
    long long wsd = 0;
    int flip = 0; ///< convolution mode: the filter is reflected
    int elementBytes = 0;
    bool channelsFirst = false; ///< x and dy: the width packed (xs_w = ys_w = 1)
    bool channelsLast = false; ///< x and dy: the channels packed (xs_c = ys_c = 1)
    // NOLINTBEGIN(modernize-avoid-c-arrays) -- mirrors the kernel argument block's C arrays
    long long xStr[4] = {}; ///< n, c, h, w element strides
    long long wStr[4] = {}; ///< k, c, r, s
    long long yStr[4] = {}; ///< n, k, h, w
    // NOLINTEND(modernize-avoid-c-arrays)
    long long xBytes = 0; ///< bytes from x's first element to one past its last
    long long yBytes = 0;
    int xAlign = 0; ///< the byte alignment the graph declares for x's pointer, at most 16
    int yAlign = 0; ///< the same for dy
};

/// The problem @p context's graph describes (backward weights), or nullopt.
std::optional<GvwProblem> parseGvwProblem(const hipdnn_plugin_sdk::ingestor::MatchContext& context);

/// A catalog config, from its metadata: the layout it reads (`nchw` or `nhwc`), the tile (cb
/// input by kb output channels), output columns a lane (p), the filter it is compiled for (fr x
/// fs taps, width stride sw, width dilation dw, width pre-padding pw), the waves a workgroup
/// (nw), whether fp16 pairs of output columns multiply on v_dot2_f32_f16 (dot2), and whether a
/// lane group walks consecutive output rows of one image with the filter's input rows in a
/// register ring (ring: channels-first, unit height stride and dilation).
struct Config
{
    std::string layout;
    int cb = 0;
    int kb = 0;
    int p = 0;
    int fr = 0;
    int fs = 0;
    int sw = 0;
    int dw = 0;
    int pw = 0;
    int nw = 0;
    int dot2 = 0;
    int ring = 0;
};

/// Accumulators a lane holds (cb kb fr fs): at most this many, so a tile stays in registers.
inline constexpr int MAX_ACCUMULATORS = 160;

/// The config @p kernel describes, or nullopt for one the kernels cannot compile: layout nchw or
/// nhwc; cb, kb 1, 2, 4, 8 or 16; p 1 to 16; fr, fs 1 to 7; sw 1 or 2; dw 1 to 4; pw at least 0
/// and the window it and the filter reach within one lane either side; nw 1, 2, 4 or 8; at most
/// MAX_ACCUMULATORS accumulators; dot2 0, or 1 with layout nchw, sw 1 and p even; ring 0, or 1
/// with layout nchw.
std::optional<Config> configOf(const hipdnn_plugin_sdk::ingestor::KernelDefinition& kernel);

/// Upper bound of the workspace a plan asks for (the splits' fp32 sums).
inline constexpr size_t WORKSPACE_CAP_BYTES = size_t{1} << 28;

/// How a config runs a problem on a device.
struct Plan
{
    Config config;
    ConvGroupValuWrwArgs args{}; ///< every field but the pointers
    int wave = 0; ///< lanes a wave
    int vbx = 0; ///< bytes a load of x
    int vby = 0; ///< bytes a load of dy
    bool ragged = false;
    bool colmask = false; ///< output columns past the row: their products are dropped
    int accumulators = 0; ///< filter elements a tile (cb kb fr fs)
    size_t workspaceBytes = 0;
    long long foldGrid = 0; ///< blocks of the fold launch (0: one split, no fold)
};

/// Lanes of a wave on @p gcnArchName when the device does not report it: 64 on gfx9, 32 on
/// gfx10 and later.
int waveOf(std::string_view gcnArchName, int reportedWarpSize);

/**
 * @brief The plan @p config runs @p problem with on a device of @p wave lanes and
 *        @p computeUnits dispatch units (0: unknown), or nullopt when the config does not serve
 *        it: another filter, width stride, dilation or padding, another layout, channels per
 *        group not a multiple of the tile, a row wider than a wave's lanes cover, or offsets
 *        within the images a pass spans past 31 bits.
 */
std::optional<Plan>
    planFor(const GvwProblem& problem, const Config& config, int wave, int64_t computeUnits);

} // namespace hip_kernel_provider::kernel_ingestor_engine::conv_group_valu_wrw

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
