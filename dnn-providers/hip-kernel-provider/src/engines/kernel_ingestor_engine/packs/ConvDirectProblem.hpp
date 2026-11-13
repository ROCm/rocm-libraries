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

#include <hipdnn_plugin_sdk/ingestor/DeviceProperties.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>

#include "engines/kernel_ingestor_engine/descriptors/conv/direct/kernels/ConvDirectArgs.h"
#include "engines/kernel_ingestor_engine/descriptors/conv/igemm/kernels/ConvIgemmArgs.h"
#include "engines/kernel_ingestor_engine/packs/ConvNaiveProblem.hpp"

/**
 * @file ConvDirectProblem.hpp
 * @brief The graph-to-kernel half of the stride-1 halo-tile direct convolution engines
 *        (CONV_DIRECT_FWD, CONV_DIRECT_BWD): which graphs they accept, how a catalog config runs
 *        one, and the workspace that takes. Kept apart from the dispatch code so the rules can be
 *        tested without a device.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::conv_direct
{

/// The filter extents (depth, height, width) the kernels are instantiated for. A 2-D filter has
/// depth 1; 1-D graphs are not served.
inline constexpr std::array<std::array<int, 3>, 7> FILTERS{{
    {1, 3, 3},
    {1, 5, 5},
    {1, 7, 7},
    {1, 1, 7},
    {1, 7, 1},
    {3, 3, 3},
    {5, 5, 5},
}};

/// A strided copy (or, for packed channels-first activations, a tiled transpose) of one
/// forward-form activation into (in) or out of (out) the workspace's packed channels-last copy.
struct Staging
{
    bool needed = false;
    bool transpose = false;
    ConvIgemmTransposeArgs transposeArgs{};
    ConvIgemmCopyArgs copy{};
    size_t workspaceOffset = 0;
    size_t bytes = 0;
};

/// A forward or backward-data convolution in the kernels' forward form: backward data is the
/// forward convolution of dy with the filter rotated and its channel roles swapped (both done by
/// the weight pre-pass), padded by F - 1 - p.
struct DirectProblem
{
    conv_naive::ConvNaiveProblem base;
    conv_naive::ConvDirection direction = conv_naive::ConvDirection::FORWARD;
    /// The forward form; pointers, tile and split are filled per config and launch.
    K2Args args{};
    int64_t xUid = 0;
    int64_t wUid = 0;
    int64_t yUid = 0;
    int elementBytes = 0;
    /// The graph filter's element strides in forward roles (k, c, t, r, s), and whether it is
    /// reflected (true convolution).
    long long wStrides[5] = {};
    int flip = 0;
    /// The forward-form input (x, or dy) and output (y, or dx), staged when not channels-last.
    Staging in;
    Staging out;
    /// Bytes of the staged activations (the start of every config's own workspace).
    size_t stagingBytes = 0;
};

/**
 * @brief The convolution @p context's graph describes in @p direction, or nullopt when the
 *        direct kernels do not serve it.
 *
 * Everything parseConvNaiveProblem gates, plus: two or three spatial axes, groups 1, stride 1
 * and dilation 1 on every axis, a filter in FILTERS, every extent, channel count and padding
 * within 32 bits, and one image of the forward-form input within 2^31 bytes (the kernels' 32-bit
 * buffer offsets). Any padding, asymmetric included (the output extent the graph declares is the
 * one computed), both convolution modes, fp32 / fp16 / bf16, and any operand layout (staged to
 * channels-last when it is not).
 */
std::optional<DirectProblem>
    parseDirectProblem(const hipdnn_plugin_sdk::ingestor::MatchContext& context,
                       conv_naive::ConvDirection direction);

/// A catalog config's compile-time choices, read from its metadata.
struct Config
{
    bool fma = false;
    std::string atom; ///< conv_igemm atom name ("fma" for the FMA kernel)
    std::string_view kernelType; ///< K1Atom* type of the atom; empty for FMA
    int wave = 0; ///< 64 MFMA, 32 WMMA, 0 FMA
    int ka = 1; ///< the atom's reduction step (weight format granule); 1 for FMA
    bool math = false; ///< three-bf16 (fp32 graph, bf16 operands): TF32 permission only
    int kd = 0, r = 0, s = 0;
    int bm = 0, bn = 0, bc = 0, wm = 0, wn = 0, vec = 0;
    int ring = 1, bl = 0, split = 0;
    int tm = 0, tn = 0, nt = 0, threads = 0, pipe = 0;
    /// The filter-row reuse kernel (kind "rows", ConvDirectRows.h): waves WP x WQ x WK, each PW
    /// output rows x 16 columns x 16 FN channels; ping-pong (pp) and staged epilogue (epi).
    bool rows = false;
    int pw = 0, fn = 0, wp = 0, wq = 0, wk = 0, pp = 0, epi = 0;
    std::string archs; ///< ship list: comma-separated arch groups, or "any"

    /// Output pixels per block (the planner's tile budget) and output channels per block.
    int blockPixels() const
    {
        return fma ? (threads / nt) * tm : rows ? wp * pw * wq * 16 : bm;
    }
    int blockColumns() const
    {
        return fma ? nt * tn : rows ? wk * fn * 16 : bn;
    }
    int blockThreads() const
    {
        return fma ? threads : rows ? wp * wq * wk * wave : wm * wn * wave;
    }
};

/// The config @p kernel describes, or nullopt for metadata the pack does not know.
std::optional<Config> configOf(const hipdnn_plugin_sdk::ingestor::KernelDefinition& kernel);

/// Whether @p archs (a config's ship list) names the device @p gcnArchName (suffix intact):
/// "any", the exact architecture, or its family ("gfx11", "gfx12").
bool shipsTo(std::string_view archs, std::string_view gcnArchName);

/// LDS per workgroup on @p gcnArchName: 160 KiB on CDNA4 (gfx950), 64 KiB on every other family.
int64_t ldsLimitOf(std::string_view gcnArchName);

/// LDS a rows config's kernel declares: two halo stages of (wp pw + r - 1) x (16 wq + s - 1)
/// pixels x bc channels of the atom's operand (ConvDirectRows.h; the staged epilogue reuses a
/// stage). 0 for a config of another kind.
int64_t rowsLdsBytes(const Config& config);

/// Whether the device runs @p config: its atom exists there, its ship list names it, and (rows)
/// its halo stages fit the device's LDS.
bool configRunsOn(const Config& config, std::string_view gcnArchName);

/// How a config runs a problem: the tile, the split, the launch grid and the workspace.
struct Plan
{
    K2Args args{}; ///< the problem with this config's tile and split; pointers filled at launch
    unsigned int grid[3] = {1, 1, 1};
    int cpad = 0; ///< reduction channels of the formatted weights (x3 for three-bf16)
    long long weightElems = 0;
    long long outElems = 0;
    size_t weightOffset = 0, weightBytes = 0;
    size_t splitInputOffset = 0, splitInputBytes = 0; ///< three-bf16 input
    size_t partialOffset = 0, partialBytes = 0; ///< split-C fp32 partials
    size_t workspaceBytes = 0;
};

/**
 * @brief The plan @p config runs @p problem with on a device of @p computeUnits, or nullopt when
 *        the config cannot (its filter or channel vector, the LDS halo bound, launch or 32-bit
 *        buffer limits) or the device reports no units (@p computeUnits < 1).
 *
 * @p computeUnits counts the units workgroups are dispatched onto (device_units::dispatchUnits:
 * CUs on gfx9, WGPs on gfx10+); the per-unit block targets below are in that unit.
 *
 * A rows config has a fixed tile (wp pw rows x 16 wq columns) and launches at most one block per
 * compute unit, each walking its share of the tiles (rowsLdsBytes: up to ~153 KiB on gfx950).
 *
 * The tile is th x tw output pixels (tw a multiple of the FMA run length) of ds planes, chosen to
 * minimize padded pixel slots plus halo loads within the config's halo bound; depth is split
 * until there are 4 (ring) or 8 blocks per compute unit. A split-C config splits the channel
 * chunks when the grid has fewer than 2 blocks per compute unit, keeping 2 chunks per split and
 * at most 16 splits.
 */
std::optional<Plan>
    planFor(const DirectProblem& problem, const Config& config, int64_t computeUnits);

} // namespace hip_kernel_provider::kernel_ingestor_engine::conv_direct

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
