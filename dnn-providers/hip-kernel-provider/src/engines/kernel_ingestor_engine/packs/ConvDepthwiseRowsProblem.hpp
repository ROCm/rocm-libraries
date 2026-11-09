// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstdint>
#include <optional>
#include <string>

#include "engines/kernel_ingestor_engine/packs/ConvDepthwiseValuProblem.hpp"

/**
 * @file ConvDepthwiseRowsProblem.hpp
 * @brief The host half of the channel-row depthwise engines (CONV_DEPTHWISE_ROWS_FWD/BWD):
 *        the config a kernel's metadata declares, whether it serves a problem, and its launch.
 *
 * The problem is the VALU depthwise engines' (parseDepthwiseValuProblem): the same graphs, the
 * same forward form of backward data. Kept apart from the dispatch code so every gate can be
 * tested without a device.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::conv_depthwise_rows
{

/// The kernel's core, in the order of its CONV_DWR_VARIANT.
enum class RowsVariant
{
    /// Forward plane tile, channels-first: LDS halo, register tile on the dilation grid.
    TILE = 0,
    /// Forward band of whole rows, channels-first: LDS halo band, one column per unit.
    BAND = 1,
    /// Backward data of a row-separable strided convolution by stride phases, a block per row.
    PHASE = 2,
    /// The same phases, channels-last, lanes over (position, channel).
    LANES = 3,
};

/// One kernel config: the KMD fields minus dtype and direction. Fields a variant does not use
/// are 0; a config that sets one is malformed and serves nothing.
struct RowsConfig
{
    std::string layout;
    int64_t variant = -1;
    /// Compile-time filter extent, 0 = read at runtime.
    int64_t kh = 0;
    int64_t kw = 0;
    /// Tile: threads along the width and height, outputs per thread along each.
    int64_t tx = 0;
    int64_t ty = 0;
    int64_t rw = 0;
    /// Tile and band: output rows per thread.
    int64_t rh = 0;
    /// Band: output rows per block.
    int64_t bh = 0;
    /// Threads per block (tile: tx * ty; band: the most it launches).
    int64_t threads = 0;
    /// Phase and lanes: the stride along the width.
    int64_t stride = 0;
    /// Phase: phase indices per thread.
    int64_t q = 0;
    /// Tile and band: the halo's LDS capacity in fp32 elements.
    int64_t lds = 0;
    /// Phase: the most taps a phase may hold (the dy window's capacity).
    int64_t mmax = 0;
};

/// The launch of a config on a problem.
struct RowsLaunch
{
    int64_t block = 0;
    int64_t grid = 0;
    /// The kernel's argument block, pointers left null: the forward form (tile, band) or the
    /// gather form (phase, lanes).
    ConvDepthwiseValuArgs args{};
    /// The filter is read reflected (CONV_DWR_FLIP).
    bool flip = false;
};

/// The variant @p value names, or nullopt.
std::optional<RowsVariant> rowsVariantOf(int64_t value);

/**
 * @brief The launch of @p config on @p problem in @p direction, or nullopt when the config does
 *        not compute it correctly or cannot launch it.
 *
 * Reads and gates every field of @p config: the layout and the variant's direction and
 * geometry (the tile and band run the forward form, so backward data only at unit stride; the
 * phases need a row-separable 1-D problem with unit dilation along the width), the filter
 * extent, the block shape, the stride, the halo against the LDS capacity and the taps per phase
 * against the window, and that every field the variant does not use is 0.
 */
std::optional<RowsLaunch> rowsLaunchFor(const conv_depthwise_valu::DepthwiseValuProblem& problem,
                                        conv_naive::ConvDirection direction,
                                        const RowsConfig& config);

/// Extent of the input a run of @p outputs outputs reads along one axis (the kernel's
/// rowsHaloExtent).
int64_t rowsHaloExtent(int64_t outputs, int64_t stride, int64_t filter, int64_t dilation);

/// Threads a band block launches (the kernel's rowsBandThreads).
int64_t rowsBandThreads(int64_t groups, int64_t wo, int64_t maxThreads);

} // namespace hip_kernel_provider::kernel_ingestor_engine::conv_depthwise_rows
