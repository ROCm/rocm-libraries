// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstdint>
#include <optional>
#include <string_view>

#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>
#include <hipdnn_plugin_sdk/ingestor/DeviceProperties.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>

#include "engines/kernel_ingestor_engine/descriptors/conv/naive/kernels/ConvNaiveArgs.h"

/**
 * @file ConvNaiveProblem.hpp
 * @brief The graph-to-kernel half of the naive convolution engines: which single-node
 *        convolution graphs they accept, and the argument block their kernel launches with.
 *
 * Shared by the three engines (forward, backward data, backward weights) because the
 * operands differ only in name. Kept apart from the dispatch code so the acceptance rules
 * and the workload gate can be tested without a device.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::conv_naive
{

enum class ConvDirection
{
    FORWARD,
    BACKWARD_DATA,
    BACKWARD_WEIGHTS,
};

/// Tensor uids in forward-convolution roles: x is dx for backward data, w is dw for
/// backward weights, y is dy for both backward directions.
struct ConvNaiveOperands
{
    int64_t x = 0;
    int64_t w = 0;
    int64_t y = 0;
};

/// Everything a launch needs, derived from the graph alone. The pointers in `args` are left
/// null; dispatch fills them from the variant pack.
struct ConvNaiveProblem
{
    ConvDirection direction = ConvDirection::FORWARD;
    hipdnn_flatbuffers_sdk::data_objects::DataType dataType
        = hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET;
    ConvNaiveOperands operands;
    int64_t spatialRank = 0;
    /// Multiply-accumulates over the whole problem; the same for every direction.
    int64_t multiplyAccumulates = 0;
    ConvNaiveArgs args{};
};

/// The engine name of each direction, as the UED declares it.
std::string_view engineName(ConvDirection direction);

/**
 * @brief The problem a single-node convolution graph describes, or nullopt when the graph
 *        is not one the naive kernel computes correctly.
 *
 * Reads and gates every field of the graph, the node and the three operands. Does not apply
 * the workload gate; see withinWorkloadBudget.
 */
std::optional<ConvNaiveProblem>
    parseConvNaiveProblem(const hipdnn_plugin_sdk::ingestor::MatchContext& context,
                          ConvDirection direction);

/**
 * @brief Multiply-accumulates each SIMD lane may spend on one launch before the problem is
 *        declined as too large for a naive kernel.
 *
 * Starting value, to be calibrated by timing the kernels across architectures: about 1 ms at
 * a conservative 1 GHz and 0.005 multiply-accumulates per lane-cycle, the low end of what the
 * MIOpen naive kernels measured on gfx950 and gfx1151 (MIOpen PR #11160 calibration).
 */
inline constexpr int64_t NAIVE_MACS_PER_LANE_BUDGET = 5000;

/// SIMD lanes per compute unit. 64 on both CDNA (4 x SIMD16) and RDNA (2 x SIMD32); RDNA
/// devices report WGPs (two CUs), which device_units::computeUnits doubles.
inline constexpr int64_t LANES_PER_COMPUTE_UNIT = 64;

/// Compute units assumed when the device does not report its count. The naive kernel is the
/// fallback every graph keeps, so an unknown count is guessed low rather than declined.
inline constexpr int64_t FALLBACK_COMPUTE_UNITS = 16;

/// Lanes that share one dw element's reduction in the backward-weights kernel (its block, a
/// fixed-order tree sum): ConvNaive.cpp's CONV_NAIVE_WRW_BLOCK.
inline constexpr int64_t NAIVE_WRW_LANES_PER_ELEMENT = 256;

/**
 * @brief True when the naive kernel finishes @p problem within budget on @p device.
 *
 * Forward and backward data: each output element is one thread that runs its
 * multiply-accumulates in sequence. Backward weights: each dw element is one block of
 * NAIVE_WRW_LANES_PER_ELEMENT lanes. The time follows the work per lane: the total over the
 * lanes the problem can occupy, the smaller of the device's lanes and the output elements'
 * lanes.
 */
bool withinWorkloadBudget(const ConvNaiveProblem& problem,
                          const hipdnn_plugin_sdk::ingestor::DeviceProperties& device);

} // namespace hip_kernel_provider::kernel_ingestor_engine::conv_naive

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
