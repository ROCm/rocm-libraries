// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file CartesianProduct.hpp
 * @brief Cartesian product generation for knob sweep autotuning
 *
 * Provides a function to compute the Cartesian product of knob sweep axes,
 * used by add_engine_sweep() to expand an EngineSweepSpec into individual
 * plan specs. Includes configurable safety limits to prevent combinatorial
 * explosion.
 */

#pragma once

#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/Logging.hpp>
#include <hipdnn_frontend/autotune/PlanSpec.hpp>

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace hipdnn_frontend
{
namespace autotune
{

/// Maximum number of combinations before returning an error
static constexpr size_t CARTESIAN_PRODUCT_LIMIT_ERROR = 10000;

/// Threshold for logging a warning about large Cartesian products
static constexpr size_t CARTESIAN_PRODUCT_LIMIT_WARNING = 1000;

/**
 * @brief Compute the Cartesian product of knob sweep axes
 *
 * Given a vector of KnobSweepAxis (each containing a knob name and a list
 * of values), produces all combinations of knob settings.
 *
 * For example, given axes:
 *   - {"SPLIT_K", {1, 2, 4}}
 *   - {"TILE_SIZE", {0, 1}}
 *
 * The result contains 6 vectors of KnobSetting:
 *   [{SPLIT_K=1, TILE_SIZE=0}, {SPLIT_K=1, TILE_SIZE=1},
 *    {SPLIT_K=2, TILE_SIZE=0}, {SPLIT_K=2, TILE_SIZE=1},
 *    {SPLIT_K=4, TILE_SIZE=0}, {SPLIT_K=4, TILE_SIZE=1}]
 *
 * @param axes Vector of KnobSweepAxis describing the knobs to sweep
 * @param[out] result Vector of knob setting combinations (each combination is
 *             a vector of KnobSetting)
 * @return Error with ErrorCode::OK on success, or an error if the product
 *         exceeds the configurable limit (10,000 combinations)
 */
inline Error computeCartesianProduct(const std::vector<KnobSweepAxis>& axes,
                                     std::vector<std::vector<KnobSetting>>& result)
{
    result.clear();

    if(axes.empty())
    {
        result.emplace_back(); // One empty combination (Cartesian product identity)
        return {ErrorCode::OK, ""};
    }

    // Any axis with no values produces an empty product
    for(const auto& axis : axes)
    {
        if(axis.values.empty())
        {
            return {ErrorCode::OK, ""};
        }
    }

    // Compute total number of combinations
    size_t totalCombinations = 1;
    for(const auto& axis : axes)
    {
        // Guard against overflow before multiplying
        if(totalCombinations > CARTESIAN_PRODUCT_LIMIT_ERROR / axis.values.size())
        {
            return {ErrorCode::INVALID_VALUE,
                    "Cartesian product exceeds limit of "
                        + std::to_string(CARTESIAN_PRODUCT_LIMIT_ERROR) + " combinations"};
        }
        totalCombinations *= axis.values.size();
    }

    if(totalCombinations > CARTESIAN_PRODUCT_LIMIT_ERROR)
    {
        return {ErrorCode::INVALID_VALUE,
                "Cartesian product of " + std::to_string(totalCombinations)
                    + " combinations exceeds limit of "
                    + std::to_string(CARTESIAN_PRODUCT_LIMIT_ERROR)};
    }

    if(totalCombinations > CARTESIAN_PRODUCT_LIMIT_WARNING)
    {
        HIPDNN_FE_LOG_WARN("Cartesian product produces " + std::to_string(totalCombinations)
                           + " combinations (warning threshold: "
                           + std::to_string(CARTESIAN_PRODUCT_LIMIT_WARNING) + ")");
    }

    result.reserve(totalCombinations);

    // Iterative Cartesian product generation starting with one empty combination
    result.emplace_back();

    for(const auto& axis : axes)
    {
        std::vector<std::vector<KnobSetting>> expanded;
        expanded.reserve(result.size() * axis.values.size());

        for(const auto& existing : result)
        {
            for(const auto& value : axis.values)
            {
                auto combo = existing;
                combo.emplace_back(axis.knobId, value);
                expanded.push_back(std::move(combo));
            }
        }

        result = std::move(expanded);
    }

    return {ErrorCode::OK, ""};
}

} // namespace autotune
} // namespace hipdnn_frontend
