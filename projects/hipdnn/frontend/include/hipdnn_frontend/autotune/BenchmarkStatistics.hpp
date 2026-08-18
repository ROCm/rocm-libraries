// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file BenchmarkStatistics.hpp
 * @brief Statistics utilities for autotune benchmarking
 *
 * Provides functions for computing mean, standard deviation, and coefficient
 * of variation from timing samples. Used by the RUN_UNTIL_STABLE strategy
 * to determine convergence.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace hipdnn_frontend::autotune::detail
{

// Compute the arithmetic mean of a range of floating-point timing samples.
// Throws std::invalid_argument if values is empty.
//
// The result is clamped to the observed [min, max] range. Floating-point
// summation and division can round the mean a few ULP outside that range when
// the samples are near-identical, which would otherwise violate the
// min <= mean <= max invariant that callers rely on (e.g. the autotune
// min/avg timing stats, where avg dipping below min by a ULP is a real,
// observed flake). The mean of a set is mathematically within its range, so
// clamping only corrects rounding noise and is a no-op for well-separated data.
template <typename T, std::enable_if_t<std::is_floating_point_v<T>, int> = 0>
T computeMean(const std::vector<T>& values)
{
    if(values.empty())
    {
        throw std::invalid_argument("computeMean: input vector must not be empty");
    }
    const auto [minIt, maxIt] = std::minmax_element(values.begin(), values.end());
    const auto sum = std::accumulate(values.begin(), values.end(), T{0});
    const auto mean = sum / static_cast<T>(values.size());
    return std::clamp(mean, *minIt, *maxIt);
}

// Compute the population standard deviation of a range of floating-point
// timing samples. Throws std::invalid_argument if values is empty.
//
// Uses population stddev (divides by N, not N-1) because the benchmarking
// samples represent the complete set of measurements, not a sample drawn
// from a larger population.
template <typename T, std::enable_if_t<std::is_floating_point_v<T>, int> = 0>
T computeStddev(const std::vector<T>& values)
{
    if(values.empty())
    {
        throw std::invalid_argument("computeStddev: input vector must not be empty");
    }
    auto mean = computeMean(values);
    auto variance = T{0};
    for(const auto& v : values)
    {
        auto diff = v - mean;
        variance += diff * diff;
    }
    variance /= static_cast<T>(values.size());
    return std::sqrt(variance);
}

// Compute the coefficient of variation (CoV) of a range of floating-point
// timing samples: stddev / mean, or 0 if mean is 0. Throws
// std::invalid_argument if values is empty.
//
// The coefficient of variation expresses timing variability as a fraction
// of the mean. A CoV below the stabilityThreshold indicates convergence
// in the RUN_UNTIL_STABLE strategy.
template <typename T, std::enable_if_t<std::is_floating_point_v<T>, int> = 0>
T computeCoefficientOfVariation(const std::vector<T>& values)
{
    if(values.empty())
    {
        throw std::invalid_argument(
            "computeCoefficientOfVariation: input vector must not be empty");
    }
    auto mean = computeMean(values);
    // Exact zero check is intentional: if all samples are exactly 0.0
    // (e.g., sub-microsecond kernel on a coarse timer), CoV is undefined.
    // A near-zero check (epsilon) would mask legitimate near-zero means
    // where CoV is meaningful (e.g., mean=1e-7, stddev=1e-8 -> CoV=0.1).
    if(mean == T{0})
    {
        return T{0};
    }
    return computeStddev(values) / mean;
}

} // namespace hipdnn_frontend::autotune::detail
