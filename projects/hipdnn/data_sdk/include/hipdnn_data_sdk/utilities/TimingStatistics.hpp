// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file TimingStatistics.hpp
 * @brief Summary statistics for benchmark timing samples
 *
 * Reduces a set of timing samples to one comparable number. Shared by the frontend's
 * autotune sweep and the plugin SDK's kernel-catalog benchmark, which both rank candidates
 * by measured cost.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace hipdnn_data_sdk::utilities::detail
{

/// Arithmetic mean of @p values.
///
/// The result is clamped to the observed [min, max] range. Floating-point summation and
/// division can round the mean a few ULP outside that range when the samples are
/// near-identical, which would otherwise violate the min <= mean <= max invariant that
/// callers rely on (a mean dipping below the minimum by one ULP is a real, observed flake).
/// The mean of a set is mathematically within its range, so clamping only corrects rounding
/// noise and is a no-op for well-separated data.
///
/// @throws std::invalid_argument if @p values is empty.
template <typename T, std::enable_if_t<std::is_floating_point_v<T>, int> = 0>
T mean(const std::vector<T>& values)
{
    if(values.empty())
    {
        throw std::invalid_argument("mean: input vector must not be empty");
    }
    const auto [minIt, maxIt] = std::minmax_element(values.begin(), values.end());
    const auto sum = std::accumulate(values.begin(), values.end(), T{0});
    const auto result = sum / static_cast<T>(values.size());
    return std::clamp(result, *minIt, *maxIt);
}

/// Median of an already-sorted sample vector; even counts average the two middle elements.
///
/// Sorting is the caller's job because every caller here already needs the sorted data.
///
/// @throws std::invalid_argument if @p sorted is empty.
template <typename T, std::enable_if_t<std::is_floating_point_v<T>, int> = 0>
T medianOfSorted(const std::vector<T>& sorted)
{
    if(sorted.empty())
    {
        throw std::invalid_argument("medianOfSorted: input vector must not be empty");
    }
    const size_t mid = sorted.size() / 2;
    if(sorted.size() % 2 == 0)
    {
        return (sorted[mid - 1] + sorted[mid]) / T{2};
    }
    return sorted[mid];
}

/// Population standard deviation of @p values (divides by N, not N-1).
///
/// Population rather than sample stddev because the timing samples are the complete set of
/// measurements taken, not a sample drawn from a larger population.
///
/// @throws std::invalid_argument if @p values is empty.
template <typename T, std::enable_if_t<std::is_floating_point_v<T>, int> = 0>
T stddev(const std::vector<T>& values)
{
    if(values.empty())
    {
        throw std::invalid_argument("stddev: input vector must not be empty");
    }
    const auto m = mean(values);
    auto variance = T{0};
    for(const auto& v : values)
    {
        const auto diff = v - m;
        variance += diff * diff;
    }
    variance /= static_cast<T>(values.size());
    return std::sqrt(variance);
}

/// Coefficient of variation of @p values: stddev / mean, or 0 when the mean is 0.
///
/// Expresses variability as a fraction of the mean, so it is comparable across candidates
/// of different absolute speeds.
///
/// The zero check is exact by intent. If every sample is exactly 0 (a sub-microsecond
/// kernel on a coarse timer) the ratio is undefined and 0 is returned. An epsilon check
/// would instead mask legitimate near-zero means where the ratio is meaningful, such as
/// mean=1e-7 with stddev=1e-8.
///
/// @throws std::invalid_argument if @p values is empty.
template <typename T, std::enable_if_t<std::is_floating_point_v<T>, int> = 0>
T coefficientOfVariation(const std::vector<T>& values)
{
    if(values.empty())
    {
        throw std::invalid_argument("coefficientOfVariation: input vector must not be empty");
    }
    const auto m = mean(values);
    if(m == T{0})
    {
        return T{0};
    }
    return stddev(values) / m;
}

/// Default modified Z-score above which a timing sample is treated as an outlier.
///
/// Matches the threshold MIOpen's tuning loop uses for the same purpose
/// (`miopen::removeHighOutliersAndGetMean`). Lower than the conventional 3.5, because GPU
/// timing samples are contaminated more often than a typical measurement.
inline constexpr double DEFAULT_OUTLIER_Z_THRESHOLD = 2.0;

/// Representative cost of a candidate: the mean of @p values after discarding the slow tail.
///
/// Answers what a candidate usually costs, which is what a caller experiences over many
/// executions. Deliberately not the fastest sample: a candidate that is usually slow but
/// occasionally lucky has the better best-case, so ranking on the fastest sample picks the
/// volatile candidate and then serves its typical, slower time on every execution after.
///
/// The trim is one-sided because the noise is. Interference only ever makes an iteration
/// slower, so the slow tail is contamination that says nothing about the candidate, while an
/// unusually fast sample is signal and is kept.
///
/// Outliers are scored by modified Z-score, `0.6745 * (x - median) / MAD`, where MAD is the
/// median absolute deviation. Median and MAD rather than mean and standard deviation,
/// because the latter are themselves distorted by the outliers being detected.
///
/// @param values Timing samples; order does not matter.
/// @param zThreshold Modified Z-score above which a sample is discarded. A non-positive
///        threshold would discard the median itself, so values below the smallest usable
///        threshold are treated as "keep everything".
/// @throws std::invalid_argument if @p values is empty.
template <typename T, std::enable_if_t<std::is_floating_point_v<T>, int> = 0>
T robustMean(const std::vector<T>& values,
             T zThreshold = static_cast<T>(DEFAULT_OUTLIER_Z_THRESHOLD))
{
    if(values.empty())
    {
        throw std::invalid_argument("robustMean: input vector must not be empty");
    }

    std::vector<T> sorted(values);
    std::sort(sorted.begin(), sorted.end());

    if(zThreshold <= T{0})
    {
        return mean(sorted);
    }

    const T median = medianOfSorted(sorted);

    std::vector<T> deviations;
    deviations.reserve(sorted.size());
    for(const auto& v : sorted)
    {
        deviations.push_back(std::abs(v - median));
    }
    std::sort(deviations.begin(), deviations.end());
    const T mad = medianOfSorted(deviations);

    // A zero MAD means at least half the samples equal the median, so there is no measurable
    // spread to score against. Every sample would divide by zero; keep them all instead.
    // Reachable in practice: a short kernel on a coarse timer quantises to one value.
    if(mad == T{0})
    {
        return mean(sorted);
    }

    // 0.6745 is the 0.75 quantile of the standard normal. It scales the MAD so the score is
    // on the same footing as a standard deviation for normally distributed samples.
    constexpr T SCALE = static_cast<T>(0.6745);

    std::vector<T> kept;
    kept.reserve(sorted.size());
    for(const auto& v : sorted)
    {
        if(SCALE * (v - median) / mad <= zThreshold)
        {
            kept.push_back(v);
        }
    }

    // The median scores exactly 0, so a positive threshold always keeps at least one sample.
    return mean(kept);
}

} // namespace hipdnn_data_sdk::utilities::detail
