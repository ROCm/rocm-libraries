// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <array>
#include <cmath>
#include <string_view>

namespace hipdnn_data_sdk::utilities
{

/// Which way a ranking metric improves.
enum class MetricDirection
{
    HIGHER_IS_BETTER,
    LOWER_IS_BETTER,
};

/**
 * @brief One registered ranking metric (RFC 0019 §4.4).
 *
 * The registry is closed: the backend needs each metric's direction to order engines.
 * Adding a metric is a row here, not an ABI change.
 */
struct RankingMetric
{
    std::string_view name;
    std::string_view units;
    MetricDirection direction;
};

/// The metric a request ranks by when it names none.
inline constexpr std::string_view DEFAULT_RANKING_METRIC = "tflops";

inline constexpr std::array<RankingMetric, 2> RANKING_METRICS{{
    {"tflops", "TFLOPS", MetricDirection::HIGHER_IS_BETTER},
    {"time", "ms", MetricDirection::LOWER_IS_BETTER},
}};

/// The registered metric named @p name, or nullptr for a name the registry does not know.
constexpr const RankingMetric* findRankingMetric(std::string_view name) noexcept
{
    for(const auto& metric : RANKING_METRICS)
    {
        if(metric.name == name)
        {
            return &metric;
        }
    }
    return nullptr;
}

/// The UHD `objective` a model of @p metric must declare: `max` or `min`.
constexpr std::string_view objectiveOf(const RankingMetric& metric) noexcept
{
    return metric.direction == MetricDirection::HIGHER_IS_BETTER ? "max" : "min";
}

/// Whether @p value is a physically meaningful value of @p metric: finite and strictly
/// positive. Zero is excluded for every direction because kernel rankings write 0 for
/// "no measurement".
inline bool isValidMetricValue(const RankingMetric& /*metric*/, double value) noexcept
{
    return std::isfinite(value) && value > 0.0;
}

/// Whether @p left is strictly better than @p right in @p metric's direction.
constexpr bool isBetterMetricValue(const RankingMetric& metric, double left, double right) noexcept
{
    return metric.direction == MetricDirection::HIGHER_IS_BETTER ? left > right : left < right;
}

} // namespace hipdnn_data_sdk::utilities
