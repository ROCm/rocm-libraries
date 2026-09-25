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
 * The registry is closed and owned by hipDNN: the backend orders engines by these
 * values, so it must know each metric's direction and units rather than trusting the
 * model that produced the number. Adding a metric is a row here, not an ABI change;
 * requests and predictions carry the metric by name.
 */
struct RankingMetric
{
    std::string_view name;
    std::string_view units;
    MetricDirection direction;
};

/// The metric a request ranks by when it names none. Engine selection before metrics
/// existed ranked by calibrated TFLOPS, so this default changes nothing for such a request.
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

/// Whether @p value is a physically meaningful value of @p metric: finite, and
/// non-negative for a throughput, strictly positive for a time.
inline bool isValidMetricValue(const RankingMetric& metric, double value) noexcept
{
    if(!std::isfinite(value))
    {
        return false;
    }
    return metric.direction == MetricDirection::HIGHER_IS_BETTER ? value >= 0.0 : value > 0.0;
}

/// Whether @p left is strictly better than @p right in @p metric's direction.
constexpr bool isBetterMetricValue(const RankingMetric& metric, double left, double right) noexcept
{
    return metric.direction == MetricDirection::HIGHER_IS_BETTER ? left > right : left < right;
}

} // namespace hipdnn_data_sdk::utilities
