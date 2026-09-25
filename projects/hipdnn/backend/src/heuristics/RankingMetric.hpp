// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "HipdnnException.hpp"

#include <hipdnn_data_sdk/utilities/RankingMetrics.hpp>

#include <string>
#include <string_view>

namespace hipdnn_backend::heuristics
{

/// @brief The registered ranking metric named @p name, or the default for an empty name.
/// @throws HipdnnException HIPDNN_STATUS_BAD_PARAM for a name the registry does not know.
///         RFC 0019 §4.4 refuses an unknown metric where the request is made: it has no
///         direction to rank by, and answering in another metric would invent a number.
inline const hipdnn_data_sdk::utilities::RankingMetric& resolveRankingMetric(std::string_view name)
{
    const auto effective = name.empty() ? hipdnn_data_sdk::utilities::DEFAULT_RANKING_METRIC : name;
    const auto* metric = hipdnn_data_sdk::utilities::findRankingMetric(effective);
    THROW_IF_NULL(metric,
                  HIPDNN_STATUS_BAD_PARAM,
                  "Unregistered ranking metric '" + std::string(effective) + "'");
    return *metric;
}

} // namespace hipdnn_backend::heuristics
