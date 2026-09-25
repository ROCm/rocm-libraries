// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <string>
#include <string_view>

#include <hipdnn_data_sdk/utilities/RankingMetrics.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/EngineConfigWrapper.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>

namespace hipdnn_plugin_sdk::heuristics
{

/// @brief The registered ranking metric named @p name, or the default for an empty name.
/// @throws HipdnnPluginException BAD_PARAM for a name the registry does not know: a
///         request in an unknown metric has no direction to rank by, and answering it in
///         another metric would invent a number no model predicted (RFC 0019 §4.4).
inline const hipdnn_data_sdk::utilities::RankingMetric& resolveRankingMetric(std::string_view name)
{
    const auto effective = name.empty() ? hipdnn_data_sdk::utilities::DEFAULT_RANKING_METRIC : name;
    const auto* metric = hipdnn_data_sdk::utilities::findRankingMetric(effective);
    if(metric == nullptr)
    {
        throw HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                    "unregistered ranking metric '" + std::string(effective) + "'");
    }
    return *metric;
}

/// @brief The ranking metric an engine configuration carries (RFC 0019 §11.4).
///
/// The metric travels in `EngineConfig.ranking_metric`, so the same field names the
/// metric a prediction is asked in and the one plan build ranks its catalog by. An
/// invalid (empty) configuration, or one that names no metric, means the default.
inline const hipdnn_data_sdk::utilities::RankingMetric&
    rankingMetric(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& config)
{
    if(!config.isValid())
    {
        return resolveRankingMetric({});
    }
    const auto* name = config.getEngineConfig().ranking_metric();
    return resolveRankingMetric(name == nullptr ? std::string_view{} : name->string_view());
}

} // namespace hipdnn_plugin_sdk::heuristics
