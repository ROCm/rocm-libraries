// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_frontend/EngineQueryTypes.hpp>

#include <nlohmann/json.hpp>

#include <map>
#include <string>
#include <utility>
#include <variant>

/// @file CandidatePageJson.hpp
/// @brief The JSON a catalog enumeration or sweep prints for an engine's candidate page.
namespace hipdnn_bench
{

/// A knob tuple as a JSON object of knob name to value.
inline nlohmann::json
    knobJson(const std::map<hipdnn_frontend::KnobType_t, hipdnn_frontend::KnobValueVariant>& knobs)
{
    nlohmann::json result = nlohmann::json::object();
    for(const auto& entry : knobs)
    {
        std::visit([&result, &entry](const auto& value) { result[entry.first] = value; },
                   entry.second);
    }
    return result;
}

/// @p page as JSON, naming the graph @p graphId: the id the graph itself carries, which a
/// corpus manifest and an engine-immediate run name it by. The page's own `graphId` is the
/// engine's structural hash of the graph, which no corpus records.
inline nlohmann::json pageJson(const hipdnn_frontend::EngineCandidatePage& page,
                               const std::string& graphId)
{
    nlohmann::json candidates = nlohmann::json::array();
    for(const auto& candidate : page.candidates)
    {
        candidates.push_back({{"id", candidate.id},
                              {"knob_settings", knobJson(candidate.variant.knobSettings)},
                              {"kernel_features", candidate.kernelFeatures}});
    }
    return {{"engine_id", page.engineId},
            {"graph_id", graphId},
            {"engine_name", page.engineName},
            {"engine_descriptor_id", page.engineDescriptorId},
            {"device_id", page.deviceId},
            {"device_arch", page.deviceArch},
            {"problem_features", page.problemFeatures},
            {"device_features", page.deviceFeatures},
            {"candidates", std::move(candidates)},
            {"total_count", page.totalCount},
            {"offset", page.offset},
            {"next_offset",
             page.nextOffset ? nlohmann::json(*page.nextOffset) : nlohmann::json(nullptr)}};
}

} // namespace hipdnn_bench
