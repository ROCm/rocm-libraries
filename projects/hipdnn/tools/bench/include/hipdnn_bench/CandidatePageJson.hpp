// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_bench/CsvOutput.hpp>
#include <hipdnn_frontend/EngineQueryTypes.hpp>
#include <hipdnn_tools/GraphContentKey.hpp>

#include <nlohmann/json.hpp>

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <variant>

/// @file CandidatePageJson.hpp
/// @brief How the bench names a graph, and the JSON a catalog enumeration or sweep prints.
namespace hipdnn_bench
{

using hipdnn_tools::graphContentKey;

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

/// @p page as JSON, naming the graph by @p graphKey (@ref graphContentKey) and the id it
/// carries, @p graphId.
inline nlohmann::json pageJson(const hipdnn_frontend::EngineCandidatePage& page,
                               const std::string& graphKey,
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
            {"graph_key", graphKey},
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

/// One `results[]` entry of a `--sweep --json` response: the candidate, its knobs and kernel
/// features, then its outcome. `is_valid` means measured, never numerically correct; an
/// untimed candidate has null timings, since a zero would read as a measurement.
inline nlohmann::json sweepResultJson(const std::string& candidateId,
                                      const nlohmann::json& knobs,
                                      const nlohmann::json& kernelFeatures,
                                      const CandidateOutcome& outcome)
{
    const bool timed = outcome.succeeded && outcome.iterationsRun > 0;
    const auto timing = [timed](float milliseconds) {
        return timed ? nlohmann::json(milliseconds) : nlohmann::json(nullptr);
    };
    return {{"candidate_id", candidateId},
            {"knob_settings", knobs},
            {"kernel_features", kernelFeatures},
            {"rank", outcome.rank},
            {"succeeded", outcome.succeeded},
            {"is_valid", timed},
            {"numerically_valid", outcome.numericallyValid},
            {"validation", outcome.validation},
            {"skip_reason", skipReason(outcome.succeeded, outcome.iterationsRun)},
            {"min_time_ms", timing(outcome.minTimeMs)},
            {"avg_time_ms", timing(outcome.avgTimeMs)},
            {"robust_time_ms", timing(outcome.robustTimeMs)},
            {"stddev_ms", timing(outcome.stddevMs)},
            {"iterations", outcome.iterationsRun},
            {"converged", outcome.converged},
            {"workspace_bytes", outcome.workspaceBytes}};
}

} // namespace hipdnn_bench
