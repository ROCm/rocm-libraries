// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphContentKey.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_frontend/EngineQueryTypes.hpp>

#include <nlohmann/json.hpp>

#include <cstddef>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>

/// @file CandidatePageJson.hpp
/// @brief How the bench names a graph, and the JSON a catalog enumeration or sweep prints.
namespace hipdnn_bench
{

/// @brief The content key of a serialized graph (or of the graph inside a graph-and-plan
/// container): `GraphContentKey::hash()` in decimal, reported as `graph_key`.
///
/// The key ignores the graph's name and id and numbers tensors by position, so two graphs
/// that compute the same thing share it. It is what a measurement is stored, deduplicated
/// and held out under. hipdnn_corpus_gen's `graphContentKey` computes the same value, so a
/// corpus manifest's `benchmark` is the key the bench reports for that file.
inline std::string graphContentKey(const uint8_t* data, size_t size)
{
    const auto graph
        = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper::fromSerializedBlob(data,
                                                                                         size);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphContentKey key{graph};
    if(!key.isUsable())
    {
        throw std::runtime_error("graph has no verifiable serialized content to key");
    }
    return std::to_string(key.hash());
}

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

} // namespace hipdnn_bench
