// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <filesystem>
#include <fstream>

#include <nlohmann/json.hpp>

// Pre-load classification for a bundle. loadGraphAndTensors() throws
// std::exception for two very different reasons — a malformed graph .json
// (authoring error) and absent .bin tensor data (DVC not pulled) — and the
// harness must react differently: FAIL on the former, SKIP on the latter
// (ALMIOPEN-1968: "Unparseable .json: fail"). These free predicates let the
// harness disambiguate before calling the loader, and are unit-tested directly
// without instantiating the harness.
namespace hipdnn_integration_tests::golden
{

// True if the bundle's graph .json is syntactically valid JSON. A parse failure
// means a malformed bundle (authoring error), not missing data.
inline bool graphJsonParses(const std::filesystem::path& jsonPath)
{
    std::ifstream stream(jsonPath);
    if(!stream)
    {
        return false;
    }
    return !nlohmann::json::parse(stream, nullptr, /*allow_exceptions=*/false).is_discarded();
}

// True if every tensor's companion .bin file exists on disk. The loader derives
// each blob path as "{stem}.tensor{uid}.bin"; absence means the tensor data was
// not fetched (DVC), which is a SKIP rather than a FAIL.
inline bool tensorDataPresent(const std::filesystem::path& jsonPath)
{
    std::ifstream stream(jsonPath);
    const auto graph = nlohmann::json::parse(stream, nullptr, /*allow_exceptions=*/false);
    if(graph.is_discarded() || !graph.contains("tensors"))
    {
        return false;
    }

    auto basePath = jsonPath;
    basePath.replace_extension();
    for(const auto& tensor : graph.at("tensors"))
    {
        if(!tensor.contains("uid"))
        {
            return false;
        }
        const auto uid = tensor.at("uid").get<int64_t>();
        const auto binPath
            = std::filesystem::path(basePath.string() + ".tensor" + std::to_string(uid) + ".bin");
        if(!std::filesystem::exists(binPath))
        {
            return false;
        }
    }
    return true;
}

} // namespace hipdnn_integration_tests::golden
