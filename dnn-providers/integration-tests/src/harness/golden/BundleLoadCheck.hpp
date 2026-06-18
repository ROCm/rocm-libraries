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
// (ALMIOPEN-1968: "Unparseable .json: fail"). checkBundlePreload() classifies a
// bundle before the loader runs, parsing the graph .json exactly once, and is
// unit-tested directly without instantiating the harness.
namespace hipdnn_integration_tests::golden
{

// Result of inspecting a bundle on disk before loading it.
//   graphJsonParses    — the graph .json is syntactically valid JSON. False
//                        means a malformed bundle (authoring error) -> FAIL.
//   tensorDataPresent  — every tensor's companion .bin file exists. False means
//                        the tensor data was not fetched (DVC) -> SKIP. Only
//                        meaningful when graphJsonParses is true.
struct BundlePreloadCheck
{
    bool graphJsonParses = false;
    bool tensorDataPresent = false;
};

// Parse the graph .json once and report both the parse result and whether all
// referenced .bin blobs are present. The loader derives each blob path as
// "{stem}.tensor{uid}.bin"; tensorDataPresent is true only if every one exists.
inline BundlePreloadCheck checkBundlePreload(const std::filesystem::path& jsonPath)
{
    BundlePreloadCheck result;

    std::ifstream stream(jsonPath);
    if(!stream)
    {
        return result;
    }

    const auto graph = nlohmann::json::parse(stream, nullptr, /*allow_exceptions=*/false);
    if(graph.is_discarded())
    {
        return result;
    }
    result.graphJsonParses = true;

    if(!graph.contains("tensors"))
    {
        return result;
    }

    auto basePath = jsonPath;
    basePath.replace_extension();
    for(const auto& tensor : graph.at("tensors"))
    {
        if(!tensor.contains("uid"))
        {
            return result;
        }
        const auto uid = tensor.at("uid").get<int64_t>();
        const auto binPath
            = std::filesystem::path(basePath.string() + ".tensor" + std::to_string(uid) + ".bin");
        if(!std::filesystem::exists(binPath))
        {
            return result;
        }
    }
    result.tensorDataPresent = true;
    return result;
}

} // namespace hipdnn_integration_tests::golden
