// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "GraphLogger.hpp"
#include "Logging.hpp"

#include <flatbuffers/flatbuffers.h>
#include <fstream>
#include <hipdnn_data_sdk/data_objects/graph_generated.h>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_data_sdk/utilities/StringUtil.hpp>
#include <hipdnn_data_sdk/utilities/json/Graph.hpp>
#include <nlohmann/json.hpp>

#include <spdlog/fmt/fmt.h>

namespace hipdnn_backend::logging
{

std::filesystem::path GraphLogger::getOutputDirectory()
{
    std::string logFilePath = hipdnn_data_sdk::utilities::trim(
        hipdnn_data_sdk::utilities::getEnv("HIPDNN_LOG_FILE", ""));

    if(!logFilePath.empty())
    {
        auto parentPath = std::filesystem::path(logFilePath).parent_path();
        if(!parentPath.empty())
        {
            return parentPath;
        }
    }

    return std::filesystem::current_path();
}

void GraphLogger::logGraph(const uint8_t* serializedGraph, size_t size)
{
    auto hash = hipdnn_data_sdk::utilities::fnv1aHash(serializedGraph, size);
    auto filename = fmt::format("graph_{:016x}.json", hash);
    auto fullPath = getOutputDirectory() / filename;

    if(std::filesystem::exists(fullPath))
    {
        return;
    }

    auto* graph = flatbuffers::GetRoot<hipdnn_data_sdk::data_objects::Graph>(serializedGraph);
    nlohmann::json j = *graph;

    std::ofstream file(fullPath);
    if(file.is_open())
    {
        file << j.dump(2);
        file.close();
        HIPDNN_BACKEND_LOG_INFO("Graph logged to {}", fullPath.string());
    }
    else
    {
        HIPDNN_BACKEND_LOG_WARN("Failed to open graph log file: {}", fullPath.string());
    }
}

} // namespace hipdnn_backend::logging
