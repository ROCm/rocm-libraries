// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "GraphLogger.hpp"
#include "Logging.hpp"

#include <atomic>
#include <flatbuffers/flatbuffers.h>
#include <fstream>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_data_sdk/utilities/StringUtil.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/utilities/Uuid.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/json/Graph.hpp>
#include <mutex>
#include <nlohmann/json.hpp>

namespace hipdnn_backend::logging
{

namespace
{
std::mutex cacheMutex;
std::filesystem::path cachedPath;
std::atomic<bool> cacheInitialized{false};
} // namespace

bool GraphLogger::isEnabled()
{
    return !getOutputDirectory().empty();
}

void GraphLogger::resetCache()
{
    cacheInitialized.store(false, std::memory_order_release);
}

std::filesystem::path GraphLogger::getOutputDirectory()
{
    // Double-checked locking: the first check is a lock-free fast path so that
    // subsequent calls avoid the mutex entirely. memory_order_acquire ensures
    // that if we read true, all prior writes to cachedPath are visible.
    if(cacheInitialized.load(std::memory_order_acquire))
    {
        return cachedPath;
    }

    // Slow path: another thread may have initialized while we waited for the
    // lock, so check again.
    const std::lock_guard<std::mutex> lock(cacheMutex);
    if(cacheInitialized.load(std::memory_order_acquire))
    {
        return cachedPath;
    }

    const std::string dirPath = hipdnn_data_sdk::utilities::trim(
        hipdnn_data_sdk::utilities::getEnv("HIPDNN_LOG_GRAPH_DIR", ""));

    if(dirPath.empty())
    {
        cachedPath.clear();
        cacheInitialized.store(true, std::memory_order_release);
        return cachedPath;
    }

    cachedPath = std::filesystem::path(dirPath);

    if(cachedPath.is_relative())
    {
        cachedPath = std::filesystem::current_path() / cachedPath;
    }

    try
    {
        std::filesystem::create_directories(cachedPath);
    }
    catch(const std::filesystem::filesystem_error& e)
    {
        HIPDNN_BACKEND_LOG_WARN(
            "Failed to create graph output directory {}: {}", cachedPath.string(), e.what());
        cachedPath.clear();
    }

    // memory_order_release pairs with the acquire above: any thread that
    // reads cacheInitialized as true is guaranteed to see the final cachedPath.
    cacheInitialized.store(true, std::memory_order_release);
    return cachedPath;
}

void GraphLogger::logGraph(const uint8_t* serializedGraph, size_t size)
{
    flatbuffers::Verifier verifier(serializedGraph, size);
    if(!hipdnn_flatbuffers_sdk::data_objects::VerifyGraphBuffer(verifier))
    {
        HIPDNN_BACKEND_LOG_WARN("Not logging a graph whose buffer failed verification");
        return;
    }

    const auto* graph
        = flatbuffers::GetRoot<hipdnn_flatbuffers_sdk::data_objects::Graph>(serializedGraph);

    // Dumps are named by the graph's ID, so re-finalizing a descriptor or replaying a serialized
    // graph reuses one file, while a graph rebuilt from scratch is a distinct graph object and
    // gets its own.
    if(graph->id() == nullptr)
    {
        HIPDNN_BACKEND_LOG_WARN("Not logging a graph without an identity; only a finalized graph "
                                "has one");
        return;
    }

    const auto graphId = hipdnn_flatbuffers_sdk::utilities::formatUuid(
        hipdnn_flatbuffers_sdk::utilities::toUuidBytes(*graph->id()));
    const auto graphName = graph->name() != nullptr ? graph->name()->str() : std::string();

    const auto outputDirectory = getOutputDirectory();
    if(outputDirectory.empty())
    {
        HIPDNN_BACKEND_LOG_WARN("Graph logging is enabled but no valid output directory is set");
        return;
    }

    const auto fullPath = outputDirectory / ("graph_" + graphId + ".json");
    if(std::filesystem::exists(fullPath))
    {
        HIPDNN_BACKEND_LOG_INFO("Skipping graph \"{}\" with id {}, already logged to {}",
                                graphName,
                                graphId,
                                fullPath.string());
        return;
    }

    const nlohmann::json graphJson = *graph;

    std::ofstream file(fullPath);
    if(file.is_open())
    {
        file << graphJson.dump(2);
        file.close();
        HIPDNN_BACKEND_LOG_INFO(
            "Writing graph \"{}\" with id {} to {}", graphName, graphId, fullPath.string());
    }
    else
    {
        HIPDNN_BACKEND_LOG_WARN("Failed to open graph log file {} for graph \"{}\" with id {}",
                                fullPath.string(),
                                graphName,
                                graphId);
    }
}

} // namespace hipdnn_backend::logging
