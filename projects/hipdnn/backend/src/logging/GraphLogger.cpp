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
#include <hipdnn_flatbuffers_sdk/utilities/json/Graph.hpp>
#include <iomanip>
#include <mutex>
#include <nlohmann/json.hpp>
#include <sstream>

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
    auto hash = hipdnn_data_sdk::utilities::fnv1aHash(serializedGraph, size);

    std::ostringstream oss;
    oss << "graph_" << std::hex << std::setfill('0') << std::setw(16) << hash << ".json";
    auto fullPath = getOutputDirectory() / oss.str();

    if(fullPath.empty())
    {
        HIPDNN_BACKEND_LOG_WARN("Graph logging is enabled but no valid output directory is set");
        return;
    }

    if(std::filesystem::exists(fullPath))
    {
        HIPDNN_BACKEND_LOG_INFO("Skipping duplicate graph logged to {}", fullPath.string());
        return;
    }

    auto* graph
        = flatbuffers::GetRoot<hipdnn_flatbuffers_sdk::data_objects::Graph>(serializedGraph);
    const nlohmann::json j = *graph;

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
