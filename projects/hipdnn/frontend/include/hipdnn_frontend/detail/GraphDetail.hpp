// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/Utilities.hpp>
#include <hipdnn_frontend/detail/BackendWrapper.hpp>
#include <hipdnn_frontend/detail/ScopedHipdnnBackendDescriptor.hpp>

#include <memory>
#include <vector>

namespace hipdnn_frontend::detail
{

inline Error hasEngineConfigs(hipdnnBackendDescriptor_t engineHeuristicDesc)
{
    int64_t availableEngineCount = 0;
    HIPDNN_RETURN_ON_BACKEND_FAILURE(
        hipdnnBackend()->backendGetAttribute(engineHeuristicDesc,
                                             HIPDNN_ATTR_ENGINEHEUR_RESULTS,
                                             HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                             0,
                                             &availableEngineCount,
                                             nullptr),
        "Failed to get attribute from the engine heuristic descriptor.");

    if(availableEngineCount == 0)
    {
        return {ErrorCode::GRAPH_NOT_SUPPORTED,
                "No engine configurations available for the graph."};
    }

    return {ErrorCode::OK, ""};
}

inline Error
    getEngineConfigs(std::vector<std::unique_ptr<ScopedHipdnnBackendDescriptor>>& engineConfigs,
                     std::vector<int64_t>& engineIds,
                     hipdnnBackendDescriptor_t engineHeuristicDesc,
                     bool getAll)
{
    int64_t availableEngineCount = 0;
    HIPDNN_RETURN_ON_BACKEND_FAILURE(
        hipdnnBackend()->backendGetAttribute(engineHeuristicDesc,
                                             HIPDNN_ATTR_ENGINEHEUR_RESULTS,
                                             HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                             0,
                                             &availableEngineCount,
                                             nullptr),
        "Failed to get attribute from the engine heuristic descriptor.");

    if(availableEngineCount == 0)
    {
        return {ErrorCode::GRAPH_NOT_SUPPORTED,
                "No engine configurations available for the graph."};
    }

    // Fetch only the top engine config unless the caller needs the full ranked
    // list (e.g. get_ranked_engine_ids, or the explicit Graph.preferred_engine_id
    // post-hoc reorder in initializeEngineConfig). HIPDNN_HEUR_CONFIG_PATH
    // reordering happens inside the SelectionHeuristic::Config built-in and is
    // already reflected in the ranked list — no extra frontend search is needed
    // for that knob.
    const int64_t requiredCount = getAll ? availableEngineCount : 1;
    std::vector<hipdnnBackendDescriptor_t> engineConfigsShallow;
    for(size_t i = 0; i < static_cast<size_t>(requiredCount); ++i)
    {
        auto engineCfgDesc
            = std::make_unique<ScopedHipdnnBackendDescriptor>(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR);

        if(engineCfgDesc == nullptr || !engineCfgDesc->valid())
        {
            return {ErrorCode::HIPDNN_BACKEND_ERROR,
                    "Failed to create engine configuration descriptor."};
        }
        engineConfigs.push_back(std::move(engineCfgDesc));
        engineConfigsShallow.push_back(engineConfigs.back()->get());
    }

    int64_t count = 0;
    HIPDNN_RETURN_ON_BACKEND_FAILURE(
        hipdnnBackend()->backendGetAttribute(engineHeuristicDesc,
                                             HIPDNN_ATTR_ENGINEHEUR_RESULTS,
                                             HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                             static_cast<int64_t>(engineConfigsShallow.size()),
                                             &count,
                                             static_cast<void*>(engineConfigsShallow.data())),
        "Failed to get engine configurations from the heuristic descriptor.");

    if(count == 0)
    {
        return {ErrorCode::GRAPH_NOT_SUPPORTED,
                "No engine configurations retrieved from the heuristic desc."};
    }

    // Finalize and get ids for engine configs
    engineIds.resize(static_cast<size_t>(count), -1);
    for(size_t i = 0; i < static_cast<size_t>(count); ++i)
    {
        auto engineConfigDesc = engineConfigsShallow[i];
        HIPDNN_RETURN_ON_BACKEND_FAILURE(hipdnnBackend()->backendFinalize(engineConfigDesc),
                                         "Failed to finalize engine config descriptor");

        hipdnnBackendDescriptor_t engineDesc = nullptr;
        HIPDNN_RETURN_ON_BACKEND_FAILURE(
            hipdnnBackend()->backendGetAttribute(engineConfigDesc,
                                                 HIPDNN_ATTR_ENGINECFG_ENGINE,
                                                 HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                 1,
                                                 nullptr,
                                                 static_cast<void*>(&engineDesc)),
            "Failed to get engine from engine configuration descriptor.");

        // Clean-up engineDesc once we no longer need it within this scope.
        const ScopedHipdnnBackendDescriptor scopedEngineDesc(engineDesc);

        HIPDNN_RETURN_ON_BACKEND_FAILURE(
            hipdnnBackend()->backendGetAttribute(engineDesc,
                                                 HIPDNN_ATTR_ENGINE_GLOBAL_INDEX,
                                                 HIPDNN_TYPE_INT64,
                                                 1,
                                                 nullptr,
                                                 &engineIds[i]),
            "Failed to get engine id from engine descriptor.");
    }

    return {ErrorCode::OK, ""};
}

} // namespace hipdnn_frontend::detail
