// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_frontend/detail/ScopedHipdnnBackendDescriptor.hpp>

namespace hipdnn_frontend::detail
{

inline Error createEngineDescriptorForGraph(ScopedHipdnnBackendDescriptor& engineDesc,
                                            hipdnnBackendDescriptor_t graphDesc,
                                            int64_t engineId)
{
    engineDesc = ScopedHipdnnBackendDescriptor(HIPDNN_BACKEND_ENGINE_DESCRIPTOR);

    HIPDNN_RETURN_ON_BACKEND_FAILURE(
        hipdnnBackend()->backendSetAttribute(engineDesc.get(),
                                             HIPDNN_ATTR_ENGINE_OPERATION_GRAPH,
                                             HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                             1,
                                             static_cast<const void*>(&graphDesc)),
        "Failed to set operation graph on the engine descriptor.");

    HIPDNN_RETURN_ON_BACKEND_FAILURE(
        hipdnnBackend()->backendSetAttribute(
            engineDesc.get(), HIPDNN_ATTR_ENGINE_GLOBAL_INDEX, HIPDNN_TYPE_INT64, 1, &engineId),
        "Failed to set engine id on the engine descriptor.");

    HIPDNN_RETURN_ON_BACKEND_FAILURE(hipdnnBackend()->backendFinalize(engineDesc.get()),
                                     "Failed to finalize engine descriptor");

    return {ErrorCode::OK, ""};
}

inline Error
    createEngineHeuristicDescriptorForGraph(ScopedHipdnnBackendDescriptor& engineHeuristicDesc,
                                            hipdnnBackendDescriptor_t graphDesc,
                                            const std::vector<HeuristicMode>& modes,
                                            bool findFirst = false)
{
    engineHeuristicDesc = ScopedHipdnnBackendDescriptor(HIPDNN_BACKEND_ENGINEHEUR_DESCRIPTOR);

    HIPDNN_RETURN_ON_BACKEND_FAILURE(
        hipdnnBackend()->backendSetAttribute(engineHeuristicDesc.get(),
                                             HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH,
                                             HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                             1,
                                             static_cast<const void*>(&graphDesc)),
        "Failed to set operation graph on the engine heuristic descriptor.");

    // TODO
    // Currently we only handle the first mode in the vector.  Once we add heuristics we will need
    // to handle using all modes that are passed in.  We currently only have 1 mode so there
    // is only 1 possibility.
    std::vector<hipdnnBackendHeurMode_t> backendModes;
    backendModes.reserve(modes.size());
    for(const auto& mode : modes)
    {
        backendModes.push_back(toBackendType(mode));
    }

    HIPDNN_RETURN_ON_BACKEND_FAILURE(
        hipdnnBackend()->backendSetAttribute(engineHeuristicDesc.get(),
                                             HIPDNN_ATTR_ENGINEHEUR_MODE,
                                             HIPDNN_TYPE_HEUR_MODE,
                                             1,
                                             backendModes.data()),
        "Failed to set mode on the engine heuristic descriptor.");

    if(findFirst)
    {
        bool findFirstValue = true;
        HIPDNN_RETURN_ON_BACKEND_FAILURE(
            hipdnnBackend()->backendSetAttribute(engineHeuristicDesc.get(),
                                                 HIPDNN_ATTR_ENGINEHEUR_FIND_FIRST_EXT,
                                                 HIPDNN_TYPE_BOOLEAN,
                                                 1,
                                                 &findFirstValue),
            "Failed to set find first on the engine heuristic descriptor.");
    }

    HIPDNN_RETURN_ON_BACKEND_FAILURE(hipdnnBackend()->backendFinalize(engineHeuristicDesc.get()),
                                     "Failed to finalize engine heuristic descriptor");

    return {ErrorCode::OK, ""};
}

} // namespace hipdnn_frontend::detail
