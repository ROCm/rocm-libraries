// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_data_sdk/utilities/PolicyNames.hpp>
#include <hipdnn_frontend/detail/ScopedHipdnnBackendDescriptor.hpp>

#include <algorithm>
#include <cstdint>
#include <vector>

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

    // Only the first mode in the vector is forwarded to the backend today.
    // HIPDNN_HEUR_MODE_FALLBACK is the only backend mode, so this collapses to
    // one element; if the backend ever gains a second mode, this loop should set
    // all of them rather than just backendModes.data()[0].
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

    // HeuristicMode::A/B name prediction POLICIES, not backend modes. RFC 0007
    // §5.3.2 makes the ordered policy list the only channel for policy selection,
    // so hash the requested policy names into HIPDNN_ATTR_ENGINEHEUR_POLICY_ORDER_EXT
    // here instead of asking the backend to infer a chain from the mode. Config
    // stays first (HIPDNN_HEUR_CONFIG_PATH rules keep winning) and StaticOrdering
    // stays last (the fallback that always succeeds). When no prediction mode is
    // requested the attribute is left unset so HIPDNN_HEUR_POLICY_ORDER and the
    // backend's built-in default keep their precedence (RFC 0007 §5.3.3).
    std::vector<int64_t> policyOrder;
    for(const auto& mode : modes)
    {
        const char* policyName = nullptr;
        if(mode == HeuristicMode::A)
        {
            policyName = hipdnn_data_sdk::utilities::MODE_A_POLICY_NAME;
        }
        else if(mode == HeuristicMode::B)
        {
            policyName = hipdnn_data_sdk::utilities::MODE_B_POLICY_NAME;
        }
        if(policyName == nullptr)
        {
            continue;
        }
        const int64_t policyId = hipdnn_data_sdk::utilities::policyNameToId(policyName);
        if(std::find(policyOrder.begin(), policyOrder.end(), policyId) == policyOrder.end())
        {
            policyOrder.push_back(policyId);
        }
    }

    if(!policyOrder.empty())
    {
        policyOrder.insert(
            policyOrder.begin(),
            hipdnn_data_sdk::utilities::policyNameToId("SelectionHeuristic::Config"));
        policyOrder.push_back(
            hipdnn_data_sdk::utilities::policyNameToId("SelectionHeuristic::StaticOrdering"));

        HIPDNN_RETURN_ON_BACKEND_FAILURE(
            hipdnnBackend()->backendSetAttribute(engineHeuristicDesc.get(),
                                                 HIPDNN_ATTR_ENGINEHEUR_POLICY_ORDER_EXT,
                                                 HIPDNN_TYPE_INT64,
                                                 static_cast<int64_t>(policyOrder.size()),
                                                 policyOrder.data()),
            "Failed to set policy order on the engine heuristic descriptor.");
    }

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
