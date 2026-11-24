// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_data_sdk/utilities/PolicyNames.hpp>
#include <hipdnn_data_sdk/utilities/VersionUtils.hpp>
#include <hipdnn_frontend/Logging.hpp>
#include <hipdnn_frontend/Types.hpp>
#include <hipdnn_frontend/detail/ScopedHipdnnBackendDescriptor.hpp>

#include <array>
#include <cstdint>
#include <string_view>

namespace hipdnn_frontend::detail
{

/// First backend version that accepts HIPDNN_HEUR_MODE_A and HIPDNN_HEUR_MODE_B. Older backends
/// of the same major version accept only HIPDNN_HEUR_MODE_FALLBACK.
inline hipdnn_data_sdk::utilities::Version heuristicModesBackendVersion()
{
    return {0, 5, 0};
}

/// Sets @p mode on an engine heuristic descriptor. A backend older than
/// heuristicModesBackendVersion() gets the encoding it understands (RFC 0005 §4.6.3): mode
/// FALLBACK, and for A or B the policy order [Config, ModeA or ModeB, StaticOrdering] on
/// HIPDNN_ATTR_ENGINEHEUR_POLICY_ORDER_EXT. A backend without that prediction policy skips it, so
/// StaticOrdering decides.
inline hipdnnStatus_t setEngineHeuristicMode(hipdnnBackendDescriptor_t heuristicDesc,
                                             hipdnnBackendHeurMode_t mode)
{
    if(mode == HIPDNN_HEUR_MODE_FALLBACK
       || hipdnnBackend()->version() >= heuristicModesBackendVersion())
    {
        return hipdnnBackend()->backendSetAttribute(
            heuristicDesc, HIPDNN_ATTR_ENGINEHEUR_MODE, HIPDNN_TYPE_HEUR_MODE, 1, &mode);
    }

    HIPDNN_FE_LOG_INFO("Backend " << hipdnnBackend()->versionString()
                                  << " predates heuristic modes A and B; sending mode FALLBACK"
                                     " with the policy order of mode "
                                  << (mode == HIPDNN_HEUR_MODE_B ? "B" : "A"));
    const hipdnnBackendHeurMode_t fallback = HIPDNN_HEUR_MODE_FALLBACK;
    const hipdnnStatus_t status = hipdnnBackend()->backendSetAttribute(
        heuristicDesc, HIPDNN_ATTR_ENGINEHEUR_MODE, HIPDNN_TYPE_HEUR_MODE, 1, &fallback);
    if(status != HIPDNN_STATUS_SUCCESS)
    {
        return status;
    }

    using hipdnn_data_sdk::utilities::policyNameToId;
    const std::array<int64_t, 3> policyOrder = {
        policyNameToId("SelectionHeuristic::Config"),
        policyNameToId(mode == HIPDNN_HEUR_MODE_B ? hipdnn_data_sdk::utilities::MODE_B_POLICY_NAME
                                                  : hipdnn_data_sdk::utilities::MODE_A_POLICY_NAME),
        policyNameToId("SelectionHeuristic::StaticOrdering")};
    return hipdnnBackend()->backendSetAttribute(heuristicDesc,
                                                HIPDNN_ATTR_ENGINEHEUR_POLICY_ORDER_EXT,
                                                HIPDNN_TYPE_INT64,
                                                static_cast<int64_t>(policyOrder.size()),
                                                policyOrder.data());
}

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

/// Creates and finalizes a heuristic descriptor for one heuristic mode, which the backend turns
/// into its policy list (RFC 0007 §5.3), as CUDNN_ATTR_ENGINEHEUR_MODE does in cuDNN.
/// @p rankingMetric names the registered metric the prediction policies rank by and the
/// result configurations carry into plan build (RFC 0019 §11.4). Empty leaves the attribute
/// unset, so HIPDNN_HEUR_RANKING_METRIC and the backend default keep deciding.
inline Error
    createEngineHeuristicDescriptorForGraph(ScopedHipdnnBackendDescriptor& engineHeuristicDesc,
                                            hipdnnBackendDescriptor_t graphDesc,
                                            HeuristicMode mode,
                                            bool findFirst = false,
                                            std::string_view rankingMetric = {})
{
    engineHeuristicDesc = ScopedHipdnnBackendDescriptor(HIPDNN_BACKEND_ENGINEHEUR_DESCRIPTOR);

    HIPDNN_RETURN_ON_BACKEND_FAILURE(
        hipdnnBackend()->backendSetAttribute(engineHeuristicDesc.get(),
                                             HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH,
                                             HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                             1,
                                             static_cast<const void*>(&graphDesc)),
        "Failed to set operation graph on the engine heuristic descriptor.");

    HIPDNN_RETURN_ON_BACKEND_FAILURE(
        setEngineHeuristicMode(engineHeuristicDesc.get(), toBackendType(mode)),
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

    if(!rankingMetric.empty())
    {
        HIPDNN_RETURN_ON_BACKEND_FAILURE(
            hipdnnBackend()->backendSetAttribute(engineHeuristicDesc.get(),
                                                 HIPDNN_ATTR_ENGINEHEUR_RANKING_METRIC_EXT,
                                                 HIPDNN_TYPE_CHAR,
                                                 static_cast<int64_t>(rankingMetric.size()),
                                                 rankingMetric.data()),
            "Failed to set ranking metric on the engine heuristic descriptor.");
    }

    HIPDNN_RETURN_ON_BACKEND_FAILURE(hipdnnBackend()->backendFinalize(engineHeuristicDesc.get()),
                                     "Failed to finalize engine heuristic descriptor");

    return {ErrorCode::OK, ""};
}

} // namespace hipdnn_frontend::detail
