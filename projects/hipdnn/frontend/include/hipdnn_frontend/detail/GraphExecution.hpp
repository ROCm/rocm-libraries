// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/Utilities.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>
#include <hipdnn_frontend/detail/BackendWrapper.hpp>
#include <hipdnn_frontend/detail/CreateBackendDescriptor.hpp>
#include <hipdnn_frontend/detail/ScopedHipdnnBackendDescriptor.hpp>
#include <hipdnn_frontend/detail/VariantPackHelpers.hpp>

#include <hipdnn_data_sdk/utilities/EngineNames.hpp>

#include <cstdint>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace hipdnn_frontend::detail
{

// Convert a tensor-attribute keyed lookup into a UID-keyed variant pack.
// Each entry's tensor must be non-null and carry a valid uid. Returns
// ErrorCode::OK on success; ErrorCode::INVALID_VALUE if any tensor is null or
// lacks a valid uid (variantPack is left partially filled).
inline Error tensorLookupToVariantPack(
    const std::unordered_map<std::shared_ptr<graph::TensorAttributes>, void*>& tensorLookup,
    std::unordered_map<int64_t, void*>& variantPack)
{
    for(const auto& [tensor, ptr] : tensorLookup)
    {
        if(tensor && tensor->has_uid())
        {
            variantPack[tensor->get_uid()] = ptr;
        }
        else
        {
            return {ErrorCode::INVALID_VALUE,
                    "Tensor in tensor lookup is null or does not have a valid uid."};
        }
    }
    return {ErrorCode::OK, ""};
}

// Resolve a backend engine ID to its human-readable name.
// Falls back to a hex string (e.g., "0x1A2B") for unknown engines.
inline std::string resolveEngineName(int64_t engineId)
{
    try
    {
        return std::string(hipdnn_data_sdk::utilities::getEngineNameFromId(engineId));
    }
    catch(const std::out_of_range&)
    {
        std::ostringstream oss;
        oss << "0x" << std::hex << engineId;
        return oss.str();
    }
}

// Execute a graph using a specific execution plan descriptor and a
// pre-built variant pack descriptor. Used by autotune() for warmup and
// timed iterations; the variant pack descriptor is built once by the caller
// and reused so its construction stays out of any timed window.
inline Error executeWithPlan(hipdnnHandle_t handle,
                             ScopedHipdnnBackendDescriptor& execPlan,
                             ScopedHipdnnBackendDescriptor& variantPackDesc)
{
    HIPDNN_RETURN_ON_BACKEND_FAILURE(
        hipdnnBackend()->backendExecute(handle, execPlan.get(), variantPackDesc.get()),
        "Execute failed.");

    return {ErrorCode::OK, ""};
}

// Query the backend for an engine's workspace size estimate.
// Returns 0 if the query fails at any step (non-fatal — workspace will
// be determined accurately at plan compilation time).
inline int64_t queryEngineWorkspaceSize(hipdnnBackendDescriptor_t graphDesc, int64_t engineId)
{
    detail::ScopedHipdnnBackendDescriptor engineDesc;
    auto createErr
        = hipdnn_frontend::detail::createEngineDescriptorForGraph(engineDesc, graphDesc, engineId);
    if(createErr.is_bad())
    {
        return 0;
    }

    auto engineConfigDesc = std::make_unique<detail::ScopedHipdnnBackendDescriptor>(
        HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR);
    auto setStatus
        = detail::hipdnnBackend()->backendSetAttribute(engineConfigDesc->get(),
                                                       HIPDNN_ATTR_ENGINECFG_ENGINE,
                                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                       1,
                                                       static_cast<const void*>(&engineDesc.get()));
    if(setStatus != HIPDNN_STATUS_SUCCESS)
    {
        return 0;
    }

    auto finStatus = detail::hipdnnBackend()->backendFinalize(engineConfigDesc->get());
    if(finStatus != HIPDNN_STATUS_SUCCESS)
    {
        return 0;
    }

    int64_t wsSize = 0;
    detail::hipdnnBackend()->backendGetAttribute(engineConfigDesc->get(),
                                                 HIPDNN_ATTR_ENGINECFG_WORKSPACE_SIZE,
                                                 HIPDNN_TYPE_INT64,
                                                 1,
                                                 nullptr,
                                                 &wsSize);
    return wsSize;
}

} // namespace hipdnn_frontend::detail
