// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/Types.hpp>
#include <hipdnn_frontend/Utilities.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>
#include <hipdnn_frontend/detail/BackendWrapper.hpp>
#include <hipdnn_frontend/detail/CreateBackendDescriptor.hpp>
#include <hipdnn_frontend/detail/ScopedHipdnnBackendDescriptor.hpp>
#include <hipdnn_frontend/detail/VariantPackHelpers.hpp>

#include <hipdnn_data_sdk/utilities/EngineNames.hpp>

#include <cmath>
#include <cstdint>
#include <memory>
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

/// Resolve a backend engine ID to its name from the static registry alone,
/// falling back to the hexadecimal ID for an engine the registry does not carry.
/// Plugin-supplied names are not visible here; those come from the backend, via
/// hipdnnGetEngineNameById_ext.
inline std::string resolveEngineName(int64_t engineId)
{
    return hipdnn_data_sdk::utilities::engineNameOrHex(engineId);
}

// Execute a graph using a specific execution plan descriptor and a
// pre-built variant pack descriptor. Used by autotune() for warmup and
// timed iterations; the variant pack descriptor is built once by the caller
// and reused so its construction stays out of any timed window.
inline Error executeWithPlan(hipdnnHandle_t handle,
                             const ScopedHipdnnBackendDescriptor& execPlan,
                             const ScopedHipdnnBackendDescriptor& variantPackDesc)
{
    HIPDNN_RETURN_ON_BACKEND_FAILURE(
        hipdnnBackend()->backendExecute(handle, execPlan.get(), variantPackDesc.get()),
        "Execute failed.");

    return {ErrorCode::OK, ""};
}

// Execute once using a caller-owned profiling context. RESET retains its events and
// gate between measurements. Autotune owns one context per comparison; public timed
// execution owns a one-shot context. Neither this helper nor the gate chooses retries.
//
// Failure cleanup releases and drains before the next candidate can run. The Error
// is constructed before cleanup, preserving the original backend diagnostic.
//
// A finite negative elapsed reading is an invalid measurement, not a backend failure: this
// returns ErrorCode::OK with `timing` left at its default INVALID/empty/timedOut=false, and
// does not replay. A non-finite (NaN/Inf) reading is a genuine backend error and returns a
// bad Error, same as any other failed step above.
inline Error executeWithPlanTimed(hipdnnHandle_t handle,
                                  const ScopedHipdnnBackendDescriptor& execPlan,
                                  const ScopedHipdnnBackendDescriptor& variantPackDesc,
                                  const ScopedHipdnnBackendDescriptor& profilingDesc,
                                  ExecutionTiming& timing,
                                  bool stalled = true)
{
    timing = {};
    if(!profilingDesc.valid())
    {
        return {ErrorCode::HIPDNN_BACKEND_ERROR, "Profiling control descriptor is not valid"};
    }

    const bool trigger = true;
    HIPDNN_RETURN_ON_BACKEND_FAILURE(
        hipdnnBackend()->backendSetAttribute(
            profilingDesc.get(), HIPDNN_ATTR_PROFILING_RESET_EXT, HIPDNN_TYPE_BOOLEAN, 1, &trigger),
        "Failed to reset profiling control descriptor");
    HIPDNN_RETURN_ON_BACKEND_FAILURE(
        hipdnnBackend()->backendSetAttribute(profilingDesc.get(),
                                             HIPDNN_ATTR_PROFILING_HANDLE_EXT,
                                             HIPDNN_TYPE_HANDLE,
                                             1,
                                             static_cast<const void*>(&handle)),
        "Failed to set handle on profiling descriptor");

    struct ResetOnFailure
    {
        const ScopedHipdnnBackendDescriptor& descriptor;
        bool finalized = false;
        ~ResetOnFailure()
        {
            if(!finalized)
            {
                const bool reset = true;
                static_cast<void>(
                    hipdnnBackend()->backendSetAttribute(descriptor.get(),
                                                         HIPDNN_ATTR_PROFILING_RESET_EXT,
                                                         HIPDNN_TYPE_BOOLEAN,
                                                         1,
                                                         &reset));
            }
        }
    } guard{profilingDesc};

    if(stalled)
    {
        HIPDNN_RETURN_ON_BACKEND_FAILURE(
            hipdnnBackend()->backendSetAttribute(profilingDesc.get(),
                                                 HIPDNN_ATTR_PROFILING_STALL_ARM_EXT,
                                                 HIPDNN_TYPE_BOOLEAN,
                                                 1,
                                                 &trigger),
            "Failed to arm profiling stall");
    }
    HIPDNN_RETURN_ON_BACKEND_FAILURE(
        hipdnnBackend()->backendSetAttribute(
            profilingDesc.get(), HIPDNN_ATTR_PROFILING_START_EXT, HIPDNN_TYPE_BOOLEAN, 1, &trigger),
        "Failed to set profiling start");

    HIPDNN_CHECK_ERROR(executeWithPlan(handle, execPlan, variantPackDesc));

    HIPDNN_RETURN_ON_BACKEND_FAILURE(
        hipdnnBackend()->backendSetAttribute(
            profilingDesc.get(), HIPDNN_ATTR_PROFILING_STOP_EXT, HIPDNN_TYPE_BOOLEAN, 1, &trigger),
        "Failed to set profiling stop");
    if(stalled)
    {
        HIPDNN_RETURN_ON_BACKEND_FAILURE(
            hipdnnBackend()->backendSetAttribute(profilingDesc.get(),
                                                 HIPDNN_ATTR_PROFILING_STALL_RELEASE_EXT,
                                                 HIPDNN_TYPE_BOOLEAN,
                                                 1,
                                                 &trigger),
            "Failed to release profiling stall");
    }
    HIPDNN_RETURN_ON_BACKEND_FAILURE(hipdnnBackend()->backendFinalize(profilingDesc.get()),
                                     "Failed to finalize profiling descriptor");
    guard.finalized = true;

    float elapsedMs = 0.0f;
    HIPDNN_RETURN_ON_BACKEND_FAILURE(
        hipdnnBackend()->backendGetAttribute(profilingDesc.get(),
                                             HIPDNN_ATTR_PROFILING_ELAPSED_MS_EXT,
                                             HIPDNN_TYPE_FLOAT,
                                             1,
                                             nullptr,
                                             &elapsedMs),
        "Failed to get profiling elapsed ms");
    bool stallUsed = false;
    HIPDNN_RETURN_ON_BACKEND_FAILURE(
        hipdnnBackend()->backendGetAttribute(profilingDesc.get(),
                                             HIPDNN_ATTR_PROFILING_STALL_USED_EXT,
                                             HIPDNN_TYPE_BOOLEAN,
                                             1,
                                             nullptr,
                                             &stallUsed),
        "Failed to get profiling stall used flag");
    bool timedOut = false;
    HIPDNN_RETURN_ON_BACKEND_FAILURE(
        hipdnnBackend()->backendGetAttribute(profilingDesc.get(),
                                             HIPDNN_ATTR_PROFILING_STALL_TIMED_OUT_EXT,
                                             HIPDNN_TYPE_BOOLEAN,
                                             1,
                                             nullptr,
                                             &timedOut),
        "Failed to get profiling stall timed-out flag");
    if(timedOut)
    {
        timing.timedOut = true;
        return {ErrorCode::OK, ""};
    }
    if(!std::isfinite(elapsedMs))
    {
        return {ErrorCode::HIPDNN_BACKEND_ERROR,
                "Backend reported a non-finite profiling elapsed time"};
    }
    if(elapsedMs < 0.0f)
    {
        // A finite negative elapsed time is a bad reading, not a backend failure: leave
        // `timing` at its default INVALID quality / empty elapsedMs / timedOut=false (set
        // above) and report success. The caller decides whether/how to retry; this
        // function never replays the measurement itself.
        return {ErrorCode::OK, ""};
    }
    timing.quality = stallUsed ? TimingQuality::DEVICE_ONLY : TimingQuality::UNSTALLED;
    timing.elapsedMs = elapsedMs;
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
