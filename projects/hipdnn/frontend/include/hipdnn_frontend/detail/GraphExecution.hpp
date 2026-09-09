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

// Run one timed execution using a fresh profiling control descriptor: create
// descriptor, set handle, optionally arm the stall gate, record START, execute
// exactly once, record STOP, release any armed stall, finalize, then read back
// elapsed time plus the stall-used / timed-out flags to classify the
// measurement. Shared by Graph::execute_timed_ext() (stalled=true, no retry)
// and autotune's benchmarkOnce() (stalled toggles the unstalled retry).
//
// `timing` is reset (empty elapsedMs, INVALID quality) before any validation and
// is only populated once every step below has succeeded. A bad Error always
// leaves `timing` at that reset state.
//
// A watchdog release means the executed plan blocked the host on its own stream
// while the stall held it, so the elapsed span contains the timeout instead of a
// measurement: quality is INVALID (elapsedMs stays empty) even though the Error
// is OK, because execution itself completed. STALL_USED_EXT distinguishes an
// active, healthy stall (DEVICE_ONLY) from a declined/skipped one
// (HOST_INCLUDED) when no timeout occurred.
inline Error executeWithPlanTimed(hipdnnHandle_t handle,
                                  const ScopedHipdnnBackendDescriptor& execPlan,
                                  const ScopedHipdnnBackendDescriptor& variantPackDesc,
                                  ExecutionTiming& timing,
                                  bool stalled = true)
{
    timing.elapsedMs.reset();
    timing.quality = TimingQuality::INVALID;

    // NOLINTNEXTLINE(misc-const-correctness)
    ScopedHipdnnBackendDescriptor profilingDesc(HIPDNN_BACKEND_PROFILING_CONTROL_EXT);
    if(!profilingDesc.valid())
    {
        return {ErrorCode::HIPDNN_BACKEND_ERROR, "Failed to create profiling control descriptor"};
    }

    HIPDNN_RETURN_ON_BACKEND_FAILURE(
        hipdnnBackend()->backendSetAttribute(profilingDesc.get(),
                                             HIPDNN_ATTR_PROFILING_HANDLE_EXT,
                                             HIPDNN_TYPE_HANDLE,
                                             1,
                                             static_cast<const void*>(&handle)),
        "Failed to set handle on profiling descriptor");

    // Stall the stream before recording start, so the measured span begins when the
    // device starts the work rather than when the host started submitting it. Arming is
    // silently skipped on a device without stream-wait-value support, and is skipped
    // altogether when `stalled` is false (the unstalled retry).
    bool stallVal = true;
    if(stalled)
    {
        HIPDNN_RETURN_ON_BACKEND_FAILURE(
            hipdnnBackend()->backendSetAttribute(profilingDesc.get(),
                                                 HIPDNN_ATTR_PROFILING_STALL_ARM_EXT,
                                                 HIPDNN_TYPE_BOOLEAN,
                                                 1,
                                                 &stallVal),
            "Failed to arm profiling stall");
    }

    bool startVal = true;
    HIPDNN_RETURN_ON_BACKEND_FAILURE(
        hipdnnBackend()->backendSetAttribute(profilingDesc.get(),
                                             HIPDNN_ATTR_PROFILING_START_EXT,
                                             HIPDNN_TYPE_BOOLEAN,
                                             1,
                                             &startVal),
        "Failed to set profiling start");

    // Exactly one execution: no warmup, no replay.
    HIPDNN_CHECK_ERROR(executeWithPlan(handle, execPlan, variantPackDesc));

    bool stopVal = true;
    HIPDNN_RETURN_ON_BACKEND_FAILURE(
        hipdnnBackend()->backendSetAttribute(
            profilingDesc.get(), HIPDNN_ATTR_PROFILING_STOP_EXT, HIPDNN_TYPE_BOOLEAN, 1, &stopVal),
        "Failed to set profiling stop");

    // Release the stall so the queued work runs. An early return before this point
    // destroys profilingDesc, and the descriptor's StallGate releases and drains; a
    // return after it leaves nothing armed.
    if(stalled)
    {
        HIPDNN_RETURN_ON_BACKEND_FAILURE(
            hipdnnBackend()->backendSetAttribute(profilingDesc.get(),
                                                 HIPDNN_ATTR_PROFILING_STALL_RELEASE_EXT,
                                                 HIPDNN_TYPE_BOOLEAN,
                                                 1,
                                                 &stallVal),
            "Failed to release profiling stall");
    }

    // Finalize synchronizes events and computes elapsed time.
    HIPDNN_RETURN_ON_BACKEND_FAILURE(hipdnnBackend()->backendFinalize(profilingDesc.get()),
                                     "Failed to finalize profiling descriptor");

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
        // timing.quality is already INVALID and timing.elapsedMs already empty from the
        // reset above; the watchdog-broken measurement is not published.
        return {ErrorCode::OK, ""};
    }

    timing.quality = stallUsed ? TimingQuality::DEVICE_ONLY : TimingQuality::HOST_INCLUDED;
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
