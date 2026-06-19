// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/Logging.hpp>
#include <hipdnn_frontend/Utilities.hpp>
#include <hipdnn_frontend/autotune/AutotuneTypes.hpp>
#include <hipdnn_frontend/autotune/KnobConstants.hpp>
#include <hipdnn_frontend/detail/GraphExecution.hpp>
#include <hipdnn_frontend/detail/ScopedHipdnnBackendDescriptor.hpp>
#include <hipdnn_frontend/knob/KnobSetting.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <string>
#include <unordered_map>
#include <vector>

namespace hipdnn_frontend::autotune::detail
{

// Run one timed iteration using a fresh profiling control descriptor.
// Creates descriptor, records START -> execute -> STOP -> finalize ->
// ELAPSED_MS. A new descriptor is created each call because
// ProfilingControlDescriptor does not support reset — setAttribute throws
// after finalize.
inline Error benchmarkOnce(hipdnnHandle_t handle,
                           ::hipdnn_frontend::detail::ScopedHipdnnBackendDescriptor& execPlan,
                           const std::unordered_map<int64_t, void*>& variantPack,
                           void* workspace,
                           float& elapsedMs)
{
    elapsedMs = 0.0f;

    // Create a fresh profiling descriptor for this iteration
    // NOLINTNEXTLINE(misc-const-correctness)
    ::hipdnn_frontend::detail::ScopedHipdnnBackendDescriptor profilingDesc(
        HIPDNN_BACKEND_PROFILING_CONTROL_EXT);
    if(!profilingDesc.valid())
    {
        return {ErrorCode::HIPDNN_BACKEND_ERROR, "Failed to create profiling control descriptor"};
    }

    // Set handle (creates HIP events)
    HIPDNN_RETURN_ON_BACKEND_FAILURE(
        ::hipdnn_frontend::detail::hipdnnBackend()->backendSetAttribute(
            profilingDesc.get(),
            HIPDNN_ATTR_PROFILING_HANDLE_EXT,
            HIPDNN_TYPE_HANDLE,
            1,
            static_cast<const void*>(&handle)),
        "Failed to set handle on profiling descriptor");

    // Record start event
    bool startVal = true;
    HIPDNN_RETURN_ON_BACKEND_FAILURE(
        ::hipdnn_frontend::detail::hipdnnBackend()->backendSetAttribute(profilingDesc.get(),
                                                                        HIPDNN_ATTR_PROFILING_START_EXT,
                                                                        HIPDNN_TYPE_BOOLEAN,
                                                                        1,
                                                                        &startVal),
        "Failed to set profiling start");

    // Execute
    HIPDNN_CHECK_ERROR(
        ::hipdnn_frontend::detail::executeWithPlan(handle, execPlan, variantPack, workspace));

    // Record stop event
    bool stopVal = true;
    HIPDNN_RETURN_ON_BACKEND_FAILURE(
        ::hipdnn_frontend::detail::hipdnnBackend()->backendSetAttribute(profilingDesc.get(),
                                                                        HIPDNN_ATTR_PROFILING_STOP_EXT,
                                                                        HIPDNN_TYPE_BOOLEAN,
                                                                        1,
                                                                        &stopVal),
        "Failed to set profiling stop");

    // Finalize synchronizes events and computes elapsed time
    HIPDNN_RETURN_ON_BACKEND_FAILURE(
        ::hipdnn_frontend::detail::hipdnnBackend()->backendFinalize(profilingDesc.get()),
        "Failed to finalize profiling descriptor");

    // Read elapsed time
    HIPDNN_RETURN_ON_BACKEND_FAILURE(
        ::hipdnn_frontend::detail::hipdnnBackend()->backendGetAttribute(
            profilingDesc.get(),
            HIPDNN_ATTR_PROFILING_ELAPSED_MS_EXT,
            HIPDNN_TYPE_FLOAT,
            1,
            nullptr,
            &elapsedMs),
        "Failed to get profiling elapsed ms");

    return {ErrorCode::OK, ""};
}

// Builds a failed AutotuneResult entry for a plan excluded by the
// workspace guard. The error message and reported workspace size use
// the actual compiled workspace; estimatedWorkspaceSize carries the
// pre-compile estimate.
inline AutotuneResult makeSkippedResult(int64_t engineId,
                                        const std::vector<KnobSetting>& knobSettings,
                                        int64_t estimatedWorkspaceSize,
                                        int64_t compiledWorkspaceSize,
                                        const AutotuneConfig& config,
                                        int64_t maxWorkspaceSize)
{
    AutotuneResult skippedResult;
    skippedResult.engineId = engineId;
    skippedResult.knobSettings = knobSettings;
    skippedResult.estimatedWorkspaceSize = estimatedWorkspaceSize;
    skippedResult.workspaceSize = compiledWorkspaceSize;
    skippedResult.succeeded = false;
    skippedResult.errorMessage = "Workspace size " + std::to_string(compiledWorkspaceSize)
                                 + " exceeds limit " + std::to_string(maxWorkspaceSize);

    skippedResult.engineName = ::hipdnn_frontend::detail::resolveEngineName(engineId);

    skippedResult.modeUsed = config.mode;
    skippedResult.ranExhaustive = false;
    skippedResult.strategyUsed = config.strategy;
    skippedResult.rank = -1;
    skippedResult.compiledPlanIndex = -1;

    return skippedResult;
}

// Builds a failed AutotuneResult entry for a plan barred by a persistent
// user filter (deselect_engines() engine ID or deselect_workspace_greater_than()).
// Mirrors the maxWorkspaceSize skipped-result shape (succeeded==false, rank==-1,
// compiledPlanIndex==-1) so deselect-barred plans surface as skipped results
// instead of silently vanishing from the benchmark loop.
inline AutotuneResult makeBarredResult(int64_t engineId,
                                       const std::vector<KnobSetting>& knobSettings,
                                       int64_t workspaceSize,
                                       const AutotuneConfig& config)
{
    AutotuneResult barredResult;
    barredResult.engineId = engineId;
    barredResult.knobSettings = knobSettings;
    barredResult.estimatedWorkspaceSize = workspaceSize;
    barredResult.workspaceSize = workspaceSize;
    barredResult.succeeded = false;
    barredResult.errorMessage = "Plan barred (engine ID or workspace deselect filter).";
    barredResult.engineName = ::hipdnn_frontend::detail::resolveEngineName(engineId);
    barredResult.modeUsed = config.mode;
    barredResult.ranExhaustive = false;
    barredResult.strategyUsed = config.strategy;
    barredResult.rank = -1;
    barredResult.compiledPlanIndex = -1;

    return barredResult;
}

// Copy knob settings while dropping the internal global.benchmarking knob,
// which is managed exclusively by autotune() in EXHAUSTIVE mode. Logs one
// warning per stripped knob, attributing it to callerName.
inline std::vector<KnobSetting> stripBenchmarkingKnob(const std::vector<KnobSetting>& settings,
                                                      const char* callerName)
{
    std::vector<KnobSetting> stripped;
    stripped.reserve(settings.size());
    for(const auto& setting : settings)
    {
        if(setting.knobId() == BENCHMARKING_KNOB_NAME)
        {
            HIPDNN_FE_LOG_WARN("Stripping internal knob '"
                               << BENCHMARKING_KNOB_NAME << "' from " << callerName << " call. "
                               << "This knob is managed by autotune() in EXHAUSTIVE mode.");
            continue;
        }
        stripped.push_back(setting);
    }
    return stripped;
}

// Rank benchmark results and select the winning plan.
//
// Sorts succeeded results (custom config.rankingFn if provided, otherwise by
// minTimeMs ascending), reassembles succeeded-then-failed, assigns 0-based
// ranks to succeeded results and -1 to failed ones, then sets activePlanIndex
// to the compiledPlanIndex of the first succeeded result. Returns a fatal error
// if no winner is found.
inline Error rankAndSelectWinner(std::vector<AutotuneResult>& allResults,
                                 const AutotuneConfig& config,
                                 size_t& activePlanIndex)
{
    // Separate succeeded and failed results
    std::vector<AutotuneResult> succeededResults;
    std::vector<AutotuneResult> failedResults;
    for(auto& r : allResults)
    {
        if(r.succeeded)
        {
            succeededResults.push_back(std::move(r));
        }
        else
        {
            failedResults.push_back(std::move(r));
        }
    }

    if(config.rankingFn)
    {
        // Pass only succeeded results to the user's ranking function
        try
        {
            config.rankingFn(succeededResults);
        }
        catch(const std::exception& e)
        {
            HIPDNN_FE_LOG_ERROR("autotune: custom ranking function threw an exception: "
                                << e.what() << ". Falling back to default ranking.");
            std::stable_sort(succeededResults.begin(),
                             succeededResults.end(),
                             [](const AutotuneResult& a, const AutotuneResult& b) {
                                 return a.minTimeMs < b.minTimeMs;
                             });
        }
        catch(...)
        {
            HIPDNN_FE_LOG_WARN("autotune: custom ranking function threw an unknown exception. "
                               "Falling back to default ranking.");
            std::stable_sort(succeededResults.begin(),
                             succeededResults.end(),
                             [](const AutotuneResult& a, const AutotuneResult& b) {
                                 return a.minTimeMs < b.minTimeMs;
                             });
        }
    }
    else
    {
        // Default ranking: succeeded engines by minTimeMs ascending
        std::stable_sort(succeededResults.begin(),
                         succeededResults.end(),
                         [](const AutotuneResult& a, const AutotuneResult& b) {
                             return a.minTimeMs < b.minTimeMs;
                         });
    }

    // Reassemble: succeeded first, then failed
    allResults.clear();
    allResults.reserve(succeededResults.size() + failedResults.size());
    for(auto& r : succeededResults)
    {
        allResults.push_back(std::move(r));
    }
    for(auto& r : failedResults)
    {
        allResults.push_back(std::move(r));
    }

    // Assign ranks: succeeded get 0-based ranks, failed get -1
    for(size_t i = 0; i < allResults.size(); ++i)
    {
        if(allResults[i].succeeded)
        {
            allResults[i].rank = static_cast<int>(i);
        }
        else
        {
            allResults[i].rank = -1;
        }
    }

    // ── Log ranking summary ────────────────────────────────────────
    {
        size_t succeededCount = 0;
        size_t failedCount = 0;
        for(const auto& r : allResults)
        {
            if(r.succeeded)
            {
                ++succeededCount;
            }
            else
            {
                ++failedCount;
            }
        }
        HIPDNN_FE_LOG_INFO("autotune: ranking complete — " << succeededCount << " succeeded, "
                                                           << failedCount << " failed");
    }

    // ── Select winner ───────────────────────────────────────────────
    // Find the first successful result and use its compiledPlanIndex
    // to set the active plan directly, avoiding the fragile O(n*m)
    // (engineId, knobSettings) search loop.
    bool winnerFound = false;
    for(const auto& result : allResults)
    {
        if(!result.succeeded || result.compiledPlanIndex < 0)
        {
            continue;
        }
        activePlanIndex = static_cast<size_t>(result.compiledPlanIndex);
        winnerFound = true;
        HIPDNN_FE_LOG_INFO("autotune: winner — engine "
                           << result.engineName << " (ID " << result.engineId
                           << "), min=" << result.minTimeMs << "ms");
        break;
    }

    if(!winnerFound)
    {
        return {ErrorCode::HIPDNN_BACKEND_ERROR,
                "All engines failed during autotuning. No winner selected."};
    }

    return {ErrorCode::OK, ""};
}

/// Synchronize the device between benchmarking phases.
///
/// Uses a one-shot ProfilingControlDescriptor to call hipDeviceSynchronize()
/// via the backend, keeping HIP calls out of the frontend header. Ensures the
/// GPU is idle from previous work before the next phase starts. Returns a
/// fatal error on any synchronization failure so the caller can mark the
/// affected plan's timing as unreliable.
inline Error syncDevice()
{
    // NOLINTNEXTLINE(misc-const-correctness)
    ::hipdnn_frontend::detail::ScopedHipdnnBackendDescriptor syncDesc(
        HIPDNN_BACKEND_PROFILING_CONTROL_EXT);
    if(!syncDesc.valid())
    {
        return {ErrorCode::HIPDNN_BACKEND_ERROR,
                "autotune: failed to create sync descriptor for device synchronization"};
    }
    bool syncVal = true;
    auto syncStatus = ::hipdnn_frontend::detail::hipdnnBackend()->backendSetAttribute(
        syncDesc.get(),
        HIPDNN_ATTR_PROFILING_DEVICE_SYNC_EXT,
        HIPDNN_TYPE_BOOLEAN,
        1,
        &syncVal);
    if(syncStatus != HIPDNN_STATUS_SUCCESS)
    {
        return {ErrorCode::HIPDNN_BACKEND_ERROR, "autotune: device sync setAttribute failed"};
    }
    // No backendFinalize() needed: sync is triggered immediately by
    // setAttribute(DEVICE_SYNC). finalize() would throw for sync-only
    // descriptors (no start/stop events recorded).
    return {ErrorCode::OK, ""};
}

} // namespace hipdnn_frontend::autotune::detail
