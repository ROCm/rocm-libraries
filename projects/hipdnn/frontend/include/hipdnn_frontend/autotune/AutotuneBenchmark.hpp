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
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace hipdnn_frontend::autotune::detail
{

// Run one timed iteration using a fresh profiling control descriptor.
// Creates descriptor, records START -> execute -> STOP -> finalize ->
// ELAPSED_MS. A new descriptor is created each call because
// ProfilingControlDescriptor does not support reset - setAttribute throws
// after finalize.
inline Error
    benchmarkOnce(hipdnnHandle_t handle,
                  ::hipdnn_frontend::detail::ScopedHipdnnBackendDescriptor& execPlan,
                  ::hipdnn_frontend::detail::ScopedHipdnnBackendDescriptor& variantPackDesc,
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
        ::hipdnn_frontend::detail::hipdnnBackend()->backendSetAttribute(
            profilingDesc.get(),
            HIPDNN_ATTR_PROFILING_START_EXT,
            HIPDNN_TYPE_BOOLEAN,
            1,
            &startVal),
        "Failed to set profiling start");

    // Execute
    HIPDNN_CHECK_ERROR(
        ::hipdnn_frontend::detail::executeWithPlan(handle, execPlan, variantPackDesc));

    // Record stop event
    bool stopVal = true;
    HIPDNN_RETURN_ON_BACKEND_FAILURE(
        ::hipdnn_frontend::detail::hipdnnBackend()->backendSetAttribute(
            profilingDesc.get(), HIPDNN_ATTR_PROFILING_STOP_EXT, HIPDNN_TYPE_BOOLEAN, 1, &stopVal),
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

// Initializes an AutotuneResult with the sub-set identity and config fields known
// before a candidate is benchmarked. The benchmark loop fills the timing, succeeded,
// rank, and compiledPlanIndex fields during/after timing.
inline AutotuneResult makeBenchmarkResult(int64_t engineId,
                                          const std::vector<KnobSetting>& knobSettings,
                                          int64_t estimatedWorkspaceSize,
                                          int64_t compiledWorkspaceSize,
                                          const AutotuneConfig& config)
{
    AutotuneResult result;
    result.engineId = engineId;
    result.knobSettings = knobSettings;
    result.estimatedWorkspaceSize = estimatedWorkspaceSize;
    result.workspaceSize = compiledWorkspaceSize;
    result.engineName = ::hipdnn_frontend::detail::resolveEngineName(engineId);
    result.modeUsed = config.mode;
    result.strategyUsed = config.strategy;

    return result;
}

// Shared factory for every non-benchmarked candidate (skipped / barred /
// finalize-failed / compile-failed / filtered). They differ only in the two
// workspace sizes and the error message; everything else is the common
// non-benchmarked values (succeeded==false, rank==-1, compiledPlanIndex==-1).
// A workspace size of -1 means "not applicable / never compiled".
inline AutotuneResult makeNonBenchmarkedResult(int64_t engineId,
                                               const std::vector<KnobSetting>& knobSettings,
                                               int64_t estimatedWorkspaceSize,
                                               int64_t compiledWorkspaceSize,
                                               const AutotuneConfig& config,
                                               const std::string& errorMessage,
                                               bool supportsExhaustive,
                                               bool ranExhaustive,
                                               const std::string& exhaustiveNotRunReason)
{
    AutotuneResult result;
    result.engineId = engineId;
    result.knobSettings = knobSettings;
    result.estimatedWorkspaceSize = estimatedWorkspaceSize;
    result.workspaceSize = compiledWorkspaceSize;
    result.succeeded = false;
    result.errorMessage = errorMessage;
    result.engineName = ::hipdnn_frontend::detail::resolveEngineName(engineId);
    result.modeUsed = config.mode;
    result.supportsExhaustive = supportsExhaustive;
    result.ranExhaustive = ranExhaustive;
    result.exhaustiveNotRunReason = exhaustiveNotRunReason;
    result.strategyUsed = config.strategy;
    result.rank = -1;
    result.compiledPlanIndex = -1;

    return result;
}

// Non-benchmarked result for a plan excluded by the runtime workspace ceiling. The
// reported workspace size is the actual compiled workspace; estimatedWorkspaceSize
// carries the pre-compile workspace size estimate.
inline AutotuneResult makeSkippedResult(int64_t engineId,
                                        const std::vector<KnobSetting>& knobSettings,
                                        int64_t estimatedWorkspaceSize,
                                        int64_t compiledWorkspaceSize,
                                        const AutotuneConfig& config,
                                        int64_t maxWorkspaceSize,
                                        bool supportsExhaustive,
                                        bool ranExhaustive,
                                        const std::string& exhaustiveNotRunReason)
{
    return makeNonBenchmarkedResult(engineId,
                                    knobSettings,
                                    estimatedWorkspaceSize,
                                    compiledWorkspaceSize,
                                    config,
                                    "Workspace size " + std::to_string(compiledWorkspaceSize)
                                        + " exceeds limit " + std::to_string(maxWorkspaceSize),
                                    supportsExhaustive,
                                    ranExhaustive,
                                    exhaustiveNotRunReason);
}

// Non-benchmarked result for a plan barred by a persistent user filter (deselect_engines()
// engine ID or deselect_workspace_greater_than()). compiledWorkspaceSize is -1
// when the plan was never compiled.
inline AutotuneResult makeBarredResult(int64_t engineId,
                                       const std::vector<KnobSetting>& knobSettings,
                                       int64_t estimatedWorkspaceSize,
                                       int64_t compiledWorkspaceSize,
                                       const AutotuneConfig& config,
                                       bool supportsExhaustive,
                                       bool ranExhaustive,
                                       const std::string& exhaustiveNotRunReason)
{
    return makeNonBenchmarkedResult(engineId,
                                    knobSettings,
                                    estimatedWorkspaceSize,
                                    compiledWorkspaceSize,
                                    config,
                                    "Plan barred (engine ID or workspace deselect filter).",
                                    supportsExhaustive,
                                    ranExhaustive,
                                    exhaustiveNotRunReason);
}

// Failed result for a plan whose execution-plan descriptor could not be compiled.
inline AutotuneResult makeCompileFailedResult(int64_t engineId,
                                              const std::vector<KnobSetting>& knobSettings,
                                              int64_t estimatedWorkspaceSize,
                                              const AutotuneConfig& config,
                                              const std::string& errorMessage,
                                              bool supportsExhaustive,
                                              bool ranExhaustive,
                                              const std::string& exhaustiveNotRunReason)
{
    return makeNonBenchmarkedResult(engineId,
                                    knobSettings,
                                    estimatedWorkspaceSize,
                                    /*compiledWorkspaceSize=*/-1,
                                    config,
                                    errorMessage,
                                    supportsExhaustive,
                                    ranExhaustive,
                                    exhaustiveNotRunReason);
}

// Failed result for a plan whose execution-plan descriptor could not be finalized.
inline AutotuneResult makeFinalizeFailedResult(int64_t engineId,
                                               const std::vector<KnobSetting>& knobSettings,
                                               const AutotuneConfig& config,
                                               const std::string& errorMessage,
                                               bool supportsExhaustive,
                                               bool ranExhaustive,
                                               const std::string& exhaustiveNotRunReason)
{
    return makeNonBenchmarkedResult(engineId,
                                    knobSettings,
                                    /*estimatedWorkspaceSize=*/-1,
                                    /*compiledWorkspaceSize=*/-1,
                                    config,
                                    errorMessage,
                                    supportsExhaustive,
                                    ranExhaustive,
                                    exhaustiveNotRunReason);
}

// Non-benchmarked result for a plan excluded by config.engineIdFilter.
inline AutotuneResult makeFilteredResult(int64_t engineId,
                                         const std::vector<KnobSetting>& knobSettings,
                                         int64_t estimatedWorkspaceSize,
                                         int64_t compiledWorkspaceSize,
                                         const AutotuneConfig& config,
                                         bool supportsExhaustive,
                                         bool ranExhaustive,
                                         const std::string& exhaustiveNotRunReason)
{
    return makeNonBenchmarkedResult(engineId,
                                    knobSettings,
                                    estimatedWorkspaceSize,
                                    compiledWorkspaceSize,
                                    config,
                                    "Plan excluded by engineIdFilter.",
                                    supportsExhaustive,
                                    ranExhaustive,
                                    exhaustiveNotRunReason);
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

    const size_t succeededCount = succeededResults.size();
    const size_t failedCount = failedResults.size();

    // Reassemble succeeded-then-failed while assigning ranks and selecting the
    // winner in a single pass. Succeeded results get 0-based ranks in sorted
    // order; failed results get -1. The winner is the first succeeded result
    // with a valid compiledPlanIndex, whose index sets the active plan directly
    // (avoiding the fragile O(n*m) (engineId, knobSettings) search loop).
    allResults.clear();
    std::optional<AutotuneResult> winner;
    for(size_t i = 0; i < succeededResults.size(); ++i)
    {
        succeededResults[i].rank = static_cast<int>(i);
        // Defensive check; a succeeded result implies valid index.
        if(!winner && succeededResults[i].compiledPlanIndex >= 0)
        {
            activePlanIndex = static_cast<size_t>(succeededResults[i].compiledPlanIndex);
            winner = succeededResults[i];
        }
        allResults.push_back(std::move(succeededResults[i]));
    }
    for(auto& result : failedResults)
    {
        result.rank = -1;
        allResults.push_back(std::move(result));
    }

    HIPDNN_FE_LOG_INFO("autotune: ranking complete - " << succeededCount << " succeeded, "
                                                       << failedCount << " failed");

    if(!winner)
    {
        return {ErrorCode::HIPDNN_BACKEND_ERROR,
                "All engines failed during autotuning. No winner selected."};
    }

    HIPDNN_FE_LOG_INFO("autotune: winner - engine " << winner->engineName << " (ID "
                                                    << winner->engineId
                                                    << "), min=" << winner->minTimeMs << "ms");

    return {ErrorCode::OK, ""};
}

// Synchronize the device between benchmarking phases.
//
// Uses a one-shot ProfilingControlDescriptor to call hipDeviceSynchronize()
// via the backend, keeping HIP calls out of the frontend header. Ensures the
// GPU is idle from previous work before the next phase starts. Returns a
// fatal error on any synchronization failure so the caller can mark the
// affected plan's timing as unreliable.
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
        syncDesc.get(), HIPDNN_ATTR_PROFILING_DEVICE_SYNC_EXT, HIPDNN_TYPE_BOOLEAN, 1, &syncVal);
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
