// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <limits>
#include <vector>

#include <hipdnn_data_sdk/logging/Logger.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>

#include "HipdnnEnginePluginExecutionContext.hpp"
#include "MiopenPlanCommon.hpp"
#include "MiopenUtils.hpp"

namespace miopen_plugin
{

namespace {

auto findBest20Solution(
    miopenHandle_t miopenHandle,
    miopenProblem_t problem,
    miopenFindOptions_t findOptions)
{
    size_t numSolutions;
    miopenSolution_t solution = nullptr;
    // Requesting only the best solution
    THROW_ON_MIOPEN_FAILURE(
        miopenFindSolutions(miopenHandle, problem, findOptions, &solution, &numSolutions, 1));

    if(solution == nullptr || numSolutions != 1)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            "miopenFindSolutions returned no solutions");
    }

    return hipdnn_data_sdk::utilities::ScopedResource<miopenSolution_t>(
        solution, [](miopenSolution_t s)
        {
            auto status = miopenDestroySolution(s);
            if(status != miopenStatusSuccess)
            {
                HIPDNN_LOG_ERROR("miopenDestroySolution failed");
            }
        });
}

auto find20SolutionByWorkspaceSize(
    miopenHandle_t miopenHandle,
    miopenProblem_t problem,
    miopenFindOptions_t findOptions,
    HipdnnEnginePluginExecutionContext::DebugMode debugMode)
{
    constexpr size_t DEBUG_MODE_MAX_REQUESTED_SOLUTIONS = 128;

    std::vector<miopenSolution_t> solutions(DEBUG_MODE_MAX_REQUESTED_SOLUTIONS);
    size_t numSolutions;

    THROW_ON_MIOPEN_FAILURE(miopenFindSolutions(
        miopenHandle, problem, findOptions, solutions.data(), &numSolutions, solutions.size()));

    if(numSolutions == 0)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            "miopenFindSolutions returned no solutions");
    }

    solutions.resize(numSolutions);

    std::vector<hipdnn_data_sdk::utilities::ScopedResource<miopenSolution_t>> scopedSolutions;
    scopedSolutions.reserve(numSolutions);

    for(auto& solution : solutions)
    {
        scopedSolutions.emplace_back(solution, [](miopenSolution_t s)
        {
            auto status = miopenDestroySolution(s);
            if(status != miopenStatusSuccess)
            {
                HIPDNN_LOG_ERROR("miopenDestroySolution failed");
            }
        });
    }

    hipdnn_data_sdk::utilities::ScopedResource<miopenSolution_t>* selectedSolution = nullptr;
    size_t selectedWorkspaceSize
        = (debugMode == HipdnnEnginePluginExecutionContext::DebugMode::FORCE_MIN_WORKSPACE)
              ? std::numeric_limits<size_t>::max()
              : 0;

    HIPDNN_LOG_INFO("Finding solution by workspace size: Found {} solutions", numSolutions);

    for(auto& scopedSolution : scopedSolutions)
    {
        size_t wsSize;
        THROW_ON_MIOPEN_FAILURE(miopenGetSolutionWorkspaceSize(scopedSolution.get(), &wsSize));

        HIPDNN_LOG_INFO("Solution: workspace_size={}", wsSize);

        if((debugMode == HipdnnEnginePluginExecutionContext::DebugMode::FORCE_MIN_WORKSPACE
            && wsSize <= selectedWorkspaceSize)
           || (debugMode == HipdnnEnginePluginExecutionContext::DebugMode::FORCE_MAX_WORKSPACE
               && wsSize >= selectedWorkspaceSize))
        {
            selectedSolution = &scopedSolution;
            selectedWorkspaceSize = wsSize;
        }
    }

    HIPDNN_LOG_INFO("Selected solution: workspace_size={}", selectedWorkspaceSize);

    return std::move(*selectedSolution);
}

} // namespace

hipdnn_data_sdk::utilities::ScopedResource<miopenSolution_t> find20Solution(
    miopenHandle_t miopenHandle,
    miopenProblem_t problem,
    const HipdnnEnginePluginExecutionContext& executionContext)
{
    miopenFindOptions_t findOptions;
    THROW_ON_MIOPEN_FAILURE(miopenCreateFindOptions(&findOptions));
    hipdnn_data_sdk::utilities::ScopedResource findOptionsRes(
        findOptions, [](miopenFindOptions_t fo) { std::ignore = miopenDestroyFindOptions(fo); });

    if(executionContext.workspaceSizeLimit().has_value())
    {
        THROW_ON_MIOPEN_FAILURE(miopenSetFindOptionWorkspaceLimit(
            findOptions, executionContext.workspaceSizeLimit().value()));
    }

    const auto debugMode = executionContext.debugMode();

    if(debugMode == HipdnnEnginePluginExecutionContext::DebugMode::FORCE_MIN_WORKSPACE
       || debugMode == HipdnnEnginePluginExecutionContext::DebugMode::FORCE_MAX_WORKSPACE)
    {
        return find20SolutionByWorkspaceSize(miopenHandle, problem, findOptions, debugMode);
    }

    return findBest20Solution(miopenHandle, problem, findOptions);
}

} // namespace miopen_plugin
