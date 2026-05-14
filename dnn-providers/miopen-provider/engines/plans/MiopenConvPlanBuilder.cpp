// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <algorithm>
#include <limits>
#include <string>

#include <hipdnn_plugin_sdk/GlobalKnobDefines.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <miopen/miopen.h>

#include "MiopenConvDescriptor.hpp"
#include "MiopenConvPlanBuilder.hpp"
#include "MiopenUtils.hpp"
#include "engines/plans/MiopenConvBwdPlan.hpp"
#include "engines/plans/MiopenConvFwdPlan.hpp"
#include "engines/plans/MiopenConvWrwPlan.hpp"

namespace miopen_plugin
{

MiopenConvPlanBuilder::MiopenConvPlanBuilder(bool deterministic)
    : _deterministic(deterministic)
{
}

namespace
{

bool isApplicableFwd(const HipdnnMiopenHandle& handle,
                     const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
                     bool deterministicEnabled)
{
    const auto& attr
        = opGraph.getNodeWrapper(0)
              .attributesAs<hipdnn_flatbuffers_sdk::data_objects::ConvolutionFwdAttributes>();

    size_t solutionCount = 0;
    try
    {
        ConvFwdParams params(attr, opGraph.getTensorMap(), deterministicEnabled);

        if(!params.validTensors())
        {
            return false;
        }

        auto status = miopenConvolutionForwardGetSolutionCount(handle.miopenHandle,
                                                               params.w().tensorDescriptor(),
                                                               params.x().tensorDescriptor(),
                                                               params.conv().convDescriptor(),
                                                               params.y().tensorDescriptor(),
                                                               &solutionCount);
        if(status != miopenStatusSuccess)
        {
            return false;
        }
    }
    catch(const std::exception& e)
    {
        HIPDNN_PLUGIN_LOG_INFO(e.what());
        return false;
    }

    return solutionCount != 0;
}

bool isApplicableBwd(const HipdnnMiopenHandle& handle,
                     const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
                     bool deterministicEnabled)
{
    const auto& attr
        = opGraph.getNodeWrapper(0)
              .attributesAs<hipdnn_flatbuffers_sdk::data_objects::ConvolutionBwdAttributes>();

    size_t solutionCount = 0;
    try
    {
        ConvBwdParams params(attr, opGraph.getTensorMap(), deterministicEnabled);

        if(!params.validTensors())
        {
            return false;
        }

        auto status = miopenConvolutionBackwardDataGetSolutionCount(handle.miopenHandle,
                                                                    params.dy().tensorDescriptor(),
                                                                    params.w().tensorDescriptor(),
                                                                    params.conv().convDescriptor(),
                                                                    params.dx().tensorDescriptor(),
                                                                    &solutionCount);
        if(status != miopenStatusSuccess)
        {
            return false;
        }
    }
    catch(const std::exception& e)
    {
        HIPDNN_PLUGIN_LOG_INFO(e.what());
        return false;
    }

    return solutionCount != 0;
}

bool isApplicableWrw(const HipdnnMiopenHandle& handle,
                     const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
                     bool deterministicEnabled)
{
    const auto& attr
        = opGraph.getNodeWrapper(0)
              .attributesAs<hipdnn_flatbuffers_sdk::data_objects::ConvolutionWrwAttributes>();

    size_t solutionCount = 0;
    try
    {
        ConvWrwParams params(attr, opGraph.getTensorMap(), deterministicEnabled);

        if(!params.validTensors())
        {
            return false;
        }

        auto status
            = miopenConvolutionBackwardWeightsGetSolutionCount(handle.miopenHandle,
                                                               params.dy().tensorDescriptor(),
                                                               params.x().tensorDescriptor(),
                                                               params.conv().convDescriptor(),
                                                               params.dw().tensorDescriptor(),
                                                               &solutionCount);
        if(status != miopenStatusSuccess)
        {
            return false;
        }
    }
    catch(const std::exception& e)
    {
        HIPDNN_PLUGIN_LOG_INFO(e.what());
        return false;
    }

    return solutionCount != 0;
}

// =============================================================================
// Workspace size range computation — hybrid query strategy
// =============================================================================
//
// MIOpen exposes two non-allocating, query-only APIs that both report something
// about workspace requirements, but neither alone gives us the {min, max} range
// we need for the workspace-size-limit knob:
//
//   1. miopenConvolution*GetWorkSpaceSize
//        Per its header doc, this returns "the minimum size of the workspace
//        that must be provided to miopenFindConvolutionForwardAlgorithm() in
//        order for the latter to find the best candidate" — i.e. the workspace
//        of the single fastest solver MIOpen would pick, not a maximum across
//        the applicable solver set. In default Fast/Hybrid Find mode the
//        implementation calls GetSolutions(maxSolutionCount=1) and returns
//        that solver's workspace (projects/miopen/src/convolution.cpp:387);
//        only when Fast/Hybrid Find fails to return a solution does it fall
//        through to a std::max over algorithm classes. We use it here as
//        `range.max` because plan execution sizes its workspace by calling
//        the same API, so by construction the value matches what MIOpen will
//        actually request.
//
//   2. miopenConvolution*GetSolutionCount / *GetSolution
//        Return a subset of applicable solutions sourced from find-db (if a
//        record exists for this problem) or from a heuristic / TunaNet fallback
//        otherwise. The header explicitly notes the result "might be based on
//        heuristics" and recommends Find for consistent performance results.
//        Each returned miopenConvSolution_t carries a per-solver workspace_size
//        field, which is reliable for the entries that ARE returned but only
//        covers the heuristic / find-db subset, NOT the full solver set.
//
// There is no public API for a true minimum across the applicable solver
// set. miopenFind*Algorithm comes closest but still deduplicates by
// algorithm class (ShrinkToFind10Results in convolutionocl.cpp:238) and
// silently drops solvers exceeding the caller-provided workspace cap; it
// also needs real device buffers and launches GPU kernels, so it isn't
// affordable at engine-selection time anyway. We therefore combine both
// query-only signals:
//
//   * Seed `range.min` and `range.max` with the GetWorkSpaceSize result,
//     then iterate the GetSolution subset updating each bound with
//     std::min / std::max. Both bounds start at the same value and only
//     move outward, so `min ≤ max` is guaranteed.
//
// If MIOpen ever exposes a direct minimum-workspace API for the full applicable
// solver set, this whole helper collapses into a single call alongside
// GetWorkSpaceSize.
// =============================================================================

MiopenConvPlanBuilder::WorkspaceSizeRange
    getWorkspaceSizeRangeFwd(const HipdnnMiopenHandle& handle,
                             const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
                             bool deterministicEnabled)
{
    const auto& attr
        = opGraph.getNodeWrapper(0)
              .attributesAs<hipdnn_flatbuffers_sdk::data_objects::ConvolutionFwdAttributes>();
    ConvFwdParams params(attr, opGraph.getTensorMap(), deterministicEnabled);

    // Seed both bounds with GetWorkSpaceSize; the loop below extends them with
    // std::min / std::max over the GetSolution subset.
    size_t maxWorkspace = 0;
    THROW_ON_MIOPEN_FAILURE(miopenConvolutionForwardGetWorkSpaceSize(handle.miopenHandle,
                                                                     params.w().tensorDescriptor(),
                                                                     params.x().tensorDescriptor(),
                                                                     params.conv().convDescriptor(),
                                                                     params.y().tensorDescriptor(),
                                                                     &maxWorkspace));

    size_t solutionCount = 0;
    THROW_ON_MIOPEN_FAILURE(miopenConvolutionForwardGetSolutionCount(handle.miopenHandle,
                                                                     params.w().tensorDescriptor(),
                                                                     params.x().tensorDescriptor(),
                                                                     params.conv().convDescriptor(),
                                                                     params.y().tensorDescriptor(),
                                                                     &solutionCount));

    if(solutionCount == 0)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM, "No solutions found for forward convolution");
    }

    std::vector<miopenConvSolution_t> solutions(solutionCount);
    size_t returnedSolutionCount = 0;
    THROW_ON_MIOPEN_FAILURE(miopenConvolutionForwardGetSolution(handle.miopenHandle,
                                                                params.w().tensorDescriptor(),
                                                                params.x().tensorDescriptor(),
                                                                params.conv().convDescriptor(),
                                                                params.y().tensorDescriptor(),
                                                                solutionCount,
                                                                &returnedSolutionCount,
                                                                solutions.data()));

    if(returnedSolutionCount == 0)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                                       "Convolution Fwd: GetSolutionCount reported "
                                                           + std::to_string(solutionCount)
                                                           + " but GetSolution returned 0");
    }

    // GetSolution may write fewer entries than solutionCount (the MIOpen header documents the
    // latter as a maximum).
    solutions.resize(returnedSolutionCount);

    HIPDNN_PLUGIN_LOG_INFO("Getting workspace size range for Convolution Fwd: Found "
                           << returnedSolutionCount << " solutions");

    // Seeded with the GetWorkSpaceSize result so min ≤ max is structurally guaranteed.
    size_t minWorkspace = maxWorkspace;
    for(const auto& solution : solutions)
    {
        HIPDNN_PLUGIN_LOG_INFO("Convolution Fwd: solution_id="
                               << solution.solution_id << ", algorithm="
                               << static_cast<int>(solution.algorithm) << ", time=" << solution.time
                               << ", workspace_size=" << solution.workspace_size);
        minWorkspace = std::min(minWorkspace, solution.workspace_size);
        maxWorkspace = std::max(maxWorkspace, solution.workspace_size);
    }

    HIPDNN_PLUGIN_LOG_INFO("Convolution Fwd: Workspace range: min=" << minWorkspace
                                                                    << ", max=" << maxWorkspace);

    return {minWorkspace, maxWorkspace};
}

MiopenConvPlanBuilder::WorkspaceSizeRange
    getWorkspaceSizeRangeBwd(const HipdnnMiopenHandle& handle,
                             const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
                             bool deterministicEnabled)
{
    const auto& attr
        = opGraph.getNodeWrapper(0)
              .attributesAs<hipdnn_flatbuffers_sdk::data_objects::ConvolutionBwdAttributes>();
    ConvBwdParams params(attr, opGraph.getTensorMap(), deterministicEnabled);

    // Seed both bounds with GetWorkSpaceSize; the loop below extends them with
    // std::min / std::max over the GetSolution subset.
    size_t maxWorkspace = 0;
    THROW_ON_MIOPEN_FAILURE(
        miopenConvolutionBackwardDataGetWorkSpaceSize(handle.miopenHandle,
                                                      params.dy().tensorDescriptor(),
                                                      params.w().tensorDescriptor(),
                                                      params.conv().convDescriptor(),
                                                      params.dx().tensorDescriptor(),
                                                      &maxWorkspace));

    size_t solutionCount = 0;
    THROW_ON_MIOPEN_FAILURE(
        miopenConvolutionBackwardDataGetSolutionCount(handle.miopenHandle,
                                                      params.dy().tensorDescriptor(),
                                                      params.w().tensorDescriptor(),
                                                      params.conv().convDescriptor(),
                                                      params.dx().tensorDescriptor(),
                                                      &solutionCount));

    if(solutionCount == 0)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM, "No solutions found for backward data convolution");
    }

    std::vector<miopenConvSolution_t> solutions(solutionCount);
    size_t returnedSolutionCount = 0;
    THROW_ON_MIOPEN_FAILURE(miopenConvolutionBackwardDataGetSolution(handle.miopenHandle,
                                                                     params.dy().tensorDescriptor(),
                                                                     params.w().tensorDescriptor(),
                                                                     params.conv().convDescriptor(),
                                                                     params.dx().tensorDescriptor(),
                                                                     solutionCount,
                                                                     &returnedSolutionCount,
                                                                     solutions.data()));

    if(returnedSolutionCount == 0)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                                       "Convolution Bwd: GetSolutionCount reported "
                                                           + std::to_string(solutionCount)
                                                           + " but GetSolution returned 0");
    }

    // GetSolution may write fewer entries than solutionCount (the MIOpen header documents the
    // latter as a maximum).
    solutions.resize(returnedSolutionCount);

    HIPDNN_PLUGIN_LOG_INFO("Getting workspace size range for Convolution Bwd: Found "
                           << returnedSolutionCount << " solutions");

    // Seeded with the GetWorkSpaceSize result so min ≤ max is structurally guaranteed.
    size_t minWorkspace = maxWorkspace;
    for(const auto& solution : solutions)
    {
        HIPDNN_PLUGIN_LOG_INFO("Convolution Bwd: solution_id="
                               << solution.solution_id << ", algorithm="
                               << static_cast<int>(solution.algorithm) << ", time=" << solution.time
                               << ", workspace_size=" << solution.workspace_size);
        minWorkspace = std::min(minWorkspace, solution.workspace_size);
        maxWorkspace = std::max(maxWorkspace, solution.workspace_size);
    }

    HIPDNN_PLUGIN_LOG_INFO("Convolution Bwd: Workspace range: min=" << minWorkspace
                                                                    << ", max=" << maxWorkspace);

    return {minWorkspace, maxWorkspace};
}

MiopenConvPlanBuilder::WorkspaceSizeRange
    getWorkspaceSizeRangeWrw(const HipdnnMiopenHandle& handle,
                             const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
                             bool deterministicEnabled)
{
    const auto& attr
        = opGraph.getNodeWrapper(0)
              .attributesAs<hipdnn_flatbuffers_sdk::data_objects::ConvolutionWrwAttributes>();
    ConvWrwParams params(attr, opGraph.getTensorMap(), deterministicEnabled);

    // Seed both bounds with GetWorkSpaceSize; the loop below extends them with
    // std::min / std::max over the GetSolution subset.
    size_t maxWorkspace = 0;
    THROW_ON_MIOPEN_FAILURE(
        miopenConvolutionBackwardWeightsGetWorkSpaceSize(handle.miopenHandle,
                                                         params.dy().tensorDescriptor(),
                                                         params.x().tensorDescriptor(),
                                                         params.conv().convDescriptor(),
                                                         params.dw().tensorDescriptor(),
                                                         &maxWorkspace));

    size_t solutionCount = 0;
    THROW_ON_MIOPEN_FAILURE(
        miopenConvolutionBackwardWeightsGetSolutionCount(handle.miopenHandle,
                                                         params.dy().tensorDescriptor(),
                                                         params.x().tensorDescriptor(),
                                                         params.conv().convDescriptor(),
                                                         params.dw().tensorDescriptor(),
                                                         &solutionCount));

    if(solutionCount == 0)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM, "No solutions found for backward weights convolution");
    }

    std::vector<miopenConvSolution_t> solutions(solutionCount);
    size_t returnedSolutionCount = 0;
    THROW_ON_MIOPEN_FAILURE(
        miopenConvolutionBackwardWeightsGetSolution(handle.miopenHandle,
                                                    params.dy().tensorDescriptor(),
                                                    params.x().tensorDescriptor(),
                                                    params.conv().convDescriptor(),
                                                    params.dw().tensorDescriptor(),
                                                    solutionCount,
                                                    &returnedSolutionCount,
                                                    solutions.data()));

    if(returnedSolutionCount == 0)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                                       "Convolution Wrw: GetSolutionCount reported "
                                                           + std::to_string(solutionCount)
                                                           + " but GetSolution returned 0");
    }

    // GetSolution may write fewer entries than solutionCount (the MIOpen header documents the
    // latter as a maximum).
    solutions.resize(returnedSolutionCount);

    HIPDNN_PLUGIN_LOG_INFO("Getting workspace size range for Convolution Wrw: Found "
                           << returnedSolutionCount << " solutions");

    // Seeded with the GetWorkSpaceSize result so min ≤ max is structurally guaranteed.
    size_t minWorkspace = maxWorkspace;
    for(const auto& solution : solutions)
    {
        HIPDNN_PLUGIN_LOG_INFO("Convolution Wrw: solution_id="
                               << solution.solution_id << ", algorithm="
                               << static_cast<int>(solution.algorithm) << ", time=" << solution.time
                               << ", workspace_size=" << solution.workspace_size);
        minWorkspace = std::min(minWorkspace, solution.workspace_size);
        maxWorkspace = std::max(maxWorkspace, solution.workspace_size);
    }

    HIPDNN_PLUGIN_LOG_INFO("Convolution Wrw: Workspace range: min=" << minWorkspace
                                                                    << ", max=" << maxWorkspace);

    return {minWorkspace, maxWorkspace};
}

size_t getMaxWorkspaceSizeFwd(const HipdnnMiopenHandle& handle,
                              const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
                              const HipdnnMiopenSettings& executionSettings,
                              bool deterministicEnabled)
{
    size_t workSpaceSize = executionSettings.selectedWorkspaceSize();

    if(workSpaceSize == 0)
    {
        const auto& attr
            = opGraph.getNodeWrapper(0)
                  .attributesAs<hipdnn_flatbuffers_sdk::data_objects::ConvolutionFwdAttributes>();
        ConvFwdParams params(attr, opGraph.getTensorMap(), deterministicEnabled);
        THROW_ON_MIOPEN_FAILURE(
            miopenConvolutionForwardGetWorkSpaceSize(handle.miopenHandle,
                                                     params.w().tensorDescriptor(),
                                                     params.x().tensorDescriptor(),
                                                     params.conv().convDescriptor(),
                                                     params.y().tensorDescriptor(),
                                                     &workSpaceSize));
    }

    return workSpaceSize;
}

size_t getMaxWorkspaceSizeBwd(const HipdnnMiopenHandle& handle,
                              const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
                              const HipdnnMiopenSettings& executionSettings,
                              bool deterministicEnabled)
{
    size_t workSpaceSize = executionSettings.selectedWorkspaceSize();

    if(workSpaceSize == 0)
    {
        const auto& attr
            = opGraph.getNodeWrapper(0)
                  .attributesAs<hipdnn_flatbuffers_sdk::data_objects::ConvolutionBwdAttributes>();
        ConvBwdParams params(attr, opGraph.getTensorMap(), deterministicEnabled);

        THROW_ON_MIOPEN_FAILURE(
            miopenConvolutionBackwardDataGetWorkSpaceSize(handle.miopenHandle,
                                                          params.dy().tensorDescriptor(),
                                                          params.w().tensorDescriptor(),
                                                          params.conv().convDescriptor(),
                                                          params.dx().tensorDescriptor(),
                                                          &workSpaceSize));
    }

    return workSpaceSize;
}

size_t getMaxWorkspaceSizeWrw(const HipdnnMiopenHandle& handle,
                              const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
                              const HipdnnMiopenSettings& executionSettings,
                              bool deterministicEnabled)
{
    size_t workSpaceSize = executionSettings.selectedWorkspaceSize();

    if(workSpaceSize == 0)
    {
        const auto& attr
            = opGraph.getNodeWrapper(0)
                  .attributesAs<hipdnn_flatbuffers_sdk::data_objects::ConvolutionWrwAttributes>();
        ConvWrwParams params(attr, opGraph.getTensorMap(), deterministicEnabled);

        THROW_ON_MIOPEN_FAILURE(
            miopenConvolutionBackwardWeightsGetWorkSpaceSize(handle.miopenHandle,
                                                             params.dy().tensorDescriptor(),
                                                             params.x().tensorDescriptor(),
                                                             params.conv().convDescriptor(),
                                                             params.dw().tensorDescriptor(),
                                                             &workSpaceSize));
    }

    return workSpaceSize;
}

void buildPlanFwd(const HipdnnMiopenHandle& handle,
                  const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
                  HipdnnMiopenContext& executionContext,
                  bool deterministicEnabled)
{
    const auto& attr
        = opGraph.getNodeWrapper(0)
              .attributesAs<hipdnn_flatbuffers_sdk::data_objects::ConvolutionFwdAttributes>();
    ConvFwdParams params(attr, opGraph.getTensorMap(), deterministicEnabled);
    auto plan = std::make_unique<ConvFwdPlan>(
        handle, std::move(params), executionContext.executionSettings());
    executionContext.setPlan(std::move(plan));
}

void buildPlanBwd(const HipdnnMiopenHandle& handle,
                  const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
                  HipdnnMiopenContext& executionContext,
                  bool deterministicEnabled)
{
    const auto& attr
        = opGraph.getNodeWrapper(0)
              .attributesAs<hipdnn_flatbuffers_sdk::data_objects::ConvolutionBwdAttributes>();
    ConvBwdParams params(attr, opGraph.getTensorMap(), deterministicEnabled);
    auto plan = std::make_unique<ConvBwdPlan>(
        handle, std::move(params), executionContext.executionSettings());
    executionContext.setPlan(std::move(plan));
}

void buildPlanWrw(const HipdnnMiopenHandle& handle,
                  const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
                  HipdnnMiopenContext& executionContext,
                  bool deterministicEnabled)
{
    const auto& attr
        = opGraph.getNodeWrapper(0)
              .attributesAs<hipdnn_flatbuffers_sdk::data_objects::ConvolutionWrwAttributes>();
    ConvWrwParams params(attr, opGraph.getTensorMap(), deterministicEnabled);
    auto plan = std::make_unique<ConvWrwPlan>(
        handle, std::move(params), executionContext.executionSettings());
    executionContext.setPlan(std::move(plan));
}

} // namespace

bool MiopenConvPlanBuilder::isApplicable(
    const HipdnnMiopenHandle& handle,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const
{
    if(opGraph.nodeCount() != 1)
    {
        HIPDNN_PLUGIN_LOG_INFO(
            "Convolution plan builder is applicable only for single node graphs. Graph "
            "has "
            << opGraph.nodeCount() << " nodes");
        return false;
    }

    if(opGraph.getNode(0).compute_data_type()
       != hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT)
    {
        HIPDNN_PLUGIN_LOG_ERROR("Convolution plan builder only supports nodes with an fp32 "
                                "compute_data_type");
        return false;
    }

    const auto& node = opGraph.getNode(0);
    bool ret = false;

    switch(node.attributes_type())
    {
    case hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::ConvolutionFwdAttributes:
        ret = isApplicableFwd(handle, opGraph, _deterministic);
        break;
    case hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::ConvolutionBwdAttributes:
        ret = isApplicableBwd(handle, opGraph, _deterministic);
        break;
    case hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::ConvolutionWrwAttributes:
        ret = isApplicableWrw(handle, opGraph, _deterministic);
        break;
    default:
        break;
    }

    if(!ret)
    {
        HIPDNN_PLUGIN_LOG_INFO("Convolution plan builder is not applicable for this graph");
    }
    return ret;
}

MiopenConvPlanBuilder::WorkspaceSizeRange MiopenConvPlanBuilder::getWorkspaceSizeRange(
    const HipdnnMiopenHandle& handle,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const
{
    if(opGraph.nodeCount() != 1)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Convolution plan builder supports only single node graphs. Graph has "
                + std::to_string(opGraph.nodeCount()) + " nodes");
    }

    const auto& node = opGraph.getNode(0);

    switch(node.attributes_type())
    {
    case hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::ConvolutionFwdAttributes:
        return getWorkspaceSizeRangeFwd(handle, opGraph, _deterministic);
    case hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::ConvolutionBwdAttributes:
        return getWorkspaceSizeRangeBwd(handle, opGraph, _deterministic);
    case hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::ConvolutionWrwAttributes:
        return getWorkspaceSizeRangeWrw(handle, opGraph, _deterministic);
    default:
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Unsupported node type for convolution plan builder: "
                + std::string(
                    hipdnn_flatbuffers_sdk::data_objects::toString(node.attributes_type())));
    }
}

size_t MiopenConvPlanBuilder::getMaxWorkspaceSize(
    const HipdnnMiopenHandle& handle,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
    const HipdnnMiopenSettings& executionSettings) const
{
    if(opGraph.nodeCount() != 1)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Convolution plan builder supports only single node graphs. Graph has "
                + std::to_string(opGraph.nodeCount()) + " nodes");
    }

    const auto& node = opGraph.getNode(0);

    switch(node.attributes_type())
    {
    case hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::ConvolutionFwdAttributes:
        return getMaxWorkspaceSizeFwd(handle, opGraph, executionSettings, _deterministic);
    case hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::ConvolutionBwdAttributes:
        return getMaxWorkspaceSizeBwd(handle, opGraph, executionSettings, _deterministic);
    case hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::ConvolutionWrwAttributes:
        return getMaxWorkspaceSizeWrw(handle, opGraph, executionSettings, _deterministic);
    default:
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Unsupported node type for convolution plan builder: "
                + std::string(
                    hipdnn_flatbuffers_sdk::data_objects::toString(node.attributes_type())));
    }
}

void MiopenConvPlanBuilder::initializeExecutionSettings(
    const HipdnnMiopenHandle& handle,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& engineConfig,
    HipdnnMiopenSettings& executionSettings) const
{
    // Read workspace size limit knob setting
    if(engineConfig.isValid()
       && engineConfig.hasKnobSetting(hipdnn_plugin_sdk::WORKSPACE_SIZE_LIMIT_KNOB_NAME))
    {
        const auto& knobSetting
            = engineConfig.getKnobSettingByName(hipdnn_plugin_sdk::WORKSPACE_SIZE_LIMIT_KNOB_NAME);

        if(knobSetting.valueType() != hipdnn_flatbuffers_sdk::data_objects::KnobValue::IntValue)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "Workspace size limit knob setting value is not an integer. Type: "
                    + std::string(hipdnn_flatbuffers_sdk::data_objects::EnumNameKnobValue(
                        knobSetting.valueType())));
        }

        auto value = knobSetting.valueAs<hipdnn_flatbuffers_sdk::data_objects::IntValue>().value();

        if(value < 0)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
                "Invalid workspace size limit value: " + std::to_string(value) + ". Must be >= 0");
        }

        const auto range = getWorkspaceSizeRange(handle, opGraph);

        if(static_cast<size_t>(value) < range.min || static_cast<size_t>(value) > range.max)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
                "Invalid workspace size limit value: " + std::to_string(value)
                    + ". Must be in range [" + std::to_string(range.min) + ", "
                    + std::to_string(range.max) + "]");
        }

        executionSettings.setWorkspaceSizeLimit(static_cast<size_t>(value));
        executionSettings.setDefaultWorkspaceSize(range.max);
    }

    if(!executionSettings.workspaceSizeLimit().has_value())
    {
        const auto maxWs = getMaxWorkspaceSize(handle, opGraph, executionSettings);
        executionSettings.setDefaultWorkspaceSize(maxWs);
    }
}

void MiopenConvPlanBuilder::buildPlan(
    const HipdnnMiopenHandle& handle,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
    [[maybe_unused]] const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig&
        engineConfig,
    HipdnnMiopenContext& executionContext) const
{
    if(opGraph.nodeCount() != 1)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Convolution plan builder supports only single node graphs. Graph has "
                + std::to_string(opGraph.nodeCount()) + " nodes");
    }

    const auto& nodeWrapper = opGraph.getNodeWrapper(0);
    const auto nodeName = nodeWrapper.name();

    switch(nodeWrapper.attributesType())
    {
    case hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::ConvolutionFwdAttributes:
        HIPDNN_PLUGIN_LOG_INFO("Building convolution fwd plan for node: " << nodeName);
        buildPlanFwd(handle, opGraph, executionContext, _deterministic);
        break;
    case hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::ConvolutionBwdAttributes:
        HIPDNN_PLUGIN_LOG_INFO("Building convolution bwd plan for node: " << nodeName);
        buildPlanBwd(handle, opGraph, executionContext, _deterministic);
        break;
    case hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::ConvolutionWrwAttributes:
        HIPDNN_PLUGIN_LOG_INFO("Building convolution wrw plan for node: " << nodeName);
        buildPlanWrw(handle, opGraph, executionContext, _deterministic);
        break;
    default:
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Unsupported node type for convolution plan builder: "
                + std::string(
                    hipdnn_flatbuffers_sdk::data_objects::toString(nodeWrapper.attributesType())));
    }
}

std::vector<hipdnn_flatbuffers_sdk::data_objects::KnobT> MiopenConvPlanBuilder::getCustomKnobs(
    const HipdnnMiopenHandle& handle,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const
{
    std::vector<hipdnn_flatbuffers_sdk::data_objects::KnobT> knobs;

    if(!isApplicable(handle, opGraph))
    {
        return knobs;
    }

    // Workspace size limit knob
    const auto range = getWorkspaceSizeRange(handle, opGraph);

    // Validate that size_t values can fit into int64_t
    if(range.min > std::numeric_limits<int64_t>::max())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            "Workspace size range minimum (" + std::to_string(range.min)
                + ") exceeds maximum representable int64_t value");
    }
    if(range.max > std::numeric_limits<int64_t>::max())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            "Workspace size range maximum (" + std::to_string(range.max)
                + ") exceeds maximum representable int64_t value");
    }

    const auto minWorkspace = static_cast<int64_t>(range.min);
    const auto maxWorkspace = static_cast<int64_t>(range.max);

    hipdnn_flatbuffers_sdk::data_objects::KnobT workspaceKnob;
    workspaceKnob.knob_id = hipdnn_plugin_sdk::WORKSPACE_SIZE_LIMIT_KNOB_NAME;
    workspaceKnob.description = "Workspace size limit in bytes";

    hipdnn_flatbuffers_sdk::data_objects::IntValueT workspaceDefaultValue;
    workspaceDefaultValue.value = maxWorkspace;
    workspaceKnob.default_value.Set(workspaceDefaultValue);

    hipdnn_flatbuffers_sdk::data_objects::IntConstraintT workspaceConstraint;
    workspaceConstraint.min_value = minWorkspace;
    workspaceConstraint.max_value = maxWorkspace;
    workspaceConstraint.step = 1;
    workspaceKnob.constraint.Set(workspaceConstraint);

    knobs.push_back(std::move(workspaceKnob));

    return knobs;
}

} // namespace miopen_plugin
