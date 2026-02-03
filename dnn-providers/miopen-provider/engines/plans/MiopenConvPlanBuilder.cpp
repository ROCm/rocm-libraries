// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#define DEBUG_WORKSPACE_LOGGING

#include <algorithm>
#include <limits>
#include <string>
#ifdef DEBUG_WORKSPACE_LOGGING
#include <iostream>
#endif

#include <hipdnn_data_sdk/logging/Logger.hpp>
#include <hipdnn_plugin_sdk/GlobalKnobDefines.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <miopen/miopen.h>

#include "MiopenConvDescriptor.hpp"
#include "MiopenConvPlanBuilder.hpp"
#include "MiopenUtils.hpp"
#include "engines/plans/MiopenConvBwdPlan.hpp"
#include "engines/plans/MiopenConvFwdPlan.hpp"
#include "engines/plans/MiopenConvWrwPlan.hpp"

namespace miopen_plugin
{

namespace
{

bool isApplicableFwd(const HipdnnEnginePluginHandle& handle,
                     const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph)
{
    const auto& attr = opGraph.getNodeWrapper(0)
                           .attributesAs<hipdnn_data_sdk::data_objects::ConvolutionFwdAttributes>();

    size_t solutionCount = 0;
    try
    {
        ConvFwdParams params(attr, opGraph.getTensorMap());

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
        HIPDNN_LOG_INFO(e.what());
        return false;
    }

    return solutionCount != 0;
}

bool isApplicableBwd(const HipdnnEnginePluginHandle& handle,
                     const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph)
{
    const auto& attr = opGraph.getNodeWrapper(0)
                           .attributesAs<hipdnn_data_sdk::data_objects::ConvolutionBwdAttributes>();

    size_t solutionCount = 0;
    try
    {
        ConvBwdParams params(attr, opGraph.getTensorMap());

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
        HIPDNN_LOG_INFO(e.what());
        return false;
    }

    return solutionCount != 0;
}

bool isApplicableWrw(const HipdnnEnginePluginHandle& handle,
                     const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph)
{
    const auto& attr = opGraph.getNodeWrapper(0)
                           .attributesAs<hipdnn_data_sdk::data_objects::ConvolutionWrwAttributes>();

    size_t solutionCount = 0;
    try
    {
        ConvWrwParams params(attr, opGraph.getTensorMap());

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
        HIPDNN_LOG_INFO(e.what());
        return false;
    }

    return solutionCount != 0;
}

IPlanBuilder::WorkspaceSizeRange getWorkspaceSizeRangeFwd(const HipdnnEnginePluginHandle& handle,
                                                           const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph)
{
    const auto& attr = opGraph.getNodeWrapper(0)
                           .attributesAs<hipdnn_data_sdk::data_objects::ConvolutionFwdAttributes>();
    ConvFwdParams params(attr, opGraph.getTensorMap());

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
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "No solutions found for forward convolution");
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

#ifdef DEBUG_WORKSPACE_LOGGING
    std::cerr << "Getting workspace size range for Convolution Fwd: Found " << returnedSolutionCount << " solutions\n";
#else
    HIPDNN_LOG_INFO("Getting workspace size range for Convolution Fwd: Found {} solutions", returnedSolutionCount);
#endif

    size_t minWorkspace = std::numeric_limits<size_t>::max();
    size_t maxWorkspace = 0;
    for(const auto& solution : solutions)
    {
#ifdef DEBUG_WORKSPACE_LOGGING
        std::cerr << "Convolution Fwd: solution_id=" << solution.solution_id
                  << ", algorithm=" << static_cast<int>(solution.algorithm)
                  << ", time=" << solution.time
                  << ", workspace_size=" << solution.workspace_size << "\n";
#else
        HIPDNN_LOG_INFO("Convolution Fwd: solution_id={}, algorithm={}, time={}, workspace_size={}",
                        solution.solution_id, static_cast<int>(solution.algorithm), solution.time, solution.workspace_size);
#endif
        minWorkspace = std::min(minWorkspace, solution.workspace_size);
        maxWorkspace = std::max(maxWorkspace, solution.workspace_size);
    }

#ifdef DEBUG_WORKSPACE_LOGGING
    std::cerr << "Convolution Fwd: Workspace range: min=" << minWorkspace
              << ", max=" << maxWorkspace << "\n";
#else
    HIPDNN_LOG_INFO("Convolution Fwd: Workspace range: min={}, max={}",
                    minWorkspace, maxWorkspace);
#endif

    return {minWorkspace, maxWorkspace};
}

IPlanBuilder::WorkspaceSizeRange getWorkspaceSizeRangeBwd(const HipdnnEnginePluginHandle& handle,
                                                           const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph)
{
    const auto& attr = opGraph.getNodeWrapper(0)
                           .attributesAs<hipdnn_data_sdk::data_objects::ConvolutionBwdAttributes>();
    ConvBwdParams params(attr, opGraph.getTensorMap());

    size_t solutionCount = 0;
    THROW_ON_MIOPEN_FAILURE(miopenConvolutionBackwardDataGetSolutionCount(handle.miopenHandle,
                                                                          params.dy().tensorDescriptor(),
                                                                          params.w().tensorDescriptor(),
                                                                          params.conv().convDescriptor(),
                                                                          params.dx().tensorDescriptor(),
                                                                          &solutionCount));

    if(solutionCount == 0)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "No solutions found for backward data convolution");
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

#ifdef DEBUG_WORKSPACE_LOGGING
    std::cerr << "Getting workspace size range for Convolution Bwd: Found " << returnedSolutionCount << " solutions\n";
#else
    HIPDNN_LOG_INFO("Getting workspace size range for Convolution Bwd: Found {} solutions", returnedSolutionCount);
#endif

    size_t minWorkspace = std::numeric_limits<size_t>::max();
    size_t maxWorkspace = 0;
    for(const auto& solution : solutions)
    {
#ifdef DEBUG_WORKSPACE_LOGGING
        std::cerr << "Convolution Bwd: solution_id=" << solution.solution_id
                  << ", algorithm=" << static_cast<int>(solution.algorithm)
                  << ", time=" << solution.time
                  << ", workspace_size=" << solution.workspace_size << "\n";
#else
        HIPDNN_LOG_INFO("Convolution Bwd: solution_id={}, algorithm={}, time={}, workspace_size={}",
                        solution.solution_id, static_cast<int>(solution.algorithm), solution.time, solution.workspace_size);
#endif
        minWorkspace = std::min(minWorkspace, solution.workspace_size);
        maxWorkspace = std::max(maxWorkspace, solution.workspace_size);
    }

#ifdef DEBUG_WORKSPACE_LOGGING
    std::cerr << "Convolution Bwd: Workspace range: min=" << minWorkspace
              << ", max=" << maxWorkspace << "\n";
#else
    HIPDNN_LOG_INFO("Convolution Bwd: Workspace range: min={}, max={}",
                    minWorkspace, maxWorkspace);
#endif

    return {minWorkspace, maxWorkspace};
}

IPlanBuilder::WorkspaceSizeRange getWorkspaceSizeRangeWrw(const HipdnnEnginePluginHandle& handle,
                                                           const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph)
{
    const auto& attr = opGraph.getNodeWrapper(0)
                           .attributesAs<hipdnn_data_sdk::data_objects::ConvolutionWrwAttributes>();
    ConvWrwParams params(attr, opGraph.getTensorMap());

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
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "No solutions found for backward weights convolution");
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

#ifdef DEBUG_WORKSPACE_LOGGING
    std::cerr << "Getting workspace size range for Convolution Wrw: Found " << returnedSolutionCount << " solutions\n";
#else
    HIPDNN_LOG_INFO("Getting workspace size range for Convolution Wrw: Found {} solutions", returnedSolutionCount);
#endif

    size_t minWorkspace = std::numeric_limits<size_t>::max();
    size_t maxWorkspace = 0;
    for(const auto& solution : solutions)
    {
#ifdef DEBUG_WORKSPACE_LOGGING
        std::cerr << "Convolution Wrw: solution_id=" << solution.solution_id
                  << ", algorithm=" << static_cast<int>(solution.algorithm)
                  << ", time=" << solution.time
                  << ", workspace_size=" << solution.workspace_size << "\n";
#else
        HIPDNN_LOG_INFO("Convolution Wrw: solution_id={}, algorithm={}, time={}, workspace_size={}",
                        solution.solution_id, static_cast<int>(solution.algorithm), solution.time, solution.workspace_size);
#endif
        minWorkspace = std::min(minWorkspace, solution.workspace_size);
        maxWorkspace = std::max(maxWorkspace, solution.workspace_size);
    }

#ifdef DEBUG_WORKSPACE_LOGGING
    std::cerr << "Convolution Wrw: Workspace range: min=" << minWorkspace
              << ", max=" << maxWorkspace << "\n";
#else
    HIPDNN_LOG_INFO("Convolution Wrw: Workspace range: min={}, max={}",
                    minWorkspace, maxWorkspace);
#endif

    return {minWorkspace, maxWorkspace};
}

size_t getMaxWorkspaceSizeFwd(const HipdnnEnginePluginHandle& handle,
                              const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph)
{
    const auto& attr = opGraph.getNodeWrapper(0)
                           .attributesAs<hipdnn_data_sdk::data_objects::ConvolutionFwdAttributes>();
    ConvFwdParams params(attr, opGraph.getTensorMap());
    size_t workSpaceSize;
    THROW_ON_MIOPEN_FAILURE(miopenConvolutionForwardGetWorkSpaceSize(handle.miopenHandle,
                                                                     params.w().tensorDescriptor(),
                                                                     params.x().tensorDescriptor(),
                                                                     params.conv().convDescriptor(),
                                                                     params.y().tensorDescriptor(),
                                                                     &workSpaceSize));

    return workSpaceSize;
}

size_t getMaxWorkspaceSizeBwd(const HipdnnEnginePluginHandle& handle,
                              const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph)
{
    const auto& attr = opGraph.getNodeWrapper(0)
                           .attributesAs<hipdnn_data_sdk::data_objects::ConvolutionBwdAttributes>();
    ConvBwdParams params(attr, opGraph.getTensorMap());
    size_t workSpaceSize;

    THROW_ON_MIOPEN_FAILURE(
        miopenConvolutionBackwardDataGetWorkSpaceSize(handle.miopenHandle,
                                                      params.dy().tensorDescriptor(),
                                                      params.w().tensorDescriptor(),
                                                      params.conv().convDescriptor(),
                                                      params.dx().tensorDescriptor(),
                                                      &workSpaceSize));

    return workSpaceSize;
}

size_t getMaxWorkspaceSizeWrw(const HipdnnEnginePluginHandle& handle,
                              const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph)
{
    const auto& attr = opGraph.getNodeWrapper(0)
                           .attributesAs<hipdnn_data_sdk::data_objects::ConvolutionWrwAttributes>();
    ConvWrwParams params(attr, opGraph.getTensorMap());
    size_t workSpaceSize;

    THROW_ON_MIOPEN_FAILURE(
        miopenConvolutionBackwardWeightsGetWorkSpaceSize(handle.miopenHandle,
                                                         params.dy().tensorDescriptor(),
                                                         params.x().tensorDescriptor(),
                                                         params.conv().convDescriptor(),
                                                         params.dw().tensorDescriptor(),
                                                         &workSpaceSize));

    return workSpaceSize;
}

void buildPlanFwd(const HipdnnEnginePluginHandle& handle,
                  const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph,
                  HipdnnEnginePluginExecutionContext& executionContext)
{
    const auto& attr = opGraph.getNodeWrapper(0)
                           .attributesAs<hipdnn_data_sdk::data_objects::ConvolutionFwdAttributes>();
    ConvFwdParams params(attr, opGraph.getTensorMap());
    auto plan = std::make_unique<ConvFwdPlan>(handle, std::move(params), executionContext);
    executionContext.setPlan(std::move(plan));
}

void buildPlanBwd(const HipdnnEnginePluginHandle& handle,
                  const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph,
                  HipdnnEnginePluginExecutionContext& executionContext)
{
    const auto& attr = opGraph.getNodeWrapper(0)
                           .attributesAs<hipdnn_data_sdk::data_objects::ConvolutionBwdAttributes>();
    ConvBwdParams params(attr, opGraph.getTensorMap());
    auto plan = std::make_unique<ConvBwdPlan>(handle, std::move(params), executionContext);
    executionContext.setPlan(std::move(plan));
}

void buildPlanWrw(const HipdnnEnginePluginHandle& handle,
                  const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph,
                  HipdnnEnginePluginExecutionContext& executionContext)
{
    const auto& attr = opGraph.getNodeWrapper(0)
                           .attributesAs<hipdnn_data_sdk::data_objects::ConvolutionWrwAttributes>();
    ConvWrwParams params(attr, opGraph.getTensorMap());
    auto plan = std::make_unique<ConvWrwPlan>(handle, std::move(params), executionContext);
    executionContext.setPlan(std::move(plan));
}

} // namespace

bool MiopenConvPlanBuilder::isApplicable(
    const HipdnnEnginePluginHandle& handle,
    const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph) const
{
    if(opGraph.nodeCount() != 1)
    {
        HIPDNN_LOG_INFO("Convolution plan builder is applicable only for single node graphs. Graph "
                        "has {} nodes",
                        opGraph.nodeCount());
        return false;
    }

    if(opGraph.getNode(0).compute_data_type() != hipdnn_data_sdk::data_objects::DataType::FLOAT)
    {
        HIPDNN_LOG_ERROR("Convolution plan builder only supports nodes with an fp32 "
                         "compute_data_type");
        return false;
    }

    const auto& node = opGraph.getNode(0);
    bool ret = false;

    switch(node.attributes_type())
    {
    case hipdnn_data_sdk::data_objects::NodeAttributes::ConvolutionFwdAttributes:
        ret = isApplicableFwd(handle, opGraph);
        break;
    case hipdnn_data_sdk::data_objects::NodeAttributes::ConvolutionBwdAttributes:
        ret = isApplicableBwd(handle, opGraph);
        break;
    case hipdnn_data_sdk::data_objects::NodeAttributes::ConvolutionWrwAttributes:
        ret = isApplicableWrw(handle, opGraph);
        break;
    default:
        break;
    }

    if(!ret)
    {
        HIPDNN_LOG_INFO("Convolution plan builder is not applicable for this graph");
    }
    return ret;
}

IPlanBuilder::WorkspaceSizeRange MiopenConvPlanBuilder::getWorkspaceSizeRange(
    const HipdnnEnginePluginHandle& handle, const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph) const
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
    case hipdnn_data_sdk::data_objects::NodeAttributes::ConvolutionFwdAttributes:
        return getWorkspaceSizeRangeFwd(handle, opGraph);
    case hipdnn_data_sdk::data_objects::NodeAttributes::ConvolutionBwdAttributes:
        return getWorkspaceSizeRangeBwd(handle, opGraph);
    case hipdnn_data_sdk::data_objects::NodeAttributes::ConvolutionWrwAttributes:
        return getWorkspaceSizeRangeWrw(handle, opGraph);
    default:
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Unsupported node type for convolution plan builder: "
                + std::string(hipdnn_data_sdk::data_objects::toString(node.attributes_type())));
    }
}

size_t MiopenConvPlanBuilder::getMaxWorkspaceSize(const HipdnnEnginePluginHandle& handle,
                                                  const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph) const
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
    case hipdnn_data_sdk::data_objects::NodeAttributes::ConvolutionFwdAttributes:
        return getMaxWorkspaceSizeFwd(handle, opGraph);
    case hipdnn_data_sdk::data_objects::NodeAttributes::ConvolutionBwdAttributes:
        return getMaxWorkspaceSizeBwd(handle, opGraph);
    case hipdnn_data_sdk::data_objects::NodeAttributes::ConvolutionWrwAttributes:
        return getMaxWorkspaceSizeWrw(handle, opGraph);
    default:
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Unsupported node type for convolution plan builder: "
                + std::string(hipdnn_data_sdk::data_objects::toString(node.attributes_type())));
    }
}

void MiopenConvPlanBuilder::buildPlan(
    const HipdnnEnginePluginHandle& handle,
    const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph,
    const hipdnn_data_sdk::flatbuffer_utilities::IEngineConfig& engineConfig,
    HipdnnEnginePluginExecutionContext& executionContext) const
{
    if(opGraph.nodeCount() != 1)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Convolution plan builder supports only single node graphs. Graph has "
                + std::to_string(opGraph.nodeCount()) + " nodes");
    }

    // Read deterministic knob setting
    bool deterministicEnabled = false;
    if(engineConfig.isValid()
       && engineConfig.hasKnobSetting(hipdnn_plugin_sdk::DETERMINISTIC_KNOB_NAME))
    {
        const auto& knobSetting
            = engineConfig.getKnobSettingByName(hipdnn_plugin_sdk::DETERMINISTIC_KNOB_NAME);
        if(knobSetting.valueType() == hipdnn_data_sdk::data_objects::KnobValue::IntValue)
        {
            auto value = knobSetting.valueAs<hipdnn_data_sdk::data_objects::IntValue>().value();
            deterministicEnabled = (value != 0);
        }
    }
    (void)deterministicEnabled; // Will be used in a follow-up PR

    const auto& nodeWrapper = opGraph.getNodeWrapper(0);
    const auto nodeName = nodeWrapper.name();

    switch(nodeWrapper.attributesType())
    {
    case hipdnn_data_sdk::data_objects::NodeAttributes::ConvolutionFwdAttributes:
        HIPDNN_LOG_INFO("Building convolution fwd plan for node: {}", nodeName);
        buildPlanFwd(handle, opGraph, executionContext);
        break;
    case hipdnn_data_sdk::data_objects::NodeAttributes::ConvolutionBwdAttributes:
        HIPDNN_LOG_INFO("Building convolution bwd plan for node: {}", nodeName);
        buildPlanBwd(handle, opGraph, executionContext);
        break;
    case hipdnn_data_sdk::data_objects::NodeAttributes::ConvolutionWrwAttributes:
        HIPDNN_LOG_INFO("Building convolution wrw plan for node: {}", nodeName);
        buildPlanWrw(handle, opGraph, executionContext);
        break;
    default:
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Unsupported node type for convolution plan builder: "
                + std::string(
                    hipdnn_data_sdk::data_objects::toString(nodeWrapper.attributesType())));
    }
}

std::vector<hipdnn_data_sdk::data_objects::KnobT> MiopenConvPlanBuilder::getCustomKnobs(
    const HipdnnEnginePluginHandle& handle,
    const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph) const
{
    std::vector<hipdnn_data_sdk::data_objects::KnobT> knobs;

    if(isApplicable(handle, opGraph))
    {
        hipdnn_data_sdk::data_objects::KnobT knob;
        knob.knob_id = hipdnn_plugin_sdk::DETERMINISTIC_KNOB_NAME;
        knob.description = "Enable deterministic mode";

        hipdnn_data_sdk::data_objects::IntValueT defaultValue;
        defaultValue.value = 0;
        knob.default_value.Set(defaultValue);

        hipdnn_data_sdk::data_objects::IntConstraintT constraint;
        constraint.min_value = 0;
        constraint.max_value = 1;
        constraint.step = 1;
        knob.constraint.Set(constraint);

        knobs.push_back(std::move(knob));
    }

    return knobs;
}

} // namespace miopen_plugin
