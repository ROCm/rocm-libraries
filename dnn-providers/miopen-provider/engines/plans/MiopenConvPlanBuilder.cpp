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

bool isApplicableFwd(const HipdnnEnginePluginHandle& handle,
                     const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph,
                     bool deterministicEnabled)
{
    const auto& attr = opGraph.getNodeWrapper(0)
                           .attributesAs<hipdnn_data_sdk::data_objects::ConvolutionFwdAttributes>();

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

bool isApplicableBwd(const HipdnnEnginePluginHandle& handle,
                     const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph,
                     bool deterministicEnabled)
{
    const auto& attr = opGraph.getNodeWrapper(0)
                           .attributesAs<hipdnn_data_sdk::data_objects::ConvolutionBwdAttributes>();

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

bool isApplicableWrw(const HipdnnEnginePluginHandle& handle,
                     const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph,
                     bool deterministicEnabled)
{
    const auto& attr = opGraph.getNodeWrapper(0)
                           .attributesAs<hipdnn_data_sdk::data_objects::ConvolutionWrwAttributes>();

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

size_t getWorkspaceSizeFwd(const HipdnnEnginePluginHandle& handle,
                           const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph,
                           bool deterministicEnabled)
{
    const auto& attr = opGraph.getNodeWrapper(0)
                           .attributesAs<hipdnn_data_sdk::data_objects::ConvolutionFwdAttributes>();
    ConvFwdParams params(attr, opGraph.getTensorMap(), deterministicEnabled);
    size_t workSpaceSize;
    THROW_ON_MIOPEN_FAILURE(miopenConvolutionForwardGetWorkSpaceSize(handle.miopenHandle,
                                                                     params.w().tensorDescriptor(),
                                                                     params.x().tensorDescriptor(),
                                                                     params.conv().convDescriptor(),
                                                                     params.y().tensorDescriptor(),
                                                                     &workSpaceSize));

    return workSpaceSize;
}

size_t getWorkspaceSizeBwd(const HipdnnEnginePluginHandle& handle,
                           const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph,
                           bool deterministicEnabled)
{
    const auto& attr = opGraph.getNodeWrapper(0)
                           .attributesAs<hipdnn_data_sdk::data_objects::ConvolutionBwdAttributes>();
    ConvBwdParams params(attr, opGraph.getTensorMap(), deterministicEnabled);
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

size_t getWorkspaceSizeWrw(const HipdnnEnginePluginHandle& handle,
                           const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph,
                           bool deterministicEnabled)
{
    const auto& attr = opGraph.getNodeWrapper(0)
                           .attributesAs<hipdnn_data_sdk::data_objects::ConvolutionWrwAttributes>();
    ConvWrwParams params(attr, opGraph.getTensorMap(), deterministicEnabled);
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
                  HipdnnEnginePluginExecutionContext& executionContext,
                  bool deterministicEnabled)
{
    const auto& attr = opGraph.getNodeWrapper(0)
                           .attributesAs<hipdnn_data_sdk::data_objects::ConvolutionFwdAttributes>();
    ConvFwdParams params(attr, opGraph.getTensorMap(), deterministicEnabled);
    auto plan = std::make_unique<ConvFwdPlan>(
        handle, std::move(params), executionContext.benchmarkingEnabled());
    executionContext.setPlan(std::move(plan));
}

void buildPlanBwd(const HipdnnEnginePluginHandle& handle,
                  const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph,
                  HipdnnEnginePluginExecutionContext& executionContext,
                  bool deterministicEnabled)
{
    const auto& attr = opGraph.getNodeWrapper(0)
                           .attributesAs<hipdnn_data_sdk::data_objects::ConvolutionBwdAttributes>();
    ConvBwdParams params(attr, opGraph.getTensorMap(), deterministicEnabled);
    auto plan = std::make_unique<ConvBwdPlan>(
        handle, std::move(params), executionContext.benchmarkingEnabled());
    executionContext.setPlan(std::move(plan));
}

void buildPlanWrw(const HipdnnEnginePluginHandle& handle,
                  const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph,
                  HipdnnEnginePluginExecutionContext& executionContext,
                  bool deterministicEnabled)
{
    const auto& attr = opGraph.getNodeWrapper(0)
                           .attributesAs<hipdnn_data_sdk::data_objects::ConvolutionWrwAttributes>();
    ConvWrwParams params(attr, opGraph.getTensorMap(), deterministicEnabled);
    auto plan = std::make_unique<ConvWrwPlan>(
        handle, std::move(params), executionContext.benchmarkingEnabled());
    executionContext.setPlan(std::move(plan));
}

} // namespace

bool MiopenConvPlanBuilder::isApplicable(
    const HipdnnEnginePluginHandle& handle,
    const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph) const
{
    if(opGraph.nodeCount() != 1)
    {
        HIPDNN_PLUGIN_LOG_INFO(
            "Convolution plan builder is applicable only for single node graphs. Graph "
            "has {} nodes",
            opGraph.nodeCount());
        return false;
    }

    if(opGraph.getNode(0).compute_data_type() != hipdnn_data_sdk::data_objects::DataType::FLOAT)
    {
        HIPDNN_PLUGIN_LOG_ERROR("Convolution plan builder only supports nodes with an fp32 "
                                "compute_data_type");
        return false;
    }

    const auto& node = opGraph.getNode(0);
    bool ret = false;

    switch(node.attributes_type())
    {
    case hipdnn_data_sdk::data_objects::NodeAttributes::ConvolutionFwdAttributes:
        ret = isApplicableFwd(handle, opGraph, _deterministic);
        break;
    case hipdnn_data_sdk::data_objects::NodeAttributes::ConvolutionBwdAttributes:
        ret = isApplicableBwd(handle, opGraph, _deterministic);
        break;
    case hipdnn_data_sdk::data_objects::NodeAttributes::ConvolutionWrwAttributes:
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

size_t MiopenConvPlanBuilder::getWorkspaceSize(
    const HipdnnEnginePluginHandle& handle,
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
        return getWorkspaceSizeFwd(handle, opGraph, _deterministic);
    case hipdnn_data_sdk::data_objects::NodeAttributes::ConvolutionBwdAttributes:
        return getWorkspaceSizeBwd(handle, opGraph, _deterministic);
    case hipdnn_data_sdk::data_objects::NodeAttributes::ConvolutionWrwAttributes:
        return getWorkspaceSizeWrw(handle, opGraph, _deterministic);
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
    [[maybe_unused]] const hipdnn_data_sdk::flatbuffer_utilities::IEngineConfig& engineConfig,
    HipdnnEnginePluginExecutionContext& executionContext) const
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
    case hipdnn_data_sdk::data_objects::NodeAttributes::ConvolutionFwdAttributes:
        HIPDNN_PLUGIN_LOG_INFO("Building convolution fwd plan for node: {}", nodeName);
        buildPlanFwd(handle, opGraph, executionContext, _deterministic);
        break;
    case hipdnn_data_sdk::data_objects::NodeAttributes::ConvolutionBwdAttributes:
        HIPDNN_PLUGIN_LOG_INFO("Building convolution bwd plan for node: {}", nodeName);
        buildPlanBwd(handle, opGraph, executionContext, _deterministic);
        break;
    case hipdnn_data_sdk::data_objects::NodeAttributes::ConvolutionWrwAttributes:
        HIPDNN_PLUGIN_LOG_INFO("Building convolution wrw plan for node: {}", nodeName);
        buildPlanWrw(handle, opGraph, executionContext, _deterministic);
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
    [[maybe_unused]] const HipdnnEnginePluginHandle& handle,
    [[maybe_unused]] const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph) const
{
    return {};
}

} // namespace miopen_plugin
