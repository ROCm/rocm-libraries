/* Copyright © Advanced Micro Devices, Inc., or its affiliates. */
/* SPDX-License-Identifier:  MIT */

#include <algorithm>
#include <limits>
#include <string>

#include <hipdnn_sdk/logging/Logger.hpp>
#include <hipdnn_sdk/plugin/PluginException.hpp>
#include <miopen/miopen.h>

#include "MiopenConvDescriptor.hpp"
#include "MiopenConvPlanBuilder.hpp"
#include "MiopenUtils.hpp"
#include "engines/plans/MiopenConvBwdPlan.hpp"
#include "engines/plans/MiopenConvFwdPlan.hpp"

namespace miopen_legacy_plugin
{

namespace
{

std::string getNodeName(const hipdnn_sdk::data_objects::Node& node)
{
    return node.name() != nullptr ? node.name()->str() : "";
}

bool isApplicableFwd(const HipdnnEnginePluginHandle& handle, const hipdnn_plugin::IGraph& opGraph)
{
    const auto& attr = opGraph.getNodeWrapper(0)
                           .attributesAs<hipdnn_sdk::data_objects::ConvolutionFwdAttributes>();

    // Check tensor attributes
    const auto& tensorMap = opGraph.getTensorMap();
    const auto& tensorAttrX = miopen_utils::findTensorAttributes(tensorMap, attr.x_tensor_uid());
    const auto& tensorAttrW = miopen_utils::findTensorAttributes(tensorMap, attr.w_tensor_uid());
    const auto& tensorAttrY = miopen_utils::findTensorAttributes(tensorMap, attr.y_tensor_uid());

    if(tensorAttrX.virtual_() || tensorAttrW.virtual_() || tensorAttrY.virtual_())
    {
        HIPDNN_LOG_WARN("All tensors must be non-virtual");
        return false;
    }

    size_t spatialDimCount;
    try
    {
        spatialDimCount = miopen_utils::getSpatialDimCount(tensorAttrX);
    }
    catch(const hipdnn_plugin::HipdnnPluginException& e)
    {
        HIPDNN_LOG_INFO(e.what());
        return false;
    }

    // Create MIOpen tensor descriptors
    const MiopenTensor tensorX(tensorAttrX);
    const MiopenTensor tensorW(tensorAttrW);
    const MiopenTensor tensorY(tensorAttrY);

    // Create MIOpen convolution descriptor
    MiopenConvDescriptor convDesc;
    try
    {
        convDesc = MiopenConvDescriptor(spatialDimCount, attr);
    }
    catch(const hipdnn_plugin::HipdnnPluginException& e)
    {
        HIPDNN_LOG_INFO(e.what());
        return false;
    }

    size_t solutionCount;
    auto status = miopenConvolutionForwardGetSolutionCount(handle.miopenHandle,
                                                           tensorW.tensorDescriptor(),
                                                           tensorX.tensorDescriptor(),
                                                           convDesc.convDescriptor(),
                                                           tensorY.tensorDescriptor(),
                                                           &solutionCount);
    if(status != miopenStatusSuccess)
    {
        return false;
    }
    return solutionCount != 0;
}

bool isApplicableBwd(const HipdnnEnginePluginHandle& handle, const hipdnn_plugin::IGraph& opGraph)
{
    const auto& attr = opGraph.getNodeWrapper(0)
                           .attributesAs<hipdnn_sdk::data_objects::ConvolutionBwdAttributes>();

    // Check tensor attributes
    const auto& tensorMap = opGraph.getTensorMap();
    const auto& tensorAttrDX = miopen_utils::findTensorAttributes(tensorMap, attr.dx_tensor_uid());
    const auto& tensorAttrW = miopen_utils::findTensorAttributes(tensorMap, attr.w_tensor_uid());
    const auto& tensorAttrDY = miopen_utils::findTensorAttributes(tensorMap, attr.dy_tensor_uid());

    if(tensorAttrDX.virtual_() || tensorAttrW.virtual_() || tensorAttrDY.virtual_())
    {
        HIPDNN_LOG_WARN("All tensors must be non-virtual");
        return false;
    }

    size_t spatialDimCount;
    try
    {
        spatialDimCount = miopen_utils::getSpatialDimCount(tensorAttrDX);
    }
    catch(const hipdnn_plugin::HipdnnPluginException& e)
    {
        HIPDNN_LOG_INFO(e.what());
        return false;
    }

    // Create MIOpen tensor descriptors
    const MiopenTensor tensorDX(tensorAttrDX);
    const MiopenTensor tensorW(tensorAttrW);
    const MiopenTensor tensorDY(tensorAttrDY);

    // Create MIOpen convolution descriptor
    MiopenConvDescriptor convDesc;
    try
    {
        convDesc = MiopenConvDescriptor(spatialDimCount, attr);
    }
    catch(const hipdnn_plugin::HipdnnPluginException& e)
    {
        HIPDNN_LOG_INFO(e.what());
        return false;
    }

    size_t solutionCount;
    auto status = miopenConvolutionBackwardDataGetSolutionCount(handle.miopenHandle,
                                                                tensorDY.tensorDescriptor(),
                                                                tensorW.tensorDescriptor(),
                                                                convDesc.convDescriptor(),
                                                                tensorDX.tensorDescriptor(),
                                                                &solutionCount);

    if(status != miopenStatusSuccess)
    {
        return false;
    }

    return solutionCount != 0;
}

size_t getWorkspaceSizeFwd(const HipdnnEnginePluginHandle& handle,
                           const hipdnn_plugin::IGraph& opGraph)
{
    const auto& attr = opGraph.getNodeWrapper(0)
                           .attributesAs<hipdnn_sdk::data_objects::ConvolutionFwdAttributes>();
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

size_t getWorkspaceSizeBwd(const HipdnnEnginePluginHandle& handle,
                           const hipdnn_plugin::IGraph& opGraph)
{
    const auto& attr = opGraph.getNodeWrapper(0)
                           .attributesAs<hipdnn_sdk::data_objects::ConvolutionBwdAttributes>();
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

void buildPlanFwd(const HipdnnEnginePluginHandle& handle,
                  const hipdnn_plugin::IGraph& opGraph,
                  HipdnnEnginePluginExecutionContext& executionContext)
{
    const auto& attr = opGraph.getNodeWrapper(0)
                           .attributesAs<hipdnn_sdk::data_objects::ConvolutionFwdAttributes>();
    ConvFwdParams params(attr, opGraph.getTensorMap());
    auto plan = std::make_unique<ConvFwdPlan>(handle, std::move(params));
    executionContext.setPlan(std::move(plan));
}

void buildPlanBwd(const HipdnnEnginePluginHandle& handle,
                  const hipdnn_plugin::IGraph& opGraph,
                  HipdnnEnginePluginExecutionContext& executionContext)
{
    const auto& attr = opGraph.getNodeWrapper(0)
                           .attributesAs<hipdnn_sdk::data_objects::ConvolutionBwdAttributes>();
    ConvBwdParams params(attr, opGraph.getTensorMap());
    auto plan = std::make_unique<ConvBwdPlan>(handle, std::move(params));
    executionContext.setPlan(std::move(plan));
}

} // namespace

bool MiopenConvPlanBuilder::isApplicable(const HipdnnEnginePluginHandle& handle,
                                         const hipdnn_plugin::IGraph& opGraph) const
{
    if(opGraph.nodeCount() != 1)
    {
        HIPDNN_LOG_INFO("Convolution plan builder is applicable only for single node graphs. Graph "
                        "has {} nodes",
                        opGraph.nodeCount());
        return false;
    }

    const auto& node = opGraph.getNode(0);
    bool ret = false;

    switch(node.attributes_type())
    {
    case hipdnn_sdk::data_objects::NodeAttributes::ConvolutionFwdAttributes:
        ret = isApplicableFwd(handle, opGraph);
        break;
    case hipdnn_sdk::data_objects::NodeAttributes::ConvolutionBwdAttributes:
        ret = isApplicableBwd(handle, opGraph);
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

size_t MiopenConvPlanBuilder::getWorkspaceSize(const HipdnnEnginePluginHandle& handle,
                                               const hipdnn_plugin::IGraph& opGraph) const
{
    if(opGraph.nodeCount() != 1)
    {
        throw hipdnn_plugin::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Convolution plan builder supports only single node graphs. Graph has "
                + std::to_string(opGraph.nodeCount()) + " nodes");
    }

    const auto& node = opGraph.getNode(0);

    switch(node.attributes_type())
    {
    case hipdnn_sdk::data_objects::NodeAttributes::ConvolutionFwdAttributes:
        return getWorkspaceSizeFwd(handle, opGraph);
    case hipdnn_sdk::data_objects::NodeAttributes::ConvolutionBwdAttributes:
        return getWorkspaceSizeBwd(handle, opGraph);
    default:
        throw hipdnn_plugin::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Unsupported node type for convolution plan builder: "
                + std::string(hipdnn_sdk::data_objects::toString(node.attributes_type())));
    }
}

void MiopenConvPlanBuilder::buildPlan(const HipdnnEnginePluginHandle& handle,
                                      const hipdnn_plugin::IGraph& opGraph,
                                      HipdnnEnginePluginExecutionContext& executionContext) const
{
    if(opGraph.nodeCount() != 1)
    {
        throw hipdnn_plugin::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Convolution plan builder supports only single node graphs. Graph has "
                + std::to_string(opGraph.nodeCount()) + " nodes");
    }

    const auto& node = opGraph.getNode(0);

    std::string nodeName = getNodeName(node);
    switch(node.attributes_type())
    {
    case hipdnn_sdk::data_objects::NodeAttributes::ConvolutionFwdAttributes:
        HIPDNN_LOG_INFO("Building convolution fwd plan for node: {}", nodeName);
        buildPlanFwd(handle, opGraph, executionContext);
        break;
    case hipdnn_sdk::data_objects::NodeAttributes::ConvolutionBwdAttributes:
        HIPDNN_LOG_INFO("Building convolution bwd plan for node: {}", nodeName);
        buildPlanBwd(handle, opGraph, executionContext);
        break;
    default:
        throw hipdnn_plugin::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Unsupported node type for convolution plan builder: "
                + std::string(hipdnn_sdk::data_objects::toString(node.attributes_type())));
    }
}

} // namespace miopen_legacy_plugin
