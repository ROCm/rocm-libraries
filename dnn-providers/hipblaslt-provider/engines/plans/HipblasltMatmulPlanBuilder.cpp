// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <limits>
#include <numeric>
#include <string>

#include <hipblaslt/hipblaslt.h>
#include <hipdnn_data_sdk/logging/Logger.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>

#include "HipblasltMatmulPlan.hpp"
#include "HipblasltMatmulPlanBuilder.hpp"

namespace hipblaslt_plugin
{
namespace
{
bool checkDataTypes(const hipdnn_plugin_sdk::TensorAttributesWrapper& tA,
                    const hipdnn_plugin_sdk::TensorAttributesWrapper& tB,
                    const hipdnn_plugin_sdk::TensorAttributesWrapper& tC)
{
    const auto& aType = tA.dataType();
    const auto& bType = tB.dataType();
    const auto& cType = tC.dataType();

    static constexpr std::array<hipdnn_data_sdk::data_objects::DataType, 3> validDataTypes
        = {hipdnn_data_sdk::data_objects::DataType::FLOAT,
           hipdnn_data_sdk::data_objects::DataType::HALF,
           hipdnn_data_sdk::data_objects::DataType::BFLOAT16};
    if(std::find(validDataTypes.begin(), validDataTypes.end(), aType) == validDataTypes.end()
       || std::find(validDataTypes.begin(), validDataTypes.end(), bType) == validDataTypes.end()
       || std::find(validDataTypes.begin(), validDataTypes.end(), cType) == validDataTypes.end())
    {
        HIPDNN_LOG_INFO("Matmul plan builder: only fp32, fp16 and bf16 are supported");
        return false;
    }

    return true;
}

void validateGraphConfiguration(const hipdnn_plugin_sdk::IGraph& opGraph)
{
    if(opGraph.nodeCount() != 1)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Matmul plan builder supports only single node graphs. Graph has "
                + std::to_string(opGraph.nodeCount()) + " nodes");
    }

    if(opGraph.getNode(0).attributes_type()
       != hipdnn_data_sdk::data_objects::NodeAttributes::MatmulAttributes)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Matmul plan builder supports only MatmulAttributes nodes");
    }

    if(opGraph.getNode(0).compute_data_type() != hipdnn_data_sdk::data_objects::DataType::FLOAT)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Matmul plan builder only supports nodes with an fp32 compute_data_type");
    }
}

} // namespace

bool HipblasltMatmulPlanBuilder::isApplicable(const HipdnnEnginePluginHandle& handle,
                                              const hipdnn_plugin_sdk::IGraph& opGraph) const
{
    try
    {
        validateGraphConfiguration(opGraph);

        const auto& attr = opGraph.getNodeWrapper(0)
                               .attributesAs<hipdnn_data_sdk::data_objects::MatmulAttributes>();

        const auto& tensorMap = opGraph.getTensorMap();
        auto aAttr = hipblaslt_utils::findTensorAttributes(tensorMap, attr.a_tensor_uid());
        auto bAttr = hipblaslt_utils::findTensorAttributes(tensorMap, attr.b_tensor_uid());
        auto cAttr = hipblaslt_utils::findTensorAttributes(tensorMap, attr.c_tensor_uid());
        if(!checkDataTypes(aAttr, bAttr, cAttr))
        {
            return false;
        }

        MatmulParams params(attr, opGraph.getTensorMap());
        MatmulPlan plan(handle, std::move(params));
        return true;
    }
    catch(const std::exception& e)
    {
        HIPDNN_LOG_INFO(e.what());
        return false;
    }
}

size_t HipblasltMatmulPlanBuilder::getWorkspaceSize(const HipdnnEnginePluginHandle& handle,
                                                    const hipdnn_plugin_sdk::IGraph& opGraph) const
{
    validateGraphConfiguration(opGraph);

    const auto& attr
        = opGraph.getNodeWrapper(0).attributesAs<hipdnn_data_sdk::data_objects::MatmulAttributes>();
    MatmulParams params(attr, opGraph.getTensorMap());
    MatmulPlan plan(handle, std::move(params));
    return plan.getWorkspaceSize(handle);
}

void HipblasltMatmulPlanBuilder::buildPlan(
    const HipdnnEnginePluginHandle& handle,
    const hipdnn_plugin_sdk::IGraph& opGraph,
    HipdnnEnginePluginExecutionContext& executionContext) const
{
    validateGraphConfiguration(opGraph);

    const auto& nodeWrapper = opGraph.getNodeWrapper(0);
    HIPDNN_LOG_INFO("Building matmul plan for node: {}", nodeWrapper.name());

    const auto& attr = nodeWrapper.attributesAs<hipdnn_data_sdk::data_objects::MatmulAttributes>();
    MatmulParams params(attr, opGraph.getTensorMap());
    auto plan = std::make_unique<MatmulPlan>(handle, std::move(params));
    executionContext.setPlan(std::move(plan));
}

} // namespace hipblaslt_plugin
