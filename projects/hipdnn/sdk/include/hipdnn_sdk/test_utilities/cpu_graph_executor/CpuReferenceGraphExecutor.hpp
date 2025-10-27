// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_sdk/plugin/flatbuffer_utilities/GraphWrapper.hpp>

#include <hipdnn_sdk/test_utilities/cpu_graph_executor/BatchnormFwdInferencePlan.hpp>
#include <hipdnn_sdk/test_utilities/cpu_graph_executor/ConvolutionBwdPlan.hpp>
#include <hipdnn_sdk/test_utilities/cpu_graph_executor/ConvolutionFwdPlan.hpp>
#include <hipdnn_sdk/test_utilities/cpu_graph_executor/PlanBuilderRegistry.hpp>
#include <hipdnn_sdk/test_utilities/cpu_graph_executor/PointwisePlan.hpp>

namespace hipdnn_sdk::test_utilities
{

class CpuReferenceGraphExecutor
{

public:
    CpuReferenceGraphExecutor() = default;

    void execute(void* graphBuffer,
                 size_t size,
                 const std::unordered_map<int64_t, void*>& variantPack)
    {
        auto graphWrap = hipdnn_plugin::GraphWrapper(graphBuffer, size);

        std::vector<std::unique_ptr<IGraphNodePlanExecutor>> planExecutors;

        //The graph in graphBuffer is guaranteed to be topologically sorted.
        for(uint32_t i = 0; i < graphWrap.nodeCount(); i++)
        {
            auto& node = graphWrap.getNode(i);
            planExecutors.push_back(buildPlanForNode(graphWrap, node));
        }

        std::vector<std::unique_ptr<ITensor>> virtualTensors;
        std::unordered_map<int64_t, void*> variantPackWithVirtualTensorsAdded
            = populateVariantPackWithMissingVirtualTensors(
                variantPack, graphWrap.getTensorMap(), virtualTensors);

        for(auto& executor : planExecutors)
        {
            executor->execute(variantPackWithVirtualTensorsAdded);
        }
    }

private:
    static std::unordered_map<int64_t, void*> populateVariantPackWithMissingVirtualTensors(
        const std::unordered_map<int64_t, void*>& variantPack,
        const std::unordered_map<int64_t, const hipdnn_sdk::data_objects::TensorAttributes*>&
            tensorMap,
        std::vector<std::unique_ptr<ITensor>>& virtualTensors)
    {
        std::unordered_map<int64_t, void*> updatedVariantPack = variantPack;

        for(const auto& [id, attr] : tensorMap)
        {
            if(attr->virtual_() && updatedVariantPack.find(id) == updatedVariantPack.end())
            {
                auto tensor = createTensorFromAttribute(*attr);
                virtualTensors.push_back(std::move(tensor));
                updatedVariantPack[id] = virtualTensors.back()->rawHostData();
            }
        }
        return updatedVariantPack;
    }

    std::unique_ptr<IGraphNodePlanExecutor>
        buildPlanForNode(const hipdnn_plugin::IGraph& graph,
                         const hipdnn_sdk::data_objects::Node& node)
    {
        // TODO: Switch this to the node's compute_type
        auto key = buildSignatureKey(node, graph.getTensorMap(), graph.getGraph().compute_type());

        const auto& planBuilder = _planRegistry.getPlanBuilder(key);
        if(!planBuilder.isApplicable(node, graph.getTensorMap()))
        {
            throw std::runtime_error("Plan builder is not applicable for the given node");
        }

        return planBuilder.buildNodePlan(graph, node);
    }

    static PlanRegistrySignatureKey buildSignatureKey(
        const hipdnn_sdk::data_objects::Node& node,
        const std::unordered_map<int64_t, const hipdnn_sdk::data_objects::TensorAttributes*>&
            tensorMap,
        const hipdnn_sdk::data_objects::DataType computeType)
    {
        switch(node.attributes_type())
        {
        case hipdnn_sdk::data_objects::NodeAttributes::BatchnormInferenceAttributes:
            return BatchnormFwdInferenceSignatureKey(node, tensorMap);
        case hipdnn_sdk::data_objects::NodeAttributes::PointwiseAttributes:
            return PointwiseSignatureKey(node, tensorMap);
        case hipdnn_sdk::data_objects::NodeAttributes::BatchnormBackwardAttributes:
            return BatchnormBwdSignatureKey(node, tensorMap);
        case hipdnn_sdk::data_objects::NodeAttributes::BatchnormAttributes:
            return BatchnormTrainSignatureKey(node, tensorMap);
        case hipdnn_sdk::data_objects::NodeAttributes::ConvolutionFwdAttributes:
            return ConvolutionFwdSignatureKey(node, tensorMap, computeType);
        case hipdnn_sdk::data_objects::NodeAttributes::ConvolutionBwdAttributes:
            return ConvolutionBwdSignatureKey(node, tensorMap, computeType);
        default:
            throw std::runtime_error("Unsupported node type for signature key generation");
        }
    }

    PlanBuilderRegistry _planRegistry;
};
}
