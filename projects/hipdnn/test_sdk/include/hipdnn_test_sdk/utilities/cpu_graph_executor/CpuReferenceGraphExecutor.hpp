// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstring>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>

#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/BatchnormFwdInferencePlan.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/BatchnormFwdInferenceWithVarianceSignatureKey.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/BlockScaleDequantizeSignatureKey.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/ConvolutionBwdPlan.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/ConvolutionFwdPlan.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/ConvolutionWrwPlan.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/LayernormFpropSignatureKey.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/MatmulPlan.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/PlanBuilderRegistry.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/PointwisePlan.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/RMSNormFwdSignatureKey.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/ReductionPlan.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/SdpaBwdSignatureKey.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/SdpaFwdSignatureKey.hpp>

namespace hipdnn_test_sdk::utilities
{

class CpuReferenceGraphExecutor
{

public:
    CpuReferenceGraphExecutor() = default;

    // NOLINTNEXTLINE(readability-convert-member-functions-to-static)
    void execute(void* graphBuffer,
                 size_t size,
                 const std::unordered_map<int64_t, void*>& variantPack)
    {
        auto graphWrap
            = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(graphBuffer, size);

        std::vector<std::unique_ptr<detail::IGraphNodePlanExecutor>> planExecutors;

        //The graph in graphBuffer is guaranteed to be topologically sorted.
        for(uint32_t i = 0; i < graphWrap.nodeCount(); i++)
        {
            auto& node = graphWrap.getNode(i);
            planExecutors.push_back(buildPlanForNode(graphWrap, node));
        }

        std::vector<std::unique_ptr<hipdnn_data_sdk::utilities::ITensor>> virtualTensors;
        const std::unordered_map<int64_t, void*> variantPackWithVirtualTensorsAdded
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
        const std::unordered_map<int64_t,
                                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
            tensorMap,
        std::vector<std::unique_ptr<hipdnn_data_sdk::utilities::ITensor>>& virtualTensors)
    {
        std::unordered_map<int64_t, void*> updatedVariantPack = variantPack;

        for(const auto& [id, attr] : tensorMap)
        {
            if(attr->virtual_() && updatedVariantPack.find(id) == updatedVariantPack.end())
            {
                auto tensor = detail::createTensorFromAttribute(*attr);
                tensor->fillWithSentinelValue();
                virtualTensors.push_back(std::move(tensor));
                updatedVariantPack[id] = virtualTensors.back()->rawHostData();
            }
        }
        return updatedVariantPack;
    }

    std::unique_ptr<detail::IGraphNodePlanExecutor>
        buildPlanForNode(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph,
                         const hipdnn_flatbuffers_sdk::data_objects::Node& node)
    {
        auto key = buildSignatureKey(node, graph.getTensorMap(), node.compute_data_type());

        const auto& planBuilder = _planRegistry.getPlanBuilder(key);
        if(!planBuilder.isApplicable(node, graph.getTensorMap()))
        {
            const std::string nodeName = node.name() == nullptr ? "" : " " + node.name()->str();
            throw std::runtime_error("Plan builder is not applicable for the given node: "
                                     + nodeName);
        }

        return planBuilder.buildNodePlan(graph, node);
    }

    static detail::PlanRegistrySignatureKey buildSignatureKey(
        const hipdnn_flatbuffers_sdk::data_objects::Node& node,
        const std::unordered_map<int64_t,
                                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
            tensorMap,
        const hipdnn_flatbuffers_sdk::data_objects::DataType computeType)
    {
        switch(node.attributes_type())
        {
        case hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::BatchnormInferenceAttributes:
            return detail::BatchnormFwdInferenceSignatureKey(node, tensorMap);
        case hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::
            BatchnormInferenceAttributesVarianceExt:
            return detail::BatchnormFwdInferenceWithVarianceSignatureKey(node, tensorMap);
        case hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::PointwiseAttributes:
            return detail::PointwiseSignatureKey(node, tensorMap);
        case hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::BatchnormBackwardAttributes:
            return detail::BatchnormBwdSignatureKey(node, tensorMap);
        case hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::BatchnormAttributes:
            return detail::BatchnormTrainSignatureKey(node, tensorMap);
        case hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::ConvolutionFwdAttributes:
            return detail::ConvolutionFwdSignatureKey(node, tensorMap, computeType);
        case hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::ConvolutionBwdAttributes:
            return detail::ConvolutionBwdSignatureKey(node, tensorMap, computeType);
        case hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::ConvolutionWrwAttributes:
            return detail::ConvolutionWrwSignatureKey(node, tensorMap, computeType);
        case hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::LayernormAttributes:
            return detail::LayernormFpropSignatureKey(node, tensorMap);
        case hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::MatmulAttributes:
            return detail::MatmulSignatureKey(node, tensorMap, computeType);
        case hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::RMSNormAttributes:
            return detail::RMSNormFwdSignatureKey(node, tensorMap);
        case hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::RMSNormBwdAttributes:
            return detail::RMSNormBwdSignatureKey(node, tensorMap);
        case hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::BlockScaleDequantizeAttributes:
            return detail::BlockScaleDequantizeSignatureKey(node, tensorMap);
        case hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::SdpaAttributes:
            return detail::SdpaFwdSignatureKey(node, tensorMap);
        case hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::SdpaBackwardAttributes:
            return detail::SdpaBwdSignatureKey(node, tensorMap);
        case hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::ReductionAttributes:
            return detail::ReductionSignatureKey(node, tensorMap, computeType);
        default:
            throw std::runtime_error("Unsupported node type for signature key generation");
        }
    }

    detail::PlanBuilderRegistry _planRegistry;
};
}
