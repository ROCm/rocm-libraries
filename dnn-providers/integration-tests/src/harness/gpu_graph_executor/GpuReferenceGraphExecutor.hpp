// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/detail/FlatbufferTensorAttributesUtils.hpp>

#include "detail/GpuPlanBuilderRegistry.hpp"
#include "harness/IReferenceGraphExecutor.hpp"

namespace hipdnn_integration_tests::gpu_graph_executor
{

class GpuReferenceGraphExecutor : public IReferenceGraphExecutor
{
public:
    GpuReferenceGraphExecutor() = default;

    void execute(void* graphBuffer,
                 size_t size,
                 const std::unordered_map<int64_t, void*>& variantPack) override
    {
        auto graphWrap
            = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(graphBuffer, size);

        std::vector<std::unique_ptr<detail::IGpuGraphNodePlanExecutor>> planExecutors;

        // The graph in graphBuffer is guaranteed to be topologically sorted.
        for(uint32_t i = 0; i < graphWrap.nodeCount(); i++)
        {
            auto& node = graphWrap.getNode(i);
            planExecutors.push_back(buildPlanForNode(graphWrap, node));
        }

        std::vector<std::unique_ptr<hipdnn_data_sdk::utilities::ITensor>> virtualTensors;
        const auto variantPackWithVirtualTensors = populateVariantPackWithMissingVirtualTensors(
            variantPack, graphWrap.getTensorMap(), virtualTensors);

        for(auto& executor : planExecutors)
        {
            executor->execute(variantPackWithVirtualTensors);
        }
    }

    bool requiresDeviceMemory() const override
    {
        return true;
    }

private:
    static std::unordered_map<int64_t, void*> populateVariantPackWithMissingVirtualTensors(
        const std::unordered_map<int64_t, void*>& variantPack,
        const std::unordered_map<int64_t,
                                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
            tensorMap,
        std::vector<std::unique_ptr<hipdnn_data_sdk::utilities::ITensor>>& virtualTensors)
    {
        auto updatedVariantPack = variantPack;

        for(const auto& [id, attr] : tensorMap)
        {
            if(attr->virtual_() && updatedVariantPack.find(id) == updatedVariantPack.end())
            {
                auto tensor = hipdnn_test_sdk::detail::createTensorFromAttribute(*attr);
                virtualTensors.push_back(std::move(tensor));

                updatedVariantPack[id] = virtualTensors.back()->rawDeviceData();
            }
        }
        return updatedVariantPack;
    }

    std::unique_ptr<detail::IGpuGraphNodePlanExecutor>
        buildPlanForNode(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph,
                         const hipdnn_flatbuffers_sdk::data_objects::Node& node)
    {
        auto key = buildSignatureKey(node, graph.getTensorMap());

        const auto& planBuilder = _planRegistry.getPlanBuilder(key);
        if(!planBuilder.isApplicable(node, graph.getTensorMap()))
        {
            const std::string nodeName
                = node.name() == nullptr ? " unknown" : " " + node.name()->str();
            throw std::runtime_error("GPU plan builder is not applicable for the given node:"
                                     + nodeName);
        }

        return planBuilder.buildNodePlan(graph, node);
    }

    static detail::GpuPlanRegistrySignatureKey buildSignatureKey(
        const hipdnn_flatbuffers_sdk::data_objects::Node& node,
        const std::unordered_map<int64_t,
                                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
            tensorMap)
    {
        using NodeAttrs = hipdnn_flatbuffers_sdk::data_objects::NodeAttributes;

        switch(node.attributes_type())
        {
        case NodeAttrs::PointwiseAttributes:
            return detail::GpuPointwiseDummySignatureKey(node, tensorMap);

        case NodeAttrs::ConvolutionFwdAttributes:
            return detail::GpuConvolutionFwdSignatureKey(node, tensorMap, node.compute_data_type());

        // Node types with no GPU plan yet - throw descriptive error
        case NodeAttrs::BatchnormInferenceAttributes:
        case NodeAttrs::BatchnormInferenceAttributesVarianceExt:
        case NodeAttrs::BatchnormBackwardAttributes:
        case NodeAttrs::BatchnormAttributes:
        case NodeAttrs::ConvolutionBwdAttributes:
        case NodeAttrs::ConvolutionWrwAttributes:
        case NodeAttrs::MatmulAttributes:
        case NodeAttrs::LayernormAttributes:
        case NodeAttrs::RMSNormAttributes:
        case NodeAttrs::RMSNormBackwardAttributes:
        case NodeAttrs::SdpaAttributes:
        case NodeAttrs::SdpaBackwardAttributes:
        case NodeAttrs::BlockScaleDequantizeAttributes:
        case NodeAttrs::BlockScaleQuantizeAttributes:
        {
            const std::string nodeName = node.name() == nullptr ? "unknown" : node.name()->str();
            throw std::runtime_error("GPU plan not yet implemented for node '" + nodeName
                                     + "'. Register a GPU plan for this operation type.");
        }

        case NodeAttrs::CustomOpAttributes:
            throw std::runtime_error("GPU reference executor does not support custom operations");

        default:
            throw std::runtime_error("Unsupported node type for GPU signature key generation");
        }
    }

    detail::GpuPlanBuilderRegistry _planRegistry;
};

} // namespace hipdnn_integration_tests::gpu_graph_executor
