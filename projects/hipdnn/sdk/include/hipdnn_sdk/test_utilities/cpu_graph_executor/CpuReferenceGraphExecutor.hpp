// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hipdnn_sdk/plugin/flatbuffer_utilities/GraphWrapper.hpp>

#include <hipdnn_sdk/test_utilities/cpu_graph_executor/BatchnormFwdInferencePlan.hpp>
#include <hipdnn_sdk/test_utilities/cpu_graph_executor/PlanBuilderRegistry.hpp>

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

        // todo future, we need to build the DAG and process it to produce a topological sequential order to execute nodes.
        // this is currently incorrect but works for single node graphs.
        for(uint32_t i = 0; i < graphWrap.nodeCount(); i++)
        {

            auto& node = graphWrap.getNode(i);
            planExecutors.push_back(buildPlanForNode(graphWrap, node));
        }

        // todo future, look through the graphs Tensor map and look for virtual tensors.
        // for each virtual tensor, create a instace of MigratableMemory(or make a host only memory class).
        // Add each new memory instance to a copy of the variant pack.
        // its not worth doing this before we know we can handle the full graph as we dont want to alloc memory
        // we dont need.

        for(auto& executor : planExecutors)
        {
            executor->execute(variantPack);
        }
    }

private:
    std::unique_ptr<IGraphNodePlanExecutor>
        buildPlanForNode(const hipdnn_plugin::IGraph& graph,
                         const hipdnn_sdk::data_objects::Node& node)
    {
        auto key = buildSignatureKey(node, graph.getTensorMap());

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
            tensorMap)
    {
        switch(node.attributes_type())
        {
        case hipdnn_sdk::data_objects::NodeAttributes::BatchnormInferenceAttributes:
            return BatchnormFwdInferenceSignatureKey(node, tensorMap);
        case hipdnn_sdk::data_objects::NodeAttributes::PointwiseAttributes:
        case hipdnn_sdk::data_objects::NodeAttributes::BatchnormBackwardAttributes:
            return BatchnormBwdSignatureKey(node, tensorMap);
        case hipdnn_sdk::data_objects::NodeAttributes::BatchnormAttributes:
            return BatchnormTrainSignatureKey(node, tensorMap);
        case hipdnn_sdk::data_objects::NodeAttributes::ConvolutionFwdAttributes:
        default:
            throw std::runtime_error("Unsupported node type for signature key generation");
        }
    }

    PlanBuilderRegistry _planRegistry;
};

}
