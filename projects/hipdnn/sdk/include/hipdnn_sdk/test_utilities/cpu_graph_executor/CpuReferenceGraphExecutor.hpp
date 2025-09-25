// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hipdnn_sdk/plugin/EnginePluginApi.h>
#include <hipdnn_sdk/plugin/PluginApiDataTypes.h>
#include <hipdnn_sdk/plugin/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_sdk/utilities/ShallowTensor.hpp>
#include <hipdnn_sdk/utilities/Tensor.hpp>
#include <hipdnn_sdk/utilities/UtilsBfp16.hpp>
#include <hipdnn_sdk/utilities/UtilsFp16.hpp>

#include <hipdnn_sdk/test_utilities/cpu_graph_executor/BatchnormFwdInferencePlan.hpp>
#include <hipdnn_sdk/test_utilities/cpu_graph_executor/PlanBuilderRegistry.hpp>

namespace hipdnn_sdk
{
namespace test_utilities
{

class CpuReferenceGraphExecutor
{
public:
    CpuReferenceGraphExecutor() = default;
    ~CpuReferenceGraphExecutor() = default;

    void execute(void* graphBuffer,
                 size_t size,
                 const std::unordered_map<int64_t, void*>& variantPack)
    {
        auto graphWrap = hipdnn_plugin::GraphWrapper(graphBuffer, size);

        std::vector<std::unique_ptr<IGraphNodePlanExecutor>> planExecutors;

        for(uint32_t i = 0; i < graphWrap.nodeCount(); i++)
        {

            auto& node = graphWrap.getNode(i);
            planExecutors.push_back(buildPlanForNode(graphWrap, node));
        }

        //todo future, look through the graphs Tensor map and look for virtual tensors.
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

        auto planBuilder = _planRegistry.getPlanBuilder(key);
        if(planBuilder == nullptr)
        {
            throw std::runtime_error("No plan builder found for given node signature");
        }

        if(!planBuilder->isApplicable(node, graph.getTensorMap()))
        {
            throw std::runtime_error("Plan builder is not applicable for the given node");
        }

        return planBuilder->buildNodePlan(graph, node);
    }

    Key buildSignatureKey(
        const hipdnn_sdk::data_objects::Node& node,
        const std::unordered_map<int64_t, const hipdnn_sdk::data_objects::TensorAttributes*>&
            tensorMap)
    {
        //todo switch on the node type .....
        const auto* nodeAttributes = node.attributes_as_BatchnormInferenceAttributes();

        auto xTensorAttr = tensorMap.at(nodeAttributes->x_tensor_uid());
        auto scaleTensorAttr = tensorMap.at(nodeAttributes->scale_tensor_uid());
        auto meanTensorAttr = tensorMap.at(nodeAttributes->mean_tensor_uid());

        return BatchnormFwdInferenceSignatureKey(
            xTensorAttr->data_type(), scaleTensorAttr->data_type(), meanTensorAttr->data_type());
    }

    PlanBuilderRegistry _planRegistry;
};

}
}
