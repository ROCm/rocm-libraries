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

    static void
        execute(void* graphBuffer, size_t size, std::unordered_map<int64_t, void*>& variantPack)
    {
        auto graphWrap = hipdnn_plugin::GraphWrapper(graphBuffer, size);

        for(uint32_t i = 0; i < graphWrap.nodeCount(); i++)
        {
            //todo
            //build up a list of planExecutors for each node in the graph
            // if all suceed to create, run each node in sequence

            auto& node = graphWrap.getNode(i);
            auto planExecutor = buildPlanForNode(graphWrap, node);
            planExecutor->execute(variantPack);
        }
    }

    static std::unique_ptr<IGraphNodePlanExecutor>
        buildPlanForNode(const hipdnn_plugin::IGraph& graph,
                         const hipdnn_sdk::data_objects::Node& node)
    {
        auto key = buildSignatureKey(node, graph.getTensorMap());

        auto it = planBuilderRegistry().find(key);
        if(it == planBuilderRegistry().end())
        {
            throw std::runtime_error("No plan builder found for given node signature");
        }

        auto planBuilder = it->second.get();
        if(!planBuilder->isApplicable(node, graph.getTensorMap()))
        {
            throw std::runtime_error("Plan builder is not applicable for the given node");
        }

        return planBuilder->buildNodePlan(graph, node);
    }

    static Key buildSignatureKey(
        const hipdnn_sdk::data_objects::Node& node,
        const std::unordered_map<int64_t, const hipdnn_sdk::data_objects::TensorAttributes*>&
            tensorMap)
    {
        //todo switch on the node type .....
        const auto* nodeAttributes = node.attributes_as_BatchnormInferenceAttributes();

        auto xTensorAttr = tensorMap.at(nodeAttributes->x_tensor_uid());
        auto scaleTensorAttr = tensorMap.at(nodeAttributes->scale_tensor_uid());
        auto meanTensorAttr = tensorMap.at(nodeAttributes->mean_tensor_uid());

        return BatchnormSignatureRegistryKey(
            xTensorAttr->data_type(), scaleTensorAttr->data_type(), meanTensorAttr->data_type());
    }
};
}
}
