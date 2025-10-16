// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_sdk/plugin/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_sdk/test_utilities/ReferenceValidationInterface.hpp>
#include <hipdnn_sdk/test_utilities/cpu_graph_executor/CpuReferenceGraphExecutor.hpp>
#include <hipdnn_sdk/test_utilities/cpu_graph_executor/GraphTensorBundle.hpp>
#include <hipdnn_sdk/utilities/Workspace.hpp>

namespace hipdnn_sdk::test_utilities
{

typedef std::unordered_map<int64_t, const hipdnn_sdk::data_objects::TensorAttributes*> TensorMap;

void executeGpuGraph(hipdnnHandle_t handle,
                     hipdnn_frontend::graph::Graph& graph,
                     GraphTensorBundle& bundle)
{

    auto result = graph.build_operation_graph(handle);
    ASSERT_EQ(result.code, hipdnn_frontend::ErrorCode::OK) << result.err_msg;

    result = graph.create_execution_plans();
    ASSERT_EQ(result.code, hipdnn_frontend::ErrorCode::OK) << result.err_msg;

    result = graph.check_support();
    ASSERT_EQ(result.code, hipdnn_frontend::ErrorCode::OK) << result.err_msg;

    result = graph.build_plans();
    ASSERT_EQ(result.code, hipdnn_frontend::ErrorCode::OK) << result.err_msg;

    int64_t workspaceSize;
    result = graph.get_workspace_size(workspaceSize);
    ASSERT_EQ(result.code, hipdnn_frontend::ErrorCode::OK) << result.err_msg;
    ASSERT_GE(workspaceSize, 0) << result.err_msg;
    Workspace workspace(static_cast<size_t>(workspaceSize));

    auto variantPack = bundle.toDeviceVariantPack();
    result = graph.execute(handle, variantPack, workspace.get());
    ASSERT_EQ(result.code, hipdnn_frontend::ErrorCode::OK) << result.err_msg;
}

void executeCpuGraph(hipdnn_frontend::graph::Graph& graph, GraphTensorBundle& bundle)
{
    auto flatbufferGraph = graph.buildFlatbufferOperationGraph();

    hipdnn_sdk::test_utilities::CpuReferenceGraphExecutor().execute(
        flatbufferGraph.data(), flatbufferGraph.size(), bundle.toHostVariantPack());
}

void defaultInitializer(GraphTensorBundle& bundle, unsigned int seed)
{
    for(auto& tensorPair : bundle.tensors)
    {
        bundle.randomizeTensor(tensorPair.first, -1.0f, 1.0f, seed);
    }
}

template <typename DataType>
void verifyGraph(hipdnnHandle_t handle,
                 hipdnn_frontend::graph::Graph& graph,
                 unsigned int seed,
                 const IReferenceValidation& validator,
                 const std::vector<int64_t>& tensorsToValidate,
                 const std::function<void(GraphTensorBundle&, unsigned int)>& initializer
                 = defaultInitializer)
{
    auto flatbufferGraph = graph.buildFlatbufferOperationGraph();
    hipdnn_plugin::GraphWrapper graphWrapper(flatbufferGraph.data(), flatbufferGraph.size());
    const auto& tensorMap = graphWrapper.getTensorMap();

    GraphTensorBundle gpuBundle(tensorMap);
    GraphTensorBundle cpuBundle(tensorMap);

    initializer(gpuBundle, seed);
    initializer(cpuBundle, seed);

    auto result = graph.validate();
    ASSERT_EQ(result.code, hipdnn_frontend::ErrorCode::OK) << result.err_msg;

    executeGpuGraph(handle, graph, gpuBundle);
    executeCpuGraph(graph, cpuBundle);

    for(const auto& tensorId : tensorsToValidate)
    {
        auto& cpuTensor = cpuBundle.tensors.at(tensorId);
        auto& gpuTensor = gpuBundle.tensors.at(tensorId);

        gpuTensor->markDeviceModified();

        bool valid = validator.allClose(*cpuTensor, *gpuTensor);
        ASSERT_TRUE(valid) << "Mismatch found in tensor with id: " << tensorId;
    }
}

} // namespace hipdnn_sdk::test_utilities
