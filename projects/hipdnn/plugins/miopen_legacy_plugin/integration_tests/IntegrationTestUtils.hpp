// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/Utilities.hpp>
#include <hipdnn_frontend/node/Node.hpp>
#include <hipdnn_sdk/plugin/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_sdk/test_utilities/ReferenceValidationInterface.hpp>
#include <hipdnn_sdk/test_utilities/cpu_graph_executor/CpuReferenceGraphExecutor.hpp>
#include <hipdnn_sdk/test_utilities/cpu_graph_executor/GraphTensorBundle.hpp>
#include <hipdnn_sdk/utilities/Workspace.hpp>

namespace hipdnn_sdk::test_utilities
{

// NOLINTBEGIN (portability-template-virtual-member-function)
template <typename DataType, typename TestCaseType>
class GraphVerifierTest : public ::testing::TestWithParam<TestCaseType>
{
protected:
    void SetUp() override
    {
        SKIP_IF_NO_DEVICES();

        // Initialize HIP
        ASSERT_EQ(hipInit(0), hipSuccess);
        ASSERT_EQ(hipGetDevice(&_deviceId), hipSuccess);

        // Note: The plugin paths has to be set before we create the hipdnn handle.
        auto pluginPath
            = std::filesystem::weakly_canonical(getCurrentExecutableDirectory() / PLUGIN_PATH);
        const std::string pluginPathStr = pluginPath.string();
        const std::array<const char*, 1> paths = {pluginPathStr.c_str()};
        ASSERT_EQ(hipdnnSetEnginePluginPaths_ext(
                      paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE),
                  HIPDNN_STATUS_SUCCESS);

        // Create handle and stream
        ASSERT_EQ(hipdnnCreate(&_handle), HIPDNN_STATUS_SUCCESS);
        ASSERT_EQ(hipStreamCreate(&_stream), hipSuccess);
        ASSERT_EQ(hipdnnSetStream(_handle, _stream), HIPDNN_STATUS_SUCCESS);
    }

    void TearDown() override
    {
        if(_handle != nullptr)
        {
            ASSERT_EQ(hipdnnDestroy(_handle), HIPDNN_STATUS_SUCCESS);
        }
        if(_stream != nullptr)
        {
            ASSERT_EQ(hipStreamDestroy(_stream), hipSuccess);
        }
    }

    virtual void runGraphTest(DataType tolerance, const TensorLayout& layout = TensorLayout::NCHW)
        = 0;

protected:
    void verifyGraph(hipdnn_frontend::graph::Graph& graph,
                     unsigned int seed,
                     const IReferenceValidation& validator)
    {
        GraphTensorBundle gpuBundle = generateBundle(graph);
        GraphTensorBundle cpuBundle = generateBundle(graph);

        verifyGraph(graph, seed, cpuBundle, gpuBundle, validator);
    }

    void verifyGraph(hipdnn_frontend::graph::Graph& graph,
                     unsigned int seed,
                     GraphTensorBundle& cpuBundle,
                     GraphTensorBundle& gpuBundle,
                     const IReferenceValidation& validator)
    {
        initializeBundle(graph, gpuBundle, seed);
        initializeBundle(graph, cpuBundle, seed);

        auto result = graph.validate();
        ASSERT_EQ(result.code, hipdnn_frontend::ErrorCode::OK) << result.err_msg;

        executeGpuGraph(_handle, graph, gpuBundle);
        executeCpuGraph(graph, cpuBundle);

        for(const auto& tensorId : getAllNonVirtualOutputTensorIds(graph))
        {
            auto& cpuTensor = cpuBundle.tensors.at(tensorId);
            auto& gpuTensor = gpuBundle.tensors.at(tensorId);

            // This tells the tensor that its data has been modified on the device side
            // All frontend graph knows is a (void*) pointer to device memory, so we need to inform the tensor
            // that the data there is now valid so that it knows to copy from device to host when requested
            // by the validation step.
            gpuTensor->markDeviceModified();

            bool valid = validator.allClose(*cpuTensor, *gpuTensor);
            ASSERT_TRUE(valid) << "Mismatch found in tensor with id: " << tensorId;
        }
    }

    virtual GraphTensorBundle generateBundle(hipdnn_frontend::graph::Graph& graph)
    {
        GraphTensorBundle bundle;

        visitGraph(graph, [&](const hipdnn_frontend::graph::INode& node) {
            for(const auto& tensorAttr : node.getNodeOutputTensorAttributes())
            {
                if(tensorAttr->get_is_virtual())
                    continue;

                const auto& tensorId = tensorAttr->get_uid();
                if(bundle.tensors.find(tensorId) == bundle.tensors.end())
                {
                    bundle.tensors.insert({tensorId, createTensorFromAttribute(*tensorAttr)});
                }
            }
            for(const auto& tensorAttr : node.getNodeInputTensorAttributes())
            {
                if(tensorAttr->get_is_virtual())
                    continue;

                const auto& tensorId = tensorAttr->get_uid();
                if(bundle.tensors.find(tensorId) == bundle.tensors.end())
                {
                    bundle.tensors.insert({tensorId, createTensorFromAttribute(*tensorAttr)});
                }
            }
        });

        return bundle;
    }

    virtual void initializeBundle([[maybe_unused]] const hipdnn_frontend::graph::Graph& graph,
                                  GraphTensorBundle& bundle,
                                  unsigned int seed)
    {
        for(auto& tensorPair : bundle.tensors)
        {
            bundle.randomizeTensor(tensorPair.first, -1.0f, 1.0f, seed);
        }
    }

private:
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

    std::vector<int64_t> getAllNonVirtualOutputTensorIds(hipdnn_frontend::graph::Graph& graph)
    {
        std::vector<int64_t> tensorIds;

        visitGraph(graph, [&](const hipdnn_frontend::graph::INode& node) {
            for(const auto& tensorAttr : node.getNodeOutputTensorAttributes())
            {
                if(tensorAttr->get_is_virtual())
                    continue;

                tensorIds.push_back(tensorAttr->get_uid());
            }
        });

        return tensorIds;
    }

    hipdnnHandle_t _handle = nullptr;
    hipStream_t _stream = nullptr;
    int _deviceId = 0;
};

// NOLINTEND (portability-template-virtual-member-function

} // namespace hipdnn_sdk::test_utilities
