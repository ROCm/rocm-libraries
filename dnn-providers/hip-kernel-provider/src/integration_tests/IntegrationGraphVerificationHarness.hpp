// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <gtest/gtest.h>
#include <hipdnn_data_sdk/utilities/Workspace.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/Utilities.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/SdkFrontendTypeConversions.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/CpuReferenceGraphExecutor.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/GraphTensorBundle.hpp>

#include <functional>

namespace hip_kernel_provider::test_utilities
{

// NOLINTBEGIN (portability-template-virtual-member-function)
template <typename DataType, typename TestCaseType>
class IntegrationGraphVerificationHarness : public ::testing::TestWithParam<TestCaseType>
{
protected:
    static constexpr float DEFAULT_MIN = -1.0f;
    static constexpr float DEFAULT_MAX = 1.0f;

    void SetUp() override
    {
        SKIP_IF_NO_DEVICES();

        ASSERT_EQ(hipInit(0), hipSuccess);
        ASSERT_EQ(hipGetDevice(&_deviceId), hipSuccess);

        auto pluginPath = std::filesystem::weakly_canonical(
            hipdnn_data_sdk::utilities::getCurrentExecutableDirectory() / PLUGIN_PATH);
        const std::string pluginPathStr = pluginPath.string();
        const std::array<const char*, 1> paths = {pluginPathStr.c_str()};
        ASSERT_EQ(hipdnnSetEnginePluginPaths_ext(
                      paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE),
                  HIPDNN_STATUS_SUCCESS);

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

protected:
    void verifyGraph(hipdnn_frontend::graph::Graph& graph, unsigned int seed)
    {
        hipdnn_test_sdk::utilities::GraphTensorBundle gpuBundle;
        hipdnn_test_sdk::utilities::GraphTensorBundle cpuBundle;
        std::vector<int64_t> outputTensorIds;

        auto result = graph.build(_handle);
        ASSERT_EQ(result.code, hipdnn_frontend::ErrorCode::OK) << result.err_msg;

        generateBundles(graph, cpuBundle, gpuBundle, outputTensorIds);

        initializeBundle(graph, gpuBundle, seed);
        initializeBundle(graph, cpuBundle, seed);

        ASSERT_NO_FATAL_FAILURE(executeGpuGraph(_handle, graph, gpuBundle));
        executeCpuGraph(graph, cpuBundle);

        ASSERT_GE(outputTensorIds.size(), 1)
            << "At least one output tensor id must be specified for validation.";

        HIPDNN_PLUGIN_LOG_INFO("Validating " << outputTensorIds.size() << " output tensors");

        for(const auto& registerValidator : _deferredValidators)
        {
            registerValidator();
        }

        for(const auto& tensorId : outputTensorIds)
        {
            auto& cpuTensor = cpuBundle.tensors.at(tensorId);
            auto& gpuTensor = gpuBundle.tensors.at(tensorId);

            gpuTensor->markDeviceModified();

            if(_tensorIdToValidatorMap.find(tensorId) == _tensorIdToValidatorMap.end())
            {
                FAIL() << "No validator registered for tensor with id: " << tensorId
                       << ", name: " << _tensorIdToNameMap.at(tensorId);
            }

            bool valid = _tensorIdToValidatorMap.at(tensorId)->allClose(*cpuTensor, *gpuTensor);
            ASSERT_TRUE(valid) << "Mismatch found in tensor with id: " << tensorId
                               << ", name: " << _tensorIdToNameMap.at(tensorId);
        }
    }

    void registerValidator(const std::shared_ptr<hipdnn_frontend::graph::TensorAttributes> attr,
                           float tolerance)
    {
        registerValidator(attr, tolerance, tolerance);
    }

    void registerValidator(const std::shared_ptr<hipdnn_frontend::graph::TensorAttributes> attr,
                           float absoluteTolerance,
                           float relativeTolerance)
    {
        _deferredValidators.emplace_back([=]() {
            _tensorIdToValidatorMap.insert(
                {attr->get_uid(),
                 hipdnn_test_sdk::utilities::createAllCloseValidator(
                     hipdnn_test_sdk::utilities::frontendToSdkDataType(attr->get_data_type()),
                     absoluteTolerance,
                     relativeTolerance)});
            _tensorIdToNameMap.insert({attr->get_uid(), attr->get_name()});
        });
    }

    virtual void generateBundles(hipdnn_frontend::graph::Graph& graph,
                                 hipdnn_test_sdk::utilities::GraphTensorBundle& cpuBundle,
                                 hipdnn_test_sdk::utilities::GraphTensorBundle& gpuBundle,
                                 std::vector<int64_t>& outputTensorIds)
    {
        graph.visit([&](const hipdnn_frontend::graph::INode& node) {
            for(const auto& tensorAttr : node.getNodeOutputTensorAttributes())
            {
                if(tryAddTensorToBundles(tensorAttr, cpuBundle, gpuBundle))
                {
                    outputTensorIds.push_back(tensorAttr->get_uid());
                }
            }
            for(const auto& tensorAttr : node.getNodeInputTensorAttributes())
            {
                tryAddTensorToBundles(tensorAttr, cpuBundle, gpuBundle);
            }
        });
    }

    virtual void initializeBundle([[maybe_unused]] const hipdnn_frontend::graph::Graph& graph,
                                  hipdnn_test_sdk::utilities::GraphTensorBundle& bundle,
                                  unsigned int seed)
    {
        for(auto& tensorPair : bundle.tensors)
        {
            bundle.randomizeTensor(tensorPair.first, DEFAULT_MIN, DEFAULT_MAX, seed);
        }
    }

    virtual hipStream_t stream() const
    {
        return _stream;
    }

private:
    void executeGpuGraph(hipdnnHandle_t handle,
                         hipdnn_frontend::graph::Graph& graph,
                         hipdnn_test_sdk::utilities::GraphTensorBundle& bundle)
    {
        int64_t workspaceSize;
        auto result = graph.get_workspace_size(workspaceSize);
        ASSERT_EQ(result.code, hipdnn_frontend::ErrorCode::OK) << result.err_msg;
        ASSERT_GE(workspaceSize, 0) << result.err_msg;
        hipdnn_data_sdk::utilities::Workspace workspace(static_cast<size_t>(workspaceSize));

        auto variantPack = bundle.toDeviceVariantPack();
        result = graph.execute(handle, variantPack, workspace.get());
        ASSERT_EQ(result.code, hipdnn_frontend::ErrorCode::OK) << result.err_msg;
    }

    void executeCpuGraph(hipdnn_frontend::graph::Graph& graph,
                         hipdnn_test_sdk::utilities::GraphTensorBundle& bundle)
    {
        auto [serializedGraph, serErr] = graph.to_binary();
        ASSERT_TRUE(serErr.is_good()) << serErr.get_message();

        hipdnn_test_sdk::utilities::CpuReferenceGraphExecutor().execute(
            serializedGraph.data(), serializedGraph.size(), bundle.toHostVariantPack());
    }

    bool tryAddTensorToBundles(
        const std::shared_ptr<hipdnn_frontend::graph::TensorAttributes>& tensorAttr,
        hipdnn_test_sdk::utilities::GraphTensorBundle& cpuBundle,
        hipdnn_test_sdk::utilities::GraphTensorBundle& gpuBundle)
    {
        int64_t tensorId = tensorAttr->get_uid();

        if(tensorAttr->get_is_virtual()
           || cpuBundle.tensors.find(tensorId) != cpuBundle.tensors.end())
        {
            return false;
        }

        cpuBundle.tensors.insert(
            {tensorId, hipdnn_test_sdk::utilities::createTensorFromAttribute(*tensorAttr)});
        gpuBundle.tensors.insert(
            {tensorId, hipdnn_test_sdk::utilities::createTensorFromAttribute(*tensorAttr)});
        _tensorIdToNameMap.insert({tensorId, tensorAttr->get_name()});

        return true;
    }

    hipdnnHandle_t _handle = nullptr;
    hipStream_t _stream = nullptr;
    int _deviceId = 0;
    std::unordered_map<int64_t, std::string> _tensorIdToNameMap;
    std::unordered_map<int64_t, std::unique_ptr<hipdnn_test_sdk::utilities::IReferenceValidation>>
        _tensorIdToValidatorMap;
    std::vector<std::function<void()>> _deferredValidators;
};

// NOLINTEND (portability-template-virtual-member-function)

} // namespace hip_kernel_provider::test_utilities
