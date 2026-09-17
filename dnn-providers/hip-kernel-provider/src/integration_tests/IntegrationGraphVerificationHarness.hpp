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
#include <limits>

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

    /// Builds fresh CPU/GPU tensor bundles for `graph`, executes once on device with an
    /// already-sized `workspace`, and compares against CpuReferenceGraphExecutor. Reseeds
    /// inputs from `seed` so repeated calls never compare stale buffers. `reductionLength`
    /// widens the tolerance for kernels that accumulate: GPU and CPU sum in different
    /// orders, so a K-term sum needs ~K*epsilon of slack where a pointwise op needs one.
    ///
    /// Unlike verifyGraph() this drives an already-built graph and validates the single
    /// output at uid 3, for suites that stage the build themselves.
    void executeAndVerify(hipdnn_frontend::graph::Graph& graph,
                          void* workspace,
                          unsigned int seed,
                          int reductionLength = 1)
    {
        hipdnn_test_sdk::utilities::GraphTensorBundle gpuBundle;
        hipdnn_test_sdk::utilities::GraphTensorBundle cpuBundle;
        graph.visit([&](const hipdnn_frontend::graph::INode& node) {
            for(const auto& tensorAttr : node.getNodeOutputTensorAttributes())
            {
                gpuBundle.addTensor(
                    *tensorAttr,
                    hipdnn_test_sdk::utilities::createTensorFromAttribute(*tensorAttr));
                cpuBundle.addTensor(
                    *tensorAttr,
                    hipdnn_test_sdk::utilities::createTensorFromAttribute(*tensorAttr));
            }
            for(const auto& tensorAttr : node.getNodeInputTensorAttributes())
            {
                if(gpuBundle.tensors.find(tensorAttr->get_uid()) == gpuBundle.tensors.end())
                {
                    gpuBundle.addTensor(
                        *tensorAttr,
                        hipdnn_test_sdk::utilities::createTensorFromAttribute(*tensorAttr));
                    cpuBundle.addTensor(
                        *tensorAttr,
                        hipdnn_test_sdk::utilities::createTensorFromAttribute(*tensorAttr));
                }
            }
        });

        for(auto& [uid, tensor] : gpuBundle.tensors)
        {
            // Per-uid offset so the operands are never byte-identical: a+a and a+b are
            // indistinguishable to allClose() when both operands hold the same bytes.
            const auto tensorSeed = seed + static_cast<unsigned int>(uid);
            gpuBundle.randomizeTensor(uid, -4.0f, 4.0f, tensorSeed);
            cpuBundle.randomizeTensor(uid, -4.0f, 4.0f, tensorSeed);
        }

        auto deviceVariantPack = gpuBundle.toDeviceVariantPack();
        auto result = graph.execute(_handle, deviceVariantPack, workspace);
        ASSERT_EQ(result.code, hipdnn_frontend::ErrorCode::OK) << result.err_msg;
        ASSERT_EQ(hipStreamSynchronize(_stream), hipSuccess);

        auto [serializedGraph, serErr] = graph.to_binary();
        ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
        hipdnn_test_sdk::utilities::CpuReferenceGraphExecutor().execute(
            serializedGraph.data(), serializedGraph.size(), cpuBundle.toHostVariantPack());

        auto& gpuOut = gpuBundle.getTensor(3);
        auto& cpuOut = cpuBundle.getTensor(3);
        gpuOut.markDeviceModified();
        const auto tolerance
            = static_cast<float>(reductionLength) * std::numeric_limits<float>::epsilon();
        EXPECT_TRUE(
            hipdnn_test_sdk::utilities::CpuFpReferenceValidation<float>(tolerance, tolerance)
                .allClose(cpuOut, gpuOut));
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

    // Exposed to subclasses driving Graph staging calls directly (build_operation_graph,
    // create_execution_plans, get_ranked_engine_ids, etc.), not only via verifyGraph().
    hipdnnHandle_t _handle = nullptr;
    hipStream_t _stream = nullptr;
    int _deviceId = 0;

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

        cpuBundle.addTensor(*tensorAttr,
                            hipdnn_test_sdk::utilities::createTensorFromAttribute(*tensorAttr));
        gpuBundle.addTensor(*tensorAttr,
                            hipdnn_test_sdk::utilities::createTensorFromAttribute(*tensorAttr));
        _tensorIdToNameMap.insert({tensorId, tensorAttr->get_name()});

        return true;
    }

    std::unordered_map<int64_t, std::string> _tensorIdToNameMap;
    std::unordered_map<int64_t, std::unique_ptr<hipdnn_test_sdk::utilities::IReferenceValidation>>
        _tensorIdToValidatorMap;
    std::vector<std::function<void()>> _deferredValidators;
};

// NOLINTEND (portability-template-virtual-member-function)

} // namespace hip_kernel_provider::test_utilities
