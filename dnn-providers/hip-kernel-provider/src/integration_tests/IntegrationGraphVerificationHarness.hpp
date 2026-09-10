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
    /// Builds @p graph, runs it on device and on the CPU reference, and validates every
    /// output against the validator registered for that tensor.
    void verifyGraph(hipdnn_frontend::graph::Graph& graph, unsigned int seed)
    {
        auto result = graph.build(_handle);
        ASSERT_EQ(result.code, hipdnn_frontend::ErrorCode::OK) << result.err_msg;

        ASSERT_NO_FATAL_FAILURE(verifyBuiltGraph(graph, seed));
    }

    /// verifyGraph() for a caller that has already built its own plans.
    ///
    /// Suites that pin an engine, set knobs or drive create_execution_plan_ext() have to
    /// stage the build themselves, so they cannot use verifyGraph()'s build step -- but
    /// everything after it is the same, and duplicating it is how a second copy came to
    /// discover outputs by a hardcoded uid, materialise virtual tensors and compare
    /// every dtype as float.
    void verifyBuiltGraph(hipdnn_frontend::graph::Graph& graph, unsigned int seed)
    {
        hipdnn_test_sdk::utilities::GraphTensorBundle gpuBundle;
        hipdnn_test_sdk::utilities::GraphTensorBundle cpuBundle;
        std::vector<int64_t> outputTensorIds;

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
        // Drained, not accumulated. A single test body may verify several graphs, and a
        // queue that kept every closure would re-run registrations for graphs that are
        // already gone.
        _deferredValidators.clear();

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

    /// Registers one validator per non-virtual output of @p graph, each built from that
    /// tensor's own data type.
    ///
    /// @p epsilonMultiple is expressed in epsilons of THIS FIXTURE'S element type, not in
    /// absolute units: a K-term sum needs ~K of them, an elementwise op needs one, and
    /// the same number then means the same thing in a FLOAT fixture and a HALF one.
    /// Hardcoding an absolute float epsilon is how a HALF comparison ends up ~4000x
    /// tighter than the type can represent.
    void registerValidatorsForOutputs(hipdnn_frontend::graph::Graph& graph,
                                      float epsilonMultiple = 1.0f)
    {
        const float tolerance
            = epsilonMultiple * static_cast<float>(std::numeric_limits<DataType>::epsilon());
        graph.visit([&](const hipdnn_frontend::graph::INode& node) {
            for(const auto& tensorAttr : node.getNodeOutputTensorAttributes())
            {
                if(!tensorAttr->get_is_virtual())
                {
                    registerValidator(tensorAttr, tolerance);
                }
            }
        });
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
            // insert_or_assign, never insert: map::insert KEEPS the existing entry, so a
            // later registration for the same uid would be silently discarded and the
            // first graph's validator would judge every graph after it. Graphs in one
            // suite routinely share output uids, and they do not share dtypes or
            // accumulation depths.
            _tensorIdToValidatorMap.insert_or_assign(
                attr->get_uid(),
                hipdnn_test_sdk::utilities::createAllCloseValidator(
                    hipdnn_test_sdk::utilities::frontendToSdkDataType(attr->get_data_type()),
                    absoluteTolerance,
                    relativeTolerance));
            _tensorIdToNameMap.insert_or_assign(attr->get_uid(), attr->get_name());
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

    /// Seeds every tensor in @p bundle.
    ///
    /// The seed is offset PER UID so two operands never hold identical bytes: `a + a` and
    /// `a + b` agree elementwise when they do, and a comparison that cannot tell them
    /// apart passes for an operation that ignores one of its inputs.
    virtual void initializeBundle([[maybe_unused]] const hipdnn_frontend::graph::Graph& graph,
                                  hipdnn_test_sdk::utilities::GraphTensorBundle& bundle,
                                  unsigned int seed)
    {
        for(auto& tensorPair : bundle.tensors)
        {
            bundle.randomizeTensor(tensorPair.first,
                                   DEFAULT_MIN,
                                   DEFAULT_MAX,
                                   seed + static_cast<unsigned int>(tensorPair.first));
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
        _tensorIdToNameMap.insert_or_assign(tensorId, tensorAttr->get_name());

        return true;
    }

    std::unordered_map<int64_t, std::string> _tensorIdToNameMap;
    std::unordered_map<int64_t, std::unique_ptr<hipdnn_test_sdk::utilities::IReferenceValidation>>
        _tensorIdToValidatorMap;
    std::vector<std::function<void()>> _deferredValidators;
};

// NOLINTEND (portability-template-virtual-member-function)

} // namespace hip_kernel_provider::test_utilities
