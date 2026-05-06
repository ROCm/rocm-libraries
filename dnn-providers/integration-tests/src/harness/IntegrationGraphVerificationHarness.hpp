// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <gtest/gtest.h>

#include <functional>
#include <hipdnn_data_sdk/utilities/Workspace.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/Utilities.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>
#include <hipdnn_frontend/node/ReductionNode.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceMiopenRmsValidation.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/SdkFrontendTypeConversions.hpp>
#include <hipdnn_test_sdk/utilities/TestTolerances.hpp>
#include <hipdnn_test_sdk/utilities/VectorLoggingUtils.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/CpuReferenceGraphExecutor.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/GraphTensorBundle.hpp>
#include <nlohmann/json.hpp>
#include <vector>

#include "harness/GraphDescription.hpp"
#include "harness/SharedHandle.hpp"
#include "harness/SupportMatrixCollector.hpp"
#include "harness/TestConfig.hpp"

namespace hipdnn_integration_tests
{

using namespace hipdnn_data_sdk;
using namespace hipdnn_frontend;

// NOLINTBEGIN (portability-template-virtual-member-function)
template <typename DataType, typename TestCaseType>
class IntegrationGraphVerificationHarness : public ::testing::TestWithParam<TestCaseType>
{
protected:
    int _deviceId = 0;
    std::string _testCaseNote;
    std::string _testCaseLayout;
    std::unordered_map<int64_t, std::string> _tensorIdToNameMap;
    std::unordered_map<int64_t, std::unique_ptr<hipdnn_test_sdk::utilities::IReferenceValidation>>
        _tensorIdToValidatorMap;
    std::vector<std::function<void()>> _deferredValidators;

    void SetUp() override
    {
        SKIP_IF_NO_DEVICES();

        // Initialize HIP
        ASSERT_EQ(hipInit(0), hipSuccess);
        ASSERT_EQ(hipGetDevice(&_deviceId), hipSuccess);
    }

    void setTestCaseNote(std::string note)
    {
        _testCaseNote = std::move(note);
    }

    void setTestCaseLayout(std::string layout)
    {
        _testCaseLayout = std::move(layout);
    }

    virtual void runGraphTest() = 0;

    // Determine tolerance for an output tensor based on the graph and
    // configured tolerance mode for the engine.
    float getTolerance(const hipdnn_frontend::graph::Graph& graph,
                       const std::shared_ptr<hipdnn_frontend::graph::TensorAttributes>& output)
    {
        ToleranceMode mode = TestConfig::get().getToleranceMode();

        if(mode == ToleranceMode::DEFAULT)
        {
            // We determine the tolerance based on the last non-PointwiseNode
            // (the root op). This will be gradually updated to use dynamic
            // calculation as possible; eventually, the tolerance will be
            // entirely dynamically determined in the default case.
            //
            // NOTE: after validate(), the graph's sub-nodes are in topological order.
            const hipdnn_frontend::graph::INode* rootOp = nullptr;
            graph.visit([&](const hipdnn_frontend::graph::INode& node) {
                if(dynamic_cast<const hipdnn_frontend::graph::PointwiseNode*>(&node) == nullptr
                   && dynamic_cast<const hipdnn_frontend::graph::Graph*>(&node) == nullptr)
                {
                    rootOp = &node;
                }
            });

            if(rootOp == nullptr)
            {
                ADD_FAILURE() << "getTolerance: no root op found in graph";
                return 0.0f;
            }

            return toleranceForNode(*rootOp, output->get_data_type());
        }

        ADD_FAILURE() << "getTolerance: unhandled tolerance mode";
        return 0.0f;
    }

    void verifyGraph(hipdnn_frontend::graph::Graph& graph, unsigned int seed)
    {
        hipdnn_test_sdk::utilities::GraphTensorBundle gpuBundle, cpuBundle;
        std::vector<int64_t> outputTensorIds;

        // Check engine support and set preferred engine before building execution plans.
        // build_operation_graph() was already called by buildGraph() in the test subclass.
        std::vector<int64_t> engineIds;
        auto status = graph.get_ranked_engine_ids(engineIds);

        // Record support information for the support matrix output
        if(SupportMatrixCollector::get().isEnabled())
        {
            std::string testName;
            auto* testInfo = ::testing::UnitTest::GetInstance()->current_test_info();
            if(testInfo != nullptr)
            {
                testName = std::string(testInfo->test_suite_name()) + "." + testInfo->name();
            }
            SupportMatrixCollector::get().recordGraphSupport(
                graph.graph_attributes.get_name(),
                describeGraph(graph),
                testName,
                status.is_good() ? engineIds : std::vector<int64_t>{},
                _testCaseNote,
                _testCaseLayout);
        }

        if(TestConfig::get().hasEngineName())
        {
            int64_t targetEngineId = TestConfig::get().getEngineId();
            if(status.is_bad()
               || std::find(engineIds.begin(), engineIds.end(), targetEngineId) == engineIds.end())
            {
                if(TestConfig::get().failOnUnsupported())
                {
                    FAIL() << "Engine " << TestConfig::get().getEngineName()
                           << " does not support this graph";
                }
                GTEST_SKIP() << "Engine " << TestConfig::get().getEngineName()
                             << " does not support this graph";
            }
            // Prererred engine must be set before create_execution_plans.
            graph.set_preferred_engine_id_ext(targetEngineId);
        }
        else
        {
            if(status.is_bad() || engineIds.empty())
            {
                if(TestConfig::get().failOnUnsupported())
                {
                    FAIL() << "No engine supports this graph";
                }
                GTEST_SKIP() << "No engine supports this graph";
            }
        }

        // --skip-graph-validation: graph is confirmed supported, exit early with PASS
        if(TestConfig::get().skipGraphValidation())
        {
            return;
        }

        // Build execution plans, engine preference set above should ensure that
        // correct engine is selected.
        auto result = graph.create_execution_plans();
        ASSERT_EQ(result.code, hipdnn_frontend::ErrorCode::OK) << result.err_msg;
        result = graph.check_support();
        ASSERT_EQ(result.code, hipdnn_frontend::ErrorCode::OK) << result.err_msg;
        result = graph.build_plans();
        ASSERT_EQ(result.code, hipdnn_frontend::ErrorCode::OK) << result.err_msg;

        generateBundles(graph, cpuBundle, gpuBundle, outputTensorIds);

        initializeBundle(graph, gpuBundle, seed);
        initializeBundle(graph, cpuBundle, seed);

        ASSERT_NO_FATAL_FAILURE(executeGpuGraph(getSharedHandle(), graph, gpuBundle));
        executeCpuGraph(graph, cpuBundle);

        ASSERT_GE(outputTensorIds.size(), 1)
            << "At least one output tensor id must be specified for "
               "validation.";

        HIPDNN_PLUGIN_LOG_INFO("Validating " << outputTensorIds.size() << " output tensors");

        // Lazily register validators after graph execution since tensor Ids and types may be
        // inferred during graph finalization
        for(const auto& registerValidator : _deferredValidators)
        {
            registerValidator();
        }

        for(const auto& tensorId : outputTensorIds)
        {
            auto& cpuTensor = cpuBundle.tensors.at(tensorId);
            auto& gpuTensor = gpuBundle.tensors.at(tensorId);

            // This tells the tensor that its data has been modified on the device side
            // All frontend graph knows is a (void*) pointer to device memory, so we need to inform
            // the tensor that the data there is now valid so that it knows to copy from device to
            // host when requested by the validation step.
            gpuTensor->markDeviceModified();

            if(_tensorIdToValidatorMap.find(tensorId) == _tensorIdToValidatorMap.end())
            {
                FAIL() << "No validator registered for tensor with id: " << tensorId
                       << ", name: " << getOutputTensorName(tensorId);
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
        // Check for per-test tolerance override from TOML config
        float finalAtol = absoluteTolerance;
        float finalRtol = relativeTolerance;

        auto* testInfo = ::testing::UnitTest::GetInstance()->current_test_info();
        if(testInfo != nullptr)
        {
            std::string testName
                = std::string(testInfo->test_suite_name()) + "." + testInfo->name();
            auto override = TestConfig::get().findToleranceOverride(testName);
            if(override.has_value())
            {
                finalAtol = override->atol;
                finalRtol = override->rtol;
                HIPDNN_PLUGIN_LOG_INFO("Tolerance override applied for " << testName
                                                                         << ": atol=" << finalAtol
                                                                         << " rtol=" << finalRtol);
            }
        }

        // Since the graph can infer properties + Ids, we defer validator registration until right
        // before validation in verifyGraph
        _deferredValidators.emplace_back([this, attr, finalAtol, finalRtol]() {
            _tensorIdToValidatorMap.insert(
                {attr->get_uid(),
                 hipdnn_test_sdk::utilities::createAllCloseValidator(
                     hipdnn_test_sdk::utilities::frontendToSdkDataType(attr->get_data_type()),
                     finalAtol,
                     finalRtol)});
            _tensorIdToNameMap.insert({attr->get_uid(), attr->get_name()});
        });
    }

    void registerRmsValidator(const std::shared_ptr<hipdnn_frontend::graph::TensorAttributes> attr,
                              float rmsThreshold)
    {
        // Since the graph can infer properties + Ids, we defer validator registration until right
        // before validation in verifyGraph
        _deferredValidators.emplace_back([this, attr, rmsThreshold]() {
            _tensorIdToValidatorMap.insert(
                {attr->get_uid(),
                 hipdnn_test_sdk::utilities::createRmsValidator(
                     hipdnn_test_sdk::utilities::frontendToSdkDataType(attr->get_data_type()),
                     rmsThreshold)});
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
            bundle.randomizeTensor(tensorPair.first, -1.0f, 1.0f, seed);
        }
    }

    static float toleranceForNode(const hipdnn_frontend::graph::INode& node,
                                  hipdnn_frontend::DataType dataType)
    {
        switch(dataType)
        {
        case hipdnn_frontend::DataType::FLOAT:
            return toleranceForNodeTyped<float>(node);
        case hipdnn_frontend::DataType::HALF:
            return toleranceForNodeTyped<half>(node);
        case hipdnn_frontend::DataType::BFLOAT16:
            return toleranceForNodeTyped<bfloat16>(node);
        default:
            ADD_FAILURE() << "toleranceForNode: unsupported data type";
            return 0.0f;
        }
    }

    template <typename T>
    static float toleranceForNodeTyped(const hipdnn_frontend::graph::INode& node)
    {
        namespace fe = hipdnn_frontend::graph;
        using namespace hipdnn_test_sdk::utilities;

        if(dynamic_cast<const fe::ConvolutionFpropNode*>(&node) != nullptr)
            return static_cast<float>(conv::getToleranceFwd<T>());
        if(dynamic_cast<const fe::ConvolutionDgradNode*>(&node) != nullptr)
            return static_cast<float>(conv::getToleranceBwd<T>());
        if(dynamic_cast<const fe::ConvolutionWgradNode*>(&node) != nullptr)
            return static_cast<float>(conv::getToleranceWrw<T>());
        if(dynamic_cast<const fe::BatchnormInferenceNodeVarianceExt*>(&node) != nullptr)
            return static_cast<float>(batchnorm::getToleranceInferenceWithVariance<T>());
        if(dynamic_cast<const fe::BatchnormInferenceNode*>(&node) != nullptr)
            return static_cast<float>(batchnorm::getToleranceInference<T>());
        if(dynamic_cast<const fe::BatchnormNode*>(&node) != nullptr)
            return static_cast<float>(batchnorm::getToleranceTraining<T>());
        if(dynamic_cast<const fe::BatchnormBackwardNode*>(&node) != nullptr)
            return static_cast<float>(batchnorm::getToleranceBackward<T>());
        if(dynamic_cast<const fe::MatmulNode*>(&node) != nullptr)
            return static_cast<float>(matmul::getTolerance<T>());
        if(dynamic_cast<const fe::ReductionNode*>(&node) != nullptr)
            return static_cast<float>(reduction::getTolerance<T>());

        ADD_FAILURE() << "toleranceForNodeTyped: unsupported node type";
        return 0.0f;
    }

    void executeGpuGraph(hipdnnHandle_t handle,
                         hipdnn_frontend::graph::Graph& graph,
                         hipdnn_test_sdk::utilities::GraphTensorBundle& bundle)
    {
        int64_t workspaceSize;
        auto result = graph.get_workspace_size(workspaceSize);
        ASSERT_EQ(result.code, hipdnn_frontend::ErrorCode::OK) << result.err_msg;
        ASSERT_GE(workspaceSize, 0) << result.err_msg;
        utilities::Workspace workspace(static_cast<size_t>(workspaceSize));

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

    std::string getOutputTensorName(int64_t tensorId)
    {
        return _tensorIdToNameMap.at(tensorId);
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
};

// NOLINTEND (portability-template-virtual-member-function)

} // namespace hipdnn_integration_tests
