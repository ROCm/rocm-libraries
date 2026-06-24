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
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceMiopenRmsValidation.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/SdkFrontendTypeConversions.hpp>
#include <hipdnn_test_sdk/utilities/TestTolerances.hpp>
#include <hipdnn_test_sdk/utilities/VectorLoggingUtils.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/GraphTensorBundle.hpp>
#include <nlohmann/json.hpp>
#include <vector>

#include "harness/GraphDescription.hpp"
#include "harness/IReferenceGraphExecutor.hpp"
#include "harness/ReferenceGraphExecutorFactory.hpp"
#include "harness/SharedHandle.hpp"
#include "harness/SupportMatrixCollector.hpp"
#include "harness/TestConfig.hpp"
#include "harness/TomlGuards.hpp"
#include "harness/input_init/SynthesizeInputs.hpp"
#include "harness/tolerance/ToleranceResolver.hpp"

namespace hipdnn_integration_tests
{

using namespace hipdnn_data_sdk;
using namespace hipdnn_frontend;

// NOLINTBEGIN (portability-template-virtual-member-function)
template <typename DataType, typename TestCaseType>
class IntegrationGraphVerificationHarness : public ::testing::TestWithParam<TestCaseType>
{
protected:
    std::string _testCaseNote;
    std::string _testCaseLayout;
    std::unordered_map<int64_t, std::string> _tensorIdToNameMap;
    std::unordered_map<int64_t, std::unique_ptr<hipdnn_test_sdk::utilities::IReferenceValidation>>
        _tensorIdToValidatorMap;
    std::vector<std::function<void()>> _deferredValidators;

    void SetUp() override
    {
        SKIP_IF_NO_DEVICES();

        // HIP initializes lazily on first runtime use; the shared hipdnn handle
        // (getSharedHandle -> hipdnnCreate) does this before any graph executes,
        // so no explicit hipInit is needed here.
        skipIfTomlMatched(currentTestName());
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

    // Determine the FINAL tolerance for an output tensor: an aggregation-policy
    // default plus the TOML per-test override, both via
    // harness/tolerance/ToleranceResolver.hpp. The resolver is keyed on the
    // serialized flatbuffer graph: we serialize with to_binary() — the same
    // pattern initializeBundle() already uses — and read the output tensor's dtype
    // from the flatbuffer.
    //
    // Policy = outputOpTolerance (the last non-Pointwise op), which reproduces
    // this harness's historical getTolerance() behavior so the C++ graph tests
    // keep their exact tolerances. (The bundle harness uses the maxAcrossNodes
    // default; the two agree for the common one-real-op + activation case.) The
    // returned value is already overridden, so registerValidator stores it as-is.
    float getTolerance(const hipdnn_frontend::graph::Graph& graph,
                       const std::shared_ptr<hipdnn_frontend::graph::TensorAttributes>& output)
    {
        ToleranceMode mode = TestConfig::get().getToleranceMode();
        if(mode != ToleranceMode::DEFAULT)
        {
            ADD_FAILURE() << "getTolerance: unhandled tolerance mode";
            return 0.0f;
        }

        auto [serialized, serErr] = graph.to_binary();
        if(serErr.code != hipdnn_frontend::ErrorCode::OK || serialized.empty())
        {
            ADD_FAILURE() << "getTolerance: graph serialization failed";
            return 0.0f;
        }

        const auto wrapper
            = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper::fromSerializedBlob(
                serialized.data(), serialized.size());
        if(!wrapper.isValid())
        {
            ADD_FAILURE() << "getTolerance: serialized graph failed verification";
            return 0.0f;
        }

        const auto& tensorMap = wrapper.getTensorMap();
        const auto it = tensorMap.find(output->get_uid());
        if(it == tensorMap.end())
        {
            ADD_FAILURE() << "getTolerance: output tensor uid " << output->get_uid()
                          << " not found in serialized graph";
            return 0.0f;
        }

        float atol = 0.0f;
        float rtol = 0.0f;
        tolerance::resolveTolerance(wrapper,
                                    it->second->data_type(),
                                    currentTestName(),
                                    atol,
                                    rtol,
                                    tolerance::outputOpTolerance);
        // getTolerance's single-float contract predates split atol/rtol; under the
        // current resolver the two are equal (same default, same override).
        return atol;
    }

    void verifyGraph(hipdnn_frontend::graph::Graph& graph, unsigned int seed)
    {
        hipdnn_test_sdk::utilities::GraphTensorBundle gpuBundle, refBundle;

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

        generateBundles(graph, refBundle, gpuBundle);

        initializeBundle(graph, gpuBundle, seed);
        initializeBundle(graph, refBundle, seed);

        ASSERT_NO_FATAL_FAILURE(executeGpuGraph(getSharedHandle(), graph, gpuBundle));
        ASSERT_NO_FATAL_FAILURE(executeReferenceGraph(graph, refBundle));

        ASSERT_GE(gpuBundle.outputTensorIds.size(), 1)
            << "At least one output tensor id must be specified for "
               "validation.";

        tolerance::warnIfMultipleOutputs(gpuBundle.outputTensorIds.size(),
                                         "IntegrationGraphVerificationHarness");

        HIPDNN_PLUGIN_LOG_INFO("Validating " << gpuBundle.outputTensorIds.size()
                                             << " output tensors");

        // Lazily register validators after graph execution since tensor Ids and types may be
        // inferred during graph finalization
        for(const auto& registerValidator : _deferredValidators)
        {
            registerValidator();
        }

        const bool referenceUsesDevice = getReferenceExecutor().requiresDeviceMemory();

        for(const auto& tensorId : gpuBundle.outputTensorIds)
        {
            auto& refTensor = refBundle.tensors.at(tensorId);
            auto& gpuTensor = gpuBundle.tensors.at(tensorId);

            // This tells the tensor that its data has been modified on the device side
            // All frontend graph knows is a (void*) pointer to device memory, so we need to inform
            // the tensor that the data there is now valid so that it knows to copy from device to
            // host when requested by the validation step.
            gpuTensor->markDeviceModified();

            // GPU reference executor writes to device memory — mark reference
            // tensors so host access triggers device-to-host sync
            if(referenceUsesDevice)
            {
                refTensor->markDeviceModified();
            }

            if(_tensorIdToValidatorMap.find(tensorId) == _tensorIdToValidatorMap.end())
            {
                FAIL() << "No validator registered for tensor with id: " << tensorId
                       << ", name: " << getOutputTensorName(tensorId);
            }

            bool valid = _tensorIdToValidatorMap.at(tensorId)->allClose(*refTensor, *gpuTensor);
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
        // Tolerances arrive already resolved (default + TOML override) from
        // getTolerance via ToleranceResolver; no override is applied here.
        const float finalAtol = absoluteTolerance;
        const float finalRtol = relativeTolerance;

        // Since the graph can infer properties + Ids, we defer validator registration until right
        // before validation in verifyGraph
        _deferredValidators.emplace_back([this, attr, finalAtol, finalRtol]() {
            auto [it, inserted] = _tensorIdToValidatorMap.insert(
                {attr->get_uid(),
                 hipdnn_test_sdk::utilities::createAllCloseValidator(
                     hipdnn_test_sdk::utilities::frontendToSdkDataType(attr->get_data_type()),
                     finalAtol,
                     finalRtol)});
            if(!inserted)
            {
                ADD_FAILURE() << "Duplicate validator for tensor " << attr->get_uid() << " ("
                              << attr->get_name() << "); keeping first registration";
            }
            _tensorIdToNameMap.insert({attr->get_uid(), attr->get_name()});
        });
    }

    void registerRmsValidator(const std::shared_ptr<hipdnn_frontend::graph::TensorAttributes> attr,
                              float rmsThreshold)
    {
        // Since the graph can infer properties + Ids, we defer validator registration until right
        // before validation in verifyGraph
        _deferredValidators.emplace_back([this, attr, rmsThreshold]() {
            auto [it, inserted] = _tensorIdToValidatorMap.insert(
                {attr->get_uid(),
                 hipdnn_test_sdk::utilities::createRmsValidator(
                     hipdnn_test_sdk::utilities::frontendToSdkDataType(attr->get_data_type()),
                     rmsThreshold)});
            if(!inserted)
            {
                ADD_FAILURE() << "Duplicate validator for tensor " << attr->get_uid() << " ("
                              << attr->get_name() << "); keeping first registration";
            }
            _tensorIdToNameMap.insert({attr->get_uid(), attr->get_name()});
        });
    }

    virtual void generateBundles(hipdnn_frontend::graph::Graph& graph,
                                 hipdnn_test_sdk::utilities::GraphTensorBundle& refBundle,
                                 hipdnn_test_sdk::utilities::GraphTensorBundle& gpuBundle)
    {
        graph.visit([&](const hipdnn_frontend::graph::INode& node) {
            for(const auto& tensorAttr : node.getNodeOutputTensorAttributes())
            {
                if(tryAddTensorToBundles(tensorAttr, refBundle, gpuBundle))
                {
                    auto uid = tensorAttr->get_uid();
                    refBundle.outputTensorIds.insert(uid);
                    gpuBundle.outputTensorIds.insert(uid);
                }
            }
            for(const auto& tensorAttr : node.getNodeInputTensorAttributes())
            {
                tryAddTensorToBundles(tensorAttr, refBundle, gpuBundle);
            }
        });
    }

    virtual void initializeBundle(const hipdnn_frontend::graph::Graph& graph,
                                  hipdnn_test_sdk::utilities::GraphTensorBundle& bundle,
                                  unsigned int seed)
    {
        bundle.sentinelFillOutputTensors();

        auto [serialized, serErr] = graph.to_binary();
        if(serErr.code != hipdnn_frontend::ErrorCode::OK || serialized.empty())
        {
            initializeBundleFallback(bundle, seed);
            return;
        }

        const auto* fb = hipdnn_flatbuffers_sdk::data_objects::GetGraph(serialized.data());
        if(fb == nullptr || fb->nodes() == nullptr)
        {
            initializeBundleFallback(bundle, seed);
            return;
        }

        std::vector<int64_t> leafInputUids;
        InputTensorMap inputs;
        for(auto& [uid, tensor] : bundle.tensors)
        {
            if(!bundle.isOutput(uid))
            {
                leafInputUids.push_back(uid);
                inputs[uid] = std::move(tensor);
            }
        }

        std::mt19937 rng(seed);
        SynthesisTracker tracker(leafInputUids, inputs);

        bool synthesisOk = true;
        for(const auto* node : *fb->nodes())
        {
            if(node == nullptr)
            {
                continue;
            }
            auto result = synthesizeNodeInputs(*node, tracker, rng);
            if(!result.filled)
            {
                synthesisOk = false;
                break;
            }
        }

        if(synthesisOk)
        {
            auto finalResult = tracker.finish("synthesis");
            synthesisOk = finalResult.filled;
        }

        for(auto& [uid, tensor] : inputs)
        {
            bundle.tensors[uid] = std::move(tensor);
        }

        if(!synthesisOk)
        {
            initializeBundleFallback(bundle, seed);
        }
    }

    void initializeBundleFallback(hipdnn_test_sdk::utilities::GraphTensorBundle& bundle,
                                  unsigned int seed)
    {
        for(auto& [uid, tensor] : bundle.tensors)
        {
            if(!bundle.isOutput(uid))
            {
                bundle.randomizeTensor(uid, -1.0f, 1.0f, seed);
            }
        }
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

    void executeReferenceGraph(hipdnn_frontend::graph::Graph& graph,
                               hipdnn_test_sdk::utilities::GraphTensorBundle& bundle)
    {
        auto [serializedGraph, serErr] = graph.to_binary();
        ASSERT_TRUE(serErr.is_good()) << serErr.get_message();

        auto& executor = getReferenceExecutor();
        const bool usesDevice = executor.requiresDeviceMemory();
        HIPDNN_PLUGIN_LOG_TRACE("executeReferenceGraph: using " << (usesDevice ? "device" : "host")
                                                                << " variant pack");
        auto variantPack = usesDevice ? bundle.toDeviceVariantPack() : bundle.toHostVariantPack();

        executor.execute(serializedGraph.data(), serializedGraph.size(), variantPack);
    }

    static IReferenceGraphExecutor& getReferenceExecutor()
    {
        static auto executor = ReferenceGraphExecutorFactory::createFromConfig();
        return *executor;
    }

    std::string getOutputTensorName(int64_t tensorId)
    {
        return _tensorIdToNameMap.at(tensorId);
    }

    bool tryAddTensorToBundles(
        const std::shared_ptr<hipdnn_frontend::graph::TensorAttributes>& tensorAttr,
        hipdnn_test_sdk::utilities::GraphTensorBundle& refBundle,
        hipdnn_test_sdk::utilities::GraphTensorBundle& gpuBundle)
    {
        int64_t tensorId = tensorAttr->get_uid();

        if(tensorAttr->get_is_virtual()
           || refBundle.tensors.find(tensorId) != refBundle.tensors.end())
        {
            return false;
        }

        refBundle.tensors.insert(
            {tensorId, hipdnn_test_sdk::utilities::createTensorFromAttribute(*tensorAttr)});
        gpuBundle.tensors.insert(
            {tensorId, hipdnn_test_sdk::utilities::createTensorFromAttribute(*tensorAttr)});
        _tensorIdToNameMap.insert({tensorId, tensorAttr->get_name()});

        return true;
    }
};

// NOLINTEND (portability-template-virtual-member-function)

} // namespace hipdnn_integration_tests
