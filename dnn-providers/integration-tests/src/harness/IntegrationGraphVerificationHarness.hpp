// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <gtest/gtest.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>

#include <hipdnn_data_sdk/utilities/Workspace.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/json/Graph.hpp>
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/Utilities.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>

#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceMiopenRmsValidation.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/SdkFrontendTypeConversions.hpp>
#include <hipdnn_test_sdk/utilities/TestTolerances.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <hipdnn_test_sdk/utilities/VectorLoggingUtils.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/GraphTensorBundle.hpp>
#include <nlohmann/json.hpp>
#include <random>
#include <vector>

#include "harness/GraphDescription.hpp"
#include "harness/IReferenceGraphExecutor.hpp"
#include "harness/ReferenceGraphExecutorFactory.hpp"
#include "harness/SharedHandle.hpp"
#include "harness/SupportMatrixCollector.hpp"
#include "harness/TestConfig.hpp"
#include "harness/TomlGuards.hpp"
#include "harness/input-init/FillInputs.hpp"
#include "harness/input-init/InputFillRecipes.hpp"
#include "harness/tolerance/ToleranceResolver.hpp"

namespace hipdnn_integration_tests
{

using namespace hipdnn_data_sdk;
using namespace hipdnn_frontend;

// Checks whether any (or the pinned --test-engine) engine supports the
// graph, skipping — or failing under --fail-on-unsupported — if not.
// build_operation_graph() must already have been called. Records
// support-matrix data (when testCaseNote/testCaseLayout are supplied) and
// pins the preferred engine when --test-engine is set. Callers that don't
// immediately return afterward must check
// ::testing::Test::IsSkipped()/HasFatalFailure() themselves.
//
// Free function (not a IntegrationGraphVerificationHarness member) so plain
// ::testing::Test fixtures that build ad hoc graphs outside the
// tiered-test-case harness (e.g. IntegrationIsSupportedExtPerformance) can
// reuse the same skip semantics instead of hard-asserting support.
inline void checkEngineSupportOrSkip(hipdnn_frontend::graph::Graph& graph,
                                     const std::string& testCaseNote = "",
                                     const std::string& testCaseLayout = "")
{
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
        SupportMatrixCollector::get().recordGraphSupport(graph.graph_attributes.get_name(),
                                                         describeGraph(graph),
                                                         testName,
                                                         status.is_good() ? engineIds
                                                                          : std::vector<int64_t>{},
                                                         testCaseNote,
                                                         testCaseLayout);
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
        // Preferred engine must be set before create_execution_plans.
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
}

// NOLINTBEGIN (portability-template-virtual-member-function)
template <typename DataType, typename TestCaseType>
class IntegrationGraphVerificationHarness : public ::testing::TestWithParam<TestCaseType>
{
protected:
    int _deviceId = 0;
    std::string _testCaseNote;
    std::string _testCaseLayout;
    InputFillRecipes _inputFillRecipes;
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

        if(auto reason = checkTomlSkip(currentTestName()))
        {
            GTEST_SKIP() << "[arch " << TestConfig::get().getCurrentArch() << "] " << *reason;
        }
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

    // Resolve tolerance for an output tensor via ToleranceResolver (max-across-nodes + TOML override).
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
        tolerance::resolveTolerance(
            wrapper, it->second->data_type(), currentTestName(), atol, rtol);
        // getTolerance's single-float contract predates split atol/rtol; under the
        // current resolver the two are equal (same default, same override).
        return atol;
    }

    // Delegates to the free hipdnn_integration_tests::checkEngineSupportOrSkip(),
    // forwarding this test case's note/layout for support-matrix recording.
    void checkEngineSupportOrSkip(hipdnn_frontend::graph::Graph& graph)
    {
        hipdnn_integration_tests::checkEngineSupportOrSkip(graph, _testCaseNote, _testCaseLayout);
    }

    void verifyGraph(hipdnn_frontend::graph::Graph& graph)
    {
        if(TestConfig::get().hasCaptureDir())
        {
            captureGraphBundle(graph);
            auto* info = ::testing::UnitTest::GetInstance()->current_test_info();
            HIPDNN_SDK_LOG_INFO("Capture-only mode: skipping execution for "
                                << (info ? info->test_suite_name() : "?") << "."
                                << (info ? info->name() : "?"));
            return;
        }

        ASSERT_NO_FATAL_FAILURE(ensureEngineSupport(graph));
        if(testing::Test::IsSkipped())
            return;

        if(TestConfig::get().skipGraphValidation())
            return;

        ASSERT_NO_FATAL_FAILURE(buildExecutionPlans(graph));

        hipdnn_test_sdk::utilities::GraphTensorBundle gpuBundle, refBundle;
        generateBundles(graph, refBundle, gpuBundle);

        auto initResult = initializeBundle(graph, gpuBundle);
        if(!initResult.filled)
        {
            GTEST_SKIP() << initResult.reason;
        }
        initResult = initializeBundle(graph, refBundle);
        if(!initResult.filled)
        {
            GTEST_SKIP() << initResult.reason;
        }

        ASSERT_NO_FATAL_FAILURE(executeGpuGraph(getSharedHandle(), graph, gpuBundle));
        ASSERT_NO_FATAL_FAILURE(executeReferenceGraph(graph, refBundle));

        ASSERT_NO_FATAL_FAILURE(validateOutputs(gpuBundle, refBundle));
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
        float finalAtol = absoluteTolerance;
        float finalRtol = relativeTolerance;
        applyTomlToleranceOverride(currentTestName(), finalAtol, finalRtol);

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

    InputFillRecipes& inputFillRecipes()
    {
        return _inputFillRecipes;
    }

    virtual FillResult initializeBundle(const hipdnn_frontend::graph::Graph& graph,
                                        hipdnn_test_sdk::utilities::GraphTensorBundle& bundle)
    {
        bundle.sentinelFillOutputTensors();

        auto [serialized, serErr] = graph.to_binary();
        if(serErr.code != hipdnn_frontend::ErrorCode::OK || serialized.empty())
        {
            return FillResult::unsupported("Graph serialization failed");
        }

        const auto* fb = hipdnn_flatbuffers_sdk::data_objects::GetGraph(serialized.data());
        if(fb == nullptr || fb->nodes() == nullptr)
        {
            return FillResult::unsupported("Graph flatbuffer is invalid");
        }

        return fillGraphInputs(*fb, bundle);
    }

    // Ranks engines for `graph` and either pins TestConfig's --test-engine as the
    // preferred engine (leaving the graph ready for create_execution_plans()/
    // build_plans(), or for Graph::build()) or GTEST_SKIP()s/FAILs the current
    // test when no suitable engine is available. GTEST_SKIP()/FAIL() only unwind
    // this function, not the caller's - callers MUST check
    // ::testing::Test::IsSkipped() (and, if they don't already ASSERT/FAIL
    // through to a return, HasFatalFailure()) and return immediately afterward.
    // Protected (rather than private) so callers that build/serialize a plan
    // manually instead of going through verifyGraph() (e.g. conv serialize
    // round-trip) can check engine support themselves first.
    void ensureEngineSupport(hipdnn_frontend::graph::Graph& graph)
    {
        checkEngineSupportOrSkip(graph);
    }

    void buildExecutionPlans(hipdnn_frontend::graph::Graph& graph)
    {
        auto result = graph.create_execution_plans();
        ASSERT_EQ(result.code, hipdnn_frontend::ErrorCode::OK) << result.err_msg;
        result = graph.check_support();
        ASSERT_EQ(result.code, hipdnn_frontend::ErrorCode::OK) << result.err_msg;
        result = graph.build_plans();
        ASSERT_EQ(result.code, hipdnn_frontend::ErrorCode::OK) << result.err_msg;
    }

    void validateOutputs(hipdnn_test_sdk::utilities::GraphTensorBundle& gpuBundle,
                         hipdnn_test_sdk::utilities::GraphTensorBundle& refBundle)
    {
        ASSERT_GE(gpuBundle.outputTensorIds.size(), 1)
            << "At least one output tensor id must be specified for validation.";

        HIPDNN_PLUGIN_LOG_INFO("Validating " << gpuBundle.outputTensorIds.size()
                                             << " output tensors");

        for(const auto& registerValidator : _deferredValidators)
        {
            registerValidator();
        }

        const bool referenceUsesDevice = getReferenceExecutor().requiresDeviceMemory();

        for(const auto& tensorId : gpuBundle.outputTensorIds)
        {
            auto& refTensor = refBundle.tensors.at(tensorId);
            auto& gpuTensor = gpuBundle.tensors.at(tensorId);

            gpuTensor->markDeviceModified();

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
            EXPECT_TRUE(valid) << "Mismatch found in tensor with id: " << tensorId
                               << ", name: " << _tensorIdToNameMap.at(tensorId);
        }
    }

    FillResult fillGraphInputs(const hipdnn_flatbuffers_sdk::data_objects::Graph& fb,
                               hipdnn_test_sdk::utilities::GraphTensorBundle& bundle)
    {
        std::vector<int64_t> leafInputUids;
        for(const auto& [uid, tensor] : bundle.tensors)
        {
            if(!bundle.isOutput(uid))
            {
                leafInputUids.push_back(uid);
            }
        }

        auto fillResult = fillInputs(fb, bundle.tensors, leafInputUids, _inputFillRecipes);
        if(!fillResult.filled)
        {
            return fillResult;
        }

        auto missing = _inputFillRecipes.unfilled(leafInputUids);
        if(!missing.empty())
        {
            std::string msg = "unfilled inputs:";
            for(const int64_t uid : missing)
            {
                msg += " uid=" + std::to_string(uid);
            }
            return FillResult::unsupported(msg);
        }

        return FillResult::ok();
    }

public:
    void captureGraphBundle(hipdnn_frontend::graph::Graph& graph)
    {
        auto* testInfo = ::testing::UnitTest::GetInstance()->current_test_info();
        if(testInfo == nullptr)
        {
            return;
        }

        const std::string suiteName = testInfo->test_suite_name();
        const std::string caseName = testInfo->name();

        auto [serialized, serErr] = graph.to_binary();
        if(serErr.code != hipdnn_frontend::ErrorCode::OK || serialized.empty())
        {
            HIPDNN_PLUGIN_LOG_WARN("capture: serialization failed for "
                                   << suiteName << "." << caseName << ": " << serErr.err_msg);
            return;
        }

        const auto* fb = hipdnn_flatbuffers_sdk::data_objects::GetGraph(serialized.data());
        if(fb == nullptr)
        {
            HIPDNN_PLUGIN_LOG_WARN("capture: null graph for " << suiteName << "." << caseName);
            return;
        }

        nlohmann::json graphJson;
        try
        {
            graphJson = *fb;
        }
        catch(const std::exception& e)
        {
            HIPDNN_PLUGIN_LOG_WARN("capture: JSON conversion failed for "
                                   << suiteName << "." << caseName << ": " << e.what());
            return;
        }

        std::string safeCaseName = caseName;
        std::replace(safeCaseName.begin(), safeCaseName.end(), '/', '_');

        const auto bundleDir = TestConfig::get().getCaptureDir() / suiteName / safeCaseName;
        std::filesystem::create_directories(bundleDir);

        const auto graphPath = bundleDir / (safeCaseName + ".json");
        {
            std::ofstream out(graphPath);
            if(!out)
            {
                HIPDNN_PLUGIN_LOG_WARN("capture: cannot write " << graphPath);
                return;
            }
            out << graphJson.dump();
        }

        nlohmann::json meta;
        meta["format_version"] = 1;
        meta["operation"] = suiteName;
        meta["generator"] = "capture-bundles";
        meta["generator_version"] = "1.0.0";
        meta["seed"] = _inputFillRecipes.globalSeed();

        if(!_inputFillRecipes.fills().empty())
        {
            meta["inputs"] = _inputFillRecipes.toJson();
        }

        meta["notes"] = "Captured from C++ graph test " + suiteName + "." + caseName;

        const auto metaPath = bundleDir / (safeCaseName + ".meta.json");
        {
            std::ofstream out(metaPath);
            if(!out)
            {
                HIPDNN_PLUGIN_LOG_WARN("capture: cannot write " << metaPath);
                return;
            }
            out << meta.dump(4);
        }

        HIPDNN_PLUGIN_LOG_INFO("capture: wrote " << graphPath);
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

        // The reference bundle keeps one element per byte for element-wise access;
        // the GPU bundle uses the packed device layout for sub-byte types (e.g. FP4
        // as two 4-bit values per byte) so the buffer can be consumed directly by
        // the kernel. Both are filled from the same seed, so they hold identical
        // logical values.
        refBundle.addTensor(*tensorAttr,
                            hipdnn_test_sdk::utilities::createTensorFromAttribute(
                                *tensorAttr, /*packSubByteElements=*/false));
        gpuBundle.addTensor(*tensorAttr,
                            hipdnn_test_sdk::utilities::createTensorFromAttribute(
                                *tensorAttr, /*packSubByteElements=*/true));
        _tensorIdToNameMap.insert({tensorId, tensorAttr->get_name()});

        return true;
    }
};

// NOLINTEND (portability-template-virtual-member-function)

} // namespace hipdnn_integration_tests
