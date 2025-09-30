// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <cmath>
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

#include <hipdnn_frontend.hpp>
#include <hipdnn_sdk/test_utilities/TestUtilities.hpp>
#include <hipdnn_sdk/utilities/MigratableMemory.hpp>
#include <hipdnn_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_sdk/utilities/Tensor.hpp>
#include <test_plugins/TestPluginConstants.hpp>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using namespace hipdnn_sdk::utilities;

namespace
{

enum class FailurePoint
{
    NONE, // No failure expected
    CREATE_EXECUTION_PLAN, // Expect failure at create execution plan
    EXECUTE // Expect failure at execute
};

struct IntegrationTestCase
{
    std::string pluginPath;
    std::string description;
    std::string graphName;
    FailurePoint expectedFailure;
    bool useManualUids;

    friend std::ostream& operator<<(std::ostream& os, const IntegrationTestCase& tc)
    {
        os << "BatchnormTestCase{" << "plugin_path: " << tc.pluginPath
           << ", description: " << tc.description << ", graph_name: " << tc.graphName
           << ", expected_failure: ";

        switch(tc.expectedFailure)
        {
        case FailurePoint::NONE:
            os << "NONE";
            break;
        case FailurePoint::CREATE_EXECUTION_PLAN:
            os << "CREATE_EXECUTION_PLAN";
            break;
        case FailurePoint::EXECUTE:
            os << "EXECUTE";
            break;
        default:
            os << "UNKNOWN";
            break;
        }

        os << ", use_manual_uids: " << (tc.useManualUids ? "true" : "false") << "}";

        return os;
    }
};

// Test class for frontend integration tests.
// This class runs end-to-end tests to ensure the frontend API works as expected.
// Notes:
//        - We are using fake test plugins to simulate different scenarios.
//        - The tests will validate the graph creation, execution plan building, and execution through the full flow.
//        - We are using a batchnorm graph since the graph doesn't really matter due to fake plugins.
class IntegrationBatchnormForwardInferenceFp32
    : public ::testing::TestWithParam<IntegrationTestCase>
{
protected:
    // Simplified tensor bundle for frontend integration tests
    template <typename Input_type, typename Intermediate_type>
    struct SimpleBatchnorm2DTensorBundle
    {
        SimpleBatchnorm2DTensorBundle(const std::vector<int64_t>& dims)
            : derivedDims(getDerivedShape(dims))
            , xTensor(Tensor<Input_type>(dims))
            , yTensor(Tensor<Input_type>(dims))
            , scaleTensor(Tensor<Intermediate_type>(derivedDims))
            , biasTensor(Tensor<Intermediate_type>(derivedDims))
            , meanTensor(Tensor<Intermediate_type>(derivedDims))
            , varianceTensor(Tensor<Intermediate_type>(derivedDims))
        {
            // Initialize with simple constant values
            xTensor.fillWithValue(static_cast<Input_type>(1.0f));
            yTensor.fillWithValue(static_cast<Input_type>(0.0f));
            scaleTensor.fillWithValue(static_cast<Intermediate_type>(1.0f));
            biasTensor.fillWithValue(static_cast<Intermediate_type>(0.0f));
            meanTensor.fillWithValue(static_cast<Intermediate_type>(0.0f));
            varianceTensor.fillWithValue(static_cast<Intermediate_type>(1.0f));
        }

        std::vector<int64_t> derivedDims;
        Tensor<Input_type> xTensor;
        Tensor<Input_type> yTensor;
        Tensor<Intermediate_type> scaleTensor;
        Tensor<Intermediate_type> biasTensor;
        Tensor<Intermediate_type> meanTensor;
        Tensor<Intermediate_type> varianceTensor;
    };

    struct BatchnormTestTensors
    {
        std::shared_ptr<TensorAttributes> x;
        std::shared_ptr<TensorAttributes> mean;
        std::shared_ptr<TensorAttributes> invVariance;
        std::shared_ptr<TensorAttributes> scale;
        std::shared_ptr<TensorAttributes> bias;
        std::shared_ptr<TensorAttributes> y;
    };

    void SetUp() override
    {
        SKIP_IF_NO_DEVICES();
    }

    void TearDown() override
    {
        if(_handle != nullptr)
        {
            ASSERT_EQ(hipdnnDestroy(_handle), HIPDNN_STATUS_SUCCESS);
        }
    }

    static hipdnnHandle_t setupEnvironmentWithPlugin(const std::string& pluginPath)
    {
        EXPECT_EQ(hipInit(0), hipSuccess);
        int deviceId = 0;
        EXPECT_EQ(hipGetDevice(&deviceId), hipSuccess);

        // Set up plugin path - load specific plugin by absolute path
        const std::array<const char*, 1> paths = {pluginPath.c_str()};
        EXPECT_EQ(hipdnnSetEnginePluginPaths_ext(
                      paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE),
                  HIPDNN_STATUS_SUCCESS);

        // Create handle
        hipdnnHandle_t handle = nullptr;
        EXPECT_EQ(hipdnnCreate(&handle), HIPDNN_STATUS_SUCCESS);

        return handle;
    }

    static std::pair<std::shared_ptr<Graph>, BatchnormTestTensors> createBatchnormTestGraphWithUids(
        const std::string& graphName,
        const SimpleBatchnorm2DTensorBundle<float, float>& tensorBundle,
        bool useManualUids)
    {
        auto graph = std::make_shared<hipdnn_frontend::graph::Graph>();
        graph->set_name(graphName);

        int64_t uid = 1;
        BatchnormTestTensors tensors;

        auto xAttr = makeTensorAttributes("X", DataType_t::FLOAT, tensorBundle.xTensor);
        if(useManualUids)
        {
            xAttr.set_uid(uid++);
        }
        tensors.x = std::make_shared<TensorAttributes>(std::move(xAttr));

        auto meanAttr = makeTensorAttributes("mean", DataType_t::FLOAT, tensorBundle.meanTensor);
        if(useManualUids)
        {
            meanAttr.set_uid(uid++);
        }
        tensors.mean = std::make_shared<TensorAttributes>(std::move(meanAttr));

        auto invVarianceAttr
            = makeTensorAttributes("inv_variance", DataType_t::FLOAT, tensorBundle.varianceTensor);
        if(useManualUids)
        {
            invVarianceAttr.set_uid(uid++);
        }
        tensors.invVariance = std::make_shared<TensorAttributes>(std::move(invVarianceAttr));

        auto scaleAttr = makeTensorAttributes("scale", DataType_t::FLOAT, tensorBundle.scaleTensor);
        if(useManualUids)
        {
            scaleAttr.set_uid(uid++);
        }
        tensors.scale = std::make_shared<TensorAttributes>(std::move(scaleAttr));

        auto biasAttr = makeTensorAttributes("bias", DataType_t::FLOAT, tensorBundle.biasTensor);
        if(useManualUids)
        {
            biasAttr.set_uid(uid++);
        }
        tensors.bias = std::make_shared<TensorAttributes>(std::move(biasAttr));

        BatchnormInferenceAttributes bnAttrs;
        bnAttrs.set_name("batchnorm_inference");

        tensors.y = graph->batchnorm_inference(
            tensors.x, tensors.mean, tensors.invVariance, tensors.scale, tensors.bias, bnAttrs);

        if(useManualUids)
        {
            tensors.y->set_uid(uid++);
        }
        tensors.y->set_data_type(DataType_t::FLOAT);

        return {graph, tensors};
    }

    static std::unordered_map<int64_t, void*>
        createVariantPack(const BatchnormTestTensors& tensors,
                          SimpleBatchnorm2DTensorBundle<float, float>& tensorBundle)
    {
        std::unordered_map<int64_t, void*> variantPack;
        variantPack[tensors.x->get_uid()] = tensorBundle.xTensor.memory().deviceData();
        variantPack[tensors.mean->get_uid()] = tensorBundle.meanTensor.memory().deviceData();
        variantPack[tensors.invVariance->get_uid()]
            = tensorBundle.varianceTensor.memory().deviceData();
        variantPack[tensors.scale->get_uid()] = tensorBundle.scaleTensor.memory().deviceData();
        variantPack[tensors.bias->get_uid()] = tensorBundle.biasTensor.memory().deviceData();
        variantPack[tensors.y->get_uid()] = tensorBundle.yTensor.memory().deviceData();

        return variantPack;
    }

    static void runGraphPipeline(const std::shared_ptr<Graph>& graph,
                                 hipdnnHandle_t handle,
                                 const BatchnormTestTensors& tensors,
                                 SimpleBatchnorm2DTensorBundle<float, float>& tensorBundle,
                                 FailurePoint expectedFailure = FailurePoint::NONE)
    {
        auto result = graph->validate();
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

        result = graph->build_operation_graph(handle);
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

        result = graph->create_execution_plans();
        if(expectedFailure == FailurePoint::CREATE_EXECUTION_PLAN)
        {
            ASSERT_NE(result.code, ErrorCode::OK) << "create_execution_plans should fail";
            return;
        }
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

        result = graph->check_support();
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

        result = graph->build_plans();
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

        ASSERT_TRUE(tensors.x->has_uid());
        ASSERT_TRUE(tensors.mean->has_uid());
        ASSERT_TRUE(tensors.invVariance->has_uid());
        ASSERT_TRUE(tensors.scale->has_uid());
        ASSERT_TRUE(tensors.bias->has_uid());
        ASSERT_TRUE(tensors.y->has_uid());

        auto variantPack = createVariantPack(tensors, tensorBundle);

        result = graph->execute(handle, variantPack, nullptr);
        if(expectedFailure == FailurePoint::EXECUTE)
        {
            ASSERT_NE(result.code, ErrorCode::OK) << "Execute should fail";
        }
        else
        {
            ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
        }
    }

    void runTest()
    {
        const auto& testCase = GetParam();

        // Setup environment with specified plugin
        _handle = setupEnvironmentWithPlugin(testCase.pluginPath);

        // Create tensor bundle
        std::vector<int64_t> dims = {2, 3, 14, 14}; // n=2, c=3, h=14, w=14
        SimpleBatchnorm2DTensorBundle<float, float> tensorBundle(dims);

        // Create graph and tensors using the unified function
        auto [graph, tensors] = createBatchnormTestGraphWithUids(
            testCase.graphName, tensorBundle, testCase.useManualUids);

        runGraphPipeline(graph, _handle, tensors, tensorBundle, testCase.expectedFailure);
    }

private:
    hipdnnHandle_t _handle = nullptr;
};

} // namespace

INSTANTIATE_TEST_SUITE_P(
    ,
    IntegrationBatchnormForwardInferenceFp32,
    ::testing::Values(
        IntegrationTestCase{hipdnn_tests::plugin_constants::testGoodPluginPath(),
                            "DefaultPluginWithManualUids",
                            "DefaultPluginBatchnormTest",
                            FailurePoint::NONE,
                            true},
        IntegrationTestCase{hipdnn_tests::plugin_constants::testGoodPluginPath(),
                            "DefaultPluginWithAutoUids",
                            "DefaultPluginBatchnormTestAutoUID",
                            FailurePoint::NONE,
                            false},
        IntegrationTestCase{hipdnn_tests::plugin_constants::testExecuteFailsPluginPath(),
                            "ExecuteFailsPlugin",
                            "ExecuteFailsPluginBatchnormTest",
                            FailurePoint::EXECUTE,
                            true},
        IntegrationTestCase{hipdnn_tests::plugin_constants::testNoApplicableEnginesAPluginPath(),
                            "NoApplicableEnginesPlugin",
                            "NoEnginesPluginBatchnormTest",
                            FailurePoint::CREATE_EXECUTION_PLAN,
                            true}),
    // Provide a custom name for each test instance
    [](const ::testing::TestParamInfo<IntegrationTestCase>& info) {
        std::string name = info.param.description;
        std::transform(name.cbegin(), name.cend(), name.begin(), [](char c) {
            return std::isalnum(c) ? c : '_';
        });
        return name;
    });

TEST_P(IntegrationBatchnormForwardInferenceFp32, ExecutePluginPipeline)
{
    runTest();
}
