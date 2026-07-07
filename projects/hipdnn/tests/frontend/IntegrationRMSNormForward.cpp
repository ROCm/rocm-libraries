// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <cmath>
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

#include <hipdnn_data_sdk/utilities/MigratableMemory.hpp>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_frontend.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <test_plugins/TestPluginConstants.hpp>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using namespace hipdnn_data_sdk::utilities;

namespace
{

enum class FailurePoint
{
    NONE, // No failure expected
    VALIDATE, // Expect failure at validate
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
    bool hasBias;
    NormFwdPhase forwardPhase;

    friend std::ostream& operator<<(std::ostream& os, const IntegrationTestCase& tc)
    {
        os << "RMSNormTestCase{" << "plugin_path: " << tc.pluginPath
           << ", description: " << tc.description << ", graph_name: " << tc.graphName
           << ", expected_failure: ";

        switch(tc.expectedFailure)
        {
        case FailurePoint::NONE:
            os << "NONE";
            break;
        case FailurePoint::VALIDATE:
            os << "VALIDATE";
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

        os << ", use_manual_uids: " << (tc.useManualUids ? "true" : "false")
           << ", has_bias: " << (tc.hasBias ? "true" : "false")
           << ", forward_phase: " << tc.forwardPhase << "}";

        return os;
    }
};

// Test class for RMSNorm frontend integration tests.
// This class runs end-to-end tests to ensure the RMSNorm frontend API works as expected.
// Notes:
//        - We are using fake test plugins to simulate different scenarios.
//        - The tests will validate the graph creation, execution plan building, and execution through the full flow.
//        - We test both bias and no-bias code paths.
class IntegrationRMSNormForwardFp32 : public ::testing::TestWithParam<IntegrationTestCase>
{
protected:
    // Simplified tensor bundle for RMSNorm frontend integration tests.
    // Under validateScaleNormalizedShape's trailing-suffix rule:
    //   - scale/bias match input except batch dim (e.g. [1, C, H, W] for [N, C, H, W])
    //   - invRms is [N, 1, 1, 1] (scale fully reduces non-batch dims)
    template <typename Input_type, typename Intermediate_type>
    struct SimpleRMSNorm2DTensorBundle
    {
        SimpleRMSNorm2DTensorBundle(const std::vector<int64_t>& dims)
            : scaleBiasDims(makeScaleBiasDims(dims))
            , invRmsDims(makeInvRmsDims(dims))
            , xTensor(Tensor<Input_type>(dims))
            , yTensor(Tensor<Input_type>(dims))
            , scaleTensor(Tensor<Intermediate_type>(scaleBiasDims))
            , biasTensor(Tensor<Intermediate_type>(scaleBiasDims))
            , invRmsTensor(Tensor<Intermediate_type>(invRmsDims))
        {
            // Initialize with simple constant values
            xTensor.fillWithValue(static_cast<Input_type>(1.0f));
            yTensor.fillWithValue(static_cast<Input_type>(0.0f));
            scaleTensor.fillWithValue(static_cast<Intermediate_type>(1.0f));
            biasTensor.fillWithValue(static_cast<Intermediate_type>(0.0f));
            invRmsTensor.fillWithValue(static_cast<Intermediate_type>(0.0f));
        }

        static std::vector<int64_t> makeScaleBiasDims(const std::vector<int64_t>& dims)
        {
            auto d = dims;
            d[0] = 1;
            return d;
        }

        static std::vector<int64_t> makeInvRmsDims(const std::vector<int64_t>& dims)
        {
            std::vector<int64_t> d(dims.size(), 1);
            d[0] = dims[0];
            return d;
        }

        std::vector<int64_t> scaleBiasDims;
        std::vector<int64_t> invRmsDims;
        Tensor<Input_type> xTensor;
        Tensor<Input_type> yTensor;
        Tensor<Intermediate_type> scaleTensor;
        Tensor<Intermediate_type> biasTensor;
        Tensor<Intermediate_type> invRmsTensor;
    };

    struct RMSNormTestTensors
    {
        std::shared_ptr<TensorAttributes> x;
        std::shared_ptr<TensorAttributes> scale;
        std::shared_ptr<TensorAttributes> bias; // nullptr when hasBias=false
        std::shared_ptr<TensorAttributes> epsilon;
        std::shared_ptr<TensorAttributes> y;
        std::shared_ptr<TensorAttributes> invRms;
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

    static std::pair<std::shared_ptr<Graph>, RMSNormTestTensors> createRMSNormTestGraphWithUids(
        const std::string& graphName,
        const SimpleRMSNorm2DTensorBundle<float, float>& tensorBundle,
        bool useManualUids,
        bool hasBias,
        NormFwdPhase forwardPhase)
    {
        auto graph = std::make_shared<hipdnn_frontend::graph::Graph>();
        graph->set_name(graphName)
            .set_io_data_type(DataType::FLOAT)
            .set_intermediate_data_type(DataType::FLOAT)
            .set_compute_data_type(DataType::FLOAT);

        int64_t uid = 1;
        RMSNormTestTensors tensors;

        auto xAttr = makeTensorAttributes("X", DataType::FLOAT, tensorBundle.xTensor);
        if(useManualUids)
        {
            xAttr.set_uid(uid++);
        }
        tensors.x = std::make_shared<TensorAttributes>(std::move(xAttr));

        auto scaleAttr = makeTensorAttributes("scale", DataType::FLOAT, tensorBundle.scaleTensor);
        if(useManualUids)
        {
            scaleAttr.set_uid(uid++);
        }
        tensors.scale = std::make_shared<TensorAttributes>(std::move(scaleAttr));

        tensors.epsilon = std::make_shared<TensorAttributes>(1e-5f);
        tensors.epsilon->set_name("epsilon");
        if(useManualUids)
        {
            tensors.epsilon->set_uid(uid++);
        }

        RMSNormAttributes attrs;
        attrs.set_name("rmsnorm");
        attrs.set_epsilon(tensors.epsilon);
        attrs.set_forward_phase(forwardPhase);

        if(hasBias)
        {
            auto biasAttr = makeTensorAttributes("bias", DataType::FLOAT, tensorBundle.biasTensor);
            if(useManualUids)
            {
                biasAttr.set_uid(uid++);
            }
            tensors.bias = std::make_shared<TensorAttributes>(std::move(biasAttr));
            attrs.set_bias(tensors.bias);
        }

        auto outputs = graph->rmsnorm(tensors.x, tensors.scale, std::move(attrs));
        tensors.y = outputs[0];
        tensors.invRms = outputs[1];

        if(useManualUids)
        {
            tensors.y->set_uid(uid++);
            if(tensors.invRms)
            {
                tensors.invRms->set_uid(uid++);
            }
        }
        tensors.y->set_data_type(DataType::FLOAT);
        if(tensors.invRms)
        {
            tensors.invRms->set_data_type(DataType::FLOAT);
        }

        return {graph, tensors};
    }

    static std::unordered_map<int64_t, void*>
        createVariantPack(const RMSNormTestTensors& tensors,
                          SimpleRMSNorm2DTensorBundle<float, float>& tensorBundle)
    {
        std::unordered_map<int64_t, void*> variantPack;
        variantPack[tensors.x->get_uid()] = tensorBundle.xTensor.memory().deviceData();
        variantPack[tensors.scale->get_uid()] = tensorBundle.scaleTensor.memory().deviceData();
        variantPack[tensors.y->get_uid()] = tensorBundle.yTensor.memory().deviceData();

        if(tensors.invRms != nullptr)
        {
            variantPack[tensors.invRms->get_uid()]
                = tensorBundle.invRmsTensor.memory().deviceData();
        }

        if(tensors.bias != nullptr)
        {
            variantPack[tensors.bias->get_uid()] = tensorBundle.biasTensor.memory().deviceData();
        }

        return variantPack;
    }

    static void runGraphPipeline(const std::shared_ptr<Graph>& graph,
                                 hipdnnHandle_t handle,
                                 const RMSNormTestTensors& tensors,
                                 SimpleRMSNorm2DTensorBundle<float, float>& tensorBundle,
                                 FailurePoint expectedFailure = FailurePoint::NONE)
    {
        auto result = graph->validate();
        if(expectedFailure == FailurePoint::VALIDATE)
        {
            ASSERT_NE(result.code, ErrorCode::OK) << "validate should fail";
            return;
        }
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
        ASSERT_TRUE(tensors.scale->has_uid());
        ASSERT_TRUE(tensors.epsilon->has_uid());
        ASSERT_TRUE(tensors.y->has_uid());
        if(tensors.invRms)
        {
            ASSERT_TRUE(tensors.invRms->has_uid());
        }

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
        const std::vector<int64_t> dims = {2, 3, 14, 14}; // n=2, c=3, h=14, w=14
        SimpleRMSNorm2DTensorBundle<float, float> tensorBundle(dims);

        // Create graph and tensors using the unified function
        auto [graph, tensors] = createRMSNormTestGraphWithUids(testCase.graphName,
                                                               tensorBundle,
                                                               testCase.useManualUids,
                                                               testCase.hasBias,
                                                               testCase.forwardPhase);

        // Verify INFERENCE mode produces nullptr for invRms
        if(testCase.forwardPhase == NormFwdPhase::INFERENCE)
        {
            ASSERT_EQ(tensors.invRms, nullptr) << "INFERENCE mode should not create invRms tensor";
        }

        runGraphPipeline(graph, _handle, tensors, tensorBundle, testCase.expectedFailure);
    }

private:
    hipdnnHandle_t _handle = nullptr;
};

} // namespace

INSTANTIATE_TEST_SUITE_P(
    ,
    IntegrationRMSNormForwardFp32,
    ::testing::Values(
        IntegrationTestCase{hipdnn_tests::plugin_constants::testGoodPluginPath(),
                            "TrainingWithManualUids",
                            "DefaultPluginRMSNormTest",
                            FailurePoint::NONE,
                            true,
                            false,
                            NormFwdPhase::TRAINING},
        IntegrationTestCase{hipdnn_tests::plugin_constants::testGoodPluginPath(),
                            "TrainingWithAutoUids",
                            "DefaultPluginRMSNormTestAutoUID",
                            FailurePoint::NONE,
                            false,
                            false,
                            NormFwdPhase::TRAINING},
        IntegrationTestCase{hipdnn_tests::plugin_constants::testGoodPluginPath(),
                            "TrainingWithBias",
                            "DefaultPluginRMSNormTestWithBias",
                            FailurePoint::NONE,
                            true,
                            true,
                            NormFwdPhase::TRAINING},
        IntegrationTestCase{hipdnn_tests::plugin_constants::testGoodPluginPath(),
                            "InferenceNoBias",
                            "InferenceRMSNormTest",
                            FailurePoint::NONE,
                            true,
                            false,
                            NormFwdPhase::INFERENCE},
        IntegrationTestCase{hipdnn_tests::plugin_constants::testGoodPluginPath(),
                            "InferenceWithBias",
                            "InferenceRMSNormTestWithBias",
                            FailurePoint::NONE,
                            true,
                            true,
                            NormFwdPhase::INFERENCE},
        IntegrationTestCase{hipdnn_tests::plugin_constants::testGoodPluginPath(),
                            "NotSetPhaseValidationFails",
                            "NotSetPhaseRMSNormTest",
                            FailurePoint::VALIDATE,
                            true,
                            false,
                            NormFwdPhase::NOT_SET},
        IntegrationTestCase{hipdnn_tests::plugin_constants::testExecuteFailsPluginPath(),
                            "ExecuteFailsPlugin",
                            "ExecuteFailsPluginRMSNormTest",
                            FailurePoint::EXECUTE,
                            true,
                            false,
                            NormFwdPhase::TRAINING},
        IntegrationTestCase{hipdnn_tests::plugin_constants::testNoApplicableEnginesAPluginPath(),
                            "NoApplicableEnginesPlugin",
                            "NoEnginesPluginRMSNormTest",
                            FailurePoint::CREATE_EXECUTION_PLAN,
                            true,
                            false,
                            NormFwdPhase::TRAINING}),
    // Provide a custom name for each test instance
    [](const ::testing::TestParamInfo<IntegrationTestCase>& info) {
        std::string name = info.param.description;
        std::transform(name.cbegin(), name.cend(), name.begin(), [](char c) {
            return std::isalnum(c) ? c : '_';
        });
        return name;
    });

TEST_P(IntegrationRMSNormForwardFp32, ExecutePluginPipeline)
{
    runTest();
}
