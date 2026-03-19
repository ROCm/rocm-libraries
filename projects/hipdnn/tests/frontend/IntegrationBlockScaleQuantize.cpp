// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <memory>
#include <vector>

#include <hipdnn_data_sdk/utilities/MigratableMemory.hpp>
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
    NONE,
    VALIDATE,
    CREATE_EXECUTION_PLAN,
    EXECUTE
};

struct IntegrationTestCase
{
    std::string pluginPath;
    std::string description;
    std::string graphName;
    FailurePoint expectedFailure;
    bool useManualUids;
    bool useTranspose;

    friend std::ostream& operator<<(std::ostream& os, const IntegrationTestCase& tc)
    {
        os << "BlockScaleQuantizeTestCase{" << "plugin_path: " << tc.pluginPath
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
           << ", use_transpose: " << (tc.useTranspose ? "true" : "false") << "}";

        return os;
    }
};

// Test class for BlockScaleQuantize frontend integration tests.
// Runs end-to-end tests using fake test plugins to validate graph creation,
// execution plan building, and execution through the full flow.
class IntegrationBlockScaleQuantizeFp32 : public ::testing::TestWithParam<IntegrationTestCase>
{
protected:
    struct SimpleBlockScaleQuantizeTensorBundle
    {
        SimpleBlockScaleQuantizeTensorBundle(const std::vector<int64_t>& xDims,
                                             const std::vector<int64_t>& scaleDims)
            : xTensor(Tensor<float>(xDims))
            , yTensor(Tensor<float>(xDims)) // output same shape as input
            , scaleTensor(Tensor<float>(scaleDims))
        {
            xTensor.fillWithValue(1.0f);
            yTensor.fillWithValue(0.0f);
            scaleTensor.fillWithValue(0.0f);
        }

        Tensor<float> xTensor;
        Tensor<float> yTensor;
        Tensor<float> scaleTensor;
    };

    struct BlockScaleQuantizeTestTensors
    {
        std::shared_ptr<TensorAttributes> x;
        std::shared_ptr<TensorAttributes> y;
        std::shared_ptr<TensorAttributes> scale;
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

        const std::array<const char*, 1> paths = {pluginPath.c_str()};
        EXPECT_EQ(hipdnnSetEnginePluginPaths_ext(
                      paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE),
                  HIPDNN_STATUS_SUCCESS);

        hipdnnHandle_t handle = nullptr;
        EXPECT_EQ(hipdnnCreate(&handle), HIPDNN_STATUS_SUCCESS);

        return handle;
    }

    static std::pair<std::shared_ptr<Graph>, BlockScaleQuantizeTestTensors>
        createBlockScaleQuantizeTestGraph(const std::string& graphName,
                                          const SimpleBlockScaleQuantizeTensorBundle& tensorBundle,
                                          bool useManualUids,
                                          bool useTranspose)
    {
        auto graph = std::make_shared<Graph>();
        graph->set_name(graphName)
            .set_io_data_type(DataType::FLOAT)
            .set_intermediate_data_type(DataType::FLOAT)
            .set_compute_data_type(DataType::FLOAT);

        int64_t uid = 1;
        BlockScaleQuantizeTestTensors tensors;

        auto xAttr = makeTensorAttributes("X", DataType::FLOAT, tensorBundle.xTensor);
        if(useManualUids)
        {
            xAttr.set_uid(uid++);
        }
        tensors.x = std::make_shared<TensorAttributes>(std::move(xAttr));

        BlockScaleQuantizeAttributes attrs;
        attrs.set_name("block_scale_quantize");
        attrs.set_block_size(32);
        attrs.set_axis(1);
        attrs.set_transpose(useTranspose);

        auto [y, scale] = graph->block_scale_quantize(tensors.x, std::move(attrs));
        tensors.y = y;
        tensors.scale = scale;

        if(useManualUids)
        {
            tensors.y->set_uid(uid++);
            tensors.scale->set_uid(uid++);
        }
        tensors.y->set_data_type(DataType::FLOAT);
        tensors.scale->set_data_type(DataType::FLOAT);

        return {graph, tensors};
    }

    static std::unordered_map<int64_t, void*>
        createVariantPack(const BlockScaleQuantizeTestTensors& tensors,
                          SimpleBlockScaleQuantizeTensorBundle& tensorBundle)
    {
        std::unordered_map<int64_t, void*> variantPack;
        variantPack[tensors.x->get_uid()] = tensorBundle.xTensor.memory().deviceData();
        variantPack[tensors.y->get_uid()] = tensorBundle.yTensor.memory().deviceData();
        variantPack[tensors.scale->get_uid()] = tensorBundle.scaleTensor.memory().deviceData();

        return variantPack;
    }

    static void runGraphPipeline(const std::shared_ptr<Graph>& graph,
                                 hipdnnHandle_t handle,
                                 const BlockScaleQuantizeTestTensors& tensors,
                                 SimpleBlockScaleQuantizeTensorBundle& tensorBundle,
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
        ASSERT_TRUE(tensors.y->has_uid());
        ASSERT_TRUE(tensors.scale->has_uid());

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

        _handle = setupEnvironmentWithPlugin(testCase.pluginPath);

        // X: [2, 64, 32, 32], Scale: [2, 2, 32, 32] (block_size=32 on channel dim)
        const std::vector<int64_t> xDims = {2, 64, 32, 32};
        const std::vector<int64_t> scaleDims = {2, 2, 32, 32};
        SimpleBlockScaleQuantizeTensorBundle tensorBundle(xDims, scaleDims);

        auto [graph, tensors] = createBlockScaleQuantizeTestGraph(
            testCase.graphName, tensorBundle, testCase.useManualUids, testCase.useTranspose);

        runGraphPipeline(graph, _handle, tensors, tensorBundle, testCase.expectedFailure);
    }

private:
    hipdnnHandle_t _handle = nullptr;
};

} // namespace

INSTANTIATE_TEST_SUITE_P(
    ,
    IntegrationBlockScaleQuantizeFp32,
    ::testing::Values(
        IntegrationTestCase{hipdnn_tests::plugin_constants::testGoodPluginPath(),
                            "WithManualUids",
                            "BlockScaleQuantizeManualUids",
                            FailurePoint::NONE,
                            true,
                            false},
        IntegrationTestCase{hipdnn_tests::plugin_constants::testGoodPluginPath(),
                            "WithAutoUids",
                            "BlockScaleQuantizeAutoUids",
                            FailurePoint::NONE,
                            false,
                            false},
        IntegrationTestCase{hipdnn_tests::plugin_constants::testGoodPluginPath(),
                            "WithTranspose",
                            "BlockScaleQuantizeTranspose",
                            FailurePoint::NONE,
                            true,
                            true},
        IntegrationTestCase{hipdnn_tests::plugin_constants::testExecuteFailsPluginPath(),
                            "ExecuteFailsPlugin",
                            "ExecuteFailsBlockScaleQuantize",
                            FailurePoint::EXECUTE,
                            true,
                            false},
        IntegrationTestCase{hipdnn_tests::plugin_constants::testNoApplicableEnginesAPluginPath(),
                            "NoApplicableEnginesPlugin",
                            "NoEnginesBlockScaleQuantize",
                            FailurePoint::CREATE_EXECUTION_PLAN,
                            true,
                            false}),
    [](const ::testing::TestParamInfo<IntegrationTestCase>& info) {
        std::string name = info.param.description;
        std::transform(name.cbegin(), name.cend(), name.begin(), [](char c) {
            return std::isalnum(c) ? c : '_';
        });
        return name;
    });

TEST_P(IntegrationBlockScaleQuantizeFp32, ExecutePluginPipeline)
{
    runTest();
}
