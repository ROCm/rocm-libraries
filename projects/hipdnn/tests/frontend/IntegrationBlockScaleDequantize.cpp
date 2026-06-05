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
    bool useNegativeScale;

    friend std::ostream& operator<<(std::ostream& os, const IntegrationTestCase& tc)
    {
        os << "BlockScaleDequantizeTestCase{" << "plugin_path: " << tc.pluginPath
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
           << ", use_negative_scale: " << (tc.useNegativeScale ? "true" : "false") << "}";

        return os;
    }
};

// Test class for BlockScaleDequantize frontend integration tests.
// Runs end-to-end tests using fake test plugins to validate graph creation,
// execution plan building, and execution through the full flow.
class IntegrationBlockScaleDequantizeFp32 : public ::testing::TestWithParam<IntegrationTestCase>
{
protected:
    struct SimpleBlockScaleDequantizeTensorBundle
    {
        SimpleBlockScaleDequantizeTensorBundle(const std::vector<int64_t>& xDims,
                                               const std::vector<int64_t>& scaleDims)
            : xTensor(Tensor<float>(xDims))
            , scaleTensor(Tensor<float>(scaleDims))
            , yTensor(Tensor<float>(xDims)) // output same shape as input
        {
            xTensor.fillWithValue(1.0f);
            scaleTensor.fillWithValue(1.0f);
            yTensor.fillWithValue(0.0f);
        }

        Tensor<float> xTensor;
        Tensor<float> scaleTensor;
        Tensor<float> yTensor;
    };

    struct BlockScaleDequantizeTestTensors
    {
        std::shared_ptr<TensorAttributes> x;
        std::shared_ptr<TensorAttributes> scale;
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

        const std::array<const char*, 1> paths = {pluginPath.c_str()};
        EXPECT_EQ(hipdnnSetEnginePluginPaths_ext(
                      paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE),
                  HIPDNN_STATUS_SUCCESS);

        hipdnnHandle_t handle = nullptr;
        EXPECT_EQ(hipdnnCreate(&handle), HIPDNN_STATUS_SUCCESS);

        return handle;
    }

    static std::pair<std::shared_ptr<Graph>, BlockScaleDequantizeTestTensors>
        createBlockScaleDequantizeTestGraph(
            const std::string& graphName,
            const SimpleBlockScaleDequantizeTensorBundle& tensorBundle,
            bool useManualUids,
            bool useNegativeScale)
    {
        auto graph = std::make_shared<Graph>();
        graph->set_name(graphName)
            .set_io_data_type(DataType::FLOAT)
            .set_intermediate_data_type(DataType::FLOAT)
            .set_compute_data_type(DataType::FLOAT);

        int64_t uid = 1;
        BlockScaleDequantizeTestTensors tensors;

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

        BlockScaleDequantizeAttributes attrs;
        attrs.set_name("block_scale_dequantize");
        attrs.set_block_size(std::vector<int32_t>{32});
        attrs.set_is_negative_scale(useNegativeScale);

        tensors.y = graph->block_scale_dequantize(tensors.x, tensors.scale, std::move(attrs));

        if(useManualUids)
        {
            tensors.y->set_uid(uid++);
        }
        tensors.y->set_data_type(DataType::FLOAT);

        return {graph, tensors};
    }

    static std::unordered_map<int64_t, void*>
        createVariantPack(const BlockScaleDequantizeTestTensors& tensors,
                          SimpleBlockScaleDequantizeTensorBundle& tensorBundle)
    {
        std::unordered_map<int64_t, void*> variantPack;
        variantPack[tensors.x->get_uid()] = tensorBundle.xTensor.memory().deviceData();
        variantPack[tensors.scale->get_uid()] = tensorBundle.scaleTensor.memory().deviceData();
        variantPack[tensors.y->get_uid()] = tensorBundle.yTensor.memory().deviceData();

        return variantPack;
    }

    static void runGraphPipeline(const std::shared_ptr<Graph>& graph,
                                 hipdnnHandle_t handle,
                                 const BlockScaleDequantizeTestTensors& tensors,
                                 SimpleBlockScaleDequantizeTensorBundle& tensorBundle,
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

        _handle = setupEnvironmentWithPlugin(testCase.pluginPath);

        // X: [2, 64, 32, 32], Scale: [2, 2, 32, 32] (block_size=32 on channel dim)
        const std::vector<int64_t> xDims = {2, 64, 32, 32};
        const std::vector<int64_t> scaleDims = {2, 2, 32, 32};
        SimpleBlockScaleDequantizeTensorBundle tensorBundle(xDims, scaleDims);

        auto [graph, tensors] = createBlockScaleDequantizeTestGraph(
            testCase.graphName, tensorBundle, testCase.useManualUids, testCase.useNegativeScale);

        runGraphPipeline(graph, _handle, tensors, tensorBundle, testCase.expectedFailure);
    }

private:
    hipdnnHandle_t _handle = nullptr;
};

} // namespace

INSTANTIATE_TEST_SUITE_P(
    ,
    IntegrationBlockScaleDequantizeFp32,
    ::testing::Values(
        IntegrationTestCase{hipdnn_tests::plugin_constants::testGoodPluginPath(),
                            "WithManualUids",
                            "BlockScaleDequantizeManualUids",
                            FailurePoint::NONE,
                            true,
                            false},
        IntegrationTestCase{hipdnn_tests::plugin_constants::testGoodPluginPath(),
                            "WithAutoUids",
                            "BlockScaleDequantizeAutoUids",
                            FailurePoint::NONE,
                            false,
                            false},
        IntegrationTestCase{hipdnn_tests::plugin_constants::testGoodPluginPath(),
                            "WithNegativeScale",
                            "BlockScaleDequantizeNegScale",
                            FailurePoint::NONE,
                            true,
                            true},
        IntegrationTestCase{hipdnn_tests::plugin_constants::testExecuteFailsPluginPath(),
                            "ExecuteFailsPlugin",
                            "ExecuteFailsBlockScaleDequantize",
                            FailurePoint::EXECUTE,
                            true,
                            false},
        IntegrationTestCase{hipdnn_tests::plugin_constants::testNoApplicableEnginesAPluginPath(),
                            "NoApplicableEnginesPlugin",
                            "NoEnginesBlockScaleDequantize",
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

TEST_P(IntegrationBlockScaleDequantizeFp32, ExecutePluginPipeline)
{
    runTest();
}
