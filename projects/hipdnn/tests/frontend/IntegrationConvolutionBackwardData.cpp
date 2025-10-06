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
        os << "ConvolutionBackwardDataTestCase{" << "plugin_path: " << tc.pluginPath
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
//        - We are using a convolution backward data graph to test the conv_dgrad operation.
class IntegrationConvolutionBackwardDataFp32 : public ::testing::TestWithParam<IntegrationTestCase>
{
protected:
    // Simplified tensor bundle for frontend integration tests
    template <typename Input_type>
    struct SimpleConvolution2DTensorBundle
    {
        SimpleConvolution2DTensorBundle(const std::vector<int64_t>& inputDims,
                                        const std::vector<int64_t>& filterDims,
                                        const std::vector<int64_t>& outputDims)
            : dyTensor(Tensor<Input_type>(outputDims))
            , wTensor(Tensor<Input_type>(filterDims))
            , dxTensor(Tensor<Input_type>(inputDims))
        {
            // Initialize with simple constant values
            dyTensor.fillWithValue(static_cast<Input_type>(1.0f));
            wTensor.fillWithValue(static_cast<Input_type>(1.0f));
            dxTensor.fillWithValue(static_cast<Input_type>(0.0f));
        }

        Tensor<Input_type> dyTensor; // Gradient of output
        Tensor<Input_type> wTensor; // Weights/filter tensor
        Tensor<Input_type> dxTensor; // Gradient of input (output)
    };

    struct ConvolutionTestTensors
    {
        std::shared_ptr<TensorAttributes> dy;
        std::shared_ptr<TensorAttributes> w;
        std::shared_ptr<TensorAttributes> dx;
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

        // Load specific plugin
        const std::array<const char*, 1> paths = {pluginPath.c_str()};
        EXPECT_EQ(hipdnnSetEnginePluginPaths_ext(
                      paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE),
                  HIPDNN_STATUS_SUCCESS);

        hipdnnHandle_t handle = nullptr;
        EXPECT_EQ(hipdnnCreate(&handle), HIPDNN_STATUS_SUCCESS);

        return handle;
    }

    static std::pair<std::shared_ptr<Graph>, ConvolutionTestTensors>
        createConvolutionBackwardDataGraphWithUids(
            const std::string& graphName,
            const SimpleConvolution2DTensorBundle<float>& tensorBundle,
            const std::vector<int64_t>& prePadding,
            const std::vector<int64_t>& postPadding,
            const std::vector<int64_t>& stride,
            const std::vector<int64_t>& dilation,
            bool useManualUids)
    {
        auto graph = std::make_shared<hipdnn_frontend::graph::Graph>();
        graph->set_name(graphName);

        int64_t uid = 1;
        ConvolutionTestTensors tensors;

        // dy tensor (gradient of output)
        auto dyAttr = makeTensorAttributes("DY", DataType_t::FLOAT, tensorBundle.dyTensor);
        if(useManualUids)
        {
            dyAttr.set_uid(uid++);
        }
        tensors.dy = std::make_shared<TensorAttributes>(std::move(dyAttr));

        // weights/filter tensor
        auto wAttr = makeTensorAttributes("W", DataType_t::FLOAT, tensorBundle.wTensor);
        if(useManualUids)
        {
            wAttr.set_uid(uid++);
        }
        tensors.w = std::make_shared<TensorAttributes>(std::move(wAttr));

        ConvDgradAttributes convAttrs;
        convAttrs.set_name("convolution_backward_data");
        convAttrs.set_pre_padding(prePadding);
        convAttrs.set_post_padding(postPadding);
        convAttrs.set_stride(stride);
        convAttrs.set_dilation(dilation);
        convAttrs.set_convolution_mode(ConvolutionMode::CROSS_CORRELATION);

        // Create the convolution backward data operation
        tensors.dx = graph->conv_dgrad(tensors.dy, tensors.w, convAttrs);

        if(useManualUids)
        {
            tensors.dx->set_uid(uid++);
        }
        tensors.dx->set_data_type(DataType_t::FLOAT);

        return {graph, tensors};
    }

    static std::unordered_map<int64_t, void*>
        createVariantPack(const ConvolutionTestTensors& tensors,
                          SimpleConvolution2DTensorBundle<float>& tensorBundle)
    {
        std::unordered_map<int64_t, void*> variantPack;
        variantPack[tensors.dy->get_uid()] = tensorBundle.dyTensor.memory().deviceData();
        variantPack[tensors.w->get_uid()] = tensorBundle.wTensor.memory().deviceData();
        variantPack[tensors.dx->get_uid()] = tensorBundle.dxTensor.memory().deviceData();

        return variantPack;
    }

    static void runGraphPipeline(const std::shared_ptr<Graph>& graph,
                                 hipdnnHandle_t handle,
                                 const ConvolutionTestTensors& tensors,
                                 SimpleConvolution2DTensorBundle<float>& tensorBundle,
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

        ASSERT_TRUE(tensors.dy->has_uid());
        ASSERT_TRUE(tensors.w->has_uid());
        ASSERT_TRUE(tensors.dx->has_uid());

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

        std::vector<int64_t> inputDims = {2, 3, 14, 14}; // DX dimensions
        std::vector<int64_t> filterDims = {3, 3, 3, 3}; // W dimensions (K, C, R, S)
        std::vector<int64_t> outputDims = {2, 3, 12, 12}; // DY dimensions

        // Input: N=2, C=3, H=14, W=14
        // Filter: K=3, C=3, R=3, S=3 (output channels = input channels for backward data)
        // Output (DY): N=2, K=3, P=12, Q=12 (calculated based on conv params)
        std::vector<int64_t> prePadding = {0, 0};
        std::vector<int64_t> postPadding = {0, 0};
        std::vector<int64_t> stride = {1, 1};
        std::vector<int64_t> dilation = {1, 1};

        SimpleConvolution2DTensorBundle<float> tensorBundle(inputDims, filterDims, outputDims);

        // Create graph and tensors with the convolution parameters
        auto [graph, tensors] = createConvolutionBackwardDataGraphWithUids(testCase.graphName,
                                                                           tensorBundle,
                                                                           prePadding,
                                                                           postPadding,
                                                                           stride,
                                                                           dilation,
                                                                           testCase.useManualUids);

        runGraphPipeline(graph, _handle, tensors, tensorBundle, testCase.expectedFailure);
    }

private:
    hipdnnHandle_t _handle = nullptr;
};

} // namespace

INSTANTIATE_TEST_SUITE_P(
    ,
    IntegrationConvolutionBackwardDataFp32,
    ::testing::Values(
        IntegrationTestCase{hipdnn_tests::plugin_constants::testGoodPluginPath(),
                            "DefaultPluginWithManualUids",
                            "DefaultPluginConvBackwardDataTest",
                            FailurePoint::NONE,
                            true},
        IntegrationTestCase{hipdnn_tests::plugin_constants::testGoodPluginPath(),
                            "DefaultPluginWithAutoUids",
                            "DefaultPluginConvBackwardDataTestAutoUID",
                            FailurePoint::NONE,
                            false},
        IntegrationTestCase{hipdnn_tests::plugin_constants::testExecuteFailsPluginPath(),
                            "ExecuteFailsPlugin",
                            "ExecuteFailsPluginConvBackwardDataTest",
                            FailurePoint::EXECUTE,
                            true},
        IntegrationTestCase{hipdnn_tests::plugin_constants::testNoApplicableEnginesAPluginPath(),
                            "NoApplicableEnginesPlugin",
                            "NoEnginesPluginConvBackwardDataTest",
                            FailurePoint::CREATE_EXECUTION_PLAN,
                            true}),
    // Provide a custom name for each test instance (C++17 compatible)
    [](const ::testing::TestParamInfo<IntegrationTestCase>& info) {
        std::string name = info.param.description;
        std::transform(name.cbegin(), name.cend(), name.begin(), [](char c) {
            return std::isalnum(static_cast<unsigned char>(c)) ? c : '_';
        });
        return name;
    });

TEST_P(IntegrationConvolutionBackwardDataFp32, ExecutePluginPipeline)
{
    runTest();
}
