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
#include <hipdnn_sdk/utilities/Workspace.hpp>

#include "test_plugins/TestPluginConstants.hpp"

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
        os << "ConvTestCase{" << "plugin_path: " << tc.pluginPath
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
template <typename Data_type>
class IntegrationConvolutionForward : public ::testing::TestWithParam<IntegrationTestCase>
{
protected:
    // Simplified tensor bundle for frontend integration tests
    struct SimpleConvolutionTensorBundle
    {
        SimpleConvolutionTensorBundle(const std::vector<int64_t>& xDims,
                                      const std::vector<int64_t>& wDims,
                                      const std::vector<int64_t>& yDims)
            : xTensor(Tensor<Data_type>(xDims))
            , wTensor(Tensor<Data_type>(wDims))
            , yTensor(Tensor<Data_type>(yDims))
        {
            // Initialize with simple constant values
            xTensor.fillWithValue(static_cast<Data_type>(1.0));
            wTensor.fillWithValue(static_cast<Data_type>(1.0));
            yTensor.fillWithValue(static_cast<Data_type>(0.0));
        }

        Tensor<Data_type> xTensor;
        Tensor<Data_type> wTensor;
        Tensor<Data_type> yTensor;
    };

    struct ConvolutionTestTensors
    {
        std::shared_ptr<TensorAttributes> x;
        std::shared_ptr<TensorAttributes> w;
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

    static std::pair<std::shared_ptr<Graph>, ConvolutionTestTensors>
        createConvTestGraphWithUids(const std::string& graphName,
                                    const SimpleConvolutionTensorBundle& tensorBundle,
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

        auto xAttr
            = makeTensorAttributes("X", getDataTypeEnumFromType<Data_type>(), tensorBundle.xTensor);
        if(useManualUids)
        {
            xAttr.set_uid(uid++);
        }
        tensors.x = std::make_shared<TensorAttributes>(std::move(xAttr));

        auto wAttr
            = makeTensorAttributes("W", getDataTypeEnumFromType<Data_type>(), tensorBundle.wTensor);
        if(useManualUids)
        {
            wAttr.set_uid(uid++);
        }
        tensors.w = std::make_shared<TensorAttributes>(std::move(wAttr));

        ConvFpropAttributes convAttrs;
        convAttrs.set_name("convolution_forward");
        convAttrs.set_pre_padding(prePadding);
        convAttrs.set_post_padding(postPadding);
        convAttrs.set_stride(stride);
        convAttrs.set_dilation(dilation);

        tensors.y = graph->conv_fprop(tensors.x, tensors.w, convAttrs);

        if(useManualUids)
        {
            tensors.y->set_uid(uid++);
        }
        tensors.y->set_output(true);

        return {graph, tensors};
    }

    static std::unordered_map<int64_t, void*>
        createVariantPack(const ConvolutionTestTensors& tensors,
                          SimpleConvolutionTensorBundle& tensorBundle)
    {
        std::unordered_map<int64_t, void*> variantPack;
        variantPack[tensors.x->get_uid()] = tensorBundle.xTensor.memory().deviceData();
        variantPack[tensors.w->get_uid()] = tensorBundle.wTensor.memory().deviceData();
        variantPack[tensors.y->get_uid()] = tensorBundle.yTensor.memory().deviceData();

        return variantPack;
    }

    static void runGraphPipeline(const std::shared_ptr<Graph>& graph,
                                 hipdnnHandle_t handle,
                                 const ConvolutionTestTensors& tensors,
                                 SimpleConvolutionTensorBundle& tensorBundle,
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

        int64_t workspaceSize;
        result = graph->get_workspace_size(workspaceSize);
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
        ASSERT_GE(workspaceSize, 0) << result.err_msg;
        Workspace workspace(static_cast<size_t>(workspaceSize));

        ASSERT_TRUE(tensors.x->has_uid());
        ASSERT_TRUE(tensors.w->has_uid());
        ASSERT_TRUE(tensors.y->has_uid());

        auto variantPack = createVariantPack(tensors, tensorBundle);

        result = graph->execute(handle, variantPack, workspace.get());
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

        std::vector<int64_t> xDims = {2, 3, 14, 14}; // n, c, h, w
        std::vector<int64_t> wDims = {3, 3, 3, 3}; // k, c, h, w
        std::vector<int64_t> yDims = {2, 3, 12, 12}; // n, k, h, w
        std::vector<int64_t> prePadding = {0, 0};
        std::vector<int64_t> postPadding = {0, 0};
        std::vector<int64_t> stride = {1, 1};
        std::vector<int64_t> dilation = {1, 1};

        // Setup environment with specified plugin
        _handle = setupEnvironmentWithPlugin(testCase.pluginPath);

        // Create tensor bundle
        SimpleConvolutionTensorBundle tensorBundle(xDims, wDims, yDims);

        // Create graph and tensors using the unified function
        auto [graph, tensors] = createConvTestGraphWithUids(testCase.graphName,
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

class IntegrationConvolutionForwardFp32 : public IntegrationConvolutionForward<float>
{
};

TEST_P(IntegrationConvolutionForwardFp32, ExecutePluginPipeline)
{
    runTest();
}

INSTANTIATE_TEST_SUITE_P(
    ,
    IntegrationConvolutionForwardFp32,
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
            return std::isalnum(static_cast<unsigned char>(c)) ? c : '_';
        });
        return name;
    });
