// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <cctype>
#include <cmath>
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

#include <hipdnn_data_sdk/utilities/MigratableMemory.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_data_sdk/utilities/Workspace.hpp>
#include <hipdnn_frontend.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "test_plugins/TestPluginConstants.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using namespace hipdnn_data_sdk::utilities;

namespace
{

enum class FailurePoint
{
    NONE,
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

    friend std::ostream& operator<<(std::ostream& os, const IntegrationTestCase& tc)
    {
        os << "MatmulTestCase{ plugin_path: " << tc.pluginPath
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

        os << ", use_manual_uids: " << (tc.useManualUids ? "true" : "false") << " }";
        return os;
    }
};

template <typename Data_type>
class IntegrationMatmul : public ::testing::TestWithParam<IntegrationTestCase>
{
protected:
    struct SimpleMatmulTensorBundle
    {
        SimpleMatmulTensorBundle(const std::vector<int64_t>& aDims,
                                 const std::vector<int64_t>& bDims,
                                 const std::vector<int64_t>& cDims)
            : aTensor(Tensor<Data_type>(aDims))
            , bTensor(Tensor<Data_type>(bDims))
            , cTensor(Tensor<Data_type>(cDims))
        {
            aTensor.fillWithValue(static_cast<Data_type>(1.0));
            bTensor.fillWithValue(static_cast<Data_type>(1.0));
            cTensor.fillWithValue(static_cast<Data_type>(0.0));
        }

        Tensor<Data_type> aTensor;
        Tensor<Data_type> bTensor;
        Tensor<Data_type> cTensor;
    };

    struct MatmulTestTensors
    {
        std::shared_ptr<TensorAttributes> a;
        std::shared_ptr<TensorAttributes> b;
        std::shared_ptr<TensorAttributes> c;
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

    static std::pair<std::shared_ptr<Graph>, MatmulTestTensors> createMatmulTestGraphWithUids(
        const std::string& graphName, const SimpleMatmulTensorBundle& bundle, bool useManualUids)
    {
        auto graph = std::make_shared<hipdnn_frontend::graph::Graph>();
        graph->set_name(graphName)
            .set_io_data_type(DataType::FLOAT)
            .set_intermediate_data_type(DataType::FLOAT)
            .set_compute_data_type(DataType::FLOAT);

        int64_t uid = 1;
        MatmulTestTensors tensors;

        auto aAttr
            = makeTensorAttributes("A", getDataTypeEnumFromType<Data_type>(), bundle.aTensor);
        if(useManualUids)
        {
            aAttr.set_uid(uid++);
        }
        tensors.a = std::make_shared<TensorAttributes>(std::move(aAttr));

        auto bAttr
            = makeTensorAttributes("B", getDataTypeEnumFromType<Data_type>(), bundle.bTensor);
        if(useManualUids)
        {
            bAttr.set_uid(uid++);
        }
        tensors.b = std::make_shared<TensorAttributes>(std::move(bAttr));

        MatmulAttributes matmulAttrs;
        matmulAttrs.set_name("matmul");

        tensors.c = graph->matmul(tensors.a, tensors.b, matmulAttrs);
        if(useManualUids)
        {
            tensors.c->set_uid(uid++);
        }
        tensors.c->set_output(true);

        return {graph, tensors};
    }

    static std::unordered_map<int64_t, void*> createVariantPack(const MatmulTestTensors& tensors,
                                                                SimpleMatmulTensorBundle& bundle)
    {
        std::unordered_map<int64_t, void*> vp;
        vp[tensors.a->get_uid()] = bundle.aTensor.memory().deviceData();
        vp[tensors.b->get_uid()] = bundle.bTensor.memory().deviceData();
        vp[tensors.c->get_uid()] = bundle.cTensor.memory().deviceData();
        return vp;
    }

    static void runGraphPipeline(const std::shared_ptr<Graph>& graph,
                                 hipdnnHandle_t handle,
                                 const MatmulTestTensors& tensors,
                                 SimpleMatmulTensorBundle& bundle,
                                 FailurePoint expectedFailure)
    {
        auto result = graph->validate();
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

        result = graph->build_operation_graph(handle);
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

        result = graph->create_execution_plans();
        if(expectedFailure == FailurePoint::CREATE_EXECUTION_PLAN)
        {
            ASSERT_NE(result.code, ErrorCode::OK);
            return;
        }
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

        result = graph->check_support();
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

        result = graph->build_plans();
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

        int64_t workspaceSize;
        result = graph->get_workspace_size(workspaceSize);
        ASSERT_EQ(result.code, ErrorCode::OK);
        ASSERT_GE(workspaceSize, 0);
        const Workspace workspace(static_cast<size_t>(workspaceSize));

        auto vp = createVariantPack(tensors, bundle);

        result = graph->execute(handle, vp, workspace.get());
        if(expectedFailure == FailurePoint::EXECUTE)
        {
            ASSERT_NE(result.code, ErrorCode::OK);
        }
        else
        {
            ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
        }
    }

    void runTest()
    {
        const auto& tc = GetParam();

        _handle = setupEnvironmentWithPlugin(tc.pluginPath);

        SimpleMatmulTensorBundle bundle({2, 3}, {3, 4}, {2, 4});

        auto [graph, tensors]
            = createMatmulTestGraphWithUids(tc.graphName, bundle, tc.useManualUids);

        runGraphPipeline(graph, _handle, tensors, bundle, tc.expectedFailure);
    }

private:
    hipdnnHandle_t _handle = nullptr;
};

} // namespace

class IntegrationMatmulFp32 : public IntegrationMatmul<float>
{
};

TEST_P(IntegrationMatmulFp32, ExecutePluginPipeline)
{
    runTest();
}

INSTANTIATE_TEST_SUITE_P(
    ,
    IntegrationMatmulFp32,
    ::testing::Values(
        IntegrationTestCase{hipdnn_tests::plugin_constants::testGoodPluginPath(),
                            "GoodPluginManualUids",
                            "MatmulTestManualUID",
                            FailurePoint::NONE,
                            true},
        IntegrationTestCase{hipdnn_tests::plugin_constants::testGoodPluginPath(),
                            "GoodPluginAutoUids",
                            "MatmulTestAutoUID",
                            FailurePoint::NONE,
                            false},
        IntegrationTestCase{hipdnn_tests::plugin_constants::testExecuteFailsPluginPath(),
                            "ExecuteFailsPlugin",
                            "MatmulTestExecuteFail",
                            FailurePoint::EXECUTE,
                            true},
        IntegrationTestCase{hipdnn_tests::plugin_constants::testNoApplicableEnginesAPluginPath(),
                            "NoEnginesPlugin",
                            "MatmulTestNoEngines",
                            FailurePoint::CREATE_EXECUTION_PLAN,
                            true}),
    [](const ::testing::TestParamInfo<IntegrationTestCase>& info) {
        std::string name = info.param.description;
        std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) {
            return std::isalnum(c) ? c : '_';
        });
        return name;
    });
