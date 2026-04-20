// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <cmath>
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

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
    bool useMeanInvVar;

    friend std::ostream& operator<<(std::ostream& os, const IntegrationTestCase& tc)
    {
        os << "LayernormBackwardTestCase{" << "plugin_path: " << tc.pluginPath
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
           << ", use_mean_inv_var: " << (tc.useMeanInvVar ? "true" : "false") << "}";

        return os;
    }
};

class IntegrationLayernormBackwardFp32 : public ::testing::TestWithParam<IntegrationTestCase>
{
protected:
    template <typename InputType, typename ScaleBiasType>
    struct SimpleLayernormTensorBundle
    {
        SimpleLayernormTensorBundle(const std::vector<int64_t>& dims)
            : normalizedDims(dims.begin() + 1, dims.end())
            , statsDims(makeStatsDims(dims))
            , dyTensor(Tensor<InputType>(dims))
            , xTensor(Tensor<InputType>(dims))
            , scaleTensor(Tensor<ScaleBiasType>(normalizedDims))
            , epsilonTensor(Tensor<ScaleBiasType>({1}))
            , meanTensor(Tensor<ScaleBiasType>(statsDims))
            , invVarianceTensor(Tensor<ScaleBiasType>(statsDims))
            , dxTensor(Tensor<InputType>(dims))
            , dscaleTensor(Tensor<InputType>(dims))
            , dbiasTensor(Tensor<InputType>(dims))
        {
            dyTensor.fillWithValue(static_cast<InputType>(1.0f));
            xTensor.fillWithValue(static_cast<InputType>(1.0f));
            scaleTensor.fillWithValue(static_cast<ScaleBiasType>(1.0f));
            epsilonTensor.fillWithValue(static_cast<ScaleBiasType>(1e-5f));
            meanTensor.fillWithValue(static_cast<ScaleBiasType>(0.0f));
            invVarianceTensor.fillWithValue(static_cast<ScaleBiasType>(0.0f));
            dxTensor.fillWithValue(static_cast<InputType>(0.0f));
            dscaleTensor.fillWithValue(static_cast<InputType>(0.0f));
            dbiasTensor.fillWithValue(static_cast<InputType>(0.0f));
        }

        std::vector<int64_t> normalizedDims;
        std::vector<int64_t> statsDims;
        Tensor<InputType> dyTensor;
        Tensor<InputType> xTensor;
        Tensor<ScaleBiasType> scaleTensor;
        Tensor<ScaleBiasType> epsilonTensor;
        Tensor<ScaleBiasType> meanTensor;
        Tensor<ScaleBiasType> invVarianceTensor;
        Tensor<InputType> dxTensor;
        Tensor<InputType> dscaleTensor;
        Tensor<InputType> dbiasTensor;

    private:
        static std::vector<int64_t> makeStatsDims(const std::vector<int64_t>& dims)
        {
            std::vector<int64_t> result(dims.size(), 1);
            result[0] = dims[0];
            return result;
        }
    };

    struct LayernormTestTensors
    {
        std::shared_ptr<TensorAttributes> dy;
        std::shared_ptr<TensorAttributes> x;
        std::shared_ptr<TensorAttributes> scale;
        std::shared_ptr<TensorAttributes> epsilon;
        std::shared_ptr<TensorAttributes> mean;
        std::shared_ptr<TensorAttributes> invVariance;
        std::shared_ptr<TensorAttributes> dx;
        std::shared_ptr<TensorAttributes> dscale;
        std::shared_ptr<TensorAttributes> dbias;
    };

    void SetUp() override
    {
        SKIP_IF_NO_DEVICES();

        ASSERT_EQ(hipInit(0), hipSuccess);
        int deviceId = 0;
        ASSERT_EQ(hipGetDevice(&deviceId), hipSuccess);
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
        const std::array<const char*, 1> paths = {pluginPath.c_str()};
        EXPECT_EQ(hipdnnSetEnginePluginPaths_ext(
                      paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE),
                  HIPDNN_STATUS_SUCCESS);

        hipdnnHandle_t handle = nullptr;
        EXPECT_EQ(hipdnnCreate(&handle), HIPDNN_STATUS_SUCCESS);

        return handle;
    }

    static std::pair<std::shared_ptr<Graph>, LayernormTestTensors> createLayernormTestGraphWithUids(
        const std::string& graphName,
        const SimpleLayernormTensorBundle<float, float>& tensorBundle,
        bool useManualUids,
        bool useMeanInvVar)
    {
        auto graph = std::make_shared<hipdnn_frontend::graph::Graph>();
        graph->set_name(graphName)
            .set_io_data_type(DataType::FLOAT)
            .set_intermediate_data_type(DataType::FLOAT)
            .set_compute_data_type(DataType::FLOAT);

        int64_t uid = 1;
        LayernormTestTensors tensors;

        auto dyAttr = makeTensorAttributes("DY", DataType::FLOAT, tensorBundle.dyTensor);
        if(useManualUids)
        {
            dyAttr.set_uid(uid++);
        }
        tensors.dy = std::make_shared<TensorAttributes>(std::move(dyAttr));

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

        auto epsilonAttr = makeTensorAttributes("epsilon", 1e-5f);
        tensors.epsilon = std::make_shared<TensorAttributes>(std::move(epsilonAttr));
        if(useManualUids)
        {
            tensors.epsilon->set_uid(uid++);
        }

        LayernormBackwardAttributes lnAttrs;
        lnAttrs.set_name("layernorm_bwd");
        lnAttrs.set_epsilon(tensors.epsilon);
        lnAttrs.set_compute_data_type(DataType::FLOAT);

        if(useMeanInvVar)
        {
            auto meanAttr = makeTensorAttributes("mean", DataType::FLOAT, tensorBundle.meanTensor);
            if(useManualUids)
            {
                meanAttr.set_uid(uid++);
            }
            tensors.mean = std::make_shared<TensorAttributes>(std::move(meanAttr));
            lnAttrs.set_mean(std::move(tensors.mean));

            auto invVarianceAttr = makeTensorAttributes(
                "inv_variance", DataType::FLOAT, tensorBundle.invVarianceTensor);
            if(useManualUids)
            {
                invVarianceAttr.set_uid(uid++);
            }
            tensors.invVariance = std::make_shared<TensorAttributes>(std::move(invVarianceAttr));
            lnAttrs.set_inv_variance(std::move(tensors.invVariance));
        }

        auto outputTensors
            = graph->layernorm_backward(tensors.dy, tensors.x, tensors.scale, lnAttrs);

        tensors.dx = outputTensors[0];
        tensors.dscale = outputTensors[1];
        tensors.dbias = outputTensors[2];

        if(useManualUids)
        {
            tensors.dx->set_uid(uid++);
            tensors.dscale->set_uid(uid++);
            tensors.dbias->set_uid(uid++);
        }
        tensors.dx->set_data_type(DataType::FLOAT);
        tensors.dscale->set_data_type(DataType::FLOAT);
        tensors.dbias->set_data_type(DataType::FLOAT);

        return {graph, tensors};
    }

    static std::unordered_map<int64_t, void*>
        createVariantPack(const LayernormTestTensors& tensors,
                          SimpleLayernormTensorBundle<float, float>& tensorBundle)
    {
        std::unordered_map<int64_t, void*> variantPack;
        variantPack[tensors.dy->get_uid()] = tensorBundle.dyTensor.memory().deviceData();
        variantPack[tensors.x->get_uid()] = tensorBundle.xTensor.memory().deviceData();
        variantPack[tensors.scale->get_uid()] = tensorBundle.scaleTensor.memory().deviceData();
        variantPack[tensors.epsilon->get_uid()] = tensorBundle.epsilonTensor.memory().deviceData();
        variantPack[tensors.dy->get_uid()] = tensorBundle.dyTensor.memory().deviceData();
        variantPack[tensors.dscale->get_uid()] = tensorBundle.dscaleTensor.memory().deviceData();
        variantPack[tensors.dbias->get_uid()] = tensorBundle.dbiasTensor.memory().deviceData();

        if(tensors.mean != nullptr)
        {
            variantPack[tensors.mean->get_uid()] = tensorBundle.meanTensor.memory().deviceData();
        }
        if(tensors.invVariance != nullptr)
        {
            variantPack[tensors.invVariance->get_uid()]
                = tensorBundle.invVarianceTensor.memory().deviceData();
        }

        return variantPack;
    }

    static void runGraphPipeline(const std::shared_ptr<Graph>& graph,
                                 hipdnnHandle_t handle,
                                 const LayernormTestTensors& tensors,
                                 SimpleLayernormTensorBundle<float, float>& tensorBundle,
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

        ASSERT_TRUE(tensors.dy->has_uid());
        ASSERT_TRUE(tensors.x->has_uid());
        ASSERT_TRUE(tensors.scale->has_uid());
        ASSERT_TRUE(tensors.epsilon->has_uid());
        ASSERT_TRUE(tensors.dx->has_uid());
        ASSERT_TRUE(tensors.dscale->has_uid());
        ASSERT_TRUE(tensors.dbias->has_uid());
        if(tensors.mean)
        {
            ASSERT_TRUE(tensors.mean->has_uid());
        }
        if(tensors.invVariance)
        {
            ASSERT_TRUE(tensors.invVariance->has_uid());
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

        _handle = setupEnvironmentWithPlugin(testCase.pluginPath);

        const std::vector<int64_t> dims = {2, 3, 14, 14};
        SimpleLayernormTensorBundle<float, float> tensorBundle(dims);

        auto [graph, tensors] = createLayernormTestGraphWithUids(
            testCase.graphName, tensorBundle, testCase.useManualUids, testCase.useMeanInvVar);

        runGraphPipeline(graph, _handle, tensors, tensorBundle, testCase.expectedFailure);
    }

private:
    hipdnnHandle_t _handle = nullptr;
};

} // namespace

INSTANTIATE_TEST_SUITE_P(
    ,
    IntegrationLayernormBackwardFp32,
    ::testing::Values(
        IntegrationTestCase{hipdnn_tests::plugin_constants::testGoodPluginPath(),
                            "ManualUids",
                            "DefaultPluginLayernormTest",
                            FailurePoint::NONE,
                            true,
                            false},
        IntegrationTestCase{hipdnn_tests::plugin_constants::testGoodPluginPath(),
                            "AutoUids",
                            "DefaultPluginLayernormTestAutoUID",
                            FailurePoint::NONE,
                            false,
                            false},
        IntegrationTestCase{hipdnn_tests::plugin_constants::testGoodPluginPath(),
                            "ManualUidsWithMeanInvVar",
                            "MeanInvVarPluginLayernormTest",
                            FailurePoint::NONE,
                            true,
                            true},
        IntegrationTestCase{hipdnn_tests::plugin_constants::testGoodPluginPath(),
                            "AutoUidsWithMeanInvVar",
                            "MeanInvVarPluginLayernormTestAutoUID",
                            FailurePoint::NONE,
                            false,
                            true},
        IntegrationTestCase{hipdnn_tests::plugin_constants::testExecuteFailsPluginPath(),
                            "ExecuteFailsPlugin",
                            "ExecuteFailsPluginLayernormTest",
                            FailurePoint::EXECUTE,
                            true,
                            true},
        IntegrationTestCase{hipdnn_tests::plugin_constants::testNoApplicableEnginesAPluginPath(),
                            "NoApplicableEnginesPlugin",
                            "NoEnginesPluginLayernormTest",
                            FailurePoint::CREATE_EXECUTION_PLAN,
                            true,
                            true}),
    [](const ::testing::TestParamInfo<IntegrationTestCase>& info) {
        std::string name = info.param.description;
        std::transform(name.cbegin(), name.cend(), name.begin(), [](char c) {
            return std::isalnum(c) ? c : '_';
        });
        return name;
    });

TEST_P(IntegrationLayernormBackwardFp32, ExecutePluginPipeline)
{
    runTest();
}
