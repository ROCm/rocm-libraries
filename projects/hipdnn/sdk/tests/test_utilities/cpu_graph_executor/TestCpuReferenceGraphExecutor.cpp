// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/Utilities.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>

#include "BatchnormGraphUtils.hpp"
#include "BatchnormTensorBundles.hpp"
#include "ConvolutionGraphUtils.hpp"
#include "PointwiseGraphUtils.hpp"
#include "PointwiseTensorBundles.hpp"

#include <hipdnn_sdk/plugin/EnginePluginApi.h>
#include <hipdnn_sdk/plugin/PluginApiDataTypes.h>
#include <hipdnn_sdk/plugin/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_sdk/test_utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_sdk/test_utilities/Seeds.hpp>
#include <hipdnn_sdk/test_utilities/cpu_graph_executor/CpuReferenceGraphExecutor.hpp>
#include <hipdnn_sdk/utilities/ShallowTensor.hpp>
#include <hipdnn_sdk/utilities/Tensor.hpp>
#include <hipdnn_sdk/utilities/UtilsBfp16.hpp>
#include <hipdnn_sdk/utilities/UtilsFp16.hpp>

using namespace hipdnn_sdk::test_utilities;
using namespace hipdnn_sdk::data_objects;
using namespace hipdnn_sdk::utilities;
using namespace ::testing;
using namespace hipdnn_sdk_test_utils;
using namespace hipdnn_plugin;

class TestCpuReferenceGraphExecutor
{
public:
    static void runBatchnormFwdTest(hipdnn_sdk::data_objects::DataType inputDataType,
                                    hipdnn_sdk::data_objects::DataType scaleBiasDataType,
                                    hipdnn_sdk::data_objects::DataType meanVarianceDataType)
    {
        unsigned int seed = getGlobalTestSeed();

        std::vector<int64_t> dims = {1, 3, 14, 14};
        auto graph = buildBatchnormFwdInferenceGraph(
            inputDataType, scaleBiasDataType, meanVarianceDataType, dims, TensorLayout::NCHW, true);

        auto result = graph->validate();
        ASSERT_EQ(result.code, hipdnn_frontend::ErrorCode::OK) << result.err_msg;

        auto flatbufferGraph = graph->buildFlatbufferOperationGraph();
        GraphWrapper graphWrapper(flatbufferGraph.data(), flatbufferGraph.size());

        BatchnormFwdTensorBundle tensorBundle(
            graphWrapper.getNodeWrapper(0), graphWrapper.getTensorMap(), seed);

        auto variantPack = tensorBundle.toHostVariantPack();

        hipdnn_sdk::test_utilities::CpuReferenceGraphExecutor().execute(
            flatbufferGraph.data(), flatbufferGraph.size(), variantPack);
    }

    template <typename InputType, typename ScaleBiasType, typename MeanVarianceType>
    static void runBatchnormBwdTest(hipdnn_sdk::data_objects::DataType inputDataType,
                                    hipdnn_sdk::data_objects::DataType scaleBiasDataType,
                                    hipdnn_sdk::data_objects::DataType meanVarianceDataType)
    {
        std::vector<int64_t> dims = {1, 3, 14, 14};
        BatchnormBwdTensorBundle<InputType, ScaleBiasType, MeanVarianceType> tensorBundle(
            dims, 1, TensorLayout::NCHW);

        auto graphTuple = buildBatchnormBwdGraph(
            tensorBundle, inputDataType, scaleBiasDataType, meanVarianceDataType);

        auto& graph = std::get<0>(graphTuple);
        auto& variantPack = std::get<1>(graphTuple);

        auto result = graph->validate();
        ASSERT_EQ(result.code, hipdnn_frontend::ErrorCode::OK) << result.err_msg;

        auto flatbufferGraph = graph->buildFlatbufferOperationGraph();

        hipdnn_sdk::test_utilities::CpuReferenceGraphExecutor().execute(
            flatbufferGraph.data(), flatbufferGraph.size(), variantPack);
    }

    template <typename InputType, typename ScaleBiasType, typename MeanVarianceType>
    static void runBatchnormTrainTest(hipdnn_sdk::data_objects::DataType inputDataType,
                                      hipdnn_sdk::data_objects::DataType scaleBiasDataType,
                                      hipdnn_sdk::data_objects::DataType meanVarianceDataType,
                                      bool useOptionalTensors = false)
    {
        std::vector<int64_t> dims = {1, 3, 14, 14};
        BatchnormTrainTensorBundle<InputType, ScaleBiasType, MeanVarianceType> tensorBundle(
            dims, 1, TensorLayout::NCHW, useOptionalTensors);

        auto graphTuple = buildBatchnormTrainGraph(tensorBundle,
                                                   inputDataType,
                                                   scaleBiasDataType,
                                                   meanVarianceDataType,
                                                   useOptionalTensors);

        auto& graph = std::get<0>(graphTuple);
        auto& variantPack = std::get<1>(graphTuple);

        auto result = graph->validate();
        ASSERT_EQ(result.code, hipdnn_frontend::ErrorCode::OK) << result.err_msg;

        auto flatbufferGraph = graph->buildFlatbufferOperationGraph();

        hipdnn_sdk::test_utilities::CpuReferenceGraphExecutor().execute(
            flatbufferGraph.data(), flatbufferGraph.size(), variantPack);
    }

    template <typename InputType, typename AccumulatorType>
    static void runConvolutionFwdTest(hipdnn_sdk::data_objects::DataType inputDataType,
                                      hipdnn_sdk::data_objects::DataType accumulatorDataType)
    {
        std::vector<int64_t> xDims = {1, 1, 2, 2};
        std::vector<int64_t> wDims = {1, 1, 1, 1};
        std::vector<int64_t> yDims = {1, 1, 2, 2};
        ConvolutionFwdTensorBundle<InputType> tensorBundle(
            xDims, wDims, yDims, 1, TensorLayout::NCHW);

        auto graphTuple
            = buildConvolutionFwdGraph(tensorBundle, inputDataType, accumulatorDataType);

        auto& graph = std::get<0>(graphTuple);
        auto& variantPack = std::get<1>(graphTuple);

        auto result = graph->validate();
        ASSERT_EQ(result.code, hipdnn_frontend::ErrorCode::OK) << result.err_msg;

        auto flatbufferGraph = graph->buildFlatbufferOperationGraph();

        hipdnn_sdk::test_utilities::CpuReferenceGraphExecutor().execute(
            flatbufferGraph.data(), flatbufferGraph.size(), variantPack);
    }

    template <typename InputType, typename AccumulatorType>
    static void runConvolutionBwdTest(hipdnn_sdk::data_objects::DataType inputDataType,
                                      hipdnn_sdk::data_objects::DataType accumulatorDataType)
    {
        std::vector<int64_t> dxDims = {1, 1, 2, 2};
        std::vector<int64_t> wDims = {1, 1, 1, 1};
        std::vector<int64_t> dyDims = {1, 1, 2, 2};
        ConvolutionBwdTensorBundle<InputType> tensorBundle(
            dxDims, wDims, dyDims, 1, TensorLayout::NCHW);

        auto graphTuple
            = buildConvolutionBwdGraph(tensorBundle, inputDataType, accumulatorDataType);

        auto& graph = std::get<0>(graphTuple);
        auto& variantPack = std::get<1>(graphTuple);

        auto result = graph->validate();
        ASSERT_EQ(result.code, hipdnn_frontend::ErrorCode::OK) << result.err_msg;

        auto flatbufferGraph = graph->buildFlatbufferOperationGraph();

        hipdnn_sdk::test_utilities::CpuReferenceGraphExecutor().execute(
            flatbufferGraph.data(), flatbufferGraph.size(), variantPack);
    }
};

TEST(TestCpuReferenceGraphExecutor, BatchnormFwdInferenceAllFloats)
{
    TestCpuReferenceGraphExecutor::runBatchnormFwdTest(
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT);
}

TEST(TestCpuReferenceGraphExecutor, BatchnormFwdInferenceAllHalfs)
{
    TestCpuReferenceGraphExecutor::runBatchnormFwdTest(
        DataType::HALF, DataType::HALF, DataType::HALF);
}

TEST(TestCpuReferenceGraphExecutor, BatchnormFwdInferenceAllBFloats)
{
    TestCpuReferenceGraphExecutor::runBatchnormFwdTest(
        DataType::BFLOAT16, DataType::BFLOAT16, DataType::BFLOAT16);
}

TEST(TestCpuReferenceGraphExecutor, SignaturesThatDontExist)
{
    EXPECT_THROW((TestCpuReferenceGraphExecutor::runBatchnormFwdTest(
                     DataType::FLOAT, DataType::HALF, DataType::HALF)),
                 std::runtime_error);

    EXPECT_THROW((TestCpuReferenceGraphExecutor::runBatchnormFwdTest(
                     DataType::FLOAT, DataType::HALF, DataType::FLOAT)),
                 std::runtime_error);
}

TEST(TestCpuReferenceGraphExecutor, BatchnormBwdAllFloats)
{
    TestCpuReferenceGraphExecutor::runBatchnormBwdTest<float, float, float>(
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT);
}

TEST(TestCpuReferenceGraphExecutor, BatchnormBwdAllHalfs)
{
    TestCpuReferenceGraphExecutor::runBatchnormBwdTest<half, half, half>(
        DataType::HALF, DataType::HALF, DataType::HALF);
}

TEST(TestCpuReferenceGraphExecutor, BatchnormBwdAllBFloat16)
{
    TestCpuReferenceGraphExecutor::runBatchnormBwdTest<hip_bfloat16, hip_bfloat16, hip_bfloat16>(
        DataType::BFLOAT16, DataType::BFLOAT16, DataType::BFLOAT16);
}

TEST(TestCpuReferenceGraphExecutor, BatchnormTrainAllFloats)
{
    TestCpuReferenceGraphExecutor::runBatchnormTrainTest<float, float, float>(
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT);

    TestCpuReferenceGraphExecutor::runBatchnormTrainTest<float, float, float>(
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, true);
}

TEST(TestCpuReferenceGraphExecutor, BatchnormTrainAllHalfs)
{
    TestCpuReferenceGraphExecutor::runBatchnormTrainTest<half, half, half>(
        DataType::HALF, DataType::HALF, DataType::HALF);
}

TEST(TestCpuReferenceGraphExecutor, BatchnormTrainAllBFloat16)
{
    TestCpuReferenceGraphExecutor::runBatchnormTrainTest<hip_bfloat16, hip_bfloat16, hip_bfloat16>(
        DataType::BFLOAT16, DataType::BFLOAT16, DataType::BFLOAT16);
}

TEST(TestCpuReferenceGraphExecutor, ConvolutionFwdAllFloats)
{
    TestCpuReferenceGraphExecutor::runConvolutionFwdTest<float, float>(DataType::FLOAT,
                                                                       DataType::FLOAT);
}
TEST(TestCpuReferenceGraphExecutor, ConvolutionFwdAllHalfs)
{
    TestCpuReferenceGraphExecutor::runConvolutionFwdTest<half, float>(DataType::HALF,
                                                                      DataType::FLOAT);
}
TEST(TestCpuReferenceGraphExecutor, ConvolutionFwdAllBFloat16)
{
    TestCpuReferenceGraphExecutor::runConvolutionFwdTest<hip_bfloat16, float>(DataType::BFLOAT16,
                                                                              DataType::FLOAT);
}

TEST(TestCpuReferenceGraphExecutor, ConvolutionBwdAllFloats)
{
    TestCpuReferenceGraphExecutor::runConvolutionBwdTest<float, float>(DataType::FLOAT,
                                                                       DataType::FLOAT);
}
TEST(TestCpuReferenceGraphExecutor, ConvolutionBwdAllHalfs)
{
    TestCpuReferenceGraphExecutor::runConvolutionBwdTest<half, float>(DataType::HALF,
                                                                      DataType::FLOAT);
}
TEST(TestCpuReferenceGraphExecutor, ConvolutionBwdAllBFloat16)
{
    TestCpuReferenceGraphExecutor::runConvolutionBwdTest<hip_bfloat16, float>(DataType::BFLOAT16,
                                                                              DataType::FLOAT);
}

// Single-node pointwise operation tests
TEST(TestCpuReferenceGraphExecutor, PointwiseUnaryReluFwd)
{
    std::vector<int64_t> inputDims = {1, 3, 4, 4};
    std::vector<int64_t> outputDims = {1, 3, 4, 4};

    auto [graph, tensorBundle, variantPack]
        = buildPointwiseUnaryGraph(inputDims,
                                   outputDims,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   hipdnn_frontend::PointwiseMode::RELU_FWD,
                                   1,
                                   TensorLayout::NCHW);

    auto flatbufferGraph = graph->buildFlatbufferOperationGraph();
    CpuReferenceGraphExecutor().execute(
        flatbufferGraph.data(), flatbufferGraph.size(), variantPack);
}

TEST(TestCpuReferenceGraphExecutor, PointwiseBinaryAdd)
{
    std::vector<int64_t> inputDims = {1, 3, 2, 2};
    std::vector<int64_t> outputDims = {1, 3, 2, 2};

    auto [graph, tensorBundle, variantPack]
        = buildPointwiseBinaryGraph(inputDims,
                                    inputDims,
                                    outputDims,
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    hipdnn_frontend::PointwiseMode::ADD,
                                    1,
                                    TensorLayout::NCHW);

    auto flatbufferGraph = graph->buildFlatbufferOperationGraph();
    CpuReferenceGraphExecutor().execute(
        flatbufferGraph.data(), flatbufferGraph.size(), variantPack);
}
