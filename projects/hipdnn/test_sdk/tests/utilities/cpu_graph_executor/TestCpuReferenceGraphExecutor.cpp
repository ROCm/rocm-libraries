// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/Utilities.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>

#include "BatchnormGraphUtils.hpp"
#include "BatchnormTensorBundles.hpp"
#include "ConvolutionGraphUtils.hpp"
#include "MatmulGraphUtils.hpp"
#include "PointwiseGraphUtils.hpp"
#include "PointwiseTensorBundles.hpp"

#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/ShallowTensor.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_data_sdk/utilities/TensorView.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/Seeds.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/CpuReferenceGraphExecutor.hpp>

using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_test_sdk::detail;
using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_data_sdk::utilities;
using namespace ::testing;
using namespace hipdnn_sdk_test_utils;
using namespace hipdnn_flatbuffers_sdk::flatbuffer_utilities;
using hipdnn_data_sdk::types::bfloat16;
using hipdnn_data_sdk::types::half;

class TestCpuReferenceGraphExecutor
{
public:
    static void
        runBatchnormFwdTest(hipdnn_flatbuffers_sdk::data_objects::DataType inputDataType,
                            hipdnn_flatbuffers_sdk::data_objects::DataType scaleBiasDataType,
                            hipdnn_flatbuffers_sdk::data_objects::DataType meanVarianceDataType,
                            hipdnn_flatbuffers_sdk::data_objects::DataType computeDataType)
    {
        const unsigned int seed = getGlobalTestSeed();

        const std::vector<int64_t> dims = {1, 3, 14, 14};
        auto graph = buildBatchnormFwdInferenceGraph(inputDataType,
                                                     scaleBiasDataType,
                                                     meanVarianceDataType,
                                                     computeDataType,
                                                     dims,
                                                     TensorLayout::NCHW,
                                                     true);

        auto result = graph->validate();
        ASSERT_EQ(result.code, hipdnn_frontend::ErrorCode::OK) << result.err_msg;

        auto [serializedGraph, serErr] = graph->to_binary();
        ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
        const GraphWrapper graphWrapper(serializedGraph.data(), serializedGraph.size());

        BatchnormFwdTensorBundle tensorBundle(
            graphWrapper.getNodeWrapper(0), graphWrapper.getTensorMap(), seed);

        auto variantPack = tensorBundle.toHostVariantPack();

        CpuReferenceGraphExecutor().execute(
            serializedGraph.data(), serializedGraph.size(), variantPack);
    }

    template <typename InputType,
              typename ScaleBiasType,
              typename MeanVarianceType,
              typename ComputeType>
    static void runBatchnormBwdTest()
    {
        auto inputDataType = nativeTypeToDataType<InputType>();
        auto scaleBiasDataType = nativeTypeToDataType<ScaleBiasType>();
        auto meanVarianceDataType = nativeTypeToDataType<MeanVarianceType>();
        auto computeDataType = nativeTypeToDataType<ComputeType>();

        const std::vector<int64_t> dims = {1, 3, 14, 14};
        BatchnormBwdTensorBundle<InputType, ScaleBiasType, MeanVarianceType> tensorBundle(
            dims, 1, TensorLayout::NCHW);

        auto graphTuple = buildBatchnormBwdGraph(
            tensorBundle, inputDataType, scaleBiasDataType, meanVarianceDataType, computeDataType);

        auto& graph = std::get<0>(graphTuple);
        auto& variantPack = std::get<1>(graphTuple);

        auto result = graph->validate();
        ASSERT_EQ(result.code, hipdnn_frontend::ErrorCode::OK) << result.err_msg;

        auto [serializedGraph, serErr] = graph->to_binary();
        ASSERT_TRUE(serErr.is_good()) << serErr.get_message();

        CpuReferenceGraphExecutor().execute(
            serializedGraph.data(), serializedGraph.size(), variantPack);
    }

    template <typename InputType,
              typename ScaleBiasType,
              typename MeanVarianceType,
              typename ComputeType>
    static void runBatchnormTrainTest(bool useOptionalTensors = false)
    {
        auto inputDataType = nativeTypeToDataType<InputType>();
        auto scaleBiasDataType = nativeTypeToDataType<ScaleBiasType>();
        auto meanVarianceDataType = nativeTypeToDataType<MeanVarianceType>();
        auto computeDataType = nativeTypeToDataType<ComputeType>();

        const std::vector<int64_t> dims = {1, 3, 14, 14};
        BatchnormTrainTensorBundle<InputType, ScaleBiasType, MeanVarianceType> tensorBundle(
            dims, 1, TensorLayout::NCHW, useOptionalTensors);

        auto graphTuple = buildBatchnormTrainGraph(tensorBundle,
                                                   inputDataType,
                                                   scaleBiasDataType,
                                                   meanVarianceDataType,
                                                   computeDataType,
                                                   useOptionalTensors);

        auto& graph = std::get<0>(graphTuple);
        auto& variantPack = std::get<1>(graphTuple);

        auto result = graph->validate();
        ASSERT_EQ(result.code, hipdnn_frontend::ErrorCode::OK) << result.err_msg;

        auto [serializedGraph, serErr] = graph->to_binary();
        ASSERT_TRUE(serErr.is_good()) << serErr.get_message();

        CpuReferenceGraphExecutor().execute(
            serializedGraph.data(), serializedGraph.size(), variantPack);
    }

    template <typename InputType, typename AccumulatorType>
    static void
        runConvolutionFwdTest(hipdnn_flatbuffers_sdk::data_objects::DataType inputDataType,
                              hipdnn_flatbuffers_sdk::data_objects::DataType accumulatorDataType)
    {
        const std::vector<int64_t> xDims = {1, 1, 2, 2};
        const std::vector<int64_t> wDims = {1, 1, 1, 1};
        const std::vector<int64_t> yDims = {1, 1, 2, 2};
        ConvolutionFwdTensorBundle<InputType> tensorBundle(
            xDims, wDims, yDims, 1, TensorLayout::NCHW);

        auto graphTuple
            = buildConvolutionFwdGraph(tensorBundle, inputDataType, accumulatorDataType);

        auto& graph = std::get<0>(graphTuple);
        auto& variantPack = std::get<1>(graphTuple);

        auto result = graph->validate();
        ASSERT_EQ(result.code, hipdnn_frontend::ErrorCode::OK) << result.err_msg;

        auto [serializedGraph, serErr] = graph->to_binary();
        ASSERT_TRUE(serErr.is_good()) << serErr.get_message();

        CpuReferenceGraphExecutor().execute(
            serializedGraph.data(), serializedGraph.size(), variantPack);
    }

    template <typename InputType, typename AccumulatorType>
    static void
        runConvolutionBwdTest(hipdnn_flatbuffers_sdk::data_objects::DataType inputDataType,
                              hipdnn_flatbuffers_sdk::data_objects::DataType accumulatorDataType)
    {
        const std::vector<int64_t> dxDims = {1, 1, 2, 2};
        const std::vector<int64_t> wDims = {1, 1, 1, 1};
        const std::vector<int64_t> dyDims = {1, 1, 2, 2};
        ConvolutionBwdTensorBundle<InputType> tensorBundle(
            dxDims, wDims, dyDims, 1, TensorLayout::NCHW);

        auto graphTuple
            = buildConvolutionBwdGraph(tensorBundle, inputDataType, accumulatorDataType);

        auto& graph = std::get<0>(graphTuple);
        auto& variantPack = std::get<1>(graphTuple);

        auto result = graph->validate();
        ASSERT_EQ(result.code, hipdnn_frontend::ErrorCode::OK) << result.err_msg;

        auto [serializedGraph, serErr] = graph->to_binary();
        ASSERT_TRUE(serErr.is_good()) << serErr.get_message();

        CpuReferenceGraphExecutor().execute(
            serializedGraph.data(), serializedGraph.size(), variantPack);
    }

    template <typename InputType, typename AccumulatorType>
    static void
        runConvolutionWrwTest(hipdnn_flatbuffers_sdk::data_objects::DataType inputDataType,
                              hipdnn_flatbuffers_sdk::data_objects::DataType accumulatorDataType)
    {
        const std::vector<int64_t> xDims = {1, 1, 2, 2};
        const std::vector<int64_t> dwDims = {1, 1, 1, 1};
        const std::vector<int64_t> dyDims = {1, 1, 2, 2};
        ConvolutionWrwTensorBundle<InputType> tensorBundle(
            xDims, dwDims, dyDims, 1, TensorLayout::NCHW);

        auto graphTuple
            = buildConvolutionWrwGraph(tensorBundle, inputDataType, accumulatorDataType);

        auto& graph = std::get<0>(graphTuple);
        auto& variantPack = std::get<1>(graphTuple);

        auto result = graph->validate();
        ASSERT_EQ(result.code, hipdnn_frontend::ErrorCode::OK) << result.err_msg;

        auto [serializedGraph, serErr] = graph->to_binary();
        ASSERT_TRUE(serErr.is_good()) << serErr.get_message();

        CpuReferenceGraphExecutor().execute(
            serializedGraph.data(), serializedGraph.size(), variantPack);
    }

    template <typename inputType, typename ComputeType>
    static void runMatmulTest(hipdnn_flatbuffers_sdk::data_objects::DataType inputDataType,
                              hipdnn_flatbuffers_sdk::data_objects::DataType computeDataType)
    {
        const std::vector<int64_t> aDims = {2, 5, 3};
        const std::vector<int64_t> bDims = {2, 3, 4};
        const std::vector<int64_t> cDims = {2, 5, 4};
        MatmulTensorBundle<inputType> tensorBundle(aDims, bDims, cDims, false, false, 1);

        auto graphTuple = buildMatmulGraph(tensorBundle, inputDataType, computeDataType);

        auto& graph = std::get<0>(graphTuple);
        auto& variantPack = std::get<1>(graphTuple);

        auto result = graph->validate();
        ASSERT_EQ(result.code, hipdnn_frontend::ErrorCode::OK) << result.err_msg;

        auto [serializedGraph, serErr] = graph->to_binary();
        ASSERT_TRUE(serErr.is_good()) << serErr.get_message();

        CpuReferenceGraphExecutor().execute(
            serializedGraph.data(), serializedGraph.size(), variantPack);
    }
};

TEST(TestCpuReferenceGraphExecutor, BatchnormFwdInferenceAllFloats)
{
    TestCpuReferenceGraphExecutor::runBatchnormFwdTest(
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT);
}

TEST(TestCpuReferenceGraphExecutor, BatchnormFwdInferenceAllHalfs)
{
    TestCpuReferenceGraphExecutor::runBatchnormFwdTest(
        DataType::HALF, DataType::HALF, DataType::HALF, DataType::HALF);
}

TEST(TestCpuReferenceGraphExecutor, BatchnormFwdInferenceAllBFloats)
{
    TestCpuReferenceGraphExecutor::runBatchnormFwdTest(
        DataType::BFLOAT16, DataType::BFLOAT16, DataType::BFLOAT16, DataType::BFLOAT16);
}

TEST(TestCpuReferenceGraphExecutor, SignaturesThatDontExist)
{
    EXPECT_THROW((TestCpuReferenceGraphExecutor::runBatchnormFwdTest(
                     DataType::FLOAT, DataType::HALF, DataType::HALF, DataType::FLOAT)),
                 std::runtime_error);

    EXPECT_THROW((TestCpuReferenceGraphExecutor::runBatchnormFwdTest(
                     DataType::FLOAT, DataType::HALF, DataType::FLOAT, DataType::FLOAT)),
                 std::runtime_error);
}

TEST(TestCpuReferenceGraphExecutor, BatchnormBwdAllFloats)
{
    TestCpuReferenceGraphExecutor::runBatchnormBwdTest<float, float, float, float>();
}

TEST(TestCpuReferenceGraphExecutor, BatchnormBwdAllHalfs)
{
    TestCpuReferenceGraphExecutor::runBatchnormBwdTest<half, half, half, half>();
}

TEST(TestCpuReferenceGraphExecutor, BatchnormBwdAllBFloat16)
{
    TestCpuReferenceGraphExecutor::runBatchnormBwdTest<bfloat16, bfloat16, bfloat16, bfloat16>();
}

TEST(TestCpuReferenceGraphExecutor, BatchnormTrainAllFloats)
{
    TestCpuReferenceGraphExecutor::runBatchnormTrainTest<float, float, float, float>();

    TestCpuReferenceGraphExecutor::runBatchnormTrainTest<float, float, float, float>(true);
}

TEST(TestCpuReferenceGraphExecutor, BatchnormTrainAllHalfs)
{
    TestCpuReferenceGraphExecutor::runBatchnormTrainTest<half, half, half, half>();
}

TEST(TestCpuReferenceGraphExecutor, BatchnormTrainAllBFloat16)
{
    TestCpuReferenceGraphExecutor::runBatchnormTrainTest<bfloat16, bfloat16, bfloat16, bfloat16>();
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
    TestCpuReferenceGraphExecutor::runConvolutionFwdTest<bfloat16, float>(DataType::BFLOAT16,
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
    TestCpuReferenceGraphExecutor::runConvolutionBwdTest<bfloat16, float>(DataType::BFLOAT16,
                                                                          DataType::FLOAT);
}

TEST(TestCpuReferenceGraphExecutor, ConvolutionWrwAllFloats)
{
    TestCpuReferenceGraphExecutor::runConvolutionWrwTest<float, float>(DataType::FLOAT,
                                                                       DataType::FLOAT);
}
TEST(TestCpuReferenceGraphExecutor, ConvolutionWrwAllHalfs)
{
    TestCpuReferenceGraphExecutor::runConvolutionWrwTest<half, float>(DataType::HALF,
                                                                      DataType::FLOAT);
}
TEST(TestCpuReferenceGraphExecutor, ConvolutionWrwAllBFloat16)
{
    TestCpuReferenceGraphExecutor::runConvolutionWrwTest<bfloat16, float>(DataType::BFLOAT16,
                                                                          DataType::FLOAT);
}

TEST(TestCpuReferenceGraphExecutor, MatmulAllFloats)
{
    TestCpuReferenceGraphExecutor::runMatmulTest<float, float>(DataType::FLOAT, DataType::FLOAT);
}
TEST(TestCpuReferenceGraphExecutor, MatmulAllHalfs)
{
    TestCpuReferenceGraphExecutor::runMatmulTest<half, float>(DataType::HALF, DataType::FLOAT);
}
TEST(TestCpuReferenceGraphExecutor, MatmulAllBFloat16)
{
    TestCpuReferenceGraphExecutor::runMatmulTest<bfloat16, float>(DataType::BFLOAT16,
                                                                  DataType::FLOAT);
}

TEST(TestCpuReferenceGraphExecutor, PointwiseBinaryAdd)
{
    const std::vector<int64_t> inputDims = {1, 3, 2, 2};
    const std::vector<int64_t> outputDims = {1, 3, 2, 2};

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

    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    CpuReferenceGraphExecutor().execute(
        serializedGraph.data(), serializedGraph.size(), variantPack);
}
