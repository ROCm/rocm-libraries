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
#include <hipdnn_sdk/utilities/TensorView.hpp>
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
                                    hipdnn_sdk::data_objects::DataType meanVarianceDataType,
                                    hipdnn_sdk::data_objects::DataType computeDataType)
    {
        unsigned int seed = getGlobalTestSeed();

        std::vector<int64_t> dims = {1, 3, 14, 14};
        auto graph = buildBatchnormFwdInferenceGraph(inputDataType,
                                                     scaleBiasDataType,
                                                     meanVarianceDataType,
                                                     computeDataType,
                                                     dims,
                                                     TensorLayout::NCHW,
                                                     true);

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

        std::vector<int64_t> dims = {1, 3, 14, 14};
        BatchnormBwdTensorBundle<InputType, ScaleBiasType, MeanVarianceType> tensorBundle(
            dims, 1, TensorLayout::NCHW);

        auto graphTuple = buildBatchnormBwdGraph(
            tensorBundle, inputDataType, scaleBiasDataType, meanVarianceDataType, computeDataType);

        auto& graph = std::get<0>(graphTuple);
        auto& variantPack = std::get<1>(graphTuple);

        auto result = graph->validate();
        ASSERT_EQ(result.code, hipdnn_frontend::ErrorCode::OK) << result.err_msg;

        auto flatbufferGraph = graph->buildFlatbufferOperationGraph();

        hipdnn_sdk::test_utilities::CpuReferenceGraphExecutor().execute(
            flatbufferGraph.data(), flatbufferGraph.size(), variantPack);
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

        std::vector<int64_t> dims = {1, 3, 14, 14};
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

    template <typename InputType, typename AccumulatorType>
    static void runConvolutionWrwTest(hipdnn_sdk::data_objects::DataType inputDataType,
                                      hipdnn_sdk::data_objects::DataType accumulatorDataType)
    {
        std::vector<int64_t> xDims = {1, 1, 2, 2};
        std::vector<int64_t> dwDims = {1, 1, 1, 1};
        std::vector<int64_t> dyDims = {1, 1, 2, 2};
        ConvolutionWrwTensorBundle<InputType> tensorBundle(
            xDims, dwDims, dyDims, 1, TensorLayout::NCHW);

        auto graphTuple
            = buildConvolutionWrwGraph(tensorBundle, inputDataType, accumulatorDataType);

        auto& graph = std::get<0>(graphTuple);
        auto& variantPack = std::get<1>(graphTuple);

        auto result = graph->validate();
        ASSERT_EQ(result.code, hipdnn_frontend::ErrorCode::OK) << result.err_msg;

        auto flatbufferGraph = graph->buildFlatbufferOperationGraph();

        hipdnn_sdk::test_utilities::CpuReferenceGraphExecutor().execute(
            flatbufferGraph.data(), flatbufferGraph.size(), variantPack);
    }

    template <typename InputType, typename AccumulatorType>
    static void runPointwiseUnaryTest(hipdnn_frontend::PointwiseMode mode,
                                      hipdnn_sdk::data_objects::DataType inputDataType,
                                      hipdnn_sdk::data_objects::DataType accumulatorDataType,
                                      float in0TensorValue = 0.0f,
                                      std::optional<float> reluLowerClip = std::nullopt,
                                      std::optional<float> reluUpperClip = std::nullopt,
                                      std::optional<float> reluLowerClipSlope = std::nullopt,
                                      std::optional<float> swishBeta = std::nullopt,
                                      std::optional<float> eluAlpha = std::nullopt,
                                      std::optional<float> softplusBeta = std::nullopt)
    {
        std::vector<int64_t> inputDims = {1, 3, 4, 4};
        std::vector<int64_t> outputDims = {1, 3, 4, 4};

        auto [graph, tensorBundle, variantPack] = buildPointwiseUnaryGraph(inputDims,
                                                                           outputDims,
                                                                           inputDataType,
                                                                           accumulatorDataType,
                                                                           inputDataType,
                                                                           mode,
                                                                           1,
                                                                           TensorLayout::NCHW,
                                                                           reluLowerClip,
                                                                           reluUpperClip,
                                                                           reluLowerClipSlope,
                                                                           swishBeta,
                                                                           eluAlpha,
                                                                           softplusBeta);

        auto result = graph->validate();
        ASSERT_EQ(result.code, hipdnn_frontend::ErrorCode::OK) << result.err_msg;

        auto flatbufferGraph = graph->buildFlatbufferOperationGraph();
        auto graphWrap
            = hipdnn_plugin::GraphWrapper(flatbufferGraph.data(), flatbufferGraph.size());
        const auto& nodeWrap = graphWrap.getNodeWrapper(0);
        const auto& attributes
            = nodeWrap.attributesAs<hipdnn_sdk::data_objects::PointwiseAttributes>();
        tensorBundle.tensors[attributes.in_0_tensor_uid()]->fillTensorWithValue(in0TensorValue);

        CpuReferenceGraphExecutor().execute(
            flatbufferGraph.data(), flatbufferGraph.size(), variantPack);

        const auto& outTensor = tensorBundle.tensors.at(attributes.out_0_tensor_uid());

        if(mode == hipdnn_frontend::PointwiseMode::RELU_FWD)
        {
            if(reluLowerClip.has_value() && reluUpperClip.has_value())
            {
                ConstTensorView<InputType> view(*outTensor);
                for(const auto& val : view)
                {
                    EXPECT_TRUE(val >= reluLowerClip.value() && val <= reluUpperClip.value());
                }
            }
            else if(reluUpperClip.has_value() && reluLowerClip == std::nullopt)
            {

                ConstTensorView<InputType> view(*outTensor);
                for(const auto& val : view)
                {
                    EXPECT_TRUE(val <= reluUpperClip.value());
                }
            }
            else if(reluLowerClipSlope.has_value())
            {
                ConstTensorView<InputType> view(*outTensor);
                for(const auto& val : view)
                {
                    if(val < 0)
                    {
                        EXPECT_NEAR(val, in0TensorValue * reluLowerClipSlope.value(), 1e-5f);
                    }
                    else
                    {
                        EXPECT_EQ(val, in0TensorValue);
                    }
                }
            }
            else
            {
                ConstTensorView<InputType> view(*outTensor);
                for(const auto& val : view)
                {
                    EXPECT_TRUE(val >= 0.0f);
                }
            }
        }
    }

    template <typename InputType, typename AccumulatorType>
    static void runPointwiseBinaryTest(hipdnn_frontend::PointwiseMode mode,
                                       hipdnn_sdk::data_objects::DataType inputDataType,
                                       hipdnn_sdk::data_objects::DataType accumulatorDataType,
                                       float in0TensorValue = 0.0f,
                                       float in1TensorValue = 0.0f,
                                       std::optional<float> reluLowerClip = std::nullopt,
                                       std::optional<float> reluUpperClip = std::nullopt,
                                       std::optional<float> reluLowerClipSlope = std::nullopt,
                                       std::optional<float> swishBeta = std::nullopt,
                                       std::optional<float> eluAlpha = std::nullopt,
                                       std::optional<float> softplusBeta = std::nullopt)
    {
        std::vector<int64_t> inputDims = {1, 3, 4, 4};
        std::vector<int64_t> outputDims = {1, 3, 4, 4};

        auto [graph, tensorBundle, variantPack] = buildPointwiseBinaryGraph(inputDims,
                                                                            inputDims,
                                                                            outputDims,
                                                                            inputDataType,
                                                                            inputDataType,
                                                                            accumulatorDataType,
                                                                            inputDataType,
                                                                            mode,
                                                                            1,
                                                                            TensorLayout::NCHW,
                                                                            reluLowerClip,
                                                                            reluUpperClip,
                                                                            reluLowerClipSlope,
                                                                            swishBeta,
                                                                            eluAlpha,
                                                                            softplusBeta);

        auto result = graph->validate();
        ASSERT_EQ(result.code, hipdnn_frontend::ErrorCode::OK) << result.err_msg;

        auto flatbufferGraph = graph->buildFlatbufferOperationGraph();
        auto graphWrap
            = hipdnn_plugin::GraphWrapper(flatbufferGraph.data(), flatbufferGraph.size());
        const auto& nodeWrap = graphWrap.getNodeWrapper(0);
        const auto& attributes
            = nodeWrap.attributesAs<hipdnn_sdk::data_objects::PointwiseAttributes>();
        tensorBundle.tensors[attributes.in_0_tensor_uid()]->fillTensorWithValue(in0TensorValue);
        tensorBundle.tensors[attributes.in_1_tensor_uid().value()]->fillTensorWithValue(
            in1TensorValue);

        CpuReferenceGraphExecutor().execute(
            flatbufferGraph.data(), flatbufferGraph.size(), variantPack);

        const auto& outTensor = tensorBundle.tensors.at(attributes.out_0_tensor_uid());

        if(mode == hipdnn_frontend::PointwiseMode::RELU_BWD)
        {
            if(reluLowerClip.has_value() && reluUpperClip.has_value())
            {
                ConstTensorView<InputType> outView(*outTensor);
                for(const auto& val : outView)
                {
                    if(in0TensorValue < reluLowerClip.value()
                       || in0TensorValue > reluUpperClip.value())
                    {
                        EXPECT_EQ(val, 0.0f);
                    }
                    else
                    {
                        EXPECT_EQ(val, in1TensorValue);
                    }
                }
            }
            else if(reluUpperClip.has_value() && reluLowerClip == std::nullopt)
            {

                ConstTensorView<InputType> view(*outTensor);
                for(const auto& val : view)
                {
                    if(in0TensorValue > reluUpperClip.value())
                    {
                        EXPECT_EQ(val, 0.0f);
                    }
                    else
                    {
                        EXPECT_EQ(val, in1TensorValue);
                    }
                }
            }
            else if(reluLowerClipSlope.has_value())
            {
                ConstTensorView<InputType> view(*outTensor);
                for(const auto& val : view)
                {
                    if(in0TensorValue < 0)
                    {
                        EXPECT_NEAR(val, in1TensorValue * reluLowerClipSlope.value(), 1e-5f);
                    }
                    else
                    {
                        EXPECT_EQ(val, in1TensorValue);
                    }
                }
            }
            else
            {
                ConstTensorView<InputType> view(*outTensor);
                for(const auto& val : view)
                {
                    EXPECT_TRUE(val >= 0.0f);
                }
            }
        }
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
    TestCpuReferenceGraphExecutor::
        runBatchnormBwdTest<hip_bfloat16, hip_bfloat16, hip_bfloat16, hip_bfloat16>();
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
    TestCpuReferenceGraphExecutor::
        runBatchnormTrainTest<hip_bfloat16, hip_bfloat16, hip_bfloat16, hip_bfloat16>();
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
    TestCpuReferenceGraphExecutor::runConvolutionWrwTest<hip_bfloat16, float>(DataType::BFLOAT16,
                                                                              DataType::FLOAT);
}

TEST(TestCpuReferenceGraphExecutor, PointwiseUnaryReluFwdFloats)
{
    //all below 0, all get clamped to 0
    TestCpuReferenceGraphExecutor::runPointwiseUnaryTest<float, float>(
        hipdnn_frontend::PointwiseMode::RELU_FWD, DataType::FLOAT, DataType::FLOAT, -10.0f);

    //all above 0, no change.
    TestCpuReferenceGraphExecutor::runPointwiseUnaryTest<float, float>(
        hipdnn_frontend::PointwiseMode::RELU_FWD, DataType::FLOAT, DataType::FLOAT, 10.0f);
}

TEST(TestCpuReferenceGraphExecutor, PointwiseUnaryReluFwdClampFloats)
{
    // buffer data all below lower bounds.
    TestCpuReferenceGraphExecutor::runPointwiseUnaryTest<float, float>(
        hipdnn_frontend::PointwiseMode::RELU_FWD,
        DataType::FLOAT,
        DataType::FLOAT,
        0.0f,
        0.1f,
        0.3f);

    // buffer data all above upper bounds.
    TestCpuReferenceGraphExecutor::runPointwiseUnaryTest<float, float>(
        hipdnn_frontend::PointwiseMode::RELU_FWD,
        DataType::FLOAT,
        DataType::FLOAT,
        2.0f,
        0.1f,
        0.3f);

    // buffer data all inside bounds
    TestCpuReferenceGraphExecutor::runPointwiseUnaryTest<float, float>(
        hipdnn_frontend::PointwiseMode::RELU_FWD,
        DataType::FLOAT,
        DataType::FLOAT,
        0.2f,
        0.1f,
        0.3f);
}

TEST(TestCpuReferenceGraphExecutor, PointwiseUnaryReluLeakyFloat)
{
    //all below 0
    TestCpuReferenceGraphExecutor::runPointwiseUnaryTest<float, float>(
        hipdnn_frontend::PointwiseMode::RELU_FWD,
        DataType::FLOAT,
        DataType::FLOAT,
        -0.5f,
        std::nullopt,
        std::nullopt,
        0.1f);

    //all above 0
    TestCpuReferenceGraphExecutor::runPointwiseUnaryTest<float, float>(
        hipdnn_frontend::PointwiseMode::RELU_FWD,
        DataType::FLOAT,
        DataType::FLOAT,
        0.5f,
        std::nullopt,
        std::nullopt,
        0.1f);
}

TEST(TestCpuReferenceGraphExecutor, PointwiseUnaryReluUpperBoundClippedFloat)
{
    //all bewlow upper bound
    TestCpuReferenceGraphExecutor::runPointwiseUnaryTest<float, float>(
        hipdnn_frontend::PointwiseMode::RELU_FWD,
        DataType::FLOAT,
        DataType::FLOAT,
        0.1f,
        std::nullopt,
        0.3f);

    //all above upper bound
    TestCpuReferenceGraphExecutor::runPointwiseUnaryTest<float, float>(
        hipdnn_frontend::PointwiseMode::RELU_FWD,
        DataType::FLOAT,
        DataType::FLOAT,
        0.9f,
        std::nullopt,
        0.3f);
}

TEST(TestCpuReferenceGraphExecutor, PointwiseBinaryReluBwdFloats)
{
    //all below 0, all get clamped to 0
    TestCpuReferenceGraphExecutor::runPointwiseBinaryTest<float, float>(
        hipdnn_frontend::PointwiseMode::RELU_BWD, DataType::FLOAT, DataType::FLOAT, -10.0f);

    //all above 0, no change.
    TestCpuReferenceGraphExecutor::runPointwiseBinaryTest<float, float>(
        hipdnn_frontend::PointwiseMode::RELU_BWD, DataType::FLOAT, DataType::FLOAT, 10.0f);
}

TEST(TestCpuReferenceGraphExecutor, PointwiseBinaryReluBwdClampFloats)
{
    // buffer data all below lower bounds.
    float in0TensorValue = -1.0f; //X
    float in1TensorValue = 5.0f; //Dy
    // all below bounds will go to 0
    TestCpuReferenceGraphExecutor::runPointwiseBinaryTest<float, float>(
        hipdnn_frontend::PointwiseMode::RELU_BWD,
        DataType::FLOAT,
        DataType::FLOAT,
        in0TensorValue,
        in1TensorValue,
        0.1f,
        0.3f);

    // buffer data all above upper bounds.
    in0TensorValue = 1.0f; //X
    in1TensorValue = 2.0f; //Dy
    //all above bounds wil go to 0
    TestCpuReferenceGraphExecutor::runPointwiseBinaryTest<float, float>(
        hipdnn_frontend::PointwiseMode::RELU_BWD,
        DataType::FLOAT,
        DataType::FLOAT,
        in0TensorValue,
        in1TensorValue,
        0.1f,
        0.3f);

    // buffer data all inside bounds
    in0TensorValue = 0.2f; //X
    in1TensorValue = 1.0f; //Dy
    // all inside bounds, all 1
    TestCpuReferenceGraphExecutor::runPointwiseBinaryTest<float, float>(
        hipdnn_frontend::PointwiseMode::RELU_BWD,
        DataType::FLOAT,
        DataType::FLOAT,
        in0TensorValue,
        in1TensorValue,
        0.1f,
        0.3f);
}

TEST(TestCpuReferenceGraphExecutor, PointwiseBinaryReluBwdLeakyFloat)
{

    //all below 0
    float in0TensorValue = -1.0f; //X
    float in1TensorValue = 5.0f; //Dy
    TestCpuReferenceGraphExecutor::runPointwiseBinaryTest<float, float>(
        hipdnn_frontend::PointwiseMode::RELU_BWD,
        DataType::FLOAT,
        DataType::FLOAT,
        in0TensorValue,
        in1TensorValue,
        std::nullopt,
        std::nullopt,
        0.1f);

    //all above 0
    in0TensorValue = 1.0f; //X
    in1TensorValue = 2.0f; //Dy
    TestCpuReferenceGraphExecutor::runPointwiseBinaryTest<float, float>(
        hipdnn_frontend::PointwiseMode::RELU_BWD,
        DataType::FLOAT,
        DataType::FLOAT,
        in0TensorValue,
        in1TensorValue,
        std::nullopt,
        std::nullopt,
        0.1f);
}

TEST(TestCpuReferenceGraphExecutor, PointwiseBinaryReluBwdUpperBoundClippedFloat)
{
    //all bewlow upper bound is active
    float in0TensorValue = 1.0f; //X
    float in1TensorValue = 5.0f; //Dy
    TestCpuReferenceGraphExecutor::runPointwiseBinaryTest<float, float>(
        hipdnn_frontend::PointwiseMode::RELU_BWD,
        DataType::FLOAT,
        DataType::FLOAT,
        in0TensorValue,
        in1TensorValue,
        std::nullopt,
        10.0f);

    //all above upper bound, zeros out
    in0TensorValue = 20.0f; //X
    in1TensorValue = 2.0f; //Dy
    TestCpuReferenceGraphExecutor::runPointwiseBinaryTest<float, float>(
        hipdnn_frontend::PointwiseMode::RELU_BWD,
        DataType::FLOAT,
        DataType::FLOAT,
        in0TensorValue,
        in1TensorValue,
        std::nullopt,
        10.0f);
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
