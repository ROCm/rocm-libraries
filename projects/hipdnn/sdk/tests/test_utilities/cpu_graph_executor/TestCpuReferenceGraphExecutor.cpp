// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/Utilities.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>

#include "BatchnormGraphUtils.hpp"
#include "BatchnormTensorBundles.hpp"
#include <hipdnn_sdk/plugin/EnginePluginApi.h>
#include <hipdnn_sdk/plugin/PluginApiDataTypes.h>
#include <hipdnn_sdk/plugin/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_sdk/test_utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_sdk/test_utilities/cpu_graph_executor/CpuReferenceGraphExecutor.hpp>
#include <hipdnn_sdk/utilities/ShallowTensor.hpp>
#include <hipdnn_sdk/utilities/Tensor.hpp>
#include <hipdnn_sdk/utilities/UtilsBfp16.hpp>
#include <hipdnn_sdk/utilities/UtilsFp16.hpp>

using namespace hipdnn_sdk::test_utilities;
using namespace hipdnn_sdk::data_objects;
using namespace hipdnn_sdk::utilities;
using namespace ::testing;

class TestCpuReferenceGraphExecutor
{
private:
public:
    static flatbuffers::FlatBufferBuilder createValidBatchnormFwdInferenceGraph(
        std::vector<int64_t> strides,
        std::vector<int64_t> dims,
        bool hasOptionalAttributes = true,
        hipdnn_sdk::data_objects::DataType inputDataType = DataType::FLOAT,
        hipdnn_sdk::data_objects::DataType scaleBiasDataType = DataType::FLOAT,
        hipdnn_sdk::data_objects::DataType meanVarianceDataType = DataType::FLOAT)
    {
        flatbuffers::FlatBufferBuilder builder;
        std::vector<::flatbuffers::Offset<hipdnn_sdk::data_objects::TensorAttributes>>
            tensorAttributes;

        std::vector<int64_t> derivedStrides = {1, strides[1], 1, 1};
        std::vector<int64_t> derivedDims = {1, dims[1], 1, 1};

        tensorAttributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
            builder, 1, "x", inputDataType, &strides, &dims));

        tensorAttributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
            builder, 2, "y", inputDataType, &strides, &dims));

        tensorAttributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
            builder, 3, "scale", scaleBiasDataType, &derivedStrides, &derivedDims));

        tensorAttributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
            builder, 4, "bias", scaleBiasDataType, &derivedStrides, &derivedDims));

        if(hasOptionalAttributes)
        {
            tensorAttributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
                builder, 5, "est_mean", meanVarianceDataType, &derivedStrides, &derivedDims));

            tensorAttributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
                builder, 6, "est_variance", meanVarianceDataType, &derivedStrides, &derivedDims));
        }

        auto bnormAttributes
            = hipdnn_sdk::data_objects::CreateBatchnormInferenceAttributes(builder,
                                                                           1, // x uid
                                                                           5, // mean uid
                                                                           6, // inv_variance uid
                                                                           3, // scale uid
                                                                           4, // bias uid
                                                                           2 // y uid
            );

        std::vector<::flatbuffers::Offset<hipdnn_sdk::data_objects::Node>> nodes;
        auto node = hipdnn_sdk::data_objects::CreateNodeDirect(
            builder,
            "batchnorm",
            hipdnn_sdk::data_objects::NodeAttributes::BatchnormInferenceAttributes,
            bnormAttributes.Union());
        nodes.push_back(node);

        auto graphOffset = hipdnn_sdk::data_objects::CreateGraphDirect(builder,
                                                                       "test",
                                                                       DataType::FLOAT,
                                                                       DataType::HALF,
                                                                       DataType::BFLOAT16,
                                                                       &tensorAttributes,
                                                                       &nodes);
        builder.Finish(graphOffset);
        return builder;
    }

    template <typename InputType, typename ScaleBiasType, typename MeanVarianceType>
    static void runBatchnormFwdTest(hipdnn_sdk::data_objects::DataType inputDataType,
                                    hipdnn_sdk::data_objects::DataType scaleBiasDataType,
                                    hipdnn_sdk::data_objects::DataType meanVarianceDataType)
    {
        std::vector<int64_t> dims = {1, 3, 14, 14};
        BatchnormFwdTensorBundle<InputType, ScaleBiasType, MeanVarianceType> tensorBundle(
            dims, 1, TensorLayout::NCHW);

        auto graphTuple = buildBatchnormFwdInferenceGraph(
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
    static void runBatchnormBwdTest(hipdnn_sdk::data_objects::DataType inputDataType,
                                    hipdnn_sdk::data_objects::DataType scaleBiasDataType,
                                    hipdnn_sdk::data_objects::DataType meanVarianceDataType)
    {
        unsigned int seed = std::random_device{}();
        std::vector<int64_t> dims = {1, 3, 14, 14};
        TensorLayout layout = TensorLayout::NCHW;

        BatchnormBwdTensorBundle<InputType, ScaleBiasType, MeanVarianceType> tensorBundle(
            dims, seed, layout);
        auto variantPack = tensorBundle.createVariantPack();

        auto batchnormBuilder = hipdnn_sdk::test_utilities::createValidBatchnormBwdGraph(
            tensorBundle.dyTensor.strides(),
            tensorBundle.dyTensor.dims(),
            true,
            inputDataType,
            scaleBiasDataType,
            meanVarianceDataType);

        auto batchnormGraph = batchnormBuilder.GetBufferPointer();

        hipdnn_sdk::test_utilities::CpuReferenceGraphExecutor().execute(
            batchnormGraph, batchnormBuilder.GetSize(), variantPack);
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
};

TEST(TestCpuReferenceGraphExecutor, BatchnormFwdInferenceAllFloats)
{
    TestCpuReferenceGraphExecutor::runBatchnormFwdTest<float, float, float>(
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT);
    TestCpuReferenceGraphExecutor::runBatchnormFwdTest<float, float, float>(
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT);
}

TEST(TestCpuReferenceGraphExecutor, BatchnormFwdInferenceAllHalfs)
{
    TestCpuReferenceGraphExecutor::runBatchnormFwdTest<half, half, half>(
        DataType::HALF, DataType::HALF, DataType::HALF);
}

TEST(TestCpuReferenceGraphExecutor, BatchnormFwdInferenceAllBFloats)
{
    TestCpuReferenceGraphExecutor::runBatchnormFwdTest<hip_bfloat16, hip_bfloat16, hip_bfloat16>(
        DataType::BFLOAT16, DataType::BFLOAT16, DataType::BFLOAT16);
}

TEST(TestCpuReferenceGraphExecutor, SignaturesThatDontExist)
{
    EXPECT_THROW((TestCpuReferenceGraphExecutor::runBatchnormFwdTest<float, half, half>(
                     DataType::FLOAT, DataType::HALF, DataType::HALF)),
                 std::runtime_error);

    EXPECT_THROW((TestCpuReferenceGraphExecutor::runBatchnormFwdTest<float, half, float>(
                     DataType::FLOAT, DataType::HALF, DataType::FLOAT)),
                 std::runtime_error);
}

TEST(TestCpuReferenceGraphExecutor, BatchnormBwdAllFloats)
{
    TestCpuReferenceGraphExecutor::runBatchnormBwdTest<float, float, float>(
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT);
    TestCpuReferenceGraphExecutor::runBatchnormBwdTest<float, float, float>(
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT);
}

TEST(TestCpuReferenceGraphExecutor, BatchnormBwdAllHalfs)
{
    // TestCpuReferenceGraphExecutor::runBatchnormBwdTest<half, half, half>(
    //     DataType::HALF, DataType::HALF, DataType::HALF);
}

TEST(TestCpuReferenceGraphExecutor, BatchnormBwdAllBFloat16)
{
    // TestCpuReferenceGraphExecutor::runBatchnormBwdTest<hip_bfloat16, hip_bfloat16, hip_bfloat16>(
    //     DataType::BFLOAT16, DataType::BFLOAT16, DataType::BFLOAT16);
}

TEST(TestCpuReferenceGraphExecutor, BatchnormTrainAllFloats)
{
    TestCpuReferenceGraphExecutor::runBatchnormTrainTest<float, float, float>(
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT);
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
