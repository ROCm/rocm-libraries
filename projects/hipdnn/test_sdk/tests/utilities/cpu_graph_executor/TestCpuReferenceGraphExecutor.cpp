// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <cstdint>
#include <vector>

#include <flatbuffers/flatbuffers.h>
#include <gtest/gtest.h>

#include <optional>

#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/Utilities.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>

#include "BatchnormGraphUtils.hpp"
#include "BatchnormTensorBundles.hpp"
#include "BlockScaleDequantizeGraphUtils.hpp"
#include "ConvolutionGraphUtils.hpp"
#include "LayernormGraphUtils.hpp"
#include "LayernormTensorBundles.hpp"
#include "MatmulGraphUtils.hpp"
#include "MoeGroupedMatmulBwdGraphUtils.hpp"
#include "MoeGroupedMatmulBwdTensorBundles.hpp"
#include "MoeGroupedMatmulGraphUtils.hpp"
#include "MoeGroupedMatmulTensorBundles.hpp"
#include "PointwiseGraphUtils.hpp"
#include "PointwiseTensorBundles.hpp"
#include "RMSNormGraphUtils.hpp"
#include "RMSNormTensorBundles.hpp"
#ifdef HIPDNN_ENABLE_SDPA
#include "SdpaGraphUtils.hpp"
#include "SdpaTensorBundles.hpp"
#endif

#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/ShallowTensor.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_data_sdk/utilities/TensorView.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/serialized_graph_and_plan_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceMoeGroupedMatmul.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceMoeGroupedMatmulBwd.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceResampleBwd.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceResampleFwd.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
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

namespace
{

// Wraps an inner graph buffer (and optional plan bytes) into an "HDGP"
// SerializedGraphAndPlan container, returning the released buffer.
flatbuffers::DetachedBuffer makeGraphAndPlanContainer(const std::vector<uint8_t>* graphBytes,
                                                      const std::vector<uint8_t>* planBytes)
{
    flatbuffers::FlatBufferBuilder builder;
    auto root = hipdnn_flatbuffers_sdk::data_objects::CreateSerializedGraphAndPlanDirect(
        builder, graphBytes, planBytes);
    hipdnn_flatbuffers_sdk::data_objects::FinishSerializedGraphAndPlanBuffer(builder, root);
    return builder.Release();
}

} // namespace

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

    template <typename InputType, typename OutputType, typename ComputeType>
    static void
        runMoeGroupedMatmulTest(hipdnn_flatbuffers_sdk::data_objects::MoeGroupedMatmulMode mode,
                                hipdnn_flatbuffers_sdk::data_objects::DataType inputDataType,
                                hipdnn_flatbuffers_sdk::data_objects::DataType outputDataType,
                                hipdnn_flatbuffers_sdk::data_objects::DataType computeDataType)
    {
        using MoeMode = hipdnn_flatbuffers_sdk::data_objects::MoeGroupedMatmulMode;
        const unsigned int seed = getGlobalTestSeed();

        constexpr int64_t EXPERTS = 2;
        constexpr int64_t HIDDEN_K = 3;
        constexpr int64_t WEIGHT_N = 4;
        constexpr int64_t TOKEN_ROWS = 6;
        const int64_t routedRows = (mode == MoeMode::GATHER) ? 7 : TOKEN_ROWS;
        const int32_t topK = (mode == MoeMode::SCATTER) ? 2 : 0;

        MoeGroupedMatmulTensorBundle<InputType> execBundle(
            EXPERTS, HIDDEN_K, WEIGHT_N, TOKEN_ROWS, routedRows, mode, topK, seed);
        MoeGroupedMatmulTensorBundle<InputType> directBundle(
            EXPERTS, HIDDEN_K, WEIGHT_N, TOKEN_ROWS, routedRows, mode, topK, seed);

        auto graphTuple = buildMoeGroupedMatmulGraph(
            execBundle, inputDataType, outputDataType, computeDataType);
        auto& graph = std::get<0>(graphTuple);
        auto& variantPack = std::get<1>(graphTuple);

        // The bundle's own output buffer is always InputType; when OutputType
        // differs (HALF/FLOAT, BFLOAT16/FLOAT), wire a correctly-typed buffer in
        // its place instead.
        Tensor<OutputType> execOutput(execBundle.outputTensor.dims(),
                                      execBundle.outputTensor.strides());
        variantPack[6] = execOutput.memory().hostData();

        auto result = graph->validate();
        ASSERT_EQ(result.code, hipdnn_frontend::ErrorCode::OK) << result.err_msg;

        auto [serializedGraph, serErr] = graph->to_binary();
        ASSERT_TRUE(serErr.is_good()) << serErr.get_message();

        CpuReferenceGraphExecutor executor;
        ASSERT_TRUE(executor.isApplicable(serializedGraph.data(), serializedGraph.size()));
        executor.execute(serializedGraph.data(), serializedGraph.size(), variantPack);

        Tensor<OutputType> directOutput(directBundle.outputTensor.dims(),
                                        directBundle.outputTensor.strides());
        CpuFpReferenceMoeGroupedMatmul::forward<InputType, InputType, OutputType, ComputeType>(
            directBundle.tokenTensor,
            directBundle.weightTensor,
            directBundle.firstTokenOffsetTensor,
            directOutput,
            mode,
            topK,
            directBundle.tokenIndexTensor.has_value() ? &(*directBundle.tokenIndexTensor) : nullptr,
            directBundle.tokenKsTensor.has_value() ? &(*directBundle.tokenKsTensor) : nullptr);

        const CpuFpReferenceValidation<OutputType> validator(0.0F, 0.0F);
        EXPECT_TRUE(validator.allClose(directOutput, execOutput));
    }

    template <typename InputType, typename DweightType, typename ComputeType>
    static void
        runMoeGroupedMatmulBwdTest(hipdnn_flatbuffers_sdk::data_objects::DataType inputDataType,
                                   hipdnn_flatbuffers_sdk::data_objects::DataType computeDataType,
                                   bool rowMajorDweight = false)
    {
        constexpr int64_t EXPERTS = 2;
        constexpr int64_t HIDDEN_K = 3;
        constexpr int64_t OUTPUT_N = 4;
        constexpr int64_t TOKEN_ROWS = 8;

        const unsigned int seed = getGlobalTestSeed();

        // Column-major is only the layout the node infers for an unset dweight, not a
        // requirement of the operation. Running the executor through a row-major dweight
        // while the direct reference keeps the column-major default makes the comparison
        // below cross-layout: allClose() indexes logically, so the values have to land in
        // the same logical positions either way.
        const std::vector<int64_t> execDweightStrides
            = rowMajorDweight ? generateStrides({EXPERTS, HIDDEN_K, OUTPUT_N})
                              : std::vector<int64_t>{};

        // Two bundles from the same seed: the executor writes through the first
        // one's buffers, leaving the second with pristine inputs for the direct
        // reference call.
        MoeGroupedMatmulBwdTensorBundle<InputType, DweightType> execBundle(
            EXPERTS, HIDDEN_K, OUTPUT_N, TOKEN_ROWS, seed, execDweightStrides);
        MoeGroupedMatmulBwdTensorBundle<InputType, DweightType> directBundle(
            EXPERTS, HIDDEN_K, OUTPUT_N, TOKEN_ROWS, seed);

        auto graphTuple = buildMoeGroupedMatmulBwdGraph(execBundle, inputDataType, computeDataType);

        auto& graph = std::get<0>(graphTuple);
        auto& variantPack = std::get<1>(graphTuple);

        auto result = graph->validate();
        ASSERT_EQ(result.code, hipdnn_frontend::ErrorCode::OK) << result.err_msg;

        auto [serializedGraph, serErr] = graph->to_binary();
        ASSERT_TRUE(serErr.is_good()) << serErr.get_message();

        CpuReferenceGraphExecutor executor;
        ASSERT_TRUE(executor.isApplicable(serializedGraph.data(), serializedGraph.size()));
        executor.execute(serializedGraph.data(), serializedGraph.size(), variantPack);

        CpuFpReferenceMoeGroupedMatmulBwd::backward<InputType, InputType, DweightType, ComputeType>(
            directBundle.doutputTensor,
            directBundle.tokenTensor,
            directBundle.firstTokenOffsetTensor,
            directBundle.dweightTensor);

        // Both sides run the same deterministic reference code over identical
        // seeded inputs, so the comparison is bit-exact.
        const CpuFpReferenceValidation<DweightType> validator(0.0F, 0.0F);
        EXPECT_TRUE(validator.allClose(directBundle.dweightTensor, execBundle.dweightTensor));
    }

    static void
        runLayernormTest(hipdnn_flatbuffers_sdk::data_objects::DataType inputDataType,
                         hipdnn_flatbuffers_sdk::data_objects::DataType scaleBiasDataType,
                         hipdnn_flatbuffers_sdk::data_objects::DataType meanInvVarianceDataType,
                         hipdnn_flatbuffers_sdk::data_objects::DataType computeDataType)
    {
        const unsigned int seed = getGlobalTestSeed();
        const std::vector<int64_t> dims = {1, 3, 14, 14};

        auto graph = buildLayernormFpropGraph(inputDataType,
                                              scaleBiasDataType,
                                              meanInvVarianceDataType,
                                              computeDataType,
                                              dims,
                                              2, // normalize over last 2 dims
                                              TensorLayout::NCHW);

        auto result = graph->validate();
        ASSERT_EQ(result.code, hipdnn_frontend::ErrorCode::OK) << result.err_msg;

        auto [serializedGraph, serErr] = graph->to_binary();
        ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
        const GraphWrapper graphWrapper(serializedGraph.data(), serializedGraph.size());

        LayernormFpropTensorBundle tensorBundle(
            graphWrapper.getNodeWrapper(0), graphWrapper.getTensorMap(), seed);

        auto variantPack = tensorBundle.toHostVariantPack();

        CpuReferenceGraphExecutor().execute(
            serializedGraph.data(), serializedGraph.size(), variantPack);
    }

    static void runLayernormBackwardTest(
        hipdnn_flatbuffers_sdk::data_objects::DataType dyDataType,
        hipdnn_flatbuffers_sdk::data_objects::DataType dxDataType,
        hipdnn_flatbuffers_sdk::data_objects::DataType scaleBiasDataType,
        hipdnn_flatbuffers_sdk::data_objects::DataType meanInvVarianceDataType,
        hipdnn_flatbuffers_sdk::data_objects::DataType computeDataType)
    {
        const unsigned int seed = getGlobalTestSeed();
        const std::vector<int64_t> dims = {1, 3, 14, 14};

        auto graph = buildLayernormBpropGraph(dyDataType,
                                              dxDataType,
                                              scaleBiasDataType,
                                              meanInvVarianceDataType,
                                              computeDataType,
                                              dims,
                                              2, // normalize over last 2 dims
                                              TensorLayout::NCHW);

        auto result = graph->validate();
        ASSERT_EQ(result.code, hipdnn_frontend::ErrorCode::OK) << result.err_msg;

        auto [serializedGraph, serErr] = graph->to_binary();
        ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
        const GraphWrapper graphWrapper(serializedGraph.data(), serializedGraph.size());

        LayernormBpropTensorBundle tensorBundle(
            graphWrapper.getNodeWrapper(0), graphWrapper.getTensorMap(), seed);

        auto variantPack = tensorBundle.toHostVariantPack();

        CpuReferenceGraphExecutor().execute(
            serializedGraph.data(), serializedGraph.size(), variantPack);
    }

    static void runRMSNormTest(hipdnn_flatbuffers_sdk::data_objects::DataType inputDataType,
                               hipdnn_flatbuffers_sdk::data_objects::DataType scaleDataType,
                               hipdnn_flatbuffers_sdk::data_objects::DataType computeDataType)
    {
        const unsigned int seed = getGlobalTestSeed();
        const std::vector<int64_t> dims = {1, 3, 14, 14};

        auto graph = buildRMSNormFwdGraph(
            inputDataType, scaleDataType, computeDataType, dims, TensorLayout::NCHW);

        auto result = graph->validate();
        ASSERT_EQ(result.code, hipdnn_frontend::ErrorCode::OK) << result.err_msg;

        auto [serializedGraph, serErr] = graph->to_binary();
        ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
        const GraphWrapper graphWrapper(serializedGraph.data(), serializedGraph.size());

        RMSNormFwdTensorBundle tensorBundle(
            graphWrapper.getNodeWrapper(0), graphWrapper.getTensorMap(), seed);

        auto variantPack = tensorBundle.toHostVariantPack();

        CpuReferenceGraphExecutor().execute(
            serializedGraph.data(), serializedGraph.size(), variantPack);
    }

    static void runRMSBwdNormTest(hipdnn_flatbuffers_sdk::data_objects::DataType inputDataType,
                                  hipdnn_flatbuffers_sdk::data_objects::DataType scaleDataType,
                                  hipdnn_flatbuffers_sdk::data_objects::DataType computeDataType)
    {
        const unsigned int seed = getGlobalTestSeed();
        const std::vector<int64_t> dims = {1, 3, 14, 14};

        auto graph = buildRMSNormBwdGraph(
            inputDataType, scaleDataType, computeDataType, dims, TensorLayout::NCHW);

        auto result = graph->validate();
        ASSERT_EQ(result.code, hipdnn_frontend::ErrorCode::OK) << result.err_msg;

        auto [serializedGraph, serErr] = graph->to_binary();
        ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
        const GraphWrapper graphWrapper(serializedGraph.data(), serializedGraph.size());

        RMSNormBwdTensorBundle tensorBundle(
            graphWrapper.getNodeWrapper(0), graphWrapper.getTensorMap(), seed);

        auto variantPack = tensorBundle.toHostVariantPack();

        CpuReferenceGraphExecutor().execute(
            serializedGraph.data(), serializedGraph.size(), variantPack);
    }

    static void runResampleFwdTest(bool generateIndex)
    {
        auto builder
            = createValidResampleFwdGraph(generateIndex ? std::optional<bool>(true) : std::nullopt);
        const GraphWrapper graphWrapper(builder.GetBufferPointer(), builder.GetSize());

        Tensor<float> xTensor({1, 1, 4, 4});
        Tensor<float> yTensor({1, 1, 2, 2});
        Tensor<float> directXTensor({1, 1, 4, 4});
        Tensor<float> directYTensor({1, 1, 2, 2});
        for(size_t i = 0; i < xTensor.elementCount(); ++i)
        {
            xTensor.memory().hostData()[i] = static_cast<float>(i + 1);
            directXTensor.memory().hostData()[i] = static_cast<float>(i + 1);
        }
        xTensor.memory().markHostModified();
        directXTensor.memory().markHostModified();

        std::unordered_map<int64_t, void*> variantPack{{1, xTensor.memory().hostData()},
                                                       {2, yTensor.memory().hostData()}};

        Tensor<int32_t> indexTensor({1, 1, 2, 2});
        Tensor<int32_t> directIndexTensor({1, 1, 2, 2});
        if(generateIndex)
        {
            variantPack[3] = indexTensor.memory().hostData();
        }

        CpuReferenceGraphExecutor().execute(
            builder.GetBufferPointer(), builder.GetSize(), variantPack);

        CpuFpReferenceResampleFwd::forward<float, float, float, int32_t>(
            directXTensor,
            directYTensor,
            {0, 0},
            {2, 2},
            {2, 2},
            ResampleMode::MAXPOOL,
            PaddingMode::ZERO_PAD,
            generateIndex ? &directIndexTensor : nullptr);

        const CpuFpReferenceValidation<float> validator(0.0f, 0.0f);
        EXPECT_TRUE(validator.allClose(directYTensor, yTensor));
        if(generateIndex)
        {
            EXPECT_EQ(indexTensor.memory().hostData()[0], directIndexTensor.memory().hostData()[0]);
            EXPECT_EQ(indexTensor.memory().hostData()[1], directIndexTensor.memory().hostData()[1]);
            EXPECT_EQ(indexTensor.memory().hostData()[2], directIndexTensor.memory().hostData()[2]);
            EXPECT_EQ(indexTensor.memory().hostData()[3], directIndexTensor.memory().hostData()[3]);
        }
    }

    static void runResampleBwdTest(hipdnn_flatbuffers_sdk::data_objects::ResampleMode resampleMode)
    {
        auto builder = createValidResampleBwdGraph(true, resampleMode);
        const GraphWrapper graphWrapper(builder.GetBufferPointer(), builder.GetSize());

        Tensor<float> dyTensor({1, 1, 2, 2});
        Tensor<float> directDyTensor({1, 1, 2, 2});
        Tensor<float> dxTensor({1, 1, 4, 4});
        Tensor<float> directDxTensor({1, 1, 4, 4});

        for(size_t i = 0; i < dyTensor.elementCount(); ++i)
        {
            const auto val = static_cast<float>(i + 1);
            dyTensor.memory().hostData()[i] = val;
            directDyTensor.memory().hostData()[i] = val;
        }
        dyTensor.memory().markHostModified();
        directDyTensor.memory().markHostModified();

        Tensor<int32_t> indexTensor({1, 1, 2, 2});
        Tensor<int32_t> directIndexTensor({1, 1, 2, 2});

        const std::unordered_map<int64_t, void*> variantPack{{1, dyTensor.memory().hostData()},
                                                             {2, dxTensor.memory().hostData()},
                                                             {3, indexTensor.memory().hostData()}};

        if(resampleMode == hipdnn_flatbuffers_sdk::data_objects::ResampleMode::MAXPOOL)
        {
            // maxpool indices with a 2x2 window on a 4x4 input of linear elements
            const std::vector<int32_t> sampleIndices = {5, 7, 13, 15};
            for(size_t i = 0; i < sampleIndices.size(); ++i)
            {
                indexTensor.memory().hostData()[i] = sampleIndices[i];
                directIndexTensor.memory().hostData()[i] = sampleIndices[i];
            }

            indexTensor.memory().markHostModified();
            directIndexTensor.memory().markHostModified();
        }

        CpuReferenceGraphExecutor().execute(
            builder.GetBufferPointer(), builder.GetSize(), variantPack);

        CpuFpReferenceResampleBwd::backward<float>(directDyTensor,
                                                   directDxTensor,
                                                   {0, 0},
                                                   {2, 2},
                                                   {2, 2},
                                                   resampleMode,
                                                   PaddingMode::ZERO_PAD,
                                                   &directIndexTensor);

        const CpuFpReferenceValidation<float> validator(0.0f, 0.0f);
        EXPECT_TRUE(validator.allClose(directDxTensor, dxTensor));
    }

#ifdef HIPDNN_ENABLE_SDPA
    template <typename InputType>
    static void runSdpaTest(hipdnn_flatbuffers_sdk::data_objects::DataType dataType)
    {
        // Q/K/V: [batch=1, heads=2, seq=4, head_dim=8]
        const std::vector<int64_t> qDims = {1, 2, 4, 8};
        const std::vector<int64_t> kDims = {1, 2, 4, 8};
        const std::vector<int64_t> vDims = {1, 2, 4, 8};

        SdpaFwdTensorBundle<InputType> tensorBundle(qDims, kDims, vDims);

        auto graphTuple = buildSdpaFwdGraph(tensorBundle, dataType);

        auto& graph = std::get<0>(graphTuple);
        auto& variantPack = std::get<1>(graphTuple);

        auto result = graph->validate();
        ASSERT_EQ(result.code, hipdnn_frontend::ErrorCode::OK) << result.err_msg;

        auto [serializedGraph, serErr] = graph->to_binary();
        ASSERT_TRUE(serErr.is_good()) << serErr.get_message();

        CpuReferenceGraphExecutor().execute(
            serializedGraph.data(), serializedGraph.size(), variantPack);
    }
#endif

    template <typename XType, typename ScaleType>
    static void
        runBlockScaleDequantizeTest(hipdnn_flatbuffers_sdk::data_objects::DataType xDataType,
                                    hipdnn_flatbuffers_sdk::data_objects::DataType scaleDataType,
                                    hipdnn_flatbuffers_sdk::data_objects::DataType yDataType,
                                    hipdnn_flatbuffers_sdk::data_objects::DataType computeDataType)
    {
        const std::vector<int64_t> xDims = {2, 32, 32, 64};
        const std::vector<int32_t> blockSize = {32};
        // scale dims: ceil(64/32) = 2 for the blocked trailing dim
        const std::vector<int64_t> scaleDims = {2, 32, 32, 2};

        BlockScaleDequantizeTensorBundle<XType, ScaleType> tensorBundle(xDims, scaleDims);

        auto graphTuple = buildBlockScaleDequantizeGraph(
            tensorBundle, xDataType, scaleDataType, yDataType, computeDataType, blockSize);

        auto& graph = std::get<0>(graphTuple);
        auto& variantPack = std::get<1>(graphTuple);

        auto result = graph->validate();
        ASSERT_EQ(result.code, hipdnn_frontend::ErrorCode::OK) << result.err_msg;

        auto [serializedGraph2, serErr2] = graph->to_binary();
        ASSERT_TRUE(serErr2.is_good()) << serErr2.get_message();

        CpuReferenceGraphExecutor().execute(
            serializedGraph2.data(), serializedGraph2.size(), variantPack);
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

TEST(TestCpuReferenceGraphExecutor, MoeGroupedMatmulNoneAllFloats)
{
    TestCpuReferenceGraphExecutor::runMoeGroupedMatmulTest<float, float, float>(
        MoeGroupedMatmulMode::NONE, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT);
}
TEST(TestCpuReferenceGraphExecutor, MoeGroupedMatmulNoneAllHalfs)
{
    TestCpuReferenceGraphExecutor::runMoeGroupedMatmulTest<half, half, float>(
        MoeGroupedMatmulMode::NONE, DataType::HALF, DataType::HALF, DataType::FLOAT);
}
TEST(TestCpuReferenceGraphExecutor, MoeGroupedMatmulNoneAllBFloat16)
{
    TestCpuReferenceGraphExecutor::runMoeGroupedMatmulTest<bfloat16, bfloat16, float>(
        MoeGroupedMatmulMode::NONE, DataType::BFLOAT16, DataType::BFLOAT16, DataType::FLOAT);
}
TEST(TestCpuReferenceGraphExecutor, MoeGroupedMatmulNoneHalfInputFloatOutput)
{
    TestCpuReferenceGraphExecutor::runMoeGroupedMatmulTest<half, float, float>(
        MoeGroupedMatmulMode::NONE, DataType::HALF, DataType::FLOAT, DataType::FLOAT);
}
TEST(TestCpuReferenceGraphExecutor, MoeGroupedMatmulNoneBFloat16InputFloatOutput)
{
    TestCpuReferenceGraphExecutor::runMoeGroupedMatmulTest<bfloat16, float, float>(
        MoeGroupedMatmulMode::NONE, DataType::BFLOAT16, DataType::FLOAT, DataType::FLOAT);
}
TEST(TestCpuReferenceGraphExecutor, MoeGroupedMatmulNoneFloatInputHalfOutput)
{
    TestCpuReferenceGraphExecutor::runMoeGroupedMatmulTest<float, half, float>(
        MoeGroupedMatmulMode::NONE, DataType::FLOAT, DataType::HALF, DataType::FLOAT);
}
TEST(TestCpuReferenceGraphExecutor, MoeGroupedMatmulNoneFloatInputBFloat16Output)
{
    TestCpuReferenceGraphExecutor::runMoeGroupedMatmulTest<float, bfloat16, float>(
        MoeGroupedMatmulMode::NONE, DataType::FLOAT, DataType::BFLOAT16, DataType::FLOAT);
}

TEST(TestCpuReferenceGraphExecutor, MoeGroupedMatmulGatherAllFloats)
{
    TestCpuReferenceGraphExecutor::runMoeGroupedMatmulTest<float, float, float>(
        MoeGroupedMatmulMode::GATHER, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT);
}
TEST(TestCpuReferenceGraphExecutor, MoeGroupedMatmulGatherAllHalfs)
{
    TestCpuReferenceGraphExecutor::runMoeGroupedMatmulTest<half, half, float>(
        MoeGroupedMatmulMode::GATHER, DataType::HALF, DataType::HALF, DataType::FLOAT);
}
TEST(TestCpuReferenceGraphExecutor, MoeGroupedMatmulGatherAllBFloat16)
{
    TestCpuReferenceGraphExecutor::runMoeGroupedMatmulTest<bfloat16, bfloat16, float>(
        MoeGroupedMatmulMode::GATHER, DataType::BFLOAT16, DataType::BFLOAT16, DataType::FLOAT);
}
TEST(TestCpuReferenceGraphExecutor, MoeGroupedMatmulGatherHalfInputFloatOutput)
{
    TestCpuReferenceGraphExecutor::runMoeGroupedMatmulTest<half, float, float>(
        MoeGroupedMatmulMode::GATHER, DataType::HALF, DataType::FLOAT, DataType::FLOAT);
}
TEST(TestCpuReferenceGraphExecutor, MoeGroupedMatmulGatherBFloat16InputFloatOutput)
{
    TestCpuReferenceGraphExecutor::runMoeGroupedMatmulTest<bfloat16, float, float>(
        MoeGroupedMatmulMode::GATHER, DataType::BFLOAT16, DataType::FLOAT, DataType::FLOAT);
}
TEST(TestCpuReferenceGraphExecutor, MoeGroupedMatmulGatherFloatInputHalfOutput)
{
    TestCpuReferenceGraphExecutor::runMoeGroupedMatmulTest<float, half, float>(
        MoeGroupedMatmulMode::GATHER, DataType::FLOAT, DataType::HALF, DataType::FLOAT);
}
TEST(TestCpuReferenceGraphExecutor, MoeGroupedMatmulGatherFloatInputBFloat16Output)
{
    TestCpuReferenceGraphExecutor::runMoeGroupedMatmulTest<float, bfloat16, float>(
        MoeGroupedMatmulMode::GATHER, DataType::FLOAT, DataType::BFLOAT16, DataType::FLOAT);
}

TEST(TestCpuReferenceGraphExecutor, MoeGroupedMatmulScatterAllFloats)
{
    TestCpuReferenceGraphExecutor::runMoeGroupedMatmulTest<float, float, float>(
        MoeGroupedMatmulMode::SCATTER, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT);
}
TEST(TestCpuReferenceGraphExecutor, MoeGroupedMatmulScatterAllHalfs)
{
    TestCpuReferenceGraphExecutor::runMoeGroupedMatmulTest<half, half, float>(
        MoeGroupedMatmulMode::SCATTER, DataType::HALF, DataType::HALF, DataType::FLOAT);
}
TEST(TestCpuReferenceGraphExecutor, MoeGroupedMatmulScatterAllBFloat16)
{
    TestCpuReferenceGraphExecutor::runMoeGroupedMatmulTest<bfloat16, bfloat16, float>(
        MoeGroupedMatmulMode::SCATTER, DataType::BFLOAT16, DataType::BFLOAT16, DataType::FLOAT);
}
TEST(TestCpuReferenceGraphExecutor, MoeGroupedMatmulScatterHalfInputFloatOutput)
{
    TestCpuReferenceGraphExecutor::runMoeGroupedMatmulTest<half, float, float>(
        MoeGroupedMatmulMode::SCATTER, DataType::HALF, DataType::FLOAT, DataType::FLOAT);
}
TEST(TestCpuReferenceGraphExecutor, MoeGroupedMatmulScatterBFloat16InputFloatOutput)
{
    TestCpuReferenceGraphExecutor::runMoeGroupedMatmulTest<bfloat16, float, float>(
        MoeGroupedMatmulMode::SCATTER, DataType::BFLOAT16, DataType::FLOAT, DataType::FLOAT);
}
TEST(TestCpuReferenceGraphExecutor, MoeGroupedMatmulScatterFloatInputHalfOutput)
{
    TestCpuReferenceGraphExecutor::runMoeGroupedMatmulTest<float, half, float>(
        MoeGroupedMatmulMode::SCATTER, DataType::FLOAT, DataType::HALF, DataType::FLOAT);
}
TEST(TestCpuReferenceGraphExecutor, MoeGroupedMatmulScatterFloatInputBFloat16Output)
{
    TestCpuReferenceGraphExecutor::runMoeGroupedMatmulTest<float, bfloat16, float>(
        MoeGroupedMatmulMode::SCATTER, DataType::FLOAT, DataType::BFLOAT16, DataType::FLOAT);
}

TEST(TestCpuReferenceGraphExecutor, MoeGroupedMatmulBwdAllFloats)
{
    TestCpuReferenceGraphExecutor::runMoeGroupedMatmulBwdTest<float, float, float>(DataType::FLOAT,
                                                                                   DataType::FLOAT);
}
TEST(TestCpuReferenceGraphExecutor, MoeGroupedMatmulBwdAllHalfs)
{
    TestCpuReferenceGraphExecutor::runMoeGroupedMatmulBwdTest<half, half, float>(DataType::HALF,
                                                                                 DataType::FLOAT);
}
TEST(TestCpuReferenceGraphExecutor, MoeGroupedMatmulBwdAllBFloat16)
{
    TestCpuReferenceGraphExecutor::runMoeGroupedMatmulBwdTest<bfloat16, bfloat16, float>(
        DataType::BFLOAT16, DataType::FLOAT);
}
TEST(TestCpuReferenceGraphExecutor, MoeGroupedMatmulBwdHalfInputFloatDweight)
{
    TestCpuReferenceGraphExecutor::runMoeGroupedMatmulBwdTest<half, float, float>(DataType::HALF,
                                                                                  DataType::FLOAT);
}
TEST(TestCpuReferenceGraphExecutor, MoeGroupedMatmulBwdBFloat16InputFloatDweight)
{
    TestCpuReferenceGraphExecutor::runMoeGroupedMatmulBwdTest<bfloat16, float, float>(
        DataType::BFLOAT16, DataType::FLOAT);
}
TEST(TestCpuReferenceGraphExecutor, MoeGroupedMatmulBwdFloatInputHalfDweight)
{
    TestCpuReferenceGraphExecutor::runMoeGroupedMatmulBwdTest<float, half, float>(DataType::FLOAT,
                                                                                  DataType::FLOAT);
}
TEST(TestCpuReferenceGraphExecutor, MoeGroupedMatmulBwdFloatInputBFloat16Dweight)
{
    TestCpuReferenceGraphExecutor::runMoeGroupedMatmulBwdTest<float, bfloat16, float>(
        DataType::FLOAT, DataType::FLOAT);
}
TEST(TestCpuReferenceGraphExecutor, MoeGroupedMatmulBwdRowMajorDweight)
{
    TestCpuReferenceGraphExecutor::runMoeGroupedMatmulBwdTest<float, float, float>(
        DataType::FLOAT, DataType::FLOAT, /*rowMajorDweight=*/true);
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

TEST(TestCpuReferenceGraphExecutor, ExecutesGraphFromContainerBlob)
{
    // Guards the regression from review thread r3364940415: the executor must
    // peel an "HDGP" SerializedGraphAndPlan container before wrapping the graph.
    // Before fromSerializedBlob() was used, execute() constructed the
    // GraphWrapper directly from the container bytes and threw "Graph is not
    // valid".
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

    // A graph-only container (no plan) must execute.
    auto graphOnly = makeGraphAndPlanContainer(&serializedGraph, nullptr);
    EXPECT_NO_THROW(
        CpuReferenceGraphExecutor().execute(graphOnly.data(), graphOnly.size(), variantPack));

    // A graph+plan container must also execute; the plan blob is ignored by the
    // reference executor.
    const std::vector<uint8_t> dummyPlan = {0x01, 0x02, 0x03, 0x04};
    auto graphAndPlan = makeGraphAndPlanContainer(&serializedGraph, &dummyPlan);
    EXPECT_NO_THROW(
        CpuReferenceGraphExecutor().execute(graphAndPlan.data(), graphAndPlan.size(), variantPack));
}

TEST(TestCpuReferenceGraphExecutor, BlockScaleDequantizeAllFloats)
{
    TestCpuReferenceGraphExecutor::runBlockScaleDequantizeTest<float, float>(
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT);
}
TEST(TestCpuReferenceGraphExecutor, BlockScaleDequantizeHalfInputFloatScale)
{
    TestCpuReferenceGraphExecutor::runBlockScaleDequantizeTest<half, float>(
        DataType::HALF, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT);
}
TEST(TestCpuReferenceGraphExecutor, BlockScaleDequantizeBFloat16InputFloatScale)
{
    TestCpuReferenceGraphExecutor::runBlockScaleDequantizeTest<bfloat16, float>(
        DataType::BFLOAT16, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT);
}

TEST(TestCpuReferenceGraphExecutor, LayernormAllFloats)
{
    TestCpuReferenceGraphExecutor::runLayernormTest(
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT);
}
TEST(TestCpuReferenceGraphExecutor, LayernormAllHalfs)
{
    TestCpuReferenceGraphExecutor::runLayernormTest(
        DataType::HALF, DataType::HALF, DataType::HALF, DataType::HALF);
}
TEST(TestCpuReferenceGraphExecutor, LayernormAllBFloat16)
{
    TestCpuReferenceGraphExecutor::runLayernormTest(
        DataType::BFLOAT16, DataType::BFLOAT16, DataType::BFLOAT16, DataType::BFLOAT16);
}

TEST(TestCpuReferenceGraphExecutor, LayernormBackwardAllFloats)
{
    TestCpuReferenceGraphExecutor::runLayernormBackwardTest(
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT);
}
TEST(TestCpuReferenceGraphExecutor, LayernormBackwardAllHalfs)
{
    TestCpuReferenceGraphExecutor::runLayernormBackwardTest(
        DataType::HALF, DataType::HALF, DataType::HALF, DataType::HALF, DataType::HALF);
}
TEST(TestCpuReferenceGraphExecutor, LayernormBackwardAllBFloat16)
{
    TestCpuReferenceGraphExecutor::runLayernormBackwardTest(DataType::BFLOAT16,
                                                            DataType::BFLOAT16,
                                                            DataType::BFLOAT16,
                                                            DataType::BFLOAT16,
                                                            DataType::BFLOAT16);
}

TEST(TestCpuReferenceGraphExecutor, RMSNormAllFloats)
{
    TestCpuReferenceGraphExecutor::runRMSNormTest(
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT);
}
TEST(TestCpuReferenceGraphExecutor, RMSNormAllHalfs)
{
    TestCpuReferenceGraphExecutor::runRMSNormTest(DataType::HALF, DataType::HALF, DataType::HALF);
}
TEST(TestCpuReferenceGraphExecutor, RMSNormAllBFloat16)
{
    TestCpuReferenceGraphExecutor::runRMSNormTest(
        DataType::BFLOAT16, DataType::BFLOAT16, DataType::BFLOAT16);
}

TEST(TestCpuReferenceGraphExecutor, RMSNormBwdAllFloats)
{
    TestCpuReferenceGraphExecutor::runRMSBwdNormTest(
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT);
}

TEST(TestCpuReferenceGraphExecutor, RMSNormBwdAllHalfs)
{
    TestCpuReferenceGraphExecutor::runRMSBwdNormTest(
        DataType::HALF, DataType::HALF, DataType::HALF);
}

TEST(TestCpuReferenceGraphExecutor, RMSNormBwdAllBFloat16)
{
    TestCpuReferenceGraphExecutor::runRMSBwdNormTest(
        DataType::BFLOAT16, DataType::BFLOAT16, DataType::BFLOAT16);
}

TEST(TestCpuReferenceGraphExecutor, ResampleFwdAllFloats)
{
    TestCpuReferenceGraphExecutor::runResampleFwdTest(false);
}

TEST(TestCpuReferenceGraphExecutor, ResampleFwdWithIndexAllFloats)
{
    TestCpuReferenceGraphExecutor::runResampleFwdTest(true);
}

TEST(TestCpuReferenceGraphExecutor, ResampleBwdMaxpool)
{
    TestCpuReferenceGraphExecutor::runResampleBwdTest(
        hipdnn_flatbuffers_sdk::data_objects::ResampleMode::MAXPOOL);
}

TEST(TestCpuReferenceGraphExecutor, ResampleBwdAvgExcludePadding)
{
    TestCpuReferenceGraphExecutor::runResampleBwdTest(
        hipdnn_flatbuffers_sdk::data_objects::ResampleMode::AVGPOOL_EXCLUDE_PADDING);
}

TEST(TestCpuReferenceGraphExecutor, ResampleBwdAvgIncludePadding)
{
    TestCpuReferenceGraphExecutor::runResampleBwdTest(
        hipdnn_flatbuffers_sdk::data_objects::ResampleMode::AVGPOOL_INCLUDE_PADDING);
}

#ifdef HIPDNN_ENABLE_SDPA
TEST(TestCpuReferenceGraphExecutor, SdpaAllFloats)
{
    TestCpuReferenceGraphExecutor::runSdpaTest<float>(DataType::FLOAT);
}
TEST(TestCpuReferenceGraphExecutor, SdpaAllHalfs)
{
    TestCpuReferenceGraphExecutor::runSdpaTest<half>(DataType::HALF);
}
TEST(TestCpuReferenceGraphExecutor, SdpaAllBFloat16)
{
    TestCpuReferenceGraphExecutor::runSdpaTest<bfloat16>(DataType::BFLOAT16);
}
#endif
