// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <cmath>
#include <optional>

#include "PointwiseGraphUtils.hpp"
#include "hipdnn_flatbuffers_sdk/data_objects/pointwise_attributes_generated.h"
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/Seeds.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/CpuReferenceGraphExecutor.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/PointwisePlan.hpp>
#include <hipdnn_test_sdk/utilities/pointwise/CpuReferencePointwise.hpp>

using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_test_sdk::detail;
using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_flatbuffers_sdk::flatbuffer_utilities;
using namespace ::testing;
using namespace hipdnn_sdk_test_utils;
using hipdnn_data_sdk::types::bfloat16;
using hipdnn_data_sdk::types::half;

class TestPointwisePlan : public ::testing::Test
{
};

TEST_F(TestPointwisePlan, ExecutePlanUnaryReluFwd)
{
    const std::vector<int64_t> inputDims = {1, 3, 4, 4};
    const std::vector<int64_t> outputDims = {1, 3, 4, 4};
    const unsigned int seed = getGlobalTestSeed();

    // Build graph using new GraphTensorBundle pattern
    auto [graph, tensorBundle, variantPack]
        = buildPointwiseUnaryGraph(inputDims,
                                   outputDims,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   hipdnn_frontend::PointwiseMode::RELU_FWD,
                                   seed,
                                   TensorLayout::NCHW);

    // Execute using CpuReferenceGraphExecutor
    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    CpuReferenceGraphExecutor{}.execute(
        serializedGraph.data(), serializedGraph.size(), variantPack);

    // Verify output is correct (non-negative since it's RELU)
    // For this test we just verify execution succeeded without throwing
    SUCCEED();
}

TEST_F(TestPointwisePlan, ExecutePlanBinaryAdd)
{
    const std::vector<int64_t> input1Dims = {1, 3, 2, 2};
    const std::vector<int64_t> input2Dims = {1, 3, 2, 2};
    const std::vector<int64_t> outputDims = {1, 3, 2, 2};
    const unsigned int seed = getGlobalTestSeed();

    // Build graph using new GraphTensorBundle pattern
    auto [graph, tensorBundle, variantPack]
        = buildPointwiseBinaryGraph(input1Dims,
                                    input2Dims,
                                    outputDims,
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    hipdnn_frontend::PointwiseMode::ADD,
                                    seed,
                                    TensorLayout::NCHW);

    // Execute using CpuReferenceGraphExecutor
    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    CpuReferenceGraphExecutor{}.execute(
        serializedGraph.data(), serializedGraph.size(), variantPack);

    // Verify execution succeeded
    SUCCEED();
}

TEST_F(TestPointwisePlan, ExecutePlanBackwardReluBwd)
{
    const std::vector<int64_t> dyDims = {1, 3, 2, 2};
    const std::vector<int64_t> xDims = {1, 3, 2, 2};
    const std::vector<int64_t> dxDims = {1, 3, 2, 2};
    const unsigned int seed = getGlobalTestSeed();

    // Build graph using new GraphTensorBundle pattern
    auto [graph, tensorBundle, variantPack]
        = buildPointwiseBinaryGraph(dyDims,
                                    xDims,
                                    dxDims,
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    hipdnn_frontend::PointwiseMode::RELU_BWD,
                                    seed,
                                    TensorLayout::NCHW);

    // Execute using CpuReferenceGraphExecutor
    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    CpuReferenceGraphExecutor{}.execute(
        serializedGraph.data(), serializedGraph.size(), variantPack);

    // Verify execution succeeded
    SUCCEED();
}

TEST_F(TestPointwisePlan, ExecutePlanUnaryGeluFwd)
{
    const std::vector<int64_t> inputDims = {1, 3, 4, 4};
    const std::vector<int64_t> outputDims = {1, 3, 4, 4};
    const unsigned int seed = getGlobalTestSeed();

    auto [graph, tensorBundle, variantPack]
        = buildPointwiseUnaryGraph(inputDims,
                                   outputDims,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   hipdnn_frontend::PointwiseMode::GELU_FWD,
                                   seed,
                                   TensorLayout::NCHW);

    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    CpuReferenceGraphExecutor{}.execute(
        serializedGraph.data(), serializedGraph.size(), variantPack);

    SUCCEED();
}

TEST_F(TestPointwisePlan, ExecutePlanUnaryGeluApproxTanhFwd)
{
    const std::vector<int64_t> inputDims = {1, 3, 4, 4};
    const std::vector<int64_t> outputDims = {1, 3, 4, 4};
    const unsigned int seed = getGlobalTestSeed();

    auto [graph, tensorBundle, variantPack]
        = buildPointwiseUnaryGraph(inputDims,
                                   outputDims,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   hipdnn_frontend::PointwiseMode::GELU_APPROX_TANH_FWD,
                                   seed,
                                   TensorLayout::NCHW);

    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    CpuReferenceGraphExecutor{}.execute(
        serializedGraph.data(), serializedGraph.size(), variantPack);

    SUCCEED();
}

TEST_F(TestPointwisePlan, ExecutePlanUnarySwishFwd)
{
    const std::vector<int64_t> inputDims = {1, 3, 4, 4};
    const std::vector<int64_t> outputDims = {1, 3, 4, 4};
    const unsigned int seed = getGlobalTestSeed();

    auto [graph, tensorBundle, variantPack]
        = buildPointwiseUnaryGraph(inputDims,
                                   outputDims,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   hipdnn_frontend::PointwiseMode::SWISH_FWD,
                                   seed,
                                   TensorLayout::NCHW);

    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    CpuReferenceGraphExecutor{}.execute(
        serializedGraph.data(), serializedGraph.size(), variantPack);

    SUCCEED();
}

TEST_F(TestPointwisePlan, ExecutePlanUnarySwishFwdWithBeta)
{
    const std::vector<int64_t> inputDims = {1, 3, 4, 4};
    const std::vector<int64_t> outputDims = {1, 3, 4, 4};
    const unsigned int seed = getGlobalTestSeed();

    auto [graph, tensorBundle, variantPack]
        = buildPointwiseUnaryGraph(inputDims,
                                   outputDims,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   hipdnn_frontend::PointwiseMode::SWISH_FWD,
                                   seed,
                                   TensorLayout::NCHW,
                                   std::nullopt,
                                   std::nullopt,
                                   std::nullopt,
                                   0.5f);

    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    const GraphWrapper graphWrapper(serializedGraph.data(), serializedGraph.size());
    const auto& attributes = graphWrapper.getNodeWrapper(0).attributesAs<PointwiseAttributes>();
    const auto* input
        = static_cast<Tensor<float>*>(&tensorBundle.getTensor(attributes.in_0_tensor_uid()));
    const auto* output
        = static_cast<Tensor<float>*>(&tensorBundle.getTensor(attributes.out_0_tensor_uid()));

    CpuReferenceGraphExecutor{}.execute(
        serializedGraph.data(), serializedGraph.size(), variantPack);

    // Swish(x) = x * sigmoid(beta * x) with beta = 0.5
    for(int64_t channel = 0; channel < 3; ++channel)
    {
        for(int64_t height = 0; height < 4; ++height)
        {
            for(int64_t width = 0; width < 4; ++width)
            {
                const auto x = static_cast<double>(input->getHostValue(0, channel, height, width));
                const double expected = x / (1.0 + std::exp(-0.5 * x));
                EXPECT_NEAR(static_cast<double>(output->getHostValue(0, channel, height, width)),
                            expected,
                            1e-5);
            }
        }
    }
}

TEST_F(TestPointwisePlan, ExecutePlanUnaryExp)
{
    const std::vector<int64_t> inputDims = {1, 3, 4, 4};
    const std::vector<int64_t> outputDims = {1, 3, 4, 4};
    const unsigned int seed = getGlobalTestSeed();

    auto [graph, tensorBundle, variantPack]
        = buildPointwiseUnaryGraph(inputDims,
                                   outputDims,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   hipdnn_frontend::PointwiseMode::EXP,
                                   seed,
                                   TensorLayout::NCHW);

    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    CpuReferenceGraphExecutor{}.execute(
        serializedGraph.data(), serializedGraph.size(), variantPack);

    SUCCEED();
}

TEST_F(TestPointwisePlan, ExecutePlanUnarySqrt)
{
    const std::vector<int64_t> inputDims = {1, 3, 4, 4};
    const std::vector<int64_t> outputDims = {1, 3, 4, 4};
    const unsigned int seed = getGlobalTestSeed();

    auto [graph, tensorBundle, variantPack]
        = buildPointwiseUnaryGraph(inputDims,
                                   outputDims,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   hipdnn_frontend::PointwiseMode::SQRT,
                                   seed,
                                   TensorLayout::NCHW);

    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    CpuReferenceGraphExecutor{}.execute(
        serializedGraph.data(), serializedGraph.size(), variantPack);

    SUCCEED();
}

TEST_F(TestPointwisePlan, ExecutePlanUnaryRsqrt)
{
    const std::vector<int64_t> inputDims = {1, 3, 4, 4};
    const std::vector<int64_t> outputDims = {1, 3, 4, 4};
    const unsigned int seed = getGlobalTestSeed();

    auto [graph, tensorBundle, variantPack]
        = buildPointwiseUnaryGraph(inputDims,
                                   outputDims,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   hipdnn_frontend::PointwiseMode::RSQRT,
                                   seed,
                                   TensorLayout::NCHW);

    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    CpuReferenceGraphExecutor{}.execute(
        serializedGraph.data(), serializedGraph.size(), variantPack);

    SUCCEED();
}

TEST_F(TestPointwisePlan, ExecutePlanUnaryReciprocal)
{
    const std::vector<int64_t> inputDims = {1, 3, 4, 4};
    const std::vector<int64_t> outputDims = {1, 3, 4, 4};
    const unsigned int seed = getGlobalTestSeed();

    auto [graph, tensorBundle, variantPack]
        = buildPointwiseUnaryGraph(inputDims,
                                   outputDims,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   hipdnn_frontend::PointwiseMode::RECIPROCAL,
                                   seed,
                                   TensorLayout::NCHW);

    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    CpuReferenceGraphExecutor{}.execute(
        serializedGraph.data(), serializedGraph.size(), variantPack);

    SUCCEED();
}

TEST_F(TestPointwisePlan, ExecutePlanUnarySin)
{
    const std::vector<int64_t> inputDims = {1, 3, 4, 4};
    const std::vector<int64_t> outputDims = {1, 3, 4, 4};
    const unsigned int seed = getGlobalTestSeed();

    auto [graph, tensorBundle, variantPack]
        = buildPointwiseUnaryGraph(inputDims,
                                   outputDims,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   hipdnn_frontend::PointwiseMode::SIN,
                                   seed,
                                   TensorLayout::NCHW);

    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    CpuReferenceGraphExecutor{}.execute(
        serializedGraph.data(), serializedGraph.size(), variantPack);

    SUCCEED();
}

TEST_F(TestPointwisePlan, ExecutePlanUnaryTan)
{
    const std::vector<int64_t> inputDims = {1, 3, 4, 4};
    const std::vector<int64_t> outputDims = {1, 3, 4, 4};
    const unsigned int seed = getGlobalTestSeed();

    auto [graph, tensorBundle, variantPack]
        = buildPointwiseUnaryGraph(inputDims,
                                   outputDims,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   hipdnn_frontend::PointwiseMode::TAN,
                                   seed,
                                   TensorLayout::NCHW);

    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    CpuReferenceGraphExecutor{}.execute(
        serializedGraph.data(), serializedGraph.size(), variantPack);

    SUCCEED();
}

TEST_F(TestPointwisePlan, ExecutePlanUnaryErf)
{
    const std::vector<int64_t> inputDims = {1, 3, 4, 4};
    const std::vector<int64_t> outputDims = {1, 3, 4, 4};
    const unsigned int seed = getGlobalTestSeed();

    auto [graph, tensorBundle, variantPack]
        = buildPointwiseUnaryGraph(inputDims,
                                   outputDims,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   hipdnn_frontend::PointwiseMode::ERF,
                                   seed,
                                   TensorLayout::NCHW);

    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    CpuReferenceGraphExecutor{}.execute(
        serializedGraph.data(), serializedGraph.size(), variantPack);

    SUCCEED();
}

TEST_F(TestPointwisePlan, ExecutePlanUnaryCeil)
{
    const std::vector<int64_t> inputDims = {1, 3, 4, 4};
    const std::vector<int64_t> outputDims = {1, 3, 4, 4};
    const unsigned int seed = getGlobalTestSeed();

    auto [graph, tensorBundle, variantPack]
        = buildPointwiseUnaryGraph(inputDims,
                                   outputDims,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   hipdnn_frontend::PointwiseMode::CEIL,
                                   seed,
                                   TensorLayout::NCHW);

    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    CpuReferenceGraphExecutor{}.execute(
        serializedGraph.data(), serializedGraph.size(), variantPack);

    SUCCEED();
}

TEST_F(TestPointwisePlan, ExecutePlanUnaryEluFwdWithAlpha)
{
    const std::vector<int64_t> inputDims = {1, 3, 4, 4};
    const std::vector<int64_t> outputDims = {1, 3, 4, 4};
    const unsigned int seed = getGlobalTestSeed();

    auto [graph, tensorBundle, variantPack]
        = buildPointwiseUnaryGraph(inputDims,
                                   outputDims,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   hipdnn_frontend::PointwiseMode::ELU_FWD,
                                   seed,
                                   TensorLayout::NCHW,
                                   std::nullopt,
                                   std::nullopt,
                                   std::nullopt,
                                   std::nullopt,
                                   0.5f);

    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    CpuReferenceGraphExecutor{}.execute(
        serializedGraph.data(), serializedGraph.size(), variantPack);

    SUCCEED();
}

TEST_F(TestPointwisePlan, ExecutePlanUnarySoftplusFwdWithBeta)
{
    const std::vector<int64_t> inputDims = {1, 3, 4, 4};
    const std::vector<int64_t> outputDims = {1, 3, 4, 4};
    const unsigned int seed = getGlobalTestSeed();

    auto [graph, tensorBundle, variantPack]
        = buildPointwiseUnaryGraph(inputDims,
                                   outputDims,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   hipdnn_frontend::PointwiseMode::SOFTPLUS_FWD,
                                   seed,
                                   TensorLayout::NCHW,
                                   std::nullopt,
                                   std::nullopt,
                                   std::nullopt,
                                   std::nullopt,
                                   std::nullopt,
                                   2.0f);

    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    CpuReferenceGraphExecutor{}.execute(
        serializedGraph.data(), serializedGraph.size(), variantPack);

    SUCCEED();
}

TEST_F(TestPointwisePlan, ExecutePlanBinaryDiv)
{
    const std::vector<int64_t> input1Dims = {1, 3, 2, 2};
    const std::vector<int64_t> input2Dims = {1, 3, 2, 2};
    const std::vector<int64_t> outputDims = {1, 3, 2, 2};
    const unsigned int seed = getGlobalTestSeed();

    // Build graph using new GraphTensorBundle pattern
    auto [graph, tensorBundle, variantPack]
        = buildPointwiseBinaryGraph(input1Dims,
                                    input2Dims,
                                    outputDims,
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    hipdnn_frontend::PointwiseMode::DIV,
                                    seed,
                                    TensorLayout::NCHW);

    // Execute using CpuReferenceGraphExecutor
    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    CpuReferenceGraphExecutor{}.execute(
        serializedGraph.data(), serializedGraph.size(), variantPack);

    // Verify execution succeeded
    SUCCEED();
}

TEST_F(TestPointwisePlan, ExecutePlanGeluBwd)
{
    const std::vector<int64_t> input1Dims = {1, 3, 2, 2};
    const std::vector<int64_t> input2Dims = {1, 3, 2, 2};
    const std::vector<int64_t> outputDims = {1, 3, 2, 2};
    const unsigned int seed = getGlobalTestSeed();

    // Build graph using new GraphTensorBundle pattern
    auto [graph, tensorBundle, variantPack]
        = buildPointwiseBinaryGraph(input1Dims,
                                    input2Dims,
                                    outputDims,
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    hipdnn_frontend::PointwiseMode::GELU_BWD,
                                    seed,
                                    TensorLayout::NCHW);

    // Execute using CpuReferenceGraphExecutor
    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    CpuReferenceGraphExecutor{}.execute(
        serializedGraph.data(), serializedGraph.size(), variantPack);

    // Verify execution succeeded
    SUCCEED();
}

TEST_F(TestPointwisePlan, ExecutePlanSoftplusBwdWithBeta)
{
    const std::vector<int64_t> input1Dims = {1, 3, 2, 2};
    const std::vector<int64_t> input2Dims = {1, 3, 2, 2};
    const std::vector<int64_t> outputDims = {1, 3, 2, 2};
    const unsigned int seed = getGlobalTestSeed();

    // Build graph using new GraphTensorBundle pattern
    auto [graph, tensorBundle, variantPack]
        = buildPointwiseBinaryGraph(input1Dims,
                                    input2Dims,
                                    outputDims,
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    hipdnn_frontend::PointwiseMode::SOFTPLUS_BWD,
                                    seed,
                                    TensorLayout::NCHW,
                                    std::nullopt,
                                    std::nullopt,
                                    std::nullopt,
                                    std::nullopt,
                                    std::nullopt,
                                    2.0f);

    // Execute using CpuReferenceGraphExecutor
    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    CpuReferenceGraphExecutor{}.execute(
        serializedGraph.data(), serializedGraph.size(), variantPack);

    // Verify execution succeeded
    SUCCEED();
}

TEST_F(TestPointwisePlan, ExecutePlanEluBwdWithAlpha)
{
    const std::vector<int64_t> input1Dims = {1, 3, 2, 2};
    const std::vector<int64_t> input2Dims = {1, 3, 2, 2};
    const std::vector<int64_t> outputDims = {1, 3, 2, 2};
    const unsigned int seed = getGlobalTestSeed();

    // Build graph using new GraphTensorBundle pattern
    auto [graph, tensorBundle, variantPack]
        = buildPointwiseBinaryGraph(input1Dims,
                                    input2Dims,
                                    outputDims,
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    hipdnn_frontend::PointwiseMode::ELU_BWD,
                                    seed,
                                    TensorLayout::NCHW,
                                    std::nullopt,
                                    std::nullopt,
                                    std::nullopt,
                                    std::nullopt,
                                    0.5f);

    // Execute using CpuReferenceGraphExecutor
    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    CpuReferenceGraphExecutor{}.execute(
        serializedGraph.data(), serializedGraph.size(), variantPack);

    // Verify execution succeeded
    SUCCEED();
}

TEST_F(TestPointwisePlan, ParameterizedReluBwdBfloat16LowerClipEdgeCase)
{
    // In bfloat16, a value higher than lowerClip can be equal to lowerClip if lowerClip itself is cast to bfloat16. However, this value should still register as higher than lowerClip.

    const float lowerClip = 0.1f;
    const float upperClip = 0.5f;
    const float lowerSlope = 0.2f;

    auto [graph, tensorBundle, variantPack]
        = buildPointwiseBinaryGraph({1, 1, 1, 1},
                                    {1, 1, 1, 1},
                                    {1, 1, 1, 1},
                                    DataType::BFLOAT16,
                                    DataType::BFLOAT16,
                                    DataType::FLOAT,
                                    DataType::BFLOAT16,
                                    hipdnn_frontend::PointwiseMode::RELU_BWD,
                                    0,
                                    TensorLayout::NCHW,
                                    lowerClip,
                                    upperClip,
                                    lowerSlope);

    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    const GraphWrapper graphWrapper(serializedGraph.data(), serializedGraph.size());
    const INodeWrapper& node = graphWrapper.getNodeWrapper(0);
    const auto& attributes
        = node.attributesAs<hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes>();

    auto* dyTensor
        = static_cast<Tensor<bfloat16>*>(&tensorBundle.getTensor(attributes.in_0_tensor_uid()));
    auto* yTensor = static_cast<Tensor<bfloat16>*>(
        &tensorBundle.getTensor(attributes.in_1_tensor_uid().value()));
    auto* outputTensor
        = static_cast<Tensor<bfloat16>*>(&tensorBundle.getTensor(attributes.out_0_tensor_uid()));

    // Set y to a value slightly above lowerClip. In bfloat16, this is equal to lowerClip
    dyTensor->setHostValue(safeTestTypeCast<bfloat16>(1.0f), 0, 0, 0, 0);
    dyTensor->markHostModified();
    yTensor->setHostValue(safeTestTypeCast<bfloat16>(0.10009765625f), 0, 0, 0, 0);
    yTensor->markHostModified();

    EXPECT_NE(lowerClip, safeTestTypeCast<float>(yTensor->getHostValue(0, 0, 0, 0)));
    EXPECT_EQ(safeTestTypeCast<bfloat16>(lowerClip), yTensor->getHostValue(0, 0, 0, 0));

    CpuReferenceGraphExecutor{}.execute(
        serializedGraph.data(), serializedGraph.size(), variantPack);

    outputTensor->markDeviceModified();

    EXPECT_EQ(outputTensor->getHostValue(0, 0, 0, 0), dyTensor->getHostValue(0, 0, 0, 0));

    SUCCEED();
}

TEST(TestPointwisePlanBuilder, PlanConstructionUnary)
{
    const std::vector<int64_t> inputDims = {1, 3, 4, 4};
    const std::vector<int64_t> outputDims = {1, 3, 4, 4};

    auto [graph, tensorBundle, variantPack]
        = buildPointwiseUnaryGraph(inputDims,
                                   outputDims,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   hipdnn_frontend::PointwiseMode::RELU_FWD,
                                   1,
                                   TensorLayout::NCHW);

    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        serializedGraph.data(), serializedGraph.size());

    const PointwisePlanBuilder<DataType::FLOAT,
                               DataType::FLOAT,
                               DataType::FLOAT,
                               DataType::FLOAT,
                               DataType::FLOAT>
        patient;
    auto builtPlan = patient.buildNodePlan(graphWrap, graphWrap.getNode(0));

    const bool result
        = dynamic_cast<PointwisePlan<float, float, float, float, float>*>(builtPlan.get())
          != nullptr;
    EXPECT_TRUE(result);
}

TEST(TestPointwisePlanBuilder, PlanConstructionBinary)
{
    const std::vector<int64_t> input1Dims = {1, 3, 2, 2};
    const std::vector<int64_t> input2Dims = {1, 3, 2, 2};
    const std::vector<int64_t> outputDims = {1, 3, 2, 2};

    auto [graph, tensorBundle, variantPack]
        = buildPointwiseBinaryGraph(input1Dims,
                                    input2Dims,
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
    auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        serializedGraph.data(), serializedGraph.size());

    const PointwisePlanBuilder<DataType::FLOAT,
                               DataType::FLOAT,
                               DataType::FLOAT,
                               DataType::FLOAT,
                               DataType::FLOAT>
        patient;
    auto builtPlan = patient.buildNodePlan(graphWrap, graphWrap.getNode(0));

    const bool result
        = dynamic_cast<PointwisePlan<float, float, float, float, float>*>(builtPlan.get())
          != nullptr;
    EXPECT_TRUE(result);
}

TEST(TestPointwisePlanBuilder, PlanConstructionUnaryGelu)
{
    const std::vector<int64_t> inputDims = {1, 3, 4, 4};
    const std::vector<int64_t> outputDims = {1, 3, 4, 4};

    auto [graph, tensorBundle, variantPack]
        = buildPointwiseUnaryGraph(inputDims,
                                   outputDims,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   hipdnn_frontend::PointwiseMode::GELU_FWD,
                                   1,
                                   TensorLayout::NCHW);

    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        serializedGraph.data(), serializedGraph.size());

    const PointwisePlanBuilder<DataType::FLOAT,
                               DataType::FLOAT,
                               DataType::FLOAT,
                               DataType::FLOAT,
                               DataType::FLOAT>
        patient;
    auto builtPlan = patient.buildNodePlan(graphWrap, graphWrap.getNode(0));

    const bool result
        = dynamic_cast<PointwisePlan<float, float, float, float, float>*>(builtPlan.get())
          != nullptr;
    EXPECT_TRUE(result);
}

TEST(TestPointwisePlanBuilder, PlanConstructionUnaryGeluApproxTanh)
{
    const std::vector<int64_t> inputDims = {1, 3, 4, 4};
    const std::vector<int64_t> outputDims = {1, 3, 4, 4};

    auto [graph, tensorBundle, variantPack]
        = buildPointwiseUnaryGraph(inputDims,
                                   outputDims,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   hipdnn_frontend::PointwiseMode::GELU_APPROX_TANH_FWD,
                                   1,
                                   TensorLayout::NCHW);

    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        serializedGraph.data(), serializedGraph.size());

    const PointwisePlanBuilder<DataType::FLOAT,
                               DataType::FLOAT,
                               DataType::FLOAT,
                               DataType::FLOAT,
                               DataType::FLOAT>
        patient;
    auto builtPlan = patient.buildNodePlan(graphWrap, graphWrap.getNode(0));

    const bool result
        = dynamic_cast<PointwisePlan<float, float, float, float, float>*>(builtPlan.get())
          != nullptr;
    EXPECT_TRUE(result);
}

TEST(TestPointwisePlanBuilder, PlanConstructionUnarySwish)
{
    const std::vector<int64_t> inputDims = {1, 3, 4, 4};
    const std::vector<int64_t> outputDims = {1, 3, 4, 4};

    auto [graph, tensorBundle, variantPack]
        = buildPointwiseUnaryGraph(inputDims,
                                   outputDims,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   hipdnn_frontend::PointwiseMode::SWISH_FWD,
                                   1,
                                   TensorLayout::NCHW);

    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        serializedGraph.data(), serializedGraph.size());

    const PointwisePlanBuilder<DataType::FLOAT,
                               DataType::FLOAT,
                               DataType::FLOAT,
                               DataType::FLOAT,
                               DataType::FLOAT>
        patient;
    auto builtPlan = patient.buildNodePlan(graphWrap, graphWrap.getNode(0));

    const bool result
        = dynamic_cast<PointwisePlan<float, float, float, float, float>*>(builtPlan.get())
          != nullptr;
    EXPECT_TRUE(result);
}

TEST(TestPointwisePlanBuilder, IsApplicableUnary)
{
    const std::vector<int64_t> inputDims = {1, 3, 4, 4};
    const std::vector<int64_t> outputDims = {1, 3, 4, 4};

    auto [graph, tensorBundle, variantPack]
        = buildPointwiseUnaryGraph(inputDims,
                                   outputDims,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   hipdnn_frontend::PointwiseMode::RELU_FWD,
                                   1,
                                   TensorLayout::NCHW);

    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        serializedGraph.data(), serializedGraph.size());

    const PointwisePlanBuilder<DataType::FLOAT,
                               DataType::FLOAT,
                               DataType::FLOAT,
                               DataType::FLOAT,
                               DataType::FLOAT>
        floatPlanBuilder;
    EXPECT_TRUE(floatPlanBuilder.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap()));

    // Test with mismatched data types
    const PointwisePlanBuilder<DataType::HALF,
                               DataType::HALF,
                               DataType::HALF,
                               DataType::FLOAT,
                               DataType::HALF>
        badTypesPlanBuilder;
    EXPECT_FALSE(badTypesPlanBuilder.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap()));
}

TEST(TestPointwisePlanBuilder, IsApplicableBinary)
{
    const std::vector<int64_t> input1Dims = {1, 3, 2, 2};
    const std::vector<int64_t> input2Dims = {1, 3, 2, 2};
    const std::vector<int64_t> outputDims = {1, 3, 2, 2};

    auto [graph, tensorBundle, variantPack]
        = buildPointwiseBinaryGraph(input1Dims,
                                    input2Dims,
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
    auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        serializedGraph.data(), serializedGraph.size());

    const PointwisePlanBuilder<DataType::FLOAT,
                               DataType::FLOAT,
                               DataType::FLOAT,
                               DataType::FLOAT,
                               DataType::FLOAT>
        floatPlanBuilder;
    EXPECT_TRUE(floatPlanBuilder.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap()));

    // Test with missing tensor - erase a tensor from the map
    auto tensorMapCopy = graphWrap.getTensorMap();
    tensorMapCopy.erase(2);
    EXPECT_FALSE(floatPlanBuilder.isApplicable(graphWrap.getNode(0), tensorMapCopy));
}

TEST(TestPointwisePlanBuilder, IsApplicableUnaryGelu)
{
    const std::vector<int64_t> inputDims = {1, 3, 4, 4};
    const std::vector<int64_t> outputDims = {1, 3, 4, 4};

    auto [graph, tensorBundle, variantPack]
        = buildPointwiseUnaryGraph(inputDims,
                                   outputDims,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   hipdnn_frontend::PointwiseMode::GELU_FWD,
                                   1,
                                   TensorLayout::NCHW);

    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        serializedGraph.data(), serializedGraph.size());

    const PointwisePlanBuilder<DataType::FLOAT,
                               DataType::FLOAT,
                               DataType::FLOAT,
                               DataType::FLOAT,
                               DataType::FLOAT>
        floatPlanBuilder;
    EXPECT_TRUE(floatPlanBuilder.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap()));

    const PointwisePlanBuilder<DataType::HALF,
                               DataType::HALF,
                               DataType::HALF,
                               DataType::FLOAT,
                               DataType::HALF>
        badTypesPlanBuilder;
    EXPECT_FALSE(badTypesPlanBuilder.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap()));
}

TEST(TestPointwisePlanBuilder, IsApplicableUnaryGeluApproxTanh)
{
    const std::vector<int64_t> inputDims = {1, 3, 4, 4};
    const std::vector<int64_t> outputDims = {1, 3, 4, 4};

    auto [graph, tensorBundle, variantPack]
        = buildPointwiseUnaryGraph(inputDims,
                                   outputDims,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   hipdnn_frontend::PointwiseMode::GELU_APPROX_TANH_FWD,
                                   1,
                                   TensorLayout::NCHW);

    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        serializedGraph.data(), serializedGraph.size());

    const PointwisePlanBuilder<DataType::FLOAT,
                               DataType::FLOAT,
                               DataType::FLOAT,
                               DataType::FLOAT,
                               DataType::FLOAT>
        floatPlanBuilder;
    EXPECT_TRUE(floatPlanBuilder.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap()));

    const PointwisePlanBuilder<DataType::HALF,
                               DataType::HALF,
                               DataType::HALF,
                               DataType::FLOAT,
                               DataType::HALF>
        badTypesPlanBuilder;
    EXPECT_FALSE(badTypesPlanBuilder.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap()));
}

TEST(TestPointwisePlanBuilder, IsApplicableUnarySwish)
{
    const std::vector<int64_t> inputDims = {1, 3, 4, 4};
    const std::vector<int64_t> outputDims = {1, 3, 4, 4};

    auto [graph, tensorBundle, variantPack]
        = buildPointwiseUnaryGraph(inputDims,
                                   outputDims,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   hipdnn_frontend::PointwiseMode::SWISH_FWD,
                                   1,
                                   TensorLayout::NCHW);

    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        serializedGraph.data(), serializedGraph.size());

    const PointwisePlanBuilder<DataType::FLOAT,
                               DataType::FLOAT,
                               DataType::FLOAT,
                               DataType::FLOAT,
                               DataType::FLOAT>
        floatPlanBuilder;
    EXPECT_TRUE(floatPlanBuilder.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap()));

    const PointwisePlanBuilder<DataType::HALF,
                               DataType::HALF,
                               DataType::HALF,
                               DataType::FLOAT,
                               DataType::HALF>
        badTypesPlanBuilder;
    EXPECT_FALSE(badTypesPlanBuilder.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap()));
}

TEST(TestPointwisePlanBuilder, PlanConstructionUnaryExp)
{
    const std::vector<int64_t> inputDims = {1, 3, 4, 4};
    const std::vector<int64_t> outputDims = {1, 3, 4, 4};

    auto [graph, tensorBundle, variantPack]
        = buildPointwiseUnaryGraph(inputDims,
                                   outputDims,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   hipdnn_frontend::PointwiseMode::EXP,
                                   1,
                                   TensorLayout::NCHW);

    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        serializedGraph.data(), serializedGraph.size());

    const PointwisePlanBuilder<DataType::FLOAT,
                               DataType::FLOAT,
                               DataType::FLOAT,
                               DataType::FLOAT,
                               DataType::FLOAT>
        patient;
    auto builtPlan = patient.buildNodePlan(graphWrap, graphWrap.getNode(0));

    const bool result
        = dynamic_cast<PointwisePlan<float, float, float, float>*>(builtPlan.get()) != nullptr;
    EXPECT_TRUE(result);
}

TEST(TestPointwisePlanBuilder, PlanConstructionBinaryDiv)
{
    const std::vector<int64_t> input1Dims = {1, 3, 2, 2};
    const std::vector<int64_t> input2Dims = {1, 3, 2, 2};
    const std::vector<int64_t> outputDims = {1, 3, 2, 2};

    auto [graph, tensorBundle, variantPack]
        = buildPointwiseBinaryGraph(input1Dims,
                                    input2Dims,
                                    outputDims,
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    hipdnn_frontend::PointwiseMode::DIV,
                                    1,
                                    TensorLayout::NCHW);

    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        serializedGraph.data(), serializedGraph.size());

    const PointwisePlanBuilder<DataType::FLOAT,
                               DataType::FLOAT,
                               DataType::FLOAT,
                               DataType::FLOAT,
                               DataType::FLOAT>
        patient;
    auto builtPlan = patient.buildNodePlan(graphWrap, graphWrap.getNode(0));

    const bool result
        = dynamic_cast<PointwisePlan<float, float, float, float>*>(builtPlan.get()) != nullptr;
    EXPECT_TRUE(result);
}

TEST(TestPointwisePlanBuilder, IsApplicableUnaryExp)
{
    const std::vector<int64_t> inputDims = {1, 3, 4, 4};
    const std::vector<int64_t> outputDims = {1, 3, 4, 4};

    auto [graph, tensorBundle, variantPack]
        = buildPointwiseUnaryGraph(inputDims,
                                   outputDims,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   hipdnn_frontend::PointwiseMode::EXP,
                                   1,
                                   TensorLayout::NCHW);

    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        serializedGraph.data(), serializedGraph.size());

    const PointwisePlanBuilder<DataType::FLOAT,
                               DataType::FLOAT,
                               DataType::FLOAT,
                               DataType::FLOAT,
                               DataType::FLOAT>
        floatPlanBuilder;
    EXPECT_TRUE(floatPlanBuilder.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap()));

    const PointwisePlanBuilder<DataType::HALF,
                               DataType::HALF,
                               DataType::HALF,
                               DataType::FLOAT,
                               DataType::HALF>
        badTypesPlanBuilder;
    EXPECT_FALSE(badTypesPlanBuilder.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap()));
}

TEST(TestPointwisePlanBuilder, PlanConstructionGenIndex)
{
    auto [graph, tensorBundle, variantPack]
        = buildPointwiseUnaryGraph({1, 3, 4, 4},
                                   {1, 3, 4, 4},
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::INT32,
                                   hipdnn_frontend::PointwiseMode::GEN_INDEX,
                                   1,
                                   TensorLayout::NCHW,
                                   std::nullopt,
                                   std::nullopt,
                                   std::nullopt,
                                   std::nullopt,
                                   std::nullopt,
                                   std::nullopt,
                                   int64_t{1});

    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        serializedGraph.data(), serializedGraph.size());

    const PointwisePlanBuilder<DataType::FLOAT,
                               DataType::FLOAT,
                               DataType::FLOAT,
                               DataType::FLOAT,
                               DataType::INT32>
        patient;
    auto builtPlan = patient.buildNodePlan(graphWrap, graphWrap.getNode(0));

    const bool result
        = dynamic_cast<PointwisePlan<float, float, float, int32_t>*>(builtPlan.get()) != nullptr;
    EXPECT_TRUE(result);
}

TEST(TestPointwisePlanBuilder, IsApplicableGenIndexWithoutAxisIsDeclined)
{
    auto [graph, tensorBundle, variantPack]
        = buildPointwiseUnaryGraph({1, 3, 4, 4},
                                   {1, 3, 4, 4},
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::INT32,
                                   hipdnn_frontend::PointwiseMode::GEN_INDEX);

    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        serializedGraph.data(), serializedGraph.size());

    const PointwisePlanBuilder<DataType::FLOAT,
                               DataType::FLOAT,
                               DataType::FLOAT,
                               DataType::FLOAT,
                               DataType::INT32>
        planBuilder;
    EXPECT_FALSE(planBuilder.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap()));
}

TEST(TestPointwisePlanBuilder, IsApplicableGenIndex)
{
    auto [graph, tensorBundle, variantPack]
        = buildPointwiseUnaryGraph({1, 3, 4, 4},
                                   {1, 3, 4, 4},
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::INT32,
                                   hipdnn_frontend::PointwiseMode::GEN_INDEX,
                                   1,
                                   TensorLayout::NCHW,
                                   std::nullopt,
                                   std::nullopt,
                                   std::nullopt,
                                   std::nullopt,
                                   std::nullopt,
                                   std::nullopt,
                                   int64_t{1});

    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        serializedGraph.data(), serializedGraph.size());

    const PointwisePlanBuilder<DataType::FLOAT,
                               DataType::FLOAT,
                               DataType::FLOAT,
                               DataType::FLOAT,
                               DataType::INT32>
        floatPlanBuilder;
    EXPECT_TRUE(floatPlanBuilder.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap()));

    const PointwisePlanBuilder<DataType::HALF,
                               DataType::HALF,
                               DataType::HALF,
                               DataType::FLOAT,
                               DataType::HALF>
        badTypesPlanBuilder;
    EXPECT_FALSE(badTypesPlanBuilder.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap()));
}

TEST(TestPointwisePlanBuilder, IsApplicableLogicalOperationsAreBooleanOnly)
{
    auto [graph, tensorBundle, variantPack]
        = buildPointwiseBinaryGraph({1, 1, 1, 4},
                                    {1, 1, 1, 4},
                                    {1, 1, 1, 4},
                                    DataType::BOOLEAN,
                                    DataType::BOOLEAN,
                                    DataType::FLOAT,
                                    DataType::BOOLEAN,
                                    hipdnn_frontend::PointwiseMode::LOGICAL_AND);

    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    const GraphWrapper graphWrapper(serializedGraph.data(), serializedGraph.size());

    const PointwisePlanBuilder<DataType::BOOLEAN,
                               DataType::BOOLEAN,
                               DataType::BOOLEAN,
                               DataType::FLOAT,
                               DataType::BOOLEAN>
        booleanPlanBuilder;
    EXPECT_TRUE(
        booleanPlanBuilder.isApplicable(graphWrapper.getNode(0), graphWrapper.getTensorMap()));

    const PointwisePlanBuilder<DataType::FLOAT,
                               DataType::FLOAT,
                               DataType::FLOAT,
                               DataType::FLOAT,
                               DataType::BOOLEAN>
        floatPlanBuilder;
    EXPECT_FALSE(
        floatPlanBuilder.isApplicable(graphWrapper.getNode(0), graphWrapper.getTensorMap()));
}

TEST(TestPointwisePlanBuilder, UnsupportedOperation)
{
    // Tensors 1 (input) and 2 (output) exist and have the right type, so only the operation decides
    // whether the builder accepts the node.
    const std::vector<int64_t> dims = {1};
    const std::vector<int64_t> strides = {1};
    flatbuffers::FlatBufferBuilder in0Builder;
    in0Builder.Finish(
        CreateTensorAttributesDirect(in0Builder, 1, "in", DataType::FLOAT, &strides, &dims));
    flatbuffers::FlatBufferBuilder out0Builder;
    out0Builder.Finish(
        CreateTensorAttributesDirect(out0Builder, 2, "out", DataType::FLOAT, &strides, &dims));
    const std::unordered_map<int64_t, const TensorAttributes*> tensorMap
        = {{1, flatbuffers::GetRoot<TensorAttributes>(in0Builder.GetBufferPointer())},
           {2, flatbuffers::GetRoot<TensorAttributes>(out0Builder.GetBufferPointer())}};

    const auto buildNode
        = [](flatbuffers::FlatBufferBuilder& fbb, PointwiseMode mode) -> const Node& {
        auto pointwiseAttributes = CreatePointwiseAttributes(fbb,
                                                             mode,
                                                             flatbuffers::nullopt,
                                                             flatbuffers::nullopt,
                                                             flatbuffers::nullopt,
                                                             flatbuffers::nullopt,
                                                             1,
                                                             flatbuffers::nullopt,
                                                             flatbuffers::nullopt,
                                                             2);
        fbb.Finish(CreateNode(fbb,
                              0,
                              DataType::FLOAT,
                              NodeAttributes::PointwiseAttributes,
                              pointwiseAttributes.Union()));
        return *flatbuffers::GetRoot<Node>(fbb.GetBufferPointer());
    };

    const PointwisePlanBuilder<DataType::FLOAT,
                               DataType::FLOAT,
                               DataType::FLOAT,
                               DataType::FLOAT,
                               DataType::FLOAT>
        planBuilder;

    flatbuffers::FlatBufferBuilder supportedBuilder;
    EXPECT_TRUE(
        planBuilder.isApplicable(buildNode(supportedBuilder, PointwiseMode::RELU_FWD), tensorMap));

    // Every classified pointwise mode is implemented, so use an unclassified (UNSET) operation.
    flatbuffers::FlatBufferBuilder unsetBuilder;
    EXPECT_FALSE(
        planBuilder.isApplicable(buildNode(unsetBuilder, PointwiseMode::UNSET), tensorMap));
}

template <typename T>
static void executeTernaryBinarySelectBroadcast()
{
    constexpr auto DATA_TYPE = nativeTypeToDataType<T>();
    const std::vector<int64_t> input0Dims = {1, 2, 1, 1};
    const std::vector<int64_t> input1Dims = {2, 1, 1, 3};
    const std::vector<int64_t> maskDims = {2, 1, 1, 3};
    const std::vector<int64_t> outputDims = {2, 2, 1, 3};

    auto [graph, tensorBundle, variantPack]
        = buildPointwiseTernaryGraph(input0Dims,
                                     input1Dims,
                                     maskDims,
                                     outputDims,
                                     DATA_TYPE,
                                     DATA_TYPE,
                                     DataType::BOOLEAN,
                                     DataType::FLOAT,
                                     DATA_TYPE,
                                     hipdnn_frontend::PointwiseMode::BINARY_SELECT);

    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    const GraphWrapper graphWrapper(serializedGraph.data(), serializedGraph.size());
    const auto& attributes = graphWrapper.getNodeWrapper(0).attributesAs<PointwiseAttributes>();

    auto* input0 = static_cast<Tensor<T>*>(&tensorBundle.getTensor(attributes.in_0_tensor_uid()));
    auto* input1
        = static_cast<Tensor<T>*>(&tensorBundle.getTensor(attributes.in_1_tensor_uid().value()));
    auto* mask
        = static_cast<Tensor<bool>*>(&tensorBundle.getTensor(attributes.in_2_tensor_uid().value()));
    auto* output = static_cast<Tensor<T>*>(&tensorBundle.getTensor(attributes.out_0_tensor_uid()));

    for(int64_t channel = 0; channel < 2; ++channel)
    {
        input0->setHostValue(static_cast<T>(10.0f + static_cast<float>(channel)), 0, channel, 0, 0);
    }
    for(int64_t batch = 0; batch < 2; ++batch)
    {
        for(int64_t width = 0; width < 3; ++width)
        {
            input1->setHostValue(
                static_cast<T>(-100.0f * static_cast<float>(batch + 1) - static_cast<float>(width)),
                batch,
                0,
                0,
                width);
            mask->setHostValue((batch + width) % 2 == 0, batch, 0, 0, width);
        }
    }
    input0->markHostModified();
    input1->markHostModified();
    mask->markHostModified();

    CpuReferenceGraphExecutor{}.execute(
        serializedGraph.data(), serializedGraph.size(), variantPack);

    for(int64_t batch = 0; batch < 2; ++batch)
    {
        for(int64_t channel = 0; channel < 2; ++channel)
        {
            for(int64_t width = 0; width < 3; ++width)
            {
                const bool selected = (batch + width) % 2 == 0;
                const float expected = selected ? 10.0f + static_cast<float>(channel)
                                                : -100.0f * static_cast<float>(batch + 1)
                                                      - static_cast<float>(width);
                EXPECT_FLOAT_EQ(static_cast<float>(output->getHostValue(batch, channel, 0, width)),
                                expected);
            }
        }
    }
}

TEST(TestTernaryPointwisePlan, ExecuteFloatBinarySelectBroadcast)
{
    executeTernaryBinarySelectBroadcast<float>();
}

TEST(TestTernaryPointwisePlan, ExecuteHalfBinarySelectBroadcast)
{
    executeTernaryBinarySelectBroadcast<half>();
}

TEST(TestTernaryPointwisePlan, ExecuteInt32BinarySelectBroadcast)
{
    executeTernaryBinarySelectBroadcast<int32_t>();
}

TEST(TestTernaryPointwisePlan, ExecuteBfloat16BinarySelectBroadcast)
{
    executeTernaryBinarySelectBroadcast<bfloat16>();
}

TEST_F(TestPointwisePlan, CompareGreaterProducesBooleanMask)
{
    auto [graph, tensorBundle, variantPack]
        = buildPointwiseBinaryGraph({1, 1, 1, 4},
                                    {1, 1, 1, 4},
                                    {1, 1, 1, 4},
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::BOOLEAN,
                                    hipdnn_frontend::PointwiseMode::CMP_GT);

    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    const GraphWrapper graphWrapper(serializedGraph.data(), serializedGraph.size());
    const auto& attributes = graphWrapper.getNodeWrapper(0).attributesAs<PointwiseAttributes>();
    auto* input0
        = static_cast<Tensor<float>*>(&tensorBundle.getTensor(attributes.in_0_tensor_uid()));
    auto* input1 = static_cast<Tensor<float>*>(
        &tensorBundle.getTensor(attributes.in_1_tensor_uid().value()));
    auto* output
        = static_cast<Tensor<bool>*>(&tensorBundle.getTensor(attributes.out_0_tensor_uid()));

    constexpr std::array INPUT0_VALUES = {-2.0f, 0.0f, 3.0f, 4.0f};
    constexpr std::array INPUT1_VALUES = {-3.0f, 0.0f, 5.0f, 1.0f};
    constexpr std::array EXPECTED = {true, false, false, true};
    for(int64_t width = 0; width < 4; ++width)
    {
        const auto widthIndex = static_cast<size_t>(width);
        input0->setHostValue(INPUT0_VALUES[widthIndex], 0, 0, 0, width);
        input1->setHostValue(INPUT1_VALUES[widthIndex], 0, 0, 0, width);
    }
    input0->markHostModified();
    input1->markHostModified();

    CpuReferenceGraphExecutor{}.execute(
        serializedGraph.data(), serializedGraph.size(), variantPack);

    for(int64_t width = 0; width < 4; ++width)
    {
        const auto widthIndex = static_cast<size_t>(width);
        EXPECT_EQ(output->getHostValue(0, 0, 0, width), EXPECTED[widthIndex]);
    }
}

TEST_F(TestPointwisePlan, ComparisonsAcceptInt32Inputs)
{
    using Mode = hipdnn_frontend::PointwiseMode;
    // The last pair differs by one above 2^24, which float cannot represent exactly.
    constexpr std::array<int32_t, 5> INPUT0_VALUES = {-2, 0, 3, 4, 16777217};
    constexpr std::array<int32_t, 5> INPUT1_VALUES = {-3, 0, 5, 1, 16777216};

    const std::vector<std::pair<Mode, bool (*)(int32_t, int32_t)>> comparisons = {
        {Mode::CMP_EQ, [](int32_t a, int32_t b) { return a == b; }},
        {Mode::CMP_NEQ, [](int32_t a, int32_t b) { return a != b; }},
        {Mode::CMP_GT, [](int32_t a, int32_t b) { return a > b; }},
        {Mode::CMP_GE, [](int32_t a, int32_t b) { return a >= b; }},
        {Mode::CMP_LT, [](int32_t a, int32_t b) { return a < b; }},
        {Mode::CMP_LE, [](int32_t a, int32_t b) { return a <= b; }},
    };

    for(const auto& [mode, compare] : comparisons)
    {
        auto [graph, tensorBundle, variantPack] = buildPointwiseBinaryGraph({1, 1, 1, 5},
                                                                            {1, 1, 1, 5},
                                                                            {1, 1, 1, 5},
                                                                            DataType::INT32,
                                                                            DataType::INT32,
                                                                            DataType::FLOAT,
                                                                            DataType::BOOLEAN,
                                                                            mode);

        auto [serializedGraph, serErr] = graph->to_binary();
        ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
        const GraphWrapper graphWrapper(serializedGraph.data(), serializedGraph.size());
        const auto& attributes = graphWrapper.getNodeWrapper(0).attributesAs<PointwiseAttributes>();
        auto* input0
            = static_cast<Tensor<int32_t>*>(&tensorBundle.getTensor(attributes.in_0_tensor_uid()));
        auto* input1 = static_cast<Tensor<int32_t>*>(
            &tensorBundle.getTensor(attributes.in_1_tensor_uid().value()));
        const auto* output
            = static_cast<Tensor<bool>*>(&tensorBundle.getTensor(attributes.out_0_tensor_uid()));

        for(int64_t width = 0; width < 5; ++width)
        {
            const auto widthIndex = static_cast<size_t>(width);
            input0->setHostValue(INPUT0_VALUES[widthIndex], 0, 0, 0, width);
            input1->setHostValue(INPUT1_VALUES[widthIndex], 0, 0, 0, width);
        }
        input0->markHostModified();
        input1->markHostModified();

        CpuReferenceGraphExecutor{}.execute(
            serializedGraph.data(), serializedGraph.size(), variantPack);

        for(int64_t width = 0; width < 5; ++width)
        {
            const auto widthIndex = static_cast<size_t>(width);
            EXPECT_EQ(output->getHostValue(0, 0, 0, width),
                      compare(INPUT0_VALUES[widthIndex], INPUT1_VALUES[widthIndex]))
                << "mode " << static_cast<int>(mode) << ", width " << width;
        }
    }
}

TEST_F(TestPointwisePlan, CompareLessProducesBooleanMask)
{
    auto [graph, tensorBundle, variantPack]
        = buildPointwiseBinaryGraph({1, 1, 1, 4},
                                    {1, 1, 1, 4},
                                    {1, 1, 1, 4},
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::BOOLEAN,
                                    hipdnn_frontend::PointwiseMode::CMP_LT);

    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    const GraphWrapper graphWrapper(serializedGraph.data(), serializedGraph.size());
    const auto& attributes = graphWrapper.getNodeWrapper(0).attributesAs<PointwiseAttributes>();
    auto* input0
        = static_cast<Tensor<float>*>(&tensorBundle.getTensor(attributes.in_0_tensor_uid()));
    auto* input1 = static_cast<Tensor<float>*>(
        &tensorBundle.getTensor(attributes.in_1_tensor_uid().value()));
    auto* output
        = static_cast<Tensor<bool>*>(&tensorBundle.getTensor(attributes.out_0_tensor_uid()));

    constexpr std::array INPUT0_VALUES = {-2.0f, 0.0f, 3.0f, 4.0f};
    constexpr std::array INPUT1_VALUES = {-3.0f, 0.0f, 5.0f, 1.0f};
    constexpr std::array EXPECTED = {false, false, true, false};
    for(int64_t width = 0; width < 4; ++width)
    {
        const auto widthIndex = static_cast<size_t>(width);
        input0->setHostValue(INPUT0_VALUES[widthIndex], 0, 0, 0, width);
        input1->setHostValue(INPUT1_VALUES[widthIndex], 0, 0, 0, width);
    }
    input0->markHostModified();
    input1->markHostModified();

    CpuReferenceGraphExecutor{}.execute(
        serializedGraph.data(), serializedGraph.size(), variantPack);

    for(int64_t width = 0; width < 4; ++width)
    {
        const auto widthIndex = static_cast<size_t>(width);
        EXPECT_EQ(output->getHostValue(0, 0, 0, width), EXPECTED[widthIndex]);
    }
}

TEST_F(TestPointwisePlan, CompareEqualProducesBooleanMask)
{
    auto [graph, tensorBundle, variantPack]
        = buildPointwiseBinaryGraph({1, 1, 1, 4},
                                    {1, 1, 1, 4},
                                    {1, 1, 1, 4},
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::BOOLEAN,
                                    hipdnn_frontend::PointwiseMode::CMP_EQ);

    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    const GraphWrapper graphWrapper(serializedGraph.data(), serializedGraph.size());
    const auto& attributes = graphWrapper.getNodeWrapper(0).attributesAs<PointwiseAttributes>();
    auto* input0
        = static_cast<Tensor<float>*>(&tensorBundle.getTensor(attributes.in_0_tensor_uid()));
    auto* input1 = static_cast<Tensor<float>*>(
        &tensorBundle.getTensor(attributes.in_1_tensor_uid().value()));
    auto* output
        = static_cast<Tensor<bool>*>(&tensorBundle.getTensor(attributes.out_0_tensor_uid()));

    constexpr std::array INPUT0_VALUES = {-2.0f, 0.0f, 3.0f, 4.0f};
    constexpr std::array INPUT1_VALUES = {-3.0f, 0.0f, 5.0f, 1.0f};
    constexpr std::array EXPECTED = {false, true, false, false};
    for(int64_t width = 0; width < 4; ++width)
    {
        const auto widthIndex = static_cast<size_t>(width);
        input0->setHostValue(INPUT0_VALUES[widthIndex], 0, 0, 0, width);
        input1->setHostValue(INPUT1_VALUES[widthIndex], 0, 0, 0, width);
    }
    input0->markHostModified();
    input1->markHostModified();

    CpuReferenceGraphExecutor{}.execute(
        serializedGraph.data(), serializedGraph.size(), variantPack);

    for(int64_t width = 0; width < 4; ++width)
    {
        const auto widthIndex = static_cast<size_t>(width);
        EXPECT_EQ(output->getHostValue(0, 0, 0, width), EXPECTED[widthIndex]);
    }
}

TEST_F(TestPointwisePlan, CompareNotEqualProducesBooleanMask)
{
    auto [graph, tensorBundle, variantPack]
        = buildPointwiseBinaryGraph({1, 1, 1, 4},
                                    {1, 1, 1, 4},
                                    {1, 1, 1, 4},
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::BOOLEAN,
                                    hipdnn_frontend::PointwiseMode::CMP_NEQ);

    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    const GraphWrapper graphWrapper(serializedGraph.data(), serializedGraph.size());
    const auto& attributes = graphWrapper.getNodeWrapper(0).attributesAs<PointwiseAttributes>();
    auto* input0
        = static_cast<Tensor<float>*>(&tensorBundle.getTensor(attributes.in_0_tensor_uid()));
    auto* input1 = static_cast<Tensor<float>*>(
        &tensorBundle.getTensor(attributes.in_1_tensor_uid().value()));
    auto* output
        = static_cast<Tensor<bool>*>(&tensorBundle.getTensor(attributes.out_0_tensor_uid()));

    constexpr std::array INPUT0_VALUES = {-2.0f, 0.0f, 3.0f, 4.0f};
    constexpr std::array INPUT1_VALUES = {-3.0f, 0.0f, 5.0f, 1.0f};
    constexpr std::array EXPECTED = {true, false, true, true};
    for(int64_t width = 0; width < 4; ++width)
    {
        const auto widthIndex = static_cast<size_t>(width);
        input0->setHostValue(INPUT0_VALUES[widthIndex], 0, 0, 0, width);
        input1->setHostValue(INPUT1_VALUES[widthIndex], 0, 0, 0, width);
    }
    input0->markHostModified();
    input1->markHostModified();

    CpuReferenceGraphExecutor{}.execute(
        serializedGraph.data(), serializedGraph.size(), variantPack);

    for(int64_t width = 0; width < 4; ++width)
    {
        const auto widthIndex = static_cast<size_t>(width);
        EXPECT_EQ(output->getHostValue(0, 0, 0, width), EXPECTED[widthIndex]);
    }
}

TEST_F(TestPointwisePlan, CompareGreaterEqualProducesBooleanMask)
{
    auto [graph, tensorBundle, variantPack]
        = buildPointwiseBinaryGraph({1, 1, 1, 4},
                                    {1, 1, 1, 4},
                                    {1, 1, 1, 4},
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::BOOLEAN,
                                    hipdnn_frontend::PointwiseMode::CMP_GE);

    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    const GraphWrapper graphWrapper(serializedGraph.data(), serializedGraph.size());
    const auto& attributes = graphWrapper.getNodeWrapper(0).attributesAs<PointwiseAttributes>();
    auto* input0
        = static_cast<Tensor<float>*>(&tensorBundle.getTensor(attributes.in_0_tensor_uid()));
    auto* input1 = static_cast<Tensor<float>*>(
        &tensorBundle.getTensor(attributes.in_1_tensor_uid().value()));
    auto* output
        = static_cast<Tensor<bool>*>(&tensorBundle.getTensor(attributes.out_0_tensor_uid()));

    constexpr std::array INPUT0_VALUES = {-2.0f, 0.0f, 3.0f, 4.0f};
    constexpr std::array INPUT1_VALUES = {-3.0f, 0.0f, 5.0f, 1.0f};
    constexpr std::array EXPECTED = {true, true, false, true};
    for(int64_t width = 0; width < 4; ++width)
    {
        const auto widthIndex = static_cast<size_t>(width);
        input0->setHostValue(INPUT0_VALUES[widthIndex], 0, 0, 0, width);
        input1->setHostValue(INPUT1_VALUES[widthIndex], 0, 0, 0, width);
    }
    input0->markHostModified();
    input1->markHostModified();

    CpuReferenceGraphExecutor{}.execute(
        serializedGraph.data(), serializedGraph.size(), variantPack);

    for(int64_t width = 0; width < 4; ++width)
    {
        const auto widthIndex = static_cast<size_t>(width);
        EXPECT_EQ(output->getHostValue(0, 0, 0, width), EXPECTED[widthIndex]);
    }
}

TEST_F(TestPointwisePlan, CompareLessEqualProducesBooleanMask)
{
    auto [graph, tensorBundle, variantPack]
        = buildPointwiseBinaryGraph({1, 1, 1, 4},
                                    {1, 1, 1, 4},
                                    {1, 1, 1, 4},
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::BOOLEAN,
                                    hipdnn_frontend::PointwiseMode::CMP_LE);

    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    const GraphWrapper graphWrapper(serializedGraph.data(), serializedGraph.size());
    const auto& attributes = graphWrapper.getNodeWrapper(0).attributesAs<PointwiseAttributes>();
    auto* input0
        = static_cast<Tensor<float>*>(&tensorBundle.getTensor(attributes.in_0_tensor_uid()));
    auto* input1 = static_cast<Tensor<float>*>(
        &tensorBundle.getTensor(attributes.in_1_tensor_uid().value()));
    auto* output
        = static_cast<Tensor<bool>*>(&tensorBundle.getTensor(attributes.out_0_tensor_uid()));

    constexpr std::array INPUT0_VALUES = {-2.0f, 0.0f, 3.0f, 4.0f};
    constexpr std::array INPUT1_VALUES = {-3.0f, 0.0f, 5.0f, 1.0f};
    constexpr std::array EXPECTED = {false, true, true, false};
    for(int64_t width = 0; width < 4; ++width)
    {
        const auto widthIndex = static_cast<size_t>(width);
        input0->setHostValue(INPUT0_VALUES[widthIndex], 0, 0, 0, width);
        input1->setHostValue(INPUT1_VALUES[widthIndex], 0, 0, 0, width);
    }
    input0->markHostModified();
    input1->markHostModified();

    CpuReferenceGraphExecutor{}.execute(
        serializedGraph.data(), serializedGraph.size(), variantPack);

    for(int64_t width = 0; width < 4; ++width)
    {
        const auto widthIndex = static_cast<size_t>(width);
        EXPECT_EQ(output->getHostValue(0, 0, 0, width), EXPECTED[widthIndex]);
    }
}

TEST_F(TestPointwisePlan, LogicalAndProducesBooleanMask)
{
    auto [graph, tensorBundle, variantPack]
        = buildPointwiseBinaryGraph({1, 1, 1, 4},
                                    {1, 1, 1, 4},
                                    {1, 1, 1, 4},
                                    DataType::BOOLEAN,
                                    DataType::BOOLEAN,
                                    DataType::FLOAT,
                                    DataType::BOOLEAN,
                                    hipdnn_frontend::PointwiseMode::LOGICAL_AND);

    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    const GraphWrapper graphWrapper(serializedGraph.data(), serializedGraph.size());
    const auto& attributes = graphWrapper.getNodeWrapper(0).attributesAs<PointwiseAttributes>();
    auto* input0
        = static_cast<Tensor<bool>*>(&tensorBundle.getTensor(attributes.in_0_tensor_uid()));
    auto* input1
        = static_cast<Tensor<bool>*>(&tensorBundle.getTensor(attributes.in_1_tensor_uid().value()));
    auto* output
        = static_cast<Tensor<bool>*>(&tensorBundle.getTensor(attributes.out_0_tensor_uid()));

    constexpr std::array INPUT0_VALUES = {true, true, false, false};
    constexpr std::array INPUT1_VALUES = {true, false, true, false};
    constexpr std::array EXPECTED = {true, false, false, false};
    for(int64_t width = 0; width < 4; ++width)
    {
        const auto widthIndex = static_cast<size_t>(width);
        input0->setHostValue(INPUT0_VALUES[widthIndex], 0, 0, 0, width);
        input1->setHostValue(INPUT1_VALUES[widthIndex], 0, 0, 0, width);
    }
    input0->markHostModified();
    input1->markHostModified();

    CpuReferenceGraphExecutor{}.execute(
        serializedGraph.data(), serializedGraph.size(), variantPack);

    for(int64_t width = 0; width < 4; ++width)
    {
        const auto widthIndex = static_cast<size_t>(width);
        EXPECT_EQ(output->getHostValue(0, 0, 0, width), EXPECTED[widthIndex]);
    }
}

TEST_F(TestPointwisePlan, LogicalNotProducesBooleanMask)
{
    auto [graph, tensorBundle, variantPack]
        = buildPointwiseUnaryGraph({1, 1, 1, 2},
                                   {1, 1, 1, 2},
                                   DataType::BOOLEAN,
                                   DataType::FLOAT,
                                   DataType::BOOLEAN,
                                   hipdnn_frontend::PointwiseMode::LOGICAL_NOT);

    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    const GraphWrapper graphWrapper(serializedGraph.data(), serializedGraph.size());
    const auto& attributes = graphWrapper.getNodeWrapper(0).attributesAs<PointwiseAttributes>();
    auto* input = static_cast<Tensor<bool>*>(&tensorBundle.getTensor(attributes.in_0_tensor_uid()));
    auto* output
        = static_cast<Tensor<bool>*>(&tensorBundle.getTensor(attributes.out_0_tensor_uid()));

    input->setHostValue(true, 0, 0, 0, 0);
    input->setHostValue(false, 0, 0, 0, 1);
    input->markHostModified();

    CpuReferenceGraphExecutor{}.execute(
        serializedGraph.data(), serializedGraph.size(), variantPack);

    EXPECT_FALSE(output->getHostValue(0, 0, 0, 0));
    EXPECT_TRUE(output->getHostValue(0, 0, 0, 1));
}

TEST_F(TestPointwisePlan, SoftplusBwdWithBetaProducesExpectedGradient)
{
    auto [graph, tensorBundle, variantPack]
        = buildPointwiseBinaryGraph({1, 1, 1, 4},
                                    {1, 1, 1, 4},
                                    {1, 1, 1, 4},
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    hipdnn_frontend::PointwiseMode::SOFTPLUS_BWD,
                                    1,
                                    TensorLayout::NCHW,
                                    std::nullopt,
                                    std::nullopt,
                                    std::nullopt,
                                    std::nullopt,
                                    std::nullopt,
                                    2.0f);

    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    const GraphWrapper graphWrapper(serializedGraph.data(), serializedGraph.size());
    const auto& attributes = graphWrapper.getNodeWrapper(0).attributesAs<PointwiseAttributes>();
    auto* dy = static_cast<Tensor<float>*>(&tensorBundle.getTensor(attributes.in_0_tensor_uid()));
    auto* x = static_cast<Tensor<float>*>(
        &tensorBundle.getTensor(attributes.in_1_tensor_uid().value()));
    auto* dx = static_cast<Tensor<float>*>(&tensorBundle.getTensor(attributes.out_0_tensor_uid()));

    constexpr std::array DY_VALUES = {1.0f, 2.0f, 3.0f, 4.0f};
    constexpr std::array X_VALUES = {-1.0f, 0.0f, 1.0f, 2.0f};
    for(int64_t width = 0; width < 4; ++width)
    {
        const auto widthIndex = static_cast<size_t>(width);
        dy->setHostValue(DY_VALUES[widthIndex], 0, 0, 0, width);
        x->setHostValue(X_VALUES[widthIndex], 0, 0, 0, width);
    }
    dy->markHostModified();
    x->markHostModified();

    CpuReferenceGraphExecutor{}.execute(
        serializedGraph.data(), serializedGraph.size(), variantPack);

    const auto gradient = [](double x) { return 1.0 / (1.0 + std::exp(-2.0 * x)); };
    for(int64_t width = 0; width < 4; ++width)
    {
        const auto widthIndex = static_cast<size_t>(width);
        const double expected = static_cast<double>(DY_VALUES[widthIndex])
                                * gradient(static_cast<double>(X_VALUES[widthIndex]));
        EXPECT_NEAR(static_cast<double>(dx->getHostValue(0, 0, 0, width)), expected, 1e-5);
    }
}

TEST_F(TestPointwisePlan, EluFwdWithoutAlphaUsesDefaultAlphaOfOne)
{
    auto [graph, tensorBundle, variantPack]
        = buildPointwiseUnaryGraph({1, 1, 1, 3},
                                   {1, 1, 1, 3},
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   hipdnn_frontend::PointwiseMode::ELU_FWD);

    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    const GraphWrapper graphWrapper(serializedGraph.data(), serializedGraph.size());
    const auto& attributes = graphWrapper.getNodeWrapper(0).attributesAs<PointwiseAttributes>();
    auto* x = static_cast<Tensor<float>*>(&tensorBundle.getTensor(attributes.in_0_tensor_uid()));
    auto* y = static_cast<Tensor<float>*>(&tensorBundle.getTensor(attributes.out_0_tensor_uid()));

    x->setHostValue(-1.0f, 0, 0, 0, 0);
    x->setHostValue(0.0f, 0, 0, 0, 1);
    x->setHostValue(2.0f, 0, 0, 0, 2);
    x->markHostModified();

    CpuReferenceGraphExecutor{}.execute(
        serializedGraph.data(), serializedGraph.size(), variantPack);

    EXPECT_NEAR(y->getHostValue(0, 0, 0, 0), -0.6321205588f, 1e-6f); // exp(-1) - 1
    EXPECT_NEAR(y->getHostValue(0, 0, 0, 1), 0.0f, 1e-6f);
    EXPECT_NEAR(y->getHostValue(0, 0, 0, 2), 2.0f, 1e-6f);
}

TEST_F(TestPointwisePlan, SoftplusBwdWithoutBetaUsesDefaultBetaOfOne)
{
    auto [graph, tensorBundle, variantPack]
        = buildPointwiseBinaryGraph({1, 1, 1, 2},
                                    {1, 1, 1, 2},
                                    {1, 1, 1, 2},
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    hipdnn_frontend::PointwiseMode::SOFTPLUS_BWD);

    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    const GraphWrapper graphWrapper(serializedGraph.data(), serializedGraph.size());
    const auto& attributes = graphWrapper.getNodeWrapper(0).attributesAs<PointwiseAttributes>();
    auto* dy = static_cast<Tensor<float>*>(&tensorBundle.getTensor(attributes.in_0_tensor_uid()));
    auto* x = static_cast<Tensor<float>*>(
        &tensorBundle.getTensor(attributes.in_1_tensor_uid().value()));
    auto* dx = static_cast<Tensor<float>*>(&tensorBundle.getTensor(attributes.out_0_tensor_uid()));

    dy->setHostValue(2.0f, 0, 0, 0, 0);
    dy->setHostValue(2.0f, 0, 0, 0, 1);
    x->setHostValue(1.0f, 0, 0, 0, 0);
    x->setHostValue(0.0f, 0, 0, 0, 1);
    dy->markHostModified();
    x->markHostModified();

    CpuReferenceGraphExecutor{}.execute(
        serializedGraph.data(), serializedGraph.size(), variantPack);

    EXPECT_NEAR(dx->getHostValue(0, 0, 0, 0), 1.4621171573f, 1e-6f); // 2 * sigmoid(1)
    EXPECT_NEAR(dx->getHostValue(0, 0, 0, 1), 1.0f, 1e-6f); // 2 * sigmoid(0)
}

TEST_F(TestPointwisePlan, GenIndexProducesCoordinateAlongAxis)
{
    auto [graph, tensorBundle, variantPack]
        = buildPointwiseUnaryGraph({1, 3, 2, 2},
                                   {1, 3, 2, 2},
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::INT32,
                                   hipdnn_frontend::PointwiseMode::GEN_INDEX,
                                   1,
                                   TensorLayout::NCHW,
                                   std::nullopt,
                                   std::nullopt,
                                   std::nullopt,
                                   std::nullopt,
                                   std::nullopt,
                                   std::nullopt,
                                   int64_t{1});

    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    const GraphWrapper graphWrapper(serializedGraph.data(), serializedGraph.size());
    const auto& attributes = graphWrapper.getNodeWrapper(0).attributesAs<PointwiseAttributes>();
    auto* output
        = static_cast<Tensor<int32_t>*>(&tensorBundle.getTensor(attributes.out_0_tensor_uid()));

    CpuReferenceGraphExecutor{}.execute(
        serializedGraph.data(), serializedGraph.size(), variantPack);

    for(int64_t channel = 0; channel < 3; ++channel)
    {
        for(int64_t height = 0; height < 2; ++height)
        {
            for(int64_t width = 0; width < 2; ++width)
            {
                EXPECT_EQ(output->getHostValue(0, channel, height, width),
                          static_cast<int32_t>(channel));
            }
        }
    }
}

TEST(TestPointwisePlanBuilder, IsApplicableTernary)
{
    auto [graph, tensorBundle, variantPack]
        = buildPointwiseTernaryGraph({1, 2, 1, 1},
                                     {2, 1, 1, 3},
                                     {2, 1, 1, 3},
                                     {2, 2, 1, 3},
                                     DataType::FLOAT,
                                     DataType::FLOAT,
                                     DataType::BOOLEAN,
                                     DataType::FLOAT,
                                     DataType::FLOAT,
                                     hipdnn_frontend::PointwiseMode::BINARY_SELECT);

    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    const GraphWrapper graphWrapper(serializedGraph.data(), serializedGraph.size());
    const auto& attributes = graphWrapper.getNodeWrapper(0).attributesAs<PointwiseAttributes>();

    const PointwisePlanBuilder<DataType::FLOAT,
                               DataType::FLOAT,
                               DataType::BOOLEAN,
                               DataType::FLOAT,
                               DataType::FLOAT>
        planBuilder;
    EXPECT_TRUE(planBuilder.isApplicable(graphWrapper.getNode(0), graphWrapper.getTensorMap()));

    const PointwisePlanBuilder<DataType::FLOAT,
                               DataType::FLOAT,
                               DataType::FLOAT,
                               DataType::FLOAT,
                               DataType::FLOAT>
        wrongMaskTypePlanBuilder;
    EXPECT_FALSE(wrongMaskTypePlanBuilder.isApplicable(graphWrapper.getNode(0),
                                                       graphWrapper.getTensorMap()));

    auto tensorMapWithoutMask = graphWrapper.getTensorMap();
    tensorMapWithoutMask.erase(attributes.in_2_tensor_uid().value());
    EXPECT_FALSE(planBuilder.isApplicable(graphWrapper.getNode(0), tensorMapWithoutMask));
}
TEST_F(TestPointwisePlan, ParametersOnTernaryModeAreIgnored)
{
    const std::vector<std::array<std::optional<float>, 4>> parameters
        = {{std::optional<float>{0.0f}, std::nullopt, std::nullopt, std::nullopt},
           {std::nullopt, std::optional<float>{0.5f}, std::nullopt, std::nullopt},
           {std::nullopt, std::nullopt, std::optional<float>{0.5f}, std::nullopt},
           {std::nullopt, std::nullopt, std::nullopt, std::optional<float>{0.5f}}};

    for(const auto& [reluLowerClip, swishBeta, eluAlpha, softplusBeta] : parameters)
    {
        auto [graph, tensorBundle, variantPack]
            = buildPointwiseTernaryGraph({1, 1, 1, 1},
                                         {1, 1, 1, 1},
                                         {1, 1, 1, 1},
                                         {1, 1, 1, 1},
                                         DataType::FLOAT,
                                         DataType::FLOAT,
                                         DataType::BOOLEAN,
                                         DataType::FLOAT,
                                         DataType::FLOAT,
                                         hipdnn_frontend::PointwiseMode::BINARY_SELECT,
                                         TensorLayout::NCHW,
                                         reluLowerClip,
                                         swishBeta,
                                         eluAlpha,
                                         softplusBeta);

        auto [serializedGraph, serErr] = graph->to_binary();
        ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
        const GraphWrapper graphWrapper(serializedGraph.data(), serializedGraph.size());
        const auto& attributes = graphWrapper.getNodeWrapper(0).attributesAs<PointwiseAttributes>();
        auto* input0
            = static_cast<Tensor<float>*>(&tensorBundle.getTensor(attributes.in_0_tensor_uid()));
        auto* input1 = static_cast<Tensor<float>*>(
            &tensorBundle.getTensor(attributes.in_1_tensor_uid().value()));
        auto* mask = static_cast<Tensor<bool>*>(
            &tensorBundle.getTensor(attributes.in_2_tensor_uid().value()));
        const auto* output
            = static_cast<Tensor<float>*>(&tensorBundle.getTensor(attributes.out_0_tensor_uid()));
        input0->setHostValue(5.0f, 0, 0, 0, 0);
        input1->setHostValue(7.0f, 0, 0, 0, 0);
        mask->setHostValue(true, 0, 0, 0, 0);
        input0->markHostModified();
        input1->markHostModified();
        mask->markHostModified();

        CpuReferenceGraphExecutor{}.execute(
            serializedGraph.data(), serializedGraph.size(), variantPack);

        EXPECT_FLOAT_EQ(output->getHostValue(0, 0, 0, 0), 5.0f);
    }
}

TEST_F(TestPointwisePlan, ParametersOnModesThatDoNotUseThemAreIgnored)
{
    const std::vector<int64_t> dims = {1, 3, 2, 2};

    auto [unaryGraph, unaryBundle, unaryPack]
        = buildPointwiseUnaryGraph(dims,
                                   dims,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   hipdnn_frontend::PointwiseMode::EXP,
                                   1,
                                   TensorLayout::NCHW,
                                   std::nullopt,
                                   std::nullopt,
                                   std::nullopt,
                                   std::nullopt,
                                   2.0f);

    auto [unarySerialized, unaryErr] = unaryGraph->to_binary();
    ASSERT_TRUE(unaryErr.is_good()) << unaryErr.get_message();
    const GraphWrapper unaryWrapper(unarySerialized.data(), unarySerialized.size());
    const auto& unaryAttributes
        = unaryWrapper.getNodeWrapper(0).attributesAs<PointwiseAttributes>();
    const auto* unaryInput
        = static_cast<Tensor<float>*>(&unaryBundle.getTensor(unaryAttributes.in_0_tensor_uid()));
    const auto* unaryOutput
        = static_cast<Tensor<float>*>(&unaryBundle.getTensor(unaryAttributes.out_0_tensor_uid()));

    CpuReferenceGraphExecutor{}.execute(unarySerialized.data(), unarySerialized.size(), unaryPack);

    for(int64_t c = 0; c < 3; ++c)
    {
        EXPECT_NEAR(unaryOutput->getHostValue(0, c, 1, 1),
                    std::exp(unaryInput->getHostValue(0, c, 1, 1)),
                    1e-5f);
    }

    auto [binaryGraph, binaryBundle, binaryPack]
        = buildPointwiseBinaryGraph(dims,
                                    dims,
                                    dims,
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    hipdnn_frontend::PointwiseMode::ADD,
                                    1,
                                    TensorLayout::NCHW,
                                    0.0f);

    auto [binarySerialized, binaryErr] = binaryGraph->to_binary();
    ASSERT_TRUE(binaryErr.is_good()) << binaryErr.get_message();
    const GraphWrapper binaryWrapper(binarySerialized.data(), binarySerialized.size());
    const auto& binaryAttributes
        = binaryWrapper.getNodeWrapper(0).attributesAs<PointwiseAttributes>();
    const auto* binaryInput0
        = static_cast<Tensor<float>*>(&binaryBundle.getTensor(binaryAttributes.in_0_tensor_uid()));
    const auto* binaryInput1 = static_cast<Tensor<float>*>(
        &binaryBundle.getTensor(binaryAttributes.in_1_tensor_uid().value()));
    const auto* binaryOutput
        = static_cast<Tensor<float>*>(&binaryBundle.getTensor(binaryAttributes.out_0_tensor_uid()));

    CpuReferenceGraphExecutor{}.execute(
        binarySerialized.data(), binarySerialized.size(), binaryPack);

    for(int64_t c = 0; c < 3; ++c)
    {
        EXPECT_FLOAT_EQ(binaryOutput->getHostValue(0, c, 1, 1),
                        binaryInput0->getHostValue(0, c, 1, 1)
                            + binaryInput1->getHostValue(0, c, 1, 1));
    }
}

TEST(TestPointwisePlanBuilder, IsApplicableGenIndexAxisOutsideOutputRankIsDeclined)
{
    for(const int64_t axis : {int64_t{4}, int64_t{-1}})
    {
        auto [graph, tensorBundle, variantPack]
            = buildPointwiseUnaryGraph({1, 3, 4, 4},
                                       {1, 3, 4, 4},
                                       DataType::FLOAT,
                                       DataType::FLOAT,
                                       DataType::INT32,
                                       hipdnn_frontend::PointwiseMode::GEN_INDEX,
                                       1,
                                       TensorLayout::NCHW,
                                       std::nullopt,
                                       std::nullopt,
                                       std::nullopt,
                                       std::nullopt,
                                       std::nullopt,
                                       std::nullopt,
                                       axis);

        auto [serializedGraph, serErr] = graph->to_binary();
        ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
        const GraphWrapper graphWrapper(serializedGraph.data(), serializedGraph.size());
        const PointwisePlanBuilder<DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::INT32>
            planBuilder;

        EXPECT_FALSE(planBuilder.isApplicable(graphWrapper.getNode(0), graphWrapper.getTensorMap()))
            << "axis " << axis;
    }
}

TEST(TestPointwisePlanBuilder, IsApplicableGenIndexAcceptsFirstAndLastAxis)
{
    for(const int64_t axis : {int64_t{0}, int64_t{3}})
    {
        auto [graph, tensorBundle, variantPack]
            = buildPointwiseUnaryGraph({2, 3, 2, 4},
                                       {2, 3, 2, 4},
                                       DataType::FLOAT,
                                       DataType::FLOAT,
                                       DataType::INT32,
                                       hipdnn_frontend::PointwiseMode::GEN_INDEX,
                                       1,
                                       TensorLayout::NCHW,
                                       std::nullopt,
                                       std::nullopt,
                                       std::nullopt,
                                       std::nullopt,
                                       std::nullopt,
                                       std::nullopt,
                                       axis);

        auto [serializedGraph, serErr] = graph->to_binary();
        ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
        const GraphWrapper graphWrapper(serializedGraph.data(), serializedGraph.size());
        const PointwisePlanBuilder<DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::INT32>
            planBuilder;

        EXPECT_TRUE(planBuilder.isApplicable(graphWrapper.getNode(0), graphWrapper.getTensorMap()))
            << "axis " << axis;
    }
}

TEST_F(TestPointwisePlan, GenIndexProducesCoordinateAlongFirstAndThirdAxis)
{
    for(const int64_t axis : {int64_t{0}, int64_t{2}})
    {
        auto [graph, tensorBundle, variantPack]
            = buildPointwiseUnaryGraph({2, 3, 2, 4},
                                       {2, 3, 2, 4},
                                       DataType::FLOAT,
                                       DataType::FLOAT,
                                       DataType::INT32,
                                       hipdnn_frontend::PointwiseMode::GEN_INDEX,
                                       1,
                                       TensorLayout::NCHW,
                                       std::nullopt,
                                       std::nullopt,
                                       std::nullopt,
                                       std::nullopt,
                                       std::nullopt,
                                       std::nullopt,
                                       axis);

        auto [serializedGraph, serErr] = graph->to_binary();
        ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
        const GraphWrapper graphWrapper(serializedGraph.data(), serializedGraph.size());
        const auto& attributes = graphWrapper.getNodeWrapper(0).attributesAs<PointwiseAttributes>();
        const auto* output
            = static_cast<Tensor<int32_t>*>(&tensorBundle.getTensor(attributes.out_0_tensor_uid()));

        CpuReferenceGraphExecutor{}.execute(
            serializedGraph.data(), serializedGraph.size(), variantPack);

        for(int64_t n = 0; n < 2; ++n)
        {
            for(int64_t c = 0; c < 3; ++c)
            {
                for(int64_t h = 0; h < 2; ++h)
                {
                    for(int64_t w = 0; w < 4; ++w)
                    {
                        const std::array<int64_t, 4> coordinate = {n, c, h, w};
                        EXPECT_EQ(output->getHostValue(n, c, h, w),
                                  static_cast<int32_t>(coordinate[static_cast<size_t>(axis)]))
                            << "axis " << axis;
                    }
                }
            }
        }
    }
}

namespace
{
// relu_lower_clip, relu_upper_clip, relu_lower_clip_slope, swish_beta, elu_alpha, softplus_beta
using ScalarAttributes = std::array<std::optional<float>, 6>;

ScalarAttributes onlyScalar(size_t index, float value)
{
    ScalarAttributes scalars;
    scalars[index] = value;
    return scalars;
}

float runUnaryOnSingleValue(hipdnn_frontend::PointwiseMode mode,
                            float x,
                            const ScalarAttributes& scalars)
{
    auto [graph, tensorBundle, variantPack] = buildPointwiseUnaryGraph({1, 1, 1, 1},
                                                                       {1, 1, 1, 1},
                                                                       DataType::FLOAT,
                                                                       DataType::FLOAT,
                                                                       DataType::FLOAT,
                                                                       mode,
                                                                       1,
                                                                       TensorLayout::NCHW,
                                                                       scalars[0],
                                                                       scalars[1],
                                                                       scalars[2],
                                                                       scalars[3],
                                                                       scalars[4],
                                                                       scalars[5]);

    auto [serializedGraph, serErr] = graph->to_binary();
    EXPECT_TRUE(serErr.is_good()) << serErr.get_message();
    const GraphWrapper graphWrapper(serializedGraph.data(), serializedGraph.size());
    const auto& attributes = graphWrapper.getNodeWrapper(0).attributesAs<PointwiseAttributes>();
    auto* input
        = static_cast<Tensor<float>*>(&tensorBundle.getTensor(attributes.in_0_tensor_uid()));
    const auto* output
        = static_cast<Tensor<float>*>(&tensorBundle.getTensor(attributes.out_0_tensor_uid()));
    input->setHostValue(x, 0, 0, 0, 0);
    input->markHostModified();

    CpuReferenceGraphExecutor{}.execute(
        serializedGraph.data(), serializedGraph.size(), variantPack);

    return output->getHostValue(0, 0, 0, 0);
}

float runBackwardOnSingleValue(hipdnn_frontend::PointwiseMode mode,
                               float dy,
                               float x,
                               const ScalarAttributes& scalars)
{
    auto [graph, tensorBundle, variantPack] = buildPointwiseBinaryGraph({1, 1, 1, 1},
                                                                        {1, 1, 1, 1},
                                                                        {1, 1, 1, 1},
                                                                        DataType::FLOAT,
                                                                        DataType::FLOAT,
                                                                        DataType::FLOAT,
                                                                        DataType::FLOAT,
                                                                        mode,
                                                                        1,
                                                                        TensorLayout::NCHW,
                                                                        scalars[0],
                                                                        scalars[1],
                                                                        scalars[2],
                                                                        scalars[3],
                                                                        scalars[4],
                                                                        scalars[5]);

    auto [serializedGraph, serErr] = graph->to_binary();
    EXPECT_TRUE(serErr.is_good()) << serErr.get_message();
    const GraphWrapper graphWrapper(serializedGraph.data(), serializedGraph.size());
    const auto& attributes = graphWrapper.getNodeWrapper(0).attributesAs<PointwiseAttributes>();
    auto* dyTensor
        = static_cast<Tensor<float>*>(&tensorBundle.getTensor(attributes.in_0_tensor_uid()));
    auto* xTensor = static_cast<Tensor<float>*>(
        &tensorBundle.getTensor(attributes.in_1_tensor_uid().value()));
    const auto* output
        = static_cast<Tensor<float>*>(&tensorBundle.getTensor(attributes.out_0_tensor_uid()));
    dyTensor->setHostValue(dy, 0, 0, 0, 0);
    xTensor->setHostValue(x, 0, 0, 0, 0);
    dyTensor->markHostModified();
    xTensor->markHostModified();

    CpuReferenceGraphExecutor{}.execute(
        serializedGraph.data(), serializedGraph.size(), variantPack);

    return output->getHostValue(0, 0, 0, 0);
}
} // namespace

TEST_F(TestPointwisePlan, ForeignScalarLeavesTheModeDefaultsInPlace)
{
    using Mode = hipdnn_frontend::PointwiseMode;
    constexpr size_t SWISH_BETA = 3;
    constexpr size_t ELU_ALPHA = 4;
    constexpr size_t SOFTPLUS_BETA = 5;
    constexpr float FOREIGN = 3.0f;

    // Expected values use the default scalar of 1; they were computed independently in double.
    EXPECT_NEAR(runUnaryOnSingleValue(Mode::ELU_FWD, -1.0f, onlyScalar(SWISH_BETA, FOREIGN)),
                -0.6321205588f,
                1e-6f);
    EXPECT_NEAR(runUnaryOnSingleValue(Mode::SOFTPLUS_FWD, 1.0f, onlyScalar(ELU_ALPHA, FOREIGN)),
                1.3132616875f,
                1e-6f);
    EXPECT_NEAR(runUnaryOnSingleValue(Mode::SWISH_FWD, 1.0f, onlyScalar(SOFTPLUS_BETA, FOREIGN)),
                0.7310585786f,
                1e-6f);

    EXPECT_NEAR(
        runBackwardOnSingleValue(Mode::ELU_BWD, 2.0f, -1.0f, onlyScalar(SWISH_BETA, FOREIGN)),
        0.7357588823f,
        1e-6f);
    EXPECT_NEAR(
        runBackwardOnSingleValue(Mode::SOFTPLUS_BWD, 2.0f, 1.0f, onlyScalar(ELU_ALPHA, FOREIGN)),
        1.4621171573f,
        1e-6f);
    EXPECT_NEAR(
        runBackwardOnSingleValue(Mode::SWISH_BWD, 2.0f, 1.0f, onlyScalar(SOFTPLUS_BETA, FOREIGN)),
        1.8553410237f,
        1e-6f);

    // RELU_BWD with its default clips (lower 0, upper max, slope 0)
    EXPECT_FLOAT_EQ(
        runBackwardOnSingleValue(Mode::RELU_BWD, 2.0f, 1.0f, onlyScalar(SWISH_BETA, FOREIGN)),
        2.0f);
    EXPECT_FLOAT_EQ(
        runBackwardOnSingleValue(Mode::RELU_BWD, 2.0f, -1.0f, onlyScalar(SWISH_BETA, FOREIGN)),
        0.0f);
}

TEST_F(TestPointwisePlan, GenIndexIgnoresEveryScalarAttribute)
{
    for(size_t scalarIndex = 0; scalarIndex < 6; ++scalarIndex)
    {
        const auto scalars = onlyScalar(scalarIndex, 0.5f);
        auto [graph, tensorBundle, variantPack]
            = buildPointwiseUnaryGraph({1, 3, 2, 2},
                                       {1, 3, 2, 2},
                                       DataType::FLOAT,
                                       DataType::FLOAT,
                                       DataType::INT32,
                                       hipdnn_frontend::PointwiseMode::GEN_INDEX,
                                       1,
                                       TensorLayout::NCHW,
                                       scalars[0],
                                       scalars[1],
                                       scalars[2],
                                       scalars[3],
                                       scalars[4],
                                       scalars[5],
                                       int64_t{1});

        auto [serializedGraph, serErr] = graph->to_binary();
        ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
        const GraphWrapper graphWrapper(serializedGraph.data(), serializedGraph.size());
        const auto& attributes = graphWrapper.getNodeWrapper(0).attributesAs<PointwiseAttributes>();
        const auto* output
            = static_cast<Tensor<int32_t>*>(&tensorBundle.getTensor(attributes.out_0_tensor_uid()));

        CpuReferenceGraphExecutor{}.execute(
            serializedGraph.data(), serializedGraph.size(), variantPack);

        for(int64_t channel = 0; channel < 3; ++channel)
        {
            for(int64_t height = 0; height < 2; ++height)
            {
                for(int64_t width = 0; width < 2; ++width)
                {
                    EXPECT_EQ(output->getHostValue(0, channel, height, width),
                              static_cast<int32_t>(channel))
                        << "scalar " << scalarIndex;
                }
            }
        }
    }
}

namespace
{

// ADD of a float first operand and a half or bfloat16 second one, one element, through the CPU
// reference with the node computing in `computeType`. The output is float.
template <typename Narrow>
void runAddOfFloatAndNarrow(
    DataType narrowType, DataType computeType, float first, float second, float& result)
{
    auto [graph, tensorBundle, variantPack]
        = buildPointwiseBinaryGraph({1, 1, 1, 1},
                                    {1, 1, 1, 1},
                                    {1, 1, 1, 1},
                                    DataType::FLOAT,
                                    narrowType,
                                    computeType,
                                    DataType::FLOAT,
                                    hipdnn_frontend::PointwiseMode::ADD);

    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    const GraphWrapper graphWrapper(serializedGraph.data(), serializedGraph.size());
    const auto& attributes = graphWrapper.getNodeWrapper(0).attributesAs<PointwiseAttributes>();
    auto* input0
        = static_cast<Tensor<float>*>(&tensorBundle.getTensor(attributes.in_0_tensor_uid()));
    auto* input1 = static_cast<Tensor<Narrow>*>(
        &tensorBundle.getTensor(attributes.in_1_tensor_uid().value()));
    auto* output
        = static_cast<Tensor<float>*>(&tensorBundle.getTensor(attributes.out_0_tensor_uid()));

    input0->setHostValue(first, 0, 0, 0, 0);
    input0->markHostModified();
    input1->setHostValue(static_cast<Narrow>(second), 0, 0, 0, 0);
    input1->markHostModified();

    CpuReferenceGraphExecutor{}.execute(
        serializedGraph.data(), serializedGraph.size(), variantPack);

    output->markDeviceModified();
    result = output->getHostValue(0, 0, 0, 0);
}

// One float element through a unary node that computes in float, stored as float.
void runUnaryOnFloat(hipdnn_frontend::PointwiseMode mode,
                     float value,
                     std::optional<float> swishBeta,
                     float& result)
{
    auto [graph, tensorBundle, variantPack] = buildPointwiseUnaryGraph({1, 1, 1, 1},
                                                                       {1, 1, 1, 1},
                                                                       DataType::FLOAT,
                                                                       DataType::FLOAT,
                                                                       DataType::FLOAT,
                                                                       mode,
                                                                       0,
                                                                       TensorLayout::NCHW,
                                                                       std::nullopt,
                                                                       std::nullopt,
                                                                       std::nullopt,
                                                                       swishBeta);

    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    const GraphWrapper graphWrapper(serializedGraph.data(), serializedGraph.size());
    const auto& attributes = graphWrapper.getNodeWrapper(0).attributesAs<PointwiseAttributes>();
    auto* input
        = static_cast<Tensor<float>*>(&tensorBundle.getTensor(attributes.in_0_tensor_uid()));
    auto* output
        = static_cast<Tensor<float>*>(&tensorBundle.getTensor(attributes.out_0_tensor_uid()));

    input->setHostValue(value, 0, 0, 0, 0);
    input->markHostModified();

    CpuReferenceGraphExecutor{}.execute(
        serializedGraph.data(), serializedGraph.size(), variantPack);

    output->markDeviceModified();
    result = output->getHostValue(0, 0, 0, 0);
}

} // namespace

// A node that computes in bfloat16 converts its operands to bfloat16 before it adds: a float
// that bfloat16 cannot hold (1 + 2^-9) reads as 1, where arithmetic in double would keep it.
TEST(TestPointwisePlanCompute, AddConvertsItsOperandsToTheComputeTypeOfBfloat16)
{
    const float first = 1.0f + std::ldexp(1.0f, -9);

    float rounded = 0.0f;
    runAddOfFloatAndNarrow<bfloat16>(DataType::BFLOAT16, DataType::BFLOAT16, first, 0.0f, rounded);
    EXPECT_EQ(rounded, 1.0f);

    float exact = 0.0f;
    runAddOfFloatAndNarrow<bfloat16>(DataType::BFLOAT16, DataType::FLOAT, first, 0.0f, exact);
    EXPECT_EQ(exact, first);
}

// The same for half, whose spacing at 1 is 2^-10: 1 + 2^-12 reads as 1.
TEST(TestPointwisePlanCompute, AddConvertsItsOperandsToTheComputeTypeOfHalf)
{
    const float first = 1.0f + std::ldexp(1.0f, -12);

    float rounded = 0.0f;
    runAddOfFloatAndNarrow<half>(DataType::HALF, DataType::HALF, first, 0.0f, rounded);
    EXPECT_EQ(rounded, 1.0f);

    float exact = 0.0f;
    runAddOfFloatAndNarrow<half>(DataType::HALF, DataType::FLOAT, first, 0.0f, exact);
    EXPECT_EQ(exact, first);
}

// The sum is rounded in the compute type before it is stored: 1 + 2^-9, both operands exact in
// bfloat16 (and 1 + 2^-12 in half), is one half step above 1 or less, and the output, a float
// that could hold it, gets 1.
TEST(TestPointwisePlanCompute, AddRoundsItsResultToTheComputeType)
{
    float bfloat16Sum = 0.0f;
    runAddOfFloatAndNarrow<bfloat16>(
        DataType::BFLOAT16, DataType::BFLOAT16, 1.0f, std::ldexp(1.0f, -9), bfloat16Sum);
    EXPECT_EQ(bfloat16Sum, 1.0f);

    float halfSum = 0.0f;
    runAddOfFloatAndNarrow<half>(
        DataType::HALF, DataType::HALF, 1.0f, std::ldexp(1.0f, -12), halfSum);
    EXPECT_EQ(halfSum, 1.0f);

    float floatSum = 0.0f;
    runAddOfFloatAndNarrow<half>(
        DataType::HALF, DataType::FLOAT, 1.0f, std::ldexp(1.0f, -12), floatSum);
    EXPECT_EQ(floatSum, 1.0f + std::ldexp(1.0f, -12));
}

// The parameters of a parameterized binary node are converted to the compute type as well. In
// bfloat16 the lower clip 0.1 is 0.10009765625, the x below is that value, so x does not exceed
// the clip, and the gradient is the slope (bfloat16 0.2). In double or float x exceeds 0.1.
TEST(TestPointwisePlanCompute, ParameterizedReluBwdConvertsItsClipsToTheComputeType)
{
    auto [graph, tensorBundle, variantPack]
        = buildPointwiseBinaryGraph({1, 1, 1, 1},
                                    {1, 1, 1, 1},
                                    {1, 1, 1, 1},
                                    DataType::BFLOAT16,
                                    DataType::BFLOAT16,
                                    DataType::BFLOAT16,
                                    DataType::BFLOAT16,
                                    hipdnn_frontend::PointwiseMode::RELU_BWD,
                                    0,
                                    TensorLayout::NCHW,
                                    0.1f,
                                    0.5f,
                                    0.2f);

    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    const GraphWrapper graphWrapper(serializedGraph.data(), serializedGraph.size());
    const auto& attributes = graphWrapper.getNodeWrapper(0).attributesAs<PointwiseAttributes>();
    auto* dyTensor
        = static_cast<Tensor<bfloat16>*>(&tensorBundle.getTensor(attributes.in_0_tensor_uid()));
    auto* xTensor = static_cast<Tensor<bfloat16>*>(
        &tensorBundle.getTensor(attributes.in_1_tensor_uid().value()));
    auto* outputTensor
        = static_cast<Tensor<bfloat16>*>(&tensorBundle.getTensor(attributes.out_0_tensor_uid()));

    dyTensor->setHostValue(static_cast<bfloat16>(1.0f), 0, 0, 0, 0);
    dyTensor->markHostModified();
    xTensor->setHostValue(static_cast<bfloat16>(0.10009765625f), 0, 0, 0, 0);
    xTensor->markHostModified();

    CpuReferenceGraphExecutor{}.execute(
        serializedGraph.data(), serializedGraph.size(), variantPack);

    outputTensor->markDeviceModified();
    EXPECT_EQ(outputTensor->getHostValue(0, 0, 0, 0), static_cast<bfloat16>(0.2f));
}

// A node that computes in float carries out the operation in float, not in double: the first
// float x below for which a float sigmoid and a double one rounded to float differ is graded
// against the float result, for the plain and the parameterized unary paths.
TEST(TestPointwisePlanCompute, UnaryFloatNodeComputesInFloat)
{
    float sigmoidInput = 0.0f;
    float sigmoidExpected = 0.0f;
    bool found = false;
    for(int i = 1; i <= 4000 && !found; ++i)
    {
        const float x = static_cast<float>(i) * 0.01f;
        const float inFloat = 1.0f / (1.0f + std::exp(-x));
        const auto inDouble = static_cast<float>(1.0 / (1.0 + std::exp(-static_cast<double>(x))));
        if(inFloat != inDouble)
        {
            sigmoidInput = x;
            sigmoidExpected = inFloat;
            found = true;
        }
    }
    ASSERT_TRUE(found);

    float sigmoid = 0.0f;
    runUnaryOnFloat(
        hipdnn_frontend::PointwiseMode::SIGMOID_FWD, sigmoidInput, std::nullopt, sigmoid);
    EXPECT_EQ(sigmoid, sigmoidExpected);

    constexpr float BETA = 0.5f;
    float swishInput = 0.0f;
    float swishExpected = 0.0f;
    found = false;
    for(int i = 1; i <= 4000 && !found; ++i)
    {
        const float x = static_cast<float>(i) * 0.01f;
        const float inFloat = x / (1.0f + std::exp(-BETA * x));
        const auto inDouble = static_cast<float>(
            static_cast<double>(x)
            / (1.0 + std::exp(-static_cast<double>(BETA) * static_cast<double>(x))));
        if(inFloat != inDouble)
        {
            swishInput = x;
            swishExpected = inFloat;
            found = true;
        }
    }
    ASSERT_TRUE(found);

    float swish = 0.0f;
    runUnaryOnFloat(hipdnn_frontend::PointwiseMode::SWISH_FWD, swishInput, BETA, swish);
    EXPECT_EQ(swish, swishExpected);
}
