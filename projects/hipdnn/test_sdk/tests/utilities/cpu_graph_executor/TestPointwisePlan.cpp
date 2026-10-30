// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

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

TEST_F(TestPointwisePlan, ExecutePlanGeluApproxTanhBwd)
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
                                    hipdnn_frontend::PointwiseMode::GELU_APPROX_TANH_BWD,
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

TEST_F(TestPointwisePlan, ExecutePlanSwishBwd)
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
                                    hipdnn_frontend::PointwiseMode::SWISH_BWD,
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

TEST_F(TestPointwisePlan, ExecutePlanSwishBwdWithBeta)
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
                                    hipdnn_frontend::PointwiseMode::SWISH_BWD,
                                    seed,
                                    TensorLayout::NCHW,
                                    std::nullopt,
                                    std::nullopt,
                                    std::nullopt,
                                    0.5f,
                                    std::nullopt,
                                    std::nullopt);

    // Execute using CpuReferenceGraphExecutor
    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    CpuReferenceGraphExecutor{}.execute(
        serializedGraph.data(), serializedGraph.size(), variantPack);

    // Verify execution succeeded
    SUCCEED();
}

TEST_F(TestPointwisePlan, ExecutePlanEluBwd)
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
                                    TensorLayout::NCHW);

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
                                    0.5f,
                                    std::nullopt);

    // Execute using CpuReferenceGraphExecutor
    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    CpuReferenceGraphExecutor{}.execute(
        serializedGraph.data(), serializedGraph.size(), variantPack);

    // Verify execution succeeded
    SUCCEED();
}

TEST_F(TestPointwisePlan, ExecutePlanSoftplusBwd)
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

TEST_F(TestPointwisePlan, ExecutePlanBinaryAddSquare)
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
                                    hipdnn_frontend::PointwiseMode::ADD_SQUARE,
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
    CpuReferenceGraphExecutor{}.execute(
        serializedGraph.data(), serializedGraph.size(), variantPack);

    SUCCEED();
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

TEST_F(TestPointwisePlan, ExecutePlanUnaryIdentity)
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
                                   hipdnn_frontend::PointwiseMode::IDENTITY,
                                   seed,
                                   TensorLayout::NCHW);

    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    CpuReferenceGraphExecutor{}.execute(
        serializedGraph.data(), serializedGraph.size(), variantPack);

    SUCCEED();
}

TEST_F(TestPointwisePlan, ExecutePlanUnaryLog)
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
                                   hipdnn_frontend::PointwiseMode::LOG,
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

TEST_F(TestPointwisePlan, ExecutePlanUnaryFloor)
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
                                   hipdnn_frontend::PointwiseMode::FLOOR,
                                   seed,
                                   TensorLayout::NCHW);

    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    CpuReferenceGraphExecutor{}.execute(
        serializedGraph.data(), serializedGraph.size(), variantPack);

    SUCCEED();
}

TEST_F(TestPointwisePlan, ExecutePlanUnaryEluFwd)
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

TEST_F(TestPointwisePlan, ExecutePlanUnarySoftplusFwd)
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
                                   TensorLayout::NCHW);

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
        = dynamic_cast<PointwisePlan<float, float, float, float>*>(builtPlan.get()) != nullptr;
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
        = dynamic_cast<PointwisePlan<float, float, float, float>*>(builtPlan.get()) != nullptr;
    EXPECT_TRUE(result);
}

TEST(TestPointwisePlanBuilder, PlanConstructionGeluBwd)
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
                                    hipdnn_frontend::PointwiseMode::GELU_BWD,
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

TEST(TestPointwisePlanBuilder, PlanConstructionGeluApproxTanhBwd)
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
                                    hipdnn_frontend::PointwiseMode::GELU_APPROX_TANH_BWD,
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

TEST(TestPointwisePlanBuilder, PlanConstructionSwishBwd)
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
                                    hipdnn_frontend::PointwiseMode::SWISH_BWD,
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

TEST(TestPointwisePlanBuilder, PlanConstructionEluBwd)
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
                                    hipdnn_frontend::PointwiseMode::ELU_BWD,
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

TEST(TestPointwisePlanBuilder, PlanConstructionSoftplusBwd)
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
                                    hipdnn_frontend::PointwiseMode::SOFTPLUS_BWD,
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

TEST(TestPointwisePlanBuilder, PlanConstructionBinaryAddSquare)
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
                                    hipdnn_frontend::PointwiseMode::ADD_SQUARE,
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
        = dynamic_cast<PointwisePlan<float, float, float, float>*>(builtPlan.get()) != nullptr;
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
        = dynamic_cast<PointwisePlan<float, float, float, float>*>(builtPlan.get()) != nullptr;
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
        = dynamic_cast<PointwisePlan<float, float, float, float>*>(builtPlan.get()) != nullptr;
    EXPECT_TRUE(result);
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

TEST(TestPointwisePlanBuilder, PlanConstructionUnaryIdentity)
{
    const std::vector<int64_t> inputDims = {1, 3, 4, 4};
    const std::vector<int64_t> outputDims = {1, 3, 4, 4};

    auto [graph, tensorBundle, variantPack]
        = buildPointwiseUnaryGraph(inputDims,
                                   outputDims,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   hipdnn_frontend::PointwiseMode::IDENTITY,
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

TEST(TestPointwisePlanBuilder, PlanConstructionUnaryLog)
{
    const std::vector<int64_t> inputDims = {1, 3, 4, 4};
    const std::vector<int64_t> outputDims = {1, 3, 4, 4};

    auto [graph, tensorBundle, variantPack]
        = buildPointwiseUnaryGraph(inputDims,
                                   outputDims,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   hipdnn_frontend::PointwiseMode::LOG,
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

TEST(TestPointwisePlanBuilder, PlanConstructionUnarySqrt)
{
    const std::vector<int64_t> inputDims = {1, 3, 4, 4};
    const std::vector<int64_t> outputDims = {1, 3, 4, 4};

    auto [graph, tensorBundle, variantPack]
        = buildPointwiseUnaryGraph(inputDims,
                                   outputDims,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   hipdnn_frontend::PointwiseMode::SQRT,
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

TEST(TestPointwisePlanBuilder, PlanConstructionUnaryRsqrt)
{
    const std::vector<int64_t> inputDims = {1, 3, 4, 4};
    const std::vector<int64_t> outputDims = {1, 3, 4, 4};

    auto [graph, tensorBundle, variantPack]
        = buildPointwiseUnaryGraph(inputDims,
                                   outputDims,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   hipdnn_frontend::PointwiseMode::RSQRT,
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

TEST(TestPointwisePlanBuilder, PlanConstructionUnaryReciprocal)
{
    const std::vector<int64_t> inputDims = {1, 3, 4, 4};
    const std::vector<int64_t> outputDims = {1, 3, 4, 4};

    auto [graph, tensorBundle, variantPack]
        = buildPointwiseUnaryGraph(inputDims,
                                   outputDims,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   hipdnn_frontend::PointwiseMode::RECIPROCAL,
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

TEST(TestPointwisePlanBuilder, PlanConstructionUnarySin)
{
    const std::vector<int64_t> inputDims = {1, 3, 4, 4};
    const std::vector<int64_t> outputDims = {1, 3, 4, 4};

    auto [graph, tensorBundle, variantPack]
        = buildPointwiseUnaryGraph(inputDims,
                                   outputDims,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   hipdnn_frontend::PointwiseMode::SIN,
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

TEST(TestPointwisePlanBuilder, PlanConstructionUnaryTan)
{
    const std::vector<int64_t> inputDims = {1, 3, 4, 4};
    const std::vector<int64_t> outputDims = {1, 3, 4, 4};

    auto [graph, tensorBundle, variantPack]
        = buildPointwiseUnaryGraph(inputDims,
                                   outputDims,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   hipdnn_frontend::PointwiseMode::TAN,
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

TEST(TestPointwisePlanBuilder, PlanConstructionUnaryErf)
{
    const std::vector<int64_t> inputDims = {1, 3, 4, 4};
    const std::vector<int64_t> outputDims = {1, 3, 4, 4};

    auto [graph, tensorBundle, variantPack]
        = buildPointwiseUnaryGraph(inputDims,
                                   outputDims,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   hipdnn_frontend::PointwiseMode::ERF,
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

TEST(TestPointwisePlanBuilder, PlanConstructionUnaryCeil)
{
    const std::vector<int64_t> inputDims = {1, 3, 4, 4};
    const std::vector<int64_t> outputDims = {1, 3, 4, 4};

    auto [graph, tensorBundle, variantPack]
        = buildPointwiseUnaryGraph(inputDims,
                                   outputDims,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   hipdnn_frontend::PointwiseMode::CEIL,
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

TEST(TestPointwisePlanBuilder, PlanConstructionUnaryFloor)
{
    const std::vector<int64_t> inputDims = {1, 3, 4, 4};
    const std::vector<int64_t> outputDims = {1, 3, 4, 4};

    auto [graph, tensorBundle, variantPack]
        = buildPointwiseUnaryGraph(inputDims,
                                   outputDims,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   hipdnn_frontend::PointwiseMode::FLOOR,
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

TEST(TestPointwisePlanBuilder, PlanConstructionUnaryElu)
{
    const std::vector<int64_t> inputDims = {1, 3, 4, 4};
    const std::vector<int64_t> outputDims = {1, 3, 4, 4};

    auto [graph, tensorBundle, variantPack]
        = buildPointwiseUnaryGraph(inputDims,
                                   outputDims,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   hipdnn_frontend::PointwiseMode::ELU_FWD,
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

TEST(TestPointwisePlanBuilder, PlanConstructionUnarySoftplus)
{
    const std::vector<int64_t> inputDims = {1, 3, 4, 4};
    const std::vector<int64_t> outputDims = {1, 3, 4, 4};

    auto [graph, tensorBundle, variantPack]
        = buildPointwiseUnaryGraph(inputDims,
                                   outputDims,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   hipdnn_frontend::PointwiseMode::SOFTPLUS_FWD,
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

TEST(TestPointwisePlanBuilder, IsApplicableGeluBwd)
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
                                    hipdnn_frontend::PointwiseMode::GELU_BWD,
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

TEST(TestPointwisePlanBuilder, IsApplicableGeluApproxTanhBwd)
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
                                    hipdnn_frontend::PointwiseMode::GELU_APPROX_TANH_BWD,
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

TEST(TestPointwisePlanBuilder, IsApplicableSwishBwd)
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
                                    hipdnn_frontend::PointwiseMode::SWISH_BWD,
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

TEST(TestPointwisePlanBuilder, IsApplicableEluBwd)
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
                                    hipdnn_frontend::PointwiseMode::ELU_BWD,
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

TEST(TestPointwisePlanBuilder, IsApplicableSoftplusBwd)
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
                                    hipdnn_frontend::PointwiseMode::SOFTPLUS_BWD,
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

TEST(TestPointwisePlanBuilder, IsApplicableBinaryDiv)
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
        floatPlanBuilder;
    EXPECT_TRUE(floatPlanBuilder.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap()));

    // Test with missing tensor - erase a tensor from the map
    auto tensorMapCopy = graphWrap.getTensorMap();
    tensorMapCopy.erase(2);
    EXPECT_FALSE(floatPlanBuilder.isApplicable(graphWrap.getNode(0), tensorMapCopy));
}

TEST(TestPointwisePlanBuilder, IsApplicableBinaryAddSquare)
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
                                    hipdnn_frontend::PointwiseMode::ADD_SQUARE,
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

TEST(TestPointwisePlanBuilder, IsApplicableUnaryIdentity)
{
    const std::vector<int64_t> inputDims = {1, 3, 4, 4};
    const std::vector<int64_t> outputDims = {1, 3, 4, 4};

    auto [graph, tensorBundle, variantPack]
        = buildPointwiseUnaryGraph(inputDims,
                                   outputDims,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   hipdnn_frontend::PointwiseMode::IDENTITY,
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

TEST(TestPointwisePlanBuilder, IsApplicableUnaryLog)
{
    const std::vector<int64_t> inputDims = {1, 3, 4, 4};
    const std::vector<int64_t> outputDims = {1, 3, 4, 4};

    auto [graph, tensorBundle, variantPack]
        = buildPointwiseUnaryGraph(inputDims,
                                   outputDims,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   hipdnn_frontend::PointwiseMode::LOG,
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

TEST(TestPointwisePlanBuilder, IsApplicableUnarySqrt)
{
    const std::vector<int64_t> inputDims = {1, 3, 4, 4};
    const std::vector<int64_t> outputDims = {1, 3, 4, 4};

    auto [graph, tensorBundle, variantPack]
        = buildPointwiseUnaryGraph(inputDims,
                                   outputDims,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   hipdnn_frontend::PointwiseMode::SQRT,
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

TEST(TestPointwisePlanBuilder, IsApplicableUnaryRsqrt)
{
    const std::vector<int64_t> inputDims = {1, 3, 4, 4};
    const std::vector<int64_t> outputDims = {1, 3, 4, 4};

    auto [graph, tensorBundle, variantPack]
        = buildPointwiseUnaryGraph(inputDims,
                                   outputDims,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   hipdnn_frontend::PointwiseMode::RSQRT,
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

TEST(TestPointwisePlanBuilder, IsApplicableUnaryReciprocal)
{
    const std::vector<int64_t> inputDims = {1, 3, 4, 4};
    const std::vector<int64_t> outputDims = {1, 3, 4, 4};

    auto [graph, tensorBundle, variantPack]
        = buildPointwiseUnaryGraph(inputDims,
                                   outputDims,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   hipdnn_frontend::PointwiseMode::RECIPROCAL,
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

TEST(TestPointwisePlanBuilder, IsApplicableUnarySin)
{
    const std::vector<int64_t> inputDims = {1, 3, 4, 4};
    const std::vector<int64_t> outputDims = {1, 3, 4, 4};

    auto [graph, tensorBundle, variantPack]
        = buildPointwiseUnaryGraph(inputDims,
                                   outputDims,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   hipdnn_frontend::PointwiseMode::SIN,
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

TEST(TestPointwisePlanBuilder, IsApplicableUnaryTan)
{
    const std::vector<int64_t> inputDims = {1, 3, 4, 4};
    const std::vector<int64_t> outputDims = {1, 3, 4, 4};

    auto [graph, tensorBundle, variantPack]
        = buildPointwiseUnaryGraph(inputDims,
                                   outputDims,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   hipdnn_frontend::PointwiseMode::TAN,
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

TEST(TestPointwisePlanBuilder, IsApplicableUnaryErf)
{
    const std::vector<int64_t> inputDims = {1, 3, 4, 4};
    const std::vector<int64_t> outputDims = {1, 3, 4, 4};

    auto [graph, tensorBundle, variantPack]
        = buildPointwiseUnaryGraph(inputDims,
                                   outputDims,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   hipdnn_frontend::PointwiseMode::ERF,
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

TEST(TestPointwisePlanBuilder, IsApplicableUnaryCeil)
{
    const std::vector<int64_t> inputDims = {1, 3, 4, 4};
    const std::vector<int64_t> outputDims = {1, 3, 4, 4};

    auto [graph, tensorBundle, variantPack]
        = buildPointwiseUnaryGraph(inputDims,
                                   outputDims,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   hipdnn_frontend::PointwiseMode::CEIL,
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

TEST(TestPointwisePlanBuilder, IsApplicableUnaryFloor)
{
    const std::vector<int64_t> inputDims = {1, 3, 4, 4};
    const std::vector<int64_t> outputDims = {1, 3, 4, 4};

    auto [graph, tensorBundle, variantPack]
        = buildPointwiseUnaryGraph(inputDims,
                                   outputDims,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   hipdnn_frontend::PointwiseMode::FLOOR,
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

TEST(TestPointwisePlanBuilder, IsApplicableUnaryElu)
{
    const std::vector<int64_t> inputDims = {1, 3, 4, 4};
    const std::vector<int64_t> outputDims = {1, 3, 4, 4};

    auto [graph, tensorBundle, variantPack]
        = buildPointwiseUnaryGraph(inputDims,
                                   outputDims,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   hipdnn_frontend::PointwiseMode::ELU_FWD,
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

TEST(TestPointwisePlanBuilder, IsApplicableUnarySoftplus)
{
    const std::vector<int64_t> inputDims = {1, 3, 4, 4};
    const std::vector<int64_t> outputDims = {1, 3, 4, 4};

    auto [graph, tensorBundle, variantPack]
        = buildPointwiseUnaryGraph(inputDims,
                                   outputDims,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   hipdnn_frontend::PointwiseMode::SOFTPLUS_FWD,
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

TEST(TestPointwisePlanBuilder, UnsupportedOperation)
{
    // Every classified pointwise mode is implemented, so use an unclassified (UNSET) operation.
    flatbuffers::FlatBufferBuilder fbb;
    auto pointwiseAttributes = CreatePointwiseAttributes(fbb,
                                                         PointwiseMode::UNSET,
                                                         flatbuffers::nullopt,
                                                         flatbuffers::nullopt,
                                                         flatbuffers::nullopt,
                                                         flatbuffers::nullopt,
                                                         1,
                                                         flatbuffers::nullopt,
                                                         flatbuffers::nullopt,
                                                         2);
    auto node = CreateNode(
        fbb, 0, DataType::FLOAT, NodeAttributes::PointwiseAttributes, pointwiseAttributes.Union());
    fbb.Finish(node);
    const auto& unsupportedNode = *flatbuffers::GetRoot<Node>(fbb.GetBufferPointer());

    const PointwisePlanBuilder<DataType::FLOAT,
                               DataType::FLOAT,
                               DataType::FLOAT,
                               DataType::FLOAT,
                               DataType::FLOAT>
        planBuilder;
    EXPECT_FALSE(planBuilder.isApplicable(unsupportedNode, {}));
}

TEST(TestPointwisePlanBuilder, PlanBuilderThrowsIfEluAlphaValueSet)
{
    const std::vector<int64_t> inputDims = {1, 3, 4, 4};
    const std::vector<int64_t> outputDims = {1, 3, 4, 4};

    auto [graph, tensorBundle, variantPack]
        = buildPointwiseUnaryGraph(inputDims,
                                   outputDims,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   hipdnn_frontend::PointwiseMode::RELU_FWD, // support op
                                   1,
                                   TensorLayout::NCHW,
                                   std::nullopt,
                                   std::nullopt,
                                   std::nullopt,
                                   std::nullopt,
                                   1.0f);

    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        serializedGraph.data(), serializedGraph.size());

    const PointwisePlanBuilder<DataType::FLOAT,
                               DataType::FLOAT,
                               DataType::FLOAT,
                               DataType::FLOAT,
                               DataType::FLOAT>
        planBuilder;
    EXPECT_THROW(planBuilder.buildNodePlan(graphWrap, graphWrap.getNode(0)), std::runtime_error);
}

TEST(TestPointwisePlanBuilder, PlanBuilderThrowsIfSoftPlusBetaValueSet)
{
    const std::vector<int64_t> inputDims = {1, 3, 4, 4};
    const std::vector<int64_t> outputDims = {1, 3, 4, 4};

    auto [graph, tensorBundle, variantPack]
        = buildPointwiseUnaryGraph(inputDims,
                                   outputDims,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   hipdnn_frontend::PointwiseMode::RELU_FWD, // support op
                                   1,
                                   TensorLayout::NCHW,
                                   std::nullopt,
                                   std::nullopt,
                                   std::nullopt,
                                   std::nullopt,
                                   std::nullopt,

                                   1.0f);

    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        serializedGraph.data(), serializedGraph.size());

    const PointwisePlanBuilder<DataType::FLOAT,
                               DataType::FLOAT,
                               DataType::FLOAT,
                               DataType::FLOAT,
                               DataType::FLOAT>
        planBuilder;
    EXPECT_THROW(planBuilder.buildNodePlan(graphWrap, graphWrap.getNode(0)), std::runtime_error);
}

TEST(TestPointwisePlanBuilder, PlanBuilderDoesNotThrowIfEluAlphaValueSetForEluBwd)
{
    const std::vector<int64_t> inputDims = {1, 3, 4, 4};
    const std::vector<int64_t> outputDims = {1, 3, 4, 4};

    auto [graph, tensorBundle, variantPack]
        = buildPointwiseBinaryGraph(inputDims,
                                    inputDims,
                                    outputDims,
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    hipdnn_frontend::PointwiseMode::ELU_BWD,
                                    1,
                                    TensorLayout::NCHW,
                                    std::nullopt,
                                    std::nullopt,
                                    std::nullopt,
                                    std::nullopt,
                                    1.0f,
                                    std::nullopt);

    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        serializedGraph.data(), serializedGraph.size());

    const PointwisePlanBuilder<DataType::FLOAT,
                               DataType::FLOAT,
                               DataType::FLOAT,
                               DataType::FLOAT,
                               DataType::FLOAT>
        planBuilder;
    EXPECT_NO_THROW(planBuilder.buildNodePlan(graphWrap, graphWrap.getNode(0)));
}

TEST(TestPointwisePlanBuilder, PlanBuilderDoesNotThrowIfSoftPlusBetaValueSetForSoftplusBwd)
{
    const std::vector<int64_t> inputDims = {1, 3, 4, 4};
    const std::vector<int64_t> outputDims = {1, 3, 4, 4};

    auto [graph, tensorBundle, variantPack]
        = buildPointwiseBinaryGraph(inputDims,
                                    inputDims,
                                    outputDims,
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
                                    1.0f);

    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        serializedGraph.data(), serializedGraph.size());

    const PointwisePlanBuilder<DataType::FLOAT,
                               DataType::FLOAT,
                               DataType::FLOAT,
                               DataType::FLOAT,
                               DataType::FLOAT>
        planBuilder;
    EXPECT_NO_THROW(planBuilder.buildNodePlan(graphWrap, graphWrap.getNode(0)));
}

TEST(TestPointwisePlanBuilder, PlanBuilderThrowsIfEluAlphaValueSetForGeluBwd)
{
    const std::vector<int64_t> inputDims = {1, 3, 4, 4};
    const std::vector<int64_t> outputDims = {1, 3, 4, 4};

    auto [graph, tensorBundle, variantPack]
        = buildPointwiseBinaryGraph(inputDims,
                                    inputDims,
                                    outputDims,
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    hipdnn_frontend::PointwiseMode::GELU_BWD,
                                    1,
                                    TensorLayout::NCHW,
                                    std::nullopt,
                                    std::nullopt,
                                    std::nullopt,
                                    std::nullopt,
                                    1.0f,
                                    std::nullopt);

    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        serializedGraph.data(), serializedGraph.size());

    const PointwisePlanBuilder<DataType::FLOAT,
                               DataType::FLOAT,
                               DataType::FLOAT,
                               DataType::FLOAT,
                               DataType::FLOAT>
        planBuilder;
    EXPECT_THROW(planBuilder.buildNodePlan(graphWrap, graphWrap.getNode(0)), std::runtime_error);
}

TEST(TestPointwisePlanBuilder, PlanBuilderThrowsIfSoftPlusBetaValueSetForSwishBwd)
{
    const std::vector<int64_t> inputDims = {1, 3, 4, 4};
    const std::vector<int64_t> outputDims = {1, 3, 4, 4};

    auto [graph, tensorBundle, variantPack]
        = buildPointwiseBinaryGraph(inputDims,
                                    inputDims,
                                    outputDims,
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    hipdnn_frontend::PointwiseMode::SWISH_BWD,
                                    1,
                                    TensorLayout::NCHW,
                                    std::nullopt,
                                    std::nullopt,
                                    std::nullopt,
                                    std::nullopt,
                                    std::nullopt,
                                    1.0f);

    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        serializedGraph.data(), serializedGraph.size());

    const PointwisePlanBuilder<DataType::FLOAT,
                               DataType::FLOAT,
                               DataType::FLOAT,
                               DataType::FLOAT,
                               DataType::FLOAT>
        planBuilder;
    EXPECT_THROW(planBuilder.buildNodePlan(graphWrap, graphWrap.getNode(0)), std::runtime_error);
}

TEST(TestPointwisePlanBuilder, PlanBuilderDoesNotThrowIfEluAlphaValueSetForEluFwd)
{
    const std::vector<int64_t> inputDims = {1, 3, 4, 4};
    const std::vector<int64_t> outputDims = {1, 3, 4, 4};

    auto [graph, tensorBundle, variantPack]
        = buildPointwiseUnaryGraph(inputDims,
                                   outputDims,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   hipdnn_frontend::PointwiseMode::ELU_FWD,
                                   1,
                                   TensorLayout::NCHW,
                                   std::nullopt,
                                   std::nullopt,
                                   std::nullopt,
                                   std::nullopt,
                                   1.0f,
                                   std::nullopt);

    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        serializedGraph.data(), serializedGraph.size());

    const PointwisePlanBuilder<DataType::FLOAT,
                               DataType::FLOAT,
                               DataType::FLOAT,
                               DataType::FLOAT,
                               DataType::FLOAT>
        planBuilder;
    EXPECT_NO_THROW(planBuilder.buildNodePlan(graphWrap, graphWrap.getNode(0)));
}

TEST(TestPointwisePlanBuilder, PlanBuilderDoesNotThrowIfSoftPlusBetaValueSetForSoftplusFwd)
{
    const std::vector<int64_t> inputDims = {1, 3, 4, 4};
    const std::vector<int64_t> outputDims = {1, 3, 4, 4};

    auto [graph, tensorBundle, variantPack]
        = buildPointwiseUnaryGraph(inputDims,
                                   outputDims,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   hipdnn_frontend::PointwiseMode::SOFTPLUS_FWD,
                                   1,
                                   TensorLayout::NCHW,
                                   std::nullopt,
                                   std::nullopt,
                                   std::nullopt,
                                   std::nullopt,
                                   std::nullopt,
                                   1.0f);

    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        serializedGraph.data(), serializedGraph.size());

    const PointwisePlanBuilder<DataType::FLOAT,
                               DataType::FLOAT,
                               DataType::FLOAT,
                               DataType::FLOAT,
                               DataType::FLOAT>
        planBuilder;
    EXPECT_NO_THROW(planBuilder.buildNodePlan(graphWrap, graphWrap.getNode(0)));
}

TEST(TestPointwisePlanBuilder, PlanBuilderThrowsIfEluAlphaValueSetForSoftplusFwd)
{
    const std::vector<int64_t> inputDims = {1, 3, 4, 4};
    const std::vector<int64_t> outputDims = {1, 3, 4, 4};

    auto [graph, tensorBundle, variantPack]
        = buildPointwiseUnaryGraph(inputDims,
                                   outputDims,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   hipdnn_frontend::PointwiseMode::SOFTPLUS_FWD,
                                   1,
                                   TensorLayout::NCHW,
                                   std::nullopt,
                                   std::nullopt,
                                   std::nullopt,
                                   std::nullopt,
                                   1.0f,
                                   std::nullopt);

    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        serializedGraph.data(), serializedGraph.size());

    const PointwisePlanBuilder<DataType::FLOAT,
                               DataType::FLOAT,
                               DataType::FLOAT,
                               DataType::FLOAT,
                               DataType::FLOAT>
        planBuilder;
    EXPECT_THROW(planBuilder.buildNodePlan(graphWrap, graphWrap.getNode(0)), std::runtime_error);
}

TEST(TestPointwisePlanBuilder, PlanBuilderThrowsIfSoftPlusBetaValueSetForEluFwd)
{
    const std::vector<int64_t> inputDims = {1, 3, 4, 4};
    const std::vector<int64_t> outputDims = {1, 3, 4, 4};

    auto [graph, tensorBundle, variantPack]
        = buildPointwiseUnaryGraph(inputDims,
                                   outputDims,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   hipdnn_frontend::PointwiseMode::ELU_FWD,
                                   1,
                                   TensorLayout::NCHW,
                                   std::nullopt,
                                   std::nullopt,
                                   std::nullopt,
                                   std::nullopt,
                                   std::nullopt,
                                   1.0f);

    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        serializedGraph.data(), serializedGraph.size());

    const PointwisePlanBuilder<DataType::FLOAT,
                               DataType::FLOAT,
                               DataType::FLOAT,
                               DataType::FLOAT,
                               DataType::FLOAT>
        planBuilder;
    EXPECT_THROW(planBuilder.buildNodePlan(graphWrap, graphWrap.getNode(0)), std::runtime_error);
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

TEST_F(TestPointwisePlan, CompareGreaterProducesBooleanMaskHalf)
{
    auto [graph, tensorBundle, variantPack]
        = buildPointwiseBinaryGraph({1, 1, 1, 4},
                                    {1, 1, 1, 4},
                                    {1, 1, 1, 4},
                                    DataType::HALF,
                                    DataType::HALF,
                                    DataType::FLOAT,
                                    DataType::BOOLEAN,
                                    hipdnn_frontend::PointwiseMode::CMP_GT);

    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    const GraphWrapper graphWrapper(serializedGraph.data(), serializedGraph.size());
    const auto& attributes = graphWrapper.getNodeWrapper(0).attributesAs<PointwiseAttributes>();
    auto* input0
        = static_cast<Tensor<half>*>(&tensorBundle.getTensor(attributes.in_0_tensor_uid()));
    auto* input1
        = static_cast<Tensor<half>*>(&tensorBundle.getTensor(attributes.in_1_tensor_uid().value()));
    auto* output
        = static_cast<Tensor<bool>*>(&tensorBundle.getTensor(attributes.out_0_tensor_uid()));

    constexpr std::array INPUT0_VALUES = {-2.0f, 0.0f, 3.0f, 4.0f};
    constexpr std::array INPUT1_VALUES = {-3.0f, 0.0f, 5.0f, 1.0f};
    constexpr std::array EXPECTED = {true, false, false, true};
    for(int64_t width = 0; width < 4; ++width)
    {
        const auto widthIndex = static_cast<size_t>(width);
        input0->setHostValue(half(INPUT0_VALUES[widthIndex]), 0, 0, 0, width);
        input1->setHostValue(half(INPUT1_VALUES[widthIndex]), 0, 0, 0, width);
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

TEST_F(TestPointwisePlan, CompareGreaterProducesBooleanMaskBfloat16)
{
    auto [graph, tensorBundle, variantPack]
        = buildPointwiseBinaryGraph({1, 1, 1, 4},
                                    {1, 1, 1, 4},
                                    {1, 1, 1, 4},
                                    DataType::BFLOAT16,
                                    DataType::BFLOAT16,
                                    DataType::FLOAT,
                                    DataType::BOOLEAN,
                                    hipdnn_frontend::PointwiseMode::CMP_GT);

    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    const GraphWrapper graphWrapper(serializedGraph.data(), serializedGraph.size());
    const auto& attributes = graphWrapper.getNodeWrapper(0).attributesAs<PointwiseAttributes>();
    auto* input0
        = static_cast<Tensor<bfloat16>*>(&tensorBundle.getTensor(attributes.in_0_tensor_uid()));
    auto* input1 = static_cast<Tensor<bfloat16>*>(
        &tensorBundle.getTensor(attributes.in_1_tensor_uid().value()));
    auto* output
        = static_cast<Tensor<bool>*>(&tensorBundle.getTensor(attributes.out_0_tensor_uid()));

    constexpr std::array INPUT0_VALUES = {-2.0f, 0.0f, 3.0f, 4.0f};
    constexpr std::array INPUT1_VALUES = {-3.0f, 0.0f, 5.0f, 1.0f};
    constexpr std::array EXPECTED = {true, false, false, true};
    for(int64_t width = 0; width < 4; ++width)
    {
        const auto widthIndex = static_cast<size_t>(width);
        input0->setHostValue(bfloat16(INPUT0_VALUES[widthIndex]), 0, 0, 0, width);
        input1->setHostValue(bfloat16(INPUT1_VALUES[widthIndex]), 0, 0, 0, width);
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

TEST_F(TestPointwisePlan, CompareEqualProducesBooleanMaskHalf)
{
    auto [graph, tensorBundle, variantPack]
        = buildPointwiseBinaryGraph({1, 1, 1, 4},
                                    {1, 1, 1, 4},
                                    {1, 1, 1, 4},
                                    DataType::HALF,
                                    DataType::HALF,
                                    DataType::FLOAT,
                                    DataType::BOOLEAN,
                                    hipdnn_frontend::PointwiseMode::CMP_EQ);

    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    const GraphWrapper graphWrapper(serializedGraph.data(), serializedGraph.size());
    const auto& attributes = graphWrapper.getNodeWrapper(0).attributesAs<PointwiseAttributes>();
    auto* input0
        = static_cast<Tensor<half>*>(&tensorBundle.getTensor(attributes.in_0_tensor_uid()));
    auto* input1
        = static_cast<Tensor<half>*>(&tensorBundle.getTensor(attributes.in_1_tensor_uid().value()));
    auto* output
        = static_cast<Tensor<bool>*>(&tensorBundle.getTensor(attributes.out_0_tensor_uid()));

    constexpr std::array INPUT0_VALUES = {-2.0f, 0.0f, 3.0f, 4.0f};
    constexpr std::array INPUT1_VALUES = {-3.0f, 0.0f, 5.0f, 1.0f};
    constexpr std::array EXPECTED = {false, true, false, false};
    for(int64_t width = 0; width < 4; ++width)
    {
        const auto widthIndex = static_cast<size_t>(width);
        input0->setHostValue(half(INPUT0_VALUES[widthIndex]), 0, 0, 0, width);
        input1->setHostValue(half(INPUT1_VALUES[widthIndex]), 0, 0, 0, width);
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

TEST_F(TestPointwisePlan, CompareEqualProducesBooleanMaskBfloat16)
{
    auto [graph, tensorBundle, variantPack]
        = buildPointwiseBinaryGraph({1, 1, 1, 4},
                                    {1, 1, 1, 4},
                                    {1, 1, 1, 4},
                                    DataType::BFLOAT16,
                                    DataType::BFLOAT16,
                                    DataType::FLOAT,
                                    DataType::BOOLEAN,
                                    hipdnn_frontend::PointwiseMode::CMP_EQ);

    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    const GraphWrapper graphWrapper(serializedGraph.data(), serializedGraph.size());
    const auto& attributes = graphWrapper.getNodeWrapper(0).attributesAs<PointwiseAttributes>();
    auto* input0
        = static_cast<Tensor<bfloat16>*>(&tensorBundle.getTensor(attributes.in_0_tensor_uid()));
    auto* input1 = static_cast<Tensor<bfloat16>*>(
        &tensorBundle.getTensor(attributes.in_1_tensor_uid().value()));
    auto* output
        = static_cast<Tensor<bool>*>(&tensorBundle.getTensor(attributes.out_0_tensor_uid()));

    constexpr std::array INPUT0_VALUES = {-2.0f, 0.0f, 3.0f, 4.0f};
    constexpr std::array INPUT1_VALUES = {-3.0f, 0.0f, 5.0f, 1.0f};
    constexpr std::array EXPECTED = {false, true, false, false};
    for(int64_t width = 0; width < 4; ++width)
    {
        const auto widthIndex = static_cast<size_t>(width);
        input0->setHostValue(bfloat16(INPUT0_VALUES[widthIndex]), 0, 0, 0, width);
        input1->setHostValue(bfloat16(INPUT1_VALUES[widthIndex]), 0, 0, 0, width);
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

TEST_F(TestPointwisePlan, CompareNotEqualProducesBooleanMaskHalf)
{
    auto [graph, tensorBundle, variantPack]
        = buildPointwiseBinaryGraph({1, 1, 1, 4},
                                    {1, 1, 1, 4},
                                    {1, 1, 1, 4},
                                    DataType::HALF,
                                    DataType::HALF,
                                    DataType::FLOAT,
                                    DataType::BOOLEAN,
                                    hipdnn_frontend::PointwiseMode::CMP_NEQ);

    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    const GraphWrapper graphWrapper(serializedGraph.data(), serializedGraph.size());
    const auto& attributes = graphWrapper.getNodeWrapper(0).attributesAs<PointwiseAttributes>();
    auto* input0
        = static_cast<Tensor<half>*>(&tensorBundle.getTensor(attributes.in_0_tensor_uid()));
    auto* input1
        = static_cast<Tensor<half>*>(&tensorBundle.getTensor(attributes.in_1_tensor_uid().value()));
    auto* output
        = static_cast<Tensor<bool>*>(&tensorBundle.getTensor(attributes.out_0_tensor_uid()));

    constexpr std::array INPUT0_VALUES = {-2.0f, 0.0f, 3.0f, 4.0f};
    constexpr std::array INPUT1_VALUES = {-3.0f, 0.0f, 5.0f, 1.0f};
    constexpr std::array EXPECTED = {true, false, true, true};
    for(int64_t width = 0; width < 4; ++width)
    {
        const auto widthIndex = static_cast<size_t>(width);
        input0->setHostValue(half(INPUT0_VALUES[widthIndex]), 0, 0, 0, width);
        input1->setHostValue(half(INPUT1_VALUES[widthIndex]), 0, 0, 0, width);
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

TEST_F(TestPointwisePlan, CompareNotEqualProducesBooleanMaskBfloat16)
{
    auto [graph, tensorBundle, variantPack]
        = buildPointwiseBinaryGraph({1, 1, 1, 4},
                                    {1, 1, 1, 4},
                                    {1, 1, 1, 4},
                                    DataType::BFLOAT16,
                                    DataType::BFLOAT16,
                                    DataType::FLOAT,
                                    DataType::BOOLEAN,
                                    hipdnn_frontend::PointwiseMode::CMP_NEQ);

    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    const GraphWrapper graphWrapper(serializedGraph.data(), serializedGraph.size());
    const auto& attributes = graphWrapper.getNodeWrapper(0).attributesAs<PointwiseAttributes>();
    auto* input0
        = static_cast<Tensor<bfloat16>*>(&tensorBundle.getTensor(attributes.in_0_tensor_uid()));
    auto* input1 = static_cast<Tensor<bfloat16>*>(
        &tensorBundle.getTensor(attributes.in_1_tensor_uid().value()));
    auto* output
        = static_cast<Tensor<bool>*>(&tensorBundle.getTensor(attributes.out_0_tensor_uid()));

    constexpr std::array INPUT0_VALUES = {-2.0f, 0.0f, 3.0f, 4.0f};
    constexpr std::array INPUT1_VALUES = {-3.0f, 0.0f, 5.0f, 1.0f};
    constexpr std::array EXPECTED = {true, false, true, true};
    for(int64_t width = 0; width < 4; ++width)
    {
        const auto widthIndex = static_cast<size_t>(width);
        input0->setHostValue(bfloat16(INPUT0_VALUES[widthIndex]), 0, 0, 0, width);
        input1->setHostValue(bfloat16(INPUT1_VALUES[widthIndex]), 0, 0, 0, width);
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

TEST_F(TestPointwisePlan, CompareGreaterEqualProducesBooleanMaskHalf)
{
    auto [graph, tensorBundle, variantPack]
        = buildPointwiseBinaryGraph({1, 1, 1, 4},
                                    {1, 1, 1, 4},
                                    {1, 1, 1, 4},
                                    DataType::HALF,
                                    DataType::HALF,
                                    DataType::FLOAT,
                                    DataType::BOOLEAN,
                                    hipdnn_frontend::PointwiseMode::CMP_GE);

    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    const GraphWrapper graphWrapper(serializedGraph.data(), serializedGraph.size());
    const auto& attributes = graphWrapper.getNodeWrapper(0).attributesAs<PointwiseAttributes>();
    auto* input0
        = static_cast<Tensor<half>*>(&tensorBundle.getTensor(attributes.in_0_tensor_uid()));
    auto* input1
        = static_cast<Tensor<half>*>(&tensorBundle.getTensor(attributes.in_1_tensor_uid().value()));
    auto* output
        = static_cast<Tensor<bool>*>(&tensorBundle.getTensor(attributes.out_0_tensor_uid()));

    constexpr std::array INPUT0_VALUES = {-2.0f, 0.0f, 3.0f, 4.0f};
    constexpr std::array INPUT1_VALUES = {-3.0f, 0.0f, 5.0f, 1.0f};
    constexpr std::array EXPECTED = {true, true, false, true};
    for(int64_t width = 0; width < 4; ++width)
    {
        const auto widthIndex = static_cast<size_t>(width);
        input0->setHostValue(half(INPUT0_VALUES[widthIndex]), 0, 0, 0, width);
        input1->setHostValue(half(INPUT1_VALUES[widthIndex]), 0, 0, 0, width);
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

TEST_F(TestPointwisePlan, CompareGreaterEqualProducesBooleanMaskBfloat16)
{
    auto [graph, tensorBundle, variantPack]
        = buildPointwiseBinaryGraph({1, 1, 1, 4},
                                    {1, 1, 1, 4},
                                    {1, 1, 1, 4},
                                    DataType::BFLOAT16,
                                    DataType::BFLOAT16,
                                    DataType::FLOAT,
                                    DataType::BOOLEAN,
                                    hipdnn_frontend::PointwiseMode::CMP_GE);

    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    const GraphWrapper graphWrapper(serializedGraph.data(), serializedGraph.size());
    const auto& attributes = graphWrapper.getNodeWrapper(0).attributesAs<PointwiseAttributes>();
    auto* input0
        = static_cast<Tensor<bfloat16>*>(&tensorBundle.getTensor(attributes.in_0_tensor_uid()));
    auto* input1 = static_cast<Tensor<bfloat16>*>(
        &tensorBundle.getTensor(attributes.in_1_tensor_uid().value()));
    auto* output
        = static_cast<Tensor<bool>*>(&tensorBundle.getTensor(attributes.out_0_tensor_uid()));

    constexpr std::array INPUT0_VALUES = {-2.0f, 0.0f, 3.0f, 4.0f};
    constexpr std::array INPUT1_VALUES = {-3.0f, 0.0f, 5.0f, 1.0f};
    constexpr std::array EXPECTED = {true, true, false, true};
    for(int64_t width = 0; width < 4; ++width)
    {
        const auto widthIndex = static_cast<size_t>(width);
        input0->setHostValue(bfloat16(INPUT0_VALUES[widthIndex]), 0, 0, 0, width);
        input1->setHostValue(bfloat16(INPUT1_VALUES[widthIndex]), 0, 0, 0, width);
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

TEST_F(TestPointwisePlan, CompareLessProducesBooleanMaskHalf)
{
    auto [graph, tensorBundle, variantPack]
        = buildPointwiseBinaryGraph({1, 1, 1, 4},
                                    {1, 1, 1, 4},
                                    {1, 1, 1, 4},
                                    DataType::HALF,
                                    DataType::HALF,
                                    DataType::FLOAT,
                                    DataType::BOOLEAN,
                                    hipdnn_frontend::PointwiseMode::CMP_LT);

    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    const GraphWrapper graphWrapper(serializedGraph.data(), serializedGraph.size());
    const auto& attributes = graphWrapper.getNodeWrapper(0).attributesAs<PointwiseAttributes>();
    auto* input0
        = static_cast<Tensor<half>*>(&tensorBundle.getTensor(attributes.in_0_tensor_uid()));
    auto* input1
        = static_cast<Tensor<half>*>(&tensorBundle.getTensor(attributes.in_1_tensor_uid().value()));
    auto* output
        = static_cast<Tensor<bool>*>(&tensorBundle.getTensor(attributes.out_0_tensor_uid()));

    constexpr std::array INPUT0_VALUES = {-2.0f, 0.0f, 3.0f, 4.0f};
    constexpr std::array INPUT1_VALUES = {-3.0f, 0.0f, 5.0f, 1.0f};
    constexpr std::array EXPECTED = {false, false, true, false};
    for(int64_t width = 0; width < 4; ++width)
    {
        const auto widthIndex = static_cast<size_t>(width);
        input0->setHostValue(half(INPUT0_VALUES[widthIndex]), 0, 0, 0, width);
        input1->setHostValue(half(INPUT1_VALUES[widthIndex]), 0, 0, 0, width);
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

TEST_F(TestPointwisePlan, CompareLessProducesBooleanMaskBfloat16)
{
    auto [graph, tensorBundle, variantPack]
        = buildPointwiseBinaryGraph({1, 1, 1, 4},
                                    {1, 1, 1, 4},
                                    {1, 1, 1, 4},
                                    DataType::BFLOAT16,
                                    DataType::BFLOAT16,
                                    DataType::FLOAT,
                                    DataType::BOOLEAN,
                                    hipdnn_frontend::PointwiseMode::CMP_LT);

    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    const GraphWrapper graphWrapper(serializedGraph.data(), serializedGraph.size());
    const auto& attributes = graphWrapper.getNodeWrapper(0).attributesAs<PointwiseAttributes>();
    auto* input0
        = static_cast<Tensor<bfloat16>*>(&tensorBundle.getTensor(attributes.in_0_tensor_uid()));
    auto* input1 = static_cast<Tensor<bfloat16>*>(
        &tensorBundle.getTensor(attributes.in_1_tensor_uid().value()));
    auto* output
        = static_cast<Tensor<bool>*>(&tensorBundle.getTensor(attributes.out_0_tensor_uid()));

    constexpr std::array INPUT0_VALUES = {-2.0f, 0.0f, 3.0f, 4.0f};
    constexpr std::array INPUT1_VALUES = {-3.0f, 0.0f, 5.0f, 1.0f};
    constexpr std::array EXPECTED = {false, false, true, false};
    for(int64_t width = 0; width < 4; ++width)
    {
        const auto widthIndex = static_cast<size_t>(width);
        input0->setHostValue(bfloat16(INPUT0_VALUES[widthIndex]), 0, 0, 0, width);
        input1->setHostValue(bfloat16(INPUT1_VALUES[widthIndex]), 0, 0, 0, width);
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

TEST_F(TestPointwisePlan, CompareLessEqualProducesBooleanMaskHalf)
{
    auto [graph, tensorBundle, variantPack]
        = buildPointwiseBinaryGraph({1, 1, 1, 4},
                                    {1, 1, 1, 4},
                                    {1, 1, 1, 4},
                                    DataType::HALF,
                                    DataType::HALF,
                                    DataType::FLOAT,
                                    DataType::BOOLEAN,
                                    hipdnn_frontend::PointwiseMode::CMP_LE);

    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    const GraphWrapper graphWrapper(serializedGraph.data(), serializedGraph.size());
    const auto& attributes = graphWrapper.getNodeWrapper(0).attributesAs<PointwiseAttributes>();
    auto* input0
        = static_cast<Tensor<half>*>(&tensorBundle.getTensor(attributes.in_0_tensor_uid()));
    auto* input1
        = static_cast<Tensor<half>*>(&tensorBundle.getTensor(attributes.in_1_tensor_uid().value()));
    auto* output
        = static_cast<Tensor<bool>*>(&tensorBundle.getTensor(attributes.out_0_tensor_uid()));

    constexpr std::array INPUT0_VALUES = {-2.0f, 0.0f, 3.0f, 4.0f};
    constexpr std::array INPUT1_VALUES = {-3.0f, 0.0f, 5.0f, 1.0f};
    constexpr std::array EXPECTED = {false, true, true, false};
    for(int64_t width = 0; width < 4; ++width)
    {
        const auto widthIndex = static_cast<size_t>(width);
        input0->setHostValue(half(INPUT0_VALUES[widthIndex]), 0, 0, 0, width);
        input1->setHostValue(half(INPUT1_VALUES[widthIndex]), 0, 0, 0, width);
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

TEST_F(TestPointwisePlan, CompareLessEqualProducesBooleanMaskBfloat16)
{
    auto [graph, tensorBundle, variantPack]
        = buildPointwiseBinaryGraph({1, 1, 1, 4},
                                    {1, 1, 1, 4},
                                    {1, 1, 1, 4},
                                    DataType::BFLOAT16,
                                    DataType::BFLOAT16,
                                    DataType::FLOAT,
                                    DataType::BOOLEAN,
                                    hipdnn_frontend::PointwiseMode::CMP_LE);

    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    const GraphWrapper graphWrapper(serializedGraph.data(), serializedGraph.size());
    const auto& attributes = graphWrapper.getNodeWrapper(0).attributesAs<PointwiseAttributes>();
    auto* input0
        = static_cast<Tensor<bfloat16>*>(&tensorBundle.getTensor(attributes.in_0_tensor_uid()));
    auto* input1 = static_cast<Tensor<bfloat16>*>(
        &tensorBundle.getTensor(attributes.in_1_tensor_uid().value()));
    auto* output
        = static_cast<Tensor<bool>*>(&tensorBundle.getTensor(attributes.out_0_tensor_uid()));

    constexpr std::array INPUT0_VALUES = {-2.0f, 0.0f, 3.0f, 4.0f};
    constexpr std::array INPUT1_VALUES = {-3.0f, 0.0f, 5.0f, 1.0f};
    constexpr std::array EXPECTED = {false, true, true, false};
    for(int64_t width = 0; width < 4; ++width)
    {
        const auto widthIndex = static_cast<size_t>(width);
        input0->setHostValue(bfloat16(INPUT0_VALUES[widthIndex]), 0, 0, 0, width);
        input1->setHostValue(bfloat16(INPUT1_VALUES[widthIndex]), 0, 0, 0, width);
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

TEST_F(TestPointwisePlan, LogicalOrProducesBooleanMask)
{
    auto [graph, tensorBundle, variantPack]
        = buildPointwiseBinaryGraph({1, 1, 1, 4},
                                    {1, 1, 1, 4},
                                    {1, 1, 1, 4},
                                    DataType::BOOLEAN,
                                    DataType::BOOLEAN,
                                    DataType::FLOAT,
                                    DataType::BOOLEAN,
                                    hipdnn_frontend::PointwiseMode::LOGICAL_OR);

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
    constexpr std::array EXPECTED = {true, true, true, false};
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

TEST_F(TestPointwisePlan, DivideBroadcastProducesExpectedValues)
{
    auto [graph, tensorBundle, variantPack]
        = buildPointwiseBinaryGraph({1, 2, 2, 2},
                                    {1, 2, 1, 1},
                                    {1, 2, 2, 2},
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    hipdnn_frontend::PointwiseMode::DIV);

    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    const GraphWrapper graphWrapper(serializedGraph.data(), serializedGraph.size());
    const auto& attributes = graphWrapper.getNodeWrapper(0).attributesAs<PointwiseAttributes>();
    auto* input0
        = static_cast<Tensor<float>*>(&tensorBundle.getTensor(attributes.in_0_tensor_uid()));
    auto* input1 = static_cast<Tensor<float>*>(
        &tensorBundle.getTensor(attributes.in_1_tensor_uid().value()));
    auto* output
        = static_cast<Tensor<float>*>(&tensorBundle.getTensor(attributes.out_0_tensor_uid()));

    // Channel 0 is divided by 2, channel 1 by 4.
    input1->setHostValue(2.0f, 0, 0, 0, 0);
    input1->setHostValue(4.0f, 0, 1, 0, 0);
    for(int64_t channel = 0; channel < 2; ++channel)
    {
        for(int64_t height = 0; height < 2; ++height)
        {
            for(int64_t width = 0; width < 2; ++width)
            {
                input0->setHostValue(static_cast<float>(8 + channel * 8 + height * 2 + width),
                                     0,
                                     channel,
                                     height,
                                     width);
            }
        }
    }
    input0->markHostModified();
    input1->markHostModified();

    CpuReferenceGraphExecutor{}.execute(
        serializedGraph.data(), serializedGraph.size(), variantPack);

    for(int64_t channel = 0; channel < 2; ++channel)
    {
        for(int64_t height = 0; height < 2; ++height)
        {
            for(int64_t width = 0; width < 2; ++width)
            {
                const auto dividend = static_cast<float>(8 + channel * 8 + height * 2 + width);
                const float divisor = channel == 0 ? 2.0f : 4.0f;
                EXPECT_NEAR(static_cast<float>(output->getHostValue(0, channel, height, width)),
                            dividend / divisor,
                            0.02f);
            }
        }
    }
}

TEST_F(TestPointwisePlan, AddSquareBroadcastProducesExpectedValues)
{
    auto [graph, tensorBundle, variantPack]
        = buildPointwiseBinaryGraph({1, 2, 2, 2},
                                    {1, 2, 1, 1},
                                    {1, 2, 2, 2},
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    hipdnn_frontend::PointwiseMode::ADD_SQUARE);

    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    const GraphWrapper graphWrapper(serializedGraph.data(), serializedGraph.size());
    const auto& attributes = graphWrapper.getNodeWrapper(0).attributesAs<PointwiseAttributes>();
    auto* input0
        = static_cast<Tensor<float>*>(&tensorBundle.getTensor(attributes.in_0_tensor_uid()));
    auto* input1 = static_cast<Tensor<float>*>(
        &tensorBundle.getTensor(attributes.in_1_tensor_uid().value()));
    auto* output
        = static_cast<Tensor<float>*>(&tensorBundle.getTensor(attributes.out_0_tensor_uid()));

    // Channel 0 adds 2^2, channel 1 adds 3^2.
    input1->setHostValue(2.0f, 0, 0, 0, 0);
    input1->setHostValue(3.0f, 0, 1, 0, 0);
    for(int64_t channel = 0; channel < 2; ++channel)
    {
        for(int64_t height = 0; height < 2; ++height)
        {
            for(int64_t width = 0; width < 2; ++width)
            {
                input0->setHostValue(
                    static_cast<float>(1 + height * 2 + width), 0, channel, height, width);
            }
        }
    }
    input0->markHostModified();
    input1->markHostModified();

    CpuReferenceGraphExecutor{}.execute(
        serializedGraph.data(), serializedGraph.size(), variantPack);

    for(int64_t channel = 0; channel < 2; ++channel)
    {
        for(int64_t height = 0; height < 2; ++height)
        {
            for(int64_t width = 0; width < 2; ++width)
            {
                const auto addend = static_cast<float>(1 + height * 2 + width);
                const float squared = channel == 0 ? 4.0f : 9.0f;
                EXPECT_NEAR(static_cast<float>(output->getHostValue(0, channel, height, width)),
                            addend + squared,
                            0.1f);
            }
        }
    }
}

TEST_F(TestPointwisePlan, DivideHalfBroadcastProducesExpectedValues)
{
    auto [graph, tensorBundle, variantPack]
        = buildPointwiseBinaryGraph({1, 2, 2, 2},
                                    {1, 2, 1, 1},
                                    {1, 2, 2, 2},
                                    DataType::HALF,
                                    DataType::HALF,
                                    DataType::FLOAT,
                                    DataType::HALF,
                                    hipdnn_frontend::PointwiseMode::DIV);

    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    const GraphWrapper graphWrapper(serializedGraph.data(), serializedGraph.size());
    const auto& attributes = graphWrapper.getNodeWrapper(0).attributesAs<PointwiseAttributes>();
    auto* input0
        = static_cast<Tensor<half>*>(&tensorBundle.getTensor(attributes.in_0_tensor_uid()));
    auto* input1
        = static_cast<Tensor<half>*>(&tensorBundle.getTensor(attributes.in_1_tensor_uid().value()));
    auto* output
        = static_cast<Tensor<half>*>(&tensorBundle.getTensor(attributes.out_0_tensor_uid()));

    // Channel 0 is divided by 2, channel 1 by 4.
    input1->setHostValue(half(2.0f), 0, 0, 0, 0);
    input1->setHostValue(half(4.0f), 0, 1, 0, 0);
    for(int64_t channel = 0; channel < 2; ++channel)
    {
        for(int64_t height = 0; height < 2; ++height)
        {
            for(int64_t width = 0; width < 2; ++width)
            {
                input0->setHostValue(half(static_cast<float>(8 + channel * 8 + height * 2 + width)),
                                     0,
                                     channel,
                                     height,
                                     width);
            }
        }
    }
    input0->markHostModified();
    input1->markHostModified();

    CpuReferenceGraphExecutor{}.execute(
        serializedGraph.data(), serializedGraph.size(), variantPack);

    for(int64_t channel = 0; channel < 2; ++channel)
    {
        for(int64_t height = 0; height < 2; ++height)
        {
            for(int64_t width = 0; width < 2; ++width)
            {
                const auto dividend = static_cast<float>(8 + channel * 8 + height * 2 + width);
                const float divisor = channel == 0 ? 2.0f : 4.0f;
                EXPECT_NEAR(static_cast<float>(output->getHostValue(0, channel, height, width)),
                            dividend / divisor,
                            0.02f);
            }
        }
    }
}

TEST_F(TestPointwisePlan, AddSquareHalfBroadcastProducesExpectedValues)
{
    auto [graph, tensorBundle, variantPack]
        = buildPointwiseBinaryGraph({1, 2, 2, 2},
                                    {1, 2, 1, 1},
                                    {1, 2, 2, 2},
                                    DataType::HALF,
                                    DataType::HALF,
                                    DataType::FLOAT,
                                    DataType::HALF,
                                    hipdnn_frontend::PointwiseMode::ADD_SQUARE);

    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    const GraphWrapper graphWrapper(serializedGraph.data(), serializedGraph.size());
    const auto& attributes = graphWrapper.getNodeWrapper(0).attributesAs<PointwiseAttributes>();
    auto* input0
        = static_cast<Tensor<half>*>(&tensorBundle.getTensor(attributes.in_0_tensor_uid()));
    auto* input1
        = static_cast<Tensor<half>*>(&tensorBundle.getTensor(attributes.in_1_tensor_uid().value()));
    auto* output
        = static_cast<Tensor<half>*>(&tensorBundle.getTensor(attributes.out_0_tensor_uid()));

    // Channel 0 adds 2^2, channel 1 adds 3^2.
    input1->setHostValue(half(2.0f), 0, 0, 0, 0);
    input1->setHostValue(half(3.0f), 0, 1, 0, 0);
    for(int64_t channel = 0; channel < 2; ++channel)
    {
        for(int64_t height = 0; height < 2; ++height)
        {
            for(int64_t width = 0; width < 2; ++width)
            {
                input0->setHostValue(
                    half(static_cast<float>(1 + height * 2 + width)), 0, channel, height, width);
            }
        }
    }
    input0->markHostModified();
    input1->markHostModified();

    CpuReferenceGraphExecutor{}.execute(
        serializedGraph.data(), serializedGraph.size(), variantPack);

    for(int64_t channel = 0; channel < 2; ++channel)
    {
        for(int64_t height = 0; height < 2; ++height)
        {
            for(int64_t width = 0; width < 2; ++width)
            {
                const auto addend = static_cast<float>(1 + height * 2 + width);
                const float squared = channel == 0 ? 4.0f : 9.0f;
                EXPECT_NEAR(static_cast<float>(output->getHostValue(0, channel, height, width)),
                            addend + squared,
                            0.1f);
            }
        }
    }
}

TEST_F(TestPointwisePlan, DivideBfloat16BroadcastProducesExpectedValues)
{
    auto [graph, tensorBundle, variantPack]
        = buildPointwiseBinaryGraph({1, 2, 2, 2},
                                    {1, 2, 1, 1},
                                    {1, 2, 2, 2},
                                    DataType::BFLOAT16,
                                    DataType::BFLOAT16,
                                    DataType::FLOAT,
                                    DataType::BFLOAT16,
                                    hipdnn_frontend::PointwiseMode::DIV);

    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    const GraphWrapper graphWrapper(serializedGraph.data(), serializedGraph.size());
    const auto& attributes = graphWrapper.getNodeWrapper(0).attributesAs<PointwiseAttributes>();
    auto* input0
        = static_cast<Tensor<bfloat16>*>(&tensorBundle.getTensor(attributes.in_0_tensor_uid()));
    auto* input1 = static_cast<Tensor<bfloat16>*>(
        &tensorBundle.getTensor(attributes.in_1_tensor_uid().value()));
    auto* output
        = static_cast<Tensor<bfloat16>*>(&tensorBundle.getTensor(attributes.out_0_tensor_uid()));

    // Channel 0 is divided by 2, channel 1 by 4.
    input1->setHostValue(bfloat16(2.0f), 0, 0, 0, 0);
    input1->setHostValue(bfloat16(4.0f), 0, 1, 0, 0);
    for(int64_t channel = 0; channel < 2; ++channel)
    {
        for(int64_t height = 0; height < 2; ++height)
        {
            for(int64_t width = 0; width < 2; ++width)
            {
                input0->setHostValue(
                    bfloat16(static_cast<float>(8 + channel * 8 + height * 2 + width)),
                    0,
                    channel,
                    height,
                    width);
            }
        }
    }
    input0->markHostModified();
    input1->markHostModified();

    CpuReferenceGraphExecutor{}.execute(
        serializedGraph.data(), serializedGraph.size(), variantPack);

    for(int64_t channel = 0; channel < 2; ++channel)
    {
        for(int64_t height = 0; height < 2; ++height)
        {
            for(int64_t width = 0; width < 2; ++width)
            {
                const auto dividend = static_cast<float>(8 + channel * 8 + height * 2 + width);
                const float divisor = channel == 0 ? 2.0f : 4.0f;
                EXPECT_NEAR(static_cast<float>(output->getHostValue(0, channel, height, width)),
                            dividend / divisor,
                            0.02f);
            }
        }
    }
}

TEST_F(TestPointwisePlan, AddSquareBfloat16BroadcastProducesExpectedValues)
{
    auto [graph, tensorBundle, variantPack]
        = buildPointwiseBinaryGraph({1, 2, 2, 2},
                                    {1, 2, 1, 1},
                                    {1, 2, 2, 2},
                                    DataType::BFLOAT16,
                                    DataType::BFLOAT16,
                                    DataType::FLOAT,
                                    DataType::BFLOAT16,
                                    hipdnn_frontend::PointwiseMode::ADD_SQUARE);

    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    const GraphWrapper graphWrapper(serializedGraph.data(), serializedGraph.size());
    const auto& attributes = graphWrapper.getNodeWrapper(0).attributesAs<PointwiseAttributes>();
    auto* input0
        = static_cast<Tensor<bfloat16>*>(&tensorBundle.getTensor(attributes.in_0_tensor_uid()));
    auto* input1 = static_cast<Tensor<bfloat16>*>(
        &tensorBundle.getTensor(attributes.in_1_tensor_uid().value()));
    auto* output
        = static_cast<Tensor<bfloat16>*>(&tensorBundle.getTensor(attributes.out_0_tensor_uid()));

    // Channel 0 adds 2^2, channel 1 adds 3^2.
    input1->setHostValue(bfloat16(2.0f), 0, 0, 0, 0);
    input1->setHostValue(bfloat16(3.0f), 0, 1, 0, 0);
    for(int64_t channel = 0; channel < 2; ++channel)
    {
        for(int64_t height = 0; height < 2; ++height)
        {
            for(int64_t width = 0; width < 2; ++width)
            {
                input0->setHostValue(bfloat16(static_cast<float>(1 + height * 2 + width)),
                                     0,
                                     channel,
                                     height,
                                     width);
            }
        }
    }
    input0->markHostModified();
    input1->markHostModified();

    CpuReferenceGraphExecutor{}.execute(
        serializedGraph.data(), serializedGraph.size(), variantPack);

    for(int64_t channel = 0; channel < 2; ++channel)
    {
        for(int64_t height = 0; height < 2; ++height)
        {
            for(int64_t width = 0; width < 2; ++width)
            {
                const auto addend = static_cast<float>(1 + height * 2 + width);
                const float squared = channel == 0 ? 4.0f : 9.0f;
                EXPECT_NEAR(static_cast<float>(output->getHostValue(0, channel, height, width)),
                            addend + squared,
                            0.1f);
            }
        }
    }
}

TEST_F(TestPointwisePlan, LogicalAndBroadcastProducesBooleanMask)
{
    auto [graph, tensorBundle, variantPack]
        = buildPointwiseBinaryGraph({1, 1, 2, 4},
                                    {1, 1, 1, 4},
                                    {1, 1, 2, 4},
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

    // The second operand is broadcast over the height dimension.
    constexpr std::array INPUT1_VALUES = {true, false, true, false};
    for(int64_t width = 0; width < 4; ++width)
    {
        const auto widthIndex = static_cast<size_t>(width);
        input1->setHostValue(INPUT1_VALUES[widthIndex], 0, 0, 0, width);
        input0->setHostValue(true, 0, 0, 0, width);
        input0->setHostValue(width < 2, 0, 0, 1, width);
    }
    input0->markHostModified();
    input1->markHostModified();

    CpuReferenceGraphExecutor{}.execute(
        serializedGraph.data(), serializedGraph.size(), variantPack);

    for(int64_t height = 0; height < 2; ++height)
    {
        for(int64_t width = 0; width < 4; ++width)
        {
            const bool first = height == 0 || width < 2;
            const bool second = INPUT1_VALUES[static_cast<size_t>(width)];
            EXPECT_EQ(output->getHostValue(0, 0, height, width), first && second);
        }
    }
}

TEST_F(TestPointwisePlan, LogicalOrBroadcastProducesBooleanMask)
{
    auto [graph, tensorBundle, variantPack]
        = buildPointwiseBinaryGraph({1, 1, 2, 4},
                                    {1, 1, 1, 4},
                                    {1, 1, 2, 4},
                                    DataType::BOOLEAN,
                                    DataType::BOOLEAN,
                                    DataType::FLOAT,
                                    DataType::BOOLEAN,
                                    hipdnn_frontend::PointwiseMode::LOGICAL_OR);

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

    // The second operand is broadcast over the height dimension.
    constexpr std::array INPUT1_VALUES = {true, false, true, false};
    for(int64_t width = 0; width < 4; ++width)
    {
        const auto widthIndex = static_cast<size_t>(width);
        input1->setHostValue(INPUT1_VALUES[widthIndex], 0, 0, 0, width);
        input0->setHostValue(true, 0, 0, 0, width);
        input0->setHostValue(width < 2, 0, 0, 1, width);
    }
    input0->markHostModified();
    input1->markHostModified();

    CpuReferenceGraphExecutor{}.execute(
        serializedGraph.data(), serializedGraph.size(), variantPack);

    for(int64_t height = 0; height < 2; ++height)
    {
        for(int64_t width = 0; width < 4; ++width)
        {
            const bool first = height == 0 || width < 2;
            const bool second = INPUT1_VALUES[static_cast<size_t>(width)];
            EXPECT_EQ(output->getHostValue(0, 0, height, width), first || second);
        }
    }
}

TEST_F(TestPointwisePlan, GeluBwdProducesExpectedGradient)
{
    auto [graph, tensorBundle, variantPack]
        = buildPointwiseBinaryGraph({1, 1, 1, 4},
                                    {1, 1, 1, 4},
                                    {1, 1, 1, 4},
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    hipdnn_frontend::PointwiseMode::GELU_BWD,
                                    1,
                                    TensorLayout::NCHW);

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

    const auto gradient = [](double x) {
        return 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)))
               + x * std::exp(-x * x / 2.0) / std::sqrt(2.0 * 3.14159265);
    };
    for(int64_t width = 0; width < 4; ++width)
    {
        const auto widthIndex = static_cast<size_t>(width);
        const double expected = static_cast<double>(DY_VALUES[widthIndex])
                                * gradient(static_cast<double>(X_VALUES[widthIndex]));
        EXPECT_NEAR(static_cast<double>(dx->getHostValue(0, 0, 0, width)), expected, 1e-5);
    }
}

TEST_F(TestPointwisePlan, GeluApproxTanhBwdProducesExpectedGradient)
{
    auto [graph, tensorBundle, variantPack]
        = buildPointwiseBinaryGraph({1, 1, 1, 4},
                                    {1, 1, 1, 4},
                                    {1, 1, 1, 4},
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    hipdnn_frontend::PointwiseMode::GELU_APPROX_TANH_BWD,
                                    1,
                                    TensorLayout::NCHW);

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

    const auto gradient = [](double x) {
        const double c = std::sqrt(2.0 / 3.14159265);
        const double tanhVal = std::tanh(c * (x + 0.044715 * x * x * x));
        return 0.5 * (1.0 + tanhVal)
               + 0.5 * x * (1.0 - tanhVal * tanhVal) * c * (1.0 + 3.0 * 0.044715 * x * x);
    };
    for(int64_t width = 0; width < 4; ++width)
    {
        const auto widthIndex = static_cast<size_t>(width);
        const double expected = static_cast<double>(DY_VALUES[widthIndex])
                                * gradient(static_cast<double>(X_VALUES[widthIndex]));
        EXPECT_NEAR(static_cast<double>(dx->getHostValue(0, 0, 0, width)), expected, 1e-5);
    }
}

TEST_F(TestPointwisePlan, SwishBwdProducesExpectedGradient)
{
    auto [graph, tensorBundle, variantPack]
        = buildPointwiseBinaryGraph({1, 1, 1, 4},
                                    {1, 1, 1, 4},
                                    {1, 1, 1, 4},
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    hipdnn_frontend::PointwiseMode::SWISH_BWD,
                                    1,
                                    TensorLayout::NCHW);

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

    const auto gradient = [](double x) {
        const double s = 1.0 / (1.0 + std::exp(-x));
        return s + x * s * (1.0 - s);
    };
    for(int64_t width = 0; width < 4; ++width)
    {
        const auto widthIndex = static_cast<size_t>(width);
        const double expected = static_cast<double>(DY_VALUES[widthIndex])
                                * gradient(static_cast<double>(X_VALUES[widthIndex]));
        EXPECT_NEAR(static_cast<double>(dx->getHostValue(0, 0, 0, width)), expected, 1e-5);
    }
}

TEST_F(TestPointwisePlan, SwishBwdWithBetaProducesExpectedGradient)
{
    auto [graph, tensorBundle, variantPack]
        = buildPointwiseBinaryGraph({1, 1, 1, 4},
                                    {1, 1, 1, 4},
                                    {1, 1, 1, 4},
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    hipdnn_frontend::PointwiseMode::SWISH_BWD,
                                    1,
                                    TensorLayout::NCHW,
                                    std::nullopt,
                                    std::nullopt,
                                    std::nullopt,
                                    0.5f,
                                    std::nullopt,
                                    std::nullopt);

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

    const auto gradient = [](double x) {
        const double s = 1.0 / (1.0 + std::exp(-0.5 * x));
        return s + 0.5 * x * s * (1.0 - s);
    };
    for(int64_t width = 0; width < 4; ++width)
    {
        const auto widthIndex = static_cast<size_t>(width);
        const double expected = static_cast<double>(DY_VALUES[widthIndex])
                                * gradient(static_cast<double>(X_VALUES[widthIndex]));
        EXPECT_NEAR(static_cast<double>(dx->getHostValue(0, 0, 0, width)), expected, 1e-5);
    }
}

TEST_F(TestPointwisePlan, EluBwdProducesExpectedGradient)
{
    auto [graph, tensorBundle, variantPack]
        = buildPointwiseBinaryGraph({1, 1, 1, 4},
                                    {1, 1, 1, 4},
                                    {1, 1, 1, 4},
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    hipdnn_frontend::PointwiseMode::ELU_BWD,
                                    1,
                                    TensorLayout::NCHW);

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

    const auto gradient = [](double x) { return x > 0.0 ? 1.0 : std::exp(x); };
    for(int64_t width = 0; width < 4; ++width)
    {
        const auto widthIndex = static_cast<size_t>(width);
        const double expected = static_cast<double>(DY_VALUES[widthIndex])
                                * gradient(static_cast<double>(X_VALUES[widthIndex]));
        EXPECT_NEAR(static_cast<double>(dx->getHostValue(0, 0, 0, width)), expected, 1e-5);
    }
}

TEST_F(TestPointwisePlan, EluBwdWithAlphaProducesExpectedGradient)
{
    auto [graph, tensorBundle, variantPack]
        = buildPointwiseBinaryGraph({1, 1, 1, 4},
                                    {1, 1, 1, 4},
                                    {1, 1, 1, 4},
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    hipdnn_frontend::PointwiseMode::ELU_BWD,
                                    1,
                                    TensorLayout::NCHW,
                                    std::nullopt,
                                    std::nullopt,
                                    std::nullopt,
                                    std::nullopt,
                                    0.5f,
                                    std::nullopt);

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

    const auto gradient = [](double x) { return x > 0.0 ? 1.0 : 0.5 * std::exp(x); };
    for(int64_t width = 0; width < 4; ++width)
    {
        const auto widthIndex = static_cast<size_t>(width);
        const double expected = static_cast<double>(DY_VALUES[widthIndex])
                                * gradient(static_cast<double>(X_VALUES[widthIndex]));
        EXPECT_NEAR(static_cast<double>(dx->getHostValue(0, 0, 0, width)), expected, 1e-5);
    }
}

TEST_F(TestPointwisePlan, SoftplusBwdProducesExpectedGradient)
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
                                    TensorLayout::NCHW);

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

    const auto gradient = [](double x) { return 1.0 / (1.0 + std::exp(-x)); };
    for(int64_t width = 0; width < 4; ++width)
    {
        const auto widthIndex = static_cast<size_t>(width);
        const double expected = static_cast<double>(DY_VALUES[widthIndex])
                                * gradient(static_cast<double>(X_VALUES[widthIndex]));
        EXPECT_NEAR(static_cast<double>(dx->getHostValue(0, 0, 0, width)), expected, 1e-5);
    }
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

TEST_F(TestPointwisePlan, GenIndexProducesCoordinateAlongAxis)
{
    auto [graph, tensorBundle, variantPack]
        = buildPointwiseUnaryGraph({1, 3, 2, 2},
                                   {1, 3, 2, 2},
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
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
        = static_cast<Tensor<float>*>(&tensorBundle.getTensor(attributes.out_0_tensor_uid()));

    CpuReferenceGraphExecutor{}.execute(
        serializedGraph.data(), serializedGraph.size(), variantPack);

    for(int64_t channel = 0; channel < 3; ++channel)
    {
        for(int64_t height = 0; height < 2; ++height)
        {
            for(int64_t width = 0; width < 2; ++width)
            {
                EXPECT_EQ(output->getHostValue(0, channel, height, width),
                          static_cast<float>(channel));
            }
        }
    }
}

TEST_F(TestPointwisePlan, GenIndexLastAxisHalfOutput)
{
    auto [graph, tensorBundle, variantPack]
        = buildPointwiseUnaryGraph({1, 2, 2, 4},
                                   {1, 2, 2, 4},
                                   DataType::HALF,
                                   DataType::FLOAT,
                                   DataType::HALF,
                                   hipdnn_frontend::PointwiseMode::GEN_INDEX,
                                   1,
                                   TensorLayout::NCHW,
                                   std::nullopt,
                                   std::nullopt,
                                   std::nullopt,
                                   std::nullopt,
                                   std::nullopt,
                                   std::nullopt,
                                   int64_t{3});

    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    const GraphWrapper graphWrapper(serializedGraph.data(), serializedGraph.size());
    const auto& attributes = graphWrapper.getNodeWrapper(0).attributesAs<PointwiseAttributes>();
    auto* output
        = static_cast<Tensor<half>*>(&tensorBundle.getTensor(attributes.out_0_tensor_uid()));

    CpuReferenceGraphExecutor{}.execute(
        serializedGraph.data(), serializedGraph.size(), variantPack);

    for(int64_t width = 0; width < 4; ++width)
    {
        EXPECT_EQ(static_cast<float>(output->getHostValue(0, 1, 1, width)),
                  static_cast<float>(width));
    }
}

TEST_F(TestPointwisePlan, GenIndexInt32Output)
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
                                   int64_t{2});

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
                          static_cast<int32_t>(height));
            }
        }
    }
}

TEST(TestPointwisePlanBuilder, PlanConstructionGenIndex)
{
    auto [graph, tensorBundle, variantPack]
        = buildPointwiseUnaryGraph({1, 3, 4, 4},
                                   {1, 3, 4, 4},
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
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
                               DataType::FLOAT>
        patient;
    auto builtPlan = patient.buildNodePlan(graphWrap, graphWrap.getNode(0));

    const bool result
        = dynamic_cast<PointwisePlan<float, float, float, float>*>(builtPlan.get()) != nullptr;
    EXPECT_TRUE(result);
}

TEST(TestPointwisePlanBuilder, IsApplicableGenIndex)
{
    auto [graph, tensorBundle, variantPack]
        = buildPointwiseUnaryGraph({1, 3, 4, 4},
                                   {1, 3, 4, 4},
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
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

TEST(TestPointwisePlanBuilder, PlanBuilderThrowsIfAxisSetForNonGenIndexMode)
{
    auto [graph, tensorBundle, variantPack]
        = buildPointwiseUnaryGraph({1, 3, 4, 4},
                                   {1, 3, 4, 4},
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   hipdnn_frontend::PointwiseMode::RELU_FWD,
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
                               DataType::FLOAT>
        planBuilder;
    EXPECT_THROW(planBuilder.buildNodePlan(graphWrap, graphWrap.getNode(0)), std::runtime_error);
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
TEST(TestPointwisePlanBuilder, RejectsParameterizedTernary)
{
    auto [graph, tensorBundle, variantPack]
        = buildPointwiseTernaryGraph({1},
                                     {1},
                                     {1},
                                     {1},
                                     DataType::FLOAT,
                                     DataType::FLOAT,
                                     DataType::BOOLEAN,
                                     DataType::FLOAT,
                                     DataType::FLOAT,
                                     hipdnn_frontend::PointwiseMode::BINARY_SELECT,
                                     TensorLayout::NCHW,
                                     0.0f);

    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    const GraphWrapper graphWrapper(serializedGraph.data(), serializedGraph.size());
    const PointwisePlanBuilder<DataType::FLOAT,
                               DataType::FLOAT,
                               DataType::BOOLEAN,
                               DataType::FLOAT,
                               DataType::FLOAT>
        planBuilder;

    EXPECT_THROW(planBuilder.buildNodePlan(graphWrapper, graphWrapper.getNode(0)),
                 std::runtime_error);
}
