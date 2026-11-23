// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <array>
#include <gtest/gtest.h>

#include "PointwiseGraphUtils.hpp"
#include "hipdnn_flatbuffers_sdk/data_objects/pointwise_attributes_generated.h"
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferDatatypeMapping.hpp>
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

template <DataType InputDataType, DataType OutputDataType>
struct IdentitySignature
{
    static constexpr DataType INPUT_DATA_TYPE = InputDataType;
    static constexpr DataType OUTPUT_DATA_TYPE = OutputDataType;
    using InputType = DataTypeToNative<InputDataType>;
    using OutputType = DataTypeToNative<OutputDataType>;
};

using IdentitySignatures
    = ::testing::Types<IdentitySignature<DataType::FLOAT, DataType::FLOAT>,
                       IdentitySignature<DataType::HALF, DataType::HALF>,
                       IdentitySignature<DataType::BFLOAT16, DataType::BFLOAT16>,
                       IdentitySignature<DataType::FLOAT, DataType::HALF>,
                       IdentitySignature<DataType::FLOAT, DataType::BFLOAT16>>;

template <typename Signature>
class PointwiseIdentityPlan : public ::testing::Test
{
};

TYPED_TEST_SUITE(PointwiseIdentityPlan, IdentitySignatures, );

TYPED_TEST(PointwiseIdentityPlan, ExecutePlanUnaryIdentityPreservesValues)
{
    using InputType = typename TypeParam::InputType;
    using OutputType = typename TypeParam::OutputType;
    constexpr auto INPUT_DATA_TYPE = TypeParam::INPUT_DATA_TYPE;
    constexpr auto OUTPUT_DATA_TYPE = TypeParam::OUTPUT_DATA_TYPE;
    auto [graph, tensorBundle, variantPack]
        = buildPointwiseUnaryGraph({1, 3, 2, 2},
                                   {1, 3, 2, 2},
                                   INPUT_DATA_TYPE,
                                   DataType::FLOAT,
                                   OUTPUT_DATA_TYPE,
                                   hipdnn_frontend::PointwiseMode::IDENTITY);
    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    const GraphWrapper graphWrapper(serializedGraph.data(), serializedGraph.size());
    const NodeWrapper nodeWrapper(&graphWrapper.getNode(0));
    const auto& attributes = nodeWrapper.attributesAs<PointwiseAttributes>();
    auto* input = static_cast<InputType*>(variantPack.at(attributes.in_0_tensor_uid()));
    const auto* output
        = static_cast<const OutputType*>(variantPack.at(attributes.out_0_tensor_uid()));
    const std::array<float, 12> expected
        = {-3.0f, -2.0f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    for(size_t i = 0; i < expected.size(); ++i)
    {
        input[i] = static_cast<InputType>(expected[i]);
    }
    CpuReferenceGraphExecutor{}.execute(
        serializedGraph.data(), serializedGraph.size(), variantPack);

    for(size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_EQ(static_cast<float>(output[i]), expected[i]) << "element " << i;
    }
}

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

TEST(TestPointwisePlanBuilder, PlanCarriesNodeComputeType)
{
    const std::vector<int64_t> dims = {1, 3, 4, 4};

    auto [graph, tensorBundle, variantPack]
        = buildPointwiseUnaryGraph(dims,
                                   dims,
                                   DataType::HALF,
                                   DataType::FLOAT,
                                   DataType::HALF,
                                   hipdnn_frontend::PointwiseMode::ABS,
                                   1,
                                   TensorLayout::NCHW);

    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        serializedGraph.data(), serializedGraph.size());

    const PointwisePlanBuilder<DataType::HALF,
                               DataType::HALF,
                               DataType::HALF,
                               DataType::FLOAT,
                               DataType::HALF>
        patient;
    ASSERT_TRUE(patient.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap()));
    auto builtPlan = patient.buildNodePlan(graphWrap, graphWrap.getNode(0));

    EXPECT_NE((dynamic_cast<PointwisePlan<hipdnn_data_sdk::types::half,
                                          hipdnn_data_sdk::types::half,
                                          hipdnn_data_sdk::types::half,
                                          hipdnn_data_sdk::types::half,
                                          float>*>(builtPlan.get())),
              nullptr);
}

// 2049 is not representable in half (11-bit significand) but is in float.
template <DataType ComputeDataTypeEnum>
static float executeHalfAddWithCompute()
{
    const std::vector<int64_t> dims = {1, 1, 1, 1};

    auto [graph, tensorBundle, variantPack]
        = buildPointwiseBinaryGraph(dims,
                                    dims,
                                    dims,
                                    DataType::HALF,
                                    DataType::HALF,
                                    ComputeDataTypeEnum,
                                    DataType::FLOAT,
                                    hipdnn_frontend::PointwiseMode::ADD,
                                    1,
                                    TensorLayout::NCHW);

    auto [serializedGraph, serErr] = graph->to_binary();
    EXPECT_TRUE(serErr.is_good()) << serErr.get_message();
    const GraphWrapper graphWrapper(serializedGraph.data(), serializedGraph.size());
    const auto& attributes = graphWrapper.getNodeWrapper(0).attributesAs<PointwiseAttributes>();

    auto* input0
        = static_cast<Tensor<half>*>(&tensorBundle.getTensor(attributes.in_0_tensor_uid()));
    auto* input1
        = static_cast<Tensor<half>*>(&tensorBundle.getTensor(attributes.in_1_tensor_uid().value()));
    auto* output
        = static_cast<Tensor<float>*>(&tensorBundle.getTensor(attributes.out_0_tensor_uid()));
    input0->setHostValue(static_cast<half>(2048.0f), 0, 0, 0, 0);
    input1->setHostValue(static_cast<half>(1.0f), 0, 0, 0, 0);
    input0->markHostModified();
    input1->markHostModified();

    const PointwisePlanBuilder<DataType::HALF,
                               DataType::HALF,
                               DataType::FLOAT,
                               ComputeDataTypeEnum,
                               DataType::FLOAT>
        planBuilder;
    auto plan = planBuilder.buildNodePlan(graphWrapper, graphWrapper.getNode(0));
    plan->execute(variantPack);

    return output->getHostValue(0, 0, 0, 0);
}

TEST(TestPointwisePlanBuilder, ExecutePlanUsesNodeComputeType)
{
    EXPECT_EQ(executeHalfAddWithCompute<DataType::FLOAT>(), 2049.0f);
    EXPECT_EQ(executeHalfAddWithCompute<DataType::HALF>(), 2048.0f);
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
        = dynamic_cast<PointwisePlan<float, float, float, float, float>*>(builtPlan.get())
          != nullptr;
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
        = dynamic_cast<PointwisePlan<float, float, float, float, float>*>(builtPlan.get())
          != nullptr;
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
        = dynamic_cast<PointwisePlan<float, float, float, int32_t, float>*>(builtPlan.get())
          != nullptr;
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

TEST_F(TestPointwisePlan, GenIndexAcceptsDoubleInput)
{
    auto [graph, tensorBundle, variantPack]
        = buildPointwiseUnaryGraph({1, 3, 2, 2},
                                   {1, 3, 2, 2},
                                   DataType::DOUBLE,
                                   DataType::DOUBLE,
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
        EXPECT_EQ(output->getHostValue(0, channel, 1, 1), static_cast<int32_t>(channel));
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

TEST(TestPointwisePlanBuilder, DeclinesNonBroadcastableShapes)
{
    auto [graph, tensorBundle, variantPack]
        = buildPointwiseBinaryGraph({1, 3, 2, 2},
                                    {1, 3, 2, 2},
                                    {1, 3, 2, 2},
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    hipdnn_frontend::PointwiseMode::ADD,
                                    1,
                                    TensorLayout::NCHW);

    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();

    // The frontend rejects non-broadcastable shapes, so edit the serialized graph instead
    auto graphT = std::unique_ptr<GraphT>(GetGraph(serializedGraph.data())->UnPack());
    const GraphWrapper original(serializedGraph.data(), serializedGraph.size());
    const auto secondInputUid
        = original.getNodeWrapper(0).attributesAs<PointwiseAttributes>().in_1_tensor_uid().value();
    for(auto& tensor : graphT->tensors)
    {
        if(tensor->uid == secondInputUid)
        {
            tensor->dims = {1, 3, 5, 2};
        }
    }
    flatbuffers::FlatBufferBuilder builder;
    builder.Finish(CreateGraph(builder, graphT.get()));
    auto badGraph = builder.Release();

    CpuReferenceGraphExecutor executor;
    EXPECT_FALSE(executor.isApplicable(badGraph.data(), badGraph.size()));
    EXPECT_THROW(executor.execute(badGraph.data(), badGraph.size(), variantPack),
                 CpuReferenceNotApplicableError);
}

TEST(TestPointwisePlanBuilder, MissingPlanBuilderIsNotApplicable)
{
    PlanBuilderRegistry registry;
    const PointwiseSignatureKey unregistered(
        PointwiseMode::ADD, DataType::INT32, DataType::HALF, DataType::INT32, DataType::INT32);

    EXPECT_THROW(registry.getPlanBuilder(unregistered), CpuReferenceNotApplicableError);
}

TEST(TestPointwisePlanBuilder, ExecuteDeclinesUnregisteredDataTypeTuple)
{
    // INT32 with HALF compute has no registered plan builder
    auto [graph, tensorBundle, variantPack]
        = buildPointwiseUnaryGraph({1, 1, 1, 1},
                                   {1, 1, 1, 1},
                                   DataType::INT32,
                                   DataType::HALF,
                                   DataType::INT32,
                                   hipdnn_frontend::PointwiseMode::ABS,
                                   1,
                                   TensorLayout::NCHW);

    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();

    CpuReferenceGraphExecutor executor;
    EXPECT_FALSE(executor.isApplicable(serializedGraph.data(), serializedGraph.size()));
    EXPECT_THROW(executor.execute(serializedGraph.data(), serializedGraph.size(), variantPack),
                 CpuReferenceNotApplicableError);
}

TEST(TestPointwisePlanBuilder, ScalarOperandTypes)
{
    const std::vector<int64_t> dims = {1, 2, 2, 2};
    const auto strides = hipdnn_data_sdk::utilities::generateStrides(dims);

    auto graph = std::make_shared<hipdnn_frontend::graph::Graph>();
    graph->set_io_data_type(hipdnn_frontend::DataType::FLOAT)
        .set_compute_data_type(hipdnn_frontend::DataType::FLOAT)
        .set_intermediate_data_type(hipdnn_frontend::DataType::FLOAT);

    auto xAttr = hipdnn_frontend::graph::makeTensorAttributes(
        "x", hipdnn_frontend::DataType::FLOAT, dims, strides);
    xAttr.set_uid(1);
    auto x = std::make_shared<hipdnn_frontend::graph::TensorAttributes>(std::move(xAttr));

    // Baked constant, runtime scalar with a default, and pure runtime scalar
    auto baked = std::make_shared<hipdnn_frontend::graph::TensorAttributes>(2.0f);
    auto runtimeWithDefault = std::make_shared<hipdnn_frontend::graph::TensorAttributes>(
        4.0f, hipdnn_frontend::graph::ScalarType::RUNTIME_PARAM);
    auto pureRuntime = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    pureRuntime->set_dim({1}).set_stride({1});
    pureRuntime->set_data_type(hipdnn_frontend::DataType::FLOAT).set_as_runtime_parameter();
    baked->set_uid(2);
    runtimeWithDefault->set_uid(3);
    pureRuntime->set_uid(4);

    hipdnn_frontend::graph::PointwiseAttributes addAttrs;
    addAttrs.set_mode(hipdnn_frontend::PointwiseMode::ADD);
    hipdnn_frontend::graph::PointwiseAttributes mulAttrs;
    mulAttrs.set_mode(hipdnn_frontend::PointwiseMode::MUL);
    hipdnn_frontend::graph::PointwiseAttributes subAttrs;
    subAttrs.set_mode(hipdnn_frontend::PointwiseMode::SUB);

    // added = x + baked, multiplied = added * runtimeWithDefault, out = pureRuntime - multiplied.
    // Each intermediate is also an output so every operation can be checked on its own.
    auto added = graph->pointwise(x, baked, addAttrs);
    auto multiplied = graph->pointwise(added, runtimeWithDefault, mulAttrs);
    auto out = graph->pointwise(pureRuntime, multiplied, subAttrs);
    int64_t outUid = 5;
    for(const auto& output : {added, multiplied, out})
    {
        output->set_uid(outUid++)
            .set_data_type(hipdnn_frontend::DataType::FLOAT)
            .set_dim(dims)
            .set_stride(strides)
            .set_output(true);
    }

    auto validateResult = graph->validate();
    ASSERT_TRUE(validateResult.is_good()) << validateResult.get_message();
    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();

    // Distinct value per element: 1..8
    Tensor<float> xData(dims);
    for(int64_t i = 0; i < 8; ++i)
    {
        xData.setHostValue(static_cast<float>(i + 1), 0, i / 4, (i / 2) % 2, i % 2);
    }
    xData.markHostModified();
    Tensor<float> addData(dims);
    Tensor<float> mulData(dims);
    Tensor<float> outData(dims);
    float pureRuntimeValue = 10.0f;
    // The baked and defaulted scalars have no entry in the variant pack
    const std::unordered_map<int64_t, void*> variantPack = {{1, xData.rawHostData()},
                                                            {4, &pureRuntimeValue},
                                                            {5, addData.rawHostData()},
                                                            {6, mulData.rawHostData()},
                                                            {7, outData.rawHostData()}};

    CpuReferenceGraphExecutor executor;
    ASSERT_TRUE(executor.isApplicable(serializedGraph.data(), serializedGraph.size()));
    executor.execute(serializedGraph.data(), serializedGraph.size(), variantPack);

    for(int64_t i = 0; i < 8; ++i)
    {
        const auto xValue = static_cast<float>(i + 1);
        const auto c = i / 4;
        const auto h = (i / 2) % 2;
        const auto w = i % 2;
        EXPECT_EQ(xData.getHostValue(0, c, h, w), xValue) << "input changed at element " << i;
        EXPECT_EQ(addData.getHostValue(0, c, h, w), xValue + 2.0f) << "element " << i;
        EXPECT_EQ(mulData.getHostValue(0, c, h, w), (xValue + 2.0f) * 4.0f) << "element " << i;
        EXPECT_EQ(outData.getHostValue(0, c, h, w), 10.0f - ((xValue + 2.0f) * 4.0f))
            << "element " << i;
    }
}

TEST(TestPointwisePlanBuilder, DeclinesNonBroadcastableFirstAndThirdInputs)
{
    for(const bool editThirdInput : {false, true})
    {
        auto [graph, tensorBundle, variantPack]
            = buildPointwiseTernaryGraph({1, 3, 2, 2},
                                         {1, 3, 2, 2},
                                         {1, 3, 2, 2},
                                         {1, 3, 2, 2},
                                         DataType::FLOAT,
                                         DataType::FLOAT,
                                         DataType::BOOLEAN,
                                         DataType::FLOAT,
                                         DataType::FLOAT,
                                         hipdnn_frontend::PointwiseMode::BINARY_SELECT);

        auto [serializedGraph, serErr] = graph->to_binary();
        ASSERT_TRUE(serErr.is_good()) << serErr.get_message();

        // The frontend rejects non-broadcastable shapes, so edit the serialized graph instead
        auto graphT = std::unique_ptr<GraphT>(GetGraph(serializedGraph.data())->UnPack());
        const GraphWrapper original(serializedGraph.data(), serializedGraph.size());
        const auto& originalAttributes
            = original.getNodeWrapper(0).attributesAs<PointwiseAttributes>();
        const auto editedUid = editThirdInput ? originalAttributes.in_2_tensor_uid().value()
                                              : originalAttributes.in_0_tensor_uid();
        for(auto& tensor : graphT->tensors)
        {
            if(tensor->uid == editedUid)
            {
                tensor->dims = {1, 3, 5, 2};
            }
        }
        flatbuffers::FlatBufferBuilder builder;
        builder.Finish(CreateGraph(builder, graphT.get()));
        auto badGraph = builder.Release();

        CpuReferenceGraphExecutor executor;
        EXPECT_FALSE(executor.isApplicable(badGraph.data(), badGraph.size()));
        EXPECT_THROW(executor.execute(badGraph.data(), badGraph.size(), variantPack),
                     CpuReferenceNotApplicableError);
    }
}

TEST(TestPointwisePlanBuilder, ScalarOperandsOfUnaryAndTernaryNodes)
{
    const std::vector<int64_t> dims = {1, 1, 1, 4};
    const auto strides = hipdnn_data_sdk::utilities::generateStrides(dims);

    auto graph = std::make_shared<hipdnn_frontend::graph::Graph>();
    graph->set_io_data_type(hipdnn_frontend::DataType::FLOAT)
        .set_compute_data_type(hipdnn_frontend::DataType::FLOAT)
        .set_intermediate_data_type(hipdnn_frontend::DataType::FLOAT);

    auto maskAttr = hipdnn_frontend::graph::makeTensorAttributes(
        "mask", hipdnn_frontend::DataType::BOOLEAN, dims, strides);
    maskAttr.set_uid(1);
    auto mask = std::make_shared<hipdnn_frontend::graph::TensorAttributes>(std::move(maskAttr));

    // The unary node reads a baked scalar, and the ternary node selects between a runtime scalar
    // with a default and the unary result
    auto absInput = std::make_shared<hipdnn_frontend::graph::TensorAttributes>(-3.0f);
    auto selectOnTrue = std::make_shared<hipdnn_frontend::graph::TensorAttributes>(
        5.0f, hipdnn_frontend::graph::ScalarType::RUNTIME_PARAM);
    absInput->set_uid(2);
    selectOnTrue->set_uid(3);

    hipdnn_frontend::graph::PointwiseAttributes absAttrs;
    absAttrs.set_mode(hipdnn_frontend::PointwiseMode::ABS);
    hipdnn_frontend::graph::PointwiseAttributes selectAttrs;
    selectAttrs.set_mode(hipdnn_frontend::PointwiseMode::BINARY_SELECT);

    auto absOut = graph->pointwise(absInput, absAttrs);
    absOut->set_uid(4)
        .set_data_type(hipdnn_frontend::DataType::FLOAT)
        .set_dim(dims)
        .set_stride(strides);
    auto selectOut = graph->pointwise(selectOnTrue, absOut, mask, selectAttrs);
    selectOut->set_uid(5)
        .set_data_type(hipdnn_frontend::DataType::FLOAT)
        .set_dim(dims)
        .set_stride(strides)
        .set_output(true);

    auto validateResult = graph->validate();
    ASSERT_TRUE(validateResult.is_good()) << validateResult.get_message();
    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();

    Tensor<bool> maskData(dims);
    for(int64_t i = 0; i < 4; ++i)
    {
        maskData.setHostValue(i % 2 == 0, 0, 0, 0, i);
    }
    maskData.markHostModified();
    Tensor<float> selectData(dims);
    // The baked and defaulted scalars have no entry in the variant pack
    const std::unordered_map<int64_t, void*> variantPack
        = {{1, maskData.rawHostData()}, {5, selectData.rawHostData()}};

    CpuReferenceGraphExecutor executor;
    ASSERT_TRUE(executor.isApplicable(serializedGraph.data(), serializedGraph.size()));
    executor.execute(serializedGraph.data(), serializedGraph.size(), variantPack);

    // Selected scalar 5 where the mask is true, otherwise abs(-3)
    for(int64_t i = 0; i < 4; ++i)
    {
        EXPECT_EQ(selectData.getHostValue(0, 0, 0, i), i % 2 == 0 ? 5.0f : 3.0f) << "element " << i;
    }
}

TEST(TestPointwisePlanBuilder, DoubleAndInt32ScalarOperands)
{
    const std::vector<int64_t> dims = {1, 1, 1, 2};
    const auto strides = hipdnn_data_sdk::utilities::generateStrides(dims);

    // FP64 scalar: 1 + 1e-12 is not representable in float
    auto doubleGraph = std::make_shared<hipdnn_frontend::graph::Graph>();
    doubleGraph->set_io_data_type(hipdnn_frontend::DataType::DOUBLE)
        .set_compute_data_type(hipdnn_frontend::DataType::DOUBLE)
        .set_intermediate_data_type(hipdnn_frontend::DataType::DOUBLE);
    auto doubleAttr = hipdnn_frontend::graph::makeTensorAttributes(
        "x", hipdnn_frontend::DataType::DOUBLE, dims, strides);
    doubleAttr.set_uid(1);
    auto doubleX
        = std::make_shared<hipdnn_frontend::graph::TensorAttributes>(std::move(doubleAttr));
    auto doubleScalar = std::make_shared<hipdnn_frontend::graph::TensorAttributes>(1e-12);
    doubleScalar->set_uid(2);
    hipdnn_frontend::graph::PointwiseAttributes addAttrs;
    addAttrs.set_mode(hipdnn_frontend::PointwiseMode::ADD);
    auto doubleOut = doubleGraph->pointwise(doubleX, doubleScalar, addAttrs);
    doubleOut->set_uid(3)
        .set_data_type(hipdnn_frontend::DataType::DOUBLE)
        .set_dim(dims)
        .set_stride(strides)
        .set_output(true);
    ASSERT_TRUE(doubleGraph->validate().is_good());
    auto [doubleSerialized, doubleErr] = doubleGraph->to_binary();
    ASSERT_TRUE(doubleErr.is_good()) << doubleErr.get_message();

    Tensor<double> doubleXData(dims);
    doubleXData.fillWithValue(1.0);
    Tensor<double> doubleOutData(dims);
    const std::unordered_map<int64_t, void*> doublePack
        = {{1, doubleXData.rawHostData()}, {3, doubleOutData.rawHostData()}};
    CpuReferenceGraphExecutor executor;
    ASSERT_TRUE(executor.isApplicable(doubleSerialized.data(), doubleSerialized.size()));
    executor.execute(doubleSerialized.data(), doubleSerialized.size(), doublePack);
    EXPECT_EQ(doubleOutData.getHostValue(0, 0, 0, 1), 1.0 + 1e-12);

    // INT32 scalar
    auto intGraph = std::make_shared<hipdnn_frontend::graph::Graph>();
    intGraph->set_io_data_type(hipdnn_frontend::DataType::INT32)
        .set_compute_data_type(hipdnn_frontend::DataType::FLOAT)
        .set_intermediate_data_type(hipdnn_frontend::DataType::INT32);
    auto intAttr = hipdnn_frontend::graph::makeTensorAttributes(
        "x", hipdnn_frontend::DataType::INT32, dims, strides);
    intAttr.set_uid(1);
    auto intX = std::make_shared<hipdnn_frontend::graph::TensorAttributes>(std::move(intAttr));
    auto intScalar = std::make_shared<hipdnn_frontend::graph::TensorAttributes>(int32_t{40});
    intScalar->set_uid(2);
    auto intOut = intGraph->pointwise(intX, intScalar, addAttrs);
    intOut->set_uid(3)
        .set_data_type(hipdnn_frontend::DataType::INT32)
        .set_dim(dims)
        .set_stride(strides)
        .set_output(true);
    ASSERT_TRUE(intGraph->validate().is_good());
    auto [intSerialized, intErr] = intGraph->to_binary();
    ASSERT_TRUE(intErr.is_good()) << intErr.get_message();

    Tensor<int32_t> intXData(dims);
    intXData.fillWithValue(2);
    Tensor<int32_t> intOutData(dims);
    const std::unordered_map<int64_t, void*> intPack
        = {{1, intXData.rawHostData()}, {3, intOutData.rawHostData()}};
    ASSERT_TRUE(executor.isApplicable(intSerialized.data(), intSerialized.size()));
    executor.execute(intSerialized.data(), intSerialized.size(), intPack);
    EXPECT_EQ(intOutData.getHostValue(0, 0, 0, 1), 42);
}

TEST(TestPointwisePlanDataTypes, DoubleAddKeepsDoublePrecision)
{
    auto [graph, tensorBundle, variantPack]
        = buildPointwiseBinaryGraph({1, 1, 1, 1},
                                    {1, 1, 1, 1},
                                    {1, 1, 1, 1},
                                    DataType::DOUBLE,
                                    DataType::DOUBLE,
                                    DataType::DOUBLE,
                                    DataType::DOUBLE,
                                    hipdnn_frontend::PointwiseMode::ADD);

    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    const GraphWrapper graphWrapper(serializedGraph.data(), serializedGraph.size());
    const auto& attributes = graphWrapper.getNodeWrapper(0).attributesAs<PointwiseAttributes>();
    auto* input0
        = static_cast<Tensor<double>*>(&tensorBundle.getTensor(attributes.in_0_tensor_uid()));
    auto* input1 = static_cast<Tensor<double>*>(
        &tensorBundle.getTensor(attributes.in_1_tensor_uid().value()));
    auto* output
        = static_cast<Tensor<double>*>(&tensorBundle.getTensor(attributes.out_0_tensor_uid()));
    input0->setHostValue(1.0, 0, 0, 0, 0);
    input1->setHostValue(1e-12, 0, 0, 0, 0);
    input0->markHostModified();
    input1->markHostModified();

    CpuReferenceGraphExecutor executor;
    ASSERT_TRUE(executor.isApplicable(serializedGraph.data(), serializedGraph.size()));
    executor.execute(serializedGraph.data(), serializedGraph.size(), variantPack);

    // 1 + 1e-12 is not representable in float
    EXPECT_EQ(output->getHostValue(0, 0, 0, 0), 1.0 + 1e-12);
}

TEST(TestPointwisePlanDataTypes, Int32Multiply)
{
    auto [graph, tensorBundle, variantPack]
        = buildPointwiseBinaryGraph({1, 1, 1, 1},
                                    {1, 1, 1, 1},
                                    {1, 1, 1, 1},
                                    DataType::INT32,
                                    DataType::INT32,
                                    DataType::FLOAT,
                                    DataType::INT32,
                                    hipdnn_frontend::PointwiseMode::MUL);

    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    const GraphWrapper graphWrapper(serializedGraph.data(), serializedGraph.size());
    const auto& attributes = graphWrapper.getNodeWrapper(0).attributesAs<PointwiseAttributes>();
    auto* input0
        = static_cast<Tensor<int32_t>*>(&tensorBundle.getTensor(attributes.in_0_tensor_uid()));
    auto* input1 = static_cast<Tensor<int32_t>*>(
        &tensorBundle.getTensor(attributes.in_1_tensor_uid().value()));
    auto* output
        = static_cast<Tensor<int32_t>*>(&tensorBundle.getTensor(attributes.out_0_tensor_uid()));
    input0->setHostValue(7, 0, 0, 0, 0);
    input1->setHostValue(-3, 0, 0, 0, 0);
    input0->markHostModified();
    input1->markHostModified();

    CpuReferenceGraphExecutor executor;
    ASSERT_TRUE(executor.isApplicable(serializedGraph.data(), serializedGraph.size()));
    executor.execute(serializedGraph.data(), serializedGraph.size(), variantPack);

    EXPECT_EQ(output->getHostValue(0, 0, 0, 0), -21);
}

TEST(TestPointwisePlanDataTypes, BooleanBinarySelect)
{
    auto [graph, tensorBundle, variantPack]
        = buildPointwiseTernaryGraph({1, 1, 1, 2},
                                     {1, 1, 1, 2},
                                     {1, 1, 1, 2},
                                     {1, 1, 1, 2},
                                     DataType::BOOLEAN,
                                     DataType::BOOLEAN,
                                     DataType::BOOLEAN,
                                     DataType::FLOAT,
                                     DataType::BOOLEAN,
                                     hipdnn_frontend::PointwiseMode::BINARY_SELECT);

    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    const GraphWrapper graphWrapper(serializedGraph.data(), serializedGraph.size());
    const auto& attributes = graphWrapper.getNodeWrapper(0).attributesAs<PointwiseAttributes>();
    auto* input0
        = static_cast<Tensor<bool>*>(&tensorBundle.getTensor(attributes.in_0_tensor_uid()));
    auto* input1
        = static_cast<Tensor<bool>*>(&tensorBundle.getTensor(attributes.in_1_tensor_uid().value()));
    auto* mask
        = static_cast<Tensor<bool>*>(&tensorBundle.getTensor(attributes.in_2_tensor_uid().value()));
    auto* output
        = static_cast<Tensor<bool>*>(&tensorBundle.getTensor(attributes.out_0_tensor_uid()));
    // Element 0 selects input0 (true), element 1 selects input1 (false)
    input0->setHostValue(true, 0, 0, 0, 0);
    input0->setHostValue(true, 0, 0, 0, 1);
    input1->setHostValue(false, 0, 0, 0, 0);
    input1->setHostValue(false, 0, 0, 0, 1);
    mask->setHostValue(true, 0, 0, 0, 0);
    mask->setHostValue(false, 0, 0, 0, 1);
    input0->markHostModified();
    input1->markHostModified();
    mask->markHostModified();

    CpuReferenceGraphExecutor executor;
    ASSERT_TRUE(executor.isApplicable(serializedGraph.data(), serializedGraph.size()));
    executor.execute(serializedGraph.data(), serializedGraph.size(), variantPack);

    EXPECT_TRUE(output->getHostValue(0, 0, 0, 0));
    EXPECT_FALSE(output->getHostValue(0, 0, 0, 1));
}

TEST(TestPointwisePlanDataTypes, Fp8E4M3AddToFloat)
{
    auto [graph, tensorBundle, variantPack]
        = buildPointwiseBinaryGraph({1, 1, 1, 1},
                                    {1, 1, 1, 1},
                                    {1, 1, 1, 1},
                                    DataType::FP8_E4M3,
                                    DataType::FP8_E4M3,
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    hipdnn_frontend::PointwiseMode::ADD);

    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    const GraphWrapper graphWrapper(serializedGraph.data(), serializedGraph.size());
    const auto& attributes = graphWrapper.getNodeWrapper(0).attributesAs<PointwiseAttributes>();
    auto* input0
        = static_cast<Tensor<fp8_e4m3>*>(&tensorBundle.getTensor(attributes.in_0_tensor_uid()));
    auto* input1 = static_cast<Tensor<fp8_e4m3>*>(
        &tensorBundle.getTensor(attributes.in_1_tensor_uid().value()));
    auto* output
        = static_cast<Tensor<float>*>(&tensorBundle.getTensor(attributes.out_0_tensor_uid()));
    input0->setHostValue(static_cast<fp8_e4m3>(1.5f), 0, 0, 0, 0);
    input1->setHostValue(static_cast<fp8_e4m3>(2.0f), 0, 0, 0, 0);
    input0->markHostModified();
    input1->markHostModified();

    CpuReferenceGraphExecutor executor;
    ASSERT_TRUE(executor.isApplicable(serializedGraph.data(), serializedGraph.size()));
    executor.execute(serializedGraph.data(), serializedGraph.size(), variantPack);

    EXPECT_EQ(output->getHostValue(0, 0, 0, 0), 3.5f);
}

TEST(TestPointwisePlanDataTypes, Fp8E5M2NegKeepsFp8Output)
{
    auto [graph, tensorBundle, variantPack]
        = buildPointwiseUnaryGraph({1, 1, 1, 1},
                                   {1, 1, 1, 1},
                                   DataType::FP8_E5M2,
                                   DataType::FLOAT,
                                   DataType::FP8_E5M2,
                                   hipdnn_frontend::PointwiseMode::NEG);

    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    const GraphWrapper graphWrapper(serializedGraph.data(), serializedGraph.size());
    const auto& attributes = graphWrapper.getNodeWrapper(0).attributesAs<PointwiseAttributes>();
    auto* input
        = static_cast<Tensor<fp8_e5m2>*>(&tensorBundle.getTensor(attributes.in_0_tensor_uid()));
    auto* output
        = static_cast<Tensor<fp8_e5m2>*>(&tensorBundle.getTensor(attributes.out_0_tensor_uid()));
    input->setHostValue(static_cast<fp8_e5m2>(2.0f), 0, 0, 0, 0);
    input->markHostModified();

    CpuReferenceGraphExecutor executor;
    ASSERT_TRUE(executor.isApplicable(serializedGraph.data(), serializedGraph.size()));
    executor.execute(serializedGraph.data(), serializedGraph.size(), variantPack);

    EXPECT_EQ(static_cast<float>(output->getHostValue(0, 0, 0, 0)), -2.0f);
}

TEST(TestPointwisePlanDataTypes, CompareGreaterWithMixedInputTypes)
{
    auto [graph, tensorBundle, variantPack]
        = buildPointwiseBinaryGraph({1, 1, 1, 2},
                                    {1, 1, 1, 2},
                                    {1, 1, 1, 2},
                                    DataType::FLOAT,
                                    DataType::HALF,
                                    DataType::FLOAT,
                                    DataType::BOOLEAN,
                                    hipdnn_frontend::PointwiseMode::CMP_GT);

    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    const GraphWrapper graphWrapper(serializedGraph.data(), serializedGraph.size());
    const auto& attributes = graphWrapper.getNodeWrapper(0).attributesAs<PointwiseAttributes>();
    auto* input0
        = static_cast<Tensor<float>*>(&tensorBundle.getTensor(attributes.in_0_tensor_uid()));
    auto* input1
        = static_cast<Tensor<half>*>(&tensorBundle.getTensor(attributes.in_1_tensor_uid().value()));
    auto* output
        = static_cast<Tensor<bool>*>(&tensorBundle.getTensor(attributes.out_0_tensor_uid()));
    input0->setHostValue(3.0f, 0, 0, 0, 0);
    input0->setHostValue(1.0f, 0, 0, 0, 1);
    input1->setHostValue(static_cast<half>(2.0f), 0, 0, 0, 0);
    input1->setHostValue(static_cast<half>(2.0f), 0, 0, 0, 1);
    input0->markHostModified();
    input1->markHostModified();

    CpuReferenceGraphExecutor executor;
    ASSERT_TRUE(executor.isApplicable(serializedGraph.data(), serializedGraph.size()));
    executor.execute(serializedGraph.data(), serializedGraph.size(), variantPack);

    EXPECT_TRUE(output->getHostValue(0, 0, 0, 0));
    EXPECT_FALSE(output->getHostValue(0, 0, 0, 1));
}

namespace
{

constexpr int64_t TEST_ELEMENT_COUNT = 4;

template <typename T>
T fromTestValue(double value)
{
    if constexpr(std::is_same_v<T, double> || std::is_integral_v<T>)
    {
        return static_cast<T>(value);
    }
    else
    {
        return static_cast<T>(static_cast<float>(value));
    }
}

template <typename T>
double toTestValue(T value)
{
    if constexpr(std::is_same_v<T, double> || std::is_integral_v<T>)
    {
        return static_cast<double>(value);
    }
    else
    {
        return static_cast<double>(static_cast<float>(value));
    }
}

// Runs a unary pointwise graph through the CPU reference and returns every output element.
template <DataType InDataType, DataType ComputeDataType, DataType OutDataType>
std::vector<double> runUnary(hipdnn_frontend::PointwiseMode mode, const std::vector<double>& values)
{
    using InType = DataTypeToNative<InDataType>;
    using OutType = DataTypeToNative<OutDataType>;
    const std::vector<int64_t> dims = {1, 1, 1, TEST_ELEMENT_COUNT};

    auto [graph, tensorBundle, variantPack] = buildPointwiseUnaryGraph(
        dims, dims, InDataType, ComputeDataType, OutDataType, mode, 1, TensorLayout::NCHW);
    auto [serializedGraph, serErr] = graph->to_binary();
    EXPECT_TRUE(serErr.is_good()) << serErr.get_message();
    const GraphWrapper graphWrapper(serializedGraph.data(), serializedGraph.size());
    const auto& attributes = graphWrapper.getNodeWrapper(0).attributesAs<PointwiseAttributes>();
    auto* input
        = static_cast<Tensor<InType>*>(&tensorBundle.getTensor(attributes.in_0_tensor_uid()));
    auto* output
        = static_cast<Tensor<OutType>*>(&tensorBundle.getTensor(attributes.out_0_tensor_uid()));
    for(int64_t i = 0; i < TEST_ELEMENT_COUNT; ++i)
    {
        input->setHostValue(fromTestValue<InType>(values[static_cast<size_t>(i)]), 0, 0, 0, i);
    }
    input->markHostModified();

    CpuReferenceGraphExecutor executor;
    EXPECT_TRUE(executor.isApplicable(serializedGraph.data(), serializedGraph.size()));
    executor.execute(serializedGraph.data(), serializedGraph.size(), variantPack);

    std::vector<double> result;
    result.reserve(values.size());
    for(int64_t i = 0; i < TEST_ELEMENT_COUNT; ++i)
    {
        result.push_back(toTestValue(output->getHostValue(0, 0, 0, i)));
    }
    return result;
}

// Runs a binary pointwise graph through the CPU reference and returns every output element.
template <DataType In0DataType,
          DataType In1DataType,
          DataType ComputeDataType,
          DataType OutDataType>
std::vector<double> runBinary(hipdnn_frontend::PointwiseMode mode,
                              const std::vector<double>& values0,
                              const std::vector<double>& values1)
{
    using In0Type = DataTypeToNative<In0DataType>;
    using In1Type = DataTypeToNative<In1DataType>;
    using OutType = DataTypeToNative<OutDataType>;
    const std::vector<int64_t> dims = {1, 1, 1, TEST_ELEMENT_COUNT};

    auto [graph, tensorBundle, variantPack] = buildPointwiseBinaryGraph(dims,
                                                                        dims,
                                                                        dims,
                                                                        In0DataType,
                                                                        In1DataType,
                                                                        ComputeDataType,
                                                                        OutDataType,
                                                                        mode,
                                                                        1,
                                                                        TensorLayout::NCHW);
    auto [serializedGraph, serErr] = graph->to_binary();
    EXPECT_TRUE(serErr.is_good()) << serErr.get_message();
    const GraphWrapper graphWrapper(serializedGraph.data(), serializedGraph.size());
    const auto& attributes = graphWrapper.getNodeWrapper(0).attributesAs<PointwiseAttributes>();
    auto* input0
        = static_cast<Tensor<In0Type>*>(&tensorBundle.getTensor(attributes.in_0_tensor_uid()));
    auto* input1 = static_cast<Tensor<In1Type>*>(
        &tensorBundle.getTensor(attributes.in_1_tensor_uid().value()));
    auto* output
        = static_cast<Tensor<OutType>*>(&tensorBundle.getTensor(attributes.out_0_tensor_uid()));
    for(int64_t i = 0; i < TEST_ELEMENT_COUNT; ++i)
    {
        input0->setHostValue(fromTestValue<In0Type>(values0[static_cast<size_t>(i)]), 0, 0, 0, i);
        input1->setHostValue(fromTestValue<In1Type>(values1[static_cast<size_t>(i)]), 0, 0, 0, i);
    }
    input0->markHostModified();
    input1->markHostModified();

    CpuReferenceGraphExecutor executor;
    EXPECT_TRUE(executor.isApplicable(serializedGraph.data(), serializedGraph.size()));
    executor.execute(serializedGraph.data(), serializedGraph.size(), variantPack);

    std::vector<double> result;
    result.reserve(values0.size());
    for(int64_t i = 0; i < TEST_ELEMENT_COUNT; ++i)
    {
        result.push_back(toTestValue(output->getHostValue(0, 0, 0, i)));
    }
    return result;
}

// Runs BINARY_SELECT with a mask of {true, false, true, false} and returns every output element.
template <DataType DataTypeEnum, DataType ComputeDataType>
std::vector<double> runSelect(const std::vector<double>& values0,
                              const std::vector<double>& values1)
{
    using Type = DataTypeToNative<DataTypeEnum>;
    const std::vector<int64_t> dims = {1, 1, 1, TEST_ELEMENT_COUNT};

    auto [graph, tensorBundle, variantPack]
        = buildPointwiseTernaryGraph(dims,
                                     dims,
                                     dims,
                                     dims,
                                     DataTypeEnum,
                                     DataTypeEnum,
                                     DataType::BOOLEAN,
                                     ComputeDataType,
                                     DataTypeEnum,
                                     hipdnn_frontend::PointwiseMode::BINARY_SELECT);
    auto [serializedGraph, serErr] = graph->to_binary();
    EXPECT_TRUE(serErr.is_good()) << serErr.get_message();
    const GraphWrapper graphWrapper(serializedGraph.data(), serializedGraph.size());
    const auto& attributes = graphWrapper.getNodeWrapper(0).attributesAs<PointwiseAttributes>();
    auto* input0
        = static_cast<Tensor<Type>*>(&tensorBundle.getTensor(attributes.in_0_tensor_uid()));
    auto* input1
        = static_cast<Tensor<Type>*>(&tensorBundle.getTensor(attributes.in_1_tensor_uid().value()));
    auto* mask
        = static_cast<Tensor<bool>*>(&tensorBundle.getTensor(attributes.in_2_tensor_uid().value()));
    auto* output
        = static_cast<Tensor<Type>*>(&tensorBundle.getTensor(attributes.out_0_tensor_uid()));
    for(int64_t i = 0; i < TEST_ELEMENT_COUNT; ++i)
    {
        input0->setHostValue(fromTestValue<Type>(values0[static_cast<size_t>(i)]), 0, 0, 0, i);
        input1->setHostValue(fromTestValue<Type>(values1[static_cast<size_t>(i)]), 0, 0, 0, i);
        mask->setHostValue(i % 2 == 0, 0, 0, 0, i);
    }
    input0->markHostModified();
    input1->markHostModified();
    mask->markHostModified();

    CpuReferenceGraphExecutor executor;
    EXPECT_TRUE(executor.isApplicable(serializedGraph.data(), serializedGraph.size()));
    executor.execute(serializedGraph.data(), serializedGraph.size(), variantPack);

    std::vector<double> result;
    result.reserve(values0.size());
    for(int64_t i = 0; i < TEST_ELEMENT_COUNT; ++i)
    {
        result.push_back(toTestValue(output->getHostValue(0, 0, 0, i)));
    }
    return result;
}

} // namespace

TEST(TestPointwisePlanDataTypes, DoubleOperations)
{
    using hipdnn_frontend::PointwiseMode;
    constexpr auto D = DataType::DOUBLE;
    const std::vector<double> a = {-2.5, -0.5, 1.5, 4.0};
    const std::vector<double> b = {1.0, -0.5, 3.0, 2.0};

    EXPECT_EQ((runUnary<D, D, D>(PointwiseMode::RELU_FWD, a)),
              (std::vector<double>{0.0, 0.0, 1.5, 4.0}));
    EXPECT_EQ((runUnary<D, D, D>(PointwiseMode::ABS, a)),
              (std::vector<double>{2.5, 0.5, 1.5, 4.0}));
    EXPECT_EQ((runUnary<D, D, D>(PointwiseMode::NEG, a)),
              (std::vector<double>{2.5, 0.5, -1.5, -4.0}));
    EXPECT_NEAR(
        (runUnary<D, D, D>(PointwiseMode::SIGMOID_FWD, a))[3], 1.0 / (1.0 + std::exp(-4.0)), 1e-12);
    EXPECT_EQ((runBinary<D, D, D, D>(PointwiseMode::SUB, a, b)),
              (std::vector<double>{-3.5, 0.0, -1.5, 2.0}));
    EXPECT_EQ((runBinary<D, D, D, DataType::BOOLEAN>(PointwiseMode::CMP_GT, a, b)),
              (std::vector<double>{0.0, 0.0, 0.0, 1.0}));
    EXPECT_EQ((runBinary<D, D, D, DataType::BOOLEAN>(PointwiseMode::CMP_LE, a, b)),
              (std::vector<double>{1.0, 1.0, 1.0, 0.0}));
    EXPECT_EQ((runSelect<D, D>(a, b)), (std::vector<double>{-2.5, -0.5, 1.5, 2.0}));
}

TEST(TestPointwisePlanDataTypes, Int32Operations)
{
    using hipdnn_frontend::PointwiseMode;
    constexpr auto I = DataType::INT32;
    constexpr auto F = DataType::FLOAT;
    const std::vector<double> a = {-5, -1, 3, 8};
    const std::vector<double> b = {2, -1, 7, 4};

    EXPECT_EQ((runUnary<I, F, I>(PointwiseMode::RELU_FWD, a)), (std::vector<double>{0, 0, 3, 8}));
    EXPECT_EQ((runBinary<I, I, F, I>(PointwiseMode::SUB, a, b)),
              (std::vector<double>{-7, 0, -4, 4}));
    EXPECT_EQ((runBinary<I, I, F, I>(PointwiseMode::MAX, a, b)),
              (std::vector<double>{2, -1, 7, 8}));
    EXPECT_EQ((runBinary<I, I, F, I>(PointwiseMode::MIN, a, b)),
              (std::vector<double>{-5, -1, 3, 4}));
    EXPECT_EQ((runBinary<I, I, F, DataType::BOOLEAN>(PointwiseMode::CMP_GT, a, b)),
              (std::vector<double>{0, 0, 0, 1}));
    EXPECT_EQ((runSelect<I, F>(a, b)), (std::vector<double>{-5, -1, 3, 4}));
}

TEST(TestPointwisePlanDataTypes, Int32ComputesInFloatPrecision)
{
    constexpr auto I = DataType::INT32;
    constexpr auto F = DataType::FLOAT;

    // 2^24 + 1 is not representable in float, the compute type of INT32 nodes
    EXPECT_EQ((runBinary<I, I, F, I>(
                  hipdnn_frontend::PointwiseMode::ADD, {16777216, 1, 2, 3}, {1, 1, 1, 1})),
              (std::vector<double>{16777216, 2, 3, 4}));
}

TEST(TestPointwisePlanDataTypes, Fp8Operations)
{
    using hipdnn_frontend::PointwiseMode;
    constexpr auto E4 = DataType::FP8_E4M3;
    constexpr auto E5 = DataType::FP8_E5M2;
    constexpr auto F = DataType::FLOAT;
    // All values are exactly representable in both FP8 formats
    const std::vector<double> a = {-2.0, -0.5, 1.0, 4.0};
    const std::vector<double> b = {1.0, -0.5, 2.0, 2.0};

    EXPECT_EQ((runUnary<E4, F, E4>(PointwiseMode::ABS, a)),
              (std::vector<double>{2.0, 0.5, 1.0, 4.0}));
    EXPECT_EQ((runUnary<E4, F, F>(PointwiseMode::RELU_FWD, a)),
              (std::vector<double>{0.0, 0.0, 1.0, 4.0}));
    EXPECT_EQ((runUnary<E5, F, F>(PointwiseMode::NEG, a)),
              (std::vector<double>{2.0, 0.5, -1.0, -4.0}));
    EXPECT_EQ((runBinary<E4, E4, F, E4>(PointwiseMode::SUB, a, b)),
              (std::vector<double>{-3.0, 0.0, -1.0, 2.0}));
    EXPECT_EQ((runBinary<E5, E5, F, E5>(PointwiseMode::ADD, a, b)),
              (std::vector<double>{-1.0, -1.0, 3.0, 6.0}));
    EXPECT_EQ((runBinary<E5, E5, F, F>(PointwiseMode::MUL, a, b)),
              (std::vector<double>{-2.0, 0.25, 2.0, 8.0}));
    EXPECT_EQ((runBinary<E4, E4, F, DataType::BOOLEAN>(PointwiseMode::CMP_GT, a, b)),
              (std::vector<double>{0, 0, 0, 1}));
    EXPECT_EQ((runBinary<E5, E5, F, DataType::BOOLEAN>(PointwiseMode::CMP_GT, a, b)),
              (std::vector<double>{0, 0, 0, 1}));
    EXPECT_EQ((runSelect<E4, F>(a, b)), (std::vector<double>{-2.0, -0.5, 1.0, 2.0}));
    EXPECT_EQ((runSelect<E5, F>(a, b)), (std::vector<double>{-2.0, -0.5, 1.0, 2.0}));
}

TEST(TestPointwisePlanDataTypes, CompareGreaterWithEveryMixedInputPair)
{
    using hipdnn_frontend::PointwiseMode;
    constexpr auto F = DataType::FLOAT;
    constexpr auto H = DataType::HALF;
    constexpr auto BF = DataType::BFLOAT16;
    constexpr auto D = DataType::DOUBLE;
    constexpr auto BOOL = DataType::BOOLEAN;
    const std::vector<double> a = {3.0, 1.0, 2.0, -1.0};
    const std::vector<double> b = {2.0, 2.0, 2.0, -2.0};
    const std::vector<double> expected = {1, 0, 0, 1};

    EXPECT_EQ((runBinary<F, H, F, BOOL>(PointwiseMode::CMP_GT, a, b)), expected);
    EXPECT_EQ((runBinary<H, F, F, BOOL>(PointwiseMode::CMP_GT, a, b)), expected);
    EXPECT_EQ((runBinary<F, BF, F, BOOL>(PointwiseMode::CMP_GT, a, b)), expected);
    EXPECT_EQ((runBinary<BF, F, F, BOOL>(PointwiseMode::CMP_GT, a, b)), expected);
    EXPECT_EQ((runBinary<H, BF, F, BOOL>(PointwiseMode::CMP_GT, a, b)), expected);
    EXPECT_EQ((runBinary<BF, H, F, BOOL>(PointwiseMode::CMP_GT, a, b)), expected);
    EXPECT_EQ((runBinary<D, F, D, BOOL>(PointwiseMode::CMP_GT, a, b)), expected);
    EXPECT_EQ((runBinary<F, D, D, BOOL>(PointwiseMode::CMP_GT, a, b)), expected);
}

TEST(TestPointwisePlanDataTypes, DeclinesOperationsThatAreNotMeaningfulForTheType)
{
    const std::vector<int64_t> dims = {1, 1, 1, 1};

    for(const auto& [dataType, mode] :
        {std::pair{DataType::INT32, hipdnn_frontend::PointwiseMode::SIGMOID_FWD},
         std::pair{DataType::INT32, hipdnn_frontend::PointwiseMode::TANH_FWD},
         std::pair{DataType::INT32, hipdnn_frontend::PointwiseMode::ABS},
         std::pair{DataType::INT32, hipdnn_frontend::PointwiseMode::NEG},
         std::pair{DataType::BOOLEAN, hipdnn_frontend::PointwiseMode::RELU_FWD}})
    {
        auto [graph, tensorBundle, variantPack] = buildPointwiseUnaryGraph(
            dims, dims, dataType, DataType::FLOAT, dataType, mode, 1, TensorLayout::NCHW);
        auto [serializedGraph, serErr] = graph->to_binary();
        ASSERT_TRUE(serErr.is_good()) << serErr.get_message();

        CpuReferenceGraphExecutor executor;
        EXPECT_FALSE(executor.isApplicable(serializedGraph.data(), serializedGraph.size()));
        EXPECT_THROW(executor.execute(serializedGraph.data(), serializedGraph.size(), variantPack),
                     CpuReferenceNotApplicableError);
    }
}
