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

TEST(TestPointwisePlanBuilder, UnsupportedOperation)
{
    const std::vector<int64_t> inputDims = {1, 3, 4, 4};
    const std::vector<int64_t> outputDims = {1, 3, 4, 4};

    auto [graph, tensorBundle, variantPack]
        = buildPointwiseUnaryGraph(inputDims,
                                   outputDims,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   hipdnn_frontend::PointwiseMode::EXP, // Not implemented
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
        planBuilder;
    EXPECT_FALSE(planBuilder.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap()));
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
    EXPECT_FALSE(planBuilder.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap()));
    EXPECT_THROW(planBuilder.buildNodePlan(graphWrap, graphWrap.getNode(0)),
                 CpuReferenceNotApplicableError);

    CpuReferenceGraphExecutor executor;
    EXPECT_FALSE(executor.isApplicable(serializedGraph.data(), serializedGraph.size()));
    EXPECT_THROW(executor.execute(serializedGraph.data(), serializedGraph.size(), variantPack),
                 CpuReferenceNotApplicableError);
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
    EXPECT_FALSE(planBuilder.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap()));
    EXPECT_THROW(planBuilder.buildNodePlan(graphWrap, graphWrap.getNode(0)),
                 CpuReferenceNotApplicableError);

    CpuReferenceGraphExecutor executor;
    EXPECT_FALSE(executor.isApplicable(serializedGraph.data(), serializedGraph.size()));
    EXPECT_THROW(executor.execute(serializedGraph.data(), serializedGraph.size(), variantPack),
                 CpuReferenceNotApplicableError);
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

    EXPECT_FALSE(planBuilder.isApplicable(graphWrapper.getNode(0), graphWrapper.getTensorMap()));
    EXPECT_THROW(planBuilder.buildNodePlan(graphWrapper, graphWrapper.getNode(0)),
                 CpuReferenceNotApplicableError);

    CpuReferenceGraphExecutor executor;
    EXPECT_FALSE(executor.isApplicable(serializedGraph.data(), serializedGraph.size()));
    EXPECT_THROW(executor.execute(serializedGraph.data(), serializedGraph.size(), variantPack),
                 CpuReferenceNotApplicableError);
}

TEST(TestPointwisePlanBuilder, DeclinesReluParamOnUnsupportedBinaryMode)
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
                                    TensorLayout::NCHW,
                                    0.0f);

    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    const GraphWrapper graphWrapper(serializedGraph.data(), serializedGraph.size());
    const PointwisePlanBuilder<DataType::FLOAT,
                               DataType::FLOAT,
                               DataType::FLOAT,
                               DataType::FLOAT,
                               DataType::FLOAT>
        planBuilder;

    EXPECT_FALSE(planBuilder.isApplicable(graphWrapper.getNode(0), graphWrapper.getTensorMap()));

    CpuReferenceGraphExecutor executor;
    EXPECT_FALSE(executor.isApplicable(serializedGraph.data(), serializedGraph.size()));
    EXPECT_THROW(executor.execute(serializedGraph.data(), serializedGraph.size(), variantPack),
                 CpuReferenceNotApplicableError);
}

TEST(TestPointwisePlanBuilder, DeclinesSwishBetaOnUnsupportedUnaryMode)
{
    auto [graph, tensorBundle, variantPack]
        = buildPointwiseUnaryGraph({1, 3, 4, 4},
                                   {1, 3, 4, 4},
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   hipdnn_frontend::PointwiseMode::ABS,
                                   1,
                                   TensorLayout::NCHW,
                                   std::nullopt,
                                   std::nullopt,
                                   std::nullopt,
                                   2.0f);

    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();

    CpuReferenceGraphExecutor executor;
    EXPECT_FALSE(executor.isApplicable(serializedGraph.data(), serializedGraph.size()));
    EXPECT_THROW(executor.execute(serializedGraph.data(), serializedGraph.size(), variantPack),
                 CpuReferenceNotApplicableError);
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

TEST(TestPointwisePlanBuilder, DeclinesEachReluAndSwishParamOnUnsupportedMode)
{
    // ADD does not use any relu clip or swish parameter. Each row sets one of the four
    // (lower clip, upper clip, lower clip slope, swish beta).
    const std::array<std::array<std::optional<float>, 4>, 4> paramSets
        = {{{1.0f, std::nullopt, std::nullopt, std::nullopt},
            {std::nullopt, 1.0f, std::nullopt, std::nullopt},
            {std::nullopt, std::nullopt, 1.0f, std::nullopt},
            {std::nullopt, std::nullopt, std::nullopt, 1.0f}}};

    for(const auto& params : paramSets)
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
                                        TensorLayout::NCHW,
                                        params[0],
                                        params[1],
                                        params[2],
                                        params[3]);

        auto [serializedGraph, serErr] = graph->to_binary();
        ASSERT_TRUE(serErr.is_good()) << serErr.get_message();

        CpuReferenceGraphExecutor executor;
        EXPECT_FALSE(executor.isApplicable(serializedGraph.data(), serializedGraph.size()));
        EXPECT_THROW(executor.execute(serializedGraph.data(), serializedGraph.size(), variantPack),
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
    EXPECT_EQ((runUnary<I, F, I>(PointwiseMode::ABS, a)), (std::vector<double>{5, 1, 3, 8}));
    EXPECT_EQ((runUnary<I, F, I>(PointwiseMode::NEG, a)), (std::vector<double>{5, 1, -3, -8}));
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
