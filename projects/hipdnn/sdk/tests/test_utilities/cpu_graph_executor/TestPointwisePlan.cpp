// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include "PointwiseGraphUtils.hpp"
#include "PointwiseTensorBundles.hpp"
#include <hipdnn_sdk/data_objects/graph_generated.h>
#include <hipdnn_sdk/plugin/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_sdk/plugin/test_utils/MockGraph.hpp>
#include <hipdnn_sdk/test_utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_sdk/test_utilities/Seeds.hpp>
#include <hipdnn_sdk/test_utilities/cpu_graph_executor/CpuReferenceGraphExecutor.hpp>
#include <hipdnn_sdk/test_utilities/cpu_graph_executor/PointwisePlan.hpp>
#include <hipdnn_sdk/test_utilities/pointwise/CpuReferencePointwise.hpp>
#include <hipdnn_sdk/utilities/ShapeUtilities.hpp>

using namespace hipdnn_sdk::test_utilities;
using namespace hipdnn_sdk::data_objects;
using namespace hipdnn_sdk::utilities;
using namespace hipdnn_plugin;
using namespace ::testing;
using namespace hipdnn_sdk_test_utils;

class TestPointwisePlan : public ::testing::Test
{
};

TEST_F(TestPointwisePlan, ExecutePlanUnaryReluFwd)
{
    std::vector<int64_t> inputDims = {1, 3, 4, 4};
    std::vector<int64_t> outputDims = {1, 3, 4, 4};
    unsigned int seed = getGlobalTestSeed();

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
    CpuReferenceGraphExecutor graphExecutor;
    auto serializedGraph = graph->buildFlatbufferOperationGraph();
    graphExecutor.execute(serializedGraph.data(), serializedGraph.size(), variantPack);

    // Verify output is correct (non-negative since it's RELU)
    // For this test we just verify execution succeeded without throwing
    SUCCEED();
}

TEST_F(TestPointwisePlan, ExecutePlanBinaryAdd)
{
    std::vector<int64_t> input1Dims = {1, 3, 2, 2};
    std::vector<int64_t> input2Dims = {1, 3, 2, 2};
    std::vector<int64_t> outputDims = {1, 3, 2, 2};
    unsigned int seed = getGlobalTestSeed();

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
    CpuReferenceGraphExecutor graphExecutor;
    auto serializedGraph = graph->buildFlatbufferOperationGraph();
    graphExecutor.execute(serializedGraph.data(), serializedGraph.size(), variantPack);

    // Verify execution succeeded
    SUCCEED();
}

TEST_F(TestPointwisePlan, ExecutePlanBackwardReluBwd)
{
    std::vector<int64_t> dyDims = {1, 3, 2, 2};
    std::vector<int64_t> xDims = {1, 3, 2, 2};
    std::vector<int64_t> dxDims = {1, 3, 2, 2};
    unsigned int seed = getGlobalTestSeed();

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
    CpuReferenceGraphExecutor graphExecutor;
    auto serializedGraph = graph->buildFlatbufferOperationGraph();
    graphExecutor.execute(serializedGraph.data(), serializedGraph.size(), variantPack);

    // Verify execution succeeded
    SUCCEED();
}

TEST(TestPointwisePlanBuilder, PlanConstructionUnary)
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
    auto graphWrap = hipdnn_plugin::GraphWrapper(flatbufferGraph.data(), flatbufferGraph.size());

    PointwisePlanBuilder<DataType::FLOAT> patient;
    auto builtPlan = patient.buildNodePlan(graphWrap, graphWrap.getNode(0));

    bool result = dynamic_cast<PointwisePlan<float>*>(builtPlan.get()) != nullptr;
    EXPECT_TRUE(result);
}

TEST(TestPointwisePlanBuilder, PlanConstructionBinary)
{
    std::vector<int64_t> input1Dims = {1, 3, 2, 2};
    std::vector<int64_t> input2Dims = {1, 3, 2, 2};
    std::vector<int64_t> outputDims = {1, 3, 2, 2};

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

    auto flatbufferGraph = graph->buildFlatbufferOperationGraph();
    auto graphWrap = hipdnn_plugin::GraphWrapper(flatbufferGraph.data(), flatbufferGraph.size());

    PointwisePlanBuilder<DataType::FLOAT> patient;
    auto builtPlan = patient.buildNodePlan(graphWrap, graphWrap.getNode(0));

    bool result = dynamic_cast<PointwisePlan<float>*>(builtPlan.get()) != nullptr;
    EXPECT_TRUE(result);
}

TEST(TestPointwisePlanBuilder, IsApplicableUnary)
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
    auto graphWrap = hipdnn_plugin::GraphWrapper(flatbufferGraph.data(), flatbufferGraph.size());

    PointwisePlanBuilder<DataType::FLOAT> floatPlanBuilder;
    EXPECT_TRUE(floatPlanBuilder.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap()));

    // Test with mismatched data types
    PointwisePlanBuilder<DataType::HALF> badTypesPlanBuilder;
    EXPECT_FALSE(badTypesPlanBuilder.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap()));
}

TEST(TestPointwisePlanBuilder, IsApplicableBinary)
{
    std::vector<int64_t> input1Dims = {1, 3, 2, 2};
    std::vector<int64_t> input2Dims = {1, 3, 2, 2};
    std::vector<int64_t> outputDims = {1, 3, 2, 2};

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

    auto flatbufferGraph = graph->buildFlatbufferOperationGraph();
    auto graphWrap = hipdnn_plugin::GraphWrapper(flatbufferGraph.data(), flatbufferGraph.size());

    PointwisePlanBuilder<DataType::FLOAT> floatPlanBuilder;
    EXPECT_TRUE(floatPlanBuilder.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap()));

    // Test with missing tensor - erase a tensor from the map
    auto tensorMapCopy = graphWrap.getTensorMap();
    tensorMapCopy.erase(2);
    EXPECT_FALSE(floatPlanBuilder.isApplicable(graphWrap.getNode(0), tensorMapCopy));
}

TEST(TestPointwisePlanBuilder, UnsupportedOperation)
{
    std::vector<int64_t> inputDims = {1, 3, 4, 4};
    std::vector<int64_t> outputDims = {1, 3, 4, 4};

    auto [graph, tensorBundle, variantPack]
        = buildPointwiseUnaryGraph(inputDims,
                                   outputDims,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   hipdnn_frontend::PointwiseMode::EXP, // Not implemented
                                   1,
                                   TensorLayout::NCHW);

    auto flatbufferGraph = graph->buildFlatbufferOperationGraph();
    auto graphWrap = hipdnn_plugin::GraphWrapper(flatbufferGraph.data(), flatbufferGraph.size());

    PointwisePlanBuilder<DataType::FLOAT> planBuilder;
    EXPECT_FALSE(planBuilder.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap()));
}
