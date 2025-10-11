// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include "PointwiseGraphUtils.hpp"
#include "PointwiseTensorBundles.hpp"
#include <hipdnn_sdk/data_objects/graph_generated.h>
#include <hipdnn_sdk/plugin/test_utils/MockGraph.hpp>
#include <hipdnn_sdk/test_utilities/CpuFpReferenceValidation.hpp>
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
protected:
    static void initTensorValues(hipdnn_sdk::data_objects::TensorAttributesT& tensorAttr,
                                 DataType dataType,
                                 const Tensor<float>& tensor,
                                 int64_t uid)
    {
        tensorAttr.data_type = dataType;
        tensorAttr.dims = tensor.dims();
        tensorAttr.strides = tensor.strides();
        tensorAttr.uid = uid;
    }
};

TEST_F(TestPointwisePlan, ExecutePlanUnaryReluFwd)
{
    double epsilon = 1e-5;
    std::vector<int64_t> inputDims = {1, 3, 4, 4};
    std::vector<int64_t> outputDims = {1, 3, 4, 4};

    unsigned int seed = 1;
    PointwiseUnaryTensorBundle<float> planTensorBundle(
        inputDims, outputDims, seed, TensorLayout::NCHW);
    PointwiseUnaryTensorBundle<float> directTensorBundle(
        inputDims, outputDims, seed, TensorLayout::NCHW);

    PointwiseParams params;
    params.mode = PointwiseMode::RELU_FWD;
    initTensorValues(params.in0Tensor, DataType::FLOAT, planTensorBundle.inputTensor, 1);
    initTensorValues(params.out0Tensor, DataType::FLOAT, planTensorBundle.outputTensor, 2);

    PointwisePlan<float> patient(std::move(params));

    std::unordered_map<int64_t, void*> variantPack;
    variantPack[1] = planTensorBundle.inputTensor.memory().hostData();
    variantPack[2] = planTensorBundle.outputTensor.memory().hostData();

    // Execute direct reference
    CpuReferencePointwiseImpl<float>::pointwiseCompute(
        PointwiseMode::RELU_FWD, directTensorBundle.outputTensor, directTensorBundle.inputTensor);

    // Execute plan
    patient.execute(variantPack);

    CpuFpReferenceValidation<float> cpuRefOutputValidation(static_cast<float>(epsilon),
                                                           static_cast<float>(epsilon));

    EXPECT_TRUE(cpuRefOutputValidation.allClose(directTensorBundle.outputTensor.memory(),
                                                planTensorBundle.outputTensor.memory()));
}

TEST_F(TestPointwisePlan, ExecutePlanBinaryAdd)
{
    double epsilon = 1e-5;
    std::vector<int64_t> input1Dims = {1, 3, 2, 2};
    std::vector<int64_t> input2Dims = {1, 3, 2, 2};
    std::vector<int64_t> outputDims = {1, 3, 2, 2};

    unsigned int seed = 1;
    PointwiseBinaryTensorBundle<float> planTensorBundle(
        input1Dims, input2Dims, outputDims, seed, TensorLayout::NCHW);
    PointwiseBinaryTensorBundle<float> directTensorBundle(
        input1Dims, input2Dims, outputDims, seed, TensorLayout::NCHW);

    PointwiseParams params;
    params.mode = PointwiseMode::ADD;
    initTensorValues(params.in0Tensor, DataType::FLOAT, planTensorBundle.input1Tensor, 1);
    initTensorValues(params.in1Tensor.emplace(), DataType::FLOAT, planTensorBundle.input2Tensor, 2);
    initTensorValues(params.out0Tensor, DataType::FLOAT, planTensorBundle.outputTensor, 3);

    PointwisePlan<float> patient(std::move(params));

    std::unordered_map<int64_t, void*> variantPack;
    variantPack[1] = planTensorBundle.input1Tensor.memory().hostData();
    variantPack[2] = planTensorBundle.input2Tensor.memory().hostData();
    variantPack[3] = planTensorBundle.outputTensor.memory().hostData();

    // Execute direct reference
    CpuReferencePointwiseImpl<float>::pointwiseCompute(PointwiseMode::ADD,
                                                       directTensorBundle.outputTensor,
                                                       directTensorBundle.input1Tensor,
                                                       directTensorBundle.input2Tensor);

    // Execute plan
    patient.execute(variantPack);

    CpuFpReferenceValidation<float> cpuRefOutputValidation(static_cast<float>(epsilon),
                                                           static_cast<float>(epsilon));

    EXPECT_TRUE(cpuRefOutputValidation.allClose(directTensorBundle.outputTensor.memory(),
                                                planTensorBundle.outputTensor.memory()));
}

TEST_F(TestPointwisePlan, ExecutePlanBackwardReluBwd)
{
    double epsilon = 1e-5;
    std::vector<int64_t> dyDims = {1, 3, 2, 2};
    std::vector<int64_t> xDims = {1, 3, 2, 2};
    std::vector<int64_t> dxDims = {1, 3, 2, 2};

    unsigned int seed = 1;
    // Use PointwiseBinaryTensorBundle for backward operations since they are binary (dy, x -> dx)
    PointwiseBinaryTensorBundle<float> planTensorBundle(
        dyDims, xDims, dxDims, seed, TensorLayout::NCHW);
    PointwiseBinaryTensorBundle<float> directTensorBundle(
        dyDims, xDims, dxDims, seed, TensorLayout::NCHW);

    PointwiseParams params;
    params.mode = PointwiseMode::RELU_BWD;
    initTensorValues(params.in0Tensor, DataType::FLOAT, planTensorBundle.input1Tensor, 1);
    initTensorValues(params.in1Tensor.emplace(), DataType::FLOAT, planTensorBundle.input2Tensor, 2);
    initTensorValues(params.out0Tensor, DataType::FLOAT, planTensorBundle.outputTensor, 3);

    PointwisePlan<float> patient(std::move(params));

    std::unordered_map<int64_t, void*> variantPack;
    variantPack[1] = planTensorBundle.input1Tensor.memory().hostData();
    variantPack[2] = planTensorBundle.input2Tensor.memory().hostData();
    variantPack[3] = planTensorBundle.outputTensor.memory().hostData();

    // Execute direct reference
    CpuReferencePointwiseImpl<float>::pointwiseCompute(PointwiseMode::RELU_BWD,
                                                       directTensorBundle.outputTensor,
                                                       directTensorBundle.input1Tensor,
                                                       directTensorBundle.input2Tensor);

    // Execute plan
    patient.execute(variantPack);

    CpuFpReferenceValidation<float> cpuRefOutputValidation(static_cast<float>(epsilon),
                                                           static_cast<float>(epsilon));

    EXPECT_TRUE(cpuRefOutputValidation.allClose(directTensorBundle.outputTensor.memory(),
                                                planTensorBundle.outputTensor.memory()));
}

TEST(TestPointwisePlanBuilder, PlanConstructionUnary)
{
    std::vector<int64_t> inputDims = {1, 3, 4, 4};
    std::vector<int64_t> outputDims = {1, 3, 4, 4};

    PointwiseUnaryTensorBundle<float> tensorBundle(inputDims, outputDims, 1, TensorLayout::NCHW);

    auto graphTuple = buildPointwiseUnaryGraph(tensorBundle,
                                               DataType::FLOAT,
                                               DataType::FLOAT,
                                               DataType::FLOAT,
                                               hipdnn_frontend::PointwiseMode::RELU_FWD);

    auto& graph = std::get<0>(graphTuple);
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

    PointwiseBinaryTensorBundle<float> tensorBundle(
        input1Dims, input2Dims, outputDims, 1, TensorLayout::NCHW);

    auto graphTuple = buildPointwiseBinaryGraph(tensorBundle,
                                                DataType::FLOAT,
                                                DataType::FLOAT,
                                                DataType::FLOAT,
                                                DataType::FLOAT,
                                                hipdnn_frontend::PointwiseMode::ADD);

    auto& graph = std::get<0>(graphTuple);
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

    PointwiseUnaryTensorBundle<float> tensorBundle(inputDims, outputDims, 1, TensorLayout::NCHW);

    auto graphTuple = buildPointwiseUnaryGraph(tensorBundle,
                                               DataType::FLOAT,
                                               DataType::FLOAT,
                                               DataType::FLOAT,
                                               hipdnn_frontend::PointwiseMode::RELU_FWD);

    auto& graph = std::get<0>(graphTuple);
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

    PointwiseBinaryTensorBundle<float> tensorBundle(
        input1Dims, input2Dims, outputDims, 1, TensorLayout::NCHW);

    auto graphTuple = buildPointwiseBinaryGraph(tensorBundle,
                                                DataType::FLOAT,
                                                DataType::FLOAT,
                                                DataType::FLOAT,
                                                DataType::FLOAT,
                                                hipdnn_frontend::PointwiseMode::ADD);

    auto& graph = std::get<0>(graphTuple);
    auto flatbufferGraph = graph->buildFlatbufferOperationGraph();

    auto graphWrap = hipdnn_plugin::GraphWrapper(flatbufferGraph.data(), flatbufferGraph.size());

    PointwisePlanBuilder<DataType::FLOAT> floatPlanBuilder;

    EXPECT_TRUE(floatPlanBuilder.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap()));

    // Test with missing tensor
    auto tensorMapCopy = graphWrap.getTensorMap();
    tensorMapCopy.erase(2);
    EXPECT_FALSE(floatPlanBuilder.isApplicable(graphWrap.getNode(0), tensorMapCopy));
}

TEST(TestPointwisePlanBuilder, UnsupportedOperation)
{
    std::vector<int64_t> inputDims = {1, 3, 4, 4};
    std::vector<int64_t> outputDims = {1, 3, 4, 4};

    PointwiseUnaryTensorBundle<float> tensorBundle(inputDims, outputDims, 1, TensorLayout::NCHW);

    // Try with an unsupported operation (not in our getSupportedUnaryOperations list)
    auto graphTuple = buildPointwiseUnaryGraph(tensorBundle,
                                               DataType::FLOAT,
                                               DataType::FLOAT,
                                               DataType::FLOAT,
                                               hipdnn_frontend::PointwiseMode::EXP);

    auto& graph = std::get<0>(graphTuple);
    auto flatbufferGraph = graph->buildFlatbufferOperationGraph();

    auto graphWrap = hipdnn_plugin::GraphWrapper(flatbufferGraph.data(), flatbufferGraph.size());

    PointwisePlanBuilder<DataType::FLOAT> planBuilder;

    EXPECT_FALSE(planBuilder.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap()));
}
