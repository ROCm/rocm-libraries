// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include "BatchnormGraphUtils.hpp"
#include "BatchnormTensorBundles.hpp"
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceBatchnorm.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/DynamicTolerancesBatchNorm.hpp>
#include <hipdnn_test_sdk/utilities/Seeds.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/BatchnormBwdPlan.hpp>

using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_test_sdk::detail;
using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_flatbuffers_sdk::flatbuffer_utilities;
using namespace ::testing;
using namespace hipdnn_sdk_test_utils;

class TestBatchnormBwdPlan : public ::testing::Test
{
protected:
    static void
        initTensorValues(hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT& tensorAttr,
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

TEST_F(TestBatchnormBwdPlan, ExecutePlan)
{
    const std::vector<int64_t> dims = {6, 3, 32, 32};
    const unsigned int seed = getGlobalTestSeed();
    BatchnormBwdTensorBundle<float, float, float> planTensorBundle(dims, seed, TensorLayout::NHWC);
    BatchnormBwdTensorBundle<float, float, float> directTensorBundle(
        dims, seed, TensorLayout::NHWC);

    BatchnormBwdParams params;
    initTensorValues(params.dyTensor, DataType::FLOAT, planTensorBundle.dyTensor, 1);
    initTensorValues(params.xTensor, DataType::FLOAT, planTensorBundle.xTensor, 2);
    params.meanTensor.emplace();
    initTensorValues(*params.meanTensor, DataType::FLOAT, planTensorBundle.meanTensor, 3);
    params.invVarianceTensor.emplace();
    initTensorValues(
        *params.invVarianceTensor, DataType::FLOAT, planTensorBundle.invVarianceTensor, 4);
    initTensorValues(params.scaleTensor, DataType::FLOAT, planTensorBundle.scaleTensor, 5);
    initTensorValues(params.dxTensor, DataType::FLOAT, planTensorBundle.dxTensor, 6);
    initTensorValues(params.dscaleTensor, DataType::FLOAT, planTensorBundle.dscaleTensor, 7);
    initTensorValues(params.dbiasTensor, DataType::FLOAT, planTensorBundle.dbiasTensor, 8);

    BatchnormBwdPlan<float, float, float, float, float, float> bwdPlan(std::move(params));

    std::unordered_map<int64_t, void*> variantPack;
    variantPack[1] = planTensorBundle.dyTensor.memory().hostData();
    variantPack[2] = planTensorBundle.xTensor.memory().hostData();
    variantPack[3] = planTensorBundle.meanTensor.memory().hostData();
    variantPack[4] = planTensorBundle.invVarianceTensor.memory().hostData();
    variantPack[5] = planTensorBundle.scaleTensor.memory().hostData();
    variantPack[6] = planTensorBundle.dxTensor.memory().hostData();
    variantPack[7] = planTensorBundle.dscaleTensor.memory().hostData();
    variantPack[8] = planTensorBundle.dbiasTensor.memory().hostData();

    CpuFpReferenceBatchnorm::backward(directTensorBundle.dyTensor,
                                      directTensorBundle.xTensor,
                                      directTensorBundle.scaleTensor,
                                      directTensorBundle.dxTensor,
                                      directTensorBundle.dscaleTensor,
                                      directTensorBundle.dbiasTensor,
                                      &directTensorBundle.meanTensor,
                                      &directTensorBundle.invVarianceTensor);

    bwdPlan.execute(variantPack);

    // Known ranges from BatchnormBwdTensorBundle constructor:
    //   dy: [-0.1, 0.1], x: [-1.0, 1.0], scale: [-0.1, 0.1]
    const auto nhw = dims[0] * dims[2] * dims[3];

    auto dbiasTol
        = batchnorm::calculateBatchnormBackwardDbiasTolerance<float, float>(-0.1, 0.1, nhw);
    auto dscaleTol = batchnorm::calculateBatchnormBackwardDscaleTolerance<float, float>(
        -0.1, 0.1, -1.0, 1.0, nhw);
    auto dxTol = batchnorm::calculateBatchnormBackwardDxTolerance<float, float>(
        -0.1, 0.1, -1.0, 1.0, -0.1, 0.1, nhw);

    const CpuFpReferenceValidation<float> dxValidation(dxTol, dxTol);
    const CpuFpReferenceValidation<float> dscaleValidation(dscaleTol, dscaleTol);
    const CpuFpReferenceValidation<float> dbiasValidation(dbiasTol, dbiasTol);

    EXPECT_TRUE(dxValidation.allClose(directTensorBundle.dxTensor, planTensorBundle.dxTensor));
    EXPECT_TRUE(
        dscaleValidation.allClose(directTensorBundle.dscaleTensor, planTensorBundle.dscaleTensor));
    EXPECT_TRUE(
        dbiasValidation.allClose(directTensorBundle.dbiasTensor, planTensorBundle.dbiasTensor));
}

TEST(TestBatchnormBwdPlanBuilder, PlanConstruction)
{
    const std::vector<int64_t> dims = {2, 1, 1, 1};
    BatchnormBwdTensorBundle<float, float, float> tensorBundle(dims, 1, TensorLayout::NCHW);

    auto graphTuple = buildBatchnormBwdGraph(
        tensorBundle, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT);

    auto& graph = std::get<0>(graphTuple);
    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();

    auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        serializedGraph.data(), serializedGraph.size());

    const BatchnormBwdPlanBuilder<DataType::FLOAT,
                                  DataType::FLOAT,
                                  DataType::FLOAT,
                                  DataType::FLOAT,
                                  DataType::FLOAT,
                                  DataType::FLOAT>
        bwdPlanBuilder;

    auto builtPlan = bwdPlanBuilder.buildNodePlan(graphWrap, graphWrap.getNode(0));

    const bool result
        = dynamic_cast<BatchnormBwdPlan<float, float, float, float, float, float>*>(builtPlan.get())
          != nullptr;
    EXPECT_TRUE(result);
}

TEST_F(TestBatchnormBwdPlan, ExecutePlanNoStats)
{
    const std::vector<int64_t> dims = {6, 3, 32, 32};
    const unsigned int seed = getGlobalTestSeed();
    BatchnormBwdTensorBundle<float, float, float> planTensorBundle(dims, seed, TensorLayout::NHWC);
    BatchnormBwdTensorBundle<float, float, float> directTensorBundle(
        dims, seed, TensorLayout::NHWC);

    BatchnormBwdParams params;
    initTensorValues(params.dyTensor, DataType::FLOAT, planTensorBundle.dyTensor, 1);
    initTensorValues(params.xTensor, DataType::FLOAT, planTensorBundle.xTensor, 2);
    // Do NOT set meanTensor or invVarianceTensor - testing NoStats case
    initTensorValues(params.scaleTensor, DataType::FLOAT, planTensorBundle.scaleTensor, 5);
    initTensorValues(params.dxTensor, DataType::FLOAT, planTensorBundle.dxTensor, 6);
    initTensorValues(params.dscaleTensor, DataType::FLOAT, planTensorBundle.dscaleTensor, 7);
    initTensorValues(params.dbiasTensor, DataType::FLOAT, planTensorBundle.dbiasTensor, 8);

    BatchnormBwdPlan<float, float, float, float, float, float> bwdPlan(std::move(params));

    std::unordered_map<int64_t, void*> variantPack;
    variantPack[1] = planTensorBundle.dyTensor.memory().hostData();
    variantPack[2] = planTensorBundle.xTensor.memory().hostData();
    variantPack[5] = planTensorBundle.scaleTensor.memory().hostData();
    variantPack[6] = planTensorBundle.dxTensor.memory().hostData();
    variantPack[7] = planTensorBundle.dscaleTensor.memory().hostData();
    variantPack[8] = planTensorBundle.dbiasTensor.memory().hostData();

    CpuFpReferenceBatchnorm::backward(directTensorBundle.dyTensor,
                                      directTensorBundle.xTensor,
                                      directTensorBundle.scaleTensor,
                                      directTensorBundle.dxTensor,
                                      directTensorBundle.dscaleTensor,
                                      directTensorBundle.dbiasTensor);

    bwdPlan.execute(variantPack);

    // Known ranges from BatchnormBwdTensorBundle constructor:
    //   dy: [-0.1, 0.1], x: [-1.0, 1.0], scale: [-0.1, 0.1]
    const auto nhw = dims[0] * dims[2] * dims[3];

    auto dbiasTol
        = batchnorm::calculateBatchnormBackwardDbiasTolerance<float, float>(-0.1, 0.1, nhw);
    auto dscaleTol = batchnorm::calculateBatchnormBackwardDscaleTolerance<float, float>(
        -0.1, 0.1, -1.0, 1.0, nhw);
    auto dxTol = batchnorm::calculateBatchnormBackwardDxTolerance<float, float>(
        -0.1, 0.1, -1.0, 1.0, -0.1, 0.1, nhw);

    const CpuFpReferenceValidation<float> dxValidation(dxTol, dxTol);
    const CpuFpReferenceValidation<float> dscaleValidation(dscaleTol, dscaleTol);
    const CpuFpReferenceValidation<float> dbiasValidation(dbiasTol, dbiasTol);

    EXPECT_TRUE(dxValidation.allClose(directTensorBundle.dxTensor, planTensorBundle.dxTensor));
    EXPECT_TRUE(
        dscaleValidation.allClose(directTensorBundle.dscaleTensor, planTensorBundle.dscaleTensor));
    EXPECT_TRUE(
        dbiasValidation.allClose(directTensorBundle.dbiasTensor, planTensorBundle.dbiasTensor));
}

TEST(TestBatchnormBwdPlanBuilder, IsApplicable)
{
    const std::vector<int64_t> dims = {2, 1, 1, 1};
    BatchnormBwdTensorBundle<float, float, float> tensorBundle(dims, 1, TensorLayout::NCHW);

    auto graphTuple = buildBatchnormBwdGraph(
        tensorBundle, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT);

    auto& graph = std::get<0>(graphTuple);
    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();

    auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        serializedGraph.data(), serializedGraph.size());

    const BatchnormBwdPlanBuilder<DataType::FLOAT,
                                  DataType::FLOAT,
                                  DataType::FLOAT,
                                  DataType::FLOAT,
                                  DataType::FLOAT,
                                  DataType::FLOAT>
        floatPlanBuilder;

    EXPECT_TRUE(floatPlanBuilder.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap()));

    const BatchnormBwdPlanBuilder<DataType::FLOAT,
                                  DataType::HALF,
                                  DataType::FLOAT,
                                  DataType::FLOAT,
                                  DataType::FLOAT,
                                  DataType::FLOAT>
        badTypesPlanBuilder;
    EXPECT_FALSE(badTypesPlanBuilder.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap()));

    auto tensorMapCopy = graphWrap.getTensorMap();
    tensorMapCopy.erase(4);
    EXPECT_FALSE(floatPlanBuilder.isApplicable(graphWrap.getNode(0), tensorMapCopy));
}
