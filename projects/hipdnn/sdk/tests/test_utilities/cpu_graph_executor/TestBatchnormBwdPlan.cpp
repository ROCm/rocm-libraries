// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include "BatchnormGraphUtils.hpp"
#include "BatchnormTensorBundles.hpp"
#include <hipdnn_sdk/data_objects/graph_generated.h>
#include <hipdnn_sdk/plugin/test_utils/MockGraph.hpp>
#include <hipdnn_sdk/test_utilities/CpuFpReferenceBatchnorm.hpp>
#include <hipdnn_sdk/test_utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_sdk/test_utilities/Seeds.hpp>
#include <hipdnn_sdk/test_utilities/TestTolerances.hpp>
#include <hipdnn_sdk/test_utilities/cpu_graph_executor/BatchnormBwdPlan.hpp>
#include <hipdnn_sdk/utilities/ShapeUtilities.hpp>

using namespace hipdnn_sdk::test_utilities;
using namespace hipdnn_sdk::data_objects;
using namespace hipdnn_sdk::utilities;
using namespace hipdnn_plugin;
using namespace ::testing;
using namespace hipdnn_sdk_test_utils;

class TestBatchnormBwdPlan : public ::testing::Test
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

TEST_F(TestBatchnormBwdPlan, ExecutePlan)
{
    std::vector<int64_t> dims = {6, 3, 32, 32};
    unsigned int seed = getGlobalTestSeed();
    BatchnormBwdTensorBundle<float, float, float> planTensorBundle(dims, seed, TensorLayout::NHWC);
    BatchnormBwdTensorBundle<float, float, float> directTensorBundle(
        dims, seed, TensorLayout::NHWC);

    BatchnormBwdParams params;
    initTensorValues(params.dyTensor, DataType::FLOAT, planTensorBundle.dyTensor, 1);
    initTensorValues(params.xTensor, DataType::FLOAT, planTensorBundle.xTensor, 2);
    initTensorValues(params.meanTensor, DataType::FLOAT, planTensorBundle.meanTensor, 3);
    initTensorValues(
        params.invVarianceTensor, DataType::FLOAT, planTensorBundle.invVarianceTensor, 4);
    initTensorValues(params.scaleTensor, DataType::FLOAT, planTensorBundle.scaleTensor, 5);
    initTensorValues(params.dxTensor, DataType::FLOAT, planTensorBundle.dxTensor, 6);
    initTensorValues(params.dscaleTensor, DataType::FLOAT, planTensorBundle.dscaleTensor, 7);
    initTensorValues(params.dbiasTensor, DataType::FLOAT, planTensorBundle.dbiasTensor, 8);

    BatchnormBwdPlan<float, float, float> bwdPlan(std::move(params));

    std::unordered_map<int64_t, void*> variantPack;
    variantPack[1] = planTensorBundle.dyTensor.memory().hostData();
    variantPack[2] = planTensorBundle.xTensor.memory().hostData();
    variantPack[3] = planTensorBundle.meanTensor.memory().hostData();
    variantPack[4] = planTensorBundle.invVarianceTensor.memory().hostData();
    variantPack[5] = planTensorBundle.scaleTensor.memory().hostData();
    variantPack[6] = planTensorBundle.dxTensor.memory().hostData();
    variantPack[7] = planTensorBundle.dscaleTensor.memory().hostData();
    variantPack[8] = planTensorBundle.dbiasTensor.memory().hostData();

    CpuFpReferenceBatchnormImpl<float, float, float>::batchnormBwd(
        directTensorBundle.dyTensor,
        directTensorBundle.xTensor,
        directTensorBundle.meanTensor,
        directTensorBundle.invVarianceTensor,
        directTensorBundle.scaleTensor,
        directTensorBundle.dxTensor,
        directTensorBundle.dscaleTensor,
        directTensorBundle.dbiasTensor);

    bwdPlan.execute(variantPack);

    CpuFpReferenceValidation<float> cpuRefOutputValidation(
        batchnorm::getToleranceBackward<float>(), batchnorm::getToleranceBackward<float>());

    EXPECT_TRUE(
        cpuRefOutputValidation.allClose(directTensorBundle.dxTensor, planTensorBundle.dxTensor));
    EXPECT_TRUE(cpuRefOutputValidation.allClose(directTensorBundle.dscaleTensor,
                                                planTensorBundle.dscaleTensor));
    EXPECT_TRUE(cpuRefOutputValidation.allClose(directTensorBundle.dbiasTensor,
                                                planTensorBundle.dbiasTensor));
}

TEST(TestBatchnormBwdPlanBuilder, PlanConstruction)
{
    std::vector<int64_t> dims = {1, 1, 1, 1};
    BatchnormBwdTensorBundle<float, float, float> tensorBundle(dims, 1, TensorLayout::NCHW);

    auto graphTuple
        = buildBatchnormBwdGraph(tensorBundle, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT);

    auto& graph = std::get<0>(graphTuple);
    auto flatbufferGraph = graph->buildFlatbufferOperationGraph();

    auto graphWrap = hipdnn_plugin::GraphWrapper(flatbufferGraph.data(), flatbufferGraph.size());

    BatchnormBwdPlanBuilder<DataType::FLOAT, DataType::FLOAT, DataType::FLOAT> bwdPlanBuilder;

    auto builtPlan = bwdPlanBuilder.buildNodePlan(graphWrap, graphWrap.getNode(0));

    bool result = dynamic_cast<BatchnormBwdPlan<float, float, float>*>(builtPlan.get()) != nullptr;
    EXPECT_TRUE(result);
}

TEST(TestBatchnormBwdPlanBuilder, IsApplicable)
{
    std::vector<int64_t> dims = {1, 1, 1, 1};
    BatchnormBwdTensorBundle<float, float, float> tensorBundle(dims, 1, TensorLayout::NCHW);

    auto graphTuple
        = buildBatchnormBwdGraph(tensorBundle, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT);

    auto& graph = std::get<0>(graphTuple);
    auto flatbufferGraph = graph->buildFlatbufferOperationGraph();

    auto graphWrap = hipdnn_plugin::GraphWrapper(flatbufferGraph.data(), flatbufferGraph.size());

    BatchnormBwdPlanBuilder<DataType::FLOAT, DataType::FLOAT, DataType::FLOAT> floatPlanBuilder;

    EXPECT_TRUE(floatPlanBuilder.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap()));

    BatchnormBwdPlanBuilder<DataType::FLOAT, DataType::HALF, DataType::FLOAT> badTypesPlanBuilder;
    EXPECT_FALSE(badTypesPlanBuilder.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap()));

    auto tensorMapCopy = graphWrap.getTensorMap();
    tensorMapCopy.erase(4);
    EXPECT_FALSE(floatPlanBuilder.isApplicable(graphWrap.getNode(0), tensorMapCopy));
}
