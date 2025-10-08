// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include "BatchnormGraphUtils.hpp"
#include "BatchnormTensorBundles.hpp"
#include <hipdnn_sdk/data_objects/graph_generated.h>
#include <hipdnn_sdk/plugin/test_utils/MockGraph.hpp>
#include <hipdnn_sdk/test_utilities/CpuFpReferenceBatchnorm.hpp>
#include <hipdnn_sdk/test_utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_sdk/test_utilities/TestTolerances.hpp>
#include <hipdnn_sdk/test_utilities/cpu_graph_executor/BatchnormFwdInferencePlan.hpp>
#include <hipdnn_sdk/utilities/ShapeUtilities.hpp>

using namespace hipdnn_sdk::test_utilities;
using namespace hipdnn_sdk::data_objects;
using namespace hipdnn_sdk::utilities;
using namespace hipdnn_plugin;
using namespace ::testing;
using namespace hipdnn_sdk_test_utils;

class TestBatchnormFwdPlan : public ::testing::Test
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

TEST_F(TestBatchnormFwdPlan, ExecutePlan)
{
    auto tolerance = batchnorm::getToleranceInference<float>();
    double epsilon = 1e-3;
    std::vector<int64_t> dims = {6, 3, 32, 32};
    unsigned int seed = 1;
    BatchnormFwdTensorBundle<float, float, float> planTensorBundle(dims, seed, TensorLayout::NHWC);
    BatchnormFwdTensorBundle<float, float, float> directTensorBundle(
        dims, seed, TensorLayout::NHWC);

    BatchnormFwdInferenceParams params;
    initTensorValues(params.xTensor, DataType::FLOAT, planTensorBundle.xTensor, 1);
    initTensorValues(params.yTensor, DataType::FLOAT, planTensorBundle.yTensor, 2);
    initTensorValues(params.biasTensor, DataType::FLOAT, planTensorBundle.biasTensor, 3);
    initTensorValues(params.scaleTensor, DataType::FLOAT, planTensorBundle.scaleTensor, 4);
    initTensorValues(params.meanTensor, DataType::FLOAT, planTensorBundle.meanTensor, 5);
    initTensorValues(
        params.invVarianceTensor, DataType::FLOAT, planTensorBundle.invVarianceTensor, 6);
    params.epsilon = epsilon;

    BatchnormFwdPlan<float, float, float> patient(std::move(params));

    std::unordered_map<int64_t, void*> variantPack;
    variantPack[1] = planTensorBundle.xTensor.memory().hostData();
    variantPack[2] = planTensorBundle.yTensor.memory().hostData();
    variantPack[3] = planTensorBundle.biasTensor.memory().hostData();
    variantPack[4] = planTensorBundle.scaleTensor.memory().hostData();
    variantPack[5] = planTensorBundle.meanTensor.memory().hostData();
    variantPack[6] = planTensorBundle.invVarianceTensor.memory().hostData();

    CpuFpReferenceBatchnormImpl<float, float>::batchnormFwdInference(
        directTensorBundle.xTensor,
        directTensorBundle.scaleTensor,
        directTensorBundle.biasTensor,
        directTensorBundle.meanTensor,
        directTensorBundle.invVarianceTensor,
        directTensorBundle.yTensor,
        epsilon);

    patient.execute(variantPack);

    CpuFpReferenceValidation<float> cpuRefOutputValidation(tolerance, tolerance);

    EXPECT_TRUE(cpuRefOutputValidation.allClose(directTensorBundle.yTensor.memory(),
                                                planTensorBundle.yTensor.memory()));
}

TEST(TestBatchnormFwdInferencePlanBuilder, PlanConstruction)
{
    std::vector<int64_t> dims = {1, 1, 1, 1};
    BatchnormFwdTensorBundle<float, float, float> tensorBundle(dims, 1, TensorLayout::NCHW);

    auto graphTuple = buildBatchnormFwdInferenceGraph(
        tensorBundle, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT);

    auto& graph = std::get<0>(graphTuple);
    auto flatbufferGraph = graph->buildFlatbufferOperationGraph();

    auto graphWrap = hipdnn_plugin::GraphWrapper(flatbufferGraph.data(), flatbufferGraph.size());

    BatchnormFwdInferencePlanBuilder<DataType::FLOAT, DataType::FLOAT, DataType::FLOAT> patient;

    auto builtPlan = patient.buildNodePlan(graphWrap, graphWrap.getNode(0));

    bool result = dynamic_cast<BatchnormFwdPlan<float, float, float>*>(builtPlan.get()) != nullptr;
    EXPECT_TRUE(result);
}

TEST(TestBatchnormFwdInferencePlanBuilder, IsApplicable)
{
    std::vector<int64_t> dims = {1, 1, 1, 1};
    BatchnormFwdTensorBundle<float, float, float> tensorBundle(dims, 1, TensorLayout::NCHW);

    auto graphTuple = buildBatchnormFwdInferenceGraph(
        tensorBundle, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT);

    auto& graph = std::get<0>(graphTuple);
    auto flatbufferGraph = graph->buildFlatbufferOperationGraph();

    auto graphWrap = hipdnn_plugin::GraphWrapper(flatbufferGraph.data(), flatbufferGraph.size());

    BatchnormFwdInferencePlanBuilder<DataType::FLOAT, DataType::FLOAT, DataType::FLOAT>
        floatPlanBuilder;

    EXPECT_TRUE(floatPlanBuilder.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap()));

    BatchnormFwdInferencePlanBuilder<DataType::FLOAT, DataType::HALF, DataType::FLOAT>
        badTypesPlanBuilder;
    EXPECT_FALSE(badTypesPlanBuilder.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap()));

    auto tensorMapCopy = graphWrap.getTensorMap();
    tensorMapCopy.erase(6);
    EXPECT_FALSE(floatPlanBuilder.isApplicable(graphWrap.getNode(0), tensorMapCopy));
}
