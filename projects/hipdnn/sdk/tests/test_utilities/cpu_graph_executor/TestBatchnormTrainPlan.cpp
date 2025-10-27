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
#include <hipdnn_sdk/test_utilities/cpu_graph_executor/BatchnormTrainPlan.hpp>
#include <hipdnn_sdk/utilities/Constants.hpp>
#include <hipdnn_sdk/utilities/ShapeUtilities.hpp>

using namespace hipdnn_sdk::test_utilities;
using namespace hipdnn_sdk::data_objects;
using namespace hipdnn_sdk::utilities;
using namespace hipdnn_plugin;
using namespace ::testing;
using namespace hipdnn_sdk_test_utils;

class TestBatchnormTrainPlan : public ::testing::Test
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

TEST_F(TestBatchnormTrainPlan, ExecutePlan)
{
    double epsilon = BATCHNORM_DEFAULT_EPSILON;
    double momentum = 0.1;
    std::vector<int64_t> dims = {6, 3, 32, 32};
    unsigned int seed = getGlobalTestSeed();
    BatchnormTrainTensorBundle<float, float, float> planTensorBundle(
        dims, seed, TensorLayout::NHWC);
    BatchnormTrainTensorBundle<float, float, float> directTensorBundle(
        dims, seed, TensorLayout::NHWC);

    BatchnormTrainParams<float> params;
    initTensorValues(params.xTensor, DataType::FLOAT, planTensorBundle.xTensor, 1);
    initTensorValues(params.scaleTensor, DataType::FLOAT, planTensorBundle.scaleTensor, 2);
    initTensorValues(params.biasTensor, DataType::FLOAT, planTensorBundle.biasTensor, 3);
    initTensorValues(params.yTensor, DataType::FLOAT, planTensorBundle.yTensor, 4);
    initTensorValues(params.meanTensor, DataType::FLOAT, planTensorBundle.meanTensor, 5);
    initTensorValues(
        params.invVarianceTensor, DataType::FLOAT, planTensorBundle.invVarianceTensor, 6);
    initTensorValues(params.epsilonTensor, DataType::FLOAT, planTensorBundle.epsilonTensor, 7);

    BatchnormTrainPlan<float, float, float> patient(std::move(params));

    std::unordered_map<int64_t, void*> variantPack;
    variantPack[1] = planTensorBundle.xTensor.memory().hostData();
    variantPack[2] = planTensorBundle.scaleTensor.memory().hostData();
    variantPack[3] = planTensorBundle.biasTensor.memory().hostData();
    variantPack[4] = planTensorBundle.yTensor.memory().hostData();
    variantPack[5] = planTensorBundle.meanTensor.memory().hostData();
    variantPack[6] = planTensorBundle.invVarianceTensor.memory().hostData();
    variantPack[7] = planTensorBundle.epsilonTensor.memory().hostData();

    CpuFpReferenceBatchnormImpl<float, float, float>::batchnormFwdTraining(
        directTensorBundle.xTensor,
        directTensorBundle.scaleTensor,
        directTensorBundle.biasTensor,
        directTensorBundle.yTensor,
        static_cast<float>(epsilon),
        static_cast<float>(momentum),
        &directTensorBundle.meanTensor,
        &directTensorBundle.invVarianceTensor,
        nullptr,
        nullptr,
        nullptr,
        nullptr);

    patient.execute(variantPack);

    CpuFpReferenceValidation<float> cpuRefOutputValidation(static_cast<float>(epsilon),
                                                           static_cast<float>(epsilon));

    EXPECT_TRUE(
        cpuRefOutputValidation.allClose(directTensorBundle.yTensor, planTensorBundle.yTensor));
    EXPECT_TRUE(cpuRefOutputValidation.allClose(directTensorBundle.meanTensor,
                                                planTensorBundle.meanTensor));
    EXPECT_TRUE(cpuRefOutputValidation.allClose(directTensorBundle.invVarianceTensor,
                                                planTensorBundle.invVarianceTensor));
}

TEST(TestBatchnormTrainPlanBuilder, PlanConstruction)
{
    std::vector<int64_t> dims = {1, 1, 1, 1};
    BatchnormTrainTensorBundle<float, float, float> tensorBundle(dims, 1, TensorLayout::NCHW);

    auto graphTuple = buildBatchnormTrainGraph(
        tensorBundle, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, false);

    auto& graph = std::get<0>(graphTuple);
    auto flatbufferGraph = graph->buildFlatbufferOperationGraph();

    auto graphWrap = hipdnn_plugin::GraphWrapper(flatbufferGraph.data(), flatbufferGraph.size());

    BatchnormTrainPlanBuilder<DataType::FLOAT, DataType::FLOAT, DataType::FLOAT> patient;

    auto builtPlan = patient.buildNodePlan(graphWrap, graphWrap.getNode(0));

    bool result
        = dynamic_cast<BatchnormTrainPlan<float, float, float>*>(builtPlan.get()) != nullptr;
    EXPECT_TRUE(result);
}

TEST(TestBatchnormTrainPlanBuilder, IsApplicable)
{
    std::vector<int64_t> dims = {1, 1, 1, 1};
    BatchnormTrainTensorBundle<float, float, float> tensorBundle(dims, 1, TensorLayout::NCHW);

    auto graphTuple = buildBatchnormTrainGraph(
        tensorBundle, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, false);

    auto& graph = std::get<0>(graphTuple);
    auto flatbufferGraph = graph->buildFlatbufferOperationGraph();

    auto graphWrap = hipdnn_plugin::GraphWrapper(flatbufferGraph.data(), flatbufferGraph.size());

    BatchnormTrainPlanBuilder<DataType::FLOAT, DataType::FLOAT, DataType::FLOAT> floatPlanBuilder;

    EXPECT_TRUE(floatPlanBuilder.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap()));

    BatchnormTrainPlanBuilder<DataType::FLOAT, DataType::HALF, DataType::FLOAT> badTypesPlanBuilder;
    EXPECT_FALSE(badTypesPlanBuilder.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap()));

    auto tensorMapCopy = graphWrap.getTensorMap();
    tensorMapCopy.erase(6);
    EXPECT_FALSE(floatPlanBuilder.isApplicable(graphWrap.getNode(0), tensorMapCopy));
}
