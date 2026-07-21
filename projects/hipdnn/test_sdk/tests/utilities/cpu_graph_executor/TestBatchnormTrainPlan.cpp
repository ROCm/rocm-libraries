// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include "BatchnormGraphUtils.hpp"
#include "BatchnormTensorBundles.hpp"
#include <hipdnn_data_sdk/utilities/Constants.hpp>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceBatchnorm.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/Seeds.hpp>
#include <hipdnn_test_sdk/utilities/TestTolerances.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/BatchnormTrainPlan.hpp>

using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_test_sdk::detail;
using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_flatbuffers_sdk::flatbuffer_utilities;
using namespace ::testing;
using namespace hipdnn_sdk_test_utils;

class TestBatchnormTrainPlan : public ::testing::Test
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

TEST_F(TestBatchnormTrainPlan, ExecutePlan)
{
    const double epsilon = BATCHNORM_DEFAULT_EPSILON;
    const double momentum = 0.1;
    const std::vector<int64_t> dims = {6, 3, 32, 32};
    const unsigned int seed = getGlobalTestSeed();
    BatchnormTrainTensorBundle<float, float, float> planTensorBundle(
        dims, seed, TensorLayout::NHWC);
    BatchnormTrainTensorBundle<float, float, float> directTensorBundle(
        dims, seed, TensorLayout::NHWC);

    BatchnormTrainParams<float> params;
    initTensorValues(params.xTensor, DataType::FLOAT, planTensorBundle.xTensor, 1);
    initTensorValues(params.scaleTensor, DataType::FLOAT, planTensorBundle.scaleTensor, 2);
    initTensorValues(params.biasTensor, DataType::FLOAT, planTensorBundle.biasTensor, 3);
    initTensorValues(params.epsilonTensor, DataType::DOUBLE, planTensorBundle.epsilonTensor, 4);
    params.epsilonTensor.value.Set(hipdnn_flatbuffers_sdk::data_objects::Float64Value(epsilon));
    initTensorValues(params.yTensor, DataType::FLOAT, planTensorBundle.yTensor, 5);

    // Initialize optional mean and invVariance tensors
    params.meanTensor = TensorAttributesT();
    initTensorValues(params.meanTensor.value(), DataType::FLOAT, planTensorBundle.meanTensor, 6);

    params.invVarianceTensor = TensorAttributesT();
    initTensorValues(
        params.invVarianceTensor.value(), DataType::FLOAT, planTensorBundle.invVarianceTensor, 7);

    BatchnormTrainPlan<float, float, float, float, float> patient(std::move(params));

    std::unordered_map<int64_t, void*> variantPack;
    variantPack[1] = planTensorBundle.xTensor.memory().hostData();
    variantPack[2] = planTensorBundle.scaleTensor.memory().hostData();
    variantPack[3] = planTensorBundle.biasTensor.memory().hostData();
    variantPack[4] = planTensorBundle.epsilonTensor.memory().hostData();
    variantPack[5] = planTensorBundle.yTensor.memory().hostData();
    variantPack[6] = planTensorBundle.meanTensor.memory().hostData();
    variantPack[7] = planTensorBundle.invVarianceTensor.memory().hostData();

    CpuFpReferenceBatchnorm::fwdTraining(directTensorBundle.xTensor,
                                         directTensorBundle.scaleTensor,
                                         directTensorBundle.biasTensor,
                                         directTensorBundle.yTensor,
                                         epsilon,
                                         momentum,
                                         &directTensorBundle.meanTensor,
                                         &directTensorBundle.invVarianceTensor);

    patient.execute(variantPack);

    auto tolerance = batchnorm::getToleranceTraining<float>();
    const CpuFpReferenceValidation<float> cpuRefOutputValidation(tolerance, tolerance);

    EXPECT_TRUE(
        cpuRefOutputValidation.allClose(directTensorBundle.yTensor, planTensorBundle.yTensor));
    EXPECT_TRUE(cpuRefOutputValidation.allClose(directTensorBundle.meanTensor,
                                                planTensorBundle.meanTensor));
    EXPECT_TRUE(cpuRefOutputValidation.allClose(directTensorBundle.invVarianceTensor,
                                                planTensorBundle.invVarianceTensor));
}

TEST(TestBatchnormTrainPlanBuilder, PlanConstruction)
{
    const std::vector<int64_t> dims = {2, 1, 1, 1};
    BatchnormTrainTensorBundle<float, float, float> tensorBundle(dims, 1, TensorLayout::NCHW);

    auto graphTuple = buildBatchnormTrainGraph(
        tensorBundle, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, false);

    auto& graph = std::get<0>(graphTuple);
    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();

    auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        serializedGraph.data(), serializedGraph.size());

    const BatchnormTrainPlanBuilder<DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::FLOAT>
        patient;

    auto builtPlan = patient.buildNodePlan(graphWrap, graphWrap.getNode(0));

    const bool result
        = dynamic_cast<BatchnormTrainPlan<float, float, float, float, float>*>(builtPlan.get())
          != nullptr;
    EXPECT_TRUE(result);
}

TEST(TestBatchnormTrainPlanBuilder, IsApplicable)
{
    const std::vector<int64_t> dims = {2, 1, 1, 1};
    BatchnormTrainTensorBundle<float, float, float> tensorBundle(dims, 1, TensorLayout::NCHW);

    auto graphTuple = buildBatchnormTrainGraph(
        tensorBundle, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, false);

    auto& graph = std::get<0>(graphTuple);
    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();

    auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        serializedGraph.data(), serializedGraph.size());

    const BatchnormTrainPlanBuilder<DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::FLOAT>
        floatPlanBuilder;

    EXPECT_TRUE(floatPlanBuilder.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap()));

    const BatchnormTrainPlanBuilder<DataType::FLOAT,
                                    DataType::HALF,
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::FLOAT>
        badTypesPlanBuilder;
    EXPECT_FALSE(badTypesPlanBuilder.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap()));

    auto tensorMapCopy = graphWrap.getTensorMap();
    tensorMapCopy.erase(6);
    EXPECT_FALSE(floatPlanBuilder.isApplicable(graphWrap.getNode(0), tensorMapCopy));
}
