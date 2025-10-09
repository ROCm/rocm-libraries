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
                                 const std::vector<int64_t>& dims,
                                 const std::vector<int64_t>& strides,
                                 int64_t uid)
    {
        tensorAttr.data_type = dataType;
        tensorAttr.dims = dims;
        tensorAttr.strides = strides;
        tensorAttr.uid = uid;
    }
};

TEST_F(TestBatchnormFwdPlan, ExecutePlan)
{
    auto tolerance = batchnorm::getToleranceInference<float>();
    double epsilon = 1e-3;
    std::vector<int64_t> dims = {6, 3, 32, 32};
    unsigned int seed = 1;
    auto graph = buildBatchnormFwdInferenceGraph(
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, dims, TensorLayout::NHWC);
    auto flatbufferGraph = graph->buildFlatbufferOperationGraph();
    GraphWrapper graphWrapper(flatbufferGraph.data(), flatbufferGraph.size());
    BatchnormFwdTensorBundle planTensorBundle(
        graphWrapper.getNodeWrapper(0), graphWrapper.getTensorMap(), seed);
    BatchnormFwdTensorBundle directTensorBundle(
        graphWrapper.getNodeWrapper(0), graphWrapper.getTensorMap(), seed);

    BatchnormFwdInferenceParams params;
    initTensorValues(params.xTensor,
                     DataType::FLOAT,
                     planTensorBundle.tensors[1]->dims(),
                     planTensorBundle.tensors[1]->strides(),
                     1);
    initTensorValues(params.scaleTensor,
                     DataType::FLOAT,
                     planTensorBundle.tensors[2]->dims(),
                     planTensorBundle.tensors[2]->strides(),
                     2);
    initTensorValues(params.biasTensor,
                     DataType::FLOAT,
                     planTensorBundle.tensors[3]->dims(),
                     planTensorBundle.tensors[3]->strides(),
                     3);
    initTensorValues(params.meanTensor,
                     DataType::FLOAT,
                     planTensorBundle.tensors[4]->dims(),
                     planTensorBundle.tensors[4]->strides(),
                     4);
    initTensorValues(params.invVarianceTensor,
                     DataType::FLOAT,
                     planTensorBundle.tensors[5]->dims(),
                     planTensorBundle.tensors[5]->strides(),
                     5);
    initTensorValues(params.yTensor,
                     DataType::FLOAT,
                     planTensorBundle.tensors[6]->dims(),
                     planTensorBundle.tensors[6]->strides(),
                     6);
    params.epsilon = epsilon;

    BatchnormFwdPlan<float, float, float> patient(std::move(params));
    std::unordered_map<int64_t, void*> variantPack = planTensorBundle.toVariantPack();

    CpuFpReferenceBatchnormImpl<float, float>::batchnormFwdInference(
        *dynamic_cast<TensorBase<float>*>(directTensorBundle.tensors[1].get()),
        *dynamic_cast<TensorBase<float>*>(directTensorBundle.tensors[2].get()),
        *dynamic_cast<TensorBase<float>*>(directTensorBundle.tensors[3].get()),
        *dynamic_cast<TensorBase<float>*>(directTensorBundle.tensors[4].get()),
        *dynamic_cast<TensorBase<float>*>(directTensorBundle.tensors[5].get()),
        *dynamic_cast<TensorBase<float>*>(directTensorBundle.tensors[6].get()),
        epsilon);

    patient.execute(variantPack);

    CpuFpReferenceValidation<float> cpuRefOutputValidation(tolerance, tolerance);
    auto& yDirect = *dynamic_cast<TensorBase<float>*>(directTensorBundle.tensors[6].get());
    auto& yPlanTensor = *dynamic_cast<TensorBase<float>*>(planTensorBundle.tensors[6].get());
    EXPECT_TRUE(cpuRefOutputValidation.allClose(yDirect.memory(), yPlanTensor.memory()));
}

TEST(TestBatchnormFwdInferencePlanBuilder, PlanConstruction)
{
    std::vector<int64_t> dims = {1, 1, 1, 1};
    auto graph = buildBatchnormFwdInferenceGraph(
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, dims, TensorLayout::NHWC);
    auto flatbufferGraph = graph->buildFlatbufferOperationGraph();
    GraphWrapper graphWrapper(flatbufferGraph.data(), flatbufferGraph.size());

    BatchnormFwdInferencePlanBuilder<DataType::FLOAT, DataType::FLOAT, DataType::FLOAT> patient;

    auto builtPlan = patient.buildNodePlan(graphWrapper, graphWrapper.getNode(0));

    bool result = dynamic_cast<BatchnormFwdPlan<float, float, float>*>(builtPlan.get()) != nullptr;
    EXPECT_TRUE(result);
}

TEST(TestBatchnormFwdInferencePlanBuilder, IsApplicable)
{
    std::vector<int64_t> dims = {1, 1, 1, 1};
    auto graph = buildBatchnormFwdInferenceGraph(
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, dims, TensorLayout::NHWC);
    auto flatbufferGraph = graph->buildFlatbufferOperationGraph();
    GraphWrapper graphWrapper(flatbufferGraph.data(), flatbufferGraph.size());

    BatchnormFwdInferencePlanBuilder<DataType::FLOAT, DataType::FLOAT, DataType::FLOAT>
        floatPlanBuilder;

    EXPECT_TRUE(
        floatPlanBuilder.isApplicable(graphWrapper.getNode(0), graphWrapper.getTensorMap()));

    BatchnormFwdInferencePlanBuilder<DataType::FLOAT, DataType::HALF, DataType::FLOAT>
        badTypesPlanBuilder;
    EXPECT_FALSE(
        badTypesPlanBuilder.isApplicable(graphWrapper.getNode(0), graphWrapper.getTensorMap()));

    auto tensorMapCopy = graphWrapper.getTensorMap();
    tensorMapCopy.erase(6);
    EXPECT_FALSE(floatPlanBuilder.isApplicable(graphWrapper.getNode(0), tensorMapCopy));
}
