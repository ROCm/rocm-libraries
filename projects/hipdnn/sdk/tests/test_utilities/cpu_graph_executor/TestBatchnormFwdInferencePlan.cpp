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

    const INodeWrapper& node = graphWrapper.getNodeWrapper(0);
    const auto& attributes
        = node.attributesAs<hipdnn_sdk::data_objects::BatchnormInferenceAttributes>();

    BatchnormFwdInferenceParams params;
    initTensorValues(params.xTensor,
                     DataType::FLOAT,
                     planTensorBundle.tensors[attributes.x_tensor_uid()]->dims(),
                     planTensorBundle.tensors[attributes.x_tensor_uid()]->strides(),
                     attributes.x_tensor_uid());
    initTensorValues(params.scaleTensor,
                     DataType::FLOAT,
                     planTensorBundle.tensors[attributes.scale_tensor_uid()]->dims(),
                     planTensorBundle.tensors[attributes.scale_tensor_uid()]->strides(),
                     attributes.scale_tensor_uid());
    initTensorValues(params.biasTensor,
                     DataType::FLOAT,
                     planTensorBundle.tensors[attributes.bias_tensor_uid()]->dims(),
                     planTensorBundle.tensors[attributes.bias_tensor_uid()]->strides(),
                     attributes.bias_tensor_uid());
    initTensorValues(params.meanTensor,
                     DataType::FLOAT,
                     planTensorBundle.tensors[attributes.mean_tensor_uid()]->dims(),
                     planTensorBundle.tensors[attributes.mean_tensor_uid()]->strides(),
                     attributes.mean_tensor_uid());
    initTensorValues(params.invVarianceTensor,
                     DataType::FLOAT,
                     planTensorBundle.tensors[attributes.inv_variance_tensor_uid()]->dims(),
                     planTensorBundle.tensors[attributes.inv_variance_tensor_uid()]->strides(),
                     attributes.inv_variance_tensor_uid());
    initTensorValues(params.yTensor,
                     DataType::FLOAT,
                     planTensorBundle.tensors[attributes.y_tensor_uid()]->dims(),
                     planTensorBundle.tensors[attributes.y_tensor_uid()]->strides(),
                     attributes.y_tensor_uid());
    params.epsilon = epsilon;

    BatchnormFwdPlan<float, float, float> patient(std::move(params));
    std::unordered_map<int64_t, void*> variantPack = planTensorBundle.toVariantPack();

    CpuFpReferenceBatchnormImpl<float, float>::batchnormFwdInference(
        *dynamic_cast<TensorBase<float>*>(
            directTensorBundle.tensors[attributes.x_tensor_uid()].get()),
        *dynamic_cast<TensorBase<float>*>(
            directTensorBundle.tensors[attributes.scale_tensor_uid()].get()),
        *dynamic_cast<TensorBase<float>*>(
            directTensorBundle.tensors[attributes.bias_tensor_uid()].get()),
        *dynamic_cast<TensorBase<float>*>(
            directTensorBundle.tensors[attributes.mean_tensor_uid()].get()),
        *dynamic_cast<TensorBase<float>*>(
            directTensorBundle.tensors[attributes.inv_variance_tensor_uid()].get()),
        *dynamic_cast<TensorBase<float>*>(
            directTensorBundle.tensors[attributes.y_tensor_uid()].get()),
        epsilon);

    patient.execute(variantPack);

    CpuFpReferenceValidation<float> cpuRefOutputValidation(tolerance, tolerance);
    auto& yDirect = *dynamic_cast<TensorBase<float>*>(
        directTensorBundle.tensors[attributes.y_tensor_uid()].get());
    auto& yPlanTensor = *dynamic_cast<TensorBase<float>*>(
        planTensorBundle.tensors[attributes.y_tensor_uid()].get());
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
