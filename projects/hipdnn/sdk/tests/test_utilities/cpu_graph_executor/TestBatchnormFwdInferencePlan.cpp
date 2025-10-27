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
#include <hipdnn_sdk/test_utilities/cpu_graph_executor/BatchnormFwdInferencePlan.hpp>
#include <hipdnn_sdk/utilities/Constants.hpp>
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
    double epsilon = BATCHNORM_DEFAULT_EPSILON;
    std::vector<int64_t> dims = {6, 3, 32, 32};
    unsigned int seed = getGlobalTestSeed();
    auto graph = buildBatchnormFwdInferenceGraph(
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, dims, TensorLayout::NHWC);
    auto flatbufferGraph = graph->buildFlatbufferOperationGraph();
    GraphWrapper graphWrapper(flatbufferGraph.data(), flatbufferGraph.size());
    const INodeWrapper& node = graphWrapper.getNodeWrapper(0);
    BatchnormFwdTensorBundle planTensorBundle(node, graphWrapper.getTensorMap(), seed);
    BatchnormFwdTensorBundle directTensorBundle(node, graphWrapper.getTensorMap(), seed);

    const auto& attributes
        = node.attributesAs<hipdnn_sdk::data_objects::BatchnormInferenceAttributes>();
    const auto& tensorMap = graphWrapper.getTensorMap();
    BatchnormFwdInferenceParams params(*tensorMap.at(attributes.x_tensor_uid()),
                                       *tensorMap.at(attributes.y_tensor_uid()),
                                       *tensorMap.at(attributes.scale_tensor_uid()),
                                       *tensorMap.at(attributes.bias_tensor_uid()),
                                       *tensorMap.at(attributes.mean_tensor_uid()),
                                       *tensorMap.at(attributes.inv_variance_tensor_uid()),
                                       epsilon);

    std::unordered_map<int64_t, void*> variantPack = planTensorBundle.toHostVariantPack();

    auto shallowXTensor = createShallowTensor<float>(
        params.xTensor, directTensorBundle.tensors[attributes.x_tensor_uid()]->rawHostData());
    auto shallowScaleTensor = createShallowTensor<float>(
        params.scaleTensor,
        directTensorBundle.tensors[attributes.scale_tensor_uid()]->rawHostData());
    auto shallowBiasTensor = createShallowTensor<float>(
        params.biasTensor, directTensorBundle.tensors[attributes.bias_tensor_uid()]->rawHostData());
    auto shallowMeanTensor = createShallowTensor<float>(
        params.meanTensor, directTensorBundle.tensors[attributes.mean_tensor_uid()]->rawHostData());
    auto shallowInvVarTensor = createShallowTensor<float>(
        params.invVarianceTensor,
        directTensorBundle.tensors[attributes.inv_variance_tensor_uid()]->rawHostData());
    auto shallowYTensor = createShallowTensor<float>(
        params.yTensor, directTensorBundle.tensors[attributes.y_tensor_uid()]->rawHostData());

    CpuFpReferenceBatchnormImpl<float, float>::batchnormFwdInference(*shallowXTensor,
                                                                     *shallowScaleTensor,
                                                                     *shallowBiasTensor,
                                                                     *shallowMeanTensor,
                                                                     *shallowInvVarTensor,
                                                                     *shallowYTensor,
                                                                     epsilon);

    BatchnormFwdPlan<float, float, float> fwdPlan(std::move(params));
    fwdPlan.execute(variantPack);

    CpuFpReferenceValidation<float> cpuRefOutputValidation(tolerance, tolerance);
    EXPECT_TRUE(cpuRefOutputValidation.allClose(
        *directTensorBundle.tensors[attributes.y_tensor_uid()].get(),
        *planTensorBundle.tensors[attributes.y_tensor_uid()].get()));
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
