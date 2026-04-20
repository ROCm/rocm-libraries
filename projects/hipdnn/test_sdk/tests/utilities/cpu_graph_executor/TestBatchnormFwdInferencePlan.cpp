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
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/BatchnormFwdInferencePlan.hpp>

using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_test_sdk::detail;
using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_flatbuffers_sdk::flatbuffer_utilities;
using namespace ::testing;
using namespace hipdnn_sdk_test_utils;

class TestBatchnormFwdPlan : public ::testing::Test
{
protected:
    static void
        initTensorValues(hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT& tensorAttr,
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
    const std::vector<int64_t> dims = {6, 3, 32, 32};
    const unsigned int seed = getGlobalTestSeed();
    auto graph = buildBatchnormFwdInferenceGraph(DataType::FLOAT,
                                                 DataType::FLOAT,
                                                 DataType::FLOAT,
                                                 DataType::FLOAT,
                                                 dims,
                                                 TensorLayout::NHWC);
    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    const GraphWrapper graphWrapper(serializedGraph.data(), serializedGraph.size());
    const INodeWrapper& node = graphWrapper.getNodeWrapper(0);
    BatchnormFwdTensorBundle planTensorBundle(node, graphWrapper.getTensorMap(), seed);
    BatchnormFwdTensorBundle directTensorBundle(node, graphWrapper.getTensorMap(), seed);

    const auto& attributes
        = node.attributesAs<hipdnn_flatbuffers_sdk::data_objects::BatchnormInferenceAttributes>();
    const auto& tensorMap = graphWrapper.getTensorMap();
    BatchnormFwdInferenceParams params(*tensorMap.at(attributes.x_tensor_uid()),
                                       *tensorMap.at(attributes.y_tensor_uid()),
                                       *tensorMap.at(attributes.scale_tensor_uid()),
                                       *tensorMap.at(attributes.bias_tensor_uid()),
                                       *tensorMap.at(attributes.mean_tensor_uid()),
                                       *tensorMap.at(attributes.inv_variance_tensor_uid()));

    const std::unordered_map<int64_t, void*> variantPack = planTensorBundle.toHostVariantPack();

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

    CpuFpReferenceBatchnorm::fwdInference(*shallowXTensor,
                                          *shallowScaleTensor,
                                          *shallowBiasTensor,
                                          *shallowMeanTensor,
                                          *shallowInvVarTensor,
                                          *shallowYTensor);

    BatchnormFwdPlan<float, float, float, float, float> fwdPlan(std::move(params));
    fwdPlan.execute(variantPack);

    const CpuFpReferenceValidation<float> cpuRefOutputValidation(tolerance, tolerance);
    EXPECT_TRUE(cpuRefOutputValidation.allClose(
        *directTensorBundle.tensors[attributes.y_tensor_uid()].get(),
        *planTensorBundle.tensors[attributes.y_tensor_uid()].get()));
}

TEST(TestBatchnormFwdInferencePlanBuilder, PlanConstruction)
{
    const std::vector<int64_t> dims = {1, 1, 1, 1};
    auto graph = buildBatchnormFwdInferenceGraph(DataType::FLOAT,
                                                 DataType::FLOAT,
                                                 DataType::FLOAT,
                                                 DataType::FLOAT,
                                                 dims,
                                                 TensorLayout::NHWC);
    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    const GraphWrapper graphWrapper(serializedGraph.data(), serializedGraph.size());

    const BatchnormFwdInferencePlanBuilder<DataType::FLOAT,
                                           DataType::FLOAT,
                                           DataType::FLOAT,
                                           DataType::FLOAT,
                                           DataType::FLOAT>
        patient;

    auto builtPlan = patient.buildNodePlan(graphWrapper, graphWrapper.getNode(0));

    const bool result
        = dynamic_cast<BatchnormFwdPlan<float, float, float, float, float>*>(builtPlan.get())
          != nullptr;
    EXPECT_TRUE(result);
}

TEST(TestBatchnormFwdInferencePlanBuilder, IsApplicable)
{
    const std::vector<int64_t> dims = {1, 1, 1, 1};
    auto graph = buildBatchnormFwdInferenceGraph(DataType::FLOAT,
                                                 DataType::FLOAT,
                                                 DataType::FLOAT,
                                                 DataType::FLOAT,
                                                 dims,
                                                 TensorLayout::NHWC);
    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    const GraphWrapper graphWrapper(serializedGraph.data(), serializedGraph.size());

    const BatchnormFwdInferencePlanBuilder<DataType::FLOAT,
                                           DataType::FLOAT,
                                           DataType::FLOAT,
                                           DataType::FLOAT,
                                           DataType::FLOAT>
        floatPlanBuilder;

    EXPECT_TRUE(
        floatPlanBuilder.isApplicable(graphWrapper.getNode(0), graphWrapper.getTensorMap()));

    const BatchnormFwdInferencePlanBuilder<DataType::FLOAT,
                                           DataType::HALF,
                                           DataType::FLOAT,
                                           DataType::FLOAT,
                                           DataType::FLOAT>
        badTypesPlanBuilder;
    EXPECT_FALSE(
        badTypesPlanBuilder.isApplicable(graphWrapper.getNode(0), graphWrapper.getTensorMap()));

    auto tensorMapCopy = graphWrapper.getTensorMap();
    tensorMapCopy.erase(6);
    EXPECT_FALSE(floatPlanBuilder.isApplicable(graphWrapper.getNode(0), tensorMapCopy));
}
