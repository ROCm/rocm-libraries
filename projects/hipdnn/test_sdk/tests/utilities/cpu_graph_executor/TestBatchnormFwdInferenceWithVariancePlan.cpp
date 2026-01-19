// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include "BatchnormGraphUtils.hpp"
#include "BatchnormTensorBundles.hpp"
#include <hipdnn_data_sdk/data_objects/graph_generated.h>
#include <hipdnn_data_sdk/utilities/Constants.hpp>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceBatchnorm.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/Seeds.hpp>
#include <hipdnn_test_sdk/utilities/TestTolerances.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/BatchnormFwdInferenceWithVariancePlan.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/BatchnormFwdInferenceWithVarianceSignatureKey.hpp>

using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_data_sdk::data_objects;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_plugin_sdk;
using namespace ::testing;
using namespace hipdnn_sdk_test_utils;

class TestBatchnormFwdWithVariancePlan : public ::testing::Test
{
protected:
    static void initTensorValues(hipdnn_data_sdk::data_objects::TensorAttributesT& tensorAttr,
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

TEST_F(TestBatchnormFwdWithVariancePlan, ExecutePlan)
{
    auto tolerance = batchnorm::getToleranceInference<float>();
    std::vector<int64_t> dims = {6, 3, 32, 32};
    unsigned int seed = getGlobalTestSeed();
    auto graph = buildBatchnormFwdInferenceWithVarianceGraph(DataType::FLOAT,
                                                             DataType::FLOAT,
                                                             DataType::FLOAT,
                                                             DataType::FLOAT,
                                                             dims,
                                                             TensorLayout::NHWC);
    auto flatbufferGraph = graph->buildFlatbufferOperationGraph();
    GraphWrapper graphWrapper(flatbufferGraph.data(), flatbufferGraph.size());
    const INodeWrapper& node = graphWrapper.getNodeWrapper(0);
    BatchnormFwdWithVarianceTensorBundle planTensorBundle(node, graphWrapper.getTensorMap(), seed);
    BatchnormFwdWithVarianceTensorBundle directTensorBundle(
        node, graphWrapper.getTensorMap(), seed);

    const auto& attributes = node.attributesAs<
        hipdnn_data_sdk::data_objects::BatchnormInferenceAttributesVarianceExt>();
    const auto& tensorMap = graphWrapper.getTensorMap();
    BatchnormFwdInferenceWithVarianceParams params(*tensorMap.at(attributes.x_tensor_uid()),
                                                   *tensorMap.at(attributes.y_tensor_uid()),
                                                   *tensorMap.at(attributes.scale_tensor_uid()),
                                                   *tensorMap.at(attributes.bias_tensor_uid()),
                                                   *tensorMap.at(attributes.mean_tensor_uid()),
                                                   *tensorMap.at(attributes.variance_tensor_uid()));

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
    auto shallowVarianceTensor = createShallowTensor<float>(
        params.varianceTensor,
        directTensorBundle.tensors[attributes.variance_tensor_uid()]->rawHostData());
    auto shallowYTensor = createShallowTensor<float>(
        params.yTensor, directTensorBundle.tensors[attributes.y_tensor_uid()]->rawHostData());

    CpuFpReferenceBatchnorm::fwdInferenceWithVariance(*shallowXTensor,
                                                      *shallowScaleTensor,
                                                      *shallowBiasTensor,
                                                      *shallowMeanTensor,
                                                      *shallowVarianceTensor,
                                                      *shallowYTensor);

    BatchnormFwdInferenceWithVariancePlan<float, float, float, float, float> fwdPlan(
        std::move(params));
    fwdPlan.execute(variantPack);

    CpuFpReferenceValidation<float> cpuRefOutputValidation(tolerance, tolerance);
    EXPECT_TRUE(cpuRefOutputValidation.allClose(
        *directTensorBundle.tensors[attributes.y_tensor_uid()].get(),
        *planTensorBundle.tensors[attributes.y_tensor_uid()].get()));
}

TEST_F(TestBatchnormFwdWithVariancePlan, ExecutePlanThrowsWhenEpsilonIsMissing)
{
    std::vector<int64_t> dims = {6, 3, 32, 32};
    // Build graph with missing epsilon (nullptr)
    auto graph = buildBatchnormFwdInferenceWithVarianceGraph(DataType::FLOAT,
                                                             DataType::FLOAT,
                                                             DataType::FLOAT,
                                                             DataType::FLOAT,
                                                             dims,
                                                             TensorLayout::NHWC);

    // Manually construct a graph with missing epsilon to verify that it throws an exception
    auto graphWithMissingEpsilon = std::make_shared<hipdnn_frontend::graph::Graph>();
    graphWithMissingEpsilon->set_name("BatchnormFwdInferenceWithVarianceTest_MissingEpsilon");

    auto strides
        = hipdnn_data_sdk::utilities::generateStrides(dims, TensorLayout::NHWC.strideOrder);
    auto derivedDims = hipdnn_data_sdk::utilities::getDerivedShape(dims);
    auto derivedStrides = hipdnn_data_sdk::utilities::generateStrides(derivedDims);

    int64_t uid = 1;
    auto xAttr = hipdnn_frontend::graph::makeTensorAttributes(
        "x", hipdnn_frontend::fromSdkType(DataType::FLOAT), dims, strides);
    xAttr.set_uid(uid++);
    auto xTensorAttr = std::make_shared<hipdnn_frontend::graph::TensorAttributes>(std::move(xAttr));

    auto scaleAttr = hipdnn_frontend::graph::makeTensorAttributes(
        "scale", hipdnn_frontend::fromSdkType(DataType::FLOAT), derivedDims, derivedStrides);
    scaleAttr.set_uid(uid++);
    auto scaleTensorAttr
        = std::make_shared<hipdnn_frontend::graph::TensorAttributes>(std::move(scaleAttr));

    auto biasAttr = hipdnn_frontend::graph::makeTensorAttributes(
        "bias", hipdnn_frontend::fromSdkType(DataType::FLOAT), derivedDims, derivedStrides);
    biasAttr.set_uid(uid++);
    auto biasTensorAttr
        = std::make_shared<hipdnn_frontend::graph::TensorAttributes>(std::move(biasAttr));

    auto meanAttr = hipdnn_frontend::graph::makeTensorAttributes(
        "mean", hipdnn_frontend::fromSdkType(DataType::FLOAT), derivedDims, derivedStrides);
    meanAttr.set_uid(uid++);
    auto meanTensorAttr
        = std::make_shared<hipdnn_frontend::graph::TensorAttributes>(std::move(meanAttr));

    auto varianceAttr = hipdnn_frontend::graph::makeTensorAttributes(
        "variance", hipdnn_frontend::fromSdkType(DataType::FLOAT), derivedDims, derivedStrides);
    varianceAttr.set_uid(uid++);
    auto varianceTensorAttr
        = std::make_shared<hipdnn_frontend::graph::TensorAttributes>(std::move(varianceAttr));

    hipdnn_frontend::graph::BatchnormInferenceAttributesVarianceExt bnAttrs;
    bnAttrs.set_name("batchnorm_fwd_inference_with_variance");
    bnAttrs.set_compute_data_type(hipdnn_frontend::fromSdkType(DataType::FLOAT));

    // Pass nullptr for epsilon
    graphWithMissingEpsilon->batchnorm_inference_variance_ext(xTensorAttr,
                                                              meanTensorAttr,
                                                              varianceTensorAttr,
                                                              scaleTensorAttr,
                                                              biasTensorAttr,
                                                              nullptr,
                                                              bnAttrs);

    EXPECT_THROW(graphWithMissingEpsilon->buildFlatbufferOperationGraph(), std::runtime_error);
}

TEST(TestBatchnormFwdInferenceWithVariancePlanBuilder, PlanConstruction)
{
    std::vector<int64_t> dims = {1, 1, 1, 1};
    auto graph = buildBatchnormFwdInferenceWithVarianceGraph(DataType::FLOAT,
                                                             DataType::FLOAT,
                                                             DataType::FLOAT,
                                                             DataType::FLOAT,
                                                             dims,
                                                             TensorLayout::NHWC);
    auto flatbufferGraph = graph->buildFlatbufferOperationGraph();
    GraphWrapper graphWrapper(flatbufferGraph.data(), flatbufferGraph.size());

    BatchnormFwdInferenceWithVariancePlanBuilder<DataType::FLOAT,
                                                 DataType::FLOAT,
                                                 DataType::FLOAT,
                                                 DataType::FLOAT,
                                                 DataType::FLOAT>
        patient;

    auto builtPlan = patient.buildNodePlan(graphWrapper, graphWrapper.getNode(0));

    bool result
        = dynamic_cast<BatchnormFwdInferenceWithVariancePlan<float, float, float, float, float>*>(
              builtPlan.get())
          != nullptr;
    EXPECT_TRUE(result);
}

TEST(TestBatchnormFwdInferenceWithVariancePlanBuilder, IsApplicable)
{
    std::vector<int64_t> dims = {1, 1, 1, 1};
    auto graph = buildBatchnormFwdInferenceWithVarianceGraph(DataType::FLOAT,
                                                             DataType::FLOAT,
                                                             DataType::FLOAT,
                                                             DataType::FLOAT,
                                                             dims,
                                                             TensorLayout::NHWC);
    auto flatbufferGraph = graph->buildFlatbufferOperationGraph();
    GraphWrapper graphWrapper(flatbufferGraph.data(), flatbufferGraph.size());

    BatchnormFwdInferenceWithVariancePlanBuilder<DataType::FLOAT,
                                                 DataType::FLOAT,
                                                 DataType::FLOAT,
                                                 DataType::FLOAT,
                                                 DataType::FLOAT>
        floatPlanBuilder;

    EXPECT_TRUE(
        floatPlanBuilder.isApplicable(graphWrapper.getNode(0), graphWrapper.getTensorMap()));

    BatchnormFwdInferenceWithVariancePlanBuilder<DataType::FLOAT,
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

TEST(TestBatchnormFwdInferenceWithVariancePlan, PlanBuilderMapContainsExpectedKeys)
{
    auto planBuilders = BatchnormFwdInferenceWithVarianceSignatureKey::getPlanBuilders();

    // Verify we have builders for common type combinations
    EXPECT_GT(planBuilders.size(), 0);

    // FP32 case
    BatchnormFwdInferenceWithVarianceSignatureKey fp32Key(
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT);
    EXPECT_TRUE(planBuilders.find(fp32Key) != planBuilders.end());

    // FP16 case with FP32 params
    BatchnormFwdInferenceWithVarianceSignatureKey fp16Key(
        DataType::HALF, DataType::FLOAT, DataType::FLOAT, DataType::HALF, DataType::FLOAT);
    EXPECT_TRUE(planBuilders.find(fp16Key) != planBuilders.end());

    // BFP16 case with FP32 params
    BatchnormFwdInferenceWithVarianceSignatureKey bfp16Key(
        DataType::BFLOAT16, DataType::FLOAT, DataType::FLOAT, DataType::BFLOAT16, DataType::FLOAT);
    EXPECT_TRUE(planBuilders.find(bfp16Key) != planBuilders.end());
}

TEST(TestBatchnormFwdInferenceWithVariancePlan, SignatureKeyHashingWorks)
{
    BatchnormFwdInferenceWithVarianceSignatureKey key1(
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT);

    BatchnormFwdInferenceWithVarianceSignatureKey key2(
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT);

    BatchnormFwdInferenceWithVarianceSignatureKey key3(
        DataType::HALF, DataType::FLOAT, DataType::FLOAT, DataType::HALF, DataType::FLOAT);

    // Same keys should be equal
    EXPECT_TRUE(key1 == key2);
    EXPECT_EQ(key1.hashSelf(), key2.hashSelf());

    // Different keys should not be equal
    EXPECT_FALSE(key1 == key3);
    // Hash collision is possible but unlikely for these specific cases
    EXPECT_NE(key1.hashSelf(), key3.hashSelf());
}

TEST(TestBatchnormFwdInferenceWithVariancePlan, NodeTypeIsCorrect)
{
    BatchnormFwdInferenceWithVarianceSignatureKey key;
    EXPECT_EQ(key.nodeType, NodeAttributes::BatchnormInferenceAttributesVarianceExt);
}
