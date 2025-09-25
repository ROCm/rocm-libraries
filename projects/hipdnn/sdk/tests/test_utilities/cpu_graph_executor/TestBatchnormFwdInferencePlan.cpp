// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <hipdnn_sdk/data_objects/graph_generated.h>
#include <hipdnn_sdk/plugin/test_utils/MockGraph.hpp>
#include <hipdnn_sdk/test_utilities/CpuFpReferenceBatchnorm.hpp>
#include <hipdnn_sdk/test_utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_sdk/test_utilities/cpu_graph_executor/BatchnormFwdInferencePlan.hpp>
#include <hipdnn_sdk/utilities/ShapeUtilities.hpp>

using namespace hipdnn_sdk::test_utilities;
using namespace hipdnn_sdk::data_objects;
using namespace hipdnn_sdk::utilities;
using namespace hipdnn_plugin;
using namespace ::testing;

template <typename InputType, typename IntermediateType>
struct BatchnormFwdTensorBundle
{
    BatchnormFwdTensorBundle(const std::vector<int64_t>& dims,
                             unsigned int seed = 1,
                             const TensorLayout& layout = TensorLayout::NCHW)
        : derivedDims(getDerivedShape(dims))
        , inputTensor(dims, layout)
        , outputTensor(dims, layout)
        , biasTensor(derivedDims)
        , scaleTensor(derivedDims)
        , meanTensor(derivedDims)
        , varianceTensor(derivedDims)
    {
        inputTensor.fillWithRandomValues(
            static_cast<InputType>(0.0f), static_cast<InputType>(1.0f), seed);

        outputTensor.fillWithRandomValues(
            static_cast<InputType>(-100.0f), static_cast<InputType>(100.0f), seed);

        scaleTensor.fillWithRandomValues(
            static_cast<IntermediateType>(0.0f), static_cast<IntermediateType>(1.0f), seed);

        biasTensor.fillWithRandomValues(
            static_cast<IntermediateType>(0.0f), static_cast<IntermediateType>(1.0f), seed);

        meanTensor.fillWithRandomValues(
            static_cast<IntermediateType>(0.0f), static_cast<IntermediateType>(1.0f), seed);

        varianceTensor.fillWithRandomValues(
            static_cast<IntermediateType>(0.1f), static_cast<IntermediateType>(1.0f), seed);
    }

    std::vector<int64_t> derivedDims;
    Tensor<InputType> inputTensor;
    Tensor<InputType> outputTensor;
    Tensor<IntermediateType> biasTensor;
    Tensor<IntermediateType> scaleTensor;
    Tensor<IntermediateType> meanTensor;
    Tensor<IntermediateType> varianceTensor;
};

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

TEST_F(TestBatchnormFwdPlan, executePlan)
{
    double epsilon = 1e-3;
    std::vector<int64_t> dims = {6, 3, 32, 32};
    unsigned int seed = 1;
    BatchnormFwdTensorBundle<float, float> planTensorBundle(dims, seed, TensorLayout::NHWC);
    BatchnormFwdTensorBundle<float, float> directTensorBundle(dims, seed, TensorLayout::NHWC);

    BatchnormFwdInferenceParams params;
    initTensorValues(params.xTensor, DataType::FLOAT, planTensorBundle.inputTensor, 1);
    initTensorValues(params.yTensor, DataType::FLOAT, planTensorBundle.outputTensor, 2);
    initTensorValues(params.biasTensor, DataType::FLOAT, planTensorBundle.biasTensor, 3);
    initTensorValues(params.scaleTensor, DataType::FLOAT, planTensorBundle.scaleTensor, 4);
    initTensorValues(params.meanTensor, DataType::FLOAT, planTensorBundle.meanTensor, 5);
    initTensorValues(params.invVarianceTensor, DataType::FLOAT, planTensorBundle.varianceTensor, 6);
    params.epsilon = epsilon;

    BatchnormFwdPlan<float, float, float> patient(std::move(params));

    std::unordered_map<int64_t, void*> variantPack;
    variantPack[1] = planTensorBundle.inputTensor.memory().hostData();
    variantPack[2] = planTensorBundle.outputTensor.memory().hostData();
    variantPack[3] = planTensorBundle.biasTensor.memory().hostData();
    variantPack[4] = planTensorBundle.scaleTensor.memory().hostData();
    variantPack[5] = planTensorBundle.meanTensor.memory().hostData();
    variantPack[6] = planTensorBundle.varianceTensor.memory().hostData();

    CpuFpReferenceBatchnormImpl<float, float>::batchnormFwdInference(
        directTensorBundle.inputTensor,
        directTensorBundle.scaleTensor,
        directTensorBundle.biasTensor,
        directTensorBundle.meanTensor,
        directTensorBundle.varianceTensor,
        directTensorBundle.outputTensor,
        epsilon);

    patient.execute(variantPack);

    CpuFpReferenceValidation<float> cpuRefOutputValidation(static_cast<float>(epsilon),
                                                           static_cast<float>(epsilon));

    EXPECT_TRUE(cpuRefOutputValidation.allClose(directTensorBundle.outputTensor.memory(),
                                                planTensorBundle.outputTensor.memory()));
}

TEST(TestBatchnormFwdInferencePlanBuilder, PlanConstruction)
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<int64_t> dims = {1, 3, 224, 224};
    std::vector<int64_t> strides = {150528, 50176, 224, 1};
    auto attributeOffset
        = CreateTensorAttributesDirect(builder, 1, "x", DataType::FLOAT, &strides, &dims);
    builder.Finish(attributeOffset);
    auto tensorAttr = flatbuffers::GetRoot<TensorAttributes>(builder.GetBufferPointer());

    // Create NodeT and set BatchnormInferenceAttributes
    auto bnormAttributes
        = hipdnn_sdk::data_objects::CreateBatchnormInferenceAttributes(builder,
                                                                       1, // x uid
                                                                       5, // mean uid
                                                                       6, // inv_variance uid
                                                                       3, // scale uid
                                                                       4, // bias uid
                                                                       2 // y uid
        );

    auto node = hipdnn_sdk::data_objects::CreateNodeDirect(
        builder,
        "batchnorm",
        hipdnn_sdk::data_objects::NodeAttributes::BatchnormInferenceAttributes,
        bnormAttributes.Union());
    builder.Finish(node);
    auto nodeRoot = flatbuffers::GetRoot<Node>(builder.GetBufferPointer());

    std::unordered_map<int64_t, const hipdnn_sdk::data_objects::TensorAttributes*> tensorMap;
    tensorMap[1] = tensorAttr;
    tensorMap[2] = tensorAttr;
    tensorMap[3] = tensorAttr;
    tensorMap[4] = tensorAttr;
    tensorMap[5] = tensorAttr;
    tensorMap[6] = tensorAttr;
    MockGraph mockGraph;
    EXPECT_CALL(mockGraph, getTensorMap()).WillRepeatedly(::testing::ReturnRef(tensorMap));

    BatchnormFwdInferencePlanBuilder<DataType::FLOAT, DataType::FLOAT, DataType::FLOAT> patient;
    auto builtPlan = patient.buildNodePlan(mockGraph, *nodeRoot);

    bool result = dynamic_cast<BatchnormFwdPlan<float, float, float>*>(builtPlan.get()) != nullptr;
    EXPECT_TRUE(result);
}

TEST(TestBatchnormFwdInferencePlanBuilder, IsApplicable)
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<int64_t> dims = {1, 3, 224, 224};
    std::vector<int64_t> strides = {150528, 50176, 224, 1};
    auto attributeOffset
        = CreateTensorAttributesDirect(builder, 1, "x", DataType::FLOAT, &strides, &dims);
    builder.Finish(attributeOffset);
    auto tensorAttr = flatbuffers::GetRoot<TensorAttributes>(builder.GetBufferPointer());

    // Create NodeT and set BatchnormInferenceAttributes
    auto bnormAttributes
        = hipdnn_sdk::data_objects::CreateBatchnormInferenceAttributes(builder,
                                                                       1, // x uid
                                                                       5, // mean uid
                                                                       6, // inv_variance uid
                                                                       3, // scale uid
                                                                       4, // bias uid
                                                                       2 // y uid
        );

    auto node = hipdnn_sdk::data_objects::CreateNodeDirect(
        builder,
        "batchnorm",
        hipdnn_sdk::data_objects::NodeAttributes::BatchnormInferenceAttributes,
        bnormAttributes.Union());
    builder.Finish(node);
    auto nodeRoot = flatbuffers::GetRoot<Node>(builder.GetBufferPointer());

    std::unordered_map<int64_t, const hipdnn_sdk::data_objects::TensorAttributes*> tensorMap;
    tensorMap[1] = tensorAttr;
    tensorMap[2] = tensorAttr;
    tensorMap[3] = tensorAttr;
    tensorMap[4] = tensorAttr;
    tensorMap[5] = tensorAttr;
    tensorMap[6] = tensorAttr;

    BatchnormFwdInferencePlanBuilder<DataType::FLOAT, DataType::FLOAT, DataType::FLOAT>
        floatPlanBuilder;
    EXPECT_TRUE(floatPlanBuilder.isApplicable(*nodeRoot, tensorMap));

    BatchnormFwdInferencePlanBuilder<DataType::FLOAT, DataType::HALF, DataType::FLOAT>
        badTypesPlanBuilder;
    EXPECT_FALSE(badTypesPlanBuilder.isApplicable(*nodeRoot, tensorMap));

    //remove a tensor and check
    tensorMap.erase(6);
    EXPECT_FALSE(floatPlanBuilder.isApplicable(*nodeRoot, tensorMap));
}
