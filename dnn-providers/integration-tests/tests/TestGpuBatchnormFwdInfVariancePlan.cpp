// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include "ScalarTestUtils.hpp"
#include "harness/gpu-graph-executor/detail/GpuPlanBuilderRegistry.hpp"
#include <cstdint>
#include <hipdnn_data_sdk/utilities/Constants.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/TestTolerances.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/CpuReferenceGraphExecutor.hpp>
#include <vector>

#include "harness/gpu-graph-executor/detail/GpuBatchnormFwdInfVariancePlan.hpp"

using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_integration_tests::gpu_graph_executor::detail;
using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_integration_tests::test_utils;
namespace
{

flatbuffers::FlatBufferBuilder
    createValidBatchnormWithVarianceInferenceGraph(const int64_t xUid,
                                                   const int64_t yUid,
                                                   const int64_t scaleUid,
                                                   const int64_t biasUid,
                                                   const int64_t meanUid,
                                                   const int64_t varianceUid,
                                                   const int64_t epsilonUid,
                                                   const std::vector<int64_t>& ioDims,
                                                   const std::vector<int64_t>& channelOnlyDims,
                                                   const std::vector<int64_t>& ioStrides,
                                                   const std::vector<int64_t>& channelOnlyStrides,
                                                   double epsilon,
                                                   const DataType ioDataType,
                                                   const DataType scaleBiasDataType,
                                                   const DataType meanVarianceDataType,
                                                   const DataType computeDataType,
                                                   const DataType epsilonDataType)
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<flatbuffers::Offset<TensorAttributes>> tensors;

    tensors.push_back(
        CreateTensorAttributesDirect(builder, xUid, "x", ioDataType, &ioStrides, &ioDims));
    tensors.push_back(
        CreateTensorAttributesDirect(builder, yUid, "y", ioDataType, &ioStrides, &ioDims));
    tensors.push_back(CreateTensorAttributesDirect(
        builder, scaleUid, "scale", scaleBiasDataType, &channelOnlyStrides, &channelOnlyDims));
    tensors.push_back(CreateTensorAttributesDirect(
        builder, biasUid, "bias", scaleBiasDataType, &channelOnlyStrides, &channelOnlyDims));
    tensors.push_back(CreateTensorAttributesDirect(
        builder, meanUid, "mean", meanVarianceDataType, &channelOnlyStrides, &channelOnlyDims));
    tensors.push_back(CreateTensorAttributesDirect(builder,
                                                   varianceUid,
                                                   "variance",
                                                   meanVarianceDataType,
                                                   &channelOnlyStrides,
                                                   &channelOnlyDims));
    tensors.push_back(
        createScalarTensorAttributes(builder, epsilonUid, epsilon, epsilonDataType, "epsilon"));

    auto attrs = CreateBatchnormInferenceAttributesVarianceExt(
        builder, xUid, meanUid, varianceUid, scaleUid, biasUid, yUid, epsilonUid);
    std::vector<flatbuffers::Offset<Node>> nodes;
    nodes.push_back(CreateNodeDirect(builder,
                                     "batchnormWithVariance_Node",
                                     computeDataType,
                                     NodeAttributes::BatchnormInferenceAttributesVarianceExt,
                                     attrs.Union()));
    auto graph = CreateGraphDirect(builder,
                                   "batchnormWithVariance_Graph",
                                   computeDataType,
                                   ioDataType,
                                   ioDataType,
                                   &tensors,
                                   &nodes);
    builder.Finish(graph);
    return builder;
}

flatbuffers::FlatBufferBuilder createBatchnormWithVarianceGraphWithRuntimeEpsilon()
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<flatbuffers::Offset<TensorAttributes>> tensors;

    const std::vector<int64_t> dims = {1, 2, 3, 4};
    const std::vector<int64_t> strides = {24, 12, 4, 1};
    const std::vector<int64_t> perChannelDims = {1, 2, 1, 1};
    const std::vector<int64_t> perChannelStrides = {2, 1, 1, 1};
    const std::vector<int64_t> scalarDims = {1};
    const Float32Value epsilonValue(1e-5f);

    tensors.push_back(
        CreateTensorAttributesDirect(builder, 1, "x", DataType::FLOAT, &strides, &dims));
    tensors.push_back(
        CreateTensorAttributesDirect(builder, 2, "y", DataType::FLOAT, &strides, &dims));
    tensors.push_back(CreateTensorAttributesDirect(
        builder, 3, "scale", DataType::FLOAT, &perChannelStrides, &perChannelDims));
    tensors.push_back(CreateTensorAttributesDirect(
        builder, 4, "bias", DataType::FLOAT, &perChannelStrides, &perChannelDims));
    tensors.push_back(CreateTensorAttributesDirect(
        builder, 5, "mean", DataType::FLOAT, &perChannelStrides, &perChannelDims));
    tensors.push_back(CreateTensorAttributesDirect(
        builder, 6, "variance", DataType::FLOAT, &perChannelStrides, &perChannelDims));
    tensors.push_back(CreateTensorAttributesDirect(builder,
                                                   7,
                                                   "epsilon",
                                                   DataType::FLOAT,
                                                   &scalarDims,
                                                   &scalarDims,
                                                   false,
                                                   TensorValue::Float32Value,
                                                   builder.CreateStruct(epsilonValue).Union(),
                                                   true));

    auto attrs = CreateBatchnormInferenceAttributesVarianceExt(builder, 1, 5, 6, 3, 4, 2, 7);

    std::vector<flatbuffers::Offset<Node>> nodes;
    nodes.push_back(CreateNodeDirect(builder,
                                     "batchnormWithVariance",
                                     DataType::FLOAT,
                                     NodeAttributes::BatchnormInferenceAttributesVarianceExt,
                                     attrs.Union()));

    auto graph = CreateGraphDirect(
        builder, "test", DataType::FLOAT, DataType::HALF, DataType::BFLOAT16, &tensors, &nodes);
    builder.Finish(graph);
    return builder;
}

} // namespace

TEST(TestGpuBatchnormFwdInfVariancePlanBuilder, PlanConstruction)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormWithVarianceInferenceGraph();
    auto graph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    const GpuBatchnormFwdInfVariancePlanBuilder<DataType::FLOAT,
                                                DataType::FLOAT,
                                                DataType::FLOAT,
                                                DataType::FLOAT,
                                                DataType::FLOAT>
        patient;

    auto builtPlan = patient.buildNodePlan(graph, graph.getNode(0));

    const bool result
        = dynamic_cast<GpuBatchnormFwdInfVariancePlan<float, float, float, float, float>*>(
              builtPlan.get())
          != nullptr;
    EXPECT_TRUE(result);
}

TEST(TestGpuBatchnormFwdInfVariancePlanBuilder, IsApplicable)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormWithVarianceInferenceGraph();
    auto graph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    const GpuBatchnormFwdInfVariancePlanBuilder<DataType::FLOAT,
                                                DataType::FLOAT,
                                                DataType::FLOAT,
                                                DataType::FLOAT,
                                                DataType::FLOAT>
        floatPlanBuilder;
    EXPECT_TRUE(floatPlanBuilder.isApplicable(graph.getNode(0), graph.getTensorMap()));

    // Half builder must not be applicable to a float graph
    const GpuBatchnormFwdInfVariancePlanBuilder<DataType::HALF,
                                                DataType::HALF,
                                                DataType::HALF,
                                                DataType::HALF,
                                                DataType::HALF>
        halfPlanBuilder;
    EXPECT_FALSE(halfPlanBuilder.isApplicable(graph.getNode(0), graph.getTensorMap()));

    // Missing input tensor must make the plan inapplicable
    auto tensorMapCopy = graph.getTensorMap();
    const auto* nodeAttributes
        = graph.getNode(0).attributes_as_BatchnormInferenceAttributesVarianceExt();
    EXPECT_NE(nodeAttributes, nullptr);
    tensorMapCopy.erase(nodeAttributes->x_tensor_uid());
    EXPECT_FALSE(floatPlanBuilder.isApplicable(graph.getNode(0), tensorMapCopy));
}

TEST(TestGpuBatchnormFwdInfVariancePlanBuilder, BuildNodePlanThrowsForWrongAttributesType)
{
    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormGraph();
    auto graph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    const GpuBatchnormFwdInfVariancePlanBuilder<DataType::FLOAT,
                                                DataType::FLOAT,
                                                DataType::FLOAT,
                                                DataType::FLOAT,
                                                DataType::FLOAT>
        patient;

    EXPECT_THROW(patient.buildNodePlan(graph, graph.getNode(0)), std::runtime_error);
}

TEST(TestGpuBatchnormFwdInfVariancePlanBuilder, IsApplicableReturnsFalseForWrongAttributesType)
{
    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormGraph();
    auto graph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    const GpuBatchnormFwdInfVariancePlanBuilder<DataType::FLOAT,
                                                DataType::FLOAT,
                                                DataType::FLOAT,
                                                DataType::FLOAT,
                                                DataType::FLOAT>
        patient;

    EXPECT_FALSE(patient.isApplicable(graph.getNode(0), graph.getTensorMap()));
}

TEST(TestGpuBatchnormFwdInfVariancePlanBuilder, IsApplicableFalseWhenEpsilonIsRuntimePassByValue)
{
    auto builder = createBatchnormWithVarianceGraphWithRuntimeEpsilon();
    auto graph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    const GpuBatchnormFwdInfVariancePlanBuilder<DataType::FLOAT,
                                                DataType::FLOAT,
                                                DataType::FLOAT,
                                                DataType::FLOAT,
                                                DataType::FLOAT>
        patient;

    EXPECT_FALSE(patient.isApplicable(graph.getNode(0), graph.getTensorMap()));
}

TEST(TestGpuBatchnormFwdInfVariancePlanBuilder,
     IsApplicableAcceptsEpsilonTypeDifferentFromComputeType)
{
    // Helper uses float for epsilon type, which differs for double type we've passed for compute
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormWithVarianceInferenceGraph(
        {150528, 50176, 224, 1},
        {1, 3, 224, 224},
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT, // input
        hipdnn_flatbuffers_sdk::data_objects::DataType::DOUBLE // compute
    );

    auto graph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    const GpuBatchnormFwdInfVariancePlanBuilder<DataType::FLOAT,
                                                DataType::FLOAT,
                                                DataType::FLOAT,
                                                DataType::FLOAT,
                                                DataType::DOUBLE>
        planBuilder;

    EXPECT_TRUE(planBuilder.isApplicable(graph.getNode(0), graph.getTensorMap()));
}

// ====================================================
// Templated helper for plan execution vs CPU reference
// ====================================================

namespace
{

template <typename IOType, typename ScaleBiasType, typename MeanVarType, typename ComputeType>
void runPlanExecuteVsCpuRef(const std::vector<int64_t>& dims,
                            const TensorLayout& layout,
                            float tolerance,
                            DataType epsilonDataType = DataType::UNSET)
{

    const auto ioStrides = generateStrides(dims, layout.strideOrder);
    const auto perChannelDims = getDerivedShape(dims); // {1, C, 1, 1}
    const auto perChannelStrides = generateStrides(perChannelDims, layout.strideOrder);

    auto ioDataType = nativeTypeToDataType<IOType>();
    auto scaleBiasDataType = nativeTypeToDataType<ScaleBiasType>();
    auto meanVarianceDataType = nativeTypeToDataType<MeanVarType>();
    auto computeDataType = nativeTypeToDataType<ComputeType>();
    if(epsilonDataType == DataType::UNSET)
    {
        epsilonDataType = computeDataType;
    }

    constexpr int64_t X_UID = 1;
    constexpr int64_t Y_UID = 2;
    constexpr int64_t SCALE_UID = 3;
    constexpr int64_t BIAS_UID = 4;
    constexpr int64_t MEAN_UID = 5;
    constexpr int64_t VARIANCE_UID = 6;
    constexpr int64_t EPSILON_UID = 7;

    // NOLINTNEXTLINE(readability-suspicious-call-argument)
    auto graphBuilder = createValidBatchnormWithVarianceInferenceGraph(X_UID,
                                                                       Y_UID,
                                                                       SCALE_UID,
                                                                       BIAS_UID,
                                                                       MEAN_UID,
                                                                       VARIANCE_UID,
                                                                       EPSILON_UID,
                                                                       dims,
                                                                       perChannelDims,
                                                                       ioStrides,
                                                                       perChannelStrides,
                                                                       BATCHNORM_DEFAULT_EPSILON,
                                                                       ioDataType,
                                                                       scaleBiasDataType,
                                                                       meanVarianceDataType,
                                                                       computeDataType,
                                                                       epsilonDataType);

    auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        graphBuilder.GetBufferPointer(), graphBuilder.GetSize());

    const auto* nodeAttributes
        = graphWrap.getNode(0).attributes_as_BatchnormInferenceAttributesVarianceExt();
    const auto& tensorMap = graphWrap.getTensorMap();

    // Construct the GPU plan directly from the tensor attributes
    GpuBatchnormFwdInfVarianceParams params(*tensorMap.at(nodeAttributes->x_tensor_uid()),
                                            *tensorMap.at(nodeAttributes->scale_tensor_uid()),
                                            *tensorMap.at(nodeAttributes->bias_tensor_uid()),
                                            *tensorMap.at(nodeAttributes->mean_tensor_uid()),
                                            *tensorMap.at(nodeAttributes->variance_tensor_uid()),
                                            *tensorMap.at(nodeAttributes->y_tensor_uid()),
                                            *tensorMap.at(nodeAttributes->epsilon_tensor_uid()));

    GpuBatchnormFwdInfVariancePlan<IOType, ScaleBiasType, MeanVarType, IOType, ComputeType> gpuPlan(
        std::move(params));

    // Prepare tensors and fill with random data
    Tensor<IOType> xTensor(dims, ioStrides);
    Tensor<IOType> cpuY(dims, ioStrides);
    Tensor<IOType> gpuY(dims, ioStrides);
    Tensor<ScaleBiasType> scaleTensor(perChannelDims, perChannelStrides);
    Tensor<ScaleBiasType> biasTensor(perChannelDims, perChannelStrides);
    Tensor<MeanVarType> meanTensor(perChannelDims, perChannelStrides);
    Tensor<MeanVarType> varianceTensor(perChannelDims, perChannelStrides);
    Tensor<ComputeType> epsilonTensor(std::vector<int64_t>{1}, std::vector<int64_t>{1});

    constexpr float MEAN_RANGE = 0.5f;
    constexpr float SCALE_BIAS_RANGE = 1.0f;
    unsigned int seed = 42;
    xTensor.fillWithRandomValues(static_cast<IOType>(-1), static_cast<IOType>(1), seed++);
    scaleTensor.fillWithRandomValues(static_cast<ScaleBiasType>(-SCALE_BIAS_RANGE),
                                     static_cast<ScaleBiasType>(SCALE_BIAS_RANGE),
                                     seed++);
    biasTensor.fillWithRandomValues(static_cast<ScaleBiasType>(-SCALE_BIAS_RANGE),
                                    static_cast<ScaleBiasType>(SCALE_BIAS_RANGE),
                                    seed++);
    meanTensor.fillWithRandomValues(
        static_cast<MeanVarType>(-MEAN_RANGE), static_cast<MeanVarType>(MEAN_RANGE), seed++);
    varianceTensor.fillWithRandomValues(
        static_cast<MeanVarType>(0.1f), static_cast<MeanVarType>(1.f), seed++);
    epsilonTensor.fillWithValue(static_cast<ComputeType>(BATCHNORM_DEFAULT_EPSILON));

    // Run the GPU reference executor
    std::unordered_map<int64_t, void*> gpuVariantPack;
    gpuVariantPack[X_UID] = xTensor.rawDeviceData();
    gpuVariantPack[Y_UID] = gpuY.rawDeviceData();
    gpuVariantPack[SCALE_UID] = scaleTensor.rawDeviceData();
    gpuVariantPack[BIAS_UID] = biasTensor.rawDeviceData();
    gpuVariantPack[MEAN_UID] = meanTensor.rawDeviceData();
    gpuVariantPack[VARIANCE_UID] = varianceTensor.rawDeviceData();
    gpuVariantPack[EPSILON_UID] = epsilonTensor.rawDeviceData();

    gpuPlan.execute(gpuVariantPack);
    gpuY.markDeviceModified();

    // Run the CPU reference executor
    std::unordered_map<int64_t, void*> cpuVariantPack;
    cpuVariantPack[nodeAttributes->x_tensor_uid()] = xTensor.rawHostData();
    cpuVariantPack[nodeAttributes->y_tensor_uid()] = cpuY.rawHostData();
    cpuVariantPack[nodeAttributes->scale_tensor_uid()] = scaleTensor.rawHostData();
    cpuVariantPack[nodeAttributes->bias_tensor_uid()] = biasTensor.rawHostData();
    cpuVariantPack[nodeAttributes->mean_tensor_uid()] = meanTensor.rawHostData();
    cpuVariantPack[nodeAttributes->variance_tensor_uid()] = varianceTensor.rawHostData();
    cpuVariantPack[nodeAttributes->epsilon_tensor_uid()] = epsilonTensor.rawHostData();

    CpuReferenceGraphExecutor cpuExecutor;
    cpuExecutor.execute(graphBuilder.GetBufferPointer(), graphBuilder.GetSize(), cpuVariantPack);
    cpuY.markHostModified();

    // Compare
    const auto* cpuYData = static_cast<const IOType*>(cpuY.rawHostData());
    const auto* gpuYData = static_cast<const IOType*>(gpuY.rawHostData());
    for(size_t i = 0; i < cpuY.elementCount(); ++i)
    {
        EXPECT_NEAR(static_cast<float>(gpuYData[i]), static_cast<float>(cpuYData[i]), tolerance)
            << "Mismatch in y at index " << i;
    }
}

} // anonymous namespace

// ============================================================================
// FP32 plan execution tests
// ============================================================================

TEST(TestGpuBatchnormFwdInfVariancePlan, ExecutePlanNchw)
{
    SKIP_IF_NO_DEVICES();

    runPlanExecuteVsCpuRef<float, float, float, float>(
        {2, 3, 4, 4}, TensorLayout::NCHW, batchnorm::getToleranceInferenceWithVariance<float>());
}

TEST(TestGpuBatchnormFwdInfVariancePlan, ExecutePlanNhwc)
{
    SKIP_IF_NO_DEVICES();

    runPlanExecuteVsCpuRef<float, float, float, float>(
        {2, 3, 4, 4}, TensorLayout::NHWC, batchnorm::getToleranceInferenceWithVariance<float>());
}

TEST(TestGpuBatchnormFwdInfVariancePlan, ExecutePlanWithDoubleEpsilon)
{
    SKIP_IF_NO_DEVICES();

    runPlanExecuteVsCpuRef<float, float, float, float>(
        {2, 3, 4, 4},
        TensorLayout::NHWC,
        batchnorm::getToleranceInferenceWithVariance<float>(),
        DataType::DOUBLE);
}

// ============================================================================
// FP16 plan execution tests
// ============================================================================

TEST(TestGpuBatchnormFwdInfVariancePlanFp16, ExecutePlanNchw)
{
    SKIP_IF_NO_DEVICES();

    runPlanExecuteVsCpuRef<half, half, half, float>(
        {2, 3, 4, 4}, TensorLayout::NCHW, batchnorm::getToleranceInferenceWithVariance<half>());
}

TEST(TestGpuBatchnormFwdInfVariancePlanFp16, ExecutePlanNhwc)
{
    SKIP_IF_NO_DEVICES();

    runPlanExecuteVsCpuRef<half, half, half, float>(
        {2, 3, 4, 4}, TensorLayout::NHWC, batchnorm::getToleranceInferenceWithVariance<half>());
}

/// ============================================================================
// BFP16 plan execution tests
// ============================================================================

TEST(TestGpuBatchnormFwdInfVariancePlanBfp16, ExecutePlanNchw)
{
    SKIP_IF_NO_DEVICES();

    runPlanExecuteVsCpuRef<bfloat16, bfloat16, bfloat16, float>(
        {2, 3, 4, 4}, TensorLayout::NCHW, batchnorm::getToleranceInferenceWithVariance<bfloat16>());
}

TEST(TestGpuBatchnormFwdInfVariancePlanBfp16, ExecutePlanNhwc)
{
    SKIP_IF_NO_DEVICES();

    runPlanExecuteVsCpuRef<bfloat16, bfloat16, bfloat16, float>(
        {2, 3, 4, 4}, TensorLayout::NHWC, batchnorm::getToleranceInferenceWithVariance<bfloat16>());
}

// ============================================================================
// Rejection test — unregistered signature
// ============================================================================

TEST(TestGpuBatchnormFwdInfVariancePlanBuilder, UnregisteredSignatureThrows)
{
    GpuPlanBuilderRegistry registry;

    const GpuBatchnormFwdInfVarianceSignatureKey unregisteredKey{
        DataType::INT8, DataType::INT8, DataType::INT8, DataType::INT8, DataType::FLOAT};

    EXPECT_THROW(registry.getPlanBuilder(unregisteredKey), std::runtime_error);
}
