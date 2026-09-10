// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "harness/gpu-graph-executor/detail/GpuBatchnormFwdInfPlan.hpp"
#include "harness/gpu-graph-executor/detail/GpuPlanBuilderRegistry.hpp"
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/TestTolerances.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/CpuReferenceGraphExecutor.hpp>

using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_integration_tests::gpu_graph_executor::detail;
using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_data_sdk::utilities;

namespace
{

flatbuffers::FlatBufferBuilder
    createValidBatchnormInferenceGraph(const int64_t xUid,
                                       const int64_t yUid,
                                       const int64_t scaleUid,
                                       const int64_t biasUid,
                                       const int64_t meanUid,
                                       const int64_t invVarianceUid,
                                       const std::vector<int64_t>& ioDims,
                                       const std::vector<int64_t>& channelOnlyDims,
                                       const std::vector<int64_t>& ioStrides,
                                       const std::vector<int64_t>& channelOnlyStrides,
                                       const DataType ioDataType,
                                       const DataType scaleBiasDataType,
                                       const DataType meanInvVarianceDataType,
                                       const DataType computeDataType)
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
        builder, meanUid, "mean", meanInvVarianceDataType, &channelOnlyStrides, &channelOnlyDims));
    tensors.push_back(CreateTensorAttributesDirect(builder,
                                                   invVarianceUid,
                                                   "variance",
                                                   meanInvVarianceDataType,
                                                   &channelOnlyStrides,
                                                   &channelOnlyDims));

    auto attrs = CreateBatchnormInferenceAttributes(
        builder, xUid, meanUid, invVarianceUid, scaleUid, biasUid, yUid);
    std::vector<flatbuffers::Offset<Node>> nodes;
    nodes.push_back(CreateNodeDirect(builder,
                                     "batchnormInf_Node",
                                     computeDataType,
                                     NodeAttributes::BatchnormInferenceAttributes,
                                     attrs.Union()));
    auto graph = CreateGraphDirect(
        builder, "batchnormInf_Graph", computeDataType, ioDataType, ioDataType, &tensors, &nodes);
    builder.Finish(graph);
    return builder;
}

} // namespace

TEST(TestGpuBatchnormFwdInfPlanBuilder, PlanConstruction)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormInferenceGraph();
    auto graph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    const GpuBatchnormFwdInfPlanBuilder<DataType::FLOAT,
                                        DataType::FLOAT,
                                        DataType::FLOAT,
                                        DataType::FLOAT,
                                        DataType::FLOAT>
        patient;

    auto builtPlan = patient.buildNodePlan(graph, graph.getNode(0));

    const bool result
        = dynamic_cast<GpuBatchnormFwdInfPlan<float, float, float, float, float>*>(builtPlan.get())
          != nullptr;
    EXPECT_TRUE(result);
}

TEST(TestGpuBatchnormFwdInfPlanBuilder, IsApplicable)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormInferenceGraph();
    auto graph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    const GpuBatchnormFwdInfPlanBuilder<DataType::FLOAT,
                                        DataType::FLOAT,
                                        DataType::FLOAT,
                                        DataType::FLOAT,
                                        DataType::FLOAT>
        floatPlanBuilder;
    EXPECT_TRUE(floatPlanBuilder.isApplicable(graph.getNode(0), graph.getTensorMap()));

    const GpuBatchnormFwdInfPlanBuilder<DataType::HALF,
                                        DataType::HALF,
                                        DataType::HALF,
                                        DataType::HALF,
                                        DataType::HALF>
        halfPlanBuilder;
    EXPECT_FALSE(halfPlanBuilder.isApplicable(graph.getNode(0), graph.getTensorMap()));

    auto tensorMapCopy = graph.getTensorMap();
    const auto* nodeAttributes = graph.getNode(0).attributes_as_BatchnormInferenceAttributes();
    ASSERT_NE(nodeAttributes, nullptr);
    tensorMapCopy.erase(nodeAttributes->x_tensor_uid());
    EXPECT_FALSE(floatPlanBuilder.isApplicable(graph.getNode(0), tensorMapCopy));
}

TEST(TestGpuBatchnormFwdInfPlanBuilder, BuildNodePlanThrowsForWrongAttributesType)
{
    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormGraph();
    auto graph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    const GpuBatchnormFwdInfPlanBuilder<DataType::FLOAT,
                                        DataType::FLOAT,
                                        DataType::FLOAT,
                                        DataType::FLOAT,
                                        DataType::FLOAT>
        patient;

    EXPECT_THROW(patient.buildNodePlan(graph, graph.getNode(0)), std::runtime_error);
}

TEST(TestGpuBatchnormFwdInfPlanBuilder, IsApplicableReturnsFalseForWrongAttributesType)
{
    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormGraph();
    auto graph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    const GpuBatchnormFwdInfPlanBuilder<DataType::FLOAT,
                                        DataType::FLOAT,
                                        DataType::FLOAT,
                                        DataType::FLOAT,
                                        DataType::FLOAT>
        patient;

    EXPECT_FALSE(patient.isApplicable(graph.getNode(0), graph.getTensorMap()));
}

// ====================================================
// Templated helper for plan execution vs CPU reference
// ====================================================

namespace
{

template <typename IOType, typename ScaleBiasType, typename MeanVarType, typename ComputeType>
void runPlanExecuteVsCpuRef(const std::vector<int64_t>& dims,
                            const TensorLayout& layout,
                            float tolerance)
{

    const auto ioStrides = generateStrides(dims, layout.strideOrder);
    const auto perChannelDims = getDerivedShape(dims); // {1, C, 1, 1}
    const auto perChannelStrides = generateStrides(perChannelDims, layout.strideOrder);

    auto ioDataType = nativeTypeToDataType<IOType>();
    auto scaleBiasDataType = nativeTypeToDataType<ScaleBiasType>();
    auto meanInvVarianceDataType = nativeTypeToDataType<MeanVarType>();
    auto computeDataType = nativeTypeToDataType<ComputeType>();

    constexpr int64_t X_UID = 1;
    constexpr int64_t Y_UID = 2;
    constexpr int64_t SCALE_UID = 3;
    constexpr int64_t BIAS_UID = 4;
    constexpr int64_t MEAN_UID = 5;
    constexpr int64_t INV_VARIANCE_UID = 6;

    auto graphBuilder = createValidBatchnormInferenceGraph(X_UID,
                                                           Y_UID,
                                                           SCALE_UID,
                                                           BIAS_UID,
                                                           MEAN_UID,
                                                           INV_VARIANCE_UID,
                                                           dims,
                                                           perChannelDims,
                                                           ioStrides,
                                                           perChannelStrides,
                                                           ioDataType,
                                                           scaleBiasDataType,
                                                           meanInvVarianceDataType,
                                                           computeDataType);

    auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        graphBuilder.GetBufferPointer(), graphBuilder.GetSize());

    const auto* nodeAttributes = graphWrap.getNode(0).attributes_as_BatchnormInferenceAttributes();
    const auto& tensorMap = graphWrap.getTensorMap();

    // Construct the GPU plan directly from the tensor attributes
    GpuBatchnormFwdInfParams params(*tensorMap.at(nodeAttributes->x_tensor_uid()),
                                    *tensorMap.at(nodeAttributes->scale_tensor_uid()),
                                    *tensorMap.at(nodeAttributes->bias_tensor_uid()),
                                    *tensorMap.at(nodeAttributes->mean_tensor_uid()),
                                    *tensorMap.at(nodeAttributes->inv_variance_tensor_uid()),
                                    *tensorMap.at(nodeAttributes->y_tensor_uid()));

    GpuBatchnormFwdInfPlan<IOType, ScaleBiasType, MeanVarType, IOType, ComputeType> gpuPlan(
        std::move(params));

    // Prepare tensors and fill with random data
    Tensor<IOType> xTensor(dims, ioStrides);
    Tensor<IOType> cpuY(dims, ioStrides);
    Tensor<IOType> gpuY(dims, ioStrides);
    Tensor<ScaleBiasType> scaleTensor(perChannelDims, perChannelStrides);
    Tensor<ScaleBiasType> biasTensor(perChannelDims, perChannelStrides);
    Tensor<MeanVarType> meanTensor(perChannelDims, perChannelStrides);
    Tensor<MeanVarType> invVarianceTensor(perChannelDims, perChannelStrides);

    constexpr unsigned int SEED = 42;
    constexpr float MEAN = 0.5f;
    constexpr float INV_VARIANCE = 1.0f;
    constexpr float SCALE_BIAS_RANGE = 1.0f;

    xTensor.fillWithRandomValues(static_cast<IOType>(-1), static_cast<IOType>(1), SEED);
    scaleTensor.fillWithRandomValues(static_cast<ScaleBiasType>(-SCALE_BIAS_RANGE),
                                     static_cast<ScaleBiasType>(SCALE_BIAS_RANGE),
                                     SEED + 1);
    biasTensor.fillWithRandomValues(static_cast<ScaleBiasType>(-SCALE_BIAS_RANGE),
                                    static_cast<ScaleBiasType>(SCALE_BIAS_RANGE),
                                    SEED + 2);
    meanTensor.fillWithValue(static_cast<MeanVarType>(MEAN));
    invVarianceTensor.fillWithValue(static_cast<MeanVarType>(INV_VARIANCE));

    // Run the GPU reference executor
    std::unordered_map<int64_t, void*> gpuVariantPack;
    gpuVariantPack[X_UID] = xTensor.rawDeviceData();
    gpuVariantPack[Y_UID] = gpuY.rawDeviceData();
    gpuVariantPack[SCALE_UID] = scaleTensor.rawDeviceData();
    gpuVariantPack[BIAS_UID] = biasTensor.rawDeviceData();
    gpuVariantPack[MEAN_UID] = meanTensor.rawDeviceData();
    gpuVariantPack[INV_VARIANCE_UID] = invVarianceTensor.rawDeviceData();

    gpuPlan.execute(gpuVariantPack);
    gpuY.markDeviceModified();

    // Run the CPU reference executor
    std::unordered_map<int64_t, void*> cpuVariantPack;
    cpuVariantPack[nodeAttributes->x_tensor_uid()] = xTensor.rawHostData();
    cpuVariantPack[nodeAttributes->y_tensor_uid()] = cpuY.rawHostData();
    cpuVariantPack[nodeAttributes->scale_tensor_uid()] = scaleTensor.rawHostData();
    cpuVariantPack[nodeAttributes->bias_tensor_uid()] = biasTensor.rawHostData();
    cpuVariantPack[nodeAttributes->mean_tensor_uid()] = meanTensor.rawHostData();
    cpuVariantPack[nodeAttributes->inv_variance_tensor_uid()] = invVarianceTensor.rawHostData();

    CpuReferenceGraphExecutor cpuExecutor;
    cpuExecutor.execute(graphBuilder.GetBufferPointer(), graphBuilder.GetSize(), cpuVariantPack);
    cpuY.markHostModified();

    // Compare
    const auto* cpuYData = static_cast<const IOType*>(cpuY.rawHostData());
    const auto* gpuYData = static_cast<const IOType*>(gpuY.rawHostData());
    for(size_t i = 0; i < cpuY.elementCount(); ++i)
    {
        EXPECT_NEAR(static_cast<float>(gpuYData[i]), static_cast<float>(cpuYData[i]), tolerance)
            << "Mismatch in dx at index " << i;
    }
}

} // anonymous namespace

// ============================================================================
// FP32 plan execution tests
// ============================================================================

TEST(TestGpuBatchnormFwdInfPlan, ExecutePlanNchw)
{
    SKIP_IF_NO_DEVICES();

    runPlanExecuteVsCpuRef<float, float, float, float>(
        {2, 3, 4, 4}, TensorLayout::NCHW, batchnorm::getToleranceInference<float>());
}

TEST(TestGpuBatchnormFwdInfPlan, ExecutePlanNhwc)
{
    SKIP_IF_NO_DEVICES();

    runPlanExecuteVsCpuRef<float, float, float, float>(
        {2, 3, 4, 4}, TensorLayout::NHWC, batchnorm::getToleranceInference<float>());
}

// ============================================================================
// FP16 plan execution tests
// ============================================================================

TEST(TestGpuBatchnormFwdInfPlanFp16, ExecutePlanNchw)
{
    SKIP_IF_NO_DEVICES();

    runPlanExecuteVsCpuRef<half, half, half, float>(
        {2, 3, 4, 4}, TensorLayout::NCHW, batchnorm::getToleranceInference<half>());
}

TEST(TestGpuBatchnormFwdInfPlanFp16, ExecutePlanNhwc)
{
    SKIP_IF_NO_DEVICES();

    runPlanExecuteVsCpuRef<half, half, half, float>(
        {2, 3, 4, 4}, TensorLayout::NHWC, batchnorm::getToleranceInference<half>());
}

/// ============================================================================
// BFP16 plan execution tests
// ============================================================================

TEST(TestGpuBatchnormFwdInfPlanBfp16, ExecutePlanNchw)
{
    SKIP_IF_NO_DEVICES();

    runPlanExecuteVsCpuRef<bfloat16, bfloat16, bfloat16, float>(
        {2, 3, 4, 4}, TensorLayout::NCHW, batchnorm::getToleranceInference<bfloat16>());
}

TEST(TestGpuBatchnormFwdInfPlanBfp16, ExecutePlanNhwc)
{
    SKIP_IF_NO_DEVICES();

    runPlanExecuteVsCpuRef<bfloat16, bfloat16, bfloat16, float>(
        {2, 3, 4, 4}, TensorLayout::NHWC, batchnorm::getToleranceInference<bfloat16>());
}

// ============================================================================
// Rejection test — unregistered signature
// ============================================================================

TEST(TestGpuBatchnormFwdInfPlanBuilder, UnregisteredSignatureThrows)
{
    GpuPlanBuilderRegistry registry;

    const GpuBatchnormFwdInfSignatureKey unregisteredKey{
        DataType::INT8, DataType::INT8, DataType::INT8, DataType::INT8, DataType::FLOAT};

    EXPECT_THROW(registry.getPlanBuilder(unregisteredKey), std::runtime_error);
}
