// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <cstdint>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <stdexcept>
#include <unordered_map>

#include <hip/hip_runtime.h>
#include <hipdnn_data_sdk/utilities/Constants.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferDatatypeMapping.hpp>
#include <hipdnn_test_sdk/utilities/TestTolerances.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/CpuReferenceGraphExecutor.hpp>

#include "LayernormBwdGraphTestUtils.hpp"
#include "harness/gpu-graph-executor/detail/GpuLayernormBwdPlan.hpp"
#include "harness/gpu-graph-executor/detail/GpuLayernormBwdSignatureKey.hpp"
#include "harness/gpu-graph-executor/detail/GpuPlanBuilderRegistry.hpp"

using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_flatbuffers_sdk::flatbuffer_utilities;
using namespace hipdnn_integration_tests::test_utils;
using namespace hipdnn_integration_tests::gpu_graph_executor::detail;
using namespace hipdnn_test_sdk::utilities;

TEST(TestGpuLayernormBwdPlanBuilder, PlanConstruction)
{
    constexpr int64_t DY_UID = 10;
    constexpr int64_t X_UID = 11;
    constexpr int64_t SCALE_UID = 12;
    constexpr int64_t DX_UID = 13;
    constexpr int64_t DSCALE_UID = 14;
    constexpr int64_t DBIAS_UID = 15;
    constexpr int64_t EPSILON_UID = 16;
    constexpr int64_t MEAN_UID = 17;
    constexpr int64_t INV_VARIANCE_UID = 18;

    const std::vector<int64_t> ioDims = {2, 3, 4, 5};
    const TensorLayout layout = TensorLayout::NCHW;
    const auto epsilon = static_cast<float>(LAYERNORM_DEFAULT_EPSILON);
    const int64_t normalizedDimCount = 2;

    auto graphBuilder = createLayernormBwdGraph(DY_UID,
                                                X_UID,
                                                SCALE_UID,
                                                DX_UID,
                                                DSCALE_UID,
                                                DBIAS_UID,
                                                EPSILON_UID,
                                                MEAN_UID,
                                                INV_VARIANCE_UID,
                                                ioDims,
                                                layout,
                                                epsilon,
                                                normalizedDimCount,
                                                DataType::FLOAT,
                                                DataType::FLOAT,
                                                DataType::FLOAT,
                                                DataType::FLOAT,
                                                DataType::FLOAT);

    auto graphWrapper = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        graphBuilder.GetBufferPointer(), graphBuilder.GetSize());

    const GpuLayernormBwdPlanBuilder<DataType::FLOAT,
                                     DataType::FLOAT,
                                     DataType::FLOAT,
                                     DataType::FLOAT,
                                     DataType::FLOAT>
        planBuilder;

    auto builtPlan = planBuilder.buildNodePlan(graphWrapper, graphWrapper.getNode(0));

    const bool result
        = dynamic_cast<GpuLayernormBwdPlan<float, float, float, float, float>*>(builtPlan.get())
          != nullptr;
    EXPECT_TRUE(result);

    // Layernorm bwd builder should not be able to build a batchnorm bwd graph
    auto batchnormGraphBuilder = createValidBatchnormBwdGraph();

    auto batchnormGraphWrapper = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        batchnormGraphBuilder.GetBufferPointer(), batchnormGraphBuilder.GetSize());

    EXPECT_THROW(planBuilder.buildNodePlan(batchnormGraphWrapper, batchnormGraphWrapper.getNode(0)),
                 std::runtime_error);
}

TEST(TestGpuLayernormBwdPlanBuilder, IsApplicable)
{
    constexpr int64_t DY_UID = 10;
    constexpr int64_t X_UID = 11;
    constexpr int64_t SCALE_UID = 12;
    constexpr int64_t DX_UID = 13;
    constexpr int64_t DSCALE_UID = 14;
    constexpr int64_t DBIAS_UID = 15;
    constexpr int64_t EPSILON_UID = 16;
    constexpr int64_t MEAN_UID = 17;
    constexpr int64_t INV_VARIANCE_UID = 18;

    const std::vector<int64_t> ioDims = {2, 3, 4, 5};
    const TensorLayout layout = TensorLayout::NCHW;
    const auto epsilon = static_cast<float>(LAYERNORM_DEFAULT_EPSILON);
    const int64_t normalizedDimCount = 2;

    auto graphBuilder = createLayernormBwdGraph(DY_UID,
                                                X_UID,
                                                SCALE_UID,
                                                DX_UID,
                                                DSCALE_UID,
                                                DBIAS_UID,
                                                EPSILON_UID,
                                                MEAN_UID,
                                                INV_VARIANCE_UID,
                                                ioDims,
                                                layout,
                                                epsilon,
                                                normalizedDimCount,
                                                DataType::FLOAT,
                                                DataType::FLOAT,
                                                DataType::FLOAT,
                                                DataType::FLOAT,
                                                DataType::FLOAT);

    auto graphWrapper = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        graphBuilder.GetBufferPointer(), graphBuilder.GetSize());

    const GpuLayernormBwdPlanBuilder<DataType::FLOAT,
                                     DataType::FLOAT,
                                     DataType::FLOAT,
                                     DataType::FLOAT,
                                     DataType::FLOAT>
        floatPlanBuilder;

    EXPECT_TRUE(
        floatPlanBuilder.isApplicable(graphWrapper.getNode(0), graphWrapper.getTensorMap()));

    // Half builder should not be applicable for a float graph
    const GpuLayernormBwdPlanBuilder<DataType::HALF,
                                     DataType::HALF,
                                     DataType::HALF,
                                     DataType::HALF,
                                     DataType::FLOAT>
        halfPlanBuilder;

    EXPECT_FALSE(
        halfPlanBuilder.isApplicable(graphWrapper.getNode(0), graphWrapper.getTensorMap()));

    // Half epsilon should not be applicable for a graph with a float epsilon
    const GpuLayernormBwdPlanBuilder<DataType::FLOAT,
                                     DataType::FLOAT,
                                     DataType::FLOAT,
                                     DataType::FLOAT,
                                     DataType::HALF>
        halfEpsilonPlanBuilder;

    EXPECT_FALSE(
        halfEpsilonPlanBuilder.isApplicable(graphWrapper.getNode(0), graphWrapper.getTensorMap()));

    // Missing tensor should return false
    auto tensorMapCopy = graphWrapper.getTensorMap();
    tensorMapCopy.erase(X_UID);
    EXPECT_FALSE(floatPlanBuilder.isApplicable(graphWrapper.getNode(0), tensorMapCopy));

    // Layernorm bwd builder should not be applicable for a batchnorm bwd graph
    auto batchnormGraphBuilder = createValidBatchnormBwdGraph();

    auto batchnormGraphWrapper = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        batchnormGraphBuilder.GetBufferPointer(), batchnormGraphBuilder.GetSize());

    EXPECT_FALSE(floatPlanBuilder.isApplicable(batchnormGraphWrapper.getNode(0),
                                               batchnormGraphWrapper.getTensorMap()));
}

// ====================================================
// Templated helper for plan execution vs CPU reference
// ====================================================

namespace
{

template <typename DyType,
          typename ScaleBiasType,
          typename MeanInvVarianceType,
          typename DxType,
          typename ComputeType>
void runPlanExecuteVsCpuRef(const std::vector<int64_t>& ioDims,
                            const TensorLayout& layout,
                            int64_t normalizedDimCount,
                            float tolerance)
{
    const auto normalizedDim = static_cast<int64_t>(ioDims.size()) - normalizedDimCount;

    auto normDims = std::vector<int64_t>(ioDims.size(), 1);
    auto batchDims = std::vector<int64_t>(ioDims.size(), 1);
    for(size_t i = 0; i < ioDims.size(); ++i)
    {
        if(static_cast<int64_t>(i) < normalizedDim)
        {
            batchDims[i] = ioDims[i];
        }
        else
        {
            normDims[i] = ioDims[i];
        }
    }

    const auto ioStrides = generateStrides(ioDims, layout.strideOrder);
    const auto normStrides = generateStrides(normDims, layout.strideOrder);
    const auto batchStrides = generateStrides(batchDims, layout.strideOrder);

    constexpr int64_t DY_UID = 1;
    constexpr int64_t X_UID = 2;
    constexpr int64_t SCALE_UID = 3;
    constexpr int64_t DX_UID = 4;
    constexpr int64_t DSCALE_UID = 5;
    constexpr int64_t DBIAS_UID = 6;
    constexpr int64_t EPSILON_UID = 7;
    constexpr int64_t MEAN_UID = 8;
    constexpr int64_t INV_VARIANCE_UID = 9;

    auto dyDataType = nativeTypeToDataType<DyType>();
    auto dxDataType = nativeTypeToDataType<DxType>();
    auto scaleBiasDataType = nativeTypeToDataType<ScaleBiasType>();
    auto meanInvVarianceDataType = nativeTypeToDataType<MeanInvVarianceType>();
    auto computeDataType = nativeTypeToDataType<ComputeType>();

    const auto epsilon = static_cast<float>(LAYERNORM_DEFAULT_EPSILON);
    auto graphBuilder = createLayernormBwdGraph(DY_UID,
                                                X_UID,
                                                SCALE_UID,
                                                DX_UID,
                                                DSCALE_UID,
                                                DBIAS_UID,
                                                EPSILON_UID,
                                                MEAN_UID,
                                                INV_VARIANCE_UID,
                                                ioDims,
                                                layout,
                                                epsilon,
                                                normalizedDimCount,
                                                dyDataType,
                                                dxDataType,
                                                scaleBiasDataType,
                                                meanInvVarianceDataType,
                                                computeDataType);

    const GraphWrapper graphWrapper(graphBuilder.GetBufferPointer(), graphBuilder.GetSize());

    const auto* nodeAttributes
        = graphWrapper.getNode(0).attributes_as_LayernormBackwardAttributes();
    const auto& tensorMap = graphWrapper.getTensorMap();

    GpuLayernormBwdParams params(
        *tensorMap.at(nodeAttributes->dy_tensor_uid()),
        *tensorMap.at(nodeAttributes->x_tensor_uid()),
        *tensorMap.at(nodeAttributes->scale_tensor_uid()),
        *tensorMap.at(nodeAttributes->dx_tensor_uid()),
        *tensorMap.at(nodeAttributes->dscale_tensor_uid()),
        *tensorMap.at(nodeAttributes->dbias_tensor_uid()),
        normalizedDimCount,
        nodeAttributes->mean_tensor_uid().has_value()
            ? tensorMap.at(nodeAttributes->mean_tensor_uid().value())
            : nullptr,
        nodeAttributes->inv_variance_tensor_uid().has_value()
            ? tensorMap.at(nodeAttributes->inv_variance_tensor_uid().value())
            : nullptr,
        nodeAttributes->epsilon_tensor_uid().has_value()
            ? tensorMap.at(nodeAttributes->epsilon_tensor_uid().value())
            : nullptr);

    GpuLayernormBwdPlan<DyType, ScaleBiasType, MeanInvVarianceType, DxType, ComputeType> gpuPlan(
        std::move(params));

    Tensor<DyType> dyTensor(ioDims, ioStrides);
    Tensor<DxType> xTensor(ioDims, ioStrides);
    Tensor<ScaleBiasType> scaleTensor(normDims, normStrides);
    Tensor<ComputeType> epsilonTensor(std::vector<int64_t>{1}, std::vector<int64_t>{1});
    Tensor<MeanInvVarianceType> meanTensor(batchDims, batchStrides);
    Tensor<MeanInvVarianceType> rstdTensor(batchDims, batchStrides);
    Tensor<DxType> cpuDx(ioDims, ioStrides);
    Tensor<ScaleBiasType> cpuDscale(normDims, normStrides);
    Tensor<ScaleBiasType> cpuDbias(normDims, normStrides);

    constexpr unsigned int SEED = 42;
    dyTensor.fillWithRandomValues(static_cast<DyType>(-1.0), static_cast<DyType>(1.0), SEED);
    xTensor.fillWithRandomValues(static_cast<DxType>(-1.0), static_cast<DxType>(1.0), SEED + 1);
    scaleTensor.fillWithRandomValues(
        static_cast<ScaleBiasType>(-1.0), static_cast<ScaleBiasType>(1.0), SEED + 2);
    epsilonTensor.fillWithValue(static_cast<ComputeType>(LAYERNORM_DEFAULT_EPSILON));
    meanTensor.fillWithRandomValues(
        static_cast<MeanInvVarianceType>(-1.0), static_cast<MeanInvVarianceType>(1.0), SEED + 3);
    rstdTensor.fillWithRandomValues(
        static_cast<MeanInvVarianceType>(-1.0), static_cast<MeanInvVarianceType>(1.0), SEED + 4);

    Tensor<DxType> gpuDx(ioDims, ioStrides);
    Tensor<ScaleBiasType> gpuDscale(normDims, normStrides);
    Tensor<ScaleBiasType> gpuDbias(normDims, normStrides);

    std::unordered_map<int64_t, void*> gpuVariantPack;
    gpuVariantPack[DY_UID] = dyTensor.rawDeviceData();
    gpuVariantPack[X_UID] = xTensor.rawDeviceData();
    gpuVariantPack[SCALE_UID] = scaleTensor.rawDeviceData();
    gpuVariantPack[DX_UID] = gpuDx.rawDeviceData();
    gpuVariantPack[DSCALE_UID] = gpuDscale.rawDeviceData();
    gpuVariantPack[DBIAS_UID] = gpuDbias.rawDeviceData();
    gpuVariantPack[EPSILON_UID] = epsilonTensor.rawDeviceData();
    if(nodeAttributes->mean_tensor_uid().has_value())
    {
        gpuVariantPack[MEAN_UID] = meanTensor.rawDeviceData();
    }
    if(nodeAttributes->inv_variance_tensor_uid().has_value())
    {
        gpuVariantPack[INV_VARIANCE_UID] = rstdTensor.rawDeviceData();
    }

    gpuPlan.execute(gpuVariantPack);
    gpuDx.markDeviceModified();
    gpuDscale.markDeviceModified();
    gpuDbias.markDeviceModified();

    std::unordered_map<int64_t, void*> cpuVariantPack;
    cpuVariantPack[DY_UID] = dyTensor.rawHostData();
    cpuVariantPack[X_UID] = xTensor.rawHostData();
    cpuVariantPack[SCALE_UID] = scaleTensor.rawHostData();
    cpuVariantPack[DX_UID] = cpuDx.rawHostData();
    cpuVariantPack[DSCALE_UID] = cpuDscale.rawHostData();
    cpuVariantPack[DBIAS_UID] = cpuDbias.rawHostData();
    cpuVariantPack[EPSILON_UID] = epsilonTensor.rawHostData();
    if(nodeAttributes->mean_tensor_uid().has_value())
    {
        cpuVariantPack[MEAN_UID] = meanTensor.rawHostData();
    }
    if(nodeAttributes->inv_variance_tensor_uid().has_value())
    {
        cpuVariantPack[INV_VARIANCE_UID] = rstdTensor.rawHostData();
    }

    CpuReferenceGraphExecutor cpuExecutor;
    cpuExecutor.execute(graphBuilder.GetBufferPointer(), graphBuilder.GetSize(), cpuVariantPack);
    cpuDx.markHostModified();
    cpuDscale.markHostModified();
    cpuDbias.markHostModified();

    const auto* cpuDxData = static_cast<const DxType*>(cpuDx.rawHostData());
    const auto* gpuDxData = static_cast<const DxType*>(gpuDx.rawHostData());
    for(size_t i = 0; i < cpuDx.elementCount(); ++i)
    {
        EXPECT_NEAR(static_cast<float>(gpuDxData[i]), static_cast<float>(cpuDxData[i]), tolerance)
            << "Mismatch in dx at index " << i;
    }
    const auto* cpuDscaleData = static_cast<const ScaleBiasType*>(cpuDscale.rawHostData());
    const auto* gpuDscaleData = static_cast<const ScaleBiasType*>(gpuDscale.rawHostData());
    for(size_t i = 0; i < cpuDscale.elementCount(); ++i)
    {
        EXPECT_NEAR(
            static_cast<float>(gpuDscaleData[i]), static_cast<float>(cpuDscaleData[i]), tolerance)
            << "Mismatch in dscale at index " << i;
    }
    const auto* cpuDbiasData = static_cast<const ScaleBiasType*>(cpuDbias.rawHostData());
    const auto* gpuDbiasData = static_cast<const ScaleBiasType*>(gpuDbias.rawHostData());
    for(size_t i = 0; i < cpuDbias.elementCount(); ++i)
    {
        EXPECT_NEAR(
            static_cast<float>(gpuDbiasData[i]), static_cast<float>(cpuDbiasData[i]), tolerance)
            << "Mismatch in dbias at index " << i;
    }
}

} // namespace

// =========================
// FP32 plan execution tests
// =========================

TEST(TestGpuLayernormBwdPlanFp32, ExecutePlanNchw)
{
    SKIP_IF_NO_DEVICES();

    runPlanExecuteVsCpuRef<float, float, float, float, float>(
        {5, 4, 3, 2}, TensorLayout::NCHW, 3, layernorm::getTolerance<float>());
}

TEST(TestGpuLayernormBwdPlanFp32, ExecutePlanNhwc)
{
    SKIP_IF_NO_DEVICES();

    runPlanExecuteVsCpuRef<float, float, float, float, float>(
        {5, 4, 3, 2}, TensorLayout::NHWC, 3, layernorm::getTolerance<float>());
}

// =========================
// FP16 plan execution tests
// =========================

TEST(TestGpuLayernormBwdPlanFp16, ExecutePlanNchw)
{
    SKIP_IF_NO_DEVICES();

    runPlanExecuteVsCpuRef<half, half, half, half, float>(
        {5, 4, 3, 2}, TensorLayout::NCHW, 3, layernorm::getTolerance<half>());
}

TEST(TestGpuLayernormBwdPlanFp16, ExecutePlanNhwc)
{
    SKIP_IF_NO_DEVICES();

    runPlanExecuteVsCpuRef<half, half, half, half, float>(
        {5, 4, 3, 2}, TensorLayout::NHWC, 3, layernorm::getTolerance<half>());
}

// =========================
// BFP16 plan execution tests
// =========================

TEST(TestGpuLayernormBwdPlanBfp16, ExecutePlanNchw)
{
    SKIP_IF_NO_DEVICES();

    runPlanExecuteVsCpuRef<bfloat16, bfloat16, bfloat16, bfloat16, float>(
        {5, 4, 3, 2}, TensorLayout::NCHW, 3, layernorm::getTolerance<bfloat16>());
}

TEST(TestGpuLayernormBwdPlanBfp16, ExecutePlanNhwc)
{
    SKIP_IF_NO_DEVICES();

    runPlanExecuteVsCpuRef<bfloat16, bfloat16, bfloat16, bfloat16, float>(
        {5, 4, 3, 2}, TensorLayout::NHWC, 3, layernorm::getTolerance<bfloat16>());
}

// ============================================================================
// Rejection test — unregistered signature
// ============================================================================

TEST(TestGpuLayernormBwdPlanBuilder, UnregisteredSignatureThrows)
{
    GpuPlanBuilderRegistry registry;

    const GpuLayernormBwdSignatureKey unregisteredKey{
        DataType::INT8, DataType::INT8, DataType::INT8, DataType::INT8, DataType::FLOAT};

    EXPECT_THROW(registry.getPlanBuilder(unregisteredKey), std::runtime_error);
}
