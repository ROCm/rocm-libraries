// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <cstdint>
#include <functional>
#include <numeric>
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
    const double epsilon = LAYERNORM_DEFAULT_EPSILON;
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
    const double epsilon = LAYERNORM_DEFAULT_EPSILON;
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

    // Missing tensor should return false
    auto tensorMapCopy = graphWrapper.getTensorMap();
    tensorMapCopy.erase(X_UID);
    EXPECT_FALSE(floatPlanBuilder.isApplicable(graphWrapper.getNode(0), tensorMapCopy));
}

// ====================================================
// Templated helper for plan execution vs CPU reference
// ====================================================

namespace
{

inline size_t elementCount(const std::vector<int64_t>& dims)
{
    return std::accumulate(dims.begin(), dims.end(), size_t{1}, std::multiplies<>());
}

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

    const auto epsilon = LAYERNORM_DEFAULT_EPSILON;
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

    auto ioCount = elementCount(ioDims);
    auto normCount = elementCount(normDims);
    auto batchCount = elementCount(batchDims);

    Tensor<DyType> cpuDy(ioDims, ioStrides);
    Tensor<DxType> cpuX(ioDims, ioStrides);
    Tensor<ScaleBiasType> cpuScale(normDims, normStrides);
    Tensor<DxType> cpuDx(ioDims, ioStrides);
    Tensor<ScaleBiasType> cpuDscale(normDims, normStrides);
    Tensor<ScaleBiasType> cpuDbias(normDims, normStrides);
    Tensor<double> cpuEpsilon(std::vector<int64_t>{1}, std::vector<int64_t>{1});
    Tensor<MeanInvVarianceType> cpuMean(batchDims, batchStrides);
    Tensor<MeanInvVarianceType> cpuRstd(batchDims, batchStrides);

    constexpr unsigned int SEED = 42;
    cpuDy.fillWithRandomValues(static_cast<DyType>(-1.0), static_cast<DyType>(1.0), SEED);
    cpuX.fillWithRandomValues(static_cast<DxType>(-1.0), static_cast<DxType>(1.0), SEED + 1);
    cpuScale.fillWithRandomValues(
        static_cast<ScaleBiasType>(-1.0), static_cast<ScaleBiasType>(1.0), SEED + 2);
    cpuDx.fillWithRandomValues(static_cast<DxType>(-1.0), static_cast<DxType>(1.0), SEED + 3);
    cpuDscale.fillWithRandomValues(
        static_cast<ScaleBiasType>(-1.0), static_cast<ScaleBiasType>(1.0), SEED + 4);
    cpuDbias.fillWithRandomValues(
        static_cast<ScaleBiasType>(-1.0), static_cast<ScaleBiasType>(1.0), SEED + 5);
    cpuEpsilon.fillWithValue(1.0);
    cpuMean.fillWithRandomValues(
        static_cast<MeanInvVarianceType>(-1.0), static_cast<MeanInvVarianceType>(1.0), SEED + 6);
    cpuRstd.fillWithRandomValues(
        static_cast<MeanInvVarianceType>(0.0), static_cast<MeanInvVarianceType>(1.0), SEED + 7);

    const Workspace gpuDy(ioCount * sizeof(DyType));
    const Workspace gpuX(ioCount * sizeof(DxType));
    const Workspace gpuScale(normCount * sizeof(ScaleBiasType));
    const Workspace gpuDx(ioCount * sizeof(DxType));
    const Workspace gpuDscale(normCount * sizeof(ScaleBiasType));
    const Workspace gpuDbias(normCount * sizeof(ScaleBiasType));
    const Workspace gpuEpsilon(sizeof(double));
    const Workspace gpuMean(batchCount * sizeof(MeanInvVarianceType));
    const Workspace gpuRstd(batchCount * sizeof(MeanInvVarianceType));

    ASSERT_EQ(
        hipMemcpy(
            gpuDy.get(), cpuDy.rawHostData(), ioCount * sizeof(DyType), hipMemcpyHostToDevice),
        hipSuccess);
    ASSERT_EQ(
        hipMemcpy(gpuX.get(), cpuX.rawHostData(), ioCount * sizeof(DxType), hipMemcpyHostToDevice),
        hipSuccess);
    ASSERT_EQ(hipMemcpy(gpuScale.get(),
                        cpuScale.rawHostData(),
                        normCount * sizeof(ScaleBiasType),
                        hipMemcpyHostToDevice),
              hipSuccess);
    ASSERT_EQ(
        hipMemcpy(
            gpuEpsilon.get(), cpuEpsilon.rawHostData(), sizeof(double), hipMemcpyHostToDevice),
        hipSuccess);
    ASSERT_EQ(hipMemcpy(gpuMean.get(),
                        cpuMean.rawHostData(),
                        batchCount * sizeof(MeanInvVarianceType),
                        hipMemcpyHostToDevice),
              hipSuccess);
    ASSERT_EQ(hipMemcpy(gpuRstd.get(),
                        cpuRstd.rawHostData(),
                        batchCount * sizeof(MeanInvVarianceType),
                        hipMemcpyHostToDevice),
              hipSuccess);

    std::unordered_map<int64_t, void*> gpuVariantPack;
    gpuVariantPack[DY_UID] = gpuDy.get();
    gpuVariantPack[X_UID] = gpuX.get();
    gpuVariantPack[SCALE_UID] = gpuScale.get();
    gpuVariantPack[DX_UID] = gpuDx.get();
    gpuVariantPack[DSCALE_UID] = gpuDscale.get();
    gpuVariantPack[DBIAS_UID] = gpuDbias.get();
    gpuVariantPack[EPSILON_UID] = gpuEpsilon.get();
    if(nodeAttributes->mean_tensor_uid().has_value())
    {
        gpuVariantPack[MEAN_UID] = gpuMean.get();
    }
    if(nodeAttributes->inv_variance_tensor_uid().has_value())
    {
        gpuVariantPack[INV_VARIANCE_UID] = gpuRstd.get();
    }

    gpuPlan.execute(gpuVariantPack);

    std::vector<DxType> gpuDxData(ioCount);
    ASSERT_EQ(
        hipMemcpy(gpuDxData.data(), gpuDx.get(), ioCount * sizeof(DxType), hipMemcpyDeviceToHost),
        hipSuccess);
    std::vector<ScaleBiasType> gpuDscaleData(normCount);
    ASSERT_EQ(hipMemcpy(gpuDscaleData.data(),
                        gpuDscale.get(),
                        normCount * sizeof(ScaleBiasType),
                        hipMemcpyDeviceToHost),
              hipSuccess);
    std::vector<ScaleBiasType> gpuDbiasData(normCount);
    ASSERT_EQ(hipMemcpy(gpuDbiasData.data(),
                        gpuDbias.get(),
                        normCount * sizeof(ScaleBiasType),
                        hipMemcpyDeviceToHost),
              hipSuccess);

    std::unordered_map<int64_t, void*> cpuVariantPack;
    cpuVariantPack[DY_UID] = cpuDy.rawHostData();
    cpuVariantPack[X_UID] = cpuX.rawHostData();
    cpuVariantPack[SCALE_UID] = cpuScale.rawHostData();
    cpuVariantPack[DX_UID] = cpuDx.rawHostData();
    cpuVariantPack[DSCALE_UID] = cpuDscale.rawHostData();
    cpuVariantPack[DBIAS_UID] = cpuDbias.rawHostData();
    cpuVariantPack[EPSILON_UID] = cpuEpsilon.rawHostData();
    if(nodeAttributes->mean_tensor_uid().has_value())
    {
        cpuVariantPack[MEAN_UID] = cpuMean.rawHostData();
    }
    if(nodeAttributes->inv_variance_tensor_uid().has_value())
    {
        cpuVariantPack[INV_VARIANCE_UID] = cpuRstd.rawHostData();
    }

    CpuReferenceGraphExecutor cpuExecutor;
    cpuExecutor.execute(graphBuilder.GetBufferPointer(), graphBuilder.GetSize(), cpuVariantPack);

    const auto* cpuDxData = static_cast<const DxType*>(cpuDx.rawHostData());
    for(size_t i = 0; i < ioCount; ++i)
    {
        EXPECT_NEAR(static_cast<float>(gpuDxData[i]), static_cast<float>(cpuDxData[i]), tolerance)
            << "Mismatch in dx at index " << i;
    }
    const auto* cpuDscaleData = static_cast<const ScaleBiasType*>(cpuDscale.rawHostData());
    for(size_t i = 0; i < normCount; ++i)
    {
        EXPECT_NEAR(
            static_cast<float>(gpuDscaleData[i]), static_cast<float>(cpuDscaleData[i]), tolerance)
            << "Mismatch in dscale at index " << i;
    }
    const auto* cpuDbiasData = static_cast<const ScaleBiasType*>(cpuDbias.rawHostData());
    for(size_t i = 0; i < normCount; ++i)
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
