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

#include "LayernormFwdGraphTestUtils.hpp"
#include "harness/gpu-graph-executor/detail/GpuLayernormFwdPlan.hpp"
#include "harness/gpu-graph-executor/detail/GpuLayernormFwdSignatureKey.hpp"
#include "harness/gpu-graph-executor/detail/GpuPlanBuilderRegistry.hpp"

using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_flatbuffers_sdk::flatbuffer_utilities;
using namespace hipdnn_integration_tests::test_utils;
using namespace hipdnn_integration_tests::gpu_graph_executor::detail;
using namespace hipdnn_test_sdk::utilities;

TEST(TestGpuLayernormFwdPlanBuilder, PlanConstruction)
{
    constexpr int64_t X_UID = 10;
    constexpr int64_t Y_UID = 11;
    constexpr int64_t SCALE_UID = 12;
    constexpr int64_t BIAS_UID = 13;
    constexpr int64_t EPSILON_UID = 14;
    constexpr int64_t MEAN_UID = 15;
    constexpr int64_t INV_VARIANCE_UID = 16;

    const std::vector<int64_t> ioDims = {2, 3, 4, 5};
    const TensorLayout layout = TensorLayout::NCHW;
    const double epsilon = LAYERNORM_DEFAULT_EPSILON;
    const int64_t normalizedDimCount = 2;

    auto graphBuilder = createLayernormFwdGraph(X_UID,
                                                Y_UID,
                                                SCALE_UID,
                                                BIAS_UID,
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

    const GpuLayernormFwdPlanBuilder<DataType::FLOAT,
                                     DataType::FLOAT,
                                     DataType::FLOAT,
                                     DataType::FLOAT,
                                     DataType::FLOAT>
        planBuilder;

    auto builtPlan = planBuilder.buildNodePlan(graphWrapper, graphWrapper.getNode(0));

    const bool result
        = dynamic_cast<GpuLayernormFwdPlan<float, float, float, float, float>*>(builtPlan.get())
          != nullptr;
    EXPECT_TRUE(result);
}

TEST(TestGpuLayernormFwdPlanBuilder, IsApplicable)
{
    constexpr int64_t X_UID = 10;
    constexpr int64_t Y_UID = 11;
    constexpr int64_t SCALE_UID = 12;
    constexpr int64_t BIAS_UID = 13;
    constexpr int64_t EPSILON_UID = 14;
    constexpr int64_t MEAN_UID = 15;
    constexpr int64_t INV_VARIANCE_UID = 16;

    const std::vector<int64_t> ioDims = {2, 3, 4, 5};
    const TensorLayout layout = TensorLayout::NCHW;
    const double epsilon = LAYERNORM_DEFAULT_EPSILON;
    const int64_t normalizedDimCount = 2;

    auto graphBuilder = createLayernormFwdGraph(X_UID,
                                                Y_UID,
                                                SCALE_UID,
                                                BIAS_UID,
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

    const GpuLayernormFwdPlanBuilder<DataType::FLOAT,
                                     DataType::FLOAT,
                                     DataType::FLOAT,
                                     DataType::FLOAT,
                                     DataType::FLOAT>
        floatPlanBuilder;

    EXPECT_TRUE(
        floatPlanBuilder.isApplicable(graphWrapper.getNode(0), graphWrapper.getTensorMap()));

    // Half builder should not be applicable for a float graph
    const GpuLayernormFwdPlanBuilder<DataType::HALF,
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

template <typename XType,
          typename ScaleBiasType,
          typename MeanInvVarianceType,
          typename YType,
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

    constexpr int64_t X_UID = 1;
    constexpr int64_t Y_UID = 2;
    constexpr int64_t SCALE_UID = 3;
    constexpr int64_t BIAS_UID = 4;
    constexpr int64_t EPSILON_UID = 5;
    constexpr int64_t MEAN_UID = 6;
    constexpr int64_t INV_VARIANCE_UID = 7;

    auto xDataType = nativeTypeToDataType<XType>();
    auto yDataType = nativeTypeToDataType<YType>();
    auto scaleBiasDataType = nativeTypeToDataType<ScaleBiasType>();
    auto meanInvVarianceDataType = nativeTypeToDataType<MeanInvVarianceType>();
    auto computeDataType = nativeTypeToDataType<ComputeType>();

    const auto epsilon = LAYERNORM_DEFAULT_EPSILON;
    auto graphBuilder = createLayernormFwdGraph(X_UID,
                                                Y_UID,
                                                SCALE_UID,
                                                BIAS_UID,
                                                EPSILON_UID,
                                                MEAN_UID,
                                                INV_VARIANCE_UID,
                                                ioDims,
                                                layout,
                                                epsilon,
                                                normalizedDimCount,
                                                xDataType,
                                                yDataType,
                                                scaleBiasDataType,
                                                meanInvVarianceDataType,
                                                computeDataType);

    const GraphWrapper graphWrapper(graphBuilder.GetBufferPointer(), graphBuilder.GetSize());

    const auto* nodeAttributes = graphWrapper.getNode(0).attributes_as_LayernormAttributes();
    const auto& tensorMap = graphWrapper.getTensorMap();

    GpuLayernormFwdParams params(
        *tensorMap.at(nodeAttributes->x_tensor_uid()),
        *tensorMap.at(nodeAttributes->y_tensor_uid()),
        *tensorMap.at(nodeAttributes->scale_tensor_uid()),
        *tensorMap.at(nodeAttributes->bias_tensor_uid()),
        *tensorMap.at(nodeAttributes->epsilon_tensor_uid()),
        normalizedDimCount,
        nodeAttributes->mean_tensor_uid().has_value()
            ? tensorMap.at(nodeAttributes->mean_tensor_uid().value())
            : nullptr,
        nodeAttributes->inv_variance_tensor_uid().has_value()
            ? tensorMap.at(nodeAttributes->inv_variance_tensor_uid().value())
            : nullptr);

    GpuLayernormFwdPlan<XType, ScaleBiasType, MeanInvVarianceType, YType, ComputeType> gpuPlan(
        std::move(params));

    auto ioCount = elementCount(ioDims);
    auto normCount = elementCount(normDims);
    auto batchCount = elementCount(batchDims);

    Tensor<XType> cpuX(ioDims, ioStrides);
    Tensor<YType> cpuY(ioDims, ioStrides);
    Tensor<ScaleBiasType> cpuScale(normDims, normStrides);
    Tensor<ScaleBiasType> cpuBias(normDims, normStrides);
    Tensor<ComputeType> cpuEpsilon(std::vector<int64_t>{1}, std::vector<int64_t>{1});
    Tensor<MeanInvVarianceType> cpuMean(batchDims, batchStrides);
    Tensor<MeanInvVarianceType> cpuRstd(batchDims, batchStrides);

    constexpr unsigned int SEED = 42;
    cpuX.fillWithRandomValues(static_cast<XType>(-1.0), static_cast<XType>(1.0), SEED);
    cpuScale.fillWithRandomValues(
        static_cast<ScaleBiasType>(-1.0), static_cast<ScaleBiasType>(1.0), SEED + 1);
    cpuBias.fillWithRandomValues(
        static_cast<ScaleBiasType>(-1.0), static_cast<ScaleBiasType>(1.0), SEED + 2);
    cpuEpsilon.fillWithValue(static_cast<ComputeType>(LAYERNORM_DEFAULT_EPSILON));

    const Workspace gpuX(ioCount * sizeof(XType));
    const Workspace gpuY(ioCount * sizeof(YType));
    const Workspace gpuScale(normCount * sizeof(ScaleBiasType));
    const Workspace gpuBias(normCount * sizeof(ScaleBiasType));
    const Workspace gpuEpsilon(sizeof(ComputeType));
    const Workspace gpuMean(batchCount * sizeof(MeanInvVarianceType));
    const Workspace gpuRstd(batchCount * sizeof(MeanInvVarianceType));

    ASSERT_EQ(
        hipMemcpy(gpuX.get(), cpuX.rawHostData(), ioCount * sizeof(XType), hipMemcpyHostToDevice),
        hipSuccess);
    ASSERT_EQ(hipMemcpy(gpuScale.get(),
                        cpuScale.rawHostData(),
                        normCount * sizeof(ScaleBiasType),
                        hipMemcpyHostToDevice),
              hipSuccess);
    ASSERT_EQ(hipMemcpy(gpuBias.get(),
                        cpuBias.rawHostData(),
                        normCount * sizeof(ScaleBiasType),
                        hipMemcpyHostToDevice),
              hipSuccess);
    ASSERT_EQ(
        hipMemcpy(
            gpuEpsilon.get(), cpuEpsilon.rawHostData(), sizeof(ComputeType), hipMemcpyHostToDevice),
        hipSuccess);

    std::unordered_map<int64_t, void*> gpuVariantPack;
    gpuVariantPack[X_UID] = gpuX.get();
    gpuVariantPack[Y_UID] = gpuY.get();
    gpuVariantPack[SCALE_UID] = gpuScale.get();
    gpuVariantPack[BIAS_UID] = gpuBias.get();
    gpuVariantPack[EPSILON_UID] = gpuEpsilon.get();
    gpuVariantPack[MEAN_UID] = gpuMean.get();
    gpuVariantPack[INV_VARIANCE_UID] = gpuRstd.get();

    gpuPlan.execute(gpuVariantPack);

    std::vector<YType> gpuYData(ioCount);
    ASSERT_EQ(
        hipMemcpy(gpuYData.data(), gpuY.get(), ioCount * sizeof(YType), hipMemcpyDeviceToHost),
        hipSuccess);

    std::unordered_map<int64_t, void*> cpuVariantPack;
    cpuVariantPack[X_UID] = cpuX.rawHostData();
    cpuVariantPack[Y_UID] = cpuY.rawHostData();
    cpuVariantPack[SCALE_UID] = cpuScale.rawHostData();
    cpuVariantPack[BIAS_UID] = cpuBias.rawHostData();
    cpuVariantPack[EPSILON_UID] = cpuEpsilon.rawHostData();
    cpuVariantPack[MEAN_UID] = cpuMean.rawHostData();
    cpuVariantPack[INV_VARIANCE_UID] = cpuRstd.rawHostData();

    CpuReferenceGraphExecutor cpuExecutor;
    cpuExecutor.execute(graphBuilder.GetBufferPointer(), graphBuilder.GetSize(), cpuVariantPack);

    const auto* cpuYData = static_cast<const YType*>(cpuY.rawHostData());
    for(size_t i = 0; i < ioCount; ++i)
    {
        EXPECT_NEAR(static_cast<float>(gpuYData[i]), static_cast<float>(cpuYData[i]), tolerance)
            << "Mismatch in y at index " << i;
    }
}

} // namespace

// =========================
// FP32 plan execution tests
// =========================

TEST(TestGpuLayernormFwdPlanFp32, ExecutePlanNchw)
{
    SKIP_IF_NO_DEVICES();

    runPlanExecuteVsCpuRef<float, float, float, float, float>(
        {5, 4, 3, 2}, TensorLayout::NCHW, 3, layernorm::getTolerance<float>());
}

TEST(TestGpuLayernormFwdPlanFp32, ExecutePlanNhwc)
{
    SKIP_IF_NO_DEVICES();

    runPlanExecuteVsCpuRef<float, float, float, float, float>(
        {5, 4, 3, 2}, TensorLayout::NHWC, 3, layernorm::getTolerance<float>());
}

// =========================
// FP16 plan execution tests
// =========================

TEST(TestGpuLayernormFwdPlanFp16, ExecutePlanNchw)
{
    SKIP_IF_NO_DEVICES();

    runPlanExecuteVsCpuRef<half, half, half, half, float>(
        {5, 4, 3, 2}, TensorLayout::NCHW, 3, layernorm::getTolerance<half>());
}

TEST(TestGpuLayernormFwdPlanFp16, ExecutePlanNhwc)
{
    SKIP_IF_NO_DEVICES();

    runPlanExecuteVsCpuRef<half, half, half, half, float>(
        {5, 4, 3, 2}, TensorLayout::NHWC, 3, layernorm::getTolerance<half>());
}

// =========================
// BFP16 plan execution tests
// =========================

TEST(TestGpuLayernormFwdPlanBfp16, ExecutePlanNchw)
{
    SKIP_IF_NO_DEVICES();

    runPlanExecuteVsCpuRef<bfloat16, bfloat16, bfloat16, bfloat16, float>(
        {5, 4, 3, 2}, TensorLayout::NCHW, 3, layernorm::getTolerance<bfloat16>());
}

TEST(TestGpuLayernormFwdPlanBfp16, ExecutePlanNhwc)
{
    SKIP_IF_NO_DEVICES();

    runPlanExecuteVsCpuRef<bfloat16, bfloat16, bfloat16, bfloat16, float>(
        {5, 4, 3, 2}, TensorLayout::NHWC, 3, layernorm::getTolerance<bfloat16>());
}

// ============================================================================
// Rejection test — unregistered signature
// ============================================================================

TEST(TestGpuLayernormFwdPlanBuilder, UnregisteredSignatureThrows)
{
    GpuPlanBuilderRegistry registry;

    const GpuLayernormFwdSignatureKey unregisteredKey{
        DataType::INT8, DataType::INT8, DataType::INT8, DataType::INT8, DataType::FLOAT};

    EXPECT_THROW(registry.getPlanBuilder(unregisteredKey), std::runtime_error);
}
