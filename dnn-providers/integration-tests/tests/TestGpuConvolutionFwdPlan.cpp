// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <cstdint>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include <hip/hip_runtime.h>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_data_sdk/utilities/Workspace.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/DynamicTolerances.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/CpuReferenceGraphExecutor.hpp>

#include "ConvolutionFwdGraphTestUtils.hpp"
#include "harness/gpu_graph_executor/detail/GpuConvolutionFwdPlan.hpp"
#include "harness/gpu_graph_executor/detail/GpuConvolutionFwdSignatureKey.hpp"
#include "harness/gpu_graph_executor/detail/GpuPlanBuilderRegistry.hpp"

using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_data_sdk::types;
using namespace hipdnn_integration_tests::test_utils;
using namespace hipdnn_integration_tests::gpu_graph_executor::detail;

TEST(TestGpuConvolutionFwdPlanBuilder, PlanConstruction)
{
    constexpr int64_t X_UID = 10;
    constexpr int64_t W_UID = 11;
    constexpr int64_t Y_UID = 12;

    const std::vector<int64_t> xDims = {1, 1, 2, 2};
    const std::vector<int64_t> wDims = {1, 1, 1, 1};
    const std::vector<int64_t> yDims = {1, 1, 2, 2};

    auto graphBuilder = createConvFwdGraph(X_UID,
                                           W_UID,
                                           Y_UID,
                                           xDims,
                                           wDims,
                                           yDims,
                                           generateStrides(xDims),
                                           generateStrides(wDims),
                                           generateStrides(yDims),
                                           {0, 0},
                                           {1, 1},
                                           {1, 1},
                                           DataType::FLOAT);

    auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        graphBuilder.GetBufferPointer(), graphBuilder.GetSize());

    const GpuConvolutionFwdPlanBuilder<DataType::FLOAT,
                                       DataType::FLOAT,
                                       DataType::FLOAT,
                                       DataType::FLOAT>
        patient;

    auto builtPlan = patient.buildNodePlan(graphWrap, graphWrap.getNode(0));

    const bool result
        = dynamic_cast<GpuConvolutionFwdPlan<float, float, float, float>*>(builtPlan.get())
          != nullptr;
    EXPECT_TRUE(result);
}

TEST(TestGpuConvolutionFwdPlanBuilder, IsApplicable)
{
    constexpr int64_t X_UID = 10;
    constexpr int64_t W_UID = 11;
    constexpr int64_t Y_UID = 12;

    const std::vector<int64_t> xDims = {1, 1, 2, 2};
    const std::vector<int64_t> wDims = {1, 1, 1, 1};
    const std::vector<int64_t> yDims = {1, 1, 2, 2};

    auto graphBuilder = createConvFwdGraph(X_UID,
                                           W_UID,
                                           Y_UID,
                                           xDims,
                                           wDims,
                                           yDims,
                                           generateStrides(xDims),
                                           generateStrides(wDims),
                                           generateStrides(yDims),
                                           {0, 0},
                                           {1, 1},
                                           {1, 1},
                                           DataType::FLOAT);

    auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        graphBuilder.GetBufferPointer(), graphBuilder.GetSize());

    const GpuConvolutionFwdPlanBuilder<DataType::FLOAT,
                                       DataType::FLOAT,
                                       DataType::FLOAT,
                                       DataType::FLOAT>
        floatPlanBuilder;

    EXPECT_TRUE(floatPlanBuilder.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap()));

    // Half builder should not be applicable for a float graph
    const GpuConvolutionFwdPlanBuilder<DataType::HALF,
                                       DataType::HALF,
                                       DataType::HALF,
                                       DataType::FLOAT>
        halfPlanBuilder;

    EXPECT_FALSE(halfPlanBuilder.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap()));

    // Missing tensor should return false
    auto tensorMapCopy = graphWrap.getTensorMap();
    tensorMapCopy.erase(W_UID);
    EXPECT_FALSE(floatPlanBuilder.isApplicable(graphWrap.getNode(0), tensorMapCopy));
}

// ============================================================================
// Templated helper for plan execution vs CPU reference
// ============================================================================

namespace
{

inline size_t elementCount(const std::vector<int64_t>& dims)
{
    size_t count = 1;
    for(auto d : dims)
    {
        count *= static_cast<size_t>(d);
    }
    return count;
}

template <typename XType, typename WType, typename YType, typename ComputeType>
void runPlanExecuteVsCpuRef(const std::vector<int64_t>& xDims,
                            const std::vector<int64_t>& wDims,
                            const std::vector<int64_t>& yDims,
                            const std::vector<int64_t>& padding,
                            const std::vector<int64_t>& stride,
                            const std::vector<int64_t>& dilation,
                            DataType xEnum,
                            DataType yEnum,
                            float tolerance)
{
    constexpr int64_t X_UID = 1;
    constexpr int64_t W_UID = 2;
    constexpr int64_t Y_UID = 3;

    auto xStrides = generateStrides(xDims);
    auto wStrides = generateStrides(wDims);
    auto yStrides = generateStrides(yDims);

    auto graphBuilder = createConvFwdGraph(X_UID,
                                           W_UID,
                                           Y_UID,
                                           xDims,
                                           wDims,
                                           yDims,
                                           xStrides,
                                           wStrides,
                                           yStrides,
                                           padding,
                                           stride,
                                           dilation,
                                           xEnum,
                                           yEnum);

    auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        graphBuilder.GetBufferPointer(), graphBuilder.GetSize());

    const auto* nodeAttributes = graphWrap.getNode(0).attributes_as_ConvolutionFwdAttributes();
    const auto& tensorMap = graphWrap.getTensorMap();

    GpuConvolutionFwdParams params(*tensorMap.at(nodeAttributes->x_tensor_uid()),
                                   *tensorMap.at(nodeAttributes->w_tensor_uid()),
                                   *tensorMap.at(nodeAttributes->y_tensor_uid()),
                                   padding,
                                   padding,
                                   stride,
                                   dilation,
                                   ConvMode::CROSS_CORRELATION);

    GpuConvolutionFwdPlan<XType, WType, YType, ComputeType> patient(std::move(params));

    auto xCount = elementCount(xDims);
    auto wCount = elementCount(wDims);
    auto yCount = elementCount(yDims);

    // Prepare CPU tensors and fill with random data
    hipdnn_data_sdk::utilities::Tensor<XType> cpuX(xDims, xStrides);
    hipdnn_data_sdk::utilities::Tensor<WType> cpuW(wDims, wStrides);
    hipdnn_data_sdk::utilities::Tensor<YType> cpuY(yDims, yStrides);

    constexpr unsigned int SEED = 42;
    cpuX.fillWithRandomValues(static_cast<XType>(-1), static_cast<XType>(1), SEED);
    cpuW.fillWithRandomValues(static_cast<WType>(-1), static_cast<WType>(1), SEED + 1);

    // Allocate device buffers (RAII — freed automatically)
    const hipdnn_data_sdk::utilities::Workspace dX(xCount * sizeof(XType));
    const hipdnn_data_sdk::utilities::Workspace dW(wCount * sizeof(WType));
    const hipdnn_data_sdk::utilities::Workspace dY(yCount * sizeof(YType));

    ASSERT_EQ(
        hipMemcpy(dX.get(), cpuX.rawHostData(), xCount * sizeof(XType), hipMemcpyHostToDevice),
        hipSuccess);
    ASSERT_EQ(
        hipMemcpy(dW.get(), cpuW.rawHostData(), wCount * sizeof(WType), hipMemcpyHostToDevice),
        hipSuccess);
    ASSERT_EQ(hipMemset(dY.get(), 0, yCount * sizeof(YType)), hipSuccess);

    // Execute
    std::unordered_map<int64_t, void*> variantPack;
    variantPack[X_UID] = dX.get();
    variantPack[W_UID] = dW.get();
    variantPack[Y_UID] = dY.get();

    patient.execute(variantPack);

    // Copy result back
    std::vector<YType> gpuYData(yCount);
    ASSERT_EQ(hipMemcpy(gpuYData.data(), dY.get(), yCount * sizeof(YType), hipMemcpyDeviceToHost),
              hipSuccess);

    // Run CPU reference executor with host pointers (same graph)
    std::unordered_map<int64_t, void*> cpuVariantPack;
    cpuVariantPack[X_UID] = cpuX.rawHostData();
    cpuVariantPack[W_UID] = cpuW.rawHostData();
    cpuVariantPack[Y_UID] = cpuY.rawHostData();

    hipdnn_test_sdk::utilities::CpuReferenceGraphExecutor cpuExecutor;
    cpuExecutor.execute(graphBuilder.GetBufferPointer(), graphBuilder.GetSize(), cpuVariantPack);

    // Compare
    const auto* cpuResult = static_cast<const YType*>(cpuY.rawHostData());
    for(size_t i = 0; i < yCount; ++i)
    {
        EXPECT_NEAR(static_cast<float>(gpuYData[i]), static_cast<float>(cpuResult[i]), tolerance)
            << "Mismatch at index " << i;
    }
}

} // anonymous namespace

// ============================================================================
// FP32 plan execution tests
// ============================================================================

TEST(TestGpuConvolutionFwdPlan, ExecutePlan)
{
    SKIP_IF_NO_DEVICES();

    runPlanExecuteVsCpuRef<float, float, float, float>({1, 1, 4, 4},
                                                       {1, 1, 3, 3},
                                                       {1, 1, 2, 2},
                                                       {0, 0},
                                                       {1, 1},
                                                       {1, 1},
                                                       DataType::FLOAT,
                                                       DataType::FLOAT,
                                                       1e-5f);
}

TEST(TestGpuConvolutionFwdPlan, ExecutePlanMultiChannel)
{
    SKIP_IF_NO_DEVICES();

    runPlanExecuteVsCpuRef<float, float, float, float>(
        {1, 3, 8, 8}, // x: batch=1, channels=3, 8x8
        {6, 3, 3, 3}, // w: 6 output filters, 3 input channels, 3x3
        {1, 6, 8, 8}, // y: with pad=1 output is same spatial size
        {1, 1}, // padding
        {1, 1}, // stride
        {1, 1}, // dilation
        DataType::FLOAT,
        DataType::FLOAT,
        1e-5f);
}

TEST(TestGpuConvolutionFwdPlan, ExecutePlanStride2)
{
    SKIP_IF_NO_DEVICES();

    runPlanExecuteVsCpuRef<float, float, float, float>(
        {1, 1, 8, 8}, // x
        {2, 1, 3, 3}, // w: 2 output filters
        {1, 2, 3, 3}, // y: stride-2 downsample => (8-3)/2+1 = 3
        {0, 0}, // padding
        {2, 2}, // stride
        {1, 1}, // dilation
        DataType::FLOAT,
        DataType::FLOAT,
        1e-5f);
}

// ============================================================================
// FP16 plan execution tests (Signatures #2 and #4)
// ============================================================================

TEST(TestGpuConvolutionFwdPlanFp16, ExecutePlan)
{
    SKIP_IF_NO_DEVICES();

    const std::vector<int64_t> wDims = {1, 1, 3, 3};
    auto tolerance
        = hipdnn_test_sdk::utilities::conv::calculateConvFpropTolerance<half, half, double>(
            -1.0, 1.0, -1.0, 1.0, wDims);

    // Signature #2: half/half/half/float
    runPlanExecuteVsCpuRef<half, half, half, float>({1, 1, 4, 4},
                                                    wDims,
                                                    {1, 1, 2, 2},
                                                    {0, 0},
                                                    {1, 1},
                                                    {1, 1},
                                                    DataType::HALF,
                                                    DataType::HALF,
                                                    tolerance);
}

TEST(TestGpuConvolutionFwdPlanFp16, ExecutePlanMixedOutput)
{
    SKIP_IF_NO_DEVICES();

    const std::vector<int64_t> wDims = {1, 1, 3, 3};
    auto tolerance
        = hipdnn_test_sdk::utilities::conv::calculateConvFpropTolerance<float, half, double>(
            -1.0, 1.0, -1.0, 1.0, wDims);

    // Signature #4: half/half/float/float (mixed output)
    runPlanExecuteVsCpuRef<half, half, float, float>({1, 1, 4, 4},
                                                     wDims,
                                                     {1, 1, 2, 2},
                                                     {0, 0},
                                                     {1, 1},
                                                     {1, 1},
                                                     DataType::HALF,
                                                     DataType::FLOAT,
                                                     tolerance);
}

// ============================================================================
// BF16 plan execution tests (Signatures #3 and #5)
// ============================================================================

TEST(TestGpuConvolutionFwdPlanBfp16, ExecutePlan)
{
    SKIP_IF_NO_DEVICES();

    const std::vector<int64_t> wDims = {1, 1, 3, 3};
    auto tolerance
        = hipdnn_test_sdk::utilities::conv::calculateConvFpropTolerance<bfloat16, bfloat16, double>(
            -1.0, 1.0, -1.0, 1.0, wDims);

    // Signature #3: bfloat16/bfloat16/bfloat16/float
    runPlanExecuteVsCpuRef<bfloat16, bfloat16, bfloat16, float>({1, 1, 4, 4},
                                                                wDims,
                                                                {1, 1, 2, 2},
                                                                {0, 0},
                                                                {1, 1},
                                                                {1, 1},
                                                                DataType::BFLOAT16,
                                                                DataType::BFLOAT16,
                                                                tolerance);
}

TEST(TestGpuConvolutionFwdPlanBfp16, ExecutePlanMixedOutput)
{
    SKIP_IF_NO_DEVICES();

    const std::vector<int64_t> wDims = {1, 1, 3, 3};
    auto tolerance
        = hipdnn_test_sdk::utilities::conv::calculateConvFpropTolerance<float, bfloat16, double>(
            -1.0, 1.0, -1.0, 1.0, wDims);

    // Signature #5: bfloat16/bfloat16/float/float (mixed output)
    runPlanExecuteVsCpuRef<bfloat16, bfloat16, float, float>({1, 1, 4, 4},
                                                             wDims,
                                                             {1, 1, 2, 2},
                                                             {0, 0},
                                                             {1, 1},
                                                             {1, 1},
                                                             DataType::BFLOAT16,
                                                             DataType::FLOAT,
                                                             tolerance);
}

// ============================================================================
// Rejection test — unregistered signature
// ============================================================================

TEST(TestGpuConvolutionFwdPlanBuilder, UnregisteredSignatureThrows)
{
    GpuPlanBuilderRegistry registry;

    // INT8 is not in the registered signatures for convolution fwd
    const GpuConvolutionFwdSignatureKey unregisteredKey{
        DataType::INT8, DataType::INT8, DataType::INT8, DataType::FLOAT};

    EXPECT_THROW(registry.getPlanBuilder(unregisteredKey), std::runtime_error);
}
