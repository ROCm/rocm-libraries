// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <cstdint>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include <hip/hip_runtime.h>
#include <hipdnn-gpu-ref/GpuFpReferenceCommon.hpp>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_data_sdk/utilities/Workspace.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/TestTolerances.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/CpuReferenceGraphExecutor.hpp>

#include "ConvolutionBwdGraphTestUtils.hpp"
#include "RaggedGraphTestUtils.hpp"
#include "harness/gpu-graph-executor/detail/GpuConvolutionBwdPlan.hpp"
#include "harness/gpu-graph-executor/detail/GpuConvolutionBwdSignatureKey.hpp"
#include "harness/gpu-graph-executor/detail/GpuPlanBuilderRegistry.hpp"

using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_data_sdk::types;
using namespace hipdnn_integration_tests::test_utils;
using namespace hipdnn_integration_tests::gpu_graph_executor::detail;
using namespace hipdnn_gpu_ref::common::gpu_fp_reference_tensor;

namespace
{

inline flatbuffers::FlatBufferBuilder
    createCustomConvBwdGraph(hipdnn_flatbuffers_sdk::data_objects::DataType inputDataType
                             = hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                             hipdnn_flatbuffers_sdk::data_objects::DataType outputDataType
                             = hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                             int64_t dyUid = 10,
                             int64_t wUid = 11,
                             int64_t dxUid = 12)
{
    const std::vector<int64_t> dxDims = {4, 4, 4, 4};
    const std::vector<int64_t> dxStrides = {64, 16, 4, 1};
    const std::vector<int64_t> wDims = {4, 4, 1, 1};
    const std::vector<int64_t> wStrides = {4, 1, 1, 1};
    const std::vector<int64_t> dyDims = {4, 4, 4, 4};
    const std::vector<int64_t> dyStrides = {64, 16, 4, 1};

    return createConvBwdGraph(dyUid,
                              wUid,
                              dxUid,
                              dyDims,
                              wDims,
                              dxDims,
                              dyStrides,
                              wStrides,
                              dxStrides,
                              {0, 0},
                              {1, 1},
                              {1, 1},
                              inputDataType,
                              outputDataType);
}

template <typename DyType, typename WType, typename DxType, typename ComputeType>
void runBwdPlanExecuteVsCpuRef(const std::vector<int64_t>& dyDims,
                               const std::vector<int64_t>& wDims,
                               const std::vector<int64_t>& dxDims,
                               const std::vector<int64_t>& padding,
                               const std::vector<int64_t>& stride,
                               const std::vector<int64_t>& dilation)
{
    constexpr int64_t DY_UID = 1;
    constexpr int64_t W_UID = 2;
    constexpr int64_t DX_UID = 3;

    auto dyStrides = generateStrides(dyDims);
    auto wStrides = generateStrides(wDims);
    auto dxStrides = generateStrides(dxDims);

    auto graphBuilder
        = createConvBwdGraph(DY_UID,
                             W_UID,
                             DX_UID,
                             dyDims,
                             wDims,
                             dxDims,
                             dyStrides,
                             wStrides,
                             dxStrides,
                             padding,
                             stride,
                             dilation,
                             hipdnn_test_sdk::utilities::nativeTypeToDataType<DyType>(),
                             hipdnn_test_sdk::utilities::nativeTypeToDataType<DxType>());

    auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        graphBuilder.GetBufferPointer(), graphBuilder.GetSize());

    const auto* nodeAttributes = graphWrap.getNode(0).attributes_as_ConvolutionBwdAttributes();
    const auto& tensorMap = graphWrap.getTensorMap();

    GpuConvolutionBwdPlanParams params(*tensorMap.at(nodeAttributes->dy_tensor_uid()),
                                       *tensorMap.at(nodeAttributes->w_tensor_uid()),
                                       *tensorMap.at(nodeAttributes->dx_tensor_uid()),
                                       padding,
                                       padding,
                                       stride,
                                       dilation,
                                       ConvMode::CROSS_CORRELATION);

    GpuConvolutionBwdPlan<DyType, WType, DxType, ComputeType> patient(std::move(params));

    auto dyCount = std::accumulate(
        dyDims.begin(), dyDims.end(), static_cast<size_t>(1), std::multiplies<size_t>());
    auto wCount = std::accumulate(
        wDims.begin(), wDims.end(), static_cast<size_t>(1), std::multiplies<size_t>());
    auto dxCount = std::accumulate(
        dxDims.begin(), dxDims.end(), static_cast<size_t>(1), std::multiplies<size_t>());

    // Prepare CPU tensors and fill with random data
    hipdnn_data_sdk::utilities::Tensor<DyType> cpuDy(dyDims, dyStrides);
    hipdnn_data_sdk::utilities::Tensor<WType> cpuW(wDims, wStrides);
    hipdnn_data_sdk::utilities::Tensor<DxType> cpuDx(dxDims, dxStrides);

    constexpr unsigned int SEED = 42;
    fillWithRandomValues(cpuDy, static_cast<DyType>(-1), static_cast<DyType>(1), SEED);
    fillWithRandomValues(cpuW, static_cast<WType>(-1), static_cast<WType>(1), SEED + 1);

    // Allocate device buffers (RAII — freed automatically)
    const hipdnn_data_sdk::utilities::Workspace dDy(dyCount * sizeof(DyType));
    const hipdnn_data_sdk::utilities::Workspace dW(wCount * sizeof(WType));
    const hipdnn_data_sdk::utilities::Workspace dDx(dxCount * sizeof(DxType));

    ASSERT_EQ(
        hipMemcpy(dDy.get(), cpuDy.rawHostData(), dyCount * sizeof(DyType), hipMemcpyHostToDevice),
        hipSuccess);
    ASSERT_EQ(
        hipMemcpy(dW.get(), cpuW.rawHostData(), wCount * sizeof(WType), hipMemcpyHostToDevice),
        hipSuccess);
    ASSERT_EQ(hipMemset(dDx.get(), 0, dxCount * sizeof(DxType)), hipSuccess);

    // Execute GPU plan
    std::unordered_map<int64_t, void*> variantPack;
    variantPack[DY_UID] = dDy.get();
    variantPack[W_UID] = dW.get();
    variantPack[DX_UID] = dDx.get();

    patient.execute(variantPack);

    // Copy result back
    std::vector<DxType> gpuDxData(dxCount);
    ASSERT_EQ(
        hipMemcpy(gpuDxData.data(), dDx.get(), dxCount * sizeof(DxType), hipMemcpyDeviceToHost),
        hipSuccess);

    // Run CPU reference executor with host pointers (same graph)
    std::unordered_map<int64_t, void*> cpuVariantPack;
    cpuVariantPack[DY_UID] = cpuDy.rawHostData();
    cpuVariantPack[W_UID] = cpuW.rawHostData();
    cpuVariantPack[DX_UID] = cpuDx.rawHostData();

    hipdnn_test_sdk::utilities::CpuReferenceGraphExecutor cpuExecutor;
    cpuExecutor.execute(graphBuilder.GetBufferPointer(), graphBuilder.GetSize(), cpuVariantPack);

    // Compare GPU output against CPU reference
    const auto* cpuResult = static_cast<const DxType*>(cpuDx.rawHostData());
    for(size_t i = 0; i < dxCount; ++i)
    {
        EXPECT_NEAR(static_cast<float>(gpuDxData[i]),
                    static_cast<float>(cpuResult[i]),
                    hipdnn_test_sdk::utilities::conv::getToleranceBwd<DxType>())
            << "Mismatch at index " << i;
    }
}

} // namespace

TEST(TestGpuConvolutionBwdPlanBuilder, PlanConstruction)
{
    auto graphBuilder = createCustomConvBwdGraph();
    auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        graphBuilder.GetBufferPointer(), graphBuilder.GetSize());

    const GpuConvolutionBwdPlanBuilder<DataType::FLOAT,
                                       DataType::FLOAT,
                                       DataType::FLOAT,
                                       DataType::FLOAT>
        patient;

    auto builtPlan = patient.buildNodePlan(graphWrap, graphWrap.getNode(0));

    const bool result
        = dynamic_cast<GpuConvolutionBwdPlan<float, float, float, float>*>(builtPlan.get())
          != nullptr;
    EXPECT_TRUE(result);
}

TEST(TestGpuConvolutionBwdPlanBuilder, IsApplicable)
{
    auto graphBuilder = createCustomConvBwdGraph();
    auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        graphBuilder.GetBufferPointer(), graphBuilder.GetSize());

    const GpuConvolutionBwdPlanBuilder<DataType::FLOAT,
                                       DataType::FLOAT,
                                       DataType::FLOAT,
                                       DataType::FLOAT>
        floatPlanBuilder;

    EXPECT_TRUE(floatPlanBuilder.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap()));

    // Half builder should not be applicable for a float graph
    const GpuConvolutionBwdPlanBuilder<DataType::HALF,
                                       DataType::HALF,
                                       DataType::HALF,
                                       DataType::FLOAT>
        halfPlanBuilder;

    EXPECT_FALSE(halfPlanBuilder.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap()));

    // Missing tensor should return false
    auto tensorMapCopy = graphWrap.getTensorMap();
    tensorMapCopy.erase(11); // W_UID
    EXPECT_FALSE(floatPlanBuilder.isApplicable(graphWrap.getNode(0), tensorMapCopy));
}

TEST(TestGpuConvolutionBwdPlanBuilder, IsNotApplicableForRaggedTensors)
{
    auto graphBuilder = createCustomConvBwdGraph();

    const GpuConvolutionBwdPlanBuilder<DataType::FLOAT,
                                       DataType::FLOAT,
                                       DataType::FLOAT,
                                       DataType::FLOAT>
        floatPlanBuilder;

    auto denseWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        graphBuilder.GetBufferPointer(), graphBuilder.GetSize());
    ASSERT_TRUE(floatPlanBuilder.isApplicable(denseWrap.getNode(0), denseWrap.getTensorMap()));

    auto ragged = markFirstTensorRagged(graphBuilder.GetBufferPointer());
    auto raggedWrap
        = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(ragged.data(), ragged.size());
    EXPECT_FALSE(floatPlanBuilder.isApplicable(raggedWrap.getNode(0), raggedWrap.getTensorMap()));
}

TEST(TestGpuConvolutionBwdPlanBuilder, IsApplicableFalseWhenDyTypeMismatched)
{
    auto graphBuilder = createCustomConvBwdGraph(DataType::HALF, DataType::FLOAT);
    auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        graphBuilder.GetBufferPointer(), graphBuilder.GetSize());

    const GpuConvolutionBwdPlanBuilder<DataType::FLOAT,
                                       DataType::FLOAT,
                                       DataType::FLOAT,
                                       DataType::FLOAT>
        patient;

    EXPECT_FALSE(patient.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap()));
}

TEST(TestGpuConvolutionBwdPlanBuilder, IsApplicableFalseWhenWTypeMismatched)
{
    constexpr int64_t DY_UID = 10;
    constexpr int64_t W_UID = 11;
    constexpr int64_t DX_UID = 12;

    const std::vector<int64_t> dxDims = {4, 4, 4, 4};
    const std::vector<int64_t> dxStrides = {64, 16, 4, 1};
    const std::vector<int64_t> wDims = {4, 4, 1, 1};
    const std::vector<int64_t> wStrides = {4, 1, 1, 1};
    const std::vector<int64_t> dyDims = {4, 4, 4, 4};
    const std::vector<int64_t> dyStrides = {64, 16, 4, 1};

    flatbuffers::FlatBufferBuilder builder;
    std::vector<flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>>
        tensors;

    tensors.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        DY_UID,
        "dy",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &dyStrides,
        &dyDims));
    tensors.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        W_UID,
        "w",
        hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
        &wStrides,
        &wDims));
    tensors.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        DX_UID,
        "dx",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &dxStrides,
        &dxDims));

    const std::vector<int64_t> padding = {0, 0};
    const std::vector<int64_t> stride = {1, 1};
    const std::vector<int64_t> dilation = {1, 1};
    auto convBwdAttrs = hipdnn_flatbuffers_sdk::data_objects::CreateConvolutionBwdAttributesDirect(
        builder,
        DY_UID,
        W_UID,
        DX_UID,
        &padding,
        &padding,
        &stride,
        &dilation,
        hipdnn_flatbuffers_sdk::data_objects::ConvMode::CROSS_CORRELATION);

    std::vector<flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Node>> nodes;
    nodes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "conv_bwd_node",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::ConvolutionBwdAttributes,
        convBwdAttrs.Union()));

    auto graph = hipdnn_flatbuffers_sdk::data_objects::CreateGraphDirect(
        builder,
        "ConvBwdTestGraph",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &tensors,
        &nodes);
    builder.Finish(graph);

    auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    const GpuConvolutionBwdPlanBuilder<DataType::FLOAT,
                                       DataType::FLOAT,
                                       DataType::FLOAT,
                                       DataType::FLOAT>
        patient;

    EXPECT_FALSE(patient.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap()));
}

TEST(TestGpuConvolutionBwdPlanBuilder, IsApplicableFalseWhenDxTypeMismatched)
{
    auto graphBuilder = createCustomConvBwdGraph(DataType::FLOAT, DataType::HALF);
    auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        graphBuilder.GetBufferPointer(), graphBuilder.GetSize());

    const GpuConvolutionBwdPlanBuilder<DataType::FLOAT,
                                       DataType::FLOAT,
                                       DataType::FLOAT,
                                       DataType::FLOAT>
        patient;

    EXPECT_FALSE(patient.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap()));
}

TEST(TestGpuConvolutionBwdPlanBuilder, IsApplicableFalseWhenComputeTypeMismatched)
{
    auto graphBuilder = createCustomConvBwdGraph();
    auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        graphBuilder.GetBufferPointer(), graphBuilder.GetSize());

    const GpuConvolutionBwdPlanBuilder<DataType::FLOAT,
                                       DataType::FLOAT,
                                       DataType::FLOAT,
                                       DataType::HALF>
        patient;

    EXPECT_FALSE(patient.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap()));
}

TEST(TestGpuConvolutionBwdPlanBuilder, BuildNodePlanThrowsForWrongAttributesType)
{
    auto builder = hipdnn_test_sdk::utilities::createValidConvFwdGraph();
    auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    const GpuConvolutionBwdPlanBuilder<DataType::FLOAT,
                                       DataType::FLOAT,
                                       DataType::FLOAT,
                                       DataType::FLOAT>
        patient;

    EXPECT_THROW(patient.buildNodePlan(graphWrap, graphWrap.getNode(0)), std::runtime_error);
}

TEST(TestGpuConvolutionBwdPlanBuilder, IsApplicableReturnsFalseForWrongAttributesType)
{
    auto builder = hipdnn_test_sdk::utilities::createValidConvFwdGraph();
    auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    const GpuConvolutionBwdPlanBuilder<DataType::FLOAT,
                                       DataType::FLOAT,
                                       DataType::FLOAT,
                                       DataType::FLOAT>
        patient;

    EXPECT_FALSE(patient.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap()));
}

TEST(TestGpuConvolutionBwdPlanBuilder, IsApplicableFalseWhenDyTensorMissing)
{
    constexpr int64_t DY_UID = 10;
    auto graphBuilder = createCustomConvBwdGraph(DataType::FLOAT, DataType::FLOAT, DY_UID);
    auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        graphBuilder.GetBufferPointer(), graphBuilder.GetSize());

    const GpuConvolutionBwdPlanBuilder<DataType::FLOAT,
                                       DataType::FLOAT,
                                       DataType::FLOAT,
                                       DataType::FLOAT>
        patient;

    auto tensorMapCopy = graphWrap.getTensorMap();
    tensorMapCopy.erase(DY_UID);

    EXPECT_FALSE(patient.isApplicable(graphWrap.getNode(0), tensorMapCopy));
}

TEST(TestGpuConvolutionBwdPlanBuilder, IsApplicableFalseWhenWTensorMissing)
{
    constexpr int64_t W_UID = 11;
    auto graphBuilder = createCustomConvBwdGraph(DataType::FLOAT, DataType::FLOAT, 10, W_UID);
    auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        graphBuilder.GetBufferPointer(), graphBuilder.GetSize());

    const GpuConvolutionBwdPlanBuilder<DataType::FLOAT,
                                       DataType::FLOAT,
                                       DataType::FLOAT,
                                       DataType::FLOAT>
        patient;

    auto tensorMapCopy = graphWrap.getTensorMap();
    tensorMapCopy.erase(W_UID);

    EXPECT_FALSE(patient.isApplicable(graphWrap.getNode(0), tensorMapCopy));
}

TEST(TestGpuConvolutionBwdPlanBuilder, IsApplicableFalseWhenDxTensorMissing)
{
    constexpr int64_t DX_UID = 12;
    auto graphBuilder = createCustomConvBwdGraph(DataType::FLOAT, DataType::FLOAT, 10, 11, DX_UID);
    auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        graphBuilder.GetBufferPointer(), graphBuilder.GetSize());

    const GpuConvolutionBwdPlanBuilder<DataType::FLOAT,
                                       DataType::FLOAT,
                                       DataType::FLOAT,
                                       DataType::FLOAT>
        patient;

    auto tensorMapCopy = graphWrap.getTensorMap();
    tensorMapCopy.erase(DX_UID);

    EXPECT_FALSE(patient.isApplicable(graphWrap.getNode(0), tensorMapCopy));
}

TEST(TestGpuConvolutionBwdPlanBuilder, UnregisteredSignatureThrows)
{
    GpuPlanBuilderRegistry registry;

    // INT8 is not in the registered signatures for convolution bwd
    const GpuConvolutionBwdSignatureKey unregisteredKey{
        DataType::INT8, DataType::INT8, DataType::INT8, DataType::FLOAT};

    EXPECT_THROW(registry.getPlanBuilder(unregisteredKey), std::runtime_error);
}

// ============================================================================
// FP32 plan execution tests
// ============================================================================

TEST(TestGpuConvolutionBwdPlan, ExecutePlan)
{
    SKIP_IF_NO_DEVICES();

    runBwdPlanExecuteVsCpuRef<float, float, float, float>(
        {1, 1, 2, 2}, {1, 1, 3, 3}, {1, 1, 4, 4}, {0, 0}, {1, 1}, {1, 1});
}

TEST(TestGpuConvolutionBwdPlan, ExecutePlanMultiChannel)
{
    SKIP_IF_NO_DEVICES();

    runBwdPlanExecuteVsCpuRef<float, float, float, float>(
        {1, 6, 8, 8}, {6, 3, 3, 3}, {1, 3, 8, 8}, {1, 1}, {1, 1}, {1, 1});
}

TEST(TestGpuConvolutionBwdPlan, ExecutePlanStride2)
{
    SKIP_IF_NO_DEVICES();

    runBwdPlanExecuteVsCpuRef<float, float, float, float>(
        {1, 2, 3, 3}, {2, 1, 3, 3}, {1, 1, 8, 8}, {0, 0}, {2, 2}, {1, 1});
}

// ============================================================================
// FP16 plan execution tests
// ============================================================================

TEST(TestGpuConvolutionBwdPlanFp16, ExecutePlan)
{
    SKIP_IF_NO_DEVICES();

    runBwdPlanExecuteVsCpuRef<half, half, half, float>(
        {1, 1, 2, 2}, {1, 1, 3, 3}, {1, 1, 4, 4}, {0, 0}, {1, 1}, {1, 1});
}

// ============================================================================
// BF16 plan execution tests
// ============================================================================

TEST(TestGpuConvolutionBwdPlanBfp16, ExecutePlan)
{
    SKIP_IF_NO_DEVICES();

    runBwdPlanExecuteVsCpuRef<bfloat16, bfloat16, bfloat16, float>(
        {1, 1, 2, 2}, {1, 1, 3, 3}, {1, 1, 4, 4}, {0, 0}, {1, 1}, {1, 1});
}
