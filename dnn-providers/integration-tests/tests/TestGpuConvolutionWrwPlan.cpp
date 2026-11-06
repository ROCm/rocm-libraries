// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <cstdint>
#include <hipdnn_data_sdk/types/Bfloat16.hpp>
#include <stdexcept>
#include <unordered_map>
#include <utility>
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

#include "ConvolutionWrwGraphTestUtils.hpp"
#include "RaggedGraphTestUtils.hpp"
#include "harness/gpu-graph-executor/detail/GpuConvolutionWrwPlan.hpp"
#include "harness/gpu-graph-executor/detail/GpuConvolutionWrwSignatureKey.hpp"
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
    createCustomConvWrwGraph(hipdnn_flatbuffers_sdk::data_objects::DataType inputDataType
                             = hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                             hipdnn_flatbuffers_sdk::data_objects::DataType outputDataType
                             = hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                             int64_t xUid = 10,
                             int64_t dyUid = 11,
                             int64_t dwUid = 12)
{
    const std::vector<int64_t> xDims = {4, 4, 4, 4};
    const std::vector<int64_t> xStrides = {64, 16, 4, 1};
    const std::vector<int64_t> dwDims = {4, 4, 1, 1};
    const std::vector<int64_t> dwStrides = {4, 1, 1, 1};
    const std::vector<int64_t> dyDims = {4, 4, 4, 4};
    const std::vector<int64_t> dyStrides = {64, 16, 4, 1};

    return createConvWrwGraph(xUid,
                              dyUid,
                              dwUid,
                              xDims,
                              dyDims,
                              dwDims,
                              xStrides,
                              dyStrides,
                              dwStrides,
                              {0, 0},
                              {1, 1},
                              {1, 1},
                              inputDataType,
                              outputDataType);
}

template <typename XType, typename DyType, typename DwType, typename ComputeType>
void runWrwPlanExecuteVsCpuRef(const std::vector<int64_t>& xDims,
                               const std::vector<int64_t>& dyDims,
                               const std::vector<int64_t>& dwDims,
                               const std::vector<int64_t>& padding,
                               const std::vector<int64_t>& stride,
                               const std::vector<int64_t>& dilation,
                               const std::vector<int64_t>& postPaddingOverride = {})
{
    constexpr int64_t X_UID = 1;
    constexpr int64_t DY_UID = 2;
    constexpr int64_t DW_UID = 3;

    auto xStrides = generateStrides(xDims);
    auto dyStrides = generateStrides(dyDims);
    auto dwStrides = generateStrides(dwDims);

    // An empty override means symmetric padding.
    const auto& postPadding = postPaddingOverride.empty() ? padding : postPaddingOverride;

    auto graphBuilder
        = createConvWrwGraphWithPadding(X_UID,
                                        DY_UID,
                                        DW_UID,
                                        xDims,
                                        dyDims,
                                        dwDims,
                                        xStrides,
                                        dyStrides,
                                        dwStrides,
                                        padding,
                                        postPadding,
                                        stride,
                                        dilation,
                                        hipdnn_test_sdk::utilities::nativeTypeToDataType<XType>(),
                                        hipdnn_test_sdk::utilities::nativeTypeToDataType<DwType>());

    auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        graphBuilder.GetBufferPointer(), graphBuilder.GetSize());

    const auto* nodeAttributes = graphWrap.getNode(0).attributes_as_ConvolutionWrwAttributes();
    const auto& tensorMap = graphWrap.getTensorMap();

    GpuConvolutionWrwPlanParams params(*tensorMap.at(nodeAttributes->x_tensor_uid()),
                                       *tensorMap.at(nodeAttributes->dy_tensor_uid()),
                                       *tensorMap.at(nodeAttributes->dw_tensor_uid()),
                                       padding,
                                       postPadding,
                                       stride,
                                       dilation,
                                       ConvMode::CROSS_CORRELATION);

    GpuConvolutionWrwPlan<XType, DyType, DwType, ComputeType> patient(std::move(params));

    auto xCount = std::accumulate(
        xDims.begin(), xDims.end(), static_cast<size_t>(1), std::multiplies<size_t>());
    auto dyCount = std::accumulate(
        dyDims.begin(), dyDims.end(), static_cast<size_t>(1), std::multiplies<size_t>());
    auto dwCount = std::accumulate(
        dwDims.begin(), dwDims.end(), static_cast<size_t>(1), std::multiplies<size_t>());

    // Prepare CPU tensors and fill with random data
    hipdnn_data_sdk::utilities::Tensor<XType> cpuX(xDims, xStrides);
    hipdnn_data_sdk::utilities::Tensor<DyType> cpuDy(dyDims, dyStrides);
    hipdnn_data_sdk::utilities::Tensor<DwType> cpuDw(dwDims, dwStrides);

    constexpr unsigned int SEED = 42;
    fillWithRandomValues(cpuX, static_cast<XType>(-1), static_cast<XType>(1), SEED);
    fillWithRandomValues(cpuDy, static_cast<DyType>(-1), static_cast<DyType>(1), SEED + 1);

    // Allocate device buffers (RAII — freed automatically)
    const hipdnn_data_sdk::utilities::Workspace dX(xCount * sizeof(XType));
    const hipdnn_data_sdk::utilities::Workspace dDy(dyCount * sizeof(DyType));
    const hipdnn_data_sdk::utilities::Workspace dDw(dwCount * sizeof(DwType));

    ASSERT_EQ(
        hipMemcpy(dX.get(), cpuX.rawHostData(), xCount * sizeof(XType), hipMemcpyHostToDevice),
        hipSuccess);
    ASSERT_EQ(
        hipMemcpy(dDy.get(), cpuDy.rawHostData(), dyCount * sizeof(DyType), hipMemcpyHostToDevice),
        hipSuccess);
    ASSERT_EQ(hipMemset(dDw.get(), 0, dwCount * sizeof(DwType)), hipSuccess);

    // Execute GPU plan
    std::unordered_map<int64_t, void*> variantPack;
    variantPack[X_UID] = dX.get();
    variantPack[DY_UID] = dDy.get();
    variantPack[DW_UID] = dDw.get();

    patient.execute(variantPack);

    // Copy result back
    std::vector<DwType> gpuDwData(dwCount);
    ASSERT_EQ(
        hipMemcpy(gpuDwData.data(), dDw.get(), dwCount * sizeof(DwType), hipMemcpyDeviceToHost),
        hipSuccess);

    // Run CPU reference executor with host pointers (same graph)
    std::unordered_map<int64_t, void*> cpuVariantPack;
    cpuVariantPack[X_UID] = cpuX.rawHostData();
    cpuVariantPack[DY_UID] = cpuDy.rawHostData();
    cpuVariantPack[DW_UID] = cpuDw.rawHostData();

    hipdnn_test_sdk::utilities::CpuReferenceGraphExecutor cpuExecutor;
    cpuExecutor.execute(graphBuilder.GetBufferPointer(), graphBuilder.GetSize(), cpuVariantPack);

    // Compare GPU output against CPU reference
    const auto* cpuResult = static_cast<const DwType*>(cpuDw.rawHostData());
    for(size_t i = 0; i < dwCount; ++i)
    {
        EXPECT_NEAR(static_cast<float>(gpuDwData[i]),
                    static_cast<float>(cpuResult[i]),
                    hipdnn_test_sdk::utilities::conv::getToleranceWrw<DwType>())
            << "Mismatch at index " << i;
    }
}

} // namespace

TEST(TestGpuConvolutionWrwPlanBuilder, PlanConstruction)
{
    auto graphBuilder = createCustomConvWrwGraph();
    auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        graphBuilder.GetBufferPointer(), graphBuilder.GetSize());

    const GpuConvolutionWrwPlanBuilder<DataType::FLOAT,
                                       DataType::FLOAT,
                                       DataType::FLOAT,
                                       DataType::FLOAT>
        patient;

    auto builtPlan = patient.buildNodePlan(graphWrap, graphWrap.getNode(0));

    const bool result
        = dynamic_cast<GpuConvolutionWrwPlan<float, float, float, float>*>(builtPlan.get())
          != nullptr;
    EXPECT_TRUE(result);
}

TEST(TestGpuConvolutionWrwPlanBuilder, IsApplicable)
{
    auto graphBuilder = createCustomConvWrwGraph();
    auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        graphBuilder.GetBufferPointer(), graphBuilder.GetSize());

    const GpuConvolutionWrwPlanBuilder<DataType::FLOAT,
                                       DataType::FLOAT,
                                       DataType::FLOAT,
                                       DataType::FLOAT>
        floatPlanBuilder;

    EXPECT_TRUE(floatPlanBuilder.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap()));

    // Half builder should not be applicable for a float graph
    const GpuConvolutionWrwPlanBuilder<DataType::HALF,
                                       DataType::HALF,
                                       DataType::HALF,
                                       DataType::FLOAT>
        halfPlanBuilder;

    EXPECT_FALSE(halfPlanBuilder.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap()));

    // Missing tensor should return false
    auto tensorMapCopy = graphWrap.getTensorMap();
    tensorMapCopy.erase(11); // DY_UID
    EXPECT_FALSE(floatPlanBuilder.isApplicable(graphWrap.getNode(0), tensorMapCopy));
}

TEST(TestGpuConvolutionWrwPlanBuilder, IsNotApplicableForRaggedTensors)
{
    auto graphBuilder = createCustomConvWrwGraph();

    const GpuConvolutionWrwPlanBuilder<DataType::FLOAT,
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

TEST(TestGpuConvolutionWrwPlanBuilder, IsApplicableFalseWhenXTypeMismatched)
{
    auto graphBuilder = createCustomConvWrwGraph(DataType::HALF, DataType::FLOAT);
    auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        graphBuilder.GetBufferPointer(), graphBuilder.GetSize());

    const GpuConvolutionWrwPlanBuilder<DataType::FLOAT,
                                       DataType::FLOAT,
                                       DataType::FLOAT,
                                       DataType::FLOAT>
        patient;

    EXPECT_FALSE(patient.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap()));
}

TEST(TestGpuConvolutionWrwPlanBuilder, IsApplicableFalseWhenDyTypeMismatched)
{
    constexpr int64_t X_UID = 10;
    constexpr int64_t DY_UID = 11;
    constexpr int64_t DW_UID = 12;

    const std::vector<int64_t> xDims = {4, 4, 4, 4};
    const std::vector<int64_t> xStrides = {64, 16, 4, 1};
    const std::vector<int64_t> dyDims = {4, 4, 4, 4};
    const std::vector<int64_t> dyStrides = {64, 16, 4, 1};
    const std::vector<int64_t> dwDims = {4, 4, 1, 1};
    const std::vector<int64_t> dwStrides = {4, 1, 1, 1};

    flatbuffers::FlatBufferBuilder builder;
    std::vector<flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>>
        tensors;

    tensors.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        X_UID,
        "x",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &xStrides,
        &xDims));
    tensors.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        DY_UID,
        "dy",
        hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
        &dyStrides,
        &dyDims));
    tensors.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        DW_UID,
        "dw",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &dwStrides,
        &dwDims));

    const std::vector<int64_t> padding = {0, 0};
    const std::vector<int64_t> stride = {1, 1};
    const std::vector<int64_t> dilation = {1, 1};
    auto convWrwAttrs = hipdnn_flatbuffers_sdk::data_objects::CreateConvolutionWrwAttributesDirect(
        builder,
        X_UID,
        DY_UID,
        DW_UID,
        &padding,
        &padding,
        &stride,
        &dilation,
        hipdnn_flatbuffers_sdk::data_objects::ConvMode::CROSS_CORRELATION);

    std::vector<flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Node>> nodes;
    nodes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "conv_wrw_node",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::ConvolutionWrwAttributes,
        convWrwAttrs.Union()));

    auto graph = hipdnn_flatbuffers_sdk::data_objects::CreateGraphDirect(
        builder,
        "ConvWrwTestGraph",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &tensors,
        &nodes);
    builder.Finish(graph);

    auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    const GpuConvolutionWrwPlanBuilder<DataType::FLOAT,
                                       DataType::FLOAT,
                                       DataType::FLOAT,
                                       DataType::FLOAT>
        patient;

    EXPECT_FALSE(patient.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap()));
}

TEST(TestGpuConvolutionWrwPlanBuilder, IsApplicableFalseWhenDwTypeMismatched)
{
    auto graphBuilder = createCustomConvWrwGraph(DataType::FLOAT, DataType::HALF);
    auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        graphBuilder.GetBufferPointer(), graphBuilder.GetSize());

    const GpuConvolutionWrwPlanBuilder<DataType::FLOAT,
                                       DataType::FLOAT,
                                       DataType::FLOAT,
                                       DataType::FLOAT>
        patient;

    EXPECT_FALSE(patient.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap()));
}

TEST(TestGpuConvolutionWrwPlanBuilder, IsApplicableFalseWhenComputeTypeMismatched)
{
    auto graphBuilder = createCustomConvWrwGraph();
    auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        graphBuilder.GetBufferPointer(), graphBuilder.GetSize());

    const GpuConvolutionWrwPlanBuilder<DataType::FLOAT,
                                       DataType::FLOAT,
                                       DataType::FLOAT,
                                       DataType::HALF>
        patient;

    EXPECT_FALSE(patient.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap()));
}

TEST(TestGpuConvolutionWrwPlanBuilder, IsApplicableFalseWhenOnlyOneOperandTypeDiffers)
{
    const GpuConvolutionWrwPlanBuilder<DataType::FLOAT,
                                       DataType::FLOAT,
                                       DataType::FLOAT,
                                       DataType::FLOAT>
        patient;

    // Each operand is changed on its own, so every per-operand type check is isolated.
    for(const auto& [name, uid] :
        std::vector<std::pair<const char*, int64_t>>{{"x", 10}, {"dy", 11}, {"dw", 12}})
    {
        SCOPED_TRACE(name);
        auto graphBuilder = createCustomConvWrwGraph();
        auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
            graphBuilder.GetBufferPointer(), graphBuilder.GetSize());
        ASSERT_TRUE(patient.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap()));

        auto* tensor = const_cast<TensorAttributes*>(graphWrap.getTensorMap().at(uid));
        ASSERT_TRUE(tensor->mutate_data_type(DataType::HALF));

        EXPECT_FALSE(patient.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap()));
    }
}

TEST(TestGpuConvolutionWrwPlanBuilder, IsApplicableFalseWhenOperandIsRuntimePassByValue)
{
    const GpuConvolutionWrwPlanBuilder<DataType::FLOAT,
                                       DataType::FLOAT,
                                       DataType::FLOAT,
                                       DataType::FLOAT>
        patient;

    for(const auto& [name, uid] :
        std::vector<std::pair<const char*, int64_t>>{{"x", 10}, {"dy", 11}, {"dw", 12}})
    {
        SCOPED_TRACE(name);
        auto graphBuilder = createCustomConvWrwGraph();
        auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
            graphBuilder.GetBufferPointer(), graphBuilder.GetSize());
        ASSERT_TRUE(patient.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap()));

        // The flag is not stored when false, so it cannot be mutated in place; swap in a
        // tensor that is marked pass-by-value instead.
        flatbuffers::FlatBufferBuilder tensorBuilder;
        const std::vector<int64_t> dims = {1};
        tensorBuilder.Finish(CreateTensorAttributesDirect(tensorBuilder,
                                                          uid,
                                                          "pbv",
                                                          DataType::FLOAT,
                                                          &dims,
                                                          &dims,
                                                          /*virtual_=*/false,
                                                          TensorValue::NONE,
                                                          /*value=*/0,
                                                          /*is_runtime_pass_by_value=*/true));
        auto tensorMap = graphWrap.getTensorMap();
        tensorMap[uid] = flatbuffers::GetRoot<TensorAttributes>(tensorBuilder.GetBufferPointer());

        EXPECT_FALSE(patient.isApplicable(graphWrap.getNode(0), tensorMap));
    }
}

TEST(TestGpuConvolutionWrwPlanBuilder, BuildNodePlanThrowsForWrongAttributesType)
{
    auto builder = hipdnn_test_sdk::utilities::createValidConvFwdGraph();
    auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    const GpuConvolutionWrwPlanBuilder<DataType::FLOAT,
                                       DataType::FLOAT,
                                       DataType::FLOAT,
                                       DataType::FLOAT>
        patient;

    EXPECT_THROW(patient.buildNodePlan(graphWrap, graphWrap.getNode(0)), std::runtime_error);
}

TEST(TestGpuConvolutionWrwPlanBuilder, IsApplicableReturnsFalseForWrongAttributesType)
{
    auto builder = hipdnn_test_sdk::utilities::createValidConvFwdGraph();
    auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    const GpuConvolutionWrwPlanBuilder<DataType::FLOAT,
                                       DataType::FLOAT,
                                       DataType::FLOAT,
                                       DataType::FLOAT>
        patient;

    EXPECT_FALSE(patient.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap()));
}

TEST(TestGpuConvolutionWrwPlanBuilder, IsApplicableFalseWhenXTensorMissing)
{
    constexpr int64_t X_UID = 10;
    auto graphBuilder = createCustomConvWrwGraph(DataType::FLOAT, DataType::FLOAT, X_UID);
    auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        graphBuilder.GetBufferPointer(), graphBuilder.GetSize());

    const GpuConvolutionWrwPlanBuilder<DataType::FLOAT,
                                       DataType::FLOAT,
                                       DataType::FLOAT,
                                       DataType::FLOAT>
        patient;

    auto tensorMapCopy = graphWrap.getTensorMap();
    tensorMapCopy.erase(X_UID);

    EXPECT_FALSE(patient.isApplicable(graphWrap.getNode(0), tensorMapCopy));
}

TEST(TestGpuConvolutionWrwPlanBuilder, IsApplicableFalseWhenDyTensorMissing)
{
    constexpr int64_t DY_UID = 11;
    auto graphBuilder = createCustomConvWrwGraph(DataType::FLOAT, DataType::FLOAT, 10, DY_UID);
    auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        graphBuilder.GetBufferPointer(), graphBuilder.GetSize());

    const GpuConvolutionWrwPlanBuilder<DataType::FLOAT,
                                       DataType::FLOAT,
                                       DataType::FLOAT,
                                       DataType::FLOAT>
        patient;

    auto tensorMapCopy = graphWrap.getTensorMap();
    tensorMapCopy.erase(DY_UID);

    EXPECT_FALSE(patient.isApplicable(graphWrap.getNode(0), tensorMapCopy));
}

TEST(TestGpuConvolutionWrwPlanBuilder, IsApplicableFalseWhenDwTensorMissing)
{
    constexpr int64_t DW_UID = 12;
    auto graphBuilder = createCustomConvWrwGraph(DataType::FLOAT, DataType::FLOAT, 10, 11, DW_UID);
    auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        graphBuilder.GetBufferPointer(), graphBuilder.GetSize());

    const GpuConvolutionWrwPlanBuilder<DataType::FLOAT,
                                       DataType::FLOAT,
                                       DataType::FLOAT,
                                       DataType::FLOAT>
        patient;

    auto tensorMapCopy = graphWrap.getTensorMap();
    tensorMapCopy.erase(DW_UID);

    EXPECT_FALSE(patient.isApplicable(graphWrap.getNode(0), tensorMapCopy));
}

TEST(TestGpuConvolutionWrwPlanBuilder, UnregisteredSignatureThrows)
{
    GpuPlanBuilderRegistry registry;

    // INT8 is not in the registered signatures for convolution wrw
    const GpuConvolutionWrwSignatureKey unregisteredKey{
        DataType::INT8, DataType::INT8, DataType::INT8, DataType::FLOAT};

    EXPECT_THROW(registry.getPlanBuilder(unregisteredKey), std::runtime_error);
}

// ============================================================================
// FP32 plan execution tests
// ============================================================================

TEST(TestGpuConvolutionWrwPlan, ExecutePlan)
{
    SKIP_IF_NO_DEVICES();

    runWrwPlanExecuteVsCpuRef<float, float, float, float>(
        {1, 1, 4, 4}, {1, 1, 2, 2}, {1, 1, 3, 3}, {0, 0}, {1, 1}, {1, 1});
}

TEST(TestGpuConvolutionWrwPlan, ExecutePlanMultiChannel)
{
    SKIP_IF_NO_DEVICES();

    runWrwPlanExecuteVsCpuRef<float, float, float, float>(
        {1, 3, 8, 8}, {1, 6, 8, 8}, {6, 3, 3, 3}, {1, 1}, {1, 1}, {1, 1});
}

TEST(TestGpuConvolutionWrwPlan, ExecutePlanStride2)
{
    SKIP_IF_NO_DEVICES();

    runWrwPlanExecuteVsCpuRef<float, float, float, float>(
        {1, 1, 8, 8}, {1, 2, 3, 3}, {2, 1, 3, 3}, {0, 0}, {2, 2}, {1, 1});
}

TEST(TestGpuConvolutionWrwPlan, ExecutePlanDilation)
{
    SKIP_IF_NO_DEVICES();

    runWrwPlanExecuteVsCpuRef<float, float, float, float>(
        {1, 2, 6, 6}, {1, 3, 4, 4}, {3, 2, 3, 3}, {1, 1}, {1, 1}, {2, 2});
}

TEST(TestGpuConvolutionWrwPlan, ExecutePlanAsymmetricPadding)
{
    SKIP_IF_NO_DEVICES();

    runWrwPlanExecuteVsCpuRef<float, float, float, float>(
        {1, 2, 6, 6}, {1, 3, 6, 5}, {3, 2, 3, 3}, {2, 0}, {1, 1}, {1, 1}, {0, 1});
}

// ============================================================================
// FP16 plan execution tests
// ============================================================================

TEST(TestGpuConvolutionWrwPlanFp16, ExecutePlan)
{
    SKIP_IF_NO_DEVICES();

    runWrwPlanExecuteVsCpuRef<half, half, half, float>(
        {1, 1, 4, 4}, {1, 1, 2, 2}, {1, 1, 3, 3}, {0, 0}, {1, 1}, {1, 1});
}

// ============================================================================
// BF16 plan execution tests
// ============================================================================

TEST(TestGpuConvolutionWrwPlanBfp16, ExecutePlan)
{
    SKIP_IF_NO_DEVICES();

    runWrwPlanExecuteVsCpuRef<bfloat16, bfloat16, bfloat16, float>(
        {1, 1, 4, 4}, {1, 1, 2, 2}, {1, 1, 3, 3}, {0, 0}, {1, 1}, {1, 1});
}

TEST(TestGpuConvolutionWrwPlanBuilder, IsNotApplicableForConvolutionMode)
{
    auto graphBuilder = createCustomConvWrwGraph();
    auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        graphBuilder.GetBufferPointer(), graphBuilder.GetSize());

    const GpuConvolutionWrwPlanBuilder<DataType::FLOAT,
                                       DataType::FLOAT,
                                       DataType::FLOAT,
                                       DataType::FLOAT>
        patient;

    EXPECT_TRUE(patient.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap()));

    // The kernels are cross-correlation only, so true convolution must be rejected.
    auto* attributes = const_cast<ConvolutionWrwAttributes*>(
        graphWrap.getNode(0).attributes_as_ConvolutionWrwAttributes());
    ASSERT_NE(attributes, nullptr);
    ASSERT_TRUE(attributes->mutate_conv_mode(ConvMode::CONVOLUTION));

    EXPECT_FALSE(patient.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap()));
}

TEST(TestGpuConvolutionWrwPlanBuilder, PureReducedPrecisionComputeSignatureThrows)
{
    GpuPlanBuilderRegistry registry;

    // Reduced-precision compute would accumulate in 16 bits, so it is not registered.
    const GpuConvolutionWrwSignatureKey halfKey{
        DataType::HALF, DataType::HALF, DataType::HALF, DataType::HALF};
    const GpuConvolutionWrwSignatureKey bfloat16Key{
        DataType::BFLOAT16, DataType::BFLOAT16, DataType::BFLOAT16, DataType::BFLOAT16};

    EXPECT_THROW(registry.getPlanBuilder(halfKey), std::runtime_error);
    EXPECT_THROW(registry.getPlanBuilder(bfloat16Key), std::runtime_error);
}
