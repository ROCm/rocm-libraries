// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cstdint>
#include <gtest/gtest.h>

#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "RaggedGraphTestUtils.hpp"
#include "harness/gpu-graph-executor/detail/GpuResamplePlan.hpp"
#include <hipdnn-gpu-ref/GpuFpReferenceCommon.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/TestTolerances.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/CpuReferenceGraphExecutor.hpp>

using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_integration_tests::gpu_graph_executor::detail;
using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_integration_tests::test_utils;
using namespace hipdnn_gpu_ref::common::gpu_fp_reference_tensor;

// =============================================================
// Test GpuResampleFwdPlan
// =============================================================

TEST(TestGpuResampleFwdPlanBuilder, PlanConstruction)
{
    auto builder = hipdnn_test_sdk::utilities::createValidResampleFwdGraph(true);
    auto graph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    const GpuResampleFwdPlanBuilder<DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::INT32>
        patient;

    auto builtPlan = patient.buildNodePlan(graph, graph.getNode(0));

    const bool result
        = dynamic_cast<GpuResampleFwdPlan<float, float, float, int32_t>*>(builtPlan.get())
          != nullptr;
    EXPECT_TRUE(result);
}

TEST(TestGpuResampleFwdPlanBuilder, IsApplicable)
{
    auto builder = hipdnn_test_sdk::utilities::createValidResampleFwdGraph();
    auto graph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    const GpuResampleFwdPlanBuilder<DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::UNSET>
        floatPlanBuilder;
    EXPECT_TRUE(floatPlanBuilder.isApplicable(graph.getNode(0), graph.getTensorMap()));

    // Half builder must not be applicable to a float graph
    const GpuResampleFwdPlanBuilder<DataType::HALF, DataType::HALF, DataType::HALF, DataType::UNSET>
        halfPlanBuilder;
    EXPECT_FALSE(halfPlanBuilder.isApplicable(graph.getNode(0), graph.getTensorMap()));

    // Missing input tensor must make the plan inapplicable
    auto tensorMapCopy = graph.getTensorMap();
    const auto* nodeAttributes = graph.getNode(0).attributes_as_ResampleFwdAttributes();
    EXPECT_NE(nodeAttributes, nullptr);
    tensorMapCopy.erase(nodeAttributes->x_tensor_uid());
    EXPECT_FALSE(floatPlanBuilder.isApplicable(graph.getNode(0), tensorMapCopy));
}

TEST(TestGpuResampleFwdPlanBuilder, IsNotApplicableForRaggedTensors)
{
    auto builder = hipdnn_test_sdk::utilities::createValidResampleFwdGraph();

    const GpuResampleFwdPlanBuilder<DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::UNSET>
        patient;

    auto denseWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        builder.GetBufferPointer(), builder.GetSize());
    ASSERT_TRUE(patient.isApplicable(denseWrap.getNode(0), denseWrap.getTensorMap()));

    auto ragged
        = hipdnn_integration_tests::test_utils::markFirstTensorRagged(builder.GetBufferPointer());
    auto raggedWrap
        = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(ragged.data(), ragged.size());
    EXPECT_FALSE(patient.isApplicable(raggedWrap.getNode(0), raggedWrap.getTensorMap()));
}

TEST(TestGpuResampleFwdPlanBuilder, BuildNodePlanThrowsForWrongAttributesType)
{
    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormGraph();
    auto graph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    const GpuResampleFwdPlanBuilder<DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::INT32>
        patient;

    EXPECT_THROW(patient.buildNodePlan(graph, graph.getNode(0)), std::runtime_error);
}

TEST(TestGpuResampleFwdPlanBuilder, IsApplicableReturnsFalseForWrongAttributesType)
{
    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormGraph();
    auto graph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    const GpuResampleFwdPlanBuilder<DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::INT32>
        patient;

    EXPECT_FALSE(patient.isApplicable(graph.getNode(0), graph.getTensorMap()));
}

TEST(TestGpuResampleFwdPlanBuilder, IsApplicableFalseWhenOutputTensorMissing)
{
    auto builder = hipdnn_test_sdk::utilities::createValidResampleFwdGraph(true);
    auto graph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    const GpuResampleFwdPlanBuilder<DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::INT32>
        patient;

    auto tensorMapCopy = graph.getTensorMap();
    const auto* nodeAttributes = graph.getNode(0).attributes_as_ResampleFwdAttributes();
    EXPECT_NE(nodeAttributes, nullptr);
    tensorMapCopy.erase(nodeAttributes->y_tensor_uid());
    EXPECT_FALSE(patient.isApplicable(graph.getNode(0), tensorMapCopy));
}

TEST(TestGpuResampleFwdPlanBuilder, IsApplicableFalseWhenTensorTypeMismatched)
{
    auto builder = hipdnn_test_sdk::utilities::createValidResampleFwdGraph();
    auto graph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    const GpuResampleFwdPlanBuilder<DataType::HALF,
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::UNSET>
        patient1;

    EXPECT_FALSE(patient1.isApplicable(graph.getNode(0), graph.getTensorMap()));

    const GpuResampleFwdPlanBuilder<DataType::FLOAT,
                                    DataType::BFLOAT16,
                                    DataType::FLOAT,
                                    DataType::UNSET>
        patient2;
    EXPECT_FALSE(patient2.isApplicable(graph.getNode(0), graph.getTensorMap()));
}

TEST(TestGpuResampleFwdPlanBuilder, IsApplicableFalseForUnsupportedModes)
{
    auto builder = hipdnn_test_sdk::utilities::createValidResampleFwdGraph(true);
    auto graph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    const GpuResampleFwdPlanBuilder<DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::INT32>
        patient;

    auto nodeAttributes = const_cast<ResampleFwdAttributes*>(
        graph.getNode(0).attributes_as_ResampleFwdAttributes());
    EXPECT_NE(nodeAttributes, nullptr);
    ASSERT_TRUE(nodeAttributes->mutate_resample_mode(
        static_cast<ResampleMode>(99))); // Invalid resample mode

    EXPECT_FALSE(patient.isApplicable(graph.getNode(0), graph.getTensorMap()));

    ASSERT_TRUE(nodeAttributes->mutate_resample_mode(ResampleMode::MAXPOOL));
    ASSERT_TRUE(
        nodeAttributes->mutate_padding_mode(static_cast<PaddingMode>(99))); // Invalid padding mode
    EXPECT_FALSE(patient.isApplicable(graph.getNode(0), graph.getTensorMap()));
}

TEST(TestGpuResampleFwdPlanBuilder, IsApplicableFalseWithIndexTensor)
{
    auto builder = hipdnn_test_sdk::utilities::createValidResampleFwdGraph();
    auto graph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    const GpuResampleFwdPlanBuilder<DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::INT32>
        patient;
    EXPECT_FALSE(patient.isApplicable(graph.getNode(0), graph.getTensorMap()));

    auto builderWithIndex = hipdnn_test_sdk::utilities::createValidResampleFwdGraph(true);
    auto graphWithIndex = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        builderWithIndex.GetBufferPointer(), builderWithIndex.GetSize());

    const GpuResampleFwdPlanBuilder<DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::UNSET>
        patientIndexUnset;
    EXPECT_FALSE(
        patientIndexUnset.isApplicable(graphWithIndex.getNode(0), graphWithIndex.getTensorMap()));

    auto tensorMapCopy = graphWithIndex.getTensorMap();
    const auto* nodeAttributes = graphWithIndex.getNode(0).attributes_as_ResampleFwdAttributes();
    EXPECT_NE(nodeAttributes, nullptr);
    tensorMapCopy.erase(nodeAttributes->index_tensor_uid().value());
    EXPECT_FALSE(patient.isApplicable(graphWithIndex.getNode(0), tensorMapCopy));
}

// =============================================================
// Test GpuResampleBwdPlan
// =============================================================

TEST(TestGpuResampleBwdPlanBuilder, PlanConstruction)
{
    auto builder = hipdnn_test_sdk::utilities::createValidResampleBwdGraph();
    auto graph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    const GpuResampleBwdPlanBuilder<DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::INT32>
        patient;

    auto builtPlan = patient.buildNodePlan(graph, graph.getNode(0));

    const bool result
        = dynamic_cast<GpuResampleBwdPlan<float, float, float, int32_t>*>(builtPlan.get())
          != nullptr;
    EXPECT_TRUE(result);
}

TEST(TestGpuResampleBwdPlanBuilder, IsApplicable)
{
    auto builder = hipdnn_test_sdk::utilities::createValidResampleBwdGraph();
    auto graph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    const GpuResampleBwdPlanBuilder<DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::INT32>
        floatPlanBuilder;
    EXPECT_TRUE(floatPlanBuilder.isApplicable(graph.getNode(0), graph.getTensorMap()));

    // Half builder must not be applicable to a float graph
    const GpuResampleBwdPlanBuilder<DataType::HALF, DataType::HALF, DataType::HALF, DataType::INT32>
        halfPlanBuilder;
    EXPECT_FALSE(halfPlanBuilder.isApplicable(graph.getNode(0), graph.getTensorMap()));

    // Missing input tensor must make the plan inapplicable
    auto tensorMapCopy = graph.getTensorMap();
    const auto* nodeAttributes = graph.getNode(0).attributes_as_ResampleBwdAttributes();
    EXPECT_NE(nodeAttributes, nullptr);
    tensorMapCopy.erase(nodeAttributes->dy_tensor_uid());
    EXPECT_FALSE(floatPlanBuilder.isApplicable(graph.getNode(0), tensorMapCopy));
}

TEST(TestGpuResampleBwdPlanBuilder, IsNotApplicableForRaggedTensors)
{
    auto builder = hipdnn_test_sdk::utilities::createValidResampleBwdGraph();

    const GpuResampleBwdPlanBuilder<DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::INT32>
        patient;

    auto denseWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        builder.GetBufferPointer(), builder.GetSize());
    ASSERT_TRUE(patient.isApplicable(denseWrap.getNode(0), denseWrap.getTensorMap()));

    auto ragged
        = hipdnn_integration_tests::test_utils::markFirstTensorRagged(builder.GetBufferPointer());
    auto raggedWrap
        = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(ragged.data(), ragged.size());
    EXPECT_FALSE(patient.isApplicable(raggedWrap.getNode(0), raggedWrap.getTensorMap()));
}

TEST(TestGpuResampleBwdPlanBuilder, BuildNodePlanThrowsForWrongAttributesType)
{
    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormBwdGraph();
    auto graph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    const GpuResampleBwdPlanBuilder<DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::INT32>
        patient;

    EXPECT_THROW(patient.buildNodePlan(graph, graph.getNode(0)), std::runtime_error);
}

TEST(TestGpuResampleBwdPlanBuilder, IsApplicableReturnsFalseForWrongAttributesType)
{
    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormBwdGraph();
    auto graph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    const GpuResampleBwdPlanBuilder<DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::INT32>
        patient;

    EXPECT_FALSE(patient.isApplicable(graph.getNode(0), graph.getTensorMap()));
}

TEST(TestGpuResampleBwdPlanBuilder, IsApplicableFalseWhenInputTensorMissing)
{
    auto builder = hipdnn_test_sdk::utilities::createValidResampleBwdGraph();
    auto graph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    const GpuResampleBwdPlanBuilder<DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::INT32>
        patient;

    auto tensorMapCopy = graph.getTensorMap();
    const auto* nodeAttributes = graph.getNode(0).attributes_as_ResampleBwdAttributes();
    EXPECT_NE(nodeAttributes, nullptr);
    tensorMapCopy.erase(nodeAttributes->dy_tensor_uid());
    EXPECT_FALSE(patient.isApplicable(graph.getNode(0), tensorMapCopy));
}

TEST(TestGpuResampleBwdPlanBuilder, IsApplicableFalseWhenTensorTypeMismatched)
{
    auto builder = hipdnn_test_sdk::utilities::createValidResampleBwdGraph();
    auto graph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    const GpuResampleBwdPlanBuilder<DataType::HALF,
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::INT32>
        patient1;

    EXPECT_FALSE(patient1.isApplicable(graph.getNode(0), graph.getTensorMap()));

    const GpuResampleBwdPlanBuilder<DataType::FLOAT,
                                    DataType::BFLOAT16,
                                    DataType::FLOAT,
                                    DataType::INT32>
        patient2;
    EXPECT_FALSE(patient2.isApplicable(graph.getNode(0), graph.getTensorMap()));
}

TEST(TestGpuResampleBwdPlanBuilder, IsApplicableFalseForUnsupportedModes)
{
    auto builder = hipdnn_test_sdk::utilities::createValidResampleBwdGraph();
    auto graph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    const GpuResampleBwdPlanBuilder<DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::INT32>
        patient;

    auto nodeAttributes = const_cast<ResampleBwdAttributes*>(
        graph.getNode(0).attributes_as_ResampleBwdAttributes());
    EXPECT_NE(nodeAttributes, nullptr);
    ASSERT_TRUE(nodeAttributes->mutate_resample_mode(
        static_cast<ResampleMode>(99))); // Invalid resample mode

    EXPECT_FALSE(patient.isApplicable(graph.getNode(0), graph.getTensorMap()));

    ASSERT_TRUE(nodeAttributes->mutate_resample_mode(ResampleMode::MAXPOOL));
    ASSERT_TRUE(
        nodeAttributes->mutate_padding_mode(static_cast<PaddingMode>(99))); // Invalid padding mode
    EXPECT_FALSE(patient.isApplicable(graph.getNode(0), graph.getTensorMap()));
}

TEST(TestGpuResampleBwdPlanBuilder, IsApplicableFalseWithIndexTensor)
{
    auto builder
        = hipdnn_test_sdk::utilities::createValidResampleBwdGraph(false, ResampleMode::MAXPOOL);
    auto graph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    const GpuResampleBwdPlanBuilder<DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::INT32>
        patient;
    EXPECT_FALSE(patient.isApplicable(graph.getNode(0), graph.getTensorMap()));

    auto builderWithIndex = hipdnn_test_sdk::utilities::createValidResampleBwdGraph(true);
    auto graphWithIndex = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        builderWithIndex.GetBufferPointer(), builderWithIndex.GetSize());

    const GpuResampleBwdPlanBuilder<DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::UNSET>
        patientIndexUnset;
    EXPECT_FALSE(
        patientIndexUnset.isApplicable(graphWithIndex.getNode(0), graphWithIndex.getTensorMap()));
    EXPECT_FALSE(patientIndexUnset.isApplicable(
        graph.getNode(0), graph.getTensorMap())); // Maxpool without index tensor should fail

    auto tensorMapCopy = graphWithIndex.getTensorMap();
    const auto* nodeAttributes = graphWithIndex.getNode(0).attributes_as_ResampleBwdAttributes();
    EXPECT_NE(nodeAttributes, nullptr);
    tensorMapCopy.erase(nodeAttributes->index_tensor_uid().value());
    EXPECT_FALSE(patient.isApplicable(graphWithIndex.getNode(0), tensorMapCopy));
}

namespace
{

template <typename XType, typename YType, typename ComputeType, typename IndexType>
void runResampleFwdPlanExecuteVsCpuRef(const std::vector<int64_t>& xDims,
                                       const TensorLayout& layout,
                                       const std::vector<int64_t>& prePadding,
                                       const std::vector<int64_t>& postPadding,
                                       const std::vector<int64_t>& stride,
                                       const std::vector<int64_t>& window,
                                       ResampleMode resampleMode,
                                       PaddingMode paddingMode)
{
    constexpr int64_t X_UID = 1;
    constexpr int64_t Y_UID = 2;
    constexpr int64_t INDEX_UID = 3;

    const auto xDataType = nativeTypeToDataType<XType>();
    const auto yDataType = nativeTypeToDataType<YType>();
    const auto computeDataType = nativeTypeToDataType<ComputeType>();
    const auto indexDataType = nativeTypeToDataType<IndexType>();
    const bool hasIndex = resampleMode == ResampleMode::MAXPOOL;

    std::vector<int64_t> yDims = {xDims[0], xDims[1]};
    for(size_t i = 0; i < prePadding.size(); ++i)
    {
        const int64_t paddedDim = xDims[i + 2] + prePadding[i] + postPadding[i];
        yDims.push_back((paddedDim - window[i]) / stride[i] + 1);
    }

    const auto xStrides = generateStrides(xDims, layout.strideOrder);
    const auto yStrides = generateStrides(yDims, layout.strideOrder);

    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>>
        tensorAttributes;

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, X_UID, "x", xDataType, &xStrides, &xDims));
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, Y_UID, "y", yDataType, &yStrides, &yDims));

    if(hasIndex)
    {
        tensorAttributes.push_back(
            hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
                builder, INDEX_UID, "index", indexDataType, &yStrides, &yDims));
    }

    auto resampleAttributes
        = hipdnn_flatbuffers_sdk::data_objects::CreateResampleFwdAttributesDirect(
            builder,
            X_UID,
            Y_UID,
            hasIndex ? ::flatbuffers::Optional<int64_t>(INDEX_UID) : ::flatbuffers::nullopt,
            &prePadding,
            &postPadding,
            &stride,
            &window,
            resampleMode,
            paddingMode,
            hasIndex ? ::flatbuffers::Optional<bool>(true) : ::flatbuffers::nullopt);

    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Node>> nodes;
    nodes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "resample_fwd",
        computeDataType,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::ResampleFwdAttributes,
        resampleAttributes.Union()));

    auto graphOffset = hipdnn_flatbuffers_sdk::data_objects::CreateGraphDirect(
        builder,
        "test",
        computeDataType,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &tensorAttributes,
        &nodes);
    builder.Finish(graphOffset);

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto* nodeAttributes = graph.getNode(0).attributes_as_ResampleFwdAttributes();
    ASSERT_NE(nodeAttributes, nullptr);
    const auto& tensorMap = graph.getTensorMap();

    GpuResampleFwdParams params(*tensorMap.at(nodeAttributes->x_tensor_uid()),
                                *tensorMap.at(nodeAttributes->y_tensor_uid()),
                                prePadding,
                                postPadding,
                                stride,
                                window,
                                nodeAttributes->resample_mode(),
                                nodeAttributes->padding_mode(),
                                hasIndex ? tensorMap.at(nodeAttributes->index_tensor_uid().value())
                                         : nullptr);
    GpuResampleFwdPlan<XType, YType, ComputeType, IndexType> gpuPlan(std::move(params));

    Tensor<XType> xTensor(xDims, xStrides);
    Tensor<YType> cpuY(yDims, yStrides);
    Tensor<YType> gpuY(yDims, yStrides);
    Tensor<IndexType> cpuIndex(yDims, yStrides);
    Tensor<IndexType> gpuIndex(yDims, yStrides);

    constexpr unsigned int SEED = 42;
    fillWithRandomValues(xTensor, static_cast<XType>(-1.0f), static_cast<XType>(1.0f), SEED);

    std::unordered_map<int64_t, void*> gpuVariantPack;
    gpuVariantPack[X_UID] = xTensor.rawDeviceData();
    gpuVariantPack[Y_UID] = gpuY.rawDeviceData();
    if(hasIndex)
    {
        gpuVariantPack[INDEX_UID] = gpuIndex.rawDeviceData();
    }
    gpuPlan.execute(gpuVariantPack);
    gpuY.markDeviceModified();
    gpuIndex.markDeviceModified();

    std::unordered_map<int64_t, void*> cpuVariantPack;
    cpuVariantPack[X_UID] = xTensor.rawHostData();
    cpuVariantPack[Y_UID] = cpuY.rawHostData();
    if(hasIndex)
    {
        cpuVariantPack[INDEX_UID] = cpuIndex.rawHostData();
    }

    CpuReferenceGraphExecutor cpuExecutor;
    cpuExecutor.execute(builder.GetBufferPointer(), builder.GetSize(), cpuVariantPack);
    cpuY.markHostModified();
    cpuIndex.markHostModified();

    const float tolerance = resample::getTolerance<YType>();
    const auto* cpuYData = static_cast<const YType*>(cpuY.rawHostData());
    const auto* gpuYData = static_cast<const YType*>(gpuY.rawHostData());
    for(size_t i = 0; i < cpuY.elementCount(); ++i)
    {
        EXPECT_NEAR(static_cast<float>(gpuYData[i]), static_cast<float>(cpuYData[i]), tolerance)
            << "Mismatch in y at index " << i;
    }

    if(hasIndex)
    {
        const auto* cpuIndexData = static_cast<const IndexType*>(cpuIndex.rawHostData());
        const auto* gpuIndexData = static_cast<const IndexType*>(gpuIndex.rawHostData());
        for(size_t i = 0; i < cpuIndex.elementCount(); ++i)
        {
            EXPECT_EQ(gpuIndexData[i], cpuIndexData[i]) << "Mismatch in index at " << i;
        }
    }
}

template <typename DyType, typename DxType, typename ComputeType, typename IndexType>
void runResampleBwdPlanExecuteVsCpuRef(const std::vector<int64_t>& dxDims,
                                       const TensorLayout& layout,
                                       const std::vector<int64_t>& prePadding,
                                       const std::vector<int64_t>& postPadding,
                                       const std::vector<int64_t>& stride,
                                       const std::vector<int64_t>& window,
                                       ResampleMode resampleMode,
                                       PaddingMode paddingMode)
{
    constexpr int64_t DY_UID = 1;
    constexpr int64_t DX_UID = 2;
    constexpr int64_t INDEX_UID = 3;

    const auto dyDataType = nativeTypeToDataType<DyType>();
    const auto dxDataType = nativeTypeToDataType<DxType>();
    const auto computeDataType = nativeTypeToDataType<ComputeType>();
    const auto indexDataType = nativeTypeToDataType<IndexType>();
    const bool hasIndex = resampleMode == ResampleMode::MAXPOOL;

    std::vector<int64_t> dyDims = {dxDims[0], dxDims[1]};
    for(size_t i = 0; i < prePadding.size(); ++i)
    {
        const int64_t paddedDim = dxDims[i + 2] + prePadding[i] + postPadding[i];
        dyDims.push_back((paddedDim - window[i]) / stride[i] + 1);
    }

    const auto dxStrides = generateStrides(dxDims, layout.strideOrder);
    const auto dyStrides = generateStrides(dyDims, layout.strideOrder);

    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>>
        tensorAttributes;

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, DY_UID, "dy", dyDataType, &dyStrides, &dyDims));
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, DX_UID, "dx", dxDataType, &dxStrides, &dxDims));

    if(hasIndex)
    {
        tensorAttributes.push_back(
            hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
                builder, INDEX_UID, "index", indexDataType, &dyStrides, &dyDims));
    }

    auto resampleAttributes
        = hipdnn_flatbuffers_sdk::data_objects::CreateResampleBwdAttributesDirect(
            builder,
            DY_UID,
            DX_UID,
            hasIndex ? ::flatbuffers::Optional<int64_t>(INDEX_UID) : ::flatbuffers::nullopt,
            &prePadding,
            &postPadding,
            &stride,
            &window,
            resampleMode,
            paddingMode);

    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Node>> nodes;
    nodes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "resample_bwd",
        computeDataType,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::ResampleBwdAttributes,
        resampleAttributes.Union()));

    auto graphOffset = hipdnn_flatbuffers_sdk::data_objects::CreateGraphDirect(
        builder,
        "test",
        computeDataType,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &tensorAttributes,
        &nodes);
    builder.Finish(graphOffset);

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto* nodeAttributes = graph.getNode(0).attributes_as_ResampleBwdAttributes();
    ASSERT_NE(nodeAttributes, nullptr);
    const auto& tensorMap = graph.getTensorMap();

    GpuResampleBwdParams params(*tensorMap.at(nodeAttributes->dy_tensor_uid()),
                                *tensorMap.at(nodeAttributes->dx_tensor_uid()),
                                prePadding,
                                postPadding,
                                stride,
                                window,
                                nodeAttributes->resample_mode(),
                                nodeAttributes->padding_mode(),
                                hasIndex ? tensorMap.at(nodeAttributes->index_tensor_uid().value())
                                         : nullptr);
    GpuResampleBwdPlan<DyType, DxType, ComputeType, IndexType> gpuPlan(std::move(params));

    Tensor<DyType> dyTensor(dyDims, dyStrides);
    Tensor<DxType> cpuDx(dxDims, dxStrides);
    Tensor<DxType> gpuDx(dxDims, dxStrides);
    Tensor<IndexType> indexTensor(dyDims, dyStrides);

    constexpr unsigned int SEED = 42;
    fillWithRandomValues(dyTensor, static_cast<DyType>(-1.0f), static_cast<DyType>(1.0f), SEED);

    // Compute sensible index tensor values using CPU reference
    if(hasIndex)
    {
        Tensor<float> xTensor(dxDims, dxStrides);
        Tensor<float> yTensor(dyDims, dyStrides);
        fillWithRandomValues(xTensor, -1.0f, 1.0f, SEED + 1);
        xTensor.memory().hostData();

        CpuFpReferenceResampleFwd::forward<float, float, float, IndexType>(
            xTensor, yTensor, prePadding, stride, window, resampleMode, paddingMode, &indexTensor);
        indexTensor.markHostModified();
    }

    std::unordered_map<int64_t, void*> gpuVariantPack;
    gpuVariantPack[DY_UID] = dyTensor.rawDeviceData();
    gpuVariantPack[DX_UID] = gpuDx.rawDeviceData();
    if(hasIndex)
    {
        gpuVariantPack[INDEX_UID] = indexTensor.rawDeviceData();
    }
    gpuPlan.execute(gpuVariantPack);
    gpuDx.markDeviceModified();

    std::unordered_map<int64_t, void*> cpuVariantPack;
    cpuVariantPack[DY_UID] = dyTensor.rawHostData();
    cpuVariantPack[DX_UID] = cpuDx.rawHostData();
    if(hasIndex)
    {
        cpuVariantPack[INDEX_UID] = indexTensor.rawHostData();
    }

    CpuReferenceGraphExecutor cpuExecutor;
    cpuExecutor.execute(builder.GetBufferPointer(), builder.GetSize(), cpuVariantPack);
    cpuDx.markHostModified();

    const float tolerance = resample::getTolerance<DxType>();
    const auto* cpuDxData = static_cast<const DxType*>(cpuDx.rawHostData());
    const auto* gpuDxData = static_cast<const DxType*>(gpuDx.rawHostData());
    for(size_t i = 0; i < cpuDx.elementCount(); ++i)
    {
        EXPECT_NEAR(static_cast<float>(gpuDxData[i]), static_cast<float>(cpuDxData[i]), tolerance)
            << "Mismatch in dx at index " << i;
    }
}

} // namespace

// =============================================================
// FP32 plan execution tests
// =============================================================

TEST(TestGpuResampleFwdPlanFp32, ExecutePlanNchw)
{
    SKIP_IF_NO_DEVICES();

    for(const auto& resampleMode : {ResampleMode::MAXPOOL,
                                    ResampleMode::AVGPOOL_EXCLUDE_PADDING,
                                    ResampleMode::AVGPOOL_INCLUDE_PADDING})
    {
        for(const auto& paddingMode : {PaddingMode::ZERO_PAD, PaddingMode::NEG_INF_PAD})
        {
            runResampleFwdPlanExecuteVsCpuRef<float, float, float, int32_t>({2, 3, 14, 14},
                                                                            TensorLayout::NCHW,
                                                                            {2, 1},
                                                                            {1, 2},
                                                                            {2, 2},
                                                                            {3, 3},
                                                                            resampleMode,
                                                                            paddingMode);
        }
    }
}

TEST(TestGpuResampleFwdPlanFp32, ExecutePlanNhwc)
{
    SKIP_IF_NO_DEVICES();

    for(const auto& resampleMode : {ResampleMode::MAXPOOL,
                                    ResampleMode::AVGPOOL_EXCLUDE_PADDING,
                                    ResampleMode::AVGPOOL_INCLUDE_PADDING})
    {
        for(const auto& paddingMode : {PaddingMode::ZERO_PAD, PaddingMode::NEG_INF_PAD})
        {
            runResampleFwdPlanExecuteVsCpuRef<float, float, float, int32_t>({2, 3, 14, 14},
                                                                            TensorLayout::NHWC,
                                                                            {2, 1},
                                                                            {1, 2},
                                                                            {2, 2},
                                                                            {3, 3},
                                                                            resampleMode,
                                                                            paddingMode);
        }
    }
}

TEST(TestGpuResampleBwdPlanFp32, ExecutePlanNchw)
{
    SKIP_IF_NO_DEVICES();

    for(const auto& resampleMode : {ResampleMode::MAXPOOL,
                                    ResampleMode::AVGPOOL_EXCLUDE_PADDING,
                                    ResampleMode::AVGPOOL_INCLUDE_PADDING})
    {
        for(const auto& paddingMode : {PaddingMode::ZERO_PAD, PaddingMode::NEG_INF_PAD})
        {
            runResampleBwdPlanExecuteVsCpuRef<float, float, float, int32_t>({2, 3, 14, 14},
                                                                            TensorLayout::NCHW,
                                                                            {2, 1},
                                                                            {1, 2},
                                                                            {2, 2},
                                                                            {3, 3},
                                                                            resampleMode,
                                                                            paddingMode);
        }
    }
}

TEST(TestGpuResampleBwdPlanFp32, ExecutePlanNhwc)
{
    SKIP_IF_NO_DEVICES();

    for(const auto& resampleMode : {ResampleMode::MAXPOOL,
                                    ResampleMode::AVGPOOL_EXCLUDE_PADDING,
                                    ResampleMode::AVGPOOL_INCLUDE_PADDING})
    {
        for(const auto& paddingMode : {PaddingMode::ZERO_PAD, PaddingMode::NEG_INF_PAD})
        {
            runResampleBwdPlanExecuteVsCpuRef<float, float, float, int32_t>({2, 3, 14, 14},
                                                                            TensorLayout::NHWC,
                                                                            {2, 1},
                                                                            {1, 2},
                                                                            {2, 2},
                                                                            {3, 3},
                                                                            resampleMode,
                                                                            paddingMode);
        }
    }
}

// =============================================================
// FP16 plan execution tests
// =============================================================

TEST(TestGpuResampleFwdPlanFp16, ExecutePlanNchw)
{
    SKIP_IF_NO_DEVICES();

    for(const auto& resampleMode : {ResampleMode::MAXPOOL,
                                    ResampleMode::AVGPOOL_EXCLUDE_PADDING,
                                    ResampleMode::AVGPOOL_INCLUDE_PADDING})
    {
        for(const auto& paddingMode : {PaddingMode::ZERO_PAD, PaddingMode::NEG_INF_PAD})
        {
            runResampleFwdPlanExecuteVsCpuRef<half, half, float, int32_t>({2, 3, 14, 14},
                                                                          TensorLayout::NCHW,
                                                                          {2, 1},
                                                                          {1, 2},
                                                                          {2, 2},
                                                                          {3, 3},
                                                                          resampleMode,
                                                                          paddingMode);
        }
    }
}

TEST(TestGpuResampleFwdPlanFp16, ExecutePlanNhwc)
{
    SKIP_IF_NO_DEVICES();

    for(const auto& resampleMode : {ResampleMode::MAXPOOL,
                                    ResampleMode::AVGPOOL_EXCLUDE_PADDING,
                                    ResampleMode::AVGPOOL_INCLUDE_PADDING})
    {
        for(const auto& paddingMode : {PaddingMode::ZERO_PAD, PaddingMode::NEG_INF_PAD})
        {
            runResampleFwdPlanExecuteVsCpuRef<half, half, float, int32_t>({2, 3, 14, 14},
                                                                          TensorLayout::NHWC,
                                                                          {2, 1},
                                                                          {1, 2},
                                                                          {2, 2},
                                                                          {3, 3},
                                                                          resampleMode,
                                                                          paddingMode);
        }
    }
}

TEST(TestGpuResampleBwdPlanFp16, ExecutePlanNchw)
{
    SKIP_IF_NO_DEVICES();

    for(const auto& resampleMode : {ResampleMode::MAXPOOL,
                                    ResampleMode::AVGPOOL_EXCLUDE_PADDING,
                                    ResampleMode::AVGPOOL_INCLUDE_PADDING})
    {
        for(const auto& paddingMode : {PaddingMode::ZERO_PAD, PaddingMode::NEG_INF_PAD})
        {
            runResampleBwdPlanExecuteVsCpuRef<half, half, float, int32_t>({2, 3, 14, 14},
                                                                          TensorLayout::NCHW,
                                                                          {2, 1},
                                                                          {1, 2},
                                                                          {2, 2},
                                                                          {3, 3},
                                                                          resampleMode,
                                                                          paddingMode);
        }
    }
}

TEST(TestGpuResampleBwdPlanFp16, ExecutePlanNhwc)
{
    SKIP_IF_NO_DEVICES();

    for(const auto& resampleMode : {ResampleMode::MAXPOOL,
                                    ResampleMode::AVGPOOL_EXCLUDE_PADDING,
                                    ResampleMode::AVGPOOL_INCLUDE_PADDING})
    {
        for(const auto& paddingMode : {PaddingMode::ZERO_PAD, PaddingMode::NEG_INF_PAD})
        {
            runResampleBwdPlanExecuteVsCpuRef<half, half, float, int32_t>({2, 3, 14, 14},
                                                                          TensorLayout::NHWC,
                                                                          {2, 1},
                                                                          {1, 2},
                                                                          {2, 2},
                                                                          {3, 3},
                                                                          resampleMode,
                                                                          paddingMode);
        }
    }
}

// =============================================================
// BFP16 plan execution tests
// =============================================================

TEST(TestGpuResampleFwdPlanBfp16, ExecutePlanNchw)
{
    SKIP_IF_NO_DEVICES();

    for(const auto& resampleMode : {ResampleMode::MAXPOOL,
                                    ResampleMode::AVGPOOL_EXCLUDE_PADDING,
                                    ResampleMode::AVGPOOL_INCLUDE_PADDING})
    {
        for(const auto& paddingMode : {PaddingMode::ZERO_PAD, PaddingMode::NEG_INF_PAD})
        {
            runResampleFwdPlanExecuteVsCpuRef<bfloat16, bfloat16, float, int32_t>(
                {2, 3, 14, 14},
                TensorLayout::NCHW,
                {2, 1},
                {1, 2},
                {2, 2},
                {3, 3},
                resampleMode,
                paddingMode);
        }
    }
}

TEST(TestGpuResampleFwdPlanBfp16, ExecutePlanNhwc)
{
    SKIP_IF_NO_DEVICES();

    for(const auto& resampleMode : {ResampleMode::MAXPOOL,
                                    ResampleMode::AVGPOOL_EXCLUDE_PADDING,
                                    ResampleMode::AVGPOOL_INCLUDE_PADDING})
    {
        for(const auto& paddingMode : {PaddingMode::ZERO_PAD, PaddingMode::NEG_INF_PAD})
        {
            runResampleFwdPlanExecuteVsCpuRef<bfloat16, bfloat16, float, int32_t>(
                {2, 3, 14, 14},
                TensorLayout::NHWC,
                {2, 1},
                {1, 2},
                {2, 2},
                {3, 3},
                resampleMode,
                paddingMode);
        }
    }
}

TEST(TestGpuResampleBwdPlanBfp16, ExecutePlanNchw)
{
    SKIP_IF_NO_DEVICES();

    for(const auto& resampleMode : {ResampleMode::MAXPOOL,
                                    ResampleMode::AVGPOOL_EXCLUDE_PADDING,
                                    ResampleMode::AVGPOOL_INCLUDE_PADDING})
    {
        for(const auto& paddingMode : {PaddingMode::ZERO_PAD, PaddingMode::NEG_INF_PAD})
        {
            runResampleBwdPlanExecuteVsCpuRef<bfloat16, bfloat16, float, int32_t>(
                {2, 3, 14, 14},
                TensorLayout::NCHW,
                {2, 1},
                {1, 2},
                {2, 2},
                {3, 3},
                resampleMode,
                paddingMode);
        }
    }
}

TEST(TestGpuResampleBwdPlanBfp16, ExecutePlanNhwc)
{
    SKIP_IF_NO_DEVICES();

    for(const auto& resampleMode : {ResampleMode::MAXPOOL,
                                    ResampleMode::AVGPOOL_EXCLUDE_PADDING,
                                    ResampleMode::AVGPOOL_INCLUDE_PADDING})
    {
        for(const auto& paddingMode : {PaddingMode::ZERO_PAD, PaddingMode::NEG_INF_PAD})
        {
            runResampleBwdPlanExecuteVsCpuRef<bfloat16, bfloat16, float, int32_t>(
                {2, 3, 14, 14},
                TensorLayout::NHWC,
                {2, 1},
                {1, 2},
                {2, 2},
                {3, 3},
                resampleMode,
                paddingMode);
        }
    }
}
