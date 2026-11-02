// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cstdint>
#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>
#include <stdexcept>
#include <string>
#include <tuple>
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

namespace
{

// The flag is omitted from the buffer while false, so it cannot be flipped in place.
flatbuffers::DetachedBuffer markTensorRuntimePassByValue(const void* graphBuffer, int64_t uid)
{
    auto graphT = std::unique_ptr<GraphT>(GetGraph(graphBuffer)->UnPack());
    for(auto& tensor : graphT->tensors)
    {
        if(tensor->uid == uid)
        {
            tensor->is_runtime_pass_by_value = true;
        }
    }

    flatbuffers::FlatBufferBuilder builder;
    builder.Finish(CreateGraph(builder, graphT.get()));
    return builder.Release();
}

} // namespace

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

TEST(TestGpuResampleFwdPlanBuilder, IsApplicableAcrossModesPaddingAndIndexPresence)
{
    const GpuResampleFwdPlanBuilder<DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::UNSET>
        noIndexBuilder;
    const GpuResampleFwdPlanBuilder<DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::INT32>
        indexBuilder;

    for(const auto mode : {ResampleMode::MAXPOOL,
                           ResampleMode::AVGPOOL_EXCLUDE_PADDING,
                           ResampleMode::AVGPOOL_INCLUDE_PADDING})
    {
        for(const auto paddingMode : {PaddingMode::ZERO_PAD, PaddingMode::NEG_INF_PAD})
        {
            for(const bool hasIndex : {false, true})
            {
                SCOPED_TRACE(testing::Message() << "mode=" << static_cast<int>(mode)
                                                << ", padding=" << static_cast<int>(paddingMode)
                                                << ", hasIndex=" << hasIndex);
                auto builder = hipdnn_test_sdk::utilities::createValidResampleFwdGraph(
                    hasIndex, mode, paddingMode);
                auto graph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
                    builder.GetBufferPointer(), builder.GetSize());

                EXPECT_EQ(noIndexBuilder.isApplicable(graph.getNode(0), graph.getTensorMap()),
                          !hasIndex);
                EXPECT_EQ(indexBuilder.isApplicable(graph.getNode(0), graph.getTensorMap()),
                          hasIndex && mode == ResampleMode::MAXPOOL);
            }
        }
    }

    auto builder = hipdnn_test_sdk::utilities::createValidResampleFwdGraph(true);
    auto graph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        builder.GetBufferPointer(), builder.GetSize());
    const auto* nodeAttributes = graph.getNode(0).attributes_as_ResampleFwdAttributes();
    ASSERT_NE(nodeAttributes, nullptr);
    auto tensorMap = graph.getTensorMap();
    tensorMap[nodeAttributes->index_tensor_uid().value()]
        = tensorMap.at(nodeAttributes->x_tensor_uid());
    EXPECT_FALSE(indexBuilder.isApplicable(graph.getNode(0), tensorMap));

    auto computeTypeBuilder = hipdnn_test_sdk::utilities::createValidResampleFwdGraph();
    auto computeTypeGraph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        computeTypeBuilder.GetBufferPointer(), computeTypeBuilder.GetSize());
    auto* node = const_cast<Node*>(&computeTypeGraph.getNode(0));
    ASSERT_TRUE(node->mutate_compute_data_type(DataType::HALF));
    EXPECT_FALSE(
        noIndexBuilder.isApplicable(computeTypeGraph.getNode(0), computeTypeGraph.getTensorMap()));
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

    auto tensorMapWithoutDx = graph.getTensorMap();
    tensorMapWithoutDx.erase(nodeAttributes->dx_tensor_uid());
    EXPECT_FALSE(patient.isApplicable(graph.getNode(0), tensorMapWithoutDx));
}

TEST(TestGpuResampleFwdPlanBuilder, IsNotApplicableForRuntimePassByValueOperand)
{
    for(const bool hasIndex : {false, true})
    {
        const auto uidOf = [](const ResampleFwdAttributes& attributes, int operand) {
            switch(operand)
            {
            case 0:
                return attributes.x_tensor_uid();
            case 1:
                return attributes.y_tensor_uid();
            default:
                return attributes.index_tensor_uid().value();
            }
        };

        for(int operand = 0; operand < (hasIndex ? 3 : 2); ++operand)
        {
            SCOPED_TRACE(testing::Message() << "operand=" << operand << ", hasIndex=" << hasIndex);
            auto builder = hipdnn_test_sdk::utilities::createValidResampleFwdGraph(hasIndex);
            auto graph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
                builder.GetBufferPointer(), builder.GetSize());
            const auto* nodeAttributes = graph.getNode(0).attributes_as_ResampleFwdAttributes();
            ASSERT_NE(nodeAttributes, nullptr);

            const GpuResampleFwdPlanBuilder<DataType::FLOAT,
                                            DataType::FLOAT,
                                            DataType::FLOAT,
                                            DataType::UNSET>
                noIndexBuilder;
            const GpuResampleFwdPlanBuilder<DataType::FLOAT,
                                            DataType::FLOAT,
                                            DataType::FLOAT,
                                            DataType::INT32>
                indexBuilder;
            const auto& patient
                = hasIndex ? static_cast<const IGpuGraphNodePlanBuilder&>(indexBuilder)
                           : static_cast<const IGpuGraphNodePlanBuilder&>(noIndexBuilder);
            ASSERT_TRUE(patient.isApplicable(graph.getNode(0), graph.getTensorMap()));

            const auto marked = markTensorRuntimePassByValue(builder.GetBufferPointer(),
                                                             uidOf(*nodeAttributes, operand));
            const auto markedGraph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
                marked.data(), marked.size());
            EXPECT_FALSE(patient.isApplicable(markedGraph.getNode(0), markedGraph.getTensorMap()));
        }
    }
}

TEST(TestGpuResampleBwdPlanBuilder, IsNotApplicableForRuntimePassByValueOperand)
{
    const GpuResampleBwdPlanBuilder<DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::INT32>
        patient;

    for(int operand = 0; operand < 3; ++operand)
    {
        SCOPED_TRACE(testing::Message() << "operand=" << operand);
        auto builder = hipdnn_test_sdk::utilities::createValidResampleBwdGraph();
        auto graph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
            builder.GetBufferPointer(), builder.GetSize());
        const auto* nodeAttributes = graph.getNode(0).attributes_as_ResampleBwdAttributes();
        ASSERT_NE(nodeAttributes, nullptr);
        ASSERT_TRUE(patient.isApplicable(graph.getNode(0), graph.getTensorMap()));

        int64_t uid = nodeAttributes->dy_tensor_uid();
        if(operand == 1)
        {
            uid = nodeAttributes->dx_tensor_uid();
        }
        else if(operand == 2)
        {
            uid = nodeAttributes->index_tensor_uid().value();
        }
        const auto marked = markTensorRuntimePassByValue(builder.GetBufferPointer(), uid);
        const auto markedGraph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
            marked.data(), marked.size());
        EXPECT_FALSE(patient.isApplicable(markedGraph.getNode(0), markedGraph.getTensorMap()));
    }
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

TEST(TestGpuResampleBwdPlanBuilder, IsApplicableAcrossModesPaddingAndIndexPresence)
{
    const GpuResampleBwdPlanBuilder<DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::UNSET>
        noIndexBuilder;
    const GpuResampleBwdPlanBuilder<DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::INT32>
        indexBuilder;

    for(const auto mode : {ResampleMode::MAXPOOL,
                           ResampleMode::AVGPOOL_EXCLUDE_PADDING,
                           ResampleMode::AVGPOOL_INCLUDE_PADDING})
    {
        for(const auto paddingMode : {PaddingMode::ZERO_PAD, PaddingMode::NEG_INF_PAD})
        {
            for(const bool hasIndex : {false, true})
            {
                SCOPED_TRACE(testing::Message() << "mode=" << static_cast<int>(mode)
                                                << ", padding=" << static_cast<int>(paddingMode)
                                                << ", hasIndex=" << hasIndex);
                auto builder
                    = hipdnn_test_sdk::utilities::createValidResampleBwdGraph(hasIndex, mode);
                auto graph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
                    builder.GetBufferPointer(), builder.GetSize());
                auto* nodeAttributes = const_cast<ResampleBwdAttributes*>(
                    graph.getNode(0).attributes_as_ResampleBwdAttributes());
                ASSERT_NE(nodeAttributes, nullptr);
                ASSERT_TRUE(nodeAttributes->mutate_padding_mode(paddingMode));

                EXPECT_EQ(noIndexBuilder.isApplicable(graph.getNode(0), graph.getTensorMap()),
                          !hasIndex && mode != ResampleMode::MAXPOOL);
                EXPECT_EQ(indexBuilder.isApplicable(graph.getNode(0), graph.getTensorMap()),
                          hasIndex && mode == ResampleMode::MAXPOOL);
            }
        }
    }

    auto builder = hipdnn_test_sdk::utilities::createValidResampleBwdGraph();
    auto graph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        builder.GetBufferPointer(), builder.GetSize());
    const auto* nodeAttributes = graph.getNode(0).attributes_as_ResampleBwdAttributes();
    ASSERT_NE(nodeAttributes, nullptr);
    auto tensorMap = graph.getTensorMap();
    tensorMap[nodeAttributes->index_tensor_uid().value()]
        = tensorMap.at(nodeAttributes->dy_tensor_uid());
    EXPECT_FALSE(indexBuilder.isApplicable(graph.getNode(0), tensorMap));

    auto computeTypeBuilder = hipdnn_test_sdk::utilities::createValidResampleBwdGraph(
        false, ResampleMode::AVGPOOL_EXCLUDE_PADDING);
    auto computeTypeGraph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        computeTypeBuilder.GetBufferPointer(), computeTypeBuilder.GetSize());
    auto* node = const_cast<Node*>(&computeTypeGraph.getNode(0));
    ASSERT_TRUE(node->mutate_compute_data_type(DataType::HALF));
    const GpuResampleBwdPlanBuilder<DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::UNSET>
        noIndexFloatBuilder;
    EXPECT_FALSE(noIndexFloatBuilder.isApplicable(computeTypeGraph.getNode(0),
                                                  computeTypeGraph.getTensorMap()));
}

namespace
{

constexpr unsigned int RESAMPLE_TEST_SEED = 42;
constexpr int64_t RESAMPLE_FIRST_UID = 1;
constexpr int64_t RESAMPLE_SECOND_UID = 2;
constexpr int64_t RESAMPLE_INDEX_UID = 3;

// Geometry shared by both directions. The big tensor is x (forward) or dx (backward);
// the small one is y (forward) or dy (backward), and the index tensor shares its shape.
struct ResampleProblem
{
    std::vector<int64_t> bigDims;
    std::vector<int64_t> bigStrides;
    std::vector<int64_t> smallDims;
    std::vector<int64_t> smallStrides;
    std::vector<int64_t> prePadding;
    std::vector<int64_t> postPadding;
    std::vector<int64_t> stride;
    std::vector<int64_t> window;
    ResampleMode resampleMode = ResampleMode::NOT_SET;
    PaddingMode paddingMode = PaddingMode::PADDING_NOT_SET;
    bool hasIndex = false;
};

ResampleProblem makeResampleProblem(const std::vector<int64_t>& bigDims,
                                    const TensorLayout& layout,
                                    const std::vector<int64_t>& prePadding,
                                    const std::vector<int64_t>& postPadding,
                                    const std::vector<int64_t>& stride,
                                    const std::vector<int64_t>& window,
                                    ResampleMode resampleMode,
                                    PaddingMode paddingMode,
                                    bool withIndex)
{
    ResampleProblem problem;
    problem.bigDims = bigDims;
    problem.smallDims = {bigDims[0], bigDims[1]};
    for(size_t i = 0; i < prePadding.size(); ++i)
    {
        const int64_t paddedDim = bigDims[i + 2] + prePadding[i] + postPadding[i];
        problem.smallDims.push_back((paddedDim - window[i]) / stride[i] + 1);
    }
    problem.bigStrides = generateStrides(problem.bigDims, layout.strideOrder);
    problem.smallStrides = generateStrides(problem.smallDims, layout.strideOrder);
    problem.prePadding = prePadding;
    problem.postPadding = postPadding;
    problem.stride = stride;
    problem.window = window;
    problem.resampleMode = resampleMode;
    problem.paddingMode = paddingMode;
    problem.hasIndex = withIndex && resampleMode == ResampleMode::MAXPOOL;
    return problem;
}

// Forward is x (first, big) -> y (second, small); backward is dy (first, small) -> dx
// (second, big). The tensor uids are RESAMPLE_{FIRST,SECOND,INDEX}_UID.
flatbuffers::FlatBufferBuilder buildResampleGraph(const ResampleProblem& problem,
                                                  bool isForward,
                                                  DataType bigType,
                                                  DataType smallType,
                                                  DataType computeType,
                                                  DataType indexType)
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<TensorAttributes>> tensorAttributes;

    if(isForward)
    {
        tensorAttributes.push_back(CreateTensorAttributesDirect(
            builder, RESAMPLE_FIRST_UID, "x", bigType, &problem.bigStrides, &problem.bigDims));
        tensorAttributes.push_back(CreateTensorAttributesDirect(builder,
                                                                RESAMPLE_SECOND_UID,
                                                                "y",
                                                                smallType,
                                                                &problem.smallStrides,
                                                                &problem.smallDims));
    }
    else
    {
        tensorAttributes.push_back(CreateTensorAttributesDirect(builder,
                                                                RESAMPLE_FIRST_UID,
                                                                "dy",
                                                                smallType,
                                                                &problem.smallStrides,
                                                                &problem.smallDims));
        tensorAttributes.push_back(CreateTensorAttributesDirect(
            builder, RESAMPLE_SECOND_UID, "dx", bigType, &problem.bigStrides, &problem.bigDims));
    }

    ::flatbuffers::Optional<int64_t> indexUid = ::flatbuffers::nullopt;
    if(problem.hasIndex)
    {
        indexUid = RESAMPLE_INDEX_UID;
        tensorAttributes.push_back(CreateTensorAttributesDirect(builder,
                                                                RESAMPLE_INDEX_UID,
                                                                "index",
                                                                indexType,
                                                                &problem.smallStrides,
                                                                &problem.smallDims));
    }

    ::flatbuffers::Offset<void> attributes;
    if(isForward)
    {
        attributes = CreateResampleFwdAttributesDirect(builder,
                                                       RESAMPLE_FIRST_UID,
                                                       RESAMPLE_SECOND_UID,
                                                       indexUid,
                                                       &problem.prePadding,
                                                       &problem.postPadding,
                                                       &problem.stride,
                                                       &problem.window,
                                                       problem.resampleMode,
                                                       problem.paddingMode,
                                                       problem.hasIndex
                                                           ? ::flatbuffers::Optional<bool>(true)
                                                           : ::flatbuffers::nullopt)
                         .Union();
    }
    else
    {
        attributes = CreateResampleBwdAttributesDirect(builder,
                                                       RESAMPLE_FIRST_UID,
                                                       RESAMPLE_SECOND_UID,
                                                       indexUid,
                                                       &problem.prePadding,
                                                       &problem.postPadding,
                                                       &problem.stride,
                                                       &problem.window,
                                                       problem.resampleMode,
                                                       problem.paddingMode)
                         .Union();
    }

    std::vector<::flatbuffers::Offset<Node>> nodes;
    nodes.push_back(CreateNodeDirect(builder,
                                     isForward ? "resample_fwd" : "resample_bwd",
                                     computeType,
                                     isForward ? NodeAttributes::ResampleFwdAttributes
                                               : NodeAttributes::ResampleBwdAttributes,
                                     attributes));

    builder.Finish(CreateGraphDirect(
        builder, "test", computeType, DataType::FLOAT, DataType::FLOAT, &tensorAttributes, &nodes));
    return builder;
}

template <typename T>
void expectTensorsNear(Tensor<T>& gpuTensor, Tensor<T>& cpuTensor, const char* name)
{
    const float tolerance = resample::getTolerance<T>();
    const auto* gpuData = static_cast<const T*>(gpuTensor.rawHostData());
    const auto* cpuData = static_cast<const T*>(cpuTensor.rawHostData());
    for(size_t i = 0; i < cpuTensor.elementCount(); ++i)
    {
        EXPECT_NEAR(static_cast<float>(gpuData[i]), static_cast<float>(cpuData[i]), tolerance)
            << "Mismatch in " << name << " at index " << i;
    }
}

template <typename T>
void expectTensorsEqual(Tensor<T>& gpuTensor, Tensor<T>& cpuTensor, const char* name)
{
    const auto* gpuData = static_cast<const T*>(gpuTensor.rawHostData());
    const auto* cpuData = static_cast<const T*>(cpuTensor.rawHostData());
    for(size_t i = 0; i < cpuTensor.elementCount(); ++i)
    {
        EXPECT_EQ(gpuData[i], cpuData[i]) << "Mismatch in " << name << " at index " << i;
    }
}

template <typename XType, typename YType, typename ComputeType, typename IndexType>
void runResampleFwdPlanExecuteVsCpuRef(const std::vector<int64_t>& xDims,
                                       const TensorLayout& layout,
                                       const std::vector<int64_t>& prePadding,
                                       const std::vector<int64_t>& postPadding,
                                       const std::vector<int64_t>& stride,
                                       const std::vector<int64_t>& window,
                                       ResampleMode resampleMode,
                                       PaddingMode paddingMode,
                                       bool withIndex = true)
{
    constexpr int64_t X_UID = RESAMPLE_FIRST_UID;
    constexpr int64_t Y_UID = RESAMPLE_SECOND_UID;
    constexpr int64_t INDEX_UID = RESAMPLE_INDEX_UID;

    const auto problem = makeResampleProblem(xDims,
                                             layout,
                                             prePadding,
                                             postPadding,
                                             stride,
                                             window,
                                             resampleMode,
                                             paddingMode,
                                             withIndex);
    const bool hasIndex = problem.hasIndex;
    const auto& yDims = problem.smallDims;
    const auto& xStrides = problem.bigStrides;
    const auto& yStrides = problem.smallStrides;

    const auto builder = buildResampleGraph(problem,
                                            /*isForward=*/true,
                                            nativeTypeToDataType<XType>(),
                                            nativeTypeToDataType<YType>(),
                                            nativeTypeToDataType<ComputeType>(),
                                            nativeTypeToDataType<IndexType>());

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

    fillWithRandomValues(
        xTensor, static_cast<XType>(-1.0f), static_cast<XType>(1.0f), RESAMPLE_TEST_SEED);

    std::unordered_map<int64_t, void*> gpuVariantPack;
    gpuVariantPack[X_UID] = xTensor.rawDeviceData();
    gpuVariantPack[Y_UID] = gpuY.rawDeviceData();
    if(hasIndex)
    {
        gpuVariantPack[INDEX_UID] = gpuIndex.rawDeviceData();
    }
    gpuPlan.execute(gpuVariantPack);
    gpuY.markDeviceModified();
    if(hasIndex)
    {
        gpuIndex.markDeviceModified();
    }

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
    if(hasIndex)
    {
        cpuIndex.markHostModified();
    }

    expectTensorsNear(gpuY, cpuY, "y");
    if(hasIndex)
    {
        expectTensorsEqual(gpuIndex, cpuIndex, "index");
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
    constexpr int64_t DY_UID = RESAMPLE_FIRST_UID;
    constexpr int64_t DX_UID = RESAMPLE_SECOND_UID;
    constexpr int64_t INDEX_UID = RESAMPLE_INDEX_UID;

    const auto problem = makeResampleProblem(dxDims,
                                             layout,
                                             prePadding,
                                             postPadding,
                                             stride,
                                             window,
                                             resampleMode,
                                             paddingMode,
                                             /*withIndex=*/true);
    const bool hasIndex = problem.hasIndex;
    const auto& dyDims = problem.smallDims;
    const auto& dxStrides = problem.bigStrides;
    const auto& dyStrides = problem.smallStrides;

    const auto builder = buildResampleGraph(problem,
                                            /*isForward=*/false,
                                            nativeTypeToDataType<DxType>(),
                                            nativeTypeToDataType<DyType>(),
                                            nativeTypeToDataType<ComputeType>(),
                                            nativeTypeToDataType<IndexType>());

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

    fillWithRandomValues(
        dyTensor, static_cast<DyType>(-1.0f), static_cast<DyType>(1.0f), RESAMPLE_TEST_SEED);

    // Compute sensible index tensor values using CPU reference
    if(hasIndex)
    {
        Tensor<float> xTensor(dxDims, dxStrides);
        Tensor<float> yTensor(dyDims, dyStrides);
        fillWithRandomValues(xTensor, -1.0f, 1.0f, RESAMPLE_TEST_SEED + 1);
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

    expectTensorsNear(gpuDx, cpuDx, "dx");
}

template <typename T>
void runResamplePlanExecuteForType(const TensorLayout& layout, bool isForward)
{
    for(const auto resampleMode : {ResampleMode::MAXPOOL,
                                   ResampleMode::AVGPOOL_EXCLUDE_PADDING,
                                   ResampleMode::AVGPOOL_INCLUDE_PADDING})
    {
        for(const auto paddingMode : {PaddingMode::ZERO_PAD, PaddingMode::NEG_INF_PAD})
        {
            if(isForward)
            {
                runResampleFwdPlanExecuteVsCpuRef<T, T, float, int32_t>({2, 3, 14, 14},
                                                                        layout,
                                                                        {2, 1},
                                                                        {1, 2},
                                                                        {2, 2},
                                                                        {3, 3},
                                                                        resampleMode,
                                                                        paddingMode);
            }
            else
            {
                runResampleBwdPlanExecuteVsCpuRef<T, T, float, int32_t>({2, 3, 14, 14},
                                                                        layout,
                                                                        {2, 1},
                                                                        {1, 2},
                                                                        {2, 2},
                                                                        {3, 3},
                                                                        resampleMode,
                                                                        paddingMode);
            }
        }
    }
}

} // namespace

class TestGpuResamplePlanExecution : public testing::TestWithParam<std::tuple<DataType, bool, bool>>
{
};

TEST_P(TestGpuResamplePlanExecution, MatchesCpuReference)
{
    SKIP_IF_NO_DEVICES();

    const auto [dataType, isNhwc, isForward] = GetParam();
    const auto& layout = isNhwc ? TensorLayout::NHWC : TensorLayout::NCHW;

    switch(dataType)
    {
    case DataType::FLOAT:
        runResamplePlanExecuteForType<float>(layout, isForward);
        break;
    case DataType::HALF:
        runResamplePlanExecuteForType<half>(layout, isForward);
        break;
    case DataType::BFLOAT16:
        runResamplePlanExecuteForType<bfloat16>(layout, isForward);
        break;
    default:
        FAIL() << "Unsupported resample test data type: " << static_cast<int>(dataType);
    }
}

INSTANTIATE_TEST_SUITE_P(
    DataTypeLayoutAndDirection,
    TestGpuResamplePlanExecution,
    testing::Combine(testing::Values(DataType::FLOAT, DataType::HALF, DataType::BFLOAT16),
                     testing::Bool(),
                     testing::Bool()));

namespace
{

struct ResampleGeometryCase
{
    std::string name;
    std::vector<int64_t> dims;
    TensorLayout layout;
    std::vector<int64_t> prePadding;
    std::vector<int64_t> postPadding;
    std::vector<int64_t> stride;
    std::vector<int64_t> window;
};

const std::vector<ResampleGeometryCase>& resampleGeometryCases()
{
    static const std::vector<ResampleGeometryCase> s_cases{
        {"Ncdhw3d",
         {1, 2, 6, 7, 8},
         TensorLayout::NCDHW,
         {1, 0, 1},
         {0, 1, 0},
         {2, 2, 2},
         {3, 2, 3}},
        {"Ndhwc3d",
         {1, 2, 6, 7, 8},
         TensorLayout::NDHWC,
         {1, 0, 1},
         {0, 1, 0},
         {2, 2, 2},
         {3, 2, 3}},
        {"StrideEqualsWindow", {1, 2, 8, 8}, TensorLayout::NCHW, {0, 0}, {0, 0}, {2, 2}, {2, 2}},
        {"WindowOfOne", {1, 2, 5, 5}, TensorLayout::NHWC, {0, 0}, {0, 0}, {1, 1}, {1, 1}},
        {"SizeOneSpatialDim", {1, 2, 1, 5}, TensorLayout::NCHW, {0, 1}, {0, 1}, {1, 1}, {1, 3}},
    };
    return s_cases;
}

} // namespace

class TestGpuResamplePlanGeometry : public testing::TestWithParam<ResampleGeometryCase>
{
};

TEST_P(TestGpuResamplePlanGeometry, FwdMatchesCpuReference)
{
    SKIP_IF_NO_DEVICES();

    const auto& geometry = GetParam();
    for(const auto resampleMode : {ResampleMode::MAXPOOL,
                                   ResampleMode::AVGPOOL_EXCLUDE_PADDING,
                                   ResampleMode::AVGPOOL_INCLUDE_PADDING})
    {
        for(const auto paddingMode : {PaddingMode::ZERO_PAD, PaddingMode::NEG_INF_PAD})
        {
            runResampleFwdPlanExecuteVsCpuRef<float, float, float, int32_t>(geometry.dims,
                                                                            geometry.layout,
                                                                            geometry.prePadding,
                                                                            geometry.postPadding,
                                                                            geometry.stride,
                                                                            geometry.window,
                                                                            resampleMode,
                                                                            paddingMode);
        }
    }
}

TEST_P(TestGpuResamplePlanGeometry, BwdMatchesCpuReference)
{
    SKIP_IF_NO_DEVICES();

    const auto& geometry = GetParam();
    for(const auto resampleMode : {ResampleMode::MAXPOOL,
                                   ResampleMode::AVGPOOL_EXCLUDE_PADDING,
                                   ResampleMode::AVGPOOL_INCLUDE_PADDING})
    {
        for(const auto paddingMode : {PaddingMode::ZERO_PAD, PaddingMode::NEG_INF_PAD})
        {
            runResampleBwdPlanExecuteVsCpuRef<float, float, float, int32_t>(geometry.dims,
                                                                            geometry.layout,
                                                                            geometry.prePadding,
                                                                            geometry.postPadding,
                                                                            geometry.stride,
                                                                            geometry.window,
                                                                            resampleMode,
                                                                            paddingMode);
        }
    }
}

INSTANTIATE_TEST_SUITE_P(Geometries,
                         TestGpuResamplePlanGeometry,
                         testing::ValuesIn(resampleGeometryCases()),
                         [](const testing::TestParamInfo<ResampleGeometryCase>& info) {
                             return info.param.name;
                         });

TEST(TestGpuResampleFwdPlanNoIndex, MaxpoolWithoutIndexMatchesCpuReference)
{
    SKIP_IF_NO_DEVICES();

    for(const auto paddingMode : {PaddingMode::ZERO_PAD, PaddingMode::NEG_INF_PAD})
    {
        runResampleFwdPlanExecuteVsCpuRef<float, float, float, int32_t>({2, 3, 14, 14},
                                                                        TensorLayout::NCHW,
                                                                        {2, 1},
                                                                        {1, 2},
                                                                        {2, 2},
                                                                        {3, 3},
                                                                        ResampleMode::MAXPOOL,
                                                                        paddingMode,
                                                                        /*withIndex=*/false);
    }
}

namespace
{

// Asymmetric padding and stride != window, so swapped arguments change the result.
ResampleProblem makeAsymmetricResampleProblem(ResampleMode resampleMode)
{
    return makeResampleProblem({1, 2, 9, 9},
                               TensorLayout::NCHW,
                               {2, 1},
                               {1, 2},
                               {2, 3},
                               {3, 2},
                               resampleMode,
                               PaddingMode::NEG_INF_PAD,
                               /*withIndex=*/true);
}

void expectParamsMatchProblem(const GpuResampleParams& params,
                              const ResampleProblem& problem,
                              bool isForward)
{
    // The input tensor is x for forward and dy for backward.
    EXPECT_EQ(params.inputTensor.uid, RESAMPLE_FIRST_UID);
    EXPECT_EQ(params.outputTensor.uid, RESAMPLE_SECOND_UID);
    EXPECT_EQ(params.inputTensor.dims, isForward ? problem.bigDims : problem.smallDims);
    EXPECT_EQ(params.outputTensor.dims, isForward ? problem.smallDims : problem.bigDims);
    EXPECT_EQ(params.prePadding, problem.prePadding);
    EXPECT_EQ(params.postPadding, problem.postPadding);
    EXPECT_EQ(params.stride, problem.stride);
    EXPECT_EQ(params.window, problem.window);
    EXPECT_EQ(params.resampleMode, problem.resampleMode);
    EXPECT_EQ(params.paddingMode, problem.paddingMode);

    ASSERT_EQ(params.indexTensor.has_value(), problem.hasIndex);
    if(problem.hasIndex)
    {
        EXPECT_EQ(params.indexTensor->uid, RESAMPLE_INDEX_UID);
        EXPECT_EQ(params.indexTensor->dims, problem.smallDims);
    }
}

} // namespace

TEST(TestGpuResampleFwdPlanBuilder, BuildNodePlanForwardsAttributesWithIndex)
{
    const auto problem = makeAsymmetricResampleProblem(ResampleMode::MAXPOOL);
    ASSERT_TRUE(problem.hasIndex);
    const auto builder = buildResampleGraph(
        problem, true, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::INT32);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const GpuResampleFwdPlanBuilder<DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::INT32>
        patient;
    const auto plan = patient.buildNodePlan(graph, graph.getNode(0));
    const auto* typedPlan
        = dynamic_cast<const GpuResampleFwdPlan<float, float, float, int32_t>*>(plan.get());
    ASSERT_NE(typedPlan, nullptr);
    expectParamsMatchProblem(typedPlan->params(), problem, /*isForward=*/true);
}

TEST(TestGpuResampleFwdPlanBuilder, BuildNodePlanForwardsAttributesWithoutIndex)
{
    const auto problem = makeAsymmetricResampleProblem(ResampleMode::AVGPOOL_EXCLUDE_PADDING);
    ASSERT_FALSE(problem.hasIndex);
    const auto builder = buildResampleGraph(
        problem, true, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::UNSET);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const GpuResampleFwdPlanBuilder<DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::UNSET>
        patient;
    const auto plan = patient.buildNodePlan(graph, graph.getNode(0));
    const auto* typedPlan
        = dynamic_cast<const GpuResampleFwdPlan<float, float, float, int32_t>*>(plan.get());
    ASSERT_NE(typedPlan, nullptr);
    expectParamsMatchProblem(typedPlan->params(), problem, /*isForward=*/true);
}

TEST(TestGpuResampleBwdPlanBuilder, BuildNodePlanForwardsAttributesWithIndex)
{
    const auto problem = makeAsymmetricResampleProblem(ResampleMode::MAXPOOL);
    ASSERT_TRUE(problem.hasIndex);
    const auto builder = buildResampleGraph(
        problem, false, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::INT32);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const GpuResampleBwdPlanBuilder<DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::INT32>
        patient;
    const auto plan = patient.buildNodePlan(graph, graph.getNode(0));
    const auto* typedPlan
        = dynamic_cast<const GpuResampleBwdPlan<float, float, float, int32_t>*>(plan.get());
    ASSERT_NE(typedPlan, nullptr);
    expectParamsMatchProblem(typedPlan->params(), problem, /*isForward=*/false);
}

TEST(TestGpuResampleBwdPlanBuilder, BuildNodePlanForwardsAttributesWithoutIndex)
{
    auto problem = makeAsymmetricResampleProblem(ResampleMode::AVGPOOL_EXCLUDE_PADDING);
    ASSERT_FALSE(problem.hasIndex);
    const auto builder = buildResampleGraph(
        problem, false, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::UNSET);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const GpuResampleBwdPlanBuilder<DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::FLOAT,
                                    DataType::UNSET>
        patient;
    const auto plan = patient.buildNodePlan(graph, graph.getNode(0));
    const auto* typedPlan
        = dynamic_cast<const GpuResampleBwdPlan<float, float, float, int32_t>*>(plan.get());
    ASSERT_NE(typedPlan, nullptr);
    expectParamsMatchProblem(typedPlan->params(), problem, /*isForward=*/false);
}
