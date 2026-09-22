// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include <hipdnn-gpu-ref/GpuFpReferenceSdpa.hpp>
#include <hipdnn_data_sdk/types/Bfloat16.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "harness/gpu-graph-executor/detail/GpuRMSNormPlan.hpp"

using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_integration_tests::gpu_graph_executor::detail;

// =============================================================
// Test GpuRMSNormFwdPlan
// =============================================================

TEST(TestGpuRMSNormFwdPlanBuilder, PlanConstruction)
{
    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormGraph();
    auto graph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    const GpuRMSNormFwdPlanBuilder<DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT>
        patient;

    auto builtPlan = patient.buildNodePlan(graph, graph.getNode(0));

    const bool result
        = dynamic_cast<GpuRMSNormFwdPlan<float, float, float, float>*>(builtPlan.get()) != nullptr;
    EXPECT_TRUE(result);
}

TEST(TestGpuRMSNormFwdPlanBuilder, IsApplicable)
{
    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormGraph();
    auto graph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    const GpuRMSNormFwdPlanBuilder<DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT>
        floatPlanBuilder;
    EXPECT_TRUE(floatPlanBuilder.isApplicable(graph.getNode(0), graph.getTensorMap()));

    // Half builder must not be applicable to a float graph
    const GpuRMSNormFwdPlanBuilder<DataType::HALF, DataType::HALF, DataType::HALF, DataType::HALF>
        halfPlanBuilder;
    EXPECT_FALSE(halfPlanBuilder.isApplicable(graph.getNode(0), graph.getTensorMap()));

    // Missing input tensor must make the plan inapplicable
    auto tensorMapCopy = graph.getTensorMap();
    const auto* nodeAttributes = graph.getNode(0).attributes_as_RMSNormAttributes();
    EXPECT_NE(nodeAttributes, nullptr);
    tensorMapCopy.erase(nodeAttributes->x_tensor_uid());
    EXPECT_FALSE(floatPlanBuilder.isApplicable(graph.getNode(0), tensorMapCopy));
}

TEST(TestGpuRMSNormFwdPlanBuilder, BuildNodePlanThrowsForWrongAttributesType)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormInferenceGraph();
    auto graph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    const GpuRMSNormFwdPlanBuilder<DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT>
        patient;

    EXPECT_THROW(patient.buildNodePlan(graph, graph.getNode(0)), std::runtime_error);
}

TEST(TestGpuRMSNormFwdPlanBuilder, IsApplicableReturnsFalseForWrongAttributesType)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormInferenceGraph();
    auto graph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    const GpuRMSNormFwdPlanBuilder<DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT>
        patient;

    EXPECT_FALSE(patient.isApplicable(graph.getNode(0), graph.getTensorMap()));
}

TEST(TestGpuRMSNormFwdPlanBuilder, IsApplicableFalseWhenScaleTensorMissing)
{
    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormGraph();
    auto graph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    const GpuRMSNormFwdPlanBuilder<DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT>
        patient;

    auto tensorMapCopy = graph.getTensorMap();
    const auto* nodeAttributes = graph.getNode(0).attributes_as_RMSNormAttributes();
    ASSERT_NE(nodeAttributes, nullptr);
    tensorMapCopy.erase(nodeAttributes->scale_tensor_uid());

    EXPECT_FALSE(patient.isApplicable(graph.getNode(0), tensorMapCopy));
}

TEST(TestGpuRMSNormFwdPlanBuilder, IsApplicableFalseWhenTensorTypeMismatched)
{
    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormGraph(
        {150528, 50176, 224, 1},
        {1, 3, 224, 224},
        hipdnn_flatbuffers_sdk::data_objects::DataType::HALF);
    auto graph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    const GpuRMSNormFwdPlanBuilder<DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT>
        patient;

    EXPECT_FALSE(patient.isApplicable(graph.getNode(0), graph.getTensorMap()));
}

// =============================================================
// Test GpuRMSNormBwdPlan
// =============================================================

TEST(TestGpuRMSNormBwdPlanBuilder, PlanConstruction)
{
    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormBwdGraph();
    auto graph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    const GpuRMSNormBwdPlanBuilder<DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT>
        patient;

    auto builtPlan = patient.buildNodePlan(graph, graph.getNode(0));

    const bool result
        = dynamic_cast<GpuRMSNormBwdPlan<float, float, float, float, float>*>(builtPlan.get())
          != nullptr;
    EXPECT_TRUE(result);
}

TEST(TestGpuRMSNormBwdPlanBuilder, IsApplicable)
{
    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormBwdGraph();
    auto graph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    const GpuRMSNormBwdPlanBuilder<DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT>
        floatPlanBuilder;
    EXPECT_TRUE(floatPlanBuilder.isApplicable(graph.getNode(0), graph.getTensorMap()));

    // Half builder must not be applicable to a float graph
    const GpuRMSNormBwdPlanBuilder<DataType::HALF,
                                   DataType::HALF,
                                   DataType::HALF,
                                   DataType::HALF,
                                   DataType::HALF>
        halfPlanBuilder;
    EXPECT_FALSE(halfPlanBuilder.isApplicable(graph.getNode(0), graph.getTensorMap()));

    // Missing input tensor must make the plan inapplicable
    auto tensorMapCopy = graph.getTensorMap();
    const auto* nodeAttributes = graph.getNode(0).attributes_as_RMSNormBackwardAttributes();
    EXPECT_NE(nodeAttributes, nullptr);
    tensorMapCopy.erase(nodeAttributes->dy_tensor_uid());
    EXPECT_FALSE(floatPlanBuilder.isApplicable(graph.getNode(0), tensorMapCopy));
}

TEST(TestGpuRMSNormBwdPlanBuilder, BuildNodePlanThrowsForWrongAttributesType)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormInferenceGraph();
    auto graph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    const GpuRMSNormBwdPlanBuilder<DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT>
        patient;

    EXPECT_THROW(patient.buildNodePlan(graph, graph.getNode(0)), std::runtime_error);
}

TEST(TestGpuRMSNormBwdPlanBuilder, IsApplicableReturnsFalseForWrongAttributesType)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormInferenceGraph();
    auto graph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    const GpuRMSNormBwdPlanBuilder<DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT>
        patient;

    EXPECT_FALSE(patient.isApplicable(graph.getNode(0), graph.getTensorMap()));
}

TEST(TestGpuRMSNormBwdPlanBuilder, IsApplicableFalseWhenInputTensorMissing)
{
    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormBwdGraph();
    auto graph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    const GpuRMSNormBwdPlanBuilder<DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT>
        patient;

    auto tensorMapCopy = graph.getTensorMap();
    const auto* nodeAttributes = graph.getNode(0).attributes_as_RMSNormBackwardAttributes();
    ASSERT_NE(nodeAttributes, nullptr);
    tensorMapCopy.erase(nodeAttributes->x_tensor_uid());

    EXPECT_FALSE(patient.isApplicable(graph.getNode(0), tensorMapCopy));
}

TEST(TestGpuRMSNormBwdPlanBuilder, IsApplicableFalseWhenDxTensorTypeMismatched)
{
    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormBwdGraph(
        {150528, 50176, 224, 1},
        {1, 3, 224, 224},
        false,
        hipdnn_flatbuffers_sdk::data_objects::DataType::HALF);
    auto graph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    const GpuRMSNormBwdPlanBuilder<DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT>
        patient;

    EXPECT_FALSE(patient.isApplicable(graph.getNode(0), graph.getTensorMap()));
}

TEST(TestGpuRMSNormBwdPlanBuilder, IsApplicableFalseWhenInvRmsTensorMissing)
{
    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormBwdGraph();
    auto graph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    const GpuRMSNormBwdPlanBuilder<DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT>
        patient;

    auto tensorMapCopy = graph.getTensorMap();
    const auto* nodeAttributes = graph.getNode(0).attributes_as_RMSNormBackwardAttributes();
    ASSERT_NE(nodeAttributes, nullptr);
    tensorMapCopy.erase(nodeAttributes->inv_rms_tensor_uid());

    EXPECT_FALSE(patient.isApplicable(graph.getNode(0), tensorMapCopy));
}

TEST(TestGpuRMSNormBwdPlanBuilder, BuildNodePlanSucceedsWithOptionalDbias)
{
    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormBwdGraph(
        {150528, 50176, 224, 1}, {1, 3, 224, 224}, true);
    auto graph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    const GpuRMSNormBwdPlanBuilder<DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT>
        patient;

    auto builtPlan = patient.buildNodePlan(graph, graph.getNode(0));
    const bool result
        = dynamic_cast<GpuRMSNormBwdPlan<float, float, float, float, float>*>(builtPlan.get())
          != nullptr;
    EXPECT_TRUE(result);
}
