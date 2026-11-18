// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <algorithm>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "RaggedGraphTestUtils.hpp"
#include "harness/gpu-graph-executor/detail/GpuPointwisePlan.hpp"
#include <hipdnn-gpu-ref/GpuFpReferenceCommon.hpp>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_integration_tests::gpu_graph_executor::detail;
using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_integration_tests::test_utils;
using namespace hipdnn_gpu_ref::common::gpu_fp_reference_tensor;

namespace
{

constexpr int64_t IN_0_UID = 1;
constexpr int64_t OUT_UID = 2;
constexpr int64_t IN_1_UID = 3;
constexpr int64_t IN_2_UID = 4;

const std::vector<int64_t> DIMS = {2, 3, 4, 4};

flatbuffers::FlatBufferBuilder createUnaryGraph(PointwiseMode mode = PointwiseMode::ABS)
{
    return createPointwiseGraph(
        PointwiseGraphSpec::unary(mode, DIMS, generateStrides(DIMS), DIMS, generateStrides(DIMS)));
}

flatbuffers::FlatBufferBuilder createBinaryGraph(PointwiseMode mode = PointwiseMode::MAX_OP)
{
    const auto strides = generateStrides(DIMS);
    return createPointwiseGraph(
        PointwiseGraphSpec::binary(mode, DIMS, strides, DIMS, strides, DIMS, strides));
}

flatbuffers::FlatBufferBuilder createTernaryGraph(DataType maskDataType = DataType::BOOLEAN,
                                                  PointwiseMode mode = PointwiseMode::BINARY_SELECT)
{
    const auto strides = generateStrides(DIMS);
    auto spec = PointwiseGraphSpec::binary(mode, DIMS, strides, DIMS, strides, DIMS, strides);
    spec.thirdInputDims = DIMS;
    spec.thirdInputStrides = strides;
    spec.thirdInputDataType = maskDataType;
    return createPointwiseGraph(spec);
}

// The graph spec uses one data type for the input and output, so CMP_GT with its boolean output
// is built here.
flatbuffers::FlatBufferBuilder createCmpGtGraph(DataType outputDataType = DataType::BOOLEAN)
{
    const auto strides = generateStrides(DIMS);

    flatbuffers::FlatBufferBuilder builder;
    std::vector<flatbuffers::Offset<TensorAttributes>> tensors;
    for(const auto& [uid, name, type] : {std::tuple{IN_0_UID, "in_0", DataType::FLOAT},
                                         std::tuple{OUT_UID, "out_0", outputDataType},
                                         std::tuple{IN_1_UID, "in_1", DataType::FLOAT}})
    {
        tensors.push_back(CreateTensorAttributesDirect(builder,
                                                       uid,
                                                       name,
                                                       type,
                                                       &strides,
                                                       &DIMS,
                                                       /*virtual_=*/false,
                                                       TensorValue::NONE,
                                                       /*value=*/0,
                                                       false));
    }

    auto pointwiseAttrs = CreatePointwiseAttributes(builder,
                                                    PointwiseMode::CMP_GT,
                                                    flatbuffers::nullopt,
                                                    flatbuffers::nullopt,
                                                    flatbuffers::nullopt,
                                                    flatbuffers::nullopt,
                                                    IN_0_UID,
                                                    IN_1_UID,
                                                    flatbuffers::nullopt,
                                                    OUT_UID);
    std::vector<flatbuffers::Offset<Node>> nodes;
    nodes.push_back(CreateNodeDirect(builder,
                                     "pointwise_node",
                                     DataType::FLOAT,
                                     NodeAttributes::PointwiseAttributes,
                                     pointwiseAttrs.Union()));
    builder.Finish(CreateGraphDirect(
        builder, "TestGraph", DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, &tensors, &nodes));
    return builder;
}

} // namespace

// =============================================================
// Test GpuPointwiseUnaryPlan
// =============================================================

TEST(TestGpuPointwiseUnaryPlanBuilder, PlanConstruction)
{
    auto builder = createUnaryGraph();
    auto graph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    const GpuPointwiseUnaryPlanBuilder<DataType::FLOAT, DataType::FLOAT, DataType::FLOAT> patient;

    auto builtPlan = patient.buildNodePlan(graph, graph.getNode(0));

    const bool result
        = dynamic_cast<GpuPointwiseUnaryPlan<float, float, float>*>(builtPlan.get()) != nullptr;
    EXPECT_TRUE(result);
}

TEST(TestGpuPointwiseUnaryPlanBuilder, IsApplicable)
{
    auto builder = createUnaryGraph();
    auto graph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    const GpuPointwiseUnaryPlanBuilder<DataType::FLOAT, DataType::FLOAT, DataType::FLOAT>
        floatPlanBuilder;
    EXPECT_TRUE(floatPlanBuilder.isApplicable(graph.getNode(0), graph.getTensorMap()));

    // Half builder must not be applicable to a float graph
    const GpuPointwiseUnaryPlanBuilder<DataType::HALF, DataType::HALF, DataType::FLOAT>
        halfPlanBuilder;
    EXPECT_FALSE(halfPlanBuilder.isApplicable(graph.getNode(0), graph.getTensorMap()));

    // Missing input tensor must make the plan inapplicable
    auto tensorMapCopy = graph.getTensorMap();
    const auto* nodeAttributes = graph.getNode(0).attributes_as_PointwiseAttributes();
    ASSERT_NE(nodeAttributes, nullptr);
    tensorMapCopy.erase(nodeAttributes->in_0_tensor_uid());
    EXPECT_FALSE(floatPlanBuilder.isApplicable(graph.getNode(0), tensorMapCopy));
}

TEST(TestGpuPointwiseUnaryPlanBuilder, IsNotApplicableForBinaryMode)
{
    auto builder = createBinaryGraph(PointwiseMode::ADD);
    auto graph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    const GpuPointwiseUnaryPlanBuilder<DataType::FLOAT, DataType::FLOAT, DataType::FLOAT> patient;

    EXPECT_FALSE(patient.isApplicable(graph.getNode(0), graph.getTensorMap()));
}

TEST(TestGpuPointwiseUnaryPlanBuilder, IsNotApplicableForRaggedTensors)
{
    auto builder = createUnaryGraph();

    const GpuPointwiseUnaryPlanBuilder<DataType::FLOAT, DataType::FLOAT, DataType::FLOAT> patient;

    auto denseWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        builder.GetBufferPointer(), builder.GetSize());
    ASSERT_TRUE(patient.isApplicable(denseWrap.getNode(0), denseWrap.getTensorMap()));

    auto ragged = markFirstTensorRagged(builder.GetBufferPointer());
    auto raggedWrap
        = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(ragged.data(), ragged.size());
    EXPECT_FALSE(patient.isApplicable(raggedWrap.getNode(0), raggedWrap.getTensorMap()));
}

TEST(TestGpuPointwiseUnaryPlanBuilder, BuildNodePlanThrowsForWrongAttributesType)
{
    auto builder = createValidBatchnormInferenceGraph();
    auto graph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    const GpuPointwiseUnaryPlanBuilder<DataType::FLOAT, DataType::FLOAT, DataType::FLOAT> patient;

    EXPECT_THROW(patient.buildNodePlan(graph, graph.getNode(0)), std::runtime_error);
}

// =============================================================
// Test GpuPointwiseBinaryPlan
// =============================================================

TEST(TestGpuPointwiseBinaryPlanBuilder, PlanConstruction)
{
    auto builder = createBinaryGraph();
    auto graph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    const GpuPointwiseBinaryPlanBuilder<DataType::FLOAT,
                                        DataType::FLOAT,
                                        DataType::FLOAT,
                                        DataType::FLOAT>
        patient;

    auto builtPlan = patient.buildNodePlan(graph, graph.getNode(0));

    const bool result
        = dynamic_cast<GpuPointwiseBinaryPlan<float, float, float, float>*>(builtPlan.get())
          != nullptr;
    EXPECT_TRUE(result);
}

TEST(TestGpuPointwiseBinaryPlanBuilder, IsApplicable)
{
    auto builder = createBinaryGraph();
    auto graph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    const GpuPointwiseBinaryPlanBuilder<DataType::FLOAT,
                                        DataType::FLOAT,
                                        DataType::FLOAT,
                                        DataType::FLOAT>
        floatPlanBuilder;
    EXPECT_TRUE(floatPlanBuilder.isApplicable(graph.getNode(0), graph.getTensorMap()));

    // Half builder must not be applicable to a float graph
    const GpuPointwiseBinaryPlanBuilder<DataType::HALF,
                                        DataType::HALF,
                                        DataType::HALF,
                                        DataType::FLOAT>
        halfPlanBuilder;
    EXPECT_FALSE(halfPlanBuilder.isApplicable(graph.getNode(0), graph.getTensorMap()));

    // Missing second input tensor must make the plan inapplicable
    auto tensorMapCopy = graph.getTensorMap();
    const auto* nodeAttributes = graph.getNode(0).attributes_as_PointwiseAttributes();
    ASSERT_NE(nodeAttributes, nullptr);
    tensorMapCopy.erase(nodeAttributes->in_1_tensor_uid().value());
    EXPECT_FALSE(floatPlanBuilder.isApplicable(graph.getNode(0), tensorMapCopy));
}

TEST(TestGpuPointwiseBinaryPlanBuilder, IsApplicableForSupportedModes)
{
    const GpuPointwiseBinaryPlanBuilder<DataType::FLOAT,
                                        DataType::FLOAT,
                                        DataType::FLOAT,
                                        DataType::FLOAT>
        patient;

    for(const auto mode : {PointwiseMode::ADD,
                           PointwiseMode::SUB,
                           PointwiseMode::MUL,
                           PointwiseMode::MAX_OP,
                           PointwiseMode::MIN_OP,
                           PointwiseMode::SIGMOID_BWD,
                           PointwiseMode::TANH_BWD,
                           PointwiseMode::RELU_BWD})
    {
        auto builder = createBinaryGraph(mode);
        auto graph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
            builder.GetBufferPointer(), builder.GetSize());
        EXPECT_TRUE(patient.isApplicable(graph.getNode(0), graph.getTensorMap()))
            << EnumNamePointwiseMode(mode);
    }
}

TEST(TestGpuPointwiseBinaryPlanBuilder, IsNotApplicableForUnsupportedMode)
{
    // DIV is registered on the CPU reference but not on the GPU one
    auto builder = createBinaryGraph(PointwiseMode::DIV);
    auto graph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    const GpuPointwiseBinaryPlanBuilder<DataType::FLOAT,
                                        DataType::FLOAT,
                                        DataType::FLOAT,
                                        DataType::FLOAT>
        patient;

    EXPECT_FALSE(patient.isApplicable(graph.getNode(0), graph.getTensorMap()));
}

TEST(TestGpuPointwiseBinaryPlanBuilder, IsNotApplicableForUnaryMode)
{
    auto builder = createUnaryGraph();
    auto graph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    const GpuPointwiseBinaryPlanBuilder<DataType::FLOAT,
                                        DataType::FLOAT,
                                        DataType::FLOAT,
                                        DataType::FLOAT>
        patient;

    EXPECT_FALSE(patient.isApplicable(graph.getNode(0), graph.getTensorMap()));
}

TEST(TestGpuPointwiseBinaryPlanBuilder, CmpGtRequiresBooleanOutput)
{
    auto booleanBuilder = createCmpGtGraph(DataType::BOOLEAN);
    auto booleanGraph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        booleanBuilder.GetBufferPointer(), booleanBuilder.GetSize());
    auto floatBuilder = createCmpGtGraph(DataType::FLOAT);
    auto floatGraph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        floatBuilder.GetBufferPointer(), floatBuilder.GetSize());

    const GpuPointwiseBinaryPlanBuilder<DataType::FLOAT,
                                        DataType::FLOAT,
                                        DataType::BOOLEAN,
                                        DataType::FLOAT>
        booleanPlanBuilder;
    EXPECT_TRUE(
        booleanPlanBuilder.isApplicable(booleanGraph.getNode(0), booleanGraph.getTensorMap()));

    // A float output is not valid for CMP_GT, even with a float builder
    const GpuPointwiseBinaryPlanBuilder<DataType::FLOAT,
                                        DataType::FLOAT,
                                        DataType::FLOAT,
                                        DataType::FLOAT>
        floatPlanBuilder;
    EXPECT_FALSE(floatPlanBuilder.isApplicable(floatGraph.getNode(0), floatGraph.getTensorMap()));
}

TEST(TestGpuPointwiseBinaryPlanBuilder, BooleanOutputIsOnlyApplicableToCmpGt)
{
    // MAX_OP graph with a boolean output tensor
    const auto strides = generateStrides(DIMS);
    auto spec = PointwiseGraphSpec::binary(
        PointwiseMode::MAX_OP, DIMS, strides, DIMS, strides, DIMS, strides);
    spec.ioDataType = DataType::BOOLEAN;
    spec.secondInputDataType = DataType::FLOAT;
    auto builder = createPointwiseGraph(spec);
    auto graph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    const GpuPointwiseBinaryPlanBuilder<DataType::BOOLEAN,
                                        DataType::FLOAT,
                                        DataType::BOOLEAN,
                                        DataType::FLOAT>
        patient;

    EXPECT_FALSE(patient.isApplicable(graph.getNode(0), graph.getTensorMap()));
}

TEST(TestGpuPointwiseBinaryPlanBuilder, IsNotApplicableForRaggedTensors)
{
    auto builder = createBinaryGraph();

    const GpuPointwiseBinaryPlanBuilder<DataType::FLOAT,
                                        DataType::FLOAT,
                                        DataType::FLOAT,
                                        DataType::FLOAT>
        patient;

    auto denseWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        builder.GetBufferPointer(), builder.GetSize());
    ASSERT_TRUE(patient.isApplicable(denseWrap.getNode(0), denseWrap.getTensorMap()));

    auto ragged = markFirstTensorRagged(builder.GetBufferPointer());
    auto raggedWrap
        = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(ragged.data(), ragged.size());
    EXPECT_FALSE(patient.isApplicable(raggedWrap.getNode(0), raggedWrap.getTensorMap()));
}

TEST(TestGpuPointwiseBinaryPlanBuilder, BuildNodePlanThrowsForWrongAttributesType)
{
    auto builder = createValidBatchnormInferenceGraph();
    auto graph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    const GpuPointwiseBinaryPlanBuilder<DataType::FLOAT,
                                        DataType::FLOAT,
                                        DataType::FLOAT,
                                        DataType::FLOAT>
        patient;

    EXPECT_THROW(patient.buildNodePlan(graph, graph.getNode(0)), std::runtime_error);
}

// =============================================================
// Test GpuPointwiseTernaryPlan
// =============================================================

TEST(TestGpuPointwiseTernaryPlanBuilder, PlanConstruction)
{
    auto builder = createTernaryGraph();
    auto graph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    const GpuPointwiseTernaryPlanBuilder<DataType::FLOAT,
                                         DataType::FLOAT,
                                         DataType::BOOLEAN,
                                         DataType::FLOAT,
                                         DataType::FLOAT>
        patient;

    auto builtPlan = patient.buildNodePlan(graph, graph.getNode(0));

    const bool result
        = dynamic_cast<GpuPointwiseTernaryPlan<float, float, bool, float>*>(builtPlan.get())
          != nullptr;
    EXPECT_TRUE(result);
}

TEST(TestGpuPointwiseTernaryPlanBuilder, IsApplicable)
{
    auto builder = createTernaryGraph();
    auto graph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    const GpuPointwiseTernaryPlanBuilder<DataType::FLOAT,
                                         DataType::FLOAT,
                                         DataType::BOOLEAN,
                                         DataType::FLOAT,
                                         DataType::FLOAT>
        floatPlanBuilder;
    EXPECT_TRUE(floatPlanBuilder.isApplicable(graph.getNode(0), graph.getTensorMap()));

    // Half builder must not be applicable to a float graph
    const GpuPointwiseTernaryPlanBuilder<DataType::HALF,
                                         DataType::HALF,
                                         DataType::BOOLEAN,
                                         DataType::HALF,
                                         DataType::FLOAT>
        halfPlanBuilder;
    EXPECT_FALSE(halfPlanBuilder.isApplicable(graph.getNode(0), graph.getTensorMap()));

    // Missing mask tensor must make the plan inapplicable
    auto tensorMapCopy = graph.getTensorMap();
    const auto* nodeAttributes = graph.getNode(0).attributes_as_PointwiseAttributes();
    ASSERT_NE(nodeAttributes, nullptr);
    tensorMapCopy.erase(nodeAttributes->in_2_tensor_uid().value());
    EXPECT_FALSE(floatPlanBuilder.isApplicable(graph.getNode(0), tensorMapCopy));
}

TEST(TestGpuPointwiseTernaryPlanBuilder, IsNotApplicableWhenMaskIsNotBoolean)
{
    auto builder = createTernaryGraph(DataType::FLOAT);
    auto graph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    const GpuPointwiseTernaryPlanBuilder<DataType::FLOAT,
                                         DataType::FLOAT,
                                         DataType::BOOLEAN,
                                         DataType::FLOAT,
                                         DataType::FLOAT>
        patient;

    EXPECT_FALSE(patient.isApplicable(graph.getNode(0), graph.getTensorMap()));
}

TEST(TestGpuPointwiseTernaryPlanBuilder, IsNotApplicableWhenModeIsNotBinarySelect)
{
    auto builder = createTernaryGraph(DataType::BOOLEAN, PointwiseMode::ADD);
    auto graph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    const GpuPointwiseTernaryPlanBuilder<DataType::FLOAT,
                                         DataType::FLOAT,
                                         DataType::BOOLEAN,
                                         DataType::FLOAT,
                                         DataType::FLOAT>
        patient;

    EXPECT_FALSE(patient.isApplicable(graph.getNode(0), graph.getTensorMap()));
}

TEST(TestGpuPointwiseTernaryPlanBuilder, IsNotApplicableForBinaryGraph)
{
    auto builder = createBinaryGraph();
    auto graph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    const GpuPointwiseTernaryPlanBuilder<DataType::FLOAT,
                                         DataType::FLOAT,
                                         DataType::BOOLEAN,
                                         DataType::FLOAT,
                                         DataType::FLOAT>
        patient;

    EXPECT_FALSE(patient.isApplicable(graph.getNode(0), graph.getTensorMap()));
}

TEST(TestGpuPointwiseTernaryPlanBuilder, IsNotApplicableForRaggedTensors)
{
    auto builder = createTernaryGraph();

    const GpuPointwiseTernaryPlanBuilder<DataType::FLOAT,
                                         DataType::FLOAT,
                                         DataType::BOOLEAN,
                                         DataType::FLOAT,
                                         DataType::FLOAT>
        patient;

    auto denseWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        builder.GetBufferPointer(), builder.GetSize());
    ASSERT_TRUE(patient.isApplicable(denseWrap.getNode(0), denseWrap.getTensorMap()));

    auto ragged = markFirstTensorRagged(builder.GetBufferPointer());
    auto raggedWrap
        = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(ragged.data(), ragged.size());
    EXPECT_FALSE(patient.isApplicable(raggedWrap.getNode(0), raggedWrap.getTensorMap()));
}

TEST(TestGpuPointwiseTernaryPlanBuilder, BuildNodePlanThrowsForWrongAttributesType)
{
    auto builder = createValidBatchnormInferenceGraph();
    auto graph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    const GpuPointwiseTernaryPlanBuilder<DataType::FLOAT,
                                         DataType::FLOAT,
                                         DataType::BOOLEAN,
                                         DataType::FLOAT,
                                         DataType::FLOAT>
        patient;

    EXPECT_THROW(patient.buildNodePlan(graph, graph.getNode(0)), std::runtime_error);
}

// =============================================================
// Test executing the plans
// =============================================================

TEST(TestGpuPointwiseBinaryPlanFp32, ExecutePlanMaxMin)
{
    SKIP_IF_NO_DEVICES();

    const GpuPointwiseBinaryPlanBuilder<DataType::FLOAT,
                                        DataType::FLOAT,
                                        DataType::FLOAT,
                                        DataType::FLOAT>
        patient;

    for(const auto mode : {PointwiseMode::MAX_OP, PointwiseMode::MIN_OP})
    {
        auto builder = createBinaryGraph(mode);
        auto graph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
            builder.GetBufferPointer(), builder.GetSize());
        auto plan = patient.buildNodePlan(graph, graph.getNode(0));

        Tensor<float> input0Tensor(DIMS);
        Tensor<float> input1Tensor(DIMS);
        Tensor<float> outputTensor(DIMS);
        fillWithRandomValues(input0Tensor, -1.0f, 1.0f, 1U);
        fillWithRandomValues(input1Tensor, -1.0f, 1.0f, 2U);
        outputTensor.fillWithValue(0);

        // Single non-const access to trigger migration as, despite a comment claiming otherwise, MigratableMemory cannot migrate via a const access
        input0Tensor.memory().hostData();
        input1Tensor.memory().hostData();

        std::unordered_map<int64_t, void*> variantPack;
        variantPack[IN_0_UID] = input0Tensor.rawDeviceData();
        variantPack[IN_1_UID] = input1Tensor.rawDeviceData();
        variantPack[OUT_UID] = outputTensor.rawDeviceData();

        plan->execute(variantPack);
        outputTensor.markDeviceModified();

        for(size_t i = 0; i < outputTensor.elementCount(); ++i)
        {
            const float x0 = static_cast<float*>(input0Tensor.rawHostData())[i];
            const float x1 = static_cast<float*>(input1Tensor.rawHostData())[i];
            const float expected
                = (mode == PointwiseMode::MAX_OP) ? std::max(x0, x1) : std::min(x0, x1);
            EXPECT_EQ(expected, static_cast<float*>(outputTensor.rawHostData())[i])
                << "Mismatch at index " << i;
        }
    }
}

TEST(TestGpuPointwiseBinaryPlanFp32, ExecutePlanCmpGt)
{
    SKIP_IF_NO_DEVICES();

    auto builder = createCmpGtGraph();
    auto graph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    const GpuPointwiseBinaryPlanBuilder<DataType::FLOAT,
                                        DataType::FLOAT,
                                        DataType::BOOLEAN,
                                        DataType::FLOAT>
        patient;
    auto plan = patient.buildNodePlan(graph, graph.getNode(0));

    Tensor<float> input0Tensor(DIMS);
    Tensor<float> input1Tensor(DIMS);
    Tensor<bool> outputTensor(DIMS);
    fillWithRandomValues(input0Tensor, -1.0f, 1.0f, 1U);
    fillWithRandomValues(input1Tensor, -1.0f, 1.0f, 2U);
    outputTensor.fillWithValue(false);

    // Single non-const access to trigger migration as, despite a comment claiming otherwise, MigratableMemory cannot migrate via a const access
    input0Tensor.memory().hostData();
    input1Tensor.memory().hostData();

    std::unordered_map<int64_t, void*> variantPack;
    variantPack[IN_0_UID] = input0Tensor.rawDeviceData();
    variantPack[IN_1_UID] = input1Tensor.rawDeviceData();
    variantPack[OUT_UID] = outputTensor.rawDeviceData();

    plan->execute(variantPack);
    outputTensor.markDeviceModified();

    for(size_t i = 0; i < outputTensor.elementCount(); ++i)
    {
        const bool expected = static_cast<float*>(input0Tensor.rawHostData())[i]
                              > static_cast<float*>(input1Tensor.rawHostData())[i];
        EXPECT_EQ(expected, static_cast<bool*>(outputTensor.rawHostData())[i])
            << "Mismatch at index " << i;
    }
}

TEST(TestGpuPointwiseTernaryPlanFp32, ExecutePlanBinarySelect)
{
    SKIP_IF_NO_DEVICES();

    auto builder = createTernaryGraph();
    auto graph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    const GpuPointwiseTernaryPlanBuilder<DataType::FLOAT,
                                         DataType::FLOAT,
                                         DataType::BOOLEAN,
                                         DataType::FLOAT,
                                         DataType::FLOAT>
        patient;
    auto plan = patient.buildNodePlan(graph, graph.getNode(0));

    Tensor<float> input0Tensor(DIMS);
    Tensor<float> input1Tensor(DIMS);
    Tensor<bool> maskTensor(DIMS);
    Tensor<float> outputTensor(DIMS);
    fillWithRandomValues(input0Tensor, -1.0f, 1.0f, 1U);
    fillWithRandomValues(input1Tensor, -1.0f, 1.0f, 2U);
    outputTensor.fillWithValue(0);

    // Single non-const access to trigger migration as, despite a comment claiming otherwise, MigratableMemory cannot migrate via a const access
    input0Tensor.memory().hostData();
    input1Tensor.memory().hostData();
    auto* maskData = maskTensor.memory().hostData();
    for(size_t i = 0; i < maskTensor.elementCount(); ++i)
    {
        maskData[i] = (i % 3) == 0;
    }
    maskTensor.memory().markHostModified();

    std::unordered_map<int64_t, void*> variantPack;
    variantPack[IN_0_UID] = input0Tensor.rawDeviceData();
    variantPack[IN_1_UID] = input1Tensor.rawDeviceData();
    variantPack[IN_2_UID] = maskTensor.rawDeviceData();
    variantPack[OUT_UID] = outputTensor.rawDeviceData();

    plan->execute(variantPack);
    outputTensor.markDeviceModified();

    for(size_t i = 0; i < outputTensor.elementCount(); ++i)
    {
        const float expected = ((i % 3) == 0) ? static_cast<float*>(input0Tensor.rawHostData())[i]
                                              : static_cast<float*>(input1Tensor.rawHostData())[i];
        EXPECT_EQ(expected, static_cast<float*>(outputTensor.rawHostData())[i])
            << "Mismatch at index " << i;
    }
}
