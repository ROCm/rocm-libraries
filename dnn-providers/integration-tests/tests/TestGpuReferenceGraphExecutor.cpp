// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <hip/hip_runtime.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <unordered_map>
#include <vector>

#include "harness/gpu_graph_executor/GpuReferenceGraphExecutor.hpp"

namespace
{

using namespace hipdnn_flatbuffers_sdk::data_objects;
using hipdnn_integration_tests::gpu_graph_executor::GpuReferenceGraphExecutor;

// Creates a minimal pointwise graph with two FLOAT tensors (input + output).
// The pointwise operation is RELU_FWD but the dummy plan ignores the operation.
flatbuffers::FlatBufferBuilder createSimplePointwiseGraph(int64_t inputUid,
                                                          int64_t outputUid,
                                                          const std::vector<int64_t>& dims,
                                                          const std::vector<int64_t>& strides)
{
    flatbuffers::FlatBufferBuilder builder;

    std::vector<flatbuffers::Offset<TensorAttributes>> tensors;
    tensors.push_back(
        CreateTensorAttributesDirect(builder, inputUid, "in_0", DataType::FLOAT, &strides, &dims));
    tensors.push_back(CreateTensorAttributesDirect(
        builder, outputUid, "out_0", DataType::FLOAT, &strides, &dims));

    auto pointwiseAttrs
        = CreatePointwiseAttributes(builder,
                                    PointwiseMode::RELU_FWD, // operation (ignored by dummy plan)
                                    flatbuffers::nullopt, // relu_lower_clip
                                    flatbuffers::nullopt, // relu_upper_clip
                                    flatbuffers::nullopt, // relu_lower_clip_slope
                                    flatbuffers::nullopt, // axis_tensor_uid
                                    inputUid, // in_0_tensor_uid
                                    flatbuffers::nullopt, // in_1_tensor_uid (unary, not needed)
                                    flatbuffers::nullopt, // in_2_tensor_uid (not needed)
                                    outputUid); // out_0_tensor_uid

    std::vector<flatbuffers::Offset<Node>> nodes;
    nodes.push_back(CreateNodeDirect(builder,
                                     "pointwise_node",
                                     DataType::FLOAT,
                                     NodeAttributes::PointwiseAttributes,
                                     pointwiseAttrs.Union()));

    auto graph = CreateGraphDirect(
        builder, "TestGraph", DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, &tensors, &nodes);

    builder.Finish(graph);
    return builder;
}

// Creates a minimal graph with a CustomOp node.
flatbuffers::FlatBufferBuilder createCustomOpGraph()
{
    flatbuffers::FlatBufferBuilder builder;

    const std::vector<int64_t> dims = {4};
    const std::vector<int64_t> strides = {1};

    std::vector<flatbuffers::Offset<TensorAttributes>> tensors;
    tensors.push_back(
        CreateTensorAttributesDirect(builder, 1, "in_0", DataType::FLOAT, &strides, &dims));
    tensors.push_back(
        CreateTensorAttributesDirect(builder, 2, "out_0", DataType::FLOAT, &strides, &dims));

    const std::vector<int64_t> inputUids = {1};
    const std::vector<int64_t> outputUids = {2};
    const std::vector<uint8_t> data;

    auto customOpAttrs
        = CreateCustomOpAttributesDirect(builder, "test.custom_op", &inputUids, &outputUids, &data);

    std::vector<flatbuffers::Offset<Node>> nodes;
    nodes.push_back(CreateNodeDirect(builder,
                                     "custom_op_node",
                                     DataType::FLOAT,
                                     NodeAttributes::CustomOpAttributes,
                                     customOpAttrs.Union()));

    auto graph = CreateGraphDirect(builder,
                                   "CustomOpTestGraph",
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   &tensors,
                                   &nodes);

    builder.Finish(graph);
    return builder;
}

// Creates a minimal graph with a BatchnormInference node (unsupported by GPU executor).
flatbuffers::FlatBufferBuilder createBatchnormInferenceGraph()
{
    flatbuffers::FlatBufferBuilder builder;

    const std::vector<int64_t> dims = {1, 2, 3, 4};
    const std::vector<int64_t> strides = {24, 12, 4, 1};

    const std::vector<int64_t> perChannelDims = {1, 2, 1, 1};
    const std::vector<int64_t> perChannelStrides = {2, 1, 1, 1};

    std::vector<flatbuffers::Offset<TensorAttributes>> tensors;
    tensors.push_back(
        CreateTensorAttributesDirect(builder, 1, "x", DataType::FLOAT, &strides, &dims));
    tensors.push_back(CreateTensorAttributesDirect(
        builder, 2, "mean", DataType::FLOAT, &perChannelStrides, &perChannelDims));
    tensors.push_back(CreateTensorAttributesDirect(
        builder, 3, "inv_variance", DataType::FLOAT, &perChannelStrides, &perChannelDims));
    tensors.push_back(CreateTensorAttributesDirect(
        builder, 4, "scale", DataType::FLOAT, &perChannelStrides, &perChannelDims));
    tensors.push_back(CreateTensorAttributesDirect(
        builder, 5, "bias", DataType::FLOAT, &perChannelStrides, &perChannelDims));
    tensors.push_back(
        CreateTensorAttributesDirect(builder, 6, "y", DataType::FLOAT, &strides, &dims));

    auto bnAttrs = CreateBatchnormInferenceAttributes(builder,
                                                      1, // x_tensor_uid
                                                      2, // mean_tensor_uid
                                                      3, // inv_variance_tensor_uid
                                                      4, // scale_tensor_uid
                                                      5, // bias_tensor_uid
                                                      6); // y_tensor_uid

    std::vector<flatbuffers::Offset<Node>> nodes;
    nodes.push_back(CreateNodeDirect(builder,
                                     "bn_inference_node",
                                     DataType::FLOAT,
                                     NodeAttributes::BatchnormInferenceAttributes,
                                     bnAttrs.Union()));

    auto graph = CreateGraphDirect(builder,
                                   "BnInferenceTestGraph",
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   &tensors,
                                   &nodes);

    builder.Finish(graph);
    return builder;
}

} // namespace

TEST(TestGpuReferenceGraphExecutor, CanBeConstructed)
{
    SKIP_IF_NO_DEVICES();

    const GpuReferenceGraphExecutor executor;
    static_cast<void>(executor);
}

TEST(TestGpuReferenceGraphExecutor, CustomOpThrows)
{
    SKIP_IF_NO_DEVICES();

    auto builder = createCustomOpGraph();

    const std::unordered_map<int64_t, void*> variantPack;

    GpuReferenceGraphExecutor executor;
    EXPECT_THROW(executor.execute(builder.GetBufferPointer(), builder.GetSize(), variantPack),
                 std::runtime_error);
}

TEST(TestGpuReferenceGraphExecutor, UnsupportedNodeTypeThrows)
{
    SKIP_IF_NO_DEVICES();

    // BatchnormInference has no GPU plan yet - should throw
    auto builder = createBatchnormInferenceGraph();

    const std::unordered_map<int64_t, void*> variantPack;

    GpuReferenceGraphExecutor executor;
    EXPECT_THROW(executor.execute(builder.GetBufferPointer(), builder.GetSize(), variantPack),
                 std::runtime_error);
}

TEST(TestGpuReferenceGraphExecutor, MissingVariantPackEntryThrows)
{
    SKIP_IF_NO_DEVICES();
    auto builder = createSimplePointwiseGraph(1, 2, {4}, {1});
    const std::unordered_map<int64_t, void*> emptyPack;
    GpuReferenceGraphExecutor executor;
    EXPECT_THROW(executor.execute(builder.GetBufferPointer(), builder.GetSize(), emptyPack),
                 std::out_of_range);
}

TEST(TestGpuReferenceGraphExecutor, PointwiseDummyAddOneExecutes)
{
    SKIP_IF_NO_DEVICES();

    constexpr int64_t INPUT_UID = 1;
    constexpr int64_t OUTPUT_UID = 2;
    const std::vector<int64_t> dims = {4};
    const std::vector<int64_t> strides = {1};

    auto builder = createSimplePointwiseGraph(INPUT_UID, OUTPUT_UID, dims, strides);

    std::array<float, 4> input = {2.0f, 3.0f, 5.0f, 7.0f};
    std::array<float, 4> output = {};

    std::unordered_map<int64_t, void*> variantPack;
    variantPack[INPUT_UID] = input.data();
    variantPack[OUTPUT_UID] = output.data();

    GpuReferenceGraphExecutor executor;
    executor.execute(builder.GetBufferPointer(), builder.GetSize(), variantPack);

    EXPECT_FLOAT_EQ(output[0], 3.0f);
    EXPECT_FLOAT_EQ(output[1], 4.0f);
    EXPECT_FLOAT_EQ(output[2], 6.0f);
    EXPECT_FLOAT_EQ(output[3], 8.0f);
}

TEST(TestGpuReferenceGraphExecutor, PointwiseDummyAddOneMultiDimensional)
{
    SKIP_IF_NO_DEVICES();

    constexpr int64_t INPUT_UID = 1;
    constexpr int64_t OUTPUT_UID = 2;
    const std::vector<int64_t> dims = {2, 3};
    const std::vector<int64_t> strides = {3, 1};

    auto builder = createSimplePointwiseGraph(INPUT_UID, OUTPUT_UID, dims, strides);

    std::array<float, 6> input = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    std::array<float, 6> output = {};

    std::unordered_map<int64_t, void*> variantPack;
    variantPack[INPUT_UID] = input.data();
    variantPack[OUTPUT_UID] = output.data();

    GpuReferenceGraphExecutor executor;
    executor.execute(builder.GetBufferPointer(), builder.GetSize(), variantPack);

    for(size_t i = 0; i < input.size(); ++i)
    {
        EXPECT_FLOAT_EQ(output[i], input[i] + 1.0f) << "Mismatch at index " << i;
    }
}
