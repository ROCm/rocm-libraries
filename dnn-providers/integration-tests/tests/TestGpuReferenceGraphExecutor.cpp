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

#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_data_sdk/utilities/Workspace.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/CpuReferenceGraphExecutor.hpp>

#include "ConvolutionFwdGraphTestUtils.hpp"
#include "harness/ReferenceCapabilityError.hpp"
#include "harness/gpu-graph-executor/GpuReferenceGraphExecutor.hpp"

namespace
{

using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_integration_tests::test_utils;
using hipdnn_integration_tests::gpu_graph_executor::GpuReferenceGraphExecutor;
using hipdnn_test_sdk::utilities::CpuReferenceGraphExecutor;

// Creates a minimal pointwise graph with two FLOAT tensors (input + output).
// The pointwise operation is RELU_FWD but the dummy plan ignores the operation.
// The runtime pass-by-value flags let a test flag the input or output operand
// PBV (Pointwise is fully supported, so a PBV operand alone is what makes the
// graph not-applicable). addUnconsumedPbvTensor adds a PBV tensor that NO node
// references, proving per-node rejection ignores tensors outside every op's
// operand set.
flatbuffers::FlatBufferBuilder createSimplePointwiseGraph(int64_t inputUid,
                                                          int64_t outputUid,
                                                          const std::vector<int64_t>& dims,
                                                          const std::vector<int64_t>& strides,
                                                          bool inputIsRuntimePassByValue = false,
                                                          bool outputIsRuntimePassByValue = false,
                                                          bool addUnconsumedPbvTensor = false)
{
    flatbuffers::FlatBufferBuilder builder;

    std::vector<flatbuffers::Offset<TensorAttributes>> tensors;
    tensors.push_back(CreateTensorAttributesDirect(builder,
                                                   inputUid,
                                                   "in_0",
                                                   DataType::FLOAT,
                                                   &strides,
                                                   &dims,
                                                   /*virtual_=*/false,
                                                   TensorValue::NONE,
                                                   /*value=*/0,
                                                   inputIsRuntimePassByValue));
    tensors.push_back(CreateTensorAttributesDirect(builder,
                                                   outputUid,
                                                   "out_0",
                                                   DataType::FLOAT,
                                                   &strides,
                                                   &dims,
                                                   /*virtual_=*/false,
                                                   TensorValue::NONE,
                                                   /*value=*/0,
                                                   outputIsRuntimePassByValue));
    if(addUnconsumedPbvTensor)
    {
        tensors.push_back(CreateTensorAttributesDirect(builder,
                                                       outputUid + 1000,
                                                       "unconsumed_pbv",
                                                       DataType::FLOAT,
                                                       &strides,
                                                       &dims,
                                                       /*virtual_=*/false,
                                                       TensorValue::NONE,
                                                       /*value=*/0,
                                                       /*is_runtime_pass_by_value=*/true));
    }

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

// Minimal ConvolutionFwd graph with the x tensor flagged runtime pass-by-value.
// Conv fwd is a supported GPU op, so the PBV flag alone must reject it. Conv has
// no scalar operand today, so this uses the shared conv builder's tensor layout
// but re-emits x with the PBV flag set.
flatbuffers::FlatBufferBuilder createConvFwdGraphWithPbvInput()
{
    const std::vector<int64_t> xDims{1, 1, 3, 3};
    const std::vector<int64_t> wDims{1, 1, 1, 1};
    const std::vector<int64_t> yDims{1, 1, 3, 3};
    const auto xStrides = generateStrides(xDims);
    const auto wStrides = generateStrides(wDims);
    const auto yStrides = generateStrides(yDims);
    const std::vector<int64_t> padding{0, 0};
    const std::vector<int64_t> convStride{1, 1};
    const std::vector<int64_t> dilation{1, 1};

    flatbuffers::FlatBufferBuilder builder;

    std::vector<flatbuffers::Offset<TensorAttributes>> tensors;
    tensors.push_back(CreateTensorAttributesDirect(builder,
                                                   1,
                                                   "x",
                                                   DataType::FLOAT,
                                                   &xStrides,
                                                   &xDims,
                                                   /*virtual_=*/false,
                                                   TensorValue::NONE,
                                                   /*value=*/0,
                                                   /*is_runtime_pass_by_value=*/true));
    tensors.push_back(
        CreateTensorAttributesDirect(builder, 2, "w", DataType::FLOAT, &wStrides, &wDims));
    tensors.push_back(
        CreateTensorAttributesDirect(builder, 3, "y", DataType::FLOAT, &yStrides, &yDims));

    auto convAttrs = CreateConvolutionFwdAttributesDirect(
        builder, 1, 2, 3, &padding, &padding, &convStride, &dilation, ConvMode::CROSS_CORRELATION);

    std::vector<flatbuffers::Offset<Node>> nodes;
    nodes.push_back(CreateNodeDirect(builder,
                                     "conv_fwd_node",
                                     DataType::FLOAT,
                                     NodeAttributes::ConvolutionFwdAttributes,
                                     convAttrs.Union()));

    auto graph = CreateGraphDirect(builder,
                                   "ConvFwdTestGraph",
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   &tensors,
                                   &nodes);
    builder.Finish(graph);
    return builder;
}

// Binary pointwise graph whose OPTIONAL second input (in_1) is runtime
// pass-by-value. This exercises the optional-operand branch of the per-op PBV
// scan: in_0/out_0 are PBV-free, only the optional operand is flagged.
flatbuffers::FlatBufferBuilder createPointwiseGraphWithPbvOptionalInput()
{
    const std::vector<int64_t> dims{4};
    const std::vector<int64_t> strides{1};

    flatbuffers::FlatBufferBuilder builder;

    std::vector<flatbuffers::Offset<TensorAttributes>> tensors;
    tensors.push_back(
        CreateTensorAttributesDirect(builder, 1, "in_0", DataType::FLOAT, &strides, &dims));
    tensors.push_back(CreateTensorAttributesDirect(builder,
                                                   2,
                                                   "in_1",
                                                   DataType::FLOAT,
                                                   &strides,
                                                   &dims,
                                                   /*virtual_=*/false,
                                                   TensorValue::NONE,
                                                   /*value=*/0,
                                                   /*is_runtime_pass_by_value=*/true));
    tensors.push_back(
        CreateTensorAttributesDirect(builder, 3, "out_0", DataType::FLOAT, &strides, &dims));

    auto pointwiseAttrs = CreatePointwiseAttributes(builder,
                                                    PointwiseMode::ADD,
                                                    flatbuffers::nullopt,
                                                    flatbuffers::nullopt,
                                                    flatbuffers::nullopt,
                                                    flatbuffers::nullopt,
                                                    1, // in_0
                                                    2, // in_1 (optional, PBV)
                                                    flatbuffers::nullopt,
                                                    3); // out_0

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

inline size_t elementCount(const std::vector<int64_t>& dims)
{
    size_t count = 1;
    for(auto d : dims)
    {
        count *= static_cast<size_t>(d);
    }
    return count;
}

template <typename T, typename ComputeT = float>
void runConvFwdExecutorVsCpu(const std::vector<int64_t>& xDims,
                             const std::vector<int64_t>& wDims,
                             const std::vector<int64_t>& yDims,
                             const std::vector<int64_t>& padding,
                             const std::vector<int64_t>& convStride,
                             const std::vector<int64_t>& dilation,
                             DataType dataType,
                             double tolerance)
{
    constexpr int64_t X_UID = 10;
    constexpr int64_t W_UID = 11;
    constexpr int64_t Y_UID = 12;

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
                                           convStride,
                                           dilation,
                                           dataType);

    auto xCount = elementCount(xDims);
    auto wCount = elementCount(wDims);
    auto yCount = elementCount(yDims);

    // Prepare CPU tensors and fill with deterministic data
    hipdnn_data_sdk::utilities::Tensor<T> cpuX(xDims, xStrides);
    hipdnn_data_sdk::utilities::Tensor<T> cpuW(wDims, wStrides);
    hipdnn_data_sdk::utilities::Tensor<T> cpuY(yDims, yStrides);

    for(size_t i = 0; i < xCount; ++i)
    {
        static_cast<T*>(cpuX.rawHostData())[i] = T(static_cast<float>(i + 1));
    }
    for(size_t i = 0; i < wCount; ++i)
    {
        static_cast<T*>(cpuW.rawHostData())[i] = T(1.0f);
    }

    // Allocate device buffers (RAII — freed automatically)
    const hipdnn_data_sdk::utilities::Workspace dX(xCount * sizeof(T));
    const hipdnn_data_sdk::utilities::Workspace dW(wCount * sizeof(T));
    const hipdnn_data_sdk::utilities::Workspace dY(yCount * sizeof(T));

    ASSERT_EQ(hipMemset(dY.get(), 0, yCount * sizeof(T)), hipSuccess);
    ASSERT_EQ(hipMemcpy(dX.get(), cpuX.rawHostData(), xCount * sizeof(T), hipMemcpyHostToDevice),
              hipSuccess);
    ASSERT_EQ(hipMemcpy(dW.get(), cpuW.rawHostData(), wCount * sizeof(T), hipMemcpyHostToDevice),
              hipSuccess);

    // Run GPU graph executor with device pointers
    std::unordered_map<int64_t, void*> variantPack;
    variantPack[X_UID] = dX.get();
    variantPack[W_UID] = dW.get();
    variantPack[Y_UID] = dY.get();

    GpuReferenceGraphExecutor gpuExecutor;
    gpuExecutor.execute(graphBuilder.GetBufferPointer(), graphBuilder.GetSize(), variantPack);

    // Copy GPU result back to host
    std::vector<T> gpuYData(yCount);
    ASSERT_EQ(hipMemcpy(gpuYData.data(), dY.get(), yCount * sizeof(T), hipMemcpyDeviceToHost),
              hipSuccess);

    // Run CPU reference executor with host pointers (same graph)
    std::unordered_map<int64_t, void*> cpuVariantPack;
    cpuVariantPack[X_UID] = cpuX.rawHostData();
    cpuVariantPack[W_UID] = cpuW.rawHostData();
    cpuVariantPack[Y_UID] = cpuY.rawHostData();

    CpuReferenceGraphExecutor cpuExecutor;
    cpuExecutor.execute(graphBuilder.GetBufferPointer(), graphBuilder.GetSize(), cpuVariantPack);

    // Compare GPU executor output against CPU executor output
    const auto* cpuResult = static_cast<const T*>(cpuY.rawHostData());
    for(size_t i = 0; i < yCount; ++i)
    {
        EXPECT_NEAR(static_cast<float>(gpuYData[i]), static_cast<float>(cpuResult[i]), tolerance)
            << "Mismatch at index " << i;
    }
}

} // namespace

TEST(TestGpuReferenceGraphExecutor, CanBeConstructed)
{
    SKIP_IF_NO_DEVICES();

    const GpuReferenceGraphExecutor executor;
    static_cast<void>(executor);
}

TEST(TestGpuReferenceGraphExecutor, IsNotApplicableForRuntimePassByValueGraph)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormFwdTrainingGraph(
        {588, 196, 14, 1},
        {1, 3, 14, 14},
        /*withMeanVariance=*/true,
        /*overrideShapeEnabled=*/false,
        /*runtimeEpsilon=*/true);
    GpuReferenceGraphExecutor executor;
    EXPECT_FALSE(executor.isApplicable(builder.GetBufferPointer(), builder.GetSize()));
}

TEST(TestGpuReferenceGraphExecutor, IsApplicableForBakedScalarGraph)
{
    const std::vector<int64_t> xDims{1, 1, 3, 3};
    const std::vector<int64_t> wDims{1, 1, 1, 1};
    const std::vector<int64_t> yDims{1, 1, 3, 3};
    auto builder = createConvFwdGraph(1,
                                      2,
                                      3,
                                      xDims,
                                      wDims,
                                      yDims,
                                      generateStrides(xDims),
                                      generateStrides(wDims),
                                      generateStrides(yDims),
                                      {0, 0},
                                      {1, 1},
                                      {1, 1},
                                      DataType::FLOAT,
                                      DataType::FLOAT);
    GpuReferenceGraphExecutor executor;
    EXPECT_TRUE(executor.isApplicable(builder.GetBufferPointer(), builder.GetSize()));
}

TEST(TestGpuReferenceGraphExecutor, IsApplicableWhenRuntimePassByValueTensorIsNotConsumed)
{
    const std::vector<int64_t> xDims{1, 1, 3, 3};
    const std::vector<int64_t> wDims{1, 1, 1, 1};
    const std::vector<int64_t> yDims{1, 1, 3, 3};
    const auto xStrides = generateStrides(xDims);
    const auto wStrides = generateStrides(wDims);
    const auto yStrides = generateStrides(yDims);

    flatbuffers::FlatBufferBuilder builder;

    std::vector<flatbuffers::Offset<TensorAttributes>> tensors;
    tensors.push_back(
        CreateTensorAttributesDirect(builder, 1, "x", DataType::FLOAT, &xStrides, &xDims));
    tensors.push_back(
        CreateTensorAttributesDirect(builder, 2, "w", DataType::FLOAT, &wStrides, &wDims));
    tensors.push_back(
        CreateTensorAttributesDirect(builder, 3, "y", DataType::FLOAT, &yStrides, &yDims));
    tensors.push_back(CreateTensorAttributesDirect(builder,
                                                   999,
                                                   "unconsumed_pbv",
                                                   DataType::FLOAT,
                                                   &xStrides,
                                                   &xDims,
                                                   /*virtual_=*/false,
                                                   TensorValue::NONE,
                                                   /*value=*/0,
                                                   /*is_runtime_pass_by_value=*/true));

    const std::vector<int64_t> padding{0, 0};
    const std::vector<int64_t> convStride{1, 1};
    const std::vector<int64_t> dilation{1, 1};
    auto convAttrs = CreateConvolutionFwdAttributesDirect(
        builder, 1, 2, 3, &padding, &padding, &convStride, &dilation, ConvMode::CROSS_CORRELATION);

    std::vector<flatbuffers::Offset<Node>> nodes;
    nodes.push_back(CreateNodeDirect(builder,
                                     "conv_fwd_node",
                                     DataType::FLOAT,
                                     NodeAttributes::ConvolutionFwdAttributes,
                                     convAttrs.Union()));

    auto graph = CreateGraphDirect(builder,
                                   "ConvFwdWithUnconsumedPbv",
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   &tensors,
                                   &nodes);
    builder.Finish(graph);

    GpuReferenceGraphExecutor executor;
    EXPECT_TRUE(executor.isApplicable(builder.GetBufferPointer(), builder.GetSize()));
}

TEST(TestGpuReferenceGraphExecutor, IsNotApplicableWhenSupportedOpTensorIsRuntimePassByValue)
{
    // Pointwise is fully supported by the GPU reference (see IsApplicableForBakedScalarGraph),
    // so flagging its input tensor runtime-PBV must be the sole cause of rejection.
    auto builder = createSimplePointwiseGraph(1, 2, {4}, {1}, /*inputIsRuntimePassByValue=*/true);
    GpuReferenceGraphExecutor executor;
    EXPECT_FALSE(executor.isApplicable(builder.GetBufferPointer(), builder.GetSize()));
}

TEST(TestGpuReferenceGraphExecutor, IsNotApplicableWhenSupportedOpOutputIsRuntimePassByValue)
{
    // The per-node PBV scan must cover EVERY operand, not just the input:
    // flagging the output tensor runtime-PBV must also reject the graph.
    auto builder = createSimplePointwiseGraph(1,
                                              2,
                                              {4},
                                              {1},
                                              /*inputIsRuntimePassByValue=*/false,
                                              /*outputIsRuntimePassByValue=*/true);
    GpuReferenceGraphExecutor executor;
    EXPECT_FALSE(executor.isApplicable(builder.GetBufferPointer(), builder.GetSize()));
}

TEST(TestGpuReferenceGraphExecutor, IsNotApplicableWhenConvInputIsRuntimePassByValue)
{
    // ConvolutionFwd is a supported GPU op; flagging its x operand runtime-PBV
    // must reject the graph, proving the per-op scan is wired for conv too.
    auto builder = createConvFwdGraphWithPbvInput();
    GpuReferenceGraphExecutor executor;
    EXPECT_FALSE(executor.isApplicable(builder.GetBufferPointer(), builder.GetSize()));
}

TEST(TestGpuReferenceGraphExecutor, IsNotApplicableWhenOptionalPointwiseOperandIsRuntimePassByValue)
{
    // Only the OPTIONAL in_1 operand is PBV (in_0/out_0 are PBV-free), exercising
    // the optional-operand branch of the scan — the flag must still reject.
    auto builder = createPointwiseGraphWithPbvOptionalInput();
    GpuReferenceGraphExecutor executor;
    EXPECT_FALSE(executor.isApplicable(builder.GetBufferPointer(), builder.GetSize()));
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

    // Use a convolution graph (a real GPU plan) so execution reaches the
    // variant-pack lookup; an empty pack must surface as std::out_of_range.
    const std::vector<int64_t> xDims = {1, 1, 4, 4};
    const std::vector<int64_t> wDims = {1, 1, 3, 3};
    const std::vector<int64_t> yDims = {1, 1, 2, 2};
    auto builder = createConvFwdGraph(10,
                                      11,
                                      12,
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

    const std::unordered_map<int64_t, void*> emptyPack;
    GpuReferenceGraphExecutor executor;
    EXPECT_THROW(executor.execute(builder.GetBufferPointer(), builder.GetSize(), emptyPack),
                 std::out_of_range);
}

TEST(TestGpuReferenceGraphExecutor, PointwiseIsApplicable)
{
    SKIP_IF_NO_DEVICES();

    auto builder = createSimplePointwiseGraph(1, 2, {4}, {1});

    GpuReferenceGraphExecutor executor;
    EXPECT_TRUE(executor.isApplicable(builder.GetBufferPointer(), builder.GetSize()));
}

TEST(TestGpuReferenceGraphExecutorFp32, ConvFwdBasicExecutes)
{
    SKIP_IF_NO_DEVICES();

    runConvFwdExecutorVsCpu<float>({1, 1, 4, 4}, // xDims
                                   {1, 1, 3, 3}, // wDims
                                   {1, 1, 2, 2}, // yDims
                                   {0, 0}, // padding
                                   {1, 1}, // stride
                                   {1, 1}, // dilation
                                   DataType::FLOAT,
                                   1e-5);
}

TEST(TestGpuReferenceGraphExecutorFp32, ConvFwdWithPaddingExecutes)
{
    SKIP_IF_NO_DEVICES();

    runConvFwdExecutorVsCpu<float>({1, 1, 4, 4}, // xDims
                                   {1, 1, 3, 3}, // wDims
                                   {1, 1, 4, 4}, // yDims (same as input due to padding=1)
                                   {1, 1}, // padding
                                   {1, 1}, // stride
                                   {1, 1}, // dilation
                                   DataType::FLOAT,
                                   1e-5);
}

TEST(TestGpuReferenceGraphExecutorFp16, ConvFwdExecutes)
{
    SKIP_IF_NO_DEVICES();

    runConvFwdExecutorVsCpu<hipdnn_data_sdk::types::half>({1, 1, 4, 4}, // xDims
                                                          {1, 1, 3, 3}, // wDims
                                                          {1, 1, 2, 2}, // yDims
                                                          {0, 0}, // padding
                                                          {1, 1}, // stride
                                                          {1, 1}, // dilation
                                                          DataType::HALF,
                                                          0.01);
}

TEST(TestGpuReferenceGraphExecutorBfp16, ConvFwdExecutes)
{
    SKIP_IF_NO_DEVICES();

    runConvFwdExecutorVsCpu<hipdnn_data_sdk::types::bfloat16>({1, 1, 4, 4}, // xDims
                                                              {1, 1, 3, 3}, // wDims
                                                              {1, 1, 2, 2}, // yDims
                                                              {0, 0}, // padding
                                                              {1, 1}, // stride
                                                              {1, 1}, // dilation
                                                              DataType::BFLOAT16,
                                                              0.1);
}

TEST(TestGpuReferenceGraphExecutorFp32, PointwiseUnaryExecutes)
{
    SKIP_IF_NO_DEVICES();

    constexpr int64_t IN_UID = 10;
    constexpr int64_t OUT_UID = 11;
    const std::vector<int64_t> dims = {2, 3, 4, 4};
    auto strides = generateStrides(dims);

    flatbuffers::FlatBufferBuilder builder;
    std::vector<flatbuffers::Offset<TensorAttributes>> tensors;
    tensors.push_back(CreateTensorAttributesDirect(builder,
                                                   IN_UID,
                                                   "in_0",
                                                   DataType::FLOAT,
                                                   &strides,
                                                   &dims,
                                                   /*virtual_=*/false,
                                                   TensorValue::NONE,
                                                   /*value=*/0,
                                                   false));
    tensors.push_back(CreateTensorAttributesDirect(builder,
                                                   OUT_UID,
                                                   "out_0",
                                                   DataType::FLOAT,
                                                   &strides,
                                                   &dims,
                                                   /*virtual_=*/false,
                                                   TensorValue::NONE,
                                                   /*value=*/0,
                                                   false));

    // Create an ABS unary pointwise operation
    auto pointwiseAttrs
        = CreatePointwiseAttributes(builder,
                                    PointwiseMode::ABS, // operation
                                    flatbuffers::nullopt, // relu_lower_clip
                                    flatbuffers::nullopt, // relu_upper_clip
                                    flatbuffers::nullopt, // relu_lower_clip_slope
                                    flatbuffers::nullopt, // axis_tensor_uid
                                    IN_UID, // in_0_tensor_uid
                                    flatbuffers::nullopt, // in_1_tensor_uid (unary, not needed)
                                    flatbuffers::nullopt, // in_2_tensor_uid (not needed)
                                    OUT_UID); // out_0_tensor_uid

    std::vector<flatbuffers::Offset<Node>> nodes;
    nodes.push_back(CreateNodeDirect(builder,
                                     "pointwise_node",
                                     DataType::FLOAT,
                                     NodeAttributes::PointwiseAttributes,
                                     pointwiseAttrs.Union()));

    auto graph = CreateGraphDirect(
        builder, "TestGraph", DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, &tensors, &nodes);

    builder.Finish(graph);

    hipdnn_data_sdk::utilities::Tensor<float> inputTensor(dims, strides);
    hipdnn_data_sdk::utilities::Tensor<float> outputTensor(dims, strides);
    inputTensor.fillWithRandomValues(-1.0f, 1.0f);
    outputTensor.fillWithValue(0);

    std::unordered_map<int64_t, void*> variantPack;
    variantPack[IN_UID] = inputTensor.rawDeviceData();
    variantPack[OUT_UID] = outputTensor.rawDeviceData();

    GpuReferenceGraphExecutor gpuExecutor;
    gpuExecutor.execute(builder.GetBufferPointer(), builder.GetSize(), variantPack);
    outputTensor.markDeviceModified();

    for(size_t i = 0; i < outputTensor.elementCount(); ++i)
    {
        const float expected = std::fabs(static_cast<float*>(inputTensor.rawHostData())[i]);
        EXPECT_EQ(expected, static_cast<float*>(outputTensor.rawHostData())[i])
            << "Mismatch at index " << i;
    }
}

TEST(TestGpuReferenceGraphExecutorFp32, PointwiseBinaryExecutes)
{
    SKIP_IF_NO_DEVICES();

    constexpr int64_t IN_0_UID = 10;
    constexpr int64_t IN_1_UID = 11;
    constexpr int64_t OUT_UID = 12;
    const std::vector<int64_t> dims = {2, 3, 4, 4};
    auto strides = generateStrides(dims);

    flatbuffers::FlatBufferBuilder builder;
    std::vector<flatbuffers::Offset<TensorAttributes>> tensors;
    tensors.push_back(CreateTensorAttributesDirect(builder,
                                                   IN_0_UID,
                                                   "in_0",
                                                   DataType::FLOAT,
                                                   &strides,
                                                   &dims,
                                                   /*virtual_=*/false,
                                                   TensorValue::NONE,
                                                   /*value=*/0,
                                                   false));
    tensors.push_back(CreateTensorAttributesDirect(builder,
                                                   IN_1_UID,
                                                   "in_1",
                                                   DataType::FLOAT,
                                                   &strides,
                                                   &dims,
                                                   /*virtual_=*/false,
                                                   TensorValue::NONE,
                                                   /*value=*/0,
                                                   false));
    tensors.push_back(CreateTensorAttributesDirect(builder,
                                                   OUT_UID,
                                                   "out_0",
                                                   DataType::FLOAT,
                                                   &strides,
                                                   &dims,
                                                   /*virtual_=*/false,
                                                   TensorValue::NONE,
                                                   /*value=*/0,
                                                   false));

    // Create an Add binary pointwise operation
    auto pointwiseAttrs
        = CreatePointwiseAttributes(builder,
                                    PointwiseMode::ADD, // operation
                                    flatbuffers::nullopt, // relu_lower_clip
                                    flatbuffers::nullopt, // relu_upper_clip
                                    flatbuffers::nullopt, // relu_lower_clip_slope
                                    flatbuffers::nullopt, // axis_tensor_uid
                                    IN_0_UID, // in_0_tensor_uid
                                    IN_1_UID, // in_1_tensor_uid (unary, not needed)
                                    flatbuffers::nullopt, // in_2_tensor_uid (not needed)
                                    OUT_UID); // out_0_tensor_uid

    std::vector<flatbuffers::Offset<Node>> nodes;
    nodes.push_back(CreateNodeDirect(builder,
                                     "pointwise_node",
                                     DataType::FLOAT,
                                     NodeAttributes::PointwiseAttributes,
                                     pointwiseAttrs.Union()));

    auto graph = CreateGraphDirect(
        builder, "TestGraph", DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, &tensors, &nodes);

    builder.Finish(graph);

    hipdnn_data_sdk::utilities::Tensor<float> input0Tensor(dims, strides);
    hipdnn_data_sdk::utilities::Tensor<float> input1Tensor(dims, strides);
    hipdnn_data_sdk::utilities::Tensor<float> outputTensor(dims, strides);
    input0Tensor.fillWithRandomValues(-1.0f, 1.0f);
    input1Tensor.fillWithRandomValues(-1.0f, 1.0f);
    outputTensor.fillWithValue(0);

    std::unordered_map<int64_t, void*> variantPack;
    variantPack[IN_0_UID] = input0Tensor.rawDeviceData();
    variantPack[IN_1_UID] = input1Tensor.rawDeviceData();
    variantPack[OUT_UID] = outputTensor.rawDeviceData();

    GpuReferenceGraphExecutor gpuExecutor;
    gpuExecutor.execute(builder.GetBufferPointer(), builder.GetSize(), variantPack);
    outputTensor.markDeviceModified();

    for(size_t i = 0; i < outputTensor.elementCount(); ++i)
    {
        const float expected = static_cast<float*>(input0Tensor.rawHostData())[i]
                               + static_cast<float*>(input1Tensor.rawHostData())[i];
        EXPECT_EQ(expected, static_cast<float*>(outputTensor.rawHostData())[i])
            << "Mismatch at index " << i;
    }
}
