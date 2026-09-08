// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <cstdint>
#include <hip/hip_runtime.h>
#include <hipdnn_data_sdk/utilities/Constants.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceLayernorm.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceMatmul.hpp>
#include <hipdnn_test_sdk/utilities/TestTolerances.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <unordered_map>
#include <vector>

#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_data_sdk/utilities/Workspace.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/CpuReferenceGraphExecutor.hpp>

#include "ConvolutionFwdGraphTestUtils.hpp"
#include "LayernormBwdGraphTestUtils.hpp"
#include "LayernormFwdGraphTestUtils.hpp"
#include "MatmulGraphTestUtils.hpp"
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

TEST(TestGpuReferenceGraphExecutorFp32, LayernormFwdIsApplicable)
{
    SKIP_IF_NO_DEVICES();

    const std::vector<int64_t> dims = {2, 3, 5, 7};
    const std::vector<int64_t> batchDims = {2, 1, 1, 1};
    const std::vector<int64_t> normDims = {1, 3, 5, 7};
    auto strides = generateStrides(dims);
    auto batchStrides = generateStrides(batchDims);
    auto normStrides = generateStrides(normDims);

    auto graphBuilder = createLayernormFwdGraph(10,
                                                11,
                                                12,
                                                13,
                                                14,
                                                15,
                                                16,
                                                dims,
                                                dims,
                                                normDims,
                                                normDims,
                                                batchDims,
                                                batchDims,
                                                strides,
                                                strides,
                                                normStrides,
                                                normStrides,
                                                batchStrides,
                                                batchStrides,
                                                static_cast<float>(LAYERNORM_DEFAULT_EPSILON),
                                                3,
                                                DataType::FLOAT,
                                                DataType::FLOAT,
                                                DataType::FLOAT,
                                                DataType::FLOAT,
                                                DataType::FLOAT);

    GpuReferenceGraphExecutor executor;
    EXPECT_TRUE(executor.isApplicable(graphBuilder.GetBufferPointer(), graphBuilder.GetSize()));
}

TEST(TestGpuReferenceGraphExecutorFp32, LayernormFwdExecutes)
{
    SKIP_IF_NO_DEVICES();

    const std::vector<int64_t> dims = {2, 3, 5, 7};
    const std::vector<int64_t> batchDims = {2, 1, 1, 1};
    const std::vector<int64_t> normDims = {1, 3, 5, 7};
    auto strides = generateStrides(dims);
    auto batchStrides = generateStrides(batchDims);
    auto normStrides = generateStrides(normDims);

    auto graphBuilder = createLayernormFwdGraph(10,
                                                11,
                                                12,
                                                13,
                                                14,
                                                15,
                                                16,
                                                dims,
                                                dims,
                                                normDims,
                                                normDims,
                                                batchDims,
                                                batchDims,
                                                strides,
                                                strides,
                                                normStrides,
                                                normStrides,
                                                batchStrides,
                                                batchStrides,
                                                static_cast<float>(LAYERNORM_DEFAULT_EPSILON),
                                                3,
                                                DataType::FLOAT,
                                                DataType::FLOAT,
                                                DataType::FLOAT,
                                                DataType::FLOAT,
                                                DataType::FLOAT);

    hipdnn_data_sdk::utilities::Tensor<float> xTensor(dims, strides);
    hipdnn_data_sdk::utilities::Tensor<float> scaleTensor(normDims, normStrides);
    hipdnn_data_sdk::utilities::Tensor<float> biasTensor(normDims, normStrides);
    hipdnn_data_sdk::utilities::Tensor<float> yTensor(dims, strides);
    hipdnn_data_sdk::utilities::Tensor<float> meanTensor(batchDims, batchStrides);
    hipdnn_data_sdk::utilities::Tensor<float> rstdTensor(batchDims, batchStrides);
    hipdnn_data_sdk::utilities::Tensor<float> epsilonTensor({1}, {1});

    xTensor.fillWithRandomValues(-1.0f, 1.0f);
    scaleTensor.fillWithRandomValues(-1.0f, 1.0f);
    biasTensor.fillWithRandomValues(-1.0f, 1.0f);
    epsilonTensor.fillWithValue(static_cast<float>(LAYERNORM_DEFAULT_EPSILON));

    std::unordered_map<int64_t, void*> variantPack;
    variantPack[10] = xTensor.rawDeviceData();
    variantPack[11] = yTensor.rawDeviceData();
    variantPack[12] = scaleTensor.rawDeviceData();
    variantPack[13] = biasTensor.rawDeviceData();
    variantPack[14] = epsilonTensor.rawDeviceData();
    variantPack[15] = meanTensor.rawDeviceData();
    variantPack[16] = rstdTensor.rawDeviceData();

    GpuReferenceGraphExecutor gpuExecutor;
    gpuExecutor.execute(graphBuilder.GetBufferPointer(), graphBuilder.GetSize(), variantPack);
    yTensor.markDeviceModified();
    meanTensor.markDeviceModified();
    rstdTensor.markDeviceModified();

    // Validate against CPU reference implementation
    hipdnn_data_sdk::utilities::Tensor<float> refYTensor(dims, strides);
    hipdnn_data_sdk::utilities::Tensor<float> refMeanTensor(batchDims, batchStrides);
    hipdnn_data_sdk::utilities::Tensor<float> refRstdTensor(batchDims, batchStrides);
    hipdnn_test_sdk::utilities::CpuFpReferenceLayernorm::fprop(xTensor,
                                                               &scaleTensor,
                                                               &biasTensor,
                                                               refYTensor,
                                                               LAYERNORM_DEFAULT_EPSILON,
                                                               3,
                                                               &refMeanTensor,
                                                               &refRstdTensor);
    refYTensor.markHostModified();
    refMeanTensor.markHostModified();
    refRstdTensor.markHostModified();

    auto* yData = static_cast<float*>(yTensor.rawHostData());
    auto* refYData = static_cast<float*>(refYTensor.rawHostData());
    for(size_t i = 0; i < yTensor.elementCount(); ++i)
    {
        EXPECT_NEAR(
            yData[i], refYData[i], hipdnn_test_sdk::utilities::layernorm::getTolerance<float>())
            << "Mismatch in y at index " << i;
    }

    auto* meanData = static_cast<float*>(meanTensor.rawHostData());
    auto* refMeanData = static_cast<float*>(refMeanTensor.rawHostData());
    for(size_t i = 0; i < meanTensor.elementCount(); ++i)
    {
        EXPECT_NEAR(meanData[i],
                    refMeanData[i],
                    hipdnn_test_sdk::utilities::layernorm::getTolerance<float>())
            << "Mismatch in mean at index " << i;
    }

    auto* rstdData = static_cast<float*>(rstdTensor.rawHostData());
    auto* refRstdData = static_cast<float*>(refRstdTensor.rawHostData());
    for(size_t i = 0; i < rstdTensor.elementCount(); ++i)
    {
        EXPECT_NEAR(rstdData[i],
                    refRstdData[i],
                    hipdnn_test_sdk::utilities::layernorm::getTolerance<float>())
            << "Mismatch in rstd at index " << i;
    }
}

TEST(TestGpuReferenceGraphExecutorFp32, LayernormBwdIsApplicable)
{
    SKIP_IF_NO_DEVICES();

    const std::vector<int64_t> dims = {2, 3, 5, 7};
    const std::vector<int64_t> batchDims = {2, 1, 1, 1};
    const std::vector<int64_t> normDims = {1, 3, 5, 7};
    auto strides = generateStrides(dims);
    auto batchStrides = generateStrides(batchDims);
    auto normStrides = generateStrides(normDims);

    auto graphBuilder = createLayernormBwdGraph(10,
                                                11,
                                                12,
                                                13,
                                                14,
                                                15,
                                                16,
                                                17,
                                                18,
                                                dims,
                                                dims,
                                                normDims,
                                                dims,
                                                normDims,
                                                normDims,
                                                batchDims,
                                                batchDims,
                                                strides,
                                                strides,
                                                normStrides,
                                                strides,
                                                normStrides,
                                                normStrides,
                                                batchStrides,
                                                batchStrides,
                                                static_cast<float>(LAYERNORM_DEFAULT_EPSILON),
                                                3,
                                                DataType::FLOAT,
                                                DataType::FLOAT,
                                                DataType::FLOAT,
                                                DataType::FLOAT,
                                                DataType::FLOAT);

    GpuReferenceGraphExecutor executor;
    EXPECT_TRUE(executor.isApplicable(graphBuilder.GetBufferPointer(), graphBuilder.GetSize()));
}

TEST(TestGpuReferenceGraphExecutorFp32, LayernormBwdExecutes)
{
    SKIP_IF_NO_DEVICES();

    const std::vector<int64_t> dims = {2, 3, 5, 7};
    const std::vector<int64_t> batchDims = {2, 1, 1, 1};
    const std::vector<int64_t> normDims = {1, 3, 5, 7};
    auto strides = generateStrides(dims);
    auto batchStrides = generateStrides(batchDims);
    auto normStrides = generateStrides(normDims);

    auto graphBuilder = createLayernormBwdGraph(10,
                                                11,
                                                12,
                                                13,
                                                14,
                                                15,
                                                16,
                                                17,
                                                18,
                                                dims,
                                                dims,
                                                normDims,
                                                dims,
                                                normDims,
                                                normDims,
                                                batchDims,
                                                batchDims,
                                                strides,
                                                strides,
                                                normStrides,
                                                strides,
                                                normStrides,
                                                normStrides,
                                                batchStrides,
                                                batchStrides,
                                                static_cast<float>(LAYERNORM_DEFAULT_EPSILON),
                                                3,
                                                DataType::FLOAT,
                                                DataType::FLOAT,
                                                DataType::FLOAT,
                                                DataType::FLOAT,
                                                DataType::FLOAT);

    hipdnn_data_sdk::utilities::Tensor<float> dyTensor(dims, strides);
    hipdnn_data_sdk::utilities::Tensor<float> xTensor(dims, strides);
    hipdnn_data_sdk::utilities::Tensor<float> scaleTensor(normDims, normStrides);
    hipdnn_data_sdk::utilities::Tensor<float> dxTensor(dims, strides);
    hipdnn_data_sdk::utilities::Tensor<float> dscaleTensor(normDims, normStrides);
    hipdnn_data_sdk::utilities::Tensor<float> dbiasTensor(normDims, normStrides);
    hipdnn_data_sdk::utilities::Tensor<float> epsilonTensor({1}, {1});
    hipdnn_data_sdk::utilities::Tensor<float> meanTensor(batchDims, batchStrides);
    hipdnn_data_sdk::utilities::Tensor<float> rstdTensor(batchDims, batchStrides);

    dyTensor.fillWithRandomValues(-1.0f, 1.0f);
    xTensor.fillWithRandomValues(-1.0f, 1.0f);
    scaleTensor.fillWithRandomValues(-1.0f, 1.0f);
    epsilonTensor.fillWithValue(static_cast<float>(LAYERNORM_DEFAULT_EPSILON));
    meanTensor.fillWithRandomValues(-1.0f, 1.0f);
    rstdTensor.fillWithRandomValues(-1.0f, 1.0f);

    std::unordered_map<int64_t, void*> variantPack;
    variantPack[10] = dyTensor.rawDeviceData();
    variantPack[11] = xTensor.rawDeviceData();
    variantPack[12] = scaleTensor.rawDeviceData();
    variantPack[13] = dxTensor.rawDeviceData();
    variantPack[14] = dscaleTensor.rawDeviceData();
    variantPack[15] = dbiasTensor.rawDeviceData();
    variantPack[16] = epsilonTensor.rawDeviceData();
    variantPack[17] = meanTensor.rawDeviceData();
    variantPack[18] = rstdTensor.rawDeviceData();

    GpuReferenceGraphExecutor gpuExecutor;
    gpuExecutor.execute(graphBuilder.GetBufferPointer(), graphBuilder.GetSize(), variantPack);
    dxTensor.markDeviceModified();
    dscaleTensor.markDeviceModified();
    dbiasTensor.markDeviceModified();

    // Validate against CPU reference implementation
    hipdnn_data_sdk::utilities::Tensor<float> refDxTensor(dims, strides);
    hipdnn_data_sdk::utilities::Tensor<float> refDscaleTensor(normDims, normStrides);
    hipdnn_data_sdk::utilities::Tensor<float> refDbiasTensor(normDims, normStrides);
    hipdnn_test_sdk::utilities::CpuFpReferenceLayernorm::bprop(dyTensor,
                                                               xTensor,
                                                               scaleTensor,
                                                               refDxTensor,
                                                               refDscaleTensor,
                                                               refDbiasTensor,
                                                               LAYERNORM_DEFAULT_EPSILON,
                                                               &meanTensor,
                                                               &rstdTensor,
                                                               3);
    refDxTensor.markHostModified();
    refDscaleTensor.markHostModified();
    refDbiasTensor.markHostModified();

    auto* dxData = static_cast<float*>(dxTensor.rawHostData());
    auto* refDxData = static_cast<float*>(refDxTensor.rawHostData());
    for(size_t i = 0; i < dxTensor.elementCount(); ++i)
    {
        EXPECT_NEAR(
            dxData[i], refDxData[i], hipdnn_test_sdk::utilities::layernorm::getTolerance<float>())
            << "Mismatch in dx at index " << i;
    }

    auto* dscaleData = static_cast<float*>(dscaleTensor.rawHostData());
    auto* refDscaleData = static_cast<float*>(refDscaleTensor.rawHostData());
    for(size_t i = 0; i < dscaleTensor.elementCount(); ++i)
    {
        EXPECT_NEAR(dscaleData[i],
                    refDscaleData[i],
                    hipdnn_test_sdk::utilities::layernorm::getTolerance<float>())
            << "Mismatch in dscale at index " << i;
    }

    auto* dbiasData = static_cast<float*>(dbiasTensor.rawHostData());
    auto* refDbiasData = static_cast<float*>(refDbiasTensor.rawHostData());
    for(size_t i = 0; i < dbiasTensor.elementCount(); ++i)
    {
        EXPECT_NEAR(dbiasData[i],
                    refDbiasData[i],
                    hipdnn_test_sdk::utilities::layernorm::getTolerance<float>())
            << "Mismatch in dbias at index " << i;
    }
}

TEST(TestGpuReferenceGraphExecutorFp32, MatmulIsApplicable)
{
    SKIP_IF_NO_DEVICES();

    const std::vector<int64_t> aDims = {2, 6, 5, 7};
    const std::vector<int64_t> bDims = {4, 3, 7, 11};
    const std::vector<int64_t> cDims = {4, 6, 5, 11};
    auto aStrides = generateStrides(aDims);
    auto bStrides = generateStrides(bDims);
    auto cStrides = generateStrides(cDims);

    auto graphBuilder = createMatmulGraph(10,
                                          11,
                                          12,
                                          aDims,
                                          aStrides,
                                          bDims,
                                          bStrides,
                                          cDims,
                                          cStrides,
                                          DataType::FLOAT,
                                          DataType::FLOAT,
                                          DataType::FLOAT,
                                          DataType::FLOAT);

    GpuReferenceGraphExecutor executor;
    EXPECT_TRUE(executor.isApplicable(graphBuilder.GetBufferPointer(), graphBuilder.GetSize()));
}

TEST(TestGpuReferenceGraphExecutorFp32, MatmulExecutes)
{
    SKIP_IF_NO_DEVICES();

    const std::vector<int64_t> aDims = {2, 6, 5, 7};
    const std::vector<int64_t> bDims = {4, 3, 7, 11};
    const std::vector<int64_t> cDims = {4, 6, 5, 11};
    auto aStrides = generateStrides(aDims);
    auto bStrides = generateStrides(bDims);
    auto cStrides = generateStrides(cDims);

    auto graphBuilder = createMatmulGraph(10,
                                          11,
                                          12,
                                          aDims,
                                          aStrides,
                                          bDims,
                                          bStrides,
                                          cDims,
                                          cStrides,
                                          DataType::FLOAT,
                                          DataType::FLOAT,
                                          DataType::FLOAT,
                                          DataType::FLOAT);

    hipdnn_data_sdk::utilities::Tensor<float> aTensor(aDims, aStrides);
    hipdnn_data_sdk::utilities::Tensor<float> bTensor(bDims, bStrides);
    hipdnn_data_sdk::utilities::Tensor<float> cTensor(cDims, cStrides);

    aTensor.fillWithRandomValues(-1.0f, 1.0f);
    bTensor.fillWithRandomValues(-1.0f, 1.0f);

    std::unordered_map<int64_t, void*> variantPack;
    variantPack[10] = aTensor.rawDeviceData();
    variantPack[11] = bTensor.rawDeviceData();
    variantPack[12] = cTensor.rawDeviceData();

    GpuReferenceGraphExecutor gpuExecutor;
    gpuExecutor.execute(graphBuilder.GetBufferPointer(), graphBuilder.GetSize(), variantPack);
    cTensor.markDeviceModified();

    // Validate against CPU reference implementation
    hipdnn_data_sdk::utilities::Tensor<float> refCTensor(cDims, cStrides);
    hipdnn_test_sdk::utilities::CpuFpReferenceMatmul::matmul(aTensor, bTensor, refCTensor);
    refCTensor.markHostModified();

    auto* cData = static_cast<float*>(cTensor.rawHostData());
    auto* refCData = static_cast<float*>(refCTensor.rawHostData());
    for(size_t i = 0; i < cTensor.elementCount(); ++i)
    {
        EXPECT_NEAR(
            cData[i], refCData[i], hipdnn_test_sdk::utilities::matmul::getTolerance<float>())
            << "Mismatch in c at index " << i;
    }
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

TEST(TestGpuReferenceGraphExecutor, RMSNormFwdIsApplicable)
{
    SKIP_IF_NO_DEVICES();

    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormGraph();

    GpuReferenceGraphExecutor executor;
    EXPECT_TRUE(executor.isApplicable(builder.GetBufferPointer(), builder.GetSize()));
}

TEST(TestGpuReferenceGraphExecutorFp32, RMSNormFwdExecutes)
{
    SKIP_IF_NO_DEVICES();

    const std::vector<int64_t> dims = {2, 3, 4, 4};
    auto strides = generateStrides(dims);

    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormGraph(strides, dims);

    std::vector<int64_t> scaleDims(dims);
    scaleDims[0] = 1;
    auto scaleStrides = hipdnn_data_sdk::utilities::generateStrides(
        scaleDims, hipdnn_data_sdk::utilities::extractStrideOrder(strides));

    hipdnn_data_sdk::utilities::Tensor<float> xTensor(dims, strides);
    hipdnn_data_sdk::utilities::Tensor<float> yTensor(dims, strides);
    hipdnn_data_sdk::utilities::Tensor<float> scaleTensor(scaleDims, scaleStrides);

    xTensor.fillWithRandomValues(-1.0f, 1.0f);
    scaleTensor.fillWithRandomValues(0.5f, 1.5f);
    yTensor.fillWithValue(0);

    std::unordered_map<int64_t, void*> variantPack;
    variantPack[1] = xTensor.rawDeviceData();
    variantPack[2] = yTensor.rawDeviceData();
    variantPack[3] = scaleTensor.rawDeviceData();

    GpuReferenceGraphExecutor gpuExecutor;
    gpuExecutor.execute(builder.GetBufferPointer(), builder.GetSize(), variantPack);
    yTensor.markDeviceModified();

    // Validate against CPU reference implementation
    hipdnn_data_sdk::utilities::Tensor<float> refYTensor(dims, strides);
    hipdnn_test_sdk::utilities::CpuFpReferenceRMSNorm::forward(
        xTensor, scaleTensor, refYTensor, 1e-5);

    auto* yHost = static_cast<float*>(yTensor.rawHostData());
    auto* refYHost = static_cast<float*>(refYTensor.rawHostData());
    for(size_t i = 0; i < yTensor.elementCount(); ++i)
    {
        EXPECT_NEAR(
            yHost[i], refYHost[i], hipdnn_test_sdk::utilities::rmsnorm::getTolerance<float>())
            << "Mismatch at index " << i;
    }
}

TEST(TestGpuReferenceGraphExecutor, RMSNormBwdIsApplicable)
{
    SKIP_IF_NO_DEVICES();

    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormBwdGraph();

    GpuReferenceGraphExecutor executor;
    EXPECT_TRUE(executor.isApplicable(builder.GetBufferPointer(), builder.GetSize()));
}

TEST(TestGpuReferenceGraphExecutorFp32, RMSNormBwdExecutes)
{
    SKIP_IF_NO_DEVICES();

    const std::vector<int64_t> dims = {2, 3, 4, 4};
    auto strides = generateStrides(dims);

    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormBwdGraph(strides, dims, true);

    std::vector<int64_t> scaleDims(dims);
    scaleDims[0] = 1;
    auto scaleStrides = hipdnn_data_sdk::utilities::generateStrides(
        scaleDims, hipdnn_data_sdk::utilities::extractStrideOrder(strides));

    std::vector<int64_t> statDims(dims.size(), 1);
    statDims[0] = dims[0];
    auto statStrides = hipdnn_data_sdk::utilities::generateStrides(
        statDims, hipdnn_data_sdk::utilities::extractStrideOrder(strides));

    hipdnn_data_sdk::utilities::Tensor<float> dyTensor(dims, strides);
    hipdnn_data_sdk::utilities::Tensor<float> xTensor(dims, strides);
    hipdnn_data_sdk::utilities::Tensor<float> scaleTensor(scaleDims, scaleStrides);
    hipdnn_data_sdk::utilities::Tensor<float> dxTensor(dims, strides);
    hipdnn_data_sdk::utilities::Tensor<float> dscaleTensor(scaleDims, scaleStrides);
    hipdnn_data_sdk::utilities::Tensor<float> dbiasTensor(scaleDims, scaleStrides);
    hipdnn_data_sdk::utilities::Tensor<float> invRmsTensor(statDims, statStrides);

    dyTensor.fillWithRandomValues(-1.0f, 1.0f);
    xTensor.fillWithRandomValues(-1.0f, 1.0f);
    scaleTensor.fillWithRandomValues(0.5f, 1.5f);
    invRmsTensor.fillWithRandomValues(0.1f, 1.0f);
    dxTensor.fillWithValue(0);
    dscaleTensor.fillWithValue(0);
    dbiasTensor.fillWithValue(0);

    std::unordered_map<int64_t, void*> variantPack;
    variantPack[1] = dyTensor.rawDeviceData();
    variantPack[2] = xTensor.rawDeviceData();
    variantPack[3] = scaleTensor.rawDeviceData();
    variantPack[4] = dxTensor.rawDeviceData();
    variantPack[5] = dscaleTensor.rawDeviceData();
    variantPack[6] = invRmsTensor.rawDeviceData();
    variantPack[7] = dbiasTensor.rawDeviceData();

    GpuReferenceGraphExecutor gpuExecutor;
    gpuExecutor.execute(builder.GetBufferPointer(), builder.GetSize(), variantPack);
    dxTensor.markDeviceModified();
    dscaleTensor.markDeviceModified();
    dbiasTensor.markDeviceModified();

    // Validate against CPU reference implementation
    hipdnn_data_sdk::utilities::Tensor<float> refDxTensor(dims, strides);
    hipdnn_data_sdk::utilities::Tensor<float> refDscaleTensor(scaleDims, scaleStrides);
    hipdnn_data_sdk::utilities::Tensor<float> refDbiasTensor(scaleDims, scaleStrides);
    hipdnn_test_sdk::utilities::CpuFpReferenceRMSNorm::backward(dyTensor,
                                                                xTensor,
                                                                scaleTensor,
                                                                invRmsTensor,
                                                                refDxTensor,
                                                                refDscaleTensor,
                                                                &refDbiasTensor);

    auto* dxHost = static_cast<float*>(dxTensor.rawHostData());
    auto* refDxHost = static_cast<float*>(refDxTensor.rawHostData());
    const auto tolerance = hipdnn_test_sdk::utilities::rmsnorm::getTolerance<float>();
    for(size_t i = 0; i < dxTensor.elementCount(); ++i)
    {
        EXPECT_NEAR(dxHost[i], refDxHost[i], tolerance) << "dx mismatch at index " << i;
    }

    auto* dscaleHost = static_cast<float*>(dscaleTensor.rawHostData());
    auto* refDscaleHost = static_cast<float*>(refDscaleTensor.rawHostData());
    auto* dbiasHost = static_cast<float*>(dbiasTensor.rawHostData());
    auto* refDbiasHost = static_cast<float*>(refDbiasTensor.rawHostData());
    for(size_t i = 0; i < dscaleTensor.elementCount(); ++i)
    {
        EXPECT_NEAR(dscaleHost[i], refDscaleHost[i], tolerance) << "dscale mismatch at index " << i;
        EXPECT_NEAR(dbiasHost[i], refDbiasHost[i], tolerance) << "dbias mismatch at index " << i;
    }
}
