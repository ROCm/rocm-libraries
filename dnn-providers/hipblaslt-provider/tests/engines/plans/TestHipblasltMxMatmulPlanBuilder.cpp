// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipblaslt/hipblaslt.h>
#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_test_sdk/utilities/DeviceQuery.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/MockGraph.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "HipdnnEnginePluginHandle.hpp"
#include "engines/plans/HipblasltMxMatmulPlanBuilder.hpp"
#include "engines/plans/MxArchSupport.hpp"

using namespace hipblaslt_plugin;
using namespace hipdnn_plugin_sdk;
using namespace hipdnn_flatbuffers_sdk::flatbuffer_utilities;
using namespace hipdnn_test_sdk::utilities;

using DT = hipdnn_flatbuffers_sdk::data_objects::DataType;

namespace
{

enum class MxTensor
{
    XA, // A input — should be non-virtual
    SCALE_A, // A scale — should be non-virtual
    YA, // A dequant output — should be virtual
    XB, // B input — should be non-virtual
    SCALE_B, // B scale — should be non-virtual
    YB, // B dequant output — should be virtual
    C, // matmul output — should be non-virtual
};

// Builds an otherwise-valid canonical MX graph with the given tensor's is_virtual
// flag flipped from its required value, to exercise the builder's virtuality checks.
flatbuffers::FlatBufferBuilder createMxGraphWithWrongVirtual(MxTensor target)
{
    // Returns the is_virtual flag for a tensor, flipped from its valid value when it
    // is the corruption target.
    const auto flag
        = [target](MxTensor self, bool valid) { return (self == target) ? !valid : valid; };

    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>>
        tensors;

    const std::vector<int64_t> xADims = {32, 128};
    const std::vector<int64_t> xAStrides = {1, 32}; // col-major (opA=T)
    const std::vector<int64_t> scaleADims = {32, 4};
    const std::vector<int64_t> scaleAStrides = {4, 1};
    const std::vector<int64_t> yADims = {32, 128};
    const std::vector<int64_t> yAStrides = {128, 1};
    const std::vector<int64_t> xBDims = {128, 32};
    const std::vector<int64_t> xBStrides = {32, 1}; // row-major (opB=N)
    const std::vector<int64_t> scaleBDims = {4, 32};
    const std::vector<int64_t> scaleBStrides = {32, 1};
    const std::vector<int64_t> yBDims = {128, 32};
    const std::vector<int64_t> yBStrides = {32, 1};
    const std::vector<int64_t> cDims = {32, 32};
    const std::vector<int64_t> cStrides = {32, 1};

    const int64_t xAUid = 1;
    const int64_t scaleAUid = 2;
    const int64_t yAUid = 3;
    const int64_t xBUid = 4;
    const int64_t scaleBUid = 5;
    const int64_t yBUid = 6;
    const int64_t cUid = 7;

    tensors.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, xAUid, "x_a", DT::FP8_E4M3, &xAStrides, &xADims, flag(MxTensor::XA, false)));
    tensors.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        scaleAUid,
        "scale_a",
        DT::FP8_E8M0,
        &scaleAStrides,
        &scaleADims,
        flag(MxTensor::SCALE_A, false)));
    tensors.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, yAUid, "y_a", DT::FLOAT, &yAStrides, &yADims, flag(MxTensor::YA, true)));
    tensors.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, xBUid, "x_b", DT::FP8_E4M3, &xBStrides, &xBDims, flag(MxTensor::XB, false)));
    tensors.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        scaleBUid,
        "scale_b",
        DT::FP8_E8M0,
        &scaleBStrides,
        &scaleBDims,
        flag(MxTensor::SCALE_B, false)));
    tensors.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, yBUid, "y_b", DT::FLOAT, &yBStrides, &yBDims, flag(MxTensor::YB, true)));
    tensors.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, cUid, "c", DT::HALF, &cStrides, &cDims, flag(MxTensor::C, false)));

    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Node>> nodes;
    const std::vector<int32_t> blockSizeVec = {32};

    auto deqAttrA
        = hipdnn_flatbuffers_sdk::data_objects::CreateBlockScaleDequantizeAttributesDirect(
            builder, xAUid, scaleAUid, yAUid, &blockSizeVec, false);
    nodes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "deq_a",
        DT::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::BlockScaleDequantizeAttributes,
        deqAttrA.Union()));

    auto deqAttrB
        = hipdnn_flatbuffers_sdk::data_objects::CreateBlockScaleDequantizeAttributesDirect(
            builder, xBUid, scaleBUid, yBUid, &blockSizeVec, false);
    nodes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "deq_b",
        DT::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::BlockScaleDequantizeAttributes,
        deqAttrB.Union()));

    auto matmulAttr
        = hipdnn_flatbuffers_sdk::data_objects::CreateMatmulAttributes(builder, yAUid, yBUid, cUid);
    nodes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "matmul",
        DT::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::MatmulAttributes,
        matmulAttr.Union()));

    auto graphOffset = hipdnn_flatbuffers_sdk::data_objects::CreateGraphDirect(
        builder, "mx_virtual_test", DT::FLOAT, DT::HALF, DT::BFLOAT16, &tensors, &nodes);
    builder.Finish(graphOffset);
    return builder;
}

} // namespace

// ===========================================================================
// Fixtures
// ===========================================================================

// CPU fixture: negative cases that reject the graph before any GPU handle use.
// No device required — the handle is default-constructed and never touched
// because every rejection happens during topology/constraint checks.
class TestHipblasltMxMatmulPlanBuilder : public ::testing::Test
{
protected:
    HipblasltMxMatmulPlanBuilder _builder;
    HipdnnEnginePluginHandle _handle{};
};

// GPU fixture: positive cases that construct a real plan via the hipBLASLt
// handle. Gated on a device + a live hipblasLt handle.
class TestGpuHipblasltMxMatmulPlanBuilder : public ::testing::Test
{
protected:
    void SetUp() override
    {
        SKIP_IF_NO_DEVICES();
        const auto archName = hipdnn_test_sdk::utilities::currentDeviceArch();
        if(!hipblaslt_plugin_test::isMxSupportedArch(archName))
        {
            GTEST_SKIP() << "MX block-scaled GEMM is not supported on " << archName;
        }
        ASSERT_EQ(hipblasLtCreate(&_handle.hipblasltHandle), HIPBLAS_STATUS_SUCCESS);
    }

    void TearDown() override
    {
        if(_handle.hipblasltHandle != nullptr)
        {
            EXPECT_EQ(hipblasLtDestroy(_handle.hipblasltHandle), HIPBLAS_STATUS_SUCCESS);
        }
    }

    HipblasltMxMatmulPlanBuilder _builder;
    HipdnnEnginePluginHandle _handle;
};

// ===========================================================================
// isApplicable — positive cases (GPU-gated via SKIP_IF_NO_DEVICES in SetUp)
// ===========================================================================

TEST_F(TestGpuHipblasltMxMatmulPlanBuilder, IsApplicableE4M3OutputHalf)
{
    auto fb = createValidMxMatmulGraph();
    GraphWrapper const graph(fb.GetBufferPointer(), fb.GetSize());
    EXPECT_TRUE(_builder.isApplicable(_handle, graph));
}

TEST_F(TestGpuHipblasltMxMatmulPlanBuilder, IsApplicableE5M2OutputBf16)
{
    auto fb = createValidMxMatmulGraph({32, 128},
                                       {1, 32},
                                       {128, 32},
                                       {32, 1},
                                       {32, 32},
                                       {32, 1},
                                       {32, 4},
                                       {4, 32},
                                       DT::FP8_E5M2,
                                       DT::BFLOAT16);
    GraphWrapper const graph(fb.GetBufferPointer(), fb.GetSize());
    EXPECT_TRUE(_builder.isApplicable(_handle, graph));
}

TEST_F(TestGpuHipblasltMxMatmulPlanBuilder, IsApplicableE4M3OutputFp32)
{
    auto fb = createValidMxMatmulGraph({32, 128},
                                       {1, 32},
                                       {128, 32},
                                       {32, 1},
                                       {32, 32},
                                       {32, 1},
                                       {32, 4},
                                       {4, 32},
                                       DT::FP8_E4M3,
                                       DT::FLOAT);
    GraphWrapper const graph(fb.GetBufferPointer(), fb.GetSize());
    EXPECT_TRUE(_builder.isApplicable(_handle, graph));
}

// Dequant nodes emitted in B-then-A order must still be recognized: A/B are
// resolved by matching dequant Y outputs to matmul inputs, not by node position.
TEST_F(TestGpuHipblasltMxMatmulPlanBuilder, IsApplicableHandlesSwappedDequantOrder)
{
    auto fb = createValidMxMatmulGraph({32, 128},
                                       {1, 32},
                                       {128, 32},
                                       {32, 1},
                                       {32, 32},
                                       {32, 1},
                                       {32, 4},
                                       {4, 32},
                                       DT::FP8_E4M3,
                                       DT::HALF,
                                       DT::FP8_E8M0,
                                       DT::FP8_E8M0,
                                       DT::FLOAT,
                                       32,
                                       false,
                                       true /*swapDequantOrder*/);
    GraphWrapper const graph(fb.GetBufferPointer(), fb.GetSize());
    EXPECT_TRUE(_builder.isApplicable(_handle, graph));
}

// ===========================================================================
// isApplicable — negative cases (all CPU-safe, no GPU call needed)
// ===========================================================================

// Plain matmul (1 node) must NOT match the MX builder
TEST_F(TestHipblasltMxMatmulPlanBuilder, IsApplicableRejectsPlainMatmul)
{
    auto fb = createValidMatmulGraph();
    GraphWrapper const graph(fb.GetBufferPointer(), fb.GetSize());
    EXPECT_FALSE(_builder.isApplicable(_handle, graph));
}

// Wrong node count (not 3)
TEST_F(TestHipblasltMxMatmulPlanBuilder, IsApplicableRejectsWrongNodeCount)
{
    MockGraph const mockGraph;
    EXPECT_CALL(mockGraph, nodeCount()).WillRepeatedly(::testing::Return(1));
    EXPECT_FALSE(_builder.isApplicable(_handle, mockGraph));
}

TEST_F(TestHipblasltMxMatmulPlanBuilder, IsApplicableRejectsWrongNodeCount4)
{
    MockGraph const mockGraph;
    EXPECT_CALL(mockGraph, nodeCount()).WillRepeatedly(::testing::Return(4));
    EXPECT_FALSE(_builder.isApplicable(_handle, mockGraph));
}

// Non-FP8 input types
TEST_F(TestHipblasltMxMatmulPlanBuilder, IsApplicableRejectsNonFp8Input)
{
    auto fb = createValidMxMatmulGraph({32, 128},
                                       {1, 32},
                                       {128, 32},
                                       {32, 1},
                                       {32, 32},
                                       {32, 1},
                                       {32, 4},
                                       {4, 32},
                                       DT::HALF,
                                       DT::HALF);
    GraphWrapper const graph(fb.GetBufferPointer(), fb.GetSize());
    EXPECT_FALSE(_builder.isApplicable(_handle, graph));
}

TEST_F(TestHipblasltMxMatmulPlanBuilder, IsApplicableRejectsNonFp8InputFloat)
{
    auto fb = createValidMxMatmulGraph({32, 128},
                                       {1, 32},
                                       {128, 32},
                                       {32, 1},
                                       {32, 32},
                                       {32, 1},
                                       {32, 4},
                                       {4, 32},
                                       DT::FLOAT,
                                       DT::HALF);
    GraphWrapper const graph(fb.GetBufferPointer(), fb.GetSize());
    EXPECT_FALSE(_builder.isApplicable(_handle, graph));
}

// FP8 output type not allowed
TEST_F(TestHipblasltMxMatmulPlanBuilder, IsApplicableRejectsFp8Output)
{
    auto fb = createValidMxMatmulGraph({32, 128},
                                       {1, 32},
                                       {128, 32},
                                       {32, 1},
                                       {32, 32},
                                       {32, 1},
                                       {32, 4},
                                       {4, 32},
                                       DT::FP8_E4M3,
                                       DT::FP8_E4M3);
    GraphWrapper const graph(fb.GetBufferPointer(), fb.GetSize());
    EXPECT_FALSE(_builder.isApplicable(_handle, graph));
}

TEST_F(TestHipblasltMxMatmulPlanBuilder, IsApplicableRejectsFp8E8M0Output)
{
    auto fb = createValidMxMatmulGraph({32, 128},
                                       {1, 32},
                                       {128, 32},
                                       {32, 1},
                                       {32, 32},
                                       {32, 1},
                                       {32, 4},
                                       {4, 32},
                                       DT::FP8_E4M3,
                                       DT::FP8_E8M0);
    GraphWrapper const graph(fb.GetBufferPointer(), fb.GetSize());
    EXPECT_FALSE(_builder.isApplicable(_handle, graph));
}

// m not divisible by 16
TEST_F(TestHipblasltMxMatmulPlanBuilder, IsApplicableRejectsMNotDiv16)
{
    auto fb = createValidMxMatmulGraph(
        {33, 128}, {1, 33}, {128, 32}, {32, 1}, {33, 32}, {32, 1}, {33, 4}, {4, 32});
    GraphWrapper const graph(fb.GetBufferPointer(), fb.GetSize());
    EXPECT_FALSE(_builder.isApplicable(_handle, graph));
}

// n not divisible by 16
TEST_F(TestHipblasltMxMatmulPlanBuilder, IsApplicableRejectsNNotDiv16)
{
    auto fb = createValidMxMatmulGraph(
        {32, 128}, {1, 32}, {128, 33}, {33, 1}, {32, 33}, {33, 1}, {32, 4}, {4, 33});
    GraphWrapper const graph(fb.GetBufferPointer(), fb.GetSize());
    EXPECT_FALSE(_builder.isApplicable(_handle, graph));
}

// K not divisible by 128 (96 is block-aligned at 32 but not 128-aligned)
TEST_F(TestHipblasltMxMatmulPlanBuilder, IsApplicableRejectsKNotDiv128)
{
    auto fb = createValidMxMatmulGraph(
        {32, 96}, {1, 32}, {96, 32}, {32, 1}, {32, 32}, {32, 1}, {32, 3}, {3, 32});
    GraphWrapper const graph(fb.GetBufferPointer(), fb.GetSize());
    EXPECT_FALSE(_builder.isApplicable(_handle, graph));
}

// block_size != 32
TEST_F(TestHipblasltMxMatmulPlanBuilder, IsApplicableRejectsWrongBlockSize)
{
    auto fb = createValidMxMatmulGraph({32, 128},
                                       {1, 32},
                                       {128, 32},
                                       {32, 1},
                                       {32, 32},
                                       {32, 1},
                                       {32, 4},
                                       {4, 32},
                                       DT::FP8_E4M3,
                                       DT::HALF,
                                       DT::FP8_E8M0,
                                       DT::FP8_E8M0,
                                       DT::FLOAT,
                                       16 /*blockSize*/);
    GraphWrapper const graph(fb.GetBufferPointer(), fb.GetSize());
    EXPECT_FALSE(_builder.isApplicable(_handle, graph));
}

// opA != T (A must be col-major for MX) — pass row-major A strides
TEST_F(TestHipblasltMxMatmulPlanBuilder, IsApplicableRejectsOpANotT)
{
    auto fb = createValidMxMatmulGraph(
        {32, 128}, {128, 1}, {128, 32}, {32, 1}, {32, 32}, {32, 1}, {32, 4}, {4, 32});
    GraphWrapper const graph(fb.GetBufferPointer(), fb.GetSize());
    EXPECT_FALSE(_builder.isApplicable(_handle, graph));
}

// opB != N (B must be row-major for MX) — pass col-major B strides
TEST_F(TestHipblasltMxMatmulPlanBuilder, IsApplicableRejectsOpBNotN)
{
    auto fb = createValidMxMatmulGraph(
        {32, 128}, {1, 32}, {128, 32}, {1, 128}, {32, 32}, {32, 1}, {32, 4}, {4, 32});
    GraphWrapper const graph(fb.GetBufferPointer(), fb.GetSize());
    EXPECT_FALSE(_builder.isApplicable(_handle, graph));
}

// Non-MX graph (batchnorm) must return false
TEST_F(TestHipblasltMxMatmulPlanBuilder, IsApplicableRejectsBatchnormGraph)
{
    auto fb = createValidBatchnormInferenceGraph();
    GraphWrapper const graph(fb.GetBufferPointer(), fb.GetSize());
    EXPECT_FALSE(_builder.isApplicable(_handle, graph));
}

// batch > 1 (hipBLASLt requires B==1 for VEC32_UE8M0) — leading batch dim of 2 must
// be rejected. All tensors carry a rank-3 leading batch dim.
TEST_F(TestHipblasltMxMatmulPlanBuilder, IsApplicableRejectsBatchGreaterThan1)
{
    auto fb = createValidMxMatmulGraph({2, 32, 128},
                                       {4096, 1, 32},
                                       {2, 128, 32},
                                       {4096, 32, 1},
                                       {2, 32, 32},
                                       {1024, 32, 1},
                                       {2, 32, 4},
                                       {2, 4, 32});
    GraphWrapper const graph(fb.GetBufferPointer(), fb.GetSize());
    EXPECT_FALSE(_builder.isApplicable(_handle, graph));
}

// Fused epilogue (4th Pointwise node) is not allowed — extra node → not 3 nodes
TEST_F(TestHipblasltMxMatmulPlanBuilder, IsApplicableRejectsEpilogue)
{
    auto fb = createValidMxMatmulGraph({32, 128},
                                       {1, 32},
                                       {128, 32},
                                       {32, 1},
                                       {32, 32},
                                       {32, 1},
                                       {32, 4},
                                       {4, 32},
                                       DT::FP8_E4M3,
                                       DT::HALF,
                                       DT::FP8_E8M0,
                                       DT::FP8_E8M0,
                                       DT::FLOAT,
                                       32,
                                       true /*withEpilogue*/);
    GraphWrapper const graph(fb.GetBufferPointer(), fb.GetSize());
    EXPECT_FALSE(_builder.isApplicable(_handle, graph));
}

// Matmul node compute data type must be FP32 (mirrors the plain matmul builder)
TEST_F(TestHipblasltMxMatmulPlanBuilder, IsApplicableRejectsNonFloatComputeType)
{
    auto fb = createValidMxMatmulGraph({32, 128},
                                       {1, 32},
                                       {128, 32},
                                       {32, 1},
                                       {32, 32},
                                       {32, 1},
                                       {32, 4},
                                       {4, 32},
                                       DT::FP8_E4M3,
                                       DT::HALF,
                                       DT::FP8_E8M0,
                                       DT::FP8_E8M0,
                                       DT::HALF /*computeType*/);
    GraphWrapper const graph(fb.GetBufferPointer(), fb.GetSize());
    EXPECT_FALSE(_builder.isApplicable(_handle, graph));
}

// Input shapes must be consistent with the output: C's N-dim (48) differs from
// B's N-dim (32) here. 48 is 16-aligned, so this fails the shape check, not the
// hipBLASLt alignment check.
TEST_F(TestHipblasltMxMatmulPlanBuilder, IsApplicableRejectsShapeMismatch)
{
    auto fb = createValidMxMatmulGraph(
        {32, 128}, {1, 32}, {128, 32}, {32, 1}, {32, 48}, {48, 1}, {32, 4}, {4, 32});
    GraphWrapper const graph(fb.GetBufferPointer(), fb.GetSize());
    EXPECT_FALSE(_builder.isApplicable(_handle, graph));
}

// Virtual/non-virtual contract: the inputs, their scales, and C are real device
// buffers (must be non-virtual); the dequant Y outputs are fused intermediates
// (must be virtual). Each case flips exactly one flag.
TEST_F(TestHipblasltMxMatmulPlanBuilder, IsApplicableRejectsVirtualFp8InputA)
{
    auto fb = createMxGraphWithWrongVirtual(MxTensor::XA);
    GraphWrapper const graph(fb.GetBufferPointer(), fb.GetSize());
    EXPECT_FALSE(_builder.isApplicable(_handle, graph));
}

TEST_F(TestHipblasltMxMatmulPlanBuilder, IsApplicableRejectsVirtualFp8InputB)
{
    auto fb = createMxGraphWithWrongVirtual(MxTensor::XB);
    GraphWrapper const graph(fb.GetBufferPointer(), fb.GetSize());
    EXPECT_FALSE(_builder.isApplicable(_handle, graph));
}

TEST_F(TestHipblasltMxMatmulPlanBuilder, IsApplicableRejectsVirtualScaleA)
{
    auto fb = createMxGraphWithWrongVirtual(MxTensor::SCALE_A);
    GraphWrapper const graph(fb.GetBufferPointer(), fb.GetSize());
    EXPECT_FALSE(_builder.isApplicable(_handle, graph));
}

TEST_F(TestHipblasltMxMatmulPlanBuilder, IsApplicableRejectsVirtualScaleB)
{
    auto fb = createMxGraphWithWrongVirtual(MxTensor::SCALE_B);
    GraphWrapper const graph(fb.GetBufferPointer(), fb.GetSize());
    EXPECT_FALSE(_builder.isApplicable(_handle, graph));
}

TEST_F(TestHipblasltMxMatmulPlanBuilder, IsApplicableRejectsNonVirtualDequantOutputA)
{
    auto fb = createMxGraphWithWrongVirtual(MxTensor::YA);
    GraphWrapper const graph(fb.GetBufferPointer(), fb.GetSize());
    EXPECT_FALSE(_builder.isApplicable(_handle, graph));
}

TEST_F(TestHipblasltMxMatmulPlanBuilder, IsApplicableRejectsNonVirtualDequantOutputB)
{
    auto fb = createMxGraphWithWrongVirtual(MxTensor::YB);
    GraphWrapper const graph(fb.GetBufferPointer(), fb.GetSize());
    EXPECT_FALSE(_builder.isApplicable(_handle, graph));
}

TEST_F(TestHipblasltMxMatmulPlanBuilder, IsApplicableRejectsVirtualOutputC)
{
    auto fb = createMxGraphWithWrongVirtual(MxTensor::C);
    GraphWrapper const graph(fb.GetBufferPointer(), fb.GetSize());
    EXPECT_FALSE(_builder.isApplicable(_handle, graph));
}

// Scale-tensor contract: VEC32_UE8M0 requires the scales to be FP8_E8M0 and to
// declare exactly one element per 32-element operand block (M*(K/32) for A,
// (K/32)*N for B). Each case corrupts exactly one scale's dtype or shape.
TEST_F(TestHipblasltMxMatmulPlanBuilder, IsApplicableRejectsWrongScaleTypeA)
{
    auto fb = createValidMxMatmulGraph({32, 128},
                                       {1, 32},
                                       {128, 32},
                                       {32, 1},
                                       {32, 32},
                                       {32, 1},
                                       {32, 4},
                                       {4, 32},
                                       DT::FP8_E4M3,
                                       DT::HALF,
                                       DT::FLOAT /*scaleAType*/);
    GraphWrapper const graph(fb.GetBufferPointer(), fb.GetSize());
    EXPECT_FALSE(_builder.isApplicable(_handle, graph));
}

TEST_F(TestHipblasltMxMatmulPlanBuilder, IsApplicableRejectsWrongScaleTypeB)
{
    auto fb = createValidMxMatmulGraph({32, 128},
                                       {1, 32},
                                       {128, 32},
                                       {32, 1},
                                       {32, 32},
                                       {32, 1},
                                       {32, 4},
                                       {4, 32},
                                       DT::FP8_E4M3,
                                       DT::HALF,
                                       DT::FP8_E8M0,
                                       DT::FLOAT /*scaleBType*/);
    GraphWrapper const graph(fb.GetBufferPointer(), fb.GetSize());
    EXPECT_FALSE(_builder.isApplicable(_handle, graph));
}

TEST_F(TestHipblasltMxMatmulPlanBuilder, IsApplicableRejectsWrongScaleShapeA)
{
    // scale_a inner dim 5 instead of K/32 = 4
    auto fb = createValidMxMatmulGraph(
        {32, 128}, {1, 32}, {128, 32}, {32, 1}, {32, 32}, {32, 1}, {32, 5}, {4, 32});
    GraphWrapper const graph(fb.GetBufferPointer(), fb.GetSize());
    EXPECT_FALSE(_builder.isApplicable(_handle, graph));
}

TEST_F(TestHipblasltMxMatmulPlanBuilder, IsApplicableRejectsWrongScaleShapeB)
{
    // scale_b outer dim 5 instead of K/32 = 4
    auto fb = createValidMxMatmulGraph(
        {32, 128}, {1, 32}, {128, 32}, {32, 1}, {32, 32}, {32, 1}, {32, 4}, {5, 32});
    GraphWrapper const graph(fb.GetBufferPointer(), fb.GetSize());
    EXPECT_FALSE(_builder.isApplicable(_handle, graph));
}

// A scale with the correct total element count but the K axis split on the wrong
// axis ([K/32, M] instead of [M, K/32]) must be rejected. A total-count-only check
// would accept this and then compute silently-wrong results on the GPU.
TEST_F(TestHipblasltMxMatmulPlanBuilder, IsApplicableRejectsTransposedScaleA)
{
    auto fb = createValidMxMatmulGraph(
        {32, 128}, {1, 32}, {128, 32}, {32, 1}, {32, 32}, {32, 1}, {4, 32}, {4, 32});
    GraphWrapper const graph(fb.GetBufferPointer(), fb.GetSize());
    EXPECT_FALSE(_builder.isApplicable(_handle, graph));
}

TEST_F(TestHipblasltMxMatmulPlanBuilder, IsApplicableRejectsTransposedScaleB)
{
    // scale_b as [N, K/32] instead of [K/32, N]
    auto fb = createValidMxMatmulGraph(
        {32, 128}, {1, 32}, {128, 32}, {32, 1}, {32, 32}, {32, 1}, {32, 4}, {32, 4});
    GraphWrapper const graph(fb.GetBufferPointer(), fb.GetSize());
    EXPECT_FALSE(_builder.isApplicable(_handle, graph));
}

// ===========================================================================
// getWorkspaceSize
// ===========================================================================

TEST_F(TestGpuHipblasltMxMatmulPlanBuilder, GetWorkspaceSizeValid)
{
    auto fb = createValidMxMatmulGraph();
    GraphWrapper const graph(fb.GetBufferPointer(), fb.GetSize());
    EXPECT_NO_THROW(_builder.getWorkspaceSize(_handle, graph));
}

TEST_F(TestHipblasltMxMatmulPlanBuilder, GetWorkspaceSizeThrowsOnInvalidGraph)
{
    auto fb = createValidMatmulGraph();
    GraphWrapper const graph(fb.GetBufferPointer(), fb.GetSize());
    EXPECT_THROW(_builder.getWorkspaceSize(_handle, graph), HipdnnPluginException);
}

// ===========================================================================
// buildPlan
// ===========================================================================

TEST_F(TestGpuHipblasltMxMatmulPlanBuilder, BuildPlanValid)
{
    auto fb = createValidMxMatmulGraph();
    GraphWrapper const graph(fb.GetBufferPointer(), fb.GetSize());
    HipdnnEnginePluginExecutionContext ctx;
    EXPECT_NO_THROW(_builder.buildPlan(_handle, graph, ctx));
    EXPECT_TRUE(ctx.hasValidPlan());
}

TEST_F(TestHipblasltMxMatmulPlanBuilder, BuildPlanThrowsOnInvalidGraph)
{
    auto fb = createValidMatmulGraph();
    GraphWrapper const graph(fb.GetBufferPointer(), fb.GetSize());
    HipdnnEnginePluginExecutionContext ctx;
    EXPECT_THROW(_builder.buildPlan(_handle, graph, ctx), HipdnnPluginException);
    EXPECT_FALSE(ctx.hasValidPlan());
}
