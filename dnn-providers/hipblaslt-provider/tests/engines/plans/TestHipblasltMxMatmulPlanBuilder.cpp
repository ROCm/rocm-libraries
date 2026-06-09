// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipblaslt/hipblaslt.h>
#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/MockGraph.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "HipdnnEnginePluginHandle.hpp"
#include "engines/plans/HipblasltMxMatmulPlanBuilder.hpp"
#include "engines/plans/MxMatmulGraphTestUtils.hpp"

using namespace hipblaslt_plugin;
using namespace hipdnn_plugin_sdk;
using namespace hipdnn_flatbuffers_sdk::flatbuffer_utilities;
using namespace hipdnn_test_sdk::utilities;

using DT = hipdnn_flatbuffers_sdk::data_objects::DataType;
using hipblaslt_plugin::test::createMxMatmulGraph;
using hipblaslt_plugin::test::ScaleOverride;
using hipblaslt_plugin::test::VirtualOverride;

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
    auto fb = createMxMatmulGraph(32, 128, 32, DT::FP8_E4M3, DT::HALF);
    GraphWrapper const graph(fb.GetBufferPointer(), fb.GetSize());
    EXPECT_TRUE(_builder.isApplicable(_handle, graph));
}

TEST_F(TestGpuHipblasltMxMatmulPlanBuilder, IsApplicableE5M2OutputBf16)
{
    auto fb = createMxMatmulGraph(32, 128, 32, DT::FP8_E5M2, DT::BFLOAT16);
    GraphWrapper const graph(fb.GetBufferPointer(), fb.GetSize());
    EXPECT_TRUE(_builder.isApplicable(_handle, graph));
}

TEST_F(TestGpuHipblasltMxMatmulPlanBuilder, IsApplicableE4M3OutputFp32)
{
    auto fb = createMxMatmulGraph(32, 128, 32, DT::FP8_E4M3, DT::FLOAT);
    GraphWrapper const graph(fb.GetBufferPointer(), fb.GetSize());
    EXPECT_TRUE(_builder.isApplicable(_handle, graph));
}

// Dequant nodes emitted in B-then-A order must still be recognized: A/B are
// resolved by matching dequant Y outputs to matmul inputs, not by node position.
TEST_F(TestGpuHipblasltMxMatmulPlanBuilder, IsApplicableHandlesSwappedDequantOrder)
{
    auto fb = createMxMatmulGraph(
        32, 128, 32, DT::FP8_E4M3, DT::HALF, 32, true, true, 1, false, true /*swapDequantOrder*/);
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
    auto fb = createMxMatmulGraph(32, 128, 32, DT::HALF, DT::HALF);
    GraphWrapper const graph(fb.GetBufferPointer(), fb.GetSize());
    EXPECT_FALSE(_builder.isApplicable(_handle, graph));
}

TEST_F(TestHipblasltMxMatmulPlanBuilder, IsApplicableRejectsNonFp8InputFloat)
{
    auto fb = createMxMatmulGraph(32, 128, 32, DT::FLOAT, DT::HALF);
    GraphWrapper const graph(fb.GetBufferPointer(), fb.GetSize());
    EXPECT_FALSE(_builder.isApplicable(_handle, graph));
}

// FP8 output type not allowed
TEST_F(TestHipblasltMxMatmulPlanBuilder, IsApplicableRejectsFp8Output)
{
    auto fb = createMxMatmulGraph(32, 128, 32, DT::FP8_E4M3, DT::FP8_E4M3);
    GraphWrapper const graph(fb.GetBufferPointer(), fb.GetSize());
    EXPECT_FALSE(_builder.isApplicable(_handle, graph));
}

TEST_F(TestHipblasltMxMatmulPlanBuilder, IsApplicableRejectsFp8E8M0Output)
{
    auto fb = createMxMatmulGraph(32, 128, 32, DT::FP8_E4M3, DT::FP8_E8M0);
    GraphWrapper const graph(fb.GetBufferPointer(), fb.GetSize());
    EXPECT_FALSE(_builder.isApplicable(_handle, graph));
}

// m not divisible by 16
TEST_F(TestHipblasltMxMatmulPlanBuilder, IsApplicableRejectsMNotDiv16)
{
    auto fb = createMxMatmulGraph(33, 128, 32, DT::FP8_E4M3, DT::HALF);
    GraphWrapper const graph(fb.GetBufferPointer(), fb.GetSize());
    EXPECT_FALSE(_builder.isApplicable(_handle, graph));
}

// n not divisible by 16
TEST_F(TestHipblasltMxMatmulPlanBuilder, IsApplicableRejectsNNotDiv16)
{
    auto fb = createMxMatmulGraph(32, 128, 33, DT::FP8_E4M3, DT::HALF);
    GraphWrapper const graph(fb.GetBufferPointer(), fb.GetSize());
    EXPECT_FALSE(_builder.isApplicable(_handle, graph));
}

// K not divisible by 128 (96 is block-aligned at 32 but not 128-aligned)
TEST_F(TestHipblasltMxMatmulPlanBuilder, IsApplicableRejectsKNotDiv128)
{
    auto fb = createMxMatmulGraph(32, 96, 32, DT::FP8_E4M3, DT::HALF);
    GraphWrapper const graph(fb.GetBufferPointer(), fb.GetSize());
    EXPECT_FALSE(_builder.isApplicable(_handle, graph));
}

// block_size != 32
TEST_F(TestHipblasltMxMatmulPlanBuilder, IsApplicableRejectsWrongBlockSize)
{
    auto fb = createMxMatmulGraph(32, 128, 32, DT::FP8_E4M3, DT::HALF, 16);
    GraphWrapper const graph(fb.GetBufferPointer(), fb.GetSize());
    EXPECT_FALSE(_builder.isApplicable(_handle, graph));
}

// opA != T (A must be col-major for MX)
TEST_F(TestHipblasltMxMatmulPlanBuilder, IsApplicableRejectsOpANotT)
{
    auto fb = createMxMatmulGraph(32, 128, 32, DT::FP8_E4M3, DT::HALF, 32, false, true);
    GraphWrapper const graph(fb.GetBufferPointer(), fb.GetSize());
    EXPECT_FALSE(_builder.isApplicable(_handle, graph));
}

// opB != N (B must be row-major for MX)
TEST_F(TestHipblasltMxMatmulPlanBuilder, IsApplicableRejectsOpBNotN)
{
    auto fb = createMxMatmulGraph(32, 128, 32, DT::FP8_E4M3, DT::HALF, 32, true, false);
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

// batch > 1 (hipBLASLt requires B==1 for VEC32_UE8M0) — leading batch dim of 2 must be rejected
TEST_F(TestHipblasltMxMatmulPlanBuilder, IsApplicableRejectsBatchGreaterThan1)
{
    auto fb = createMxMatmulGraph(32, 128, 32, DT::FP8_E4M3, DT::HALF, 32, true, true, 2);
    GraphWrapper const graph(fb.GetBufferPointer(), fb.GetSize());
    EXPECT_FALSE(_builder.isApplicable(_handle, graph));
}

// Fused epilogue (4th Pointwise node) is not allowed — extra node → not 3 nodes
TEST_F(TestHipblasltMxMatmulPlanBuilder, IsApplicableRejectsEpilogue)
{
    auto fb = createMxMatmulGraph(
        32, 128, 32, DT::FP8_E4M3, DT::HALF, 32, true, true, 1, true /*withEpilogue*/);
    GraphWrapper const graph(fb.GetBufferPointer(), fb.GetSize());
    EXPECT_FALSE(_builder.isApplicable(_handle, graph));
}

// Matmul node compute data type must be FP32 (mirrors the plain matmul builder)
TEST_F(TestHipblasltMxMatmulPlanBuilder, IsApplicableRejectsNonFloatComputeType)
{
    auto fb = createMxMatmulGraph(32,
                                  128,
                                  32,
                                  DT::FP8_E4M3,
                                  DT::HALF,
                                  32,
                                  true,
                                  true,
                                  1,
                                  false,
                                  false,
                                  DT::HALF /*matmulComputeType*/);
    GraphWrapper const graph(fb.GetBufferPointer(), fb.GetSize());
    EXPECT_FALSE(_builder.isApplicable(_handle, graph));
}

// Input shapes must be consistent with the output: D's N-dim (48) differs from
// B's N-dim (32) here. 48 is 16-aligned, so this fails the shape check, not the
// hipBLASLt alignment check.
TEST_F(TestHipblasltMxMatmulPlanBuilder, IsApplicableRejectsShapeMismatch)
{
    auto fb = createMxMatmulGraph(32,
                                  128,
                                  32,
                                  DT::FP8_E4M3,
                                  DT::HALF,
                                  32,
                                  true,
                                  true,
                                  1,
                                  false,
                                  false,
                                  DT::FLOAT,
                                  48 /*dnOverride*/);
    GraphWrapper const graph(fb.GetBufferPointer(), fb.GetSize());
    EXPECT_FALSE(_builder.isApplicable(_handle, graph));
}

// Virtual/non-virtual contract: the FP8 inputs, their scales, and D are real
// device buffers (must be non-virtual); the dequant Y outputs are fused
// intermediates (must be virtual). Each override flips exactly one flag.
TEST_F(TestHipblasltMxMatmulPlanBuilder, IsApplicableRejectsVirtualFp8InputA)
{
    auto fb = createMxMatmulGraph(32,
                                  128,
                                  32,
                                  DT::FP8_E4M3,
                                  DT::HALF,
                                  32,
                                  true,
                                  true,
                                  1,
                                  false,
                                  false,
                                  DT::FLOAT,
                                  0,
                                  VirtualOverride::XA_VIRTUAL);
    GraphWrapper const graph(fb.GetBufferPointer(), fb.GetSize());
    EXPECT_FALSE(_builder.isApplicable(_handle, graph));
}

TEST_F(TestHipblasltMxMatmulPlanBuilder, IsApplicableRejectsVirtualFp8InputB)
{
    auto fb = createMxMatmulGraph(32,
                                  128,
                                  32,
                                  DT::FP8_E4M3,
                                  DT::HALF,
                                  32,
                                  true,
                                  true,
                                  1,
                                  false,
                                  false,
                                  DT::FLOAT,
                                  0,
                                  VirtualOverride::XB_VIRTUAL);
    GraphWrapper const graph(fb.GetBufferPointer(), fb.GetSize());
    EXPECT_FALSE(_builder.isApplicable(_handle, graph));
}

TEST_F(TestHipblasltMxMatmulPlanBuilder, IsApplicableRejectsVirtualScaleA)
{
    auto fb = createMxMatmulGraph(32,
                                  128,
                                  32,
                                  DT::FP8_E4M3,
                                  DT::HALF,
                                  32,
                                  true,
                                  true,
                                  1,
                                  false,
                                  false,
                                  DT::FLOAT,
                                  0,
                                  VirtualOverride::SCALE_A_VIRTUAL);
    GraphWrapper const graph(fb.GetBufferPointer(), fb.GetSize());
    EXPECT_FALSE(_builder.isApplicable(_handle, graph));
}

TEST_F(TestHipblasltMxMatmulPlanBuilder, IsApplicableRejectsVirtualScaleB)
{
    auto fb = createMxMatmulGraph(32,
                                  128,
                                  32,
                                  DT::FP8_E4M3,
                                  DT::HALF,
                                  32,
                                  true,
                                  true,
                                  1,
                                  false,
                                  false,
                                  DT::FLOAT,
                                  0,
                                  VirtualOverride::SCALE_B_VIRTUAL);
    GraphWrapper const graph(fb.GetBufferPointer(), fb.GetSize());
    EXPECT_FALSE(_builder.isApplicable(_handle, graph));
}

TEST_F(TestHipblasltMxMatmulPlanBuilder, IsApplicableRejectsNonVirtualDequantOutputA)
{
    auto fb = createMxMatmulGraph(32,
                                  128,
                                  32,
                                  DT::FP8_E4M3,
                                  DT::HALF,
                                  32,
                                  true,
                                  true,
                                  1,
                                  false,
                                  false,
                                  DT::FLOAT,
                                  0,
                                  VirtualOverride::YA_NON_VIRTUAL);
    GraphWrapper const graph(fb.GetBufferPointer(), fb.GetSize());
    EXPECT_FALSE(_builder.isApplicable(_handle, graph));
}

TEST_F(TestHipblasltMxMatmulPlanBuilder, IsApplicableRejectsNonVirtualDequantOutputB)
{
    auto fb = createMxMatmulGraph(32,
                                  128,
                                  32,
                                  DT::FP8_E4M3,
                                  DT::HALF,
                                  32,
                                  true,
                                  true,
                                  1,
                                  false,
                                  false,
                                  DT::FLOAT,
                                  0,
                                  VirtualOverride::YB_NON_VIRTUAL);
    GraphWrapper const graph(fb.GetBufferPointer(), fb.GetSize());
    EXPECT_FALSE(_builder.isApplicable(_handle, graph));
}

TEST_F(TestHipblasltMxMatmulPlanBuilder, IsApplicableRejectsVirtualOutputD)
{
    auto fb = createMxMatmulGraph(32,
                                  128,
                                  32,
                                  DT::FP8_E4M3,
                                  DT::HALF,
                                  32,
                                  true,
                                  true,
                                  1,
                                  false,
                                  false,
                                  DT::FLOAT,
                                  0,
                                  VirtualOverride::D_VIRTUAL);
    GraphWrapper const graph(fb.GetBufferPointer(), fb.GetSize());
    EXPECT_FALSE(_builder.isApplicable(_handle, graph));
}

// Scale-tensor contract: VEC32_UE8M0 requires the scales to be FP8_E8M0 and to
// declare exactly one element per 32-element operand block (M*(K/32) for A,
// (K/32)*N for B). Each override corrupts exactly one scale's dtype or shape.
TEST_F(TestHipblasltMxMatmulPlanBuilder, IsApplicableRejectsWrongScaleTypeA)
{
    auto fb = createMxMatmulGraph(32,
                                  128,
                                  32,
                                  DT::FP8_E4M3,
                                  DT::HALF,
                                  32,
                                  true,
                                  true,
                                  1,
                                  false,
                                  false,
                                  DT::FLOAT,
                                  0,
                                  VirtualOverride::NONE,
                                  ScaleOverride::A_WRONG_TYPE);
    GraphWrapper const graph(fb.GetBufferPointer(), fb.GetSize());
    EXPECT_FALSE(_builder.isApplicable(_handle, graph));
}

TEST_F(TestHipblasltMxMatmulPlanBuilder, IsApplicableRejectsWrongScaleTypeB)
{
    auto fb = createMxMatmulGraph(32,
                                  128,
                                  32,
                                  DT::FP8_E4M3,
                                  DT::HALF,
                                  32,
                                  true,
                                  true,
                                  1,
                                  false,
                                  false,
                                  DT::FLOAT,
                                  0,
                                  VirtualOverride::NONE,
                                  ScaleOverride::B_WRONG_TYPE);
    GraphWrapper const graph(fb.GetBufferPointer(), fb.GetSize());
    EXPECT_FALSE(_builder.isApplicable(_handle, graph));
}

TEST_F(TestHipblasltMxMatmulPlanBuilder, IsApplicableRejectsWrongScaleShapeA)
{
    auto fb = createMxMatmulGraph(32,
                                  128,
                                  32,
                                  DT::FP8_E4M3,
                                  DT::HALF,
                                  32,
                                  true,
                                  true,
                                  1,
                                  false,
                                  false,
                                  DT::FLOAT,
                                  0,
                                  VirtualOverride::NONE,
                                  ScaleOverride::A_WRONG_SHAPE);
    GraphWrapper const graph(fb.GetBufferPointer(), fb.GetSize());
    EXPECT_FALSE(_builder.isApplicable(_handle, graph));
}

TEST_F(TestHipblasltMxMatmulPlanBuilder, IsApplicableRejectsWrongScaleShapeB)
{
    auto fb = createMxMatmulGraph(32,
                                  128,
                                  32,
                                  DT::FP8_E4M3,
                                  DT::HALF,
                                  32,
                                  true,
                                  true,
                                  1,
                                  false,
                                  false,
                                  DT::FLOAT,
                                  0,
                                  VirtualOverride::NONE,
                                  ScaleOverride::B_WRONG_SHAPE);
    GraphWrapper const graph(fb.GetBufferPointer(), fb.GetSize());
    EXPECT_FALSE(_builder.isApplicable(_handle, graph));
}

// ===========================================================================
// getWorkspaceSize
// ===========================================================================

TEST_F(TestGpuHipblasltMxMatmulPlanBuilder, GetWorkspaceSizeValid)
{
    auto fb = createMxMatmulGraph(32, 128, 32, DT::FP8_E4M3, DT::HALF);
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
    auto fb = createMxMatmulGraph(32, 128, 32, DT::FP8_E4M3, DT::HALF);
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
