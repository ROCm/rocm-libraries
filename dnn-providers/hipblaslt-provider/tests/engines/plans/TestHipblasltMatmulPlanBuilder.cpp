// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipblaslt/hipblaslt.h>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/MockGraph.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "HipdnnEnginePluginHandle.hpp"
#include "engines/plans/HipblasltMatmulPlanBuilder.hpp"

using namespace hipblaslt_plugin;
using namespace hipdnn_plugin_sdk;
using namespace hipdnn_flatbuffers_sdk::flatbuffer_utilities;
using namespace hipdnn_test_sdk::utilities;

class TestHipblasltMatmulPlanBuilder : public ::testing::Test
{
protected:
    HipblasltMatmulPlanBuilder _planBuilder;
    HipdnnEnginePluginHandle _handle{};
};

class TestGpuHipblasltMatmulPlanBuilder : public ::testing::Test
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

    HipblasltMatmulPlanBuilder _planBuilder;
    HipdnnEnginePluginHandle _handle;
};

// ---------------------------------------------------------------------------
// isApplicable — negative cases (CPU)
// ---------------------------------------------------------------------------

TEST_F(TestHipblasltMatmulPlanBuilder, IsApplicableRejectsTooManyNodes)
{
    MockGraph const mockGraph;
    EXPECT_CALL(mockGraph, nodeCount()).WillRepeatedly(::testing::Return(4));
    EXPECT_FALSE(_planBuilder.isApplicable(_handle, mockGraph));
}

TEST_F(TestHipblasltMatmulPlanBuilder, IsApplicableRejectsBatchnorm)
{
    auto builder = createValidBatchnormInferenceGraph();
    GraphWrapper const graph(builder.GetBufferPointer(), builder.GetSize());
    EXPECT_FALSE(_planBuilder.isApplicable(_handle, graph));
}

TEST_F(TestHipblasltMatmulPlanBuilder, IsApplicableRejectsMismatchedRanks)
{
    std::vector<int64_t> const aDims = {2, 4, 8};
    std::vector<int64_t> const aStrides = {32, 8, 1};
    std::vector<int64_t> const bDims = {8, 5};
    std::vector<int64_t> const bStrides = {5, 1};
    std::vector<int64_t> const cDims = {2, 4, 5};
    std::vector<int64_t> const cStrides = {20, 5, 1};

    auto builder = createValidMatmulGraph(aDims, aStrides, bDims, bStrides, cDims, cStrides);
    GraphWrapper const graph(builder.GetBufferPointer(), builder.GetSize());
    EXPECT_FALSE(_planBuilder.isApplicable(_handle, graph));
}

TEST_F(TestHipblasltMatmulPlanBuilder, IsApplicableRejectsIncompatibleBatch)
{
    std::vector<int64_t> const aDims = {2, 4, 8};
    std::vector<int64_t> const aStrides = {32, 8, 1};
    std::vector<int64_t> const bDims = {3, 8, 5};
    std::vector<int64_t> const bStrides = {40, 5, 1};
    std::vector<int64_t> const cDims = {6, 4, 5};
    std::vector<int64_t> const cStrides = {20, 5, 1};

    auto builder = createValidMatmulGraph(aDims, aStrides, bDims, bStrides, cDims, cStrides);
    GraphWrapper const graph(builder.GetBufferPointer(), builder.GetSize());
    EXPECT_FALSE(_planBuilder.isApplicable(_handle, graph));
}

TEST_F(TestHipblasltMatmulPlanBuilder, IsApplicableRejectsUnsupportedInputType)
{
    std::vector<int64_t> const aDims = {2, 4, 8};
    std::vector<int64_t> const aStrides = {32, 8, 1};
    std::vector<int64_t> const bDims = {2, 8, 5};
    std::vector<int64_t> const bStrides = {40, 5, 1};
    std::vector<int64_t> const cDims = {2, 4, 5};
    std::vector<int64_t> const cStrides = {20, 5, 1};

    auto builder = createValidMatmulGraph(aDims,
                                          aStrides,
                                          bDims,
                                          bStrides,
                                          cDims,
                                          cStrides,
                                          hipdnn_flatbuffers_sdk::data_objects::DataType::INT32);
    GraphWrapper const graph(builder.GetBufferPointer(), builder.GetSize());
    EXPECT_FALSE(_planBuilder.isApplicable(_handle, graph));
}

TEST_F(TestHipblasltMatmulPlanBuilder, IsApplicableRejectsUnsupportedComputeType)
{
    flatbuffers::FlatBufferBuilder const builder = createValidMatmulGraph();

    auto mutableGraph
        = hipdnn_flatbuffers_sdk::data_objects::GetMutableGraph(builder.GetBufferPointer());
    mutableGraph->mutable_nodes()->GetMutableObject(0)->mutate_compute_data_type(
        hipdnn_flatbuffers_sdk::data_objects::DataType::HALF);

    GraphWrapper const graph(builder.GetBufferPointer(), builder.GetSize());
    EXPECT_FALSE(_planBuilder.isApplicable(_handle, graph));
}

TEST_F(TestHipblasltMatmulPlanBuilder, IsApplicableRejectsUnsupportedActivationComputeType)
{
    auto builder = createValidMatmulBiasActivGraph(
        {4, 8},
        {8, 1},
        {8, 5},
        {5, 1},
        {4, 5},
        {5, 1},
        true,
        hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::GELU_APPROX_TANH_FWD);

    auto mutableGraph
        = hipdnn_flatbuffers_sdk::data_objects::GetMutableGraph(builder.GetBufferPointer());
    mutableGraph->mutable_nodes()->GetMutableObject(2)->mutate_compute_data_type(
        hipdnn_flatbuffers_sdk::data_objects::DataType::HALF);

    GraphWrapper const graph(builder.GetBufferPointer(), builder.GetSize());
    EXPECT_FALSE(_planBuilder.isApplicable(_handle, graph));
}

// ---------------------------------------------------------------------------
// isApplicable — positive cases (GPU)
// ---------------------------------------------------------------------------

TEST_F(TestGpuHipblasltMatmulPlanBuilder, IsApplicableAcceptsBroadcastBatch)
{
    std::vector<int64_t> const aDims = {2, 4, 8};
    std::vector<int64_t> const aStrides = {32, 8, 1};
    std::vector<int64_t> const bDims = {1, 8, 5};
    std::vector<int64_t> const bStrides = {40, 5, 1};
    std::vector<int64_t> const cDims = {2, 4, 5};
    std::vector<int64_t> const cStrides = {20, 5, 1};

    auto builder = createValidMatmulGraph(aDims, aStrides, bDims, bStrides, cDims, cStrides);
    GraphWrapper const graph(builder.GetBufferPointer(), builder.GetSize());
    EXPECT_TRUE(_planBuilder.isApplicable(_handle, graph));
}

TEST_F(TestGpuHipblasltMatmulPlanBuilder, IsApplicableAcceptsMatmul)
{
    auto builder = createValidMatmulGraph();
    GraphWrapper const graph(builder.GetBufferPointer(), builder.GetSize());
    EXPECT_TRUE(_planBuilder.isApplicable(_handle, graph));
}

TEST_F(TestGpuHipblasltMatmulPlanBuilder, IsApplicableAcceptsMatmulBias)
{
    auto builder = createValidMatmulBiasActivGraph(
        {4, 8},
        {8, 1},
        {8, 5},
        {5, 1},
        {4, 5},
        {5, 1},
        true,
        hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::UNSET);
    GraphWrapper const graph(builder.GetBufferPointer(), builder.GetSize());
    EXPECT_TRUE(_planBuilder.isApplicable(_handle, graph));
}

TEST_F(TestGpuHipblasltMatmulPlanBuilder, IsApplicableAcceptsMatmulBiasGelu)
{
    auto builder = createValidMatmulBiasActivGraph(
        {4, 8},
        {8, 1},
        {8, 5},
        {5, 1},
        {4, 5},
        {5, 1},
        true,
        hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::GELU_APPROX_TANH_FWD);
    GraphWrapper const graph(builder.GetBufferPointer(), builder.GetSize());
    EXPECT_TRUE(_planBuilder.isApplicable(_handle, graph));
}

TEST_F(TestGpuHipblasltMatmulPlanBuilder, IsApplicableAcceptsMatmulRelu)
{
    auto builder = createValidMatmulBiasActivGraph(
        {4, 8},
        {8, 1},
        {8, 5},
        {5, 1},
        {4, 5},
        {5, 1},
        false,
        hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RELU_FWD,
        0.0f);
    GraphWrapper const graph(builder.GetBufferPointer(), builder.GetSize());
    EXPECT_TRUE(_planBuilder.isApplicable(_handle, graph));
}

TEST_F(TestGpuHipblasltMatmulPlanBuilder, IsApplicableAcceptsMatmulBiasSwish)
{
    auto builder = createValidMatmulBiasActivGraph(
        {4, 8},
        {8, 1},
        {8, 5},
        {5, 1},
        {4, 5},
        {5, 1},
        true,
        hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::SWISH_FWD,
        std::nullopt,
        std::nullopt,
        1.0f);
    GraphWrapper const graph(builder.GetBufferPointer(), builder.GetSize());
    EXPECT_TRUE(_planBuilder.isApplicable(_handle, graph));
}

TEST_F(TestGpuHipblasltMatmulPlanBuilder, IsApplicableAcceptsMatmulBiasClamp)
{
    auto builder = createValidMatmulBiasActivGraph(
        {4, 8},
        {8, 1},
        {8, 5},
        {5, 1},
        {4, 5},
        {5, 1},
        true,
        hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RELU_FWD,
        0.f,
        6.f);
    GraphWrapper const graph(builder.GetBufferPointer(), builder.GetSize());
    EXPECT_TRUE(_planBuilder.isApplicable(_handle, graph));
}

// ---------------------------------------------------------------------------
// getWorkspaceSize — negative cases (CPU)
// ---------------------------------------------------------------------------

TEST_F(TestHipblasltMatmulPlanBuilder, GetWorkspaceSizeThrowsOnTooManyNodes)
{
    MockGraph const mockGraph;
    EXPECT_CALL(mockGraph, nodeCount()).WillRepeatedly(::testing::Return(4));
    EXPECT_THROW(_planBuilder.getWorkspaceSize(_handle, mockGraph), HipdnnPluginException);
}

TEST_F(TestHipblasltMatmulPlanBuilder, GetWorkspaceSizeThrowsOnBatchnorm)
{
    auto builder = createValidBatchnormInferenceGraph();
    GraphWrapper const graph(builder.GetBufferPointer(), builder.GetSize());
    EXPECT_THROW(_planBuilder.getWorkspaceSize(_handle, graph), HipdnnPluginException);
}

// ---------------------------------------------------------------------------
// getWorkspaceSize — positive cases (GPU)
// ---------------------------------------------------------------------------

TEST_F(TestGpuHipblasltMatmulPlanBuilder, GetWorkspaceSizeMatmul)
{
    auto builder = createValidMatmulGraph();
    GraphWrapper const graph(builder.GetBufferPointer(), builder.GetSize());
    EXPECT_NO_THROW(_planBuilder.getWorkspaceSize(_handle, graph));
}

TEST_F(TestGpuHipblasltMatmulPlanBuilder, GetWorkspaceSizeMatmulBias)
{
    auto builder = createValidMatmulBiasActivGraph(
        {4, 8},
        {8, 1},
        {8, 5},
        {5, 1},
        {4, 5},
        {5, 1},
        true,
        hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::UNSET);
    GraphWrapper const graph(builder.GetBufferPointer(), builder.GetSize());
    EXPECT_NO_THROW(_planBuilder.getWorkspaceSize(_handle, graph));
}

TEST_F(TestGpuHipblasltMatmulPlanBuilder, GetWorkspaceSizeMatmulBiasGelu)
{
    auto builder = createValidMatmulBiasActivGraph(
        {4, 8},
        {8, 1},
        {8, 5},
        {5, 1},
        {4, 5},
        {5, 1},
        true,
        hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::GELU_APPROX_TANH_FWD);
    GraphWrapper const graph(builder.GetBufferPointer(), builder.GetSize());
    EXPECT_NO_THROW(_planBuilder.getWorkspaceSize(_handle, graph));
}

TEST_F(TestGpuHipblasltMatmulPlanBuilder, GetWorkspaceSizeMatmulRelu)
{
    auto builder = createValidMatmulBiasActivGraph(
        {4, 8},
        {8, 1},
        {8, 5},
        {5, 1},
        {4, 5},
        {5, 1},
        false,
        hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RELU_FWD,
        0.0f);
    GraphWrapper const graph(builder.GetBufferPointer(), builder.GetSize());
    EXPECT_NO_THROW(_planBuilder.getWorkspaceSize(_handle, graph));
}

// ---------------------------------------------------------------------------
// buildPlan — negative cases (CPU)
// ---------------------------------------------------------------------------

TEST_F(TestHipblasltMatmulPlanBuilder, BuildPlanThrowsOnTooManyNodes)
{
    MockGraph const mockGraph;
    EXPECT_CALL(mockGraph, nodeCount()).WillRepeatedly(::testing::Return(4));
    HipdnnEnginePluginExecutionContext ctx;

    EXPECT_THROW(_planBuilder.buildPlan(_handle, mockGraph, ctx), HipdnnPluginException);
    EXPECT_FALSE(ctx.hasValidPlan());
}

TEST_F(TestHipblasltMatmulPlanBuilder, BuildPlanThrowsOnBatchnorm)
{
    auto builder = createValidBatchnormInferenceGraph();
    GraphWrapper const graph(builder.GetBufferPointer(), builder.GetSize());
    HipdnnEnginePluginExecutionContext ctx;

    EXPECT_THROW(_planBuilder.buildPlan(_handle, graph, ctx), HipdnnPluginException);
    EXPECT_FALSE(ctx.hasValidPlan());
}

// ---------------------------------------------------------------------------
// buildPlan — positive cases (GPU)
// ---------------------------------------------------------------------------

TEST_F(TestGpuHipblasltMatmulPlanBuilder, BuildPlanMatmul)
{
    auto builder = createValidMatmulGraph();
    GraphWrapper const graph(builder.GetBufferPointer(), builder.GetSize());
    HipdnnEnginePluginExecutionContext ctx;

    EXPECT_NO_THROW(_planBuilder.buildPlan(_handle, graph, ctx));
    EXPECT_TRUE(ctx.hasValidPlan());
}

TEST_F(TestGpuHipblasltMatmulPlanBuilder, BuildPlanMatmulBias)
{
    auto builder = createValidMatmulBiasActivGraph(
        {4, 8},
        {8, 1},
        {8, 5},
        {5, 1},
        {4, 5},
        {5, 1},
        true,
        hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::UNSET);
    GraphWrapper const graph(builder.GetBufferPointer(), builder.GetSize());
    HipdnnEnginePluginExecutionContext ctx;

    EXPECT_NO_THROW(_planBuilder.buildPlan(_handle, graph, ctx));
    EXPECT_TRUE(ctx.hasValidPlan());
}

TEST_F(TestGpuHipblasltMatmulPlanBuilder, BuildPlanMatmulBiasGelu)
{
    auto builder = createValidMatmulBiasActivGraph(
        {4, 8},
        {8, 1},
        {8, 5},
        {5, 1},
        {4, 5},
        {5, 1},
        true,
        hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::GELU_APPROX_TANH_FWD);
    GraphWrapper const graph(builder.GetBufferPointer(), builder.GetSize());
    HipdnnEnginePluginExecutionContext ctx;

    EXPECT_NO_THROW(_planBuilder.buildPlan(_handle, graph, ctx));
    EXPECT_TRUE(ctx.hasValidPlan());
}

TEST_F(TestGpuHipblasltMatmulPlanBuilder, BuildPlanMatmulRelu)
{
    auto builder = createValidMatmulBiasActivGraph(
        {4, 8},
        {8, 1},
        {8, 5},
        {5, 1},
        {4, 5},
        {5, 1},
        false,
        hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RELU_FWD,
        0.0f);
    GraphWrapper const graph(builder.GetBufferPointer(), builder.GetSize());
    HipdnnEnginePluginExecutionContext ctx;

    EXPECT_NO_THROW(_planBuilder.buildPlan(_handle, graph, ctx));
    EXPECT_TRUE(ctx.hasValidPlan());
}
