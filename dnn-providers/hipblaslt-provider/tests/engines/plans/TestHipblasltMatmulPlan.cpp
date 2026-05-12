// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipblaslt/hipblaslt.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "HipdnnEnginePluginHandle.hpp"
#include "engines/plans/HipblasltMatmulPlan.hpp"

using namespace hipblaslt_plugin;
using namespace hipdnn_plugin_sdk;
using namespace hipdnn_flatbuffers_sdk::flatbuffer_utilities;
using namespace hipdnn_test_sdk::utilities;

class TestGpuMatmulPlan : public ::testing::Test
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

    HipdnnEnginePluginHandle _handle;
};

TEST(TestMatmulParams, InitializesAllTensorsFromValidGraph)
{
    // Create a valid matmul graph
    auto builder = createValidMatmulGraph();
    GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    // Get the matmul node and attributes
    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_MatmulAttributes();
    ASSERT_NE(attrs, nullptr);

    // Construct params
    MatmulParams params(*attrs, graph.getTensorMap());

    // All required tensors should be accessible
    EXPECT_NO_THROW(params.a());
    EXPECT_NO_THROW(params.b());
    EXPECT_NO_THROW(params.c());
    EXPECT_NO_THROW(params.desc());
    EXPECT_FALSE(params.biasUid().has_value());
}

TEST(TestMatmulParams, InitializesWithBiasAndActivation)
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
    GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    auto* matmulAttrs = graph.getNode(0).attributes_as_MatmulAttributes();
    auto* biasAttrs = graph.getNode(1).attributes_as_PointwiseAttributes();
    auto* activAttrs = graph.getNode(2).attributes_as_PointwiseAttributes();
    ASSERT_NE(matmulAttrs, nullptr);
    ASSERT_NE(biasAttrs, nullptr);
    ASSERT_NE(activAttrs, nullptr);

    MatmulParams params(*matmulAttrs, biasAttrs, activAttrs, graph.getTensorMap());

    EXPECT_NO_THROW(params.a());
    EXPECT_NO_THROW(params.b());
    EXPECT_NO_THROW(params.c());
    EXPECT_NO_THROW(params.desc());
    EXPECT_TRUE(params.biasUid().has_value());
}

TEST(TestMatmulParams, InitializesWithActivationOnly)
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
    GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    auto* matmulAttrs = graph.getNode(0).attributes_as_MatmulAttributes();
    auto* activAttrs = graph.getNode(1).attributes_as_PointwiseAttributes();
    ASSERT_NE(matmulAttrs, nullptr);
    ASSERT_NE(activAttrs, nullptr);

    MatmulParams params(*matmulAttrs, nullptr, activAttrs, graph.getTensorMap());

    EXPECT_NO_THROW(params.a());
    EXPECT_NO_THROW(params.b());
    EXPECT_NO_THROW(params.c());
    EXPECT_FALSE(params.biasUid().has_value());
}

TEST(TestMatmulParams, InitializesWithBiasOnly)
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
    GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    auto* matmulAttrs = graph.getNode(0).attributes_as_MatmulAttributes();
    auto* biasAttrs = graph.getNode(1).attributes_as_PointwiseAttributes();
    ASSERT_NE(matmulAttrs, nullptr);
    ASSERT_NE(biasAttrs, nullptr);

    MatmulParams params(*matmulAttrs, biasAttrs, nullptr, graph.getTensorMap());

    EXPECT_NO_THROW(params.a());
    EXPECT_NO_THROW(params.b());
    EXPECT_NO_THROW(params.c());
    EXPECT_NO_THROW(params.desc());
    EXPECT_TRUE(params.biasUid().has_value());
}

TEST(TestMatmulParams, BiasUidMatchesBiasTensor)
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
    GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    auto* matmulAttrs = graph.getNode(0).attributes_as_MatmulAttributes();
    auto* biasAttrs = graph.getNode(1).attributes_as_PointwiseAttributes();

    MatmulParams params(*matmulAttrs, biasAttrs, nullptr, graph.getTensorMap());

    ASSERT_TRUE(params.biasUid().has_value());
    // The bias UID should correspond to a tensor in the graph's tensor map
    EXPECT_NE(graph.getTensorMap().find(params.biasUid().value()), graph.getTensorMap().end());
}

TEST(TestMatmulParams, RowMajor)
{
    // Row-major storage has stride[last] == 1
    std::vector<int64_t> aDims = {4, 8};
    std::vector<int64_t> aStrides = {8, 1}; // Row-major
    std::vector<int64_t> bDims = {8, 5};
    std::vector<int64_t> bStrides = {5, 1}; // Row-major
    std::vector<int64_t> cDims = {4, 5};
    std::vector<int64_t> cStrides = {5, 1};

    auto builder = createValidMatmulGraph(aDims, aStrides, bDims, bStrides, cDims, cStrides);
    GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_MatmulAttributes();
    ASSERT_NE(attrs, nullptr);

    // Should not throw for valid row-major strides
    EXPECT_NO_THROW(MatmulParams(*attrs, graph.getTensorMap()));
}

TEST(TestMatmulParams, ColumnMajor)
{
    // Column-major storage has stride[last-1] == 1
    std::vector<int64_t> aDims = {4, 8};
    std::vector<int64_t> aStrides = {1, 4}; // Column-major
    std::vector<int64_t> bDims = {8, 5};
    std::vector<int64_t> bStrides = {1, 8}; // Column-major
    std::vector<int64_t> cDims = {4, 5};
    std::vector<int64_t> cStrides = {5, 1};

    auto builder = createValidMatmulGraph(aDims, aStrides, bDims, bStrides, cDims, cStrides);
    GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_MatmulAttributes();
    ASSERT_NE(attrs, nullptr);

    // Should not throw for valid column-major strides
    EXPECT_NO_THROW(MatmulParams(*attrs, graph.getTensorMap()));
}

TEST(TestMatmulParams, BatchBroadcastWithABatchOne)
{
    // [1, M, K] x [2, K, N] -> [2, M, N]
    std::vector<int64_t> aDims = {1, 4, 8};
    std::vector<int64_t> aStrides = {32, 8, 1};
    std::vector<int64_t> bDims = {2, 8, 5};
    std::vector<int64_t> bStrides = {40, 5, 1};
    std::vector<int64_t> cDims = {2, 4, 5};
    std::vector<int64_t> cStrides = {20, 5, 1};

    auto builder = createValidMatmulGraph(aDims, aStrides, bDims, bStrides, cDims, cStrides);
    GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_MatmulAttributes();
    ASSERT_NE(attrs, nullptr);

    // Should not throw for valid broadcast configuration
    EXPECT_NO_THROW(MatmulParams(*attrs, graph.getTensorMap()));
}

TEST(TestMatmulParams, BatchBroadcastWithBBatchOne)
{
    // [3, M, K] x [1, K, N] -> [3, M, N]
    std::vector<int64_t> aDims = {3, 4, 8};
    std::vector<int64_t> aStrides = {32, 8, 1};
    std::vector<int64_t> bDims = {1, 8, 5};
    std::vector<int64_t> bStrides = {40, 5, 1};
    std::vector<int64_t> cDims = {3, 4, 5};
    std::vector<int64_t> cStrides = {20, 5, 1};

    auto builder = createValidMatmulGraph(aDims, aStrides, bDims, bStrides, cDims, cStrides);
    GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_MatmulAttributes();
    ASSERT_NE(attrs, nullptr);

    // Should not throw for valid broadcast configuration
    EXPECT_NO_THROW(MatmulParams(*attrs, graph.getTensorMap()));
}

TEST(TestMatmulParams, ThrowsOnDifferentRanks)
{
    // [M, K] x [2, K, N] -> [2, M, N]
    std::vector<int64_t> aDims = {4, 8};
    std::vector<int64_t> aStrides = {8, 1};
    std::vector<int64_t> bDims = {2, 8, 5};
    std::vector<int64_t> bStrides = {40, 5, 1};
    std::vector<int64_t> cDims = {2, 4, 5};
    std::vector<int64_t> cStrides = {20, 5, 1};

    auto builder = createValidMatmulGraph(aDims, aStrides, bDims, bStrides, cDims, cStrides);
    GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_MatmulAttributes();
    ASSERT_NE(attrs, nullptr);

    // Should throw due to mismatched ranks
    EXPECT_THROW(MatmulParams(*attrs, graph.getTensorMap()), HipdnnPluginException);
}

TEST(TestMatmulParams, ThrowsOnIncompatibleBatchDimensions)
{
    // [2, M, K] x [3, K, N] -> [3, M, N]
    std::vector<int64_t> aDims = {2, 4, 8};
    std::vector<int64_t> aStrides = {32, 8, 1};
    std::vector<int64_t> bDims = {3, 8, 5};
    std::vector<int64_t> bStrides = {40, 5, 1};
    std::vector<int64_t> cDims = {3, 4, 5};
    std::vector<int64_t> cStrides = {20, 5, 1};

    auto builder = createValidMatmulGraph(aDims, aStrides, bDims, bStrides, cDims, cStrides);
    GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_MatmulAttributes();
    ASSERT_NE(attrs, nullptr);

    // Should throw due to incompatible batch dimensions
    EXPECT_THROW(MatmulParams(*attrs, graph.getTensorMap()), HipdnnPluginException);
}

TEST(TestMatmulParams, ThrowsOnOutputBatchMismatch)
{
    // [2, M, K] x [2, K, N] -> [3, M, N]
    std::vector<int64_t> aDims = {2, 4, 8};
    std::vector<int64_t> aStrides = {32, 8, 1};
    std::vector<int64_t> bDims = {2, 8, 5};
    std::vector<int64_t> bStrides = {40, 5, 1};
    std::vector<int64_t> cDims = {3, 4, 5};
    std::vector<int64_t> cStrides = {20, 5, 1};

    auto builder = createValidMatmulGraph(aDims, aStrides, bDims, bStrides, cDims, cStrides);
    GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_MatmulAttributes();
    ASSERT_NE(attrs, nullptr);

    // Should throw due to output batch mismatch
    EXPECT_THROW(MatmulParams(*attrs, graph.getTensorMap()), HipdnnPluginException);
}

TEST(TestMatmulParams, ThrowsOnOutputBatchSmallerThanMax)
{
    // [1, M, K] x [3, K, N] -> [2, M, N]
    std::vector<int64_t> aDims = {1, 4, 8};
    std::vector<int64_t> aStrides = {32, 8, 1};
    std::vector<int64_t> bDims = {3, 8, 5};
    std::vector<int64_t> bStrides = {40, 5, 1};
    std::vector<int64_t> cDims = {2, 4, 5};
    std::vector<int64_t> cStrides = {20, 5, 1};

    auto builder = createValidMatmulGraph(aDims, aStrides, bDims, bStrides, cDims, cStrides);
    GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_MatmulAttributes();
    ASSERT_NE(attrs, nullptr);

    EXPECT_THROW(MatmulParams(*attrs, graph.getTensorMap()), HipdnnPluginException);
}

TEST(TestMatmulParams, Tensor4DBroadcastWithUnitBatchB)
{
    // [3, 3, m, k] x [1, 1, k, n] -> [3, 3, m, n]
    std::vector<int64_t> aDims = {3, 3, 4, 8};
    std::vector<int64_t> aStrides = {96, 32, 8, 1};
    std::vector<int64_t> bDims = {1, 1, 8, 5};
    std::vector<int64_t> bStrides = {40, 40, 5, 1};
    std::vector<int64_t> cDims = {3, 3, 4, 5};
    std::vector<int64_t> cStrides = {60, 20, 5, 1};

    auto builder = createValidMatmulGraph(aDims, aStrides, bDims, bStrides, cDims, cStrides);
    GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_MatmulAttributes();
    ASSERT_NE(attrs, nullptr);

    // Should not throw - B batch is all 1s, broadcasting is valid
    EXPECT_NO_THROW(MatmulParams(*attrs, graph.getTensorMap()));
}

TEST(TestMatmulParams, Tensor4DThrowsOnIncompatibleBatch)
{
    // [3, 3, m, k] x [1, 3, k, n] -> [3, 3, m, n]
    std::vector<int64_t> aDims = {3, 3, 4, 8};
    std::vector<int64_t> aStrides = {96, 32, 8, 1};
    std::vector<int64_t> bDims = {1, 3, 8, 5};
    std::vector<int64_t> bStrides = {120, 40, 5, 1};
    std::vector<int64_t> cDims = {3, 3, 4, 5};
    std::vector<int64_t> cStrides = {60, 20, 5, 1};

    auto builder = createValidMatmulGraph(aDims, aStrides, bDims, bStrides, cDims, cStrides);
    GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_MatmulAttributes();
    ASSERT_NE(attrs, nullptr);

    // Should throw - batch dimensions are incompatible (9 vs 3, neither is 1)
    EXPECT_THROW(MatmulParams(*attrs, graph.getTensorMap()), HipdnnPluginException);
}

TEST(TestMatmulParams, Tensor4DThrowsOnOutputBatchMismatch)
{
    // [3, 1, m, k] x [1, 3, k, n] -> [3, 3, m, n]
    std::vector<int64_t> aDims = {3, 1, 4, 8};
    std::vector<int64_t> aStrides = {32, 32, 8, 1};
    std::vector<int64_t> bDims = {1, 3, 8, 5};
    std::vector<int64_t> bStrides = {120, 40, 5, 1};
    std::vector<int64_t> cDims = {3, 3, 4, 5};
    std::vector<int64_t> cStrides = {60, 20, 5, 1};

    auto builder = createValidMatmulGraph(aDims, aStrides, bDims, bStrides, cDims, cStrides);
    GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_MatmulAttributes();
    ASSERT_NE(attrs, nullptr);

    // Should throw - strided input batches
    EXPECT_THROW(MatmulParams(*attrs, graph.getTensorMap()), HipdnnPluginException);
}

TEST_F(TestGpuMatmulPlan, CreatesPlanWithValidGraph)
{
    // Create a valid matmul graph
    auto builder = createValidMatmulGraph();
    GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    // Get the matmul node and attributes
    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_MatmulAttributes();
    ASSERT_NE(attrs, nullptr);

    // Construct params
    MatmulParams params(*attrs, graph.getTensorMap());

    // Create plan - should not throw
    EXPECT_NO_THROW(MatmulPlan(_handle, std::move(params)));
}

TEST_F(TestGpuMatmulPlan, PlanReturnsValidWorkspaceSize)
{
    // Create a valid matmul graph
    auto builder = createValidMatmulGraph();
    GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_MatmulAttributes();
    ASSERT_NE(attrs, nullptr);

    MatmulParams params(*attrs, graph.getTensorMap());
    MatmulPlan plan(_handle, std::move(params));

    // Workspace size should be >= 0
    EXPECT_GE(plan.getWorkspaceSize(_handle), 0u);
}

TEST_F(TestGpuMatmulPlan, CreatesPlanWithColumnMajorInputs)
{
    // Column-major strides
    std::vector<int64_t> aDims = {4, 8};
    std::vector<int64_t> aStrides = {1, 4}; // Column-major
    std::vector<int64_t> bDims = {8, 5};
    std::vector<int64_t> bStrides = {1, 8}; // Column-major
    std::vector<int64_t> cDims = {4, 5};
    std::vector<int64_t> cStrides = {5, 1};

    auto builder = createValidMatmulGraph(aDims, aStrides, bDims, bStrides, cDims, cStrides);
    GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_MatmulAttributes();
    ASSERT_NE(attrs, nullptr);

    MatmulParams params(*attrs, graph.getTensorMap());

    // Create plan - should not throw for column-major inputs
    EXPECT_NO_THROW(MatmulPlan(_handle, std::move(params)));
}

TEST_F(TestGpuMatmulPlan, CreatesPlanWithLargerMatrices)
{
    // Larger matrices for testing
    std::vector<int64_t> aDims = {64, 128};
    std::vector<int64_t> aStrides = {128, 1};
    std::vector<int64_t> bDims = {128, 256};
    std::vector<int64_t> bStrides = {256, 1};
    std::vector<int64_t> cDims = {64, 256};
    std::vector<int64_t> cStrides = {256, 1};

    auto builder = createValidMatmulGraph(aDims, aStrides, bDims, bStrides, cDims, cStrides);
    GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_MatmulAttributes();
    ASSERT_NE(attrs, nullptr);

    MatmulParams params(*attrs, graph.getTensorMap());

    // Create plan - should not throw
    EXPECT_NO_THROW(MatmulPlan(_handle, std::move(params)));
}

TEST_F(TestGpuMatmulPlan, CreatesPlanWithBatchedMatmul)
{
    // Batched matmul: [batch, M, K] x [batch, K, N] -> [batch, M, N]
    std::vector<int64_t> aDims = {2, 4, 8};
    std::vector<int64_t> aStrides = {32, 8, 1};
    std::vector<int64_t> bDims = {2, 8, 5};
    std::vector<int64_t> bStrides = {40, 5, 1};
    std::vector<int64_t> cDims = {2, 4, 5};
    std::vector<int64_t> cStrides = {20, 5, 1};

    auto builder = createValidMatmulGraph(aDims, aStrides, bDims, bStrides, cDims, cStrides);
    GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_MatmulAttributes();
    ASSERT_NE(attrs, nullptr);

    MatmulParams params(*attrs, graph.getTensorMap());

    // Create plan - should not throw
    EXPECT_NO_THROW(MatmulPlan(_handle, std::move(params)));
}

TEST_F(TestGpuMatmulPlan, CreatesPlanWithHalfPrecision)
{
    // Half precision matmul
    std::vector<int64_t> aDims = {16, 32};
    std::vector<int64_t> aStrides = {32, 1};
    std::vector<int64_t> bDims = {32, 64};
    std::vector<int64_t> bStrides = {64, 1};
    std::vector<int64_t> cDims = {16, 64};
    std::vector<int64_t> cStrides = {64, 1};

    auto builder = createValidMatmulGraph(aDims,
                                          aStrides,
                                          bDims,
                                          bStrides,
                                          cDims,
                                          cStrides,
                                          hipdnn_flatbuffers_sdk::data_objects::DataType::HALF);
    GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_MatmulAttributes();
    ASSERT_NE(attrs, nullptr);

    MatmulParams params(*attrs, graph.getTensorMap());

    // Create plan - should not throw
    EXPECT_NO_THROW(MatmulPlan(_handle, std::move(params)));
}

TEST_F(TestGpuMatmulPlan, CreatesPlanWithBFloat16)
{
    // BFloat16 matmul
    std::vector<int64_t> aDims = {16, 32};
    std::vector<int64_t> aStrides = {32, 1};
    std::vector<int64_t> bDims = {32, 64};
    std::vector<int64_t> bStrides = {64, 1};
    std::vector<int64_t> cDims = {16, 64};
    std::vector<int64_t> cStrides = {64, 1};

    auto builder = createValidMatmulGraph(aDims,
                                          aStrides,
                                          bDims,
                                          bStrides,
                                          cDims,
                                          cStrides,
                                          hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16);
    GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_MatmulAttributes();
    ASSERT_NE(attrs, nullptr);

    MatmulParams params(*attrs, graph.getTensorMap());

    // Create plan - should not throw
    EXPECT_NO_THROW(MatmulPlan(_handle, std::move(params)));
}

TEST_F(TestGpuMatmulPlan, CreatesPlanWithBiasAndGelu)
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
    GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    auto* matmulAttrs = graph.getNode(0).attributes_as_MatmulAttributes();
    auto* biasAttrs = graph.getNode(1).attributes_as_PointwiseAttributes();
    auto* activAttrs = graph.getNode(2).attributes_as_PointwiseAttributes();
    ASSERT_NE(matmulAttrs, nullptr);
    ASSERT_NE(biasAttrs, nullptr);
    ASSERT_NE(activAttrs, nullptr);

    MatmulParams params(*matmulAttrs, biasAttrs, activAttrs, graph.getTensorMap());

    EXPECT_NO_THROW(MatmulPlan(_handle, std::move(params)));
}

TEST_F(TestGpuMatmulPlan, CreatesPlanWithBiasAndRelu)
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
        0.0f);
    GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    auto* matmulAttrs = graph.getNode(0).attributes_as_MatmulAttributes();
    auto* biasAttrs = graph.getNode(1).attributes_as_PointwiseAttributes();
    auto* activAttrs = graph.getNode(2).attributes_as_PointwiseAttributes();
    ASSERT_NE(matmulAttrs, nullptr);
    ASSERT_NE(biasAttrs, nullptr);
    ASSERT_NE(activAttrs, nullptr);

    MatmulParams params(*matmulAttrs, biasAttrs, activAttrs, graph.getTensorMap());

    EXPECT_NO_THROW(MatmulPlan(_handle, std::move(params)));
}

TEST_F(TestGpuMatmulPlan, CreatesPlanWithActivationOnly)
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
    GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    auto* matmulAttrs = graph.getNode(0).attributes_as_MatmulAttributes();
    auto* activAttrs = graph.getNode(1).attributes_as_PointwiseAttributes();
    ASSERT_NE(matmulAttrs, nullptr);
    ASSERT_NE(activAttrs, nullptr);

    MatmulParams params(*matmulAttrs, nullptr, activAttrs, graph.getTensorMap());

    EXPECT_NO_THROW(MatmulPlan(_handle, std::move(params)));
}

TEST_F(TestGpuMatmulPlan, CreatesPlanWithBiasAndSwish)
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
    GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    auto* matmulAttrs = graph.getNode(0).attributes_as_MatmulAttributes();
    auto* biasAttrs = graph.getNode(1).attributes_as_PointwiseAttributes();
    auto* activAttrs = graph.getNode(2).attributes_as_PointwiseAttributes();
    ASSERT_NE(matmulAttrs, nullptr);
    ASSERT_NE(biasAttrs, nullptr);
    ASSERT_NE(activAttrs, nullptr);

    MatmulParams params(*matmulAttrs, biasAttrs, activAttrs, graph.getTensorMap());

    EXPECT_NO_THROW(MatmulPlan(_handle, std::move(params)));
}

TEST_F(TestGpuMatmulPlan, CreatesPlanWithBiasOnly)
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
    GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    auto* matmulAttrs = graph.getNode(0).attributes_as_MatmulAttributes();
    auto* biasAttrs = graph.getNode(1).attributes_as_PointwiseAttributes();
    ASSERT_NE(matmulAttrs, nullptr);
    ASSERT_NE(biasAttrs, nullptr);

    MatmulParams params(*matmulAttrs, biasAttrs, nullptr, graph.getTensorMap());

    EXPECT_NO_THROW(MatmulPlan(_handle, std::move(params)));
}

TEST_F(TestGpuMatmulPlan, PlanReturnValidWorkspaceSizeWithBiasOnly)
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
    GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    auto* matmulAttrs = graph.getNode(0).attributes_as_MatmulAttributes();
    auto* biasAttrs = graph.getNode(1).attributes_as_PointwiseAttributes();
    ASSERT_NE(matmulAttrs, nullptr);
    ASSERT_NE(biasAttrs, nullptr);

    MatmulParams params(*matmulAttrs, biasAttrs, nullptr, graph.getTensorMap());
    MatmulPlan plan(_handle, std::move(params));

    EXPECT_GE(plan.getWorkspaceSize(_handle), 0u);
}

TEST_F(TestGpuMatmulPlan, PlanReturnValidWorkspaceSizeWithBiasActivation)
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
    GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    auto* matmulAttrs = graph.getNode(0).attributes_as_MatmulAttributes();
    auto* biasAttrs = graph.getNode(1).attributes_as_PointwiseAttributes();
    auto* activAttrs = graph.getNode(2).attributes_as_PointwiseAttributes();
    ASSERT_NE(matmulAttrs, nullptr);
    ASSERT_NE(biasAttrs, nullptr);
    ASSERT_NE(activAttrs, nullptr);

    MatmulParams params(*matmulAttrs, biasAttrs, activAttrs, graph.getTensorMap());
    MatmulPlan plan(_handle, std::move(params));

    EXPECT_GE(plan.getWorkspaceSize(_handle), 0u);
}
