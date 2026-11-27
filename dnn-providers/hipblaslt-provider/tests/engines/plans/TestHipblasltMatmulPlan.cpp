// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipblaslt/hipblaslt.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <optional>
#include <string>
#include <vector>

#include "HipdnnEnginePluginHandle.hpp"
#include "TestWorkarounds.hpp"
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
        SKIP_IF_WORKAROUND_ISSUE_9962(_handle);
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
    const auto builder = createValidMatmulGraph();
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    // Get the matmul node and attributes
    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_MatmulAttributes();
    ASSERT_NE(attrs, nullptr);

    // Construct params
    const MatmulParams params(*attrs, graph.getTensorMap());

    // All required tensors should be accessible
    EXPECT_NO_THROW(params.a());
    EXPECT_NO_THROW(params.b());
    EXPECT_NO_THROW(params.c());
    EXPECT_NO_THROW(params.desc());
    EXPECT_FALSE(params.biasUid().has_value());
}

TEST(TestMatmulParams, InitializesWithBiasAndActivation)
{
    const auto builder = createValidMatmulBiasActivGraph(
        {4, 8},
        {8, 1},
        {8, 5},
        {5, 1},
        {4, 5},
        {5, 1},
        true,
        hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::GELU_APPROX_TANH_FWD);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    auto* matmulAttrs = graph.getNode(0).attributes_as_MatmulAttributes();
    auto* biasAttrs = graph.getNode(1).attributes_as_PointwiseAttributes();
    auto* activAttrs = graph.getNode(2).attributes_as_PointwiseAttributes();
    ASSERT_NE(matmulAttrs, nullptr);
    ASSERT_NE(biasAttrs, nullptr);
    ASSERT_NE(activAttrs, nullptr);

    const MatmulParams params(*matmulAttrs, biasAttrs, activAttrs, graph.getTensorMap());

    EXPECT_NO_THROW(params.a());
    EXPECT_NO_THROW(params.b());
    EXPECT_NO_THROW(params.c());
    EXPECT_NO_THROW(params.desc());
    EXPECT_TRUE(params.biasUid().has_value());
}

TEST(TestMatmulParams, InitializesWithActivationOnly)
{
    const auto builder = createValidMatmulBiasActivGraph(
        {4, 8},
        {8, 1},
        {8, 5},
        {5, 1},
        {4, 5},
        {5, 1},
        false,
        hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RELU_FWD,
        0.0f);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    auto* matmulAttrs = graph.getNode(0).attributes_as_MatmulAttributes();
    auto* activAttrs = graph.getNode(1).attributes_as_PointwiseAttributes();
    ASSERT_NE(matmulAttrs, nullptr);
    ASSERT_NE(activAttrs, nullptr);

    const MatmulParams params(*matmulAttrs, nullptr, activAttrs, graph.getTensorMap());

    EXPECT_NO_THROW(params.a());
    EXPECT_NO_THROW(params.b());
    EXPECT_NO_THROW(params.c());
    EXPECT_FALSE(params.biasUid().has_value());
}

TEST(TestMatmulParams, InitializesWithBiasOnly)
{
    const auto builder = createValidMatmulBiasActivGraph(
        {4, 8},
        {8, 1},
        {8, 5},
        {5, 1},
        {4, 5},
        {5, 1},
        true,
        hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::UNSET);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    auto* matmulAttrs = graph.getNode(0).attributes_as_MatmulAttributes();
    auto* biasAttrs = graph.getNode(1).attributes_as_PointwiseAttributes();
    ASSERT_NE(matmulAttrs, nullptr);
    ASSERT_NE(biasAttrs, nullptr);

    const MatmulParams params(*matmulAttrs, biasAttrs, nullptr, graph.getTensorMap());

    EXPECT_NO_THROW(params.a());
    EXPECT_NO_THROW(params.b());
    EXPECT_NO_THROW(params.c());
    EXPECT_NO_THROW(params.desc());
    EXPECT_TRUE(params.biasUid().has_value());
}

TEST(TestMatmulParams, BiasUidMatchesBiasTensor)
{
    const auto builder = createValidMatmulBiasActivGraph(
        {4, 8},
        {8, 1},
        {8, 5},
        {5, 1},
        {4, 5},
        {5, 1},
        true,
        hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::UNSET);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    auto* matmulAttrs = graph.getNode(0).attributes_as_MatmulAttributes();
    auto* biasAttrs = graph.getNode(1).attributes_as_PointwiseAttributes();

    const MatmulParams params(*matmulAttrs, biasAttrs, nullptr, graph.getTensorMap());

    ASSERT_TRUE(params.biasUid().has_value());
    // The bias UID should correspond to a tensor in the graph's tensor map
    EXPECT_NE(graph.getTensorMap().find(params.biasUid().value()), graph.getTensorMap().end());
}

TEST(TestMatmulParams, RowMajor)
{
    // Row-major storage has stride[last] == 1
    const std::vector<int64_t> aDims = {4, 8};
    const std::vector<int64_t> aStrides = {8, 1}; // Row-major
    const std::vector<int64_t> bDims = {8, 5};
    const std::vector<int64_t> bStrides = {5, 1}; // Row-major
    const std::vector<int64_t> cDims = {4, 5};
    const std::vector<int64_t> cStrides = {5, 1};

    const auto builder = createValidMatmulGraph(aDims, aStrides, bDims, bStrides, cDims, cStrides);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_MatmulAttributes();
    ASSERT_NE(attrs, nullptr);

    // Should not throw for valid row-major strides
    EXPECT_NO_THROW(MatmulParams(*attrs, graph.getTensorMap()));
}

TEST(TestMatmulParams, ColumnMajor)
{
    // Column-major storage has stride[last-1] == 1
    const std::vector<int64_t> aDims = {4, 8};
    const std::vector<int64_t> aStrides = {1, 4}; // Column-major
    const std::vector<int64_t> bDims = {8, 5};
    const std::vector<int64_t> bStrides = {1, 8}; // Column-major
    const std::vector<int64_t> cDims = {4, 5};
    const std::vector<int64_t> cStrides = {5, 1};

    const auto builder = createValidMatmulGraph(aDims, aStrides, bDims, bStrides, cDims, cStrides);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_MatmulAttributes();
    ASSERT_NE(attrs, nullptr);

    // Should not throw for valid column-major strides
    EXPECT_NO_THROW(MatmulParams(*attrs, graph.getTensorMap()));
}

// With K == 1, A [M, 1] and B [1, N] carry strides that no stride-1 rule accepts.
TEST(TestMatmulParams, AcceptsSizeOneKOperandsWithNonCanonicalStrides)
{
    const std::vector<int64_t> cDims = {4, 5};
    const std::vector<int64_t> cStrides = {5, 1};

    {
        // A [M, 1] with column-style strides.
        const auto builder
            = createValidMatmulGraph({4, 1}, {1, 4}, {1, 5}, {5, 1}, cDims, cStrides);
        const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
        auto* attrs = graph.getNode(0).attributes_as_MatmulAttributes();
        ASSERT_NE(attrs, nullptr);
        EXPECT_NO_THROW(MatmulParams(*attrs, graph.getTensorMap()));
    }
    {
        // B [1, N] with strides that are neither 1.
        const auto builder
            = createValidMatmulGraph({4, 1}, {1, 1}, {1, 5}, {99, 100}, cDims, cStrides);
        const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
        auto* attrs = graph.getNode(0).attributes_as_MatmulAttributes();
        ASSERT_NE(attrs, nullptr);
        EXPECT_NO_THROW(MatmulParams(*attrs, graph.getTensorMap()));
    }
}

namespace
{
hipblasOperation_t getTransAttribute(const HipblasltMatmulDesc& desc,
                                     hipblasLtMatmulDescAttributes_t attr)
{
    hipblasOperation_t value = HIPBLAS_OP_N;
    size_t sizeWritten = 0;
    EXPECT_EQ(hipblasLtMatmulDescGetAttribute(
                  desc.matmulDesc(), attr, static_cast<void*>(&value), sizeof(value), &sizeWritten),
              HIPBLAS_STATUS_SUCCESS);
    EXPECT_EQ(sizeWritten, sizeof(value));
    return value;
}
} // namespace

// B [1, N] with strides {99, 100} has a size-1 row dimension and a last stride that is not 1, so
// matrixOrder reads it as column-major and operandTranspose returns OP_T. The descriptor is in
// the swapped BLAS frame (C^T = B^T * A^T): its TRANSA is B's transpose and its TRANSB is A's.
// Fails if the transpose of a size-1 operand is derived from its raw strides.
TEST(TestMatmulParams, SizeOneRowOperandWithNonCanonicalStridesIsTransposed)
{
    const auto builder = createValidMatmulGraph({4, 1}, {1, 1}, {1, 5}, {99, 100}, {4, 5}, {5, 1});
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
    auto* attrs = graph.getNode(0).attributes_as_MatmulAttributes();
    ASSERT_NE(attrs, nullptr);

    const MatmulParams params(*attrs, graph.getTensorMap());

    EXPECT_EQ(getTransAttribute(params.desc(), HIPBLASLT_MATMUL_DESC_TRANSA), HIPBLAS_OP_T);
    EXPECT_EQ(getTransAttribute(params.desc(), HIPBLASLT_MATMUL_DESC_TRANSB), HIPBLAS_OP_N);
}

namespace
{
// Rewrites the strides of the tensor at `index` in the graph's tensor list to {1, rows}
// (column-major for a [rows, cols] tensor).
void makeTensorColumnMajor(flatbuffers::FlatBufferBuilder& builder,
                           flatbuffers::uoffset_t index,
                           int64_t rows)
{
    auto* graph = flatbuffers::GetMutableRoot<hipdnn_flatbuffers_sdk::data_objects::Graph>(
        builder.GetBufferPointer());
    auto* strides = graph->mutable_tensors()->GetMutableObject(index)->mutable_strides();
    strides->Mutate(0, 1);
    strides->Mutate(1, rows);
}
} // namespace

TEST(TestMatmulParams, ThrowsOnColumnMajorOutput)
{
    const std::vector<int64_t> aDims = {4, 8};
    const std::vector<int64_t> aStrides = {8, 1};
    const std::vector<int64_t> bDims = {8, 5};
    const std::vector<int64_t> bStrides = {5, 1};
    const std::vector<int64_t> cDims = {4, 5};
    const std::vector<int64_t> cStrides = {1, 4}; // Column-major output

    const auto builder = createValidMatmulGraph(aDims, aStrides, bDims, bStrides, cDims, cStrides);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    auto* attrs = graph.getNode(0).attributes_as_MatmulAttributes();
    ASSERT_NE(attrs, nullptr);

    EXPECT_THROW(MatmulParams(*attrs, graph.getTensorMap()), HipdnnPluginException);
}

// The final output of a fused graph supplies C's layout, so its orientation decides.
TEST(TestMatmulParams, ThrowsOnColumnMajorFusedFinalOutput)
{
    auto builder = createValidMatmulBiasActivGraph();
    // Tensor order: A, B, C_matmul, bias, C_bias, C (final output).
    makeTensorColumnMajor(builder, 5, 4);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    auto* matmulAttrs = graph.getNode(0).attributes_as_MatmulAttributes();
    auto* biasAttrs = graph.getNode(1).attributes_as_PointwiseAttributes();
    auto* activAttrs = graph.getNode(2).attributes_as_PointwiseAttributes();
    ASSERT_NE(matmulAttrs, nullptr);
    ASSERT_NE(biasAttrs, nullptr);
    ASSERT_NE(activAttrs, nullptr);

    EXPECT_THROW(MatmulParams(*matmulAttrs, biasAttrs, activAttrs, graph.getTensorMap()),
                 HipdnnPluginException);
}

// An intermediate that never reaches C/D does not constrain the output layout.
TEST(TestMatmulParams, FusedIntermediateLayoutDoesNotAffectRowMajorOutput)
{
    auto builder = createValidMatmulBiasActivGraph();
    makeTensorColumnMajor(builder, 2, 4); // C_matmul
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    auto* matmulAttrs = graph.getNode(0).attributes_as_MatmulAttributes();
    auto* biasAttrs = graph.getNode(1).attributes_as_PointwiseAttributes();
    auto* activAttrs = graph.getNode(2).attributes_as_PointwiseAttributes();
    ASSERT_NE(matmulAttrs, nullptr);
    ASSERT_NE(biasAttrs, nullptr);
    ASSERT_NE(activAttrs, nullptr);

    EXPECT_NO_THROW(MatmulParams(*matmulAttrs, biasAttrs, activAttrs, graph.getTensorMap()));
}

// A [M, 1] output is row-major whatever stride its size-1 dim carries.
TEST(TestMatmulParams, AcceptsSingleColumnOutputWithNonCanonicalStride)
{
    const std::vector<int64_t> aDims = {4, 8};
    const std::vector<int64_t> aStrides = {8, 1};
    const std::vector<int64_t> bDims = {8, 1};
    const std::vector<int64_t> bStrides = {1, 1};
    const std::vector<int64_t> cDims = {4, 1};
    const std::vector<int64_t> cStrides = {1, 4};

    const auto builder = createValidMatmulGraph(aDims, aStrides, bDims, bStrides, cDims, cStrides);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    auto* attrs = graph.getNode(0).attributes_as_MatmulAttributes();
    ASSERT_NE(attrs, nullptr);

    EXPECT_NO_THROW(MatmulParams(*attrs, graph.getTensorMap()));
}

TEST(TestMatmulParams, BatchBroadcastWithABatchOne)
{
    // [1, M, K] x [2, K, N] -> [2, M, N]
    const std::vector<int64_t> aDims = {1, 4, 8};
    const std::vector<int64_t> aStrides = {32, 8, 1};
    const std::vector<int64_t> bDims = {2, 8, 5};
    const std::vector<int64_t> bStrides = {40, 5, 1};
    const std::vector<int64_t> cDims = {2, 4, 5};
    const std::vector<int64_t> cStrides = {20, 5, 1};

    const auto builder = createValidMatmulGraph(aDims, aStrides, bDims, bStrides, cDims, cStrides);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_MatmulAttributes();
    ASSERT_NE(attrs, nullptr);

    // Should not throw for valid broadcast configuration
    EXPECT_NO_THROW(MatmulParams(*attrs, graph.getTensorMap()));
}

TEST(TestMatmulParams, BatchBroadcastWithBBatchOne)
{
    // [3, M, K] x [1, K, N] -> [3, M, N]
    const std::vector<int64_t> aDims = {3, 4, 8};
    const std::vector<int64_t> aStrides = {32, 8, 1};
    const std::vector<int64_t> bDims = {1, 8, 5};
    const std::vector<int64_t> bStrides = {40, 5, 1};
    const std::vector<int64_t> cDims = {3, 4, 5};
    const std::vector<int64_t> cStrides = {20, 5, 1};

    const auto builder = createValidMatmulGraph(aDims, aStrides, bDims, bStrides, cDims, cStrides);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_MatmulAttributes();
    ASSERT_NE(attrs, nullptr);

    // Should not throw for valid broadcast configuration
    EXPECT_NO_THROW(MatmulParams(*attrs, graph.getTensorMap()));
}

TEST(TestMatmulParams, ThrowsOnDifferentRanks)
{
    // [M, K] x [2, K, N] -> [2, M, N]
    const std::vector<int64_t> aDims = {4, 8};
    const std::vector<int64_t> aStrides = {8, 1};
    const std::vector<int64_t> bDims = {2, 8, 5};
    const std::vector<int64_t> bStrides = {40, 5, 1};
    const std::vector<int64_t> cDims = {2, 4, 5};
    const std::vector<int64_t> cStrides = {20, 5, 1};

    const auto builder = createValidMatmulGraph(aDims, aStrides, bDims, bStrides, cDims, cStrides);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_MatmulAttributes();
    ASSERT_NE(attrs, nullptr);

    // Should throw due to mismatched ranks
    EXPECT_THROW(MatmulParams(*attrs, graph.getTensorMap()), HipdnnPluginException);
}

TEST(TestMatmulParams, ThrowsOnIncompatibleBatchDimensions)
{
    // [2, M, K] x [3, K, N] -> [3, M, N]
    const std::vector<int64_t> aDims = {2, 4, 8};
    const std::vector<int64_t> aStrides = {32, 8, 1};
    const std::vector<int64_t> bDims = {3, 8, 5};
    const std::vector<int64_t> bStrides = {40, 5, 1};
    const std::vector<int64_t> cDims = {3, 4, 5};
    const std::vector<int64_t> cStrides = {20, 5, 1};

    const auto builder = createValidMatmulGraph(aDims, aStrides, bDims, bStrides, cDims, cStrides);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_MatmulAttributes();
    ASSERT_NE(attrs, nullptr);

    // Should throw due to incompatible batch dimensions
    EXPECT_THROW(MatmulParams(*attrs, graph.getTensorMap()), HipdnnPluginException);
}

TEST(TestMatmulParams, ThrowsOnOutputBatchMismatch)
{
    // [2, M, K] x [2, K, N] -> [3, M, N]
    const std::vector<int64_t> aDims = {2, 4, 8};
    const std::vector<int64_t> aStrides = {32, 8, 1};
    const std::vector<int64_t> bDims = {2, 8, 5};
    const std::vector<int64_t> bStrides = {40, 5, 1};
    const std::vector<int64_t> cDims = {3, 4, 5};
    const std::vector<int64_t> cStrides = {20, 5, 1};

    const auto builder = createValidMatmulGraph(aDims, aStrides, bDims, bStrides, cDims, cStrides);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_MatmulAttributes();
    ASSERT_NE(attrs, nullptr);

    // Should throw due to output batch mismatch
    EXPECT_THROW(MatmulParams(*attrs, graph.getTensorMap()), HipdnnPluginException);
}

TEST(TestMatmulParams, ThrowsOnOutputBatchSmallerThanMax)
{
    // [1, M, K] x [3, K, N] -> [2, M, N]
    const std::vector<int64_t> aDims = {1, 4, 8};
    const std::vector<int64_t> aStrides = {32, 8, 1};
    const std::vector<int64_t> bDims = {3, 8, 5};
    const std::vector<int64_t> bStrides = {40, 5, 1};
    const std::vector<int64_t> cDims = {2, 4, 5};
    const std::vector<int64_t> cStrides = {20, 5, 1};

    const auto builder = createValidMatmulGraph(aDims, aStrides, bDims, bStrides, cDims, cStrides);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_MatmulAttributes();
    ASSERT_NE(attrs, nullptr);

    EXPECT_THROW(MatmulParams(*attrs, graph.getTensorMap()), HipdnnPluginException);
}

TEST(TestMatmulParams, Tensor4DBroadcastWithUnitBatchB)
{
    // [3, 3, m, k] x [1, 1, k, n] -> [3, 3, m, n]
    const std::vector<int64_t> aDims = {3, 3, 4, 8};
    const std::vector<int64_t> aStrides = {96, 32, 8, 1};
    const std::vector<int64_t> bDims = {1, 1, 8, 5};
    const std::vector<int64_t> bStrides = {40, 40, 5, 1};
    const std::vector<int64_t> cDims = {3, 3, 4, 5};
    const std::vector<int64_t> cStrides = {60, 20, 5, 1};

    const auto builder = createValidMatmulGraph(aDims, aStrides, bDims, bStrides, cDims, cStrides);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_MatmulAttributes();
    ASSERT_NE(attrs, nullptr);

    // Should not throw - B batch is all 1s, broadcasting is valid
    EXPECT_NO_THROW(MatmulParams(*attrs, graph.getTensorMap()));
}

TEST(TestMatmulParams, Tensor4DThrowsOnIncompatibleBatch)
{
    // [3, 3, m, k] x [1, 3, k, n] -> [3, 3, m, n]
    const std::vector<int64_t> aDims = {3, 3, 4, 8};
    const std::vector<int64_t> aStrides = {96, 32, 8, 1};
    const std::vector<int64_t> bDims = {1, 3, 8, 5};
    const std::vector<int64_t> bStrides = {120, 40, 5, 1};
    const std::vector<int64_t> cDims = {3, 3, 4, 5};
    const std::vector<int64_t> cStrides = {60, 20, 5, 1};

    const auto builder = createValidMatmulGraph(aDims, aStrides, bDims, bStrides, cDims, cStrides);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_MatmulAttributes();
    ASSERT_NE(attrs, nullptr);

    // Should throw - batch dimensions are incompatible (9 vs 3, neither is 1)
    EXPECT_THROW(MatmulParams(*attrs, graph.getTensorMap()), HipdnnPluginException);
}

TEST(TestMatmulParams, Tensor4DThrowsOnOutputBatchMismatch)
{
    // [3, 1, m, k] x [1, 3, k, n] -> [3, 3, m, n]
    const std::vector<int64_t> aDims = {3, 1, 4, 8};
    const std::vector<int64_t> aStrides = {32, 32, 8, 1};
    const std::vector<int64_t> bDims = {1, 3, 8, 5};
    const std::vector<int64_t> bStrides = {120, 40, 5, 1};
    const std::vector<int64_t> cDims = {3, 3, 4, 5};
    const std::vector<int64_t> cStrides = {60, 20, 5, 1};

    const auto builder = createValidMatmulGraph(aDims, aStrides, bDims, bStrides, cDims, cStrides);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_MatmulAttributes();
    ASSERT_NE(attrs, nullptr);

    // Should throw - strided input batches
    EXPECT_THROW(MatmulParams(*attrs, graph.getTensorMap()), HipdnnPluginException);
}

namespace
{
int64_t stridedBatchOffset(const HipblasltMatrixLayout& layout)
{
    int64_t offset = -1;
    size_t written = 0;
    EXPECT_EQ(hipblasLtMatrixLayoutGetAttribute(layout.matrixLayout(),
                                                HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                                &offset,
                                                sizeof(offset),
                                                &written),
              HIPBLAS_STATUS_SUCCESS);
    EXPECT_EQ(written, sizeof(offset));
    return offset;
}
} // namespace

// A batch axis of extent one never advances, so the C offset is the stride of the next
// batch axis with extent above one.
TEST(TestMatmulParams, CLayoutBatchOffsetIgnoresSingletonBatchAxis)
{
    // The virtual matmul output is packed; the final output keeps a stride of 5 on the
    // singleton axis.
    const auto fused = createValidMatmulBiasActivGraph(
        {2, 1, 4, 8},
        {32, 32, 8, 1},
        {2, 1, 8, 5},
        {40, 40, 5, 1},
        {2, 1, 4, 5},
        {20, 5, 5, 1},
        false,
        hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RELU_FWD);
    const auto builder = mutateGraph(fused, [](auto& g) {
        for(const auto& tensor : g.tensors)
        {
            if(tensor->name == "C_matmul")
            {
                tensor->strides = {20, 20, 5, 1};
            }
        }
    });
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    auto* matmulAttrs = graph.getNode(0).attributes_as_MatmulAttributes();
    auto* activAttrs = graph.getNode(1).attributes_as_PointwiseAttributes();
    ASSERT_NE(matmulAttrs, nullptr);
    ASSERT_NE(activAttrs, nullptr);

    std::optional<MatmulParams> params;
    ASSERT_NO_THROW(params.emplace(*matmulAttrs, nullptr, activAttrs, graph.getTensorMap()));
    EXPECT_EQ(stridedBatchOffset(params->c()), 20);
}

// A padded final output keeps its own batch stride: C's offset comes from the final output,
// not from the packed virtual matmul output.
TEST(TestMatmulParams, CLayoutBatchOffsetTakesFinalOutputStrideWhenPadded)
{
    const auto fused = createValidMatmulBiasActivGraph(
        {2, 4, 8},
        {32, 8, 1},
        {2, 8, 5},
        {40, 5, 1},
        {2, 4, 5},
        {20, 5, 1},
        false,
        hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RELU_FWD);
    const auto builder = mutateGraph(fused, [](auto& g) {
        for(const auto& tensor : g.tensors)
        {
            if(tensor->name == "C")
            {
                tensor->strides = {27, 5, 1};
            }
        }
    });
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    auto* matmulAttrs = graph.getNode(0).attributes_as_MatmulAttributes();
    auto* activAttrs = graph.getNode(1).attributes_as_PointwiseAttributes();
    ASSERT_NE(matmulAttrs, nullptr);
    ASSERT_NE(activAttrs, nullptr);

    std::optional<MatmulParams> params;
    ASSERT_NO_THROW(params.emplace(*matmulAttrs, nullptr, activAttrs, graph.getTensorMap()));
    EXPECT_EQ(stridedBatchOffset(params->c()), 27);
}

// Fails if a fused final output whose batch stride (10) is smaller than one output matrix
// (20 elements, so batch 0 row 2 and batch 1 row 0 share an address) is accepted just because
// the virtual matmul output is packed.
TEST(TestMatmulParams, ThrowsOnFusedFinalOutputBatchStrideBelowMatrixSpan)
{
    const auto fused = createValidMatmulBiasActivGraph(
        {2, 4, 8},
        {32, 8, 1},
        {2, 8, 5},
        {40, 5, 1},
        {2, 4, 5},
        {20, 5, 1},
        false,
        hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RELU_FWD);
    const auto builder = mutateGraph(fused, [](auto& g) {
        for(const auto& tensor : g.tensors)
        {
            if(tensor->name == "C")
            {
                tensor->strides = {10, 5, 1};
            }
        }
    });
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    auto* matmulAttrs = graph.getNode(0).attributes_as_MatmulAttributes();
    auto* activAttrs = graph.getNode(1).attributes_as_PointwiseAttributes();
    ASSERT_NE(matmulAttrs, nullptr);
    ASSERT_NE(activAttrs, nullptr);

    EXPECT_THROW(MatmulParams(*matmulAttrs, nullptr, activAttrs, graph.getTensorMap()),
                 HipdnnPluginException);
}

// Fails if a plain matmul output whose batch stride is smaller than one output matrix, or zero,
// is accepted, or if a stride exactly equal to the matrix span ([2, 4, 5] with row stride 5
// spans 20 elements) is declined. The padded case is covered by
// CLayoutBatchOffsetTakesFinalOutputStrideWhenPadded.
TEST(TestMatmulParams, PlainOutputBatchStrideMustCoverOneMatrix)
{
    const auto makeGraph = [](const std::vector<int64_t>& cStrides) {
        return createValidMatmulGraph(
            {2, 4, 8}, {32, 8, 1}, {2, 8, 5}, {40, 5, 1}, {2, 4, 5}, cStrides);
    };

    for(const auto& cStrides : {std::vector<int64_t>{10, 5, 1}, std::vector<int64_t>{0, 5, 1}})
    {
        const auto builder = makeGraph(cStrides);
        const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
        auto* attrs = graph.getNode(0).attributes_as_MatmulAttributes();
        ASSERT_NE(attrs, nullptr);

        EXPECT_THROW(MatmulParams(*attrs, graph.getTensorMap()), HipdnnPluginException)
            << "batch stride " << cStrides[0];
    }

    const auto builder = makeGraph({20, 5, 1});
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
    auto* attrs = graph.getNode(0).attributes_as_MatmulAttributes();
    ASSERT_NE(attrs, nullptr);

    std::optional<MatmulParams> params;
    ASSERT_NO_THROW(params.emplace(*attrs, graph.getTensorMap()));
    EXPECT_EQ(stridedBatchOffset(params->c()), 20);
}

TEST(TestMatmulParams, ALayoutBatchOffsetIgnoresSingletonBatchAxis)
{
    const auto builder = createValidMatmulGraph(
        {4, 1, 4, 8}, {32, 7, 8, 1}, {4, 1, 8, 5}, {40, 40, 5, 1}, {4, 1, 4, 5}, {20, 20, 5, 1});
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    auto* attrs = graph.getNode(0).attributes_as_MatmulAttributes();
    ASSERT_NE(attrs, nullptr);

    std::optional<MatmulParams> params;
    ASSERT_NO_THROW(params.emplace(*attrs, graph.getTensorMap()));
    EXPECT_EQ(stridedBatchOffset(params->a()), 32);
}

// With three batch axes the outer strides must account for every axis inside them.
TEST(TestMatmulParams, ALayoutBatchOffsetNestsThreeBatchAxes)
{
    const auto makeGraph = [](const std::vector<int64_t>& aStrides) {
        return createValidMatmulGraph({2, 2, 3, 4, 8},
                                      aStrides,
                                      {2, 2, 3, 8, 5},
                                      {480, 240, 80, 5, 1},
                                      {2, 2, 3, 4, 5},
                                      {240, 120, 40, 5, 1});
    };

    {
        const auto builder = makeGraph({192, 96, 32, 8, 1});
        const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
        auto* attrs = graph.getNode(0).attributes_as_MatmulAttributes();
        ASSERT_NE(attrs, nullptr);

        std::optional<MatmulParams> params;
        ASSERT_NO_THROW(params.emplace(*attrs, graph.getTensorMap()));
        EXPECT_EQ(stridedBatchOffset(params->a()), 32);
    }
    {
        const auto builder = makeGraph({200, 96, 32, 8, 1});
        const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
        auto* attrs = graph.getNode(0).attributes_as_MatmulAttributes();
        ASSERT_NE(attrs, nullptr);

        try
        {
            const MatmulParams params(*attrs, graph.getTensorMap());
            FAIL() << "non-nesting outer batch stride must throw";
        }
        catch(const HipdnnPluginException& e)
        {
            EXPECT_NE(std::string(e.what()).find("are not uniform"), std::string::npos) << e.what();
        }
    }
}

TEST_F(TestGpuMatmulPlan, CreatesPlanWithValidGraph)
{
    // Create a valid matmul graph
    const auto builder = createValidMatmulGraph();
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

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
    const auto builder = createValidMatmulGraph();
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_MatmulAttributes();
    ASSERT_NE(attrs, nullptr);

    MatmulParams params(*attrs, graph.getTensorMap());
    const MatmulPlan plan(_handle, std::move(params));

    // Workspace size should be >= 0
    EXPECT_GE(plan.getWorkspaceSize(_handle), 0u);
}

TEST_F(TestGpuMatmulPlan, CreatesPlanWithColumnMajorInputs)
{
    // Column-major strides
    const std::vector<int64_t> aDims = {4, 8};
    const std::vector<int64_t> aStrides = {1, 4}; // Column-major
    const std::vector<int64_t> bDims = {8, 5};
    const std::vector<int64_t> bStrides = {1, 8}; // Column-major
    const std::vector<int64_t> cDims = {4, 5};
    const std::vector<int64_t> cStrides = {5, 1};

    const auto builder = createValidMatmulGraph(aDims, aStrides, bDims, bStrides, cDims, cStrides);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

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
    const std::vector<int64_t> aDims = {64, 128};
    const std::vector<int64_t> aStrides = {128, 1};
    const std::vector<int64_t> bDims = {128, 256};
    const std::vector<int64_t> bStrides = {256, 1};
    const std::vector<int64_t> cDims = {64, 256};
    const std::vector<int64_t> cStrides = {256, 1};

    const auto builder = createValidMatmulGraph(aDims, aStrides, bDims, bStrides, cDims, cStrides);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

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
    const std::vector<int64_t> aDims = {2, 4, 8};
    const std::vector<int64_t> aStrides = {32, 8, 1};
    const std::vector<int64_t> bDims = {2, 8, 5};
    const std::vector<int64_t> bStrides = {40, 5, 1};
    const std::vector<int64_t> cDims = {2, 4, 5};
    const std::vector<int64_t> cStrides = {20, 5, 1};

    const auto builder = createValidMatmulGraph(aDims, aStrides, bDims, bStrides, cDims, cStrides);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

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
    const std::vector<int64_t> aDims = {16, 32};
    const std::vector<int64_t> aStrides = {32, 1};
    const std::vector<int64_t> bDims = {32, 64};
    const std::vector<int64_t> bStrides = {64, 1};
    const std::vector<int64_t> cDims = {16, 64};
    const std::vector<int64_t> cStrides = {64, 1};

    const auto builder
        = createValidMatmulGraph(aDims,
                                 aStrides,
                                 bDims,
                                 bStrides,
                                 cDims,
                                 cStrides,
                                 hipdnn_flatbuffers_sdk::data_objects::DataType::HALF);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

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
    const std::vector<int64_t> aDims = {16, 32};
    const std::vector<int64_t> aStrides = {32, 1};
    const std::vector<int64_t> bDims = {32, 64};
    const std::vector<int64_t> bStrides = {64, 1};
    const std::vector<int64_t> cDims = {16, 64};
    const std::vector<int64_t> cStrides = {64, 1};

    const auto builder
        = createValidMatmulGraph(aDims,
                                 aStrides,
                                 bDims,
                                 bStrides,
                                 cDims,
                                 cStrides,
                                 hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_MatmulAttributes();
    ASSERT_NE(attrs, nullptr);

    MatmulParams params(*attrs, graph.getTensorMap());

    // Create plan - should not throw
    EXPECT_NO_THROW(MatmulPlan(_handle, std::move(params)));
}

TEST_F(TestGpuMatmulPlan, CreatesPlanWithBiasAndGelu)
{
    const auto builder = createValidMatmulBiasActivGraph(
        {4, 8},
        {8, 1},
        {8, 5},
        {5, 1},
        {4, 5},
        {5, 1},
        true,
        hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::GELU_APPROX_TANH_FWD);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

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
    const auto builder = createValidMatmulBiasActivGraph(
        {4, 8},
        {8, 1},
        {8, 5},
        {5, 1},
        {4, 5},
        {5, 1},
        true,
        hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RELU_FWD,
        0.0f);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

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
    const auto builder = createValidMatmulBiasActivGraph(
        {4, 8},
        {8, 1},
        {8, 5},
        {5, 1},
        {4, 5},
        {5, 1},
        false,
        hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RELU_FWD,
        0.0f);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    auto* matmulAttrs = graph.getNode(0).attributes_as_MatmulAttributes();
    auto* activAttrs = graph.getNode(1).attributes_as_PointwiseAttributes();
    ASSERT_NE(matmulAttrs, nullptr);
    ASSERT_NE(activAttrs, nullptr);

    MatmulParams params(*matmulAttrs, nullptr, activAttrs, graph.getTensorMap());

    EXPECT_NO_THROW(MatmulPlan(_handle, std::move(params)));
}

TEST_F(TestGpuMatmulPlan, CreatesPlanWithBiasAndSwish)
{
    const auto builder = createValidMatmulBiasActivGraph(
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
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

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
    const auto builder = createValidMatmulBiasActivGraph(
        {4, 8},
        {8, 1},
        {8, 5},
        {5, 1},
        {4, 5},
        {5, 1},
        true,
        hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::UNSET);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    auto* matmulAttrs = graph.getNode(0).attributes_as_MatmulAttributes();
    auto* biasAttrs = graph.getNode(1).attributes_as_PointwiseAttributes();
    ASSERT_NE(matmulAttrs, nullptr);
    ASSERT_NE(biasAttrs, nullptr);

    MatmulParams params(*matmulAttrs, biasAttrs, nullptr, graph.getTensorMap());

    EXPECT_NO_THROW(MatmulPlan(_handle, std::move(params)));
}

TEST_F(TestGpuMatmulPlan, PlanReturnValidWorkspaceSizeWithBiasOnly)
{
    const auto builder = createValidMatmulBiasActivGraph(
        {4, 8},
        {8, 1},
        {8, 5},
        {5, 1},
        {4, 5},
        {5, 1},
        true,
        hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::UNSET);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    auto* matmulAttrs = graph.getNode(0).attributes_as_MatmulAttributes();
    auto* biasAttrs = graph.getNode(1).attributes_as_PointwiseAttributes();
    ASSERT_NE(matmulAttrs, nullptr);
    ASSERT_NE(biasAttrs, nullptr);

    MatmulParams params(*matmulAttrs, biasAttrs, nullptr, graph.getTensorMap());
    const MatmulPlan plan(_handle, std::move(params));

    EXPECT_GE(plan.getWorkspaceSize(_handle), 0u);
}

TEST_F(TestGpuMatmulPlan, PlanReturnValidWorkspaceSizeWithBiasActivation)
{
    const auto builder = createValidMatmulBiasActivGraph(
        {4, 8},
        {8, 1},
        {8, 5},
        {5, 1},
        {4, 5},
        {5, 1},
        true,
        hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::GELU_APPROX_TANH_FWD);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    auto* matmulAttrs = graph.getNode(0).attributes_as_MatmulAttributes();
    auto* biasAttrs = graph.getNode(1).attributes_as_PointwiseAttributes();
    auto* activAttrs = graph.getNode(2).attributes_as_PointwiseAttributes();
    ASSERT_NE(matmulAttrs, nullptr);
    ASSERT_NE(biasAttrs, nullptr);
    ASSERT_NE(activAttrs, nullptr);

    MatmulParams params(*matmulAttrs, biasAttrs, activAttrs, graph.getTensorMap());
    const MatmulPlan plan(_handle, std::move(params));

    EXPECT_GE(plan.getWorkspaceSize(_handle), 0u);
}
