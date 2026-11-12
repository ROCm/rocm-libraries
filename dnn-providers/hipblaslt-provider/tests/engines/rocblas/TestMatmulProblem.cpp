// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <functional>
#include <limits>
#include <string>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

#include "common/GemmFormulation.hpp"
#include "common/MatmulProblem.hpp"
#include "engines/rocblas/MatmulTestGraphs.hpp"

/**
 * @file TestMatmulProblem.cpp
 * @brief The backend-neutral GEMM a matmul graph resolves to: the column-major call for every
 *        storage order, batching and broadcast, and one refusal per gated field.
 */
namespace
{

using namespace hipblaslt_plugin;
using namespace hipblaslt_plugin::test;

std::optional<GemmCall> resolve(const MatmulGraphSpec& spec)
{
    const MatmulGraph graph(spec);
    return gemmForMatmul(graph.graph());
}

// ---------------------------------------------------------------------------
// Storage orders
// ---------------------------------------------------------------------------

class TestMatmulProblemLayouts : public ::testing::TestWithParam<std::tuple<bool, bool, bool>>
{
};

TEST_P(TestMatmulProblemLayouts, ResolvesTheColumnMajorCall)
{
    const auto [aColumnMajor, bColumnMajor, cColumnMajor] = GetParam();
    constexpr int64_t M = 5;
    constexpr int64_t N = 7;
    constexpr int64_t K = 3;
    const auto call = resolve(matmulSpec(M, N, K, {}, aColumnMajor, bColumnMajor, cColumnMajor));
    ASSERT_TRUE(call.has_value());
    EXPECT_EQ(call->k, K);
    EXPECT_EQ(call->batchCount, 1);

    if(cColumnMajor)
    {
        // c = a b, with each operand transposed when it is stored row-major.
        EXPECT_EQ(call->m, M);
        EXPECT_EQ(call->n, N);
        EXPECT_EQ(call->a.uid, A_UID);
        EXPECT_EQ(call->b.uid, B_UID);
        EXPECT_EQ(call->transposeA, !aColumnMajor);
        EXPECT_EQ(call->transposeB, !bColumnMajor);
        EXPECT_EQ(call->a.leadingDimension, aColumnMajor ? M : K);
        EXPECT_EQ(call->b.leadingDimension, bColumnMajor ? K : N);
        EXPECT_EQ(call->d.leadingDimension, M);
    }
    else
    {
        // c^T = b^T a^T, so b becomes the call's first operand.
        EXPECT_EQ(call->m, N);
        EXPECT_EQ(call->n, M);
        EXPECT_EQ(call->a.uid, B_UID);
        EXPECT_EQ(call->b.uid, A_UID);
        EXPECT_EQ(call->transposeA, bColumnMajor);
        EXPECT_EQ(call->transposeB, aColumnMajor);
        EXPECT_EQ(call->a.leadingDimension, bColumnMajor ? K : N);
        EXPECT_EQ(call->b.leadingDimension, aColumnMajor ? M : K);
        EXPECT_EQ(call->d.leadingDimension, N);
    }
    EXPECT_EQ(call->d.uid, C_UID);
}

INSTANTIATE_TEST_SUITE_P(,
                         TestMatmulProblemLayouts,
                         ::testing::Combine(::testing::Bool(),
                                            ::testing::Bool(),
                                            ::testing::Bool()));

// ---------------------------------------------------------------------------
// Batching
// ---------------------------------------------------------------------------

TEST(TestMatmulProblem, CollapsesNestedBatchAxesIntoOneStride)
{
    const auto call = resolve(matmulSpec(4, 6, 5, {2, 3}, false, false, false));
    ASSERT_TRUE(call.has_value());
    EXPECT_EQ(call->batchCount, 6);
    EXPECT_EQ(call->b.batchStride, 4 * 5); // a, the call's second operand
    EXPECT_EQ(call->a.batchStride, 5 * 6); // b
    EXPECT_EQ(call->d.batchStride, 4 * 6);
}

TEST(TestMatmulProblem, BroadcastsAnOperandWhoseBatchAxesAreAllOne)
{
    const auto broadcastA = resolve(matmulSpec(4, 6, 5, {3}, false, false, false, true, false));
    ASSERT_TRUE(broadcastA.has_value());
    EXPECT_EQ(broadcastA->batchCount, 3);
    EXPECT_EQ(broadcastA->b.batchStride, 0);
    EXPECT_NE(broadcastA->a.batchStride, 0);

    const auto broadcastB = resolve(matmulSpec(4, 6, 5, {3}, false, false, false, false, true));
    ASSERT_TRUE(broadcastB.has_value());
    EXPECT_EQ(broadcastB->a.batchStride, 0);
}

TEST(TestMatmulProblem, AcceptsAHalfOutputForHalfInputsAndAFloatOutput)
{
    auto spec = matmulSpec(4, 6, 5, {}, false, false, false);
    spec.a.dataType = spec.b.dataType = spec.c.dataType = data_objects::DataType::HALF;
    EXPECT_TRUE(resolve(spec).has_value());
    spec.c.dataType = data_objects::DataType::FLOAT;
    EXPECT_TRUE(resolve(spec).has_value());
    spec.a.dataType = spec.b.dataType = data_objects::DataType::BFLOAT16;
    EXPECT_TRUE(resolve(spec).has_value());
}

// ---------------------------------------------------------------------------
// One refusal per gated field
// ---------------------------------------------------------------------------

struct RefusalCase
{
    std::string name;
    std::function<void(MatmulGraphSpec&)> mutate;
};

std::vector<RefusalCase> refusalCases()
{
    using data_objects::DataType;
    return {
        {"OverrideShapeEnabled", [](MatmulGraphSpec& s) { s.overrideShapeEnabled = true; }},
        {"ComputeTypeHalf", [](MatmulGraphSpec& s) { s.computeType = DataType::HALF; }},
        {"OutputAliasesInput", [](MatmulGraphSpec& s) { s.cUid = A_UID; }},
        {"VirtualOperand", [](MatmulGraphSpec& s) { s.b.isVirtual = true; }},
        {"RuntimePassByValue", [](MatmulGraphSpec& s) { s.a.runtimePassByValue = true; }},
        {"Int8",
         [](MatmulGraphSpec& s) { s.a.dataType = s.b.dataType = s.c.dataType = DataType::INT8; }},
        {"InputsDiffer", [](MatmulGraphSpec& s) { s.b.dataType = DataType::HALF; }},
        {"NarrowerOutput", [](MatmulGraphSpec& s) { s.c.dataType = DataType::HALF; }},
        {"RanksDiffer",
         [](MatmulGraphSpec& s) {
             s.c.dims.insert(s.c.dims.begin(), 1);
             s.c.strides.insert(s.c.strides.begin(), 0);
         }},
        {"InnerExtentsDisagree",
         [](MatmulGraphSpec& s) {
             s.b.dims[s.b.dims.size() - 2] += 1;
             s.b.strides = matrixStrides(s.b.dims, false);
         }},
        {"OutputExtentWrong",
         [](MatmulGraphSpec& s) {
             s.c.dims.back() += 1;
             s.c.strides = matrixStrides(s.c.dims, false);
         }},
        {"NoUnitStride",
         [](MatmulGraphSpec& s) {
             s.a.strides[s.a.strides.size() - 1] = 2;
             s.a.strides[s.a.strides.size() - 2] = 2 * s.a.dims.back() * 2;
         }},
        {"NegativeStride", [](MatmulGraphSpec& s) { s.c.strides[0] = -1; }},
        {"BatchAxesDisagree",
         [](MatmulGraphSpec& s) {
             s.a.dims[0] = 2; // c has 3
             s.a.strides = matrixStrides(s.a.dims, false);
         }},
        {"BatchAxesDoNotNest",
         [](MatmulGraphSpec& s) {
             // Batch [3, 2]: the outer stride is not 2 x the inner.
             s.a.dims = {3, 2, 4, 5};
             s.b.dims = {3, 2, 5, 6};
             s.c.dims = {3, 2, 4, 6};
             s.a.strides = matrixStrides(s.a.dims, false);
             s.b.strides = matrixStrides(s.b.dims, false);
             s.c.strides = matrixStrides(s.c.dims, false);
             s.a.strides[0] += 1;
         }},
        {"OutputMatricesOverlap", [](MatmulGraphSpec& s) { s.c.strides[0] = 1; }},
    };
}

class TestMatmulProblemRefuses : public ::testing::TestWithParam<RefusalCase>
{
};

TEST_P(TestMatmulProblemRefuses, OneField)
{
    auto spec = matmulSpec(4, 6, 5, {3}, false, false, false);
    ASSERT_TRUE(resolve(spec).has_value());
    GetParam().mutate(spec);
    EXPECT_FALSE(resolve(spec).has_value());
}

INSTANTIATE_TEST_SUITE_P(,
                         TestMatmulProblemRefuses,
                         ::testing::ValuesIn(refusalCases()),
                         [](const auto& info) { return info.param.name; });

/// An 8 x 8 x 8 fp32 call over three matrices 64 elements apart.
GemmCall smallCall()
{
    GemmCall call;
    call.m = call.n = call.k = 8;
    call.batchCount = 3;
    for(auto* operand : {&call.a, &call.b, &call.d})
    {
        operand->dataType = hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT;
        operand->leadingDimension = 8;
        operand->batchStride = 64;
    }
    return call;
}

TEST(TestMatmulProblem, WideBatchStridesSplitIntoOneCallPerMatrix)
{
    const GemmCall narrow = smallCall();
    GemmCall wide = smallCall();
    wide.a.batchStride = MAX_BATCH_STRIDE + 1; // an image past 2^31 elements of the next
    wide.b.batchStride = 0; // a broadcast operand stays put
    wide.d.offset = 5;
    wide.accumulate = true;

    const auto split = fitLibraryLimits({narrow, wide});
    ASSERT_EQ(split.size(), 4u);
    EXPECT_EQ(split[0].batchCount, 3); // within the limit: untouched
    EXPECT_EQ(split[0].a.batchStride, 64);
    for(int64_t matrix = 0; matrix < 3; ++matrix)
    {
        const auto& one = split[static_cast<size_t>(1 + matrix)];
        EXPECT_EQ(one.batchCount, 1);
        EXPECT_EQ(one.a.offset, matrix * (MAX_BATCH_STRIDE + 1));
        EXPECT_EQ(one.b.offset, 0);
        EXPECT_EQ(one.d.offset, 5 + matrix * 64);
        EXPECT_EQ(one.a.batchStride, 0);
        EXPECT_EQ(one.d.batchStride, 0);
        EXPECT_TRUE(one.accumulate);
        EXPECT_EQ(one.m, 8);
    }
}

TEST(TestMatmulProblem, LongExtentsSplitIntoChunks)
{
    // n past the limit, b stored k x n: columns of b and of d.
    GemmCall wide = smallCall();
    wide.batchCount = 1;
    wide.n = 2 * MAX_GEMM_EXTENT;
    auto split = fitLibraryLimits({wide});
    ASSERT_EQ(split.size(), 2u);
    EXPECT_EQ(split[1].b.offset, MAX_GEMM_EXTENT * 8);
    EXPECT_EQ(split[1].d.offset, MAX_GEMM_EXTENT * 8);
    EXPECT_EQ(split[1].a.offset, 0);
    EXPECT_FALSE(split[1].accumulate); // disjoint columns of d

    // k past the limit into fp32: the later chunk adds to the first.
    GemmCall deep = smallCall();
    deep.batchCount = 1;
    deep.k = MAX_GEMM_EXTENT + 1;
    split = fitLibraryLimits({deep});
    ASSERT_EQ(split.size(), 2u);
    EXPECT_FALSE(split[0].accumulate);
    EXPECT_TRUE(split[1].accumulate);
    EXPECT_EQ(split[1].k, 1);
    EXPECT_EQ(split[1].a.offset, MAX_GEMM_EXTENT * 8); // a stored m x k: k walks columns
    EXPECT_EQ(split[1].b.offset, MAX_GEMM_EXTENT); // b stored k x n: k walks rows
    EXPECT_EQ(split[1].d.offset, 0);
}

TEST(TestMatmulProblem, RocblasChunkedReductionsIntoNarrowOutputsAreFlagged)
{
    namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
    // bf16, a stored m x k (not transposed): k walks lda. rocBLAS's chunk is the largest k with
    // k * lda * 2 bytes under 2^31.
    GemmCall call = smallCall();
    call.batchCount = 1;
    for(auto* operand : {&call.a, &call.b, &call.d})
    {
        operand->dataType = data_objects::DataType::BFLOAT16;
    }
    const int64_t lda = 8193 * 8193;
    call.a.leadingDimension = lda;
    const int64_t chunk = ROCBLAS_REDUCTION_BYTES / (lda * 2);
    ASSERT_EQ(chunk, 15);
    call.k = chunk;
    EXPECT_FALSE(rocblasRoundsReductionChunks(call));
    call.k = chunk + 1; // the 16-channel backward data of one 8193 x 8193 image
    EXPECT_TRUE(rocblasRoundsReductionChunks(call));
    EXPECT_GT(call.k * call.a.leadingDimension * 2, ROCBLAS_REDUCTION_BYTES);

    // An fp32 d sums the chunks in fp32: one rounding per chunk at fp32, as a split k would.
    GemmCall fp32Out = call;
    fp32Out.d.dataType = data_objects::DataType::FLOAT;
    EXPECT_FALSE(rocblasRoundsReductionChunks(fp32Out));

    // a transposed: k walks unit stride, whatever lda.
    GemmCall unitStride = call;
    unitStride.transposeA = true;
    EXPECT_FALSE(rocblasRoundsReductionChunks(unitStride));

    // b transposed: k walks ldb.
    GemmCall viaB = call;
    viaB.a.leadingDimension = 8;
    EXPECT_FALSE(rocblasRoundsReductionChunks(viaB));
    viaB.transposeB = true;
    viaB.b.leadingDimension = lda;
    EXPECT_TRUE(rocblasRoundsReductionChunks(viaB));

    // fp16 the same; an overflowing stride flags any k past 1.
    GemmCall half = call;
    half.a.dataType = half.b.dataType = half.d.dataType = data_objects::DataType::HALF;
    EXPECT_TRUE(rocblasRoundsReductionChunks(half));
    half.a.leadingDimension = std::numeric_limits<int64_t>::max();
    half.k = 1;
    EXPECT_FALSE(rocblasRoundsReductionChunks(half));
    half.k = 2;
    EXPECT_TRUE(rocblasRoundsReductionChunks(half));
}

} // namespace
