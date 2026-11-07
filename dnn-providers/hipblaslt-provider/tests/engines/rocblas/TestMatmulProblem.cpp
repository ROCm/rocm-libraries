// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <functional>
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

TEST(TestMatmulProblem, WideBatchStridesSplitIntoOneCallPerMatrix)
{
    namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
    GemmCall narrow;
    narrow.m = narrow.n = narrow.k = 8;
    narrow.batchCount = 3;
    for(auto* operand : {&narrow.a, &narrow.b, &narrow.d})
    {
        operand->dataType = data_objects::DataType::FLOAT;
        operand->leadingDimension = 8;
        operand->batchStride = 64;
    }
    GemmCall wide = narrow;
    wide.a.batchStride = MAX_BATCH_STRIDE + 1; // an image past 2^31 elements of the next
    wide.b.batchStride = 0; // a broadcast operand stays put
    wide.d.offset = 5;
    wide.accumulate = true;

    const auto split = splitWideBatches({narrow, wide});
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

} // namespace
