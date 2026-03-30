// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocm_ck/resolve.hpp>

#include <gtest/gtest.h>

using namespace rocm_ck;

// ============================================================================
// Simple GemmOp resolution
// ============================================================================

TEST(Resolve, ResolvesSimpleGemmToThreeTensors)
{
    constexpr auto r = resolve(
        Signature{.dtype = DataType::FP16, .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}});

    EXPECT_EQ(r.num_tensors, 3);
}

TEST(Resolve, CascadesSignatureDtypeToAllGemmTensors)
{
    constexpr auto r = resolve(
        Signature{.dtype = DataType::FP16, .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}});

    EXPECT_EQ(r.tensor("A").dtype, DataType::FP16);
    EXPECT_EQ(r.tensor("B").dtype, DataType::FP16);
    EXPECT_EQ(r.tensor("C").dtype, DataType::FP16);
}

TEST(Resolve, AssignsRank2ToGemmTensors)
{
    constexpr auto r = resolve(
        Signature{.dtype = DataType::FP16, .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}});

    EXPECT_EQ(r.tensor("A").rank, 2);
    EXPECT_EQ(r.tensor("B").rank, 2);
    EXPECT_EQ(r.tensor("C").rank, 2);
}

TEST(Resolve, AssignsRowColRowLayoutToGemmTensors)
{
    constexpr auto r = resolve(
        Signature{.dtype = DataType::FP16, .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}});

    EXPECT_EQ(r.tensor("A").layout, Layout::Row);
    EXPECT_EQ(r.tensor("B").layout, Layout::Col);
    EXPECT_EQ(r.tensor("C").layout, Layout::Row);
}

// ============================================================================
// Custom tensor names
// ============================================================================

TEST(Resolve, AcceptsCustomTensorNames)
{
    constexpr auto r = resolve(
        Signature{.dtype = DataType::FP16, .ops = {GemmOp{.lhs = "X", .rhs = "Y", .out = "Z"}}});

    EXPECT_EQ(r.tensor("X").rank, 2);
    EXPECT_EQ(r.tensor("Y").rank, 2);
    EXPECT_EQ(r.tensor("Z").rank, 2);
}

// ============================================================================
// dtype cascade
// ============================================================================

TEST(Resolve, CascadesBF16DtypeToAllTensors)
{
    constexpr auto r = resolve(
        Signature{.dtype = DataType::BF16, .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}});

    EXPECT_EQ(r.tensor("A").dtype, DataType::BF16);
    EXPECT_EQ(r.tensor("C").dtype, DataType::BF16);
}

TEST(Resolve, AllowsPerTensorDtypeOverride)
{
    constexpr auto r = resolve(Signature{.dtype   = DataType::FP16,
                                         .tensors = {Tensor{.name = "C", .dtype = DataType::FP32}},
                                         .ops     = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}});

    EXPECT_EQ(r.tensor("C").dtype, DataType::FP32);
    EXPECT_EQ(r.tensor("A").dtype, DataType::FP16); // cascade still applies to A
}

// ============================================================================
// Explicit tensor rank/layout overrides
// ============================================================================

TEST(Resolve, AllowsPerTensorRankOverride)
{
    constexpr auto r = resolve(Signature{.dtype   = DataType::FP16,
                                         .tensors = {Tensor{.name = "A", .rank = 3}},
                                         .ops     = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}});

    EXPECT_EQ(r.tensor("A").rank, 3);
}

TEST(Resolve, AllowsPerTensorLayoutOverride)
{
    // Override B from default Col to Row (R×R layout)
    constexpr auto r = resolve(Signature{.dtype   = DataType::FP16,
                                         .tensors = {Tensor{.name = "B", .layout = Layout::Row}},
                                         .ops     = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}});

    EXPECT_EQ(r.tensor("A").layout, Layout::Row); // default preserved
    EXPECT_EQ(r.tensor("B").layout, Layout::Row); // overridden from Col
    EXPECT_EQ(r.tensor("C").layout, Layout::Row); // default preserved
}

TEST(Resolve, AllowsMultipleLayoutOverrides)
{
    // Override both A and B (C×C layout)
    constexpr auto r = resolve(Signature{.dtype   = DataType::FP16,
                                         .tensors = {Tensor{.name = "A", .layout = Layout::Col},
                                                     Tensor{.name = "B", .layout = Layout::Col}},
                                         .ops     = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}});

    EXPECT_EQ(r.tensor("A").layout, Layout::Col);
    EXPECT_EQ(r.tensor("B").layout, Layout::Col);
    EXPECT_EQ(r.tensor("C").layout, Layout::Row); // default preserved
}

// ============================================================================
// GEMM + Add + Relu chain
// ============================================================================

TEST(Resolve, ResolvesGemmAddReluToSixTensors)
{
    constexpr auto r = resolve(Signature{.dtype = DataType::FP16,
                                         .ops   = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"},
                                                   AddOp{.lhs = "C", .rhs = "bias", .out = "D"},
                                                   ReluOp{.in = "D", .out = "E"}}});

    EXPECT_EQ(r.num_tensors, 6); // A, B, C, bias, D, E
}

TEST(Resolve, PropagatesRankAndLayoutThroughEpilogueChain)
{
    constexpr auto r = resolve(Signature{.dtype = DataType::FP16,
                                         .ops   = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"},
                                                   AddOp{.lhs = "C", .rhs = "bias", .out = "D"},
                                                   ReluOp{.in = "D", .out = "E"}}});

    EXPECT_EQ(r.tensor("C").rank, 2);
    EXPECT_EQ(r.tensor("bias").rank, 2);
    EXPECT_EQ(r.tensor("bias").layout, Layout::Row);
    EXPECT_EQ(r.tensor("D").rank, 2);
    EXPECT_EQ(r.tensor("D").layout, Layout::Row);
    EXPECT_EQ(r.tensor("E").rank, 2);
    EXPECT_EQ(r.tensor("E").layout, Layout::Row);
}

TEST(Resolve, AssignsSequentialIndicesToChainedOps)
{
    constexpr auto r = resolve(Signature{.dtype = DataType::FP16,
                                         .ops   = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"},
                                                   AddOp{.lhs = "C", .rhs = "bias", .out = "D"},
                                                   ReluOp{.in = "D", .out = "E"}}});

    EXPECT_EQ(r.tensor_index("A"), 0);
    EXPECT_EQ(r.tensor_index("B"), 1);
    EXPECT_EQ(r.tensor_index("C"), 2);
    EXPECT_EQ(r.tensor_index("bias"), 3);
    EXPECT_EQ(r.tensor_index("D"), 4);
    EXPECT_EQ(r.tensor_index("E"), 5);
}

// ============================================================================
// Standalone AddOp
// ============================================================================

TEST(Resolve, ResolvesStandaloneAddWithoutImpliedRank)
{
    constexpr auto r = resolve(
        Signature{.dtype = DataType::FP32, .ops = {AddOp{.lhs = "A", .rhs = "B", .out = "C"}}});

    EXPECT_EQ(r.num_tensors, 3);
    EXPECT_EQ(r.tensor("A").rank, 0);              // no op implies rank
    EXPECT_EQ(r.tensor("A").layout, Layout::Auto); // no op implies layout
}

// ============================================================================
// Conflict detection — redundant identical sets are silent
// ============================================================================

TEST(Resolve, AllowsRedundantIdenticalLayoutFromTwoGemmOps)
{
    // GemmOp1 outputs "C" as Row. GemmOp2 uses "C" as lhs (also Row).
    // Two ops set the same layout → no conflict.
    constexpr auto r = resolve(Signature{.dtype = DataType::FP16,
                                         .ops   = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"},
                                                   GemmOp{.lhs = "C", .rhs = "D", .out = "E"}}});

    EXPECT_EQ(r.tensor("C").layout, Layout::Row);
    EXPECT_EQ(r.tensor("C").rank, 2);
}

TEST(Resolve, AllowsPropagationThroughAddWithConsistentLayout)
{
    // GemmOp sets C=Row. AddOp connects C to bias and D.
    // Propagation sets bias and D to Row (matching C) → no conflict.
    constexpr auto r = resolve(Signature{.dtype = DataType::FP16,
                                         .ops   = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"},
                                                   AddOp{.lhs = "C", .rhs = "bias", .out = "D"}}});

    EXPECT_EQ(r.tensor("C").layout, Layout::Row);
    EXPECT_EQ(r.tensor("bias").layout, Layout::Row);
    EXPECT_EQ(r.tensor("D").layout, Layout::Row);
}

// ============================================================================
// FMHA pattern: two GemmOps + SoftmaxOp
// ============================================================================

TEST(Resolve, ResolvesFMHATwoGemmSoftmaxPattern)
{
    constexpr auto r = resolve(Signature{.dtype = DataType::FP16,
                                         .ops   = {GemmOp{.lhs = "Q", .rhs = "K", .out = "S"},
                                                   SoftmaxOp{.in = "S", .out = "P"},
                                                   GemmOp{.lhs = "P", .rhs = "V", .out = "O"}}});

    EXPECT_EQ(r.num_tensors, 6); // Q, K, S, P, V, O
    EXPECT_EQ(r.tensor("Q").rank, 2);
    EXPECT_EQ(r.tensor("S").rank, 2);
    EXPECT_EQ(r.tensor("P").rank, 2); // propagated via SoftmaxOp
    EXPECT_EQ(r.tensor("O").rank, 2);
}

// ============================================================================
// Scalar tracking
// ============================================================================

TEST(Resolve, PreservesScalarNamesAndDtypes)
{
    constexpr auto r =
        resolve(Signature{.dtype   = DataType::FP16,
                          .scalars = {Scalar{.name = "alpha", .dtype = DataType::FP32},
                                      Scalar{.name = "beta", .dtype = DataType::FP32}},
                          .ops     = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}});

    EXPECT_EQ(r.num_scalars, 2);
    EXPECT_EQ(r.scalar("alpha").dtype, DataType::FP32);
    EXPECT_EQ(r.scalar("beta").dtype, DataType::FP32);
    EXPECT_EQ(r.scalar_index("alpha"), 0);
    EXPECT_EQ(r.scalar_index("beta"), 1);
}

TEST(Resolve, ReportsZeroScalarsWhenNoneDeclared)
{
    constexpr auto r = resolve(
        Signature{.dtype = DataType::FP16, .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}});

    EXPECT_EQ(r.num_scalars, 0);
}

// ============================================================================
// find_tensor / find_scalar (constexpr, not consteval — returns -1 on miss)
// ============================================================================

TEST(Resolve, FindsTensorByName)
{
    constexpr auto r = resolve(
        Signature{.dtype = DataType::FP16, .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}});

    EXPECT_EQ(r.find_tensor("A"), 0);
    EXPECT_EQ(r.find_tensor("C"), 2);
}

TEST(Resolve, ReturnsNegativeOneForUnknownTensor)
{
    constexpr auto r = resolve(
        Signature{.dtype = DataType::FP16, .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}});

    EXPECT_EQ(r.find_tensor("Z"), -1);
}

TEST(Resolve, ReturnsNegativeOneForUnknownScalar)
{
    constexpr auto r = resolve(
        Signature{.dtype = DataType::FP16, .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}});

    EXPECT_EQ(r.find_scalar("nonexistent"), -1);
}

// ============================================================================
// C++20 concepts
// ============================================================================

TEST(Concepts, ClassifiesAddAndMulAsBinaryOpLike)
{
    EXPECT_TRUE(BinaryOpLike<AddOp>);
    EXPECT_TRUE(BinaryOpLike<MulOp>);
    EXPECT_FALSE(BinaryOpLike<ReluOp>);
    EXPECT_FALSE(BinaryOpLike<SoftmaxOp>);
}

TEST(Concepts, ClassifiesActivationsAsUnaryOpLike)
{
    EXPECT_TRUE(UnaryOpLike<ReluOp>);
    EXPECT_TRUE(UnaryOpLike<FastGeluOp>);
    EXPECT_TRUE(UnaryOpLike<GeluOp>);
    EXPECT_TRUE(UnaryOpLike<SiluOp>);
    EXPECT_TRUE(UnaryOpLike<SigmoidOp>);
    EXPECT_TRUE(UnaryOpLike<SoftmaxOp>);
    EXPECT_FALSE(UnaryOpLike<AddOp>);
    EXPECT_FALSE(UnaryOpLike<GemmOp>);
}

TEST(Concepts, ClassifiesGemmOpAsBinaryButNotUnary)
{
    // GemmOp has lhs/rhs/out AND is special-cased, not generic BinaryOpLike
    // (it has .lhs, .rhs, .out but is handled separately in register_slots)
    EXPECT_TRUE(BinaryOpLike<GemmOp>); // structurally matches, but dispatch special-cases it
    EXPECT_FALSE(UnaryOpLike<GemmOp>);
}

TEST(Concepts, ClassifiesScaleOpAsUnaryNotBinary)
{
    EXPECT_TRUE(UnaryOpLike<ScaleOp>);
    EXPECT_FALSE(BinaryOpLike<ScaleOp>);
}
