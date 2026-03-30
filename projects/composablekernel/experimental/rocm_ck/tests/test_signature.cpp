// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocm_ck/signature.hpp>

#include <gtest/gtest.h>

using namespace rocm_ck;

// ============================================================================
// Signature construction
// ============================================================================

TEST(Signature, DefaultsToNoDtype)
{
    constexpr Signature sig{};
    EXPECT_FALSE(sig.dtype.has_value());
}

TEST(Signature, StoresExplicitDtype)
{
    constexpr Signature sig{.dtype = DataType::FP16};
    EXPECT_TRUE(sig.dtype.has_value());
    EXPECT_EQ(*sig.dtype, DataType::FP16);
}

// ============================================================================
// Tensor
// ============================================================================

TEST(Tensor, DefaultsToAutoLayoutAndRankZero)
{
    constexpr Tensor t{.name = "A"};
    EXPECT_EQ(t.name, "A");
    EXPECT_FALSE(t.dtype.has_value());
    EXPECT_EQ(t.rank, 0);
    EXPECT_EQ(t.layout, Layout::Auto);
}

TEST(Tensor, StoresAllExplicitFields)
{
    constexpr Tensor t{.name = "Q", .dtype = DataType::FP32, .rank = 3, .layout = Layout::Row};
    EXPECT_EQ(t.name, "Q");
    EXPECT_EQ(*t.dtype, DataType::FP32);
    EXPECT_EQ(t.rank, 3);
    EXPECT_EQ(t.layout, Layout::Row);
}

// ============================================================================
// Scalar
// ============================================================================

TEST(Scalar, DefaultsToFP32Dtype)
{
    constexpr Scalar s{.name = "alpha"};
    EXPECT_EQ(s.name, "alpha");
    EXPECT_EQ(s.dtype, DataType::FP32);
}

TEST(Scalar, StoresExplicitDtype)
{
    constexpr Scalar s{.name = "scale", .dtype = DataType::FP16};
    EXPECT_EQ(s.dtype, DataType::FP16);
}

// ============================================================================
// Op variant
// ============================================================================

TEST(Op, DefaultsToMonostate)
{
    constexpr Op op{};
    EXPECT_TRUE(std::holds_alternative<std::monostate>(op));
}

TEST(Op, HoldsGemmOp)
{
    constexpr Op op = GemmOp{.lhs = "A", .rhs = "B", .out = "C"};
    EXPECT_TRUE(std::holds_alternative<GemmOp>(op));
}

TEST(Op, HoldsAllUnaryOpTypes)
{
    constexpr Op relu = ReluOp{.in = "X", .out = "Y"};
    EXPECT_TRUE(std::holds_alternative<ReluOp>(relu));

    constexpr Op gelu = FastGeluOp{.in = "X", .out = "Y"};
    EXPECT_TRUE(std::holds_alternative<FastGeluOp>(gelu));

    constexpr Op sigmoid = SigmoidOp{.in = "X", .out = "Y"};
    EXPECT_TRUE(std::holds_alternative<SigmoidOp>(sigmoid));
}

TEST(Op, HoldsAllBinaryOpTypes)
{
    constexpr Op add = AddOp{.lhs = "X", .rhs = "Y", .out = "Z"};
    EXPECT_TRUE(std::holds_alternative<AddOp>(add));

    constexpr Op mul = MulOp{.lhs = "X", .rhs = "Y", .out = "Z"};
    EXPECT_TRUE(std::holds_alternative<MulOp>(mul));
}

// ============================================================================
// GemmOp defaults
// ============================================================================

TEST(GemmOp, DefaultsAccDtypeToFP32)
{
    constexpr GemmOp gemm{.lhs = "A", .rhs = "B", .out = "C"};
    EXPECT_EQ(gemm.acc_dtype, DataType::FP32);
}

// ============================================================================
// Capacity constants
// ============================================================================

TEST(Signature, DefinesExpectedCapacityLimits)
{
    EXPECT_EQ(kMaxTensors, 16);
    EXPECT_EQ(kMaxScalars, 16);
    EXPECT_EQ(kMaxOps, 8);
}
