// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocm_ck/physical_tensor.hpp>

#include <gtest/gtest.h>

using namespace rocm_ck;

// TensorName has consteval constructor and operators — all comparisons must
// happen at compile time. We use constexpr variables and EXPECT_TRUE to
// report results at runtime.

// ============================================================================
// TensorName
// ============================================================================

TEST(TensorName, MatchesSingleCharacterName)
{
    constexpr bool match    = TensorName("A") == "A";
    constexpr bool no_match = TensorName("A") == "B";
    EXPECT_TRUE(match);
    EXPECT_FALSE(no_match);
}

TEST(TensorName, MatchesExactStringOnly)
{
    constexpr bool match  = TensorName("bias") == "bias";
    constexpr bool prefix = TensorName("bias") == "bia";
    constexpr bool longer = TensorName("bias") == "biases";
    EXPECT_TRUE(match);
    EXPECT_FALSE(prefix);
    EXPECT_FALSE(longer);
}

TEST(TensorName, Accepts15CharMaxLength)
{
    // 15 chars is the max (16 - null)
    constexpr bool match = TensorName("123456789012345") == "123456789012345";
    EXPECT_TRUE(match);
}

TEST(TensorName, SupportsEmptyString)
{
    constexpr TensorName tn("");
    EXPECT_EQ(tn.len, 0);
    constexpr bool match = tn == "";
    EXPECT_TRUE(match);
}

TEST(TensorName, SupportsThreeWayComparison)
{
    constexpr TensorName a("A");
    constexpr TensorName a2("A");
    constexpr TensorName b("B");
    constexpr bool eq = (a <=> a2) == 0;
    constexpr bool ne = (a <=> b) != 0;
    EXPECT_TRUE(eq);
    EXPECT_TRUE(ne);
}

// ============================================================================
// PhysicalTensor
// ============================================================================

TEST(PhysicalTensor, InitializesWithFP32RowAndSlotZero)
{
    constexpr PhysicalTensor pt{};
    EXPECT_EQ(pt.dtype, DataType::FP32);
    EXPECT_EQ(pt.layout, Layout::Row);
    EXPECT_EQ(pt.args_slot, 0);
}

TEST(PhysicalTensor, StoresAllFieldsFromConstruction)
{
    constexpr PhysicalTensor pt{TensorName("bias"), DataType::FP16, Layout::Col, 3};
    constexpr bool name_match = pt.name == "bias";
    EXPECT_TRUE(name_match);
    EXPECT_EQ(pt.dtype, DataType::FP16);
    EXPECT_EQ(pt.layout, Layout::Col);
    EXPECT_EQ(pt.args_slot, 3);
}

// ============================================================================
// Capacity constant
// ============================================================================

TEST(PhysicalTensor, LimitsCapacityTo8) { EXPECT_EQ(kMaxPhysicalTensors, 8); }
