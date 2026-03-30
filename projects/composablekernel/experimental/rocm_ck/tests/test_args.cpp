// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocm_ck/args.hpp>

#include <gtest/gtest.h>

#include <cstddef>
#include <type_traits>

using namespace rocm_ck;

// ============================================================================
// TensorArg ABI
// ============================================================================

TEST(TensorArg, IsTriviallyCopyable) { EXPECT_TRUE(std::is_trivially_copyable_v<TensorArg>); }

TEST(TensorArg, HasStandardLayout) { EXPECT_TRUE(std::is_standard_layout_v<TensorArg>); }

TEST(TensorArg, Occupies80Bytes)
{
    // ptr(8) + lengths(6*4=24) + strides(6*8=48) = 80
    EXPECT_EQ(sizeof(TensorArg), 80);
}

TEST(TensorArg, AlignsTo8Bytes) { EXPECT_EQ(alignof(TensorArg), 8); }

TEST(TensorArg, PlacesFieldsAtExpectedOffsets)
{
    EXPECT_EQ(offsetof(TensorArg, ptr), 0);
    EXPECT_EQ(offsetof(TensorArg, lengths), 8);
    EXPECT_EQ(offsetof(TensorArg, strides), 32);
}

// ============================================================================
// ScalarValue ABI
// ============================================================================

TEST(ScalarValue, IsTriviallyCopyable) { EXPECT_TRUE(std::is_trivially_copyable_v<ScalarValue>); }

TEST(ScalarValue, Occupies8Bytes)
{
    // Union of float(4), int32(4), uint32(4), double(8) -> 8 bytes
    EXPECT_EQ(sizeof(ScalarValue), 8);
}

// ============================================================================
// Args ABI
// ============================================================================

TEST(Args, IsTriviallyCopyable) { EXPECT_TRUE(std::is_trivially_copyable_v<Args>); }

TEST(Args, HasStandardLayout) { EXPECT_TRUE(std::is_standard_layout_v<Args>); }

TEST(Args, Occupies1552Bytes)
{
    // 16 tensors * 80 + 16 scalars * 8 + batch_count(4) + pad(4)
    // + 16 batch_strides * 8 + workspace_ptr(8) = 1280 + 128 + 8 + 128 + 8 = 1552
    EXPECT_EQ(sizeof(Args), 1552);
}

TEST(Args, AlignsTo8Bytes) { EXPECT_EQ(alignof(Args), 8); }

TEST(Args, FitsWithin4KBKernargBudget)
{
    // HSA minimum kernarg size is 4096 bytes
    EXPECT_LE(sizeof(Args), 4096);
}

// ============================================================================
// Capacity constants
// ============================================================================

TEST(Args, DefinesExpectedCapacityLimits)
{
    EXPECT_EQ(kMaxRank, 6);
    EXPECT_EQ(kMaxTensors, 16);
    EXPECT_EQ(kMaxScalars, 16);
}

// ============================================================================
// ScalarValue union access
// ============================================================================

TEST(ScalarValue, StoresAndRetrievesFloat)
{
    ScalarValue sv{};
    sv.f32 = 3.14f;
    EXPECT_FLOAT_EQ(sv.f32, 3.14f);
}

TEST(ScalarValue, StoresAndRetrievesInt32)
{
    ScalarValue sv{};
    sv.i32 = -42;
    EXPECT_EQ(sv.i32, -42);
}

TEST(ScalarValue, StoresAndRetrievesDouble)
{
    ScalarValue sv{};
    sv.f64 = 2.718281828;
    EXPECT_DOUBLE_EQ(sv.f64, 2.718281828);
}
