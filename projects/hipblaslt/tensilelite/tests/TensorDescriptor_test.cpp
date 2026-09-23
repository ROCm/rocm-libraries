// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <Tensile/DataTypes.hpp>
#include <Tensile/TensorDescriptor.hpp>
#include <Tensile/Utils.hpp>

// Regression coverage for the fix to TensorDescriptor::elementBytes() and
// totalAllocatedBytes() after PR #6499 changed BaseTypeInfo::ElementSize from
// container size (sizeof(T)) to per-element segment size (sizeof(T) / Packing).
//
// Two prior bugs are exercised here:
//  1) elementBytes() asserted m_dataType != Float4/Float6/BFloat6, which fired
//     in Debug builds for valid MX descriptors.
//  2) totalAllocatedBytes() divided by info.packing a second time, returning
//     N/4 bytes for N Float4 elements instead of the correct N/2.
//
// All tests are pure CPU and do not require a GPU.
//
// Float6/BFloat6/Float4 are registered via TypeInfo<Float[*]x[*]> when the
// TENSILE_USE_FP6 / TENSILE_USE_BF6 / TENSILE_USE_FP4 macros are defined
// (see tensilelite/src/DataTypes.cpp registerAllTypeInfo). When those macros
// are off (e.g. on Windows) the same enums are still registered with the
// equivalent elementSize / packing via registerThinOcpFpTypesWhenNoExtOcp's
// addIfMissing fallback, so the per-element-bytes assertions below hold
// unconditionally.

namespace
{
    using TensileLite::TensorDescriptor;
    using TensileLite::multiplyElementSize;

    void expectElementAndTotalBytes(rocisa::DataType dt,
                                    float            expectedElementBytes,
                                    size_t           rows,
                                    size_t           cols)
    {
        TensorDescriptor desc("test", dt, {rows, cols});

        EXPECT_FLOAT_EQ(desc.elementBytes(), expectedElementBytes)
            << "elementBytes mismatch for dataType="
            << static_cast<int>(dt);

        EXPECT_EQ(desc.totalAllocatedBytes(),
                  multiplyElementSize(desc.totalAllocatedElements(),
                                      desc.elementBytes()))
            << "totalAllocatedBytes mismatch for dataType="
            << static_cast<int>(dt);
    }
} // namespace

TEST(TensorDescriptor, ElementBytes_Float)
{
    // Control: a fully-aligned, non-packed type.
    expectElementAndTotalBytes(rocisa::DataType::Float, 4.0f, 64, 128);
}

TEST(TensorDescriptor, ElementBytes_Half)
{
    expectElementAndTotalBytes(rocisa::DataType::Half, 2.0f, 64, 128);
}

TEST(TensorDescriptor, ElementBytes_Float8)
{
    // Float8 has packing == 1, so elementSize is the full 1 byte.
    expectElementAndTotalBytes(rocisa::DataType::Float8, 1.0f, 64, 128);
}

TEST(TensorDescriptor, ElementBytes_Float4)
{
    // Float4 (registered as Float4x2 when TENSILE_USE_FP4 is defined, or via
    // addIfMissing otherwise) has packing == 2 and per-element size 0.5 byte.
    // Pick 64x128 = 8192 elements, divisible by packing == 2.
    expectElementAndTotalBytes(rocisa::DataType::Float4, 0.5f, 64, 128);
}

TEST(TensorDescriptor, ElementBytes_Float6)
{
    // Float6: per-element size 0.75 byte. With TENSILE_USE_FP6 the type is
    // registered as Float6x32 (packing == 32); without TENSILE_USE_FP6 the
    // addIfMissing fallback registers it with packing == 16. 64x128 = 8192
    // elements is divisible by both.
    expectElementAndTotalBytes(rocisa::DataType::Float6, 0.75f, 64, 128);
}

TEST(TensorDescriptor, ElementBytes_BFloat6)
{
    // Same packing/elementSize story as Float6.
    expectElementAndTotalBytes(rocisa::DataType::BFloat6, 0.75f, 64, 128);
}

// Regression test: in Debug builds the previous elementBytes() implementation
// hard-asserted that m_dataType was none of {Float4, Float6, BFloat6}. Simply
// reaching this point without aborting (under -DNDEBUG=0) is the signal that
// the fix is in place.
TEST(TensorDescriptor, ElementBytes_DoesNotAbortOnPackedMxTypes)
{
    {
        TensorDescriptor desc("f4", rocisa::DataType::Float4, {64, 128});
        EXPECT_FLOAT_EQ(desc.elementBytes(), 0.5f);
    }
    {
        TensorDescriptor desc("f6", rocisa::DataType::Float6, {64, 128});
        EXPECT_FLOAT_EQ(desc.elementBytes(), 0.75f);
    }
    {
        TensorDescriptor desc("bf6", rocisa::DataType::BFloat6, {64, 128});
        EXPECT_FLOAT_EQ(desc.elementBytes(), 0.75f);
    }
}

// Regression test: pre-fix, totalAllocatedBytes() divided by info.packing a
// second time. For Float4 over N elements the correct byte count is N/2, but
// the buggy version returned N/4. Pin the expected value explicitly.
TEST(TensorDescriptor, TotalAllocatedBytes_Float4_NotDoubleDivided)
{
    TensorDescriptor desc("f4", rocisa::DataType::Float4, {64, 128});
    ASSERT_EQ(desc.totalAllocatedElements(), size_t{64} * 128);
    EXPECT_EQ(desc.totalAllocatedBytes(), (size_t{64} * 128) / 2);
}

TEST(TensorDescriptor, TotalAllocatedBytes_Float6_NotDoubleDivided)
{
    TensorDescriptor desc("f6", rocisa::DataType::Float6, {64, 128});
    ASSERT_EQ(desc.totalAllocatedElements(), size_t{64} * 128);
    // 0.75 * N = (3 * N) / 4
    EXPECT_EQ(desc.totalAllocatedBytes(), (size_t{64} * 128 * 3) / 4);
}
