// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file TestVariantPackBuilder.cpp
 * @brief Covers buffer sizing, where being wrong yields a number rather than an error.
 *
 * Both ways of getting this wrong pass a benchmark run. Under-sizing a padded tensor means
 * the kernel writes past the buffer and the corpus records the time it took to do so;
 * mis-sizing a sub-byte type allocates nothing at all. Neither raises anything a fleet would
 * notice, so the arithmetic is pinned here against tensors the test defines.
 */

#include <gtest/gtest.h>

#include <hipdnn_bench/VariantPackBuilder.hpp>

#include <cstdint>
#include <limits>
#include <vector>

namespace hipdnn_bench
{

TEST(TestVariantPackBuilder, SpansThePackedTensorExactly)
{
    // A packed tensor's span is its element count, which is the only case where the naive
    // count-based sizing happens to be right -- and the reason it survives casual testing.
    EXPECT_EQ(elementSpan({2, 3, 4}, {12, 4, 1}), 24);
    EXPECT_EQ(elementSpan({1}, {1}), 1);
}

TEST(TestVariantPackBuilder, SpansAPaddedTensorPastItsElementCount)
{
    // A 4x4 tensor with a row stride of 8: element [3][3] sits at 3*8 + 3 = 27, so 28 must be
    // addressable while the tensor holds only 16. Sizing by element count allocates 16 and the
    // kernel writes twelve elements past the end -- and still returns a time.
    EXPECT_EQ(elementSpan({4, 4}, {8, 1}), 28);
    EXPECT_GT(elementSpan({4, 4}, {8, 1}), 4 * 4);
}

TEST(TestVariantPackBuilder, TakesTheLastIndexOfEveryDimensionAtOnce)
{
    // The furthest element is reached with every index at its maximum, so the span is a sum
    // over dimensions. Taking the largest single term instead would size this at 13 and come
    // up four elements short.
    //
    //   (2-1)*12 + (3-1)*4 + (4-1)*1 + 1 = 12 + 8 + 3 + 1 = 24
    EXPECT_EQ(elementSpan({2, 3, 4}, {12, 4, 1}), 24);
    EXPECT_GT(elementSpan({2, 3, 4}, {12, 4, 1}), (2 - 1) * 12 + 1);
}

TEST(TestVariantPackBuilder, RefusesAMalformedTensorRatherThanSizingIt)
{
    EXPECT_EQ(elementSpan({}, {}), 0);
    EXPECT_EQ(elementSpan({2, 3}, {1}), 0) << "rank mismatch";
    EXPECT_EQ(elementSpan({2, 0}, {1, 1}), 0) << "zero extent";
    EXPECT_EQ(elementSpan({2, -1}, {1, 1}), 0) << "negative extent";
}

TEST(TestVariantPackBuilder, SizesSubByteTypesInBitsNotBytes)
{
    using hipdnn_frontend::DataType;

    // Four-bit types are the reason the arithmetic is in bits. Ten FP4 elements are five
    // bytes; a per-element byte size would be 0 (allocating nothing) or 1 (harmless but
    // wasteful), and only the first of those is a bug that runs.
    EXPECT_EQ(tensorBytes({10}, {1}, DataType::FP4_E2M1), 5);
    EXPECT_EQ(tensorBytes({10}, {1}, DataType::INT4), 5);

    // Odd spans round up: nine elements is 36 bits, which needs five bytes.
    EXPECT_EQ(tensorBytes({9}, {1}, DataType::FP4_E2M1), 5);
}

TEST(TestVariantPackBuilder, SizesTheOrdinaryTypes)
{
    using hipdnn_frontend::DataType;
    EXPECT_EQ(tensorBytes({4, 4}, {4, 1}, DataType::FLOAT), 64);
    EXPECT_EQ(tensorBytes({4, 4}, {4, 1}, DataType::HALF), 32);
    EXPECT_EQ(tensorBytes({4, 4}, {4, 1}, DataType::BFLOAT16), 32);
    EXPECT_EQ(tensorBytes({4, 4}, {4, 1}, DataType::DOUBLE), 128);
    EXPECT_EQ(tensorBytes({4, 4}, {4, 1}, DataType::INT8), 16);
}

TEST(TestVariantPackBuilder, RefusesATypeItCannotSize)
{
    using hipdnn_frontend::DataType;

    // Refusing by name beats guessing a width. A guess that is too small is a buffer overrun
    // that still produces a time.
    EXPECT_FALSE(tensorBytes({4}, {1}, DataType::NOT_SET).has_value());
    EXPECT_EQ(elementBits(DataType::NOT_SET), 0);
}

TEST(TestVariantPackBuilder, RefusesASpanThatWouldOverflow)
{
    using hipdnn_frontend::DataType;

    // The corpus search proposes across orders of magnitude, so this is reachable rather than
    // theoretical. Wrapping would produce a small, plausible allocation.
    const auto huge = std::numeric_limits<int64_t>::max() / 4;
    EXPECT_FALSE(tensorBytes({huge, 2}, {2, 1}, DataType::DOUBLE).has_value());
}

} // namespace hipdnn_bench
