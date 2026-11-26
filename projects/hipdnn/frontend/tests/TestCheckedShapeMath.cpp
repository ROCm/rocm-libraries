// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#include <gtest/gtest.h>
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/node/detail/CheckedShapeMath.hpp>

#include <cstdint>
#include <limits>
#include <string>

using namespace hipdnn_frontend;
using hipdnn_frontend::detail::checkedAdd;
using hipdnn_frontend::detail::checkedDilatedKernelSize;
using hipdnn_frontend::detail::checkedMul;
using hipdnn_frontend::detail::checkedPaddedNumerator;
using hipdnn_frontend::detail::checkedStridedExtent;
using hipdnn_frontend::detail::checkedSub;
using hipdnn_frontend::detail::checkedTransposedExtent;

namespace
{
constexpr int64_t K_MAX = std::numeric_limits<int64_t>::max();
constexpr int64_t K_MIN = std::numeric_limits<int64_t>::min();
constexpr int64_t K_UNTOUCHED = 12345;
constexpr int64_t K_TWO_POW_31 = int64_t{1} << 31;
constexpr int64_t K_TWO_POW_32 = int64_t{1} << 32;
// INT64_MAX = 7 * 73 * 127 * 337 * 92737 * 649657, so 7 * K_MAX_OVER_7 is exactly the maximum
constexpr int64_t K_MAX_OVER_7 = K_MAX / 7;
static_assert(K_MAX_OVER_7 * 7 == K_MAX);
} // namespace

TEST(TestCheckedShapeMath, AddRejectsResultsOutsideInt64)
{
    int64_t out = K_UNTOUCHED;
    EXPECT_FALSE(checkedAdd(K_MAX, 1, out));
    EXPECT_FALSE(checkedAdd(1, K_MAX, out));
    EXPECT_FALSE(checkedAdd(K_MIN, -1, out));
    EXPECT_FALSE(checkedAdd(-1, K_MIN, out));
    EXPECT_EQ(out, K_UNTOUCHED);
}

TEST(TestCheckedShapeMath, AddAcceptsResultsAtInt64Limits)
{
    int64_t out = 0;
    EXPECT_TRUE(checkedAdd(K_MAX - 1, 1, out));
    EXPECT_EQ(out, K_MAX);
    EXPECT_TRUE(checkedAdd(K_MIN + 1, -1, out));
    EXPECT_EQ(out, K_MIN);
    EXPECT_TRUE(checkedAdd(K_MAX, K_MIN, out));
    EXPECT_EQ(out, -1);
}

TEST(TestCheckedShapeMath, SubRejectsResultsOutsideInt64)
{
    int64_t out = K_UNTOUCHED;
    EXPECT_FALSE(checkedSub(K_MIN, 1, out));
    EXPECT_FALSE(checkedSub(K_MAX, -1, out));
    EXPECT_FALSE(checkedSub(0, K_MIN, out));
    EXPECT_EQ(out, K_UNTOUCHED);
}

TEST(TestCheckedShapeMath, SubAcceptsResultsAtInt64Limits)
{
    int64_t out = 0;
    EXPECT_TRUE(checkedSub(K_MIN + 1, 1, out));
    EXPECT_EQ(out, K_MIN);
    EXPECT_TRUE(checkedSub(-1, K_MIN, out));
    EXPECT_EQ(out, K_MAX);
}

TEST(TestCheckedShapeMath, MulRejectsMinTimesMinusOne)
{
    int64_t out = K_UNTOUCHED;
    EXPECT_FALSE(checkedMul(K_MIN, -1, out));
    EXPECT_FALSE(checkedMul(-1, K_MIN, out));
    EXPECT_EQ(out, K_UNTOUCHED);
}

TEST(TestCheckedShapeMath, MulRejectsEachSignCombinationPastTheLimit)
{
    int64_t out = K_UNTOUCHED;
    EXPECT_FALSE(checkedMul(K_TWO_POW_32, K_TWO_POW_31, out));
    EXPECT_FALSE(checkedMul(-K_TWO_POW_32, -K_TWO_POW_31, out));
    EXPECT_FALSE(checkedMul(K_TWO_POW_32, -K_TWO_POW_32, out));
    EXPECT_FALSE(checkedMul(-K_TWO_POW_32, K_TWO_POW_32, out));
    EXPECT_FALSE(checkedMul(-2, -(int64_t{1} << 62), out));
    EXPECT_EQ(out, K_UNTOUCHED);
}

TEST(TestCheckedShapeMath, MulAcceptsEachSignCombinationWithinTheLimit)
{
    int64_t out = 0;
    EXPECT_TRUE(checkedMul(K_TWO_POW_31, K_TWO_POW_31, out));
    EXPECT_EQ(out, int64_t{1} << 62);
    EXPECT_TRUE(checkedMul(-K_TWO_POW_31, -K_TWO_POW_31, out));
    EXPECT_EQ(out, int64_t{1} << 62);
    // -2^32 * 2^31 is exactly the minimum, in either operand order
    EXPECT_TRUE(checkedMul(-K_TWO_POW_32, K_TWO_POW_31, out));
    EXPECT_EQ(out, K_MIN);
    EXPECT_TRUE(checkedMul(K_TWO_POW_31, -K_TWO_POW_32, out));
    EXPECT_EQ(out, K_MIN);
    EXPECT_TRUE(checkedMul(K_MAX, -1, out));
    EXPECT_EQ(out, -K_MAX);
    EXPECT_TRUE(checkedMul(K_MIN, 1, out));
    EXPECT_EQ(out, K_MIN);
    EXPECT_TRUE(checkedMul(K_MIN, 0, out));
    EXPECT_EQ(out, 0);
    EXPECT_TRUE(checkedMul(0, K_MIN, out));
    EXPECT_EQ(out, 0);
}

// A product exactly at the maximum fits and the next multiple of the larger operand does not.
// The accepts fail if the positive-positive bound `a <= maxValue / b` (or the negative-negative
// bound `b >= maxValue / a`) in checkedMul becomes strict; the rejects fail if it is loosened
// by one.
TEST(TestCheckedShapeMath, MulAcceptsTheMaximumAndRejectsOnePastItForLikeSigns)
{
    int64_t out = 0;
    EXPECT_TRUE(checkedMul(7, K_MAX_OVER_7, out));
    EXPECT_EQ(out, K_MAX);
    EXPECT_TRUE(checkedMul(K_MAX, 1, out));
    EXPECT_EQ(out, K_MAX);
    EXPECT_TRUE(checkedMul(-7, -K_MAX_OVER_7, out));
    EXPECT_EQ(out, K_MAX);
    EXPECT_TRUE(checkedMul(-1, -K_MAX, out));
    EXPECT_EQ(out, K_MAX);

    out = K_UNTOUCHED;
    // 7 * (K_MAX_OVER_7 + 1) = K_MAX + 7
    EXPECT_FALSE(checkedMul(7, K_MAX_OVER_7 + 1, out));
    EXPECT_FALSE(checkedMul(-7, -(K_MAX_OVER_7 + 1), out));
    EXPECT_EQ(out, K_UNTOUCHED);
}

TEST(TestCheckedShapeMath, DilatedKernelSizeComputesTheTerm)
{
    int64_t out = 0;
    auto error = checkedDilatedKernelSize("Node", 0, 2, 3, out);
    EXPECT_EQ(error.code, ErrorCode::OK) << error.err_msg;
    EXPECT_EQ(out, 5);
}

TEST(TestCheckedShapeMath, DilatedKernelSizeRejectsOverflow)
{
    int64_t out = K_UNTOUCHED;
    // (2^62 + 1) * 4 overflows
    auto error = checkedDilatedKernelSize("Node", 1, (int64_t{1} << 62) + 1, 5, out);
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_NE(error.err_msg.find("Node: dilated kernel size overflows int64 at spatial index 1"),
              std::string::npos)
        << error.err_msg;
    EXPECT_EQ(out, K_UNTOUCHED);

    // kernel - 1 overflows
    error = checkedDilatedKernelSize("Node", 0, 1, K_MIN, out);
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_NE(error.err_msg.find("overflows int64"), std::string::npos) << error.err_msg;
    EXPECT_EQ(out, K_UNTOUCHED);
}

// Removing the checked `+ 1` (checkedAdd(scaled, 1, result)) in checkedDilatedKernelSize fails
// this test: it is the only step that overflows.
TEST(TestCheckedShapeMath, DilatedKernelSizeRejectsOverflowOfTheFinalIncrement)
{
    int64_t out = K_UNTOUCHED;
    // kernel - 1 = 1 and max * 1 = max fit; max + 1 does not
    auto error = checkedDilatedKernelSize("Node", 1, K_MAX, 2, out);
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_NE(error.err_msg.find("Node: dilated kernel size overflows int64 at spatial index 1"),
              std::string::npos)
        << error.err_msg;
    EXPECT_EQ(out, K_UNTOUCHED);
}

TEST(TestCheckedShapeMath, PaddedNumeratorComputesTheTerm)
{
    int64_t out = 0;
    auto error = checkedPaddedNumerator("Node", 0, 8, 1, 2, 3, out);
    EXPECT_EQ(error.code, ErrorCode::OK) << error.err_msg;
    EXPECT_EQ(out, 8);

    error = checkedPaddedNumerator("Node", 0, 2, 0, 0, 5, out);
    EXPECT_EQ(error.code, ErrorCode::OK) << error.err_msg;
    EXPECT_EQ(out, -3);
}

TEST(TestCheckedShapeMath, PaddedNumeratorRejectsIntermediateOverflow)
{
    int64_t out = K_UNTOUCHED;
    // 8 + max overflows even though subtracting the span would bring the value back
    auto error = checkedPaddedNumerator("Node", 0, 8, K_MAX, 0, K_MAX, out);
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_NE(error.err_msg.find("Node: padded extent minus span overflows int64"),
              std::string::npos)
        << error.err_msg;
    EXPECT_EQ(out, K_UNTOUCHED);

    error = checkedPaddedNumerator("Node", 0, 0, 0, 0, K_MIN, out);
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_EQ(out, K_UNTOUCHED);
}

// Removing the checked `+ postPad` (checkedAdd(prePadded, postPad, padded)) in
// checkedPaddedNumerator fails this test: it is the only step that overflows, and its wrapped
// value minus the span would fit.
TEST(TestCheckedShapeMath, PaddedNumeratorRejectsOverflowOfThePostPadding)
{
    int64_t out = K_UNTOUCHED;
    // 1 + (max - 1) = max fits; max + max does not (it wraps to -2, and -2 - 3 would fit)
    auto error = checkedPaddedNumerator("Node", 1, 1, K_MAX - 1, K_MAX, 3, out);
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_NE(error.err_msg.find("padded extent minus span overflows int64 at spatial index 1"),
              std::string::npos)
        << error.err_msg;
    EXPECT_EQ(out, K_UNTOUCHED);
}

TEST(TestCheckedShapeMath, StridedExtentTruncatesAsPlainDivision)
{
    int64_t out = 0;
    auto error = checkedStridedExtent("Node", 0, 7, 2, out);
    EXPECT_EQ(error.code, ErrorCode::OK) << error.err_msg;
    EXPECT_EQ(out, 4);

    error = checkedStridedExtent("Node", 0, -1, 2, out);
    EXPECT_EQ(error.code, ErrorCode::OK) << error.err_msg;
    EXPECT_EQ(out, 1);
}

TEST(TestCheckedShapeMath, StridedExtentRejectsOverflowOfTheFinalIncrement)
{
    int64_t out = K_UNTOUCHED;
    auto error = checkedStridedExtent("Node", 2, K_MAX, 1, out);
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_NE(error.err_msg.find("Node: output extent overflows int64 at spatial index 2"),
              std::string::npos)
        << error.err_msg;
    EXPECT_EQ(out, K_UNTOUCHED);

    error = checkedStridedExtent("Node", 0, K_MIN, -1, out);
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_EQ(out, K_UNTOUCHED);
}

TEST(TestCheckedShapeMath, TransposedExtentComputesTheTerm)
{
    int64_t out = 0;
    // (4 - 1) * 2 - 1 - 1 + 3
    auto error = checkedTransposedExtent("Node", 0, 4, 2, 1, 1, 3, out);
    EXPECT_EQ(error.code, ErrorCode::OK) << error.err_msg;
    EXPECT_EQ(out, 7);
}

TEST(TestCheckedShapeMath, TransposedExtentRejectsOverflow)
{
    int64_t out = K_UNTOUCHED;
    // (5 - 1) * (2^62 + 1) overflows
    auto error = checkedTransposedExtent("Node", 0, 5, (int64_t{1} << 62) + 1, 0, 0, 1, out);
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_NE(error.err_msg.find("Node: input extent overflows int64 at spatial index 0"),
              std::string::npos)
        << error.err_msg;
    EXPECT_EQ(out, K_UNTOUCHED);

    // 0 - max - max underflows
    error = checkedTransposedExtent("Node", 0, 1, 1, K_MAX, K_MAX, 1, out);
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_EQ(out, K_UNTOUCHED);
}

// Each case overflows at exactly one step of checkedTransposedExtent, and the value that step
// would wrap to carries through the later steps without another overflow. Removing the checked
// step a case names fails that case.
TEST(TestCheckedShapeMath, TransposedExtentRejectsOverflowAtEachRemainingStep)
{
    int64_t out = K_UNTOUCHED;
    // checkedSub(dy, 1, dyMinusOne): min - 1 underflows; its wrap, max, times 1 minus 0 plus 0
    // fits
    auto error = checkedTransposedExtent("Node", 0, K_MIN, 1, 0, 0, 0, out);
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_NE(error.err_msg.find("Node: input extent overflows int64 at spatial index 0"),
              std::string::npos)
        << error.err_msg;
    EXPECT_EQ(out, K_UNTOUCHED);

    // checkedSub(strided, prePad, lessPrePad): (2 - 1) * max = max fits; max - (-1) overflows;
    // its wrap, min, minus 0 plus 1 fits
    error = checkedTransposedExtent("Node", 1, 2, K_MAX, -1, 0, 1, out);
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_NE(error.err_msg.find("Node: input extent overflows int64 at spatial index 1"),
              std::string::npos)
        << error.err_msg;
    EXPECT_EQ(out, K_UNTOUCHED);

    // checkedAdd(unpadded, window, result): (2 - 1) * max - 0 - 0 = max fits; max + 1 overflows
    error = checkedTransposedExtent("Node", 2, 2, K_MAX, 0, 0, 1, out);
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_NE(error.err_msg.find("Node: input extent overflows int64 at spatial index 2"),
              std::string::npos)
        << error.err_msg;
    EXPECT_EQ(out, K_UNTOUCHED);
}
