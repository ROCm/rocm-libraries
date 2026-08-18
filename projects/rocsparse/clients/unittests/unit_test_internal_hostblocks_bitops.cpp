/*! \file */
/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

//
// Unit tests for rocSPARSE internal host bit-twiddling helpers: clz
// (count-leading-zeros / MSB position) and fnp2 (next power of two). Split out
// of unit_test_internal_hostblocks.cpp by topic.
//
// NOTE ON TARGET: these are pure *host* helpers, but their headers pull in
// rocsparse_common.hpp, which uses HIP device intrinsics that only compile
// under `-x hip`. This file therefore builds into the GPU test binary
// (rocsparse-unit-test-device), NOT the host-only rocsparse-unit-test. The
// tests themselves run host code and launch no kernels.
//
#include "unit_test_utils.hpp"

#include "rocsparse_common.hpp" // fnp2
#include "rocsparse_utility.hpp" // clz

#include <gtest/gtest.h>

// clz(n) returns the 1-based position of the most-significant set bit
// (floor(log2(n)) + 1) for n > 0, and 0 for n == 0. This identity holds for
// both the 32-bit and the ILP64 (64-bit) definitions of rocsparse_int, so the
// expected values below are configuration-independent.
TEST(internal_hostblocks_bitops, clz)
{
    EXPECT_EQ(rocsparse::clz(0), 0); // documented n == 0 special case
    EXPECT_EQ(rocsparse::clz(1), 1);
    EXPECT_EQ(rocsparse::clz(2), 2);
    EXPECT_EQ(rocsparse::clz(3), 2);
    EXPECT_EQ(rocsparse::clz(4), 3);
    EXPECT_EQ(rocsparse::clz(5), 3);
    EXPECT_EQ(rocsparse::clz(7), 3);
    EXPECT_EQ(rocsparse::clz(8), 4);
    EXPECT_EQ(rocsparse::clz(15), 4);
    EXPECT_EQ(rocsparse::clz(16), 5);
    EXPECT_EQ(rocsparse::clz(17), 5);
    EXPECT_EQ(rocsparse::clz(255), 8);
    EXPECT_EQ(rocsparse::clz(256), 9);
    EXPECT_EQ(rocsparse::clz(1023), 10);
    EXPECT_EQ(rocsparse::clz(1024), 11);
    // Highest bit representable without touching the sign bit of a 32-bit
    // rocsparse_int (bit 30).
    EXPECT_EQ(rocsparse::clz(static_cast<rocsparse_int>(1) << 30), 31);
}

// fnp2(x) rounds x up to the next power of two. Powers of two map to
// themselves; the empty/zero input wraps to 0 (x-- underflows then x++ returns
// to 0), which is the documented edge-case behavior we lock in here.
TEST(internal_hostblocks_bitops, fnp2)
{
    EXPECT_EQ(rocsparse::fnp2(0u), 0u); // edge case: wraps back to 0
    EXPECT_EQ(rocsparse::fnp2(1u), 1u);
    EXPECT_EQ(rocsparse::fnp2(2u), 2u);
    EXPECT_EQ(rocsparse::fnp2(3u), 4u);
    EXPECT_EQ(rocsparse::fnp2(4u), 4u);
    EXPECT_EQ(rocsparse::fnp2(5u), 8u);
    EXPECT_EQ(rocsparse::fnp2(7u), 8u);
    EXPECT_EQ(rocsparse::fnp2(8u), 8u);
    EXPECT_EQ(rocsparse::fnp2(9u), 16u);
    EXPECT_EQ(rocsparse::fnp2(15u), 16u);
    EXPECT_EQ(rocsparse::fnp2(16u), 16u);
    EXPECT_EQ(rocsparse::fnp2(17u), 32u);
    EXPECT_EQ(rocsparse::fnp2(1000u), 1024u);
    EXPECT_EQ(rocsparse::fnp2(1024u), 1024u);
    EXPECT_EQ(rocsparse::fnp2(1u << 30), 1u << 30);
    EXPECT_EQ(rocsparse::fnp2(1u << 31), 1u << 31); // 2^31 is already a power of 2
}

// NOTE: the sibling `flp2` (floor power of two) case is PARKED for
// self-containment on ut-infra. It lives in rocsparse_csrmv_adaptive_analysis.hpp,
// a header added by the sibling AISPARSE-642 level2 SpMV series that is not in
// this stack's base. Reinstate it here once AISPARSE-642 lands in ut-infra.
// See kim/internal_ut_findings.md.
