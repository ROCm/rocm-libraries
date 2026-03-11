// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "asan_helpers.hpp"
#include <gtest/gtest.h>

// Compile-time tests for ROCSOLVER_ASAN_VALUE
// If any of these fail, the translation unit will not compile.

// Basic branch selection
static_assert(ROCSOLVER_ASAN_VALUE(1, 2) == 1 || ROCSOLVER_ASAN_VALUE(1, 2) == 2,
              "ROCSOLVER_ASAN_VALUE selects one of its arguments");

// Values representative of actual ASAN block-size reductions
static_assert(ROCSOLVER_ASAN_VALUE(256, 1024) == (rocsolver_enable_asan ? 256 : 1024),
              "ROCSOLVER_ASAN_VALUE matches rocsolver_enable_asan");
static_assert(ROCSOLVER_ASAN_VALUE(4, 16) == (rocsolver_enable_asan ? 4 : 16));

// Identical values (degenerate case)
static_assert(ROCSOLVER_ASAN_VALUE(42, 42) == 42);

// Zero as a valid value
static_assert(ROCSOLVER_ASAN_VALUE(0, 512) == (rocsolver_enable_asan ? 0 : 512));

// rocsolver_enable_asan is a well-formed constexpr bool
static_assert(rocsolver_enable_asan || !rocsolver_enable_asan,
              "rocsolver_enable_asan is a valid bool");

// ROCSOLVER_ASAN_VALUE is consistent with rocsolver_enable_asan
static_assert(ROCSOLVER_ASAN_VALUE(true, false) == rocsolver_enable_asan,
              "ROCSOLVER_ASAN_VALUE(true, false) matches rocsolver_enable_asan");

// Minimal runtime test so the file registers in gtest output
TEST(AsanHelpers, AsanValue)
{
    // Runtime mirror of the static_asserts above — verifies macro expansion and linkage
    EXPECT_EQ(ROCSOLVER_ASAN_VALUE(1, 2), rocsolver_enable_asan ? 1 : 2);
    EXPECT_EQ(ROCSOLVER_ASAN_VALUE(256, 1024), rocsolver_enable_asan ? 256 : 1024);
    EXPECT_EQ(ROCSOLVER_ASAN_VALUE(true, false), rocsolver_enable_asan);
}
