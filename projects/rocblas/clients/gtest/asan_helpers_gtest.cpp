// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "internal/asan_helpers.hpp"
#include <gtest/gtest.h>

// Compile-time tests for rocblas::conditional_v
// If any of these fail, the translation unit will not compile.

using rocblas::conditional_v;

// Basic true/false branch selection
static_assert(conditional_v<true, 1, 2> == 1, "true selects IfTrue");
static_assert(conditional_v<false, 1, 2> == 2, "false selects IfFalse");

// Values representative of actual ASAN block-size reductions
static_assert(conditional_v<true, 256, 1024> == 256, "ASAN-reduced block size");
static_assert(conditional_v<false, 256, 1024> == 1024, "normal block size");
static_assert(conditional_v<true, 4, 16> == 4);
static_assert(conditional_v<false, 4, 16> == 16);

// Identical values (degenerate case)
static_assert(conditional_v<true, 42, 42> == 42);
static_assert(conditional_v<false, 42, 42> == 42);

// Zero as a valid value
static_assert(conditional_v<true, 0, 512> == 0);
static_assert(conditional_v<false, 0, 512> == 512);

// rocblas_enable_asan is a well-formed constexpr bool
static_assert(rocblas_enable_asan || !rocblas_enable_asan, "rocblas_enable_asan is a valid bool");

// conditional_v works with rocblas_enable_asan as the condition
static_assert(conditional_v<rocblas_enable_asan, 256, 1024> == (rocblas_enable_asan ? 256 : 1024));

// Minimal runtime test so the file registers in gtest output
TEST(AsanHelpers, ConditionalV)
{
    // Runtime mirror of the static_asserts above — verifies ODR and linkage
    EXPECT_EQ((conditional_v<true, 1, 2>), 1);
    EXPECT_EQ((conditional_v<false, 1, 2>), 2);
    EXPECT_EQ((conditional_v<rocblas_enable_asan, 256, 1024>), rocblas_enable_asan ? 256 : 1024);
}
