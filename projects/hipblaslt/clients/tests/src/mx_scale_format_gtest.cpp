// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Host test for which TensileLite scale layout gfx950 keeps. The helper
// compares scale types before it looks at the arch string, and it does not
// query the device.

#include <gtest/gtest.h>

#include "../../../library/src/amd_detail/rocblaslt/include/rocblaslt-types.h"
#include "../../../library/src/amd_detail/rocblaslt/src/include/mx_scale_format.hpp"

namespace
{
    using Scale = RocblasltContractionProblem::ScalingFormat;

    TEST(MXScaleFormat_smoke, gfx950_keeps_noswizzle_for_packed_and_preswizzle_for_1001)
    {
        // Packed scale 3 on both sides, or packed plus no scale, keeps NoSwizzle (0).
        // Packed on one side and pre-swizzled on the other matches no layout.
        EXPECT_EQ(rocblaslt_gfx950_required_mx_scale_format(
                      "gfx950", Scale::Block_32_UE8M0, Scale::Block_32_UE8M0),
                  0);
        EXPECT_EQ(rocblaslt_gfx950_required_mx_scale_format(
                      "gfx950:xnack-", Scale::Block_32_UE8M0, Scale::None),
                  0);
        EXPECT_EQ(rocblaslt_gfx950_required_mx_scale_format(
                      "gfx950", Scale::Block_32_UE8M0_32_8_EXT, Scale::Block_32_UE8M0),
                  -2);
        EXPECT_TRUE(rocblaslt_kernel_mx_scale_format_ok(0, 0));
        EXPECT_FALSE(rocblaslt_kernel_mx_scale_format_ok(0, 1));

        // Pre-swizzled scale 1001 keeps HostPreSwizzle (1) only.
        EXPECT_EQ(rocblaslt_gfx950_required_mx_scale_format(
                      "gfx950", Scale::Block_32_UE8M0_32_8_EXT, Scale::Block_32_UE8M0_32_8_EXT),
                  1);
        EXPECT_TRUE(rocblaslt_kernel_mx_scale_format_ok(1, 1));
        EXPECT_FALSE(rocblaslt_kernel_mx_scale_format_ok(1, 0));

        // Non-MX scales and other archs are not filtered. -1 allows every layout.
        EXPECT_EQ(
            rocblaslt_gfx950_required_mx_scale_format("gfx950", Scale::Vector, Scale::Vector), -1);
        EXPECT_EQ(rocblaslt_gfx950_required_mx_scale_format(
                      "gfx950", Scale::Block_16_UE8M0, Scale::Block_16_UE8M0),
                  -1);
        EXPECT_EQ(rocblaslt_gfx950_required_mx_scale_format(
                      "gfx1250", Scale::Block_32_UE8M0, Scale::Block_32_UE8M0),
                  -1);
        EXPECT_EQ(rocblaslt_gfx950_required_mx_scale_format(
                      "gfx942", Scale::Block_32_UE8M0, Scale::Block_32_UE8M0),
                  -1);
        EXPECT_EQ(rocblaslt_gfx950_required_mx_scale_format(
                      nullptr, Scale::Block_32_UE8M0, Scale::None),
                  -1);
        EXPECT_TRUE(rocblaslt_kernel_mx_scale_format_ok(-1, 0));
        EXPECT_TRUE(rocblaslt_kernel_mx_scale_format_ok(-1, 1));
        EXPECT_TRUE(rocblaslt_kernel_mx_scale_format_ok(-1, 2));
    }
}
