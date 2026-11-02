// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Host test for the gfx950 packed-scale reject. The helper compares scale
// types before it looks at the arch string, and it does not query the device.

#include <gtest/gtest.h>

#include "../../../library/src/amd_detail/rocblaslt/include/rocblaslt-types.h"
#include "../../../library/src/amd_detail/rocblaslt/src/include/mx_scale_format.hpp"

namespace
{
    using Scale = RocblasltContractionProblem::ScalingFormat;

    TEST(MXScaleFormat_smoke, packed_gfx950_rejects_before_search)
    {
        EXPECT_TRUE(rocblaslt_reject_gfx950_packed_scale("gfx950", Scale::Block_32_UE8M0, Scale::Block_32_UE8M0));
        EXPECT_TRUE(rocblaslt_reject_gfx950_packed_scale(
            "gfx950:xnack-", Scale::Block_32_UE8M0, Scale::None));
        // One packed side is enough. The pre-swizzled side does not keep the kernel.
        EXPECT_TRUE(rocblaslt_reject_gfx950_packed_scale(
            "gfx950", Scale::Block_32_UE8M0_32_8_EXT, Scale::Block_32_UE8M0));

        // Pre-swizzled scales still go through solution search.
        EXPECT_FALSE(rocblaslt_reject_gfx950_packed_scale(
            "gfx950", Scale::Block_32_UE8M0_32_8_EXT, Scale::Block_32_UE8M0_32_8_EXT));
        // Non-packed scales do not scan as a reject, including public-vector ordinal 2.
        EXPECT_FALSE(rocblaslt_reject_gfx950_packed_scale("gfx950", Scale::Vector, Scale::Vector));
        EXPECT_FALSE(
            rocblaslt_reject_gfx950_packed_scale("gfx950", Scale::Block_16_UE8M0, Scale::Block_16_UE8M0));

        // Packed scales on other archs still go through solution search.
        EXPECT_FALSE(rocblaslt_reject_gfx950_packed_scale(
            "gfx1250", Scale::Block_32_UE8M0, Scale::Block_32_UE8M0));
        EXPECT_FALSE(rocblaslt_reject_gfx950_packed_scale(
            "gfx1250", Scale::Block_32_UE8M0_32_8_EXT, Scale::Block_32_UE8M0_32_8_EXT));
        EXPECT_FALSE(
            rocblaslt_reject_gfx950_packed_scale("gfx942", Scale::Block_32_UE8M0, Scale::Block_32_UE8M0));
        EXPECT_FALSE(rocblaslt_reject_gfx950_packed_scale(nullptr, Scale::Block_32_UE8M0, Scale::None));
    }
}
