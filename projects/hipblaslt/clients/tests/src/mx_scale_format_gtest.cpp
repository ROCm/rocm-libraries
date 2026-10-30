// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Host-only smoke test for the gfx950 scale-format request. The helper has no
// HIP dependency (see mx_scale_format.hpp). Included by relative path so this
// stays off the internal rocblaslt include dir, same as arch_revision_gtest.cpp.
// Ordinals are RocblasltContractionProblem::ScalingFormat, not the public
// HIPBLASLT_MATMUL_MATRIX_SCALE_* values.

#include <gtest/gtest.h>

#include "../../../library/src/amd_detail/rocblaslt/src/include/mx_scale_format.hpp"

namespace
{
    TEST(MXScaleFormat_smoke, request_for_arch_and_scale_ordinals)
    {
        // gfx950 Block_32_UE8M0 matches no kernel layout.
        EXPECT_EQ(rocblaslt_mx_scale_format_request("gfx950", 3, 3), -2);
        EXPECT_EQ(rocblaslt_mx_scale_format_request("gfx950:xnack-", 3, 0), -2);
        // Format 1001 still requests HostPreSwizzle.
        EXPECT_EQ(rocblaslt_mx_scale_format_request("gfx950", 9, 9), 1);
        // A natural-packed side cannot be read by a HostPreSwizzle kernel.
        EXPECT_EQ(rocblaslt_mx_scale_format_request("gfx950", 9, 3), -2);
        // Public outer-vector is ScalingFormat::Vector (2), not scale 3.
        EXPECT_EQ(rocblaslt_mx_scale_format_request("gfx950", 2, 2), -1);
        // Other block scales stay on today's matching.
        EXPECT_EQ(rocblaslt_mx_scale_format_request("gfx950", 4, 4), -1);
        // gfx1250 scale 3 stays InMemorySwizzle via an unspecified request.
        EXPECT_EQ(rocblaslt_mx_scale_format_request("gfx1250", 3, 3), -1);
        EXPECT_EQ(rocblaslt_mx_scale_format_request("gfx1250", 9, 9), -1);
        EXPECT_EQ(rocblaslt_mx_scale_format_request("gfx942", 3, 3), -1);
    }
}
