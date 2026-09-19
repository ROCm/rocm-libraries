// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <cstddef>
#include <miopen/miopen.h>
#include <miopen/version.h>
#include <gtest/gtest.h>

// The reported version comes from the loaded library, the expected one from the headers
// this test compiled against; a mismatch means the two were not built together.
TEST(CPU_VersionApi_NONE, ReportedVersionMatchesHeader)
{
    std::size_t major = 0, minor = 0, patch = 0;
    ASSERT_EQ(miopenGetVersion(&major, &minor, &patch), miopenStatusSuccess);
    EXPECT_EQ(major, static_cast<std::size_t>(MIOPEN_VERSION_MAJOR));
    EXPECT_EQ(minor, static_cast<std::size_t>(MIOPEN_VERSION_MINOR));
    EXPECT_EQ(patch, static_cast<std::size_t>(MIOPEN_VERSION_PATCH));
}
