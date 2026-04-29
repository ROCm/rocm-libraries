// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "tests/common/TestWorkarounds.hpp"

#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include <gtest/gtest.h>

#include <regex>
#include <string>

using ::test_common::workarounds_detail::queryCurrentDeviceArch;
using ::test_common::workarounds_detail::stripArchFeatureSuffix;

TEST(TestWorkarounds, StripArchFeatureSuffix_NoColon_ReturnsInput)
{
    EXPECT_EQ(stripArchFeatureSuffix("gfx90a"), "gfx90a");
}

TEST(TestWorkarounds, StripArchFeatureSuffix_WithFeatures_StripsAtFirstColon)
{
    EXPECT_EQ(stripArchFeatureSuffix("gfx90a:sramecc+:xnack-"), "gfx90a");
}

TEST(TestWorkarounds, StripArchFeatureSuffix_EmptyInput_ReturnsEmpty)
{
    EXPECT_EQ(stripArchFeatureSuffix(""), "");
}

TEST(TestWorkarounds, StripArchFeatureSuffix_LeadingColon_ReturnsEmpty)
{
    EXPECT_EQ(stripArchFeatureSuffix(":xnack-"), "");
}

TEST(GpuTestWorkarounds, QueryCurrentDeviceArch_ReturnsValidArchString)
{
    SKIP_IF_NO_DEVICES();

    const auto arch = queryCurrentDeviceArch();

    ASSERT_FALSE(arch.empty()) << "Expected non-empty arch string from device 0";
    EXPECT_EQ(arch.find(':'), std::string::npos)
        << "Arch string should have feature suffix stripped: " << arch;
    EXPECT_TRUE(std::regex_match(arch, std::regex{"^gfx[0-9a-f]+$"}))
        << "Arch string does not match expected gfx<digits> pattern: " << arch;
}
