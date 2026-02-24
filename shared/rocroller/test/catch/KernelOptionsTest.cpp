// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocRoller/Context.hpp>
#include <rocRoller/KernelOptions.hpp>
#include <rocRoller/KernelOptions_detail.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

using namespace rocRoller;

TEST_CASE("KernelOptions ToString", "[kernel-options]")
{
    KernelOptions opts;
    std::string   output = opts.toString();

    REQUIRE_THAT(output, Catch::Matchers::ContainsSubstring("Kernel Options:"));
    REQUIRE_THAT(output, Catch::Matchers::ContainsSubstring("  logLevel:"));
    REQUIRE_THAT(output, Catch::Matchers::ContainsSubstring("  alwaysWaitAfterLoad:"));
    REQUIRE_THAT(output, Catch::Matchers::ContainsSubstring("  alwaysWaitBeforeBranch:"));
    REQUIRE_THAT(output, Catch::Matchers::ContainsSubstring("scaleSkipPermlane:"));
}

TEST_CASE("KernelOptions scaleSkipPermlane default is None",
          "[kernel-options][scale-skip-permlane]")
{
    KernelOptions opts;
    REQUIRE(opts->scaleSkipPermlane == ScaleSkipPermlaneMode::None);
}

TEST_CASE("KernelOptions scaleSkipPermlane PreSwizzleScaleGFX950 in ToString",
          "[kernel-options][scale-skip-permlane]")
{
    KernelOptionValues values;
    values.scaleSkipPermlane = ScaleSkipPermlaneMode::PreSwizzleScaleGFX950;
    KernelOptions opts(std::move(values));
    REQUIRE_THAT(opts.toString(), Catch::Matchers::ContainsSubstring("PreSwizzleScaleGFX950"));
}

TEST_CASE("KernelOptions scaleSkipPermlane PreSwizzleScale in ToString",
          "[kernel-options][scale-skip-permlane]")
{
    KernelOptionValues values;
    values.scaleSkipPermlane = ScaleSkipPermlaneMode::PreSwizzleScale;
    KernelOptions opts(std::move(values));
    REQUIRE_THAT(opts.toString(), Catch::Matchers::ContainsSubstring("PreSwizzleScale"));
}
