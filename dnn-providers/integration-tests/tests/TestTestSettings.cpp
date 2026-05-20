// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

#include "harness/TestSettings.hpp"

using hipdnn_integration_tests::TestSettings;

// NOLINTBEGIN(readability-identifier-naming) -- gtest macro-generated names

namespace
{

// Helper to create a temporary TOML file that is auto-deleted on destruction.
class TempTomlFile
{
public:
    explicit TempTomlFile(const std::string& content)
        : _path(std::filesystem::temp_directory_path()
                / ("test_settings_" + std::to_string(std::rand()) + ".toml"))
    {
        std::ofstream ofs(_path);
        ofs << content;
    }

    ~TempTomlFile()
    {
        std::filesystem::remove(_path);
    }

    TempTomlFile(const TempTomlFile&) = delete;
    TempTomlFile& operator=(const TempTomlFile&) = delete;
    TempTomlFile(TempTomlFile&&) = delete;
    TempTomlFile& operator=(TempTomlFile&&) = delete;

    const std::filesystem::path& path() const
    {
        return _path;
    }

private:
    std::filesystem::path _path;
};

} // namespace

// ---------------------------------------------------------------------------
// Parsing
// ---------------------------------------------------------------------------

TEST(TestSettingsParser, ParsesValidTomlWithOverrides)
{
    const TempTomlFile file(R"(
[meta]
version = 1

[[tolerance_overrides]]
filters = ["*ConvFwd*Fp16*"]
atol = 1e-3
rtol = 1e-2

[[tolerance_overrides]]
filters = ["*ConvFwd*Fp32*"]
atol = 1e-5
rtol = 1e-4
)");

    const TestSettings settings(file.path());
    EXPECT_EQ(settings.toleranceOverrideCount(), 2U);
}

TEST(TestSettingsParser, ParsesValidTomlWithNoOverrides)
{
    const TempTomlFile file(R"(
[meta]
version = 1
)");

    const TestSettings settings(file.path());
    EXPECT_EQ(settings.toleranceOverrideCount(), 0U);
}

TEST(TestSettingsParser, ThrowsOnMissingVersion)
{
    const TempTomlFile file(R"(
[meta]
)");

    EXPECT_THROW(const TestSettings settings(file.path()), std::runtime_error);
}

TEST(TestSettingsParser, ThrowsOnUnsupportedVersion)
{
    const TempTomlFile file(R"(
[meta]
version = 99
)");

    EXPECT_THROW(const TestSettings settings(file.path()), std::runtime_error);
}

TEST(TestSettingsParser, ThrowsOnMissingFilters)
{
    const TempTomlFile file(R"(
[meta]
version = 1

[[tolerance_overrides]]
atol = 1e-3
rtol = 1e-2
)");

    EXPECT_THROW(const TestSettings settings(file.path()), std::runtime_error);
}

TEST(TestSettingsParser, ThrowsOnMissingAtol)
{
    const TempTomlFile file(R"(
[meta]
version = 1

[[tolerance_overrides]]
filters = ["*test*"]
rtol = 1e-2
)");

    EXPECT_THROW(const TestSettings settings(file.path()), std::runtime_error);
}

TEST(TestSettingsParser, ThrowsOnMissingRtol)
{
    const TempTomlFile file(R"(
[meta]
version = 1

[[tolerance_overrides]]
filters = ["*test*"]
atol = 1e-3
)");

    EXPECT_THROW(const TestSettings settings(file.path()), std::runtime_error);
}

TEST(TestSettingsParser, ThrowsOnNonexistentFile)
{
    EXPECT_THROW(const TestSettings settings("/nonexistent/path.toml"), std::exception);
}

// ---------------------------------------------------------------------------
// Filter matching
// ---------------------------------------------------------------------------

TEST(TestSettingsParser, FindOverrideMatchesWildcard)
{
    const TempTomlFile file(R"(
[meta]
version = 1

[[tolerance_overrides]]
filters = ["*ConvFwd*Fp16*"]
atol = 1e-3
rtol = 1e-2
)");

    const TestSettings settings(file.path());

    auto result = settings.findToleranceOverride(
        "IntegrationGpuConvFwd2dFp16/Smoke.Correctness/NCHW_params");
    ASSERT_TRUE(result.has_value());
    EXPECT_FLOAT_EQ(result->atol, 1e-3F);
    EXPECT_FLOAT_EQ(result->rtol, 1e-2F);
}

TEST(TestSettingsParser, FindOverrideReturnsNulloptWhenNoMatch)
{
    const TempTomlFile file(R"(
[meta]
version = 1

[[tolerance_overrides]]
filters = ["*ConvFwd*Fp16*"]
atol = 1e-3
rtol = 1e-2
)");

    const TestSettings settings(file.path());

    auto result
        = settings.findToleranceOverride("IntegrationGpuBatchnormFp32/Smoke.Correctness/params");
    EXPECT_FALSE(result.has_value());
}

TEST(TestSettingsParser, FindOverrideReturnsNulloptWhenNoOverrides)
{
    const TempTomlFile file(R"(
[meta]
version = 1
)");

    const TestSettings settings(file.path());

    auto result = settings.findToleranceOverride("AnyTestName");
    EXPECT_FALSE(result.has_value());
}

TEST(TestSettingsParser, FindOverrideMatchesMultipleFiltersInEntry)
{
    const TempTomlFile file(R"(
[meta]
version = 1

[[tolerance_overrides]]
filters = ["*Fp16*", "*Half*"]
atol = 1e-3
rtol = 1e-2
)");

    const TestSettings settings(file.path());

    // Should match the first filter
    auto result1
        = settings.findToleranceOverride("IntegrationGpuConvFwd2dFp16/Smoke.Correctness/params");
    ASSERT_TRUE(result1.has_value());

    // Should match the second filter
    auto result2
        = settings.findToleranceOverride("IntegrationGpuConvFwdHalf/Smoke.Correctness/params");
    ASSERT_TRUE(result2.has_value());
}

// ---------------------------------------------------------------------------
// Precedence: later entries win
// ---------------------------------------------------------------------------

TEST(TestSettingsParser, LaterEntriesTakePrecedence)
{
    const TempTomlFile file(R"(
[meta]
version = 1

[[tolerance_overrides]]
filters = ["*ConvFwd*"]
atol = 1e-3
rtol = 1e-2

[[tolerance_overrides]]
filters = ["*ConvFwd*Fp16*"]
atol = 5e-3
rtol = 5e-2
)");

    const TestSettings settings(file.path());

    // Matches both entries - the later (more specific) one should win
    auto result
        = settings.findToleranceOverride("IntegrationGpuConvFwd2dFp16/Smoke.Correctness/params");
    ASSERT_TRUE(result.has_value());
    EXPECT_FLOAT_EQ(result->atol, 5e-3F);
    EXPECT_FLOAT_EQ(result->rtol, 5e-2F);
}

TEST(TestSettingsParser, EarlierEntryUsedWhenLaterDoesNotMatch)
{
    const TempTomlFile file(R"(
[meta]
version = 1

[[tolerance_overrides]]
filters = ["*ConvFwd*"]
atol = 1e-3
rtol = 1e-2

[[tolerance_overrides]]
filters = ["*ConvFwd*Fp16*"]
atol = 5e-3
rtol = 5e-2
)");

    const TestSettings settings(file.path());

    // Matches only the first entry (Fp32, not Fp16)
    auto result
        = settings.findToleranceOverride("IntegrationGpuConvFwd2dFp32/Smoke.Correctness/params");
    ASSERT_TRUE(result.has_value());
    EXPECT_FLOAT_EQ(result->atol, 1e-3F);
    EXPECT_FLOAT_EQ(result->rtol, 1e-2F);
}

// ---------------------------------------------------------------------------
// [[test_skips]] parsing
// ---------------------------------------------------------------------------

TEST(TestSettingsParser, ParsesValidTomlWithSkips)
{
    const TempTomlFile file(R"(
[meta]
version = 1

[[test_skips]]
archs   = ["gfx1100", "gfx1101"]
filters = ["*Activation*Fp16*"]
reason  = "missing activation kernel on RDNA3"

[[test_skips]]
archs   = ["gfx942"]
filters = ["*ConvFwd3d*"]
reason  = "3d conv unimplemented on MI300"
)");

    const TestSettings settings(file.path());
    EXPECT_EQ(settings.skipEntryCount(), 2U);
}

TEST(TestSettingsParser, SkipsCoexistWithOverrides)
{
    const TempTomlFile file(R"(
[meta]
version = 1

[[tolerance_overrides]]
filters = ["*Fp16*"]
atol = 1e-3
rtol = 1e-2

[[test_skips]]
archs   = ["gfx942"]
filters = ["*Conv*"]
reason  = "test"
)");

    const TestSettings settings(file.path());
    EXPECT_EQ(settings.toleranceOverrideCount(), 1U);
    EXPECT_EQ(settings.skipEntryCount(), 1U);
}

TEST(TestSettingsParser, ParsesSkipWithoutArchsAsGlobal)
{
    // 'archs' omitted => global skip on every arch.
    const TempTomlFile file(R"(
[meta]
version = 1

[[test_skips]]
filters = ["*KnownBroken*"]
reason  = "tracked in #1234"
)");

    const TestSettings settings(file.path());
    EXPECT_EQ(settings.skipEntryCount(), 1U);
}

TEST(TestSettingsParser, ParsesSkipWithEmptyArchsAsGlobal)
{
    // 'archs = []' is also a global skip.
    const TempTomlFile file(R"(
[meta]
version = 1

[[test_skips]]
archs   = []
filters = ["*KnownBroken*"]
reason  = "tracked in #1234"
)");

    const TestSettings settings(file.path());
    EXPECT_EQ(settings.skipEntryCount(), 1U);
}

TEST(TestSettingsParser, ThrowsOnMissingSkipFilters)
{
    const TempTomlFile file(R"(
[meta]
version = 1

[[test_skips]]
archs  = ["gfx942"]
reason = "r"
)");

    EXPECT_THROW(const TestSettings settings(file.path()), std::runtime_error);
}

TEST(TestSettingsParser, ThrowsOnMissingSkipReason)
{
    const TempTomlFile file(R"(
[meta]
version = 1

[[test_skips]]
archs   = ["gfx942"]
filters = ["*test*"]
)");

    EXPECT_THROW(const TestSettings settings(file.path()), std::runtime_error);
}

TEST(TestSettingsParser, ThrowsOnEmptySkipReason)
{
    const TempTomlFile file(R"(
[meta]
version = 1

[[test_skips]]
archs   = ["gfx942"]
filters = ["*test*"]
reason  = ""
)");

    EXPECT_THROW(const TestSettings settings(file.path()), std::runtime_error);
}

// ---------------------------------------------------------------------------
// [[test_skips]] matching
// ---------------------------------------------------------------------------

TEST(TestSettingsParser, FindSkipMatchesArchSubstring)
{
    const TempTomlFile file(R"(
[meta]
version = 1

[[test_skips]]
archs   = ["gfx942"]
filters = ["*ConvFwd*"]
reason  = "no kernel"
)");

    const TestSettings settings(file.path());

    // Bare arch in TOML must match against the full ROCm-formatted device string.
    auto result
        = settings.findSkip("IntegrationGpuConvFwd2dFp16.Test", "gfx942:sramecc+:xnack-", "linux");
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, "no kernel");
}

TEST(TestSettingsParser, FindSkipMatchesExactArchString)
{
    // A more-qualified arch string in TOML still matches when the device
    // string is identical (the substring check is symmetric for equality).
    const TempTomlFile file(R"(
[meta]
version = 1

[[test_skips]]
archs   = ["gfx942:xnack-"]
filters = ["*Conv*"]
reason  = "xnack-only issue"
)");

    const TestSettings settings(file.path());

    auto matched
        = settings.findSkip("IntegrationGpuConvFwd2dFp16.Test", "gfx942:sramecc+:xnack-", "linux");
    ASSERT_FALSE(matched.has_value())
        << "literal substring 'gfx942:xnack-' should not appear in 'gfx942:sramecc+:xnack-'";

    auto matchedExact
        = settings.findSkip("IntegrationGpuConvFwd2dFp16.Test", "gfx942:xnack-", "linux");
    ASSERT_TRUE(matchedExact.has_value());
}

TEST(TestSettingsParser, FindSkipReturnsNulloptOnArchMismatch)
{
    const TempTomlFile file(R"(
[meta]
version = 1

[[test_skips]]
archs   = ["gfx1100"]
filters = ["*ConvFwd*"]
reason  = "rdna3 only"
)");

    const TestSettings settings(file.path());

    auto result
        = settings.findSkip("IntegrationGpuConvFwd2dFp16.Test", "gfx942:sramecc+:xnack-", "linux");
    EXPECT_FALSE(result.has_value());
}

TEST(TestSettingsParser, FindSkipReturnsNulloptOnFilterMismatch)
{
    const TempTomlFile file(R"(
[meta]
version = 1

[[test_skips]]
archs   = ["gfx942"]
filters = ["*Batchnorm*"]
reason  = "batchnorm only"
)");

    const TestSettings settings(file.path());

    auto result
        = settings.findSkip("IntegrationGpuConvFwd2dFp16.Test", "gfx942:sramecc+:xnack-", "linux");
    EXPECT_FALSE(result.has_value());
}

TEST(TestSettingsParser, FindSkipArchScopedReturnsNulloptOnEmptyDeviceArch)
{
    // Arch-scoped rule + empty device arch => substring search fails, no skip.
    const TempTomlFile file(R"(
[meta]
version = 1

[[test_skips]]
archs   = ["gfx942"]
filters = ["*ConvFwd*"]
reason  = "r"
)");

    const TestSettings settings(file.path());

    auto result = settings.findSkip("IntegrationGpuConvFwd2dFp16.Test", "", "linux");
    EXPECT_FALSE(result.has_value());
}

TEST(TestSettingsParser, FindSkipGlobalRuleMatchesAnyArch)
{
    const TempTomlFile file(R"(
[meta]
version = 1

[[test_skips]]
filters = ["*KnownBroken*"]
reason  = "global"
)");

    const TestSettings settings(file.path());

    // Global rule fires on any arch.
    EXPECT_TRUE(
        settings.findSkip("KnownBroken.Test", "gfx942:sramecc+:xnack-", "linux").has_value());
    EXPECT_TRUE(settings.findSkip("KnownBroken.Test", "gfx1100", "linux").has_value());
    EXPECT_TRUE(settings.findSkip("KnownBroken.Test", "anything", "linux").has_value());
    // ...even when the device arch could not be detected.
    EXPECT_TRUE(settings.findSkip("KnownBroken.Test", "", "linux").has_value());

    // But filter still has to match.
    EXPECT_FALSE(settings.findSkip("Other.Test", "gfx942", "linux").has_value());
}

TEST(TestSettingsParser, FindSkipReturnsNulloptWhenNoSkipsConfigured)
{
    const TempTomlFile file(R"(
[meta]
version = 1
)");

    const TestSettings settings(file.path());

    auto result = settings.findSkip("AnyTest.Name", "gfx942", "linux");
    EXPECT_FALSE(result.has_value());
}

TEST(TestSettingsParser, FindSkipFirstMatchWins)
{
    const TempTomlFile file(R"(
[meta]
version = 1

[[test_skips]]
archs   = ["gfx942"]
filters = ["*ConvFwd*"]
reason  = "first"

[[test_skips]]
archs   = ["gfx942"]
filters = ["*ConvFwd*Fp16*"]
reason  = "second more specific"
)");

    const TestSettings settings(file.path());

    auto result = settings.findSkip("IntegrationGpuConvFwd2dFp16.Test", "gfx942", "linux");
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, "first");
}

TEST(TestSettingsParser, FindSkipMatchesAnyArchInList)
{
    const TempTomlFile file(R"(
[meta]
version = 1

[[test_skips]]
archs   = ["gfx1100", "gfx1101", "gfx942"]
filters = ["*ConvFwd*"]
reason  = "broad arch list"
)");

    const TestSettings settings(file.path());

    EXPECT_TRUE(
        settings.findSkip("IntegrationGpuConvFwd.Test", "gfx942:xnack-", "linux").has_value());
    EXPECT_TRUE(settings.findSkip("IntegrationGpuConvFwd.Test", "gfx1100", "linux").has_value());
    EXPECT_FALSE(settings.findSkip("IntegrationGpuConvFwd.Test", "gfx906", "linux").has_value());
}

// ---------------------------------------------------------------------------
// [[test_skips]] platforms
// ---------------------------------------------------------------------------

TEST(TestSettingsParser, ParsesSkipWithPlatforms)
{
    const TempTomlFile file(R"(
[meta]
version = 1

[[test_skips]]
archs     = ["gfx1100"]
platforms = ["windows"]
filters   = ["*Foo*"]
reason    = "windows + RDNA3 only"
)");

    const TestSettings settings(file.path());
    EXPECT_EQ(settings.skipEntryCount(), 1U);
}

TEST(TestSettingsParser, FindSkipMatchesPlatform)
{
    const TempTomlFile file(R"(
[meta]
version = 1

[[test_skips]]
platforms = ["windows"]
filters   = ["*Foo*"]
reason    = "windows-only"
)");

    const TestSettings settings(file.path());

    auto winMatch = settings.findSkip("Foo.Test", "gfx942", "windows");
    ASSERT_TRUE(winMatch.has_value());
    EXPECT_EQ(*winMatch, "windows-only");

    auto linuxMiss = settings.findSkip("Foo.Test", "gfx942", "linux");
    EXPECT_FALSE(linuxMiss.has_value());
}

TEST(TestSettingsParser, FindSkipMatchesAnyPlatformInList)
{
    const TempTomlFile file(R"(
[meta]
version = 1

[[test_skips]]
platforms = ["windows", "linux"]
filters   = ["*Foo*"]
reason    = "broken everywhere"
)");

    const TestSettings settings(file.path());

    EXPECT_TRUE(settings.findSkip("Foo.Test", "gfx942", "windows").has_value());
    EXPECT_TRUE(settings.findSkip("Foo.Test", "gfx942", "linux").has_value());
}

TEST(TestSettingsParser, FindSkipPlatformIsExactNotSubstring)
{
    // Unlike archs, platform matching is exact equality. "win" must not match
    // "windows".
    const TempTomlFile file(R"(
[meta]
version = 1

[[test_skips]]
platforms = ["win"]
filters   = ["*Foo*"]
reason    = "should not fire"
)");

    const TestSettings settings(file.path());
    auto result = settings.findSkip("Foo.Test", "gfx942", "windows");
    EXPECT_FALSE(result.has_value());
}

TEST(TestSettingsParser, FindSkipOmittedPlatformsMatchesAny)
{
    // No 'platforms' field => entry matches any platform (the existing
    // arch-only behavior is unchanged).
    const TempTomlFile file(R"(
[meta]
version = 1

[[test_skips]]
archs   = ["gfx942"]
filters = ["*Foo*"]
reason  = "arch-only rule"
)");

    const TestSettings settings(file.path());

    EXPECT_TRUE(settings.findSkip("Foo.Test", "gfx942", "windows").has_value());
    EXPECT_TRUE(settings.findSkip("Foo.Test", "gfx942", "linux").has_value());
}

TEST(TestSettingsParser, FindSkipArchAndPlatformBothRequired)
{
    // When both 'archs' and 'platforms' are set, BOTH must match.
    const TempTomlFile file(R"(
[meta]
version = 1

[[test_skips]]
archs     = ["gfx1100"]
platforms = ["windows"]
filters   = ["*Foo*"]
reason    = "windows + gfx1100 only"
)");

    const TestSettings settings(file.path());

    // Both match
    EXPECT_TRUE(settings.findSkip("Foo.Test", "gfx1100", "windows").has_value());
    // Wrong arch
    EXPECT_FALSE(settings.findSkip("Foo.Test", "gfx942", "windows").has_value());
    // Wrong platform
    EXPECT_FALSE(settings.findSkip("Foo.Test", "gfx1100", "linux").has_value());
}

// NOLINTEND(readability-identifier-naming)
