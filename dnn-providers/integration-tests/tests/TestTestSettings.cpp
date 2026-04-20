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

// NOLINTEND(readability-identifier-naming)
