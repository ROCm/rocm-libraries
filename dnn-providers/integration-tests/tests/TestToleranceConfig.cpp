// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

#include "harness/ToleranceConfig.hpp"

using hipdnn_integration_tests::ToleranceConfig;

// NOLINTBEGIN(readability-identifier-naming) -- gtest macro-generated names

namespace
{

// Helper to create a temporary TOML file that is auto-deleted on destruction.
class TempTomlFile
{
public:
    explicit TempTomlFile(const std::string& content)
        : _path(std::filesystem::temp_directory_path()
                / ("test_tolerance_config_" + std::to_string(std::rand()) + ".toml"))
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

TEST(TestToleranceConfig, ParsesValidTomlWithOverrides)
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

    const ToleranceConfig config(file.path());
    EXPECT_EQ(config.overrideCount(), 2U);
}

TEST(TestToleranceConfig, ParsesValidTomlWithNoOverrides)
{
    const TempTomlFile file(R"(
[meta]
version = 1
)");

    const ToleranceConfig config(file.path());
    EXPECT_EQ(config.overrideCount(), 0U);
}

TEST(TestToleranceConfig, ThrowsOnMissingVersion)
{
    const TempTomlFile file(R"(
[meta]
)");

    EXPECT_THROW(const ToleranceConfig config(file.path()), std::runtime_error);
}

TEST(TestToleranceConfig, ThrowsOnUnsupportedVersion)
{
    const TempTomlFile file(R"(
[meta]
version = 99
)");

    EXPECT_THROW(const ToleranceConfig config(file.path()), std::runtime_error);
}

TEST(TestToleranceConfig, ThrowsOnMissingFilters)
{
    const TempTomlFile file(R"(
[meta]
version = 1

[[tolerance_overrides]]
atol = 1e-3
rtol = 1e-2
)");

    EXPECT_THROW(const ToleranceConfig config(file.path()), std::runtime_error);
}

TEST(TestToleranceConfig, ThrowsOnMissingAtol)
{
    const TempTomlFile file(R"(
[meta]
version = 1

[[tolerance_overrides]]
filters = ["*test*"]
rtol = 1e-2
)");

    EXPECT_THROW(const ToleranceConfig config(file.path()), std::runtime_error);
}

TEST(TestToleranceConfig, ThrowsOnMissingRtol)
{
    const TempTomlFile file(R"(
[meta]
version = 1

[[tolerance_overrides]]
filters = ["*test*"]
atol = 1e-3
)");

    EXPECT_THROW(const ToleranceConfig config(file.path()), std::runtime_error);
}

TEST(TestToleranceConfig, ThrowsOnNonexistentFile)
{
    EXPECT_THROW(const ToleranceConfig config("/nonexistent/path.toml"), std::exception);
}

// ---------------------------------------------------------------------------
// Filter matching
// ---------------------------------------------------------------------------

TEST(TestToleranceConfig, FindOverrideMatchesWildcard)
{
    const TempTomlFile file(R"(
[meta]
version = 1

[[tolerance_overrides]]
filters = ["*ConvFwd*Fp16*"]
atol = 1e-3
rtol = 1e-2
)");

    const ToleranceConfig config(file.path());

    auto result = config.findOverride("IntegrationGpuConvFwd2dFp16/Smoke.Correctness/NCHW_params");
    ASSERT_TRUE(result.has_value());
    EXPECT_FLOAT_EQ(result->atol, 1e-3F);
    EXPECT_FLOAT_EQ(result->rtol, 1e-2F);
}

TEST(TestToleranceConfig, FindOverrideReturnsNulloptWhenNoMatch)
{
    const TempTomlFile file(R"(
[meta]
version = 1

[[tolerance_overrides]]
filters = ["*ConvFwd*Fp16*"]
atol = 1e-3
rtol = 1e-2
)");

    const ToleranceConfig config(file.path());

    auto result = config.findOverride("IntegrationGpuBatchnormFp32/Smoke.Correctness/params");
    EXPECT_FALSE(result.has_value());
}

TEST(TestToleranceConfig, FindOverrideReturnsNulloptWhenNoOverrides)
{
    const TempTomlFile file(R"(
[meta]
version = 1
)");

    const ToleranceConfig config(file.path());

    auto result = config.findOverride("AnyTestName");
    EXPECT_FALSE(result.has_value());
}

TEST(TestToleranceConfig, FindOverrideMatchesMultipleFiltersInEntry)
{
    const TempTomlFile file(R"(
[meta]
version = 1

[[tolerance_overrides]]
filters = ["*Fp16*", "*Half*"]
atol = 1e-3
rtol = 1e-2
)");

    const ToleranceConfig config(file.path());

    // Should match the first filter
    auto result1 = config.findOverride("IntegrationGpuConvFwd2dFp16/Smoke.Correctness/params");
    ASSERT_TRUE(result1.has_value());

    // Should match the second filter
    auto result2 = config.findOverride("IntegrationGpuConvFwdHalf/Smoke.Correctness/params");
    ASSERT_TRUE(result2.has_value());
}

// ---------------------------------------------------------------------------
// Precedence: later entries win
// ---------------------------------------------------------------------------

TEST(TestToleranceConfig, LaterEntriesTakePrecedence)
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

    const ToleranceConfig config(file.path());

    // Matches both entries - the later (more specific) one should win
    auto result = config.findOverride("IntegrationGpuConvFwd2dFp16/Smoke.Correctness/params");
    ASSERT_TRUE(result.has_value());
    EXPECT_FLOAT_EQ(result->atol, 5e-3F);
    EXPECT_FLOAT_EQ(result->rtol, 5e-2F);
}

TEST(TestToleranceConfig, EarlierEntryUsedWhenLaterDoesNotMatch)
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

    const ToleranceConfig config(file.path());

    // Matches only the first entry (Fp32, not Fp16)
    auto result = config.findOverride("IntegrationGpuConvFwd2dFp32/Smoke.Correctness/params");
    ASSERT_TRUE(result.has_value());
    EXPECT_FLOAT_EQ(result->atol, 1e-3F);
    EXPECT_FLOAT_EQ(result->rtol, 1e-2F);
}

// NOLINTEND(readability-identifier-naming)
