// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <filesystem>
#include <fstream>
#include <string>

#include <gtest/gtest.h>

#include <hipdnn_test_sdk/utilities/ScopedEnvironmentVariableSetter.hpp>

#include "tests/utilities/ScratchDirectory.hpp"

/**
 * @file TestScratchDirectory.cpp
 * @brief Not gated on the ingestor: the helper is plain filesystem code, and a build with
 *        the ingestor off should still catch a regression in it.
 */
namespace hip_kernel_provider::tests
{
namespace
{

using hipdnn_test_sdk::utilities::ScopedDirectory;
using hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter;

constexpr const char* SCRATCH_LABEL = "scratchsuite";

[[nodiscard]] unsigned long counterOf(const std::filesystem::path& claimed)
{
    const std::string name = claimed.filename().string();
    const auto separator = name.rfind('_');
    EXPECT_NE(separator, std::string::npos) << name;
    return std::stoul(name.substr(separator + 1));
}

/// Points every variable temp_directory_path() consults at one place. Windows reads TMP
/// then TEMP; libstdc++ reads TMPDIR, TMP, TEMP, TEMPDIR. Setting all four means the test
/// does not depend on which of them the standard library happens to prefer.
class ScopedTempDirOverride
{
public:
    explicit ScopedTempDirOverride(const std::string& value)
        : _tmpdir("TMPDIR", value)
        , _tmp("TMP", value)
        , _temp("TEMP", value)
        , _tempdir("TEMPDIR", value)
    {
    }

private:
    ScopedEnvironmentVariableSetter _tmpdir;
    ScopedEnvironmentVariableSetter _tmp;
    ScopedEnvironmentVariableSetter _temp;
    ScopedEnvironmentVariableSetter _tempdir;
};

/// The counter only moves forward, so re-creating a name a claim already returned would
/// never collide: the name to squat on has to be computed from it. The sentinel is what
/// separates retry from clear -- a helper that cleared the taken name would pass every
/// other assertion here.
TEST(TestScratchDirectory, WalksPastANameSomethingElseAlreadyHoldsAndLeavesItIntact)
{
    const ScopedDirectory first = claimScratchDirectory(SCRATCH_LABEL);
    const std::string name = first.path().filename().string();
    const auto separator = name.rfind('_');
    ASSERT_NE(separator, std::string::npos) << name;
    const unsigned long firstCounter = counterOf(first.path());

    const std::filesystem::path squatted
        = first.path().parent_path()
          / (name.substr(0, separator + 1) + std::to_string(firstCounter + 1));
    ASSERT_TRUE(std::filesystem::create_directory(squatted));
    const std::filesystem::path sentinel = squatted / "held-by-someone-else";
    std::ofstream{sentinel} << "occupied";
    ASSERT_TRUE(std::filesystem::exists(sentinel));

    const ScopedDirectory second = claimScratchDirectory(SCRATCH_LABEL);
    EXPECT_NE(second.path(), squatted);
    EXPECT_TRUE(std::filesystem::is_directory(second.path()));
    EXPECT_TRUE(std::filesystem::is_directory(squatted));
    EXPECT_TRUE(std::filesystem::exists(sentinel));
    EXPECT_GT(counterOf(second.path()), firstCounter);

    std::filesystem::remove_all(squatted);
}

/// WalksPastANameSomethingElseAlreadyHoldsAndLeavesItIntact computes the next name by
/// taking the trailing counter apart, so it only proves anything for as long as the names
/// keep that shape.
TEST(TestScratchDirectory, NamesTheDirectoryAfterItsLabelAndACounter)
{
    const ScopedDirectory claimed = claimScratchDirectory(SCRATCH_LABEL);
    const std::string name = claimed.path().filename().string();

    EXPECT_EQ(name.rfind(std::string("hkp_") + SCRATCH_LABEL + '_', 0), 0U) << name;
    const auto separator = name.rfind('_');
    ASSERT_NE(separator, std::string::npos) << name;
    EXPECT_EQ(name.find_first_not_of("0123456789", separator + 1), std::string::npos) << name;
    EXPECT_TRUE(std::filesystem::is_directory(claimed.path()));
}

/// A handler with the two catches ordered the other way round compiles, passes the two cases
/// above, and reports a missing temp directory as a shortage of free names.
TEST(TestScratchDirectory, ReportsAnUnusableTempDirectoryRatherThanNameExhaustion)
{
    const ScopedDirectory real = claimScratchDirectory(SCRATCH_LABEL);
    const std::filesystem::path absent = real.path() / "no-such-directory";
    ASSERT_FALSE(std::filesystem::exists(absent));

    const ScopedTempDirOverride override(absent.string());
    EXPECT_THROW((void)claimScratchDirectory(SCRATCH_LABEL), std::filesystem::filesystem_error);
}

} // namespace
} // namespace hip_kernel_provider::tests
