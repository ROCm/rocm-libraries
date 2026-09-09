// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <string>

#include "ScratchDirectory.hpp"

#include <hipdnn_test_sdk/utilities/ScopedEnvironmentVariableSetter.hpp>

using hipdnn_integration_tests::scratch::makeDir;
using hipdnn_test_sdk::utilities::ScopedDirectory;
using hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter;

// NOLINTBEGIN(readability-identifier-naming) -- gtest macro-generated names

namespace
{

constexpr const char* SCRATCH_PREFIX = "scratchsuite_";

unsigned long long counterOf(const std::filesystem::path& claimed)
{
    const std::string name = claimed.filename().string();
    const auto separator = name.rfind('_');
    EXPECT_NE(separator, std::string::npos) << name;
    return std::stoull(name.substr(separator + 1));
}

// Points every variable temp_directory_path() consults at one place. Windows reads TMP
// then TEMP; libstdc++ reads TMPDIR, TMP, TEMP, TEMPDIR. Setting all four means the test
// does not depend on which of them the standard library happens to prefer.
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

} // namespace

// The counter only moves forward, so the name to squat on has to be computed from the one
// already claimed. The sentinel separates retry from clear: a helper that called remove_all
// on the taken name would pass every other assertion here and still delete a live sibling
// process's fixture under `ctest -j`.
TEST(TestScratchDirectory, WalksPastANameSomethingElseAlreadyHoldsAndLeavesItIntact)
{
    const ScopedDirectory first = makeDir(SCRATCH_PREFIX);
    const std::string name = first.path().filename().string();
    const auto separator = name.rfind('_');
    ASSERT_NE(separator, std::string::npos) << name;
    const unsigned long long firstCounter = counterOf(first.path());

    const std::filesystem::path squatted
        = first.path().parent_path()
          / (name.substr(0, separator + 1) + std::to_string(firstCounter + 1));
    ASSERT_TRUE(std::filesystem::create_directory(squatted));
    const std::filesystem::path sentinel = squatted / "held-by-someone-else";
    std::ofstream{sentinel} << "occupied";
    ASSERT_TRUE(std::filesystem::exists(sentinel));

    const ScopedDirectory second = makeDir(SCRATCH_PREFIX);
    EXPECT_NE(second.path(), squatted);
    EXPECT_TRUE(std::filesystem::is_directory(second.path()));
    EXPECT_TRUE(std::filesystem::is_directory(squatted));
    EXPECT_TRUE(std::filesystem::exists(sentinel));
    EXPECT_GT(counterOf(second.path()), firstCounter);

    std::filesystem::remove_all(squatted);
}

// WalksPastANameSomethingElseAlreadyHoldsAndLeavesItIntact computes the next name by
// taking the trailing counter apart, so it only proves anything for as long as the names
// keep that shape.
TEST(TestScratchDirectory, NamesTheDirectoryAfterItsPrefixAndACounter)
{
    const ScopedDirectory claimed = makeDir(SCRATCH_PREFIX);
    const std::string name = claimed.path().filename().string();

    EXPECT_EQ(name.rfind(SCRATCH_PREFIX, 0), 0U) << name;
    EXPECT_EQ(name.find_first_not_of("0123456789", std::string(SCRATCH_PREFIX).size()),
              std::string::npos)
        << name;
    EXPECT_TRUE(std::filesystem::is_directory(claimed.path()));
}

// A handler with the two catches ordered the other way round compiles, passes the two cases
// above, and reports an unwritable temp directory as a shortage of free names.
TEST(TestScratchDirectory, ReportsAnUnusableTempDirectoryRatherThanNameExhaustion)
{
    const ScopedDirectory real = makeDir(SCRATCH_PREFIX);
    const std::filesystem::path absent = real.path() / "no-such-directory";
    ASSERT_FALSE(std::filesystem::exists(absent));

    const ScopedTempDirOverride override(absent.string());
    EXPECT_THROW((void)makeDir(SCRATCH_PREFIX), std::filesystem::filesystem_error);
}

// NOLINTEND(readability-identifier-naming)
