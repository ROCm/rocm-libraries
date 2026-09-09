// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <filesystem>
#include <fstream>
#include <string>

#include <gtest/gtest.h>

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

constexpr const char* SCRATCH_LABEL = "scratchsuite";

[[nodiscard]] unsigned long counterOf(const std::filesystem::path& claimed)
{
    const std::string name = claimed.filename().string();
    const auto separator = name.rfind('_');
    EXPECT_NE(separator, std::string::npos) << name;
    return std::stoul(name.substr(separator + 1));
}

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
    // Separates retry from clear: a helper that cleared the taken name would pass every other
    // assertion here.
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

// WalksPastANameSomethingElseAlreadyHoldsAndLeavesItIntact takes the trailing counter apart,
// so it only proves anything while the names keep this shape.
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

// The base is named directly rather than pointed at through TMP/TEMP/TMPDIR: an override the
// platform ignores would leave this claiming under the real temp dir. See ScratchDirectory.hpp.
TEST(TestScratchDirectory, ReportsAnUnusableTempDirectoryRatherThanNameExhaustion)
{
    const ScopedDirectory real = claimScratchDirectory(SCRATCH_LABEL);
    const std::filesystem::path absent = real.path() / "no-such-directory";
    ASSERT_FALSE(std::filesystem::exists(absent));

    // With the two catches ordered the other way round, this reports as name exhaustion.
    EXPECT_THROW((void)claimScratchDirectoryUnder(absent, SCRATCH_LABEL),
                 std::filesystem::filesystem_error);
}

} // namespace
} // namespace hip_kernel_provider::tests
