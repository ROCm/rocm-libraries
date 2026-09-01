// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <atomic>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <hipdnn_data_sdk/utilities/CacheRoot.hpp>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_test_sdk/utilities/ScopedEnvironmentVariableSetter.hpp>
#include <optional>
#include <string>

using namespace hipdnn_data_sdk::utilities;
using hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter;

namespace
{

std::filesystem::path makeUniqueTempDir()
{
    static std::atomic<int> s_counter{0};
    const auto unique = std::to_string(::testing::UnitTest::GetInstance()->random_seed()) + "_"
                        + std::to_string(s_counter++);
    return std::filesystem::temp_directory_path() / ("hipdnn_test_cacheroot_" + unique);
}

} // namespace

/// Every test here neutralizes HIPDNN_DISABLE_CACHE for its own duration. An ambient
/// truthy value in the runner's environment makes cacheRoot() return empty before it
/// reads anything else, which would turn each case below into a test of the kill switch
/// it did not mean to exercise.
class TestCacheRoot : public ::testing::Test
{
protected:
    void SetUp() override
    {
        _disableCache.emplace("HIPDNN_DISABLE_CACHE", "");
    }

    void TearDown() override
    {
        _disableCache.reset();
        if(!_cleanupPath.empty())
        {
            std::error_code ignored;
            std::filesystem::remove_all(_cleanupPath, ignored);
        }
    }

    std::filesystem::path _cleanupPath;

private:
    std::optional<ScopedEnvironmentVariableSetter> _disableCache;
};

TEST_F(TestCacheRoot, UnsetEnvResolvesToPlatformDefaultAndDirectoryExists)
{
    const auto fakeHome = makeUniqueTempDir();
    _cleanupPath = fakeHome;
    std::filesystem::remove_all(fakeHome);

#if defined(__linux__)
    const ScopedEnvironmentVariableSetter home("HOME", fakeHome.string());
#else
    const ScopedEnvironmentVariableSetter home("USERPROFILE", fakeHome.string());
#endif
    unsetEnv("HIPDNN_CACHE_DIR");

    const auto root = cacheRoot();

    ASSERT_FALSE(root.empty());
    EXPECT_TRUE(std::filesystem::is_directory(root));
    EXPECT_EQ(root.string().rfind(fakeHome.string(), 0), 0u);
}

TEST_F(TestCacheRoot, CustomWritableDirIsUsedAndCreated)
{
    const auto customDir = makeUniqueTempDir();
    _cleanupPath = customDir;
    std::filesystem::remove_all(customDir);

    const ScopedEnvironmentVariableSetter cacheDir("HIPDNN_CACHE_DIR", customDir.string());

    const auto root = cacheRoot();

    EXPECT_EQ(root, customDir);
    EXPECT_TRUE(std::filesystem::is_directory(root));
}

/// The advertised degradation path. Uses a location no user can create -- /proc rejects
/// mkdir for root as well -- rather than a permission bit, since ROCm CI containers run
/// as root and root ignores directory permissions. A GTEST_SKIP() here would leave the
/// behaviour uncovered exactly where it runs.
TEST_F(TestCacheRoot, UnwritableLocationDegradesInsteadOfThrowing)
{
#if defined(__linux__)
    const std::string target = "/proc/hipdnn-cache-root-must-fail";
#else
    // A reserved device name: CreateDirectoryW refuses it at any privilege level.
    const std::string target = "C:\\NUL\\hipdnn-cache-root-must-fail";
#endif
    const ScopedEnvironmentVariableSetter cacheDir("HIPDNN_CACHE_DIR", target);

    std::filesystem::path root;
    EXPECT_NO_THROW(root = cacheRoot());

    EXPECT_TRUE(root.empty()) << "cacheRoot() returned " << root
                              << " for a location that cannot be created";
    std::error_code ignored;
    EXPECT_FALSE(std::filesystem::exists(target, ignored));
}

TEST_F(TestCacheRoot, PathCollidingWithAnExistingFileDegrades)
{
    const auto parentDir = makeUniqueTempDir();
    _cleanupPath = parentDir;
    std::filesystem::remove_all(parentDir);
    std::filesystem::create_directories(parentDir);

    const auto filePath = parentDir / "not_a_directory";
    {
        std::ofstream(filePath) << "occupied";
    }

    const ScopedEnvironmentVariableSetter cacheDir("HIPDNN_CACHE_DIR", filePath.string());

    std::filesystem::path root;
    EXPECT_NO_THROW(root = cacheRoot());

    EXPECT_TRUE(root.empty());
}

class TestCacheRootDisableTokens : public ::testing::TestWithParam<std::string>
{
};

/// The documented kill switch. Its token set matches HIPDNN_FORCE_BENCHMARKING's,
/// case-insensitive and whitespace-trimmed, and a truthy value wins over an explicit
/// HIPDNN_CACHE_DIR -- so this also pins the precedence and the no-directory guarantee.
TEST_P(TestCacheRootDisableTokens, ATruthyValueDisablesTheCacheAndOutranksCacheDir)
{
    const auto customDir = makeUniqueTempDir();
    std::filesystem::remove_all(customDir);

    const ScopedEnvironmentVariableSetter cacheDir("HIPDNN_CACHE_DIR", customDir.string());
    const ScopedEnvironmentVariableSetter disabled("HIPDNN_DISABLE_CACHE", GetParam());

    EXPECT_TRUE(cacheRoot().empty()) << "\"" << GetParam() << "\" did not disable the cache";
    std::error_code ignored;
    EXPECT_FALSE(std::filesystem::exists(customDir, ignored))
        << "the kill switch created a directory it promises never to touch";
}

INSTANTIATE_TEST_SUITE_P(
    TruthyTokensAndCaseVariants,
    TestCacheRootDisableTokens,
    ::testing::Values("1", "true", "TRUE", "  True  ", "on", "yes", "enable", "enabled"));

class TestCacheRootNonDisablingTokens : public ::testing::TestWithParam<std::string>
{
};

/// A typo must fail OPEN: an unrecognized value leaves the cache enabled rather than
/// silently turning it off, which would be indistinguishable from a broken cache.
TEST_P(TestCacheRootNonDisablingTokens, AnUnrecognizedValueLeavesTheCacheEnabled)
{
    const auto customDir = makeUniqueTempDir();
    std::filesystem::remove_all(customDir);

    const ScopedEnvironmentVariableSetter cacheDir("HIPDNN_CACHE_DIR", customDir.string());
    const ScopedEnvironmentVariableSetter disabled("HIPDNN_DISABLE_CACHE", GetParam());

    const auto root = cacheRoot();
    EXPECT_EQ(root, customDir) << "\"" << GetParam() << "\" was treated as truthy";

    std::error_code ignored;
    std::filesystem::remove_all(customDir, ignored);
}

INSTANTIATE_TEST_SUITE_P(UnsetEmptyAndTypos,
                         TestCacheRootNonDisablingTokens,
                         ::testing::Values("", "0", "false", "off", "no", "disable", "ture"));
