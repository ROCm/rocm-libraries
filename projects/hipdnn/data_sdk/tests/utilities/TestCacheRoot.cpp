// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <atomic>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <hipdnn_data_sdk/utilities/CacheRoot.hpp>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_test_sdk/utilities/ScopedEnvironmentVariableSetter.hpp>
#include <string>

#if defined(__linux__)
#include <sys/stat.h>
#endif

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

class TestCacheRoot : public ::testing::Test
{
protected:
    void TearDown() override
    {
        if(!_cleanupPath.empty())
        {
            std::error_code ignored;
            std::filesystem::remove_all(_cleanupPath, ignored);
        }
    }

    std::filesystem::path _cleanupPath;
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
    // The default is rooted under the fake home directory expandUser() resolved to.
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

#if defined(__linux__)
TEST_F(TestCacheRoot, UnwritableLocationDegradesInsteadOfThrowing)
{
    const auto parentDir = makeUniqueTempDir();
    _cleanupPath = parentDir;
    std::filesystem::remove_all(parentDir);
    std::filesystem::create_directories(parentDir);
    ::chmod(parentDir.c_str(), 0500); // read + execute only, no write

    const auto target = parentDir / "subcache";
    const ScopedEnvironmentVariableSetter cacheDir("HIPDNN_CACHE_DIR", target.string());

    std::filesystem::path root;
    EXPECT_NO_THROW(root = cacheRoot());

    EXPECT_TRUE(root.empty());

    // Restore write permission so TearDown() can remove the directory.
    ::chmod(parentDir.c_str(), 0700);
}
#endif // defined(__linux__)

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
