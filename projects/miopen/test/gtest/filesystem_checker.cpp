// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <miopen/filesystem_checker.hpp>
#include <gtest/gtest.h>
#include "gtest_common.hpp"

#ifdef __linux__
#include <linux/magic.h>
#include <sys/vfs.h>

namespace fs = miopen::fs;

TEST(CPU_FilesystemChecker_NONE, LocalFilesystemAndMissingAbsoluteAncestor)
{
    // procfs provides a local, read-only filesystem without requiring CI to mount one.
    const fs::path ancestor = "/proc/self";
    struct statfs stat;
    if(statfs(ancestor.c_str(), &stat) != 0 || stat.f_type != PROC_SUPER_MAGIC)
        GTEST_SKIP() << "procfs is unavailable";

    const auto missing = ancestor / "miopen-filesystem-checker-missing" / "cache";
    ASSERT_FALSE(fs::exists(missing.parent_path()));
    const auto& checker = miopen::GetFilesystemChecker();
    ScopedEnvironment<std::string> log_level{MIOPEN_LOG_LEVEL, "3"};

    testing::internal::CaptureStderr();
    const bool ancestor_is_networked = checker.IsNetworkedFilesystem(ancestor);
    const bool missing_is_networked  = checker.IsNetworkedFilesystem(missing);
    const auto errors                = testing::internal::GetCapturedStderr();

    EXPECT_FALSE(ancestor_is_networked);
    EXPECT_FALSE(missing_is_networked);
    EXPECT_TRUE(errors.empty()) << errors;
    EXPECT_FALSE(fs::exists(missing.parent_path()));
}

TEST(CPU_FilesystemChecker_NONE, MissingRelativePathUsesCurrentFilesystem)
{
    const fs::path missing = "miopen-filesystem-checker-missing-relative/cache";
    ASSERT_FALSE(fs::exists(missing.parent_path()));
    const auto& checker = miopen::GetFilesystemChecker();
    ScopedEnvironment<std::string> log_level{MIOPEN_LOG_LEVEL, "3"};

    testing::internal::CaptureStderr();
    const bool current_is_networked = checker.IsNetworkedFilesystem(fs::current_path());
    const bool missing_is_networked = checker.IsNetworkedFilesystem(missing);
    const auto errors               = testing::internal::GetCapturedStderr();

    EXPECT_EQ(missing_is_networked, current_is_networked);
    EXPECT_TRUE(errors.empty()) << errors;
    EXPECT_FALSE(fs::exists(missing.parent_path()));
}

#endif // __linux__
