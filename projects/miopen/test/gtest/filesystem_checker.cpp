// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <miopen/filesystem_checker.hpp>
#include <gtest/gtest.h>
#include "gtest_common.hpp"

#ifdef __linux__
#include <linux/magic.h>
#include <sys/vfs.h>
#include <unistd.h>

namespace fs = miopen::fs;

namespace {
struct FilesystemTypeCase
{
    unsigned long type;
    bool network;
    const char* name;
};
} // namespace

class CPU_FilesystemTypes_NONE : public testing::TestWithParam<FilesystemTypeCase>
{
};

TEST_P(CPU_FilesystemTypes_NONE, ClassifiesNetworkAndLocalStorage)
{
    EXPECT_EQ(miopen::IsNetworkedFilesystemType(GetParam().type), GetParam().network);
}

INSTANTIATE_TEST_SUITE_P(Unit,
                         CPU_FilesystemTypes_NONE,
                         testing::Values(FilesystemTypeCase{0x6969, true, "NFS"},
                                         FilesystemTypeCase{0x517b, true, "SMB"},
                                         FilesystemTypeCase{0xfe534d42, true, "SMB2"},
                                         FilesystemTypeCase{0xff534d42, true, "CIFS"},
                                         FilesystemTypeCase{0x73757245, true, "Coda"},
                                         FilesystemTypeCase{0x7461636f, true, "OCFS2"},
                                         FilesystemTypeCase{0x5346414f, true, "AFS"},
                                         FilesystemTypeCase{0x0bd00bd0, true, "Lustre"},
                                         FilesystemTypeCase{0x00c36400, true, "Ceph"},
                                         FilesystemTypeCase{0x47504653, true, "GPFS"},
                                         FilesystemTypeCase{0x19830326, true, "BeeGFS"},
                                         FilesystemTypeCase{0xaad7aaea, true, "PanFS"},
                                         FilesystemTypeCase{0x01021997, true, "NineP"},
                                         FilesystemTypeCase{0xef51, false, "Ext2Old"},
                                         FilesystemTypeCase{0xef53, false, "Ext"},
                                         FilesystemTypeCase{0x01021994, false, "Tmpfs"},
                                         FilesystemTypeCase{0x794c7630, false, "Overlay"},
                                         FilesystemTypeCase{PROC_SUPER_MAGIC, false, "Proc"},
                                         FilesystemTypeCase{0, false, "Unknown"}),
                         [](const testing::TestParamInfo<FilesystemTypeCase>& info) {
                             return info.param.name;
                         });

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

TEST(CPU_FilesystemChecker_NONE, MissingRelativeAncestorUsesCurrentFilesystem)
{
    const fs::path missing =
        fs::path{"miopen-filesystem-checker-missing-" + std::to_string(getpid())} / "cache";
    ASSERT_FALSE(fs::exists(missing.parent_path()));
    struct statfs stat;
    ASSERT_EQ(statfs(".", &stat), 0);

    const auto& checker = miopen::GetFilesystemChecker();
    ScopedEnvironment<std::string> log_level{MIOPEN_LOG_LEVEL, "3"};

    testing::internal::CaptureStderr();
    const bool ancestor_is_networked = checker.IsNetworkedFilesystem(".");
    const bool missing_is_networked  = checker.IsNetworkedFilesystem(missing);
    const auto errors                = testing::internal::GetCapturedStderr();

    // Equal classifications alone could pass accidentally on a local filesystem
    // if statfs failed. Also require that detection reports no filesystem error.
    EXPECT_EQ(missing_is_networked, ancestor_is_networked);
    EXPECT_TRUE(errors.empty()) << errors;
    EXPECT_FALSE(fs::exists(missing.parent_path()));
}
#endif // __linux__
