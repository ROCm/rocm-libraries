// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Host-only unit tests for cgroup_memory_probe.hpp: no GPU/HIP calls, no real cgroup setup.

#include "cgroup_memory_probe.hpp"

#include <gtest/gtest.h>

#include <map>
#include <string>

using hipblaslt_client::cgroup_available_memory_from;
using hipblaslt_client::cgroup_mount;
using hipblaslt_client::cgroup_paths;
using hipblaslt_client::parse_cgroup_size_token;
using hipblaslt_client::parse_mountinfo;
using hipblaslt_client::parse_proc_self_cgroup;
using hipblaslt_client::pick_cgroup_mount;
using hipblaslt_client::resolve_cgroup_directory;

namespace
{
    size_t const kUnlimitedSentinel = static_cast<size_t>(1) << 62;
    size_t const kGiB               = 1024u * 1024u * 1024u; // 1 GiB; fake sysfs strings below are byte counts in the same units

    using FakeFs = std::map<std::string, std::string>;

    char const kDefaultMountinfo[] =
        "100 90 0:1 / /sys/fs/cgroup rw shared:1 - cgroup2 cgroup2 rw\n"
        "101 90 0:2 / /sys/fs/cgroup/memory rw shared:1 - cgroup memory rw\n";

    size_t headroom(cgroup_paths const& paths,
                    FakeFs const&         files,
                    char const*           mountinfo = kDefaultMountinfo)
    {
        return cgroup_available_memory_from(paths, mountinfo, files);
    }
} // namespace

// ---------------------------------------------------------------------------
// parse_proc_self_cgroup
// ---------------------------------------------------------------------------

struct ParseProcSelfCgroupCase
{
    const char* name;
    const char* input;
    const char* expected_v2;
    const char* expected_v1;
};

class ParseProcSelfCgroupTest : public testing::TestWithParam<ParseProcSelfCgroupCase>
{
};

TEST_P(ParseProcSelfCgroupTest, ParsesExpectedPaths)
{
    auto const& param = GetParam();
    auto        paths = parse_proc_self_cgroup(param.input);
    EXPECT_EQ(paths.v2, param.expected_v2);
    EXPECT_EQ(paths.v1, param.expected_v1);
}

INSTANTIATE_TEST_SUITE_P(
    CgroupParse,
    ParseProcSelfCgroupTest,
    testing::Values(
        ParseProcSelfCgroupCase{"v2_unified", "0::/user.slice/app.slice\n", "/user.slice/app.slice", ""},
        ParseProcSelfCgroupCase{"v1_memory_only", "5:memory:/docker/abc\n", "", "/docker/abc"},
        ParseProcSelfCgroupCase{
            "v1_comounted", "12:memory,cpu:/same/path\n", "", "/same/path"},
        ParseProcSelfCgroupCase{
            "cpu_only_path_with_memory_substring", "12:cpu:/foo/memory/bar\n", "", ""},
        ParseProcSelfCgroupCase{"hybrid",
                                "12:memory,cpu:/docker/abc\n0::/user.slice/app.slice\n",
                                "/user.slice/app.slice",
                                "/docker/abc"},
        ParseProcSelfCgroupCase{"empty", "", "", ""},
        ParseProcSelfCgroupCase{"malformed", "garbage\n", "", ""},
        ParseProcSelfCgroupCase{"trailing_crlf", "0::/path\r\n", "/path", ""}),
    [](testing::TestParamInfo<ParseProcSelfCgroupCase> const& info) { return info.param.name; });

// ---------------------------------------------------------------------------
// parse_cgroup_size_token
// ---------------------------------------------------------------------------

TEST(ParseCgroupSizeToken, MaxMeansUnlimited)
{
    size_t out = 0;
    ASSERT_TRUE(parse_cgroup_size_token("max", out, kUnlimitedSentinel));
    EXPECT_EQ(out, std::numeric_limits<size_t>::max());
}

TEST(ParseCgroupSizeToken, ParsesNumericLimit)
{
    size_t out = 0;
    ASSERT_TRUE(parse_cgroup_size_token("1073741824", out, kUnlimitedSentinel));
    EXPECT_EQ(out, kGiB);
}

TEST(ParseCgroupSizeToken, HugeSentinelMeansUnlimited)
{
    size_t out = 0;
    ASSERT_TRUE(parse_cgroup_size_token("9223372036854771712", out, kUnlimitedSentinel));
    EXPECT_EQ(out, std::numeric_limits<size_t>::max());
}

TEST(ParseCgroupSizeToken, RejectsNonNumeric)
{
    size_t out = 123;
    EXPECT_FALSE(parse_cgroup_size_token("abc", out, kUnlimitedSentinel));
}

TEST(ParseCgroupSizeToken, RejectsEmpty)
{
    size_t out = 123;
    EXPECT_FALSE(parse_cgroup_size_token("", out, kUnlimitedSentinel));
}

// ---------------------------------------------------------------------------
// mountinfo resolution
// ---------------------------------------------------------------------------

TEST(ParseMountinfo, FindsCoMountedV1MemoryController)
{
    char const mountinfo[] =
        "200 199 0:55 / /sys/fs/cgroup/cpu,memory rw shared:1 - cgroup cpu,memory rw,memory,cpu\n";
    auto mounts = parse_mountinfo(mountinfo);
    ASSERT_EQ(mounts.size(), 1u);
    EXPECT_EQ(mounts[0].mountpoint, "/sys/fs/cgroup/cpu,memory");

    cgroup_paths paths;
    paths.v1 = "/docker/abc";
    auto const* mount = pick_cgroup_mount(mounts, false, paths.v1);
    ASSERT_NE(mount, nullptr);
    EXPECT_EQ(resolve_cgroup_directory(*mount, paths.v1),
              "/sys/fs/cgroup/cpu,memory/docker/abc");
}

TEST(ResolveCgroupDirectory, NonRootV2MountRoot)
{
    char const mountinfo[] =
        "300 299 0:56 /user.slice /sys/fs/cgroup/user.slice rw shared:1 - cgroup2 cgroup2 rw\n";
    auto mounts = parse_mountinfo(mountinfo);
    ASSERT_EQ(mounts.size(), 1u);

    cgroup_paths paths;
    paths.v2 = "/user.slice/app";
    auto const* mount = pick_cgroup_mount(mounts, true, paths.v2);
    ASSERT_NE(mount, nullptr);
    EXPECT_EQ(resolve_cgroup_directory(*mount, paths.v2), "/sys/fs/cgroup/user.slice/app");
}

// ---------------------------------------------------------------------------
// cgroup_available_memory_from
// ---------------------------------------------------------------------------

TEST(CgroupHeadroomFrom, NoPathsMeansUnlimited)
{
    FakeFs files;
    EXPECT_EQ(headroom({}, files), std::numeric_limits<size_t>::max());
}

TEST(CgroupHeadroomFrom, LeafLimitMinusUsage)
{
    FakeFs files;
    files["/sys/fs/cgroup/user.slice/app/memory.max"]     = "1073741824";
    files["/sys/fs/cgroup/user.slice/app/memory.current"] = "536870912";

    cgroup_paths paths;
    paths.v2 = "/user.slice/app";
    EXPECT_EQ(headroom(paths, files), kGiB / 2);
}

TEST(CgroupHeadroomFrom, MaxAtLeafDefersToParent)
{
    FakeFs files;
    files["/sys/fs/cgroup/user.slice/app/memory.max"]     = "max";
    files["/sys/fs/cgroup/user.slice/app/memory.current"] = "0";
    files["/sys/fs/cgroup/user.slice/memory.max"]         = "2147483648";
    files["/sys/fs/cgroup/user.slice/memory.current"]     = "536870912";

    cgroup_paths paths;
    paths.v2 = "/user.slice/app";
    EXPECT_EQ(headroom(paths, files), kGiB + kGiB / 2);
}

TEST(CgroupHeadroomFrom, NestedAncestorTightestWins)
{
    FakeFs files;
    // Parent 8 GiB cap / 6 GiB used -> 2 GiB left; child 4 GiB / 1 GiB -> 3 GiB left (parent wins).
    files["/sys/fs/cgroup/user.slice/memory.max"]         = "8589934592";
    files["/sys/fs/cgroup/user.slice/memory.current"]     = "6442450944";
    files["/sys/fs/cgroup/user.slice/app/memory.max"]     = "4294967296";
    files["/sys/fs/cgroup/user.slice/app/memory.current"] = "1073741824";

    cgroup_paths paths;
    paths.v2 = "/user.slice/app";
    EXPECT_EQ(headroom(paths, files), 2u * kGiB);
}

TEST(CgroupHeadroomFrom, MissingUsageDefaultsToZero)
{
    FakeFs files;
    files["/sys/fs/cgroup/user.slice/app/memory.max"] = "1073741824";

    cgroup_paths paths;
    paths.v2 = "/user.slice/app";
    EXPECT_EQ(headroom(paths, files), kGiB);
}

TEST(CgroupHeadroomFrom, MissingLimitSkipsLevel)
{
    FakeFs files;
    files["/sys/fs/cgroup/user.slice/app/memory.current"] = "1000";

    cgroup_paths paths;
    paths.v2 = "/user.slice/app";
    EXPECT_EQ(headroom(paths, files), std::numeric_limits<size_t>::max());
}

TEST(CgroupHeadroomFrom, FullyUsedCgroupHasZeroHeadroom)
{
    FakeFs files;
    files["/sys/fs/cgroup/user.slice/app/memory.max"]     = "1073741824";
    files["/sys/fs/cgroup/user.slice/app/memory.current"] = "1073741824";

    cgroup_paths paths;
    paths.v2 = "/user.slice/app";
    EXPECT_EQ(headroom(paths, files), 0u);
}

TEST(CgroupHeadroomFrom, V1HierarchyUsesLegacyFilenames)
{
    FakeFs files;
    files["/sys/fs/cgroup/memory/docker/abc/memory.limit_in_bytes"] = "2147483648";
    files["/sys/fs/cgroup/memory/docker/abc/memory.usage_in_bytes"] = "536870912";

    cgroup_paths paths;
    paths.v1 = "/docker/abc";
    EXPECT_EQ(headroom(paths, files), kGiB + kGiB / 2);
}

TEST(CgroupHeadroomFrom, CoMountedV1Mountpoint)
{
    char const mountinfo[] =
        "200 199 0:55 / /sys/fs/cgroup/cpu,memory rw shared:1 - cgroup cpu,memory rw,memory,cpu\n";
    FakeFs files;
    files["/sys/fs/cgroup/cpu,memory/docker/abc/memory.limit_in_bytes"] = "1073741824";
    files["/sys/fs/cgroup/cpu,memory/docker/abc/memory.usage_in_bytes"] = "536870912";

    cgroup_paths paths;
    paths.v1 = "/docker/abc";
    EXPECT_EQ(headroom(paths, files, mountinfo), kGiB / 2);
}

TEST(CgroupHeadroomFrom, NonRootV2MountRoot)
{
    char const mountinfo[] =
        "300 299 0:56 /user.slice /sys/fs/cgroup/user.slice rw shared:1 - cgroup2 cgroup2 rw\n";
    FakeFs files;
    files["/sys/fs/cgroup/user.slice/app/memory.max"]     = "1073741824";
    files["/sys/fs/cgroup/user.slice/app/memory.current"] = "536870912";

    cgroup_paths paths;
    paths.v2 = "/user.slice/app";
    EXPECT_EQ(headroom(paths, files, mountinfo), kGiB / 2);
}

TEST(CgroupHeadroomFrom, MinAcrossV1AndV2Walks)
{
    FakeFs files;
    files["/sys/fs/cgroup/user.slice/app/memory.max"]               = "4294967296";
    files["/sys/fs/cgroup/user.slice/app/memory.current"]           = "0";
    files["/sys/fs/cgroup/memory/docker/abc/memory.limit_in_bytes"] = "1073741824";
    files["/sys/fs/cgroup/memory/docker/abc/memory.usage_in_bytes"] = "536870912";

    cgroup_paths paths;
    paths.v2 = "/user.slice/app";
    paths.v1 = "/docker/abc";
    EXPECT_EQ(headroom(paths, files), kGiB / 2);
}
