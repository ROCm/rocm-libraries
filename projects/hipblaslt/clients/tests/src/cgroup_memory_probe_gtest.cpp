// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Host-only unit tests for cgroup_memory_probe.hpp: no GPU/HIP calls, no real cgroup setup.

#include "cgroup_memory_probe.hpp"

#include <gtest/gtest.h>

#include <map>
#include <string>

using hipblaslt_client::cgroup_available_memory_from;
using hipblaslt_client::cgroup_paths;
using hipblaslt_client::parse_cgroup_size_token;
using hipblaslt_client::parse_proc_self_cgroup;

namespace
{
    size_t const kUnlimitedSentinel = static_cast<size_t>(1) << 62;
    size_t const kGiB               = 1024u * 1024u * 1024u; // 1 GiB; fake sysfs strings below are byte counts in the same units

    struct FakeFs
    {
        std::map<std::string, std::string> files;

        bool read_size(std::string const& path, size_t& out) const
        {
            auto it = files.find(path);
            if(it == files.end())
                return false;
            return parse_cgroup_size_token(it->second, out, kUnlimitedSentinel);
        }
    };

    size_t headroom(cgroup_paths const& paths, FakeFs const& fs)
    {
        return cgroup_available_memory_from(
            paths, [&](std::string const& path, size_t& out) { return fs.read_size(path, out); });
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
// cgroup_available_memory_from
// ---------------------------------------------------------------------------

TEST(CgroupHeadroomFrom, NoPathsMeansUnlimited)
{
    FakeFs fs;
    EXPECT_EQ(headroom({}, fs), std::numeric_limits<size_t>::max());
}

TEST(CgroupHeadroomFrom, LeafLimitMinusUsage)
{
    FakeFs fs;
    fs.files["/sys/fs/cgroup/user.slice/app/memory.max"]     = "1073741824";
    fs.files["/sys/fs/cgroup/user.slice/app/memory.current"] = "536870912";

    cgroup_paths paths;
    paths.v2 = "/user.slice/app";
    EXPECT_EQ(headroom(paths, fs), kGiB / 2);
}

TEST(CgroupHeadroomFrom, MaxAtLeafDefersToParent)
{
    FakeFs fs;
    fs.files["/sys/fs/cgroup/user.slice/app/memory.max"]     = "max";
    fs.files["/sys/fs/cgroup/user.slice/app/memory.current"] = "0";
    fs.files["/sys/fs/cgroup/user.slice/memory.max"]         = "2147483648";
    fs.files["/sys/fs/cgroup/user.slice/memory.current"]     = "536870912";

    cgroup_paths paths;
    paths.v2 = "/user.slice/app";
    EXPECT_EQ(headroom(paths, fs), kGiB + kGiB / 2);
}

TEST(CgroupHeadroomFrom, NestedAncestorTightestWins)
{
    FakeFs fs;
    // Parent 8 GiB cap / 6 GiB used -> 2 GiB left; child 4 GiB / 1 GiB -> 3 GiB left (parent wins).
    fs.files["/sys/fs/cgroup/user.slice/memory.max"]         = "8589934592";
    fs.files["/sys/fs/cgroup/user.slice/memory.current"]     = "6442450944";
    fs.files["/sys/fs/cgroup/user.slice/app/memory.max"]     = "4294967296";
    fs.files["/sys/fs/cgroup/user.slice/app/memory.current"] = "1073741824";

    cgroup_paths paths;
    paths.v2 = "/user.slice/app";
    EXPECT_EQ(headroom(paths, fs), 2u * kGiB);
}

TEST(CgroupHeadroomFrom, MissingUsageDefaultsToZero)
{
    FakeFs fs;
    fs.files["/sys/fs/cgroup/user.slice/app/memory.max"] = "1073741824";

    cgroup_paths paths;
    paths.v2 = "/user.slice/app";
    EXPECT_EQ(headroom(paths, fs), kGiB);
}

TEST(CgroupHeadroomFrom, MissingLimitSkipsLevel)
{
    FakeFs fs;
    fs.files["/sys/fs/cgroup/user.slice/app/memory.current"] = "1000";

    cgroup_paths paths;
    paths.v2 = "/user.slice/app";
    EXPECT_EQ(headroom(paths, fs), std::numeric_limits<size_t>::max());
}

TEST(CgroupHeadroomFrom, FullyUsedCgroupHasZeroHeadroom)
{
    FakeFs fs;
    fs.files["/sys/fs/cgroup/user.slice/app/memory.max"]     = "1073741824";
    fs.files["/sys/fs/cgroup/user.slice/app/memory.current"] = "1073741824";

    cgroup_paths paths;
    paths.v2 = "/user.slice/app";
    EXPECT_EQ(headroom(paths, fs), 0u);
}

TEST(CgroupHeadroomFrom, V1HierarchyUsesLegacyFilenames)
{
    FakeFs fs;
    fs.files["/sys/fs/cgroup/memory/docker/abc/memory.limit_in_bytes"]  = "2147483648";
    fs.files["/sys/fs/cgroup/memory/docker/abc/memory.usage_in_bytes"] = "536870912";

    cgroup_paths paths;
    paths.v1 = "/docker/abc";
    EXPECT_EQ(headroom(paths, fs), kGiB + kGiB / 2);
}

TEST(CgroupHeadroomFrom, MinAcrossV1AndV2Walks)
{
    FakeFs fs;
    fs.files["/sys/fs/cgroup/user.slice/app/memory.max"]                    = "4294967296";
    fs.files["/sys/fs/cgroup/user.slice/app/memory.current"]                = "0";
    fs.files["/sys/fs/cgroup/memory/docker/abc/memory.limit_in_bytes"]      = "1073741824";
    fs.files["/sys/fs/cgroup/memory/docker/abc/memory.usage_in_bytes"]     = "536870912";

    cgroup_paths paths;
    paths.v2 = "/user.slice/app";
    paths.v1 = "/docker/abc";
    EXPECT_EQ(headroom(paths, fs), kGiB / 2);
}
