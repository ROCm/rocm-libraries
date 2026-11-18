/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#include "hipsparse_local_handle.hpp"

#include <cstdio>
#include <cstring>
#include <gtest/gtest.h>
#include <string>

// The pre_checkin prefix is the name CI selects. TheRock matches pre_checkin/*
// and Jenkins matches *checkin*. A plain TEST() has no such prefix, so it
// never runs in either lane.
class get_version : public testing::TestWithParam<int>
{
};

TEST_P(get_version, returns_library_version)
{
    hipsparseLocalHandle_t handle;

    int version = 0;
    ASSERT_EQ(hipsparseGetVersion(handle, &version), HIPSPARSE_STATUS_SUCCESS);

    // Packed the same way both backends write it. Unpacked the way the public
    // header documents: major is version / 100000, minor is (version / 100)
    // mod 1000, and patch is version mod 100.
    constexpr int expected
        = hipsparseVersionMajor * 100000 + hipsparseVersionMinor * 100 + hipsparseVersionPatch;

    EXPECT_EQ(version, expected);
    EXPECT_EQ(version / 100000, hipsparseVersionMajor);
    EXPECT_EQ(version / 100 % 1000, hipsparseVersionMinor);
    EXPECT_EQ(version % 100, hipsparseVersionPatch);
}

TEST_P(get_version, rejects_null_handle)
{
    int version = -1;

    EXPECT_EQ(hipsparseGetVersion(nullptr, &version), HIPSPARSE_STATUS_NOT_INITIALIZED);
    EXPECT_EQ(version, -1);
}

TEST_P(get_version, rejects_null_version)
{
    hipsparseLocalHandle_t handle;

    EXPECT_EQ(hipsparseGetVersion(handle, nullptr), HIPSPARSE_STATUS_INVALID_VALUE);
}

INSTANTIATE_TEST_SUITE_P(pre_checkin, get_version, testing::Values(0));

namespace
{
    // Same two-level stringification the backends use on hipsparseVersionTweak.
    // The implementation copies at most 64 characters of that text.
    std::string revision_prefix()
    {
#define HIPSPARSE_STRINGIFY_IMPL(text) #text
#define HIPSPARSE_STRINGIFY(text) HIPSPARSE_STRINGIFY_IMPL(text)
        const std::string tweak = HIPSPARSE_STRINGIFY(hipsparseVersionTweak);
#undef HIPSPARSE_STRINGIFY
#undef HIPSPARSE_STRINGIFY_IMPL
        return tweak.substr(0, 64);
    }
}

class get_git_revision : public testing::TestWithParam<int>
{
};

TEST_P(get_git_revision, returns_revision)
{
    hipsparseLocalHandle_t handle;

    // Both backends snprintf into a 256-byte buffer.
    char revision[256];
    std::memset(revision, 0, sizeof(revision));
    ASSERT_EQ(hipsparseGetGitRevision(handle, revision), HIPSPARSE_STATUS_SUCCESS);
    EXPECT_LT(std::strlen(revision), sizeof(revision));

    const std::string text(revision);
    const std::string prefix = revision_prefix();
#if defined(CUDART_VERSION)
    const std::string backend = " (cuSPARSE ";
#else
    const std::string backend = " (rocSPARSE ";
#endif
    const std::string head = prefix + backend;

    ASSERT_GE(text.size(), head.size());
    EXPECT_EQ(text.compare(0, head.size(), head), 0);
    EXPECT_EQ(text.back(), ')');

    int         major   = -1;
    int         minor   = -1;
    int         patch   = -1;
    const char* numbers = text.c_str() + head.size();
#if defined(CUDART_VERSION)
    ASSERT_EQ(std::sscanf(numbers, "%d.%d.%d)", &major, &minor, &patch), 3);
#else
    ASSERT_EQ(std::sscanf(numbers, "%d.%d.%d-", &major, &minor, &patch), 3);
    // A build without Git metadata leaves ROCSPARSE_VERSION_TWEAK empty, so
    // the text after the hyphen may be empty, as in "(rocSPARSE 5.2.0-)".
    const char* backend_revision = std::strchr(numbers, '-');
    ASSERT_NE(backend_revision, nullptr);
#endif
    EXPECT_GE(major, 0);
    EXPECT_GE(minor, 0);
    EXPECT_GE(patch, 0);
    EXPECT_LT(minor, 1000);
    EXPECT_LT(patch, 100);
}

TEST_P(get_git_revision, rejects_null_handle)
{
    char revision[256];
    std::memset(revision, 'x', sizeof(revision));

    EXPECT_EQ(hipsparseGetGitRevision(nullptr, revision), HIPSPARSE_STATUS_NOT_INITIALIZED);
    EXPECT_EQ(revision[0], 'x');
}

TEST_P(get_git_revision, rejects_null_revision)
{
    hipsparseLocalHandle_t handle;

    EXPECT_EQ(hipsparseGetGitRevision(handle, nullptr), HIPSPARSE_STATUS_INVALID_VALUE);
}

INSTANTIATE_TEST_SUITE_P(pre_checkin, get_git_revision, testing::Values(0));
