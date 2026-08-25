// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "node_factory.h"
#include "plan.h"
#include "repo.h"

#include "../../../shared/device_properties.h"
#include "rocfft/rocfft.h"

#include <gtest/gtest.h>
#include <vector>

// Rebuilding a cached twiddle table is wasted work and memory that no accuracy
// test would notice.
class repo_test : public ::testing::Test
{
protected:
    void SetUp() override
    {
        ASSERT_EQ(rocfft_setup(), rocfft_status_success);
        deviceProp = get_curr_device_prop();
    }

    void TearDown() override
    {
        Repo::Clear();
        rocfft_cleanup();
    }

    hipDeviceProp_t deviceProp;
};

TEST_F(repo_test, twiddles_1d_are_reused)
{
    const std::vector<size_t> radices{8, 8};

    auto first = Repo::GetTwiddles1D(64, 0, rocfft_precision_single, deviceProp, 0, false, radices);
    auto second
        = Repo::GetTwiddles1D(64, 0, rocfft_precision_single, deviceProp, 0, false, radices);

    ASSERT_NE(first.first, nullptr);
    EXPECT_EQ(first.first, second.first);
    EXPECT_EQ(first.second, second.second);
}

TEST_F(repo_test, twiddles_1d_differ_by_length)
{
    auto small = Repo::GetTwiddles1D(64, 0, rocfft_precision_single, deviceProp, 0, false, {8, 8});
    auto large
        = Repo::GetTwiddles1D(128, 0, rocfft_precision_single, deviceProp, 0, false, {8, 16});

    ASSERT_NE(small.first, nullptr);
    ASSERT_NE(large.first, nullptr);
    EXPECT_NE(small.first, large.first);
}

TEST_F(repo_test, twiddles_1d_differ_by_precision)
{
    const std::vector<size_t> radices{8, 8};

    auto single
        = Repo::GetTwiddles1D(64, 0, rocfft_precision_single, deviceProp, 0, false, radices);
    auto twice = Repo::GetTwiddles1D(64, 0, rocfft_precision_double, deviceProp, 0, false, radices);

    ASSERT_NE(single.first, nullptr);
    ASSERT_NE(twice.first, nullptr);
    EXPECT_NE(single.first, twice.first);
}

TEST_F(repo_test, twiddles_2d_are_reused)
{
    auto first = Repo::GetTwiddles2D(
        64, 64, rocfft_precision_single, deviceProp, false, false, {8, 8}, {8, 8});
    auto second = Repo::GetTwiddles2D(
        64, 64, rocfft_precision_single, deviceProp, false, false, {8, 8}, {8, 8});

    ASSERT_NE(first.first, nullptr);
    EXPECT_EQ(first.first, second.first);
}

TEST_F(repo_test, chirp_is_reused)
{
    auto first  = Repo::GetChirp(1009, rocfft_precision_single, deviceProp);
    auto second = Repo::GetChirp(1009, rocfft_precision_single, deviceProp);

    ASSERT_NE(first.first, nullptr);
    EXPECT_EQ(first.first, second.first);
    EXPECT_EQ(first.second, second.second);
}
