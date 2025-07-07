// Copyright (c) 2017-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include <gtest/gtest.h>
#include <stdio.h>

#include <cmath>
#include <numeric>
#include <random>
#include <vector>

#include <rocrand/rocrand_mrg31k3p.h>

TEST(mrg31k3pTest, host_rocrand_init_consistency)
{
    // Test the consistency of rocrand_init when the same seed and subsequence are given
    constexpr size_t test_size = 10000;

    std::vector<unsigned long long> seeds        = {1, 12, 123, 1234, 12345};
    std::vector<unsigned long long> subsequences = {54321, 4321, 321, 21, 1};

    rocrand_state_mrg31k3p state1;
    rocrand_state_mrg31k3p state2;

    for(const unsigned long long& seed : seeds)
    {
        for(const unsigned long long& subsequence : subsequences)
        {
            rocrand_init(seed, subsequence, 0, &state1);
            rocrand_init(seed, subsequence, 0, &state2);

            for(size_t i = 0; i < test_size; i++)
                ASSERT_EQ(rocrand(&state1), rocrand(&state2));
        }
    }
}

TEST(mrg31k3pTest, host_rocrand_offet_consistency)
{
    // Test the consistency of rocrand_init when the same seed and subsequence are given
    // but with a different offset

    std::vector<unsigned long long> seeds        = {1, 12, 123, 1234, 12345};
    std::vector<unsigned long long> subsequences = {54321, 4321, 321, 21, 1};
    std::vector<unsigned long long> offsets      = {1, 10, 100, 1000, 10000};

    rocrand_state_mrg31k3p state1;
    rocrand_state_mrg31k3p state2;

    for(const unsigned long long& seed : seeds)
    {
        for(const unsigned long long& subsequence : subsequences)
        {
            for(const unsigned long long& offset : offsets)
            {
                rocrand_init(seed, subsequence, 0, &state1);
                rocrand_init(seed, subsequence, offset, &state2);

                for(size_t i = 0; i < offset; i++)
                    rocrand(&state1);

                ASSERT_EQ(rocrand(&state1), rocrand(&state2));
            }
        }
    }
}

TEST(mrg31k3pTest, rocrand_check_uniform_property)
{
    // Test to ensure that rocrand gives a uniform distribution

    constexpr size_t size = 10000;

    rocrand_state_mrg31k3p state;
    rocrand_init(123456, 654321, 0, &state);

    std::vector<unsigned int> output(size);

    for(size_t i = 0; i < size; i++)
        output[i] = rocrand(&state);

    constexpr double a = 0;
    constexpr double b = 1;

    const double expected_mean    = (a + b) / 2;
    const double expected_std_dev = (b - a) / std::sqrt(12);

    const unsigned int mini = std::numeric_limits<unsigned int>::min();
    const unsigned int maxi = std::numeric_limits<unsigned int>::max();

    double actual_mean = std::accumulate(output.begin(),
                                         output.end(),
                                         (double)0.0,
                                         [=](double acc, unsigned int x)
                                         {
                                             double converted
                                                 = (a + static_cast<double>(x - mini) * (b - a))
                                                   / (static_cast<double>(maxi - mini));
                                             return acc + converted;
                                         })
                         / static_cast<double>(size);

    double actual_std_dev = std::accumulate(output.begin(),
                                            output.end(),
                                            (double)0.0,
                                            [=](double acc, unsigned int x)
                                            {
                                                double converted
                                                    = (a + static_cast<double>(x - mini) * (b - a))
                                                      / (static_cast<double>(maxi - mini));
                                                return acc + std::pow(converted - actual_mean, 2);
                                            });
    actual_std_dev        = std::sqrt(actual_std_dev / static_cast<double>(size));

    // make sure results are within 5% of expected values
    ASSERT_NEAR(expected_mean, actual_mean, expected_mean * 0.05);
    ASSERT_NEAR(expected_std_dev, actual_std_dev, expected_std_dev * 0.05);
}

TEST(mrg31k3pTest, host_skipahead)
{
    // Test the consistency of skipahead when the same seed and subsequence are given

    std::vector<unsigned long long> seeds        = {1, 12, 123, 1234, 12345};
    std::vector<unsigned long long> subsequences = {54321, 4321, 321, 21, 1};
    std::vector<unsigned long long> offsets      = {1, 10, 100, 1000, 10000};

    rocrand_state_mrg31k3p state1;
    rocrand_state_mrg31k3p state2;

    for(const unsigned long long& seed : seeds)
    {
        for(const unsigned long long& subsequence : subsequences)
        {
            for(const unsigned long long& offset : offsets)
            {
                rocrand_init(seed, subsequence, 0, &state1);
                rocrand_init(seed, subsequence, offset, &state2);

                skipahead(offset, &state1);

                ASSERT_EQ(rocrand(&state1), rocrand(&state2));
            }
        }
    }
}

TEST(mrg31k3pTest, host_skipahead_subsequence)
{
    // Test the consistency of skipahead when the same seed and subsequence are given

    std::vector<unsigned long long> seeds        = {1, 12, 123, 1234, 12345};
    std::vector<unsigned long long> subsequences = {54321, 4321, 321, 21, 1};

    rocrand_state_mrg31k3p state1;
    rocrand_state_mrg31k3p state2;

    for(const unsigned long long& seed : seeds)
    {
        for(const unsigned long long& subsequence : subsequences)
        {
            rocrand_init(seed, 0, 0, &state1);
            rocrand_init(seed, subsequence, 0, &state2);

            skipahead_subsequence(subsequence, &state1);

            ASSERT_EQ(rocrand(&state1), rocrand(&state2));
        }
    }
}
