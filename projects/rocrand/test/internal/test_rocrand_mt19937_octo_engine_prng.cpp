// Copyright (c) 2022-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "../test_common.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>

#include <rng/mt19937_octo_engine.hpp>

#define MAXI(x) std::numeric_limits<x>::max()
#define MINI(x) std::numeric_limits<x>::min()

// Normalize betweem 0 and 1
#define NORMALIZE(x, type) \
    static_cast<double>(x - MINI(type)) / static_cast<double>(MAXI(type) - MINI(type))

/* #################################################

                TEST HOST SIDE

   ###############################################*/

TEST(Mt19937OctoEngineTest, test_host_gather)
{
    // Check if a mt199370 Octo Engine state contatins the correct arrangement listed below

    /// Thread 0 has element   0, thread 1 has element 113, thread 2 has element 170,
    /// thread 3 had element 283, thread 4 has element 340, thread 5 has element 397,
    /// thread 6 has element 510, thread 7 has element 567.
    /// Thread i for i in [0, 7) has the following elements (ipt = items_per_thread):
    /// [  1 + ipt * i,   1 + ipt * (i + 1)), [398 + ipt * i, 398 + ipt * (i + 1)), [171 + ipt * i, 171 + ipt * (i + 1)),
    /// [568 + ipt * i, 568 + ipt * (i + 1)), [341 + ipt * i, 341 + ipt * (i + 1)), [114 + ipt * i, 114 + ipt * (i + 1)),
    /// [511 + ipt * i, 511 + ipt * (i + 1)), [284 + ipt * i, 284 + ipt * (i + 1)), [ 57 + ipt * i,  57 + ipt * (i + 1)),
    /// [454 + ipt * i, 454 + ipt * (i + 1)), [227 + ipt * i, 227 + ipt * (i + 1))
    ///

    namespace constants = rocrand_impl::host::mt19937_constants;

    std::vector<unsigned int> src(constants::n);
    std::iota(src.begin(), src.end(), 0);

    const std::vector<unsigned int> offsets = {1, 398, 171, 568, 341, 114, 511, 284, 57, 454, 227};
    const std::vector<unsigned int> special_elem = {0, 113, 170, 283, 340, 397, 510, 567};
    const unsigned int              ipt          = 7;
    const unsigned int              vpt          = 1 + ipt * 11;

    for(size_t tid = 0; tid < 8; tid++)
    {
        rocrand_impl::host::mt19937_octo_engine test_engine;
        test_engine.gather(src.data(), dim3(tid, 0, 0));

        std::vector<unsigned int> expected_items;

        for(const unsigned int& offset : offsets)
        {
            auto left  = offset + ipt * tid;
            auto right = (offset + ipt * (tid + 1)); // no need to -1 since insert is exclusive

            expected_items.insert(expected_items.begin(), src.begin() + left, src.begin() + right);
        }

        expected_items.insert(expected_items.begin(), special_elem[tid]);

        std::sort(expected_items.begin(), expected_items.end());

        std::vector<unsigned int> actual_items(vpt);

        for(size_t i = 0; i < vpt; i++)
            actual_items[i] = test_engine.get(i);

        std::sort(actual_items.begin(), actual_items.end());

        for(size_t i = 0; i < vpt; i++)
            ASSERT_EQ(expected_items[i], actual_items[i]);
    }
}

unsigned int comp(unsigned int a, unsigned int b, unsigned int c)
{
    namespace constants = rocrand_impl::host::mt19937_constants;

    unsigned int x  = (a & constants::upper_mask) | (b & constants::lower_mask);
    unsigned int xA = x >> 1;
    if(x & 1UL)
        xA ^= constants::matrix_a;

    x = c ^ xA;
    return x;
}

TEST(Mt19937OctoEngineTest, comp_test)
{
    //Test the twister step
    namespace constants = rocrand_impl::host::mt19937_constants;

    std::vector<unsigned int> src(constants::n);
    std::iota(src.begin(), src.end(), 0);

    // the actual thread should not matter
    rocrand_impl::host::mt19937_octo_engine test_engine;
    test_engine.gather(src.data(), dim3(0, 0, 0));

    for(size_t i = 0; i < constants::n; i++)
    {
        long long k = i;
        long long j = k - (constants::n - 1);
        if(j < 0)
            j += constants::n;

        long long m = k - (constants::n - constants::m);
        if(m < 0)
            m += constants::n;

        ASSERT_EQ(comp(k, j, m), test_engine.comp(k, j, m))
            << k << " " << j << " " << m << std::endl;
    }
}

TEST(Mt19937OctoEngineTest, gen_next_n_consistency_test)
{
    namespace constants = rocrand_impl::host::mt19937_constants;

    std::vector<unsigned int> src(constants::n);
    std::iota(src.begin(), src.end(), 0);

    rocrand_impl::host::mt19937_octo_engine octo_engine_a[8];
    rocrand_impl::host::mt19937_octo_engine octo_engine_b[8];

    for(size_t i = 0; i < 8; i++)
    {
        octo_engine_a[i].gather(src.data(), dim3(i));
        octo_engine_b[i].gather(src.data(), dim3(i));
    }

    constexpr size_t test_size = 1000;

    for(size_t _ = 0; _ < test_size; _++)
    {
        rocrand_impl::host::mt19937_octo_engine::gen_next_n(octo_engine_a);
        rocrand_impl::host::mt19937_octo_engine::gen_next_n(octo_engine_b);

        for(size_t tid = 0; tid < 8; tid++)
        {
            for(size_t i = 0; i < 78; i++)
            {
                ASSERT_EQ(octo_engine_a[tid].get(i), octo_engine_b[tid].get(i));
            }
        }
    }
}

TEST(Mt19937OctoEngineTest, uniform_dis_test)
{
    // checks if we are uniformly distributed
    namespace constants = rocrand_impl::host::mt19937_constants;

    std::vector<unsigned int> src(constants::n);
    std::iota(src.begin(), src.end(), 0);

    rocrand_impl::host::mt19937_octo_engine octo_engine[8];

    for(size_t i = 0; i < 8; i++)
    {
        octo_engine[i].gather(src.data(), dim3(i));
    }

    constexpr size_t test_size = 1000;

    std::vector<double> output;

    for(size_t _ = 0; _ < test_size; _++)
    {
        rocrand_impl::host::mt19937_octo_engine::gen_next_n(octo_engine);

        for(size_t tid = 0; tid < 8; tid++)
        {
            for(size_t i = 0; i < 78; i++)
            {
                unsigned int x = octo_engine[tid].get(i);

                double normalized = NORMALIZE(x, unsigned int);

                output.push_back(normalized);
            }
        }
    }

    double actual_mean
        = std::accumulate(output.begin(), output.end(), (double)0) / static_cast<double>(output.size());
    double actual_std_dev
        = std::accumulate(output.begin(),
                          output.end(),
                          (double)0,
                          [=](double acc, double x) { return acc + POWF(x - actual_mean, 2); });
    actual_std_dev        = std::sqrt(actual_std_dev / static_cast<double>(output.size() - 1));

    double expected_mean    = 0.5;
    double expected_std_dev = 0.288675134595; //sqrt(1/12)

    ASSERT_NEAR(expected_mean, actual_mean, expected_mean * 0.01);
    ASSERT_NEAR(expected_std_dev, actual_std_dev, expected_std_dev * 0.01);
}

TEST(Mt19937OctoEngineAccessorTest, load_test)
{
    namespace constants = rocrand_impl::host::mt19937_constants;

    std::vector<unsigned int> src(constants::n);
    std::iota(src.begin(), src.end(), 0);

    const unsigned int ipt = 7;
    const unsigned int vpt = 1 + ipt * 11;

    rocrand_impl::host::mt19937_octo_engine_accessor<8> accessor(src.data());

    for(size_t tid = 0; tid < 8; tid++)
    {
        auto accessor_engine = accessor.load(tid);

        for(size_t i = 0; i < vpt; i++)
        {
            ASSERT_EQ(accessor_engine.get(i), src[i * 8 + tid]);
        }
    }
}

TEST(Mt19937OctoEngineAccessorTest, load_value_test)
{
    namespace constants = rocrand_impl::host::mt19937_constants;

    std::vector<unsigned int> src(constants::n);
    std::iota(src.begin(), src.end(), 0);

    const unsigned int ipt = 7;
    const unsigned int vpt = 1 + ipt * 11;

    rocrand_impl::host::mt19937_octo_engine_accessor<8> accessor(src.data());

    for(size_t tid = 0; tid < 8; tid++)
    {
        auto engine = accessor.load(tid);

        for(size_t i = 0; i < vpt; i++)
        {
            ASSERT_EQ(engine.get(i), accessor.load_value(tid, i));
        }
    }
}

TEST(Mt19937OctoEngineAccessorTest, save_test)
{
    namespace constants = rocrand_impl::host::mt19937_constants;

    std::vector<unsigned int> src(constants::n);
    std::iota(src.begin(), src.end(), 0);

    const unsigned int ipt = 7;
    const unsigned int vpt = 1 + ipt * 11;

    std::vector<rocrand_impl::host::mt19937_octo_engine> octo_engine(8);
    rocrand_impl::host::mt19937_octo_engine_accessor<8>  accessor(src.data());

    for(size_t tid = 0; tid < 8; tid++)
    {

        octo_engine[tid].gather(src.data(), dim3(tid, 0, 0));

        accessor.save(tid, octo_engine[tid]);

        for(size_t i = 0; i < vpt; i++)
        {
            ASSERT_EQ(octo_engine[tid].get(i), accessor.load_value(tid, i));
        }
    }
}
