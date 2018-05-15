// MIT License
//
// Copyright (c) 2017 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <algorithm>
#include <iostream>
#include <random>
#include <vector>

// Google Test
#include <gtest/gtest.h>
// HC API
#include <hcc/hc.hpp>
// rocPRIM API
#include <rocprim/rocprim.hpp>

#include "test_utils.hpp"

namespace rp = rocprim;

template<
    class Key,
    class Value,
    unsigned int BlockSize
>
struct params
{
    using key_type = Key;
    using value_type = Value;
    static constexpr unsigned int block_size = BlockSize;
};

template<typename Params>
class RocprimBlockSortTests : public ::testing::Test {
public:
    using key_type = typename Params::key_type;
    using value_type = typename Params::value_type;
    static constexpr unsigned int block_size = Params::block_size;
};

typedef ::testing::Types<
    // Power of 2 BlockSize
    params<unsigned int, int, 64U>,
    params<int, int, 128U>,
    params<unsigned int, int, 256U>,
    params<unsigned short, char, 1024U>,

    // Non-power of 2 BlockSize
    params<double, unsigned int, 65U>,
    params<float, int, 37U>,
    params<long long, char, 510U>,
    params<unsigned int, long long, 162U>,
    params<unsigned char, float, 255U>
> BlockSizes;

TYPED_TEST_CASE(RocprimBlockSortTests, BlockSizes);

TYPED_TEST(RocprimBlockSortTests, SortKey)
{
    using key_type = typename TestFixture::key_type;
    const size_t block_size = TestFixture::block_size;
    const size_t size = block_size * 1134;

    // Generate data
    std::vector<key_type> output = test_utils::get_random_data<key_type>(size, -100, 100);

    // Calculate expected results on host
    std::vector<key_type> expected(output);

    for(size_t i = 0; i < output.size() / block_size; i++)
    {
        std::sort(
            expected.begin() + (i * block_size),
            expected.begin() + ((i + 1) * block_size)
        );
    }

    hc::array_view<key_type, 1> d_output(output.size(), output.data());
    hc::parallel_for_each(
        hc::extent<1>(output.size()).tile(block_size),
        [=](hc::tiled_index<1> i) [[hc]]
        {
            key_type value = d_output[i];
            rp::block_sort<key_type, block_size> bsort;
            bsort.sort(value);
            d_output[i] = value;
        }
    );

    d_output.synchronize();
    for(size_t i = 0; i < output.size(); i++)
    {
        ASSERT_EQ(output[i], expected[i]);
    }
}

TYPED_TEST(RocprimBlockSortTests, SortKeyValue)
{
    using key_type = typename TestFixture::key_type;
    using value_type = typename TestFixture::value_type;
    const size_t block_size = TestFixture::block_size;
    const size_t size = block_size * 1134;

    // Generate data
    std::vector<key_type> output_key(size);
    for(size_t i = 0; i < output_key.size() / block_size; i++)
    {
        std::iota(
            output_key.begin() + (i * block_size),
            output_key.begin() + ((i + 1) * block_size),
            0
        );

        std::shuffle(
            output_key.begin() + (i * block_size),
            output_key.begin() + ((i + 1) * block_size),
            std::mt19937{std::random_device{}()}
        );
    }
    std::vector<value_type> output_value = test_utils::get_random_data<value_type>(size, -100, 100);

    // Combine vectors to form pairs with key and value
    std::vector<std::pair<key_type, value_type>> target(size);
    for (unsigned i = 0; i < target.size(); i++)
        target[i] = std::make_pair(output_key[i], output_value[i]);

    // Calculate expected results on host
    std::vector<std::pair<key_type, value_type>> expected(target);

    for(size_t i = 0; i < expected.size() / block_size; i++)
    {
        std::sort(
            expected.begin() + (i * block_size),
            expected.begin() + ((i + 1) * block_size)
        );
    }

    hc::array_view<key_type, 1> d_output_key(output_key.size(), output_key.data());
    hc::array_view<value_type, 1> d_output_value(output_value.size(), output_value.data());
    hc::parallel_for_each(
        hc::extent<1>(output_key.size()).tile(block_size),
        [=](hc::tiled_index<1> i) [[hc]]
        {
            key_type key = d_output_key[i];
            value_type value = d_output_value[i];
            rp::block_sort<key_type, block_size, value_type> bsort;
            bsort.sort(key, value);
            d_output_key[i] = key;
            d_output_value[i] = value;
        }
    );

    d_output_key.synchronize();
    d_output_value.synchronize();
    for(size_t i = 0; i < expected.size(); i++)
    {
        ASSERT_EQ(output_key[i], expected[i].first);
        ASSERT_EQ(output_value[i], expected[i].second);
    }
}

template<class Key, class Value>
struct key_value_comparator
{
    bool operator()(const std::pair<Key, Value>& lhs, const std::pair<Key, Value>& rhs)
    {
        return lhs.first > rhs.first;
    }
};

TYPED_TEST(RocprimBlockSortTests, CustomSortKeyValue)
{
    using key_type = typename TestFixture::key_type;
    using value_type = typename TestFixture::value_type;
    const size_t block_size = TestFixture::block_size;
    const size_t size = block_size * 1134;

    // Generate data
    std::vector<key_type> output_key(size);
    for(size_t i = 0; i < output_key.size() / block_size; i++)
    {
        std::iota(
            output_key.begin() + (i * block_size),
            output_key.begin() + ((i + 1) * block_size),
            0
        );

        std::shuffle(
            output_key.begin() + (i * block_size),
            output_key.begin() + ((i + 1) * block_size),
            std::mt19937{std::random_device{}()}
        );
    }
    std::vector<value_type> output_value = test_utils::get_random_data<value_type>(size, -100, 100);

    // Combine vectors to form pairs with key and value
    std::vector<std::pair<key_type, value_type>> target(size);
    for (unsigned i = 0; i < target.size(); i++)
        target[i] = std::make_pair(output_key[i], output_value[i]);

    // Calculate expected results on host
    std::vector<std::pair<key_type, value_type>> expected(target);

    for(size_t i = 0; i < expected.size() / block_size; i++)
    {
        std::sort(
            expected.begin() + (i * block_size),
            expected.begin() + ((i + 1) * block_size),
            key_value_comparator<key_type, value_type>()
        );
    }

    hc::array_view<key_type, 1> d_output_key(output_key.size(), output_key.data());
    hc::array_view<value_type, 1> d_output_value(output_value.size(), output_value.data());
    hc::parallel_for_each(
        hc::extent<1>(output_key.size()).tile(block_size),
        [=](hc::tiled_index<1> i) [[hc]]
        {
            key_type key = d_output_key[i];
            value_type value = d_output_value[i];
            rp::block_sort<key_type, block_size, value_type> bsort;
            bsort.sort(key, value, rocprim::greater<key_type>());
            d_output_key[i] = key;
            d_output_value[i] = value;
        }
    );

    d_output_key.synchronize();
    d_output_value.synchronize();
    for(size_t i = 0; i < expected.size(); i++)
    {
        ASSERT_EQ(output_key[i], expected[i].first);
        ASSERT_EQ(output_value[i], expected[i].second);
    }
}

