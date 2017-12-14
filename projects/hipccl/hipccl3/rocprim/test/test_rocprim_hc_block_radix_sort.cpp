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
#include <functional>
#include <iostream>
#include <type_traits>
#include <vector>

// Google Test
#include <gtest/gtest.h>
// HC API
#include <hcc/hc.hpp>
// rocPRIM
#include <block/block_radix_sort.hpp>

#include "test_utils.hpp"

namespace rp = rocprim;

template<
    class T,
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    unsigned int StartBit = 0,
    unsigned int EndBit = sizeof(T) * 8
>
struct params
{
    using type = T;
    static constexpr unsigned int block_size = BlockSize;
    static constexpr unsigned int items_per_thread = ItemsPerThread;
    static constexpr unsigned int start_bit = StartBit;
    static constexpr unsigned int end_bit = EndBit;
};

template<class Params>
class RocprimBlockRadixSort : public ::testing::Test {
public:
    using params = Params;
};

typedef ::testing::Types<
    // Power of 2 Blocksize
    params<unsigned int, 64U, 1>,
    params<int, 128U, 1>,
    params<unsigned int, 256U, 1>,
    params<unsigned short, 1024U, 1>,

    // Non-power of 2 Blocksize
    params<double, 65U, 1>,
    params<float, 37U, 1>,
    params<long long, 510U, 1>,
    params<unsigned int, 162U, 1>,
    params<unsigned char, 255U, 1>,

    // Power of 2 Blocksize and ItemsPerThread > 1
    params<float, 64U, 2>,
    params<int, 1024U, 4>,
    params<unsigned short, 256U, 7>,

    // Non-power of 2 Blocksize and ItemsPerThread > 1
    params<double, 33U, 5>,
    params<char, 464U, 2>,
    params<unsigned short, 100U, 3>,
    params<short, 234U, 9>,

    // StartBits and EndBits
    params<unsigned long long, 64U, 1, 8, 20>,
    params<unsigned int, 162U, 2, 3, 12>
> Params;

TYPED_TEST_CASE(RocprimBlockRadixSort, Params);

template<class T, unsigned int StartBit, unsigned int EndBit>
struct comparator
{
    static_assert(std::is_unsigned<T>::value, "Test supports start and bits only for unsigned integers");

    bool operator()(const T& lhs, const T& rhs)
    {
        auto mask = (1ull << (EndBit - StartBit)) - 1;
        auto l = static_cast<unsigned long long>(lhs);
        auto r = static_cast<unsigned long long>(rhs);
        return ((l >> StartBit) & mask) < ((r >> StartBit) & mask);
    }
};

template<class T>
struct comparator<T, 0, sizeof(T) * 8>
{
    bool operator()(const T& lhs, const T& rhs)
    {
        return lhs < rhs;
    }
};

TYPED_TEST(RocprimBlockRadixSort, SortKeys)
{
    hc::accelerator acc;

    using T = typename TestFixture::params::type;
    constexpr size_t block_size = TestFixture::params::block_size;
    constexpr size_t items_per_thread = TestFixture::params::items_per_thread;
    constexpr unsigned int start_bit = TestFixture::params::start_bit;
    constexpr unsigned int end_bit = TestFixture::params::end_bit;
    constexpr size_t items_per_block = block_size * items_per_thread;
    // Given block size not supported
    if(block_size > get_max_tile_size(acc))
    {
        return;
    }

    const size_t size = items_per_block * 1134;
    // Generate data
    std::vector<T> output;
    if(std::is_floating_point<T>::value)
        output = get_random_data<T>(size, (T)-1000, (T)+1000);
    else
        output = get_random_data<T>(size, std::numeric_limits<T>::min(), std::numeric_limits<T>::max());

    // Calulcate expected results on host
    std::vector<T> expected(output);
    for(size_t i = 0; i < output.size() / items_per_block; i++)
    {
        std::stable_sort(
            expected.begin() + (i * items_per_block),
            expected.begin() + ((i + 1) * items_per_block),
            comparator<T, start_bit, end_bit>()
        );
    }

    hc::array_view<T, 1> d_output(output.size(), output.data());
    hc::parallel_for_each(
        acc.get_default_view(),
        hc::extent<1>(output.size() / items_per_thread).tile(block_size),
        [=](hc::tiled_index<1> idx) [[hc]]
        {
            const unsigned int thread_id = idx.global[0];

            T keys[items_per_thread];
            for(unsigned int i = 0; i < items_per_thread; i++)
            {
                keys[i] = d_output[thread_id * items_per_thread + i];
            }

            rp::block_radix_sort<T, block_size, items_per_thread> bsort;
            bsort.sort(keys, start_bit, end_bit);

            for(unsigned int i = 0; i < items_per_thread; i++)
            {
                d_output[thread_id * items_per_thread + i] = keys[i];
            }
        }
    );

    d_output.synchronize();
    for(size_t i = 0; i < output.size(); i++)
    {
        ASSERT_EQ(output[i], expected[i]);
    }
}
