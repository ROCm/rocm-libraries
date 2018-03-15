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

#include <iostream>
#include <vector>

// Google Test
#include <gtest/gtest.h>
// HC API
#include <hcc/hc.hpp>
// rocPRIM API
#include <rocprim/rocprim.hpp>

#include "test_utils.hpp"

namespace rp = rocprim;

// Params for tests
template<
    class T,
    unsigned int BlockSize = 256U,
    unsigned int ItemsPerThread = 1U,
    rocprim::block_reduce_algorithm Algorithm = rocprim::block_reduce_algorithm::using_warp_reduce
>
struct params
{
    using type = T;
    static constexpr rocprim::block_reduce_algorithm algorithm = Algorithm;
    static constexpr unsigned int block_size = BlockSize;
    static constexpr unsigned int items_per_thread = ItemsPerThread;
};

// ---------------------------------------------------------
// Test for reduce ops taking single input value
// ---------------------------------------------------------

template<class Params>
class RocprimBlockReduceSingleValueTests : public ::testing::Test
{
public:
    using type = typename Params::type;
    static constexpr rocprim::block_reduce_algorithm algorithm = Params::algorithm;
    static constexpr unsigned int block_size = Params::block_size;
};

typedef ::testing::Types<
    // -----------------------------------------------------------------------
    // rocprim::block_reduce_algorithm::using_warp_reduce
    // -----------------------------------------------------------------------
    params<int, 64U>,
    params<int, 128U>,
    params<int, 192U>,
    params<int, 256U>,
    params<int, 512U>,
    params<int, 1024U>,
    params<int, 65U>,
    params<int, 37U>,
    params<int, 129U>,
    params<int, 162U>,
    params<int, 255U>,
    // uint tests
    params<unsigned int, 64U>,
    params<unsigned int, 256U>,
    params<unsigned int, 377U>,
    // long tests
    params<long, 64U>,
    params<long, 256U>,
    params<long, 377U>,
    // -----------------------------------------------------------------------
    // rocprim::block_reduce_algorithm::raking_reduce
    // -----------------------------------------------------------------------
    params<int, 64U, 1, rocprim::block_reduce_algorithm::raking_reduce>,
    params<int, 128U, 1, rocprim::block_reduce_algorithm::raking_reduce>,
    params<int, 192U, 1, rocprim::block_reduce_algorithm::raking_reduce>,
    params<int, 256U, 1, rocprim::block_reduce_algorithm::raking_reduce>,
    params<int, 512U, 1, rocprim::block_reduce_algorithm::raking_reduce>,
    params<int, 1024U, 1, rocprim::block_reduce_algorithm::raking_reduce>,
    params<unsigned long, 65U, 1, rocprim::block_reduce_algorithm::raking_reduce>,
    params<long, 37U, 1, rocprim::block_reduce_algorithm::raking_reduce>,
    params<short, 162U, 1, rocprim::block_reduce_algorithm::raking_reduce>,
    params<unsigned int, 255U, 1, rocprim::block_reduce_algorithm::raking_reduce>,
    params<int, 377U, 1, rocprim::block_reduce_algorithm::raking_reduce>,
    params<unsigned char, 377U, 1, rocprim::block_reduce_algorithm::raking_reduce>
> SingleValueTestParams;

TYPED_TEST_CASE(RocprimBlockReduceSingleValueTests, SingleValueTestParams);

TYPED_TEST(RocprimBlockReduceSingleValueTests, Reduce)
{
    using T = typename TestFixture::type;
    constexpr auto algorithm = TestFixture::algorithm;
    constexpr size_t block_size = TestFixture::block_size;

    hc::accelerator acc;
    // Given block size not supported
    if(block_size > test_utils::get_max_tile_size(acc))
    {
        return;
    }

    const size_t size = block_size * 113;
    // Generate data
    std::vector<T> output = test_utils::get_random_data<T>(size, 2, 200);
    std::vector<T> output_reductions(size / block_size);

    // Calculate expected results on host
    std::vector<T> expected_reductions(output_reductions.size(), 0);
    for(size_t i = 0; i < output.size() / block_size; i++)
    {
        T value = 0;
        for(size_t j = 0; j < block_size; j++)
        {
            auto idx = i * block_size + j;
            value += output[idx];
        }
        expected_reductions[i] = value;
    }

    hc::array_view<T, 1> d_output(output.size(), output.data());
    hc::array_view<T, 1> d_output_r(
        output_reductions.size(), output_reductions.data()
    );
    hc::parallel_for_each(
        acc.get_default_view(),
        hc::extent<1>(output.size()).tile(block_size),
        [=](hc::tiled_index<1> i) [[hc]]
        {
            T value = d_output[i];
            //T reduction;
            rp::block_reduce<T, block_size, algorithm> breduce;
            breduce.reduce(value, value);
            //d_output[i] = value;
            if(i.local[0] == 0)
            {
                d_output_r[i.tile[0]] = value;
            }
        }
    );

    d_output.synchronize();
    d_output_r.synchronize();
    for(size_t i = 0; i < output_reductions.size(); i++)
    {
        ASSERT_EQ(output_reductions[i], expected_reductions[i]);
    }
}

TYPED_TEST(RocprimBlockReduceSingleValueTests, ReduceMultiplies)
{
    using T = typename TestFixture::type;
    constexpr auto algorithm = TestFixture::algorithm;
    constexpr size_t block_size = TestFixture::block_size;

    hc::accelerator acc;
    // Given block size not supported
    if(block_size > test_utils::get_max_tile_size(acc))
    {
        return;
    }

    const size_t size = block_size * 113;
    // Generate data
    std::vector<T> output(size, 1);
    auto two_places = test_utils::get_random_data<unsigned int>(size/32, 0, size-1);
    for(auto i : two_places)
    {
        output[i] = T(2);
    }
    std::vector<T> output_reductions(size / block_size);

    // Calculate expected results on host
    std::vector<T> expected_reductions(output_reductions.size(), 0);
    for(size_t i = 0; i < output.size() / block_size; i++)
    {
        T value = 1;
        for(size_t j = 0; j < block_size; j++)
        {
            auto idx = i * block_size + j;
            value *= output[idx];
        }
        expected_reductions[i] = value;
    }

    hc::array_view<T, 1> d_output(output.size(), output.data());
    hc::array_view<T, 1> d_output_r(
        output_reductions.size(), output_reductions.data()
    );
    hc::parallel_for_each(
        acc.get_default_view(),
        hc::extent<1>(output.size()).tile(block_size),
        [=](hc::tiled_index<1> i) [[hc]]
        {
            T value = d_output[i];
            //T reduction;
            rp::block_reduce<T, block_size, algorithm> breduce;
            breduce.reduce(value, value, rocprim::multiplies<T>());
            //d_output[i] = value;
            if(i.local[0] == 0)
            {
                d_output_r[i.tile[0]] = value;
            }
        }
    );

    d_output.synchronize();
    d_output_r.synchronize();
    for(size_t i = 0; i < output_reductions.size(); i++)
    {
        ASSERT_EQ(output_reductions[i], expected_reductions[i]);
    }
}

TYPED_TEST(RocprimBlockReduceSingleValueTests, ReduceValid)
{
    using T = typename TestFixture::type;
    constexpr auto algorithm = TestFixture::algorithm;
    constexpr size_t block_size = TestFixture::block_size;
    const unsigned int valid_items = test_utils::get_random_value(block_size - 10, block_size);

    hc::accelerator acc;
    // Given block size not supported
    if(block_size > test_utils::get_max_tile_size(acc))
    {
        return;
    }

    const size_t size = block_size * 113;
    // Generate data
    std::vector<T> output = test_utils::get_random_data<T>(size, 2, 200);
    std::vector<T> output_reductions(size / block_size);

    // Calculate expected results on host
    std::vector<T> expected_reductions(output_reductions.size(), 0);
    for(size_t i = 0; i < output.size() / block_size; i++)
    {
        T value = 0;
        for(size_t j = 0; j < valid_items; j++)
        {
            auto idx = i * block_size + j;
            value += output[idx];
        }
        expected_reductions[i] = value;
    }

    hc::array_view<T, 1> d_output(output.size(), output.data());
    hc::array_view<T, 1> d_output_r(
        output_reductions.size(), output_reductions.data()
    );
    hc::parallel_for_each(
        acc.get_default_view(),
        hc::extent<1>(output.size()).tile(block_size),
        [=](hc::tiled_index<1> i) [[hc]]
        {
            T value = d_output[i];
            //T reduction;
            rp::block_reduce<T, block_size, algorithm> breduce;
            breduce.reduce(value, value, valid_items);
            //d_output[i] = value;
            if(i.local[0] == 0)
            {
                d_output_r[i.tile[0]] = value;
            }
        }
    );

    d_output.synchronize();
    d_output_r.synchronize();
    for(size_t i = 0; i < output_reductions.size(); i++)
    {
        ASSERT_EQ(output_reductions[i], expected_reductions[i]);
    }
}

template<class Params>
class RocprimBlockReduceInputArrayTests : public ::testing::Test
{
public:
    using type = typename Params::type;
    static constexpr unsigned int block_size = Params::block_size;
    static constexpr rocprim::block_reduce_algorithm algorithm = Params::algorithm;
    static constexpr unsigned int items_per_thread = Params::items_per_thread;
};

typedef ::testing::Types<
    // -----------------------------------------------------------------------
    // rocprim::block_reduce_algorithm::using_warp_reduce
    // -----------------------------------------------------------------------
    params<float, 6U,   32>,
    params<float, 32,   2>,
    params<unsigned int, 256,  3>,
    params<int, 512,  4>,
    params<float, 1024, 1>,
    params<float, 37,   2>,
    params<float, 65,   5>,
    params<float, 162,  7>,
    params<float, 255,  15>,
    // -----------------------------------------------------------------------
    // rocprim::block_reduce_algorithm::raking_reduce
    // -----------------------------------------------------------------------
    params<float, 6U,   32, rocprim::block_reduce_algorithm::raking_reduce>,
    params<float, 32,   2,  rocprim::block_reduce_algorithm::raking_reduce>,
    params<int, 256,  3,  rocprim::block_reduce_algorithm::raking_reduce>,
    params<unsigned int, 512,  4,  rocprim::block_reduce_algorithm::raking_reduce>,
    params<float, 1024, 1,  rocprim::block_reduce_algorithm::raking_reduce>,
    params<float, 37,   2,  rocprim::block_reduce_algorithm::raking_reduce>,
    params<float, 65,   5,  rocprim::block_reduce_algorithm::raking_reduce>,
    params<float, 162,  7,  rocprim::block_reduce_algorithm::raking_reduce>,
    params<float, 255,  15, rocprim::block_reduce_algorithm::raking_reduce>
> InputArrayTestParams;

TYPED_TEST_CASE(RocprimBlockReduceInputArrayTests, InputArrayTestParams);

TYPED_TEST(RocprimBlockReduceInputArrayTests, Reduce)
{
    using T = typename TestFixture::type;
    constexpr auto algorithm = TestFixture::algorithm;
    constexpr size_t block_size = TestFixture::block_size;
    constexpr size_t items_per_thread = TestFixture::items_per_thread;

    hc::accelerator acc;
    // Given block size not supported
    if(block_size > test_utils::get_max_tile_size(acc))
    {
        return;
    }

    const size_t items_per_block = block_size * items_per_thread;
    const size_t size = items_per_block * 37;
    // Generate data
    std::vector<T> output = test_utils::get_random_data<T>(size, 2, 200);

    // Output reduce results
    std::vector<T> output_reductions(size / block_size);

    // Calculate expected results on host
    std::vector<T> expected_reductions(output_reductions.size(), 0);
    for(size_t i = 0; i < output.size() / items_per_block; i++)
    {
        T value = 0;
        for(size_t j = 0; j < items_per_block; j++)
        {
            auto idx = i * items_per_block + j;
            value += output[idx];
        }
        expected_reductions[i] = value;
    }

    // global/grid size
    const size_t global_size = output.size()/items_per_thread;
    hc::array_view<T, 1> d_output(output.size(), output.data());
    hc::array_view<T, 1> d_output_r(
        output_reductions.size(), output_reductions.data()
    );
    hc::parallel_for_each(
        acc.get_default_view(),
        hc::extent<1>(global_size).tile(block_size),
        [=](hc::tiled_index<1> i) [[hc]]
        {
            size_t idx = i.global[0] * items_per_thread;

            // load
            T in_out[items_per_thread];
            for(unsigned int j = 0; j < items_per_thread; j++)
            {
                in_out[j] = d_output[idx + j];
            }

            rp::block_reduce<T, block_size, algorithm> breduce;
            T reduction;
            breduce.reduce(in_out, reduction);

            if(i.local[0] == 0)
            {
                d_output_r[i.tile[0]] = reduction;
            }
        }
    );

    d_output.synchronize();
    d_output_r.synchronize();
    for(size_t i = 0; i < output_reductions.size(); i++)
    {
        ASSERT_NEAR(
            output_reductions[i], expected_reductions[i],
            static_cast<T>(0.05) * expected_reductions[i]
        );
    }
}
