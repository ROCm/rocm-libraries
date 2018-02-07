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
#include <type_traits>

// Google Test
#include <gtest/gtest.h>
// HC API
#include <hcc/hc.hpp>
#include <hcc/hc_short_vector.hpp>
// rocPRIM API
#include <rocprim.hpp>

#include "test_utils.hpp"

namespace rp = rocprim;

template<
    class T,
    class U,
    unsigned int ItemsPerThread,
    bool ShouldBeVectorized
>
struct params
{
    using type = T;
    using vector_type = U;
    static constexpr unsigned int items_per_thread = ItemsPerThread;
    static constexpr bool should_be_vectorized = ShouldBeVectorized;
};

template<
    rp::block_load_method Load,
    rp::block_store_method Store,
    unsigned int BlockSize
>
struct class_params
{
    static constexpr rp::block_load_method load_method = Load;
    static constexpr rp::block_store_method store_method = Store;
    static constexpr unsigned int block_size = BlockSize;
};

template<class ClassParams>
class RocprimBlockLoadStoreClassTests : public ::testing::Test {
public:
    using params = ClassParams;
};

template<class Params>
class RocprimVectorizationTests : public ::testing::Test {
public:
    using params = Params;
};

typedef ::testing::Types<
    class_params<rp::block_load_method::block_load_direct,
                 rp::block_store_method::block_store_direct, 64U>,
    class_params<rp::block_load_method::block_load_direct,
                 rp::block_store_method::block_store_direct, 128U>,
    class_params<rp::block_load_method::block_load_direct,
                 rp::block_store_method::block_store_direct, 256U>,
    class_params<rp::block_load_method::block_load_direct,
                 rp::block_store_method::block_store_direct, 512U>,

    class_params<rp::block_load_method::block_load_vectorize,
                 rp::block_store_method::block_store_vectorize, 64U>,
    class_params<rp::block_load_method::block_load_vectorize,
                 rp::block_store_method::block_store_vectorize, 128U>,
    class_params<rp::block_load_method::block_load_vectorize,
                 rp::block_store_method::block_store_vectorize, 256U>,
    class_params<rp::block_load_method::block_load_vectorize,
                 rp::block_store_method::block_store_vectorize, 512U>,

    class_params<rp::block_load_method::block_load_transpose,
                 rp::block_store_method::block_store_transpose, 64U>,
    class_params<rp::block_load_method::block_load_transpose,
                 rp::block_store_method::block_store_transpose, 128U>,
    class_params<rp::block_load_method::block_load_transpose,
                 rp::block_store_method::block_store_transpose, 256U>,
    class_params<rp::block_load_method::block_load_transpose,
                 rp::block_store_method::block_store_transpose, 512U>,

    class_params<rp::block_load_method::block_load_warp_transpose,
                 rp::block_store_method::block_store_warp_transpose, 64U>,
    class_params<rp::block_load_method::block_load_warp_transpose,
                 rp::block_store_method::block_store_warp_transpose, 128U>,
    class_params<rp::block_load_method::block_load_warp_transpose,
                 rp::block_store_method::block_store_warp_transpose, 256U>,
    class_params<rp::block_load_method::block_load_warp_transpose,
                 rp::block_store_method::block_store_warp_transpose, 512U>
> ClassParams;

typedef ::testing::Types<
    params<int, int, 3, false>,
    params<int, rp::detail::int4, 4, true>,
    params<int, int, 7, false>,
    params<int, rp::detail::int4, 8, true>,
    params<int, int, 11, false>,
    params<int, rp::detail::int4, 16, true>,

    params<char, char, 3, false>,
    params<char, rp::detail::char4, 4, true>,
    params<char, char, 7, false>,
    params<char, rp::detail::char4, 8, true>,
    params<char, char, 11, false>,
    params<char, rp::detail::char4, 16, true>,

    params<short, short, 3, false>,
    params<short, rp::detail::short4, 4, true>,
    params<short, short, 7, false>,
    params<short, rp::detail::short4, 8, true>,
    params<short, short, 11, false>,
    params<short, rp::detail::short4, 16, true>,

    params<float, int, 3, false>,
    params<float, rp::detail::int4, 4, true>,
    params<float, int, 7, false>,
    params<float, rp::detail::int4, 8, true>,
    params<float, int, 11, false>,
    params<float, rp::detail::int4, 16, true>,

    params<hc::short_vector::int2, rp::detail::int2, 3, false>,
    params<hc::short_vector::int2, rp::detail::int4, 4, true>,
    params<hc::short_vector::int2, rp::detail::int2, 7, false>,
    params<hc::short_vector::int2, rp::detail::int4, 8, true>,
    params<hc::short_vector::int2, rp::detail::int2, 11, false>,
    params<hc::short_vector::int2, rp::detail::int4, 16, true>,

    params<hc::short_vector::float2, rp::detail::int2, 3, false>,
    params<hc::short_vector::float2, rp::detail::int4, 4, true>,
    params<hc::short_vector::float2, rp::detail::int2, 7, false>,
    params<hc::short_vector::float2, rp::detail::int4, 8, true>,
    params<hc::short_vector::float2, rp::detail::int2, 11, false>,
    params<hc::short_vector::float2, rp::detail::int4, 16, true>,

    params<hc::short_vector::char4, int, 3, false>,
    params<hc::short_vector::char4, rp::detail::int4, 4, true>,
    params<hc::short_vector::char4, int, 7, false>,
    params<hc::short_vector::char4, rp::detail::int4, 8, true>,
    params<hc::short_vector::char4, int, 11, false>,
    params<hc::short_vector::char4, rp::detail::int4, 16, true>
> Params;

TYPED_TEST_CASE(RocprimBlockLoadStoreClassTests, ClassParams);
TYPED_TEST_CASE(RocprimVectorizationTests, Params);

TYPED_TEST(RocprimBlockLoadStoreClassTests, LoadStoreClass)
{
    hc::accelerator acc;

    constexpr size_t block_size = TestFixture::params::block_size;
    constexpr rp::block_load_method load_method = TestFixture::params::load_method;
    constexpr rp::block_store_method store_method = TestFixture::params::store_method;
    // Given block size not supported
    if(block_size > get_max_tile_size(acc) || (block_size & (block_size - 1)) != 0)
    {
        return;
    }

    const size_t items_per_thread = 4;
    constexpr auto items_per_block = block_size * items_per_thread;
    const size_t size = items_per_block * 113;
    const auto grid_size = size / items_per_thread;
    // Generate data
    std::vector<int> input = get_random_data<int>(size, -100, 100);
    std::vector<int> output(input.size(), 0);

    // Calculate expected results on host
    std::vector<int> expected(input.size(), 0);
    for (size_t i = 0; i < 113; i++)
    {
        size_t block_offset = i * items_per_block;
        for (size_t j = 0; j < items_per_block; j++)
        {
            expected[j + block_offset] = input[j + block_offset];
        }
    }

    hc::array_view<int, 1> d_input(input.size(), input.data());
    hc::array_view<int, 1> d_output(output.size(), output.data());
    hc::parallel_for_each(
        acc.get_default_view(),
        hc::extent<1>(grid_size).tile(block_size),
        [=](hc::tiled_index<1> i) [[hc]]
        {
            int t[items_per_thread];
            int offset = i.tile[0] * block_size * items_per_thread;
            rp::block_load<int, block_size, items_per_thread, load_method> load;
            rp::block_store<int, block_size, items_per_thread, store_method> store;
            load.load(d_input.data() + offset, t);
            store.store(d_output.data() + offset, t);
        }
    );

    d_input.synchronize();
    d_output.synchronize();
    for(size_t i = 0; i < output.size(); i++)
    {
        ASSERT_EQ(output[i], expected[i]);
    }
}

TYPED_TEST(RocprimBlockLoadStoreClassTests, LoadStoreClassValid)
{
    hc::accelerator acc;

    constexpr size_t block_size = TestFixture::params::block_size;
    constexpr rp::block_load_method load_method = TestFixture::params::load_method;
    constexpr rp::block_store_method store_method = TestFixture::params::store_method;
    // Given block size not supported
    if(block_size > get_max_tile_size(acc) || (block_size & (block_size - 1)) != 0)
    {
        return;
    }

    const size_t items_per_thread = 4;
    constexpr auto items_per_block = block_size * items_per_thread;
    const size_t size = items_per_block * 113;
    const auto grid_size = size / items_per_thread;
    const size_t valid = items_per_block - 32;
    // Generate data
    std::vector<int> input = get_random_data<int>(size, -100, 100);
    std::vector<int> output(input.size(), 0);

    // Calculate expected results on host
    std::vector<int> expected(input.size(), 0);
    for (size_t i = 0; i < 113; i++)
    {
        size_t block_offset = i * items_per_block;
        for (size_t j = 0; j < items_per_block; j++)
        {
            if (j < valid)
            {
                expected[j + block_offset] = input[j + block_offset];
            }
        }
    }

    hc::array_view<int, 1> d_input(input.size(), input.data());
    hc::array_view<int, 1> d_output(output.size(), output.data());
    hc::parallel_for_each(
        acc.get_default_view(),
        hc::extent<1>(grid_size).tile(block_size),
        [=](hc::tiled_index<1> i) [[hc]]
        {
            int t[items_per_thread];
            int offset = i.tile[0] * block_size * items_per_thread;
            rp::block_load<int, block_size, items_per_thread, load_method> load;
            rp::block_store<int, block_size, items_per_thread, store_method> store;
            load.load(d_input.data() + offset, t, valid);
            store.store(d_output.data() + offset, t, valid);
        }
    );

    d_input.synchronize();
    d_output.synchronize();
    for(size_t i = 0; i < output.size(); i++)
    {
        ASSERT_EQ(output[i], expected[i]);
    }
}

TYPED_TEST(RocprimBlockLoadStoreClassTests, LoadStoreClassDefault)
{
    hc::accelerator acc;

    constexpr size_t block_size = TestFixture::params::block_size;
    constexpr rp::block_load_method load_method = TestFixture::params::load_method;
    constexpr rp::block_store_method store_method = TestFixture::params::store_method;
    // Given block size not supported
    if(block_size > get_max_tile_size(acc) || (block_size & (block_size - 1)) != 0)
    {
        return;
    }

    const size_t items_per_thread = 4;
    constexpr auto items_per_block = block_size * items_per_thread;
    const size_t size = items_per_block * 113;
    const auto grid_size = size / items_per_thread;
    const size_t valid = items_per_thread + 1;
    int _default = -1;
    // Generate data
    std::vector<int> input = get_random_data<int>(size, -100, 100);
    std::vector<int> output(input.size(), 0);

    // Calculate expected results on host
    std::vector<int> expected(input.size(), _default);
    for (size_t i = 0; i < 113; i++)
    {
        size_t block_offset = i * items_per_block;
        for (size_t j = 0; j < items_per_block; j++)
        {
            if (j < valid)
            {
                expected[j + block_offset] = input[j + block_offset];
            }
        }
    }

    hc::array_view<int, 1> d_input(input.size(), input.data());
    hc::array_view<int, 1> d_output(output.size(), output.data());
    hc::parallel_for_each(
        acc.get_default_view(),
        hc::extent<1>(grid_size).tile(block_size),
        [=](hc::tiled_index<1> i) [[hc]]
        {
            int t[items_per_thread];
            int offset = i.tile[0] * block_size * items_per_thread;
            rp::block_load<int, block_size, items_per_thread, load_method> load;
            rp::block_store<int, block_size, items_per_thread, store_method> store;
            load.load(d_input.data() + offset, t, valid, _default);
            store.store(d_output.data() + offset, t);
        }
    );

    d_input.synchronize();
    d_output.synchronize();
    for(size_t i = 0; i < output.size(); i++)
    {
        ASSERT_EQ(output[i], expected[i]);
    }
}

TYPED_TEST(RocprimVectorizationTests, IsVectorizable)
{
    using T = typename TestFixture::params::type;
    constexpr size_t items_per_thread = TestFixture::params::items_per_thread;
    constexpr bool should_be_vectorized = TestFixture::params::should_be_vectorized;
    bool input = rp::detail::is_vectorizable<T, items_per_thread>();
    ASSERT_EQ(input, should_be_vectorized);
}

TYPED_TEST(RocprimVectorizationTests, MatchVectorType)
{
    using T = typename TestFixture::params::type;
    using U = typename TestFixture::params::vector_type;
    constexpr size_t items_per_thread = TestFixture::params::items_per_thread;
    typedef typename rp::detail::match_vector_type<T, items_per_thread>::type Vector;
    bool input = std::is_same<Vector, U>::value;
    EXPECT_TRUE(input);
}

