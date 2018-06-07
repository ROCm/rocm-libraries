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
#include <rocprim/rocprim.hpp>

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
    class Type,
    rp::block_load_method Load,
    rp::block_store_method Store,
    unsigned int BlockSize,
    unsigned int ItemsPerThread
>
struct class_params
{
    using type = Type;
    static constexpr rp::block_load_method load_method = Load;
    static constexpr rp::block_store_method store_method = Store;
    static constexpr unsigned int block_size = BlockSize;
    static constexpr unsigned int items_per_thread = ItemsPerThread;
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
    // block_load_direct
    class_params<int, rp::block_load_method::block_load_direct,
                 rp::block_store_method::block_store_direct, 64U, 1>,
    class_params<int, rp::block_load_method::block_load_direct,
                 rp::block_store_method::block_store_direct, 64U, 4>,
    class_params<int, rp::block_load_method::block_load_direct,
                 rp::block_store_method::block_store_direct, 256U, 1>,
    class_params<int, rp::block_load_method::block_load_direct,
                 rp::block_store_method::block_store_direct, 256U, 4>,
    class_params<int, rp::block_load_method::block_load_direct,
                 rp::block_store_method::block_store_direct, 512U, 1>,
    class_params<int, rp::block_load_method::block_load_direct,
                 rp::block_store_method::block_store_direct, 512U, 4>,

    class_params<double, rp::block_load_method::block_load_direct,
                 rp::block_store_method::block_store_direct, 64U, 1>,
    class_params<double, rp::block_load_method::block_load_direct,
                 rp::block_store_method::block_store_direct, 64U, 4>,
    class_params<double, rp::block_load_method::block_load_direct,
                 rp::block_store_method::block_store_direct, 256U, 1>,
    class_params<double, rp::block_load_method::block_load_direct,
                 rp::block_store_method::block_store_direct, 256U, 4>,
    class_params<double, rp::block_load_method::block_load_direct,
                 rp::block_store_method::block_store_direct, 512U, 1>,
    class_params<double, rp::block_load_method::block_load_direct,
                 rp::block_store_method::block_store_direct, 512U, 4>,

    class_params<test_utils::custom_test_type<int>, rp::block_load_method::block_load_direct,
                 rp::block_store_method::block_store_direct, 64U, 1>,
    class_params<test_utils::custom_test_type<int>, rp::block_load_method::block_load_direct,
                 rp::block_store_method::block_store_direct, 64U, 4>,
    class_params<test_utils::custom_test_type<double>, rp::block_load_method::block_load_direct,
                 rp::block_store_method::block_store_direct, 256U, 1>,
    class_params<test_utils::custom_test_type<double>, rp::block_load_method::block_load_direct,
                 rp::block_store_method::block_store_direct, 256U, 4>,

    // block_load_vectorize
    class_params<int, rp::block_load_method::block_load_vectorize,
                 rp::block_store_method::block_store_vectorize, 64U, 1>,
    class_params<int, rp::block_load_method::block_load_vectorize,
                 rp::block_store_method::block_store_vectorize, 64U, 4>,
    class_params<int, rp::block_load_method::block_load_vectorize,
                 rp::block_store_method::block_store_vectorize, 256U, 1>,
    class_params<int, rp::block_load_method::block_load_vectorize,
                 rp::block_store_method::block_store_vectorize, 256U, 4>,
    class_params<int, rp::block_load_method::block_load_vectorize,
                 rp::block_store_method::block_store_vectorize, 512U, 1>,
    class_params<int, rp::block_load_method::block_load_vectorize,
                 rp::block_store_method::block_store_vectorize, 512U, 4>,

    class_params<double, rp::block_load_method::block_load_vectorize,
                 rp::block_store_method::block_store_vectorize, 64U, 1>,
    class_params<double, rp::block_load_method::block_load_vectorize,
                 rp::block_store_method::block_store_vectorize, 64U, 4>,
    class_params<double, rp::block_load_method::block_load_vectorize,
                 rp::block_store_method::block_store_vectorize, 256U, 1>,
    class_params<double, rp::block_load_method::block_load_vectorize,
                 rp::block_store_method::block_store_vectorize, 256U, 4>,
    class_params<double, rp::block_load_method::block_load_vectorize,
                 rp::block_store_method::block_store_vectorize, 512U, 1>,
    class_params<double, rp::block_load_method::block_load_vectorize,
                 rp::block_store_method::block_store_vectorize, 512U, 4>,

    class_params<test_utils::custom_test_type<int>, rp::block_load_method::block_load_vectorize,
                 rp::block_store_method::block_store_vectorize, 64U, 1>,
    class_params<test_utils::custom_test_type<int>, rp::block_load_method::block_load_vectorize,
                 rp::block_store_method::block_store_vectorize, 64U, 4>,
    class_params<test_utils::custom_test_type<double>, rp::block_load_method::block_load_vectorize,
                 rp::block_store_method::block_store_vectorize, 256U, 1>,
    class_params<test_utils::custom_test_type<double>, rp::block_load_method::block_load_vectorize,
                 rp::block_store_method::block_store_vectorize, 256U, 4>,

    // block_load_transpose
    class_params<int, rp::block_load_method::block_load_transpose,
                 rp::block_store_method::block_store_transpose, 64U, 1>,
    class_params<int, rp::block_load_method::block_load_transpose,
                 rp::block_store_method::block_store_transpose, 64U, 4>,
    class_params<int, rp::block_load_method::block_load_transpose,
                 rp::block_store_method::block_store_transpose, 256U, 1>,
    class_params<int, rp::block_load_method::block_load_transpose,
                 rp::block_store_method::block_store_transpose, 256U, 4>,
    class_params<int, rp::block_load_method::block_load_transpose,
                 rp::block_store_method::block_store_transpose, 512U, 1>,
    class_params<int, rp::block_load_method::block_load_transpose,
                 rp::block_store_method::block_store_transpose, 512U, 4>,

    class_params<double, rp::block_load_method::block_load_transpose,
                 rp::block_store_method::block_store_transpose, 64U, 1>,
    class_params<double, rp::block_load_method::block_load_transpose,
                 rp::block_store_method::block_store_transpose, 64U, 4>,
    class_params<double, rp::block_load_method::block_load_transpose,
                 rp::block_store_method::block_store_transpose, 256U, 1>,
    class_params<double, rp::block_load_method::block_load_transpose,
                 rp::block_store_method::block_store_transpose, 256U, 4>,
    class_params<double, rp::block_load_method::block_load_transpose,
                 rp::block_store_method::block_store_transpose, 512U, 1>,
    class_params<double, rp::block_load_method::block_load_transpose,
                 rp::block_store_method::block_store_transpose, 512U, 4>,

    class_params<test_utils::custom_test_type<int>, rp::block_load_method::block_load_transpose,
                 rp::block_store_method::block_store_transpose, 64U, 1>,
    class_params<test_utils::custom_test_type<int>, rp::block_load_method::block_load_transpose,
                 rp::block_store_method::block_store_transpose, 64U, 4>,
    class_params<test_utils::custom_test_type<double>, rp::block_load_method::block_load_transpose,
                 rp::block_store_method::block_store_transpose, 256U, 1>,
    class_params<test_utils::custom_test_type<double>, rp::block_load_method::block_load_transpose,
                 rp::block_store_method::block_store_transpose, 256U, 4>

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

    using Type = typename TestFixture::params::type;
    constexpr size_t block_size = TestFixture::params::block_size;
    constexpr rp::block_load_method load_method = TestFixture::params::load_method;
    constexpr rp::block_store_method store_method = TestFixture::params::store_method;
    // Given block size not supported
    if(block_size > test_utils::get_max_tile_size(acc) || (block_size & (block_size - 1)) != 0)
    {
        return;
    }

    const size_t items_per_thread = TestFixture::params::items_per_thread;
    constexpr auto items_per_block = block_size * items_per_thread;
    const size_t size = items_per_block * 113;
    const auto grid_size = size / items_per_thread;
    // Generate data
    std::vector<Type> input = test_utils::get_random_data<Type>(size, -100, 100);
    std::vector<Type> output(input.size(), 0);

    // Calculate expected results on host
    std::vector<Type> expected(input.size(), 0);
    for (size_t i = 0; i < 113; i++)
    {
        size_t block_offset = i * items_per_block;
        for (size_t j = 0; j < items_per_block; j++)
        {
            expected[j + block_offset] = input[j + block_offset];
        }
    }

    hc::array_view<Type, 1> d_input(input.size(), input.data());
    hc::array_view<Type, 1> d_output(output.size(), output.data());
    hc::parallel_for_each(
        acc.get_default_view(),
        hc::extent<1>(grid_size).tile(block_size),
        [=](hc::tiled_index<1> i) [[hc]]
        {
            Type t[items_per_thread];
            int offset = i.tile[0] * block_size * items_per_thread;
            rp::block_load<Type, block_size, items_per_thread, load_method> load;
            rp::block_store<Type, block_size, items_per_thread, store_method> store;
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

    using Type = typename TestFixture::params::type;
    constexpr size_t block_size = TestFixture::params::block_size;
    constexpr rp::block_load_method load_method = TestFixture::params::load_method;
    constexpr rp::block_store_method store_method = TestFixture::params::store_method;
    // Given block size not supported
    if(block_size > test_utils::get_max_tile_size(acc) || (block_size & (block_size - 1)) != 0)
    {
        return;
    }

    const size_t items_per_thread = TestFixture::params::items_per_thread;
    constexpr auto items_per_block = block_size * items_per_thread;
    const size_t size = items_per_block * 113;
    const auto grid_size = size / items_per_thread;
    const size_t valid = items_per_block - 32;
    // Generate data
    std::vector<Type> input = test_utils::get_random_data<Type>(size, -100, 100);
    std::vector<Type> output(input.size(), 0);

    // Calculate expected results on host
    std::vector<Type> expected(input.size(), 0);
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

    hc::array_view<Type, 1> d_input(input.size(), input.data());
    hc::array_view<Type, 1> d_output(output.size(), output.data());
    hc::parallel_for_each(
        acc.get_default_view(),
        hc::extent<1>(grid_size).tile(block_size),
        [=](hc::tiled_index<1> i) [[hc]]
        {
            Type t[items_per_thread];
            int offset = i.tile[0] * block_size * items_per_thread;
            rp::block_load<Type, block_size, items_per_thread, load_method> load;
            rp::block_store<Type, block_size, items_per_thread, store_method> store;
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

    using Type = typename TestFixture::params::type;
    constexpr size_t block_size = TestFixture::params::block_size;
    constexpr rp::block_load_method load_method = TestFixture::params::load_method;
    constexpr rp::block_store_method store_method = TestFixture::params::store_method;
    // Given block size not supported
    if(block_size > test_utils::get_max_tile_size(acc) || (block_size & (block_size - 1)) != 0)
    {
        return;
    }

    const size_t items_per_thread = TestFixture::params::items_per_thread;
    constexpr auto items_per_block = block_size * items_per_thread;
    const size_t size = items_per_block * 113;
    const auto grid_size = size / items_per_thread;
    const size_t valid = items_per_thread + 1;
    int _default = -1;
    // Generate data
    std::vector<Type> input = test_utils::get_random_data<Type>(size, -100, 100);
    std::vector<Type> output(input.size(), 0);

    // Calculate expected results on host
    std::vector<Type> expected(input.size(), _default);
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

    hc::array_view<Type, 1> d_input(input.size(), input.data());
    hc::array_view<Type, 1> d_output(output.size(), output.data());
    hc::parallel_for_each(
        acc.get_default_view(),
        hc::extent<1>(grid_size).tile(block_size),
        [=](hc::tiled_index<1> i) [[hc]]
        {
            Type t[items_per_thread];
            int offset = i.tile[0] * block_size * items_per_thread;
            rp::block_load<Type, block_size, items_per_thread, load_method> load;
            rp::block_store<Type, block_size, items_per_thread, store_method> store;
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
    ASSERT_TRUE(input);
}

