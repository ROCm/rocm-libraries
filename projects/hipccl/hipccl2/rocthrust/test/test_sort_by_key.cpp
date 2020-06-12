/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *  Modifications Copyright© 2019 Advanced Micro Devices, Inc. All rights reserved.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#include <thrust/functional.h>
#include <thrust/iterator/retag.h>
#include <thrust/sort.h>

#include "test_header.hpp"

TESTS_DEFINE(SortByKeyTests, FullTestsParams);
TESTS_DEFINE(SortByKeyPrimitiveTests, NumericalTestsParams);

template <typename RandomAccessIterator1, typename RandomAccessIterator2>
void sort_by_key(my_system& system,
                 RandomAccessIterator1,
                 RandomAccessIterator1,
                 RandomAccessIterator2)
{
    system.validate_dispatch();
}

TEST(SortByKeyTests, TestSortByKeyDispatchExplicit)
{
    thrust::device_vector<int> vec(1);

    my_system sys(0);
    thrust::sort_by_key(sys, vec.begin(), vec.begin(), vec.begin());

    ASSERT_EQ(true, sys.is_valid());
}

template <typename RandomAccessIterator1, typename RandomAccessIterator2>
void sort_by_key(my_tag,
                 RandomAccessIterator1 keys_first,
                 RandomAccessIterator1,
                 RandomAccessIterator2)
{
    *keys_first = 13;
}

TEST(SortByKeyTests, TestSortByKeyDispatchImplicit)
{
    thrust::device_vector<int> vec(1);

    thrust::sort_by_key(thrust::retag<my_tag>(vec.begin()),
                        thrust::retag<my_tag>(vec.begin()),
                        thrust::retag<my_tag>(vec.begin()));

    ASSERT_EQ(13, vec.front());
}

template <class Vector>
void InitializeSimpleKeyValueSortTest(Vector& unsorted_keys,
                                      Vector& unsorted_values,
                                      Vector& sorted_keys,
                                      Vector& sorted_values)
{
    using T = typename Vector::value_type;

    unsorted_keys.resize(7);
    unsorted_values.resize(7);
    unsorted_keys[0]   = T(1);
    unsorted_values[0] = T(0);
    unsorted_keys[1]   = T(3);
    unsorted_values[1] = T(1);
    unsorted_keys[2]   = T(6);
    unsorted_values[2] = T(2);
    unsorted_keys[3]   = T(5);
    unsorted_values[3] = T(3);
    unsorted_keys[4]   = T(2);
    unsorted_values[4] = T(4);
    unsorted_keys[5]   = T(0);
    unsorted_values[5] = T(5);
    unsorted_keys[6]   = T(4);
    unsorted_values[6] = T(6);

    sorted_keys.resize(7);
    sorted_values.resize(7);
    sorted_keys[0]   = T(0);
    sorted_values[1] = T(0);
    sorted_keys[1]   = T(1);
    sorted_values[3] = T(1);
    sorted_keys[2]   = T(2);
    sorted_values[6] = T(2);
    sorted_keys[3]   = T(3);
    sorted_values[5] = T(3);
    sorted_keys[4]   = T(4);
    sorted_values[2] = T(4);
    sorted_keys[5]   = T(5);
    sorted_values[0] = T(5);
    sorted_keys[6]   = T(6);
    sorted_values[4] = T(6);
}

TYPED_TEST(SortByKeyTests, TestSortByKeySimple)
{
    using Vector = typename TestFixture::input_type;

    Vector unsorted_keys, unsorted_values;
    Vector sorted_keys, sorted_values;

    InitializeSimpleKeyValueSortTest(unsorted_keys, unsorted_values, sorted_keys, sorted_values);

    thrust::sort_by_key(unsorted_keys.begin(), unsorted_keys.end(), unsorted_values.begin());

    ASSERT_EQ(unsorted_keys, sorted_keys);
    ASSERT_EQ(unsorted_values, sorted_values);
}

TYPED_TEST(SortByKeyPrimitiveTests, TestSortAscendingKeyValue)
{
    using T = typename TestFixture::input_type;

    for(auto size : get_sizes())
    {
        SCOPED_TRACE(testing::Message() << "with size = " << size);
        for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
        {
            unsigned int seed_value
                = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
            SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

            thrust::host_vector<T> h_keys = get_random_data<T>(
                size, std::numeric_limits<T>::min(), std::numeric_limits<T>::max(), seed_value);

            thrust::device_vector<T> d_keys = h_keys;

            thrust::host_vector<T>   h_values = h_keys;
            thrust::device_vector<T> d_values = d_keys;

            thrust::sort_by_key(h_keys.begin(), h_keys.end(), h_values.begin(), thrust::less<T>());
            thrust::sort_by_key(d_keys.begin(), d_keys.end(), d_values.begin(), thrust::less<T>());

            ASSERT_EQ(h_keys, d_keys);
            ASSERT_EQ(h_values, d_values);
        }
    }
}

TEST(SortByKeyTests, TestSortDescendingKeyValue)
{
    for(auto size : get_sizes())
    {
        SCOPED_TRACE(testing::Message() << "with size = " << size);
        for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
        {
            unsigned int seed_value
                = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
            SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

            thrust::host_vector<int> h_keys = get_random_data<int>(
                size, std::numeric_limits<int>::min(), std::numeric_limits<int>::max(), seed_value);

            thrust::device_vector<int> d_keys = h_keys;

            thrust::host_vector<int>   h_values = h_keys;
            thrust::device_vector<int> d_values = d_keys;

            thrust::sort_by_key(
                h_keys.begin(), h_keys.end(), h_values.begin(), thrust::greater<int>());
            thrust::sort_by_key(
                d_keys.begin(), d_keys.end(), d_values.begin(), thrust::greater<int>());

            ASSERT_EQ(h_keys, d_keys);
            ASSERT_EQ(h_values, d_values);
        }
    }
}

TEST(SortByKeyTests, TestSortByKeyBool)
{
    const size_t size = 10027;

    for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        thrust::host_vector<bool> h_keys = get_random_data<bool>(
            size, std::numeric_limits<bool>::min(), std::numeric_limits<bool>::max(), seed_value);

        thrust::host_vector<int> h_values = get_random_data<int>(
            size,
            std::numeric_limits<int>::min(),
            std::numeric_limits<int>::max(),
            seed_value + seed_value_addition
        );

        thrust::device_vector<bool> d_keys   = h_keys;
        thrust::device_vector<int>  d_values = h_values;

        thrust::sort_by_key(h_keys.begin(), h_keys.end(), h_values.begin());
        thrust::sort_by_key(d_keys.begin(), d_keys.end(), d_values.begin());

        ASSERT_EQ(h_keys, d_keys);
        ASSERT_EQ(h_values, d_values);
    }
}

TEST(SortByKeyTests, TestSortByKeyBoolDescending)
{
    const size_t size = 10027;

    for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        thrust::host_vector<bool> h_keys = get_random_data<bool>(
            size, std::numeric_limits<bool>::min(), std::numeric_limits<bool>::max(), seed_value);

        thrust::host_vector<int> h_values = get_random_data<int>(
            size, std::numeric_limits<int>::min(), std::numeric_limits<int>::max(), seed_value);

        thrust::device_vector<bool> d_keys   = h_keys;
        thrust::device_vector<int>  d_values = h_values;

        thrust::sort_by_key(
            h_keys.begin(), h_keys.end(), h_values.begin(), thrust::greater<bool>());
        thrust::sort_by_key(
            d_keys.begin(), d_keys.end(), d_values.begin(), thrust::greater<bool>());

        ASSERT_EQ(h_keys, d_keys);
        ASSERT_EQ(h_values, d_values);
    }
}

//TODO: refactor this test into a different set of tests
__global__
THRUST_HIP_ATTRIBUTE_WORK_GROUP_SIZE_RANGE_DEFAULT
void SortByKeyKernel(int const N, int* keys, short* values)
{
    if(threadIdx.x == 0)
    {
        thrust::device_ptr<int>   keys_begin(keys);
        thrust::device_ptr<int>   keys_end(keys + N);
        thrust::device_ptr<short> val(values);
        thrust::sort_by_key(thrust::hip::par, keys_begin, keys_end, val);
    }
}

TEST(SortByKeyTests, TestSortByKeyDevice)
{
    std::vector<size_t> sizes = {0, 1, 2, 4, 6, 12, 16, 24, 32, 64, 84, 128, 160, 256};

    for(auto size : sizes)
    {
        SCOPED_TRACE(testing::Message() << "with size= " << size);
        for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
        {
            unsigned int seed_value
                = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
            SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

            thrust::host_vector<int> h_keys = get_random_data<int>(size, 0, size, seed_value);

            thrust::host_vector<short> h_values
                = get_random_data<short>(size,
                                         std::numeric_limits<short>::min(),
                                         std::numeric_limits<short>::max(),
                                         seed_value);

            thrust::device_vector<int>   d_keys   = h_keys;
            thrust::device_vector<short> d_values = h_values;

            thrust::sort_by_key(h_keys.begin(), h_keys.end(), h_values.begin());
            hipLaunchKernelGGL(SortByKeyKernel,
                               dim3(1, 1, 1),
                               dim3(128, 1, 1),
                               0,
                               0,
                               size,
                               thrust::raw_pointer_cast(&d_keys[0]),
                               thrust::raw_pointer_cast(&d_values[0]));

            ASSERT_EQ(h_keys, d_keys);
            // Only keys are compared here, the sequential stable_merge_sort that's used in
            // CUDA and HIP don't generate the correct value sorting
        }
    }
}
