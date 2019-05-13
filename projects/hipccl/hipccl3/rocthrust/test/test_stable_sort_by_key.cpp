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

TESTS_DEFINE(StableSortByKeyTests, UnsignedIntegerTestsParams);
TESTS_DEFINE(StableSortByKeyVectorTests, VectorIntegerTestsParams);
TESTS_DEFINE(StableSortByKeyVectorPrimitiveTests, IntegerTestsParams);

template <typename RandomAccessIterator1, typename RandomAccessIterator2>
void stable_sort_by_key(my_system& system,
                        RandomAccessIterator1,
                        RandomAccessIterator1,
                        RandomAccessIterator2)
{
    system.validate_dispatch();
}

TEST(StableSortByKeyTests, TestStableSortByKeyDispatchExplicit)
{
    thrust::device_vector<int> vec(1);

    my_system sys(0);
    thrust::stable_sort_by_key(sys, vec.begin(), vec.begin(), vec.begin());

    ASSERT_EQ(true, sys.is_valid());
}

template <typename RandomAccessIterator1, typename RandomAccessIterator2>
void stable_sort_by_key(my_tag,
                        RandomAccessIterator1 keys_first,
                        RandomAccessIterator1,
                        RandomAccessIterator2)
{
    *keys_first = 13;
}

TEST(StableSortByKeyTests, TestStableSortByKeyDispatchImplicit)
{
    thrust::device_vector<int> vec(1);

    thrust::stable_sort_by_key(thrust::retag<my_tag>(vec.begin()),
                               thrust::retag<my_tag>(vec.begin()),
                               thrust::retag<my_tag>(vec.begin()));

    ASSERT_EQ(13, vec.front());
}

template <typename T>
struct less_div_10
{
    __host__ __device__ bool operator()(const T& lhs, const T& rhs) const
    {
        return ((int)lhs) / 10 < ((int)rhs) / 10;
    }
};

template <class Vector>
void InitializeSimpleStableKeyValueSortTest(Vector& unsorted_keys,
                                            Vector& unsorted_values,
                                            Vector& sorted_keys,
                                            Vector& sorted_values)
{
    unsorted_keys.resize(9);
    unsorted_values.resize(9);
    unsorted_keys[0]   = 25;
    unsorted_values[0] = 0;
    unsorted_keys[1]   = 14;
    unsorted_values[1] = 1;
    unsorted_keys[2]   = 35;
    unsorted_values[2] = 2;
    unsorted_keys[3]   = 16;
    unsorted_values[3] = 3;
    unsorted_keys[4]   = 26;
    unsorted_values[4] = 4;
    unsorted_keys[5]   = 34;
    unsorted_values[5] = 5;
    unsorted_keys[6]   = 36;
    unsorted_values[6] = 6;
    unsorted_keys[7]   = 24;
    unsorted_values[7] = 7;
    unsorted_keys[8]   = 15;
    unsorted_values[8] = 8;

    sorted_keys.resize(9);
    sorted_values.resize(9);
    sorted_keys[0]   = 14;
    sorted_values[0] = 1;
    sorted_keys[1]   = 16;
    sorted_values[1] = 3;
    sorted_keys[2]   = 15;
    sorted_values[2] = 8;
    sorted_keys[3]   = 25;
    sorted_values[3] = 0;
    sorted_keys[4]   = 26;
    sorted_values[4] = 4;
    sorted_keys[5]   = 24;
    sorted_values[5] = 7;
    sorted_keys[6]   = 35;
    sorted_values[6] = 2;
    sorted_keys[7]   = 34;
    sorted_values[7] = 5;
    sorted_keys[8]   = 36;
    sorted_values[8] = 6;
}

TYPED_TEST(StableSortByKeyVectorTests, TestStableSortByKeySimple)
{
    using Vector = typename TestFixture::input_type;
    using T      = typename Vector::value_type;

    Vector unsorted_keys, unsorted_values;
    Vector sorted_keys, sorted_values;

    InitializeSimpleStableKeyValueSortTest(
        unsorted_keys, unsorted_values, sorted_keys, sorted_values);

    thrust::stable_sort_by_key(
        unsorted_keys.begin(), unsorted_keys.end(), unsorted_values.begin(), less_div_10<T>());

    ASSERT_EQ(unsorted_keys, sorted_keys);
    ASSERT_EQ(unsorted_values, sorted_values);
}

TYPED_TEST(StableSortByKeyVectorPrimitiveTests, TestStableSortByKey)
{
    using T = typename TestFixture::input_type;

    for(auto size : get_sizes())
    {
        thrust::host_vector<T> h_keys = get_random_data<T>(
            size, std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
        thrust::device_vector<T> d_keys = h_keys;

        thrust::host_vector<T> h_values = get_random_data<T>(
            size, std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
        thrust::device_vector<T> d_values = h_values;

        thrust::stable_sort_by_key(h_keys.begin(), h_keys.end(), h_values.begin());
        thrust::stable_sort_by_key(d_keys.begin(), d_keys.end(), d_values.begin());

        ASSERT_EQ(h_keys, d_keys);
        ASSERT_EQ(h_values, d_values);
    }
}