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

TESTS_DEFINE(StableSortTests, UnsignedIntegerTestsParams);
TESTS_DEFINE(StableSortVectorTests, VectorIntegerTestsParams);

template <typename RandomAccessIterator>
void stable_sort(my_system& system, RandomAccessIterator, RandomAccessIterator)
{
    system.validate_dispatch();
}

TEST(StableSortTests, TestStableSortDispatchExplicit)
{
    thrust::device_vector<int> vec(1);

    my_system sys(0);
    thrust::stable_sort(sys, vec.begin(), vec.begin());

    ASSERT_EQ(true, sys.is_valid());
}

template <typename RandomAccessIterator>
void stable_sort(my_tag, RandomAccessIterator first, RandomAccessIterator)
{
    *first = 13;
}

TEST(StableSortTests, TestStableSortDispatchImplicit)
{
    thrust::device_vector<int> vec(1);

    thrust::stable_sort(thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.begin()));

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
void InitializeSimpleStableKeySortTest(Vector& unsorted_keys, Vector& sorted_keys)
{
    unsorted_keys.resize(9);
    unsorted_keys[0] = 25;
    unsorted_keys[1] = 14;
    unsorted_keys[2] = 35;
    unsorted_keys[3] = 16;
    unsorted_keys[4] = 26;
    unsorted_keys[5] = 34;
    unsorted_keys[6] = 36;
    unsorted_keys[7] = 24;
    unsorted_keys[8] = 15;

    sorted_keys.resize(9);
    sorted_keys[0] = 14;
    sorted_keys[1] = 16;
    sorted_keys[2] = 15;
    sorted_keys[3] = 25;
    sorted_keys[4] = 26;
    sorted_keys[5] = 24;
    sorted_keys[6] = 35;
    sorted_keys[7] = 34;
    sorted_keys[8] = 36;
}

TYPED_TEST(StableSortVectorTests, TestStableSortSimple)
{
    using Vector = typename TestFixture::input_type;
    using T      = typename Vector::value_type;

    Vector unsorted_keys;
    Vector sorted_keys;

    InitializeSimpleStableKeySortTest(unsorted_keys, sorted_keys);

    thrust::stable_sort(unsorted_keys.begin(), unsorted_keys.end(), less_div_10<T>());

    ASSERT_EQ(unsorted_keys, sorted_keys);
}

TYPED_TEST(StableSortTests, TestStableSort)
{
    using T = typename TestFixture::input_type;

    for(auto size : get_sizes())
    {
        thrust::host_vector<T> h_data = get_random_data<T>(
            size, std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
        thrust::device_vector<T> d_data = h_data;

        thrust::stable_sort(h_data.begin(), h_data.end(), less_div_10<T>());
        thrust::stable_sort(d_data.begin(), d_data.end(), less_div_10<T>());

        ASSERT_EQ(h_data, d_data);
    }
}

template <typename T>
struct comp_mod3
{
    T* table;

    comp_mod3(T* table)
        : table(table)
    {
    }

    __host__ __device__ bool operator()(T a, T b) const
    {
        return table[(int)a] < table[(int)b];
    }
};

TYPED_TEST(StableSortVectorTests, TestStableSortWithIndirection)
{
    using Vector = typename TestFixture::input_type;
    using T      = typename Vector::value_type;

    Vector data(7);
    data[0] = T(1);
    data[1] = T(3);
    data[2] = T(5);
    data[3] = T(3);
    data[4] = T(0);
    data[5] = T(2);
    data[6] = T(1);

    Vector table(6);
    table[0] = T(0);
    table[1] = T(1);
    table[2] = T(2);
    table[3] = T(0);
    table[4] = T(1);
    table[5] = T(2);

    thrust::stable_sort(
        data.begin(), data.end(), comp_mod3<T>(thrust::raw_pointer_cast(&table[0])));

    ASSERT_EQ(data[0], T(3));
    ASSERT_EQ(data[1], T(3));
    ASSERT_EQ(data[2], T(0));
    ASSERT_EQ(data[3], T(1));
    ASSERT_EQ(data[4], T(1));
    ASSERT_EQ(data[5], T(5));
    ASSERT_EQ(data[6], T(2));
}
