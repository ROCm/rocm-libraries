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

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/retag.h>
#include <thrust/sort.h>

#include "test_header.hpp"

template <class Key, class Item, class CompareFunction = thrust::less<Key>>
struct ParamsSort
{
    using key_type         = Key;
    using value_type       = Item;
    using compare_function = CompareFunction;
};

template <class ParamsSort>
class SortTests : public ::testing::Test
{
public:
    using key_type         = typename ParamsSort::key_type;
    using value_type       = typename ParamsSort::value_type;
    using compare_function = typename ParamsSort::compare_function;
};

typedef ::testing::Types<ParamsSort<unsigned short, int, thrust::less<unsigned short>>,
                         ParamsSort<unsigned short, int, thrust::greater<unsigned short>>,
                         ParamsSort<unsigned short, int, custom_compare_less<unsigned short>>,
                         ParamsSort<unsigned short, double>,
                         ParamsSort<int, long long>>
    SortTestsParams;

TYPED_TEST_CASE(SortTests, SortTestsParams);

TESTS_DEFINE(SortVector, FullTestsParams);
TESTS_DEFINE(SortVectorPrimitives, NumericalTestsParams);

TYPED_TEST(SortTests, Sort)
{
    using key_type         = typename TestFixture::key_type;
    using compare_function = typename TestFixture::compare_function;

    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    for(auto size : get_sizes())
    {
        SCOPED_TRACE(testing::Message() << "with size = " << size);

        thrust::host_vector<key_type> h_keys;
        if(std::is_floating_point<key_type>::value)
        {
            for(auto seed : get_seeds())
            {
                SCOPED_TRACE(testing::Message() << "with seed= " << seed);

                h_keys = get_random_data<key_type>(
                    size, (key_type)-1000, (key_type) + 1000, seed);
            }
        }
        else
        {
            for(auto seed : get_seeds())
            {
                SCOPED_TRACE(testing::Message() << "with seed= " << seed);

                h_keys = get_random_data<key_type>(size,
                                                   std::numeric_limits<key_type>::min(),
                                                   std::numeric_limits<key_type>::max(),
                                                   seed);
            }
        }

        // Calculate expected results on host
        thrust::host_vector<key_type> expected(h_keys);
        thrust::sort(expected.begin(), expected.end(), compare_function());

        thrust::device_vector<key_type> d_keys(h_keys);
        thrust::sort(d_keys.begin(), d_keys.end(), compare_function());

        h_keys = d_keys;
        for(size_t i = 0; i < size; i++)
        {
            ASSERT_EQ(h_keys[i], expected[i]) << "where index = " << i;
        }
    }
}

TYPED_TEST(SortTests, SortByKey)
{
    using key_type         = typename TestFixture::key_type;
    using value_type       = typename TestFixture::value_type;
    using compare_function = typename TestFixture::compare_function;

    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    for(auto size : get_sizes())
    {
        SCOPED_TRACE(testing::Message() << "with size = " << size);

        // Check if non-stable sort can be used (no equal keys with different values)
        if(size > static_cast<size_t>(std::numeric_limits<key_type>::max()))
            continue;

        thrust::host_vector<key_type> h_keys(size);
        std::iota(h_keys.begin(), h_keys.end(), 0);
        std::shuffle(
            h_keys.begin(), h_keys.end(), std::default_random_engine(std::random_device{}()));

        thrust::host_vector<value_type> h_values(size);
        std::iota(h_values.begin(), h_values.end(), 0);

        // Calculate expected results on host
        thrust::host_vector<key_type>   expected_keys(h_keys);
        thrust::host_vector<value_type> expected_values(h_values);
        thrust::sort_by_key(expected_keys.begin(),
                            expected_keys.end(),
                            expected_values.begin(),
                            compare_function());

        thrust::device_vector<key_type>   d_keys(h_keys);
        thrust::device_vector<value_type> d_values(h_values);
        thrust::sort_by_key(d_keys.begin(), d_keys.end(), d_values.begin(), compare_function());

        h_keys   = d_keys;
        h_values = d_values;
        for(size_t i = 0; i < size; i++)
        {
            ASSERT_EQ(h_keys[i], expected_keys[i]) << "where index = " << i;
            ASSERT_EQ(h_values[i], expected_values[i]) << "where index = " << i;
        }
    }
}

TYPED_TEST(SortTests, StableSort)
{
    using key_type         = typename TestFixture::key_type;
    using compare_function = typename TestFixture::compare_function;

    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    for(auto size : get_sizes())
    {
        SCOPED_TRACE(testing::Message() << "with size = " << size);

        thrust::host_vector<key_type> h_keys;
        if(std::is_floating_point<key_type>::value)
        {
            for(auto seed : get_seeds())
            {
                SCOPED_TRACE(testing::Message() << "with seed= " << seed);

                h_keys = get_random_data<key_type>(
                    size, (key_type)-1000, (key_type) + 1000, seed);
            }
        }
        else
        {
            for(auto seed : get_seeds())
            {
                SCOPED_TRACE(testing::Message() << "with seed= " << seed);

                h_keys = get_random_data<key_type>(size,
                                                   std::numeric_limits<key_type>::min(),
                                                   std::numeric_limits<key_type>::max(),
                                                   seed);
            }
        }

        // Calculate expected results on host
        thrust::host_vector<key_type> expected(h_keys);
        thrust::stable_sort(expected.begin(), expected.end(), compare_function());

        thrust::device_vector<key_type> d_keys(h_keys);
        thrust::stable_sort(d_keys.begin(), d_keys.end(), compare_function());

        h_keys = d_keys;
        for(size_t i = 0; i < size; i++)
        {
            ASSERT_EQ(h_keys[i], expected[i]) << "where index = " << i;
        }
    }
}

TYPED_TEST(SortTests, StableSortByKey)
{
    using key_type         = typename TestFixture::key_type;
    using value_type       = typename TestFixture::value_type;
    using compare_function = typename TestFixture::compare_function;

    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    for(auto size : get_sizes())
    {
        SCOPED_TRACE(testing::Message() << "with size = " << size);

        thrust::host_vector<key_type> h_keys;
        if(std::is_floating_point<key_type>::value)
        {
            for(auto seed : get_seeds())
            {
                SCOPED_TRACE(testing::Message() << "with seed= " << seed);

                h_keys = get_random_data<key_type>(
                    size, (key_type)-1000, (key_type) + 1000, seed);
            }
        }
        else
        {
            for(auto seed : get_seeds())
            {
                SCOPED_TRACE(testing::Message() << "with seed= " << seed);

                h_keys = get_random_data<key_type>(size,
                                                   std::numeric_limits<key_type>::min(),
                                                   std::numeric_limits<key_type>::max(),
                                                   seed);
            }
        }

        thrust::host_vector<value_type> h_values(size);
        std::iota(h_values.begin(), h_values.end(), 0);

        // Calculate expected results on host
        thrust::host_vector<key_type>   expected_keys(h_keys);
        thrust::host_vector<value_type> expected_values(h_values);
        thrust::stable_sort_by_key(expected_keys.begin(),
                                   expected_keys.end(),
                                   expected_values.begin(),
                                   compare_function());

        thrust::device_vector<key_type>   d_keys(h_keys);
        thrust::device_vector<value_type> d_values(h_values);
        thrust::stable_sort_by_key(
            d_keys.begin(), d_keys.end(), d_values.begin(), compare_function());

        h_keys   = d_keys;
        h_values = d_values;
        for(size_t i = 0; i < size; i++)
        {
            ASSERT_EQ(h_keys[i], expected_keys[i]) << "where index = " << i;
            ASSERT_EQ(h_values[i], expected_values[i]) << "where index = " << i;
        }
    }
}

template <typename RandomAccessIterator>
void sort(my_system& system, RandomAccessIterator, RandomAccessIterator)
{
    system.validate_dispatch();
}

TEST(SortTests, TestSortDispatchExplicit)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    thrust::device_vector<int> vec(1);

    my_system sys(0);
    thrust::sort(sys, vec.begin(), vec.begin());

    ASSERT_EQ(true, sys.is_valid());
}

template <typename RandomAccessIterator>
void sort(my_tag, RandomAccessIterator first, RandomAccessIterator)
{
    *first = 13;
}

TEST(SortTests, TestSortDispatchImplicit)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    thrust::device_vector<int> vec(1);

    thrust::sort(thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.begin()));

    ASSERT_EQ(13, vec.front());
}

template <class Vector>
void InitializeSimpleKeySortTest(Vector& unsorted_keys, Vector& sorted_keys)
{
    using T = typename Vector::value_type;

    unsorted_keys.resize(7);
    unsorted_keys[0] = T(1);
    unsorted_keys[1] = T(3);
    unsorted_keys[2] = T(6);
    unsorted_keys[3] = T(5);
    unsorted_keys[4] = T(2);
    unsorted_keys[5] = T(0);
    unsorted_keys[6] = T(4);

    sorted_keys.resize(7);
    sorted_keys[0] = T(0);
    sorted_keys[1] = T(1);
    sorted_keys[2] = T(2);
    sorted_keys[3] = T(3);
    sorted_keys[4] = T(4);
    sorted_keys[5] = T(5);
    sorted_keys[6] = T(6);
}

TYPED_TEST(SortVector, TestSortSimple)
{
    using Vector = typename TestFixture::input_type;

    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    Vector unsorted_keys;
    Vector sorted_keys;

    InitializeSimpleKeySortTest(unsorted_keys, sorted_keys);

    thrust::sort(unsorted_keys.begin(), unsorted_keys.end());

    ASSERT_EQ(unsorted_keys, sorted_keys);
}

TYPED_TEST(SortVectorPrimitives, TestSortAscendingKey)
{
    using T = typename TestFixture::input_type;

    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    for(auto size : get_sizes())
    {
        SCOPED_TRACE(testing::Message() << "with size= " << size);

        for(auto seed : get_seeds())
        {
            SCOPED_TRACE(testing::Message() << "with seed= " << seed);

            thrust::host_vector<T> h_data = get_random_data<T>(
                size, std::numeric_limits<T>::min(), std::numeric_limits<T>::max(), seed);
            thrust::device_vector<T> d_data = h_data;

            thrust::sort(h_data.begin(), h_data.end(), thrust::less<T>());
            thrust::sort(d_data.begin(), d_data.end(), thrust::less<T>());

            ASSERT_EQ(h_data, d_data);
        }
    }
}

TEST(SortTests, TestSortDescendingKey)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());
    
    for(auto size : { 10027 })
    {
        SCOPED_TRACE(testing::Message() << "with size= " << size);

        for(auto seed : get_seeds())
        {
            SCOPED_TRACE(testing::Message() << "with seed= " << seed);

            thrust::host_vector<int> h_data = get_random_data<int>(
                size, std::numeric_limits<int>::min(), std::numeric_limits<int>::max(), seed);
            thrust::device_vector<int> d_data = h_data;

            thrust::sort(h_data.begin(), h_data.end(), thrust::greater<int>());
            thrust::sort(d_data.begin(), d_data.end(), thrust::greater<int>());

            ASSERT_EQ(h_data, d_data);
        }
    }
}

TEST(SortTests, TestSortBool)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());
    
    for(auto size : { 10027 })
    {
        SCOPED_TRACE(testing::Message() << "with size= " << size);

        for(auto seed : get_seeds())
        {
            SCOPED_TRACE(testing::Message() << "with seed= " << seed);

            thrust::host_vector<bool> h_data = get_random_data<bool>(
                size, std::numeric_limits<bool>::min(), std::numeric_limits<bool>::max(), seed);

            thrust::device_vector<bool> d_data = h_data;

            thrust::sort(h_data.begin(), h_data.end());
            thrust::sort(d_data.begin(), d_data.end());

            ASSERT_EQ(h_data, d_data);
        }
    }
}

TEST(SortTests, TestSortBoolDescending)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    for(auto size : { 10027 })
    {
        SCOPED_TRACE(testing::Message() << "with size= " << size);

        for(auto seed : get_seeds())
        {
            SCOPED_TRACE(testing::Message() << "with seed= " << seed);

            thrust::host_vector<bool> h_data = get_random_data<bool>(
                size, std::numeric_limits<bool>::min(), std::numeric_limits<bool>::max(), seed);

            thrust::device_vector<bool> d_data = h_data;

            thrust::sort(h_data.begin(), h_data.end(), thrust::greater<bool>());
            thrust::sort(d_data.begin(), d_data.end(), thrust::greater<bool>());

            ASSERT_EQ(h_data, d_data);
        }
    }
}

//TODO: refactor this test into a different set of tests
__global__
THRUST_HIP_LAUNCH_BOUNDS_DEFAULT
void SortKernel(int const N, int* array)
{
    if(threadIdx.x == 0)
    {
        thrust::device_ptr<int> begin(array);
        thrust::device_ptr<int> end(array + N);
        thrust::sort(thrust::hip::par, begin, end);
    }
}

TEST(SortTests, TestSortDevice)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());
    
    for(auto size: {0, 1, 2, 4, 6, 12, 16, 24, 32, 64, 84, 128, 160, 256} )
    {
        SCOPED_TRACE(testing::Message() << "with size= " << size);

        for(auto seed : get_seeds())
        {
            SCOPED_TRACE(testing::Message() << "with seed= " << seed);

            thrust::host_vector<int> h_data = get_random_data<int>(size, 0, size, seed);

            thrust::device_vector<int> d_data = h_data;

            thrust::sort(h_data.begin(), h_data.end());
            hipLaunchKernelGGL(SortKernel,
                               dim3(1, 1, 1),
                               dim3(128, 1, 1),
                               0,
                               0,
                               size,
                               thrust::raw_pointer_cast(&d_data[0]));

            ASSERT_EQ(h_data, d_data);
        }
    }
}
