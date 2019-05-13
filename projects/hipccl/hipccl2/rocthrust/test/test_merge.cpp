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
#include <thrust/extrema.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/retag.h>
#include <thrust/merge.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include "test_header.hpp"

TESTS_DEFINE(MergeTests, FullTestsParams);
TESTS_DEFINE(PrimitiveMergeTests, NumericalTestsParams);

TEST(MergeTests, UsingHip)
{
    ASSERT_EQ(THRUST_DEVICE_SYSTEM, THRUST_DEVICE_SYSTEM_HIP);
}

TYPED_TEST(MergeTests, MergeSimple)
{
    using Vector   = typename TestFixture::input_type;
    using Iterator = typename Vector::iterator;

    Vector a(3), b(4);

    a[0] = 0;
    a[1] = 2;
    a[2] = 4;
    b[0] = 0;
    b[1] = 3;
    b[2] = 3;
    b[3] = 4;

    Vector ref(7);
    ref[0] = 0;
    ref[1] = 0;
    ref[2] = 2;
    ref[3] = 3;
    ref[4] = 3;
    ref[5] = 4;
    ref[6] = 4;

    Vector result(7);

    Iterator end = thrust::merge(a.begin(), a.end(), b.begin(), b.end(), result.begin());

    EXPECT_EQ(result.end(), end);
    ASSERT_EQ(ref, result);
}

template <typename InputIterator1, typename InputIterator2, typename OutputIterator>
OutputIterator merge(my_system& system,
                     InputIterator1,
                     InputIterator1,
                     InputIterator2,
                     InputIterator2,
                     OutputIterator result)
{
    system.validate_dispatch();
    return result;
}

TEST(MergeTests, MergeDispatchExplicit)
{
    thrust::device_vector<int> vec(1);

    my_system sys(0);
    thrust::merge(sys, vec.begin(), vec.begin(), vec.begin(), vec.begin(), vec.begin());

    ASSERT_EQ(true, sys.is_valid());
}

template <typename InputIterator1, typename InputIterator2, typename OutputIterator>
OutputIterator merge(
    my_tag, InputIterator1, InputIterator1, InputIterator2, InputIterator2, OutputIterator result)
{
    *result = 13;
    return result;
}

TEST(MergeTests, MergeDispatchImplicit)
{
    thrust::device_vector<int> vec(1);

    thrust::merge(thrust::retag<my_tag>(vec.begin()),
                  thrust::retag<my_tag>(vec.begin()),
                  thrust::retag<my_tag>(vec.begin()),
                  thrust::retag<my_tag>(vec.begin()),
                  thrust::retag<my_tag>(vec.begin()));

    ASSERT_EQ(13, vec.front());
}

TYPED_TEST(PrimitiveMergeTests, MergeWithRandomData)
{
    using T = typename TestFixture::input_type;

    const std::vector<size_t> sizes = get_sizes();
    for(auto size : sizes)
    {
        size_t expanded_sizes[]   = {0, 1, size / 2, size, size + 1, 2 * size};
        size_t num_expanded_sizes = sizeof(expanded_sizes) / sizeof(size_t);

        thrust::host_vector<T> random = get_random_data<unsigned short int>(
            size + *thrust::max_element(expanded_sizes, expanded_sizes + num_expanded_sizes),
            0,
            255);
        thrust::host_vector<T> h_a(random.begin(), random.begin() + size);
        thrust::host_vector<T> h_b(random.begin() + size, random.end());

        thrust::stable_sort(h_a.begin(), h_a.end());
        thrust::stable_sort(h_b.begin(), h_b.end());

        thrust::device_vector<T> d_a = h_a;
        thrust::device_vector<T> d_b = h_b;

        for(size_t i = 0; i < num_expanded_sizes; i++)
        {
            size_t expanded_size = expanded_sizes[i];

            thrust::host_vector<T>   h_result(size + expanded_size);
            thrust::device_vector<T> d_result(size + expanded_size);

            typename thrust::host_vector<T>::iterator   h_end;
            typename thrust::device_vector<T>::iterator d_end;

            h_end = thrust::merge(
                h_a.begin(), h_a.end(), h_b.begin(), h_b.begin() + expanded_size, h_result.begin());
            h_result.resize(h_end - h_result.begin());

            d_end = thrust::merge(
                d_a.begin(), d_a.end(), d_b.begin(), d_b.begin() + expanded_size, d_result.begin());
            d_result.resize(d_end - d_result.begin());

            thrust::host_vector<T> d_result_h = d_result;
            ASSERT_EQ(h_result, d_result_h);
        }
    }
}

TYPED_TEST(PrimitiveMergeTests, MergeToDiscardIterator)
{
    using T = typename TestFixture::input_type;

    const std::vector<size_t> sizes = get_sizes();
    for(auto size : sizes)
    {
        thrust::host_vector<T> h_a = get_random_data<T>(
            size, std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
        thrust::host_vector<T> h_b = get_random_data<T>(
            size, std::numeric_limits<T>::min(), std::numeric_limits<T>::max());

        thrust::stable_sort(h_a.begin(), h_a.end());
        thrust::stable_sort(h_b.begin(), h_b.end());

        thrust::device_vector<T> d_a = h_a;
        thrust::device_vector<T> d_b = h_b;

        thrust::discard_iterator<> h_result = thrust::merge(
            h_a.begin(), h_a.end(), h_b.begin(), h_b.end(), thrust::make_discard_iterator());

        thrust::discard_iterator<> d_result = thrust::merge(
            d_a.begin(), d_a.end(), d_b.begin(), d_b.end(), thrust::make_discard_iterator());

        thrust::discard_iterator<> reference(2 * size);

        ASSERT_EQ(reference, h_result);
        ASSERT_EQ(reference, d_result);
    }
}

TYPED_TEST(PrimitiveMergeTests, MergeDescending)
{
    using T = typename TestFixture::input_type;

    const std::vector<size_t> sizes = get_sizes();
    for(auto size : sizes)
    {
        thrust::host_vector<T> h_a = get_random_data<T>(
            size, std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
        thrust::host_vector<T> h_b = get_random_data<T>(
            size, std::numeric_limits<T>::min(), std::numeric_limits<T>::max());

        thrust::stable_sort(h_a.begin(), h_a.end(), thrust::greater<T>());
        thrust::stable_sort(h_b.begin(), h_b.end(), thrust::greater<T>());

        thrust::device_vector<T> d_a = h_a;
        thrust::device_vector<T> d_b = h_b;

        thrust::host_vector<T>   h_result(h_a.size() + h_b.size());
        thrust::device_vector<T> d_result(d_a.size() + d_b.size());

        typename thrust::host_vector<T>::iterator   h_end;
        typename thrust::device_vector<T>::iterator d_end;

        h_end = thrust::merge(
            h_a.begin(), h_a.end(), h_b.begin(), h_b.end(), h_result.begin(), thrust::greater<T>());

        d_end = thrust::merge(
            d_a.begin(), d_a.end(), d_b.begin(), d_b.end(), d_result.begin(), thrust::greater<T>());

        ASSERT_EQ(h_result, d_result);
    }
}
