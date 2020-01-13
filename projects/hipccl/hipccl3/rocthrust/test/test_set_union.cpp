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

#include <thrust/extrema.h>
#include <thrust/functional.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/retag.h>
#include <thrust/set_operations.h>
#include <thrust/sort.h>

#include "test_header.hpp"

TESTS_DEFINE(SetUnionTests, FullTestsParams);
TESTS_DEFINE(SetUnionPrimitiveTests, NumericalTestsParams);

template <typename InputIterator1, typename InputIterator2, typename OutputIterator>
OutputIterator set_union(my_system& system,
                         InputIterator1,
                         InputIterator1,
                         InputIterator2,
                         InputIterator2,
                         OutputIterator result)
{
    system.validate_dispatch();
    return result;
}

TEST(SetUnionTests, TestSetUnionDispatchExplicit)
{
    thrust::device_vector<int> vec(1);

    my_system sys(0);
    thrust::set_union(sys, vec.begin(), vec.begin(), vec.begin(), vec.begin(), vec.begin());

    ASSERT_EQ(true, sys.is_valid());
}

template <typename InputIterator1, typename InputIterator2, typename OutputIterator>
OutputIterator set_union(
    my_tag, InputIterator1, InputIterator1, InputIterator2, InputIterator2, OutputIterator result)
{
    *result = 13;
    return result;
}

TEST(SetUnionTests, TestSetUnionDispatchImplicit)
{
    thrust::device_vector<int> vec(1);

    thrust::set_union(thrust::retag<my_tag>(vec.begin()),
                      thrust::retag<my_tag>(vec.begin()),
                      thrust::retag<my_tag>(vec.begin()),
                      thrust::retag<my_tag>(vec.begin()),
                      thrust::retag<my_tag>(vec.begin()));

    ASSERT_EQ(13, vec.front());
}

TYPED_TEST(SetUnionTests, TestSetUnionSimple)
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

    Vector ref(5);
    ref[0] = 0;
    ref[1] = 2;
    ref[2] = 3;
    ref[3] = 3;
    ref[4] = 4;

    Vector result(5);

    Iterator end = thrust::set_union(a.begin(), a.end(), b.begin(), b.end(), result.begin());

    EXPECT_EQ(result.end(), end);
    ASSERT_EQ(ref, result);
}

TYPED_TEST(SetUnionTests, TestSetUnionWithEquivalentElementsSimple)
{
    using Vector   = typename TestFixture::input_type;
    using Iterator = typename Vector::iterator;

    Vector a(3), b(5);

    a[0] = 0;
    a[1] = 2;
    a[2] = 2;
    b[0] = 0;
    b[1] = 2;
    b[2] = 2;
    b[3] = 2;
    b[4] = 3;

    Vector ref(5);
    ref[0] = 0;
    ref[1] = 2;
    ref[2] = 2;
    ref[3] = 2;
    ref[4] = 3;

    Vector result(5);

    Iterator end = thrust::set_union(a.begin(), a.end(), b.begin(), b.end(), result.begin());

    EXPECT_EQ(result.end(), end);
    ASSERT_EQ(ref, result);
}

TYPED_TEST(SetUnionPrimitiveTests, TestSetUnion)
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

            h_end = thrust::set_union(
                h_a.begin(), h_a.end(), h_b.begin(), h_b.begin() + expanded_size, h_result.begin());
            h_result.resize(h_end - h_result.begin());

            d_end = thrust::set_union(
                d_a.begin(), d_a.end(), d_b.begin(), d_b.begin() + expanded_size, d_result.begin());
            d_result.resize(d_end - d_result.begin());

            ASSERT_EQ(h_result, d_result);
        }
    }
}

TYPED_TEST(SetUnionPrimitiveTests, TestSetUnionToDiscardIterator)
{
    using T = typename TestFixture::input_type;

    const std::vector<size_t> sizes = get_sizes();

    for(auto size : sizes)
    {
        thrust::host_vector<T> temp = get_random_data<T>(
            2 * size, std::numeric_limits<T>::min(), std::numeric_limits<T>::max());

        thrust::host_vector<T> h_a(temp.begin(), temp.begin() + size);
        thrust::host_vector<T> h_b(temp.begin() + size, temp.end());

        thrust::sort(h_a.begin(), h_a.end());
        thrust::sort(h_b.begin(), h_b.end());

        thrust::device_vector<T> d_a = h_a;
        thrust::device_vector<T> d_b = h_b;

        thrust::discard_iterator<> h_result;
        thrust::discard_iterator<> d_result;

        thrust::host_vector<T>                    h_reference(2 * size);
        typename thrust::host_vector<T>::iterator h_end = thrust::set_union(
            h_a.begin(), h_a.end(), h_b.begin(), h_b.end(), h_reference.begin());
        h_reference.erase(h_end, h_reference.end());

        h_result = thrust::set_union(
            h_a.begin(), h_a.end(), h_b.begin(), h_b.end(), thrust::make_discard_iterator());

        d_result = thrust::set_union(
            d_a.begin(), d_a.end(), d_b.begin(), d_b.end(), thrust::make_discard_iterator());

        thrust::discard_iterator<> reference(h_reference.size());

        EXPECT_EQ(reference, h_result);
        EXPECT_EQ(reference, d_result);
    }
}
