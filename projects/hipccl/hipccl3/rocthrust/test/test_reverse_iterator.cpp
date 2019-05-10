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

#include <thrust/iterator/reverse_iterator.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>

#include "test_header.hpp"

TESTS_DEFINE(ReverseIteratorTests, FullTestsParams);

TESTS_DEFINE(PrimitiveReverseIteratorTests, NumericalTestsParams);

TEST(ReverseIteratorTests, UsingHip)
{
    ASSERT_EQ(THRUST_DEVICE_SYSTEM, THRUST_DEVICE_SYSTEM_HIP);
}

TEST(ReverseIteratorTests, ReverseIteratorCopyConstructor)
{
    thrust::host_vector<int> h_v(1, 13);

    thrust::reverse_iterator<thrust::host_vector<int>::iterator> h_iter0(h_v.end());
    thrust::reverse_iterator<thrust::host_vector<int>::iterator> h_iter1(h_iter0);

    ASSERT_EQ(h_iter0, h_iter1);
    ASSERT_EQ(*h_iter0, *h_iter1);

    thrust::device_vector<int> d_v(1, 13);

    thrust::reverse_iterator<thrust::device_vector<int>::iterator> d_iter2(d_v.end());
    thrust::reverse_iterator<thrust::device_vector<int>::iterator> d_iter3(d_iter2);

    ASSERT_EQ(d_iter2, d_iter3);
    ASSERT_EQ(*d_iter2, *d_iter3);
}

TEST(ReverseIteratorTests, ReverseIteratorIncrement)
{
    thrust::host_vector<int> h_v(4);
    thrust::sequence(h_v.begin(), h_v.end());

    thrust::reverse_iterator<thrust::host_vector<int>::iterator> h_iter(h_v.end());

    ASSERT_EQ(*h_iter, 3);

    h_iter++;
    ASSERT_EQ(*h_iter, 2);

    h_iter++;
    ASSERT_EQ(*h_iter, 1);

    h_iter++;
    ASSERT_EQ(*h_iter, 0);

    thrust::device_vector<int> d_v(4);
    thrust::sequence(d_v.begin(), d_v.end());

    thrust::reverse_iterator<thrust::device_vector<int>::iterator> d_iter(d_v.end());

    ASSERT_EQ(*d_iter, 3);

    d_iter++;
    ASSERT_EQ(*d_iter, 2);

    d_iter++;
    ASSERT_EQ(*d_iter, 1);

    d_iter++;
    ASSERT_EQ(*d_iter, 0);
}

TYPED_TEST(ReverseIteratorTests, ReverseIteratorCopy)
{
    using Vector = typename TestFixture::input_type;
    using T      = typename Vector::value_type;

    Vector source(4);
    source[0] = (T)10;
    source[1] = (T)20;
    source[2] = (T)30;
    source[3] = (T)40;

    Vector destination(4, 0);

    thrust::copy(thrust::make_reverse_iterator(source.end()),
                 thrust::make_reverse_iterator(source.begin()),
                 destination.begin());

    ASSERT_EQ(destination[0], (T)40);
    ASSERT_EQ(destination[1], (T)30);
    ASSERT_EQ(destination[2], (T)20);
    ASSERT_EQ(destination[3], (T)10);
}

TYPED_TEST(PrimitiveReverseIteratorTests, ReverseIteratorExclusiveScanSimple)
{
    using T = typename TestFixture::input_type;

    const size_t size = 10;

    T                      error_margin = (T)0.01 * size;
    thrust::host_vector<T> h_data(size);
    thrust::sequence(h_data.begin(), h_data.end());

    thrust::device_vector<T> d_data = h_data;

    thrust::host_vector<T>   h_result(h_data.size());
    thrust::device_vector<T> d_result(d_data.size());

    thrust::exclusive_scan(thrust::make_reverse_iterator(h_data.end()),
                           thrust::make_reverse_iterator(h_data.begin()),
                           h_result.begin());

    thrust::exclusive_scan(thrust::make_reverse_iterator(d_data.end()),
                           thrust::make_reverse_iterator(d_data.begin()),
                           d_result.begin());

    thrust::host_vector<T> h_result_d(d_result);
    for(size_t i = 0; i < size; i++)
        ASSERT_NEAR(h_result[i], h_result_d[i], error_margin);
}

TYPED_TEST(PrimitiveReverseIteratorTests, ReverseIteratorExclusiveScan)
{
    using T = typename TestFixture::input_type;

    const std::vector<size_t> sizes = get_sizes();
    for(auto size : sizes)
    {
        T                      error_margin = (T)0.01 * size;
        thrust::host_vector<T> h_data       = get_random_data<T>(size, 0, 10);

        thrust::device_vector<T> d_data = h_data;

        thrust::host_vector<T>   h_result(size);
        thrust::device_vector<T> d_result(size);

        thrust::exclusive_scan(thrust::make_reverse_iterator(h_data.end()),
                               thrust::make_reverse_iterator(h_data.begin()),
                               h_result.begin());

        thrust::exclusive_scan(thrust::make_reverse_iterator(d_data.end()),
                               thrust::make_reverse_iterator(d_data.begin()),
                               d_result.begin());

        thrust::host_vector<T> h_result_d(d_result);
        for(size_t i = 0; i < size; i++)
            ASSERT_NEAR(h_result[i], h_result_d[i], error_margin);
    }
};
