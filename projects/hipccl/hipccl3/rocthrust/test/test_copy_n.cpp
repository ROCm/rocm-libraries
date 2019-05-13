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

#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/retag.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sequence.h>

#include "test_header.hpp"

TESTS_DEFINE(CopyNTests, FullTestsParams);
TESTS_DEFINE(CopyNPrimitiveTests, NumericalTestsParams);

TYPED_TEST(CopyNPrimitiveTests, TestCopyNFromConstIterator)
{
    using T = typename TestFixture::input_type;

    std::vector<T> v(5);
    v[0] = T(0);
    v[1] = T(1);
    v[2] = T(2);
    v[3] = T(3);
    v[4] = T(4);

    typename std::vector<T>::const_iterator begin = v.begin();

    // copy to host_vector
    thrust::host_vector<T>                    h(5, (T)10);
    typename thrust::host_vector<T>::iterator h_result = thrust::copy_n(begin, h.size(), h.begin());
    ASSERT_EQ(h[0], T(0));
    ASSERT_EQ(h[1], T(1));
    ASSERT_EQ(h[2], T(2));
    ASSERT_EQ(h[3], T(3));
    ASSERT_EQ(h[4], T(4));
    ASSERT_EQ_QUIET(h_result, h.end());

    // copy to device_vector
    thrust::device_vector<T>                    d(5, (T)10);
    typename thrust::device_vector<T>::iterator d_result
        = thrust::copy_n(begin, d.size(), d.begin());
    ASSERT_EQ(d[0], T(0));
    ASSERT_EQ(d[1], T(1));
    ASSERT_EQ(d[2], T(2));
    ASSERT_EQ(d[3], T(3));
    ASSERT_EQ(d[4], T(4));
    ASSERT_EQ_QUIET(d_result, d.end());
}

TYPED_TEST(CopyNPrimitiveTests, TestCopyNToDiscardIterator)
{
    using T = typename TestFixture::input_type;

    thrust::host_vector<T>   h_input(5, 1);
    thrust::device_vector<T> d_input = h_input;

    // copy from host_vector
    thrust::discard_iterator<> h_result
        = thrust::copy_n(h_input.begin(), h_input.size(), thrust::make_discard_iterator());

    // copy from device_vector
    thrust::discard_iterator<> d_result
        = thrust::copy_n(d_input.begin(), d_input.size(), thrust::make_discard_iterator());

    thrust::discard_iterator<> reference(5);

    ASSERT_EQ_QUIET(reference, h_result);
    ASSERT_EQ_QUIET(reference, d_result);
}

TYPED_TEST(CopyNTests, TestCopyNMatchingTypes)
{
    using Vector = typename TestFixture::input_type;
    using T      = typename Vector::value_type;

    Vector v(5);
    v[0] = T(0);
    v[1] = T(1);
    v[2] = T(2);
    v[3] = T(3);
    v[4] = T(4);

    // copy to host_vector
    thrust::host_vector<T>                    h(5, (T)10);
    typename thrust::host_vector<T>::iterator h_result
        = thrust::copy_n(v.begin(), v.size(), h.begin());
    ASSERT_EQ(h[0], T(0));
    ASSERT_EQ(h[1], T(1));
    ASSERT_EQ(h[2], T(2));
    ASSERT_EQ(h[3], T(3));
    ASSERT_EQ(h[4], T(4));
    ASSERT_EQ_QUIET(h_result, h.end());

    // copy to device_vector
    thrust::device_vector<T>                    d(5, (T)10);
    typename thrust::device_vector<T>::iterator d_result
        = thrust::copy_n(v.begin(), v.size(), d.begin());
    ASSERT_EQ(d[0], T(0));
    ASSERT_EQ(d[1], T(1));
    ASSERT_EQ(d[2], T(2));
    ASSERT_EQ(d[3], T(3));
    ASSERT_EQ(d[4], T(4));
    ASSERT_EQ_QUIET(d_result, d.end());
}

TYPED_TEST(CopyNTests, TestCopyNMixedTypes)
{
    using Vector = typename TestFixture::input_type;
    using T      = typename Vector::value_type;

    Vector v(5);
    v[0] = T(0);
    v[1] = T(1);
    v[2] = T(2);
    v[3] = T(3);
    v[4] = T(4);

    // copy to host_vector with different type
    thrust::host_vector<float>                    h(5, (float)10);
    typename thrust::host_vector<float>::iterator h_result
        = thrust::copy_n(v.begin(), v.size(), h.begin());

    ASSERT_EQ(h[0], T(0));
    ASSERT_EQ(h[1], T(1));
    ASSERT_EQ(h[2], T(2));
    ASSERT_EQ(h[3], T(3));
    ASSERT_EQ(h[4], T(4));
    ASSERT_EQ_QUIET(h_result, h.end());

    // copy to device_vector with different type
    thrust::device_vector<float>                    d(5, (float)10);
    typename thrust::device_vector<float>::iterator d_result
        = thrust::copy_n(v.begin(), v.size(), d.begin());
    ASSERT_EQ(d[0], T(0));
    ASSERT_EQ(d[1], T(1));
    ASSERT_EQ(d[2], T(2));
    ASSERT_EQ(d[3], T(3));
    ASSERT_EQ(d[4], T(4));
    ASSERT_EQ_QUIET(d_result, d.end());
}

TEST(CopyNTests, TestCopyNVectorBool)
{
    std::vector<bool> v(3);
    v[0] = true;
    v[1] = false;
    v[2] = true;

    thrust::host_vector<bool>   h(3);
    thrust::device_vector<bool> d(3);

    thrust::copy_n(v.begin(), v.size(), h.begin());
    thrust::copy_n(v.begin(), v.size(), d.begin());

    ASSERT_EQ(h[0], true);
    ASSERT_EQ(h[1], false);
    ASSERT_EQ(h[2], true);

    ASSERT_EQ(d[0], true);
    ASSERT_EQ(d[1], false);
    ASSERT_EQ(d[2], true);
}

TYPED_TEST(CopyNTests, TestCopyNListTo)
{
    using Vector = typename TestFixture::input_type;
    using T      = typename Vector::value_type;

    // copy from list to Vector
    std::list<T> l;
    l.push_back(0);
    l.push_back(1);
    l.push_back(2);
    l.push_back(3);
    l.push_back(4);

    Vector v(l.size());

    typename Vector::iterator v_result = thrust::copy_n(l.begin(), l.size(), v.begin());

    ASSERT_EQ(v[0], 0);
    ASSERT_EQ(v[1], 1);
    ASSERT_EQ(v[2], 2);
    ASSERT_EQ(v[3], 3);
    ASSERT_EQ(v[4], 4);
    ASSERT_EQ_QUIET(v_result, v.end());

    l.clear();

    thrust::copy_n(v.begin(), v.size(), std::back_insert_iterator<std::list<T>>(l));

    ASSERT_EQ(l.size(), 5);

    typename std::list<T>::const_iterator iter = l.begin();
    ASSERT_EQ(*iter, 0);
    iter++;
    ASSERT_EQ(*iter, 1);
    iter++;
    ASSERT_EQ(*iter, 2);
    iter++;
    ASSERT_EQ(*iter, 3);
    iter++;
    ASSERT_EQ(*iter, 4);
    iter++;
}

TYPED_TEST(CopyNTests, TestCopyNCountingIterator)
{
    using Vector = typename TestFixture::input_type;
    using T      = typename Vector::value_type;

    thrust::counting_iterator<T> iter(1);

    Vector vec(4);

    thrust::copy_n(iter, 4, vec.begin());

    ASSERT_EQ(vec[0], T(1));
    ASSERT_EQ(vec[1], T(2));
    ASSERT_EQ(vec[2], T(3));
    ASSERT_EQ(vec[3], T(4));
}

TYPED_TEST(CopyNTests, TestCopyNZipIterator)
{
    using Vector = typename TestFixture::input_type;
    using T      = typename Vector::value_type;

    Vector v1(3);
    v1[0] = T(1);
    v1[1] = T(2);
    v1[2] = T(3);
    Vector v2(3);
    v2[0] = T(4);
    v2[1] = T(5);
    v2[2] = T(6);
    Vector v3(3, T(0));
    Vector v4(3, T(0));

    thrust::copy_n(thrust::make_zip_iterator(thrust::make_tuple(v1.begin(), v2.begin())),
                   3,
                   thrust::make_zip_iterator(thrust::make_tuple(v3.begin(), v4.begin())));

    ASSERT_EQ(v1, v3);
    ASSERT_EQ(v2, v4);
}

TYPED_TEST(CopyNTests, TestCopyNConstantIteratorToZipIterator)
{
    using Vector = typename TestFixture::input_type;
    using T      = typename Vector::value_type;

    Vector v1(3, T(0));
    Vector v2(3, T(0));

    thrust::copy_n(thrust::make_constant_iterator(thrust::tuple<T, T>(4, 7)),
                   v1.size(),
                   thrust::make_zip_iterator(thrust::make_tuple(v1.begin(), v2.begin())));

    ASSERT_EQ(v1[0], T(4));
    ASSERT_EQ(v1[1], T(4));
    ASSERT_EQ(v1[2], T(4));
    ASSERT_EQ(v2[0], T(7));
    ASSERT_EQ(v2[1], T(7));
    ASSERT_EQ(v2[2], T(7));
}

template <typename InputIterator, typename Size, typename OutputIterator>
OutputIterator copy_n(my_system& system, InputIterator, Size, OutputIterator result)
{
    system.validate_dispatch();
    return result;
}

TEST(CopyNTests, TestCopyNDispatchExplicit)
{
    thrust::device_vector<int> vec(1);

    my_system sys(0);
    thrust::copy_n(sys, vec.begin(), 1, vec.begin());

    ASSERT_EQ(true, sys.is_valid());
}

template <typename InputIterator, typename Size, typename OutputIterator>
OutputIterator copy_n(my_tag, InputIterator, Size, OutputIterator result)
{
    *result = 13;
    return result;
}

TEST(CopyNTests, TestCopyNDispatchImplicit)
{
    thrust::device_vector<int> vec(1);

    thrust::copy_n(thrust::retag<my_tag>(vec.begin()), 1, thrust::retag<my_tag>(vec.begin()));

    ASSERT_EQ(13, vec.front());
}
