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
#include <thrust/functional.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/retag.h>
#include <thrust/tabulate.h>

#include "test_header.hpp"

TESTS_DEFINE(MismatchTests, FullTestsParams);

TYPED_TEST(MismatchTests, TestMismatchSimple)
{
    using Vector = typename TestFixture::input_type;
    using T      = typename Vector::value_type;

    Vector a(4);
    Vector b(4);
    a[0] = T(1);
    b[0] = T(1);
    a[1] = T(2);
    b[1] = T(2);
    a[2] = T(3);
    b[2] = T(4);
    a[3] = T(4);
    b[3] = T(3);

    ASSERT_EQ(thrust::mismatch(a.begin(), a.end(), b.begin()).first - a.begin(), 2);
    ASSERT_EQ(thrust::mismatch(a.begin(), a.end(), b.begin()).second - b.begin(), 2);

    b[2] = T(3);

    ASSERT_EQ(thrust::mismatch(a.begin(), a.end(), b.begin()).first - a.begin(), 3);
    ASSERT_EQ(thrust::mismatch(a.begin(), a.end(), b.begin()).second - b.begin(), 3);

    b[3] = T(4);

    ASSERT_EQ(thrust::mismatch(a.begin(), a.end(), b.begin()).first - a.begin(), 4);
    ASSERT_EQ(thrust::mismatch(a.begin(), a.end(), b.begin()).second - b.begin(), 4);
}

template <typename InputIterator1, typename InputIterator2>
thrust::pair<InputIterator1, InputIterator2>
    mismatch(my_system& system, InputIterator1 first, InputIterator1, InputIterator2)
{
    system.validate_dispatch();
    return thrust::make_pair(first, first);
}

TEST(MismatchTests, TestMismatchDispatchExplicit)
{
    thrust::device_vector<int> vec(1);

    my_system sys(0);
    thrust::mismatch(sys, vec.begin(), vec.begin(), vec.begin());

    ASSERT_EQ(true, sys.is_valid());
}

template <typename InputIterator1, typename InputIterator2>
thrust::pair<InputIterator1, InputIterator2>
    mismatch(my_tag, InputIterator1 first, InputIterator1, InputIterator2)
{
    *first = 13;
    return thrust::make_pair(first, first);
}

TEST(MismatchTests, TestMismatchDispatchImplicit)
{
    thrust::device_vector<int> vec(1);

    thrust::mismatch(thrust::retag<my_tag>(vec.begin()),
                     thrust::retag<my_tag>(vec.begin()),
                     thrust::retag<my_tag>(vec.begin()));

    ASSERT_EQ(13, vec.front());
}
