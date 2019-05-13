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

#include <thrust/iterator/retag.h>
#include <thrust/sort.h>

#include "test_header.hpp"

TESTS_DEFINE(IsSortedUntilTests, FullTestsParams);
TESTS_DEFINE(IsSortedUntilVectorTests, VectorSignedIntegerTestsParams);

TYPED_TEST(IsSortedUntilVectorTests, TestIsSortedUntilSimple)
{
    using Vector = typename TestFixture::input_type;
    using T      = typename Vector::value_type;

    typedef typename Vector::iterator Iterator;

    Vector v(4);
    v[0] = 0;
    v[1] = 5;
    v[2] = 8;
    v[3] = 0;

    Iterator first = v.begin();

    Iterator last = v.begin() + 0;
    Iterator ref  = last;
    ASSERT_EQ_QUIET(ref, thrust::is_sorted_until(first, last));

    last = v.begin() + 1;
    ref  = last;
    ASSERT_EQ_QUIET(ref, thrust::is_sorted_until(first, last));

    last = v.begin() + 2;
    ref  = last;
    ASSERT_EQ_QUIET(ref, thrust::is_sorted_until(first, last));

    last = v.begin() + 3;
    ref  = v.begin() + 3;
    ASSERT_EQ_QUIET(ref, thrust::is_sorted_until(first, last));

    last = v.begin() + 4;
    ref  = v.begin() + 3;
    ASSERT_EQ_QUIET(ref, thrust::is_sorted_until(first, last));

    last = v.begin() + 3;
    ref  = v.begin() + 3;
    ASSERT_EQ_QUIET(ref, thrust::is_sorted_until(first, last, thrust::less<T>()));

    last = v.begin() + 4;
    ref  = v.begin() + 3;
    ASSERT_EQ_QUIET(ref, thrust::is_sorted_until(first, last, thrust::less<T>()));

    last = v.begin() + 1;
    ref  = v.begin() + 1;
    ASSERT_EQ_QUIET(ref, thrust::is_sorted_until(first, last, thrust::greater<T>()));

    last = v.begin() + 4;
    ref  = v.begin() + 1;
    ASSERT_EQ_QUIET(ref, thrust::is_sorted_until(first, last, thrust::greater<T>()));

    first = v.begin() + 2;
    last  = v.begin() + 4;
    ref   = v.begin() + 4;
    ASSERT_EQ_QUIET(ref, thrust::is_sorted_until(first, last, thrust::greater<T>()));
}

TYPED_TEST(IsSortedUntilVectorTests, TestIsSortedUntilRepeatedElements)
{
    using Vector = typename TestFixture::input_type;
    Vector v(10);

    v[0] = 0;
    v[1] = 1;
    v[2] = 1;
    v[3] = 2;
    v[4] = 3;
    v[5] = 4;
    v[6] = 5;
    v[7] = 5;
    v[8] = 5;
    v[9] = 6;

    ASSERT_EQ_QUIET(v.end(), thrust::is_sorted_until(v.begin(), v.end()));
}

TYPED_TEST(IsSortedUntilVectorTests, TestIsSortedUntil)
{
    using Vector = typename TestFixture::input_type;
    using T      = typename Vector::value_type;

    const size_t n = (1 << 16) + 13;

    Vector v = get_random_data<T>(n, std::numeric_limits<T>::min(), std::numeric_limits<T>::max());

    v[0] = 1;
    v[1] = 0;

    ASSERT_EQ_QUIET(v.begin() + 1, thrust::is_sorted_until(v.begin(), v.end()));

    thrust::sort(v.begin(), v.end());

    ASSERT_EQ_QUIET(v.end(), thrust::is_sorted_until(v.begin(), v.end()));
}

template <typename ForwardIterator>
ForwardIterator is_sorted_until(my_system& system, ForwardIterator first, ForwardIterator)
{
    system.validate_dispatch();
    return first;
}

TEST(IsSortedUntilTests, TestIsSortedUntilExplicit)
{
    thrust::device_vector<int> vec(1);

    my_system sys(0);
    thrust::is_sorted_until(sys, vec.begin(), vec.end());

    ASSERT_EQ(true, sys.is_valid());
}

template <typename ForwardIterator>
ForwardIterator is_sorted_until(my_tag, ForwardIterator first, ForwardIterator)
{
    *first = 13;
    return first;
}

TEST(IsSortedUntilTests, TestIsSortedUntilImplicit)
{
    thrust::device_vector<int> vec(1);

    thrust::is_sorted_until(thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.end()));

    ASSERT_EQ(13, vec.front());
}
