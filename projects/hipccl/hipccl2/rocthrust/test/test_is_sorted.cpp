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

TESTS_DEFINE(IsSortedTests, FullTestsParams);
TESTS_DEFINE(IsSortedVectorTests, VectorSignedIntegerTestsParams);

TYPED_TEST(IsSortedVectorTests, TestIsSortedSimple)
{
    using Vector = typename TestFixture::input_type;
    using T      = typename Vector::value_type;

    Vector v(4);
    v[0] = 0;
    v[1] = 5;
    v[2] = 8;
    v[3] = 0;

    ASSERT_EQ(thrust::is_sorted(v.begin(), v.begin() + 0), true);
    ASSERT_EQ(thrust::is_sorted(v.begin(), v.begin() + 1), true);

    // the following line crashes gcc 4.3
#if(__GNUC__ == 4) && (__GNUC_MINOR__ == 3)
    // do nothing
#else
    // compile this line on other compilers
    ASSERT_EQ(thrust::is_sorted(v.begin(), v.begin() + 2), true);
#endif // GCC

    ASSERT_EQ(thrust::is_sorted(v.begin(), v.begin() + 3), true);
    ASSERT_EQ(thrust::is_sorted(v.begin(), v.begin() + 4), false);

    ASSERT_EQ(thrust::is_sorted(v.begin(), v.begin() + 3, thrust::less<T>()), true);

    ASSERT_EQ(thrust::is_sorted(v.begin(), v.begin() + 1, thrust::greater<T>()), true);
    ASSERT_EQ(thrust::is_sorted(v.begin(), v.begin() + 4, thrust::greater<T>()), false);

    ASSERT_EQ(thrust::is_sorted(v.begin(), v.end()), false);
}

TYPED_TEST(IsSortedVectorTests, TestIsSortedRepeatedElements)
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

    ASSERT_EQ(true, thrust::is_sorted(v.begin(), v.end()));
}

TYPED_TEST(IsSortedVectorTests, TestIsSorted)
{
    using Vector = typename TestFixture::input_type;
    using T      = typename Vector::value_type;

    const size_t n = (1 << 16) + 13;

    Vector v = get_random_data<T>(n, std::numeric_limits<T>::min(), std::numeric_limits<T>::max());

    v[0] = 1;
    v[1] = 0;

    ASSERT_EQ(thrust::is_sorted(v.begin(), v.end()), false);

    thrust::sort(v.begin(), v.end());

    ASSERT_EQ(thrust::is_sorted(v.begin(), v.end()), true);
}

template <typename InputIterator>
bool is_sorted(my_system& system, InputIterator, InputIterator)
{
    system.validate_dispatch();
    return false;
}

TEST(IsSortedTests, TestIsSortedDispatchExplicit)
{
    thrust::device_vector<int> vec(1);

    my_system sys(0);
    thrust::is_sorted(sys, vec.begin(), vec.end());

    ASSERT_EQ(true, sys.is_valid());
}

template <typename InputIterator>
bool is_sorted(my_tag, InputIterator first, InputIterator)
{
    *first = 13;
    return false;
}

TEST(IsSortedTests, TestIsSortedDispatchImplicit)
{
    thrust::device_vector<int> vec(1);

    thrust::is_sorted(thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.end()));

    ASSERT_EQ(13, vec.front());
}
