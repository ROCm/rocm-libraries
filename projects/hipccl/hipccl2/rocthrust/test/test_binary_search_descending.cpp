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

#include <thrust/binary_search.h>
#include <thrust/functional.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include "test_header.hpp"

TESTS_DEFINE(BinarySearchDescendingTests, FullTestsParams);

TYPED_TEST(BinarySearchDescendingTests, TestScalarLowerBoundDescendingSimple)
{
    using Vector = typename TestFixture::input_type;
    using T      = typename Vector::value_type;
    Vector vec(5);

    vec[0] = 8;
    vec[1] = 7;
    vec[2] = 5;
    vec[3] = 2;
    vec[4] = 0;

    ASSERT_EQ_QUIET(vec.begin() + 4,
                    thrust::lower_bound(vec.begin(), vec.end(), 0, thrust::greater<T>()));
    ASSERT_EQ_QUIET(vec.begin() + 4,
                    thrust::lower_bound(vec.begin(), vec.end(), 1, thrust::greater<T>()));
    ASSERT_EQ_QUIET(vec.begin() + 3,
                    thrust::lower_bound(vec.begin(), vec.end(), 2, thrust::greater<T>()));
    ASSERT_EQ_QUIET(vec.begin() + 3,
                    thrust::lower_bound(vec.begin(), vec.end(), 3, thrust::greater<T>()));
    ASSERT_EQ_QUIET(vec.begin() + 3,
                    thrust::lower_bound(vec.begin(), vec.end(), 4, thrust::greater<T>()));
    ASSERT_EQ_QUIET(vec.begin() + 2,
                    thrust::lower_bound(vec.begin(), vec.end(), 5, thrust::greater<T>()));
    ASSERT_EQ_QUIET(vec.begin() + 2,
                    thrust::lower_bound(vec.begin(), vec.end(), 6, thrust::greater<T>()));
    ASSERT_EQ_QUIET(vec.begin() + 1,
                    thrust::lower_bound(vec.begin(), vec.end(), 7, thrust::greater<T>()));
    ASSERT_EQ_QUIET(vec.begin() + 0,
                    thrust::lower_bound(vec.begin(), vec.end(), 8, thrust::greater<T>()));
    ASSERT_EQ_QUIET(vec.begin() + 0,
                    thrust::lower_bound(vec.begin(), vec.end(), 9, thrust::greater<T>()));
}

TYPED_TEST(BinarySearchDescendingTests, TestScalarUpperBoundDescendingSimple)
{
    using Vector = typename TestFixture::input_type;
    Vector vec(5);

    vec[0] = 8;
    vec[1] = 7;
    vec[2] = 5;
    vec[3] = 2;
    vec[4] = 0;

    ASSERT_EQ_QUIET(vec.begin() + 5,
                    thrust::upper_bound(vec.begin(), vec.end(), 0, thrust::greater<int>()));
    ASSERT_EQ_QUIET(vec.begin() + 4,
                    thrust::upper_bound(vec.begin(), vec.end(), 1, thrust::greater<int>()));
    ASSERT_EQ_QUIET(vec.begin() + 4,
                    thrust::upper_bound(vec.begin(), vec.end(), 2, thrust::greater<int>()));
    ASSERT_EQ_QUIET(vec.begin() + 3,
                    thrust::upper_bound(vec.begin(), vec.end(), 3, thrust::greater<int>()));
    ASSERT_EQ_QUIET(vec.begin() + 3,
                    thrust::upper_bound(vec.begin(), vec.end(), 4, thrust::greater<int>()));
    ASSERT_EQ_QUIET(vec.begin() + 3,
                    thrust::upper_bound(vec.begin(), vec.end(), 5, thrust::greater<int>()));
    ASSERT_EQ_QUIET(vec.begin() + 2,
                    thrust::upper_bound(vec.begin(), vec.end(), 6, thrust::greater<int>()));
    ASSERT_EQ_QUIET(vec.begin() + 2,
                    thrust::upper_bound(vec.begin(), vec.end(), 7, thrust::greater<int>()));
    ASSERT_EQ_QUIET(vec.begin() + 1,
                    thrust::upper_bound(vec.begin(), vec.end(), 8, thrust::greater<int>()));
    ASSERT_EQ_QUIET(vec.begin() + 0,
                    thrust::upper_bound(vec.begin(), vec.end(), 9, thrust::greater<int>()));
}

TYPED_TEST(BinarySearchDescendingTests, TestScalarBinarySearchDescendingSimple)
{
    using Vector = typename TestFixture::input_type;
    Vector vec(5);

    vec[0] = 8;
    vec[1] = 7;
    vec[2] = 5;
    vec[3] = 2;
    vec[4] = 0;

    ASSERT_EQ(true, thrust::binary_search(vec.begin(), vec.end(), 0, thrust::greater<int>()));
    ASSERT_EQ(false, thrust::binary_search(vec.begin(), vec.end(), 1, thrust::greater<int>()));
    ASSERT_EQ(true, thrust::binary_search(vec.begin(), vec.end(), 2, thrust::greater<int>()));
    ASSERT_EQ(false, thrust::binary_search(vec.begin(), vec.end(), 3, thrust::greater<int>()));
    ASSERT_EQ(false, thrust::binary_search(vec.begin(), vec.end(), 4, thrust::greater<int>()));
    ASSERT_EQ(true, thrust::binary_search(vec.begin(), vec.end(), 5, thrust::greater<int>()));
    ASSERT_EQ(false, thrust::binary_search(vec.begin(), vec.end(), 6, thrust::greater<int>()));
    ASSERT_EQ(true, thrust::binary_search(vec.begin(), vec.end(), 7, thrust::greater<int>()));
    ASSERT_EQ(true, thrust::binary_search(vec.begin(), vec.end(), 8, thrust::greater<int>()));
    ASSERT_EQ(false, thrust::binary_search(vec.begin(), vec.end(), 9, thrust::greater<int>()));
}

TYPED_TEST(BinarySearchDescendingTests, TestScalarEqualRangeDescendingSimple)
{
    using Vector = typename TestFixture::input_type;
    Vector vec(5);

    vec[0] = 8;
    vec[1] = 7;
    vec[2] = 5;
    vec[3] = 2;
    vec[4] = 0;

    ASSERT_EQ_QUIET(vec.begin() + 4,
                    thrust::equal_range(vec.begin(), vec.end(), 0, thrust::greater<int>()).first);
    ASSERT_EQ_QUIET(vec.begin() + 4,
                    thrust::equal_range(vec.begin(), vec.end(), 1, thrust::greater<int>()).first);
    ASSERT_EQ_QUIET(vec.begin() + 3,
                    thrust::equal_range(vec.begin(), vec.end(), 2, thrust::greater<int>()).first);
    ASSERT_EQ_QUIET(vec.begin() + 3,
                    thrust::equal_range(vec.begin(), vec.end(), 3, thrust::greater<int>()).first);
    ASSERT_EQ_QUIET(vec.begin() + 3,
                    thrust::equal_range(vec.begin(), vec.end(), 4, thrust::greater<int>()).first);
    ASSERT_EQ_QUIET(vec.begin() + 2,
                    thrust::equal_range(vec.begin(), vec.end(), 5, thrust::greater<int>()).first);
    ASSERT_EQ_QUIET(vec.begin() + 2,
                    thrust::equal_range(vec.begin(), vec.end(), 6, thrust::greater<int>()).first);
    ASSERT_EQ_QUIET(vec.begin() + 1,
                    thrust::equal_range(vec.begin(), vec.end(), 7, thrust::greater<int>()).first);
    ASSERT_EQ_QUIET(vec.begin() + 0,
                    thrust::equal_range(vec.begin(), vec.end(), 8, thrust::greater<int>()).first);
    ASSERT_EQ_QUIET(vec.begin() + 0,
                    thrust::equal_range(vec.begin(), vec.end(), 9, thrust::greater<int>()).first);

    ASSERT_EQ_QUIET(vec.begin() + 5,
                    thrust::equal_range(vec.begin(), vec.end(), 0, thrust::greater<int>()).second);
    ASSERT_EQ_QUIET(vec.begin() + 4,
                    thrust::equal_range(vec.begin(), vec.end(), 1, thrust::greater<int>()).second);
    ASSERT_EQ_QUIET(vec.begin() + 4,
                    thrust::equal_range(vec.begin(), vec.end(), 2, thrust::greater<int>()).second);
    ASSERT_EQ_QUIET(vec.begin() + 3,
                    thrust::equal_range(vec.begin(), vec.end(), 3, thrust::greater<int>()).second);
    ASSERT_EQ_QUIET(vec.begin() + 3,
                    thrust::equal_range(vec.begin(), vec.end(), 4, thrust::greater<int>()).second);
    ASSERT_EQ_QUIET(vec.begin() + 3,
                    thrust::equal_range(vec.begin(), vec.end(), 5, thrust::greater<int>()).second);
    ASSERT_EQ_QUIET(vec.begin() + 2,
                    thrust::equal_range(vec.begin(), vec.end(), 6, thrust::greater<int>()).second);
    ASSERT_EQ_QUIET(vec.begin() + 2,
                    thrust::equal_range(vec.begin(), vec.end(), 7, thrust::greater<int>()).second);
    ASSERT_EQ_QUIET(vec.begin() + 1,
                    thrust::equal_range(vec.begin(), vec.end(), 8, thrust::greater<int>()).second);
    ASSERT_EQ_QUIET(vec.begin() + 0,
                    thrust::equal_range(vec.begin(), vec.end(), 9, thrust::greater<int>()).second);
}
