/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *  Modifications Copyright© 2019-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include <thrust/advance.h>
#include <thrust/sequence.h>

#include "test_header.hpp"

TESTS_DEFINE(AdvanceVectorTests, VectorSignedIntegerTestsParams);

TYPED_TEST(AdvanceVectorTests, TestAdvance)
{
    using Vector = typename TestFixture::input_type;
    using T      = typename Vector::value_type;

    using Iterator = typename Vector::iterator;

    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    Vector v(100);
    thrust::sequence(v.begin(), v.end());

    Iterator i = v.begin();

    thrust::advance(i, 7);

    ASSERT_EQ(*i, T(7));

    thrust::advance(i, 13);

    ASSERT_EQ(*i, T(20));

    thrust::advance(i, -10);

    ASSERT_EQ(*i, T(10));
}

TYPED_TEST(AdvanceVectorTests, TestNext)
{
    using Vector = typename TestFixture::input_type;
    using T      = typename Vector::value_type;

    using Iterator = typename Vector::iterator;

    Vector v(10);
    thrust::sequence(v.begin(), v.end());

    Iterator const i0 = v.begin();

    Iterator const i1 = thrust::next(i0);

    ASSERT_EQ(*i0, T(0));
    ASSERT_EQ(*i1, T(1));

    Iterator const i2 = thrust::next(i1, 8);

    ASSERT_EQ(*i0, T(0));
    ASSERT_EQ(*i1, T(1));
    ASSERT_EQ(*i2, T(9));

    Iterator const i3 = thrust::next(i2, -4);

    ASSERT_EQ(*i0, T(0));
    ASSERT_EQ(*i1, T(1));
    ASSERT_EQ(*i2, T(9));
    ASSERT_EQ(*i3, T(5));
}

TYPED_TEST(AdvanceVectorTests, TestPrev)
{
    using Vector = typename TestFixture::input_type;
    using T      = typename Vector::value_type;

    using Iterator = typename Vector::iterator;

    Vector v(10);
    thrust::sequence(v.begin(), v.end());

    Iterator const i0 = v.end();

    Iterator const i1 = thrust::prev(i0);

    ASSERT_EQ_QUIET(i0, v.end());
    ASSERT_EQ(*i1, T(9));

    Iterator const i2 = thrust::prev(i1, 8);

    ASSERT_EQ_QUIET(i0, v.end());
    ASSERT_EQ(*i1, T(9));
    ASSERT_EQ(*i2, T(1));

    Iterator const i3 = thrust::prev(i2, -4);

    ASSERT_EQ_QUIET(i0, v.end());
    ASSERT_EQ(*i1, T(9));
    ASSERT_EQ(*i2, T(1));
    ASSERT_EQ(*i3, T(5));
}
