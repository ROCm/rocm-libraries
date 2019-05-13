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

#include <thrust/advance.h>
#include <thrust/sequence.h>

#include "test_header.hpp"

TESTS_DEFINE(AdvanceVectorTests, VectorSignedIntegerTestsParams);

// TODO expand this with other iterator types (forward, bidirectional, etc.)

TYPED_TEST(AdvanceVectorTests, TestAdvance)
{
    using Vector = typename TestFixture::input_type;
    using T      = typename Vector::value_type;

    typedef typename Vector::iterator Iterator;

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
