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
#include <thrust/distance.h>
#include <thrust/host_vector.h>

#include "test_header.hpp"

TESTS_DEFINE(DistanceTests, FullTestsParams);

TYPED_TEST(DistanceTests, TestDistance)
{
    using Vector   = typename TestFixture::input_type;
    using Iterator = typename Vector::iterator;

    Vector v(100);

    Iterator i = v.begin();

    ASSERT_EQ(thrust::distance(i, v.end()), 100);

    i++;

    ASSERT_EQ(thrust::distance(i, v.end()), 99);

    i += 49;

    ASSERT_EQ(thrust::distance(i, v.end()), 50);

    ASSERT_EQ(thrust::distance(i, i), 0);
}

TYPED_TEST(DistanceTests, TestDistanceLarge)
{
    using Vector   = typename TestFixture::input_type;
    using Iterator = typename Vector::iterator;

    Vector v(1000);

    Iterator i = v.begin();

    ASSERT_EQ(thrust::distance(i, v.end()), 1000);

    i++;

    ASSERT_EQ(thrust::distance(i, v.end()), 999);

    i += 49;

    ASSERT_EQ(thrust::distance(i, v.end()), 950);

    i += 950;

    ASSERT_EQ(thrust::distance(i, v.end()), 0);

    ASSERT_EQ(thrust::distance(i, i), 0);
}
