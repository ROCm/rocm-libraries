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

#include <thrust/functional.h>
#include <thrust/iterator/retag.h>
#include <thrust/partition.h>

#include "test_header.hpp"

TESTS_DEFINE(IsPartitionedTests, FullTestsParams);

TESTS_DEFINE(IsPartitionedVectorTests, VectorSignedIntegerTestsParams);

template <typename T>
struct is_even
{
    __host__ __device__ bool operator()(T x) const
    {
        return ((int)x % 2) == 0;
    }
};

TYPED_TEST(IsPartitionedVectorTests, TestIsPartitionedSimple)
{
    using Vector = typename TestFixture::input_type;
    using T      = typename Vector::value_type;

    Vector v(4);
    v[0] = 1;
    v[1] = 1;
    v[2] = 1;
    v[3] = 0;

    // empty partition
    ASSERT_EQ_QUIET(true, thrust::is_partitioned(v.begin(), v.begin(), thrust::identity<T>()));

    // one element true partition
    ASSERT_EQ_QUIET(true, thrust::is_partitioned(v.begin(), v.begin() + 1, thrust::identity<T>()));

    // just true partition
    ASSERT_EQ_QUIET(true, thrust::is_partitioned(v.begin(), v.begin() + 2, thrust::identity<T>()));

    // both true & false partitions
    ASSERT_EQ_QUIET(true, thrust::is_partitioned(v.begin(), v.end(), thrust::identity<T>()));

    // one element false partition
    ASSERT_EQ_QUIET(true, thrust::is_partitioned(v.begin() + 3, v.end(), thrust::identity<T>()));

    v[0] = 1;
    v[1] = 0;
    v[2] = 1;
    v[3] = 1;

    // not partitioned
    ASSERT_EQ_QUIET(false, thrust::is_partitioned(v.begin(), v.end(), thrust::identity<T>()));
}

TYPED_TEST(IsPartitionedVectorTests, TestIsPartitioned)
{
    using Vector = typename TestFixture::input_type;
    using T      = typename Vector::value_type;

    const size_t n = (1 << 16) + 13;

    Vector v = get_random_data<T>(n, std::numeric_limits<T>::min(), std::numeric_limits<T>::max());

    v[0] = 1;
    v[1] = 0;

    ASSERT_EQ(false, thrust::is_partitioned(v.begin(), v.end(), is_even<T>()));

    thrust::partition(v.begin(), v.end(), is_even<T>());

    ASSERT_EQ(true, thrust::is_partitioned(v.begin(), v.end(), is_even<T>()));
}

template <typename InputIterator, typename Predicate>
__host__ __device__ bool is_partitioned(my_system& system, InputIterator, InputIterator, Predicate)
{
    system.validate_dispatch();
    return false;
}

TEST(IsPartitionedTests, TestIsPartitionedDispatchExplicit)
{
    thrust::device_vector<int> vec(1);

    my_system sys(0);
    thrust::is_partitioned(sys, vec.begin(), vec.end(), 0);

    ASSERT_EQ(true, sys.is_valid());
}

template <typename InputIterator, typename Predicate>
__host__ __device__ bool is_partitioned(my_tag, InputIterator first, InputIterator, Predicate)
{
    *first = 13;
    return false;
}

TEST(IsPartitionedTests, TestIsPartitionedDispatchImplicit)
{
    thrust::device_vector<int> vec(1);

    thrust::is_partitioned(thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.end()), 0);

    ASSERT_EQ(13, vec.front());
}