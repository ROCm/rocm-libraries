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

#include <thrust/functional.h>
#include <thrust/iterator/retag.h>
#include <thrust/partition.h>

#include "test_param_fixtures.hpp"
#include "test_real_assertions.hpp"
#include "test_utils.hpp"

TESTS_DEFINE(IsPartitionedTests, FullTestsParams);

TESTS_DEFINE(IsPartitionedVectorTests, VectorSignedIntegerTestsParams);

template <typename T>
struct is_even
{
  THRUST_HOST_DEVICE bool operator()(T x) const
  {
    return ((int) x % 2) == 0;
  }
};

TYPED_TEST(IsPartitionedVectorTests, TestIsPartitionedSimple)
{
  using Vector = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  Vector v{1, 1, 1, 0};

  // empty partition
  ASSERT_EQ_QUIET(true, thrust::is_partitioned(v.begin(), v.begin(), ::internal::identity{}));

  // one element true partition
  ASSERT_EQ_QUIET(true, thrust::is_partitioned(v.begin(), v.begin() + 1, ::internal::identity{}));

  // just true partition
  ASSERT_EQ_QUIET(true, thrust::is_partitioned(v.begin(), v.begin() + 2, ::internal::identity{}));

  // both true & false partitions
  ASSERT_EQ_QUIET(true, thrust::is_partitioned(v.begin(), v.end(), ::internal::identity{}));

  // one element false partition
  ASSERT_EQ_QUIET(true, thrust::is_partitioned(v.begin() + 3, v.end(), ::internal::identity{}));

  v = {1, 0, 1, 1};

  // not partitioned
  ASSERT_EQ_QUIET(false, thrust::is_partitioned(v.begin(), v.end(), ::internal::identity{}));
}

TYPED_TEST(IsPartitionedVectorTests, TestIsPartitioned)
{
  using Vector = typename TestFixture::input_type;
  using T      = typename Vector::value_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  const size_t n = (1 << 16) + 13;

  for (auto seed : get_seeds())
  {
    SCOPED_TRACE(testing::Message() << "with seed= " << seed);

    Vector v = get_random_data<T>(n, get_default_limits<T>::min(), get_default_limits<T>::max(), seed);

    v[0] = 1;
    v[1] = 0;

    ASSERT_EQ(false, thrust::is_partitioned(v.begin(), v.end(), is_even<T>()));

    thrust::partition(v.begin(), v.end(), is_even<T>());

    ASSERT_EQ(true, thrust::is_partitioned(v.begin(), v.end(), is_even<T>()));
  }
}

template <typename InputIterator, typename Predicate>
bool is_partitioned(my_system& system, InputIterator /*first*/, InputIterator, Predicate)
{
  system.validate_dispatch();
  return false;
}

TEST(IsPartitionedTests, TestIsPartitionedDispatchExplicit)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::device_vector<int> vec(1);

  my_system sys(0);
  thrust::is_partitioned(sys, vec.begin(), vec.end(), 0);

  ASSERT_EQ(true, sys.is_valid());
}

template <typename InputIterator, typename Predicate>
bool is_partitioned(my_tag, InputIterator first, InputIterator, Predicate)
{
  *first = 13;
  return false;
}

TEST(IsPartitionedTests, TestIsPartitionedDispatchImplicit)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::device_vector<int> vec(1);

  thrust::is_partitioned(thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.end()), 0);

  ASSERT_EQ(13, vec.front());
}
