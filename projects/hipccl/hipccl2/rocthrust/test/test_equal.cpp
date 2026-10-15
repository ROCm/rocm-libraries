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

#include <thrust/device_free.h>
#include <thrust/device_malloc.h>
#include <thrust/equal.h>
#include <thrust/functional.h>
#include <thrust/iterator/retag.h>

#include "test_param_fixtures.hpp"
#include "test_utils.hpp"

TESTS_DEFINE(EqualTests, FullTestsParams);
TESTS_DEFINE(EqualsPrimitiveTests, NumericalTestsParams);

TYPED_TEST(EqualTests, TestEqualSimple)
{
  using Vector = typename TestFixture::input_type;
  using T      = typename Vector::value_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  Vector v1{5, 2, 0, 0, 0};
  Vector v2{5, 2, 0, 6, 1};

  ASSERT_EQ(thrust::equal(v1.begin(), v1.end(), v1.begin()), true);
  ASSERT_EQ(thrust::equal(v1.begin(), v1.end(), v2.begin()), false);
  ASSERT_EQ(thrust::equal(v2.begin(), v2.end(), v2.begin()), true);

  ASSERT_EQ(thrust::equal(v1.begin(), v1.begin() + 0, v1.begin()), true);
  ASSERT_EQ(thrust::equal(v1.begin(), v1.begin() + 1, v1.begin()), true);
  ASSERT_EQ(thrust::equal(v1.begin(), v1.begin() + 3, v2.begin()), true);
  ASSERT_EQ(thrust::equal(v1.begin(), v1.begin() + 4, v2.begin()), false);

  ASSERT_EQ(thrust::equal(v1.begin(), v1.end(), v2.begin(), thrust::less_equal<T>()), true);
  ASSERT_EQ(thrust::equal(v1.begin(), v1.end(), v2.begin(), thrust::greater<T>()), false);
}

TYPED_TEST(EqualsPrimitiveTests, TestEqual)
{
  using T = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  for (auto size : get_sizes())
  {
    SCOPED_TRACE(testing::Message() << "with size= " << size);

    for (auto seed : get_seeds())
    {
      SCOPED_TRACE(testing::Message() << "with seed= " << seed);

      thrust::host_vector<T> h_data1 =
        get_random_data<T>(size, get_default_limits<T>::min(), get_default_limits<T>::max(), seed);
      thrust::host_vector<T> h_data2 = get_random_data<T>(
        size, get_default_limits<T>::min(), get_default_limits<T>::max(), seed + seed_value_addition);
      thrust::device_vector<T> d_data1 = h_data1;
      thrust::device_vector<T> d_data2 = h_data2;

      // empty ranges
      ASSERT_EQ(thrust::equal(h_data1.begin(), h_data1.begin(), h_data1.begin()), true);
      ASSERT_EQ(thrust::equal(d_data1.begin(), d_data1.begin(), d_data1.begin()), true);

      // symmetric cases
      ASSERT_EQ(thrust::equal(h_data1.begin(), h_data1.end(), h_data1.begin()), true);
      ASSERT_EQ(thrust::equal(d_data1.begin(), d_data1.end(), d_data1.begin()), true);

      if (size > 0)
      {
        h_data1[0] = 0;
        h_data2[0] = 1;
        d_data1[0] = 0;
        d_data2[0] = 1;

        // different vectors
        ASSERT_EQ(thrust::equal(h_data1.begin(), h_data1.end(), h_data2.begin()), false);
        ASSERT_EQ(thrust::equal(d_data1.begin(), d_data1.end(), d_data2.begin()), false);

        // different predicates
        ASSERT_EQ(thrust::equal(h_data1.begin(), h_data1.begin() + 1, h_data2.begin(), thrust::less<T>()), true);
        ASSERT_EQ(thrust::equal(d_data1.begin(), d_data1.begin() + 1, d_data2.begin(), thrust::less<T>()), true);
        ASSERT_EQ(thrust::equal(h_data1.begin(), h_data1.begin() + 1, h_data2.begin(), thrust::greater<T>()), false);
        ASSERT_EQ(thrust::equal(d_data1.begin(), d_data1.begin() + 1, d_data2.begin(), thrust::greater<T>()), false);
      }
    }
  }
}

template <typename InputIterator1, typename InputIterator2>
bool equal(my_system& system, InputIterator1 /*first*/, InputIterator1, InputIterator2)
{
  system.validate_dispatch();
  return false;
}

TEST(EqualTests, TestEqualDispatchExplicit)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::device_vector<int> vec(1);

  my_system sys(0);
  thrust::equal(sys, vec.begin(), vec.end(), vec.begin());

  ASSERT_EQ(true, sys.is_valid());
}

template <typename InputIterator1, typename InputIterator2>
bool equal(my_tag, InputIterator1 first, InputIterator1, InputIterator2)
{
  *first = 13;
  return false;
}

TEST(EqualTests, TestEqualDispatchImplicit)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::device_vector<int> vec(1);

  thrust::equal(
    thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.end()), thrust::retag<my_tag>(vec.begin()));

  ASSERT_EQ(13, vec.front());
}

struct only_set_when_both_expected
{
  long long expected;
  bool* flag;

  THRUST_DEVICE bool operator()(long long x, long long y)
  {
    if (x == expected && y == expected)
    {
      *flag = true;
    }

    return x == y;
  }
};

void TestEqualWithBigIndexesHelper(int magnitude)
{
  thrust::counting_iterator<long long> begin(1);
  thrust::counting_iterator<long long> end = begin + (1ll << magnitude);
  ASSERT_EQ(thrust::distance(begin, end), 1ll << magnitude);

  thrust::device_ptr<bool> has_executed = thrust::device_malloc<bool>(1);
  *has_executed                         = false;

  only_set_when_both_expected fn = {(1ll << magnitude) - 1, thrust::raw_pointer_cast(has_executed)};

  ASSERT_EQ(thrust::equal(thrust::device, begin, end, begin, fn), true);

  bool has_executed_h = *has_executed;
  thrust::device_free(has_executed);

  ASSERT_EQ(has_executed_h, true);
}

TEST(EqualTests, TestEqualWithBigIndexes)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  TestEqualWithBigIndexesHelper(30);
  TestEqualWithBigIndexesHelper(31);
  TestEqualWithBigIndexesHelper(32);
  TestEqualWithBigIndexesHelper(33);
}
