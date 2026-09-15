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

#include <thrust/count.h>
#include <thrust/iterator/retag.h>
#include <thrust/universal_vector.h>

#include "test_param_fixtures.hpp"
#include "test_utils.hpp"

using VectorTestsParams = ::testing::Types<
  Params<thrust::host_vector<signed char>>,
  Params<thrust::host_vector<short>>,
  Params<thrust::host_vector<int>>,
  Params<thrust::host_vector<float>>,
  Params<thrust::host_vector<int, thrust::mr::stateless_resource_allocator<int, thrust::host_memory_resource>>>,
  Params<thrust::device_vector<signed char>>,
  Params<thrust::device_vector<short>>,
  Params<thrust::device_vector<int>>,
  Params<thrust::device_vector<float>>,
  Params<thrust::device_vector<int, thrust::mr::stateless_resource_allocator<int, thrust::device_memory_resource>>>,
  Params<thrust::universal_vector<int>>,
  Params<thrust::universal_host_pinned_vector<int>>>;

TESTS_DEFINE(CountTests, FullTestsParams);
TESTS_DEFINE(CountPrimitiveTests, NumericalTestsParams);
TESTS_DEFINE(CountVectorTests, VectorTestsParams);

TYPED_TEST(CountVectorTests, TestCountSimple)
{
  using Vector = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  Vector data{1, 1, 0, 0, 1};

  ASSERT_EQ(thrust::count(data.begin(), data.end(), 0), 2);
  ASSERT_EQ(thrust::count(data.begin(), data.end(), 1), 3);
  ASSERT_EQ(thrust::count(data.begin(), data.end(), 2), 0);
}

TYPED_TEST(CountPrimitiveTests, TestCount)
{
  using T = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  for (auto size : get_sizes())
  {
    SCOPED_TRACE(testing::Message() << "with size= " << size);

    for (auto seed : get_seeds())
    {
      SCOPED_TRACE(testing::Message() << "with seed= " << seed);

      thrust::host_vector<T> h_data =
        get_random_data<T>(size, get_default_limits<T>::min(), get_default_limits<T>::max(), seed);
      thrust::device_vector<T> d_data = h_data;

      size_t cpu_result = thrust::count(h_data.begin(), h_data.end(), T(5));
      size_t gpu_result = thrust::count(d_data.begin(), d_data.end(), T(5));

      ASSERT_EQ(cpu_result, gpu_result);
    }
  }
}

template <typename T>
struct greater_than_five
{
  THRUST_HOST_DEVICE bool operator()(const T& x) const
  {
    return x > 5;
  }
};

TYPED_TEST(CountTests, TestCountIfSimple)
{
  using Vector = typename TestFixture::input_type;
  using T      = typename Vector::value_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  Vector data{1, 6, 1, 9, 2};

  ASSERT_EQ(thrust::count_if(data.begin(), data.end(), greater_than_five<T>()), 2);
}

TYPED_TEST(CountTests, TestCountIf)
{
  using Vector = typename TestFixture::input_type;
  using T      = typename Vector::value_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  for (auto size : get_sizes())
  {
    SCOPED_TRACE(testing::Message() << "with size= " << size);

    for (auto seed : get_seeds())
    {
      SCOPED_TRACE(testing::Message() << "with seed= " << seed);

      thrust::host_vector<T> h_data =
        get_random_data<T>(size, get_default_limits<T>::min(), get_default_limits<T>::max(), seed);
      thrust::device_vector<T> d_data = h_data;

      size_t cpu_result = thrust::count_if(h_data.begin(), h_data.end(), greater_than_five<T>());
      size_t gpu_result = thrust::count_if(d_data.begin(), d_data.end(), greater_than_five<T>());

      ASSERT_EQ(cpu_result, gpu_result);
    }
  }
}

TYPED_TEST(CountVectorTests, TestCountFromConstIteratorSimple)
{
  using Vector = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  Vector data{1, 1, 0, 0, 1};

  ASSERT_EQ(thrust::count(data.cbegin(), data.cend(), 0), 2);
  ASSERT_EQ(thrust::count(data.cbegin(), data.cend(), 1), 3);
  ASSERT_EQ(thrust::count(data.cbegin(), data.cend(), 2), 0);
}

template <typename InputIterator, typename EqualityComparable>
int count(my_system& system, InputIterator, InputIterator, EqualityComparable x)
{
  system.validate_dispatch();
  return x;
}

TEST(CountTests, TestCountDispatchExplicit)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::device_vector<int> vec(1);

  my_system sys(0);
  thrust::count(sys, vec.begin(), vec.end(), 13);

  ASSERT_EQ(true, sys.is_valid());
}

template <typename InputIterator, typename EqualityComparable>
int count(my_tag, InputIterator /*first*/, InputIterator, EqualityComparable x)
{
  return x;
}

TEST(CountTests, TestCountDispatchImplicit)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::device_vector<int> vec(1);

  auto result = thrust::count(thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.end()), 13);

  ASSERT_EQ(13, result);
}

void TestCountWithBigIndexesHelper(int magnitude)
{
  thrust::counting_iterator<long long> begin(1);
  thrust::counting_iterator<long long> end = begin + (1ll << magnitude);
  ASSERT_EQ(thrust::distance(begin, end), 1ll << magnitude);

  long long result = thrust::count(thrust::device, begin, end, (1ll << magnitude) - 17);

  ASSERT_EQ(result, 1);
}

TEST(CountTests, TestCountWithBigIndexes)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  TestCountWithBigIndexesHelper(30);
#ifndef THRUST_FORCE_32_BIT_OFFSET_TYPE
  TestCountWithBigIndexesHelper(31);
  TestCountWithBigIndexesHelper(32);
  TestCountWithBigIndexesHelper(33);
#endif
}
