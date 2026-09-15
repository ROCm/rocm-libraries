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
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/inner_product.h>
#include <thrust/iterator/retag.h>
#include <thrust/universal_vector.h>

#include <cmath>

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

using IntegralTypes = ::testing::Types<
  Params<char>,
  Params<signed char>,
  Params<unsigned char>,
  Params<short>,
  Params<unsigned short>,
  Params<int>,
  Params<unsigned int>,
  Params<long>,
  Params<unsigned long>,
  Params<long long>,
  Params<unsigned long long>>;

TESTS_DEFINE(InnerProductTests, FullTestsParams);
TESTS_DEFINE(PrimitiveInnerProductTests, NumericalTestsParams);
TESTS_DEFINE(VectorUnitInnerProductTests, VectorTestsParams);
TESTS_DEFINE(VariableUnitInnerProductTests, IntegralTypes);

TEST(InnerProductTests, UsingHip)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  ASSERT_EQ(THRUST_DEVICE_SYSTEM, THRUST_DEVICE_SYSTEM_HIP);
}

TYPED_TEST(VectorUnitInnerProductTests, TestInnerProductSimple)
{
  using Vector = typename TestFixture::input_type;
  using T      = typename Vector::value_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  Vector v1{1, -2, 3};
  Vector v2{-4, 5, 6};

  T init   = 3;
  T result = thrust::inner_product(v1.begin(), v1.end(), v2.begin(), init);
  ASSERT_EQ(result, 7);
}

template <typename InputIterator1, typename InputIterator2, typename OutputType>
int inner_product(my_system& system, InputIterator1, InputIterator1, InputIterator2, OutputType)
{
  system.validate_dispatch();
  return 13;
}

TEST(InnerProductTests, TestInnerProductDispatchExplicit)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::device_vector<int> vec;

  my_system sys(0);
  thrust::inner_product(sys, vec.begin(), vec.end(), vec.begin(), 0);

  ASSERT_EQ(true, sys.is_valid());
}

template <typename InputIterator1, typename InputIterator2, typename OutputType>
int inner_product(my_tag, InputIterator1, InputIterator1, InputIterator2, OutputType)
{
  return 13;
}

TEST(InnerProductTests, TestInnerProductDispatchImplicit)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::device_vector<int> vec;

  int result = thrust::inner_product(
    thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.end()), thrust::retag<my_tag>(vec.begin()), 0);

  ASSERT_EQ(13, result);
}

TYPED_TEST(VectorUnitInnerProductTests, TestInnerProductWithOperator)
{
  using Vector = typename TestFixture::input_type;
  using T      = typename Vector::value_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  Vector v1{1, -2, 3};
  Vector v2{-1, 3, 6};

  // compute (v1 - v2) and perform a multiplies reduction
  T init   = 3;
  T result = thrust::inner_product(v1.begin(), v1.end(), v2.begin(), init, thrust::multiplies<T>(), thrust::minus<T>());
  ASSERT_EQ(result, 90);
}

TYPED_TEST(VariableUnitInnerProductTests, TestInnerProduct)
{
  using T = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  for (auto size : get_sizes())
  {
    SCOPED_TRACE(testing::Message() << "with size= " << size);

    thrust::host_vector<T> h_v1 = random_integers<T>(size);
    thrust::host_vector<T> h_v2 = random_integers<T>(size);

    thrust::device_vector<T> d_v1 = h_v1;
    thrust::device_vector<T> d_v2 = h_v2;

    T init = 13;

    T expected = thrust::inner_product(h_v1.begin(), h_v1.end(), h_v2.begin(), init);
    T result   = thrust::inner_product(d_v1.begin(), d_v1.end(), d_v2.begin(), init);

    ASSERT_EQ(expected, result);
  }
}

struct only_set_when_both_expected
{
  long long expected;
  bool* flag;

  THRUST_HOST_DEVICE long long operator()(long long x, long long y)
  {
    if (x == expected && y == expected)
    {
      *flag = true;
    }

    return x == y;
  }
};

void TestInnerProductWithBigIndexesHelper(int magnitude)
{
  thrust::counting_iterator<long long> begin(1);
  thrust::counting_iterator<long long> end = begin + (1ll << magnitude);
  ASSERT_EQ(thrust::distance(begin, end), 1ll << magnitude);

  thrust::device_ptr<bool> has_executed = thrust::device_malloc<bool>(1);
  *has_executed                         = false;

  only_set_when_both_expected fn = {(1ll << magnitude) - 1, thrust::raw_pointer_cast(has_executed)};

  ASSERT_EQ(thrust::inner_product(thrust::device, begin, end, begin, 0ll, thrust::plus<long long>(), fn),
            (1ll << magnitude));

  bool has_executed_h = *has_executed;
  thrust::device_free(has_executed);

  ASSERT_EQ(has_executed_h, true);
}

TEST(InnerProductTests, TestInnerProductWithBigIndexes)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  TestInnerProductWithBigIndexesHelper(30);
#ifndef THRUST_FORCE_32_BIT_OFFSET_TYPE
  TestInnerProductWithBigIndexesHelper(31);
  TestInnerProductWithBigIndexesHelper(32);
  TestInnerProductWithBigIndexesHelper(33);
#endif
}

TEST(InnerProductTests, TestInnerProductPlaceholders)
{ // Regression test for NVIDIA/thrust#1178
  using namespace thrust::placeholders;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::device_vector<float> v1(100, 1.f);
  thrust::device_vector<float> v2(100, 1.f);

  auto result = thrust::inner_product(v1.begin(), v1.end(), v2.begin(), 0.0f, thrust::plus<float>{}, _1 * _2 + 1.0f);

  auto error_margin = 1e-4 * (::std::abs(result) + 200.f) * 1e-4;
  ASSERT_NEAR(static_cast<double>(result), static_cast<double>(200.f), static_cast<double>(error_margin));
}

template <class T>
T clip_infinity(T val)
{
  T min = std::numeric_limits<T>::min();
  T max = std::numeric_limits<T>::max();
  if (val > max)
  {
    return max;
  }
  if (val < min)
  {
    return min;
  }
  return val;
}

TYPED_TEST(PrimitiveInnerProductTests, InnerProductWithRandomData)
{
  using T = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  for (auto size : get_sizes())
  {
    SCOPED_TRACE(testing::Message() << "with size= " << size);

    T error_margin = static_cast<T>(0.01) * size;

    T min = saturate_cast<T>(-10);
    T max = saturate_cast<T>(10);

    for (auto seed : get_seeds())
    {
      SCOPED_TRACE(testing::Message() << "with seed= " << seed);

      thrust::host_vector<T> h_v1 = get_random_data<T>(size, min, max, seed);
      thrust::host_vector<T> h_v2 = get_random_data<T>(size, min, max, seed + seed_value_addition);

      thrust::device_vector<T> d_v1 = h_v1;
      thrust::device_vector<T> d_v2 = h_v2;

      T init = 13;

      T expected = thrust::inner_product(h_v1.begin(), h_v1.end(), h_v2.begin(), init);
      T result   = thrust::inner_product(d_v1.begin(), d_v1.end(), d_v2.begin(), init);

      ASSERT_NEAR(clip_infinity<T>(expected), clip_infinity<T>(result), error_margin);
    }
  }
};
