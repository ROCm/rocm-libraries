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

#include <thrust/find.h>
#include <thrust/functional.h>
#include <thrust/iterator/retag.h>
#include <thrust/sequence.h>

#include "test_param_fixtures.hpp"
#include "test_utils.hpp"

TESTS_DEFINE(FindTestsVector, FullTestsParams);
TESTS_DEFINE(FindTests, NumericalTestsParams);

template <typename T>
struct equal_to_value_pred
{
  T value;

  equal_to_value_pred(T value)
      : value(value)
  {}

  THRUST_HOST_DEVICE bool operator()(T v) const
  {
    return v == value;
  }
};

template <typename T>
struct not_equal_to_value_pred
{
  T value;

  not_equal_to_value_pred(T value)
      : value(value)
  {}

  THRUST_HOST_DEVICE bool operator()(T v) const
  {
    return v != value;
  }
};

template <typename T>
struct less_than_value_pred
{
  T value;

  less_than_value_pred(T value)
      : value(value)
  {}

  THRUST_HOST_DEVICE bool operator()(T v) const
  {
    return v < value;
  }
};

TYPED_TEST(FindTestsVector, TestFindSimple)
{
  using Vector = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  Vector vec{1, 2, 3, 3, 5};

  ASSERT_EQ(thrust::find(vec.begin(), vec.end(), 0) - vec.begin(), 5);
  ASSERT_EQ(thrust::find(vec.begin(), vec.end(), 1) - vec.begin(), 0);
  ASSERT_EQ(thrust::find(vec.begin(), vec.end(), 2) - vec.begin(), 1);
  ASSERT_EQ(thrust::find(vec.begin(), vec.end(), 3) - vec.begin(), 2);
  ASSERT_EQ(thrust::find(vec.begin(), vec.end(), 4) - vec.begin(), 5);
  ASSERT_EQ(thrust::find(vec.begin(), vec.end(), 5) - vec.begin(), 4);
}

template <typename InputIterator, typename T>
InputIterator find(my_system& system, InputIterator first, InputIterator, const T&)
{
  system.validate_dispatch();
  return first;
}

TEST(FindTests, TestFindDispatchExplicit)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::device_vector<int> vec(1);

  my_system sys(0);
  thrust::find(sys, vec.begin(), vec.end(), 0);

  ASSERT_EQ(true, sys.is_valid());
}

template <typename InputIterator, typename T>
InputIterator find(my_tag, InputIterator first, InputIterator, const T&)
{
  *first = 13;
  return first;
}

TEST(FindTests, TestFindDispatchImplicit)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::device_vector<int> vec(1);

  thrust::find(thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.end()), 0);

  ASSERT_EQ(13, vec.front());
}

TYPED_TEST(FindTestsVector, TestFindIfSimple)
{
  using Vector = typename TestFixture::input_type;
  using T      = typename Vector::value_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  Vector vec{1, 2, 3, 3, 5};

  ASSERT_EQ(thrust::find_if(vec.begin(), vec.end(), equal_to_value_pred<T>(0)) - vec.begin(), 5);
  ASSERT_EQ(thrust::find_if(vec.begin(), vec.end(), equal_to_value_pred<T>(1)) - vec.begin(), 0);
  ASSERT_EQ(thrust::find_if(vec.begin(), vec.end(), equal_to_value_pred<T>(2)) - vec.begin(), 1);
  ASSERT_EQ(thrust::find_if(vec.begin(), vec.end(), equal_to_value_pred<T>(3)) - vec.begin(), 2);
  ASSERT_EQ(thrust::find_if(vec.begin(), vec.end(), equal_to_value_pred<T>(4)) - vec.begin(), 5);
  ASSERT_EQ(thrust::find_if(vec.begin(), vec.end(), equal_to_value_pred<T>(5)) - vec.begin(), 4);
}

template <typename InputIterator, typename Predicate>
InputIterator find_if(my_system& system, InputIterator first, InputIterator, Predicate)
{
  system.validate_dispatch();
  return first;
}

TEST(FindTests, TestFindIfDispatchExplicit)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::device_vector<int> vec(1);

  my_system sys(0);
  thrust::find_if(sys, vec.begin(), vec.end(), ::internal::identity{});

  ASSERT_EQ(true, sys.is_valid());
}

template <typename InputIterator, typename Predicate>
InputIterator find_if(my_tag, InputIterator first, InputIterator, Predicate)
{
  *first = 13;
  return first;
}

TEST(FindTests, TestFindIfDispatchImplicit)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::device_vector<int> vec(1);

  thrust::find_if(thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.end()), ::internal::identity{});

  ASSERT_EQ(13, vec.front());
}

TYPED_TEST(FindTestsVector, TestFindIfNotSimple)
{
  using Vector = typename TestFixture::input_type;
  using T      = typename Vector::value_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  Vector vec{0, 1, 2, 3, 4};

  ASSERT_EQ(0, thrust::find_if_not(vec.begin(), vec.end(), less_than_value_pred<T>(0)) - vec.begin());
  ASSERT_EQ(1, thrust::find_if_not(vec.begin(), vec.end(), less_than_value_pred<T>(1)) - vec.begin());
  ASSERT_EQ(2, thrust::find_if_not(vec.begin(), vec.end(), less_than_value_pred<T>(2)) - vec.begin());
  ASSERT_EQ(3, thrust::find_if_not(vec.begin(), vec.end(), less_than_value_pred<T>(3)) - vec.begin());
  ASSERT_EQ(4, thrust::find_if_not(vec.begin(), vec.end(), less_than_value_pred<T>(4)) - vec.begin());
  ASSERT_EQ(5, thrust::find_if_not(vec.begin(), vec.end(), less_than_value_pred<T>(5)) - vec.begin());
}

template <typename InputIterator, typename Predicate>
InputIterator find_if_not(my_system& system, InputIterator first, InputIterator, Predicate)
{
  system.validate_dispatch();
  return first;
}

TEST(FindTests, TestFindIfNotDispatchExplicit)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::device_vector<int> vec(1);

  my_system sys(0);
  thrust::find_if_not(sys, vec.begin(), vec.end(), ::internal::identity{});

  ASSERT_EQ(true, sys.is_valid());
}

template <typename InputIterator, typename Predicate>
InputIterator find_if_not(my_tag, InputIterator first, InputIterator, Predicate)
{
  *first = 13;
  return first;
}

TEST(FindTests, TestFindIfNotDispatchImplicit)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::device_vector<int> vec(1);

  thrust::find_if_not(thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.end()), ::internal::identity{});

  ASSERT_EQ(13, vec.front());
}

TYPED_TEST(FindTests, TestFind)
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

      typename thrust::host_vector<T>::iterator h_iter;
      typename thrust::device_vector<T>::iterator d_iter;

      h_iter = thrust::find(h_data.begin(), h_data.end(), T(0));
      d_iter = thrust::find(d_data.begin(), d_data.end(), T(0));
      ASSERT_EQ(h_iter - h_data.begin(), d_iter - d_data.begin());

      for (size_t i = 1; i < size; i *= 2)
      {
        T sample = h_data[i];
        h_iter   = thrust::find(h_data.begin(), h_data.end(), sample);
        d_iter   = thrust::find(d_data.begin(), d_data.end(), sample);
        ASSERT_EQ(h_iter - h_data.begin(), d_iter - d_data.begin());
      }
    }
  }
}

TYPED_TEST(FindTests, TestFindIf)
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

      typename thrust::host_vector<T>::iterator h_iter;
      typename thrust::device_vector<T>::iterator d_iter;

      h_iter = thrust::find_if(h_data.begin(), h_data.end(), equal_to_value_pred<T>(0));
      d_iter = thrust::find_if(d_data.begin(), d_data.end(), equal_to_value_pred<T>(0));
      ASSERT_EQ(h_iter - h_data.begin(), d_iter - d_data.begin());

      for (size_t i = 1; i < size; i *= 2)
      {
        T sample = h_data[i];
        h_iter   = thrust::find_if(h_data.begin(), h_data.end(), equal_to_value_pred<T>(sample));
        d_iter   = thrust::find_if(d_data.begin(), d_data.end(), equal_to_value_pred<T>(sample));
        ASSERT_EQ(h_iter - h_data.begin(), d_iter - d_data.begin());
      }
    }
  }
}

TYPED_TEST(FindTests, TestFindIfNot)
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

      typename thrust::host_vector<T>::iterator h_iter;
      typename thrust::device_vector<T>::iterator d_iter;

      h_iter = thrust::find_if_not(h_data.begin(), h_data.end(), not_equal_to_value_pred<T>(0));
      d_iter = thrust::find_if_not(d_data.begin(), d_data.end(), not_equal_to_value_pred<T>(0));
      ASSERT_EQ(h_iter - h_data.begin(), d_iter - d_data.begin());

      for (size_t i = 1; i < size; i *= 2)
      {
        T sample = h_data[i];
        h_iter   = thrust::find_if_not(h_data.begin(), h_data.end(), not_equal_to_value_pred<T>(sample));
        d_iter   = thrust::find_if_not(d_data.begin(), d_data.end(), not_equal_to_value_pred<T>(sample));
        ASSERT_EQ(h_iter - h_data.begin(), d_iter - d_data.begin());
      }
    }
  }
}

void TestFindWithBigIndexesHelper(int magnitude)
{
  thrust::counting_iterator<long long> begin(1);
  thrust::counting_iterator<long long> end = begin + (1ll << magnitude);
  ASSERT_EQ(thrust::distance(begin, end), 1ll << magnitude);

  thrust::detail::intmax_t distance_low_value = thrust::distance(begin, thrust::find(thrust::device, begin, end, 17));

  thrust::detail::intmax_t distance_high_value =
    thrust::distance(begin, thrust::find(thrust::device, begin, end, (1ll << magnitude) - 17));

  ASSERT_EQ(distance_low_value, 16);
  ASSERT_EQ(distance_high_value, (1ll << magnitude) - 18);
}

TEST(FindTests, TestFindWithBigIndexes)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  TestFindWithBigIndexesHelper(30);
  TestFindWithBigIndexesHelper(31);
  TestFindWithBigIndexesHelper(32);
  TestFindWithBigIndexesHelper(33);
}

namespace
{

class Weird
{
  int value;

public:
  THRUST_HOST_DEVICE Weird(int val, int)
      : value(val)
  {}

  friend THRUST_HOST_DEVICE bool operator==(int x, Weird y)
  {
    return x == y.value;
  }
};

} // namespace

TEST(FindTests, TestFindAsymmetricEquality)
{ // Regression test for NVIDIA/thrust#1229
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::host_vector<int> v(1000);
  thrust::sequence(v.begin(), v.end());
  thrust::device_vector<int> dv(v);
  auto result = thrust::find(dv.begin(), dv.end(), Weird(333, 0));
  ASSERT_EQ(*result, 333);
  ASSERT_EQ(result - dv.begin(), 333);
}

__global__ THRUST_HIP_LAUNCH_BOUNDS_DEFAULT void FindKernel(int const N, int* in_array, int value, int* out_array)
{
  if (threadIdx.x == 0)
  {
    thrust::device_ptr<int> in_begin(in_array);
    thrust::device_ptr<int> in_end(in_array + N);

    auto x       = thrust::find(thrust::hip::par, in_begin, in_end, value);
    out_array[0] = x - in_begin;
  }
}
TEST(FindTests, TestFindDevice)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());
  for (auto size : get_sizes())
  {
    SCOPED_TRACE(testing::Message() << "with size= " << size);
    for (auto seed : get_seeds())
    {
      SCOPED_TRACE(testing::Message() << "with seed= " << seed);

      thrust::host_vector<int> h_data =
        get_random_data<int>(size, get_default_limits<int>::min(), get_default_limits<int>::max(), seed);
      thrust::device_vector<int> d_data = h_data;
      thrust::device_vector<int> d_output(1);

      int h_index = thrust::find(h_data.begin(), h_data.end(), 0) - h_data.begin();

      hipLaunchKernelGGL(
        FindKernel,
        dim3(1, 1, 1),
        dim3(128, 1, 1),
        0,
        0,
        size,
        thrust::raw_pointer_cast(&d_data[0]),
        0,
        thrust::raw_pointer_cast(&d_output[0]));
      ASSERT_EQ(h_index, d_output[0]);

      // for(size_t i = 1; i < size; i *= 2)
      // {
      //     T sample = h_data[i];
      //     h_iter   = thrust::find(h_data.begin(), h_data.end(), sample);
      //     d_iter   = thrust::find(d_data.begin(), d_data.end(), sample);
      //     ASSERT_EQ(h_iter - h_data.begin(), d_iter - d_data.begin());
      // }
    }
  }
}
