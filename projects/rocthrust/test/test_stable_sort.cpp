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
#include <thrust/sort.h>

#include "test_param_fixtures.hpp"
#include "test_real_assertions.hpp"
#include "test_utils.hpp"

TESTS_DEFINE(StableSortTests, UnsignedIntegerTestsParams);
TESTS_DEFINE(StableSortVectorTests, VectorIntegerTestsParams);

template <typename RandomAccessIterator>
void stable_sort(my_system& system, RandomAccessIterator, RandomAccessIterator)
{
  system.validate_dispatch();
}

TEST(StableSortTests, TestStableSortDispatchExplicit)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::device_vector<int> vec(1);

  my_system sys(0);
  thrust::stable_sort(sys, vec.begin(), vec.begin());

  ASSERT_EQ(true, sys.is_valid());
}

template <typename RandomAccessIterator>
void stable_sort(my_tag, RandomAccessIterator first, RandomAccessIterator)
{
  *first = 13;
}

TEST(StableSortTests, TestStableSortDispatchImplicit)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::device_vector<int> vec(1);

  thrust::stable_sort(thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.begin()));

  ASSERT_EQ(13, vec.front());
}

template <typename T>
struct less_div_10
{
  THRUST_HOST_DEVICE bool operator()(const T& lhs, const T& rhs) const
  {
    return ((int) lhs) / 10 < ((int) rhs) / 10;
  }
};

template <class Vector>
void InitializeSimpleStableKeySortTest(Vector& unsorted_keys, Vector& sorted_keys)
{
  unsorted_keys.resize(9);
  unsorted_keys = {25, 14, 35, 16, 26, 34, 36, 24, 15};

  sorted_keys.resize(9);
  sorted_keys = {14, 16, 15, 25, 26, 24, 35, 34, 36};
}

TYPED_TEST(StableSortVectorTests, TestStableSortSimple)
{
  using Vector = typename TestFixture::input_type;
  using T      = typename Vector::value_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  Vector unsorted_keys;
  Vector sorted_keys;

  InitializeSimpleStableKeySortTest(unsorted_keys, sorted_keys);

  thrust::stable_sort(unsorted_keys.begin(), unsorted_keys.end(), less_div_10<T>());

  ASSERT_EQ(unsorted_keys, sorted_keys);
}

TYPED_TEST(StableSortTests, TestStableSort)
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

      thrust::stable_sort(h_data.begin(), h_data.end(), less_div_10<T>());
      thrust::stable_sort(d_data.begin(), d_data.end(), less_div_10<T>());

      ASSERT_EQ(h_data, d_data);
    }
  }
}

TYPED_TEST(StableSortTests, TestStableSortSemantics)
{
  using T = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  for (auto size : get_sizes())
  {
    SCOPED_TRACE(testing::Message() << "with size= " << size);

    thrust::host_vector<T> h_data   = random_integers<T>(size);
    thrust::device_vector<T> d_data = h_data;

    thrust::stable_sort(h_data.begin(), h_data.end(), less_div_10<T>());
    thrust::stable_sort(d_data.begin(), d_data.end(), less_div_10<T>());

    ASSERT_EQ(h_data, d_data);
  }
}

template <typename T>
struct comp_mod3
{
  T* table;

  comp_mod3(T* table)
      : table(table)
  {}

  THRUST_HOST_DEVICE bool operator()(T a, T b)
  {
    return table[(int) a] < table[(int) b];
  }
};

TYPED_TEST(StableSortVectorTests, TestStableSortWithIndirection)
{
  using Vector = typename TestFixture::input_type;
  // add numbers modulo 3 with external lookup table
  using T = typename Vector::value_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  Vector data{1, 3, 5, 3, 0, 2, 1};
  Vector table{0, 1, 2, 0, 1, 2};

  thrust::stable_sort(data.begin(), data.end(), comp_mod3<T>(thrust::raw_pointer_cast(&table[0])));

  Vector ref{3, 3, 0, 1, 1, 5, 2};
  ASSERT_EQ(data, ref);
}

#ifndef _WIN32
__global__ THRUST_HIP_LAUNCH_BOUNDS_DEFAULT void StableSortKernel(int const N, int* array)
{
  if (threadIdx.x == 0)
  {
    thrust::device_ptr<int> begin(array);
    thrust::device_ptr<int> end(array + N);
    thrust::stable_sort(thrust::hip::par, begin, end);
  }
}

TEST(StableSortTests, TestStableSortDevice)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  for (auto size : get_sizes())
  {
    SCOPED_TRACE(testing::Message() << "with size= " << size);

    for (auto seed : get_seeds())
    {
      SCOPED_TRACE(testing::Message() << "with seed= " << seed);

      thrust::host_vector<int> h_data = get_random_data<int>(size, 0, size, seed);

      thrust::device_vector<int> d_data = h_data;

      thrust::stable_sort(h_data.begin(), h_data.end());
      hipLaunchKernelGGL(
        StableSortKernel, dim3(1, 1, 1), dim3(128, 1, 1), 0, 0, size, thrust::raw_pointer_cast(&d_data[0]));

      ASSERT_EQ(h_data, d_data);
    }
  }
}
#endif
