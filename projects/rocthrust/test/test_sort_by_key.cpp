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

TESTS_DEFINE(SortByKeyTests, FullTestsParams);
TESTS_DEFINE(SortByKeyPrimitiveTests, NumericalTestsParams);

template <typename RandomAccessIterator1, typename RandomAccessIterator2>
void sort_by_key(my_system& system, RandomAccessIterator1, RandomAccessIterator1, RandomAccessIterator2)
{
  system.validate_dispatch();
}

TEST(SortByKeyTests, TestSortByKeyDispatchExplicit)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::device_vector<int> vec(1);

  my_system sys(0);
  thrust::sort_by_key(sys, vec.begin(), vec.begin(), vec.begin());

  ASSERT_EQ(true, sys.is_valid());
}

template <typename RandomAccessIterator1, typename RandomAccessIterator2>
void sort_by_key(my_tag, RandomAccessIterator1 keys_first, RandomAccessIterator1, RandomAccessIterator2)
{
  *keys_first = 13;
}

TEST(SortByKeyTests, TestSortByKeyDispatchImplicit)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::device_vector<int> vec(1);

  thrust::sort_by_key(
    thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.begin()));

  ASSERT_EQ(13, vec.front());
}

template <class Vector>
void InitializeSimpleKeyValueSortTest(
  Vector& unsorted_keys, Vector& unsorted_values, Vector& sorted_keys, Vector& sorted_values)
{
  unsorted_keys.resize(7);
  unsorted_keys = {1, 3, 6, 5, 2, 0, 4};
  unsorted_values.resize(7);
  unsorted_values = {0, 1, 2, 3, 4, 5, 6};

  sorted_keys.resize(7);
  sorted_keys = {0, 1, 2, 3, 4, 5, 6};
  sorted_values.resize(7);
  sorted_values = {5, 0, 4, 1, 6, 3, 2};
}

TYPED_TEST(SortByKeyTests, TestSortByKeySimple)
{
  using Vector = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  Vector unsorted_keys, unsorted_values;
  Vector sorted_keys, sorted_values;

  InitializeSimpleKeyValueSortTest(unsorted_keys, unsorted_values, sorted_keys, sorted_values);

  thrust::sort_by_key(unsorted_keys.begin(), unsorted_keys.end(), unsorted_values.begin());

  ASSERT_EQ(unsorted_keys, sorted_keys);
  ASSERT_EQ(unsorted_values, sorted_values);
}

TYPED_TEST(SortByKeyPrimitiveTests, TestSortAscendingKeyValue)
{
  using T = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  for (auto size : get_sizes())
  {
    SCOPED_TRACE(testing::Message() << "with size= " << size);

    for (auto seed : get_seeds())
    {
      SCOPED_TRACE(testing::Message() << "with seed= " << seed);

      thrust::host_vector<T> h_keys =
        get_random_data<T>(size, get_default_limits<T>::min(), get_default_limits<T>::max(), seed);
      thrust::device_vector<T> d_keys = h_keys;

      thrust::host_vector<T> h_values   = h_keys;
      thrust::device_vector<T> d_values = d_keys;

      thrust::sort_by_key(h_keys.begin(), h_keys.end(), h_values.begin(), thrust::less<T>());
      thrust::sort_by_key(d_keys.begin(), d_keys.end(), d_values.begin(), thrust::less<T>());

      ASSERT_EQ(h_keys, d_keys);
      ASSERT_EQ(h_values, d_values);
    }
  }
}

TEST(SortByKeyTests, TestSortDescendingKeyValue)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  for (auto size : get_sizes())
  {
    SCOPED_TRACE(testing::Message() << "with size= " << size);

    for (auto seed : get_seeds())
    {
      SCOPED_TRACE(testing::Message() << "with seed= " << seed);

      thrust::host_vector<int> h_keys =
        get_random_data<int>(size, get_default_limits<int>::min(), get_default_limits<int>::max(), seed);
      thrust::device_vector<int> d_keys = h_keys;

      thrust::host_vector<int> h_values   = h_keys;
      thrust::device_vector<int> d_values = d_keys;

      thrust::sort_by_key(h_keys.begin(), h_keys.end(), h_values.begin(), thrust::greater<int>());
      thrust::sort_by_key(d_keys.begin(), d_keys.end(), d_values.begin(), thrust::greater<int>());

      ASSERT_EQ(h_keys, d_keys);
      ASSERT_EQ(h_values, d_values);
    }
  }
}

TEST(SortByKeyTests, TestSortByKeyBool)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  const size_t n = 10027;

  for (auto seed : get_seeds())
  {
    SCOPED_TRACE(testing::Message() << "with seed= " << seed);

    thrust::host_vector<bool> h_keys =
      get_random_data<bool>(n, get_default_limits<bool>::min(), get_default_limits<bool>::max(), seed);
    thrust::host_vector<int> h_values = get_random_data<int>(
      n, get_default_limits<int>::min(), get_default_limits<int>::max(), seed + seed_value_addition);

    thrust::device_vector<bool> d_keys  = h_keys;
    thrust::device_vector<int> d_values = h_values;

    thrust::sort_by_key(h_keys.begin(), h_keys.end(), h_values.begin());
    thrust::sort_by_key(d_keys.begin(), d_keys.end(), d_values.begin());

    ASSERT_EQ(h_keys, d_keys);
    ASSERT_EQ(h_values, d_values);
  }
}

TEST(SortByKeyTests, TestSortByKeyBoolDescending)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  const size_t n = 10027;

  for (auto seed : get_seeds())
  {
    SCOPED_TRACE(testing::Message() << "with seed= " << seed);

    thrust::host_vector<bool> h_keys =
      get_random_data<bool>(n, get_default_limits<bool>::min(), get_default_limits<bool>::max(), seed);
    thrust::host_vector<int> h_values =
      get_random_data<int>(n, get_default_limits<int>::min(), get_default_limits<int>::max(), seed);

    thrust::device_vector<bool> d_keys  = h_keys;
    thrust::device_vector<int> d_values = h_values;

    thrust::sort_by_key(h_keys.begin(), h_keys.end(), h_values.begin(), thrust::greater<bool>());
    thrust::sort_by_key(d_keys.begin(), d_keys.end(), d_values.begin(), thrust::greater<bool>());

    ASSERT_EQ(h_keys, d_keys);
    ASSERT_EQ(h_values, d_values);
  }
}

#ifndef _WIN32
__global__ THRUST_HIP_LAUNCH_BOUNDS_DEFAULT void SortByKeyKernel(int const N, int* keys, short* values)
{
  if (threadIdx.x == 0)
  {
    thrust::device_ptr<int> keys_begin(keys);
    thrust::device_ptr<int> keys_end(keys + N);
    thrust::device_ptr<short> val(values);
    thrust::sort_by_key(thrust::hip::par, keys_begin, keys_end, val);
  }
}

TEST(SortByKeyTests, TestSortByKeyDevice)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  for (auto size : get_sizes())
  {
    SCOPED_TRACE(testing::Message() << "with size= " << size);

    for (auto seed : get_seeds())
    {
      SCOPED_TRACE(testing::Message() << "with seed= " << seed);

      thrust::host_vector<int> h_keys = get_random_data<int>(size, 0, size, seed);

      thrust::host_vector<short> h_values =
        get_random_data<short>(size, get_default_limits<short>::min(), get_default_limits<short>::max(), seed);

      thrust::device_vector<int> d_keys     = h_keys;
      thrust::device_vector<short> d_values = h_values;

      thrust::sort_by_key(h_keys.begin(), h_keys.end(), h_values.begin());
      hipLaunchKernelGGL(
        SortByKeyKernel,
        dim3(1, 1, 1),
        dim3(128, 1, 1),
        0,
        0,
        size,
        thrust::raw_pointer_cast(&d_keys[0]),
        thrust::raw_pointer_cast(&d_values[0]));

      ASSERT_EQ(h_keys, d_keys);
      // Only keys are compared here, the sequential stable_merge_sort that's used in
      // CUDA and HIP don't generate the correct value sorting
    }
  }
}
#endif
