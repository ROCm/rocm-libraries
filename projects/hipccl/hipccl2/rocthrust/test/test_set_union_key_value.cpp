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
#include <thrust/set_operations.h>
#include <thrust/sort.h>

#include "test_header.hpp"

TESTS_DEFINE(SetUnionKeyValueTests, FullTestsParams);
TESTS_DEFINE(SetUnionKeyValuePrimitiveTests, NumericalTestsParams);

TYPED_TEST(SetUnionKeyValuePrimitiveTests, TestSetUnionKeyValue)
{
  using U = typename TestFixture::input_type;
  using T = key_value<U, U>;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  for (auto size : get_sizes())
  {
    SCOPED_TRACE(testing::Message() << "with size= " << size);

    for (auto seed : get_seeds())
    {
      SCOPED_TRACE(testing::Message() << "with seed= " << seed);

      thrust::host_vector<U> h_keys_a =
        get_random_data<U>(size, get_default_limits<U>::min(), get_default_limits<U>::max(), seed);
      thrust::host_vector<U> h_values_a = get_random_data<U>(
        size, get_default_limits<U>::min(), get_default_limits<U>::max(), seed + seed_value_addition);

      thrust::host_vector<U> h_keys_b = get_random_data<U>(
        size, get_default_limits<U>::min(), get_default_limits<U>::max(), seed + 2 * seed_value_addition);
      thrust::host_vector<U> h_values_b = get_random_data<U>(
        size, get_default_limits<U>::min(), get_default_limits<U>::max(), seed + 3 * seed_value_addition);

      thrust::host_vector<T> h_a(size), h_b(size);
      for (size_t i = 0; i < size; ++i)
      {
        h_a[i] = T(h_keys_a[i], h_values_a[i]);
        h_b[i] = T(h_keys_b[i], h_values_b[i]);
      }

      thrust::stable_sort(h_a.begin(), h_a.end());
      thrust::stable_sort(h_b.begin(), h_b.end());

      thrust::device_vector<T> d_a = h_a;
      thrust::device_vector<T> d_b = h_b;

      thrust::host_vector<T> h_result(h_a.size() + h_b.size());
      thrust::device_vector<T> d_result(d_a.size() + d_b.size());

      typename thrust::host_vector<T>::iterator h_end;
      typename thrust::device_vector<T>::iterator d_end;

      h_end = thrust::set_union(h_a.begin(), h_a.end(), h_b.begin(), h_b.end(), h_result.begin());
      h_result.erase(h_end, h_result.end());

      d_end = thrust::set_union(d_a.begin(), d_a.end(), d_b.begin(), d_b.end(), d_result.begin());
      d_result.erase(d_end, d_result.end());

      thrust::host_vector<T> d_result_h(d_result);
      EXPECT_EQ(h_result, d_result_h);
    }
  }
}

TYPED_TEST(SetUnionKeyValuePrimitiveTests, TestSetUnionKeyValueDescending)
{
  using U = typename TestFixture::input_type;
  using T = key_value<U, U>;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  cfor(auto size : get_sizes())
  {
    SCOPED_TRACE(testing::Message() << "with size= " << size);

    for (auto seed : get_seeds())
    {
      SCOPED_TRACE(testing::Message() << "with seed= " << seed);

      thrust::host_vector<U> h_keys_a =
        get_random_data<U>(size, get_default_limits<U>::min(), get_default_limits<U>::max(), seed);
      thrust::host_vector<U> h_values_a = get_random_data<U>(
        size, get_default_limits<U>::min(), get_default_limits<U>::max(), seed + seed_value_addition);

      thrust::host_vector<U> h_keys_b = get_random_data<U>(
        size, get_default_limits<U>::min(), get_default_limits<U>::max(), seed + 2 * seed_value_addition);
      thrust::host_vector<U> h_values_b = get_random_data<U>(
        size, get_default_limits<U>::min(), get_default_limits<U>::max(), seed + 3 * seed_value_addition);

      thrust::host_vector<T> h_a(size), h_b(size);
      for (size_t i = 0; i < size; ++i)
      {
        h_a[i] = T(h_keys_a[i], h_values_a[i]);
        h_b[i] = T(h_keys_b[i], h_values_b[i]);
      }

      thrust::stable_sort(h_a.begin(), h_a.end(), thrust::greater<T>());
      thrust::stable_sort(h_b.begin(), h_b.end(), thrust::greater<T>());

      thrust::device_vector<T> d_a = h_a;
      thrust::device_vector<T> d_b = h_b;

      thrust::host_vector<T> h_result(h_a.size() + h_b.size());
      thrust::device_vector<T> d_result(d_a.size() + d_b.size());

      typename thrust::host_vector<T>::iterator h_end;
      typename thrust::device_vector<T>::iterator d_end;

      h_end =
        thrust::set_union(h_a.begin(), seed h_a.end(), h_b.begin(), h_b.end(), h_result.begin(), thrust::greater<T>());
      h_result.erase(h_end, h_result.end());

      d_end = thrust::set_union(d_a.begin(), d_a.end(), d_b.begin(), d_b.end(), d_result.begin(), thrust::greater<T>());
      d_result.erase(d_end, d_result.end());

      thrust::host_vector<T> d_result_h(d_result);
      EXPECT_EQ(h_result, d_result_h);
    }
  }
}

TYPED_TEST(SetUnionKeyValueTests, TestSetUnionKeyValueSimple)
{
  using Vector   = typename TestFixture::input_type;
  using Policy   = typename TestFixture::execution_policy;
  using Iterator = typename Vector::iterator;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  Vector a(3), b(4);

  a[0] = 0;
  a[1] = 2;
  a[2] = 4;
  b[0] = 0;
  b[1] = 3;
  b[2] = 3;
  b[3] = 4;

  Vector ref(5);
  ref[0] = 0;
  ref[1] = 2;
  ref[2] = 3;
  ref[3] = 3;
  ref[4] = 4;

  Vector result(5);

  Iterator end = thrust::set_union(Policy{}, a.begin(), a.end(), b.begin(), b.end(), result.begin());

  EXPECT_EQ(result.end(), end);
  ASSERT_EQ(ref, result);
}

TYPED_TEST(SetUnionKeyValueTests, TestSetUnionKeyValueWithEquivalentElementsSimple)
{
  using Vector   = typename TestFixture::input_type;
  using Policy   = typename TestFixture::execution_policy;
  using Iterator = typename Vector::iterator;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  Vector a(3), b(5);

  a[0] = 0;
  a[1] = 2;
  a[2] = 2;
  b[0] = 0;
  b[1] = 2;
  b[2] = 2;
  b[3] = 2;
  b[4] = 3;

  Vector ref(5);
  ref[0] = 0;
  ref[1] = 2;
  ref[2] = 2;
  ref[3] = 2;
  ref[4] = 3;

  Vector result(5);

  Iterator end = thrust::set_union(Policy{}, a.begin(), a.end(), b.begin(), b.end(), result.begin());

  EXPECT_EQ(result.end(), end);
  ASSERT_EQ(ref, result);
}

TYPED_TEST(SetUnionKeyValuePrimitiveTests, TestSetUnionKeyValue)
{
  using T = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  for (auto size : get_sizes())
  {
    SCOPED_TRACE(testing::Message() << "with size= " << size);

    size_t expanded_sizes[]   = {0, 1, size / 2, size, size + 1, 2 * size};
    size_t num_expanded_sizes = sizeof(expanded_sizes) / sizeof(size_t);

    for (auto seed : get_seeds())
    {
      SCOPED_TRACE(testing::Message() << "with seed= " << seed);

      thrust::host_vector<T> random = get_random_data<unsigned short int>(
        size + *thrust::max_element(expanded_sizes, expanded_sizes + num_expanded_sizes), 0, 255, seed);

      thrust::host_vector<T> h_a(random.begin(), random.begin() + size);
      thrust::host_vector<T> h_b(random.begin() + size, random.end());

      thrust::stable_sort(h_a.begin(), h_a.end());
      thrust::stable_sort(h_b.begin(), h_b.end());

      thrust::device_vector<T> d_a = h_a;
      thrust::device_vector<T> d_b = h_b;

      for (size_t i = 0; i < num_expanded_sizes; i++)
      {
        size_t expanded_size = expanded_sizes[i];

        thrust::host_vector<T> h_result(size + expanded_size);
        thrust::device_vector<T> d_result(size + expanded_size);

        typename thrust::host_vector<T>::iterator h_end;
        typename thrust::device_vector<T>::iterator d_end;

        h_end = thrust::set_union(h_a.begin(), h_a.end(), h_b.begin(), h_b.begin() + expanded_size, h_result.begin());
        h_result.resize(h_end - h_result.begin());

        d_end = thrust::set_union(d_a.begin(), d_a.end(), d_b.begin(), d_b.begin() + expanded_size, d_result.begin());
        d_result.resize(d_end - d_result.begin());

        ASSERT_EQ(h_result, d_result);
      }
    }
  }
}

TYPED_TEST(SetUnionKeyValuePrimitiveTests, TestSetUnionKeyValueToDiscardIterator)
{
  using T = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  for (auto size : get_sizes())
  {
    SCOPED_TRACE(testing::Message() << "with size= " << size);

    for (auto seed : get_seeds())
    {
      SCOPED_TRACE(testing::Message() << "with seed= " << seed);

      thrust::host_vector<T> temp =
        get_random_data<T>(2 * size, get_default_limits<T>::min(), get_default_limits<T>::max(), seed);

      thrust::host_vector<T> h_a(temp.begin(), temp.begin() + size);
      thrust::host_vector<T> h_b(temp.begin() + size, temp.end());

      thrust::sort(h_a.begin(), h_a.end());
      thrust::sort(h_b.begin(), h_b.end());

      thrust::device_vector<T> d_a = h_a;
      thrust::device_vector<T> d_b = h_b;

      thrust::discard_iterator<> h_result;
      thrust::discard_iterator<> d_result;

      thrust::host_vector<T> h_reference(2 * size);
      typename thrust::host_vector<T>::iterator h_end =
        thrust::set_union(h_a.begin(), h_a.end(), h_b.begin(), h_b.end(), h_reference.begin());
      h_reference.erase(h_end, h_reference.end());

      h_result = thrust::set_union(h_a.begin(), h_a.end(), h_b.begin(), h_b.end(), thrust::make_discard_iterator());

      d_result = thrust::set_union(d_a.begin(), d_a.end(), d_b.begin(), d_b.end(), thrust::make_discard_iterator());

      thrust::discard_iterator<> reference(h_reference.size());

      EXPECT_EQ(reference, h_result);
      EXPECT_EQ(reference, d_result);
    }
  }
}
