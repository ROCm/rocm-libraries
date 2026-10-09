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

#include <thrust/device_malloc_allocator.h>
#include <thrust/sequence.h>

#include "test_param_fixtures.hpp"
#include "test_real_assertions.hpp"
#include "test_utils.hpp"

#if !_THRUST_HAS_DEVICE_SYSTEM_STD
#  include <utility>
#endif

TESTS_DEFINE(VectorInsertTests, FullTestsParams);
TESTS_DEFINE(VectorInsertPrimitiveTests, NumericalTestsParams);

TYPED_TEST(VectorInsertTests, TestVectorRangeInsertSimple)
{
  using Vector = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  Vector v1(5);
  thrust::sequence(v1.begin(), v1.end());

  // test when insertion range fits inside capacity
  // and the size of the insertion is greater than the number
  // of displaced elements
  Vector v2(3);
  v2.reserve(10);
  thrust::sequence(v2.begin(), v2.end());

  size_t new_size       = v2.size() + v1.size();
  size_t insertion_size = v1.end() - v1.begin();
  size_t num_displaced  = v2.end() - (v2.begin() + 1);

  ASSERT_EQ(true, v2.capacity() >= new_size);
  ASSERT_EQ(true, insertion_size > num_displaced);

  v2.insert(v2.begin() + 1, v1.begin(), v1.end());

  Vector ref{0, 0, 1, 2, 3, 4, 1, 2};
  ASSERT_EQ(ref, v2);

  ASSERT_EQ(8lu, v2.size());
  ASSERT_EQ(10lu, v2.capacity());

  // test when insertion range fits inside capacity
  // and the size of the insertion is equal to the number
  // of displaced elements
  Vector v3(5);
  v3.reserve(10);
  thrust::sequence(v3.begin(), v3.end());

  new_size       = v3.size() + v1.size();
  insertion_size = v1.end() - v1.begin();
  num_displaced  = v3.end() - v3.begin();

  ASSERT_EQ(true, v3.capacity() >= new_size);
  ASSERT_EQ(true, insertion_size == num_displaced);

  v3.insert(v3.begin(), v1.begin(), v1.end());
  ref = {0, 1, 2, 3, 4, 0, 1, 2, 3, 4};
  ASSERT_EQ(ref, v3);

  ASSERT_EQ(10lu, v3.size());
  ASSERT_EQ(10lu, v3.capacity());

  // test when insertion range fits inside capacity
  // and the size of the insertion is less than the
  // number of displaced elements
  Vector v4(5);
  v4.reserve(10);
  thrust::sequence(v4.begin(), v4.end());

  new_size       = v4.size() + v1.size();
  insertion_size = (v1.begin() + 3) - v1.begin();
  num_displaced  = v4.end() - (v4.begin() + 1);

  ASSERT_EQ(true, v4.capacity() >= new_size);
  ASSERT_EQ(true, insertion_size < num_displaced);

  v4.insert(v4.begin() + 1, v1.begin(), v1.begin() + 3);

  ref = {0, 0, 1, 2, 1, 2, 3, 4};
  ASSERT_EQ(ref, v4);

  ASSERT_EQ(8lu, v4.size());
  ASSERT_EQ(10lu, v4.capacity());

  // test when insertion range does not fit inside capacity
  Vector v5(5);
  thrust::sequence(v5.begin(), v5.end());

  new_size = v5.size() + v1.size();

  ASSERT_EQ(true, v5.capacity() < new_size);

  v5.insert(v5.begin() + 1, v1.begin(), v1.end());

  ref = {0, 0, 1, 2, 3, 4, 1, 2, 3, 4};
  ASSERT_EQ(ref, v5);

  ASSERT_EQ(10lu, v5.size());
}

TYPED_TEST(VectorInsertPrimitiveTests, TestVectorRangeInsert)
{
  using T = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  for (auto size : get_sizes())
  {
    SCOPED_TRACE(testing::Message() << "with size= " << size);

    for (auto seed : get_seeds())
    {
      SCOPED_TRACE(testing::Message() << "with seed= " << seed);

      thrust::host_vector<T> h_src =
        get_random_data<T>(size + 3, get_default_limits<T>::min(), get_default_limits<T>::max(), seed);
      thrust::host_vector<T> h_dst = get_random_data<T>(
        size, get_default_limits<T>::min(), get_default_limits<T>::max(), seed + seed_value_addition);

      thrust::device_vector<T> d_src = h_src;
      thrust::device_vector<T> d_dst = h_dst;

      // choose insertion range at random
      size_t begin = size > 0 ? (size_t) h_src[size] % size : 0;
      size_t end   = size > 0 ? (size_t) h_src[size + 1] % size : 0;
      if (end < begin)
      {
        using _THRUST_STD::swap;
        swap(begin, end);
      }

      // choose insertion position at random
      size_t position = size > 0 ? (size_t) h_src[size + 2] % size : 0;

      // insert on host
      h_dst.insert(h_dst.begin() + position, h_src.begin() + begin, h_src.begin() + end);

      // insert on device
      d_dst.insert(d_dst.begin() + position, d_src.begin() + begin, d_src.begin() + end);

      ASSERT_EQ(h_dst, d_dst);
    }
  }
} // end TestVectorRangeInsert

TYPED_TEST(VectorInsertTests, TestVectorFillInsertSimple)
{
  using Vector = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  // test when insertion range fits inside capacity
  // and the size of the insertion is greater than the number
  // of displaced elements
  Vector v1(3);
  v1.reserve(10);
  thrust::sequence(v1.begin(), v1.end());

  size_t insertion_size = 5;
  size_t new_size       = v1.size() + insertion_size;
  size_t num_displaced  = v1.end() - (v1.begin() + 1);

  ASSERT_EQ(true, v1.capacity() >= new_size);
  ASSERT_EQ(true, insertion_size > num_displaced);

  v1.insert(v1.begin() + 1, insertion_size, 13);

  Vector ref{0, 13, 13, 13, 13, 13, 1, 2};
  ASSERT_EQ(ref, v1);

  ASSERT_EQ(8lu, v1.size());
  ASSERT_EQ(10lu, v1.capacity());

  // test when insertion range fits inside capacity
  // and the size of the insertion is equal to the number
  // of displaced elements
  Vector v2(5);
  v2.reserve(10);
  thrust::sequence(v2.begin(), v2.end());

  insertion_size = 5;
  new_size       = v2.size() + insertion_size;
  num_displaced  = v2.end() - v2.begin();

  ASSERT_EQ(true, v2.capacity() >= new_size);
  ASSERT_EQ(true, insertion_size == num_displaced);

  v2.insert(v2.begin(), insertion_size, 13);

  ref = {13, 13, 13, 13, 13, 0, 1, 2, 3, 4};
  ASSERT_EQ(ref, v2);

  ASSERT_EQ(10lu, v2.size());
  ASSERT_EQ(10lu, v2.capacity());

  // test when insertion range fits inside capacity
  // and the size of the insertion is less than the
  // number of displaced elements
  Vector v3(5);
  v3.reserve(10);
  thrust::sequence(v3.begin(), v3.end());

  insertion_size = 3;
  new_size       = v3.size() + insertion_size;
  num_displaced  = v3.end() - (v3.begin() + 1);

  ASSERT_EQ(true, v3.capacity() >= new_size);
  ASSERT_EQ(true, insertion_size < num_displaced);

  v3.insert(v3.begin() + 1, insertion_size, 13);

  ref = {0, 13, 13, 13, 1, 2, 3, 4};
  ASSERT_EQ(ref, v3);

  ASSERT_EQ(8lu, v3.size());
  ASSERT_EQ(10lu, v3.capacity());

  // test when insertion range does not fit inside capacity
  Vector v4(5);
  thrust::sequence(v4.begin(), v4.end());

  insertion_size = 5;
  new_size       = v4.size() + insertion_size;

  ASSERT_EQ(true, v4.capacity() < new_size);

  v4.insert(v4.begin() + 1, insertion_size, 13);

  ref = {0, 13, 13, 13, 13, 13, 1, 2, 3, 4};
  ASSERT_EQ(ref, v4);

  ASSERT_EQ(10lu, v4.size());
} // end TestVectorFillInsertSimple

TYPED_TEST(VectorInsertPrimitiveTests, TestVectorFillInsert)
{
  using T = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  for (auto size : get_sizes())
  {
    SCOPED_TRACE(testing::Message() << "with size= " << size);

    for (auto seed : get_seeds())
    {
      SCOPED_TRACE(testing::Message() << "with seed= " << seed);

      thrust::host_vector<T> h_dst =
        get_random_data<T>(size + 2, get_default_limits<T>::min(), get_default_limits<T>::max(), seed);

      thrust::device_vector<T> d_dst = h_dst;

      // choose insertion position at random
      size_t position = size > 0 ? (size_t) h_dst[size] % size : 0;

      // choose insertion size at random
      size_t insertion_size = size > 0 ? (size_t) h_dst[size] % size : 13;

      // insert on host
      h_dst.insert(h_dst.begin() + position, insertion_size, 13);

      // insert on device
      d_dst.insert(d_dst.begin() + position, insertion_size, 13);

      ASSERT_EQ(h_dst, d_dst);
    }
  }
} // end TestVectorFillInsert
