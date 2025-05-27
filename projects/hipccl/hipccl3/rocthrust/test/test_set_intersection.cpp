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

#include <thrust/extrema.h>
#include <thrust/functional.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/retag.h>
#include <thrust/set_operations.h>
#include <thrust/sort.h>

#include "test_header.hpp"

TESTS_DEFINE(SetIntersectionTests, FullTestsParams);
TESTS_DEFINE(SetIntersectionPrimitiveTests, NumericalTestsParams);

template <typename InputIterator1, typename InputIterator2, typename OutputIterator>
OutputIterator set_intersection(
  my_system& system, InputIterator1, InputIterator1, InputIterator2, InputIterator2, OutputIterator result)
{
  system.validate_dispatch();
  return result;
}

TEST(SetIntersectionTests, TestSetIntersectionDispatchExplicit)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::device_vector<int> vec(1);

  my_system sys(0);
  thrust::set_intersection(sys, vec.begin(), vec.begin(), vec.begin(), vec.begin(), vec.begin());

  ASSERT_EQ(true, sys.is_valid());
}

template <typename InputIterator1, typename InputIterator2, typename OutputIterator>
OutputIterator
set_intersection(my_tag, InputIterator1, InputIterator1, InputIterator2, InputIterator2, OutputIterator result)
{
  *result = 13;
  return result;
}

TEST(SetIntersectionTests, TestSetIntersectionDispatchImplicit)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::device_vector<int> vec(1);

  thrust::set_intersection(
    thrust::retag<my_tag>(vec.begin()),
    thrust::retag<my_tag>(vec.begin()),
    thrust::retag<my_tag>(vec.begin()),
    thrust::retag<my_tag>(vec.begin()),
    thrust::retag<my_tag>(vec.begin()));

  ASSERT_EQ(13, vec.front());
}

TYPED_TEST(SetIntersectionTests, TestSetIntersectionSimple)
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

  Vector ref(2);
  ref[0] = 0;
  ref[1] = 4;

  Vector result(2);

  Iterator end = thrust::set_intersection(Policy{}, a.begin(), a.end(), b.begin(), b.end(), result.begin());

  EXPECT_EQ(result.end(), end);
  ASSERT_EQ(ref, result);
}

TYPED_TEST(SetIntersectionPrimitiveTests, TestSetIntersectionSimple)
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

        h_end =
          thrust::set_intersection(h_a.begin(), h_a.end(), h_b.begin(), h_b.begin() + expanded_size, h_result.begin());
        h_result.resize(h_end - h_result.begin());

        d_end =
          thrust::set_intersection(d_a.begin(), d_a.end(), d_b.begin(), d_b.begin() + expanded_size, d_result.begin());
        d_result.resize(d_end - d_result.begin());

        ASSERT_EQ(h_result, d_result);
      }
    }
  }
}

TYPED_TEST(SetIntersectionPrimitiveTests, TestSetIntersectionToDiscardIterator)
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

      thrust::host_vector<T> h_reference(size);
      typename thrust::host_vector<T>::iterator h_end =
        thrust::set_intersection(h_a.begin(), h_a.end(), h_b.begin(), h_b.end(), h_reference.begin());
      h_reference.erase(h_end, h_reference.end());

      h_result =
        thrust::set_intersection(h_a.begin(), h_a.end(), h_b.begin(), h_b.end(), thrust::make_discard_iterator());

      d_result =
        thrust::set_intersection(d_a.begin(), d_a.end(), d_b.begin(), d_b.end(), thrust::make_discard_iterator());

      thrust::discard_iterator<> reference(h_reference.size());

      EXPECT_EQ(reference, h_result);
      EXPECT_EQ(reference, d_result);
    }
  }
}

TYPED_TEST(SetIntersectionPrimitiveTests, TestSetIntersectionEquivalentRanges)
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
        get_random_data<T>(size, get_default_limits<T>::min(), get_default_limits<T>::max(), seed);

      thrust::host_vector<T> h_a = temp;
      thrust::sort(h_a.begin(), h_a.end());
      thrust::host_vector<T> h_b = h_a;

      thrust::device_vector<T> d_a = h_a;
      thrust::device_vector<T> d_b = h_b;

      thrust::host_vector<T> h_result(size);
      thrust::device_vector<T> d_result(size);

      typename thrust::host_vector<T>::iterator h_end;
      typename thrust::device_vector<T>::iterator d_end;

      h_end = thrust::set_intersection(h_a.begin(), h_a.end(), h_b.begin(), h_b.end(), h_result.begin());
      h_result.resize(h_end - h_result.begin());

      d_end = thrust::set_intersection(d_a.begin(), d_a.end(), d_b.begin(), d_b.end(), d_result.begin());

      d_result.resize(d_end - d_result.begin());

      ASSERT_EQ(h_result, d_result);
    }
  }
}

TYPED_TEST(SetIntersectionPrimitiveTests, TestSetIntersectionMultiset)
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

      // restrict elements to [min,13)
      for (typename thrust::host_vector<T>::iterator i = temp.begin(); i != temp.end(); ++i)
      {
        int temp = static_cast<int>(*i);
        temp %= 13;
        *i = temp;
      }

      thrust::host_vector<T> h_a(temp.begin(), temp.begin() + size);
      thrust::host_vector<T> h_b(temp.begin() + size, temp.end());

      thrust::sort(h_a.begin(), h_a.end());
      thrust::sort(h_b.begin(), h_b.end());

      thrust::device_vector<T> d_a = h_a;
      thrust::device_vector<T> d_b = h_b;

      thrust::host_vector<T> h_result(size);
      thrust::device_vector<T> d_result(size);

      typename thrust::host_vector<T>::iterator h_end;
      typename thrust::device_vector<T>::iterator d_end;

      h_end = thrust::set_intersection(h_a.begin(), h_a.end(), h_b.begin(), h_b.end(), h_result.begin());
      h_result.resize(h_end - h_result.begin());

      d_end = thrust::set_intersection(d_a.begin(), d_a.end(), d_b.begin(), d_b.end(), d_result.begin());

      d_result.resize(d_end - d_result.begin());

      ASSERT_EQ(h_result, d_result);
    }
  }
}
