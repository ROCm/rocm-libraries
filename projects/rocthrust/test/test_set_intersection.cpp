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

#include "test_param_fixtures.hpp"
#include "test_real_assertions.hpp"
#include "test_utils.hpp"

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
  using Iterator = typename Vector::iterator;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  Vector a{0, 2, 4}, b{0, 3, 3, 4};
  Vector ref{0, 4};

  Vector result(2);

  Iterator end = thrust::set_intersection(a.begin(), a.end(), b.begin(), b.end(), result.begin());

  ASSERT_EQ_QUIET(result.end(), end);
  ASSERT_EQ(ref, result);
}

TYPED_TEST(SetIntersectionPrimitiveTests, TestSetIntersection)
{
  using T = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  for (auto size : get_sizes())
  {
    SCOPED_TRACE(testing::Message() << "with size= " << size);

    size_t expanded_sizes[]   = {0, 1, size / 2, size, size + 1, 2 * size};
    size_t num_expanded_sizes = sizeof(expanded_sizes) / sizeof(size_t);

    thrust::host_vector<T> random =
      random_integers<int8_t>(size + *thrust::max_element(expanded_sizes, expanded_sizes + num_expanded_sizes));

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

      h_end = thrust::set_intersection(h_a.begin(), h_a.end(), h_b.begin(), h_b.begin() + size, h_result.begin());
      h_result.resize(h_end - h_result.begin());

      d_end = thrust::set_intersection(d_a.begin(), d_a.end(), d_b.begin(), d_b.begin() + size, d_result.begin());
      d_result.resize(d_end - d_result.begin());

      ASSERT_EQ(h_result, d_result);
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

      ASSERT_EQ_QUIET(reference, h_result);
      ASSERT_EQ_QUIET(reference, d_result);
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

      thrust::host_vector<T> vec =
        get_random_data<int>(2 * size, get_default_limits<int>::min(), get_default_limits<int>::max(), seed);

      // restrict elements to [min,13)
      for (typename thrust::host_vector<T>::iterator i = vec.begin(); i != vec.end(); ++i)
      {
        int tmp = static_cast<int>(*i);
        tmp %= 13;
        *i = static_cast<T>(tmp);
      }

      thrust::host_vector<T> h_a(vec.begin(), vec.begin() + size);
      thrust::host_vector<T> h_b(vec.begin() + size, vec.end());

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

// FIXME: disabled on Windows, because it causes a failure on the internal CI system in one specific configuration.
// That failure will be tracked in a new NVBug, this is disabled to unblock submitting all the other changes.
#if THRUST_HOST_COMPILER != THRUST_HOST_COMPILER_MSVC
void TestSetDifferenceWithBigIndexesHelper(int magnitude)
{
  thrust::counting_iterator<long long> begin1(0);
  thrust::counting_iterator<long long> begin2 = begin1 + (1ll << magnitude);
  thrust::counting_iterator<long long> end1   = begin2 + 1;
  thrust::counting_iterator<long long> end2   = begin2 + (1ll << magnitude);
  ASSERT_EQ(thrust::distance(begin2, end1), 1);

  thrust::device_vector<long long> result;
  result.resize(1);
  thrust::set_intersection(thrust::device, begin1, end1, begin2, end2, result.begin());

  thrust::host_vector<long long> expected;
  expected.push_back(*begin2);

  ASSERT_EQ(result, expected);
}

TEST(SetIntersectionTests, TestSetDifferenceWithBigIndexes)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

#  ifndef THRUST_FORCE_32_BIT_OFFSET_TYPE
  TestSetDifferenceWithBigIndexesHelper(30);
  TestSetDifferenceWithBigIndexesHelper(31);
  TestSetDifferenceWithBigIndexesHelper(32);
  TestSetDifferenceWithBigIndexesHelper(33);
#  endif
}
#endif
