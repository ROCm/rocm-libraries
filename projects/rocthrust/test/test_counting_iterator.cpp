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

#include <thrust/binary_search.h>
#include <thrust/distance.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/detail/iterator_traits.h>
#include <thrust/sort.h>

#include <cstdint>

#include "test_param_fixtures.hpp"
#include "test_real_assertions.hpp"
#include "test_utils.hpp"

#include _THRUST_STD_INCLUDE(iterator)
#include _THRUST_STD_INCLUDE(type_traits)

TESTS_DEFINE(CountingIteratorTests, NumericalTestsParams);

THRUST_DIAG_PUSH
THRUST_DIAG_SUPPRESS_MSVC(4244 4267) // possible loss of data

// ensure that we properly support thrust::counting_iterator from _THRUST_STD
TEST(CountingIteratorTests, TestIteratorTraits)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  using It       = _THRUST_STD::iterator_traits<thrust::counting_iterator<int>>;
  using category = thrust::detail::iterator_category_with_system_and_traversal<::std::random_access_iterator_tag,
                                                                               thrust::any_system_tag,
                                                                               thrust::random_access_traversal_tag>;

  static_assert(_THRUST_STD::is_same<It::difference_type, ptrdiff_t>::value, "");
  static_assert(_THRUST_STD::is_same<It::value_type, int>::value, "");
  static_assert(_THRUST_STD::is_same<It::pointer, void>::value, "");
  static_assert(_THRUST_STD::is_same<It::reference, signed int>::value, "");
  static_assert(_THRUST_STD::is_same<It::iterator_category, category>::value, "");

  static_assert(::thrust::detail::is_cpp17_random_access_iterator<thrust::counting_iterator<int>>::value, "");
}

TYPED_TEST(CountingIteratorTests, TestCountingDefaultConstructor)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());
  using T = typename TestFixture::input_type;

  thrust::counting_iterator<T> iter0;
  ASSERT_EQ(*iter0, T{});
}

TEST(CountingIteratorTests, TestCountingIteratorCopyConstructor)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::counting_iterator<int> iter0(100);

  thrust::counting_iterator<int> iter1(iter0);

  ASSERT_EQ_QUIET(iter0, iter1);
  ASSERT_EQ(*iter0, *iter1);

  // construct from related space
  thrust::counting_iterator<int, thrust::host_system_tag> h_iter = iter0;
  ASSERT_EQ(*iter0, *h_iter);

  thrust::counting_iterator<int, thrust::device_system_tag> d_iter = iter0;
  ASSERT_EQ(*iter0, *d_iter);
}
static_assert(_THRUST_STD::is_trivially_copy_constructible<thrust::counting_iterator<int>>::value, "");
static_assert(_THRUST_STD::is_trivially_copyable<thrust::counting_iterator<int>>::value, "");

TEST(CountingIteratorTests, TestCountingIteratorIncrement)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::counting_iterator<int> iter(0);

  ASSERT_EQ(*iter, 0);

  iter++;

  ASSERT_EQ(*iter, 1);

  iter++;
  iter++;

  ASSERT_EQ(*iter, 3);

  iter += 5;

  ASSERT_EQ(*iter, 8);

  iter -= 10;

  ASSERT_EQ(*iter, -2);
}

TEST(CountingIteratorTests, TestCountingIteratorComparison)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::counting_iterator<int> iter1(0);
  thrust::counting_iterator<int> iter2(0);

  ASSERT_EQ(iter1 - iter2, 0);
  ASSERT_EQ(iter1 == iter2, true);

  iter1++;

  ASSERT_EQ(iter1 - iter2, 1);
  ASSERT_EQ(iter1 == iter2, false);

  iter2++;

  ASSERT_EQ(iter1 - iter2, 0);
  ASSERT_EQ(iter1 == iter2, true);

  iter1 += 100;
  iter2 += 100;

  ASSERT_EQ(iter1 - iter2, 0);
  ASSERT_EQ(iter1 == iter2, true);
}

TEST(CountingIteratorTests, TestCountingIteratorFloatComparison)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::counting_iterator<float> iter1(0);
  thrust::counting_iterator<float> iter2(0);

  ASSERT_EQ(iter1 - iter2, 0);
  ASSERT_EQ(iter1 == iter2, true);
  ASSERT_EQ(iter1 < iter2, false);
  ASSERT_EQ(iter2 < iter1, false);

  iter1++;

  ASSERT_EQ(iter1 - iter2, 1);
  ASSERT_EQ(iter1 == iter2, false);
  ASSERT_EQ(iter2 < iter1, true);
  ASSERT_EQ(iter1 < iter2, false);

  iter2++;

  ASSERT_EQ(iter1 - iter2, 0);
  ASSERT_EQ(iter1 == iter2, true);
  ASSERT_EQ(iter1 < iter2, false);
  ASSERT_EQ(iter2 < iter1, false);

  iter1 += 100;
  iter2 += 100;

  ASSERT_EQ(iter1 - iter2, 0);
  ASSERT_EQ(iter1 == iter2, true);
  ASSERT_EQ(iter1 < iter2, false);
  ASSERT_EQ(iter2 < iter1, false);

  thrust::counting_iterator<float> iter3(0);
  thrust::counting_iterator<float> iter4(0.5);

  ASSERT_EQ(iter3 - iter4, 0);
  ASSERT_EQ(iter3 == iter4, true);
  ASSERT_EQ(iter3 < iter4, false);
  ASSERT_EQ(iter4 < iter3, false);

  iter3++; // iter3 = 1.0, iter4 = 0.5

  ASSERT_EQ(iter3 - iter4, 0);
  ASSERT_EQ(iter3 == iter4, true);
  ASSERT_EQ(iter3 < iter4, false);
  ASSERT_EQ(iter4 < iter3, false);

  iter4++; // iter3 = 1.0, iter4 = 1.5

  ASSERT_EQ(iter3 - iter4, 0);
  ASSERT_EQ(iter3 == iter4, true);
  ASSERT_EQ(iter3 < iter4, false);
  ASSERT_EQ(iter4 < iter3, false);

  iter4++; // iter3 = 1.0, iter4 = 2.5

  ASSERT_EQ(iter3 - iter4, -1);
  ASSERT_EQ(iter4 - iter3, 1);
  ASSERT_EQ(iter3 == iter4, false);
  ASSERT_EQ(iter3 < iter4, true);
  ASSERT_EQ(iter4 < iter3, false);
}

TEST(CountingIteratorTests, TestCountingIteratorDistance)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::counting_iterator<int> iter1(0);
  thrust::counting_iterator<int> iter2(5);

  ASSERT_EQ(thrust::distance(iter1, iter2), 5);

  iter1++;

  ASSERT_EQ(thrust::distance(iter1, iter2), 4);

  iter2 += 100;

  ASSERT_EQ(thrust::distance(iter1, iter2), 104);

  iter2 += 1000;

  ASSERT_EQ(thrust::distance(iter1, iter2), 1104);
}

TEST(CountingIteratorTests, TestCountingIteratorUnsignedType)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::counting_iterator<unsigned int> iter0(0);
  thrust::counting_iterator<unsigned int> iter1(5);

  ASSERT_EQ(iter1 - iter0, 5);
  ASSERT_EQ(iter0 - iter1, -5);
  ASSERT_EQ(iter0 != iter1, true);
  ASSERT_EQ(iter0 < iter1, true);
  ASSERT_EQ(iter1 < iter0, false);
}

TEST(CountingIteratorTests, TestCountingIteratorLowerBound)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  size_t n       = 10000;
  const size_t M = 100;

  for (auto seed : get_seeds())
  {
    SCOPED_TRACE(testing::Message() << "with seed= " << seed);

    thrust::host_vector<unsigned int> h_data = get_random_data<unsigned int>(
      n, get_default_limits<unsigned int>::min(), get_default_limits<unsigned int>::max(), seed);
    for (unsigned int i = 0; i < n; ++i)
    {
      h_data[i] %= M;
    }

    thrust::sort(h_data.begin(), h_data.end());

    thrust::device_vector<unsigned int> d_data = h_data;

    thrust::counting_iterator<unsigned int> search_begin(0);
    thrust::counting_iterator<unsigned int> search_end(M);

    thrust::host_vector<unsigned int> h_result(M);
    thrust::device_vector<unsigned int> d_result(M);

    thrust::lower_bound(h_data.begin(), h_data.end(), search_begin, search_end, h_result.begin());

    thrust::lower_bound(d_data.begin(), d_data.end(), search_begin, search_end, d_result.begin());

    ASSERT_EQ(h_result, d_result);
  }
}

TEST(CountingIteratorTests, TestCountingIteratorDifference)
{
  using Iterator   = thrust::counting_iterator<std::uint64_t>;
  using Difference = thrust::iterator_difference<Iterator>::type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  Difference diff = std::numeric_limits<std::uint32_t>::max() + 1;

  Iterator first(0);
  Iterator last = first + diff;

  ASSERT_EQ(diff, last - first);
}

THRUST_DIAG_POP
