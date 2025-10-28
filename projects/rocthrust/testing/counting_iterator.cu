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
#include <thrust/detail/libcxx_wrapper/std/__iterator/iterator_traits.h>
#include <thrust/distance.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sort.h>

#include <cstdint>

#include <unittest/unittest.h>

#include _THRUST_STD_INCLUDE(iterator)
#include _THRUST_STD_INCLUDE(type_traits)

template <typename ValueType, typename DifferenceType>
inline constexpr bool diff_type_is =
  _THRUST_STD::is_same_v<typename thrust::counting_iterator<ValueType>::difference_type, DifferenceType>;

static_assert(diff_type_is<int8_t, int>);
static_assert(diff_type_is<uint8_t, int>);
static_assert(diff_type_is<int16_t, int>);
static_assert(diff_type_is<uint16_t, int>);
static_assert(diff_type_is<int32_t, ptrdiff_t>);
static_assert(diff_type_is<uint32_t, ptrdiff_t>);
static_assert(diff_type_is<int64_t, ptrdiff_t>);
static_assert(diff_type_is<uint64_t, ptrdiff_t>);
#if !defined(THRUST_DISABLE_INT128_SUPPORT) && (defined(__linux__) || defined(__LP64__)) \
  && ((THRUST_COMPILER(NVRTC) && defined(__CUDACC_RTC_INT128__)) || defined(__SIZEOF_INT128__))
static_assert(diff_type_is<__int128_t, ptrdiff_t>);
static_assert(diff_type_is<__uint128_t, ptrdiff_t>);
#endif
static_assert(diff_type_is<float, ptrdiff_t>);
static_assert(diff_type_is<double, ptrdiff_t>);

struct custom_int
{
  THRUST_HOST_DEVICE custom_int(int) {}
  THRUST_HOST_DEVICE operator int() const;
};
static_assert(thrust::detail::is_numeric<custom_int>::value);

static_assert(diff_type_is<custom_int, ptrdiff_t>);

THRUST_DIAG_PUSH
THRUST_DIAG_SUPPRESS_MSVC(4244 4267) // possible loss of data

// ensure that we properly support thrust::counting_iterator from _THRUST_STD
void TestCountingIteratorTraits()
{
  using it       = thrust::counting_iterator<int>;
  using traits   = _THRUST_STD::iterator_traits<it>;
  using category = thrust::detail::iterator_category_with_system_and_traversal<_THRUST_STD::random_access_iterator_tag,
                                                                               thrust::any_system_tag,
                                                                               thrust::random_access_traversal_tag>;

  static_assert(_THRUST_STD::is_same_v<traits::difference_type, ptrdiff_t>);
  static_assert(_THRUST_STD::is_same_v<traits::value_type, int>);
  static_assert(_THRUST_STD::is_same_v<traits::pointer, void>);
  static_assert(_THRUST_STD::is_same_v<traits::reference, signed int>);
  static_assert(_THRUST_STD::is_same_v<traits::iterator_category, category>);

  static_assert(_THRUST_STD::is_same_v<thrust::iterator_traversal_t<it>, thrust::random_access_traversal_tag>);

  static_assert(::internal::is_cpp17_random_access_iterator<it>::value);

#if _THRUST_HAS_DEVICE_SYSTEM_STD || THRUST_STD_VER >= 2020
  static_assert(!_THRUST_STD::output_iterator<it, int>);
  static_assert(_THRUST_STD::input_iterator<it>);
  static_assert(_THRUST_STD::forward_iterator<it>);
  static_assert(_THRUST_STD::bidirectional_iterator<it>);
  static_assert(_THRUST_STD::random_access_iterator<it>);
  static_assert(!_THRUST_STD::contiguous_iterator<it>);
#endif
}
DECLARE_UNITTEST(TestCountingIteratorTraits);

template <typename T>
void TestCountingDefaultConstructor()
{
  thrust::counting_iterator<T> iter0;
  ASSERT_EQUAL(*iter0, T{});
}
DECLARE_GENERIC_UNITTEST(TestCountingDefaultConstructor);

void TestCountingIteratorCopyConstructor()
{
  thrust::counting_iterator<int> iter0(100);

  thrust::counting_iterator<int> iter1(iter0);

  ASSERT_EQUAL_QUIET(iter0, iter1);
  ASSERT_EQUAL(*iter0, *iter1);

  // construct from related space
  thrust::counting_iterator<int, thrust::host_system_tag> h_iter = iter0;
  ASSERT_EQUAL(*iter0, *h_iter);

  thrust::counting_iterator<int, thrust::device_system_tag> d_iter = iter0;
  ASSERT_EQUAL(*iter0, *d_iter);
}
DECLARE_UNITTEST(TestCountingIteratorCopyConstructor);
static_assert(_THRUST_STD::is_trivially_copy_constructible<thrust::counting_iterator<int>>::value, "");
static_assert(_THRUST_STD::is_trivially_copyable<thrust::counting_iterator<int>>::value, "");

void TestCountingIteratorIncrement()
{
  thrust::counting_iterator<int> iter(0);

  ASSERT_EQUAL(*iter, 0);

  iter++;

  ASSERT_EQUAL(*iter, 1);

  iter++;
  iter++;

  ASSERT_EQUAL(*iter, 3);

  iter += 5;

  ASSERT_EQUAL(*iter, 8);

  iter -= 10;

  ASSERT_EQUAL(*iter, -2);
}
DECLARE_UNITTEST(TestCountingIteratorIncrement);

void TestCountingIteratorComparison()
{
  thrust::counting_iterator<int> iter1(0);
  thrust::counting_iterator<int> iter2(0);

  ASSERT_EQUAL(iter1 - iter2, 0);
  ASSERT_EQUAL(iter1 == iter2, true);

  iter1++;

  ASSERT_EQUAL(iter1 - iter2, 1);
  ASSERT_EQUAL(iter1 == iter2, false);

  iter2++;

  ASSERT_EQUAL(iter1 - iter2, 0);
  ASSERT_EQUAL(iter1 == iter2, true);

  iter1 += 100;
  iter2 += 100;

  ASSERT_EQUAL(iter1 - iter2, 0);
  ASSERT_EQUAL(iter1 == iter2, true);
}
DECLARE_UNITTEST(TestCountingIteratorComparison);

void TestCountingIteratorFloatComparison()
{
  thrust::counting_iterator<float> iter1(0);
  thrust::counting_iterator<float> iter2(0);

  ASSERT_EQUAL(iter1 - iter2, 0);
  ASSERT_EQUAL(iter1 == iter2, true);
  ASSERT_EQUAL(iter1 < iter2, false);
  ASSERT_EQUAL(iter2 < iter1, false);

  iter1++;

  ASSERT_EQUAL(iter1 - iter2, 1);
  ASSERT_EQUAL(iter1 == iter2, false);
  ASSERT_EQUAL(iter2 < iter1, true);
  ASSERT_EQUAL(iter1 < iter2, false);

  iter2++;

  ASSERT_EQUAL(iter1 - iter2, 0);
  ASSERT_EQUAL(iter1 == iter2, true);
  ASSERT_EQUAL(iter1 < iter2, false);
  ASSERT_EQUAL(iter2 < iter1, false);

  iter1 += 100;
  iter2 += 100;

  ASSERT_EQUAL(iter1 - iter2, 0);
  ASSERT_EQUAL(iter1 == iter2, true);
  ASSERT_EQUAL(iter1 < iter2, false);
  ASSERT_EQUAL(iter2 < iter1, false);

  thrust::counting_iterator<float> iter3(0);
  thrust::counting_iterator<float> iter4(0.5);

  ASSERT_EQUAL(iter3 - iter4, 0);
  ASSERT_EQUAL(iter3 == iter4, true);
  ASSERT_EQUAL(iter3 < iter4, false);
  ASSERT_EQUAL(iter4 < iter3, false);

  iter3++; // iter3 = 1.0, iter4 = 0.5

  ASSERT_EQUAL(iter3 - iter4, 0);
  ASSERT_EQUAL(iter3 == iter4, true);
  ASSERT_EQUAL(iter3 < iter4, false);
  ASSERT_EQUAL(iter4 < iter3, false);

  iter4++; // iter3 = 1.0, iter4 = 1.5

  ASSERT_EQUAL(iter3 - iter4, 0);
  ASSERT_EQUAL(iter3 == iter4, true);
  ASSERT_EQUAL(iter3 < iter4, false);
  ASSERT_EQUAL(iter4 < iter3, false);

  iter4++; // iter3 = 1.0, iter4 = 2.5

  ASSERT_EQUAL(iter3 - iter4, -1);
  ASSERT_EQUAL(iter4 - iter3, 1);
  ASSERT_EQUAL(iter3 == iter4, false);
  ASSERT_EQUAL(iter3 < iter4, true);
  ASSERT_EQUAL(iter4 < iter3, false);
}
DECLARE_UNITTEST(TestCountingIteratorFloatComparison);

void TestCountingIteratorDistance()
{
  thrust::counting_iterator<int> iter1(0);
  thrust::counting_iterator<int> iter2(5);

  ASSERT_EQUAL(thrust::distance(iter1, iter2), 5);

  iter1++;

  ASSERT_EQUAL(thrust::distance(iter1, iter2), 4);

  iter2 += 100;

  ASSERT_EQUAL(thrust::distance(iter1, iter2), 104);
}
DECLARE_UNITTEST(TestCountingIteratorDistance);

void TestCountingIteratorUnsignedType()
{
  thrust::counting_iterator<unsigned int> iter0(0);
  thrust::counting_iterator<unsigned int> iter1(5);

  ASSERT_EQUAL(iter1 - iter0, 5);
  ASSERT_EQUAL(iter0 - iter1, -5);
  ASSERT_EQUAL(iter0 != iter1, true);
  ASSERT_EQUAL(iter0 < iter1, true);
  ASSERT_EQUAL(iter1 < iter0, false);
}
DECLARE_UNITTEST(TestCountingIteratorUnsignedType);

void TestCountingIteratorLowerBound()
{
  size_t n       = 10000;
  const size_t M = 100;

  thrust::host_vector<unsigned int> h_data = unittest::random_integers<unsigned int>(n);
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

  ASSERT_EQUAL(h_result, d_result);
}
DECLARE_UNITTEST(TestCountingIteratorLowerBound);

void TestCountingIteratorDifference()
{
  using Iterator   = thrust::counting_iterator<std::uint64_t>;
  using Difference = thrust::detail::it_difference_t<Iterator>;

  Difference diff = std::numeric_limits<std::uint32_t>::max() + 1;

  Iterator first(0);
  Iterator last = first + diff;

  ASSERT_EQUAL(diff, last - first);
}
DECLARE_UNITTEST(TestCountingIteratorDifference);

THRUST_DIAG_POP
