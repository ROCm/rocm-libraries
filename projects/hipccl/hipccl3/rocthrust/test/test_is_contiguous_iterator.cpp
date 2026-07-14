/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *  Modifications Copyright© 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include <thrust/detail/static_assert.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/type_traits/is_contiguous_iterator.h>
#include <thrust/type_traits/unwrap_contiguous_iterator.h>

#include <array>
#include <deque>
#include <iterator>
#include <list>
#include <map>
#include <set>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "test_param_fixtures.hpp"
#include "test_utils.hpp"

#if !_THRUST_HAS_DEVICE_SYSTEM_STD
#  include <type_traits>
#  include <utility>
#endif

TESTS_DEFINE(IsContiguousIteratorTests, FullTestsParams);

static_assert(thrust::is_contiguous_iterator_v<std::string::iterator>);
static_assert(thrust::is_contiguous_iterator_v<std::wstring::iterator>);
static_assert(thrust::is_contiguous_iterator_v<std::string_view::iterator>);
static_assert(thrust::is_contiguous_iterator_v<std::wstring_view::iterator>);
static_assert(!thrust::is_contiguous_iterator_v<std::vector<bool>::iterator>);

TYPED_TEST(IsContiguousIteratorTests, test_is_contiguous_iterator)
{
  using Vector = typename TestFixture::input_type;
  using T      = typename Vector::value_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  static_assert(thrust::is_contiguous_iterator_v<T*>);
  static_assert(thrust::is_contiguous_iterator_v<T const*>);
  static_assert(thrust::is_contiguous_iterator_v<thrust::device_ptr<T>>);
  static_assert(thrust::is_contiguous_iterator_v<typename std::vector<T>::iterator>);
  static_assert(!thrust::is_contiguous_iterator_v<typename std::vector<T>::reverse_iterator>);
  static_assert(thrust::is_contiguous_iterator_v<typename std::array<T, 1>::iterator>);
  static_assert(!thrust::is_contiguous_iterator_v<typename std::list<T>::iterator>);
  static_assert(!thrust::is_contiguous_iterator_v<typename std::deque<T>::iterator>);
  static_assert(!thrust::is_contiguous_iterator_v<typename std::set<T>::iterator>);
  static_assert(!thrust::is_contiguous_iterator_v<typename std::multiset<T>::iterator>);
  static_assert(!thrust::is_contiguous_iterator_v<typename std::map<T, T>::iterator>);
  static_assert(!thrust::is_contiguous_iterator_v<typename std::multimap<T, T>::iterator>);
  static_assert(!thrust::is_contiguous_iterator_v<typename std::unordered_set<T>::iterator>);
  static_assert(!thrust::is_contiguous_iterator_v<typename std::unordered_multiset<T>::iterator>);
  static_assert(!thrust::is_contiguous_iterator_v<typename std::unordered_map<T, T>::iterator>);
  static_assert(!thrust::is_contiguous_iterator_v<typename std::unordered_multimap<T, T>::iterator>);
  static_assert(!thrust::is_contiguous_iterator_v<std::istream_iterator<T>>);
  static_assert(!thrust::is_contiguous_iterator_v<std::ostream_iterator<T>>);
}

TYPED_TEST(IsContiguousIteratorTests, test_is_contiguous_iterator_cvref)
{
  using Vector = typename TestFixture::input_type;
  using T      = typename Vector::value_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  static_assert(thrust::is_contiguous_iterator_v<T* const>);
  static_assert(thrust::is_contiguous_iterator_v<T* volatile>);
  static_assert(thrust::is_contiguous_iterator_v<T*&>);
  static_assert(thrust::is_contiguous_iterator_v<T* const&>);
  static_assert(thrust::is_contiguous_iterator_v<T* volatile&>);

  static_assert(!thrust::is_contiguous_iterator_v<std::vector<bool>::iterator const>);
  static_assert(!thrust::is_contiguous_iterator_v<std::vector<bool>::iterator volatile>);
  static_assert(!thrust::is_contiguous_iterator_v<std::vector<bool>::iterator&>);
  static_assert(!thrust::is_contiguous_iterator_v<std::vector<bool>::iterator const&>);
  static_assert(!thrust::is_contiguous_iterator_v<std::vector<bool>::iterator volatile&>);
}

TYPED_TEST(IsContiguousIteratorTests, test_is_contiguous_iterator_vectors)
{
  using Vector = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  static_assert(thrust::is_contiguous_iterator_v<typename Vector::iterator>);
}

struct expect_pointer
{};
struct expect_passthrough
{};

template <typename IteratorT, typename PointerT, typename expected_unwrapped_type /* = expect_[pointer|passthrough] */>
struct check_unwrapped_iterator
{
  using unwrapped_t =
    _THRUST_STD::remove_reference_t<decltype(thrust::try_unwrap_contiguous_iterator(_THRUST_STD::declval<IteratorT>()))>;

  static constexpr bool value =
    std::is_same<expected_unwrapped_type, expect_pointer>::value
      ? std::is_same<unwrapped_t, PointerT>::value
      : std::is_same<unwrapped_t, IteratorT>::value;
};

TYPED_TEST(IsContiguousIteratorTests, test_try_unwrap_contiguous_iterator)
{
  using Vector = typename TestFixture::input_type;
  using T      = typename Vector::value_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  // Raw pointers should pass whether expecting pointers or passthrough.
  static_assert(check_unwrapped_iterator<T*, T*, expect_pointer>::value);
  static_assert(check_unwrapped_iterator<T*, T*, expect_passthrough>::value);
  static_assert(check_unwrapped_iterator<T const*, T const*, expect_pointer>::value);
  static_assert(check_unwrapped_iterator<T const*, T const*, expect_passthrough>::value);

  static_assert(check_unwrapped_iterator<thrust::device_ptr<T>, T*, expect_pointer>::value);
  static_assert(check_unwrapped_iterator<thrust::device_ptr<T const>, T const*, expect_pointer>::value);
  static_assert(check_unwrapped_iterator<typename std::vector<T>::iterator, T*, expect_pointer>::value);
  static_assert(check_unwrapped_iterator<typename std::vector<T>::reverse_iterator, T*, expect_passthrough>::value);
  static_assert(check_unwrapped_iterator<typename std::array<T, 1>::iterator, T*, expect_pointer>::value);
  static_assert(check_unwrapped_iterator<typename std::array<T const, 1>::iterator, T const*, expect_pointer>::value);
  static_assert(check_unwrapped_iterator<typename std::list<T>::iterator, T*, expect_passthrough>::value);
  static_assert(check_unwrapped_iterator<typename std::deque<T>::iterator, T*, expect_passthrough>::value);
  static_assert(check_unwrapped_iterator<typename std::set<T>::iterator, T*, expect_passthrough>::value);
  static_assert(check_unwrapped_iterator<typename std::multiset<T>::iterator, T*, expect_passthrough>::value);
  static_assert(
    check_unwrapped_iterator<typename std::map<T, T>::iterator, std::pair<T const, T>*, expect_passthrough>::value);
  static_assert(
    check_unwrapped_iterator<typename std::multimap<T, T>::iterator, std::pair<T const, T>*, expect_passthrough>::value);
  static_assert(check_unwrapped_iterator<typename std::unordered_set<T>::iterator, T*, expect_passthrough>::value);
  static_assert(check_unwrapped_iterator<typename std::unordered_multiset<T>::iterator, T*, expect_passthrough>::value);
  static_assert(
    check_unwrapped_iterator<typename std::unordered_map<T, T>::iterator, std::pair<T const, T>*, expect_passthrough>::
      value);
  static_assert(check_unwrapped_iterator<typename std::unordered_multimap<T, T>::iterator,
                                         std::pair<T const, T>*,
                                         expect_passthrough>::value);
  static_assert(check_unwrapped_iterator<std::istream_iterator<T>, T*, expect_passthrough>::value);
  static_assert(check_unwrapped_iterator<std::ostream_iterator<T>, void, expect_passthrough>::value);
}
