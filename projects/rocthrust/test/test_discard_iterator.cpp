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

#include <thrust/iterator/detail/iterator_traits.h>
#include <thrust/iterator/discard_iterator.h>

#include "test_param_fixtures.hpp"
#include "test_utils.hpp"

#include _THRUST_STD_INCLUDE(type_traits)

#if !_THRUST_HAS_DEVICE_SYSTEM_STD
#  include <iterator>
#endif

// ensure that we properly support thrust::discard_iterator from _THRUST_STD
TEST(DiscardIteratorTests, TestDiscardIteratorTraits)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  using it       = thrust::discard_iterator<>;
  using traits   = _THRUST_STD::iterator_traits<it>;
  using category = thrust::detail::iterator_category_with_system_and_traversal<_THRUST_STD::random_access_iterator_tag,
                                                                               thrust::any_system_tag,
                                                                               thrust::random_access_traversal_tag>;

  static_assert(_THRUST_STD::is_same_v<traits::difference_type, ptrdiff_t>);
  static_assert(_THRUST_STD::is_same_v<traits::value_type, thrust::detail::any_assign>);
  static_assert(_THRUST_STD::is_same_v<traits::pointer, void>);
  static_assert(_THRUST_STD::is_same_v<traits::reference, thrust::detail::any_assign&>);
  static_assert(_THRUST_STD::is_same_v<traits::iterator_category, category>);

  static_assert(_THRUST_STD::is_same_v<thrust::iterator_traversal_t<it>, thrust::random_access_traversal_tag>);

  static_assert(::thrust::detail::is_cpp17_random_access_iterator<it>::value);

#if _THRUST_HAS_DEVICE_SYSTEM_STD || THRUST_CPP_DIALECT >= 2020
  static_assert(_THRUST_STD::output_iterator<it, int>);
  static_assert(_THRUST_STD::input_iterator<it>);
  static_assert(_THRUST_STD::forward_iterator<it>);
  static_assert(_THRUST_STD::bidirectional_iterator<it>);
  static_assert(_THRUST_STD::random_access_iterator<it>);
  static_assert(!_THRUST_STD::contiguous_iterator<it>);
#endif
}

TEST(DiscardIteratorTests, TestDiscardIteratorIncrement)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::discard_iterator<> lhs(0);
  thrust::discard_iterator<> rhs(0);

  ASSERT_EQ(0, lhs - rhs);

  lhs++;

  ASSERT_EQ(1, lhs - rhs);

  lhs++;
  lhs++;

  ASSERT_EQ(3, lhs - rhs);

  lhs += 5;

  ASSERT_EQ(8, lhs - rhs);

  lhs -= 10;

  ASSERT_EQ(-2, lhs - rhs);
}
static_assert(_THRUST_STD::is_trivially_copy_constructible<thrust::discard_iterator<>>::value, "");
static_assert(_THRUST_STD::is_trivially_copyable<thrust::discard_iterator<>>::value, "");

TEST(DiscardIteratorTests, TestDiscardIteratorComparison)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::discard_iterator<> iter1(0);
  thrust::discard_iterator<> iter2(0);

  ASSERT_EQ(0, iter1 - iter2);
  ASSERT_EQ(true, iter1 == iter2);

  iter1++;

  ASSERT_EQ(1, iter1 - iter2);
  ASSERT_EQ(false, iter1 == iter2);

  iter2++;

  ASSERT_EQ(0, iter1 - iter2);
  ASSERT_EQ(true, iter1 == iter2);

  iter1 += 100;
  iter2 += 100;

  ASSERT_EQ(0, iter1 - iter2);
  ASSERT_EQ(true, iter1 == iter2);
}

TEST(DiscardIteratorTests, TestMakeDiscardIterator)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::discard_iterator<> iter0 = thrust::make_discard_iterator(13);

  *iter0 = 7;

  thrust::discard_iterator<> iter1 = thrust::make_discard_iterator(7);

  *iter1 = 13;

  ASSERT_EQ(6, iter0 - iter1);
}

TEST(DiscardIteratorTests, TestZippedDiscardIterator)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  using namespace thrust;

  using IteratorTuple1 = tuple<discard_iterator<>>;
  using ZipIterator1   = zip_iterator<IteratorTuple1>;

  IteratorTuple1 t = thrust::make_tuple(thrust::make_discard_iterator());

  ZipIterator1 z_iter1_first = thrust::make_zip_iterator(t);
  ZipIterator1 z_iter1_last  = z_iter1_first + 10;
  for (; z_iter1_first != z_iter1_last; ++z_iter1_first)
  {
    ;
  }

  ASSERT_EQ(10, thrust::get<0>(z_iter1_first.get_iterator_tuple()) - thrust::make_discard_iterator());

  using IteratorTuple2 = tuple<int*, discard_iterator<>>;
  using ZipIterator2   = zip_iterator<IteratorTuple2>;

  ZipIterator2 z_iter_first = thrust::make_zip_iterator(thrust::make_tuple((int*) 0, thrust::make_discard_iterator()));
  ZipIterator2 z_iter_last  = z_iter_first + 10;

  for (; z_iter_first != z_iter_last; ++z_iter_first)
  {
    ;
  }

  ASSERT_EQ(10, thrust::get<1>(z_iter_first.get_iterator_tuple()) - thrust::make_discard_iterator());
}

TEST(DiscardIteratorTests, UsingHip)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  ASSERT_EQ(THRUST_DEVICE_SYSTEM, THRUST_DEVICE_SYSTEM_HIP);
}
