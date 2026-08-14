/*
 *  Copyright 2025 NVIDIA Corporation
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

#include <thrust/detail/libcxx_wrapper/std/__iterator/iterator_traits.h>
#include <thrust/distance.h>
#include <thrust/iterator/offset_iterator.h>
#include <thrust/universal_vector.h>

#include "test_param_fixtures.hpp"
#include "test_real_assertions.hpp"
#include "test_utils.hpp"

#include _THRUST_STD_INCLUDE(iterator)

#if !_THRUST_HAS_DEVICE_SYSTEM_STD
#  include <type_traits>
#endif

using VectorTestsParams = ::testing::Types<
  Params<thrust::host_vector<signed char>>,
  Params<thrust::host_vector<short>>,
  Params<thrust::host_vector<int>>,
  Params<thrust::host_vector<float>>,
  Params<thrust::host_vector<int, thrust::mr::stateless_resource_allocator<int, thrust::host_memory_resource>>>,
  Params<thrust::device_vector<signed char>>,
  Params<thrust::device_vector<short>>,
  Params<thrust::device_vector<int>>,
  Params<thrust::device_vector<float>>,
  Params<thrust::device_vector<int, thrust::mr::stateless_resource_allocator<int, thrust::device_memory_resource>>>,
  Params<thrust::universal_vector<int>>,
  Params<thrust::universal_host_pinned_vector<int>>>;

TESTS_DEFINE(OffsetIteratorTests, VectorTestsParams);

// ensure that we properly support thrust::counting_iterator from _THRUST_STD
TEST(OffsetIteratorTests, TestOffsetIteratorTraits)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  using base_it    = thrust::host_vector<int>::iterator;
  using it         = thrust::offset_iterator<base_it>;
  using traits     = _THRUST_STD::iterator_traits<it>;
  using vec_traits = _THRUST_STD::iterator_traits<base_it>;

  static_assert(_THRUST_STD::is_same_v<traits::difference_type, vec_traits::difference_type>);
  static_assert(_THRUST_STD::is_same_v<traits::value_type, vec_traits::value_type>);
  static_assert(_THRUST_STD::is_same_v<traits::pointer, vec_traits::pointer>);
  static_assert(_THRUST_STD::is_same_v<traits::reference, vec_traits::reference>);
  static_assert(_THRUST_STD::is_same_v<traits::iterator_category, vec_traits::iterator_category>);

  static_assert(_THRUST_STD::is_same_v<thrust::iterator_traversal_t<it>, thrust::random_access_traversal_tag>);

  static_assert(::internal::is_cpp17_random_access_iterator<it>::value);

#if _THRUST_HAS_DEVICE_SYSTEM_STD || THRUST_STD_VER >= 2020
  static_assert(_THRUST_STD::output_iterator<it, int>);
  static_assert(_THRUST_STD::input_iterator<it>);
  static_assert(_THRUST_STD::forward_iterator<it>);
  static_assert(_THRUST_STD::bidirectional_iterator<it>);
  static_assert(_THRUST_STD::random_access_iterator<it>);
  static_assert(!_THRUST_STD::contiguous_iterator<it>);
#endif
}

TYPED_TEST(OffsetIteratorTests, TestOffsetConstructor)
{
  using Vector = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::offset_iterator<int*> iter0;
  ASSERT_EQ(iter0.base(), static_cast<int*>(nullptr));
  ASSERT_EQ(iter0.offset(), 0);

  Vector v{42, 43};
  thrust::offset_iterator iter1(v.begin());
  ASSERT_EQ_QUIET(iter1.base(), v.begin());
  ASSERT_EQ(iter1.offset(), 0);
  ASSERT_EQ(*iter1, 42);

  thrust::offset_iterator iter2(v.begin(), 1);
  ASSERT_EQ_QUIET(iter2.base(), v.begin());
  ASSERT_EQ(iter2.offset(), 1);
  ASSERT_EQ(*iter2, 43);

  ptrdiff_t offset = 1;
  thrust::offset_iterator iter3(v.begin(), &offset);
  ASSERT_EQ_QUIET(iter3.base(), v.begin());
  ASSERT_EQ(iter3.offset(), &offset);
  ASSERT_EQ(*iter3.offset(), 1);
  ASSERT_EQ(*iter3, 43);
}

TYPED_TEST(OffsetIteratorTests, TestOffsetIteratorCopyConstructorAndAssignment)
{
  using Vector = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  Vector v{42, 43};

  // value offset
  {
    thrust::offset_iterator iter0(v.begin());
#if THRUST_COMPILER(MSVC) // MSVC cannot deduce the template arguments from the copy ctor
    decltype(iter0) iter1(iter0);
#else // THRUST_COMPILER(MSVC)
    thrust::offset_iterator iter1(iter0);
#endif // THRUST_COMPILER(MSVC)
    ASSERT_EQ(iter0 == iter1, true);
    ASSERT_EQ(*iter0 == *iter1, true);

    thrust::offset_iterator iter2(v.begin() + 1);
    ASSERT_EQ(iter0 != iter2, true);
    ASSERT_EQ(*iter0 != *iter2, true);

    iter2 = iter0;
    ASSERT_EQ(iter0 == iter2, true);
    ASSERT_EQ(*iter0 == *iter2, true);
  }

  // indirect offset
  {
    const typename Vector::iterator::difference_type offset = 0;
    thrust::offset_iterator iter0(v.begin(), &offset);

#if THRUST_COMPILER(MSVC) // MSVC cannot deduce the template arguments from the copy ctor
    decltype(iter0) iter1(iter0);
#else // THRUST_COMPILER(MSVC)
    thrust::offset_iterator iter1(iter0);
#endif // THRUST_COMPILER(MSVC)
    ASSERT_EQ(iter0 == iter1, true);
    ASSERT_EQ(*iter0 == *iter1, true);

    thrust::offset_iterator iter2(v.begin() + 1, &offset);
    ASSERT_EQ(iter0 != iter2, true);
    ASSERT_EQ(*iter0 != *iter2, true);

    iter2 = iter0;
    ASSERT_EQ(iter0 == iter2, true);
    ASSERT_EQ(*iter0 == *iter2, true);
  }
}

TYPED_TEST(OffsetIteratorTests, TestOffsetIteratorIncrement)
{
  using Vector = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  auto test = [](auto iter) {
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
  };

  const Vector v{-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8};
  test(thrust::offset_iterator(v.begin() + 1, 1));
  const typename Vector::iterator::difference_type offset = 1;
  test(thrust::offset_iterator(v.begin() + 1, &offset));
}

TYPED_TEST(OffsetIteratorTests, TestOffsetIteratorMutation)
{
  using Vector = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  {
    Vector v{-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8};
    thrust::offset_iterator it(v.begin() + 1, 1);
    *it = 42;
    ++it;
    *it = 43;
    ++it.offset();
    *it = 44;
    ASSERT_EQ(v, (Vector{-2, -1, 42, 43, 44, 3, 4, 5, 6, 7, 8}));
  }
  {
    Vector v{-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8};
    typename Vector::iterator::difference_type offset = 1;
    thrust::offset_iterator it(v.begin() + 1, &offset);
    *it = 42;
    ++it;
    *it    = 43;
    offset = 2;
    *it    = 44;
    ASSERT_EQ(v, (Vector{-2, -1, 42, 43, 44, 3, 4, 5, 6, 7, 8}));
  }
}

TYPED_TEST(OffsetIteratorTests, TestOffsetIteratorComparisonAndDistance)
{
  using Vector = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  auto test = [](auto iter1, auto iter2) {
    ASSERT_EQ(iter1 == iter2, true);
    ASSERT_EQ(iter1 - iter2, 0);
    ASSERT_EQ(thrust::distance(iter1, iter2), 0);

    iter1++;
    ASSERT_EQ(iter1 == iter2, false);
    ASSERT_EQ(iter1 - iter2, 1);
    ASSERT_EQ(thrust::distance(iter1, iter2), -1);

    iter2++;
    ASSERT_EQ(iter1 == iter2, true);
    ASSERT_EQ(iter1 - iter2, 0);
    ASSERT_EQ(thrust::distance(iter1, iter2), 0);

    iter1 += 100;
    iter2 += 100;
    ASSERT_EQ(iter1 == iter2, true);
    ASSERT_EQ(iter1 - iter2, 0);
    ASSERT_EQ(thrust::distance(iter1, iter2), 0);

    iter1 -= 5;
    ASSERT_EQ(iter1 == iter2, false);
    ASSERT_EQ(iter1 - iter2, -5);
    ASSERT_EQ(thrust::distance(iter1, iter2), 5);
  };

  Vector v(101);
  test(thrust::offset_iterator(v.begin()), thrust::offset_iterator(v.begin()));
  const typename Vector::iterator::difference_type offset = 0;
  test(thrust::offset_iterator(v.begin(), &offset), thrust::offset_iterator(v.begin(), &offset));
}

TYPED_TEST(OffsetIteratorTests, TestOffsetIteratorLateValue)
{
  using Vector = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  typename Vector::difference_type offset;
  Vector v{0, 1, 2, 3, 4, 5, 6, 7, 8};
  thrust::offset_iterator iter(v.begin(), &offset);
  offset = 2; // we provide the offset value **after** constructing the iterator
  ASSERT_EQ(*iter, 2);
}

TYPED_TEST(OffsetIteratorTests, TestOffsetIteratorIndirectValueFancyIterator)
{
  using Vector = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  using thrust::placeholders::_1;

  Vector v{0, 1, 2, 3, 4, 5, 6, 7, 8};
  thrust::device_vector<typename Vector::difference_type> offsets{2};
  auto it = thrust::make_transform_iterator(offsets.begin(), _1 * 3);
  thrust::offset_iterator iter(v.begin(), it);
  ASSERT_EQ(*iter, 6);
}
