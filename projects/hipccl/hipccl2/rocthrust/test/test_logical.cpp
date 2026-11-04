/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *  Modifications Copyright© 2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include <thrust/logical.h>

#include "test_param_fixtures.hpp"
#include "test_utils.hpp"

TESTS_DEFINE(LogicalTests, FullTestsParams);
TESTS_DEFINE(LogicalTestsOne, SignedIntegerTestsParams);

TYPED_TEST(LogicalTests, TestAllOf)
{
  using Vector = typename TestFixture::input_type;
  using T      = typename Vector::value_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  Vector v(3, T{1});

  ASSERT_EQ(thrust::all_of(v.begin(), v.end(), ::internal::identity{}), true);

  v[1] = T{0};

  ASSERT_EQ(thrust::all_of(v.begin(), v.end(), ::internal::identity{}), false);

  ASSERT_EQ(thrust::all_of(v.begin() + 0, v.begin() + 0, ::internal::identity{}), true);
  ASSERT_EQ(thrust::all_of(v.begin() + 0, v.begin() + 1, ::internal::identity{}), true);
  ASSERT_EQ(thrust::all_of(v.begin() + 0, v.begin() + 2, ::internal::identity{}), false);
  ASSERT_EQ(thrust::all_of(v.begin() + 1, v.begin() + 2, ::internal::identity{}), false);
}

template <class InputIterator, class Predicate>
bool all_of(my_system& system, InputIterator, InputIterator, Predicate)
{
  system.validate_dispatch();
  return false;
}

TEST(LogicalTestsOne, TestAllOfDispatchExplicit)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::device_vector<int> vec(1);

  my_system sys(0);
  thrust::all_of(sys, vec.begin(), vec.end(), 0);

  ASSERT_EQ(true, sys.is_valid());
}

template <class InputIterator, class Predicate>
bool all_of(my_tag, InputIterator first, InputIterator, Predicate)
{
  *first = 13;
  return false;
}

TEST(LogicalTestsOne, TestAllOfDispatchImplicit)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::device_vector<int> vec(1);

  thrust::all_of(thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.end()), 0);

  ASSERT_EQ(13, vec.front());
}

TYPED_TEST(LogicalTests, TestAnyOf)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  using Vector = typename TestFixture::input_type;
  using T      = typename Vector::value_type;

  Vector v(3, T{1});

  ASSERT_EQ(thrust::any_of(v.begin(), v.end(), ::internal::identity{}), true);

  v[1] = 0;

  ASSERT_EQ(thrust::any_of(v.begin(), v.end(), ::internal::identity{}), true);

  ASSERT_EQ(thrust::any_of(v.begin() + 0, v.begin() + 0, ::internal::identity{}), false);
  ASSERT_EQ(thrust::any_of(v.begin() + 0, v.begin() + 1, ::internal::identity{}), true);
  ASSERT_EQ(thrust::any_of(v.begin() + 0, v.begin() + 2, ::internal::identity{}), true);
  ASSERT_EQ(thrust::any_of(v.begin() + 1, v.begin() + 2, ::internal::identity{}), false);
}

template <class InputIterator, class Predicate>
bool any_of(my_system& system, InputIterator, InputIterator, Predicate)
{
  system.validate_dispatch();
  return false;
}

TEST(LogicalTestsOne, TestAnyOfDispatchExplicit)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::device_vector<int> vec(1);

  my_system sys(0);
  thrust::any_of(sys, vec.begin(), vec.end(), 0);

  ASSERT_EQ(true, sys.is_valid());
}

template <class InputIterator, class Predicate>
bool any_of(my_tag, InputIterator first, InputIterator, Predicate)
{
  *first = 13;
  return false;
}

TEST(LogicalTestsOne, TestAnyOfDispatchImplicit)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::device_vector<int> vec(1);

  thrust::any_of(thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.end()), 0);

  ASSERT_EQ(13, vec.front());
}

TYPED_TEST(LogicalTests, TestNoneOf)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  using Vector = typename TestFixture::input_type;
  using T      = typename Vector::value_type;

  Vector v(3, T{1});

  ASSERT_EQ(thrust::none_of(v.begin(), v.end(), ::internal::identity{}), false);

  v[1] = 0;

  ASSERT_EQ(thrust::none_of(v.begin(), v.end(), ::internal::identity{}), false);

  ASSERT_EQ(thrust::none_of(v.begin() + 0, v.begin() + 0, ::internal::identity{}), true);
  ASSERT_EQ(thrust::none_of(v.begin() + 0, v.begin() + 1, ::internal::identity{}), false);
  ASSERT_EQ(thrust::none_of(v.begin() + 0, v.begin() + 2, ::internal::identity{}), false);
  ASSERT_EQ(thrust::none_of(v.begin() + 1, v.begin() + 2, ::internal::identity{}), true);
}

template <class InputIterator, class Predicate>
bool none_of(my_system& system, InputIterator, InputIterator, Predicate)
{
  system.validate_dispatch();
  return false;
}

TEST(LogicalTestsOne, TestNoneOfDispatchExplicit)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::device_vector<int> vec(1);

  my_system sys(0);
  thrust::none_of(sys, vec.begin(), vec.end(), 0);

  ASSERT_EQ(true, sys.is_valid());
}

template <class InputIterator, class Predicate>
bool none_of(my_tag, InputIterator first, InputIterator, Predicate)
{
  *first = 13;
  return false;
}

TEST(LogicalTestsOne, TestNoneOfDispatchImplicit)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::device_vector<int> vec(1);

  thrust::none_of(thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.end()), 0);

  ASSERT_EQ(13, vec.front());
}
