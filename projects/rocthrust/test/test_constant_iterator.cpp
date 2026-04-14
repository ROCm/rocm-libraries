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

#include <thrust/copy.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>

#include "test_param_fixtures.hpp"
#include "test_utils.hpp"

#include _THRUST_STD_INCLUDE(type_traits)

TESTS_DEFINE(ConstantIteratorTests, VectorSignedTestsParams);

TEST(ConstantIteratorTests, UsingHip)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  ASSERT_EQ(THRUST_DEVICE_SYSTEM, THRUST_DEVICE_SYSTEM_HIP);
}

TEST(ConstantIteratorTests, TestConstantIteratorConstructFromConvertibleSystem)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  using namespace thrust;

  constant_iterator<int> default_system(13);

  constant_iterator<int, use_default, host_system_tag> host_system = default_system;
  ASSERT_EQ(*default_system, *host_system);

  constant_iterator<int, use_default, device_system_tag> device_system = default_system;
  ASSERT_EQ(*default_system, *device_system);
}

TEST(ConstantIteratorTests, TestConstantIteratorIncrement)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  using namespace thrust;

  constant_iterator<int> lhs(0, 0);
  constant_iterator<int> rhs(0, 0);

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
static_assert(_THRUST_STD::is_trivially_copy_constructible<thrust::constant_iterator<int>>::value, "");
static_assert(_THRUST_STD::is_trivially_copyable<thrust::constant_iterator<int>>::value, "");

TEST(ConstantIteratorTests, TestConstantIteratorIncrementBig)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  long long int n = 10000000000ULL;

  thrust::constant_iterator<long long int> begin(1);
  thrust::constant_iterator<long long int> end = begin + n;

  ASSERT_EQ(thrust::distance(begin, end), n);
}

TEST(ConstantIteratorTests, TestConstantIteratorComparison)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  using namespace thrust;

  constant_iterator<int> iter1(0);
  constant_iterator<int> iter2(0);

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

TEST(ConstantIteratorTests, TestMakeConstantIterator)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  using namespace thrust;

  // test one argument version
  constant_iterator<int> iter0 = make_constant_iterator<int>(13);

  ASSERT_EQ(13, *iter0);

  // test two argument version
  constant_iterator<int, thrust::detail::intmax_t> iter1 = make_constant_iterator<int, thrust::detail::intmax_t>(13, 7);

  ASSERT_EQ(13, *iter1);
  ASSERT_EQ(7, iter1 - iter0);
}

TYPED_TEST(ConstantIteratorTests, TestConstantIteratorCopy)
{
  using Vector = typename TestFixture::input_type;
  using namespace thrust;

  using ValueType = typename Vector::value_type;
  using ConstIter = constant_iterator<ValueType>;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  Vector result(4);

  ConstIter first = make_constant_iterator<ValueType>(7);
  ConstIter last  = first + result.size();
  thrust::copy(first, last, result.begin());

  Vector ref(4, 7);
  ASSERT_EQ(ref, result);
}

TYPED_TEST(ConstantIteratorTests, TestConstantIteratorTransform)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  using Vector = typename TestFixture::input_type;
  using namespace thrust;

  using T         = typename Vector::value_type;
  using ConstIter = constant_iterator<T>;

  Vector result(4);

  ConstIter first1 = make_constant_iterator<T>(7);
  ConstIter last1  = first1 + result.size();
  ConstIter first2 = make_constant_iterator<T>(3);

  thrust::transform(first1, last1, result.begin(), thrust::negate<T>());

  Vector ref(4, -7);
  ASSERT_EQ(ref, result);

  thrust::transform(first1, last1, first2, result.begin(), thrust::plus<T>());

  ref = Vector(4, 10);
  ASSERT_EQ(ref, result);
}

TYPED_TEST(ConstantIteratorTests, TestConstantIteratorReduce)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  using namespace thrust;

  using T         = int;
  using ConstIter = constant_iterator<T>;

  ConstIter first = make_constant_iterator<T>(7);
  ConstIter last  = first + 4;

  T sum = thrust::reduce(first, last);

  ASSERT_EQ(sum, 4 * 7);
}
