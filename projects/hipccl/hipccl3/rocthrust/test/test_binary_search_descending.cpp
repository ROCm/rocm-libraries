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
#include <thrust/functional.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include "test_param_fixtures.hpp"
#include "test_real_assertions.hpp"
#include "test_utils.hpp"

TESTS_DEFINE(BinarySearchDescendingTests, FullTestsParams);

//////////////////////
// Scalar Functions //
//////////////////////

TYPED_TEST(BinarySearchDescendingTests, TestScalarLowerBoundDescendingSimple)
{
  using Vector = typename TestFixture::input_type;
  using T      = typename Vector::value_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  Vector vec{8, 7, 5, 2, 0};

  ASSERT_EQ_QUIET(vec.begin() + 4, thrust::lower_bound(vec.begin(), vec.end(), T{0}, thrust::greater<T>()));
  ASSERT_EQ_QUIET(vec.begin() + 4, thrust::lower_bound(vec.begin(), vec.end(), T{1}, thrust::greater<T>()));
  ASSERT_EQ_QUIET(vec.begin() + 3, thrust::lower_bound(vec.begin(), vec.end(), T{2}, thrust::greater<T>()));
  ASSERT_EQ_QUIET(vec.begin() + 3, thrust::lower_bound(vec.begin(), vec.end(), T{3}, thrust::greater<T>()));
  ASSERT_EQ_QUIET(vec.begin() + 3, thrust::lower_bound(vec.begin(), vec.end(), T{4}, thrust::greater<T>()));
  ASSERT_EQ_QUIET(vec.begin() + 2, thrust::lower_bound(vec.begin(), vec.end(), T{5}, thrust::greater<T>()));
  ASSERT_EQ_QUIET(vec.begin() + 2, thrust::lower_bound(vec.begin(), vec.end(), T{6}, thrust::greater<T>()));
  ASSERT_EQ_QUIET(vec.begin() + 1, thrust::lower_bound(vec.begin(), vec.end(), T{7}, thrust::greater<T>()));
  ASSERT_EQ_QUIET(vec.begin() + 0, thrust::lower_bound(vec.begin(), vec.end(), T{8}, thrust::greater<T>()));
  ASSERT_EQ_QUIET(vec.begin() + 0, thrust::lower_bound(vec.begin(), vec.end(), T{9}, thrust::greater<T>()));
}

TYPED_TEST(BinarySearchDescendingTests, TestScalarUpperBoundDescendingSimple)
{
  using Vector = typename TestFixture::input_type;
  using T      = typename Vector::value_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  Vector vec{8, 7, 5, 2, 0};

  ASSERT_EQ_QUIET(vec.begin() + 5, thrust::upper_bound(vec.begin(), vec.end(), T{0}, thrust::greater<T>()));
  ASSERT_EQ_QUIET(vec.begin() + 4, thrust::upper_bound(vec.begin(), vec.end(), T{1}, thrust::greater<T>()));
  ASSERT_EQ_QUIET(vec.begin() + 4, thrust::upper_bound(vec.begin(), vec.end(), T{2}, thrust::greater<T>()));
  ASSERT_EQ_QUIET(vec.begin() + 3, thrust::upper_bound(vec.begin(), vec.end(), T{3}, thrust::greater<T>()));
  ASSERT_EQ_QUIET(vec.begin() + 3, thrust::upper_bound(vec.begin(), vec.end(), T{4}, thrust::greater<T>()));
  ASSERT_EQ_QUIET(vec.begin() + 3, thrust::upper_bound(vec.begin(), vec.end(), T{5}, thrust::greater<T>()));
  ASSERT_EQ_QUIET(vec.begin() + 2, thrust::upper_bound(vec.begin(), vec.end(), T{6}, thrust::greater<T>()));
  ASSERT_EQ_QUIET(vec.begin() + 2, thrust::upper_bound(vec.begin(), vec.end(), T{7}, thrust::greater<T>()));
  ASSERT_EQ_QUIET(vec.begin() + 1, thrust::upper_bound(vec.begin(), vec.end(), T{8}, thrust::greater<T>()));
  ASSERT_EQ_QUIET(vec.begin() + 0, thrust::upper_bound(vec.begin(), vec.end(), T{9}, thrust::greater<T>()));
}

TYPED_TEST(BinarySearchDescendingTests, TestScalarBinarySearchDescendingSimple)
{
  using Vector = typename TestFixture::input_type;
  using T      = typename Vector::value_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  Vector vec{8, 7, 5, 2, 0};

  ASSERT_EQ(true, thrust::binary_search(vec.begin(), vec.end(), T{0}, thrust::greater<T>()));
  ASSERT_EQ(false, thrust::binary_search(vec.begin(), vec.end(), T{1}, thrust::greater<T>()));
  ASSERT_EQ(true, thrust::binary_search(vec.begin(), vec.end(), T{2}, thrust::greater<T>()));
  ASSERT_EQ(false, thrust::binary_search(vec.begin(), vec.end(), T{3}, thrust::greater<T>()));
  ASSERT_EQ(false, thrust::binary_search(vec.begin(), vec.end(), T{4}, thrust::greater<T>()));
  ASSERT_EQ(true, thrust::binary_search(vec.begin(), vec.end(), T{5}, thrust::greater<T>()));
  ASSERT_EQ(false, thrust::binary_search(vec.begin(), vec.end(), T{6}, thrust::greater<T>()));
  ASSERT_EQ(true, thrust::binary_search(vec.begin(), vec.end(), T{7}, thrust::greater<T>()));
  ASSERT_EQ(true, thrust::binary_search(vec.begin(), vec.end(), T{8}, thrust::greater<T>()));
  ASSERT_EQ(false, thrust::binary_search(vec.begin(), vec.end(), T{9}, thrust::greater<T>()));
}

TYPED_TEST(BinarySearchDescendingTests, TestScalarEqualRangeDescendingSimple)
{
  using Vector = typename TestFixture::input_type;
  using T      = typename Vector::value_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  Vector vec{8, 7, 5, 2, 0};

  ASSERT_EQ_QUIET(vec.begin() + 4, thrust::equal_range(vec.begin(), vec.end(), T{0}, thrust::greater<T>()).first);
  ASSERT_EQ_QUIET(vec.begin() + 4, thrust::equal_range(vec.begin(), vec.end(), T{1}, thrust::greater<T>()).first);
  ASSERT_EQ_QUIET(vec.begin() + 3, thrust::equal_range(vec.begin(), vec.end(), T{2}, thrust::greater<T>()).first);
  ASSERT_EQ_QUIET(vec.begin() + 3, thrust::equal_range(vec.begin(), vec.end(), T{3}, thrust::greater<T>()).first);
  ASSERT_EQ_QUIET(vec.begin() + 3, thrust::equal_range(vec.begin(), vec.end(), T{4}, thrust::greater<T>()).first);
  ASSERT_EQ_QUIET(vec.begin() + 2, thrust::equal_range(vec.begin(), vec.end(), T{5}, thrust::greater<T>()).first);
  ASSERT_EQ_QUIET(vec.begin() + 2, thrust::equal_range(vec.begin(), vec.end(), T{6}, thrust::greater<T>()).first);
  ASSERT_EQ_QUIET(vec.begin() + 1, thrust::equal_range(vec.begin(), vec.end(), T{7}, thrust::greater<T>()).first);
  ASSERT_EQ_QUIET(vec.begin() + 0, thrust::equal_range(vec.begin(), vec.end(), T{8}, thrust::greater<T>()).first);
  ASSERT_EQ_QUIET(vec.begin() + 0, thrust::equal_range(vec.begin(), vec.end(), T{9}, thrust::greater<T>()).first);

  ASSERT_EQ_QUIET(vec.begin() + 5, thrust::equal_range(vec.begin(), vec.end(), T{0}, thrust::greater<T>()).second);
  ASSERT_EQ_QUIET(vec.begin() + 4, thrust::equal_range(vec.begin(), vec.end(), T{1}, thrust::greater<T>()).second);
  ASSERT_EQ_QUIET(vec.begin() + 4, thrust::equal_range(vec.begin(), vec.end(), T{2}, thrust::greater<T>()).second);
  ASSERT_EQ_QUIET(vec.begin() + 3, thrust::equal_range(vec.begin(), vec.end(), T{3}, thrust::greater<T>()).second);
  ASSERT_EQ_QUIET(vec.begin() + 3, thrust::equal_range(vec.begin(), vec.end(), T{4}, thrust::greater<T>()).second);
  ASSERT_EQ_QUIET(vec.begin() + 3, thrust::equal_range(vec.begin(), vec.end(), T{5}, thrust::greater<T>()).second);
  ASSERT_EQ_QUIET(vec.begin() + 2, thrust::equal_range(vec.begin(), vec.end(), T{6}, thrust::greater<T>()).second);
  ASSERT_EQ_QUIET(vec.begin() + 2, thrust::equal_range(vec.begin(), vec.end(), T{7}, thrust::greater<T>()).second);
  ASSERT_EQ_QUIET(vec.begin() + 1, thrust::equal_range(vec.begin(), vec.end(), T{8}, thrust::greater<T>()).second);
  ASSERT_EQ_QUIET(vec.begin() + 0, thrust::equal_range(vec.begin(), vec.end(), T{9}, thrust::greater<T>()).second);
}
