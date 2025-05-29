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

#include <thrust/functional.h>
#include <thrust/set_operations.h>
#include <thrust/sort.h>

#include "test_header.hpp"

TESTS_DEFINE(SetDifferenceDescendingTests, FullTestsParams);
TESTS_DEFINE(SetDifferenceDescendingPrimitiveTests, NumericalTestsParams);

TYPED_TEST(SetDifferenceDescendingTests, TestSetDifferenceDescendingSimple)
{
  using Vector   = typename TestFixture::input_type;
  using Policy   = typename TestFixture::execution_policy;
  using T        = typename Vector::value_type;
  using Iterator = typename Vector::iterator;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  Vector a(4), b(5);

  a[0] = 5;
  a[1] = 4;
  a[2] = 2;
  a[3] = 0;
  b[0] = 6;
  b[1] = 4;
  b[2] = 3;
  b[3] = 3;
  b[4] = 0;

  Vector ref(2);
  ref[0] = 5;
  ref[1] = 2;

  Vector result(2);

  Iterator end =
    thrust::set_difference(Policy{}, a.begin(), a.end(), b.begin(), b.end(), result.begin(), thrust::greater<T>());

  EXPECT_EQ(result.end(), end);
  ASSERT_EQ(ref, result);
}

TYPED_TEST(SetDifferenceDescendingPrimitiveTests, TestSetDifferenceDescending)
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

      thrust::sort(h_a.begin(), h_a.end(), thrust::greater<T>());
      thrust::sort(h_b.begin(), h_b.end(), thrust::greater<T>());

      thrust::device_vector<T> d_a = h_a;
      thrust::device_vector<T> d_b = h_b;

      thrust::host_vector<T> h_result(size);
      thrust::device_vector<T> d_result(size);

      typename thrust::host_vector<T>::iterator h_end;
      typename thrust::device_vector<T>::iterator d_end;

      h_end =
        thrust::set_difference(h_a.begin(), h_a.end(), h_b.begin(), h_b.end(), h_result.begin(), thrust::greater<T>());
      h_result.resize(h_end - h_result.begin());

      d_end =
        thrust::set_difference(d_a.begin(), d_a.end(), d_b.begin(), d_b.end(), d_result.begin(), thrust::greater<T>());

      d_result.resize(d_end - d_result.begin());

      ASSERT_EQ(h_result, d_result);
    }
  }
}
