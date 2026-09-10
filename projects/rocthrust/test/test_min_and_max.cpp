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

#include <thrust/extrema.h>

#include "test_param_fixtures.hpp"
#include "test_real_assertions.hpp"
#include "test_utils.hpp"

using NumericTypes = ::testing::Types<
  Params<char>,
  Params<signed char>,
  Params<unsigned char>,
  Params<short>,
  Params<unsigned short>,
  Params<int>,
  Params<unsigned int>,
  Params<long>,
  Params<unsigned long>,
  Params<long long>,
  Params<unsigned long long>,
  Params<float>,
  Params<double>>;

TESTS_DEFINE(MinAndMaxTests, NumericTypes);

TYPED_TEST(MinAndMaxTests, TestMin)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  using T = typename TestFixture::input_type;

  // 2 < 3
  T two(2), three(3);
  ASSERT_EQ(two, thrust::min THRUST_PREVENT_MACRO_SUBSTITUTION(two, three));
  ASSERT_EQ(two, thrust::min THRUST_PREVENT_MACRO_SUBSTITUTION(two, three, thrust::less<T>()));

  ASSERT_EQ(two, thrust::min THRUST_PREVENT_MACRO_SUBSTITUTION(three, two));
  ASSERT_EQ(two, thrust::min THRUST_PREVENT_MACRO_SUBSTITUTION(three, two, thrust::less<T>()));

  ASSERT_EQ(three, thrust::min THRUST_PREVENT_MACRO_SUBSTITUTION(two, three, thrust::greater<T>()));
  ASSERT_EQ(three, thrust::min THRUST_PREVENT_MACRO_SUBSTITUTION(three, two, thrust::greater<T>()));

  using KV = key_value<T, T>;
  KV two_and_two(two, two);
  KV two_and_three(two, three);

  // the first element breaks ties
  ASSERT_EQ_QUIET(two_and_two, thrust::min THRUST_PREVENT_MACRO_SUBSTITUTION(two_and_two, two_and_three));
  ASSERT_EQ_QUIET(two_and_three, thrust::min THRUST_PREVENT_MACRO_SUBSTITUTION(two_and_three, two_and_two));

  ASSERT_EQ_QUIET(two_and_two,
                  thrust::min THRUST_PREVENT_MACRO_SUBSTITUTION(two_and_two, two_and_three, thrust::less<KV>()));
  ASSERT_EQ_QUIET(two_and_three,
                  thrust::min THRUST_PREVENT_MACRO_SUBSTITUTION(two_and_three, two_and_two, thrust::less<KV>()));

  ASSERT_EQ_QUIET(two_and_two,
                  thrust::min THRUST_PREVENT_MACRO_SUBSTITUTION(two_and_two, two_and_three, thrust::greater<KV>()));
  ASSERT_EQ_QUIET(two_and_three,
                  thrust::min THRUST_PREVENT_MACRO_SUBSTITUTION(two_and_three, two_and_two, thrust::greater<KV>()));
}

TYPED_TEST(MinAndMaxTests, TestMax)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  using T = typename TestFixture::input_type;

  // 2 < 3
  T two(2), three(3);
  ASSERT_EQ(three, thrust::max THRUST_PREVENT_MACRO_SUBSTITUTION(two, three));
  ASSERT_EQ(three, thrust::max THRUST_PREVENT_MACRO_SUBSTITUTION(two, three, thrust::less<T>()));

  ASSERT_EQ(three, thrust::max THRUST_PREVENT_MACRO_SUBSTITUTION(three, two));
  ASSERT_EQ(three, thrust::max THRUST_PREVENT_MACRO_SUBSTITUTION(three, two, thrust::less<T>()));

  ASSERT_EQ(two, thrust::max THRUST_PREVENT_MACRO_SUBSTITUTION(two, three, thrust::greater<T>()));
  ASSERT_EQ(two, thrust::max THRUST_PREVENT_MACRO_SUBSTITUTION(three, two, thrust::greater<T>()));

  using KV = key_value<T, T>;
  KV two_and_two(two, two);
  KV two_and_three(two, three);

  // the first element breaks ties
  ASSERT_EQ_QUIET(two_and_two, thrust::max THRUST_PREVENT_MACRO_SUBSTITUTION(two_and_two, two_and_three));
  ASSERT_EQ_QUIET(two_and_three, thrust::max THRUST_PREVENT_MACRO_SUBSTITUTION(two_and_three, two_and_two));

  ASSERT_EQ_QUIET(two_and_two,
                  thrust::max THRUST_PREVENT_MACRO_SUBSTITUTION(two_and_two, two_and_three, thrust::less<KV>()));
  ASSERT_EQ_QUIET(two_and_three,
                  thrust::max THRUST_PREVENT_MACRO_SUBSTITUTION(two_and_three, two_and_two, thrust::less<KV>()));

  ASSERT_EQ_QUIET(two_and_two,
                  thrust::max THRUST_PREVENT_MACRO_SUBSTITUTION(two_and_two, two_and_three, thrust::greater<KV>()));
  ASSERT_EQ_QUIET(two_and_three,
                  thrust::max THRUST_PREVENT_MACRO_SUBSTITUTION(two_and_three, two_and_two, thrust::greater<KV>()));
}
