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

#include <thrust/extrema.h>

#include <unittest/unittest.h>

template <typename T>
struct TestMin
{
  void operator()(void)
  {
    // 2 < 3
    T two(2), three(3);
    ASSERT_EQUAL(two, thrust::min THRUST_PREVENT_MACRO_SUBSTITUTION(two, three));
    ASSERT_EQUAL(two, thrust::min THRUST_PREVENT_MACRO_SUBSTITUTION(two, three, thrust::less<T>()));

    ASSERT_EQUAL(two, thrust::min THRUST_PREVENT_MACRO_SUBSTITUTION(three, two));
    ASSERT_EQUAL(two, thrust::min THRUST_PREVENT_MACRO_SUBSTITUTION(three, two, thrust::less<T>()));

    ASSERT_EQUAL(three, thrust::min THRUST_PREVENT_MACRO_SUBSTITUTION(two, three, thrust::greater<T>()));
    ASSERT_EQUAL(three, thrust::min THRUST_PREVENT_MACRO_SUBSTITUTION(three, two, thrust::greater<T>()));

    using KV = key_value<T, T>;
    KV two_and_two(two, two);
    KV two_and_three(two, three);

    // the first element breaks ties
    ASSERT_EQUAL_QUIET(two_and_two, thrust::min THRUST_PREVENT_MACRO_SUBSTITUTION(two_and_two, two_and_three));
    ASSERT_EQUAL_QUIET(two_and_three, thrust::min THRUST_PREVENT_MACRO_SUBSTITUTION(two_and_three, two_and_two));

    ASSERT_EQUAL_QUIET(two_and_two,
                       thrust::min THRUST_PREVENT_MACRO_SUBSTITUTION(two_and_two, two_and_three, thrust::less<KV>()));
    ASSERT_EQUAL_QUIET(two_and_three,
                       thrust::min THRUST_PREVENT_MACRO_SUBSTITUTION(two_and_three, two_and_two, thrust::less<KV>()));

    ASSERT_EQUAL_QUIET(
      two_and_two, thrust::min THRUST_PREVENT_MACRO_SUBSTITUTION(two_and_two, two_and_three, thrust::greater<KV>()));
    ASSERT_EQUAL_QUIET(
      two_and_three, thrust::min THRUST_PREVENT_MACRO_SUBSTITUTION(two_and_three, two_and_two, thrust::greater<KV>()));
  }
};
SimpleUnitTest<TestMin, NumericTypes> TestMinInstance;

template <typename T>
struct TestMax
{
  void operator()(void)
  {
    // 2 < 3
    T two(2), three(3);
    ASSERT_EQUAL(three, thrust::max THRUST_PREVENT_MACRO_SUBSTITUTION(two, three));
    ASSERT_EQUAL(three, thrust::max THRUST_PREVENT_MACRO_SUBSTITUTION(two, three, thrust::less<T>()));

    ASSERT_EQUAL(three, thrust::max THRUST_PREVENT_MACRO_SUBSTITUTION(three, two));
    ASSERT_EQUAL(three, thrust::max THRUST_PREVENT_MACRO_SUBSTITUTION(three, two, thrust::less<T>()));

    ASSERT_EQUAL(two, thrust::max THRUST_PREVENT_MACRO_SUBSTITUTION(two, three, thrust::greater<T>()));
    ASSERT_EQUAL(two, thrust::max THRUST_PREVENT_MACRO_SUBSTITUTION(three, two, thrust::greater<T>()));

    using KV = key_value<T, T>;
    KV two_and_two(two, two);
    KV two_and_three(two, three);

    // the first element breaks ties
    ASSERT_EQUAL_QUIET(two_and_two, thrust::max THRUST_PREVENT_MACRO_SUBSTITUTION(two_and_two, two_and_three));
    ASSERT_EQUAL_QUIET(two_and_three, thrust::max THRUST_PREVENT_MACRO_SUBSTITUTION(two_and_three, two_and_two));

    ASSERT_EQUAL_QUIET(two_and_two,
                       thrust::max THRUST_PREVENT_MACRO_SUBSTITUTION(two_and_two, two_and_three, thrust::less<KV>()));
    ASSERT_EQUAL_QUIET(two_and_three,
                       thrust::max THRUST_PREVENT_MACRO_SUBSTITUTION(two_and_three, two_and_two, thrust::less<KV>()));

    ASSERT_EQUAL_QUIET(
      two_and_two, thrust::max THRUST_PREVENT_MACRO_SUBSTITUTION(two_and_two, two_and_three, thrust::greater<KV>()));
    ASSERT_EQUAL_QUIET(
      two_and_three, thrust::max THRUST_PREVENT_MACRO_SUBSTITUTION(two_and_three, two_and_two, thrust::greater<KV>()));
  }
};
SimpleUnitTest<TestMax, NumericTypes> TestMaxInstance;
