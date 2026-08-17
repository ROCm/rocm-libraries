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

#include <thrust/detail/config.h>

#include <thrust/detail/tuple_algorithms.h>
#include <thrust/type_traits/integer_sequence.h>

#include <unittest/unittest.h>

// FIXME: Replace with C++14 style `thrust::square<>` when we have it.
struct custom_square
{
  template <typename T>
  THRUST_HOST_DEVICE T operator()(T v) const
  {
    return v * v;
  }
};

struct custom_square_inplace
{
  template <typename T>
  THRUST_HOST_DEVICE void operator()(T& v) const
  {
    v *= v;
  }
};

void test_tuple_subset()
{
  auto t0 = std::make_tuple(0, 2, 3.14);

  auto t1 = thrust::tuple_subset(t0, thrust::index_sequence<2, 0>{});

  ASSERT_EQUAL_QUIET(t1, std::make_tuple(3.14, 0));
}
DECLARE_UNITTEST(test_tuple_subset);

void test_tuple_transform()
{
  auto t0 = std::make_tuple(0, 2, 3.14);

  auto t1 = thrust::tuple_transform(t0, custom_square{});

  ASSERT_EQUAL_QUIET(t1, std::make_tuple(0, 4, 9.8596));
}
DECLARE_UNITTEST(test_tuple_transform);

void test_tuple_for_each()
{
  auto t = std::make_tuple(0, 2, 3.14);

  thrust::tuple_for_each(t, custom_square_inplace{});

  ASSERT_EQUAL_QUIET(t, std::make_tuple(0, 4, 9.8596));
}
DECLARE_UNITTEST(test_tuple_for_each);
