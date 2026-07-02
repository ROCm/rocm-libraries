/*
 *  Copyright 2026 NVIDIA Corporation
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

#include <thrust/detail/libcxx_wrapper/__cccl_config.h>

THRUST_DIAG_PUSH
// gcc 10 and 11 wrongly warn about an out-of-bounds access in TestWritingStridedIteratorToStructMember
#if THRUST_COMPILER(GCC, >=, 10) && THRUST_COMPILER(GCC, <, 12)
THRUST_DIAG_SUPPRESS_GCC("-Warray-bounds")
#endif // THRUST_COMPILER(GCC, >=, 10) && THRUST_COMPILER(GCC, <, 12)

#include <thrust/device_vector.h>
#include <thrust/iterator/strided_iterator.h>
#include <thrust/universal_vector.h>

#include _THRUST_STD_INCLUDE(array)
#include _THRUST_STD_INCLUDE(utility)

#include <algorithm>
#include <numeric>

#include "test_param_fixtures.hpp"
#include "test_real_assertions.hpp"
#include "test_utils.hpp"

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

TESTS_DEFINE(StridedIteratorTests, NumericalTestsParams);
TESTS_DEFINE(StridedIteratorVectorTests, VectorTestsParams);

TEST(StridedIteratorTests, TestReadingStridedIterator)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::host_vector<int> v(21);
  std::iota(v.begin(), v.end(), -4);
  auto iter = thrust::make_strided_iterator(v.begin() + 4, 2);

  ASSERT_EQ(*iter, 0);
  iter++;
  ASSERT_EQ(*iter, 2);
  iter++;
  iter++;
  ASSERT_EQ(*iter, 6);
  iter += 5;
  ASSERT_EQ(*iter, 16);
  iter -= 10;
  ASSERT_EQ(*iter, -4);
}

TYPED_TEST(StridedIteratorVectorTests, TestWritingStridedIterator)
{
  using Vector = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());
  // iterate over all second elements (runtime stride)
  {
    Vector v(10);
    auto iter = thrust::make_strided_iterator(v.begin(), 2);
    ASSERT_EQ(v, (Vector{0, 0, 0, 0, 0, 0, 0, 0, 0, 0}));
    *iter = 33;
    ASSERT_EQ(v, (Vector{33, 0, 0, 0, 0, 0, 0, 0, 0, 0}));
    auto iter2 = iter + 1;
    *iter2     = 34;
    ASSERT_EQ(v, (Vector{33, 0, 34, 0, 0, 0, 0, 0, 0, 0}));
    thrust::fill(iter + 2, iter + 4, 42);
    ASSERT_EQ(v, (Vector{33, 0, 34, 0, 42, 0, 42, 0, 0, 0}));
  }

  // iterate over all second elements (static stride)
  {
    Vector v(10);
    auto iter = thrust::make_strided_iterator<2>(v.begin());
    thrust::fill(iter, iter + 3, 42);
    ASSERT_EQ(v, (Vector{42, 0, 42, 0, 42, 0, 0, 0, 0, 0}));
  }
}

TEST(StridedIteratorTests, TestWritingStridedIteratorToStructMember)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());
  using pair            = _THRUST_STD::pair<int, double>;
  using arr_of_pairs    = _THRUST_STD::array<pair, 4>;
  const auto data       = arr_of_pairs{{{1, 2}, {3, 4}, {5, 6}, {7, 8}}};
  const auto reference  = arr_of_pairs{{{1, 1337}, {3, 1337}, {5, 1337}, {7, 1337}}};
  constexpr auto stride = sizeof(pair) / sizeof(double);
  static_assert(stride == 2);

  // iterate over all second elements (runtime stride)
  {
    auto arr  = data;
    auto iter = thrust::make_strided_iterator(&arr[0].second, stride);
    thrust::fill(iter, iter + 4, 1337);
    ASSERT_EQ(arr == reference, true);
  }

  // iterate over all second elements (static stride)
  {
    auto arr  = data;
    auto iter = thrust::make_strided_iterator<stride>(&arr[0].second);
    thrust::fill(iter, iter + 4, 1337);
    ASSERT_EQ(arr == reference, true);
  }
}

THRUST_DIAG_POP
