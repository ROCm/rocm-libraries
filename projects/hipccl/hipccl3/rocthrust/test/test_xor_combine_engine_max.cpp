/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *  Modifications Copyright© 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include <thrust/random/detail/xor_combine_engine_max.h>

#include "test_utils.hpp"

TEST(XorCombineEngineMaxTests, TestLog2)
{
  static_assert(thrust::random::detail::log2(1u) == 0u);
  static_assert(thrust::random::detail::log2(2u) == 1u);
  static_assert(thrust::random::detail::log2(3u) == 1u);
  static_assert(thrust::random::detail::log2(4u) == 2u);
  static_assert(thrust::random::detail::log2(5u) == 2u);
  static_assert(thrust::random::detail::log2(6u) == 2u);
  static_assert(thrust::random::detail::log2(7u) == 2u);
  static_assert(thrust::random::detail::log2(8u) == 3u);
  static_assert(thrust::random::detail::log2(9u) == 3u);
  static_assert(thrust::random::detail::log2(15u) == 3u);
  static_assert(thrust::random::detail::log2(16u) == 4u);
  static_assert(thrust::random::detail::log2(17u) == 4u);
  static_assert(thrust::random::detail::log2(127u) == 6u);
  static_assert(thrust::random::detail::log2(128u) == 7u);
  static_assert(thrust::random::detail::log2(129u) == 7u);
  static_assert(thrust::random::detail::log2(256u) == 8u);
  static_assert(thrust::random::detail::log2(511u) == 8u);
  static_assert(thrust::random::detail::log2(512u) == 9u);
}
