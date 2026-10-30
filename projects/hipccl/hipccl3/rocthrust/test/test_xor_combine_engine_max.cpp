// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Modifications Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <thrust/random/detail/xor_combine_engine_max.h>

#include "test_param_fixtures.hpp"
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
