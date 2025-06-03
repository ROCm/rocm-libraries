/*
 *  Copyright 2024 NVIDIA Corporation
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

#include <thrust/detail/functional/address_stability.h>

#include <unittest/unittest.h>

struct my_plus
{
  THRUST_HOST_DEVICE auto operator()(int a, int b) const -> int
  {
    return a + b;
  }
};

void TestAddressStability()
{
  using ::thrust::detail::proclaim_copyable_arguments;
  using ::thrust::detail::proclaims_copyable_arguments;

  static_assert(!proclaims_copyable_arguments<thrust::plus<int>>::value, "");
  static_assert(proclaims_copyable_arguments<decltype(proclaim_copyable_arguments(thrust::plus<int>{}))>::value, "");

  static_assert(!proclaims_copyable_arguments<my_plus>::value, "");
  static_assert(proclaims_copyable_arguments<decltype(proclaim_copyable_arguments(my_plus{}))>::value, "");
}
DECLARE_UNITTEST(TestAddressStability);
