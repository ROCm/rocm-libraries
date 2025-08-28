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

#if !_THRUST_HAS_DEVICE_SYSTEM_STD
#  include <functional>
#endif

struct addable
{
  THRUST_HOST_DEVICE friend auto operator+(const addable&, const addable&) -> addable
  {
    return addable{};
  }
};

void TestAddressStabilityLibcuxx()
{
  using ::thrust::detail::proclaim_copyable_arguments;
  using ::thrust::detail::proclaims_copyable_arguments;

  // libcu++ function objects with known types
  static_assert(proclaims_copyable_arguments<_THRUST_STD::plus<int>>::value, "");
  static_assert(!proclaims_copyable_arguments<_THRUST_STD::plus<>>::value, "");

  // libcu++ function objects with unknown types
  static_assert(!proclaims_copyable_arguments<_THRUST_STD::plus<addable>>::value, "");
  static_assert(!proclaims_copyable_arguments<_THRUST_STD::plus<>>::value, "");

  // libcu++ function objects with unknown types and opt-in
  static_assert(
    proclaims_copyable_arguments<decltype(proclaim_copyable_arguments(_THRUST_STD::plus<addable>{}))>::value, "");
  static_assert(proclaims_copyable_arguments<decltype(proclaim_copyable_arguments(_THRUST_STD::plus<>{}))>::value, "");
}
DECLARE_UNITTEST(TestAddressStabilityLibcuxx);

void TestAddressStabilityThrust()
{
  using ::thrust::detail::proclaim_copyable_arguments;
  using ::thrust::detail::proclaims_copyable_arguments;

  // thrust function objects with known types
  static_assert(proclaims_copyable_arguments<thrust::plus<int>>::value, "");
  static_assert(!proclaims_copyable_arguments<thrust::plus<>>::value, "");

  // thrust function objects with unknown types
  static_assert(!proclaims_copyable_arguments<thrust::plus<addable>>::value, "");
  static_assert(!proclaims_copyable_arguments<thrust::plus<>>::value, "");

  // thrust function objects with unknown types and opt-in
  static_assert(proclaims_copyable_arguments<decltype(proclaim_copyable_arguments(thrust::plus<addable>{}))>::value,
                "");
  static_assert(proclaims_copyable_arguments<decltype(proclaim_copyable_arguments(thrust::plus<>{}))>::value, "");
}
DECLARE_UNITTEST(TestAddressStabilityThrust);

template <typename T>
struct my_plus
{
  THRUST_HOST_DEVICE auto operator()(T a, T b) const -> T
  {
    return a + b;
  }
};

void TestAddressStabilityUserDefinedFunctionObject()
{
  using ::thrust::detail::proclaim_copyable_arguments;
  using ::thrust::detail::proclaims_copyable_arguments;

  // by-value overload
  static_assert(!proclaims_copyable_arguments<my_plus<int>>::value, "");

  // by-value overload with opt-in
  static_assert(proclaims_copyable_arguments<decltype(proclaim_copyable_arguments(my_plus<int>{}))>::value, "");

  // by-reference overload
  static_assert(!proclaims_copyable_arguments<my_plus<int&>>::value, "");
  static_assert(!proclaims_copyable_arguments<my_plus<const int&>>::value, "");
  static_assert(!proclaims_copyable_arguments<my_plus<int&&>>::value, "");
  static_assert(!proclaims_copyable_arguments<my_plus<const int&&>>::value, "");

  // by-reference overload with opt-in
  static_assert(proclaims_copyable_arguments<decltype(proclaim_copyable_arguments(my_plus<int&>{}))>::value, "");
  static_assert(proclaims_copyable_arguments<decltype(proclaim_copyable_arguments(my_plus<const int&>{}))>::value, "");
  static_assert(proclaims_copyable_arguments<decltype(proclaim_copyable_arguments(my_plus<int&&>{}))>::value, "");
  static_assert(proclaims_copyable_arguments<decltype(proclaim_copyable_arguments(my_plus<const int&&>{}))>::value, "");
}
DECLARE_UNITTEST(TestAddressStabilityUserDefinedFunctionObject);

void TestAddressStabilityLambda()
{
  using ::thrust::detail::proclaim_copyable_arguments;
  using ::thrust::detail::proclaims_copyable_arguments;

  {
    auto l = [](const int& i) {
      return i + 2;
    };
    static_assert(!proclaims_copyable_arguments<decltype(l)>::value, "");
    auto pr_l = proclaim_copyable_arguments(l);
    ASSERT_EQUAL(pr_l(3), 5);
    static_assert(proclaims_copyable_arguments<decltype(pr_l)>::value, "");
  }

  {
    auto l = [] THRUST_DEVICE(const int& i) {
      return i + 2;
    };
    static_assert(!proclaims_copyable_arguments<decltype(l)>::value, "");
    auto pr_device_l = proclaim_copyable_arguments(l);
    (void) &pr_device_l;
    static_assert(proclaims_copyable_arguments<decltype(pr_device_l)>::value, "");
  }

  {
    auto l = [] THRUST_HOST_DEVICE(const int& i) {
      return i + 2;
    };
    static_assert(!proclaims_copyable_arguments<decltype(l)>::value, "");
    auto pr_l = proclaim_copyable_arguments(l);
    ASSERT_EQUAL(pr_l(3), 5);
    static_assert(proclaims_copyable_arguments<decltype(pr_l)>::value, "");
  }
}
DECLARE_UNITTEST(TestAddressStabilityLambda);
