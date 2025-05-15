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
#include <thrust/merge.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include <unittest/unittest.h>

#if THRUST_DEVICE_SYSTEM != THRUST_DEVICE_SYSTEM_CUDA
#  include <type_traits>
#endif

template <typename T, typename CompareOp, typename... Args>
auto call_merge(Args&&... args) -> decltype(thrust::merge(std::forward<Args>(args)...))
{
#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
  THRUST_IF_CONSTEXPR (::cuda::std::is_void<CompareOp>::value)
#else
  THRUST_IF_CONSTEXPR (::std::is_void<CompareOp>::value)
#endif
  {
    return thrust::merge(std::forward<Args>(args)...);
  }
  else
  {
    // TODO(bgruber): remove next line in C++17 and pass CompareOp{} directly to stable_sort
#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
    using C = ::cuda::std::__conditional_t<::cuda::std::is_void<CompareOp>::value, thrust::less<T>, CompareOp>;
#else
    using C = ::std::conditional_t<::std::is_void<CompareOp>::value, thrust::less<T>, CompareOp>;
#endif
    return thrust::merge(std::forward<Args>(args)..., C{});
  }
#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
  _CCCL_UNREACHABLE();
#endif
}

template <typename U, typename CompareOp = void>
void TestMergeKeyValue(size_t n)
{
  using T = key_value<U, U>;

  const auto h_keys_a   = unittest::random_integers<U>(n);
  const auto h_values_a = unittest::random_integers<U>(n);

  const auto h_keys_b   = unittest::random_integers<U>(n);
  const auto h_values_b = unittest::random_integers<U>(n);

  thrust::host_vector<T> h_a(n), h_b(n);
  for (size_t i = 0; i < n; ++i)
  {
    h_a[i] = T(h_keys_a[i], h_values_a[i]);
    h_b[i] = T(h_keys_b[i], h_values_b[i]);
  }

#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
  THRUST_IF_CONSTEXPR (::cuda::std::is_void<CompareOp>::value)
#else
  THRUST_IF_CONSTEXPR (::std::is_void<CompareOp>::value)
#endif
  {
    thrust::stable_sort(h_a.begin(), h_a.end());
    thrust::stable_sort(h_b.begin(), h_b.end());
  }
  else
  {
    // TODO(bgruber): remove next line in C++17 and pass CompareOp{} directly to stable_sort
#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
    using C = ::cuda::std::__conditional_t<::cuda::std::is_void<CompareOp>::value, thrust::less<T>, CompareOp>;
#else
    using C = ::std::conditional_t<::std::is_void<CompareOp>::value, thrust::less<T>, CompareOp>;
#endif
    thrust::stable_sort(h_a.begin(), h_a.end(), C{});
    thrust::stable_sort(h_b.begin(), h_b.end(), C{});
  }

  const thrust::device_vector<T> d_a = h_a;
  const thrust::device_vector<T> d_b = h_b;

  thrust::host_vector<T> h_result(h_a.size() + h_b.size());
  thrust::device_vector<T> d_result(d_a.size() + d_b.size());

  const auto h_end = call_merge<T, CompareOp>(h_a.begin(), h_a.end(), h_b.begin(), h_b.end(), h_result.begin());
  const auto d_end = call_merge<T, CompareOp>(d_a.begin(), d_a.end(), d_b.begin(), d_b.end(), d_result.begin());

  ASSERT_EQUAL_QUIET(h_result, d_result);
  ASSERT_EQUAL(true, h_end == h_result.end());
  ASSERT_EQUAL(true, d_end == d_result.end());
}
DECLARE_VARIABLE_UNITTEST(TestMergeKeyValue);

template <typename U>
void TestMergeKeyValueDescending(size_t n)
{
  TestMergeKeyValue<U, thrust::greater<key_value<U, U>>>(n);
}
DECLARE_VARIABLE_UNITTEST(TestMergeKeyValueDescending);
