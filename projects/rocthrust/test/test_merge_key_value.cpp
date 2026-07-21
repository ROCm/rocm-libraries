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

#include <thrust/functional.h>
#include <thrust/merge.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include "test_param_fixtures.hpp"
#include "test_real_assertions.hpp"
#include "test_utils.hpp"

#if !_THRUST_HAS_DEVICE_SYSTEM_STD
#  include <type_traits>
#endif

template <typename Input, typename CompareFunction = thrust::less<key_value<Input, Input>>>
struct ParamsMerge
{
  using input_type       = Input;
  using compare_function = CompareFunction;
};

template <typename ParamsMerge>
class MergeKeyValueTestsClass : public ::testing::Test
{
public:
  using input_type       = typename ParamsMerge::input_type;
  using compare_function = typename ParamsMerge::compare_function;
};

using MergeKeyValueTestsParams = ::testing::Types<
  ParamsMerge<short, thrust::less<key_value<short, short>>>,
  ParamsMerge<int, thrust::less<key_value<int, int>>>,
  ParamsMerge<long long, thrust::less<key_value<long long, long long>>>,
  ParamsMerge<unsigned short, thrust::less<key_value<unsigned short, unsigned short>>>,
  ParamsMerge<unsigned int, thrust::less<key_value<unsigned int, unsigned int>>>,
  ParamsMerge<unsigned long long, thrust::less<key_value<unsigned long long, unsigned long long>>>,
  ParamsMerge<float, thrust::less<key_value<float, float>>>,
  ParamsMerge<double, thrust::less<key_value<double, double>>>,
  ParamsMerge<short, thrust::greater<key_value<short, short>>>,
  ParamsMerge<int, thrust::greater<key_value<int, int>>>,
  ParamsMerge<long long, thrust::greater<key_value<long long, long long>>>,
  ParamsMerge<unsigned short, thrust::greater<key_value<unsigned short, unsigned short>>>,
  ParamsMerge<unsigned int, thrust::greater<key_value<unsigned int, unsigned int>>>,
  ParamsMerge<unsigned long long, thrust::greater<key_value<unsigned long long, unsigned long long>>>,
  ParamsMerge<float, thrust::greater<key_value<float, float>>>,
  ParamsMerge<double, thrust::greater<key_value<double, double>>>>;

TYPED_TEST_SUITE(MergeKeyValueTestsClass, MergeKeyValueTestsParams);

TESTS_DEFINE(MergeKeyValueTests, FullTestsParams);

template <typename T, typename CompareOp, typename... Args>
auto call_merge(Args&&... args) -> decltype(thrust::merge(std::forward<Args>(args)...))
{
  THRUST_IF_CONSTEXPR (_THRUST_STD::is_void<CompareOp>::value)
  {
    return thrust::merge(std::forward<Args>(args)...);
  }
  else
  {
    // TODO(bgruber): remove next line in C++17 and pass CompareOp{} directly to stable_sort
    using C = _THRUST_STD::conditional_t<_THRUST_STD::is_void<CompareOp>::value, thrust::less<T>, CompareOp>;
    return thrust::merge(std::forward<Args>(args)..., C{});
  }
  __builtin_unreachable();
}

TYPED_TEST(MergeKeyValueTestsClass, TestMergeKeyValue)
{
  using U                = typename TestFixture::input_type;
  using compare_function = typename TestFixture::compare_function;
  using T                = key_value<U, U>;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  for (auto size : get_sizes())
  {
    SCOPED_TRACE(testing::Message() << "with size= " << size);

    for (auto seed : get_seeds())
    {
      SCOPED_TRACE(testing::Message() << "with seed= " << seed);

      const auto h_keys_a = get_random_data<U>(size, get_default_limits<U>::min(), get_default_limits<U>::max(), seed);
      const auto h_values_a = get_random_data<U>(
        size, get_default_limits<U>::min(), get_default_limits<U>::max(), seed + seed_value_addition);

      const auto h_keys_b = get_random_data<U>(
        size, get_default_limits<U>::min(), get_default_limits<U>::max(), seed + 2 * seed_value_addition);
      const auto h_values_b = get_random_data<U>(
        size, get_default_limits<U>::min(), get_default_limits<U>::max(), seed + 3 * seed_value_addition);

      thrust::host_vector<T> h_a(size), h_b(size);
      for (size_t i = 0; i < size; ++i)
      {
        h_a[i] = T(h_keys_a[i], h_values_a[i]);
        h_b[i] = T(h_keys_b[i], h_values_b[i]);
      }

      THRUST_IF_CONSTEXPR (_THRUST_STD::is_void<compare_function>::value)
      {
        thrust::stable_sort(h_a.begin(), h_a.end());
        thrust::stable_sort(h_b.begin(), h_b.end());
      }
      else
      {
        // TODO(bgruber): remove next line in C++17 and pass compare_function{} directly to stable_sort
        using C =
          _THRUST_STD::conditional_t<_THRUST_STD::is_void<compare_function>::value, thrust::less<T>, compare_function>;
        thrust::stable_sort(h_a.begin(), h_a.end(), C{});
        thrust::stable_sort(h_b.begin(), h_b.end(), C{});
      }

      const thrust::device_vector<T> d_a = h_a;
      const thrust::device_vector<T> d_b = h_b;

      thrust::host_vector<T> h_result(h_a.size() + h_b.size());
      thrust::device_vector<T> d_result(d_a.size() + d_b.size());

      const auto h_end =
        call_merge<T, compare_function>(h_a.begin(), h_a.end(), h_b.begin(), h_b.end(), h_result.begin());
      const auto d_end =
        call_merge<T, compare_function>(d_a.begin(), d_a.end(), d_b.begin(), d_b.end(), d_result.begin());

      ASSERT_EQ_QUIET(h_result, d_result);
      ASSERT_TRUE(h_end == h_result.end());
      ASSERT_TRUE(d_end == d_result.end());
    }
  }
}
