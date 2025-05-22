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

#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/retag.h>
#include <thrust/merge.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include "test_header.hpp"

template <class Input, class CompareFunction = thrust::less<Input>>
struct ParamsMerge
{
  using input_type       = Input;
  using compare_function = CompareFunction;
};

template <class ParamsMerge>
class MergeByKeyTestsClass : public ::testing::Test
{
public:
  using input_type       = typename ParamsMerge::input_type;
  using compare_function = typename ParamsMerge::compare_function;
};

using MergeByKeyTestsParams = ::testing::Types<
  ParamsMerge<short, thrust::less<short>>,
  ParamsMerge<int, thrust::less<int>>,
  ParamsMerge<long long, thrust::less<long long>>,
  ParamsMerge<unsigned short, thrust::less<unsigned short>>,
  ParamsMerge<unsigned int, thrust::less<unsigned int>>,
  ParamsMerge<unsigned long long, thrust::less<unsigned long int>>,
  ParamsMerge<float, thrust::less<float>>,
  ParamsMerge<double, thrust::less<double>>,
  ParamsMerge<short, thrust::greater<short>>,
  ParamsMerge<int, thrust::greater<int>>,
  ParamsMerge<long long, thrust::greater<long long>>,
  ParamsMerge<unsigned short, thrust::greater<unsigned short>>,
  ParamsMerge<unsigned int, thrust::greater<unsigned int>>,
  ParamsMerge<unsigned long long, thrust::greater<unsigned long int>>,
  ParamsMerge<float, thrust::greater<float>>,
  ParamsMerge<double, thrust::greater<double>>>;

TYPED_TEST_SUITE(MergeByKeyTestsClass, MergeByKeyTestsParams);

TESTS_DEFINE(MergeByKeyTests, FullTestsParams);
TESTS_DEFINE(PrimitiveMergeByKeyTests, NumericalTestsParams);

TEST(MergeByKeyTests, UsingHip)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  ASSERT_EQ(THRUST_DEVICE_SYSTEM, THRUST_DEVICE_SYSTEM_HIP);
}

TYPED_TEST(MergeByKeyTests, MergeByKeySimple)
{
  using Vector = typename TestFixture::input_type;
  using Policy = typename TestFixture::execution_policy;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  const Vector a_key{0, 2, 4}, a_val{13, 7, 42}, b_key{0, 3, 3, 4}, b_val{42, 42, 7, 13};
  Vector ref_key{0, 0, 2, 3, 3, 4, 4}, ref_val{13, 42, 7, 42, 7, 42, 13};

  Vector result_key(7), result_val(7);

  const auto ends = thrust::merge_by_key(
    Policy{},
    a_key.begin(),
    a_key.end(),
    b_key.begin(),
    b_key.end(),
    a_val.begin(),
    b_val.begin(),
    result_key.begin(),
    result_val.begin());

  EXPECT_EQ(result_key.end(), ends.first);
  EXPECT_EQ(result_val.end(), ends.second);
  ASSERT_EQ(ref_key, result_key);
  ASSERT_EQ(ref_val, result_val);
}

template <typename InputIterator1,
          typename InputIterator2,
          typename InputIterator3,
          typename InputIterator4,
          typename OutputIterator1,
          typename OutputIterator2>
thrust::pair<OutputIterator1, OutputIterator2> merge_by_key(
  my_system& system,
  InputIterator1,
  InputIterator1,
  InputIterator2,
  InputIterator2,
  InputIterator3,
  InputIterator4,
  OutputIterator1 keys_result,
  OutputIterator2 values_result)
{
  system.validate_dispatch();
  return thrust::make_pair(keys_result, values_result);
}

TEST(MergeByKeyTests, MergeByKeyDispatchExplicit)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::device_vector<int> vec(1);

  my_system sys(0);
  thrust::merge_by_key(
    sys, vec.begin(), vec.begin(), vec.begin(), vec.begin(), vec.begin(), vec.begin(), vec.begin(), vec.begin());

  ASSERT_EQ(true, sys.is_valid());
}

template <typename InputIterator1,
          typename InputIterator2,
          typename InputIterator3,
          typename InputIterator4,
          typename OutputIterator1,
          typename OutputIterator2>
thrust::pair<OutputIterator1, OutputIterator2> merge_by_key(
  my_tag,
  InputIterator1,
  InputIterator1,
  InputIterator2,
  InputIterator2,
  InputIterator3,
  InputIterator4,
  OutputIterator1 keys_result,
  OutputIterator2 values_result)
{
  *keys_result = 13;
  return thrust::make_pair(keys_result, values_result);
}

TEST(MergeByKeyTests, MergeByKeyDispatchImplicit)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::device_vector<int> vec(1);

  thrust::merge_by_key(
    thrust::retag<my_tag>(vec.begin()),
    thrust::retag<my_tag>(vec.begin()),
    thrust::retag<my_tag>(vec.begin()),
    thrust::retag<my_tag>(vec.begin()),
    thrust::retag<my_tag>(vec.begin()),
    thrust::retag<my_tag>(vec.begin()),
    thrust::retag<my_tag>(vec.begin()),
    thrust::retag<my_tag>(vec.begin()));

  ASSERT_EQ(13, vec.front());
}

template <typename T, typename CompareOp, typename... Args>
auto call_merge_by_key(Args&&... args) -> decltype(thrust::merge_by_key(std::forward<Args>(args)...))
{
  THRUST_IF_CONSTEXPR (std::is_void<CompareOp>::value)
  {
    return thrust::merge_by_key(std::forward<Args>(args)...);
  }
  else
  {
    using C = std::conditional_t<std::is_void<CompareOp>::value, thrust::less<T>, CompareOp>;
    return thrust::merge_by_key(std::forward<Args>(args)..., C{});
  }
}

// ascending and descending
TYPED_TEST(MergeByKeyTestsClass, TestMergeByKeyWithRandomData)
{
  using T                = typename TestFixture::input_type;
  using compare_function = typename TestFixture::compare_function;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  for (auto size : get_sizes())
  {
    SCOPED_TRACE(testing::Message() << "with size= " << size);

    for (auto seed : get_seeds())
    {
      SCOPED_TRACE(testing::Message() << "with seed= " << seed);

      thrust::host_vector<T> random_keys = get_random_data<unsigned short int>(size, 0, 255, seed);
      thrust::host_vector<T> random_vals =
        get_random_data<unsigned short int>(size, 0, 255, seed + seed_value_addition);

      size_t denominators[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};

      for (const auto& denom : denominators)
      {
        size_t size_a = size / denom;

        thrust::host_vector<T> h_a_keys(random_keys.begin(), random_keys.begin() + size_a);
        thrust::host_vector<T> h_b_keys(random_keys.begin() + size_a, random_keys.end());

        thrust::host_vector<T> h_a_vals(random_vals.begin(), random_vals.begin() + size_a);
        thrust::host_vector<T> h_b_vals(random_vals.begin() + size_a, random_vals.end());

        THRUST_IF_CONSTEXPR (std::is_void<compare_function>::value)
        {
          thrust::stable_sort(h_a_keys.begin(), h_a_keys.end());
          thrust::stable_sort(h_b_keys.begin(), h_b_keys.end());
        }
        else
        {
          using C = std::conditional_t<std::is_void<compare_function>::value, thrust::less<T>, compare_function>;
          thrust::stable_sort(h_a_keys.begin(), h_a_keys.end(), C{});
          thrust::stable_sort(h_b_keys.begin(), h_b_keys.end(), C{});
        }

        const thrust::device_vector<T> d_a_keys = h_a_keys;
        const thrust::device_vector<T> d_b_keys = h_b_keys;

        const thrust::device_vector<T> d_a_vals = h_a_vals;
        const thrust::device_vector<T> d_b_vals = h_b_vals;

        thrust::host_vector<T> h_result_keys(size);
        thrust::host_vector<T> h_result_vals(size);

        thrust::device_vector<T> d_result_keys(size);
        thrust::device_vector<T> d_result_vals(size);

        const auto h_end = call_merge_by_key<T, compare_function>(
          h_a_keys.begin(),
          h_a_keys.end(),
          h_b_keys.begin(),
          h_b_keys.end(),
          h_a_vals.begin(),
          h_b_vals.begin(),
          h_result_keys.begin(),
          h_result_vals.begin());
        h_result_keys.erase(h_end.first, h_result_keys.end());
        h_result_vals.erase(h_end.second, h_result_vals.end());

        const auto d_end = call_merge_by_key<T, compare_function>(
          d_a_keys.begin(),
          d_a_keys.end(),
          d_b_keys.begin(),
          d_b_keys.end(),
          d_a_vals.begin(),
          d_b_vals.begin(),
          d_result_keys.begin(),
          d_result_vals.begin());
        d_result_keys.erase(d_end.first, d_result_keys.end());
        d_result_vals.erase(d_end.second, d_result_vals.end());

        thrust::host_vector<T> d_result_keys_h = d_result_keys;
        thrust::host_vector<T> d_result_vals_h = d_result_vals;

        ASSERT_EQ(h_result_keys, d_result_keys_h);
        ASSERT_EQ(h_result_vals, d_result_vals_h);
        ASSERT_TRUE(h_end.first == h_result_keys.end());
        ASSERT_TRUE(h_end.second == h_result_vals.end());
        ASSERT_TRUE(d_end.first == d_result_keys.end());
        ASSERT_TRUE(d_end.second == d_result_vals.end());
      }
    }
  }
}

TYPED_TEST(PrimitiveMergeByKeyTests, MergeByKeyToDiscardIterator)
{
  using T = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  for (auto size : get_sizes())
  {
    SCOPED_TRACE(testing::Message() << "with size= " << size);

    for (auto seed : get_seeds())
    {
      SCOPED_TRACE(testing::Message() << "with seed= " << seed);

      auto h_a_keys = get_random_data<T>(size, get_default_limits<T>::min(), get_default_limits<T>::max(), seed);
      auto h_b_keys = get_random_data<T>(
        size, get_default_limits<T>::min(), get_default_limits<T>::max(), seed + seed_value_addition);

      const auto h_a_vals = get_random_data<T>(
        size, get_default_limits<T>::min(), get_default_limits<T>::max(), seed + 2 * seed_value_addition);
      const auto h_b_vals = get_random_data<T>(
        size, get_default_limits<T>::min(), get_default_limits<T>::max(), seed + 3 * seed_value_addition);

      thrust::stable_sort(h_a_keys.begin(), h_a_keys.end());
      thrust::stable_sort(h_b_keys.begin(), h_b_keys.end());

      const thrust::device_vector<T> d_a_keys = h_a_keys;
      const thrust::device_vector<T> d_b_keys = h_b_keys;

      const thrust::device_vector<T> d_a_vals = h_a_vals;
      const thrust::device_vector<T> d_b_vals = h_b_vals;

      using discard_pair = thrust::pair<thrust::discard_iterator<>, thrust::discard_iterator<>>;

      const discard_pair h_result = thrust::merge_by_key(
        h_a_keys.begin(),
        h_a_keys.end(),
        h_b_keys.begin(),
        h_b_keys.end(),
        h_a_vals.begin(),
        h_b_vals.begin(),
        thrust::make_discard_iterator(),
        thrust::make_discard_iterator());

      const discard_pair d_result = thrust::merge_by_key(
        d_a_keys.begin(),
        d_a_keys.end(),
        d_b_keys.begin(),
        d_b_keys.end(),
        d_a_vals.begin(),
        d_b_vals.begin(),
        thrust::make_discard_iterator(),
        thrust::make_discard_iterator());

      thrust::discard_iterator<> reference(2 * size);

      ASSERT_EQ(reference, h_result.first);
      ASSERT_EQ(reference, h_result.second);
      ASSERT_EQ(reference, d_result.first);
      ASSERT_EQ(reference, d_result.second);
    }
  }
}

template <class T>
__global__ THRUST_HIP_LAUNCH_BOUNDS_DEFAULT void
MergeByKeyKernel(int const N, T* keys_A, T* keys_B, T* values_A, T* values_B, T* keys_result, T* values_result)
{
  if (threadIdx.x == 0)
  {
    thrust::device_ptr<int> keyA_begin(keys_A);
    thrust::device_ptr<int> keyA_end(keys_A + N);
    thrust::device_ptr<int> keyB_begin(keys_B);
    thrust::device_ptr<int> keyB_end(keys_B + N);
    thrust::device_ptr<int> valuesA_begin(values_A);
    thrust::device_ptr<int> valuesB_begin(values_B);
    thrust::device_ptr<int> keys_result_begin(keys_result);
    thrust::device_ptr<int> values_result_begin(values_result);

    thrust::merge_by_key(
      thrust::hip::par,
      keyA_begin,
      keyA_end,
      keyB_begin,
      keyB_end,
      valuesA_begin,
      valuesB_begin,
      keys_result_begin,
      values_result_begin);
  }
}
TEST(MergeByKeyTests, TestMergeByKeyDevice)
{
  using T = int;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  for (auto size : get_sizes())
  {
    SCOPED_TRACE(testing::Message() << "with size= " << size);

    for (auto seed : get_seeds())
    {
      SCOPED_TRACE(testing::Message() << "with seed= " << seed);

      thrust::host_vector<T> h_keys_a =
        get_random_data<T>(size, get_default_limits<T>::min(), get_default_limits<T>::max(), seed);
      thrust::host_vector<T> h_keys_b = get_random_data<T>(
        size, get_default_limits<T>::min(), get_default_limits<T>::max(), seed + seed_value_addition);

      thrust::host_vector<T> h_values_a =
        get_random_data<T>(size, get_default_limits<T>::min(), get_default_limits<T>::max(), seed);
      thrust::host_vector<T> h_values_b = get_random_data<T>(
        size, get_default_limits<T>::min(), get_default_limits<T>::max(), seed + seed_value_addition);

      thrust::stable_sort(h_keys_a.begin(), h_keys_a.end(), thrust::greater<T>());
      thrust::stable_sort(h_keys_b.begin(), h_keys_b.end(), thrust::greater<T>());

      thrust::device_vector<T> d_keys_a = h_keys_a;
      thrust::device_vector<T> d_keys_b = h_keys_b;

      thrust::device_vector<T> d_values_a = h_values_a;
      thrust::device_vector<T> d_values_b = h_values_b;

      thrust::host_vector<T> h_keys_result(h_keys_a.size() + h_keys_b.size());
      thrust::device_vector<T> d_keys_result(d_keys_a.size() + d_keys_b.size());

      thrust::host_vector<T> h_values_result(h_values_a.size() + h_values_b.size());
      thrust::device_vector<T> d_values_result(d_values_a.size() + d_values_b.size());

      typename thrust::host_vector<T>::iterator h_end;
      typename thrust::device_vector<T>::iterator d_end;

      thrust::merge_by_key(
        h_keys_a.begin(),
        h_keys_a.end(),
        h_keys_b.begin(),
        h_keys_b.end(),
        h_values_a.begin(),
        h_values_b.begin(),
        h_keys_result.begin(),
        h_values_result.begin());

      hipLaunchKernelGGL(
        MergeByKeyKernel,
        dim3(1, 1, 1),
        dim3(128, 1, 1),
        0,
        0,
        size,
        thrust::raw_pointer_cast(&d_keys_a[0]),
        thrust::raw_pointer_cast(&d_keys_b[0]),
        thrust::raw_pointer_cast(&d_values_a[0]),
        thrust::raw_pointer_cast(&d_values_b[0]),
        thrust::raw_pointer_cast(&d_keys_result[0]),
        thrust::raw_pointer_cast(&d_values_result[0]));

      ASSERT_EQ(h_keys_result, d_keys_result);
      ASSERT_EQ(h_values_result, d_values_result);
    }
  }
}
