/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *  Modifications Copyright© 2019-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/distance.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/retag.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/unique.h>

#include _THRUST_STD_INCLUDE(functional)
#include _THRUST_STD_INCLUDE(utility)

#include <cstdint>
#include <new>

#include "test_param_fixtures.hpp"
#include "test_real_assertions.hpp"
#include "test_utils.hpp"

#if _THRUST_HAS_DEVICE_SYSTEM_STD
#  include _THRUST_STD_INCLUDE(array)
#else
#  include <iterator>
#endif

TESTS_DEFINE(UniqueTests, FullTestsParams);

TESTS_DEFINE(UniqueIntegralTests, IntegerTestsParams);

template <typename ForwardIterator>
ForwardIterator unique(my_system& system, ForwardIterator first, ForwardIterator)
{
  system.validate_dispatch();
  return first;
}

TEST(UniqueTests, TestUniqueDispatchExplicit)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::device_vector<int> vec(1);

  my_system sys(0);
  thrust::unique(sys, vec.begin(), vec.begin());

  ASSERT_EQ(true, sys.is_valid());
}

template <typename ForwardIterator>
ForwardIterator unique(my_tag, ForwardIterator first, ForwardIterator)
{
  *first = 13;
  return first;
}

TEST(UniqueTests, TestUniqueDispatchImplicit)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::device_vector<int> vec(1);

  thrust::unique(thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.begin()));

  ASSERT_EQ(13, vec.front());
}

template <typename InputIterator, typename OutputIterator>
OutputIterator unique_copy(my_system& system, InputIterator, InputIterator, OutputIterator result)
{
  system.validate_dispatch();
  return result;
}

TEST(UniqueTests, TestUniqueCopyDispatchExplicit)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::device_vector<int> vec(1);

  my_system sys(0);
  thrust::unique_copy(sys, vec.begin(), vec.begin(), vec.begin());

  ASSERT_EQ(true, sys.is_valid());
}

template <typename InputIterator, typename OutputIterator>
OutputIterator unique_copy(my_tag, InputIterator, InputIterator, OutputIterator result)
{
  *result = 13;
  return result;
}

TEST(UniqueTests, TestUniqueCopyDispatchImplicit)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::device_vector<int> vec(1);

  thrust::unique_copy(
    thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.begin()));

  ASSERT_EQ(13, vec.front());
}

template <typename ForwardIterator>
typename _THRUST_STD::iterator_traits<ForwardIterator>::difference_type
unique_count(my_system& system, ForwardIterator, ForwardIterator)
{
  system.validate_dispatch();
  return 0;
}

TEST(UniqueTests, TestUniqueCountDispatchExplicit)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::device_vector<int> vec(1);

  my_system sys(0);
  thrust::unique_count(sys, vec.begin(), vec.begin());

  ASSERT_EQ(true, sys.is_valid());
}

template <typename ForwardIterator>
typename _THRUST_STD::iterator_traits<ForwardIterator>::difference_type
unique_count(my_tag, ForwardIterator, ForwardIterator)
{
  return 13;
}

TEST(UniqueTests, TestUniqueCountDispatchImplicit)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::device_vector<int> vec(1);

  auto result = thrust::unique_count(thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.begin()));

  ASSERT_EQ(13, result);
}

template <typename T>
struct is_equal_div_10_unique
{
  THRUST_HOST_DEVICE bool operator()(const T x, const T& y) const
  {
    return ((int) x / 10) == ((int) y / 10);
  }
};

TYPED_TEST(UniqueTests, TestUniqueSimple)
{
  using Vector = typename TestFixture::input_type;
  using T      = typename Vector::value_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  Vector data{11, 11, 12, 20, 29, 21, 21, 31, 31, 37};

  typename Vector::iterator new_last;

  new_last = thrust::unique(data.begin(), data.end());

  ASSERT_EQ(new_last - data.begin(), 7);
  data.resize(7);
  Vector ref{11, 12, 20, 29, 21, 31, 37};
  ASSERT_EQ(data, ref);

  new_last = thrust::unique(data.begin(), new_last, is_equal_div_10_unique<T>());

  ASSERT_EQ(new_last - data.begin(), 3);
  ref.resize(3);
  data.resize(3);
  ref = {11, 20, 31};
  ASSERT_EQ(data, ref);
}

TYPED_TEST(UniqueIntegralTests, TestUnique)
{
  using T = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  for (auto size : get_sizes())
  {
    SCOPED_TRACE(testing::Message() << "with size= " << size);

    for (auto seed : get_seeds())
    {
      SCOPED_TRACE(testing::Message() << "with seed= " << seed);

      thrust::host_vector<T> h_data =
        get_random_data<T>(size, get_default_limits<T>::min(), get_default_limits<T>::max(), seed);
      thrust::device_vector<T> d_data = h_data;

      typename thrust::host_vector<T>::iterator h_new_last;
      typename thrust::device_vector<T>::iterator d_new_last;

      h_new_last = thrust::unique(h_data.begin(), h_data.end());
      d_new_last = thrust::unique(d_data.begin(), d_data.end());

      ASSERT_EQ(h_new_last - h_data.begin(), d_new_last - d_data.begin());

      h_data.resize(h_new_last - h_data.begin());
      d_data.resize(d_new_last - d_data.begin());

      ASSERT_EQ(h_data, d_data);
    }
  }
}

TYPED_TEST(UniqueTests, TestUniqueCopySimple)
{
  using Vector = typename TestFixture::input_type;
  using T      = typename Vector::value_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  Vector data{11, 11, 12, 20, 29, 21, 21, 31, 31, 37};
  Vector output(10, -1);

  typename Vector::iterator new_last;

  new_last = thrust::unique_copy(data.begin(), data.end(), output.begin());

  ASSERT_EQ(new_last - output.begin(), 7);
  output.resize(7);
  Vector ref{11, 12, 20, 29, 21, 31, 37};
  ASSERT_EQ(output, ref);

  new_last = thrust::unique_copy(output.begin(), new_last, data.begin(), is_equal_div_10_unique<T>());

  ASSERT_EQ(new_last - data.begin(), 3);
  ref.resize(3);
  data.resize(3);
  ref = {11, 20, 31};
  ASSERT_EQ(data, ref);
}

TYPED_TEST(UniqueIntegralTests, TestUniqueCopy)
{
  using T = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  for (auto size : get_sizes())
  {
    SCOPED_TRACE(testing::Message() << "with size= " << size);

    for (auto seed : get_seeds())
    {
      SCOPED_TRACE(testing::Message() << "with seed= " << seed);

      thrust::host_vector<T> h_data =
        get_random_data<T>(size, get_default_limits<T>::min(), get_default_limits<T>::max(), seed);
      thrust::device_vector<T> d_data = h_data;

      thrust::host_vector<T> h_output(size);
      thrust::device_vector<T> d_output(size);

      typename thrust::host_vector<T>::iterator h_new_last;
      typename thrust::device_vector<T>::iterator d_new_last;

      h_new_last = thrust::unique_copy(h_data.begin(), h_data.end(), h_output.begin());
      d_new_last = thrust::unique_copy(d_data.begin(), d_data.end(), d_output.begin());

      ASSERT_EQ(h_new_last - h_output.begin(), d_new_last - d_output.begin());

      h_data.resize(h_new_last - h_output.begin());
      d_data.resize(d_new_last - d_output.begin());

      ASSERT_EQ(h_output, d_output);
    }
  }
}

TYPED_TEST(UniqueIntegralTests, TestUniqueCopyToDiscardIterator)
{
  using T = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  for (auto size : get_sizes())
  {
    SCOPED_TRACE(testing::Message() << "with size= " << size);

    for (auto seed : get_seeds())
    {
      SCOPED_TRACE(testing::Message() << "with seed= " << seed);

      thrust::host_vector<T> h_data =
        get_random_data<T>(size, get_default_limits<T>::min(), get_default_limits<T>::max(), seed);
      thrust::device_vector<T> d_data = h_data;

      thrust::host_vector<T> h_unique = h_data;
      h_unique.erase(thrust::unique(h_unique.begin(), h_unique.end()), h_unique.end());

      thrust::discard_iterator<> reference(h_unique.size());

      typename thrust::device_vector<T>::iterator d_new_last;

      thrust::discard_iterator<> h_result =
        thrust::unique_copy(h_data.begin(), h_data.end(), thrust::make_discard_iterator());

      thrust::discard_iterator<> d_result =
        thrust::unique_copy(d_data.begin(), d_data.end(), thrust::make_discard_iterator());

      ASSERT_EQ_QUIET(reference, h_result);
      ASSERT_EQ_QUIET(reference, d_result);
    }
  }
}

TYPED_TEST(UniqueTests, TestUniqueCountSimple)
{
  using Vector = typename TestFixture::input_type;
  using T      = typename Vector::value_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  Vector data{11, 11, 12, 20, 29, 21, 21, 31, 31, 37};

  int count = thrust::unique_count(data.begin(), data.end());

  ASSERT_EQ(count, 7);

  int div_10_count = thrust::unique_count(data.begin(), data.end(), is_equal_div_10_unique<T>());

  ASSERT_EQ(div_10_count, 3);
}

TYPED_TEST(UniqueIntegralTests, TestUniqueCount)
{
  using T = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  for (auto size : get_sizes())
  {
    SCOPED_TRACE(testing::Message() << "with size= " << size);

    thrust::host_vector<T> h_data   = random_integers<bool>(size);
    thrust::device_vector<T> d_data = h_data;

    int h_count{};
    int d_count{};

    h_count = thrust::unique_count(h_data.begin(), h_data.end());
    d_count = thrust::unique_count(d_data.begin(), d_data.end());

    ASSERT_EQ(h_count, d_count);
  }
}

#if _THRUST_HAS_DEVICE_SYSTEM_STD
template <typename T, std::size_t N>
using DeviceArray = _THRUST_STD::array<T, N>;
#else // !_THRUST_HAS_DEVICE_SYSTEM_STD
template <typename T, std::size_t N>
struct DeviceArray
{
  T data[N];

  // Host and device-compatible equality operator
  __host__ __device__ bool operator==(const DeviceArray& other) const
  {
    for (std::size_t i = 0; i < N; ++i)
    {
      if (data[i] != other.data[i])
      {
        return false;
      }
    }
    return true;
  }
};
#endif // _THRUST_HAS_DEVICE_SYSTEM_STD

TYPED_TEST(UniqueTests, TestUniqueMemoryAccess)
{
  using Vector = typename TestFixture::input_type;
  using T      = typename Vector::value_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::device_vector<DeviceArray<T, 100>> v(10);
  thrust::unique(v.begin(), v.end());
}

__global__ THRUST_HIP_LAUNCH_BOUNDS_DEFAULT void UniqueKernel(int const N, int* in_array, int* out_size)
{
  if (threadIdx.x == 0)
  {
    thrust::device_ptr<int> in_begin(in_array);
    thrust::device_ptr<int> in_end(in_array + N);

    thrust::device_vector<int>::iterator last = thrust::unique(thrust::hip::par, in_begin, in_end);
    out_size[0]                               = last - thrust::device_vector<int>::iterator(in_begin);
  }
}

TEST(UniqueTests, TestUniqueDevice)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  for (auto size : get_sizes())
  {
    SCOPED_TRACE(testing::Message() << "with size= " << size);

    for (auto seed : get_seeds())
    {
      SCOPED_TRACE(testing::Message() << "with seed= " << seed);

      thrust::host_vector<int> h_data   = get_random_data<int>(size, 0, 15, seed);
      thrust::device_vector<int> d_data = h_data;
      thrust::device_vector<int> d_output_size(1, 0);

      typename thrust::host_vector<int>::iterator h_new_last;
      typename thrust::device_vector<int>::iterator d_new_last;

      h_new_last = thrust::unique(h_data.begin(), h_data.end());

      hipLaunchKernelGGL(
        UniqueKernel,
        dim3(1, 1, 1),
        dim3(128, 1, 1),
        0,
        0,
        size,
        thrust::raw_pointer_cast(&d_data[0]),
        thrust::raw_pointer_cast(&d_output_size[0]));
      HIP_CHECK(hipGetLastError());

      ASSERT_EQ(h_new_last - h_data.begin(), d_output_size[0]);

      h_data.resize(h_new_last - h_data.begin());
      d_data.resize(d_output_size[0]);

      ASSERT_EQ(h_data, d_data);
    }
  }
}

template <typename T>
struct div_n_equality_op
{
  T div;
  THRUST_HOST_DEVICE bool operator()(const T x, const T& y) const
  {
    return (x / div) == (y / div);
  }
};

template <typename T>
struct multiply_n
{
  T multiplier;
  THRUST_HOST_DEVICE T operator()(T x)
  {
    return x * multiplier;
  }
};

void TestUniqueWithMagnitude(int magnitude)
{
  using offset_t      = std::int64_t;
  using equality_op_t = div_n_equality_op<offset_t>;

  offset_t run_length_of_equal_items = offset_t{10};
  equality_op_t equality_op          = equality_op_t{run_length_of_equal_items};

  // Prepare input
  offset_t num_items = offset_t{1ll} << magnitude;
  thrust::counting_iterator<offset_t> begin(offset_t{0});
  auto end = begin + num_items;
  ASSERT_EQ(static_cast<offset_t>(thrust::distance(begin, end)), num_items);

  offset_t expected_num_unique = (num_items + run_length_of_equal_items - 1) / run_length_of_equal_items;
  thrust::device_vector<offset_t> unique_out(expected_num_unique);
  auto unique_out_end = thrust::unique_copy(begin, end, unique_out.begin(), equality_op);

  // Ensure number of selected items are correct
  offset_t num_selected_out = static_cast<offset_t>(thrust::distance(unique_out.begin(), unique_out_end));
  ASSERT_EQ(num_selected_out, expected_num_unique);
  unique_out.resize(expected_num_unique);

  // Ensure selected items are correct
  auto expected_out_it     = thrust::make_transform_iterator(begin, multiply_n<offset_t>{run_length_of_equal_items});
  bool all_results_correct = thrust::equal(unique_out.begin(), unique_out.end(), expected_out_it);
  ASSERT_TRUE(all_results_correct);
}

TEST(UniqueTests, TestUniqueWithLargeNumberOfItems)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  try
  {
    for (int mag : {30, 31, 32, 33})
    {
      TestUniqueWithMagnitude(mag);
    }
  }
  catch (std::bad_alloc&)
  {
    // if we run out of memory, just skip the test
    GTEST_SKIP() << "Out of device memory";
  }
}

struct check_valid_item_op
{
  std::uint32_t* error_counter{};
  int expected_upper_bound{};

  __device__ bool operator()(const int lhs, const int rhs) const
  {
    if (lhs > expected_upper_bound || rhs > expected_upper_bound)
    {
      if (error_counter)
      {
        atomicAdd(error_counter, 1);
      }
      return false;
    }
    return lhs == rhs;
  }
};

TEST(UniqueTests, TestUniqueWithCustomEqualityOp)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  using Vector = thrust::device_vector<int>;
  using T      = Vector::value_type;

  auto constexpr num_items = 1000;
  auto data                = thrust::make_counting_iterator(T{0});

  thrust::device_vector<std::uint32_t> error_counter(1, 0);
  auto const error_counter_ptr = thrust::raw_pointer_cast(error_counter.data());

  Vector unique_out(num_items);
  auto unique_out_end = thrust::unique_copy(
    data, data + num_items, unique_out.begin(), check_valid_item_op{error_counter_ptr, num_items - 1});

  auto num_selected_out = thrust::distance(unique_out.begin(), unique_out_end);
  ASSERT_EQ(num_selected_out, num_items);
  ASSERT_EQ(error_counter[0], std::uint32_t{0});
  bool all_results_correct = thrust::equal(unique_out.cbegin(), unique_out.cend(), data);
  ASSERT_TRUE(all_results_correct);
}

template <typename F>
struct NonConstAdapter
{
  F f;
  NonConstAdapter(const F& func)
      : f(func)
  {}

  template <typename... Args>
  THRUST_DEVICE auto operator()(Args&&... args) -> decltype(f(_THRUST_STD::forward<Args>(args)...))
  {
    return f(_THRUST_STD::forward<Args>(args)...);
  }
};

TEST(UniqueTests, TestUniqueWithCustomEqualityOpMutable)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::device_vector<int> in = {1, 1, 2, 3, 4, 4, 5};
  thrust::unique(thrust::hip::par, in.begin(), in.end(), NonConstAdapter(_THRUST_STD::equal_to<>{}));
}
