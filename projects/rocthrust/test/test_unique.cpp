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
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/retag.h>
#include <thrust/unique.h>

#include "test_param_fixtures.hpp"
#include "test_real_assertions.hpp"
#include "test_utils.hpp"

#if _THRUST_HAS_DEVICE_SYSTEM_STD
#  include _THRUST_STD_INCLUDE(array)
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
typename thrust::iterator_traits<ForwardIterator>::difference_type
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
typename thrust::iterator_traits<ForwardIterator>::difference_type unique_count(my_tag, ForwardIterator, ForwardIterator)
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

      ASSERT_EQ(h_new_last - h_data.begin(), d_output_size[0]);

      h_data.resize(h_new_last - h_data.begin());
      d_data.resize(d_output_size[0]);

      ASSERT_EQ(h_data, d_data);
    }
  }
}
