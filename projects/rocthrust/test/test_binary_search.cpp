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

#include <thrust/binary_search.h>
#include <thrust/iterator/retag.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/universal_vector.h>

#include "test_param_fixtures.hpp"
#include "test_real_assertions.hpp"
#include "test_utils.hpp"

TESTS_DEFINE(SingleValueTests, NumericalTestsParams);

template <typename T>
bool host_compare(const T& lhs, const T& rhs)
{
  return lhs < rhs;
}

template <typename T>
__device__ bool device_compare(const T& lhs, const T& rhs)
{
  return lhs < rhs;
}

template <typename T, size_t items_per_thread, size_t block_size, class DeviceFunc>
__global__ THRUST_HIP_LAUNCH_BOUNDS_DEFAULT void
single_value_kernel(T* device_search, T* device_input, size_t* device_output, const size_t N, DeviceFunc f)
{
  constexpr size_t items_per_block = items_per_thread * block_size;
  const size_t offset              = (blockIdx.x * items_per_block) + (threadIdx.x * items_per_thread);

  for (size_t i = 0; i < items_per_thread; i++)
  {
    device_output[offset + i] = f(device_search, device_search + N, device_input[offset + i]);
  }
}

template <typename T, class ExpectedFunction, class ThrustDeviceFunction, class ThrustHostFunction>
void RunSingleValueTest(const ExpectedFunction& ef, const ThrustDeviceFunction& df, const ThrustHostFunction& hf)
{
  constexpr size_t grid_size        = 1024;
  constexpr size_t items_per_thread = 12;
  constexpr size_t block_size       = 32;
  constexpr size_t items_per_block  = items_per_thread * block_size;
  constexpr size_t size             = items_per_block * grid_size;

  double maxi = static_cast<double>(std::numeric_limits<T>::max());
  double mini = static_cast<double>(std::numeric_limits<T>::min());

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dis(mini, maxi);

  T* host_search = new T[size];
  T* host_input  = new T[size];
  for (size_t i = 0; i < size; i++)
  {
    host_search[i] = static_cast<T>(dis(gen));
    host_input[i]  = static_cast<T>(dis(gen));
  }

  std::sort(host_search, host_search + size);

  size_t* host_expected = new size_t[size];
  for (size_t i = 0; i < size; i++)
  {
    host_expected[i] = ef(host_search, host_search + size, host_input[i]);
  }

  size_t* host_thrust_output = new size_t[size];
  for (size_t i = 0; i < size; i++)
  {
    host_thrust_output[i] = hf(host_search, host_search + size, host_input[i]);
  }

  T* device_search;
  T* device_input;
  HIP_CHECK(hipMalloc(&device_search, sizeof(T) * size));
  HIP_CHECK(hipMalloc(&device_input, sizeof(size_t) * size));
  HIP_CHECK(hipMemcpy(device_search, host_search, sizeof(T) * size, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(device_input, host_input, sizeof(T) * size, hipMemcpyHostToDevice));

  size_t* device_output;
  HIP_CHECK(hipMalloc(&device_output, sizeof(size_t) * size));

  hipLaunchKernelGGL(
    HIP_KERNEL_NAME(single_value_kernel<T, items_per_thread, block_size>),
    dim3(grid_size),
    dim3(block_size),
    0,
    0,
    device_search,
    device_input,
    device_output,
    size,
    df);

  size_t* device_thrust_output = new size_t[size];
  HIP_CHECK(hipMemcpy(device_thrust_output, device_output, sizeof(size_t) * size, hipMemcpyDeviceToHost));

  for (size_t i = 0; i < size; i++)
  {
    ASSERT_EQ(host_expected[i], device_thrust_output[i]);
    ASSERT_EQ(host_expected[i], host_thrust_output[i]);
  }

  delete[] host_search;
  delete[] host_expected;
  delete[] host_thrust_output;
  delete[] device_thrust_output;

  HIP_CHECK(hipFree(device_search));
  HIP_CHECK(hipFree(device_input));
  HIP_CHECK(hipFree(device_output));
}

TYPED_TEST(SingleValueTests, LowerBound)
{
  using T = typename TestFixture::input_type;
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  RunSingleValueTest<T>(
    [=](T* begin, T* end, const T& value) {
      return std::lower_bound(begin, end, value) - begin;
    },
    [=] __device__(T * begin, T * end, const T& value) {
      return thrust::lower_bound(thrust::device, begin, end, value) - begin;
    },
    [=](T* begin, T* end, const T& value) {
      return thrust::lower_bound(begin, end, value) - begin;
    });
}

TYPED_TEST(SingleValueTests, LowerBoundWithCustomComp)
{
  using T = typename TestFixture::input_type;
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  RunSingleValueTest<T>(
    [=](T* begin, T* end, const T& value) {
      return std::lower_bound(begin, end, value, host_compare<T>) - begin;
    },
    [=] __device__(T * begin, T * end, const T& value) {
      return thrust::lower_bound(thrust::device, begin, end, value, device_compare<T>) - begin;
    },
    [=](T* begin, T* end, const T& value) {
      return thrust::lower_bound(begin, end, value, host_compare<T>) - begin;
    });
}

TYPED_TEST(SingleValueTests, UpperBound)
{
  using T = typename TestFixture::input_type;
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  RunSingleValueTest<T>(
    [=](T* begin, T* end, const T& value) {
      return std::upper_bound(begin, end, value) - begin;
    },
    [=] __device__(T * begin, T * end, const T& value) {
      return thrust::upper_bound(thrust::device, begin, end, value) - begin;
    },
    [=](T* begin, T* end, const T& value) {
      return thrust::upper_bound(begin, end, value) - begin;
    });
}

TYPED_TEST(SingleValueTests, UpperBoundWithCustomComp)
{
  using T = typename TestFixture::input_type;
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  RunSingleValueTest<T>(
    [=](T* begin, T* end, const T& value) {
      return std::upper_bound(begin, end, value, host_compare<T>) - begin;
    },
    [=] __device__(T * begin, T * end, const T& value) {
      return thrust::upper_bound(thrust::device, begin, end, value, device_compare<T>) - begin;
    },
    [=](T* begin, T* end, const T& value) {
      return thrust::upper_bound(begin, end, value, host_compare<T>) - begin;
    });
}

TYPED_TEST(SingleValueTests, TestSingleValueBinarySearch)
{
  using T = typename TestFixture::input_type;
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  RunSingleValueTest<T>(
    [=](T* begin, T* end, const T& value) {
      return std::binary_search(begin, end, value);
    },
    [=] __device__(T * begin, T * end, const T& value) {
      return thrust::binary_search(thrust::device, begin, end, value);
    },
    [=](T* begin, T* end, const T& value) {
      return thrust::binary_search(begin, end, value);
    });
}

TYPED_TEST(SingleValueTests, TestSingleValueBinarySearchWithCustomComp)
{
  using T = typename TestFixture::input_type;
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  RunSingleValueTest<T>(
    [=](T* begin, T* end, const T& value) {
      return std::binary_search(begin, end, value, host_compare<T>);
    },
    [=] __device__(T * begin, T * end, const T& value) {
      return thrust::binary_search(thrust::device, begin, end, value, device_compare<T>);
    },
    [=](T* begin, T* end, const T& value) {
      return thrust::binary_search(begin, end, value, host_compare<T>);
    });
}

TYPED_TEST(SingleValueTests, EqualRange)
{
  using T = typename TestFixture::input_type;
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  RunSingleValueTest<T>(
    [=](T* begin, T* end, const T& value) {
      auto out = std::equal_range(begin, end, value);
      return out.second - out.first;
    },
    [=] __device__(T * begin, T * end, const T& value) {
      auto out = thrust::equal_range(thrust::device, begin, end, value);
      return out.second - out.first;
    },
    [=](T* begin, T* end, const T& value) {
      auto out = thrust::equal_range(begin, end, value);
      return out.second - out.first;
    });
}

TYPED_TEST(SingleValueTests, EqualRangeWithCustomComp)
{
  using T = typename TestFixture::input_type;
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  RunSingleValueTest<T>(
    [=](T* begin, T* end, const T& value) {
      auto out = std::equal_range(begin, end, value, host_compare<T>);
      return out.second - out.first;
    },
    [=] __device__(T * begin, T * end, const T& value) {
      auto out = thrust::equal_range(thrust::device, begin, end, value, device_compare<T>);
      return out.second - out.first;
    },
    [=](T* begin, T* end, const T& value) {
      auto out = thrust::equal_range(begin, end, value, host_compare<T>);
      return out.second - out.first;
    });
}

TESTS_DEFINE(VectorTests, NumericalTestsParams);

template <typename T, size_t items_per_thread, size_t block_size, class DeviceFunc>
__global__ THRUST_HIP_LAUNCH_BOUNDS_DEFAULT void
multiple_value_kernel(T* device_search_arr, T* device_input, size_t* device_output, const DeviceFunc& f)
{
  constexpr size_t items_per_block = items_per_thread * block_size;
  const size_t offset              = (items_per_block * blockIdx.x) + (items_per_thread * threadIdx.x);

  f(device_search_arr,
    device_search_arr + items_per_block,
    device_input + offset,
    device_input + offset + items_per_thread,
    device_output + offset);
}

template <typename T, class ExpectedFunction, class ThrustDeviceFunction, class ThrustHostFunction>
void RunVectorTest(const ExpectedFunction& ef, const ThrustDeviceFunction& df, const ThrustHostFunction& hf)
{
  constexpr size_t grid_size        = 1024;
  constexpr size_t items_per_thread = 12;
  constexpr size_t block_size       = 32;
  constexpr size_t items_per_block  = items_per_thread * block_size;
  constexpr size_t size             = items_per_block * grid_size;

  double maxi = static_cast<double>(std::numeric_limits<T>::max());
  double mini = static_cast<double>(std::numeric_limits<T>::min());

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dis(mini, maxi);

  T* host_search_arr = new T[items_per_block];
  for (size_t i = 0; i < items_per_block; i++)
  {
    host_search_arr[i] = static_cast<T>(dis(gen));
  }

  std::sort(host_search_arr, host_search_arr + items_per_block);

  T* host_input = new T[size];
  for (size_t grid_idx = 0; grid_idx < grid_size; grid_idx++)
  {
    size_t offset = grid_idx * items_per_block;
    for (size_t i = 0; i < items_per_block; i++)
    {
      host_input[offset + i] = static_cast<T>(dis(gen));
    }
  }

  size_t* thrust_host_output   = new size_t[size];
  size_t* thrust_device_output = new size_t[size];
  size_t* expected_output      = new size_t[size];

  for (size_t bIdx = 0; bIdx < grid_size; bIdx++)
  {
    size_t offset = bIdx * items_per_block;

    // getting thrust host output
    hf(host_search_arr,
       host_search_arr + items_per_block,
       host_input + offset,
       host_input + offset + items_per_block,
       thrust_host_output + offset);
    // getting expected output
    for (size_t i = 0; i < items_per_block; i++)
    {
      expected_output[offset + i] = ef(host_search_arr, host_search_arr + items_per_block, host_input[offset + i]);
    }
  }

  // getting thrust device output
  T *device_search_arr, *device_input;
  size_t* device_output;
  HIP_CHECK(hipMalloc(&device_search_arr, sizeof(T) * items_per_block));
  HIP_CHECK(hipMalloc(&device_input, sizeof(T) * size));
  HIP_CHECK(hipMalloc(&device_output, sizeof(size_t) * size));

  HIP_CHECK(hipMemcpy(device_search_arr, host_search_arr, sizeof(T) * items_per_block, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(device_input, host_input, sizeof(T) * size, hipMemcpyHostToDevice));

  hipLaunchKernelGGL(
    HIP_KERNEL_NAME(multiple_value_kernel<T, items_per_thread, block_size>),
    dim3(grid_size),
    dim3(block_size),
    0,
    0,
    device_search_arr,
    device_input,
    device_output,
    df);
  HIP_CHECK(hipMemcpy(thrust_device_output, device_output, sizeof(size_t) * size, hipMemcpyDeviceToHost));

  for (size_t i = 0; i < size; i++)
  {
    ASSERT_EQ(expected_output[i], thrust_host_output[i]);
    ASSERT_EQ(expected_output[i], thrust_device_output[i]);
  }

  delete[] host_search_arr;
  delete[] host_input;
  delete[] thrust_host_output;
  delete[] thrust_device_output;
  delete[] expected_output;

  HIP_CHECK(hipFree(device_search_arr));
  HIP_CHECK(hipFree(device_input));
  HIP_CHECK(hipFree(device_output));
}

TYPED_TEST(VectorTests, LowerBound)
{
  using T = typename TestFixture::input_type;
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  RunVectorTest<T>(
    [=](T* begin, T* end, const T& value) {
      return std::lower_bound(begin, end, value) - begin;
    },
    [=] __device__(T * s_begin, T * s_end, T * i_begin, T * i_end, size_t * out) {
      thrust::lower_bound(thrust::device, s_begin, s_end, i_begin, i_end, out);
    },
    [=](T* s_begin, T* s_end, T* i_begin, T* i_end, size_t* out) {
      thrust::lower_bound(s_begin, s_end, i_begin, i_end, out);
    });
}

TYPED_TEST(VectorTests, LowerBoundWithCustomComp)
{
  using T = typename TestFixture::input_type;
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  RunVectorTest<T>(
    [=](T* begin, T* end, const T& value) {
      return std::lower_bound(begin, end, value, host_compare<T>) - begin;
    },
    [=] __device__(T * s_begin, T * s_end, T * i_begin, T * i_end, size_t * out) {
      thrust::lower_bound(thrust::device, s_begin, s_end, i_begin, i_end, out, device_compare<T>);
    },
    [=](T* s_begin, T* s_end, T* i_begin, T* i_end, size_t* out) {
      thrust::lower_bound(s_begin, s_end, i_begin, i_end, out, host_compare<T>);
    });
}

TYPED_TEST(VectorTests, UpperBound)
{
  using T = typename TestFixture::input_type;
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  RunVectorTest<T>(
    [=](T* begin, T* end, const T& value) {
      return std::upper_bound(begin, end, value) - begin;
    },
    [=] __device__(T * s_begin, T * s_end, T * i_begin, T * i_end, size_t * out) {
      thrust::upper_bound(thrust::device, s_begin, s_end, i_begin, i_end, out);
    },
    [=](T* s_begin, T* s_end, T* i_begin, T* i_end, size_t* out) {
      thrust::upper_bound(s_begin, s_end, i_begin, i_end, out);
    });
}

TYPED_TEST(VectorTests, UpperBoundWithCustomComp)
{
  using T = typename TestFixture::input_type;
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  RunVectorTest<T>(
    [=](T* begin, T* end, const T& value) {
      return std::upper_bound(begin, end, value, host_compare<T>) - begin;
    },
    [=] __device__(T * s_begin, T * s_end, T * i_begin, T * i_end, size_t * out) {
      thrust::upper_bound(thrust::device, s_begin, s_end, i_begin, i_end, out, device_compare<T>);
    },
    [=](T* s_begin, T* s_end, T* i_begin, T* i_end, size_t* out) {
      thrust::upper_bound(s_begin, s_end, i_begin, i_end, out, host_compare<T>);
    });
}

TYPED_TEST(VectorTests, BinarySearch)
{
  using T = typename TestFixture::input_type;
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  RunVectorTest<T>(
    [=](T* begin, T* end, const T& value) {
      return std::binary_search(begin, end, value);
    },
    [=] __device__(T * s_begin, T * s_end, T * i_begin, T * i_end, size_t * out) {
      thrust::binary_search(thrust::device, s_begin, s_end, i_begin, i_end, out);
    },
    [=](T* s_begin, T* s_end, T* i_begin, T* i_end, size_t* out) {
      thrust::binary_search(s_begin, s_end, i_begin, i_end, out);
    });
}

TYPED_TEST(VectorTests, BinarySearchWithCustomComp)
{
  using T = typename TestFixture::input_type;
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  RunVectorTest<T>(
    [=](T* begin, T* end, const T& value) {
      return std::binary_search(begin, end, value, host_compare<T>);
    },
    [=] __device__(T * s_begin, T * s_end, T * i_begin, T * i_end, size_t * out) {
      thrust::binary_search(thrust::device, s_begin, s_end, i_begin, i_end, out, device_compare<T>);
    },
    [=](T* s_begin, T* s_end, T* i_begin, T* i_end, size_t* out) {
      thrust::binary_search(s_begin, s_end, i_begin, i_end, out, host_compare<T>);
    });
}

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

TESTS_DEFINE(BinarySearchTests, FullTestsParams);
TESTS_DEFINE(BinarySearchVectorTests, VectorTestsParams);

THRUST_DIAG_PUSH
THRUST_DIAG_SUPPRESS_MSVC(4244 4267) // possible loss of data

//////////////////////
// Scalar Functions //
//////////////////////

template <typename T>
struct init_scalar
{
  T operator()(T t)
  {
    return t;
  }
};

template <typename T>
struct init_tuple
{
  thrust::tuple<T, T> operator()(T t)
  {
    return thrust::make_tuple(t, t);
  }
};

template <class Vector, class Initializer>
void TestScalarLowerBoundSimple(Initializer init)
{
  Vector vec{init(0), init(2), init(5), init(7), init(8)};

  ASSERT_EQ(thrust::lower_bound(vec.begin(), vec.end(), init(0)) - vec.begin(), 0);
  ASSERT_EQ(thrust::lower_bound(vec.begin(), vec.end(), init(1)) - vec.begin(), 1);
  ASSERT_EQ(thrust::lower_bound(vec.begin(), vec.end(), init(2)) - vec.begin(), 1);
  ASSERT_EQ(thrust::lower_bound(vec.begin(), vec.end(), init(3)) - vec.begin(), 2);
  ASSERT_EQ(thrust::lower_bound(vec.begin(), vec.end(), init(4)) - vec.begin(), 2);
  ASSERT_EQ(thrust::lower_bound(vec.begin(), vec.end(), init(5)) - vec.begin(), 2);
  ASSERT_EQ(thrust::lower_bound(vec.begin(), vec.end(), init(6)) - vec.begin(), 3);
  ASSERT_EQ(thrust::lower_bound(vec.begin(), vec.end(), init(7)) - vec.begin(), 3);
  ASSERT_EQ(thrust::lower_bound(vec.begin(), vec.end(), init(8)) - vec.begin(), 4);
  ASSERT_EQ(thrust::lower_bound(vec.begin(), vec.end(), init(9)) - vec.begin(), 5);
}

TYPED_TEST(BinarySearchTests, TestScalarLowerBoundSimple)
{
  using Vector = typename TestFixture::input_type;
  using T      = typename Vector::value_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  TestScalarLowerBoundSimple<Vector>(init_scalar<T>());
}

TEST(BinarySearchTests, TestTupleLowerBoundSimple)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  {
    using Vector = thrust::device_vector<thrust::tuple<int, int>>;
    TestScalarLowerBoundSimple<Vector>(init_tuple<int>());
  }
  {
    using Vector = thrust::host_vector<thrust::tuple<int, int>>;
    TestScalarLowerBoundSimple<Vector>(init_tuple<int>());
  }
}

// accepts device_vector
template <typename Vector, typename Initializer>
void test_scalar_lower_bound_haystack(Initializer init)
{
  Vector haystack{init(0), init(2), init(5), init(7), init(8)};

  Vector needles{init(1), init(6)};

  thrust::device_vector<int> indices(needles.size());

  thrust::lower_bound(haystack.begin(), haystack.end(), needles.begin(), needles.end(), indices.begin());

  thrust::device_vector<int> expected{1, 3};

  ASSERT_EQ(indices, expected);
}

TEST(BinarySearchTests, TestTupleLowerBoundHayStack)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  {
    using Vector = thrust::device_vector<int>;
    test_scalar_lower_bound_haystack<Vector>(init_scalar<int>());
  }
  {
    using Vector = thrust::device_vector<thrust::tuple<int, int>>;
    test_scalar_lower_bound_haystack<Vector>(init_tuple<int>());
  }
}

template <typename ForwardIterator, typename LessThanComparable>
ForwardIterator
lower_bound(my_system& system, ForwardIterator first, ForwardIterator /*last*/, const LessThanComparable& /*value*/)
{
  system.validate_dispatch();
  return first;
}

TEST(BinarySearchTests, TestScalarLowerBoundDispatchExplicit)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::device_vector<int> vec(1);

  my_system sys(0);
  thrust::lower_bound(sys, vec.begin(), vec.end(), 0);

  ASSERT_EQ(true, sys.is_valid());
}

template <typename ForwardIterator, typename LessThanComparable>
ForwardIterator lower_bound(my_tag, ForwardIterator first, ForwardIterator /*last*/, const LessThanComparable& /*value*/)
{
  *first = 13;
  return first;
}

TEST(BinarySearchTests, TestScalarLowerBoundDispatchImplicit)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::device_vector<int> vec(1);

  thrust::lower_bound(thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.end()), 0);

  ASSERT_EQ(13, vec.front());
}

template <class Vector, typename Initializer>
void TestScalarUpperBoundSimple(Initializer init)
{
  Vector vec{init(0), init(2), init(5), init(7), init(8)};

  ASSERT_EQ(thrust::upper_bound(vec.begin(), vec.end(), init(0)) - vec.begin(), 1);
  ASSERT_EQ(thrust::upper_bound(vec.begin(), vec.end(), init(1)) - vec.begin(), 1);
  ASSERT_EQ(thrust::upper_bound(vec.begin(), vec.end(), init(2)) - vec.begin(), 2);
  ASSERT_EQ(thrust::upper_bound(vec.begin(), vec.end(), init(3)) - vec.begin(), 2);
  ASSERT_EQ(thrust::upper_bound(vec.begin(), vec.end(), init(4)) - vec.begin(), 2);
  ASSERT_EQ(thrust::upper_bound(vec.begin(), vec.end(), init(5)) - vec.begin(), 3);
  ASSERT_EQ(thrust::upper_bound(vec.begin(), vec.end(), init(6)) - vec.begin(), 3);
  ASSERT_EQ(thrust::upper_bound(vec.begin(), vec.end(), init(7)) - vec.begin(), 4);
  ASSERT_EQ(thrust::upper_bound(vec.begin(), vec.end(), init(8)) - vec.begin(), 5);
  ASSERT_EQ(thrust::upper_bound(vec.begin(), vec.end(), init(9)) - vec.begin(), 5);
}

TYPED_TEST(BinarySearchTests, TestScalarUpperBoundSimple)
{
  using Vector = typename TestFixture::input_type;
  using T      = typename Vector::value_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  TestScalarUpperBoundSimple<Vector>(init_scalar<T>());
}

TEST(BinarySearchTests, TestTupleUpperBoundSimple)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  {
    using Vector = thrust::device_vector<thrust::tuple<int, int>>;
    TestScalarUpperBoundSimple<Vector>(init_tuple<int>());
  }
  {
    using Vector = thrust::host_vector<thrust::tuple<int, int>>;
    TestScalarUpperBoundSimple<Vector>(init_tuple<int>());
  }
}

// accepts device_vector
template <typename Vector, typename Initializer>
void test_scalar_upper_bound_haystack(Initializer init)
{
  Vector haystack{init(0), init(2), init(5), init(7), init(8)};

  Vector needles{init(1), init(6)};

  thrust::device_vector<int> indices(needles.size());

  thrust::upper_bound(haystack.begin(), haystack.end(), needles.begin(), needles.end(), indices.begin());

  thrust::device_vector<int> expected{1, 3};

  ASSERT_EQ(indices, expected);
}

TEST(BinarySearchTests, TestTupleUpperBoundHayStack)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  {
    using Vector = thrust::device_vector<int>;
    test_scalar_upper_bound_haystack<Vector>(init_scalar<int>());
  }
  {
    using Vector = thrust::device_vector<thrust::tuple<int, int>>;
    test_scalar_upper_bound_haystack<Vector>(init_tuple<int>());
  }
}

template <typename ForwardIterator, typename LessThanComparable>
ForwardIterator
upper_bound(my_system& system, ForwardIterator first, ForwardIterator /*last*/, const LessThanComparable& /*value*/)
{
  system.validate_dispatch();
  return first;
}

TEST(BinarySearchTests, TestScalarUpperBoundDispatchExplicit)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::device_vector<int> vec(1);

  my_system sys(0);
  thrust::upper_bound(sys, vec.begin(), vec.end(), 0);

  ASSERT_EQ(true, sys.is_valid());
}

template <typename ForwardIterator, typename LessThanComparable>
ForwardIterator upper_bound(my_tag, ForwardIterator first, ForwardIterator /*last*/, const LessThanComparable& /*value*/)
{
  *first = 13;
  return first;
}

TEST(BinarySearchTests, TestScalarUpperBoundDispatchImplicit)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::device_vector<int> vec(1);

  thrust::upper_bound(thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.end()), 0);

  ASSERT_EQ(13, vec.front());
}

template <class Vector, typename Initializer>
void TestScalarBinarySearchSimple(Initializer init)
{
  Vector vec{init(0), init(2), init(5), init(7), init(8)};

  ASSERT_EQ(thrust::binary_search(vec.begin(), vec.end(), init(0)), true);
  ASSERT_EQ(thrust::binary_search(vec.begin(), vec.end(), init(1)), false);
  ASSERT_EQ(thrust::binary_search(vec.begin(), vec.end(), init(2)), true);
  ASSERT_EQ(thrust::binary_search(vec.begin(), vec.end(), init(3)), false);
  ASSERT_EQ(thrust::binary_search(vec.begin(), vec.end(), init(4)), false);
  ASSERT_EQ(thrust::binary_search(vec.begin(), vec.end(), init(5)), true);
  ASSERT_EQ(thrust::binary_search(vec.begin(), vec.end(), init(6)), false);
  ASSERT_EQ(thrust::binary_search(vec.begin(), vec.end(), init(7)), true);
  ASSERT_EQ(thrust::binary_search(vec.begin(), vec.end(), init(8)), true);
  ASSERT_EQ(thrust::binary_search(vec.begin(), vec.end(), init(9)), false);
}

TYPED_TEST(BinarySearchTests, TestScalarBinarySearchSimple)
{
  using Vector = typename TestFixture::input_type;
  using T      = typename Vector::value_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  TestScalarBinarySearchSimple<Vector>(init_scalar<T>());
}

TEST(BinarySearchTests, TestTupleBinarySearchSimple)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  {
    using Vector = thrust::device_vector<thrust::tuple<int, int>>;
    TestScalarBinarySearchSimple<Vector>(init_tuple<int>());
  }
  {
    using Vector = thrust::host_vector<thrust::tuple<int, int>>;
    TestScalarBinarySearchSimple<Vector>(init_tuple<int>());
  }
}

// accepts device_vector
template <typename Vector, typename Initializer>
void test_scalar_binary_search_haystack(Initializer init)
{
  Vector haystack{init(0), init(2), init(5), init(7), init(8)};

  Vector needles{init(3), init(5)};

  thrust::device_vector<bool> indices(needles.size());

  thrust::binary_search(haystack.begin(), haystack.end(), needles.begin(), needles.end(), indices.begin());

  thrust::device_vector<bool> expected{false, true};

  ASSERT_EQ(indices, expected);
}

TEST(BinarySearchTests, TestTupleBinarySearchHayStack)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  {
    using Vector = thrust::device_vector<int>;
    test_scalar_binary_search_haystack<Vector>(init_scalar<int>());
  }
  {
    using Vector = thrust::device_vector<thrust::tuple<int, int>>;
    test_scalar_binary_search_haystack<Vector>(init_tuple<int>());
  }
}

template <typename ForwardIterator, typename LessThanComparable>
bool binary_search(
  my_system& system, ForwardIterator /*first*/, ForwardIterator /*last*/, const LessThanComparable& /*value*/)
{
  system.validate_dispatch();
  return false;
}

TEST(BinarySearchTests, TestScalarBinarySearchDispatchExplicit)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::device_vector<int> vec(1);

  my_system sys(0);
  thrust::binary_search(sys, vec.begin(), vec.end(), 0);

  ASSERT_EQ(true, sys.is_valid());
}

template <typename ForwardIterator, typename LessThanComparable>
bool binary_search(my_tag, ForwardIterator first, ForwardIterator /*last*/, const LessThanComparable& /*value*/)
{
  *first = 13;
  return false;
}

TEST(BinarySearchTests, TestScalarBinarySearchDispatchImplicit)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::device_vector<int> vec(1);

  thrust::binary_search(thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.end()), 0);

  ASSERT_EQ(13, vec.front());
}

TYPED_TEST(BinarySearchVectorTests, TestScalarEqualRangeSimple)
{
  using Vector = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  Vector vec{0, 2, 5, 7, 8};

  ASSERT_EQ(thrust::equal_range(vec.begin(), vec.end(), 0).first - vec.begin(), 0);
  ASSERT_EQ(thrust::equal_range(vec.begin(), vec.end(), 1).first - vec.begin(), 1);
  ASSERT_EQ(thrust::equal_range(vec.begin(), vec.end(), 2).first - vec.begin(), 1);
  ASSERT_EQ(thrust::equal_range(vec.begin(), vec.end(), 3).first - vec.begin(), 2);
  ASSERT_EQ(thrust::equal_range(vec.begin(), vec.end(), 4).first - vec.begin(), 2);
  ASSERT_EQ(thrust::equal_range(vec.begin(), vec.end(), 5).first - vec.begin(), 2);
  ASSERT_EQ(thrust::equal_range(vec.begin(), vec.end(), 6).first - vec.begin(), 3);
  ASSERT_EQ(thrust::equal_range(vec.begin(), vec.end(), 7).first - vec.begin(), 3);
  ASSERT_EQ(thrust::equal_range(vec.begin(), vec.end(), 8).first - vec.begin(), 4);
  ASSERT_EQ(thrust::equal_range(vec.begin(), vec.end(), 9).first - vec.begin(), 5);

  ASSERT_EQ(thrust::equal_range(vec.begin(), vec.end(), 0).second - vec.begin(), 1);
  ASSERT_EQ(thrust::equal_range(vec.begin(), vec.end(), 1).second - vec.begin(), 1);
  ASSERT_EQ(thrust::equal_range(vec.begin(), vec.end(), 2).second - vec.begin(), 2);
  ASSERT_EQ(thrust::equal_range(vec.begin(), vec.end(), 3).second - vec.begin(), 2);
  ASSERT_EQ(thrust::equal_range(vec.begin(), vec.end(), 4).second - vec.begin(), 2);
  ASSERT_EQ(thrust::equal_range(vec.begin(), vec.end(), 5).second - vec.begin(), 3);
  ASSERT_EQ(thrust::equal_range(vec.begin(), vec.end(), 6).second - vec.begin(), 3);
  ASSERT_EQ(thrust::equal_range(vec.begin(), vec.end(), 7).second - vec.begin(), 4);
  ASSERT_EQ(thrust::equal_range(vec.begin(), vec.end(), 8).second - vec.begin(), 5);
  ASSERT_EQ(thrust::equal_range(vec.begin(), vec.end(), 9).second - vec.begin(), 5);
}

template <typename ForwardIterator, typename LessThanComparable>
thrust::pair<ForwardIterator, ForwardIterator>
equal_range(my_system& system, ForwardIterator first, ForwardIterator /*last*/, const LessThanComparable& /*value*/)
{
  system.validate_dispatch();
  return thrust::make_pair(first, first);
}

TEST(BinarySearchTests, TestScalarEqualRangeDispatchExplicit)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::device_vector<int> vec(1);

  my_system sys(0);
  thrust::equal_range(sys, vec.begin(), vec.end(), 0);

  ASSERT_EQ(true, sys.is_valid());
}

template <typename ForwardIterator, typename LessThanComparable>
thrust::pair<ForwardIterator, ForwardIterator>
equal_range(my_tag, ForwardIterator first, ForwardIterator /*last*/, const LessThanComparable& /*value*/)
{
  *first = 13;
  return thrust::make_pair(first, first);
}

TEST(BinarySearchTests, TestScalarEqualRangeDispatchImplicit)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::device_vector<int> vec(1);

  thrust::equal_range(thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.end()), 0);

  ASSERT_EQ(13, vec.front());
}

TEST(BinarySearchTests, TestEqualRangeExecutionPolicy)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  using thrust_exec_policy_t =
    thrust::detail::execute_with_allocator<thrust::device_allocator<char>, thrust::hip_rocprim::execute_on_stream_base>;

  constexpr int data[]  = {1, 2, 3, 4, 4, 5, 6, 7, 8, 9};
  constexpr size_t size = sizeof(data) / sizeof(data[0]);
  constexpr int key     = 4;
  thrust::device_vector<int> d_data(data, data + size);

  thrust::pair<thrust::device_vector<int>::iterator, thrust::device_vector<int>::iterator> range = thrust::equal_range(
    thrust_exec_policy_t(thrust::hip_rocprim::execute_on_stream_base<thrust_exec_policy_t>(hipStreamPerThread),
                         thrust::device_allocator<char>()),
    d_data.begin(),
    d_data.end(),
    key);

  ASSERT_EQ(*range.first, 4);
  ASSERT_EQ(*range.second, 5);
}

__global__ THRUST_HIP_LAUNCH_BOUNDS_DEFAULT void
BinarySearchKernel(int const N, int* in_array, int* result_array, int search_value)
{
  if (threadIdx.x == 0)
  {
    thrust::device_ptr<int> begin(in_array);
    thrust::device_ptr<int> end(in_array + N);
    result_array[search_value] = thrust::binary_search(thrust::hip::par, begin, end, search_value);
  }
}

TEST(BinarySearchTests, TestBinarySearchDevice)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());
  for (auto size : get_sizes())
  {
    SCOPED_TRACE(testing::Message() << "with size= " << size);

    for (auto seed : get_seeds())
    {
      SCOPED_TRACE(testing::Message() << "with seed= " << seed);

      thrust::host_vector<int> h_data   = get_random_data<int>(size, 0, size, seed);
      thrust::device_vector<int> d_data = h_data;

      thrust::host_vector<int> h_result(size * 2, -1);
      thrust::device_vector<int> d_result(size * 2, -1);

      for (int search_value = 0; search_value < (int) size * 2; search_value++)
      {
        SCOPED_TRACE(testing::Message() << "searching for " << search_value);

        h_result[search_value] = thrust::binary_search(h_data.begin(), h_data.end(), search_value);
        hipLaunchKernelGGL(
          BinarySearchKernel,
          dim3(1, 1, 1),
          dim3(128, 1, 1),
          0,
          0,
          size,
          thrust::raw_pointer_cast(&d_data[0]),
          thrust::raw_pointer_cast(&d_result[0]),
          search_value);
      }
      ASSERT_EQ(h_result, d_result);
    }
  }
}

THRUST_DIAG_POP

void TestBoundsWithBigIndexesHelper(int magnitude)
{
  thrust::counting_iterator<long long> begin(1);
  thrust::counting_iterator<long long> end = begin + (1ll << magnitude);
  ASSERT_EQ(thrust::distance(begin, end), 1ll << magnitude);

  thrust::detail::intmax_t distance_low_value =
    thrust::distance(begin, thrust::lower_bound(thrust::device, begin, end, 17));

  thrust::detail::intmax_t distance_high_value =
    thrust::distance(begin, thrust::lower_bound(thrust::device, begin, end, (1ll << magnitude) - 17));

  ASSERT_EQ(distance_low_value, 16);
  ASSERT_EQ(distance_high_value, (1ll << magnitude) - 18);

  distance_low_value = thrust::distance(begin, thrust::upper_bound(thrust::device, begin, end, 17));

  distance_high_value =
    thrust::distance(begin, thrust::upper_bound(thrust::device, begin, end, (1ll << magnitude) - 17));

  ASSERT_EQ(distance_low_value, 17);
  ASSERT_EQ(distance_high_value, (1ll << magnitude) - 17);
}

TEST(BinarySearchTests, TestBoundsWithBigIndexes)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  TestBoundsWithBigIndexesHelper(30);
  TestBoundsWithBigIndexesHelper(31);
  TestBoundsWithBigIndexesHelper(32);
  TestBoundsWithBigIndexesHelper(33);
}
