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

#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/retag.h>
#include <thrust/reduce.h>
#include <thrust/universal_vector.h>

#include <limits>

#include "test_param_fixtures.hpp"
#include "test_utils.hpp"

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

TESTS_DEFINE(ReduceTests, FullTestsParams);
TESTS_DEFINE(ReduceIntegerTests, UnsignedIntegerTestsParams);
TESTS_DEFINE(ReducePrimitiveTests, NumericalTestsParams);
TESTS_DEFINE(ReduceVectorUnitTests, VectorTestsParams);

template <typename T>
struct plus_mod_10
{
  THRUST_HOST_DEVICE T operator()(T lhs, T rhs) const
  {
    return ((lhs % 10) + (rhs % 10)) % 10;
  }
};

template <typename T>
struct is_equal_div_10_reduce
{
  THRUST_HOST_DEVICE bool operator()(const T x, const T& y) const
  {
    return ((int) x / 10) == ((int) y / 10);
  }
};

TYPED_TEST(ReduceVectorUnitTests, TestReduceSimple)
{
  using Vector = typename TestFixture::input_type;
  using T      = typename Vector::value_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  Vector v{1, -2, 3};

  // no initializer
  ASSERT_EQ(thrust::reduce(v.begin(), v.end()), 2);

  // with initializer
  ASSERT_EQ(thrust::reduce(v.begin(), v.end(), (T) 10), 12);
}

template <typename InputIterator>
int reduce(my_system& system, InputIterator, InputIterator)
{
  system.validate_dispatch();
  return 13;
}

TEST(ReduceTests, TestReduceDispatchExplicit)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::device_vector<int> vec;

  my_system sys(0);
  thrust::reduce(sys, vec.begin(), vec.end());

  ASSERT_EQ(true, sys.is_valid());
}

template <typename InputIterator>
int reduce(my_tag, InputIterator, InputIterator)
{
  return 13;
}

TEST(ReduceTests, TestReduceDispatchImplicit)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::device_vector<int> vec;

  int result = thrust::reduce(thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.end()));

  ASSERT_EQ(13, result);
}

TYPED_TEST(ReducePrimitiveTests, TestReduce)
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

      T init = 13;

      T h_result = thrust::reduce(h_data.begin(), h_data.end(), init);
      T d_result = thrust::reduce(d_data.begin(), d_data.end(), init);

      test_equality(h_result, d_result, size - 1);
    }
  }
}

TYPED_TEST(ReduceTests, TestReduceMixedTypes)
{
  using Vector = typename TestFixture::input_type;
  using T      = typename Vector::value_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  // make sure we get types for default args and operators correct
  if constexpr (std::is_floating_point<T>::value)
  {
    Vector float_input{1.5, 2.5, 3.5, 4.5};

    // float -> int should use using plus<int> operator by default
    ASSERT_EQ(thrust::reduce(float_input.begin(), float_input.end(), (int) 0), 10);
  }
  else
  {
    Vector int_input{1, 2, 3, 4};
    // int -> float should use using plus<float> operator by default
    ASSERT_EQ(thrust::reduce(int_input.begin(), int_input.end(), (float) 0.5), 10.5);
  }
}

TYPED_TEST(ReduceIntegerTests, TestReduceWithOperator)
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

      T init = 3;

      T cpu_result = thrust::reduce(h_data.begin(), h_data.end(), init, plus_mod_10<T>());
      T gpu_result = thrust::reduce(d_data.begin(), d_data.end(), init, plus_mod_10<T>());

      ASSERT_EQ(cpu_result, gpu_result);
    }
  }
}

template <typename T>
struct plus_mod3
{
  T* table;

  plus_mod3(T* table)
      : table(table)
  {}

  THRUST_HOST_DEVICE T operator()(T a, T b)
  {
    return table[(int) (a + b)];
  }
};

TYPED_TEST(ReduceTests, TestReduceWithIndirection)
{
  // add numbers modulo 3 with external lookup table
  using Vector = typename TestFixture::input_type;
  using T      = typename Vector::value_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  Vector data{0, 1, 2, 1, 2, 0, 1};

  Vector table{0, 1, 2, 0, 1, 2};

  T result = thrust::reduce(data.begin(), data.end(), T(0), plus_mod3<T>(thrust::raw_pointer_cast(&table[0])));

  ASSERT_EQ(result, T(1));
}

TYPED_TEST(ReducePrimitiveTests, TestReduceCountingIterator)
{
  using T = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  size_t const n = 15 * sizeof(T);

  ASSERT_LE(T(n), truncate_to_max_representable<T>(n));

  thrust::counting_iterator<T, thrust::host_system_tag> h_first   = thrust::make_counting_iterator<T>(0);
  thrust::counting_iterator<T, thrust::device_system_tag> d_first = thrust::make_counting_iterator<T>(0);

  T init = random_integer<T>();

  T h_result = thrust::reduce(h_first, h_first + n, init);
  T d_result = thrust::reduce(d_first, d_first + n, init);

  // we use ASSERT_NEAR because we're testing floating point types
  if (std::is_floating_point<T>::value)
  {
    ASSERT_NEAR(h_result, d_result, h_result * 0.01);
  }
  else
  {
    ASSERT_EQ(h_result, d_result);
  }
}

void TestReduceWithBigIndexesHelper(int magnitude)
{
  thrust::constant_iterator<long long> begin(1);
  thrust::constant_iterator<long long> end = begin + (1ll << magnitude);
  ASSERT_EQ(thrust::distance(begin, end), 1ll << magnitude);

  long long result = thrust::reduce(thrust::device, begin, end);

  ASSERT_EQ(result, 1ll << magnitude);
}

TEST(ReduceTests, TestReduceWithBigIndexes)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  TestReduceWithBigIndexesHelper(30);
#ifndef THRUST_FORCE_32_BIT_OFFSET_TYPE
  TestReduceWithBigIndexesHelper(31);
  TestReduceWithBigIndexesHelper(32);
  TestReduceWithBigIndexesHelper(33);
#endif
}

__global__ THRUST_HIP_LAUNCH_BOUNDS_DEFAULT void ReduceKernel(int const N, int* in_array, int* result)
{
  if (threadIdx.x == 0)
  {
    thrust::device_ptr<int> in_begin(in_array);
    thrust::device_ptr<int> in_end(in_array + N);

    result[0] = thrust::reduce(thrust::hip::par, in_begin, in_end);
  }
}

TEST(ReduceTests, TestReduceDevice)
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
      thrust::device_vector<int> d_output(1, 0);

      int h_output = thrust::reduce(h_data.begin(), h_data.end());

      hipLaunchKernelGGL(
        ReduceKernel,
        dim3(1, 1, 1),
        dim3(128, 1, 1),
        0,
        0,
        size,
        thrust::raw_pointer_cast(&d_data[0]),
        thrust::raw_pointer_cast(&d_output[0]));

      ASSERT_EQ(h_output, d_output[0]);
    }
  }
}
