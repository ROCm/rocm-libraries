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

#include <thrust/detail/config.h>

#include <thrust/device_free.h>
#include <thrust/device_malloc.h>
#include <thrust/functional.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/retag.h>
#include <thrust/scan.h>

#include "test_param_fixtures.hpp"
#include "test_real_assertions.hpp"
#include "test_utils.hpp"

#include _THRUST_STD_INCLUDE(array)

TESTS_DEFINE(ScanTests, FullTestsParams);

TESTS_DEFINE(ScanVariablesTests, NumericalTestsParams);

TESTS_DEFINE(ScanVectorTests, VectorSignedIntegerTestsParams);

using MixedParams = ::testing::Types<Params<std::tuple<thrust::host_vector<int>, thrust::host_vector<float>>>,
                                     Params<std::tuple<thrust::device_vector<int>, thrust::device_vector<float>>>>;

TESTS_DEFINE(ScanMixedTests, MixedParams);

template <typename T>
struct max_functor
{
  THRUST_HOST_DEVICE T operator()(T rhs, T lhs) const
  {
    return thrust::max(rhs, lhs);
  }
};

TYPED_TEST(ScanVectorTests, TestScanSimple)
{
  using Vector = typename TestFixture::input_type;
  using T      = typename Vector::value_type;

  typename Vector::iterator iter;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  Vector input(5);
  Vector result(5);
  Vector output(5);

  input = {1, 3, -2, 4, -5};
  Vector input_copy(input);

  // inclusive scan
  iter   = thrust::inclusive_scan(input.begin(), input.end(), output.begin());
  result = {1, 4, 2, 6, 1};
  ASSERT_EQ(std::size_t(iter - output.begin()), input.size());
  ASSERT_EQ(input, input_copy);
  ASSERT_EQ(output, result);

  // exclusive scan
  iter   = thrust::exclusive_scan(input.begin(), input.end(), output.begin(), T(0));
  result = {0, 1, 4, 2, 6};
  ASSERT_EQ(std::size_t(iter - output.begin()), input.size());
  ASSERT_EQ(input, input_copy);
  ASSERT_EQ(output, result);

  // exclusive scan with init
  iter   = thrust::exclusive_scan(input.begin(), input.end(), output.begin(), T(3));
  result = {3, 4, 7, 5, 9};
  ASSERT_EQ(std::size_t(iter - output.begin()), input.size());
  ASSERT_EQ(input, input_copy);
  ASSERT_EQ(output, result);

  // inclusive scan with op
  iter   = thrust::inclusive_scan(input.begin(), input.end(), output.begin(), thrust::plus<T>());
  result = {1, 4, 2, 6, 1};
  ASSERT_EQ(std::size_t(iter - output.begin()), input.size());
  ASSERT_EQ(input, input_copy);
  ASSERT_EQ(output, result);

  // inclusive scan with init and op
  iter   = thrust::inclusive_scan(input.begin(), input.end(), output.begin(), T(-1), thrust::multiplies<T>());
  result = {-1, -3, 6, 24, -120};
  ASSERT_EQ(std::size_t(iter - output.begin()), input.size());
  ASSERT_EQ(input, input_copy);
  ASSERT_EQ(output, result);

  // exclusive scan with init and op
  iter   = thrust::exclusive_scan(input.begin(), input.end(), output.begin(), T(3), thrust::plus<T>());
  result = {3, 4, 7, 5, 9};
  ASSERT_EQ(std::size_t(iter - output.begin()), input.size());
  ASSERT_EQ(input, input_copy);
  ASSERT_EQ(output, result);

  // inplace inclusive scan
  input  = input_copy;
  iter   = thrust::inclusive_scan(input.begin(), input.end(), input.begin());
  result = {1, 4, 2, 6, 1};
  ASSERT_EQ(std::size_t(iter - input.begin()), input.size());
  ASSERT_EQ(input, result);

  // inplace inclusive scan with init and op
  input  = input_copy;
  iter   = thrust::inclusive_scan(input.begin(), input.end(), input.begin(), T(3), thrust::plus<T>());
  result = {4, 7, 5, 9, 4};
  ASSERT_EQ(std::size_t(iter - input.begin()), input.size());
  ASSERT_EQ(input, result);

  // inplace exclusive scan with init
  input  = input_copy;
  iter   = thrust::exclusive_scan(input.begin(), input.end(), input.begin(), T(3));
  result = {3, 4, 7, 5, 9};
  ASSERT_EQ(std::size_t(iter - input.begin()), input.size());
  ASSERT_EQ(input, result);

  // inplace exclusive scan with implicit init=0
  input  = input_copy;
  iter   = thrust::exclusive_scan(input.begin(), input.end(), input.begin());
  result = {0, 1, 4, 2, 6};
  ASSERT_EQ(std::size_t(iter - input.begin()), input.size());
  ASSERT_EQ(input, result);
}

template <typename InputIterator, typename OutputIterator>
OutputIterator inclusive_scan(my_system& system, InputIterator, InputIterator, OutputIterator result)
{
  system.validate_dispatch();
  return result;
}

TEST(ScanTests, TestInclusiveScanDispatchExplicit)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::device_vector<int> vec(1);

  my_system sys(0);
  thrust::inclusive_scan(sys, vec.begin(), vec.begin(), vec.begin());

  ASSERT_EQ(true, sys.is_valid());
}

template <typename InputIterator, typename OutputIterator>
OutputIterator inclusive_scan(my_tag, InputIterator, InputIterator, OutputIterator result)
{
  *result = 13;
  return result;
}

TEST(ScanTests, TestInclusiveScanDispatchImplicit)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::device_vector<int> vec(1);

  thrust::inclusive_scan(
    thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.begin()));

  ASSERT_EQ(13, vec.front());
}

template <typename InputIterator, typename OutputIterator>
OutputIterator exclusive_scan(my_system& system, InputIterator, InputIterator, OutputIterator result)
{
  system.validate_dispatch();
  return result;
}

TEST(ScanTests, TestExclusiveScanDispatchExplicit)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::device_vector<int> vec(1);

  my_system sys(0);
  thrust::exclusive_scan(sys, vec.begin(), vec.begin(), vec.begin());

  ASSERT_EQ(true, sys.is_valid());
}

template <typename InputIterator, typename OutputIterator>
OutputIterator exclusive_scan(my_tag, InputIterator, InputIterator, OutputIterator result)
{
  *result = 13;
  return result;
}

TEST(ScanTests, TestExclusiveScanDispatchImplicit)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::device_vector<int> vec(1);

  thrust::exclusive_scan(
    thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.begin()));

  ASSERT_EQ(13, vec.front());
}

TEST(ScanTests, TestInclusiveScan32)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  using T  = int;
  size_t n = 32;

  for (auto seed : get_seeds())
  {
    SCOPED_TRACE(testing::Message() << "with seed= " << seed);

    thrust::host_vector<T> h_input =
      get_random_data<T>(n, get_default_limits<T>::min(), get_default_limits<T>::max(), seed);
    thrust::device_vector<T> d_input = h_input;

    thrust::host_vector<T> h_output(n);
    thrust::device_vector<T> d_output(n);

    thrust::inclusive_scan(h_input.begin(), h_input.end(), h_output.begin());
    thrust::inclusive_scan(d_input.begin(), d_input.end(), d_output.begin());

    ASSERT_EQ(d_output, h_output);
  }
}

TEST(ScanTests, TestExclusiveScan32)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  using T  = int;
  size_t n = 32;
  T init   = 13;

  for (auto seed : get_seeds())
  {
    SCOPED_TRACE(testing::Message() << "with seed= " << seed);

    thrust::host_vector<T> h_input =
      get_random_data<T>(n, get_default_limits<T>::min(), get_default_limits<T>::max(), seed);
    thrust::device_vector<T> d_input = h_input;

    thrust::host_vector<T> h_output(n);
    thrust::device_vector<T> d_output(n);

    thrust::exclusive_scan(h_input.begin(), h_input.end(), h_output.begin(), init);
    thrust::exclusive_scan(d_input.begin(), d_input.end(), d_output.begin(), init);

    ASSERT_EQ(d_output, h_output);
  }
}

template <class IntVector, class FloatVector>
void TestScanMixedTypes()
{
  // make sure we get types for default args and operators correct
  IntVector int_input{1, 2, 3, 4};
  FloatVector float_input{1.5, 2.5, 3.5, 4.5};
  IntVector int_output(4);
  FloatVector float_output(4);

  // float -> int should use plus<void> operator and float accumulator by default
  thrust::inclusive_scan(float_input.begin(), float_input.end(), int_output.begin());
  ASSERT_EQ(int_output[0], 1); // in: 1.5 accum: 1.5f out: 1
  ASSERT_EQ(int_output[1], 4); // in: 2.5 accum: 4.0f out: 4
  ASSERT_EQ(int_output[2], 7); // in: 3.5 accum: 7.5f out: 7
  ASSERT_EQ(int_output[3], 12); // in: 4.5 accum: 12.f out: 12

  // float -> float with plus<int> operator (float accumulator)
  thrust::inclusive_scan(float_input.begin(), float_input.end(), float_output.begin(), thrust::plus<int>());
  ASSERT_EQ(float_output[0], 1.5f); // in: 1.5 accum: 1.5f out: 1.5f
  ASSERT_EQ(float_output[1], 3.0f); // in: 2.5 accum: 3.0f out: 3.0f
  ASSERT_EQ(float_output[2], 6.0f); // in: 3.5 accum: 6.0f out: 6.0f
  ASSERT_EQ(float_output[3], 10.0f); // in: 4.5 accum: 10.f out: 10.f

  // float -> int should use plus<void> operator and float accumulator by default
  thrust::exclusive_scan(float_input.begin(), float_input.end(), int_output.begin());
  ASSERT_EQ(int_output[0], 0); // out: 0.0f  in: 1.5 accum: 1.5f
  ASSERT_EQ(int_output[1], 1); // out: 1.5f  in: 2.5 accum: 4.0f
  ASSERT_EQ(int_output[2], 4); // out: 4.0f  in: 3.5 accum: 7.5f
  ASSERT_EQ(int_output[3], 7); // out: 7.5f  in: 4.5 accum: 12.f

  // float -> int should use plus<> operator and float accumulator by default
  thrust::exclusive_scan(float_input.begin(), float_input.end(), int_output.begin(), (float) 5.5);
  ASSERT_EQ(int_output[0], 5); // out: 5.5f  in: 1.5 accum: 7.0f
  ASSERT_EQ(int_output[1], 7); // out: 7.0f  in: 2.5 accum: 9.5f
  ASSERT_EQ(int_output[2], 9); // out: 9.5f  in: 3.5 accum: 13.0f
  ASSERT_EQ(int_output[3], 13); // out: 13.f  in: 4.5 accum: 17.4f

  // int -> float should use using plus<> operator and int accumulator by default
  thrust::inclusive_scan(int_input.begin(), int_input.end(), float_output.begin());
  ASSERT_EQ(float_output[0], 1.f); // in: 1 accum: 1  out: 1
  ASSERT_EQ(float_output[1], 3.f); // in: 2 accum: 3  out: 3
  ASSERT_EQ(float_output[2], 6.f); // in: 3 accum: 6  out: 6
  ASSERT_EQ(float_output[3], 10.f); // in: 4 accum: 10 out: 10

  // int -> float + float init_value should use using plus<> operator and
  // float accumulator by default
  thrust::exclusive_scan(int_input.begin(), int_input.end(), float_output.begin(), (float) 5.5);
  ASSERT_EQ(float_output[0], 5.5f); // out: 5.5f  in: 1 accum: 6.5f
  ASSERT_EQ(float_output[1], 6.5f); // out: 6.0f  in: 2 accum: 8.5f
  ASSERT_EQ(float_output[2], 8.5f); // out: 8.0f  in: 3 accum: 11.5f
  ASSERT_EQ(float_output[3], 11.5f); // out: 11.f  in: 4 accum: 15.5f
}
TEST(ScanTests, TestScanMixedTypesHost)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  TestScanMixedTypes<thrust::host_vector<int>, thrust::host_vector<float>>();
}
TEST(ScanTests, TestScanMixedTypesDevice)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  TestScanMixedTypes<thrust::device_vector<int>, thrust::device_vector<float>>();
}

TYPED_TEST(ScanVariablesTests, TestScanWithOperator)
{
  using T = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  for (auto size : get_sizes())
  {
    SCOPED_TRACE(testing::Message() << "with size= " << size);

    for (auto seed : get_seeds())
    {
      SCOPED_TRACE(testing::Message() << "with seed= " << seed);

      thrust::host_vector<T> h_input =
        get_random_data<T>(size, get_default_limits<T>::min(), get_default_limits<T>::max(), seed);
      thrust::device_vector<T> d_input = h_input;

      thrust::host_vector<T> h_output(size);
      thrust::device_vector<T> d_output(size);

      thrust::inclusive_scan(h_input.begin(), h_input.end(), h_output.begin(), max_functor<T>());
      thrust::inclusive_scan(d_input.begin(), d_input.end(), d_output.begin(), max_functor<T>());
      ASSERT_EQ(d_output, h_output);

      thrust::exclusive_scan(h_input.begin(), h_input.end(), h_output.begin(), T(13), max_functor<T>());
      thrust::exclusive_scan(d_input.begin(), d_input.end(), d_output.begin(), T(13), max_functor<T>());
      ASSERT_EQ(d_output, h_output);
    }
  }
}

TYPED_TEST(ScanVariablesTests, TestScanWithOperatorToDiscardIterator)
{
  using T = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  for (auto size : get_sizes())
  {
    SCOPED_TRACE(testing::Message() << "with size= " << size);

    for (auto seed : get_seeds())
    {
      SCOPED_TRACE(testing::Message() << "with seed= " << seed);

      thrust::host_vector<T> h_input =
        get_random_data<T>(size, get_default_limits<T>::min(), get_default_limits<T>::max(), seed);
      thrust::device_vector<T> d_input = h_input;

      thrust::discard_iterator<> reference(size);

      thrust::discard_iterator<> h_result =
        thrust::inclusive_scan(h_input.begin(), h_input.end(), thrust::make_discard_iterator(), max_functor<T>());

      thrust::discard_iterator<> d_result =
        thrust::inclusive_scan(d_input.begin(), d_input.end(), thrust::make_discard_iterator(), max_functor<T>());

      ASSERT_EQ_QUIET(reference, h_result);
      ASSERT_EQ_QUIET(reference, d_result);

      h_result = thrust::exclusive_scan(
        h_input.begin(), h_input.end(), thrust::make_discard_iterator(), T(13), max_functor<T>());

      d_result = thrust::exclusive_scan(
        d_input.begin(), d_input.end(), thrust::make_discard_iterator(), T(13), max_functor<T>());

      ASSERT_EQ_QUIET(reference, h_result);
      ASSERT_EQ_QUIET(reference, d_result);
    }
  }
}

TYPED_TEST(ScanVariablesTests, TestScan)
{
  using T = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  for (auto size : get_sizes())
  {
    SCOPED_TRACE(testing::Message() << "with size= " << size);

    for (auto seed : get_seeds())
    {
      SCOPED_TRACE(testing::Message() << "with seed= " << seed);

      thrust::host_vector<T> h_input =
        get_random_data<T>(size, get_default_limits<T>::min(), get_default_limits<T>::max(), seed);
      thrust::device_vector<T> d_input = h_input;

      thrust::host_vector<T> h_output(size);
      thrust::device_vector<T> d_output(size);

      thrust::inclusive_scan(h_input.begin(), h_input.end(), h_output.begin());
      thrust::inclusive_scan(d_input.begin(), d_input.end(), d_output.begin());
      test_equality_scan(h_output, d_output);

      thrust::exclusive_scan(h_input.begin(), h_input.end(), h_output.begin());
      thrust::exclusive_scan(d_input.begin(), d_input.end(), d_output.begin());
      test_equality_scan(h_output, d_output);

      thrust::exclusive_scan(h_input.begin(), h_input.end(), h_output.begin(), (T) 11);
      thrust::exclusive_scan(d_input.begin(), d_input.end(), d_output.begin(), (T) 11);
      test_equality_scan(h_output, d_output);

      // in-place scans
      h_output = h_input;
      d_output = d_input;
      thrust::inclusive_scan(h_output.begin(), h_output.end(), h_output.begin());
      thrust::inclusive_scan(d_output.begin(), d_output.end(), d_output.begin());
      test_equality_scan(h_output, d_output);

      h_output = h_input;
      d_output = d_input;
      thrust::exclusive_scan(h_output.begin(), h_output.end(), h_output.begin());
      thrust::exclusive_scan(d_output.begin(), d_output.end(), d_output.begin());
      test_equality_scan(h_output, d_output);
    }
  }
}

TYPED_TEST(ScanVariablesTests, TestScanToDiscardIterator)
{
  using T = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  for (auto size : get_sizes())
  {
    SCOPED_TRACE(testing::Message() << "with size= " << size);

    for (auto seed : get_seeds())
    {
      SCOPED_TRACE(testing::Message() << "with seed= " << seed);

      thrust::host_vector<T> h_input =
        get_random_data<T>(size, get_default_limits<T>::min(), get_default_limits<T>::max(), seed);
      thrust::device_vector<T> d_input = h_input;

      thrust::discard_iterator<> h_result =
        thrust::inclusive_scan(h_input.begin(), h_input.end(), thrust::make_discard_iterator());

      thrust::discard_iterator<> d_result =
        thrust::inclusive_scan(d_input.begin(), d_input.end(), thrust::make_discard_iterator());

      thrust::discard_iterator<> reference(size);

      ASSERT_EQ_QUIET(reference, h_result);
      ASSERT_EQ_QUIET(reference, d_result);

      h_result = thrust::exclusive_scan(h_input.begin(), h_input.end(), thrust::make_discard_iterator(), (T) 11);

      d_result = thrust::exclusive_scan(d_input.begin(), d_input.end(), thrust::make_discard_iterator(), (T) 11);

      ASSERT_EQ_QUIET(reference, h_result);
      ASSERT_EQ_QUIET(reference, d_result);
    }
  }
}

TEST(ScanTests, TestScanMixedTypes)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  const unsigned int n = 113;

  for (auto seed : get_seeds())
  {
    SCOPED_TRACE(testing::Message() << "with seed= " << seed);

    thrust::host_vector<unsigned int> h_input = get_random_data<unsigned int>(
      n, get_default_limits<unsigned int>::min(), get_default_limits<unsigned int>::max(), seed);
    for (size_t i = 0; i < n; i++)
    {
      h_input[i] %= 10;
    }
    thrust::device_vector<unsigned int> d_input = h_input;

    thrust::host_vector<float> h_float_output(n);
    thrust::device_vector<float> d_float_output(n);
    thrust::host_vector<int> h_int_output(n);
    thrust::device_vector<int> d_int_output(n);

    // mixed input/output types
    thrust::inclusive_scan(h_input.begin(), h_input.end(), h_float_output.begin());
    thrust::inclusive_scan(d_input.begin(), d_input.end(), d_float_output.begin());
    ASSERT_EQ(d_float_output, h_float_output);

    thrust::exclusive_scan(h_input.begin(), h_input.end(), h_float_output.begin(), (float) 3.5);
    thrust::exclusive_scan(d_input.begin(), d_input.end(), d_float_output.begin(), (float) 3.5);
    ASSERT_EQ(d_float_output, h_float_output);

    thrust::exclusive_scan(h_input.begin(), h_input.end(), h_float_output.begin(), (int) 3);
    thrust::exclusive_scan(d_input.begin(), d_input.end(), d_float_output.begin(), (int) 3);
    ASSERT_EQ(d_float_output, h_float_output);

    thrust::exclusive_scan(h_input.begin(), h_input.end(), h_int_output.begin(), (int) 3);
    thrust::exclusive_scan(d_input.begin(), d_input.end(), d_int_output.begin(), (int) 3);
    ASSERT_EQ(d_int_output, h_int_output);

    thrust::exclusive_scan(h_input.begin(), h_input.end(), h_int_output.begin(), (float) 3.5);
    thrust::exclusive_scan(d_input.begin(), d_input.end(), d_int_output.begin(), (float) 3.5);
    ASSERT_EQ(d_int_output, h_int_output);
  }
}

template <typename T, unsigned int N>
void _TestScanWithLargeTypes()
{
  size_t n = (1024 * 1024) / sizeof(FixedVector<T, N>);

  thrust::host_vector<FixedVector<T, N>> h_input(n);
  thrust::host_vector<FixedVector<T, N>> h_output(n);

  for (size_t i = 0; i < h_input.size(); i++)
  {
    h_input[i] = FixedVector<T, N>(static_cast<T>(i));
  }

  thrust::device_vector<FixedVector<T, N>> d_input = h_input;
  thrust::device_vector<FixedVector<T, N>> d_output(n);

  thrust::inclusive_scan(h_input.begin(), h_input.end(), h_output.begin());
  thrust::inclusive_scan(d_input.begin(), d_input.end(), d_output.begin());

  ASSERT_EQ_QUIET(h_output, d_output);

  thrust::exclusive_scan(h_input.begin(), h_input.end(), h_output.begin(), FixedVector<T, N>(0));
  thrust::exclusive_scan(d_input.begin(), d_input.end(), d_output.begin(), FixedVector<T, N>(0));

  ASSERT_EQ_QUIET(h_output, d_output);
}

TEST(ScanTests, TestScanWithLargeTypes)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  _TestScanWithLargeTypes<int, 1>();

#if !defined(__QNX__)
  _TestScanWithLargeTypes<int, 8>();
  _TestScanWithLargeTypes<int, 64>();
#else
//  KNOWN_FAILURE;
#endif
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

TYPED_TEST(ScanVectorTests, TestInclusiveScanWithIndirection)
{
  // add numbers modulo 3 with external lookup table
  using Vector = typename TestFixture::input_type;
  using T      = typename Vector::value_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  Vector data{0, 1, 2, 1, 2, 0, 1};
  Vector table{0, 1, 2, 0, 1, 2};
  thrust::inclusive_scan(data.begin(), data.end(), data.begin(), plus_mod3<T>(thrust::raw_pointer_cast(&table[0])));

  ASSERT_EQ(data, (Vector{0, 1, 0, 1, 0, 0, 1}));
}

__global__ THRUST_HIP_LAUNCH_BOUNDS_DEFAULT void InclusiveScanKernel(int const N, int* in_array, int* out_array)
{
  if (threadIdx.x == 0)
  {
    thrust::device_ptr<int> in_begin(in_array);
    thrust::device_ptr<int> in_end(in_array + N);
    thrust::device_ptr<int> out_begin(out_array);

    thrust::inclusive_scan(thrust::hip::par, in_begin, in_end, out_begin);
  }
}

TEST(ScanTests, TestInclusiveScanDevice)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());
  for (auto size : get_sizes())
  {
    SCOPED_TRACE(testing::Message() << "with size= " << size);

    for (auto seed : get_seeds())
    {
      SCOPED_TRACE(testing::Message() << "with seed= " << seed);

      thrust::host_vector<int> h_data = get_random_data<int>(size, 0, size, seed);
      thrust::host_vector<int> h_output(size);

      thrust::inclusive_scan(h_data.begin(), h_data.end(), h_output.begin());
      thrust::device_vector<int> d_data = h_data;
      thrust::device_vector<int> d_output(size);
      hipLaunchKernelGGL(
        InclusiveScanKernel,
        dim3(1, 1, 1),
        dim3(128, 1, 1),
        0,
        0,
        size,
        thrust::raw_pointer_cast(&d_data[0]),
        thrust::raw_pointer_cast(&d_output[0]));

      ASSERT_EQ(h_output, d_output);
    }
  }
}

__global__ THRUST_HIP_LAUNCH_BOUNDS_DEFAULT void ExclusiveScanKernel(int const N, int* in_array, int* out_array)
{
  if (threadIdx.x == 0)
  {
    thrust::device_ptr<int> in_begin(in_array);
    thrust::device_ptr<int> in_end(in_array + N);
    thrust::device_ptr<int> out_begin(out_array);

    thrust::exclusive_scan(thrust::hip::par, in_begin, in_end, out_begin);
  }
}

TEST(ScanTests, TestExclusiveScanDevice)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());
  for (auto size : get_sizes())
  {
    SCOPED_TRACE(testing::Message() << "with size= " << size);

    for (auto seed : get_seeds())
    {
      SCOPED_TRACE(testing::Message() << "with seed= " << seed);

      thrust::host_vector<int> h_data = get_random_data<int>(size, 0, size, seed);
      thrust::host_vector<int> h_output(size);

      thrust::exclusive_scan(h_data.begin(), h_data.end(), h_output.begin());
      thrust::device_vector<int> d_data = h_data;
      thrust::device_vector<int> d_output(size);
      hipLaunchKernelGGL(
        ExclusiveScanKernel,
        dim3(1, 1, 1),
        dim3(128, 1, 1),
        0,
        0,
        size,
        thrust::raw_pointer_cast(&d_data[0]),
        thrust::raw_pointer_cast(&d_output[0]));

      ASSERT_EQ(h_output, d_output);
    }
  }
}

template <typename T>
struct const_ref_plus_mod3
{
  T* table;

  const_ref_plus_mod3(T* table)
      : table(table)
  {}

  THRUST_HOST_DEVICE const T& operator()(T a, T b)
  {
    return table[(int) (a + b)];
  }
};

TYPED_TEST(ScanTests, TestInclusiveScanWithConstAccumulator)
{
  // add numbers modulo 3 with external lookup table
  using Vector = typename TestFixture::input_type;
  using T      = typename Vector::value_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  Vector data{0, 1, 2, 1, 2, 0, 1};
  Vector table{0, 1, 2, 0, 1, 2};
  thrust::inclusive_scan(
    data.begin(), data.end(), data.begin(), const_ref_plus_mod3<T>(thrust::raw_pointer_cast(&table[0])));

  ASSERT_EQ(data, (Vector{0, 1, 0, 1, 0, 0, 1}));
}

struct only_set_when_expected_it
{
  long long expected;
  bool* flag;

  THRUST_HOST_DEVICE only_set_when_expected_it operator++() const
  {
    return *this;
  }
  THRUST_HOST_DEVICE only_set_when_expected_it operator*() const
  {
    return *this;
  }
  template <typename Difference>
  THRUST_HOST_DEVICE only_set_when_expected_it operator+(Difference) const
  {
    return *this;
  }
  template <typename Index>
  THRUST_HOST_DEVICE only_set_when_expected_it operator[](Index) const
  {
    return *this;
  }

  THRUST_DEVICE void operator=(long long value) const
  {
    if (value == expected)
    {
      *flag = true;
    }
  }
};

THRUST_NAMESPACE_BEGIN
template <>
struct iterator_traits<only_set_when_expected_it>
{
  using value_type = long long;
  using reference  = only_set_when_expected_it;
};
THRUST_NAMESPACE_END

namespace std
{
template <>
struct iterator_traits<only_set_when_expected_it>
{
  using value_type = long long;
  using reference  = only_set_when_expected_it;
};
} // namespace std

void TestInclusiveScanWithBigIndexesHelper(int magnitude)
{
  thrust::constant_iterator<long long> begin(1);
  thrust::constant_iterator<long long> end = begin + (1ll << magnitude);
  ASSERT_EQ(thrust::distance(begin, end), 1ll << magnitude);

  thrust::device_ptr<bool> has_executed = thrust::device_malloc<bool>(1);
  *has_executed                         = false;

  only_set_when_expected_it out = {(1ll << magnitude), thrust::raw_pointer_cast(has_executed)};

  thrust::inclusive_scan(thrust::device, begin, end, out);

  bool has_executed_h = *has_executed;
  thrust::device_free(has_executed);

  ASSERT_EQ(has_executed_h, true);
}

TEST(ScanTests, TestInclusiveScanWithBigIndexes)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  TestInclusiveScanWithBigIndexesHelper(30);
  TestInclusiveScanWithBigIndexesHelper(31);
#ifndef THRUST_FORCE_32_BIT_OFFSET_TYPE
  TestInclusiveScanWithBigIndexesHelper(32);
  TestInclusiveScanWithBigIndexesHelper(33);
#endif
}

void TestExclusiveScanWithBigIndexesHelper(int magnitude)
{
  thrust::constant_iterator<long long> begin(1);
  thrust::constant_iterator<long long> end = begin + (1ll << magnitude);
  ASSERT_EQ(thrust::distance(begin, end), 1ll << magnitude);

  thrust::device_ptr<bool> has_executed = thrust::device_malloc<bool>(1);
  *has_executed                         = false;

  only_set_when_expected_it out = {(1ll << magnitude) - 1, thrust::raw_pointer_cast(has_executed)};

  thrust::exclusive_scan(thrust::device, begin, end, out, 0ll);

  bool has_executed_h = *has_executed;
  thrust::device_free(has_executed);

  ASSERT_EQ(has_executed_h, true);
}

TEST(ScanTests, TestExclusiveScanWithBigIndexes)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  TestExclusiveScanWithBigIndexesHelper(30);
  TestExclusiveScanWithBigIndexesHelper(31);
#ifndef THRUST_FORCE_32_BIT_OFFSET_TYPE
  TestExclusiveScanWithBigIndexesHelper(32);
  TestExclusiveScanWithBigIndexesHelper(33);
#endif
}

struct Int
{
  int i{};
  THRUST_HOST_DEVICE explicit Int(int num)
      : i(num)
  {}
  THRUST_HOST_DEVICE Int()
      : i{}
  {}
  THRUST_HOST_DEVICE Int operator+(Int const& o) const
  {
    return Int{this->i + o.i};
  }
};

TEST(ScanTests, TestInclusiveScanWithUserDefinedType)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::device_vector<Int> vec(5, Int{1});

  thrust::inclusive_scan(thrust::device, vec.cbegin(), vec.cend(), vec.begin());

  ASSERT_EQ(static_cast<Int>(vec.back()).i, 5);
}

// Represents a permutation as a tuple of integers, see also: https://en.wikipedia.org/wiki/Permutation
// We need a distinct type (instead of an alias) for operator<< to be found via ADL
struct permutation_t : _THRUST_STD::array<int, 5>
{
  permutation_t() = default;

  constexpr THRUST_HOST_DEVICE permutation_t(int a, int b, int c, int d, int e)
      : _THRUST_STD::array<int, 5>{a, b, c, d, e}
  {}

  friend std::ostream& operator<<(std::ostream& os, const permutation_t& p)
  {
    os << '{';
    for (std::size_t i = 0; i < p.size(); i++)
    {
      if (i > 0)
      {
        os << ", ";
      }
      os << p[i];
    }
    return os << '}';
  }

  friend THRUST_HOST_DEVICE bool operator==(const permutation_t& lhs, const permutation_t& rhs)
  {
    auto lhs_ptr = lhs.data();
    auto rhs_ptr = rhs.data();
    for (size_t i = 0; i < lhs.size(); ++i)
    {
      if (lhs_ptr[i] != rhs_ptr[i])
      {
        return false;
      }
    }
    return true;
  }
};

// Composes two permutations. This operation is associative, but not commutative.
struct composition_op_t
{
  THRUST_HOST_DEVICE permutation_t operator()(permutation_t lhs, permutation_t rhs) const
  {
    permutation_t result;
    // Get raw pointers to the underlying data to avoid operator[] which
    // results in debug-assert calls (and __glibcxx_assert_fail) on device.
    auto lhs_ptr    = lhs.data();
    auto rhs_ptr    = rhs.data();
    auto result_ptr = result.data();
    for (std::size_t i = 0; i < lhs.size(); i++)
    {
      result_ptr[i] = rhs_ptr[lhs_ptr[i]];
    }
    return result;
  }
};

TEST(ScanTests, TestInclusiveScanWithNonCommutativeOp)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  const thrust::device_vector<permutation_t> input = {
    {3, 2, 0, 1, 4},
    {2, 4, 0, 1, 3},
    {3, 2, 1, 4, 0},
    {4, 3, 1, 0, 2},
    {0, 3, 2, 4, 1},
    {3, 2, 1, 0, 4},
    {3, 4, 1, 2, 0},
    {4, 2, 1, 0, 3},
    {4, 0, 1, 3, 2},
    {0, 2, 3, 1, 4}};
  thrust::device_vector<permutation_t> output(10);
  constexpr auto identity = permutation_t{0, 1, 2, 3, 4};

  thrust::inclusive_scan(input.begin(), input.end(), output.begin(), composition_op_t{});
  ASSERT_EQ(
    output,
    (thrust::device_vector<permutation_t>{
      {3, 2, 0, 1, 4},
      {1, 0, 2, 4, 3},
      {2, 3, 1, 0, 4},
      {1, 0, 3, 4, 2},
      {3, 0, 4, 1, 2},
      {0, 3, 4, 2, 1},
      {3, 2, 0, 1, 4},
      {0, 1, 4, 2, 3},
      {4, 0, 2, 1, 3},
      {4, 0, 3, 2, 1}}));

  thrust::exclusive_scan(input.begin(), input.end(), output.begin(), identity, composition_op_t{});
  ASSERT_EQ(
    output,
    (thrust::device_vector<permutation_t>{
      {0, 1, 2, 3, 4},
      {3, 2, 0, 1, 4},
      {1, 0, 2, 4, 3},
      {2, 3, 1, 0, 4},
      {1, 0, 3, 4, 2},
      {3, 0, 4, 1, 2},
      {0, 3, 4, 2, 1},
      {3, 2, 0, 1, 4},
      {0, 1, 4, 2, 3},
      {4, 0, 2, 1, 3}}));
}
