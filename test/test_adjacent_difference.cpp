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

#include <thrust/adjacent_difference.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/retag.h>

#include "test_header.hpp"

#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HIP

TESTS_DEFINE(AdjacentDifferenceTests, FullTestsParams);
TESTS_DEFINE(AdjacentDifferenceVariableTests, NumericalTestsParams);

TYPED_TEST(AdjacentDifferenceTests, TestAdjacentDifferenceSimple)
{
  using Vector = typename TestFixture::input_type;
  using Policy = typename TestFixture::execution_policy;
  using T      = typename Vector::value_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  Vector input(3);
  Vector output(3);
  input[0] = 1;
  input[1] = 4;
  input[2] = 6;

  typename Vector::iterator result;

  result = thrust::adjacent_difference(Policy{}, input.begin(), input.end(), output.begin());

  ASSERT_EQ(result - output.begin(), 3);
  ASSERT_EQ(output[0], T(1));
  ASSERT_EQ(output[1], T(3));
  ASSERT_EQ(output[2], T(2));

  result = thrust::adjacent_difference(Policy{}, input.begin(), input.end(), output.begin(), thrust::plus<T>());

  ASSERT_EQ(result - output.begin(), 3);
  ASSERT_EQ(output[0], T(1));
  ASSERT_EQ(output[1], T(5));
  ASSERT_EQ(output[2], T(10));

  // test in-place operation, result and first are permitted to be the same
  result = thrust::adjacent_difference(Policy{}, input.begin(), input.end(), input.begin());

  ASSERT_EQ(result - input.begin(), 3);
  ASSERT_EQ(input[0], T(1));
  ASSERT_EQ(input[1], T(3));
  ASSERT_EQ(input[2], T(2));
}

TYPED_TEST(AdjacentDifferenceVariableTests, TestAdjacentDifference)
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

      typename thrust::host_vector<T>::iterator h_result;
      typename thrust::device_vector<T>::iterator d_result;

      h_result = thrust::adjacent_difference(h_input.begin(), h_input.end(), h_output.begin());
      d_result = thrust::adjacent_difference(d_input.begin(), d_input.end(), d_output.begin());

      ASSERT_EQ(h_result - h_output.begin(), size);
      ASSERT_EQ(d_result - d_output.begin(), size);
      ASSERT_EQ_QUIET(h_output, d_output);

      h_result = thrust::adjacent_difference(h_input.begin(), h_input.end(), h_output.begin(), thrust::plus<T>());
      d_result = thrust::adjacent_difference(d_input.begin(), d_input.end(), d_output.begin(), thrust::plus<T>());

      ASSERT_EQ(h_result - h_output.begin(), size);
      ASSERT_EQ(d_result - d_output.begin(), size);
      ASSERT_EQ_QUIET(h_output, d_output);

      // in-place operation
      h_result = thrust::adjacent_difference(h_input.begin(), h_input.end(), h_input.begin(), thrust::plus<T>());
      d_result = thrust::adjacent_difference(d_input.begin(), d_input.end(), d_input.begin(), thrust::plus<T>());

      ASSERT_EQ(h_result - h_input.begin(), size);
      ASSERT_EQ(d_result - d_input.begin(), size);
      ASSERT_EQ_QUIET(h_input, h_output); // computed previously
      ASSERT_EQ_QUIET(d_input, d_output); // computed previously
    }
  }
}

TYPED_TEST(AdjacentDifferenceVariableTests, TestAdjacentDifferenceInPlaceWithRelatedIteratorTypes)
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

      typename thrust::host_vector<T>::iterator h_result;
      typename thrust::device_vector<T>::iterator d_result;

      h_result = thrust::adjacent_difference(h_input.begin(), h_input.end(), h_output.begin(), thrust::plus<T>());
      d_result = thrust::adjacent_difference(d_input.begin(), d_input.end(), d_output.begin(), thrust::plus<T>());

      // in-place operation with different iterator types
      h_result = thrust::adjacent_difference(h_input.cbegin(), h_input.cend(), h_input.begin(), thrust::plus<T>());
      d_result = thrust::adjacent_difference(d_input.cbegin(), d_input.cend(), d_input.begin(), thrust::plus<T>());

      ASSERT_EQ(h_result - h_input.begin(), size);
      ASSERT_EQ(d_result - d_input.begin(), size);
      ASSERT_EQ_QUIET(h_output, h_input); // reference computed previously
      ASSERT_EQ_QUIET(d_output, d_input); // reference computed previously
    }
  }
}

TYPED_TEST(AdjacentDifferenceVariableTests, TestAdjacentDifferenceDiscardIterator)
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

      thrust::discard_iterator<> h_result;
      thrust::discard_iterator<> d_result;

      h_result = thrust::adjacent_difference(h_input.begin(), h_input.end(), thrust::make_discard_iterator());
      d_result = thrust::adjacent_difference(d_input.begin(), d_input.end(), thrust::make_discard_iterator());

      thrust::discard_iterator<> reference(static_cast<std::ptrdiff_t>(size));

      ASSERT_EQ_QUIET(reference, h_result);
      ASSERT_EQ_QUIET(reference, d_result);
    }
  }
}

template <typename InputIterator, typename OutputIterator>
OutputIterator adjacent_difference(my_system& system, InputIterator, InputIterator, OutputIterator result)
{
  system.validate_dispatch();
  return result;
}

TEST(AdjacentDifferenceTests, TestAdjacentDifferenceDispatchExplicit)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::device_vector<int> d_input(1);

  my_system sys(0);
  thrust::adjacent_difference(sys, d_input.begin(), d_input.end(), d_input.begin());

  ASSERT_EQ(true, sys.is_valid());
}

template <typename InputIterator, typename OutputIterator>
OutputIterator adjacent_difference(my_tag, InputIterator, InputIterator, OutputIterator result)
{
  *result = 13;
  return result;
}

TEST(AdjacentDifferenceTests, TestAdjacentDifferenceDispatchImplicit)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::device_vector<int> d_input(1);

  thrust::adjacent_difference(thrust::retag<my_tag>(d_input.begin()),
                              thrust::retag<my_tag>(d_input.end()),
                              thrust::retag<my_tag>(d_input.begin()));

  ASSERT_EQ(13, d_input.front());
}

__global__ THRUST_HIP_LAUNCH_BOUNDS_DEFAULT void AdjacentDifferenceKernel(int const N, int* array)
{
  if (threadIdx.x == 0)
  {
    thrust::device_ptr<int> begin(array);
    thrust::device_ptr<int> end(array + N);
    // thrust::advance(begin,2);
    thrust::adjacent_difference(thrust::hip::par, begin, end, begin);
  }
}

TEST(AdjacentDifferenceTests, TestAdjacentDifferenceDevice)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());
  for (auto size : get_sizes())
  {
    SCOPED_TRACE(testing::Message() << "with size= " << size);

    for (auto seed : get_seeds())
    {
      SCOPED_TRACE(testing::Message() << "with seed= " << seed);

      thrust::host_vector<int> h_data   = get_random_data<int>(size, 0, static_cast<int>(size), seed);
      thrust::device_vector<int> d_data = h_data;
      thrust::adjacent_difference(h_data.begin(), h_data.end(), h_data.begin());
      hipLaunchKernelGGL(
        AdjacentDifferenceKernel, dim3(1, 1, 1), dim3(128, 1, 1), 0, 0, size, thrust::raw_pointer_cast(&d_data[0]));

      ASSERT_EQ(h_data, d_data);
    }
  }
}

#endif // THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HIP
