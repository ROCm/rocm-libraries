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
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/iterator/retag.h>
#include <thrust/transform_scan.h>

#include "test_header.hpp"

TESTS_DEFINE(TransformScanTests, FullTestsParams);
TESTS_DEFINE(TransformScanVariablesTests, NumericalTestsParams);
TESTS_DEFINE(TransformScanVectorTests, VectorSignedIntegerTestsParams);

template <typename InputIterator, typename OutputIterator, typename UnaryFunction, typename AssociativeOperator>
__host__ __device__ OutputIterator transform_inclusive_scan(
  my_system& system, InputIterator, InputIterator, OutputIterator result, UnaryFunction, AssociativeOperator)
{
  system.validate_dispatch();
  return result;
}

TEST(TransformScanTests, TestTransformInclusiveScanDispatchExplicit)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::device_vector<int> vec(1);

  my_system sys(0);
  thrust::transform_inclusive_scan(sys, vec.begin(), vec.begin(), vec.begin(), 0, 0);

  ASSERT_EQ(true, sys.is_valid());
}

template <typename InputIterator, typename OutputIterator, typename UnaryFunction, typename AssociativeOperator>
__host__ __device__ OutputIterator transform_inclusive_scan(
  my_tag, InputIterator, InputIterator, OutputIterator result, UnaryFunction, AssociativeOperator)
{
  *result = 13;
  return result;
}

TEST(TransformScanTests, TestTransformInclusiveScanDispatchImplicit)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::device_vector<int> vec(1);

  thrust::transform_inclusive_scan(
    thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.begin()), 0, 0);

  ASSERT_EQ(13, vec.front());
}

template <typename InputIterator, typename OutputIterator, typename UnaryFunction, typename T, typename AssociativeOperator>
__host__ __device__ OutputIterator transform_exclusive_scan(
  my_system& system, InputIterator, InputIterator, OutputIterator result, UnaryFunction, T, AssociativeOperator)
{
  system.validate_dispatch();
  return result;
}

TEST(TransformScanTests, TestTransformExclusiveScanDispatchExplicit)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::device_vector<int> vec(1);

  my_system sys(0);
  thrust::transform_exclusive_scan(sys, vec.begin(), vec.begin(), vec.begin(), 0, 0, 0);

  ASSERT_EQ(true, sys.is_valid());
}

template <typename InputIterator, typename OutputIterator, typename UnaryFunction, typename T, typename AssociativeOperator>
__host__ __device__ OutputIterator transform_exclusive_scan(
  my_tag, InputIterator, InputIterator, OutputIterator result, UnaryFunction, T, AssociativeOperator)
{
  *result = 13;
  return result;
}

TEST(TransformScanTests, TestTransformExclusiveScanDispatchImplicit)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::device_vector<int> vec(1);

  thrust::transform_exclusive_scan(
    thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.begin()), 0, 0, 0);

  ASSERT_EQ(13, vec.front());
}

TYPED_TEST(TransformScanVectorTests, TestTransformScanSimple)
{
  using Vector = typename TestFixture::input_type;
  using T      = typename Vector::value_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  typename Vector::iterator iter;

  Vector input(5);
  Vector result(5);
  Vector output(5);

  input[0] = 1;
  input[1] = 3;
  input[2] = -2;
  input[3] = 4;
  input[4] = -5;

  Vector input_copy(input);

  // inclusive scan
  iter = thrust::transform_inclusive_scan(
    input.begin(), input.end(), output.begin(), thrust::negate<T>(), thrust::plus<T>());
  result[0] = -1;
  result[1] = -4;
  result[2] = -2;
  result[3] = -6;
  result[4] = -1;
  ASSERT_EQ(iter - output.begin(), input.size());
  ASSERT_EQ(input, input_copy);
  ASSERT_EQ(output, result);

  // exclusive scan with 0 init
  iter = thrust::transform_exclusive_scan(
    input.begin(), input.end(), output.begin(), thrust::negate<T>(), 0, thrust::plus<T>());
  result[0] = 0;
  result[1] = -1;
  result[2] = -4;
  result[3] = -2;
  result[4] = -6;
  ASSERT_EQ(iter - output.begin(), input.size());
  ASSERT_EQ(input, input_copy);
  ASSERT_EQ(output, result);

  // exclusive scan with nonzero init
  iter = thrust::transform_exclusive_scan(
    input.begin(), input.end(), output.begin(), thrust::negate<T>(), 3, thrust::plus<T>());
  result[0] = 3;
  result[1] = 2;
  result[2] = -1;
  result[3] = 1;
  result[4] = -3;
  ASSERT_EQ(iter - output.begin(), input.size());
  ASSERT_EQ(input, input_copy);
  ASSERT_EQ(output, result);

  // inplace inclusive scan
  input = input_copy;
  iter =
    thrust::transform_inclusive_scan(input.begin(), input.end(), input.begin(), thrust::negate<T>(), thrust::plus<T>());
  result[0] = -1;
  result[1] = -4;
  result[2] = -2;
  result[3] = -6;
  result[4] = -1;
  ASSERT_EQ(iter - input.begin(), input.size());
  ASSERT_EQ(input, result);

  // inplace exclusive scan with init
  input = input_copy;
  iter  = thrust::transform_exclusive_scan(
    input.begin(), input.end(), input.begin(), thrust::negate<T>(), 3, thrust::plus<T>());
  result[0] = 3;
  result[1] = 2;
  result[2] = -1;
  result[3] = 1;
  result[4] = -3;
  ASSERT_EQ(iter - input.begin(), input.size());
  ASSERT_EQ(input, result);
}

TYPED_TEST(TransformScanVariablesTests, TestTransformScan)
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

      thrust::transform_inclusive_scan(
        h_input.begin(), h_input.end(), h_output.begin(), thrust::negate<T>(), thrust::plus<T>());
      thrust::transform_inclusive_scan(
        d_input.begin(), d_input.end(), d_output.begin(), thrust::negate<T>(), thrust::plus<T>());
      test_equality_scan(h_output, d_output);

      thrust::transform_exclusive_scan(
        h_input.begin(), h_input.end(), h_output.begin(), thrust::negate<T>(), (T) 11, thrust::plus<T>());
      thrust::transform_exclusive_scan(
        d_input.begin(), d_input.end(), d_output.begin(), thrust::negate<T>(), (T) 11, thrust::plus<T>());
      test_equality_scan(h_output, d_output);

      // in-place scans
      h_output = h_input;
      d_output = d_input;
      thrust::transform_inclusive_scan(
        h_output.begin(), h_output.end(), h_output.begin(), thrust::negate<T>(), thrust::plus<T>());
      thrust::transform_inclusive_scan(
        d_output.begin(), d_output.end(), d_output.begin(), thrust::negate<T>(), thrust::plus<T>());
      test_equality_scan(h_output, d_output);

      h_output = h_input;
      d_output = d_input;
      thrust::transform_exclusive_scan(
        h_output.begin(), h_output.end(), h_output.begin(), thrust::negate<T>(), (T) 11, thrust::plus<T>());
      thrust::transform_exclusive_scan(
        d_output.begin(), d_output.end(), d_output.begin(), thrust::negate<T>(), (T) 11, thrust::plus<T>());
      test_equality_scan(h_output, d_output);
    }
  }
};

TYPED_TEST(TransformScanVectorTests, TestTransformScanCountingIterator)
{
  using Vector = typename TestFixture::input_type;
  using T      = typename Vector::value_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  using space = typename thrust::iterator_system<typename Vector::iterator>::type;

  thrust::counting_iterator<T, space> first(1);

  Vector result(3);

  thrust::transform_inclusive_scan(first, first + 3, result.begin(), thrust::negate<T>(), thrust::plus<T>());

  ASSERT_EQ(result[0], -1);
  ASSERT_EQ(result[1], -3);
  ASSERT_EQ(result[2], -6);
}

TYPED_TEST(TransformScanVariablesTests, TestTransformScanToDiscardIterator)
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

      thrust::discard_iterator<> h_result = thrust::transform_inclusive_scan(
        h_input.begin(), h_input.end(), thrust::make_discard_iterator(), thrust::negate<T>(), thrust::plus<T>());

      thrust::discard_iterator<> d_result = thrust::transform_inclusive_scan(
        d_input.begin(), d_input.end(), thrust::make_discard_iterator(), thrust::negate<T>(), thrust::plus<T>());
      ASSERT_EQ_QUIET(reference, h_result);
      ASSERT_EQ_QUIET(reference, d_result);

      h_result = thrust::transform_exclusive_scan(
        h_input.begin(), h_input.end(), thrust::make_discard_iterator(), thrust::negate<T>(), (T) 11, thrust::plus<T>());

      d_result = thrust::transform_exclusive_scan(
        d_input.begin(), d_input.end(), thrust::make_discard_iterator(), thrust::negate<T>(), (T) 11, thrust::plus<T>());

      ASSERT_EQ_QUIET(reference, h_result);
      ASSERT_EQ_QUIET(reference, d_result);
    }
  }
}

TYPED_TEST(TransformScanVariablesTests, TestValueCategoryDeduction)
{
  using T = typename TestFixture::input_type;

  thrust::device_vector<T> vec;

  T a_h[10] = {5, 0, 5, 8, 6, 7, 5, 3, 0, 9};
  vec.assign((T*) a_h, a_h + 10);

  thrust::transform_inclusive_scan(
    thrust::device, vec.cbegin(), vec.cend(), vec.begin(), thrust::identity<>{}, thrust::maximum<>{});

  ASSERT_EQ(T{5}, vec[0]);
  ASSERT_EQ(T{5}, vec[1]);
  ASSERT_EQ(T{5}, vec[2]);
  ASSERT_EQ(T{8}, vec[3]);
  ASSERT_EQ(T{8}, vec[4]);
  ASSERT_EQ(T{8}, vec[5]);
  ASSERT_EQ(T{8}, vec[6]);
  ASSERT_EQ(T{8}, vec[7]);
  ASSERT_EQ(T{8}, vec[8]);
  ASSERT_EQ(T{9}, vec[9]);

  vec.assign((T*) a_h, a_h + 10);
  thrust::transform_exclusive_scan(
    thrust::device, vec.cbegin(), vec.cend(), vec.begin(), thrust::identity<>{}, T{}, thrust::maximum<>{});

  ASSERT_EQ(T{0}, vec[0]);
  ASSERT_EQ(T{5}, vec[1]);
  ASSERT_EQ(T{5}, vec[2]);
  ASSERT_EQ(T{5}, vec[3]);
  ASSERT_EQ(T{8}, vec[4]);
  ASSERT_EQ(T{8}, vec[5]);
  ASSERT_EQ(T{8}, vec[6]);
  ASSERT_EQ(T{8}, vec[7]);
  ASSERT_EQ(T{8}, vec[8]);
  ASSERT_EQ(T{8}, vec[9]);
}

TEST(TransformScanTests, TestTransformScanConstAccumulator)
{
  using Vector = thrust::device_vector<int>;
  using T      = Vector::value_type;

  Vector::iterator iter;

  Vector input(5);
  Vector reference(5);
  Vector output(5);

  input[0] = 1;
  input[1] = 3;
  input[2] = -2;
  input[3] = 4;
  input[4] = -5;

  thrust::transform_inclusive_scan(input.begin(), input.end(), output.begin(), thrust::identity<T>(), thrust::plus<T>());
  thrust::inclusive_scan(input.begin(), input.end(), reference.begin(), thrust::plus<T>());

  ASSERT_EQ(output, reference);
}
