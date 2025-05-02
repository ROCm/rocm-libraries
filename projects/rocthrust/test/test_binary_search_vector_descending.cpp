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
#include <thrust/detail/allocator/allocator_traits.h>
#include <thrust/functional.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include "test_param_fixtures.hpp"
#include "test_real_assertions.hpp"
#include "test_utils.hpp"

TESTS_DEFINE(BinarySearchVectorDescendingTests, FullTestsParams);
TESTS_DEFINE(BinarySearchVectorDescendingIntegerTests, SignedIntegerTestsParams);

//////////////////////
// Vector Functions //
//////////////////////

// convert xxx_vector<T1> to xxx_vector<T2>
template <class ExampleVector, typename NewType>
struct vector_like
{
  using alloc        = typename ExampleVector::allocator_type;
  using alloc_traits = typename thrust::detail::allocator_traits<alloc>;
  using new_alloc    = typename alloc_traits::template rebind_alloc<NewType>;
  using type         = thrust::detail::vector_base<NewType, new_alloc>;
};

TYPED_TEST(BinarySearchVectorDescendingTests, TestVectorLowerBoundDescendingSimple)
{
  using Vector = typename TestFixture::input_type;
  using T      = typename Vector::value_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  Vector vec{8, 7, 5, 2, 0};

  Vector input(10);
  thrust::sequence(input.begin(), input.end());

  using int_type  = typename Vector::difference_type;
  using IntVector = typename vector_like<Vector, int_type>::type;

  // test with integral output type
  IntVector integral_output(10);
  typename IntVector::iterator output_end = thrust::lower_bound(
    vec.begin(), vec.end(), input.begin(), input.end(), integral_output.begin(), thrust::greater<T>());

  ASSERT_EQ_QUIET(integral_output.end(), output_end);

  IntVector ref{4, 4, 3, 3, 3, 2, 2, 1, 0, 0};
  ASSERT_EQ(ref, integral_output);
}

TYPED_TEST(BinarySearchVectorDescendingTests, TestVectorUpperBoundDescendingSimple)
{
  using Vector = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  Vector vec{8, 7, 5, 2, 0};

  Vector input(10);
  thrust::sequence(input.begin(), input.end());

  using int_type  = typename Vector::difference_type;
  using T         = typename Vector::value_type;
  using IntVector = typename vector_like<Vector, int_type>::type;

  // test with integral output type
  IntVector integral_output(10);
  typename IntVector::iterator output_end = thrust::upper_bound(
    vec.begin(), vec.end(), input.begin(), input.end(), integral_output.begin(), thrust::greater<T>());

  ASSERT_EQ_QUIET(output_end, integral_output.end());

  IntVector ref{5, 4, 4, 3, 3, 3, 2, 2, 1, 0};
  ASSERT_EQ(ref, integral_output);
}

TYPED_TEST(BinarySearchVectorDescendingTests, TestVectorBinarySearchDescendingSimple)
{
  using Vector = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  Vector vec{8, 7, 5, 2, 0};

  Vector input(10);
  thrust::sequence(input.begin(), input.end());

  using BoolVector = typename vector_like<Vector, bool>::type;
  using int_type   = typename Vector::difference_type;
  using T          = typename Vector::value_type;
  using IntVector  = typename vector_like<Vector, int_type>::type;

  // test with boolean output type
  BoolVector bool_output(10);
  typename BoolVector::iterator bool_output_end = thrust::binary_search(
    vec.begin(), vec.end(), input.begin(), input.end(), bool_output.begin(), thrust::greater<T>());

  ASSERT_EQ_QUIET(bool_output_end, bool_output.end());

  BoolVector bool_ref{true, false, true, false, false, true, false, true, true, false};
  ASSERT_EQ(bool_ref, bool_output);

  // test with integral output type
  IntVector integral_output(10, 2);
  typename IntVector::iterator int_output_end = thrust::binary_search(
    vec.begin(), vec.end(), input.begin(), input.end(), integral_output.begin(), thrust::greater<T>());

  ASSERT_EQ_QUIET(int_output_end, integral_output.end());

  IntVector int_ref{1, 0, 1, 0, 0, 1, 0, 1, 1, 0};

  ASSERT_EQ(int_ref, integral_output);
}

TYPED_TEST(BinarySearchVectorDescendingIntegerTests, TestVectorLowerBoundDescending)
{
  using T = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  for (auto size : get_sizes())
  {
    SCOPED_TRACE(testing::Message() << "with size= " << size);

    for (auto seed : get_seeds())
    {
      SCOPED_TRACE(testing::Message() << "with seed= " << seed);

      thrust::host_vector<T> h_vec =
        get_random_data<T>(size, get_default_limits<T>::min(), get_default_limits<T>::max(), seed);
      thrust::sort(h_vec.begin(), h_vec.end(), thrust::greater<T>());
      thrust::device_vector<T> d_vec = h_vec;

      thrust::host_vector<T> h_input = get_random_data<T>(
        2 * size, get_default_limits<T>::min(), get_default_limits<T>::max(), seed + seed_value_addition);
      thrust::device_vector<T> d_input = h_input;

      using int_type = typename thrust::host_vector<T>::difference_type;
      thrust::host_vector<int_type> h_output(2 * size);
      thrust::device_vector<int_type> d_output(2 * size);

      thrust::lower_bound(
        h_vec.begin(), h_vec.end(), h_input.begin(), h_input.end(), h_output.begin(), thrust::greater<T>());
      thrust::lower_bound(
        d_vec.begin(), d_vec.end(), d_input.begin(), d_input.end(), d_output.begin(), thrust::greater<T>());

      ASSERT_EQ(h_output, d_output);
    }
  }
}

TYPED_TEST(BinarySearchVectorDescendingIntegerTests, TestVectorUpperBoundDescending)
{
  using T = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  for (auto size : get_sizes())
  {
    SCOPED_TRACE(testing::Message() << "with size= " << size);

    for (auto seed : get_seeds())
    {
      SCOPED_TRACE(testing::Message() << "with seed= " << seed);

      thrust::host_vector<T> h_vec =
        get_random_data<T>(size, get_default_limits<T>::min(), get_default_limits<T>::max(), seed);
      thrust::sort(h_vec.begin(), h_vec.end(), thrust::greater<T>());
      thrust::device_vector<T> d_vec = h_vec;

      thrust::host_vector<T> h_input = get_random_data<T>(
        2 * size, get_default_limits<T>::min(), get_default_limits<T>::max(), seed + seed_value_addition);
      thrust::device_vector<T> d_input = h_input;

      using int_type = typename thrust::host_vector<T>::difference_type;
      thrust::host_vector<int_type> h_output(2 * size);
      thrust::device_vector<int_type> d_output(2 * size);

      thrust::upper_bound(
        h_vec.begin(), h_vec.end(), h_input.begin(), h_input.end(), h_output.begin(), thrust::greater<T>());
      thrust::upper_bound(
        d_vec.begin(), d_vec.end(), d_input.begin(), d_input.end(), d_output.begin(), thrust::greater<T>());

      ASSERT_EQ(h_output, d_output);
    }
  }
}

TYPED_TEST(BinarySearchVectorDescendingIntegerTests, TestVectorBinarySearchDescending)
{
  using T = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  for (auto size : get_sizes())
  {
    SCOPED_TRACE(testing::Message() << "with size= " << size);

    for (auto seed : get_seeds())
    {
      SCOPED_TRACE(testing::Message() << "with seed= " << seed);

      thrust::host_vector<T> h_vec =
        get_random_data<T>(size, get_default_limits<T>::min(), get_default_limits<T>::max(), seed);
      thrust::sort(h_vec.begin(), h_vec.end(), thrust::greater<T>());
      thrust::device_vector<T> d_vec = h_vec;

      thrust::host_vector<T> h_input = get_random_data<T>(
        2 * size, get_default_limits<T>::min(), get_default_limits<T>::max(), seed + seed_value_addition);
      thrust::device_vector<T> d_input = h_input;

      using int_type = typename thrust::host_vector<T>::difference_type;
      thrust::host_vector<int_type> h_output(2 * size);
      thrust::device_vector<int_type> d_output(2 * size);

      thrust::binary_search(
        h_vec.begin(), h_vec.end(), h_input.begin(), h_input.end(), h_output.begin(), thrust::greater<T>());
      thrust::binary_search(
        d_vec.begin(), d_vec.end(), d_input.begin(), d_input.end(), d_output.begin(), thrust::greater<T>());

      ASSERT_EQ(h_output, d_output);
    }
  }
}
