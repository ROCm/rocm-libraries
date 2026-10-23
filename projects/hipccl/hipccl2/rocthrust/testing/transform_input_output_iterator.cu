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

#include <thrust/copy.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_input_output_iterator.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>

#include <unittest/unittest.h>

// There is an unfortunate miscompilation of the gcc-13 vectorizer leading to OOB writes
// Adding this attribute suffices that this miscompilation does not appear anymore
#if defined(THRUST_HOST_COMPILER_GCC) && __GNUC__ >= 13
#  define THRUST_DISABLE_BROKEN_GCC_VECTORIZER __attribute__((optimize("no-tree-vectorize")))
#else // defined(THRUST_HOST_COMPILER_GCC) && __GNUC__ < 13
#  define THRUST_DISABLE_BROKEN_GCC_VECTORIZER
#endif

template <class Vector>
THRUST_DISABLE_BROKEN_GCC_VECTORIZER void TestTransformInputOutputIterator()
{
  using T = typename Vector::value_type;

  using InputFunction  = thrust::negate<T>;
  using OutputFunction = thrust::square<T>;
  using Iterator       = typename Vector::iterator;

  Vector input(4);
  Vector squared(4);
  Vector negated(4);

  // initialize input
  thrust::sequence(input.begin(), input.end(), 1);

  // construct transform_iterator
  thrust::transform_input_output_iterator<InputFunction, OutputFunction, Iterator> transform_iter(
    squared.begin(), InputFunction(), OutputFunction());

  // transform_iter writes squared value
  thrust::copy(input.begin(), input.end(), transform_iter);

  Vector gold_squared{1, 4, 9, 16};

  ASSERT_EQUAL(squared, gold_squared);

  // negated value read from transform_iter
  thrust::copy_n(transform_iter, squared.size(), negated.begin());

  Vector gold_negated{-1, -4, -9, -16};

  ASSERT_EQUAL(negated, gold_negated);
}
DECLARE_VECTOR_UNITTEST(TestTransformInputOutputIterator);

template <class Vector>
THRUST_DISABLE_BROKEN_GCC_VECTORIZER void TestMakeTransformInputOutputIterator()
{
  using T = typename Vector::value_type;

  using InputFunction  = thrust::negate<T>;
  using OutputFunction = thrust::square<T>;

  Vector input(4);
  Vector negated(4);
  Vector squared(4);

  // initialize input
  thrust::sequence(input.begin(), input.end(), 1);

  // negated value read from transform iterator
  thrust::copy_n(thrust::make_transform_input_output_iterator(input.begin(), InputFunction(), OutputFunction()),
                 input.size(),
                 negated.begin());

  Vector gold_negated{-1, -2, -3, -4};

  ASSERT_EQUAL(negated, gold_negated);

  // squared value written by transform iterator
  thrust::copy(negated.begin(),
               negated.end(),
               thrust::make_transform_input_output_iterator(squared.begin(), InputFunction(), OutputFunction()));

  Vector gold_squared{1, 4, 9, 16};

  ASSERT_EQUAL(squared, gold_squared);
}
DECLARE_VECTOR_UNITTEST(TestMakeTransformInputOutputIterator);

template <typename T>
struct TestTransformInputOutputIteratorScan
{
  void operator()(const size_t n)
  {
    thrust::host_vector<T> h_data   = unittest::random_samples<T>(n);
    thrust::device_vector<T> d_data = h_data;

    thrust::host_vector<T> h_result(n);
    thrust::device_vector<T> d_result(n);

    // run on host (uses forward iterator negate)
    thrust::inclusive_scan(
      thrust::make_transform_input_output_iterator(h_data.begin(), thrust::negate<T>(), ::internal::identity{}),
      thrust::make_transform_input_output_iterator(h_data.end(), thrust::negate<T>(), ::internal::identity{}),
      h_result.begin());
    // run on device (uses reverse iterator negate)
    thrust::inclusive_scan(
      d_data.begin(),
      d_data.end(),
      thrust::make_transform_input_output_iterator(d_result.begin(), thrust::square<T>(), thrust::negate<T>()));

    ASSERT_EQUAL(h_result, d_result);
  }
};
VariableUnitTest<TestTransformInputOutputIteratorScan, IntegralTypes> TestTransformInputOutputIteratorScanInstance;
