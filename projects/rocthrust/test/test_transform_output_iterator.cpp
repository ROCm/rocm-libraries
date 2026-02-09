/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *  Modifications Copyright© 2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <thrust/universal_vector.h>

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

using SignedIntegralTestsParams =
  ::testing::Types<Params<signed char>, Params<short>, Params<int>, Params<long>, Params<long long>>;

TESTS_DEFINE(TransformOutputIteratorVectorTests, VectorTestsParams);
TESTS_DEFINE(TransformOutputIteratorSignedIntegralTests, SignedIntegralTestsParams);

TYPED_TEST(TransformOutputIteratorVectorTests, TestTransformOutputIterator)
{
  using Vector = typename TestFixture::input_type;
  using T      = typename Vector::value_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  using UnaryFunction = thrust::square<T>;
  using Iterator      = typename Vector::iterator;

  Vector input(4);
  Vector output(4);

  // initialize input
  thrust::sequence(input.begin(), input.end(), T{1});

  // construct transform_iterator
  thrust::transform_output_iterator<UnaryFunction, Iterator> output_iter(output.begin(), UnaryFunction());

  thrust::copy(input.begin(), input.end(), output_iter);

  Vector gold_output{1, 4, 9, 16};

  ASSERT_EQ(output, gold_output);
}

TYPED_TEST(TransformOutputIteratorVectorTests, TestMakeTransformOutputIterator)
{
  using Vector = typename TestFixture::input_type;
  using T      = typename Vector::value_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  using UnaryFunction = thrust::square<T>;

  Vector input(4);
  Vector output(4);

  // initialize input
  thrust::sequence(input.begin(), input.end(), 1);

  thrust::copy(input.begin(), input.end(), thrust::make_transform_output_iterator(output.begin(), UnaryFunction()));

  Vector gold_output{1, 4, 9, 16};
  ASSERT_EQ(output, gold_output);
}

TYPED_TEST(TransformOutputIteratorSignedIntegralTests, TestTransformOutputIteratorScan)
{
  using T = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  for (auto size : get_sizes())
  {
    SCOPED_TRACE(testing::Message() << "with size= " << size);

    thrust::host_vector<T> h_data   = random_samples<T>(size);
    thrust::device_vector<T> d_data = h_data;

    thrust::host_vector<T> h_result(size);
    thrust::device_vector<T> d_result(size);

    // run on host
    thrust::inclusive_scan(thrust::make_transform_iterator(h_data.begin(), thrust::negate<T>()),
                           thrust::make_transform_iterator(h_data.end(), thrust::negate<T>()),
                           h_result.begin());
    // run on device
    thrust::inclusive_scan(
      d_data.begin(), d_data.end(), thrust::make_transform_output_iterator(d_result.begin(), thrust::negate<T>()));

    ASSERT_EQ(h_result, d_result);
  }
}
