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

#include <thrust/iterator/reverse_iterator.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>

#include "test_param_fixtures.hpp"
#include "test_real_assertions.hpp"
#include "test_utils.hpp"

#include _THRUST_STD_INCLUDE(type_traits)

TESTS_DEFINE(ReverseIteratorTests, FullTestsParams);

TESTS_DEFINE(PrimitiveReverseIteratorTests, NumericalTestsParams);

TEST(ReverseIteratorTests, UsingHip)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  ASSERT_EQ(THRUST_DEVICE_SYSTEM, THRUST_DEVICE_SYSTEM_HIP);
}

TEST(ReverseIteratorTests, TestReverseIteratorCopyConstructor)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::host_vector<int> h_v(1, 13);

  thrust::reverse_iterator<thrust::host_vector<int>::iterator> h_iter0(h_v.end());
  thrust::reverse_iterator<thrust::host_vector<int>::iterator> h_iter1(h_iter0);

  ASSERT_EQ_QUIET(h_iter0, h_iter1);
  ASSERT_EQ(*h_iter0, *h_iter1);

  thrust::device_vector<int> d_v(1, 13);

  thrust::reverse_iterator<thrust::device_vector<int>::iterator> d_iter2(d_v.end());
  thrust::reverse_iterator<thrust::device_vector<int>::iterator> d_iter3(d_iter2);

  ASSERT_EQ_QUIET(d_iter2, d_iter3);
  ASSERT_EQ(*d_iter2, *d_iter3);
}
static_assert(_THRUST_STD::is_trivially_copy_constructible<thrust::reverse_iterator<int*>>::value, "");
static_assert(_THRUST_STD::is_trivially_copyable<thrust::reverse_iterator<int*>>::value, "");

TEST(ReverseIteratorTests, TestReverseIteratorIncrement)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::host_vector<int> h_v(4);
  thrust::sequence(h_v.begin(), h_v.end());

  thrust::reverse_iterator<thrust::host_vector<int>::iterator> h_iter(h_v.end());

  ASSERT_EQ(*h_iter, 3);

  h_iter++;
  ASSERT_EQ(*h_iter, 2);

  h_iter++;
  ASSERT_EQ(*h_iter, 1);

  h_iter++;
  ASSERT_EQ(*h_iter, 0);

  thrust::device_vector<int> d_v(4);
  thrust::sequence(d_v.begin(), d_v.end());

  thrust::reverse_iterator<thrust::device_vector<int>::iterator> d_iter(d_v.end());

  ASSERT_EQ(*d_iter, 3);

  d_iter++;
  ASSERT_EQ(*d_iter, 2);

  d_iter++;
  ASSERT_EQ(*d_iter, 1);

  d_iter++;
  ASSERT_EQ(*d_iter, 0);
}

TYPED_TEST(ReverseIteratorTests, TestReverseIteratorCopy)
{
  using Vector = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  Vector source{10, 20, 30, 40};

  Vector destination(8, 0); // arm gcc is complaining here

  thrust::copy(
    thrust::make_reverse_iterator(source.end()), thrust::make_reverse_iterator(source.begin()), destination.begin());

  destination.resize(4);
  Vector ref{40, 30, 20, 10};
  ASSERT_EQ(destination, ref);
}

TYPED_TEST(PrimitiveReverseIteratorTests, TestReverseIteratorExclusiveScanSimple)
{
  using T = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  const size_t n = 10;

  thrust::host_vector<T> h_data(n);
  thrust::sequence(h_data.begin(), h_data.end());

  thrust::device_vector<T> d_data = h_data;

  thrust::host_vector<T> h_result(h_data.size());
  thrust::device_vector<T> d_result(d_data.size());

  thrust::exclusive_scan(
    thrust::make_reverse_iterator(h_data.end()), thrust::make_reverse_iterator(h_data.begin()), h_result.begin());

  thrust::exclusive_scan(
    thrust::make_reverse_iterator(d_data.end()), thrust::make_reverse_iterator(d_data.begin()), d_result.begin());

  ASSERT_EQ_QUIET(h_result, d_result);
}

TYPED_TEST(PrimitiveReverseIteratorTests, TestReverseIteratorExclusiveScan)
{
  using T = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  for (auto size : get_sizes())
  {
    SCOPED_TRACE(testing::Message() << "with size= " << size);

    thrust::host_vector<T> h_data = random_samples<T>(size);

    thrust::device_vector<T> d_data = h_data;

    thrust::host_vector<T> h_result(size);
    thrust::device_vector<T> d_result(size);

    thrust::exclusive_scan(
      thrust::make_reverse_iterator(h_data.end()), thrust::make_reverse_iterator(h_data.begin()), h_result.begin());

    thrust::exclusive_scan(
      thrust::make_reverse_iterator(d_data.end()), thrust::make_reverse_iterator(d_data.begin()), d_result.begin());

    ASSERT_EQ_QUIET(h_result, d_result);
  }
}
