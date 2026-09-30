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

#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/retag.h>
#include <thrust/reverse.h>
#include <thrust/universal_vector.h>

#include <cstddef>

#include "test_param_fixtures.hpp"
#include "test_real_assertions.hpp"
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

using ReverseTypes = ::testing::Types<Params<int8_t>, Params<int16_t>, Params<int32_t>>;

TESTS_DEFINE(ReverseTests, VectorTestsParams);
TESTS_DEFINE(ReverseVariableUnitTest, ReverseTypes);

TYPED_TEST(ReverseTests, TestReverseSimple)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  using Vector = typename TestFixture::input_type;

  Vector data{1, 2, 3, 4, 5};

  thrust::reverse(data.begin(), data.end());

  Vector ref{5, 4, 3, 2, 1};

  ASSERT_EQ(ref, data);
}

template <typename BidirectionalIterator>
void reverse(my_system& system, BidirectionalIterator, BidirectionalIterator)
{
  system.validate_dispatch();
}

TEST(ReverseTests, TestReverseDispatchExplicit)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::device_vector<int> vec(1);

  my_system sys(0);
  thrust::reverse(sys, vec.begin(), vec.begin());

  ASSERT_EQ(true, sys.is_valid());
}

template <typename BidirectionalIterator>
void reverse(my_tag, BidirectionalIterator first, BidirectionalIterator)
{
  *first = 13;
}

TEST(ReverseTests, TestReverseDispatchImplicit)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::device_vector<int> vec(1);

  thrust::reverse(thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.begin()));

  ASSERT_EQ(13, vec.front());
}

TYPED_TEST(ReverseTests, TestReverseCopySimple)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  using Vector = typename TestFixture::input_type;

  using Iterator = typename Vector::iterator;

  Vector input{1, 2, 3, 4, 5};
  Vector output(8); // arm GCC is complaining about destination size

  Iterator iter = thrust::reverse_copy(input.begin(), input.end(), output.begin());

  output.resize(5);
  Vector ref{5, 4, 3, 2, 1};
  ASSERT_EQ(5, iter - output.begin());
  ASSERT_EQ(ref, output);
}

template <typename BidirectionalIterator, typename OutputIterator>
OutputIterator reverse_copy(my_system& system, BidirectionalIterator, BidirectionalIterator, OutputIterator result)
{
  system.validate_dispatch();
  return result;
}

TEST(ReverseTests, TestReverseCopyDispatchExplicit)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::device_vector<int> vec(1);

  my_system sys(0);
  thrust::reverse_copy(sys, vec.begin(), vec.end(), vec.begin());

  ASSERT_EQ(true, sys.is_valid());
}

template <typename BidirectionalIterator, typename OutputIterator>
OutputIterator reverse_copy(my_tag, BidirectionalIterator, BidirectionalIterator, OutputIterator result)
{
  *result = 13;
  return result;
}

TEST(ReverseTests, TestReverseCopyDispatchImplicit)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::device_vector<int> vec(1);

  thrust::reverse_copy(
    thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.end()), thrust::retag<my_tag>(vec.begin()));

  ASSERT_EQ(13, vec.front());
}

TYPED_TEST(ReverseVariableUnitTest, TestReverse)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  using T = typename TestFixture::input_type;

  for (auto size : get_sizes())
  {
    SCOPED_TRACE(testing::Message() << "with size= " << size);

    thrust::host_vector<T> h_data   = random_integers<T>(size);
    thrust::device_vector<T> d_data = h_data;

    thrust::reverse(h_data.begin(), h_data.end());
    thrust::reverse(d_data.begin(), d_data.end());

    ASSERT_EQ(h_data, d_data);
  }
}

TYPED_TEST(ReverseVariableUnitTest, TestReverseCopy)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  using T = typename TestFixture::input_type;

  for (auto size : get_sizes())
  {
    SCOPED_TRACE(testing::Message() << "with size= " << size);

    thrust::host_vector<T> h_data   = random_integers<T>(size);
    thrust::device_vector<T> d_data = h_data;

    thrust::host_vector<T> h_result(size);
    thrust::device_vector<T> d_result(size);

    thrust::reverse_copy(h_data.begin(), h_data.end(), h_result.begin());
    thrust::reverse_copy(d_data.begin(), d_data.end(), d_result.begin());

    ASSERT_EQ(h_result, d_result);
  }
}

TYPED_TEST(ReverseVariableUnitTest, TestReverseCopyToDiscardIterator)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  using T = typename TestFixture::input_type;

  for (auto size : get_sizes())
  {
    SCOPED_TRACE(testing::Message() << "with size= " << size);

    thrust::host_vector<T> h_data   = random_integers<T>(size);
    thrust::device_vector<T> d_data = h_data;

    thrust::discard_iterator<> h_result =
      thrust::reverse_copy(h_data.begin(), h_data.end(), thrust::make_discard_iterator());

    thrust::discard_iterator<> d_result =
      thrust::reverse_copy(d_data.begin(), d_data.end(), thrust::make_discard_iterator());

    thrust::discard_iterator<> reference(size);

    ASSERT_EQ_QUIET(reference, h_result);
    ASSERT_EQ_QUIET(reference, d_result);
  }
}
