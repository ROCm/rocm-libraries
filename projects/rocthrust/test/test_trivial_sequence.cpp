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

#include <thrust/detail/trivial_sequence.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/type_traits/is_contiguous_iterator.h>
#include <thrust/universal_vector.h>

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

TESTS_DEFINE(TrivialSequenceTests, VectorTestsParams);

template <typename Iterator>
void test_func(Iterator first, Iterator last)
{
  using System = typename thrust::iterator_system<Iterator>::type;
  System system;
  thrust::detail::trivial_sequence<Iterator, System> ts(system, first, last);
  using ValueType = typename thrust::iterator_traits<Iterator>::value_type;

  ASSERT_EQ_QUIET((ValueType) ts.begin()[0], ValueType(0, 11));
  ASSERT_EQ_QUIET((ValueType) ts.begin()[1], ValueType(2, 11));
  ASSERT_EQ_QUIET((ValueType) ts.begin()[2], ValueType(1, 13));
  ASSERT_EQ_QUIET((ValueType) ts.begin()[3], ValueType(0, 10));
  ASSERT_EQ_QUIET((ValueType) ts.begin()[4], ValueType(1, 12));

  ts.begin()[0] = ValueType(0, 0);
  ts.begin()[1] = ValueType(0, 0);
  ts.begin()[2] = ValueType(0, 0);
  ts.begin()[3] = ValueType(0, 0);
  ts.begin()[4] = ValueType(0, 0);

  using TrivialIterator = typename thrust::detail::trivial_sequence<Iterator, System>::iterator_type;

  ASSERT_EQ((bool) thrust::is_contiguous_iterator<Iterator>::value, false);
  ASSERT_EQ((bool) thrust::is_contiguous_iterator<TrivialIterator>::value, true);
}

TYPED_TEST(TrivialSequenceTests, TestTrivialSequence)
{
  using Vector = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  Vector A{0, 2, 1, 0, 1};
  Vector B{11, 11, 13, 10, 12};

  test_func(thrust::make_zip_iterator(thrust::make_tuple(A.begin(), B.begin())),
            thrust::make_zip_iterator(thrust::make_tuple(A.end(), B.end())));

  Vector refA{0, 2, 1, 0, 1};
  ASSERT_EQ(A, refA);
  // ensure that values weren't modified
  Vector refB{11, 11, 13, 10, 12};
  ASSERT_EQ(B, refB);
}
