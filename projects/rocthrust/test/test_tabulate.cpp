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
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/retag.h>
#include <thrust/tabulate.h>
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

TESTS_DEFINE(TabulateTests, FullTestsParams);
TESTS_DEFINE(TabulatePrimitiveTests, NumericalTestsParams);
TESTS_DEFINE(TabulateVectorTests, VectorTestsParams)

template <typename ForwardIterator, typename UnaryOperation>
void tabulate(my_system& system, ForwardIterator, ForwardIterator, UnaryOperation)
{
  system.validate_dispatch();
}

TEST(TabulateTests, TestTabulateDispatchExplicit)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::device_vector<int> vec(1);

  my_system sys(0);
  thrust::tabulate(sys, vec.begin(), vec.end(), ::internal::identity{});

  ASSERT_EQ(true, sys.is_valid());
}

template <typename ForwardIterator, typename UnaryOperation>
void tabulate(my_tag, ForwardIterator first, ForwardIterator, UnaryOperation)
{
  *first = 13;
}

TEST(TabulateTests, TestTabulateDispatchImplicit)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::device_vector<int> vec(1);

  thrust::tabulate(thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.end()), ::internal::identity{});

  ASSERT_EQ(13, vec.front());
}

TYPED_TEST(TabulateVectorTests, TestTabulateSimple)
{
  using Vector = typename TestFixture::input_type;
  using namespace thrust::placeholders;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  Vector v(5);

  thrust::tabulate(v.begin(), v.end(), ::internal::identity{});

  Vector ref{0, 1, 2, 3, 4};
  ASSERT_EQ(v, ref);

  thrust::tabulate(v.begin(), v.end(), -_1);

  ref = {0, -1, -2, -3, -4};
  ASSERT_EQ(v, ref);

  thrust::tabulate(v.begin(), v.end(), _1 * _1 * _1);

  ref = {0, 1, 8, 27, 64};
  ASSERT_EQ(v, ref);
}

template <class OutputType>
struct nonconst_op
{
  THRUST_HIP_FUNCTION
  OutputType operator()(size_t idx)
  {
    return (OutputType) (idx >= 3);
  }
};

TYPED_TEST(TabulateTests, TestTabulateSimpleNonConstOP)
{
  using Vector = typename TestFixture::input_type;
  using T      = typename Vector::value_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  Vector v(5);

  thrust::tabulate(v.begin(), v.end(), nonconst_op<T>());

  ASSERT_EQ(v[0], T(0));
  ASSERT_EQ(v[1], T(0));
  ASSERT_EQ(v[2], T(0));
  ASSERT_EQ(v[3], T(1));
  ASSERT_EQ(v[4], T(1));
}

TYPED_TEST(TabulatePrimitiveTests, TestTabulate)
{
  using T = typename TestFixture::input_type;
  using namespace thrust::placeholders;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  for (auto size : get_sizes())
  {
    SCOPED_TRACE(testing::Message() << "with size = " << size);

    thrust::host_vector<T> h_data(size);
    thrust::device_vector<T> d_data(size);

    thrust::tabulate(h_data.begin(), h_data.end(), _1 * _1 + 13);
    thrust::tabulate(d_data.begin(), d_data.end(), _1 * _1 + 13);

    ASSERT_EQ(h_data, d_data);

    thrust::tabulate(h_data.begin(), h_data.end(), (_1 - 7) * _1);
    thrust::tabulate(d_data.begin(), d_data.end(), (_1 - 7) * _1);

    ASSERT_EQ(h_data, d_data);
  }
}

TEST(TabulateTests, TestTabulateToDiscardIterator)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  for (auto size : get_sizes())
  {
    SCOPED_TRACE(testing::Message() << "with size = " << size);

    thrust::tabulate(thrust::discard_iterator<thrust::device_system_tag>(),
                     thrust::discard_iterator<thrust::device_system_tag>(size),
                     ::internal::identity{});
  }

  // nothing to check -- just make sure it compiles
}
