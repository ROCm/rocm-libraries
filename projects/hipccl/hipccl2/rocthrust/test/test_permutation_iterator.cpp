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
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <thrust/transform_reduce.h>
#include <thrust/universal_vector.h>

#include "test_param_fixtures.hpp"
#include "test_utils.hpp"

#include _THRUST_STD_INCLUDE(type_traits)

using IntegralVectorTestsParams =
  ::testing::Types<Params<thrust::host_vector<signed char>>,
                   Params<thrust::host_vector<short>>,
                   Params<thrust::host_vector<int>>,
                   Params<thrust::device_vector<signed char>>,
                   Params<thrust::device_vector<short>>,
                   Params<thrust::device_vector<int>>,
                   Params<thrust::universal_vector<int>>,
                   Params<thrust::universal_host_pinned_vector<int>>>;

TESTS_DEFINE(PermutationIteratorTests, FullTestsParams);
TESTS_DEFINE(PermutationIteratorIntegralVectorTests, IntegralVectorTestsParams);

TEST(PermutationIteratorTests, UsingHip)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  ASSERT_EQ(THRUST_DEVICE_SYSTEM, THRUST_DEVICE_SYSTEM_HIP);
}

TYPED_TEST(PermutationIteratorTests, TestPermutationIteratorSimple)
{
  using Vector   = typename TestFixture::input_type;
  using T        = typename Vector::value_type;
  using Iterator = typename Vector::iterator;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  Vector source(8);
  Vector indices{3, 0, 5, 7};

  // initialize input
  thrust::sequence(source.begin(), source.end(), 1);

  thrust::permutation_iterator<Iterator, Iterator> begin(source.begin(), indices.begin());
  thrust::permutation_iterator<Iterator, Iterator> end(source.begin(), indices.end());

  ASSERT_EQ(end - begin, 4);
  ASSERT_EQ((begin + 4) == end, true);

  ASSERT_EQ((T) *begin, 4);

  begin++;
  end--;

  ASSERT_EQ((T) *begin, 1);
  ASSERT_EQ((T) *end, 8);
  ASSERT_EQ(end - begin, 2);

  end--;

  *begin = 10;
  *end   = 20;

  Vector ref{10, 2, 3, 4, 5, 20, 7, 8};
  ASSERT_EQ(source, ref);
}
static_assert(_THRUST_STD::is_trivially_copy_constructible<thrust::permutation_iterator<int*, int*>>::value, "");
static_assert(_THRUST_STD::is_trivially_copyable<thrust::permutation_iterator<int*, int*>>::value, "");

TYPED_TEST(PermutationIteratorTests, TestPermutationIteratorGather)
{
  using Vector   = typename TestFixture::input_type;
  using Iterator = typename Vector::iterator;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  Vector source(8);
  Vector indices{3, 0, 5, 7};
  Vector output(4, 10);

  // initialize input
  thrust::sequence(source.begin(), source.end(), 1);

  thrust::permutation_iterator<Iterator, Iterator> p_source(source.begin(), indices.begin());

  thrust::copy(p_source, p_source + 4, output.begin());

  Vector ref{4, 1, 6, 8};
  ASSERT_EQ(output, ref);
}

TYPED_TEST(PermutationIteratorTests, TestPermutationIteratorScatter)
{
  using Vector   = typename TestFixture::input_type;
  using Iterator = typename Vector::iterator;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  Vector source(4, 10);
  Vector indices{3, 0, 5, 7};
  Vector output(8);

  // initialize output
  thrust::sequence(output.begin(), output.end(), 1);

  // construct transform_iterator
  thrust::permutation_iterator<Iterator, Iterator> p_output(output.begin(), indices.begin());

  thrust::copy(source.begin(), source.end(), p_output);

  Vector ref{10, 2, 3, 10, 5, 10, 7, 10};
  ASSERT_EQ(output, ref);
}

TYPED_TEST(PermutationIteratorTests, TestMakePermutationIterator)
{
  using Vector = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  Vector source(8);
  Vector indices{3, 0, 5, 7};
  Vector output(4, 10);

  // initialize input
  thrust::sequence(source.begin(), source.end(), 1);

  thrust::copy(thrust::make_permutation_iterator(source.begin(), indices.begin()),
               thrust::make_permutation_iterator(source.begin(), indices.begin()) + 4,
               output.begin());

  Vector ref{4, 1, 6, 8};
  ASSERT_EQ(output, ref);
}

TYPED_TEST(PermutationIteratorIntegralVectorTests, TestPermutationIteratorReduce)
{
  using Vector   = typename TestFixture::input_type;
  using T        = typename Vector::value_type;
  using Iterator = typename Vector::iterator;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  Vector source(8);
  Vector indices{3, 0, 5, 7};
  Vector output(4, 10);

  // initialize input
  thrust::sequence(source.begin(), source.end(), 1);

  // construct transform_iterator
  thrust::permutation_iterator<Iterator, Iterator> iter(source.begin(), indices.begin());

  T result1 = thrust::reduce(thrust::make_permutation_iterator(source.begin(), indices.begin()),
                             thrust::make_permutation_iterator(source.begin(), indices.begin()) + 4);

  ASSERT_EQ(result1, 19);

  T result2 = thrust::transform_reduce(
    thrust::make_permutation_iterator(source.begin(), indices.begin()),
    thrust::make_permutation_iterator(source.begin(), indices.begin()) + 4,
    thrust::negate<T>(),
    T(0),
    thrust::plus<T>());
  ASSERT_EQ(result2, -19);
};

TEST(PermutationIteratorTests, TestPermutationIteratorHostDeviceGather)
{
  using T              = int;
  using HostVector     = thrust::host_vector<T>;
  using DeviceVector   = thrust::host_vector<T>;
  using HostIterator   = HostVector::iterator;
  using DeviceIterator = DeviceVector::iterator;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  HostVector h_source(8);
  HostVector h_indices{3, 0, 5, 7};
  HostVector h_output(4, 10);

  DeviceVector d_source(8);
  DeviceVector d_indices(h_indices);
  DeviceVector d_output(4, 10);

  // initialize source
  thrust::sequence(h_source.begin(), h_source.end(), 1);
  thrust::sequence(d_source.begin(), d_source.end(), 1);

  thrust::permutation_iterator<HostIterator, HostIterator> p_h_source(h_source.begin(), h_indices.begin());
  thrust::permutation_iterator<DeviceIterator, DeviceIterator> p_d_source(d_source.begin(), d_indices.begin());

  // gather host->device
  thrust::copy(p_h_source, p_h_source + 4, d_output.begin());

  DeviceVector dref{4, 1, 6, 8};
  ASSERT_EQ(d_output, dref);

  // gather device->host
  thrust::copy(p_d_source, p_d_source + 4, h_output.begin());

  HostVector href{4, 1, 6, 8};
  ASSERT_EQ(h_output, href);
}

TEST(PermutationIteratorTests, TestPermutationIteratorHostDeviceScatter)
{
  using T              = int;
  using HostVector     = thrust::host_vector<T>;
  using DeviceVector   = thrust::host_vector<T>;
  using HostIterator   = HostVector::iterator;
  using DeviceIterator = DeviceVector::iterator;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  HostVector h_source(4, 10);
  HostVector h_indices{3, 0, 5, 7};
  HostVector h_output(8);

  DeviceVector d_source(4, 10);
  DeviceVector d_indices(h_indices);
  DeviceVector d_output(8);

  // initialize source
  thrust::sequence(h_output.begin(), h_output.end(), 1);
  thrust::sequence(d_output.begin(), d_output.end(), 1);

  thrust::permutation_iterator<HostIterator, HostIterator> p_h_output(h_output.begin(), h_indices.begin());
  thrust::permutation_iterator<DeviceIterator, DeviceIterator> p_d_output(d_output.begin(), d_indices.begin());

  // scatter host->device
  thrust::copy(h_source.begin(), h_source.end(), p_d_output);

  DeviceVector dref{10, 2, 3, 10, 5, 10, 7, 10};
  ASSERT_EQ(d_output, dref);

  // scatter device->host
  thrust::copy(d_source.begin(), d_source.end(), p_h_output);

  HostVector href(dref);
  ASSERT_EQ(h_output, href);
}

TYPED_TEST(PermutationIteratorTests, TestPermutationIteratorWithCountingIterator)
{
  using Vector = typename TestFixture::input_type;
  using T      = typename Vector::value_type;
  using diff_t = typename thrust::counting_iterator<T>::difference_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::counting_iterator<T> input(0), index(0);

  // test copy()
  {
    Vector output(4, 0);

    auto first = thrust::make_permutation_iterator(input, index);
    auto last  = thrust::make_permutation_iterator(input, index + static_cast<diff_t>(output.size()));

    thrust::copy(first, last, output.begin());

    Vector ref{0, 1, 2, 3};
    ASSERT_EQ(output, ref);
  }

  // test copy()
  {
    Vector output(4, 0);

    thrust::transform(thrust::make_permutation_iterator(input, index),
                      thrust::make_permutation_iterator(input, index + 4),
                      output.begin(),
                      ::internal::identity{});

    Vector ref{0, 1, 2, 3};
    ASSERT_EQ(output, ref);
  }
}
