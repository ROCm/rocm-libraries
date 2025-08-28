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

#include <thrust/generate.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/retag.h>

#include "test_param_fixtures.hpp"
#include "test_real_assertions.hpp"
#include "test_utils.hpp"

using VectorParams = ::testing::Types<Params<thrust::host_vector<short>>, Params<thrust::host_vector<int>>>;

TESTS_DEFINE(GenerateTests, FullTestsParams);
TESTS_DEFINE(GenerateVectorTests, VectorParams);
TESTS_DEFINE(GenerateVariablesTests, NumericalTestsParams);

TEST(ReplaceTests, UsingHip)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  ASSERT_EQ(THRUST_DEVICE_SYSTEM, THRUST_DEVICE_SYSTEM_HIP);
}

THRUST_DIAG_PUSH
THRUST_DIAG_SUPPRESS_MSVC(4244 4267) // possible loss of data

template <typename T>
struct return_value
{
  T val;

  return_value() {}
  return_value(T v)
      : val(v)
  {}

  THRUST_HOST_DEVICE T operator()(void)
  {
    return val;
  }
};

TYPED_TEST(GenerateVectorTests, TestGenerateSimple)
{
  using Vector = typename TestFixture::input_type;
  using T      = typename Vector::value_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  Vector result(5);

  T value = 13;

  return_value<T> f(value);

  thrust::generate(result.begin(), result.end(), f);

  Vector ref(result.size(), value);
  ASSERT_EQ(result, ref);
}

template <typename ForwardIterator, typename Generator>
void generate(my_system& system, ForwardIterator /*first*/, ForwardIterator, Generator)
{
  system.validate_dispatch();
}

TEST(GenerateTests, TestGenerateDispatchExplicit)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::device_vector<int> vec(1);

  my_system sys(0);
  thrust::generate(sys, vec.begin(), vec.end(), 0);

  ASSERT_EQ(true, sys.is_valid());
}

template <typename ForwardIterator, typename Generator>
void generate(my_tag, ForwardIterator first, ForwardIterator, Generator)
{
  *first = 13;
}

TEST(GenerateTests, TestGenerateDispatchImplicit)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::device_vector<int> vec(1);

  thrust::generate(thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.end()), 0);

  ASSERT_EQ(13, vec.front());
}

TYPED_TEST(GenerateVariablesTests, TestGenerate)
{
  using T = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  for (auto size : get_sizes())
  {
    SCOPED_TRACE(testing::Message() << "with size= " << size);

    thrust::host_vector<T> h_result(size);
    thrust::device_vector<T> d_result(size);

    T value = 13;
    return_value<T> f(value);

    thrust::generate(h_result.begin(), h_result.end(), f);
    thrust::generate(d_result.begin(), d_result.end(), f);

    ASSERT_EQ(h_result, d_result);
  }
}

TYPED_TEST(GenerateVariablesTests, TestGenerateToDiscardIterator)
{
  using T = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  T value = 13;
  return_value<T> f(value);

  thrust::discard_iterator<thrust::host_system_tag> h_first;
  thrust::generate(h_first, h_first + 10, f);

  thrust::discard_iterator<thrust::device_system_tag> d_first;
  thrust::generate(d_first, d_first + 10, f);

  // there's nothing to actually check except that it compiles
}

TYPED_TEST(GenerateVectorTests, TestGenerateNSimple)
{
  using Vector = typename TestFixture::input_type;
  using T      = typename Vector::value_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  Vector result(5);

  T value = 13;

  return_value<T> f(value);

  thrust::generate_n(result.begin(), result.size(), f);

  Vector ref(result.size(), value);
  ASSERT_EQ(result, ref);
}

template <typename ForwardIterator, typename Size, typename Generator>
ForwardIterator generate_n(my_system& system, ForwardIterator first, Size, Generator)
{
  system.validate_dispatch();
  return first;
}

TEST(GenerateTests, TestGenerateNDispatchExplicit)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::device_vector<int> vec(1);

  my_system sys(0);
  thrust::generate_n(sys, vec.begin(), vec.size(), 0);

  ASSERT_EQ(true, sys.is_valid());
}

template <typename ForwardIterator, typename Size, typename Generator>
ForwardIterator generate_n(my_tag, ForwardIterator first, Size, Generator)
{
  *first = 13;
  return first;
}

TEST(GenerateTests, TestGenerateNDispatchImplicit)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::device_vector<int> vec(1);

  thrust::generate_n(thrust::retag<my_tag>(vec.begin()), vec.size(), 0);

  ASSERT_EQ(13, vec.front());
}

TYPED_TEST(GenerateVariablesTests, TestGenerateNToDiscardIterator)
{
  using T = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  for (auto size : get_sizes())
  {
    SCOPED_TRACE(testing::Message() << "with size= " << size);

    T value = 13;
    return_value<T> f(value);

    thrust::discard_iterator<thrust::host_system_tag> h_result =
      thrust::generate_n(thrust::discard_iterator<thrust::host_system_tag>(), size, f);

    thrust::discard_iterator<thrust::device_system_tag> d_result =
      thrust::generate_n(thrust::discard_iterator<thrust::device_system_tag>(), size, f);

    thrust::discard_iterator<> reference(size);

    ASSERT_EQ_QUIET(reference, h_result);
    ASSERT_EQ_QUIET(reference, d_result);
  }
}

TYPED_TEST(GenerateVectorTests, TestGenerateZipIterator)
{
  using Vector = typename TestFixture::input_type;
  using T      = typename Vector::value_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  Vector v1(3, T(0));
  Vector v2(3, T(0));

  thrust::generate(thrust::make_zip_iterator(thrust::make_tuple(v1.begin(), v2.begin())),
                   thrust::make_zip_iterator(thrust::make_tuple(v1.end(), v2.end())),
                   return_value<thrust::tuple<T, T>>(thrust::tuple<T, T>(4, 7)));

  Vector ref1(3, 4);
  Vector ref2(3, 7);
  ASSERT_EQ(v1, ref1);
  ASSERT_EQ(v2, ref2);
}

TEST(GenerateTests, TestGenerateTuple)
{
  using T     = int;
  using Tuple = thrust::tuple<T, T>;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::host_vector<Tuple> h(3, Tuple(0, 0));
  thrust::device_vector<Tuple> d(3, Tuple(0, 0));

  thrust::generate(h.begin(), h.end(), return_value<Tuple>(Tuple(4, 7)));
  thrust::generate(d.begin(), d.end(), return_value<Tuple>(Tuple(4, 7)));

  ASSERT_EQ_QUIET(h, d);
}

THRUST_DIAG_POP
