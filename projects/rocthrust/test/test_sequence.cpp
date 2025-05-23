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

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/retag.h>
#include <thrust/sequence.h>

#include "test_header.hpp"

TESTS_DEFINE(SequenceTests, FullTestsParams);
TESTS_DEFINE(PrimitiveSequenceTests, NumericalTestsParams);

TEST(SequenceTests, UsingHip)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  ASSERT_EQ(THRUST_DEVICE_SYSTEM, THRUST_DEVICE_SYSTEM_HIP);
}

template <typename ForwardIterator>
void sequence(my_system& system, ForwardIterator, ForwardIterator)
{
  system.validate_dispatch();
}

TEST(SequenceTests, SequenceDispatchExplicit)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::device_vector<int> vec(1);

  my_system sys(0);
  thrust::sequence(sys, vec.begin(), vec.end());

  ASSERT_EQ(true, sys.is_valid());
}

template <typename ForwardIterator>
void sequence(my_tag, ForwardIterator first, ForwardIterator)
{
  *first = 13;
}

TEST(SequenceTests, SequenceDispatchImplicit)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::device_vector<int> vec(1);

  thrust::sequence(thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.end()));

  ASSERT_EQ(13, vec.front());
}

TYPED_TEST(SequenceTests, SequenceSimple)
{
  using Vector = typename TestFixture::input_type;
  using Policy = typename TestFixture::execution_policy;
  using T      = typename Vector::value_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  Vector v(5);

  thrust::sequence(Policy{}, v.begin(), v.end());

  ASSERT_EQ(v[0], 0);
  ASSERT_EQ(v[1], 1);
  ASSERT_EQ(v[2], 2);
  ASSERT_EQ(v[3], 3);
  ASSERT_EQ(v[4], 4);

  thrust::sequence(Policy{}, v.begin(), v.end(), (T) 10);

  ASSERT_EQ(v[0], 10);
  ASSERT_EQ(v[1], 11);
  ASSERT_EQ(v[2], 12);
  ASSERT_EQ(v[3], 13);
  ASSERT_EQ(v[4], 14);

  thrust::sequence(Policy{}, v.begin(), v.end(), (T) 10, (T) 2);

  ASSERT_EQ(v[0], 10);
  ASSERT_EQ(v[1], 12);
  ASSERT_EQ(v[2], 14);
  ASSERT_EQ(v[3], 16);
  ASSERT_EQ(v[4], 18);
}

TYPED_TEST(PrimitiveSequenceTests, SequencesWithVariableLength)
{
  using T = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  T error_margin = (T) 0.01;
  for (auto size : get_sizes())
  {
    size_t step_size = (size * 0.01) + 1;

    thrust::host_vector<T> h_data(size);
    thrust::device_vector<T> d_data(size);

    thrust::sequence(h_data.begin(), h_data.end());
    thrust::sequence(d_data.begin(), d_data.end());

    thrust::host_vector<T> h_data_d = d_data;
    for (size_t i = 0; i < size; i += step_size)
    {
      ASSERT_NEAR(h_data[i], h_data_d[i], error_margin);
    }

    thrust::sequence(h_data.begin(), h_data.end(), T(10));
    thrust::sequence(d_data.begin(), d_data.end(), T(10));

    h_data_d = d_data;
    for (size_t i = 0; i < size; i += step_size)
    {
      ASSERT_NEAR(h_data[i], h_data_d[i], error_margin);
    }

    thrust::sequence(h_data.begin(), h_data.end(), T(10), T(2));
    thrust::sequence(d_data.begin(), d_data.end(), T(10), T(2));

    h_data_d = d_data;
    for (size_t i = 0; i < size; i += step_size)
    {
      ASSERT_NEAR(h_data[i], h_data_d[i], error_margin);
    }

    thrust::sequence(h_data.begin(), h_data.end(), size_t(10), size_t(2));
    thrust::sequence(d_data.begin(), d_data.end(), size_t(10), size_t(2));

    h_data_d = d_data;
    for (size_t i = 0; i < size; i += step_size)
    {
      ASSERT_NEAR(h_data[i], h_data_d[i], error_margin);
    }
  }
}

TYPED_TEST(PrimitiveSequenceTests, SequenceToDiscardIterator)
{
  using T = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  for (auto size : get_sizes())
  {
    thrust::host_vector<T> h_data(size);
    thrust::device_vector<T> d_data(size);

    thrust::sequence(thrust::discard_iterator<thrust::device_system_tag>(),
                     thrust::discard_iterator<thrust::device_system_tag>(13),
                     T(10),
                     T(2));
  }
  // nothing to check -- just make sure it compiles
}

// A class that doesnt accept conversion from size_t but can be multiplied by a scalar
struct Vector
{
  Vector() = default;
  // Explicitly disable construction from size_t
  Vector(std::size_t) = delete;
  __host__ __device__ Vector(int x_, int y_)
      : x{x_}
      , y{y_}
  {}
  Vector(const Vector&)            = default;
  Vector& operator=(const Vector&) = default;

  int x, y;
};

// Vector-Vector addition
__host__ __device__ Vector operator+(const Vector a, const Vector b)
{
  return Vector{a.x + b.x, a.y + b.y};
}
// Vector-Scalar Multiplication
__host__ __device__ Vector operator*(const int a, const Vector b)
{
  return Vector{a * b.x, a * b.y};
}
__host__ __device__ Vector operator*(const Vector b, const int a)
{
  return Vector{a * b.x, a * b.y};
}

TEST(SequenceTests, TestSequenceNoSizeTConversion)
{
  thrust::device_vector<Vector> m(64);
  thrust::sequence(m.begin(), m.end(), ::Vector{0, 0}, ::Vector{1, 2});

  for (std::size_t i = 0; i < m.size(); ++i)
  {
    const ::Vector v = m[i];
    ASSERT_EQ(static_cast<std::size_t>(v.x), i);
    ASSERT_EQ(static_cast<std::size_t>(v.y), 2 * i);
  }
}
