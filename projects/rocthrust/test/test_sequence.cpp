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

#include <thrust/complex.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/retag.h>
#include <thrust/sequence.h>

#include "test_param_fixtures.hpp"
#include "test_utils.hpp"

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

TEST(SequenceTests, TestSequenceDispatchExplicit)
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

TEST(SequenceTests, TestSequenceDispatchImplicit)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::device_vector<int> vec(1);

  thrust::sequence(thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.end()));

  ASSERT_EQ(13, vec.front());
}

TYPED_TEST(SequenceTests, TestSequenceSimple)
{
  using Vector     = typename TestFixture::input_type;
  using value_type = typename Vector::value_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  Vector v(5);

  thrust::sequence(v.begin(), v.end());

  Vector ref{0, 1, 2, 3, 4};
  ASSERT_EQ(v, ref);

  thrust::sequence(v.begin(), v.end(), value_type{10});

  ref = {10, 11, 12, 13, 14};
  ASSERT_EQ(v, ref);

  thrust::sequence(v.begin(), v.end(), value_type{10}, value_type{2});

  ref = {10, 12, 14, 16, 18};
  ASSERT_EQ(v, ref);
}

TYPED_TEST(PrimitiveSequenceTests, TestSequence)
{
  using T = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  for (auto size : get_sizes())
  {
    SCOPED_TRACE(testing::Message() << "with size= " << size);

    thrust::host_vector<T> h_data(size);
    thrust::device_vector<T> d_data(size);

    thrust::sequence(h_data.begin(), h_data.end());
    thrust::sequence(d_data.begin(), d_data.end());

    ASSERT_EQ(h_data, d_data);

    thrust::sequence(h_data.begin(), h_data.end(), T(10));
    thrust::sequence(d_data.begin(), d_data.end(), T(10));

    ASSERT_EQ(h_data, d_data);

    thrust::sequence(h_data.begin(), h_data.end(), T(10), T(2));
    thrust::sequence(d_data.begin(), d_data.end(), T(10), T(2));

    ASSERT_EQ(h_data, d_data);

    thrust::sequence(h_data.begin(), h_data.end(), T(10), T(2));
    thrust::sequence(d_data.begin(), d_data.end(), T(10), T(2));

    ASSERT_EQ(h_data, d_data);
  }
}

TYPED_TEST(PrimitiveSequenceTests, TestSequenceToDiscardIterator)
{
  using T = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  for (auto size : get_sizes())
  {
    SCOPED_TRACE(testing::Message() << "with size= " << size);

    thrust::host_vector<T> h_data(size);
    thrust::device_vector<T> d_data(size);

    thrust::sequence(thrust::discard_iterator<thrust::device_system_tag>(),
                     thrust::discard_iterator<thrust::device_system_tag>(13),
                     T(10),
                     T(2));
  }
  // nothing to check -- just make sure it compiles
}

TEST(SequenceTests, TestSequenceComplex)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::device_vector<thrust::complex<double>> m(64);
  thrust::sequence(m.begin(), m.end());
}

// A class that does not accept conversion from size_t but can be multiplied by a scalar
struct Vector
{
  Vector() = default;
  // Explicitly disable construction from size_t
  Vector(std::size_t) = delete;
  THRUST_HOST_DEVICE Vector(int x_, int y_)
      : x{x_}
      , y{y_}
  {}
  Vector(const Vector&)            = default;
  Vector& operator=(const Vector&) = default;

  int x, y;
};

// Vector-Vector addition
THRUST_HOST_DEVICE Vector operator+(const Vector a, const Vector b)
{
  return Vector{a.x + b.x, a.y + b.y};
}

// Vector-Scalar Multiplication
// Multiplication by std::size_t is required by thrust::sequence.
THRUST_HOST_DEVICE Vector operator*(const std::size_t a, const Vector b)
{
  return Vector{static_cast<int>(a) * b.x, static_cast<int>(a) * b.y};
}
THRUST_HOST_DEVICE Vector operator*(const Vector b, const std::size_t a)
{
  return Vector{static_cast<int>(a) * b.x, static_cast<int>(a) * b.y};
}

TEST(SequenceTests, TestSequenceNoSizeTConversion)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::device_vector<Vector> m(64);
  thrust::sequence(m.begin(), m.end(), ::Vector{0, 0}, ::Vector{1, 2});

  for (std::size_t i = 0; i < m.size(); ++i)
  {
    const ::Vector v = m[i];
    ASSERT_EQ(static_cast<std::size_t>(v.x), i);
    ASSERT_EQ(static_cast<std::size_t>(v.y), 2 * i);
  }
}
