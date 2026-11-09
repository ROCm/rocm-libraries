/*
 *  Copyright 2026 NVIDIA Corporation
 *  Modifications Copyright© 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/retag.h>
#include <thrust/reduce.h>
#include <thrust/universal_vector.h>

#include <limits>

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

TESTS_DEFINE(ReduceIntoTests, FullTestsParams);
TESTS_DEFINE(ReduceIntoIntegerTests, UnsignedIntegerTestsParams);
TESTS_DEFINE(ReduceIntoPrimitiveTests, NumericalTestsParams);
TESTS_DEFINE(ReduceIntoVectorUnitTests, VectorTestsParams);

template <typename T>
struct plus_mod_10
{
  THRUST_HOST_DEVICE T operator()(T lhs, T rhs) const
  {
    return ((lhs % 10) + (rhs % 10)) % 10;
  }
};

TYPED_TEST(ReduceIntoVectorUnitTests, TestReduceIntoSimple)
{
  using Vector = typename TestFixture::input_type;
  using T      = typename Vector::value_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  Vector i{1, -2, 3};
  Vector o(1);

  // no initializer
  thrust::reduce_into(i.begin(), i.end(), o.begin());
  EXPECT_EQ(o[0], 2);

  // with initializer
  thrust::reduce_into(i.begin(), i.end(), o.begin(), T(10));
  EXPECT_EQ(o[0], 12);
}

template <typename InputIterator, typename OutputIterator>
void reduce_into(my_system& system, InputIterator, InputIterator, OutputIterator output)
{
  system.validate_dispatch();
  *output = 13;
}

TEST(ReduceIntoTests, TestReduceIntoDispatchExplicit)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::device_vector<int> i;
  thrust::device_vector<int> o(1);

  my_system sys(0);
  thrust::reduce_into(sys, i.begin(), i.end(), o.begin());

  EXPECT_TRUE(sys.is_valid());
  EXPECT_EQ(o[0], 13);
}

template <typename InputIterator, typename OutputIterator>
void reduce_into(my_tag, InputIterator, InputIterator, OutputIterator output)
{
  *output = 13;
}

TEST(ReduceIntoTests, TestReduceIntoDispatchImplicit)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::device_vector<int> i;
  thrust::device_vector<int> o(1);

  thrust::reduce_into(
    thrust::retag<my_tag>(i.begin()), thrust::retag<my_tag>(i.end()), thrust::retag<my_tag>(o.begin()));

  EXPECT_EQ(o[0], 13);
}

TYPED_TEST(ReduceIntoPrimitiveTests, TestReduceInto)
{
  using T = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  for (auto size : get_sizes())
  {
    SCOPED_TRACE(testing::Message() << "with size= " << size);

    for (auto seed : get_seeds())
    {
      SCOPED_TRACE(testing::Message() << "with seed= " << seed);

      thrust::host_vector<T> h_data =
        get_random_data<T>(size, get_default_limits<T>::min(), get_default_limits<T>::max(), seed);
      thrust::device_vector<T> d_data = h_data;
      thrust::host_vector<T> h_result(1);
      thrust::device_vector<T> d_result(1);

      T init = 13;

      thrust::reduce_into(h_data.begin(), h_data.end(), h_result.begin(), init);
      thrust::reduce_into(d_data.begin(), d_data.end(), d_result.begin(), init);

      test_equality(h_result, d_result, size - 1);
    }
  }
}

TYPED_TEST(ReduceIntoTests, TestReduceIntoMixedTypes)
{
  using Vector = typename TestFixture::input_type;
  using T      = typename Vector::value_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  // make sure we get types for default args and operators correct
  if constexpr (std::is_floating_point<T>::value)
  {
    Vector int_input{1, 2, 3, 4};

    Vector float_input{1.5, 2.5, 3.5, 4.5};

    // float -> int should use using plus<int> operator by default
    thrust::host_vector<int> int_output(1);
    thrust::reduce_into(float_input.begin(), float_input.end(), int_output.begin(), int(0));
    EXPECT_EQ(int_output[0], 10);

    // int -> float should use using plus<float> operator by default
    thrust::host_vector<float> float_output(1);
    thrust::reduce_into(int_input.begin(), int_input.end(), float_output.begin(), float(0.5));
    EXPECT_EQ(float_output[0], 10.5);
  }
}

TYPED_TEST(ReduceIntoIntegerTests, TestReduceIntoWithOperator)
{
  using T = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  for (auto size : get_sizes())
  {
    SCOPED_TRACE(testing::Message() << "with size= " << size);

    for (auto seed : get_seeds())
    {
      SCOPED_TRACE(testing::Message() << "with seed= " << seed);

      thrust::host_vector<T> h_data =
        get_random_data<T>(size, get_default_limits<T>::min(), get_default_limits<T>::max(), seed);
      thrust::device_vector<T> d_data = h_data;
      thrust::host_vector<T> h_result(1);
      thrust::device_vector<T> d_result(1);

      T init = 3;

      thrust::reduce_into(h_data.begin(), h_data.end(), h_result.begin(), init, plus_mod_10<T>());
      thrust::reduce_into(d_data.begin(), d_data.end(), d_result.begin(), init, plus_mod_10<T>());

      ASSERT_EQ(h_result, d_result);
    }
  }
}

template <typename T>
struct plus_mod3
{
  T* table;

  plus_mod3(T* table)
      : table(table)
  {}

  THRUST_HOST_DEVICE T operator()(T a, T b)
  {
    return table[(int) (a + b)];
  }
};

TYPED_TEST(ReduceIntoTests, TestReduceIntoWithIndirection)
{
  // add numbers modulo 3 with external lookup table
  using Vector = typename TestFixture::input_type;
  using T      = typename Vector::value_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  Vector data{0, 1, 2, 1, 2, 0, 1};

  Vector table{0, 1, 2, 0, 1, 2};

  Vector result(1);

  thrust::reduce_into(data.begin(), data.end(), result.begin(), T(0), plus_mod3<T>(thrust::raw_pointer_cast(&table[0])));

  ASSERT_EQ(result[0], T(1));
}

TYPED_TEST(ReduceIntoPrimitiveTests, TestReduceIntoCountingIterator)
{
  using T = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  size_t const n = 15 * sizeof(T);

  ASSERT_LE(T(n), truncate_to_max_representable<T>(n));

  thrust::counting_iterator<T, thrust::host_system_tag> h_first   = thrust::make_counting_iterator<T>(0);
  thrust::counting_iterator<T, thrust::device_system_tag> d_first = thrust::make_counting_iterator<T>(0);
  thrust::host_vector<T> h_result(1);
  thrust::device_vector<T> d_result(1);

  T init = random_integer<T>();

  thrust::reduce_into(h_first, h_first + n, h_result.begin(), init);
  thrust::reduce_into(d_first, d_first + n, d_result.begin(), init);

  // we use ASSERT_NEAR because we're testing floating point types
  if (std::is_floating_point<T>::value)
  {
    ASSERT_NEAR(h_result[0], d_result[0], h_result[0] * 0.01);
  }
  else
  {
    ASSERT_EQ(h_result[0], d_result[0]);
  }
}
