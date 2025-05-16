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

#include <thrust/functional.h>
#include <thrust/transform.h>
#include <thrust/universal_vector.h>

#include <algorithm>
#include <cstddef>
#include <functional>

#include "test_param_fixtures.hpp"
#include "test_utils.hpp"

THRUST_DIAG_PUSH
THRUST_DIAG_SUPPRESS_MSVC(4244 4267) // possible loss of data

using AllTypesParams = ::testing::Types<
  Params<thrust::host_vector<int8_t>>,
  Params<thrust::host_vector<uint8_t>>,
  Params<thrust::host_vector<int16_t>>,
  Params<thrust::host_vector<uint16_t>>,
  Params<thrust::host_vector<int32_t>>,
  Params<thrust::host_vector<uint32_t>>,
  Params<thrust::host_vector<int64_t>>,
  Params<thrust::host_vector<uint64_t>>,
  Params<thrust::host_vector<float>>,
  Params<thrust::device_vector<int8_t>>,
  Params<thrust::device_vector<uint8_t>>,
  Params<thrust::device_vector<int16_t>>,
  Params<thrust::device_vector<uint16_t>>,
  Params<thrust::device_vector<int32_t>>,
  Params<thrust::device_vector<uint32_t>>,
  Params<thrust::device_vector<int64_t>>,
  Params<thrust::device_vector<uint64_t>>,
  Params<thrust::device_vector<float>>>;

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
  Params<thrust::device_vector<
    int,
    thrust::mr::stateless_resource_allocator<int, thrust::universal_host_pinned_memory_resource>>>>;

using IntegralVectorTestsParams =
  ::testing::Types<Params<thrust::host_vector<signed char>>,
                   Params<thrust::host_vector<short>>,
                   Params<thrust::host_vector<int>>,
                   Params<thrust::device_vector<signed char>>,
                   Params<thrust::device_vector<short>>,
                   Params<thrust::device_vector<int>>,
                   Params<thrust::universal_vector<int>>,
                   Params<thrust::device_vector<
                     int,
                     thrust::mr::stateless_resource_allocator<int, thrust::universal_host_pinned_memory_resource>>>>;

TESTS_DEFINE(AllTypesTests, AllTypesParams);
TESTS_DEFINE(VectorTests, VectorTestsParams);
TESTS_DEFINE(IntegralVectorTests, IntegralVectorTestsParams);

// There is a unfortunate miscompilation of the gcc-11 vectorizer leading to OOB writes
// Adding this attribute suffices that this miscompilation does not appear anymore
#if (THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_GCC) && __GNUC__ >= 11
#  define THRUST_DISABLE_BROKEN_GCC_VECTORIZER __attribute__((optimize("no-tree-vectorize")))
#else
#  define THRUST_DISABLE_BROKEN_GCC_VECTORIZER
#endif

const size_t NUM_SAMPLES = 10000;

template <class InputVector, class OutputVector, class Operator, class ReferenceOperator>
THRUST_DISABLE_BROKEN_GCC_VECTORIZER void TestUnaryFunctional()
{
  using InputType  = typename InputVector::value_type;
  using OutputType = typename OutputVector::value_type;

  thrust::host_vector<InputType> std_input = random_samples<InputType>(NUM_SAMPLES);
  thrust::host_vector<OutputType> std_output(NUM_SAMPLES);

  InputVector input = std_input;
  OutputVector output(NUM_SAMPLES);

  thrust::transform(input.begin(), input.end(), output.begin(), Operator());
  thrust::transform(std_input.begin(), std_input.end(), std_output.begin(), ReferenceOperator());

  ASSERT_EQ(output, std_output);
}

template <class InputVector, class OutputVector, class Operator, class ReferenceOperator>
THRUST_DISABLE_BROKEN_GCC_VECTORIZER void TestBinaryFunctional()
{
  using InputType  = typename InputVector::value_type;
  using OutputType = typename OutputVector::value_type;

  thrust::host_vector<InputType> std_input1 = random_samples<InputType>(NUM_SAMPLES);
  thrust::host_vector<InputType> std_input2 = random_samples<InputType>(NUM_SAMPLES);
  thrust::host_vector<OutputType> std_output(NUM_SAMPLES);

  // Replace zeros to avoid divide by zero exceptions
  std::replace(std_input2.begin(), std_input2.end(), (InputType) 0, (InputType) 1);

  InputVector input1 = std_input1;
  InputVector input2 = std_input2;
  OutputVector output(NUM_SAMPLES);

  thrust::transform(input1.begin(), input1.end(), input2.begin(), output.begin(), Operator());
  thrust::transform(std_input1.begin(), std_input1.end(), std_input2.begin(), std_output.begin(), ReferenceOperator());

  // Note: FP division is not bit-equal, even when nvcc is invoked with --prec-div
  ASSERT_EQ(output, std_output);
}

// op(T) -> T
#define DECLARE_UNARY_ARITHMETIC_FUNCTIONAL_UNITTEST(operator_name, OperatorName)                           \
  TYPED_TEST(AllTypesTests, Test##OperatorName##Functional)                                                 \
  {                                                                                                         \
    using Vector    = typename TestFixture::input_type;                                                     \
    using data_type = typename Vector::value_type;                                                          \
                                                                                                            \
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());                \
                                                                                                            \
    TestUnaryFunctional<Vector, Vector, thrust::operator_name<data_type>, std::operator_name<data_type>>(); \
  }

// op(T) -> bool
#define DECLARE_UNARY_LOGICAL_FUNCTIONAL_UNITTEST(operator_name, OperatorName)                              \
  TYPED_TEST(AllTypesTests, Test##OperatorName##Functional)                                                 \
  {                                                                                                         \
    using Vector    = typename TestFixture::input_type;                                                     \
    using data_type = typename Vector::value_type;                                                          \
                                                                                                            \
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());                \
                                                                                                            \
    TestUnaryFunctional<Vector, Vector, thrust::operator_name<data_type>, std::operator_name<data_type>>(); \
  }

// op(T,T) -> T
#define DECLARE_BINARY_ARITHMETIC_FUNCTIONAL_UNITTEST(operator_name, OperatorName)                           \
  TYPED_TEST(AllTypesTests, Test##OperatorName##Functional)                                                  \
  {                                                                                                          \
    using Vector    = typename TestFixture::input_type;                                                      \
    using data_type = typename Vector::value_type;                                                           \
                                                                                                             \
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());                 \
                                                                                                             \
    TestBinaryFunctional<Vector, Vector, thrust::operator_name<data_type>, std::operator_name<data_type>>(); \
  }

// op(T,T) -> T (for integer T only)
#define DECLARE_BINARY_INTEGER_ARITHMETIC_FUNCTIONAL_UNITTEST(operator_name, OperatorName)                   \
  TYPED_TEST(AllTypesTests, Test##OperatorName##Functional)                                                  \
  {                                                                                                          \
    using Vector    = typename TestFixture::input_type;                                                      \
    using data_type = typename Vector::value_type;                                                           \
                                                                                                             \
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());                 \
                                                                                                             \
    TestBinaryFunctional<Vector, Vector, thrust::operator_name<data_type>, std::operator_name<data_type>>(); \
  }

// op(T,T) -> bool
#define DECLARE_BINARY_LOGICAL_FUNCTIONAL_UNITTEST(operator_name, OperatorName)                              \
  TYPED_TEST(AllTypesTests, Test##OperatorName##Functional)                                                  \
  {                                                                                                          \
    using Vector    = typename TestFixture::input_type;                                                      \
    using data_type = typename Vector::value_type;                                                           \
                                                                                                             \
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());                 \
                                                                                                             \
    TestBinaryFunctional<Vector, Vector, thrust::operator_name<data_type>, std::operator_name<data_type>>(); \
  }

THRUST_DIAG_PUSH
THRUST_DIAG_SUPPRESS_MSVC(4146) // warning C4146: unary minus operator applied to unsigned type, result still
                                // unsigned

// Create the unit tests
DECLARE_UNARY_ARITHMETIC_FUNCTIONAL_UNITTEST(negate, Negate);
THRUST_DIAG_POP
DECLARE_UNARY_LOGICAL_FUNCTIONAL_UNITTEST(logical_not, LogicalNot);

// Ad-hoc testing for other functionals
TYPED_TEST(VectorTests, TestIdentityFunctional) THRUST_DISABLE_BROKEN_GCC_VECTORIZER
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  using Vector = typename TestFixture::input_type;
  using T      = typename Vector::value_type;

  Vector input(4);
  input[0] = 0;
  input[1] = 1;
  input[2] = 2;
  input[3] = 3;

  Vector output(4);

  thrust::transform(input.begin(), input.end(), output.begin(), thrust::identity<T>());

  ASSERT_EQ(input, output);
}

TYPED_TEST(VectorTests, TestProject1stFunctional) THRUST_DISABLE_BROKEN_GCC_VECTORIZER
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  using Vector = typename TestFixture::input_type;
  using T      = typename Vector::value_type;

  Vector lhs(4);
  Vector rhs(4);
  lhs[0] = 0;
  rhs[0] = 3;
  lhs[1] = 1;
  rhs[1] = 4;
  lhs[2] = 2;
  rhs[2] = 5;
  lhs[2] = 3;
  rhs[2] = 6;

  Vector output(4);

  thrust::transform(lhs.begin(), lhs.end(), rhs.begin(), output.begin(), thrust::project1st<T, T>());

  ASSERT_EQ(output, lhs);
}

TYPED_TEST(VectorTests, TestProject2ndFunctional) THRUST_DISABLE_BROKEN_GCC_VECTORIZER
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  using Vector = typename TestFixture::input_type;
  using T      = typename Vector::value_type;

  Vector lhs(4);
  Vector rhs(4);
  lhs[0] = 0;
  rhs[0] = 3;
  lhs[1] = 1;
  rhs[1] = 4;
  lhs[2] = 2;
  rhs[2] = 5;
  lhs[2] = 3;
  rhs[2] = 6;

  Vector output(4);

  thrust::transform(lhs.begin(), lhs.end(), rhs.begin(), output.begin(), thrust::project2nd<T, T>());

  ASSERT_EQ(output, rhs);
}

TYPED_TEST(VectorTests, TestMaximumFunctional) THRUST_DISABLE_BROKEN_GCC_VECTORIZER
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  using Vector = typename TestFixture::input_type;
  using T      = typename Vector::value_type;

  Vector input1(4);
  Vector input2(4);
  input1[0] = 8;
  input1[1] = 3;
  input1[2] = 7;
  input1[3] = 7;
  input2[0] = 5;
  input2[1] = 6;
  input2[2] = 9;
  input2[3] = 3;

  Vector output(4);

  thrust::transform(input1.begin(), input1.end(), input2.begin(), output.begin(), thrust::maximum<T>());

  ASSERT_EQ(output[0], 8);
  ASSERT_EQ(output[1], 6);
  ASSERT_EQ(output[2], 9);
  ASSERT_EQ(output[3], 7);
}

TYPED_TEST(VectorTests, TestMinimumFunctional) THRUST_DISABLE_BROKEN_GCC_VECTORIZER
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  using Vector = typename TestFixture::input_type;
  using T      = typename Vector::value_type;

  Vector input1(4);
  Vector input2(4);
  input1[0] = 8;
  input1[1] = 3;
  input1[2] = 7;
  input1[3] = 7;
  input2[0] = 5;
  input2[1] = 6;
  input2[2] = 9;
  input2[3] = 3;

  Vector output(4);

  thrust::transform(input1.begin(), input1.end(), input2.begin(), output.begin(), thrust::minimum<T>());

  ASSERT_EQ(output[0], 5);
  ASSERT_EQ(output[1], 3);
  ASSERT_EQ(output[2], 7);
  ASSERT_EQ(output[3], 3);
}

TYPED_TEST(IntegralVectorTests, TestNot1) THRUST_DISABLE_BROKEN_GCC_VECTORIZER
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  using Vector = typename TestFixture::input_type;
  using T      = typename Vector::value_type;

  Vector input(5);
  input[0] = 1;
  input[1] = 0;
  input[2] = 1;
  input[3] = 1;
  input[4] = 0;

  Vector output(5);

  thrust::transform(input.begin(), input.end(), output.begin(), thrust::not_fn(thrust::identity<T>()));

  ASSERT_EQ(output[0], 0);
  ASSERT_EQ(output[1], 1);
  ASSERT_EQ(output[2], 0);
  ASSERT_EQ(output[3], 0);
  ASSERT_EQ(output[4], 1);
}

// GCC 11 fails to build this test case with a spurious error in a
// very specific scenario:
// - GCC 11
// - CPP system for both host and device
// - C++11 dialect
#if !(defined(THRUST_GCC_VERSION) && THRUST_GCC_VERSION >= 110000 && THRUST_GCC_VERSION < 120000          \
      && THRUST_HOST_SYSTEM == THRUST_HOST_SYSTEM_CPP && THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CPP \
      && THRUST_CPP_DIALECT == 2011)

TYPED_TEST(VectorTests, TestNot2) THRUST_DISABLE_BROKEN_GCC_VECTORIZER
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  using Vector = typename TestFixture::input_type;
  using T      = typename Vector::value_type;

  Vector input1(5);
  Vector input2(5);
  input1[0] = 1;
  input1[1] = 0;
  input1[2] = 1;
  input1[3] = 1;
  input1[4] = 0;
  input2[0] = 1;
  input2[1] = 1;
  input2[2] = 0;
  input2[3] = 1;
  input2[4] = 1;

  Vector output(5);

  thrust::transform(input1.begin(), input1.end(), input2.begin(), output.begin(), thrust::not_fn(thrust::equal_to<T>()));

  ASSERT_EQ(output[0], 0);
  ASSERT_EQ(output[1], 1);
  ASSERT_EQ(output[2], 1);
  ASSERT_EQ(output[3], 0);
  ASSERT_EQ(output[4], 1);
}

#endif // Weird GCC11 failure case

THRUST_DIAG_POP
