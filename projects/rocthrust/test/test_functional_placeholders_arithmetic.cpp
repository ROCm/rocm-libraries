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
#include <thrust/iterator/constant_iterator.h>
#include <thrust/transform.h>
#include <thrust/universal_vector.h>

#include "test_param_fixtures.hpp"
#include "test_utils.hpp"

THRUST_DIAG_PUSH
THRUST_DIAG_SUPPRESS_MSVC(4244) // warning C4244: '=': conversion from 'int' to '_Ty', possible loss of data

using ThirtyTwoBitTypesParams =
  ::testing::Types<Params<thrust::host_vector<int>>,
                   Params<thrust::host_vector<unsigned int>>,
                   Params<thrust::host_vector<float>>,
                   Params<thrust::device_vector<int>>,
                   Params<thrust::device_vector<unsigned int>>,
                   Params<thrust::device_vector<float>>>;

using SmallIntegralTypesParams =
  ::testing::Types<Params<thrust::host_vector<char>>,
                   Params<thrust::host_vector<signed char>>,
                   Params<thrust::host_vector<unsigned char>>,
                   Params<thrust::host_vector<short>>,
                   Params<thrust::host_vector<unsigned short>>,
                   Params<thrust::device_vector<char>>,
                   Params<thrust::device_vector<signed char>>,
                   Params<thrust::device_vector<unsigned char>>,
                   Params<thrust::device_vector<short>>,
                   Params<thrust::device_vector<unsigned short>>>;

using UnaryTestsParams = ::testing::Types<
  Params<thrust::host_vector<signed char>>,
  Params<thrust::host_vector<short>>,
  Params<thrust::host_vector<int>>,
  Params<thrust::device_vector<char>>,
  Params<thrust::device_vector<signed char>>,
  Params<thrust::device_vector<short>>,
  Params<thrust::device_vector<int>>,
  Params<thrust::universal_vector<int>>,
  Params<thrust::device_vector<
    int,
    thrust::mr::stateless_resource_allocator<int, thrust::universal_host_pinned_memory_resource>>>>;

TESTS_DEFINE(BinaryArithmeticTests, ThirtyTwoBitTypesParams);
TESTS_DEFINE(ModulusTests, SmallIntegralTypesParams);
TESTS_DEFINE(UnaryTests, UnaryTestsParams);

#define BINARY_FUNCTIONAL_PLACEHOLDERS_TEST(name, op, reference_functor, test_type)                             \
  TYPED_TEST(test_type, TestFunctionalPlaceholders##name)                                                       \
  {                                                                                                             \
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());                    \
                                                                                                                \
    static const size_t num_samples = 10000;                                                                    \
    const size_t zero               = 0;                                                                        \
    using Vector                    = typename TestFixture::input_type;                                         \
    using T                         = typename Vector::value_type;                                              \
                                                                                                                \
    Vector lhs = random_samples<T>(num_samples);                                                                \
    Vector rhs = random_samples<T>(num_samples);                                                                \
    thrust::replace(rhs.begin(), rhs.end(), T(0), T(1));                                                        \
                                                                                                                \
    Vector reference(lhs.size());                                                                               \
    Vector result(lhs.size());                                                                                  \
    using namespace thrust::placeholders;                                                                       \
                                                                                                                \
    thrust::transform(lhs.begin(), lhs.end(), rhs.begin(), reference.begin(), reference_functor<T>());          \
    thrust::transform(lhs.begin(), lhs.end(), rhs.begin(), result.begin(), _1 op _2);                           \
    ASSERT_EQ(reference, result);                                                                               \
                                                                                                                \
    thrust::transform(                                                                                          \
      lhs.begin(), lhs.end(), thrust::make_constant_iterator<T>(1), reference.begin(), reference_functor<T>()); \
    thrust::transform(lhs.begin(), lhs.end(), result.begin(), _1 op T(1));                                      \
    ASSERT_EQ(reference, result);                                                                               \
                                                                                                                \
    thrust::transform(                                                                                          \
      thrust::make_constant_iterator<T>(1, zero),                                                               \
      thrust::make_constant_iterator<T>(1, num_samples),                                                        \
      rhs.begin(),                                                                                              \
      reference.begin(),                                                                                        \
      reference_functor<T>());                                                                                  \
    thrust::transform(rhs.begin(), rhs.end(), result.begin(), T(1) op _1);                                      \
    ASSERT_EQ(reference, result);                                                                               \
  }

BINARY_FUNCTIONAL_PLACEHOLDERS_TEST(Plus, +, thrust::plus, BinaryArithmeticTests);
BINARY_FUNCTIONAL_PLACEHOLDERS_TEST(Minus, -, thrust::minus, BinaryArithmeticTests);
BINARY_FUNCTIONAL_PLACEHOLDERS_TEST(Multiplies, *, thrust::multiplies, BinaryArithmeticTests);
BINARY_FUNCTIONAL_PLACEHOLDERS_TEST(Divides, /, thrust::divides, BinaryArithmeticTests);
BINARY_FUNCTIONAL_PLACEHOLDERS_TEST(Modulus, %, thrust::modulus, ModulusTests);

#define UNARY_FUNCTIONAL_PLACEHOLDERS_TEST(name, reference_operator, functor)                \
  TYPED_TEST(UnaryTests, TestFunctionalPlaceholders##name)                                   \
  {                                                                                          \
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest()); \
                                                                                             \
    static const size_t num_samples = 10000;                                                 \
    using Vector                    = typename TestFixture::input_type;                      \
    using T                         = typename Vector::value_type;                           \
                                                                                             \
    Vector input = random_samples<T>(num_samples);                                           \
                                                                                             \
    Vector reference(input.size());                                                          \
    thrust::transform(input.begin(), input.end(), reference.begin(), functor<T>());          \
                                                                                             \
    using namespace thrust::placeholders;                                                    \
    Vector result(input.size());                                                             \
    thrust::transform(input.begin(), input.end(), result.begin(), reference_operator _1);    \
                                                                                             \
    ASSERT_EQ(reference, result);                                                            \
  }

template <typename T>
struct unary_plus_reference
{
  THRUST_HOST_DEVICE T operator()(const T& x) const
  { // Static cast to undo integral promotion
    return static_cast<T>(+x);
  }
};

UNARY_FUNCTIONAL_PLACEHOLDERS_TEST(UnaryPlus, +, unary_plus_reference);
UNARY_FUNCTIONAL_PLACEHOLDERS_TEST(Negate, -, thrust::negate);

THRUST_DIAG_POP
