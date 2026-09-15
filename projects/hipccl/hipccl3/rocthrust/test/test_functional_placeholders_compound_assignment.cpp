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

using IntegralVectorTestsParams =
  ::testing::Types<Params<thrust::host_vector<signed char>>,
                   Params<thrust::host_vector<short>>,
                   Params<thrust::host_vector<int>>,
                   Params<thrust::device_vector<signed char>>,
                   Params<thrust::device_vector<short>>,
                   Params<thrust::device_vector<int>>,
                   Params<thrust::universal_vector<int>>,
                   Params<thrust::universal_host_pinned_vector<int>>>;

TESTS_DEFINE(ThirtyTwoBitTypesTests, ThirtyTwoBitTypesParams);
TESTS_DEFINE(SmallIntegralTypesTests, SmallIntegralTypesParams);
TESTS_DEFINE(IntegralVectorTests, IntegralVectorTestsParams);

#define BINARY_FUNCTIONAL_PLACEHOLDERS_TEST(name, op, reference_functor, test_type)                        \
  TYPED_TEST(test_type, TestFunctionalPlaceholders##name)                                                  \
  {                                                                                                        \
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());               \
                                                                                                           \
    const size_t num_samples = 10000;                                                                      \
    using Vector             = typename TestFixture::input_type;                                           \
    using T                  = typename Vector::value_type;                                                \
    Vector lhs               = random_samples<T>(num_samples);                                             \
    Vector rhs               = random_samples<T>(num_samples);                                             \
    thrust::replace(rhs.begin(), rhs.end(), T(0), T(1));                                                   \
                                                                                                           \
    Vector lhs_reference = lhs;                                                                            \
    Vector reference(lhs.size());                                                                          \
    Vector result(lhs_reference.size());                                                                   \
    using namespace thrust::placeholders;                                                                  \
                                                                                                           \
    thrust::transform(                                                                                     \
      lhs_reference.begin(), lhs_reference.end(), rhs.begin(), reference.begin(), reference_functor<T>()); \
    thrust::transform(lhs.begin(), lhs.end(), rhs.begin(), result.begin(), _1 op _2);                      \
    ASSERT_EQ(reference, result);                                                                          \
    ASSERT_EQ(lhs_reference, lhs);                                                                         \
                                                                                                           \
    thrust::transform(                                                                                     \
      lhs_reference.begin(),                                                                               \
      lhs_reference.end(),                                                                                 \
      thrust::make_constant_iterator<T>(1),                                                                \
      reference.begin(),                                                                                   \
      reference_functor<T>());                                                                             \
    thrust::transform(lhs.begin(), lhs.end(), rhs.begin(), result.begin(), _1 op T(1));                    \
    ASSERT_EQ(reference, result);                                                                          \
    ASSERT_EQ(lhs_reference, lhs);                                                                         \
  }

template <typename T>
struct plus_equal_reference
{
  THRUST_HOST_DEVICE T& operator()(T& lhs, const T& rhs) const
  {
    return lhs += rhs;
  }
};

template <typename T>
struct minus_equal_reference
{
  THRUST_HOST_DEVICE T& operator()(T& lhs, const T& rhs) const
  {
    return lhs -= rhs;
  }
};

template <typename T>
struct multiplies_equal_reference
{
  THRUST_HOST_DEVICE T& operator()(T& lhs, const T& rhs) const
  {
    return lhs *= rhs;
  }
};

template <typename T>
struct divides_equal_reference
{
  THRUST_HOST_DEVICE T& operator()(T& lhs, const T& rhs) const
  {
    return lhs /= rhs;
  }
};

template <typename T>
struct modulus_equal_reference
{
  THRUST_HOST_DEVICE T& operator()(T& lhs, const T& rhs) const
  {
    return lhs %= rhs;
  }
};

BINARY_FUNCTIONAL_PLACEHOLDERS_TEST(PlusEqual, +=, plus_equal_reference, ThirtyTwoBitTypesTests);
BINARY_FUNCTIONAL_PLACEHOLDERS_TEST(MinusEqual, -=, minus_equal_reference, ThirtyTwoBitTypesTests);
BINARY_FUNCTIONAL_PLACEHOLDERS_TEST(MultipliesEqual, *=, multiplies_equal_reference, ThirtyTwoBitTypesTests);
BINARY_FUNCTIONAL_PLACEHOLDERS_TEST(DividesEqual, /=, divides_equal_reference, ThirtyTwoBitTypesTests);
BINARY_FUNCTIONAL_PLACEHOLDERS_TEST(ModulusEqual, %=, modulus_equal_reference, SmallIntegralTypesTests);

template <typename T>
struct bit_and_equal_reference
{
  THRUST_HOST_DEVICE T& operator()(T& lhs, const T& rhs) const
  {
    return lhs &= rhs;
  }
};

template <typename T>
struct bit_or_equal_reference
{
  THRUST_HOST_DEVICE T& operator()(T& lhs, const T& rhs) const
  {
    return lhs |= rhs;
  }
};

template <typename T>
struct bit_xor_equal_reference
{
  THRUST_HOST_DEVICE T& operator()(T& lhs, const T& rhs) const
  {
    return lhs ^= rhs;
  }
};

template <typename T>
struct bit_lshift_equal_reference
{
  THRUST_HOST_DEVICE T& operator()(T& lhs, const T& rhs) const
  {
    return lhs <<= rhs;
  }
};

template <typename T>
struct bit_rshift_equal_reference
{
  THRUST_HOST_DEVICE T& operator()(T& lhs, const T& rhs) const
  {
    return lhs >>= rhs;
  }
};

BINARY_FUNCTIONAL_PLACEHOLDERS_TEST(BitAndEqual, &=, bit_and_equal_reference, SmallIntegralTypesTests);
BINARY_FUNCTIONAL_PLACEHOLDERS_TEST(BitOrEqual, |=, bit_or_equal_reference, SmallIntegralTypesTests);
BINARY_FUNCTIONAL_PLACEHOLDERS_TEST(BitXorEqual, ^=, bit_xor_equal_reference, SmallIntegralTypesTests);

// XXX ptxas produces an error
void TestFunctionalPlaceholdersBitLshiftEqualDevice()
{
  GTEST_NONFATAL_FAILURE_("Known failure");
}
// XXX KNOWN_FAILURE this until the above works
void TestFunctionalPlaceholdersBitLshiftEqualHost()
{
  GTEST_NONFATAL_FAILURE_("Known failure");
}
// BINARY_FUNCTIONAL_PLACEHOLDERS_TEST(BitLshiftEqual, <<=, bit_lshift_equal_reference, SmallIntegralTypesTests);

BINARY_FUNCTIONAL_PLACEHOLDERS_TEST(BitRshiftEqual, >>=, bit_rshift_equal_reference, SmallIntegralTypesTests);

template <typename T>
struct prefix_increment_reference
{
  THRUST_HOST_DEVICE T& operator()(T& x) const
  {
    return ++x;
  }
};

template <typename T>
struct suffix_increment_reference
{
  THRUST_HOST_DEVICE T operator()(T& x) const
  {
    return x++;
  }
};

template <typename T>
struct prefix_decrement_reference
{
  THRUST_HOST_DEVICE T& operator()(T& x) const
  {
    return --x;
  }
};

template <typename T>
struct suffix_decrement_reference
{
  THRUST_HOST_DEVICE T operator()(T& x) const
  {
    return x--;
  }
};

#define PREFIX_FUNCTIONAL_PLACEHOLDERS_TEST(name, reference_operator, functor)                                \
  TYPED_TEST(IntegralVectorTests, TestFunctionalPlaceholdersPrefix##name)                                     \
  {                                                                                                           \
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());                  \
                                                                                                              \
    const size_t num_samples = 10000;                                                                         \
    using Vector             = typename TestFixture::input_type;                                              \
    using T                  = typename Vector::value_type;                                                   \
    Vector input             = random_samples<T>(num_samples);                                                \
                                                                                                              \
    Vector input_reference = input;                                                                           \
    Vector reference(input.size());                                                                           \
    thrust::transform(input.begin(), input.end(), reference.begin(), functor<T>());                           \
                                                                                                              \
    using namespace thrust::placeholders;                                                                     \
    Vector result(input_reference.size());                                                                    \
    thrust::transform(input_reference.begin(), input_reference.end(), result.begin(), reference_operator _1); \
                                                                                                              \
    ASSERT_EQ(input_reference, input);                                                                        \
    ASSERT_EQ(reference, result);                                                                             \
  }

PREFIX_FUNCTIONAL_PLACEHOLDERS_TEST(Increment, ++, prefix_increment_reference);
PREFIX_FUNCTIONAL_PLACEHOLDERS_TEST(Decrement, --, prefix_decrement_reference);

#define SUFFIX_FUNCTIONAL_PLACEHOLDERS_TEST(name, reference_operator, functor)                                \
  TYPED_TEST(IntegralVectorTests, TestFunctionalPlaceholdersSuffix##name)                                     \
  {                                                                                                           \
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());                  \
                                                                                                              \
    const size_t num_samples = 10000;                                                                         \
    using Vector             = typename TestFixture::input_type;                                              \
    using T                  = typename Vector::value_type;                                                   \
    Vector input             = random_samples<T>(num_samples);                                                \
                                                                                                              \
    Vector input_reference = input;                                                                           \
    Vector reference(input.size());                                                                           \
    thrust::transform(input.begin(), input.end(), reference.begin(), functor<T>());                           \
                                                                                                              \
    using namespace thrust::placeholders;                                                                     \
    Vector result(input_reference.size());                                                                    \
    thrust::transform(input_reference.begin(), input_reference.end(), result.begin(), _1 reference_operator); \
                                                                                                              \
    ASSERT_EQ(input_reference, input);                                                                        \
    ASSERT_EQ(reference, result);                                                                             \
  }

SUFFIX_FUNCTIONAL_PLACEHOLDERS_TEST(Increment, ++, suffix_increment_reference);
SUFFIX_FUNCTIONAL_PLACEHOLDERS_TEST(Decrement, --, suffix_decrement_reference);
