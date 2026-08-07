/*
 *  Copyright 2008-2024 NVIDIA Corporation
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
#include <thrust/detail/config.h>

#include <thrust/complex.h>
#include <thrust/detail/type_traits.h>

#include <complex>
#include <iostream>
#include <sstream>

#include "test_imag_assertions.hpp"
#include "test_param_fixtures.hpp"
#include "test_utils.hpp"

#if !_THRUST_HAS_DEVICE_SYSTEM_STD
#  include <type_traits>
#endif

#define CHECK_CORRECT(stComplex, ttComplex, lreal, limag, rreal, rimag)                \
  do                                                                                   \
  {                                                                                    \
    SCOPED_TRACE(testing::Message()                                                    \
                 << "\nLHS --- Real: " << lreal << " Imag: " << limag << std::endl     \
                 << "RHS --- Real: " << rreal << " Imag: " << rimag << std::endl       \
                 << "AC: " << rreal * lreal << " BD: " << limag * rimag << std::endl); \
    if (std::isinf(stComplex.real()))                                                  \
    {                                                                                  \
      ASSERT_TRUE(std::isinf(ttComplex.real()));                                       \
    }                                                                                  \
    else if (std::isnan(stComplex.real()))                                             \
    {                                                                                  \
      ASSERT_TRUE(std::isnan(ttComplex.real()));                                       \
    }                                                                                  \
    else                                                                               \
    {                                                                                  \
      auto eps = stComplex.real() < 0.01 ? 1e-2 : abs(stComplex.real() * 0.05);        \
      ASSERT_NEAR(stComplex.real(), ttComplex.real(), eps);                            \
    }                                                                                  \
    if (std::isinf(stComplex.imag()))                                                  \
    {                                                                                  \
      ASSERT_TRUE(std::isinf(ttComplex.imag()));                                       \
    }                                                                                  \
    else if (std::isnan(stComplex.imag()))                                             \
    {                                                                                  \
      ASSERT_TRUE(std::isnan(ttComplex.imag()));                                       \
    }                                                                                  \
    else                                                                               \
    {                                                                                  \
      auto eps = stComplex.imag() < 0.01 ? 1e-2 : abs(stComplex.imag() * 0.05);        \
      ASSERT_NEAR(stComplex.imag(), ttComplex.imag(), eps);                            \
    }                                                                                  \
  } while (0)

TESTS_DEFINE(ComplexTests, FloatTestsParams);

TYPED_TEST(ComplexTests, TestComplexConstructors)
{
  using T = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  for (auto seed : get_seeds())
  {
    SCOPED_TRACE(testing::Message() << "with seed= " << seed);

    thrust::host_vector<T> data = get_random_data<T>(2, saturate_cast<T>(-1000), saturate_cast<T>(1000), seed);

    thrust::complex<T> a(data[0], data[1]);
    thrust::complex<T> b(a);
    thrust::complex<float> float_b(a);
    a = thrust::complex<T>(data[0], data[1]);
    ASSERT_NEAR_COMPLEX(a, b);

    a = thrust::complex<T>(data[0]);
    ASSERT_EQ(data[0], a.real());
    ASSERT_EQ(T(0), a.imag());

    a = thrust::complex<T>();
    ASSERT_NEAR_COMPLEX(a, std::complex<T>(0));

    a = thrust::complex<T>(thrust::complex<float>(static_cast<float>(data[0]), static_cast<float>(data[1])));
    ASSERT_NEAR_COMPLEX(a, float_b);

    a = thrust::complex<T>(thrust::complex<double>(static_cast<double>(data[0]), static_cast<double>(data[1])));
    ASSERT_NEAR_COMPLEX(a, b);

    a = thrust::complex<T>(std::complex<float>(static_cast<float>(data[0]), static_cast<float>(data[1])));
    ASSERT_NEAR_COMPLEX(a, float_b);

    a = thrust::complex<T>(std::complex<double>(static_cast<double>(data[0]), static_cast<double>(data[1])));
    ASSERT_NEAR_COMPLEX(a, b);
  }
}

THRUST_DIAG_SUPPRESS_MSVC(4244) // conversion from 'const T1' to 'const T', possible loss of data

/*
   The following tests do not check for the numerical accuracy of the operations.
   That is tested in a separate program (complex_accuracy.cpp) which requires mpfr,
   and takes a lot of time to run.
 */

namespace
{

// Helper construction to create a double from float and
// vice versa to test thrust::complex promoting operators.
template <typename T>
struct other_floating_point_type
{};

template <>
struct other_floating_point_type<float>
{
  using type = double;
};

template <>
struct other_floating_point_type<double>
{
  using type = float;
};

template <typename T>
using other_floating_point_type_t = typename other_floating_point_type<T>::type;

} // anonymous namespace

TYPED_TEST(ComplexTests, TestComplexSizeAndAlignment)
{
  using T = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  THRUST_STATIC_ASSERT(sizeof(thrust::complex<T>) == sizeof(T) * 2);
  THRUST_STATIC_ASSERT(alignof(thrust::complex<T>) == alignof(T) * 2);

  THRUST_STATIC_ASSERT(sizeof(thrust::complex<T const>) == sizeof(T) * 2);
  THRUST_STATIC_ASSERT(alignof(thrust::complex<T const>) == alignof(T) * 2);
}

TYPED_TEST(ComplexTests, TestComplexConstructionAndAssignment)
{
  using T = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::host_vector<T> data = random_samples<T>(2);

  const T real = data[0];
  const T imag = data[1];

  {
    const thrust::complex<T> construct_from_real_and_imag(real, imag);
    ASSERT_EQ(real, construct_from_real_and_imag.real());
    ASSERT_EQ(imag, construct_from_real_and_imag.imag());
  }

  {
    const thrust::complex<T> construct_from_real(real);
    ASSERT_EQ(real, construct_from_real.real());
    ASSERT_EQ(T(), construct_from_real.imag());
  }

  {
    const thrust::complex<T> expected(real, imag);
    thrust::complex<T> construct_from_copy(expected);
    ASSERT_EQ(expected.real(), construct_from_copy.real());
    ASSERT_EQ(expected.imag(), construct_from_copy.imag());
  }

  {
    thrust::complex<T> construct_from_move(thrust::complex<T>(real, imag));
    ASSERT_EQ(real, construct_from_move.real());
    ASSERT_EQ(imag, construct_from_move.imag());
  }

  {
    thrust::complex<T> copy_assign{};
    const thrust::complex<T> expected{real, imag};
    copy_assign = expected;
    ASSERT_EQ(expected.real(), copy_assign.real());
    ASSERT_EQ(expected.imag(), copy_assign.imag());
  }

  {
    thrust::complex<T> move_assign{};
    const thrust::complex<T> expected{real, imag};
    move_assign = thrust::complex<T>{real, imag};
    ASSERT_EQ(expected.real(), move_assign.real());
    ASSERT_EQ(expected.imag(), move_assign.imag());
  }

  {
    thrust::complex<T> assign_from_lvalue_T{};
    const thrust::complex<T> expected{real};
    const T to_be_copied = real;
    assign_from_lvalue_T = to_be_copied;
    ASSERT_EQ(expected.real(), assign_from_lvalue_T.real());
    ASSERT_EQ(expected.imag(), assign_from_lvalue_T.imag());
  }

  {
    thrust::complex<T> assign_from_rvalue_T{};
    const thrust::complex<T> expected{real};
    assign_from_rvalue_T = T(real);
    ASSERT_EQ(expected.real(), assign_from_rvalue_T.real());
    ASSERT_EQ(expected.imag(), assign_from_rvalue_T.imag());
  }

  {
    const std::complex<T> expected(real, imag);
    const thrust::complex<T> copy_from_std(expected);
    ASSERT_EQ(expected.real(), copy_from_std.real());
    ASSERT_EQ(expected.imag(), copy_from_std.imag());
  }

  {
    thrust::complex<T> assign_from_lvalue_std{};
    const std::complex<T> expected(real, imag);
    assign_from_lvalue_std = expected;
    ASSERT_EQ(expected.real(), assign_from_lvalue_std.real());
    ASSERT_EQ(expected.imag(), assign_from_lvalue_std.imag());
  }

  {
    thrust::complex<T> assign_from_rvalue_std{};
    assign_from_rvalue_std = std::complex<T>(real, imag);
    ASSERT_EQ(real, assign_from_rvalue_std.real());
    ASSERT_EQ(imag, assign_from_rvalue_std.imag());
  }
}

TYPED_TEST(ComplexTests, TestComplexConstructionAndAssignmentWithPromoting)
{
  using T = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  using T0 = T;
  using T1 = other_floating_point_type_t<T0>;

  thrust::host_vector<T0> data = random_samples<T0>(2);

  const T0 real_T0 = data[0];
  const T0 imag_T0 = data[1];

  const T1 real_T1 = static_cast<T1>(real_T0);
  const T1 imag_T1 = static_cast<T1>(imag_T0);

  {
    const thrust::complex<T0> construct_from_real_and_imag(real_T1, imag_T1);
    ASSERT_EQ(real_T0, construct_from_real_and_imag.real());
    ASSERT_EQ(imag_T0, construct_from_real_and_imag.imag());
  }

  {
    const thrust::complex<T0> construct_from_real(real_T1);
    ASSERT_EQ(real_T0, construct_from_real.real());
    ASSERT_EQ(T0(), construct_from_real.imag());
  }

  {
    const thrust::complex<T1> expected(real_T1, imag_T1);
    thrust::complex<T0> construct_from_copy(expected);
    ASSERT_EQ(real_T0, construct_from_copy.real());
    ASSERT_EQ(imag_T0, construct_from_copy.imag());
  }

  {
    thrust::complex<T0> construct_from_move(thrust::complex<T1>(real_T1, imag_T1));
    ASSERT_EQ(real_T0, construct_from_move.real());
    ASSERT_EQ(imag_T0, construct_from_move.imag());
  }

  {
    thrust::complex<T0> copy_assign{};
    const thrust::complex<T1> expected{real_T1, imag_T1};
    copy_assign = expected;
    ASSERT_EQ(expected.real(), copy_assign.real());
    ASSERT_EQ(expected.imag(), copy_assign.imag());
  }

  {
    thrust::complex<T0> assign_from_lvalue_T{};
    const thrust::complex<T1> expected{real_T1};
    const T1 to_be_copied = real_T1;
    assign_from_lvalue_T  = to_be_copied;
    ASSERT_EQ(expected.real(), assign_from_lvalue_T.real());
    ASSERT_EQ(expected.imag(), assign_from_lvalue_T.imag());
  }

  {
    const std::complex<T1> expected(real_T1, imag_T1);
    const thrust::complex<T0> copy_from_std(expected);
    ASSERT_EQ(expected.real(), copy_from_std.real());
    ASSERT_EQ(expected.imag(), copy_from_std.imag());
  }

  {
    thrust::complex<T1> assign_from_lvalue_std{};
    const std::complex<T0> expected(real_T1, imag_T1);
    assign_from_lvalue_std = expected;
    ASSERT_EQ(expected.real(), assign_from_lvalue_std.real());
    ASSERT_EQ(expected.imag(), assign_from_lvalue_std.imag());
  }

  {
    thrust::complex<T0> assign_from_rvalue_std{};
    assign_from_rvalue_std = std::complex<T1>(real_T1, imag_T1);
    ASSERT_EQ(real_T0, assign_from_rvalue_std.real());
    ASSERT_EQ(imag_T1, assign_from_rvalue_std.imag());
  }
}

TYPED_TEST(ComplexTests, TestComplexGetters)
{
  using T = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  for (auto seed : get_seeds())
  {
    SCOPED_TRACE(testing::Message() << "with seed= " << seed);

    thrust::host_vector<T> data =
      get_random_data<T>(2, get_default_limits<T>::min(), get_default_limits<T>::max(), seed);

    thrust::complex<T> z(data[0], data[1]);

    ASSERT_EQ(data[0], z.real());
    ASSERT_EQ(data[1], z.imag());

    z.real(data[1]);
    z.imag(data[0]);
    ASSERT_EQ(data[1], z.real());
    ASSERT_EQ(data[0], z.imag());

    volatile thrust::complex<T> v(data[0], data[1]);

    ASSERT_EQ(data[0], v.real());
    ASSERT_EQ(data[1], v.imag());

    v.real(data[1]);
    v.imag(data[0]);
    ASSERT_EQ(data[1], v.real());
    ASSERT_EQ(data[0], v.imag());
  }
}

TYPED_TEST(ComplexTests, TestComplexComparisionOperators)
{
  using T = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  {
    thrust::host_vector<T> data = random_samples<T>(1);

    const T a = data[0];
    const T b = data[0] + T(1.0);

    ASSERT_EQ(thrust::complex<T>(a, b) == thrust::complex<T>(a, b), true);
    ASSERT_EQ(thrust::complex<T>(a, T()) == a, true);
    ASSERT_EQ(a == thrust::complex<T>(a, T()), true);

    ASSERT_EQ(thrust::complex<T>(a, b) == std::complex<T>(a, b), true);
    ASSERT_EQ(std::complex<T>(a, b) == thrust::complex<T>(a, b), true);

    ASSERT_EQ(thrust::complex<T>(a, b) != thrust::complex<T>(b, a), true);
    ASSERT_EQ(thrust::complex<T>(a, T()) != b, true);
    ASSERT_EQ(b != thrust::complex<T>(a, T()), true);

    ASSERT_EQ(thrust::complex<T>(a, b) != a, true);
    ASSERT_EQ(a != thrust::complex<T>(a, b), true);

    ASSERT_EQ(thrust::complex<T>(a, b) != std::complex<T>(b, a), true);
    ASSERT_EQ(std::complex<T>(a, b) != thrust::complex<T>(b, a), true);
  }

  // Testing comparison operators with promoted types.
  // These tests don't use random numbers on purpose since `T0(x) == T1(x)` will
  // not be true for all x.
  {
    using T0 = T;
    using T1 = other_floating_point_type_t<T0>;

    ASSERT_EQ(thrust::complex<T0>(1.0, 2.0) == thrust::complex<T1>(1.0, 2.0), true);
    ASSERT_EQ(thrust::complex<T0>(1.0, T0()) == T1(1.0), true);
    ASSERT_EQ(T1(1.0) == thrust::complex<T0>(1.0, 0.0), true);

    ASSERT_EQ(thrust::complex<T0>(1.0, 2.0) == std::complex<T1>(1.0, 2.0), true);
    ASSERT_EQ(std::complex<T0>(1.0, 2.0) == thrust::complex<T1>(1.0, 2.0), true);

    ASSERT_EQ(thrust::complex<T0>(1.0, 2.0) != T1(1.0), true);
    ASSERT_EQ(T1(1.0) != thrust::complex<T0>(1.0, 2.0), true);
  }
}

TYPED_TEST(ComplexTests, TestComplexMemberOperators)
{
  using T = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  {
    thrust::host_vector<T> data = random_samples<T>(5);

    thrust::complex<T> a_thrust(data[0], data[1]);
    const thrust::complex<T> b_thrust(data[2], data[3]);

    std::complex<T> a_std(a_thrust);
    const std::complex<T> b_std(b_thrust);

    a_thrust += b_thrust;
    a_std += b_std;
    ASSERT_NEAR_COMPLEX(a_thrust, a_std);

    a_thrust -= b_thrust;
    a_std -= b_std;
    ASSERT_NEAR_COMPLEX(a_thrust, a_std);

    a_thrust *= b_thrust;
    a_std *= b_std;
    ASSERT_NEAR_COMPLEX(a_thrust, a_std);

    a_thrust /= b_thrust;
    a_std /= b_std;
    ASSERT_NEAR_COMPLEX(a_thrust, a_std);

    // arithmetic operators with `double` and `float`
    const T real = data[4];

    a_thrust += real;
    a_std += std::complex<T>(real);
    ASSERT_NEAR_COMPLEX(a_thrust, a_std);

    a_thrust -= real;
    a_std -= std::complex<T>(real);
    ASSERT_NEAR_COMPLEX(a_thrust, a_std);

    a_thrust *= real;
    a_std *= std::complex<T>(real);
    ASSERT_NEAR_COMPLEX(a_thrust, a_std);

    a_thrust /= real;
    a_std /= std::complex<T>(real);
    ASSERT_NEAR_COMPLEX(a_thrust, a_std);

    // casting operator
    a_std = (std::complex<T>) a_thrust;
    ASSERT_EQ(a_thrust.real(), a_std.real());
    ASSERT_EQ(a_thrust.imag(), a_std.imag());
  }

  // Testing arithmetic member operators with promoted types.
  {
    using T0 = T;
    using T1 = other_floating_point_type_t<T0>;

    thrust::host_vector<T0> data = random_samples<T0>(5);

    thrust::complex<T0> a_thrust(data[0], data[1]);
    const thrust::complex<T1> b_thrust(data[2], data[3]);

    std::complex<T1> a_std(data[0], data[1]);
    const std::complex<T1> b_std(data[2], data[3]);

    // The following tests require that thrust::complex and std::complex are `almost` equal.
    ASSERT_NEAR_COMPLEX(a_thrust, a_std);
    ASSERT_NEAR_COMPLEX(b_thrust, b_thrust);

    a_thrust += b_thrust;
    a_std += b_std;
    ASSERT_NEAR_COMPLEX(a_thrust, a_std);

    a_thrust -= b_thrust;
    a_std -= b_std;
    ASSERT_NEAR_COMPLEX(a_thrust, a_std);

    a_thrust *= b_thrust;
    a_std *= b_std;
    ASSERT_NEAR_COMPLEX(a_thrust, a_std);

    a_thrust /= b_thrust;
    a_std /= b_std;
    ASSERT_NEAR_COMPLEX(a_thrust, a_std);

    // Testing arithmetic member operators with another floating point type.
    const T1 e = data[2];

    a_thrust += e;
    a_std += std::complex<T1>(e);
    ASSERT_NEAR_COMPLEX(a_thrust, a_std);

    a_thrust -= e;
    a_std -= std::complex<T1>(e);
    ASSERT_NEAR_COMPLEX(a_thrust, a_std);

    a_thrust *= e;
    a_std *= std::complex<T1>(e);
    ASSERT_NEAR_COMPLEX(a_thrust, a_std);

    a_thrust /= e;
    a_std /= std::complex<T1>(e);
    ASSERT_NEAR_COMPLEX(a_thrust, a_std);
  }
}

TYPED_TEST(ComplexTests, TestComplexBasicArithmetic)
{
  using T = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  for (auto seed : get_seeds())
  {
    SCOPED_TRACE(testing::Message() << "with seed= " << seed);

    thrust::host_vector<T> data = get_random_data<T>(2, saturate_cast<T>(-100), saturate_cast<T>(100), seed);

    const thrust::complex<T> a(data[0], data[1]);
    const std::complex<T> b(a);

    // Test the basic arithmetic functions against std

    ASSERT_NEAR(thrust::abs(a), std::abs(b), T(0.01));
    ASSERT_NEAR(thrust::arg(a), std::arg(b), T(0.01));
    ASSERT_NEAR(thrust::norm(a), std::norm(b), T(0.01));

    ASSERT_EQ(thrust::conj(a), std::conj(b));
    static_assert(_THRUST_STD::is_same<thrust::complex<T>, decltype(thrust::conj(a))>::value, "");

    ASSERT_NEAR_COMPLEX(thrust::polar(data[0], data[1]), std::polar(data[0], data[1]));
    static_assert(_THRUST_STD::is_same<thrust::complex<T>, decltype(thrust::polar(data[0], data[1]))>::value, "");

    // random_samples does not seem to produce infinities so proj(z) == z
    ASSERT_EQ(thrust::proj(a), a);
    static_assert(_THRUST_STD::is_same<thrust::complex<T>, decltype(thrust::proj(a))>::value, "");
  }
}

TYPED_TEST(ComplexTests, TestComplexBinaryArithmetic)
{
  using T = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  {
    thrust::host_vector<T> data = random_samples<T>(5);

    const thrust::complex<T> a(data[0], data[1]);
    const thrust::complex<T> b(data[2], data[3]);
    const T real = data[4];

    ASSERT_NEAR_COMPLEX(a * b, std::complex<T>(a) * std::complex<T>(b));
    ASSERT_NEAR_COMPLEX(a * real, std::complex<T>(a) * real);
    ASSERT_NEAR_COMPLEX(real * b, real * std::complex<T>(b));

    ASSERT_NEAR_COMPLEX(a / b, std::complex<T>(a) / std::complex<T>(b));
    ASSERT_NEAR_COMPLEX(a / real, std::complex<T>(a) / real);
    ASSERT_NEAR_COMPLEX(real / b, real / std::complex<T>(b));

    ASSERT_EQ(a + b, std::complex<T>(a) + std::complex<T>(b));
    ASSERT_EQ(a + real, std::complex<T>(a) + real);
    ASSERT_EQ(real + b, real + std::complex<T>(b));

    ASSERT_EQ(a - b, std::complex<T>(a) - std::complex<T>(b));
    ASSERT_EQ(a - real, std::complex<T>(a) - real);
    ASSERT_EQ(real - b, real - std::complex<T>(b));
  }

  // Testing binary arithmetic with promoted types.
  {
    using T0 = T;
    using T1 = other_floating_point_type_t<T0>;

    thrust::host_vector<T0> data = unittest::random_samples<T0>(5);

    const thrust::complex<T0> a_thrust(data[0], data[1]);
    const thrust::complex<T1> b_thrust(data[2], data[3]);
    const thrust::complex<T0> a_std(data[0], data[1]);
    const thrust::complex<T0> b_std(data[2], data[3]);

    const T0 real_T0 = data[4];
    const T1 real_T1 = static_cast<T1>(real_T0);

    ASSERT_NEAR_COMPLEX(a_thrust * b_thrust, a_std * b_std);
    ASSERT_NEAR_COMPLEX(a_thrust * real_T1, a_std * real_T0);
    ASSERT_NEAR_COMPLEX(real_T0 * b_thrust, real_T0 * b_std);

    ASSERT_NEAR_COMPLEX(a_thrust / b_thrust, a_std / b_std);
    ASSERT_NEAR_COMPLEX(a_thrust / real_T1, a_std / real_T0);
    ASSERT_NEAR_COMPLEX(real_T0 / b_thrust, real_T0 / b_std);

    ASSERT_NEAR_COMPLEX(a_thrust + b_thrust, a_std + b_std);
    ASSERT_NEAR_COMPLEX(a_thrust + real_T1, a_std + real_T0);
    ASSERT_NEAR_COMPLEX(real_T0 + b_thrust, real_T0 + b_std);

    ASSERT_NEAR_COMPLEX(a_thrust - b_thrust, a_std - b_std);
    ASSERT_NEAR_COMPLEX(a_thrust - real_T1, a_std - real_T0);
    ASSERT_NEAR_COMPLEX(real_T0 - b_thrust, real_T0 - b_std);
  }
}

TYPED_TEST(ComplexTests, TestComplexUnaryArithmetic)
{
  using T = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  for (auto seed : get_seeds())
  {
    SCOPED_TRACE(testing::Message() << "with seed= " << seed);

    thrust::host_vector<T> data =
      get_random_data<T>(2, get_default_limits<T>::min(), get_default_limits<T>::max(), seed);

    const thrust::complex<T> a(data[0], data[1]);

    ASSERT_EQ(+a, a);
    ASSERT_EQ(-a, a * (-1.0));
  }
}

TYPED_TEST(ComplexTests, TestComplexExponentialFunctions)
{
  using T = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  for (auto seed : get_seeds())
  {
    SCOPED_TRACE(testing::Message() << "with seed= " << seed);

    thrust::host_vector<T> data = get_random_data<T>(2, -100, 100, seed);

    const thrust::complex<T> a(data[0], data[1]);
    const std::complex<T> b(a);

    ASSERT_NEAR_COMPLEX(thrust::exp(a), std::exp(b));
    ASSERT_NEAR_COMPLEX(thrust::log(a), std::log(b));
    ASSERT_NEAR_COMPLEX(thrust::log10(a), std::log10(b));
    static_assert(_THRUST_STD::is_same<thrust::complex<T>, decltype(thrust::exp(a))>::value, "");
    static_assert(_THRUST_STD::is_same<thrust::complex<T>, decltype(thrust::log(a))>::value, "");
    static_assert(_THRUST_STD::is_same<thrust::complex<T>, decltype(thrust::log10(a))>::value, "");
  }
}

TYPED_TEST(ComplexTests, TestComplexPowerFunctions)
{
  using T = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  {
    thrust::host_vector<T> data = random_samples<T>(4);

    const thrust::complex<T> a_thrust(data[0], data[1]);
    const thrust::complex<T> b_thrust(data[2], data[3]);
    const std::complex<T> a_std(a_thrust);
    const std::complex<T> b_std(b_thrust);

    ASSERT_NEAR_COMPLEX(thrust::pow(a_thrust, b_thrust), std::pow(a_std, b_std));
    static_assert(_THRUST_STD::is_same<thrust::complex<T>, decltype(thrust::pow(a_thrust, b_thrust))>::value, "");
    ASSERT_NEAR_COMPLEX(thrust::pow(a_thrust, b_thrust.real()), std::pow(a_std, b_std.real()));
    static_assert(_THRUST_STD::is_same<thrust::complex<T>, decltype(thrust::pow(a_thrust, b_thrust.real()))>::value,
                  "");
    ASSERT_NEAR_COMPLEX(thrust::pow(a_thrust.real(), b_thrust), std::pow(a_std.real(), b_std));
    static_assert(_THRUST_STD::is_same<thrust::complex<T>, decltype(thrust::pow(a_thrust.real(), b_thrust))>::value,
                  "");

    ASSERT_NEAR_COMPLEX(thrust::pow(a_thrust, 4), std::pow(a_std, 4));
    static_assert(_THRUST_STD::is_same<thrust::complex<T>, decltype(thrust::pow(a_thrust, 4))>::value, "");

    ASSERT_NEAR_COMPLEX(thrust::sqrt(a_thrust), std::sqrt(a_std));
    static_assert(_THRUST_STD::is_same<thrust::complex<T>, decltype(thrust::sqrt(a_thrust))>::value, "");
  }

  // Test power functions with promoted types.
  {
    using T0       = T;
    using T1       = other_floating_point_type_t<T0>;
    using promoted = _THRUST_STD::common_type_t<T0, T1>;

    thrust::host_vector<T0> data = random_samples<T0>(4);

    const thrust::complex<T0> a_thrust(data[0], data[1]);
    const thrust::complex<T1> b_thrust(data[2], data[3]);
    const std::complex<T0> a_std(data[0], data[1]);
    const std::complex<T0> b_std(data[2], data[3]);

    ASSERT_NEAR_COMPLEX(thrust::pow(a_thrust, b_thrust), std::pow(a_std, b_std));
    static_assert(_THRUST_STD::is_same<thrust::complex<promoted>, decltype(thrust::pow(a_thrust, b_thrust))>::value,
                  "");
    ASSERT_NEAR_COMPLEX(thrust::pow(b_thrust, a_thrust), std::pow(b_std, a_std));
    static_assert(_THRUST_STD::is_same<thrust::complex<promoted>, decltype(thrust::pow(b_thrust, a_thrust))>::value,
                  "");
    ASSERT_NEAR_COMPLEX(thrust::pow(a_thrust, b_thrust.real()), std::pow(a_std, b_std.real()));
    static_assert(
      _THRUST_STD::is_same<thrust::complex<promoted>, decltype(thrust::pow(a_thrust, b_thrust.real()))>::value, "");
    ASSERT_NEAR_COMPLEX(thrust::pow(b_thrust, a_thrust.real()), std::pow(b_std, a_std.real()));
    static_assert(
      _THRUST_STD::is_same<thrust::complex<promoted>, decltype(thrust::pow(b_thrust, a_thrust.real()))>::value, "");
    ASSERT_NEAR_COMPLEX(thrust::pow(a_thrust.real(), b_thrust), std::pow(a_std.real(), b_std));
    static_assert(
      _THRUST_STD::is_same<thrust::complex<promoted>, decltype(thrust::pow(a_thrust.real(), b_thrust))>::value, "");
    ASSERT_NEAR_COMPLEX(thrust::pow(b_thrust.real(), a_thrust), std::pow(b_std.real(), a_std));
    static_assert(
      _THRUST_STD::is_same<thrust::complex<promoted>, decltype(thrust::pow(b_thrust.real(), a_thrust))>::value, "");
  }
}

TYPED_TEST(ComplexTests, TestComplexTrigonometricFunctions)
{
  using T = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  for (auto seed : get_seeds())
  {
    SCOPED_TRACE(testing::Message() << "with seed= " << seed);

    thrust::host_vector<T> data = get_random_data<T>(2, saturate_cast<T>(-1), saturate_cast<T>(1), seed);

    const thrust::complex<T> a(data[0], data[1]);
    const std::complex<T> c(a);

    ASSERT_NEAR_COMPLEX(thrust::cos(a), std::cos(c));
    ASSERT_NEAR_COMPLEX(thrust::sin(a), std::sin(c));
    ASSERT_NEAR_COMPLEX(thrust::tan(a), std::tan(c));
    static_assert(_THRUST_STD::is_same<thrust::complex<T>, decltype(thrust::cos(a))>::value, "");
    static_assert(_THRUST_STD::is_same<thrust::complex<T>, decltype(thrust::sin(a))>::value, "");
    static_assert(_THRUST_STD::is_same<thrust::complex<T>, decltype(thrust::tan(a))>::value, "");

    ASSERT_NEAR_COMPLEX(thrust::cosh(a), std::cosh(c));
    ASSERT_NEAR_COMPLEX(thrust::sinh(a), std::sinh(c));
    ASSERT_NEAR_COMPLEX(thrust::tanh(a), std::tanh(c));
    static_assert(_THRUST_STD::is_same<thrust::complex<T>, decltype(thrust::cosh(a))>::value, "");
    static_assert(_THRUST_STD::is_same<thrust::complex<T>, decltype(thrust::sinh(a))>::value, "");
    static_assert(_THRUST_STD::is_same<thrust::complex<T>, decltype(thrust::tanh(a))>::value, "");

    ASSERT_NEAR_COMPLEX(thrust::acos(a), std::acos(c));
    ASSERT_NEAR_COMPLEX(thrust::asin(a), std::asin(c));
    ASSERT_NEAR_COMPLEX(thrust::atan(a), std::atan(c));
    static_assert(_THRUST_STD::is_same<thrust::complex<T>, decltype(thrust::acos(a))>::value, "");
    static_assert(_THRUST_STD::is_same<thrust::complex<T>, decltype(thrust::asin(a))>::value, "");
    static_assert(_THRUST_STD::is_same<thrust::complex<T>, decltype(thrust::atan(a))>::value, "");

    ASSERT_NEAR_COMPLEX(thrust::acosh(a), std::acosh(c));
    ASSERT_NEAR_COMPLEX(thrust::asinh(a), std::asinh(c));
    ASSERT_NEAR_COMPLEX(thrust::atanh(a), std::atanh(c));
    static_assert(_THRUST_STD::is_same<thrust::complex<T>, decltype(thrust::acosh(a))>::value, "");
    static_assert(_THRUST_STD::is_same<thrust::complex<T>, decltype(thrust::asinh(a))>::value, "");
    static_assert(_THRUST_STD::is_same<thrust::complex<T>, decltype(thrust::atanh(a))>::value, "");
  }
}

TYPED_TEST(ComplexTests, TestComplexStreamOperators)
{
  using T = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  for (auto seed : get_seeds())
  {
    SCOPED_TRACE(testing::Message() << "with seed= " << seed);

    thrust::host_vector<T> data = get_random_data<T>(2, saturate_cast<T>(-1000), saturate_cast<T>(1000), seed);
    const thrust::complex<T> a(data[0], data[1]);

    std::stringstream out;
    out << a;
    thrust::complex<T> b;
    out >> b;
    ASSERT_NEAR_COMPLEX(a, b);
  }
}

TYPED_TEST(ComplexTests, TestComplexStdComplexDeviceInterop)
{
  using T = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::host_vector<T> data = random_samples<T>(6);
  std::vector<std::complex<T>> vec(10);
  vec[0] = std::complex<T>(data[0], data[1]);
  vec[1] = std::complex<T>(data[2], data[3]);
  vec[2] = std::complex<T>(data[4], data[5]);

  thrust::device_vector<thrust::complex<T>> device_vec = vec;
  ASSERT_EQ(vec[0].real(), thrust::complex<T>(device_vec[0]).real());
  ASSERT_EQ(vec[0].imag(), thrust::complex<T>(device_vec[0]).imag());
  ASSERT_EQ(vec[1].real(), thrust::complex<T>(device_vec[1]).real());
  ASSERT_EQ(vec[1].imag(), thrust::complex<T>(device_vec[1]).imag());
  ASSERT_EQ(vec[2].real(), thrust::complex<T>(device_vec[2]).real());
  ASSERT_EQ(vec[2].imag(), thrust::complex<T>(device_vec[2]).imag());
}

namespace
{

template <typename T>
struct user_complex
{
  THRUST_HOST_DEVICE user_complex(T, T) {}
  THRUST_HOST_DEVICE user_complex(const thrust::complex<T>&) {}
};

} // anonymous namespace

TYPED_TEST(ComplexTests, TestComplexExplicitConstruction)
{
  using T = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  const thrust::complex<T> input(42.0, 1337.0);
  const user_complex<T> result = thrust::exp(input);
  (void) result;
}

TESTS_PAIRS_DEFINE(ComplexPairsTests, PairsTestsParams)

TYPED_TEST(ComplexPairsTests, TestAssignOperator)
{
  using T = typename TestFixture::first_type;
  using U = typename TestFixture::second_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  constexpr size_t test_it = 123456;

  const double tmini = static_cast<double>(std::numeric_limits<T>::min());
  const double tmaxi = static_cast<double>(std::numeric_limits<T>::max());

  const double umini = static_cast<double>(std::numeric_limits<U>::min());
  const double umaxi = static_cast<double>(std::numeric_limits<U>::max());

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> tdis(tmini, tmaxi);
  std::uniform_real_distribution<double> udis(umini, umaxi);

  for (size_t i = 0; i < test_it; i++)
  {
    thrust::complex<T> thrustNum;
    std::complex<T> expectedNum;
    T treal = tdis(gen);

    thrustNum   = treal;
    expectedNum = treal;

    ASSERT_EQ(thrustNum.imag(), expectedNum.imag());
    ASSERT_EQ(thrustNum.real(), expectedNum.real());

    U ureal = static_cast<T>(udis(gen));
    U uimag = static_cast<T>(udis(gen));
    thrust::complex<U> thrustOrigin(static_cast<U>(ureal), static_cast<U>(uimag));

    thrustNum = thrustOrigin;
    ASSERT_EQ(thrustNum.imag(), static_cast<T>(thrustOrigin.imag()));
    ASSERT_EQ(thrustNum.real(), static_cast<T>(thrustOrigin.real()));

    ureal = static_cast<T>(udis(gen));
    uimag = static_cast<T>(udis(gen));
    std::complex<U> stdOriginU(static_cast<U>(ureal), static_cast<U>(uimag));

    thrustNum   = stdOriginU;
    expectedNum = stdOriginU;

    ASSERT_EQ(thrustNum.imag(), expectedNum.imag());
    ASSERT_EQ(thrustNum.real(), expectedNum.real());

    treal   = tdis(gen);
    T timag = tdis(gen);
    std::complex<T> stdOriginT(static_cast<T>(treal), static_cast<T>(timag));

    thrustNum   = stdOriginT;
    expectedNum = stdOriginT;

    ASSERT_EQ(thrustNum.imag(), expectedNum.imag());
    ASSERT_EQ(thrustNum.real(), expectedNum.real());
  }
}

template <typename T, typename U, class StdComplexOp, class StdScalarOp, class ThrustComplexOp, class ThrustScalarOp>
void run_compound_tests(
  const StdComplexOp& standard_complex_operator,
  const StdScalarOp& standard_scalar_operator,
  const ThrustComplexOp& thrust_complex_operator,
  const ThrustScalarOp& thrust_scalar_operator,
  T mini = -500000,
  T maxi = 500000)
{
  constexpr T test_it = 15;
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<T> dis(mini, maxi);

  T inc = (maxi - mini) / test_it;

  for (T treal = mini; treal <= maxi; treal += inc)
  {
    for (T timag = mini; timag <= maxi; timag += inc)
    {
      for (U ureal = mini; ureal <= maxi; ureal += inc)
      {
        for (U uimag = mini; uimag <= maxi; uimag += inc)
        {
          std::complex<T> stComplex(treal, timag);
          std::complex<U> suComplex(ureal, uimag);

          thrust::complex<T> ttComplex(treal, timag);
          thrust::complex<U> tuComplex(ureal, uimag);

          standard_complex_operator(stComplex, suComplex);
          thrust_complex_operator(ttComplex, tuComplex);

          CHECK_CORRECT(stComplex, ttComplex, treal, timag, ureal, uimag);

          stComplex = std::complex<T>(treal, timag);
          ttComplex = thrust::complex<T>(treal, timag);

          standard_scalar_operator(stComplex, ureal);
          thrust_scalar_operator(ttComplex, ureal);

          CHECK_CORRECT(stComplex, ttComplex, treal, timag, ureal, uimag);

          stComplex = std::complex<T>(treal, timag);
          ttComplex = thrust::complex<T>(treal, timag);

          standard_scalar_operator(stComplex, uimag);
          thrust_scalar_operator(ttComplex, uimag);

          CHECK_CORRECT(stComplex, ttComplex, treal, timag, ureal, uimag);
        }
      }
    }
  }
}

/**
 * Due to hipcc compiler issue, std complex functions for
 * complex number is not working properly. To get
 * around this we will have to calculate the real and
 * imaginary parts separately manually.
 */

TYPED_TEST(ComplexPairsTests, TestCompoundPlusOperator)
{
  using T = typename TestFixture::first_type;
  using U = typename TestFixture::second_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  run_compound_tests<T, U>(
    [=](std::complex<T>& lhs, const std::complex<U>& rhs) {
      using type = typename ::internal::promoted_numerical_type<T, U>::type;
      lhs        = std::complex<type>(lhs.real() + rhs.real(), lhs.imag() + rhs.imag());
    },
    [=](std::complex<T>& lhs, const U& rhs) {
      using type = typename ::internal::promoted_numerical_type<T, U>::type;
      lhs        = std::complex<type>(lhs.real() + rhs, lhs.imag());
    },
    [=](thrust::complex<T>& lhs, const thrust::complex<U>& rhs) {
      lhs += rhs;
    },
    [=](thrust::complex<T>& lhs, const U& rhs) {
      lhs += rhs;
    });
}

TYPED_TEST(ComplexPairsTests, TestCompoundMinusOperator)
{
  using T = typename TestFixture::first_type;
  using U = typename TestFixture::second_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  run_compound_tests<T, U>(
    [=](std::complex<T>& lhs, const std::complex<U>& rhs) {
      using type = typename ::internal::promoted_numerical_type<T, U>::type;
      lhs        = std::complex<type>(lhs.real() - rhs.real(), lhs.imag() - rhs.imag());
    },
    [=](std::complex<T>& lhs, const U& rhs) {
      using type = typename ::internal::promoted_numerical_type<T, U>::type;
      lhs        = std::complex<type>(lhs.real() - rhs, lhs.imag());
    },
    [=](thrust::complex<T>& lhs, const thrust::complex<U>& rhs) {
      lhs -= rhs;
    },
    [=](thrust::complex<T>& lhs, const U& rhs) {
      lhs -= rhs;
    });
}

TYPED_TEST(ComplexPairsTests, TestCompoundMultiplyOperator)
{
  using T = typename TestFixture::first_type;
  using U = typename TestFixture::second_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  run_compound_tests<T, U>(
    [=](std::complex<T>& lhs, const std::complex<U>& rhs) {
      // (a + bi)(c + di) = ac + i(ad + bc) - bd
      using type = typename ::internal::promoted_numerical_type<T, U>::type;

      type a = static_cast<type>(lhs.real());
      type b = static_cast<type>(lhs.imag());
      type c = static_cast<type>(rhs.real());
      type d = static_cast<type>(rhs.imag());

      type ac   = a * c;
      type bd   = b * d;
      type real = ac - bd;
      type imag = (a * d) + (b * c);

      lhs = std::complex<type>(real, imag);
    },
    [=](std::complex<T>& lhs, const U& rhs) {
      lhs = std::complex<T>(lhs.real() * rhs, lhs.imag() * rhs);
    },
    [=](thrust::complex<T>& lhs, const thrust::complex<U>& rhs) {
      lhs *= rhs;
    },
    [=](thrust::complex<T>& lhs, const U& rhs) {
      lhs *= rhs;
    });
}

TYPED_TEST(ComplexPairsTests, TestCompoundDivisionOperator)
{
  using T = typename TestFixture::first_type;
  using U = typename TestFixture::second_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  run_compound_tests<T, U>(
    [=](std::complex<T>& lhs, const std::complex<U>& rhs) {
      using type = typename ::internal::promoted_numerical_type<T, U>::type;

      // (a + bi) / (c + di) = ((ac + bd) + (bc -ad)i) / (c^2 + d^2)
      type a = static_cast<type>(lhs.real());
      type b = static_cast<type>(lhs.imag());
      type c = static_cast<type>(rhs.real());
      type d = static_cast<type>(rhs.imag());

      type ac    = a * c;
      type bd    = b * d;
      type bc    = b * c;
      type ad    = a * d;
      type denom = c * c + d * d;
      type real  = (ac + bd) / denom;
      type imag  = (bc - ad) / denom;

      lhs = std::complex<type>(real, imag);
    },
    [=](std::complex<T>& lhs, const U& rhs) {
      lhs /= rhs;
    },
    [=](thrust::complex<T>& lhs, const thrust::complex<U>& rhs) {
      lhs /= rhs;
    },
    [=](thrust::complex<T>& lhs, const U& rhs) {
      lhs /= rhs;
    });
}

template <typename T, typename U, class EqualOp>
void run_equality_operator_tests(const EqualOp& f, const bool scalar = false)
{
  constexpr T test_size = 123456;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<T> dis(std::numeric_limits<T>::min(), std::numeric_limits<T>::max());

  for (size_t i = 0; i < test_size; i++)
  {
    T treal = dis(gen);
    T timag = dis(gen);

    U ureal;
    U uimag;

    if (i % 2)
    {
      ureal = dis(gen);
      uimag = dis(gen);
    }
    else
    {
      ureal = treal;
      uimag = timag;
    }

    thrust::complex<T> tcomplex;
    thrust::complex<U> ucomplex;
    if (scalar)
    {
      tcomplex = thrust::complex<T>(treal);
      ucomplex = thrust::complex<U>(ureal);
    }
    else
    {
      tcomplex = thrust::complex<T>(treal, timag);
      ucomplex = thrust::complex<U>(ureal, uimag);
    }

    ASSERT_EQ((treal == ureal && (scalar ? true : timag == uimag)), f(tcomplex, ucomplex));
  }
}

TYPED_TEST(ComplexPairsTests, TestEqualOperator_ThrustComplex_ThrustComplex)
{
  using T = typename TestFixture::first_type;
  using U = typename TestFixture::second_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  run_equality_operator_tests<T, U>([=](const thrust::complex<T>& lhs, const thrust::complex<U>& rhs) {
    return lhs == rhs;
  });
}

TYPED_TEST(ComplexPairsTests, TestEqualOperator_ThrustComplex_StdComplex)
{
  using T = typename TestFixture::first_type;
  using U = typename TestFixture::second_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  run_equality_operator_tests<T, U>([=](const thrust::complex<T>& lhs, const thrust::complex<U>& rhs) {
    std::complex<U> rhs_(rhs.real(), rhs.imag());
    return lhs == rhs_;
  });
}

TYPED_TEST(ComplexPairsTests, TestEqualOperator_StdComplex_ThrustComplex)
{
  using T = typename TestFixture::first_type;
  using U = typename TestFixture::second_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  run_equality_operator_tests<T, U>([=](const thrust::complex<T>& lhs, const thrust::complex<U>& rhs) {
    std::complex<T> lhs_(lhs.real(), lhs.imag());
    return lhs_ == rhs;
  });
}

TYPED_TEST(ComplexPairsTests, TestEqualOperator_Scalar_ThrustComplex)
{
  using T = typename TestFixture::first_type;
  using U = typename TestFixture::second_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  run_equality_operator_tests<T, U>(
    [=](const thrust::complex<T>& lhs, const thrust::complex<U>& rhs) {
      return lhs.real() == rhs;
    },
    true);
}

TYPED_TEST(ComplexPairsTests, TestEqualOperator_ThrustComplex_Scalar)
{
  using T = typename TestFixture::first_type;
  using U = typename TestFixture::second_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  run_equality_operator_tests<T, U>(
    [=](const thrust::complex<T>& lhs, const thrust::complex<U>& rhs) {
      return lhs == rhs.real();
    },
    true);
}

template <typename T, typename U, class InequalOp>
void run_inequality_operator_tests(const InequalOp& f, const bool scalar = false)
{
  constexpr T test_size = 123456;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<T> dis(std::numeric_limits<T>::min(), std::numeric_limits<T>::max());

  for (size_t i = 0; i < test_size; i++)
  {
    T treal = dis(gen);
    T timag = dis(gen);

    U ureal;
    U uimag;

    if (i % 2)
    {
      ureal = dis(gen);
      uimag = dis(gen);
    }
    else
    {
      ureal = treal;
      uimag = timag;
    }

    thrust::complex<T> tcomplex;
    thrust::complex<U> ucomplex;
    if (scalar)
    {
      tcomplex = thrust::complex<T>(treal);
      ucomplex = thrust::complex<U>(ureal);
    }
    else
    {
      tcomplex = thrust::complex<T>(treal, timag);
      ucomplex = thrust::complex<U>(ureal, uimag);
    }

    ASSERT_EQ(!(treal == ureal && (scalar ? true : timag == uimag)), f(tcomplex, ucomplex));
  }
}

TYPED_TEST(ComplexPairsTests, TestInequalOperator_ThrustComplex_ThrustComplex)
{
  using T = typename TestFixture::first_type;
  using U = typename TestFixture::second_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  run_inequality_operator_tests<T, U>([=](const thrust::complex<T>& lhs, const thrust::complex<U>& rhs) {
    return lhs != rhs;
  });
}

TYPED_TEST(ComplexPairsTests, TestInequalOperator_ThrustComplex_StdComplex)
{
  using T = typename TestFixture::first_type;
  using U = typename TestFixture::second_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  run_inequality_operator_tests<T, U>([=](const thrust::complex<T>& lhs, const thrust::complex<U>& rhs) {
    std::complex<U> rhs_(rhs.real(), rhs.imag());
    return lhs != rhs_;
  });
}

TYPED_TEST(ComplexPairsTests, TestInequalOperator_StdComplex_ThrustComplex)
{
  using T = typename TestFixture::first_type;
  using U = typename TestFixture::second_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  run_inequality_operator_tests<T, U>([=](const thrust::complex<T>& lhs, const thrust::complex<U>& rhs) {
    std::complex<T> lhs_(lhs.real(), lhs.imag());
    return lhs_ != rhs;
  });
}

TYPED_TEST(ComplexPairsTests, TestInequalOperator_Scalar_ThrustComplex)
{
  using T = typename TestFixture::first_type;
  using U = typename TestFixture::second_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  run_inequality_operator_tests<T, U>(
    [=](const thrust::complex<T>& lhs, const thrust::complex<U>& rhs) {
      return lhs.real() != rhs;
    },
    true);
}

TYPED_TEST(ComplexPairsTests, TestInequalOperator_ThrustComplex_Scalar)
{
  using T = typename TestFixture::first_type;
  using U = typename TestFixture::second_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  run_inequality_operator_tests<T, U>(
    [=](const thrust::complex<T>& lhs, const thrust::complex<U>& rhs) {
      return lhs != rhs.real();
    },
    true);
}
