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

#include "test_imag_assertions.hpp"
#include "test_param_fixtures.hpp"
#include "test_utils.hpp"

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

TYPED_TEST(ComplexTests, TestComplexMemberOperators)
{
  using T = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  for (auto seed : get_seeds())
  {
    SCOPED_TRACE(testing::Message() << "with seed= " << seed);

    thrust::host_vector<T> data_a = get_random_data<T>(2, 10000, 10000, seed);

    thrust::host_vector<T> data_b = get_random_data<T>(2, 10000, 10000, seed + seed_value_addition);

    thrust::complex<T> a(data_a[0], data_a[1]);
    thrust::complex<T> b(data_b[0], data_b[1]);

    std::complex<T> c(a);
    std::complex<T> d(b);

    a += b;
    c += d;
    ASSERT_NEAR_COMPLEX(a, c);

    a -= b;
    c -= d;
    ASSERT_NEAR_COMPLEX(a, c);

    a *= b;
    c *= d;
    ASSERT_NEAR_COMPLEX(a, c);

    a /= b;
    c /= d;
    ASSERT_NEAR_COMPLEX(a, c);

    // casting operator
    c = (std::complex<T>) a;
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

    thrust::complex<T> a(data[0], data[1]);
    std::complex<T> b(a);

    // Test the basic arithmetic functions against std

    ASSERT_NEAR(abs(a), abs(b), T(0.01));

    ASSERT_NEAR(arg(a), arg(b), T(0.01));

    ASSERT_NEAR(norm(a), norm(b), T(0.01));

    ASSERT_EQ(conj(a), conj(b));

    ASSERT_NEAR_COMPLEX(thrust::polar(data[0], data[1]), std::polar(data[0], data[1]));

    // random_samples does not seem to produce infinities so proj(z) == z
    ASSERT_EQ(proj(a), a);
  }
}

TYPED_TEST(ComplexTests, TestComplexBinaryArithmetic)
{
  using T = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  for (auto seed : get_seeds())
  {
    SCOPED_TRACE(testing::Message() << "with seed= " << seed);

    thrust::host_vector<T> data_a = get_random_data<T>(2, -10000, 10000, seed);

    thrust::host_vector<T> data_b = get_random_data<T>(2, -10000, 10000, seed + seed_value_addition);

    thrust::complex<T> a(data_a[0], data_a[1]);
    thrust::complex<T> b(data_b[0], data_b[1]);

    ASSERT_NEAR_COMPLEX(a * b, std::complex<T>(a) * std::complex<T>(b));
    ASSERT_NEAR_COMPLEX(a * data_b[0], std::complex<T>(a) * data_b[0]);
    ASSERT_NEAR_COMPLEX(data_a[0] * b, data_a[0] * std::complex<T>(b));

    ASSERT_NEAR_COMPLEX(a / b, std::complex<T>(a) / std::complex<T>(b));
    ASSERT_NEAR_COMPLEX(a / data_b[0], std::complex<T>(a) / data_b[0]);
    ASSERT_NEAR_COMPLEX(data_a[0] / b, data_a[0] / std::complex<T>(b));

    ASSERT_EQ(a + b, std::complex<T>(a) + std::complex<T>(b));
    ASSERT_EQ(a + data_b[0], std::complex<T>(a) + data_b[0]);
    ASSERT_EQ(data_a[0] + b, data_a[0] + std::complex<T>(b));

    ASSERT_EQ(a - b, std::complex<T>(a) - std::complex<T>(b));
    ASSERT_EQ(a - data_b[0], std::complex<T>(a) - data_b[0]);
    ASSERT_EQ(data_a[0] - b, data_a[0] - std::complex<T>(b));
  }
}

TYPED_TEST(ComplexTests, TestComplexUnaryArithmetic)
{
  using T = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  for (auto seed : get_seeds())
  {
    SCOPED_TRACE(testing::Message() << "with seed= " << seed);

    thrust::host_vector<T> data_a =
      get_random_data<T>(2, get_default_limits<T>::min(), get_default_limits<T>::max(), seed);

    thrust::complex<T> a(data_a[0], data_a[1]);

    ASSERT_EQ(+a, +std::complex<T>(a));
    ASSERT_EQ(-a, -std::complex<T>(a));
  }
}

TYPED_TEST(ComplexTests, TestComplexExponentialFunctions)
{
  using T = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  for (auto seed : get_seeds())
  {
    SCOPED_TRACE(testing::Message() << "with seed= " << seed);

    thrust::host_vector<T> data_a = get_random_data<T>(2, -100, 100, seed);

    thrust::complex<T> a(data_a[0], data_a[1]);
    std::complex<T> b(a);

    ASSERT_NEAR_COMPLEX(exp(a), exp(b));
    ASSERT_NEAR_COMPLEX(log(a), log(b));
    ASSERT_NEAR_COMPLEX(log10(a), log10(b));
  }
}

TYPED_TEST(ComplexTests, TestComplexPowerFunctions)
{
  using T = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  for (auto seed : get_seeds())
  {
    SCOPED_TRACE(testing::Message() << "with seed= " << seed);

    thrust::host_vector<T> data_a = get_random_data<T>(2, -100, 100, seed);

    thrust::host_vector<T> data_b = get_random_data<T>(2, -100, 100, seed + seed_value_addition);

    thrust::complex<T> a(data_a[0], data_a[1]);
    thrust::complex<T> b(data_b[0], data_b[1]);
    std::complex<T> c(a);
    std::complex<T> d(b);

    ASSERT_NEAR_COMPLEX(pow(a, b), pow(c, d));
    ASSERT_NEAR_COMPLEX(pow(a, b.real()), pow(c, d.real()));
    ASSERT_NEAR_COMPLEX(pow(a.real(), b), pow(c.real(), d));

    ASSERT_NEAR_COMPLEX(sqrt(a), sqrt(c));
  }
}

TYPED_TEST(ComplexTests, TestComplexTrigonometricFunctions)
{
  using T = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  for (auto seed : get_seeds())
  {
    SCOPED_TRACE(testing::Message() << "with seed= " << seed);

    thrust::host_vector<T> data_a = get_random_data<T>(2, saturate_cast<T>(-1), saturate_cast<T>(1), seed);

    thrust::complex<T> a(data_a[0], data_a[1]);
    std::complex<T> c(a);

    ASSERT_NEAR_COMPLEX(cos(a), cos(c));
    ASSERT_NEAR_COMPLEX(sin(a), sin(c));
    ASSERT_NEAR_COMPLEX(tan(a), tan(c));
    ASSERT_NEAR_COMPLEX(cosh(a), cosh(c));
    ASSERT_NEAR_COMPLEX(sinh(a), sinh(c));
    ASSERT_NEAR_COMPLEX(tanh(a), tanh(c));

    ASSERT_NEAR_COMPLEX(acos(a), acos(c));
    ASSERT_NEAR_COMPLEX(asin(a), asin(c));
    ASSERT_NEAR_COMPLEX(atan(a), atan(c));
    ASSERT_NEAR_COMPLEX(acosh(a), acosh(c));
    ASSERT_NEAR_COMPLEX(asinh(a), asinh(c));
    ASSERT_NEAR_COMPLEX(atanh(a), atanh(c));
  }
}

TYPED_TEST(ComplexTests, TestComplexStreamOperators)
{
  using T = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  for (auto seed : get_seeds())
  {
    SCOPED_TRACE(testing::Message() << "with seed= " << seed);

    thrust::host_vector<T> data_a = get_random_data<T>(2, saturate_cast<T>(-1000), saturate_cast<T>(1000), seed);

    thrust::complex<T> a(data_a[0], data_a[1]);
    std::stringstream out;
    out << a;
    thrust::complex<T> b;
    out >> b;
    ASSERT_NEAR_COMPLEX(a, b);
  }
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
      using type = typename thrust::detail::promoted_numerical_type<T, U>::type;
      lhs        = std::complex<type>(lhs.real() + rhs.real(), lhs.imag() + rhs.imag());
    },
    [=](std::complex<T>& lhs, const U& rhs) {
      using type = typename thrust::detail::promoted_numerical_type<T, U>::type;
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
      using type = typename thrust::detail::promoted_numerical_type<T, U>::type;
      lhs        = std::complex<type>(lhs.real() - rhs.real(), lhs.imag() - rhs.imag());
    },
    [=](std::complex<T>& lhs, const U& rhs) {
      using type = typename thrust::detail::promoted_numerical_type<T, U>::type;
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
      using type = typename thrust::detail::promoted_numerical_type<T, U>::type;

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
      using type = typename thrust::detail::promoted_numerical_type<T, U>::type;

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
