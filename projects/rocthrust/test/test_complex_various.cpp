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

#include <cmath>
#include <complex>
#include <random>

#include <gtest/gtest.h>

#ifndef M_PI
#  define M_PI 3.1415926535897932
#endif

#define CHECK_CORRECT(std_complex, thrust_complex, real_eps, imag_eps, real_val, imag_val)                   \
  do                                                                                                         \
  {                                                                                                          \
    SCOPED_TRACE(                                                                                            \
      testing::Message()                                                                                     \
      << std::endl                                                                                           \
      << "Input Real Value: " << real_val << " Input Imaginary Value: " << imag_val << std::endl             \
      << "Std Output Real Value: " << std_complex.real() << " Std Output Imag Value: " << std_complex.imag() \
      << std::endl                                                                                           \
      << "Thrust Output Real Value: " << thrust_complex.real()                                               \
      << " Thrust Output Imag Value: " << thrust_complex.imag() << std::endl);                               \
    if (std::isinf(std_complex.real()))                                                                      \
    {                                                                                                        \
      ASSERT_TRUE(std::isinf(thrust_complex.real()));                                                        \
    }                                                                                                        \
    else if (std::isnan(std_complex.real()))                                                                 \
    {                                                                                                        \
      ASSERT_TRUE(std::isnan(thrust_complex.real()));                                                        \
    }                                                                                                        \
    else                                                                                                     \
    {                                                                                                        \
      ASSERT_NEAR(std_complex.real(), thrust_complex.real(), real_eps);                                      \
    }                                                                                                        \
    if (std::isinf(std_complex.imag()))                                                                      \
    {                                                                                                        \
      ASSERT_TRUE(std::isinf(thrust_complex.imag()));                                                        \
    }                                                                                                        \
    else if (std::isnan(std_complex.imag()))                                                                 \
    {                                                                                                        \
      ASSERT_TRUE(std::isnan(thrust_complex.imag()));                                                        \
    }                                                                                                        \
    else                                                                                                     \
    {                                                                                                        \
      ASSERT_NEAR(std_complex.imag(), thrust_complex.imag(), imag_eps);                                      \
    }                                                                                                        \
  } while (0)

#define GET_EPS(x) ((std::abs(x) < 0.05) ? 1e-1 : std::abs(x) * 0.05)

// This test suite aims to test vairous different implementations
// in the thrust/detail/complex directory

using FloatDouble = ::testing::Types<float, double>;

template <class Type>
class VariousComplexTest : public ::testing::Test
{
public:
  using T = Type;
};

template <typename T, bool SingleParam, class StdFunc, class ThrustFunc>
void run_rng_test(const StdFunc& sf, const ThrustFunc& tf)
{
  constexpr size_t test_size = 10000;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<T> dis(std::numeric_limits<T>::min(), std::numeric_limits<T>::max());

  for (size_t i = 0; i < test_size; i++)
  {
    T r_num = dis(gen);
    T expected, actual;
    if constexpr (SingleParam)
    {
      expected = sf(r_num);
      actual   = tf(r_num);
    }
    else
    {
      T r_num_ = dis(gen);
      expected = sf(r_num, r_num_);
      actual   = tf(r_num, r_num_);
    }

    if (std::isinf(expected))
    {
      ASSERT_TRUE(std::isinf(actual));
    }
    else if (std::isnan(expected))
    {
      ASSERT_TRUE(std::isnan(actual));
    }
    else
    {
      ASSERT_NEAR(expected, actual, expected * 0.01);
    }
  }
}

template <typename T, class StdFunc, class ThrustFunc>
void run_trig_tests(
  const StdFunc& std_func, const ThrustFunc& thrust_func, double rmini, double rmaxi, double imini, double imaxi)
{
  // To run N tests but with N^2 algorithim, we must sqrt(N)
  // so that total tests is still N
  const double test_size = std::sqrt(100000);

  const double real_inc = (rmaxi - rmini) / test_size;
  const double imag_inc = (imaxi - imini) / test_size;

  for (double real = rmini; real <= rmaxi; real += real_inc)
  {
    for (double imag = imini; imag <= imaxi; imag += imag_inc)
    {
      thrust::complex<T> thrust_complex(real, imag);
      std::complex<T> std_complex(real, imag);

      thrust::complex<T> thrust_out = thrust_func(thrust_complex);
      std::complex<T> std_out       = std_func(std_complex);

      T real_eps = GET_EPS(std_out.real());
      T imag_eps = GET_EPS(std_out.imag());

      CHECK_CORRECT(std_out, thrust_out, real_eps, imag_eps, real, imag);
    }
  }
}

// ===================================================================
//
//                            C99 MATH
//
// ===================================================================

TYPED_TEST_SUITE(VariousComplexTest, FloatDouble);
TYPED_TEST(VariousComplexTest, getInf)
{
  using T = typename TestFixture::T;

  T inf = thrust::detail::complex::infinity<T>();
  ASSERT_TRUE(std::isinf(inf));
}

#if defined _MSC_VER

TYPED_TEST(VariousComplexTest, isinf)
{
  using T = typename TestFixture::T;
  T inf   = thrust::detail::complex::infinity<T>();
  ASSERT_EQ(std::isinf(inf), thrust::detail::complex::isinf(inf));

  run_rng_test<T, true>(
    [](const T& x) {
      return std::isinf(x);
    },
    [](const T& x) {
      return thrust::detail::complex::isinf(x);
    });
}

TYPED_TEST(VariousComplexTest, isnan)
{
  using T = typename TestFixture::T;
  T nan   = std::numeric_limits<T>::quiet_NaN();
  ASSERT_EQ(std::isnan(nan), thrust::detail::complex::isnan(nan));

  run_rng_test<T, true>(
    [](const T& x) {
      return std::isnan(x);
    },
    [](const T& x) {
      return thrust::detail::complex::isnan(x);
    });
}

TYPED_TEST(VariousComplexTest, signbit)
{
  using T = typename TestFixture::T;
  run_rng_test<T, true>(
    [](const T& x) {
      return std::signbit(x);
    },
    [](const T& x) {
      return thrust::detail::complex::signbit(x);
    });
}

TYPED_TEST(VariousComplexTest, isfinite)
{
  using T = typename TestFixture::T;
  run_rng_test<T, true>(
    [](const T& x) {
      return std::isfinite(x);
    },
    [](const T& x) {
      return thrust::detail::complex::isfinite(x);
    });
}

TEST(VariousComplexTest, copysign)
{
  run_rng_test<double, false>(
    [](const double& x, const double& y) {
      return std::copysign(x, y);
    },
    [](const double& x, const double& y) {
      return thrust::detail::complex::copysign(x, y);
    });
}

TEST(VariousComplexTest, copysignf)
{
  run_rng_test<float, false>(
    [](const float x, const float y) {
      return std::copysignf(x, y);
    },
    [](const float x, const float y) {
      return thrust::detail::complex::copysignf(x, y);
    });
}

#  if !defined(__CUDACC__) && !defined(_NVHPC_CUDA)

TYPED_TEST(VariousComplexTest, log1p)
{
  using T = typename TestFixture::T;
  run_rng_test<T, true>(
    [](const T& x) {
      return std::log1p(x);
    },
    [](const T& x) {
      return thrust::detail::complex::log1p(x);
    });
}

TYPED_TEST(VariousComplexTest, log1pf)
{
  using T = typename TestFixture::T;
  run_rng_test<T, true>(
    [](const T& x) {
      return std::log1pf(x);
    },
    [](const T& x) {
      return thrust::detail::complex::log1pf(x);
    });
}
#  endif

#  if _MSC_VER <= 1500 && !defined(__clang__)

TEST(VariousComplexTest, hypot)
{
  run_rng_test<double, false>(
    [](const double& x, const double& y) {
      return std::hypot(x, y);
    },
    [](const double& x, const double& y) {
      return thrust::detail::complex::hypot(x, y);
    });
}

TEST(VariousComplexTest, hypotf)
{
  run_rng_test<float, false>(
    [](const float& x, const float& y) {
      return std::hypotf(x, y);
    },
    [](const float& x, const float& y) {
      return thrust::detail::complex::hypotf(x, y);
    });
}

#  endif // _MSC_VER <= 1500

#endif

/**
 * Due to hipcc compiler issue, std trig functions for
 * complex number is not working properly. To get
 * around this we will have to calculate the real and
 * imaginary parts separately manually.
 */

// ===================================================================
//
//                          CATRIG & CATRIGF
//
// ===================================================================

#ifndef _WIN32

TYPED_TEST(VariousComplexTest, asinh)
{
  /**
   * Due to hipcc compiler issues, negative range for the real is not working
   * for std::asinh.
   */

  using T = typename TestFixture::T;
  run_trig_tests<T>(
    [](std::complex<T>& x) {
      return std::asinh(x);
    },
    [](thrust::complex<T>& x) {
      return thrust::asinh(x);
    },
    0,
    10000,
    -1,
    1);
}

TYPED_TEST(VariousComplexTest, asinh_special)
{
  // test special cases for asinh
  using T = typename TestFixture::T;

  T qNan   = std::numeric_limits<T>::quiet_NaN();
  T posInf = std::numeric_limits<T>::infinity();

  thrust::complex<T> zero_zero(0, 0);
  thrust::complex<T> out = thrust::asinh(zero_zero);
  ASSERT_EQ(0, out.real());
  ASSERT_EQ(0, out.imag());

  thrust::complex<T> x_inf(1000, posInf);
  out = thrust::asinh(x_inf);
  ASSERT_TRUE(std::isinf(out.real()));
  ASSERT_NEAR(M_PI / 2, out.imag(), M_PI / 2 * 0.001);

  thrust::complex<T> x_nan(1000, qNan);
  out = thrust::asinh(x_nan);
  ASSERT_TRUE(std::isnan(out.real()));
  ASSERT_TRUE(std::isnan(out.imag()));

  thrust::complex<T> inf_x(posInf, 10000);
  out = thrust::asinh(inf_x);
  ASSERT_TRUE(std::isinf(out.real()));
  ASSERT_EQ(0, out.imag());

  thrust::complex<T> inf_inf(posInf, posInf);
  out = thrust::asinh(inf_inf);
  ASSERT_TRUE(std::isinf(out.real()));
  ASSERT_NEAR(M_PI / 4, out.imag(), M_PI / 4 * 0.001);

  thrust::complex<T> inf_nan(posInf, qNan);
  out = thrust::asinh(inf_nan);
  ASSERT_TRUE(std::isinf(out.real()));
  ASSERT_TRUE(std::isnan(out.imag()));

  thrust::complex<T> nan_zero(qNan, 0);
  out = thrust::asinh(nan_zero);
  ASSERT_TRUE(std::isnan(out.real()));
  ASSERT_EQ(0, out.imag());

  thrust::complex<T> nan_x(qNan, 1000);
  out = thrust::asinh(nan_x);
  ASSERT_TRUE(std::isnan(out.real()));
  ASSERT_TRUE(std::isnan(out.imag()));

  thrust::complex<T> nan_inf(qNan, posInf);
  out = thrust::asinh(nan_inf);
  ASSERT_TRUE(std::isinf(out.real()));
  ASSERT_TRUE(std::isnan(out.imag()));

  thrust::complex<T> nan_nan(qNan, qNan);
  out = thrust::asinh(nan_nan);
  ASSERT_TRUE(std::isnan(out.real()));
  ASSERT_TRUE(std::isnan(out.imag()));
}

TYPED_TEST(VariousComplexTest, asin)
{
  /**
   * Due to hipcc compiler issues, negative range for the real and imaginary is not working
   * for std::asin. Also a poistive range greater than 1000 for the imaginary cause issues too
   */

  using T = typename TestFixture::T;

  run_trig_tests<T>(
    [](std::complex<T>& x) {
      return std::asin(x);
    },
    [](thrust::complex<T>& x) {
      return thrust::asin(x);
    },
    -1,
    1,
    0,
    1000);
}

TYPED_TEST(VariousComplexTest, acosh)
{
  using T     = typename TestFixture::T;
  T max_range = std::sqrt(std::numeric_limits<T>::max());

  run_trig_tests<T>(
    [](std::complex<T>& x) {
      return std::acosh(x);
    },
    [](thrust::complex<T>& x) {
      return thrust::acosh(x);
    },
    1,
    max_range,
    -max_range,
    max_range);
}

TYPED_TEST(VariousComplexTest, acosh_special)
{
  // test special cases for acosh
  using T = typename TestFixture::T;

  T qNan   = std::numeric_limits<T>::quiet_NaN();
  T posInf = std::numeric_limits<T>::infinity();
  T negInf = -std::numeric_limits<T>::infinity();

  thrust::complex<T> zero_zero(0, 0);
  thrust::complex<T> out = thrust::acosh(zero_zero);
  ASSERT_EQ(0, out.real());
  ASSERT_NEAR(M_PI * 0.5, out.imag(), M_PI * 0.005);

  thrust::complex<T> x_inf(1000, posInf);
  out = thrust::acosh(x_inf);
  ASSERT_TRUE(std::isinf(out.real()));
  ASSERT_NEAR(M_PI / 2, out.imag(), M_PI / 2 * 0.001);

  thrust::complex<T> x_nan(1000, qNan);
  out = thrust::acosh(x_nan);
  ASSERT_TRUE(std::isnan(out.real()));
  ASSERT_TRUE(std::isnan(out.imag()));

  thrust::complex<T> neg_inf_x(negInf, 10000);
  out = thrust::acosh(neg_inf_x);
  ASSERT_TRUE(std::isinf(out.real()));
  ASSERT_NEAR(M_PI, out.imag(), M_PI * 0.001);

  thrust::complex<T> inf_x(posInf, 10000);
  out = thrust::acosh(inf_x);
  ASSERT_TRUE(std::isinf(out.real()));
  ASSERT_EQ(0, out.imag());

  thrust::complex<T> neg_inf_pos_inf(negInf, posInf);
  out = thrust::acosh(neg_inf_pos_inf);
  ASSERT_EQ(posInf, out.real());
  ASSERT_NEAR(M_PI * 0.75, out.imag(), M_PI * 0.0075);

  thrust::complex<T> pos_inf_nan(posInf, qNan);
  out = thrust::acosh(pos_inf_nan);
  ASSERT_EQ(posInf, out.real());
  ASSERT_TRUE(std::isnan(out.imag()));

  thrust::complex<T> neg_inf_nan(negInf, qNan);
  out = thrust::acosh(neg_inf_nan);
  ASSERT_EQ(posInf, out.real());
  ASSERT_TRUE(std::isnan(out.imag()));

  thrust::complex<T> nan_x(qNan, 1000);
  out = thrust::acosh(nan_x);
  ASSERT_TRUE(std::isnan(out.real()));
  ASSERT_TRUE(std::isnan(out.imag()));

  thrust::complex<T> nan_inf(qNan, posInf);
  out = thrust::acosh(nan_inf);
  ASSERT_EQ(posInf, out.real());
  ASSERT_TRUE(std::isnan(out.imag()));

  thrust::complex<T> nan_nan(qNan, qNan);
  out = thrust::acosh(nan_nan);
  ASSERT_TRUE(std::isnan(out.real()));
  ASSERT_TRUE(std::isnan(out.imag()));
}

TYPED_TEST(VariousComplexTest, acos)
{
  using T = typename TestFixture::T;
  run_trig_tests<T>(
    [](std::complex<T>& x) {
      return std::acos(x);
    },
    [](thrust::complex<T>& x) {
      return thrust::acos(x);
    },
    -1,
    1,
    -1000,
    1000);
}

TYPED_TEST(VariousComplexTest, acos_special)
{
  // test special cases for acos
  using T = typename TestFixture::T;

  T qNan   = std::numeric_limits<T>::quiet_NaN();
  T posInf = std::numeric_limits<T>::infinity();
  T negInf = -std::numeric_limits<T>::infinity();

  thrust::complex<T> zero_zero(0, 0);
  thrust::complex<T> out = thrust::acos(zero_zero);
  ASSERT_NEAR(M_PI * 0.5, out.real(), M_PI * 0.005);
  ASSERT_EQ(0, out.imag());

  thrust::complex<T> zero_nan(0, qNan);
  out = thrust::acos(zero_nan);
  ASSERT_NEAR(M_PI * 0.5, out.real(), M_PI * 0.005);
  ASSERT_TRUE(std::isnan(out.imag()));

  thrust::complex<T> x_inf(1000, posInf);
  out = thrust::acos(x_inf);
  ASSERT_NEAR(M_PI * 0.5, out.real(), M_PI * 0.005);
  ASSERT_EQ(negInf, out.imag());

  thrust::complex<T> x_nan(1000, qNan);
  out = thrust::acos(x_nan);
  ASSERT_TRUE(std::isnan(out.real()));
  ASSERT_TRUE(std::isnan(out.imag()));

  thrust::complex<T> neg_inf_x(negInf, 10000);
  out = thrust::acos(neg_inf_x);
  ASSERT_NEAR(M_PI, out.real(), M_PI * 0.001);
  ASSERT_EQ(negInf, out.imag());

  thrust::complex<T> pos_inf_x(posInf, 10000);
  out = thrust::acos(pos_inf_x);
  ASSERT_EQ(0, out.real());
  ASSERT_EQ(negInf, out.imag());

  thrust::complex<T> neg_inf_pos_inf(negInf, posInf);
  out = thrust::acos(neg_inf_pos_inf);
  ASSERT_NEAR(M_PI * 0.75, out.real(), M_PI * 0.0075);
  ASSERT_EQ(negInf, out.imag());

  thrust::complex<T> pos_inf_pos_inf(posInf, posInf);
  out = thrust::acos(pos_inf_pos_inf);
  ASSERT_NEAR(M_PI * 0.25, out.real(), M_PI * 0.0025);
  ASSERT_EQ(negInf, out.imag());

  thrust::complex<T> pos_inf_nan(posInf, qNan);
  out = thrust::acos(pos_inf_nan);
  ASSERT_TRUE(std::isnan(out.real()));
  ASSERT_TRUE(std::isinf(out.imag()));

  thrust::complex<T> nan_x(qNan, 1000);
  out = thrust::acos(nan_x);
  ASSERT_TRUE(std::isnan(out.real()));
  ASSERT_TRUE(std::isnan(out.imag()));

  thrust::complex<T> nan_inf(qNan, posInf);
  out = thrust::acos(nan_inf);
  ASSERT_TRUE(std::isnan(out.real()));
  ASSERT_EQ(negInf, out.imag());

  thrust::complex<T> nan_nan(qNan, qNan);
  out = thrust::acos(nan_nan);
  ASSERT_TRUE(std::isnan(out.real()));
  ASSERT_TRUE(std::isnan(out.imag()));
}

TYPED_TEST(VariousComplexTest, atanh)
{
  using T = typename TestFixture::T;
  run_trig_tests<T>(
    [](std::complex<T>& x) {
      return std::atanh(x);
    },
    [](thrust::complex<T>& x) {
      return thrust::atanh(x);
    },
    -1,
    1,
    -1000,
    1000);
}

TYPED_TEST(VariousComplexTest, catanh)
{
  using T = typename TestFixture::T;
  run_trig_tests<T>(
    [](std::complex<T>& x) {
      return std::atanh(x);
    },
    [](thrust::complex<T>& x) {
      return thrust::detail::complex::catanh(x);
    },
    -1,
    1,
    -1000,
    1000);
}

TYPED_TEST(VariousComplexTest, atanh_special)
{
  // test special cases for atanh
  using T = typename TestFixture::T;

  T qNan   = std::numeric_limits<T>::quiet_NaN();
  T posInf = std::numeric_limits<T>::infinity();

  thrust::complex<T> zero_zero(0, 0);
  thrust::complex<T> out = thrust::atanh(zero_zero);
  ASSERT_EQ(0, out.real());
  ASSERT_EQ(0, out.imag());

  thrust::complex<T> zero_nan(0, qNan);
  out = thrust::atanh(zero_nan);
  ASSERT_TRUE(std::isnan(out.imag()));

  thrust::complex<T> one_zero(1, 0);
  out = thrust::atanh(one_zero);
  ASSERT_EQ(posInf, out.real());
  ASSERT_EQ(0, out.imag());

  thrust::complex<T> x_inf(1000, posInf);
  out = thrust::atanh(x_inf);
  ASSERT_EQ(0, out.real());
  ASSERT_NEAR(M_PI * 0.5, out.imag(), M_PI * 0.005);

  thrust::complex<T> x_nan(1000, qNan);
  out = thrust::atanh(x_nan);
  ASSERT_TRUE(std::isnan(out.real()));
  ASSERT_TRUE(std::isnan(out.imag()));

  thrust::complex<T> pos_inf_x(posInf, 10000);
  out = thrust::atanh(pos_inf_x);
  ASSERT_EQ(0, out.real());
  ASSERT_NEAR(M_PI * 0.5, out.imag(), M_PI * 0.005);

  thrust::complex<T> pos_inf_nan(posInf, qNan);
  out = thrust::atanh(pos_inf_nan);
  ASSERT_EQ(0, out.real());
  ASSERT_TRUE(std::isnan(out.imag()));

  thrust::complex<T> nan_x(qNan, 1000);
  out = thrust::atanh(nan_x);
  ASSERT_TRUE(std::isnan(out.real()));
  ASSERT_TRUE(std::isnan(out.imag()));

  thrust::complex<T> nan_inf(qNan, posInf);
  out = thrust::atanh(nan_inf);
  ASSERT_EQ(0, out.real());
  ASSERT_NEAR(M_PI * 0.5, out.imag(), M_PI * 0.005);

  thrust::complex<T> nan_nan(qNan, qNan);
  out = thrust::atanh(nan_nan);
  ASSERT_TRUE(std::isnan(out.real()));
  ASSERT_TRUE(std::isnan(out.imag()));
}

TYPED_TEST(VariousComplexTest, atan)
{
  using T = typename TestFixture::T;
  run_trig_tests<T>(
    [](std::complex<T>& x) {
      return std::atan(x);
    },
    [](thrust::complex<T>& x) {
      return thrust::atan(x);
    },
    -1,
    1,
    -1000,
    1000);
}

TYPED_TEST(VariousComplexTest, catan)
{
  using T = typename TestFixture::T;
  run_trig_tests<T>(
    [](std::complex<T>& x) {
      return std::atan(x);
    },
    [](thrust::complex<T>& x) {
      return thrust::detail::complex::catan(x);
    },
    -1,
    1,
    -1000,
    1000);
}

TEST(VariousComplexTest, acos_long_double)
{
  run_trig_tests<long double>(
    [](std::complex<long double>& x) {
      return std::acos(x);
    },
    [](thrust::complex<long double>& x) {
      return thrust::acos(x);
    },
    -1,
    1,
    0,
    1);
}

TEST(VariousComplexTest, asin_long_double)
{
  run_trig_tests<long double>(
    [](std::complex<long double>& x) {
      return std::asin(x);
    },
    [](thrust::complex<long double>& x) {
      return thrust::asin(x);
    },
    -1,
    1,
    0,
    1);
}

TEST(VariousComplexTest, atan_long_double)
{
  run_trig_tests<long double>(
    [](std::complex<long double>& x) {
      return std::atan(x);
    },
    [](thrust::complex<long double>& x) {
      return thrust::atan(x);
    },
    -1,
    1,
    0,
    1);
}

TEST(VariousComplexTest, acosh_long_double)
{
  run_trig_tests<long double>(
    [](std::complex<long double>& x) {
      return std::acosh(x);
    },
    [](thrust::complex<long double>& x) {
      return thrust::acosh(x);
    },
    1,
    100000,
    0,
    10000);
}

TEST(VariousComplexTest, asinh_long_double)
{
  run_trig_tests<long double>(
    [](std::complex<long double>& x) {
      return std::asinh(x);
    },
    [](thrust::complex<long double>& x) {
      return thrust::asinh(x);
    },
    -1,
    1,
    0,
    1);
}

TEST(VariousComplexTest, atanh_long_double)
{
  run_trig_tests<long double>(
    [](std::complex<long double>& x) {
      return std::atanh(x);
    },
    [](thrust::complex<long double>& x) {
      return thrust::atanh(x);
    },
    -1,
    1,
    0,
    1);
}

// ===================================================================
//
//                            CCOSH & CCOSHF
//
// ===================================================================

TYPED_TEST(VariousComplexTest, cosh)
{
  using T = typename TestFixture::T;
  run_trig_tests<T>(
    [](std::complex<T>& x) {
      // cos(a + bi) = cosh(a) * cos(b) + isinh(a) * sin(b)
      double real = std::cosh((double) x.real()) * std::cos((double) x.imag());
      double imag = std::sinh((double) x.real()) * std::sin((double) x.imag());

      return std::complex<T>((T) real, (T) imag);
    },
    [](thrust::complex<T>& x) {
      return thrust::cosh(x);
    },
    -710,
    710,
    std::numeric_limits<T>::min(),
    std::numeric_limits<T>::max());
  //-710 and 710 since cosh and sinh is quadratic and anything above 710 will result
  // in out of bounds even for double!
}

TYPED_TEST(VariousComplexTest, cos)
{
  using T = typename TestFixture::T;
  run_trig_tests<T>(
    [](std::complex<T>& x) {
      // cos(a + bi) = cos(a) * cosh(b) − isin(a) * sinh(b)
      double real = std::cos((double) x.real()) * std::cosh((double) x.imag());
      double imag = std::sin((double) x.real()) * std::sinh((double) x.imag());

      SCOPED_TRACE(testing::Message() << real << " " << imag << std::endl);

      return std::complex<T>((T) real, (T) -imag);
    },
    [](thrust::complex<T>& x) {
      return thrust::cos(x);
    },
    std::numeric_limits<T>::min(),
    std::numeric_limits<T>::max(),
    -710,
    710);
  //-710 and 710 since cosh and sinh is quadratic and anything above 710 will result
  // in out of bounds even for double!
}

// ===================================================================
//
//                            CEXP & CEXPF
//
// ===================================================================

TYPED_TEST(VariousComplexTest, exp)
{
  using T = typename TestFixture::T;
  run_trig_tests<T>(
    [](std::complex<T>& x) {
      return std::exp(x);
    },
    [](thrust::complex<T>& x) {
      return thrust::exp(x);
    },
    std::numeric_limits<T>::min(),
    std::numeric_limits<T>::max(),
    std::numeric_limits<T>::min(),
    std::numeric_limits<T>::max());
}

// // ===================================================================
// //
// //                            CLOG & CLOGF
// //
// // ===================================================================

template <typename DataType>
std::complex<DataType> stdLog(std::complex<DataType>& x)
{
  double t_real = x.real();
  double t_imag = x.imag();

  double r     = std::sqrt(std::pow(t_real, 2) + std::pow(t_imag, 2));
  double theta = std::atan2(t_imag, t_real);

  double real = std::log(r);
  double imag = theta;

  return std::complex<DataType>((DataType) real, (DataType) imag);
}

TYPED_TEST(VariousComplexTest, log)
{
  using T     = typename TestFixture::T;
  T max_range = std::sqrt(std::numeric_limits<T>::max() / 5);

  run_trig_tests<T>(
    stdLog<T>,
    [](thrust::complex<T>& x) {
      return thrust::log(x);
    },
    -max_range,
    max_range,
    -max_range,
    max_range);
}

TEST(VariousComplexTest, logf_special_cases)
{
  using T = float;

  auto run_test = [=](T real, T imag) {
    std::complex<T> std_in(real, imag);
    thrust::complex<T> thrust_in(real, imag);

    std::complex<T> std_out       = stdLog<T>(std_in);
    thrust::complex<T> thrust_out = thrust::log(thrust_in);

    T real_eps = GET_EPS(std_out.real());
    T imag_eps = GET_EPS(std_out.imag());

    CHECK_CORRECT(std_out, thrust_out, real_eps, imag_eps, real, imag);
  };

  union converted
  {
    float out;
    uint32_t in;
  } c;

  // Testing clogf for case of ay > 1e34
  run_test(2e34f, 3e34f);
  // checking cases where real is 1, and imaginary is less than 1
  run_test(1.0f, 5e-20f);
  run_test(1.0f, 0.5);

  // checking cases where real or imag is less than 1e-6
  run_test(5e-7f, 0.5);
  run_test(0.5f, 5e-7f);
  run_test(0.0, 0.5);
  run_test(0.5, 0.0);

  // checking cases where real or imag is greater than 1e6
  run_test(5e7f, 0.5);
  run_test(0.5f, 5e7f);

  c.in = 0x7f800000;
  run_test(c.out, 0.5); // inf
  run_test(0.5, c.out); // inf

  // checking case where r <= 0.7
  run_test(0.55, 0.55);

  // checking NAN cases
  c.in = 0xffc00001;
  run_test(c.out, c.out);
}

TYPED_TEST(VariousComplexTest, log10)
{
  using T     = typename TestFixture::T;
  T max_range = std::sqrt(std::numeric_limits<T>::max() / 5);

  auto std_log10 = [](std::complex<T>& x) {
    double t_real = x.real();
    double t_imag = x.imag();

    double r     = std::sqrt(std::pow(t_real, 2) + std::pow(t_imag, 2));
    double theta = std::atan2(t_imag, t_real);

    double real = std::log10(r);
    double imag = std::log10(std::exp(1.0)) * theta;

    return std::complex<T>((T) real, (T) imag);
  };

  run_trig_tests<T>(
    std_log10,
    [](thrust::complex<T>& x) {
      return thrust::log10(x);
    },
    -max_range,
    max_range,
    -max_range,
    max_range);

  run_trig_tests<T>(
    std_log10,
    [](thrust::complex<T>& x) {
      return thrust::log10(x);
    },
    -5,
    5,
    -5,
    5);
}

// // ===================================================================
// //
// //                            CSIN & CSINF
// //
// // ===================================================================

TYPED_TEST(VariousComplexTest, sinh)
{
  using T = typename TestFixture::T;
  run_trig_tests<T>(
    [](std::complex<T>& x) {
      /**
       * Due to hipcc compiler issue, std::sinh(complex)
       * is not working properly. Instead we
       * will have to calculate this manually
       */

      // sinh(a + bi) = sinh(a) * cos(b) + icosh(a) * sin(b)

      double real = std::sinh((double) x.real()) * std::cos((double) x.imag());
      double imag = std::cosh((double) x.real()) * std::sin((double) x.imag());

      return std::complex<T>((T) real, (T) imag);
    },
    [](thrust::complex<T>& x) {
      return thrust::sinh(x);
    },
    std::numeric_limits<T>::min(),
    std::numeric_limits<T>::max(),
    -710,
    710);
  //-710 and 710 since cosh and sinh is quadratic and anything above 710 will result
  // in out of bounds even for double!
}

TYPED_TEST(VariousComplexTest, sin)
{
  using T = typename TestFixture::T;
  run_trig_tests<T>(
    [](std::complex<T>& x) {
      // sin(a + bi) = sin(a) * cosh(b) + icos(a) * sinh(b)

      double real = std::sin((double) x.real()) * std::cosh((double) x.imag());
      double imag = std::cos((double) x.real()) * std::sinh((double) x.imag());

      return std::complex<T>((T) real, (T) imag);
    },
    [](thrust::complex<T>& x) {
      return thrust::sin(x);
    },
    std::numeric_limits<T>::min(),
    std::numeric_limits<T>::max(),
    -710,
    710);
  //-710 and 710 since cosh and sinh is quadratic and anything above 710 will result
  // in out of bounds even for double!
}

// // ===================================================================
// //
// //                         CSQRT & CSQRTF
// //
// // ===================================================================

TYPED_TEST(VariousComplexTest, sqrt)
{
  using T = typename TestFixture::T;

  T max_range = std::sqrt(std::numeric_limits<T>::max() / 5);

  run_trig_tests<T>(
    [](std::complex<T>& x) {
      return std::sqrt(x);
    },
    [](thrust::complex<T>& x) {
      return thrust::sqrt(x);
    },
    -max_range,
    max_range,
    -max_range,
    max_range);
}

// // ===================================================================
// //
// //                            CTAN & CTANF
// //
// // ===================================================================

TYPED_TEST(VariousComplexTest, tanh)
{
  using T = typename TestFixture::T;
  run_trig_tests<T>(
    [](std::complex<T>& x) {
      // tanh(a + bi) = sinh(2a) / (cosh(b) + cos(2a) + isin(2b) / (cosh(b) + cos(2a)

      double t_real = (double) x.real();
      double t_imag = (double) x.imag();

      double denom = std::cosh(2.0 * t_real) + std::cos(2 * t_imag);

      T real = std::sinh(2.0 * t_real) / denom;
      T imag = std::sin(2.0 * t_imag) / denom;

      return std::complex<T>((T) real, (T) imag);
    },
    [](thrust::complex<T>& x) {
      return thrust::tanh(x);
    },
    -710 / 2,
    710 / 2,
    std::numeric_limits<T>::min() / 2,
    std::numeric_limits<T>::max() / 2);
}

TYPED_TEST(VariousComplexTest, tan)
{
  using T = typename TestFixture::T;
  run_trig_tests<T>(
    [](std::complex<T>& x) {
      // tan(a + bi) = sin(2a) / (cos(b) + cosh(2a)) + isinh(2b) / (cos(b) + cosh(2a))

      double t_real = (double) x.real();
      double t_imag = (double) x.imag();

      double denom = std::cos(2.0 * t_real) + std::cosh(2 * t_imag);

      T real = std::sin(2.0 * t_real) / denom;
      T imag = std::sinh(2.0 * t_imag) / denom;

      return std::complex<T>((T) real, (T) imag);
    },
    [](thrust::complex<T>& x) {
      return thrust::tan(x);
    },
    std::numeric_limits<T>::min() / 2,
    std::numeric_limits<T>::max() / 2,
    -710 / 2,
    710 / 2);
  //-710 / 2 and 710 / 2 since cosh and sinh is quadratic and anything above 710 will result
  // in out of bounds even for double! We also have to acount the 2 multiplier);
}
#endif // _WIN32
