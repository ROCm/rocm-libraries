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

#include "test_header.hpp"

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

    if constexpr (SingleParam)
    {
      ASSERT_EQ(sf(r_num), tf(r_num));
    }
    else
    {
      T r_num_ = dis(gen);
      ASSERT_EQ(sf(r_num, r_num_), tf(r_num, r_num_));
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

  const T real_inc = (rmaxi - rmini) / test_size;
  const T imag_inc = (imaxi - imini) / test_size;

  for (double real = rmini; real <= rmaxi; real += real_inc)
  {
    for (double imag = imini; imag <= imaxi; imag += imag_inc)
    {
      thrust::complex<T> thrust_complex(real, imag);
      std::complex<T> std_complex(real, imag);

      thrust::complex<T> thrust_out = thrust_func(thrust_complex);
      std::complex<T> std_out       = std_func(std_complex);

      T real_eps = (std::abs(std_out.real()) < 0.05) ? 1e-1 : std::abs(std_out.real() * 0.01);
      T imag_eps = (std::abs(std_out.imag()) < 0.05) ? 1e-1 : std::abs(std_out.imag() * 0.01);

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
  using T = TestFixture::T;

  T inf = thrust::detail::complex::infinity<T>();
  ASSERT_TRUE(std::isinf(inf));
}

#if defined _MSC_VER

TYPED_TEST(VariousComplexTest, isinf)
{
  using T = TestFixture::T;
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
  using T = TestFixture::T;
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
  using T = TestFixture::T;
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
  using T = TestFixture::T;
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
  using T = TestFixture::T;
  run_rng_test<T>(
    [](const T& x) {
      return std::log1p(x);
    },
    [](const T& x) {
      return thrust::detail::complex::log1p(x);
    });
}

TYPED_TEST(VariousComplexTest, log1pf)
{
  using T = TestFixture::T;
  run_rng_test<T>(
    [](const T& x) {
      return std::log1pf(x);
    },
    [](const T& x) {
      return thrust::detail::complex::log1pf(x);
    });
}
#  endif // __HIP__

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

//TODO: CALCULATE ARC FUNCTIONS MANUALLY
TYPED_TEST(VariousComplexTest, asinh)
{
  using T = TestFixture::T;
  run_trig_tests<T>(
    [](std::complex<T>& x) {
      return std::asinh(x);
    },
    [](thrust::complex<T>& x) {
      return thrust::asinh(x);
    },
    -1,
    1,
    0,
    1);
}

TYPED_TEST(VariousComplexTest, asin)
{
  using T = TestFixture::T;
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
    1);
}

TYPED_TEST(VariousComplexTest, acosh)
{
  using T = TestFixture::T;
  run_trig_tests<T>(
    [](std::complex<T>& x) {
      return std::acosh(x);
    },
    [](thrust::complex<T>& x) {
      return thrust::acosh(x);
    },
    -1,
    1,
    0,
    1);
}

TYPED_TEST(VariousComplexTest, acos)
{
  using T = TestFixture::T;
  run_trig_tests<T>(
    [](std::complex<T>& x) {
      return std::acos(x);
    },
    [](thrust::complex<T>& x) {
      return thrust::acos(x);
    },
    -1,
    1,
    0,
    1);
}

TYPED_TEST(VariousComplexTest, atanh)
{
  using T = TestFixture::T;
  run_trig_tests<T>(
    [](std::complex<T>& x) {
      return std::atanh(x);
    },
    [](thrust::complex<T>& x) {
      return thrust::atanh(x);
    },
    -1,
    1,
    0,
    1);
}

TYPED_TEST(VariousComplexTest, atan)
{
  using T = TestFixture::T;
  run_trig_tests<T>(
    [](std::complex<T>& x) {
      return std::atan(x);
    },
    [](thrust::complex<T>& x) {
      return thrust::atan(x);
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
  using T = TestFixture::T;
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
  using T = TestFixture::T;
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
  using T = TestFixture::T;
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

TYPED_TEST(VariousComplexTest, log)
{
  using T     = TestFixture::T;
  T max_range = std::sqrt(std::numeric_limits<T>::max() / 5);
  run_trig_tests<T>(
    [](std::complex<T>& x) {
      double t_real = x.real();
      double t_imag = x.imag();

      double r     = std::sqrt(std::pow(t_real, 2) + std::pow(t_imag, 2));
      double theta = std::atan2(t_imag, t_real);

      double real = std::log(r);
      double imag = theta;

      return std::complex<T>((T) real, (T) imag);
    },
    [](thrust::complex<T>& x) {
      return thrust::log(x);
    },
    -max_range,
    max_range,
    -max_range,
    max_range);
}

TYPED_TEST(VariousComplexTest, log10)
{
  using T     = TestFixture::T;
  T max_range = std::sqrt(std::numeric_limits<T>::max() / 5);
  run_trig_tests<T>(
    [](std::complex<T>& x) {
      double t_real = x.real();
      double t_imag = x.imag();

      double r     = std::sqrt(std::pow(t_real, 2) + std::pow(t_imag, 2));
      double theta = std::atan2(t_imag, t_real);

      double real = std::log10(r);
      double imag = std::log10(std::exp(1.0)) * theta;

      return std::complex<T>((T) real, (T) imag);
    },
    [](thrust::complex<T>& x) {
      return thrust::log10(x);
    },
    -max_range,
    max_range,
    -max_range,
    max_range);
}

// // ===================================================================
// //
// //                            CSIN & CSINF
// //
// // ===================================================================

TYPED_TEST(VariousComplexTest, sinh)
{
  using T = TestFixture::T;
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
  using T = TestFixture::T;
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
  using T = TestFixture::T;

  T max_range = std::sqrt(std::numeric_limits<T>::max() / 5);

  run_trig_tests<T>(
    [](std::complex<T>& x) {
      // sqrt(a + bi) = sqrt(a + sqrt(a^2 + b^2)/2) + i * ((b / |b|) * sqrt((-a + sqrt(a^2 + b^2))/2))

      double t_real = (double) x.real();
      double t_imag = (double) x.imag();

      double sqrt_portion = std::sqrt(std::pow(t_real, 2) + std::pow(t_imag, 2));

      double real = std::sqrt((t_real + sqrt_portion) / 2);
      double imag = (t_imag < 0 ? -1 : 1) * std::sqrt((-t_real + sqrt_portion) / 2);

      return std::complex<T>((T) real, (T) imag);
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
  using T = TestFixture::T;
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
  using T = TestFixture::T;
  run_trig_tests<T>(
    [](std::complex<T>& x) {
      // tan(a + bi) = sin(2a) / (cos(b) + cosh(2a) + isinh(2b) / (cos(b) + cosh(2a)

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
