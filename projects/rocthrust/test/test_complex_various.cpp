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
#include <random>

#include "test_header.hpp"

#define CHECK_CORRECT(std_complex, thrust_complex, real_eps, imag_eps)  \
  do                                                                    \
  {                                                                     \
    if (std::isinf(std_complex.real()))                                 \
    {                                                                   \
      ASSERT_TRUE(std::isinf(thrust_complex.real()));                   \
    }                                                                   \
    else if (std::isnan(std_complex.real()))                            \
    {                                                                   \
      ASSERT_TRUE(std::isnan(thrust_complex.real()));                   \
    }                                                                   \
    else                                                                \
    {                                                                   \
      ASSERT_NEAR(std_complex.real(), thrust_complex.real(), real_eps); \
    }                                                                   \
    if (std::isinf(std_complex.imag()))                                 \
    {                                                                   \
      ASSERT_TRUE(std::isinf(thrust_complex.imag()));                   \
    }                                                                   \
    else if (std::isnan(std_complex.imag()))                            \
    {                                                                   \
      ASSERT_TRUE(std::isnan(thrust_complex.imag()));                   \
    }                                                                   \
    else                                                                \
    {                                                                   \
      ASSERT_NEAR(std_complex.imag(), thrust_complex.imag(), imag_eps); \
    }                                                                   \
  } while (0)

// This test suite aims to test vairous different implementations
// in the thrust/detail/complex directory

using FloatDouble = ::testing::Types<float, double>;

template <class Type>
class c99MathTest : public ::testing::Test
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
void run_trig_tests(const StdFunc& std_func, const ThrustFunc& thrust_func, const T mini = -1e5, T maxi = 1e5)
{
  constexpr size_t test_size = 100000;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution <T> dis(mini, maxi);

  for (size_t i = 0; i < test_size; i++)
  {
    T real = dis(gen);
    T imag = dis(gen);

    thrust::complex <T> thrust_complex(real, imag);
    std::complex <T> std_complex(real, imag);

    thrust::complex <T> thrust_out = thrust_func(thrust_complex);
    std::complex <T> std_out       = std_func(std_complex);

    T real_eps = (std::abs(std_out.real()) < 0.05) ? 1e-1 : std::abs(std_out.real() * 0.01);
    T imag_eps = (std::abs(std_out.imag()) < 0.05) ? 1e-1 : std::abs(std_out.imag() * 0.01);

    CHECK_CORRECT(std_out, thrust_out, real_eps, imag_eps);
  }
}

// ===================================================================
//
//                            C99 MATH
//
// ===================================================================

TYPED_TEST_SUITE(c99MathTest, FloatDouble);
TYPED_TEST(c99MathTest, getInf)
{
  using T = TestFixture::T;

  T inf = thrust::detail::complex::infinity<T>();
  ASSERT_TRUE(std::isinf(inf));
}

#if defined _MSC_VER

TYPED_TEST(c99MathTest, isinf)
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

TYPED_TEST(c99MathTest, isnan)
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

TYPED_TEST(c99MathTest, signbit)
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

TYPED_TEST(c99MathTest, isfinite)
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

TEST(c99MathTest, copysign)
{
  run_rng_test<double, false>(
    [](const double& x, const double& y) {
      return std::copysign(x, y);
    },
    [](const double& x, const double& y) {
      return thrust::detail::complex::copysign(x, y);
    });
}

TEST(c99MathTest, copysignf)
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

TYPED_TEST(c99MathTest, log1p)
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

TYPED_TEST(c99MathTest, log1pf)
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

TEST(c99MathTest, hypot)
{
  run_rng_test<double, false>(
    [](const double& x, const double& y) {
      return std::hypot(x, y);
    },
    [](const double& x, const double& y) {
      return thrust::detail::complex::hypot(x, y);
    });
}

TEST(c99MathTest, hypotf)
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

// ===================================================================
//
//                             CATRIG
//
// ===================================================================

TEST(catrigTest, asinh_small_range)
{
  run_trig_tests<double>(
    [](std::complex<double>& x) {
      return std::asinh(x);
    },
    [](thrust::complex<double>& x) {
      return thrust::asinh(x);
    });
}

TEST(catrigTest, asin_small_range)
{
  run_trig_tests<double>(
    [](std::complex<double>& x) {
      return std::asin(x);
    },
    [](thrust::complex<double>& x) {
      return thrust::asin(x);
    });
}

TEST(catrigTest, acos_small_range)
{
  run_trig_tests<double>(
    [](std::complex<double>& x) {
      return std::acos(x);
    },
    [](thrust::complex<double>& x) {
      return thrust::acos(x);
    });
}

TEST(catrigTest, acosh_small_range)
{
  run_trig_tests<double>(
    [](std::complex<double>& x) {
      return std::acosh(x);
    },
    [](thrust::complex<double>& x) {
      return thrust::acosh(x);
    });
}

TEST(catrigTest, atan_small_range)
{
  run_trig_tests<double>(
    [](std::complex<double>& x) {
      return std::atan(x);
    },
    [](thrust::complex<double>& x) {
      return thrust::atan(x);
    });
}

TEST(catrigTest, atanh_small_range)
{
  run_trig_tests<double>(
    [](std::complex<double>& x) {
      return std::atanh(x);
    },
    [](thrust::complex<double>& x) {
      return thrust::atanh(x);
    });
}

TEST(catrigTest, asinh_large_range)
{
  run_trig_tests<double>(
    [](std::complex<double>& x) {
      return std::asinh(x);
    },
    [](thrust::complex<double>& x) {
      return thrust::asinh(x);
    },
    std::numeric_limits<double>::min(),
    std::numeric_limits<double>::max());
}

TEST(catrigTest, asin_large_range)
{
  run_trig_tests<double>(
    [](std::complex<double>& x) {
      return std::asin(x);
    },
    [](thrust::complex<double>& x) {
      return thrust::asin(x);
    },
    std::numeric_limits<double>::min(),
    std::numeric_limits<double>::max());
}

TEST(catrigTest, acos_large_range)
{
  run_trig_tests<double>(
    [](std::complex<double>& x) {
      return std::acos(x);
    },
    [](thrust::complex<double>& x) {
      return thrust::acos(x);
    },
    std::numeric_limits<double>::min(),
    std::numeric_limits<double>::max());
}

TEST(catrigTest, acosh_large_range)
{
  run_trig_tests<double>(
    [](std::complex<double>& x) {
      return std::acosh(x);
    },
    [](thrust::complex<double>& x) {
      return thrust::acosh(x);
    },
    std::numeric_limits<double>::min(),
    std::numeric_limits<double>::max());
}

TEST(catrigTest, atan_large_range)
{
  run_trig_tests<double>(
    [](std::complex<double>& x) {
      return std::atan(x);
    },
    [](thrust::complex<double>& x) {
      return thrust::atan(x);
    },
    std::numeric_limits<double>::min(),
    std::numeric_limits<double>::max());
}

TEST(catrigTest, atanh_large_range)
{
  run_trig_tests<double>(
    [](std::complex<double>& x) {
      return std::atanh(x);
    },
    [](thrust::complex<double>& x) {
      return thrust::atanh(x);
    },
    std::numeric_limits<double>::min(),
    std::numeric_limits<double>::max());
}

// ===================================================================
//
//                            CATRIGF
//
// ===================================================================

TEST(catrigTest, asinhf_small_range)
{
  run_trig_tests<float>(
    [](std::complex<float>& x) {
      return std::asinh(x);
    },
    [](thrust::complex<float>& x) {
      return thrust::asinh(x);
    });
}

TEST(catrigTest, asinf_small_range)
{
  run_trig_tests<float>(
    [](std::complex<float>& x) {
      return std::asin(x);
    },
    [](thrust::complex<float>& x) {
      return thrust::asin(x);
    });
}

TEST(catrigTest, acosf_small_range)
{
  run_trig_tests<float>(
    [](std::complex<float>& x) {
      return std::acos(x);
    },
    [](thrust::complex<float>& x) {
      return thrust::acos(x);
    });
}

TEST(catrigTest, acoshf_small_range)
{
  run_trig_tests<float>(
    [](std::complex<float>& x) {
      return std::acosh(x);
    },
    [](thrust::complex<float>& x) {
      return thrust::acosh(x);
    });
}

TEST(catrigTest, atanf_small_range)
{
  run_trig_tests<float>(
    [](std::complex<float>& x) {
      return std::atan(x);
    },
    [](thrust::complex<float>& x) {
      return thrust::atan(x);
    });
}

TEST(catrigTest, atanhf_small_range)
{
  run_trig_tests<float>(
    [](std::complex<float>& x) {
      return std::atanh(x);
    },
    [](thrust::complex<float>& x) {
      return thrust::atanh(x);
    });
}

TEST(catrigTest, asinhf_large_range)
{
  run_trig_tests<float>(
    [](std::complex<float>& x) {
      return std::asinh(x);
    },
    [](thrust::complex<float>& x) {
      return thrust::asinh(x);
    },
    std::numeric_limits<float>::min(),
    std::numeric_limits<float>::max());
}

TEST(catrigTest, asinf_large_range)
{
  run_trig_tests<float>(
    [](std::complex<float>& x) {
      return std::asin(x);
    },
    [](thrust::complex<float>& x) {
      return thrust::asin(x);
    },
    std::numeric_limits<float>::min(),
    std::numeric_limits<float>::max());
}

TEST(catrigTest, acosf_large_range)
{
  run_trig_tests<float>(
    [](std::complex<float>& x) {
      return std::acos(x);
    },
    [](thrust::complex<float>& x) {
      return thrust::acos(x);
    },
    std::numeric_limits<float>::min(),
    std::numeric_limits<float>::max());
}

TEST(catrigTest, acoshf_large_range)
{
  run_trig_tests<float>(
    [](std::complex<float>& x) {
      return std::acosh(x);
    },
    [](thrust::complex<float>& x) {
      return thrust::acosh(x);
    },
    std::numeric_limits<float>::min(),
    std::numeric_limits<float>::max());
}

TEST(catrigTest, atanf_large_range)
{
  run_trig_tests<float>(
    [](std::complex<float>& x) {
      return std::atan(x);
    },
    [](thrust::complex<float>& x) {
      return thrust::atan(x);
    },
    std::numeric_limits<float>::min(),
    std::numeric_limits<float>::max());
}

TEST(catrigTest, atanhf_large_range)
{
  run_trig_tests<float>(
    [](std::complex<float>& x) {
      return std::atanh(x);
    },
    [](thrust::complex<float>& x) {
      return thrust::atanh(x);
    },
    std::numeric_limits<float>::min(),
    std::numeric_limits<float>::max());
}
