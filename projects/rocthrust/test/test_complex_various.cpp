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
