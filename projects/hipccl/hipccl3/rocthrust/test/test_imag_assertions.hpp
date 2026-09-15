// Copyright (C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#pragma once

#include <cfloat>
#include <cmath>

#include <gtest/gtest.h>

#include "test_param_fixtures.hpp"
#include "test_utils.hpp"

template <typename T>
testing::AssertionResult ComplexCompare(
  const char* expr1,
  const char* expr2,
  const char* abs_error_expr,
  thrust::complex<T> val1,
  thrust::complex<T> val2,
  thrust::complex<double> abs_error)
{
  double real_diff;
  if (std::isinf(val1.real()))
  {
    real_diff = std::isinf(val2.real()) ? 0.0 : std::numeric_limits<double>::infinity();
  }
  else if (std::isnan(val1.real()))
  {
    real_diff = std::isnan(val2.real()) ? 0.0 : std::numeric_limits<double>::infinity();
  }
  else
  {
    real_diff = fabs((double) val1.real() - (double) val2.real());
  }

  double imag_diff;
  if (std::isinf(val1.imag()))
  {
    imag_diff = std::isinf(val2.imag()) ? 0.0 : std::numeric_limits<double>::infinity();
  }
  else if (std::isnan(val1.imag()))
  {
    imag_diff = std::isnan(val2.imag()) ? 0.0 : std::numeric_limits<double>::infinity();
  }
  else
  {
    imag_diff = fabs((double) val1.imag() - (double) val2.imag());
  }

  if (real_diff == 0 && imag_diff == 0)
  {
    return testing::AssertionSuccess();
  }

  const thrust::complex<double> diff(real_diff, imag_diff);
  const thrust::complex<double> tol_diff(std::max(0.1 * (fabs(val1.real() + val2.real()) + abs_error.real()), 0.01),
                                         std::max(0.1 * (fabs(val1.imag() + val2.imag()) + abs_error.imag()), 0.01));

  if ((diff.real() != 0 && diff.real() > tol_diff.real()) || (diff.imag() != 0 && diff.imag() > tol_diff.imag()))
  {
    return testing::AssertionFailure()
        << "The difference between " << expr1 << " and " << expr2 << " is " << diff << ", which exceeds "
        << abs_error_expr << ", where\n"
        << expr1 << " evaluates to " << val1 << ",\n"
        << expr2 << " evaluates to " << val2 << ", and\n"
        << abs_error_expr << " evaluates to " << tol_diff << ".";
  }
  else
  {
    return testing::AssertionSuccess();
  }
}

template <typename T>
testing::AssertionResult ComplexNearPredFormat(
  const char* expr1,
  const char* expr2,
  const char* abs_error_expr,
  thrust::complex<T> val1,
  thrust::complex<T> val2,
  thrust::complex<double> abs_error)
{
  return ComplexCompare(expr1, expr2, abs_error_expr, val1, val2, abs_error);
}

template <typename T>
testing::AssertionResult ComplexVectorNearPredFormat(
  const char* expr1,
  const char* expr2,
  const char* abs_error_expr,
  thrust::host_vector<T> val1,
  thrust::device_vector<T> val2,
  thrust::complex<double> abs_error)
{
  thrust::host_vector<T> vector1(val1);
  thrust::host_vector<T> vector2(val2);

  if (vector1.size() != vector2.size())
  {
    return testing::AssertionFailure()
        << "The difference between " << expr1 << " and " << expr2 << " are the sizes: " << vector1.size() << ", "
        << vector2.size() << ".";
  }

  for (unsigned int i = 0; i < vector1.size(); i++)
  {
    testing::AssertionResult result = ComplexCompare(expr1, expr2, abs_error_expr, vector1[i], vector2[i], abs_error);
    if (testing::AssertionSuccess() != result)
    {
      return result;
    }
  }
  return testing::AssertionSuccess();
}

#define ASSERT_NEAR_COMPLEX_ERROR(val1, val2, abs_error) \
  ASSERT_PRED_FORMAT3(ComplexNearPredFormat<typename decltype(val1)::value_type>, val1, val2, abs_error)

#define ASSERT_NEAR_COMPLEX(val1, val2) \
  ASSERT_NEAR_COMPLEX_ERROR(            \
    val1, val2, thrust::complex<T>(std::numeric_limits<T>::epsilon(), std::numeric_limits<T>::epsilon()))

#define ASSERT_NEAR_COMPLEX_VECTOR_ERROR(val1, val2, abs_error) \
  ASSERT_PRED_FORMAT3(ComplexVectorNearPredFormat<typename decltype(val1)::value_type>, val1, val2, abs_error)

#define ASSERT_NEAR_COMPLEX_VECTOR(val1, val2) \
  ASSERT_NEAR_COMPLEX_VECTOR_ERROR(            \
    val1, val2, thrust::complex<T>(std::numeric_limits<T>::epsilon(), std::numeric_limits<T>::epsilon()))
