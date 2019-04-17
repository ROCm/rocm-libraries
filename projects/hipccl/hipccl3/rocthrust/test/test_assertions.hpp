/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *  Modifications Copyright© 2019 Advanced Micro Devices, Inc. All rights reserved.
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

// Google Test
#include <gtest/gtest.h>

// TODO: Fix the complex
//#include <thrust/complex.h>

template <typename T1, typename T2>
testing::AssertionResult CmpHelperEQQuite(const char* lhs_expression,
                                          const char* rhs_expression,
                                          const T1& lhs,
                                          const T2& rhs) {
  if (lhs == rhs) {
    return testing::AssertionSuccess();
  }

  testing::Message msg;
  msg << "Expressions during equality check:";
  msg << "\n  " << lhs_expression;
  msg << "\n  " << rhs_expression;

  return testing::AssertionFailure() << msg;
}

#define ASSERT_EQ_QUIET(val1, val2)\
  ASSERT_PRED_FORMAT2(CmpHelperEQQuite, \
                      val1, val2)

// TODO: Fix the complex
/*template <typename T>
testing::AssertionResult ComplexNearPredFormat(const char* expr1,
                                               const char* expr2,
                                               const char* abs_error_expr,
                                               thrust::complex<T> val1,
                                               thrust::complex<T> val2,
                                               thrust::complex<double> abs_error){
  const thrust::complex<double> diff(fabs((double)val1.real() - (double)val2.real())
                                    ,fabs((double)val1.imag() - (double)val2.imag()));
  if (diff.real() <= abs_error.real() &&
      diff.imag() <= abs_error.imag())
    return testing::AssertionSuccess();

  return testing::AssertionFailure()
      << "The difference between " << expr1 << " and " << expr2
      << " is " << diff << ", which exceeds " << abs_error_expr << ", where\n"
      << expr1 << " evaluates to " << val1 << ",\n"
      << expr2 << " evaluates to " << val2 << ", and\n"
      << abs_error_expr << " evaluates to " << abs_error << ".";
}

#define ASSERT_NEAR_COMPLEX_ERROR(val1, val2, abs_error)\
  ASSERT_PRED_FORMAT3(ComplexNearPredFormat<typename decltype(val1)::value_type>, \
                      val1, val2, abs_error)


#define ASSERT_NEAR_COMPLEX(val1, val2)\
  ASSERT_NEAR_COMPLEX_ERROR(val1, val2, thrust::complex<T>(std::numeric_limits<T>::epsilon(),std::numeric_limits<T>::epsilon()))
*/
