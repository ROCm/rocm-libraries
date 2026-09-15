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

/**
 * test_real_assertions will contain assertions for normal numbers only,
 * complex assertions will be in a seperate file
 */

#pragma once

#include <thrust/host_vector.h>

#include <cfloat>
#include <cmath>

#include <gtest/gtest.h>

#include "test_param_fixtures.hpp"
#include "test_utils.hpp"

template <typename T1, typename T2>
testing::AssertionResult
CmpHelperEQQuite(const char* lhs_expression, const char* rhs_expression, const T1& lhs, const T2& rhs)
{
  if (lhs == rhs)
  {
    return testing::AssertionSuccess();
  }

  testing::Message msg;
  msg << "Expressions during equality check:";
  msg << "\n  " << lhs_expression;
  msg << "\n  " << rhs_expression;

  return testing::AssertionFailure() << msg;
}

#define ASSERT_EQ_QUIET(val1, val2) ASSERT_PRED_FORMAT2(CmpHelperEQQuite, val1, val2)

template <typename T1, typename T2>
testing::AssertionResult
CmpHelperNEQuite(const char* lhs_expression, const char* rhs_expression, const T1& lhs, const T2& rhs)
{
  if (lhs != rhs)
  {
    return testing::AssertionSuccess();
  }

  testing::Message msg;
  msg << "Expressions during equality check:";
  msg << "\n  " << lhs_expression;
  msg << "\n  " << rhs_expression;

  return testing::AssertionFailure() << msg;
}

#define ASSERT_NE_QUIET(val1, val2) ASSERT_PRED_FORMAT2(CmpHelperNEQuite, val1, val2)

template <typename T>
testing::AssertionResult bitwise_equal(const char* a_expr, const char* b_expr, const T& a, const T& b)
{
  if (std::memcmp(&a, &b, sizeof(T)) == 0)
  {
    return testing::AssertionSuccess();
  }

  // googletest's operator<< doesn't see the above overload for int128_t
  std::stringstream a_str;
  std::stringstream b_str;
  a_str << std::hexfloat << a;
  b_str << std::hexfloat << b;

  return testing::AssertionFailure()
      << "Expected strict/bitwise equality of these values: " << std::endl
      << "     " << a_expr << ": " << std::hexfloat << a_str.str() << std::endl
      << "     " << b_expr << ": " << std::hexfloat << b_str.str() << std::endl;
}

#define ASSERT_BITWISE_EQ(a, b) ASSERT_PRED_FORMAT2(bitwise_equal, a, b)

template <typename IterA, typename IterB>
void assert_bit_eq(IterA result_begin, IterA result_end, IterB expected_begin, IterB expected_end)
{
  using value_a_t = typename std::iterator_traits<IterA>::value_type;
  using value_b_t = typename std::iterator_traits<IterB>::value_type;

  ASSERT_EQ(std::distance(result_begin, result_end), std::distance(expected_begin, expected_end));
  auto result_it   = result_begin;
  auto expected_it = expected_begin;
  for (size_t index = 0; result_it != result_end; ++result_it, ++expected_it, ++index)
  {
    // The cast is needed, because the argument can be an std::vector<bool> iterator, which's operator*
    // returns a proxy object that must be converted to bool
    const auto result   = static_cast<value_a_t>(*result_it);
    const auto expected = static_cast<value_b_t>(*expected_it);

    ASSERT_BITWISE_EQ(result, expected) << "where index = " << index;
  }
}

template <typename T>
void assert_bit_eq(const thrust::host_vector<T>& result, const thrust::host_vector<T>& expected)
{
  assert_bit_eq(result.begin(), result.end(), expected.begin(), expected.end());
}

#define ASSERT_THROWS_EQ_WITH_FILE_AND_LINE(EXPR, EXCEPTION_TYPE, VALUE, FILE_, LINE_)                  \
  {                                                                                                     \
    threw_status THRUST_PP_CAT2(__s, LINE_) = did_not_throw;                                            \
    try                                                                                                 \
    {                                                                                                   \
      EXPR;                                                                                             \
    }                                                                                                   \
    catch (EXCEPTION_TYPE const& THRUST_PP_CAT2(__e, LINE_))                                            \
    {                                                                                                   \
      if (VALUE == THRUST_PP_CAT2(__e, LINE_))                                                          \
        THRUST_PP_CAT2(__s, LINE_) = threw_right_type;                                                  \
      else                                                                                              \
        THRUST_PP_CAT2(__s, LINE_) = threw_right_type_but_wrong_value;                                  \
    }                                                                                                   \
    catch (...)                                                                                         \
    {                                                                                                   \
      THRUST_PP_CAT2(__s, LINE_) = threw_wrong_type;                                                    \
    }                                                                                                   \
    check_assert_throws(THRUST_PP_CAT2(__s, LINE_), THRUST_PP_STRINGIZE(EXCEPTION_TYPE), FILE_, LINE_); \
  }

#define ASSERT_THROWS_EQ(EXPR, EXCEPTION_TYPE, VALUE) \
  ASSERT_THROWS_EQ_WITH_FILE_AND_LINE(EXPR, EXCEPTION_TYPE, VALUE, __FILE__, __LINE__)
