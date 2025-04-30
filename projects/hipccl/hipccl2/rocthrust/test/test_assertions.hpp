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
#include <thrust/host_vector.h>

#include <gtest/gtest.h>

#include <cfloat>
#include <cmath>

template <typename T1, typename T2>
testing::AssertionResult CmpHelperEQQuite(const char* lhs_expression,
                                          const char* rhs_expression,
                                          const T1&   lhs,
                                          const T2&   rhs)
{
    if(lhs == rhs)
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

template <typename T>
testing::AssertionResult ComplexCompare(const char*             expr1,
                                        const char*             expr2,
                                        const char*             abs_error_expr,
                                        thrust::complex<T>      val1,
                                        thrust::complex<T>      val2,
                                        thrust::complex<double> abs_error)
{

    double real_diff;
    if(std::isinf(val1.real()))
    {
        real_diff = std::isinf(val2.real()) ? 0.0 : std::numeric_limits<double>::infinity();
    }
    else if(std::isnan(val1.real()))
    {
        real_diff = std::isnan(val2.real()) ? 0.0 : std::numeric_limits<double>::infinity();
    }
    else
    {
        real_diff = fabs((double)val1.real() - (double)val2.real());
    }

    double imag_diff;
    if(std::isinf(val1.imag()))
    {
        imag_diff = std::isinf(val2.imag()) ? 0.0 : std::numeric_limits<double>::infinity();
    }
    else if(std::isnan(val1.imag()))
    {
        imag_diff = std::isnan(val2.imag()) ? 0.0 : std::numeric_limits<double>::infinity();
    }
    else
    {
        imag_diff = fabs((double)val1.imag() - (double)val2.imag());
    }

    if(real_diff == 0 && imag_diff == 0)
        return testing::AssertionSuccess();

    const thrust::complex<double> diff(real_diff, imag_diff);
    const thrust::complex<double> tol_diff(
        std::max( 0.1 * (fabs(val1.real() + val2.real()) + abs_error.real()), 0.01 ),
        std::max( 0.1 * (fabs(val1.imag() + val2.imag()) + abs_error.imag()), 0.01 ));

    if((diff.real() != 0 && diff.real() > tol_diff.real())
       || (diff.imag() != 0 && diff.imag() > tol_diff.imag()))
        return testing::AssertionFailure()
               << "The difference between " << expr1 << " and " << expr2 << " is " << diff
               << ", which exceeds " << abs_error_expr << ", where\n"
               << expr1 << " evaluates to " << val1 << ",\n"
               << expr2 << " evaluates to " << val2 << ", and\n"
               << abs_error_expr << " evaluates to " << tol_diff << ".";
    else
        return testing::AssertionSuccess();
}

template <typename T>
testing::AssertionResult ComplexNearPredFormat(const char*             expr1,
                                               const char*             expr2,
                                               const char*             abs_error_expr,
                                               thrust::complex<T>      val1,
                                               thrust::complex<T>      val2,
                                               thrust::complex<double> abs_error)
{

    return ComplexCompare(expr1, expr2, abs_error_expr, val1, val2, abs_error);
}

template <typename T>
testing::AssertionResult ComplexVectorNearPredFormat(const char*              expr1,
                                                     const char*              expr2,
                                                     const char*              abs_error_expr,
                                                     thrust::host_vector<T>   val1,
                                                     thrust::device_vector<T> val2,
                                                     thrust::complex<double>  abs_error)
{

    thrust::host_vector<T> vector1(val1);
    thrust::host_vector<T> vector2(val2);

    if(vector1.size() != vector2.size())
    {
        return testing::AssertionFailure()
               << "The difference between " << expr1 << " and " << expr2
               << " are the sizes: " << vector1.size() << ", " << vector2.size() << ".";
    }

    for(unsigned int i = 0; i < vector1.size(); i++)
    {
        testing::AssertionResult result
            = ComplexCompare(expr1, expr2, abs_error_expr, vector1[i], vector2[i], abs_error);
        if(testing::AssertionSuccess() != result)
            return result;
    }
    return testing::AssertionSuccess();
}

#define ASSERT_NEAR_COMPLEX_ERROR(val1, val2, abs_error) \
    ASSERT_PRED_FORMAT3(                                 \
        ComplexNearPredFormat<typename decltype(val1)::value_type>, val1, val2, abs_error)

#define ASSERT_NEAR_COMPLEX(val1, val2) \
    ASSERT_NEAR_COMPLEX_ERROR(          \
        val1,                           \
        val2,                           \
        thrust::complex<T>(std::numeric_limits<T>::epsilon(), std::numeric_limits<T>::epsilon()))

#define ASSERT_NEAR_COMPLEX_VECTOR_ERROR(val1, val2, abs_error) \
    ASSERT_PRED_FORMAT3(                                        \
        ComplexVectorNearPredFormat<typename decltype(val1)::value_type>, val1, val2, abs_error)

#define ASSERT_NEAR_COMPLEX_VECTOR(val1, val2) \
    ASSERT_NEAR_COMPLEX_VECTOR_ERROR(          \
        val1,                                  \
        val2,                                  \
        thrust::complex<T>(std::numeric_limits<T>::epsilon(), std::numeric_limits<T>::epsilon()))

template <typename T>
testing::AssertionResult
bitwise_equal(const char* a_expr, const char* b_expr, const T& a, const T& b)
{
    if(std::memcmp(&a, &b, sizeof(T)) == 0)
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
    for(size_t index = 0; result_it != result_end; ++result_it, ++expected_it, ++index)
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
