/* ************************************************************************
 * Copyright (C) 2018-2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 *
 * ************************************************************************ */

/* =====================================================================
    Google Near check: ASSERT_NEAR( elementof(A), elementof(B))
   =================================================================== */

/*!\file
 * \brief compares two results (usually, CPU and GPU results); provides Google Near check.
 */

#pragma once

#include "rocblas.h"
#include "rocblas_math.hpp"
#include "rocblas_test.hpp"
#include "rocblas_vector.hpp"
#include "unit.hpp"

// sqrt(0.5) factor for complex cutoff calculations
constexpr double sqrthalf = 0.7071067811865475244;

template <class T>
inline bool reduction_requires_near(const Arguments& arg, int64_t n)
{
    return arg.initialization == rocblas_initialization::hpl
           || (std::is_same_v<T, rocblas_half> && n > 10000);
}

// Sum error tolerance for large sums. If multiplied by the number of items
// in the sum you get an expected a normalized upper bound on error.
// The sum exponent needs to also be considered for the error tolerance

template <class T>
inline constexpr double sum_error_tolerance = get_epsilon<T>();

template <>
inline constexpr double sum_error_tolerance<rocblas_bfloat16> = 1 / 100.0;

template <>
inline constexpr double sum_error_tolerance<rocblas_half> = 1 / 900.0;

template <>
inline constexpr double sum_error_tolerance<rocblas_float_complex> = 1 / 10000.0;

template <>
inline constexpr double sum_error_tolerance<rocblas_double_complex> = 1 / 1000000.0;

template <class Tc, class Ti, class To>
inline constexpr double sum_error_tolerance_for_gfx11 = get_epsilon<Tc>();

template <>
inline constexpr double sum_error_tolerance_for_gfx11<float, rocblas_bfloat16, float> = 1 / 10.0;

template <>
inline constexpr double
    sum_error_tolerance_for_gfx11<float, rocblas_bfloat16, rocblas_bfloat16> = 1 / 10.0;

template <>
inline constexpr double sum_error_tolerance_for_gfx11<float, rocblas_half, float> = 1 / 100.0;

template <>
inline constexpr double sum_error_tolerance_for_gfx11<float, rocblas_half, rocblas_half> = 1
                                                                                           / 100.0;
template <>
inline constexpr double
    sum_error_tolerance_for_gfx11<rocblas_half, rocblas_half, rocblas_half> = 1 / 100.0;

template <>
inline constexpr double sum_error_tolerance_for_gfx11<rocblas_float_complex,
                                                      rocblas_float_complex,
                                                      rocblas_float_complex> = 1 / 10000.0;

template <>
inline constexpr double sum_error_tolerance_for_gfx11<rocblas_double_complex,
                                                      rocblas_double_complex,
                                                      rocblas_double_complex> = 1 / 1000000.0;

template <typename T>
double sum_near_tolerance(int64_t n, real_t<T> sum)
{
    double count     = (n == 1 || n > 4) ? sqrt(n) : 2.0;
    double tolerance = sum_error_tolerance<T> * 2.0 * count;
    if(sum != 0)
    {
        tolerance *= std::abs(sum);
    }
    return tolerance;
}

template <typename Tc, typename Ti, typename To>
double dot_near_tolerance(int arch_major, int64_t n, To sum)
{
    double tolerance;
    if(arch_major == 11 && sizeof(To) == 2)
    {
        double count = (n == 1 || n > 4) ? sqrt(n) : 2.0;
        tolerance    = sum_error_tolerance_for_gfx11<Tc, Ti, To> * 2.0 * count;
        if(sum != 0)
        {
            tolerance *= rocblas_abs(sum);
        }
    }
    else
    {
        tolerance = sum_error_tolerance<Tc> * n;
    }

    return tolerance;
}

#ifndef GOOGLE_TEST
#define NEAR_CHECK(M, N, lda, strideA, hCPU, hGPU, batch_count, err, NEAR_ASSERT)
#define NEAR_CHECK_B(M, N, lda, hCPU, hGPU, batch_count, err, NEAR_ASSERT)
#else

// Also used for vectors with lda used for inc, which may be negative
#define NEAR_CHECK(M, N, lda, strideA, hCPU, hGPU, batch_count, err, NEAR_ASSERT) \
    do                                                                            \
    {                                                                             \
        for(int64_t k = 0; k < batch_count; k++)                                  \
            for(int64_t j = 0; j < N; j++)                                        \
            {                                                                     \
                int64_t offset = lda >= 0 ? 0 : int64_t(lda) * (1 - N);           \
                offset += j * int64_t(lda) + k * strideA;                         \
                size_t idx = offset;                                              \
                for(size_t i = 0; i < M; i++)                                     \
                {                                                                 \
                    if(rocblas_isnan(hCPU[i + idx]))                              \
                    {                                                             \
                        ASSERT_TRUE(rocblas_isnan(hGPU[i + idx]));                \
                    }                                                             \
                    else                                                          \
                    {                                                             \
                        NEAR_ASSERT(hCPU[i + idx], hGPU[i + idx], err);           \
                    }                                                             \
                }                                                                 \
            }                                                                     \
    } while(0)

// Also used for vectors with lda used for inc, which may be negative
#define NEAR_CHECK_B(M, N, lda, hCPU, hGPU, batch_count, err, NEAR_ASSERT)    \
    do                                                                        \
    {                                                                         \
        for(size_t k = 0; k < batch_count; k++)                               \
            for(int64_t j = 0; j < N; j++)                                    \
            {                                                                 \
                int64_t offset = lda >= 0 ? 0 : int64_t(lda) * (1 - N);       \
                offset += j * int64_t(lda);                                   \
                size_t idx = offset;                                          \
                for(size_t i = 0; i < M; i++)                                 \
                {                                                             \
                    if(rocblas_isnan(hCPU[k][i + idx]))                       \
                    {                                                         \
                        ASSERT_TRUE(rocblas_isnan(hGPU[k][i + idx]));         \
                    }                                                         \
                    else                                                      \
                    {                                                         \
                        NEAR_ASSERT(hCPU[k][i + idx], hGPU[k][i + idx], err); \
                    }                                                         \
                }                                                             \
            }                                                                 \
    } while(0)

#endif

#define NEAR_ASSERT_HALF(a, b, err) ASSERT_NEAR(double(a), double(b), err)

#define NEAR_ASSERT_BF16(a, b, err) ASSERT_NEAR(double(rocblas_bfloat16(a)), double(b), err)

#define NEAR_ASSERT_COMPLEX(a, b, err)                  \
    do                                                  \
    {                                                   \
        auto ta = (a), tb = (b);                        \
        ASSERT_NEAR(std::real(ta), std::real(tb), err); \
        ASSERT_NEAR(std::imag(ta), std::imag(tb), err); \
    } while(0)

// TODO: Replace std::remove_cv_t with std::type_identity_t in C++20
// It is only used to make T_hpa non-deduced
template <typename T, typename T_hpa = T>
inline void near_check_general(int64_t                        M,
                               int64_t                        N,
                               int64_t                        lda,
                               const std::remove_cv_t<T_hpa>* hCPU,
                               const T*                       hGPU,
                               double                         abs_error)
{
    NEAR_CHECK(M, N, lda, 0, hCPU, hGPU, 1, abs_error, ASSERT_NEAR);
}

template <>
inline void near_check_general(int64_t             M,
                               int64_t             N,
                               int64_t             lda,
                               const rocblas_half* hCPU,
                               const rocblas_half* hGPU,
                               double              abs_error)
{
    NEAR_CHECK(M, N, lda, 0, hCPU, hGPU, 1, abs_error, NEAR_ASSERT_HALF);
}

template <>
inline void near_check_general<rocblas_bfloat16, float>(int64_t                 M,
                                                        int64_t                 N,
                                                        int64_t                 lda,
                                                        const float*            hCPU,
                                                        const rocblas_bfloat16* hGPU,
                                                        double                  abs_error)
{
    NEAR_CHECK(M, N, lda, 0, hCPU, hGPU, 1, abs_error, NEAR_ASSERT_BF16);
}

template <>
inline void near_check_general(int64_t                      M,
                               int64_t                      N,
                               int64_t                      lda,
                               const rocblas_float_complex* hCPU,
                               const rocblas_float_complex* hGPU,
                               double                       abs_error)
{
    abs_error *= sqrthalf;
    NEAR_CHECK(M, N, lda, 0, hCPU, hGPU, 1, abs_error, NEAR_ASSERT_COMPLEX);
}

template <>
inline void near_check_general(int64_t                       M,
                               int64_t                       N,
                               int64_t                       lda,
                               const rocblas_double_complex* hCPU,
                               const rocblas_double_complex* hGPU,
                               double                        abs_error)
{
    abs_error *= sqrthalf;
    NEAR_CHECK(M, N, lda, 0, hCPU, hGPU, 1, abs_error, NEAR_ASSERT_COMPLEX);
}

template <typename T, typename T_hpa = T>
inline void near_check_general(int64_t                        M,
                               int64_t                        N,
                               int64_t                        lda,
                               rocblas_stride                 strideA,
                               const std::remove_cv_t<T_hpa>* hCPU,
                               const T*                       hGPU,
                               int64_t                        batch_count,
                               double                         abs_error)
{
    NEAR_CHECK(M, N, lda, strideA, hCPU, hGPU, batch_count, abs_error, ASSERT_NEAR);
}

template <>
inline void near_check_general(int64_t             M,
                               int64_t             N,
                               int64_t             lda,
                               rocblas_stride      strideA,
                               const rocblas_half* hCPU,
                               const rocblas_half* hGPU,
                               int64_t             batch_count,
                               double              abs_error)
{
    NEAR_CHECK(M, N, lda, strideA, hCPU, hGPU, batch_count, abs_error, NEAR_ASSERT_HALF);
}

template <>
inline void near_check_general<rocblas_bfloat16, float>(int64_t                 M,
                                                        int64_t                 N,
                                                        int64_t                 lda,
                                                        rocblas_stride          strideA,
                                                        const float*            hCPU,
                                                        const rocblas_bfloat16* hGPU,
                                                        int64_t                 batch_count,
                                                        double                  abs_error)
{
    NEAR_CHECK(M, N, lda, strideA, hCPU, hGPU, batch_count, abs_error, NEAR_ASSERT_BF16);
}

template <>
inline void near_check_general(int64_t                      M,
                               int64_t                      N,
                               int64_t                      lda,
                               rocblas_stride               strideA,
                               const rocblas_float_complex* hCPU,
                               const rocblas_float_complex* hGPU,
                               int64_t                      batch_count,
                               double                       abs_error)
{
    abs_error *= sqrthalf;
    NEAR_CHECK(M, N, lda, strideA, hCPU, hGPU, batch_count, abs_error, NEAR_ASSERT_COMPLEX);
}

template <>
inline void near_check_general(int64_t                       M,
                               int64_t                       N,
                               int64_t                       lda,
                               rocblas_stride                strideA,
                               const rocblas_double_complex* hCPU,
                               const rocblas_double_complex* hGPU,
                               int64_t                       batch_count,
                               double                        abs_error)
{
    abs_error *= sqrthalf;
    NEAR_CHECK(M, N, lda, strideA, hCPU, hGPU, batch_count, abs_error, NEAR_ASSERT_COMPLEX);
}

template <typename T, typename T_hpa = T>
void near_check_general(int64_t                                    M,
                        int64_t                                    N,
                        int64_t                                    lda,
                        const host_vector<std::remove_cv_t<T_hpa>> hCPU[],
                        const host_vector<T>                       hGPU[],
                        int64_t                                    batch_count,
                        double                                     abs_error)
{
    NEAR_CHECK_B(M, N, lda, hCPU, hGPU, batch_count, abs_error, ASSERT_NEAR);
}

template <>
inline void near_check_general(int64_t                         M,
                               int64_t                         N,
                               int64_t                         lda,
                               const host_vector<rocblas_half> hCPU[],
                               const host_vector<rocblas_half> hGPU[],
                               int64_t                         batch_count,
                               double                          abs_error)
{
    NEAR_CHECK_B(M, N, lda, hCPU, hGPU, batch_count, abs_error, NEAR_ASSERT_HALF);
}
template <>
inline void near_check_general<rocblas_bfloat16, float>(int64_t                             M,
                                                        int64_t                             N,
                                                        int64_t                             lda,
                                                        const host_vector<float>            hCPU[],
                                                        const host_vector<rocblas_bfloat16> hGPU[],
                                                        int64_t batch_count,
                                                        double  abs_error)
{
    NEAR_CHECK_B(M, N, lda, hCPU, hGPU, batch_count, abs_error, NEAR_ASSERT_BF16);
}

template <>
inline void near_check_general(int64_t                                  M,
                               int64_t                                  N,
                               int64_t                                  lda,
                               const host_vector<rocblas_float_complex> hCPU[],
                               const host_vector<rocblas_float_complex> hGPU[],
                               int64_t                                  batch_count,
                               double                                   abs_error)
{
    abs_error *= sqrthalf;
    NEAR_CHECK_B(M, N, lda, hCPU, hGPU, batch_count, abs_error, NEAR_ASSERT_COMPLEX);
}

template <>
inline void near_check_general(int64_t                                   M,
                               int64_t                                   N,
                               int64_t                                   lda,
                               const host_vector<rocblas_double_complex> hCPU[],
                               const host_vector<rocblas_double_complex> hGPU[],
                               int64_t                                   batch_count,
                               double                                    abs_error)
{
    abs_error *= sqrthalf;
    NEAR_CHECK_B(M, N, lda, hCPU, hGPU, batch_count, abs_error, NEAR_ASSERT_COMPLEX);
}

template <typename T, typename T_hpa = T>
inline void near_check_general(int64_t                              M,
                               int64_t                              N,
                               int64_t                              lda,
                               const std::remove_cv_t<T_hpa>* const hCPU[],
                               const T* const                       hGPU[],
                               int64_t                              batch_count,
                               double                               abs_error)
{
    NEAR_CHECK_B(M, N, lda, hCPU, hGPU, batch_count, abs_error, ASSERT_NEAR);
}

template <>
inline void near_check_general(int64_t                   M,
                               int64_t                   N,
                               int64_t                   lda,
                               const rocblas_half* const hCPU[],
                               const rocblas_half* const hGPU[],
                               int64_t                   batch_count,
                               double                    abs_error)
{
    NEAR_CHECK_B(M, N, lda, hCPU, hGPU, batch_count, abs_error, NEAR_ASSERT_HALF);
}

template <>
inline void near_check_general<rocblas_bfloat16, float>(int64_t                       M,
                                                        int64_t                       N,
                                                        int64_t                       lda,
                                                        const float* const            hCPU[],
                                                        const rocblas_bfloat16* const hGPU[],
                                                        int64_t                       batch_count,
                                                        double                        abs_error)
{
    NEAR_CHECK_B(M, N, lda, hCPU, hGPU, batch_count, abs_error, NEAR_ASSERT_BF16);
}

template <>
inline void near_check_general(int64_t                            M,
                               int64_t                            N,
                               int64_t                            lda,
                               const rocblas_float_complex* const hCPU[],
                               const rocblas_float_complex* const hGPU[],
                               int64_t                            batch_count,
                               double                             abs_error)
{
    NEAR_CHECK_B(M, N, lda, hCPU, hGPU, batch_count, abs_error, NEAR_ASSERT_COMPLEX);
}

template <>
inline void near_check_general(int64_t                             M,
                               int64_t                             N,
                               int64_t                             lda,
                               const rocblas_double_complex* const hCPU[],
                               const rocblas_double_complex* const hGPU[],
                               int64_t                             batch_count,
                               double                              abs_error)
{
    NEAR_CHECK_B(M, N, lda, hCPU, hGPU, batch_count, abs_error, NEAR_ASSERT_COMPLEX);
}
