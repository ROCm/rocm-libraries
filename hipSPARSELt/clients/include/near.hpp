/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2022-2023 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

/* =====================================================================
    Google Near check: ASSERT_NEAR( elementof(A), elementof(B))
   =================================================================== */

/*!\file
 * \brief compares two results (usually, CPU and GPU results); provides Google Near check.
 */

#pragma once

#include "hipsparselt_math.hpp"
#include "hipsparselt_test.hpp"
#include "hipsparselt_vector.hpp"
#include <hipsparselt/hipsparselt.h>

#ifndef GOOGLE_TEST
#define NEAR_CHECK(M, N, lda, strideA, hCPU, hGPU, batch_count, err, NEAR_ASSERT)
#define NEAR_CHECK_B(M, N, lda, hCPU, hGPU, batch_count, err, NEAR_ASSERT)
#else

#define NEAR_CHECK(M, N, lda, strideA, hCPU, hGPU, batch_count, err, NEAR_ASSERT) \
    do                                                                            \
    {                                                                             \
        for(size_t k = 0; k < batch_count; k++)                                   \
            for(size_t j = 0; j < N; j++)                                         \
                for(size_t i = 0; i < M; i++)                                     \
                    NEAR_ASSERT(hCPU[i + j * size_t(lda) + k * strideA],          \
                                hGPU[i + j * size_t(lda) + k * strideA],          \
                                err);                                             \
    } while(0)

#define NEAR_CHECK_B(M, N, lda, hCPU, hGPU, batch_count, err, NEAR_ASSERT)                    \
    do                                                                                        \
    {                                                                                         \
        for(size_t k = 0; k < batch_count; k++)                                               \
            for(size_t j = 0; j < N; j++)                                                     \
                for(size_t i = 0; i < M; i++)                                                 \
                    if(hipsparselt_isnan(hCPU[k][i + j * size_t(lda)]))                       \
                    {                                                                         \
                        ASSERT_TRUE(hipsparselt_isnan(hGPU[k][i + j * size_t(lda)]));         \
                    }                                                                         \
                    else                                                                      \
                    {                                                                         \
                        NEAR_ASSERT(                                                          \
                            hCPU[k][i + j * size_t(lda)], hGPU[k][i + j * size_t(lda)], err); \
                    }                                                                         \
    } while(0)

#endif

#define NEAR_ASSERT_HALF(a, b, err) ASSERT_NEAR(double(a), double(b), err)

#define NEAR_ASSERT_BF16(a, b, err) ASSERT_NEAR(double(a), double(b), err)

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
inline void near_check_general(
    int64_t M, int64_t N, int64_t lda, const __half* hCPU, const __half* hGPU, double abs_error)
{
    NEAR_CHECK(M, N, lda, 0, hCPU, hGPU, 1, abs_error, NEAR_ASSERT_HALF);
}

template <>
inline void near_check_general<hip_bfloat16, float>(int64_t             M,
                                                    int64_t             N,
                                                    int64_t             lda,
                                                    const float*        hCPU,
                                                    const hip_bfloat16* hGPU,
                                                    double              abs_error)
{
    NEAR_CHECK(M, N, lda, 0, hCPU, hGPU, 1, abs_error, NEAR_ASSERT_BF16);
}

template <typename T, typename T_hpa = T>
inline void near_check_general(int64_t                        M,
                               int64_t                        N,
                               int64_t                        lda,
                               int64_t                        strideA,
                               const std::remove_cv_t<T_hpa>* hCPU,
                               const T*                       hGPU,
                               int64_t                        batch_count,
                               double                         abs_error)
{
    NEAR_CHECK(M, N, lda, strideA, hCPU, hGPU, batch_count, abs_error, ASSERT_NEAR);
}

template <>
inline void near_check_general(int64_t       M,
                               int64_t       N,
                               int64_t       lda,
                               int64_t       strideA,
                               const __half* hCPU,
                               const __half* hGPU,
                               int64_t       batch_count,
                               double        abs_error)
{
    NEAR_CHECK(M, N, lda, strideA, hCPU, hGPU, batch_count, abs_error, NEAR_ASSERT_HALF);
}

template <>
inline void near_check_general<hip_bfloat16, float>(int64_t             M,
                                                    int64_t             N,
                                                    int64_t             lda,
                                                    int64_t             strideA,
                                                    const float*        hCPU,
                                                    const hip_bfloat16* hGPU,
                                                    int64_t             batch_count,
                                                    double              abs_error)
{
    NEAR_CHECK(M, N, lda, strideA, hCPU, hGPU, batch_count, abs_error, NEAR_ASSERT_BF16);
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
inline void near_check_general(int64_t                   M,
                               int64_t                   N,
                               int64_t                   lda,
                               const host_vector<__half> hCPU[],
                               const host_vector<__half> hGPU[],
                               int64_t                   batch_count,
                               double                    abs_error)
{
    NEAR_CHECK_B(M, N, lda, hCPU, hGPU, batch_count, abs_error, NEAR_ASSERT_HALF);
}
template <>
inline void near_check_general<hip_bfloat16, float>(int64_t                         M,
                                                    int64_t                         N,
                                                    int64_t                         lda,
                                                    const host_vector<float>        hCPU[],
                                                    const host_vector<hip_bfloat16> hGPU[],
                                                    int64_t                         batch_count,
                                                    double                          abs_error)
{
    NEAR_CHECK_B(M, N, lda, hCPU, hGPU, batch_count, abs_error, NEAR_ASSERT_BF16);
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
inline void near_check_general(int64_t             M,
                               int64_t             N,
                               int64_t             lda,
                               const __half* const hCPU[],
                               const __half* const hGPU[],
                               int64_t             batch_count,
                               double              abs_error)
{
    NEAR_CHECK_B(M, N, lda, hCPU, hGPU, batch_count, abs_error, NEAR_ASSERT_HALF);
}

template <>
inline void near_check_general<hip_bfloat16, float>(int64_t                   M,
                                                    int64_t                   N,
                                                    int64_t                   lda,
                                                    const float* const        hCPU[],
                                                    const hip_bfloat16* const hGPU[],
                                                    int64_t                   batch_count,
                                                    double                    abs_error)
{
    NEAR_CHECK_B(M, N, lda, hCPU, hGPU, batch_count, abs_error, NEAR_ASSERT_BF16);
}
