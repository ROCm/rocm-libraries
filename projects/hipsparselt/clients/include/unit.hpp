/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2022-2025 Advanced Micro Devices, Inc.
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

/* ========================================Gtest Unit Check
 * ==================================================== */

/*! \brief gtest unit compare two matrices float/double/complex */

#pragma once

#include "hipsparselt_math.hpp"
#include "hipsparselt_test.hpp"
#include "hipsparselt_vector.hpp"
#include <hip/hip_fp8.h>
#include <hipsparselt/hipsparselt.h>

#ifndef GOOGLE_TEST
#define UNIT_CHECK(M, N, lda, strideA, hCPU, hGPU, batch_count, UNIT_ASSERT_EQ)
#define UNIT_CHECK_B(M, N, lda, hCPU, hGPU, batch_count, UNIT_ASSERT_EQ)
#else
#define UNIT_CHECK(M, N, lda, strideA, hCPU, hGPU, batch_count, UNIT_ASSERT_EQ)                  \
    do                                                                                           \
    {                                                                                            \
        for(size_t k = 0; k < batch_count; k++)                                                  \
            for(size_t j = 0; j < N; j++)                                                        \
                for(size_t i = 0; i < M; i++)                                                    \
                    if(hipsparselt_isnan(hCPU[i + j * size_t(lda) + k * strideA]))               \
                    {                                                                            \
                        ASSERT_TRUE(hipsparselt_isnan(hGPU[i + j * size_t(lda) + k * strideA])); \
                    }                                                                            \
                    else                                                                         \
                    {                                                                            \
                        UNIT_ASSERT_EQ(hCPU[i + j * size_t(lda) + k * strideA],                  \
                                       hGPU[i + j * size_t(lda) + k * strideA]);                 \
                    }                                                                            \
    } while(0)

#define UNIT_CHECK_B(M, N, lda, hCPU, hGPU, batch_count, UNIT_ASSERT_EQ)              \
    do                                                                                \
    {                                                                                 \
        for(size_t k = 0; k < batch_count; k++)                                       \
            for(size_t j = 0; j < N; j++)                                             \
                for(size_t i = 0; i < M; i++)                                         \
                    if(hipsparselt_isnan(hCPU[k][i + j * size_t(lda)]))               \
                    {                                                                 \
                        ASSERT_TRUE(hipsparselt_isnan(hGPU[k][i + j * size_t(lda)])); \
                    }                                                                 \
                    else                                                              \
                    {                                                                 \
                        UNIT_ASSERT_EQ(hCPU[k][i + j * size_t(lda)],                  \
                                       hGPU[k][i + j * size_t(lda)]);                 \
                    }                                                                 \
    } while(0)

//#define ASSERT_HALF_EQ(a, b) ASSERT_FLOAT_EQ(float(a), float(b))
//#define ASSERT_BF16_EQ(a, b) ASSERT_FLOAT_EQ(float(a), float(b))
#if defined(__HIP_PLATFORM_AMD__)
#define ASSERT_HALF_EQ(a, b)                                                         \
    do                                                                               \
    {                                                                                \
        union _HALF                                                                  \
        {                                                                            \
            uint16_t x;                                                              \
            _Float16 data;                                                           \
        };                                                                           \
        _HALF a_half  = {__half_raw(a).x};                                           \
        _HALF b_half  = {__half_raw(b).x};                                           \
        auto  absA    = (a_half.data > 0) ? a_half.data : -a_half.data;              \
        auto  absB    = (b_half.data > 0) ? b_half.data : -b_half.data;              \
        auto  absDiff = (a_half.data - b_half.data > 0) ? a_half.data - b_half.data  \
                                                        : b_half.data - a_half.data; \
        ASSERT_TRUE(absDiff / (absA + absB + 1) < 0.01);                             \
    } while(0)

#else
#define ASSERT_HALF_EQ(a, b)                             \
    do                                                   \
    {                                                       \
	auto zero    = __half(0.0);                         \
        auto absA    = (a > zero) ? a : negate(a);          \
        auto absB    = (b > zero) ? b : negate(b);          \
        auto absDiff = (a - b > zero) ? a - b : b - a;      \
        ASSERT_TRUE(absDiff / (absA + absB + __half(1.0)) < __half(0.01)); \
    } while(0)
#endif

#define ASSERT_BF16_EQ(a, b)                                                                      \
    do                                                                                            \
    {                                                                                             \
        const hip_bfloat16 bf16A    = static_cast<hip_bfloat16>(a);                               \
        const hip_bfloat16 bf16B    = static_cast<hip_bfloat16>(b);                               \
        const hip_bfloat16 bf16Zero = static_cast<hip_bfloat16>(0.0f);                            \
        const hip_bfloat16 bf16One  = static_cast<hip_bfloat16>(1.0f);                            \
        hip_bfloat16       absA     = (bf16A > bf16Zero) ? bf16A : negate(bf16A);                 \
        hip_bfloat16       absB     = (bf16B > bf16Zero) ? bf16B : negate(bf16B);                 \
        hip_bfloat16       absDiff  = (bf16A - bf16B > bf16Zero) ? bf16A - bf16B : bf16B - bf16A; \
        ASSERT_TRUE(absDiff / (absA + absB + bf16One) < static_cast<hip_bfloat16>(0.1f));         \
    } while(0)

#define ASSERT_FP8_EQ(a, b)                                                 \
    do                                                                      \
    {                                                                       \
        const float f32A = static_cast<float>(a);                           \
        const float f32B = static_cast<float>(b);                           \
        float absA    = (f32A > 0.0f) ? f32A : negate(f32A);                \
        float absB    = (f32B > 0.0f) ? f32B : negate(f32B);                \
        float absDiff = (f32A - f32B > 0.0f) ? f32A - f32B : f32B - f32A;   \
        ASSERT_TRUE(absDiff / (absA + absB + 1.0f) < 0.125f);               \
        /*tolerance * eps = 2 * 0.0625; 2*eps needed for SR*/               \
    } while(0)


#define ASSERT_BF8_EQ(a, b)                                                 \
    do                                                                      \
    {                                                                       \
        const float f32A = static_cast<float>(a);                           \
        const float f32B = static_cast<float>(b);                           \
        float absA    = (f32A > 0.0f) ? f32A : negate(f32A);                \
        float absB    = (f32B > 0.0f) ? f32B : negate(f32B);                \
        float absDiff = (f32A - f32B > 0.0f) ? f32A - f32B : f32B - f32A;   \
        ASSERT_TRUE(absDiff / (absA + absB + 1.0f) < 0.25f);                \
        /*tolerance * eps= 2 * 0.125; 2*eps needed for SR*/                 \
    } while(0)

// Compare float to hip_bfloat16
// Allow the hip_bfloat16 to match the rounded or truncated value of float
// Only call ASSERT_FLOAT_EQ with the rounded value if the truncated value does not match
#include <gtest/internal/gtest-internal.h>
#define ASSERT_FLOAT_BF16_EQ(a, b)                                   \
    do                                                               \
    {                                                                \
        using testing::internal::FloatingPoint;                      \
        if(!FloatingPoint<float>(b).AlmostEquals(                    \
               FloatingPoint<float>(float_to_bfloat16_truncate(a)))) \
            ASSERT_FLOAT_EQ(b, hip_bfloat16(a));                     \
    } while(0)

#define ASSERT_FLOAT_COMPLEX_EQ(a, b)                  \
    do                                                 \
    {                                                  \
        auto ta = (a), tb = (b);                       \
        ASSERT_FLOAT_EQ(std::real(ta), std::real(tb)); \
        ASSERT_FLOAT_EQ(std::imag(ta), std::imag(tb)); \
    } while(0)

#define ASSERT_DOUBLE_COMPLEX_EQ(a, b)                  \
    do                                                  \
    {                                                   \
        auto ta = (a), tb = (b);                        \
        ASSERT_DOUBLE_EQ(std::real(ta), std::real(tb)); \
        ASSERT_DOUBLE_EQ(std::imag(ta), std::imag(tb)); \
    } while(0)

#endif // GOOGLE_TEST

// TODO: Replace std::remove_cv_t with std::type_identity_t in C++20
// It is only used to make T_hpa non-deduced
template <typename T, typename T_hpa = T>
void unit_check_general(
    int64_t M, int64_t N, int64_t lda, const std::remove_cv_t<T_hpa>* hCPU, const T* hGPU);

template <>
inline void unit_check_general(
    int64_t M, int64_t N, int64_t lda, const hip_bfloat16* hCPU, const hip_bfloat16* hGPU)
{
    UNIT_CHECK(M, N, lda, 0, hCPU, hGPU, 1, ASSERT_BF16_EQ);
}

template <>
inline void unit_check_general<hip_bfloat16, float>(
    int64_t M, int64_t N, int64_t lda, const float* hCPU, const hip_bfloat16* hGPU)
{
    UNIT_CHECK(M, N, lda, 0, hCPU, hGPU, 1, ASSERT_FLOAT_BF16_EQ);
}

template <>
inline void
    unit_check_general(int64_t M, int64_t N, int64_t lda, const __half* hCPU, const __half* hGPU)
{
    UNIT_CHECK(M, N, lda, 0, hCPU, hGPU, 1, ASSERT_HALF_EQ);
}

template <>
inline void
    unit_check_general(int64_t M, int64_t N, int64_t lda, const float* hCPU, const float* hGPU)
{
    UNIT_CHECK(M, N, lda, 0, hCPU, hGPU, 1, ASSERT_FLOAT_EQ);
}

template <>
inline void
    unit_check_general(int64_t M, int64_t N, int64_t lda, const double* hCPU, const double* hGPU)
{
    UNIT_CHECK(M, N, lda, 0, hCPU, hGPU, 1, ASSERT_DOUBLE_EQ);
}

template <>
inline void
    unit_check_general(int64_t M, int64_t N, int64_t lda, const int64_t* hCPU, const int64_t* hGPU)
{
    UNIT_CHECK(M, N, lda, 0, hCPU, hGPU, 1, ASSERT_EQ);
}

template <>
inline void
    unit_check_general(int64_t M, int64_t N, int64_t lda, const int8_t* hCPU, const int8_t* hGPU)
{
    UNIT_CHECK(M, N, lda, 0, hCPU, hGPU, 1, ASSERT_EQ);
}

#ifdef HIPSPARSELT_CLIENT_ENABLE_FP8_OCP
template <>
inline void unit_check_general(
    int64_t M, int64_t N, int64_t lda, const hipsparselt_fp8_e4m3* hCPU, const hipsparselt_fp8_e4m3* hGPU)
{
    UNIT_CHECK(M, N, lda, 0, hCPU, hGPU, 1, ASSERT_FP8_EQ);
}

template <>
inline void unit_check_general(
    int64_t M, int64_t N, int64_t lda, const hipsparselt_fp8_e5m2* hCPU, const hipsparselt_fp8_e5m2* hGPU)
{
    UNIT_CHECK(M, N, lda, 0, hCPU, hGPU, 1, ASSERT_BF8_EQ);
}
#endif

template <typename T, typename T_hpa = T>
void unit_check_general(int64_t                        M,
                        int64_t                        N,
                        int64_t                        lda,
                        int64_t                        strideA,
                        const std::remove_cv_t<T_hpa>* hCPU,
                        const T*                       hGPU,
                        int64_t                        batch_count);

template <>
inline void unit_check_general(int64_t             M,
                               int64_t             N,
                               int64_t             lda,
                               int64_t             strideA,
                               const hip_bfloat16* hCPU,
                               const hip_bfloat16* hGPU,
                               int64_t             batch_count)
{
    UNIT_CHECK(M, N, lda, strideA, hCPU, hGPU, batch_count, ASSERT_BF16_EQ);
}

template <>
inline void unit_check_general<hip_bfloat16, float>(int64_t             M,
                                                    int64_t             N,
                                                    int64_t             lda,
                                                    int64_t             strideA,
                                                    const float*        hCPU,
                                                    const hip_bfloat16* hGPU,
                                                    int64_t             batch_count)
{
    UNIT_CHECK(M, N, lda, strideA, hCPU, hGPU, batch_count, ASSERT_FLOAT_BF16_EQ);
}

template <>
inline void unit_check_general(int64_t       M,
                               int64_t       N,
                               int64_t       lda,
                               int64_t       strideA,
                               const __half* hCPU,
                               const __half* hGPU,
                               int64_t       batch_count)
{
    UNIT_CHECK(M, N, lda, strideA, hCPU, hGPU, batch_count, ASSERT_HALF_EQ);
}

template <>
inline void unit_check_general(int64_t      M,
                               int64_t      N,
                               int64_t      lda,
                               int64_t      strideA,
                               const float* hCPU,
                               const float* hGPU,
                               int64_t      batch_count)
{
    UNIT_CHECK(M, N, lda, strideA, hCPU, hGPU, batch_count, ASSERT_FLOAT_EQ);
}

template <>
inline void unit_check_general(int64_t       M,
                               int64_t       N,
                               int64_t       lda,
                               int64_t       strideA,
                               const double* hCPU,
                               const double* hGPU,
                               int64_t       batch_count)
{
    UNIT_CHECK(M, N, lda, strideA, hCPU, hGPU, batch_count, ASSERT_DOUBLE_EQ);
}

template <>
inline void unit_check_general(int64_t        M,
                               int64_t        N,
                               int64_t        lda,
                               int64_t        strideA,
                               const int64_t* hCPU,
                               const int64_t* hGPU,
                               int64_t        batch_count)
{
    UNIT_CHECK(M, N, lda, strideA, hCPU, hGPU, batch_count, ASSERT_EQ);
}

template <>
inline void unit_check_general(int64_t       M,
                               int64_t       N,
                               int64_t       lda,
                               int64_t       strideA,
                               const int8_t* hCPU,
                               const int8_t* hGPU,
                               int64_t       batch_count)
{
    UNIT_CHECK(M, N, lda, strideA, hCPU, hGPU, batch_count, ASSERT_EQ);
}

#ifdef HIPSPARSELT_CLIENT_ENABLE_FP8_OCP
template <>
inline void unit_check_general(int64_t               M,
                               int64_t               N,
                               int64_t               lda,
                               int64_t               strideA,
                               const hipsparselt_fp8_e4m3* hCPU,
                               const hipsparselt_fp8_e4m3* hGPU,
                               int64_t               batch_count)
{
    UNIT_CHECK(M, N, lda, strideA, hCPU, hGPU, batch_count, ASSERT_FP8_EQ);
}

template <>
inline void unit_check_general(int64_t               M,
                               int64_t               N,
                               int64_t               lda,
                               int64_t               strideA,
                               const hipsparselt_fp8_e5m2* hCPU,
                               const hipsparselt_fp8_e5m2* hGPU,
                               int64_t               batch_count)
{
    UNIT_CHECK(M, N, lda, strideA, hCPU, hGPU, batch_count, ASSERT_BF8_EQ);
}
#endif

template <typename T, typename T_hpa = T>
void unit_check_general(int64_t                                    M,
                        int64_t                                    N,
                        int64_t                                    lda,
                        const host_vector<std::remove_cv_t<T_hpa>> hCPU[],
                        const host_vector<T>                       hGPU[],
                        int64_t                                    batch_count);

template <>
inline void unit_check_general(int64_t                         M,
                               int64_t                         N,
                               int64_t                         lda,
                               const host_vector<hip_bfloat16> hCPU[],
                               const host_vector<hip_bfloat16> hGPU[],
                               int64_t                         batch_count)
{
    UNIT_CHECK_B(M, N, lda, hCPU, hGPU, batch_count, ASSERT_BF16_EQ);
}

template <>
inline void unit_check_general<hip_bfloat16, float>(int64_t                         M,
                                                    int64_t                         N,
                                                    int64_t                         lda,
                                                    const host_vector<float>        hCPU[],
                                                    const host_vector<hip_bfloat16> hGPU[],
                                                    int64_t                         batch_count)
{
    UNIT_CHECK_B(M, N, lda, hCPU, hGPU, batch_count, ASSERT_FLOAT_BF16_EQ);
}

template <>
inline void unit_check_general(int64_t                   M,
                               int64_t                   N,
                               int64_t                   lda,
                               const host_vector<__half> hCPU[],
                               const host_vector<__half> hGPU[],
                               int64_t                   batch_count)
{
    UNIT_CHECK_B(M, N, lda, hCPU, hGPU, batch_count, ASSERT_HALF_EQ);
}

template <>
inline void unit_check_general(int64_t                M,
                               int64_t                N,
                               int64_t                lda,
                               const host_vector<int> hCPU[],
                               const host_vector<int> hGPU[],
                               int64_t                batch_count)
{
    UNIT_CHECK_B(M, N, lda, hCPU, hGPU, batch_count, ASSERT_EQ);
}

template <>
inline void unit_check_general(int64_t                   M,
                               int64_t                   N,
                               int64_t                   lda,
                               const host_vector<int8_t> hCPU[],
                               const host_vector<int8_t> hGPU[],
                               int64_t                   batch_count)
{
    UNIT_CHECK_B(M, N, lda, hCPU, hGPU, batch_count, ASSERT_EQ);
}

template <>
inline void unit_check_general(int64_t                  M,
                               int64_t                  N,
                               int64_t                  lda,
                               const host_vector<float> hCPU[],
                               const host_vector<float> hGPU[],
                               int64_t                  batch_count)
{
    UNIT_CHECK_B(M, N, lda, hCPU, hGPU, batch_count, ASSERT_FLOAT_EQ);
}

template <>
inline void unit_check_general(int64_t                   M,
                               int64_t                   N,
                               int64_t                   lda,
                               const host_vector<double> hCPU[],
                               const host_vector<double> hGPU[],
                               int64_t                   batch_count)
{
    UNIT_CHECK_B(M, N, lda, hCPU, hGPU, batch_count, ASSERT_DOUBLE_EQ);
}

template <typename T, typename T_hpa = T>
void unit_check_general(int64_t                              M,
                        int64_t                              N,
                        int64_t                              lda,
                        const std::remove_cv_t<T_hpa>* const hCPU[],
                        const T* const                       hGPU[],
                        int64_t                              batch_count);

template <>
inline void unit_check_general(int64_t                   M,
                               int64_t                   N,
                               int64_t                   lda,
                               const hip_bfloat16* const hCPU[],
                               const hip_bfloat16* const hGPU[],
                               int64_t                   batch_count)
{
    UNIT_CHECK_B(M, N, lda, hCPU, hGPU, batch_count, ASSERT_BF16_EQ);
}

template <>
inline void unit_check_general<hip_bfloat16, float>(int64_t                   M,
                                                    int64_t                   N,
                                                    int64_t                   lda,
                                                    const float* const        hCPU[],
                                                    const hip_bfloat16* const hGPU[],
                                                    int64_t                   batch_count)
{
    UNIT_CHECK_B(M, N, lda, hCPU, hGPU, batch_count, ASSERT_FLOAT_BF16_EQ);
}

template <>
inline void unit_check_general(int64_t             M,
                               int64_t             N,
                               int64_t             lda,
                               const __half* const hCPU[],
                               const __half* const hGPU[],
                               int64_t             batch_count)
{
    UNIT_CHECK_B(M, N, lda, hCPU, hGPU, batch_count, ASSERT_HALF_EQ);
}

template <>
inline void unit_check_general(int64_t          M,
                               int64_t          N,
                               int64_t          lda,
                               const int* const hCPU[],
                               const int* const hGPU[],
                               int64_t          batch_count)
{
    UNIT_CHECK_B(M, N, lda, hCPU, hGPU, batch_count, ASSERT_EQ);
}

template <>
inline void unit_check_general(int64_t             M,
                               int64_t             N,
                               int64_t             lda,
                               const int8_t* const hCPU[],
                               const int8_t* const hGPU[],
                               int64_t             batch_count)
{
    UNIT_CHECK_B(M, N, lda, hCPU, hGPU, batch_count, ASSERT_EQ);
}

template <>
inline void unit_check_general(int64_t            M,
                               int64_t            N,
                               int64_t            lda,
                               const float* const hCPU[],
                               const float* const hGPU[],
                               int64_t            batch_count)
{
    UNIT_CHECK_B(M, N, lda, hCPU, hGPU, batch_count, ASSERT_FLOAT_EQ);
}

template <>
inline void unit_check_general(int64_t             M,
                               int64_t             N,
                               int64_t             lda,
                               const double* const hCPU[],
                               const double* const hGPU[],
                               int64_t             batch_count)
{
    UNIT_CHECK_B(M, N, lda, hCPU, hGPU, batch_count, ASSERT_DOUBLE_EQ);
}

#ifdef HIPSPARSELT_CLIENT_ENABLE_FP8_OCP
template <>
inline void unit_check_general(int64_t                     M,
                               int64_t                     N,
                               int64_t                     lda,
                               const hipsparselt_fp8_e4m3* const hCPU[],
                               const hipsparselt_fp8_e4m3* const hGPU[],
                               int64_t                     batch_count)
{
    unit_check_general(M, N, lda, &hCPU[0], &hGPU[0], batch_count);
}

template <>
inline void unit_check_general(int64_t                     M,
                               int64_t                     N,
                               int64_t                     lda,
                               const hipsparselt_fp8_e5m2* const hCPU[],
                               const hipsparselt_fp8_e5m2* const hGPU[],
                               int64_t                     batch_count)
{
    unit_check_general(M, N, lda, &hCPU[0], &hGPU[0], batch_count);
}
#endif

template <typename T>
inline void trsm_err_res_check(T max_error, int64_t M, T forward_tolerance, T eps)
{
#ifdef GOOGLE_TEST
    ASSERT_LE(max_error, forward_tolerance * eps * M);
#endif
}

template <typename T>
constexpr double get_epsilon()
{
    return std::numeric_limits<T>::epsilon();
}

template <typename T>
inline int64_t unit_check_diff(
    int64_t M, int64_t N, int64_t lda, int64_t stride, T* hCPU, T* hGPU, int64_t batch_count)
{
    using c_type  = std::conditional_t<std::is_same<__half, T>::value
#ifdef HIPSPARSELT_CLIENT_ENABLE_FP8_OCP
	    || std::is_same<hipsparselt_fp8_e4m3, T>::value || std::is_same<hipsparselt_fp8_e5m2, T>::value
#endif
	    , float, T>;
    int64_t error = 0;
    do
    {
        for(size_t k = 0; k < batch_count; k++)
            for(size_t j = 0; j < N; j++)
                for(size_t i = 0; i < M; i++)
                    if(hipsparselt_isnan(hCPU[i + j * size_t(lda) + k * stride]))
                    {
                        error += hipsparselt_isnan(hGPU[i + j * size_t(lda) + k * stride]) ? 0 : 1;
                    }
                    else
                    {
                        error += static_cast<c_type>(hCPU[i + j * size_t(lda) + k * stride])
                                         == static_cast<c_type>(
                                             hGPU[i + j * size_t(lda) + k * stride])
                                     ? 0
                                     : 1;
                    }
    } while(0);
    return error;
}
