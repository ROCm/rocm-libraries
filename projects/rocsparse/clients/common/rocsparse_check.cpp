/*! \file */
/* ************************************************************************
 * Copyright (C) 2021-2025 Advanced Micro Devices, Inc. All rights Reserved.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */
#include "rocsparse_check.hpp"
#include "utility.hpp"

#ifdef GOOGLE_TEST
#include <gtest/gtest.h>
#endif

#ifndef GOOGLE_TEST

#include <iostream>

#define ASSERT_TRUE(cond)                                      \
    do                                                         \
    {                                                          \
        if(!(cond))                                            \
        {                                                      \
            std::cerr << "ASSERT_TRUE() failed." << std::endl; \
            exit(EXIT_FAILURE);                                \
        }                                                      \
    } while(0)

#define ASSERT_EQ(state1, state2)                                                              \
    do                                                                                         \
    {                                                                                          \
        if(state1 != state2)                                                                   \
        {                                                                                      \
            std::cerr.precision(16);                                                           \
            std::cerr << "ASSERT_EQ(" << state1 << ", " << state2 << ") failed." << std::endl; \
            exit(EXIT_FAILURE);                                                                \
        }                                                                                      \
    } while(0)

#define ASSERT_FLOAT_EQ ASSERT_EQ
#define ASSERT_DOUBLE_EQ ASSERT_EQ
#endif

#define ASSERT_FLOAT_COMPLEX_EQ(a, b)                \
    do                                               \
    {                                                \
        ASSERT_FLOAT_EQ(std::real(a), std::real(b)); \
        ASSERT_FLOAT_EQ(std::imag(a), std::imag(b)); \
    } while(0)

#define ASSERT_DOUBLE_COMPLEX_EQ(a, b)                \
    do                                                \
    {                                                 \
        ASSERT_DOUBLE_EQ(std::real(a), std::real(b)); \
        ASSERT_DOUBLE_EQ(std::imag(a), std::imag(b)); \
    } while(0)

#define ROCSPARSE_UNIT_CHECK(M, N, A, LDA, B, LDB, UNIT_ASSERT_EQ)  \
    do                                                              \
    {                                                               \
        for(int64_t j = 0; j < N; ++j)                              \
        {                                                           \
            for(int64_t i = 0; i < M; ++i)                          \
            {                                                       \
                if(rocsparse_isnan(A[i + j * LDA]))                 \
                {                                                   \
                    ASSERT_TRUE(rocsparse_isnan(B[i + j * LDB]));   \
                }                                                   \
                else                                                \
                {                                                   \
                    UNIT_ASSERT_EQ(A[i + j * LDA], B[i + j * LDB]); \
                }                                                   \
            }                                                       \
        }                                                           \
    } while(0)

template <>
void unit_check_general(
    int64_t M, int64_t N, const _Float16* A, int64_t LDA, const _Float16* B, int64_t LDB)
{
    ROCSPARSE_UNIT_CHECK(M, N, A, LDA, B, LDB, ASSERT_FLOAT_EQ);
}

template <>
void unit_check_general(int64_t                   M,
                        int64_t                   N,
                        const rocsparse_bfloat16* A,
                        int64_t                   LDA,
                        const rocsparse_bfloat16* B,
                        int64_t                   LDB)
{
    ROCSPARSE_UNIT_CHECK(M, N, A, LDA, B, LDB, ASSERT_FLOAT_EQ);
}

template <>
void unit_check_general(
    int64_t M, int64_t N, const float* A, int64_t LDA, const float* B, int64_t LDB)
{
    ROCSPARSE_UNIT_CHECK(M, N, A, LDA, B, LDB, ASSERT_FLOAT_EQ);
}

template <>
void unit_check_general(
    int64_t M, int64_t N, const double* A, int64_t LDA, const double* B, int64_t LDB)
{
    ROCSPARSE_UNIT_CHECK(M, N, A, LDA, B, LDB, ASSERT_DOUBLE_EQ);
}

template <>
void unit_check_general(int64_t                        M,
                        int64_t                        N,
                        const rocsparse_float_complex* A,
                        int64_t                        LDA,
                        const rocsparse_float_complex* B,
                        int64_t                        LDB)
{
    ROCSPARSE_UNIT_CHECK(M, N, A, LDA, B, LDB, ASSERT_FLOAT_COMPLEX_EQ);
}

template <>
void unit_check_general(int64_t                         M,
                        int64_t                         N,
                        const rocsparse_double_complex* A,
                        int64_t                         LDA,
                        const rocsparse_double_complex* B,
                        int64_t                         LDB)
{
    ROCSPARSE_UNIT_CHECK(M, N, A, LDA, B, LDB, ASSERT_DOUBLE_COMPLEX_EQ);
}

template <>
void unit_check_general(
    int64_t M, int64_t N, const int8_t* A, int64_t LDA, const int8_t* B, int64_t LDB)
{
    ROCSPARSE_UNIT_CHECK(M, N, A, LDA, B, LDB, ASSERT_EQ);
}

template <>
void unit_check_general(
    int64_t M, int64_t N, const int32_t* A, int64_t LDA, const int32_t* B, int64_t LDB)
{
    ROCSPARSE_UNIT_CHECK(M, N, A, LDA, B, LDB, ASSERT_EQ);
}

template <>
void unit_check_general(
    int64_t M, int64_t N, const uint8_t* A, int64_t LDA, const uint8_t* B, int64_t LDB)
{
    ROCSPARSE_UNIT_CHECK(M, N, A, LDA, B, LDB, ASSERT_EQ);
}

template <>
void unit_check_general(
    int64_t M, int64_t N, const uint32_t* A, int64_t LDA, const uint32_t* B, int64_t LDB)
{
    ROCSPARSE_UNIT_CHECK(M, N, A, LDA, B, LDB, ASSERT_EQ);
}

template <>
void unit_check_general(
    int64_t M, int64_t N, const int64_t* A, int64_t LDA, const int64_t* B, int64_t LDB)
{
    ROCSPARSE_UNIT_CHECK(M, N, A, LDA, B, LDB, ASSERT_EQ);
}
template <>
void unit_check_general(
    int64_t M, int64_t N, const size_t* A, int64_t LDA, const size_t* B, int64_t LDB)
{
    ROCSPARSE_UNIT_CHECK(M, N, A, LDA, B, LDB, ASSERT_EQ);
}

template <>
void unit_check_enum(const rocsparse_index_base a, const rocsparse_index_base b)
{
    ASSERT_TRUE(a == b);
}

template <>
void unit_check_enum(const rocsparse_order a, const rocsparse_order b)
{
    ASSERT_TRUE(a == b);
}

template <>
void unit_check_enum(const rocsparse_direction a, const rocsparse_direction b)
{
    ASSERT_TRUE(a == b);
}

template <>
void unit_check_enum(const rocsparse_datatype a, const rocsparse_datatype b)
{
    ASSERT_TRUE(a == b);
}

template <>
void unit_check_enum(const rocsparse_indextype a, const rocsparse_indextype b)
{
    ASSERT_TRUE(a == b);
}

#define MAX_TOL_MULTIPLIER 4

template <typename T>
void near_check_general_template(int64_t            M,
                                 int64_t            N,
                                 const T*           A,
                                 int64_t            LDA,
                                 const T*           B,
                                 int64_t            LDB,
                                 floating_data_t<T> tol = default_tolerance<T>::value)
{
    int                tolm            = 1;
    bool               passed          = true;
    floating_data_t<T> min_passing_tol = 0;
    for(int64_t j = 0; j < N; ++j)
    {
        for(int64_t i = 0; i < M; ++i)
        {
            T compare_val = std::max(rocsparse_abs(A[i + j * LDA] * tol),
                                     10 * std::numeric_limits<T>::epsilon());
#ifdef GOOGLE_TEST
            if(rocsparse_isnan(A[i + j * LDA]))
            {
                ASSERT_TRUE(rocsparse_isnan(B[i + j * LDB]));
            }
            else if(rocsparse_isinf(A[i + j * LDA]))
            {
                ASSERT_TRUE(rocsparse_isinf(B[i + j * LDB]));
            }
            else
#endif
            {
                int k;
                for(k = 1; k <= MAX_TOL_MULTIPLIER; ++k)
                {
                    if(rocsparse_abs(A[i + j * LDA] - B[i + j * LDB]) <= compare_val * k)
                    {
                        break;
                    }
                }

                if(k > MAX_TOL_MULTIPLIER)
                {
                    if(rocsparse_abs(A[i + j * LDA] - B[i + j * LDB]) > compare_val)
                    {
                        if(passed)
                        {
                            std::cerr.precision(12);
                            std::cerr
                                << "ASSERT_NEAR(" << A[i + j * LDA] << ", " << B[i + j * LDB]
                                << ") failed: " << rocsparse_abs(A[i + j * LDA] - B[i + j * LDB])
                                << " exceeds permissive range [" << compare_val << ","
                                << compare_val * MAX_TOL_MULTIPLIER << " ]" << std::endl;
                        }

                        min_passing_tol = std::max(min_passing_tol,
                                                   rocsparse_abs(A[i + j * LDA] - B[i + j * LDB])
                                                       / std::max(rocsparse_abs(A[i + j * LDA]),
                                                                  rocsparse_abs(B[i + j * LDB])));
                        passed          = false;
                    }
                }
                tolm = std::max(tolm, k);
            }
        }
    }
    if(!passed)
    {
        std::cerr << "Test failed with tol = " << tol << ". Relaxing the tolerance to at least "
                  << min_passing_tol << " would make the test pass." << std::endl;
    }
#ifdef GOOGLE_TEST
    ASSERT_TRUE(passed);
#else
    if(!passed)
        exit(EXIT_FAILURE);
#endif
    if(tolm > 1)
    {
        std::cerr << "WARNING near_check has been permissive with a tolerance multiplier equal to "
                  << tolm << std::endl;
    }
}

template <>
void near_check_general_template(int64_t                        M,
                                 int64_t                        N,
                                 const rocsparse_float_complex* A,
                                 int64_t                        LDA,
                                 const rocsparse_float_complex* B,
                                 int64_t                        LDB,
                                 float                          tol)
{
    int   tolm            = 1;
    bool  passed          = true;
    float min_passing_tol = 0;
    for(int64_t j = 0; j < N; ++j)
    {
        for(int64_t i = 0; i < M; ++i)
        {
            float compare_val = std::max(rocsparse_abs(A[i + j * LDA]) * tol,
                                         10 * std::numeric_limits<float>::epsilon());
#ifdef GOOGLE_TEST
            if(rocsparse_isnan(A[i + j * LDA]))
            {
                ASSERT_TRUE(rocsparse_isnan(B[i + j * LDB]));
            }
            else if(rocsparse_isinf(A[i + j * LDA]))
            {
                ASSERT_TRUE(rocsparse_isinf(B[i + j * LDB]));
            }
            else
#endif
            {
                int k;
                for(k = 1; k <= MAX_TOL_MULTIPLIER; ++k)
                {
                    if(rocsparse_abs(A[i + j * LDA] - B[i + j * LDB]) <= compare_val * k)
                    {
                        break;
                    }
                }

                if(k > MAX_TOL_MULTIPLIER)
                {
                    if(rocsparse_abs(A[i + j * LDA] - B[i + j * LDB]) > compare_val)
                    {
                        if(passed)
                        {
                            std::cerr.precision(16);
                            std::cerr
                                << "ASSERT_NEAR(" << A[i + j * LDA] << ", " << B[i + j * LDB]
                                << ") failed: " << rocsparse_abs(A[i + j * LDA] - B[i + j * LDB])
                                << " exceeds permissive range [" << compare_val << ","
                                << compare_val * MAX_TOL_MULTIPLIER << " ]" << std::endl;
                        }

                        float min_passing_tol_tmp = rocsparse_abs(A[i + j * LDA] - B[i + j * LDB])
                                                    / std::max(rocsparse_abs(A[i + j * LDA]),
                                                               rocsparse_abs(B[i + j * LDB]));

                        min_passing_tol = std::max(min_passing_tol, min_passing_tol_tmp);
                        passed          = false;
                    }
                }
                tolm = std::max(tolm, k);
            }
        }
    }
    if(!passed)
    {
        std::cerr << "Test failed with tol = " << tol << ". Relaxing the tolerance to at least "
                  << min_passing_tol << " would make the test pass." << std::endl;
    }
#ifdef GOOGLE_TEST
    ASSERT_TRUE(passed);
#else
    if(!passed)
        exit(EXIT_FAILURE);
#endif
    if(tolm > 1)
    {
        std::cerr << "WARNING near_check has been permissive with a tolerance multiplier equal to "
                  << tolm << std::endl;
    }
}

template <>
void near_check_general_template(int64_t                         M,
                                 int64_t                         N,
                                 const rocsparse_double_complex* A,
                                 int64_t                         LDA,
                                 const rocsparse_double_complex* B,
                                 int64_t                         LDB,
                                 double                          tol)
{
    int    tolm            = 1;
    bool   passed          = true;
    double min_passing_tol = 0;
    for(int64_t j = 0; j < N; ++j)
    {
        for(int64_t i = 0; i < M; ++i)
        {
            double compare_val = std::max(rocsparse_abs(A[i + j * LDA]) * tol,
                                          10 * std::numeric_limits<double>::epsilon());
#ifdef GOOGLE_TEST
            if(rocsparse_isnan(A[i + j * LDA]))
            {
                ASSERT_TRUE(rocsparse_isnan(B[i + j * LDB]));
            }
            else if(rocsparse_isinf(A[i + j * LDA]))
            {
                ASSERT_TRUE(rocsparse_isinf(B[i + j * LDB]));
            }
            else
#endif
            {
                int k;
                for(k = 1; k <= MAX_TOL_MULTIPLIER; ++k)
                {
                    if(rocsparse_abs(A[i + j * LDA] - B[i + j * LDB]) <= compare_val * k)
                    {
                        break;
                    }
                }

                if(k > MAX_TOL_MULTIPLIER)
                {
                    if(rocsparse_abs(A[i + j * LDA] - B[i + j * LDB]) > compare_val)
                    {
                        if(passed)
                        {
                            std::cerr.precision(16);
                            std::cerr
                                << "ASSERT_NEAR(" << A[i + j * LDA] << ", " << B[i + j * LDB]
                                << ") failed: " << rocsparse_abs(A[i + j * LDA] - B[i + j * LDB])
                                << " exceeds permissive range [" << compare_val << ","
                                << compare_val * MAX_TOL_MULTIPLIER << " ]" << std::endl;
                        }

                        double min_passing_tol_tmp = rocsparse_abs(A[i + j * LDA] - B[i + j * LDB])
                                                     / std::max(rocsparse_abs(A[i + j * LDA]),
                                                                rocsparse_abs(B[i + j * LDB]));

                        min_passing_tol = std::max(min_passing_tol, min_passing_tol_tmp);
                        passed          = false;
                    }
                }
                tolm = std::max(tolm, k);
            }
        }
    }
    if(!passed)
    {
        std::cerr << "Test failed with tol = " << tol << ". Relaxing the tolerance to at least "
                  << min_passing_tol << " would make the test pass." << std::endl;
    }
#ifdef GOOGLE_TEST
    ASSERT_TRUE(passed);
#else
    if(!passed)
        exit(EXIT_FAILURE);
#endif
    if(tolm > 1)
    {
        std::cerr << "WARNING near_check has been permissive with a tolerance multiplier equal to "
                  << tolm << std::endl;
    }
}

template <typename T>
void near_check_general(
    int64_t M, int64_t N, const T* A, int64_t LDA, const T* B, int64_t LDB, floating_data_t<T> tol)
{
    near_check_general_template(M, N, A, LDA, B, LDB, tol);
}

#define INSTANTIATE(TYPE)                                       \
    template void near_check_general(int64_t               M,   \
                                     int64_t               N,   \
                                     const TYPE*           A,   \
                                     int64_t               LDA, \
                                     const TYPE*           B,   \
                                     int64_t               LDB, \
                                     floating_data_t<TYPE> tol)

INSTANTIATE(int32_t);
INSTANTIATE(_Float16);
INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex);

#undef INSTANTIATE

void unit_check_garray(rocsparse_indextype ind_type,
                       int64_t             size,
                       const void*         source,
                       const void*         target)
{
    void* s;
    CHECK_HIP_ERROR(rocsparse_hipHostMalloc(&s, rocsparse_indextype_sizeof(ind_type) * size));
    CHECK_HIP_ERROR(
        hipMemcpy(s, source, rocsparse_indextype_sizeof(ind_type) * size, hipMemcpyDeviceToHost));
    void* t;
    CHECK_HIP_ERROR(rocsparse_hipHostMalloc(&t, rocsparse_indextype_sizeof(ind_type) * size));
    CHECK_HIP_ERROR(
        hipMemcpy(t, target, rocsparse_indextype_sizeof(ind_type) * size, hipMemcpyDeviceToHost));
    switch(ind_type)
    {
    case rocsparse_indextype_i32:
    {
        unit_check_segments<int32_t>(size, (const int32_t*)s, (const int32_t*)t);
        break;
    }
    case rocsparse_indextype_i64:
    {
        unit_check_segments<int64_t>(size, (const int64_t*)s, (const int64_t*)t);
        break;
    }
    case rocsparse_indextype_u16:
    {
        break;
    }
    }
    CHECK_HIP_ERROR(rocsparse_hipHostFree(s));
    CHECK_HIP_ERROR(rocsparse_hipHostFree(t));
}

void unit_check_garray(rocsparse_datatype val_type,
                       int64_t            size,
                       const void*        source,
                       const void*        target)
{
    void* s;
    CHECK_HIP_ERROR(rocsparse_hipHostMalloc(&s, rocsparse_datatype_sizeof(val_type) * size));
    CHECK_HIP_ERROR(
        hipMemcpy(s, source, rocsparse_datatype_sizeof(val_type) * size, hipMemcpyDeviceToHost));
    void* t;
    CHECK_HIP_ERROR(rocsparse_hipHostMalloc(&t, rocsparse_datatype_sizeof(val_type) * size));
    CHECK_HIP_ERROR(
        hipMemcpy(t, target, rocsparse_datatype_sizeof(val_type) * size, hipMemcpyDeviceToHost));
    switch(val_type)
    {
    case rocsparse_datatype_f32_r:
    {
        unit_check_segments<float>(size, (const float*)s, (const float*)t);
        break;
    }
    case rocsparse_datatype_f32_c:
    {
        unit_check_segments<rocsparse_float_complex>(
            size, (const rocsparse_float_complex*)s, (const rocsparse_float_complex*)t);
        break;
    }
    case rocsparse_datatype_f64_r:
    {
        unit_check_segments<double>(size, (const double*)s, (const double*)t);
        break;
    }
    case rocsparse_datatype_f64_c:
    {
        unit_check_segments<rocsparse_double_complex>(
            size, (const rocsparse_double_complex*)s, (const rocsparse_double_complex*)t);
        break;
    }
    case rocsparse_datatype_i32_r:
    {
        unit_check_segments<int32_t>(size, (const int32_t*)s, (const int32_t*)t);
        break;
    }
    case rocsparse_datatype_u32_r:
    {
        //      unit_check_segments<uint32_t>(size,(const uint32_t*) source, (const uint32_t*) t);
        break;
    }
    case rocsparse_datatype_f16_r:
    {
        unit_check_segments<_Float16>(size, (const _Float16*)source, (const _Float16*)t);
        break;
    }
    case rocsparse_datatype_i8_r:
    {
        unit_check_segments<int8_t>(size, (const int8_t*)s, (const int8_t*)t);
        break;
    }
    case rocsparse_datatype_u8_r:
    {
        unit_check_segments<uint8_t>(size, (const uint8_t*)source, (const uint8_t*)target);
        break;
    }
    case rocsparse_datatype_bf16_r:
    {
        unit_check_segments<rocsparse_bfloat16>(
            size, (const rocsparse_bfloat16*)source, (const rocsparse_bfloat16*)t);
        break;
    }
    }
    CHECK_HIP_ERROR(rocsparse_hipHostFree(s));
    CHECK_HIP_ERROR(rocsparse_hipHostFree(t));
}
