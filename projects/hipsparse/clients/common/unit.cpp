/* ************************************************************************
 * Copyright (C) 2018-2019 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "unit.hpp"

#include <algorithm>
#include <hip/hip_runtime_api.h>
#include <hipsparse.h>
#include <limits>

#ifdef GOOGLE_TEST
#include <gtest/gtest.h>
#else
#ifdef NDEBUG
#undef NDEBUG
#include <assert.h>
#define NDEBUG
#else
#include <assert.h>
#endif
#endif

/* ========================================Gtest Unit Check
 * ==================================================== */

/*! \brief Template: gtest unit compare two matrices float/double/complex */
// Do not put a wrapper over ASSERT_FLOAT_EQ, sincer assert exit the current function NOT the test
// case
// a wrapper will cause the loop keep going

template <>
void unit_check_general(int64_t M, int64_t N, int64_t lda, int8_t* hCPU, int8_t* hGPU)
{
    for(int64_t j = 0; j < N; j++)
    {
        for(int64_t i = 0; i < M; i++)
        {
#ifdef GOOGLE_TEST
            ASSERT_FLOAT_EQ(hCPU[i + j * lda], hGPU[i + j * lda]);
#else
            assert(hCPU[i + j * lda] == hGPU[i + j * lda]);
#endif
        }
    }
}

template <>
void unit_check_general(int64_t M, int64_t N, int64_t lda, float* hCPU, float* hGPU)
{
    for(int64_t j = 0; j < N; j++)
    {
        for(int64_t i = 0; i < M; i++)
        {
#ifdef GOOGLE_TEST
            ASSERT_FLOAT_EQ(hCPU[i + j * lda], hGPU[i + j * lda]);
#else
            assert(hCPU[i + j * lda] == hGPU[i + j * lda]);
#endif
        }
    }
}

template <>
void unit_check_general(int64_t M, int64_t N, int64_t lda, double* hCPU, double* hGPU)
{
    for(int64_t j = 0; j < N; j++)
    {
        for(int64_t i = 0; i < M; i++)
        {
#ifdef GOOGLE_TEST
            ASSERT_DOUBLE_EQ(hCPU[i + j * lda], hGPU[i + j * lda]);
#else
            assert(hCPU[i + j * lda] == hGPU[i + j * lda]);
#endif
        }
    }
}

template <>
void unit_check_general(int64_t M, int64_t N, int64_t lda, hipComplex* hCPU, hipComplex* hGPU)
{
    for(int64_t j = 0; j < N; j++)
    {
        for(int64_t i = 0; i < M; i++)
        {
#ifdef GOOGLE_TEST
            ASSERT_FLOAT_EQ(hCPU[i + j * lda].x, hGPU[i + j * lda].x);
            ASSERT_FLOAT_EQ(hCPU[i + j * lda].y, hGPU[i + j * lda].y);
#else
            assert(hCPU[i + j * lda].x == hGPU[i + j * lda].x);
            assert(hCPU[i + j * lda].y == hGPU[i + j * lda].y);
#endif
        }
    }
}

template <>
void unit_check_general(
    int64_t M, int64_t N, int64_t lda, hipDoubleComplex* hCPU, hipDoubleComplex* hGPU)
{
    for(int64_t j = 0; j < N; j++)
    {
        for(int64_t i = 0; i < M; i++)
        {
#ifdef GOOGLE_TEST
            ASSERT_DOUBLE_EQ(hCPU[i + j * lda].x, hGPU[i + j * lda].x);
            ASSERT_DOUBLE_EQ(hCPU[i + j * lda].y, hGPU[i + j * lda].y);
#else
            assert(hCPU[i + j * lda].x == hGPU[i + j * lda].x);
            assert(hCPU[i + j * lda].y == hGPU[i + j * lda].y);
#endif
        }
    }
}

template <>
void unit_check_general(int64_t M, int64_t N, int64_t lda, int* hCPU, int* hGPU)
{
    for(int64_t j = 0; j < N; j++)
    {
        for(int64_t i = 0; i < M; i++)
        {
#ifdef GOOGLE_TEST
            ASSERT_EQ(hCPU[i + j * lda], hGPU[i + j * lda]);
#else
            assert(hCPU[i + j * lda] == hGPU[i + j * lda]);
#endif
        }
    }
}

template <>
void unit_check_general(int64_t M, int64_t N, int64_t lda, int64_t* hCPU, int64_t* hGPU)
{
    for(int64_t j = 0; j < N; j++)
    {
        for(int64_t i = 0; i < M; i++)
        {
#ifdef GOOGLE_TEST
            ASSERT_EQ(hCPU[i + j * lda], hGPU[i + j * lda]);
#else
            assert(hCPU[i + j * lda] == hGPU[i + j * lda]);
#endif
        }
    }
}

template <>
void unit_check_general(int64_t M, int64_t N, int64_t lda, size_t* hCPU, size_t* hGPU)
{
    for(int64_t j = 0; j < N; j++)
    {
        for(int64_t i = 0; i < M; i++)
        {
#ifdef GOOGLE_TEST
            ASSERT_EQ(hCPU[i + j * lda], hGPU[i + j * lda]);
#else
            assert(hCPU[i + j * lda] == hGPU[i + j * lda]);
#endif
        }
    }
}

/*! \brief Template: gtest unit compare two matrices float/double/complex */
// Do not put a wrapper over ASSERT_FLOAT_EQ, since assert exit the current function NOT the test
// case
// a wrapper will cause the loop keep going

template <>
void unit_check_near(int64_t M, int64_t N, int64_t lda, float* hCPU, float* hGPU)
{
    for(int64_t j = 0; j < N; j++)
    {
        for(int64_t i = 0; i < M; i++)
        {
            float compare_val = std::max(std::abs(hCPU[i + j * lda] * 1e-3f),
                                         10 * std::numeric_limits<float>::epsilon());
#ifdef GOOGLE_TEST
            ASSERT_NEAR(hCPU[i + j * lda], hGPU[i + j * lda], compare_val);
#else
            assert(std::abs(hCPU[i + j * lda] - hGPU[i + j * lda]) < compare_val);
#endif
        }
    }
}

template <>
void unit_check_near(int64_t M, int64_t N, int64_t lda, double* hCPU, double* hGPU)
{
    for(int64_t j = 0; j < N; j++)
    {
        for(int64_t i = 0; i < M; i++)
        {
            double compare_val = std::max(std::abs(hCPU[i + j * lda] * 1e-10),
                                          10 * std::numeric_limits<double>::epsilon());
#ifdef GOOGLE_TEST
            ASSERT_NEAR(hCPU[i + j * lda], hGPU[i + j * lda], compare_val);
#else
            assert(std::abs(hCPU[i + j * lda] - hGPU[i + j * lda]) < compare_val);
#endif
        }
    }
}

template <>
void unit_check_near(int64_t M, int64_t N, int64_t lda, hipComplex* hCPU, hipComplex* hGPU)
{
    for(int64_t j = 0; j < N; j++)
    {
        for(int64_t i = 0; i < M; i++)
        {
            hipComplex compare_val
                = make_hipFloatComplex(std::max(std::abs(hCPU[i + j * lda].x * 1e-3f),
                                                10 * std::numeric_limits<float>::epsilon()),
                                       std::max(std::abs(hCPU[i + j * lda].y * 1e-3f),
                                                10 * std::numeric_limits<float>::epsilon()));

#ifdef GOOGLE_TEST
            ASSERT_NEAR(hCPU[i + j * lda].x, hGPU[i + j * lda].x, compare_val.x);
            ASSERT_NEAR(hCPU[i + j * lda].y, hGPU[i + j * lda].y, compare_val.y);
#else
            assert(std::abs(hCPU[i + j * lda].x - hGPU[i + j * lda].x) < compare_val.x);
            assert(std::abs(hCPU[i + j * lda].y - hGPU[i + j * lda].y) < compare_val.y);
#endif
        }
    }
}

template <>
void unit_check_near(
    int64_t M, int64_t N, int64_t lda, hipDoubleComplex* hCPU, hipDoubleComplex* hGPU)
{
    for(int64_t j = 0; j < N; j++)
    {
        for(int64_t i = 0; i < M; i++)
        {
            hipDoubleComplex compare_val
                = make_hipDoubleComplex(std::max(std::abs(hCPU[i + j * lda].x * 1e-10),
                                                 10 * std::numeric_limits<double>::epsilon()),
                                        std::max(std::abs(hCPU[i + j * lda].y * 1e-10),
                                                 10 * std::numeric_limits<double>::epsilon()));
#ifdef GOOGLE_TEST
            ASSERT_NEAR(hCPU[i + j * lda].x, hGPU[i + j * lda].x, compare_val.x);
            ASSERT_NEAR(hCPU[i + j * lda].y, hGPU[i + j * lda].y, compare_val.y);
#else
            assert(std::abs(hCPU[i + j * lda].x - hGPU[i + j * lda].x) < compare_val.x);
            assert(std::abs(hCPU[i + j * lda].y - hGPU[i + j * lda].y) < compare_val.y);
#endif
        }
    }
}
