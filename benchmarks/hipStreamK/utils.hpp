/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2021-2025 Advanced Micro Devices, Inc.
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

//Borrowed from rocWMMA samples
#pragma once

#include <cblas.h>
#include <iostream>
#include <rocwmma/rocwmma.hpp>

using namespace rocwmma;

// Helper macro for HIP errors
#ifndef CHECK_HIP_ERROR
#define CHECK_HIP_ERROR(status)                   \
    if(status != hipSuccess)                      \
    {                                             \
        fprintf(stderr,                           \
                "hip error: '%s'(%d) at %s:%d\n", \
                hipGetErrorString(status),        \
                status,                           \
                __FILE__,                         \
                __LINE__);                        \
        exit(EXIT_FAILURE);                       \
    }
#endif

// Host matrix data random initialization
template <typename DataT>
__host__ static inline void fillRand(DataT* mat, uint32_t m, uint32_t n)
{
    auto randInit = []() {
        srand(time(0));
        return 0u;
    };
    static auto init = randInit();
#pragma omp parallel for
    for(int i = 0; i < m; ++i)
    {
        auto rando = rand() % 5u;
        for(int j = 0; j < n; j++)
        {
            // Assign random integer values within 0-64, alternating
            // sign if the value is a multiple of 3
            auto value     = (rando + j) % 5u;
            mat[i * n + j] = ((value % 3u == 0u) && std::is_signed<DataT>::value)
                                 ? -static_cast<DataT>(value)
                                 : static_cast<DataT>(value);
        }
    }
}

// Host GEMM validation
template <typename LayoutA, typename LayoutB, typename LayoutC, typename LayoutD = LayoutC>
__host__ void gemm_cpu_h(uint32_t         m,
                         uint32_t         n,
                         uint32_t         k,
                         float16_t const* a,
                         float16_t const* b,
                         float16_t const* c,
                         float16_t*       d,
                         float            alpha,
                         float            beta)
{
    std::vector<float> floatA(m * k);
    std::vector<float> floatB(k * n);
    std::vector<float> floatD(m * n);

#pragma omp parallel for
    for(std::size_t i = 0; i != floatA.size(); ++i)
    {
        floatA[i] = static_cast<float>(a[i]);
    }

#pragma omp parallel for
    for(std::size_t i = 0; i != floatB.size(); ++i)
    {
        floatB[i] = static_cast<float>(b[i]);
    }

#pragma omp parallel for
    for(std::size_t i = 0; i != floatD.size(); ++i)
    {
        floatD[i] = static_cast<float>(c[i]);
    }
    bool transA = std::is_same<LayoutA, rocwmma::row_major>::value;
    bool transB = std::is_same<LayoutB, rocwmma::row_major>::value;
    cblas_sgemm(transA ? CblasRowMajor : CblasColMajor,
                transA ? CblasTrans : CblasNoTrans,
                transB ? CblasTrans : CblasNoTrans,
                m,
                n,
                k,
                alpha,
                floatA.data(),
                transA ? k : m,
                floatB.data(),
                transB ? n : k,
                beta,
                floatD.data(),
                m);

#pragma omp parallel for
    for(std::size_t i = 0; i != floatD.size(); ++i)
    {
        d[i] = static_cast<float16_t>(floatD[i]);
    }
}
template <typename LayoutA, typename LayoutB, typename LayoutC, typename LayoutD = LayoutC>
__host__ void gemm_cpu_h(uint32_t     m,
                         uint32_t     n,
                         uint32_t     k,
                         float const* a,
                         float const* b,
                         float const* c,
                         float*       d,
                         float        alpha,
                         float        beta)
{
#pragma omp parallel for
    for(std::size_t i = 0; i != m * n; ++i)
    {
        d[i] = c[i];
    }
    bool transA = std::is_same<LayoutA, rocwmma::row_major>::value;
    bool transB = std::is_same<LayoutB, rocwmma::row_major>::value;
    cblas_sgemm(transA ? CblasRowMajor : CblasColMajor,
                transA ? CblasTrans : CblasNoTrans,
                transB ? CblasTrans : CblasNoTrans,
                m,
                n,
                k,
                alpha,
                a,
                transA ? k : m,
                b,
                transB ? n : k,
                beta,
                d,
                m);
}
