/*! \file */
/* ************************************************************************
 * Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
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

/*! \file
 *  \brief flops.hpp provides floating point counts of Sparse Linear Algebra Subprograms
 *  of Level 1, 2 and 3.
 */

#pragma once
#ifndef FLOPS_HPP
#define FLOPS_HPP

#include "hipsparse.h"

// Compute gflops
inline double get_gpu_gflops(double gpu_time_used, double gflop_count)
{
    return gflop_count / gpu_time_used * 1e6;
}

template <typename F, typename... Ts>
inline double get_gpu_gflops(double gpu_time_used, F count, Ts... ts)
{
    return get_gpu_gflops(gpu_time_used, count(ts...));
}

/*
 * ===========================================================================
 *    level 1 SPARSE
 * ===========================================================================
 */
template <typename I>
constexpr double axpyi_gflop_count(I nnz)
{
    return (2.0 * nnz) / 1e9;
}

template <typename I>
constexpr double axpby_gflop_count(I nnz)
{
    return (3.0 * nnz) / 1e9;
}

template <typename I>
constexpr double doti_gflop_count(I nnz)
{
    return (2.0 * nnz) / 1e9;
}

template <typename I>
constexpr double roti_gflop_count(I nnz)
{
    return (6.0 * nnz) / 1e9;
}

/*
 * ===========================================================================
 *    level 2 SPARSE
 * ===========================================================================
 */
template <typename I, typename J>
constexpr double spmv_gflop_count(J M, I nnz, bool beta = false)
{
    return (2.0 * nnz + (beta ? M : 0)) / 1e9;
}

template <typename I, typename J>
constexpr double csrsv_gflop_count(J M, I nnz, hipsparseDiagType_t diag)
{
    return (2.0 * nnz + M + (diag == HIPSPARSE_DIAG_TYPE_NON_UNIT ? M : 0)) / 1e9;
}

template <typename I, typename J>
constexpr double spsv_gflop_count(J M, I nnz, hipsparseDiagType_t diag)
{
    return csrsv_gflop_count(M, nnz, diag);
}

template <typename I>
constexpr double gemvi_gflop_count(I M, I nnz)
{
    return (M + 2.0 * nnz * M) / 1e9;
}

/*
 * ===========================================================================
 *    level 3 SPARSE
 * ===========================================================================
 */
constexpr double bsrmm_gflop_count(int N, int nnzb, int block_dim, int nnz_C, bool beta = false)
{
    return (2.0 * nnzb * block_dim * block_dim * N + (beta ? nnz_C : 0)) / 1e9;
}

template <typename I, typename J>
constexpr double csrmm_gflop_count(J N, I nnz_A, I nnz_C, bool beta = false)
{
    // Multiplication by 2 comes from 1 addition and 1 multiplication in product. Multiplication
    // by alpha and beta not counted.
    return (2.0 * nnz_A * N + (beta ? nnz_C : 0)) / 1e9;
}

template <typename I, typename J>
constexpr double gemmi_gflop_count(J M, I nnz_B, I nnz_C, bool beta = false)
{
    // Multiplication by 2 comes from 1 addition and 1 multiplication in product. Multiplication
    // by alpha and beta not counted.
    return (2.0 * nnz_B * M + (beta ? nnz_C : 0)) / 1e9;
}

template <typename I, typename J>
constexpr double spmm_gflop_count(J N, I nnz_A, I nnz_C, bool beta = false)
{
    return csrmm_gflop_count(N, nnz_A, nnz_C, beta);
}

template <typename I, typename J>
constexpr double sddmm_gflop_count(J K, I nnz, bool beta = false)
{
    return (size_t(nnz) * ((size_t(K) + (size_t(K) - 1)) + 1 + ((beta) ? 2 : 0))) / 1e9;
}

/*
 * ===========================================================================
 *    extra SPARSE
 * ===========================================================================
 */
template <typename T>
constexpr double csrgeam_gflop_count(int nnz_A, int nnz_B, int nnz_C, const T* alpha, const T* beta)
{
    // Flop counter
    double flops = 0.0;

    if(alpha && beta)
    {
        // Count alpha * A
        flops += static_cast<double>(nnz_A);

        // Count beta * B
        flops += static_cast<double>(nnz_B);

        // Count A + B
        flops += static_cast<double>(nnz_C);
    }
    else if(!alpha)
    {
        // Count beta * B
        flops += static_cast<double>(nnz_B);
    }
    else
    {
        // Count alpha * A
        flops += static_cast<double>(nnz_A);
    }

    return flops / 1e9;
}

template <typename T, typename I = int, typename J = int>
constexpr double csrgemm_gflop_count(J                    M,
                                     const I*             csr_row_ptr_A,
                                     const J*             csr_col_ind_A,
                                     const I*             csr_row_ptr_B,
                                     hipsparseIndexBase_t baseA)
{
    // Flop counter
    double flops = 0.0;

    // Loop over rows of A
    for(J i = 0; i < M; ++i)
    {
        I row_begin_A = csr_row_ptr_A[i] - baseA;
        I row_end_A   = csr_row_ptr_A[i + 1] - baseA;

        // Loop over columns of A
        for(I j = row_begin_A; j < row_end_A; ++j)
        {
            // Current column of A
            J col_A = csr_col_ind_A[j] - baseA;

            // Count flops generated by A * B
            flops += 2.0 * (csr_row_ptr_B[col_A + 1] - csr_row_ptr_B[col_A]) + 1.0;
        }
    }

    return flops / 1e9;
}

#endif // FLOPS_HPP
