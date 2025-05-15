/*! \file */
/* ************************************************************************
 * Copyright (C) 2019-2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse.h"

/*
 * ===========================================================================
 *    level 1 SPARSE
 * ===========================================================================
 */
constexpr double axpyi_gflop_count(rocsparse_int nnz)
{
    return (2.0 * nnz) / 1e9;
}

template <typename I>
constexpr double axpby_gflop_count(I nnz)
{
    return (3.0 * nnz) / 1e9;
}

constexpr double doti_gflop_count(rocsparse_int nnz)
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
constexpr double csrsv_gflop_count(J M, I nnz, rocsparse_diag_type diag)
{
    return (2.0 * nnz + M + (diag == rocsparse_diag_type_non_unit ? M : 0)) / 1e9;
}

template <typename I, typename J>
constexpr double spsv_gflop_count(J M, I nnz, rocsparse_diag_type diag)
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
constexpr double bsrmm_gflop_count(rocsparse_int N,
                                   rocsparse_int nnzb,
                                   rocsparse_int block_dim,
                                   rocsparse_int nnz_C,
                                   bool          beta = false)
{
    return (2.0 * nnzb * block_dim * block_dim * N + (beta ? nnz_C : 0)) / 1e9;
}

constexpr double gebsrmm_gflop_count(rocsparse_int N,
                                     rocsparse_int nnzb,
                                     rocsparse_int row_block_dim,
                                     rocsparse_int col_block_dim,
                                     rocsparse_int nnz_C,
                                     bool          beta = false)
{
    return (2.0 * nnzb * row_block_dim * col_block_dim * N + (beta ? nnz_C : 0)) / 1e9;
}

template <typename I, typename J>
constexpr double csrmm_gflop_count(J N, I nnz_A, I nnz_C, bool beta = false)
{
    // Multiplication by 2 comes from 1 addition and 1 multiplication in product. Multiplication
    // by alpha and beta not counted.
    return (2.0 * nnz_A * N + (beta ? nnz_C : 0)) / 1e9;
}

template <typename I, typename J>
constexpr double spmm_gflop_count(J N, I nnz_A, I nnz_C, bool beta = false)
{
    return csrmm_gflop_count(N, nnz_A, nnz_C, beta);
}

template <rocsparse_format FORMAT>
struct rocsparse_gflop_count
{
    template <typename T, typename I, typename J>
    static constexpr double sddmm(J M, J N, I nnz, J K, bool beta = false);
};

template <>
struct rocsparse_gflop_count<rocsparse_format_coo>
{
    template <typename I, typename J>
    static constexpr double sddmm(J M, J N, I nnz, J K, bool beta = false)
    {
        size_t innz = (size_t)nnz;
        return (innz * ((size_t(K) + (size_t(K) - 1)) + 1 + ((beta) ? 2 : 0))) / 1e9;
    }
};

template <>
struct rocsparse_gflop_count<rocsparse_format_csr>
{
    template <typename I, typename J>
    static constexpr double sddmm(J M, J N, I nnz, J K, bool beta = false)
    {
        return rocsparse_gflop_count<rocsparse_format_coo>::sddmm(M, N, nnz, K, beta);
    }
};

template <>
struct rocsparse_gflop_count<rocsparse_format_csc>
{
    template <typename I, typename J>
    static constexpr double sddmm(J M, J N, I nnz, J K, bool beta = false)
    {
        return rocsparse_gflop_count<rocsparse_format_coo>::sddmm(M, N, nnz, K, beta);
    }
};

template <>
struct rocsparse_gflop_count<rocsparse_format_coo_aos>
{
    template <typename I, typename J>
    static constexpr double sddmm(J M, J N, I nnz, J K, bool beta = false)
    {
        return rocsparse_gflop_count<rocsparse_format_coo>::sddmm(M, N, nnz, K, beta);
    }
};

template <>
struct rocsparse_gflop_count<rocsparse_format_ell>
{
    template <typename I, typename J>
    static constexpr double sddmm(J M, J N, I nnz, J K, bool beta = false)
    {
        return rocsparse_gflop_count<rocsparse_format_coo>::sddmm(M, N, nnz, K, beta);
    }
};

/*
 * ===========================================================================
 *    extra SPARSE
 * ===========================================================================
 */
template <typename T, typename I = rocsparse_int, typename J = rocsparse_int>
constexpr double bsrgemm_gflop_count(J                    Mb,
                                     J                    block_dim,
                                     const T*             alpha,
                                     const I*             bsr_row_ptr_A,
                                     const J*             bsr_col_ind_A,
                                     const I*             bsr_row_ptr_B,
                                     const T*             beta,
                                     const I*             bsr_row_ptr_D,
                                     rocsparse_index_base baseA)
{
    // Flop counter
    double flops = 0.0;

    // Loop over rows of A
    for(J i = 0; i < Mb; ++i)
    {
        if(alpha)
        {
            I row_begin_A = bsr_row_ptr_A[i] - baseA;
            I row_end_A   = bsr_row_ptr_A[i + 1] - baseA;

            // Loop over columns of A
            for(I j = row_begin_A; j < row_end_A; ++j)
            {
                // Current column of A
                J col_A = bsr_col_ind_A[j] - baseA;

                // Count flops generated by alpha * A * B
                flops += 2.0 * block_dim * block_dim * block_dim
                             * (bsr_row_ptr_B[col_A + 1] - bsr_row_ptr_B[col_A])
                         + 1.0;
            }
        }

        if(beta)
        {
            // Count flops generated by beta * D
            flops += block_dim * block_dim * (bsr_row_ptr_D[i + 1] - bsr_row_ptr_D[i]);
        }
    }

    return flops / 1e9;
}

template <typename T>
constexpr double bsrgeam_gflop_count(rocsparse_int block_dim,
                                     rocsparse_int nnzb_A,
                                     rocsparse_int nnzb_B,
                                     rocsparse_int nnzb_C,
                                     const T*      alpha,
                                     const T*      beta)
{
    // Flop counter
    double flops = 0.0;

    if(alpha && beta)
    {
        // Count alpha * A
        flops += static_cast<double>(nnzb_A) * block_dim * block_dim;

        // Count beta * B
        flops += static_cast<double>(nnzb_B) * block_dim * block_dim;

        // Count A + B
        flops += static_cast<double>(nnzb_C) * block_dim * block_dim;
        ;
    }
    else if(!alpha)
    {
        // Count beta * B
        flops += static_cast<double>(nnzb_B) * block_dim * block_dim;
    }
    else
    {
        // Count alpha * A
        flops += static_cast<double>(nnzb_A) * block_dim * block_dim;
    }

    return flops / 1e9;
}

template <typename T, typename I = rocsparse_int, typename J = rocsparse_int>
constexpr double csrgeam_gflop_count(I nnz_A, I nnz_B, I nnz_C, const T* alpha, const T* beta)
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

template <typename T, typename I = rocsparse_int, typename J = rocsparse_int>
constexpr double csrgemm_gflop_count(J                    M,
                                     const T*             alpha,
                                     const I*             csr_row_ptr_A,
                                     const J*             csr_col_ind_A,
                                     const I*             csr_row_ptr_B,
                                     const T*             beta,
                                     const I*             csr_row_ptr_D,
                                     rocsparse_index_base baseA)
{
    // Flop counter
    double flops = 0.0;

    // Loop over rows of A
    for(J i = 0; i < M; ++i)
    {
        if(alpha)
        {
            I row_begin_A = csr_row_ptr_A[i] - baseA;
            I row_end_A   = csr_row_ptr_A[i + 1] - baseA;

            // Loop over columns of A
            for(I j = row_begin_A; j < row_end_A; ++j)
            {
                // Current column of A
                J col_A = csr_col_ind_A[j] - baseA;

                // Count flops generated by alpha * A * B
                flops += 2.0 * (csr_row_ptr_B[col_A + 1] - csr_row_ptr_B[col_A]) + 1.0;
            }
        }

        if(beta)
        {
            // Count flops generated by beta * D
            flops += (csr_row_ptr_D[i + 1] - csr_row_ptr_D[i]);
        }
    }

    return flops / 1e9;
}

#endif // FLOPS_HPP
