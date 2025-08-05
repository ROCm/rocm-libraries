/*! \file */
/* ************************************************************************
 * Copyright (C) 2020-2025 Advanced Micro Devices, Inc. All rights Reserved.
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
 *  \brief gbyte.hpp provides data transfer counts of Sparse Linear Algebra Subprograms
 *  of Level 1, 2 and 3.
 */

#pragma once
#ifndef GBYTE_HPP
#define GBYTE_HPP

#include "rocsparse.h"

/*
 * ===========================================================================
 *    level 1 SPARSE
 * ===========================================================================
 */
template <typename T, typename I>
constexpr double axpby_gbyte_count(I nnz)
{
    return (nnz * sizeof(I) + (3.0 * nnz) * sizeof(T)) / 1e9;
}

template <typename X, typename Y, typename I>
constexpr double doti_gbyte_count(I nnz)
{
    return (nnz * sizeof(I) + nnz * (sizeof(X) + sizeof(Y))) / 1e9;
}

template <typename T, typename I>
constexpr double gthr_gbyte_count(I nnz)
{
    return (nnz * sizeof(I) + (2.0 * nnz) * sizeof(T)) / 1e9;
}

template <typename T>
constexpr double gthrz_gbyte_count(rocsparse_int nnz)
{
    return (nnz * sizeof(rocsparse_int) + (2.0 * nnz) * sizeof(T)) / 1e9;
}

template <typename T, typename I>
constexpr double roti_gbyte_count(I nnz)
{
    return (nnz * sizeof(I) + (3.0 * nnz) * sizeof(T)) / 1e9;
}

template <typename T, typename I>
constexpr double sctr_gbyte_count(I nnz)
{
    return (nnz * sizeof(I) + (2.0 * nnz) * sizeof(T)) / 1e9;
}

/*
 * ===========================================================================
 *    level 2 SPARSE
 * ===========================================================================
 */
template <typename A, typename X, typename Y, typename I, typename J>
constexpr double bsrmv_gbyte_count(J mb, J nb, I nnzb, J block_dim, bool beta = false)
{
    return (sizeof(I) * (mb + 1) + sizeof(J) * nnzb + sizeof(A) * nnzb * block_dim * block_dim
            + sizeof(Y) * (mb * block_dim + (beta ? mb * block_dim : 0))
            + sizeof(X) * (nb * block_dim))
           / 1e9;
}

template <typename T, typename I, typename J>
constexpr double bsrmv_gbyte_count(J mb, J nb, I nnzb, J block_dim, bool beta = false)
{
    return bsrmv_gbyte_count<T, T, T>(mb, nb, nnzb, block_dim, beta);
}

template <typename A, typename X, typename Y, typename I>
constexpr double coomv_gbyte_count(I M, I N, int64_t nnz, bool beta = false)
{
    return (sizeof(I) * 2.0 * nnz + sizeof(A) * nnz + sizeof(Y) * (M + (beta ? M : 0))
            + sizeof(X) * N)
           / 1e9;
}

template <typename T, typename I>
constexpr double coomv_gbyte_count(I M, I N, int64_t nnz, bool beta = false)
{
    return coomv_gbyte_count<T, T, T>(M, N, nnz, beta);
}

template <typename A, typename X, typename Y, typename I, typename J>
constexpr double csrmv_gbyte_count(J M, J N, I nnz, bool beta = false)
{
    return (sizeof(I) * (M + 1) + sizeof(J) * nnz + sizeof(A) * nnz
            + sizeof(Y) * (M + (beta ? M : 0)) + sizeof(X) * N)
           / 1e9;
}

template <typename T, typename I, typename J>
constexpr double csrmv_gbyte_count(J M, J N, I nnz, bool beta = false)
{
    return csrmv_gbyte_count<T, T, T>(M, N, nnz, beta);
}

template <typename A, typename X, typename Y, typename I, typename J>
constexpr double cscmv_gbyte_count(J M, J N, I nnz, bool beta = false)
{
    return (sizeof(I) * (N + 1) + sizeof(J) * nnz + sizeof(A) * nnz
            + sizeof(Y) * (M + (beta ? M : 0)) + sizeof(X) * N)
           / 1e9;
}

template <typename T, typename I, typename J>
constexpr double cscmv_gbyte_count(J M, J N, I nnz, bool beta = false)
{
    return cscmv_gbyte_count<T, T, T>(M, N, nnz, beta);
}

template <typename A, typename X, typename Y, typename I>
constexpr double ellmv_gbyte_count(I M, I N, int64_t nnz, bool beta = false)
{
    return (sizeof(I) * nnz + sizeof(A) * nnz + sizeof(Y) * (M + (beta ? M : 0)) + sizeof(X) * N)
           / 1e9;
}

template <typename T, typename I>
constexpr double ellmv_gbyte_count(I M, I N, int64_t nnz, bool beta = false)
{
    return ellmv_gbyte_count<T, T, T>(M, N, nnz, beta);
}

template <typename A, typename X, typename Y, typename I, typename J>
constexpr double
    gebsrmv_gbyte_count(J mb, J nb, I nnzb, J row_block_dim, J col_block_dim, bool beta = false)
{
    return (sizeof(I) * (mb + 1) + sizeof(J) * nnzb
            + sizeof(A) * nnzb * row_block_dim * col_block_dim
            + sizeof(Y) * (mb * row_block_dim + (beta ? mb * row_block_dim : 0))
            + sizeof(X) * (nb * col_block_dim))
           / 1e9;
}

template <typename T, typename I, typename J>
constexpr double
    gebsrmv_gbyte_count(J mb, J nb, I nnzb, J row_block_dim, J col_block_dim, bool beta = false)
{
    return gebsrmv_gbyte_count<T, T, T>(mb, nb, nnzb, row_block_dim, col_block_dim, beta);
}

template <typename T>
constexpr double bsrsv_gbyte_count(rocsparse_int mb, rocsparse_int nnzb, rocsparse_int bsr_dim)
{
    return ((mb + 1 + nnzb) * sizeof(rocsparse_int)
            + (bsr_dim * (mb + mb + nnzb * bsr_dim)) * sizeof(T))
           / 1e9;
}

template <typename T, typename I, typename J>
constexpr double csrsv_gbyte_count(J M, I nnz)
{
    return ((M + 1) * sizeof(I) + nnz * sizeof(J) + (M + M + nnz) * sizeof(T)) / 1e9;
}

template <typename T, typename I>
constexpr double coosv_gbyte_count(I M, int64_t nnz)
{
    return (2 * nnz * sizeof(I) + (M + M + nnz) * sizeof(T)) / 1e9;
}

template <typename T, typename I>
constexpr double gemvi_gbyte_count(I m, I nnz, bool beta = false)
{
    return ((nnz) * sizeof(I) + (m * nnz + nnz + m + (beta ? m : 0)) * sizeof(T)) / 1e9;
}

/*
 * ===========================================================================
 *    level 3 SPARSE
 * ===========================================================================
 */
template <typename T>
constexpr double bsrmm_gbyte_count(rocsparse_int Mb,
                                   rocsparse_int nnzb,
                                   rocsparse_int block_dim,
                                   rocsparse_int nnz_B,
                                   rocsparse_int nnz_C,
                                   bool          beta = false)
{
    //reads
    size_t reads = (Mb + 1 + nnzb) * sizeof(rocsparse_int)
                   + (block_dim * block_dim * nnzb + nnz_B + (beta ? nnz_C : 0)) * sizeof(T);

    //writes
    size_t writes = nnz_C * sizeof(T);

    return (reads + writes) / 1e9;
}

template <typename T>
constexpr double gebsrmm_gbyte_count(rocsparse_int Mb,
                                     rocsparse_int nnzb,
                                     rocsparse_int row_block_dim,
                                     rocsparse_int col_block_dim,
                                     rocsparse_int nnz_B,
                                     rocsparse_int nnz_C,
                                     bool          beta = false)
{
    //reads
    size_t reads
        = (Mb + 1 + nnzb) * sizeof(rocsparse_int)
          + (row_block_dim * col_block_dim * nnzb + nnz_B + (beta ? nnz_C : 0)) * sizeof(T);

    //writes
    size_t writes = nnz_C * sizeof(T);

    return (reads + writes) / 1e9;
}

template <typename A, typename B, typename C, typename I>
constexpr double bellmm_gbyte_count(I Mb, I width, I block_dim, I nnz_B, I nnz_C, bool beta = false)
{
    return (sizeof(A) * Mb * width * block_dim * block_dim + sizeof(I) * Mb * width
            + sizeof(B) * nnz_B + sizeof(C) * (nnz_C + (beta ? nnz_C : 0)))
           / 1e9;
}

template <typename A, typename B, typename C, typename I, typename J>
constexpr double csrmm_gbyte_count(J M, I nnz_A, I nnz_B, I nnz_C, bool beta = false)
{
    return ((M + 1) * sizeof(I) + nnz_A * sizeof(J) + nnz_A * sizeof(A) + nnz_B * sizeof(B)
            + (nnz_C + (beta ? nnz_C : 0)) * sizeof(C))
           / 1e9;
}

template <typename A, typename B, typename C, typename I, typename J>
constexpr double csrmm_batched_gbyte_count(J    M,
                                           I    nnz_A,
                                           I    nnz_B,
                                           I    nnz_C,
                                           J    batch_count_A,
                                           J    batch_count_B,
                                           J    batch_count_C,
                                           bool beta = false)
{
    // read A matrix
    size_t readA = batch_count_A * ((M + 1) * sizeof(I) + nnz_A * sizeof(J) + nnz_A * sizeof(A));

    // read B matrix
    size_t readB = batch_count_B * nnz_B * sizeof(B);

    // read C matrix
    size_t readC = batch_count_C * (beta ? nnz_C : 0) * sizeof(C);

    // write C matrix
    size_t writeC = batch_count_C * nnz_C * sizeof(C);

    return (readA + readB + readC + writeC) / 1e9;
}

template <typename A, typename B, typename C, typename I, typename J>
constexpr double cscmm_gbyte_count(J N, I nnz_A, I nnz_B, I nnz_C, bool beta = false)
{
    return csrmm_gbyte_count<A, B, C>(N, nnz_A, nnz_B, nnz_C, beta);
}

template <typename A, typename B, typename C, typename I, typename J>
constexpr double cscmm_batched_gbyte_count(J    N,
                                           I    nnz_A,
                                           I    nnz_B,
                                           I    nnz_C,
                                           J    batch_count_A,
                                           J    batch_count_B,
                                           J    batch_count_C,
                                           bool beta = false)
{
    // read A matrix
    size_t readA = batch_count_A * ((N + 1) * sizeof(I) + nnz_A * sizeof(J) + nnz_A * sizeof(A));

    // read B matrix
    size_t readB = batch_count_B * nnz_B * sizeof(B);

    // read C matrix
    size_t readC = batch_count_C * (beta ? nnz_C : 0) * sizeof(C);

    // write C matrix
    size_t writeC = batch_count_C * nnz_C * sizeof(C);

    return (readA + readB + readC + writeC) / 1e9;
}

template <typename A, typename B, typename C, typename I>
constexpr double coomm_gbyte_count(int64_t nnz_A, int64_t nnz_B, int64_t nnz_C, bool beta = false)
{
    return (2.0 * nnz_A * sizeof(I) + nnz_A * sizeof(A) + nnz_B * sizeof(B)
            + (nnz_C + (beta ? nnz_C : 0)) * sizeof(C))
           / 1e9;
}

template <typename A, typename B, typename C, typename I>
constexpr double coomm_batched_gbyte_count(I       M,
                                           int64_t nnz_A,
                                           int64_t nnz_B,
                                           int64_t nnz_C,
                                           I       batch_count_A,
                                           I       batch_count_B,
                                           I       batch_count_C,
                                           bool    beta = false)
{
    // read A matrix
    size_t readA = batch_count_A * (nnz_A * sizeof(I) + nnz_A * sizeof(I) + nnz_A * sizeof(A));

    // read B matrix
    size_t readB = batch_count_B * nnz_B * sizeof(B);

    // read C matrix
    size_t readC = batch_count_C * (beta ? nnz_C : 0) * sizeof(C);

    // write C matrix
    size_t writeC = batch_count_C * nnz_C * sizeof(C);

    return (readA + readB + readC + writeC) / 1e9;
}

template <rocsparse_format FORMAT>
struct rocsparse_gbyte_count
{
    template <typename T, typename I, typename J>
    static constexpr double sddmm(J M, J N, I nnz, J K, bool beta = false);
};

template <>
struct rocsparse_gbyte_count<rocsparse_format_csr>
{
    template <typename T, typename I, typename J>
    static constexpr double sddmm(J M, J N, I nnz, J K, bool beta = false)
    {
        size_t l = ((size_t)nnz);
        return ((size_t(M) + 1) * sizeof(I) + l * sizeof(J)
                + (l * (K * 2 + ((beta) ? 1 : 0)) * sizeof(T)))
               / 1e9;
    }
};

template <>
struct rocsparse_gbyte_count<rocsparse_format_csc>
{
    template <typename T, typename I, typename J>
    static constexpr double sddmm(J M, J N, I nnz, J K, bool beta = false)
    {
        size_t l = ((size_t)nnz);
        return ((size_t(N) + 1) * sizeof(I) + l * sizeof(J)
                + (l * (K * 2 + ((beta) ? 1 : 0)) * sizeof(T)))
               / 1e9;
    }
};

template <>
struct rocsparse_gbyte_count<rocsparse_format_coo>
{
    template <typename T, typename I, typename J>
    static constexpr double sddmm(J M, J N, I nnz, J K, bool beta = false)
    {
        size_t l = ((size_t)nnz);
        return (l * 2 * sizeof(I) + l * (K * 2 + ((beta) ? 1 : 0)) * sizeof(T)) / 1e9;
    }
};

template <>
struct rocsparse_gbyte_count<rocsparse_format_coo_aos>
{
    template <typename T, typename I, typename J>
    static constexpr double sddmm(J M, J N, I nnz, J K, bool beta = false)
    {
        size_t l = ((size_t)nnz);
        return (l * 2 * sizeof(I) + (l * (K * 2 + ((beta) ? 1 : 0)) * sizeof(T))) / 1e9;
    }
};

template <>
struct rocsparse_gbyte_count<rocsparse_format_ell>
{
    template <typename T, typename I, typename J>
    static constexpr double sddmm(J M, J N, I nnz, J K, bool beta = false)
    {
        size_t l = ((size_t)nnz);
        return (l * sizeof(J) + (l * (K * 2 + ((beta) ? 1 : 0)) * sizeof(T)))
               / static_cast<size_t>(1e9);
    }
};

/*
 * ===========================================================================
 *    extra SPARSE
 * ===========================================================================
 */
template <typename T>
constexpr double bsrgeam_gbyte_count(rocsparse_int Mb,
                                     rocsparse_int block_dim,
                                     rocsparse_int nnzb_A,
                                     rocsparse_int nnzb_B,
                                     rocsparse_int nnzb_C,
                                     const T*      alpha,
                                     const T*      beta)
{
    double size_A = alpha ? sizeof(rocsparse_int) * (Mb + 1.0 + nnzb_A)
                                + sizeof(T) * block_dim * block_dim * nnzb_A
                          : 0.0;
    double size_B = beta ? sizeof(rocsparse_int) * (Mb + 1.0 + nnzb_B)
                               + sizeof(T) * block_dim * block_dim * nnzb_B
                         : 0.0;
    double size_C
        = sizeof(rocsparse_int) * (Mb + 1.0 + nnzb_C) + sizeof(T) * nnzb_C * block_dim * block_dim;

    return (size_A + size_B + size_C) / 1e9;
}

template <typename T, typename I = rocsparse_int, typename J = rocsparse_int>
constexpr double csrgeam_gbyte_count(J M, I nnz_A, I nnz_B, I nnz_C, const T* alpha, const T* beta)
{
    double size_A = alpha ? (M + 1.0) * sizeof(I) + nnz_A * sizeof(J) + nnz_A * sizeof(T) : 0.0;
    double size_B = beta ? (M + 1.0) * sizeof(I) + nnz_B * sizeof(J) + nnz_B * sizeof(T) : 0.0;
    double size_C = (M + 1.0) * sizeof(I) + nnz_C * sizeof(J) + nnz_C * sizeof(T);

    return (size_A + size_B + size_C) / 1e9;
}

template <typename T, typename I = rocsparse_int, typename J = rocsparse_int>
constexpr double csrgemm_gbyte_count(
    J M, J N, J K, I nnz_A, I nnz_B, I nnz_C, I nnz_D, const T* alpha, const T* beta)
{
    double size_A = alpha ? (M + 1.0) * sizeof(I) + nnz_A * sizeof(J) + nnz_A * sizeof(T) : 0.0;
    double size_B = alpha ? (K + 1.0) * sizeof(I) + nnz_B * sizeof(J) + nnz_B * sizeof(T) : 0.0;
    double size_C = (M + 1.0) * sizeof(I) + nnz_C * sizeof(J) + nnz_C * sizeof(T);
    double size_D = beta ? (M + 1.0) * sizeof(I) + nnz_D * sizeof(J) + nnz_D * sizeof(T) : 0.0;

    return (size_A + size_B + size_C + size_D) / 1e9;
}

template <typename T, typename I = rocsparse_int, typename J = rocsparse_int>
constexpr double bsrgemm_gbyte_count(J        Mb,
                                     J        Nb,
                                     J        Kb,
                                     J        block_dim,
                                     I        nnzb_A,
                                     I        nnzb_B,
                                     I        nnzb_C,
                                     I        nnzb_D,
                                     const T* alpha,
                                     const T* beta)
{
    double size_A = alpha ? sizeof(I) * (Mb + 1.0) + sizeof(J) * nnzb_A
                                + sizeof(T) * block_dim * block_dim * nnzb_A
                          : 0.0;
    double size_B = alpha ? sizeof(I) * (Kb + 1.0) + sizeof(J) * nnzb_B
                                + sizeof(T) * block_dim * block_dim * nnzb_B
                          : 0.0;
    double size_C
        = sizeof(I) * (Mb + 1.0) + sizeof(J) * nnzb_C + sizeof(T) * block_dim * block_dim * nnzb_C;
    double size_D = beta ? sizeof(I) * (Mb + 1.0) + sizeof(J) * nnzb_D
                               + sizeof(T) * block_dim * block_dim * nnzb_D
                         : 0.0;

    return (size_A + size_B + size_C + size_D) / 1e9;
}

/*
 * ===========================================================================
 *    precond SPARSE
 * ===========================================================================
 */
template <typename T>
constexpr double bsric0_gbyte_count(rocsparse_int Mb, rocsparse_int block_dim, rocsparse_int nnzb)
{
    return ((Mb + 1 + nnzb) * sizeof(rocsparse_int)
            + 2.0 * block_dim * block_dim * nnzb * sizeof(T))
           / 1e9;
}

template <typename T>
constexpr double bsrilu0_gbyte_count(rocsparse_int Mb, rocsparse_int block_dim, rocsparse_int nnzb)
{
    return ((Mb + 1 + nnzb) * sizeof(rocsparse_int)
            + 2.0 * block_dim * block_dim * nnzb * sizeof(T))
           / 1e9;
}

template <typename T>
constexpr double csric0_gbyte_count(rocsparse_int M, rocsparse_int nnz)
{
    return ((M + 1 + nnz) * sizeof(rocsparse_int) + 2.0 * nnz * sizeof(T)) / 1e9;
}

template <typename T>
constexpr double csrilu0_gbyte_count(rocsparse_int M, rocsparse_int nnz)
{
    return ((M + 1 + nnz) * sizeof(rocsparse_int) + 2.0 * nnz * sizeof(T)) / 1e9;
}

template <typename T>
constexpr double gtsv_gbyte_count(rocsparse_int M, rocsparse_int N)
{
    return ((3 * M + 2 * M * N) * sizeof(T)) / 1e9;
}

template <typename T>
constexpr double gtsv_strided_batch_gbyte_count(rocsparse_int M, rocsparse_int N)
{
    return ((3 * M * N + 2 * M * N) * sizeof(T)) / 1e9;
}

template <typename T>
constexpr double gtsv_interleaved_batch_gbyte_count(rocsparse_int M, rocsparse_int N)
{
    return ((3 * M * N + 2 * M * N) * sizeof(T)) / 1e9;
}

template <typename T>
constexpr double gpsv_interleaved_batch_gbyte_count(rocsparse_int M, rocsparse_int N)
{
    return ((5 * M * N + 2 * M * N) * sizeof(T)) / 1e9;
}

/*
 * ===========================================================================
 *    conversion SPARSE
 * ===========================================================================
 */
template <typename T>
constexpr double nnz_gbyte_count(rocsparse_int M, rocsparse_int N, rocsparse_direction dir)
{
    return ((M * N) * sizeof(T)
            + ((rocsparse_direction_row == dir) ? M : N) * sizeof(rocsparse_int))
           / 1e9;
}

template <rocsparse_direction DIRA, typename T, typename I, typename J>
constexpr double dense2csx_gbyte_count(J M, J N, I nnz)
{
    const J      L             = (rocsparse_direction_row == DIRA) ? M : N;
    const size_t write_csx_ptr = (L + 1) * sizeof(I);
    const size_t read_csx_ptr  = (L + 1) * sizeof(I);
    const size_t build_csx_ptr = write_csx_ptr + read_csx_ptr;

    const size_t write_csx  = nnz * sizeof(T) + nnz * sizeof(J) + (L + 1) * sizeof(I);
    const size_t read_dense = M * N * sizeof(T);
    return (read_dense + build_csx_ptr + write_csx) / 1e9;
}

template <typename T, typename I>
constexpr double dense2coo_gbyte_count(I M, I N, I nnz)
{
    size_t reads  = (M * N) * sizeof(T);
    size_t writes = 2 * nnz * sizeof(I) + nnz * sizeof(T);

    return (reads + writes) / 1e9;
}

template <typename T>
constexpr double prune_dense2csr_gbyte_count(rocsparse_int M, rocsparse_int N, rocsparse_int nnz)
{
    size_t reads = M * N * sizeof(T);

    size_t writes = (M + 1 + nnz) * sizeof(rocsparse_int) + nnz * sizeof(T);

    return (reads + writes) / 1e9;
}

template <typename T>
constexpr double
    prune_dense2csr_by_percentage_gbyte_count(rocsparse_int M, rocsparse_int N, rocsparse_int nnz)
{
    size_t reads = M * N * sizeof(T);

    size_t writes = (M + 1 + nnz) * sizeof(rocsparse_int) + nnz * sizeof(T);

    return (reads + writes) / 1e9;
}

template <rocsparse_direction DIRA, typename T, typename I, typename J>
constexpr double csx2dense_gbyte_count(J M, J N, I nnz)
{
    const J      L        = (rocsparse_direction_row == DIRA) ? M : N;
    const size_t read_csx = nnz * sizeof(T) + nnz * sizeof(J) + (L + 1) * sizeof(I);
    const size_t write_dense
        = M * N * sizeof(T) + nnz * sizeof(T); // set to zero + nnz assignments.
    return (read_csx + write_dense) / 1e9;
}

template <typename T, typename I>
constexpr double coo2dense_gbyte_count(I M, I N, I nnz)
{
    size_t reads  = 2 * nnz * sizeof(I) + nnz * sizeof(T);
    size_t writes = (M * N) * sizeof(T);

    return (reads + writes) / 1e9;
}

template <typename T>
constexpr double csr2coo_gbyte_count(rocsparse_int M, rocsparse_int nnz)
{
    return (M + 1 + nnz) * sizeof(rocsparse_int) / 1e9;
}

template <typename T>
constexpr double coo2csr_gbyte_count(rocsparse_int M, rocsparse_int nnz)
{
    return (M + 1 + nnz) * sizeof(rocsparse_int) / 1e9;
}

template <typename T>
constexpr double csr2csc_gbyte_count(rocsparse_int    M,
                                     rocsparse_int    N,
                                     rocsparse_int    nnz,
                                     rocsparse_action action)
{
    return ((M + N + 2 + 2.0 * nnz) * sizeof(rocsparse_int)
            + (action == rocsparse_action_numeric ? (2.0 * nnz) * sizeof(T) : 0.0))
           / 1e9;
}

template <typename T>
constexpr double gebsr2gebsc_gbyte_count(rocsparse_int    Mb,
                                         rocsparse_int    Nb,
                                         rocsparse_int    nnzb,
                                         rocsparse_int    row_block_dim,
                                         rocsparse_int    col_block_dim,
                                         rocsparse_action action)
{
    return ((Mb + Nb + 2 + 2.0 * nnzb) * sizeof(rocsparse_int)
            + (action == rocsparse_action_numeric
                   ? (2.0 * nnzb * row_block_dim * col_block_dim) * sizeof(T)
                   : 0.0))
           / 1e9;
}

template <typename T>
constexpr double csr2ell_gbyte_count(rocsparse_int M, rocsparse_int nnz, rocsparse_int ell_nnz)
{
    return ((M + 1.0 + ell_nnz) * sizeof(rocsparse_int) + (nnz + ell_nnz) * sizeof(T)) / 1e9;
}

template <typename T>
constexpr double ell2csr_gbyte_count(rocsparse_int M, rocsparse_int csr_nnz, rocsparse_int ell_nnz)
{
    return ((M + 1.0 + ell_nnz) * sizeof(rocsparse_int) + (csr_nnz + ell_nnz) * sizeof(T)) / 1e9;
}

template <typename T>
constexpr double csr2hyb_gbyte_count(rocsparse_int M,
                                     rocsparse_int nnz,
                                     rocsparse_int ell_nnz,
                                     rocsparse_int coo_nnz)
{
    return ((M + 1.0 + ell_nnz + 2.0 * coo_nnz) * sizeof(rocsparse_int)
            + (nnz + ell_nnz + coo_nnz) * sizeof(T))
           / 1e9;
}

template <typename T>
constexpr double hyb2csr_gbyte_count(rocsparse_int M,
                                     rocsparse_int csr_nnz,
                                     rocsparse_int ell_nnz,
                                     rocsparse_int coo_nnz)
{
    return ((M + 1.0 + csr_nnz + ell_nnz + 2.0 * coo_nnz) * sizeof(rocsparse_int)
            + (csr_nnz + ell_nnz + coo_nnz) * sizeof(T))
           / 1e9;
}

template <typename T>
constexpr double bsr2csr_gbyte_count(rocsparse_int Mb, rocsparse_int block_dim, rocsparse_int nnzb)
{
    // reads
    size_t reads
        = nnzb * block_dim * block_dim * sizeof(T) + (Mb + 1 + nnzb) * sizeof(rocsparse_int);

    // writes
    size_t writes = nnzb * block_dim * block_dim * sizeof(T)
                    + (Mb * block_dim + 1 + nnzb * block_dim * block_dim) * sizeof(rocsparse_int);

    return (reads + writes) / 1e9;
}

template <typename T>
constexpr double gebsr2csr_gbyte_count(rocsparse_int Mb,
                                       rocsparse_int row_block_dim,
                                       rocsparse_int col_block_dim,
                                       rocsparse_int nnzb)
{
    // reads
    size_t reads = nnzb * row_block_dim * col_block_dim * sizeof(T)
                   + (Mb + 1 + nnzb) * sizeof(rocsparse_int);

    // writes
    size_t writes
        = nnzb * row_block_dim * col_block_dim * sizeof(T)
          + (Mb * row_block_dim + 1 + nnzb * row_block_dim * col_block_dim) * sizeof(rocsparse_int);

    return (reads + writes) / 1e9;
}

template <typename T>
constexpr double gebsr2gebsr_gbyte_count(rocsparse_int Mb_A,
                                         rocsparse_int Mb_C,
                                         rocsparse_int row_block_dim_A,
                                         rocsparse_int col_block_dim_A,
                                         rocsparse_int row_block_dim_C,
                                         rocsparse_int col_block_dim_C,
                                         rocsparse_int nnzb_A,
                                         rocsparse_int nnzb_C)
{
    // reads
    size_t reads = nnzb_A * row_block_dim_A * col_block_dim_A * sizeof(T)
                   + (Mb_A + 1 + nnzb_A) * sizeof(rocsparse_int);

    // writes
    size_t writes = nnzb_C * row_block_dim_C * col_block_dim_C * sizeof(T)
                    + (Mb_C + 1 + nnzb_C) * sizeof(rocsparse_int);

    return (reads + writes) / 1e9;
}

template <typename T>
constexpr double csr2bsr_gbyte_count(rocsparse_int M,
                                     rocsparse_int Mb,
                                     rocsparse_int nnz,
                                     rocsparse_int nnzb,
                                     rocsparse_int block_dim)
{
    // reads
    size_t reads = (M + 1 + nnz) * sizeof(rocsparse_int) + nnz * sizeof(T);

    // writes
    size_t writes = (Mb + 1 + nnzb * block_dim * block_dim) * sizeof(rocsparse_int)
                    + (nnzb * block_dim * block_dim) * sizeof(T);

    return (reads + writes) / 1e9;
}

template <typename T>
constexpr double csr2gebsr_gbyte_count(rocsparse_int M,
                                       rocsparse_int Mb,
                                       rocsparse_int nnz,
                                       rocsparse_int nnzb,
                                       rocsparse_int row_block_dim,
                                       rocsparse_int col_block_dim)
{
    // reads
    size_t reads = (M + 1 + nnz) * sizeof(rocsparse_int) + nnz * sizeof(T);

    // writes
    size_t writes = (Mb + 1 + nnzb * row_block_dim * col_block_dim) * sizeof(rocsparse_int)
                    + (nnzb * row_block_dim * col_block_dim) * sizeof(T);

    return (reads + writes) / 1e9;
}

template <typename T>
constexpr double
    csr2csr_compress_gbyte_count(rocsparse_int M, rocsparse_int nnz_A, rocsparse_int nnz_C)
{
    size_t reads = (M + 1 + nnz_A) * sizeof(rocsparse_int) + nnz_A * sizeof(T);

    size_t writes = (M + 1 + nnz_C) * sizeof(rocsparse_int) + nnz_C * sizeof(T);

    return (reads + writes) / 1e9;
}

template <typename T>
constexpr double
    prune_csr2csr_gbyte_count(rocsparse_int M, rocsparse_int nnz_A, rocsparse_int nnz_C)
{
    // reads
    size_t reads = (M + 1 + nnz_A) * sizeof(rocsparse_int) + nnz_A * sizeof(T);

    // writes
    size_t writes = (M + 1 + nnz_C) * sizeof(rocsparse_int) + nnz_C * sizeof(T);

    return (reads + writes) / 1e9;
}

template <typename T>
constexpr double prune_csr2csr_by_percentage_gbyte_count(rocsparse_int M,
                                                         rocsparse_int nnz_A,
                                                         rocsparse_int nnz_C)
{
    // reads
    size_t reads = (M + 1 + nnz_A) * sizeof(rocsparse_int) + nnz_A * sizeof(T);

    // writes
    size_t writes = (M + 1 + nnz_C) * sizeof(rocsparse_int) + nnz_C * sizeof(T);

    return (reads + writes) / 1e9;
}

template <typename T>
constexpr double identity_gbyte_count(rocsparse_int N)
{
    return N * sizeof(rocsparse_int) / 1e9;
}

template <typename T>
constexpr double inverse_permutation_gbyte_count(rocsparse_int N)
{
    return (sizeof(rocsparse_int) * N * 2) / 1e9;
}

template <typename T>
constexpr double csrsort_gbyte_count(rocsparse_int M, rocsparse_int nnz, bool permute)
{
    return ((2.0 * M + 2.0 + 2.0 * nnz + (permute ? 2.0 * nnz : 0.0)) * sizeof(rocsparse_int))
           / 1e9;
}

template <typename T>
constexpr double cscsort_gbyte_count(rocsparse_int N, rocsparse_int nnz, bool permute)
{
    return ((2.0 * N + 2.0 + 2.0 * nnz + (permute ? 2.0 * nnz : 0.0)) * sizeof(rocsparse_int))
           / 1e9;
}

template <typename T>
constexpr double coosort_gbyte_count(rocsparse_int nnz, bool permute)
{
    return ((4.0 * nnz + (permute ? 2.0 * nnz : 0.0)) * sizeof(rocsparse_int)) / 1e9;
}

/*
 * ===========================================================================
 *    utility SPARSE
 * ===========================================================================
 */
template <typename T, typename I, typename J>
constexpr double check_matrix_csr_gbyte_count(J m, I nnz)
{
    return (sizeof(I) * (m + 1) + sizeof(J) * nnz + sizeof(T) * nnz) / 1e9;
}

template <typename T, typename I, typename J>
constexpr double check_matrix_csc_gbyte_count(J n, I nnz)
{
    return check_matrix_csr_gbyte_count<T>(n, nnz);
}

template <typename T, typename I>
constexpr double check_matrix_coo_gbyte_count(I nnz)
{
    return (2.0 * sizeof(I) * nnz + sizeof(T) * nnz) / 1e9;
}

template <typename T, typename I, typename J>
constexpr double check_matrix_gebsr_gbyte_count(J mb, I nnzb, J row_block_dim, J col_block_dim)
{
    return (sizeof(I) * (mb + 1) + sizeof(J) * nnzb
            + sizeof(T) * nnzb * row_block_dim * col_block_dim)
           / 1e9;
}

template <typename T, typename I, typename J>
constexpr double check_matrix_gebsc_gbyte_count(J nb, I nnzb, J row_block_dim, J col_block_dim)
{
    return check_matrix_gebsr_gbyte_count<T>(nb, nnzb, row_block_dim, col_block_dim);
}

template <typename T, typename I>
constexpr double check_matrix_ell_gbyte_count(I ell_nnz)
{
    return (sizeof(I) * ell_nnz + sizeof(T) * ell_nnz) / 1e9;
}

template <typename T, typename I>
constexpr double check_matrix_hyb_gbyte_count(I ell_nnz, I coo_nnz)
{
    return (sizeof(I) * (ell_nnz + 2.0 * coo_nnz) + sizeof(T) * (ell_nnz + coo_nnz)) / 1e9;
}

#endif // GBYTE_HPP
