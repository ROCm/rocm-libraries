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
 *  \brief gbyte.hpp provides data transfer counts of Sparse Linear Algebra Subprograms
 *  of Level 1, 2 and 3.
 */

#pragma once
#ifndef GBYTE_HPP
#define GBYTE_HPP

// Compute gbytes
inline double get_gpu_gbyte(double gpu_time_used, double gbyte_count)
{
    return gbyte_count / gpu_time_used * 1e6;
}

template <typename F, typename... Ts>
inline double get_gpu_gbyte(double gpu_time_used, F count, Ts... ts)
{
    return get_gpu_gbyte(gpu_time_used, count(ts...));
}

inline double get_gpu_time_msec(double gpu_time_used)
{
    return gpu_time_used / 1e3;
}

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

template <typename T, typename I>
constexpr double gthrz_gbyte_count(I nnz)
{
    return (nnz * sizeof(I) + (2.0 * nnz) * sizeof(T)) / 1e9;
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

template <typename T>
constexpr double bsrsv_gbyte_count(int mb, int nnzb, int bsr_dim)
{
    return ((mb + 1 + nnzb) * sizeof(int) + (bsr_dim * (mb + mb + nnzb * bsr_dim)) * sizeof(T))
           / 1e9;
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
constexpr double
    bsrmm_gbyte_count(int Mb, int nnzb, int block_dim, int nnz_B, int nnz_C, bool beta = false)
{
    //reads
    size_t reads = (Mb + 1 + nnzb) * sizeof(int)
                   + (block_dim * block_dim * nnzb + nnz_B + (beta ? nnz_C : 0)) * sizeof(T);

    //writes
    size_t writes = nnz_C * sizeof(T);

    return (reads + writes) / 1e9;
}

template <typename T, typename I, typename J>
constexpr double csrmm_gbyte_count(J M, I nnz_A, I nnz_B, I nnz_C, bool beta = false)
{
    return ((M + 1) * sizeof(I) + nnz_A * sizeof(J)
            + (nnz_A + nnz_B + nnz_C + (beta ? nnz_C : 0)) * sizeof(T))
           / 1e9;
}

template <typename T, typename I, typename J>
constexpr double cscmm_gbyte_count(J N, I nnz_A, I nnz_B, I nnz_C, bool beta = false)
{
    return csrmm_gbyte_count<T>(N, nnz_A, nnz_B, nnz_C, beta);
}

template <typename T, typename I>
constexpr double coomm_gbyte_count(int64_t nnz_A, int64_t nnz_B, int64_t nnz_C, bool beta = false)
{
    return (2.0 * nnz_A * sizeof(I) + (nnz_A + nnz_B + nnz_C + (beta ? nnz_C : 0)) * sizeof(T))
           / 1e9;
}

template <typename T, typename I, typename J>
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
    size_t readA = batch_count_A * ((M + 1) * sizeof(I) + nnz_A * sizeof(J) + nnz_A * sizeof(T));

    // read B matrix
    size_t readB = batch_count_B * nnz_B * sizeof(T);

    // read C matrix
    size_t readC = batch_count_C * (beta ? nnz_C : 0) * sizeof(T);

    // write C matrix
    size_t writeC = batch_count_C * nnz_C * sizeof(T);

    return (readA + readB + readC + writeC) / 1e9;
}

template <typename T, typename I, typename J>
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
    size_t readA = batch_count_A * ((N + 1) * sizeof(I) + nnz_A * sizeof(J) + nnz_A * sizeof(T));

    // read B matrix
    size_t readB = batch_count_B * nnz_B * sizeof(T);

    // read C matrix
    size_t readC = batch_count_C * (beta ? nnz_C : 0) * sizeof(T);

    // write C matrix
    size_t writeC = batch_count_C * nnz_C * sizeof(T);

    return (readA + readB + readC + writeC) / 1e9;
}

template <typename T, typename I>
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
    size_t readA = batch_count_A * (nnz_A * sizeof(I) + nnz_A * sizeof(I) + nnz_A * sizeof(T));

    // read B matrix
    size_t readB = batch_count_B * nnz_B * sizeof(T);

    // read C matrix
    size_t readC = batch_count_C * (beta ? nnz_C : 0) * sizeof(T);

    // write C matrix
    size_t writeC = batch_count_C * nnz_C * sizeof(T);

    return (readA + readB + readC + writeC) / 1e9;
}

template <typename T, typename I, typename J>
constexpr double gemmi_gbyte_count(J N, I nnz_B, I nnz_A, I nnz_C, bool beta = false)
{
    return ((N + 1) * sizeof(I) + nnz_B * sizeof(J)
            + (nnz_B + nnz_A + nnz_C + (beta ? nnz_C : 0)) * sizeof(T))
           / 1e9;
}

template <typename T, typename I, typename J>
constexpr double sddmm_csr_gbyte_count(J M, J N, J K, I nnz, bool beta = false)
{
    return ((size_t(M) + 1) * sizeof(I) + size_t(nnz) * sizeof(J)
            + (size_t(nnz) * (K * 2 + ((beta) ? 1 : 0)) * sizeof(T)))
           / 1e9;
}

template <typename T, typename I, typename J>
constexpr double sddmm_csc_gbyte_count(J M, J N, J K, I nnz, bool beta = false)
{
    return ((size_t(N) + 1) * sizeof(I) + size_t(nnz) * sizeof(J)
            + (size_t(nnz) * (K * 2 + ((beta) ? 1 : 0)) * sizeof(T)))
           / 1e9;
}

template <typename T, typename I, typename J>
constexpr double sddmm_coo_gbyte_count(J M, J N, J K, I nnz, bool beta = false)
{
    return (size_t(nnz) * 2 * sizeof(I) + size_t(nnz) * (K * 2 + ((beta) ? 1 : 0)) * sizeof(T))
           / 1e9;
}

template <typename T, typename I, typename J>
constexpr double sddmm_coo_aos_gbyte_count(J M, J N, J K, I nnz, bool beta = false)
{
    return (size_t(nnz) * 2 * sizeof(I) + (size_t(nnz) * (K * 2 + ((beta) ? 1 : 0)) * sizeof(T)))
           / 1e9;
}

/*
 * ===========================================================================
 *    precond SPARSE
 * ===========================================================================
 */
template <typename T>
constexpr double bsric0_gbyte_count(int Mb, int block_dim, int nnzb)
{
    return ((Mb + 1 + nnzb) * sizeof(int) + 2.0 * block_dim * block_dim * nnzb * sizeof(T)) / 1e9;
}

template <typename T>
constexpr double bsrilu0_gbyte_count(int Mb, int block_dim, int nnzb)
{
    return ((Mb + 1 + nnzb) * sizeof(int) + 2.0 * block_dim * block_dim * nnzb * sizeof(T)) / 1e9;
}

template <typename T>
constexpr double csric0_gbyte_count(int M, int nnz)
{
    return ((M + 1 + nnz) * sizeof(int) + 2.0 * nnz * sizeof(T)) / 1e9;
}

template <typename T>
constexpr double csrilu0_gbyte_count(int M, int nnz)
{
    return ((M + 1 + nnz) * sizeof(int) + 2.0 * nnz * sizeof(T)) / 1e9;
}

template <typename T>
constexpr double gtsv_gbyte_count(int M, int N)
{
    return ((3 * M + 2 * M * N) * sizeof(T)) / 1e9;
}

template <typename T>
constexpr double gtsv_strided_batch_gbyte_count(int M, int N)
{
    return ((3 * M * N + 2 * M * N) * sizeof(T)) / 1e9;
}

template <typename T>
constexpr double gtsv_interleaved_batch_gbyte_count(int M, int N)
{
    return ((3 * M * N + 2 * M * N) * sizeof(T)) / 1e9;
}

template <typename T>
constexpr double gpsv_interleaved_batch_gbyte_count(int M, int N)
{
    return ((5 * M * N + 2 * M * N) * sizeof(T)) / 1e9;
}

/*
 * ===========================================================================
 *    conversion SPARSE
 * ===========================================================================
 */
template <typename T>
constexpr double nnz_gbyte_count(int M, int N, hipsparseDirection_t dir)
{
    return ((M * N) * sizeof(T) + ((dir == HIPSPARSE_DIRECTION_ROW) ? M : N) * sizeof(int)) / 1e9;
}

template <typename T>
constexpr double bsr2csr_gbyte_count(int Mb, int block_dim, int nnzb)
{
    // reads
    size_t reads = nnzb * block_dim * block_dim * sizeof(T) + (Mb + 1 + nnzb) * sizeof(int);

    // writes
    size_t writes = nnzb * block_dim * block_dim * sizeof(T)
                    + (Mb * block_dim + 1 + nnzb * block_dim * block_dim) * sizeof(int);

    return (reads + writes) / 1e9;
}

template <typename T>
constexpr double csr2coo_gbyte_count(int M, int nnz)
{
    return (M + 1 + nnz) * sizeof(int) / 1e9;
}

template <typename T>
constexpr double coo2csr_gbyte_count(int M, int nnz)
{
    return (M + 1 + nnz) * sizeof(int) / 1e9;
}

template <typename T>
constexpr double csr2csc_gbyte_count(int M, int N, int nnz, hipsparseAction_t action)
{
    return ((M + N + 2 + 2.0 * nnz) * sizeof(int)
            + (action == HIPSPARSE_ACTION_NUMERIC ? (2.0 * nnz) * sizeof(T) : 0.0))
           / 1e9;
}

template <typename T>
constexpr double csr2hyb_gbyte_count(int M, int nnz, int ell_nnz, int coo_nnz)
{
    return ((M + 1.0 + ell_nnz + 2.0 * coo_nnz) * sizeof(int)
            + (nnz + ell_nnz + coo_nnz) * sizeof(T))
           / 1e9;
}

template <typename T>
constexpr double hyb2csr_gbyte_count(int M, int csr_nnz, int ell_nnz, int coo_nnz)
{
    return ((M + 1.0 + csr_nnz + ell_nnz + 2.0 * coo_nnz) * sizeof(int)
            + (csr_nnz + ell_nnz + coo_nnz) * sizeof(T))
           / 1e9;
}

template <typename T>
constexpr double csr2bsr_gbyte_count(int M, int Mb, int nnz, int nnzb, int block_dim)
{
    // reads
    size_t reads = (M + 1 + nnz) * sizeof(int) + nnz * sizeof(T);

    // writes
    size_t writes = (Mb + 1 + nnzb * block_dim * block_dim) * sizeof(int)
                    + (nnzb * block_dim * block_dim) * sizeof(T);

    return (reads + writes) / 1e9;
}

template <typename T>
constexpr double
    csr2gebsr_gbyte_count(int M, int Mb, int nnz, int nnzb, int row_block_dim, int col_block_dim)
{
    // reads
    size_t reads = (M + 1 + nnz) * sizeof(int) + nnz * sizeof(T);

    // writes
    size_t writes = (Mb + 1 + nnzb * row_block_dim * col_block_dim) * sizeof(int)
                    + (nnzb * row_block_dim * col_block_dim) * sizeof(T);

    return (reads + writes) / 1e9;
}

template <typename T>
constexpr double csr2csr_compress_gbyte_count(int M, int nnz_A, int nnz_C)
{
    size_t reads = (M + 1 + nnz_A) * sizeof(int) + nnz_A * sizeof(T);

    size_t writes = (M + 1 + nnz_C) * sizeof(int) + nnz_C * sizeof(T);

    return (reads + writes) / 1e9;
}

template <hipsparseDirection_t DIRA, typename T, typename I, typename J>
constexpr double csx2dense_gbyte_count(J M, J N, I nnz)
{
    J      L           = (DIRA == HIPSPARSE_DIRECTION_ROW) ? M : N;
    size_t read_csx    = nnz * sizeof(T) + nnz * sizeof(J) + (L + 1) * sizeof(I);
    size_t write_dense = M * N * sizeof(T) + nnz * sizeof(T);
    return (read_csx + write_dense) / 1e9;
}

template <hipsparseDirection_t DIRA, typename T, typename I, typename J>
constexpr double dense2csx_gbyte_count(J M, J N, I nnz)
{
    J      L             = (DIRA == HIPSPARSE_DIRECTION_ROW) ? M : N;
    size_t write_csx_ptr = (L + 1) * sizeof(I);
    size_t read_csx_ptr  = (L + 1) * sizeof(I);
    size_t build_csx_ptr = write_csx_ptr + read_csx_ptr;

    size_t write_csx  = nnz * sizeof(T) + nnz * sizeof(J) + (L + 1) * sizeof(I);
    size_t read_dense = M * N * sizeof(T);
    return (read_dense + build_csx_ptr + write_csx) / 1e9;
}

template <typename T, typename I>
constexpr double dense2coo_gbyte_count(I M, I N, I nnz)
{
    size_t reads  = (M * N) * sizeof(T);
    size_t writes = 2 * nnz * sizeof(I) + nnz * sizeof(T);

    return (reads + writes) / 1e9;
}

template <typename T, typename I>
constexpr double coo2dense_gbyte_count(I M, I N, I nnz)
{
    size_t reads  = 2 * nnz * sizeof(I) + nnz * sizeof(T);
    size_t writes = (M * N) * sizeof(T);

    return (reads + writes) / 1e9;
}

constexpr double csrsort_gbyte_count(int M, int nnz, bool permute)
{
    return ((2.0 * M + 2.0 + 2.0 * nnz + (permute ? 2.0 * nnz : 0.0)) * sizeof(int)) / 1e9;
}

constexpr double cscsort_gbyte_count(int N, int nnz, bool permute)
{
    return ((2.0 * N + 2.0 + 2.0 * nnz + (permute ? 2.0 * nnz : 0.0)) * sizeof(int)) / 1e9;
}

constexpr double coosort_gbyte_count(int nnz, bool permute)
{
    return ((4.0 * nnz + (permute ? 2.0 * nnz : 0.0)) * sizeof(int)) / 1e9;
}

template <typename T>
constexpr double gebsr2csr_gbyte_count(int Mb, int row_block_dim, int col_block_dim, int nnzb)
{
    // reads
    size_t reads = nnzb * row_block_dim * col_block_dim * sizeof(T) + (Mb + 1 + nnzb) * sizeof(int);

    // writes
    size_t writes = nnzb * row_block_dim * col_block_dim * sizeof(T)
                    + (Mb * row_block_dim + 1 + nnzb * row_block_dim * col_block_dim) * sizeof(int);

    return (reads + writes) / 1e9;
}

template <typename T>
constexpr double gebsr2gebsc_gbyte_count(
    int Mb, int Nb, int nnzb, int row_block_dim, int col_block_dim, hipsparseAction_t action)
{
    return ((Mb + Nb + 2 + 2.0 * nnzb) * sizeof(int)
            + (action == HIPSPARSE_ACTION_NUMERIC
                   ? (2.0 * nnzb * row_block_dim * col_block_dim) * sizeof(T)
                   : 0.0))
           / 1e9;
}

template <typename T>
constexpr double gebsr2gebsr_gbyte_count(int Mb_A,
                                         int Mb_C,
                                         int row_block_dim_A,
                                         int col_block_dim_A,
                                         int row_block_dim_C,
                                         int col_block_dim_C,
                                         int nnzb_A,
                                         int nnzb_C)
{
    // reads
    size_t reads = nnzb_A * row_block_dim_A * col_block_dim_A * sizeof(T)
                   + (Mb_A + 1 + nnzb_A) * sizeof(int);

    // writes
    size_t writes = nnzb_C * row_block_dim_C * col_block_dim_C * sizeof(T)
                    + (Mb_C + 1 + nnzb_C) * sizeof(int);

    return (reads + writes) / 1e9;
}

constexpr double identity_gbyte_count(int N)
{
    return N * sizeof(int) / 1e9;
}

template <typename T>
constexpr double prune_csr2csr_gbyte_count(int M, int nnz_A, int nnz_C)
{
    // reads
    size_t reads = (M + 1 + nnz_A) * sizeof(int) + nnz_A * sizeof(T);

    // writes
    size_t writes = (M + 1 + nnz_C) * sizeof(int) + nnz_C * sizeof(T);

    return (reads + writes) / 1e9;
}

template <typename T>
constexpr double prune_csr2csr_by_percentage_gbyte_count(int M, int nnz_A, int nnz_C)
{
    // reads
    size_t reads = (M + 1 + nnz_A) * sizeof(int) + nnz_A * sizeof(T);

    // writes
    size_t writes = (M + 1 + nnz_C) * sizeof(int) + nnz_C * sizeof(T);

    return (reads + writes) / 1e9;
}

template <typename T>
constexpr double prune_dense2csr_gbyte_count(int M, int N, int nnz)
{
    size_t reads = M * N * sizeof(T);

    size_t writes = (M + 1 + nnz) * sizeof(int) + nnz * sizeof(T);

    return (reads + writes) / 1e9;
}

template <typename T>
constexpr double prune_dense2csr_by_percentage_gbyte_count(int M, int N, int nnz)
{
    size_t reads = M * N * sizeof(T);

    size_t writes = (M + 1 + nnz) * sizeof(int) + nnz * sizeof(T);

    return (reads + writes) / 1e9;
}

/*
 * ===========================================================================
 *    extra SPARSE
 * ===========================================================================
 */
template <typename T>
constexpr double
    csrgeam_gbyte_count(int M, int nnz_A, int nnz_B, int nnz_C, const T* alpha, const T* beta)
{
    double size_A = alpha ? (M + 1.0 + nnz_A) * sizeof(int) + nnz_A * sizeof(T) : 0.0;
    double size_B = beta ? (M + 1.0 + nnz_B) * sizeof(int) + nnz_B * sizeof(T) : 0.0;
    double size_C = (M + 1.0 + nnz_C) * sizeof(int) + nnz_C * sizeof(T);

    return (size_A + size_B + size_C) / 1e9;
}

template <typename T, typename I = int, typename J = int>
constexpr double csrgemm_gbyte_count(J M, J N, J K, I nnz_A, I nnz_B, I nnz_C)
{
    double size_A = (M + 1.0) * sizeof(I) + nnz_A * sizeof(J) + nnz_A * sizeof(T);
    double size_B = (K + 1.0) * sizeof(I) + nnz_B * sizeof(J) + nnz_B * sizeof(T);
    double size_C = (M + 1.0) * sizeof(I) + nnz_C * sizeof(J) + nnz_C * sizeof(T);

    return (size_A + size_B + size_C) / 1e9;
}

#endif // GBYTE_HPP
