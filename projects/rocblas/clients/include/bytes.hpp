/* ************************************************************************
 * Copyright (C) 2020-2026 Advanced Micro Devices, Inc. All rights reserved.
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
 * ************************************************************************/

#pragma once

#include "rocblas.h"

#include <algorithm>

/*!\file
 * \brief provides bandwidth measure as byte counts Basic Linear Algebra Subprograms (BLAS) of
 * Level 1, 2, 3. Where possible we are using the values of NOP from the legacy BLAS files
 * [sdcz]blas[23]time.f for byte counts.
 *
 * A count is the data a call moves between device memory and the GPU, as in BLAS++ (blas::Gbyte):
 * each input is read once, each output is written once, and an operand that is both (such as y
 * when beta != 0, or A in ger) is read and written. Matrices count the stored elements only: the
 * triangle of symmetric, Hermitian and triangular matrices (also packed) and the band of banded ones.
 */

/*
 * ===========================================================================
 *    Auxiliary
 * ===========================================================================
 */

/* \brief byte counts of SET/GET_MATRIX/_ASYNC */
template <typename T>
constexpr double set_get_matrix_gbyte_count(int64_t rows, int64_t cols)
{
    // calls done in pairs for timing so x 2.0
    return (sizeof(T) * rows * cols * 2.0) / 1e9;
}

/* \brief byte counts of SET/GET_VECTOR/_ASYNC */
template <typename T>
constexpr double set_get_vector_gbyte_count(int64_t n)
{
    // calls done in pairs for timing so x 2.0
    return (sizeof(T) * n * 2.0) / 1e9;
}

/*
 * ===========================================================================
 *    level 1 BLAS
 * ===========================================================================
 */

/* \brief byte counts of ASUM */
template <typename T>
constexpr double asum_gbyte_count(int64_t n)
{
    return (sizeof(T) * n) / 1e9;
}

/* \brief byte counts of AXPY */
template <typename T>
constexpr double axpy_gbyte_count(int64_t n)
{
    return (sizeof(T) * 3.0 * n) / 1e9;
}

/* \brief byte counts of COPY */
template <typename T>
constexpr double copy_gbyte_count(int64_t n)
{
    return (sizeof(T) * 2.0 * n) / 1e9;
}

/* \brief byte counts of DOT */
template <typename T>
constexpr double dot_gbyte_count(int64_t n)
{
    return (sizeof(T) * 2.0 * n) / 1e9;
}

/* \brief byte counts of IAMAX AND IAMIN */
template <typename T>
constexpr double iamax_iamin_gbyte_count(int64_t n)
{
    return (sizeof(T) * n) / 1e9;
}

/* \brief byte counts of NRM2 */
template <typename T>
constexpr double nrm2_gbyte_count(int64_t n)
{
    return (sizeof(T) * n) / 1e9;
}

/* \brief byte counts of SCAL */
template <typename T>
constexpr double scal_gbyte_count(int64_t n)
{
    return (sizeof(T) * 2.0 * n) / 1e9;
}

/* \brief byte counts of SWAP */
template <typename T>
constexpr double swap_gbyte_count(int64_t n)
{
    return (sizeof(T) * 4.0 * n) / 1e9;
}

/* \brief byte counts of ROT */
template <typename T>
constexpr double rot_gbyte_count(int64_t n)
{
    return (sizeof(T) * 4.0 * n) / 1e9; //2 loads and 2 stores
}

/* \brief byte counts of ROTM */
template <typename T>
constexpr double rotm_gbyte_count(int64_t n, T flag)
{
    //No load and store operations when flag is set to -2.0
    if(flag != -2.0)
    {
        return (sizeof(T) * 4.0 * n) / 1e9; //2 loads and 2 stores
    }
    else
    {
        return 0;
    }
}

/*
 * ===========================================================================
 *    level 2 BLAS
 * ===========================================================================
 */

inline size_t tri_count(int64_t n)
{
    return size_t(n) * (1 + n) / 2;
}

inline size_t rocblas_packed_matrix_size(int64_t n)
{
    return (size_t(n) * (1 + n)) / 2;
}

// Elements in the band of an m x n matrix with kl subdiagonals and ku superdiagonals
inline double band_count(int64_t m, int64_t n, int64_t kl, int64_t ku)
{
    double count = 0;
    for(int64_t d = -std::min(kl, m - 1); d <= std::min(ku, n - 1); d++)
        count += d < 0 ? std::min(m + d, n) : std::min(m, n - d);
    return count;
}

/* \brief byte counts of GBMV */
template <typename T>
constexpr double
    gbmv_gbyte_count(rocblas_operation transA, int64_t m, int64_t n, int64_t kl, int64_t ku)
{
    int64_t dim_x = transA == rocblas_operation_none ? n : m;
    int64_t dim_y = transA == rocblas_operation_none ? m : n;
    // read band of A, x, y; write y
    return (sizeof(T) * (band_count(m, n, kl, ku) + dim_x + 2.0 * dim_y)) / 1e9;
}

/* \brief byte counts of GEMV */
template <typename T>
constexpr double gemv_gbyte_count(rocblas_operation transA, int64_t m, int64_t n)
{
    int64_t dim_x = transA == rocblas_operation_none ? n : m;
    int64_t dim_y = transA == rocblas_operation_none ? m : n;
    // read A, x, y; write y
    return (sizeof(T) * (double(m) * n + dim_x + 2.0 * dim_y)) / 1e9;
}

/* \brief byte counts of GER */
template <typename T>
constexpr double ger_gbyte_count(int64_t m, int64_t n)
{
    // read and write of A + read of x and y
    return (sizeof(T) * (2 * m * n + m + n)) / 1e9;
}

/* \brief byte counts of HEMV */
template <typename T>
constexpr double hemv_gbyte_count(int64_t n)
{
    // read triangle of A, x, y; write y
    return (sizeof(T) * (((n * (n + 1.0)) / 2.0) + 3.0 * n)) / 1e9;
}

/* \brief byte counts of HBMV */
template <typename T>
constexpr double hbmv_gbyte_count(int64_t n, int64_t k)
{
    // read band of A (main diagonal and k superdiagonals), x, y; write y
    return (sizeof(T) * (band_count(n, n, 0, k) + 3.0 * n)) / 1e9;
}

/* \brief byte counts of HPMV */
template <typename T>
constexpr double hpmv_gbyte_count(int64_t n)
{
    // read triangle of A, x, y; write y
    return (sizeof(T) * (((n * (n + 1.0)) / 2.0) + 3.0 * n)) / 1e9;
}

/* \brief byte counts of HPR */
template <typename T>
constexpr double hpr_gbyte_count(int64_t n)
{
    // read and write of A + read of x
    return (sizeof(T) * (tri_count(n) * 2 + n)) / 1e9;
}

/* \brief byte counts of HPR2 */
template <typename T>
constexpr double hpr2_gbyte_count(int64_t n)
{
    // read and write of A + read of x and y
    return (sizeof(T) * (tri_count(n) * 2 + 2.0 * n)) / 1e9;
}

/* \brief byte counts of SYMV */
template <typename T>
constexpr double symv_gbyte_count(int64_t n)
{
    // read triangle of A, x, y; write y
    return (sizeof(T) * (tri_count(n) + 3.0 * n)) / 1e9;
}

/* \brief byte counts of SPMV */
template <typename T>
constexpr double spmv_gbyte_count(int64_t n)
{
    // read triangle of A, x, y; write y
    return (sizeof(T) * (tri_count(n) + 3.0 * n)) / 1e9;
}

/* \brief byte counts of SBMV */
template <typename T>
constexpr double sbmv_gbyte_count(int64_t n, int64_t k)
{
    // read band of A (main diagonal and k superdiagonals), x, y; write y
    return (sizeof(T) * (band_count(n, n, 0, k) + 3.0 * n)) / 1e9;
}

/* \brief byte counts of HER */
template <typename T>
constexpr double her_gbyte_count(int64_t n)
{
    // read and write of A + read of x
    return (sizeof(T) * (tri_count(n) * 2 + n)) / 1e9;
}

/* \brief byte counts of HER2 */
template <typename T>
constexpr double her2_gbyte_count(int64_t n)
{
    // read and write of A + read of x and y
    return (sizeof(T) * (tri_count(n) * 2 + 2 * n)) / 1e9;
}

/* \brief byte counts of SPR */
template <typename T>
constexpr double spr_gbyte_count(int64_t n)
{
    // read and write of A + read of x
    return (sizeof(T) * (tri_count(n) * 2 + n)) / 1e9;
}

/* \brief byte counts of SPR2 */
template <typename T>
constexpr double spr2_gbyte_count(int64_t n)
{
    // read and write of A + read of x and y
    return (sizeof(T) * (tri_count(n) * 2 + n * 2)) / 1e9;
}

/* \brief byte counts of SYR */
template <typename T>
constexpr double syr_gbyte_count(int64_t n)
{
    // read and write of A + read of x
    return (sizeof(T) * (tri_count(n) * 2 + n)) / 1e9;
}

/* \brief byte counts of SYR2 */
template <typename T>
constexpr double syr2_gbyte_count(int64_t n)
{
    // read and write of A + read of x and y
    return (sizeof(T) * (tri_count(n) * 2 + n * 2)) / 1e9;
}

/* \brief byte counts of TBMV */
template <typename T>
constexpr double tbmv_gbyte_count(int64_t n, int64_t k)
{
    // read band of A (main diagonal and k off-diagonals), x; write x
    return (sizeof(T) * (band_count(n, n, 0, k) + 2.0 * n)) / 1e9;
}

/* \brief byte counts of TPMV */
template <typename T>
constexpr double tpmv_gbyte_count(int64_t n)
{
    // read triangle of A, x; write x
    return (sizeof(T) * (tri_count(n) + 2.0 * n)) / 1e9;
}

/* \brief byte counts of TRMV */
template <typename T>
constexpr double trmv_gbyte_count(int64_t n)
{
    // read triangle of A, x; write x
    return (sizeof(T) * ((n * (n + 1.0)) / 2 + 2 * n)) / 1e9;
}

/* \brief byte counts of TPSV */
template <typename T>
constexpr double tpsv_gbyte_count(int64_t n)
{
    // read triangle of A, x; write x
    return (sizeof(T) * (tri_count(n) + 2.0 * n)) / 1e9;
}

/*
 * ===========================================================================
 *    level 3 BLAS
 * ===========================================================================
 */

/* \brief byte counts of SYRK */
template <typename T>
constexpr double syrk_gbyte_count(int64_t n, int64_t k)
{
    // read A, triangle of C; write triangle of C
    return (sizeof(T) * (double(n) * k + 2.0 * tri_count(n))) / 1e9;
}

/* \brief byte counts of HERK */
template <typename T>
constexpr double herk_gbyte_count(int64_t n, int64_t k)
{
    return syrk_gbyte_count<T>(n, k);
}
