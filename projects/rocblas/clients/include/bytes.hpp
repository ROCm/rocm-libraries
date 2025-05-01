/* ************************************************************************
 * Copyright (C) 2020-2023 Advanced Micro Devices, Inc. All rights reserved.
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

/*!\file
 * \brief provides bandwidth measure as byte counts Basic Linear Algebra Subprograms (BLAS) of
 * Level 1, 2, 3. Where possible we are using the values of NOP from the legacy BLAS files
 * [sdcz]blas[23]time.f for byte counts.
 */

/*
 * ===========================================================================
 *    Auxiliary
 * ===========================================================================
 */

/* \brief byte counts of SET/GET_MATRIX/_ASYNC */
template <typename T>
constexpr double set_get_matrix_gbyte_count(int64_t m, int64_t n)
{
    // calls done in pairs for timing so x 2.0
    return (sizeof(T) * m * n * 2.0) / 1e9;
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

/* \brief byte counts of GBMV */
template <typename T>
constexpr double
    gbmv_gbyte_count(rocblas_operation transA, int64_t m, int64_t n, int64_t kl, int64_t ku)
{
    size_t dim_x = transA == rocblas_operation_none ? n : m;

    int64_t k1      = dim_x < kl ? dim_x : kl;
    int64_t k2      = dim_x < ku ? dim_x : ku;
    int64_t d1      = ((k1 * dim_x) - (k1 * (k1 + 1) / 2));
    int64_t d2      = ((k2 * dim_x) - (k2 * (k2 + 1) / 2));
    double  num_els = double(d1 + d2 + dim_x);
    return (sizeof(T) * (num_els)) / 1e9;
}

/* \brief byte counts of GEMV */
template <typename T>
constexpr double gemv_gbyte_count(rocblas_operation transA, int64_t m, int64_t n)
{
    return (sizeof(T) * (m * n + 2 * (transA == rocblas_operation_none ? n : m))) / 1e9;
}

/* \brief byte counts of GER */
template <typename T>
constexpr double ger_gbyte_count(int64_t m, int64_t n)
{
    return (sizeof(T) * (m * n + m + n)) / 1e9;
}

/* \brief byte counts of HEMV */
template <typename T>
constexpr double hemv_gbyte_count(int64_t n)
{
    return (sizeof(T) * (((n * (n + 1.0)) / 2.0) + 3.0 * n)) / 1e9;
}

/* \brief byte counts of HBMV */
template <typename T>
constexpr double hbmv_gbyte_count(int64_t n, int64_t k)
{
    int64_t k1 = k < n ? k : n;
    return (sizeof(T) * (n * k1 - ((k1 * (k1 + 1)) / 2.0) + 3 * n)) / 1e9;
}

/* \brief byte counts of HPMV */
template <typename T>
constexpr double hpmv_gbyte_count(int64_t n)
{
    return (sizeof(T) * ((n * (n + 1.0)) / 2.0) + 3.0 * n) / 1e9;
}

/* \brief byte counts of HPR */
template <typename T>
constexpr double hpr_gbyte_count(int64_t n)
{
    return (sizeof(T) * (tri_count(n) + n)) / 1e9;
}

/* \brief byte counts of HPR2 */
template <typename T>
constexpr double hpr2_gbyte_count(int64_t n)
{
    return (sizeof(T) * (tri_count(n) + 2.0 * n)) / 1e9;
}

/* \brief byte counts of SYMV */
template <typename T>
constexpr double symv_gbyte_count(int64_t n)
{
    return (sizeof(T) * (tri_count(n) + n)) / 1e9;
}

/* \brief byte counts of SPMV */
template <typename T>
constexpr double spmv_gbyte_count(int64_t n)
{
    return (sizeof(T) * (tri_count(n) + n)) / 1e9;
}

/* \brief byte counts of SBMV */
template <typename T>
constexpr double sbmv_gbyte_count(int64_t n, int64_t k)
{
    int64_t k1 = k < n ? k : n - 1;
    return (sizeof(T) * (tri_count(n) - tri_count(n - (k1 + 1)) + n)) / 1e9;
}

/* \brief byte counts of HER */
template <typename T>
constexpr double her_gbyte_count(int64_t n)
{
    return (sizeof(T) * (tri_count(n) + n)) / 1e9;
}

/* \brief byte counts of HER2 */
template <typename T>
constexpr double her2_gbyte_count(int64_t n)
{
    return (sizeof(T) * (tri_count(n) + 2 * n)) / 1e9;
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
constexpr double tbmv_gbyte_count(int64_t m, int64_t k)
{
    int64_t k1 = k < m ? k : m;
    return (sizeof(T) * (m * k1 - ((k1 * (k1 + 1)) / 2.0) + 3 * m)) / 1e9;
}

/* \brief byte counts of TPMV */
template <typename T>
constexpr double tpmv_gbyte_count(int64_t m)
{
    return (sizeof(T) * tri_count(m)) / 1e9;
}

/* \brief byte counts of TRMV */
template <typename T>
constexpr double trmv_gbyte_count(int64_t m)
{
    return (sizeof(T) * ((m * (m + 1.0)) / 2 + 2 * m)) / 1e9;
}

/* \brief byte counts of TPSV */
template <typename T>
constexpr double tpsv_gbyte_count(int64_t n)
{
    return (sizeof(T) * (tri_count(n) + n)) / 1e9;
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
    int64_t k1 = k < n ? k : n - 1;
    return (sizeof(T) * (tri_count(n) + n * k)) / 1e9;
}

/* \brief byte counts of HERK */
template <typename T>
constexpr double herk_gbyte_count(int64_t n, int64_t k)
{
    return syrk_gbyte_count<T>(n, k);
}
