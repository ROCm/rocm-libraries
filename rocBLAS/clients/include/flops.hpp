/* ************************************************************************
 * Copyright (C) 2018-2023 Advanced Micro Devices, Inc. All rights reserved.
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
 * \brief provides Floating point counts of Basic Linear Algebra Subprograms (BLAS) of Level 1, 2,
 * 3. Where possible we are using the values of NOP from the legacy BLAS files [sdcz]blas[23]time.f
 * for flop count.
 */

inline size_t sym_tri_count(int64_t n)
{
    return size_t(n) * (1 + n) / 2;
}

/*
 * ===========================================================================
 *    level 1 BLAS
 * ===========================================================================
 */

// asum
template <typename T>
constexpr double asum_gflop_count(int64_t n)
{
    return (2.0 * n) / 1e9;
}
template <>
constexpr double asum_gflop_count<rocblas_float_complex>(int64_t n)
{
    return (4.0 * n) / 1e9;
}
template <>
constexpr double asum_gflop_count<rocblas_double_complex>(int64_t n)
{
    return (4.0 * n) / 1e9;
}

// axpy
template <typename T>
constexpr double axpy_gflop_count(int64_t n)
{
    return (2.0 * n) / 1e9;
}
template <>
constexpr double axpy_gflop_count<rocblas_float_complex>(int64_t n)
{
    return (8.0 * n) / 1e9; // 6 for complex-complex multiply, 2 for c-c add
}
template <>
constexpr double axpy_gflop_count<rocblas_double_complex>(int64_t n)
{
    return (8.0 * n) / 1e9;
}

// dot
template <bool CONJ, typename T>
constexpr double dot_gflop_count(int64_t n)
{
    return (2.0 * n) / 1e9;
}
template <>
constexpr double dot_gflop_count<false, rocblas_float_complex>(int64_t n)
{
    return (8.0 * n) / 1e9; // 6 for each c-c multiply, 2 for each c-c add
}
template <>
constexpr double dot_gflop_count<false, rocblas_double_complex>(int64_t n)
{
    return (8.0 * n) / 1e9;
}
template <>
constexpr double dot_gflop_count<true, rocblas_float_complex>(int64_t n)
{
    return (9.0 * n) / 1e9; // regular dot (8n) + 1n for complex conjugate
}
template <>
constexpr double dot_gflop_count<true, rocblas_double_complex>(int64_t n)
{
    return (9.0 * n) / 1e9;
}

// nrm2
template <typename T>
constexpr double nrm2_gflop_count(int64_t n)
{
    return (2.0 * n) / 1e9;
}

template <>
constexpr double nrm2_gflop_count<rocblas_float_complex>(int64_t n)
{
    return (6.0 * n + 2.0 * n) / 1e9;
}

template <>
constexpr double nrm2_gflop_count<rocblas_double_complex>(int64_t n)
{
    return nrm2_gflop_count<rocblas_float_complex>(n);
}

// scal
template <typename T, typename U>
constexpr double scal_gflop_count(int64_t n)
{
    return (1.0 * n) / 1e9;
}
template <>
constexpr double scal_gflop_count<rocblas_float_complex, rocblas_float_complex>(int64_t n)
{
    return (6.0 * n) / 1e9; // 6 for c-c multiply
}
template <>
constexpr double scal_gflop_count<rocblas_double_complex, rocblas_double_complex>(int64_t n)
{
    return (6.0 * n) / 1e9;
}
template <>
constexpr double scal_gflop_count<rocblas_float_complex, float>(int64_t n)
{
    return (2.0 * n) / 1e9; // 2 for real-complex multiply
}
template <>
constexpr double scal_gflop_count<rocblas_double_complex, double>(int64_t n)
{
    return (2.0 * n) / 1e9;
}

// rot
template <typename Tx, typename Ty, typename Tc, typename Ts>
constexpr double rot_gflop_count(int64_t n)
{
    return (6.0 * n) / 1e9; //4 real multiplication, 1 addition , 1 subtraction
}
template <>
constexpr double
    rot_gflop_count<rocblas_float_complex, rocblas_float_complex, float, rocblas_float_complex>(
        int64_t n)
{
    return (20.0 * n)
           / 1e9; // (6*2 n for c-c multiply)+(2*2 n for real-complex multiply) + 2n for c-c add + 2n for c-c sub
}
template <>
constexpr double
    rot_gflop_count<rocblas_float_complex, rocblas_float_complex, float, float>(int64_t n)
{
    return (12.0 * n) / 1e9; // (2*4 n for real-complex multiply) + 2n for c-c add + 2n for c-c sub
}
template <>
constexpr double
    rot_gflop_count<rocblas_double_complex, rocblas_double_complex, double, rocblas_double_complex>(
        int64_t n)
{
    return (20.0 * n) / 1e9;
}
template <>
constexpr double
    rot_gflop_count<rocblas_double_complex, rocblas_double_complex, double, double>(int64_t n)
{
    return (12.0 * n) / 1e9;
}

// rotm
template <typename Tx>
constexpr double rotm_gflop_count(int64_t n, Tx flag)
{
    //No floating point operations when flag is set to -2.0
    if(flag != -2.0)
    {
        if(flag < 0)
            return (6.0 * n) / 1e9; // 4 real multiplication, 2 addition
        else
            return (4.0 * n) / 1e9; // 2 real multiplication, 2 addition
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

/* \brief floating point counts of tpmv */
template <typename T>
constexpr double tpmv_gflop_count(int64_t m)
{
    return (double(m) * m) / 1e9;
}

template <>
constexpr double tpmv_gflop_count<rocblas_float_complex>(int64_t m)
{
    return (4.0 * double(m) * m) / 1e9;
}

template <>
constexpr double tpmv_gflop_count<rocblas_double_complex>(int64_t m)
{
    return tpmv_gflop_count<rocblas_float_complex>(m);
}

/* \brief floating point counts of trmv */
template <typename T>
constexpr double trmv_gflop_count(int64_t m)
{
    return (double(m) * m) / 1e9;
}

template <>
constexpr double trmv_gflop_count<rocblas_float_complex>(int64_t m)
{
    return (4.0 * double(m) * m) / 1e9;
}

template <>
constexpr double trmv_gflop_count<rocblas_double_complex>(int64_t m)
{
    return trmv_gflop_count<rocblas_float_complex>(m);
}

/* \brief floating point counts of GBMV */
template <typename T>
constexpr double
    gbmv_gflop_count(rocblas_operation transA, int64_t m, int64_t n, int64_t kl, int64_t ku)
{
    int64_t dim_x = transA == rocblas_operation_none ? n : m;
    int64_t k1    = dim_x < kl ? dim_x : kl;
    int64_t k2    = dim_x < ku ? dim_x : ku;

    // kl and ku ops, plus main diagonal ops
    double d1 = ((2 * k1 * dim_x) - (k1 * (k1 + 1))) + dim_x;
    double d2 = ((2 * k2 * dim_x) - (k2 * (k2 + 1))) + 2 * dim_x;

    // add y operations
    return (d1 + d2 + 2 * dim_x) / 1e9;
}

template <>
constexpr double gbmv_gflop_count<rocblas_float_complex>(
    rocblas_operation transA, int64_t m, int64_t n, int64_t kl, int64_t ku)
{
    int64_t dim_x = transA == rocblas_operation_none ? n : m;
    int64_t k1    = dim_x < kl ? dim_x : kl;
    int64_t k2    = dim_x < ku ? dim_x : ku;

    double d1 = 4 * ((2 * k1 * dim_x) - (k1 * (k1 + 1))) + 6 * dim_x;
    double d2 = 4 * ((2 * k2 * dim_x) - (k2 * (k2 + 1))) + 8 * dim_x;

    return (d1 + d2 + 8 * dim_x) / 1e9;
}

template <>
constexpr double gbmv_gflop_count<rocblas_double_complex>(
    rocblas_operation transA, int64_t m, int64_t n, int64_t kl, int64_t ku)
{
    int64_t dim_x = transA == rocblas_operation_none ? n : m;
    int64_t k1    = dim_x < kl ? dim_x : kl;
    int64_t k2    = dim_x < ku ? dim_x : ku;

    double d1 = 4 * ((2 * k1 * dim_x) - (k1 * (k1 + 1))) + 6 * dim_x;
    double d2 = 4 * ((2 * k2 * dim_x) - (k2 * (k2 + 1))) + 8 * dim_x;

    return (d1 + d2 + 8 * dim_x) / 1e9;
}

/* \brief floating point counts of GEMV */
template <typename T>
constexpr double gemv_gflop_count(rocblas_operation transA, int64_t m, int64_t n)
{
    return (2.0 * double(m) * n + 2.0 * (transA == rocblas_operation_none ? double(m) : double(n)))
           / 1e9;
}
template <>
constexpr double
    gemv_gflop_count<rocblas_float_complex>(rocblas_operation transA, int64_t m, int64_t n)
{
    return (8.0 * double(m) * n + 6.0 * (transA == rocblas_operation_none ? double(m) : double(n)))
           / 1e9;
}

template <>
constexpr double
    gemv_gflop_count<rocblas_double_complex>(rocblas_operation transA, int64_t m, int64_t n)
{
    return gemv_gflop_count<rocblas_float_complex>(transA, m, n);
}

/* \brief floating point counts of HBMV */
template <typename T>
constexpr double hbmv_gflop_count(int64_t n, int64_t k)
{
    double k1 = k < n ? k : n;
    return (8.0 * ((2 * k1 + 1) * n - k1 * (k1 + 1)) + 8 * n) / 1e9;
}

/* \brief floating point counts of HEMV */
template <typename T>
constexpr double hemv_gflop_count(int64_t n)
{
    return (8.0 * double(n) * n + 8.0 * double(n)) / 1e9;
}

/* \brief floating point counts of HER */
template <typename T>
constexpr double her_gflop_count(int64_t n)
{
    return (4.0 * double(n) * n) / 1e9;
}

/* \brief floating point counts of HER2 */
template <typename T>
constexpr double her2_gflop_count(int64_t n)
{
    return (8.0 * (double(n) + 1) * n) / 1e9;
}

/* \brief floating point counts of HPMV */
template <typename T>
constexpr double hpmv_gflop_count(int64_t n)
{
    return (8.0 * double(n) * n + 8.0 * double(n)) / 1e9;
}

/* \brief floating point counts of HPR */
template <typename T>
constexpr double hpr_gflop_count(int64_t n)
{
    return (4.0 * double(n) * n) / 1e9;
}

/* \brief floating point counts of HPR2 */
template <typename T>
constexpr double hpr2_gflop_count(int64_t n)
{
    return (8.0 * (double(n) + 1) * n) / 1e9;
}

/* \brief floating point counts or TBSV */
template <typename T>
constexpr double tbsv_gflop_count(int64_t n, int64_t k)
{
    double k1 = std::min(k, n);
    return ((2.0 * n * k1 - k1 * (k1 + 1)) + n) / 1e9;
}

template <>
constexpr double tbsv_gflop_count<rocblas_float_complex>(int64_t n, int64_t k)
{
    double k1 = std::min(k, n);
    return (4.0 * (2.0 * n * k1 - k1 * (k1 + 1)) + 4.0 * n) / 1e9;
}

template <>
constexpr double tbsv_gflop_count<rocblas_double_complex>(int64_t n, int64_t k)
{
    return tbsv_gflop_count<rocblas_float_complex>(n, k);
}

/* \brief floating point counts of TRSV */
template <typename T>
constexpr double trsv_gflop_count(int64_t m)
{
    return (double(m) * m) / 1e9;
}

template <>
constexpr double trsv_gflop_count<rocblas_float_complex>(int64_t n)
{
    return (4.0 * double(n) * n) / 1e9;
}

template <>
constexpr double trsv_gflop_count<rocblas_double_complex>(int64_t n)
{
    return trsv_gflop_count<rocblas_float_complex>(n);
}

/* \brief floating point counts of TBMV */
template <typename T>
constexpr double tbmv_gflop_count(int64_t m, int64_t k)
{
    double k1 = k < m ? k : m;
    return ((2 * m * k1 - k1 * (k1 + 1)) + m) / 1e9;
}

template <>
constexpr double tbmv_gflop_count<rocblas_float_complex>(int64_t m, int64_t k)
{
    double k1 = k < m ? k : m;
    return (4 * (2 * m * k1 - k1 * (k1 + 1)) + 4 * m) / 1e9;
}

template <>
constexpr double tbmv_gflop_count<rocblas_double_complex>(int64_t m, int64_t k)
{
    double k1 = k < m ? k : m;
    return (4 * (2 * m * k1 - k1 * (k1 + 1)) + 4 * m) / 1e9;
}

/* \brief floating point counts of TPSV */
template <typename T>
constexpr double tpsv_gflop_count(int64_t n)
{
    return (double(n) * n) / 1e9;
}

template <>
constexpr double tpsv_gflop_count<rocblas_float_complex>(int64_t n)
{
    return (4.0 * double(n) * n) / 1e9;
}

template <>
constexpr double tpsv_gflop_count<rocblas_double_complex>(int64_t n)
{
    return tpsv_gflop_count<rocblas_float_complex>(n);
}

/* \brief floating point counts of SY(HE)MV */
template <typename T>
constexpr double symv_gflop_count(int64_t n)
{
    return (2.0 * double(n) * n + 2.0 * n) / 1e9;
}

template <>
constexpr double symv_gflop_count<rocblas_float_complex>(int64_t n)
{
    return 4.0 * symv_gflop_count<rocblas_float>(n);
}

template <>
constexpr double symv_gflop_count<rocblas_double_complex>(int64_t n)
{
    return symv_gflop_count<rocblas_float_complex>(n);
}

/* \brief floating point counts of SPMV */
template <typename T>
constexpr double spmv_gflop_count(int64_t n)
{
    return (2.0 * double(n) * n + 2.0 * n) / 1e9;
}

/* \brief floating point counts of SBMV */
template <typename T>
constexpr double sbmv_gflop_count(int64_t n, int64_t k)
{
    double k1 = k < n ? k : n;
    return (2.0 * ((2.0 * k1 + 1) * n - k1 * (k1 + 1)) + 2.0 * n) / 1e9;
}

/* \brief floating point counts of SPR */
template <typename T>
constexpr double spr_gflop_count(int64_t n)
{
    return (double(n) * (n + 1.0) + n) / 1e9;
}

/* \brief floating point counts of SPR2 */
template <typename T>
constexpr double spr2_gflop_count(int64_t n)
{
    return (2.0 * (double(n) + 1.0) * n + 2.0 * n) / 1e9;
}

/* \brief floating point counts of GER */
template <typename T>
constexpr double ger_gflop_count(int64_t m, int64_t n)
{
    return (6 * (double(m) * n + std::min(m, n)) + 2 * double(m) * n) / 1e9;
}

template <>
constexpr double ger_gflop_count<float>(int64_t m, int64_t n)
{
    return ((2.0 * double(m) * n) + std::min(m, n)) / 1e9;
}

template <>
constexpr double ger_gflop_count<double>(int64_t m, int64_t n)
{
    return ger_gflop_count<float>(m, n);
}

/* \brief floating point counts of SYR */
template <typename T>
constexpr double syr_gflop_count(int64_t n)
{
    return (n * (double(n) + 1.0) + n) / 1e9;
}

/* \brief floating point counts of SYR2 */
template <typename T>
constexpr double syr2_gflop_count(int64_t n)
{
    return (2.0 * (double(n) + 1.0) * n + 2.0 * double(n)) / 1e9;
}

/*
 * ===========================================================================
 *    level 3 BLAS
 * ===========================================================================
 */

/* \brief floating point counts of GEMM */
template <typename T>
constexpr double gemm_gflop_count(int64_t m, int64_t n, int64_t k)
{
    if(k == 0)
        k = 1;
    return (2.0 * m * n * k) / 1e9;
}

template <>
constexpr double gemm_gflop_count<rocblas_float_complex>(int64_t m, int64_t n, int64_t k)
{
    if(k == 0)
        k = 1;
    return (8.0 * m * n * k) / 1e9;
}

template <>
constexpr double gemm_gflop_count<rocblas_double_complex>(int64_t m, int64_t n, int64_t k)
{
    return gemm_gflop_count<rocblas_float_complex>(m, n, k);
}

/* \brief floating point counts of GEAM */
template <typename T>
constexpr double geam_gflop_count(int64_t m, int64_t n)
{
    return (3.0 * m * n) / 1e9;
}

template <>
constexpr double geam_gflop_count<rocblas_float_complex>(int64_t m, int64_t n)
{
    return (14.0 * m * n) / 1e9;
}

template <>
constexpr double geam_gflop_count<rocblas_double_complex>(int64_t m, int64_t n)
{
    return (14.0 * m * n) / 1e9;
}

/* \brief floating point counts of DGMM */
template <typename T>
constexpr double dgmm_gflop_count(int64_t m, int64_t n)
{
    return (m * n) / 1e9;
}

template <>
constexpr double dgmm_gflop_count<rocblas_float_complex>(int64_t m, int64_t n)
{
    return (6 * m * n) / 1e9;
}

template <>
constexpr double dgmm_gflop_count<rocblas_double_complex>(int64_t m, int64_t n)
{
    return (6 * m * n) / 1e9;
}

/* \brief floating point counts of HEMM */
template <typename T>
constexpr double hemm_gflop_count(rocblas_side side, int64_t m, int64_t n)
{
    int k = side == rocblas_side_left ? m : n;
    return (8.0 * double(m) * k * n) / 1e9;
}

/* \brief floating point counts of HERK */
template <typename T>
constexpr double herk_gflop_count(int64_t n, int64_t k)
{
    return (4.0 * double(n) * n * k) / 1e9;
}

/* \brief floating point counts of HER2K */
template <typename T>
constexpr double her2k_gflop_count(int64_t n, int64_t k)
{
    return (8.0 * double(n) * n * k) / 1e9;
}

/* \brief floating point counts of HERKX */
template <typename T>
constexpr double herkx_gflop_count(int64_t n, int64_t k)
{
    return (4.0 * double(n) * n * k) / 1e9;
}

/* \brief floating point counts of SYMM */
template <typename T>
constexpr double symm_gflop_count(rocblas_side side, int64_t m, int64_t n)
{
    int k = side == rocblas_side_left ? m : n;
    return (2.0 * double(m) * k * n) / 1e9;
}

template <>
constexpr double symm_gflop_count<rocblas_float_complex>(rocblas_side side, int64_t m, int64_t n)
{
    return 4.0 * symm_gflop_count<float>(side, m, n);
}

template <>
constexpr double symm_gflop_count<rocblas_double_complex>(rocblas_side side, int64_t m, int64_t n)
{
    return symm_gflop_count<rocblas_float_complex>(side, m, n);
}

/* \brief floating point counts of SYRK */
template <typename T>
constexpr double syrk_gflop_count(int64_t n, int64_t k)
{
    return (double(n) * n * k) / 1e9;
}

template <>
constexpr double syrk_gflop_count<rocblas_float_complex>(int64_t n, int64_t k)
{
    return 4.0 * syrk_gflop_count<float>(n, k);
}

template <>
constexpr double syrk_gflop_count<rocblas_double_complex>(int64_t n, int64_t k)
{
    return syrk_gflop_count<rocblas_float_complex>(n, k);
}

/* \brief floating point counts of SYR2K */
template <typename T>
constexpr double syr2k_gflop_count(int64_t n, int64_t k)
{
    return (2.0 * double(n) * n * k) / 1e9;
}

template <>
constexpr double syr2k_gflop_count<rocblas_float_complex>(int64_t n, int64_t k)
{
    return 4.0 * syr2k_gflop_count<float>(n, k);
}

template <>
constexpr double syr2k_gflop_count<rocblas_double_complex>(int64_t n, int64_t k)
{
    return syr2k_gflop_count<rocblas_float_complex>(n, k);
}

/* \brief floating point counts of SYRKX */
template <typename T>
constexpr double syrkx_gflop_count(int64_t n, int64_t k)
{
    return (2 * k * sym_tri_count(n)) / 1e9;
}

template <>
constexpr double syrkx_gflop_count<rocblas_float_complex>(int64_t n, int64_t k)
{
    return 4.0 * syrkx_gflop_count<float>(n, k);
}

template <>
constexpr double syrkx_gflop_count<rocblas_double_complex>(int64_t n, int64_t k)
{
    return syrkx_gflop_count<rocblas_float_complex>(n, k);
}

/* \brief floating point counts of TRMM */
template <typename T>
constexpr double trmm_gflop_count(int64_t m, int64_t n, rocblas_side side)
{
    int k = side == rocblas_side_left ? m : n;
    return (double(m) * k * n) / 1e9;
}

template <>
constexpr double trmm_gflop_count<rocblas_float_complex>(int64_t m, int64_t n, rocblas_side side)
{
    return 4.0 * trmm_gflop_count<float>(m, n, side);
}

template <>
constexpr double trmm_gflop_count<rocblas_double_complex>(int64_t m, int64_t n, rocblas_side side)
{
    return trmm_gflop_count<rocblas_float_complex>(m, n, side);
}

/* \brief floating point counts of TRSM */
template <typename T>
constexpr double trsm_gflop_count(int64_t m, int64_t n, int64_t k)
{
    return (1.0 * m * n * k) / 1e9;
}

template <>
constexpr double trsm_gflop_count<rocblas_float_complex>(int64_t m, int64_t n, int64_t k)
{
    return 4.0 * trsm_gflop_count<float>(m, n, k);
}

template <>
constexpr double trsm_gflop_count<rocblas_double_complex>(int64_t m, int64_t n, int64_t k)
{
    return trsm_gflop_count<rocblas_float_complex>(m, n, k);
}

/* \brief floating point counts of TRTRI */
template <typename T>
constexpr double trtri_gflop_count(int64_t n)
{
    return (1.0 * n * n * n) / 3e9;
}

template <>
constexpr double trtri_gflop_count<rocblas_float_complex>(int64_t n)
{
    return (8.0 * n * n * n) / 3e9;
}

template <>
constexpr double trtri_gflop_count<rocblas_double_complex>(int64_t n)
{
    return (8.0 * n * n * n) / 3e9;
}

/* \brief floating point counts of GEAM_MIN_PLUS and GEAM_PLUS_MIN in geam_ex*/
template <typename T>
constexpr double geam_min_plus_gflop_count(int64_t m, int64_t n, int64_t k)
{
    return (2.0 * m * n * k) / 1e9;
}
