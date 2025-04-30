/* ************************************************************************
 * Copyright (C) 2018-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include <cblas.h>

#include "rocblas.h"
#include <type_traits>

// fwd declaration for lapack_utilities
template <typename T, typename U>
void ref_scal(int64_t n, T alpha, U x, int64_t incx);

#include "lapack_utilities.hpp"

/*
 * ===========================================================================
 *    level 1 BLAS
 * ===========================================================================
 */

// iamax
template <typename T>
void ref_iamax(int64_t n, const T* x, int64_t incx, int64_t* result);

template <>
inline void ref_iamax(int64_t n, const float* x, int64_t incx, int64_t* result)
{
    *result = (int64_t)cblas_isamax(n, x, incx);
}

template <>
inline void ref_iamax(int64_t n, const double* x, int64_t incx, int64_t* result)
{
    *result = (int64_t)cblas_idamax(n, x, incx);
}

template <>
inline void ref_iamax(int64_t n, const rocblas_float_complex* x, int64_t incx, int64_t* result)
{
    *result = (int64_t)cblas_icamax(n, x, incx);
}

template <>
inline void ref_iamax(int64_t n, const rocblas_double_complex* x, int64_t incx, int64_t* result)
{
    *result = (int64_t)cblas_izamax(n, x, incx);
}

// asum
template <typename T>
void ref_asum(int64_t n, const T* x, int64_t incx, real_t<T>* result);

template <>
inline void ref_asum(int64_t n, const double* x, int64_t incx, double* result)
{
    *result = cblas_dasum(n, x, incx);
}

template <>
inline void ref_asum(int64_t n, const rocblas_float_complex* x, int64_t incx, float* result)
{
    *result = cblas_scasum(n, x, incx);
}

template <>
inline void ref_asum(int64_t n, const rocblas_double_complex* x, int64_t incx, double* result)
{
    *result = cblas_dzasum(n, x, incx);
}

// axpy
template <typename T>
void ref_axpy(int64_t n, T alpha, T* x, int64_t incx, T* y, int64_t incy);

template <>
inline void ref_axpy(int64_t n, float alpha, float* x, int64_t incx, float* y, int64_t incy)
{
    cblas_saxpy(n, alpha, x, incx, y, incy);
}

template <>
inline void ref_axpy(int64_t n, double alpha, double* x, int64_t incx, double* y, int64_t incy)
{
    cblas_daxpy(n, alpha, x, incx, y, incy);
}

template <>
inline void ref_axpy(int64_t                n,
                     rocblas_float_complex  alpha,
                     rocblas_float_complex* x,
                     int64_t                incx,
                     rocblas_float_complex* y,
                     int64_t                incy)
{
    cblas_caxpy(n, &alpha, x, incx, y, incy);
}

template <>
inline void ref_axpy(int64_t                 n,
                     rocblas_double_complex  alpha,
                     rocblas_double_complex* x,
                     int64_t                 incx,
                     rocblas_double_complex* y,
                     int64_t                 incy)
{
    cblas_zaxpy(n, &alpha, x, incx, y, incy);
}

// copy
template <typename T>
void ref_copy(int64_t n, T* x, int64_t incx, T* y, int64_t incy);

template <>
inline void ref_copy(int64_t n, float* x, int64_t incx, float* y, int64_t incy)
{
    cblas_scopy(n, x, incx, y, incy);
}

template <>
inline void ref_copy(int64_t n, double* x, int64_t incx, double* y, int64_t incy)
{
    cblas_dcopy(n, x, incx, y, incy);
}

template <>
inline void ref_copy(
    int64_t n, rocblas_float_complex* x, int64_t incx, rocblas_float_complex* y, int64_t incy)
{
    cblas_ccopy(n, x, incx, y, incy);
}

template <>
inline void ref_copy(
    int64_t n, rocblas_double_complex* x, int64_t incx, rocblas_double_complex* y, int64_t incy)
{
    cblas_zcopy(n, x, incx, y, incy);
}

// dot
template <typename T, typename Tr>
void ref_dot(int64_t n, const T* x, int64_t incx, const T* y, int64_t incy, Tr* result);

template <>
inline void
    ref_dot(int64_t n, const float* x, int64_t incx, const float* y, int64_t incy, float* result)
{
    *result = cblas_sdot(n, x, incx, y, incy);
}

template <>
inline void
    ref_dot(int64_t n, const double* x, int64_t incx, const double* y, int64_t incy, double* result)
{
    *result = cblas_ddot(n, x, incx, y, incy);
}

template <>
inline void ref_dot(int64_t                      n,
                    const rocblas_float_complex* x,
                    int64_t                      incx,
                    const rocblas_float_complex* y,
                    int64_t                      incy,
                    rocblas_float_complex*       result)
{
    cblas_cdotu_sub(n, x, incx, y, incy, result);
}

template <>
inline void ref_dot(int64_t                       n,
                    const rocblas_double_complex* x,
                    int64_t                       incx,
                    const rocblas_double_complex* y,
                    int64_t                       incy,
                    rocblas_double_complex*       result)
{
    cblas_zdotu_sub(n, x, incx, y, incy, result);
}

// dotc
template <typename T, typename Tr>
void ref_dotc(int64_t n, const T* x, int64_t incx, const T* y, int64_t incy, Tr* result);

template <>
inline void ref_dotc(int64_t                      n,
                     const rocblas_float_complex* x,
                     int64_t                      incx,
                     const rocblas_float_complex* y,
                     int64_t                      incy,
                     rocblas_float_complex*       result)
{
    cblas_cdotc_sub(n, x, incx, y, incy, result);
}

template <>
inline void ref_dotc(int64_t                       n,
                     const rocblas_double_complex* x,
                     int64_t                       incx,
                     const rocblas_double_complex* y,
                     int64_t                       incy,
                     rocblas_double_complex*       result)
{
    cblas_zdotc_sub(n, x, incx, y, incy, result);
}

// nrm2
template <typename T>
void ref_nrm2(int64_t n, const T* x, int64_t incx, real_t<T>* result);

template <>
inline void ref_nrm2(int64_t n, const double* x, int64_t incx, double* result)
{
    *result = cblas_dnrm2(n, x, incx);
}

template <>
inline void ref_nrm2(int64_t n, const rocblas_float_complex* x, int64_t incx, float* result)
{
    *result = cblas_scnrm2(n, x, incx);
}

template <>
inline void ref_nrm2(int64_t n, const rocblas_double_complex* x, int64_t incx, double* result)
{
    *result = cblas_dznrm2(n, x, incx);
}

// scal ILP64
template <typename T, typename U>
void ref_scal(int64_t n, T alpha, U x, int64_t incx);

// swap
template <typename T>
inline void ref_swap(int64_t n, T* x, int64_t incx, T* y, int64_t incy);

template <>
inline void ref_swap(int64_t n, float* x, int64_t incx, float* y, int64_t incy)
{
    cblas_sswap(n, x, incx, y, incy);
}

template <>
inline void ref_swap(int64_t n, double* x, int64_t incx, double* y, int64_t incy)
{
    cblas_dswap(n, x, incx, y, incy);
}

template <>
inline void ref_swap(
    int64_t n, rocblas_float_complex* x, int64_t incx, rocblas_float_complex* y, int64_t incy)
{
    cblas_cswap(n, x, incx, y, incy);
}

template <>
inline void ref_swap(
    int64_t n, rocblas_double_complex* x, int64_t incx, rocblas_double_complex* y, int64_t incy)
{
    cblas_zswap(n, x, incx, y, incy);
}

// rot
template <typename Tx, typename Ty, typename Tc, typename Ts>
void ref_rot(int64_t n, Tx* x, int64_t incx, Ty* y, int64_t incy, const Tc* c, const Ts* s);

template <>
inline void ref_rot(
    int64_t n, float* x, int64_t incx, float* y, int64_t incy, const float* c, const float* s)
{
    cblas_srot(n, x, incx, y, incy, *c, *s);
}

template <>
inline void ref_rot(
    int64_t n, double* x, int64_t incx, double* y, int64_t incy, const double* c, const double* s)
{
    cblas_drot(n, x, incx, y, incy, *c, *s);
}

template <>
inline void ref_rot(int64_t                      n,
                    rocblas_float_complex*       x,
                    int64_t                      incx,
                    rocblas_float_complex*       y,
                    int64_t                      incy,
                    const float*                 c,
                    const rocblas_float_complex* s)
{
    ref_lapack_xrot(n, x, incx, y, incy, *c, *s);
}

template <>
inline void ref_rot(int64_t                n,
                    rocblas_float_complex* x,
                    int64_t                incx,
                    rocblas_float_complex* y,
                    int64_t                incy,
                    const float*           c,
                    const float*           s)
{
    ref_lapack_xrot(n, x, incx, y, incy, *c, *s);
}

template <>
inline void ref_rot(int64_t                       n,
                    rocblas_double_complex*       x,
                    int64_t                       incx,
                    rocblas_double_complex*       y,
                    int64_t                       incy,
                    const double*                 c,
                    const rocblas_double_complex* s)
{
    ref_lapack_xrot(n, x, incx, y, incy, *c, *s);
}

template <>
inline void ref_rot(int64_t                 n,
                    rocblas_double_complex* x,
                    int64_t                 incx,
                    rocblas_double_complex* y,
                    int64_t                 incy,
                    const double*           c,
                    const double*           s)
{
    ref_lapack_xrot(n, x, incx, y, incy, *c, *s);
}

// for rot_ex
template <>
inline void ref_rot(int64_t                      n,
                    rocblas_float_complex*       x,
                    int64_t                      incx,
                    rocblas_float_complex*       y,
                    int64_t                      incy,
                    const rocblas_float_complex* c,
                    const rocblas_float_complex* s)
{
    const float c_real = std::real(*c);
    ref_lapack_xrot(n, x, incx, y, incy, c_real, *s);
}

template <>
inline void ref_rot(int64_t                       n,
                    rocblas_double_complex*       x,
                    int64_t                       incx,
                    rocblas_double_complex*       y,
                    int64_t                       incy,
                    const rocblas_double_complex* c,
                    const rocblas_double_complex* s)
{
    const double c_real = std::real(*c);
    ref_lapack_xrot(n, x, incx, y, incy, c_real, *s);
}

// rotg
template <typename T, typename U>
inline void ref_rotg(T* a, T* b, U* c, T* s);

template <>
inline void ref_rotg(float* a, float* b, float* c, float* s)
{
    cblas_srotg(a, b, c, s);
}

template <>
inline void ref_rotg(double* a, double* b, double* c, double* s)
{
    cblas_drotg(a, b, c, s);
}

template <>
inline void
    ref_rotg(rocblas_float_complex* a, rocblas_float_complex* b, float* c, rocblas_float_complex* s)
{
    ref_lapack_xrotg(*a, *b, *c, *s);
}

template <>
inline void ref_rotg(rocblas_double_complex* a,
                     rocblas_double_complex* b,
                     double*                 c,
                     rocblas_double_complex* s)
{
    ref_lapack_xrotg(*a, *b, *c, *s);
}

// rotm

template <typename T>
inline void ref_rotm(int64_t n, T* x, int64_t incx, T* y, int64_t incy, const T* p);

template <>
inline void ref_rotm(int64_t n, float* x, int64_t incx, float* y, int64_t incy, const float* p)
{
    cblas_srotm(n, x, incx, y, incy, p);
}

template <>
inline void ref_rotm(int64_t n, double* x, int64_t incx, double* y, int64_t incy, const double* p)
{
    cblas_drotm(n, x, incx, y, incy, p);
}

// rotmg

template <typename T>
inline void ref_rotmg(T* d1, T* d2, T* b1, const T* b2, T* p);

template <>
inline void ref_rotmg(float* d1, float* d2, float* b1, const float* b2, float* p)
{
    cblas_srotmg(d1, d2, b1, *b2, p);
}

template <>
inline void ref_rotmg(double* d1, double* d2, double* b1, const double* b2, double* p)
{
    cblas_drotmg(d1, d2, b1, *b2, p);
}

/*
 * ===========================================================================
 *    level 2 BLAS
 * ===========================================================================
 */

// gbmv
template <typename T>
void ref_gbmv(rocblas_operation transA,
              int64_t           m,
              int64_t           n,
              int64_t           kl,
              int64_t           ku,
              T                 alpha,
              T*                A,
              int64_t           lda,
              T*                x,
              int64_t           incx,
              T                 beta,
              T*                y,
              int64_t           incy);

template <>
inline void ref_gbmv(rocblas_operation transA,
                     int64_t           m,
                     int64_t           n,
                     int64_t           kl,
                     int64_t           ku,
                     float             alpha,
                     float*            A,
                     int64_t           lda,
                     float*            x,
                     int64_t           incx,
                     float             beta,
                     float*            y,
                     int64_t           incy)
{
    cblas_sgbmv(CblasColMajor,
                CBLAS_TRANSPOSE(transA),
                m,
                n,
                kl,
                ku,
                alpha,
                A,
                lda,
                x,
                incx,
                beta,
                y,
                incy);
}

template <>
inline void ref_gbmv(rocblas_operation transA,
                     int64_t           m,
                     int64_t           n,
                     int64_t           kl,
                     int64_t           ku,
                     double            alpha,
                     double*           A,
                     int64_t           lda,
                     double*           x,
                     int64_t           incx,
                     double            beta,
                     double*           y,
                     int64_t           incy)
{
    cblas_dgbmv(CblasColMajor,
                CBLAS_TRANSPOSE(transA),
                m,
                n,
                kl,
                ku,
                alpha,
                A,
                lda,
                x,
                incx,
                beta,
                y,
                incy);
}

template <>
inline void ref_gbmv(rocblas_operation      transA,
                     int64_t                m,
                     int64_t                n,
                     int64_t                kl,
                     int64_t                ku,
                     rocblas_float_complex  alpha,
                     rocblas_float_complex* A,
                     int64_t                lda,
                     rocblas_float_complex* x,
                     int64_t                incx,
                     rocblas_float_complex  beta,
                     rocblas_float_complex* y,
                     int64_t                incy)
{
    cblas_cgbmv(CblasColMajor,
                CBLAS_TRANSPOSE(transA),
                m,
                n,
                kl,
                ku,
                &alpha,
                A,
                lda,
                x,
                incx,
                &beta,
                y,
                incy);
}

template <>
inline void ref_gbmv(rocblas_operation       transA,
                     int64_t                 m,
                     int64_t                 n,
                     int64_t                 kl,
                     int64_t                 ku,
                     rocblas_double_complex  alpha,
                     rocblas_double_complex* A,
                     int64_t                 lda,
                     rocblas_double_complex* x,
                     int64_t                 incx,
                     rocblas_double_complex  beta,
                     rocblas_double_complex* y,
                     int64_t                 incy)
{
    cblas_zgbmv(CblasColMajor,
                CBLAS_TRANSPOSE(transA),
                m,
                n,
                kl,
                ku,
                &alpha,
                A,
                lda,
                x,
                incx,
                &beta,
                y,
                incy);
}

// gemv
template <typename Ti, typename To, typename Ta>
void ref_gemv(rocblas_operation transA,
              int64_t           m,
              int64_t           n,
              Ta                alpha,
              const Ti*         A,
              int64_t           lda,
              const Ti*         x,
              int64_t           incx,
              Ta                beta,
              To*               y,
              int64_t           incy);

// tbmv
template <typename T>
void ref_tbmv(rocblas_fill      uplo,
              rocblas_operation transA,
              rocblas_diagonal  diag,
              int64_t           m,
              int64_t           k,
              T*                A,
              int64_t           lda,
              T*                x,
              int64_t           incx);

template <>
inline void ref_tbmv(rocblas_fill      uplo,
                     rocblas_operation transA,
                     rocblas_diagonal  diag,
                     int64_t           m,
                     int64_t           k,
                     float*            A,
                     int64_t           lda,
                     float*            x,
                     int64_t           incx)
{
    cblas_stbmv(CblasColMajor,
                CBLAS_UPLO(uplo),
                CBLAS_TRANSPOSE(transA),
                CBLAS_DIAG(diag),
                m,
                k,
                A,
                lda,
                x,
                incx);
}

template <>
inline void ref_tbmv(rocblas_fill      uplo,
                     rocblas_operation transA,
                     rocblas_diagonal  diag,
                     int64_t           m,
                     int64_t           k,
                     double*           A,
                     int64_t           lda,
                     double*           x,
                     int64_t           incx)
{
    cblas_dtbmv(CblasColMajor,
                CBLAS_UPLO(uplo),
                CBLAS_TRANSPOSE(transA),
                CBLAS_DIAG(diag),
                m,
                k,
                A,
                lda,
                x,
                incx);
}

template <>
inline void ref_tbmv(rocblas_fill           uplo,
                     rocblas_operation      transA,
                     rocblas_diagonal       diag,
                     int64_t                m,
                     int64_t                k,
                     rocblas_float_complex* A,
                     int64_t                lda,
                     rocblas_float_complex* x,
                     int64_t                incx)
{
    cblas_ctbmv(CblasColMajor,
                CBLAS_UPLO(uplo),
                CBLAS_TRANSPOSE(transA),
                CBLAS_DIAG(diag),
                m,
                k,
                A,
                lda,
                x,
                incx);
}

template <>
inline void ref_tbmv(rocblas_fill            uplo,
                     rocblas_operation       transA,
                     rocblas_diagonal        diag,
                     int64_t                 m,
                     int64_t                 k,
                     rocblas_double_complex* A,
                     int64_t                 lda,
                     rocblas_double_complex* x,
                     int64_t                 incx)
{
    cblas_ztbmv(CblasColMajor,
                CBLAS_UPLO(uplo),
                CBLAS_TRANSPOSE(transA),
                CBLAS_DIAG(diag),
                m,
                k,
                A,
                lda,
                x,
                incx);
}

// tbsv
template <typename T>
void ref_tbsv(rocblas_fill      uplo,
              rocblas_operation transA,
              rocblas_diagonal  diag,
              int64_t           n,
              int64_t           k,
              const T*          A,
              int64_t           lda,
              T*                x,
              int64_t           incx);

template <>
inline void ref_tbsv(rocblas_fill      uplo,
                     rocblas_operation transA,
                     rocblas_diagonal  diag,
                     int64_t           n,
                     int64_t           k,
                     const float*      A,
                     int64_t           lda,
                     float*            x,
                     int64_t           incx)
{
    cblas_stbsv(CblasColMajor,
                CBLAS_UPLO(uplo),
                CBLAS_TRANSPOSE(transA),
                CBLAS_DIAG(diag),
                n,
                k,
                A,
                lda,
                x,
                incx);
}

template <>
inline void ref_tbsv(rocblas_fill      uplo,
                     rocblas_operation transA,
                     rocblas_diagonal  diag,
                     int64_t           n,
                     int64_t           k,
                     const double*     A,
                     int64_t           lda,
                     double*           x,
                     int64_t           incx)
{
    cblas_dtbsv(CblasColMajor,
                CBLAS_UPLO(uplo),
                CBLAS_TRANSPOSE(transA),
                CBLAS_DIAG(diag),
                n,
                k,
                A,
                lda,
                x,
                incx);
}

template <>
inline void ref_tbsv(rocblas_fill                 uplo,
                     rocblas_operation            transA,
                     rocblas_diagonal             diag,
                     int64_t                      n,
                     int64_t                      k,
                     const rocblas_float_complex* A,
                     int64_t                      lda,
                     rocblas_float_complex*       x,
                     int64_t                      incx)
{
    cblas_ctbsv(CblasColMajor,
                CBLAS_UPLO(uplo),
                CBLAS_TRANSPOSE(transA),
                CBLAS_DIAG(diag),
                n,
                k,
                A,
                lda,
                x,
                incx);
}

template <>
inline void ref_tbsv(rocblas_fill                  uplo,
                     rocblas_operation             transA,
                     rocblas_diagonal              diag,
                     int64_t                       n,
                     int64_t                       k,
                     const rocblas_double_complex* A,
                     int64_t                       lda,
                     rocblas_double_complex*       x,
                     int64_t                       incx)
{
    cblas_ztbsv(CblasColMajor,
                CBLAS_UPLO(uplo),
                CBLAS_TRANSPOSE(transA),
                CBLAS_DIAG(diag),
                n,
                k,
                A,
                lda,
                x,
                incx);
}

// tpsv
template <typename T>
void ref_tpsv(rocblas_fill      uplo,
              rocblas_operation transA,
              rocblas_diagonal  diag,
              int64_t           n,
              const T*          AP,
              T*                x,
              int64_t           incx);

template <>
inline void ref_tpsv(rocblas_fill      uplo,
                     rocblas_operation transA,
                     rocblas_diagonal  diag,
                     int64_t           n,
                     const float*      AP,
                     float*            x,
                     int64_t           incx)
{
    cblas_stpsv(
        CblasColMajor, CBLAS_UPLO(uplo), CBLAS_TRANSPOSE(transA), CBLAS_DIAG(diag), n, AP, x, incx);
}

template <>
inline void ref_tpsv(rocblas_fill      uplo,
                     rocblas_operation transA,
                     rocblas_diagonal  diag,
                     int64_t           n,
                     const double*     AP,
                     double*           x,
                     int64_t           incx)
{
    cblas_dtpsv(
        CblasColMajor, CBLAS_UPLO(uplo), CBLAS_TRANSPOSE(transA), CBLAS_DIAG(diag), n, AP, x, incx);
}

template <>
inline void ref_tpsv(rocblas_fill                 uplo,
                     rocblas_operation            transA,
                     rocblas_diagonal             diag,
                     int64_t                      n,
                     const rocblas_float_complex* AP,
                     rocblas_float_complex*       x,
                     int64_t                      incx)
{
    cblas_ctpsv(
        CblasColMajor, CBLAS_UPLO(uplo), CBLAS_TRANSPOSE(transA), CBLAS_DIAG(diag), n, AP, x, incx);
}

template <>
inline void ref_tpsv(rocblas_fill                  uplo,
                     rocblas_operation             transA,
                     rocblas_diagonal              diag,
                     int64_t                       n,
                     const rocblas_double_complex* AP,
                     rocblas_double_complex*       x,
                     int64_t                       incx)
{
    cblas_ztpsv(
        CblasColMajor, CBLAS_UPLO(uplo), CBLAS_TRANSPOSE(transA), CBLAS_DIAG(diag), n, AP, x, incx);
}

// trsv
template <typename T>
void ref_trsv(rocblas_fill      uplo,
              rocblas_operation transA,
              rocblas_diagonal  diag,
              int64_t           m,
              const T*          A,
              int64_t           lda,
              T*                x,
              int64_t           incx);

template <>
inline void ref_trsv(rocblas_fill      uplo,
                     rocblas_operation transA,
                     rocblas_diagonal  diag,
                     int64_t           m,
                     const float*      A,
                     int64_t           lda,
                     float*            x,
                     int64_t           incx)
{
    cblas_strsv(CblasColMajor,
                CBLAS_UPLO(uplo),
                CBLAS_TRANSPOSE(transA),
                CBLAS_DIAG(diag),
                m,
                A,
                lda,
                x,
                incx);
}

template <>
inline void ref_trsv(rocblas_fill      uplo,
                     rocblas_operation transA,
                     rocblas_diagonal  diag,
                     int64_t           m,
                     const double*     A,
                     int64_t           lda,
                     double*           x,
                     int64_t           incx)
{
    cblas_dtrsv(CblasColMajor,
                CBLAS_UPLO(uplo),
                CBLAS_TRANSPOSE(transA),
                CBLAS_DIAG(diag),
                m,
                A,
                lda,
                x,
                incx);
}

template <>
inline void ref_trsv(rocblas_fill                 uplo,
                     rocblas_operation            transA,
                     rocblas_diagonal             diag,
                     int64_t                      m,
                     const rocblas_float_complex* A,
                     int64_t                      lda,
                     rocblas_float_complex*       x,
                     int64_t                      incx)
{
    cblas_ctrsv(CblasColMajor,
                CBLAS_UPLO(uplo),
                CBLAS_TRANSPOSE(transA),
                CBLAS_DIAG(diag),
                m,
                A,
                lda,
                x,
                incx);
}

template <>
inline void ref_trsv(rocblas_fill                  uplo,
                     rocblas_operation             transA,
                     rocblas_diagonal              diag,
                     int64_t                       m,
                     const rocblas_double_complex* A,
                     int64_t                       lda,
                     rocblas_double_complex*       x,
                     int64_t                       incx)
{
    cblas_ztrsv(CblasColMajor,
                CBLAS_UPLO(uplo),
                CBLAS_TRANSPOSE(transA),
                CBLAS_DIAG(diag),
                m,
                A,
                lda,
                x,
                incx);
}

// tpmv
template <typename T>
void ref_tpmv(rocblas_fill      uplo,
              rocblas_operation transA,
              rocblas_diagonal  diag,
              int64_t           m,
              const T*          A,
              T*                x,
              int64_t           incx);

template <>
inline void ref_tpmv(rocblas_fill      uplo,
                     rocblas_operation transA,
                     rocblas_diagonal  diag,
                     int64_t           m,
                     const float*      A,
                     float*            x,
                     int64_t           incx)
{
    cblas_stpmv(
        CblasColMajor, CBLAS_UPLO(uplo), CBLAS_TRANSPOSE(transA), CBLAS_DIAG(diag), m, A, x, incx);
}

template <>
inline void ref_tpmv(rocblas_fill      uplo,
                     rocblas_operation transA,
                     rocblas_diagonal  diag,
                     int64_t           m,
                     const double*     A,
                     double*           x,
                     int64_t           incx)
{
    cblas_dtpmv(
        CblasColMajor, CBLAS_UPLO(uplo), CBLAS_TRANSPOSE(transA), CBLAS_DIAG(diag), m, A, x, incx);
}

template <>
inline void ref_tpmv(rocblas_fill                 uplo,
                     rocblas_operation            transA,
                     rocblas_diagonal             diag,
                     int64_t                      m,
                     const rocblas_float_complex* A,
                     rocblas_float_complex*       x,
                     int64_t                      incx)
{
    cblas_ctpmv(
        CblasColMajor, CBLAS_UPLO(uplo), CBLAS_TRANSPOSE(transA), CBLAS_DIAG(diag), m, A, x, incx);
}

template <>
inline void ref_tpmv(rocblas_fill                  uplo,
                     rocblas_operation             transA,
                     rocblas_diagonal              diag,
                     int64_t                       m,
                     const rocblas_double_complex* A,
                     rocblas_double_complex*       x,
                     int64_t                       incx)
{
    cblas_ztpmv(
        CblasColMajor, CBLAS_UPLO(uplo), CBLAS_TRANSPOSE(transA), CBLAS_DIAG(diag), m, A, x, incx);
}

// trmv
template <typename T>
void ref_trmv(rocblas_fill      uplo,
              rocblas_operation transA,
              rocblas_diagonal  diag,
              int64_t           m,
              const T*          A,
              int64_t           lda,
              T*                x,
              int64_t           incx);

template <>
inline void ref_trmv(rocblas_fill      uplo,
                     rocblas_operation transA,
                     rocblas_diagonal  diag,
                     int64_t           m,
                     const float*      A,
                     int64_t           lda,
                     float*            x,
                     int64_t           incx)
{
    cblas_strmv(CblasColMajor,
                CBLAS_UPLO(uplo),
                CBLAS_TRANSPOSE(transA),
                CBLAS_DIAG(diag),
                m,
                A,
                lda,
                x,
                incx);
}

template <>
inline void ref_trmv(rocblas_fill      uplo,
                     rocblas_operation transA,
                     rocblas_diagonal  diag,
                     int64_t           m,
                     const double*     A,
                     int64_t           lda,
                     double*           x,
                     int64_t           incx)
{
    cblas_dtrmv(CblasColMajor,
                CBLAS_UPLO(uplo),
                CBLAS_TRANSPOSE(transA),
                CBLAS_DIAG(diag),
                m,
                A,
                lda,
                x,
                incx);
}

template <>
inline void ref_trmv(rocblas_fill                 uplo,
                     rocblas_operation            transA,
                     rocblas_diagonal             diag,
                     int64_t                      m,
                     const rocblas_float_complex* A,
                     int64_t                      lda,
                     rocblas_float_complex*       x,
                     int64_t                      incx)
{
    cblas_ctrmv(CblasColMajor,
                CBLAS_UPLO(uplo),
                CBLAS_TRANSPOSE(transA),
                CBLAS_DIAG(diag),
                m,
                A,
                lda,
                x,
                incx);
}

template <>
inline void ref_trmv(rocblas_fill                  uplo,
                     rocblas_operation             transA,
                     rocblas_diagonal              diag,
                     int64_t                       m,
                     const rocblas_double_complex* A,
                     int64_t                       lda,
                     rocblas_double_complex*       x,
                     int64_t                       incx)
{
    cblas_ztrmv(CblasColMajor,
                CBLAS_UPLO(uplo),
                CBLAS_TRANSPOSE(transA),
                CBLAS_DIAG(diag),
                m,
                A,
                lda,
                x,
                incx);
}

// sbmv
template <typename T>
void ref_sbmv(rocblas_fill uplo,
              int64_t      n,
              int64_t      k,
              T            alpha,
              T*           A,
              int64_t      lda,
              T*           x,
              int64_t      incx,
              T            beta,
              T*           y,
              int64_t      incy);

template <>
inline void ref_sbmv(rocblas_fill uplo,
                     int64_t      n,
                     int64_t      k,
                     float        alpha,
                     float*       A,
                     int64_t      lda,
                     float*       x,
                     int64_t      incx,
                     float        beta,
                     float*       y,
                     int64_t      incy)
{
    cblas_ssbmv(CblasColMajor, CBLAS_UPLO(uplo), n, k, alpha, A, lda, x, incx, beta, y, incy);
}

template <>
inline void ref_sbmv(rocblas_fill uplo,
                     int64_t      n,
                     int64_t      k,
                     double       alpha,
                     double*      A,
                     int64_t      lda,
                     double*      x,
                     int64_t      incx,
                     double       beta,
                     double*      y,
                     int64_t      incy)
{
    cblas_dsbmv(CblasColMajor, CBLAS_UPLO(uplo), n, k, alpha, A, lda, x, incx, beta, y, incy);
}

// spmv
template <typename T>
void ref_spmv(
    rocblas_fill uplo, int64_t n, T alpha, T* A, T* x, int64_t incx, T beta, T* y, int64_t incy);

template <>
inline void ref_spmv(rocblas_fill uplo,
                     int64_t      n,
                     float        alpha,
                     float*       A,
                     float*       x,
                     int64_t      incx,
                     float        beta,
                     float*       y,
                     int64_t      incy)
{
    cblas_sspmv(CblasColMajor, CBLAS_UPLO(uplo), n, alpha, A, x, incx, beta, y, incy);
}

template <>
inline void ref_spmv(rocblas_fill uplo,
                     int64_t      n,
                     double       alpha,
                     double*      A,
                     double*      x,
                     int64_t      incx,
                     double       beta,
                     double*      y,
                     int64_t      incy)
{
    cblas_dspmv(CblasColMajor, CBLAS_UPLO(uplo), n, alpha, A, x, incx, beta, y, incy);
}

// symv
template <typename T>
void ref_symv(rocblas_fill uplo,
              int64_t      n,
              T            alpha,
              T*           A,
              int64_t      lda,
              T*           x,
              int64_t      incx,
              T            beta,
              T*           y,
              int64_t      incy);

template <>
inline void ref_symv(rocblas_fill uplo,
                     int64_t      n,
                     float        alpha,
                     float*       A,
                     int64_t      lda,
                     float*       x,
                     int64_t      incx,
                     float        beta,
                     float*       y,
                     int64_t      incy)
{
    cblas_ssymv(CblasColMajor, CBLAS_UPLO(uplo), n, alpha, A, lda, x, incx, beta, y, incy);
}

template <>
inline void ref_symv(rocblas_fill uplo,
                     int64_t      n,
                     double       alpha,
                     double*      A,
                     int64_t      lda,
                     double*      x,
                     int64_t      incx,
                     double       beta,
                     double*      y,
                     int64_t      incy)
{
    cblas_dsymv(CblasColMajor, CBLAS_UPLO(uplo), n, alpha, A, lda, x, incx, beta, y, incy);
}

template <>
inline void ref_symv(rocblas_fill           uplo,
                     int64_t                n,
                     rocblas_float_complex  alpha,
                     rocblas_float_complex* A,
                     int64_t                lda,
                     rocblas_float_complex* x,
                     int64_t                incx,
                     rocblas_float_complex  beta,
                     rocblas_float_complex* y,
                     int64_t                incy)
{
    ref_lapack_xsymv(uplo, n, alpha, A, lda, x, incx, beta, y, incy);
}

template <>
inline void ref_symv(rocblas_fill            uplo,
                     int64_t                 n,
                     rocblas_double_complex  alpha,
                     rocblas_double_complex* A,
                     int64_t                 lda,
                     rocblas_double_complex* x,
                     int64_t                 incx,
                     rocblas_double_complex  beta,
                     rocblas_double_complex* y,
                     int64_t                 incy)
{
    ref_lapack_xsymv(uplo, n, alpha, A, lda, x, incx, beta, y, incy);
}

template <typename T>
void ref_spr(rocblas_fill uplo, int64_t n, T alpha, T* x, int64_t incx, T* A);

template <>
inline void ref_spr(rocblas_fill uplo, int64_t n, float alpha, float* x, int64_t incx, float* A)
{
    cblas_sspr(CblasColMajor, CBLAS_UPLO(uplo), n, alpha, x, incx, A);
}

template <>
inline void ref_spr(rocblas_fill uplo, int64_t n, double alpha, double* x, int64_t incx, double* A)
{
    cblas_dspr(CblasColMajor, CBLAS_UPLO(uplo), n, alpha, x, incx, A);
}

template <>
inline void ref_spr(rocblas_fill           uplo,
                    int64_t                n,
                    rocblas_float_complex  alpha,
                    rocblas_float_complex* x,
                    int64_t                incx,
                    rocblas_float_complex* A)
{
    ref_lapack_xspr(uplo, n, alpha, x, incx, A);
}

template <>
inline void ref_spr(rocblas_fill            uplo,
                    int64_t                 n,
                    rocblas_double_complex  alpha,
                    rocblas_double_complex* x,
                    int64_t                 incx,
                    rocblas_double_complex* A)
{
    ref_lapack_xspr(uplo, n, alpha, x, incx, A);
}

// spr2
template <typename T>
void ref_spr2(rocblas_fill uplo, int64_t n, T alpha, T* x, int64_t incx, T* y, int64_t incy, T* A);

template <>
inline void ref_spr2(rocblas_fill uplo,
                     int64_t      n,
                     float        alpha,
                     float*       x,
                     int64_t      incx,
                     float*       y,
                     int64_t      incy,
                     float*       A)
{
    cblas_sspr2(CblasColMajor, CBLAS_UPLO(uplo), n, alpha, x, incx, y, incy, A);
}

template <>
inline void ref_spr2(rocblas_fill uplo,
                     int64_t      n,
                     double       alpha,
                     double*      x,
                     int64_t      incx,
                     double*      y,
                     int64_t      incy,
                     double*      A)
{
    cblas_dspr2(CblasColMajor, CBLAS_UPLO(uplo), n, alpha, x, incx, y, incy, A);
}

// ger maps to ger, geru, gerc
template <typename T, bool CONJ>
void ref_ger(
    int64_t m, int64_t n, T alpha, T* x, int64_t incx, T* y, int64_t incy, T* A, int64_t lda);

template <>
inline void ref_ger<float, false>(int64_t m,
                                  int64_t n,
                                  float   alpha,
                                  float*  x,
                                  int64_t incx,
                                  float*  y,
                                  int64_t incy,
                                  float*  A,
                                  int64_t lda)
{
    cblas_sger(CblasColMajor, m, n, alpha, x, incx, y, incy, A, lda);
}

template <>
inline void ref_ger<double, false>(int64_t m,
                                   int64_t n,
                                   double  alpha,
                                   double* x,
                                   int64_t incx,
                                   double* y,
                                   int64_t incy,
                                   double* A,
                                   int64_t lda)
{
    cblas_dger(CblasColMajor, m, n, alpha, x, incx, y, incy, A, lda);
}

template <>
inline void ref_ger<rocblas_float_complex, false>(int64_t                m,
                                                  int64_t                n,
                                                  rocblas_float_complex  alpha,
                                                  rocblas_float_complex* x,
                                                  int64_t                incx,
                                                  rocblas_float_complex* y,
                                                  int64_t                incy,
                                                  rocblas_float_complex* A,
                                                  int64_t                lda)
{
    cblas_cgeru(CblasColMajor, m, n, &alpha, x, incx, y, incy, A, lda);
}

template <>
inline void ref_ger<rocblas_double_complex, false>(int64_t                 m,
                                                   int64_t                 n,
                                                   rocblas_double_complex  alpha,
                                                   rocblas_double_complex* x,
                                                   int64_t                 incx,
                                                   rocblas_double_complex* y,
                                                   int64_t                 incy,
                                                   rocblas_double_complex* A,
                                                   int64_t                 lda)
{
    cblas_zgeru(CblasColMajor, m, n, &alpha, x, incx, y, incy, A, lda);
}

template <>
inline void ref_ger<rocblas_float_complex, true>(int64_t                m,
                                                 int64_t                n,
                                                 rocblas_float_complex  alpha,
                                                 rocblas_float_complex* x,
                                                 int64_t                incx,
                                                 rocblas_float_complex* y,
                                                 int64_t                incy,
                                                 rocblas_float_complex* A,
                                                 int64_t                lda)
{
    cblas_cgerc(CblasColMajor, m, n, &alpha, x, incx, y, incy, A, lda);
}

template <>
inline void ref_ger<rocblas_double_complex, true>(int64_t                 m,
                                                  int64_t                 n,
                                                  rocblas_double_complex  alpha,
                                                  rocblas_double_complex* x,
                                                  int64_t                 incx,
                                                  rocblas_double_complex* y,
                                                  int64_t                 incy,
                                                  rocblas_double_complex* A,
                                                  int64_t                 lda)
{
    cblas_zgerc(CblasColMajor, m, n, &alpha, x, incx, y, incy, A, lda);
}

// syr
template <typename T>
inline void ref_syr(rocblas_fill uplo, int64_t n, T alpha, T* xa, int64_t incx, T* A, int64_t lda);

template <>
inline void ref_syr(
    rocblas_fill uplo, int64_t n, float alpha, float* x, int64_t incx, float* A, int64_t lda)
{
    cblas_ssyr(CblasColMajor, CBLAS_UPLO(uplo), n, alpha, x, incx, A, lda);
}

template <>
inline void ref_syr(
    rocblas_fill uplo, int64_t n, double alpha, double* x, int64_t incx, double* A, int64_t lda)
{
    cblas_dsyr(CblasColMajor, CBLAS_UPLO(uplo), n, alpha, x, incx, A, lda);
}

template <>
inline void ref_syr(rocblas_fill           uplo,
                    int64_t                n,
                    rocblas_float_complex  alpha,
                    rocblas_float_complex* xa,
                    int64_t                incx,
                    rocblas_float_complex* A,
                    int64_t                lda)
{
    ref_lapack_xsyr(uplo, n, alpha, xa, incx, A, lda);
}

template <>
inline void ref_syr(rocblas_fill            uplo,
                    int64_t                 n,
                    rocblas_double_complex  alpha,
                    rocblas_double_complex* xa,
                    int64_t                 incx,
                    rocblas_double_complex* A,
                    int64_t                 lda)
{
    ref_lapack_xsyr(uplo, n, alpha, xa, incx, A, lda);
}

// syr2
template <typename T>
inline void ref_syr2(rocblas_fill uplo,
                     int64_t      n,
                     T            alpha,
                     T*           x,
                     int64_t      incx,
                     T*           y,
                     int64_t      incy,
                     T*           A,
                     int64_t      lda);

template <>
inline void ref_syr2(rocblas_fill uplo,
                     int64_t      n,
                     float        alpha,
                     float*       x,
                     int64_t      incx,
                     float*       y,
                     int64_t      incy,
                     float*       A,
                     int64_t      lda)
{
    cblas_ssyr2(CblasColMajor, CBLAS_UPLO(uplo), n, alpha, x, incx, y, incy, A, lda);
}

template <>
inline void ref_syr2(rocblas_fill uplo,
                     int64_t      n,
                     double       alpha,
                     double*      x,
                     int64_t      incx,
                     double*      y,
                     int64_t      incy,
                     double*      A,
                     int64_t      lda)
{
    cblas_dsyr2(CblasColMajor, CBLAS_UPLO(uplo), n, alpha, x, incx, y, incy, A, lda);
}

template <>
inline void ref_syr2(rocblas_fill           uplo,
                     int64_t                n,
                     rocblas_float_complex  alpha,
                     rocblas_float_complex* x,
                     int64_t                incx,
                     rocblas_float_complex* y,
                     int64_t                incy,
                     rocblas_float_complex* A,
                     int64_t                lda)
{
    ref_lapack_xsyr2(uplo, n, alpha, x, incx, y, incy, A, lda);
}

template <>
inline void ref_syr2(rocblas_fill            uplo,
                     int64_t                 n,
                     rocblas_double_complex  alpha,
                     rocblas_double_complex* x,
                     int64_t                 incx,
                     rocblas_double_complex* y,
                     int64_t                 incy,
                     rocblas_double_complex* A,
                     int64_t                 lda)
{
    ref_lapack_xsyr2(uplo, n, alpha, x, incx, y, incy, A, lda);
}

// hbmv
template <typename T>
void ref_hbmv(rocblas_fill uplo,
              int64_t      n,
              int64_t      k,
              T            alpha,
              T*           A,
              int64_t      lda,
              T*           x,
              int64_t      incx,
              T            beta,
              T*           y,
              int64_t      incy);

template <>
inline void ref_hbmv(rocblas_fill           uplo,
                     int64_t                n,
                     int64_t                k,
                     rocblas_float_complex  alpha,
                     rocblas_float_complex* A,
                     int64_t                lda,
                     rocblas_float_complex* x,
                     int64_t                incx,
                     rocblas_float_complex  beta,
                     rocblas_float_complex* y,
                     int64_t                incy)
{
    cblas_chbmv(CblasColMajor, CBLAS_UPLO(uplo), n, k, &alpha, A, lda, x, incx, &beta, y, incy);
}

template <>
inline void ref_hbmv(rocblas_fill            uplo,
                     int64_t                 n,
                     int64_t                 k,
                     rocblas_double_complex  alpha,
                     rocblas_double_complex* A,
                     int64_t                 lda,
                     rocblas_double_complex* x,
                     int64_t                 incx,
                     rocblas_double_complex  beta,
                     rocblas_double_complex* y,
                     int64_t                 incy)
{
    cblas_zhbmv(CblasColMajor, CBLAS_UPLO(uplo), n, k, &alpha, A, lda, x, incx, &beta, y, incy);
}

// hemv
template <typename T>
void ref_hemv(rocblas_fill uplo,
              int64_t      n,
              T            alpha,
              T*           A,
              int64_t      lda,
              T*           x,
              int64_t      incx,
              T            beta,
              T*           y,
              int64_t      incy);

template <>
inline void ref_hemv(rocblas_fill           uplo,
                     int64_t                n,
                     rocblas_float_complex  alpha,
                     rocblas_float_complex* A,
                     int64_t                lda,
                     rocblas_float_complex* x,
                     int64_t                incx,
                     rocblas_float_complex  beta,
                     rocblas_float_complex* y,
                     int64_t                incy)
{
    cblas_chemv(CblasColMajor, CBLAS_UPLO(uplo), n, &alpha, A, lda, x, incx, &beta, y, incy);
}

template <>
inline void ref_hemv(rocblas_fill            uplo,
                     int64_t                 n,
                     rocblas_double_complex  alpha,
                     rocblas_double_complex* A,
                     int64_t                 lda,
                     rocblas_double_complex* x,
                     int64_t                 incx,
                     rocblas_double_complex  beta,
                     rocblas_double_complex* y,
                     int64_t                 incy)
{
    cblas_zhemv(CblasColMajor, CBLAS_UPLO(uplo), n, &alpha, A, lda, x, incx, &beta, y, incy);
}

// her
template <typename T>
void ref_her(rocblas_fill uplo, int64_t n, real_t<T> alpha, T* x, int64_t incx, T* A, int64_t lda);

template <>
inline void ref_her(rocblas_fill           uplo,
                    int64_t                n,
                    float                  alpha,
                    rocblas_float_complex* x,
                    int64_t                incx,
                    rocblas_float_complex* A,
                    int64_t                lda)
{
    cblas_cher(CblasColMajor, CBLAS_UPLO(uplo), n, alpha, x, incx, A, lda);
}

template <>
inline void ref_her(rocblas_fill            uplo,
                    int64_t                 n,
                    double                  alpha,
                    rocblas_double_complex* x,
                    int64_t                 incx,
                    rocblas_double_complex* A,
                    int64_t                 lda)
{
    cblas_zher(CblasColMajor, CBLAS_UPLO(uplo), n, alpha, x, incx, A, lda);
}

// her2
template <typename T>
void ref_her2(rocblas_fill uplo,
              int64_t      n,
              T            alpha,
              T*           x,
              int64_t      incx,
              T*           y,
              int64_t      incy,
              T*           A,
              int64_t      lda);

template <>
inline void ref_her2(rocblas_fill           uplo,
                     int64_t                n,
                     rocblas_float_complex  alpha,
                     rocblas_float_complex* x,
                     int64_t                incx,
                     rocblas_float_complex* y,
                     int64_t                incy,
                     rocblas_float_complex* A,
                     int64_t                lda)
{
    cblas_cher2(CblasColMajor, CBLAS_UPLO(uplo), n, &alpha, x, incx, y, incy, A, lda);
}

template <>
inline void ref_her2(rocblas_fill            uplo,
                     int64_t                 n,
                     rocblas_double_complex  alpha,
                     rocblas_double_complex* x,
                     int64_t                 incx,
                     rocblas_double_complex* y,
                     int64_t                 incy,
                     rocblas_double_complex* A,
                     int64_t                 lda)
{
    cblas_zher2(CblasColMajor, CBLAS_UPLO(uplo), n, &alpha, x, incx, y, incy, A, lda);
}

// hpmv
template <typename T>
void ref_hpmv(
    rocblas_fill uplo, int64_t n, T alpha, T* A, T* x, int64_t incx, T beta, T* y, int64_t incy);

template <>
inline void ref_hpmv(rocblas_fill           uplo,
                     int64_t                n,
                     rocblas_float_complex  alpha,
                     rocblas_float_complex* A,
                     rocblas_float_complex* x,
                     int64_t                incx,
                     rocblas_float_complex  beta,
                     rocblas_float_complex* y,
                     int64_t                incy)
{
    cblas_chpmv(CblasColMajor, CBLAS_UPLO(uplo), n, &alpha, A, x, incx, &beta, y, incy);
}

template <>
inline void ref_hpmv(rocblas_fill            uplo,
                     int64_t                 n,
                     rocblas_double_complex  alpha,
                     rocblas_double_complex* A,
                     rocblas_double_complex* x,
                     int64_t                 incx,
                     rocblas_double_complex  beta,
                     rocblas_double_complex* y,
                     int64_t                 incy)
{
    cblas_zhpmv(CblasColMajor, CBLAS_UPLO(uplo), n, &alpha, A, x, incx, &beta, y, incy);
}

// hpr
template <typename T>
void ref_hpr(rocblas_fill uplo, int64_t n, real_t<T> alpha, T* x, int64_t incx, T* A);

template <>
inline void ref_hpr(rocblas_fill           uplo,
                    int64_t                n,
                    float                  alpha,
                    rocblas_float_complex* x,
                    int64_t                incx,
                    rocblas_float_complex* A)
{
    cblas_chpr(CblasColMajor, CBLAS_UPLO(uplo), n, alpha, x, incx, A);
}

template <>
inline void ref_hpr(rocblas_fill            uplo,
                    int64_t                 n,
                    double                  alpha,
                    rocblas_double_complex* x,
                    int64_t                 incx,
                    rocblas_double_complex* A)
{
    cblas_zhpr(CblasColMajor, CBLAS_UPLO(uplo), n, alpha, x, incx, A);
}

// hpr2
template <typename T>
void ref_hpr2(rocblas_fill uplo, int64_t n, T alpha, T* x, int64_t incx, T* y, int64_t incy, T* A);

template <>
inline void ref_hpr2(rocblas_fill           uplo,
                     int64_t                n,
                     rocblas_float_complex  alpha,
                     rocblas_float_complex* x,
                     int64_t                incx,
                     rocblas_float_complex* y,
                     int64_t                incy,
                     rocblas_float_complex* A)
{
    cblas_chpr2(CblasColMajor, CBLAS_UPLO(uplo), n, &alpha, x, incx, y, incy, A);
}

template <>
inline void ref_hpr2(rocblas_fill            uplo,
                     int64_t                 n,
                     rocblas_double_complex  alpha,
                     rocblas_double_complex* x,
                     int64_t                 incx,
                     rocblas_double_complex* y,
                     int64_t                 incy,
                     rocblas_double_complex* A)
{
    cblas_zhpr2(CblasColMajor, CBLAS_UPLO(uplo), n, &alpha, x, incx, y, incy, A);
}

/*
 * ===========================================================================
 *    level 3 BLAS
 * ===========================================================================
 */

// dgmm
template <typename T>
void ref_dgmm(rocblas_side side,
              int64_t      m,
              int64_t      n,
              T*           A,
              int64_t      lda,
              T*           x,
              int64_t      incx,
              T*           C,
              int64_t      ldc);

// geam
template <typename T>
void ref_geam(rocblas_operation transa,
              rocblas_operation transb,
              int64_t           m,
              int64_t           n,
              T*                alpha,
              T*                A,
              int64_t           lda,
              T*                beta,
              T*                B,
              int64_t           ldb,
              T*                C,
              int64_t           ldc);

// gemm

template <typename TiA,
          typename TiB,
          typename To,
          typename Tc,
          typename std::enable_if<std::is_same<To, Tc>{}, int>::type = 0>
inline void f8_to_cblas_sgemm(rocblas_operation transA,
                              rocblas_operation transB,
                              int64_t           m,
                              int64_t           n,
                              int64_t           k,
                              Tc                alpha,
                              const TiA*        A,
                              int64_t           lda,
                              const TiB*        B,
                              int64_t           ldb,
                              Tc                beta,
                              To*               C,
                              int64_t           ldc)
{
    // cblas does not support rocblas_float8, so convert to higher precision float
    // This will give more precise result which is acceptable for testing

    size_t sizeA = (transA == rocblas_operation_none ? k : m) * size_t(lda);
    size_t sizeB = (transB == rocblas_operation_none ? n : k) * size_t(ldb);
    size_t sizeC = n * size_t(ldc);

    host_vector<Tc> A_float(sizeA), B_float(sizeB), C_float(sizeC);

    for(size_t i = 0; i < sizeA; i++)
        A_float[i] = static_cast<float>(A[i]);
    for(size_t i = 0; i < sizeB; i++)
        B_float[i] = static_cast<float>(B[i]);

    // just directly cast, since transA, transB are integers in the enum
    // printf("transA: rocblas =%d, cblas=%d\n", transA, static_cast<CBLAS_TRANSPOSE>(transA) );
    cblas_sgemm(CblasColMajor,
                static_cast<CBLAS_TRANSPOSE>(transA),
                static_cast<CBLAS_TRANSPOSE>(transB),
                m,
                n,
                k,
                alpha,
                A_float,
                lda,
                B_float,
                ldb,
                beta,
                C,
                ldc);
}

// gemm
template <typename TiA,
          typename TiB,
          typename To,
          typename Tc,
          typename std::enable_if<!std::is_same<To, Tc>{}, int>::type = 0>
inline void f8_to_cblas_sgemm(rocblas_operation transA,
                              rocblas_operation transB,
                              int64_t           m,
                              int64_t           n,
                              int64_t           k,
                              Tc                alpha,
                              const TiA*        A,
                              int64_t           lda,
                              const TiB*        B,
                              int64_t           ldb,
                              Tc                beta,
                              To*               C,
                              int64_t           ldc)
{
    // cblas does not support rocblas_float8, so convert to higher precision float
    // This will give more precise result which is acceptable for testing

    size_t sizeA = (transA == rocblas_operation_none ? k : m) * size_t(lda);
    size_t sizeB = (transB == rocblas_operation_none ? n : k) * size_t(ldb);
    size_t sizeC = n * size_t(ldc);

    host_vector<Tc> A_float(sizeA), B_float(sizeB), C_float(sizeC);

    for(size_t i = 0; i < sizeA; i++)
        A_float[i] = static_cast<float>(A[i]);
    for(size_t i = 0; i < sizeB; i++)
        B_float[i] = static_cast<float>(B[i]);
    for(size_t i = 0; i < sizeC; i++)
        C_float[i] = static_cast<float>(C[i]);

    // just directly cast, since transA, transB are integers in the enum
    // printf("transA: rocblas =%d, cblas=%d\n", transA, static_cast<CBLAS_TRANSPOSE>(transA) );
    cblas_sgemm(CblasColMajor,
                static_cast<CBLAS_TRANSPOSE>(transA),
                static_cast<CBLAS_TRANSPOSE>(transB),
                m,
                n,
                k,
                alpha,
                A_float,
                lda,
                B_float,
                ldb,
                beta,
                C_float,
                ldc);

    for(size_t i = 0; i < sizeC; i++)
        C[i] = To(C_float[i]);
}

template <typename Ti, typename To, typename Tc>
void ref_gemm(rocblas_operation                    transA,
              rocblas_operation                    transB,
              int64_t                              m,
              int64_t                              n,
              int64_t                              k,
              Tc                                   alpha,
              const Ti*                            A,
              int64_t                              lda,
              const Ti*                            B,
              int64_t                              ldb,
              Tc                                   beta,
              To*                                  C,
              int64_t                              ldc,
              rocblas_bfloat16::rocblas_truncate_t round
              = rocblas_bfloat16::rocblas_truncate_t::rocblas_round_near_even);

//GEMMT
template <typename T>
void ref_gemmt(rocblas_fill      uplo,
               rocblas_operation transA,
               rocblas_operation transB,
               int64_t           N,
               int64_t           K,
               T                 alpha,
               T*                A,
               int64_t           lda,
               T*                B,
               int64_t           ldb,
               T                 beta,
               T*                C,
               int64_t           ldc);

// symm
template <typename T>
void ref_symm(rocblas_side side,
              rocblas_fill uplo,
              int64_t      m,
              int64_t      n,
              T            alpha,
              const T*     A,
              int64_t      lda,
              const T*     B,
              int64_t      ldb,
              T            beta,
              T*           C,
              int64_t      ldc);

// syrk
template <typename T>
void ref_syrk(rocblas_fill      uplo,
              rocblas_operation transA,
              int64_t           n,
              int64_t           k,
              T                 alpha,
              const T*          A,
              int64_t           lda,
              T                 beta,
              T*                C,
              int64_t           ldc);

// syr2k
template <typename T>
void ref_syr2k(rocblas_fill      uplo,
               rocblas_operation transA,
               int64_t           n,
               int64_t           k,
               T                 alpha,
               const T*          A,
               int64_t           lda,
               const T*          B,
               int64_t           ldb,
               T                 beta,
               T*                C,
               int64_t           ldc);

// hemm
template <typename T>
void ref_hemm(rocblas_side side,
              rocblas_fill uplo,
              int64_t      m,
              int64_t      n,
              const T*     alpha,
              const T*     A,
              int64_t      lda,
              const T*     B,
              int64_t      ldb,
              const T*     beta,
              T*           C,
              int64_t      ldc);

// herk
template <typename T, typename U>
void ref_herk(rocblas_fill      uplo,
              rocblas_operation transA,
              int64_t           n,
              int64_t           k,
              U                 alpha,
              const T*          A,
              int64_t           lda,
              U                 beta,
              T*                C,
              int64_t           ldc);

// her2k
template <typename T>
void ref_her2k(rocblas_fill      uplo,
               rocblas_operation transA,
               int64_t           n,
               int64_t           k,
               const T*          alpha,
               const T*          A,
               int64_t           lda,
               const T*          B,
               int64_t           ldb,
               const real_t<T>*  beta,
               T*                C,
               int64_t           ldc);

// cblas_geam_min_plus doesn't exist. implementation in cpp
template <typename T>
void ref_geam_min_plus(rocblas_operation transA,
                       rocblas_operation transB,
                       int64_t           m,
                       int64_t           n,
                       int64_t           k,
                       const T           alpha,
                       const T*          A,
                       int64_t           lda,
                       const T*          B,
                       int64_t           ldb,
                       const T           beta,
                       const T*          C,
                       int64_t           ldc,
                       T*                D,
                       int64_t           ldd);

template <typename T>
void ref_geam_plus_min(rocblas_operation transA,
                       rocblas_operation transB,
                       int64_t           m,
                       int64_t           n,
                       int64_t           k,
                       const T           alpha,
                       const T*          A,
                       int64_t           lda,
                       const T*          B,
                       int64_t           ldb,
                       const T           beta,
                       const T*          C,
                       int64_t           ldc,
                       T*                D,
                       int64_t           ldd);

// cblas_herkx doesn't exist. implementation in cpp
template <typename T, typename U = real_t<T>>
void ref_herkx(rocblas_fill      uplo,
               rocblas_operation transA,
               int64_t           n,
               int64_t           k,
               const T*          alpha,
               const T*          A,
               int64_t           lda,
               const T*          B,
               int64_t           ldb,
               const U*          beta,
               T*                C,
               int64_t           ldc);

// trsm
template <typename T>
void ref_trsm(rocblas_side      side,
              rocblas_fill      uplo,
              rocblas_operation transA,
              rocblas_diagonal  diag,
              int64_t           m,
              int64_t           n,
              T                 alpha,
              const T*          A,
              int64_t           lda,
              T*                B,
              int64_t           ldb);

// trmm
template <typename T>
void ref_trmm(rocblas_side      side,
              rocblas_fill      uplo,
              rocblas_operation transA,
              rocblas_diagonal  diag,
              int64_t           m,
              int64_t           n,
              T                 alpha,
              const T*          A,
              int64_t           lda,
              T*                B,
              int64_t           ldb);

/* ============================================================================================ */
