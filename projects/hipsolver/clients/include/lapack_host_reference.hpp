/* ************************************************************************
 * Copyright (C) 2020-2024 Advanced Micro Devices, Inc. All rights reserved.
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
 * ************************************************************************ */

#pragma once

#include "hipsolver.h"
#include "hipsolver_datatype2string.hpp"

// BLAS

template <typename T>
void cpu_gemm(hipsolverOperation_t transA,
              hipsolverOperation_t transB,
              int                  m,
              int                  n,
              int                  k,
              T                    alpha,
              T*                   A,
              int                  lda,
              T*                   B,
              int                  ldb,
              T                    beta,
              T*                   C,
              int                  ldc);

template <typename T>
void cpu_symm_hemm(hipsolverSideMode_t side,
                   hipsolverFillMode_t uplo,
                   int                 m,
                   int                 n,
                   T                   alpha,
                   T*                  A,
                   int                 lda,
                   T*                  B,
                   int                 ldb,
                   T                   beta,
                   T*                  C,
                   int                 ldc);

template <typename T>
void cpu_symv_hemv(hipsolverFillMode_t uplo,
                   int                 n,
                   T                   alpha,
                   T*                  A,
                   int                 lda,
                   T*                  x,
                   int                 incx,
                   T                   beta,
                   T*                  y,
                   int                 incy);

template <typename T>
void cpu_trmm(hipsolverSideMode_t  side,
              hipsolverFillMode_t  uplo,
              hipsolverOperation_t transA,
              char                 diag,
              int                  m,
              int                  n,
              T                    alpha,
              T*                   A,
              int                  lda,
              T*                   B,
              int                  ldb);

template <typename T>
void cpu_trsm(hipsolverSideMode_t  side,
              hipsolverFillMode_t  uplo,
              hipsolverOperation_t transA,
              char                 diag,
              int                  m,
              int                  n,
              T                    alpha,
              T*                   A,
              int                  lda,
              T*                   B,
              int                  ldb);

// LAPACK

template <typename T>
void cpu_lacgv(int n, T* x, int incx);

template <typename T>
void cpu_larf(
    hipsolverSideMode_t side, int m, int n, T* x, int incx, T* alpha, T* A, int lda, T* work);

template <typename T>
void cpu_orgbr_ungbr(hipsolverSideMode_t side,
                     int                 m,
                     int                 n,
                     int                 k,
                     T*                  A,
                     int                 lda,
                     T*                  Ipiv,
                     T*                  work,
                     int                 size_w,
                     int*                info);

template <typename T>
void cpu_orgqr_ungqr(int m, int n, int k, T* A, int lda, T* Ipiv, T* work, int sizeW, int* info);

template <typename T>
void cpu_orgtr_ungtr(
    hipsolverFillMode_t uplo, int n, T* A, int lda, T* Ipiv, T* work, int size_w, int* info);

template <typename T>
void cpu_ormqr_unmqr(hipsolverSideMode_t  side,
                     hipsolverOperation_t trans,
                     int                  m,
                     int                  n,
                     int                  k,
                     T*                   A,
                     int                  lda,
                     T*                   Ipiv,
                     T*                   C,
                     int                  ldc,
                     T*                   work,
                     int                  sizeW,
                     int*                 info);

template <typename T>
void cpu_ormtr_unmtr(hipsolverSideMode_t  side,
                     hipsolverFillMode_t  uplo,
                     hipsolverOperation_t trans,
                     int                  m,
                     int                  n,
                     T*                   A,
                     int                  lda,
                     T*                   Ipiv,
                     T*                   C,
                     int                  ldc,
                     T*                   work,
                     int                  sizeW,
                     int*                 info);

template <typename T, typename S>
void cpu_gebrd(
    int m, int n, T* A, int lda, S* D, S* E, T* tauq, T* taup, T* work, int size_w, int* info);

template <typename T>
void cpu_gels(hipsolverOperation_t transR,
              int                  m,
              int                  n,
              int                  nrhs,
              T*                   A,
              int                  lda,
              T*                   B,
              int                  ldb,
              T*                   work,
              int                  lwork,
              int*                 info);

template <typename T>
void cpu_geqrf(int m, int n, T* A, int lda, T* ipiv, T* work, int sizeW, int* info);

template <typename T>
void cpu_gesv(int n, int nrhs, T* A, int lda, int* ipiv, T* B, int ldb, int* info);

template <typename T, typename W>
void cpu_gesvd(char leftv,
               char rightv,
               int  m,
               int  n,
               T*   A,
               int  lda,
               W*   S,
               T*   U,
               int  ldu,
               T*   V,
               int  ldv,
               T*   work,
               int  lwork,
               W*   E,
               int* info);

template <typename T, typename W>
void cpu_gesvdx(hipsolverEigMode_t leftv,
                hipsolverEigMode_t rightv,
                char               srange,
                int                m,
                int                n,
                T*                 A,
                int                lda,
                W                  vl,
                W                  vu,
                int                il,
                int                iu,
                int*               nsv,
                W*                 S,
                T*                 U,
                int                ldu,
                T*                 V,
                int                ldv,
                T*                 work,
                int                lwork,
                W*                 rwork,
                int*               iwork,
                int*               info);

template <typename T>
void cpu_getrf(int m, int n, T* A, int lda, int* ipiv, int* info);

template <typename T>
void cpu_getrs(hipsolverOperation_t trans,
               int                  n,
               int                  nrhs,
               T*                   A,
               int                  lda,
               int*                 ipiv,
               T*                   B,
               int                  ldb,
               int*                 info);

template <typename T>
void cpu_potrf(hipsolverFillMode_t uplo, int n, T* A, int lda, int* info);

template <typename T>
void cpu_potri(hipsolverFillMode_t uplo, int n, T* A, int lda, int* info);

template <typename T>
void cpu_potrs(hipsolverFillMode_t uplo, int n, int nrhs, T* A, int lda, T* B, int ldb, int* info);

template <typename T, typename S>
void cpu_syevd_heevd(hipsolverEigMode_t  evect,
                     hipsolverFillMode_t uplo,
                     int                 n,
                     T*                  A,
                     int                 lda,
                     S*                  W,
                     T*                  work,
                     int                 lwork,
                     S*                  rwork,
                     int                 lrwork,
                     int*                iwork,
                     int                 liwork,
                     int*                info);

template <typename T, typename S>
void cpu_syevx_heevx(hipsolverEigMode_t  evect,
                     hipsolverEigRange_t erange,
                     hipsolverFillMode_t uplo,
                     int                 n,
                     T*                  A,
                     int                 lda,
                     S                   vl,
                     S                   vu,
                     int                 il,
                     int                 iu,
                     S                   abstol,
                     int*                nev,
                     S*                  W,
                     T*                  Z,
                     int                 ldz,
                     T*                  work,
                     int                 lwork,
                     S*                  rwork,
                     int*                iwork,
                     int*                ifail,
                     int*                info);

template <typename T, typename S>
void cpu_sygvd_hegvd(hipsolverEigType_t  itype,
                     hipsolverEigMode_t  evect,
                     hipsolverFillMode_t uplo,
                     int                 n,
                     T*                  A,
                     int                 lda,
                     T*                  B,
                     int                 ldb,
                     S*                  W,
                     T*                  work,
                     int                 lwork,
                     S*                  rwork,
                     int                 lrwork,
                     int*                iwork,
                     int                 liwork,
                     int*                info);

template <typename T, typename S>
void cpu_sygvx_hegvx(hipsolverEigType_t  itype,
                     hipsolverEigMode_t  evect,
                     hipsolverEigRange_t erange,
                     hipsolverFillMode_t uplo,
                     int                 n,
                     T*                  A,
                     int                 lda,
                     T*                  B,
                     int                 ldb,
                     S                   vl,
                     S                   vu,
                     int                 il,
                     int                 iu,
                     S                   abstol,
                     int*                nev,
                     S*                  W,
                     T*                  Z,
                     int                 ldz,
                     T*                  work,
                     int                 lwork,
                     S*                  rwork,
                     int*                iwork,
                     int*                ifail,
                     int*                info);

template <typename T, typename S>
void cpu_sytrd_hetrd(
    hipsolverFillMode_t uplo, int n, T* A, int lda, S* D, S* E, T* tau, T* work, int size_w);

template <typename T>
void cpu_sytrf(
    hipsolverFillMode_t uplo, int n, T* A, int lda, int* ipiv, T* work, int lwork, int* info);
