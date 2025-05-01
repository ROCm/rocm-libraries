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

#include "../include/lapack_host_reference.hpp"
#include "hipsolver.h"

/*!\file
 * \brief provide template functions interfaces to BLAS and LAPACK interfaces, it is
 * only used for testing, not part of the GPU library
 */

/*************************************************************************/
// Function declarations for LAPACK-provided functions with gfortran-style
// name mangling (lowercase name with trailing underscore).

#ifdef __cplusplus
extern "C" {
#endif

void sgemm_(char*  transA,
            char*  transB,
            int*   m,
            int*   n,
            int*   k,
            float* alpha,
            float* A,
            int*   lda,
            float* B,
            int*   ldb,
            float* beta,
            float* C,
            int*   ldc);
void dgemm_(char*   transA,
            char*   transB,
            int*    m,
            int*    n,
            int*    k,
            double* alpha,
            double* A,
            int*    lda,
            double* B,
            int*    ldb,
            double* beta,
            double* C,
            int*    ldc);
void cgemm_(char*             transA,
            char*             transB,
            int*              m,
            int*              n,
            int*              k,
            hipsolverComplex* alpha,
            hipsolverComplex* A,
            int*              lda,
            hipsolverComplex* B,
            int*              ldb,
            hipsolverComplex* beta,
            hipsolverComplex* C,
            int*              ldc);
void zgemm_(char*                   transA,
            char*                   transB,
            int*                    m,
            int*                    n,
            int*                    k,
            hipsolverDoubleComplex* alpha,
            hipsolverDoubleComplex* A,
            int*                    lda,
            hipsolverDoubleComplex* B,
            int*                    ldb,
            hipsolverDoubleComplex* beta,
            hipsolverDoubleComplex* C,
            int*                    ldc);

void ssymm_(char*  side,
            char*  uplo,
            int*   m,
            int*   n,
            float* alpha,
            float* A,
            int*   lda,
            float* B,
            int*   ldb,
            float* beta,
            float* C,
            int*   ldc);
void dsymm_(char*   side,
            char*   uplo,
            int*    m,
            int*    n,
            double* alpha,
            double* A,
            int*    lda,
            double* B,
            int*    ldb,
            double* beta,
            double* C,
            int*    ldc);
void chemm_(char*             side,
            char*             uplo,
            int*              m,
            int*              n,
            hipsolverComplex* alpha,
            hipsolverComplex* A,
            int*              lda,
            hipsolverComplex* B,
            int*              ldb,
            hipsolverComplex* beta,
            hipsolverComplex* C,
            int*              ldc);
void zhemm_(char*                   side,
            char*                   uplo,
            int*                    m,
            int*                    n,
            hipsolverDoubleComplex* alpha,
            hipsolverDoubleComplex* A,
            int*                    lda,
            hipsolverDoubleComplex* B,
            int*                    ldb,
            hipsolverDoubleComplex* beta,
            hipsolverDoubleComplex* C,
            int*                    ldc);

void ssymv_(char*  uplo,
            int*   n,
            float* alpha,
            float* A,
            int*   lda,
            float* x,
            int*   incx,
            float* beta,
            float* y,
            int*   incy);
void dsymv_(char*   uplo,
            int*    n,
            double* alpha,
            double* A,
            int*    lda,
            double* x,
            int*    incx,
            double* beta,
            double* y,
            int*    incy);
void chemv_(char*             uplo,
            int*              n,
            hipsolverComplex* alpha,
            hipsolverComplex* A,
            int*              lda,
            hipsolverComplex* x,
            int*              incx,
            hipsolverComplex* beta,
            hipsolverComplex* y,
            int*              incy);
void zhemv_(char*                   uplo,
            int*                    n,
            hipsolverDoubleComplex* alpha,
            hipsolverDoubleComplex* A,
            int*                    lda,
            hipsolverDoubleComplex* x,
            int*                    incx,
            hipsolverDoubleComplex* beta,
            hipsolverDoubleComplex* y,
            int*                    incy);

void strmm_(char*  side,
            char*  uplo,
            char*  transA,
            char*  diag,
            int*   m,
            int*   n,
            float* alpha,
            float* A,
            int*   lda,
            float* B,
            int*   ldb);
void dtrmm_(char*   side,
            char*   uplo,
            char*   transA,
            char*   diag,
            int*    m,
            int*    n,
            double* alpha,
            double* A,
            int*    lda,
            double* B,
            int*    ldb);
void ctrmm_(char*             side,
            char*             uplo,
            char*             transA,
            char*             diag,
            int*              m,
            int*              n,
            hipsolverComplex* alpha,
            hipsolverComplex* A,
            int*              lda,
            hipsolverComplex* B,
            int*              ldb);
void ztrmm_(char*                   side,
            char*                   uplo,
            char*                   transA,
            char*                   diag,
            int*                    m,
            int*                    n,
            hipsolverDoubleComplex* alpha,
            hipsolverDoubleComplex* A,
            int*                    lda,
            hipsolverDoubleComplex* B,
            int*                    ldb);

void strsm_(char*  side,
            char*  uplo,
            char*  transA,
            char*  diag,
            int*   m,
            int*   n,
            float* alpha,
            float* A,
            int*   lda,
            float* B,
            int*   ldb);
void dtrsm_(char*   side,
            char*   uplo,
            char*   transA,
            char*   diag,
            int*    m,
            int*    n,
            double* alpha,
            double* A,
            int*    lda,
            double* B,
            int*    ldb);
void ctrsm_(char*             side,
            char*             uplo,
            char*             transA,
            char*             diag,
            int*              m,
            int*              n,
            hipsolverComplex* alpha,
            hipsolverComplex* A,
            int*              lda,
            hipsolverComplex* B,
            int*              ldb);
void ztrsm_(char*                   side,
            char*                   uplo,
            char*                   transA,
            char*                   diag,
            int*                    m,
            int*                    n,
            hipsolverDoubleComplex* alpha,
            hipsolverDoubleComplex* A,
            int*                    lda,
            hipsolverDoubleComplex* B,
            int*                    ldb);

void clacgv_(int* n, hipsolverComplex* x, int* incx);
void zlacgv_(int* n, hipsolverDoubleComplex* x, int* incx);

void slarf_(
    char* side, int* m, int* n, float* x, int* incx, float* alpha, float* A, int* lda, float* work);
void dlarf_(char*   side,
            int*    m,
            int*    n,
            double* x,
            int*    incx,
            double* alpha,
            double* A,
            int*    lda,
            double* work);
void clarf_(char*             side,
            int*              m,
            int*              n,
            hipsolverComplex* x,
            int*              incx,
            hipsolverComplex* alpha,
            hipsolverComplex* A,
            int*              lda,
            hipsolverComplex* work);
void zlarf_(char*                   side,
            int*                    m,
            int*                    n,
            hipsolverDoubleComplex* x,
            int*                    incx,
            hipsolverDoubleComplex* alpha,
            hipsolverDoubleComplex* A,
            int*                    lda,
            hipsolverDoubleComplex* work);

void sorgbr_(char*  vect,
             int*   m,
             int*   n,
             int*   k,
             float* A,
             int*   lda,
             float* Ipiv,
             float* work,
             int*   size_w,
             int*   info);
void dorgbr_(char*   vect,
             int*    m,
             int*    n,
             int*    k,
             double* A,
             int*    lda,
             double* Ipiv,
             double* work,
             int*    size_w,
             int*    info);
void cungbr_(char*             vect,
             int*              m,
             int*              n,
             int*              k,
             hipsolverComplex* A,
             int*              lda,
             hipsolverComplex* Ipiv,
             hipsolverComplex* work,
             int*              size_w,
             int*              info);
void zungbr_(char*                   vect,
             int*                    m,
             int*                    n,
             int*                    k,
             hipsolverDoubleComplex* A,
             int*                    lda,
             hipsolverDoubleComplex* Ipiv,
             hipsolverDoubleComplex* work,
             int*                    size_w,
             int*                    info);

void sorgqr_(
    int* m, int* n, int* k, float* A, int* lda, float* ipiv, float* work, int* lwork, int* info);
void dorgqr_(
    int* m, int* n, int* k, double* A, int* lda, double* ipiv, double* work, int* lwork, int* info);
void cungqr_(int*              m,
             int*              n,
             int*              k,
             hipsolverComplex* A,
             int*              lda,
             hipsolverComplex* ipiv,
             hipsolverComplex* work,
             int*              lwork,
             int*              info);
void zungqr_(int*                    m,
             int*                    n,
             int*                    k,
             hipsolverDoubleComplex* A,
             int*                    lda,
             hipsolverDoubleComplex* ipiv,
             hipsolverDoubleComplex* work,
             int*                    lwork,
             int*                    info);

void sorgtr_(
    char* uplo, int* n, float* A, int* lda, float* Ipiv, float* work, int* size_w, int* info);
void dorgtr_(
    char* uplo, int* n, double* A, int* lda, double* Ipiv, double* work, int* size_w, int* info);
void cungtr_(char*             uplo,
             int*              n,
             hipsolverComplex* A,
             int*              lda,
             hipsolverComplex* Ipiv,
             hipsolverComplex* work,
             int*              size_w,
             int*              info);
void zungtr_(char*                   uplo,
             int*                    n,
             hipsolverDoubleComplex* A,
             int*                    lda,
             hipsolverDoubleComplex* Ipiv,
             hipsolverDoubleComplex* work,
             int*                    size_w,
             int*                    info);

void sormqr_(char*  side,
             char*  trans,
             int*   m,
             int*   n,
             int*   k,
             float* A,
             int*   lda,
             float* ipiv,
             float* C,
             int*   ldc,
             float* work,
             int*   sizeW,
             int*   info);
void dormqr_(char*   side,
             char*   trans,
             int*    m,
             int*    n,
             int*    k,
             double* A,
             int*    lda,
             double* ipiv,
             double* C,
             int*    ldc,
             double* work,
             int*    sizeW,
             int*    info);
void cunmqr_(char*             side,
             char*             trans,
             int*              m,
             int*              n,
             int*              k,
             hipsolverComplex* A,
             int*              lda,
             hipsolverComplex* ipiv,
             hipsolverComplex* C,
             int*              ldc,
             hipsolverComplex* work,
             int*              sizeW,
             int*              info);
void zunmqr_(char*                   side,
             char*                   trans,
             int*                    m,
             int*                    n,
             int*                    k,
             hipsolverDoubleComplex* A,
             int*                    lda,
             hipsolverDoubleComplex* ipiv,
             hipsolverDoubleComplex* C,
             int*                    ldc,
             hipsolverDoubleComplex* work,
             int*                    sizeW,
             int*                    info);

void sormtr_(char*  side,
             char*  uplo,
             char*  trans,
             int*   m,
             int*   n,
             float* A,
             int*   lda,
             float* ipiv,
             float* C,
             int*   ldc,
             float* work,
             int*   sizeW,
             int*   info);
void dormtr_(char*   side,
             char*   uplo,
             char*   trans,
             int*    m,
             int*    n,
             double* A,
             int*    lda,
             double* ipiv,
             double* C,
             int*    ldc,
             double* work,
             int*    sizeW,
             int*    info);
void cunmtr_(char*             side,
             char*             uplo,
             char*             trans,
             int*              m,
             int*              n,
             hipsolverComplex* A,
             int*              lda,
             hipsolverComplex* ipiv,
             hipsolverComplex* C,
             int*              ldc,
             hipsolverComplex* work,
             int*              sizeW,
             int*              info);
void zunmtr_(char*                   side,
             char*                   uplo,
             char*                   trans,
             int*                    m,
             int*                    n,
             hipsolverDoubleComplex* A,
             int*                    lda,
             hipsolverDoubleComplex* ipiv,
             hipsolverDoubleComplex* C,
             int*                    ldc,
             hipsolverDoubleComplex* work,
             int*                    sizeW,
             int*                    info);

void sgebrd_(int*   m,
             int*   n,
             float* A,
             int*   lda,
             float* D,
             float* E,
             float* tauq,
             float* taup,
             float* work,
             int*   size_w,
             int*   info);
void dgebrd_(int*    m,
             int*    n,
             double* A,
             int*    lda,
             double* D,
             double* E,
             double* tauq,
             double* taup,
             double* work,
             int*    size_w,
             int*    info);
void cgebrd_(int*              m,
             int*              n,
             hipsolverComplex* A,
             int*              lda,
             float*            D,
             float*            E,
             hipsolverComplex* tauq,
             hipsolverComplex* taup,
             hipsolverComplex* work,
             int*              size_w,
             int*              info);
void zgebrd_(int*                    m,
             int*                    n,
             hipsolverDoubleComplex* A,
             int*                    lda,
             double*                 D,
             double*                 E,
             hipsolverDoubleComplex* tauq,
             hipsolverDoubleComplex* taup,
             hipsolverDoubleComplex* work,
             int*                    size_w,
             int*                    info);

void sgels_(char*  trans,
            int*   m,
            int*   n,
            int*   nrhs,
            float* A,
            int*   lda,
            float* B,
            int*   ldb,
            float* work,
            int*   lwork,
            int*   info);
void dgels_(char*   trans,
            int*    m,
            int*    n,
            int*    nrhs,
            double* A,
            int*    lda,
            double* B,
            int*    ldb,
            double* work,
            int*    lwork,
            int*    info);
void cgels_(char*             trans,
            int*              m,
            int*              n,
            int*              nrhs,
            hipsolverComplex* A,
            int*              lda,
            hipsolverComplex* B,
            int*              ldb,
            hipsolverComplex* work,
            int*              lwork,
            int*              info);
void zgels_(char*                   trans,
            int*                    m,
            int*                    n,
            int*                    nrhs,
            hipsolverDoubleComplex* A,
            int*                    lda,
            hipsolverDoubleComplex* B,
            int*                    ldb,
            hipsolverDoubleComplex* work,
            int*                    lwork,
            int*                    info);

void sgeqrf_(int* m, int* n, float* A, int* lda, float* ipiv, float* work, int* lwork, int* info);
void dgeqrf_(
    int* m, int* n, double* A, int* lda, double* ipiv, double* work, int* lwork, int* info);
void cgeqrf_(int*              m,
             int*              n,
             hipsolverComplex* A,
             int*              lda,
             hipsolverComplex* ipiv,
             hipsolverComplex* work,
             int*              lwork,
             int*              info);
void zgeqrf_(int*                    m,
             int*                    n,
             hipsolverDoubleComplex* A,
             int*                    lda,
             hipsolverDoubleComplex* ipiv,
             hipsolverDoubleComplex* work,
             int*                    lwork,
             int*                    info);

void sgesv_(int* n, int* nrhs, float* A, int* lda, int* ipiv, float* B, int* ldb, int* info);
void dgesv_(int* n, int* nrhs, double* A, int* lda, int* ipiv, double* B, int* ldb, int* info);
void cgesv_(int*              n,
            int*              nrhs,
            hipsolverComplex* A,
            int*              lda,
            int*              ipiv,
            hipsolverComplex* B,
            int*              ldb,
            int*              info);
void zgesv_(int*                    n,
            int*                    nrhs,
            hipsolverDoubleComplex* A,
            int*                    lda,
            int*                    ipiv,
            hipsolverDoubleComplex* B,
            int*                    ldb,
            int*                    info);

void sgesvd_(char*  jobu,
             char*  jobv,
             int*   m,
             int*   n,
             float* A,
             int*   lda,
             float* S,
             float* U,
             int*   ldu,
             float* V,
             int*   ldv,
             float* E,
             int*   lwork,
             int*   info);
void dgesvd_(char*   jobu,
             char*   jobv,
             int*    m,
             int*    n,
             double* A,
             int*    lda,
             double* S,
             double* U,
             int*    ldu,
             double* V,
             int*    ldv,
             double* E,
             int*    lwork,
             int*    info);
void cgesvd_(char*             jobu,
             char*             jobv,
             int*              m,
             int*              n,
             hipsolverComplex* A,
             int*              lda,
             float*            S,
             hipsolverComplex* U,
             int*              ldu,
             hipsolverComplex* V,
             int*              ldv,
             hipsolverComplex* work,
             int*              lwork,
             float*            E,
             int*              info);
void zgesvd_(char*                   jobu,
             char*                   jobv,
             int*                    m,
             int*                    n,
             hipsolverDoubleComplex* A,
             int*                    lda,
             double*                 S,
             hipsolverDoubleComplex* U,
             int*                    ldu,
             hipsolverDoubleComplex* V,
             int*                    ldv,
             hipsolverDoubleComplex* work,
             int*                    lwork,
             double*                 E,
             int*                    info);

void sgesvdx_(char*  jobu,
              char*  jobv,
              char*  srange,
              int*   m,
              int*   n,
              float* A,
              int*   lda,
              float* vl,
              float* vu,
              int*   il,
              int*   iu,
              int*   nsv,
              float* S,
              float* U,
              int*   ldu,
              float* V,
              int*   ldv,
              float* work,
              int*   lwork,
              int*   iwork,
              int*   info);
void dgesvdx_(char*   jobu,
              char*   jobv,
              char*   srange,
              int*    m,
              int*    n,
              double* A,
              int*    lda,
              double* vl,
              double* vu,
              int*    il,
              int*    iu,
              int*    nsv,
              double* S,
              double* U,
              int*    ldu,
              double* V,
              int*    ldv,
              double* work,
              int*    lwork,
              int*    iwork,
              int*    info);
void cgesvdx_(char*             jobu,
              char*             jobv,
              char*             srange,
              int*              m,
              int*              n,
              hipsolverComplex* A,
              int*              lda,
              float*            vl,
              float*            vu,
              int*              il,
              int*              iu,
              int*              nsv,
              float*            S,
              hipsolverComplex* U,
              int*              ldu,
              hipsolverComplex* V,
              int*              ldv,
              hipsolverComplex* work,
              int*              lwork,
              float*            rwork,
              int*              iwork,
              int*              info);
void zgesvdx_(char*                   jobu,
              char*                   jobv,
              char*                   srange,
              int*                    m,
              int*                    n,
              hipsolverDoubleComplex* A,
              int*                    lda,
              double*                 vl,
              double*                 vu,
              int*                    il,
              int*                    iu,
              int*                    nsv,
              double*                 S,
              hipsolverDoubleComplex* U,
              int*                    ldu,
              hipsolverDoubleComplex* V,
              int*                    ldv,
              hipsolverDoubleComplex* work,
              int*                    lwork,
              double*                 rwork,
              int*                    iwork,
              int*                    info);

void sgetrf_(int* m, int* n, float* A, int* lda, int* ipiv, int* info);
void dgetrf_(int* m, int* n, double* A, int* lda, int* ipiv, int* info);
void cgetrf_(int* m, int* n, hipsolverComplex* A, int* lda, int* ipiv, int* info);
void zgetrf_(int* m, int* n, hipsolverDoubleComplex* A, int* lda, int* ipiv, int* info);

void sgetrs_(
    char* trans, int* n, int* nrhs, float* A, int* lda, int* ipiv, float* B, int* ldb, int* info);
void dgetrs_(
    char* trans, int* n, int* nrhs, double* A, int* lda, int* ipiv, double* B, int* ldb, int* info);
void cgetrs_(char*             trans,
             int*              n,
             int*              nrhs,
             hipsolverComplex* A,
             int*              lda,
             int*              ipiv,
             hipsolverComplex* B,
             int*              ldb,
             int*              info);
void zgetrs_(char*                   trans,
             int*                    n,
             int*                    nrhs,
             hipsolverDoubleComplex* A,
             int*                    lda,
             int*                    ipiv,
             hipsolverDoubleComplex* B,
             int*                    ldb,
             int*                    info);

void spotrf_(char* uplo, int* m, float* A, int* lda, int* info);
void dpotrf_(char* uplo, int* m, double* A, int* lda, int* info);
void cpotrf_(char* uplo, int* m, hipsolverComplex* A, int* lda, int* info);
void zpotrf_(char* uplo, int* m, hipsolverDoubleComplex* A, int* lda, int* info);

void spotri_(char* uplo, int* n, float* A, int* lda, int* info);
void dpotri_(char* uplo, int* n, double* A, int* lda, int* info);
void cpotri_(char* uplo, int* n, hipsolverComplex* A, int* lda, int* info);
void zpotri_(char* uplo, int* n, hipsolverDoubleComplex* A, int* lda, int* info);

void spotrs_(char* uplo, int* n, int* nrhs, float* A, int* lda, float* B, int* ldb, int* info);
void dpotrs_(char* uplo, int* n, int* nrhs, double* A, int* lda, double* B, int* ldb, int* info);
void cpotrs_(char*             uplo,
             int*              n,
             int*              nrhs,
             hipsolverComplex* A,
             int*              lda,
             hipsolverComplex* B,
             int*              ldb,
             int*              info);
void zpotrs_(char*                   uplo,
             int*                    n,
             int*                    nrhs,
             hipsolverDoubleComplex* A,
             int*                    lda,
             hipsolverDoubleComplex* B,
             int*                    ldb,
             int*                    info);

void ssyevd_(char*  evect,
             char*  uplo,
             int*   n,
             float* A,
             int*   lda,
             float* W,
             float* work,
             int*   lwork,
             int*   iwork,
             int*   liwork,
             int*   info);
void dsyevd_(char*   evect,
             char*   uplo,
             int*    n,
             double* A,
             int*    lda,
             double* W,
             double* work,
             int*    lwork,
             int*    iwork,
             int*    liwork,
             int*    info);
void cheevd_(char*             evect,
             char*             uplo,
             int*              n,
             hipsolverComplex* A,
             int*              lda,
             float*            W,
             hipsolverComplex* work,
             int*              lwork,
             float*            rwork,
             int*              lrwork,
             int*              iwork,
             int*              liwork,
             int*              info);
void zheevd_(char*                   evect,
             char*                   uplo,
             int*                    n,
             hipsolverDoubleComplex* A,
             int*                    lda,
             double*                 W,
             hipsolverDoubleComplex* work,
             int*                    lwork,
             double*                 rwork,
             int*                    lrwork,
             int*                    iwork,
             int*                    liwork,
             int*                    info);

void ssyevx_(char*  evect,
             char*  erange,
             char*  uplo,
             int*   n,
             float* A,
             int*   lda,
             float* vl,
             float* vu,
             int*   il,
             int*   iu,
             float* abstol,
             int*   nev,
             float* W,
             float* Z,
             int*   ldz,
             float* work,
             int*   lwork,
             int*   iwork,
             int*   ifail,
             int*   info);
void dsyevx_(char*   evect,
             char*   erange,
             char*   uplo,
             int*    n,
             double* A,
             int*    lda,
             double* vl,
             double* vu,
             int*    il,
             int*    iu,
             double* abstol,
             int*    nev,
             double* W,
             double* Z,
             int*    ldz,
             double* work,
             int*    lwork,
             int*    iwork,
             int*    ifail,
             int*    info);
void cheevx_(char*             evect,
             char*             erange,
             char*             uplo,
             int*              n,
             hipsolverComplex* A,
             int*              lda,
             float*            vl,
             float*            vu,
             int*              il,
             int*              iu,
             float*            abstol,
             int*              nev,
             float*            W,
             hipsolverComplex* Z,
             int*              ldz,
             hipsolverComplex* work,
             int*              lwork,
             float*            rwork,
             int*              iwork,
             int*              ifail,
             int*              info);
void zheevx_(char*                   evect,
             char*                   erange,
             char*                   uplo,
             int*                    n,
             hipsolverDoubleComplex* A,
             int*                    lda,
             double*                 vl,
             double*                 vu,
             int*                    il,
             int*                    iu,
             double*                 abstol,
             int*                    nev,
             double*                 W,
             hipsolverDoubleComplex* Z,
             int*                    ldz,
             hipsolverDoubleComplex* work,
             int*                    lwork,
             double*                 rwork,
             int*                    iwork,
             int*                    ifail,
             int*                    info);

void ssygvd_(int*   itype,
             char*  evect,
             char*  uplo,
             int*   n,
             float* A,
             int*   lda,
             float* B,
             int*   ldb,
             float* W,
             float* work,
             int*   lwork,
             int*   iwork,
             int*   liwork,
             int*   info);
void dsygvd_(int*    itype,
             char*   evect,
             char*   uplo,
             int*    n,
             double* A,
             int*    lda,
             double* B,
             int*    ldb,
             double* W,
             double* work,
             int*    lwork,
             int*    iwork,
             int*    liwork,
             int*    info);
void chegvd_(int*              itype,
             char*             evect,
             char*             uplo,
             int*              n,
             hipsolverComplex* A,
             int*              lda,
             hipsolverComplex* B,
             int*              ldb,
             float*            W,
             hipsolverComplex* work,
             int*              lwork,
             float*            rwork,
             int*              lrwork,
             int*              iwork,
             int*              liwork,
             int*              info);
void zhegvd_(int*                    itype,
             char*                   evect,
             char*                   uplo,
             int*                    n,
             hipsolverDoubleComplex* A,
             int*                    lda,
             hipsolverDoubleComplex* B,
             int*                    ldb,
             double*                 W,
             hipsolverDoubleComplex* work,
             int*                    lwork,
             double*                 rwork,
             int*                    lrwork,
             int*                    iwork,
             int*                    liwork,
             int*                    info);

void ssygvx_(int*   itype,
             char*  evect,
             char*  erange,
             char*  uplo,
             int*   n,
             float* A,
             int*   lda,
             float* B,
             int*   ldb,
             float* vl,
             float* vu,
             int*   il,
             int*   iu,
             float* abstol,
             int*   nev,
             float* W,
             float* Z,
             int*   ldz,
             float* work,
             int*   lwork,
             int*   iwork,
             int*   ifail,
             int*   info);
void dsygvx_(int*    itype,
             char*   evect,
             char*   erange,
             char*   uplo,
             int*    n,
             double* A,
             int*    lda,
             double* B,
             int*    ldb,
             double* vl,
             double* vu,
             int*    il,
             int*    iu,
             double* abstol,
             int*    nev,
             double* W,
             double* Z,
             int*    ldz,
             double* work,
             int*    lwork,
             int*    iwork,
             int*    ifail,
             int*    info);
void chegvx_(int*              itype,
             char*             evect,
             char*             erange,
             char*             uplo,
             int*              n,
             hipsolverComplex* A,
             int*              lda,
             hipsolverComplex* B,
             int*              ldb,
             float*            vl,
             float*            vu,
             int*              il,
             int*              iu,
             float*            abstol,
             int*              nev,
             float*            W,
             hipsolverComplex* Z,
             int*              ldz,
             hipsolverComplex* work,
             int*              lwork,
             float*            rwork,
             int*              iwork,
             int*              ifail,
             int*              info);
void zhegvx_(int*                    itype,
             char*                   evect,
             char*                   erange,
             char*                   uplo,
             int*                    n,
             hipsolverDoubleComplex* A,
             int*                    lda,
             hipsolverDoubleComplex* B,
             int*                    ldb,
             double*                 vl,
             double*                 vu,
             int*                    il,
             int*                    iu,
             double*                 abstol,
             int*                    nev,
             double*                 W,
             hipsolverDoubleComplex* Z,
             int*                    ldz,
             hipsolverDoubleComplex* work,
             int*                    lwork,
             double*                 rwork,
             int*                    iwork,
             int*                    ifail,
             int*                    info);

void ssytrd_(char*  uplo,
             int*   n,
             float* A,
             int*   lda,
             float* D,
             float* E,
             float* tau,
             float* work,
             int*   size_w,
             int*   info);
void dsytrd_(char*   uplo,
             int*    n,
             double* A,
             int*    lda,
             double* D,
             double* E,
             double* tau,
             double* work,
             int*    size_w,
             int*    info);
void chetrd_(char*             uplo,
             int*              n,
             hipsolverComplex* A,
             int*              lda,
             float*            D,
             float*            E,
             hipsolverComplex* tau,
             hipsolverComplex* work,
             int*              size_w,
             int*              info);
void zhetrd_(char*                   uplo,
             int*                    n,
             hipsolverDoubleComplex* A,
             int*                    lda,
             double*                 D,
             double*                 E,
             hipsolverDoubleComplex* tau,
             hipsolverDoubleComplex* work,
             int*                    size_w,
             int*                    info);

void ssytrf_(char* uplo, int* n, float* A, int* lda, int* ipiv, float* work, int* lwork, int* info);
void dsytrf_(
    char* uplo, int* n, double* A, int* lda, int* ipiv, double* work, int* lwork, int* info);
void csytrf_(char*             uplo,
             int*              n,
             hipsolverComplex* A,
             int*              lda,
             int*              ipiv,
             hipsolverComplex* work,
             int*              lwork,
             int*              info);
void zsytrf_(char*                   uplo,
             int*                    n,
             hipsolverDoubleComplex* A,
             int*                    lda,
             int*                    ipiv,
             hipsolverDoubleComplex* work,
             int*                    lwork,
             int*                    info);

#ifdef __cplusplus
}
#endif
/************************************************************************/

/************************************************************************/
// These are templated BLAS functions used in hipSOLVER clients code

// gemm
template <>
void cpu_gemm<float>(hipsolverOperation_t transA,
                     hipsolverOperation_t transB,
                     int                  m,
                     int                  n,
                     int                  k,
                     float                alpha,
                     float*               A,
                     int                  lda,
                     float*               B,
                     int                  ldb,
                     float                beta,
                     float*               C,
                     int                  ldc)
{
    char transAC = hipsolver2char_operation(transA);
    char transBC = hipsolver2char_operation(transB);
    sgemm_(&transAC, &transBC, &m, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
}

template <>
void cpu_gemm<double>(hipsolverOperation_t transA,
                      hipsolverOperation_t transB,
                      int                  m,
                      int                  n,
                      int                  k,
                      double               alpha,
                      double*              A,
                      int                  lda,
                      double*              B,
                      int                  ldb,
                      double               beta,
                      double*              C,
                      int                  ldc)
{
    char transAC = hipsolver2char_operation(transA);
    char transBC = hipsolver2char_operation(transB);
    dgemm_(&transAC, &transBC, &m, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
}

template <>
void cpu_gemm<hipsolverComplex>(hipsolverOperation_t transA,
                                hipsolverOperation_t transB,
                                int                  m,
                                int                  n,
                                int                  k,
                                hipsolverComplex     alpha,
                                hipsolverComplex*    A,
                                int                  lda,
                                hipsolverComplex*    B,
                                int                  ldb,
                                hipsolverComplex     beta,
                                hipsolverComplex*    C,
                                int                  ldc)
{
    char transAC = hipsolver2char_operation(transA);
    char transBC = hipsolver2char_operation(transB);
    cgemm_(&transAC, &transBC, &m, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
}

template <>
void cpu_gemm<hipsolverDoubleComplex>(hipsolverOperation_t    transA,
                                      hipsolverOperation_t    transB,
                                      int                     m,
                                      int                     n,
                                      int                     k,
                                      hipsolverDoubleComplex  alpha,
                                      hipsolverDoubleComplex* A,
                                      int                     lda,
                                      hipsolverDoubleComplex* B,
                                      int                     ldb,
                                      hipsolverDoubleComplex  beta,
                                      hipsolverDoubleComplex* C,
                                      int                     ldc)
{
    char transAC = hipsolver2char_operation(transA);
    char transBC = hipsolver2char_operation(transB);
    zgemm_(&transAC, &transBC, &m, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
}

// symm & hemm
template <>
void cpu_symm_hemm<float>(hipsolverSideMode_t side,
                          hipsolverFillMode_t uplo,
                          int                 m,
                          int                 n,
                          float               alpha,
                          float*              A,
                          int                 lda,
                          float*              B,
                          int                 ldb,
                          float               beta,
                          float*              C,
                          int                 ldc)
{
    char sideC = hipsolver2char_side(side);
    char uploC = hipsolver2char_fill(uplo);
    ssymm_(&sideC, &uploC, &m, &n, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
}

template <>
void cpu_symm_hemm<double>(hipsolverSideMode_t side,
                           hipsolverFillMode_t uplo,
                           int                 m,
                           int                 n,
                           double              alpha,
                           double*             A,
                           int                 lda,
                           double*             B,
                           int                 ldb,
                           double              beta,
                           double*             C,
                           int                 ldc)
{
    char sideC = hipsolver2char_side(side);
    char uploC = hipsolver2char_fill(uplo);
    dsymm_(&sideC, &uploC, &m, &n, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
}

template <>
void cpu_symm_hemm<hipsolverComplex>(hipsolverSideMode_t side,
                                     hipsolverFillMode_t uplo,
                                     int                 m,
                                     int                 n,
                                     hipsolverComplex    alpha,
                                     hipsolverComplex*   A,
                                     int                 lda,
                                     hipsolverComplex*   B,
                                     int                 ldb,
                                     hipsolverComplex    beta,
                                     hipsolverComplex*   C,
                                     int                 ldc)
{
    char sideC = hipsolver2char_side(side);
    char uploC = hipsolver2char_fill(uplo);
    chemm_(&sideC, &uploC, &m, &n, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
}

template <>
void cpu_symm_hemm<hipsolverDoubleComplex>(hipsolverSideMode_t     side,
                                           hipsolverFillMode_t     uplo,
                                           int                     m,
                                           int                     n,
                                           hipsolverDoubleComplex  alpha,
                                           hipsolverDoubleComplex* A,
                                           int                     lda,
                                           hipsolverDoubleComplex* B,
                                           int                     ldb,
                                           hipsolverDoubleComplex  beta,
                                           hipsolverDoubleComplex* C,
                                           int                     ldc)
{
    char sideC = hipsolver2char_side(side);
    char uploC = hipsolver2char_fill(uplo);
    zhemm_(&sideC, &uploC, &m, &n, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
}

// symv/hemv
template <>
void cpu_symv_hemv<float>(hipsolverFillMode_t uplo,
                          int                 n,
                          float               alpha,
                          float*              A,
                          int                 lda,
                          float*              x,
                          int                 incx,
                          float               beta,
                          float*              y,
                          int                 incy)
{
    char uploC = hipsolver2char_fill(uplo);
    ssymv_(&uploC, &n, &alpha, A, &lda, x, &incx, &beta, y, &incy);
}

template <>
void cpu_symv_hemv<double>(hipsolverFillMode_t uplo,
                           int                 n,
                           double              alpha,
                           double*             A,
                           int                 lda,
                           double*             x,
                           int                 incx,
                           double              beta,
                           double*             y,
                           int                 incy)
{
    char uploC = hipsolver2char_fill(uplo);
    dsymv_(&uploC, &n, &alpha, A, &lda, x, &incx, &beta, y, &incy);
}

template <>
void cpu_symv_hemv<hipsolverComplex>(hipsolverFillMode_t uplo,
                                     int                 n,
                                     hipsolverComplex    alpha,
                                     hipsolverComplex*   A,
                                     int                 lda,
                                     hipsolverComplex*   x,
                                     int                 incx,
                                     hipsolverComplex    beta,
                                     hipsolverComplex*   y,
                                     int                 incy)
{
    char uploC = hipsolver2char_fill(uplo);
    chemv_(&uploC, &n, &alpha, A, &lda, x, &incx, &beta, y, &incy);
}

template <>
void cpu_symv_hemv<hipsolverDoubleComplex>(hipsolverFillMode_t     uplo,
                                           int                     n,
                                           hipsolverDoubleComplex  alpha,
                                           hipsolverDoubleComplex* A,
                                           int                     lda,
                                           hipsolverDoubleComplex* x,
                                           int                     incx,
                                           hipsolverDoubleComplex  beta,
                                           hipsolverDoubleComplex* y,
                                           int                     incy)
{
    char uploC = hipsolver2char_fill(uplo);
    zhemv_(&uploC, &n, &alpha, A, &lda, x, &incx, &beta, y, &incy);
}

// trmm
template <>
void cpu_trmm<float>(hipsolverSideMode_t  side,
                     hipsolverFillMode_t  uplo,
                     hipsolverOperation_t transA,
                     char                 diag,
                     int                  m,
                     int                  n,
                     float                alpha,
                     float*               A,
                     int                  lda,
                     float*               B,
                     int                  ldb)
{
    char sideC  = hipsolver2char_side(side);
    char uploC  = hipsolver2char_fill(uplo);
    char transC = hipsolver2char_operation(transA);
    strmm_(&sideC, &uploC, &transC, &diag, &m, &n, &alpha, A, &lda, B, &ldb);
}

template <>
void cpu_trmm<double>(hipsolverSideMode_t  side,
                      hipsolverFillMode_t  uplo,
                      hipsolverOperation_t transA,
                      char                 diag,
                      int                  m,
                      int                  n,
                      double               alpha,
                      double*              A,
                      int                  lda,
                      double*              B,
                      int                  ldb)
{
    char sideC  = hipsolver2char_side(side);
    char uploC  = hipsolver2char_fill(uplo);
    char transC = hipsolver2char_operation(transA);
    dtrmm_(&sideC, &uploC, &transC, &diag, &m, &n, &alpha, A, &lda, B, &ldb);
}

template <>
void cpu_trmm<hipsolverComplex>(hipsolverSideMode_t  side,
                                hipsolverFillMode_t  uplo,
                                hipsolverOperation_t transA,
                                char                 diag,
                                int                  m,
                                int                  n,
                                hipsolverComplex     alpha,
                                hipsolverComplex*    A,
                                int                  lda,
                                hipsolverComplex*    B,
                                int                  ldb)
{
    char sideC  = hipsolver2char_side(side);
    char uploC  = hipsolver2char_fill(uplo);
    char transC = hipsolver2char_operation(transA);
    ctrmm_(&sideC, &uploC, &transC, &diag, &m, &n, &alpha, A, &lda, B, &ldb);
}

template <>
void cpu_trmm<hipsolverDoubleComplex>(hipsolverSideMode_t     side,
                                      hipsolverFillMode_t     uplo,
                                      hipsolverOperation_t    transA,
                                      char                    diag,
                                      int                     m,
                                      int                     n,
                                      hipsolverDoubleComplex  alpha,
                                      hipsolverDoubleComplex* A,
                                      int                     lda,
                                      hipsolverDoubleComplex* B,
                                      int                     ldb)
{
    char sideC  = hipsolver2char_side(side);
    char uploC  = hipsolver2char_fill(uplo);
    char transC = hipsolver2char_operation(transA);
    ztrmm_(&sideC, &uploC, &transC, &diag, &m, &n, &alpha, A, &lda, B, &ldb);
}

// trsm
template <>
void cpu_trsm<float>(hipsolverSideMode_t  side,
                     hipsolverFillMode_t  uplo,
                     hipsolverOperation_t transA,
                     char                 diag,
                     int                  m,
                     int                  n,
                     float                alpha,
                     float*               A,
                     int                  lda,
                     float*               B,
                     int                  ldb)
{
    char sideC  = hipsolver2char_side(side);
    char uploC  = hipsolver2char_fill(uplo);
    char transC = hipsolver2char_operation(transA);
    strsm_(&sideC, &uploC, &transC, &diag, &m, &n, &alpha, A, &lda, B, &ldb);
}

template <>
void cpu_trsm<double>(hipsolverSideMode_t  side,
                      hipsolverFillMode_t  uplo,
                      hipsolverOperation_t transA,
                      char                 diag,
                      int                  m,
                      int                  n,
                      double               alpha,
                      double*              A,
                      int                  lda,
                      double*              B,
                      int                  ldb)
{
    char sideC  = hipsolver2char_side(side);
    char uploC  = hipsolver2char_fill(uplo);
    char transC = hipsolver2char_operation(transA);
    dtrsm_(&sideC, &uploC, &transC, &diag, &m, &n, &alpha, A, &lda, B, &ldb);
}

template <>
void cpu_trsm<hipsolverComplex>(hipsolverSideMode_t  side,
                                hipsolverFillMode_t  uplo,
                                hipsolverOperation_t transA,
                                char                 diag,
                                int                  m,
                                int                  n,
                                hipsolverComplex     alpha,
                                hipsolverComplex*    A,
                                int                  lda,
                                hipsolverComplex*    B,
                                int                  ldb)
{
    char sideC  = hipsolver2char_side(side);
    char uploC  = hipsolver2char_fill(uplo);
    char transC = hipsolver2char_operation(transA);
    ctrsm_(&sideC, &uploC, &transC, &diag, &m, &n, &alpha, A, &lda, B, &ldb);
}

template <>
void cpu_trsm<hipsolverDoubleComplex>(hipsolverSideMode_t     side,
                                      hipsolverFillMode_t     uplo,
                                      hipsolverOperation_t    transA,
                                      char                    diag,
                                      int                     m,
                                      int                     n,
                                      hipsolverDoubleComplex  alpha,
                                      hipsolverDoubleComplex* A,
                                      int                     lda,
                                      hipsolverDoubleComplex* B,
                                      int                     ldb)
{
    char sideC  = hipsolver2char_side(side);
    char uploC  = hipsolver2char_fill(uplo);
    char transC = hipsolver2char_operation(transA);
    ztrsm_(&sideC, &uploC, &transC, &diag, &m, &n, &alpha, A, &lda, B, &ldb);
}

/************************************************************************/
// These are templated LAPACK functions used in hipSOLVER clients code

// lacgv
template <>
void cpu_lacgv<hipsolverComplex>(int n, hipsolverComplex* x, int incx)
{
    clacgv_(&n, x, &incx);
}

template <>
void cpu_lacgv<hipsolverDoubleComplex>(int n, hipsolverDoubleComplex* x, int incx)
{
    zlacgv_(&n, x, &incx);
}

// larf
template <>
void cpu_larf<float>(hipsolverSideMode_t sideR,
                     int                 m,
                     int                 n,
                     float*              x,
                     int                 incx,
                     float*              alpha,
                     float*              A,
                     int                 lda,
                     float*              work)
{
    char side = hipsolver2char_side(sideR);
    slarf_(&side, &m, &n, x, &incx, alpha, A, &lda, work);
}

template <>
void cpu_larf<double>(hipsolverSideMode_t sideR,
                      int                 m,
                      int                 n,
                      double*             x,
                      int                 incx,
                      double*             alpha,
                      double*             A,
                      int                 lda,
                      double*             work)
{
    char side = hipsolver2char_side(sideR);
    dlarf_(&side, &m, &n, x, &incx, alpha, A, &lda, work);
}

template <>
void cpu_larf<hipsolverComplex>(hipsolverSideMode_t sideR,
                                int                 m,
                                int                 n,
                                hipsolverComplex*   x,
                                int                 incx,
                                hipsolverComplex*   alpha,
                                hipsolverComplex*   A,
                                int                 lda,
                                hipsolverComplex*   work)
{
    char side = hipsolver2char_side(sideR);
    clarf_(&side, &m, &n, x, &incx, alpha, A, &lda, work);
}

template <>
void cpu_larf<hipsolverDoubleComplex>(hipsolverSideMode_t     sideR,
                                      int                     m,
                                      int                     n,
                                      hipsolverDoubleComplex* x,
                                      int                     incx,
                                      hipsolverDoubleComplex* alpha,
                                      hipsolverDoubleComplex* A,
                                      int                     lda,
                                      hipsolverDoubleComplex* work)
{
    char side = hipsolver2char_side(sideR);
    zlarf_(&side, &m, &n, x, &incx, alpha, A, &lda, work);
}

// orgbr & ungbr
template <>
void cpu_orgbr_ungbr<float>(hipsolverSideMode_t side,
                            int                 m,
                            int                 n,
                            int                 k,
                            float*              A,
                            int                 lda,
                            float*              Ipiv,
                            float*              work,
                            int                 size_w,
                            int*                info)
{
    char vect;
    if(side == HIPSOLVER_SIDE_LEFT)
        vect = 'Q';
    else
        vect = 'P';
    sorgbr_(&vect, &m, &n, &k, A, &lda, Ipiv, work, &size_w, info);
}

template <>
void cpu_orgbr_ungbr<double>(hipsolverSideMode_t side,
                             int                 m,
                             int                 n,
                             int                 k,
                             double*             A,
                             int                 lda,
                             double*             Ipiv,
                             double*             work,
                             int                 size_w,
                             int*                info)
{
    char vect;
    if(side == HIPSOLVER_SIDE_LEFT)
        vect = 'Q';
    else
        vect = 'P';
    dorgbr_(&vect, &m, &n, &k, A, &lda, Ipiv, work, &size_w, info);
}

template <>
void cpu_orgbr_ungbr<hipsolverComplex>(hipsolverSideMode_t side,
                                       int                 m,
                                       int                 n,
                                       int                 k,
                                       hipsolverComplex*   A,
                                       int                 lda,
                                       hipsolverComplex*   Ipiv,
                                       hipsolverComplex*   work,
                                       int                 size_w,
                                       int*                info)
{
    char vect;
    if(side == HIPSOLVER_SIDE_LEFT)
        vect = 'Q';
    else
        vect = 'P';
    cungbr_(&vect, &m, &n, &k, A, &lda, Ipiv, work, &size_w, info);
}

template <>
void cpu_orgbr_ungbr<hipsolverDoubleComplex>(hipsolverSideMode_t     side,
                                             int                     m,
                                             int                     n,
                                             int                     k,
                                             hipsolverDoubleComplex* A,
                                             int                     lda,
                                             hipsolverDoubleComplex* Ipiv,
                                             hipsolverDoubleComplex* work,
                                             int                     size_w,
                                             int*                    info)
{
    char vect;
    if(side == HIPSOLVER_SIDE_LEFT)
        vect = 'Q';
    else
        vect = 'P';
    zungbr_(&vect, &m, &n, &k, A, &lda, Ipiv, work, &size_w, info);
}

// orgqr & ungqr
template <>
void cpu_orgqr_ungqr<float>(
    int m, int n, int k, float* A, int lda, float* ipiv, float* work, int lwork, int* info)
{
    sorgqr_(&m, &n, &k, A, &lda, ipiv, work, &lwork, info);
}

template <>
void cpu_orgqr_ungqr<double>(
    int m, int n, int k, double* A, int lda, double* ipiv, double* work, int lwork, int* info)
{
    dorgqr_(&m, &n, &k, A, &lda, ipiv, work, &lwork, info);
}

template <>
void cpu_orgqr_ungqr<hipsolverComplex>(int               m,
                                       int               n,
                                       int               k,
                                       hipsolverComplex* A,
                                       int               lda,
                                       hipsolverComplex* ipiv,
                                       hipsolverComplex* work,
                                       int               lwork,
                                       int*              info)
{
    cungqr_(&m, &n, &k, A, &lda, ipiv, work, &lwork, info);
}

template <>
void cpu_orgqr_ungqr<hipsolverDoubleComplex>(int                     m,
                                             int                     n,
                                             int                     k,
                                             hipsolverDoubleComplex* A,
                                             int                     lda,
                                             hipsolverDoubleComplex* ipiv,
                                             hipsolverDoubleComplex* work,
                                             int                     lwork,
                                             int*                    info)
{
    zungqr_(&m, &n, &k, A, &lda, ipiv, work, &lwork, info);
}

// orgtr & ungtr
template <>
void cpu_orgtr_ungtr<float>(hipsolverFillMode_t uplo,
                            int                 n,
                            float*              A,
                            int                 lda,
                            float*              Ipiv,
                            float*              work,
                            int                 size_w,
                            int*                info)
{
    char uploC = hipsolver2char_fill(uplo);
    sorgtr_(&uploC, &n, A, &lda, Ipiv, work, &size_w, info);
}

template <>
void cpu_orgtr_ungtr<double>(hipsolverFillMode_t uplo,
                             int                 n,
                             double*             A,
                             int                 lda,
                             double*             Ipiv,
                             double*             work,
                             int                 size_w,
                             int*                info)
{
    char uploC = hipsolver2char_fill(uplo);
    dorgtr_(&uploC, &n, A, &lda, Ipiv, work, &size_w, info);
}

template <>
void cpu_orgtr_ungtr<hipsolverComplex>(hipsolverFillMode_t uplo,
                                       int                 n,
                                       hipsolverComplex*   A,
                                       int                 lda,
                                       hipsolverComplex*   Ipiv,
                                       hipsolverComplex*   work,
                                       int                 size_w,
                                       int*                info)
{
    char uploC = hipsolver2char_fill(uplo);
    cungtr_(&uploC, &n, A, &lda, Ipiv, work, &size_w, info);
}

template <>
void cpu_orgtr_ungtr<hipsolverDoubleComplex>(hipsolverFillMode_t     uplo,
                                             int                     n,
                                             hipsolverDoubleComplex* A,
                                             int                     lda,
                                             hipsolverDoubleComplex* Ipiv,
                                             hipsolverDoubleComplex* work,
                                             int                     size_w,
                                             int*                    info)
{
    char uploC = hipsolver2char_fill(uplo);
    zungtr_(&uploC, &n, A, &lda, Ipiv, work, &size_w, info);
}

// ormqr & unmqr
template <>
void cpu_ormqr_unmqr<float>(hipsolverSideMode_t  side,
                            hipsolverOperation_t trans,
                            int                  m,
                            int                  n,
                            int                  k,
                            float*               A,
                            int                  lda,
                            float*               ipiv,
                            float*               C,
                            int                  ldc,
                            float*               work,
                            int                  lwork,
                            int*                 info)
{
    char sideC  = hipsolver2char_side(side);
    char transC = hipsolver2char_operation(trans);

    sormqr_(&sideC, &transC, &m, &n, &k, A, &lda, ipiv, C, &ldc, work, &lwork, info);
}

template <>
void cpu_ormqr_unmqr<double>(hipsolverSideMode_t  side,
                             hipsolverOperation_t trans,
                             int                  m,
                             int                  n,
                             int                  k,
                             double*              A,
                             int                  lda,
                             double*              ipiv,
                             double*              C,
                             int                  ldc,
                             double*              work,
                             int                  lwork,
                             int*                 info)
{
    char sideC  = hipsolver2char_side(side);
    char transC = hipsolver2char_operation(trans);

    dormqr_(&sideC, &transC, &m, &n, &k, A, &lda, ipiv, C, &ldc, work, &lwork, info);
}

template <>
void cpu_ormqr_unmqr<hipsolverComplex>(hipsolverSideMode_t  side,
                                       hipsolverOperation_t trans,
                                       int                  m,
                                       int                  n,
                                       int                  k,
                                       hipsolverComplex*    A,
                                       int                  lda,
                                       hipsolverComplex*    ipiv,
                                       hipsolverComplex*    C,
                                       int                  ldc,
                                       hipsolverComplex*    work,
                                       int                  lwork,
                                       int*                 info)
{
    char sideC  = hipsolver2char_side(side);
    char transC = hipsolver2char_operation(trans);

    cunmqr_(&sideC, &transC, &m, &n, &k, A, &lda, ipiv, C, &ldc, work, &lwork, info);
}

template <>
void cpu_ormqr_unmqr<hipsolverDoubleComplex>(hipsolverSideMode_t     side,
                                             hipsolverOperation_t    trans,
                                             int                     m,
                                             int                     n,
                                             int                     k,
                                             hipsolverDoubleComplex* A,
                                             int                     lda,
                                             hipsolverDoubleComplex* ipiv,
                                             hipsolverDoubleComplex* C,
                                             int                     ldc,
                                             hipsolverDoubleComplex* work,
                                             int                     lwork,
                                             int*                    info)
{
    char sideC  = hipsolver2char_side(side);
    char transC = hipsolver2char_operation(trans);

    zunmqr_(&sideC, &transC, &m, &n, &k, A, &lda, ipiv, C, &ldc, work, &lwork, info);
}

// ormtr & unmtr
template <>
void cpu_ormtr_unmtr<float>(hipsolverSideMode_t  side,
                            hipsolverFillMode_t  uplo,
                            hipsolverOperation_t trans,
                            int                  m,
                            int                  n,
                            float*               A,
                            int                  lda,
                            float*               ipiv,
                            float*               C,
                            int                  ldc,
                            float*               work,
                            int                  lwork,
                            int*                 info)
{
    char sideC  = hipsolver2char_side(side);
    char uploC  = hipsolver2char_fill(uplo);
    char transC = hipsolver2char_operation(trans);

    sormtr_(&sideC, &uploC, &transC, &m, &n, A, &lda, ipiv, C, &ldc, work, &lwork, info);
}

template <>
void cpu_ormtr_unmtr<double>(hipsolverSideMode_t  side,
                             hipsolverFillMode_t  uplo,
                             hipsolverOperation_t trans,
                             int                  m,
                             int                  n,
                             double*              A,
                             int                  lda,
                             double*              ipiv,
                             double*              C,
                             int                  ldc,
                             double*              work,
                             int                  lwork,
                             int*                 info)
{
    char sideC  = hipsolver2char_side(side);
    char uploC  = hipsolver2char_fill(uplo);
    char transC = hipsolver2char_operation(trans);

    dormtr_(&sideC, &uploC, &transC, &m, &n, A, &lda, ipiv, C, &ldc, work, &lwork, info);
}

template <>
void cpu_ormtr_unmtr<hipsolverComplex>(hipsolverSideMode_t  side,
                                       hipsolverFillMode_t  uplo,
                                       hipsolverOperation_t trans,
                                       int                  m,
                                       int                  n,
                                       hipsolverComplex*    A,
                                       int                  lda,
                                       hipsolverComplex*    ipiv,
                                       hipsolverComplex*    C,
                                       int                  ldc,
                                       hipsolverComplex*    work,
                                       int                  lwork,
                                       int*                 info)
{
    char sideC  = hipsolver2char_side(side);
    char uploC  = hipsolver2char_fill(uplo);
    char transC = hipsolver2char_operation(trans);

    cunmtr_(&sideC, &uploC, &transC, &m, &n, A, &lda, ipiv, C, &ldc, work, &lwork, info);
}

template <>
void cpu_ormtr_unmtr<hipsolverDoubleComplex>(hipsolverSideMode_t     side,
                                             hipsolverFillMode_t     uplo,
                                             hipsolverOperation_t    trans,
                                             int                     m,
                                             int                     n,
                                             hipsolverDoubleComplex* A,
                                             int                     lda,
                                             hipsolverDoubleComplex* ipiv,
                                             hipsolverDoubleComplex* C,
                                             int                     ldc,
                                             hipsolverDoubleComplex* work,
                                             int                     lwork,
                                             int*                    info)
{
    char sideC  = hipsolver2char_side(side);
    char uploC  = hipsolver2char_fill(uplo);
    char transC = hipsolver2char_operation(trans);

    zunmtr_(&sideC, &uploC, &transC, &m, &n, A, &lda, ipiv, C, &ldc, work, &lwork, info);
}

// gebrd
template <>
void cpu_gebrd<float, float>(int    m,
                             int    n,
                             float* A,
                             int    lda,
                             float* D,
                             float* E,
                             float* tauq,
                             float* taup,
                             float* work,
                             int    size_w,
                             int*   info)
{
    sgebrd_(&m, &n, A, &lda, D, E, tauq, taup, work, &size_w, info);
}

template <>
void cpu_gebrd<double, double>(int     m,
                               int     n,
                               double* A,
                               int     lda,
                               double* D,
                               double* E,
                               double* tauq,
                               double* taup,
                               double* work,
                               int     size_w,
                               int*    info)
{
    dgebrd_(&m, &n, A, &lda, D, E, tauq, taup, work, &size_w, info);
}

template <>
void cpu_gebrd<hipsolverComplex, float>(int               m,
                                        int               n,
                                        hipsolverComplex* A,
                                        int               lda,
                                        float*            D,
                                        float*            E,
                                        hipsolverComplex* tauq,
                                        hipsolverComplex* taup,
                                        hipsolverComplex* work,
                                        int               size_w,
                                        int*              info)
{
    cgebrd_(&m, &n, A, &lda, D, E, tauq, taup, work, &size_w, info);
}

template <>
void cpu_gebrd<hipsolverDoubleComplex, double>(int                     m,
                                               int                     n,
                                               hipsolverDoubleComplex* A,
                                               int                     lda,
                                               double*                 D,
                                               double*                 E,
                                               hipsolverDoubleComplex* tauq,
                                               hipsolverDoubleComplex* taup,
                                               hipsolverDoubleComplex* work,
                                               int                     size_w,
                                               int*                    info)
{
    zgebrd_(&m, &n, A, &lda, D, E, tauq, taup, work, &size_w, info);
}

// gels
template <>
void cpu_gels<float>(hipsolverOperation_t transR,
                     int                  m,
                     int                  n,
                     int                  nrhs,
                     float*               A,
                     int                  lda,
                     float*               B,
                     int                  ldb,
                     float*               work,
                     int                  lwork,
                     int*                 info)
{
    char trans = hipsolver2char_operation(transR);
    sgels_(&trans, &m, &n, &nrhs, A, &lda, B, &ldb, work, &lwork, info);
}

template <>
void cpu_gels<double>(hipsolverOperation_t transR,
                      int                  m,
                      int                  n,
                      int                  nrhs,
                      double*              A,
                      int                  lda,
                      double*              B,
                      int                  ldb,
                      double*              work,
                      int                  lwork,
                      int*                 info)
{
    char trans = hipsolver2char_operation(transR);
    dgels_(&trans, &m, &n, &nrhs, A, &lda, B, &ldb, work, &lwork, info);
}

template <>
void cpu_gels<hipsolverComplex>(hipsolverOperation_t transR,
                                int                  m,
                                int                  n,
                                int                  nrhs,
                                hipsolverComplex*    A,
                                int                  lda,
                                hipsolverComplex*    B,
                                int                  ldb,
                                hipsolverComplex*    work,
                                int                  lwork,
                                int*                 info)
{
    char trans = hipsolver2char_operation(transR);
    cgels_(&trans, &m, &n, &nrhs, A, &lda, B, &ldb, work, &lwork, info);
}

template <>
void cpu_gels<hipsolverDoubleComplex>(hipsolverOperation_t    transR,
                                      int                     m,
                                      int                     n,
                                      int                     nrhs,
                                      hipsolverDoubleComplex* A,
                                      int                     lda,
                                      hipsolverDoubleComplex* B,
                                      int                     ldb,
                                      hipsolverDoubleComplex* work,
                                      int                     lwork,
                                      int*                    info)
{
    char trans = hipsolver2char_operation(transR);
    zgels_(&trans, &m, &n, &nrhs, A, &lda, B, &ldb, work, &lwork, info);
}

// geqrf
template <>
void cpu_geqrf<float>(
    int m, int n, float* A, int lda, float* ipiv, float* work, int lwork, int* info)
{
    sgeqrf_(&m, &n, A, &lda, ipiv, work, &lwork, info);
}

template <>
void cpu_geqrf<double>(
    int m, int n, double* A, int lda, double* ipiv, double* work, int lwork, int* info)
{
    dgeqrf_(&m, &n, A, &lda, ipiv, work, &lwork, info);
}

template <>
void cpu_geqrf<hipsolverComplex>(int               m,
                                 int               n,
                                 hipsolverComplex* A,
                                 int               lda,
                                 hipsolverComplex* ipiv,
                                 hipsolverComplex* work,
                                 int               lwork,
                                 int*              info)
{
    cgeqrf_(&m, &n, A, &lda, ipiv, work, &lwork, info);
}

template <>
void cpu_geqrf<hipsolverDoubleComplex>(int                     m,
                                       int                     n,
                                       hipsolverDoubleComplex* A,
                                       int                     lda,
                                       hipsolverDoubleComplex* ipiv,
                                       hipsolverDoubleComplex* work,
                                       int                     lwork,
                                       int*                    info)
{
    zgeqrf_(&m, &n, A, &lda, ipiv, work, &lwork, info);
}

// gesv
template <>
void cpu_gesv<float>(int n, int nrhs, float* A, int lda, int* ipiv, float* B, int ldb, int* info)
{
    sgesv_(&n, &nrhs, A, &lda, ipiv, B, &ldb, info);
}

template <>
void cpu_gesv<double>(int n, int nrhs, double* A, int lda, int* ipiv, double* B, int ldb, int* info)
{
    dgesv_(&n, &nrhs, A, &lda, ipiv, B, &ldb, info);
}

template <>
void cpu_gesv<hipsolverComplex>(int               n,
                                int               nrhs,
                                hipsolverComplex* A,
                                int               lda,
                                int*              ipiv,
                                hipsolverComplex* B,
                                int               ldb,
                                int*              info)
{
    cgesv_(&n, &nrhs, A, &lda, ipiv, B, &ldb, info);
}

template <>
void cpu_gesv<hipsolverDoubleComplex>(int                     n,
                                      int                     nrhs,
                                      hipsolverDoubleComplex* A,
                                      int                     lda,
                                      int*                    ipiv,
                                      hipsolverDoubleComplex* B,
                                      int                     ldb,
                                      int*                    info)
{
    zgesv_(&n, &nrhs, A, &lda, ipiv, B, &ldb, info);
}

// gesvd
template <>
void cpu_gesvd(char   jobu,
               char   jobv,
               int    m,
               int    n,
               float* A,
               int    lda,
               float* S,
               float* U,
               int    ldu,
               float* V,
               int    ldv,
               float* work,
               int    lwork,
               float* E,
               int*   info)
{
    sgesvd_(&jobu, &jobv, &m, &n, A, &lda, S, U, &ldu, V, &ldv, E, &lwork, info);
}

template <>
void cpu_gesvd(char    jobu,
               char    jobv,
               int     m,
               int     n,
               double* A,
               int     lda,
               double* S,
               double* U,
               int     ldu,
               double* V,
               int     ldv,
               double* work,
               int     lwork,
               double* E,
               int*    info)
{
    dgesvd_(&jobu, &jobv, &m, &n, A, &lda, S, U, &ldu, V, &ldv, E, &lwork, info);
}

template <>
void cpu_gesvd(char              jobu,
               char              jobv,
               int               m,
               int               n,
               hipsolverComplex* A,
               int               lda,
               float*            S,
               hipsolverComplex* U,
               int               ldu,
               hipsolverComplex* V,
               int               ldv,
               hipsolverComplex* work,
               int               lwork,
               float*            E,
               int*              info)
{
    cgesvd_(&jobu, &jobv, &m, &n, A, &lda, S, U, &ldu, V, &ldv, work, &lwork, E, info);
}

template <>
void cpu_gesvd(char                    jobu,
               char                    jobv,
               int                     m,
               int                     n,
               hipsolverDoubleComplex* A,
               int                     lda,
               double*                 S,
               hipsolverDoubleComplex* U,
               int                     ldu,
               hipsolverDoubleComplex* V,
               int                     ldv,
               hipsolverDoubleComplex* work,
               int                     lwork,
               double*                 E,
               int*                    info)
{
    zgesvd_(&jobu, &jobv, &m, &n, A, &lda, S, U, &ldu, V, &ldv, work, &lwork, E, info);
}

// gesvdx
template <>
void cpu_gesvdx(hipsolverEigMode_t leftv,
                hipsolverEigMode_t rightv,
                char               srange,
                int                m,
                int                n,
                float*             A,
                int                lda,
                float              vl,
                float              vu,
                int                il,
                int                iu,
                int*               nsv,
                float*             S,
                float*             U,
                int                ldu,
                float*             V,
                int                ldv,
                float*             work,
                int                lwork,
                float*             rwork,
                int*               iwork,
                int*               info)
{
    char jobu = hipsolver2char_evect(leftv);
    char jobv = hipsolver2char_evect(rightv);
    sgesvdx_(&jobu,
             &jobv,
             &srange,
             &m,
             &n,
             A,
             &lda,
             &vl,
             &vu,
             &il,
             &iu,
             nsv,
             S,
             U,
             &ldu,
             V,
             &ldv,
             work,
             &lwork,
             iwork,
             info);
}

template <>
void cpu_gesvdx(hipsolverEigMode_t leftv,
                hipsolverEigMode_t rightv,
                char               srange,
                int                m,
                int                n,
                double*            A,
                int                lda,
                double             vl,
                double             vu,
                int                il,
                int                iu,
                int*               nsv,
                double*            S,
                double*            U,
                int                ldu,
                double*            V,
                int                ldv,
                double*            work,
                int                lwork,
                double*            rwork,
                int*               iwork,
                int*               info)
{
    char jobu = hipsolver2char_evect(leftv);
    char jobv = hipsolver2char_evect(rightv);
    dgesvdx_(&jobu,
             &jobv,
             &srange,
             &m,
             &n,
             A,
             &lda,
             &vl,
             &vu,
             &il,
             &iu,
             nsv,
             S,
             U,
             &ldu,
             V,
             &ldv,
             work,
             &lwork,
             iwork,
             info);
}

template <>
void cpu_gesvdx(hipsolverEigMode_t leftv,
                hipsolverEigMode_t rightv,
                char               srange,
                int                m,
                int                n,
                hipsolverComplex*  A,
                int                lda,
                float              vl,
                float              vu,
                int                il,
                int                iu,
                int*               nsv,
                float*             S,
                hipsolverComplex*  U,
                int                ldu,
                hipsolverComplex*  V,
                int                ldv,
                hipsolverComplex*  work,
                int                lwork,
                float*             rwork,
                int*               iwork,
                int*               info)
{
    char jobu = hipsolver2char_evect(leftv);
    char jobv = hipsolver2char_evect(rightv);
    cgesvdx_(&jobu,
             &jobv,
             &srange,
             &m,
             &n,
             A,
             &lda,
             &vl,
             &vu,
             &il,
             &iu,
             nsv,
             S,
             U,
             &ldu,
             V,
             &ldv,
             work,
             &lwork,
             rwork,
             iwork,
             info);
}

template <>
void cpu_gesvdx(hipsolverEigMode_t      leftv,
                hipsolverEigMode_t      rightv,
                char                    srange,
                int                     m,
                int                     n,
                hipsolverDoubleComplex* A,
                int                     lda,
                double                  vl,
                double                  vu,
                int                     il,
                int                     iu,
                int*                    nsv,
                double*                 S,
                hipsolverDoubleComplex* U,
                int                     ldu,
                hipsolverDoubleComplex* V,
                int                     ldv,
                hipsolverDoubleComplex* work,
                int                     lwork,
                double*                 rwork,
                int*                    iwork,
                int*                    info)
{
    char jobu = hipsolver2char_evect(leftv);
    char jobv = hipsolver2char_evect(rightv);
    zgesvdx_(&jobu,
             &jobv,
             &srange,
             &m,
             &n,
             A,
             &lda,
             &vl,
             &vu,
             &il,
             &iu,
             nsv,
             S,
             U,
             &ldu,
             V,
             &ldv,
             work,
             &lwork,
             rwork,
             iwork,
             info);
}

// getrf
template <>
void cpu_getrf<float>(int m, int n, float* A, int lda, int* ipiv, int* info)
{
    sgetrf_(&m, &n, A, &lda, ipiv, info);
}

template <>
void cpu_getrf<double>(int m, int n, double* A, int lda, int* ipiv, int* info)
{
    dgetrf_(&m, &n, A, &lda, ipiv, info);
}

template <>
void cpu_getrf<hipsolverComplex>(int m, int n, hipsolverComplex* A, int lda, int* ipiv, int* info)
{
    cgetrf_(&m, &n, A, &lda, ipiv, info);
}

template <>
void cpu_getrf<hipsolverDoubleComplex>(
    int m, int n, hipsolverDoubleComplex* A, int lda, int* ipiv, int* info)
{
    zgetrf_(&m, &n, A, &lda, ipiv, info);
}

// getrs
template <>
void cpu_getrs<float>(hipsolverOperation_t trans,
                      int                  n,
                      int                  nrhs,
                      float*               A,
                      int                  lda,
                      int*                 ipiv,
                      float*               B,
                      int                  ldb,
                      int*                 info)
{
    char transC = hipsolver2char_operation(trans);
    sgetrs_(&transC, &n, &nrhs, A, &lda, ipiv, B, &ldb, info);
}

template <>
void cpu_getrs<double>(hipsolverOperation_t trans,
                       int                  n,
                       int                  nrhs,
                       double*              A,
                       int                  lda,
                       int*                 ipiv,
                       double*              B,
                       int                  ldb,
                       int*                 info)
{
    char transC = hipsolver2char_operation(trans);
    dgetrs_(&transC, &n, &nrhs, A, &lda, ipiv, B, &ldb, info);
}

template <>
void cpu_getrs<hipsolverComplex>(hipsolverOperation_t trans,
                                 int                  n,
                                 int                  nrhs,
                                 hipsolverComplex*    A,
                                 int                  lda,
                                 int*                 ipiv,
                                 hipsolverComplex*    B,
                                 int                  ldb,
                                 int*                 info)
{
    char transC = hipsolver2char_operation(trans);
    cgetrs_(&transC, &n, &nrhs, A, &lda, ipiv, B, &ldb, info);
}

template <>
void cpu_getrs<hipsolverDoubleComplex>(hipsolverOperation_t    trans,
                                       int                     n,
                                       int                     nrhs,
                                       hipsolverDoubleComplex* A,
                                       int                     lda,
                                       int*                    ipiv,
                                       hipsolverDoubleComplex* B,
                                       int                     ldb,
                                       int*                    info)
{
    char transC = hipsolver2char_operation(trans);
    zgetrs_(&transC, &n, &nrhs, A, &lda, ipiv, B, &ldb, info);
}

// potrf
template <>
void cpu_potrf<float>(hipsolverFillMode_t uplo, int n, float* A, int lda, int* info)
{
    char uploC = hipsolver2char_fill(uplo);
    spotrf_(&uploC, &n, A, &lda, info);
}

template <>
void cpu_potrf<double>(hipsolverFillMode_t uplo, int n, double* A, int lda, int* info)
{
    char uploC = hipsolver2char_fill(uplo);
    dpotrf_(&uploC, &n, A, &lda, info);
}

template <>
void cpu_potrf<hipsolverComplex>(
    hipsolverFillMode_t uplo, int n, hipsolverComplex* A, int lda, int* info)
{
    char uploC = hipsolver2char_fill(uplo);
    cpotrf_(&uploC, &n, A, &lda, info);
}

template <>
void cpu_potrf<hipsolverDoubleComplex>(
    hipsolverFillMode_t uplo, int n, hipsolverDoubleComplex* A, int lda, int* info)
{
    char uploC = hipsolver2char_fill(uplo);
    zpotrf_(&uploC, &n, A, &lda, info);
}

// potri
template <>
void cpu_potri(hipsolverFillMode_t uplo, int n, float* A, int lda, int* info)
{
    char uploC = hipsolver2char_fill(uplo);
    spotri_(&uploC, &n, A, &lda, info);
}

template <>
void cpu_potri(hipsolverFillMode_t uplo, int n, double* A, int lda, int* info)
{
    char uploC = hipsolver2char_fill(uplo);
    dpotri_(&uploC, &n, A, &lda, info);
}

template <>
void cpu_potri(hipsolverFillMode_t uplo, int n, hipsolverComplex* A, int lda, int* info)
{
    char uploC = hipsolver2char_fill(uplo);
    cpotri_(&uploC, &n, A, &lda, info);
}

template <>
void cpu_potri(hipsolverFillMode_t uplo, int n, hipsolverDoubleComplex* A, int lda, int* info)
{
    char uploC = hipsolver2char_fill(uplo);
    zpotri_(&uploC, &n, A, &lda, info);
}

// potrs
template <>
void cpu_potrs(
    hipsolverFillMode_t uplo, int n, int nrhs, float* A, int lda, float* B, int ldb, int* info)
{
    char uploC = hipsolver2char_fill(uplo);
    spotrs_(&uploC, &n, &nrhs, A, &lda, B, &ldb, info);
}

template <>
void cpu_potrs(
    hipsolverFillMode_t uplo, int n, int nrhs, double* A, int lda, double* B, int ldb, int* info)
{
    char uploC = hipsolver2char_fill(uplo);
    dpotrs_(&uploC, &n, &nrhs, A, &lda, B, &ldb, info);
}

template <>
void cpu_potrs(hipsolverFillMode_t uplo,
               int                 n,
               int                 nrhs,
               hipsolverComplex*   A,
               int                 lda,
               hipsolverComplex*   B,
               int                 ldb,
               int*                info)
{
    char uploC = hipsolver2char_fill(uplo);
    cpotrs_(&uploC, &n, &nrhs, A, &lda, B, &ldb, info);
}

template <>
void cpu_potrs(hipsolverFillMode_t     uplo,
               int                     n,
               int                     nrhs,
               hipsolverDoubleComplex* A,
               int                     lda,
               hipsolverDoubleComplex* B,
               int                     ldb,
               int*                    info)
{
    char uploC = hipsolver2char_fill(uplo);
    zpotrs_(&uploC, &n, &nrhs, A, &lda, B, &ldb, info);
}

// syevd & heevd
template <>
void cpu_syevd_heevd<float, float>(hipsolverEigMode_t  evect,
                                   hipsolverFillMode_t uplo,
                                   int                 n,
                                   float*              A,
                                   int                 lda,
                                   float*              W,
                                   float*              work,
                                   int                 lwork,
                                   float*              rwork,
                                   int                 lrwork,
                                   int*                iwork,
                                   int                 liwork,
                                   int*                info)
{
    char evectC = hipsolver2char_evect(evect);
    char uploC  = hipsolver2char_fill(uplo);
    ssyevd_(&evectC, &uploC, &n, A, &lda, W, rwork, &lrwork, iwork, &liwork, info);
}

template <>
void cpu_syevd_heevd<double, double>(hipsolverEigMode_t  evect,
                                     hipsolverFillMode_t uplo,
                                     int                 n,
                                     double*             A,
                                     int                 lda,
                                     double*             W,
                                     double*             work,
                                     int                 lwork,
                                     double*             rwork,
                                     int                 lrwork,
                                     int*                iwork,
                                     int                 liwork,
                                     int*                info)
{
    char evectC = hipsolver2char_evect(evect);
    char uploC  = hipsolver2char_fill(uplo);
    dsyevd_(&evectC, &uploC, &n, A, &lda, W, rwork, &lrwork, iwork, &liwork, info);
}

template <>
void cpu_syevd_heevd<hipsolverComplex, float>(hipsolverEigMode_t  evect,
                                              hipsolverFillMode_t uplo,
                                              int                 n,
                                              hipsolverComplex*   A,
                                              int                 lda,
                                              float*              W,
                                              hipsolverComplex*   work,
                                              int                 lwork,
                                              float*              rwork,
                                              int                 lrwork,
                                              int*                iwork,
                                              int                 liwork,
                                              int*                info)
{
    char evectC = hipsolver2char_evect(evect);
    char uploC  = hipsolver2char_fill(uplo);
    cheevd_(&evectC, &uploC, &n, A, &lda, W, work, &lwork, rwork, &lrwork, iwork, &liwork, info);
}

template <>
void cpu_syevd_heevd<hipsolverDoubleComplex, double>(hipsolverEigMode_t      evect,
                                                     hipsolverFillMode_t     uplo,
                                                     int                     n,
                                                     hipsolverDoubleComplex* A,
                                                     int                     lda,
                                                     double*                 W,
                                                     hipsolverDoubleComplex* work,
                                                     int                     lwork,
                                                     double*                 rwork,
                                                     int                     lrwork,
                                                     int*                    iwork,
                                                     int                     liwork,
                                                     int*                    info)
{
    char evectC = hipsolver2char_evect(evect);
    char uploC  = hipsolver2char_fill(uplo);
    zheevd_(&evectC, &uploC, &n, A, &lda, W, work, &lwork, rwork, &lrwork, iwork, &liwork, info);
}

// syevx & heevx
template <>
void cpu_syevx_heevx<float, float>(hipsolverEigMode_t  evect,
                                   hipsolverEigRange_t erange,
                                   hipsolverFillMode_t uplo,
                                   int                 n,
                                   float*              A,
                                   int                 lda,
                                   float               vl,
                                   float               vu,
                                   int                 il,
                                   int                 iu,
                                   float               abstol,
                                   int*                nev,
                                   float*              W,
                                   float*              Z,
                                   int                 ldz,
                                   float*              work,
                                   int                 lwork,
                                   float*              rwork,
                                   int*                iwork,
                                   int*                ifail,
                                   int*                info)
{
    char evectC  = hipsolver2char_evect(evect);
    char erangeC = hipsolver2char_erange(erange);
    char uploC   = hipsolver2char_fill(uplo);
    ssyevx_(&evectC,
            &erangeC,
            &uploC,
            &n,
            A,
            &lda,
            &vl,
            &vu,
            &il,
            &iu,
            &abstol,
            nev,
            W,
            Z,
            &ldz,
            work,
            &lwork,
            iwork,
            ifail,
            info);
}

template <>
void cpu_syevx_heevx<double, double>(hipsolverEigMode_t  evect,
                                     hipsolverEigRange_t erange,
                                     hipsolverFillMode_t uplo,
                                     int                 n,
                                     double*             A,
                                     int                 lda,
                                     double              vl,
                                     double              vu,
                                     int                 il,
                                     int                 iu,
                                     double              abstol,
                                     int*                nev,
                                     double*             W,
                                     double*             Z,
                                     int                 ldz,
                                     double*             work,
                                     int                 lwork,
                                     double*             rwork,
                                     int*                iwork,
                                     int*                ifail,
                                     int*                info)
{
    char evectC  = hipsolver2char_evect(evect);
    char erangeC = hipsolver2char_erange(erange);
    char uploC   = hipsolver2char_fill(uplo);
    dsyevx_(&evectC,
            &erangeC,
            &uploC,
            &n,
            A,
            &lda,
            &vl,
            &vu,
            &il,
            &iu,
            &abstol,
            nev,
            W,
            Z,
            &ldz,
            work,
            &lwork,
            iwork,
            ifail,
            info);
}

template <>
void cpu_syevx_heevx<hipsolverComplex, float>(hipsolverEigMode_t  evect,
                                              hipsolverEigRange_t erange,
                                              hipsolverFillMode_t uplo,
                                              int                 n,
                                              hipsolverComplex*   A,
                                              int                 lda,
                                              float               vl,
                                              float               vu,
                                              int                 il,
                                              int                 iu,
                                              float               abstol,
                                              int*                nev,
                                              float*              W,
                                              hipsolverComplex*   Z,
                                              int                 ldz,
                                              hipsolverComplex*   work,
                                              int                 lwork,
                                              float*              rwork,
                                              int*                iwork,
                                              int*                ifail,
                                              int*                info)
{
    char evectC  = hipsolver2char_evect(evect);
    char erangeC = hipsolver2char_erange(erange);
    char uploC   = hipsolver2char_fill(uplo);
    cheevx_(&evectC,
            &erangeC,
            &uploC,
            &n,
            A,
            &lda,
            &vl,
            &vu,
            &il,
            &iu,
            &abstol,
            nev,
            W,
            Z,
            &ldz,
            work,
            &lwork,
            rwork,
            iwork,
            ifail,
            info);
}

template <>
void cpu_syevx_heevx<hipsolverDoubleComplex, double>(hipsolverEigMode_t      evect,
                                                     hipsolverEigRange_t     erange,
                                                     hipsolverFillMode_t     uplo,
                                                     int                     n,
                                                     hipsolverDoubleComplex* A,
                                                     int                     lda,
                                                     double                  vl,
                                                     double                  vu,
                                                     int                     il,
                                                     int                     iu,
                                                     double                  abstol,
                                                     int*                    nev,
                                                     double*                 W,
                                                     hipsolverDoubleComplex* Z,
                                                     int                     ldz,
                                                     hipsolverDoubleComplex* work,
                                                     int                     lwork,
                                                     double*                 rwork,
                                                     int*                    iwork,
                                                     int*                    ifail,
                                                     int*                    info)
{
    char evectC  = hipsolver2char_evect(evect);
    char erangeC = hipsolver2char_erange(erange);
    char uploC   = hipsolver2char_fill(uplo);
    zheevx_(&evectC,
            &erangeC,
            &uploC,
            &n,
            A,
            &lda,
            &vl,
            &vu,
            &il,
            &iu,
            &abstol,
            nev,
            W,
            Z,
            &ldz,
            work,
            &lwork,
            rwork,
            iwork,
            ifail,
            info);
}

// sygvd & hegvd
template <>
void cpu_sygvd_hegvd<float, float>(hipsolverEigType_t  itype,
                                   hipsolverEigMode_t  evect,
                                   hipsolverFillMode_t uplo,
                                   int                 n,
                                   float*              A,
                                   int                 lda,
                                   float*              B,
                                   int                 ldb,
                                   float*              W,
                                   float*              work,
                                   int                 lwork,
                                   float*              rwork,
                                   int                 lrwork,
                                   int*                iwork,
                                   int                 liwork,
                                   int*                info)
{
    int  itypeI = hipsolver2char_eform(itype) - '0';
    char evectC = hipsolver2char_evect(evect);
    char uploC  = hipsolver2char_fill(uplo);
    ssygvd_(
        &itypeI, &evectC, &uploC, &n, A, &lda, B, &ldb, W, rwork, &lrwork, iwork, &liwork, info);
}

template <>
void cpu_sygvd_hegvd<double, double>(hipsolverEigType_t  itype,
                                     hipsolverEigMode_t  evect,
                                     hipsolverFillMode_t uplo,
                                     int                 n,
                                     double*             A,
                                     int                 lda,
                                     double*             B,
                                     int                 ldb,
                                     double*             W,
                                     double*             work,
                                     int                 lwork,
                                     double*             rwork,
                                     int                 lrwork,
                                     int*                iwork,
                                     int                 liwork,
                                     int*                info)
{
    int  itypeI = hipsolver2char_eform(itype) - '0';
    char evectC = hipsolver2char_evect(evect);
    char uploC  = hipsolver2char_fill(uplo);
    dsygvd_(
        &itypeI, &evectC, &uploC, &n, A, &lda, B, &ldb, W, rwork, &lrwork, iwork, &liwork, info);
}

template <>
void cpu_sygvd_hegvd<hipsolverComplex, float>(hipsolverEigType_t  itype,
                                              hipsolverEigMode_t  evect,
                                              hipsolverFillMode_t uplo,
                                              int                 n,
                                              hipsolverComplex*   A,
                                              int                 lda,
                                              hipsolverComplex*   B,
                                              int                 ldb,
                                              float*              W,
                                              hipsolverComplex*   work,
                                              int                 lwork,
                                              float*              rwork,
                                              int                 lrwork,
                                              int*                iwork,
                                              int                 liwork,
                                              int*                info)
{
    int  itypeI = hipsolver2char_eform(itype) - '0';
    char evectC = hipsolver2char_evect(evect);
    char uploC  = hipsolver2char_fill(uplo);
    chegvd_(&itypeI,
            &evectC,
            &uploC,
            &n,
            A,
            &lda,
            B,
            &ldb,
            W,
            work,
            &lwork,
            rwork,
            &lrwork,
            iwork,
            &liwork,
            info);
}

template <>
void cpu_sygvd_hegvd<hipsolverDoubleComplex, double>(hipsolverEigType_t      itype,
                                                     hipsolverEigMode_t      evect,
                                                     hipsolverFillMode_t     uplo,
                                                     int                     n,
                                                     hipsolverDoubleComplex* A,
                                                     int                     lda,
                                                     hipsolverDoubleComplex* B,
                                                     int                     ldb,
                                                     double*                 W,
                                                     hipsolverDoubleComplex* work,
                                                     int                     lwork,
                                                     double*                 rwork,
                                                     int                     lrwork,
                                                     int*                    iwork,
                                                     int                     liwork,
                                                     int*                    info)
{
    int  itypeI = hipsolver2char_eform(itype) - '0';
    char evectC = hipsolver2char_evect(evect);
    char uploC  = hipsolver2char_fill(uplo);
    zhegvd_(&itypeI,
            &evectC,
            &uploC,
            &n,
            A,
            &lda,
            B,
            &ldb,
            W,
            work,
            &lwork,
            rwork,
            &lrwork,
            iwork,
            &liwork,
            info);
}

// sygvx & hegvx
template <>
void cpu_sygvx_hegvx<float, float>(hipsolverEigType_t  itype,
                                   hipsolverEigMode_t  evect,
                                   hipsolverEigRange_t erange,
                                   hipsolverFillMode_t uplo,
                                   int                 n,
                                   float*              A,
                                   int                 lda,
                                   float*              B,
                                   int                 ldb,
                                   float               vl,
                                   float               vu,
                                   int                 il,
                                   int                 iu,
                                   float               abstol,
                                   int*                nev,
                                   float*              W,
                                   float*              Z,
                                   int                 ldz,
                                   float*              work,
                                   int                 lwork,
                                   float*              rwork,
                                   int*                iwork,
                                   int*                ifail,
                                   int*                info)
{
    int  itypeI  = hipsolver2char_eform(itype) - '0';
    char evectC  = hipsolver2char_evect(evect);
    char erangeC = hipsolver2char_erange(erange);
    char uploC   = hipsolver2char_fill(uplo);
    ssygvx_(&itypeI,
            &evectC,
            &erangeC,
            &uploC,
            &n,
            A,
            &lda,
            B,
            &ldb,
            &vl,
            &vu,
            &il,
            &iu,
            &abstol,
            nev,
            W,
            Z,
            &ldz,
            work,
            &lwork,
            iwork,
            ifail,
            info);
}

template <>
void cpu_sygvx_hegvx<double, double>(hipsolverEigType_t  itype,
                                     hipsolverEigMode_t  evect,
                                     hipsolverEigRange_t erange,
                                     hipsolverFillMode_t uplo,
                                     int                 n,
                                     double*             A,
                                     int                 lda,
                                     double*             B,
                                     int                 ldb,
                                     double              vl,
                                     double              vu,
                                     int                 il,
                                     int                 iu,
                                     double              abstol,
                                     int*                nev,
                                     double*             W,
                                     double*             Z,
                                     int                 ldz,
                                     double*             work,
                                     int                 lwork,
                                     double*             rwork,
                                     int*                iwork,
                                     int*                ifail,
                                     int*                info)
{
    int  itypeI  = hipsolver2char_eform(itype) - '0';
    char evectC  = hipsolver2char_evect(evect);
    char erangeC = hipsolver2char_erange(erange);
    char uploC   = hipsolver2char_fill(uplo);
    dsygvx_(&itypeI,
            &evectC,
            &erangeC,
            &uploC,
            &n,
            A,
            &lda,
            B,
            &ldb,
            &vl,
            &vu,
            &il,
            &iu,
            &abstol,
            nev,
            W,
            Z,
            &ldz,
            work,
            &lwork,
            iwork,
            ifail,
            info);
}

template <>
void cpu_sygvx_hegvx<hipsolverComplex, float>(hipsolverEigType_t  itype,
                                              hipsolverEigMode_t  evect,
                                              hipsolverEigRange_t erange,
                                              hipsolverFillMode_t uplo,
                                              int                 n,
                                              hipsolverComplex*   A,
                                              int                 lda,
                                              hipsolverComplex*   B,
                                              int                 ldb,
                                              float               vl,
                                              float               vu,
                                              int                 il,
                                              int                 iu,
                                              float               abstol,
                                              int*                nev,
                                              float*              W,
                                              hipsolverComplex*   Z,
                                              int                 ldz,
                                              hipsolverComplex*   work,
                                              int                 lwork,
                                              float*              rwork,
                                              int*                iwork,
                                              int*                ifail,
                                              int*                info)
{
    int  itypeI  = hipsolver2char_eform(itype) - '0';
    char evectC  = hipsolver2char_evect(evect);
    char erangeC = hipsolver2char_erange(erange);
    char uploC   = hipsolver2char_fill(uplo);
    chegvx_(&itypeI,
            &evectC,
            &erangeC,
            &uploC,
            &n,
            A,
            &lda,
            B,
            &ldb,
            &vl,
            &vu,
            &il,
            &iu,
            &abstol,
            nev,
            W,
            Z,
            &ldz,
            work,
            &lwork,
            rwork,
            iwork,
            ifail,
            info);
}

template <>
void cpu_sygvx_hegvx<hipsolverDoubleComplex, double>(hipsolverEigType_t      itype,
                                                     hipsolverEigMode_t      evect,
                                                     hipsolverEigRange_t     erange,
                                                     hipsolverFillMode_t     uplo,
                                                     int                     n,
                                                     hipsolverDoubleComplex* A,
                                                     int                     lda,
                                                     hipsolverDoubleComplex* B,
                                                     int                     ldb,
                                                     double                  vl,
                                                     double                  vu,
                                                     int                     il,
                                                     int                     iu,
                                                     double                  abstol,
                                                     int*                    nev,
                                                     double*                 W,
                                                     hipsolverDoubleComplex* Z,
                                                     int                     ldz,
                                                     hipsolverDoubleComplex* work,
                                                     int                     lwork,
                                                     double*                 rwork,
                                                     int*                    iwork,
                                                     int*                    ifail,
                                                     int*                    info)
{
    int  itypeI  = hipsolver2char_eform(itype) - '0';
    char evectC  = hipsolver2char_evect(evect);
    char erangeC = hipsolver2char_erange(erange);
    char uploC   = hipsolver2char_fill(uplo);
    zhegvx_(&itypeI,
            &evectC,
            &erangeC,
            &uploC,
            &n,
            A,
            &lda,
            B,
            &ldb,
            &vl,
            &vu,
            &il,
            &iu,
            &abstol,
            nev,
            W,
            Z,
            &ldz,
            work,
            &lwork,
            rwork,
            iwork,
            ifail,
            info);
}

// sytrd & hetrd
template <>
void cpu_sytrd_hetrd<float, float>(hipsolverFillMode_t uplo,
                                   int                 n,
                                   float*              A,
                                   int                 lda,
                                   float*              D,
                                   float*              E,
                                   float*              tau,
                                   float*              work,
                                   int                 size_w)
{
    int  info;
    char uploC = hipsolver2char_fill(uplo);
    ssytrd_(&uploC, &n, A, &lda, D, E, tau, work, &size_w, &info);
}

template <>
void cpu_sytrd_hetrd<double, double>(hipsolverFillMode_t uplo,
                                     int                 n,
                                     double*             A,
                                     int                 lda,
                                     double*             D,
                                     double*             E,
                                     double*             tau,
                                     double*             work,
                                     int                 size_w)
{
    int  info;
    char uploC = hipsolver2char_fill(uplo);
    dsytrd_(&uploC, &n, A, &lda, D, E, tau, work, &size_w, &info);
}

template <>
void cpu_sytrd_hetrd<hipsolverComplex, float>(hipsolverFillMode_t uplo,
                                              int                 n,
                                              hipsolverComplex*   A,
                                              int                 lda,
                                              float*              D,
                                              float*              E,
                                              hipsolverComplex*   tau,
                                              hipsolverComplex*   work,
                                              int                 size_w)
{
    int  info;
    char uploC = hipsolver2char_fill(uplo);
    chetrd_(&uploC, &n, A, &lda, D, E, tau, work, &size_w, &info);
}

template <>
void cpu_sytrd_hetrd<hipsolverDoubleComplex, double>(hipsolverFillMode_t     uplo,
                                                     int                     n,
                                                     hipsolverDoubleComplex* A,
                                                     int                     lda,
                                                     double*                 D,
                                                     double*                 E,
                                                     hipsolverDoubleComplex* tau,
                                                     hipsolverDoubleComplex* work,
                                                     int                     size_w)
{
    int  info;
    char uploC = hipsolver2char_fill(uplo);
    zhetrd_(&uploC, &n, A, &lda, D, E, tau, work, &size_w, &info);
}

// sytrf
template <>
void cpu_sytrf<float>(hipsolverFillMode_t uplo,
                      int                 n,
                      float*              A,
                      int                 lda,
                      int*                ipiv,
                      float*              work,
                      int                 lwork,
                      int*                info)
{
    char uploC = hipsolver2char_fill(uplo);
    ssytrf_(&uploC, &n, A, &lda, ipiv, work, &lwork, info);
}

template <>
void cpu_sytrf<double>(hipsolverFillMode_t uplo,
                       int                 n,
                       double*             A,
                       int                 lda,
                       int*                ipiv,
                       double*             work,
                       int                 lwork,
                       int*                info)
{
    char uploC = hipsolver2char_fill(uplo);
    dsytrf_(&uploC, &n, A, &lda, ipiv, work, &lwork, info);
}

template <>
void cpu_sytrf<hipsolverComplex>(hipsolverFillMode_t uplo,
                                 int                 n,
                                 hipsolverComplex*   A,
                                 int                 lda,
                                 int*                ipiv,
                                 hipsolverComplex*   work,
                                 int                 lwork,
                                 int*                info)
{
    char uploC = hipsolver2char_fill(uplo);
    csytrf_(&uploC, &n, A, &lda, ipiv, work, &lwork, info);
}

template <>
void cpu_sytrf<hipsolverDoubleComplex>(hipsolverFillMode_t     uplo,
                                       int                     n,
                                       hipsolverDoubleComplex* A,
                                       int                     lda,
                                       int*                    ipiv,
                                       hipsolverDoubleComplex* work,
                                       int                     lwork,
                                       int*                    info)
{
    char uploC = hipsolver2char_fill(uplo);
    zsytrf_(&uploC, &n, A, &lda, ipiv, work, &lwork, info);
}
