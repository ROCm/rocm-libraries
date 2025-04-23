/* **************************************************************************
 * Copyright (C) 2016-2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 * *************************************************************************/

#include <rocblas/rocblas.h>

#include "lapack_host_reference.hpp"

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

float slange_(char* norm, int* m, int* n, float* A, int* lda, float* work);
double dlange_(char* norm, int* m, int* n, double* A, int* lda, double* work);
float clange_(char* norm, int* m, int* n, rocblas_float_complex* A, int* lda, float* work);
double zlange_(char* norm, int* m, int* n, rocblas_double_complex* A, int* lda, double* work);

void sgecon_(char* norm,
             int* n,
             float* A,
             int* lda,
             float* anorm,
             float* rcond,
             float* work,
             int* iwork,
             int* info);
void dgecon_(char* norm,
             int* n,
             double* A,
             int* lda,
             double* anorm,
             double* rcond,
             double* work,
             int* iwork,
             int* info);
void cgecon_(char* norm,
             int* n,
             rocblas_float_complex* A,
             int* lda,
             float* anorm,
             float* rcond,
             rocblas_float_complex* work,
             float* rwork,
             int* info);
void zgecon_(char* norm,
             int* n,
             rocblas_double_complex* A,
             int* lda,
             double* anorm,
             double* rcond,
             rocblas_double_complex* work,
             double* rwork,
             int* info);

void saxpy_(int* n, float* alpha, float* x, int* incx, float* y, int* incy);
void daxpy_(int* n, double* alpha, double* x, int* incx, double* y, int* incy);
void caxpy_(int* n,
            rocblas_float_complex* alpha,
            rocblas_float_complex* x,
            int* incx,
            rocblas_float_complex* y,
            int* incy);
void zaxpy_(int* n,
            rocblas_double_complex* alpha,
            rocblas_double_complex* x,
            int* incx,
            rocblas_double_complex* y,
            int* incy);

void sgemv_(char* transA,
            int* m,
            int* n,
            float* alpha,
            float* A,
            int* lda,
            float* x,
            int* incx,
            float* beta,
            float* y,
            int* incy);
void dgemv_(char* transA,
            int* m,
            int* n,
            double* alpha,
            double* A,
            int* lda,
            double* x,
            int* incx,
            double* beta,
            double* y,
            int* incy);
void cgemv_(char* transA,
            int* m,
            int* n,
            rocblas_float_complex* alpha,
            rocblas_float_complex* A,
            int* lda,
            rocblas_float_complex* x,
            int* incx,
            rocblas_float_complex* beta,
            rocblas_float_complex* y,
            int* incy);
void zgemv_(char* transA,
            int* m,
            int* n,
            rocblas_double_complex* alpha,
            rocblas_double_complex* A,
            int* lda,
            rocblas_double_complex* x,
            int* incx,
            rocblas_double_complex* beta,
            rocblas_double_complex* y,
            int* incy);

void sgemm_(char* transA,
            char* transB,
            int* m,
            int* n,
            int* k,
            float* alpha,
            float* A,
            int* lda,
            float* B,
            int* ldb,
            float* beta,
            float* C,
            int* ldc);
void dgemm_(char* transA,
            char* transB,
            int* m,
            int* n,
            int* k,
            double* alpha,
            double* A,
            int* lda,
            double* B,
            int* ldb,
            double* beta,
            double* C,
            int* ldc);
void cgemm_(char* transA,
            char* transB,
            int* m,
            int* n,
            int* k,
            rocblas_float_complex* alpha,
            rocblas_float_complex* A,
            int* lda,
            rocblas_float_complex* B,
            int* ldb,
            rocblas_float_complex* beta,
            rocblas_float_complex* C,
            int* ldc);
void zgemm_(char* transA,
            char* transB,
            int* m,
            int* n,
            int* k,
            rocblas_double_complex* alpha,
            rocblas_double_complex* A,
            int* lda,
            rocblas_double_complex* B,
            int* ldb,
            rocblas_double_complex* beta,
            rocblas_double_complex* C,
            int* ldc);

void ssymv_(char* uplo,
            int* n,
            float* alpha,
            float* A,
            int* lda,
            float* x,
            int* incx,
            float* beta,
            float* y,
            int* incy);
void dsymv_(char* uplo,
            int* n,
            double* alpha,
            double* A,
            int* lda,
            double* x,
            int* incx,
            double* beta,
            double* y,
            int* incy);
void chemv_(char* uplo,
            int* n,
            rocblas_float_complex* alpha,
            rocblas_float_complex* A,
            int* lda,
            rocblas_float_complex* x,
            int* incx,
            rocblas_float_complex* beta,
            rocblas_float_complex* y,
            int* incy);
void zhemv_(char* uplo,
            int* n,
            rocblas_double_complex* alpha,
            rocblas_double_complex* A,
            int* lda,
            rocblas_double_complex* x,
            int* incx,
            rocblas_double_complex* beta,
            rocblas_double_complex* y,
            int* incy);

void ssymm_(char* side,
            char* uplo,
            int* m,
            int* n,
            float* alpha,
            float* A,
            int* lda,
            float* B,
            int* ldb,
            float* beta,
            float* C,
            int* ldc);
void dsymm_(char* side,
            char* uplo,
            int* m,
            int* n,
            double* alpha,
            double* A,
            int* lda,
            double* B,
            int* ldb,
            double* beta,
            double* C,
            int* ldc);
void chemm_(char* side,
            char* uplo,
            int* m,
            int* n,
            rocblas_float_complex* alpha,
            rocblas_float_complex* A,
            int* lda,
            rocblas_float_complex* B,
            int* ldb,
            rocblas_float_complex* beta,
            rocblas_float_complex* C,
            int* ldc);
void zhemm_(char* side,
            char* uplo,
            int* m,
            int* n,
            rocblas_double_complex* alpha,
            rocblas_double_complex* A,
            int* lda,
            rocblas_double_complex* B,
            int* ldb,
            rocblas_double_complex* beta,
            rocblas_double_complex* C,
            int* ldc);

void strmm_(char* side,
            char* uplo,
            char* transA,
            char* diag,
            int* m,
            int* n,
            float* alpha,
            float* A,
            int* lda,
            float* B,
            int* ldb);
void dtrmm_(char* side,
            char* uplo,
            char* transA,
            char* diag,
            int* m,
            int* n,
            double* alpha,
            double* A,
            int* lda,
            double* B,
            int* ldb);
void ctrmm_(char* side,
            char* uplo,
            char* transA,
            char* diag,
            int* m,
            int* n,
            rocblas_float_complex* alpha,
            rocblas_float_complex* A,
            int* lda,
            rocblas_float_complex* B,
            int* ldb);
void ztrmm_(char* side,
            char* uplo,
            char* transA,
            char* diag,
            int* m,
            int* n,
            rocblas_double_complex* alpha,
            rocblas_double_complex* A,
            int* lda,
            rocblas_double_complex* B,
            int* ldb);

void strsm_(char* side,
            char* uplo,
            char* transA,
            char* diag,
            int* m,
            int* n,
            float* alpha,
            float* A,
            int* lda,
            float* B,
            int* ldb);
void dtrsm_(char* side,
            char* uplo,
            char* transA,
            char* diag,
            int* m,
            int* n,
            double* alpha,
            double* A,
            int* lda,
            double* B,
            int* ldb);
void ctrsm_(char* side,
            char* uplo,
            char* transA,
            char* diag,
            int* m,
            int* n,
            rocblas_float_complex* alpha,
            rocblas_float_complex* A,
            int* lda,
            rocblas_float_complex* B,
            int* ldb);
void ztrsm_(char* side,
            char* uplo,
            char* transA,
            char* diag,
            int* m,
            int* n,
            rocblas_double_complex* alpha,
            rocblas_double_complex* A,
            int* lda,
            rocblas_double_complex* B,
            int* ldb);

void strsv_(char* uplo, char* transA, char* diag, int* n, float* A, int* lda, float* x, int* incx);
void dtrsv_(char* uplo, char* transA, char* diag, int* n, double* A, int* lda, double* x, int* incx);
void ctrsv_(char* uplo,
            char* transA,
            char* diag,
            int* n,
            rocblas_float_complex* A,
            int* lda,
            rocblas_float_complex* x,
            int* incx);
void ztrsv_(char* uplo,
            char* transA,
            char* diag,
            int* n,
            rocblas_double_complex* A,
            int* lda,
            rocblas_double_complex* x,
            int* incx);

void strtri_(char* uplo, char* diag, int* n, float* A, int* lda, int* info);
void dtrtri_(char* uplo, char* diag, int* n, double* A, int* lda, int* info);
void ctrtri_(char* uplo, char* diag, int* n, rocblas_float_complex* A, int* lda, int* info);
void ztrtri_(char* uplo, char* diag, int* n, rocblas_double_complex* A, int* lda, int* info);

void sgetrf_(int* m, int* n, float* A, int* lda, int* ipiv, int* info);
void dgetrf_(int* m, int* n, double* A, int* lda, int* ipiv, int* info);
void cgetrf_(int* m, int* n, rocblas_float_complex* A, int* lda, int* ipiv, int* info);
void zgetrf_(int* m, int* n, rocblas_double_complex* A, int* lda, int* ipiv, int* info);

void spotf2_(char* uplo, int* n, float* A, int* lda, int* info);
void dpotf2_(char* uplo, int* n, double* A, int* lda, int* info);
void cpotf2_(char* uplo, int* n, rocblas_float_complex* A, int* lda, int* info);
void zpotf2_(char* uplo, int* n, rocblas_double_complex* A, int* lda, int* info);

void spotrf_(char* uplo, int* n, float* A, int* lda, int* info);
void dpotrf_(char* uplo, int* n, double* A, int* lda, int* info);
void cpotrf_(char* uplo, int* n, rocblas_float_complex* A, int* lda, int* info);
void zpotrf_(char* uplo, int* n, rocblas_double_complex* A, int* lda, int* info);

void spotrs_(char* uplo, int* n, int* nrhs, float* A, int* lda, float* B, int* ldb, int* info);
void dpotrs_(char* uplo, int* n, int* nrhs, double* A, int* lda, double* B, int* ldb, int* info);
void cpotrs_(char* uplo,
             int* n,
             int* nrhs,
             rocblas_float_complex* A,
             int* lda,
             rocblas_float_complex* B,
             int* ldb,
             int* info);
void zpotrs_(char* uplo,
             int* n,
             int* nrhs,
             rocblas_double_complex* A,
             int* lda,
             rocblas_double_complex* B,
             int* ldb,
             int* info);

void sposv_(char* uplo, int* n, int* nrhs, float* A, int* lda, float* B, int* ldb, int* info);
void dposv_(char* uplo, int* n, int* nrhs, double* A, int* lda, double* B, int* ldb, int* info);
void cposv_(char* uplo,
            int* n,
            int* nrhs,
            rocblas_float_complex* A,
            int* lda,
            rocblas_float_complex* B,
            int* ldb,
            int* info);
void zposv_(char* uplo,
            int* n,
            int* nrhs,
            rocblas_double_complex* A,
            int* lda,
            rocblas_double_complex* B,
            int* ldb,
            int* info);

void spotri_(char* uplo, int* n, float* A, int* lda, int* info);
void dpotri_(char* uplo, int* n, double* A, int* lda, int* info);
void cpotri_(char* uplo, int* n, rocblas_float_complex* A, int* lda, int* info);
void zpotri_(char* uplo, int* n, rocblas_double_complex* A, int* lda, int* info);

void sgetf2_(int* m, int* n, float* A, int* lda, int* ipiv, int* info);
void dgetf2_(int* m, int* n, double* A, int* lda, int* ipiv, int* info);
void cgetf2_(int* m, int* n, rocblas_float_complex* A, int* lda, int* ipiv, int* info);
void zgetf2_(int* m, int* n, rocblas_double_complex* A, int* lda, int* ipiv, int* info);

void sgetrs_(char* trans, int* n, int* nrhs, float* A, int* lda, int* ipiv, float* B, int* ldb, int* info);
void dgetrs_(char* trans, int* n, int* nrhs, double* A, int* lda, int* ipiv, double* B, int* ldb, int* info);
void cgetrs_(char* trans,
             int* n,
             int* nrhs,
             rocblas_float_complex* A,
             int* lda,
             int* ipiv,
             rocblas_float_complex* B,
             int* ldb,
             int* info);
void zgetrs_(char* trans,
             int* n,
             int* nrhs,
             rocblas_double_complex* A,
             int* lda,
             int* ipiv,
             rocblas_double_complex* B,
             int* ldb,
             int* info);

void sgesv_(int* n, int* nrhs, float* A, int* lda, int* ipiv, float* B, int* ldb, int* info);
void dgesv_(int* n, int* nrhs, double* A, int* lda, int* ipiv, double* B, int* ldb, int* info);
void cgesv_(int* n,
            int* nrhs,
            rocblas_float_complex* A,
            int* lda,
            int* ipiv,
            rocblas_float_complex* B,
            int* ldb,
            int* info);
void zgesv_(int* n,
            int* nrhs,
            rocblas_double_complex* A,
            int* lda,
            int* ipiv,
            rocblas_double_complex* B,
            int* ldb,
            int* info);

void sgels_(char* trans,
            int* m,
            int* n,
            int* nrhs,
            float* A,
            int* lda,
            float* B,
            int* ldb,
            float* work,
            int* lwork,
            int* info);
void dgels_(char* trans,
            int* m,
            int* n,
            int* nrhs,
            double* A,
            int* lda,
            double* B,
            int* ldb,
            double* work,
            int* lwork,
            int* info);
void cgels_(char* trans,
            int* m,
            int* n,
            int* nrhs,
            rocblas_float_complex* A,
            int* lda,
            rocblas_float_complex* B,
            int* ldb,
            rocblas_float_complex* work,
            int* lwork,
            int* info);
void zgels_(char* trans,
            int* m,
            int* n,
            int* nrhs,
            rocblas_double_complex* A,
            int* lda,
            rocblas_double_complex* B,
            int* ldb,
            rocblas_double_complex* work,
            int* lwork,
            int* info);

void sgetri_(int* n, float* A, int* lda, int* ipiv, float* work, int* lwork, int* info);
void dgetri_(int* n, double* A, int* lda, int* ipiv, double* work, int* lwork, int* info);
void cgetri_(int* n,
             rocblas_float_complex* A,
             int* lda,
             int* ipiv,
             rocblas_float_complex* work,
             int* lwork,
             int* info);
void zgetri_(int* n,
             rocblas_double_complex* A,
             int* lda,
             int* ipiv,
             rocblas_double_complex* work,
             int* lwork,
             int* info);

void strtri_(char* uplo, char* diag, int* n, float* A, int* lda, int* info);
void dtrtri_(char* uplo, char* diag, int* n, double* A, int* lda, int* info);
void ctrtri_(char* uplo, char* diag, int* n, rocblas_float_complex* A, int* lda, int* info);
void ztrtri_(char* uplo, char* diag, int* n, rocblas_double_complex* A, int* lda, int* info);

void slarfg_(int* n, float* alpha, float* x, int* incx, float* tau);
void dlarfg_(int* n, double* alpha, double* x, int* incx, double* tau);
void clarfg_(int* n,
             rocblas_float_complex* alpha,
             rocblas_float_complex* x,
             int* incx,
             rocblas_float_complex* tau);
void zlarfg_(int* n,
             rocblas_double_complex* alpha,
             rocblas_double_complex* x,
             int* incx,
             rocblas_double_complex* tau);

void slarf_(char* side, int* m, int* n, float* x, int* incx, float* alpha, float* A, int* lda, float* work);
void dlarf_(char* side,
            int* m,
            int* n,
            double* x,
            int* incx,
            double* alpha,
            double* A,
            int* lda,
            double* work);
void clarf_(char* side,
            int* m,
            int* n,
            rocblas_float_complex* x,
            int* incx,
            rocblas_float_complex* alpha,
            rocblas_float_complex* A,
            int* lda,
            rocblas_float_complex* work);
void zlarf_(char* side,
            int* m,
            int* n,
            rocblas_double_complex* x,
            int* incx,
            rocblas_double_complex* alpha,
            rocblas_double_complex* A,
            int* lda,
            rocblas_double_complex* work);

void slarft_(char* direct, char* storev, int* n, int* k, float* V, int* ldv, float* tau, float* T, int* ldt);
void dlarft_(char* direct,
             char* storev,
             int* n,
             int* k,
             double* V,
             int* ldv,
             double* tau,
             double* T,
             int* ldt);
void clarft_(char* direct,
             char* storev,
             int* n,
             int* k,
             rocblas_float_complex* V,
             int* ldv,
             rocblas_float_complex* tau,
             rocblas_float_complex* T,
             int* ldt);
void zlarft_(char* direct,
             char* storev,
             int* n,
             int* k,
             rocblas_double_complex* V,
             int* ldv,
             rocblas_double_complex* tau,
             rocblas_double_complex* T,
             int* ldt);

void sbdsqr_(char* uplo,
             int* n,
             int* nv,
             int* nu,
             int* nc,
             float* D,
             float* E,
             float* V,
             int* ldv,
             float* U,
             int* ldu,
             float* C,
             int* ldc,
             float* W,
             int* info);
void dbdsqr_(char* uplo,
             int* n,
             int* nv,
             int* nu,
             int* nc,
             double* D,
             double* E,
             double* V,
             int* ldv,
             double* U,
             int* ldu,
             double* C,
             int* ldc,
             double* W,
             int* info);
void cbdsqr_(char* uplo,
             int* n,
             int* nv,
             int* nu,
             int* nc,
             float* D,
             float* E,
             rocblas_float_complex* V,
             int* ldv,
             rocblas_float_complex* U,
             int* ldu,
             rocblas_float_complex* C,
             int* ldc,
             float* W,
             int* info);
void zbdsqr_(char* uplo,
             int* n,
             int* nv,
             int* nu,
             int* nc,
             double* D,
             double* E,
             rocblas_double_complex* V,
             int* ldv,
             rocblas_double_complex* U,
             int* ldu,
             rocblas_double_complex* C,
             int* ldc,
             double* W,
             int* info);

void slarfb_(char* side,
             char* trans,
             char* direct,
             char* storev,
             int* m,
             int* n,
             int* k,
             float* V,
             int* ldv,
             float* T,
             int* ldt,
             float* A,
             int* lda,
             float* W,
             int* ldw);
void dlarfb_(char* side,
             char* trans,
             char* direct,
             char* storev,
             int* m,
             int* n,
             int* k,
             double* V,
             int* ldv,
             double* T,
             int* ldt,
             double* A,
             int* lda,
             double* W,
             int* ldw);
void clarfb_(char* side,
             char* trans,
             char* direct,
             char* storev,
             int* m,
             int* n,
             int* k,
             rocblas_float_complex* V,
             int* ldv,
             rocblas_float_complex* T,
             int* ldt,
             rocblas_float_complex* A,
             int* lda,
             rocblas_float_complex* W,
             int* ldw);
void zlarfb_(char* side,
             char* trans,
             char* direct,
             char* storev,
             int* m,
             int* n,
             int* k,
             rocblas_double_complex* V,
             int* ldv,
             rocblas_double_complex* T,
             int* ldt,
             rocblas_double_complex* A,
             int* lda,
             rocblas_double_complex* W,
             int* ldw);

void slasr_(char* side, char* pivot, char* direct, int* m, int* n, float* C, float* S, float* A, int* lda);
void dlasr_(char* side, char* pivot, char* direct, int* m, int* n, double* C, double* S, double* A, int* lda);
void clasr_(char* side,
            char* pivot,
            char* direct,
            int* m,
            int* n,
            float* C,
            float* S,
            rocblas_float_complex* A,
            int* lda);
void zlasr_(char* side,
            char* pivot,
            char* direct,
            int* m,
            int* n,
            double* C,
            double* S,
            rocblas_double_complex* A,
            int* lda);

void slatrd_(char* uplo, int* n, int* k, float* A, int* lda, float* E, float* tau, float* W, int* ldw);
void dlatrd_(char* uplo, int* n, int* k, double* A, int* lda, double* E, double* tau, double* W, int* ldw);
void clatrd_(char* uplo,
             int* n,
             int* k,
             rocblas_float_complex* A,
             int* lda,
             float* E,
             rocblas_float_complex* tau,
             rocblas_float_complex* W,
             int* ldw);
void zlatrd_(char* uplo,
             int* n,
             int* k,
             rocblas_double_complex* A,
             int* lda,
             double* E,
             rocblas_double_complex* tau,
             rocblas_double_complex* W,
             int* ldw);

void slabrd_(int* m,
             int* n,
             int* nb,
             float* A,
             int* lda,
             float* D,
             float* E,
             float* tauq,
             float* taup,
             float* X,
             int* ldx,
             float* Y,
             int* ldy);
void dlabrd_(int* m,
             int* n,
             int* nb,
             double* A,
             int* lda,
             double* D,
             double* E,
             double* tauq,
             double* taup,
             double* X,
             int* ldx,
             double* Y,
             int* ldy);
void clabrd_(int* m,
             int* n,
             int* nb,
             rocblas_float_complex* A,
             int* lda,
             float* D,
             float* E,
             rocblas_float_complex* tauq,
             rocblas_float_complex* taup,
             rocblas_float_complex* X,
             int* ldx,
             rocblas_float_complex* Y,
             int* ldy);
void zlabrd_(int* m,
             int* n,
             int* nb,
             rocblas_double_complex* A,
             int* lda,
             double* D,
             double* E,
             rocblas_double_complex* tauq,
             rocblas_double_complex* taup,
             rocblas_double_complex* X,
             int* ldx,
             rocblas_double_complex* Y,
             int* ldy);

void sgeqr2_(int* m, int* n, float* A, int* lda, float* ipiv, float* work, int* info);
void dgeqr2_(int* m, int* n, double* A, int* lda, double* ipiv, double* work, int* info);
void cgeqr2_(int* m,
             int* n,
             rocblas_float_complex* A,
             int* lda,
             rocblas_float_complex* ipiv,
             rocblas_float_complex* work,
             int* info);
void zgeqr2_(int* m,
             int* n,
             rocblas_double_complex* A,
             int* lda,
             rocblas_double_complex* ipiv,
             rocblas_double_complex* work,
             int* info);

void sgeqrf_(int* m, int* n, float* A, int* lda, float* ipiv, float* work, int* lwork, int* info);
void dgeqrf_(int* m, int* n, double* A, int* lda, double* ipiv, double* work, int* lwork, int* info);
void cgeqrf_(int* m,
             int* n,
             rocblas_float_complex* A,
             int* lda,
             rocblas_float_complex* ipiv,
             rocblas_float_complex* work,
             int* lwork,
             int* info);
void zgeqrf_(int* m,
             int* n,
             rocblas_double_complex* A,
             int* lda,
             rocblas_double_complex* ipiv,
             rocblas_double_complex* work,
             int* lwork,
             int* info);

void sgerq2_(int* m, int* n, float* A, int* lda, float* ipiv, float* work, int* info);
void dgerq2_(int* m, int* n, double* A, int* lda, double* ipiv, double* work, int* info);
void cgerq2_(int* m,
             int* n,
             rocblas_float_complex* A,
             int* lda,
             rocblas_float_complex* ipiv,
             rocblas_float_complex* work,
             int* info);
void zgerq2_(int* m,
             int* n,
             rocblas_double_complex* A,
             int* lda,
             rocblas_double_complex* ipiv,
             rocblas_double_complex* work,
             int* info);

void sgerqf_(int* m, int* n, float* A, int* lda, float* ipiv, float* work, int* lwork, int* info);
void dgerqf_(int* m, int* n, double* A, int* lda, double* ipiv, double* work, int* lwork, int* info);
void cgerqf_(int* m,
             int* n,
             rocblas_float_complex* A,
             int* lda,
             rocblas_float_complex* ipiv,
             rocblas_float_complex* work,
             int* lwork,
             int* info);
void zgerqf_(int* m,
             int* n,
             rocblas_double_complex* A,
             int* lda,
             rocblas_double_complex* ipiv,
             rocblas_double_complex* work,
             int* lwork,
             int* info);

void sgeql2_(int* m, int* n, float* A, int* lda, float* ipiv, float* work, int* info);
void dgeql2_(int* m, int* n, double* A, int* lda, double* ipiv, double* work, int* info);
void cgeql2_(int* m,
             int* n,
             rocblas_float_complex* A,
             int* lda,
             rocblas_float_complex* ipiv,
             rocblas_float_complex* work,
             int* info);
void zgeql2_(int* m,
             int* n,
             rocblas_double_complex* A,
             int* lda,
             rocblas_double_complex* ipiv,
             rocblas_double_complex* work,
             int* info);

void sgeqlf_(int* m, int* n, float* A, int* lda, float* ipiv, float* work, int* lwork, int* info);
void dgeqlf_(int* m, int* n, double* A, int* lda, double* ipiv, double* work, int* lwork, int* info);
void cgeqlf_(int* m,
             int* n,
             rocblas_float_complex* A,
             int* lda,
             rocblas_float_complex* ipiv,
             rocblas_float_complex* work,
             int* lwork,
             int* info);
void zgeqlf_(int* m,
             int* n,
             rocblas_double_complex* A,
             int* lda,
             rocblas_double_complex* ipiv,
             rocblas_double_complex* work,
             int* lwork,
             int* info);

void sgelq2_(int* m, int* n, float* A, int* lda, float* ipiv, float* work, int* info);
void dgelq2_(int* m, int* n, double* A, int* lda, double* ipiv, double* work, int* info);
void cgelq2_(int* m,
             int* n,
             rocblas_float_complex* A,
             int* lda,
             rocblas_float_complex* ipiv,
             rocblas_float_complex* work,
             int* info);
void zgelq2_(int* m,
             int* n,
             rocblas_double_complex* A,
             int* lda,
             rocblas_double_complex* ipiv,
             rocblas_double_complex* work,
             int* info);

void sgelqf_(int* m, int* n, float* A, int* lda, float* ipiv, float* work, int* lwork, int* info);
void dgelqf_(int* m, int* n, double* A, int* lda, double* ipiv, double* work, int* lwork, int* info);
void cgelqf_(int* m,
             int* n,
             rocblas_float_complex* A,
             int* lda,
             rocblas_float_complex* ipiv,
             rocblas_float_complex* work,
             int* lwork,
             int* info);
void zgelqf_(int* m,
             int* n,
             rocblas_double_complex* A,
             int* lda,
             rocblas_double_complex* ipiv,
             rocblas_double_complex* work,
             int* lwork,
             int* info);

void clacgv_(int* n, rocblas_float_complex* x, int* incx);
void zlacgv_(int* n, rocblas_double_complex* x, int* incx);

void clacpy_(char* uplo,
             int* m,
             int* n,
             rocblas_float_complex* A,
             int* lda,
             rocblas_float_complex* B,
             int* ldb);
void zlacpy_(char* uplo,
             int* m,
             int* n,
             rocblas_double_complex* A,
             int* lda,
             rocblas_double_complex* B,
             int* ldb);

void slaswp_(int* n, float* A, int* lda, int* k1, int* k2, int* ipiv, int* inc);
void dlaswp_(int* n, double* A, int* lda, int* k1, int* k2, int* ipiv, int* inc);
void claswp_(int* n, rocblas_float_complex* A, int* lda, int* k1, int* k2, int* ipiv, int* inc);
void zlaswp_(int* n, rocblas_double_complex* A, int* lda, int* k1, int* k2, int* ipiv, int* inc);

void slauum_(char* uplo, int* n, float* A, int* lda, int* info);
void dlauum_(char* uplo, int* n, double* A, int* lda, int* info);
void clauum_(char* uplo, int* n, rocblas_float_complex* A, int* lda, int* info);
void zlauum_(char* uplo, int* n, rocblas_double_complex* A, int* lda, int* info);

void sorg2r_(int* m, int* n, int* k, float* A, int* lda, float* ipiv, float* work, int* info);
void dorg2r_(int* m, int* n, int* k, double* A, int* lda, double* ipiv, double* work, int* info);
void cung2r_(int* m,
             int* n,
             int* k,
             rocblas_float_complex* A,
             int* lda,
             rocblas_float_complex* ipiv,
             rocblas_float_complex* work,
             int* info);
void zung2r_(int* m,
             int* n,
             int* k,
             rocblas_double_complex* A,
             int* lda,
             rocblas_double_complex* ipiv,
             rocblas_double_complex* work,
             int* info);

void sorgqr_(int* m, int* n, int* k, float* A, int* lda, float* ipiv, float* work, int* lwork, int* info);
void dorgqr_(int* m, int* n, int* k, double* A, int* lda, double* ipiv, double* work, int* lwork, int* info);
void cungqr_(int* m,
             int* n,
             int* k,
             rocblas_float_complex* A,
             int* lda,
             rocblas_float_complex* ipiv,
             rocblas_float_complex* work,
             int* lwork,
             int* info);
void zungqr_(int* m,
             int* n,
             int* k,
             rocblas_double_complex* A,
             int* lda,
             rocblas_double_complex* ipiv,
             rocblas_double_complex* work,
             int* lwork,
             int* info);

void sorgl2_(int* m, int* n, int* k, float* A, int* lda, float* ipiv, float* work, int* info);
void dorgl2_(int* m, int* n, int* k, double* A, int* lda, double* ipiv, double* work, int* info);
void cungl2_(int* m,
             int* n,
             int* k,
             rocblas_float_complex* A,
             int* lda,
             rocblas_float_complex* ipiv,
             rocblas_float_complex* work,
             int* info);
void zungl2_(int* m,
             int* n,
             int* k,
             rocblas_double_complex* A,
             int* lda,
             rocblas_double_complex* ipiv,
             rocblas_double_complex* work,
             int* info);

void sorglq_(int* m, int* n, int* k, float* A, int* lda, float* ipiv, float* work, int* lwork, int* info);
void dorglq_(int* m, int* n, int* k, double* A, int* lda, double* ipiv, double* work, int* lwork, int* info);
void cunglq_(int* m,
             int* n,
             int* k,
             rocblas_float_complex* A,
             int* lda,
             rocblas_float_complex* ipiv,
             rocblas_float_complex* work,
             int* lwork,
             int* info);
void zunglq_(int* m,
             int* n,
             int* k,
             rocblas_double_complex* A,
             int* lda,
             rocblas_double_complex* ipiv,
             rocblas_double_complex* work,
             int* lwork,
             int* info);

void sorg2l_(int* m, int* n, int* k, float* A, int* lda, float* ipiv, float* work, int* info);
void dorg2l_(int* m, int* n, int* k, double* A, int* lda, double* ipiv, double* work, int* info);
void cung2l_(int* m,
             int* n,
             int* k,
             rocblas_float_complex* A,
             int* lda,
             rocblas_float_complex* ipiv,
             rocblas_float_complex* work,
             int* info);
void zung2l_(int* m,
             int* n,
             int* k,
             rocblas_double_complex* A,
             int* lda,
             rocblas_double_complex* ipiv,
             rocblas_double_complex* work,
             int* info);

void sorgql_(int* m, int* n, int* k, float* A, int* lda, float* ipiv, float* work, int* lwork, int* info);
void dorgql_(int* m, int* n, int* k, double* A, int* lda, double* ipiv, double* work, int* lwork, int* info);
void cungql_(int* m,
             int* n,
             int* k,
             rocblas_float_complex* A,
             int* lda,
             rocblas_float_complex* ipiv,
             rocblas_float_complex* work,
             int* lwork,
             int* info);
void zungql_(int* m,
             int* n,
             int* k,
             rocblas_double_complex* A,
             int* lda,
             rocblas_double_complex* ipiv,
             rocblas_double_complex* work,
             int* lwork,
             int* info);

void sorgbr_(char* vect,
             int* m,
             int* n,
             int* k,
             float* A,
             int* lda,
             float* Ipiv,
             float* work,
             int* size_w,
             int* info);
void dorgbr_(char* vect,
             int* m,
             int* n,
             int* k,
             double* A,
             int* lda,
             double* Ipiv,
             double* work,
             int* size_w,
             int* info);
void cungbr_(char* vect,
             int* m,
             int* n,
             int* k,
             rocblas_float_complex* A,
             int* lda,
             rocblas_float_complex* Ipiv,
             rocblas_float_complex* work,
             int* size_w,
             int* info);
void zungbr_(char* vect,
             int* m,
             int* n,
             int* k,
             rocblas_double_complex* A,
             int* lda,
             rocblas_double_complex* Ipiv,
             rocblas_double_complex* work,
             int* size_w,
             int* info);

void sorgtr_(char* uplo, int* n, float* A, int* lda, float* Ipiv, float* work, int* size_w, int* info);
void dorgtr_(char* uplo, int* n, double* A, int* lda, double* Ipiv, double* work, int* size_w, int* info);
void cungtr_(char* uplo,
             int* n,
             rocblas_float_complex* A,
             int* lda,
             rocblas_float_complex* Ipiv,
             rocblas_float_complex* work,
             int* size_w,
             int* info);
void zungtr_(char* uplo,
             int* n,
             rocblas_double_complex* A,
             int* lda,
             rocblas_double_complex* Ipiv,
             rocblas_double_complex* work,
             int* size_w,
             int* info);

void sorm2r_(char* side,
             char* trans,
             int* m,
             int* n,
             int* k,
             float* A,
             int* lda,
             float* ipiv,
             float* C,
             int* ldc,
             float* work,
             int* info);
void dorm2r_(char* side,
             char* trans,
             int* m,
             int* n,
             int* k,
             double* A,
             int* lda,
             double* ipiv,
             double* C,
             int* ldc,
             double* work,
             int* info);
void cunm2r_(char* side,
             char* trans,
             int* m,
             int* n,
             int* k,
             rocblas_float_complex* A,
             int* lda,
             rocblas_float_complex* ipiv,
             rocblas_float_complex* C,
             int* ldc,
             rocblas_float_complex* work,
             int* info);
void zunm2r_(char* side,
             char* trans,
             int* m,
             int* n,
             int* k,
             rocblas_double_complex* A,
             int* lda,
             rocblas_double_complex* ipiv,
             rocblas_double_complex* C,
             int* ldc,
             rocblas_double_complex* work,
             int* info);

void sormqr_(char* side,
             char* trans,
             int* m,
             int* n,
             int* k,
             float* A,
             int* lda,
             float* ipiv,
             float* C,
             int* ldc,
             float* work,
             int* sizeW,
             int* info);
void dormqr_(char* side,
             char* trans,
             int* m,
             int* n,
             int* k,
             double* A,
             int* lda,
             double* ipiv,
             double* C,
             int* ldc,
             double* work,
             int* sizeW,
             int* info);
void cunmqr_(char* side,
             char* trans,
             int* m,
             int* n,
             int* k,
             rocblas_float_complex* A,
             int* lda,
             rocblas_float_complex* ipiv,
             rocblas_float_complex* C,
             int* ldc,
             rocblas_float_complex* work,
             int* sizeW,
             int* info);
void zunmqr_(char* side,
             char* trans,
             int* m,
             int* n,
             int* k,
             rocblas_double_complex* A,
             int* lda,
             rocblas_double_complex* ipiv,
             rocblas_double_complex* C,
             int* ldc,
             rocblas_double_complex* work,
             int* sizeW,
             int* info);

void sorml2_(char* side,
             char* trans,
             int* m,
             int* n,
             int* k,
             float* A,
             int* lda,
             float* ipiv,
             float* C,
             int* ldc,
             float* work,
             int* info);
void dorml2_(char* side,
             char* trans,
             int* m,
             int* n,
             int* k,
             double* A,
             int* lda,
             double* ipiv,
             double* C,
             int* ldc,
             double* work,
             int* info);
void cunml2_(char* side,
             char* trans,
             int* m,
             int* n,
             int* k,
             rocblas_float_complex* A,
             int* lda,
             rocblas_float_complex* ipiv,
             rocblas_float_complex* C,
             int* ldc,
             rocblas_float_complex* work,
             int* info);
void zunml2_(char* side,
             char* trans,
             int* m,
             int* n,
             int* k,
             rocblas_double_complex* A,
             int* lda,
             rocblas_double_complex* ipiv,
             rocblas_double_complex* C,
             int* ldc,
             rocblas_double_complex* work,
             int* info);

void sormlq_(char* side,
             char* trans,
             int* m,
             int* n,
             int* k,
             float* A,
             int* lda,
             float* ipiv,
             float* C,
             int* ldc,
             float* work,
             int* sizeW,
             int* info);
void dormlq_(char* side,
             char* trans,
             int* m,
             int* n,
             int* k,
             double* A,
             int* lda,
             double* ipiv,
             double* C,
             int* ldc,
             double* work,
             int* sizeW,
             int* info);
void cunmlq_(char* side,
             char* trans,
             int* m,
             int* n,
             int* k,
             rocblas_float_complex* A,
             int* lda,
             rocblas_float_complex* ipiv,
             rocblas_float_complex* C,
             int* ldc,
             rocblas_float_complex* work,
             int* sizeW,
             int* info);
void zunmlq_(char* side,
             char* trans,
             int* m,
             int* n,
             int* k,
             rocblas_double_complex* A,
             int* lda,
             rocblas_double_complex* ipiv,
             rocblas_double_complex* C,
             int* ldc,
             rocblas_double_complex* work,
             int* sizeW,
             int* info);

void sorm2l_(char* side,
             char* trans,
             int* m,
             int* n,
             int* k,
             float* A,
             int* lda,
             float* ipiv,
             float* C,
             int* ldc,
             float* work,
             int* info);
void dorm2l_(char* side,
             char* trans,
             int* m,
             int* n,
             int* k,
             double* A,
             int* lda,
             double* ipiv,
             double* C,
             int* ldc,
             double* work,
             int* info);
void cunm2l_(char* side,
             char* trans,
             int* m,
             int* n,
             int* k,
             rocblas_float_complex* A,
             int* lda,
             rocblas_float_complex* ipiv,
             rocblas_float_complex* C,
             int* ldc,
             rocblas_float_complex* work,
             int* info);
void zunm2l_(char* side,
             char* trans,
             int* m,
             int* n,
             int* k,
             rocblas_double_complex* A,
             int* lda,
             rocblas_double_complex* ipiv,
             rocblas_double_complex* C,
             int* ldc,
             rocblas_double_complex* work,
             int* info);

void sormql_(char* side,
             char* trans,
             int* m,
             int* n,
             int* k,
             float* A,
             int* lda,
             float* ipiv,
             float* C,
             int* ldc,
             float* work,
             int* sizeW,
             int* info);
void dormql_(char* side,
             char* trans,
             int* m,
             int* n,
             int* k,
             double* A,
             int* lda,
             double* ipiv,
             double* C,
             int* ldc,
             double* work,
             int* sizeW,
             int* info);
void cunmql_(char* side,
             char* trans,
             int* m,
             int* n,
             int* k,
             rocblas_float_complex* A,
             int* lda,
             rocblas_float_complex* ipiv,
             rocblas_float_complex* C,
             int* ldc,
             rocblas_float_complex* work,
             int* sizeW,
             int* info);
void zunmql_(char* side,
             char* trans,
             int* m,
             int* n,
             int* k,
             rocblas_double_complex* A,
             int* lda,
             rocblas_double_complex* ipiv,
             rocblas_double_complex* C,
             int* ldc,
             rocblas_double_complex* work,
             int* sizeW,
             int* info);

void sormbr_(char* vect,
             char* side,
             char* trans,
             int* m,
             int* n,
             int* k,
             float* A,
             int* lda,
             float* ipiv,
             float* C,
             int* ldc,
             float* work,
             int* sizeW,
             int* info);
void dormbr_(char* vect,
             char* side,
             char* trans,
             int* m,
             int* n,
             int* k,
             double* A,
             int* lda,
             double* ipiv,
             double* C,
             int* ldc,
             double* work,
             int* sizeW,
             int* info);
void cunmbr_(char* vect,
             char* side,
             char* trans,
             int* m,
             int* n,
             int* k,
             rocblas_float_complex* A,
             int* lda,
             rocblas_float_complex* ipiv,
             rocblas_float_complex* C,
             int* ldc,
             rocblas_float_complex* work,
             int* sizeW,
             int* info);
void zunmbr_(char* vect,
             char* side,
             char* trans,
             int* m,
             int* n,
             int* k,
             rocblas_double_complex* A,
             int* lda,
             rocblas_double_complex* ipiv,
             rocblas_double_complex* C,
             int* ldc,
             rocblas_double_complex* work,
             int* sizeW,
             int* info);

void sormtr_(char* side,
             char* uplo,
             char* trans,
             int* m,
             int* n,
             float* A,
             int* lda,
             float* ipiv,
             float* C,
             int* ldc,
             float* work,
             int* sizeW,
             int* info);
void dormtr_(char* side,
             char* uplo,
             char* trans,
             int* m,
             int* n,
             double* A,
             int* lda,
             double* ipiv,
             double* C,
             int* ldc,
             double* work,
             int* sizeW,
             int* info);
void cunmtr_(char* side,
             char* uplo,
             char* trans,
             int* m,
             int* n,
             rocblas_float_complex* A,
             int* lda,
             rocblas_float_complex* ipiv,
             rocblas_float_complex* C,
             int* ldc,
             rocblas_float_complex* work,
             int* sizeW,
             int* info);
void zunmtr_(char* side,
             char* uplo,
             char* trans,
             int* m,
             int* n,
             rocblas_double_complex* A,
             int* lda,
             rocblas_double_complex* ipiv,
             rocblas_double_complex* C,
             int* ldc,
             rocblas_double_complex* work,
             int* sizeW,
             int* info);

void sgebd2_(int* m,
             int* n,
             float* A,
             int* lda,
             float* D,
             float* E,
             float* tauq,
             float* taup,
             float* work,
             int* info);
void dgebd2_(int* m,
             int* n,
             double* A,
             int* lda,
             double* D,
             double* E,
             double* tauq,
             double* taup,
             double* work,
             int* info);
void cgebd2_(int* m,
             int* n,
             rocblas_float_complex* A,
             int* lda,
             float* D,
             float* E,
             rocblas_float_complex* tauq,
             rocblas_float_complex* taup,
             rocblas_float_complex* work,
             int* info);
void zgebd2_(int* m,
             int* n,
             rocblas_double_complex* A,
             int* lda,
             double* D,
             double* E,
             rocblas_double_complex* tauq,
             rocblas_double_complex* taup,
             rocblas_double_complex* work,
             int* info);

void sgebrd_(int* m,
             int* n,
             float* A,
             int* lda,
             float* D,
             float* E,
             float* tauq,
             float* taup,
             float* work,
             int* size_w,
             int* info);
void dgebrd_(int* m,
             int* n,
             double* A,
             int* lda,
             double* D,
             double* E,
             double* tauq,
             double* taup,
             double* work,
             int* size_w,
             int* info);
void cgebrd_(int* m,
             int* n,
             rocblas_float_complex* A,
             int* lda,
             float* D,
             float* E,
             rocblas_float_complex* tauq,
             rocblas_float_complex* taup,
             rocblas_float_complex* work,
             int* size_w,
             int* info);
void zgebrd_(int* m,
             int* n,
             rocblas_double_complex* A,
             int* lda,
             double* D,
             double* E,
             rocblas_double_complex* tauq,
             rocblas_double_complex* taup,
             rocblas_double_complex* work,
             int* size_w,
             int* info);

void ssytrd_(char* uplo,
             int* n,
             float* A,
             int* lda,
             float* D,
             float* E,
             float* tau,
             float* work,
             int* size_w,
             int* info);
void dsytrd_(char* uplo,
             int* n,
             double* A,
             int* lda,
             double* D,
             double* E,
             double* tau,
             double* work,
             int* size_w,
             int* info);
void chetrd_(char* uplo,
             int* n,
             rocblas_float_complex* A,
             int* lda,
             float* D,
             float* E,
             rocblas_float_complex* tau,
             rocblas_float_complex* work,
             int* size_w,
             int* info);
void zhetrd_(char* uplo,
             int* n,
             rocblas_double_complex* A,
             int* lda,
             double* D,
             double* E,
             rocblas_double_complex* tau,
             rocblas_double_complex* work,
             int* size_w,
             int* info);

void ssytd2_(char* uplo, int* n, float* A, int* lda, float* D, float* E, float* tau, int* info);
void dsytd2_(char* uplo, int* n, double* A, int* lda, double* D, double* E, double* tau, int* info);
void chetd2_(char* uplo,
             int* n,
             rocblas_float_complex* A,
             int* lda,
             float* D,
             float* E,
             rocblas_float_complex* tau,
             int* info);
void zhetd2_(char* uplo,
             int* n,
             rocblas_double_complex* A,
             int* lda,
             double* D,
             double* E,
             rocblas_double_complex* tau,
             int* info);

void sgesvd_(char* jobu,
             char* jobv,
             int* m,
             int* n,
             float* A,
             int* lda,
             float* S,
             float* U,
             int* ldu,
             float* V,
             int* ldv,
             float* E,
             int* lwork,
             int* info);
void dgesvd_(char* jobu,
             char* jobv,
             int* m,
             int* n,
             double* A,
             int* lda,
             double* S,
             double* U,
             int* ldu,
             double* V,
             int* ldv,
             double* E,
             int* lwork,
             int* info);
void cgesvd_(char* jobu,
             char* jobv,
             int* m,
             int* n,
             rocblas_float_complex* A,
             int* lda,
             float* S,
             rocblas_float_complex* U,
             int* ldu,
             rocblas_float_complex* V,
             int* ldv,
             rocblas_float_complex* work,
             int* lwork,
             float* E,
             int* info);
void zgesvd_(char* jobu,
             char* jobv,
             int* m,
             int* n,
             rocblas_double_complex* A,
             int* lda,
             double* S,
             rocblas_double_complex* U,
             int* ldu,
             rocblas_double_complex* V,
             int* ldv,
             rocblas_double_complex* work,
             int* lwork,
             double* E,
             int* info);

void sgesvdx_(char* jobu,
              char* jobv,
              char* srange,
              int* m,
              int* n,
              float* A,
              int* lda,
              float* vl,
              float* vu,
              int* il,
              int* iu,
              int* nsv,
              float* S,
              float* U,
              int* ldu,
              float* V,
              int* ldv,
              float* work,
              int* lwork,
              int* iwork,
              int* info);
void dgesvdx_(char* jobu,
              char* jobv,
              char* srange,
              int* m,
              int* n,
              double* A,
              int* lda,
              double* vl,
              double* vu,
              int* il,
              int* iu,
              int* nsv,
              double* S,
              double* U,
              int* ldu,
              double* V,
              int* ldv,
              double* work,
              int* lwork,
              int* iwork,
              int* info);
void cgesvdx_(char* jobu,
              char* jobv,
              char* srange,
              int* m,
              int* n,
              rocblas_float_complex* A,
              int* lda,
              float* vl,
              float* vu,
              int* il,
              int* iu,
              int* nsv,
              float* S,
              rocblas_float_complex* U,
              int* ldu,
              rocblas_float_complex* V,
              int* ldv,
              rocblas_float_complex* work,
              int* lwork,
              float* rwork,
              int* iwork,
              int* info);
void zgesvdx_(char* jobu,
              char* jobv,
              char* srange,
              int* m,
              int* n,
              rocblas_double_complex* A,
              int* lda,
              double* vl,
              double* vu,
              int* il,
              int* iu,
              int* nsv,
              double* S,
              rocblas_double_complex* U,
              int* ldu,
              rocblas_double_complex* V,
              int* ldv,
              rocblas_double_complex* work,
              int* lwork,
              double* rwork,
              int* iwork,
              int* info);

void ssterf_(int* n, float* D, float* E, int* info);
void dsterf_(int* n, double* D, double* E, int* info);

void ssteqr_(char* evect, int* n, float* D, float* E, float* C, int* ldc, float* work, int* info);
void dsteqr_(char* evect, int* n, double* D, double* E, double* C, int* ldc, double* work, int* info);
void csteqr_(char* evect,
             int* n,
             float* D,
             float* E,
             rocblas_float_complex* C,
             int* ldc,
             float* work,
             int* info);
void zsteqr_(char* evect,
             int* n,
             double* D,
             double* E,
             rocblas_double_complex* C,
             int* ldc,
             double* work,
             int* info);

void sstedc_(char* evect,
             int* n,
             float* D,
             float* E,
             float* C,
             int* ldc,
             float* work,
             int* lwork,
             int* iwork,
             int* liwork,
             int* info);
void dstedc_(char* evect,
             int* n,
             double* D,
             double* E,
             double* C,
             int* ldc,
             double* work,
             int* lwork,
             int* iwork,
             int* liwork,
             int* info);
void cstedc_(char* evect,
             int* n,
             float* D,
             float* E,
             rocblas_float_complex* C,
             int* ldc,
             rocblas_float_complex* work,
             int* lwork,
             float* rwork,
             int* lrwork,
             int* iwork,
             int* liwork,
             int* info);
void zstedc_(char* evect,
             int* n,
             double* D,
             double* E,
             rocblas_double_complex* C,
             int* ldc,
             rocblas_double_complex* work,
             int* lwork,
             double* rwork,
             int* lrwork,
             int* iwork,
             int* liwork,
             int* info);

void sstebz_(char* erange,
             char* order,
             int* n,
             float* vl,
             float* vu,
             int* il,
             int* iu,
             float* abstol,
             float* D,
             float* E,
             int* nev,
             int* nsplit,
             float* W,
             int* iblock,
             int* isplit,
             float* work,
             int* iwork,
             int* info);
void dstebz_(char* erange,
             char* order,
             int* n,
             double* vl,
             double* vu,
             int* il,
             int* iu,
             double* abstol,
             double* D,
             double* E,
             int* nev,
             int* nsplit,
             double* W,
             int* iblock,
             int* isplit,
             double* work,
             int* iwork,
             int* info);

void sstein_(int* n,
             float* D,
             float* E,
             int* nev,
             float* W,
             int* iblock,
             int* isplit,
             float* Z,
             int* ldz,
             float* work,
             int* iwork,
             int* ifail,
             int* info);
void dstein_(int* n,
             double* D,
             double* E,
             int* nev,
             double* W,
             int* iblock,
             int* isplit,
             double* Z,
             int* ldz,
             double* work,
             int* iwork,
             int* ifail,
             int* info);
void cstein_(int* n,
             float* D,
             float* E,
             int* nev,
             float* W,
             int* iblock,
             int* isplit,
             rocblas_float_complex* Z,
             int* ldz,
             float* work,
             int* iwork,
             int* ifail,
             int* info);
void zstein_(int* n,
             double* D,
             double* E,
             int* nev,
             double* W,
             int* iblock,
             int* isplit,
             rocblas_double_complex* Z,
             int* ldz,
             double* work,
             int* iwork,
             int* ifail,
             int* info);

void ssygs2_(int* itype, char* uplo, int* n, float* A, int* lda, float* B, int* ldb, int* info);
void dsygs2_(int* itype, char* uplo, int* n, double* A, int* lda, double* B, int* ldb, int* info);
void chegs2_(int* itype,
             char* uplo,
             int* n,
             rocblas_float_complex* A,
             int* lda,
             rocblas_float_complex* B,
             int* ldb,
             int* info);
void zhegs2_(int* itype,
             char* uplo,
             int* n,
             rocblas_double_complex* A,
             int* lda,
             rocblas_double_complex* B,
             int* ldb,
             int* info);

void ssygst_(int* itype, char* uplo, int* n, float* A, int* lda, float* B, int* ldb, int* info);
void dsygst_(int* itype, char* uplo, int* n, double* A, int* lda, double* B, int* ldb, int* info);
void chegst_(int* itype,
             char* uplo,
             int* n,
             rocblas_float_complex* A,
             int* lda,
             rocblas_float_complex* B,
             int* ldb,
             int* info);
void zhegst_(int* itype,
             char* uplo,
             int* n,
             rocblas_double_complex* A,
             int* lda,
             rocblas_double_complex* B,
             int* ldb,
             int* info);

void ssyev_(char* evect,
            char* uplo,
            int* n,
            float* A,
            int* lda,
            float* W,
            float* work,
            int* lwork,
            int* info);
void dsyev_(char* evect,
            char* uplo,
            int* n,
            double* A,
            int* lda,
            double* W,
            double* work,
            int* lwork,
            int* info);
void cheev_(char* evect,
            char* uplo,
            int* n,
            rocblas_float_complex* A,
            int* lda,
            float* W,
            rocblas_float_complex* work,
            int* lwork,
            float* rwork,
            int* info);
void zheev_(char* evect,
            char* uplo,
            int* n,
            rocblas_double_complex* A,
            int* lda,
            double* W,
            rocblas_double_complex* work,
            int* lwork,
            double* rwork,
            int* info);

void ssyevd_(char* evect,
             char* uplo,
             int* n,
             float* A,
             int* lda,
             float* W,
             float* work,
             int* lwork,
             int* iwork,
             int* liwork,
             int* info);
void dsyevd_(char* evect,
             char* uplo,
             int* n,
             double* A,
             int* lda,
             double* W,
             double* work,
             int* lwork,
             int* iwork,
             int* liwork,
             int* info);
void cheevd_(char* evect,
             char* uplo,
             int* n,
             rocblas_float_complex* A,
             int* lda,
             float* W,
             rocblas_float_complex* work,
             int* lwork,
             float* rwork,
             int* lrwork,
             int* iwork,
             int* liwork,
             int* info);
void zheevd_(char* evect,
             char* uplo,
             int* n,
             rocblas_double_complex* A,
             int* lda,
             double* W,
             rocblas_double_complex* work,
             int* lwork,
             double* rwork,
             int* lrwork,
             int* iwork,
             int* liwork,
             int* info);

void ssyevx_(char* evect,
             char* erange,
             char* uplo,
             int* n,
             float* A,
             int* lda,
             float* vl,
             float* vu,
             int* il,
             int* iu,
             float* abstol,
             int* nev,
             float* W,
             float* Z,
             int* ldz,
             float* work,
             int* lwork,
             int* iwork,
             int* ifail,
             int* info);
void dsyevx_(char* evect,
             char* erange,
             char* uplo,
             int* n,
             double* A,
             int* lda,
             double* vl,
             double* vu,
             int* il,
             int* iu,
             double* abstol,
             int* nev,
             double* W,
             double* Z,
             int* ldz,
             double* work,
             int* lwork,
             int* iwork,
             int* ifail,
             int* info);
void cheevx_(char* evect,
             char* erange,
             char* uplo,
             int* n,
             rocblas_float_complex* A,
             int* lda,
             float* vl,
             float* vu,
             int* il,
             int* iu,
             float* abstol,
             int* nev,
             float* W,
             rocblas_float_complex* Z,
             int* ldz,
             rocblas_float_complex* work,
             int* lwork,
             float* rwork,
             int* iwork,
             int* ifail,
             int* info);
void zheevx_(char* evect,
             char* erange,
             char* uplo,
             int* n,
             rocblas_double_complex* A,
             int* lda,
             double* vl,
             double* vu,
             int* il,
             int* iu,
             double* abstol,
             int* nev,
             double* W,
             rocblas_double_complex* Z,
             int* ldz,
             rocblas_double_complex* work,
             int* lwork,
             double* rwork,
             int* iwork,
             int* ifail,
             int* info);

void ssygv_(int* itype,
            char* evect,
            char* uplo,
            int* n,
            float* A,
            int* lda,
            float* B,
            int* ldb,
            float* W,
            float* work,
            int* lwork,
            int* info);
void dsygv_(int* itype,
            char* evect,
            char* uplo,
            int* n,
            double* A,
            int* lda,
            double* B,
            int* ldb,
            double* W,
            double* work,
            int* lwork,
            int* info);
void chegv_(int* itype,
            char* evect,
            char* uplo,
            int* n,
            rocblas_float_complex* A,
            int* lda,
            rocblas_float_complex* B,
            int* ldb,
            float* W,
            rocblas_float_complex* work,
            int* lwork,
            float* rwork,
            int* info);
void zhegv_(int* itype,
            char* evect,
            char* uplo,
            int* n,
            rocblas_double_complex* A,
            int* lda,
            rocblas_double_complex* B,
            int* ldb,
            double* W,
            rocblas_double_complex* work,
            int* lwork,
            double* rwork,
            int* info);

void ssygvd_(int* itype,
             char* evect,
             char* uplo,
             int* n,
             float* A,
             int* lda,
             float* B,
             int* ldb,
             float* W,
             float* work,
             int* lwork,
             int* iwork,
             int* liwork,
             int* info);
void dsygvd_(int* itype,
             char* evect,
             char* uplo,
             int* n,
             double* A,
             int* lda,
             double* B,
             int* ldb,
             double* W,
             double* work,
             int* lwork,
             int* iwork,
             int* liwork,
             int* info);
void chegvd_(int* itype,
             char* evect,
             char* uplo,
             int* n,
             rocblas_float_complex* A,
             int* lda,
             rocblas_float_complex* B,
             int* ldb,
             float* W,
             rocblas_float_complex* work,
             int* lwork,
             float* rwork,
             int* lrwork,
             int* iwork,
             int* liwork,
             int* info);
void zhegvd_(int* itype,
             char* evect,
             char* uplo,
             int* n,
             rocblas_double_complex* A,
             int* lda,
             rocblas_double_complex* B,
             int* ldb,
             double* W,
             rocblas_double_complex* work,
             int* lwork,
             double* rwork,
             int* lrwork,
             int* iwork,
             int* liwork,
             int* info);

void ssygvx_(int* itype,
             char* evect,
             char* erange,
             char* uplo,
             int* n,
             float* A,
             int* lda,
             float* B,
             int* ldb,
             float* vl,
             float* vu,
             int* il,
             int* iu,
             float* abstol,
             int* m,
             float* W,
             float* Z,
             int* ldz,
             float* work,
             int* lwork,
             int* iwork,
             int* ifail,
             int* info);
void dsygvx_(int* itype,
             char* evect,
             char* erange,
             char* uplo,
             int* n,
             double* A,
             int* lda,
             double* B,
             int* ldb,
             double* vl,
             double* vu,
             int* il,
             int* iu,
             double* abstol,
             int* m,
             double* W,
             double* Z,
             int* ldz,
             double* work,
             int* lwork,
             int* iwork,
             int* ifail,
             int* info);
void chegvx_(int* itype,
             char* evect,
             char* erange,
             char* uplo,
             int* n,
             rocblas_float_complex* A,
             int* lda,
             rocblas_float_complex* B,
             int* ldb,
             float* vl,
             float* vu,
             int* il,
             int* iu,
             float* abstol,
             int* m,
             float* W,
             rocblas_float_complex* Z,
             int* ldz,
             rocblas_float_complex* work,
             int* lwork,
             float* rwork,
             int* iwork,
             int* ifail,
             int* info);
void zhegvx_(int* itype,
             char* evect,
             char* erange,
             char* uplo,
             int* n,
             rocblas_double_complex* A,
             int* lda,
             rocblas_double_complex* B,
             int* ldb,
             double* vl,
             double* vu,
             int* il,
             int* iu,
             double* abstol,
             int* m,
             double* W,
             rocblas_double_complex* Z,
             int* ldz,
             rocblas_double_complex* work,
             int* lwork,
             double* rwork,
             int* iwork,
             int* ifail,
             int* info);

void slasyf_(char* uplo,
             int* n,
             int* nb,
             int* kb,
             float* A,
             int* lda,
             int* ipiv,
             float* W,
             int* ldw,
             int* info);
void dlasyf_(char* uplo,
             int* n,
             int* nb,
             int* kb,
             double* A,
             int* lda,
             int* ipiv,
             double* W,
             int* ldw,
             int* info);
void clasyf_(char* uplo,
             int* n,
             int* nb,
             int* kb,
             rocblas_float_complex* A,
             int* lda,
             int* ipiv,
             rocblas_float_complex* W,
             int* ldw,
             int* info);
void zlasyf_(char* uplo,
             int* n,
             int* nb,
             int* kb,
             rocblas_double_complex* A,
             int* lda,
             int* ipiv,
             rocblas_double_complex* W,
             int* ldw,
             int* info);

void ssytf2_(char* uplo, int* n, float* A, int* lda, int* ipiv, int* info);
void dsytf2_(char* uplo, int* n, double* A, int* lda, int* ipiv, int* info);
void csytf2_(char* uplo, int* n, rocblas_float_complex* A, int* lda, int* ipiv, int* info);
void zsytf2_(char* uplo, int* n, rocblas_double_complex* A, int* lda, int* ipiv, int* info);

void ssytrf_(char* uplo, int* n, float* A, int* lda, int* ipiv, float* work, int* lwork, int* info);
void dsytrf_(char* uplo, int* n, double* A, int* lda, int* ipiv, double* work, int* lwork, int* info);
void csytrf_(char* uplo,
             int* n,
             rocblas_float_complex* A,
             int* lda,
             int* ipiv,
             rocblas_float_complex* work,
             int* lwork,
             int* info);
void zsytrf_(char* uplo,
             int* n,
             rocblas_double_complex* A,
             int* lda,
             int* ipiv,
             rocblas_double_complex* work,
             int* lwork,
             int* info);

void sbdsvdx_(char* uplo,
              char* svect,
              char* srange,
              int* n,
              float* D,
              float* E,
              float* vl,
              float* vu,
              int* il,
              int* iu,
              int* nsv,
              float* S,
              float* Z,
              int* ldz,
              float* work,
              int* iwork,
              int* info);
void dbdsvdx_(char* uplo,
              char* svect,
              char* srange,
              int* n,
              double* D,
              double* E,
              double* vl,
              double* vu,
              int* il,
              int* iu,
              int* nsv,
              double* S,
              double* Z,
              int* ldz,
              double* work,
              int* iwork,
              int* info);

#ifdef __cplusplus
}
#endif
/************************************************************************/

/************************************************************************/
// These are templated functions used in rocSOLVER clients code

// lange

template <>
float cpu_lange<float, float>(char norm,
                              rocblas_int m,
                              rocblas_int n,
                              float* A,
                              rocblas_int lda,
                              float* work)
{
    return slange_(&norm, &m, &n, A, &lda, work);
}

template <>
double cpu_lange<double, double>(char norm,
                                 rocblas_int m,
                                 rocblas_int n,
                                 double* A,
                                 rocblas_int lda,
                                 double* work)
{
    return dlange_(&norm, &m, &n, A, &lda, work);
}

template <>
float cpu_lange<rocblas_float_complex, float>(char norm,
                                              rocblas_int m,
                                              rocblas_int n,
                                              rocblas_float_complex* A,
                                              rocblas_int lda,
                                              float* work)
{
    return clange_(&norm, &m, &n, A, &lda, work);
}

template <>
double cpu_lange<rocblas_double_complex, double>(char norm,
                                                 rocblas_int m,
                                                 rocblas_int n,
                                                 rocblas_double_complex* A,
                                                 rocblas_int lda,
                                                 double* work)
{
    return zlange_(&norm, &m, &n, A, &lda, work);
}

// gecon

template <>
float cpu_gecon<float, float>(char norm,
                              rocblas_int n,
                              float* A,
                              rocblas_int lda,
                              float anorm,
                              float* work,
                              float* rwork,
                              rocblas_int* iwork)
{
    float rcond;
    rocblas_int info;
    sgecon_(&norm, &n, A, &lda, &anorm, &rcond, work, iwork, &info);
    return rcond;
}

template <>
double cpu_gecon<double, double>(char norm,
                                 rocblas_int n,
                                 double* A,
                                 rocblas_int lda,
                                 double anorm,
                                 double* work,
                                 double* rwork,
                                 rocblas_int* iwork)
{
    double rcond;
    rocblas_int info;
    dgecon_(&norm, &n, A, &lda, &anorm, &rcond, work, iwork, &info);
    return rcond;
}

template <>
float cpu_gecon<rocblas_float_complex, float>(char norm,
                                              rocblas_int n,
                                              rocblas_float_complex* A,
                                              rocblas_int lda,
                                              float anorm,
                                              rocblas_float_complex* work,
                                              float* rwork,
                                              rocblas_int* iwork)
{
    float rcond;
    rocblas_int info;
    cgecon_(&norm, &n, A, &lda, &anorm, &rcond, work, rwork, &info);
    return rcond;
}

template <>
double cpu_gecon<rocblas_double_complex, double>(char norm,
                                                 rocblas_int n,
                                                 rocblas_double_complex* A,
                                                 rocblas_int lda,
                                                 double anorm,
                                                 rocblas_double_complex* work,
                                                 double* rwork,
                                                 rocblas_int* iwork)
{
    double rcond;
    rocblas_int info;
    zgecon_(&norm, &n, A, &lda, &anorm, &rcond, work, rwork, &info);
    return rcond;
}

// axpy

template <>
void cpu_axpy<float>(rocblas_int n, float alpha, float* x, rocblas_int incx, float* y, rocblas_int incy)
{
    saxpy_(&n, &alpha, x, &incx, y, &incy);
}

template <>
void cpu_axpy<double>(rocblas_int n, double alpha, double* x, rocblas_int incx, double* y, rocblas_int incy)
{
    daxpy_(&n, &alpha, x, &incx, y, &incy);
}

template <>
void cpu_axpy<rocblas_float_complex>(rocblas_int n,
                                     rocblas_float_complex alpha,
                                     rocblas_float_complex* x,
                                     rocblas_int incx,
                                     rocblas_float_complex* y,
                                     rocblas_int incy)
{
    caxpy_(&n, &alpha, x, &incx, y, &incy);
}

template <>
void cpu_axpy<rocblas_double_complex>(rocblas_int n,
                                      rocblas_double_complex alpha,
                                      rocblas_double_complex* x,
                                      rocblas_int incx,
                                      rocblas_double_complex* y,
                                      rocblas_int incy)
{
    zaxpy_(&n, &alpha, x, &incx, y, &incy);
}

// lacgv

template <>
void cpu_lacgv<rocblas_float_complex>(rocblas_int n, rocblas_float_complex* x, rocblas_int incx)
{
    clacgv_(&n, x, &incx);
}

template <>
void cpu_lacgv<rocblas_double_complex>(rocblas_int n, rocblas_double_complex* x, rocblas_int incx)
{
    zlacgv_(&n, x, &incx);
}

// lacpy
template <>
void cpu_lacpy<rocblas_float_complex>(rocblas_fill uplo,
                                      rocblas_int m,
                                      rocblas_int n,
                                      rocblas_float_complex* A,
                                      rocblas_int lda,
                                      rocblas_float_complex* B,
                                      rocblas_int ldb)
{
    char uploC = rocblas2char_fill(uplo);
    clacpy_(&uploC, &m, &n, A, &lda, B, &ldb);
}

template <>
void cpu_lacpy<rocblas_double_complex>(rocblas_fill uplo,
                                       rocblas_int m,
                                       rocblas_int n,
                                       rocblas_double_complex* A,
                                       rocblas_int lda,
                                       rocblas_double_complex* B,
                                       rocblas_int ldb)
{
    char uploC = rocblas2char_fill(uplo);
    zlacpy_(&uploC, &m, &n, A, &lda, B, &ldb);
}

// laswp

template <>
void cpu_laswp<float>(rocblas_int n,
                      float* A,
                      rocblas_int lda,
                      rocblas_int k1,
                      rocblas_int k2,
                      rocblas_int* ipiv,
                      rocblas_int inc)
{
    slaswp_(&n, A, &lda, &k1, &k2, ipiv, &inc);
}

template <>
void cpu_laswp<double>(rocblas_int n,
                       double* A,
                       rocblas_int lda,
                       rocblas_int k1,
                       rocblas_int k2,
                       rocblas_int* ipiv,
                       rocblas_int inc)
{
    dlaswp_(&n, A, &lda, &k1, &k2, ipiv, &inc);
}

template <>
void cpu_laswp<rocblas_float_complex>(rocblas_int n,
                                      rocblas_float_complex* A,
                                      rocblas_int lda,
                                      rocblas_int k1,
                                      rocblas_int k2,
                                      rocblas_int* ipiv,
                                      rocblas_int inc)
{
    claswp_(&n, A, &lda, &k1, &k2, ipiv, &inc);
}

template <>
void cpu_laswp<rocblas_double_complex>(rocblas_int n,
                                       rocblas_double_complex* A,
                                       rocblas_int lda,
                                       rocblas_int k1,
                                       rocblas_int k2,
                                       rocblas_int* ipiv,
                                       rocblas_int inc)
{
    zlaswp_(&n, A, &lda, &k1, &k2, ipiv, &inc);
}

// larfg

template <>
void cpu_larfg<float>(rocblas_int n, float* alpha, float* x, rocblas_int incx, float* tau)
{
    slarfg_(&n, alpha, x, &incx, tau);
}

template <>
void cpu_larfg<double>(rocblas_int n, double* alpha, double* x, rocblas_int incx, double* tau)
{
    dlarfg_(&n, alpha, x, &incx, tau);
}

template <>
void cpu_larfg<rocblas_float_complex>(rocblas_int n,
                                      rocblas_float_complex* alpha,
                                      rocblas_float_complex* x,
                                      rocblas_int incx,
                                      rocblas_float_complex* tau)
{
    clarfg_(&n, alpha, x, &incx, tau);
}

template <>
void cpu_larfg<rocblas_double_complex>(rocblas_int n,
                                       rocblas_double_complex* alpha,
                                       rocblas_double_complex* x,
                                       rocblas_int incx,
                                       rocblas_double_complex* tau)
{
    zlarfg_(&n, alpha, x, &incx, tau);
}

// larf

template <>
void cpu_larf<float>(rocblas_side sideR,
                     rocblas_int m,
                     rocblas_int n,
                     float* x,
                     rocblas_int incx,
                     float* alpha,
                     float* A,
                     rocblas_int lda,
                     float* work)
{
    char side = rocblas2char_side(sideR);
    slarf_(&side, &m, &n, x, &incx, alpha, A, &lda, work);
}

template <>
void cpu_larf<double>(rocblas_side sideR,
                      rocblas_int m,
                      rocblas_int n,
                      double* x,
                      rocblas_int incx,
                      double* alpha,
                      double* A,
                      rocblas_int lda,
                      double* work)
{
    char side = rocblas2char_side(sideR);
    dlarf_(&side, &m, &n, x, &incx, alpha, A, &lda, work);
}

template <>
void cpu_larf<rocblas_float_complex>(rocblas_side sideR,
                                     rocblas_int m,
                                     rocblas_int n,
                                     rocblas_float_complex* x,
                                     rocblas_int incx,
                                     rocblas_float_complex* alpha,
                                     rocblas_float_complex* A,
                                     rocblas_int lda,
                                     rocblas_float_complex* work)
{
    char side = rocblas2char_side(sideR);
    clarf_(&side, &m, &n, x, &incx, alpha, A, &lda, work);
}

template <>
void cpu_larf<rocblas_double_complex>(rocblas_side sideR,
                                      rocblas_int m,
                                      rocblas_int n,
                                      rocblas_double_complex* x,
                                      rocblas_int incx,
                                      rocblas_double_complex* alpha,
                                      rocblas_double_complex* A,
                                      rocblas_int lda,
                                      rocblas_double_complex* work)
{
    char side = rocblas2char_side(sideR);
    zlarf_(&side, &m, &n, x, &incx, alpha, A, &lda, work);
}

// larft

template <>
void cpu_larft<float>(rocblas_direct directR,
                      rocblas_storev storevR,
                      rocblas_int n,
                      rocblas_int k,
                      float* V,
                      rocblas_int ldv,
                      float* tau,
                      float* T,
                      rocblas_int ldt)
{
    char direct = rocblas2char_direct(directR);
    char storev = rocblas2char_storev(storevR);
    slarft_(&direct, &storev, &n, &k, V, &ldv, tau, T, &ldt);
}

template <>
void cpu_larft<double>(rocblas_direct directR,
                       rocblas_storev storevR,
                       rocblas_int n,
                       rocblas_int k,
                       double* V,
                       rocblas_int ldv,
                       double* tau,
                       double* T,
                       rocblas_int ldt)
{
    char direct = rocblas2char_direct(directR);
    char storev = rocblas2char_storev(storevR);
    dlarft_(&direct, &storev, &n, &k, V, &ldv, tau, T, &ldt);
}

template <>
void cpu_larft<rocblas_float_complex>(rocblas_direct directR,
                                      rocblas_storev storevR,
                                      rocblas_int n,
                                      rocblas_int k,
                                      rocblas_float_complex* V,
                                      rocblas_int ldv,
                                      rocblas_float_complex* tau,
                                      rocblas_float_complex* T,
                                      rocblas_int ldt)
{
    char direct = rocblas2char_direct(directR);
    char storev = rocblas2char_storev(storevR);
    clarft_(&direct, &storev, &n, &k, V, &ldv, tau, T, &ldt);
}

template <>
void cpu_larft<rocblas_double_complex>(rocblas_direct directR,
                                       rocblas_storev storevR,
                                       rocblas_int n,
                                       rocblas_int k,
                                       rocblas_double_complex* V,
                                       rocblas_int ldv,
                                       rocblas_double_complex* tau,
                                       rocblas_double_complex* T,
                                       rocblas_int ldt)
{
    char direct = rocblas2char_direct(directR);
    char storev = rocblas2char_storev(storevR);
    zlarft_(&direct, &storev, &n, &k, V, &ldv, tau, T, &ldt);
}

// larfb

template <>
void cpu_larfb<float>(rocblas_side sideR,
                      rocblas_operation transR,
                      rocblas_direct directR,
                      rocblas_storev storevR,
                      rocblas_int m,
                      rocblas_int n,
                      rocblas_int k,
                      float* V,
                      rocblas_int ldv,
                      float* T,
                      rocblas_int ldt,
                      float* A,
                      rocblas_int lda,
                      float* W,
                      rocblas_int ldw)
{
    char side = rocblas2char_side(sideR);
    char trans = rocblas2char_operation(transR);
    char direct = rocblas2char_direct(directR);
    char storev = rocblas2char_storev(storevR);
    slarfb_(&side, &trans, &direct, &storev, &m, &n, &k, V, &ldv, T, &ldt, A, &lda, W, &ldw);
}

template <>
void cpu_larfb<double>(rocblas_side sideR,
                       rocblas_operation transR,
                       rocblas_direct directR,
                       rocblas_storev storevR,
                       rocblas_int m,
                       rocblas_int n,
                       rocblas_int k,
                       double* V,
                       rocblas_int ldv,
                       double* T,
                       rocblas_int ldt,
                       double* A,
                       rocblas_int lda,
                       double* W,
                       rocblas_int ldw)
{
    char side = rocblas2char_side(sideR);
    char trans = rocblas2char_operation(transR);
    char direct = rocblas2char_direct(directR);
    char storev = rocblas2char_storev(storevR);
    dlarfb_(&side, &trans, &direct, &storev, &m, &n, &k, V, &ldv, T, &ldt, A, &lda, W, &ldw);
}

template <>
void cpu_larfb<rocblas_float_complex>(rocblas_side sideR,
                                      rocblas_operation transR,
                                      rocblas_direct directR,
                                      rocblas_storev storevR,
                                      rocblas_int m,
                                      rocblas_int n,
                                      rocblas_int k,
                                      rocblas_float_complex* V,
                                      rocblas_int ldv,
                                      rocblas_float_complex* T,
                                      rocblas_int ldt,
                                      rocblas_float_complex* A,
                                      rocblas_int lda,
                                      rocblas_float_complex* W,
                                      rocblas_int ldw)
{
    char side = rocblas2char_side(sideR);
    char trans = rocblas2char_operation(transR);
    char direct = rocblas2char_direct(directR);
    char storev = rocblas2char_storev(storevR);
    clarfb_(&side, &trans, &direct, &storev, &m, &n, &k, V, &ldv, T, &ldt, A, &lda, W, &ldw);
}

template <>
void cpu_larfb<rocblas_double_complex>(rocblas_side sideR,
                                       rocblas_operation transR,
                                       rocblas_direct directR,
                                       rocblas_storev storevR,
                                       rocblas_int m,
                                       rocblas_int n,
                                       rocblas_int k,
                                       rocblas_double_complex* V,
                                       rocblas_int ldv,
                                       rocblas_double_complex* T,
                                       rocblas_int ldt,
                                       rocblas_double_complex* A,
                                       rocblas_int lda,
                                       rocblas_double_complex* W,
                                       rocblas_int ldw)
{
    char side = rocblas2char_side(sideR);
    char trans = rocblas2char_operation(transR);
    char direct = rocblas2char_direct(directR);
    char storev = rocblas2char_storev(storevR);
    zlarfb_(&side, &trans, &direct, &storev, &m, &n, &k, V, &ldv, T, &ldt, A, &lda, W, &ldw);
}

// lasr

template <>
void cpu_lasr<float>(rocblas_side sideR,
                     rocblas_pivot pivotR,
                     rocblas_direct directR,
                     rocblas_int m,
                     rocblas_int n,
                     float* C,
                     float* S,
                     float* A,
                     rocblas_int lda)
{
    char side = rocblas2char_side(sideR);
    char pivot = rocblas2char_pivot(pivotR);
    char direct = rocblas2char_direct(directR);
    slasr_(&side, &pivot, &direct, &m, &n, C, S, A, &lda);
}

template <>
void cpu_lasr<double>(rocblas_side sideR,
                      rocblas_pivot pivotR,
                      rocblas_direct directR,
                      rocblas_int m,
                      rocblas_int n,
                      double* C,
                      double* S,
                      double* A,
                      rocblas_int lda)
{
    char side = rocblas2char_side(sideR);
    char pivot = rocblas2char_pivot(pivotR);
    char direct = rocblas2char_direct(directR);
    dlasr_(&side, &pivot, &direct, &m, &n, C, S, A, &lda);
}

template <>
void cpu_lasr<rocblas_float_complex>(rocblas_side sideR,
                                     rocblas_pivot pivotR,
                                     rocblas_direct directR,
                                     rocblas_int m,
                                     rocblas_int n,
                                     float* C,
                                     float* S,
                                     rocblas_float_complex* A,
                                     rocblas_int lda)
{
    char side = rocblas2char_side(sideR);
    char pivot = rocblas2char_pivot(pivotR);
    char direct = rocblas2char_direct(directR);
    clasr_(&side, &pivot, &direct, &m, &n, C, S, A, &lda);
}

template <>
void cpu_lasr<rocblas_double_complex>(rocblas_side sideR,
                                      rocblas_pivot pivotR,
                                      rocblas_direct directR,
                                      rocblas_int m,
                                      rocblas_int n,
                                      double* C,
                                      double* S,
                                      rocblas_double_complex* A,
                                      rocblas_int lda)
{
    char side = rocblas2char_side(sideR);
    char pivot = rocblas2char_pivot(pivotR);
    char direct = rocblas2char_direct(directR);
    zlasr_(&side, &pivot, &direct, &m, &n, C, S, A, &lda);
}

// lauum
template <>
void cpu_lauum(rocblas_fill uploR, rocblas_int n, float* A, rocblas_int lda)
{
    rocblas_int info;
    char uplo = rocblas2char_fill(uploR);
    slauum_(&uplo, &n, A, &lda, &info);
}

template <>
void cpu_lauum(rocblas_fill uploR, rocblas_int n, double* A, rocblas_int lda)
{
    rocblas_int info;
    char uplo = rocblas2char_fill(uploR);
    dlauum_(&uplo, &n, A, &lda, &info);
}

template <>
void cpu_lauum(rocblas_fill uploR, rocblas_int n, rocblas_float_complex* A, rocblas_int lda)
{
    rocblas_int info;
    char uplo = rocblas2char_fill(uploR);
    clauum_(&uplo, &n, A, &lda, &info);
}

template <>
void cpu_lauum(rocblas_fill uploR, rocblas_int n, rocblas_double_complex* A, rocblas_int lda)
{
    rocblas_int info;
    char uplo = rocblas2char_fill(uploR);
    zlauum_(&uplo, &n, A, &lda, &info);
}

// bdsqr
template <>
void cpu_bdsqr(rocblas_fill uplo,
               rocblas_int n,
               rocblas_int nv,
               rocblas_int nu,
               rocblas_int nc,
               float* D,
               float* E,
               float* V,
               rocblas_int ldv,
               float* U,
               rocblas_int ldu,
               float* C,
               rocblas_int ldc,
               float* work,
               rocblas_int* info)
{
    char uploC = (uplo == rocblas_fill_upper) ? 'U' : 'L';
    sbdsqr_(&uploC, &n, &nv, &nu, &nc, D, E, V, &ldv, U, &ldu, C, &ldc, work, info);
}

template <>
void cpu_bdsqr(rocblas_fill uplo,
               rocblas_int n,
               rocblas_int nv,
               rocblas_int nu,
               rocblas_int nc,
               double* D,
               double* E,
               double* V,
               rocblas_int ldv,
               double* U,
               rocblas_int ldu,
               double* C,
               rocblas_int ldc,
               double* work,
               rocblas_int* info)
{
    char uploC = (uplo == rocblas_fill_upper) ? 'U' : 'L';
    dbdsqr_(&uploC, &n, &nv, &nu, &nc, D, E, V, &ldv, U, &ldu, C, &ldc, work, info);
}

template <>
void cpu_bdsqr(rocblas_fill uplo,
               rocblas_int n,
               rocblas_int nv,
               rocblas_int nu,
               rocblas_int nc,
               float* D,
               float* E,
               rocblas_float_complex* V,
               rocblas_int ldv,
               rocblas_float_complex* U,
               rocblas_int ldu,
               rocblas_float_complex* C,
               rocblas_int ldc,
               float* work,
               rocblas_int* info)
{
    char uploC = (uplo == rocblas_fill_upper) ? 'U' : 'L';
    cbdsqr_(&uploC, &n, &nv, &nu, &nc, D, E, V, &ldv, U, &ldu, C, &ldc, work, info);
}

template <>
void cpu_bdsqr(rocblas_fill uplo,
               rocblas_int n,
               rocblas_int nv,
               rocblas_int nu,
               rocblas_int nc,
               double* D,
               double* E,
               rocblas_double_complex* V,
               rocblas_int ldv,
               rocblas_double_complex* U,
               rocblas_int ldu,
               rocblas_double_complex* C,
               rocblas_int ldc,
               double* work,
               rocblas_int* info)
{
    char uploC = (uplo == rocblas_fill_upper) ? 'U' : 'L';
    zbdsqr_(&uploC, &n, &nv, &nu, &nc, D, E, V, &ldv, U, &ldu, C, &ldc, work, info);
}

// gesvd
template <>
void cpu_gesvd(rocblas_svect leftv,
               rocblas_svect rightv,
               rocblas_int m,
               rocblas_int n,
               float* A,
               rocblas_int lda,
               float* S,
               float* U,
               rocblas_int ldu,
               float* V,
               rocblas_int ldv,
               float* work,
               rocblas_int lwork,
               float* rwork,
               rocblas_int* info)
{
    char jobu = rocblas2char_svect(leftv);
    char jobv = rocblas2char_svect(rightv);
    sgesvd_(&jobu, &jobv, &m, &n, A, &lda, S, U, &ldu, V, &ldv, work, &lwork, info);
}

template <>
void cpu_gesvd(rocblas_svect leftv,
               rocblas_svect rightv,
               rocblas_int m,
               rocblas_int n,
               double* A,
               rocblas_int lda,
               double* S,
               double* U,
               rocblas_int ldu,
               double* V,
               rocblas_int ldv,
               double* work,
               rocblas_int lwork,
               double* rwork,
               rocblas_int* info)
{
    char jobu = rocblas2char_svect(leftv);
    char jobv = rocblas2char_svect(rightv);
    dgesvd_(&jobu, &jobv, &m, &n, A, &lda, S, U, &ldu, V, &ldv, work, &lwork, info);
}

template <>
void cpu_gesvd(rocblas_svect leftv,
               rocblas_svect rightv,
               rocblas_int m,
               rocblas_int n,
               rocblas_float_complex* A,
               rocblas_int lda,
               float* S,
               rocblas_float_complex* U,
               rocblas_int ldu,
               rocblas_float_complex* V,
               rocblas_int ldv,
               rocblas_float_complex* work,
               rocblas_int lwork,
               float* rwork,
               rocblas_int* info)
{
    char jobu = rocblas2char_svect(leftv);
    char jobv = rocblas2char_svect(rightv);
    cgesvd_(&jobu, &jobv, &m, &n, A, &lda, S, U, &ldu, V, &ldv, work, &lwork, rwork, info);
}

template <>
void cpu_gesvd(rocblas_svect leftv,
               rocblas_svect rightv,
               rocblas_int m,
               rocblas_int n,
               rocblas_double_complex* A,
               rocblas_int lda,
               double* S,
               rocblas_double_complex* U,
               rocblas_int ldu,
               rocblas_double_complex* V,
               rocblas_int ldv,
               rocblas_double_complex* work,
               rocblas_int lwork,
               double* rwork,
               rocblas_int* info)
{
    char jobu = rocblas2char_svect(leftv);
    char jobv = rocblas2char_svect(rightv);
    zgesvd_(&jobu, &jobv, &m, &n, A, &lda, S, U, &ldu, V, &ldv, work, &lwork, rwork, info);
}

// gesvdx
template <>
void cpu_gesvdx(rocblas_svect leftv,
                rocblas_svect rightv,
                rocblas_srange srange,
                rocblas_int m,
                rocblas_int n,
                float* A,
                rocblas_int lda,
                float vl,
                float vu,
                rocblas_int il,
                rocblas_int iu,
                rocblas_int* nsv,
                float* S,
                float* U,
                rocblas_int ldu,
                float* V,
                rocblas_int ldv,
                float* work,
                rocblas_int lwork,
                float* rwork,
                rocblas_int* iwork,
                rocblas_int* info)
{
    char jobu = rocblas2char_svect(leftv, true);
    char jobv = rocblas2char_svect(rightv, true);
    char srangeC = rocblas2char_srange(srange);
    sgesvdx_(&jobu, &jobv, &srangeC, &m, &n, A, &lda, &vl, &vu, &il, &iu, nsv, S, U, &ldu, V, &ldv,
             work, &lwork, iwork, info);
}

template <>
void cpu_gesvdx(rocblas_svect leftv,
                rocblas_svect rightv,
                rocblas_srange srange,
                rocblas_int m,
                rocblas_int n,
                double* A,
                rocblas_int lda,
                double vl,
                double vu,
                rocblas_int il,
                rocblas_int iu,
                rocblas_int* nsv,
                double* S,
                double* U,
                rocblas_int ldu,
                double* V,
                rocblas_int ldv,
                double* work,
                rocblas_int lwork,
                double* rwork,
                rocblas_int* iwork,
                rocblas_int* info)
{
    char jobu = rocblas2char_svect(leftv, true);
    char jobv = rocblas2char_svect(rightv, true);
    char srangeC = rocblas2char_srange(srange);
    dgesvdx_(&jobu, &jobv, &srangeC, &m, &n, A, &lda, &vl, &vu, &il, &iu, nsv, S, U, &ldu, V, &ldv,
             work, &lwork, iwork, info);
}

template <>
void cpu_gesvdx(rocblas_svect leftv,
                rocblas_svect rightv,
                rocblas_srange srange,
                rocblas_int m,
                rocblas_int n,
                rocblas_float_complex* A,
                rocblas_int lda,
                float vl,
                float vu,
                rocblas_int il,
                rocblas_int iu,
                rocblas_int* nsv,
                float* S,
                rocblas_float_complex* U,
                rocblas_int ldu,
                rocblas_float_complex* V,
                rocblas_int ldv,
                rocblas_float_complex* work,
                rocblas_int lwork,
                float* rwork,
                rocblas_int* iwork,
                rocblas_int* info)
{
    char jobu = rocblas2char_svect(leftv, true);
    char jobv = rocblas2char_svect(rightv, true);
    char srangeC = rocblas2char_srange(srange);
    cgesvdx_(&jobu, &jobv, &srangeC, &m, &n, A, &lda, &vl, &vu, &il, &iu, nsv, S, U, &ldu, V, &ldv,
             work, &lwork, rwork, iwork, info);
}

template <>
void cpu_gesvdx(rocblas_svect leftv,
                rocblas_svect rightv,
                rocblas_srange srange,
                rocblas_int m,
                rocblas_int n,
                rocblas_double_complex* A,
                rocblas_int lda,
                double vl,
                double vu,
                rocblas_int il,
                rocblas_int iu,
                rocblas_int* nsv,
                double* S,
                rocblas_double_complex* U,
                rocblas_int ldu,
                rocblas_double_complex* V,
                rocblas_int ldv,
                rocblas_double_complex* work,
                rocblas_int lwork,
                double* rwork,
                rocblas_int* iwork,
                rocblas_int* info)
{
    char jobu = rocblas2char_svect(leftv, true);
    char jobv = rocblas2char_svect(rightv, true);
    char srangeC = rocblas2char_srange(srange);
    zgesvdx_(&jobu, &jobv, &srangeC, &m, &n, A, &lda, &vl, &vu, &il, &iu, nsv, S, U, &ldu, V, &ldv,
             work, &lwork, rwork, iwork, info);
}

// latrd
template <>
void cpu_latrd<float, float>(rocblas_fill uplo,
                             rocblas_int n,
                             rocblas_int k,
                             float* A,
                             rocblas_int lda,
                             float* E,
                             float* tau,
                             float* W,
                             rocblas_int ldw)
{
    char uploC = rocblas2char_fill(uplo);
    slatrd_(&uploC, &n, &k, A, &lda, E, tau, W, &ldw);
}

template <>
void cpu_latrd<double, double>(rocblas_fill uplo,
                               rocblas_int n,
                               rocblas_int k,
                               double* A,
                               rocblas_int lda,
                               double* E,
                               double* tau,
                               double* W,
                               rocblas_int ldw)
{
    char uploC = rocblas2char_fill(uplo);
    dlatrd_(&uploC, &n, &k, A, &lda, E, tau, W, &ldw);
}

template <>
void cpu_latrd<rocblas_float_complex, float>(rocblas_fill uplo,
                                             rocblas_int n,
                                             rocblas_int k,
                                             rocblas_float_complex* A,
                                             rocblas_int lda,
                                             float* E,
                                             rocblas_float_complex* tau,
                                             rocblas_float_complex* W,
                                             rocblas_int ldw)
{
    char uploC = rocblas2char_fill(uplo);
    clatrd_(&uploC, &n, &k, A, &lda, E, tau, W, &ldw);
}

template <>
void cpu_latrd<rocblas_double_complex, double>(rocblas_fill uplo,
                                               rocblas_int n,
                                               rocblas_int k,
                                               rocblas_double_complex* A,
                                               rocblas_int lda,
                                               double* E,
                                               rocblas_double_complex* tau,
                                               rocblas_double_complex* W,
                                               rocblas_int ldw)
{
    char uploC = rocblas2char_fill(uplo);
    zlatrd_(&uploC, &n, &k, A, &lda, E, tau, W, &ldw);
}

// labrd
template <>
void cpu_labrd<float, float>(rocblas_int m,
                             rocblas_int n,
                             rocblas_int nb,
                             float* A,
                             rocblas_int lda,
                             float* D,
                             float* E,
                             float* tauq,
                             float* taup,
                             float* X,
                             rocblas_int ldx,
                             float* Y,
                             rocblas_int ldy)
{
    slabrd_(&m, &n, &nb, A, &lda, D, E, tauq, taup, X, &ldx, Y, &ldy);
}

template <>
void cpu_labrd<double, double>(rocblas_int m,
                               rocblas_int n,
                               rocblas_int nb,
                               double* A,
                               rocblas_int lda,
                               double* D,
                               double* E,
                               double* tauq,
                               double* taup,
                               double* X,
                               rocblas_int ldx,
                               double* Y,
                               rocblas_int ldy)
{
    dlabrd_(&m, &n, &nb, A, &lda, D, E, tauq, taup, X, &ldx, Y, &ldy);
}

template <>
void cpu_labrd<rocblas_float_complex, float>(rocblas_int m,
                                             rocblas_int n,
                                             rocblas_int nb,
                                             rocblas_float_complex* A,
                                             rocblas_int lda,
                                             float* D,
                                             float* E,
                                             rocblas_float_complex* tauq,
                                             rocblas_float_complex* taup,
                                             rocblas_float_complex* X,
                                             rocblas_int ldx,
                                             rocblas_float_complex* Y,
                                             rocblas_int ldy)
{
    clabrd_(&m, &n, &nb, A, &lda, D, E, tauq, taup, X, &ldx, Y, &ldy);
}

template <>
void cpu_labrd<rocblas_double_complex, double>(rocblas_int m,
                                               rocblas_int n,
                                               rocblas_int nb,
                                               rocblas_double_complex* A,
                                               rocblas_int lda,
                                               double* D,
                                               double* E,
                                               rocblas_double_complex* tauq,
                                               rocblas_double_complex* taup,
                                               rocblas_double_complex* X,
                                               rocblas_int ldx,
                                               rocblas_double_complex* Y,
                                               rocblas_int ldy)
{
    zlabrd_(&m, &n, &nb, A, &lda, D, E, tauq, taup, X, &ldx, Y, &ldy);
}

// orgqr & ungqr
template <>
void cpu_orgqr_ungqr<float>(rocblas_int m,
                            rocblas_int n,
                            rocblas_int k,
                            float* A,
                            rocblas_int lda,
                            float* ipiv,
                            float* work,
                            rocblas_int lwork)
{
    int info;
    sorgqr_(&m, &n, &k, A, &lda, ipiv, work, &lwork, &info);
}

template <>
void cpu_orgqr_ungqr<double>(rocblas_int m,
                             rocblas_int n,
                             rocblas_int k,
                             double* A,
                             rocblas_int lda,
                             double* ipiv,
                             double* work,
                             rocblas_int lwork)
{
    int info;
    dorgqr_(&m, &n, &k, A, &lda, ipiv, work, &lwork, &info);
}

template <>
void cpu_orgqr_ungqr<rocblas_float_complex>(rocblas_int m,
                                            rocblas_int n,
                                            rocblas_int k,
                                            rocblas_float_complex* A,
                                            rocblas_int lda,
                                            rocblas_float_complex* ipiv,
                                            rocblas_float_complex* work,
                                            rocblas_int lwork)
{
    int info;
    cungqr_(&m, &n, &k, A, &lda, ipiv, work, &lwork, &info);
}

template <>
void cpu_orgqr_ungqr<rocblas_double_complex>(rocblas_int m,
                                             rocblas_int n,
                                             rocblas_int k,
                                             rocblas_double_complex* A,
                                             rocblas_int lda,
                                             rocblas_double_complex* ipiv,
                                             rocblas_double_complex* work,
                                             rocblas_int lwork)
{
    int info;
    zungqr_(&m, &n, &k, A, &lda, ipiv, work, &lwork, &info);
}

// org2r & ung2r
template <>
void cpu_org2r_ung2r<float>(rocblas_int m,
                            rocblas_int n,
                            rocblas_int k,
                            float* A,
                            rocblas_int lda,
                            float* ipiv,
                            float* work)
{
    int info;
    sorg2r_(&m, &n, &k, A, &lda, ipiv, work, &info);
}

template <>
void cpu_org2r_ung2r<double>(rocblas_int m,
                             rocblas_int n,
                             rocblas_int k,
                             double* A,
                             rocblas_int lda,
                             double* ipiv,
                             double* work)
{
    int info;
    dorg2r_(&m, &n, &k, A, &lda, ipiv, work, &info);
}

template <>
void cpu_org2r_ung2r<rocblas_float_complex>(rocblas_int m,
                                            rocblas_int n,
                                            rocblas_int k,
                                            rocblas_float_complex* A,
                                            rocblas_int lda,
                                            rocblas_float_complex* ipiv,
                                            rocblas_float_complex* work)
{
    int info;
    cung2r_(&m, &n, &k, A, &lda, ipiv, work, &info);
}

template <>
void cpu_org2r_ung2r<rocblas_double_complex>(rocblas_int m,
                                             rocblas_int n,
                                             rocblas_int k,
                                             rocblas_double_complex* A,
                                             rocblas_int lda,
                                             rocblas_double_complex* ipiv,
                                             rocblas_double_complex* work)
{
    int info;
    zung2r_(&m, &n, &k, A, &lda, ipiv, work, &info);
}

// orglq & unglq
template <>
void cpu_orglq_unglq<float>(rocblas_int m,
                            rocblas_int n,
                            rocblas_int k,
                            float* A,
                            rocblas_int lda,
                            float* ipiv,
                            float* work,
                            rocblas_int lwork)
{
    int info;
    sorglq_(&m, &n, &k, A, &lda, ipiv, work, &lwork, &info);
}

template <>
void cpu_orglq_unglq<double>(rocblas_int m,
                             rocblas_int n,
                             rocblas_int k,
                             double* A,
                             rocblas_int lda,
                             double* ipiv,
                             double* work,
                             rocblas_int lwork)
{
    int info;
    dorglq_(&m, &n, &k, A, &lda, ipiv, work, &lwork, &info);
}

template <>
void cpu_orglq_unglq<rocblas_float_complex>(rocblas_int m,
                                            rocblas_int n,
                                            rocblas_int k,
                                            rocblas_float_complex* A,
                                            rocblas_int lda,
                                            rocblas_float_complex* ipiv,
                                            rocblas_float_complex* work,
                                            rocblas_int lwork)
{
    int info;
    cunglq_(&m, &n, &k, A, &lda, ipiv, work, &lwork, &info);
}

template <>
void cpu_orglq_unglq<rocblas_double_complex>(rocblas_int m,
                                             rocblas_int n,
                                             rocblas_int k,
                                             rocblas_double_complex* A,
                                             rocblas_int lda,
                                             rocblas_double_complex* ipiv,
                                             rocblas_double_complex* work,
                                             rocblas_int lwork)
{
    int info;
    zunglq_(&m, &n, &k, A, &lda, ipiv, work, &lwork, &info);
}

// orgl2 & ungl2
template <>
void cpu_orgl2_ungl2<float>(rocblas_int m,
                            rocblas_int n,
                            rocblas_int k,
                            float* A,
                            rocblas_int lda,
                            float* ipiv,
                            float* work)
{
    int info;
    sorgl2_(&m, &n, &k, A, &lda, ipiv, work, &info);
}

template <>
void cpu_orgl2_ungl2<double>(rocblas_int m,
                             rocblas_int n,
                             rocblas_int k,
                             double* A,
                             rocblas_int lda,
                             double* ipiv,
                             double* work)
{
    int info;
    dorgl2_(&m, &n, &k, A, &lda, ipiv, work, &info);
}

template <>
void cpu_orgl2_ungl2<rocblas_float_complex>(rocblas_int m,
                                            rocblas_int n,
                                            rocblas_int k,
                                            rocblas_float_complex* A,
                                            rocblas_int lda,
                                            rocblas_float_complex* ipiv,
                                            rocblas_float_complex* work)
{
    int info;
    cungl2_(&m, &n, &k, A, &lda, ipiv, work, &info);
}

template <>
void cpu_orgl2_ungl2<rocblas_double_complex>(rocblas_int m,
                                             rocblas_int n,
                                             rocblas_int k,
                                             rocblas_double_complex* A,
                                             rocblas_int lda,
                                             rocblas_double_complex* ipiv,
                                             rocblas_double_complex* work)
{
    int info;
    zungl2_(&m, &n, &k, A, &lda, ipiv, work, &info);
}

// orgql & ungql
template <>
void cpu_orgql_ungql<float>(rocblas_int m,
                            rocblas_int n,
                            rocblas_int k,
                            float* A,
                            rocblas_int lda,
                            float* ipiv,
                            float* work,
                            rocblas_int lwork)
{
    int info;
    sorgql_(&m, &n, &k, A, &lda, ipiv, work, &lwork, &info);
}

template <>
void cpu_orgql_ungql<double>(rocblas_int m,
                             rocblas_int n,
                             rocblas_int k,
                             double* A,
                             rocblas_int lda,
                             double* ipiv,
                             double* work,
                             rocblas_int lwork)
{
    int info;
    dorgql_(&m, &n, &k, A, &lda, ipiv, work, &lwork, &info);
}

template <>
void cpu_orgql_ungql<rocblas_float_complex>(rocblas_int m,
                                            rocblas_int n,
                                            rocblas_int k,
                                            rocblas_float_complex* A,
                                            rocblas_int lda,
                                            rocblas_float_complex* ipiv,
                                            rocblas_float_complex* work,
                                            rocblas_int lwork)
{
    int info;
    cungql_(&m, &n, &k, A, &lda, ipiv, work, &lwork, &info);
}

template <>
void cpu_orgql_ungql<rocblas_double_complex>(rocblas_int m,
                                             rocblas_int n,
                                             rocblas_int k,
                                             rocblas_double_complex* A,
                                             rocblas_int lda,
                                             rocblas_double_complex* ipiv,
                                             rocblas_double_complex* work,
                                             rocblas_int lwork)
{
    int info;
    zungql_(&m, &n, &k, A, &lda, ipiv, work, &lwork, &info);
}

// org2l & ung2l
template <>
void cpu_org2l_ung2l<float>(rocblas_int m,
                            rocblas_int n,
                            rocblas_int k,
                            float* A,
                            rocblas_int lda,
                            float* ipiv,
                            float* work)
{
    int info;
    sorg2l_(&m, &n, &k, A, &lda, ipiv, work, &info);
}

template <>
void cpu_org2l_ung2l<double>(rocblas_int m,
                             rocblas_int n,
                             rocblas_int k,
                             double* A,
                             rocblas_int lda,
                             double* ipiv,
                             double* work)
{
    int info;
    dorg2l_(&m, &n, &k, A, &lda, ipiv, work, &info);
}

template <>
void cpu_org2l_ung2l<rocblas_float_complex>(rocblas_int m,
                                            rocblas_int n,
                                            rocblas_int k,
                                            rocblas_float_complex* A,
                                            rocblas_int lda,
                                            rocblas_float_complex* ipiv,
                                            rocblas_float_complex* work)
{
    int info;
    cung2l_(&m, &n, &k, A, &lda, ipiv, work, &info);
}

template <>
void cpu_org2l_ung2l<rocblas_double_complex>(rocblas_int m,
                                             rocblas_int n,
                                             rocblas_int k,
                                             rocblas_double_complex* A,
                                             rocblas_int lda,
                                             rocblas_double_complex* ipiv,
                                             rocblas_double_complex* work)
{
    int info;
    zung2l_(&m, &n, &k, A, &lda, ipiv, work, &info);
}

// orgbr & ungbr
template <>
void cpu_orgbr_ungbr<float>(rocblas_storev storev,
                            rocblas_int m,
                            rocblas_int n,
                            rocblas_int k,
                            float* A,
                            rocblas_int lda,
                            float* Ipiv,
                            float* work,
                            rocblas_int size_w)
{
    int info;
    char vect;
    if(storev == rocblas_column_wise)
        vect = 'Q';
    else
        vect = 'P';
    sorgbr_(&vect, &m, &n, &k, A, &lda, Ipiv, work, &size_w, &info);
}

template <>
void cpu_orgbr_ungbr<double>(rocblas_storev storev,
                             rocblas_int m,
                             rocblas_int n,
                             rocblas_int k,
                             double* A,
                             rocblas_int lda,
                             double* Ipiv,
                             double* work,
                             rocblas_int size_w)
{
    int info;
    char vect;
    if(storev == rocblas_column_wise)
        vect = 'Q';
    else
        vect = 'P';
    dorgbr_(&vect, &m, &n, &k, A, &lda, Ipiv, work, &size_w, &info);
}

template <>
void cpu_orgbr_ungbr<rocblas_float_complex>(rocblas_storev storev,
                                            rocblas_int m,
                                            rocblas_int n,
                                            rocblas_int k,
                                            rocblas_float_complex* A,
                                            rocblas_int lda,
                                            rocblas_float_complex* Ipiv,
                                            rocblas_float_complex* work,
                                            rocblas_int size_w)
{
    int info;
    char vect;
    if(storev == rocblas_column_wise)
        vect = 'Q';
    else
        vect = 'P';
    cungbr_(&vect, &m, &n, &k, A, &lda, Ipiv, work, &size_w, &info);
}

template <>
void cpu_orgbr_ungbr<rocblas_double_complex>(rocblas_storev storev,
                                             rocblas_int m,
                                             rocblas_int n,
                                             rocblas_int k,
                                             rocblas_double_complex* A,
                                             rocblas_int lda,
                                             rocblas_double_complex* Ipiv,
                                             rocblas_double_complex* work,
                                             rocblas_int size_w)
{
    int info;
    char vect;
    if(storev == rocblas_column_wise)
        vect = 'Q';
    else
        vect = 'P';
    zungbr_(&vect, &m, &n, &k, A, &lda, Ipiv, work, &size_w, &info);
}

// orgtr & ungtr
template <>
void cpu_orgtr_ungtr<float>(rocblas_fill uplo,
                            rocblas_int n,
                            float* A,
                            rocblas_int lda,
                            float* Ipiv,
                            float* work,
                            rocblas_int size_w)
{
    int info;
    char uploC = rocblas2char_fill(uplo);
    sorgtr_(&uploC, &n, A, &lda, Ipiv, work, &size_w, &info);
}

template <>
void cpu_orgtr_ungtr<double>(rocblas_fill uplo,
                             rocblas_int n,
                             double* A,
                             rocblas_int lda,
                             double* Ipiv,
                             double* work,
                             rocblas_int size_w)
{
    int info;
    char uploC = rocblas2char_fill(uplo);
    dorgtr_(&uploC, &n, A, &lda, Ipiv, work, &size_w, &info);
}

template <>
void cpu_orgtr_ungtr<rocblas_float_complex>(rocblas_fill uplo,
                                            rocblas_int n,
                                            rocblas_float_complex* A,
                                            rocblas_int lda,
                                            rocblas_float_complex* Ipiv,
                                            rocblas_float_complex* work,
                                            rocblas_int size_w)
{
    int info;
    char uploC = rocblas2char_fill(uplo);
    cungtr_(&uploC, &n, A, &lda, Ipiv, work, &size_w, &info);
}

template <>
void cpu_orgtr_ungtr<rocblas_double_complex>(rocblas_fill uplo,
                                             rocblas_int n,
                                             rocblas_double_complex* A,
                                             rocblas_int lda,
                                             rocblas_double_complex* Ipiv,
                                             rocblas_double_complex* work,
                                             rocblas_int size_w)
{
    int info;
    char uploC = rocblas2char_fill(uplo);
    zungtr_(&uploC, &n, A, &lda, Ipiv, work, &size_w, &info);
}

// ormqr & unmqr
template <>
void cpu_ormqr_unmqr<float>(rocblas_side side,
                            rocblas_operation trans,
                            rocblas_int m,
                            rocblas_int n,
                            rocblas_int k,
                            float* A,
                            rocblas_int lda,
                            float* ipiv,
                            float* C,
                            rocblas_int ldc,
                            float* work,
                            rocblas_int lwork)
{
    int info;
    char sideC = rocblas2char_side(side);
    char transC = rocblas2char_operation(trans);

    sormqr_(&sideC, &transC, &m, &n, &k, A, &lda, ipiv, C, &ldc, work, &lwork, &info);
}

template <>
void cpu_ormqr_unmqr<double>(rocblas_side side,
                             rocblas_operation trans,
                             rocblas_int m,
                             rocblas_int n,
                             rocblas_int k,
                             double* A,
                             rocblas_int lda,
                             double* ipiv,
                             double* C,
                             rocblas_int ldc,
                             double* work,
                             rocblas_int lwork)
{
    int info;
    char sideC = rocblas2char_side(side);
    char transC = rocblas2char_operation(trans);

    dormqr_(&sideC, &transC, &m, &n, &k, A, &lda, ipiv, C, &ldc, work, &lwork, &info);
}

template <>
void cpu_ormqr_unmqr<rocblas_float_complex>(rocblas_side side,
                                            rocblas_operation trans,
                                            rocblas_int m,
                                            rocblas_int n,
                                            rocblas_int k,
                                            rocblas_float_complex* A,
                                            rocblas_int lda,
                                            rocblas_float_complex* ipiv,
                                            rocblas_float_complex* C,
                                            rocblas_int ldc,
                                            rocblas_float_complex* work,
                                            rocblas_int lwork)
{
    int info;
    char sideC = rocblas2char_side(side);
    char transC = rocblas2char_operation(trans);

    cunmqr_(&sideC, &transC, &m, &n, &k, A, &lda, ipiv, C, &ldc, work, &lwork, &info);
}

template <>
void cpu_ormqr_unmqr<rocblas_double_complex>(rocblas_side side,
                                             rocblas_operation trans,
                                             rocblas_int m,
                                             rocblas_int n,
                                             rocblas_int k,
                                             rocblas_double_complex* A,
                                             rocblas_int lda,
                                             rocblas_double_complex* ipiv,
                                             rocblas_double_complex* C,
                                             rocblas_int ldc,
                                             rocblas_double_complex* work,
                                             rocblas_int lwork)
{
    int info;
    char sideC = rocblas2char_side(side);
    char transC = rocblas2char_operation(trans);

    zunmqr_(&sideC, &transC, &m, &n, &k, A, &lda, ipiv, C, &ldc, work, &lwork, &info);
}

// orm2r & unm2r
template <>
void cpu_orm2r_unm2r<float>(rocblas_side side,
                            rocblas_operation trans,
                            rocblas_int m,
                            rocblas_int n,
                            rocblas_int k,
                            float* A,
                            rocblas_int lda,
                            float* ipiv,
                            float* C,
                            rocblas_int ldc,
                            float* work)
{
    int info;
    char sideC = rocblas2char_side(side);
    char transC = rocblas2char_operation(trans);

    sorm2r_(&sideC, &transC, &m, &n, &k, A, &lda, ipiv, C, &ldc, work, &info);
}

template <>
void cpu_orm2r_unm2r<double>(rocblas_side side,
                             rocblas_operation trans,
                             rocblas_int m,
                             rocblas_int n,
                             rocblas_int k,
                             double* A,
                             rocblas_int lda,
                             double* ipiv,
                             double* C,
                             rocblas_int ldc,
                             double* work)
{
    int info;
    char sideC = rocblas2char_side(side);
    char transC = rocblas2char_operation(trans);

    dorm2r_(&sideC, &transC, &m, &n, &k, A, &lda, ipiv, C, &ldc, work, &info);
}

template <>
void cpu_orm2r_unm2r<rocblas_float_complex>(rocblas_side side,
                                            rocblas_operation trans,
                                            rocblas_int m,
                                            rocblas_int n,
                                            rocblas_int k,
                                            rocblas_float_complex* A,
                                            rocblas_int lda,
                                            rocblas_float_complex* ipiv,
                                            rocblas_float_complex* C,
                                            rocblas_int ldc,
                                            rocblas_float_complex* work)
{
    int info;
    char sideC = rocblas2char_side(side);
    char transC = rocblas2char_operation(trans);

    cunm2r_(&sideC, &transC, &m, &n, &k, A, &lda, ipiv, C, &ldc, work, &info);
}

template <>
void cpu_orm2r_unm2r<rocblas_double_complex>(rocblas_side side,
                                             rocblas_operation trans,
                                             rocblas_int m,
                                             rocblas_int n,
                                             rocblas_int k,
                                             rocblas_double_complex* A,
                                             rocblas_int lda,
                                             rocblas_double_complex* ipiv,
                                             rocblas_double_complex* C,
                                             rocblas_int ldc,
                                             rocblas_double_complex* work)
{
    int info;
    char sideC = rocblas2char_side(side);
    char transC = rocblas2char_operation(trans);

    zunm2r_(&sideC, &transC, &m, &n, &k, A, &lda, ipiv, C, &ldc, work, &info);
}

// ormlq & unmlq
template <>
void cpu_ormlq_unmlq<float>(rocblas_side side,
                            rocblas_operation trans,
                            rocblas_int m,
                            rocblas_int n,
                            rocblas_int k,
                            float* A,
                            rocblas_int lda,
                            float* ipiv,
                            float* C,
                            rocblas_int ldc,
                            float* work,
                            rocblas_int lwork)
{
    int info;
    char sideC = rocblas2char_side(side);
    char transC = rocblas2char_operation(trans);

    sormlq_(&sideC, &transC, &m, &n, &k, A, &lda, ipiv, C, &ldc, work, &lwork, &info);
}

template <>
void cpu_ormlq_unmlq<double>(rocblas_side side,
                             rocblas_operation trans,
                             rocblas_int m,
                             rocblas_int n,
                             rocblas_int k,
                             double* A,
                             rocblas_int lda,
                             double* ipiv,
                             double* C,
                             rocblas_int ldc,
                             double* work,
                             rocblas_int lwork)
{
    int info;
    char sideC = rocblas2char_side(side);
    char transC = rocblas2char_operation(trans);

    dormlq_(&sideC, &transC, &m, &n, &k, A, &lda, ipiv, C, &ldc, work, &lwork, &info);
}

template <>
void cpu_ormlq_unmlq<rocblas_float_complex>(rocblas_side side,
                                            rocblas_operation trans,
                                            rocblas_int m,
                                            rocblas_int n,
                                            rocblas_int k,
                                            rocblas_float_complex* A,
                                            rocblas_int lda,
                                            rocblas_float_complex* ipiv,
                                            rocblas_float_complex* C,
                                            rocblas_int ldc,
                                            rocblas_float_complex* work,
                                            rocblas_int lwork)
{
    int info;
    char sideC = rocblas2char_side(side);
    char transC = rocblas2char_operation(trans);

    cunmlq_(&sideC, &transC, &m, &n, &k, A, &lda, ipiv, C, &ldc, work, &lwork, &info);
}

template <>
void cpu_ormlq_unmlq<rocblas_double_complex>(rocblas_side side,
                                             rocblas_operation trans,
                                             rocblas_int m,
                                             rocblas_int n,
                                             rocblas_int k,
                                             rocblas_double_complex* A,
                                             rocblas_int lda,
                                             rocblas_double_complex* ipiv,
                                             rocblas_double_complex* C,
                                             rocblas_int ldc,
                                             rocblas_double_complex* work,
                                             rocblas_int lwork)
{
    int info;
    char sideC = rocblas2char_side(side);
    char transC = rocblas2char_operation(trans);

    zunmlq_(&sideC, &transC, &m, &n, &k, A, &lda, ipiv, C, &ldc, work, &lwork, &info);
}

// orml2 & unml2
template <>
void cpu_orml2_unml2<float>(rocblas_side side,
                            rocblas_operation trans,
                            rocblas_int m,
                            rocblas_int n,
                            rocblas_int k,
                            float* A,
                            rocblas_int lda,
                            float* ipiv,
                            float* C,
                            rocblas_int ldc,
                            float* work)
{
    int info;
    char sideC = rocblas2char_side(side);
    char transC = rocblas2char_operation(trans);

    sorml2_(&sideC, &transC, &m, &n, &k, A, &lda, ipiv, C, &ldc, work, &info);
}

template <>
void cpu_orml2_unml2<double>(rocblas_side side,
                             rocblas_operation trans,
                             rocblas_int m,
                             rocblas_int n,
                             rocblas_int k,
                             double* A,
                             rocblas_int lda,
                             double* ipiv,
                             double* C,
                             rocblas_int ldc,
                             double* work)
{
    int info;
    char sideC = rocblas2char_side(side);
    char transC = rocblas2char_operation(trans);

    dorml2_(&sideC, &transC, &m, &n, &k, A, &lda, ipiv, C, &ldc, work, &info);
}

template <>
void cpu_orml2_unml2<rocblas_float_complex>(rocblas_side side,
                                            rocblas_operation trans,
                                            rocblas_int m,
                                            rocblas_int n,
                                            rocblas_int k,
                                            rocblas_float_complex* A,
                                            rocblas_int lda,
                                            rocblas_float_complex* ipiv,
                                            rocblas_float_complex* C,
                                            rocblas_int ldc,
                                            rocblas_float_complex* work)
{
    int info;
    char sideC = rocblas2char_side(side);
    char transC = rocblas2char_operation(trans);

    cunml2_(&sideC, &transC, &m, &n, &k, A, &lda, ipiv, C, &ldc, work, &info);
}

template <>
void cpu_orml2_unml2<rocblas_double_complex>(rocblas_side side,
                                             rocblas_operation trans,
                                             rocblas_int m,
                                             rocblas_int n,
                                             rocblas_int k,
                                             rocblas_double_complex* A,
                                             rocblas_int lda,
                                             rocblas_double_complex* ipiv,
                                             rocblas_double_complex* C,
                                             rocblas_int ldc,
                                             rocblas_double_complex* work)
{
    int info;
    char sideC = rocblas2char_side(side);
    char transC = rocblas2char_operation(trans);

    zunml2_(&sideC, &transC, &m, &n, &k, A, &lda, ipiv, C, &ldc, work, &info);
}

// ormql & unmql
template <>
void cpu_ormql_unmql<float>(rocblas_side side,
                            rocblas_operation trans,
                            rocblas_int m,
                            rocblas_int n,
                            rocblas_int k,
                            float* A,
                            rocblas_int lda,
                            float* ipiv,
                            float* C,
                            rocblas_int ldc,
                            float* work,
                            rocblas_int lwork)
{
    int info;
    char sideC = rocblas2char_side(side);
    char transC = rocblas2char_operation(trans);

    sormql_(&sideC, &transC, &m, &n, &k, A, &lda, ipiv, C, &ldc, work, &lwork, &info);
}

template <>
void cpu_ormql_unmql<double>(rocblas_side side,
                             rocblas_operation trans,
                             rocblas_int m,
                             rocblas_int n,
                             rocblas_int k,
                             double* A,
                             rocblas_int lda,
                             double* ipiv,
                             double* C,
                             rocblas_int ldc,
                             double* work,
                             rocblas_int lwork)
{
    int info;
    char sideC = rocblas2char_side(side);
    char transC = rocblas2char_operation(trans);

    dormql_(&sideC, &transC, &m, &n, &k, A, &lda, ipiv, C, &ldc, work, &lwork, &info);
}

template <>
void cpu_ormql_unmql<rocblas_float_complex>(rocblas_side side,
                                            rocblas_operation trans,
                                            rocblas_int m,
                                            rocblas_int n,
                                            rocblas_int k,
                                            rocblas_float_complex* A,
                                            rocblas_int lda,
                                            rocblas_float_complex* ipiv,
                                            rocblas_float_complex* C,
                                            rocblas_int ldc,
                                            rocblas_float_complex* work,
                                            rocblas_int lwork)
{
    int info;
    char sideC = rocblas2char_side(side);
    char transC = rocblas2char_operation(trans);

    cunmql_(&sideC, &transC, &m, &n, &k, A, &lda, ipiv, C, &ldc, work, &lwork, &info);
}

template <>
void cpu_ormql_unmql<rocblas_double_complex>(rocblas_side side,
                                             rocblas_operation trans,
                                             rocblas_int m,
                                             rocblas_int n,
                                             rocblas_int k,
                                             rocblas_double_complex* A,
                                             rocblas_int lda,
                                             rocblas_double_complex* ipiv,
                                             rocblas_double_complex* C,
                                             rocblas_int ldc,
                                             rocblas_double_complex* work,
                                             rocblas_int lwork)
{
    int info;
    char sideC = rocblas2char_side(side);
    char transC = rocblas2char_operation(trans);

    zunmql_(&sideC, &transC, &m, &n, &k, A, &lda, ipiv, C, &ldc, work, &lwork, &info);
}

// orm2l & unm2l
template <>
void cpu_orm2l_unm2l<float>(rocblas_side side,
                            rocblas_operation trans,
                            rocblas_int m,
                            rocblas_int n,
                            rocblas_int k,
                            float* A,
                            rocblas_int lda,
                            float* ipiv,
                            float* C,
                            rocblas_int ldc,
                            float* work)
{
    int info;
    char sideC = rocblas2char_side(side);
    char transC = rocblas2char_operation(trans);

    sorm2l_(&sideC, &transC, &m, &n, &k, A, &lda, ipiv, C, &ldc, work, &info);
}

template <>
void cpu_orm2l_unm2l<double>(rocblas_side side,
                             rocblas_operation trans,
                             rocblas_int m,
                             rocblas_int n,
                             rocblas_int k,
                             double* A,
                             rocblas_int lda,
                             double* ipiv,
                             double* C,
                             rocblas_int ldc,
                             double* work)
{
    int info;
    char sideC = rocblas2char_side(side);
    char transC = rocblas2char_operation(trans);

    dorm2l_(&sideC, &transC, &m, &n, &k, A, &lda, ipiv, C, &ldc, work, &info);
}

template <>
void cpu_orm2l_unm2l<rocblas_float_complex>(rocblas_side side,
                                            rocblas_operation trans,
                                            rocblas_int m,
                                            rocblas_int n,
                                            rocblas_int k,
                                            rocblas_float_complex* A,
                                            rocblas_int lda,
                                            rocblas_float_complex* ipiv,
                                            rocblas_float_complex* C,
                                            rocblas_int ldc,
                                            rocblas_float_complex* work)
{
    int info;
    char sideC = rocblas2char_side(side);
    char transC = rocblas2char_operation(trans);

    cunm2l_(&sideC, &transC, &m, &n, &k, A, &lda, ipiv, C, &ldc, work, &info);
}

template <>
void cpu_orm2l_unm2l<rocblas_double_complex>(rocblas_side side,
                                             rocblas_operation trans,
                                             rocblas_int m,
                                             rocblas_int n,
                                             rocblas_int k,
                                             rocblas_double_complex* A,
                                             rocblas_int lda,
                                             rocblas_double_complex* ipiv,
                                             rocblas_double_complex* C,
                                             rocblas_int ldc,
                                             rocblas_double_complex* work)
{
    int info;
    char sideC = rocblas2char_side(side);
    char transC = rocblas2char_operation(trans);

    zunm2l_(&sideC, &transC, &m, &n, &k, A, &lda, ipiv, C, &ldc, work, &info);
}

// ormbr & unmbr
template <>
void cpu_ormbr_unmbr<float>(rocblas_storev storev,
                            rocblas_side side,
                            rocblas_operation trans,
                            rocblas_int m,
                            rocblas_int n,
                            rocblas_int k,
                            float* A,
                            rocblas_int lda,
                            float* ipiv,
                            float* C,
                            rocblas_int ldc,
                            float* work,
                            rocblas_int lwork)
{
    int info;
    char sideC = rocblas2char_side(side);
    char transC = rocblas2char_operation(trans);
    char vect;
    if(storev == rocblas_column_wise)
        vect = 'Q';
    else
        vect = 'P';

    sormbr_(&vect, &sideC, &transC, &m, &n, &k, A, &lda, ipiv, C, &ldc, work, &lwork, &info);
}

template <>
void cpu_ormbr_unmbr<double>(rocblas_storev storev,
                             rocblas_side side,
                             rocblas_operation trans,
                             rocblas_int m,
                             rocblas_int n,
                             rocblas_int k,
                             double* A,
                             rocblas_int lda,
                             double* ipiv,
                             double* C,
                             rocblas_int ldc,
                             double* work,
                             rocblas_int lwork)
{
    int info;
    char sideC = rocblas2char_side(side);
    char transC = rocblas2char_operation(trans);
    char vect;
    if(storev == rocblas_column_wise)
        vect = 'Q';
    else
        vect = 'P';

    dormbr_(&vect, &sideC, &transC, &m, &n, &k, A, &lda, ipiv, C, &ldc, work, &lwork, &info);
}

template <>
void cpu_ormbr_unmbr<rocblas_float_complex>(rocblas_storev storev,
                                            rocblas_side side,
                                            rocblas_operation trans,
                                            rocblas_int m,
                                            rocblas_int n,
                                            rocblas_int k,
                                            rocblas_float_complex* A,
                                            rocblas_int lda,
                                            rocblas_float_complex* ipiv,
                                            rocblas_float_complex* C,
                                            rocblas_int ldc,
                                            rocblas_float_complex* work,
                                            rocblas_int lwork)
{
    int info;
    char sideC = rocblas2char_side(side);
    char transC = rocblas2char_operation(trans);
    char vect;
    if(storev == rocblas_column_wise)
        vect = 'Q';
    else
        vect = 'P';

    cunmbr_(&vect, &sideC, &transC, &m, &n, &k, A, &lda, ipiv, C, &ldc, work, &lwork, &info);
}

template <>
void cpu_ormbr_unmbr<rocblas_double_complex>(rocblas_storev storev,
                                             rocblas_side side,
                                             rocblas_operation trans,
                                             rocblas_int m,
                                             rocblas_int n,
                                             rocblas_int k,
                                             rocblas_double_complex* A,
                                             rocblas_int lda,
                                             rocblas_double_complex* ipiv,
                                             rocblas_double_complex* C,
                                             rocblas_int ldc,
                                             rocblas_double_complex* work,
                                             rocblas_int lwork)
{
    int info;
    char sideC = rocblas2char_side(side);
    char transC = rocblas2char_operation(trans);
    char vect;
    if(storev == rocblas_column_wise)
        vect = 'Q';
    else
        vect = 'P';

    zunmbr_(&vect, &sideC, &transC, &m, &n, &k, A, &lda, ipiv, C, &ldc, work, &lwork, &info);
}

// ormtr & unmtr
template <>
void cpu_ormtr_unmtr<float>(rocblas_side side,
                            rocblas_fill uplo,
                            rocblas_operation trans,
                            rocblas_int m,
                            rocblas_int n,
                            float* A,
                            rocblas_int lda,
                            float* ipiv,
                            float* C,
                            rocblas_int ldc,
                            float* work,
                            rocblas_int lwork)
{
    int info;
    char sideC = rocblas2char_side(side);
    char uploC = rocblas2char_fill(uplo);
    char transC = rocblas2char_operation(trans);

    sormtr_(&sideC, &uploC, &transC, &m, &n, A, &lda, ipiv, C, &ldc, work, &lwork, &info);
}

template <>
void cpu_ormtr_unmtr<double>(rocblas_side side,
                             rocblas_fill uplo,
                             rocblas_operation trans,
                             rocblas_int m,
                             rocblas_int n,
                             double* A,
                             rocblas_int lda,
                             double* ipiv,
                             double* C,
                             rocblas_int ldc,
                             double* work,
                             rocblas_int lwork)
{
    int info;
    char sideC = rocblas2char_side(side);
    char uploC = rocblas2char_fill(uplo);
    char transC = rocblas2char_operation(trans);

    dormtr_(&sideC, &uploC, &transC, &m, &n, A, &lda, ipiv, C, &ldc, work, &lwork, &info);
}

template <>
void cpu_ormtr_unmtr<rocblas_float_complex>(rocblas_side side,
                                            rocblas_fill uplo,
                                            rocblas_operation trans,
                                            rocblas_int m,
                                            rocblas_int n,
                                            rocblas_float_complex* A,
                                            rocblas_int lda,
                                            rocblas_float_complex* ipiv,
                                            rocblas_float_complex* C,
                                            rocblas_int ldc,
                                            rocblas_float_complex* work,
                                            rocblas_int lwork)
{
    int info;
    char sideC = rocblas2char_side(side);
    char uploC = rocblas2char_fill(uplo);
    char transC = rocblas2char_operation(trans);

    cunmtr_(&sideC, &uploC, &transC, &m, &n, A, &lda, ipiv, C, &ldc, work, &lwork, &info);
}

template <>
void cpu_ormtr_unmtr<rocblas_double_complex>(rocblas_side side,
                                             rocblas_fill uplo,
                                             rocblas_operation trans,
                                             rocblas_int m,
                                             rocblas_int n,
                                             rocblas_double_complex* A,
                                             rocblas_int lda,
                                             rocblas_double_complex* ipiv,
                                             rocblas_double_complex* C,
                                             rocblas_int ldc,
                                             rocblas_double_complex* work,
                                             rocblas_int lwork)
{
    int info;
    char sideC = rocblas2char_side(side);
    char uploC = rocblas2char_fill(uplo);
    char transC = rocblas2char_operation(trans);

    zunmtr_(&sideC, &uploC, &transC, &m, &n, A, &lda, ipiv, C, &ldc, work, &lwork, &info);
}

// gemv
template <>
void cpu_gemv<float>(rocblas_operation transA,
                     rocblas_int m,
                     rocblas_int n,
                     float alpha,
                     float* A,
                     rocblas_int lda,
                     float* x,
                     rocblas_int incx,
                     float beta,
                     float* y,
                     rocblas_int incy)
{
    char transAC = rocblas2char_operation(transA);
    sgemv_(&transAC, &m, &n, &alpha, A, &lda, x, &incx, &beta, y, &incy);
}

template <>
void cpu_gemv<double>(rocblas_operation transA,
                      rocblas_int m,
                      rocblas_int n,
                      double alpha,
                      double* A,
                      rocblas_int lda,
                      double* x,
                      rocblas_int incx,
                      double beta,
                      double* y,
                      rocblas_int incy)
{
    char transAC = rocblas2char_operation(transA);
    dgemv_(&transAC, &m, &n, &alpha, A, &lda, x, &incx, &beta, y, &incy);
}

template <>
void cpu_gemv<rocblas_float_complex>(rocblas_operation transA,
                                     rocblas_int m,
                                     rocblas_int n,
                                     rocblas_float_complex alpha,
                                     rocblas_float_complex* A,
                                     rocblas_int lda,
                                     rocblas_float_complex* x,
                                     rocblas_int incx,
                                     rocblas_float_complex beta,
                                     rocblas_float_complex* y,
                                     rocblas_int incy)
{
    char transAC = rocblas2char_operation(transA);
    cgemv_(&transAC, &m, &n, &alpha, A, &lda, x, &incx, &beta, y, &incy);
}

template <>
void cpu_gemv<rocblas_double_complex>(rocblas_operation transA,
                                      rocblas_int m,
                                      rocblas_int n,
                                      rocblas_double_complex alpha,
                                      rocblas_double_complex* A,
                                      rocblas_int lda,
                                      rocblas_double_complex* x,
                                      rocblas_int incx,
                                      rocblas_double_complex beta,
                                      rocblas_double_complex* y,
                                      rocblas_int incy)
{
    char transAC = rocblas2char_operation(transA);
    zgemv_(&transAC, &m, &n, &alpha, A, &lda, x, &incx, &beta, y, &incy);
}

// gemm
template <>
void cpu_gemm<float>(rocblas_operation transA,
                     rocblas_operation transB,
                     rocblas_int m,
                     rocblas_int n,
                     rocblas_int k,
                     float alpha,
                     float* A,
                     rocblas_int lda,
                     float* B,
                     rocblas_int ldb,
                     float beta,
                     float* C,
                     rocblas_int ldc)
{
    char transAC = rocblas2char_operation(transA);
    char transBC = rocblas2char_operation(transB);
    sgemm_(&transAC, &transBC, &m, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
}

template <>
void cpu_gemm<double>(rocblas_operation transA,
                      rocblas_operation transB,
                      rocblas_int m,
                      rocblas_int n,
                      rocblas_int k,
                      double alpha,
                      double* A,
                      rocblas_int lda,
                      double* B,
                      rocblas_int ldb,
                      double beta,
                      double* C,
                      rocblas_int ldc)
{
    char transAC = rocblas2char_operation(transA);
    char transBC = rocblas2char_operation(transB);
    dgemm_(&transAC, &transBC, &m, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
}

template <>
void cpu_gemm<rocblas_float_complex>(rocblas_operation transA,
                                     rocblas_operation transB,
                                     rocblas_int m,
                                     rocblas_int n,
                                     rocblas_int k,
                                     rocblas_float_complex alpha,
                                     rocblas_float_complex* A,
                                     rocblas_int lda,
                                     rocblas_float_complex* B,
                                     rocblas_int ldb,
                                     rocblas_float_complex beta,
                                     rocblas_float_complex* C,
                                     rocblas_int ldc)
{
    char transAC = rocblas2char_operation(transA);
    char transBC = rocblas2char_operation(transB);
    cgemm_(&transAC, &transBC, &m, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
}

template <>
void cpu_gemm<rocblas_double_complex>(rocblas_operation transA,
                                      rocblas_operation transB,
                                      rocblas_int m,
                                      rocblas_int n,
                                      rocblas_int k,
                                      rocblas_double_complex alpha,
                                      rocblas_double_complex* A,
                                      rocblas_int lda,
                                      rocblas_double_complex* B,
                                      rocblas_int ldb,
                                      rocblas_double_complex beta,
                                      rocblas_double_complex* C,
                                      rocblas_int ldc)
{
    char transAC = rocblas2char_operation(transA);
    char transBC = rocblas2char_operation(transB);
    zgemm_(&transAC, &transBC, &m, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
}

// symv & hemv
template <>
void cpu_symv_hemv<float>(rocblas_fill uplo,
                          rocblas_int n,
                          float alpha,
                          float* A,
                          rocblas_int lda,
                          float* x,
                          rocblas_int incx,
                          float beta,
                          float* y,
                          rocblas_int incy)
{
    char uploC = rocblas2char_fill(uplo);
    ssymv_(&uploC, &n, &alpha, A, &lda, x, &incx, &beta, y, &incy);
}

template <>
void cpu_symv_hemv<double>(rocblas_fill uplo,
                           rocblas_int n,
                           double alpha,
                           double* A,
                           rocblas_int lda,
                           double* x,
                           rocblas_int incx,
                           double beta,
                           double* y,
                           rocblas_int incy)
{
    char uploC = rocblas2char_fill(uplo);
    dsymv_(&uploC, &n, &alpha, A, &lda, x, &incx, &beta, y, &incy);
}

template <>
void cpu_symv_hemv<rocblas_float_complex>(rocblas_fill uplo,
                                          rocblas_int n,
                                          rocblas_float_complex alpha,
                                          rocblas_float_complex* A,
                                          rocblas_int lda,
                                          rocblas_float_complex* x,
                                          rocblas_int incx,
                                          rocblas_float_complex beta,
                                          rocblas_float_complex* y,
                                          rocblas_int incy)
{
    char uploC = rocblas2char_fill(uplo);
    chemv_(&uploC, &n, &alpha, A, &lda, x, &incx, &beta, y, &incy);
}

template <>
void cpu_symv_hemv<rocblas_double_complex>(rocblas_fill uplo,
                                           rocblas_int n,
                                           rocblas_double_complex alpha,
                                           rocblas_double_complex* A,
                                           rocblas_int lda,
                                           rocblas_double_complex* x,
                                           rocblas_int incx,
                                           rocblas_double_complex beta,
                                           rocblas_double_complex* y,
                                           rocblas_int incy)
{
    char uploC = rocblas2char_fill(uplo);
    zhemv_(&uploC, &n, &alpha, A, &lda, x, &incx, &beta, y, &incy);
}

// symm & hemm
template <>
void cpu_symm_hemm<float>(rocblas_side side,
                          rocblas_fill uplo,
                          rocblas_int m,
                          rocblas_int n,
                          float alpha,
                          float* A,
                          rocblas_int lda,
                          float* B,
                          rocblas_int ldb,
                          float beta,
                          float* C,
                          rocblas_int ldc)
{
    char sideC = rocblas2char_side(side);
    char uploC = rocblas2char_fill(uplo);
    ssymm_(&sideC, &uploC, &m, &n, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
}

template <>
void cpu_symm_hemm<double>(rocblas_side side,
                           rocblas_fill uplo,
                           rocblas_int m,
                           rocblas_int n,
                           double alpha,
                           double* A,
                           rocblas_int lda,
                           double* B,
                           rocblas_int ldb,
                           double beta,
                           double* C,
                           rocblas_int ldc)
{
    char sideC = rocblas2char_side(side);
    char uploC = rocblas2char_fill(uplo);
    dsymm_(&sideC, &uploC, &m, &n, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
}

template <>
void cpu_symm_hemm<rocblas_float_complex>(rocblas_side side,
                                          rocblas_fill uplo,
                                          rocblas_int m,
                                          rocblas_int n,
                                          rocblas_float_complex alpha,
                                          rocblas_float_complex* A,
                                          rocblas_int lda,
                                          rocblas_float_complex* B,
                                          rocblas_int ldb,
                                          rocblas_float_complex beta,
                                          rocblas_float_complex* C,
                                          rocblas_int ldc)
{
    char sideC = rocblas2char_side(side);
    char uploC = rocblas2char_fill(uplo);
    chemm_(&sideC, &uploC, &m, &n, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
}

template <>
void cpu_symm_hemm<rocblas_double_complex>(rocblas_side side,
                                           rocblas_fill uplo,
                                           rocblas_int m,
                                           rocblas_int n,
                                           rocblas_double_complex alpha,
                                           rocblas_double_complex* A,
                                           rocblas_int lda,
                                           rocblas_double_complex* B,
                                           rocblas_int ldb,
                                           rocblas_double_complex beta,
                                           rocblas_double_complex* C,
                                           rocblas_int ldc)
{
    char sideC = rocblas2char_side(side);
    char uploC = rocblas2char_fill(uplo);
    zhemm_(&sideC, &uploC, &m, &n, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
}

// trsm
template <>
void cpu_trsm<float>(rocblas_side side,
                     rocblas_fill uplo,
                     rocblas_operation transA,
                     rocblas_diagonal diag,
                     rocblas_int m,
                     rocblas_int n,
                     float alpha,
                     float* A,
                     rocblas_int lda,
                     float* B,
                     rocblas_int ldb)
{
    char sideC = rocblas2char_side(side);
    char uploC = rocblas2char_fill(uplo);
    char transC = rocblas2char_operation(transA);
    char diagC = rocblas2char_diagonal(diag);
    strsm_(&sideC, &uploC, &transC, &diagC, &m, &n, &alpha, A, &lda, B, &ldb);
}

template <>
void cpu_trsm<double>(rocblas_side side,
                      rocblas_fill uplo,
                      rocblas_operation transA,
                      rocblas_diagonal diag,
                      rocblas_int m,
                      rocblas_int n,
                      double alpha,
                      double* A,
                      rocblas_int lda,
                      double* B,
                      rocblas_int ldb)
{
    char sideC = rocblas2char_side(side);
    char uploC = rocblas2char_fill(uplo);
    char transC = rocblas2char_operation(transA);
    char diagC = rocblas2char_diagonal(diag);
    dtrsm_(&sideC, &uploC, &transC, &diagC, &m, &n, &alpha, A, &lda, B, &ldb);
}

template <>
void cpu_trsm<rocblas_float_complex>(rocblas_side side,
                                     rocblas_fill uplo,
                                     rocblas_operation transA,
                                     rocblas_diagonal diag,
                                     rocblas_int m,
                                     rocblas_int n,
                                     rocblas_float_complex alpha,
                                     rocblas_float_complex* A,
                                     rocblas_int lda,
                                     rocblas_float_complex* B,
                                     rocblas_int ldb)
{
    char sideC = rocblas2char_side(side);
    char uploC = rocblas2char_fill(uplo);
    char transC = rocblas2char_operation(transA);
    char diagC = rocblas2char_diagonal(diag);
    ctrsm_(&sideC, &uploC, &transC, &diagC, &m, &n, &alpha, A, &lda, B, &ldb);
}

template <>
void cpu_trsm<rocblas_double_complex>(rocblas_side side,
                                      rocblas_fill uplo,
                                      rocblas_operation transA,
                                      rocblas_diagonal diag,
                                      rocblas_int m,
                                      rocblas_int n,
                                      rocblas_double_complex alpha,
                                      rocblas_double_complex* A,
                                      rocblas_int lda,
                                      rocblas_double_complex* B,
                                      rocblas_int ldb)
{
    char sideC = rocblas2char_side(side);
    char uploC = rocblas2char_fill(uplo);
    char transC = rocblas2char_operation(transA);
    char diagC = rocblas2char_diagonal(diag);
    ztrsm_(&sideC, &uploC, &transC, &diagC, &m, &n, &alpha, A, &lda, B, &ldb);
}
/*
// trsv
template <>
void cpu_trsv<float>(rocblas_fill uplo,
                     rocblas_operation transA,
                     rocblas_diagonal diag,
                     rocblas_int n,
                     float* A,
                     rocblas_int lda,
                     float* x,
                     rocblas_int incx)
{
    char uploC = rocblas2char_fill(uplo);
    char transC = rocblas2char_operation(transA);
    char diagC = rocblas2char_diagonal(diag);
    strsv_(&uploC, &transC, &diagC, &n, A, &lda, x, &incx);
}

template <>
void cpu_trsv<double>(rocblas_fill uplo,
                      rocblas_operation transA,
                      rocblas_diagonal diag,
                      rocblas_int n,
                      double* A,
                      rocblas_int lda,
                      double* x,
                      rocblas_int incx)
{
    char uploC = rocblas2char_fill(uplo);
    char transC = rocblas2char_operation(transA);
    char diagC = rocblas2char_diagonal(diag);
    dtrsv_(&uploC, &transC, &diagC, &n, A, &lda, x, &incx);
}

template <>
void cpu_trsv<rocblas_float_complex>(rocblas_fill uplo,
                                     rocblas_operation transA,
                                     rocblas_diagonal diag,
                                     rocblas_int n,
                                     rocblas_float_complex* A,
                                     rocblas_int lda,
                                     rocblas_float_complex* x,
                                     rocblas_int incx)
{
    char uploC = rocblas2char_fill(uplo);
    char transC = rocblas2char_operation(transA);
    char diagC = rocblas2char_diagonal(diag);
    ctrsv_(&uploC, &transC, &diagC, &n, A, &lda, x, &incx);
}

template <>
void cpu_trsv<rocblas_double_complex>(rocblas_fill uplo,
                                      rocblas_operation transA,
                                      rocblas_diagonal diag,
                                      rocblas_int n,
                                      rocblas_double_complex* A,
                                      rocblas_int lda,
                                      rocblas_double_complex* x,
                                      rocblas_int incx)
{
    char uploC = rocblas2char_fill(uplo);
    char transC = rocblas2char_operation(transA);
    char diagC = rocblas2char_diagonal(diag);
    ztrsv_(&uploC, &transC, &diagC, &n, A, &lda, x, &incx);
}
*/
// trmm
template <>
void cpu_trmm<float>(rocblas_side side,
                     rocblas_fill uplo,
                     rocblas_operation transA,
                     rocblas_diagonal diag,
                     rocblas_int m,
                     rocblas_int n,
                     float alpha,
                     float* A,
                     rocblas_int lda,
                     float* B,
                     rocblas_int ldb)
{
    char sideC = rocblas2char_side(side);
    char uploC = rocblas2char_fill(uplo);
    char transC = rocblas2char_operation(transA);
    char diagC = rocblas2char_diagonal(diag);
    strmm_(&sideC, &uploC, &transC, &diagC, &m, &n, &alpha, A, &lda, B, &ldb);
}

template <>
void cpu_trmm<double>(rocblas_side side,
                      rocblas_fill uplo,
                      rocblas_operation transA,
                      rocblas_diagonal diag,
                      rocblas_int m,
                      rocblas_int n,
                      double alpha,
                      double* A,
                      rocblas_int lda,
                      double* B,
                      rocblas_int ldb)
{
    char sideC = rocblas2char_side(side);
    char uploC = rocblas2char_fill(uplo);
    char transC = rocblas2char_operation(transA);
    char diagC = rocblas2char_diagonal(diag);
    dtrmm_(&sideC, &uploC, &transC, &diagC, &m, &n, &alpha, A, &lda, B, &ldb);
}

template <>
void cpu_trmm<rocblas_float_complex>(rocblas_side side,
                                     rocblas_fill uplo,
                                     rocblas_operation transA,
                                     rocblas_diagonal diag,
                                     rocblas_int m,
                                     rocblas_int n,
                                     rocblas_float_complex alpha,
                                     rocblas_float_complex* A,
                                     rocblas_int lda,
                                     rocblas_float_complex* B,
                                     rocblas_int ldb)
{
    char sideC = rocblas2char_side(side);
    char uploC = rocblas2char_fill(uplo);
    char transC = rocblas2char_operation(transA);
    char diagC = rocblas2char_diagonal(diag);
    ctrmm_(&sideC, &uploC, &transC, &diagC, &m, &n, &alpha, A, &lda, B, &ldb);
}

template <>
void cpu_trmm<rocblas_double_complex>(rocblas_side side,
                                      rocblas_fill uplo,
                                      rocblas_operation transA,
                                      rocblas_diagonal diag,
                                      rocblas_int m,
                                      rocblas_int n,
                                      rocblas_double_complex alpha,
                                      rocblas_double_complex* A,
                                      rocblas_int lda,
                                      rocblas_double_complex* B,
                                      rocblas_int ldb)
{
    char sideC = rocblas2char_side(side);
    char uploC = rocblas2char_fill(uplo);
    char transC = rocblas2char_operation(transA);
    char diagC = rocblas2char_diagonal(diag);
    ztrmm_(&sideC, &uploC, &transC, &diagC, &m, &n, &alpha, A, &lda, B, &ldb);
}

// potf2
template <>
void cpu_potf2(rocblas_fill uplo, rocblas_int n, float* A, rocblas_int lda, rocblas_int* info)
{
    char uploC = rocblas2char_fill(uplo);
    spotf2_(&uploC, &n, A, &lda, info);
}

template <>
void cpu_potf2(rocblas_fill uplo, rocblas_int n, double* A, rocblas_int lda, rocblas_int* info)
{
    char uploC = rocblas2char_fill(uplo);
    dpotf2_(&uploC, &n, A, &lda, info);
}

template <>
void cpu_potf2(rocblas_fill uplo,
               rocblas_int n,
               rocblas_float_complex* A,
               rocblas_int lda,
               rocblas_int* info)
{
    char uploC = rocblas2char_fill(uplo);
    cpotf2_(&uploC, &n, A, &lda, info);
}

template <>
void cpu_potf2(rocblas_fill uplo,
               rocblas_int n,
               rocblas_double_complex* A,
               rocblas_int lda,
               rocblas_int* info)
{
    char uploC = rocblas2char_fill(uplo);
    zpotf2_(&uploC, &n, A, &lda, info);
}

// potrf
template <>
void cpu_potrf(rocblas_fill uplo, rocblas_int n, float* A, rocblas_int lda, rocblas_int* info)
{
    char uploC = rocblas2char_fill(uplo);
    spotrf_(&uploC, &n, A, &lda, info);
}

template <>
void cpu_potrf(rocblas_fill uplo, rocblas_int n, double* A, rocblas_int lda, rocblas_int* info)
{
    char uploC = rocblas2char_fill(uplo);
    dpotrf_(&uploC, &n, A, &lda, info);
}

template <>
void cpu_potrf(rocblas_fill uplo,
               rocblas_int n,
               rocblas_float_complex* A,
               rocblas_int lda,
               rocblas_int* info)
{
    char uploC = rocblas2char_fill(uplo);
    cpotrf_(&uploC, &n, A, &lda, info);
}

template <>
void cpu_potrf(rocblas_fill uplo,
               rocblas_int n,
               rocblas_double_complex* A,
               rocblas_int lda,
               rocblas_int* info)
{
    char uploC = rocblas2char_fill(uplo);
    zpotrf_(&uploC, &n, A, &lda, info);
}

// potrs
template <>
void cpu_potrs(rocblas_fill uplo,
               rocblas_int n,
               rocblas_int nrhs,
               float* A,
               rocblas_int lda,
               float* B,
               rocblas_int ldb)
{
    int info;
    char uploC = rocblas2char_fill(uplo);
    spotrs_(&uploC, &n, &nrhs, A, &lda, B, &ldb, &info);
}

template <>
void cpu_potrs(rocblas_fill uplo,
               rocblas_int n,
               rocblas_int nrhs,
               double* A,
               rocblas_int lda,
               double* B,
               rocblas_int ldb)
{
    int info;
    char uploC = rocblas2char_fill(uplo);
    dpotrs_(&uploC, &n, &nrhs, A, &lda, B, &ldb, &info);
}

template <>
void cpu_potrs(rocblas_fill uplo,
               rocblas_int n,
               rocblas_int nrhs,
               rocblas_float_complex* A,
               rocblas_int lda,
               rocblas_float_complex* B,
               rocblas_int ldb)
{
    int info;
    char uploC = rocblas2char_fill(uplo);
    cpotrs_(&uploC, &n, &nrhs, A, &lda, B, &ldb, &info);
}

template <>
void cpu_potrs(rocblas_fill uplo,
               rocblas_int n,
               rocblas_int nrhs,
               rocblas_double_complex* A,
               rocblas_int lda,
               rocblas_double_complex* B,
               rocblas_int ldb)
{
    int info;
    char uploC = rocblas2char_fill(uplo);
    zpotrs_(&uploC, &n, &nrhs, A, &lda, B, &ldb, &info);
}

// posv
template <>
void cpu_posv(rocblas_fill uplo,
              rocblas_int n,
              rocblas_int nrhs,
              float* A,
              rocblas_int lda,
              float* B,
              rocblas_int ldb,
              rocblas_int* info)
{
    char uploC = rocblas2char_fill(uplo);
    sposv_(&uploC, &n, &nrhs, A, &lda, B, &ldb, info);
}

template <>
void cpu_posv(rocblas_fill uplo,
              rocblas_int n,
              rocblas_int nrhs,
              double* A,
              rocblas_int lda,
              double* B,
              rocblas_int ldb,
              rocblas_int* info)
{
    char uploC = rocblas2char_fill(uplo);
    dposv_(&uploC, &n, &nrhs, A, &lda, B, &ldb, info);
}

template <>
void cpu_posv(rocblas_fill uplo,
              rocblas_int n,
              rocblas_int nrhs,
              rocblas_float_complex* A,
              rocblas_int lda,
              rocblas_float_complex* B,
              rocblas_int ldb,
              rocblas_int* info)
{
    char uploC = rocblas2char_fill(uplo);
    cposv_(&uploC, &n, &nrhs, A, &lda, B, &ldb, info);
}

template <>
void cpu_posv(rocblas_fill uplo,
              rocblas_int n,
              rocblas_int nrhs,
              rocblas_double_complex* A,
              rocblas_int lda,
              rocblas_double_complex* B,
              rocblas_int ldb,
              rocblas_int* info)
{
    char uploC = rocblas2char_fill(uplo);
    zposv_(&uploC, &n, &nrhs, A, &lda, B, &ldb, info);
}

// potri
template <>
void cpu_potri(rocblas_fill uplo, rocblas_int n, float* A, rocblas_int lda, rocblas_int* info)
{
    char uploC = rocblas2char_fill(uplo);
    spotri_(&uploC, &n, A, &lda, info);
}

template <>
void cpu_potri(rocblas_fill uplo, rocblas_int n, double* A, rocblas_int lda, rocblas_int* info)
{
    char uploC = rocblas2char_fill(uplo);
    dpotri_(&uploC, &n, A, &lda, info);
}

template <>
void cpu_potri(rocblas_fill uplo,
               rocblas_int n,
               rocblas_float_complex* A,
               rocblas_int lda,
               rocblas_int* info)
{
    char uploC = rocblas2char_fill(uplo);
    cpotri_(&uploC, &n, A, &lda, info);
}

template <>
void cpu_potri(rocblas_fill uplo,
               rocblas_int n,
               rocblas_double_complex* A,
               rocblas_int lda,
               rocblas_int* info)
{
    char uploC = rocblas2char_fill(uplo);
    zpotri_(&uploC, &n, A, &lda, info);
}

// getf2
template <>
void cpu_getf2(rocblas_int m, rocblas_int n, float* A, rocblas_int lda, rocblas_int* ipiv, rocblas_int* info)
{
    sgetf2_(&m, &n, A, &lda, ipiv, info);
}

template <>
void cpu_getf2(rocblas_int m,
               rocblas_int n,
               double* A,
               rocblas_int lda,
               rocblas_int* ipiv,
               rocblas_int* info)
{
    dgetf2_(&m, &n, A, &lda, ipiv, info);
}

template <>
void cpu_getf2(rocblas_int m,
               rocblas_int n,
               rocblas_float_complex* A,
               rocblas_int lda,
               rocblas_int* ipiv,
               rocblas_int* info)
{
    cgetf2_(&m, &n, A, &lda, ipiv, info);
}

template <>
void cpu_getf2(rocblas_int m,
               rocblas_int n,
               rocblas_double_complex* A,
               rocblas_int lda,
               rocblas_int* ipiv,
               rocblas_int* info)
{
    zgetf2_(&m, &n, A, &lda, ipiv, info);
}

// getrf
template <>
void cpu_getrf<float>(rocblas_int m,
                      rocblas_int n,
                      float* A,
                      rocblas_int lda,
                      rocblas_int* ipiv,
                      rocblas_int* info)
{
    sgetrf_(&m, &n, A, &lda, ipiv, info);
}

template <>
void cpu_getrf<double>(rocblas_int m,
                       rocblas_int n,
                       double* A,
                       rocblas_int lda,
                       rocblas_int* ipiv,
                       rocblas_int* info)
{
    dgetrf_(&m, &n, A, &lda, ipiv, info);
}

template <>
void cpu_getrf<rocblas_float_complex>(rocblas_int m,
                                      rocblas_int n,
                                      rocblas_float_complex* A,
                                      rocblas_int lda,
                                      rocblas_int* ipiv,
                                      rocblas_int* info)
{
    cgetrf_(&m, &n, A, &lda, ipiv, info);
}

template <>
void cpu_getrf<rocblas_double_complex>(rocblas_int m,
                                       rocblas_int n,
                                       rocblas_double_complex* A,
                                       rocblas_int lda,
                                       rocblas_int* ipiv,
                                       rocblas_int* info)
{
    zgetrf_(&m, &n, A, &lda, ipiv, info);
}

// getrs
template <>
void cpu_getrs<float>(rocblas_operation trans,
                      rocblas_int n,
                      rocblas_int nrhs,
                      float* A,
                      rocblas_int lda,
                      rocblas_int* ipiv,
                      float* B,
                      rocblas_int ldb)
{
    rocblas_int info;
    char transC = rocblas2char_operation(trans);
    sgetrs_(&transC, &n, &nrhs, A, &lda, ipiv, B, &ldb, &info);
}

template <>
void cpu_getrs<double>(rocblas_operation trans,
                       rocblas_int n,
                       rocblas_int nrhs,
                       double* A,
                       rocblas_int lda,
                       rocblas_int* ipiv,
                       double* B,
                       rocblas_int ldb)
{
    rocblas_int info;
    char transC = rocblas2char_operation(trans);
    dgetrs_(&transC, &n, &nrhs, A, &lda, ipiv, B, &ldb, &info);
}

template <>
void cpu_getrs<rocblas_float_complex>(rocblas_operation trans,
                                      rocblas_int n,
                                      rocblas_int nrhs,
                                      rocblas_float_complex* A,
                                      rocblas_int lda,
                                      rocblas_int* ipiv,
                                      rocblas_float_complex* B,
                                      rocblas_int ldb)
{
    rocblas_int info;
    char transC = rocblas2char_operation(trans);
    cgetrs_(&transC, &n, &nrhs, A, &lda, ipiv, B, &ldb, &info);
}

template <>
void cpu_getrs<rocblas_double_complex>(rocblas_operation trans,
                                       rocblas_int n,
                                       rocblas_int nrhs,
                                       rocblas_double_complex* A,
                                       rocblas_int lda,
                                       rocblas_int* ipiv,
                                       rocblas_double_complex* B,
                                       rocblas_int ldb)
{
    rocblas_int info;
    char transC = rocblas2char_operation(trans);
    zgetrs_(&transC, &n, &nrhs, A, &lda, ipiv, B, &ldb, &info);
}

// gesv
template <>
void cpu_gesv<float>(rocblas_int n,
                     rocblas_int nrhs,
                     float* A,
                     rocblas_int lda,
                     rocblas_int* ipiv,
                     float* B,
                     rocblas_int ldb,
                     rocblas_int* info)
{
    sgesv_(&n, &nrhs, A, &lda, ipiv, B, &ldb, info);
}

template <>
void cpu_gesv<double>(rocblas_int n,
                      rocblas_int nrhs,
                      double* A,
                      rocblas_int lda,
                      rocblas_int* ipiv,
                      double* B,
                      rocblas_int ldb,
                      rocblas_int* info)
{
    dgesv_(&n, &nrhs, A, &lda, ipiv, B, &ldb, info);
}

template <>
void cpu_gesv<rocblas_float_complex>(rocblas_int n,
                                     rocblas_int nrhs,
                                     rocblas_float_complex* A,
                                     rocblas_int lda,
                                     rocblas_int* ipiv,
                                     rocblas_float_complex* B,
                                     rocblas_int ldb,
                                     rocblas_int* info)
{
    cgesv_(&n, &nrhs, A, &lda, ipiv, B, &ldb, info);
}

template <>
void cpu_gesv<rocblas_double_complex>(rocblas_int n,
                                      rocblas_int nrhs,
                                      rocblas_double_complex* A,
                                      rocblas_int lda,
                                      rocblas_int* ipiv,
                                      rocblas_double_complex* B,
                                      rocblas_int ldb,
                                      rocblas_int* info)
{
    zgesv_(&n, &nrhs, A, &lda, ipiv, B, &ldb, info);
}

// gels
template <>
void cpu_gels<float>(rocblas_operation transR,
                     rocblas_int m,
                     rocblas_int n,
                     rocblas_int nrhs,
                     float* A,
                     rocblas_int lda,
                     float* B,
                     rocblas_int ldb,
                     float* work,
                     rocblas_int lwork,
                     rocblas_int* info)
{
    char trans = rocblas2char_operation(transR);
    sgels_(&trans, &m, &n, &nrhs, A, &lda, B, &ldb, work, &lwork, info);
}

template <>
void cpu_gels<double>(rocblas_operation transR,
                      rocblas_int m,
                      rocblas_int n,
                      rocblas_int nrhs,
                      double* A,
                      rocblas_int lda,
                      double* B,
                      rocblas_int ldb,
                      double* work,
                      rocblas_int lwork,
                      rocblas_int* info)
{
    char trans = rocblas2char_operation(transR);
    dgels_(&trans, &m, &n, &nrhs, A, &lda, B, &ldb, work, &lwork, info);
}

template <>
void cpu_gels<rocblas_float_complex>(rocblas_operation transR,
                                     rocblas_int m,
                                     rocblas_int n,
                                     rocblas_int nrhs,
                                     rocblas_float_complex* A,
                                     rocblas_int lda,
                                     rocblas_float_complex* B,
                                     rocblas_int ldb,
                                     rocblas_float_complex* work,
                                     rocblas_int lwork,
                                     rocblas_int* info)
{
    char trans = rocblas2char_operation(transR);
    cgels_(&trans, &m, &n, &nrhs, A, &lda, B, &ldb, work, &lwork, info);
}

template <>
void cpu_gels<rocblas_double_complex>(rocblas_operation transR,
                                      rocblas_int m,
                                      rocblas_int n,
                                      rocblas_int nrhs,
                                      rocblas_double_complex* A,
                                      rocblas_int lda,
                                      rocblas_double_complex* B,
                                      rocblas_int ldb,
                                      rocblas_double_complex* work,
                                      rocblas_int lwork,
                                      rocblas_int* info)
{
    char trans = rocblas2char_operation(transR);
    zgels_(&trans, &m, &n, &nrhs, A, &lda, B, &ldb, work, &lwork, info);
}

// trtri
template <>
void cpu_trtri<float>(rocblas_fill uplo,
                      rocblas_diagonal diag,
                      rocblas_int n,
                      float* A,
                      rocblas_int lda,
                      rocblas_int* info)
{
    char uploC = rocblas2char_fill(uplo);
    char diagC = rocblas2char_diagonal(diag);
    strtri_(&uploC, &diagC, &n, A, &lda, info);
}

template <>
void cpu_trtri<double>(rocblas_fill uplo,
                       rocblas_diagonal diag,
                       rocblas_int n,
                       double* A,
                       rocblas_int lda,
                       rocblas_int* info)
{
    char uploC = rocblas2char_fill(uplo);
    char diagC = rocblas2char_diagonal(diag);
    dtrtri_(&uploC, &diagC, &n, A, &lda, info);
}

template <>
void cpu_trtri<rocblas_float_complex>(rocblas_fill uplo,
                                      rocblas_diagonal diag,
                                      rocblas_int n,
                                      rocblas_float_complex* A,
                                      rocblas_int lda,
                                      rocblas_int* info)
{
    char uploC = rocblas2char_fill(uplo);
    char diagC = rocblas2char_diagonal(diag);
    ctrtri_(&uploC, &diagC, &n, A, &lda, info);
}

template <>
void cpu_trtri<rocblas_double_complex>(rocblas_fill uplo,
                                       rocblas_diagonal diag,
                                       rocblas_int n,
                                       rocblas_double_complex* A,
                                       rocblas_int lda,
                                       rocblas_int* info)
{
    char uploC = rocblas2char_fill(uplo);
    char diagC = rocblas2char_diagonal(diag);
    ztrtri_(&uploC, &diagC, &n, A, &lda, info);
}

// getri
template <>
void cpu_getri<float>(rocblas_int n,
                      float* A,
                      rocblas_int lda,
                      rocblas_int* ipiv,
                      float* work,
                      rocblas_int lwork,
                      rocblas_int* info)
{
    sgetri_(&n, A, &lda, ipiv, work, &lwork, info);
}

template <>
void cpu_getri<double>(rocblas_int n,
                       double* A,
                       rocblas_int lda,
                       rocblas_int* ipiv,
                       double* work,
                       rocblas_int lwork,
                       rocblas_int* info)
{
    dgetri_(&n, A, &lda, ipiv, work, &lwork, info);
}

template <>
void cpu_getri<rocblas_float_complex>(rocblas_int n,
                                      rocblas_float_complex* A,
                                      rocblas_int lda,
                                      rocblas_int* ipiv,
                                      rocblas_float_complex* work,
                                      rocblas_int lwork,
                                      rocblas_int* info)
{
    cgetri_(&n, A, &lda, ipiv, work, &lwork, info);
}

template <>
void cpu_getri<rocblas_double_complex>(rocblas_int n,
                                       rocblas_double_complex* A,
                                       rocblas_int lda,
                                       rocblas_int* ipiv,
                                       rocblas_double_complex* work,
                                       rocblas_int lwork,
                                       rocblas_int* info)
{
    zgetri_(&n, A, &lda, ipiv, work, &lwork, info);
}

// geqrf
template <>
void cpu_geqrf<float>(rocblas_int m,
                      rocblas_int n,
                      float* A,
                      rocblas_int lda,
                      float* ipiv,
                      float* work,
                      rocblas_int lwork)
{
    int info;
    sgeqrf_(&m, &n, A, &lda, ipiv, work, &lwork, &info);
}

template <>
void cpu_geqrf<double>(rocblas_int m,
                       rocblas_int n,
                       double* A,
                       rocblas_int lda,
                       double* ipiv,
                       double* work,
                       rocblas_int lwork)
{
    int info;
    dgeqrf_(&m, &n, A, &lda, ipiv, work, &lwork, &info);
}

template <>
void cpu_geqrf<rocblas_float_complex>(rocblas_int m,
                                      rocblas_int n,
                                      rocblas_float_complex* A,
                                      rocblas_int lda,
                                      rocblas_float_complex* ipiv,
                                      rocblas_float_complex* work,
                                      rocblas_int lwork)
{
    int info;
    cgeqrf_(&m, &n, A, &lda, ipiv, work, &lwork, &info);
}

template <>
void cpu_geqrf<rocblas_double_complex>(rocblas_int m,
                                       rocblas_int n,
                                       rocblas_double_complex* A,
                                       rocblas_int lda,
                                       rocblas_double_complex* ipiv,
                                       rocblas_double_complex* work,
                                       rocblas_int lwork)
{
    int info;
    zgeqrf_(&m, &n, A, &lda, ipiv, work, &lwork, &info);
}

// geqr2
template <>
void cpu_geqr2<float>(rocblas_int m, rocblas_int n, float* A, rocblas_int lda, float* ipiv, float* work)
{
    int info;
    sgeqr2_(&m, &n, A, &lda, ipiv, work, &info);
}

template <>
void cpu_geqr2<double>(rocblas_int m, rocblas_int n, double* A, rocblas_int lda, double* ipiv, double* work)
{
    int info;
    dgeqr2_(&m, &n, A, &lda, ipiv, work, &info);
}

template <>
void cpu_geqr2<rocblas_float_complex>(rocblas_int m,
                                      rocblas_int n,
                                      rocblas_float_complex* A,
                                      rocblas_int lda,
                                      rocblas_float_complex* ipiv,
                                      rocblas_float_complex* work)
{
    int info;
    cgeqr2_(&m, &n, A, &lda, ipiv, work, &info);
}

template <>
void cpu_geqr2<rocblas_double_complex>(rocblas_int m,
                                       rocblas_int n,
                                       rocblas_double_complex* A,
                                       rocblas_int lda,
                                       rocblas_double_complex* ipiv,
                                       rocblas_double_complex* work)
{
    int info;
    zgeqr2_(&m, &n, A, &lda, ipiv, work, &info);
}

// gerqf
template <>
void cpu_gerqf<float>(rocblas_int m,
                      rocblas_int n,
                      float* A,
                      rocblas_int lda,
                      float* ipiv,
                      float* work,
                      rocblas_int lwork)
{
    int info;
    sgerqf_(&m, &n, A, &lda, ipiv, work, &lwork, &info);
}

template <>
void cpu_gerqf<double>(rocblas_int m,
                       rocblas_int n,
                       double* A,
                       rocblas_int lda,
                       double* ipiv,
                       double* work,
                       rocblas_int lwork)
{
    int info;
    dgerqf_(&m, &n, A, &lda, ipiv, work, &lwork, &info);
}

template <>
void cpu_gerqf<rocblas_float_complex>(rocblas_int m,
                                      rocblas_int n,
                                      rocblas_float_complex* A,
                                      rocblas_int lda,
                                      rocblas_float_complex* ipiv,
                                      rocblas_float_complex* work,
                                      rocblas_int lwork)
{
    int info;
    cgerqf_(&m, &n, A, &lda, ipiv, work, &lwork, &info);
}

template <>
void cpu_gerqf<rocblas_double_complex>(rocblas_int m,
                                       rocblas_int n,
                                       rocblas_double_complex* A,
                                       rocblas_int lda,
                                       rocblas_double_complex* ipiv,
                                       rocblas_double_complex* work,
                                       rocblas_int lwork)
{
    int info;
    zgerqf_(&m, &n, A, &lda, ipiv, work, &lwork, &info);
}

// gerq2
template <>
void cpu_gerq2<float>(rocblas_int m, rocblas_int n, float* A, rocblas_int lda, float* ipiv, float* work)
{
    int info;
    sgerq2_(&m, &n, A, &lda, ipiv, work, &info);
}

template <>
void cpu_gerq2<double>(rocblas_int m, rocblas_int n, double* A, rocblas_int lda, double* ipiv, double* work)
{
    int info;
    dgerq2_(&m, &n, A, &lda, ipiv, work, &info);
}

template <>
void cpu_gerq2<rocblas_float_complex>(rocblas_int m,
                                      rocblas_int n,
                                      rocblas_float_complex* A,
                                      rocblas_int lda,
                                      rocblas_float_complex* ipiv,
                                      rocblas_float_complex* work)
{
    int info;
    cgerq2_(&m, &n, A, &lda, ipiv, work, &info);
}

template <>
void cpu_gerq2<rocblas_double_complex>(rocblas_int m,
                                       rocblas_int n,
                                       rocblas_double_complex* A,
                                       rocblas_int lda,
                                       rocblas_double_complex* ipiv,
                                       rocblas_double_complex* work)
{
    int info;
    zgerq2_(&m, &n, A, &lda, ipiv, work, &info);
}

// geqlf
template <>
void cpu_geqlf<float>(rocblas_int m,
                      rocblas_int n,
                      float* A,
                      rocblas_int lda,
                      float* ipiv,
                      float* work,
                      rocblas_int lwork)
{
    int info;
    sgeqlf_(&m, &n, A, &lda, ipiv, work, &lwork, &info);
}

template <>
void cpu_geqlf<double>(rocblas_int m,
                       rocblas_int n,
                       double* A,
                       rocblas_int lda,
                       double* ipiv,
                       double* work,
                       rocblas_int lwork)
{
    int info;
    dgeqlf_(&m, &n, A, &lda, ipiv, work, &lwork, &info);
}

template <>
void cpu_geqlf<rocblas_float_complex>(rocblas_int m,
                                      rocblas_int n,
                                      rocblas_float_complex* A,
                                      rocblas_int lda,
                                      rocblas_float_complex* ipiv,
                                      rocblas_float_complex* work,
                                      rocblas_int lwork)
{
    int info;
    cgeqlf_(&m, &n, A, &lda, ipiv, work, &lwork, &info);
}

template <>
void cpu_geqlf<rocblas_double_complex>(rocblas_int m,
                                       rocblas_int n,
                                       rocblas_double_complex* A,
                                       rocblas_int lda,
                                       rocblas_double_complex* ipiv,
                                       rocblas_double_complex* work,
                                       rocblas_int lwork)
{
    int info;
    zgeqlf_(&m, &n, A, &lda, ipiv, work, &lwork, &info);
}

// geql2
template <>
void cpu_geql2<float>(rocblas_int m, rocblas_int n, float* A, rocblas_int lda, float* ipiv, float* work)
{
    int info;
    sgeql2_(&m, &n, A, &lda, ipiv, work, &info);
}

template <>
void cpu_geql2<double>(rocblas_int m, rocblas_int n, double* A, rocblas_int lda, double* ipiv, double* work)
{
    int info;
    dgeql2_(&m, &n, A, &lda, ipiv, work, &info);
}

template <>
void cpu_geql2<rocblas_float_complex>(rocblas_int m,
                                      rocblas_int n,
                                      rocblas_float_complex* A,
                                      rocblas_int lda,
                                      rocblas_float_complex* ipiv,
                                      rocblas_float_complex* work)
{
    int info;
    cgeql2_(&m, &n, A, &lda, ipiv, work, &info);
}

template <>
void cpu_geql2<rocblas_double_complex>(rocblas_int m,
                                       rocblas_int n,
                                       rocblas_double_complex* A,
                                       rocblas_int lda,
                                       rocblas_double_complex* ipiv,
                                       rocblas_double_complex* work)
{
    int info;
    zgeql2_(&m, &n, A, &lda, ipiv, work, &info);
}

// gelqf
template <>
void cpu_gelqf<float>(rocblas_int m,
                      rocblas_int n,
                      float* A,
                      rocblas_int lda,
                      float* ipiv,
                      float* work,
                      rocblas_int lwork)
{
    int info;
    sgelqf_(&m, &n, A, &lda, ipiv, work, &lwork, &info);
}

template <>
void cpu_gelqf<double>(rocblas_int m,
                       rocblas_int n,
                       double* A,
                       rocblas_int lda,
                       double* ipiv,
                       double* work,
                       rocblas_int lwork)
{
    int info;
    dgelqf_(&m, &n, A, &lda, ipiv, work, &lwork, &info);
}

template <>
void cpu_gelqf<rocblas_float_complex>(rocblas_int m,
                                      rocblas_int n,
                                      rocblas_float_complex* A,
                                      rocblas_int lda,
                                      rocblas_float_complex* ipiv,
                                      rocblas_float_complex* work,
                                      rocblas_int lwork)
{
    int info;
    cgelqf_(&m, &n, A, &lda, ipiv, work, &lwork, &info);
}

template <>
void cpu_gelqf<rocblas_double_complex>(rocblas_int m,
                                       rocblas_int n,
                                       rocblas_double_complex* A,
                                       rocblas_int lda,
                                       rocblas_double_complex* ipiv,
                                       rocblas_double_complex* work,
                                       rocblas_int lwork)
{
    int info;
    zgelqf_(&m, &n, A, &lda, ipiv, work, &lwork, &info);
}

// gelq2
template <>
void cpu_gelq2<float>(rocblas_int m, rocblas_int n, float* A, rocblas_int lda, float* ipiv, float* work)
{
    int info;
    sgelq2_(&m, &n, A, &lda, ipiv, work, &info);
}

template <>
void cpu_gelq2<double>(rocblas_int m, rocblas_int n, double* A, rocblas_int lda, double* ipiv, double* work)
{
    int info;
    dgelq2_(&m, &n, A, &lda, ipiv, work, &info);
}

template <>
void cpu_gelq2<rocblas_float_complex>(rocblas_int m,
                                      rocblas_int n,
                                      rocblas_float_complex* A,
                                      rocblas_int lda,
                                      rocblas_float_complex* ipiv,
                                      rocblas_float_complex* work)
{
    int info;
    cgelq2_(&m, &n, A, &lda, ipiv, work, &info);
}

template <>
void cpu_gelq2<rocblas_double_complex>(rocblas_int m,
                                       rocblas_int n,
                                       rocblas_double_complex* A,
                                       rocblas_int lda,
                                       rocblas_double_complex* ipiv,
                                       rocblas_double_complex* work)
{
    int info;
    zgelq2_(&m, &n, A, &lda, ipiv, work, &info);
}

// gebd2
template <>
void cpu_gebd2<float, float>(rocblas_int m,
                             rocblas_int n,
                             float* A,
                             rocblas_int lda,
                             float* D,
                             float* E,
                             float* tauq,
                             float* taup,
                             float* work)
{
    int info;
    sgebd2_(&m, &n, A, &lda, D, E, tauq, taup, work, &info);
}

template <>
void cpu_gebd2<double, double>(rocblas_int m,
                               rocblas_int n,
                               double* A,
                               rocblas_int lda,
                               double* D,
                               double* E,
                               double* tauq,
                               double* taup,
                               double* work)
{
    int info;
    dgebd2_(&m, &n, A, &lda, D, E, tauq, taup, work, &info);
}

template <>
void cpu_gebd2<rocblas_float_complex, float>(rocblas_int m,
                                             rocblas_int n,
                                             rocblas_float_complex* A,
                                             rocblas_int lda,
                                             float* D,
                                             float* E,
                                             rocblas_float_complex* tauq,
                                             rocblas_float_complex* taup,
                                             rocblas_float_complex* work)
{
    int info;
    cgebd2_(&m, &n, A, &lda, D, E, tauq, taup, work, &info);
}

template <>
void cpu_gebd2<rocblas_double_complex, double>(rocblas_int m,
                                               rocblas_int n,
                                               rocblas_double_complex* A,
                                               rocblas_int lda,
                                               double* D,
                                               double* E,
                                               rocblas_double_complex* tauq,
                                               rocblas_double_complex* taup,
                                               rocblas_double_complex* work)
{
    int info;
    zgebd2_(&m, &n, A, &lda, D, E, tauq, taup, work, &info);
}

// gebrd
template <>
void cpu_gebrd<float, float>(rocblas_int m,
                             rocblas_int n,
                             float* A,
                             rocblas_int lda,
                             float* D,
                             float* E,
                             float* tauq,
                             float* taup,
                             float* work,
                             rocblas_int size_w)
{
    int info;
    sgebrd_(&m, &n, A, &lda, D, E, tauq, taup, work, &size_w, &info);
}

template <>
void cpu_gebrd<double, double>(rocblas_int m,
                               rocblas_int n,
                               double* A,
                               rocblas_int lda,
                               double* D,
                               double* E,
                               double* tauq,
                               double* taup,
                               double* work,
                               rocblas_int size_w)
{
    int info;
    dgebrd_(&m, &n, A, &lda, D, E, tauq, taup, work, &size_w, &info);
}

template <>
void cpu_gebrd<rocblas_float_complex, float>(rocblas_int m,
                                             rocblas_int n,
                                             rocblas_float_complex* A,
                                             rocblas_int lda,
                                             float* D,
                                             float* E,
                                             rocblas_float_complex* tauq,
                                             rocblas_float_complex* taup,
                                             rocblas_float_complex* work,
                                             rocblas_int size_w)
{
    int info;
    cgebrd_(&m, &n, A, &lda, D, E, tauq, taup, work, &size_w, &info);
}

template <>
void cpu_gebrd<rocblas_double_complex, double>(rocblas_int m,
                                               rocblas_int n,
                                               rocblas_double_complex* A,
                                               rocblas_int lda,
                                               double* D,
                                               double* E,
                                               rocblas_double_complex* tauq,
                                               rocblas_double_complex* taup,
                                               rocblas_double_complex* work,
                                               rocblas_int size_w)
{
    int info;
    zgebrd_(&m, &n, A, &lda, D, E, tauq, taup, work, &size_w, &info);
}

// sytrd & hetrd
template <>
void cpu_sytrd_hetrd<float, float>(rocblas_fill uplo,
                                   rocblas_int n,
                                   float* A,
                                   rocblas_int lda,
                                   float* D,
                                   float* E,
                                   float* tau,
                                   float* work,
                                   rocblas_int size_w)
{
    int info;
    char uploC = rocblas2char_fill(uplo);
    ssytrd_(&uploC, &n, A, &lda, D, E, tau, work, &size_w, &info);
}

template <>
void cpu_sytrd_hetrd<double, double>(rocblas_fill uplo,
                                     rocblas_int n,
                                     double* A,
                                     rocblas_int lda,
                                     double* D,
                                     double* E,
                                     double* tau,
                                     double* work,
                                     rocblas_int size_w)
{
    int info;
    char uploC = rocblas2char_fill(uplo);
    dsytrd_(&uploC, &n, A, &lda, D, E, tau, work, &size_w, &info);
}

template <>
void cpu_sytrd_hetrd<rocblas_float_complex, float>(rocblas_fill uplo,
                                                   rocblas_int n,
                                                   rocblas_float_complex* A,
                                                   rocblas_int lda,
                                                   float* D,
                                                   float* E,
                                                   rocblas_float_complex* tau,
                                                   rocblas_float_complex* work,
                                                   rocblas_int size_w)
{
    int info;
    char uploC = rocblas2char_fill(uplo);
    chetrd_(&uploC, &n, A, &lda, D, E, tau, work, &size_w, &info);
}

template <>
void cpu_sytrd_hetrd<rocblas_double_complex, double>(rocblas_fill uplo,
                                                     rocblas_int n,
                                                     rocblas_double_complex* A,
                                                     rocblas_int lda,
                                                     double* D,
                                                     double* E,
                                                     rocblas_double_complex* tau,
                                                     rocblas_double_complex* work,
                                                     rocblas_int size_w)
{
    int info;
    char uploC = rocblas2char_fill(uplo);
    zhetrd_(&uploC, &n, A, &lda, D, E, tau, work, &size_w, &info);
}

// sytd2 & hetd2
template <>
void cpu_sytd2_hetd2<float, float>(rocblas_fill uplo,
                                   rocblas_int n,
                                   float* A,
                                   rocblas_int lda,
                                   float* D,
                                   float* E,
                                   float* tau)
{
    int info;
    char uploC = rocblas2char_fill(uplo);
    ssytd2_(&uploC, &n, A, &lda, D, E, tau, &info);
}

template <>
void cpu_sytd2_hetd2<double, double>(rocblas_fill uplo,
                                     rocblas_int n,
                                     double* A,
                                     rocblas_int lda,
                                     double* D,
                                     double* E,
                                     double* tau)
{
    int info;
    char uploC = rocblas2char_fill(uplo);
    dsytd2_(&uploC, &n, A, &lda, D, E, tau, &info);
}

template <>
void cpu_sytd2_hetd2<rocblas_float_complex, float>(rocblas_fill uplo,
                                                   rocblas_int n,
                                                   rocblas_float_complex* A,
                                                   rocblas_int lda,
                                                   float* D,
                                                   float* E,
                                                   rocblas_float_complex* tau)
{
    int info;
    char uploC = rocblas2char_fill(uplo);
    chetd2_(&uploC, &n, A, &lda, D, E, tau, &info);
}

template <>
void cpu_sytd2_hetd2<rocblas_double_complex, double>(rocblas_fill uplo,
                                                     rocblas_int n,
                                                     rocblas_double_complex* A,
                                                     rocblas_int lda,
                                                     double* D,
                                                     double* E,
                                                     rocblas_double_complex* tau)
{
    int info;
    char uploC = rocblas2char_fill(uplo);
    zhetd2_(&uploC, &n, A, &lda, D, E, tau, &info);
}

// sterf
template <>
void cpu_sterf<float>(rocblas_int n, float* D, float* E)
{
    int info;
    ssterf_(&n, D, E, &info);
}

template <>
void cpu_sterf<double>(rocblas_int n, double* D, double* E)
{
    int info;
    dsterf_(&n, D, E, &info);
}

// steqr
template <>
void cpu_steqr<float, float>(rocblas_evect evect,
                             rocblas_int n,
                             float* D,
                             float* E,
                             float* C,
                             rocblas_int ldc,
                             float* work,
                             rocblas_int* info)
{
    char evectC = rocblas2char_evect(evect);
    ssteqr_(&evectC, &n, D, E, C, &ldc, work, info);
}

template <>
void cpu_steqr<double, double>(rocblas_evect evect,
                               rocblas_int n,
                               double* D,
                               double* E,
                               double* C,
                               rocblas_int ldc,
                               double* work,
                               rocblas_int* info)
{
    char evectC = rocblas2char_evect(evect);
    dsteqr_(&evectC, &n, D, E, C, &ldc, work, info);
}
template <>
void cpu_steqr<rocblas_float_complex, float>(rocblas_evect evect,
                                             rocblas_int n,
                                             float* D,
                                             float* E,
                                             rocblas_float_complex* C,
                                             rocblas_int ldc,
                                             float* work,
                                             rocblas_int* info)
{
    char evectC = rocblas2char_evect(evect);
    csteqr_(&evectC, &n, D, E, C, &ldc, work, info);
}

template <>
void cpu_steqr<rocblas_double_complex, double>(rocblas_evect evect,
                                               rocblas_int n,
                                               double* D,
                                               double* E,
                                               rocblas_double_complex* C,
                                               rocblas_int ldc,
                                               double* work,
                                               rocblas_int* info)
{
    char evectC = rocblas2char_evect(evect);
    zsteqr_(&evectC, &n, D, E, C, &ldc, work, info);
}

// stedc
template <>
void cpu_stedc<float, float>(rocblas_evect evect,
                             rocblas_int n,
                             float* D,
                             float* E,
                             float* C,
                             rocblas_int ldc,
                             float* work,
                             rocblas_int lwork,
                             float* rwork,
                             rocblas_int lrwork,
                             rocblas_int* iwork,
                             rocblas_int liwork,
                             rocblas_int* info)
{
    char evectC = rocblas2char_evect(evect);
    sstedc_(&evectC, &n, D, E, C, &ldc, rwork, &lrwork, iwork, &liwork, info);
}

template <>
void cpu_stedc<double, double>(rocblas_evect evect,
                               rocblas_int n,
                               double* D,
                               double* E,
                               double* C,
                               rocblas_int ldc,
                               double* work,
                               rocblas_int lwork,
                               double* rwork,
                               rocblas_int lrwork,
                               rocblas_int* iwork,
                               rocblas_int liwork,
                               rocblas_int* info)
{
    char evectC = rocblas2char_evect(evect);
    dstedc_(&evectC, &n, D, E, C, &ldc, rwork, &lrwork, iwork, &liwork, info);
}
template <>
void cpu_stedc<rocblas_float_complex, float>(rocblas_evect evect,
                                             rocblas_int n,
                                             float* D,
                                             float* E,
                                             rocblas_float_complex* C,
                                             rocblas_int ldc,
                                             rocblas_float_complex* work,
                                             rocblas_int lwork,
                                             float* rwork,
                                             rocblas_int lrwork,
                                             rocblas_int* iwork,
                                             rocblas_int liwork,
                                             rocblas_int* info)
{
    char evectC = rocblas2char_evect(evect);
    cstedc_(&evectC, &n, D, E, C, &ldc, work, &lwork, rwork, &lrwork, iwork, &liwork, info);
}

template <>
void cpu_stedc<rocblas_double_complex, double>(rocblas_evect evect,
                                               rocblas_int n,
                                               double* D,
                                               double* E,
                                               rocblas_double_complex* C,
                                               rocblas_int ldc,
                                               rocblas_double_complex* work,
                                               rocblas_int lwork,
                                               double* rwork,
                                               rocblas_int lrwork,
                                               rocblas_int* iwork,
                                               rocblas_int liwork,
                                               rocblas_int* info)
{
    char evectC = rocblas2char_evect(evect);
    zstedc_(&evectC, &n, D, E, C, &ldc, work, &lwork, rwork, &lrwork, iwork, &liwork, info);
}

// stebz
template <>
void cpu_stebz<float>(rocblas_erange erange,
                      rocblas_eorder eorder,
                      rocblas_int n,
                      float vl,
                      float vu,
                      rocblas_int il,
                      rocblas_int iu,
                      float abstol,
                      float* D,
                      float* E,
                      rocblas_int* m,
                      rocblas_int* nsplit,
                      float* W,
                      rocblas_int* iblock,
                      rocblas_int* isplit,
                      float* work,
                      rocblas_int* iwork,
                      rocblas_int* info)
{
    char erangeC = rocblas2char_erange(erange);
    char eorderC = rocblas2char_eorder(eorder);
    sstebz_(&erangeC, &eorderC, &n, &vl, &vu, &il, &iu, &abstol, D, E, m, nsplit, W, iblock, isplit,
            work, iwork, info);
}

template <>
void cpu_stebz<double>(rocblas_erange erange,
                       rocblas_eorder eorder,
                       rocblas_int n,
                       double vl,
                       double vu,
                       rocblas_int il,
                       rocblas_int iu,
                       double abstol,
                       double* D,
                       double* E,
                       rocblas_int* m,
                       rocblas_int* nsplit,
                       double* W,
                       rocblas_int* iblock,
                       rocblas_int* isplit,
                       double* work,
                       rocblas_int* iwork,
                       rocblas_int* info)
{
    char erangeC = rocblas2char_erange(erange);
    char eorderC = rocblas2char_eorder(eorder);
    dstebz_(&erangeC, &eorderC, &n, &vl, &vu, &il, &iu, &abstol, D, E, m, nsplit, W, iblock, isplit,
            work, iwork, info);
}

// stein
template <>
void cpu_stein<float, float>(rocblas_int n,
                             float* D,
                             float* E,
                             rocblas_int* nev,
                             float* W,
                             rocblas_int* iblock,
                             rocblas_int* isplit,
                             float* Z,
                             rocblas_int ldz,
                             float* work,
                             rocblas_int* iwork,
                             rocblas_int* ifail,
                             rocblas_int* info)
{
    sstein_(&n, D, E, nev, W, iblock, isplit, Z, &ldz, work, iwork, ifail, info);
}

template <>
void cpu_stein<double, double>(rocblas_int n,
                               double* D,
                               double* E,
                               rocblas_int* nev,
                               double* W,
                               rocblas_int* iblock,
                               rocblas_int* isplit,
                               double* Z,
                               rocblas_int ldz,
                               double* work,
                               rocblas_int* iwork,
                               rocblas_int* ifail,
                               rocblas_int* info)
{
    dstein_(&n, D, E, nev, W, iblock, isplit, Z, &ldz, work, iwork, ifail, info);
}

template <>
void cpu_stein<rocblas_float_complex, float>(rocblas_int n,
                                             float* D,
                                             float* E,
                                             rocblas_int* nev,
                                             float* W,
                                             rocblas_int* iblock,
                                             rocblas_int* isplit,
                                             rocblas_float_complex* Z,
                                             rocblas_int ldz,
                                             float* work,
                                             rocblas_int* iwork,
                                             rocblas_int* ifail,
                                             rocblas_int* info)
{
    cstein_(&n, D, E, nev, W, iblock, isplit, Z, &ldz, work, iwork, ifail, info);
}

template <>
void cpu_stein<rocblas_double_complex, double>(rocblas_int n,
                                               double* D,
                                               double* E,
                                               rocblas_int* nev,
                                               double* W,
                                               rocblas_int* iblock,
                                               rocblas_int* isplit,
                                               rocblas_double_complex* Z,
                                               rocblas_int ldz,
                                               double* work,
                                               rocblas_int* iwork,
                                               rocblas_int* ifail,
                                               rocblas_int* info)
{
    zstein_(&n, D, E, nev, W, iblock, isplit, Z, &ldz, work, iwork, ifail, info);
}

// sygs2 & hegs2
template <>
void cpu_sygs2_hegs2<float>(rocblas_eform itype,
                            rocblas_fill uplo,
                            rocblas_int n,
                            float* A,
                            rocblas_int lda,
                            float* B,
                            rocblas_int ldb)
{
    rocblas_int info;
    int itypeI = rocblas2char_eform(itype) - '0';
    char uploC = rocblas2char_fill(uplo);
    ssygs2_(&itypeI, &uploC, &n, A, &lda, B, &ldb, &info);
}

template <>
void cpu_sygs2_hegs2<double>(rocblas_eform itype,
                             rocblas_fill uplo,
                             rocblas_int n,
                             double* A,
                             rocblas_int lda,
                             double* B,
                             rocblas_int ldb)
{
    rocblas_int info;
    int itypeI = rocblas2char_eform(itype) - '0';
    char uploC = rocblas2char_fill(uplo);
    dsygs2_(&itypeI, &uploC, &n, A, &lda, B, &ldb, &info);
}

template <>
void cpu_sygs2_hegs2<rocblas_float_complex>(rocblas_eform itype,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            rocblas_float_complex* A,
                                            rocblas_int lda,
                                            rocblas_float_complex* B,
                                            rocblas_int ldb)
{
    rocblas_int info;
    int itypeI = rocblas2char_eform(itype) - '0';
    char uploC = rocblas2char_fill(uplo);
    chegs2_(&itypeI, &uploC, &n, A, &lda, B, &ldb, &info);
}

template <>
void cpu_sygs2_hegs2<rocblas_double_complex>(rocblas_eform itype,
                                             rocblas_fill uplo,
                                             rocblas_int n,
                                             rocblas_double_complex* A,
                                             rocblas_int lda,
                                             rocblas_double_complex* B,
                                             rocblas_int ldb)
{
    rocblas_int info;
    int itypeI = rocblas2char_eform(itype) - '0';
    char uploC = rocblas2char_fill(uplo);
    zhegs2_(&itypeI, &uploC, &n, A, &lda, B, &ldb, &info);
}

// sygst & hegst
template <>
void cpu_sygst_hegst<float>(rocblas_eform itype,
                            rocblas_fill uplo,
                            rocblas_int n,
                            float* A,
                            rocblas_int lda,
                            float* B,
                            rocblas_int ldb)
{
    rocblas_int info;
    int itypeI = rocblas2char_eform(itype) - '0';
    char uploC = rocblas2char_fill(uplo);
    ssygst_(&itypeI, &uploC, &n, A, &lda, B, &ldb, &info);
}

template <>
void cpu_sygst_hegst<double>(rocblas_eform itype,
                             rocblas_fill uplo,
                             rocblas_int n,
                             double* A,
                             rocblas_int lda,
                             double* B,
                             rocblas_int ldb)
{
    rocblas_int info;
    int itypeI = rocblas2char_eform(itype) - '0';
    char uploC = rocblas2char_fill(uplo);
    dsygst_(&itypeI, &uploC, &n, A, &lda, B, &ldb, &info);
}

template <>
void cpu_sygst_hegst<rocblas_float_complex>(rocblas_eform itype,
                                            rocblas_fill uplo,
                                            rocblas_int n,
                                            rocblas_float_complex* A,
                                            rocblas_int lda,
                                            rocblas_float_complex* B,
                                            rocblas_int ldb)
{
    rocblas_int info;
    int itypeI = rocblas2char_eform(itype) - '0';
    char uploC = rocblas2char_fill(uplo);
    chegst_(&itypeI, &uploC, &n, A, &lda, B, &ldb, &info);
}

template <>
void cpu_sygst_hegst<rocblas_double_complex>(rocblas_eform itype,
                                             rocblas_fill uplo,
                                             rocblas_int n,
                                             rocblas_double_complex* A,
                                             rocblas_int lda,
                                             rocblas_double_complex* B,
                                             rocblas_int ldb)
{
    rocblas_int info;
    int itypeI = rocblas2char_eform(itype) - '0';
    char uploC = rocblas2char_fill(uplo);
    zhegst_(&itypeI, &uploC, &n, A, &lda, B, &ldb, &info);
}

// syev & heev
template <>
void cpu_syev_heev<float, float>(rocblas_evect evect,
                                 rocblas_fill uplo,
                                 rocblas_int n,
                                 float* A,
                                 rocblas_int lda,
                                 float* W,
                                 float* work,
                                 rocblas_int lwork,
                                 float* rwork,
                                 rocblas_int lrwork,
                                 rocblas_int* info)
{
    char evectC = rocblas2char_evect(evect);
    char uploC = rocblas2char_fill(uplo);
    ssyev_(&evectC, &uploC, &n, A, &lda, W, rwork, &lrwork, info);
}

template <>
void cpu_syev_heev<double, double>(rocblas_evect evect,
                                   rocblas_fill uplo,
                                   rocblas_int n,
                                   double* A,
                                   rocblas_int lda,
                                   double* W,
                                   double* work,
                                   rocblas_int lwork,
                                   double* rwork,
                                   rocblas_int lrwork,
                                   rocblas_int* info)
{
    char evectC = rocblas2char_evect(evect);
    char uploC = rocblas2char_fill(uplo);
    dsyev_(&evectC, &uploC, &n, A, &lda, W, rwork, &lrwork, info);
}

template <>
void cpu_syev_heev<rocblas_float_complex, float>(rocblas_evect evect,
                                                 rocblas_fill uplo,
                                                 rocblas_int n,
                                                 rocblas_float_complex* A,
                                                 rocblas_int lda,
                                                 float* W,
                                                 rocblas_float_complex* work,
                                                 rocblas_int lwork,
                                                 float* rwork,
                                                 rocblas_int lrwork,
                                                 rocblas_int* info)
{
    char evectC = rocblas2char_evect(evect);
    char uploC = rocblas2char_fill(uplo);
    cheev_(&evectC, &uploC, &n, A, &lda, W, work, &lwork, rwork, info);
}

template <>
void cpu_syev_heev<rocblas_double_complex, double>(rocblas_evect evect,
                                                   rocblas_fill uplo,
                                                   rocblas_int n,
                                                   rocblas_double_complex* A,
                                                   rocblas_int lda,
                                                   double* W,
                                                   rocblas_double_complex* work,
                                                   rocblas_int lwork,
                                                   double* rwork,
                                                   rocblas_int lrwork,
                                                   rocblas_int* info)
{
    char evectC = rocblas2char_evect(evect);
    char uploC = rocblas2char_fill(uplo);
    zheev_(&evectC, &uploC, &n, A, &lda, W, work, &lwork, rwork, info);
}

// syevd & heevd
template <>
void cpu_syevd_heevd<float, float>(rocblas_evect evect,
                                   rocblas_fill uplo,
                                   rocblas_int n,
                                   float* A,
                                   rocblas_int lda,
                                   float* W,
                                   float* work,
                                   rocblas_int lwork,
                                   float* rwork,
                                   rocblas_int lrwork,
                                   rocblas_int* iwork,
                                   rocblas_int liwork,
                                   rocblas_int* info)
{
    char evectC = rocblas2char_evect(evect);
    char uploC = rocblas2char_fill(uplo);
    ssyevd_(&evectC, &uploC, &n, A, &lda, W, rwork, &lrwork, iwork, &liwork, info);
}

template <>
void cpu_syevd_heevd<double, double>(rocblas_evect evect,
                                     rocblas_fill uplo,
                                     rocblas_int n,
                                     double* A,
                                     rocblas_int lda,
                                     double* W,
                                     double* work,
                                     rocblas_int lwork,
                                     double* rwork,
                                     rocblas_int lrwork,
                                     rocblas_int* iwork,
                                     rocblas_int liwork,
                                     rocblas_int* info)
{
    char evectC = rocblas2char_evect(evect);
    char uploC = rocblas2char_fill(uplo);
    dsyevd_(&evectC, &uploC, &n, A, &lda, W, rwork, &lrwork, iwork, &liwork, info);
}

template <>
void cpu_syevd_heevd<rocblas_float_complex, float>(rocblas_evect evect,
                                                   rocblas_fill uplo,
                                                   rocblas_int n,
                                                   rocblas_float_complex* A,
                                                   rocblas_int lda,
                                                   float* W,
                                                   rocblas_float_complex* work,
                                                   rocblas_int lwork,
                                                   float* rwork,
                                                   rocblas_int lrwork,
                                                   rocblas_int* iwork,
                                                   rocblas_int liwork,
                                                   rocblas_int* info)
{
    char evectC = rocblas2char_evect(evect);
    char uploC = rocblas2char_fill(uplo);
    cheevd_(&evectC, &uploC, &n, A, &lda, W, work, &lwork, rwork, &lrwork, iwork, &liwork, info);
}

template <>
void cpu_syevd_heevd<rocblas_double_complex, double>(rocblas_evect evect,
                                                     rocblas_fill uplo,
                                                     rocblas_int n,
                                                     rocblas_double_complex* A,
                                                     rocblas_int lda,
                                                     double* W,
                                                     rocblas_double_complex* work,
                                                     rocblas_int lwork,
                                                     double* rwork,
                                                     rocblas_int lrwork,
                                                     rocblas_int* iwork,
                                                     rocblas_int liwork,
                                                     rocblas_int* info)
{
    char evectC = rocblas2char_evect(evect);
    char uploC = rocblas2char_fill(uplo);
    zheevd_(&evectC, &uploC, &n, A, &lda, W, work, &lwork, rwork, &lrwork, iwork, &liwork, info);
}

// syevx & heevx
template <>
void cpu_syevx_heevx<float, float>(rocblas_evect evect,
                                   rocblas_erange erange,
                                   rocblas_fill uplo,
                                   rocblas_int n,
                                   float* A,
                                   rocblas_int lda,
                                   float vl,
                                   float vu,
                                   rocblas_int il,
                                   rocblas_int iu,
                                   float abstol,
                                   rocblas_int* nev,
                                   float* W,
                                   float* Z,
                                   rocblas_int ldz,
                                   float* work,
                                   rocblas_int lwork,
                                   float* rwork,
                                   rocblas_int* iwork,
                                   rocblas_int* ifail,
                                   rocblas_int* info)
{
    char evectC = rocblas2char_evect(evect);
    char erangeC = rocblas2char_erange(erange);
    char uploC = rocblas2char_fill(uplo);
    ssyevx_(&evectC, &erangeC, &uploC, &n, A, &lda, &vl, &vu, &il, &iu, &abstol, nev, W, Z, &ldz,
            work, &lwork, iwork, ifail, info);
}

template <>
void cpu_syevx_heevx<double, double>(rocblas_evect evect,
                                     rocblas_erange erange,
                                     rocblas_fill uplo,
                                     rocblas_int n,
                                     double* A,
                                     rocblas_int lda,
                                     double vl,
                                     double vu,
                                     rocblas_int il,
                                     rocblas_int iu,
                                     double abstol,
                                     rocblas_int* nev,
                                     double* W,
                                     double* Z,
                                     rocblas_int ldz,
                                     double* work,
                                     rocblas_int lwork,
                                     double* rwork,
                                     rocblas_int* iwork,
                                     rocblas_int* ifail,
                                     rocblas_int* info)
{
    char evectC = rocblas2char_evect(evect);
    char erangeC = rocblas2char_erange(erange);
    char uploC = rocblas2char_fill(uplo);
    dsyevx_(&evectC, &erangeC, &uploC, &n, A, &lda, &vl, &vu, &il, &iu, &abstol, nev, W, Z, &ldz,
            work, &lwork, iwork, ifail, info);
}

template <>
void cpu_syevx_heevx<rocblas_float_complex, float>(rocblas_evect evect,
                                                   rocblas_erange erange,
                                                   rocblas_fill uplo,
                                                   rocblas_int n,
                                                   rocblas_float_complex* A,
                                                   rocblas_int lda,
                                                   float vl,
                                                   float vu,
                                                   rocblas_int il,
                                                   rocblas_int iu,
                                                   float abstol,
                                                   rocblas_int* nev,
                                                   float* W,
                                                   rocblas_float_complex* Z,
                                                   rocblas_int ldz,
                                                   rocblas_float_complex* work,
                                                   rocblas_int lwork,
                                                   float* rwork,
                                                   rocblas_int* iwork,
                                                   rocblas_int* ifail,
                                                   rocblas_int* info)
{
    char evectC = rocblas2char_evect(evect);
    char erangeC = rocblas2char_erange(erange);
    char uploC = rocblas2char_fill(uplo);
    cheevx_(&evectC, &erangeC, &uploC, &n, A, &lda, &vl, &vu, &il, &iu, &abstol, nev, W, Z, &ldz,
            work, &lwork, rwork, iwork, ifail, info);
}

template <>
void cpu_syevx_heevx<rocblas_double_complex, double>(rocblas_evect evect,
                                                     rocblas_erange erange,
                                                     rocblas_fill uplo,
                                                     rocblas_int n,
                                                     rocblas_double_complex* A,
                                                     rocblas_int lda,
                                                     double vl,
                                                     double vu,
                                                     rocblas_int il,
                                                     rocblas_int iu,
                                                     double abstol,
                                                     rocblas_int* nev,
                                                     double* W,
                                                     rocblas_double_complex* Z,
                                                     rocblas_int ldz,
                                                     rocblas_double_complex* work,
                                                     rocblas_int lwork,
                                                     double* rwork,
                                                     rocblas_int* iwork,
                                                     rocblas_int* ifail,
                                                     rocblas_int* info)
{
    char evectC = rocblas2char_evect(evect);
    char erangeC = rocblas2char_erange(erange);
    char uploC = rocblas2char_fill(uplo);
    zheevx_(&evectC, &erangeC, &uploC, &n, A, &lda, &vl, &vu, &il, &iu, &abstol, nev, W, Z, &ldz,
            work, &lwork, rwork, iwork, ifail, info);
}

// sygv & hegv
template <>
void cpu_sygv_hegv<float, float>(rocblas_eform itype,
                                 rocblas_evect evect,
                                 rocblas_fill uplo,
                                 rocblas_int n,
                                 float* A,
                                 rocblas_int lda,
                                 float* B,
                                 rocblas_int ldb,
                                 float* W,
                                 float* work,
                                 rocblas_int lwork,
                                 float* rwork,
                                 rocblas_int* info)
{
    int itypeI = rocblas2char_eform(itype) - '0';
    char evectC = rocblas2char_evect(evect);
    char uploC = rocblas2char_fill(uplo);
    ssygv_(&itypeI, &evectC, &uploC, &n, A, &lda, B, &ldb, W, work, &lwork, info);
}

template <>
void cpu_sygv_hegv<double, double>(rocblas_eform itype,
                                   rocblas_evect evect,
                                   rocblas_fill uplo,
                                   rocblas_int n,
                                   double* A,
                                   rocblas_int lda,
                                   double* B,
                                   rocblas_int ldb,
                                   double* W,
                                   double* work,
                                   rocblas_int lwork,
                                   double* rwork,
                                   rocblas_int* info)
{
    int itypeI = rocblas2char_eform(itype) - '0';
    char evectC = rocblas2char_evect(evect);
    char uploC = rocblas2char_fill(uplo);
    dsygv_(&itypeI, &evectC, &uploC, &n, A, &lda, B, &ldb, W, work, &lwork, info);
}

template <>
void cpu_sygv_hegv<rocblas_float_complex, float>(rocblas_eform itype,
                                                 rocblas_evect evect,
                                                 rocblas_fill uplo,
                                                 rocblas_int n,
                                                 rocblas_float_complex* A,
                                                 rocblas_int lda,
                                                 rocblas_float_complex* B,
                                                 rocblas_int ldb,
                                                 float* W,
                                                 rocblas_float_complex* work,
                                                 rocblas_int lwork,
                                                 float* rwork,
                                                 rocblas_int* info)
{
    int itypeI = rocblas2char_eform(itype) - '0';
    char evectC = rocblas2char_evect(evect);
    char uploC = rocblas2char_fill(uplo);
    chegv_(&itypeI, &evectC, &uploC, &n, A, &lda, B, &ldb, W, work, &lwork, rwork, info);
}

template <>
void cpu_sygv_hegv<rocblas_double_complex, double>(rocblas_eform itype,
                                                   rocblas_evect evect,
                                                   rocblas_fill uplo,
                                                   rocblas_int n,
                                                   rocblas_double_complex* A,
                                                   rocblas_int lda,
                                                   rocblas_double_complex* B,
                                                   rocblas_int ldb,
                                                   double* W,
                                                   rocblas_double_complex* work,
                                                   rocblas_int lwork,
                                                   double* rwork,
                                                   rocblas_int* info)
{
    int itypeI = rocblas2char_eform(itype) - '0';
    char evectC = rocblas2char_evect(evect);
    char uploC = rocblas2char_fill(uplo);
    zhegv_(&itypeI, &evectC, &uploC, &n, A, &lda, B, &ldb, W, work, &lwork, rwork, info);
}

// sygvd & hegvd
template <>
void cpu_sygvd_hegvd<float, float>(rocblas_eform itype,
                                   rocblas_evect evect,
                                   rocblas_fill uplo,
                                   rocblas_int n,
                                   float* A,
                                   rocblas_int lda,
                                   float* B,
                                   rocblas_int ldb,
                                   float* W,
                                   float* work,
                                   rocblas_int lwork,
                                   float* rwork,
                                   rocblas_int lrwork,
                                   rocblas_int* iwork,
                                   rocblas_int liwork,
                                   rocblas_int* info)
{
    int itypeI = rocblas2char_eform(itype) - '0';
    char evectC = rocblas2char_evect(evect);
    char uploC = rocblas2char_fill(uplo);
    ssygvd_(&itypeI, &evectC, &uploC, &n, A, &lda, B, &ldb, W, rwork, &lrwork, iwork, &liwork, info);
}

template <>
void cpu_sygvd_hegvd<double, double>(rocblas_eform itype,
                                     rocblas_evect evect,
                                     rocblas_fill uplo,
                                     rocblas_int n,
                                     double* A,
                                     rocblas_int lda,
                                     double* B,
                                     rocblas_int ldb,
                                     double* W,
                                     double* work,
                                     rocblas_int lwork,
                                     double* rwork,
                                     rocblas_int lrwork,
                                     rocblas_int* iwork,
                                     rocblas_int liwork,
                                     rocblas_int* info)
{
    int itypeI = rocblas2char_eform(itype) - '0';
    char evectC = rocblas2char_evect(evect);
    char uploC = rocblas2char_fill(uplo);
    dsygvd_(&itypeI, &evectC, &uploC, &n, A, &lda, B, &ldb, W, rwork, &lrwork, iwork, &liwork, info);
}

template <>
void cpu_sygvd_hegvd<rocblas_float_complex, float>(rocblas_eform itype,
                                                   rocblas_evect evect,
                                                   rocblas_fill uplo,
                                                   rocblas_int n,
                                                   rocblas_float_complex* A,
                                                   rocblas_int lda,
                                                   rocblas_float_complex* B,
                                                   rocblas_int ldb,
                                                   float* W,
                                                   rocblas_float_complex* work,
                                                   rocblas_int lwork,
                                                   float* rwork,
                                                   rocblas_int lrwork,
                                                   rocblas_int* iwork,
                                                   rocblas_int liwork,
                                                   rocblas_int* info)
{
    int itypeI = rocblas2char_eform(itype) - '0';
    char evectC = rocblas2char_evect(evect);
    char uploC = rocblas2char_fill(uplo);
    chegvd_(&itypeI, &evectC, &uploC, &n, A, &lda, B, &ldb, W, work, &lwork, rwork, &lrwork, iwork,
            &liwork, info);
}

template <>
void cpu_sygvd_hegvd<rocblas_double_complex, double>(rocblas_eform itype,
                                                     rocblas_evect evect,
                                                     rocblas_fill uplo,
                                                     rocblas_int n,
                                                     rocblas_double_complex* A,
                                                     rocblas_int lda,
                                                     rocblas_double_complex* B,
                                                     rocblas_int ldb,
                                                     double* W,
                                                     rocblas_double_complex* work,
                                                     rocblas_int lwork,
                                                     double* rwork,
                                                     rocblas_int lrwork,
                                                     rocblas_int* iwork,
                                                     rocblas_int liwork,
                                                     rocblas_int* info)
{
    int itypeI = rocblas2char_eform(itype) - '0';
    char evectC = rocblas2char_evect(evect);
    char uploC = rocblas2char_fill(uplo);
    zhegvd_(&itypeI, &evectC, &uploC, &n, A, &lda, B, &ldb, W, work, &lwork, rwork, &lrwork, iwork,
            &liwork, info);
}

// sygvx & hegvx
template <>
void cpu_sygvx_hegvx<float, float>(rocblas_eform itype,
                                   rocblas_evect evect,
                                   rocblas_erange erange,
                                   rocblas_fill uplo,
                                   rocblas_int n,
                                   float* A,
                                   rocblas_int lda,
                                   float* B,
                                   rocblas_int ldb,
                                   float vl,
                                   float vu,
                                   rocblas_int il,
                                   rocblas_int iu,
                                   float abstol,
                                   rocblas_int* m,
                                   float* W,
                                   float* Z,
                                   rocblas_int ldz,
                                   float* work,
                                   rocblas_int lwork,
                                   float* rwork,
                                   rocblas_int* iwork,
                                   rocblas_int* ifail,
                                   rocblas_int* info)
{
    int itypeI = rocblas2char_eform(itype) - '0';
    char evectC = rocblas2char_evect(evect);
    char erangeC = rocblas2char_erange(erange);
    char uploC = rocblas2char_fill(uplo);
    ssygvx_(&itypeI, &evectC, &erangeC, &uploC, &n, A, &lda, B, &ldb, &vl, &vu, &il, &iu, &abstol,
            m, W, Z, &ldz, work, &lwork, iwork, ifail, info);
}

template <>
void cpu_sygvx_hegvx<double, double>(rocblas_eform itype,
                                     rocblas_evect evect,
                                     rocblas_erange erange,
                                     rocblas_fill uplo,
                                     rocblas_int n,
                                     double* A,
                                     rocblas_int lda,
                                     double* B,
                                     rocblas_int ldb,
                                     double vl,
                                     double vu,
                                     rocblas_int il,
                                     rocblas_int iu,
                                     double abstol,
                                     rocblas_int* m,
                                     double* W,
                                     double* Z,
                                     rocblas_int ldz,
                                     double* work,
                                     rocblas_int lwork,
                                     double* rwork,
                                     rocblas_int* iwork,
                                     rocblas_int* ifail,
                                     rocblas_int* info)
{
    int itypeI = rocblas2char_eform(itype) - '0';
    char evectC = rocblas2char_evect(evect);
    char erangeC = rocblas2char_erange(erange);
    char uploC = rocblas2char_fill(uplo);
    dsygvx_(&itypeI, &evectC, &erangeC, &uploC, &n, A, &lda, B, &ldb, &vl, &vu, &il, &iu, &abstol,
            m, W, Z, &ldz, work, &lwork, iwork, ifail, info);
}

template <>
void cpu_sygvx_hegvx<rocblas_float_complex, float>(rocblas_eform itype,
                                                   rocblas_evect evect,
                                                   rocblas_erange erange,
                                                   rocblas_fill uplo,
                                                   rocblas_int n,
                                                   rocblas_float_complex* A,
                                                   rocblas_int lda,
                                                   rocblas_float_complex* B,
                                                   rocblas_int ldb,
                                                   float vl,
                                                   float vu,
                                                   rocblas_int il,
                                                   rocblas_int iu,
                                                   float abstol,
                                                   rocblas_int* m,
                                                   float* W,
                                                   rocblas_float_complex* Z,
                                                   rocblas_int ldz,
                                                   rocblas_float_complex* work,
                                                   rocblas_int lwork,
                                                   float* rwork,
                                                   rocblas_int* iwork,
                                                   rocblas_int* ifail,
                                                   rocblas_int* info)
{
    int itypeI = rocblas2char_eform(itype) - '0';
    char evectC = rocblas2char_evect(evect);
    char erangeC = rocblas2char_erange(erange);
    char uploC = rocblas2char_fill(uplo);
    chegvx_(&itypeI, &evectC, &erangeC, &uploC, &n, A, &lda, B, &ldb, &vl, &vu, &il, &iu, &abstol,
            m, W, Z, &ldz, work, &lwork, rwork, iwork, ifail, info);
}

template <>
void cpu_sygvx_hegvx<rocblas_double_complex, double>(rocblas_eform itype,
                                                     rocblas_evect evect,
                                                     rocblas_erange erange,
                                                     rocblas_fill uplo,
                                                     rocblas_int n,
                                                     rocblas_double_complex* A,
                                                     rocblas_int lda,
                                                     rocblas_double_complex* B,
                                                     rocblas_int ldb,
                                                     double vl,
                                                     double vu,
                                                     rocblas_int il,
                                                     rocblas_int iu,
                                                     double abstol,
                                                     rocblas_int* m,
                                                     double* W,
                                                     rocblas_double_complex* Z,
                                                     rocblas_int ldz,
                                                     rocblas_double_complex* work,
                                                     rocblas_int lwork,
                                                     double* rwork,
                                                     rocblas_int* iwork,
                                                     rocblas_int* ifail,
                                                     rocblas_int* info)
{
    int itypeI = rocblas2char_eform(itype) - '0';
    char evectC = rocblas2char_evect(evect);
    char erangeC = rocblas2char_erange(erange);
    char uploC = rocblas2char_fill(uplo);
    zhegvx_(&itypeI, &evectC, &erangeC, &uploC, &n, A, &lda, B, &ldb, &vl, &vu, &il, &iu, &abstol,
            m, W, Z, &ldz, work, &lwork, rwork, iwork, ifail, info);
}

// lasyf
template <>
void cpu_lasyf<float>(rocblas_fill uplo,
                      rocblas_int n,
                      rocblas_int nb,
                      rocblas_int* kb,
                      float* A,
                      rocblas_int lda,
                      rocblas_int* ipiv,
                      float* W,
                      rocblas_int ldw,
                      rocblas_int* info)
{
    char uploC = rocblas2char_fill(uplo);
    slasyf_(&uploC, &n, &nb, kb, A, &lda, ipiv, W, &ldw, info);
}

template <>
void cpu_lasyf<double>(rocblas_fill uplo,
                       rocblas_int n,
                       rocblas_int nb,
                       rocblas_int* kb,
                       double* A,
                       rocblas_int lda,
                       rocblas_int* ipiv,
                       double* W,
                       rocblas_int ldw,
                       rocblas_int* info)
{
    char uploC = rocblas2char_fill(uplo);
    dlasyf_(&uploC, &n, &nb, kb, A, &lda, ipiv, W, &ldw, info);
}

template <>
void cpu_lasyf<rocblas_float_complex>(rocblas_fill uplo,
                                      rocblas_int n,
                                      rocblas_int nb,
                                      rocblas_int* kb,
                                      rocblas_float_complex* A,
                                      rocblas_int lda,
                                      rocblas_int* ipiv,
                                      rocblas_float_complex* W,
                                      rocblas_int ldw,
                                      rocblas_int* info)
{
    char uploC = rocblas2char_fill(uplo);
    clasyf_(&uploC, &n, &nb, kb, A, &lda, ipiv, W, &ldw, info);
}

template <>
void cpu_lasyf<rocblas_double_complex>(rocblas_fill uplo,
                                       rocblas_int n,
                                       rocblas_int nb,
                                       rocblas_int* kb,
                                       rocblas_double_complex* A,
                                       rocblas_int lda,
                                       rocblas_int* ipiv,
                                       rocblas_double_complex* W,
                                       rocblas_int ldw,
                                       rocblas_int* info)
{
    char uploC = rocblas2char_fill(uplo);
    zlasyf_(&uploC, &n, &nb, kb, A, &lda, ipiv, W, &ldw, info);
}

// sytf2
template <>
void cpu_sytf2<float>(rocblas_fill uplo,
                      rocblas_int n,
                      float* A,
                      rocblas_int lda,
                      rocblas_int* ipiv,
                      rocblas_int* info)
{
    char uploC = rocblas2char_fill(uplo);
    ssytf2_(&uploC, &n, A, &lda, ipiv, info);
}

template <>
void cpu_sytf2<double>(rocblas_fill uplo,
                       rocblas_int n,
                       double* A,
                       rocblas_int lda,
                       rocblas_int* ipiv,
                       rocblas_int* info)
{
    char uploC = rocblas2char_fill(uplo);
    dsytf2_(&uploC, &n, A, &lda, ipiv, info);
}

template <>
void cpu_sytf2<rocblas_float_complex>(rocblas_fill uplo,
                                      rocblas_int n,
                                      rocblas_float_complex* A,
                                      rocblas_int lda,
                                      rocblas_int* ipiv,
                                      rocblas_int* info)
{
    char uploC = rocblas2char_fill(uplo);
    csytf2_(&uploC, &n, A, &lda, ipiv, info);
}

template <>
void cpu_sytf2<rocblas_double_complex>(rocblas_fill uplo,
                                       rocblas_int n,
                                       rocblas_double_complex* A,
                                       rocblas_int lda,
                                       rocblas_int* ipiv,
                                       rocblas_int* info)
{
    char uploC = rocblas2char_fill(uplo);
    zsytf2_(&uploC, &n, A, &lda, ipiv, info);
}

// sytrf
template <>
void cpu_sytrf<float>(rocblas_fill uplo,
                      rocblas_int n,
                      float* A,
                      rocblas_int lda,
                      rocblas_int* ipiv,
                      float* work,
                      rocblas_int lwork,
                      rocblas_int* info)
{
    char uploC = rocblas2char_fill(uplo);
    ssytrf_(&uploC, &n, A, &lda, ipiv, work, &lwork, info);
}

template <>
void cpu_sytrf<double>(rocblas_fill uplo,
                       rocblas_int n,
                       double* A,
                       rocblas_int lda,
                       rocblas_int* ipiv,
                       double* work,
                       rocblas_int lwork,
                       rocblas_int* info)
{
    char uploC = rocblas2char_fill(uplo);
    dsytrf_(&uploC, &n, A, &lda, ipiv, work, &lwork, info);
}

template <>
void cpu_sytrf<rocblas_float_complex>(rocblas_fill uplo,
                                      rocblas_int n,
                                      rocblas_float_complex* A,
                                      rocblas_int lda,
                                      rocblas_int* ipiv,
                                      rocblas_float_complex* work,
                                      rocblas_int lwork,
                                      rocblas_int* info)
{
    char uploC = rocblas2char_fill(uplo);
    csytrf_(&uploC, &n, A, &lda, ipiv, work, &lwork, info);
}

template <>
void cpu_sytrf<rocblas_double_complex>(rocblas_fill uplo,
                                       rocblas_int n,
                                       rocblas_double_complex* A,
                                       rocblas_int lda,
                                       rocblas_int* ipiv,
                                       rocblas_double_complex* work,
                                       rocblas_int lwork,
                                       rocblas_int* info)
{
    char uploC = rocblas2char_fill(uplo);
    zsytrf_(&uploC, &n, A, &lda, ipiv, work, &lwork, info);
}

// bdsvdx
template <>
void cpu_bdsvdx<float>(rocblas_fill uplo,
                       rocblas_svect svect,
                       rocblas_srange srange,
                       rocblas_int n,
                       float* D,
                       float* E,
                       float vl,
                       float vu,
                       rocblas_int il,
                       rocblas_int iu,
                       rocblas_int* nsv,
                       float* S,
                       float* Z,
                       rocblas_int ldz,
                       float* work,
                       rocblas_int* iwork,
                       rocblas_int* info)
{
    char uploC = rocblas2char_fill(uplo);
    char svectC = rocblas2char_svect(svect, true);
    char srangeC = rocblas2char_srange(srange);
    sbdsvdx_(&uploC, &svectC, &srangeC, &n, D, E, &vl, &vu, &il, &iu, nsv, S, Z, &ldz, work, iwork,
             info);
}

template <>
void cpu_bdsvdx<double>(rocblas_fill uplo,
                        rocblas_svect svect,
                        rocblas_srange srange,
                        rocblas_int n,
                        double* D,
                        double* E,
                        double vl,
                        double vu,
                        rocblas_int il,
                        rocblas_int iu,
                        rocblas_int* nsv,
                        double* S,
                        double* Z,
                        rocblas_int ldz,
                        double* work,
                        rocblas_int* iwork,
                        rocblas_int* info)
{
    char uploC = rocblas2char_fill(uplo);
    char svectC = rocblas2char_svect(svect, true);
    char srangeC = rocblas2char_srange(srange);
    dbdsvdx_(&uploC, &svectC, &srangeC, &n, D, E, &vl, &vu, &il, &iu, nsv, S, Z, &ldz, work, iwork,
             info);
}
