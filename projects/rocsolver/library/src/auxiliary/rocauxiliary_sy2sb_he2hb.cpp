/* **************************************************************************
 * Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "rocauxiliary_sy2sb_he2hb.hpp"
#include "exceptions.hpp"

ROCSOLVER_BEGIN_NAMESPACE

/*
 * ===========================================================================
 *    he2hb is not intended for inclusion in the public API.
 *    It is used as a step in eigenvalue solvers (heev*).
 * ===========================================================================
 */

//------------------------------------------------------------------------------
// Reduces Hermitian/symmetric matrix A to Hermitian/symmetric band form
// by a unitary similarity transformation:
//      Q^H A Q = A_band.
//
//  handle      rocblas_handle.
//  n           Matrix dimension. n >= 0.
//  kd          Matrix bandwidth. kd >= 1.
//  nb          Block size. nb >= kd and nb is a multiple of kd.
//  A           n-by-n Hermitian matrix, with both lower and upper entries set.
//              On output, Householder vectors V overwrite the lower portion of A,
//              sub-diagonal kd and below. The rest of A is destroyed.
//  Aband       n-by-n band matrix, with space for kd - 1 super-diagonals and
//              2*kd - 1 sub-diagonals. On output, the main diagonal and kd
//              sub-diagonals are set; other entries are destroyed.
//  ldab        Leading dimension of Aband. ldab >= 3*kd - 1.
//  tau         Householder tau values, length n - kd.
//
template <typename T, typename I, typename U>
rocblas_status rocsolver_sy2sb_he2hb_impl(rocblas_handle handle,
                                          const rocblas_fill uplo,
                                          const I n,
                                          const I kd,
                                          const I nb,
                                          U A,
                                          const I lda,
                                          T* Aband,
                                          const I ldab,
                                          T* tau)
try
{
    ROCSOLVER_ENTER_TOP("sy2sb_he2hb", "--uplo", uplo, "-n", n, "-kd", kd, "-nb", nb, "--lda", lda,
                        "--ldab", ldab);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st
        = rocsolver_sy2sb_he2hb_argCheck(handle, uplo, n, kd, nb, A, lda, Aband, ldab, tau);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    I shiftA = 0;

    // normal (non-batched non-strided) execution
    rocblas_stride strideA = 0;
    rocblas_stride strideAb = 0;
    rocblas_stride strideTau = 0;
    I batch_count = 1;

    // memory workspace sizes:
    // size for constants in rocblas calls
    size_t size_scalars;
    // size of arrays of pointers (for batched cases) and re-usable workspace
    size_t size_workArr;
    // extra requirements
    size_t size_D, size_V, size_W, size_X, size_Z, size_work;
    rocsolver_sy2sb_he2hb_getMemorySize<false, T, I>(uplo, n, kd, nb, batch_count, &size_scalars,
                                                     &size_D, &size_V, &size_W, &size_X, &size_Z,
                                                     &size_work, &size_workArr);

    if(rocblas_is_device_memory_size_query(handle))
    {
        return rocblas_set_optimal_device_memory_size(handle, size_scalars, size_D, size_V, size_W,
                                                      size_X, size_Z, size_work, size_workArr);
    }

    // memory workspace allocation
    rocblas_device_malloc mem(handle, size_scalars, size_D, size_V, size_W, size_X, size_Z,
                              size_work, size_workArr);

    if(!mem)
        return rocblas_status_memory_error;

    T* scalars = (T*)mem[0];
    T* D = (T*)mem[1];
    T* V = (T*)mem[2];
    T* W = (T*)mem[3];
    T* X = (T*)mem[4];
    T* Z = (T*)mem[5];
    T* work = (T*)mem[6];
    T** workArr = (T**)mem[7];
    if(size_scalars > 0)
        init_scalars(handle, scalars);

    // execution
    return rocsolver_sy2sb_he2hb_template<false, false, T, I>(handle, uplo, n, kd, nb, // opts
                                                              A, shiftA, lda, strideA, // A
                                                              Aband, ldab, strideAb, // Aband
                                                              tau, strideTau, // tau
                                                              batch_count, scalars, D, V, W, X, Z,
                                                              work, workArr);
}
catch(...)
{
    return exception2rocblas_status();
}

ROCSOLVER_END_NAMESPACE

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

ROCSOLVER_EXPORT rocblas_status rocsolver_ssy2sb(rocblas_handle handle,
                                                 const rocblas_fill uplo,
                                                 const rocblas_int n,
                                                 const rocblas_int kd,
                                                 const rocblas_int nb,
                                                 float* A,
                                                 const rocblas_int lda,
                                                 float* Aband,
                                                 const rocblas_int ldab,
                                                 float* tau)
{
#ifdef ROCSOLVER_ENABLE_EIG_2STAGE
    return rocsolver::rocsolver_sy2sb_he2hb_impl<float, rocblas_int>(handle, uplo, n, kd, nb, A,
                                                                     lda, Aband, ldab, tau);
#else
    return rocblas_status_not_implemented;
#endif
}

ROCSOLVER_EXPORT rocblas_status rocsolver_dsy2sb(rocblas_handle handle,
                                                 const rocblas_fill uplo,
                                                 const rocblas_int n,
                                                 const rocblas_int kd,
                                                 const rocblas_int nb,
                                                 double* A,
                                                 const rocblas_int lda,
                                                 double* Aband,
                                                 const rocblas_int ldab,
                                                 double* tau)
{
#ifdef ROCSOLVER_ENABLE_EIG_2STAGE
    return rocsolver::rocsolver_sy2sb_he2hb_impl<double, rocblas_int>(handle, uplo, n, kd, nb, A,
                                                                      lda, Aband, ldab, tau);
#else
    return rocblas_status_not_implemented;
#endif
}

ROCSOLVER_EXPORT rocblas_status rocsolver_che2hb(rocblas_handle handle,
                                                 const rocblas_fill uplo,
                                                 const rocblas_int n,
                                                 const rocblas_int kd,
                                                 const rocblas_int nb,
                                                 rocblas_float_complex* A,
                                                 const rocblas_int lda,
                                                 rocblas_float_complex* Aband,
                                                 const rocblas_int ldab,
                                                 rocblas_float_complex* tau)
{
#ifdef ROCSOLVER_ENABLE_EIG_2STAGE
    return rocsolver::rocsolver_sy2sb_he2hb_impl<rocblas_float_complex, rocblas_int>(
        handle, uplo, n, kd, nb, A, lda, Aband, ldab, tau);
#else
    return rocblas_status_not_implemented;
#endif
}

ROCSOLVER_EXPORT rocblas_status rocsolver_zhe2hb(rocblas_handle handle,
                                                 const rocblas_fill uplo,
                                                 const rocblas_int n,
                                                 const rocblas_int kd,
                                                 const rocblas_int nb,
                                                 rocblas_double_complex* A,
                                                 const rocblas_int lda,
                                                 rocblas_double_complex* Aband,
                                                 const rocblas_int ldab,
                                                 rocblas_double_complex* tau)
{
#ifdef ROCSOLVER_ENABLE_EIG_2STAGE
    return rocsolver::rocsolver_sy2sb_he2hb_impl<rocblas_double_complex, rocblas_int>(
        handle, uplo, n, kd, nb, A, lda, Aband, ldab, tau);
#else
    return rocblas_status_not_implemented;
#endif
}

ROCSOLVER_EXPORT rocblas_status rocsolver_ssy2sb_64(rocblas_handle handle,
                                                    const rocblas_fill uplo,
                                                    const int64_t n,
                                                    const int64_t kd,
                                                    const int64_t nb,
                                                    float* A,
                                                    const int64_t lda,
                                                    float* Aband,
                                                    const int64_t ldab,
                                                    float* tau)
{
#if defined(ROCSOLVER_ENABLE_EIG_2STAGE) && defined(HAVE_ROCBLAS_64)
    return rocsolver::rocsolver_sy2sb_he2hb_impl<float, int64_t>(handle, uplo, n, kd, nb, A, lda,
                                                                 Aband, ldab, tau);
#else
    return rocblas_status_not_implemented;
#endif
}

ROCSOLVER_EXPORT rocblas_status rocsolver_dsy2sb_64(rocblas_handle handle,
                                                    const rocblas_fill uplo,
                                                    const int64_t n,
                                                    const int64_t kd,
                                                    const int64_t nb,
                                                    double* A,
                                                    const int64_t lda,
                                                    double* Aband,
                                                    const int64_t ldab,
                                                    double* tau)
{
#if defined(ROCSOLVER_ENABLE_EIG_2STAGE) && defined(HAVE_ROCBLAS_64)
    return rocsolver::rocsolver_sy2sb_he2hb_impl<double, int64_t>(handle, uplo, n, kd, nb, A, lda,
                                                                  Aband, ldab, tau);
#else
    return rocblas_status_not_implemented;
#endif
}

ROCSOLVER_EXPORT rocblas_status rocsolver_che2hb_64(rocblas_handle handle,
                                                    const rocblas_fill uplo,
                                                    const int64_t n,
                                                    const int64_t kd,
                                                    const int64_t nb,
                                                    rocblas_float_complex* A,
                                                    const int64_t lda,
                                                    rocblas_float_complex* Aband,
                                                    const int64_t ldab,
                                                    rocblas_float_complex* tau)
{
#if defined(ROCSOLVER_ENABLE_EIG_2STAGE) && defined(HAVE_ROCBLAS_64)
    return rocsolver::rocsolver_sy2sb_he2hb_impl<rocblas_float_complex, int64_t>(
        handle, uplo, n, kd, nb, A, lda, Aband, ldab, tau);
#else
    return rocblas_status_not_implemented;
#endif
}

ROCSOLVER_EXPORT rocblas_status rocsolver_zhe2hb_64(rocblas_handle handle,
                                                    const rocblas_fill uplo,
                                                    const int64_t n,
                                                    const int64_t kd,
                                                    const int64_t nb,
                                                    rocblas_double_complex* A,
                                                    const int64_t lda,
                                                    rocblas_double_complex* Aband,
                                                    const int64_t ldab,
                                                    rocblas_double_complex* tau)
{
#if defined(ROCSOLVER_ENABLE_EIG_2STAGE) && defined(HAVE_ROCBLAS_64)
    return rocsolver::rocsolver_sy2sb_he2hb_impl<rocblas_double_complex, int64_t>(
        handle, uplo, n, kd, nb, A, lda, Aband, ldab, tau);
#else
    return rocblas_status_not_implemented;
#endif
}

} // extern C
