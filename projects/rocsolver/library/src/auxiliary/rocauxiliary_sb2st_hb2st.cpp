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

#include "rocauxiliary_sb2st_hb2st.hpp"
#include "exceptions.hpp"

ROCSOLVER_BEGIN_NAMESPACE

/*
 * ===========================================================================
 *    hb2st is not intended for inclusion in the public API.
 *    It is used as a step in eigenvalue solvers (heev*).
 * ===========================================================================
 */

//------------------------------------------------------------------------------
// Reduces Hermitian/symmetric band matrix A to real symmetric tridiagonal form
// by a unitary similarity transformation:
//      Q^H A_band Q = A_tri.
//
//  handle      Rocblas_handle.
//  uplo        Lower or upper; only lower currently supported.
//  n           Matrix dimension. n >= 0.
//  kd          Matrix bandwidth. kd >= 1.
//  Aband       n-by-n band matrix, with space for kd - 1 super-diagonals and
//              2*kd - 1 sub-diagonals. On input, diagonal and kd sub-diagonals
//              must be set.
//  ldab        Leading dimension of Aband. ldab >= 3*kd - 1.
//  D           Vector of length n. On output, diagonal of tridiagonal A_tri.
//  E           Vector of length n-1. On output, sub-diagonal of tridiagonal A_tri.
//  V           Array of Householder vectors, of size ldv*nv, where
//              number of tiles nt = ceil( (n - 1) / kd ), and
//              number of vectors nv = kd*nt*(nt + 1)/2.
//  ldv         Leading dimension of V. ldv >= 2*kd - 1.
//  tau         Vector of Householder tau factors, of length nv.
//
template <typename T, typename I, typename S, typename U>
rocblas_status rocsolver_sb2st_hb2st_impl(rocblas_handle handle,
                                          rocblas_fill uplo,
                                          const I n,
                                          const I kd,
                                          U Aband,
                                          const I ldab,
                                          S* D,
                                          S* E,
                                          U V,
                                          const I ldv,
                                          T* tau)
try
{
    ROCSOLVER_ENTER_TOP("sb2st_hb2st", "-n", n, "--kd", kd, "--ldab", ldab, "--ldv", ldv);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st
        = rocsolver_sb2st_hb2st_argCheck(handle, uplo, n, kd, ldab, ldv, Aband, D, E, V, tau);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_stride shiftA = 0;

    // normal (non-batched non-strided) execution
    rocblas_stride strideA = 0;
    rocblas_stride strideD = 0;
    rocblas_stride strideE = 0;
    rocblas_stride strideV = 0;
    rocblas_stride strideTau = 0;
    I batch_count = 1;

    // memory workspace sizes:
    // size of reusable workspace
    size_t size_work;
    rocsolver_sb2st_hb2st_getMemorySize<false, T, I, S>(n, kd, batch_count, &size_work);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle, size_work);

    // no memory workspace allocation
    assert(size_work == 0);

    // execution
    return rocsolver_sb2st_hb2st_template<false, false, T, I>(
        handle, uplo, n, kd, Aband, shiftA, ldab, strideA, D, strideD, E, strideE, V, ldv, strideV,
        tau, strideTau, batch_count);
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

ROCSOLVER_EXPORT rocblas_status rocsolver_ssb2st(rocblas_handle handle,
                                                 rocblas_fill uplo,
                                                 const rocblas_int n,
                                                 const rocblas_int kd,
                                                 float* Aband,
                                                 const rocblas_int ldab,
                                                 float* D,
                                                 float* E,
                                                 float* V,
                                                 const rocblas_int ldv,
                                                 float* tau)
{
#ifdef ROCSOLVER_ENABLE_EIG_2STAGE
    return rocsolver::rocsolver_sb2st_hb2st_impl<float, rocblas_int>(handle, uplo, n, kd, Aband,
                                                                     ldab, D, E, V, ldv, tau);
#else
    return rocblas_status_not_implemented;
#endif
}

ROCSOLVER_EXPORT rocblas_status rocsolver_dsb2st(rocblas_handle handle,
                                                 rocblas_fill uplo,
                                                 const rocblas_int n,
                                                 const rocblas_int kd,
                                                 double* Aband,
                                                 const rocblas_int ldab,
                                                 double* D,
                                                 double* E,
                                                 double* V,
                                                 const rocblas_int ldv,
                                                 double* tau)
{
#ifdef ROCSOLVER_ENABLE_EIG_2STAGE
    return rocsolver::rocsolver_sb2st_hb2st_impl<double, rocblas_int>(handle, uplo, n, kd, Aband,
                                                                      ldab, D, E, V, ldv, tau);
#else
    return rocblas_status_not_implemented;
#endif
}

ROCSOLVER_EXPORT rocblas_status rocsolver_chb2st(rocblas_handle handle,
                                                 rocblas_fill uplo,
                                                 const rocblas_int n,
                                                 const rocblas_int kd,
                                                 rocblas_float_complex* Aband,
                                                 const rocblas_int ldab,
                                                 float* D,
                                                 float* E,
                                                 rocblas_float_complex* V,
                                                 const rocblas_int ldv,
                                                 rocblas_float_complex* tau)
{
#ifdef ROCSOLVER_ENABLE_EIG_2STAGE
    return rocsolver::rocsolver_sb2st_hb2st_impl<rocblas_float_complex, rocblas_int>(
        handle, uplo, n, kd, Aband, ldab, D, E, V, ldv, tau);
#else
    return rocblas_status_not_implemented;
#endif
}

ROCSOLVER_EXPORT rocblas_status rocsolver_zhb2st(rocblas_handle handle,
                                                 rocblas_fill uplo,
                                                 const rocblas_int n,
                                                 const rocblas_int kd,
                                                 rocblas_double_complex* Aband,
                                                 const rocblas_int ldab,
                                                 double* D,
                                                 double* E,
                                                 rocblas_double_complex* V,
                                                 const rocblas_int ldv,
                                                 rocblas_double_complex* tau)
{
#ifdef ROCSOLVER_ENABLE_EIG_2STAGE
    return rocsolver::rocsolver_sb2st_hb2st_impl<rocblas_double_complex, rocblas_int>(
        handle, uplo, n, kd, Aband, ldab, D, E, V, ldv, tau);
#else
    return rocblas_status_not_implemented;
#endif
}

ROCSOLVER_EXPORT rocblas_status rocsolver_ssb2st_64(rocblas_handle handle,
                                                    rocblas_fill uplo,
                                                    const int64_t n,
                                                    const int64_t kd,
                                                    float* Aband,
                                                    const int64_t ldab,
                                                    float* D,
                                                    float* E,
                                                    float* V,
                                                    const int64_t ldv,
                                                    float* tau)
{
#if defined(ROCSOLVER_ENABLE_EIG_2STAGE) && defined(HAVE_ROCBLAS_64)
    return rocsolver::rocsolver_sb2st_hb2st_impl<float, int64_t>(handle, uplo, n, kd, Aband, ldab,
                                                                 D, E, V, ldv, tau);
#else
    return rocblas_status_not_implemented;
#endif
}

ROCSOLVER_EXPORT rocblas_status rocsolver_dsb2st_64(rocblas_handle handle,
                                                    rocblas_fill uplo,
                                                    const int64_t n,
                                                    const int64_t kd,
                                                    double* Aband,
                                                    const int64_t ldab,
                                                    double* D,
                                                    double* E,
                                                    double* V,
                                                    const int64_t ldv,
                                                    double* tau)
{
#if defined(ROCSOLVER_ENABLE_EIG_2STAGE) && defined(HAVE_ROCBLAS_64)
    return rocsolver::rocsolver_sb2st_hb2st_impl<double, int64_t>(handle, uplo, n, kd, Aband, ldab,
                                                                  D, E, V, ldv, tau);
#else
    return rocblas_status_not_implemented;
#endif
}

ROCSOLVER_EXPORT rocblas_status rocsolver_chb2st_64(rocblas_handle handle,
                                                    rocblas_fill uplo,
                                                    const int64_t n,
                                                    const int64_t kd,
                                                    rocblas_float_complex* Aband,
                                                    const int64_t ldab,
                                                    float* D,
                                                    float* E,
                                                    rocblas_float_complex* V,
                                                    const int64_t ldv,
                                                    rocblas_float_complex* tau)
{
#if defined(ROCSOLVER_ENABLE_EIG_2STAGE) && defined(HAVE_ROCBLAS_64)
    return rocsolver::rocsolver_sb2st_hb2st_impl<rocblas_float_complex, int64_t>(
        handle, uplo, n, kd, Aband, ldab, D, E, V, ldv, tau);
#else
    return rocblas_status_not_implemented;
#endif
}

ROCSOLVER_EXPORT rocblas_status rocsolver_zhb2st_64(rocblas_handle handle,
                                                    rocblas_fill uplo,
                                                    const int64_t n,
                                                    const int64_t kd,
                                                    rocblas_double_complex* Aband,
                                                    const int64_t ldab,
                                                    double* D,
                                                    double* E,
                                                    rocblas_double_complex* V,
                                                    const int64_t ldv,
                                                    rocblas_double_complex* tau)
{
#if defined(ROCSOLVER_ENABLE_EIG_2STAGE) && defined(HAVE_ROCBLAS_64)
    return rocsolver::rocsolver_sb2st_hb2st_impl<rocblas_double_complex, int64_t>(
        handle, uplo, n, kd, Aband, ldab, D, E, V, ldv, tau);
#else
    return rocblas_status_not_implemented;
#endif
}

} // extern C
