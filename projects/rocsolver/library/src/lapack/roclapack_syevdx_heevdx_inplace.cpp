/* **************************************************************************
 * Copyright (C) 2021-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include "roclapack_syevdx_heevdx_inplace.hpp"

ROCSOLVER_BEGIN_NAMESPACE

/*
 * ===========================================================================
 *    syevdx/heevdx_inplace is not intended for inclusion in the public API. It
 *    exists to provide a syevdx/heevdx method with a signature identical to
 *    the cuSOLVER implementation, for use exclusively in hipSOLVER.
 *
 *    TODO: The current implementation is based on syevx. It will need to be
 *    updated to syevdx at a later date.
 * ===========================================================================
 */

template <typename T, typename S, typename U>
rocblas_status rocsolver_syevdx_heevdx_inplace_impl(rocblas_handle handle,
                                                    const rocblas_evect evect,
                                                    const rocblas_erange erange,
                                                    const rocblas_fill uplo,
                                                    const rocblas_int n,
                                                    U A,
                                                    const rocblas_int lda,
                                                    const S vl,
                                                    const S vu,
                                                    const rocblas_int il,
                                                    const rocblas_int iu,
                                                    const S abstol,
                                                    rocblas_int* h_nev,
                                                    S* W,
                                                    rocblas_int* info)
{
    const char* name = (!rocblas_is_complex<T> ? "syevdx_inplace" : "heevdx_inplace");
    ROCSOLVER_ENTER_TOP(name, "--evect", evect, "--erange", erange, "--uplo", uplo, "-n", n, "--lda",
                        lda, "--vl", vl, "--vu", vu, "--il", il, "--iu", iu, "--abstol", abstol);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st = rocsolver_syevdx_heevdx_inplace_argCheck(
        handle, evect, erange, uplo, n, A, lda, vl, vu, il, iu, h_nev, W, info);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_int shiftA = 0;

    // normal (non-batched non-strided) execution
    rocblas_stride strideA = 0;
    rocblas_stride strideW = 0;
    rocblas_int batch_count = 1;

    // memory workspace sizes:
    // size for constants in rocblas calls
    size_t size_scalars;
    // size of reusable workspaces (for calling SYTRD/HETRD, STEBZ, STEIN, and ORMTR/UNMTR)
    size_t size_work1, size_work2, size_work3, size_work4, size_work5, size_work6_ifail;
    // size for temporary arrays
    size_t size_D, size_E, size_iblock, size_isplit_map, size_tau, size_nev, size_nsplit_workArr;

    rocsolver_syevdx_heevdx_inplace_getMemorySize<false, T, S>(
        evect, uplo, n, batch_count, &size_scalars, &size_work1, &size_work2, &size_work3,
        &size_work4, &size_work5, &size_work6_ifail, &size_D, &size_E, &size_iblock,
        &size_isplit_map, &size_tau, &size_nev, &size_nsplit_workArr);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(
            handle, size_scalars, size_work1, size_work2, size_work3, size_work4, size_work5,
            size_work6_ifail, size_D, size_E, size_iblock, size_isplit_map, size_tau, size_nev,
            size_nsplit_workArr);

    // memory workspace allocation
    void *scalars, *work1, *work2, *work3, *work4, *work5, *work6_ifail, *D, *E, *iblock,
        *isplit_map, *tau, *d_nev, *nsplit_workArr;
    rocblas_device_malloc mem(handle, size_scalars, size_work1, size_work2, size_work3, size_work4,
                              size_work5, size_work6_ifail, size_D, size_E, size_iblock,
                              size_isplit_map, size_tau, size_nev, size_nsplit_workArr);

    if(!mem)
        return rocblas_status_memory_error;

    scalars = mem[0];
    work1 = mem[1];
    work2 = mem[2];
    work3 = mem[3];
    work4 = mem[4];
    work5 = mem[5];
    work6_ifail = mem[6];
    D = mem[7];
    E = mem[8];
    iblock = mem[9];
    isplit_map = mem[10];
    tau = mem[11];
    d_nev = mem[12];
    nsplit_workArr = mem[13];
    if(size_scalars > 0)
        init_scalars(handle, (T*)scalars);

    // execution
    return rocsolver_syevdx_heevdx_inplace_template<false, false, T>(
        handle, evect, erange, uplo, n, A, shiftA, lda, strideA, vl, vu, il, iu, abstol, h_nev, W,
        strideW, info, batch_count, (T*)scalars, work1, work2, work3, work4, work5,
        (rocblas_int*)work6_ifail, (S*)D, (S*)E, (rocblas_int*)iblock, (rocblas_int*)isplit_map,
        (T*)tau, (rocblas_int*)d_nev, nsplit_workArr);
}

ROCSOLVER_END_NAMESPACE

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

ROCSOLVER_EXPORT rocblas_status rocsolver_ssyevdx_inplace(rocblas_handle handle,
                                                          const rocblas_evect evect,
                                                          const rocblas_erange erange,
                                                          const rocblas_fill uplo,
                                                          const rocblas_int n,
                                                          float* A,
                                                          const rocblas_int lda,
                                                          const float vl,
                                                          const float vu,
                                                          const rocblas_int il,
                                                          const rocblas_int iu,
                                                          const float abstol,
                                                          rocblas_int* h_nev,
                                                          float* W,
                                                          rocblas_int* info)
{
    return rocsolver::rocsolver_syevdx_heevdx_inplace_impl<float>(
        handle, evect, erange, uplo, n, A, lda, vl, vu, il, iu, abstol, h_nev, W, info);
}

ROCSOLVER_EXPORT rocblas_status rocsolver_dsyevdx_inplace(rocblas_handle handle,
                                                          const rocblas_evect evect,
                                                          const rocblas_erange erange,
                                                          const rocblas_fill uplo,
                                                          const rocblas_int n,
                                                          double* A,
                                                          const rocblas_int lda,
                                                          const double vl,
                                                          const double vu,
                                                          const rocblas_int il,
                                                          const rocblas_int iu,
                                                          const double abstol,
                                                          rocblas_int* h_nev,
                                                          double* W,
                                                          rocblas_int* info)
{
    return rocsolver::rocsolver_syevdx_heevdx_inplace_impl<double>(
        handle, evect, erange, uplo, n, A, lda, vl, vu, il, iu, abstol, h_nev, W, info);
}

ROCSOLVER_EXPORT rocblas_status rocsolver_cheevdx_inplace(rocblas_handle handle,
                                                          const rocblas_evect evect,
                                                          const rocblas_erange erange,
                                                          const rocblas_fill uplo,
                                                          const rocblas_int n,
                                                          rocblas_float_complex* A,
                                                          const rocblas_int lda,
                                                          const float vl,
                                                          const float vu,
                                                          const rocblas_int il,
                                                          const rocblas_int iu,
                                                          const float abstol,
                                                          rocblas_int* h_nev,
                                                          float* W,
                                                          rocblas_int* info)
{
    return rocsolver::rocsolver_syevdx_heevdx_inplace_impl<rocblas_float_complex>(
        handle, evect, erange, uplo, n, A, lda, vl, vu, il, iu, abstol, h_nev, W, info);
}

ROCSOLVER_EXPORT rocblas_status rocsolver_zheevdx_inplace(rocblas_handle handle,
                                                          const rocblas_evect evect,
                                                          const rocblas_erange erange,
                                                          const rocblas_fill uplo,
                                                          const rocblas_int n,
                                                          rocblas_double_complex* A,
                                                          const rocblas_int lda,
                                                          const double vl,
                                                          const double vu,
                                                          const rocblas_int il,
                                                          const rocblas_int iu,
                                                          const double abstol,
                                                          rocblas_int* h_nev,
                                                          double* W,
                                                          rocblas_int* info)
{
    return rocsolver::rocsolver_syevdx_heevdx_inplace_impl<rocblas_double_complex>(
        handle, evect, erange, uplo, n, A, lda, vl, vu, il, iu, abstol, h_nev, W, info);
}

} // extern C
