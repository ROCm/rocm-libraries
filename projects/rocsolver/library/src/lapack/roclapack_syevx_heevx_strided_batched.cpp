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

#include "roclapack_syevx_heevx.hpp"

ROCSOLVER_BEGIN_NAMESPACE

template <typename T, typename S, typename U>
rocblas_status rocsolver_syevx_heevx_strided_batched_impl(rocblas_handle handle,
                                                          const rocblas_evect evect,
                                                          const rocblas_erange erange,
                                                          const rocblas_fill uplo,
                                                          const rocblas_int n,
                                                          U A,
                                                          const rocblas_int lda,
                                                          const rocblas_stride strideA,
                                                          const S vl,
                                                          const S vu,
                                                          const rocblas_int il,
                                                          const rocblas_int iu,
                                                          const S abstol,
                                                          rocblas_int* nev,
                                                          S* W,
                                                          const rocblas_stride strideW,
                                                          U Z,
                                                          const rocblas_int ldz,
                                                          const rocblas_stride strideZ,
                                                          rocblas_int* ifail,
                                                          const rocblas_stride strideF,
                                                          rocblas_int* info,
                                                          const rocblas_int batch_count)
{
    const char* name = (!rocblas_is_complex<T> ? "syevx_strided_batched" : "heevx_strided_batched");
    ROCSOLVER_ENTER_TOP(name, "--evect", evect, "--erange", erange, "--uplo", uplo, "-n", n,
                        "--lda", lda, "--strideA", strideA, "--vl", vl, "--vu", vu, "--il", il,
                        "--iu", iu, "--abstol", abstol, "--strideW", strideW, "--ldz", ldz,
                        "--strideZ", strideZ, "--strideF", strideF, "--batch_count", batch_count);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st
        = rocsolver_syevx_heevx_argCheck(handle, evect, erange, uplo, n, A, lda, vl, vu, il, iu,
                                         nev, W, Z, ldz, ifail, info, batch_count);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_int shiftA = 0;
    rocblas_int shiftZ = 0;

    // memory workspace sizes:
    // size for constants in rocblas calls
    size_t size_scalars;
    // size of reusable workspaces (for calling SYTRD/HETRD, STEBZ, STEIN, and ORMTR/UNMTR)
    size_t size_work1, size_work2, size_work3, size_work4, size_work5, size_work6;
    // size for temporary arrays
    size_t size_D, size_E, size_iblock, size_isplit_map, size_tau, size_nsplit_workArr;

    rocsolver_syevx_heevx_getMemorySize<false, T, S>(
        evect, uplo, n, batch_count, &size_scalars, &size_work1, &size_work2, &size_work3,
        &size_work4, &size_work5, &size_work6, &size_D, &size_E, &size_iblock, &size_isplit_map,
        &size_tau, &size_nsplit_workArr);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(
            handle, size_scalars, size_work1, size_work2, size_work3, size_work4, size_work5,
            size_work6, size_D, size_E, size_iblock, size_isplit_map, size_tau, size_nsplit_workArr);

    // memory workspace allocation
    void *scalars, *work1, *work2, *work3, *work4, *work5, *work6, *D, *E, *iblock, *isplit_map,
        *tau, *nsplit_workArr;
    rocblas_device_malloc mem(handle, size_scalars, size_work1, size_work2, size_work3, size_work4,
                              size_work5, size_work6, size_D, size_E, size_iblock, size_isplit_map,
                              size_tau, size_nsplit_workArr);

    if(!mem)
        return rocblas_status_memory_error;

    scalars = mem[0];
    work1 = mem[1];
    work2 = mem[2];
    work3 = mem[3];
    work4 = mem[4];
    work5 = mem[5];
    work6 = mem[6];
    D = mem[7];
    E = mem[8];
    iblock = mem[9];
    isplit_map = mem[10];
    tau = mem[11];
    nsplit_workArr = mem[12];
    if(size_scalars > 0)
        init_scalars(handle, (T*)scalars);

    // execution
    return rocsolver_syevx_heevx_template<false, true, T>(
        handle, evect, erange, uplo, n, A, shiftA, lda, strideA, vl, vu, il, iu, abstol, nev, W,
        strideW, Z, shiftZ, ldz, strideZ, ifail, strideF, info, batch_count, (T*)scalars, work1,
        work2, work3, work4, work5, work6, (S*)D, (S*)E, (rocblas_int*)iblock,
        (rocblas_int*)isplit_map, (T*)tau, nsplit_workArr);
}

ROCSOLVER_END_NAMESPACE

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocsolver_ssyevx_strided_batched(rocblas_handle handle,
                                                const rocblas_evect evect,
                                                const rocblas_erange erange,
                                                const rocblas_fill uplo,
                                                const rocblas_int n,
                                                float* A,
                                                const rocblas_int lda,
                                                const rocblas_stride strideA,
                                                const float vl,
                                                const float vu,
                                                const rocblas_int il,
                                                const rocblas_int iu,
                                                const float abstol,
                                                rocblas_int* nev,
                                                float* W,
                                                const rocblas_stride strideW,
                                                float* Z,
                                                const rocblas_int ldz,
                                                const rocblas_stride strideZ,
                                                rocblas_int* ifail,
                                                const rocblas_stride strideF,
                                                rocblas_int* info,
                                                const rocblas_int batch_count)
{
    return rocsolver::rocsolver_syevx_heevx_strided_batched_impl<float>(
        handle, evect, erange, uplo, n, A, lda, strideA, vl, vu, il, iu, abstol, nev, W, strideW, Z,
        ldz, strideZ, ifail, strideF, info, batch_count);
}

rocblas_status rocsolver_dsyevx_strided_batched(rocblas_handle handle,
                                                const rocblas_evect evect,
                                                const rocblas_erange erange,
                                                const rocblas_fill uplo,
                                                const rocblas_int n,
                                                double* A,
                                                const rocblas_int lda,
                                                const rocblas_stride strideA,
                                                const double vl,
                                                const double vu,
                                                const rocblas_int il,
                                                const rocblas_int iu,
                                                const double abstol,
                                                rocblas_int* nev,
                                                double* W,
                                                const rocblas_stride strideW,
                                                double* Z,
                                                const rocblas_int ldz,
                                                const rocblas_stride strideZ,
                                                rocblas_int* ifail,
                                                const rocblas_stride strideF,
                                                rocblas_int* info,
                                                const rocblas_int batch_count)
{
    return rocsolver::rocsolver_syevx_heevx_strided_batched_impl<double>(
        handle, evect, erange, uplo, n, A, lda, strideA, vl, vu, il, iu, abstol, nev, W, strideW, Z,
        ldz, strideZ, ifail, strideF, info, batch_count);
}

rocblas_status rocsolver_cheevx_strided_batched(rocblas_handle handle,
                                                const rocblas_evect evect,
                                                const rocblas_erange erange,
                                                const rocblas_fill uplo,
                                                const rocblas_int n,
                                                rocblas_float_complex* A,
                                                const rocblas_int lda,
                                                const rocblas_stride strideA,
                                                const float vl,
                                                const float vu,
                                                const rocblas_int il,
                                                const rocblas_int iu,
                                                const float abstol,
                                                rocblas_int* nev,
                                                float* W,
                                                const rocblas_stride strideW,
                                                rocblas_float_complex* Z,
                                                const rocblas_int ldz,
                                                const rocblas_stride strideZ,
                                                rocblas_int* ifail,
                                                const rocblas_stride strideF,
                                                rocblas_int* info,
                                                const rocblas_int batch_count)
{
    return rocsolver::rocsolver_syevx_heevx_strided_batched_impl<rocblas_float_complex>(
        handle, evect, erange, uplo, n, A, lda, strideA, vl, vu, il, iu, abstol, nev, W, strideW, Z,
        ldz, strideZ, ifail, strideF, info, batch_count);
}

rocblas_status rocsolver_zheevx_strided_batched(rocblas_handle handle,
                                                const rocblas_evect evect,
                                                const rocblas_erange erange,
                                                const rocblas_fill uplo,
                                                const rocblas_int n,
                                                rocblas_double_complex* A,
                                                const rocblas_int lda,
                                                const rocblas_stride strideA,
                                                const double vl,
                                                const double vu,
                                                const rocblas_int il,
                                                const rocblas_int iu,
                                                const double abstol,
                                                rocblas_int* nev,
                                                double* W,
                                                const rocblas_stride strideW,
                                                rocblas_double_complex* Z,
                                                const rocblas_int ldz,
                                                const rocblas_stride strideZ,
                                                rocblas_int* ifail,
                                                const rocblas_stride strideF,
                                                rocblas_int* info,
                                                const rocblas_int batch_count)
{
    return rocsolver::rocsolver_syevx_heevx_strided_batched_impl<rocblas_double_complex>(
        handle, evect, erange, uplo, n, A, lda, strideA, vl, vu, il, iu, abstol, nev, W, strideW, Z,
        ldz, strideZ, ifail, strideF, info, batch_count);
}

} // extern C
