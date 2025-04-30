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

#include "roclapack_sygvx_hegvx.hpp"

ROCSOLVER_BEGIN_NAMESPACE

template <typename T, typename S, typename U>
rocblas_status rocsolver_sygvx_hegvx_batched_impl(rocblas_handle handle,
                                                  const rocblas_eform itype,
                                                  const rocblas_evect evect,
                                                  const rocblas_erange erange,
                                                  const rocblas_fill uplo,
                                                  const rocblas_int n,
                                                  U A,
                                                  const rocblas_int lda,
                                                  U B,
                                                  const rocblas_int ldb,
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
                                                  rocblas_int* ifail,
                                                  const rocblas_stride strideF,
                                                  rocblas_int* info,
                                                  const rocblas_int batch_count)
{
    const char* name = (!rocblas_is_complex<T> ? "sygvx_batched" : "hegvx_batched");
    ROCSOLVER_ENTER_TOP(name, "--itype", itype, "--evect", evect, "--erange", erange, "--uplo",
                        uplo, "-n", n, "--lda", lda, "--ldb", ldb, "--vl", vl, "--vu", vu, "--il",
                        il, "--iu", iu, "--abstol", abstol, "--strideW", strideW, "--ldz", ldz,
                        "--strideF", strideF, "--batch_count", batch_count);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st
        = rocsolver_sygvx_hegvx_argCheck(handle, itype, evect, erange, uplo, n, A, lda, B, ldb, vl,
                                         vu, il, iu, nev, W, Z, ldz, ifail, info, batch_count);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_int shiftA = 0;
    rocblas_int shiftB = 0;
    rocblas_int shiftZ = 0;

    // batched execution
    rocblas_stride strideA = 0;
    rocblas_stride strideB = 0;
    rocblas_stride strideZ = 0;

    // memory workspace sizes:
    // size for constants in rocblas calls
    size_t size_scalars;
    // size of reusable workspaces (and for calling TRSM, SYGST/HEGST, and SYEVX/HEEVX)
    bool optim_mem;
    size_t size_work1, size_work2, size_work3, size_work4, size_work5, size_work6;
    // extra requirements for calling SYEVX/HEEVX
    size_t size_D, size_E, size_iblock, size_isplit, size_tau;
    // extra requirements for calling POTRF and SYEVX/HEEVX
    size_t size_work7_workArr;
    // size of temporary info array
    size_t size_iinfo;
    rocsolver_sygvx_hegvx_getMemorySize<true, false, T, S>(
        itype, evect, uplo, n, batch_count, &size_scalars, &size_work1, &size_work2, &size_work3,
        &size_work4, &size_work5, &size_work6, &size_D, &size_E, &size_iblock, &size_isplit,
        &size_tau, &size_work7_workArr, &size_iinfo, &optim_mem);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle, size_scalars, size_work1, size_work2,
                                                      size_work3, size_work4, size_work5, size_work6,
                                                      size_D, size_E, size_iblock, size_isplit,
                                                      size_tau, size_work7_workArr, size_iinfo);

    // memory workspace allocation
    void *scalars, *work1, *work2, *work3, *work4, *work5, *work6, *D, *E, *iblock, *isplit, *tau,
        *work7_workArr, *iinfo;
    rocblas_device_malloc mem(handle, size_scalars, size_work1, size_work2, size_work3, size_work4,
                              size_work5, size_work6, size_D, size_E, size_iblock, size_isplit,
                              size_tau, size_work7_workArr, size_iinfo);

    if(!mem)
        return rocblas_status_memory_error;

    scalars = mem[0];
    work1 = mem[1];
    work2 = mem[2];
    work3 = mem[3];
    work4 = mem[4];
    work5 = mem[5], work6 = mem[6], D = mem[7];
    E = mem[8];
    iblock = mem[9];
    isplit = mem[10];
    tau = mem[11];
    work7_workArr = mem[12];
    iinfo = mem[13];
    if(size_scalars > 0)
        init_scalars(handle, (T*)scalars);

    // execution
    return rocsolver_sygvx_hegvx_template<true, false, T>(
        handle, itype, evect, erange, uplo, n, A, shiftA, lda, strideA, B, shiftB, ldb, strideB, vl,
        vu, il, iu, abstol, nev, W, strideW, Z, shiftZ, ldz, strideZ, ifail, strideF, info,
        batch_count, (T*)scalars, work1, work2, work3, work4, work5, work6, (S*)D, (S*)E,
        (rocblas_int*)iblock, (rocblas_int*)isplit, (T*)tau, work7_workArr, (rocblas_int*)iinfo,
        optim_mem);
}

ROCSOLVER_END_NAMESPACE

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocsolver_ssygvx_batched(rocblas_handle handle,
                                        const rocblas_eform itype,
                                        const rocblas_evect evect,
                                        const rocblas_erange erange,
                                        const rocblas_fill uplo,
                                        const rocblas_int n,
                                        float* const A[],
                                        const rocblas_int lda,
                                        float* const B[],
                                        const rocblas_int ldb,
                                        const float vl,
                                        const float vu,
                                        const rocblas_int il,
                                        const rocblas_int iu,
                                        const float abstol,
                                        rocblas_int* nev,
                                        float* W,
                                        const rocblas_stride strideW,
                                        float* const Z[],
                                        const rocblas_int ldz,
                                        rocblas_int* ifail,
                                        const rocblas_stride strideF,
                                        rocblas_int* info,
                                        const rocblas_int batch_count)
{
    return rocsolver::rocsolver_sygvx_hegvx_batched_impl<float>(
        handle, itype, evect, erange, uplo, n, A, lda, B, ldb, vl, vu, il, iu, abstol, nev, W,
        strideW, Z, ldz, ifail, strideF, info, batch_count);
}

rocblas_status rocsolver_dsygvx_batched(rocblas_handle handle,
                                        const rocblas_eform itype,
                                        const rocblas_evect evect,
                                        const rocblas_erange erange,
                                        const rocblas_fill uplo,
                                        const rocblas_int n,
                                        double* const A[],
                                        const rocblas_int lda,
                                        double* const B[],
                                        const rocblas_int ldb,
                                        const double vl,
                                        const double vu,
                                        const rocblas_int il,
                                        const rocblas_int iu,
                                        const double abstol,
                                        rocblas_int* nev,
                                        double* W,
                                        const rocblas_stride strideW,
                                        double* const Z[],
                                        const rocblas_int ldz,
                                        rocblas_int* ifail,
                                        const rocblas_stride strideF,
                                        rocblas_int* info,
                                        const rocblas_int batch_count)
{
    return rocsolver::rocsolver_sygvx_hegvx_batched_impl<double>(
        handle, itype, evect, erange, uplo, n, A, lda, B, ldb, vl, vu, il, iu, abstol, nev, W,
        strideW, Z, ldz, ifail, strideF, info, batch_count);
}

rocblas_status rocsolver_chegvx_batched(rocblas_handle handle,
                                        const rocblas_eform itype,
                                        const rocblas_evect evect,
                                        const rocblas_erange erange,
                                        const rocblas_fill uplo,
                                        const rocblas_int n,
                                        rocblas_float_complex* const A[],
                                        const rocblas_int lda,
                                        rocblas_float_complex* const B[],
                                        const rocblas_int ldb,
                                        const float vl,
                                        const float vu,
                                        const rocblas_int il,
                                        const rocblas_int iu,
                                        const float abstol,
                                        rocblas_int* nev,
                                        float* W,
                                        const rocblas_stride strideW,
                                        rocblas_float_complex* const Z[],
                                        const rocblas_int ldz,
                                        rocblas_int* ifail,
                                        const rocblas_stride strideF,
                                        rocblas_int* info,
                                        const rocblas_int batch_count)
{
    return rocsolver::rocsolver_sygvx_hegvx_batched_impl<rocblas_float_complex>(
        handle, itype, evect, erange, uplo, n, A, lda, B, ldb, vl, vu, il, iu, abstol, nev, W,
        strideW, Z, ldz, ifail, strideF, info, batch_count);
}

rocblas_status rocsolver_zhegvx_batched(rocblas_handle handle,
                                        const rocblas_eform itype,
                                        const rocblas_evect evect,
                                        const rocblas_erange erange,
                                        const rocblas_fill uplo,
                                        const rocblas_int n,
                                        rocblas_double_complex* const A[],
                                        const rocblas_int lda,
                                        rocblas_double_complex* const B[],
                                        const rocblas_int ldb,
                                        const double vl,
                                        const double vu,
                                        const rocblas_int il,
                                        const rocblas_int iu,
                                        const double abstol,
                                        rocblas_int* nev,
                                        double* W,
                                        const rocblas_stride strideW,
                                        rocblas_double_complex* const Z[],
                                        const rocblas_int ldz,
                                        rocblas_int* ifail,
                                        const rocblas_stride strideF,
                                        rocblas_int* info,
                                        const rocblas_int batch_count)
{
    return rocsolver::rocsolver_sygvx_hegvx_batched_impl<rocblas_double_complex>(
        handle, itype, evect, erange, uplo, n, A, lda, B, ldb, vl, vu, il, iu, abstol, nev, W,
        strideW, Z, ldz, ifail, strideF, info, batch_count);
}

} // extern C
