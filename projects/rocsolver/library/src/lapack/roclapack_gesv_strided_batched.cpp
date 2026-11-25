/* **************************************************************************
 * Copyright (C) 2019-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "exceptions.hpp"
#include "roclapack_gesv.hpp"

ROCSOLVER_BEGIN_NAMESPACE

template <typename T, typename I, typename U>
rocblas_status rocsolver_gesv_strided_batched_impl(rocblas_handle handle,
                                                   const I n,
                                                   const I nrhs,
                                                   U A,
                                                   const I lda,
                                                   const rocblas_stride strideA,
                                                   I* ipiv,
                                                   const rocblas_stride strideP,
                                                   U B,
                                                   const I ldb,
                                                   const rocblas_stride strideB,
                                                   I* info,
                                                   const I batch_count_arg)
try
{
    ROCSOLVER_ENTER_TOP("gesv_strided_batched", "-n", n, "--nrhs", nrhs, "--lda", lda, "--strideA",
                        strideA, "--strideP", strideP, "--ldb", ldb, "--strideB", strideB,
                        "--batch_count", batch_count_arg);

    using S = decltype(std::real(T{}));

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st
        = rocsolver_gesv_argCheck(handle, n, nrhs, lda, ldb, A, B, ipiv, info, batch_count_arg);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    I shiftA = 0;
    I shiftB = 0;

    // split batch
    I const nsweep = ceildiv(batch_count_arg, MAX_BATCH_COUNT);
    I const batch_count_split = (nsweep == 0) ? 0 : ceildiv(batch_count_arg, nsweep);

    // memory workspace sizes:
    rocsolver_workspace_helper work_helper;
    rocsolver_gesv_getMemorySize<false, true, T>(handle, n, nrhs, A, shiftA, lda, strideA, ipiv,
                                                 strideP, B, shiftB, ldb, strideB, info,
                                                 batch_count_split, &work_helper);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle, work_helper.get_total_size<T>());

    // memory workspace allocation
    rocblas_device_malloc mem(handle, work_helper.get_total_size<T>());

    if(!mem)
        return rocblas_status_memory_error;

    ROCBLAS_CHECK(work_helper.assign_buffer<T>(handle, mem[0]));

    for(I isweep = 0; isweep < nsweep; isweep++)
    {
        I const bid = isweep * batch_count_split;
        I const batch_count = std::min(batch_count_arg - bid, batch_count_split);

        auto const A_sweep = (A == nullptr) ? nullptr : A + bid * strideA;
        auto const B_sweep = (B == nullptr) ? nullptr : B + bid * strideB;
        auto const ipiv_sweep = (ipiv == nullptr) ? nullptr : ipiv + bid * strideP;
        auto const info_sweep = (info == nullptr) ? nullptr : info + bid;

        // execution
        ROCBLAS_CHECK(rocsolver_gesv_template<false, true, T>(
            handle, n, nrhs, A_sweep, shiftA, lda, strideA, ipiv_sweep, strideP, B_sweep, shiftB,
            ldb, strideB, info_sweep, batch_count, &work_helper));
    }

    return rocblas_status_success;
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

extern "C" rocblas_status rocsolver_sgesv_strided_batched(rocblas_handle handle,
                                                          const rocblas_int n,
                                                          const rocblas_int nrhs,
                                                          float* A,
                                                          const rocblas_int lda,
                                                          const rocblas_stride strideA,
                                                          rocblas_int* ipiv,
                                                          const rocblas_stride strideP,
                                                          float* B,
                                                          const rocblas_int ldb,
                                                          const rocblas_stride strideB,
                                                          rocblas_int* info,
                                                          const rocblas_int batch_count)
{
    return rocsolver::rocsolver_gesv_strided_batched_impl<float>(
        handle, n, nrhs, A, lda, strideA, ipiv, strideP, B, ldb, strideB, info, batch_count);
}

extern "C" rocblas_status rocsolver_dgesv_strided_batched(rocblas_handle handle,
                                                          const rocblas_int n,
                                                          const rocblas_int nrhs,
                                                          double* A,
                                                          const rocblas_int lda,
                                                          const rocblas_stride strideA,
                                                          rocblas_int* ipiv,
                                                          const rocblas_stride strideP,
                                                          double* B,
                                                          const rocblas_int ldb,
                                                          const rocblas_stride strideB,
                                                          rocblas_int* info,
                                                          const rocblas_int batch_count)
{
    return rocsolver::rocsolver_gesv_strided_batched_impl<double>(
        handle, n, nrhs, A, lda, strideA, ipiv, strideP, B, ldb, strideB, info, batch_count);
}

extern "C" rocblas_status rocsolver_cgesv_strided_batched(rocblas_handle handle,
                                                          const rocblas_int n,
                                                          const rocblas_int nrhs,
                                                          rocblas_float_complex* A,
                                                          const rocblas_int lda,
                                                          const rocblas_stride strideA,
                                                          rocblas_int* ipiv,
                                                          const rocblas_stride strideP,
                                                          rocblas_float_complex* B,
                                                          const rocblas_int ldb,
                                                          const rocblas_stride strideB,
                                                          rocblas_int* info,
                                                          const rocblas_int batch_count)
{
    return rocsolver::rocsolver_gesv_strided_batched_impl<rocblas_float_complex>(
        handle, n, nrhs, A, lda, strideA, ipiv, strideP, B, ldb, strideB, info, batch_count);
}

extern "C" rocblas_status rocsolver_zgesv_strided_batched(rocblas_handle handle,
                                                          const rocblas_int n,
                                                          const rocblas_int nrhs,
                                                          rocblas_double_complex* A,
                                                          const rocblas_int lda,
                                                          const rocblas_stride strideA,
                                                          rocblas_int* ipiv,
                                                          const rocblas_stride strideP,
                                                          rocblas_double_complex* B,
                                                          const rocblas_int ldb,
                                                          const rocblas_stride strideB,
                                                          rocblas_int* info,
                                                          const rocblas_int batch_count)
{
    return rocsolver::rocsolver_gesv_strided_batched_impl<rocblas_double_complex>(
        handle, n, nrhs, A, lda, strideA, ipiv, strideP, B, ldb, strideB, info, batch_count);
}
