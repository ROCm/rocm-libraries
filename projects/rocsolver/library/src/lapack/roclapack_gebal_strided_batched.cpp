/* **************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
#include "roclapack_gebal.hpp"

ROCSOLVER_BEGIN_NAMESPACE

template <typename T, typename I, typename S, typename U>
rocblas_status rocsolver_gebal_strided_batched_impl(rocblas_handle handle,
                                                    const rocsolver_balance job,
                                                    const I n,
                                                    U A,
                                                    const I lda,
                                                    const rocblas_stride strideA,
                                                    I* ilo,
                                                    I* ihi,
                                                    S* scale,
                                                    const rocblas_stride strideS,
                                                    const I batch_count)
try
{
    ROCSOLVER_ENTER_TOP("gebal_strided_batched", "--job", job, "-n", n, "--lda", lda, "--strideA",
                        strideA, "--strideS", strideS, "--batch_count", batch_count);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st
        = rocsolver_gebal_argCheck(handle, job, n, lda, A, ilo, ihi, scale, batch_count);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_stride shiftA = 0;

    // memory workspace sizes:
    // size of the workspace (the counts of the permutation step, and the state and
    // partial results of the multi-block algorithm for large matrices)
    size_t size_work;
    rocsolver_gebal_getMemorySize<T>(job, n, batch_count, &size_work);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle, size_work);

    // memory workspace allocation
    void* work;
    rocblas_device_malloc mem(handle, size_work);
    if(!mem)
        return rocblas_status_memory_error;

    work = mem[0];

    // execution
    return rocsolver_gebal_template<false, true, T>(handle, job, n, A, shiftA, lda, strideA, ilo,
                                                    ihi, scale, strideS, batch_count, (I*)work);
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

rocblas_status rocsolver_sgebal_strided_batched(rocblas_handle handle,
                                                const rocsolver_balance job,
                                                const rocblas_int n,
                                                float* A,
                                                const rocblas_int lda,
                                                const rocblas_stride strideA,
                                                rocblas_int* ilo,
                                                rocblas_int* ihi,
                                                float* scale,
                                                const rocblas_stride strideS,
                                                const rocblas_int batch_count)
{
#if defined(ROCSOLVER_ENABLE_BALANCE)
    return rocsolver::rocsolver_gebal_strided_batched_impl<float>(
        handle, job, n, A, lda, strideA, ilo, ihi, scale, strideS, batch_count);
#else
    return rocblas_status_not_implemented;
#endif
}

rocblas_status rocsolver_dgebal_strided_batched(rocblas_handle handle,
                                                const rocsolver_balance job,
                                                const rocblas_int n,
                                                double* A,
                                                const rocblas_int lda,
                                                const rocblas_stride strideA,
                                                rocblas_int* ilo,
                                                rocblas_int* ihi,
                                                double* scale,
                                                const rocblas_stride strideS,
                                                const rocblas_int batch_count)
{
#if defined(ROCSOLVER_ENABLE_BALANCE)
    return rocsolver::rocsolver_gebal_strided_batched_impl<double>(
        handle, job, n, A, lda, strideA, ilo, ihi, scale, strideS, batch_count);
#else
    return rocblas_status_not_implemented;
#endif
}

rocblas_status rocsolver_cgebal_strided_batched(rocblas_handle handle,
                                                const rocsolver_balance job,
                                                const rocblas_int n,
                                                rocblas_float_complex* A,
                                                const rocblas_int lda,
                                                const rocblas_stride strideA,
                                                rocblas_int* ilo,
                                                rocblas_int* ihi,
                                                float* scale,
                                                const rocblas_stride strideS,
                                                const rocblas_int batch_count)
{
#if defined(ROCSOLVER_ENABLE_BALANCE)
    return rocsolver::rocsolver_gebal_strided_batched_impl<rocblas_float_complex>(
        handle, job, n, A, lda, strideA, ilo, ihi, scale, strideS, batch_count);
#else
    return rocblas_status_not_implemented;
#endif
}

rocblas_status rocsolver_zgebal_strided_batched(rocblas_handle handle,
                                                const rocsolver_balance job,
                                                const rocblas_int n,
                                                rocblas_double_complex* A,
                                                const rocblas_int lda,
                                                const rocblas_stride strideA,
                                                rocblas_int* ilo,
                                                rocblas_int* ihi,
                                                double* scale,
                                                const rocblas_stride strideS,
                                                const rocblas_int batch_count)
{
#if defined(ROCSOLVER_ENABLE_BALANCE)
    return rocsolver::rocsolver_gebal_strided_batched_impl<rocblas_double_complex>(
        handle, job, n, A, lda, strideA, ilo, ihi, scale, strideS, batch_count);
#else
    return rocblas_status_not_implemented;
#endif
}

} // extern C
