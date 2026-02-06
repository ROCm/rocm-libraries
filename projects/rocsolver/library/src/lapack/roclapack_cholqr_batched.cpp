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

#include "roclapack_cholqr.hpp"

ROCSOLVER_BEGIN_NAMESPACE

template <typename T, typename I, typename S = decltype(std::real(T{}))>
rocblas_status rocsolver_cholqr_batched_impl(rocblas_handle handle,
                                             const I m,
                                             const I n,
                                             T* const A[],
                                             const I lda,
                                             T* R,
                                             const I ldr,
                                             const rocblas_stride strideR,
                                             S* sigma,
                                             const rocsolver_cholqr_algo algo,
                                             I* info,
                                             const I batch_count)
{
    ROCSOLVER_ENTER_TOP("cholqr_batched", "-m", m, "-n", n, "--lda", lda, "--ldr", ldr, "--strideR",
                        strideR, "--algo", algo, "--batch_count", batch_count);

    if(!handle)
        return rocblas_status_invalid_handle;

    // working with unshifted arrays
    rocblas_stride shiftA = 0;
    rocblas_stride shiftR = 0;

    // batched execution
    rocblas_stride strideA = 0;

    // argument checking
    rocblas_status st = rocsolver_cholqr_argCheck<T>(handle, m, n, A, lda, strideA, R, ldr, strideR,
                                                     sigma, algo, info, batch_count);
    if(st != rocblas_status_continue)
        return st;

    // memory workspace sizes:
    size_t size_work = 0;
    bool optim_mem;
    size_t size_work1, size_work2, size_work3, size_work4;
    size_t size_workArr;
    rocsolver_cholqr_getMemorySize<true, false, T>(m, n, lda, ldr, batch_count, algo, &size_work1,
                                                   &size_work2, &size_work3, &size_work4,
                                                   &size_workArr, &optim_mem, &size_work);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle, size_work1, size_work2, size_work3,
                                                      size_work4, size_workArr, size_work);

    // memory workspace allocation
    void *work1, *work2, *work3, *work4, *workArr;
    rocblas_device_malloc mem(handle, size_work1, size_work2, size_work3, size_work4, size_workArr,
                              size_work);

    if(!mem)
        return rocblas_status_memory_error;

    work1 = mem[0];
    work2 = mem[1];
    work3 = mem[2];
    work4 = mem[3];
    workArr = mem[4];
    void* const work = (void*)mem[5];

    // execution
    return rocsolver_cholqr_template<true, false, T>(
        handle, m, n, A, shiftA, lda, strideA, R, shiftR, ldr, strideR, sigma, algo, info,
        batch_count, work1, work2, work3, work4, (T**)workArr, optim_mem, work, size_work);
}

ROCSOLVER_END_NAMESPACE

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocsolver_scholqr_batched(rocblas_handle handle,
                                         const rocblas_int m,
                                         const rocblas_int n,
                                         float* const A[],
                                         const rocblas_int lda,
                                         float* R,
                                         const rocblas_int ldr,
                                         const rocblas_stride strideR,
                                         float* sigma,
                                         const rocsolver_cholqr_algo algo,
                                         rocblas_int* info,
                                         const rocblas_int batch_count)
{
    return rocsolver::rocsolver_cholqr_batched_impl<float>(handle, m, n, A, lda, R, ldr, strideR,
                                                           sigma, algo, info, batch_count);
}

rocblas_status rocsolver_dcholqr_batched(rocblas_handle handle,
                                         const rocblas_int m,
                                         const rocblas_int n,
                                         double* const A[],
                                         const rocblas_int lda,
                                         double* R,
                                         const rocblas_int ldr,
                                         const rocblas_stride strideR,
                                         double* sigma,
                                         const rocsolver_cholqr_algo algo,
                                         rocblas_int* info,
                                         const rocblas_int batch_count)
{
    return rocsolver::rocsolver_cholqr_batched_impl<double>(handle, m, n, A, lda, R, ldr, strideR,
                                                            sigma, algo, info, batch_count);
}

rocblas_status rocsolver_ccholqr_batched(rocblas_handle handle,
                                         const rocblas_int m,
                                         const rocblas_int n,
                                         rocblas_float_complex* const A[],
                                         const rocblas_int lda,
                                         rocblas_float_complex* R,
                                         const rocblas_int ldr,
                                         const rocblas_stride strideR,
                                         float* sigma,
                                         const rocsolver_cholqr_algo algo,
                                         rocblas_int* info,
                                         const rocblas_int batch_count)
{
    return rocsolver::rocsolver_cholqr_batched_impl<rocblas_float_complex>(
        handle, m, n, A, lda, R, ldr, strideR, sigma, algo, info, batch_count);
}

rocblas_status rocsolver_zcholqr_batched(rocblas_handle handle,
                                         const rocblas_int m,
                                         const rocblas_int n,
                                         rocblas_double_complex* const A[],
                                         const rocblas_int lda,
                                         rocblas_double_complex* R,
                                         const rocblas_int ldr,
                                         const rocblas_stride strideR,
                                         double* sigma,
                                         const rocsolver_cholqr_algo algo,
                                         rocblas_int* info,
                                         const rocblas_int batch_count)
{
    return rocsolver::rocsolver_cholqr_batched_impl<rocblas_double_complex>(
        handle, m, n, A, lda, R, ldr, strideR, sigma, algo, info, batch_count);
}

rocblas_status rocsolver_scholqr_batched_64(rocblas_handle handle,
                                            const int64_t m,
                                            const int64_t n,
                                            float* const A[],
                                            const int64_t lda,
                                            float* R,
                                            const int64_t ldr,
                                            const rocblas_stride strideR,
                                            float* sigma,
                                            const rocsolver_cholqr_algo algo,
                                            int64_t* info,
                                            const int64_t batch_count)
{
#ifdef HAVE_ROCBLAS_64
    return rocsolver::rocsolver_cholqr_batched_impl<float>(handle, m, n, A, lda, R, ldr, strideR,
                                                           sigma, algo, info, batch_count);
#else
    return rocblas_status_not_implemented;
#endif
}

rocblas_status rocsolver_dcholqr_batched_64(rocblas_handle handle,
                                            const int64_t m,
                                            const int64_t n,
                                            double* const A[],
                                            const int64_t lda,
                                            double* R,
                                            const int64_t ldr,
                                            const rocblas_stride strideR,
                                            double* sigma,
                                            const rocsolver_cholqr_algo algo,
                                            int64_t* info,
                                            const int64_t batch_count)
{
#ifdef HAVE_ROCBLAS_64
    return rocsolver::rocsolver_cholqr_batched_impl<double>(handle, m, n, A, lda, R, ldr, strideR,
                                                            sigma, algo, info, batch_count);
#else
    return rocblas_status_not_implemented;
#endif
}

rocblas_status rocsolver_ccholqr_batched_64(rocblas_handle handle,
                                            const int64_t m,
                                            const int64_t n,
                                            rocblas_float_complex* const A[],
                                            const int64_t lda,
                                            rocblas_float_complex* R,
                                            const int64_t ldr,
                                            const rocblas_stride strideR,
                                            float* sigma,
                                            const rocsolver_cholqr_algo algo,
                                            int64_t* info,
                                            const int64_t batch_count)
{
#ifdef HAVE_ROCBLAS_64
    return rocsolver::rocsolver_cholqr_batched_impl<rocblas_float_complex>(
        handle, m, n, A, lda, R, ldr, strideR, sigma, algo, info, batch_count);
#else
    return rocblas_status_not_implemented;
#endif
}

rocblas_status rocsolver_zcholqr_batched_64(rocblas_handle handle,
                                            const int64_t m,
                                            const int64_t n,
                                            rocblas_double_complex* const A[],
                                            const int64_t lda,
                                            rocblas_double_complex* R,
                                            const int64_t ldr,
                                            const rocblas_stride strideR,
                                            double* sigma,
                                            const rocsolver_cholqr_algo algo,
                                            int64_t* info,
                                            const int64_t batch_count)
{
#ifdef HAVE_ROCBLAS_64
    return rocsolver::rocsolver_cholqr_batched_impl<rocblas_double_complex>(
        handle, m, n, A, lda, R, ldr, strideR, sigma, algo, info, batch_count);
#else
    return rocblas_status_not_implemented;
#endif
}

} // extern C
