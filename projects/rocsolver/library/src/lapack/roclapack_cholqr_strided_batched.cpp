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
#include "roclapack_cholqr.hpp"

ROCSOLVER_BEGIN_NAMESPACE

template <typename T, typename I, typename S = decltype(std::real(T{}))>
rocblas_status rocsolver_cholqr_strided_batched_impl(rocblas_handle handle,
                                                     const rocsolver_cholqr_shift cholshift,
                                                     const rocblas_int cholnum,
                                                     const I m,
                                                     const I n,
                                                     T* A,
                                                     const I lda,
                                                     const rocblas_stride strideA,
                                                     T* W,
                                                     const I ldw,
                                                     const rocblas_stride strideW,
                                                     S* sigma,
                                                     I* nr,
                                                     const I batch_count)
try
{
    ROCSOLVER_ENTER_TOP("cholqr_strided_batched", "--cholshift", cholshift, "--cholnum", cholnum,
                        "-m", m, "-n", n, "--lda", lda, "--strideA", strideA, "--ldw", ldw,
                        "--strideW", strideW, "--batch_count", batch_count);

    if(!handle)
        return rocblas_status_invalid_handle;

    // working with unshifted arrays
    rocblas_stride shiftA = 0;
    rocblas_stride shiftW = 0;

    // argument checking
    rocblas_status st = rocsolver_cholqr_argCheck<T>(
        handle, cholshift, cholnum, m, n, A, lda, strideA, W, ldw, strideW, sigma, nr, batch_count);
    if(st != rocblas_status_continue)
        return st;

    // memory workspace sizes:
    size_t size_W1;
    size_t size_Acpy;
    // size of workspace (for calling TRSM)
    bool optim_mem;
    size_t size_work1, size_work2, size_work3, size_work4;
    // size of workspace (for calling POTRF)
    size_t size_scalars, size_pivots, size_iinfo;
    // size of arrays of pointers (for batched cases)
    size_t size_workArr;
    rocsolver_cholqr_getMemorySize<false, true, T>(
        cholshift, cholnum, m, n, lda, ldw, batch_count, &size_scalars, &size_work1, &size_work2,
        &size_work3, &size_work4, &size_pivots, &size_iinfo, &size_W1, &size_Acpy, &size_workArr,
        &optim_mem);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle, size_scalars, size_work1, size_work2,
                                                      size_work3, size_work4, size_pivots,
                                                      size_iinfo, size_W1, size_Acpy, size_workArr);

    // memory workspace allocation
    void *scalars, *work1, *work2, *work3, *work4, *pivots, *iinfo, *W1, *Acpy, *workArr;
    rocblas_device_malloc mem(handle, size_scalars, size_work1, size_work2, size_work3, size_work4,
                              size_pivots, size_iinfo, size_W1, size_Acpy, size_workArr);

    if(!mem)
        return rocblas_status_memory_error;

    scalars = mem[0];
    work1 = mem[1];
    work2 = mem[2];
    work3 = mem[3];
    work4 = mem[4];
    pivots = mem[5];
    iinfo = mem[6];
    W1 = mem[7];
    Acpy = mem[8];
    workArr = mem[9];
    if(size_scalars > 0)
        init_scalars(handle, (T*)scalars);

    // execution
    return rocsolver_cholqr_template<false, true, T>(
        handle, cholshift, cholnum, m, n, A, shiftA, lda, strideA, W, shiftW, ldw, strideW, sigma,
        nr, batch_count, (T*)scalars, work1, work2, work3, work4, (T*)pivots, (I*)iinfo, (T*)W1,
        (T*)Acpy, (T**)workArr, optim_mem);
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

rocblas_status rocsolver_scholqr_strided_batched(rocblas_handle handle,
                                                 const rocsolver_cholqr_shift cholshift,
                                                 const rocblas_int cholnum,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 float* A,
                                                 const rocblas_int lda,
                                                 const rocblas_stride strideA,
                                                 float* W,
                                                 const rocblas_int ldw,
                                                 const rocblas_stride strideW,
                                                 float* sigma,
                                                 rocblas_int* nr,
                                                 const rocblas_int batch_count)
{
#if defined(ROCSOLVER_ENABLE_CHOLQR)
    return rocsolver::rocsolver_cholqr_strided_batched_impl<float>(
        handle, cholshift, cholnum, m, n, A, lda, strideA, W, ldw, strideW, sigma, nr, batch_count);
#else
    return rocblas_status_not_implemented;
#endif
}

rocblas_status rocsolver_dcholqr_strided_batched(rocblas_handle handle,
                                                 const rocsolver_cholqr_shift cholshift,
                                                 const rocblas_int cholnum,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 double* A,
                                                 const rocblas_int lda,
                                                 const rocblas_stride strideA,
                                                 double* W,
                                                 const rocblas_int ldw,
                                                 const rocblas_stride strideW,
                                                 double* sigma,
                                                 rocblas_int* nr,
                                                 const rocblas_int batch_count)
{
#if defined(ROCSOLVER_ENABLE_CHOLQR)
    return rocsolver::rocsolver_cholqr_strided_batched_impl<double>(
        handle, cholshift, cholnum, m, n, A, lda, strideA, W, ldw, strideW, sigma, nr, batch_count);
#else
    return rocblas_status_not_implemented;
#endif
}

rocblas_status rocsolver_ccholqr_strided_batched(rocblas_handle handle,
                                                 const rocsolver_cholqr_shift cholshift,
                                                 const rocblas_int cholnum,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 rocblas_float_complex* A,
                                                 const rocblas_int lda,
                                                 const rocblas_stride strideA,
                                                 rocblas_float_complex* W,
                                                 const rocblas_int ldw,
                                                 const rocblas_stride strideW,
                                                 float* sigma,
                                                 rocblas_int* nr,
                                                 const rocblas_int batch_count)
{
#if defined(ROCSOLVER_ENABLE_CHOLQR)
    return rocsolver::rocsolver_cholqr_strided_batched_impl<rocblas_float_complex>(
        handle, cholshift, cholnum, m, n, A, lda, strideA, W, ldw, strideW, sigma, nr, batch_count);
#else
    return rocblas_status_not_implemented;
#endif
}

rocblas_status rocsolver_zcholqr_strided_batched(rocblas_handle handle,
                                                 const rocsolver_cholqr_shift cholshift,
                                                 const rocblas_int cholnum,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 rocblas_double_complex* A,
                                                 const rocblas_int lda,
                                                 const rocblas_stride strideA,
                                                 rocblas_double_complex* W,
                                                 const rocblas_int ldw,
                                                 const rocblas_stride strideW,
                                                 double* sigma,
                                                 rocblas_int* nr,
                                                 const rocblas_int batch_count)
{
#if defined(ROCSOLVER_ENABLE_CHOLQR)
    return rocsolver::rocsolver_cholqr_strided_batched_impl<rocblas_double_complex>(
        handle, cholshift, cholnum, m, n, A, lda, strideA, W, ldw, strideW, sigma, nr, batch_count);
#else
    return rocblas_status_not_implemented;
#endif
}

rocblas_status rocsolver_scholqr_strided_batched_64(rocblas_handle handle,
                                                    const rocsolver_cholqr_shift cholshift,
                                                    const rocblas_int cholnum,
                                                    const int64_t m,
                                                    const int64_t n,
                                                    float* A,
                                                    const int64_t lda,
                                                    const rocblas_stride strideA,
                                                    float* W,
                                                    const int64_t ldw,
                                                    const rocblas_stride strideW,
                                                    float* sigma,
                                                    int64_t* nr,
                                                    const int64_t batch_count)
{
#if defined(HAVE_ROCBLAS_64) && defined(ROCSOLVER_ENABLE_CHOLQR)
    return rocsolver::rocsolver_cholqr_strided_batched_impl<float>(
        handle, cholshift, cholnum, m, n, A, lda, strideA, W, ldw, strideW, sigma, nr, batch_count);
#else
    return rocblas_status_not_implemented;
#endif
}

rocblas_status rocsolver_dcholqr_strided_batched_64(rocblas_handle handle,
                                                    const rocsolver_cholqr_shift cholshift,
                                                    const rocblas_int cholnum,
                                                    const int64_t m,
                                                    const int64_t n,
                                                    double* A,
                                                    const int64_t lda,
                                                    const rocblas_stride strideA,
                                                    double* W,
                                                    const int64_t ldw,
                                                    const rocblas_stride strideW,
                                                    double* sigma,
                                                    int64_t* nr,
                                                    const int64_t batch_count)
{
#if defined(HAVE_ROCBLAS_64) && defined(ROCSOLVER_ENABLE_CHOLQR)
    return rocsolver::rocsolver_cholqr_strided_batched_impl<double>(
        handle, cholshift, cholnum, m, n, A, lda, strideA, W, ldw, strideW, sigma, nr, batch_count);
#else
    return rocblas_status_not_implemented;
#endif
}

rocblas_status rocsolver_ccholqr_strided_batched_64(rocblas_handle handle,
                                                    const rocsolver_cholqr_shift cholshift,
                                                    const rocblas_int cholnum,
                                                    const int64_t m,
                                                    const int64_t n,
                                                    rocblas_float_complex* A,
                                                    const int64_t lda,
                                                    const rocblas_stride strideA,
                                                    rocblas_float_complex* W,
                                                    const int64_t ldw,
                                                    const rocblas_stride strideW,
                                                    float* sigma,
                                                    int64_t* nr,
                                                    const int64_t batch_count)
{
#if defined(HAVE_ROCBLAS_64) && defined(ROCSOLVER_ENABLE_CHOLQR)
    return rocsolver::rocsolver_cholqr_strided_batched_impl<rocblas_float_complex>(
        handle, cholshift, cholnum, m, n, A, lda, strideA, W, ldw, strideW, sigma, nr, batch_count);
#else
    return rocblas_status_not_implemented;
#endif
}

rocblas_status rocsolver_zcholqr_strided_batched_64(rocblas_handle handle,
                                                    const rocsolver_cholqr_shift cholshift,
                                                    const rocblas_int cholnum,
                                                    const int64_t m,
                                                    const int64_t n,
                                                    rocblas_double_complex* A,
                                                    const int64_t lda,
                                                    const rocblas_stride strideA,
                                                    rocblas_double_complex* W,
                                                    const int64_t ldw,
                                                    const rocblas_stride strideW,
                                                    double* sigma,
                                                    int64_t* nr,
                                                    const int64_t batch_count)
{
#if defined(HAVE_ROCBLAS_64) && defined(ROCSOLVER_ENABLE_CHOLQR)
    return rocsolver::rocsolver_cholqr_strided_batched_impl<rocblas_double_complex>(
        handle, cholshift, cholnum, m, n, A, lda, strideA, W, ldw, strideW, sigma, nr, batch_count);
#else
    return rocblas_status_not_implemented;
#endif
}

} // extern C
