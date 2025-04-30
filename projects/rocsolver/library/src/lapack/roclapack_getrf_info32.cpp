/* **************************************************************************
 * Copyright (C) 2019-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include "roclapack_getrf.hpp"

ROCSOLVER_BEGIN_NAMESPACE

/*
 * ===========================================================================
 *    getrf_info32 is not intended for inclusion in the public API. It
 *    exists to provide a 64-bit getrf method with a signature identical to
 *    the cuBLAS implementation, for use exclusively in hipSOLVER.
 * ===========================================================================
 */

template <typename T, typename U>
rocblas_status rocsolver_getrf_info32_impl(rocblas_handle handle,
                                           const int64_t m,
                                           const int64_t n,
                                           U A,
                                           const int64_t lda,
                                           int64_t* ipiv,
                                           rocblas_int* info,
                                           const bool pivot)
{
    const char* name = (pivot ? "getrf" : "getrf_npvt");
    ROCSOLVER_ENTER_TOP(name, "-m", m, "-n", n, "--lda", lda);

    using S = decltype(std::real(T{}));

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st = rocsolver_getf2_getrf_argCheck(handle, m, n, lda, A, ipiv, info, pivot);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_stride shiftA = 0;
    rocblas_stride shiftP = 0;

    // normal (non-batched non-strided) execution
    int64_t inca = 1;
    rocblas_stride strideA = 0;
    rocblas_stride strideP = 0;
    int64_t batch_count = 1;

    // memory workspace sizes:
    // size for constants in rocblas calls
    size_t size_scalars;
    // size of reusable workspace (and for calling TRSM)
    bool optim_mem;
    size_t size_work1, size_work2, size_work3, size_work4;
    // extra requirements for calling GETF2
    size_t size_pivotval, size_pivotidx;
    // size to store info about singularity of each subblock
    size_t size_iinfo, size_iipiv;

    rocsolver_getrf_getMemorySize<false, false, T>(
        m, n, pivot, batch_count, &size_scalars, &size_work1, &size_work2, &size_work3, &size_work4,
        &size_pivotval, &size_pivotidx, &size_iipiv, &size_iinfo, &optim_mem, lda);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle, size_scalars, size_work1, size_work2,
                                                      size_work3, size_work4, size_pivotval,
                                                      size_pivotidx, size_iipiv, size_iinfo);

    // memory workspace allocation
    void *scalars, *work1, *work2, *work3, *work4, *pivotval, *pivotidx, *iinfo, *iipiv;
    rocblas_device_malloc mem(handle, size_scalars, size_work1, size_work2, size_work3, size_work4,
                              size_pivotval, size_pivotidx, size_iipiv, size_iinfo);

    if(!mem)
        return rocblas_status_memory_error;

    scalars = mem[0];
    work1 = mem[1];
    work2 = mem[2];
    work3 = mem[3];
    work4 = mem[4];
    pivotval = mem[5];
    pivotidx = mem[6];
    iipiv = mem[7];
    iinfo = mem[8];
    if(size_scalars > 0)
        init_scalars(handle, (T*)scalars);

    // execution
    return rocsolver_getrf_template<false, false, T>(
        handle, m, n, A, shiftA, inca, lda, strideA, ipiv, shiftP, strideP, info, batch_count,
        (T*)scalars, work1, work2, work3, work4, (T*)pivotval, (int64_t*)pivotidx, (int64_t*)iipiv,
        (rocblas_int*)iinfo, optim_mem, pivot);
}

ROCSOLVER_END_NAMESPACE

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

ROCSOLVER_EXPORT rocblas_status rocsolver_sgetrf_info32(rocblas_handle handle,
                                                        const int64_t m,
                                                        const int64_t n,
                                                        float* A,
                                                        const int64_t lda,
                                                        int64_t* ipiv,
                                                        rocblas_int* info)
{
#ifdef HAVE_ROCBLAS_64
    return rocsolver::rocsolver_getrf_info32_impl<float>(handle, m, n, A, lda, ipiv, info, true);
#else
    return rocblas_status_not_implemented;
#endif
}

ROCSOLVER_EXPORT rocblas_status rocsolver_dgetrf_info32(rocblas_handle handle,
                                                        const int64_t m,
                                                        const int64_t n,
                                                        double* A,
                                                        const int64_t lda,
                                                        int64_t* ipiv,
                                                        rocblas_int* info)
{
#ifdef HAVE_ROCBLAS_64
    return rocsolver::rocsolver_getrf_info32_impl<double>(handle, m, n, A, lda, ipiv, info, true);
#else
    return rocblas_status_not_implemented;
#endif
}

ROCSOLVER_EXPORT rocblas_status rocsolver_cgetrf_info32(rocblas_handle handle,
                                                        const int64_t m,
                                                        const int64_t n,
                                                        rocblas_float_complex* A,
                                                        const int64_t lda,
                                                        int64_t* ipiv,
                                                        rocblas_int* info)
{
#ifdef HAVE_ROCBLAS_64
    return rocsolver::rocsolver_getrf_info32_impl<rocblas_float_complex>(handle, m, n, A, lda, ipiv,
                                                                         info, true);
#else
    return rocblas_status_not_implemented;
#endif
}

ROCSOLVER_EXPORT rocblas_status rocsolver_zgetrf_info32(rocblas_handle handle,
                                                        const int64_t m,
                                                        const int64_t n,
                                                        rocblas_double_complex* A,
                                                        const int64_t lda,
                                                        int64_t* ipiv,
                                                        rocblas_int* info)
{
#ifdef HAVE_ROCBLAS_64
    return rocsolver::rocsolver_getrf_info32_impl<rocblas_double_complex>(handle, m, n, A, lda,
                                                                          ipiv, info, true);
#else
    return rocblas_status_not_implemented;
#endif
}

ROCSOLVER_EXPORT rocblas_status rocsolver_sgetrf_npvt_info32(rocblas_handle handle,
                                                             const int64_t m,
                                                             const int64_t n,
                                                             float* A,
                                                             const int64_t lda,
                                                             rocblas_int* info)
{
#ifdef HAVE_ROCBLAS_64
    int64_t* ipiv = nullptr;
    return rocsolver::rocsolver_getrf_info32_impl<float>(handle, m, n, A, lda, ipiv, info, false);
#else
    return rocblas_status_not_implemented;
#endif
}

ROCSOLVER_EXPORT rocblas_status rocsolver_dgetrf_npvt_info32(rocblas_handle handle,
                                                             const int64_t m,
                                                             const int64_t n,
                                                             double* A,
                                                             const int64_t lda,
                                                             rocblas_int* info)
{
#ifdef HAVE_ROCBLAS_64
    int64_t* ipiv = nullptr;
    return rocsolver::rocsolver_getrf_info32_impl<double>(handle, m, n, A, lda, ipiv, info, false);
#else
    return rocblas_status_not_implemented;
#endif
}

ROCSOLVER_EXPORT rocblas_status rocsolver_cgetrf_npvt_info32(rocblas_handle handle,
                                                             const int64_t m,
                                                             const int64_t n,
                                                             rocblas_float_complex* A,
                                                             const int64_t lda,
                                                             rocblas_int* info)
{
#ifdef HAVE_ROCBLAS_64
    int64_t* ipiv = nullptr;
    return rocsolver::rocsolver_getrf_info32_impl<rocblas_float_complex>(handle, m, n, A, lda, ipiv,
                                                                         info, false);
#else
    return rocblas_status_not_implemented;
#endif
}

ROCSOLVER_EXPORT rocblas_status rocsolver_zgetrf_npvt_info32(rocblas_handle handle,
                                                             const int64_t m,
                                                             const int64_t n,
                                                             rocblas_double_complex* A,
                                                             const int64_t lda,
                                                             rocblas_int* info)
{
#ifdef HAVE_ROCBLAS_64
    int64_t* ipiv = nullptr;
    return rocsolver::rocsolver_getrf_info32_impl<rocblas_double_complex>(handle, m, n, A, lda,
                                                                          ipiv, info, false);
#else
    return rocblas_status_not_implemented;
#endif
}

} // extern C
