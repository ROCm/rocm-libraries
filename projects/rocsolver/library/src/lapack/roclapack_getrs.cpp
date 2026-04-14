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

#include "roclapack_getrs.hpp"

ROCSOLVER_BEGIN_NAMESPACE

template <typename T, typename I>
rocblas_status rocsolver_getrs_impl(rocblas_handle handle,
                                    const rocblas_operation trans,
                                    const I n,
                                    const I nrhs,
                                    T* A,
                                    const I lda,
                                    const I* ipiv,
                                    T* B,
                                    const I ldb)
{
    ROCSOLVER_ENTER_TOP("getrs", "--trans", trans, "-n", n, "--nrhs", nrhs, "--lda", lda, "--ldb",
                        ldb);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st = rocsolver_getrs_argCheck(handle, trans, n, nrhs, lda, ldb, A, B, ipiv);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_stride shiftA = 0;
    rocblas_stride shiftB = 0;

    // normal (non-batched non-strided) execution
    I inca = 1;
    I incb = 1;
    rocblas_stride strideA = 0;
    rocblas_stride strideB = 0;
    rocblas_stride strideP = 0;
    I batch_count = 1;

    // memory workspace sizes:
    // size of workspace (for calling TRSM)
    bool optim_mem;
    size_t size_work1, size_work2, size_work3, size_work4;
    rocsolver_getrs_getMemorySize<false, false, T>(trans, n, nrhs, batch_count, &size_work1,
                                                   &size_work2, &size_work3, &size_work4,
                                                   &optim_mem, lda, ldb);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle, size_work1, size_work2, size_work3,
                                                      size_work4);

    // memory workspace allocation
    void *work1, *work2, *work3, *work4;
    rocblas_device_malloc mem(handle, size_work1, size_work2, size_work3, size_work4);

    if(!mem)
        return rocblas_status_memory_error;

    work1 = mem[0];
    work2 = mem[1];
    work3 = mem[2];
    work4 = mem[3];

    // execution
    return rocsolver_getrs_template<false, false, T>(
        handle, trans, n, nrhs, A, shiftA, inca, lda, strideA, ipiv, strideP, B, shiftB, incb, ldb,
        strideB, batch_count, work1, work2, work3, work4, optim_mem, true);
}

ROCSOLVER_END_NAMESPACE

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocsolver_sgetrs(rocblas_handle handle,
                                const rocblas_operation trans,
                                const rocblas_int n,
                                const rocblas_int nrhs,
                                float* A,
                                const rocblas_int lda,
                                const rocblas_int* ipiv,
                                float* B,
                                const rocblas_int ldb)
{
    return rocsolver::rocsolver_getrs_impl<float>(handle, trans, n, nrhs, A, lda, ipiv, B, ldb);
}

rocblas_status rocsolver_dgetrs(rocblas_handle handle,
                                const rocblas_operation trans,
                                const rocblas_int n,
                                const rocblas_int nrhs,
                                double* A,
                                const rocblas_int lda,
                                const rocblas_int* ipiv,
                                double* B,
                                const rocblas_int ldb)
{
    return rocsolver::rocsolver_getrs_impl<double>(handle, trans, n, nrhs, A, lda, ipiv, B, ldb);
}

rocblas_status rocsolver_cgetrs(rocblas_handle handle,
                                const rocblas_operation trans,
                                const rocblas_int n,
                                const rocblas_int nrhs,
                                rocblas_float_complex* A,
                                const rocblas_int lda,
                                const rocblas_int* ipiv,
                                rocblas_float_complex* B,
                                const rocblas_int ldb)
{
    return rocsolver::rocsolver_getrs_impl<rocblas_float_complex>(handle, trans, n, nrhs, A, lda,
                                                                  ipiv, B, ldb);
}

rocblas_status rocsolver_zgetrs(rocblas_handle handle,
                                const rocblas_operation trans,
                                const rocblas_int n,
                                const rocblas_int nrhs,
                                rocblas_double_complex* A,
                                const rocblas_int lda,
                                const rocblas_int* ipiv,
                                rocblas_double_complex* B,
                                const rocblas_int ldb)
{
    return rocsolver::rocsolver_getrs_impl<rocblas_double_complex>(handle, trans, n, nrhs, A, lda,
                                                                   ipiv, B, ldb);
}

rocblas_status rocsolver_sgetrs_64(rocblas_handle handle,
                                   const rocblas_operation trans,
                                   const int64_t n,
                                   const int64_t nrhs,
                                   float* A,
                                   const int64_t lda,
                                   const int64_t* ipiv,
                                   float* B,
                                   const int64_t ldb)
{
#ifdef HAVE_ROCBLAS_64
    return rocsolver::rocsolver_getrs_impl<float>(handle, trans, n, nrhs, A, lda, ipiv, B, ldb);
#else
    return rocblas_status_not_implemented;
#endif
}

rocblas_status rocsolver_dgetrs_64(rocblas_handle handle,
                                   const rocblas_operation trans,
                                   const int64_t n,
                                   const int64_t nrhs,
                                   double* A,
                                   const int64_t lda,
                                   const int64_t* ipiv,
                                   double* B,
                                   const int64_t ldb)
{
#ifdef HAVE_ROCBLAS_64
    return rocsolver::rocsolver_getrs_impl<double>(handle, trans, n, nrhs, A, lda, ipiv, B, ldb);
#else
    return rocblas_status_not_implemented;
#endif
}

rocblas_status rocsolver_cgetrs_64(rocblas_handle handle,
                                   const rocblas_operation trans,
                                   const int64_t n,
                                   const int64_t nrhs,
                                   rocblas_float_complex* A,
                                   const int64_t lda,
                                   const int64_t* ipiv,
                                   rocblas_float_complex* B,
                                   const int64_t ldb)
{
#ifdef HAVE_ROCBLAS_64
    return rocsolver::rocsolver_getrs_impl<rocblas_float_complex>(handle, trans, n, nrhs, A, lda,
                                                                  ipiv, B, ldb);
#else
    return rocblas_status_not_implemented;
#endif
}

rocblas_status rocsolver_zgetrs_64(rocblas_handle handle,
                                   const rocblas_operation trans,
                                   const int64_t n,
                                   const int64_t nrhs,
                                   rocblas_double_complex* A,
                                   const int64_t lda,
                                   const int64_t* ipiv,
                                   rocblas_double_complex* B,
                                   const int64_t ldb)
{
#ifdef HAVE_ROCBLAS_64
    return rocsolver::rocsolver_getrs_impl<rocblas_double_complex>(handle, trans, n, nrhs, A, lda,
                                                                   ipiv, B, ldb);
#else
    return rocblas_status_not_implemented;
#endif
}

} // extern C
