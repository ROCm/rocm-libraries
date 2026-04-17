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

#include "roclapack_sytrs.hpp"

ROCSOLVER_BEGIN_NAMESPACE

template <typename T, typename I, typename Istride>
rocblas_status rocsolver_sytrs_batched_impl(rocblas_handle handle,
                                            rocblas_fill const uplo,

                                            I const n,
                                            I const nrhs,

                                            T* const A[],
                                            I const lda,

                                            I* const ipiv,
                                            Istride const strideP,

                                            T* const B[],
                                            I const ldb,

                                            I const batch_count)
{
    ROCSOLVER_ENTER_TOP("sytrs_batched", "--uplo", uplo, "-n", n, "--nrhs", nrhs, "--lda", lda,
                        "--strideP", strideP, "--ldb", ldb, "--batch_count", batch_count);

    if(!handle)
    {
        return rocblas_status_invalid_handle;
    }

    // argument checking
    {
        rocblas_status st
            = rocsolver_sytrs_argCheck(handle, uplo, n, nrhs, lda, ldb, A, B, ipiv, batch_count);

	bool const is_ok = (st == rocblas_status_continue) ||
		           (st == rocblas_status_success);
        if(!is_ok)
        {
            return st;
        }
    }

    // working with unshifted arrays
    Istride shiftA = 0;
    Istride shiftB = 0;
    Istride strideA = Istride(lda) * n;
    Istride strideB = Istride(ldb) * nrhs;

    // ----------------------
    // memory workspace sizes:
    // ----------------------
    size_t size_work = 0;
    {
        auto const istat
            = rocsolver_sytrs_getMemorySize<T>(handle, n, nrhs, batch_count, lda, ldb, &size_work);
        bool const is_ok = (istat == rocblas_status_success) || (istat == rocblas_status_continue);
        if(!is_ok)
        {
            return (istat);
        }
    }

    if(rocblas_is_device_memory_size_query(handle))
    {
        return rocblas_set_optimal_device_memory_size(handle, size_work);
    }

    // ---------------------------
    // memory workspace allocation
    // ---------------------------
    rocblas_device_malloc mem(handle, size_work);

    if(!mem)
    {
        return rocblas_status_memory_error;
    }

    void* const work = static_cast<void*>(mem[0]);

    // execution
    return rocsolver_sytrs_template<T>(handle, uplo, n, nrhs,

                                       A, shiftA, lda, strideA,

                                       ipiv, strideP,

                                       B, shiftB, ldb, strideB,

                                       batch_count, work, size_work);
}

ROCSOLVER_END_NAMESPACE

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocsolver_ssytrs_batched(rocblas_handle handle,
                                        rocblas_fill const uplo,
                                        rocblas_int const n,
                                        rocblas_int const nrhs,
                                        float* const A[],
                                        rocblas_int const lda,
                                        rocblas_int* const ipiv,
                                        rocblas_stride const strideP,
                                        float* const B[],
                                        rocblas_int const ldb,
                                        rocblas_int const batch_count)
{
    return rocsolver::rocsolver_sytrs_batched_impl<float>(handle, uplo, n, nrhs, A, lda, ipiv,
                                                          strideP, B, ldb, batch_count);
}

rocblas_status rocsolver_dsytrs_batched(rocblas_handle handle,
                                        rocblas_fill const uplo,
                                        rocblas_int const n,
                                        rocblas_int const nrhs,
                                        double* const A[],
                                        rocblas_int const lda,
                                        rocblas_int* const ipiv,
                                        rocblas_stride const strideP,
                                        double* const B[],
                                        rocblas_int const ldb,
                                        rocblas_int const batch_count)
{
    return rocsolver::rocsolver_sytrs_batched_impl<double>(handle, uplo, n, nrhs, A, lda, ipiv,
                                                           strideP, B, ldb, batch_count);
}

rocblas_status rocsolver_csytrs_batched(rocblas_handle handle,
                                        rocblas_fill const uplo,
                                        rocblas_int const n,
                                        rocblas_int const nrhs,
                                        rocblas_float_complex* const A[],
                                        rocblas_int const lda,
                                        rocblas_int* const ipiv,
                                        rocblas_stride const strideP,
                                        rocblas_float_complex* const B[],
                                        rocblas_int const ldb,
                                        rocblas_int const batch_count)
{
    return rocsolver::rocsolver_sytrs_batched_impl<rocblas_float_complex>(
        handle, uplo, n, nrhs, A, lda, ipiv, strideP, B, ldb, batch_count);
}

rocblas_status rocsolver_zsytrs_batched(rocblas_handle handle,
                                        rocblas_fill const uplo,
                                        rocblas_int const n,
                                        rocblas_int const nrhs,
                                        rocblas_double_complex* const A[],
                                        rocblas_int const lda,
                                        rocblas_int* const ipiv,
                                        rocblas_stride const strideP,
                                        rocblas_double_complex* const B[],
                                        rocblas_int const ldb,
                                        rocblas_int const batch_count)
{
    return rocsolver::rocsolver_sytrs_batched_impl<rocblas_double_complex>(
        handle, uplo, n, nrhs, A, lda, ipiv, strideP, B, ldb, batch_count);
}

rocblas_status rocsolver_ssytrs_batched_64(rocblas_handle handle,
                                           rocblas_fill const uplo,
                                           int64_t const n,
                                           int64_t const nrhs,
                                           float* const A[],
                                           int64_t const lda,
                                           int64_t* const ipiv,
                                           rocblas_stride const strideP,
                                           float* const B[],
                                           int64_t const ldb,
                                           int64_t const batch_count)
{
#ifdef HAVE_ROCBLAS_64
    return rocsolver::rocsolver_sytrs_batched_impl<float, int64_t>(
        handle, uplo, n, nrhs, A, lda, ipiv, strideP, B, ldb, batch_count);
#else
    return rocblas_status_not_implemented;
#endif
}

rocblas_status rocsolver_dsytrs_batched_64(rocblas_handle handle,
                                           rocblas_fill const uplo,
                                           int64_t const n,
                                           int64_t const nrhs,
                                           double* const A[],
                                           int64_t const lda,
                                           int64_t* const ipiv,
                                           rocblas_stride const strideP,
                                           double* const B[],
                                           int64_t const ldb,
                                           int64_t const batch_count)
{
#ifdef HAVE_ROCBLAS_64
    return rocsolver::rocsolver_sytrs_batched_impl<double, int64_t>(
        handle, uplo, n, nrhs, A, lda, ipiv, strideP, B, ldb, batch_count);
#else
    return rocblas_status_not_implemented;
#endif
}

rocblas_status rocsolver_csytrs_batched_64(rocblas_handle handle,
                                           rocblas_fill const uplo,
                                           int64_t const n,
                                           int64_t const nrhs,
                                           rocblas_float_complex* const A[],
                                           int64_t const lda,
                                           int64_t* const ipiv,
                                           rocblas_stride const strideP,
                                           rocblas_float_complex* const B[],
                                           int64_t const ldb,
                                           int64_t const batch_count)
{
#ifdef HAVE_ROCBLAS_64
    return rocsolver::rocsolver_sytrs_batched_impl<rocblas_float_complex, int64_t>(
        handle, uplo, n, nrhs, A, lda, ipiv, strideP, B, ldb, batch_count);
#else
    return rocblas_status_not_implemented;
#endif
}

rocblas_status rocsolver_zsytrs_batched_64(rocblas_handle handle,
                                           rocblas_fill const uplo,
                                           int64_t const n,
                                           int64_t const nrhs,
                                           rocblas_double_complex* const A[],
                                           int64_t const lda,
                                           int64_t* const ipiv,
                                           rocblas_stride const strideP,
                                           rocblas_double_complex* const B[],
                                           int64_t const ldb,
                                           int64_t const batch_count)
{
#ifdef HAVE_ROCBLAS_64
    return rocsolver::rocsolver_sytrs_batched_impl<rocblas_double_complex, int64_t>(
        handle, uplo, n, nrhs, A, lda, ipiv, strideP, B, ldb, batch_count);
#else
    return rocblas_status_not_implemented;
#endif
}

} // extern C
