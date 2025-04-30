/* **************************************************************************
 * Copyright (C) 2020-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include "roclapack_gels_outofplace.hpp"

ROCSOLVER_BEGIN_NAMESPACE

/*
 * ===========================================================================
 *    gels_outofplace is not intended for inclusion in the public API. It
 *    exists to provide a gels method with a signature identical to
 *    the cuSOLVER implementation, for use exclusively in hipSOLVER.
 * ===========================================================================
 */

template <typename T, typename U, bool COMPLEX = rocblas_is_complex<T>>
rocblas_status rocsolver_gels_outofplace_impl(rocblas_handle handle,
                                              rocblas_operation trans,
                                              const rocblas_int m,
                                              const rocblas_int n,
                                              const rocblas_int nrhs,
                                              U A,
                                              const rocblas_int lda,
                                              U B,
                                              const rocblas_int ldb,
                                              U X,
                                              const rocblas_int ldx,
                                              rocblas_int* info)
{
    ROCSOLVER_ENTER_TOP("gels_outofplace", "--trans", trans, "-m", m, "-n", n, "--nrhs", nrhs,
                        "--lda", lda, "--ldb", ldb, "--ldx", ldx);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st = rocsolver_gels_outofplace_argCheck<COMPLEX>(handle, trans, m, n, nrhs, A,
                                                                    lda, B, ldb, X, ldx, info);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    const rocblas_int shiftA = 0;
    const rocblas_int shiftB = 0;
    const rocblas_int shiftX = 0;

    // normal (non-batched non-strided) execution
    const rocblas_stride strideA = 0;
    const rocblas_stride strideB = 0;
    const rocblas_stride strideX = 0;
    const rocblas_int batch_count = 1;

    // memory workspace sizes:
    // size for constants in rocblas calls
    size_t size_scalars;
    // size of workspace (for calling GEQRF/GELQF, ORMQR/ORMLQ, and TRSM)
    bool optim_mem;
    size_t size_work_x_temp, size_workArr_temp_arr, size_diag_trfac_invA,
        size_trfact_workTrmm_invA_arr, size_ipiv;
    // extra requirements to copy B
    size_t size_savedB;
    rocsolver_gels_outofplace_getMemorySize<false, false, T>(
        trans, m, n, nrhs, batch_count, &size_scalars, &size_work_x_temp, &size_workArr_temp_arr,
        &size_diag_trfac_invA, &size_trfact_workTrmm_invA_arr, &size_ipiv, &size_savedB, &optim_mem);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(
            handle, size_scalars, size_work_x_temp, size_workArr_temp_arr, size_diag_trfac_invA,
            size_trfact_workTrmm_invA_arr, size_ipiv, size_savedB);

    // memory workspace allocation
    void *scalars, *work_x_temp, *workArr_temp_arr, *diag_trfac_invA, *trfact_workTrmm_invA_arr,
        *ipiv, *savedB;
    rocblas_device_malloc mem(handle, size_scalars, size_work_x_temp, size_workArr_temp_arr,
                              size_diag_trfac_invA, size_trfact_workTrmm_invA_arr, size_ipiv,
                              size_savedB);

    if(!mem)
        return rocblas_status_memory_error;

    scalars = mem[0];
    work_x_temp = mem[1];
    workArr_temp_arr = mem[2];
    diag_trfac_invA = mem[3];
    trfact_workTrmm_invA_arr = mem[4];
    ipiv = mem[5];
    savedB = mem[6];
    if(size_scalars > 0)
        init_scalars(handle, (T*)scalars);

    // execution
    return rocsolver_gels_outofplace_template<false, false, T>(
        handle, trans, m, n, nrhs, A, shiftA, lda, strideA, B, shiftB, ldb, strideB, X, shiftX, ldx,
        strideX, info, batch_count, (T*)scalars, (T*)work_x_temp, (T*)workArr_temp_arr,
        (T*)diag_trfac_invA, (T**)trfact_workTrmm_invA_arr, (T*)ipiv, (T*)savedB, optim_mem);
}

ROCSOLVER_END_NAMESPACE

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

ROCSOLVER_EXPORT rocblas_status rocsolver_sgels_outofplace(rocblas_handle handle,
                                                           rocblas_operation trans,
                                                           const rocblas_int m,
                                                           const rocblas_int n,
                                                           const rocblas_int nrhs,
                                                           float* A,
                                                           const rocblas_int lda,
                                                           float* B,
                                                           const rocblas_int ldb,
                                                           float* X,
                                                           const rocblas_int ldx,
                                                           rocblas_int* info)
{
    return rocsolver::rocsolver_gels_outofplace_impl<float>(handle, trans, m, n, nrhs, A, lda, B,
                                                            ldb, X, ldx, info);
}

ROCSOLVER_EXPORT rocblas_status rocsolver_dgels_outofplace(rocblas_handle handle,
                                                           rocblas_operation trans,
                                                           const rocblas_int m,
                                                           const rocblas_int n,
                                                           const rocblas_int nrhs,
                                                           double* A,
                                                           const rocblas_int lda,
                                                           double* B,
                                                           const rocblas_int ldb,
                                                           double* X,
                                                           const rocblas_int ldx,
                                                           rocblas_int* info)
{
    return rocsolver::rocsolver_gels_outofplace_impl<double>(handle, trans, m, n, nrhs, A, lda, B,
                                                             ldb, X, ldx, info);
}

ROCSOLVER_EXPORT rocblas_status rocsolver_cgels_outofplace(rocblas_handle handle,
                                                           rocblas_operation trans,
                                                           const rocblas_int m,
                                                           const rocblas_int n,
                                                           const rocblas_int nrhs,
                                                           rocblas_float_complex* A,
                                                           const rocblas_int lda,
                                                           rocblas_float_complex* B,
                                                           const rocblas_int ldb,
                                                           rocblas_float_complex* X,
                                                           const rocblas_int ldx,
                                                           rocblas_int* info)
{
    return rocsolver::rocsolver_gels_outofplace_impl<rocblas_float_complex>(
        handle, trans, m, n, nrhs, A, lda, B, ldb, X, ldx, info);
}

ROCSOLVER_EXPORT rocblas_status rocsolver_zgels_outofplace(rocblas_handle handle,
                                                           rocblas_operation trans,
                                                           const rocblas_int m,
                                                           const rocblas_int n,
                                                           const rocblas_int nrhs,
                                                           rocblas_double_complex* A,
                                                           const rocblas_int lda,
                                                           rocblas_double_complex* B,
                                                           const rocblas_int ldb,
                                                           rocblas_double_complex* X,
                                                           const rocblas_int ldx,
                                                           rocblas_int* info)
{
    return rocsolver::rocsolver_gels_outofplace_impl<rocblas_double_complex>(
        handle, trans, m, n, nrhs, A, lda, B, ldb, X, ldx, info);
}

} // extern C
