/* ************************************************************************
 * Copyright (C) 2021-2025 Advanced Micro Devices, Inc.
 * *************************************************************************/

#include "roclapack_syevd_heevd.hpp"

ROCSOLVER_BEGIN_NAMESPACE

template <typename T, typename S, typename W>
rocblas_status rocsolver_syevd_heevd_impl(rocblas_handle handle,
                                          const rocblas_evect evect,
                                          const rocblas_fill uplo,
                                          const rocblas_int n,
                                          W A,
                                          const rocblas_int lda,
                                          S* D,
                                          S* E,
                                          rocblas_int* info)
{
    const char* name = (!rocblas_is_complex<T> ? "syevd" : "heevd");
    ROCSOLVER_ENTER_TOP(name, "--evect", evect, "--uplo", uplo, "-n", n, "--lda", lda);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st = rocsolver_syevd_heevd_argCheck(handle, evect, uplo, n, A, lda, D, E, info);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_int shiftA = 0;

    // normal (non-batched non-strided) execution
    rocblas_stride strideA = 0;
    rocblas_stride strideD = 0;
    rocblas_stride strideE = 0;
    rocblas_int batch_count = 1;

    // memory workspace sizes:
    bool optim_mem;
    // size for constants in rocblas calls
    size_t size_scalars;
    // size of reusable workspaces
    size_t size_work1;
    size_t size_work2;
    size_t size_work3;
    size_t size_work4;
    size_t size_tmptau_W;
    // extra space for call stedc
    size_t size_splits, size_tmpz;
    // size of array of pointers (only for batched case)
    size_t size_workArr;
    // size for temporary householder scalars
    size_t size_tau;

    rocsolver_syevd_heevd_getMemorySize<false, false, T, S>(
        handle, evect, uplo, n, batch_count, &size_scalars, &size_work1, &size_work2, &size_work3,
        &size_work4, &size_tmpz, &size_splits, &size_tmptau_W, &size_tau, &size_workArr, &optim_mem);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle, size_scalars, size_work1, size_work2,
                                                      size_work3, size_work4, size_tmpz, size_splits,
                                                      size_tmptau_W, size_tau, size_workArr);

    // memory workspace allocation
    void *scalars, *work1, *work2, *work3, *work4, *tmpz, *splits, *tmptau_W, *tau, *workArr;
    rocblas_device_malloc mem(handle, size_scalars, size_work1, size_work2, size_work3, size_work4,
                              size_tmpz, size_splits, size_tmptau_W, size_tau, size_workArr);

    if(!mem)
        return rocblas_status_memory_error;

    scalars = mem[0];
    work1 = mem[1];
    work2 = mem[2];
    work3 = mem[3];
    work4 = mem[4];
    tmpz = mem[5];
    splits = mem[6];
    tmptau_W = mem[7];
    tau = mem[8];
    workArr = mem[9];
    if(size_scalars > 0)
        init_scalars(handle, (T*)scalars);

    // execution
    return rocsolver_syevd_heevd_template<false, false, T>(
        handle, evect, uplo, n, A, shiftA, lda, strideA, D, strideD, E, strideE, info, batch_count,
        (T*)scalars, work1, work2, work3, work4, (S*)tmpz, (rocblas_int*)splits, (T*)tmptau_W,
        (T*)tau, (T**)workArr, optim_mem);
}

ROCSOLVER_END_NAMESPACE

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocsolver_ssyevd(rocblas_handle handle,
                                const rocblas_evect evect,
                                const rocblas_fill uplo,
                                const rocblas_int n,
                                float* A,
                                const rocblas_int lda,
                                float* D,
                                float* E,
                                rocblas_int* info)
{
    return rocsolver::rocsolver_syevd_heevd_impl<float>(handle, evect, uplo, n, A, lda, D, E, info);
}

rocblas_status rocsolver_dsyevd(rocblas_handle handle,
                                const rocblas_evect evect,
                                const rocblas_fill uplo,
                                const rocblas_int n,
                                double* A,
                                const rocblas_int lda,
                                double* D,
                                double* E,
                                rocblas_int* info)
{
    return rocsolver::rocsolver_syevd_heevd_impl<double>(handle, evect, uplo, n, A, lda, D, E, info);
}

rocblas_status rocsolver_cheevd(rocblas_handle handle,
                                const rocblas_evect evect,
                                const rocblas_fill uplo,
                                const rocblas_int n,
                                rocblas_float_complex* A,
                                const rocblas_int lda,
                                float* D,
                                float* E,
                                rocblas_int* info)
{
    return rocsolver::rocsolver_syevd_heevd_impl<rocblas_float_complex>(handle, evect, uplo, n, A,
                                                                        lda, D, E, info);
}

rocblas_status rocsolver_zheevd(rocblas_handle handle,
                                const rocblas_evect evect,
                                const rocblas_fill uplo,
                                const rocblas_int n,
                                rocblas_double_complex* A,
                                const rocblas_int lda,
                                double* D,
                                double* E,
                                rocblas_int* info)
{
    return rocsolver::rocsolver_syevd_heevd_impl<rocblas_double_complex>(handle, evect, uplo, n, A,
                                                                         lda, D, E, info);
}

} // extern C
