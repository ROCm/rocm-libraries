/************************************************************************
 * Copyright (C) 2022-2024 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#pragma once

#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"

ROCSOLVER_BEGIN_NAMESPACE

template <typename T>
rocblas_status rocsolver_lauum_argCheck(rocblas_handle handle,
                                        const rocblas_fill uplo,
                                        const rocblas_int n,
                                        T A,
                                        const rocblas_int lda)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    if(uplo != rocblas_fill_upper && uplo != rocblas_fill_lower)
        return rocblas_status_invalid_value;

    // 2. invalid size
    if(n < 0 || lda < n)
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if(n && !A)
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <typename T>
void rocsolver_lauum_getMemorySize(const rocblas_int n, const rocblas_int batch_count, size_t* size_work)
{
    *size_work = 0;

    // if quick return, no workspace is needed
    if(n == 0 || batch_count == 0)
        return;

    // size of workspace
    *size_work = sizeof(T) * n * n * batch_count;
}

template <typename T, typename U>
rocblas_status rocsolver_lauum_template(rocblas_handle handle,
                                        const rocblas_fill uplo,
                                        const rocblas_int n,
                                        U A,
                                        const rocblas_int shiftA,
                                        const rocblas_int lda,
                                        const rocblas_stride strideA,
                                        const rocblas_int batch_count,
                                        T* work)
{
    ROCSOLVER_ENTER("lauum", "uplo:", uplo, "n:", n, "shiftA:", shiftA, "lda:", lda,
                    "strideA:", strideA, "bc:", batch_count);

    // quick return
    if(n == 0 || batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // everything must be executed with scalars on the host
    rocblas_pointer_mode old_mode;
    rocblas_get_pointer_mode(handle, &old_mode);
    rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);

    rocblas_int strideW = n * n;
    rocblas_int blocks = (n - 1) / BS2 + 1;
    dim3 grid(blocks, blocks, batch_count);
    dim3 threads(BS2, BS2);
    T one = 1;
    T zero = 0;

    // put the triangular factor of interest in work
    ROCSOLVER_LAUNCH_KERNEL(set_zero<T>, grid, threads, 0, stream, n, n, work, 0, n, strideW);
    ROCSOLVER_LAUNCH_KERNEL(copy_mat<T>, grid, threads, 0, stream, n, n, A, shiftA, lda, strideA,
                            (T*)work, 0, n, strideW, no_mask{}, uplo);

    rocblas_side side = (uplo == rocblas_fill_upper) ? rocblas_side_right : rocblas_side_left;

    // work = work * A' or work = A' * work
    rocblasCall_trmm(handle, side, uplo, rocblas_operation_conjugate_transpose,
                     rocblas_diagonal_non_unit, n, n, &one, 0, A, shiftA, lda, strideA, work, 0, n,
                     strideW, batch_count);

    // copy the new factor into the relevant triangle of A leaving the rest untouched
    ROCSOLVER_LAUNCH_KERNEL(copy_mat<T>, grid, threads, 0, stream, n, n, work, 0, n, strideW, A,
                            shiftA, lda, strideA, no_mask{}, uplo);

    rocblas_set_pointer_mode(handle, old_mode);
    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE
