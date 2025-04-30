/* **************************************************************************
 * Copyright (C) 2020-2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include "rocsolver/rocsolver.h"

// We need to include extern definitions for these inline functions to ensure
// that librocsolver.so will contain these symbols for FFI or when inlining
// is disabled.

extern "C" {

// deprecated functions use deprecated types
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

rocsolver_status rocsolver_create_handle(rocsolver_handle* handle)
{
    const rocblas_status stat = rocblas_create_handle(handle);
    if(stat != rocblas_status_success)
    {
        return stat;
    }
    return rocblas_set_pointer_mode(*handle, rocblas_pointer_mode_device);
}

rocsolver_status rocsolver_destroy_handle(rocsolver_handle handle)
{
    return rocblas_destroy_handle(handle);
}

rocsolver_status rocsolver_set_stream(rocsolver_handle handle, hipStream_t stream)
{
    return rocblas_set_stream(handle, stream);
}

rocsolver_status rocsolver_get_stream(rocsolver_handle handle, hipStream_t* stream)
{
    return rocblas_get_stream(handle, stream);
}

rocsolver_status rocsolver_set_vector(rocsolver_int n,
                                      rocsolver_int elem_size,
                                      const void* x,
                                      rocsolver_int incx,
                                      void* y,
                                      rocsolver_int incy)
{
    return rocblas_set_vector(n, elem_size, x, incx, y, incy);
}

rocsolver_status rocsolver_get_vector(rocsolver_int n,
                                      rocsolver_int elem_size,
                                      const void* x,
                                      rocsolver_int incx,
                                      void* y,
                                      rocsolver_int incy)
{
    return rocblas_get_vector(n, elem_size, x, incx, y, incy);
}

rocsolver_status rocsolver_set_matrix(rocsolver_int rows,
                                      rocsolver_int cols,
                                      rocsolver_int elem_size,
                                      const void* a,
                                      rocsolver_int lda,
                                      void* b,
                                      rocsolver_int ldb)
{
    return rocblas_set_matrix(rows, cols, elem_size, a, lda, b, ldb);
}

rocsolver_status rocsolver_get_matrix(rocsolver_int rows,
                                      rocsolver_int cols,
                                      rocsolver_int elem_size,
                                      const void* a,
                                      rocsolver_int lda,
                                      void* b,
                                      rocsolver_int ldb)
{
    return rocblas_get_matrix(rows, cols, elem_size, a, lda, b, ldb);
}

#pragma GCC diagnostic pop // re-enable deprecation warnings
}
