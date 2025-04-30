/* ************************************************************************
 * Copyright (C) 2016-2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 *
 * ************************************************************************ */
#include "logging.hpp"
#include "rocblas_block_sizes.h"
#include "rocblas_trtri.hpp"
#include "utility.hpp"

namespace
{

    template <typename>
    constexpr char rocblas_trtri_name[] = "unknown";
    template <>
    constexpr char rocblas_trtri_name<float>[] = "rocblas_strtri_batched";
    template <>
    constexpr char rocblas_trtri_name<double>[] = "rocblas_dtrtri_batched";
    template <>
    constexpr char rocblas_trtri_name<rocblas_float_complex>[] = "rocblas_ctrtri_batched";
    template <>
    constexpr char rocblas_trtri_name<rocblas_double_complex>[] = "rocblas_ztrtri_batched";

    template <typename T>
    rocblas_status rocblas_trtri_batched_impl(rocblas_handle   handle,
                                              rocblas_fill     uplo,
                                              rocblas_diagonal diag,
                                              rocblas_int      n,
                                              const T* const   A[],
                                              rocblas_int      lda,
                                              T* const         invA[],
                                              rocblas_int      ldinvA,
                                              rocblas_int      batch_count)
    {
        if(!handle)
            return rocblas_status_invalid_handle;

        // Compute the optimal size for temporary device memory
        size_t els   = rocblas_internal_trtri_temp_elements(n, 1);
        size_t size  = els * batch_count * sizeof(T);
        size_t sizep = batch_count * sizeof(T*);
        if(handle->is_device_memory_size_query())
        {
            if(!n || !batch_count)
                return rocblas_status_size_unchanged;
            return handle->set_optimal_device_memory_size(size, sizep);
        }

        auto                    layer_mode     = handle->layer_mode;
        auto                    check_numerics = handle->check_numerics;
        rocblas_internal_logger logger;

        if(layer_mode & rocblas_layer_mode_log_trace)
            logger.log_trace(
                handle, rocblas_trtri_name<T>, uplo, diag, n, A, lda, invA, ldinvA, batch_count);

        if(layer_mode & rocblas_layer_mode_log_profile)
            logger.log_profile(handle,
                               rocblas_trtri_name<T>,
                               "uplo",
                               rocblas_fill_letter(uplo),
                               "diag",
                               rocblas_diag_letter(diag),
                               "N",
                               n,
                               "lda",
                               lda,
                               "ldinvA",
                               ldinvA,
                               "batch_count",
                               batch_count);

        rocblas_status arg_status
            = rocblas_trtri_arg_check(handle, uplo, diag, n, A, lda, invA, ldinvA, batch_count);
        if(arg_status != rocblas_status_continue)
            return arg_status;

        // Allocate memory
        auto  w_mem = handle->device_malloc(size, sizep);
        void *w_C_tmp, *w_C_tmp_arr;
        if(!w_mem)
        {
            return rocblas_status_memory_error;
        }

        if(size)
        {
            w_C_tmp     = w_mem[0];
            w_C_tmp_arr = w_mem[1];

            // kernel to copy strided_batched array to batched format
            constexpr int ARRAY_NB = 128;
            RETURN_IF_ROCBLAS_ERROR(setup_batched_array<ARRAY_NB>(
                handle->get_stream(), (T*)w_C_tmp, els, (T**)w_C_tmp_arr, batch_count));
        }

        if(check_numerics)
        {
            bool           is_input = true;
            rocblas_status trtri_check_numerics_status
                = rocblas_trtri_check_numerics(rocblas_trtri_name<T>,
                                               handle,
                                               uplo,
                                               n,
                                               A,
                                               lda,
                                               0,
                                               invA,
                                               ldinvA,
                                               0,
                                               batch_count,
                                               check_numerics,
                                               is_input);
            if(trtri_check_numerics_status != rocblas_status_success)
                return trtri_check_numerics_status;
        }

        rocblas_status status = rocblas_internal_trtri_batched_template(handle,
                                                                        uplo,
                                                                        diag,
                                                                        n,
                                                                        A,
                                                                        0,
                                                                        lda,
                                                                        0,
                                                                        0,
                                                                        invA,
                                                                        0,
                                                                        ldinvA,
                                                                        0,
                                                                        0,
                                                                        batch_count,
                                                                        1,
                                                                        (T* const*)w_C_tmp_arr);

        if(status != rocblas_status_success)
            return status;

        if(check_numerics)
        {
            bool           is_input = false;
            rocblas_status trtri_check_numerics_status
                = rocblas_trtri_check_numerics(rocblas_trtri_name<T>,
                                               handle,
                                               uplo,
                                               n,
                                               A,
                                               lda,
                                               0,
                                               invA,
                                               ldinvA,
                                               0,
                                               batch_count,
                                               check_numerics,
                                               is_input);
            if(trtri_check_numerics_status != rocblas_status_success)
                return trtri_check_numerics_status;
        }
        return status;
    }

}

extern "C" {
rocblas_status rocblas_strtri_batched(rocblas_handle     handle,
                                      rocblas_fill       uplo,
                                      rocblas_diagonal   diag,
                                      rocblas_int        n,
                                      const float* const A[],
                                      rocblas_int        lda,
                                      float* const       invA[],
                                      rocblas_int        ldinvA,
                                      rocblas_int        batch_count)
try
{
    return rocblas_trtri_batched_impl(handle, uplo, diag, n, A, lda, invA, ldinvA, batch_count);
}
catch(...)
{
    return exception_to_rocblas_status();
}

rocblas_status rocblas_dtrtri_batched(rocblas_handle      handle,
                                      rocblas_fill        uplo,
                                      rocblas_diagonal    diag,
                                      rocblas_int         n,
                                      const double* const A[],
                                      rocblas_int         lda,
                                      double* const       invA[],
                                      rocblas_int         ldinvA,
                                      rocblas_int         batch_count)
try
{
    return rocblas_trtri_batched_impl(handle, uplo, diag, n, A, lda, invA, ldinvA, batch_count);
}
catch(...)
{
    return exception_to_rocblas_status();
}

rocblas_status rocblas_ctrtri_batched(rocblas_handle                     handle,
                                      rocblas_fill                       uplo,
                                      rocblas_diagonal                   diag,
                                      rocblas_int                        n,
                                      const rocblas_float_complex* const A[],
                                      rocblas_int                        lda,
                                      rocblas_float_complex* const       invA[],
                                      rocblas_int                        ldinvA,
                                      rocblas_int                        batch_count)
try
{
    return rocblas_trtri_batched_impl(handle, uplo, diag, n, A, lda, invA, ldinvA, batch_count);
}
catch(...)
{
    return exception_to_rocblas_status();
}

rocblas_status rocblas_ztrtri_batched(rocblas_handle                      handle,
                                      rocblas_fill                        uplo,
                                      rocblas_diagonal                    diag,
                                      rocblas_int                         n,
                                      const rocblas_double_complex* const A[],
                                      rocblas_int                         lda,
                                      rocblas_double_complex* const       invA[],
                                      rocblas_int                         ldinvA,
                                      rocblas_int                         batch_count)
try
{
    return rocblas_trtri_batched_impl(handle, uplo, diag, n, A, lda, invA, ldinvA, batch_count);
}
catch(...)
{
    return exception_to_rocblas_status();
}

} // extern "C"
