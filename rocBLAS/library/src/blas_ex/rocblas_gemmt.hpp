/* ************************************************************************
 * Copyright (C) 2016-2024 Advanced Micro Devices, Inc. All rights reserved.
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
 * ************************************************************************ */

#pragma once

#include "check_numerics_matrix.hpp"
#include "handle.hpp"
#include "int64_helpers.hpp"

/*******************************************************************************
 * Validate Arguments
 ******************************************************************************/
template <typename API_INT, typename TScal, typename TConstPtr, typename TPtr>
inline rocblas_status rocblas_gemmt_arg_check(rocblas_handle    handle,
                                              rocblas_fill      uplo,
                                              rocblas_operation trans_a,
                                              rocblas_operation trans_b,
                                              API_INT           n,
                                              API_INT           k,
                                              TScal             alpha,
                                              TConstPtr         A,
                                              int64_t           lda,
                                              TConstPtr         B,
                                              int64_t           ldb,
                                              TScal             beta,
                                              TPtr              C,
                                              int64_t           ldc,
                                              API_INT           batch_count = 1)
{
    // handle must be valid
    if(!handle)
        return rocblas_status_invalid_handle;

    if(trans_a != rocblas_operation_none && trans_a != rocblas_operation_transpose
       && trans_a != rocblas_operation_conjugate_transpose)
        return rocblas_status_invalid_value;
    if(trans_b != rocblas_operation_none && trans_b != rocblas_operation_transpose
       && trans_b != rocblas_operation_conjugate_transpose)
        return rocblas_status_invalid_value;
    if(uplo != rocblas_fill_lower && uplo != rocblas_fill_upper)
        return rocblas_status_invalid_value;

    // sizes must not be negative
    if(n < 0 || k < 0 || batch_count < 0)
        return rocblas_status_invalid_size;

    rocblas_int num_rows_a = trans_a == rocblas_operation_none ? n : k;
    rocblas_int num_rows_b = trans_b == rocblas_operation_none ? k : n;
    rocblas_int num_rows_c = n;

    // leading dimensions must be valid
    if(num_rows_a > lda || num_rows_b > ldb || num_rows_c > ldc)
        return rocblas_status_invalid_size;

    // quick return 0 is valid in BLAS
    // Note: k==0 is not a quick return, because C must still be multiplied by beta
    if(!n || !batch_count)
        return rocblas_status_success;

    if((k > 0 && !alpha) || !beta)
        return rocblas_status_invalid_pointer;

    if(handle->pointer_mode == rocblas_pointer_mode_host)
    {
        bool calcAB = k > 0 && *alpha != 0;

        if(!calcAB && *beta == 1)
            return rocblas_status_success;

        if((calcAB && (!A || !B)) || ((calcAB || *beta != 1) && !C))
            return rocblas_status_invalid_pointer;
    }

    return rocblas_status_continue;
}

/*
 * ===========================================================================
 *    template interface
 * ===========================================================================
 */

template <typename API_INT, typename TScal, typename TConstPtr, typename TPtr>
rocblas_status rocblas_internal_gemmt_launcher(rocblas_handle    handle,
                                               rocblas_fill      uplo,
                                               rocblas_operation trans_a,
                                               rocblas_operation trans_b,
                                               rocblas_int       n,
                                               API_INT           k,
                                               const TScal*      alpha,
                                               TConstPtr         A,
                                               API_INT           lda,
                                               rocblas_stride    stride_a,
                                               TConstPtr         B,
                                               API_INT           ldb,
                                               rocblas_stride    stride_b,
                                               const TScal*      beta,
                                               TPtr              C,
                                               API_INT           ldc,
                                               rocblas_stride    stride_c,
                                               rocblas_int       batch_count);

template <typename T, typename API_INT, typename TConstPtr, typename TPtr>
rocblas_status rocblas_gemmt_check_numerics(const char*       function_name,
                                            rocblas_handle    handle,
                                            rocblas_fill      uplo,
                                            rocblas_operation trans_a,
                                            rocblas_operation trans_b,
                                            API_INT           n,
                                            API_INT           k,
                                            TConstPtr         A,
                                            API_INT           lda,
                                            rocblas_stride    stride_a,
                                            TConstPtr         B,
                                            API_INT           ldb,
                                            rocblas_stride    stride_b,
                                            TPtr              C,
                                            API_INT           ldc,
                                            rocblas_stride    stride_c,
                                            API_INT           batch_count,
                                            const int         check_numerics,
                                            bool              is_input)
{
    rocblas_status check_numerics_status = rocblas_status_success;

    if(is_input)
    {
        check_numerics_status
            = rocblas_internal_check_numerics_matrix_template(function_name,
                                                              handle,
                                                              trans_a,
                                                              rocblas_fill_full,
                                                              rocblas_client_general_matrix,
                                                              n,
                                                              k,
                                                              A,
                                                              0,
                                                              lda,
                                                              stride_a,
                                                              batch_count,
                                                              check_numerics,
                                                              is_input);
        if(check_numerics_status != rocblas_status_success)
            return check_numerics_status;

        check_numerics_status
            = rocblas_internal_check_numerics_matrix_template(function_name,
                                                              handle,
                                                              trans_b,
                                                              rocblas_fill_full,
                                                              rocblas_client_general_matrix,
                                                              k,
                                                              n,
                                                              B,
                                                              0,
                                                              ldb,
                                                              stride_b,
                                                              batch_count,
                                                              check_numerics,
                                                              is_input);
        if(check_numerics_status != rocblas_status_success)
            return check_numerics_status;
    }

    check_numerics_status
        = rocblas_internal_check_numerics_matrix_template(function_name,
                                                          handle,
                                                          rocblas_operation_none,
                                                          uplo,
                                                          rocblas_client_triangular_matrix,
                                                          n,
                                                          n,
                                                          C,
                                                          0,
                                                          ldc,
                                                          stride_c,
                                                          batch_count,
                                                          check_numerics,
                                                          is_input);

    return check_numerics_status;
}
