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
 * ************************************************************************ */

#ifndef ROCBLAS_BETA_H
#define ROCBLAS_BETA_H

#include "rocblas-export.h"
#include "rocblas-macros.h"
#include "rocblas-types.h"

/*!\file
 * \brief rocblas_functions.h provides Basic Linear Algebra Subprograms of Level 1, 2 and 3,
 *  using HIP optimized for AMD HCC-based GPU hardware. This library can also run on CUDA-based
 * NVIDIA GPUs.
 *  This file exposes C99 BLAS interface
 */

/*
 * ===========================================================================
 *   README: Please follow the naming convention
 *   Big case for matrix, e.g. matrix A, B, C   GEMM (C = A*B)
 *   Lower case for vector, e.g. vector x, y    GEMV (y = A*x)
 * ===========================================================================
 */

#ifdef __cplusplus
extern "C" {
#endif

ROCBLAS_DEPRECATED_MSG(
    "rocblas_gemm_ex_get_solutions is a beta feature and is subject to change in future releases")
/*! @{
    \brief <b> BLAS BETA API </b>

    \details
    gemm_ex_get_solutions gets the indices for all the solutions that can solve a corresponding
    call to gemm_ex. Which solution is used by gemm_ex is controlled by the solution_index
    parameter.

    All parameters correspond to gemm_ex except for list_array and list_size, which are used as
    input and output for getting the solution indices. If list_array is NULL, list_size is an
    output and will be filled with the number of solutions that can solve the GEMM. If list_array
    is not NULL, then it must be pointing to an array with at least list_size elements and will
    be filled with the solution indices that can solve the GEMM: the number of elements filled is
    min(list_size, # of solutions).

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    transA    [rocblas_operation]
              specifies the form of op( A ).
    @param[in]
    transB    [rocblas_operation]
              specifies the form of op( B ).
    @param[in]
    m         [rocblas_int]
              matrix dimension m.
    @param[in]
    n         [rocblas_int]
              matrix dimension n.
    @param[in]
    k         [rocblas_int]
              matrix dimension k.
    @param[in]
    alpha     [const void *]
              device pointer or host pointer specifying the scalar alpha. Same datatype as compute_type.
    @param[in]
    a         [void *]
              device pointer storing matrix A.
    @param[in]
    a_type    [rocblas_datatype]
              specifies the datatype of matrix A.
    @param[in]
    lda       [rocblas_int]
              specifies the leading dimension of A.
    @param[in]
    b         [void *]
              device pointer storing matrix B.
    @param[in]
    b_type    [rocblas_datatype]
              specifies the datatype of matrix B.
    @param[in]
    ldb       [rocblas_int]
              specifies the leading dimension of B.
    @param[in]
    beta      [const void *]
              device pointer or host pointer specifying the scalar beta. Same datatype as compute_type.
    @param[in]
    c         [void *]
              device pointer storing matrix C.
    @param[in]
    c_type    [rocblas_datatype]
              specifies the datatype of matrix C.
    @param[in]
    ldc       [rocblas_int]
              specifies the leading dimension of C.
    @param[out]
    d         [void *]
              device pointer storing matrix D.
              If d and c pointers are to the same matrix then d_type must equal c_type and ldd must equal ldc
              or the respective invalid status will be returned.
    @param[in]
    d_type    [rocblas_datatype]
              specifies the datatype of matrix D.
    @param[in]
    ldd       [rocblas_int]
              specifies the leading dimension of D.
    @param[in]
    compute_type
              [rocblas_datatype]
              specifies the datatype of computation.
    @param[in]
    algo      [rocblas_gemm_algo]
              enumerant specifying the algorithm type.
    @param[in]
    flags     [uint32_t]
              optional gemm flags.
    @param[out]
    list_array [rocblas_int *]
               output array for solution indices or NULL if getting number of solutions
    @param[in,out]
    list_size  [rocblas_int *]
               size of list_array if getting solution indices or output with number of solutions
               if list_array is NULL

    ********************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_gemm_ex_get_solutions(rocblas_handle    handle,
                                                            rocblas_operation transA,
                                                            rocblas_operation transB,
                                                            rocblas_int       m,
                                                            rocblas_int       n,
                                                            rocblas_int       k,
                                                            const void*       alpha,
                                                            const void*       a,
                                                            rocblas_datatype  a_type,
                                                            rocblas_int       lda,
                                                            const void*       b,
                                                            rocblas_datatype  b_type,
                                                            rocblas_int       ldb,
                                                            const void*       beta,
                                                            const void*       c,
                                                            rocblas_datatype  c_type,
                                                            rocblas_int       ldc,
                                                            void*             d,
                                                            rocblas_datatype  d_type,
                                                            rocblas_int       ldd,
                                                            rocblas_datatype  compute_type,
                                                            rocblas_gemm_algo algo,
                                                            uint32_t          flags,
                                                            rocblas_int*      list_array,
                                                            rocblas_int*      list_size);

//! @}

ROCBLAS_DEPRECATED_MSG("rocblas_gemm_ex_get_solutions_by_type is a beta feature and is subject to "
                       "change in future releases")
/*! @{
    \brief <b> BLAS BETA API </b>

    \details
    rocblas_gemm_ex_get_solutions_by_type gets the indices for all the solutions that match the
    given types for gemm_ex. Which solution is used by gemm_ex is controlled by the
    solution_index parameter.

    If list_array is NULL, list_size is an output and will be filled with the number of solutions
    that can solve the GEMM. If list_array is not NULL, then it must be pointing to an array with
    at least list_size elements and will be filled with the solution indices that can solve the
    GEMM: the number of elements filled is min(list_size, # of solutions).

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    input_type
              [rocblas_datatype]
              specifies the datatype of matrix A.
    @param[in]
    output_type
              [rocblas_datatype]
              specifies the datatype of matrix D.
    @param[in]
    compute_type
              [rocblas_datatype]
              specifies the datatype of computation.
    @param[in]
    flags     [uint32_t]
              optional gemm flags.
    @param[out]
    list_array [rocblas_int *]
               output array for solution indices or NULL if getting number of solutions
    @param[in,out]
    list_size  [rocblas_int *]
               size of list_array if getting solution indices or output with number of solutions
               if list_array is NULL

    ********************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_gemm_ex_get_solutions_by_type(rocblas_handle   handle,
                                                                    rocblas_datatype input_type,
                                                                    rocblas_datatype output_type,
                                                                    rocblas_datatype compute_type,
                                                                    uint32_t         flags,
                                                                    rocblas_int*     list_array,
                                                                    rocblas_int*     list_size);

//! @}

ROCBLAS_DEPRECATED_MSG(
    "rocblas_gemm_batched_ex_get_solutions is a beta feature and is subject to change "
    "in future releases")
/*! @{
    \brief <b> BLAS BETA API </b>

    \details
    rocblas_gemm_batched_ex_get_solutions gets the indices for all the solutions that can solve a
    corresponding call to gemm_batched_ex. Which solution is used by gemm_batched_ex is
    controlled by the solution_index parameter.

    All parameters correspond to gemm_batched_ex except for list_array and list_size, which are
    used as input and output for getting the solution indices. If list_array is NULL, list_size is
    an output and will be filled with the number of solutions that can solve the GEMM. If
    list_array is not NULL, then it must be pointing to an array with at least list_size elements
    and will be filled with the solution indices that can solve the GEMM: the number of elements
    filled is min(list_size, # of solutions).

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    transA    [rocblas_operation]
              specifies the form of op( A ).
    @param[in]
    transB    [rocblas_operation]
              specifies the form of op( B ).
    @param[in]
    m         [rocblas_int]
              matrix dimension m.
    @param[in]
    n         [rocblas_int]
              matrix dimension n.
    @param[in]
    k         [rocblas_int]
              matrix dimension k.
    @param[in]
    alpha     [const void *]
              device pointer or host pointer specifying the scalar alpha. Same datatype as compute_type.
    @param[in]
    a         [void *]
              device pointer storing array of pointers to each matrix A_i.
    @param[in]
    a_type    [rocblas_datatype]
              specifies the datatype of each matrix A_i.
    @param[in]
    lda       [rocblas_int]
              specifies the leading dimension of each A_i.
    @param[in]
    b         [void *]
              device pointer storing array of pointers to each matrix B_i.
    @param[in]
    b_type    [rocblas_datatype]
              specifies the datatype of each matrix B_i.
    @param[in]
    ldb       [rocblas_int]
              specifies the leading dimension of each B_i.
    @param[in]
    beta      [const void *]
              device pointer or host pointer specifying the scalar beta. Same datatype as compute_type.
    @param[in]
    c         [void *]
              device array of device pointers to each matrix C_i.
    @param[in]
    c_type    [rocblas_datatype]
              specifies the datatype of each matrix C_i.
    @param[in]
    ldc       [rocblas_int]
              specifies the leading dimension of each C_i.
    @param[out]
    d         [void *]
              device array of device pointers to each matrix D_i.
              If d and c are the same array of matrix pointers then d_type must equal c_type and ldd must equal ldc
              or the respective invalid status will be returned.
    @param[in]
    d_type    [rocblas_datatype]
              specifies the datatype of each matrix D_i.
    @param[in]
    ldd       [rocblas_int]
              specifies the leading dimension of each D_i.
    @param[in]
    batch_count
              [rocblas_int]
              number of gemm operations in the batch.
    @param[in]
    compute_type
              [rocblas_datatype]
              specifies the datatype of computation.
    @param[in]
    algo      [rocblas_gemm_algo]
              enumerant specifying the algorithm type.
    @param[in]
    flags     [uint32_t]
              optional gemm flags.
    @param[out]
    list_array [rocblas_int *]
               output array for solution indices or NULL if getting number of solutions
    @param[in,out]
    list_size  [rocblas_int *]
               size of list_array if getting solution indices or output with number of solutions
               if list_array is NULL

    ********************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_gemm_batched_ex_get_solutions(rocblas_handle    handle,
                                                                    rocblas_operation transA,
                                                                    rocblas_operation transB,
                                                                    rocblas_int       m,
                                                                    rocblas_int       n,
                                                                    rocblas_int       k,
                                                                    const void*       alpha,
                                                                    const void*       a,
                                                                    rocblas_datatype  a_type,
                                                                    rocblas_int       lda,
                                                                    const void*       b,
                                                                    rocblas_datatype  b_type,
                                                                    rocblas_int       ldb,
                                                                    const void*       beta,
                                                                    const void*       c,
                                                                    rocblas_datatype  c_type,
                                                                    rocblas_int       ldc,
                                                                    void*             d,
                                                                    rocblas_datatype  d_type,
                                                                    rocblas_int       ldd,
                                                                    rocblas_int       batch_count,
                                                                    rocblas_datatype  compute_type,
                                                                    rocblas_gemm_algo algo,
                                                                    uint32_t          flags,
                                                                    rocblas_int*      list_array,
                                                                    rocblas_int*      list_size);

//! @}

ROCBLAS_DEPRECATED_MSG("rocblas_gemm_batched_ex_get_solutions_by_type is a beta feature and is "
                       "subject to change in future releases")
/*! @{
    \brief <b> BLAS BETA API </b>

    \details
    rocblas_gemm_batched_ex_get_solutions_by_type gets the indices for all the solutions that
    match the given types for gemm_batched_ex. Which solution is used by gemm_ex is controlled
    by the solution_index parameter.

    If list_array is NULL, list_size is an output and will be filled with the number of solutions
    that can solve the GEMM. If list_array is not NULL, then it must be pointing to an array with
    at least list_size elements and will be filled with the solution indices that can solve the
    GEMM: the number of elements filled is min(list_size, # of solutions).

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    input_type
              [rocblas_datatype]
              specifies the datatype of matrix A.
    @param[in]
    output_type
              [rocblas_datatype]
              specifies the datatype of matrix D.
    @param[in]
    compute_type
              [rocblas_datatype]
              specifies the datatype of computation.
    @param[in]
    flags     [uint32_t]
              optional gemm flags.
    @param[out]
    list_array [rocblas_int *]
               output array for solution indices or NULL if getting number of solutions
    @param[in,out]
    list_size  [rocblas_int *]
               size of list_array if getting solution indices or output with number of solutions
               if list_array is NULL

    ********************************************************************/
ROCBLAS_EXPORT rocblas_status
    rocblas_gemm_batched_ex_get_solutions_by_type(rocblas_handle   handle,
                                                  rocblas_datatype input_type,
                                                  rocblas_datatype output_type,
                                                  rocblas_datatype compute_type,
                                                  uint32_t         flags,
                                                  rocblas_int*     list_array,
                                                  rocblas_int*     list_size);

//! @}

ROCBLAS_DEPRECATED_MSG(
    "rocblas_gemm_strided_batched_ex_get_solutions is a beta feature and is subject "
    "to change in future releases")
/*! @{
    \brief <b> BLAS BETA API </b>

    \details
    gemm_strided_batched_ex_get_solutions gets the indices for all the solutions that can solve a
    corresponding call to gemm_strided_batched_ex. Which solution is used by
    gemm_strided_batched_ex is controlled by the solution_index parameter.

    All parameters correspond to gemm_strided_batched_ex except for list_array and list_size,
    which are used as input and output for getting the solution indices. If list_array is NULL,
    list_size is an output and will be filled with the number of solutions that can solve the
    GEMM. If list_array is not NULL, then it must be pointing to an array with at least list_size
    elements and will be filled with the solution indices that can solve the GEMM: the number of
    elements filled is min(list_size, # of solutions).

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    transA    [rocblas_operation]
              specifies the form of op( A ).
    @param[in]
    transB    [rocblas_operation]
              specifies the form of op( B ).
    @param[in]
    m         [rocblas_int]
              matrix dimension m.
    @param[in]
    n         [rocblas_int]
              matrix dimension n.
    @param[in]
    k         [rocblas_int]
              matrix dimension k.
    @param[in]
    alpha     [const void *]
              device pointer or host pointer specifying the scalar alpha. Same datatype as compute_type.
    @param[in]
    a         [void *]
              device pointer pointing to first matrix A_1.
    @param[in]
    a_type    [rocblas_datatype]
              specifies the datatype of each matrix A_i.
    @param[in]
    lda       [rocblas_int]
              specifies the leading dimension of each A_i.
    @param[in]
    stride_a  [rocblas_stride]
              specifies stride from start of one A_i matrix to the next A_(i + 1).
    @param[in]
    b         [void *]
              device pointer pointing to first matrix B_1.
    @param[in]
    b_type    [rocblas_datatype]
              specifies the datatype of each matrix B_i.
    @param[in]
    ldb       [rocblas_int]
              specifies the leading dimension of each B_i.
    @param[in]
    stride_b  [rocblas_stride]
              specifies stride from start of one B_i matrix to the next B_(i + 1).
    @param[in]
    beta      [const void *]
              device pointer or host pointer specifying the scalar beta. Same datatype as compute_type.
    @param[in]
    c         [void *]
              device pointer pointing to first matrix C_1.
    @param[in]
    c_type    [rocblas_datatype]
              specifies the datatype of each matrix C_i.
    @param[in]
    ldc       [rocblas_int]
              specifies the leading dimension of each C_i.
    @param[in]
    stride_c  [rocblas_stride]
              specifies stride from start of one C_i matrix to the next C_(i + 1).
    @param[out]
    d         [void *]
              device pointer storing each matrix D_i.
              If d and c pointers are to the same matrix then d_type must equal c_type and ldd must equal ldc
              and stride_d must equal stride_c or the respective invalid status will be returned.
    @param[in]
    d_type    [rocblas_datatype]
              specifies the datatype of each matrix D_i.
    @param[in]
    ldd       [rocblas_int]
              specifies the leading dimension of each D_i.
    @param[in]
    stride_d  [rocblas_stride]
              specifies stride from start of one D_i matrix to the next D_(i + 1).
    @param[in]
    batch_count
              [rocblas_int]
              number of gemm operations in the batch.
    @param[in]
    compute_type
              [rocblas_datatype]
              specifies the datatype of computation.
    @param[in]
    algo      [rocblas_gemm_algo]
              enumerant specifying the algorithm type.
    @param[in]
    flags     [uint32_t]
              optional gemm flags.
    @param[out]
    list_array [rocblas_int *]
               output array for solution indices or NULL if getting number of solutions
    @param[in,out]
    list_size  [rocblas_int *]
               size of list_array if getting solution indices or output with number of solutions
               if list_array is NULL

    ********************************************************************/
ROCBLAS_EXPORT rocblas_status
    rocblas_gemm_strided_batched_ex_get_solutions(rocblas_handle    handle,
                                                  rocblas_operation transA,
                                                  rocblas_operation transB,
                                                  rocblas_int       m,
                                                  rocblas_int       n,
                                                  rocblas_int       k,
                                                  const void*       alpha,
                                                  const void*       a,
                                                  rocblas_datatype  a_type,
                                                  rocblas_int       lda,
                                                  rocblas_stride    stride_a,
                                                  const void*       b,
                                                  rocblas_datatype  b_type,
                                                  rocblas_int       ldb,
                                                  rocblas_stride    stride_b,
                                                  const void*       beta,
                                                  const void*       c,
                                                  rocblas_datatype  c_type,
                                                  rocblas_int       ldc,
                                                  rocblas_stride    stride_c,
                                                  void*             d,
                                                  rocblas_datatype  d_type,
                                                  rocblas_int       ldd,
                                                  rocblas_stride    stride_d,
                                                  rocblas_int       batch_count,
                                                  rocblas_datatype  compute_type,
                                                  rocblas_gemm_algo algo,
                                                  uint32_t          flags,
                                                  rocblas_int*      list_array,
                                                  rocblas_int*      list_size);

//! @}

#ifdef __cplusplus
}
#endif

#endif /* ROCBLAS_BETA_H */
