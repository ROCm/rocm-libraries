/* **************************************************************************
 * Copyright (C) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#pragma once

#include <rocblas/rocblas.h>

#ifdef HAVE_ROCSPARSE
#include <rocsparse/rocsparse.h>
#else
#include "rocblas_utility.hpp"

typedef enum rocsparse_status_
{
    rocsparse_status_success = 0,
    rocsparse_status_invalid_handle = 1,
    rocsparse_status_not_implemented = 2,
    rocsparse_status_invalid_pointer = 3,
    rocsparse_status_invalid_size = 4,
    rocsparse_status_memory_error = 5,
    rocsparse_status_internal_error = 6,
    rocsparse_status_invalid_value = 7,
    rocsparse_status_arch_mismatch = 8,
    rocsparse_status_zero_pivot = 9,
    rocsparse_status_not_initialized = 10,
    rocsparse_status_type_mismatch = 11,
    rocsparse_status_requires_sorted_storage = 12,
    rocsparse_status_thrown_exception = 13,
    rocsparse_status_continue = 14
} rocsparse_status;

typedef enum rocsparse_data_status_
{
    rocsparse_data_status_success = 0, /**< success. */
    rocsparse_data_status_inf = 1, /**< An inf value detected. */
    rocsparse_data_status_nan = 2, /**< An nan value detected. */
    rocsparse_data_status_invalid_offset_ptr = 3, /**< An invalid row pointer offset detected. */
    rocsparse_data_status_invalid_index = 4, /**< An invalid row indice detected. */
    rocsparse_data_status_duplicate_entry = 5, /**< Duplicate indice detected. */
    rocsparse_data_status_invalid_sorting = 6, /**< Incorrect sorting detected. */
    rocsparse_data_status_invalid_fill = 7 /**< Incorrect fill mode detected. */
} rocsparse_data_status;

#endif /* HAVE_ROCSPARSE */

ROCSOLVER_BEGIN_NAMESPACE

constexpr auto rocsparse2string_status(rocsparse_status status)
{
    switch(status)
    {
    case rocsparse_status_success: return "rocsparse_status_success";
    case rocsparse_status_invalid_handle: return "rocsparse_status_invalid_handle";
    case rocsparse_status_not_implemented: return "rocsparse_status_not_implemented";
    case rocsparse_status_invalid_pointer: return "rocsparse_status_invalid_pointer";
    case rocsparse_status_invalid_size: return "rocsparse_status_invalid_size";
    case rocsparse_status_memory_error: return "rocsparse_status_memory_error";
    case rocsparse_status_internal_error: return "rocsparse_status_internal_error";
    case rocsparse_status_invalid_value: return "rocsparse_status_invalid_value";
    case rocsparse_status_arch_mismatch: return "rocsparse_status_arch_mismatch";
    case rocsparse_status_zero_pivot: return "rocsparse_status_zero_pivot";
    case rocsparse_status_not_initialized: return "rocsparse_status_not_initialized";
    case rocsparse_status_type_mismatch: return "rocsparse_status_type_mismatch";
    case rocsparse_status_requires_sorted_storage:
        return "rocsparse_status_requires_sorted_storage";
    default: return "unknown";
    }
}

constexpr auto rocsparse2rocblas_status(rocsparse_status status)
{
    switch(status)
    {
    case rocsparse_status_success: return rocblas_status_success;
    case rocsparse_status_invalid_handle: return rocblas_status_invalid_handle;
    case rocsparse_status_not_implemented: return rocblas_status_not_implemented;
    case rocsparse_status_invalid_pointer: return rocblas_status_invalid_pointer;
    case rocsparse_status_invalid_size: return rocblas_status_invalid_size;
    case rocsparse_status_memory_error: return rocblas_status_memory_error;
    case rocsparse_status_invalid_value: return rocblas_status_invalid_value;
    default: return rocblas_status_internal_error;
    }
}

ROCSOLVER_END_NAMESPACE

#ifndef HAVE_ROCSPARSE

typedef enum rocsparse_operation_
{
    rocsparse_operation_none = 111, /**< Operate with matrix. */
    rocsparse_operation_transpose = 112, /**< Operate with transpose. */
    rocsparse_operation_conjugate_transpose = 113 /**< Operate with conj. transpose. */
} rocsparse_operation;

typedef enum rocsparse_analysis_policy_
{
    rocsparse_analysis_policy_reuse = 0, /**< try to re-use meta data. */
    rocsparse_analysis_policy_force = 1 /**< force to re-build meta data. */
} rocsparse_analysis_policy;

typedef enum rocsparse_solve_policy_
{
    rocsparse_solve_policy_auto = 0 /**< automatically decide on level information. */
} rocsparse_solve_policy;

typedef enum rocsparse_index_base_
{
    rocsparse_index_base_zero = 0, /**< zero based indexing. */
    rocsparse_index_base_one = 1 /**< one based indexing. */
} rocsparse_index_base;

typedef enum rocsparse_matrix_type_
{
    rocsparse_matrix_type_general = 0, /**< general matrix type. */
    rocsparse_matrix_type_symmetric = 1, /**< symmetric matrix type. */
    rocsparse_matrix_type_hermitian = 2, /**< hermitian matrix type. */
    rocsparse_matrix_type_triangular = 3 /**< triangular matrix type. */
} rocsparse_matrix_type;

typedef enum rocsparse_fill_mode_
{
    rocsparse_fill_mode_lower = 0, /**< lower triangular part is stored. */
    rocsparse_fill_mode_upper = 1 /**< upper triangular part is stored. */
} rocsparse_fill_mode;

typedef enum rocsparse_diag_type_
{
    rocsparse_diag_type_non_unit = 0, /**< diagonal entries are non-unity. */
    rocsparse_diag_type_unit = 1 /**< diagonal entries are unity */
} rocsparse_diag_type;

#endif /* HAVE_ROCSPARSE */

#define ROCSPARSE_CHECK(...)                                     \
    {                                                            \
        rocsparse_status _status = (__VA_ARGS__);                \
        if(_status != rocsparse_status_success)                  \
            return rocsolver::rocsparse2rocblas_status(_status); \
    }
#define THROW_IF_ROCSPARSE_ERROR(...)                           \
    {                                                           \
        rocsparse_status _status = (__VA_ARGS__);               \
        if(_status != rocsparse_status_success)                 \
            throw rocsolver::rocsparse2rocblas_status(_status); \
    }

#ifndef HAVE_ROCSPARSE

#if defined(rocsparse_ILP64)
typedef int64_t rocsparse_int;
#else
typedef int32_t rocsparse_int;
#endif

typedef struct _rocsparse_handle* rocsparse_handle;
typedef struct _rocsparse_mat_descr* rocsparse_mat_descr;
typedef struct _rocsparse_mat_info* rocsparse_mat_info;

typedef struct
{
    float x, y;
} rocsparse_float_complex;

typedef struct
{
    double x, y;
} rocsparse_double_complex;

ROCSOLVER_BEGIN_NAMESPACE

typedef rocsparse_status (*fp_rocsparse_create_handle)(rocsparse_handle* handle);
extern fp_rocsparse_create_handle g_sparse_create_handle;
#define rocsparse_create_handle ::rocsolver::g_sparse_create_handle

typedef rocsparse_status (*fp_rocsparse_destroy_handle)(rocsparse_handle handle);
extern fp_rocsparse_destroy_handle g_sparse_destroy_handle;
#define rocsparse_destroy_handle ::rocsolver::g_sparse_destroy_handle

typedef rocsparse_status (*fp_rocsparse_set_stream)(rocsparse_handle handle, hipStream_t stream);
extern fp_rocsparse_set_stream g_sparse_set_stream;
#define rocsparse_set_stream ::rocsolver::g_sparse_set_stream

typedef rocsparse_status (*fp_rocsparse_create_mat_descr)(rocsparse_mat_descr* descr);
extern fp_rocsparse_create_mat_descr g_sparse_create_mat_descr;
#define rocsparse_create_mat_descr ::rocsolver::g_sparse_create_mat_descr

typedef rocsparse_status (*fp_rocsparse_destroy_mat_descr)(rocsparse_mat_descr descr);
extern fp_rocsparse_destroy_mat_descr g_sparse_destroy_mat_descr;
#define rocsparse_destroy_mat_descr ::rocsolver::g_sparse_destroy_mat_descr

typedef rocsparse_status (*fp_rocsparse_set_mat_type)(rocsparse_mat_descr descr,
                                                      rocsparse_matrix_type type);
extern fp_rocsparse_set_mat_type g_sparse_set_mat_type;
#define rocsparse_set_mat_type ::rocsolver::g_sparse_set_mat_type

typedef rocsparse_status (*fp_rocsparse_set_mat_index_base)(rocsparse_mat_descr descr,
                                                            rocsparse_index_base base);
extern fp_rocsparse_set_mat_index_base g_sparse_set_mat_index_base;
#define rocsparse_set_mat_index_base ::rocsolver::g_sparse_set_mat_index_base

typedef rocsparse_status (*fp_rocsparse_set_mat_fill_mode)(rocsparse_mat_descr descr,
                                                           rocsparse_fill_mode fill_mode);
extern fp_rocsparse_set_mat_fill_mode g_sparse_set_mat_fill_mode;
#define rocsparse_set_mat_fill_mode ::rocsolver::g_sparse_set_mat_fill_mode

typedef rocsparse_status (*fp_rocsparse_set_mat_diag_type)(rocsparse_mat_descr descr,
                                                           rocsparse_diag_type diag_type);
extern fp_rocsparse_set_mat_diag_type g_sparse_set_mat_diag_type;
#define rocsparse_set_mat_diag_type ::rocsolver::g_sparse_set_mat_diag_type

typedef rocsparse_status (*fp_rocsparse_create_mat_info)(rocsparse_mat_info* info);
extern fp_rocsparse_create_mat_info g_sparse_create_mat_info;
#define rocsparse_create_mat_info ::rocsolver::g_sparse_create_mat_info

typedef rocsparse_status (*fp_rocsparse_destroy_mat_info)(rocsparse_mat_info info);
extern fp_rocsparse_destroy_mat_info g_sparse_destroy_mat_info;
#define rocsparse_destroy_mat_info ::rocsolver::g_sparse_destroy_mat_info

typedef rocsparse_status (*fp_rocsparse_scsrilu0_buffer_size)(rocsparse_handle handle,
                                                              rocsparse_int m,
                                                              rocsparse_int nnz,
                                                              const rocsparse_mat_descr descr,
                                                              const float* csr_val,
                                                              const rocsparse_int* csr_row_ptr,
                                                              const rocsparse_int* csr_col_ind,
                                                              rocsparse_mat_info info,
                                                              size_t* buffer_size);
extern fp_rocsparse_scsrilu0_buffer_size g_sparse_scsrilu0_buffer_size;
#define rocsparse_scsrilu0_buffer_size ::rocsolver::g_sparse_scsrilu0_buffer_size

typedef rocsparse_status (*fp_rocsparse_dcsrilu0_buffer_size)(rocsparse_handle handle,
                                                              rocsparse_int m,
                                                              rocsparse_int nnz,
                                                              const rocsparse_mat_descr descr,
                                                              const double* csr_val,
                                                              const rocsparse_int* csr_row_ptr,
                                                              const rocsparse_int* csr_col_ind,
                                                              rocsparse_mat_info info,
                                                              size_t* buffer_size);
extern fp_rocsparse_dcsrilu0_buffer_size g_sparse_dcsrilu0_buffer_size;
#define rocsparse_dcsrilu0_buffer_size ::rocsolver::g_sparse_dcsrilu0_buffer_size

typedef rocsparse_status (*fp_rocsparse_ccsrilu0_buffer_size)(rocsparse_handle handle,
                                                              rocsparse_int m,
                                                              rocsparse_int nnz,
                                                              const rocsparse_mat_descr descr,
                                                              const rocsparse_float_complex* csr_val,
                                                              const rocsparse_int* csr_row_ptr,
                                                              const rocsparse_int* csr_col_ind,
                                                              rocsparse_mat_info info,
                                                              size_t* buffer_size);
extern fp_rocsparse_ccsrilu0_buffer_size g_sparse_ccsrilu0_buffer_size;
#define rocsparse_ccsrilu0_buffer_size ::rocsolver::g_sparse_ccsrilu0_buffer_size

typedef rocsparse_status (*fp_rocsparse_zcsrilu0_buffer_size)(rocsparse_handle handle,
                                                              rocsparse_int m,
                                                              rocsparse_int nnz,
                                                              const rocsparse_mat_descr descr,
                                                              const rocsparse_double_complex* csr_val,
                                                              const rocsparse_int* csr_row_ptr,
                                                              const rocsparse_int* csr_col_ind,
                                                              rocsparse_mat_info info,
                                                              size_t* buffer_size);
extern fp_rocsparse_zcsrilu0_buffer_size g_sparse_zcsrilu0_buffer_size;
#define rocsparse_zcsrilu0_buffer_size ::rocsolver::g_sparse_zcsrilu0_buffer_size

typedef rocsparse_status (*fp_rocsparse_scsric0_buffer_size)(rocsparse_handle handle,
                                                             rocsparse_int m,
                                                             rocsparse_int nnz,
                                                             const rocsparse_mat_descr descr,
                                                             const float* csr_val,
                                                             const rocsparse_int* csr_row_ptr,
                                                             const rocsparse_int* csr_col_ind,
                                                             rocsparse_mat_info info,
                                                             size_t* buffer_size);
extern fp_rocsparse_scsric0_buffer_size g_sparse_scsric0_buffer_size;
#define rocsparse_scsric0_buffer_size ::rocsolver::g_sparse_scsric0_buffer_size

typedef rocsparse_status (*fp_rocsparse_dcsric0_buffer_size)(rocsparse_handle handle,
                                                             rocsparse_int m,
                                                             rocsparse_int nnz,
                                                             const rocsparse_mat_descr descr,
                                                             const double* csr_val,
                                                             const rocsparse_int* csr_row_ptr,
                                                             const rocsparse_int* csr_col_ind,
                                                             rocsparse_mat_info info,
                                                             size_t* buffer_size);
extern fp_rocsparse_dcsric0_buffer_size g_sparse_dcsric0_buffer_size;
#define rocsparse_dcsric0_buffer_size ::rocsolver::g_sparse_dcsric0_buffer_size

typedef rocsparse_status (*fp_rocsparse_ccsric0_buffer_size)(rocsparse_handle handle,
                                                             rocsparse_int m,
                                                             rocsparse_int nnz,
                                                             const rocsparse_mat_descr descr,
                                                             const rocsparse_float_complex* csr_val,
                                                             const rocsparse_int* csr_row_ptr,
                                                             const rocsparse_int* csr_col_ind,
                                                             rocsparse_mat_info info,
                                                             size_t* buffer_size);
extern fp_rocsparse_ccsric0_buffer_size g_sparse_ccsric0_buffer_size;
#define rocsparse_ccsric0_buffer_size ::rocsolver::g_sparse_ccsric0_buffer_size

typedef rocsparse_status (*fp_rocsparse_zcsric0_buffer_size)(rocsparse_handle handle,
                                                             rocsparse_int m,
                                                             rocsparse_int nnz,
                                                             const rocsparse_mat_descr descr,
                                                             const rocsparse_double_complex* csr_val,
                                                             const rocsparse_int* csr_row_ptr,
                                                             const rocsparse_int* csr_col_ind,
                                                             rocsparse_mat_info info,
                                                             size_t* buffer_size);
extern fp_rocsparse_zcsric0_buffer_size g_sparse_zcsric0_buffer_size;
#define rocsparse_zcsric0_buffer_size ::rocsolver::g_sparse_zcsric0_buffer_size

typedef rocsparse_status (*fp_rocsparse_scsric0_analysis)(rocsparse_handle handle,
                                                          rocsparse_int m,
                                                          rocsparse_int nnz,
                                                          const rocsparse_mat_descr descr,
                                                          const float* csr_val,
                                                          const rocsparse_int* csr_row_ptr,
                                                          const rocsparse_int* csr_col_ind,
                                                          rocsparse_mat_info info,
                                                          rocsparse_analysis_policy analysis,
                                                          rocsparse_solve_policy solve,
                                                          void* temp_buffer);
extern fp_rocsparse_scsric0_analysis g_sparse_scsric0_analysis;
#define rocsparse_scsric0_analysis ::rocsolver::g_sparse_scsric0_analysis

typedef rocsparse_status (*fp_rocsparse_dcsric0_analysis)(rocsparse_handle handle,
                                                          rocsparse_int m,
                                                          rocsparse_int nnz,
                                                          const rocsparse_mat_descr descr,
                                                          const double* csr_val,
                                                          const rocsparse_int* csr_row_ptr,
                                                          const rocsparse_int* csr_col_ind,
                                                          rocsparse_mat_info info,
                                                          rocsparse_analysis_policy analysis,
                                                          rocsparse_solve_policy solve,
                                                          void* temp_buffer);
extern fp_rocsparse_dcsric0_analysis g_sparse_dcsric0_analysis;
#define rocsparse_dcsric0_analysis ::rocsolver::g_sparse_dcsric0_analysis

typedef rocsparse_status (*fp_rocsparse_ccsric0_analysis)(rocsparse_handle handle,
                                                          rocsparse_int m,
                                                          rocsparse_int nnz,
                                                          const rocsparse_mat_descr descr,
                                                          const rocsparse_float_complex* csr_val,
                                                          const rocsparse_int* csr_row_ptr,
                                                          const rocsparse_int* csr_col_ind,
                                                          rocsparse_mat_info info,
                                                          rocsparse_analysis_policy analysis,
                                                          rocsparse_solve_policy solve,
                                                          void* temp_buffer);
extern fp_rocsparse_ccsric0_analysis g_sparse_ccsric0_analysis;
#define rocsparse_ccsric0_analysis ::rocsolver::g_sparse_ccsric0_analysis

typedef rocsparse_status (*fp_rocsparse_zcsric0_analysis)(rocsparse_handle handle,
                                                          rocsparse_int m,
                                                          rocsparse_int nnz,
                                                          const rocsparse_mat_descr descr,
                                                          const rocsparse_double_complex* csr_val,
                                                          const rocsparse_int* csr_row_ptr,
                                                          const rocsparse_int* csr_col_ind,
                                                          rocsparse_mat_info info,
                                                          rocsparse_analysis_policy analysis,
                                                          rocsparse_solve_policy solve,
                                                          void* temp_buffer);
extern fp_rocsparse_zcsric0_analysis g_sparse_zcsric0_analysis;
#define rocsparse_zcsric0_analysis ::rocsolver::g_sparse_zcsric0_analysis

typedef rocsparse_status (*fp_rocsparse_scsrsm_analysis)(rocsparse_handle handle,
                                                         rocsparse_operation trans_A,
                                                         rocsparse_operation trans_B,
                                                         rocsparse_int m,
                                                         rocsparse_int nrhs,
                                                         rocsparse_int nnz,
                                                         const float* alpha,
                                                         const rocsparse_mat_descr descr,
                                                         const float* csr_val,
                                                         const rocsparse_int* csr_row_ptr,
                                                         const rocsparse_int* csr_col_ind,
                                                         const float* B,
                                                         rocsparse_int ldb,
                                                         rocsparse_mat_info info,
                                                         rocsparse_analysis_policy analysis,
                                                         rocsparse_solve_policy solve,
                                                         void* temp_buffer);
extern fp_rocsparse_scsrsm_analysis g_sparse_scsrsm_analysis;
#define rocsparse_scsrsm_analysis ::rocsolver::g_sparse_scsrsm_analysis

typedef rocsparse_status (*fp_rocsparse_dcsrsm_analysis)(rocsparse_handle handle,
                                                         rocsparse_operation trans_A,
                                                         rocsparse_operation trans_B,
                                                         rocsparse_int m,
                                                         rocsparse_int nrhs,
                                                         rocsparse_int nnz,
                                                         const double* alpha,
                                                         const rocsparse_mat_descr descr,
                                                         const double* csr_val,
                                                         const rocsparse_int* csr_row_ptr,
                                                         const rocsparse_int* csr_col_ind,
                                                         const double* B,
                                                         rocsparse_int ldb,
                                                         rocsparse_mat_info info,
                                                         rocsparse_analysis_policy analysis,
                                                         rocsparse_solve_policy solve,
                                                         void* temp_buffer);
extern fp_rocsparse_dcsrsm_analysis g_sparse_dcsrsm_analysis;
#define rocsparse_dcsrsm_analysis ::rocsolver::g_sparse_dcsrsm_analysis

typedef rocsparse_status (*fp_rocsparse_ccsrsm_analysis)(rocsparse_handle handle,
                                                         rocsparse_operation trans_A,
                                                         rocsparse_operation trans_B,
                                                         rocsparse_int m,
                                                         rocsparse_int nrhs,
                                                         rocsparse_int nnz,
                                                         const rocsparse_float_complex* alpha,
                                                         const rocsparse_mat_descr descr,
                                                         const rocsparse_float_complex* csr_val,
                                                         const rocsparse_int* csr_row_ptr,
                                                         const rocsparse_int* csr_col_ind,
                                                         const rocsparse_float_complex* B,
                                                         rocsparse_int ldb,
                                                         rocsparse_mat_info info,
                                                         rocsparse_analysis_policy analysis,
                                                         rocsparse_solve_policy solve,
                                                         void* temp_buffer);
extern fp_rocsparse_ccsrsm_analysis g_sparse_ccsrsm_analysis;
#define rocsparse_ccsrsm_analysis ::rocsolver::g_sparse_ccsrsm_analysis

typedef rocsparse_status (*fp_rocsparse_zcsrsm_analysis)(rocsparse_handle handle,
                                                         rocsparse_operation trans_A,
                                                         rocsparse_operation trans_B,
                                                         rocsparse_int m,
                                                         rocsparse_int nrhs,
                                                         rocsparse_int nnz,
                                                         const rocsparse_double_complex* alpha,
                                                         const rocsparse_mat_descr descr,
                                                         const rocsparse_double_complex* csr_val,
                                                         const rocsparse_int* csr_row_ptr,
                                                         const rocsparse_int* csr_col_ind,
                                                         const rocsparse_double_complex* B,
                                                         rocsparse_int ldb,
                                                         rocsparse_mat_info info,
                                                         rocsparse_analysis_policy analysis,
                                                         rocsparse_solve_policy solve,
                                                         void* temp_buffer);
extern fp_rocsparse_zcsrsm_analysis g_sparse_zcsrsm_analysis;
#define rocsparse_zcsrsm_analysis ::rocsolver::g_sparse_zcsrsm_analysis

typedef rocsparse_status (*fp_rocsparse_scsrsm_buffer_size)(rocsparse_handle handle,
                                                            rocsparse_operation trans_A,
                                                            rocsparse_operation trans_B,
                                                            rocsparse_int m,
                                                            rocsparse_int nrhs,
                                                            rocsparse_int nnz,
                                                            const float* alpha,
                                                            const rocsparse_mat_descr descr,
                                                            const float* csr_val,
                                                            const rocsparse_int* csr_row_ptr,
                                                            const rocsparse_int* csr_col_ind,
                                                            const float* B,
                                                            rocsparse_int ldb,
                                                            rocsparse_mat_info info,
                                                            rocsparse_solve_policy policy,
                                                            size_t* buffer_size);
extern fp_rocsparse_scsrsm_buffer_size g_sparse_scsrsm_buffer_size;
#define rocsparse_scsrsm_buffer_size ::rocsolver::g_sparse_scsrsm_buffer_size

typedef rocsparse_status (*fp_rocsparse_dcsrsm_buffer_size)(rocsparse_handle handle,
                                                            rocsparse_operation trans_A,
                                                            rocsparse_operation trans_B,
                                                            rocsparse_int m,
                                                            rocsparse_int nrhs,
                                                            rocsparse_int nnz,
                                                            const double* alpha,
                                                            const rocsparse_mat_descr descr,
                                                            const double* csr_val,
                                                            const rocsparse_int* csr_row_ptr,
                                                            const rocsparse_int* csr_col_ind,
                                                            const double* B,
                                                            rocsparse_int ldb,
                                                            rocsparse_mat_info info,
                                                            rocsparse_solve_policy policy,
                                                            size_t* buffer_size);
extern fp_rocsparse_dcsrsm_buffer_size g_sparse_dcsrsm_buffer_size;
#define rocsparse_dcsrsm_buffer_size ::rocsolver::g_sparse_dcsrsm_buffer_size

typedef rocsparse_status (*fp_rocsparse_ccsrsm_buffer_size)(rocsparse_handle handle,
                                                            rocsparse_operation trans_A,
                                                            rocsparse_operation trans_B,
                                                            rocsparse_int m,
                                                            rocsparse_int nrhs,
                                                            rocsparse_int nnz,
                                                            const rocsparse_float_complex* alpha,
                                                            const rocsparse_mat_descr descr,
                                                            const rocsparse_float_complex* csr_val,
                                                            const rocsparse_int* csr_row_ptr,
                                                            const rocsparse_int* csr_col_ind,
                                                            const rocsparse_float_complex* B,
                                                            rocsparse_int ldb,
                                                            rocsparse_mat_info info,
                                                            rocsparse_solve_policy policy,
                                                            size_t* buffer_size);
extern fp_rocsparse_ccsrsm_buffer_size g_sparse_ccsrsm_buffer_size;
#define rocsparse_ccsrsm_buffer_size ::rocsolver::g_sparse_ccsrsm_buffer_size

typedef rocsparse_status (*fp_rocsparse_zcsrsm_buffer_size)(rocsparse_handle handle,
                                                            rocsparse_operation trans_A,
                                                            rocsparse_operation trans_B,
                                                            rocsparse_int m,
                                                            rocsparse_int nrhs,
                                                            rocsparse_int nnz,
                                                            const rocsparse_double_complex* alpha,
                                                            const rocsparse_mat_descr descr,
                                                            const rocsparse_double_complex* csr_val,
                                                            const rocsparse_int* csr_row_ptr,
                                                            const rocsparse_int* csr_col_ind,
                                                            const rocsparse_double_complex* B,
                                                            rocsparse_int ldb,
                                                            rocsparse_mat_info info,
                                                            rocsparse_solve_policy policy,
                                                            size_t* buffer_size);
extern fp_rocsparse_zcsrsm_buffer_size g_sparse_zcsrsm_buffer_size;
#define rocsparse_zcsrsm_buffer_size ::rocsolver::g_sparse_zcsrsm_buffer_size

typedef rocsparse_status (*fp_rocsparse_scsrsm_solve)(rocsparse_handle handle,
                                                      rocsparse_operation trans_A,
                                                      rocsparse_operation trans_B,
                                                      rocsparse_int m,
                                                      rocsparse_int nrhs,
                                                      rocsparse_int nnz,
                                                      const float* alpha,
                                                      const rocsparse_mat_descr descr,
                                                      const float* csr_val,
                                                      const rocsparse_int* csr_row_ptr,
                                                      const rocsparse_int* csr_col_ind,
                                                      float* B,
                                                      rocsparse_int ldb,
                                                      rocsparse_mat_info info,
                                                      rocsparse_solve_policy policy,
                                                      void* temp_buffer);
extern fp_rocsparse_scsrsm_solve g_sparse_scsrsm_solve;
#define rocsparse_scsrsm_solve ::rocsolver::g_sparse_scsrsm_solve

typedef rocsparse_status (*fp_rocsparse_dcsrsm_solve)(rocsparse_handle handle,
                                                      rocsparse_operation trans_A,
                                                      rocsparse_operation trans_B,
                                                      rocsparse_int m,
                                                      rocsparse_int nrhs,
                                                      rocsparse_int nnz,
                                                      const double* alpha,
                                                      const rocsparse_mat_descr descr,
                                                      const double* csr_val,
                                                      const rocsparse_int* csr_row_ptr,
                                                      const rocsparse_int* csr_col_ind,
                                                      double* B,
                                                      rocsparse_int ldb,
                                                      rocsparse_mat_info info,
                                                      rocsparse_solve_policy policy,
                                                      void* temp_buffer);
extern fp_rocsparse_dcsrsm_solve g_sparse_dcsrsm_solve;
#define rocsparse_dcsrsm_solve ::rocsolver::g_sparse_dcsrsm_solve

typedef rocsparse_status (*fp_rocsparse_ccsrsm_solve)(rocsparse_handle handle,
                                                      rocsparse_operation trans_A,
                                                      rocsparse_operation trans_B,
                                                      rocsparse_int m,
                                                      rocsparse_int nrhs,
                                                      rocsparse_int nnz,
                                                      const rocsparse_float_complex* alpha,
                                                      const rocsparse_mat_descr descr,
                                                      const rocsparse_float_complex* csr_val,
                                                      const rocsparse_int* csr_row_ptr,
                                                      const rocsparse_int* csr_col_ind,
                                                      rocsparse_float_complex* B,
                                                      rocsparse_int ldb,
                                                      rocsparse_mat_info info,
                                                      rocsparse_solve_policy policy,
                                                      void* temp_buffer);
extern fp_rocsparse_ccsrsm_solve g_sparse_ccsrsm_solve;
#define rocsparse_ccsrsm_solve ::rocsolver::g_sparse_ccsrsm_solve

typedef rocsparse_status (*fp_rocsparse_zcsrsm_solve)(rocsparse_handle handle,
                                                      rocsparse_operation trans_A,
                                                      rocsparse_operation trans_B,
                                                      rocsparse_int m,
                                                      rocsparse_int nrhs,
                                                      rocsparse_int nnz,
                                                      const rocsparse_double_complex* alpha,
                                                      const rocsparse_mat_descr descr,
                                                      const rocsparse_double_complex* csr_val,
                                                      const rocsparse_int* csr_row_ptr,
                                                      const rocsparse_int* csr_col_ind,
                                                      rocsparse_double_complex* B,
                                                      rocsparse_int ldb,
                                                      rocsparse_mat_info info,
                                                      rocsparse_solve_policy policy,
                                                      void* temp_buffer);
extern fp_rocsparse_zcsrsm_solve g_sparse_zcsrsm_solve;
#define rocsparse_zcsrsm_solve ::rocsolver::g_sparse_zcsrsm_solve

typedef rocsparse_status (*fp_rocsparse_scsrilu0_analysis)(rocsparse_handle handle,
                                                           rocsparse_int m,
                                                           rocsparse_int nnz,
                                                           const rocsparse_mat_descr descr,
                                                           const float* csr_val,
                                                           const rocsparse_int* csr_row_ptr,
                                                           const rocsparse_int* csr_col_ind,
                                                           rocsparse_mat_info info,
                                                           rocsparse_analysis_policy analysis,
                                                           rocsparse_solve_policy solve,
                                                           void* temp_buffer);
extern fp_rocsparse_scsrilu0_analysis g_sparse_scsrilu0_analysis;
#define rocsparse_scsrilu0_analysis ::rocsolver::g_sparse_scsrilu0_analysis

typedef rocsparse_status (*fp_rocsparse_dcsrilu0_analysis)(rocsparse_handle handle,
                                                           rocsparse_int m,
                                                           rocsparse_int nnz,
                                                           const rocsparse_mat_descr descr,
                                                           const double* csr_val,
                                                           const rocsparse_int* csr_row_ptr,
                                                           const rocsparse_int* csr_col_ind,
                                                           rocsparse_mat_info info,
                                                           rocsparse_analysis_policy analysis,
                                                           rocsparse_solve_policy solve,
                                                           void* temp_buffer);
extern fp_rocsparse_dcsrilu0_analysis g_sparse_dcsrilu0_analysis;
#define rocsparse_dcsrilu0_analysis ::rocsolver::g_sparse_dcsrilu0_analysis

typedef rocsparse_status (*fp_rocsparse_ccsrilu0_analysis)(rocsparse_handle handle,
                                                           rocsparse_int m,
                                                           rocsparse_int nnz,
                                                           const rocsparse_mat_descr descr,
                                                           const rocsparse_float_complex* csr_val,
                                                           const rocsparse_int* csr_row_ptr,
                                                           const rocsparse_int* csr_col_ind,
                                                           rocsparse_mat_info info,
                                                           rocsparse_analysis_policy analysis,
                                                           rocsparse_solve_policy solve,
                                                           void* temp_buffer);
extern fp_rocsparse_ccsrilu0_analysis g_sparse_ccsrilu0_analysis;
#define rocsparse_ccsrilu0_analysis ::rocsolver::g_sparse_ccsrilu0_analysis

typedef rocsparse_status (*fp_rocsparse_zcsrilu0_analysis)(rocsparse_handle handle,
                                                           rocsparse_int m,
                                                           rocsparse_int nnz,
                                                           const rocsparse_mat_descr descr,
                                                           const rocsparse_double_complex* csr_val,
                                                           const rocsparse_int* csr_row_ptr,
                                                           const rocsparse_int* csr_col_ind,
                                                           rocsparse_mat_info info,
                                                           rocsparse_analysis_policy analysis,
                                                           rocsparse_solve_policy solve,
                                                           void* temp_buffer);
extern fp_rocsparse_zcsrilu0_analysis g_sparse_zcsrilu0_analysis;
#define rocsparse_zcsrilu0_analysis ::rocsolver::g_sparse_zcsrilu0_analysis

typedef rocsparse_status (*fp_rocsparse_scsrilu0)(rocsparse_handle handle,
                                                  rocsparse_int m,
                                                  rocsparse_int nnz,
                                                  const rocsparse_mat_descr descr,
                                                  float* csr_val,
                                                  const rocsparse_int* csr_row_ptr,
                                                  const rocsparse_int* csr_col_ind,
                                                  rocsparse_mat_info info,
                                                  rocsparse_solve_policy policy,
                                                  void* temp_buffer);
extern fp_rocsparse_scsrilu0 g_sparse_scsrilu0;
#define rocsparse_scsrilu0 ::rocsolver::g_sparse_scsrilu0

typedef rocsparse_status (*fp_rocsparse_dcsrilu0)(rocsparse_handle handle,
                                                  rocsparse_int m,
                                                  rocsparse_int nnz,
                                                  const rocsparse_mat_descr descr,
                                                  double* csr_val,
                                                  const rocsparse_int* csr_row_ptr,
                                                  const rocsparse_int* csr_col_ind,
                                                  rocsparse_mat_info info,
                                                  rocsparse_solve_policy policy,
                                                  void* temp_buffer);
extern fp_rocsparse_dcsrilu0 g_sparse_dcsrilu0;
#define rocsparse_dcsrilu0 ::rocsolver::g_sparse_dcsrilu0

typedef rocsparse_status (*fp_rocsparse_ccsrilu0)(rocsparse_handle handle,
                                                  rocsparse_int m,
                                                  rocsparse_int nnz,
                                                  const rocsparse_mat_descr descr,
                                                  rocsparse_float_complex* csr_val,
                                                  const rocsparse_int* csr_row_ptr,
                                                  const rocsparse_int* csr_col_ind,
                                                  rocsparse_mat_info info,
                                                  rocsparse_solve_policy policy,
                                                  void* temp_buffer);
extern fp_rocsparse_ccsrilu0 g_sparse_ccsrilu0;
#define rocsparse_ccsrilu0 ::rocsolver::g_sparse_ccsrilu0

typedef rocsparse_status (*fp_rocsparse_zcsrilu0)(rocsparse_handle handle,
                                                  rocsparse_int m,
                                                  rocsparse_int nnz,
                                                  const rocsparse_mat_descr descr,
                                                  rocsparse_double_complex* csr_val,
                                                  const rocsparse_int* csr_row_ptr,
                                                  const rocsparse_int* csr_col_ind,
                                                  rocsparse_mat_info info,
                                                  rocsparse_solve_policy policy,
                                                  void* temp_buffer);
extern fp_rocsparse_zcsrilu0 g_sparse_zcsrilu0;
#define rocsparse_zcsrilu0 ::rocsolver::g_sparse_zcsrilu0

typedef rocsparse_status (*fp_rocsparse_scsric0)(rocsparse_handle handle,
                                                 rocsparse_int m,
                                                 rocsparse_int nnz,
                                                 const rocsparse_mat_descr descr,
                                                 float* csr_val,
                                                 const rocsparse_int* csr_row_ptr,
                                                 const rocsparse_int* csr_col_ind,
                                                 rocsparse_mat_info info,
                                                 rocsparse_solve_policy policy,
                                                 void* temp_buffer);
extern fp_rocsparse_scsric0 g_sparse_scsric0;
#define rocsparse_scsric0 ::rocsolver::g_sparse_scsric0

typedef rocsparse_status (*fp_rocsparse_dcsric0)(rocsparse_handle handle,
                                                 rocsparse_int m,
                                                 rocsparse_int nnz,
                                                 const rocsparse_mat_descr descr,
                                                 double* csr_val,
                                                 const rocsparse_int* csr_row_ptr,
                                                 const rocsparse_int* csr_col_ind,
                                                 rocsparse_mat_info info,
                                                 rocsparse_solve_policy policy,
                                                 void* temp_buffer);
extern fp_rocsparse_dcsric0 g_sparse_dcsric0;
#define rocsparse_dcsric0 ::rocsolver::g_sparse_dcsric0

typedef rocsparse_status (*fp_rocsparse_ccsric0)(rocsparse_handle handle,
                                                 rocsparse_int m,
                                                 rocsparse_int nnz,
                                                 const rocsparse_mat_descr descr,
                                                 rocsparse_float_complex* csr_val,
                                                 const rocsparse_int* csr_row_ptr,
                                                 const rocsparse_int* csr_col_ind,
                                                 rocsparse_mat_info info,
                                                 rocsparse_solve_policy policy,
                                                 void* temp_buffer);
extern fp_rocsparse_ccsric0 g_sparse_ccsric0;
#define rocsparse_ccsric0 ::rocsolver::g_sparse_ccsric0

typedef rocsparse_status (*fp_rocsparse_zcsric0)(rocsparse_handle handle,
                                                 rocsparse_int m,
                                                 rocsparse_int nnz,
                                                 const rocsparse_mat_descr descr,
                                                 rocsparse_double_complex* csr_val,
                                                 const rocsparse_int* csr_row_ptr,
                                                 const rocsparse_int* csr_col_ind,
                                                 rocsparse_mat_info info,
                                                 rocsparse_solve_policy policy,
                                                 void* temp_buffer);
extern fp_rocsparse_zcsric0 g_sparse_zcsric0;
#define rocsparse_zcsric0 ::rocsolver::g_sparse_zcsric0

ROCSOLVER_END_NAMESPACE

#endif /* HAVE_ROCSPARSE */

ROCSOLVER_BEGIN_NAMESPACE

// csrilu0 buffer
inline rocsparse_status rocsparseCall_csrilu0_buffer_size(rocsparse_handle sphandle,
                                                          rocblas_int n,
                                                          rocblas_int nnz,
                                                          rocsparse_mat_descr descr,
                                                          float* val,
                                                          rocblas_int* ptr,
                                                          rocblas_int* ind,
                                                          rocsparse_mat_info info,
                                                          size_t* size)
{
    return rocsparse_scsrilu0_buffer_size(sphandle, n, nnz, descr, val, ptr, ind, info, size);
}

inline rocsparse_status rocsparseCall_csrilu0_buffer_size(rocsparse_handle sphandle,
                                                          rocblas_int n,
                                                          rocblas_int nnz,
                                                          rocsparse_mat_descr descr,
                                                          double* val,
                                                          rocblas_int* ptr,
                                                          rocblas_int* ind,
                                                          rocsparse_mat_info info,
                                                          size_t* size)
{
    return rocsparse_dcsrilu0_buffer_size(sphandle, n, nnz, descr, val, ptr, ind, info, size);
}

// csric0 buffer
inline rocsparse_status rocsparseCall_csric0_buffer_size(rocsparse_handle sphandle,
                                                         rocblas_int n,
                                                         rocblas_int nnz,
                                                         rocsparse_mat_descr descr,
                                                         float* val,
                                                         rocblas_int* ptr,
                                                         rocblas_int* ind,
                                                         rocsparse_mat_info info,
                                                         size_t* size)
{
    return rocsparse_scsric0_buffer_size(sphandle, n, nnz, descr, val, ptr, ind, info, size);
}

inline rocsparse_status rocsparseCall_csric0_buffer_size(rocsparse_handle sphandle,
                                                         rocblas_int n,
                                                         rocblas_int nnz,
                                                         rocsparse_mat_descr descr,
                                                         double* val,
                                                         rocblas_int* ptr,
                                                         rocblas_int* ind,
                                                         rocsparse_mat_info info,
                                                         size_t* size)
{
    return rocsparse_dcsric0_buffer_size(sphandle, n, nnz, descr, val, ptr, ind, info, size);
}

// csrilu0 analysis
inline rocsparse_status rocsparseCall_csrilu0_analysis(rocsparse_handle sphandle,
                                                       rocblas_int n,
                                                       rocblas_int nnz,
                                                       rocsparse_mat_descr descr,
                                                       float* val,
                                                       rocblas_int* ptr,
                                                       rocblas_int* ind,
                                                       rocsparse_mat_info info,
                                                       rocsparse_analysis_policy analysis,
                                                       rocsparse_solve_policy solve,
                                                       void* buffer)
{
    return rocsparse_scsrilu0_analysis(sphandle, n, nnz, descr, val, ptr, ind, info, analysis,
                                       solve, buffer);
}

inline rocsparse_status rocsparseCall_csrilu0_analysis(rocsparse_handle sphandle,
                                                       rocblas_int n,
                                                       rocblas_int nnz,
                                                       rocsparse_mat_descr descr,
                                                       double* val,
                                                       rocblas_int* ptr,
                                                       rocblas_int* ind,
                                                       rocsparse_mat_info info,
                                                       rocsparse_analysis_policy analysis,
                                                       rocsparse_solve_policy solve,
                                                       void* buffer)
{
    return rocsparse_dcsrilu0_analysis(sphandle, n, nnz, descr, val, ptr, ind, info, analysis,
                                       solve, buffer);
}

// csric0 analysis
inline rocsparse_status rocsparseCall_csric0_analysis(rocsparse_handle sphandle,
                                                      rocblas_int n,
                                                      rocblas_int nnz,
                                                      rocsparse_mat_descr descr,
                                                      float* val,
                                                      rocblas_int* ptr,
                                                      rocblas_int* ind,
                                                      rocsparse_mat_info info,
                                                      rocsparse_analysis_policy analysis,
                                                      rocsparse_solve_policy solve,
                                                      void* buffer)
{
    return rocsparse_scsric0_analysis(sphandle, n, nnz, descr, val, ptr, ind, info, analysis, solve,
                                      buffer);
}

inline rocsparse_status rocsparseCall_csric0_analysis(rocsparse_handle sphandle,
                                                      rocblas_int n,
                                                      rocblas_int nnz,
                                                      rocsparse_mat_descr descr,
                                                      double* val,
                                                      rocblas_int* ptr,
                                                      rocblas_int* ind,
                                                      rocsparse_mat_info info,
                                                      rocsparse_analysis_policy analysis,
                                                      rocsparse_solve_policy solve,
                                                      void* buffer)
{
    return rocsparse_dcsric0_analysis(sphandle, n, nnz, descr, val, ptr, ind, info, analysis, solve,
                                      buffer);
}

// csrilu0
inline rocsparse_status rocsparseCall_csrilu0(rocsparse_handle sphandle,
                                              rocblas_int n,
                                              rocblas_int nnz,
                                              rocsparse_mat_descr descr,
                                              float* val,
                                              rocblas_int* ptr,
                                              rocblas_int* ind,
                                              rocsparse_mat_info info,
                                              rocsparse_solve_policy solve,
                                              void* buffer)
{
    return rocsparse_scsrilu0(sphandle, n, nnz, descr, val, ptr, ind, info, solve, buffer);
}

inline rocsparse_status rocsparseCall_csrilu0(rocsparse_handle sphandle,
                                              rocblas_int n,
                                              rocblas_int nnz,
                                              rocsparse_mat_descr descr,
                                              double* val,
                                              rocblas_int* ptr,
                                              rocblas_int* ind,
                                              rocsparse_mat_info info,
                                              rocsparse_solve_policy solve,
                                              void* buffer)
{
    return rocsparse_dcsrilu0(sphandle, n, nnz, descr, val, ptr, ind, info, solve, buffer);
}

// csric0
inline rocsparse_status rocsparseCall_csric0(rocsparse_handle sphandle,
                                             rocblas_int n,
                                             rocblas_int nnz,
                                             rocsparse_mat_descr descr,
                                             float* val,
                                             rocblas_int* ptr,
                                             rocblas_int* ind,
                                             rocsparse_mat_info info,
                                             rocsparse_solve_policy solve,
                                             void* buffer)
{
    return rocsparse_scsric0(sphandle, n, nnz, descr, val, ptr, ind, info, solve, buffer);
}

inline rocsparse_status rocsparseCall_csric0(rocsparse_handle sphandle,
                                             rocblas_int n,
                                             rocblas_int nnz,
                                             rocsparse_mat_descr descr,
                                             double* val,
                                             rocblas_int* ptr,
                                             rocblas_int* ind,
                                             rocsparse_mat_info info,
                                             rocsparse_solve_policy solve,
                                             void* buffer)
{
    return rocsparse_dcsric0(sphandle, n, nnz, descr, val, ptr, ind, info, solve, buffer);
}

// csrsm buffer
inline rocsparse_status rocsparseCall_csrsm_buffer_size(rocsparse_handle sphandle,
                                                        rocsparse_operation transA,
                                                        rocsparse_operation transB,
                                                        rocblas_int n,
                                                        rocblas_int nrhs,
                                                        rocblas_int nnz,
                                                        float* alpha,
                                                        rocsparse_mat_descr descr,
                                                        float* val,
                                                        rocblas_int* ptr,
                                                        rocblas_int* ind,
                                                        float* B,
                                                        rocblas_int ldb,
                                                        rocsparse_mat_info info,
                                                        rocsparse_solve_policy solve,
                                                        size_t* size)
{
    return rocsparse_scsrsm_buffer_size(sphandle, transA, transB, n, nrhs, nnz, alpha, descr, val,
                                        ptr, ind, B, ldb, info, solve, size);
}

inline rocsparse_status rocsparseCall_csrsm_buffer_size(rocsparse_handle sphandle,
                                                        rocsparse_operation transA,
                                                        rocsparse_operation transB,
                                                        rocblas_int n,
                                                        rocblas_int nrhs,
                                                        rocblas_int nnz,
                                                        double* alpha,
                                                        rocsparse_mat_descr descr,
                                                        double* val,
                                                        rocblas_int* ptr,
                                                        rocblas_int* ind,
                                                        double* B,
                                                        rocblas_int ldb,
                                                        rocsparse_mat_info info,
                                                        rocsparse_solve_policy solve,
                                                        size_t* size)
{
    return rocsparse_dcsrsm_buffer_size(sphandle, transA, transB, n, nrhs, nnz, alpha, descr, val,
                                        ptr, ind, B, ldb, info, solve, size);
}

// csrsm analysis
inline rocsparse_status rocsparseCall_csrsm_analysis(rocsparse_handle sphandle,
                                                     rocsparse_operation transA,
                                                     rocsparse_operation transB,
                                                     rocblas_int n,
                                                     rocblas_int nrhs,
                                                     rocblas_int nnz,
                                                     float* alpha,
                                                     rocsparse_mat_descr descr,
                                                     float* val,
                                                     rocblas_int* ptr,
                                                     rocblas_int* ind,
                                                     float* B,
                                                     rocblas_int ldb,
                                                     rocsparse_mat_info info,
                                                     rocsparse_analysis_policy analysis,
                                                     rocsparse_solve_policy solve,
                                                     void* buffer)
{
    return rocsparse_scsrsm_analysis(sphandle, transA, transB, n, nrhs, nnz, alpha, descr, val, ptr,
                                     ind, B, ldb, info, analysis, solve, buffer);
}

inline rocsparse_status rocsparseCall_csrsm_analysis(rocsparse_handle sphandle,
                                                     rocsparse_operation transA,
                                                     rocsparse_operation transB,
                                                     rocblas_int n,
                                                     rocblas_int nrhs,
                                                     rocblas_int nnz,
                                                     double* alpha,
                                                     rocsparse_mat_descr descr,
                                                     double* val,
                                                     rocblas_int* ptr,
                                                     rocblas_int* ind,
                                                     double* B,
                                                     rocblas_int ldb,
                                                     rocsparse_mat_info info,
                                                     rocsparse_analysis_policy analysis,
                                                     rocsparse_solve_policy solve,
                                                     void* buffer)
{
    return rocsparse_dcsrsm_analysis(sphandle, transA, transB, n, nrhs, nnz, alpha, descr, val, ptr,
                                     ind, B, ldb, info, analysis, solve, buffer);
}

//csrsm solve
inline rocsparse_status rocsparseCall_csrsm_solve(rocsparse_handle sphandle,
                                                  rocsparse_operation transA,
                                                  rocsparse_operation transB,
                                                  rocblas_int n,
                                                  rocblas_int nrhs,
                                                  rocblas_int nnz,
                                                  float* alpha,
                                                  rocsparse_mat_descr descr,
                                                  float* val,
                                                  rocblas_int* ptr,
                                                  rocblas_int* ind,
                                                  float* B,
                                                  rocblas_int ldb,
                                                  rocsparse_mat_info info,
                                                  rocsparse_solve_policy solve,
                                                  void* buffer)
{
    return rocsparse_scsrsm_solve(sphandle, transA, transB, n, nrhs, nnz, alpha, descr, val, ptr,
                                  ind, B, ldb, info, solve, buffer);
}

inline rocsparse_status rocsparseCall_csrsm_solve(rocsparse_handle sphandle,
                                                  rocsparse_operation transA,
                                                  rocsparse_operation transB,
                                                  rocblas_int n,
                                                  rocblas_int nrhs,
                                                  rocblas_int nnz,
                                                  double* alpha,
                                                  rocsparse_mat_descr descr,
                                                  double* val,
                                                  rocblas_int* ptr,
                                                  rocblas_int* ind,
                                                  double* B,
                                                  rocblas_int ldb,
                                                  rocsparse_mat_info info,
                                                  rocsparse_solve_policy solve,
                                                  void* buffer)
{
    return rocsparse_dcsrsm_solve(sphandle, transA, transB, n, nrhs, nnz, alpha, descr, val, ptr,
                                  ind, B, ldb, info, solve, buffer);
}

ROCSOLVER_END_NAMESPACE
