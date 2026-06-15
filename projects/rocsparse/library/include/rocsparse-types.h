/* ************************************************************************
 * Copyright (C) 2018-2026 Advanced Micro Devices, Inc. All rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

/*! \file
 * \brief rocsparse-types.h defines data types used by rocsparse
 */

#ifndef ROCSPARSE_TYPES_H
#define ROCSPARSE_TYPES_H

#include "rocsparse-complex-types.h"
#include "rocsparse_bfloat16.h"

#include <float.h>
#include <stddef.h>
#include <stdint.h>

/// \cond DO_NOT_DOCUMENT
#define ROCSPARSE_KERNEL_W(MAX_THREADS_PER_BLOCK, MIN_WARPS_PER_EXECUTION_UNIT) \
    __launch_bounds__(MAX_THREADS_PER_BLOCK, MIN_WARPS_PER_EXECUTION_UNIT) static __global__
#define ROCSPARSE_KERNEL(MAX_THREADS_PER_BLOCK) \
    __launch_bounds__(MAX_THREADS_PER_BLOCK) static __global__
#define ROCSPARSE_DEVICE_ILF static __device__ __forceinline__
/// \endcond

/*! \ingroup types_module
 *  \brief Specifies the rocSPARSE integer type (defaults to \p int32_t).
 *
 *  \note When rocSPARSE is built with \p rocsparse_ILP64, \ref rocsparse_int has a typedef to \p int64_t.
 */
#if defined(rocsparse_ILP64)
typedef int64_t rocsparse_int;
#else
typedef int32_t rocsparse_int;
#endif

/// \cond DO_NOT_DOCUMENT
/* Forward declaration of hipStream_t */
typedef struct ihipStream_t* hipStream_t;
/// \endcond

/*! \ingroup types_module
 *  \brief Handle to the rocSPARSE library context queue.
 *
 *  \details
 *  The rocSPARSE handle is a structure holding the rocSPARSE library context. It must
 *  be initialized using rocsparse_create_handle(), and the returned handle must be
 *  passed to all subsequent library function calls. It should be destroyed at the end
 *  using rocsparse_destroy_handle().
 */
typedef struct _rocsparse_handle* rocsparse_handle;

/*! \ingroup types_module
 *  \brief Descriptor of the error.
 *
 *  \details
 *  The rocSPARSE error descriptor is a structure holding the information related to an error
 *  that occurred during the execution of a rocSPARSE routine.
 *  It should be destroyed using rocsparse_destroy_error().
 */
typedef struct _rocsparse_error* rocsparse_error;

/*! \ingroup types_module
 *  \brief Descriptor of the matrix.
 *
 *  \details
 *  The rocSPARSE matrix descriptor is a structure holding all properties of a matrix.
 *  It must be initialized using rocsparse_create_mat_descr(), and the returned
 *  descriptor must be passed to all subsequent library calls that involve the matrix.
 *  It should be destroyed at the end using rocsparse_destroy_mat_descr().
 */
typedef struct _rocsparse_mat_descr* rocsparse_mat_descr;

/*! \ingroup types_module
 *  \brief HYB matrix storage format.
 *
 *  \details
 *  The rocSPARSE HYB matrix structure holds the HYB matrix. It must be initialized using
 *  rocsparse_create_hyb_mat(), and the returned HYB matrix must be passed to all
 *  subsequent library calls that involve the matrix. It should be destroyed at the end
 *  using rocsparse_destroy_hyb_mat().
 */
typedef struct _rocsparse_hyb_mat* rocsparse_hyb_mat;

/*! \ingroup types_module
 *  \brief Info structure to hold all matrix meta data.
 *
 *  \details
 *  The rocSPARSE matrix info is a structure holding all matrix information that is
 *  gathered during analysis routines. It must be initialized using
 *  rocsparse_create_mat_info(), and the returned info structure must be passed to all
 *  subsequent library calls that require additional matrix information. It should be
 *  destroyed at the end using rocsparse_destroy_mat_info().
 */
typedef struct _rocsparse_mat_info* rocsparse_mat_info;

// Generic API

/*! \ingroup types_module
 *  \brief Generic API descriptor of the sparse vector.
 *
 *  \details
 *  The rocSPARSE sparse vector descriptor is a structure holding all properties of a sparse vector.
 *  It must be initialized using rocsparse_create_spvec_descr(), and the returned
 *  descriptor must be passed to all subsequent generic API library calls that involve the sparse vector.
 *  It should be destroyed at the end using rocsparse_destroy_spvec_descr().
 */
typedef struct _rocsparse_spvec_descr* rocsparse_spvec_descr;

/*! \ingroup types_module
 *  \brief Generic API descriptor of the sparse vector.
 *
 *  \details
 *  The rocSPARSE constant sparse vector descriptor is a structure holding all properties of a sparse vector.
 *  It must be initialized using \ref rocsparse_create_const_spvec_descr(), and the returned
 *  descriptor must be passed to all subsequent generic API library calls that involve the sparse vector.
 *  It should be destroyed at the end using rocsparse_destroy_spvec_descr().
 */
typedef struct _rocsparse_spvec_descr const* rocsparse_const_spvec_descr;

/*! \ingroup types_module
 *  \brief Generic API descriptor of the sparse matrix.
 *
 *  \details
 *  The rocSPARSE sparse matrix descriptor is a structure holding all properties of a sparse matrix.
 *  It must be initialized using rocsparse_create_coo_descr(), rocsparse_create_coo_aos_descr(),
 *  rocsparse_create_bsr_descr(), rocsparse_create_csr_descr(), rocsparse_create_csc_descr(),
 *  rocsparse_create_ell_descr(), or rocsparse_create_bell_descr(), and the returned
 *  descriptor must be passed to all subsequent generic API library calls that involve the sparse matrix.
 *  It should be destroyed at the end using rocsparse_destroy_spmat_descr().
 */
typedef struct _rocsparse_spmat_descr* rocsparse_spmat_descr;

/*! \ingroup types_module
 *  \brief Generic API descriptor of the sparse matrix.
 *
 *  \details
 *  The rocSPARSE constant sparse matrix descriptor is a structure holding all properties of a sparse matrix.
 *  It must be initialized using rocsparse_create_const_coo_descr(),
 *  rocsparse_create_const_csr_descr(), rocsparse_create_const_csc_descr(),
 *  rocsparse_create_const_bsr_descr(), or rocsparse_create_const_bell_descr(), and the returned
 *  descriptor must be passed to all subsequent generic API library calls that involve the sparse matrix.
 *  It should be destroyed at the end using rocsparse_destroy_spmat_descr().
 */
typedef struct _rocsparse_spmat_descr const* rocsparse_const_spmat_descr;

/*! \ingroup types_module
 *  \brief Generic API descriptor of the dense vector.
 *
 *  \details
 *  The rocSPARSE dense vector descriptor is a structure holding all properties of a dense vector.
 *  It must be initialized using rocsparse_create_dnvec_descr(), and the returned
 *  descriptor must be passed to all subsequent generic API library calls that involve the dense vector.
 *  It should be destroyed at the end using rocsparse_destroy_dnvec_descr().
 */
typedef struct _rocsparse_dnvec_descr* rocsparse_dnvec_descr;

/*! \ingroup types_module
 *  \brief Generic API descriptor of the dense vector.
 *
 *  \details
 *  The rocSPARSE constant dense vector descriptor is a structure holding all properties of a dense vector.
 *  It must be initialized using rocsparse_create_const_dnvec_descr(), and the returned
 *  descriptor must be passed to all subsequent generic API library calls that involve the dense vector.
 *  It should be destroyed at the end using rocsparse_destroy_dnvec_descr().
 */
typedef struct _rocsparse_dnvec_descr const* rocsparse_const_dnvec_descr;

/*! \ingroup types_module
 *  \brief Generic API descriptor of the dense matrix.
 *
 *  \details
 *  The rocSPARSE dense matrix descriptor is a structure holding all properties of a dense matrix.
 *  It must be initialized using rocsparse_create_dnmat_descr(), and the returned
 *  descriptor must be passed to all subsequent generic API library calls that involve the dense matrix.
 *  It should be destroyed at the end using rocsparse_destroy_dnmat_descr().
 */
typedef struct _rocsparse_dnmat_descr* rocsparse_dnmat_descr;

/*! \ingroup types_module
 *  \brief Generic API descriptor of the dense matrix.
 *
 *  \details
 *  The rocSPARSE constant dense matrix descriptor is a structure holding all properties of a dense matrix.
 *  It must be initialized using rocsparse_create_const_dnmat_descr(), and the returned
 *  descriptor must be passed to all subsequent generic API library calls that involve the dense matrix.
 *  It should be destroyed at the end using rocsparse_destroy_dnmat_descr().
 */
typedef struct _rocsparse_dnmat_descr const* rocsparse_const_dnmat_descr;

/*! \ingroup types_module
 *  \brief Coloring info structure to hold data gathered during analysis and later used in
 *  rocSPARSE sparse matrix coloring routines.
 *
 *  \details
 *  The rocSPARSE color info is a structure holding coloring data that is
 *  gathered during analysis routines. It must be initialized using
 *  rocsparse_create_color_info(), and the returned info structure must be passed to all
 *  subsequent library calls that require coloring information. It should be
 *  destroyed at the end using rocsparse_destroy_color_info().
 */
typedef struct _rocsparse_color_info* rocsparse_color_info;

/*! \ingroup types_module
 * \brief \p rocsparse_sparse_to_sparse_descr is a structure holding the rocSPARSE sparse_to_sparse
 * descriptor data. It must be initialized using
 * the rocsparse_create_sparse_to_sparse_descr() routine. It should be destroyed at the
 * end using rocsparse_destroy_sparse_to_sparse_descr().
 */
typedef struct _rocsparse_sparse_to_sparse_descr* rocsparse_sparse_to_sparse_descr;

/*! \ingroup types_module
 * \brief \p rocsparse_extract_descr is a structure holding the rocSPARSE extract
 * descriptor data. It must be initialized using
 * the rocsparse_create_extract_descr() routine. It should be destroyed at the
 * end using rocsparse_destroy_extract_descr().
 */
typedef struct _rocsparse_extract_descr* rocsparse_extract_descr;

/*! \ingroup types_module
 * \brief \p rocsparse_spgeam_descr is a structure holding the rocSPARSE spgeam
 * descriptor data. It must be initialized using
 * the rocsparse_create_spgeam_descr() routine. It should be destroyed at the
 * end using rocsparse_destroy_spgeam_descr().
 */
typedef struct _rocsparse_spgeam_descr* rocsparse_spgeam_descr;

/*! \ingroup types_module
 * \brief \p rocsparse_spmv_descr is a structure holding the rocSPARSE spmv
 * descriptor data. It must be initialized using
 * the rocsparse_create_spmv_descr() routine. It should be destroyed at the
 * end using rocsparse_destroy_spmv_descr().
 */
typedef struct _rocsparse_spmv_descr* rocsparse_spmv_descr;

/*! \ingroup types_module
 * \brief \p rocsparse_sptrsv_descr is a structure holding the rocSPARSE sptrsv
 * descriptor data. It must be initialized using
 * the rocsparse_create_sptrsv_descr() or rocsparse_sptrsv_descr_create() routine. It should be destroyed at the
 * end using rocsparse_destroy_sptrsv_descr() or rocsparse_sptrsv_descr_destroy().
 */
typedef struct _rocsparse_sptrsv_descr* rocsparse_sptrsv_descr;

/*! \ingroup types_module
 * \brief \p rocsparse_sptrsm_descr is a structure holding the rocSPARSE sptrsm
 * descriptor data. It must be initialized using
 * the rocsparse_create_sptrsm_descr() routine. It should be destroyed at the
 * end using rocsparse_destroy_sptrsm_descr().
 */
typedef struct _rocsparse_sptrsm_descr* rocsparse_sptrsm_descr;

/*! \ingroup types_module
 * \brief \p rocsparse_spic0_descr is a structure holding the rocSPARSE spic0
 * descriptor data. It must be initialized using
 * the rocsparse_spic0_descr_create() routine. It should be destroyed at the
 * end using rocsparse_spic0_descr_destroy().
 */
typedef struct _rocsparse_spic0_descr* rocsparse_spic0_descr;

/*! \ingroup types_module
 * \brief \p rocsparse_spilu0_descr is a structure holding the rocSPARSE spilu0
 * descriptor data. It must be initialized using
 * the rocsparse_spilu0_descr_create() routine. It should be destroyed at the
 * end using rocsparse_spilu0_descr_destroy().
 */
typedef struct _rocsparse_spilu0_descr* rocsparse_spilu0_descr;

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup types_module
 *  \brief Specify whether the matrix is to be transposed or not.
 *
 *  \details
 *  The \ref rocsparse_operation indicates the operation performed with the given matrix.
 */
typedef enum rocsparse_operation_
{
    rocsparse_operation_none                = 111, /**< Operate with matrix. */
    rocsparse_operation_transpose           = 112, /**< Operate with transpose. */
    rocsparse_operation_conjugate_transpose = 113 /**< Operate with conj. transpose. */
} rocsparse_operation;

/*! \ingroup types_module
 *  \brief Specify the matrix index base.
 *
 *  \details
 *  The \ref rocsparse_index_base indicates the index base of the indices. For a
 *  given \ref rocsparse_mat_descr, the \ref rocsparse_index_base can be set using
 *  rocsparse_set_mat_index_base(). The current \ref rocsparse_index_base of a matrix
 *  can be obtained by rocsparse_get_mat_index_base().
 */
typedef enum rocsparse_index_base_
{
    rocsparse_index_base_zero = 0, /**< Zero-based indexing. */
    rocsparse_index_base_one  = 1 /**< One-based indexing. */
} rocsparse_index_base;

/*! \ingroup types_module
 *  \brief Specify the matrix type.
 *
 *  \details
 *  The \ref rocsparse_matrix_type indicates the type of a matrix. For a given
 *  \ref rocsparse_mat_descr, the \ref rocsparse_matrix_type can be set using
 *  rocsparse_set_mat_type(). The current \ref rocsparse_matrix_type of a matrix can be
 *  obtained by rocsparse_get_mat_type().
 *
 *  For the matrix types \ref rocsparse_matrix_type_symmetric, \ref rocsparse_matrix_type_hermitian,
 *  and \ref rocsparse_matrix_type_triangular, only the upper or lower part of the matrix
 *  (specified by setting the \ref rocsparse_fill_mode) is assumed to be stored. The purpose of this
 *  is to minimize the amount of memory required to store the matrix.
 *
 *  Routines that accept \ref rocsparse_matrix_type_symmetric or \ref rocsparse_matrix_type_hermitian
 *  will only read from the stored upper or lower part of the matrix but will perform the computation
 *  as if the full symmetric/Hermitian matrix existed. For example, when computing \f$y=A*x\f$, where
 *  A is symmetric and only the lower part is stored, internally the multiplication will be performed
 *  in two steps. First, the computation \f$y=(L+D)*x\f$ will be performed. Secondly, the multiplication
 *  will be completed by performing \f$y=L^T*x + y\f$. This second step involves a transposed
 *  multiplication, which is slower. For this reason, where space allows, it is faster to store the
 *  entire symmetric matrix and use \ref rocsparse_matrix_type_general instead of
 *  \ref rocsparse_matrix_type_symmetric.
 */
typedef enum rocsparse_matrix_type_
{
    rocsparse_matrix_type_general    = 0, /**< General matrix type. */
    rocsparse_matrix_type_symmetric  = 1, /**< Symmetric matrix type. */
    rocsparse_matrix_type_hermitian  = 2, /**< Hermitian matrix type. */
    rocsparse_matrix_type_triangular = 3 /**< Triangular matrix type. */
} rocsparse_matrix_type;

/*! \ingroup types_module
 *  \brief Indicates if the diagonal entries are unity.
 *
 *  \details
 *  The \ref rocsparse_diag_type indicates whether the diagonal entries of a matrix are
 *  unity or not. If \ref rocsparse_diag_type_unit is specified, all present diagonal
 *  values will be ignored. For a given \ref rocsparse_mat_descr, the
 *  \ref rocsparse_diag_type can be set using rocsparse_set_mat_diag_type(). The current
 *  \ref rocsparse_diag_type of a matrix can be obtained by
 *  rocsparse_get_mat_diag_type().
 */
typedef enum rocsparse_diag_type_
{
    rocsparse_diag_type_non_unit = 0, /**< Diagonal entries are non-unity. */
    rocsparse_diag_type_unit     = 1 /**< Diagonal entries are unity. */
} rocsparse_diag_type;

/*! \ingroup types_module
 *  \brief Specify the matrix fill mode.
 *
 *  \details
 *  The \ref rocsparse_fill_mode indicates whether the lower or the upper part is stored
 *  in a sparse triangular matrix. For a given \ref rocsparse_mat_descr, the
 *  \ref rocsparse_fill_mode can be set using rocsparse_set_mat_fill_mode(). The current
 *  \ref rocsparse_fill_mode of a matrix can be obtained by
 *  rocsparse_get_mat_fill_mode().
 */
typedef enum rocsparse_fill_mode_
{
    rocsparse_fill_mode_lower = 0, /**< Lower triangular part is stored. */
    rocsparse_fill_mode_upper = 1 /**< Upper triangular part is stored. */
} rocsparse_fill_mode;

/*! \ingroup types_module
 *  \brief Specify whether the matrix is stored sorted or not.
 *
 *  \details
 *  The \ref rocsparse_storage_mode indicates whether the matrix is stored as sorted or not.
 *  For a given \ref rocsparse_mat_descr, the \ref rocsparse_storage_mode can be set
 *  using rocsparse_set_mat_storage_mode(). The current \ref rocsparse_storage_mode of a
 *  matrix can be obtained by rocsparse_get_mat_storage_mode().
 */
typedef enum rocsparse_storage_mode_
{
    rocsparse_storage_mode_sorted   = 0, /**< Matrix is sorted. */
    rocsparse_storage_mode_unsorted = 1 /**< Matrix is unsorted. */
} rocsparse_storage_mode;

/*! \ingroup types_module
 *  \brief Specify where the operation is performed on.
 *
 *  \details
 *  The \ref rocsparse_action indicates whether the operation is performed on the full
 *  matrix or only on the sparsity pattern of the matrix.
 */
typedef enum rocsparse_action_
{
    rocsparse_action_symbolic = 0, /**< Operate only on indices. */
    rocsparse_action_numeric  = 1 /**< Operate on data and indices. */
} rocsparse_action;

/*! \ingroup types_module
 *  \brief Specify the matrix direction.
 *
 *  \details
 *  The \ref rocsparse_direction indicates whether a dense matrix should be parsed by
 *  rows or by columns, assuming column-major storage.
 */
typedef enum rocsparse_direction_
{
    rocsparse_direction_row    = 0, /**< Parse the matrix by rows. */
    rocsparse_direction_column = 1 /**< Parse the matrix by columns. */
} rocsparse_direction;

/*! \ingroup types_module
 *  \brief HYB matrix partitioning type.
 *
 *  \details
 *  The \ref rocsparse_hyb_partition type indicates how the hybrid format partitioning
 *  between COO and ELL storage formats is performed.
 */
typedef enum rocsparse_hyb_partition_
{
    rocsparse_hyb_partition_auto = 0, /**< Automatically decide on ELL nnz per row. */
    rocsparse_hyb_partition_user = 1, /**< User-provided ELL nnz per row. */
    rocsparse_hyb_partition_max  = 2 /**< Max ELL nnz per row, no COO part. */
} rocsparse_hyb_partition;

/*! \ingroup types_module
 *  \brief Specify policy in analysis functions.
 *
 *  \details
 *  The \ref rocsparse_analysis_policy specifies whether gathered analysis data should be
 *  reused or not. If meta data from, for example, a previous \ref rocsparse_scsrilu0_analysis
 *  "rocsparse_Xcsrilu0_analysis()" call is available, it can be reused for subsequent calls to, for example,
 *  \ref rocsparse_scsrsv_analysis "rocsparse_Xcsrsv_analysis()" and greatly improve performance
 *  of the analysis function.
 */
typedef enum rocsparse_analysis_policy_
{
    rocsparse_analysis_policy_reuse = 0, /**< Try to reuse meta data. */
    rocsparse_analysis_policy_force = 1 /**< Force to rebuild meta data. */
} rocsparse_analysis_policy;

/*! \ingroup types_module
 *  \brief Specify policy in triangular solvers and factorizations.
 *
 *  \details
 *  This is a placeholder.
 */
typedef enum rocsparse_solve_policy_
{
    rocsparse_solve_policy_auto = 0 /**< Automatically decide on level information. */
} rocsparse_solve_policy;

/*! \ingroup types_module
 *  \brief Indicates if the pointer is a device pointer or host pointer.
 *
 *  \details
 *  The \ref rocsparse_pointer_mode indicates whether scalar values are passed by
 *  reference on the host or device. The \ref rocsparse_pointer_mode can be changed by
 *  rocsparse_set_pointer_mode(). The current pointer mode in use can be obtained by
 *  rocsparse_get_pointer_mode().
 */
typedef enum rocsparse_pointer_mode_
{
    rocsparse_pointer_mode_host   = 0, /**< Scalar pointers are in host memory. */
    rocsparse_pointer_mode_device = 1 /**< Scalar pointers are in device memory. */
} rocsparse_pointer_mode;

/*! \ingroup types_module
 *  \brief Indicates if the layer is active through the use of a bitmask.
 *
 *  \details
 *  The \ref rocsparse_layer_mode bit mask indicates the logging characteristics.
 */
typedef enum rocsparse_layer_mode
{
    rocsparse_layer_mode_none      = 0x0, /**< Layer is not active. */
    rocsparse_layer_mode_log_trace = 0x1, /**< Layer is in logging mode. */
    rocsparse_layer_mode_log_bench = 0x2, /**< Layer is in benchmarking mode (deprecated). */
    rocsparse_layer_mode_log_debug = 0x4 /**< Layer is in debug mode. */
} rocsparse_layer_mode;

/*! \ingroup types_module
 *  \brief List of rocSPARSE status codes definition.
 *
 *  \details
 *  This is a list of the \ref rocsparse_status types that are used by the rocSPARSE
 *  library.
 */
typedef enum rocsparse_status_
{
    rocsparse_status_success                 = 0, /**< success. */
    rocsparse_status_invalid_handle          = 1, /**< handle not initialized, invalid, or null. */
    rocsparse_status_not_implemented         = 2, /**< function is not implemented. */
    rocsparse_status_invalid_pointer         = 3, /**< invalid pointer parameter. */
    rocsparse_status_invalid_size            = 4, /**< invalid size parameter. */
    rocsparse_status_memory_error            = 5, /**< failed memory allocation, copy or dealloc. */
    rocsparse_status_internal_error          = 6, /**< other internal library failure. */
    rocsparse_status_invalid_value           = 7, /**< invalid value parameter. */
    rocsparse_status_arch_mismatch           = 8, /**< device arch is not supported. */
    rocsparse_status_zero_pivot              = 9, /**< encountered zero pivot. */
    rocsparse_status_not_initialized         = 10, /**< descriptor has not been initialized. */
    rocsparse_status_type_mismatch           = 11, /**< index types do not match. */
    rocsparse_status_requires_sorted_storage = 12, /**< sorted storage required. */
    rocsparse_status_thrown_exception        = 13, /**< exception being thrown. */
    rocsparse_status_continue                = 14 /**< Nothing preventing the function to proceed */
} rocsparse_status;

/*! \ingroup types_module
 *  \brief List of rocSPARSE data status code definitions.
 *
 *  \details
 *  This is a list of the \ref rocsparse_data_status types that are used by the rocSPARSE
 *  library in the matrix check routines.
 */
typedef enum rocsparse_data_status_
{
    rocsparse_data_status_success            = 0, /**< Success. */
    rocsparse_data_status_inf                = 1, /**< An inf value detected. */
    rocsparse_data_status_nan                = 2, /**< A nan value detected. */
    rocsparse_data_status_invalid_offset_ptr = 3, /**< An invalid row pointer offset detected. */
    rocsparse_data_status_invalid_index      = 4, /**< An invalid row index detected. */
    rocsparse_data_status_duplicate_entry    = 5, /**< Duplicate indices detected. */
    rocsparse_data_status_invalid_sorting    = 6, /**< Incorrect sorting detected. */
    rocsparse_data_status_invalid_fill       = 7 /**< Incorrect fill mode detected. */
} rocsparse_data_status;

/*! \ingroup types_module
 *  \brief List of rocSPARSE index types.
 *
 *  \details
 *  Indicates the index width of a rocSPARSE index type.
 */
typedef enum rocsparse_indextype_
{
    rocsparse_indextype_u16
    [[deprecated("rocsparse_indextype_u16 is no longer supported and will be removed in a future "
                 "release. Use "
                 "rocsparse_indextype_i32 or rocsparse_indextype_i64 instead.")]]
    = 1, /**< 16-bit unsigned integer. \deprecated This index type is unsupported and will be
            removed in a future release. Use \ref rocsparse_indextype_i32 or \ref
            rocsparse_indextype_i64 instead. */
    rocsparse_indextype_i32 = 2, /**< 32-bit signed integer. */
    rocsparse_indextype_i64 = 3 /**< 64-bit signed integer. */
} rocsparse_indextype;

/*! \ingroup types_module
 *  \brief List of rocSPARSE data types.
 *
 *  \details
 *  Indicates the precision width of data stored in a rocSPARSE type.
 */
typedef enum rocsparse_datatype_
{
    rocsparse_datatype_f16_r  = 150, /**< 16-bit floating point, real. */
    rocsparse_datatype_f32_r  = 151, /**< 32-bit floating point, real. */
    rocsparse_datatype_f64_r  = 152, /**< 64-bit floating point, real. */
    rocsparse_datatype_f32_c  = 154, /**< 32-bit floating point, complex. */
    rocsparse_datatype_f64_c  = 155, /**< 64-bit floating point, complex. */
    rocsparse_datatype_i8_r   = 160, /**<  8-bit signed integer, real */
    rocsparse_datatype_u8_r   = 161, /**<  8-bit unsigned integer, real */
    rocsparse_datatype_i32_r  = 162, /**< 32-bit signed integer, real */
    rocsparse_datatype_u32_r  = 163, /**< 32-bit unsigned integer, real */
    rocsparse_datatype_bf16_r = 168 /**< 16-bit brain floating point, real */
} rocsparse_datatype;

/*! \ingroup types_module
 *  \brief List of sparse matrix formats.
 *
 *  \details
 *  This is a list of supported \ref rocsparse_format types that are used to describe a
 *  sparse matrix.
 */
typedef enum rocsparse_format_
{
    rocsparse_format_coo     = 0, /**< COO sparse matrix format. */
    rocsparse_format_coo_aos = 1, /**< COO AoS sparse matrix format. */
    rocsparse_format_csr     = 2, /**< CSR sparse matrix format. */
    rocsparse_format_csc     = 3, /**< CSC sparse matrix format. */
    rocsparse_format_ell     = 4, /**< ELL sparse matrix format. */
    rocsparse_format_bell    = 5, /**< Blocked ELL sparse matrix format. */
    rocsparse_format_bsr     = 6, /**< BSR sparse matrix format. */
    rocsparse_format_sell    = 7 /**< Sliced ELL sparse matrix format. */
} rocsparse_format;

/*! \ingroup types_module
 *  \brief List of dense matrix ordering.
 *
 *  \details
 *  This is a list of supported \ref rocsparse_order types that are used to describe the
 *  memory layout of a dense matrix.
 */
typedef enum rocsparse_order_
{
    rocsparse_order_row    = 0, /**< Row major. */
    rocsparse_order_column = 1 /**< Column major. */
} rocsparse_order;

/*! \ingroup types_module
 *  \brief List of sparse matrix attributes.
 */
typedef enum rocsparse_spmat_attribute_
{
    rocsparse_spmat_fill_mode    = 0, /**< Fill mode attribute. */
    rocsparse_spmat_diag_type    = 1, /**< Diag type attribute. */
    rocsparse_spmat_matrix_type  = 2, /**< Matrix type attribute. */
    rocsparse_spmat_storage_mode = 3 /**< Matrix storage attribute. */
} rocsparse_spmat_attribute;

/*! \ingroup types_module
   *  \brief List of sparse-to-sparse algorithms.
   *
   *  \details
   *  This is a list of supported \ref rocsparse_sparse_to_sparse_alg types that are used to perform
   *  sparse-to-sparse conversion.
   */
typedef enum rocsparse_sparse_to_sparse_alg_
{
    rocsparse_sparse_to_sparse_alg_default
    = 0, /**< Default sparse-to-sparse algorithm for the given format. */
} rocsparse_sparse_to_sparse_alg;

/*! \ingroup types_module
   *  \brief List of sparse-to-sparse stages.
   *
   *  \details
   *  This is a list of possible stages during sparse-to-sparse conversion. The typical order is
   *  \ref rocsparse_sparse_to_sparse_stage_analysis, then \ref rocsparse_sparse_to_sparse_stage_compute.
   */
typedef enum rocsparse_sparse_to_sparse_stage_
{
    rocsparse_sparse_to_sparse_stage_analysis = 0, /**< Data analysis. */
    rocsparse_sparse_to_sparse_stage_compute  = 1 /**< Performs the actual conversion. */
} rocsparse_sparse_to_sparse_stage;

/*! \ingroup types_module
   *  \brief List of extract algorithms.
   *
   *  \details
   *  This is a list of supported \ref rocsparse_extract_alg types that are used to perform
   *  the submatrix extraction.
   */
typedef enum rocsparse_extract_alg_
{
    rocsparse_extract_alg_default = 0, /**< Default extract algorithm for the given format. */
} rocsparse_extract_alg;

/*! \ingroup types_module
   *  \brief List of extract stages.
   *
   *  \details
   *  The analysis \ref rocsparse_extract_stage_analysis must be done before the first call of the calculation function \ref rocsparse_extract_stage_compute.
   */
typedef enum rocsparse_extract_stage_
{
    rocsparse_extract_stage_analysis = 0, /**< Data analysis. */
    rocsparse_extract_stage_compute  = 1 /**< Performs the actual extraction. */
} rocsparse_extract_stage;

/*! \ingroup types_module
 *  \brief List of Iterative ILU0 algorithms.
 *
 *  \details
 *  This is a list of supported \ref rocsparse_itilu0_alg types that are used to perform
 *  the iterative ILU0 algorithm.
 */
typedef enum rocsparse_itilu0_alg_
{
    rocsparse_itilu0_alg_default = 0, /**< Asynchronous ITILU0 algorithm with in-place storage. */
    rocsparse_itilu0_alg_async_inplace
    = 1, /**< Asynchronous ITILU0 algorithm with in-place storage. */
    rocsparse_itilu0_alg_async_split
    = 2, /**< Asynchronous ITILU0 algorithm with explicit storage splitting. */
    rocsparse_itilu0_alg_sync_split
    = 3, /**< Synchronous ITILU0 algorithm with explicit storage splitting. */
    rocsparse_itilu0_alg_sync_split_fusion [[deprecated]]
    = 4 /**< Semi-synchronous ITILU0 algorithm with explicit storage splitting. */
} rocsparse_itilu0_alg;

/*! \ingroup types_module
 *  \brief List of iterative ILU0 options.
 *
 *  \details
 *  This is a list of supported \ref rocsparse_itilu0_option options that are used to perform
 *  the iterative ILU0 algorithm.
 */
typedef enum rocsparse_itilu0_option_
{
    rocsparse_itilu0_option_verbose                = 1, /**< Compute a stopping criteria. */
    rocsparse_itilu0_option_stopping_criteria      = 2, /**< Compute a stopping criteria. */
    rocsparse_itilu0_option_compute_nrm_correction = 4, /**< Compute correction. */
    rocsparse_itilu0_option_compute_nrm_residual   = 8, /**< Compute residual. */
    rocsparse_itilu0_option_convergence_history    = 16, /**< Log convergence history. */
    rocsparse_itilu0_option_coo_format             = 32 /**< Use internal coordinate format. */
} rocsparse_itilu0_option;

/*! \ingroup types_module
 *  \brief List of interleaved gtsv algorithms.
 *
 *  \details
 *  This is a list of supported \ref rocsparse_gtsv_interleaved_alg types that are used to perform
 *  interleaved tridiagonal solve.
 */
typedef enum rocsparse_gtsv_interleaved_alg_
{
    rocsparse_gtsv_interleaved_alg_default
    = 0, /**< Solve interleaved gtsv using QR algorithm (stable). */
    rocsparse_gtsv_interleaved_alg_thomas
    = 1, /**< Solve interleaved gtsv using Thomas algorithm (unstable). */
    rocsparse_gtsv_interleaved_alg_lu
    = 2, /**< Solve interleaved gtsv using LU algorithm (stable). */
    rocsparse_gtsv_interleaved_alg_qr
    = 3 /**< Solve interleaved gtsv using QR algorithm (stable). */
} rocsparse_gtsv_interleaved_alg;

/*! \ingroup types_module
 *  \brief List of check matrix stages.
 *
 *  \details
 *  This is a list of possible stages during check matrix computation. The typical order is
 *  \ref rocsparse_check_spmat_stage_buffer_size, then \ref rocsparse_check_spmat_stage_compute.
 */
typedef enum rocsparse_check_spmat_stage_
{
    rocsparse_check_spmat_stage_buffer_size = 0, /**< Returns the required buffer size. */
    rocsparse_check_spmat_stage_compute     = 1, /**< Performs check. */
} rocsparse_check_spmat_stage;

/*! \ingroup types_module
 *  \brief List of inputs to the SpMV descriptor.
 *
 *  \details
 *  This is a list of possible inputs to the SpMV descriptor.
 */
typedef enum rocsparse_spmv_input_
{
    rocsparse_spmv_input_alg, /**< Select algorithm for input on a SpMV descriptor. */
    rocsparse_spmv_input_operation, /**< Select matrix transpose operation for input on a SpMV descriptor. */
    rocsparse_spmv_input_scalar_datatype, /**< Select scalar datatype for input on a SpMV descriptor. */
    rocsparse_spmv_input_compute_datatype, /**< Select compute datatype for input on a SpMV descriptor. */
    rocsparse_spmv_input_nnz_use_starting_block_ids, /**< Configure usage of starting block IDs for non-zero split. */
    rocsparse_spmv_input_enable_extra /**< Enable/disable extra vectors computation for SpMV descriptor. */
} rocsparse_spmv_input;

/*! \ingroup types_module
 *  \brief List of SpMV-Version2 stages.
 *
 *  \details
 *  This is a list of possible stages during SpMV-Version2 computation.
 */
typedef enum rocsparse_v2_spmv_stage_
{
    rocsparse_v2_spmv_stage_analysis, /**< Analysis of the data. */
    rocsparse_v2_spmv_stage_compute /**< Performs the actual SpMV computation. */
} rocsparse_v2_spmv_stage;

/*! \ingroup types_module
 *  \brief List of SpMV stages.
 *
 *  \details
 *  This is a list of possible stages during SpMV computation. The typical order is
 *  \ref rocsparse_spmv_stage_buffer_size, \ref rocsparse_spmv_stage_preprocess, and \ref rocsparse_spmv_stage_compute.
 */
typedef enum rocsparse_spmv_stage_
{
    rocsparse_spmv_stage_buffer_size = 1, /**< Returns the required buffer size. */
    rocsparse_spmv_stage_preprocess  = 2, /**< Preprocess data. */
    rocsparse_spmv_stage_compute     = 3 /**< Performs the actual SpMV computation. */
} rocsparse_spmv_stage;

/*! \ingroup types_module
 *  \brief List of SpMV algorithms.
 *
 *  \details
 *  This is a list of supported \ref rocsparse_spmv_alg types that are used to perform
 *  matrix vector product.
 */
typedef enum rocsparse_spmv_alg_
{
    rocsparse_spmv_alg_default      = 0, /**< Default SpMV algorithm for the given format. */
    rocsparse_spmv_alg_coo          = 1, /**< COO SpMV algorithm 1 (segmented) for COO matrices. */
    rocsparse_spmv_alg_csr_adaptive = 2, /**< CSR SpMV algorithm 1 (adaptive) for CSR matrices. */
    rocsparse_spmv_alg_csr_rowsplit = 3, /**< CSR SpMV algorithm 2 (rowsplit) for CSR matrices. */
    rocsparse_spmv_alg_ell          = 4, /**< ELL SpMV algorithm for ELL matrices. */
    rocsparse_spmv_alg_coo_atomic   = 5, /**< COO SpMV algorithm 2 (atomic) for COO matrices. */
    rocsparse_spmv_alg_bsr          = 6, /**< BSR SpMV algorithm 1 for BSR matrices. */
    rocsparse_spmv_alg_csr_lrb      = 7, /**< CSR SpMV algorithm 3 (LRB) for CSR matrices. */
    rocsparse_spmv_alg_csr_nnzsplit = 8, /**< CSR SpMV algorithm 4 (nnzsplit) for CSR matrices. */
    rocsparse_spmv_alg_sell         = 9, /**< SLICED ELL SpMV algorithm for SLICED ELL matrices. */
    rocsparse_spmv_alg_csr_stream [[deprecated]]
    = rocsparse_spmv_alg_csr_rowsplit /**< CSR SpMV algorithm 2 (stream) for CSR matrices. */
} rocsparse_spmv_alg;

/*! \ingroup types_module
 *  \brief List of SpSV algorithms.
 *
 *  \details
 *  This is a list of supported \ref rocsparse_spsv_alg types that are used to perform
 *  triangular solve.
 */
typedef enum rocsparse_spsv_alg_
{
    rocsparse_spsv_alg_default = 0, /**< Default SpSV algorithm for the given format. */
} rocsparse_spsv_alg;

/*! \ingroup types_module
 *  \brief List of SpSV stages.
 *
 *  \details
 *  This is a list of possible stages during SpSV computation. The typical order is
 *  \ref rocsparse_spsv_stage_buffer_size, \ref rocsparse_spsv_stage_preprocess, and \ref rocsparse_spsv_stage_compute.
 */
typedef enum rocsparse_spsv_stage_
{
    rocsparse_spsv_stage_buffer_size = 1, /**< Returns the required buffer size. */
    rocsparse_spsv_stage_preprocess  = 2, /**< Preprocess data. */
    rocsparse_spsv_stage_compute     = 3 /**< Performs the actual SpSV computation. */
} rocsparse_spsv_stage;

/*! \ingroup types_module
 *  \brief List of SpITSV algorithms.
 *
 *  \details
 *  This is a list of supported \ref rocsparse_spitsv_alg types that are used to perform
 *  triangular solve.
 */
typedef enum rocsparse_spitsv_alg_
{
    rocsparse_spitsv_alg_default = 0, /**< Default SpITSV algorithm for the given format. */
} rocsparse_spitsv_alg;

/*! \ingroup types_module
 *  \brief List of SpITSV stages.
 *
 *  \details
 *  This is a list of possible stages during SpITSV computation. The typical order is
 *  \ref rocsparse_spitsv_stage_buffer_size, \ref rocsparse_spitsv_stage_preprocess, and \ref rocsparse_spitsv_stage_compute.
 */
typedef enum rocsparse_spitsv_stage_
{
    rocsparse_spitsv_stage_buffer_size = 1, /**< Returns the required buffer size. */
    rocsparse_spitsv_stage_preprocess  = 2, /**< Preprocess data. */
    rocsparse_spitsv_stage_compute     = 3 /**< Performs the actual SpITSV computation. */
} rocsparse_spitsv_stage;

/*! \ingroup types_module
 *  \brief List of SpSM algorithms.
 *
 *  \details
 *  This is a list of supported \ref rocsparse_spsm_alg types that are used to perform
 *  triangular solve.
 */
typedef enum rocsparse_spsm_alg_
{
    rocsparse_spsm_alg_default = 0, /**< Default SpSM algorithm for the given format. */
} rocsparse_spsm_alg;

/*! \ingroup types_module
 *  \brief List of SpSM stages.
 *
 *  \details
 *  This is a list of possible stages during SpSM computation. The typical order is
 *  \ref rocsparse_spsm_stage_buffer_size, \ref rocsparse_spsm_stage_preprocess, and \ref rocsparse_spsm_stage_compute.
 */
typedef enum rocsparse_spsm_stage_
{
    rocsparse_spsm_stage_buffer_size = 1, /**< Returns the required buffer size. */
    rocsparse_spsm_stage_preprocess  = 2, /**< Preprocess data. */
    rocsparse_spsm_stage_compute     = 3 /**< Performs the actual SpSM computation. */
} rocsparse_spsm_stage;

/*! \ingroup types_module
*  \brief List of SpMM algorithms.
*
*  \details
*  This is a list of supported \ref rocsparse_spmm_alg types that are used to perform
*  matrix vector product.
*/
typedef enum rocsparse_spmm_alg_
{
    rocsparse_spmm_alg_default = 0, /**< Default SpMM algorithm for the given format. */
    rocsparse_spmm_alg_csr, /**< SpMM algorithm for the CSR format using row split and shared memory. */
    rocsparse_spmm_alg_coo_segmented, /**< SpMM algorithm for the COO format using segmented scan. */
    rocsparse_spmm_alg_coo_atomic, /**< SpMM algorithm for the COO format using atomics. */
    rocsparse_spmm_alg_csr_row_split, /**< SpMM algorithm for the CSR format using row split and shfl. */
    rocsparse_spmm_alg_csr_merge, /**< SpMM algorithm for CSR format using the nnz split algorithm. This is the same as \p rocsparse_spmm_alg_csr_nnz_split. */
    rocsparse_spmm_alg_coo_segmented_atomic, /**< SpMM algorithm for the COO format using segmented scan and atomics. */
    rocsparse_spmm_alg_bell, /**< SpMM algorithm for the Blocked ELL format. */
    rocsparse_spmm_alg_bsr, /**< SpMM algorithm for the BSR format. */
    rocsparse_spmm_alg_csr_merge_path, /**< SpMM algorithm for the CSR format using the merge path algorithm. */
    rocsparse_spmm_alg_csr_nnz_split
    = rocsparse_spmm_alg_csr_merge /**< SpMM algorithm for the CSR format using the nnz split algorithm. */
} rocsparse_spmm_alg;

/*! \ingroup types_module
 *  \brief List of sddmm algorithms.
 *
 *  \details
 *  This is a list of supported \ref rocsparse_sddmm_alg types that are used to perform
 *  matrix vector product.
 */
typedef enum rocsparse_sddmm_alg_
{
    rocsparse_sddmm_alg_default = 0, /**< Default sddmm algorithm for the given format. */
    rocsparse_sddmm_alg_dense   = 1 /**< Sddmm algorithm using dense blas operations. */
} rocsparse_sddmm_alg;

/*! \ingroup types_module
 *  \brief List of sparse-to-dense algorithms.
 *
 *  \details
 *  This is a list of supported \ref rocsparse_sparse_to_dense_alg types that are used to perform
 *  sparse-to-dense conversion.
 */
typedef enum rocsparse_sparse_to_dense_alg_
{
    rocsparse_sparse_to_dense_alg_default
    = 0, /**< Default sparse-to-dense algorithm for the given format. */
} rocsparse_sparse_to_dense_alg;

/*! \ingroup types_module
 *  \brief List of dense-to-sparse algorithms.
 *
 *  \details
 *  This is a list of supported \ref rocsparse_dense_to_sparse_alg types that are used to perform
 *  dense-to-sparse conversion.
 */
typedef enum rocsparse_dense_to_sparse_alg_
{
    rocsparse_dense_to_sparse_alg_default
    = 0, /**< Default dense to sparse algorithm for the given format. */
} rocsparse_dense_to_sparse_alg;

/*! \ingroup types_module
 *  \brief List of SpMM stages.
 *
 *  \details
 *  This is a list of possible stages during SpMM computation. The typical order is
 *  \ref rocsparse_spmm_stage_buffer_size, \ref rocsparse_spmm_stage_preprocess, and \ref rocsparse_spmm_stage_compute.
 */
typedef enum rocsparse_spmm_stage_
{
    rocsparse_spmm_stage_buffer_size = 1, /**< Returns the required buffer size. */
    rocsparse_spmm_stage_preprocess  = 2, /**< Preprocess data. */
    rocsparse_spmm_stage_compute     = 3 /**< Performs the actual SpMM computation. */
} rocsparse_spmm_stage;

/*! \ingroup types_module
 *  \brief List of SpGEMM stages.
 *
 *  \details
 *  This is a list of possible stages during SpGEMM computation. The typical order is
 *  \ref rocsparse_spgemm_stage_buffer_size, \ref rocsparse_spgemm_stage_nnz, and \ref rocsparse_spgemm_stage_compute.
 */
typedef enum rocsparse_spgemm_stage_
{
    rocsparse_spgemm_stage_buffer_size = 1, /**< Returns the required buffer size. */
    rocsparse_spgemm_stage_nnz         = 2, /**< Computes the number of non-zero entries. */
    rocsparse_spgemm_stage_compute     = 3, /**< Performs the actual SpGEMM computation. */
    rocsparse_spgemm_stage_symbolic    = 4, /**< Performs the actual SpGEMM symbolic computation. */
    rocsparse_spgemm_stage_numeric     = 5 /**< Performs the actual SpGEMM numeric computation. */
} rocsparse_spgemm_stage;

/*! \ingroup types_module
 *  \brief List of SpGEMM algorithms.
 *
 *  \details
 *  This is a list of supported \ref rocsparse_spgemm_alg types that are used to perform a
 *  sparse matrix sparse matrix product.
 */
typedef enum rocsparse_spgemm_alg_
{
    rocsparse_spgemm_alg_default = 0 /**< Default SpGEMM algorithm for the given format. */
} rocsparse_spgemm_alg;

/*! \ingroup types_module
 *  \brief List of singularity types encountered in triangular solves and incomplete factorizations.
 */
typedef enum rocsparse_singularity_
{
    rocsparse_singularity_none, /**< No singularity detected. */
    rocsparse_singularity_symbolic, /**< The sparsity pattern inherently prevents a full rank, for example, a missing diagonal element. */
    rocsparse_singularity_numeric_exact, /**< An exact zero was encountered during numerical calculation. */
    rocsparse_singularity_numeric_near, /**< A near zero was encountered during numerical calculation, that is, within a given tolerance. */
} rocsparse_singularity;

/*! \ingroup types_module
 *  \brief List of SpTRSV algorithms.
 *
 *  \details
 *  This is a list of supported \ref rocsparse_sptrsv_alg types that are used to perform
 *  triangular solve.
 */
typedef enum rocsparse_sptrsv_alg_
{
    rocsparse_sptrsv_alg_default = 0, /**< Default SpTRSV algorithm for the given format. */
} rocsparse_sptrsv_alg;

/*! \ingroup types_module
 *  \brief List of SpTRSV stages.
 *
 *  \details
 *  This is a list of possible stages during SpTRSV computation.
 */
typedef enum rocsparse_sptrsv_stage_
{
    rocsparse_sptrsv_stage_analysis, /**< Analysis. */
    rocsparse_sptrsv_stage_compute /**< Performs the actual SpTRSV computation. */
} rocsparse_sptrsv_stage;

/*! \ingroup types_module
 *  \brief List of inputs to the SpTRSV descriptor.
 *
 *  \details
 *  This is a list of possible inputs to the SpTRSV descriptor.
 */
typedef enum rocsparse_sptrsv_input_
{
    rocsparse_sptrsv_input_alg, /**< Select algorithm \ref rocsparse_sptrsv_alg for input on a SpTRSV descriptor. */
    rocsparse_sptrsv_input_operation, /**< Select matrix operation \ref rocsparse_operation for input on a SpTRSV descriptor. */
    rocsparse_sptrsv_input_scalar_datatype, /**< Select scalar datatype \ref rocsparse_datatype for input on a SpTRSV descriptor. */
    rocsparse_sptrsv_input_compute_datatype, /**< Select compute datatype \ref rocsparse_datatype for input on a SpTRSV descriptor. */
    rocsparse_sptrsv_input_scalar_alpha, /**< Select scalar alpha pointer for input on a SpTRSV descriptor. */
    rocsparse_sptrsv_input_analysis_policy /**< Select the analysis policy \ref rocsparse_analysis_policy for input on a SpTRSV descriptor. */
} rocsparse_sptrsv_input;

/*! \ingroup types_module
 *  \brief List of outputs to SpTRSV descriptor.
 *
 *  \details
 *  This is a list of possible outputs to the SpTRSV descriptor.
 */
typedef enum rocsparse_sptrsv_output_
{
    rocsparse_sptrsv_output_zero_pivot_position, /**< Get zero pivot \p int64_t based position for output from the SpTRSV descriptor. */
    rocsparse_sptrsv_output_singularity, /**< Get the type of \ref rocsparse_singularity detected during SpTRSV calculation for output from the SpTRSV descriptor. */
    rocsparse_sptrsv_output_singularity_position /**< Get the singularity \p int64_t based position for output from the SpTRSV descriptor. */
} rocsparse_sptrsv_output;

/*! \ingroup types_module
 *  \brief List of SpTRSM algorithms.
 *
 *  \details
 *  This is a list of supported \ref rocsparse_sptrsm_alg types that are used to perform
 *  triangular solve.
 */
typedef enum rocsparse_sptrsm_alg_
{
    rocsparse_sptrsm_alg_default = 0, /**< Default SpTRSM algorithm for the given format. */
} rocsparse_sptrsm_alg;

/*! \ingroup types_module
 *  \brief List of SpTRSM stages.
 *
 *  \details
 *  This is a list of possible stages during SpTRSM computation.
 */
typedef enum rocsparse_sptrsm_stage_
{
    rocsparse_sptrsm_stage_analysis, /**< Analysis. */
    rocsparse_sptrsm_stage_compute /**< Performs the actual SpTRSM computation. */
} rocsparse_sptrsm_stage;

/*! \ingroup types_module
 *  \brief List of inputs to the SpTRSM descriptor.
 *
 *  \details
 *  This is a list of possible inputs to the SpTRSM descriptor.
 */
typedef enum rocsparse_sptrsm_input_
{
    rocsparse_sptrsm_input_alg, /**< Select algorithm \ref rocsparse_sptrsm_alg for input on a SpTRSM descriptor. */
    rocsparse_sptrsm_input_operation_A, /**< Select matrix A operation \ref rocsparse_operation for input on a SpTRSM descriptor. */
    rocsparse_sptrsm_input_operation_X, /**< Select matrix X operation \ref rocsparse_operation for input on a SpTRSM descriptor. */
    rocsparse_sptrsm_input_compute_datatype, /**< Select compute datatype \ref rocsparse_datatype for input on a SpTRSM descriptor. */
    rocsparse_sptrsm_input_scalar_datatype, /**< Select scalar datatype \ref rocsparse_datatype for input on a SpTRSM descriptor. */
    rocsparse_sptrsm_input_scalar_alpha, /**< Select scalar alpha pointer for input on a SpTRSM descriptor. This datatype is used as the compute type. */
    rocsparse_sptrsm_input_analysis_policy /**< Select the analysis policy \ref rocsparse_analysis_policy for input on a SpTRSM descriptor. */
} rocsparse_sptrsm_input;

/*! \ingroup types_module
 *  \brief List of outputs to SpTRSM descriptor.
 *
 *  \details
 *  This is a list of possible outputs to the SpTRSM descriptor.
 */
typedef enum rocsparse_sptrsm_output_
{
    rocsparse_sptrsm_output_zero_pivot_position /**< Get zero pivot \p int64_t based position for output from the SpTRSM descriptor and synchronously return zero_pivot. */
} rocsparse_sptrsm_output;

/*! \ingroup types_module
 *  \brief List of SpIC0 algorithms.
 *
 *  \details
 *  This is a list of supported \ref rocsparse_spic0_alg types that are used to perform the incomplete Cholesky factorization
 *  of level 0.
 */
typedef enum rocsparse_spic0_alg_
{
    rocsparse_spic0_alg_default
} rocsparse_spic0_alg;

/*! \ingroup types_module
 *  \brief List of SpIC0 stages.
 *
 *  \details
 *  This is a list of possible stages during SpIC0 computation.
 */
typedef enum rocsparse_spic0_stage_
{
    rocsparse_spic0_stage_analysis, /**< Analysis. */
    rocsparse_spic0_stage_compute /**< Performs the actual SpIC0 computation. */
} rocsparse_spic0_stage;

/*! \ingroup types_module
 *  \brief List of inputs to the SpIC0 descriptor.
 *
 *  \details
 *  This is a list of possible inputs to the SpIC0 descriptor.
 */
typedef enum rocsparse_spic0_input_
{
    rocsparse_spic0_input_alg, /**< Select algorithm \ref rocsparse_spic0_alg for input on a SpIC0 descriptor. */
    rocsparse_spic0_input_analysis_policy, /**< Select the analysis policy \ref rocsparse_analysis_policy for input on a SpIC0 descriptor. */
    rocsparse_spic0_input_compute_datatype, /**< Select compute datatype \ref rocsparse_datatype for input on a SpIC0 descriptor. */
    rocsparse_spic0_input_boost_enable, /**< Enable diagonal boosting for input on a SpIC0 descriptor. */
    rocsparse_spic0_input_boost_tolerance, /**< Select diagonal boosting tolerance on a SpIC0 descriptor. */
    rocsparse_spic0_input_boost_value, /**< Select diagonal boosting value on a SpIC0 descriptor. */
    rocsparse_spic0_input_singularity_tolerance, /**< Select singularity tolerance for input on a SpIC0 descriptor. */
} rocsparse_spic0_input;

/*! \ingroup types_module
 *  \brief List of outputs to SpIC0 descriptor.
 *
 *  \details
 *  This is a list of possible outputs to the SpIC0 descriptor.
 */
typedef enum rocsparse_spic0_output_
{
    rocsparse_spic0_output_singularity, /**< Get the type of \ref rocsparse_singularity detected during SpIC0 calculation for output from the SpIC0 descriptor. */
    rocsparse_spic0_output_singularity_position, /**< Get the singularity \p int64_t based position for output from the SpIC0 descriptor. */
} rocsparse_spic0_output;

/*! \ingroup types_module
 *  \brief List of SpILU0 algorithms.
 *
 *  \details
 *  This is a list of supported \ref rocsparse_spilu0_alg types that are used to perform the incomplete LU factorization
 *  of level 0.
 */
typedef enum rocsparse_spilu0_alg_
{
    rocsparse_spilu0_alg_default
} rocsparse_spilu0_alg;

/*! \ingroup types_module
 *  \brief List of SpILU0 stages.
 *
 *  \details
 *  This is a list of possible stages during SpILU0 computation.
 */
typedef enum rocsparse_spilu0_stage_
{
    rocsparse_spilu0_stage_analysis, /**< Analysis. */
    rocsparse_spilu0_stage_compute, /**< Performs the actual SpILU0 computation. */
} rocsparse_spilu0_stage;

/*! \ingroup types_module
 *  \brief List of inputs to the SpILU0 descriptor.
 *
 *  \details
 *  This is a list of possible inputs to the SpILU0 descriptor.
 */
typedef enum rocsparse_spilu0_input_
{
    rocsparse_spilu0_input_alg, /**< Select algorithm \ref rocsparse_spilu0_alg for input on a SpILU0 descriptor. */
    rocsparse_spilu0_input_analysis_policy, /**< Select the analysis policy \ref rocsparse_analysis_policy for input on a SpILU0 descriptor. */
    rocsparse_spilu0_input_compute_datatype, /**< Select compute datatype \ref rocsparse_datatype for input on a SpILU0 descriptor. */
    rocsparse_spilu0_input_boost_enable, /**< Enable diagonal boosting for input on a SpILU0 descriptor. */
    rocsparse_spilu0_input_boost_tolerance, /**< Select diagonal boosting tolerance on a SpILU0 descriptor. */
    rocsparse_spilu0_input_boost_value, /**< Select diagonal boosting value on a SpILU0 descriptor. */
    rocsparse_spilu0_input_singularity_tolerance, /**< Select singularity tolerance for input on a SpILU0 descriptor. */
} rocsparse_spilu0_input;

/*! \ingroup types_module
 *  \brief List of outputs to the SpILU0 descriptor.
 *
 *  \details
 *  This is a list of possible outputs to the SpILU0 descriptor.
 */
typedef enum rocsparse_spilu0_output_
{
    rocsparse_spilu0_output_singularity, /**< Get the type of \ref rocsparse_singularity detected during SpILU0 calculation for output from the SpILU0 descriptor. */
    rocsparse_spilu0_output_singularity_position, /**< Get the singularity \p int64_t based position for output from the SpILU0 descriptor. */
} rocsparse_spilu0_output;

/*! \ingroup types_module
 *  \brief List of SpGEAM stages.
 *
 *  \details
 *  This is a list of possible stages during SpGEAM computation. The typical order is
 *  \p rocsparse_spgeam_stage_buffer_size, \p rocsparse_spgeam_stage_analysis, and \p rocsparse_spgeam_stage_compute.
 */
typedef enum rocsparse_spgeam_stage_
{
    rocsparse_spgeam_stage_analysis = 1, /**< Computes number of non-zero entries. */
    rocsparse_spgeam_stage_compute  = 2, /**< Performs the actual SpGEAM computation. */
    rocsparse_spgeam_stage_symbolic_analysis
    = 3, /**< Performs only the symbolic analysis SpGEAM computation to fill the column indices array. */
    rocsparse_spgeam_stage_symbolic_compute
    = 4, /**< Performs only the symbolic SpGEAM computation to fill the column indices array. */
    rocsparse_spgeam_stage_numeric_analysis
    = 5, /**< Performs only the numeric analysis SpGEAM computation to fill the values array. */
    rocsparse_spgeam_stage_numeric_compute
    = 6 /**< Performs only the numeric SpGEAM computation to fill the values array. */
} rocsparse_spgeam_stage;

/*! \ingroup types_module
 *  \brief List of inputs to the SpGEAM descriptor.
 *
 *  \details
 *  This is a list of possible inputs to the SpGEAM descriptor.
 */
typedef enum rocsparse_spgeam_input_
{
    rocsparse_spgeam_input_alg, /**< Select algorithm for input on a SpGEAM descriptor. */
    rocsparse_spgeam_input_scalar_datatype, /**< Select scalar data type for input on a SpGEAM descriptor. */
    rocsparse_spgeam_input_compute_datatype, /**< Select compute data type for input on a SpGEAM descriptor. */
    rocsparse_spgeam_input_operation_A, /**< Select A matrix transpose operation for input on a SpGEAM descriptor. */
    rocsparse_spgeam_input_operation_B, /**< Select B matrix transpose operation for input on a SpGEAM descriptor. */
    rocsparse_spgeam_input_scalar_alpha, /**< Select scalar multiplier alpha for input on a SpGEAM descriptor. */
    rocsparse_spgeam_input_scalar_beta /**< Select scalar multiplier beta for input on a SpGEAM descriptor. */
} rocsparse_spgeam_input;

/*! \ingroup types_module
 *  \brief List of outputs to SpGEAM descriptor.
 *
 *  \details
 *  This is a list of possible outputs to the SpGEAM descriptor.
 */
typedef enum rocsparse_spgeam_output_
{
    rocsparse_spgeam_output_nnz /**< Select nnz count for output from the SpGEAM descriptor. */
} rocsparse_spgeam_output;

/*! \ingroup types_module
 *  \brief List of SpGEAM algorithms.
 *
 *  \details
 *  This is a list of supported \ref rocsparse_spgeam_alg types that are used to perform
 *  sparse matrix sparse matrix product.
 */
typedef enum rocsparse_spgeam_alg_
{
    rocsparse_spgeam_alg_default = 0 /**< Default SpGEAM algorithm for the given format. */
} rocsparse_spgeam_alg;

/*! \ingroup types_module
 *  \brief List of gpsv algorithms.
 *
 *  \details
 *  This is a list of supported \ref rocsparse_gpsv_interleaved_alg types that are used to solve
 *  pentadiagonal linear systems.
 */
typedef enum rocsparse_gpsv_interleaved_alg_
{
    rocsparse_gpsv_interleaved_alg_default = 0, /**< Default gpsv algorithm. */
    rocsparse_gpsv_interleaved_alg_qr      = 1 /**< QR algorithm. */
} rocsparse_gpsv_interleaved_alg;

#ifdef __cplusplus
}
#endif

#endif /* ROCSPARSE_TYPES_H */
