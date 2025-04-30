/* ************************************************************************
 * Copyright (c) 2022-2024 Advanced Micro Devices, Inc.
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
 * \brief rocsparselt-types.h defines data types used by rocsparselt
 */

#pragma once
#ifndef _ROCSPARSELT_TYPES_H_
#define _ROCSPARSELT_TYPES_H_

#include <float.h>
#include <hip/library_types.h>
#include <stddef.h>
#include <stdint.h>

#define ROCSPARSELT_KERNEL __global__
#define ROCSPARSELT_DEVICE_ILF __device__

/*! \ingroup types_module
 *  \brief Handle to the rocSPARSELt library context queue.
 *
 *  \details
 *  The rocSPARSELt handle is a structure holding the rocSPARSELt library context. It must
 *  be initialized using \ref rocsparselt_create_handle and the returned handle must be
 *  passed to all subsequent library function calls. It should be destroyed at the end
 *  using \ref rocsparselt_destroy_handle.
 */
typedef struct
{
    uint8_t data[11024];
} rocsparselt_handle;

/*! \ingroup types_module
 *  \brief Descriptor of the matrix.
 *
 *  \details
 *  The rocSPARSELt matrix descriptor is a structure holding all properties of a matrix.
 *  It must be initialized using \ref rocsparselt_create_mat_descr and the returned
 *  descriptor must be passed to all subsequent library calls that involve the matrix.
 *  It should be destroyed at the end using \ref rocsparselt_destroy_mat_descr.
 */
typedef struct
{
    uint8_t data[11024];
} rocsparselt_mat_descr;

/*! \ingroup types_module
 *  \brief Descriptor of the matrix multiplication operation
 *
 *  \details
 *  The rocSPARSELt matrix multiplication descriptor is a structure holding
 *  the description of the matrix multiplication operation.
 *  It is initialized with \ref rocsparselt_matmul_descr_init function.
 */
typedef struct
{
    uint8_t data[11024];
} rocsparselt_matmul_descr;

/*! \ingroup types_module
 *  \brief Descriptor of the matrix multiplication algorithm.
 *
 *  \details
 *  It is initialized with \ref rocsparselt_matmul_alg_selection_init function.
 */
typedef struct
{
    uint8_t data[11024];
} rocsparselt_matmul_alg_selection;

/*! \ingroup types_module
 *  \brief Descriptor of the matrix multiplication execution plan
 *
 *  \details
 *  The rocSPARSELt matrix multiplication execution plan descriptor is a structure holding
 *  all the information necessary to execute the rocsparselt_matmul() operation.
 *  It is initialized and destroyed with \ref rocsparselt_matmul_plan_init
 *  and \ref rocsparselt_matmul_plan_destroy functions respectively.
 */
typedef struct
{
    uint8_t data[11024];
} rocsparselt_matmul_plan;

// Generic API

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup types_module
 *  \brief Specify whether the matrix is to be transposed or not.
 *
 *  \details
 *  The \ref rocsparselt_operation indicates the operation performed with the given matrix.
 */
typedef enum rocsparselt_operation_
{
    rocsparselt_operation_none                = 111, /**< Operate with matrix. */
    rocsparselt_operation_transpose           = 112, /**< Operate with transpose. */
    rocsparselt_operation_conjugate_transpose = 113 /**< Operate with conj. transpose. */
} rocsparselt_operation;

/*! \ingroup types_module
 *  \brief List of rocsparse status codes definition.
 *
 *  \details
 *  This is a list of the \ref rocsparselt_status types that are used by the rocSPARSE
 *  library.
 */
typedef enum rocsparselt_status_
{
    rocsparselt_status_success                 = 0, /**< success. */
    rocsparselt_status_invalid_handle          = 1, /**< handle not initialized, invalid or null. */
    rocsparselt_status_not_implemented         = 2, /**< function is not implemented. */
    rocsparselt_status_invalid_pointer         = 3, /**< invalid pointer parameter. */
    rocsparselt_status_invalid_size            = 4, /**< invalid size parameter. */
    rocsparselt_status_memory_error            = 5, /**< failed memory allocation, copy, dealloc. */
    rocsparselt_status_internal_error          = 6, /**< other internal library failure. */
    rocsparselt_status_invalid_value           = 7, /**< invalid value parameter. */
    rocsparselt_status_arch_mismatch           = 8, /**< device arch is not supported. */
    rocsparselt_status_zero_pivot              = 9, /**< encountered zero pivot. */
    rocsparselt_status_not_initialized         = 10, /**< descriptor has not been initialized. */
    rocsparselt_status_type_mismatch           = 11, /**< index types do not match. */
    rocsparselt_status_requires_sorted_storage = 12, /**< sorted storage required. */
    rocsparselt_status_continue                = 13 /**< nothing preventing function to proceed. */
} rocsparselt_status;

/*! \ingroup types_module
 *  \brief List of dense matrix ordering.
 *
 *  \details
 *  This is a list of supported \ref rocsparselt_order types that are used to describe the
 *  memory layout of a dense matrix
 */
typedef enum rocsparselt_order_
{
    rocsparselt_order_row    = 0, /**< Row major. */
    rocsparselt_order_column = 1 /**< Column major. */
} rocsparselt_order;

/*! \ingroup types_module
 *  \brief Indicates if the pointer is device pointer or host pointer.
 *
 *  \details
 *  The \ref rocsparselt_pointer_mode indicates whether scalar values (alpha/beta) are passed by
 *  reference on the host or device.
 *  Note, only support rocsparselt_pointer_mode_host.
 */
typedef enum rocsparselt_pointer_mode_
{
    rocsparselt_pointer_mode_host   = 0, /**< scalar pointers are in host memory. */
    rocsparselt_pointer_mode_device = 1 /**< scalar pointers are in device memory. */
} rocsparselt_pointer_mode;

/*! \ingroup types_module
 *  \brief Indicates if layer is active with bitmask.
 *
 *  \details
 *  The \ref rocsparselt_layer_mode bit mask indicates the logging characteristics.
 */
typedef enum rocsparselt_layer_mode
{
    rocsparselt_layer_mode_none      = 0, /**< layer is not active. */
    rocsparselt_layer_mode_log_error = 1, /**< layer is in error mode. */
    rocsparselt_layer_mode_log_trace = 2, /**< layer is in trace mode. */
    rocsparselt_layer_mode_log_hints = 4, /**< layer is in hints mode. */
    rocsparselt_layer_mode_log_info  = 8, /**< layer is in info mode. */
    rocsparselt_layer_mode_log_api   = 16, /**< layer is in api mode. */
} rocsparselt_layer_mode;

/*! \ingroup types_module
 *  \brief Indicates if layer is active with level.
 *
 *  \details
 *  The \ref rocsparselt_layer_level number indicates the logging characteristics.
 *  A higher log level will show logs including the lower log level.
 */
typedef enum rocsparselt_layer_level
{
    rocsparselt_layer_level_none      = 0, /**< layer is not active. */
    rocsparselt_layer_level_log_error = 1, /**< layer is in error mode. */
    rocsparselt_layer_level_log_trace = 2, /**< layer is in trace mode. */
    rocsparselt_layer_level_log_hints = 3, /**< layer is in hints mode. */
    rocsparselt_layer_level_log_info  = 4, /**< layer is in info mode. */
    rocsparselt_layer_level_log_api   = 5, /**< layer is in api mode. */
} rocsparselt_layer_level;

/*! \ingroup types_module
 *  \brief Specify the sparsity of the structured matrix.
 *
 *  \details
 *  The enumerator specifies the sparsity ratio of the structured matrix as
 *  sparsity = nnz / total elements
 *  The sparsity property is used in the rocsparselt_structured_descr_init() function.
 */
typedef enum rocsparselt_sparsity_
{
    rocsparselt_sparsity_50_percent = 0, /**< 50% sparsity ratio - 2:4 */
} rocsparselt_sparsity;

/*! \ingroup types_module
 *  \brief Specify the matrix type.
 *
 *  \details
 */
typedef enum rocsparselt_matrix_type_
{
    rocsparselt_matrix_type_unknown    = 0,
    rocsparselt_matrix_type_dense      = 1, /**< dense matrix type. */
    rocsparselt_matrix_type_structured = 2, /**< structured matrix type. */
} rocsparselt_matrix_type;

/*! \ingroup types_module
 *  \brief Specify the compute precision modes of the matrix
 *
 *  \details
 */
typedef enum rocsparselt_compute_type_
{
    rocsparselt_compute_f32 = 300, /**< 32-bit floating-point precision. */
    rocsparselt_compute_i32 = 301 /**< 32-bit integer precision. */
} rocsparselt_compute_type;

/*! \ingroup types_module
 *  \brief Specify the additional attributes of a matrix descriptor
 *
 *  \details
 *  The rocsparselt_mat_descr_attribute is used in the
 *  \ref rocsparselt_mat_descr_set_attribute and \ref rocsparselt_mat_descr_get_attribute functions
 */
typedef enum rocsparselt_mat_descr_attribute_
{
    rocsparselt_mat_num_batches  = 0, /**< number of matrices in a batch. */
    rocsparselt_mat_batch_stride = 1 /**< s
    tride between consecutive matrices in a batch expressed in terms of matrix elements. */
} rocsparselt_mat_descr_attribute;

/*! \ingroup types_module
 *  \brief Specify the additional attributes of a matrix multiplication descriptor
 *
 *  \details
 *  The rocsparselt_matmul_descr_attribute_ is used in the
 *  \ref rocsparselt_matmul_descr_set_attribute and \ref rocsparselt_matmul_descr_get_attribute functions
 */
typedef enum rocsparselt_matmul_descr_attribute_
{
    rocsparselt_matmul_activation_relu = 0, /**< ReLU activation function. */
    rocsparselt_matmul_activation_relu_upperbound
    = 1, /**< Upper bound of the ReLU activation function. */
    rocsparselt_matmul_activation_relu_threshold
    = 2, /**< Lower threshold of the ReLU activation function. */
    rocsparselt_matmul_activation_gelu = 3, /**< GeLU activation function. */
    rocsparselt_matmul_activation_gelu_scaling
    = 4, /** Scaling coefficient for the GeLU activation function. It implies gelu is endable */
    rocsparselt_matmul_alpha_vector_scaling
    = 5, /** Enable/Disable alpha vector (per-channel) scaling */
    rocsparselt_matmul_beta_vector_scaling
    = 6, /** Enable/Disable beta vector (per-channel) scaling */
    rocsparselt_matmul_bias_pointer
    = 7, /**< Bias pointer. The bias vector size must equal to the number of rows of the output matrix (D). */
    rocsparselt_matmul_bias_stride
    = 8, /**< Bias stride between consecutive bias vectors. 0 means broadcast the first bias vector. */
    rocsparselt_matmul_activation_abs       = 9, /**< ABS activation function. */
    rocsparselt_matmul_activation_leakyrelu = 10, /**< LeakyReLU activation function. */
    rocsparselt_matmul_activation_leakyrelu_alpha
    = 11, /**< Alpha value of the LeakyReLU activation function. */
    rocsparselt_matmul_activation_sigmoid = 12, /**< Sigmoid activation function. */
    rocsparselt_matmul_activation_tanh    = 13, /**< Tanh activation function. */
    rocsparselt_matmul_activation_tanh_alpha
    = 14, /**< Alpha value of the Tanh activation function. */
    rocsparselt_matmul_activation_tanh_beta
    = 15, /**< Beta value of the Tanh activation function. */
    rocsparselt_matmul_bias_type = 16, /**< Precision of bias >*/
    rocsparselt_matmul_activation_none, /**< activation function is disabled. */
} rocsparselt_matmul_descr_attribute;

/*! \ingroup types_module
 *  \brief Specify the algorithm for matrix-matrix multiplication.
 *
 *  \details
 *  The \ref rocsparselt_matmul_alg is used in the \ref rocsparselt_matmul_alg_selection_init function.
 */
typedef enum rocsparselt_matmul_alg_
{
    rocsparselt_matmul_alg_default = 0, /**< Default algorithm. */
} rocsparselt_matmul_alg;

/*! \ingroup types_module
 *  \brief Specify the matrix multiplication algorithm attributes.
 *
 *  \details
 *  The \ref rocsparselt_matmul_alg_attribute is used in the
 *  \ref rocsparselt_matmul_alg_get_attribute and \ref rocsparselt_matmul_alg_set_attribute functions.
 */
typedef enum rocsparselt_matmul_alg_attribute_
{
    rocsparselt_matmul_alg_config_id     = 0, /**< Algorithm ID (set and query). */
    rocsparselt_matmul_alg_config_max_id = 1, /**< Algorithm ID limit (query only). */
    rocsparselt_matmul_search_iterations
    = 2, /**< Number of iterations (kernel launches per algorithm)
                                                  for rocsparselt_matmul_search, default=10. */
    rocsparselt_matmul_split_k
    = 3, /**< Split-K factor, default=not set. Valid range: [1, K]. Value 1 is equivalent to the Split-K feature is disabled */
    rocsparselt_matmul_split_k_mode
    = 4, /**< Number of kernels to call for Split-K. Values are specified in rocsparselt_split_k_mode. */
    rocsparselt_matmul_split_k_buffers
    = 5, /**< Device memory buffers to store partial results for the reduction. The valid range is [1, SplitK - 1] */
} rocsparselt_matmul_alg_attribute;

/*! \ingroup types_module
 *  \brief Specify the pruning algorithm to apply to the structured matrix before the compression.
 *
 *  \details
 *  The \ref rocsparselt_prune_alg is used in the \ref rocsparselt_smfmac_prune function.
 */
typedef enum rocsparselt_prune_alg_
{
    rocsparselt_prune_smfmac_tile = 0,
    rocsparselt_prune_smfmac_strip
    = 1, /**< - Zero-out two values in a 1x4 strip to maximize the L1-norm of the resulting strip. */
} rocsparselt_prune_alg;

/*! \brief Indicates if atomics operations are allowed. Not allowing atomic operations
*    may generally improve determinism and repeatability of results at a cost of performance */
typedef enum rocsparselt_atomics_mode_
{
    /*! \brief Algorithms will refrain from atomics where applicable */
    rocsparselt_atomics_not_allowed = 0,
    /*! \brief Algorithms will take advantage of atomics where applicable */
    rocsparselt_atomics_allowed = 1,
} rocsparselt_atomics_mode;

typedef enum rocsparselt_split_k_mode_
{
    rocsparselt_splik_k_mode_one_kernel
    = 0, /**< Use the same SP-MM kernel to do the final reduction */
    rocsparselt_split_k_mode_two_kernels = 1, /**< Use anoghter kernel to do the final reduction */
} rocsparselt_split_k_mode;

#ifdef __cplusplus
}
#endif

#endif /* _ROCSPARSELT_TYPES_H_ */
