/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2022-2025 Advanced Micro Devices, Inc.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

/*! \file
*  \brief hipsparselt.h provides Sparse Linear Algebra Subprograms
*  of Prune, Compressed and Matrix Multiplication, using HIP optimized for AMD GPU hardware.
*/

//! HIP = Heterogeneous-compute Interface for Portability
//!
//! Define a extremely thin runtime layer that allows source code to be compiled
//! unmodified through either AMD HCC or NVCC. Key features tend to be in the spirit
//! and terminology of CUDA, but with a portable path to other GPUs as well.
//!
//! This is the master include file for hipSPARSELt, wrapping around hipSPARSELt and
//! cuSPARSELt "version 0.3".
//

/*! \defgroup types_module Data types
 *
 *
 *  \defgroup library_module Library management functions
 *  Provides the library handle
 *
 *  \defgroup matrix_desc_module Matrix descriptor functions
 *  Defines sparse and dense matrices
 *
 *  \defgroup matmul_module Matrix multiply (matmul) functions
 *  Multiplies sparse matrices
 *
 *  \defgroup matmul_desc_module Matmul descriptor functions
 *  Defines how to matrix multiply
 *
 *  \defgroup matmul_algo_module Matmul algorithm functions
 *  Provides algorithms for matrix multiply
 *
 *  \defgroup helper_module Helper functions
 *  Required for subsequent library calls
 *
 *  \defgroup aux_module Auxilary functions
 *  Initializes hipSPARSELt for the current HIP device
 */

#pragma once
#ifndef _HIPSPARSELT_H_
#define _HIPSPARSELT_H_

#include <hipsparse/hipsparse.h>
#include <hipsparselt/hipsparselt-export.h>
#include <hipsparselt/hipsparselt-version.h>

#include <hip/hip_complex.h>
#include <hip/hip_runtime_api.h>

#if defined(__HIP_PLATFORM_AMD__)
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_fp8.h>
#include <hip/library_types.h>
#else
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <library_types.h>
#endif

/* Opaque structures holding information */
// clang-format off

#if defined(__HIP_PLATFORM_AMD__)
/*! \ingroup types_module
 *  \brief Handle to the hipSPARSELt library context queue.
 *
 *  \details
 *  The hipSPARSELt handle is a structure holding the hipSPARSELt library context. It must
 *  be initialized using \ref hipsparseLtInit and the returned handle must be
 *  passed to all subsequent library function calls. It should be destroyed at the end
 *  using \ref hipsparseLtDestroy.
 */
typedef struct hipsparseLtHandle_t {uint8_t data[11024];} hipsparseLtHandle_t;

/*! \ingroup types_module
 *  \brief Descriptor of the matrix.
 *
 *  \details
 *  The hipSPARSELt matrix descriptor is a structure holding all properties of a matrix.
 *  It must be initialized using \ref hipsparseLtDenseDescriptorInit and the returned
 *  descriptor must be passed to all subsequent library calls that involve the matrix.
 *  It should be destroyed at the end using \ref hipsparseLtMatDescriptorDestroy.
 */
typedef struct hipsparseLtMatDescriptor_t {uint8_t data[11024];} hipsparseLtMatDescriptor_t;

/*! \ingroup types_module
 *  \brief Descriptor of the matrix multiplication operation
 *
 *  \details
 *  The hipSPARSELt matrix multiplication descriptor is a structure holding
 *  the description of the matrix multiplication operation.
 *  It is initialized with \ref hipsparseLtMatmulDescriptorInit function.
 */
typedef struct hipsparseLtMatmulDescriptor_t {uint8_t data[11024];} hipsparseLtMatmulDescriptor_t;

/*! \ingroup types_module
 *  \brief Descriptor of the matrix multiplication algorithm.
 *
 *  \details
 *  It is initialized with \ref hipsparseLtMatmulAlgSelectionInit function.
 */
typedef struct hipsparseLtMatmulAlgSelection_t {uint8_t data[11024];} hipsparseLtMatmulAlgSelection_t;

/*! \ingroup types_module
 *  \brief Descriptor of the matrix multiplication execution plan
 *
 *  \details
 *  The hipSPARSELt matrix multiplication execution plan descriptor is a structure holding
 *  all the information necessary to execute the \ref hipsparseLtMatmul operation.
 *  It is initialized and destroyed with \ref hipsparseLtMatmulPlanInit
 *  and \ref hipsparseLtMatmulPlanDestroy functions respectively.
 */
typedef struct hipsparseLtMatmulPlan_t {uint8_t data[11024];} hipsparseLtMatmulPlan_t;
#elif defined(__HIP_PLATFORM_NVIDIA__)
typedef __nv_bfloat16 hip_bfloat16;
typedef struct {uint8_t data[11024];} hipsparseLtHandle_t;
typedef struct {uint8_t data[11024];} hipsparseLtMatDescriptor_t;
typedef struct {uint8_t data[11024];} hipsparseLtMatmulDescriptor_t;
typedef struct {uint8_t data[11024];} hipsparseLtMatmulAlgSelection_t;
typedef struct {uint8_t data[11024];} hipsparseLtMatmulPlan_t;
#endif

/*! \ingroup types_module
 *  \brief Specify the sparsity of the structured matrix.
 *
 *  \details
 *  The enumerator specifies the sparsity ratio of the structured matrix as
 *  sparsity = nnz / total elements
 *  The sparsity property is used in the \ref hipsparseLtStructuredDescriptorInit function.
 */
typedef enum {
   HIPSPARSELT_SPARSITY_50_PERCENT /**< 50% sparsity ratio - 1:2 for tf32 and float,
                                                             2:4 for half, bfloat16, int */
} hipsparseLtSparsity_t;

/*! \ingroup types_module
 *  \brief Specify the additional attributes of a matrix descriptor
 *
 *  \details
 *  The hipsparseLtMatDescAttribute_t is used in the
 *  \ref hipsparseLtMatDescSetAttribute and \ref hipsparseLtMatDescGetAttribute functions
 */
typedef enum {
   HIPSPARSELT_MAT_NUM_BATCHES,     /**< number of matrices in a batch. READ/WRITE */
   HIPSPARSELT_MAT_BATCH_STRIDE,    /**< stride between consecutive matrices in a batch expressed in terms of matrix elements. READ/WRITE */
} hipsparseLtMatDescAttribute_t;

/*! \ingroup types_module
 *  \brief Specify the compute precision modes of the matrix
 *
 *  \details
 */
typedef enum {
   HIPSPARSELT_COMPUTE_16F = 0,     /**< 16-bit floating-point precision. CUDA backend only. */
   HIPSPARSELT_COMPUTE_32I,         /**< 32-bit integer precision */
   HIPSPARSELT_COMPUTE_32F,         /**< 32-bit floating-point precision. HIP backend only. */
   HIPSPARSELT_COMPUTE_TF32,        /**< 32-bit floating point value are rounded to TF32 before the computation.
                                       CUDA backend only. */
   HIPSPARSELT_COMPUTE_TF32_FAST    /**< 32-bit floating point value are truncated to TF32 before the computation.
                                       CUDA backend only. */
} hipsparseLtComputetype_t;

/*! \ingroup types_module
 *  \brief Specify the additional attributes of a matrix multiplication descriptor
 *
 *  \details
 *  The hipsparseLtMatmulDescAttribute_t is used in the
 *  \ref hipsparseLtMatmulDescSetAttribute and \ref hipsparseLtMatmulDescGetAttribute functions
 */
typedef enum {
   HIPSPARSELT_MATMUL_ACTIVATION_RELU = 0,             /**< ReLU activation function. */
   HIPSPARSELT_MATMUL_ACTIVATION_RELU_UPPERBOUND = 1,  /**< Upper bound of the ReLU activation function. */
   HIPSPARSELT_MATMUL_ACTIVATION_RELU_THRESHOLD = 2,   /**< Lower threshold of the ReLU activation function. */
   HIPSPARSELT_MATMUL_ACTIVATION_GELU = 3,             /**< GeLU activation function. */
   HIPSPARSELT_MATMUL_ACTIVATION_GELU_SCALING = 4,     /**< Scaling coefficient for the GeLU activation function. It implies gelu is endable */
   HIPSPARSELT_MATMUL_ALPHA_VECTOR_SCALING = 5,        /**< Enable/Disable alpha vector (per-channel) scaling */
   HIPSPARSELT_MATMUL_BETA_VECTOR_SCALING = 6,         /**< Enable/Disable beta vector (per-channel) scaling */
   HIPSPARSELT_MATMUL_BIAS_STRIDE = 7,                 /**< Bias pointer. The bias vector size must equal to the number of rows of the output matrix (D). */
   HIPSPARSELT_MATMUL_BIAS_POINTER = 8,                /**< Bias stride between consecutive bias vectors. 0 means broadcast the first bias vector. */
   HIPSPARSELT_MATMUL_ACTIVATION_ABS = 9,              /**< ABS activation function. HIP backend only */
   HIPSPARSELT_MATMUL_ACTIVATION_LEAKYRELU = 10,       /**< LeakyReLU activation function. HIP backend only */
   HIPSPARSELT_MATMUL_ACTIVATION_LEAKYRELU_ALPHA = 11, /**< Alpha value of the LeakyReLU activation function. HIP backend only */
   HIPSPARSELT_MATMUL_ACTIVATION_SIGMOID = 12,         /**< Sigmoid activation function. HIP backend only */
   HIPSPARSELT_MATMUL_ACTIVATION_TANH = 13,            /**< Tanh activation function. HIP backend only */
   HIPSPARSELT_MATMUL_ACTIVATION_TANH_ALPHA = 14,      /**< Alpha value of the Tanh activation function. HIP backend only */
   HIPSPARSELT_MATMUL_ACTIVATION_TANH_BETA = 15,       /**< Beta value of the Tanh activation function. HIP backend only */
   HIPSPARSELT_MATMUL_BIAS_TYPE = 16,                  /**< Precision of Bias. HIP backend only,
                                                            When Input's datatype is FP16 - Bias type can be FP16 or FP32. (default FP16)
                                                            When Input's datatype is BF16 - Bias type can be BF16 or FP32. (default BF16)
                                                            In other cases - Bias type is FP32.*/
   HIPSPARSELT_MATMUL_SPARSE_MAT_POINTER = 17,         /**< Pointer to the pruned sparse matrix. */
} hipsparseLtMatmulDescAttribute_t;

/*! \ingroup types_module
 *  \brief Specify the algorithm for matrix-matrix multiplication.
 *
 *  \details
 *  The \ref hipsparseLtMatmulAlg_t is used in the \ref hipsparseLtMatmulAlgSelectionInit function.
 */
typedef enum {
   HIPSPARSELT_MATMUL_ALG_DEFAULT
} hipsparseLtMatmulAlg_t;

/*! \ingroup types_module
 *  \brief Specify the matrix multiplication algorithm attributes.
 *
 *  \details
 *  The \ref hipsparseLtMatmulAlgAttribute_t is used in the
 *  \ref hipsparseLtMatmulAlgGetAttribute and \ref hipsparseLtMatmulAlgSetAttribute functions.
 */
typedef enum {
   HIPSPARSELT_MATMUL_ALG_CONFIG_ID = 0,     // READ/WRITE
   HIPSPARSELT_MATMUL_ALG_CONFIG_MAX_ID = 1, // READ-ONLY
   HIPSPARSELT_MATMUL_SEARCH_ITERATIONS = 2,  // READ/WRITE
   HIPSPARSELT_MATMUL_SPLIT_K = 3,
   HIPSPARSELT_MATMUL_SPLIT_K_MODE = 4,
   HIPSPARSELT_MATMUL_SPLIT_K_BUFFERS = 5,
} hipsparseLtMatmulAlgAttribute_t;

/*! \ingroup types_module
 *  \brief Specify the pruning algorithm to apply to the structured matrix before the compression.
 *
 *  \details
 *  The \ref hipsparseLtPruneAlg_t is used in the \ref hipsparseLtSpMMAPrune and \ref hipsparseLtSpMMAPrune2 function.
 */
typedef enum {
   HIPSPARSELT_PRUNE_SPMMA_TILE  = 0, /**< Zero out eight elements in a 4x4 tile, nonzero elements have the maximum L1-norm value in all combinations in the tile. Exactly two elements in each row and column. */
   HIPSPARSELT_PRUNE_SPMMA_STRIP = 1, /**< Zero out two elements in a 1x4 strip, nonzero elements have the maximum L1-norm value in all combinations in the strip.*/
} hipsparseLtPruneAlg_t;

/*! \ingroup types_module
 *  \brief Specify the split k mode value.
 *
 *  \details
 *  The \ref hipsparseLtSplitKMode_t is used by HIPSPARSELT_MATMUL_SPLIT_K_MODE attribute in \ref hipsparseLtMatmulAlgAttribute_t.
 */
typedef enum {
   HIPSPARSELT_SPLIT_K_MODE_ONE_KERNEL = 0,  /**< Use the same SP-MM kernel to do the final reduction */
   HIPSPARSELT_SPLIT_K_MODE_TWO_KERNELS = 1, /**< Use another kernel to do the final reduction */
} hipsparseLtSplitKMode_t;

// clang-format on

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup aux_module
 *  \brief Initialize hipSPARSELt for the current HIP device
 *
 *  \details
 *  \p hipsparseLtInitialize Initialize hipSPARSELt for the current HIP device, to avoid costly startup time at the first call on that device.
 *  Only work when using HIP backend.
 *
 */
HIPSPARSELT_EXPORT
void hipsparseLtInitialize();

/*! \ingroup library_module
 *  \brief Retrive the version number of the hipSPARSELt library.
 *
 *  \details
 *  \p hipsparseLtGetVersion return the version number of the hipSPARSELt library.
 *
 *  @param[in]
 *  handle   hipsparselt library handle.
 *  @param[out]
 *  version  the version number of the library.
 *
 *  \retval HIPSPARSE_STATUS_SUCCESS
 *  \retval HIPSPARSE_STATUS_INVALID_VALUE \p handle is invalid.
 */
HIPSPARSELT_EXPORT
hipsparseStatus_t hipsparseLtGetVersion(const hipsparseLtHandle_t* handle, int* version);

/*! \ingroup library_module
 *  \brief Retrive the value of the requested property.
 *
 *  \details
 *  \p hipsparseLtGetProperty return the value of the requested property.
 *
 *  @param[in]
 *  propertyType   property type. hipLibraryPropertyType (defined in library_types.h)
 *  @param[out]
 *  value          value of the requested property.
 *
 *  \retval HIPSPARSE_STATUS_SUCCESS
 */
HIPSPARSELT_EXPORT
hipsparseStatus_t hipsparseLtGetProperty(hipLibraryPropertyType propertyType, int* value);

HIPSPARSELT_EXPORT
hipsparseStatus_t hipsparseLtGetGitRevision(hipsparseLtHandle_t handle, char* rev);

HIPSPARSELT_EXPORT
hipsparseStatus_t hipsparseLtGetArchName(char** archName);

/*! \ingroup library_module
 *  \brief Create a hipsparselt handle
 *
 *  \details
 *  \p hipsparseLtInit creates the hipSPARSELt library context. It must be
 *  initialized before any other hipSPARSELt API function is invoked and must be passed to
 *  all subsequent library function calls. The handle should be destroyed at the end
 *  using hipsparseLtDestroy_handle().
 *
 *  @param[out]
 *  handle  hipsparselt library handle
 *
 *  \retval HIPSPARSE_STATUS_SUCCESS the initialization succeeded.
 *  \retval HIPSPARSE_STATUS_INVALID_VALUE \p handle is invalid.
 */
/* hipSPARSE initialization and management routines */
HIPSPARSELT_EXPORT
hipsparseStatus_t hipsparseLtInit(hipsparseLtHandle_t* handle);

/*! \ingroup library_module
 *  \brief Destroy a hipsparselt handle
 *
 *  \details
 *  \p hipsparseLtDestroy destroys the hipSPARSELt library context and releases all
 *  resources used by the hipSPARSELt library.
 *
 *  @param[in]
 *  handle  hipsparselt library handle
 *
 *  \retval HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
 *  \retval HIPSPARSE_STATUS_NOT_INITIALIZED \p handle is invalid.
 */
HIPSPARSELT_EXPORT
hipsparseStatus_t hipsparseLtDestroy(const hipsparseLtHandle_t* handle);

/* matrix descriptor */
/*! \ingroup matrix_desc_module
 *  \brief Create a descriptor for dense matrix
 *  \details
 *  \p hipsparseLtDenseDescriptorInit creates a matrix descriptor It initializes
 *  It should be destroyed at the end using \ref hipsparseLtMatDescriptorDestroy().
 *
 *  @param[in]
 *  handle     the hipsparselt handle
 *  @param[out]
 *  matDescr   the pointer to the dense matrix descriptor
 *  @param[in]
 *  rows       number of rows
 *  @param[in]
 *  cols       number of columns
 *  @param[in]
 *  ld         leading dimension
 *  @param[in]
 *  alignment  memory alignment in bytes (not used by HIP backend)
 *  @param[in]
 *  valueType  data type of the matrix. see \ref hipDataType
 *  @param[in]
 *  order      memory layout. \p HIPSPARSE_ORDER_COL or \p HIPSPARSE_ORDER_ROW.
 *
 *  \retval HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
 *  \retval HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p descr , \p rows , \p cols , \p ld  is invalid.
 *  \retval HIPSPARSE_STATUS_NOT_SUPPORTED \p rows , \p cols , \p ld , \p alignment , \p valueType or \p order is not supported.
 */
HIPSPARSELT_EXPORT
hipsparseStatus_t hipsparseLtDenseDescriptorInit(const hipsparseLtHandle_t*  handle,
                                                 hipsparseLtMatDescriptor_t* matDescr,
                                                 int64_t                     rows,
                                                 int64_t                     cols,
                                                 int64_t                     ld,
                                                 uint32_t                    alignment,
                                                 hipDataType                 valueType,
                                                 hipsparseOrder_t            order);

/*! \ingroup matrix_desc_module
 *  \brief Create a descriptor for structured matrix
 *  \details
 *  \p hipsparseLtStructuredDescriptorInit creates a matrix descriptor It initializes
 *  It should be destroyed at the end using \ref hipsparseLtMatDescriptorDestroy().
 *
 *  @param[in]
 *  handle     the hipsparselt handle
 *  @param[out]
 *  matDescr   the pointer to the dense matrix descriptor
 *  @param[in]
 *  rows       number of rows
 *  @param[in]
 *  cols       number of columns
 *  @param[in]
 *  ld         leading dimension
 *  @param[in]
 *  alignment  memory alignment in bytes (not used by HIP backend)
 *  @param[in]
 *  valueType  data type of the matrix. see \ref hipDataType
 *  @param[in]
 *  order      memory layout. \p HIPSPARSE_ORDER_COL or \p HIPSPARSE_ORDER_ROW.

 *  @param[in]
 *  sparsity   matrix sparsity ratio. see \ref hipsparseLtSparsity_t
 *
 *  \retval HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
 *  \retval HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p descr , \p rows , \p cols , \p ld  is invalid.
 *  \retval HIPSPARSE_STATUS_NOT_SUPPORTED \p rows , \p cols , \p ld , \p alignment , \p valueType or \p order is not supported.
 */
HIPSPARSELT_EXPORT
hipsparseStatus_t hipsparseLtStructuredDescriptorInit(const hipsparseLtHandle_t*  handle,
                                                      hipsparseLtMatDescriptor_t* matDescr,
                                                      int64_t                     rows,
                                                      int64_t                     cols,
                                                      int64_t                     ld,
                                                      uint32_t                    alignment,
                                                      hipDataType                 valueType,
                                                      hipsparseOrder_t            order,
                                                      hipsparseLtSparsity_t       sparsity);

/*! \ingroup matrix_desc_module
 *  \brief Destroy a matrix descriptor
 *
 *  \details
 *  \p hipsparseLtMatDescriptorDestroy destroys a matrix descriptor and releases all
 *  resources used by the descriptor
 *
 *  @param[in]
 *  matDescr   the matrix descriptor
 *
 *  \retval HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
 *  \retval HIPSPARSE_STATUS_INVALID_VALUE \p descr is invalid.
 */
HIPSPARSELT_EXPORT
hipsparseStatus_t hipsparseLtMatDescriptorDestroy(const hipsparseLtMatDescriptor_t* matDescr);

/*! \ingroup matrix_desc_module
 *  \brief Specify the matrix attribute of a matrix descriptor
 *
 *  \details
 *  \p hipsparseLtMatDescSetAttribute sets the value of the specified attribute belonging
 *  to matrix descr such as number of batches and their stride.
 *
 *  @param[in]
 *  handle          the hipsparselt handle
 *  @param[inout]
 *  matDescr        the matrix descriptor
 *  @param[in]
 *  matAttribute    \ref HIPSPARSELT_MAT_NUM_BATCHES, \ref HIPSPARSELT_MAT_BATCH_STRIDE.
 *  @param[in]
 *  data            pointer to the value to which the specified attribute will be set.
 *  @param[in]
 *  dataSize        size in bytes of the attribute value used for verification.
 *
 *  \retval HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
 *  \retval HIPSPARSE_STATUS_INVALID_VALUE \p handle , \p matmulDescr , \p data or \p dataSize is invalid.
 */
HIPSPARSELT_EXPORT
hipsparseStatus_t hipsparseLtMatDescSetAttribute(const hipsparseLtHandle_t*    handle,
                                                 hipsparseLtMatDescriptor_t*   matDescr,
                                                 hipsparseLtMatDescAttribute_t matAttribute,
                                                 const void*                   data,
                                                 size_t                        dataSize);

/*! \ingroup matrix_desc_module
 *  \brief Get the matrix type of a matrix descriptor
 *
 *  \details
 *  \p hipsparseLtMatDescGetAttribute returns the matrix attribute of a matrix descriptor
 *
 *
 *  @param[in]
 *  handle          the hipsparselt handle
 *  @param[in]
 *  matDescr        the matrix descriptor
 *  @param[in]
 *  matAttribute    \ref HIPSPARSELT_MAT_NUM_BATCHES, \ref HIPSPARSELT_MAT_BATCH_STRIDE.
 *  @param[inout]
 *  data            the memory address containing the attribute value retrieved by this function
 *  @param[in]
 *  dataSize        size in bytes of the attribute value used for verification.
 *
 *  \retval HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
 *  \retval HIPSPARSE_STATUS_INVALID_VALUE \p handle , \p descr , \p data or \p dataSize is invalid.
 */
HIPSPARSELT_EXPORT
hipsparseStatus_t hipsparseLtMatDescGetAttribute(const hipsparseLtHandle_t*        handle,
                                                 const hipsparseLtMatDescriptor_t* matDescr,
                                                 hipsparseLtMatDescAttribute_t     matAttribute,
                                                 void*                             data,
                                                 size_t                            dataSize);

/* matmul descriptor */
/*! \ingroup matmul_desc_module
 *  \brief  Initializes the matrix multiplication descriptor.
 *
 *  \details
 *  \p hipsparseLtMatmulDescriptorInit creates a matrix multiplication descriptor.
 *
 *  @param[in]
 *  handle          the hipsparselt handle
 *  @param[inout]
 *  matmulDescr     the matrix multiplication descriptor
 *  @param[in]
 *  opA             hipsparse operation for Matrix A. \p HIPSPARSE_OPERATION_NON_TRANSPOSE or \p HIPSPARSE_OPERATION_TRANSPOSE
 *  @param[in]
 *  opB             hipsparse operation for Matrix B. \p HIPSPARSE_OPERATION_NON_TRANSPOSE or \p HIPSPARSE_OPERATION_TRANSPOSE
 *  @param[in]
 *  matA            the matrix descriptor (only and must one of matA or matB is a structured sparsity matrix)
 *  @param[in]
 *  matB            the matrix descriptor (only and must one of matA or matB is a structured sparsity matrix)
 *  @param[in]
 *  matC            the matrix descriptor (dense matrix)
 *  @param[in]
 *  matD            the matrix descriptor (dense matrix)
 *  @param[in]
 *  computeType     size in bytes of the attribute value used for verification.
 *
 *  \retval HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
 *  \retval HIPSPARSE_STATUS_INVALID_VALUE \p handle , \p matmulDescr , \p opA , \p opB , \p matA , \p matB , \p matC , \p matD or \p computeType ,is invalid.
 *  \retval HIPSPARSE_STATUS_NOT_SUPPORTED \p opA , \p opB or \p computeType is invalid.
 */
HIPSPARSELT_EXPORT
hipsparseStatus_t hipsparseLtMatmulDescriptorInit(const hipsparseLtHandle_t*        handle,
                                                  hipsparseLtMatmulDescriptor_t*    matmulDescr,
                                                  hipsparseOperation_t              opA,
                                                  hipsparseOperation_t              opB,
                                                  const hipsparseLtMatDescriptor_t* matA,
                                                  const hipsparseLtMatDescriptor_t* matB,
                                                  const hipsparseLtMatDescriptor_t* matC,
                                                  const hipsparseLtMatDescriptor_t* matD,
                                                  hipsparseLtComputetype_t          computeType);

/*! \ingroup matmul_desc_module
 *  \brief Specify the matrix attribute of a matrix descriptor
 *
 *  \details
 *  \p hipsparseLtMatmulDescSetAttribute sets the value of the specified attribute belonging
 *  to matrix descr such as number of batches and their stride.
 *
 *  @param[in]
 *  handle          the hipsparselt handle
 *  @param[inout]
 *  matmulDescr     the matrix multiplication descriptor
 *  @param[in]
 *  matmulAttribute see \ref hipsparseLtMatmulDescAttribute_t
 *  @param[in]
 *  data            pointer to the value to which the specified attribute will be set.
 *  @param[in]
 *  dataSize        size in bytes of the attribute value used for verification.
 *
 *  \retval HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
 *  \retval HIPSPARSE_STATUS_INVALID_VALUE \p handle , \p matDescr , \p data or \p dataSize is invalid.
 *  \retval HIPSPARSE_STATUS_NOT_SUPPORTED \p hipsparseLtMatmulDescAttribute_t is not supported.
 */
HIPSPARSELT_EXPORT
hipsparseStatus_t
    hipsparseLtMatmulDescSetAttribute(const hipsparseLtHandle_t*       handle,
                                      hipsparseLtMatmulDescriptor_t*   matmulDescr,
                                      hipsparseLtMatmulDescAttribute_t matmulAttribute,
                                      const void*                      data,
                                      size_t                           dataSize);

/*! \ingroup matmul_desc_module
 *  \brief Get the matrix type of a matrix descriptor
 *
 *  \details
 *  \p hipsparseLtMatmulDescGetAttribute returns the matrix attribute of a matrix descriptor
 *
 *
 *  @param[in]
 *  handle           the hipsparselt handle
 *  @param[in]
 *  matmulDescr      the matrix multiplication descriptor
 *  @param[in]
 *  matmulAttribute  see \ref hipsparseLtMatmulDescAttribute_t
 *  @param[inout]
 *  data             the memory address containing the attribute value retrieved by this function
 *  @param[in]
 *  dataSize         size in bytes of the attribute value used for verification.
 *
 *  \retval HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
 *  \retval HIPSPARSE_STATUS_INVALID_VALUE \p handle , \p matDescr , \p data or \p dataSize is invalid.
 *  \retval HIPSPARSE_STATUS_NOT_SUPPORTED \p hipsparseLtMatmulDescAttribute_t is not supported.
 */
HIPSPARSELT_EXPORT
hipsparseStatus_t
    hipsparseLtMatmulDescGetAttribute(const hipsparseLtHandle_t*           handle,
                                      const hipsparseLtMatmulDescriptor_t* matmulDescr,
                                      hipsparseLtMatmulDescAttribute_t     matmulAttribute,
                                      void*                                data,
                                      size_t                               dataSize);

/* algorithm selection */
/*! \ingroup matmul_algo_module
 *  \brief Initializes the algorithm selection descriptor
 *  \details
 *  \p hipsparseLtMatmulAlgSelectionInit creates a algorithm selection descriptor.
 *
 *  @param[in]
 *  handle           the hipsparselt handle
 *  @param[out]
 *  algSelection     the pointer to the algorithm selection descriptor
 *  @param[in]
 *  matmulDescr      the matrix multiplication descriptor
 *  @param[in]
 *  alg              the algorithm used to do the matrix multiplication.
 *
 *  \retval HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
 *  \retval HIPSPARSE_STATUS_INVALID_VALUE \p handle , \p matmulDescr or \p algSelection is invalid.
 */
HIPSPARSELT_EXPORT
hipsparseStatus_t
    hipsparseLtMatmulAlgSelectionInit(const hipsparseLtHandle_t*           handle,
                                      hipsparseLtMatmulAlgSelection_t*     algSelection,
                                      const hipsparseLtMatmulDescriptor_t* matmulDescr,
                                      hipsparseLtMatmulAlg_t               alg);

/*! \ingroup matmul_algo_module
 *  \brief Specify the algorithm attribute of a algorithm selection descriptor
 *
 *  \details
 *  \p hipsparseLtMatmulAlgSetAttribute sets the value of the specified attribute
 *  belonging to algorithm selection descriptor.
 *
 *  @param[in]
 *  handle          the hipsparselt handle
 *  @param[inout]
 *  algSelection    the algorithm selection descriptor
 *  @param[in]
 *  attribute       attributes are specify in \ref hipsparseLtMatmulAlgAttribute_t
 *  @param[in]
 *  data            pointer to the value to which the specified attribute will be set.
 *  @param[in]
 *  dataSize        size in bytes of the attribute value used for verification.
 *
 *  \retval HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
 *  \retval HIPSPARSE_STATUS_INVALID_VALUE \p handle , \p  algSelection , \p attribute , \p data or \p dataSize is invalid.
 *  \retval HIPSPARSE_STATUS_NOT_SUPPORTED \p attribute is not supported.
 */
HIPSPARSELT_EXPORT
hipsparseStatus_t hipsparseLtMatmulAlgSetAttribute(const hipsparseLtHandle_t*       handle,
                                                   hipsparseLtMatmulAlgSelection_t* algSelection,
                                                   hipsparseLtMatmulAlgAttribute_t  attribute,
                                                   const void*                      data,
                                                   size_t                           dataSize);

/*! \ingroup matmul_algo_module
 *  \brief Get the specific algorithm attribute from algorithm selection descriptor
 *
 *  \details
 *  \p hipsparseLtMatmulAlgGetAttribute returns the value of the queried attribute belonging
 *  to algorithm selection descriptor.
 *
 *
 *  @param[in]
 *  handle          the hipsparselt handle
 *  @param[in]
 *  algSelection    the algorithm selection descriptor
 *  @param[in]
 *  attribute       attributes are specify in \ref hipsparseLtMatmulAlgAttribute_t
 *  @param[inout]
 *  data            the memory address containing the attribute value retrieved by this function
 *  @param[in]
 *  dataSize        size in bytes of the attribute value used for verification.
 *
 *  \retval HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
 *  \retval HIPSPARSE_STATUS_INVALID_VALUE \p handle , \p  algSelection , \p attribute , \p data or \p dataSize is invalid.
 *  \retval HIPSPARSE_STATUS_NOT_SUPPORTED \p attribute is not supported.
 */
HIPSPARSELT_EXPORT
hipsparseStatus_t
    hipsparseLtMatmulAlgGetAttribute(const hipsparseLtHandle_t*             handle,
                                     const hipsparseLtMatmulAlgSelection_t* algSelection,
                                     hipsparseLtMatmulAlgAttribute_t        attribute,
                                     void*                                  data,
                                     size_t                                 dataSize);

/* matmul plan */
/*! \ingroup matmul_module
 *  \brief Determines the required workspace size.
 *  \details
 *  \p hipsparseLtMatmulGetWorkspace determines the required workspace size
 *  associated to the selected algorithm.
 *
 *  @param[in]
 *  handle           hipsparselt library handle
 *  @param[in]
 *  plan             the matrix multiplication plan descriptor.
 *  @param[out]
 *  workspaceSize    Workspace size in bytes
 *
 *  \retval HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
 *  \retval HIPSPARSE_STATUS_INVALID_VALUE \p handle , \p algSelection or \p workspaceSize is invalid.
 */
HIPSPARSELT_EXPORT
hipsparseStatus_t hipsparseLtMatmulGetWorkspace(const hipsparseLtHandle_t*     handle,
                                                const hipsparseLtMatmulPlan_t* plan,
                                                size_t*                        workspaceSize);

/*! \ingroup matmul_module
 *  \brief Initializes the matrix multiplication plan descriptor
 *  \details
 *  \p hipsparseLtMatmulPlanInit creates a matrix multiplication plan descriptor.
 *  It should be destroyed at the end using \ref hipsparseLtMatmulPlanDestroy.
 *
 *  @param[in]
 *  handle           hipsparselt library handle
 *  @param[out]
 *  plan             the matrix multiplication plan descriptor
 *  @param[in]
 *  matmulDescr      the matrix multiplication descriptor
 *  @param[in]
 *  algSelection     the algorithm selection descriptor
 *
 *  \retval HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
 *  \retval HIPSPARSE_STATUS_INVALID_VALUE \p handle , \p plan , \p matmulDescr , \p algSelection or \p workspaceSize is invalid. \ref HIPSPARSELT_MAT_NUM_BATCHES from matrix A to D are inconisistent
 */
HIPSPARSELT_EXPORT
hipsparseStatus_t hipsparseLtMatmulPlanInit(const hipsparseLtHandle_t*             handle,
                                            hipsparseLtMatmulPlan_t*               plan,
                                            const hipsparseLtMatmulDescriptor_t*   matmulDescr,
                                            const hipsparseLtMatmulAlgSelection_t* algSelection);

/*! \ingroup matmul_module
 *  \brief Destroy a matrix multiplication plan descriptor
 *  \details
 *  \p hipsparseLtMatmulPlanDestroy releases the resources used by an instance
 *  of the matrix multiplication plan. This function is the last call with a specific plan
 *  instance.
 *
 *  @param[in]
 *  plan the matrix multiplication plan descriptor
 *
 *  \retval HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
 *  \retval HIPSPARSE_STATUS_INVALID_VALUE \p plan is invalid.
 */
HIPSPARSELT_EXPORT
hipsparseStatus_t hipsparseLtMatmulPlanDestroy(const hipsparseLtMatmulPlan_t* plan);

/* matmul execution */
/*! \ingroup matmul_module
 *  \brief Sparse matrix dense matrix multiplication
 *
 *  \details
 *  \p hipsparseLtMatmul computes the matrix multiplication of matrices A and B to
 *  produce the output matrix D, according to the following operation:
 *  \f[
 *    D := Activation(\alpha \cdot op(A) \cdot op(B) + \beta \cdot C + bias) * scale
 *  \f]
 *
 *  \note
 *  This function is non blocking and executed asynchronously with respect to the host.
 *  It may return before the actual computation has finished.
 *
 *  \note
 *  Currently, only supports the case where D has the same shape of C.
 *
 *  @param[in]
 *  handle      hipsparselt library handle
 *  @param[in]
 *  plan        Matrix multiplication plan
 *  @param[in]
 *  alpha       scalar \f$\alpha\f$. (float)
 *  @param[in]
 *  d_A         Pointer to the structured matrix A
 *  @param[in]
 *  d_B         Pointer to the dense matrix B
 *  @param[in]
 *  beta        scalar \f$\beta\f$. (float)
 *  @param[in]
 *  d_C         Pointer to the dense matrix C
 *  @param[out]
 *  d_D         Pointer to the dense matrix D
 *  @param[in]
 *  workspace   Pointor to the worksapce
 *  @param[in]
 *  streams     Pointer to HIP stream array for the computation
 *  @param[in]
 *  numStreams  Number of HIP streams in \p streams

 *  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
 *  \retval     HIPSPARSE_STATUS_NOT_INITIALIZED \p handle or \p plan is invalid.
 *  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p plan \p alpha, \p d_A, \p d_B, \p beta, \p d_C , \p d_D , \p workspace \p streams or \p numStreams is invalid.
 *  \retval     HIPSPARSE_STATUS_NOT_SUPPORTED the problme is not supported.
 */
HIPSPARSELT_EXPORT
hipsparseStatus_t hipsparseLtMatmul(const hipsparseLtHandle_t*     handle,
                                    const hipsparseLtMatmulPlan_t* plan,
                                    const void*                    alpha,
                                    const void*                    d_A,
                                    const void*                    d_B,
                                    const void*                    beta,
                                    const void*                    d_C,
                                    void*                          d_D,
                                    void*                          workspace,
                                    hipStream_t*                   streams,
                                    int32_t                        numStreams);

/*! \ingroup matmul_module
 *  \brief Sparse matrix dense matrix multiplication
 *
 *  \details
 *  \p hipsparseLtMatmulSearch evaluates all available algorithms for the matrix multiplication
 *  and automatically updates the plan by selecting the fastest one.
 *  The functionality is intended to be used for auto-tuning purposes when the same operation
 *  is repeated multiple times over different inputs.
 *
 *  \note
 *  This function's behavior is the same of \ref hipsparseLtMatmul
 *
 *  \note
 *  d_C and d_D must be two different memory buffers, otherwise the output will be incorrect.
 *
 *  \note
 *  This function is NOT asynchronous with respect to streams[0] (blocking call)
 *
 *  \note
 *  The number of iterations for the evaluation can be set by using
 *  hipsparseLtMatmulAlgSetAttribute() with HIPSPARSELT_MATMUL_SEARCH_ITERATIONS.
 *
 *  \note
 *	The selected algorithm id can be retrieved by using
 *
 *
 *  @param[in]
 *  handle      hipsparselt library handle
 *  @param[in]
 *  plan        Matrix multiplication plan
 *  @param[in]
 *  alpha       scalar \f$\alpha\f$. (float)
 *  @param[in]
 *  d_A         Pointer to the structured matrix A
 *  @param[in]
 *  d_B         Pointer to the dense matrix B
 *  @param[in]
 *  beta        scalar \f$\beta\f$. (float)
 *  @param[in]
 *  d_C         Pointer to the dense matrix C
 *  @param[out]
 *  d_D         Pointer to the dense matrix D
 *  @param[in]
 *  workspace   Pointor to the worksapce
 *  @param[in]
 *  streams     Pointer to HIP stream array for the computation
 *  @param[in]
 *  numStreams  Number of HIP streams in \p streams
 *
 *  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
 *  \retval     HIPSPARSE_STATUS_NOT_INITIALIZED \p handle or \p plan is invalid.
 *  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p plan \p alpha, \p d_A, \p d_B, \p beta, \p d_C , \p d_D , \p workspace \p streams or \p numStreams is invalid.
 *  \retval     HIPSPARSE_STATUS_NOT_SUPPORTED the problme is not supported.
 */
HIPSPARSELT_EXPORT
hipsparseStatus_t hipsparseLtMatmulSearch(const hipsparseLtHandle_t* handle,
                                          hipsparseLtMatmulPlan_t*   plan,
                                          const void*                alpha,
                                          const void*                d_A,
                                          const void*                d_B,
                                          const void*                beta,
                                          const void*                d_C,
                                          void*                      d_D,
                                          void*                      workspace,
                                          hipStream_t*               streams,
                                          int32_t                    numStreams);

/* helper */
// prune
/*! \ingroup helper_module
 *  \brief Purnes a dense matrix.
 *
 *  \details
 *  \p hipsparseLtSpMMAPrune prunes a dense matrix d_in according to the specified
 *  algorithm pruneAlg, HIPSPARSELT_PRUNE_SPMMA_TILE or HIPSPARSELT_PRUNE_SPMMA_STRIP.
 *
 *  \note
 *	The function requires no extra storage.
 *
 *  \note
 *  This function supports asynchronous execution with respect to stream.
 *
 *  @param[in]
 *  handle      hipsparselt library handle
 *  @param[in]
 *  matmulDescr matrix multiplication descriptor.
 *  @param[in]
 *  d_in        pointer to the dense matrix.
 *  @param[out]
 *  d_out       pointer to the pruned matrix.
 *  @param[in]
 *  pruneAlg    pruning algorithm.
 *  @param[in]
 *  stream      HIP stream for the computation.
 *
 *  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
 *  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle , \p matmulDescr , \p d_in or \p d_out is invalid.
 */
HIPSPARSELT_EXPORT
hipsparseStatus_t hipsparseLtSpMMAPrune(const hipsparseLtHandle_t*           handle,
                                        const hipsparseLtMatmulDescriptor_t* matmulDescr,
                                        const void*                          d_in,
                                        void*                                d_out,
                                        hipsparseLtPruneAlg_t                pruneAlg,
                                        hipStream_t                          stream);

/*! \ingroup helper_module
 *  \brief checks the correctness of the pruning structure for a given matrix.
 *
 *  \details
 *  \p hipsparseLtSpMMAPruneCheck checks the correctness of the pruning structure for a given matrix.
 *  Contents in the given matrix must be sparsity 2:4.
 *
 *
 *  @param[in]
 *  handle      hipsparselt library handle
 *  @param[in]
 *  matmulDescr matrix multiplication descriptor.
 *  @param[in]
 *  d_in        pointer to the matrix to check.
 *  @param[out]
 *  d_valid     validation results (0 correct, 1 wrong).
 *  @param[in]
 *  stream      HIP stream for the computation.
 *
 *  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
 *  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle , \p matmulDescr , \p d_in or \p d_valid is invalid.
 */
HIPSPARSELT_EXPORT
hipsparseStatus_t hipsparseLtSpMMAPruneCheck(const hipsparseLtHandle_t*           handle,
                                             const hipsparseLtMatmulDescriptor_t* matmulDescr,
                                             const void*                          d_in,
                                             int*                                 d_valid,
                                             hipStream_t                          stream);

/*! \ingroup helper_module
 *  \brief Purnes a dense matrix.
 *
 *  \details
 *  \p hipsparseLtSpMMAPrune2 prunes a dense matrix d_in according to the specified
 *  algorithm pruneAlg, HIPSPARSELT_PRUNE_SPMMA_TILE or HIPSPARSELT_PRUNE_SPMMA_STRIP.
 *
 *  \note
 *	The function requires no extra storage.
 *
 *  \note
 *  This function supports asynchronous execution with respect to stream.
 *
 *  @param[in]
 *  handle         hipsparselt library handle
 *  @param[in]
 *  sparseMatDescr structured(sparse) matrix descriptor.
 *  @param[in]
 *  isSparseA      specify if the structured (sparse) matrix is in the first position (matA or matB) (Currently, only support matA)
 *  @param[in]
 *  op             operation that will be applied to the structured (sparse) matrix in the multiplication
 *  @param[in]
 *  d_in           pointer to the dense matrix.
 *  @param[out]
 *  d_out          pointer to the pruned matrix.
 *  @param[in]
 *  pruneAlg       pruning algorithm.
 *  @param[in]
 *  stream         HIP stream for the computation.
 *
 *  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
 *  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle , \p sparseMatDescr , \p op , \p d_in  or \p d_out is invalid.
 *  \retval     HIPSPARSE_STATUS_NOT_SUPPORTED the problem is not support
 */
HIPSPARSELT_EXPORT
hipsparseStatus_t hipsparseLtSpMMAPrune2(const hipsparseLtHandle_t*        handle,
                                         const hipsparseLtMatDescriptor_t* sparseMatDescr,
                                         int                               isSparseA,
                                         hipsparseOperation_t              op,
                                         const void*                       d_in,
                                         void*                             d_out,
                                         hipsparseLtPruneAlg_t             pruneAlg,
                                         hipStream_t                       stream);

/*! \ingroup helper_module
 *  \brief checks the correctness of the pruning structure for a given matrix.
 *
 *  \details
 *  \p hipsparseLtSpMMAPruneCheck2 checks the correctness of the pruning structure for a given matrix.
 *  Contents in the given matrix must be sparsity 2:4.
 *
 *  @param[in]
 *  handle         hipsparselt library handle
 *  @param[in]
 *  sparseMatDescr structured(sparse) matrix descriptor.
 *  @param[in]
 *  isSparseA      specify if the structured (sparse) matrix is in the first position (matA or matB) (HIP backend only support matA)
 *  @param[in]
 *  op             operation that will be applied to the structured (sparse) matrix in the multiplication
 *  @param[in]
 *  d_in           pointer to the matrix to check.
 *  @param[out]
 *  d_valid        validation results (0 correct, 1 wrong).
 *  @param[in]
 *  stream         HIP stream for the computation.
 *
 *  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
 *  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle , \p sparseMatDescr , \p op , \p d_in or \p d_valid is invalid.
 *  \retval     HIPSPARSE_STATUS_NOT_SUPPORTED the problem is not support
 */
HIPSPARSELT_EXPORT
hipsparseStatus_t hipsparseLtSpMMAPruneCheck2(const hipsparseLtHandle_t*        handle,
                                              const hipsparseLtMatDescriptor_t* sparseMatDescr,
                                              int                               isSparseA,
                                              hipsparseOperation_t              op,
                                              const void*                       d_in,
                                              int*                              d_valid,
                                              hipStream_t                       stream);

// compression
/*! \ingroup helper_module
 *  \brief provide the size of the compressed matrix.
 *
 *  \details
 *  \p HIPSPARSE_STATUS_INVALID_VALUE provides the size of the compressed matrix
 *  to be allocated before calling \ref hipsparseLtSpMMACompress() or \ref hipsparseLtSpMMACompress2().
 *
 *  @param[in]
 *  handle                hipsparselt library handle
 *  @param[in]
 *  plan                  matrix multiplication plan descriptor.
 *  @param[out]
 *  compressedSize        size in bytes of the compressed matrix.
 *  @param[out]
 *  compressBufferSize    size in bytes for the buffer needed for the matrix compression.
 *
 *  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
 *  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle , \p plan , \p compressedSize or \p compressBufferSize is invalid.
 */
HIPSPARSELT_EXPORT
hipsparseStatus_t hipsparseLtSpMMACompressedSize(const hipsparseLtHandle_t*     handle,
                                                 const hipsparseLtMatmulPlan_t* plan,
                                                 size_t*                        compressedSize,
                                                 size_t*                        compressBufferSize);

/*! \ingroup helper_module
 *  \brief compresses a dense matrix to structured matrix.
 *
 *  \details
 *  \p hipsparseLtSpMMACompress compresses a dense matrix d_dense.
 *  The compressed matrix is intended to be used as the first/second operand A/B
 *  in the \ref hipsparseLtMatmul() function.
 *
 *  @param[in]
 *  handle             handle to the hipsparselt library context queue.
 *  @param[in]
 *  plan               matrix multiplication plan descriptor.
 *  @param[in]
 *  d_dense            pointer to the dense matrix.
 *  @param[out]
 *  d_compressed       compressed matrix and metadata.
 *  @param[out]
 *  d_compressBuffer   temporary buffer for the compression.
 *  @param[in]
 *  stream             HIP stream for the computation.
 *
 *  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
 *  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle , \p plan , \p d_dense or \p d_compressed is invalid.
 *  \retval     HIPSPARSE_STATUS_NOT_SUPPORTED the problem is not support
 */
HIPSPARSELT_EXPORT
hipsparseStatus_t hipsparseLtSpMMACompress(const hipsparseLtHandle_t*     handle,
                                           const hipsparseLtMatmulPlan_t* plan,
                                           const void*                    d_dense,
                                           void*                          d_compressed,
                                           void*                          d_compressBuffer,
                                           hipStream_t                    stream);

/*! \ingroup helper_module
 *  \brief provide the size of the compressed matrix.
 *
 *  \details
 *  \p hipsparseLtSpMMACompressedSize2 provides the size of the compressed matrix
 *  to be allocated before calling \ref hipsparseLtSpMMACompress or \ref hipsparseLtSpMMACompress2
 *
 *  @param[in]
 *  handle                hipsparselt library handle
 *  @param[in]
 *  sparseMatDescr        structured(sparse) matrix descriptor.
 *  @param[out]
 *  compressedSize        size in bytes of the compressed matrix.
 *  @param[out]
 *  compressBufferSize    size in bytes for the buffer needed for the matrix compression.
 *
 *  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
 *  \retval     HIPSPARSE_STATUS_NOT_INITIALIZED \p handle , \p sparseMatDescr , \p compressedSize or \p compressBufferSize is invalid.
 *  \retval     HIPSPARSE_STATUS_NOT_SUPPORTED the problem is not support
 */
HIPSPARSELT_EXPORT
hipsparseStatus_t hipsparseLtSpMMACompressedSize2(const hipsparseLtHandle_t*        handle,
                                                  const hipsparseLtMatDescriptor_t* sparseMatDescr,
                                                  size_t*                           compressedSize,
                                                  size_t* compressBufferSize);

/*! \ingroup helper_module
 *  \brief compresses a dense matrix to structured matrix.
 *
 *  \details
 *  \p hipsparseLtSpMMACompress2 compresses a dense matrix d_dense.
 *  The compressed matrix is intended to be used as the first/second operand A/B
 *  in the \ref hipsparseLtMatmul() function.
 *
 *  @param[in]
 *  handle             handle to the hipsparselt library context queue.
 *  @param[in]
 *  sparseMatDescr     structured(sparse) matrix descriptor.
 *  @param[in]
 *  isSparseA          specify if the structured (sparse) matrix is in the first position (matA or matB) (HIP backend only support matA)
 *  @param[in]
 *  op                 operation that will be applied to the structured (sparse) matrix in the multiplication
 *  @param[in]
 *  d_dense            pointer to the dense matrix.
 *  @param[out]
 *  d_compressed       compressed matrix and metadata
 *  @param[out]
 *  d_compressBuffer   temporary buffer for the compression.
 *  @param[in]
 *  stream             HIP stream for the computation.
 *
 *  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
 *  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle , \p sparseMatDescr , \p op , \p d_dense or \p d_compressed is invalid.
 *  \retval     HIPSPARSE_STATUS_NOT_SUPPORTED the problem is not support
 */
HIPSPARSELT_EXPORT
hipsparseStatus_t hipsparseLtSpMMACompress2(const hipsparseLtHandle_t*        handle,
                                            const hipsparseLtMatDescriptor_t* sparseMatDescr,
                                            int                               isSparseA,
                                            hipsparseOperation_t              op,
                                            const void*                       d_dense,
                                            void*                             d_compressed,
                                            void*                             d_compressBuffer,
                                            hipStream_t                       stream);

#ifdef __cplusplus
}
#endif

#endif // _HIPSPARSELT_H_
