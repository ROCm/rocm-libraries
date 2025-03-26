/* ************************************************************************
* Copyright (C) 2018-2025 Advanced Micro Devices, Inc. All rights Reserved.
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
*  \brief hipsparse.h provides Sparse Linear Algebra Subprograms
*  of Level 1, 2 and 3, using HIP optimized for AMD GPU hardware.
*/

// HIP = Heterogeneous-compute Interface for Portability
//
// Define a extremely thin runtime layer that allows source code to be compiled
// unmodified through either AMD HCC or NVCC. Key features tend to be in the spirit
// and terminology of CUDA, but with a portable path to other accelerators as well.
//
// This is the master include file for hipSPARSE, wrapping around rocSPARSE and
// cuSPARSE "version 2".

#ifndef HIPSPARSE_H
#define HIPSPARSE_H

#include "hipsparse-export.h"
#include "hipsparse-version.h"

#include <hip/hip_complex.h>
#include <hip/hip_runtime.h>

/// \cond DO_NOT_DOCUMENT
#define DEPRECATED_CUDA_12000(warning)
#define DEPRECATED_CUDA_11000(warning)
#define DEPRECATED_CUDA_10000(warning)
#define DEPRECATED_CUDA_9000(warning)

#ifdef __cplusplus
#ifndef __has_cpp_attribute
#define __has_cpp_attribute(X) 0
#endif
#define HIPSPARSE_HAS_DEPRECATED_MSG __has_cpp_attribute(deprecated) >= 201309L
#else
#ifndef __has_c_attribute
#define __has_c_attribute(X) 0
#endif
#define HIPSPARSE_HAS_DEPRECATED_MSG __has_c_attribute(deprecated) >= 201904L
#endif

#if HIPSPARSE_HAS_DEPRECATED_MSG
#define HIPSPARSE_DEPRECATED_MSG(MSG) [[deprecated(MSG)]]
#else
#define HIPSPARSE_DEPRECATED_MSG(MSG) HIPSPARSE_DEPRECATED // defined in hipsparse-export.h
#endif
/// \endcond

#if defined(CUDART_VERSION)
#if CUDART_VERSION < 10000
#undef DEPRECATED_CUDA_9000
#define DEPRECATED_CUDA_9000(warning) HIPSPARSE_DEPRECATED_MSG(warning)
#elif CUDART_VERSION < 11000
#undef DEPRECATED_CUDA_10000
#define DEPRECATED_CUDA_10000(warning) HIPSPARSE_DEPRECATED_MSG(warning)
#elif CUDART_VERSION < 12000
#undef DEPRECATED_CUDA_11000
#define DEPRECATED_CUDA_11000(warning) HIPSPARSE_DEPRECATED_MSG(warning)
#elif CUDART_VERSION < 13000
#undef DEPRECATED_CUDA_12000
#define DEPRECATED_CUDA_12000(warning) HIPSPARSE_DEPRECATED_MSG(warning)
#endif
#endif

/// \cond DO_NOT_DOCUMENT
// Forward declarations
struct bsrsv2Info;
struct bsrsm2Info;
struct bsrilu02Info;
struct bsric02Info;
struct csrsv2Info;
struct csrsm2Info;
struct csrilu02Info;
struct csric02Info;
struct csrgemm2Info;
struct pruneInfo;
struct csru2csrInfo;
/// \endcond

/*! \ingroup types_module
 *  \brief Handle to the hipSPARSE library context queue.
 *
 *  \details
 *  The hipSPARSE handle is a structure holding the hipSPARSE library context. It must
 *  be initialized using hipsparseCreate() and the returned handle must be passed to all
 *  subsequent library function calls. It should be destroyed at the end using
 *  hipsparseDestroy().
 */
typedef void* hipsparseHandle_t;

/*! \ingroup types_module
 *  \brief Descriptor of the matrix.
 *
 *  \details
 *  The hipSPARSE matrix descriptor is a structure holding all properties of a matrix.
 *  It must be initialized using hipsparseCreateMatDescr() and the returned descriptor
 *  must be passed to all subsequent library calls that involve the matrix. It should be
 *  destroyed at the end using hipsparseDestroyMatDescr().
 */
typedef void* hipsparseMatDescr_t;

/*! \ingroup types_module
 *  \brief HYB matrix storage format.
 *
 *  \details
 *  The hipSPARSE HYB matrix structure holds the HYB matrix. It must be initialized using
 *  hipsparseCreateHybMat() and the returned HYB matrix must be passed to all subsequent
 *  library calls that involve the matrix. It should be destroyed at the end using
 *  hipsparseDestroyHybMat().
 */
typedef void* hipsparseHybMat_t;

/*! \ingroup types_module
 *  \brief Pointer type to opaque structure holding coloring info.
 *
 *  \details
 *  The hipSPARSE ColorInfo structure holds the coloring information. It must be
 *  initialized using hipsparseCreateColorInfo() and the returned structure must be
 *  passed to all subsequent library calls that involve the coloring. It should be
 *  destroyed at the end using hipsparseDestroyColorInfo().
 */
typedef void* hipsparseColorInfo_t;

/*! \ingroup types_module
 *  \brief Pointer type to opaque structure holding bsrsv2 info.
 *
 *  \details
 *  The hipSPARSE bsrsv2 structure holds the information used by hipsparseXbsrsv2_zeroPivot(), 
 *  \ref hipsparseSbsrsv2_bufferSize "hipsparseXbsrsv2_bufferSize()", \ref hipsparseSbsrsv2_bufferSizeExt 
 *  "hipsparseXbsrsv2_bufferSizeExt()", \ref hipsparseSbsrsv2_analysis "hipsparseXbsrsv2_analysis()", 
 *  and \ref hipsparseSbsrsv2_solve "hipsparseXbsrsv2_solve()". It must be initialized using 
 *  hipsparseCreateBsrsv2Info() and the returned structure must be passed to all subsequent library calls 
 *  that involve bsrsv2. It should be destroyed at the end using hipsparseDestroyBsrsv2Info().
 */
typedef struct bsrsv2Info* bsrsv2Info_t;

/*! \ingroup types_module
 *  \brief Pointer type to opaque structure holding bsrsm2 info.
 *
 *  \details
 *  The hipSPARSE bsrsm2 structure holds the information used by hipsparseXbsrsm2_zeroPivot(), 
 *  \ref hipsparseSbsrsm2_bufferSize "hipsparseXbsrsm2_bufferSize()", \ref hipsparseSbsrsm2_analysis 
 *  "hipsparseXbsrsm2_analysis()", and \ref hipsparseSbsrsm2_solve "hipsparseXbsrsm2_solve()". It 
 *  must be initialized using hipsparseCreateBsrsm2Info() and the returned structure must be
 *  passed to all subsequent library calls that involve bsrsm2. It should be
 *  destroyed at the end using hipsparseDestroyBsrsm2Info().
 */
typedef struct bsrsm2Info* bsrsm2Info_t;

/*! \ingroup types_module
 *  \brief Pointer type to opaque structure holding bsrilu02 info.
 *
 *  \details
 *  The hipSPARSE bsrilu02 structure holds the information used by hipsparseXbsrilu02_zeroPivot(), 
 *  \ref hipsparseSbsrilu02_numericBoost "hipsparseXbsrilu02_numericBoost()", \ref hipsparseSbsrilu02_bufferSize 
 *  "hipsparseXbsrilu02_bufferSize()", \ref hipsparseSbsrilu02_analysis "hipsparseXbsrilu02_analysis()", 
 *  and \ref hipsparseSbsrilu02 "hipsparseXbsrilu02()". It must be initialized using hipsparseCreateBsrilu02Info() 
 *  and the returned structure must be passed to all subsequent library calls that involve bsrilu02. It should be
 *  destroyed at the end using hipsparseDestroyBsrilu02Info().
 */
typedef struct bsrilu02Info* bsrilu02Info_t;

/*! \ingroup types_module
 *  \brief Pointer type to opaque structure holding bsric02 info.
 *
 *  \details
 *  The hipSPARSE bsric02 structure holds the information used by hipsparseXbsric02_zeroPivot(), 
 *  \ref hipsparseSbsric02_bufferSize "hipsparseXbsric02_bufferSize()", \ref hipsparseSbsric02_analysis 
 *  "hipsparseXbsric02_analysis()", and \ref hipsparseSbsric02 "hipsparseXbsric02()". It must be initialized using 
 *  hipsparseCreateBsric02Info() and the returned structure must be passed to all subsequent library calls 
 *  that involve bsric02. It should be destroyed at the end using hipsparseDestroyBsric02Info().
 */
typedef struct bsric02Info* bsric02Info_t;

/*! \ingroup types_module
 *  \brief Pointer type to opaque structure holding csrsv2 info.
 *
 *  \details
 *  The hipSPARSE csrsv2 structure holds the information used by hipsparseXcsrsv2_zeroPivot(), 
 *  \ref hipsparseScsrsv2_bufferSize "hipsparseXcsrsv2_bufferSize()", \ref hipsparseScsrsv2_analysis
 *  "hipsparseXcsrsv2_analysis()", and \ref hipsparseScsrsv2_solve "hipsparseXcsrsv2_solve()". It must be initialized using 
 *  hipsparseCreateCsrsv2Info() and the returned structure must be passed to all subsequent library calls 
 *  that involve csrsv2. It should be destroyed at the end using hipsparseDestroyCsrsv2Info().
 */
typedef struct csrsv2Info* csrsv2Info_t;

/*! \ingroup types_module
 *  \brief Pointer type to opaque structure holding csrsm2 info.
 *
 *  \details
 *  The hipSPARSE csrsm2 structure holds the information used by hipsparseXcsrsm2_zeroPivot(), 
 *  \ref hipsparseScsrsm2_bufferSizeExt "hipsparseXcsrsm2_bufferSizeExt()", \ref hipsparseScsrsm2_analysis 
 *  "hipsparseXcsrsm2_analysis()", and \ref hipsparseScsrsm2_solve "hipsparseXcsrsm2_solve()". It must be initialized using 
 *  hipsparseCreateCsrsm2Info() and the returned structure must be passed to all subsequent library calls 
 *  that involve csrsm2. It should be destroyed at the end using hipsparseDestroyCsrsm2Info().
 */
typedef struct csrsm2Info* csrsm2Info_t;

/*! \ingroup types_module
 *  \brief Pointer type to opaque structure holding csrilu02 info.
 *
 *  \details
 *  The hipSPARSE csrilu02 structure holds the information used by hipsparseXcsrilu02_zeroPivot(), 
 *  \ref hipsparseScsrilu02_numericBoost "hipsparseXcsrilu02_numericBoost()", \ref hipsparseScsrilu02_bufferSize 
 *  "hipsparseXcsrilu02_bufferSize()", \ref hipsparseScsrilu02_analysis "hipsparseXcsrilu02_analysis()", 
 *  and \ref hipsparseScsrilu02 "hipsparseXcsrilu02()". It must be initialized using hipsparseCreateCsrilu02Info() 
 *  and the returned structure must be passed to all subsequent library calls that involve csrilu02. It should be
 *  destroyed at the end using hipsparseDestroyCsrilu02Info().
 */
typedef struct csrilu02Info* csrilu02Info_t;

/*! \ingroup types_module
 *  \brief Pointer type to opaque structure holding csric02 info.
 *
 *  \details
 *  The hipSPARSE csric02 structure holds the information used by hipsparseXcsric02_zeroPivot(), 
 *  \ref hipsparseScsric02_bufferSize "hipsparseXcsric02_bufferSize()", \ref hipsparseScsric02_analysis 
 *  "hipsparseXcsric02_analysis()", and \ref hipsparseScsric02 "hipsparseXcsric02()". It must be 
 *  initialized using hipsparseCreateCsric02Info() and the returned structure must be passed to all 
 *  subsequent library calls that involve csric02. It should be destroyed at the end using 
 *  hipsparseDestroyCsric02Info().
 */
typedef struct csric02Info* csric02Info_t;

/*! \ingroup types_module
 *  \brief Pointer type to opaque structure holding csrgemm2 info.
 *
 *  \details
 *  The hipSPARSE csrgemm2 structure holds the information used by \ref hipsparseScsrgemm2_bufferSizeExt 
 *  "hipsparseXcsrgemm2_bufferSizeExt()", hipsparseXcsrgemm2Nnz(), and \ref hipsparseScsrgemm2 "hipsparseXcsrgemm2()". 
 *  It must be initialized using hipsparseCreateCsrgemm2Info() and the returned structure must be passed to all 
 *  subsequent library calls that involve csrgemm2. It should be destroyed at the end using 
 *  hipsparseDestroyCsrgemm2Info().
 */
typedef struct csrgemm2Info* csrgemm2Info_t;

/*! \ingroup types_module
 *  \brief Pointer type to opaque structure holding prune info.
 *
 *  \details
 *  The hipSPARSE prune structure holds the information used by \ref hipsparseSpruneDense2csrByPercentage_bufferSize 
 *  "hipsparseXpruneDense2csrByPercentage_bufferSize()", \ref hipsparseSpruneDense2csrByPercentage_bufferSizeExt
 *  "hipsparseXpruneDense2csrByPercentage_bufferSizeExt()", \ref hipsparseSpruneCsr2csrByPercentage_bufferSize 
 *  "hipsparseXpruneCsr2csrByPercentage_bufferSize()", \ref hipsparseSpruneCsr2csrByPercentage_bufferSizeExt 
 *  "hipsparseXpruneCsr2csrByPercentage_bufferSizeExt()", \ref hipsparseSpruneDense2csrNnzByPercentage 
 *  "hipsparseXpruneDense2csrNnzByPercentage()", \ref hipsparseSpruneCsr2csrNnzByPercentage 
 *  "hipsparseXpruneCsr2csrNnzByPercentage()", \ref hipsparseSpruneDense2csrByPercentage 
 *  "hipsparseXpruneDense2csrByPercentage()", and \ref hipsparseSpruneCsr2csrByPercentage 
 *  "hipsparseXpruneCsr2csrByPercentage()". It must be initialized using hipsparseCreatePruneInfo() and the 
 *  returned structure must be passed to all subsequent library calls that involve prune. It should be 
 *  destroyed at the end using hipsparseDestroyPruneInfo().
 */
typedef struct pruneInfo* pruneInfo_t;

/*! \ingroup types_module
 *  \brief Pointer type to opaque structure holding csru2csr info.
 *
 *  \details
 *  The hipSPARSE csru2csr structure holds the information used by \ref hipsparseScsru2csr_bufferSizeExt 
 *  "hipsparseXcsru2csr_bufferSizeExt()", \ref hipsparseScsru2csr "hipsparseXcsru2csr()", and 
 *  \ref hipsparseScsr2csru "hipsparseXcsr2csru()". It must be initialized using hipsparseCreateCsru2csrInfo() 
 *  and the returned structure must be passed to all subsequent library calls that involve csru2csr. It should be 
 *  destroyed at the end using hipsparseDestroyCsru2csrInfo().
 */
typedef struct csru2csrInfo* csru2csrInfo_t;

// clang-format off

/*! \ingroup types_module
 *  \brief List of hipsparse status codes definition.
 *
 *  \details
 *  This is a list of the \ref hipsparseStatus_t types that are used by the hipSPARSE
 *  library.
 */
#if(!defined(CUDART_VERSION))
typedef enum {
    HIPSPARSE_STATUS_SUCCESS                   = 0, /**< Function succeeds */
    HIPSPARSE_STATUS_NOT_INITIALIZED           = 1, /**< hipSPARSE was not initialized */
    HIPSPARSE_STATUS_ALLOC_FAILED              = 2, /**< Resource allocation failed */
    HIPSPARSE_STATUS_INVALID_VALUE             = 3, /**< Unsupported value was passed to the function */
    HIPSPARSE_STATUS_ARCH_MISMATCH             = 4, /**< Device architecture not supported */
    HIPSPARSE_STATUS_MAPPING_ERROR             = 5, /**< Access to GPU memory space failed */
    HIPSPARSE_STATUS_EXECUTION_FAILED          = 6, /**< GPU program failed to execute */
    HIPSPARSE_STATUS_INTERNAL_ERROR            = 7, /**< An internal hipSPARSE operation failed */
    HIPSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED = 8, /**< Matrix type not supported */
    HIPSPARSE_STATUS_ZERO_PIVOT                = 9, /**< Zero pivot was computed */
    HIPSPARSE_STATUS_NOT_SUPPORTED             = 10, /**< Operation is not supported */
    HIPSPARSE_STATUS_INSUFFICIENT_RESOURCES    = 11 /**< Resources are insufficient */
} hipsparseStatus_t;
#else
#if(CUDART_VERSION >= 11003)
typedef enum {
    HIPSPARSE_STATUS_SUCCESS                   = 0, /**< Function succeeds */
    HIPSPARSE_STATUS_NOT_INITIALIZED           = 1, /**< hipSPARSE was not initialized */
    HIPSPARSE_STATUS_ALLOC_FAILED              = 2, /**< Resource allocation failed */
    HIPSPARSE_STATUS_INVALID_VALUE             = 3, /**< Unsupported value was passed to the function */
    HIPSPARSE_STATUS_ARCH_MISMATCH             = 4, /**< Device architecture not supported */
    HIPSPARSE_STATUS_MAPPING_ERROR             = 5, /**< Access to GPU memory space failed */
    HIPSPARSE_STATUS_EXECUTION_FAILED          = 6, /**< GPU program failed to execute */
    HIPSPARSE_STATUS_INTERNAL_ERROR            = 7, /**< An internal hipSPARSE operation failed */
    HIPSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED = 8, /**< Matrix type not supported */
    HIPSPARSE_STATUS_ZERO_PIVOT                = 9, /**< Zero pivot was computed */
    HIPSPARSE_STATUS_NOT_SUPPORTED             = 10, /**< Operation is not supported */
    HIPSPARSE_STATUS_INSUFFICIENT_RESOURCES    = 11 /**< Resources are insufficient */
} hipsparseStatus_t;
#elif(CUDART_VERSION >= 10010)
typedef enum {
    HIPSPARSE_STATUS_SUCCESS                   = 0, /**< Function succeeds */
    HIPSPARSE_STATUS_NOT_INITIALIZED           = 1, /**< hipSPARSE was not initialized */
    HIPSPARSE_STATUS_ALLOC_FAILED              = 2, /**< Resource allocation failed */
    HIPSPARSE_STATUS_INVALID_VALUE             = 3, /**< Unsupported value was passed to the function */
    HIPSPARSE_STATUS_ARCH_MISMATCH             = 4, /**< Device architecture not supported */
    HIPSPARSE_STATUS_MAPPING_ERROR             = 5, /**< Access to GPU memory space failed */
    HIPSPARSE_STATUS_EXECUTION_FAILED          = 6, /**< GPU program failed to execute */
    HIPSPARSE_STATUS_INTERNAL_ERROR            = 7, /**< An internal hipSPARSE operation failed */
    HIPSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED = 8, /**< Matrix type not supported */
    HIPSPARSE_STATUS_ZERO_PIVOT                = 9, /**< Zero pivot was computed */
    HIPSPARSE_STATUS_NOT_SUPPORTED             = 10 /**< Operation is not supported */
} hipsparseStatus_t;
#endif
#endif

/*! \ingroup types_module
 *  \brief Indicates if the pointer is device pointer or host pointer.
 *
 *  \details
 *  The \ref hipsparsePointerMode_t indicates whether scalar values are passed by
 *  reference on the host or device. The \ref hipsparsePointerMode_t can be changed by
 *  hipsparseSetPointerMode(). The currently used pointer mode can be obtained by
 *  hipsparseGetPointerMode().
 */
typedef enum {
    HIPSPARSE_POINTER_MODE_HOST   = 0, /**< Scalar pointers are in host memory */
    HIPSPARSE_POINTER_MODE_DEVICE = 1 /**< Scalar pointers are in device memory */
} hipsparsePointerMode_t;

/*! \ingroup types_module
 *  \brief Specify where the operation is performed on.
 *
 *  \details
 *  The \ref hipsparseAction_t indicates whether the operation is performed on the full
 *  matrix, or only on the sparsity pattern of the matrix.
 */
typedef enum {
    HIPSPARSE_ACTION_SYMBOLIC = 0, /**< Operate only on indices */
    HIPSPARSE_ACTION_NUMERIC  = 1 /**< Operate on data and indices */
} hipsparseAction_t;

/*! \ingroup types_module
 *  \brief Specify the matrix type.
 *
 *  \details
 *  The \ref hipsparseMatrixType_t indices the type of a matrix. For a given
 *  \ref hipsparseMatDescr_t, the \ref hipsparseMatrixType_t can be set using
 *  hipsparseSetMatType(). The current \ref hipsparseMatrixType_t of a matrix can be
 *  obtained by hipsparseGetMatType().
 */
typedef enum {
    HIPSPARSE_MATRIX_TYPE_GENERAL    = 0, /**< General matrix type */
    HIPSPARSE_MATRIX_TYPE_SYMMETRIC  = 1, /**< Symmetric matrix type */
    HIPSPARSE_MATRIX_TYPE_HERMITIAN  = 2, /**< Hermitian matrix type */
    HIPSPARSE_MATRIX_TYPE_TRIANGULAR = 3 /**< Triangular matrix type */
} hipsparseMatrixType_t;

/*! \ingroup types_module
 *  \brief Specify the matrix fill mode.
 *
 *  \details
 *  The \ref hipsparseFillMode_t indicates whether the lower or the upper part is stored
 *  in a sparse triangular matrix. For a given \ref hipsparseMatDescr_t, the
 *  \ref hipsparseFillMode_t can be set using hipsparseSetMatFillMode(). The current
 *  \ref hipsparseFillMode_t of a matrix can be obtained by hipsparseGetMatFillMode().
 */
typedef enum {
    HIPSPARSE_FILL_MODE_LOWER = 0, /**< Lower triangular part is stored */
    HIPSPARSE_FILL_MODE_UPPER = 1 /**< Upper triangular part is stored */
} hipsparseFillMode_t;

/*! \ingroup types_module
 *  \brief Indicates if the diagonal entries are unity.
 *
 *  \details
 *  The \ref hipsparseDiagType_t indicates whether the diagonal entries of a matrix are
 *  unity or not. If \ref HIPSPARSE_DIAG_TYPE_UNIT is specified, all present diagonal
 *  values will be ignored. For a given \ref hipsparseMatDescr_t, the
 *  \ref hipsparseDiagType_t can be set using hipsparseSetMatDiagType(). The current
 *  \ref hipsparseDiagType_t of a matrix can be obtained by hipsparseGetMatDiagType().
 */
typedef enum {
    HIPSPARSE_DIAG_TYPE_NON_UNIT = 0, /**< Diagonal entries are non-unity */
    HIPSPARSE_DIAG_TYPE_UNIT     = 1  /**< Diagonal entries are unity */
} hipsparseDiagType_t;

/*! \ingroup types_module
 *  \brief Specify the matrix index base.
 *
 *  \details
 *  The \ref hipsparseIndexBase_t indicates the index base of the indices. For a
 *  given \ref hipsparseMatDescr_t, the \ref hipsparseIndexBase_t can be set using
 *  hipsparseSetMatIndexBase(). The current \ref hipsparseIndexBase_t of a matrix
 *  can be obtained by hipsparseGetMatIndexBase().
 */
typedef enum {
    HIPSPARSE_INDEX_BASE_ZERO = 0, /**< Zero based indexing */
    HIPSPARSE_INDEX_BASE_ONE  = 1  /**< One based indexing */
} hipsparseIndexBase_t;

/*! \ingroup types_module
 *  \brief Specify whether the matrix is to be transposed or not.
 *
 *  \details
 *  The \ref hipsparseOperation_t indicates the operation performed with the given matrix.
 */
typedef enum {
    HIPSPARSE_OPERATION_NON_TRANSPOSE       = 0, /**< Operate with matrix */
    HIPSPARSE_OPERATION_TRANSPOSE           = 1, /**< Operate with transpose */
    HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE = 2  /**< Operate with conj. transpose */
} hipsparseOperation_t;

/*! \ingroup types_module
 *  \brief HYB matrix partitioning type.
 *
 *  \details
 *  The \ref hipsparseHybPartition_t type indicates how the hybrid format partitioning
 *  between COO and ELL storage formats is performed.
 */
typedef enum {
    HIPSPARSE_HYB_PARTITION_AUTO = 0, /**< Automatically decide on ELL nnz per row */
    HIPSPARSE_HYB_PARTITION_USER = 1, /**< User given ELL nnz per row */
    HIPSPARSE_HYB_PARTITION_MAX  = 2  /**< Max ELL nnz per row, no COO part */
} hipsparseHybPartition_t;

/*! \ingroup types_module
 *  \brief Specify policy in triangular solvers and factorizations.
 *
 *  \details
 *  The \ref hipsparseSolvePolicy_t type indicates the solve policy for the triangular
 *  solve.
 */
typedef enum {
    HIPSPARSE_SOLVE_POLICY_NO_LEVEL  = 0, /**< No level information generated */
    HIPSPARSE_SOLVE_POLICY_USE_LEVEL = 1  /**< Generate level information */
} hipsparseSolvePolicy_t;

/// \cond DO_NOT_DOCUMENT
// Note: Add back to types.rst if we get documentation for this in the future
typedef enum {
    HIPSPARSE_SIDE_LEFT  = 0,
    HIPSPARSE_SIDE_RIGHT = 1
} hipsparseSideMode_t;
/// \endcond

/*! \ingroup types_module
 *  \brief Specify the matrix direction.
 *
 *  \details
 *  The \ref hipsparseDirection_t indicates whether a dense matrix should be parsed by
 *  rows or by columns, assuming column-major storage.
 */
typedef enum {
    HIPSPARSE_DIRECTION_ROW = 0, /**< Parse the matrix by rows */
    HIPSPARSE_DIRECTION_COLUMN = 1 /**< Parse the matrix by columns */
} hipsparseDirection_t;

// clang-format on

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup aux_module
 *  \brief Create a hipsparse handle
 *
 *  \details
 *  \p hipsparseCreate creates the hipSPARSE library context. It must be
 *  initialized before any other hipSPARSE API function is invoked and must be passed to
 *  all subsequent library function calls. The handle should be destroyed at the end
 *  using hipsparseDestroy().
 */
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCreate(hipsparseHandle_t* handle);

/*! \ingroup aux_module
 *  \brief Destroy a hipsparse handle
 *
 *  \details
 *  \p hipsparseDestroy destroys the hipSPARSE library context and releases all
 *  resources used by the hipSPARSE library.
 */
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDestroy(hipsparseHandle_t handle);

#if(!defined(CUDART_VERSION) || CUDART_VERSION > 10000)
/*! \ingroup aux_module
 *  \brief Return the string representation of a hipSPARSE status's matching backend status enum name
 *
 *  \details
 *  \p hipsparseGetErrorName takes a hipSPARSE status as input and first converts it to the matching backend 
 *  status (either rocsparse_status or cusparseStatus_t). It then returns the string representation of this status 
 *  enum name. If the status is not recognized, the function returns "Unrecognized status code".
 *
 *  For example, hipsparseGetErrorName(HIPSPARSE_STATUS_SUCCESS) on a system with a rocSPARSE backend will 
 *  return "rocsparse_status_success". On a system with a cuSPARSE backend this function would return 
 *  "CUSPARSE_STATUS_SUCCESS".
 */
HIPSPARSE_EXPORT
const char* hipsparseGetErrorName(hipsparseStatus_t status);

/*! \ingroup aux_module
 *  \brief Return the hipSPARSE status's matching backend status description as a string
 *
 *  \details
 *  \p hipsparseGetErrorString takes a hipSPARSE status as input and first converts it to the matching backend 
 *  status (either rocsparse_status or cusparseStatus_t). It then returns the string description of this status.
 *  If the status is not recognized, the function returns "Unrecognized status code".
 */
HIPSPARSE_EXPORT
const char* hipsparseGetErrorString(hipsparseStatus_t status);
#endif

/*! \ingroup aux_module
 *  \brief Get hipSPARSE version
 *
 *  \details
 *  \p hipsparseGetVersion gets the hipSPARSE library version number.
 *  - patch = version % 100
 *  - minor = version / 100 % 1000
 *  - major = version / 100000
 */
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseGetVersion(hipsparseHandle_t handle, int* version);

/*! \ingroup aux_module
 *  \brief Get hipSPARSE git revision
 *
 *  \details
 *  \p hipsparseGetGitRevision gets the hipSPARSE library git commit revision (SHA-1).
 */
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseGetGitRevision(hipsparseHandle_t handle, char* rev);

/*! \ingroup aux_module
 *  \brief Specify user defined HIP stream
 *
 *  \details
 *  \p hipsparseSetStream specifies the stream to be used by the hipSPARSE library
 *  context and all subsequent function calls.
 */
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSetStream(hipsparseHandle_t handle, hipStream_t streamId);

/*! \ingroup aux_module
 *  \brief Get current stream from library context
 *
 *  \details
 *  \p hipsparseGetStream gets the hipSPARSE library context stream which is currently
 *  used for all subsequent function calls.
 */
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseGetStream(hipsparseHandle_t handle, hipStream_t* streamId);

/*! \ingroup aux_module
 *  \brief Specify pointer mode
 *
 *  \details
 *  \p hipsparseSetPointerMode specifies the pointer mode to be used by the hipSPARSE
 *  library context and all subsequent function calls. By default, all values are passed
 *  by reference on the host. Valid pointer modes are \ref HIPSPARSE_POINTER_MODE_HOST
 *  or \ref HIPSPARSE_POINTER_MODE_DEVICE.
 */
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSetPointerMode(hipsparseHandle_t handle, hipsparsePointerMode_t mode);

/*! \ingroup aux_module
 *  \brief Get current pointer mode from library context
 *
 *  \details
 *  \p hipsparseGetPointerMode gets the hipSPARSE library context pointer mode which
 *  is currently used for all subsequent function calls.
 */
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseGetPointerMode(hipsparseHandle_t handle, hipsparsePointerMode_t* mode);

/*! \ingroup aux_module
 *  \brief Create a matrix descriptor
 *  \details
 *  \p hipsparseCreateMatDescr creates a matrix descriptor. It initializes
 *  \ref hipsparseMatrixType_t to \ref HIPSPARSE_MATRIX_TYPE_GENERAL and
 *  \ref hipsparseIndexBase_t to \ref HIPSPARSE_INDEX_BASE_ZERO. It should be destroyed
 *  at the end using hipsparseDestroyMatDescr().
 */
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCreateMatDescr(hipsparseMatDescr_t* descrA);

/*! \ingroup aux_module
 *  \brief Destroy a matrix descriptor
 *
 *  \details
 *  \p hipsparseDestroyMatDescr destroys a matrix descriptor and releases all
 *  resources used by the descriptor.
 */
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDestroyMatDescr(hipsparseMatDescr_t descrA);

/*! \ingroup aux_module
 *  \brief Copy a matrix descriptor
 *  \details
 *  \p hipsparseCopyMatDescr copies a matrix descriptor. Both, source and destination
 *  matrix descriptors must be initialized prior to calling \p hipsparseCopyMatDescr.
 */
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCopyMatDescr(hipsparseMatDescr_t dest, const hipsparseMatDescr_t src);

/*! \ingroup aux_module
 *  \brief Specify the matrix type of a matrix descriptor
 *
 *  \details
 *  \p hipsparseSetMatType sets the matrix type of a matrix descriptor. Valid
 *  matrix types are \ref HIPSPARSE_MATRIX_TYPE_GENERAL,
 *  \ref HIPSPARSE_MATRIX_TYPE_SYMMETRIC, \ref HIPSPARSE_MATRIX_TYPE_HERMITIAN or
 *  \ref HIPSPARSE_MATRIX_TYPE_TRIANGULAR.
 */
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSetMatType(hipsparseMatDescr_t descrA, hipsparseMatrixType_t type);

/*! \ingroup aux_module
 *  \brief Get the matrix type of a matrix descriptor
 *
 *  \details
 *  \p hipsparseGetMatType returns the matrix type of a matrix descriptor.
 */
HIPSPARSE_EXPORT
hipsparseMatrixType_t hipsparseGetMatType(const hipsparseMatDescr_t descrA);

/*! \ingroup aux_module
 *  \brief Specify the matrix fill mode of a matrix descriptor
 *
 *  \details
 *  \p hipsparseSetMatFillMode sets the matrix fill mode of a matrix descriptor.
 *  Valid fill modes are \ref HIPSPARSE_FILL_MODE_LOWER or
 *  \ref HIPSPARSE_FILL_MODE_UPPER.
 */
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSetMatFillMode(hipsparseMatDescr_t descrA, hipsparseFillMode_t fillMode);

/*! \ingroup aux_module
 *  \brief Get the matrix fill mode of a matrix descriptor
 *
 *  \details
 *  \p hipsparseGetMatFillMode returns the matrix fill mode of a matrix descriptor.
 */
HIPSPARSE_EXPORT
hipsparseFillMode_t hipsparseGetMatFillMode(const hipsparseMatDescr_t descrA);

/*! \ingroup aux_module
 *  \brief Specify the matrix diagonal type of a matrix descriptor
 *
 *  \details
 *  \p hipsparseSetMatDiagType sets the matrix diagonal type of a matrix
 *  descriptor. Valid diagonal types are \ref HIPSPARSE_DIAG_TYPE_UNIT or
 *  \ref HIPSPARSE_DIAG_TYPE_NON_UNIT.
 */
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSetMatDiagType(hipsparseMatDescr_t descrA, hipsparseDiagType_t diagType);

/*! \ingroup aux_module
 *  \brief Get the matrix diagonal type of a matrix descriptor
 *
 *  \details
 *  \p hipsparseGetMatDiagType returns the matrix diagonal type of a matrix
 *  descriptor.
 */
HIPSPARSE_EXPORT
hipsparseDiagType_t hipsparseGetMatDiagType(const hipsparseMatDescr_t descrA);

/*! \ingroup aux_module
 *  \brief Specify the index base of a matrix descriptor
 *
 *  \details
 *  \p hipsparseSetMatIndexBase sets the index base of a matrix descriptor. Valid
 *  options are \ref HIPSPARSE_INDEX_BASE_ZERO or \ref HIPSPARSE_INDEX_BASE_ONE.
 */
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSetMatIndexBase(hipsparseMatDescr_t descrA, hipsparseIndexBase_t base);

/*! \ingroup aux_module
 *  \brief Get the index base of a matrix descriptor
 *
 *  \details
 *  \p hipsparseGetMatIndexBase returns the index base of a matrix descriptor.
 */
HIPSPARSE_EXPORT
hipsparseIndexBase_t hipsparseGetMatIndexBase(const hipsparseMatDescr_t descrA);

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 11000)
/*! \ingroup aux_module
 *  \brief Create a \p HYB matrix structure
 *
 *  \details
 *  \p hipsparseCreateHybMat creates a structure that holds the matrix in \p HYB
 *  storage format. It should be destroyed at the end using hipsparseDestroyHybMat().
 */
DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCreateHybMat(hipsparseHybMat_t* hybA);

/*! \ingroup aux_module
 *  \brief Destroy a \p HYB matrix structure
 *
 *  \details
 *  \p hipsparseDestroyHybMat destroys a \p HYB structure.
 */
DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDestroyHybMat(hipsparseHybMat_t hybA);
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
/* Info structures */
/*! \ingroup aux_module
 *  \brief Create a bsrsv2 info structure
 *
 *  \details
 *  \p hipsparseCreateBsrsv2Info creates a structure that holds the bsrsv2 info data
 *  that is gathered during the analysis routines available. It should be destroyed
 *  at the end using hipsparseDestroyBsrsv2Info().
 */
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCreateBsrsv2Info(bsrsv2Info_t* info);
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
/*! \ingroup aux_module
 *  \brief Destroy a bsrsv2 info structure
 *
 *  \details
 *  \p hipsparseDestroyBsrsv2Info destroys a bsrsv2 info structure.
 */
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDestroyBsrsv2Info(bsrsv2Info_t info);
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
/* Info structures */
/*! \ingroup aux_module
 *  \brief Create a bsrsm2 info structure
 *
 *  \details
 *  \p hipsparseCreateBsrsm2Info creates a structure that holds the bsrsm2 info data
 *  that is gathered during the analysis routines available. It should be destroyed
 *  at the end using hipsparseDestroyBsrsm2Info().
 */
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCreateBsrsm2Info(bsrsm2Info_t* info);
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
/*! \ingroup aux_module
 *  \brief Destroy a bsrsm2 info structure
 *
 *  \details
 *  \p hipsparseDestroyBsrsm2Info destroys a bsrsm2 info structure.
 */
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDestroyBsrsm2Info(bsrsm2Info_t info);
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
/* Info structures */
/*! \ingroup aux_module
 *  \brief Create a bsrilu02 info structure
 *
 *  \details
 *  \p hipsparseCreateBsrilu02Info creates a structure that holds the bsrilu02 info data
 *  that is gathered during the analysis routines available. It should be destroyed
 *  at the end using hipsparseDestroyBsrilu02Info().
 */
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCreateBsrilu02Info(bsrilu02Info_t* info);
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
/*! \ingroup aux_module
 *  \brief Destroy a bsrilu02 info structure
 *
 *  \details
 *  \p hipsparseDestroyBsrilu02Info destroys a bsrilu02 info structure.
 */
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDestroyBsrilu02Info(bsrilu02Info_t info);
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
/* Info structures */
/*! \ingroup aux_module
 *  \brief Create a bsric02 info structure
 *
 *  \details
 *  \p hipsparseCreateBsric02Info creates a structure that holds the bsric02 info data
 *  that is gathered during the analysis routines available. It should be destroyed
 *  at the end using hipsparseDestroyBsric02Info().
 */
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCreateBsric02Info(bsric02Info_t* info);
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
/*! \ingroup aux_module
 *  \brief Destroy a bsric02 info structure
 *
 *  \details
 *  \p hipsparseDestroyBsric02Info destroys a bsric02 info structure.
 */
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDestroyBsric02Info(bsric02Info_t info);
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 12000)
/* Info structures */
/*! \ingroup aux_module
 *  \brief Create a csrsv2 info structure
 *
 *  \details
 *  \p hipsparseCreateCsrsv2Info creates a structure that holds the csrsv2 info data
 *  that is gathered during the analysis routines available. It should be destroyed
 *  at the end using hipsparseDestroyCsrsv2Info().
 */
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCreateCsrsv2Info(csrsv2Info_t* info);

/*! \ingroup aux_module
 *  \brief Destroy a csrsv2 info structure
 *
 *  \details
 *  \p hipsparseDestroyCsrsv2Info destroys a csrsv2 info structure.
 */
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDestroyCsrsv2Info(csrsv2Info_t info);

/* Info structures */
/*! \ingroup aux_module
 *  \brief Create a csrsm2 info structure
 *
 *  \details
 *  \p hipsparseCreateCsrsm2Info creates a structure that holds the csrsm2 info data
 *  that is gathered during the analysis routines available. It should be destroyed
 *  at the end using hipsparseDestroyCsrsm2Info().
 */
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCreateCsrsm2Info(csrsm2Info_t* info);

/*! \ingroup aux_module
 *  \brief Destroy a csrsm2 info structure
 *
 *  \details
 *  \p hipsparseDestroyCsrsm2Info destroys a csrsm2 info structure.
 */
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDestroyCsrsm2Info(csrsm2Info_t info);
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
/* Info structures */
/*! \ingroup aux_module
 *  \brief Create a csrilu02 info structure
 *
 *  \details
 *  \p hipsparseCreateCsrilu02Info creates a structure that holds the csrilu02 info data
 *  that is gathered during the analysis routines available. It should be destroyed
 *  at the end using hipsparseDestroyCsrilu02Info().
 */
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCreateCsrilu02Info(csrilu02Info_t* info);
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
/*! \ingroup aux_module
 *  \brief Destroy a csrilu02 info structure
 *
 *  \details
 *  \p hipsparseDestroyCsrilu02Info destroys a csrilu02 info structure.
 */
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDestroyCsrilu02Info(csrilu02Info_t info);
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
/* Info structures */
/*! \ingroup aux_module
 *  \brief Create a csric02 info structure
 *
 *  \details
 *  \p hipsparseCreateCsric02Info creates a structure that holds the csric02 info data
 *  that is gathered during the analysis routines available. It should be destroyed
 *  at the end using hipsparseDestroyCsric02Info().
 */
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCreateCsric02Info(csric02Info_t* info);
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
/*! \ingroup aux_module
 *  \brief Destroy a csric02 info structure
 *
 *  \details
 *  \p hipsparseDestroyCsric02Info destroys a csric02 info structure.
 */
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDestroyCsric02Info(csric02Info_t info);
#endif

/* Info structures */
/*! \ingroup aux_module
 *  \brief Create a csru2csr info structure
 *
 *  \details
 *  \p hipsparseCreateCsru2csrInfo creates a structure that holds the csru2csr info data
 *  that is gathered during the analysis routines available. It should be destroyed
 *  at the end using hipsparseDestroyCsru2csrInfo().
 */
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCreateCsru2csrInfo(csru2csrInfo_t* info);

/*! \ingroup aux_module
 *  \brief Destroy a csru2csr info structure
 *
 *  \details
 *  \p hipsparseDestroyCsru2csrInfo destroys a csru2csr info structure.
 */
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDestroyCsru2csrInfo(csru2csrInfo_t info);

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
/* Info structures */
/*! \ingroup aux_module
 *  \brief Create a color info structure
 *
 *  \details
 *  \p hipsparseCreateColorInfo creates a structure that holds the color info data
 *  that is gathered during the analysis routines available. It should be destroyed
 *  at the end using hipsparseDestroyColorInfo().
 */
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCreateColorInfo(hipsparseColorInfo_t* info);
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
/*! \ingroup aux_module
 *  \brief Destroy a color info structure
 *
 *  \details
 *  \p hipsparseDestroyColorInfo destroys a color info structure.
 */
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDestroyColorInfo(hipsparseColorInfo_t info);
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 12000)
/* Info structures */
/*! \ingroup aux_module
 *  \brief Create a csrgemm2 info structure
 *
 *  \details
 *  \p hipsparseCreateCsrgemm2Info creates a structure that holds the csrgemm2 info data
 *  that is gathered during the analysis routines available. It should be destroyed
 *  at the end using hipsparseDestroyCsrgemm2Info().
 */
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCreateCsrgemm2Info(csrgemm2Info_t* info);

/*! \ingroup aux_module
 *  \brief Destroy a csrgemm2 info structure
 *
 *  \details
 *  \p hipsparseDestroyCsrgemm2Info destroys a csrgemm2 info structure.
 */
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDestroyCsrgemm2Info(csrgemm2Info_t info);
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
/* Info structures */
/*! \ingroup aux_module
 *  \brief Create a prune info structure
 *
 *  \details
 *  \p hipsparseCreatePruneInfo creates a structure that holds the prune info data
 *  that is gathered during the analysis routines available. It should be destroyed
 *  at the end using hipsparseDestroyPruneInfo().
 */
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCreatePruneInfo(pruneInfo_t* info);
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
/*! \ingroup aux_module
 *  \brief Destroy a prune info structure
 *
 *  \details
 *  \p hipsparseDestroyPruneInfo destroys a prune info structure.
 */
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDestroyPruneInfo(pruneInfo_t info);
#endif

/*
* ===========================================================================
*    level 1 SPARSE
* ===========================================================================
*/

#include "internal/level1/hipsparse_axpyi.h"
#include "internal/level1/hipsparse_dotci.h"
#include "internal/level1/hipsparse_doti.h"
#include "internal/level1/hipsparse_gthr.h"
#include "internal/level1/hipsparse_gthrz.h"
#include "internal/level1/hipsparse_roti.h"
#include "internal/level1/hipsparse_sctr.h"

/*
* ===========================================================================
*    level 2 SPARSE
* ===========================================================================
*/

#include "internal/level2/hipsparse_bsrmv.h"
#include "internal/level2/hipsparse_bsrsv.h"
#include "internal/level2/hipsparse_bsrxmv.h"
#include "internal/level2/hipsparse_csrmv.h"
#include "internal/level2/hipsparse_csrsv.h"
#include "internal/level2/hipsparse_gemvi.h"
#include "internal/level2/hipsparse_hybmv.h"

/*
* ===========================================================================
*    level 3 SPARSE
* ===========================================================================
*/

#include "internal/level3/hipsparse_bsrmm.h"
#include "internal/level3/hipsparse_bsrsm.h"
#include "internal/level3/hipsparse_csrmm.h"
#include "internal/level3/hipsparse_csrsm.h"
#include "internal/level3/hipsparse_gemmi.h"

/*
* ===========================================================================
*    extra SPARSE
* ===========================================================================
*/

#include "internal/extra/hipsparse_csrgeam.h"
#include "internal/extra/hipsparse_csrgemm.h"

/*
* ===========================================================================
*    preconditioner SPARSE
* ===========================================================================
*/

#include "internal/precond/hipsparse_bsric0.h"
#include "internal/precond/hipsparse_bsrilu0.h"
#include "internal/precond/hipsparse_csric0.h"
#include "internal/precond/hipsparse_csrilu0.h"
#include "internal/precond/hipsparse_gpsv_interleaved_batch.h"
#include "internal/precond/hipsparse_gtsv.h"
#include "internal/precond/hipsparse_gtsv_interleaved_batch.h"
#include "internal/precond/hipsparse_gtsv_nopivot.h"
#include "internal/precond/hipsparse_gtsv_strided_batch.h"

/*
* ===========================================================================
*    Sparse Format Conversions
* ===========================================================================
*/

/*! \ingroup types_module
 *  \brief List of hipsparse csr2csc algorithms.
 *
 *  \details
 *  This is a list of the \ref hipsparseCsr2CscAlg_t algorithms that can be used by the hipSPARSE
 *  library routines \ref hipsparseCsr2cscEx2_bufferSize and \ref hipsparseCsr2cscEx2.
 */
#if(!defined(CUDART_VERSION))
typedef enum
{
    HIPSPARSE_CSR2CSC_ALG_DEFAULT = 0,
    HIPSPARSE_CSR2CSC_ALG1        = 1,
    HIPSPARSE_CSR2CSC_ALG2        = 2
} hipsparseCsr2CscAlg_t;
#else
#if(CUDART_VERSION >= 12000)
typedef enum
{
    HIPSPARSE_CSR2CSC_ALG_DEFAULT = 0,
    HIPSPARSE_CSR2CSC_ALG1        = 1
} hipsparseCsr2CscAlg_t;
#elif(CUDART_VERSION >= 10010 && CUDART_VERSION < 12000)
typedef enum
{
    HIPSPARSE_CSR2CSC_ALG1 = 1,
    HIPSPARSE_CSR2CSC_ALG2 = 2
} hipsparseCsr2CscAlg_t;
#endif
#endif

#include "internal/conversion/hipsparse_bsr2csr.h"
#include "internal/conversion/hipsparse_coo2csr.h"
#include "internal/conversion/hipsparse_coosort.h"
#include "internal/conversion/hipsparse_create_identity_permutation.h"
#include "internal/conversion/hipsparse_csc2dense.h"
#include "internal/conversion/hipsparse_cscsort.h"
#include "internal/conversion/hipsparse_csr2bsr.h"
#include "internal/conversion/hipsparse_csr2coo.h"
#include "internal/conversion/hipsparse_csr2csc.h"
#include "internal/conversion/hipsparse_csr2csr_compress.h"
#include "internal/conversion/hipsparse_csr2csru.h"
#include "internal/conversion/hipsparse_csr2dense.h"
#include "internal/conversion/hipsparse_csr2gebsr.h"
#include "internal/conversion/hipsparse_csr2hyb.h"
#include "internal/conversion/hipsparse_csrsort.h"
#include "internal/conversion/hipsparse_csru2csr.h"
#include "internal/conversion/hipsparse_dense2csc.h"
#include "internal/conversion/hipsparse_dense2csr.h"
#include "internal/conversion/hipsparse_gebsr2csr.h"
#include "internal/conversion/hipsparse_gebsr2gebsc.h"
#include "internal/conversion/hipsparse_gebsr2gebsr.h"
#include "internal/conversion/hipsparse_hyb2csr.h"
#include "internal/conversion/hipsparse_nnz.h"
#include "internal/conversion/hipsparse_nnz_compress.h"
#include "internal/conversion/hipsparse_prune_csr2csr.h"
#include "internal/conversion/hipsparse_prune_csr2csr_by_percentage.h"
#include "internal/conversion/hipsparse_prune_dense2csr.h"
#include "internal/conversion/hipsparse_prune_dense2csr_by_percentage.h"

/*
* ===========================================================================
*    reordering SPARSE
* ===========================================================================
*/

#include "internal/reorder/hipsparse_csrcolor.h"

/*
* ===========================================================================
*    generic SPARSE
* ===========================================================================
*/

/*! \ingroup types_module
 *  \brief Generic API opaque structure holding information for a sparse vector
 *
 *  \details
 *  The hipSPARSE descriptor is an opaque structure holding information for a sparse vector. It must
 *  be initialized using hipsparseCreateSpVec() and the returned descriptor 
 *  is used in hipSPARSE generic API's involving sparse vectors. It should be destroyed at the end using
 *  hipsparseDestroySpVec().
 */
#if(!defined(CUDART_VERSION) || CUDART_VERSION > 10010 \
    || (CUDART_VERSION == 10010 && CUDART_10_1_UPDATE_VERSION == 1))
typedef void* hipsparseSpVecDescr_t;
#endif

/*! \ingroup types_module
 *  \brief Generic API opaque structure holding information for a dense vector
 *
 *  \details
 *  The hipSPARSE descriptor is an opaque structure holding information for a dense vector. It must
 *  be initialized using hipsparseCreateDnVec() and the returned descriptor 
 *  is used in hipSPARSE generic API's involving dense vectors. It should be destroyed at the end using
 *  hipsparseDestroyDnVec().
 */
#if(!defined(CUDART_VERSION) || CUDART_VERSION > 10010 \
    || (CUDART_VERSION == 10010 && CUDART_10_1_UPDATE_VERSION == 1))
typedef void* hipsparseDnVecDescr_t;
#endif

/*! \ingroup types_module
 *  \brief Generic API opaque structure holding information for a sparse matrix
 *
 *  \details
 *  The hipSPARSE descriptor is an opaque structure holding information for a sparse matrix. It must
 *  be initialized using either hipsparseCreateCoo() (for COO format), hipsparseCreateCooAoS() (for COO AOS format).
 *  hipsparseCreateCsr() (for CSR format), hipsparseCreateCsc() (for CSC format) or hipsparseCreateBlockedEll() 
 *  (for Blocked ELL format). The returned descriptor is used in hipSPARSE generic API's involving sparse matrices. 
 *  It should be destroyed at the end using hipsparseDestroySpMat().
 */
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 10010)
typedef void* hipsparseSpMatDescr_t;
#endif

/*! \ingroup types_module
 *  \brief Generic API opaque structure holding information for a dense matrix
 *
 *  \details
 *  The hipSPARSE descriptor is an opaque structure holding information for a dense matrix. It must
 *  be initialized using hipsparseCreateDnMat() and the returned descriptor 
 *  is used in hipSPARSE generic API's involving dense matrices. It should be destroyed at the end using
 *  hipsparseDestroyDnMat().
 */
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 10010)
typedef void* hipsparseDnMatDescr_t;
#endif

/*! \ingroup types_module
 *  \brief Generic API opaque structure holding information for a sparse vector
 *
 *  \details
 *  The hipSPARSE (const) descriptor is an opaque structure holding information for a sparse vector. It must
 *  be initialized using hipsparseCreateConstSpVec() and the returned descriptor 
 *  is used in hipSPARSE generic API's involving sparse vectors. It should be destroyed at the end using
 *  hipsparseDestroySpVec().
 */
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)
typedef void const* hipsparseConstSpVecDescr_t;
#endif

/*! \ingroup types_module
 *  \brief Generic API opaque structure holding information for a dense vector
 *
 *  \details
 *  The hipSPARSE (const) descriptor is an opaque structure holding information for a dense vector. It must
 *  be initialized using hipsparseCreateConstDnVec() and the returned descriptor 
 *  is used in hipSPARSE generic API's involving dense vectors. It should be destroyed at the end using
 *  hipsparseDestroyDnVec().
 */
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)
typedef void const* hipsparseConstDnVecDescr_t;
#endif

/*! \ingroup types_module
 *  \brief Generic API opaque structure holding information for a sparse matrix
 *
 *  \details
 *  The hipSPARSE (const) descriptor is an opaque structure holding information for a sparse matrix. It must
 *  be initialized using hipsparseCreateConstSpMat() and the returned descriptor 
 *  is used in hipSPARSE generic API's involving sparse matrices. It should be destroyed at the end using
 *  hipsparseDestroySpMat().
 */
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)
typedef void const* hipsparseConstSpMatDescr_t;
#endif

/*! \ingroup types_module
 *  \brief Generic API opaque structure holding information for a dense matrix
 *
 *  \details
 *  The hipSPARSE (const) descriptor is an opaque structure holding information for a dense matrix. It must
 *  be initialized using hipsparseCreateConstDnMat() and the returned descriptor 
 *  is used in hipSPARSE generic API's involving dense matrices. It should be destroyed at the end using
 *  hipsparseDestroyDnMat().
 */
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)
typedef void const* hipsparseConstDnMatDescr_t;
#endif

/// \cond DO_NOT_DOCUMENT
// Forward declarations
struct hipsparseSpGEMMDescr;
struct hipsparseSpSVDescr;
struct hipsparseSpSMDescr;
/// \endcond

/*! \ingroup types_module
 *  \brief Generic API opaque structure holding information for a SpGEMM calculations
 *
 *  \details
 *  The hipSPARSE descriptor is an opaque structure holding information that is used in hipsparseSpGEMM_workEstimation(), 
 *  hipsparseSpGEMMreuse_workEstimation(), hipsparseSpGEMMreuse_nnz(), hipsparseSpGEMM_compute(), 
 *  hipsparseSpGEMMreuse_compute(), hipsparseSpGEMM_copy(), and hipsparseSpGEMMreuse_copy(). It must
 *  be initialized using hipsparseSpGEMM_createDescr(). It should be destroyed at the end using
 *  hipsparseSpGEMM_destroyDescr().
 */
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11000)
typedef struct hipsparseSpGEMMDescr* hipsparseSpGEMMDescr_t;
#endif

/*! \ingroup types_module
 *  \brief Generic API opaque structure holding information for a SpSV calculations
 *
 *  \details
 *  The hipSPARSE descriptor is an opaque structure holding information that is used in hipsparseSpSV_bufferSize(), 
 *  hipsparseSpSV_analysis(), and hipsparseSpSV_solve(). It must be initialized using hipsparseSpSV_createDescr(). 
 *  It should be destroyed at the end using hipsparseSpSV_destroyDescr().
 */
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11030)
typedef struct hipsparseSpSVDescr* hipsparseSpSVDescr_t;
#endif

/*! \ingroup types_module
 *  \brief Generic API opaque structure holding information for a SpSM calculations
 *
 *  \details
 *  The hipSPARSE descriptor is an opaque structure holding information that is used in hipsparseSpSM_bufferSize(), 
 *  hipsparseSpSM_analysis(), and hipsparseSpSM_solve(). It must be initialized using hipsparseSpSM_createDescr(). 
 *  It should be destroyed at the end using hipsparseSpSM_destroyDescr().
 */
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11031)
typedef struct hipsparseSpSMDescr* hipsparseSpSMDescr_t;
#endif

/* Generic API types */

/*! \ingroup generic_module
 *  \brief List of hipsparse sparse matrix formats.
 *
 *  \details
 *  This is a list of the \ref hipsparseFormat_t types that are used by the hipSPARSE
 *  library.
 */
#if(!defined(CUDART_VERSION))
typedef enum
{
    HIPSPARSE_FORMAT_CSR         = 1, /* Compressed Sparse Row */
    HIPSPARSE_FORMAT_CSC         = 2, /* Compressed Sparse Column */
    HIPSPARSE_FORMAT_COO         = 3, /* Coordinate - Structure of Arrays */
    HIPSPARSE_FORMAT_COO_AOS     = 4, /* Coordinate - Array of Structures */
    HIPSPARSE_FORMAT_BLOCKED_ELL = 5 /* Blocked ELL */
} hipsparseFormat_t;
#else
#if(CUDART_VERSION >= 12000)
typedef enum
{
    HIPSPARSE_FORMAT_CSR         = 1, /* Compressed Sparse Row */
    HIPSPARSE_FORMAT_CSC         = 2, /* Compressed Sparse Column */
    HIPSPARSE_FORMAT_COO         = 3, /* Coordinate - Structure of Arrays */
    HIPSPARSE_FORMAT_BLOCKED_ELL = 5 /* Blocked ELL */
} hipsparseFormat_t;
#elif(CUDART_VERSION >= 11021 && CUDART_VERSION < 12000)
typedef enum
{
    HIPSPARSE_FORMAT_CSR         = 1, /* Compressed Sparse Row */
    HIPSPARSE_FORMAT_CSC         = 2, /* Compressed Sparse Column */
    HIPSPARSE_FORMAT_COO         = 3, /* Coordinate - Structure of Arrays */
    HIPSPARSE_FORMAT_COO_AOS     = 4, /* Coordinate - Array of Structures */
    HIPSPARSE_FORMAT_BLOCKED_ELL = 5 /* Blocked ELL */
} hipsparseFormat_t;
#elif(CUDART_VERSION >= 10010 && CUDART_VERSION < 11021)
typedef enum
{
    HIPSPARSE_FORMAT_CSR     = 1, /* Compressed Sparse Row */
    HIPSPARSE_FORMAT_COO     = 3, /* Coordinate - Structure of Arrays */
    HIPSPARSE_FORMAT_COO_AOS = 4, /* Coordinate - Array of Structures */
} hipsparseFormat_t;
#endif
#endif

/*! \ingroup generic_module
 *  \brief List of hipsparse dense matrix memory layout ordering.
 *
 *  \details
 *  This is a list of the \ref hipsparseOrder_t types that are used by the hipSPARSE
 *  library.
 */
#if(!defined(CUDART_VERSION))
typedef enum
{
    HIPSPARSE_ORDER_COLUMN HIPSPARSE_DEPRECATED_MSG("Please use HIPSPARSE_ORDER_COL instead")
    = 1, /**< Column major */
    HIPSPARSE_ORDER_COL = 1, /**< Column major */
    HIPSPARSE_ORDER_ROW = 2 /**< Row major */
} hipsparseOrder_t;
#else
#if(CUDART_VERSION >= 11000)
typedef enum
{
    HIPSPARSE_ORDER_COLUMN HIPSPARSE_DEPRECATED_MSG("Please use HIPSPARSE_ORDER_COL instead")
    = 1, /**< Column major */
    HIPSPARSE_ORDER_COL = 1, /**< Column major */
    HIPSPARSE_ORDER_ROW = 2 /**< Row major */
} hipsparseOrder_t;
#elif(CUDART_VERSION >= 10010)
typedef enum
{
    HIPSPARSE_ORDER_COLUMN HIPSPARSE_DEPRECATED_MSG("Please use HIPSPARSE_ORDER_COL instead")
    = 1, /**< Column major */
    HIPSPARSE_ORDER_COL = 1 /**< Column major */
} hipsparseOrder_t;
#endif
#endif

/*! \ingroup generic_module
 *  \brief List of hipsparse index type used by sparse matrix indices.
 *
 *  \details
 *  This is a list of the \ref hipsparseIndexType_t types that are used by the hipSPARSE
 *  library.
 */
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 10010)
typedef enum
{
    HIPSPARSE_INDEX_16U = 1, /**< 16 bit unsigned integer indices */
    HIPSPARSE_INDEX_32I = 2, /**< 32 bit signed integer indices */
    HIPSPARSE_INDEX_64I = 3 /**< 64 bit signed integer indices */
} hipsparseIndexType_t;
#endif

/*! \ingroup generic_module
 *  \brief List of hipsparse SpMV algorithms.
 *
 *  \details
 *  This is a list of the \ref hipsparseSpMVAlg_t types that are used by the hipSPARSE
 *  library.
 */
#if(!defined(CUDART_VERSION))
typedef enum
{
    HIPSPARSE_MV_ALG_DEFAULT   = 0,
    HIPSPARSE_COOMV_ALG        = 1,
    HIPSPARSE_CSRMV_ALG1       = 2,
    HIPSPARSE_CSRMV_ALG2       = 3,
    HIPSPARSE_SPMV_ALG_DEFAULT = 0,
    HIPSPARSE_SPMV_COO_ALG1    = 1,
    HIPSPARSE_SPMV_CSR_ALG1    = 2,
    HIPSPARSE_SPMV_CSR_ALG2    = 3,
    HIPSPARSE_SPMV_COO_ALG2    = 4
} hipsparseSpMVAlg_t;
#else
#if(CUDART_VERSION >= 12000)
typedef enum
{
    HIPSPARSE_SPMV_ALG_DEFAULT = 0,
    HIPSPARSE_SPMV_COO_ALG1    = 1,
    HIPSPARSE_SPMV_CSR_ALG1    = 2,
    HIPSPARSE_SPMV_CSR_ALG2    = 3,
    HIPSPARSE_SPMV_COO_ALG2    = 4
} hipsparseSpMVAlg_t;
#elif(CUDART_VERSION >= 11021 && CUDART_VERSION < 12000)
typedef enum
{
    HIPSPARSE_MV_ALG_DEFAULT   = 0,
    HIPSPARSE_COOMV_ALG        = 1,
    HIPSPARSE_CSRMV_ALG1       = 2,
    HIPSPARSE_CSRMV_ALG2       = 3,
    HIPSPARSE_SPMV_ALG_DEFAULT = 0,
    HIPSPARSE_SPMV_COO_ALG1    = 1,
    HIPSPARSE_SPMV_CSR_ALG1    = 2,
    HIPSPARSE_SPMV_CSR_ALG2    = 3,
    HIPSPARSE_SPMV_COO_ALG2    = 4
} hipsparseSpMVAlg_t;
#elif(CUDART_VERSION >= 10010 && CUDART_VERSION < 11021)
typedef enum
{
    HIPSPARSE_MV_ALG_DEFAULT = 0,
    HIPSPARSE_COOMV_ALG      = 1,
    HIPSPARSE_CSRMV_ALG1     = 2,
    HIPSPARSE_CSRMV_ALG2     = 3
} hipsparseSpMVAlg_t;
#endif
#endif

/*! \ingroup generic_module
 *  \brief List of hipsparse SpMM algorithms.
 *
 *  \details
 *  This is a list of the \ref hipsparseSpMMAlg_t types that are used by the hipSPARSE
 *  library.
 */
#if(!defined(CUDART_VERSION))
typedef enum
{
    HIPSPARSE_MM_ALG_DEFAULT        = 0,
    HIPSPARSE_COOMM_ALG1            = 1,
    HIPSPARSE_COOMM_ALG2            = 2,
    HIPSPARSE_COOMM_ALG3            = 3,
    HIPSPARSE_CSRMM_ALG1            = 4,
    HIPSPARSE_SPMM_ALG_DEFAULT      = 0,
    HIPSPARSE_SPMM_COO_ALG1         = 1,
    HIPSPARSE_SPMM_COO_ALG2         = 2,
    HIPSPARSE_SPMM_COO_ALG3         = 3,
    HIPSPARSE_SPMM_COO_ALG4         = 5,
    HIPSPARSE_SPMM_CSR_ALG1         = 4,
    HIPSPARSE_SPMM_CSR_ALG2         = 6,
    HIPSPARSE_SPMM_CSR_ALG3         = 12,
    HIPSPARSE_SPMM_BLOCKED_ELL_ALG1 = 13
} hipsparseSpMMAlg_t;
#else
#if(CUDART_VERSION >= 12000)
typedef enum
{
    HIPSPARSE_SPMM_ALG_DEFAULT      = 0,
    HIPSPARSE_SPMM_COO_ALG1         = 1,
    HIPSPARSE_SPMM_COO_ALG2         = 2,
    HIPSPARSE_SPMM_COO_ALG3         = 3,
    HIPSPARSE_SPMM_COO_ALG4         = 5,
    HIPSPARSE_SPMM_CSR_ALG1         = 4,
    HIPSPARSE_SPMM_CSR_ALG2         = 6,
    HIPSPARSE_SPMM_CSR_ALG3         = 12,
    HIPSPARSE_SPMM_BLOCKED_ELL_ALG1 = 13
} hipsparseSpMMAlg_t;
#elif(CUDART_VERSION >= 11021 && CUDART_VERSION < 12000)
typedef enum
{
    HIPSPARSE_MM_ALG_DEFAULT        = 0,
    HIPSPARSE_COOMM_ALG1            = 1,
    HIPSPARSE_COOMM_ALG2            = 2,
    HIPSPARSE_COOMM_ALG3            = 3,
    HIPSPARSE_CSRMM_ALG1            = 4,
    HIPSPARSE_SPMM_ALG_DEFAULT      = 0,
    HIPSPARSE_SPMM_COO_ALG1         = 1,
    HIPSPARSE_SPMM_COO_ALG2         = 2,
    HIPSPARSE_SPMM_COO_ALG3         = 3,
    HIPSPARSE_SPMM_COO_ALG4         = 5,
    HIPSPARSE_SPMM_CSR_ALG1         = 4,
    HIPSPARSE_SPMM_CSR_ALG2         = 6,
    HIPSPARSE_SPMM_CSR_ALG3         = 12,
    HIPSPARSE_SPMM_BLOCKED_ELL_ALG1 = 13
} hipsparseSpMMAlg_t;
#elif(CUDART_VERSION >= 11003 && CUDART_VERSION < 11021)
typedef enum
{
    HIPSPARSE_MM_ALG_DEFAULT        = 0,
    HIPSPARSE_COOMM_ALG1            = 1,
    HIPSPARSE_COOMM_ALG2            = 2,
    HIPSPARSE_COOMM_ALG3            = 3,
    HIPSPARSE_CSRMM_ALG1            = 4,
    HIPSPARSE_SPMM_ALG_DEFAULT      = 0,
    HIPSPARSE_SPMM_COO_ALG1         = 1,
    HIPSPARSE_SPMM_COO_ALG2         = 2,
    HIPSPARSE_SPMM_COO_ALG3         = 3,
    HIPSPARSE_SPMM_COO_ALG4         = 5,
    HIPSPARSE_SPMM_CSR_ALG1         = 4,
    HIPSPARSE_SPMM_CSR_ALG2         = 6,
    HIPSPARSE_SPMM_BLOCKED_ELL_ALG1 = 13
} hipsparseSpMMAlg_t;
#elif(CUDART_VERSION >= 10010 && CUDART_VERSION < 11003)
typedef enum
{
    HIPSPARSE_MM_ALG_DEFAULT = 0,
    HIPSPARSE_COOMM_ALG1     = 1,
    HIPSPARSE_COOMM_ALG2     = 2,
    HIPSPARSE_COOMM_ALG3     = 3,
    HIPSPARSE_CSRMM_ALG1     = 4
} hipsparseSpMMAlg_t;
#endif
#endif

/*! \ingroup generic_module
 *  \brief List of hipsparse SparseToDense algorithms.
 *
 *  \details
 *  This is a list of the \ref hipsparseSparseToDenseAlg_t types that are used by the hipSPARSE
 *  library.
 */
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11020)
typedef enum
{
    HIPSPARSE_SPARSETODENSE_ALG_DEFAULT = 0,
} hipsparseSparseToDenseAlg_t;
#endif

/*! \ingroup generic_module
 *  \brief List of hipsparse DenseToSparse algorithms.
 *
 *  \details
 *  This is a list of the \ref hipsparseDenseToSparseAlg_t types that are used by the hipSPARSE
 *  library.
 */
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11020)
typedef enum
{
    HIPSPARSE_DENSETOSPARSE_ALG_DEFAULT = 0,
} hipsparseDenseToSparseAlg_t;
#endif

/*! \ingroup generic_module
 *  \brief List of hipsparse SDDMM algorithms.
 *
 *  \details
 *  This is a list of the \ref hipsparseSDDMMAlg_t types that are used by the hipSPARSE
 *  library.
 */
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11022)
typedef enum
{
    HIPSPARSE_SDDMM_ALG_DEFAULT = 0
} hipsparseSDDMMAlg_t;
#endif

/*! \ingroup generic_module
 *  \brief List of hipsparse SpSV algorithms.
 *
 *  \details
 *  This is a list of the \ref hipsparseSpSVAlg_t types that are used by the hipSPARSE
 *  library.
 */
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11030)
typedef enum
{
    HIPSPARSE_SPSV_ALG_DEFAULT = 0
} hipsparseSpSVAlg_t;
#endif

/*! \ingroup generic_module
 *  \brief List of hipsparse SpSM algorithms.
 *
 *  \details
 *  This is a list of the \ref hipsparseSpSMAlg_t types that are used by the hipSPARSE
 *  library.
 */
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11031)
typedef enum
{
    HIPSPARSE_SPSM_ALG_DEFAULT = 0
} hipsparseSpSMAlg_t;
#endif

/*! \ingroup generic_module
 *  \brief List of hipsparse attributes.
 *
 *  \details
 *  This is a list of the \ref hipsparseSpMatAttribute_t types that are used by the hipSPARSE
 *  library.
 */
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11030)
typedef enum
{
    HIPSPARSE_SPMAT_FILL_MODE = 0, /**< Fill mode attribute */
    HIPSPARSE_SPMAT_DIAG_TYPE = 1 /**< Diag type attribute */
} hipsparseSpMatAttribute_t;
#endif

/*! \ingroup generic_module
 *  \brief List of hipsparse SpGEMM algorithms.
 *
 *  \details
 *  This is a list of the \ref hipsparseSpGEMMAlg_t types that are used by the hipSPARSE
 *  library.
 */
#if(!defined(CUDART_VERSION))
typedef enum
{
    HIPSPARSE_SPGEMM_DEFAULT                  = 0,
    HIPSPARSE_SPGEMM_CSR_ALG_DETERMINISTIC    = 1,
    HIPSPARSE_SPGEMM_CSR_ALG_NONDETERMINISTIC = 2,
    HIPSPARSE_SPGEMM_ALG1                     = 3,
    HIPSPARSE_SPGEMM_ALG2                     = 4,
    HIPSPARSE_SPGEMM_ALG3                     = 5
} hipsparseSpGEMMAlg_t;
#else
#if(CUDART_VERSION >= 12000)
typedef enum
{
    HIPSPARSE_SPGEMM_DEFAULT                  = 0,
    HIPSPARSE_SPGEMM_CSR_ALG_DETERMINISTIC    = 1,
    HIPSPARSE_SPGEMM_CSR_ALG_NONDETERMINISTIC = 2,
    HIPSPARSE_SPGEMM_ALG1                     = 3,
    HIPSPARSE_SPGEMM_ALG2                     = 4,
    HIPSPARSE_SPGEMM_ALG3                     = 5
} hipsparseSpGEMMAlg_t;
#elif(CUDART_VERSION >= 11031 && CUDART_VERSION < 12000)
typedef enum
{
    HIPSPARSE_SPGEMM_DEFAULT                  = 0,
    HIPSPARSE_SPGEMM_CSR_ALG_DETERMINISTIC    = 1,
    HIPSPARSE_SPGEMM_CSR_ALG_NONDETERMINISTIC = 2,
} hipsparseSpGEMMAlg_t;
#elif(CUDART_VERSION >= 11000)
typedef enum
{
    HIPSPARSE_SPGEMM_DEFAULT = 0
} hipsparseSpGEMMAlg_t;
#endif
#endif

/* Sparse vector API */

/*! \ingroup generic_module
*  \brief Create a sparse vector.
*
*  \details
*  \p hipsparseCreateSpVec creates a sparse vector descriptor. It should be
*  destroyed at the end using hipsparseDestroySpVec().
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION > 10010 \
    || (CUDART_VERSION == 10010 && CUDART_10_1_UPDATE_VERSION == 1))
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCreateSpVec(hipsparseSpVecDescr_t* spVecDescr,
                                       int64_t                size,
                                       int64_t                nnz,
                                       void*                  indices,
                                       void*                  values,
                                       hipsparseIndexType_t   idxType,
                                       hipsparseIndexBase_t   idxBase,
                                       hipDataType            valueType);
#endif

/*! \ingroup generic_module
*  \brief Create a const sparse vector.
*
*  \details
*  \p hipsparseCreateConstSpVec creates a const sparse vector descriptor. It should be
*  destroyed at the end using hipsparseDestroySpVec().
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCreateConstSpVec(hipsparseConstSpVecDescr_t* spVecDescr,
                                            int64_t                     size,
                                            int64_t                     nnz,
                                            const void*                 indices,
                                            const void*                 values,
                                            hipsparseIndexType_t        idxType,
                                            hipsparseIndexBase_t        idxBase,
                                            hipDataType                 valueType);
#endif

/*! \ingroup generic_module
*  \brief Destroy a sparse vector.
*
*  \details
*  \p hipsparseDestroySpVec destroys a sparse vector descriptor and releases all
*  resources used by the descriptor.
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDestroySpVec(hipsparseConstSpVecDescr_t spVecDescr);
#elif(CUDART_VERSION > 10010 || (CUDART_VERSION == 10010 && CUDART_10_1_UPDATE_VERSION == 1))
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDestroySpVec(hipsparseSpVecDescr_t spVecDescr);
#endif

/*! \ingroup generic_module
*  \brief Get the fields of the sparse vector descriptor.
*
*  \details
*  \p hipsparseSpVecGet gets the fields of the sparse vector descriptor
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION > 10010 \
    || (CUDART_VERSION == 10010 && CUDART_10_1_UPDATE_VERSION == 1))
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpVecGet(const hipsparseSpVecDescr_t spVecDescr,
                                    int64_t*                    size,
                                    int64_t*                    nnz,
                                    void**                      indices,
                                    void**                      values,
                                    hipsparseIndexType_t*       idxType,
                                    hipsparseIndexBase_t*       idxBase,
                                    hipDataType*                valueType);
#endif

/*! \ingroup generic_module
*  \brief Get the fields of the const sparse vector descriptor.
*
*  \details
*  \p hipsparseConstSpVecGet gets the fields of the const sparse vector descriptor
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseConstSpVecGet(hipsparseConstSpVecDescr_t spVecDescr,
                                         int64_t*                   size,
                                         int64_t*                   nnz,
                                         const void**               indices,
                                         const void**               values,
                                         hipsparseIndexType_t*      idxType,
                                         hipsparseIndexBase_t*      idxBase,
                                         hipDataType*               valueType);
#endif

/*! \ingroup generic_module
*  \brief Get index base of a sparse vector.
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpVecGetIndexBase(const hipsparseConstSpVecDescr_t spVecDescr,
                                             hipsparseIndexBase_t*            idxBase);
#elif(CUDART_VERSION > 10010 || (CUDART_VERSION == 10010 && CUDART_10_1_UPDATE_VERSION == 1))
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpVecGetIndexBase(const hipsparseSpVecDescr_t spVecDescr,
                                             hipsparseIndexBase_t*       idxBase);
#endif

/*! \ingroup generic_module
*  \brief Get pointer to a sparse vector data array.
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION > 10010 \
    || (CUDART_VERSION == 10010 && CUDART_10_1_UPDATE_VERSION == 1))
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpVecGetValues(const hipsparseSpVecDescr_t spVecDescr, void** values);
#endif

/*! \ingroup generic_module
*  \brief Get pointer to a sparse vector data array.
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseConstSpVecGetValues(hipsparseConstSpVecDescr_t spVecDescr,
                                               const void**               values);
#endif

/*! \ingroup generic_module
*  \brief Set pointer of a sparse vector data array.
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION > 10010 \
    || (CUDART_VERSION == 10010 && CUDART_10_1_UPDATE_VERSION == 1))
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpVecSetValues(hipsparseSpVecDescr_t spVecDescr, void* values);
#endif

/* Sparse matrix API */

/*! \ingroup generic_module
*  \brief Create a sparse COO matrix descriptor
*  \details
*  \p hipsparseCreateCoo creates a sparse COO matrix descriptor. It should be
*  destroyed at the end using \p hipsparseDestroySpMat.
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 10010)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCreateCoo(hipsparseSpMatDescr_t* spMatDescr,
                                     int64_t                rows,
                                     int64_t                cols,
                                     int64_t                nnz,
                                     void*                  cooRowInd,
                                     void*                  cooColInd,
                                     void*                  cooValues,
                                     hipsparseIndexType_t   cooIdxType,
                                     hipsparseIndexBase_t   idxBase,
                                     hipDataType            valueType);
#endif

/*! \ingroup generic_module
*  \brief Create a sparse COO matrix descriptor
*  \details
*  \p hipsparseCreateConstCoo creates a sparse COO matrix descriptor. It should be
*  destroyed at the end using \p hipsparseDestroySpMat.
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCreateConstCoo(hipsparseConstSpMatDescr_t* spMatDescr,
                                          int64_t                     rows,
                                          int64_t                     cols,
                                          int64_t                     nnz,
                                          const void*                 cooRowInd,
                                          const void*                 cooColInd,
                                          const void*                 cooValues,
                                          hipsparseIndexType_t        cooIdxType,
                                          hipsparseIndexBase_t        idxBase,
                                          hipDataType                 valueType);
#endif

/*! \ingroup generic_module
*  \brief Create a sparse COO (AoS) matrix descriptor
*  \details
*  \p hipsparseCreateCooAoS creates a sparse COO (AoS) matrix descriptor. It should be
*  destroyed at the end using \p hipsparseDestroySpMat.
*/
#if(!defined(CUDART_VERSION) || (CUDART_VERSION >= 10010 && CUDART_VERSION < 12000))
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCreateCooAoS(hipsparseSpMatDescr_t* spMatDescr,
                                        int64_t                rows,
                                        int64_t                cols,
                                        int64_t                nnz,
                                        void*                  cooInd,
                                        void*                  cooValues,
                                        hipsparseIndexType_t   cooIdxType,
                                        hipsparseIndexBase_t   idxBase,
                                        hipDataType            valueType);
#endif

/*! \ingroup generic_module
*  \brief Create a sparse CSR matrix descriptor
*  \details
*  \p hipsparseCreateCsr creates a sparse CSR matrix descriptor. It should be
*  destroyed at the end using \p hipsparseDestroySpMat.
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 10010)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCreateCsr(hipsparseSpMatDescr_t* spMatDescr,
                                     int64_t                rows,
                                     int64_t                cols,
                                     int64_t                nnz,
                                     void*                  csrRowOffsets,
                                     void*                  csrColInd,
                                     void*                  csrValues,
                                     hipsparseIndexType_t   csrRowOffsetsType,
                                     hipsparseIndexType_t   csrColIndType,
                                     hipsparseIndexBase_t   idxBase,
                                     hipDataType            valueType);
#endif

/*! \ingroup generic_module
*  \brief Create a sparse CSR matrix descriptor
*  \details
*  \p hipsparseCreateConstCsr creates a sparse CSR matrix descriptor. It should be
*  destroyed at the end using \p hipsparseDestroySpMat.
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 12001)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCreateConstCsr(hipsparseConstSpMatDescr_t* spMatDescr,
                                          int64_t                     rows,
                                          int64_t                     cols,
                                          int64_t                     nnz,
                                          const void*                 csrRowOffsets,
                                          const void*                 csrColInd,
                                          const void*                 csrValues,
                                          hipsparseIndexType_t        csrRowOffsetsType,
                                          hipsparseIndexType_t        csrColIndType,
                                          hipsparseIndexBase_t        idxBase,
                                          hipDataType                 valueType);
#endif

/*! \ingroup generic_module
*  \brief Create a sparse CSC matrix descriptor
*  \details
*  \p hipsparseCreateCsc creates a sparse CSC matrix descriptor. It should be
*  destroyed at the end using \p hipsparseDestroySpMat.
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11020)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCreateCsc(hipsparseSpMatDescr_t* spMatDescr,
                                     int64_t                rows,
                                     int64_t                cols,
                                     int64_t                nnz,
                                     void*                  cscColOffsets,
                                     void*                  cscRowInd,
                                     void*                  cscValues,
                                     hipsparseIndexType_t   cscColOffsetsType,
                                     hipsparseIndexType_t   cscRowIndType,
                                     hipsparseIndexBase_t   idxBase,
                                     hipDataType            valueType);
#endif

/*! \ingroup generic_module
*  \brief Create a sparse CSC matrix descriptor
*  \details
*  \p hipsparseCreateConstCsc creates a sparse CSC matrix descriptor. It should be
*  destroyed at the end using \p hipsparseDestroySpMat.
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCreateConstCsc(hipsparseConstSpMatDescr_t* spMatDescr,
                                          int64_t                     rows,
                                          int64_t                     cols,
                                          int64_t                     nnz,
                                          const void*                 cscColOffsets,
                                          const void*                 cscRowInd,
                                          const void*                 cscValues,
                                          hipsparseIndexType_t        cscColOffsetsType,
                                          hipsparseIndexType_t        cscRowIndType,
                                          hipsparseIndexBase_t        idxBase,
                                          hipDataType                 valueType);
#endif

/*! \ingroup generic_module
*  \brief Create a sparse Blocked ELL matrix descriptor
*  \details
*  \p hipsparseCreateCsr creates a sparse Blocked ELL matrix descriptor. It should be
*  destroyed at the end using \p hipsparseDestroySpMat.
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11021)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCreateBlockedEll(hipsparseSpMatDescr_t* spMatDescr,
                                            int64_t                rows,
                                            int64_t                cols,
                                            int64_t                ellBlockSize,
                                            int64_t                ellCols,
                                            void*                  ellColInd,
                                            void*                  ellValue,
                                            hipsparseIndexType_t   ellIdxType,
                                            hipsparseIndexBase_t   idxBase,
                                            hipDataType            valueType);
#endif

/*! \ingroup generic_module
*  \brief Create a sparse Blocked ELL matrix descriptor
*  \details
*  \p hipsparseCreateConstBlockedEll creates a sparse Blocked ELL matrix descriptor. It should be
*  destroyed at the end using \p hipsparseDestroySpMat.
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCreateConstBlockedEll(hipsparseConstSpMatDescr_t* spMatDescr,
                                                 int64_t                     rows,
                                                 int64_t                     cols,
                                                 int64_t                     ellBlockSize,
                                                 int64_t                     ellCols,
                                                 const void*                 ellColInd,
                                                 const void*                 ellValue,
                                                 hipsparseIndexType_t        ellIdxType,
                                                 hipsparseIndexBase_t        idxBase,
                                                 hipDataType                 valueType);
#endif

/*! \ingroup generic_module
*  \brief Destroy a sparse matrix descriptor
*  \details
*  \p hipsparseDestroySpMat destroys a sparse matrix descriptor and releases all
*  resources used by the descriptor.
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDestroySpMat(hipsparseConstSpMatDescr_t spMatDescr);
#elif(CUDART_VERSION >= 10010)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDestroySpMat(hipsparseSpMatDescr_t spMatDescr);
#endif

/*! \ingroup generic_module
*  \brief Get pointers of a sparse COO matrix
*  \details
*  \p hipsparseCooGet gets the fields of the sparse COO matrix descriptor
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 10010)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCooGet(const hipsparseSpMatDescr_t spMatDescr,
                                  int64_t*                    rows,
                                  int64_t*                    cols,
                                  int64_t*                    nnz,
                                  void**                      cooRowInd,
                                  void**                      cooColInd,
                                  void**                      cooValues,
                                  hipsparseIndexType_t*       idxType,
                                  hipsparseIndexBase_t*       idxBase,
                                  hipDataType*                valueType);
#endif

/*! \ingroup generic_module
*  \brief Get pointers of a sparse COO matrix
*  \details
*  \p hipsparseConstCooGet gets the fields of the sparse COO matrix descriptor
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseConstCooGet(hipsparseConstSpMatDescr_t spMatDescr,
                                       int64_t*                   rows,
                                       int64_t*                   cols,
                                       int64_t*                   nnz,
                                       const void**               cooRowInd,
                                       const void**               cooColInd,
                                       const void**               cooValues,
                                       hipsparseIndexType_t*      idxType,
                                       hipsparseIndexBase_t*      idxBase,
                                       hipDataType*               valueType);
#endif

/*! \ingroup generic_module
*  \brief Get pointers of a sparse COO (AoS) matrix
*  \details
*  \p hipsparseCooAoSGet gets the fields of the sparse COO (AoS) matrix descriptor
*/
#if(!defined(CUDART_VERSION) || (CUDART_VERSION >= 10010 && CUDART_VERSION < 12000))
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCooAoSGet(const hipsparseSpMatDescr_t spMatDescr,
                                     int64_t*                    rows,
                                     int64_t*                    cols,
                                     int64_t*                    nnz,
                                     void**                      cooInd,
                                     void**                      cooValues,
                                     hipsparseIndexType_t*       idxType,
                                     hipsparseIndexBase_t*       idxBase,
                                     hipDataType*                valueType);
#endif

/*! \ingroup generic_module
*  \brief Get pointers of a sparse CSR matrix
*  \details
*  \p hipsparseCsrGet gets the fields of the sparse CSR matrix descriptor
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 10010)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCsrGet(const hipsparseSpMatDescr_t spMatDescr,
                                  int64_t*                    rows,
                                  int64_t*                    cols,
                                  int64_t*                    nnz,
                                  void**                      csrRowOffsets,
                                  void**                      csrColInd,
                                  void**                      csrValues,
                                  hipsparseIndexType_t*       csrRowOffsetsType,
                                  hipsparseIndexType_t*       csrColIndType,
                                  hipsparseIndexBase_t*       idxBase,
                                  hipDataType*                valueType);
#endif

/*! \ingroup generic_module
*  \brief Get pointers of a sparse CSR matrix
*  \details
*  \p hipsparseConstCsrGet gets the fields of the sparse CSR matrix descriptor
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 12001)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseConstCsrGet(hipsparseConstSpMatDescr_t spMatDescr,
                                       int64_t*                   rows,
                                       int64_t*                   cols,
                                       int64_t*                   nnz,
                                       const void**               csrRowOffsets,
                                       const void**               csrColInd,
                                       const void**               csrValues,
                                       hipsparseIndexType_t*      csrRowOffsetsType,
                                       hipsparseIndexType_t*      csrColIndType,
                                       hipsparseIndexBase_t*      idxBase,
                                       hipDataType*               valueType);
#endif

/*! \ingroup generic_module
*  \brief Get pointers of a sparse CSC matrix
*  \details
*  \p hipsparseCscGet gets the fields of the sparse CSC matrix descriptor
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 12001)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCscGet(const hipsparseSpMatDescr_t spMatDescr,
                                  int64_t*                    rows,
                                  int64_t*                    cols,
                                  int64_t*                    nnz,
                                  void**                      cscColOffsets,
                                  void**                      cscRowInd,
                                  void**                      cscValues,
                                  hipsparseIndexType_t*       cscColOffsetsType,
                                  hipsparseIndexType_t*       cscRowIndType,
                                  hipsparseIndexBase_t*       idxBase,
                                  hipDataType*                valueType);
#endif

/*! \ingroup generic_module
*  \brief Get pointers of a sparse CSC matrix
*  \details
*  \p hipsparseConstCscGet gets the fields of the sparse CSC matrix descriptor
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 12001)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseConstCscGet(hipsparseConstSpMatDescr_t spMatDescr,
                                       int64_t*                   rows,
                                       int64_t*                   cols,
                                       int64_t*                   nnz,
                                       const void**               cscColOffsets,
                                       const void**               cscRowInd,
                                       const void**               cscValues,
                                       hipsparseIndexType_t*      cscColOffsetsType,
                                       hipsparseIndexType_t*      cscRowIndType,
                                       hipsparseIndexBase_t*      idxBase,
                                       hipDataType*               valueType);
#endif

/*! \ingroup generic_module
*  \brief Get pointers of a sparse blocked ELL matrix
*  \details
*  \p hipsparseBlockedEllGet gets the fields of the sparse blocked ELL matrix descriptor
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11021)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseBlockedEllGet(const hipsparseSpMatDescr_t spMatDescr,
                                         int64_t*                    rows,
                                         int64_t*                    cols,
                                         int64_t*                    ellBlockSize,
                                         int64_t*                    ellCols,
                                         void**                      ellColInd,
                                         void**                      ellValue,
                                         hipsparseIndexType_t*       ellIdxType,
                                         hipsparseIndexBase_t*       idxBase,
                                         hipDataType*                valueType);
#endif

/*! \ingroup generic_module
*  \brief Get pointers of a sparse blocked ELL matrix
*  \details
*  \p hipsparseConstBlockedEllGet gets the fields of the sparse blocked ELL matrix descriptor
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseConstBlockedEllGet(hipsparseConstSpMatDescr_t spMatDescr,
                                              int64_t*                   rows,
                                              int64_t*                   cols,
                                              int64_t*                   ellBlockSize,
                                              int64_t*                   ellCols,
                                              const void**               ellColInd,
                                              const void**               ellValue,
                                              hipsparseIndexType_t*      ellIdxType,
                                              hipsparseIndexBase_t*      idxBase,
                                              hipDataType*               valueType);
#endif

/*! \ingroup generic_module
*  \brief Set pointers of a sparse CSR matrix
*  \details
*  \p hipsparseCsrSetPointers sets the fields of the sparse CSR matrix descriptor
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCsrSetPointers(hipsparseSpMatDescr_t spMatDescr,
                                          void*                 csrRowOffsets,
                                          void*                 csrColInd,
                                          void*                 csrValues);
#endif

/*! \ingroup generic_module
*  \brief Set pointers of a sparse CSC matrix
*  \details
*  \p hipsparseCscSetPointers sets the fields of the sparse CSC matrix descriptor
*/
#if(!defined(CUDART_VERSION))
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCscSetPointers(hipsparseSpMatDescr_t spMatDescr,
                                          void*                 cscColOffsets,
                                          void*                 cscRowInd,
                                          void*                 cscValues);
#endif

/*! \ingroup generic_module
*  \brief Set pointers of a sparse COO matrix
*  \details
*  \p hipsparseCooSetPointers sets the fields of the sparse COO matrix descriptor
*/
#if(!defined(CUDART_VERSION))
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCooSetPointers(hipsparseSpMatDescr_t spMatDescr,
                                          void*                 cooRowInd,
                                          void*                 cooColInd,
                                          void*                 cooValues);
#endif

/*! \ingroup generic_module
*  \brief Get the sizes of a sparse matrix
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpMatGetSize(hipsparseConstSpMatDescr_t spMatDescr,
                                        int64_t*                   rows,
                                        int64_t*                   cols,
                                        int64_t*                   nnz);
#elif(CUDART_VERSION >= 10010)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpMatGetSize(hipsparseSpMatDescr_t spMatDescr,
                                        int64_t*              rows,
                                        int64_t*              cols,
                                        int64_t*              nnz);
#endif

/*! \ingroup generic_module
*  \brief Get the format of a sparse matrix
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpMatGetFormat(hipsparseConstSpMatDescr_t spMatDescr,
                                          hipsparseFormat_t*         format);
#elif(CUDART_VERSION >= 10010)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpMatGetFormat(const hipsparseSpMatDescr_t spMatDescr,
                                          hipsparseFormat_t*          format);
#endif

/*! \ingroup generic_module
*  \brief Get the index base of a sparse matrix
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpMatGetIndexBase(hipsparseConstSpMatDescr_t spMatDescr,
                                             hipsparseIndexBase_t*      idxBase);
#elif(CUDART_VERSION >= 10010)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpMatGetIndexBase(const hipsparseSpMatDescr_t spMatDescr,
                                             hipsparseIndexBase_t*       idxBase);
#endif

/*! \ingroup generic_module
*  \brief Get the pointer of the values array of a sparse matrix
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 10010)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpMatGetValues(hipsparseSpMatDescr_t spMatDescr, void** values);
#endif

/*! \ingroup generic_module
*  \brief Get the pointer of the values array of a sparse matrix
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseConstSpMatGetValues(hipsparseConstSpMatDescr_t spMatDescr,
                                               const void**               values);
#endif

/*! \ingroup generic_module
*  \brief Set the pointer of the values array of a sparse matrix
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 10010)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpMatSetValues(hipsparseSpMatDescr_t spMatDescr, void* values);
#endif

/*! \ingroup generic_module
*  \brief Get the batch count of the sparse matrix
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpMatGetStridedBatch(hipsparseConstSpMatDescr_t spMatDescr,
                                                int*                       batchCount);
#elif(CUDART_VERSION >= 10010)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpMatGetStridedBatch(hipsparseSpMatDescr_t spMatDescr, int* batchCount);
#endif

/*! \ingroup generic_module
*  \brief Set the batch count of the sparse matrix
*/
#if(!defined(CUDART_VERSION) || (CUDART_VERSION >= 10010 && CUDART_VERSION < 12000))
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpMatSetStridedBatch(hipsparseSpMatDescr_t spMatDescr, int batchCount);
#endif

/*! \ingroup generic_module
*  \brief Set the batch count and stride of the sparse COO matrix
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCooSetStridedBatch(hipsparseSpMatDescr_t spMatDescr,
                                              int                   batchCount,
                                              int64_t               batchStride);
#endif

/*! \ingroup generic_module
*  \brief Set the batch count and stride of the sparse CSR matrix
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCsrSetStridedBatch(hipsparseSpMatDescr_t spMatDescr,
                                              int                   batchCount,
                                              int64_t               offsetsBatchStride,
                                              int64_t               columnsValuesBatchStride);
#endif

/*! \ingroup generic_module
*  \brief Get attribute from sparse matrix descriptor
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)
HIPSPARSE_EXPORT
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpMatGetAttribute(hipsparseConstSpMatDescr_t spMatDescr,
                                             hipsparseSpMatAttribute_t  attribute,
                                             void*                      data,
                                             size_t                     dataSize);
#elif(CUDART_VERSION >= 11030)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpMatGetAttribute(hipsparseSpMatDescr_t     spMatDescr,
                                             hipsparseSpMatAttribute_t attribute,
                                             void*                     data,
                                             size_t                    dataSize);
#endif

/*! \ingroup generic_module
*  \brief Set attribute in sparse matrix descriptor
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11030)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpMatSetAttribute(hipsparseSpMatDescr_t     spMatDescr,
                                             hipsparseSpMatAttribute_t attribute,
                                             const void*               data,
                                             size_t                    dataSize);
#endif

/* Dense vector API */

/*! \ingroup generic_module
*  \brief Create dense vector
*  \details
*  \p hipsparseCreateDnVec creates a dense vector descriptor. It should be
*  destroyed at the end using hipsparseDestroyDnVec().
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION > 10010 \
    || (CUDART_VERSION == 10010 && CUDART_10_1_UPDATE_VERSION == 1))
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCreateDnVec(hipsparseDnVecDescr_t* dnVecDescr,
                                       int64_t                size,
                                       void*                  values,
                                       hipDataType            valueType);
#endif

/*! \ingroup generic_module
*  \brief Create dense vector
*  \details
*  \p hipsparseCreateConstDnVec creates a dense vector descriptor. It should be
*  destroyed at the end using hipsparseDestroyDnVec().
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCreateConstDnVec(hipsparseConstDnVecDescr_t* dnVecDescr,
                                            int64_t                     size,
                                            const void*                 values,
                                            hipDataType                 valueType);
#endif

/*! \ingroup generic_module
*  \brief Destroy dense vector
*  \details
*  \p hipsparseDestroyDnVec destroys a dense vector descriptor and releases all
*  resources used by the descriptor.
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDestroyDnVec(hipsparseConstDnVecDescr_t dnVecDescr);
#elif(CUDART_VERSION > 10010 || (CUDART_VERSION == 10010 && CUDART_10_1_UPDATE_VERSION == 1))
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDestroyDnVec(hipsparseDnVecDescr_t dnVecDescr);
#endif

/*! \ingroup generic_module
*  \brief Get the fields from a dense vector
*  \details
*  \p hipsparseDnVecGet gets the fields of the dense vector descriptor
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION > 10010 \
    || (CUDART_VERSION == 10010 && CUDART_10_1_UPDATE_VERSION == 1))
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDnVecGet(const hipsparseDnVecDescr_t dnVecDescr,
                                    int64_t*                    size,
                                    void**                      values,
                                    hipDataType*                valueType);
#endif

/*! \ingroup generic_module
*  \brief Get the fields from a dense vector
*  \details
*  \p hipsparseConstDnVecGet gets the fields of the dense vector descriptor
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseConstDnVecGet(hipsparseConstDnVecDescr_t dnVecDescr,
                                         int64_t*                   size,
                                         const void**               values,
                                         hipDataType*               valueType);
#endif

/*! \ingroup generic_module
*  \brief Get value pointer from a dense vector
*  \details
*  \p hipsparseDnVecGetValues gets the fields of the dense vector descriptor
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION > 10010 \
    || (CUDART_VERSION == 10010 && CUDART_10_1_UPDATE_VERSION == 1))
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDnVecGetValues(const hipsparseDnVecDescr_t dnVecDescr, void** values);
#endif

/*! \ingroup generic_module
*  \brief Get value pointer from a dense vector
*  \details
*  \p hipsparseConstDnVecGetValues gets the fields of the dense vector descriptor
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 12001)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseConstDnVecGetValues(hipsparseConstDnVecDescr_t dnVecDescr,
                                               const void**               values);
#endif

/*! \ingroup generic_module
*  \brief Set value pointer of a dense vector
*  \details
*  \p hipsparseDnVecSetValues sets the fields of the dense vector descriptor
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION > 10010 \
    || (CUDART_VERSION == 10010 && CUDART_10_1_UPDATE_VERSION == 1))
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDnVecSetValues(hipsparseDnVecDescr_t dnVecDescr, void* values);
#endif

/* Dense matrix API */

/* Description: Create dense matrix */

/*! \ingroup generic_module
*  \brief Create dense matrix
*  \details
*  \p hipsparseCreateDnMat creates a dense matrix descriptor. It should be
*  destroyed at the end using hipsparseDestroyDnMat().
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 10010)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCreateDnMat(hipsparseDnMatDescr_t* dnMatDescr,
                                       int64_t                rows,
                                       int64_t                cols,
                                       int64_t                ld,
                                       void*                  values,
                                       hipDataType            valueType,
                                       hipsparseOrder_t       order);
#endif

/*! \ingroup generic_module
*  \brief Create dense matrix
*  \details
*  \p hipsparseCreateConstDnMat creates a dense matrix descriptor. It should be
*  destroyed at the end using hipsparseDestroyDnMat().
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCreateConstDnMat(hipsparseConstDnMatDescr_t* dnMatDescr,
                                            int64_t                     rows,
                                            int64_t                     cols,
                                            int64_t                     ld,
                                            const void*                 values,
                                            hipDataType                 valueType,
                                            hipsparseOrder_t            order);
#endif

/*! \ingroup generic_module
*  \brief Destroy dense matrix
*  \details
*  \p hipsparseDestroyDnMat destroys a dense matrix descriptor and releases all
*  resources used by the descriptor.
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDestroyDnMat(hipsparseConstDnMatDescr_t dnMatDescr);
#elif(CUDART_VERSION >= 10010)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDestroyDnMat(hipsparseDnMatDescr_t dnMatDescr);
#endif

/*! \ingroup generic_module
*  \brief Get fields from a dense matrix
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 10010)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDnMatGet(const hipsparseDnMatDescr_t dnMatDescr,
                                    int64_t*                    rows,
                                    int64_t*                    cols,
                                    int64_t*                    ld,
                                    void**                      values,
                                    hipDataType*                valueType,
                                    hipsparseOrder_t*           order);
#endif

/*! \ingroup generic_module
*  \brief Get fields from a dense matrix
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseConstDnMatGet(hipsparseConstDnMatDescr_t dnMatDescr,
                                         int64_t*                   rows,
                                         int64_t*                   cols,
                                         int64_t*                   ld,
                                         const void**               values,
                                         hipDataType*               valueType,
                                         hipsparseOrder_t*          order);
#endif

/*! \ingroup generic_module
*  \brief Get value pointer from a dense matrix
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 10010)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDnMatGetValues(const hipsparseDnMatDescr_t dnMatDescr, void** values);
#endif

/*! \ingroup generic_module
*  \brief Get value pointer from a dense matrix
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseConstDnMatGetValues(hipsparseConstDnMatDescr_t dnMatDescr,
                                               const void**               values);
#endif

/*! \ingroup generic_module
*  \brief Set value pointer of a dense matrix
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 10010)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDnMatSetValues(hipsparseDnMatDescr_t dnMatDescr, void* values);
#endif

/*! \ingroup generic_module
*  \brief Get the batch count and batch stride of the dense matrix
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDnMatGetStridedBatch(hipsparseConstDnMatDescr_t dnMatDescr,
                                                int*                       batchCount,
                                                int64_t*                   batchStride);
#elif(CUDART_VERSION >= 10010)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDnMatGetStridedBatch(hipsparseDnMatDescr_t dnMatDescr,
                                                int*                  batchCount,
                                                int64_t*              batchStride);
#endif

/*! \ingroup generic_module
*  \brief Set the batch count and batch stride of the dense matrix
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 10010)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDnMatSetStridedBatch(hipsparseDnMatDescr_t dnMatDescr,
                                                int                   batchCount,
                                                int64_t               batchStride);
#endif

/*
* ===========================================================================
*    Generic
* ===========================================================================
*/

#include "internal/generic/hipsparse_axpby.h"
#include "internal/generic/hipsparse_dense2sparse.h"
#include "internal/generic/hipsparse_gather.h"
#include "internal/generic/hipsparse_rot.h"
#include "internal/generic/hipsparse_scatter.h"
#include "internal/generic/hipsparse_sddmm.h"
#include "internal/generic/hipsparse_sparse2dense.h"
#include "internal/generic/hipsparse_spgemm.h"
#include "internal/generic/hipsparse_spgemm_reuse.h"
#include "internal/generic/hipsparse_spmm.h"
#include "internal/generic/hipsparse_spmv.h"
#include "internal/generic/hipsparse_spsm.h"
#include "internal/generic/hipsparse_spsv.h"
#include "internal/generic/hipsparse_spvv.h"

#ifdef __cplusplus
}
#endif

#endif // HIPSPARSE_H
