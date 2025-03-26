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

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 11000)
/*! \ingroup extra_module
*  \brief Sparse matrix sparse matrix addition using CSR storage format
*
*  \details
*  \p hipsparseXcsrgeamNnz computes the total CSR non-zero elements and the CSR row
*  offsets, that point to the start of every row of the sparse CSR matrix, of the
*  resulting matrix C. It is assumed that \p csrRowPtrC has been allocated with
*  size \p m+1. The desired index base in the output CSR matrix is set in the 
*  \ref hipsparseMatDescr_t. See \ref hipsparseSetMatIndexBase().
*
*  For full code example, see \ref hipsparseScsrgeam().
*
*  \note
*  As indicated, \p nnzTotalDevHostPtr can point either to host or device memory. This is controlled 
*  by setting the pointer mode. See \ref hipsparseSetPointerMode().
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*
*  \note
*  Currently, only \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*
*  @param[in]
*  handle          handle to the hipsparse library context queue.
*  @param[in]
*  m               number of rows of the sparse CSR matrix \f$A\f$, \f$B\f$ and \f$C\f$.
*  @param[in]
*  n               number of columns of the sparse CSR matrix \f$A\f$, \f$B\f$ and \f$C\f$.
*  @param[in]
*  descrA          descriptor of the sparse CSR matrix \f$A\f$. Currently, only
*                  \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*  @param[in]
*  nnzA            number of non-zero entries of the sparse CSR matrix \f$A\f$.
*  @param[in]
*  csrRowPtrA      array of \p m+1 elements that point to the start of every row of the
*                  sparse CSR matrix \f$A\f$.
*  @param[in]
*  csrColIndA      array of \p nnzA elements containing the column indices of the
*                  sparse CSR matrix \f$A\f$.
*  @param[in]
*  descrB          descriptor of the sparse CSR matrix \f$B\f$. Currently, only
*                  \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*  @param[in]
*  nnzB            number of non-zero entries of the sparse CSR matrix \f$B\f$.
*  @param[in]
*  csrRowPtrB      array of \p m+1 elements that point to the start of every row of the
*                  sparse CSR matrix \f$B\f$.
*  @param[in]
*  csrColIndB      array of \p nnzB elements containing the column indices of the
*                  sparse CSR matrix \f$B\f$.
*  @param[in]
*  descrC          descriptor of the sparse CSR matrix \f$C\f$. Currently, only
*                  \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*  @param[out]
*  csrRowPtrC      array of \p m+1 elements that point to the start of every row of the
*                  sparse CSR matrix \f$C\f$.
*  @param[out]
*  nnzTotalDevHostPtr pointer to the number of non-zero entries of the sparse CSR
*                     matrix \f$C\f$. \p nnzTotalDevHostPtr can be a host or device pointer.
*
*  \retval HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p m, \p n, \p nnzA, \p nnzB, \p descrA, 
*          \p csrRowPtrA, \p csrColIndA, \p descrB, \p csrRowPtrB, \p csrColIndB, \p descrC, 
*          \p csrRowPtrC or \p nnzTotalDevHostPtr is invalid.
*  \retval HIPSPARSE_STATUS_NOT_SUPPORTED
*          \ref hipsparseMatrixType_t != \ref HIPSPARSE_MATRIX_TYPE_GENERAL.
*/
DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseXcsrgeamNnz(hipsparseHandle_t         handle,
                                       int                       m,
                                       int                       n,
                                       const hipsparseMatDescr_t descrA,
                                       int                       nnzA,
                                       const int*                csrRowPtrA,
                                       const int*                csrColIndA,
                                       const hipsparseMatDescr_t descrB,
                                       int                       nnzB,
                                       const int*                csrRowPtrB,
                                       const int*                csrColIndB,
                                       const hipsparseMatDescr_t descrC,
                                       int*                      csrRowPtrC,
                                       int*                      nnzTotalDevHostPtr);

/*! \ingroup extra_module
*  \brief Sparse matrix sparse matrix addition using CSR storage format
*
*  \details
*  \p hipsparseXcsrgeam multiplies the scalar \f$\alpha\f$ with the sparse
*  \f$m \times n\f$ matrix \f$A\f$, defined in CSR storage format, multiplies the
*  scalar \f$\beta\f$ with the sparse \f$m \times n\f$ matrix \f$B\f$, defined in CSR
*  storage format, and adds both resulting matrices to obtain the sparse
*  \f$m \times n\f$ matrix \f$C\f$, defined in CSR storage format, such that
*  \f[
*    C := \alpha \cdot A + \beta \cdot B.
*  \f]
*
*  This computation involves a multi step process. First the user must allocate \p csrRowPtrC
*  to have size \p m+1. The user then calls \ref hipsparseXcsrgeamNnz which fills in the \p csrRowPtrC
*  array as well as computes the total number of nonzeros in C, \p nnzC. The user then allocates both 
*  arrays \p csrColIndC and \p csrValC to have size \p nnzC and calls \p hipsparseXcsrgeam to complete 
*  the computation. The desired index base in the output CSR matrix C is set in the 
*  \ref hipsparseMatDescr_t \p descrC. See \ref hipsparseSetMatIndexBase().
*
*  \note Both scalars \f$\alpha\f$ and \f$beta\f$ have to be valid.
*  \note Currently, only \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*  \note This function is non blocking and executed asynchronously with respect to the
*        host. It may return before the actual computation has finished.
*
*  @param[in]
*  handle          handle to the hipsparse library context queue.
*  @param[in]
*  m               number of rows of the sparse CSR matrix \f$A\f$, \f$B\f$ and \f$C\f$.
*  @param[in]
*  n               number of columns of the sparse CSR matrix \f$A\f$, \f$B\f$ and \f$C\f$.
*  @param[in]
*  alpha           scalar \f$\alpha\f$.
*  @param[in]
*  descrA          descriptor of the sparse CSR matrix \f$A\f$. Currently, only
*                  \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*  @param[in]
*  nnzA            number of non-zero entries of the sparse CSR matrix \f$A\f$.
*  @param[in]
*  csrValA         array of \p nnzA elements of the sparse CSR matrix \f$A\f$.
*  @param[in]
*  csrRowPtrA      array of \p m+1 elements that point to the start of every row of the
*                  sparse CSR matrix \f$A\f$.
*  @param[in]
*  csrColIndA      array of \p nnzA elements containing the column indices of the
*                  sparse CSR matrix \f$A\f$.
*  @param[in]
*  beta            scalar \f$\beta\f$.
*  @param[in]
*  descrB          descriptor of the sparse CSR matrix \f$B\f$. Currently, only
*                  \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*  @param[in]
*  nnzB            number of non-zero entries of the sparse CSR matrix \f$B\f$.
*  @param[in]
*  csrValB         array of \p nnzB elements of the sparse CSR matrix \f$B\f$.
*  @param[in]
*  csrRowPtrB      array of \p m+1 elements that point to the start of every row of the
*                  sparse CSR matrix \f$B\f$.
*  @param[in]
*  csrColIndB      array of \p nnzB elements containing the column indices of the
*                  sparse CSR matrix \f$B\f$.
*  @param[in]
*  descrC          descriptor of the sparse CSR matrix \f$C\f$. Currently, only
*                  \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*  @param[out]
*  csrValC         array of elements of the sparse CSR matrix \f$C\f$.
*  @param[in]
*  csrRowPtrC      array of \p m+1 elements that point to the start of every row of the
*                  sparse CSR matrix \f$C\f$.
*  @param[out]
*  csrColIndC      array of elements containing the column indices of the
*                  sparse CSR matrix \f$C\f$.
*
*  \retval HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p m, \p n, \p nnzA, \p nnzB, 
*          \p alpha, \p descrA, \p csrValA, \p csrRowPtrA, \p csrColIndA, \p beta, 
*          \p descrB, \p csrValB, \p csrRowPtrB, \p csrColIndB, \p descrC, \p csrValC,
*          \p csrRowPtrC or \p csrColIndC is invalid.
*  \retval HIPSPARSE_STATUS_NOT_SUPPORTED
*          \ref hipsparseMatrixType_t != \ref HIPSPARSE_MATRIX_TYPE_GENERAL.
*
*  \par Example
*  \code{.c}
*    int m = 4;
*    int n = 4;
*    int nnzA = 9;
*    int nnzB = 6;
*    
*    float alpha{1.0f};
*    float beta{1.0f};
*
*    // A, B, and C are m×n
*
*    // A
*    // 1 0 0 2
*    // 3 4 0 0
*    // 5 6 7 8
*    // 0 0 9 0
*    std::vector<int> hcsrRowPtrA = {0, 2, 4, 8, 9};
*    std::vector<int> hcsrColIndA = {0, 3, 0, 1, 0, 1, 2, 3, 2};
*    std::vector<float> hcsrValA = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
*
*    // B
*    // 0 1 0 0
*    // 1 0 1 0
*    // 0 1 0 1
*    // 0 0 1 0
*    std::vector<int> hcsrRowPtrB = {0, 1, 3, 5, 6};
*    std::vector<int> hcsrColIndB = {1, 0, 2, 1, 3, 2};
*    std::vector<float> hcsrValB = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
*
*    // Device memory management: Allocate and copy A, B
*    int* dcsrRowPtrA;
*    int* dcsrColIndA;
*    float* dcsrValA;
*    int* dcsrRowPtrB;
*    int* dcsrColIndB;
*    float* dcsrValB;
*    int* dcsrRowPtrC;
*    hipMalloc((void**)&dcsrRowPtrA, (m + 1) * sizeof(int));
*    hipMalloc((void**)&dcsrColIndA, nnzA * sizeof(int));
*    hipMalloc((void**)&dcsrValA, nnzA * sizeof(float));
*    hipMalloc((void**)&dcsrRowPtrB, (m + 1) * sizeof(int));
*    hipMalloc((void**)&dcsrColIndB, nnzB * sizeof(int));
*    hipMalloc((void**)&dcsrValB, nnzB * sizeof(float));
*    hipMalloc((void**)&dcsrRowPtrC, (m + 1) * sizeof(int));
*
*    hipMemcpy(dcsrRowPtrA, hcsrRowPtrA.data(), (m + 1) * sizeof(int), hipMemcpyHostToDevice);
*    hipMemcpy(dcsrColIndA, hcsrColIndA.data(), nnzA * sizeof(int), hipMemcpyHostToDevice);
*    hipMemcpy(dcsrValA, hcsrValA.data(), nnzA * sizeof(float), hipMemcpyHostToDevice);
*    hipMemcpy(dcsrRowPtrB, hcsrRowPtrB.data(), (m + 1) * sizeof(int), hipMemcpyHostToDevice);
*    hipMemcpy(dcsrColIndB, hcsrColIndB.data(), nnzB * sizeof(int), hipMemcpyHostToDevice);
*    hipMemcpy(dcsrValB, hcsrValB.data(), nnzB * sizeof(float), hipMemcpyHostToDevice);
*
*    hipsparseHandle_t handle;
*    hipsparseCreate(&handle);
*
*    hipsparseMatDescr_t descrA;
*    hipsparseCreateMatDescr(&descrA);
*
*    hipsparseMatDescr_t descrB;
*    hipsparseCreateMatDescr(&descrB);
*
*    hipsparseMatDescr_t descrC;
*    hipsparseCreateMatDescr(&descrC);
*
*    int nnzC;
*    hipsparseXcsrgeamNnz(handle,
*                        m,
*                        n,
*                        descrA,
*                        nnzA,
*                        dcsrRowPtrA,
*                        dcsrColIndA,
*                        descrB,
*                        nnzB,
*                        dcsrRowPtrB,
*                        dcsrColIndB,
*                        descrC,
*                        dcsrRowPtrC,
*                        &nnzC);
*
*    int* dcsrColIndC = nullptr;
*    float* dcsrValC = nullptr;
*    hipMalloc((void**)&dcsrColIndC, sizeof(int) * nnzC);
*    hipMalloc((void**)&dcsrValC, sizeof(float) * nnzC);
*
*    hipsparseScsrgeam(handle,
*                      m,
*                      n,
*                      &alpha,
*                      descrA,
*                      nnzA,
*                      dcsrValA,
*                      dcsrRowPtrA,
*                      dcsrColIndA,
*                      &beta,
*                      descrB,
*                      nnzB,
*                      dcsrValB,
*                      dcsrRowPtrB,
*                      dcsrColIndB,
*                      descrC,
*                      dcsrValC,
*                      dcsrRowPtrC,
*                      dcsrColIndC);
*
*    hipFree(dcsrRowPtrA);
*    hipFree(dcsrColIndA);
*    hipFree(dcsrValA);
*    hipFree(dcsrRowPtrB);
*    hipFree(dcsrColIndB);
*    hipFree(dcsrValB);
*    hipFree(dcsrRowPtrC);
*    hipFree(dcsrColIndC);
*    hipFree(dcsrValC);
*
*    hipsparseDestroyMatDescr(descrA);
*    hipsparseDestroyMatDescr(descrB);
*    hipsparseDestroyMatDescr(descrC);
*    hipsparseDestroy(handle);
*  \endcode
*/
/**@{*/
DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseScsrgeam(hipsparseHandle_t         handle,
                                    int                       m,
                                    int                       n,
                                    const float*              alpha,
                                    const hipsparseMatDescr_t descrA,
                                    int                       nnzA,
                                    const float*              csrValA,
                                    const int*                csrRowPtrA,
                                    const int*                csrColIndA,
                                    const float*              beta,
                                    const hipsparseMatDescr_t descrB,
                                    int                       nnzB,
                                    const float*              csrValB,
                                    const int*                csrRowPtrB,
                                    const int*                csrColIndB,
                                    const hipsparseMatDescr_t descrC,
                                    float*                    csrValC,
                                    int*                      csrRowPtrC,
                                    int*                      csrColIndC);
DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDcsrgeam(hipsparseHandle_t         handle,
                                    int                       m,
                                    int                       n,
                                    const double*             alpha,
                                    const hipsparseMatDescr_t descrA,
                                    int                       nnzA,
                                    const double*             csrValA,
                                    const int*                csrRowPtrA,
                                    const int*                csrColIndA,
                                    const double*             beta,
                                    const hipsparseMatDescr_t descrB,
                                    int                       nnzB,
                                    const double*             csrValB,
                                    const int*                csrRowPtrB,
                                    const int*                csrColIndB,
                                    const hipsparseMatDescr_t descrC,
                                    double*                   csrValC,
                                    int*                      csrRowPtrC,
                                    int*                      csrColIndC);
DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCcsrgeam(hipsparseHandle_t         handle,
                                    int                       m,
                                    int                       n,
                                    const hipComplex*         alpha,
                                    const hipsparseMatDescr_t descrA,
                                    int                       nnzA,
                                    const hipComplex*         csrValA,
                                    const int*                csrRowPtrA,
                                    const int*                csrColIndA,
                                    const hipComplex*         beta,
                                    const hipsparseMatDescr_t descrB,
                                    int                       nnzB,
                                    const hipComplex*         csrValB,
                                    const int*                csrRowPtrB,
                                    const int*                csrColIndB,
                                    const hipsparseMatDescr_t descrC,
                                    hipComplex*               csrValC,
                                    int*                      csrRowPtrC,
                                    int*                      csrColIndC);
DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZcsrgeam(hipsparseHandle_t         handle,
                                    int                       m,
                                    int                       n,
                                    const hipDoubleComplex*   alpha,
                                    const hipsparseMatDescr_t descrA,
                                    int                       nnzA,
                                    const hipDoubleComplex*   csrValA,
                                    const int*                csrRowPtrA,
                                    const int*                csrColIndA,
                                    const hipDoubleComplex*   beta,
                                    const hipsparseMatDescr_t descrB,
                                    int                       nnzB,
                                    const hipDoubleComplex*   csrValB,
                                    const int*                csrRowPtrB,
                                    const int*                csrColIndB,
                                    const hipsparseMatDescr_t descrC,
                                    hipDoubleComplex*         csrValC,
                                    int*                      csrRowPtrC,
                                    int*                      csrColIndC);
/**@}*/
#endif

/*! \ingroup extra_module
*  \brief Sparse matrix sparse matrix multiplication using CSR storage format
*
*  \details
*  \p hipsparseXcsrgeam2_bufferSizeExt returns the size of the temporary storage buffer
*  in bytes that is required by \ref hipsparseXcsrgeam2Nnz() and \ref hipsparseScsrgeam2 
*  "hipsparseXcsrgeam2()". The temporary storage buffer must be allocated by the user. 
*
*  \note
*  Currently, only \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*
*  @param[in]
*  handle             handle to the hipsparse library context queue.
*  @param[in]
*  m                  number of rows of the sparse CSR matrix \f$A\f$, \f$B\f$ and \f$C\f$.
*  @param[in]
*  n                  number of columns of the sparse CSR matrix \f$A\f$, \f$B\f$ and \f$C\f$.
*  @param[in]
*  alpha              scalar \f$\alpha\f$.
*  @param[in]
*  descrA             descriptor of the sparse CSR matrix \f$A\f$. Currently, only
*                     \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*  @param[in]
*  nnzA               number of non-zero entries of the sparse CSR matrix \f$A\f$.
*  @param[in]
*  csrSortedValA      array of \p nnzA elements of the sparse CSR matrix \f$A\f$.
*  @param[in]
*  csrSortedRowPtrA   array of \p m+1 elements that point to the start of every row of the
*                     sparse CSR matrix \f$A\f$.
*  @param[in]
*  csrSortedColIndA   array of \p nnzA elements containing the column indices of the
*                     sparse CSR matrix \f$A\f$.
*  @param[in]
*  beta               scalar \f$\beta\f$.
*  @param[in]
*  descrB             descriptor of the sparse CSR matrix \f$B\f$. Currently, only
*                     \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*  @param[in]
*  nnzB               number of non-zero entries of the sparse CSR matrix \f$B\f$.
*  @param[in]
*  csrSortedValB      array of \p nnzB elements of the sparse CSR matrix \f$B\f$.
*  @param[in]
*  csrSortedRowPtrB   array of \p m+1 elements that point to the start of every row of the
*                     sparse CSR matrix \f$B\f$.
*  @param[in]
*  csrSortedColIndB   array of \p nnzB elements containing the column indices of the
*                     sparse CSR matrix \f$B\f$.
*  @param[in]
*  descrC             descriptor of the sparse CSR matrix \f$C\f$. Currently, only
*                     \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*  @param[out]
*  csrSortedValC      array of elements of the sparse CSR matrix \f$C\f$.
*  @param[in]
*  csrSortedRowPtrC   array of \p m+1 elements that point to the start of every row of the
*                     sparse CSR matrix \f$C\f$.
*  @param[out]
*  csrSortedColIndC   array of elements containing the column indices of the
*                     sparse CSR matrix \f$C\f$.
*  @param[out]
*  pBufferSizeInBytes number of bytes of the temporary storage buffer required by
*                     hipsparseXcsrgeam2Nnz(), hipsparseScsrgeam2(),
*                     hipsparseDcsrgeam2(), hipsparseCcsrgeam2(),
*                     hipsparseZcsrgeam2().
*
*  \retval HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p m, \p n, \p nnzA, \p nnzB, 
*          \p alpha, \p descrA, \p csrSortedValA, \p csrSortedRowPtrA, \p csrSortedColIndA, 
*          \p beta, \p descrB, \p csrSortedValB, \p csrSortedRowPtrB, \p csrSortedColIndB, 
*          \p descrC, \p csrSortedValC, \p csrSortedRowPtrC, \p csrSortedColIndC, or 
*          \p pBufferSizeInBytes is invalid.
*  \retval HIPSPARSE_STATUS_NOT_SUPPORTED
*          \ref hipsparseMatrixType_t != \ref HIPSPARSE_MATRIX_TYPE_GENERAL.
*/
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseScsrgeam2_bufferSizeExt(hipsparseHandle_t         handle,
                                                   int                       m,
                                                   int                       n,
                                                   const float*              alpha,
                                                   const hipsparseMatDescr_t descrA,
                                                   int                       nnzA,
                                                   const float*              csrSortedValA,
                                                   const int*                csrSortedRowPtrA,
                                                   const int*                csrSortedColIndA,
                                                   const float*              beta,
                                                   const hipsparseMatDescr_t descrB,
                                                   int                       nnzB,
                                                   const float*              csrSortedValB,
                                                   const int*                csrSortedRowPtrB,
                                                   const int*                csrSortedColIndB,
                                                   const hipsparseMatDescr_t descrC,
                                                   const float*              csrSortedValC,
                                                   const int*                csrSortedRowPtrC,
                                                   const int*                csrSortedColIndC,
                                                   size_t*                   pBufferSizeInBytes);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDcsrgeam2_bufferSizeExt(hipsparseHandle_t         handle,
                                                   int                       m,
                                                   int                       n,
                                                   const double*             alpha,
                                                   const hipsparseMatDescr_t descrA,
                                                   int                       nnzA,
                                                   const double*             csrSortedValA,
                                                   const int*                csrSortedRowPtrA,
                                                   const int*                csrSortedColIndA,
                                                   const double*             beta,
                                                   const hipsparseMatDescr_t descrB,
                                                   int                       nnzB,
                                                   const double*             csrSortedValB,
                                                   const int*                csrSortedRowPtrB,
                                                   const int*                csrSortedColIndB,
                                                   const hipsparseMatDescr_t descrC,
                                                   const double*             csrSortedValC,
                                                   const int*                csrSortedRowPtrC,
                                                   const int*                csrSortedColIndC,
                                                   size_t*                   pBufferSizeInBytes);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCcsrgeam2_bufferSizeExt(hipsparseHandle_t         handle,
                                                   int                       m,
                                                   int                       n,
                                                   const hipComplex*         alpha,
                                                   const hipsparseMatDescr_t descrA,
                                                   int                       nnzA,
                                                   const hipComplex*         csrSortedValA,
                                                   const int*                csrSortedRowPtrA,
                                                   const int*                csrSortedColIndA,
                                                   const hipComplex*         beta,
                                                   const hipsparseMatDescr_t descrB,
                                                   int                       nnzB,
                                                   const hipComplex*         csrSortedValB,
                                                   const int*                csrSortedRowPtrB,
                                                   const int*                csrSortedColIndB,
                                                   const hipsparseMatDescr_t descrC,
                                                   const hipComplex*         csrSortedValC,
                                                   const int*                csrSortedRowPtrC,
                                                   const int*                csrSortedColIndC,
                                                   size_t*                   pBufferSizeInBytes);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZcsrgeam2_bufferSizeExt(hipsparseHandle_t         handle,
                                                   int                       m,
                                                   int                       n,
                                                   const hipDoubleComplex*   alpha,
                                                   const hipsparseMatDescr_t descrA,
                                                   int                       nnzA,
                                                   const hipDoubleComplex*   csrSortedValA,
                                                   const int*                csrSortedRowPtrA,
                                                   const int*                csrSortedColIndA,
                                                   const hipDoubleComplex*   beta,
                                                   const hipsparseMatDescr_t descrB,
                                                   int                       nnzB,
                                                   const hipDoubleComplex*   csrSortedValB,
                                                   const int*                csrSortedRowPtrB,
                                                   const int*                csrSortedColIndB,
                                                   const hipsparseMatDescr_t descrC,
                                                   const hipDoubleComplex*   csrSortedValC,
                                                   const int*                csrSortedRowPtrC,
                                                   const int*                csrSortedColIndC,
                                                   size_t*                   pBufferSizeInBytes);
/**@}*/

/*! \ingroup extra_module
*  \brief Sparse matrix sparse matrix addition using CSR storage format
*
*  \details
*  \p hipsparseXcsrgeam2Nnz computes the total CSR non-zero elements and the CSR row
*  offsets, that point to the start of every row of the sparse CSR matrix, of the
*  resulting matrix C. It is assumed that \p csrRowPtrC has been allocated with
*  size \p m+1. The required buffer size can be obtained by 
*  \ref hipsparseScsrgeam2_bufferSizeExt "hipsparseXcsrgeam2_bufferSizeExt()". The 
*  desired index base in the output CSR matrix C is set in the \ref hipsparseMatDescr_t 
*  \p descrC. See \ref hipsparseSetMatIndexBase().
*
*  \note
*  As indicated, \p nnzTotalDevHostPtr can point either to host or device memory. This is controlled 
*  by setting the pointer mode. See \ref hipsparseSetPointerMode().
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*  \note
*  Currently, only \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*
*  @param[in]
*  handle             handle to the hipsparse library context queue.
*  @param[in]
*  m                  number of rows of the sparse CSR matrix \f$A\f$, \f$B\f$ and \f$C\f$.
*  @param[in]
*  n                  number of columns of the sparse CSR matrix \f$A\f$, \f$B\f$ and \f$C\f$.
*  @param[in]
*  descrA             descriptor of the sparse CSR matrix \f$A\f$. Currently, only
*                     \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*  @param[in]
*  nnzA               number of non-zero entries of the sparse CSR matrix \f$A\f$.
*  @param[in]
*  csrSortedRowPtrA   array of \p m+1 elements that point to the start of every row of the
*                     sparse CSR matrix \f$A\f$.
*  @param[in]
*  csrSortedColIndA   array of \p nnzA elements containing the column indices of the
*                     sparse CSR matrix \f$A\f$.
*  @param[in]
*  descrB             descriptor of the sparse CSR matrix \f$B\f$. Currently, only
*                     \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*  @param[in]
*  nnzB               number of non-zero entries of the sparse CSR matrix \f$B\f$.
*  @param[in]
*  csrSortedRowPtrB   array of \p m+1 elements that point to the start of every row of the
*                     sparse CSR matrix \f$B\f$.
*  @param[in]
*  csrSortedColIndB   array of \p nnzB elements containing the column indices of the
*                     sparse CSR matrix \f$B\f$.
*  @param[in]
*  descrC             descriptor of the sparse CSR matrix \f$C\f$. Currently, only
*                     \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*  @param[in]
*  csrSortedRowPtrC   array of \p m+1 elements that point to the start of every row of the
*                     sparse CSR matrix \f$C\f$.
*  @param[out]
*  nnzTotalDevHostPtr pointer to the number of non-zero entries of the sparse CSR
*                     matrix \f$C\f$. \p nnzTotalDevHostPtr can be a host or device pointer.
*  @param[in]
*  workspace          temporary storage buffer allocated by the user.
*
*  \retval HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p m, \p n, \p nnzA, \p nnzB, 
*          \p descrA, \p csrSortedRowPtrA, \p csrSortedColIndA, \p descrB, \p csrSortedRowPtrB, 
*          \p csrSortedColIndB, \p descrC, \p csrSortedRowPtrC or \p nnzTotalDevHostPtr is invalid.
*  \retval HIPSPARSE_STATUS_NOT_SUPPORTED
*          \ref hipsparseMatrixType_t != \ref HIPSPARSE_MATRIX_TYPE_GENERAL.
*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseXcsrgeam2Nnz(hipsparseHandle_t         handle,
                                        int                       m,
                                        int                       n,
                                        const hipsparseMatDescr_t descrA,
                                        int                       nnzA,
                                        const int*                csrSortedRowPtrA,
                                        const int*                csrSortedColIndA,
                                        const hipsparseMatDescr_t descrB,
                                        int                       nnzB,
                                        const int*                csrSortedRowPtrB,
                                        const int*                csrSortedColIndB,
                                        const hipsparseMatDescr_t descrC,
                                        int*                      csrSortedRowPtrC,
                                        int*                      nnzTotalDevHostPtr,
                                        void*                     workspace);

/*! \ingroup extra_module
*  \brief Sparse matrix sparse matrix addition using CSR storage format
*
*  \details
*  \p hipsparseXcsrgeam2 multiplies the scalar \f$\alpha\f$ with the sparse
*  \f$m \times n\f$ matrix \f$A\f$, defined in CSR storage format, multiplies the
*  scalar \f$\beta\f$ with the sparse \f$m \times n\f$ matrix \f$B\f$, defined in CSR
*  storage format, and adds both resulting matrices to obtain the sparse
*  \f$m \times n\f$ matrix \f$C\f$, defined in CSR storage format, such that
*  \f[
*    C := \alpha \cdot A + \beta \cdot B.
*  \f]
*
*  This computation involves a multi step process. First the user must call 
*  \ref hipsparseScsrgeam2_bufferSizeExt "hipsparseXcsrgeam2_bufferSizeExt()" in order to determine the 
*  required user allocated temporary buffer size. The user then allocates this buffer and also allocates 
*  \p csrRowPtrC to have size \p m+1. Both the temporary storage buffer and \p csrRowPtrC array are then 
*  passed to \ref hipsparseXcsrgeam2Nnz which fills in the \p csrRowPtrC array as well as computes the total 
*  number of nonzeros in C, \p nnzC. The user then allocates both arrays \p csrColIndC and \p csrValC to have 
*  size \p nnzC and calls \p hipsparseXcsrgeam2 to complete the computation. The desired index base in 
*  the output CSR matrix C is set in the \ref hipsparseMatDescr_t \p descrC. See \ref hipsparseSetMatIndexBase().
*
*  \note Both scalars \f$\alpha\f$ and \f$beta\f$ have to be valid.
*  \note Currently, only \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*  \note This function is non blocking and executed asynchronously with respect to the
*        host. It may return before the actual computation has finished.
*
*  @param[in]
*  handle           handle to the hipsparse library context queue.
*  @param[in]
*  m                number of rows of the sparse CSR matrix \f$A\f$, \f$B\f$ and \f$C\f$.
*  @param[in]
*  n                number of columns of the sparse CSR matrix \f$A\f$, \f$B\f$ and \f$C\f$.
*  @param[in]
*  alpha            scalar \f$\alpha\f$.
*  @param[in]
*  descrA           descriptor of the sparse CSR matrix \f$A\f$. Currently, only
*                   \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*  @param[in]
*  nnzA             number of non-zero entries of the sparse CSR matrix \f$A\f$.
*  @param[in]
*  csrSortedValA    array of \p nnzA elements of the sparse CSR matrix \f$A\f$.
*  @param[in]
*  csrSortedRowPtrA array of \p m+1 elements that point to the start of every row of the
*                   sparse CSR matrix \f$A\f$.
*  @param[in]
*  csrSortedColIndA array of \p nnzA elements containing the column indices of the
*                   sparse CSR matrix \f$A\f$.
*  @param[in]
*  beta             scalar \f$\beta\f$.
*  @param[in]
*  descrB           descriptor of the sparse CSR matrix \f$B\f$. Currently, only
*                   \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*  @param[in]
*  nnzB             number of non-zero entries of the sparse CSR matrix \f$B\f$.
*  @param[in]
*  csrSortedValB    array of \p nnzB elements of the sparse CSR matrix \f$B\f$.
*  @param[in]
*  csrSortedRowPtrB array of \p m+1 elements that point to the start of every row of the
*                   sparse CSR matrix \f$B\f$.
*  @param[in]
*  csrSortedColIndB array of \p nnzB elements containing the column indices of the
*                   sparse CSR matrix \f$B\f$.
*  @param[in]
*  descrC           descriptor of the sparse CSR matrix \f$C\f$. Currently, only
*                   \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*  @param[out]
*  csrSortedValC    array of elements of the sparse CSR matrix \f$C\f$.
*  @param[in]
*  csrSortedRowPtrC array of \p m+1 elements that point to the start of every row of the
*                   sparse CSR matrix \f$C\f$.
*  @param[out]
*  csrSortedColIndC array of elements containing the column indices of the
*                   sparse CSR matrix \f$C\f$.
*  @param[in]
*  pBuffer          temporary storage buffer allocated by the user.
*
*  \retval HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p m, \p n, \p nnzA, \p nnzB, 
*          \p alpha, \p descrA, \p csrSortedValA, \p csrSortedRowPtrA, \p csrSortedColIndA, \p beta, 
*          \p descrB, \p csrSortedValB, \p csrSortedRowPtrB, \p csrSortedColIndB, \p descrC, \p csrSortedValC,
*          \p csrSortedRowPtrC, \p csrSortedColIndC or \p pBuffer is invalid.
*  \retval HIPSPARSE_STATUS_NOT_SUPPORTED
*          \ref hipsparseMatrixType_t != \ref HIPSPARSE_MATRIX_TYPE_GENERAL.
*
*  \par Example
*  \code{.c}
*    int m = 4;
*    int n = 4;
*    int nnzA = 9;
*    int nnzB = 6;
*    
*    float alpha{1.0f};
*    float beta{1.0f};
*
*    // A, B, and C are m×n
*
*    // A
*    // 1 0 0 2
*    // 3 4 0 0
*    // 5 6 7 8
*    // 0 0 9 0
*    std::vector<int> hcsrRowPtrA = {0, 2, 4, 8, 9};
*    std::vector<int> hcsrColIndA = {0, 3, 0, 1, 0, 1, 2, 3, 2};
*    std::vector<float> hcsrValA = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
*
*    // B
*    // 0 1 0 0
*    // 1 0 1 0
*    // 0 1 0 1
*    // 0 0 1 0
*    std::vector<int> hcsrRowPtrB = {0, 1, 3, 5, 6};
*    std::vector<int> hcsrColIndB = {1, 0, 2, 1, 3, 2};
*    std::vector<float> hcsrValB = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
*
*    // Device memory management: Allocate and copy A, B
*    int* dcsrRowPtrA;
*    int* dcsrColIndA;
*    float* dcsrValA;
*    int* dcsrRowPtrB;
*    int* dcsrColIndB;
*    float* dcsrValB;
*    int* dcsrRowPtrC;
*    hipMalloc((void**)&dcsrRowPtrA, (m + 1) * sizeof(int));
*    hipMalloc((void**)&dcsrColIndA, nnzA * sizeof(int));
*    hipMalloc((void**)&dcsrValA, nnzA * sizeof(float));
*    hipMalloc((void**)&dcsrRowPtrB, (m + 1) * sizeof(int));
*    hipMalloc((void**)&dcsrColIndB, nnzB * sizeof(int));
*    hipMalloc((void**)&dcsrValB, nnzB * sizeof(float));
*    hipMalloc((void**)&dcsrRowPtrC, (m + 1) * sizeof(int));
*
*    hipMemcpy(dcsrRowPtrA, hcsrRowPtrA.data(), (m + 1) * sizeof(int), hipMemcpyHostToDevice);
*    hipMemcpy(dcsrColIndA, hcsrColIndA.data(), nnzA * sizeof(int), hipMemcpyHostToDevice);
*    hipMemcpy(dcsrValA, hcsrValA.data(), nnzA * sizeof(float), hipMemcpyHostToDevice);
*    hipMemcpy(dcsrRowPtrB, hcsrRowPtrB.data(), (m + 1) * sizeof(int), hipMemcpyHostToDevice);
*    hipMemcpy(dcsrColIndB, hcsrColIndB.data(), nnzB * sizeof(int), hipMemcpyHostToDevice);
*    hipMemcpy(dcsrValB, hcsrValB.data(), nnzB * sizeof(float), hipMemcpyHostToDevice);
*
*    hipsparseHandle_t handle;
*    hipsparseCreate(&handle);
*
*    hipsparseMatDescr_t descrA;
*    hipsparseCreateMatDescr(&descrA);
*
*    hipsparseMatDescr_t descrB;
*    hipsparseCreateMatDescr(&descrB);
*
*    hipsparseMatDescr_t descrC;
*    hipsparseCreateMatDescr(&descrC);
*
*    size_t bufferSize;
*    hipsparseScsrgeam2_bufferSizeExt(handle, 
*                                     m, 
*                                     n, 
*                                     &alpha, 
*                                     descrA, 
*                                     nnzA, 
*                                     dcsrValA, 
*                                     dcsrRowPtrA, 
*                                     dcsrColIndA, 
*                                     &beta, 
*                                     descrB, 
*                                     nnzB, 
*                                     dcsrValB, 
*                                     dcsrRowPtrB, 
*                                     dcsrColIndB, 
*                                     descrC, 
*                                     nullptr, 
*                                     dcsrRowPtrC, 
*                                     nullptr, 
*                                     &bufferSize);
*
*    void* dbuffer = nullptr;
*    hipMalloc((void**)&dbuffer, bufferSize);
*
*    int nnzC;
*    hipsparseXcsrgeam2Nnz(handle,
*                        m,
*                        n,
*                        descrA,
*                        nnzA,
*                        dcsrRowPtrA,
*                        dcsrColIndA,
*                        descrB,
*                        nnzB,
*                        dcsrRowPtrB,
*                        dcsrColIndB,
*                        descrC,
*                        dcsrRowPtrC,
*                        &nnzC,
*                        dbuffer);
*
*    int* dcsrColIndC = nullptr;
*    float* dcsrValC = nullptr;
*    hipMalloc((void**)&dcsrColIndC, sizeof(int) * nnzC);
*    hipMalloc((void**)&dcsrValC, sizeof(float) * nnzC);
*
*    hipsparseScsrgeam2(handle,
*                      m,
*                      n,
*                      &alpha,
*                      descrA,
*                      nnzA,
*                      dcsrValA,
*                      dcsrRowPtrA,
*                      dcsrColIndA,
*                      &beta,
*                      descrB,
*                      nnzB,
*                      dcsrValB,
*                      dcsrRowPtrB,
*                      dcsrColIndB,
*                      descrC,
*                      dcsrValC,
*                      dcsrRowPtrC,
*                      dcsrColIndC,
*                      dbuffer);
*
*    hipFree(dcsrRowPtrA);
*    hipFree(dcsrColIndA);
*    hipFree(dcsrValA);
*    hipFree(dcsrRowPtrB);
*    hipFree(dcsrColIndB);
*    hipFree(dcsrValB);
*    hipFree(dcsrRowPtrC);
*    hipFree(dcsrColIndC);
*    hipFree(dcsrValC);
*
*    hipFree(dbuffer);
*
*    hipsparseDestroyMatDescr(descrA);
*    hipsparseDestroyMatDescr(descrB);
*    hipsparseDestroyMatDescr(descrC);
*    hipsparseDestroy(handle);
*  \endcode
*/
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseScsrgeam2(hipsparseHandle_t         handle,
                                     int                       m,
                                     int                       n,
                                     const float*              alpha,
                                     const hipsparseMatDescr_t descrA,
                                     int                       nnzA,
                                     const float*              csrSortedValA,
                                     const int*                csrSortedRowPtrA,
                                     const int*                csrSortedColIndA,
                                     const float*              beta,
                                     const hipsparseMatDescr_t descrB,
                                     int                       nnzB,
                                     const float*              csrSortedValB,
                                     const int*                csrSortedRowPtrB,
                                     const int*                csrSortedColIndB,
                                     const hipsparseMatDescr_t descrC,
                                     float*                    csrSortedValC,
                                     int*                      csrSortedRowPtrC,
                                     int*                      csrSortedColIndC,
                                     void*                     pBuffer);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDcsrgeam2(hipsparseHandle_t         handle,
                                     int                       m,
                                     int                       n,
                                     const double*             alpha,
                                     const hipsparseMatDescr_t descrA,
                                     int                       nnzA,
                                     const double*             csrSortedValA,
                                     const int*                csrSortedRowPtrA,
                                     const int*                csrSortedColIndA,
                                     const double*             beta,
                                     const hipsparseMatDescr_t descrB,
                                     int                       nnzB,
                                     const double*             csrSortedValB,
                                     const int*                csrSortedRowPtrB,
                                     const int*                csrSortedColIndB,
                                     const hipsparseMatDescr_t descrC,
                                     double*                   csrSortedValC,
                                     int*                      csrSortedRowPtrC,
                                     int*                      csrSortedColIndC,
                                     void*                     pBuffer);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCcsrgeam2(hipsparseHandle_t         handle,
                                     int                       m,
                                     int                       n,
                                     const hipComplex*         alpha,
                                     const hipsparseMatDescr_t descrA,
                                     int                       nnzA,
                                     const hipComplex*         csrSortedValA,
                                     const int*                csrSortedRowPtrA,
                                     const int*                csrSortedColIndA,
                                     const hipComplex*         beta,
                                     const hipsparseMatDescr_t descrB,
                                     int                       nnzB,
                                     const hipComplex*         csrSortedValB,
                                     const int*                csrSortedRowPtrB,
                                     const int*                csrSortedColIndB,
                                     const hipsparseMatDescr_t descrC,
                                     hipComplex*               csrSortedValC,
                                     int*                      csrSortedRowPtrC,
                                     int*                      csrSortedColIndC,
                                     void*                     pBuffer);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZcsrgeam2(hipsparseHandle_t         handle,
                                     int                       m,
                                     int                       n,
                                     const hipDoubleComplex*   alpha,
                                     const hipsparseMatDescr_t descrA,
                                     int                       nnzA,
                                     const hipDoubleComplex*   csrSortedValA,
                                     const int*                csrSortedRowPtrA,
                                     const int*                csrSortedColIndA,
                                     const hipDoubleComplex*   beta,
                                     const hipsparseMatDescr_t descrB,
                                     int                       nnzB,
                                     const hipDoubleComplex*   csrSortedValB,
                                     const int*                csrSortedRowPtrB,
                                     const int*                csrSortedColIndB,
                                     const hipsparseMatDescr_t descrC,
                                     hipDoubleComplex*         csrSortedValC,
                                     int*                      csrSortedRowPtrC,
                                     int*                      csrSortedColIndC,
                                     void*                     pBuffer);
/**@}*/

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 11000)
/*! \ingroup extra_module
*  \brief Sparse matrix sparse matrix multiplication using CSR storage format
*
*  \details
*  \p hipsparseXcsrgemmNnz computes the total CSR non-zero elements and the CSR row
*  offsets, that point to the start of every row of the sparse CSR matrix, of the
*  resulting multiplied matrix C. It is assumed that \p csrRowPtrC has been allocated
*  with size \p m+1. The desired index base in the output CSR matrix C is set in the 
*  \ref hipsparseMatDescr_t \p descrC. See \ref hipsparseSetMatIndexBase().
*
*  \note
*  As indicated, \p nnzTotalDevHostPtr can point either to host or device memory. This is controlled 
*  by setting the pointer mode. See \ref hipsparseSetPointerMode().
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*
*  \note
*  Please note, that for matrix products with more than 8192 intermediate products per
*  row, additional temporary storage buffer is allocated by the algorithm.
*
*  \note
*  Currently, only \p transA == \p transB == \ref HIPSPARSE_OPERATION_NON_TRANSPOSE is
*  supported.
*
*  \note
*  Currently, only \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*
*  @param[in]
*  handle          handle to the hipsparse library context queue.
*  @param[in]
*  transA          matrix \f$A\f$ operation type.
*  @param[in]
*  transB          matrix \f$B\f$ operation type.
*  @param[in]
*  m               number of rows of the sparse CSR matrix \f$op(A)\f$ and \f$C\f$.
*  @param[in]
*  n               number of columns of the sparse CSR matrix \f$op(B)\f$ and
*                  \f$C\f$.
*  @param[in]
*  k               number of columns of the sparse CSR matrix \f$op(A)\f$ and number of
*                  rows of the sparse CSR matrix \f$op(B)\f$.
*  @param[in]
*  descrA          descriptor of the sparse CSR matrix \f$A\f$. Currently, only
*                  \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*  @param[in]
*  nnzA            number of non-zero entries of the sparse CSR matrix \f$A\f$.
*  @param[in]
*  csrRowPtrA      array of \p m+1 elements (\f$op(A) == A\f$, \p k+1 otherwise)
*                  that point to the start of every row of the sparse CSR matrix
*                  \f$op(A)\f$.
*  @param[in]
*  csrColIndA      array of \p nnzA elements containing the column indices of the
*                  sparse CSR matrix \f$A\f$.
*  @param[in]
*  descrB          descriptor of the sparse CSR matrix \f$B\f$. Currently, only
*                  \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*  @param[in]
*  nnzB            number of non-zero entries of the sparse CSR matrix \f$B\f$.
*  @param[in]
*  csrRowPtrB      array of \p k+1 elements (\f$op(B) == B\f$, \p m+1 otherwise)
*                  that point to the start of every row of the sparse CSR matrix
*                  \f$op(B)\f$.
*  @param[in]
*  csrColIndB      array of \p nnzB elements containing the column indices of the
*                  sparse CSR matrix \f$B\f$.
*  @param[in]
*  descrC          descriptor of the sparse CSR matrix \f$C\f$. Currently, only
*                  \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*  @param[in]
*  csrRowPtrC      array of \p m+1 elements that point to the start of every row of the
*                  sparse CSR matrix \f$C\f$.
*  @param[inout]
*  nnzTotalDevHostPtr pointer to the number of non-zero entries of the sparse CSR
*                     matrix \f$C\f$. \p nnzTotalDevHostPtr can be a host or device pointer.
*
*  \retval HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p m, \p n, \p k, \p nnzA, \p nnzB, \p nnzC, 
*          \p descrA, \p csrRowPtrA, \p csrColIndA, \p descrB, \p csrRowPtrB, \p csrColIndB, 
*          \p descrC, \p csrRowPtrC or \p nnzTotalDevHostPtr is invalid.
*  \retval HIPSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED
*          \p transA != \ref HIPSPARSE_OPERATION_NON_TRANSPOSE,
*          \p transB != \ref HIPSPARSE_OPERATION_NON_TRANSPOSE, or
*          \ref hipsparseMatrixType_t != \ref HIPSPARSE_MATRIX_TYPE_GENERAL.
*/
DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseXcsrgemmNnz(hipsparseHandle_t         handle,
                                       hipsparseOperation_t      transA,
                                       hipsparseOperation_t      transB,
                                       int                       m,
                                       int                       n,
                                       int                       k,
                                       const hipsparseMatDescr_t descrA,
                                       int                       nnzA,
                                       const int*                csrRowPtrA,
                                       const int*                csrColIndA,
                                       const hipsparseMatDescr_t descrB,
                                       int                       nnzB,
                                       const int*                csrRowPtrB,
                                       const int*                csrColIndB,
                                       const hipsparseMatDescr_t descrC,
                                       int*                      csrRowPtrC,
                                       int*                      nnzTotalDevHostPtr);
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 11000)
/*! \ingroup extra_module
*  \brief Sparse matrix sparse matrix multiplication using CSR storage format
*
*  \details
*  \p hipsparseXcsrgemm multiplies the sparse \f$m \times k\f$ matrix \f$op(A)\f$, defined in
*  CSR storage format with the sparse \f$k \times n\f$ matrix \f$op(B)\f$, defined in CSR
*  storage format, and stores the result in the sparse \f$m \times n\f$ matrix \f$C\f$,
*  defined in CSR storage format, such that
*  \f[
*    C := op(A) \cdot op(B),
*  \f]
*  with
*  \f[
*    op(A) = \left\{
*    \begin{array}{ll}
*        A,   & \text{if transA == HIPSPARSE_OPERATION_NON_TRANSPOSE} \\
*        A^T, & \text{if transA == HIPSPARSE_OPERATION_TRANSPOSE} \\
*        A^H, & \text{if transA == HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE}
*    \end{array}
*    \right.
*  \f]
*  and
*  \f[
*    op(B) = \left\{
*    \begin{array}{ll}
*        B,   & \text{if transB == HIPSPARSE_OPERATION_NON_TRANSPOSE} \\
*        B^T, & \text{if transB == HIPSPARSE_OPERATION_TRANSPOSE} \\
*        B^H, & \text{if transB == HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE}
*    \end{array}
*    \right.
*  \f]
*
*  This computation involves a multi step process. First the user must allocate \p csrRowPtrC
*  to have size \p m+1. The user then calls \ref hipsparseXcsrgemmNnz which fills in the \p csrRowPtrC
*  array as well as computes the total number of nonzeros in C, \p nnzC. The user then allocates both 
*  arrays \p csrColIndC and \p csrValC to have size \p nnzC and calls \p hipsparseXcsrgemm to complete 
*  the computation. The desired index base in the output CSR matrix C is set in the 
*  \ref hipsparseMatDescr_t \p descrC. See \ref hipsparseSetMatIndexBase().
*
*  \note Currently, only \p transA == \ref HIPSPARSE_OPERATION_NON_TRANSPOSE is supported.
*  \note Currently, only \p transB == \ref HIPSPARSE_OPERATION_NON_TRANSPOSE is supported.
*  \note Currently, only \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*  \note This function is non blocking and executed asynchronously with respect to the
*        host. It may return before the actual computation has finished.
*  \note Please note, that for matrix products with more than 4096 non-zero entries per
*  row, additional temporary storage buffer is allocated by the algorithm.
*
*  @param[in]
*  handle          handle to the hipsparse library context queue.
*  @param[in]
*  transA          matrix \f$A\f$ operation type.
*  @param[in]
*  transB          matrix \f$B\f$ operation type.
*  @param[in]
*  m               number of rows of the sparse CSR matrix \f$op(A)\f$ and \f$C\f$.
*  @param[in]
*  n               number of columns of the sparse CSR matrix \f$op(B)\f$ and
*                  \f$C\f$.
*  @param[in]
*  k               number of columns of the sparse CSR matrix \f$op(A)\f$ and number of
*                  rows of the sparse CSR matrix \f$op(B)\f$.
*  @param[in]
*  descrA          descriptor of the sparse CSR matrix \f$A\f$. Currently, only
*                  \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*  @param[in]
*  nnzA            number of non-zero entries of the sparse CSR matrix \f$A\f$.
*  @param[in]
*  csrValA         array of \p nnzA elements of the sparse CSR matrix \f$A\f$.
*  @param[in]
*  csrRowPtrA      array of \p m+1 elements (\f$op(A) == A\f$, \p k+1 otherwise)
*                  that point to the start of every row of the sparse CSR matrix
*                  \f$op(A)\f$.
*  @param[in]
*  csrColIndA      array of \p nnzA elements containing the column indices of the
*                  sparse CSR matrix \f$A\f$.
*  @param[in]
*  descrB          descriptor of the sparse CSR matrix \f$B\f$. Currently, only
*                  \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*  @param[in]
*  nnzB            number of non-zero entries of the sparse CSR matrix \f$B\f$.
*  @param[in]
*  csrValB         array of \p nnzB elements of the sparse CSR matrix \f$B\f$.
*  @param[in]
*  csrRowPtrB      array of \p k+1 elements (\f$op(B) == B\f$, \p m+1 otherwise)
*                  that point to the start of every row of the sparse CSR matrix
*                  \f$op(B)\f$.
*  @param[in]
*  csrColIndB      array of \p nnzB elements containing the column indices of the
*                  sparse CSR matrix \f$B\f$.
*  @param[in]
*  descrC          descriptor of the sparse CSR matrix \f$C\f$. Currently, only
*                  \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*  @param[out]
*  csrValC         array of \p nnzC elements of the sparse CSR matrix \f$C\f$.
*  @param[in]
*  csrRowPtrC      array of \p m+1 elements that point to the start of every row of the
*                  sparse CSR matrix \f$C\f$.
*  @param[out]
*  csrColIndC      array of \p nnzC elements containing the column indices of the
*                  sparse CSR matrix \f$C\f$.
*
*  \retval HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p m, \p n, \p k, \p nnzA, \p nnzB,
*          \p descrA, \p csrValA, \p csrRowPtrA, \p csrColIndA, \p descrB, \p csrValB, 
*          \p csrRowPtrB, \p csrColIndB, \p descrC, \p csrValC, \p csrRowPtrC, \p csrColIndC 
*          is invalid.
*  \retval HIPSPARSE_STATUS_ALLOC_FAILED additional buffer for long rows could not be
*          allocated.
*  \retval HIPSPARSE_STATUS_NOT_SUPPORTED
*          \p transA != \ref HIPSPARSE_OPERATION_NON_TRANSPOSE,
*          \p transB != \ref HIPSPARSE_OPERATION_NON_TRANSPOSE, or
*          \ref hipsparseMatrixType_t != \ref HIPSPARSE_MATRIX_TYPE_GENERAL.
*
*  \par Example
*  \code{.c}
*    int m = 4;
*    int k = 3;
*    int n = 2;
*    int nnzA = 7;
*    int nnzB = 3;
*
*    hipsparseOperation_t transA = HIPSPARSE_OPERATION_NON_TRANSPOSE;
*    hipsparseOperation_t transB = HIPSPARSE_OPERATION_NON_TRANSPOSE;
*
*    // A, B, and C are mxk, kxn, and m×n
*
*    // A
*    // 1 0 0
*    // 3 4 0
*    // 5 6 7
*    // 0 0 9
*    std::vector<int> hcsrRowPtrA = {0, 1, 3, 6, 7};
*    std::vector<int> hcsrColIndA = {0, 0, 1, 0, 1, 2, 2};
*    std::vector<float> hcsrValA = {1.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 9.0f};
*
*    // B
*    // 0 1
*    // 1 0
*    // 0 1
*    std::vector<int> hcsrRowPtrB = {0, 1, 2, 3};
*    std::vector<int> hcsrColIndB = {1, 0, 1};
*    std::vector<float> hcsrValB = {1.0f, 1.0f, 1.0f};
*
*    // Device memory management: Allocate and copy A, B
*    int* dcsrRowPtrA;
*    int* dcsrColIndA;
*    float* dcsrValA;
*    int* dcsrRowPtrB;
*    int* dcsrColIndB;
*    float* dcsrValB;
*    int* dcsrRowPtrC;
*    hipMalloc((void**)&dcsrRowPtrA, (m + 1) * sizeof(int));
*    hipMalloc((void**)&dcsrColIndA, nnzA * sizeof(int));
*    hipMalloc((void**)&dcsrValA, nnzA * sizeof(float));
*    hipMalloc((void**)&dcsrRowPtrB, (m + 1) * sizeof(int));
*    hipMalloc((void**)&dcsrColIndB, nnzB * sizeof(int));
*    hipMalloc((void**)&dcsrValB, nnzB * sizeof(float));
*    hipMalloc((void**)&dcsrRowPtrC, (m + 1) * sizeof(int));
*
*    hipMemcpy(dcsrRowPtrA, hcsrRowPtrA.data(), (m + 1) * sizeof(int), hipMemcpyHostToDevice);
*    hipMemcpy(dcsrColIndA, hcsrColIndA.data(), nnzA * sizeof(int), hipMemcpyHostToDevice);
*    hipMemcpy(dcsrValA, hcsrValA.data(), nnzA * sizeof(float), hipMemcpyHostToDevice);
*    hipMemcpy(dcsrRowPtrB, hcsrRowPtrB.data(), (m + 1) * sizeof(int), hipMemcpyHostToDevice);
*    hipMemcpy(dcsrColIndB, hcsrColIndB.data(), nnzB * sizeof(int), hipMemcpyHostToDevice);
*    hipMemcpy(dcsrValB, hcsrValB.data(), nnzB * sizeof(float), hipMemcpyHostToDevice);
*
*    hipsparseHandle_t handle;
*    hipsparseCreate(&handle);
*
*    hipsparseMatDescr_t descrA;
*    hipsparseCreateMatDescr(&descrA);
*
*    hipsparseMatDescr_t descrB;
*    hipsparseCreateMatDescr(&descrB);
*
*    hipsparseMatDescr_t descrC;
*    hipsparseCreateMatDescr(&descrC);
*
*    int nnzC;
*    hipsparseXcsrgemmNnz(handle,
*                    transA,
*                    transB,
*                    m,
*                    n,
*                    k,
*                    descrA,
*                    nnzA,
*                    dcsrRowPtrA,
*                    dcsrColIndA,
*                    descrB,
*                    nnzB,
*                    dcsrRowPtrB,
*                    dcsrColIndB,
*                    descrC,
*                    dcsrRowPtrC,
*                    &nnzC);
*
*    int* dcsrColIndC = nullptr;
*    float* dcsrValC = nullptr;
*    hipMalloc((void**)&dcsrColIndC, sizeof(int) * nnzC);
*    hipMalloc((void**)&dcsrValC, sizeof(float) * nnzC);
*
*    hipsparseScsrgemm(handle, 
*                      transA, 
*                      transB, 
*                      m, 
*                      n, 
*                      k, 
*                      descrA, 
*                      nnzA, 
*                      dcsrValA, 
*                      dcsrRowPtrA, 
*                      dcsrColIndA, 
*                      descrB, 
*                      nnzB, 
*                      dcsrValB, 
*                      dcsrRowPtrB, 
*                      dcsrColIndB, 
*                      descrC, 
*                      dcsrValC, 
*                      dcsrRowPtrC, 
*                      dcsrColIndC);
*
*    hipFree(dcsrRowPtrA);
*    hipFree(dcsrColIndA);
*    hipFree(dcsrValA);
*    hipFree(dcsrRowPtrB);
*    hipFree(dcsrColIndB);
*    hipFree(dcsrValB);
*    hipFree(dcsrRowPtrC);
*    hipFree(dcsrColIndC);
*    hipFree(dcsrValC);
*
*    hipsparseDestroyMatDescr(descrA);
*    hipsparseDestroyMatDescr(descrB);
*    hipsparseDestroyMatDescr(descrC);
*    hipsparseDestroy(handle);
*  \endcode
*/
/**@{*/
DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseScsrgemm(hipsparseHandle_t         handle,
                                    hipsparseOperation_t      transA,
                                    hipsparseOperation_t      transB,
                                    int                       m,
                                    int                       n,
                                    int                       k,
                                    const hipsparseMatDescr_t descrA,
                                    int                       nnzA,
                                    const float*              csrValA,
                                    const int*                csrRowPtrA,
                                    const int*                csrColIndA,
                                    const hipsparseMatDescr_t descrB,
                                    int                       nnzB,
                                    const float*              csrValB,
                                    const int*                csrRowPtrB,
                                    const int*                csrColIndB,
                                    const hipsparseMatDescr_t descrC,
                                    float*                    csrValC,
                                    const int*                csrRowPtrC,
                                    int*                      csrColIndC);
DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDcsrgemm(hipsparseHandle_t         handle,
                                    hipsparseOperation_t      transA,
                                    hipsparseOperation_t      transB,
                                    int                       m,
                                    int                       n,
                                    int                       k,
                                    const hipsparseMatDescr_t descrA,
                                    int                       nnzA,
                                    const double*             csrValA,
                                    const int*                csrRowPtrA,
                                    const int*                csrColIndA,
                                    const hipsparseMatDescr_t descrB,
                                    int                       nnzB,
                                    const double*             csrValB,
                                    const int*                csrRowPtrB,
                                    const int*                csrColIndB,
                                    const hipsparseMatDescr_t descrC,
                                    double*                   csrValC,
                                    const int*                csrRowPtrC,
                                    int*                      csrColIndC);
DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCcsrgemm(hipsparseHandle_t         handle,
                                    hipsparseOperation_t      transA,
                                    hipsparseOperation_t      transB,
                                    int                       m,
                                    int                       n,
                                    int                       k,
                                    const hipsparseMatDescr_t descrA,
                                    int                       nnzA,
                                    const hipComplex*         csrValA,
                                    const int*                csrRowPtrA,
                                    const int*                csrColIndA,
                                    const hipsparseMatDescr_t descrB,
                                    int                       nnzB,
                                    const hipComplex*         csrValB,
                                    const int*                csrRowPtrB,
                                    const int*                csrColIndB,
                                    const hipsparseMatDescr_t descrC,
                                    hipComplex*               csrValC,
                                    const int*                csrRowPtrC,
                                    int*                      csrColIndC);
DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZcsrgemm(hipsparseHandle_t         handle,
                                    hipsparseOperation_t      transA,
                                    hipsparseOperation_t      transB,
                                    int                       m,
                                    int                       n,
                                    int                       k,
                                    const hipsparseMatDescr_t descrA,
                                    int                       nnzA,
                                    const hipDoubleComplex*   csrValA,
                                    const int*                csrRowPtrA,
                                    const int*                csrColIndA,
                                    const hipsparseMatDescr_t descrB,
                                    int                       nnzB,
                                    const hipDoubleComplex*   csrValB,
                                    const int*                csrRowPtrB,
                                    const int*                csrColIndB,
                                    const hipsparseMatDescr_t descrC,
                                    hipDoubleComplex*         csrValC,
                                    const int*                csrRowPtrC,
                                    int*                      csrColIndC);
/**@}*/
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 12000)
/*! \ingroup extra_module
*  \brief Sparse matrix sparse matrix multiplication using CSR storage format
*
*  \details
*  \p hipsparseXcsrgemm2_bufferSizeExt returns the size of the temporary storage buffer
*  in bytes that is required by \ref hipsparseXcsrgemm2Nnz() and \ref hipsparseScsrgemm2 
*  "hipsparseXcsrgemm2()". The temporary storage buffer must be allocated by the user.
*
*  \note
*  Please note, that for matrix products with more than 4096 non-zero entries per row,
*  additional temporary storage buffer is allocated by the algorithm.
*
*  \note
*  Please note, that for matrix products with more than 8192 intermediate products per
*  row, additional temporary storage buffer is allocated by the algorithm.
*
*  \note
*  Currently, only \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*
*  @param[in]
*  handle          handle to the hipsparse library context queue.
*  @param[in]
*  m               number of rows of the sparse CSR matrix \f$op(A)\f$ and \f$C\f$.
*  @param[in]
*  n               number of columns of the sparse CSR matrix \f$op(B)\f$ and
*                  \f$C\f$.
*  @param[in]
*  k               number of columns of the sparse CSR matrix \f$op(A)\f$ and number of
*                  rows of the sparse CSR matrix \f$op(B)\f$.
*  @param[in]
*  alpha           scalar \f$\alpha\f$.
*  @param[in]
*  descrA          descriptor of the sparse CSR matrix \f$A\f$. Currently, only
*                  \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*  @param[in]
*  nnzA            number of non-zero entries of the sparse CSR matrix \f$A\f$.
*  @param[in]
*  csrRowPtrA      array of \p m+1 elements (\f$op(A) == A\f$, \p k+1 otherwise)
*                  that point to the start of every row of the sparse CSR matrix
*                  \f$op(A)\f$.
*  @param[in]
*  csrColIndA      array of \p nnzA elements containing the column indices of the
*                  sparse CSR matrix \f$A\f$.
*  @param[in]
*  descrB          descriptor of the sparse CSR matrix \f$B\f$. Currently, only
*                  \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*  @param[in]
*  nnzB            number of non-zero entries of the sparse CSR matrix \f$B\f$.
*  @param[in]
*  csrRowPtrB      array of \p k+1 elements (\f$op(B) == B\f$, \p m+1 otherwise)
*                  that point to the start of every row of the sparse CSR matrix
*                  \f$op(B)\f$.
*  @param[in]
*  csrColIndB      array of \p nnzB elements containing the column indices of the
*                  sparse CSR matrix \f$B\f$.
*  @param[in]
*  beta            scalar \f$\beta\f$.
*  @param[in]
*  descrD          descriptor of the sparse CSR matrix \f$D\f$. Currently, only
*                  \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*  @param[in]
*  nnzD            number of non-zero entries of the sparse CSR matrix \f$D\f$.
*  @param[in]
*  csrRowPtrD      array of \p m+1 elements that point to the start of every row of the
*                  sparse CSR matrix \f$D\f$.
*  @param[in]
*  csrColIndD      array of \p nnzD elements containing the column indices of the sparse
*                  CSR matrix \f$D\f$.
*  @param[inout]
*  info            structure that holds meta data for the sparse CSR matrix \f$C\f$.
*  @param[out]
*  pBufferSizeInBytes number of bytes of the temporary storage buffer required by
*                     hipsparseXcsrgemm2Nnz(), hipsparseScsrgemm2(), hipsparseDcsrgemm2(),
*                     hipsparseCcsrgemm2() and hipsparseZcsrgemm2().
*
*  \retval HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p m, \p n, \p k, \p nnzA, \p nnzB, \p nnz_D, 
*          \p alpha, \p beta, \p descrA, \p csrRowPtrA, \p csrColIndA, \p descrB, \p csrRowPtrB, 
*          \p csrColIndB, \p descrD, \p csrRowPtrD, \p csrColIndD, \p info or \p pBufferSizeInBytes 
*          is invalid.
*  \retval HIPSPARSE_STATUS_NOT_SUPPORTED
*          \ref hipsparseMatrixType_t != \ref HIPSPARSE_MATRIX_TYPE_GENERAL.
*/
/**@{*/
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseScsrgemm2_bufferSizeExt(hipsparseHandle_t         handle,
                                                   int                       m,
                                                   int                       n,
                                                   int                       k,
                                                   const float*              alpha,
                                                   const hipsparseMatDescr_t descrA,
                                                   int                       nnzA,
                                                   const int*                csrRowPtrA,
                                                   const int*                csrColIndA,
                                                   const hipsparseMatDescr_t descrB,
                                                   int                       nnzB,
                                                   const int*                csrRowPtrB,
                                                   const int*                csrColIndB,
                                                   const float*              beta,
                                                   const hipsparseMatDescr_t descrD,
                                                   int                       nnzD,
                                                   const int*                csrRowPtrD,
                                                   const int*                csrColIndD,
                                                   csrgemm2Info_t            info,
                                                   size_t*                   pBufferSizeInBytes);
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDcsrgemm2_bufferSizeExt(hipsparseHandle_t         handle,
                                                   int                       m,
                                                   int                       n,
                                                   int                       k,
                                                   const double*             alpha,
                                                   const hipsparseMatDescr_t descrA,
                                                   int                       nnzA,
                                                   const int*                csrRowPtrA,
                                                   const int*                csrColIndA,
                                                   const hipsparseMatDescr_t descrB,
                                                   int                       nnzB,
                                                   const int*                csrRowPtrB,
                                                   const int*                csrColIndB,
                                                   const double*             beta,
                                                   const hipsparseMatDescr_t descrD,
                                                   int                       nnzD,
                                                   const int*                csrRowPtrD,
                                                   const int*                csrColIndD,
                                                   csrgemm2Info_t            info,
                                                   size_t*                   pBufferSizeInBytes);
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCcsrgemm2_bufferSizeExt(hipsparseHandle_t         handle,
                                                   int                       m,
                                                   int                       n,
                                                   int                       k,
                                                   const hipComplex*         alpha,
                                                   const hipsparseMatDescr_t descrA,
                                                   int                       nnzA,
                                                   const int*                csrRowPtrA,
                                                   const int*                csrColIndA,
                                                   const hipsparseMatDescr_t descrB,
                                                   int                       nnzB,
                                                   const int*                csrRowPtrB,
                                                   const int*                csrColIndB,
                                                   const hipComplex*         beta,
                                                   const hipsparseMatDescr_t descrD,
                                                   int                       nnzD,
                                                   const int*                csrRowPtrD,
                                                   const int*                csrColIndD,
                                                   csrgemm2Info_t            info,
                                                   size_t*                   pBufferSizeInBytes);
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZcsrgemm2_bufferSizeExt(hipsparseHandle_t         handle,
                                                   int                       m,
                                                   int                       n,
                                                   int                       k,
                                                   const hipDoubleComplex*   alpha,
                                                   const hipsparseMatDescr_t descrA,
                                                   int                       nnzA,
                                                   const int*                csrRowPtrA,
                                                   const int*                csrColIndA,
                                                   const hipsparseMatDescr_t descrB,
                                                   int                       nnzB,
                                                   const int*                csrRowPtrB,
                                                   const int*                csrColIndB,
                                                   const hipDoubleComplex*   beta,
                                                   const hipsparseMatDescr_t descrD,
                                                   int                       nnzD,
                                                   const int*                csrRowPtrD,
                                                   const int*                csrColIndD,
                                                   csrgemm2Info_t            info,
                                                   size_t*                   pBufferSizeInBytes);
/**@}*/
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 12000)
/*! \ingroup extra_module
*  \brief Sparse matrix sparse matrix multiplication using CSR storage format
*
*  \details
*  \p hipsparseXcsrgemm2Nnz computes the total CSR non-zero elements and the CSR row
*  offsets, that point to the start of every row of the sparse CSR matrix, of the
*  resulting multiplied matrix C. It is assumed that \p csrRowPtrC has been allocated
*  with size \p m+1. The required buffer size can be obtained by 
*  \ref hipsparseScsrgemm2_bufferSizeExt "hipsparseXcsrgemm2_bufferSizeExt()". The desired 
*  index base in the output CSR matrix C is set in the \ref hipsparseMatDescr_t \p descrC. 
*  See \ref hipsparseSetMatIndexBase().
*
*  \note
*  As indicated, \p nnzTotalDevHostPtr can point either to host or device memory. This is controlled 
*  by setting the pointer mode. See \ref hipsparseSetPointerMode().
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*
*  \note
*  Please note, that for matrix products with more than 8192 intermediate products per
*  row, additional temporary storage buffer is allocated by the algorithm.
*
*  \note
*  Currently, only \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*
*  @param[in]
*  handle          handle to the hipsparse library context queue.
*  @param[in]
*  m               number of rows of the sparse CSR matrix \f$op(A)\f$ and \f$C\f$.
*  @param[in]
*  n               number of columns of the sparse CSR matrix \f$op(B)\f$ and
*                  \f$C\f$.
*  @param[in]
*  k               number of columns of the sparse CSR matrix \f$op(A)\f$ and number of
*                  rows of the sparse CSR matrix \f$op(B)\f$.
*  @param[in]
*  descrA          descriptor of the sparse CSR matrix \f$A\f$. Currently, only
*                  \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*  @param[in]
*  nnzA            number of non-zero entries of the sparse CSR matrix \f$A\f$.
*  @param[in]
*  csrRowPtrA      array of \p m+1 elements (\f$op(A) == A\f$, \p k+1 otherwise)
*                  that point to the start of every row of the sparse CSR matrix
*                  \f$op(A)\f$.
*  @param[in]
*  csrColIndA      array of \p nnzA elements containing the column indices of the
*                  sparse CSR matrix \f$A\f$.
*  @param[in]
*  descrB          descriptor of the sparse CSR matrix \f$B\f$. Currently, only
*                  \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*  @param[in]
*  nnzB            number of non-zero entries of the sparse CSR matrix \f$B\f$.
*  @param[in]
*  csrRowPtrB      array of \p k+1 elements (\f$op(B) == B\f$, \p m+1 otherwise)
*                  that point to the start of every row of the sparse CSR matrix
*                  \f$op(B)\f$.
*  @param[in]
*  csrColIndB      array of \p nnzB elements containing the column indices of the
*                  sparse CSR matrix \f$B\f$.
*  @param[in]
*  descrD          descriptor of the sparse CSR matrix \f$D\f$. Currently, only
*                  \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*  @param[in]
*  nnzD            number of non-zero entries of the sparse CSR matrix \f$D\f$.
*  @param[in]
*  csrRowPtrD      array of \p m+1 elements that point to the start of every row of the
*                  sparse CSR matrix \f$D\f$.
*  @param[in]
*  csrColIndD      array of \p nnzD elements containing the column indices of the sparse
*                  CSR matrix \f$D\f$.
*  @param[in]
*  descrC          descriptor of the sparse CSR matrix \f$C\f$. Currently, only
*                  \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*  @param[out]
*  csrRowPtrC      array of \p m+1 elements that point to the start of every row of the
*                  sparse CSR matrix \f$C\f$.
*  @param[out]
*  nnzTotalDevHostPtr pointer to the number of non-zero entries of the sparse CSR
*                     matrix \f$C\f$.
*  @param[in]
*  info            structure that holds meta data for the sparse CSR matrix \f$C\f$.
*  @param[in]
*  pBuffer         temporary storage buffer allocated by the user, size is returned
*                  by hipsparseScsrgemm2_bufferSizeExt(), hipsparseDcsrgemm2_bufferSizeExt(), 
*                  hipsparseZcsrgemm2_bufferSizeExt() or hipsparseZcsrgemm2_bufferSizeExt().
*
*  \retval HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p m, \p n, \p k, \p nnzA, \p nnzB, \p nnzD,
*          \p descrA, \p csrRowPtrA, \p csrColIndA, \p descrB, \p csrRowPtrB, \p csrColIndB,
*          \p descrD, \p csrRowPtrD, \p csrColIndD, \p descrC, \p csrRowPtrC, \p nnzTotalDevHostPtr, 
*          \p info or \p pBuffer is invalid.
*  \retval HIPSPARSE_STATUS_ALLOC_FAILED additional buffer for long rows could not be
*          allocated.
*  \retval HIPSPARSE_STATUS_NOT_SUPPORTED
*          \ref hipsparseMatrixType_t != \ref HIPSPARSE_MATRIX_TYPE_GENERAL.
*/
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseXcsrgemm2Nnz(hipsparseHandle_t         handle,
                                        int                       m,
                                        int                       n,
                                        int                       k,
                                        const hipsparseMatDescr_t descrA,
                                        int                       nnzA,
                                        const int*                csrRowPtrA,
                                        const int*                csrColIndA,
                                        const hipsparseMatDescr_t descrB,
                                        int                       nnzB,
                                        const int*                csrRowPtrB,
                                        const int*                csrColIndB,
                                        const hipsparseMatDescr_t descrD,
                                        int                       nnzD,
                                        const int*                csrRowPtrD,
                                        const int*                csrColIndD,
                                        const hipsparseMatDescr_t descrC,
                                        int*                      csrRowPtrC,
                                        int*                      nnzTotalDevHostPtr,
                                        const csrgemm2Info_t      info,
                                        void*                     pBuffer);
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 12000)
/*! \ingroup extra_module
*  \brief Sparse matrix sparse matrix multiplication using CSR storage format
*
*  \details
*  \p hipsparseXcsrgemm2 multiplies the scalar \f$\alpha\f$ with the sparse
*  \f$m \times k\f$ matrix \f$A\f$, defined in CSR storage format, and the sparse
*  \f$k \times n\f$ matrix \f$B\f$, defined in CSR storage format, and adds the result
*  to the sparse \f$m \times n\f$ matrix \f$D\f$ that is multiplied by \f$\beta\f$. The
*  final result is stored in the sparse \f$m \times n\f$ matrix \f$C\f$, defined in CSR
*  storage format, such
*  that
*  \f[
*    C := \alpha \cdot A \cdot B + \beta \cdot D
*  \f]
*
*  This computation involves a multi step process. First the user must call
*  \ref hipsparseScsrgemm2_bufferSizeExt "hipsparseXcsrgemm2_bufferSizeExt()" in order to 
*  determine the required user allocated temporary buffer size. The user then allocates this 
*  buffer and also allocates \p csrRowPtrC to have size \p m+1. Both the temporary storage 
*  buffer and \p csrRowPtrC array are then passed to \ref hipsparseXcsrgemm2Nnz which fills 
*  in the \p csrRowPtrC array as well as computes the total number of nonzeros in C, \p nnzC. 
*  The user then allocates both arrays \p csrColIndC and \p csrValC to have size \p nnzC and 
*  calls \p hipsparseXcsrgemm2 to complete the computation. The desired index base in the output 
*  CSR matrix C is set in the \ref hipsparseMatDescr_t \p descrC. See \ref hipsparseSetMatIndexBase().
*
*  \note If \f$\alpha == 0\f$, then \f$C = \beta \cdot D\f$ will be computed.
*  \note If \f$\beta == 0\f$, then \f$C = \alpha \cdot A \cdot B\f$ will be computed.
*  \note \f$\alpha == beta == 0\f$ is invalid.
*  \note Currently, only \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*  \note This function is non blocking and executed asynchronously with respect to the
*        host. It may return before the actual computation has finished.
*  \note Please note, that for matrix products with more than 4096 non-zero entries per
*  row, additional temporary storage buffer is allocated by the algorithm.
*
*  @param[in]
*  handle          handle to the hipsparse library context queue.
*  @param[in]
*  m               number of rows of the sparse CSR matrix \f$op(A)\f$ and \f$C\f$.
*  @param[in]
*  n               number of columns of the sparse CSR matrix \f$op(B)\f$ and
*                  \f$C\f$.
*  @param[in]
*  k               number of columns of the sparse CSR matrix \f$op(A)\f$ and number of
*                  rows of the sparse CSR matrix \f$op(B)\f$.
*  @param[in]
*  alpha           scalar \f$\alpha\f$.
*  @param[in]
*  descrA          descriptor of the sparse CSR matrix \f$A\f$. Currently, only
*                  \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*  @param[in]
*  nnzA            number of non-zero entries of the sparse CSR matrix \f$A\f$.
*  @param[in]
*  csrValA         array of \p nnzA elements of the sparse CSR matrix \f$A\f$.
*  @param[in]
*  csrRowPtrA      array of \p m+1 elements (\f$op(A) == A\f$, \p k+1 otherwise)
*                  that point to the start of every row of the sparse CSR matrix
*                  \f$op(A)\f$.
*  @param[in]
*  csrColIndA      array of \p nnzA elements containing the column indices of the
*                  sparse CSR matrix \f$A\f$.
*  @param[in]
*  descrB          descriptor of the sparse CSR matrix \f$B\f$. Currently, only
*                  \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*  @param[in]
*  nnzB            number of non-zero entries of the sparse CSR matrix \f$B\f$.
*  @param[in]
*  csrValB         array of \p nnzB elements of the sparse CSR matrix \f$B\f$.
*  @param[in]
*  csrRowPtrB      array of \p k+1 elements (\f$op(B) == B\f$, \p m+1 otherwise)
*                  that point to the start of every row of the sparse CSR matrix
*                  \f$op(B)\f$.
*  @param[in]
*  csrColIndB      array of \p nnzB elements containing the column indices of the
*                  sparse CSR matrix \f$B\f$.
*  @param[in]
*  beta            scalar \f$\beta\f$.
*  @param[in]
*  descrD          descriptor of the sparse CSR matrix \f$D\f$. Currently, only
*                  \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*  @param[in]
*  nnzD            number of non-zero entries of the sparse CSR matrix \f$D\f$.
*  @param[in]
*  csrValD         array of \p nnzD elements of the sparse CSR matrix \f$D\f$.
*  @param[in]
*  csrRowPtrD      array of \p m+1 elements that point to the start of every row of the
*                  sparse CSR matrix \f$D\f$.
*  @param[in]
*  csrColIndD      array of \p nnzD elements containing the column indices of the
*                  sparse CSR matrix \f$D\f$.
*  @param[in]
*  descrC          descriptor of the sparse CSR matrix \f$C\f$. Currently, only
*                  \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*  @param[out]
*  csrValC         array of \p nnzC elements of the sparse CSR matrix \f$C\f$.
*  @param[in]
*  csrRowPtrC      array of \p m+1 elements that point to the start of every row of the
*                  sparse CSR matrix \f$C\f$.
*  @param[out]
*  csrColIndC      array of \p nnzC elements containing the column indices of the
*                  sparse CSR matrix \f$C\f$.
*  @param[in]
*  info            structure that holds meta data for the sparse CSR matrix \f$C\f$.
*  @param[in]
*  pBuffer         temporary storage buffer allocated by the user, size is returned
*                  by hipsparseScsrgemm2_bufferSizeExt(), hipsparseDcsrgemm2_bufferSizeExt(), 
*                  hipsparseCcsrgemm2_bufferSizeExt() or hipsparseZcsrgemm2_bufferSizeExt().
*
*  \retval HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p m, \p n, \p k, \p nnzA, \p nnzB, 
*          \p nnzD, \p alpha, \p beta, \p descrA, \p csrValA, \p csrRowPtrA, \p csrColIndA, 
*          \p descrB, \p csrValB, \p csrRowPtrB, \p csrColIndB, \p descrD, \p csrValD, 
*          \p csrRowPtrD, \p csrColIndD, \p csrValC, \p csrRowPtrC, \p csrColIndC, \p info 
*          or \p pBuffer is invalid.
*  \retval HIPSPARSE_STATUS_ALLOC_FAILED additional buffer for long rows could not be
*          allocated.
*  \retval HIPSPARSE_STATUS_NOT_SUPPORTED
*          \ref hipsparseMatrixType_t != \ref HIPSPARSE_MATRIX_TYPE_GENERAL.
*
*  \par Example
*  \code{.c}
*    int m = 4;
*    int k = 3;
*    int n = 2;
*    int nnzA = 7;
*    int nnzB = 3;
*    int nnzD = 6;
*
*    float alpha{1.0f};
*    float beta{1.0f};
*
*    // A, B, and C are mxk, kxn, and m×n
*
*    // A
*    // 1 0 0
*    // 3 4 0
*    // 5 6 7
*    // 0 0 9
*    std::vector<int> hcsrRowPtrA = {0, 1, 3, 6, 7};
*    std::vector<int> hcsrColIndA = {0, 0, 1, 0, 1, 2, 2};
*    std::vector<float> hcsrValA = {1.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 9.0f};
*
*    // B
*    // 0 1
*    // 1 0
*    // 0 1
*    std::vector<int> hcsrRowPtrB = {0, 1, 2, 3};
*    std::vector<int> hcsrColIndB = {1, 0, 1};
*    std::vector<float> hcsrValB = {1.0f, 1.0f, 1.0f};
*
*    // D
*    // 0 1
*    // 2 3
*    // 4 5
*    // 0 6
*    std::vector<int> hcsrRowPtrD = {0, 1, 3, 5, 6};
*    std::vector<int> hcsrColIndD = {1, 0, 1, 0, 1, 1};
*    std::vector<float> hcsrValD = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
*
*    // Device memory management: Allocate and copy A, B
*    int* dcsrRowPtrA;
*    int* dcsrColIndA;
*    float* dcsrValA;
*    int* dcsrRowPtrB;
*    int* dcsrColIndB;
*    float* dcsrValB;
*    int* dcsrRowPtrD;
*    int* dcsrColIndD;
*    float* dcsrValD;
*    int* dcsrRowPtrC;
*    hipMalloc((void**)&dcsrRowPtrA, (m + 1) * sizeof(int));
*    hipMalloc((void**)&dcsrColIndA, nnzA * sizeof(int));
*    hipMalloc((void**)&dcsrValA, nnzA * sizeof(float));
*    hipMalloc((void**)&dcsrRowPtrB, (k + 1) * sizeof(int));
*    hipMalloc((void**)&dcsrColIndB, nnzB * sizeof(int));
*    hipMalloc((void**)&dcsrValB, nnzB * sizeof(float));
*    hipMalloc((void**)&dcsrRowPtrD, (m + 1) * sizeof(int));
*    hipMalloc((void**)&dcsrColIndD, nnzD * sizeof(int));
*    hipMalloc((void**)&dcsrValD, nnzD * sizeof(float));
*    hipMalloc((void**)&dcsrRowPtrC, (m + 1) * sizeof(int));
*
*    hipMemcpy(dcsrRowPtrA, hcsrRowPtrA.data(), (m + 1) * sizeof(int), hipMemcpyHostToDevice);
*    hipMemcpy(dcsrColIndA, hcsrColIndA.data(), nnzA * sizeof(int), hipMemcpyHostToDevice);
*    hipMemcpy(dcsrValA, hcsrValA.data(), nnzA * sizeof(float), hipMemcpyHostToDevice);
*    hipMemcpy(dcsrRowPtrB, hcsrRowPtrB.data(), (k + 1) * sizeof(int), hipMemcpyHostToDevice);
*    hipMemcpy(dcsrColIndB, hcsrColIndB.data(), nnzB * sizeof(int), hipMemcpyHostToDevice);
*    hipMemcpy(dcsrValB, hcsrValB.data(), nnzB * sizeof(float), hipMemcpyHostToDevice);
*    hipMemcpy(dcsrRowPtrD, hcsrRowPtrD.data(), (m + 1) * sizeof(int), hipMemcpyHostToDevice);
*    hipMemcpy(dcsrColIndD, hcsrColIndD.data(), nnzD * sizeof(int), hipMemcpyHostToDevice);
*    hipMemcpy(dcsrValD, hcsrValD.data(), nnzD * sizeof(float), hipMemcpyHostToDevice);
*
*    hipsparseHandle_t handle;
*    hipsparseCreate(&handle);
*
*    hipsparseMatDescr_t descrA;
*    hipsparseCreateMatDescr(&descrA);
*
*    hipsparseMatDescr_t descrB;
*    hipsparseCreateMatDescr(&descrB);
*
*    hipsparseMatDescr_t descrC;
*    hipsparseCreateMatDescr(&descrC);
*
*    hipsparseMatDescr_t descrD;
*    hipsparseCreateMatDescr(&descrD);
*
*    csrgemm2Info_t info;
*    hipsparseCreateCsrgemm2Info(&info);
*
*    size_t bufferSize;
*    hipsparseScsrgemm2_bufferSizeExt(handle,
*                                     m,
*                                     n,
*                                     k,
*                                     &alpha,
*                                     descrA,
*                                     nnzA,
*                                     dcsrRowPtrA,
*                                     dcsrColIndA,
*                                     descrB,
*                                     nnzB,
*                                     dcsrRowPtrB,
*                                     dcsrColIndB,
*                                     &beta,
*                                     descrD,
*                                     nnzD,
*                                     dcsrRowPtrD,
*                                     dcsrColIndD,
*                                     info,
*                                     &bufferSize);
*
*    void* dbuffer = nullptr;
*    hipMalloc((void**)&dbuffer, bufferSize);
*
*    int nnzC;
*    hipsparseXcsrgemm2Nnz(handle,
*                    m,
*                    n,
*                    k,
*                    descrA,
*                    nnzA,
*                    dcsrRowPtrA,
*                    dcsrColIndA,
*                    descrB,
*                    nnzB,
*                    dcsrRowPtrB,
*                    dcsrColIndB,
*                    descrD,
*                    nnzD,
*                    dcsrRowPtrD,
*                    dcsrColIndD,
*                    descrC,
*                    dcsrRowPtrC,
*                    &nnzC,
*                    info,
*                    dbuffer);
*
*    int* dcsrColIndC = nullptr;
*    float* dcsrValC = nullptr;
*    hipMalloc((void**)&dcsrColIndC, sizeof(int) * nnzC);
*    hipMalloc((void**)&dcsrValC, sizeof(float) * nnzC);
*
*    hipsparseScsrgemm2(handle, 
*                      m, 
*                      n, 
*                      k,
*                      &alpha, 
*                      descrA, 
*                      nnzA, 
*                      dcsrValA, 
*                      dcsrRowPtrA, 
*                      dcsrColIndA, 
*                      descrB, 
*                      nnzB, 
*                      dcsrValB, 
*                      dcsrRowPtrB, 
*                      dcsrColIndB, 
*                      &beta,
*                      descrD,
*                      nnzD,
*                      dcsrValD, 
*                      dcsrRowPtrD, 
*                      dcsrColIndD, 
*                      descrC, 
*                      dcsrValC, 
*                      dcsrRowPtrC, 
*                      dcsrColIndC,
*                      info,
*                      dbuffer);
*
*    hipFree(dcsrRowPtrA);
*    hipFree(dcsrColIndA);
*    hipFree(dcsrValA);
*    hipFree(dcsrRowPtrB);
*    hipFree(dcsrColIndB);
*    hipFree(dcsrValB);
*    hipFree(dcsrRowPtrC);
*    hipFree(dcsrColIndC);
*    hipFree(dcsrValC);
*    hipFree(dcsrRowPtrD);
*    hipFree(dcsrColIndD);
*    hipFree(dcsrValD);
*
*    hipFree(dbuffer);
*
*    hipsparseDestroyMatDescr(descrA);
*    hipsparseDestroyMatDescr(descrB);
*    hipsparseDestroyMatDescr(descrC);
*    hipsparseDestroyMatDescr(descrD);
*    hipsparseDestroyCsrgemm2Info(info);
*
*    hipsparseDestroy(handle);
*  \endcode
*/
/**@{*/
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseScsrgemm2(hipsparseHandle_t         handle,
                                     int                       m,
                                     int                       n,
                                     int                       k,
                                     const float*              alpha,
                                     const hipsparseMatDescr_t descrA,
                                     int                       nnzA,
                                     const float*              csrValA,
                                     const int*                csrRowPtrA,
                                     const int*                csrColIndA,
                                     const hipsparseMatDescr_t descrB,
                                     int                       nnzB,
                                     const float*              csrValB,
                                     const int*                csrRowPtrB,
                                     const int*                csrColIndB,
                                     const float*              beta,
                                     const hipsparseMatDescr_t descrD,
                                     int                       nnzD,
                                     const float*              csrValD,
                                     const int*                csrRowPtrD,
                                     const int*                csrColIndD,
                                     const hipsparseMatDescr_t descrC,
                                     float*                    csrValC,
                                     const int*                csrRowPtrC,
                                     int*                      csrColIndC,
                                     const csrgemm2Info_t      info,
                                     void*                     pBuffer);
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDcsrgemm2(hipsparseHandle_t         handle,
                                     int                       m,
                                     int                       n,
                                     int                       k,
                                     const double*             alpha,
                                     const hipsparseMatDescr_t descrA,
                                     int                       nnzA,
                                     const double*             csrValA,
                                     const int*                csrRowPtrA,
                                     const int*                csrColIndA,
                                     const hipsparseMatDescr_t descrB,
                                     int                       nnzB,
                                     const double*             csrValB,
                                     const int*                csrRowPtrB,
                                     const int*                csrColIndB,
                                     const double*             beta,
                                     const hipsparseMatDescr_t descrD,
                                     int                       nnzD,
                                     const double*             csrValD,
                                     const int*                csrRowPtrD,
                                     const int*                csrColIndD,
                                     const hipsparseMatDescr_t descrC,
                                     double*                   csrValC,
                                     const int*                csrRowPtrC,
                                     int*                      csrColIndC,
                                     const csrgemm2Info_t      info,
                                     void*                     pBuffer);
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCcsrgemm2(hipsparseHandle_t         handle,
                                     int                       m,
                                     int                       n,
                                     int                       k,
                                     const hipComplex*         alpha,
                                     const hipsparseMatDescr_t descrA,
                                     int                       nnzA,
                                     const hipComplex*         csrValA,
                                     const int*                csrRowPtrA,
                                     const int*                csrColIndA,
                                     const hipsparseMatDescr_t descrB,
                                     int                       nnzB,
                                     const hipComplex*         csrValB,
                                     const int*                csrRowPtrB,
                                     const int*                csrColIndB,
                                     const hipComplex*         beta,
                                     const hipsparseMatDescr_t descrD,
                                     int                       nnzD,
                                     const hipComplex*         csrValD,
                                     const int*                csrRowPtrD,
                                     const int*                csrColIndD,
                                     const hipsparseMatDescr_t descrC,
                                     hipComplex*               csrValC,
                                     const int*                csrRowPtrC,
                                     int*                      csrColIndC,
                                     const csrgemm2Info_t      info,
                                     void*                     pBuffer);
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZcsrgemm2(hipsparseHandle_t         handle,
                                     int                       m,
                                     int                       n,
                                     int                       k,
                                     const hipDoubleComplex*   alpha,
                                     const hipsparseMatDescr_t descrA,
                                     int                       nnzA,
                                     const hipDoubleComplex*   csrValA,
                                     const int*                csrRowPtrA,
                                     const int*                csrColIndA,
                                     const hipsparseMatDescr_t descrB,
                                     int                       nnzB,
                                     const hipDoubleComplex*   csrValB,
                                     const int*                csrRowPtrB,
                                     const int*                csrColIndB,
                                     const hipDoubleComplex*   beta,
                                     const hipsparseMatDescr_t descrD,
                                     int                       nnzD,
                                     const hipDoubleComplex*   csrValD,
                                     const int*                csrRowPtrD,
                                     const int*                csrColIndD,
                                     const hipsparseMatDescr_t descrC,
                                     hipDoubleComplex*         csrValC,
                                     const int*                csrRowPtrC,
                                     int*                      csrColIndC,
                                     const csrgemm2Info_t      info,
                                     void*                     pBuffer);
/**@}*/
#endif

/*
* ===========================================================================
*    preconditioner SPARSE
* ===========================================================================
*/

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
/*! \ingroup precond_module
 *  \brief Incomplete LU factorization with 0 fill-ins and no pivoting using BSR storage
 *  format
 *
 *  \details
 *  \p hipsparseXbsrilu02_zeroPivot returns \ref HIPSPARSE_STATUS_ZERO_PIVOT, if either a
 *  structural or numerical zero has been found during \ref hipsparseSbsrilu02_analysis 
 *  "hipsparseXbsrilu02_analysis()" or \ref hipsparseSbsrilu02 "hipsparseXbsrilu02()" 
 *  computation. The first zero pivot \f$j\f$ at \f$A_{j,j}\f$ is stored in \p position, 
 *  using same index base as the BSR matrix.
 *
 *  \p position can be in host or device memory. If no zero pivot has been found,
 *  \p position is set to -1 and \ref HIPSPARSE_STATUS_SUCCESS is returned instead.
 *
 *  \note
 *  If a zero pivot is found, \p position \f$=j\f$ means that either the diagonal block
 *  \f$A_{j,j}\f$ is missing (structural zero) or the diagonal block \f$A_{j,j}\f$ is not
 *  invertible (numerical zero).
 *
 *  \note \p hipsparseXbsrilu02_zeroPivot is a blocking function. It might influence
 *  performance negatively.
 *
 *  @param[in]
 *  handle      handle to the hipsparse library context queue.
 *  @param[in]
 *  info        structure that holds the information collected during the analysis step.
 *  @param[inout]
 *  position    pointer to zero pivot \f$j\f$, can be in host or device memory.
 *
 *  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
 *  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p info or \p position pointer is
 *              invalid.
 *  \retval     HIPSPARSE_STATUS_INTERNAL_ERROR an internal error occurred.
 *  \retval     HIPSPARSE_STATUS_ZERO_PIVOT zero pivot has been found.
 */
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t
    hipsparseXbsrilu02_zeroPivot(hipsparseHandle_t handle, bsrilu02Info_t info, int* position);
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
/*! \ingroup precond_module
 *  \brief Incomplete LU factorization with 0 fill-ins and no pivoting using BSR storage
 *  format
 *
 *  \details
 *  \p hipsparseXbsrilu02_numericBoost enables the user to replace a numerical value in
 *  an incomplete LU factorization. \p tol is used to determine whether a numerical value
 *  is replaced by \p boost_val, such that \f$A_{j,j} = \text{boost_val}\f$ if
 *  \f$\text{tol} \ge \left|A_{j,j}\right|\f$.
 *
 *  \note The boost value is enabled by setting \p enable_boost to 1 or disabled by
 *  setting \p enable_boost to 0.
 *
 *  \note \p tol and \p boost_val can be in host or device memory.
 *
 *  @param[in]
 *  handle        handle to the hipsparse library context queue.
 *  @param[in]
 *  info          structure that holds the information collected during the analysis step.
 *  @param[in]
 *  enable_boost  enable/disable numeric boost.
 *  @param[in]
 *  tol           tolerance to determine whether a numerical value is replaced or not.
 *  @param[in]
 *  boost_val     boost value to replace a numerical value.
 *
 *  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
 *  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p info, \p tol or \p boost_val pointer
 *              is invalid.
 *  \retval     HIPSPARSE_STATUS_INTERNAL_ERROR an internal error occurred.
*/
/**@{*/
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSbsrilu02_numericBoost(
    hipsparseHandle_t handle, bsrilu02Info_t info, int enable_boost, double* tol, float* boost_val);

DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDbsrilu02_numericBoost(hipsparseHandle_t handle,
                                                  bsrilu02Info_t    info,
                                                  int               enable_boost,
                                                  double*           tol,
                                                  double*           boost_val);

DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCbsrilu02_numericBoost(hipsparseHandle_t handle,
                                                  bsrilu02Info_t    info,
                                                  int               enable_boost,
                                                  double*           tol,
                                                  hipComplex*       boost_val);

DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZbsrilu02_numericBoost(hipsparseHandle_t handle,
                                                  bsrilu02Info_t    info,
                                                  int               enable_boost,
                                                  double*           tol,
                                                  hipDoubleComplex* boost_val);
/**@}*/
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
/*! \ingroup precond_module
 *  \brief Incomplete LU factorization with 0 fill-ins and no pivoting using BSR storage
 *  format
 *
 *  \details
 *  \p hipsparseXbsrilu02_bufferSize returns the size of the temporary storage buffer
 *  in bytes that is required by \ref hipsparseSbsrilu02_analysis "hipsparseXbsrilu02_analysis()" 
 *  and \ref hipsparseSbsrilu02 "hipsparseXbsrilu02()". The temporary storage buffer must be 
 *  allocated by the user.
 *
 *  @param[in]
 *  handle             handle to the hipsparse library context queue.
 *  @param[in]
 *  dirA               direction that specifies whether to count nonzero elements by \ref HIPSPARSE_DIRECTION_ROW 
 *                     or by \ref HIPSPARSE_DIRECTION_COLUMN.
 *  @param[in]
 *  mb                 number of block rows in the sparse BSR matrix.
 *  @param[in]
 *  nnzb               number of non-zero block entries of the sparse BSR matrix.
 *  @param[in]
 *  descrA             descriptor of the sparse BSR matrix.
 *  @param[in]
 *  bsrSortedValA      array of length \p nnzb*blockDim*blockDim containing the values of the sparse BSR matrix.
 *  @param[in]
 *  bsrSortedRowPtrA   array of \p mb+1 elements that point to the start of every block row of the
 *                     sparse BSR matrix.
 *  @param[in]
 *  bsrSortedColIndA   array of \p nnzb elements containing the block column indices of the sparse BSR matrix.
 *  @param[in]
 *  blockDim           the block dimension of the BSR matrix. Between 1 and m where \p m=mb*blockDim.
 *  @param[out]
 *  info               structure that holds the information collected during the analysis step.
 *  @param[out]
 *  pBufferSizeInBytes number of bytes of the temporary storage buffer required by
 *                     hipsparseSbsrilu02_analysis(), hipsparseDbsrilu02_analysis(),
 *                     hipsparseCbsrilu02_analysis(), hipsparseZbsrilu02_analysis(),
 *                     hipsparseSbsrilu02(), hipsparseDbsrilu02(), hipsparseCbsrilu02()
 *                     and hipsparseZbsrilu02().
 *
 *  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
 *  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p mb, \p nnzb, \p blockDim, \p descrA, 
 *              \p bsrSortedValA, \p bsrSortedRowPtrA, \p bsrSortedColIndA, \p info or 
 *              \p pBufferSizeInBytes pointer is invalid.
 *  \retval     HIPSPARSE_STATUS_INTERNAL_ERROR an internal error occurred.
 *  \retval     HIPSPARSE_STATUS_NOT_SUPPORTED
 *              \ref hipsparseMatrixType_t != \ref HIPSPARSE_MATRIX_TYPE_GENERAL.
*/
/**@{*/
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSbsrilu02_bufferSize(hipsparseHandle_t         handle,
                                                hipsparseDirection_t      dirA,
                                                int                       mb,
                                                int                       nnzb,
                                                const hipsparseMatDescr_t descrA,
                                                float*                    bsrSortedValA,
                                                const int*                bsrSortedRowPtrA,
                                                const int*                bsrSortedColIndA,
                                                int                       blockDim,
                                                bsrilu02Info_t            info,
                                                int*                      pBufferSizeInBytes);
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDbsrilu02_bufferSize(hipsparseHandle_t         handle,
                                                hipsparseDirection_t      dirA,
                                                int                       mb,
                                                int                       nnzb,
                                                const hipsparseMatDescr_t descrA,
                                                double*                   bsrSortedValA,
                                                const int*                bsrSortedRowPtrA,
                                                const int*                bsrSortedColIndA,
                                                int                       blockDim,
                                                bsrilu02Info_t            info,
                                                int*                      pBufferSizeInBytes);
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCbsrilu02_bufferSize(hipsparseHandle_t         handle,
                                                hipsparseDirection_t      dirA,
                                                int                       mb,
                                                int                       nnzb,
                                                const hipsparseMatDescr_t descrA,
                                                hipComplex*               bsrSortedValA,
                                                const int*                bsrSortedRowPtrA,
                                                const int*                bsrSortedColIndA,
                                                int                       blockDim,
                                                bsrilu02Info_t            info,
                                                int*                      pBufferSizeInBytes);
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZbsrilu02_bufferSize(hipsparseHandle_t         handle,
                                                hipsparseDirection_t      dirA,
                                                int                       mb,
                                                int                       nnzb,
                                                const hipsparseMatDescr_t descrA,
                                                hipDoubleComplex*         bsrSortedValA,
                                                const int*                bsrSortedRowPtrA,
                                                const int*                bsrSortedColIndA,
                                                int                       blockDim,
                                                bsrilu02Info_t            info,
                                                int*                      pBufferSizeInBytes);
/**@}*/
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
/*! \ingroup precond_module
 *  \brief Incomplete LU factorization with 0 fill-ins and no pivoting using BSR storage
 *  format
 *
 *  \details
 *  \p hipsparseXbsrilu02_analysis performs the analysis step for \ref hipsparseSbsrilu02 
 *  "hipsparseXbsrilu02()".
 *
 *  \note
 *  If the matrix sparsity pattern changes, the gathered information will become invalid.
 *
 *  \note
 *  This function is non blocking and executed asynchronously with respect to the host.
 *  It may return before the actual computation has finished.
 *
 *  @param[in]
 *  handle           handle to the hipsparse library context queue.
 *  @param[in]
 *  dirA             direction that specified whether to count nonzero elements by
 *                   \ref HIPSPARSE_DIRECTION_ROW or by \ref HIPSPARSE_DIRECTION_COLUMN.
 *  @param[in]
 *  mb               number of block rows in the sparse BSR matrix.
 *  @param[in]
 *  nnzb             number of non-zero block entries of the sparse BSR matrix.
 *  @param[in]
 *  descrA           descriptor of the sparse BSR matrix.
 *  @param[in]
 *  bsrSortedValA    array of length \p nnzb*blockDim*blockDim containing the values of the sparse BSR matrix.
 *  @param[in]
 *  bsrSortedRowPtrA array of \p mb+1 elements that point to the start of every block row of the
 *                   sparse BSR matrix.
 *  @param[in]
 *  bsrSortedColIndA array of \p nnzb elements containing the block column indices of the sparse BSR matrix.
 *  @param[in]
 *  blockDim         the block dimension of the BSR matrix. Between 1 and m where \p m=mb*blockDim.
 *  @param[out]
 *  info             structure that holds the information collected during the analysis step.
 *  @param[in]
 *  policy           \ref HIPSPARSE_SOLVE_POLICY_NO_LEVEL or \ref HIPSPARSE_SOLVE_POLICY_USE_LEVEL.
 *  @param[in]
 *  pBuffer          temporary storage buffer allocated by the user.
 *
 *  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
 *  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p mb, \p nnzb, \p blockDim, \p descrA, 
 *              \p bsrSortedValA, \p bsrSortedRowPtrA, \p bsrSortedColIndA, \p info or \p pBuffer pointer 
 *              is invalid.
 *  \retval     HIPSPARSE_STATUS_INTERNAL_ERROR an internal error occurred.
 *  \retval     HIPSPARSE_STATUS_NOT_SUPPORTED
 *              \ref hipsparseMatrixType_t != \ref HIPSPARSE_MATRIX_TYPE_GENERAL.
 */
/**@{*/
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSbsrilu02_analysis(hipsparseHandle_t         handle,
                                              hipsparseDirection_t      dirA,
                                              int                       mb,
                                              int                       nnzb,
                                              const hipsparseMatDescr_t descrA,
                                              float*                    bsrSortedValA,
                                              const int*                bsrSortedRowPtrA,
                                              const int*                bsrSortedColIndA,
                                              int                       blockDim,
                                              bsrilu02Info_t            info,
                                              hipsparseSolvePolicy_t    policy,
                                              void*                     pBuffer);
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDbsrilu02_analysis(hipsparseHandle_t         handle,
                                              hipsparseDirection_t      dirA,
                                              int                       mb,
                                              int                       nnzb,
                                              const hipsparseMatDescr_t descrA,
                                              double*                   bsrSortedValA,
                                              const int*                bsrSortedRowPtrA,
                                              const int*                bsrSortedColIndA,
                                              int                       blockDim,
                                              bsrilu02Info_t            info,
                                              hipsparseSolvePolicy_t    policy,
                                              void*                     pBuffer);
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCbsrilu02_analysis(hipsparseHandle_t         handle,
                                              hipsparseDirection_t      dirA,
                                              int                       mb,
                                              int                       nnzb,
                                              const hipsparseMatDescr_t descrA,
                                              hipComplex*               bsrSortedValA,
                                              const int*                bsrSortedRowPtrA,
                                              const int*                bsrSortedColIndA,
                                              int                       blockDim,
                                              bsrilu02Info_t            info,
                                              hipsparseSolvePolicy_t    policy,
                                              void*                     pBuffer);
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZbsrilu02_analysis(hipsparseHandle_t         handle,
                                              hipsparseDirection_t      dirA,
                                              int                       mb,
                                              int                       nnzb,
                                              const hipsparseMatDescr_t descrA,
                                              hipDoubleComplex*         bsrSortedValA,
                                              const int*                bsrSortedRowPtrA,
                                              const int*                bsrSortedColIndA,
                                              int                       blockDim,
                                              bsrilu02Info_t            info,
                                              hipsparseSolvePolicy_t    policy,
                                              void*                     pBuffer);
/**@}*/
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
/*! \ingroup precond_module
 *  \brief Incomplete LU factorization with 0 fill-ins and no pivoting using BSR storage
 *  format
 *
 *  \details
 *  \p hipsparseXbsrilu02 computes the incomplete LU factorization with 0 fill-ins and no
 *  pivoting of a sparse \f$mb \times mb\f$ BSR matrix \f$A\f$, such that
 *  \f[
 *    A \approx LU
 *  \f]
 *
 *  \p hipsparseXbsrilu02 requires a user allocated temporary buffer. Its size is
 *  returned by \ref hipsparseSbsrilu02_bufferSize "hipsparseXbsrilu02_bufferSize()". 
 *  Furthermore, analysis meta data is required. It can be obtained by 
 *  \ref hipsparseSbsrilu02_analysis "hipsparseXbsrilu02_analysis()". \p hipsparseXbsrilu02
 *  reports the first zero pivot (either numerical or structural zero). The zero pivot
 *  status can be obtained by calling hipsparseXbsrilu02_zeroPivot().
 *
 *  \note
 *  This function is non blocking and executed asynchronously with respect to the host.
 *  It may return before the actual computation has finished.
 *
 *  @param[in]
 *  handle             handle to the hipsparse library context queue.
 *  @param[in]
 *  dirA               direction that specified whether to count nonzero elements by
 *                     \ref HIPSPARSE_DIRECTION_ROW or by \ref HIPSPARSE_DIRECTION_COLUMN.
 *  @param[in]
 *  mb                 number of block rows in the sparse BSR matrix.
 *  @param[in]
 *  nnzb               number of non-zero block entries of the sparse BSR matrix.
 *  @param[in]
 *  descrA             descriptor of the sparse BSR matrix.
 *  @param[inout]
 *  bsrSortedValA_valM array of length \p nnzb*blockDim*blockDim containing the values of the sparse BSR matrix.
 *  @param[in]
 *  bsrSortedRowPtrA   array of \p mb+1 elements that point to the start of every block row of the
 *                     sparse BSR matrix.
 *  @param[in]
 *  bsrSortedColIndA   array of \p nnzb elements containing the block column indices of the sparse BSR matrix.
 *  @param[in]
 *  blockDim           the block dimension of the BSR matrix. Between 1 and m where \p m=mb*blockDim.
 *  @param[in]
 *  info               structure that holds the information collected during the analysis step.
 *  @param[in]
 *  policy             \ref HIPSPARSE_SOLVE_POLICY_NO_LEVEL or \ref HIPSPARSE_SOLVE_POLICY_USE_LEVEL.
 *  @param[in]
 *  pBuffer            temporary storage buffer allocated by the user.
 *
 *  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
 *  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p mb, \p nnzb, \p blockDim, \p descrA, 
 *              \p bsrSortedValA_valM, \p bsrSortedRowPtrA or \p bsrSortedColIndA pointer is invalid.
 *  \retval     HIPSPARSE_STATUS_ARCH_MISMATCH the device is not supported.
 *  \retval     HIPSPARSE_STATUS_INTERNAL_ERROR an internal error occurred.
 *  \retval     HIPSPARSE_STATUS_NOT_SUPPORTED
 *              \ref hipsparseMatrixType_t != \ref HIPSPARSE_MATRIX_TYPE_GENERAL.
 */
/**@{*/
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSbsrilu02(hipsparseHandle_t         handle,
                                     hipsparseDirection_t      dirA,
                                     int                       mb,
                                     int                       nnzb,
                                     const hipsparseMatDescr_t descrA,
                                     float*                    bsrSortedValA_valM,
                                     /* matrix A values are updated inplace
                                        to be the preconditioner M values */
                                     const int*             bsrSortedRowPtrA,
                                     const int*             bsrSortedColIndA,
                                     int                    blockDim,
                                     bsrilu02Info_t         info,
                                     hipsparseSolvePolicy_t policy,
                                     void*                  pBuffer);
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDbsrilu02(hipsparseHandle_t         handle,
                                     hipsparseDirection_t      dirA,
                                     int                       mb,
                                     int                       nnzb,
                                     const hipsparseMatDescr_t descrA,
                                     double*                   bsrSortedValA_valM,
                                     /* matrix A values are updated inplace
                                        to be the preconditioner M values */
                                     const int*             bsrSortedRowPtrA,
                                     const int*             bsrSortedColIndA,
                                     int                    blockDim,
                                     bsrilu02Info_t         info,
                                     hipsparseSolvePolicy_t policy,
                                     void*                  pBuffer);
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCbsrilu02(hipsparseHandle_t         handle,
                                     hipsparseDirection_t      dirA,
                                     int                       mb,
                                     int                       nnzb,
                                     const hipsparseMatDescr_t descrA,
                                     hipComplex*               bsrSortedValA_valM,
                                     /* matrix A values are updated inplace
                                        to be the preconditioner M values */
                                     const int*             bsrSortedRowPtrA,
                                     const int*             bsrSortedColIndA,
                                     int                    blockDim,
                                     bsrilu02Info_t         info,
                                     hipsparseSolvePolicy_t policy,
                                     void*                  pBuffer);
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZbsrilu02(hipsparseHandle_t         handle,
                                     hipsparseDirection_t      dirA,
                                     int                       mb,
                                     int                       nnzb,
                                     const hipsparseMatDescr_t descrA,
                                     hipDoubleComplex*         bsrSortedValA_valM,
                                     /* matrix A values are updated inplace
                                        to be the preconditioner M values */
                                     const int*             bsrSortedRowPtrA,
                                     const int*             bsrSortedColIndA,
                                     int                    blockDim,
                                     bsrilu02Info_t         info,
                                     hipsparseSolvePolicy_t policy,
                                     void*                  pBuffer);
/**@}*/
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
/*! \ingroup precond_module
*  \brief Incomplete LU factorization with 0 fill-ins and no pivoting using CSR
*  storage format
*
*  \details
*  \p hipsparseXcsrilu02_zeroPivot returns \ref HIPSPARSE_STATUS_ZERO_PIVOT, if either a
*  structural or numerical zero has been found during \ref hipsparseScsrilu02 "hipsparseXcsrilu02()" 
*  computation. The first zero pivot \f$j\f$ at \f$A_{j,j}\f$ is stored in \p position, using same
*  index base as the CSR matrix.
*
*  \p position can be in host or device memory. If no zero pivot has been found,
*  \p position is set to -1 and \ref HIPSPARSE_STATUS_SUCCESS is returned instead.
*
*  \note \p hipsparseXcsrilu02_zeroPivot is a blocking function. It might influence
*  performance negatively.
*
*  @param[in]
*  handle      handle to the hipsparse library context queue.
*  @param[in]
*  info        structure that holds the information collected during the analysis step.
*  @param[inout]
*  position    pointer to zero pivot \f$j\f$, can be in host or device memory.
*
*  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p info or \p position pointer is
*              invalid.
*  \retval     HIPSPARSE_STATUS_INTERNAL_ERROR an internal error occurred.
*  \retval     HIPSPARSE_STATUS_ZERO_PIVOT zero pivot has been found.
*/
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t
    hipsparseXcsrilu02_zeroPivot(hipsparseHandle_t handle, csrilu02Info_t info, int* position);
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
/*! \ingroup precond_module
 *  \brief Incomplete LU factorization with 0 fill-ins and no pivoting using CSR storage
 *  format
 *
 *  \details
 *  \p hipsparseXcsrilu02_numericBoost enables the user to replace a numerical value in
 *  an incomplete LU factorization. \p tol is used to determine whether a numerical value
 *  is replaced by \p boost_val, such that \f$A_{j,j} = \text{boost_val}\f$ if
 *  \f$\text{tol} \ge \left|A_{j,j}\right|\f$.
 *
 *  \note The boost value is enabled by setting \p enable_boost to 1 or disabled by
 *  setting \p enable_boost to 0.
 *
 *  \note \p tol and \p boost_val can be in host or device memory.
 *
 *  @param[in]
 *  handle          handle to the hipsparse library context queue.
 *  @param[in]
 *  info            structure that holds the information collected during the analysis step.
 *  @param[in]
 *  enable_boost    enable/disable numeric boost.
 *  @param[in]
 *  tol             tolerance to determine whether a numerical value is replaced or not.
 *  @param[in]
 *  boost_val       boost value to replace a numerical value.
 *
 *  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
 *  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p info, \p tol or \p boost_val pointer
 *              is invalid.
 *  \retval     HIPSPARSE_STATUS_INTERNAL_ERROR an internal error occurred.
 */
/**@{*/
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseScsrilu02_numericBoost(
    hipsparseHandle_t handle, csrilu02Info_t info, int enable_boost, double* tol, float* boost_val);

DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDcsrilu02_numericBoost(hipsparseHandle_t handle,
                                                  csrilu02Info_t    info,
                                                  int               enable_boost,
                                                  double*           tol,
                                                  double*           boost_val);

DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCcsrilu02_numericBoost(hipsparseHandle_t handle,
                                                  csrilu02Info_t    info,
                                                  int               enable_boost,
                                                  double*           tol,
                                                  hipComplex*       boost_val);

DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZcsrilu02_numericBoost(hipsparseHandle_t handle,
                                                  csrilu02Info_t    info,
                                                  int               enable_boost,
                                                  double*           tol,
                                                  hipDoubleComplex* boost_val);
/**@}*/
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
/*! \ingroup precond_module
*  \brief Incomplete LU factorization with 0 fill-ins and no pivoting using CSR
*  storage format
*
*  \details
*  \p hipsparseXcsrilu02_bufferSize returns the size of the temporary storage buffer
*  in bytes that is required by \ref hipsparseScsrilu02_analysis "hipsparseXcsrilu02_analysis()" 
*  and \ref hipsparseScsrilu02 "hipsparseXcsrilu02()". The temporary storage buffer 
*  must be allocated by the user.
*
*  @param[in]
*  handle             handle to the hipsparse library context queue.
*  @param[in]
*  m                  number of rows of the sparse CSR matrix.
*  @param[in]
*  nnz                number of non-zero entries of the sparse CSR matrix.
*  @param[in]
*  descrA             descriptor of the sparse CSR matrix.
*  @param[in]
*  csrSortedValA      array of \p nnz elements of the sparse CSR matrix.
*  @param[in]
*  csrSortedRowPtrA   array of \p m+1 elements that point to the start of every row of the
*                     sparse CSR matrix.
*  @param[in]
*  csrSortedColIndA   array of \p nnz elements containing the column indices of the sparse
*                     CSR matrix.
*  @param[out]
*  info               structure that holds the information collected during the analysis step.
*  @param[out]
*  pBufferSizeInBytes number of bytes of the temporary storage buffer required by
*                     hipsparseScsrilu02_analysis(), hipsparseDcsrilu02_analysis(),
*                     hipsparseCcsrilu02_analysis(), hipsparseZcsrilu02_analysis(),
*                     hipsparseScsrilu02(), hipsparseDcsrilu02(), hipsparseCcsrilu02() and
*                     hipsparseZcsrilu02().
*
*  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p m, \p nnz, \p descrA, \p csrSortedValA, 
*              \p csrSortedRowPtrA, \p csrSortedColIndA, \p info or \p pBufferSizeInBytes pointer 
*              is invalid.
*  \retval     HIPSPARSE_STATUS_INTERNAL_ERROR an internal error occurred.
*/
/**@{*/
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseScsrilu02_bufferSize(hipsparseHandle_t         handle,
                                                int                       m,
                                                int                       nnz,
                                                const hipsparseMatDescr_t descrA,
                                                float*                    csrSortedValA,
                                                const int*                csrSortedRowPtrA,
                                                const int*                csrSortedColIndA,
                                                csrilu02Info_t            info,
                                                int*                      pBufferSizeInBytes);
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDcsrilu02_bufferSize(hipsparseHandle_t         handle,
                                                int                       m,
                                                int                       nnz,
                                                const hipsparseMatDescr_t descrA,
                                                double*                   csrSortedValA,
                                                const int*                csrSortedRowPtrA,
                                                const int*                csrSortedColIndA,
                                                csrilu02Info_t            info,
                                                int*                      pBufferSizeInBytes);
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCcsrilu02_bufferSize(hipsparseHandle_t         handle,
                                                int                       m,
                                                int                       nnz,
                                                const hipsparseMatDescr_t descrA,
                                                hipComplex*               csrSortedValA,
                                                const int*                csrSortedRowPtrA,
                                                const int*                csrSortedColIndA,
                                                csrilu02Info_t            info,
                                                int*                      pBufferSizeInBytes);
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZcsrilu02_bufferSize(hipsparseHandle_t         handle,
                                                int                       m,
                                                int                       nnz,
                                                const hipsparseMatDescr_t descrA,
                                                hipDoubleComplex*         csrSortedValA,
                                                const int*                csrSortedRowPtrA,
                                                const int*                csrSortedColIndA,
                                                csrilu02Info_t            info,
                                                int*                      pBufferSizeInBytes);
/**@}*/
#endif

/*! \ingroup precond_module
*  \brief Incomplete LU factorization with 0 fill-ins and no pivoting using CSR
*  storage format
*
*  \details
*  \p hipsparseXcsrilu02_bufferSizeExt returns the size of the temporary storage buffer
*  in bytes that is required by \ref hipsparseScsrilu02_analysis "hipsparseXcsrilu02_analysis()" 
*  and \ref hipsparseScsrilu02 "hipsparseXcsrilu02()". The temporary storage buffer 
*  must be allocated by the user.
*
*  @param[in]
*  handle             handle to the hipsparse library context queue.
*  @param[in]
*  m                  number of rows of the sparse CSR matrix.
*  @param[in]
*  nnz                number of non-zero entries of the sparse CSR matrix.
*  @param[in]
*  descrA             descriptor of the sparse CSR matrix.
*  @param[in]
*  csrSortedValA      array of \p nnz elements of the sparse CSR matrix.
*  @param[in]
*  csrSortedRowPtrA   array of \p m+1 elements that point to the start of every row of the
*                     sparse CSR matrix.
*  @param[in]
*  csrSortedColIndA   array of \p nnz elements containing the column indices of the sparse
*                     CSR matrix.
*  @param[out]
*  info               structure that holds the information collected during the analysis step.
*  @param[out]
*  pBufferSizeInBytes number of bytes of the temporary storage buffer required by
*                     hipsparseScsrilu02_analysis(), hipsparseDcsrilu02_analysis(),
*                     hipsparseCcsrilu02_analysis(), hipsparseZcsrilu02_analysis(),
*                     hipsparseScsrilu02(), hipsparseDcsrilu02(), hipsparseCcsrilu02() and
*                     hipsparseZcsrilu02().
*
*  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p m, \p nnz, \p descrA, \p csrSortedValA, 
*              \p csrSortedRowPtrA, \p csrSortedColIndA, \p info or \p pBufferSizeInBytes pointer 
*              is invalid.
*  \retval     HIPSPARSE_STATUS_INTERNAL_ERROR an internal error occurred.
*/
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseScsrilu02_bufferSizeExt(hipsparseHandle_t         handle,
                                                   int                       m,
                                                   int                       nnz,
                                                   const hipsparseMatDescr_t descrA,
                                                   float*                    csrSortedValA,
                                                   const int*                csrSortedRowPtrA,
                                                   const int*                csrSortedColIndA,
                                                   csrilu02Info_t            info,
                                                   size_t*                   pBufferSizeInBytes);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDcsrilu02_bufferSizeExt(hipsparseHandle_t         handle,
                                                   int                       m,
                                                   int                       nnz,
                                                   const hipsparseMatDescr_t descrA,
                                                   double*                   csrSortedValA,
                                                   const int*                csrSortedRowPtrA,
                                                   const int*                csrSortedColIndA,
                                                   csrilu02Info_t            info,
                                                   size_t*                   pBufferSizeInBytes);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCcsrilu02_bufferSizeExt(hipsparseHandle_t         handle,
                                                   int                       m,
                                                   int                       nnz,
                                                   const hipsparseMatDescr_t descrA,
                                                   hipComplex*               csrSortedValA,
                                                   const int*                csrSortedRowPtrA,
                                                   const int*                csrSortedColIndA,
                                                   csrilu02Info_t            info,
                                                   size_t*                   pBufferSizeInBytes);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZcsrilu02_bufferSizeExt(hipsparseHandle_t         handle,
                                                   int                       m,
                                                   int                       nnz,
                                                   const hipsparseMatDescr_t descrA,
                                                   hipDoubleComplex*         csrSortedValA,
                                                   const int*                csrSortedRowPtrA,
                                                   const int*                csrSortedColIndA,
                                                   csrilu02Info_t            info,
                                                   size_t*                   pBufferSizeInBytes);
/**@}*/

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
/*! \ingroup precond_module
*  \brief Incomplete LU factorization with 0 fill-ins and no pivoting using CSR
*  storage format
*
*  \details
*  \p hipsparseXcsrilu02_analysis performs the analysis step for \ref hipsparseScsrilu02 
*  "hipsparseXcsrilu02()".
*
*  \note
*  If the matrix sparsity pattern changes, the gathered information will become invalid.
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*
*  @param[in]
*  handle           handle to the hipsparse library context queue.
*  @param[in]
*  m                number of rows of the sparse CSR matrix.
*  @param[in]
*  nnz              number of non-zero entries of the sparse CSR matrix.
*  @param[in]
*  descrA           descriptor of the sparse CSR matrix.
*  @param[in]
*  csrSortedValA    array of \p nnz elements of the sparse CSR matrix.
*  @param[in]
*  csrSortedRowPtrA array of \p m+1 elements that point to the start of every row of the
*                   sparse CSR matrix.
*  @param[in]
*  csrSortedColIndA array of \p nnz elements containing the column indices of the sparse
*                   CSR matrix.
*  @param[out]
*  info             structure that holds the information collected during
*                   the analysis step.
*  @param[in]
*  policy           \ref HIPSPARSE_SOLVE_POLICY_NO_LEVEL or \ref HIPSPARSE_SOLVE_POLICY_USE_LEVEL.
*  @param[in]
*  pBuffer          temporary storage buffer allocated by the user.
*
*  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p m, \p nnz, \p descrA, \p csrSortedValA, 
*              \p csrSortedRowPtrA, \p csrSortedColIndA, \p info or \p pBuffer pointer is invalid.
*  \retval     HIPSPARSE_STATUS_INTERNAL_ERROR an internal error occurred.
*/
/**@{*/
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseScsrilu02_analysis(hipsparseHandle_t         handle,
                                              int                       m,
                                              int                       nnz,
                                              const hipsparseMatDescr_t descrA,
                                              const float*              csrSortedValA,
                                              const int*                csrSortedRowPtrA,
                                              const int*                csrSortedColIndA,
                                              csrilu02Info_t            info,
                                              hipsparseSolvePolicy_t    policy,
                                              void*                     pBuffer);
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDcsrilu02_analysis(hipsparseHandle_t         handle,
                                              int                       m,
                                              int                       nnz,
                                              const hipsparseMatDescr_t descrA,
                                              const double*             csrSortedValA,
                                              const int*                csrSortedRowPtrA,
                                              const int*                csrSortedColIndA,
                                              csrilu02Info_t            info,
                                              hipsparseSolvePolicy_t    policy,
                                              void*                     pBuffer);
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCcsrilu02_analysis(hipsparseHandle_t         handle,
                                              int                       m,
                                              int                       nnz,
                                              const hipsparseMatDescr_t descrA,
                                              const hipComplex*         csrSortedValA,
                                              const int*                csrSortedRowPtrA,
                                              const int*                csrSortedColIndA,
                                              csrilu02Info_t            info,
                                              hipsparseSolvePolicy_t    policy,
                                              void*                     pBuffer);
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZcsrilu02_analysis(hipsparseHandle_t         handle,
                                              int                       m,
                                              int                       nnz,
                                              const hipsparseMatDescr_t descrA,
                                              const hipDoubleComplex*   csrSortedValA,
                                              const int*                csrSortedRowPtrA,
                                              const int*                csrSortedColIndA,
                                              csrilu02Info_t            info,
                                              hipsparseSolvePolicy_t    policy,
                                              void*                     pBuffer);
/**@}*/
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
/*! \ingroup precond_module
*  \brief Incomplete LU factorization with 0 fill-ins and no pivoting using CSR
*  storage format
*
*  \details
*  \p hipsparseXcsrilu02 computes the incomplete LU factorization with 0 fill-ins and no
*  pivoting of a sparse \f$m \times m\f$ CSR matrix \f$A\f$, such that
*  \f[
*    A \approx LU
*  \f]
*
*  \p hipsparseXcsrilu02 requires a user allocated temporary buffer. Its size is returned
*  by \ref hipsparseScsrilu02_bufferSize "hipsparseXcsrilu02_bufferSize()" or 
*  \ref hipsparseScsrilu02_bufferSizeExt "hipsparseXcsrilu02_bufferSizeExt()". Furthermore,
*  analysis meta data is required. It can be obtained by \ref hipsparseScsrilu02_analysis 
*  "hipsparseXcsrilu02_analysis()". \p hipsparseXcsrilu02 reports the first zero pivot 
*  (either numerical or structural zero). The zero pivot status can be obtained by calling 
*  \ref hipsparseXcsrilu02_zeroPivot().
*
*  \note
*  The sparse CSR matrix has to be sorted. This can be achieved by calling
*  hipsparseXcsrsort().
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*
*  @param[in]
*  handle             handle to the hipsparse library context queue.
*  @param[in]
*  m                  number of rows of the sparse CSR matrix.
*  @param[in]
*  nnz                number of non-zero entries of the sparse CSR matrix.
*  @param[in]
*  descrA             descriptor of the sparse CSR matrix.
*  @param[inout]
*  csrSortedValA_valM array of \p nnz elements of the sparse CSR matrix.
*  @param[in]
*  csrSortedRowPtrA   array of \p m+1 elements that point to the start
*                     of every row of the sparse CSR matrix.
*  @param[in]
*  csrSortedColIndA   array of \p nnz elements containing the column indices of the sparse
*                     CSR matrix.
*  @param[in]
*  info               structure that holds the information collected during the analysis step.
*  @param[in]
*  policy             \ref HIPSPARSE_SOLVE_POLICY_NO_LEVEL or \ref HIPSPARSE_SOLVE_POLICY_USE_LEVEL.
*  @param[in]
*  pBuffer            temporary storage buffer allocated by the user.
*
*  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p m, \p nnz, \p descrA, \p csrSortedValA_valM, 
*              \p csrSortedRowPtrA or \p csrSortedColIndA pointer is invalid.
*  \retval     HIPSPARSE_STATUS_ARCH_MISMATCH the device is not supported.
*  \retval     HIPSPARSE_STATUS_INTERNAL_ERROR an internal error occurred.
*/
/**@{*/
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseScsrilu02(hipsparseHandle_t         handle,
                                     int                       m,
                                     int                       nnz,
                                     const hipsparseMatDescr_t descrA,
                                     float*                    csrSortedValA_valM,
                                     /* matrix A values are updated inplace
                                        to be the preconditioner M values */
                                     const int*             csrSortedRowPtrA,
                                     const int*             csrSortedColIndA,
                                     csrilu02Info_t         info,
                                     hipsparseSolvePolicy_t policy,
                                     void*                  pBuffer);
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDcsrilu02(hipsparseHandle_t         handle,
                                     int                       m,
                                     int                       nnz,
                                     const hipsparseMatDescr_t descrA,
                                     double*                   csrSortedValA_valM,
                                     /* matrix A values are updated inplace
                                        to be the preconditioner M values */
                                     const int*             csrSortedRowPtrA,
                                     const int*             csrSortedColIndA,
                                     csrilu02Info_t         info,
                                     hipsparseSolvePolicy_t policy,
                                     void*                  pBuffer);
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCcsrilu02(hipsparseHandle_t         handle,
                                     int                       m,
                                     int                       nnz,
                                     const hipsparseMatDescr_t descrA,
                                     hipComplex*               csrSortedValA_valM,
                                     /* matrix A values are updated inplace
                                        to be the preconditioner M values */
                                     const int*             csrSortedRowPtrA,
                                     const int*             csrSortedColIndA,
                                     csrilu02Info_t         info,
                                     hipsparseSolvePolicy_t policy,
                                     void*                  pBuffer);
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZcsrilu02(hipsparseHandle_t         handle,
                                     int                       m,
                                     int                       nnz,
                                     const hipsparseMatDescr_t descrA,
                                     hipDoubleComplex*         csrSortedValA_valM,
                                     /* matrix A values are updated inplace
                                        to be the preconditioner M values */
                                     const int*             csrSortedRowPtrA,
                                     const int*             csrSortedColIndA,
                                     csrilu02Info_t         info,
                                     hipsparseSolvePolicy_t policy,
                                     void*                  pBuffer);
/**@}*/
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
/*! \ingroup precond_module
 *  \brief Incomplete Cholesky factorization with 0 fill-ins and no pivoting using BSR
 *  storage format
 *
 *  \details
 *  \p hipsparseXbsric02_zeroPivot returns \ref HIPSPARSE_STATUS_ZERO_PIVOT, if either a
 *  structural or numerical zero has been found during \ref hipsparseSbsric02_analysis 
 *  "hipsparseXbsric02_analysis()" or \ref hipsparseSbsric02 "hipsparseXbsric02()" computation. 
 *  The first zero pivot \f$j\f$ at \f$A_{j,j}\f$ is stored in \p position, using same index 
 *  base as the BSR matrix.
 *
 *  \p position can be in host or device memory. If no zero pivot has been found,
 *  \p position is set to -1 and \ref HIPSPARSE_STATUS_SUCCESS is returned instead.
 *
 *  \note
 *  If a zero pivot is found, \p position=j means that either the diagonal block \p A(j,j)
 *  is missing (structural zero) or the diagonal block \p A(j,j) is not positive definite
 *  (numerical zero).
 *
 *  \note \p hipsparseXbsric02_zeroPivot is a blocking function. It might influence
 *  performance negatively.
 *
 *  @param[in]
 *  handle      handle to the hipsparse library context queue.
 *  @param[in]
 *  info        structure that holds the information collected during the analysis step.
 *  @param[inout]
 *  position    pointer to zero pivot \f$j\f$, can be in host or device memory.
 *
 *  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
 *  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p info or \p position pointer is
 *              invalid.
 *  \retval     HIPSPARSE_STATUS_INTERNAL_ERROR an internal error occurred.
 *  \retval     HIPSPARSE_STATUS_ZERO_PIVOT zero pivot has been found.
 */
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t
    hipsparseXbsric02_zeroPivot(hipsparseHandle_t handle, bsric02Info_t info, int* position);
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
/*! \ingroup precond_module
 *  \brief Incomplete Cholesky factorization with 0 fill-ins and no pivoting using BSR
 *  storage format
 *
 *  \details
 *  \p hipsparseXbsric02_bufferSize returns the size of the temporary storage buffer
 *  in bytes that is required by \ref hipsparseSbsric02_analysis "hipsparseXbsric02_analysis()" 
 *  and \ref hipsparseSbsric02 "hipsparseXbsric02()". The temporary storage buffer must be 
 *  allocated by the user.
 *
 *  @param[in]
 *  handle             handle to the hipsparse library context queue.
 *  @param[in]
 *  dirA               direction that specifies whether to count nonzero elements by \ref HIPSPARSE_DIRECTION_ROW 
 *                     or by \ref HIPSPARSE_DIRECTION_COLUMN.
 *  @param[in]
 *  mb                 number of block rows in the sparse BSR matrix.
 *  @param[in]
 *  nnzb               number of non-zero block entries of the sparse BSR matrix.
 *  @param[in]
 *  descrA             descriptor of the sparse BSR matrix.
 *  @param[in]
 *  bsrValA            array of length \p nnzb*blockDim*blockDim containing the values of the sparse BSR matrix.
 *  @param[in]
 *  bsrRowPtrA         array of \p mb+1 elements that point to the start of every block row of the
 *                     sparse BSR matrix.
 *  @param[in]
 *  bsrColIndA         array of \p nnzb elements containing the block column indices of the sparse BSR matrix.
 *  @param[in]
 *  blockDim           the block dimension of the BSR matrix. Between 1 and m where \p m=mb*blockDim.
 *  @param[out]
 *  info               structure that holds the information collected during the analysis step.
 *  @param[out]
 *  pBufferSizeInBytes number of bytes of the temporary storage buffer required by
 *                     hipsparseSbsric02_analysis(), hipsparseDbsric02_analysis(),
 *                     hipsparseCbsric02_analysis(), hipsparseZbsric02_analysis(),
 *                     hipsparseSbsric02(), hipsparseDbsric02(), hipsparseCbsric02()
 *                     and hipsparseZbsric02().
 *
 *  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
 *  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p mb, \p nnzb, \p blockDim, \p descrA, 
 *              \p bsrValA, \p bsrRowPtrA, \p bsrColIndA, \p info or \p pBufferSizeInBytes pointer 
 *              is invalid.
 *  \retval     HIPSPARSE_STATUS_INTERNAL_ERROR an internal error occurred.
 *  \retval     HIPSPARSE_STATUS_NOT_SUPPORTED
 *              \ref hipsparseMatrixType_t != \ref HIPSPARSE_MATRIX_TYPE_GENERAL.
 */
/**@{*/
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSbsric02_bufferSize(hipsparseHandle_t         handle,
                                               hipsparseDirection_t      dirA,
                                               int                       mb,
                                               int                       nnzb,
                                               const hipsparseMatDescr_t descrA,
                                               float*                    bsrValA,
                                               const int*                bsrRowPtrA,
                                               const int*                bsrColIndA,
                                               int                       blockDim,
                                               bsric02Info_t             info,
                                               int*                      pBufferSizeInBytes);
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDbsric02_bufferSize(hipsparseHandle_t         handle,
                                               hipsparseDirection_t      dirA,
                                               int                       mb,
                                               int                       nnzb,
                                               const hipsparseMatDescr_t descrA,
                                               double*                   bsrValA,
                                               const int*                bsrRowPtrA,
                                               const int*                bsrColIndA,
                                               int                       blockDim,
                                               bsric02Info_t             info,
                                               int*                      pBufferSizeInBytes);
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCbsric02_bufferSize(hipsparseHandle_t         handle,
                                               hipsparseDirection_t      dirA,
                                               int                       mb,
                                               int                       nnzb,
                                               const hipsparseMatDescr_t descrA,
                                               hipComplex*               bsrValA,
                                               const int*                bsrRowPtrA,
                                               const int*                bsrColIndA,
                                               int                       blockDim,
                                               bsric02Info_t             info,
                                               int*                      pBufferSizeInBytes);
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZbsric02_bufferSize(hipsparseHandle_t         handle,
                                               hipsparseDirection_t      dirA,
                                               int                       mb,
                                               int                       nnzb,
                                               const hipsparseMatDescr_t descrA,
                                               hipDoubleComplex*         bsrValA,
                                               const int*                bsrRowPtrA,
                                               const int*                bsrColIndA,
                                               int                       blockDim,
                                               bsric02Info_t             info,
                                               int*                      pBufferSizeInBytes);
/**@}*/
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
/*! \ingroup precond_module
 *  \brief Incomplete Cholesky factorization with 0 fill-ins and no pivoting using BSR
 *  storage format
 *
 *  \details
 *  \p hipsparseXbsric02_analysis performs the analysis step for \ref hipsparseSbsric02 
 *  "hipsparseXbsric02()".
 *
 *  \note
 *  If the matrix sparsity pattern changes, the gathered information will become invalid.
 *
 *  \note
 *  This function is non blocking and executed asynchronously with respect to the host.
 *  It may return before the actual computation has finished.
 *
 *  @param[in]
 *  handle      handle to the hipsparse library context queue.
 *  @param[in]
 *  dirA        direction that specified whether to count nonzero elements by \ref HIPSPARSE_DIRECTION_ROW or by
 *              \ref HIPSPARSE_DIRECTION_COLUMN.
 *  @param[in]
 *  mb          number of block rows in the sparse BSR matrix.
 *  @param[in]
 *  nnzb        number of non-zero block entries of the sparse BSR matrix.
 *  @param[in]
 *  descrA      descriptor of the sparse BSR matrix.
 *  @param[in]
 *  bsrValA     array of length \p nnzb*blockDim*blockDim containing the values of the sparse BSR matrix.
 *  @param[in]
 *  bsrRowPtrA  array of \p mb+1 elements that point to the start of every block row of the
 *              sparse BSR matrix.
 *  @param[in]
 *  bsrColIndA  array of \p nnzb elements containing the block column indices of the sparse BSR matrix.
 *  @param[in]
 *  blockDim    the block dimension of the BSR matrix. Between 1 and m where \p m=mb*blockDim.
 *  @param[out]
 *  info        structure that holds the information collected during the analysis step.
 *  @param[in]
 *  policy      \ref HIPSPARSE_SOLVE_POLICY_NO_LEVEL or \ref HIPSPARSE_SOLVE_POLICY_USE_LEVEL.
 *  @param[in]
 *  pBuffer     temporary storage buffer allocated by the user.
 *
 *  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
 *  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p mb, \p nnzb, \p blockDim, \p descrA, 
 *              \p bsrValA, \p bsrRowPtrA, \p bsrColIndA, \p info or \p pBuffer pointer is invalid.
 *  \retval     HIPSPARSE_STATUS_INTERNAL_ERROR an internal error occurred.
 *  \retval     HIPSPARSE_STATUS_NOT_SUPPORTED
 *              \ref hipsparseMatrixType_t != \ref HIPSPARSE_MATRIX_TYPE_GENERAL.
 */
/**@{*/
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSbsric02_analysis(hipsparseHandle_t         handle,
                                             hipsparseDirection_t      dirA,
                                             int                       mb,
                                             int                       nnzb,
                                             const hipsparseMatDescr_t descrA,
                                             const float*              bsrValA,
                                             const int*                bsrRowPtrA,
                                             const int*                bsrColIndA,
                                             int                       blockDim,
                                             bsric02Info_t             info,
                                             hipsparseSolvePolicy_t    policy,
                                             void*                     pBuffer);
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDbsric02_analysis(hipsparseHandle_t         handle,
                                             hipsparseDirection_t      dirA,
                                             int                       mb,
                                             int                       nnzb,
                                             const hipsparseMatDescr_t descrA,
                                             const double*             bsrValA,
                                             const int*                bsrRowPtrA,
                                             const int*                bsrColIndA,
                                             int                       blockDim,
                                             bsric02Info_t             info,
                                             hipsparseSolvePolicy_t    policy,
                                             void*                     pBuffer);
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCbsric02_analysis(hipsparseHandle_t         handle,
                                             hipsparseDirection_t      dirA,
                                             int                       mb,
                                             int                       nnzb,
                                             const hipsparseMatDescr_t descrA,
                                             const hipComplex*         bsrValA,
                                             const int*                bsrRowPtrA,
                                             const int*                bsrColIndA,
                                             int                       blockDim,
                                             bsric02Info_t             info,
                                             hipsparseSolvePolicy_t    policy,
                                             void*                     pBuffer);
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZbsric02_analysis(hipsparseHandle_t         handle,
                                             hipsparseDirection_t      dirA,
                                             int                       mb,
                                             int                       nnzb,
                                             const hipsparseMatDescr_t descrA,
                                             const hipDoubleComplex*   bsrValA,
                                             const int*                bsrRowPtrA,
                                             const int*                bsrColIndA,
                                             int                       blockDim,
                                             bsric02Info_t             info,
                                             hipsparseSolvePolicy_t    policy,
                                             void*                     pBuffer);
/**@}*/
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
/*! \ingroup precond_module
 *  \brief Incomplete Cholesky factorization with 0 fill-ins and no pivoting using BSR
 *  storage format
 *
 *  \details
 *  \p hipsparseXbsric02 computes the incomplete Cholesky factorization with 0 fill-ins
 *  and no pivoting of a sparse \f$mb \times mb\f$ BSR matrix \f$A\f$, such that
 *  \f[
 *    A \approx LL^T
 *  \f]
 *
 *  \p hipsparseXbsric02 requires a user allocated temporary buffer. Its size is returned
 *  by \ref hipsparseSbsric02_bufferSize "hipsparseXbsric02_bufferSize()". Furthermore, 
 *  analysis meta data is required. It can be obtained by \ref hipsparseSbsric02_analysis 
 *  "hipsparseXbsric02_analysis()". \p hipsparseXbsric02 reports the first zero pivot 
 *  (either numerical or structural zero). The zero pivot status can be obtained by calling 
 *  \ref hipsparseXbsric02_zeroPivot().
 *
 *  \note
 *  This function is non blocking and executed asynchronously with respect to the host.
 *  It may return before the actual computation has finished.
 *
 *  @param[in]
 *  handle      handle to the hipsparse library context queue.
 *  @param[in]
 *  dirA        direction that specified whether to count nonzero elements by \ref HIPSPARSE_DIRECTION_ROW or by
 *              \ref HIPSPARSE_DIRECTION_COLUMN.
 *  @param[in]
 *  mb          number of block rows in the sparse BSR matrix.
 *  @param[in]
 *  nnzb        number of non-zero block entries of the sparse BSR matrix.
 *  @param[in]
 *  descrA      descriptor of the sparse BSR matrix.
 *  @param[inout]
 *  bsrValA     array of length \p nnzb*blockDim*blockDim containing the values of the sparse BSR matrix.
 *  @param[in]
 *  bsrRowPtrA  array of \p mb+1 elements that point to the start of every block row of the
 *              sparse BSR matrix.
 *  @param[in]
 *  bsrColIndA  array of \p nnzb elements containing the block column indices of the sparse BSR matrix.
 *  @param[in]
 *  blockDim    the block dimension of the BSR matrix. Between 1 and m where \p m=mb*blockDim.
 *  @param[in]
 *  info        structure that holds the information collected during the analysis step.
 *  @param[in]
 *  policy      \ref HIPSPARSE_SOLVE_POLICY_NO_LEVEL or \ref HIPSPARSE_SOLVE_POLICY_USE_LEVEL.
 *  @param[in]
 *  pBuffer     temporary storage buffer allocated by the user.
 *
 *  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
 *  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p mb, \p nnzb, \p blockDim, \p descrA, 
 *              \p bsrValA, \p bsrRowPtrA, or \p bsrColIndA pointer is invalid.
 *  \retval     HIPSPARSE_STATUS_ARCH_MISMATCH the device is not supported.
 *  \retval     HIPSPARSE_STATUS_INTERNAL_ERROR an internal error occurred.
 *  \retval     HIPSPARSE_STATUS_NOT_SUPPORTED
 *              \ref hipsparseMatrixType_t != \ref HIPSPARSE_MATRIX_TYPE_GENERAL.
 */
/**@{*/
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSbsric02(hipsparseHandle_t         handle,
                                    hipsparseDirection_t      dirA,
                                    int                       mb,
                                    int                       nnzb,
                                    const hipsparseMatDescr_t descrA,
                                    float*                    bsrValA,
                                    const int*                bsrRowPtrA,
                                    const int*                bsrColIndA,
                                    int                       blockDim,
                                    bsric02Info_t             info,
                                    hipsparseSolvePolicy_t    policy,
                                    void*                     pBuffer);
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDbsric02(hipsparseHandle_t         handle,
                                    hipsparseDirection_t      dirA,
                                    int                       mb,
                                    int                       nnzb,
                                    const hipsparseMatDescr_t descrA,
                                    double*                   bsrValA,
                                    const int*                bsrRowPtrA,
                                    const int*                bsrColIndA,
                                    int                       blockDim,
                                    bsric02Info_t             info,
                                    hipsparseSolvePolicy_t    policy,
                                    void*                     pBuffer);
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCbsric02(hipsparseHandle_t         handle,
                                    hipsparseDirection_t      dirA,
                                    int                       mb,
                                    int                       nnzb,
                                    const hipsparseMatDescr_t descrA,
                                    hipComplex*               bsrValA,
                                    const int*                bsrRowPtrA,
                                    const int*                bsrColIndA,
                                    int                       blockDim,
                                    bsric02Info_t             info,
                                    hipsparseSolvePolicy_t    policy,
                                    void*                     pBuffer);
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZbsric02(hipsparseHandle_t         handle,
                                    hipsparseDirection_t      dirA,
                                    int                       mb,
                                    int                       nnzb,
                                    const hipsparseMatDescr_t descrA,
                                    hipDoubleComplex*         bsrValA,
                                    const int*                bsrRowPtrA,
                                    const int*                bsrColIndA,
                                    int                       blockDim,
                                    bsric02Info_t             info,
                                    hipsparseSolvePolicy_t    policy,
                                    void*                     pBuffer);
/**@}*/
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
/*! \ingroup precond_module
*  \brief Incomplete Cholesky factorization with 0 fill-ins and no pivoting using CSR
*  storage format
*
*  \details
*  \p hipsparseXcsric02_zeroPivot returns \ref HIPSPARSE_STATUS_ZERO_PIVOT, if either a
*  structural or numerical zero has been found during \ref hipsparseScsric02_analysis 
*  "hipsparseXcsric02_analysis()" or \ref hipsparseScsric02 "hipsparseXcsric02()" 
*  computation. The first zero pivot \f$j\f$ at \f$A_{j,j}\f$ is stored in \p position, 
*  using same index base as the CSR matrix.
*
*  \p position can be in host or device memory. If no zero pivot has been found,
*  \p position is set to -1 and \ref HIPSPARSE_STATUS_SUCCESS is returned instead.
*
*  \note \p hipsparseXcsric02_zeroPivot is a blocking function. It might influence
*  performance negatively.
*
*  @param[in]
*  handle      handle to the hipsparse library context queue.
*  @param[in]
*  info        structure that holds the information collected during the analysis step.
*  @param[inout]
*  position    pointer to zero pivot \f$j\f$, can be in host or device memory.
*
*  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p info or \p position pointer is
*              invalid.
*  \retval     HIPSPARSE_STATUS_INTERNAL_ERROR an internal error occurred.
*  \retval     HIPSPARSE_STATUS_ZERO_PIVOT zero pivot has been found.
*/
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t
    hipsparseXcsric02_zeroPivot(hipsparseHandle_t handle, csric02Info_t info, int* position);
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
/*! \ingroup precond_module
*  \brief Incomplete Cholesky factorization with 0 fill-ins and no pivoting using CSR
*  storage format
*
*  \details
*  \p hipsparseXcsric02_bufferSize returns the size of the temporary storage buffer in bytes
*  that is required by \ref hipsparseScsric02_analysis "hipsparseXcsric02_analysis()" and 
*  \ref hipsparseScsric02 "hipsparseXcsric02()".
*
*  @param[in]
*  handle             handle to the hipsparse library context queue.
*  @param[in]
*  m                  number of rows of the sparse CSR matrix.
*  @param[in]
*  nnz                number of non-zero entries of the sparse CSR matrix.
*  @param[in]
*  descrA             descriptor of the sparse CSR matrix.
*  @param[in]
*  csrSortedValA      array of \p nnz elements of the sparse CSR matrix.
*  @param[in]
*  csrSortedRowPtrA   array of \p m+1 elements that point to the start of every row of the
*                     sparse CSR matrix.
*  @param[in]
*  csrSortedColIndA   array of \p nnz elements containing the column indices of the sparse
*                     CSR matrix.
*  @param[out]
*  info               structure that holds the information collected during the analysis step.
*  @param[out]
*  pBufferSizeInBytes number of bytes of the temporary storage buffer required by
*                     hipsparseScsric02_analysis(), hipsparseDcsric02_analysis(),
*                     hipsparseCcsric02_analysis(), hipsparseZcsric02_analysis(),
*                     hipsparseScsric02(), hipsparseDcsric02(), hipsparseCcsric02()
*                     and hipsparseZcsric02().
*
*  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p m, \p nnz, \p descrA, \p csrSortedValA, 
*              \p csrSortedRowPtrA, \p csrSortedColIndA, \p info or \p pBufferSizeInBytes pointer is 
*              invalid.
*  \retval     HIPSPARSE_STATUS_INTERNAL_ERROR an internal error occurred.
*  \retval     HIPSPARSE_STATUS_NOT_SUPPORTED
*              \ref hipsparseMatrixType_t != \ref HIPSPARSE_MATRIX_TYPE_GENERAL.
*/
/**@{*/
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseScsric02_bufferSize(hipsparseHandle_t         handle,
                                               int                       m,
                                               int                       nnz,
                                               const hipsparseMatDescr_t descrA,
                                               float*                    csrSortedValA,
                                               const int*                csrSortedRowPtrA,
                                               const int*                csrSortedColIndA,
                                               csric02Info_t             info,
                                               int*                      pBufferSizeInBytes);
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDcsric02_bufferSize(hipsparseHandle_t         handle,
                                               int                       m,
                                               int                       nnz,
                                               const hipsparseMatDescr_t descrA,
                                               double*                   csrSortedValA,
                                               const int*                csrSortedRowPtrA,
                                               const int*                csrSortedColIndA,
                                               csric02Info_t             info,
                                               int*                      pBufferSizeInBytes);
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCcsric02_bufferSize(hipsparseHandle_t         handle,
                                               int                       m,
                                               int                       nnz,
                                               const hipsparseMatDescr_t descrA,
                                               hipComplex*               csrSortedValA,
                                               const int*                csrSortedRowPtrA,
                                               const int*                csrSortedColIndA,
                                               csric02Info_t             info,
                                               int*                      pBufferSizeInBytes);
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZcsric02_bufferSize(hipsparseHandle_t         handle,
                                               int                       m,
                                               int                       nnz,
                                               const hipsparseMatDescr_t descrA,
                                               hipDoubleComplex*         csrSortedValA,
                                               const int*                csrSortedRowPtrA,
                                               const int*                csrSortedColIndA,
                                               csric02Info_t             info,
                                               int*                      pBufferSizeInBytes);
/**@}*/
#endif

/*! \ingroup precond_module
*  \brief Incomplete Cholesky factorization with 0 fill-ins and no pivoting using CSR
*  storage format
*
*  \details
*  \p hipsparseXcsric02_bufferSizeExt returns the size of the temporary storage buffer
*  in bytes that is required by \ref hipsparseScsric02_analysis "hipsparseXcsric02_analysis()" 
*  and \ref hipsparseScsric02 "hipsparseXcsric02()".
*
*  @param[in]
*  handle             handle to the hipsparse library context queue.
*  @param[in]
*  m                  number of rows of the sparse CSR matrix.
*  @param[in]
*  nnz                number of non-zero entries of the sparse CSR matrix.
*  @param[in]
*  descrA             descriptor of the sparse CSR matrix.
*  @param[in]
*  csrSortedValA      array of \p nnz elements of the sparse CSR matrix.
*  @param[in]
*  csrSortedRowPtrA   array of \p m+1 elements that point to the start of every row of the
*                     sparse CSR matrix.
*  @param[in]
*  csrSortedColIndA   array of \p nnz elements containing the column indices of the sparse
*                     CSR matrix.
*  @param[out]
*  info               structure that holds the information collected during the analysis step.
*  @param[out]
*  pBufferSizeInBytes number of bytes of the temporary storage buffer required by
*                     hipsparseScsric02_analysis(), hipsparseDcsric02_analysis(),
*                     hipsparseCcsric02_analysis(), hipsparseZcsric02_analysis(),
*                     hipsparseScsric02(), hipsparseDcsric02(), hipsparseCcsric02()
*                     and hipsparseZcsric02().
*
*  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p m, \p nnz, \p descrA, \p csrSortedValA, 
*              \p csrSortedRowPtrA, \p csrSortedColIndA, \p info or \p pBufferSizeInBytes pointer is 
*              invalid.
*  \retval     HIPSPARSE_STATUS_INTERNAL_ERROR an internal error occurred.
*  \retval     HIPSPARSE_STATUS_NOT_SUPPORTED
*              \ref hipsparseMatrixType_t != \ref HIPSPARSE_MATRIX_TYPE_GENERAL.
*/
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseScsric02_bufferSizeExt(hipsparseHandle_t         handle,
                                                  int                       m,
                                                  int                       nnz,
                                                  const hipsparseMatDescr_t descrA,
                                                  float*                    csrSortedValA,
                                                  const int*                csrSortedRowPtrA,
                                                  const int*                csrSortedColIndA,
                                                  csric02Info_t             info,
                                                  size_t*                   pBufferSizeInBytes);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDcsric02_bufferSizeExt(hipsparseHandle_t         handle,
                                                  int                       m,
                                                  int                       nnz,
                                                  const hipsparseMatDescr_t descrA,
                                                  double*                   csrSortedValA,
                                                  const int*                csrSortedRowPtrA,
                                                  const int*                csrSortedColIndA,
                                                  csric02Info_t             info,
                                                  size_t*                   pBufferSizeInBytes);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCcsric02_bufferSizeExt(hipsparseHandle_t         handle,
                                                  int                       m,
                                                  int                       nnz,
                                                  const hipsparseMatDescr_t descrA,
                                                  hipComplex*               csrSortedValA,
                                                  const int*                csrSortedRowPtrA,
                                                  const int*                csrSortedColIndA,
                                                  csric02Info_t             info,
                                                  size_t*                   pBufferSizeInBytes);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZcsric02_bufferSizeExt(hipsparseHandle_t         handle,
                                                  int                       m,
                                                  int                       nnz,
                                                  const hipsparseMatDescr_t descrA,
                                                  hipDoubleComplex*         csrSortedValA,
                                                  const int*                csrSortedRowPtrA,
                                                  const int*                csrSortedColIndA,
                                                  csric02Info_t             info,
                                                  size_t*                   pBufferSizeInBytes);
/**@}*/

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
/*! \ingroup precond_module
*  \brief Incomplete Cholesky factorization with 0 fill-ins and no pivoting using CSR
*  storage format
*
*  \details
*  \p hipsparseXcsric02_analysis performs the analysis step for \ref hipsparseScsric02 
*  "hipsparseXcsric02()".
*
*  \note
*  If the matrix sparsity pattern changes, the gathered information will become invalid.
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*
*  @param[in]
*  handle           handle to the hipsparse library context queue.
*  @param[in]
*  m                number of rows of the sparse CSR matrix.
*  @param[in]
*  nnz              number of non-zero entries of the sparse CSR matrix.
*  @param[in]
*  descrA           descriptor of the sparse CSR matrix.
*  @param[in]
*  csrSortedValA    array of \p nnz elements of the sparse CSR matrix.
*  @param[in]
*  csrSortedRowPtrA array of \p m+1 elements that point to the start of every row of the
*                   sparse CSR matrix.
*  @param[in]
*  csrSortedColIndA array of \p nnz elements containing the column indices of the sparse
*                   CSR matrix.
*  @param[out]
*  info             structure that holds the information collected during
*                   the analysis step.
*  @param[in]
*  policy           \ref HIPSPARSE_SOLVE_POLICY_NO_LEVEL or \ref HIPSPARSE_SOLVE_POLICY_USE_LEVEL.
*  @param[in]
*  pBuffer          temporary storage buffer allocated by the user.
*
*  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p m, \p nnz, \p descrA, \p csrSortedValA, 
*              \p csrSortedRowPtrA, \p csrSortedColIndA, \p info or \p pBuffer pointer is invalid.
*  \retval     HIPSPARSE_STATUS_INTERNAL_ERROR an internal error occurred.
*  \retval     HIPSPARSE_STATUS_NOT_SUPPORTED
*              \ref hipsparseMatrixType_t != \ref HIPSPARSE_MATRIX_TYPE_GENERAL.
*/
/**@{*/
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseScsric02_analysis(hipsparseHandle_t         handle,
                                             int                       m,
                                             int                       nnz,
                                             const hipsparseMatDescr_t descrA,
                                             const float*              csrSortedValA,
                                             const int*                csrSortedRowPtrA,
                                             const int*                csrSortedColIndA,
                                             csric02Info_t             info,
                                             hipsparseSolvePolicy_t    policy,
                                             void*                     pBuffer);
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDcsric02_analysis(hipsparseHandle_t         handle,
                                             int                       m,
                                             int                       nnz,
                                             const hipsparseMatDescr_t descrA,
                                             const double*             csrSortedValA,
                                             const int*                csrSortedRowPtrA,
                                             const int*                csrSortedColIndA,
                                             csric02Info_t             info,
                                             hipsparseSolvePolicy_t    policy,
                                             void*                     pBuffer);
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCcsric02_analysis(hipsparseHandle_t         handle,
                                             int                       m,
                                             int                       nnz,
                                             const hipsparseMatDescr_t descrA,
                                             const hipComplex*         csrSortedValA,
                                             const int*                csrSortedRowPtrA,
                                             const int*                csrSortedColIndA,
                                             csric02Info_t             info,
                                             hipsparseSolvePolicy_t    policy,
                                             void*                     pBuffer);
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZcsric02_analysis(hipsparseHandle_t         handle,
                                             int                       m,
                                             int                       nnz,
                                             const hipsparseMatDescr_t descrA,
                                             const hipDoubleComplex*   csrSortedValA,
                                             const int*                csrSortedRowPtrA,
                                             const int*                csrSortedColIndA,
                                             csric02Info_t             info,
                                             hipsparseSolvePolicy_t    policy,
                                             void*                     pBuffer);
/**@}*/
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
/*! \ingroup precond_module
*  \brief Incomplete Cholesky factorization with 0 fill-ins and no pivoting using CSR
*  storage format
*
*  \details
*  \p hipsparseXcsric02 computes the incomplete Cholesky factorization with 0 fill-ins
*  and no pivoting of a sparse \f$m \times m\f$ CSR matrix \f$A\f$, such that
*  \f[
*    A \approx LL^T
*  \f]
*
*  \p hipsparseXcsric02 requires a user allocated temporary buffer. Its size is returned
*  by \ref hipsparseScsric02_bufferSize "hipsparseXcsric02_bufferSize()" or 
*  \ref hipsparseScsric02_bufferSizeExt "hipsparseXcsric02_bufferSizeExt()". Furthermore,
*  analysis meta data is required. It can be obtained by \ref hipsparseScsric02_analysis 
*  "hipsparseXcsric02_analysis()". \p hipsparseXcsric02 reports the first zero pivot 
*  (either numerical or structural zero). The zero pivot status can be obtained by calling 
*  \ref hipsparseXcsric02_zeroPivot().
*
*  \note
*  The sparse CSR matrix has to be sorted. This can be achieved by calling
*  hipsparseXcsrsort().
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*
*  @param[in]
*  handle             handle to the hipsparse library context queue.
*  @param[in]
*  m                  number of rows of the sparse CSR matrix.
*  @param[in]
*  nnz                number of non-zero entries of the sparse CSR matrix.
*  @param[in]
*  descrA             descriptor of the sparse CSR matrix.
*  @param[inout]
*  csrSortedValA_valM array of \p nnz elements of the sparse CSR matrix.
*  @param[in]
*  csrSortedRowPtrA   array of \p m+1 elements that point to the start
*                     of every row of the sparse CSR matrix.
*  @param[in]
*  csrSortedColIndA   array of \p nnz elements containing the column indices of the sparse
*                     CSR matrix.
*  @param[in]
*  info               structure that holds the information collected during the analysis step.
*  @param[in]
*  policy             \ref HIPSPARSE_SOLVE_POLICY_NO_LEVEL or \ref HIPSPARSE_SOLVE_POLICY_USE_LEVEL.
*  @param[in]
*  pBuffer            temporary storage buffer allocated by the user.
*
*  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p m, \p nnz, \p descrA, \p csrSortedValA_valM, 
*              \p csrSortedRowPtrA or \p csrSortedColIndA pointer is invalid.
*  \retval     HIPSPARSE_STATUS_ARCH_MISMATCH the device is not supported.
*  \retval     HIPSPARSE_STATUS_INTERNAL_ERROR an internal error occurred.
*  \retval     HIPSPARSE_STATUS_NOT_SUPPORTED
*              \ref hipsparseMatrixType_t != \ref HIPSPARSE_MATRIX_TYPE_GENERAL.
*/
/**@{*/
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseScsric02(hipsparseHandle_t         handle,
                                    int                       m,
                                    int                       nnz,
                                    const hipsparseMatDescr_t descrA,
                                    float*                    csrSortedValA_valM,
                                    /* matrix A values are updated inplace
                                        to be the preconditioner M values */
                                    const int*             csrSortedRowPtrA,
                                    const int*             csrSortedColIndA,
                                    csric02Info_t          info,
                                    hipsparseSolvePolicy_t policy,
                                    void*                  pBuffer);
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDcsric02(hipsparseHandle_t         handle,
                                    int                       m,
                                    int                       nnz,
                                    const hipsparseMatDescr_t descrA,
                                    double*                   csrSortedValA_valM,
                                    /* matrix A values are updated inplace
                                        to be the preconditioner M values */
                                    const int*             csrSortedRowPtrA,
                                    const int*             csrSortedColIndA,
                                    csric02Info_t          info,
                                    hipsparseSolvePolicy_t policy,
                                    void*                  pBuffer);
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCcsric02(hipsparseHandle_t         handle,
                                    int                       m,
                                    int                       nnz,
                                    const hipsparseMatDescr_t descrA,
                                    hipComplex*               csrSortedValA_valM,
                                    /* matrix A values are updated inplace
                                        to be the preconditioner M values */
                                    const int*             csrSortedRowPtrA,
                                    const int*             csrSortedColIndA,
                                    csric02Info_t          info,
                                    hipsparseSolvePolicy_t policy,
                                    void*                  pBuffer);
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZcsric02(hipsparseHandle_t         handle,
                                    int                       m,
                                    int                       nnz,
                                    const hipsparseMatDescr_t descrA,
                                    hipDoubleComplex*         csrSortedValA_valM,
                                    /* matrix A values are updated inplace
                                        to be the preconditioner M values */
                                    const int*             csrSortedRowPtrA,
                                    const int*             csrSortedColIndA,
                                    csric02Info_t          info,
                                    hipsparseSolvePolicy_t policy,
                                    void*                  pBuffer);
/**@}*/
#endif

/*! \ingroup precond_module
*  \brief Tridiagonal solver with pivoting
*
*  \details
*  \p hipsparseXgtsv2_bufferSize returns the size of the temporary storage buffer
*  in bytes that is required by \ref hipsparseSgtsv2 "hipsparseXgtsv2()". The 
*  temporary storage buffer must be allocated by the user.
*/
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSgtsv2_bufferSizeExt(hipsparseHandle_t handle,
                                                int               m,
                                                int               n,
                                                const float*      dl,
                                                const float*      d,
                                                const float*      du,
                                                const float*      B,
                                                int               ldb,
                                                size_t*           pBufferSizeInBytes);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDgtsv2_bufferSizeExt(hipsparseHandle_t handle,
                                                int               m,
                                                int               n,
                                                const double*     dl,
                                                const double*     d,
                                                const double*     du,
                                                const double*     B,
                                                int               db,
                                                size_t*           pBufferSizeInBytes);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCgtsv2_bufferSizeExt(hipsparseHandle_t handle,
                                                int               m,
                                                int               n,
                                                const hipComplex* dl,
                                                const hipComplex* d,
                                                const hipComplex* du,
                                                const hipComplex* B,
                                                int               ldb,
                                                size_t*           pBufferSizeInBytes);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZgtsv2_bufferSizeExt(hipsparseHandle_t       handle,
                                                int                     m,
                                                int                     n,
                                                const hipDoubleComplex* dl,
                                                const hipDoubleComplex* d,
                                                const hipDoubleComplex* du,
                                                const hipDoubleComplex* B,
                                                int                     ldb,
                                                size_t*                 pBufferSizeInBytes);
/**@}*/

/*! \ingroup precond_module
*  \brief Tridiagonal solver with pivoting
*
*  \details
*  \p hipsparseXgtsv2 solves a tridiagonal system for multiple right hand sides using pivoting.
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*/
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSgtsv2(hipsparseHandle_t handle,
                                  int               m,
                                  int               n,
                                  const float*      dl,
                                  const float*      d,
                                  const float*      du,
                                  float*            B,
                                  int               ldb,
                                  void*             pBuffer);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDgtsv2(hipsparseHandle_t handle,
                                  int               m,
                                  int               n,
                                  const double*     dl,
                                  const double*     d,
                                  const double*     du,
                                  double*           B,
                                  int               ldb,
                                  void*             pBuffer);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCgtsv2(hipsparseHandle_t handle,
                                  int               m,
                                  int               n,
                                  const hipComplex* dl,
                                  const hipComplex* d,
                                  const hipComplex* du,
                                  hipComplex*       B,
                                  int               ldb,
                                  void*             pBuffer);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZgtsv2(hipsparseHandle_t       handle,
                                  int                     m,
                                  int                     n,
                                  const hipDoubleComplex* dl,
                                  const hipDoubleComplex* d,
                                  const hipDoubleComplex* du,
                                  hipDoubleComplex*       B,
                                  int                     ldb,
                                  void*                   pBuffer);
/**@}*/

/*! \ingroup precond_module
*  \brief Tridiagonal solver (no pivoting)
*
*  \details
*  \p hipsparseXgtsv2_nopivot_bufferSizeExt returns the size of the temporary storage
*  buffer in bytes that is required by \ref hipsparseSgtsv2_nopivot "hipsparseXgtsv2_nopivot()". 
*  The temporary storage buffer must be allocated by the user.
*
*  @param[in]
*  handle             handle to the hipsparse library context queue.
*  @param[in]
*  m                  size of the tri-diagonal linear system (must be >= 2).
*  @param[in]
*  n                  number of columns in the dense matrix B.
*  @param[in]
*  dl                 lower diagonal of tri-diagonal system. First entry must be zero.
*  @param[in]
*  d                  main diagonal of tri-diagonal system.
*  @param[in]
*  du                 upper diagonal of tri-diagonal system. Last entry must be zero.
*  @param[in]
*  B                  Dense matrix of size ( \p ldb, \p n ).
*  @param[in]
*  ldb                Leading dimension of B. Must satisfy \p ldb >= max(1, m).
*  @param[out]
*  pBufferSizeInBytes number of bytes of the temporary storage buffer required by
*                     hipsparseSgtsv2_nopivot(), hipsparseDgtsv2_nopivot(), hipsparseCgtsv2_nopivot()
*                     and hipsparseZgtsv2_nopivot().
*
*  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p m, \p n, \p ldb, \p dl, \p d, \p du,
*              \p B or \p pBufferSizeInBytes pointer is invalid.
*  \retval     HIPSPARSE_STATUS_INTERNAL_ERROR an internal error occurred.
*/
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSgtsv2_nopivot_bufferSizeExt(hipsparseHandle_t handle,
                                                        int               m,
                                                        int               n,
                                                        const float*      dl,
                                                        const float*      d,
                                                        const float*      du,
                                                        const float*      B,
                                                        int               ldb,
                                                        size_t*           pBufferSizeInBytes);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDgtsv2_nopivot_bufferSizeExt(hipsparseHandle_t handle,
                                                        int               m,
                                                        int               n,
                                                        const double*     dl,
                                                        const double*     d,
                                                        const double*     du,
                                                        const double*     B,
                                                        int               ldb,
                                                        size_t*           pBufferSizeInBytes);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCgtsv2_nopivot_bufferSizeExt(hipsparseHandle_t handle,
                                                        int               m,
                                                        int               n,
                                                        const hipComplex* dl,
                                                        const hipComplex* d,
                                                        const hipComplex* du,
                                                        const hipComplex* B,
                                                        int               ldb,
                                                        size_t*           pBufferSizeInBytes);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZgtsv2_nopivot_bufferSizeExt(hipsparseHandle_t       handle,
                                                        int                     m,
                                                        int                     n,
                                                        const hipDoubleComplex* dl,
                                                        const hipDoubleComplex* d,
                                                        const hipDoubleComplex* du,
                                                        const hipDoubleComplex* B,
                                                        int                     ldb,
                                                        size_t*                 pBufferSizeInBytes);
/**@}*/

/*! \ingroup precond_module
*  \brief Tridiagonal solver (no pivoting)
*
*  \details
*  \p hipsparseXgtsv2_nopivot solves a tridiagonal linear system for multiple right-hand sides
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*
*  @param[in]
*  handle      handle to the hipsparse library context queue.
*  @param[in]
*  m           size of the tri-diagonal linear system (must be >= 2).
*  @param[in]
*  n           number of columns in the dense matrix B.
*  @param[in]
*  dl          lower diagonal of tri-diagonal system. First entry must be zero.
*  @param[in]
*  d           main diagonal of tri-diagonal system.
*  @param[in]
*  du          upper diagonal of tri-diagonal system. Last entry must be zero.
*  @param[inout]
*  B           Dense matrix of size ( \p ldb, \p n ).
*  @param[in]
*  ldb         Leading dimension of B. Must satisfy \p ldb >= max(1, m).
*  @param[in]
*  pBuffer     temporary storage buffer allocated by the user.
*
*  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p m, \p n, \p ldb, \p dl, \p d,
*              \p du, \p B or \p pBuffer pointer is invalid.
*  \retval     HIPSPARSE_STATUS_INTERNAL_ERROR an internal error occurred.
*/
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSgtsv2_nopivot(hipsparseHandle_t handle,
                                          int               m,
                                          int               n,
                                          const float*      dl,
                                          const float*      d,
                                          const float*      du,
                                          float*            B,
                                          int               ldb,
                                          void*             pBuffer);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDgtsv2_nopivot(hipsparseHandle_t handle,
                                          int               m,
                                          int               n,
                                          const double*     dl,
                                          const double*     d,
                                          const double*     du,
                                          double*           B,
                                          int               ldb,
                                          void*             pBuffer);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCgtsv2_nopivot(hipsparseHandle_t handle,
                                          int               m,
                                          int               n,
                                          const hipComplex* dl,
                                          const hipComplex* d,
                                          const hipComplex* du,
                                          hipComplex*       B,
                                          int               ldb,
                                          void*             pBuffer);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZgtsv2_nopivot(hipsparseHandle_t       handle,
                                          int                     m,
                                          int                     n,
                                          const hipDoubleComplex* dl,
                                          const hipDoubleComplex* d,
                                          const hipDoubleComplex* du,
                                          hipDoubleComplex*       B,
                                          int                     ldb,
                                          void*                   pBuffer);
/**@}*/

/*! \ingroup precond_module
*  \brief Strided Batch tridiagonal solver (no pivoting)
*
*  \details
*  \p hipsparseXgtsv2StridedBatch_bufferSizeExt returns the size of the temporary storage
*  buffer in bytes that is required by \ref hipsparseSgtsv2StridedBatch "hipsparseXgtsv2StridedBatch()". 
*  The temporary storage buffer must be allocated by the user.
*
*  @param[in]
*  handle             handle to the hipsparse library context queue.
*  @param[in]
*  m                  size of the tri-diagonal linear system.
*  @param[in]
*  dl                 lower diagonal of tri-diagonal system where the ith system lower diagonal starts at 
*                     \p dl+batchStride*i.
*  @param[in]
*  d                  main diagonal of tri-diagonal system where the ith system diagonal starts at 
*                     \p d+batchStride*i.
*  @param[in]
*  du                 upper diagonal of tri-diagonal system where the ith system upper diagonal starts at 
*                     \p du+batchStride*i.
*  @param[inout]
*  x                  Dense array of righthand-sides where the ith righthand-side starts at \p x+batchStride*i.
*  @param[in]
*  batchCount         The number of systems to solve.
*  @param[in]
*  batchStride        The number of elements that separate each system. Must satisfy \p batchStride >= m.
*  @param[out]
*  pBufferSizeInBytes number of bytes of the temporary storage buffer required by
*                     hipsparseSgtsv2StridedBatch(), hipsparseDgtsv2StridedBatch(), 
*                     hipsparseCgtsv2StridedBatch() and hipsparseZgtsv2StridedBatch().
*
*  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p m, \p batchCount, \p batchStride, \p dl, 
*              \p d, \p du, \p x or \p pBufferSizeInBytes pointer is invalid.
*  \retval     HIPSPARSE_STATUS_INTERNAL_ERROR an internal error occurred.
*/
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSgtsv2StridedBatch_bufferSizeExt(hipsparseHandle_t handle,
                                                            int               m,
                                                            const float*      dl,
                                                            const float*      d,
                                                            const float*      du,
                                                            const float*      x,
                                                            int               batchCount,
                                                            int               batchStride,
                                                            size_t*           pBufferSizeInBytes);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDgtsv2StridedBatch_bufferSizeExt(hipsparseHandle_t handle,
                                                            int               m,
                                                            const double*     dl,
                                                            const double*     d,
                                                            const double*     du,
                                                            const double*     x,
                                                            int               batchCount,
                                                            int               batchStride,
                                                            size_t*           pBufferSizeInBytes);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCgtsv2StridedBatch_bufferSizeExt(hipsparseHandle_t handle,
                                                            int               m,
                                                            const hipComplex* dl,
                                                            const hipComplex* d,
                                                            const hipComplex* du,
                                                            const hipComplex* x,
                                                            int               batchCount,
                                                            int               batchStride,
                                                            size_t*           pBufferSizeInBytes);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZgtsv2StridedBatch_bufferSizeExt(hipsparseHandle_t       handle,
                                                            int                     m,
                                                            const hipDoubleComplex* dl,
                                                            const hipDoubleComplex* d,
                                                            const hipDoubleComplex* du,
                                                            const hipDoubleComplex* x,
                                                            int                     batchCount,
                                                            int                     batchStride,
                                                            size_t* pBufferSizeInBytes);
/**@}*/

/*! \ingroup precond_module
*  \brief Strided Batch tridiagonal solver (no pivoting)
*
*  \details
*  \p hipsparseXgtsv2StridedBatch solves a batched tridiagonal linear system
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*
*  @param[in]
*  handle      handle to the hipsparse library context queue.
*  @param[in]
*  m           size of the tri-diagonal linear system (must be >= 2).
*  @param[in]
*  dl          lower diagonal of tri-diagonal system. First entry must be zero.
*  @param[in]
*  d           main diagonal of tri-diagonal system.
*  @param[in]
*  du          upper diagonal of tri-diagonal system. Last entry must be zero.
*  @param[inout]
*  x           Dense array of righthand-sides where the ith righthand-side starts at \p x+batchStride*i.
*  @param[in]
*  batchCount  The number of systems to solve.
*  @param[in]
*  batchStride The number of elements that separate each system. Must satisfy \p batchStride >= m.
*  @param[in]
*  pBuffer     temporary storage buffer allocated by the user.
*
*  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p m, \p batchCount, \p batchStride, \p dl, \p d,
*              \p du, \p x or \p pBuffer pointer is invalid.
*  \retval     HIPSPARSE_STATUS_INTERNAL_ERROR an internal error occurred.
*/
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSgtsv2StridedBatch(hipsparseHandle_t handle,
                                              int               m,
                                              const float*      dl,
                                              const float*      d,
                                              const float*      du,
                                              float*            x,
                                              int               batchCount,
                                              int               batchStride,
                                              void*             pBuffer);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDgtsv2StridedBatch(hipsparseHandle_t handle,
                                              int               m,
                                              const double*     dl,
                                              const double*     d,
                                              const double*     du,
                                              double*           x,
                                              int               batchCount,
                                              int               batchStride,
                                              void*             pBuffer);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCgtsv2StridedBatch(hipsparseHandle_t handle,
                                              int               m,
                                              const hipComplex* dl,
                                              const hipComplex* d,
                                              const hipComplex* du,
                                              hipComplex*       x,
                                              int               batchCount,
                                              int               batchStride,
                                              void*             pBuffer);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZgtsv2StridedBatch(hipsparseHandle_t       handle,
                                              int                     m,
                                              const hipDoubleComplex* dl,
                                              const hipDoubleComplex* d,
                                              const hipDoubleComplex* du,
                                              hipDoubleComplex*       x,
                                              int                     batchCount,
                                              int                     batchStride,
                                              void*                   pBuffer);
/**@}*/

/*! \ingroup precond_module
*  \brief Interleaved Batch tridiagonal solver
*
*  \details
*  \p hipsparseXgtsvInterleavedBatch_bufferSizeExt returns the size of the temporary storage
*  buffer in bytes that is required by \ref hipsparseSgtsvInterleavedBatch "hipsparseXgtsvInterleavedBatch()". 
*  The temporary storage buffer must be allocated by the user.
*
*  @param[in]
*  handle             handle to the hipsparse library context queue.
*  @param[in]
*  algo               Algorithm to use when solving tridiagonal systems. Options are thomas ( \p algo=0 ),
*                     LU ( \p algo=1 ), or QR ( \p algo=2 ). Thomas algorithm is the fastest but is not
*                     stable while LU and QR are slower but are stable.
*  @param[in]
*  m                  size of the tri-diagonal linear system.
*  @param[in]
*  dl                 lower diagonal of tri-diagonal system. The first element of the lower diagonal must be zero.
*  @param[in]
*  d                  main diagonal of tri-diagonal system.
*  @param[in]
*  du                 upper diagonal of tri-diagonal system. The last element of the upper diagonal must be zero.
*  @param[inout]
*  x                  Dense array of righthand-sides with dimension \p batchCount by \p m.
*  @param[in]
*  batchCount         The number of systems to solve.
*  @param[out]
*  pBufferSizeInBytes number of bytes of the temporary storage buffer required by
*                     hipsparseSgtsvInterleavedBatch(), hipsparseDgtsvInterleavedBatch(), hipsparseCgtsvInterleavedBatch()
*                     and hipsparseZgtsvInterleavedBatch().
*
*  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p m, \p batchCount, \p dl, \p d, \p du,
*              \p x or \p pBufferSizeInBytes pointer is invalid.
*  \retval     HIPSPARSE_STATUS_INTERNAL_ERROR an internal error occurred.
*/
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSgtsvInterleavedBatch_bufferSizeExt(hipsparseHandle_t handle,
                                                               int               algo,
                                                               int               m,
                                                               const float*      dl,
                                                               const float*      d,
                                                               const float*      du,
                                                               const float*      x,
                                                               int               batchCount,
                                                               size_t* pBufferSizeInBytes);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDgtsvInterleavedBatch_bufferSizeExt(hipsparseHandle_t handle,
                                                               int               algo,
                                                               int               m,
                                                               const double*     dl,
                                                               const double*     d,
                                                               const double*     du,
                                                               const double*     x,
                                                               int               batchCount,
                                                               size_t* pBufferSizeInBytes);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCgtsvInterleavedBatch_bufferSizeExt(hipsparseHandle_t handle,
                                                               int               algo,
                                                               int               m,
                                                               const hipComplex* dl,
                                                               const hipComplex* d,
                                                               const hipComplex* du,
                                                               const hipComplex* x,
                                                               int               batchCount,
                                                               size_t* pBufferSizeInBytes);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZgtsvInterleavedBatch_bufferSizeExt(hipsparseHandle_t       handle,
                                                               int                     algo,
                                                               int                     m,
                                                               const hipDoubleComplex* dl,
                                                               const hipDoubleComplex* d,
                                                               const hipDoubleComplex* du,
                                                               const hipDoubleComplex* x,
                                                               int                     batchCount,
                                                               size_t* pBufferSizeInBytes);
/**@}*/

/*! \ingroup precond_module
*  \brief Interleaved Batch tridiagonal solver
*
*  \details
*  \p hipsparseXgtsvInterleavedBatch solves a batched tridiagonal linear system
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*
*  @param[in]
*  handle      handle to the hipsparse library context queue.
*  @param[in]
*  algo        Algorithm to use when solving tridiagonal systems. Options are thomas ( \p algo=0 ),
*              LU ( \p algo=1 ), or QR ( \p algo=2 ). Thomas algorithm is the fastest but is not
*              stable while LU and QR are slower but are stable.
*  @param[in]
*  m           size of the tri-diagonal linear system.
*  @param[inout]
*  dl          lower diagonal of tri-diagonal system. The first element of the lower diagonal must be zero.
*  @param[inout]
*  d           main diagonal of tri-diagonal system.
*  @param[inout]
*  du          upper diagonal of tri-diagonal system. The last element of the upper diagonal must be zero.
*  @param[inout]
*  x           Dense array of righthand-sides with dimension \p batchCount by \p m.
*  @param[in]
*  batchCount  The number of systems to solve.
*  @param[in]
*  pBuffer     temporary storage buffer allocated by the user.
*
*  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p m, \p batchCount, \p dl, \p d,
*              \p du, \p x or \p pBuffer pointer is invalid.
*  \retval     HIPSPARSE_STATUS_INTERNAL_ERROR an internal error occurred.
*/
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSgtsvInterleavedBatch(hipsparseHandle_t handle,
                                                 int               algo,
                                                 int               m,
                                                 float*            dl,
                                                 float*            d,
                                                 float*            du,
                                                 float*            x,
                                                 int               batchCount,
                                                 void*             pBuffer);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDgtsvInterleavedBatch(hipsparseHandle_t handle,
                                                 int               algo,
                                                 int               m,
                                                 double*           dl,
                                                 double*           d,
                                                 double*           du,
                                                 double*           x,
                                                 int               batchCount,
                                                 void*             pBuffer);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCgtsvInterleavedBatch(hipsparseHandle_t handle,
                                                 int               algo,
                                                 int               m,
                                                 hipComplex*       dl,
                                                 hipComplex*       d,
                                                 hipComplex*       du,
                                                 hipComplex*       x,
                                                 int               batchCount,
                                                 void*             pBuffer);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZgtsvInterleavedBatch(hipsparseHandle_t handle,
                                                 int               algo,
                                                 int               m,
                                                 hipDoubleComplex* dl,
                                                 hipDoubleComplex* d,
                                                 hipDoubleComplex* du,
                                                 hipDoubleComplex* x,
                                                 int               batchCount,
                                                 void*             pBuffer);
/**@}*/

/*! \ingroup precond_module
*  \brief Interleaved Batch pentadiagonal solver
*
*  \details
*  \p hipsparseXgpsvInterleavedBatch_bufferSizeExt returns the size of the temporary storage
*  buffer in bytes that is required by \ref hipsparseSgpsvInterleavedBatch "hipsparseXgpsvInterleavedBatch()". 
*  The temporary storage buffer must be allocated by the user.
*
*  @param[in]
*  handle             handle to the hipsparse library context queue.
*  @param[in]
*  algo               algorithm to solve the linear system.
*  @param[in]
*  m                  size of the pentadiagonal linear system.
*  @param[in]
*  ds                 lower diagonal (distance 2) of pentadiagonal system. First two entries
*                     must be zero.
*  @param[in]
*  dl                 lower diagonal of pentadiagonal system. First entry must be zero.
*  @param[in]
*  d                  main diagonal of pentadiagonal system.
*  @param[in]
*  du                 upper diagonal of pentadiagonal system. Last entry must be zero.
*  @param[in]
*  dw                 upper diagonal (distance 2) of pentadiagonal system. Last two entries
*                     must be zero.
*  @param[in]
*  x                  Dense array of right-hand-sides with dimension \p batchCount by \p m.
*  @param[in]
*  batchCount         The number of systems to solve.
*  @param[out]
*  pBufferSizeInBytes Number of bytes of the temporary storage buffer required.
*
*  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p m, \p alg, \p batchCount, \p ds, \p dl, 
*              \p d, \p du, \p dw, \p x or \p pBufferSizeInBytes pointer is invalid.
*  \retval     HIPSPARSE_STATUS_INTERNAL_ERROR an internal error occurred.
*/
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSgpsvInterleavedBatch_bufferSizeExt(hipsparseHandle_t handle,
                                                               int               algo,
                                                               int               m,
                                                               const float*      ds,
                                                               const float*      dl,
                                                               const float*      d,
                                                               const float*      du,
                                                               const float*      dw,
                                                               const float*      x,
                                                               int               batchCount,
                                                               size_t* pBufferSizeInBytes);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDgpsvInterleavedBatch_bufferSizeExt(hipsparseHandle_t handle,
                                                               int               algo,
                                                               int               m,
                                                               const double*     ds,
                                                               const double*     dl,
                                                               const double*     d,
                                                               const double*     du,
                                                               const double*     dw,
                                                               const double*     x,
                                                               int               batchCount,
                                                               size_t* pBufferSizeInBytes);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCgpsvInterleavedBatch_bufferSizeExt(hipsparseHandle_t handle,
                                                               int               algo,
                                                               int               m,
                                                               const hipComplex* ds,
                                                               const hipComplex* dl,
                                                               const hipComplex* d,
                                                               const hipComplex* du,
                                                               const hipComplex* dw,
                                                               const hipComplex* x,
                                                               int               batchCount,
                                                               size_t* pBufferSizeInBytes);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZgpsvInterleavedBatch_bufferSizeExt(hipsparseHandle_t       handle,
                                                               int                     algo,
                                                               int                     m,
                                                               const hipDoubleComplex* ds,
                                                               const hipDoubleComplex* dl,
                                                               const hipDoubleComplex* d,
                                                               const hipDoubleComplex* du,
                                                               const hipDoubleComplex* dw,
                                                               const hipDoubleComplex* x,
                                                               int                     batchCount,
                                                               size_t* pBufferSizeInBytes);

/**@}*/

/*! \ingroup precond_module
*  \brief Interleaved Batch pentadiagonal solver
*
*  \details
*  \p hipsparseXgpsvInterleavedBatch solves a batched pentadiagonal linear system
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*
*  @param[in]
*  handle      handle to the hipsparse library context queue.
*  @param[in]
*  algo        algorithm to solve the linear system.
*  @param[in]
*  m           size of the pentadiagonal linear system.
*  @param[inout]
*  ds          lower diagonal (distance 2) of pentadiagonal system. First two entries
*              must be zero.
*  @param[inout]
*  dl          lower diagonal of pentadiagonal system. First entry must be zero.
*  @param[inout]
*  d           main diagonal of pentadiagonal system.
*  @param[inout]
*  du          upper diagonal of pentadiagonal system. Last entry must be zero.
*  @param[inout]
*  dw          upper diagonal (distance 2) of pentadiagonal system. Last two entries
*              must be zero.
*  @param[inout]
*  x           Dense array of right-hand-sides with dimension \p batchCount by \p m.
*  @param[in]
*  batchCount  The number of systems to solve.
*  @param[in]
*  pBuffer     Temporary storage buffer allocated by the user.
*
*  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p m, \p alg, \p batchCount, \p ds, 
*              \p dl, \p d, \p du, \p dw, \p x or \p pBuffer pointer is invalid.
*  \retval     HIPSPARSE_STATUS_INTERNAL_ERROR an internal error occurred.
*/
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSgpsvInterleavedBatch(hipsparseHandle_t handle,
                                                 int               algo,
                                                 int               m,
                                                 float*            ds,
                                                 float*            dl,
                                                 float*            d,
                                                 float*            du,
                                                 float*            dw,
                                                 float*            x,
                                                 int               batchCount,
                                                 void*             pBuffer);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDgpsvInterleavedBatch(hipsparseHandle_t handle,
                                                 int               algo,
                                                 int               m,
                                                 double*           ds,
                                                 double*           dl,
                                                 double*           d,
                                                 double*           du,
                                                 double*           dw,
                                                 double*           x,
                                                 int               batchCount,
                                                 void*             pBuffer);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCgpsvInterleavedBatch(hipsparseHandle_t handle,
                                                 int               algo,
                                                 int               m,
                                                 hipComplex*       ds,
                                                 hipComplex*       dl,
                                                 hipComplex*       d,
                                                 hipComplex*       du,
                                                 hipComplex*       dw,
                                                 hipComplex*       x,
                                                 int               batchCount,
                                                 void*             pBuffer);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZgpsvInterleavedBatch(hipsparseHandle_t handle,
                                                 int               algo,
                                                 int               m,
                                                 hipDoubleComplex* ds,
                                                 hipDoubleComplex* dl,
                                                 hipDoubleComplex* d,
                                                 hipDoubleComplex* du,
                                                 hipDoubleComplex* dw,
                                                 hipDoubleComplex* x,
                                                 int               batchCount,
                                                 void*             pBuffer);

/**@}*/

/*
* ===========================================================================
*    Sparse Format Conversions
* ===========================================================================
*/

/*! \ingroup conv_module
*  \brief
*  \p hipsparseXnnz computes the number of nonzero elements per row or column and the total
*  number of nonzero elements in a dense matrix.
*
*  \details
*  For example, given the dense matrix:
*  \f[
*    \begin{bmatrix}
*    1 & 0 & 0 & 2 \\
*    3 & 4 & 0 & 0 \\
*    5 & 0 & 6 & 7
*    \end{bmatrix}
*  \f]
*
*  then using \p dirA == \ref HIPSPARSE_DIRECTION_ROW results in: 
*  \f[
*    \begin{align}
*    \text{nnzPerRowColumn} &= \begin{bmatrix} 2 & 2 & 3 \end{bmatrix} \\
*    \text{nnzTotalDevHostPtr} &= 7
*    \end{align}
*  \f]
*
*  while using \p dirA == \ref HIPSPARSE_DIRECTION_COLUMN results in:
*  \f[
*    \begin{align}
*    \text{nnzPerRowColumn} &= \begin{bmatrix} 3 & 1 & 1 & 2 \end{bmatrix} \\ 
*    \text{nnzTotalDevHostPtr} &= 7
*    \end{align}
*  \f]
*
*  The array \p nnzPerRowColumn must be allocated by the user before calling \p hipsparseXnnz and
*  has length equal to \p m if \p dirA == \ref HIPSPARSE_DIRECTION_ROW or \p n if 
*  \p dirA == \ref HIPSPARSE_DIRECTION_COLUMN.
*
*  For a complete code example on its usage, see the example found with hipsparseSdense2csr().
* 
*  \note
*  As indicated, \p nnzTotalDevHostPtr can point either to host or device memory. This is controlled 
*  by setting the pointer mode. See \ref hipsparseSetPointerMode().
*
*  \note
*  The routine does support asynchronous execution if the pointer mode is set to device.
*
*  @param[in]
*  handle             handle to the rocsparse library context queue.
*  @param[in]
*  dirA               direction that specified whether to count nonzero elements by \ref HIPSPARSE_DIRECTION_ROW 
*                     or by \ref HIPSPARSE_DIRECTION_COLUMN.
*  @param[in]
*  m                  number of rows of the dense matrix \p A.
*  @param[in]
*  n                  number of columns of the dense matrix \p A.
*  @param[in]
*  descrA             the descriptor of the dense matrix \p A.
*  @param[in]
*  A                  array of dimensions (\p lda, \p n)
*  @param[in]
*  lda                leading dimension of dense array \p A.
*  @param[out]
*  nnzPerRowColumn    array of size \p m or \p n containing the number of nonzero elements per row or column, respectively.
*  @param[out]
*  nnzTotalDevHostPtr total number of nonzero elements in device or host memory.
*
*  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p m, \p n, \p lda, \p A or \p nnzPerRowColumn or 
*              \p nnzTotalDevHostPtr pointer is invalid.
*/
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSnnz(hipsparseHandle_t         handle,
                                hipsparseDirection_t      dirA,
                                int                       m,
                                int                       n,
                                const hipsparseMatDescr_t descrA,
                                const float*              A,
                                int                       lda,
                                int*                      nnzPerRowColumn,
                                int*                      nnzTotalDevHostPtr);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDnnz(hipsparseHandle_t         handle,
                                hipsparseDirection_t      dirA,
                                int                       m,
                                int                       n,
                                const hipsparseMatDescr_t descrA,
                                const double*             A,
                                int                       lda,
                                int*                      nnzPerRowColumn,
                                int*                      nnzTotalDevHostPtr);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCnnz(hipsparseHandle_t         handle,
                                hipsparseDirection_t      dirA,
                                int                       m,
                                int                       n,
                                const hipsparseMatDescr_t descrA,
                                const hipComplex*         A,
                                int                       lda,
                                int*                      nnzPerRowColumn,
                                int*                      nnzTotalDevHostPtr);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZnnz(hipsparseHandle_t         handle,
                                hipsparseDirection_t      dirA,
                                int                       m,
                                int                       n,
                                const hipsparseMatDescr_t descrA,
                                const hipDoubleComplex*   A,
                                int                       lda,
                                int*                      nnzPerRowColumn,
                                int*                      nnzTotalDevHostPtr);
/**@}*/

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 12000)
/*! \ingroup conv_module
*  \brief
*  \p hipsparseXdense2csr converts the matrix A in dense format into a sparse matrix in CSR format.
*
*  \details
*  Given a dense, column ordered, matrix \p A with leading dimension \p ld where \p ld>=m, 
*  \p hipsparseXdense2csr converts the matrix to a sparse CSR format matrix. All the parameters 
*  are assumed to have been pre-allocated by the user and the arrays are filled in based on number 
*  of nonzeros per row, which can be pre-computed with \ref hipsparseSnnz "hipsparseXnnz()". The 
*  desired index base in the output CSR matrix is set in the \ref hipsparseMatDescr_t. See 
*  \ref hipsparseSetMatIndexBase(). 
*
*  As an example, if using index base zero (i.e. the default) and the dense 
*  matrix:
*
*  \f[
*    \begin{bmatrix}
*    1 & 0 & 0 & 2 \\
*    3 & 4 & 0 & 0 \\
*    5 & 0 & 6 & 7
*    \end{bmatrix}
*  \f]
*
*  The conversion results in the CSR arrays:
*
*  \f[
*    \begin{align}
*    \text{csrRowPtr} &= \begin{bmatrix} 0 & 2 & 4 & 7 \end{bmatrix} \\
*    \text{csrColInd} &= \begin{bmatrix} 0 & 3 & 0 & 1 & 0 & 2 & 3 \end{bmatrix} \\
*    \text{csrVal} &= \begin{bmatrix} 1 & 2 & 3 & 4 & 5 & 6 & 7 \end{bmatrix} \\
*    \end{align}
*  \f]
*
*  \note
*  It is executed asynchronously with respect to the host and may return control to the
*  application on the host before the entire result is ready.
*
*  @param[in]
*  handle       handle to the hipsparse library context queue.
*  @param[in]
*  m            number of rows of the dense matrix \p A.
*  @param[in]
*  n            number of columns of the dense matrix \p A.
*  @param[in]
*  descr        the descriptor of the dense matrix \p A, the supported matrix type is  \ref HIPSPARSE_MATRIX_TYPE_GENERAL and also 
*               any valid value of the \ref hipsparseIndexBase_t.
*  @param[in]
*  A            array of dimensions (\p ld, \p n)
*  @param[in]
*  ld           leading dimension of dense array \p A.
*  @param[in]
*  nnzPerRow    array of size \p n containing the number of non-zero elements per row.
*  @param[out]
*  csrVal       array of nnz ( = \p csrRowPtr[m] - \p csrRowPtr[0] ) nonzero elements of matrix \p A.
*  @param[out]
*  csrRowPtr    integer array of \p m+1 elements that contains the start of every row and the end of the last row plus one.
*  @param[out]
*  csrColInd    integer array of nnz ( = \p csrRowPtr[m] - \p csrRowPtr[0] ) column indices of the non-zero elements of matrix \p A.
*
*  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p m, \p n, \p ld, \p A, \p nnzPerRow, \p csrVal \p csrRowPtr or \p csrColInd
*              pointer is invalid.
*
*  \par Example
*  \code{.c}
*    // hipSPARSE handle
*    hipsparseHandle_t handle;
*    hipsparseCreate(&handle);
*
*    // Matrix descriptor
*    hipsparseMatDescr_t descr;
*    hipsparseCreateMatDescr(&descr);
*
*    // Dense matrix in column order
*    //     1 2 0 3 0
*    // A = 0 4 5 0 0
*    //     6 0 0 7 8
*    float hdense_A[15] = {1.0f, 0.0f, 6.0f, 2.0f, 4.0f, 0.0f, 0.0f, 5.0f, 0.0f, 3.0f, 0.0f, 7.0f, 0.0f, 0.0f, 8.0f};
*
*    int m         = 3;
*    int n         = 5;
*    hipsparseDirection_t dir = HIPSPARSE_DIRECTION_ROW;
*
*    float* ddense_A = nullptr;
*    hipMalloc((void**)&ddense_A, sizeof(float) * m * n);
*    hipMemcpy(ddense_A, hdense_A, sizeof(float) * m * n, hipMemcpyHostToDevice);
*
*    // Allocate memory for the nnz_per_row_columns array
*    int* dnnz_per_row;
*    hipMalloc((void**)&dnnz_per_row, sizeof(int) * m);
*
*    int nnz_A;
*    hipsparseSnnz(handle, dir, m, n, descr, ddense_A, m, dnnz_per_row, &nnz_A);
*
*    // Allocate sparse CSR matrix
*    int* dcsrRowPtr = nullptr;
*    int* dcsrColInd = nullptr;
*    float* dcsrVal = nullptr;
*    hipMalloc((void**)&dcsrRowPtr, sizeof(int) * (m + 1));
*    hipMalloc((void**)&dcsrColInd, sizeof(int) * nnz_A);
*    hipMalloc((void**)&dcsrVal, sizeof(float) * nnz_A);
*
*    hipsparseSdense2csr(handle, m, n, descr, ddense_A, m, dnnz_per_row, dcsrVal, dcsrRowPtr, dcsrColInd);
*
*    hipFree(dcsrRowPtr);
*    hipFree(dcsrColInd);
*    hipFree(dcsrVal);
*    hipFree(dnnz_per_row);
*    hipFree(ddense_A);
*
*    hipsparseDestroyMatDescr(descr);
*    hipsparseDestroy(handle);
*  \endcode
*/
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSdense2csr(hipsparseHandle_t         handle,
                                      int                       m,
                                      int                       n,
                                      const hipsparseMatDescr_t descr,
                                      const float*              A,
                                      int                       ld,
                                      const int*                nnzPerRow,
                                      float*                    csrVal,
                                      int*                      csrRowPtr,
                                      int*                      csrColInd);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDdense2csr(hipsparseHandle_t         handle,
                                      int                       m,
                                      int                       n,
                                      const hipsparseMatDescr_t descr,
                                      const double*             A,
                                      int                       ld,
                                      const int*                nnzPerRow,
                                      double*                   csrVal,
                                      int*                      csrRowPtr,
                                      int*                      csrColInd);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCdense2csr(hipsparseHandle_t         handle,
                                      int                       m,
                                      int                       n,
                                      const hipsparseMatDescr_t descr,
                                      const hipComplex*         A,
                                      int                       ld,
                                      const int*                nnzPerRow,
                                      hipComplex*               csrVal,
                                      int*                      csrRowPtr,
                                      int*                      csrColInd);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZdense2csr(hipsparseHandle_t         handle,
                                      int                       m,
                                      int                       n,
                                      const hipsparseMatDescr_t descr,
                                      const hipDoubleComplex*   A,
                                      int                       ld,
                                      const int*                nnzPerRow,
                                      hipDoubleComplex*         csrVal,
                                      int*                      csrRowPtr,
                                      int*                      csrColInd);
/**@}*/
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
/*! \ingroup conv_module
*  \brief
*  \p hipsparseXpruneDense2csr_bufferSize computes the the size of the user allocated temporary storage buffer 
*  used when converting a dense matrix to a pruned CSR matrix. 
*
*  \details
*  Specifically given an input dense column ordered matrix A, with leading dimension \p lda where \p lda>=m, 
*  the resulting pruned sparse CSR matrix C is computed using:
*  \f[ 
*   |C(i,j)| = A(i, j) \text{  if |A(i, j)| > threshold}
*  \f]
*
*  The first step in this conversion is to determine the required user allocated buffer size 
*  using \p hipsparseXpruneDense2csr_bufferSize() that will be passed to the subsequent steps of the conversion. 
*  Once the buffer size has been determined the user must allocate it. This user allocated buffer is then passed 
*  to \ref hipsparseSpruneDense2csrNnz "hipsparseXpruneDense2csrNnz()" and \ref hipsparseSpruneDense2csr 
*  "hipsparseXpruneDense2csr()" to complete the conversion. The user is responsible to then free the buffer once 
*  the conversion has been completed. 
*
*  See hipsparseSpruneDense2csr() for a full code example.
*
*  @param[in]
*  handle             handle to the hipsparse library context queue.
*  @param[in]
*  m                  number of rows of the dense matrix \p A.
*  @param[in]
*  n                  number of columns of the dense matrix \p A.
*  @param[in]
*  A                  array of dimensions (\p lda, \p n)
*  @param[in]
*  lda                leading dimension of dense array \p A.
*  @param[in]
*  threshold          pointer to the pruning non-negative threshold which can exist in either host or device memory.
*  @param[in]
*  descr              the descriptor of the dense matrix \p A, the supported matrix type is  \ref HIPSPARSE_MATRIX_TYPE_GENERAL 
*                     and also any valid value of the \ref hipsparseIndexBase_t.
*  @param[in]
*  csrVal             array of nnz ( = \p csrRowPtr[m] - \p csrRowPtr[0] ) nonzero elements of matrix \p A.
*  @param[in]
*  csrRowPtr          integer array of \p m+1 elements that contains the start of every row and the end of the last row plus one.
*  @param[in]
*  csrColInd          integer array of nnz ( = \p csrRowPtr[m] - \p csrRowPtr[0] ) column indices of the non-zero elements of matrix \p A.
*  @param[out]
*  pBufferSizeInBytes number of bytes of the temporary storage buffer required by
*                     hipsparseSpruneDense2csrNnz(), hipsparseDpruneDense2csrNnz(),
*                     hipsparseSpruneDense2csr() and hipsparseDpruneDense2csr().
*
*  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle or \p pBufferSizeInBytes pointer is invalid.
*  \retval     HIPSPARSE_STATUS_INTERNAL_ERROR an internal error occurred.
*/
/**@{*/
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpruneDense2csr_bufferSize(hipsparseHandle_t         handle,
                                                      int                       m,
                                                      int                       n,
                                                      const float*              A,
                                                      int                       lda,
                                                      const float*              threshold,
                                                      const hipsparseMatDescr_t descr,
                                                      const float*              csrVal,
                                                      const int*                csrRowPtr,
                                                      const int*                csrColInd,
                                                      size_t*                   pBufferSizeInBytes);
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDpruneDense2csr_bufferSize(hipsparseHandle_t         handle,
                                                      int                       m,
                                                      int                       n,
                                                      const double*             A,
                                                      int                       lda,
                                                      const double*             threshold,
                                                      const hipsparseMatDescr_t descr,
                                                      const double*             csrVal,
                                                      const int*                csrRowPtr,
                                                      const int*                csrColInd,
                                                      size_t*                   pBufferSizeInBytes);

DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpruneDense2csr_bufferSizeExt(hipsparseHandle_t         handle,
                                                         int                       m,
                                                         int                       n,
                                                         const float*              A,
                                                         int                       lda,
                                                         const float*              threshold,
                                                         const hipsparseMatDescr_t descr,
                                                         const float*              csrVal,
                                                         const int*                csrRowPtr,
                                                         const int*                csrColInd,
                                                         size_t* pBufferSizeInBytes);

DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDpruneDense2csr_bufferSizeExt(hipsparseHandle_t         handle,
                                                         int                       m,
                                                         int                       n,
                                                         const double*             A,
                                                         int                       lda,
                                                         const double*             threshold,
                                                         const hipsparseMatDescr_t descr,
                                                         const double*             csrVal,
                                                         const int*                csrRowPtr,
                                                         const int*                csrColInd,
                                                         size_t* pBufferSizeInBytes);
/**@}*/
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
/*! \ingroup conv_module
*  \brief
*  \p hipsparseXpruneDense2csrNnz function computes the number of nonzero elements per row and the total 
*  number of nonzero elements in a dense matrix once the elements less than the (non-negative) threshold are 
*  pruned from the matrix.
*
*  \details
*  Specifically given an input dense column ordered matrix A, with leading dimension \p lda where \p lda>=m, 
*  the resulting pruned sparse CSR matrix C is computed using:
*  \f[ 
*   |C(i,j)| = A(i, j) \text{  if |A(i, j)| > threshold}
*  \f]
*
*  First the user must determine the size of the required temporary buffer using the routine 
*  \ref hipsparseSpruneDense2csr_bufferSize "hipsparseXpruneDense2csr_bufferSize()" and then allocate it. Next 
*  the user allocates \p csrRowPtr with size \p m+1. Then the passes both the temporary storage buffer as well 
*  as \p csrRowPtr to \p hipsparseXpruneDense2csrNnz in order to determine the total number of non-zeros that 
*  will exist in the sparse CSR matrix C (after pruning has been performed on A) as well as fill the output CSR 
*  row pointer array \p csrRowPtr.
*
*  For example, given the dense matrix:
*
*  \f[
*    \begin{bmatrix}
*    6 & 2 & 3 & 7 \\
*    5 & 6 & 7 & 8 \\
*    5 & 4 & 8 & 1
*    \end{bmatrix}
*  \f]
*
*  and the \p threshold value 5, the resulting matrix after pruning is:
*
*  \f[
*    \begin{bmatrix}
*    6 & 0 & 0 & 7 \\
*    0 & 6 & 7 & 8 \\
*    0 & 0 & 8 & 0
*    \end{bmatrix}
*  \f]
*
*  and corresponding row pointer array and non-zero count:
*
*  \f[
*    \begin{align}
*    \text{csrRowPtr} &= \begin{bmatrix} 0 & 2 & 5 & 6 \end{bmatrix} \\
*    \text{nnzTotalDevHostPtr} &= 6
*    \end{align}
*  \f]
*
*  The above example assumes a zero index base for the output CSR matrix. We can set the desired index base 
*  in the output CSR matrix by setting it in the \ref hipsparseMatDescr_t. See \ref hipsparseSetMatIndexBase().
*
*  For a full code example on how to use this routine, see hipsparseSpruneDense2csr().  
*
*  \note
*  The routine does support asynchronous execution if the pointer mode is set to device.
*
*  @param[in]
*  handle             handle to the hipsparse library context queue.
*  @param[in]
*  m                  number of rows of the dense matrix \p A.
*  @param[in]
*  n                  number of columns of the dense matrix \p A.
*  @param[in]
*  A                  array of dimensions (\p lda, \p n)
*  @param[in]
*  lda                leading dimension of dense array \p A.
*  @param[in]
*  threshold          pointer to the pruning non-negative threshold which can exist in either host or device memory.
*  @param[in]
*  descr              the descriptor of the dense matrix \p A.
*  @param[out]
*  csrRowPtr          integer array of \p m+1 elements that contains the start of every row and the end of the last row plus one.
*  @param[out]
*  nnzTotalDevHostPtr total number of nonzero elements in device or host memory.
*  @param[out]
*  buffer             buffer allocated by the user whose size is determined by calling 
*                     \ref hipsparseSpruneDense2csr_bufferSize "hipsparseXpruneDense2csr_bufferSize()" or 
*                     \ref hipsparseSpruneDense2csr_bufferSizeExt "hipsparseXpruneDense2csr_bufferSizeExt()".
*
*  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p m, \p n, \p lda, \p A, \p threshold, \p descr, \p csrRowPtr
*              \p nnzTotalDevHostPtr or \p buffer pointer is invalid.
*/
/**@{*/
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpruneDense2csrNnz(hipsparseHandle_t         handle,
                                              int                       m,
                                              int                       n,
                                              const float*              A,
                                              int                       lda,
                                              const float*              threshold,
                                              const hipsparseMatDescr_t descr,
                                              int*                      csrRowPtr,
                                              int*                      nnzTotalDevHostPtr,
                                              void*                     buffer);

DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDpruneDense2csrNnz(hipsparseHandle_t         handle,
                                              int                       m,
                                              int                       n,
                                              const double*             A,
                                              int                       lda,
                                              const double*             threshold,
                                              const hipsparseMatDescr_t descr,
                                              int*                      csrRowPtr,
                                              int*                      nnzTotalDevHostPtr,
                                              void*                     buffer);
/**@}*/
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
/*! \ingroup conv_module
*  \brief
*  \p hipsparseXpruneDense2csr converts the matrix A in dense format into a sparse matrix in CSR format
*  while pruning values that are less than the (non-negative) threshold. All the parameters are assumed
*  to have been pre-allocated by the user.
*
*  \details
*  Specifically given an input dense column ordered matrix A, with leading dimension \p lda where \p lda>=m, 
*  the resulting pruned sparse CSR matrix C is computed using:
*  \f[ 
*   |C(i,j)| = A(i, j) \text{  if |A(i, j)| > threshold}
*  \f]
*
*  The user first calls \ref hipsparseSpruneDense2csr_bufferSize "hipsparseXpruneDense2csr_bufferSize()" to 
*  determine the size of the required user allocate temporary storage buffer. The user then allocates this 
*  buffer. Next, the user allocates \p csrRowPtr to have \p m+1 elements and then calls 
*  \ref hipsparseSpruneDense2csrNnz "hipsparseXpruneDense2csrNnz()" which fills in the \p csrRowPtr array 
*  and stores the number of elements that are larger than the pruning \p threshold in \p nnzTotalDevHostPtr. 
*  The user then allocates \p csrColInd and \p csrVal to have size \p nnzTotalDevHostPtr and completes the 
*  conversion by calling \p hipsparseXpruneDense2csr().
*
*  For example, performing these steps with the dense input matrix A:
*  \f[
*    \begin{bmatrix}
*    6 & 2 & 3 & 7 \\
*    5 & 6 & 7 & 8 \\
*    5 & 4 & 8 & 1
*    \end{bmatrix}
*  \f]
*
*  and the \p threshold value 5, results in the pruned matrix C:
*
*  \f[
*    \begin{bmatrix}
*    6 & 0 & 0 & 7 \\
*    0 & 6 & 7 & 8 \\
*    0 & 0 & 8 & 0
*    \end{bmatrix}
*  \f]
*
*  and corresponding CSR row, column, and values arrays:
*
*  \f[
*    \begin{align}
*    \text{csrRowPtr} &= \begin{bmatrix} 0 & 2 & 5 & 6 \end{bmatrix} \\
*    \text{csrColInd} &= \begin{bmatrix} 0 & 3 & 1 & 2 & 3 & 2 \end{bmatrix} \\
*    \text{csrVal} &= \begin{bmatrix} 6 & 7 & 6 & 7 & 8 & 8 \end{bmatrix} \\
*    \end{align}
*  \f]
*
*  \note 
*  The routine \p hipsparseXpruneDense2csr() is executed asynchronously with respect to the host and may 
*  return control to the application on the host before the entire result is ready.
*
*  @param[in]
*  handle      handle to the hipsparse library context queue.
*  @param[in]
*  m           number of rows of the dense matrix \p A.
*  @param[in]
*  n           number of columns of the dense matrix \p A.
*  @param[in]
*  A           array of dimensions (\p lda, \p n)
*  @param[in]
*  lda         leading dimension of dense array \p A.
*  @param[in]
*  threshold   pointer to the non-negative pruning threshold which can exist in either host or device memory.
*  @param[in]
*  descr       the descriptor of the dense matrix \p A, the supported matrix type is  \ref HIPSPARSE_MATRIX_TYPE_GENERAL 
*              and also any valid value of the \ref hipsparseIndexBase_t.
*  @param[out]
*  csrVal      array of nnz ( = \p csrRowPtr[m] - \p csrRowPtr[0] ) nonzero elements of matrix \p A.
*  @param[in]
*  csrRowPtr  integer array of \p m+1 elements that contains the start of every row and the end of the last row plus one.
*  @param[out]
*  csrColInd  integer array of nnz ( = \p csrRowPtr[m] - \p csrRowPtr[0] ) column indices of the non-zero elements of matrix \p A.
*
*  @param[in]
*  buffer     temporary storage buffer allocated by the user, size is returned by
*             \ref hipsparseSpruneDense2csr_bufferSize "hipsparseXpruneDense2csr_bufferSize()" or 
*             \ref hipsparseSpruneDense2csr_bufferSizeExt "hipsparseXpruneDense2csr_bufferSizeExt()".
*
*  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p m, \p n, \p lda, \p A, \p descr, \p threshold, \p csrVal
*              \p csrRowPtr, \p csrColInd, \p buffer pointer is invalid.
*
*  \par Example
*  \code{.c}
*    // hipSPARSE handle
*    hipsparseHandle_t handle;
*    hipsparseCreate(&handle);
*
*    // Matrix descriptor
*    hipsparseMatDescr_t descr;
*    hipsparseCreateMatDescr(&descr);
*
*    // Dense matrix in column order
*    //     1 2 0 3 0
*    // A = 0 4 5 0 0
*    //     6 0 0 7 8
*    float hdense_A[15] = {1.0f, 0.0f, 6.0f, 2.0f, 4.0f, 0.0f, 0.0f, 5.0f, 0.0f, 3.0f, 0.0f, 7.0f, 0.0f, 0.0f, 8.0f};
*
*    int m         = 3;
*    int n         = 5;
*    int lda       = m;
*    float threshold = 4.0f;
*
*    float* ddense_A = nullptr;
*    hipMalloc((void**)&ddense_A, sizeof(float) * lda * n);
*    hipMemcpy(ddense_A, hdense_A, sizeof(float) * lda * n, hipMemcpyHostToDevice);
*
*    // Allocate sparse CSR matrix
*    int* dcsrRowPtr = nullptr;
*    hipMalloc((void**)&dcsrRowPtr, sizeof(int) * (m + 1));
*
*    size_t bufferSize;
*    hipsparseSpruneDense2csr_bufferSize(handle, m, n, ddense_A, lda, &threshold, descr, nullptr, dcsrRowPtr, nullptr, &bufferSize);
*
*    void* dbuffer = nullptr;
*    hipMalloc((void**)&dbuffer, bufferSize);
*
*    int nnz_A;
*    hipsparseSpruneDense2csrNnz(handle, m, n, ddense_A, lda, &threshold, descr, dcsrRowPtr, &nnz_A, dbuffer);
*
*    int* dcsrColInd = nullptr;
*    float* dcsrVal = nullptr;
*    hipMalloc((void**)&dcsrColInd, sizeof(int) * nnz_A);
*    hipMalloc((void**)&dcsrVal, sizeof(float) * nnz_A);
*
*    hipsparseSpruneDense2csr(handle, m, n, ddense_A, lda, &threshold, descr, dcsrVal, dcsrRowPtr, dcsrColInd, dbuffer);
*
*    hipFree(dcsrRowPtr);
*    hipFree(dcsrColInd);
*    hipFree(dcsrVal);
*    hipFree(ddense_A);
*    hipFree(dbuffer);
*
*    hipsparseDestroyMatDescr(descr);
*    hipsparseDestroy(handle);
*  \endcode
*/
/**@{*/
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpruneDense2csr(hipsparseHandle_t         handle,
                                           int                       m,
                                           int                       n,
                                           const float*              A,
                                           int                       lda,
                                           const float*              threshold,
                                           const hipsparseMatDescr_t descr,
                                           float*                    csrVal,
                                           const int*                csrRowPtr,
                                           int*                      csrColInd,
                                           void*                     buffer);

DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDpruneDense2csr(hipsparseHandle_t         handle,
                                           int                       m,
                                           int                       n,
                                           const double*             A,
                                           int                       lda,
                                           const double*             threshold,
                                           const hipsparseMatDescr_t descr,
                                           double*                   csrVal,
                                           const int*                csrRowPtr,
                                           int*                      csrColInd,
                                           void*                     buffer);
/**@}*/
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
/*! \ingroup conv_module
*  \brief
*  \p hipsparseXpruneDense2csrByPercentage_bufferSize computes the size of the user allocated temporary 
*  storage buffer used when converting a dense matrix to a pruned CSR matrix where the pruning is done 
*  based on a percantage.
*
*  \details
*  When converting and pruning a dense matrix A to a CSR matrix by percentage the
*  following steps are performed. First the user calls
*  \p hipsparseXpruneDense2csrByPercentage_bufferSize which determines the size of the
*  temporary storage buffer. Once determined, this buffer must be allocated by the user.
*  Next the user allocates the \p csrRowPtr array to have \p m+1 elements and calls
*  \ref hipsparseSpruneDense2csrNnzByPercentage "hipsparseXpruneDense2csrNnzByPercentage()". 
*  Finally the user finishes the conversion by allocating the \p csrColInd and \p csrVal arrays 
*  (whose size is determined by the value at \p nnzTotalDevHostPtr) and calling 
*  \ref hipsparseSpruneDense2csrByPercentage "hipsparseXpruneDense2csrByPercentage()".
*
*  The pruning by percentage works by first sorting the absolute values of the dense
*  matrix \p A. We then determine a position in this sorted array by
*  \f[
*    pos = ceil(m \cdot n \cdot (percentage/100)) - 1 \\
*    pos = \min(pos, m \cdot n-1) \\
*    pos = \max(pos, 0) \\
*    threshold = sorted_A[pos]
*  \f]
*
*  Once we have this threshold we prune values in the dense matrix \p A as in
*  \ref hipsparseSpruneDense2csr "hipsparseXpruneDense2csr()". 
*
*  \note
*  It is executed asynchronously with respect to the host and may return control to the 
*  application on the host before the entire result is ready.
*
*  @param[in]
*  handle             handle to the hipsparse library context queue.
*  @param[in]
*  m                  number of rows of the dense matrix \p A.
*  @param[in]
*  n                  number of columns of the dense matrix \p A.
*  @param[in]
*  A                  array of dimensions (\p lda, \p n)
*  @param[in]
*  lda                leading dimension of dense array \p A.
*  @param[in]
*  percentage         \p percentage>=0 and \p percentage<=100.
*  @param[in]
*  descr              the descriptor of the dense matrix \p A, the supported matrix type is  \ref HIPSPARSE_MATRIX_TYPE_GENERAL and also 
*                     any valid value of the \ref hipsparseIndexBase_t.
*  @param[in]
*  csrVal             array of nnz ( = \p csrRowPtr[m] - \p csrRowPtr[0] ) nonzero elements of matrix \p A.
*  @param[in]
*  csrRowPtr          integer array of \p m+1 elements that contains the start of every row and the end of the last row plus one.
*  @param[in]
*  csrColInd          integer array of nnz ( = \p csrRowPtr[m] - \p csrRowPtr[0] ) column indices of the non-zero elements of matrix \p A.
*  @param[in]
*  info               prune information structure
*  @param[out]
*  pBufferSizeInBytes number of bytes of the temporary storage buffer required by
*                     hipsparseSpruneDense2csrNnzByPercentage(), hipsparseDpruneDense2csrNnzByPercentage().
*
*  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE the \p handle or \p pBufferSizeInBytes pointer is invalid.
*  \retval     HIPSPARSE_STATUS_INTERNAL_ERROR an internal error occurred.
*/
/**@{*/
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpruneDense2csrByPercentage_bufferSize(hipsparseHandle_t handle,
                                                                  int               m,
                                                                  int               n,
                                                                  const float*      A,
                                                                  int               lda,
                                                                  float             percentage,
                                                                  const hipsparseMatDescr_t descr,
                                                                  const float*              csrVal,
                                                                  const int*  csrRowPtr,
                                                                  const int*  csrColInd,
                                                                  pruneInfo_t info,
                                                                  size_t*     pBufferSizeInBytes);
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDpruneDense2csrByPercentage_bufferSize(hipsparseHandle_t handle,
                                                                  int               m,
                                                                  int               n,
                                                                  const double*     A,
                                                                  int               lda,
                                                                  double            percentage,
                                                                  const hipsparseMatDescr_t descr,
                                                                  const double*             csrVal,
                                                                  const int*  csrRowPtr,
                                                                  const int*  csrColInd,
                                                                  pruneInfo_t info,
                                                                  size_t*     pBufferSizeInBytes);
/**@}*/
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
/*! \ingroup conv_module
*  \brief
*  This function computes the size of the user allocated temporary storage buffer used
*  when converting and pruning by percentage a dense matrix to a CSR matrix.
*
*  \details
*  When converting and pruning a dense matrix A to a CSR matrix by percentage the
*  following steps are performed. First the user calls
*  \p hipsparseXpruneDense2csrByPercentage_bufferSizeExt which determines the size of the
*  temporary storage buffer. Once determined, this buffer must be allocated by the user.
*  Next the user allocates the \p csrRowPtr array to have \p m+1 elements and calls
*  \ref hipsparseSpruneDense2csrNnzByPercentage "hipsparseXpruneDense2csrNnzByPercentage()". 
*  Finally the user finishes the conversion by allocating the \p csrColInd and \p csrVal arrays 
*  (whos size is determined by the value at \p nnzTotalDevHostPtr) and calling 
*  \ref hipsparseSpruneDense2csrByPercentage "hipsparseXpruneDense2csrByPercentage()".
*
*  The pruning by percentage works by first sorting the absolute values of the dense
*  matrix \p A. We then determine a position in this sorted array by
*  \f[
*    pos = ceil(m \cdot n \cdot (percentage/100)) - 1 \\
*    pos = \min(pos, m \cdot n-1) \\
*    pos = \max(pos, 0) \\
*    threshold = sorted_A[pos]
*  \f]
*
*  Once we have this threshold we prune values in the dense matrix \p A as in
*  \ref hipsparseSpruneDense2csr "hipsparseXpruneDense2csr()". 
*
*  \note
*  It is executed asynchronously with respect to the host and may return control to the 
*  application on the host before the entire result is ready.
*
*  @param[in]
*  handle             handle to the hipsparse library context queue.
*  @param[in]
*  m                  number of rows of the dense matrix \p A.
*  @param[in]
*  n                  number of columns of the dense matrix \p A.
*  @param[in]
*  A                  array of dimensions (\p lda, \p n)
*  @param[in]
*  lda                leading dimension of dense array \p A.
*  @param[in]
*  percentage         \p percentage>=0 and \p percentage<=100.
*  @param[in]
*  descr              the descriptor of the dense matrix \p A, the supported matrix type is  \ref HIPSPARSE_MATRIX_TYPE_GENERAL and also 
*                     any valid value of the \ref hipsparseIndexBase_t.
*  @param[in]
*  csrVal             array of nnz ( = \p csrRowPtr[m] - \p csrRowPtr[0] ) nonzero elements of matrix \p A.
*  @param[in]
*  csrRowPtr          integer array of \p m+1 elements that contains the start of every row and the end of the last row plus one.
*  @param[in]
*  csrColInd          integer array of nnz ( = \p csrRowPtr[m] - \p csrRowPtr[0] ) column indices of the non-zero elements of matrix \p A.
*  @param[in]
*  info               prune information structure
*  @param[out]
*  pBufferSizeInBytes number of bytes of the temporary storage buffer required by
*                     hipsparseSpruneDense2csrNnzByPercentage(), hipsparseDpruneDense2csrNnzByPercentage().
*
*  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE the \p handle or \p pBufferSizeInBytes pointer is invalid.
*  \retval     HIPSPARSE_STATUS_INTERNAL_ERROR an internal error occurred.
*/
/**@{*/
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t
    hipsparseSpruneDense2csrByPercentage_bufferSizeExt(hipsparseHandle_t         handle,
                                                       int                       m,
                                                       int                       n,
                                                       const float*              A,
                                                       int                       lda,
                                                       float                     percentage,
                                                       const hipsparseMatDescr_t descr,
                                                       const float*              csrVal,
                                                       const int*                csrRowPtr,
                                                       const int*                csrColInd,
                                                       pruneInfo_t               info,
                                                       size_t* pBufferSizeInBytes);

DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t
    hipsparseDpruneDense2csrByPercentage_bufferSizeExt(hipsparseHandle_t         handle,
                                                       int                       m,
                                                       int                       n,
                                                       const double*             A,
                                                       int                       lda,
                                                       double                    percentage,
                                                       const hipsparseMatDescr_t descr,
                                                       const double*             csrVal,
                                                       const int*                csrRowPtr,
                                                       const int*                csrColInd,
                                                       pruneInfo_t               info,
                                                       size_t* pBufferSizeInBytes);
/**@}*/
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
/*! \ingroup conv_module
*  \brief
*  This function computes the number of nonzero elements per row and the total number of
*  nonzero elements in a dense matrix when converting and pruning by percentage a dense
*  matrix to a CSR matrix.
*
*  \details
*  When converting and pruning a dense matrix A to a CSR matrix by percentage the
*  following steps are performed. First the user calls
*  \ref hipsparseSpruneDense2csrByPercentage_bufferSize "hipsparseXpruneDense2csrByPercentage_bufferSize()" 
*  which determines the size of the temporary storage buffer. Once determined, this buffer must be allocated 
*  by the user. Next the user allocates the \p csrRowPtr array to have \p m+1 elements and calls
*  \p hipsparseXpruneDense2csrNnzByPercentage. Finally the user finishes the conversion
*  by allocating the \p csrColInd and \p csrVal arrays (whos size is determined by the value
*  at \p nnzTotalDevHostPtr) and calling \ref hipsparseSpruneDense2csrByPercentage 
*  "hipsparseXpruneDense2csrByPercentage()".
*
*  The pruning by percentage works by first sorting the absolute values of the dense
*  matrix \p A. We then determine a position in this sorted array by
*  \f[
*    pos = ceil(m \cdot n \cdot (percentage/100)) - 1 \\
*    pos = \min(pos, m \cdot n-1) \\
*    pos = \max(pos, 0) \\
*    threshold = sorted_A[pos]
*  \f]
*
*  Once we have this threshold we prune values in the dense matrix \p A as in
*  \ref hipsparseSpruneDense2csr "hipsparseXpruneDense2csr()".
*
*  \note
*  The routine does support asynchronous execution if the pointer mode is set to device.
*
*  @param[in]
*  handle             handle to the hipsparse library context queue.
*  @param[in]
*  m                  number of rows of the dense matrix \p A.
*  @param[in]
*  n                  number of columns of the dense matrix \p A.
*  @param[in]
*  A                  array of dimensions (\p lda, \p n)
*  @param[in]
*  lda                leading dimension of dense array \p A.
*  @param[in]
*  percentage         \p percentage>=0 and \p percentage<=100.
*  @param[in]
*  descr              the descriptor of the dense matrix \p A.
*  @param[out]
*  csrRowPtr          integer array of \p m+1 elements that contains the start of every row and the end of the last row plus one.
*  @param[out]
*  nnzTotalDevHostPtr total number of nonzero elements in device or host memory.
*  @param[in]
*  info               prune information structure
*  @param[out]
*  buffer             buffer allocated by the user whose size is determined by calling 
*                     \ref hipsparseSpruneDense2csrByPercentage_bufferSize "hipsparseXpruneDense2csrByPercentage_bufferSize()" 
*                     or \ref hipsparseSpruneDense2csrByPercentage_bufferSizeExt "hipsparseXpruneDense2csrByPercentage_bufferSizeExt()".
*
*  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p m, \p n, \p lda, \p percentage, \p A, \p descr, \p info, \p csrRowPtr
*              \p nnzTotalDevHostPtr or \p buffer pointer is invalid.
*/
/**@{*/
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpruneDense2csrNnzByPercentage(hipsparseHandle_t         handle,
                                                          int                       m,
                                                          int                       n,
                                                          const float*              A,
                                                          int                       lda,
                                                          float                     percentage,
                                                          const hipsparseMatDescr_t descr,
                                                          int*                      csrRowPtr,
                                                          int*        nnzTotalDevHostPtr,
                                                          pruneInfo_t info,
                                                          void*       buffer);

DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDpruneDense2csrNnzByPercentage(hipsparseHandle_t         handle,
                                                          int                       m,
                                                          int                       n,
                                                          const double*             A,
                                                          int                       lda,
                                                          double                    percentage,
                                                          const hipsparseMatDescr_t descr,
                                                          int*                      csrRowPtr,
                                                          int*        nnzTotalDevHostPtr,
                                                          pruneInfo_t info,
                                                          void*       buffer);
/**@}*/
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
/*! \ingroup conv_module
*  \brief
*  This function computes the number of nonzero elements per row and the total number of
*  nonzero elements in a dense matrix when converting and pruning by percentage a dense
*  matrix to a CSR matrix.
*
*  \details
*  When converting and pruning a dense matrix A to a CSR matrix by percentage the
*  following steps are performed. First the user calls \ref hipsparseSpruneDense2csrByPercentage_bufferSize 
*  "hipsparseXpruneDense2csrByPercentage_bufferSize()" which determines the size of the
*  temporary storage buffer. Once determined, this buffer must be allocated by the user.
*  Next the user allocates the \p csrRowPtr array to have \p m+1 elements and calls
*  \ref hipsparseSpruneDense2csrNnzByPercentage "hipsparseXpruneDense2csrNnzByPercentage()". Finally the 
*  user finishes the conversion by allocating the \p csrColInd and \p csrVal arrays (whos size is 
*  determined by the value at \p nnzTotalDevHostPtr) and calling \p hipsparseXpruneDense2csrByPercentage.
*
*  The pruning by percentage works by first sorting the absolute values of the dense
*  matrix \p A. We then determine a position in this sorted array by
*  \f[
*    pos = ceil(m \ cdot n \cdot (percentage/100)) - 1 \\
*    pos = \min(pos, m \cdot n-1) \\
*    pos = \max(pos, 0) \\
*    threshold = sorted_A[pos]
*  \f]
*
*  Once we have this threshold we prune values in the dense matrix \p A as in
*  \ref hipsparseSpruneDense2csr "hipsparseXpruneDense2csr()".
*
*  \note
*  The routine does support asynchronous execution if the pointer mode is set to device.
*
*  @param[in]
*  handle      handle to the hipsparse library context queue.
*  @param[in]
*  m           number of rows of the dense matrix \p A.
*  @param[in]
*  n           number of columns of the dense matrix \p A.
*  @param[in]
*  A           array of dimensions (\p lda, \p n)
*  @param[in]
*  lda         leading dimension of dense array \p A.
*  @param[in]
*  percentage  \p percentage>=0 and \p percentage<=100.
*  @param[in]
*  descr       the descriptor of the dense matrix \p A, the supported matrix type is  \ref HIPSPARSE_MATRIX_TYPE_GENERAL and 
*              also any valid value of the \ref hipsparseIndexBase_t.
*  @param[out]
*  csrVal      array of nnz ( = \p csrRowPtr[m] - \p csrRowPtr[0] ) nonzero elements of matrix \p A.
*  @param[in]
*  csrRowPtr   integer array of \p m+1 elements that contains the start of every row and the end of the last row plus one.
*  @param[out]
*  csrColInd   integer array of nnz ( = \p csrRowPtr[m] - \p csrRowPtr[0] ) column indices of the non-zero elements of matrix \p A.
*  @param[in]
*  info prune  information structure
*  @param[in]
*  buffer      temporary storage buffer allocated by the user, size is returned by
*              \ref hipsparseSpruneDense2csrByPercentage_bufferSize "hipsparseXpruneDense2csrByPercentage_bufferSize()" or 
*              \ref hipsparseSpruneDense2csrByPercentage_bufferSizeExt "hipsparseXpruneDense2csrByPercentage_bufferSizeExt()".
*
*  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p m, \p n, \p lda, \p percentage, \p A, \p descr, \p info, \p csrVal
*              \p csrRowPtr, \p csrColInd or \p buffer pointer is invalid.
*
*  \par Example
*  \code{.c}
*    // hipSPARSE handle
*    hipsparseHandle_t handle;
*    hipsparseCreate(&handle);
*
*    // Matrix descriptor
*    hipsparseMatDescr_t descr;
*    hipsparseCreateMatDescr(&descr);
*
*    // Dense matrix in column order
*    //     1 2 0 3 0
*    // A = 0 4 5 0 0
*    //     6 0 0 7 8
*    float hdense_A[15] = {1.0f, 0.0f, 6.0f, 2.0f, 4.0f, 0.0f, 0.0f, 5.0f, 0.0f, 3.0f, 0.0f, 7.0f, 0.0f, 0.0f, 8.0f};
*
*    int m         = 3;
*    int n         = 5;
*    int lda       = m;
*    float percentage = 70.0f;
*
*    float* ddense_A = nullptr;
*    hipMalloc((void**)&ddense_A, sizeof(float) * lda * n);
*    hipMemcpy(ddense_A, hdense_A, sizeof(float) * lda * n, hipMemcpyHostToDevice);
*
*    // Allocate sparse CSR matrix
*    int* dcsrRowPtr = nullptr;
*    hipMalloc((void**)&dcsrRowPtr, sizeof(int) * (m + 1));
*
*    pruneInfo_t info;
*    hipsparseCreatePruneInfo(&info);
*
*    size_t bufferSize;
*    hipsparseSpruneDense2csrByPercentage_bufferSize(handle, m, n, ddense_A, lda, percentage, descr, nullptr, dcsrRowPtr, nullptr, info, &bufferSize);
*
*    void* dbuffer = nullptr;
*    hipMalloc((void**)&dbuffer, bufferSize);
*
*    int nnz_A;
*    hipsparseSpruneDense2csrNnzByPercentage(handle, m, n, ddense_A, lda, percentage, descr, dcsrRowPtr, &nnz_A, info, dbuffer);
*
*    int* dcsrColInd = nullptr;
*    float* dcsrVal = nullptr;
*    hipMalloc((void**)&dcsrColInd, sizeof(int) * nnz_A);
*    hipMalloc((void**)&dcsrVal, sizeof(float) * nnz_A);
*
*    hipsparseSpruneDense2csrByPercentage(handle, m, n, ddense_A, lda, percentage, descr, dcsrVal, dcsrRowPtr, dcsrColInd, info, dbuffer);
*
*    hipFree(dcsrRowPtr);
*    hipFree(dcsrColInd);
*    hipFree(dcsrVal);
*    hipFree(ddense_A);
*    hipFree(dbuffer);
*
*    hipsparseDestroyPruneInfo(info);
*    hipsparseDestroyMatDescr(descr);
*    hipsparseDestroy(handle);
*  \endcode
*/
/**@{*/
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpruneDense2csrByPercentage(hipsparseHandle_t         handle,
                                                       int                       m,
                                                       int                       n,
                                                       const float*              A,
                                                       int                       lda,
                                                       float                     percentage,
                                                       const hipsparseMatDescr_t descr,
                                                       float*                    csrVal,
                                                       const int*                csrRowPtr,
                                                       int*                      csrColInd,
                                                       pruneInfo_t               info,
                                                       void*                     buffer);

DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDpruneDense2csrByPercentage(hipsparseHandle_t         handle,
                                                       int                       m,
                                                       int                       n,
                                                       const double*             A,
                                                       int                       lda,
                                                       double                    percentage,
                                                       const hipsparseMatDescr_t descr,
                                                       double*                   csrVal,
                                                       const int*                csrRowPtr,
                                                       int*                      csrColInd,
                                                       pruneInfo_t               info,
                                                       void*                     buffer);
/**@}*/
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 12000)
/*! \ingroup conv_module
*  \brief
*  \p hipsparseXdense2csc converts the matrix A in dense format into a sparse matrix in CSC format.
*
*  \details
*  Given a dense, column ordered, matrix \p A with leading dimension \p ld where \p ld>=m, 
*  \p hipsparseXdense2csc converts the matrix to a sparse CSC format matrix.
*  All the parameters are assumed to have been pre-allocated by the user and the arrays
*  are filled in based on number of nonzeros per row, which can be pre-computed with 
*  \ref hipsparseSnnz "hipsparseXnnz()". We can set the desired index base in the output CSC 
*  matrix by setting it in the \ref hipsparseMatDescr_t. See \ref hipsparseSetMatIndexBase(). 
*
*  As an example, if using index base zero (i.e. the default) and the dense 
*  matrix:
*
*  \f[
*    \begin{bmatrix}
*    1 & 0 & 0 & 2 \\
*    3 & 4 & 0 & 0 \\
*    5 & 0 & 6 & 7
*    \end{bmatrix}
*  \f]
*
*  where the \p A values have column ordering with leading dimension \p ld=m:
*  \f[
*    \text{A} &= \begin{bmatrix} 1 & 3 & 5 & 0 & 4 & 0 & 0 & 0 & 6 & 2 & 0 & 7 \end{bmatrix} \\
*  \f]
*
*  the conversion results in the CSC arrays:
*
*  \f[
*    \begin{align}
*    \text{cscRowInd} &= \begin{bmatrix} 0 & 1 & 2 & 1 & 2 & 0 & 2 \end{bmatrix} \\
*    \text{cscColPtr} &= \begin{bmatrix} 0 & 3 & 4 & 5 & 7 \end{bmatrix} \\
*    \text{cscVal} &= \begin{bmatrix} 1 & 3 & 5 & 4 & 6 & 2 & 7 \end{bmatrix} \\
*    \end{align}
*  \f]
*
*  This function works very similar to \ref hipsparseSdense2csr "hipsparseXdense2csr()". 
&  See hipsparseSdense2csr() for a code example.
*
*  \note
*  It is executed asynchronously with respect to the host and may return control to the
*  application on the host before the entire result is ready.
*
*  @param[in]
*  handle       handle to the hipsparse library context queue.
*  @param[in]
*  m            number of rows of the dense matrix \p A.
*  @param[in]
*  n            number of columns of the dense matrix \p A.
*  @param[in]
*  descr        the descriptor of the dense matrix \p A, the supported matrix type is  \ref HIPSPARSE_MATRIX_TYPE_GENERAL and also 
*               any valid value of the \ref hipsparseIndexBase_t.
*  @param[in]
*  A            array of dimensions (\p ld, \p n)
*  @param[in]
*  ld           leading dimension of dense array \p A.
*  @param[in]
*  nnzPerColumn array of size \p n containing the number of non-zero elements per column.
*  @param[out]
*  cscVal       array of nnz ( = \p cscColPtr[n] - \p cscColPtr[0] ) nonzero elements of matrix \p A.
*  @param[out]
*  cscRowInd    integer array of nnz ( = \p cscColPtr[n] - \p cscColPtr[0] ) column indices of the non-zero elements of matrix \p A.
*  @param[out]
*  cscColPtr    integer array of \p n+1 elements that contains the start of every column and the end of the last column plus one.
*
*  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p m, \p n, \p ld, \p A, \p nnzPerColumn or \p cscVal \p cscColPtr 
*              or \p cscRowInd pointer is invalid.
*/
/**@{*/
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSdense2csc(hipsparseHandle_t         handle,
                                      int                       m,
                                      int                       n,
                                      const hipsparseMatDescr_t descr,
                                      const float*              A,
                                      int                       ld,
                                      const int*                nnzPerColumn,
                                      float*                    cscVal,
                                      int*                      cscRowInd,
                                      int*                      cscColPtr);
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDdense2csc(hipsparseHandle_t         handle,
                                      int                       m,
                                      int                       n,
                                      const hipsparseMatDescr_t descr,
                                      const double*             A,
                                      int                       ld,
                                      const int*                nnzPerColumn,
                                      double*                   cscVal,
                                      int*                      cscRowInd,
                                      int*                      cscColPtr);
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCdense2csc(hipsparseHandle_t         handle,
                                      int                       m,
                                      int                       n,
                                      const hipsparseMatDescr_t descr,
                                      const hipComplex*         A,
                                      int                       ld,
                                      const int*                nnzPerColumn,
                                      hipComplex*               cscVal,
                                      int*                      cscRowInd,
                                      int*                      cscColPtr);
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZdense2csc(hipsparseHandle_t         handle,
                                      int                       m,
                                      int                       n,
                                      const hipsparseMatDescr_t descr,
                                      const hipDoubleComplex*   A,
                                      int                       ld,
                                      const int*                nnzPerColumn,
                                      hipDoubleComplex*         cscVal,
                                      int*                      cscRowInd,
                                      int*                      cscColPtr);
/**@}*/
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 12000)
/*! \ingroup conv_module
*  \brief
*  \p hipsparseXcsr2dense function converts the sparse matrix in CSR format into a dense matrix.
*
*  \details
*  Given the input CSR matrix of size \p mxn, the routine writes the matrix to the dense array \p A such 
*  that \p A has leading dimension \p ld and is column ordered. This means that \p A has size \p ldxn where
*  \p ld>=m. All the parameters are assumed to have been pre-allocated by the user. If the input CSR matrix 
*  has index base of one, it must be set in the \ref hipsparseMatDescr_t. See \ref hipsparseSetMatIndexBase() 
*  prior to calling \p hipsparseXcsr2dense.
*
*  For example, consider the sparse CSR matrix:
*  \f[
*    \begin{align}
*    \text{csrRowPtr} &= \begin{bmatrix} 0 & 2 & 4 & 7 \end{bmatrix} \\
*    \text{csrColInd} &= \begin{bmatrix} 0 & 3 & 0 & 1 & 0 & 2 & 3 \end{bmatrix} \\
*    \text{csrVal} &= \begin{bmatrix} 1 & 2 & 3 & 4 & 5 & 6 & 7 \end{bmatrix} \\
*    \end{align}
*  \f]
*
*  \p hipsparseXcsr2dense is used to convert to the dense matrix:
*  \f[
*    \begin{bmatrix}
*    1 & 0 & 0 & 2 \\
*    3 & 4 & 0 & 0 \\
*    5 & 0 & 6 & 7
*    \end{bmatrix}
*  \f]
*
*  where the values in the \p A array are column ordered:
*  \f[
*    \text{A} &= \begin{bmatrix} 1 & 3 & 5 & 0 & 4 & 0 & 0 & 0 & 6 & 2 & 0 & 7 \end{bmatrix} \\
*  \f]
*
*  \note
*  It is executed asynchronously with respect to the host and may return control to the application 
*  on the host before the entire result is ready.
*
*  @param[in]
*  handle      handle to the hipsparse library context queue.
*  @param[in]
*  m           number of rows of the dense matrix \p A.
*  @param[in]
*  n           number of columns of the dense matrix \p A.
*  @param[in]
*  descr       the descriptor of the dense matrix \p A, the supported matrix type is \ref HIPSPARSE_MATRIX_TYPE_GENERAL and 
*              also any valid value of the \ref hipsparseIndexBase_t.
*  @param[in]
*  csrVal      array of nnz ( = \p csrRowPtr[m] - \p csrRowPtr[0] ) nonzero elements of matrix \p A.
*  @param[in]
*  csrRowPtr   integer array of \p m+1 elements that contains the start of every row and the end of the last row plus one.
*  @param[in]
*  csrColInd   integer array of nnz ( = \p csrRowPtr[m] - \p csrRowPtr[0] ) column indices of the non-zero elements of matrix \p A.
*  @param[out]
*  A           array of dimensions (\p ld, \p n)
*  @param[out]
*  ld          leading dimension of dense array \p A.
*
*  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p m, \p n, \p ld, \p A, \p csrVal, \p csrRowPtr or \p csrColInd
*              pointer is invalid.
*
*  \par Example
*  \code{.c}
*    // hipSPARSE handle
*    hipsparseHandle_t handle;
*    hipsparseCreate(&handle);
*
*    // Matrix descriptor
*    hipsparseMatDescr_t descr;
*    hipsparseCreateMatDescr(&descr);
*
*    // Sparse matrix in CSR format
*    //     1 2 0 3 0
*    // A = 0 4 5 0 0
*    //     6 0 0 7 8
*    int hcsrRowPtr[4] = {0, 3, 5, 8};
*    int hcsrColInd[8] = {0, 1, 3, 1, 2, 0, 3, 4};
*    float hcsrVal[8]   = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f}; 
*
*    int m         = 3;
*    int n         = 5;
*    int ld        = 3;
*    int nnz       = 8;
*
*    int* dcsrRowPtr = nullptr;
*    int* dcsrColInd = nullptr;
*    float* dcsrVal = nullptr;
*    hipMalloc((void**)&dcsrRowPtr, sizeof(int) * (m + 1));
*    hipMalloc((void**)&dcsrColInd, sizeof(int) * nnz);
*    hipMalloc((void**)&dcsrVal, sizeof(float) * nnz);
*
*    hipMemcpy(dcsrRowPtr, hcsrRowPtr, sizeof(int) * (m + 1), hipMemcpyHostToDevice);
*    hipMemcpy(dcsrColInd, hcsrColInd, sizeof(int) * nnz, hipMemcpyHostToDevice);
*    hipMemcpy(dcsrVal, hcsrVal, sizeof(float) * nnz, hipMemcpyHostToDevice);
*
*    float* ddense_A = nullptr;
*    hipMalloc((void**)&ddense_A, sizeof(float) * ld * n);
*
*    hipsparseScsr2dense(handle, m, n, descr, dcsrVal, dcsrRowPtr, dcsrColInd, ddense_A, ld);
*
*    hipFree(dcsrRowPtr);
*    hipFree(dcsrColInd);
*    hipFree(dcsrVal);
*    hipFree(ddense_A);
*
*    hipsparseDestroyMatDescr(descr);
*    hipsparseDestroy(handle);
*  \endcode
*/
/**@{*/
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseScsr2dense(hipsparseHandle_t         handle,
                                      int                       m,
                                      int                       n,
                                      const hipsparseMatDescr_t descr,
                                      const float*              csrVal,
                                      const int*                csrRowPtr,
                                      const int*                csrColInd,
                                      float*                    A,
                                      int                       ld);
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDcsr2dense(hipsparseHandle_t         handle,
                                      int                       m,
                                      int                       n,
                                      const hipsparseMatDescr_t descr,
                                      const double*             csrVal,
                                      const int*                csrRowPtr,
                                      const int*                csrColInd,
                                      double*                   A,
                                      int                       ld);
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCcsr2dense(hipsparseHandle_t         handle,
                                      int                       m,
                                      int                       n,
                                      const hipsparseMatDescr_t descr,
                                      const hipComplex*         csrVal,
                                      const int*                csrRowPtr,
                                      const int*                csrColInd,
                                      hipComplex*               A,
                                      int                       ld);
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZcsr2dense(hipsparseHandle_t         handle,
                                      int                       m,
                                      int                       n,
                                      const hipsparseMatDescr_t descr,
                                      const hipDoubleComplex*   csrVal,
                                      const int*                csrRowPtr,
                                      const int*                csrColInd,
                                      hipDoubleComplex*         A,
                                      int                       ld);
/**@}*/
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 12000)
/*! \ingroup conv_module
*  \brief
*  \p hipsparseXcsc2dense function converts the sparse matrix in CSC format into a dense matrix.
*
*  \details
*  Given the input CSC matrix of size \p mxn, the routine writes the matrix to the dense array \p A such 
*  that \p A has leading dimension \p ld and is column ordered. This means that \p A has size \p ldxn where
*  \p ld>=m. All the parameters are assumed to have been pre-allocated by the user. If the input CSC matrix 
*  has index base of one, it must be set in the \ref hipsparseMatDescr_t. See \ref hipsparseSetMatIndexBase() 
*  prior to calling \p hipsparseXcsc2dense.
*
*  For example, consider the sparse CSC matrix:
*  \f[
*    \begin{align}
*    \text{cscRowInd} &= \begin{bmatrix} 0 & 1 & 2 & 1 & 2 & 0 & 2 \end{bmatrix} \\
*    \text{cscColPtr} &= \begin{bmatrix} 0 & 3 & 4 & 5 & 7 \end{bmatrix} \\
*    \text{cscVal} &= \begin{bmatrix} 1 & 3 & 5 & 4 & 6 & 2 & 7 \end{bmatrix} \\
*    \end{align}
*  \f]
*
*  \p hipsparseXcsc2dense is used to convert to the dense matrix:
*  \f[
*    \begin{bmatrix}
*    1 & 0 & 0 & 2 \\
*    3 & 4 & 0 & 0 \\
*    5 & 0 & 6 & 7
*    \end{bmatrix}
*  \f]
*
*  where the values in the \p A array are column ordered:
*  \f[
*    \text{A} &= \begin{bmatrix} 1 & 3 & 5 & 0 & 4 & 0 & 0 & 0 & 6 & 2 & 0 & 7 \end{bmatrix} \\
*  \f]
*
*  \note
*  It is executed asynchronously with respect to the host and may return control to the application 
*  on the host before the entire result is ready.
*
*  @param[in]
*  handle      handle to the hipsparse library context queue.
*  @param[in]
*  m           number of rows of the dense matrix \p A.
*  @param[in]
*  n           number of columns of the dense matrix \p A.
*  @param[in]
*  descr       the descriptor of the dense matrix \p A, the supported matrix type is \ref HIPSPARSE_MATRIX_TYPE_GENERAL and also 
*              any valid value of the \ref hipsparseIndexBase_t.
*  @param[in]
*  cscVal      array of nnz ( = \p cscColPtr[n] - \p cscColPtr[0] ) nonzero elements of matrix \p A.
*  @param[in]
*  cscRowInd   integer array of nnz ( = \p cscColPtr[n] - \p cscColPtr[0] ) column indices of the non-zero elements of matrix \p A.
*  @param[in]
*  cscColPtr   integer array of \p n+1 elements that contains the start of every column and the end of the last column plus one.
*  @param[out]
*  A           array of dimensions (\p ld, \p n)
*  @param[out]
*  ld          leading dimension of dense array \p A.
*
*  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p m, \p n, \p ld, \p A, \p cscVal \p cscColPtr 
*              or \p cscRowInd pointer is invalid.
*/
/**@{*/
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseScsc2dense(hipsparseHandle_t         handle,
                                      int                       m,
                                      int                       n,
                                      const hipsparseMatDescr_t descr,
                                      const float*              cscVal,
                                      const int*                cscRowInd,
                                      const int*                cscColPtr,
                                      float*                    A,
                                      int                       ld);
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDcsc2dense(hipsparseHandle_t         handle,
                                      int                       m,
                                      int                       n,
                                      const hipsparseMatDescr_t descr,
                                      const double*             cscVal,
                                      const int*                cscRowInd,
                                      const int*                cscColPtr,
                                      double*                   A,
                                      int                       ld);
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCcsc2dense(hipsparseHandle_t         handle,
                                      int                       m,
                                      int                       n,
                                      const hipsparseMatDescr_t descr,
                                      const hipComplex*         cscVal,
                                      const int*                cscRowInd,
                                      const int*                cscColPtr,
                                      hipComplex*               A,
                                      int                       ld);
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZcsc2dense(hipsparseHandle_t         handle,
                                      int                       m,
                                      int                       n,
                                      const hipsparseMatDescr_t descr,
                                      const hipDoubleComplex*   cscVal,
                                      const int*                cscRowInd,
                                      const int*                cscColPtr,
                                      hipDoubleComplex*         A,
                                      int                       ld);
/**@}*/
#endif

/*! \ingroup conv_module
*  \brief
*  This function computes the number of nonzero block columns per row and the total number of nonzero blocks in a sparse
*  BSR matrix given a sparse CSR matrix as input.
*
*  \details
*  Consider the matrix:
*  \f[
*    \begin{bmatrix}
*    1 & 0 & 0 & 2 \\
*    3 & 4 & 0 & 0 \\
*    5 & 0 & 6 & 7 \\
*    1 & 2 & 3 & 4
*    \end{bmatrix}
*  \f]
*
*  stored as a sparse CSR matrix. This function computes both the BSR row pointer array as well as the total number 
*  of non-zero blocks that results when converting the CSR matrix to the BSR format. Assuming a block dimension of 2, 
*  the above matrix once converted to BSR format looks like:
*
*  \f[
*   \left[ 
*    \begin{array}{c | c} 
*      \begin{array}{c c} 
*       1 & 0 \\ 
*       3 & 4 
*      \end{array} & 
*      \begin{array}{c c} 
*       0 & 2 \\ 
*       0 & 0 
*      \end{array} \\ 
*    \hline 
*      \begin{array}{c c} 
*       5 & 0 \\ 
*       1 & 2 
*      \end{array} & 
*      \begin{array}{c c} 
*       6 & 7 \\ 
*       3 & 4 
*      \end{array} \\ 
*   \end{array} 
*  \right] 
*  \f]
*
*  and the resulting BSR row pointer array and total non-zero blocks once \p hipsparseXcsr2bsrNnz has been called:
*
*  \f[
*    \begin{align}
*    \text{bsrRowPtrC} &= \begin{bmatrix} 0 & 2 & 4 \end{bmatrix} \\
*    \text{bsrNnzb} &= 4
*    \end{align}
*  \f]
*
*  In general, when converting a CSR matrix of size \p m x \p n to a BSR matrix, the resulting BSR matrix will have size 
*  \p mb x \p nb where \p mb and \p nb equal:
*
*  \f[
*    \begin{align}
*    \text{mb} &= \text{(m - 1) / blockDim + 1} \\
*    \text{nb} &= \text{(n - 1) / blockDim + 1}
*    \end{align}
*  \f]
*
*  In particular, it may be the case that \p blockDim does not divide evenly into \p m and/or \p n. In these cases, the 
*  CSR matrix is expanded in size in order to fit full BSR blocks. For example, using the original CSR matrix and block 
*  dimension 3 instead of 2, the function \p hipsparseXcsr2bsrNnz computes the BSR row pointer array and total number of 
*  non-zero blocks for the BSR matrix:
*
*  \f[
*   \left[ 
*    \begin{array}{c | c} 
*      \begin{array}{c c c} 
*       1 & 0 & 0 \\ 
*       3 & 4 & 0 \\
*       5 & 0 & 6
*      \end{array} & 
*      \begin{array}{c c c} 
*       2 & 0 & 0 \\ 
*       0 & 0 & 0 \\
*       7 & 0 & 0
*      \end{array} \\ 
*    \hline 
*      \begin{array}{c c c} 
*       1 & 2 & 3 \\ 
*       0 & 0 & 0 \\
*       0 & 0 & 0
*      \end{array} & 
*      \begin{array}{c c c} 
*       4 & 0 & 0 \\ 
*       0 & 0 & 0 \\
*       0 & 0 & 0
*      \end{array} \\ 
*   \end{array} 
*  \right] 
*  \f]
*
*  See hipsparseScsr2bsr() for full code example.
*
*  \note
*  The routine does support asynchronous execution if the pointer mode is set to device.
*
*  @param[in]
*  handle      handle to the hipsparse library context queue.
*  @param[in]
*  dirA        direction that specified whether to count nonzero elements by \ref HIPSPARSE_DIRECTION_ROW or by
*              \ref HIPSPARSE_DIRECTION_COLUMN.
*  @param[in]
*  m           number of rows of the sparse CSR matrix.
*  @param[in]
*  n           number of columns of the sparse CSR matrix.
*  @param[in]
*  descrA      descriptor of the sparse CSR matrix. Currently, only \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*  @param[in]
*  csrRowPtrA  integer array containing \p m+1 elements that point to the start of each row of the CSR matrix
*  @param[in]
*  csrColIndA  integer array of the column indices for each non-zero element in the CSR matrix
*  @param[in]
*  blockDim    the block dimension of the BSR matrix. Between \f$1\f$ and \f$\min(m, n)\f$
*  @param[in]
*  descrC      descriptor of the sparse BSR matrix. Currently, only \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*  @param[out]
*  bsrRowPtrC  integer array containing \p mb+1 elements that point to the start of each block row of the BSR matrix
*  @param[out]
*  bsrNnzb     total number of nonzero elements in device or host memory.
*
*  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p m, \p n, \p blockDim, \p csrRowPtrA, \p csrColIndA, 
*              \p bsrRowPtrC or \p bsrNnzb pointer is invalid.
*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseXcsr2bsrNnz(hipsparseHandle_t         handle,
                                       hipsparseDirection_t      dirA,
                                       int                       m,
                                       int                       n,
                                       const hipsparseMatDescr_t descrA,
                                       const int*                csrRowPtrA,
                                       const int*                csrColIndA,
                                       int                       blockDim,
                                       const hipsparseMatDescr_t descrC,
                                       int*                      bsrRowPtrC,
                                       int*                      bsrNnzb);

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
/*! \ingroup conv_module
*  This function is used as the first step in converting a CSR matrix to a compressed CSR matrix.
*
*  \details
*  Given a sparse CSR matrix and a non-negative tolerance, this function computes how many entries would be left
*  in each row of the matrix if elements less than the tolerance were removed. It also computes the total number
*  of remaining elements in the matrix. 
*
*  Specifically given an input sparse matrix A in CSR format, the resulting compressed sparse CSR matrix C is 
*  computed using:
*  \f[ 
*   C(i,j) = A(i, j) \text{  if |A(i, j)| > tol}
*  \f]
*
*  The user first allocates \p nnzPerRow with size \p m elements. Then calling \p hipsparseXnnz_compress, 
*  the function fills in the \p nnzPerRow array and sets the total number of nonzeros found in \p nnzC.
*
*  See hipsparseScsr2csr_compress() for full code example.
*
*  \note
*  In the case of complex matrices only the magnitude of the real part of \p tol is used.
*
*  @param[in]
*  handle        handle to the hipsparse library context queue.
*  @param[in]
*  m             number of rows of the sparse CSR matrix.
*  @param[in]
*  descrA        the descriptor of the sparse CSR matrix.
*  @param[in]
*  csrValA       array of \p nnzA elements of the sparse CSR matrix.
*  @param[in]
*  csrRowPtrA    array of \p m+1 elements that point to the start of every row of the
*                uncompressed sparse CSR matrix.
*  @param[out]
*  nnzPerRow     array of length \p m containing the number of entries that will be kept per row in
*                the final compressed CSR matrix.
*  @param[out]
*  nnzC          number of elements in the column indices and values arrays of the compressed
*                sparse CSR matrix. Can be either host or device pointer.
*  @param[in]
*  tol           the non-negative tolerance used for compression. If \p tol is complex then only the magnitude
*                of the real part is used. Entries in the input uncompressed CSR array that are below the tolerance
*                are removed in output compressed CSR matrix.
*
*  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p m, \p n, \p tol, \p csrValA, \p csrRowPtrA, \p nnzPerRow or \p nnzC
*              pointer is invalid.
*
*  \par Example
*  \code{.c}
*    // hipSPARSE handle
*    hipsparseHandle_t handle;
*    hipsparseCreate(&handle);
*
*    // Matrix descriptor
*    hipsparseMatDescr_t descr_A;
*    hipsparseCreateMatDescr(&descr_A);
*
*    //     1 2 0 3 0
*    // A = 0 4 5 0 0
*    //     6 0 0 7 8
*    float tol = 4.2f;
*
*    int m     = 3;
*    int n     = 5;
*    int nnz_A = 8;
*
*    int hcsrRowPtr_A[4] = {0, 3, 5, 8};             
*    float hcsrVal_A[8]   = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
*
*    int* dcsrRowPtr_A = nullptr;
*    float* dcsrVal_A = nullptr;
*    hipMalloc((void**)&dcsrRowPtr_A, sizeof(int) * (m + 1));
*    hipMalloc((void**)&dcsrVal_A, sizeof(float) * nnz_A);
*
*    hipMemcpy(dcsrRowPtr_A, hcsrRowPtr_A, sizeof(int) * (m + 1), hipMemcpyHostToDevice);
*    hipMemcpy(dcsrVal_A, hcsrVal_A, sizeof(float) * nnz_A, hipMemcpyHostToDevice);
*
*    // Allocate memory for the nnz_per_row array
*    int* dnnz_per_row;
*    hipMalloc((void**)&dnnz_per_row, sizeof(int) * m);
*
*    // Call snnz_compress() which fills in nnz_per_row array and finds the number
*    // of entries that will be in the compressed CSR matrix
*    int nnz_C;
*    hipsparseSnnz_compress(handle, m, descr_A, dcsrVal_A, dcsrRowPtr_A, dnnz_per_row, &nnz_C, tol);
*
*    hipFree(dcsrRowPtr_A);
*    hipFree(dcsrVal_A);
*    hipFree(dnnz_per_row);
*
*    hipsparseDestroyMatDescr(descr_A);
*    hipsparseDestroy(handle);
*  \endcode
*/
/**@{*/
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSnnz_compress(hipsparseHandle_t         handle,
                                         int                       m,
                                         const hipsparseMatDescr_t descrA,
                                         const float*              csrValA,
                                         const int*                csrRowPtrA,
                                         int*                      nnzPerRow,
                                         int*                      nnzC,
                                         float                     tol);

DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDnnz_compress(hipsparseHandle_t         handle,
                                         int                       m,
                                         const hipsparseMatDescr_t descrA,
                                         const double*             csrValA,
                                         const int*                csrRowPtrA,
                                         int*                      nnzPerRow,
                                         int*                      nnzC,
                                         double                    tol);

DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCnnz_compress(hipsparseHandle_t         handle,
                                         int                       m,
                                         const hipsparseMatDescr_t descrA,
                                         const hipComplex*         csrValA,
                                         const int*                csrRowPtrA,
                                         int*                      nnzPerRow,
                                         int*                      nnzC,
                                         hipComplex                tol);

DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZnnz_compress(hipsparseHandle_t         handle,
                                         int                       m,
                                         const hipsparseMatDescr_t descrA,
                                         const hipDoubleComplex*   csrValA,
                                         const int*                csrRowPtrA,
                                         int*                      nnzPerRow,
                                         int*                      nnzC,
                                         hipDoubleComplex          tol);
/**@}*/
#endif

/*! \ingroup conv_module
*  \brief Convert a sparse CSR matrix into a sparse COO matrix
*
*  \details
*  \p hipsparseXcsr2coo converts the CSR array containing the row offsets, that point
*  to the start of every row, into a COO array of row indices. All arrays are assumed 
*  to be allocated by the user prior to calling \p hipsparseXcsr2coo.
*
*  For example, given the CSR row pointer array (assuming zero index base):
*  \f[
*    \begin{align}
*    \text{csrRowPtr} &= \begin{bmatrix} 0 & 1 & 3 & 4 \end{bmatrix}
*    \end{align}
*  \f]
*
*  Calling \p hipsparseXcsr2coo() results in the COO row indices array:
*  \f[
*    \begin{align}
*    \text{cooRowInd} &= \begin{bmatrix} 0 & 1 & 1 & 2 \end{bmatrix}
*    \end{align}
*  \f]
*
*  \note
*  It can also be used to convert a CSC array containing the column offsets into a COO
*  array of column indices.
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*
*  @param[in]
*  handle      handle to the hipsparse library context queue.
*  @param[in]
*  csrRowPtr   array of \p m+1 elements that point to the start of every row
*              of the sparse CSR matrix.
*  @param[in]
*  nnz         number of non-zero entries of the sparse CSR matrix.
*  @param[in]
*  m           number of rows of the sparse CSR matrix.
*  @param[out]
*  cooRowInd   array of \p nnz elements containing the row indices of the sparse COO
*              matrix.
*  @param[in]
*  idxBase    \ref HIPSPARSE_INDEX_BASE_ZERO or \ref HIPSPARSE_INDEX_BASE_ONE.
*
*  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p m, \p nnz, \p csrRowPtr or \p cooRowInd
*              pointer is invalid.
*  \retval     HIPSPARSE_STATUS_ARCH_MISMATCH the device is not supported.
*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseXcsr2coo(hipsparseHandle_t    handle,
                                    const int*           csrRowPtr,
                                    int                  nnz,
                                    int                  m,
                                    int*                 cooRowInd,
                                    hipsparseIndexBase_t idxBase);

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 11000)
/*! \ingroup conv_module
*  \brief Convert a sparse CSR matrix into a sparse CSC matrix
*
*  \details
*  \p hipsparseXcsr2csc converts a CSR matrix into a CSC matrix. \p hipsparseXcsr2csc
*  can also be used to convert a CSC matrix into a CSR matrix. \p copyValues decides
*  whether \p cscSortedVal is being filled during conversion (\ref HIPSPARSE_ACTION_NUMERIC)
*  or not (\ref HIPSPARSE_ACTION_SYMBOLIC).
*
*  For example given the matrix:
*  \f[
*    \begin{bmatrix}
*    1 & 0 & 0 & 2 \\
*    3 & 4 & 0 & 0 \\
*    5 & 0 & 6 & 7
*    \end{bmatrix}
*  \f]
*
*  Represented using the sparse CSR format as:
*  \f[
*    \begin{align}
*    \text{csrSortedRowPtr} &= \begin{bmatrix} 0 & 2 & 4 & 7 \end{bmatrix} \\
*    \text{csrSortedColInd} &= \begin{bmatrix} 0 & 3 & 0 & 1 & 0 & 2 & 3 \end{bmatrix} \\
*    \text{csrSortedVal} &= \begin{bmatrix} 1 & 2 & 3 & 4 & 5 & 6 & 7 \end{bmatrix}
*    \end{align}
*  \f]
*
*  this function converts to sparse CSC format:
*  \f[
*    \begin{align}
*    \text{cscSortedRowInd} &= \begin{bmatrix} 0 & 1 & 2 & 1 & 2 & 0 & 2 \end{bmatrix} \\
*    \text{cscSortedColPtr} &= \begin{bmatrix} 0 & 3 & 4 & 5 & 7 \end{bmatrix} \\
*    \text{cscSortedVal} &= \begin{bmatrix} 1 & 3 & 5 & 4 & 6 & 2 & 7 \end{bmatrix}
*    \end{align}
*  \f]
*
*  The CSC arrays, \p cscSortedRowInd, \p cscSortedColPtr, and \p cscSortedVal must be allocated by the 
*  user prior to calling \p hipsparseXcsr2csc().
*
*  \note
*  The resulting matrix can also be seen as the transpose of the input matrix.
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*
*  @param[in]
*  handle          handle to the hipsparse library context queue.
*  @param[in]
*  m               number of rows of the sparse CSR matrix.
*  @param[in]
*  n               number of columns of the sparse CSR matrix.
*  @param[in]
*  nnz             number of non-zero entries of the sparse CSR matrix.
*  @param[in]
*  csrSortedVal    array of \p nnz elements of the sparse CSR matrix.
*  @param[in]
*  csrSortedRowPtr array of \p m+1 elements that point to the start of every row of the
*                  sparse CSR matrix.
*  @param[in]
*  csrSortedColInd array of \p nnz elements containing the column indices of the sparse
*                  CSR matrix.
*  @param[out]
*  cscSortedVal    array of \p nnz elements of the sparse CSC matrix.
*  @param[out]
*  cscSortedRowInd array of \p nnz elements containing the row indices of the sparse CSC
*                  matrix.
*  @param[out]
*  cscSortedColPtr array of \p n+1 elements that point to the start of every column of the
*                  sparse CSC matrix.
*  @param[in]
*  copyValues      \ref HIPSPARSE_ACTION_SYMBOLIC or \ref HIPSPARSE_ACTION_NUMERIC.
*  @param[in]
*  idxBase         \ref HIPSPARSE_INDEX_BASE_ZERO or \ref HIPSPARSE_INDEX_BASE_ONE.
*
*  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p m, \p n, \p nnz, \p csrSortedVal, \p csrSortedRowPtr,
*              \p csrSortedColInd, \p cscSortedVal, \p cscSortedRowInd or \p cscSortedColPtr pointer is invalid.
*  \retval     HIPSPARSE_STATUS_ARCH_MISMATCH the device is not supported.
*  \retval     HIPSPARSE_STATUS_INTERNAL_ERROR an internal error occurred.
*
*  \par Example
*  \code{.c}
*    // hipSPARSE handle
*    hipsparseHandle_t handle;
*    hipsparseCreate(&handle);
*
*    // Sparse matrix in CSR format
*    //     1 2 0 3 0
*    // A = 0 4 5 0 0
*    //     6 0 0 7 8
*    int hcsrRowPtr[4] = {0, 3, 5, 8};
*    int hcsrColInd[8] = {0, 1, 3, 1, 2, 0, 3, 4};
*    float hcsrVal[8]   = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f}; 
*
*    int m         = 3;
*    int n         = 5;
*    int nnz       = 8;
*    hipsparseIndexBase_t base = HIPSPARSE_INDEX_BASE_ZERO;
*    hipsparseAction_t action = HIPSPARSE_ACTION_NUMERIC;
*
*    int* dcsrRowPtr = nullptr;
*    int* dcsrColInd = nullptr;
*    float* dcsrVal = nullptr;
*    hipMalloc((void**)&dcsrRowPtr, sizeof(int) * (m + 1));
*    hipMalloc((void**)&dcsrColInd, sizeof(int) * nnz);
*    hipMalloc((void**)&dcsrVal, sizeof(float) * nnz);
*
*    hipMemcpy(dcsrRowPtr, hcsrRowPtr, sizeof(int) * (m + 1), hipMemcpyHostToDevice);
*    hipMemcpy(dcsrColInd, hcsrColInd, sizeof(int) * nnz, hipMemcpyHostToDevice);
*    hipMemcpy(dcsrVal, hcsrVal, sizeof(float) * nnz, hipMemcpyHostToDevice);
*
*    int* dcscRowInd = nullptr;
*    int* dcscColPtr = nullptr;
*    float* dcsc_val   = nullptr;
*    hipMalloc((void**)&dcscRowInd, sizeof(int) * nnz);
*    hipMalloc((void**)&dcscColPtr, sizeof(int) * (n + 1));
*    hipMalloc((void**)&dcsc_val, sizeof(float) * nnz);
*
*    hipsparseScsr2csc(handle, m, n, nnz, dcsrVal, dcsrRowPtr, dcsrColInd, dcsc_val, dcscRowInd, dcscColPtr, action, base);
*
*    hipFree(dcsrRowPtr);
*    hipFree(dcsrColInd);
*    hipFree(dcsrVal);
*    
*    hipFree(dcscRowInd);
*    hipFree(dcscColPtr);
*    hipFree(dcsc_val);
*
*    hipsparseDestroy(handle);
*  \endcode
*/
/**@{*/
DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseScsr2csc(hipsparseHandle_t    handle,
                                    int                  m,
                                    int                  n,
                                    int                  nnz,
                                    const float*         csrSortedVal,
                                    const int*           csrSortedRowPtr,
                                    const int*           csrSortedColInd,
                                    float*               cscSortedVal,
                                    int*                 cscSortedRowInd,
                                    int*                 cscSortedColPtr,
                                    hipsparseAction_t    copyValues,
                                    hipsparseIndexBase_t idxBase);
DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDcsr2csc(hipsparseHandle_t    handle,
                                    int                  m,
                                    int                  n,
                                    int                  nnz,
                                    const double*        csrSortedVal,
                                    const int*           csrSortedRowPtr,
                                    const int*           csrSortedColInd,
                                    double*              cscSortedVal,
                                    int*                 cscSortedRowInd,
                                    int*                 cscSortedColPtr,
                                    hipsparseAction_t    copyValues,
                                    hipsparseIndexBase_t idxBase);
DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCcsr2csc(hipsparseHandle_t    handle,
                                    int                  m,
                                    int                  n,
                                    int                  nnz,
                                    const hipComplex*    csrSortedVal,
                                    const int*           csrSortedRowPtr,
                                    const int*           csrSortedColInd,
                                    hipComplex*          cscSortedVal,
                                    int*                 cscSortedRowInd,
                                    int*                 cscSortedColPtr,
                                    hipsparseAction_t    copyValues,
                                    hipsparseIndexBase_t idxBase);
DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZcsr2csc(hipsparseHandle_t       handle,
                                    int                     m,
                                    int                     n,
                                    int                     nnz,
                                    const hipDoubleComplex* csrSortedVal,
                                    const int*              csrSortedRowPtr,
                                    const int*              csrSortedColInd,
                                    hipDoubleComplex*       cscSortedVal,
                                    int*                    cscSortedRowInd,
                                    int*                    cscSortedColPtr,
                                    hipsparseAction_t       copyValues,
                                    hipsparseIndexBase_t    idxBase);
/**@}*/
#endif

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

#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 10010)
/*! \ingroup conv_module
*  \brief This function computes the size of the user allocated temporary storage buffer used
*  when converting a sparse CSR matrix into a sparse CSC matrix.
*
*  \details
*  \p hipsparseCsr2cscEx2_bufferSize calculates the required user allocated temporary buffer needed 
*  by \ref hipsparseCsr2cscEx2 to convert a CSR matrix into a CSC matrix. \ref hipsparseCsr2cscEx2
*  can also be used to convert a CSC matrix into a CSR matrix. \p copyValues decides
*  whether \p cscVal is being filled during conversion (\ref HIPSPARSE_ACTION_NUMERIC)
*  or not (\ref HIPSPARSE_ACTION_SYMBOLIC).
*
*  \note
*  The resulting matrix can also be seen as the transpose of the input matrix.
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*
*  @param[in]
*  handle             handle to the hipsparse library context queue.
*  @param[in]
*  m                  number of rows of the sparse CSR matrix.
*  @param[in]
*  n                  number of columns of the sparse CSR matrix.
*  @param[in]
*  nnz                number of non-zero entries of the sparse CSR matrix.
*  @param[in]
*  csrVal             array of \p nnz elements of the sparse CSR matrix.
*  @param[in]
*  csrRowPtr          array of \p m+1 elements that point to the start of every row of the
*                     sparse CSR matrix.
*  @param[in]
*  csrColInd          array of \p nnz elements containing the column indices of the sparse
*                     CSR matrix.
*  @param[in]
*  cscVal             array of \p nnz elements of the sparse CSC matrix.
*  @param[in]
*  cscColPtr          array of \p n+1 elements that point to the start of every column of the
*                     sparse CSC matrix.
*  @param[in]
*  cscRowInd          array of \p nnz elements containing the row indices of the sparse
*                     CSC matrix.
*  @param[in]
*  valType            The data type of the values arrays \p csrVal and \p cscVal. Can be HIP_R_32F, 
*                     HIP_R_64F, HIP_C_32F or HIP_C_64F
*  @param[in]
*  copyValues         \ref HIPSPARSE_ACTION_SYMBOLIC or \ref HIPSPARSE_ACTION_NUMERIC.
*  @param[in]
*  idxBase            \ref HIPSPARSE_INDEX_BASE_ZERO or \ref HIPSPARSE_INDEX_BASE_ONE.
*  @param[in]
*  alg                HIPSPARSE_CSR2CSC_ALG_DEFAULT, HIPSPARSE_CSR2CSC_ALG1 or HIPSPARSE_CSR2CSC_ALG2.
*  @param[out]
*  pBufferSizeInBytes number of bytes of the temporary storage buffer required by
*                     hipsparseCsr2cscEx2().
*
*  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p m, \p n, \p nnz, \p csrRowPtr, \p csrColInd or
*              \p pBufferSizeInBytes pointer is invalid.
*  \retval     HIPSPARSE_STATUS_INTERNAL_ERROR an internal error occurred.
*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCsr2cscEx2_bufferSize(hipsparseHandle_t     handle,
                                                 int                   m,
                                                 int                   n,
                                                 int                   nnz,
                                                 const void*           csrVal,
                                                 const int*            csrRowPtr,
                                                 const int*            csrColInd,
                                                 void*                 cscVal,
                                                 int*                  cscColPtr,
                                                 int*                  cscRowInd,
                                                 hipDataType           valType,
                                                 hipsparseAction_t     copyValues,
                                                 hipsparseIndexBase_t  idxBase,
                                                 hipsparseCsr2CscAlg_t alg,
                                                 size_t*               pBufferSizeInBytes);
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 10010)
/*! \ingroup conv_module
*  \brief Convert a sparse CSR matrix into a sparse CSC matrix
*
*  \details
*  \p hipsparseCsr2cscEx2 converts a CSR matrix into a CSC matrix. \p hipsparseCsr2cscEx2
*  can also be used to convert a CSC matrix into a CSR matrix. \p copyValues decides
*  whether \p cscVal is being filled during conversion (\ref HIPSPARSE_ACTION_NUMERIC)
*  or not (\ref HIPSPARSE_ACTION_SYMBOLIC).
*
*  \note
*  The resulting matrix can also be seen as the transpose of the input matrix.
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*
*  @param[in]
*  handle      handle to the hipsparse library context queue.
*  @param[in]
*  m           number of rows of the sparse CSR matrix.
*  @param[in]
*  n           number of columns of the sparse CSR matrix.
*  @param[in]
*  nnz         number of non-zero entries of the sparse CSR matrix.
*  @param[in]
*  csrVal      array of \p nnz elements of the sparse CSR matrix.
*  @param[in]
*  csrRowPtr   array of \p m+1 elements that point to the start of every row of the
*              sparse CSR matrix.
*  @param[in]
*  csrColInd   array of \p nnz elements containing the column indices of the sparse
*              CSR matrix.
*  @param[in]
*  cscVal      array of \p nnz elements of the sparse CSC matrix.
*  @param[in]
*  cscColPtr   array of \p n+1 elements that point to the start of every column of the
*              sparse CSC matrix.
*  @param[in]
*  cscRowInd   array of \p nnz elements containing the row indices of the sparse
*              CSC matrix.
*  @param[in]
*  valType     The data type of the values arrays \p csrVal and \p cscVal. Can be HIP_R_32F, 
*              HIP_R_64F, HIP_C_32F or HIP_C_64F
*  @param[in]
*  copyValues  \ref HIPSPARSE_ACTION_SYMBOLIC or \ref HIPSPARSE_ACTION_NUMERIC.
*  @param[in]
*  idxBase     \ref HIPSPARSE_INDEX_BASE_ZERO or \ref HIPSPARSE_INDEX_BASE_ONE.
*  @param[in]
*  alg         HIPSPARSE_CSR2CSC_ALG_DEFAULT, HIPSPARSE_CSR2CSC_ALG1 or HIPSPARSE_CSR2CSC_ALG2.
*  @param[in]
*  buffer      temporary storage buffer allocated by the user, size is returned by
*              hipsparseCsr2cscEx2_bufferSize().
*
*  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p m, \p n, \p nnz, \p csrRowPtr, \p csrColInd or
*              \p pBufferSizeInBytes pointer is invalid.
*  \retval     HIPSPARSE_STATUS_INTERNAL_ERROR an internal error occurred.
*
*  \par Example
*  \code{.c}
*    // hipSPARSE handle
*    hipsparseHandle_t handle;
*    hipsparseCreate(&handle);
*
*    // Sparse matrix in CSR format
*    //     1 2 0 3 0
*    // A = 0 4 5 0 0
*    //     6 0 0 7 8
*    int hcsrRowPtr[4] = {0, 3, 5, 8};
*    int hcsrColInd[8] = {0, 1, 3, 1, 2, 0, 3, 4};
*    float hcsrVal[8]   = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f}; 
*
*    int m         = 3;
*    int n         = 5;
*    int nnz       = 8;
*    hipsparseIndexBase_t base = HIPSPARSE_INDEX_BASE_ZERO;
*    hipsparseAction_t action = HIPSPARSE_ACTION_NUMERIC;
*    hipsparseCsr2CscAlg_t alg = HIPSPARSE_CSR2CSC_ALG_DEFAULT;
*
*    int* dcsrRowPtr = nullptr;
*    int* dcsrColInd = nullptr;
*    float* dcsrVal = nullptr;
*    hipMalloc((void**)&dcsrRowPtr, sizeof(int) * (m + 1));
*    hipMalloc((void**)&dcsrColInd, sizeof(int) * nnz);
*    hipMalloc((void**)&dcsrVal, sizeof(float) * nnz);
*
*    hipMemcpy(dcsrRowPtr, hcsrRowPtr, sizeof(int) * (m + 1), hipMemcpyHostToDevice);
*    hipMemcpy(dcsrColInd, hcsrColInd, sizeof(int) * nnz, hipMemcpyHostToDevice);
*    hipMemcpy(dcsrVal, hcsrVal, sizeof(float) * nnz, hipMemcpyHostToDevice);
*
*    int* dcscRowInd = nullptr;
*    int* dcscColPtr = nullptr;
*    float* dcsc_val   = nullptr;
*    hipMalloc((void**)&dcscRowInd, sizeof(int) * nnz);
*    hipMalloc((void**)&dcscColPtr, sizeof(int) * (n + 1));
*    hipMalloc((void**)&dcsc_val, sizeof(float) * nnz);
*
*    size_t bufferSize;
*    hipsparseCsr2cscEx2_bufferSize(handle, 
*                                   m, 
*                                   n, 
*                                   nnz, 
*                                   dcsrVal, 
*                                   dcsrRowPtr, 
*                                   dcsrColInd, 
*                                   dcsc_val, 
*                                   dcscColPtr, 
*                                   dcscRowInd,
*                                   HIP_R_32F, 
*                                   action, 
*                                   base, 
*                                   alg, 
*                                   &bufferSize);
*
*    void* dbuffer = nullptr;
*    hipMalloc((void**)&dbuffer, bufferSize);
*
*    hipsparseCsr2cscEx2(handle, 
*                        m, 
*                        n, 
*                        nnz, 
*                        dcsrVal, 
*                        dcsrRowPtr, 
*                        dcsrColInd, 
*                        dcsc_val, 
*                        dcscColPtr, 
*                        dcscRowInd, 
*                        HIP_R_32F, 
*                        action, 
*                        base, 
*                        alg, 
*                        dbuffer);
*
*    hipFree(dcsrRowPtr);
*    hipFree(dcsrColInd);
*    hipFree(dcsrVal);
*    
*    hipFree(dcscRowInd);
*    hipFree(dcscColPtr);
*    hipFree(dcsc_val);
*
*    hipFree(dbuffer);
*
*    hipsparseDestroy(handle);
*  \endcode
*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCsr2cscEx2(hipsparseHandle_t     handle,
                                      int                   m,
                                      int                   n,
                                      int                   nnz,
                                      const void*           csrVal,
                                      const int*            csrRowPtr,
                                      const int*            csrColInd,
                                      void*                 cscVal,
                                      int*                  cscColPtr,
                                      int*                  cscRowInd,
                                      hipDataType           valType,
                                      hipsparseAction_t     copyValues,
                                      hipsparseIndexBase_t  idxBase,
                                      hipsparseCsr2CscAlg_t alg,
                                      void*                 buffer);
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 11000)
/*! \ingroup conv_module
*  \brief Convert a sparse CSR matrix into a sparse HYB matrix
*
*  \details
*  \p hipsparseXcsr2hyb converts a CSR matrix into a HYB matrix. It is assumed
*  that \p hyb has been initialized with \ref hipsparseCreateHybMat().
*
*  \note
*  This function requires a significant amount of storage for the HYB matrix,
*  depending on the matrix structure.
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*
*  @param[in]
*  handle            handle to the hipsparse library context queue.
*  @param[in]
*  m                 number of rows of the sparse CSR matrix.
*  @param[in]
*  n                 number of columns of the sparse CSR matrix.
*  @param[in]
*  descrA            descriptor of the sparse CSR matrix. Currently, only
*                    \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*  @param[in]
*  csrSortedValA     array containing the values of the sparse CSR matrix.
*  @param[in]
*  csrSortedRowPtrA  array of \p m+1 elements that point to the start of every row of the
*                    sparse CSR matrix.
*  @param[in]
*  csrSortedColIndA  array containing the column indices of the sparse CSR matrix.
*  @param[out]
*  hybA              sparse matrix in HYB format.
*  @param[in]
*  userEllWidth      width of the ELL part of the HYB matrix (only required if
*                    \p partitionType == \ref HIPSPARSE_HYB_PARTITION_USER).
*  @param[in]
*  partitionType     \ref HIPSPARSE_HYB_PARTITION_AUTO (recommended),
*                    \ref HIPSPARSE_HYB_PARTITION_USER or
*                    \ref HIPSPARSE_HYB_PARTITION_MAX.
*
*  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p m, \p n, \p userEllWidth, \p partitionType, 
*              \p descrA, \p hybA, \p csrSortedValA, \p csrSortedRowPtrA or \p csrSortedColIndA pointer is invalid.
*  \retval     HIPSPARSE_STATUS_ALLOC_FAILED the buffer for the HYB matrix could not be
*              allocated.
*  \retval     HIPSPARSE_STATUS_INTERNAL_ERROR an internal error occurred.
*  \retval     HIPSPARSE_STATUS_NOT_SUPPORTED
*              \ref hipsparseMatrixType_t != \ref HIPSPARSE_MATRIX_TYPE_GENERAL.
*
*  \par Example
*  \code{.c}
*    // hipSPARSE handle
*    hipsparseHandle_t handle;
*    hipsparseCreate(&handle);
*
*    hipsparseMatDescr_t descr;
*    hipsparseCreateMatDescr(&descr);
*
*    // Sparse matrix in CSR format
*    //     1 2 0 3 0
*    // A = 0 4 5 0 0
*    //     6 0 0 7 8
*    int hcsrRowPtr[4] = {0, 3, 5, 8};
*    int hcsrColInd[8] = {0, 1, 3, 1, 2, 0, 3, 4};
*    float hcsrVal[8]   = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f}; 
*
*    int m         = 3;
*    int n         = 5;
*    int nnz       = 8;
*    int userEllWidth = 2;
*    hipsparseHybPartition_t partitionType = HIPSPARSE_HYB_PARTITION_AUTO;
*
*    int* dcsrRowPtr = nullptr;
*    int* dcsrColInd = nullptr;
*    float* dcsrVal = nullptr;
*    hipMalloc((void**)&dcsrRowPtr, sizeof(int) * (m + 1));
*    hipMalloc((void**)&dcsrColInd, sizeof(int) * nnz);
*    hipMalloc((void**)&dcsrVal, sizeof(float) * nnz);
*
*    hipMemcpy(dcsrRowPtr, hcsrRowPtr, sizeof(int) * (m + 1), hipMemcpyHostToDevice);
*    hipMemcpy(dcsrColInd, hcsrColInd, sizeof(int) * nnz, hipMemcpyHostToDevice);
*    hipMemcpy(dcsrVal, hcsrVal, sizeof(float) * nnz, hipMemcpyHostToDevice);
*
*    hipsparseHybMat_t hyb;
*    hipsparseCreateHybMat(&hyb);
*
*    hipsparseScsr2hyb(handle, m, n, descr, dcsrVal, dcsrRowPtr, dcsrColInd, hyb, userEllWidth, partitionType);
*
*    hipFree(dcsrRowPtr);
*    hipFree(dcsrColInd);
*    hipFree(dcsrVal);
*
*    hipsparseDestroyHybMat(hyb);    
*    hipsparseDestroyMatDescr(descr);
*
*    hipsparseDestroy(handle);
*  \endcode
*/
/**@{*/
DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseScsr2hyb(hipsparseHandle_t         handle,
                                    int                       m,
                                    int                       n,
                                    const hipsparseMatDescr_t descrA,
                                    const float*              csrSortedValA,
                                    const int*                csrSortedRowPtrA,
                                    const int*                csrSortedColIndA,
                                    hipsparseHybMat_t         hybA,
                                    int                       userEllWidth,
                                    hipsparseHybPartition_t   partitionType);
DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDcsr2hyb(hipsparseHandle_t         handle,
                                    int                       m,
                                    int                       n,
                                    const hipsparseMatDescr_t descrA,
                                    const double*             csrSortedValA,
                                    const int*                csrSortedRowPtrA,
                                    const int*                csrSortedColIndA,
                                    hipsparseHybMat_t         hybA,
                                    int                       userEllWidth,
                                    hipsparseHybPartition_t   partitionType);
DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCcsr2hyb(hipsparseHandle_t         handle,
                                    int                       m,
                                    int                       n,
                                    const hipsparseMatDescr_t descrA,
                                    const hipComplex*         csrSortedValA,
                                    const int*                csrSortedRowPtrA,
                                    const int*                csrSortedColIndA,
                                    hipsparseHybMat_t         hybA,
                                    int                       userEllWidth,
                                    hipsparseHybPartition_t   partitionType);
DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZcsr2hyb(hipsparseHandle_t         handle,
                                    int                       m,
                                    int                       n,
                                    const hipsparseMatDescr_t descrA,
                                    const hipDoubleComplex*   csrSortedValA,
                                    const int*                csrSortedRowPtrA,
                                    const int*                csrSortedColIndA,
                                    hipsparseHybMat_t         hybA,
                                    int                       userEllWidth,
                                    hipsparseHybPartition_t   partitionType);
/**@}*/
#endif

/*! \ingroup conv_module
*  \brief Convert a sparse GEBSR matrix into a sparse GEBSC matrix
*
*  \details
*  \p hipsparseXgebsr2gebsc_bufferSize returns the size of the temporary storage buffer
*  required by \ref hipsparseSgebsr2gebsc "hipsparseXgebsr2gebsc()" and is the first step 
*  in converting a sparse matrix in GEBSR format to a sparse matrix in GEBSC format. Once 
*  the size of the temporary storage buffer has been determined, it must be allocated by the user.
*
*  See hipsparseSgebsr2gebsc() for a complete code example.
*
*  @param[in]
*  handle             handle to the hipsparse library context queue.
*  @param[in]
*  mb                 number of rows of the sparse GEneral BSR matrix.
*  @param[in]
*  nb                 number of columns of the sparse GEneral BSR matrix.
*  @param[in]
*  nnzb               number of non-zero entries of the sparse GEneral BSR matrix.
*  @param[in]
*  bsrVal             array of \p nnzb*rowBlockDim*colBlockDim containing the values of the sparse GEneral 
*                     BSR matrix.
*  @param[in]
*  bsrRowPtr          array of \p mb+1 elements that point to the start of every row of the
*                     sparse GEneral BSR matrix.
*  @param[in]
*  bsrColInd          array of \p nnzb elements containing the column indices of the sparse
*                     GEneral BSR matrix.
*  @param[in]
*  rowBlockDim        row size of the blocks in the sparse general BSR matrix.
*  @param[in]
*  colBlockDim        col size of the blocks in the sparse general BSR matrix.
*  @param[out]
*  pBufferSizeInBytes number of bytes of the temporary storage buffer required by
*                     hipsparseSgebsr2gebsc(), hipsparseDgebsr2gebsc(), hipsparseCgebsr2gebsc() and
*                     hipsparseZgebsr2gebsc().
*
*  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p mb, \p nb, \p nnzb, \p bsrRowPtr, \p bsrColInd 
*              or \p pBufferSizeInBytes pointer is invalid.
*  \retval     rocsparse_status_internal_error an internal error occurred.
*/
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSgebsr2gebsc_bufferSize(hipsparseHandle_t handle,
                                                   int               mb,
                                                   int               nb,
                                                   int               nnzb,
                                                   const float*      bsrVal,
                                                   const int*        bsrRowPtr,
                                                   const int*        bsrColInd,
                                                   int               rowBlockDim,
                                                   int               colBlockDim,
                                                   size_t*           pBufferSizeInBytes);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDgebsr2gebsc_bufferSize(hipsparseHandle_t handle,
                                                   int               mb,
                                                   int               nb,
                                                   int               nnzb,
                                                   const double*     bsrVal,
                                                   const int*        bsrRowPtr,
                                                   const int*        bsrColInd,
                                                   int               rowBlockDim,
                                                   int               colBlockDim,
                                                   size_t*           pBufferSizeInBytes);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCgebsr2gebsc_bufferSize(hipsparseHandle_t handle,
                                                   int               mb,
                                                   int               nb,
                                                   int               nnzb,
                                                   const hipComplex* bsrVal,
                                                   const int*        bsrRowPtr,
                                                   const int*        bsrColInd,
                                                   int               rowBlockDim,
                                                   int               colBlockDim,
                                                   size_t*           pBufferSizeInBytes);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZgebsr2gebsc_bufferSize(hipsparseHandle_t       handle,
                                                   int                     mb,
                                                   int                     nb,
                                                   int                     nnzb,
                                                   const hipDoubleComplex* bsrVal,
                                                   const int*              bsrRowPtr,
                                                   const int*              bsrColInd,
                                                   int                     rowBlockDim,
                                                   int                     colBlockDim,
                                                   size_t*                 pBufferSizeInBytes);
/**@}*/

/*! \ingroup conv_module
*  \brief Convert a sparse GEBSR matrix into a sparse GEBSC matrix
*
*  \details
*  \p hipsparseXgebsr2gebsc converts a GEBSR matrix into a GEBSC matrix. \p hipsparseXgebsr2gebsc
*  can also be used to convert a GEBSC matrix into a GEBSR matrix. \p copyValues decides
*  whether \p bscVal is being filled during conversion (\ref HIPSPARSE_ACTION_NUMERIC)
*  or not (\ref HIPSPARSE_ACTION_SYMBOLIC).
*
*  \p hipsparseXgebsr2gebsc requires extra temporary storage buffer that has to be allocated
*  by the user. Storage buffer size can be determined by \ref hipsparseSgebsr2gebsc_bufferSize 
*  "hipsparseXgebsr2gebsc_bufferSize()".
*
*  For example, given the GEBSR matrix:
*  \f[
*   \left[ 
*    \begin{array}{c | c} 
*      \begin{array}{c c} 
*       1 & 2 \\ 
*       3 & 4 \\
*       6 & 0 
*      \end{array} & 
*      \begin{array}{c c} 
*       0 & 2 \\ 
*       0 & 0 \\
*       3 & 4 
*      \end{array} \\ 
*    \hline 
*      \begin{array}{c c} 
*       5 & 0 \\ 
*       1 & 2 \\
*       3 & 4 
*      \end{array} & 
*      \begin{array}{c c} 
*       6 & 7 \\ 
*       3 & 4 \\
*       3 & 4 
*      \end{array} \\ 
*   \end{array} 
*  \right] 
*  \f]
*
*  represented with the arrays:
*  \f[
*    \begin{align}
*    \text{bsrRowPtr} &= \begin{bmatrix} 0 & 2 & 4 \end{bmatrix} \\
*    \text{bsrColInd} &= \begin{bmatrix} 0 & 1 & 0 & 1 \end{bmatrix} \\
*    \text{bsrVal} &= \begin{bmatrix} 1 & 2 & 3 & 4 & 6 & 0 & 0 & 2 & 0 & 0 & 3 & 4 & 5 & 0 & 1 & 2 & 3 & 4 & 6 & 7 & 3 & 4 & 3 & 4 \end{bmatrix}
*    \end{align}
*  \f]
*
*  this function converts the matrix to GEBSC format:
*  \f[
*    \begin{align}
*    \text{bscRowInd} &= \begin{bmatrix} 0 & 1 & 0 & 1 \end{bmatrix} \\
*    \text{bscColPtr} &= \begin{bmatrix} 0 & 2 & 4 \end{bmatrix} \\
*    \text{bscVal} &= \begin{bmatrix} 1 & 2 & 3 & 4 & 6 & 0 & 5 & 0 & 1 & 2 & 3 & 4 & 0 & 2 & 0 & 0 & 3 & 4 & 6 & 7 & 3 & 4 & 3 & 4 \end{bmatrix}
*    \end{align}
*  \f]
*
*  The GEBSC arrays, \p bscRowInd, \p bscColPtr, and \p bscVal must be allocated by the user prior 
*  to calling \p hipsparseXgebsr2gebsc(). The \p bscRowInd array has size \p nnzb, the \p bscColPtr 
*  array has size \p nb+1, and the \p bscVal array has size \p nnzb*rowBlockDim*colBlockDim.
*
*  \note
*  The resulting matrix can also be seen as the transpose of the input matrix.
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*
*  @param[in]
*  handle      handle to the hipsparse library context queue.
*  @param[in]
*  mb          number of rows of the sparse GEneral BSR matrix.
*  @param[in]
*  nb          number of columns of the sparse GEneral BSR matrix.
*  @param[in]
*  nnzb        number of non-zero entries of the sparse GEneral BSR matrix.
*  @param[in]
*  bsrVal      array of \p nnzb * \p rowBlockDim * \p colBlockDim  elements of the sparse GEneral BSR matrix.
*  @param[in]
*  bsrRowPtr   array of \p m+1 elements that point to the start of every row of the
*              sparse GEneral BSR matrix.
*  @param[in]
*  bsrColInd   array of \p nnz elements containing the column indices of the sparse
*              GEneral BSR matrix.
*  @param[in]
*  rowBlockDim row size of the blocks in the sparse general BSR matrix.
*  @param[in]
*  colBlockDim col size of the blocks in the sparse general BSR matrix.
*  @param[out]
*  bscVal      array of \p nnz elements of the sparse BSC matrix.
*  @param[out]
*  bscRowInd   array of \p nnz elements containing the row indices of the sparse BSC
*              matrix.
*  @param[out]
*  bscColPtr   array of \p n+1 elements that point to the start of every column of the
*              sparse BSC matrix.
*  @param[in]
*  copyValues  \ref HIPSPARSE_ACTION_SYMBOLIC or \ref HIPSPARSE_ACTION_NUMERIC.
*  @param[in]
*  idxBase     \ref HIPSPARSE_INDEX_BASE_ZERO or \ref HIPSPARSE_INDEX_BASE_ONE.
*  @param[in]
*  temp_buffer temporary storage buffer allocated by the user, size is returned by
*              \ref hipsparseSgebsr2gebsc_bufferSize "hipsparseXgebsr2gebsc_bufferSize()".
*
*  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p mb, \p nb, \p nnzb, \p bsrVal, 
*              \p bsrRowPtr, \p bsrColInd, \p bscVal, \p bscRowInd, \p bscColPtr or
*              \p temp_buffer pointer is invalid.
*  \retval     HIPSPARSE_STATUS_ARCH_MISMATCH the device is not supported.
*  \retval     HIPSPARSE_STATUS_INTERNAL_ERROR an internal error occurred.
*
*  \par Example
*  \code{.c}
*    // hipSPARSE handle
*    hipsparseHandle_t handle;
*    hipsparseCreate(&handle);
*
*    // Sparse matrix in BSR format
*    //     1 2 | 0 3 | 0 0 
*    //     0 4 | 5 0 | 0 1
*    // A = 6 0 | 0 7 | 8 0
*    //     --------------- 
*    //     0 0 | 3 0 | 2 2
*    //     1 0 | 0 0 | 4 3 
*    //     7 2 | 0 0 | 1 4
*    int hbsrRowPtr[3] = {0, 3, 6};
*    int hbsrColInd[6] = {0, 1, 2, 0, 1, 2};
*    float hbsrVal[36]  = {1.0f, 2.0f, 0.0f, 4.0f, 6.0f, 0.0f, 
*                           0.0f, 3.0f, 5.0f, 0.0f, 0.0f, 7.0f, 
*                           0.0f, 0.0f, 0.0f, 1.0f, 8.0f, 0.0f, 
*                           0.0f, 0.0f, 1.0f, 0.0f, 7.0f, 2.0f,
*                           3.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
*                           2.0f, 2.0f, 4.0f, 3.0f, 1.0f, 4.0f}; 
*
*    int m           = 6;
*    int n           = 6;
*    int rowBlockDim = 3;
*    int colBlockDim = 2;
*    int nnzb        = 6;
*    hipsparseDirection_t dir = HIPSPARSE_DIRECTION_ROW;
*    hipsparseAction_t    action = HIPSPARSE_ACTION_NUMERIC;
*    hipsparseIndexBase_t base = HIPSPARSE_INDEX_BASE_ZERO;
*
*    int mb = (m + rowBlockDim - 1) / rowBlockDim;
*    int nb = (n + colBlockDim - 1) / colBlockDim;
*
*    int* dbsrRowPtr = nullptr;
*    int* dbsrColInd = nullptr;
*    float* dbsrVal = nullptr;
*    hipMalloc((void**)&dbsrRowPtr, sizeof(int) * (mb + 1));
*    hipMalloc((void**)&dbsrColInd, sizeof(int) * nnzb);
*    hipMalloc((void**)&dbsrVal, sizeof(float) * rowBlockDim * colBlockDim * nnzb);
*
*    hipMemcpy(dbsrRowPtr, hbsrRowPtr, sizeof(int) * (mb + 1), hipMemcpyHostToDevice);
*    hipMemcpy(dbsrColInd, hbsrColInd, sizeof(int) * nnzb, hipMemcpyHostToDevice);
*    hipMemcpy(dbsrVal, hbsrVal, sizeof(float) * rowBlockDim * colBlockDim * nnzb, hipMemcpyHostToDevice);
*
*    int* dbscRowInd = nullptr;
*    int* dbscColPtr = nullptr;
*    float* dbscVal = nullptr;
*    hipMalloc((void**)&dbscRowInd, sizeof(int) * nnzb);
*    hipMalloc((void**)&dbscColPtr, sizeof(int) * (nb + 1));
*    hipMalloc((void**)&dbscVal, sizeof(float) * rowBlockDim * colBlockDim * nnzb);
*
*    size_t bufferSize;
*    hipsparseSgebsr2gebsc_bufferSize(handle, 
*                                     mb, 
*                                     nb, 
*                                     nnzb, 
*                                     dbsrVal, 
*                                     dbsrRowPtr, 
*                                     dbsrColInd, 
*                                     rowBlockDim, 
*                                     colBlockDim, 
*                                     &bufferSize);
*
*    void* dbuffer = nullptr;
*    hipMalloc((void**)&dbuffer, bufferSize);
*
*    hipsparseSgebsr2gebsc(handle, 
*                          mb, 
*                          nb, 
*                          nnzb, 
*                          dbsrVal, 
*                          dbsrRowPtr, 
*                          dbsrColInd, 
*                          rowBlockDim, 
*                          colBlockDim, 
*                          dbscVal, 
*                          dbscRowInd, 
*                          dbscColPtr, 
*                          action, 
*                          base, 
*                          dbuffer);
*
*    hipFree(dbsrRowPtr);
*    hipFree(dbsrColInd);
*    hipFree(dbsrVal);
*    
*    hipFree(dbscRowInd);
*    hipFree(dbscColPtr);
*    hipFree(dbscVal);
*
*    hipFree(dbuffer);
*
*    hipsparseDestroy(handle);
*  \endcode
*/
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSgebsr2gebsc(hipsparseHandle_t    handle,
                                        int                  mb,
                                        int                  nb,
                                        int                  nnzb,
                                        const float*         bsrVal,
                                        const int*           bsrRowPtr,
                                        const int*           bsrColInd,
                                        int                  rowBlockDim,
                                        int                  colBlockDim,
                                        float*               bscVal,
                                        int*                 bscRowInd,
                                        int*                 bscColPtr,
                                        hipsparseAction_t    copyValues,
                                        hipsparseIndexBase_t idxBase,
                                        void*                temp_buffer);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDgebsr2gebsc(hipsparseHandle_t    handle,
                                        int                  mb,
                                        int                  nb,
                                        int                  nnzb,
                                        const double*        bsrVal,
                                        const int*           bsrRowPtr,
                                        const int*           bsrColInd,
                                        int                  rowBlockDim,
                                        int                  colBlockDim,
                                        double*              bscVal,
                                        int*                 bscRowInd,
                                        int*                 bscColPtr,
                                        hipsparseAction_t    copyValues,
                                        hipsparseIndexBase_t idxBase,
                                        void*                temp_buffer);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCgebsr2gebsc(hipsparseHandle_t    handle,
                                        int                  mb,
                                        int                  nb,
                                        int                  nnzb,
                                        const hipComplex*    bsrVal,
                                        const int*           bsrRowPtr,
                                        const int*           bsrColInd,
                                        int                  rowBlockDim,
                                        int                  colBlockDim,
                                        hipComplex*          bscVal,
                                        int*                 bscRowInd,
                                        int*                 bscColPtr,
                                        hipsparseAction_t    copyValues,
                                        hipsparseIndexBase_t idxBase,
                                        void*                temp_buffer);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZgebsr2gebsc(hipsparseHandle_t       handle,
                                        int                     mb,
                                        int                     nb,
                                        int                     nnzb,
                                        const hipDoubleComplex* bsrVal,
                                        const int*              bsrRowPtr,
                                        const int*              bsrColInd,
                                        int                     rowBlockDim,
                                        int                     colBlockDim,
                                        hipDoubleComplex*       bscVal,
                                        int*                    bscRowInd,
                                        int*                    bscColPtr,
                                        hipsparseAction_t       copyValues,
                                        hipsparseIndexBase_t    idxBase,
                                        void*                   temp_buffer);
/**@}*/

/*! \ingroup conv_module
*  \brief Convert a sparse CSR matrix into a sparse GEBSR matrix
*
*  \details
*  \p hipsparseXcsr2gebsr_bufferSize returns the size of the temporary buffer that
*  is required by \ref hipsparseXcsr2gebsrNnz and \ref hipsparseScsr2gebsr "hipsparseXcsr2gebsr()". 
*  Once the temporary buffer size has been determined, it must be allocated by the user prior 
*  to calling \ref hipsparseXcsr2gebsrNnz and \ref hipsparseScsr2gebsr "hipsparseXcsr2gebsr()". 
*
*  See hipsparseScsr2gebsr() for complete code example.
*
*  \note
*  The routine does support asynchronous execution if the pointer mode is set to device.
*
*  @param[in]
*  handle             handle to the hipsparse library context queue.
*  @param[in]
*  dir                direction that specified whether to count nonzero elements by \ref HIPSPARSE_DIRECTION_ROW 
*                     or by \ref HIPSPARSE_DIRECTION_COLUMN.
*  @param[in]
*  m                  number of rows of the sparse CSR matrix.
*  @param[in]
*  n                  number of columns of the sparse CSR matrix.
*  @param[in]
*  csr_descr          descriptor of the sparse CSR matrix. Currently, only
*                     \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*  @param[in]
*  csrVal             array of \p nnz elements containing the values of the sparse CSR matrix.
*  @param[in]
*  csrRowPtr          integer array containing \p m+1 elements that point to the start of each row of the CSR matrix
*  @param[in]
*  csrColInd          integer array of the column indices for each non-zero element in the CSR matrix
*  @param[in]
*  rowBlockDim        the row block dimension of the GEneral BSR matrix. Between 1 and \p m
*  @param[in]
*  colBlockDim        the col block dimension of the GEneral BSR matrix. Between 1 and \p n
*  @param[out]
*  pBufferSizeInBytes number of bytes of the temporary storage buffer required by \ref hipsparseXcsr2gebsrNnz() 
*                     and \ref hipsparseScsr2gebsr "hipsparseXcsr2gebsr()".
*
*  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p m, \p n, \p rowBlockDim, \p colBlockDim, \p csrVal, 
*              \p csrRowPtr, \p csrColInd or \p pBufferSizeInBytes pointer is invalid.
*/
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseScsr2gebsr_bufferSize(hipsparseHandle_t         handle,
                                                 hipsparseDirection_t      dir,
                                                 int                       m,
                                                 int                       n,
                                                 const hipsparseMatDescr_t csr_descr,
                                                 const float*              csrVal,
                                                 const int*                csrRowPtr,
                                                 const int*                csrColInd,
                                                 int                       rowBlockDim,
                                                 int                       colBlockDim,
                                                 size_t*                   pBufferSizeInBytes);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDcsr2gebsr_bufferSize(hipsparseHandle_t         handle,
                                                 hipsparseDirection_t      dir,
                                                 int                       m,
                                                 int                       n,
                                                 const hipsparseMatDescr_t csr_descr,
                                                 const double*             csrVal,
                                                 const int*                csrRowPtr,
                                                 const int*                csrColInd,
                                                 int                       rowBlockDim,
                                                 int                       colBlockDim,
                                                 size_t*                   pBufferSizeInBytes);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCcsr2gebsr_bufferSize(hipsparseHandle_t         handle,
                                                 hipsparseDirection_t      dir,
                                                 int                       m,
                                                 int                       n,
                                                 const hipsparseMatDescr_t csr_descr,
                                                 const hipComplex*         csrVal,
                                                 const int*                csrRowPtr,
                                                 const int*                csrColInd,
                                                 int                       rowBlockDim,
                                                 int                       colBlockDim,
                                                 size_t*                   pBufferSizeInBytes);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZcsr2gebsr_bufferSize(hipsparseHandle_t         handle,
                                                 hipsparseDirection_t      dir,
                                                 int                       m,
                                                 int                       n,
                                                 const hipsparseMatDescr_t csr_descr,
                                                 const hipDoubleComplex*   csrVal,
                                                 const int*                csrRowPtr,
                                                 const int*                csrColInd,
                                                 int                       rowBlockDim,
                                                 int                       colBlockDim,
                                                 size_t*                   pBufferSizeInBytes);
/**@}*/

/*! \ingroup conv_module
*  \brief
*  This function computes the number of nonzero block columns per row and the total number of nonzero blocks in a sparse
*  GEBSR matrix given a sparse CSR matrix as input.
*
*  \details
*  This is the second step in conveting a CSR matrix to a GEBSR matrix. The user must first call 
*  \ref hipsparseScsr2gebsr_bufferSize "hipsparseXcsr2gebsr_bufferSize()" to determine the size of 
*  the required temporary storage buffer. The user then allocates this buffer as well as the 
*  \p bsrRowPtr array ( size \p mb+1 ) and passes both to \p hipsparseXcsr2gebsrNnz(). This second 
*  step then computes the number of nonzero block columns per row and the total number of nonzero blocks. 
*
*  In general, when converting a CSR matrix of size \p m x \p n to a GEBSR matrix, the resulting GEBSR matrix will have size 
*  \p mb x \p nb where \p mb and \p nb equal:
*  \f[
*    \begin{align}
*    \text{mb} &= \text{(m - 1) / rowBlockDim + 1} \\
*    \text{nb} &= \text{(n - 1) / colBlockDim + 1}
*    \end{align}
*  \f]
*
*  For example given a matrix:
*  \f[
*    \begin{bmatrix}
*    1 & 0 & 0 & 2 & 4 & 0 \\
*    3 & 4 & 0 & 0 & 5 & 1 \\
*    5 & 0 & 6 & 7 & 6 & 2
*    \end{bmatrix}
*  \f]
*
*  represented in CSR format with the arrays:
*  \f[
*    \begin{align}
*    \text{csrRowPtr} &= \begin{bmatrix} 0 & 3 & 7 & 12 \end{bmatrix} \\
*    \text{csrColInd} &= \begin{bmatrix} 0 & 3 & 4 & 0 & 1 & 4 & 5 & 0 & 2 & 3 & 4 & 5 \end{bmatrix} \\
*    \text{csrVal} &= \begin{bmatrix} 1 & 2 & 4 & 3 & 4 & 5 & 1 & 5 & 6 & 7 & 6 & 2 \end{bmatrix}
*    \end{align}
*  \f]
*
*  the \p bsrRowPtr array and total nonzero block count will be filled with:
*  \f[
*    \begin{align}
*    \text{bsrRowPtr} &= \begin{bmatrix} 0 & 3 \end{bmatrix} \\
*    \text{*bsrNnzDevhost} &= 3
*    \end{align}
*  \f]
*
*  after calling \p hipsparseXcsr2gebsrNnz with \p rowBlockDim=3 and \p colBlockDim=2.
*
*  \note
*  As indicated, \p bsrNnzDevhost can point either to host or device memory. This is controlled 
*  by setting the pointer mode. See \ref hipsparseSetPointerMode().
*
*  It may be the case that \p rowBlockDim does not divide evenly into \p m and/or that \p colBlockDim does not divide 
*  evenly into \p n. In these cases, the CSR matrix is expanded in size in order to fit full GEBSR blocks. For example, 
*  using the original CSR matrix but this time with \p rowBlockDim=2 and \p colBlockDim=3, the function 
*  \p hipsparseXcsr2gebsrNnz computes the GEBSR row pointer array and total number of non-zero blocks for the GEBSR matrix:
*
*  \f[
*   \left[ 
*    \begin{array}{c | c} 
*      \begin{array}{c c c} 
*       1 & 0 & 0 \\ 
*       3 & 4 & 0
*      \end{array} & 
*      \begin{array}{c c c} 
*       2 & 4 & 0 \\ 
*       0 & 5 & 1
*      \end{array} \\
*    \hline 
*      \begin{array}{c c c} 
*       5 & 0 & 6 \\ 
*       0 & 0 & 0
*      \end{array} &
*      \begin{array}{c c c} 
*       7 & 6 & 2 \\ 
*       0 & 0 & 0
*      \end{array}
*   \end{array} 
*  \right] 
*  \f]
*
*  See hipsparseScsr2gebsr() for full code example.
*
*  @param[in]
*  handle        handle to the hipsparse library context queue.
*  @param[in]
*  dir           direction that specified whether to count nonzero elements by \ref HIPSPARSE_DIRECTION_ROW or by
*                \ref HIPSPARSE_DIRECTION_COLUMN.
*  @param[in]
*  m             number of rows of the sparse CSR matrix.
*  @param[in]
*  n             number of columns of the sparse CSR matrix.
*  @param[in]
*  csr_descr     descriptor of the sparse CSR matrix. Currently, only
*                \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*  @param[in]
*  csrRowPtr     integer array containing \p m+1 elements that point to the start of each row of the CSR matrix
*  @param[in]
*  csrColInd     integer array of the column indices for each non-zero element in the CSR matrix
*  @param[in]
*  bsr_descr     descriptor of the sparse GEneral BSR matrix. Currently, only
*                \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*  @param[out]
*  bsrRowPtr     integer array containing \p mb+1 elements that point to the start of each block row of the General BSR matrix
*
*  @param[in]
*  rowBlockDim   the row block dimension of the GEneral BSR matrix. Between \f$1\f$ and \f$\min(m, n)\f$
*
*  @param[in]
*  colBlockDim   the col block dimension of the GEneral BSR matrix. Between \f$1\f$ and \f$\min(m, n)\f$
*
*  @param[out]
*  bsrNnzDevhost total number of nonzero elements in device or host memory.
*
*  @param[in]
*  pbuffer       buffer allocated by the user whose size is determined by calling \ref hipsparseScsr2gebsr_bufferSize 
*                "hipsparseXcsr2gebsr_bufferSize()".
*
*  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p m, \p n, \p rowBlockDim, \p colBlockDim, \p csrRowPtr, 
*              \p csrColInd, \p bsrRowPtr or \p bsrNnzDevhost pointer is invalid.
*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseXcsr2gebsrNnz(hipsparseHandle_t         handle,
                                         hipsparseDirection_t      dir,
                                         int                       m,
                                         int                       n,
                                         const hipsparseMatDescr_t csr_descr,
                                         const int*                csrRowPtr,
                                         const int*                csrColInd,
                                         const hipsparseMatDescr_t bsr_descr,
                                         int*                      bsrRowPtr,
                                         int                       rowBlockDim,
                                         int                       colBlockDim,
                                         int*                      bsrNnzDevhost,
                                         void*                     pbuffer);

/*! \ingroup conv_module
*  \brief Convert a sparse CSR matrix into a sparse GEBSR matrix
*
*  \details
*  \p hipsparseXcsr2gebsr converts a CSR matrix into a GEBSR matrix. It is assumed,
*  that \p bsrVal, \p bsrColInd and \p bsrRowPtr are allocated. Allocation size
*  for \p bsrRowPtr is computed as \p mb+1 where \p mb is the number of block rows in
*  the GEBSR matrix. The number of nonzero blocks in the resulting GEBSR matrix 
*  is computed using \ref hipsparseXcsr2gebsrNnz which also fills in \p bsrRowPtr.
*
*  In more detail, \p hipsparseXcsr2gebsr is the third and final step on the conversion from CSR to GEBSR.
*  The user first determines the size of the required user allocated temporary storage buffer using 
*  \ref hipsparseScsr2gebsr_bufferSize "hipsparseXcsr2gebsr_bufferSize()". The user then allocates this buffer 
*  as well as the row pointer array \p bsrRowPtr with size \p mb+1, where \p mb is the number of block rows 
*  in the GEBSR matrix and \p nb is the number of block columns in GEBSR matrix:
*
*  \f[
*    \begin{align}
*    \text{mb} &= \text{(m - 1) / rowBlockDim + 1} \\
*    \text{nb} &= \text{(n - 1) / colBlockDim + 1}
*    \end{align}
*  \f]
*
*  Both the temporary storage buffer and the GEBSR row pointer array are then passed to \ref hipsparseXcsr2gebsrNnz 
*  which fills the GEBSR row pointer array \p bsrRowPtr and also computes the number of nonzero blocks, 
*  \p bsrNnzDevhost, that will exist in the GEBSR matrix. The user then allocates both the GEBSR column indices array 
*  \p bsrColInd with size \p bsrNnzDevhost as well as the GEBSR values array \p bsrVal with size 
*  \p bsrNnzDevhost*rowBlockDim*colBlockDim. Finally, with all arrays allocated, the conversion is completed by calling
*  \p hipsparseXcsr2gebsr.
*
*  For example, assuming the matrix:
*  \f[
*    \begin{bmatrix}
*    1 & 0 & 0 & 2 & 4 & 0 \\
*    3 & 4 & 0 & 0 & 5 & 1 \\
*    5 & 0 & 6 & 7 & 6 & 2
*    \end{bmatrix}
*  \f]
*
*  represented in CSR format with the arrays:
*  \f[
*    \begin{align}
*    \text{csrRowPtr} &= \begin{bmatrix} 0 & 3 & 7 & 12 \end{bmatrix} \\
*    \text{csrColInd} &= \begin{bmatrix} 0 & 3 & 4 & 0 & 1 & 4 & 5 & 0 & 2 & 3 & 4 & 5 \end{bmatrix} \\
*    \text{csrVal} &= \begin{bmatrix} 1 & 2 & 4 & 3 & 4 & 5 & 1 & 5 & 6 & 7 & 6 & 2 \end{bmatrix}
*    \end{align}
*  \f]
*
*  then using \p rowBlockDim=3 and \p colBlockDim=2, the final GEBSR matrix is:
*  \f[
*   \left[ 
*    \begin{array}{c | c} 
*      \begin{array}{c c} 
*       1 & 0 \\ 
*       3 & 4 \\
*       3 & 0
*      \end{array} & 
*      \begin{array}{c c} 
*       0 & 2 \\ 
*       0 & 0 \\
*       6 & 7
*      \end{array} & 
*      \begin{array}{c c} 
*       4 & 0 \\ 
*       5 & 1 \\
*       6 & 2
*      \end{array}
*   \end{array} 
*  \right] 
*  \f]
*
*  and is represented with the arrays:
*  \f[
*    \begin{align}
*    \text{bsrRowPtr} &= \begin{bmatrix} 0 & 3 \end{bmatrix} \\
*    \text{bsrColInd} &= \begin{bmatrix} 0 & 1 & 2 \end{bmatrix} \\
*    \text{bsrVal} &= \begin{bmatrix} 1 & 0 & 3 & 4 & 3 & 0 & 0 & 2 & 0 & 0 & 6 & 7 & 4 & 0 & 5 & 1 & 6 & 2 \end{bmatrix}
*    \end{align}
*  \f]
*
*  The above example assumes that the blocks are row ordered. If instead the blocks are column ordered, the \p bsrVal arrays 
*  becomes:
*  \f[
*    \begin{align}
*    \text{bsrVal} &= \begin{bmatrix} 1 & 3 & 3 & 0 & 4 & 0 & 0 & 0 & 6 & 2 & 0 & 7 & 4 & 5 & 6 & 0 & 1 & 2 \end{bmatrix}
*    \end{align}
*  \f]
*
*  The block order direction is determined by \p dir.
*
*  It may be the case that \p rowBlockDim does not divide evenly into \p m and/or that \p colBlockDim does not divide 
*  evenly into \p n. In these cases, the CSR matrix is expanded in size in order to fit full GEBSR blocks. For example, 
*  using the original CSR matrix but this time with \p rowBlockDim=2 and \p colBlockDim=3, the resulting GEBSR matrix
*  would looks like:
*
*  \f[
*   \left[ 
*    \begin{array}{c | c} 
*      \begin{array}{c c c} 
*       1 & 0 & 0 \\ 
*       3 & 4 & 0
*      \end{array} & 
*      \begin{array}{c c c} 
*       2 & 4 & 0 \\ 
*       0 & 5 & 1
*      \end{array} \\
*    \hline 
*      \begin{array}{c c c} 
*       5 & 0 & 6 \\ 
*       0 & 0 & 0
*      \end{array} &
*      \begin{array}{c c c} 
*       7 & 6 & 2 \\ 
*       0 & 0 & 0
*      \end{array}
*   \end{array} 
*  \right] 
*  \f]
*
*  @param[in]
*  handle       handle to the hipsparse library context queue.
*  @param[in]
*  dir          the storage format of the blocks, \ref HIPSPARSE_DIRECTION_ROW or \ref HIPSPARSE_DIRECTION_COLUMN
*  @param[in]
*  m            number of rows in the sparse CSR matrix.
*  @param[in]
*  n            number of columns in the sparse CSR matrix.
*  @param[in]
*  csr_descr    descriptor of the sparse CSR matrix. Currently, only
*               \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*  @param[in]
*  csrVal       array of \p nnz elements containing the values of the sparse CSR matrix.
*  @param[in]
*  csrRowPtr    array of \p m+1 elements that point to the start of every row of the
*               sparse CSR matrix.
*  @param[in]
*  csrColInd    array of \p nnz elements containing the column indices of the sparse CSR matrix.
*  @param[in]
*  bsr_descr    descriptor of the sparse BSR matrix. Currently, only
*               \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*  @param[out]
*  bsrVal       array of \p nnzb* \p rowBlockDim* \p colBlockDim containing the values of the sparse BSR matrix.
*  @param[out]
*  bsrRowPtr    array of \p mb+1 elements that point to the start of every block row of the
*               sparse BSR matrix.
*  @param[out]
*  bsrColInd    array of \p nnzb elements containing the block column indices of the sparse BSR matrix.
*  @param[in]
*  rowBlockDim  row size of the blocks in the sparse GEneral BSR matrix.
*  @param[in]
*  colBlockDim  col size of the blocks in the sparse GEneral BSR matrix.
*  @param[in]
*  pbuffer      buffer allocated by the user whose size is determined by calling \ref hipsparseScsr2gebsr_bufferSize 
*               "hipsparseXcsr2gebsr_bufferSize()".
*
*  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p m, \p n, \p rowBlockDim, \p colBlockDim, \p bsrVal,
*              \p bsrRowPtr, \p bsrColInd, \p csrVal, \p csrRowPtr or \p csrColInd pointer is invalid.
*
*  \par Example
*  \code{.c}
*    // hipSPARSE handle
*    hipsparseHandle_t handle;
*    hipsparseCreate(&handle);
*
*    hipsparseMatDescr_t csr_descr;
*    hipsparseCreateMatDescr(&csr_descr);
*
*    hipsparseMatDescr_t bsr_descr;
*    hipsparseCreateMatDescr(&bsr_descr);
*
*    // Sparse matrix in CSR format
*    //     1 2 0 3 0 0 
*    //     0 4 5 0 0 1
*    // A = 6 0 0 7 8 0
*    //     0 0 3 0 2 2
*    //     1 0 0 0 4 3 
*    //     7 2 0 0 1 4
*    int hcsrRowPtr[7] = {0, 3, 6, 9, 12, 15, 19};
*    int hcsrColInd[19] = {0, 1, 3, 1, 2, 5, 0, 3, 4, 2, 4, 5, 0, 4, 5, 0, 1, 4, 5};
*    float hcsrVal[19]   = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 1.0f, 6.0f, 7.0f, 8.0f, 3.0f, 2.0f, 2.0f,
*                           1.0f, 4.0f, 3.0f, 7.0f, 2.0f, 1.0f, 4.0f}; 
*
*    int m           = 6;
*    int n           = 6;
*    int nnz         = 19;
*    int rowBlockDim = 3;
*    int colBlockDim = 2;
*    hipsparseDirection_t dir = HIPSPARSE_DIRECTION_ROW;
*    hipsparseIndexBase_t base = HIPSPARSE_INDEX_BASE_ZERO;
*
*    int mb = (m + rowBlockDim - 1) / rowBlockDim;
*    int nb = (n + colBlockDim - 1) / colBlockDim;
*
*    int* dcsrRowPtr = nullptr;
*    int* dcsrColInd = nullptr;
*    float* dcsrVal = nullptr;
*    hipMalloc((void**)&dcsrRowPtr, sizeof(int) * (m + 1));
*    hipMalloc((void**)&dcsrColInd, sizeof(int) * nnz);
*    hipMalloc((void**)&dcsrVal, sizeof(float) * nnz);
*
*    hipMemcpy(dcsrRowPtr, hcsrRowPtr, sizeof(int) * (m + 1), hipMemcpyHostToDevice);
*    hipMemcpy(dcsrColInd, hcsrColInd, sizeof(int) * nnz, hipMemcpyHostToDevice);
*    hipMemcpy(dcsrVal, hcsrVal, sizeof(float) * nnz, hipMemcpyHostToDevice);
*
*    int* dbsrRowPtr = nullptr;
*    hipMalloc((void**)&dbsrRowPtr, sizeof(int) * (mb + 1));
*
*    size_t bufferSize;
*    hipsparseScsr2gebsr_bufferSize(handle, 
*                                   dir, 
*                                   m, 
*                                   n, 
*                                   csr_descr, 
*                                   dcsrVal, 
*                                   dcsrRowPtr, 
*                                   dcsrColInd, 
*                                   rowBlockDim, 
*                                   colBlockDim, 
*                                   &bufferSize);
*
*    void* dbuffer = nullptr;
*    hipMalloc((void**)&dbuffer, bufferSize);
*
*    int nnzb;
*    hipsparseXcsr2gebsrNnz(handle, 
*                           dir, 
*                           m, 
*                           n, 
*                           csr_descr, 
*                           dcsrRowPtr, 
*                           dcsrColInd, 
*                           bsr_descr, 
*                           dbsrRowPtr, 
*                           rowBlockDim, 
*                           colBlockDim, 
*                           &nnzb, 
*                           dbuffer);
*
*    int* dbsrColInd = nullptr;
*    float* dbsrVal = nullptr;
*    hipMalloc((void**)&dbsrColInd, sizeof(int) * nnzb);
*    hipMalloc((void**)&dbsrVal, sizeof(float) * rowBlockDim * colBlockDim * nnzb);
*
*    hipsparseScsr2gebsr(handle, 
*                        dir, 
*                        m, 
*                        n, 
*                        csr_descr, 
*                        dcsrVal, 
*                        dcsrRowPtr, 
*                        dcsrColInd, 
*                        bsr_descr, 
*                        dbsrVal, 
*                        dbsrRowPtr, 
*                        dbsrColInd, 
*                        rowBlockDim, 
*                        colBlockDim, 
*                        dbuffer);
*
*    hipFree(dcsrRowPtr);
*    hipFree(dcsrColInd);
*    hipFree(dcsrVal);
*    
*    hipFree(dbsrRowPtr);
*    hipFree(dbsrColInd);
*    hipFree(dbsrVal);
*
*    hipFree(dbuffer);
*
*    hipsparseDestroyMatDescr(csr_descr);
*    hipsparseDestroyMatDescr(bsr_descr);
*    hipsparseDestroy(handle);
*  \endcode
*/
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseScsr2gebsr(hipsparseHandle_t         handle,
                                      hipsparseDirection_t      dir,
                                      int                       m,
                                      int                       n,
                                      const hipsparseMatDescr_t csr_descr,
                                      const float*              csrVal,
                                      const int*                csrRowPtr,
                                      const int*                csrColInd,
                                      const hipsparseMatDescr_t bsr_descr,
                                      float*                    bsrVal,
                                      int*                      bsrRowPtr,
                                      int*                      bsrColInd,
                                      int                       rowBlockDim,
                                      int                       colBlockDim,
                                      void*                     pbuffer);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDcsr2gebsr(hipsparseHandle_t         handle,
                                      hipsparseDirection_t      dir,
                                      int                       m,
                                      int                       n,
                                      const hipsparseMatDescr_t csr_descr,
                                      const double*             csrVal,
                                      const int*                csrRowPtr,
                                      const int*                csrColInd,
                                      const hipsparseMatDescr_t bsr_descr,
                                      double*                   bsrVal,
                                      int*                      bsrRowPtr,
                                      int*                      bsrColInd,
                                      int                       rowBlockDim,
                                      int                       colBlockDim,
                                      void*                     pbuffer);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCcsr2gebsr(hipsparseHandle_t         handle,
                                      hipsparseDirection_t      dir,
                                      int                       m,
                                      int                       n,
                                      const hipsparseMatDescr_t csr_descr,
                                      const hipComplex*         csrVal,
                                      const int*                csrRowPtr,
                                      const int*                csrColInd,
                                      const hipsparseMatDescr_t bsr_descr,
                                      hipComplex*               bsrVal,
                                      int*                      bsrRowPtr,
                                      int*                      bsrColInd,
                                      int                       rowBlockDim,
                                      int                       colBlockDim,
                                      void*                     pbuffer);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZcsr2gebsr(hipsparseHandle_t         handle,
                                      hipsparseDirection_t      dir,
                                      int                       m,
                                      int                       n,
                                      const hipsparseMatDescr_t csr_descr,
                                      const hipDoubleComplex*   csrVal,
                                      const int*                csrRowPtr,
                                      const int*                csrColInd,
                                      const hipsparseMatDescr_t bsr_descr,
                                      hipDoubleComplex*         bsrVal,
                                      int*                      bsrRowPtr,
                                      int*                      bsrColInd,
                                      int                       rowBlockDim,
                                      int                       colBlockDim,
                                      void*                     pbuffer);
/**@}*/

/*! \ingroup conv_module
*  \brief Convert a sparse CSR matrix into a sparse BSR matrix
*
*  \details
*  \p hipsparseXcsr2bsr completes the conversion of a CSR matrix into a BSR matrix. 
*  It is assumed, that \p bsrValC, \p bsrColIndC and \p bsrRowPtrC are allocated. The 
*  allocation size for \p bsrRowPtr is computed as \p mb+1 where \p mb is the number of 
*  block rows in the BSR matrix defined as:
*
*  \f[
*    \begin{align}
*    \text{mb} &= \text{(m - 1) / blockDim + 1}
*    \end{align}
*  \f]
*
*  The allocation size for \p bsrColIndC, i.e. \p bsrNnzb, is computed using
*  \ref hipsparseXcsr2bsrNnz() which also fills the \p bsrRowPtrC array. The allocation size 
*  for \p bsrValC is then equal to:
*
*  \f[
*    \text{bsrNnzb * blockDim * blockDim}
*  \f]
*
*  For example, given the CSR matrix:
*  \f[
*    \begin{bmatrix}
*    1 & 0 & 0 & 2 \\
*    3 & 4 & 0 & 0 \\
*    5 & 0 & 6 & 7 \\
*    1 & 2 & 3 & 4
*    \end{bmatrix}
*  \f]
*
*  The resulting BSR matrix using block dimension 2 would look like:
*  \f[
*   \left[ 
*    \begin{array}{c | c} 
*      \begin{array}{c c} 
*       1 & 0 \\ 
*       3 & 4 
*      \end{array} & 
*      \begin{array}{c c} 
*       0 & 2 \\ 
*       0 & 0 
*      \end{array} \\ 
*    \hline 
*      \begin{array}{c c} 
*       5 & 0 \\ 
*       1 & 2 
*      \end{array} & 
*      \begin{array}{c c} 
*       6 & 7 \\ 
*       3 & 4 
*      \end{array} \\ 
*   \end{array} 
*  \right] 
*  \f]
*
*  The call to \ref hipsparseXcsr2bsrNnz results in the BSR row pointer array:
*  \f[
*    \begin{align}
*    \text{bsrRowPtrC} &= \begin{bmatrix} 0 & 2 & 4 \end{bmatrix} \\
*    \end{align}
*  \f]
*
*  and the call to \p hipsparseXcsr2bsr completes the conversion resulting in the BSR column indices and values arrays:
*  \f[
*    \begin{align}
*    \text{bsrColIndC} &= \begin{bmatrix} 0 & 1 & 0 & 1 \end{bmatrix} \\
*    \text{bsrValC} &= \begin{bmatrix} 1 & 0 & 3 & 4 & 0 & 2 & 0 & 0 & 5 & 0 & 1 & 2 & 6 & 7 & 3 & 4 \end{bmatrix} \\
*    \end{align}
*  \f]
*
*  The \p dirA parameter determines the order of the BSR block values. The example above uses row order. Using column ordering
*  would result instead in the BSR values array:
*
*  \f[
*    \text{bsrValC} &= \begin{bmatrix} 1 & 3 & 0 & 4 & 0 & 0 & 2 & 0 & 5 & 1 & 0 & 2 & 6 & 3 & 7 & 4 \end{bmatrix} \\
*  \f]
*
*  \note
*  \p hipsparseXcsr2bsr requires extra temporary storage that is allocated internally if
*  \p blockDim>16
*
*  @param[in]
*  handle       handle to the hipsparse library context queue.
*  @param[in]
*  dirA         the storage format of the blocks, \ref HIPSPARSE_DIRECTION_ROW or \ref HIPSPARSE_DIRECTION_COLUMN
*  @param[in]
*  m            number of rows in the sparse CSR matrix.
*  @param[in]
*  n            number of columns in the sparse CSR matrix.
*  @param[in]
*  descrA       descriptor of the sparse CSR matrix. Currently, only
*               \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*  @param[in]
*  csrValA      array of \p nnz elements containing the values of the sparse CSR matrix.
*  @param[in]
*  csrRowPtrA   array of \p m+1 elements that point to the start of every row of the
*               sparse CSR matrix.
*  @param[in]
*  csrColIndA   array of \p nnz elements containing the column indices of the sparse CSR matrix.
*  @param[in]
*  blockDim     size of the blocks in the sparse BSR matrix.
*  @param[in]
*  descrC       descriptor of the sparse BSR matrix. Currently, only
*               \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*  @param[out]
*  bsrValC      array of \p nnzb*blockDim*blockDim containing the values of the sparse BSR matrix.
*  @param[out]
*  bsrRowPtrC   array of \p mb+1 elements that point to the start of every block row of the
*               sparse BSR matrix.
*  @param[out]
*  bsrColIndC   array of \p nnzb elements containing the block column indices of the sparse BSR matrix.
*
*  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p m, \p n, \p blockDim, \p bsrValC, \p bsrRowPtrC, 
*              \p bsrColIndC, \p csrValA, \p csrRowPtrA or \p csrColIndA pointer is invalid.
*
*  \par Example
*  \code{.c}
*    // hipSPARSE handle
*    hipsparseHandle_t handle;
*    hipsparseCreate(&handle);
*
*    // Matrix descriptor
*    hipsparseMatDescr_t csr_descr;
*    hipsparseCreateMatDescr(&csr_descr);
*
*    hipsparseMatDescr_t bsr_descr;
*    hipsparseCreateMatDescr(&bsr_descr);
*
*    // Sparse matrix in CSR format
*    //     1 2 0 3 0
*    // A = 0 4 5 0 0
*    //     6 0 0 7 8
*    int hcsrRowPtr[4] = {0, 3, 5, 8};
*    int hcsrColInd[8] = {0, 1, 3, 1, 2, 0, 3, 4};
*    float hcsrVal[8]   = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f}; 
*
*    int m         = 3;
*    int n         = 5;
*    int nnz       = 8;
*    int blockDim  = 3;
*    hipsparseDirection_t dir = HIPSPARSE_DIRECTION_ROW;
*
*    int mb = (m + blockDim - 1) / blockDim;
*    int nb = (n + blockDim - 1) / blockDim;
*
*    int* dcsrRowPtr = nullptr;
*    int* dcsrColInd = nullptr;
*    float* dcsrVal = nullptr;
*    hipMalloc((void**)&dcsrRowPtr, sizeof(int) * (m + 1));
*    hipMalloc((void**)&dcsrColInd, sizeof(int) * nnz);
*    hipMalloc((void**)&dcsrVal, sizeof(float) * nnz);
*
*    hipMemcpy(dcsrRowPtr, hcsrRowPtr, sizeof(int) * (m + 1), hipMemcpyHostToDevice);
*    hipMemcpy(dcsrColInd, hcsrColInd, sizeof(int) * nnz, hipMemcpyHostToDevice);
*    hipMemcpy(dcsrVal, hcsrVal, sizeof(float) * nnz, hipMemcpyHostToDevice);
*
*    int* dbsrRowPtr = nullptr;
*    hipMalloc((void**)&dbsrRowPtr, sizeof(int) * (mb + 1));
*
*    int nnzb;
*    hipsparseXcsr2bsrNnz(handle, dir, m, n, csr_descr, dcsrRowPtr, dcsrColInd, blockDim, bsr_descr, dbsrRowPtr, &nnzb);
*
*    int* dbsrColInd = nullptr;
*    float* dbsrVal = nullptr;
*    hipMalloc((void**)&dbsrColInd, sizeof(int) * nnzb);
*    hipMalloc((void**)&dbsrVal, sizeof(float) * blockDim * blockDim * nnzb);
*
*    hipsparseScsr2bsr(handle, dir, m, n, csr_descr, dcsrVal, dcsrRowPtr, dcsrColInd, blockDim, bsr_descr, dbsrVal, dbsrRowPtr, dbsrColInd);
*
*    hipFree(dcsrRowPtr);
*    hipFree(dcsrColInd);
*    hipFree(dcsrVal);
*   
*    hipFree(dbsrRowPtr);
*    hipFree(dbsrColInd);
*    hipFree(dbsrVal);
*
*    hipsparseDestroyMatDescr(csr_descr);
*    hipsparseDestroyMatDescr(bsr_descr);
*    hipsparseDestroy(handle);
*  \endcode
*/
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseScsr2bsr(hipsparseHandle_t         handle,
                                    hipsparseDirection_t      dirA,
                                    int                       m,
                                    int                       n,
                                    const hipsparseMatDescr_t descrA,
                                    const float*              csrValA,
                                    const int*                csrRowPtrA,
                                    const int*                csrColIndA,
                                    int                       blockDim,
                                    const hipsparseMatDescr_t descrC,
                                    float*                    bsrValC,
                                    int*                      bsrRowPtrC,
                                    int*                      bsrColIndC);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDcsr2bsr(hipsparseHandle_t         handle,
                                    hipsparseDirection_t      dirA,
                                    int                       m,
                                    int                       n,
                                    const hipsparseMatDescr_t descrA,
                                    const double*             csrValA,
                                    const int*                csrRowPtrA,
                                    const int*                csrColIndA,
                                    int                       blockDim,
                                    const hipsparseMatDescr_t descrC,
                                    double*                   bsrValC,
                                    int*                      bsrRowPtrC,
                                    int*                      bsrColIndC);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCcsr2bsr(hipsparseHandle_t         handle,
                                    hipsparseDirection_t      dirA,
                                    int                       m,
                                    int                       n,
                                    const hipsparseMatDescr_t descrA,
                                    const hipComplex*         csrValA,
                                    const int*                csrRowPtrA,
                                    const int*                csrColIndA,
                                    int                       blockDim,
                                    const hipsparseMatDescr_t descrC,
                                    hipComplex*               bsrValC,
                                    int*                      bsrRowPtrC,
                                    int*                      bsrColIndC);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZcsr2bsr(hipsparseHandle_t         handle,
                                    hipsparseDirection_t      dirA,
                                    int                       m,
                                    int                       n,
                                    const hipsparseMatDescr_t descrA,
                                    const hipDoubleComplex*   csrValA,
                                    const int*                csrRowPtrA,
                                    const int*                csrColIndA,
                                    int                       blockDim,
                                    const hipsparseMatDescr_t descrC,
                                    hipDoubleComplex*         bsrValC,
                                    int*                      bsrRowPtrC,
                                    int*                      bsrColIndC);
/**@}*/

/*! \ingroup conv_module
*  \brief Convert a sparse BSR matrix into a sparse CSR matrix
*
*  \details
*  \p hipsparseXbsr2csr converts a BSR matrix into a CSR matrix. It is assumed,
*  that \p csrValC, \p csrColIndC and \p csrRowPtrC are allocated. Allocation size
*  for \p csrRowPtrC is computed by the number of block rows multiplied by the block
*  dimension plus one. Allocation for \p csrValC and \p csrColInd is computed by the
*  the number of blocks in the BSR matrix multiplied by the block dimension squared.
*
*  For example, given the BSR matrix using block dimension 2:
*  \f[
*   \left[ 
*    \begin{array}{c | c} 
*      \begin{array}{c c} 
*       1 & 0 \\ 
*       3 & 4 
*      \end{array} & 
*      \begin{array}{c c} 
*       0 & 2 \\ 
*       0 & 0 
*      \end{array} \\ 
*    \hline 
*      \begin{array}{c c} 
*       5 & 0 \\ 
*       1 & 2 
*      \end{array} & 
*      \begin{array}{c c} 
*       6 & 7 \\ 
*       3 & 4 
*      \end{array} \\ 
*   \end{array} 
*  \right] 
*  \f]
*
*  The resulting CSR matrix row pointer, column indices, and values arrays are:
*  \f[
*    \begin{align}
*    \text{csrRowPtrC} &= \begin{bmatrix} 0 & 4 & 8 & 12 & 16 \end{bmatrix} \\
*    \text{csrColIndC} &= \begin{bmatrix} 0 & 1 & 2 & 3 & 0 & 1 & 2 & 3 & 0 & 1 & 2 & 3 & 0 & 1 & 2 & 3 \end{bmatrix} \\
*    \text{csrValC} &= \begin{bmatrix} 1 & 0 & 0 & 2 & 3 & 4 & 0 & 0 & 5 & 0 & 6 & 7 & 1 & 2 & 3 & 4 \end{bmatrix} \\
*    \end{align}
*  \f]
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*
*  @param[in]
*  handle      handle to the hipsparse library context queue.
*  @param[in]
*  dirA        the storage format of the blocks, \ref HIPSPARSE_DIRECTION_ROW or \ref HIPSPARSE_DIRECTION_COLUMN
*  @param[in]
*  mb          number of block rows in the sparse BSR matrix.
*  @param[in]
*  nb          number of block columns in the sparse BSR matrix.
*  @param[in]
*  descrA      descriptor of the sparse BSR matrix. Currently, only
*              \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*  @param[in]
*  bsrValA     array of \p nnzb*blockDim*blockDim containing the values of the sparse BSR matrix.
*  @param[in]
*  bsrRowPtrA  array of \p mb+1 elements that point to the start of every block row of the
*              sparse BSR matrix.
*  @param[in]
*  bsrColIndA  array of \p nnzb elements containing the block column indices of the sparse BSR matrix.
*  @param[in]
*  blockDim    size of the blocks in the sparse BSR matrix.
*  @param[in]
*  descrC      descriptor of the sparse CSR matrix. Currently, only
*              \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*  @param[out]
*  csrValC     array of \p nnzb*blockDim*blockDim elements containing the values of the sparse CSR matrix.
*  @param[out]
*  csrRowPtrC  array of \p m+1 where \p m=mb*blockDim elements that point to the start of every row of the
*              sparse CSR matrix.
*  @param[out]
*  csrColIndC  array of \p nnzb*blockDim*blockDim elements containing the column indices of the sparse CSR matrix.
*
*  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p mb, \p nb, \p blockDim, \p bsrValA,
*              \p bsrRowPtrA, \p bsrColIndA, \p csrValC, \p csrRowPtrC or \p csrColIndC pointer is invalid.
*
*  \par Example
*  \code{.c}
*    // hipSPARSE handle
*    hipsparseHandle_t handle;
*    hipsparseCreate(&handle);
*
*    hipsparseMatDescr_t csr_descr;
*    hipsparseCreateMatDescr(&csr_descr);
*
*    hipsparseMatDescr_t bsr_descr;
*    hipsparseCreateMatDescr(&bsr_descr);
*
*    // Sparse matrix in BSR format
*    //     1 2 | 0 3 | 0 0 
*    //     0 4 | 5 0 | 0 1
*    //     ---------------
*    // A = 6 0 | 0 7 | 8 0 
*    //     0 0 | 3 0 | 2 2
*    //     ---------------
*    //     1 0 | 0 0 | 4 3 
*    //     7 2 | 0 0 | 1 4
*    int hbsrRowPtr[4] = {0, 3, 6, 8};
*    int hbsrColInd[8] = {0, 1, 2, 0, 1, 2, 0, 2};
*    float hbsrVal[32]  = {1.0f, 2.0f, 0.0f, 4.0f, 
*                            0.0f, 3.0f, 5.0f, 0.0f, 
*                            0.0f, 0.0f, 0.0f, 1.0f,
*                            6.0f, 0.0f, 0.0f, 0.0f, 
*                            0.0f, 7.0f, 3.0f, 0.0f,
*                            8.0f, 0.0f, 2.0f, 2.0f,
*                            1.0f, 0.0f, 7.0f, 2.0f,
*                            4.0f, 3.0f, 1.0f, 4.0f}; 
*
*    int m        = 6;
*    int n        = 6;
*    int nnz      = 32;
*    int mb       = 3;
*    int nb       = 3;
*    int nnzb     = 8;
*    int blockDim = 2;
*    hipsparseDirection_t dir = HIPSPARSE_DIRECTION_ROW;
*
*    int* dbsrRowPtr = nullptr;
*    int* dbsrColInd = nullptr;
*    float* dbsrVal = nullptr;
*    hipMalloc((void**)&dbsrRowPtr, sizeof(int) * (mb + 1));
*    hipMalloc((void**)&dbsrColInd, sizeof(int) * nnzb);
*    hipMalloc((void**)&dbsrVal, sizeof(float) * blockDim * blockDim * nnzb);
*
*    hipMemcpy(dbsrRowPtr, hbsrRowPtr, sizeof(int) * (mb + 1), hipMemcpyHostToDevice);
*    hipMemcpy(dbsrColInd, hbsrColInd, sizeof(int) * nnzb, hipMemcpyHostToDevice);
*    hipMemcpy(dbsrVal, hbsrVal, sizeof(float) * blockDim * blockDim * nnzb, hipMemcpyHostToDevice);
*
*    int* dcsrRowPtr = nullptr;
*    int* dcsrColInd = nullptr;
*    float* dcsrVal = nullptr;
*    hipMalloc((void**)&dcsrRowPtr, sizeof(int) * (m + 1));
*    hipMalloc((void**)&dcsrColInd, sizeof(int) * nnz);
*    hipMalloc((void**)&dcsrVal, sizeof(float) * nnz);
*
*    hipsparseSbsr2csr(handle, 
*                      dir, 
*                      mb, 
*                      nb, 
*                      bsr_descr, 
*                      dbsrVal, 
*                      dbsrRowPtr, 
*                      dbsrColInd, 
*                      blockDim, 
*                      csr_descr, 
*                      dcsrVal, 
*                      dcsrRowPtr, 
*                      dcsrColInd);
*
*    hipFree(dbsrRowPtr);
*    hipFree(dbsrColInd);
*    hipFree(dbsrVal);
*    
*    hipFree(dcsrRowPtr);
*    hipFree(dcsrColInd);
*    hipFree(dcsrVal);
*
*    hipsparseDestroyMatDescr(csr_descr);
*    hipsparseDestroyMatDescr(bsr_descr);
*    hipsparseDestroy(handle);
*  \endcode
*/
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSbsr2csr(hipsparseHandle_t         handle,
                                    hipsparseDirection_t      dirA,
                                    int                       mb,
                                    int                       nb,
                                    const hipsparseMatDescr_t descrA,
                                    const float*              bsrValA,
                                    const int*                bsrRowPtrA,
                                    const int*                bsrColIndA,
                                    int                       blockDim,
                                    const hipsparseMatDescr_t descrC,
                                    float*                    csrValC,
                                    int*                      csrRowPtrC,
                                    int*                      csrColIndC);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDbsr2csr(hipsparseHandle_t         handle,
                                    hipsparseDirection_t      dirA,
                                    int                       mb,
                                    int                       nb,
                                    const hipsparseMatDescr_t descrA,
                                    const double*             bsrValA,
                                    const int*                bsrRowPtrA,
                                    const int*                bsrColIndA,
                                    int                       blockDim,
                                    const hipsparseMatDescr_t descrC,
                                    double*                   csrValC,
                                    int*                      csrRowPtrC,
                                    int*                      csrColIndC);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCbsr2csr(hipsparseHandle_t         handle,
                                    hipsparseDirection_t      dirA,
                                    int                       mb,
                                    int                       nb,
                                    const hipsparseMatDescr_t descrA,
                                    const hipComplex*         bsrValA,
                                    const int*                bsrRowPtrA,
                                    const int*                bsrColIndA,
                                    int                       blockDim,
                                    const hipsparseMatDescr_t descrC,
                                    hipComplex*               csrValC,
                                    int*                      csrRowPtrC,
                                    int*                      csrColIndC);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZbsr2csr(hipsparseHandle_t         handle,
                                    hipsparseDirection_t      dirA,
                                    int                       mb,
                                    int                       nb,
                                    const hipsparseMatDescr_t descrA,
                                    const hipDoubleComplex*   bsrValA,
                                    const int*                bsrRowPtrA,
                                    const int*                bsrColIndA,
                                    int                       blockDim,
                                    const hipsparseMatDescr_t descrC,
                                    hipDoubleComplex*         csrValC,
                                    int*                      csrRowPtrC,
                                    int*                      csrColIndC);
/**@}*/

/*! \ingroup conv_module
*  \brief Convert a sparse GEBSR matrix into a sparse CSR matrix
*
*  \details
*  \p hipsparseXgebsr2csr converts a GEBSR matrix into a CSR matrix. It is assumed,
*  that \p csrValC, \p csrColIndC and \p csrRowPtrC are already allocated prior to 
*  calling \p hipsparseXgebsr2csr. Allocation size for \p csrRowPtrC equals 
*  \p m+1 where:
*
*  \f[
*    \begin{align}
*    \text{m} &= \text{mb * rowBlockDim} \\
*    \text{n} &= \text{nb * colBlockDim}
*    \end{align}
*  \f]
*
*  Allocation size for \p csrValC and \p csrColIndC is computed by the the number of blocks in the GEBSR 
*  matrix, \p nnzb, multiplied by the product of the block dimensions, i.e. \p nnz=nnzb*rocBlockDim*colBlockDim.
*
*  For example, given the GEBSR matrix:
*  \f[
*   \left[ 
*    \begin{array}{c | c | c} 
*      \begin{array}{c c} 
*       6 & 2 \\ 
*       1 & 4 \\ 
*       5 & 4 
*      \end{array} & 
*      \begin{array}{c c} 
*       0 & 3 \\ 
*       5 & 0 \\ 
*       0 & 7 
*      \end{array} &
*      \begin{array}{c c} 
*       0 & 0 \\ 
*       0 & 0 \\ 
*       0 & 0 
*      \end{array} \\ 
*    \hline 
*      \begin{array}{c c} 
*       0 & 0 \\ 
*       0 & 0 \\ 
*       0 & 0 
*      \end{array} & 
*      \begin{array}{c c} 
*       3 & 0 \\ 
*       0 & 0 \\ 
*       0 & 7 
*      \end{array} &
*      \begin{array}{c c} 
*       2 & 2 \\ 
*       4 & 3 \\ 
*       1 & 4 
*      \end{array} \\ 
*   \end{array} 
*  \right] 
*  \f]
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*
*  @param[in]
*  handle      handle to the hipsparse library context queue.
*  @param[in]
*  dirA        the storage format of the blocks, \ref HIPSPARSE_DIRECTION_ROW or \ref HIPSPARSE_DIRECTION_COLUMN
*  @param[in]
*  mb          number of block rows in the sparse general BSR matrix.
*  @param[in]
*  nb          number of block columns in the sparse general BSR matrix.
*  @param[in]
*  descrA      descriptor of the sparse general BSR matrix. Currently, only
*              \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*  @param[in]
*  bsrValA      array of \p nnzb*rowBlockDim*colBlockDim containing the values of the sparse BSR matrix.
*  @param[in]
*  bsrRowPtrA  array of \p mb+1 elements that point to the start of every block row of the
*              sparse BSR matrix.
*  @param[in]
*  bsrColIndA  array of \p nnzb elements containing the block column indices of the sparse BSR matrix.
*  @param[in]
*  rowBlockDim row size of the blocks in the sparse general BSR matrix.
*  @param[in]
*  colBlockDim column size of the blocks in the sparse general BSR matrix.
*  @param[in]
*  descrC      descriptor of the sparse CSR matrix. Currently, only
*              \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*  @param[out]
*  csrValC     array of \p nnzb*rowBlockDim*colBlockDim elements containing the values of the sparse CSR matrix.
*  @param[out]
*  csrRowPtrC  array of \p m+1 where \p m=mb*rowBlockDim elements that point to the start of every row of the
*              sparse CSR matrix.
*  @param[out]
*  csrColIndC  array of \p nnzb*block_dim*block_dim elements containing the column indices of the sparse CSR matrix.
*
*  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p mb, \p nb, \p block_dim, \p bsrValA,
*              \p bsrRowPtrA, \p bsrColIndA, \p csrValC, \p csrRowPtrC or \p csrColIndC pointer is invalid.
*
*  \par Example
*  \code{.c}
*    // hipSPARSE handle
*    hipsparseHandle_t handle;
*    hipsparseCreate(&handle);
*
*    hipsparseMatDescr_t csr_descr;
*    hipsparseCreateMatDescr(&csr_descr);
*
*    hipsparseMatDescr_t bsr_descr;
*    hipsparseCreateMatDescr(&bsr_descr);
*
*    // Sparse matrix in GEBSR format
*    //     1 2 | 0 3 | 0 0 
*    //     0 4 | 5 0 | 0 1
*    // A = 6 0 | 0 7 | 8 0
*    //     --------------- 
*    //     0 0 | 3 0 | 2 2
*    //     1 0 | 0 0 | 4 3 
*    //     7 2 | 0 0 | 1 4
*    int hbsrRowPtr[3] = {0, 3, 6};
*    int hbsrColInd[6] = {0, 1, 2, 0, 1, 2};
*    float hbsrVal[36]  = {1.0f, 2.0f, 0.0f, 4.0f, 6.0f, 0.0f, 
*                           0.0f, 3.0f, 5.0f, 0.0f, 0.0f, 7.0f,
*                           0.0f, 0.0f, 0.0f, 1.0f, 8.0f, 0.0f, 
*                           0.0f, 0.0f, 1.0f, 0.0f, 7.0f, 2.0f,
*                           3.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 
*                           2.0f, 2.0f, 4.0f, 3.0f, 1.0f, 4.0f}; 
*
*    int m        = 6;
*    int n        = 6;
*    int nnz      = 36;
*    int mb       = 2;
*    int nb       = 3;
*    int nnzb     = 6;
*    int rowBlockDim = 3;
*    int colBlockDim = 2;
*    hipsparseDirection_t dir = HIPSPARSE_DIRECTION_ROW;
*
*    int* dbsrRowPtr = nullptr;
*    int* dbsrColInd = nullptr;
*    float* dbsrVal = nullptr;
*    hipMalloc((void**)&dbsrRowPtr, sizeof(int) * (mb + 1));
*    hipMalloc((void**)&dbsrColInd, sizeof(int) * nnzb);
*    hipMalloc((void**)&dbsrVal, sizeof(float) * rowBlockDim * colBlockDim * nnzb);
*
*    hipMemcpy(dbsrRowPtr, hbsrRowPtr, sizeof(int) * (mb + 1), hipMemcpyHostToDevice);
*    hipMemcpy(dbsrColInd, hbsrColInd, sizeof(int) * nnzb, hipMemcpyHostToDevice);
*    hipMemcpy(dbsrVal, hbsrVal, sizeof(float) * rowBlockDim * colBlockDim * nnzb, hipMemcpyHostToDevice);
*
*    int* dcsrRowPtr = nullptr;
*    int* dcsrColInd = nullptr;
*    float* dcsrVal = nullptr;
*    hipMalloc((void**)&dcsrRowPtr, sizeof(int) * (m + 1));
*    hipMalloc((void**)&dcsrColInd, sizeof(int) * nnz);
*    hipMalloc((void**)&dcsrVal, sizeof(float) * nnz);
*
*    hipsparseSgebsr2csr(handle, 
*                        dir, 
*                        mb, 
*                        nb, 
*                        bsr_descr, 
*                        dbsrVal, 
*                        dbsrRowPtr, 
*                        dbsrColInd, 
*                        rowBlockDim, 
*                        colBlockDim, 
*                        csr_descr, 
*                        dcsrVal, 
*                        dcsrRowPtr, 
*                        dcsrColInd);
*
*    hipFree(dbsrRowPtr);
*    hipFree(dbsrColInd);
*    hipFree(dbsrVal);
*    
*    hipFree(dcsrRowPtr);
*    hipFree(dcsrColInd);
*    hipFree(dcsrVal);
*
*    hipsparseDestroyMatDescr(csr_descr);
*    hipsparseDestroyMatDescr(bsr_descr);
*    hipsparseDestroy(handle);
*  \endcode
*/
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSgebsr2csr(hipsparseHandle_t         handle,
                                      hipsparseDirection_t      dirA,
                                      int                       mb,
                                      int                       nb,
                                      const hipsparseMatDescr_t descrA,
                                      const float*              bsrValA,
                                      const int*                bsrRowPtrA,
                                      const int*                bsrColIndA,
                                      int                       rowBlockDim,
                                      int                       colBlockDim,
                                      const hipsparseMatDescr_t descrC,
                                      float*                    csrValC,
                                      int*                      csrRowPtrC,
                                      int*                      csrColIndC);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDgebsr2csr(hipsparseHandle_t         handle,
                                      hipsparseDirection_t      dirA,
                                      int                       mb,
                                      int                       nb,
                                      const hipsparseMatDescr_t descrA,
                                      const double*             bsrValA,
                                      const int*                bsrRowPtrA,
                                      const int*                bsrColIndA,
                                      int                       rowBlockDim,
                                      int                       colBlockDim,
                                      const hipsparseMatDescr_t descrC,
                                      double*                   csrValC,
                                      int*                      csrRowPtrC,
                                      int*                      csrColIndC);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCgebsr2csr(hipsparseHandle_t         handle,
                                      hipsparseDirection_t      dirA,
                                      int                       mb,
                                      int                       nb,
                                      const hipsparseMatDescr_t descrA,
                                      const hipComplex*         bsrValA,
                                      const int*                bsrRowPtrA,
                                      const int*                bsrColIndA,
                                      int                       rowBlockDim,
                                      int                       colBlockDim,
                                      const hipsparseMatDescr_t descrC,
                                      hipComplex*               csrValC,
                                      int*                      csrRowPtrC,
                                      int*                      csrColIndC);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZgebsr2csr(hipsparseHandle_t         handle,
                                      hipsparseDirection_t      dirA,
                                      int                       mb,
                                      int                       nb,
                                      const hipsparseMatDescr_t descrA,
                                      const hipDoubleComplex*   bsrValA,
                                      const int*                bsrRowPtrA,
                                      const int*                bsrColIndA,
                                      int                       rowBlockDim,
                                      int                       colBlockDim,
                                      const hipsparseMatDescr_t descrC,
                                      hipDoubleComplex*         csrValC,
                                      int*                      csrRowPtrC,
                                      int*                      csrColIndC);
/**@}*/

/*! \ingroup conv_module
*  \brief Convert a sparse CSR matrix into a compressed sparse CSR matrix
*
*  \details
*  \p hipsparseXcsr2csr_compress converts a CSR matrix into a compressed CSR matrix by
*  removing entries in the input CSR matrix that are below a non-negative threshold \p tol:
*
*  \f[ 
*   C(i,j) = A(i, j) \text{  if |A(i, j)| > tol}
*  \f]
*
*  The user must first call \ref hipsparseSnnz_compress "hipsparseXnnz_compress()" to determine the number 
*  of nonzeros per row as well as the total number of nonzeros that will exist in resulting compressed CSR 
*  matrix. The user then uses this information to allocate the column indices array \p csrColIndC and the 
*  values array \p csrValC. The user then calls \p hipsparseXcsr2csr_compress to complete the conversion.
*
*  \note
*  In the case of complex matrices only the magnitude of the real part of \p tol is used.
*
*  @param[in]
*  handle        handle to the hipsparse library context queue.
*  @param[in]
*  m             number of rows of the sparse CSR matrix.
*  @param[in]
*  n             number of columns of the sparse CSR matrix.
*  @param[in]
*  descrA        matrix descriptor for the CSR matrix
*  @param[in]
*  csrValA       array of \p nnzA elements of the sparse CSR matrix.
*  @param[in]
*  csrRowPtrA    array of \p m+1 elements that point to the start of every row of the
*                uncompressed sparse CSR matrix.
*  @param[in]
*  csrColIndA    array of \p nnzA elements containing the column indices of the uncompressed
*                sparse CSR matrix.
*  @param[in]
*  nnzA          number of elements in the column indices and values arrays of the uncompressed
*                sparse CSR matrix.
*  @param[in]
*  nnzPerRow     array of length \p m containing the number of entries that will be kept per row in
*                the final compressed CSR matrix.
*  @param[out]
*  csrValC       array of \p nnzC elements of the compressed sparse CSC matrix.
*  @param[out]
*  csrRowPtrC    array of \p m+1 elements that point to the start of every column of the compressed
*                sparse CSR matrix.
*  @param[out]
*  csrColIndC    array of \p nnzC elements containing the row indices of the compressed
*                sparse CSR matrix.
*  @param[in]
*  tol           the non-negative tolerance used for compression. If \p tol is complex then only the magnitude
*                of the real part is used. Entries in the input uncompressed CSR array that are below the tolerance
*                are removed in output compressed CSR matrix.
*
*  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p m, \p n, \p nnzA, \p tol, \p csrValA, \p csrRowPtrA,
*              \p csrColIndA, \p csrValC, \p csrRowPtrC, \p csrColIndC or \p nnzPerRow pointer is invalid.
*
*  \par Example
*  \code{.c}
*    // hipSPARSE handle
*    hipsparseHandle_t handle;
*    hipsparseCreate(&handle);
*
*    // Matrix descriptor
*    hipsparseMatDescr_t descr;
*    hipsparseCreateMatDescr(&descr);
*
*    // Sparse matrix in CSR format
*    //     1 2 0 3 0
*    // A = 0 4 5 0 0
*    //     6 0 0 7 8
*    int hcsrRowPtrA[4] = {0, 3, 5, 8};
*    int hcsrColIndA[8] = {0, 1, 3, 1, 2, 0, 3, 4};
*    float hcsrValA[8]   = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f}; 
*
*    int m    = 3;
*    int n    = 5;
*    int nnzA = 8;
*
*    float tol = 5.9f;
*    
*    int* dcsrRowPtrA = nullptr;
*    int* dcsrColIndA = nullptr;
*    float* dcsrValA = nullptr;
*    hipMalloc((void**)&dcsrRowPtrA, sizeof(int) * (m + 1));
*    hipMalloc((void**)&dcsrColIndA, sizeof(int) * nnzA);
*    hipMalloc((void**)&dcsrValA, sizeof(float) * nnzA);
*
*    hipMemcpy(dcsrRowPtrA, hcsrRowPtrA, sizeof(int) * (m + 1), hipMemcpyHostToDevice);
*    hipMemcpy(dcsrColIndA, hcsrColIndA, sizeof(int) * nnzA, hipMemcpyHostToDevice);
*    hipMemcpy(dcsrValA, hcsrValA, sizeof(float) * nnzA, hipMemcpyHostToDevice);
*
*    // Allocate memory for the nnz_per_row array
*    int* dnnz_per_row;
*    hipMalloc((void**)&dnnz_per_row, sizeof(int) * m);
*
*    // Call snnz_compress() which fills in nnz_per_row array and finds the number
*    // of entries that will be in the compressed CSR matrix
*    int nnzC;
*    hipsparseSnnz_compress(handle, m, descr, dcsrValA, dcsrRowPtrA, dnnz_per_row, &nnzC, tol);
*
*    int* dcsrRowPtrC = nullptr;
*    int* dcsrColIndC = nullptr;
*    float* dcsrValC = nullptr;
*    hipMalloc((void**)&dcsrRowPtrC, sizeof(int) * (m + 1));
*    hipMalloc((void**)&dcsrColIndC, sizeof(int) * nnzC);
*    hipMalloc((void**)&dcsrValC, sizeof(float) * nnzC);
*
*    hipsparseScsr2csr_compress(handle,
*                               m,
*                               n,
*                               descr,
*                               dcsrValA,
*                               dcsrColIndA,
*                               dcsrRowPtrA,
*                               nnzA,
*                               dnnz_per_row,
*                               dcsrValC,
*                               dcsrColIndC,
*                               dcsrRowPtrC,
*                               tol);
*
*    hipFree(dcsrRowPtrA);
*    hipFree(dcsrColIndA);
*    hipFree(dcsrValA);
*    
*    hipFree(dcsrRowPtrC);
*    hipFree(dcsrColIndC);
*    hipFree(dcsrValC);
*
*    hipFree(dnnz_per_row);
*
*    hipsparseDestroyMatDescr(descr);
*    hipsparseDestroy(handle);
*  \endcode
*/
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseScsr2csr_compress(hipsparseHandle_t         handle,
                                             int                       m,
                                             int                       n,
                                             const hipsparseMatDescr_t descrA,
                                             const float*              csrValA,
                                             const int*                csrColIndA,
                                             const int*                csrRowPtrA,
                                             int                       nnzA,
                                             const int*                nnzPerRow,
                                             float*                    csrValC,
                                             int*                      csrColIndC,
                                             int*                      csrRowPtrC,
                                             float                     tol);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDcsr2csr_compress(hipsparseHandle_t         handle,
                                             int                       m,
                                             int                       n,
                                             const hipsparseMatDescr_t descrA,
                                             const double*             csrValA,
                                             const int*                csrColIndA,
                                             const int*                csrRowPtrA,
                                             int                       nnzA,
                                             const int*                nnzPerRow,
                                             double*                   csrValC,
                                             int*                      csrColIndC,
                                             int*                      csrRowPtrC,
                                             double                    tol);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCcsr2csr_compress(hipsparseHandle_t         handle,
                                             int                       m,
                                             int                       n,
                                             const hipsparseMatDescr_t descrA,
                                             const hipComplex*         csrValA,
                                             const int*                csrColIndA,
                                             const int*                csrRowPtrA,
                                             int                       nnzA,
                                             const int*                nnzPerRow,
                                             hipComplex*               csrValC,
                                             int*                      csrColIndC,
                                             int*                      csrRowPtrC,
                                             hipComplex                tol);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZcsr2csr_compress(hipsparseHandle_t         handle,
                                             int                       m,
                                             int                       n,
                                             const hipsparseMatDescr_t descrA,
                                             const hipDoubleComplex*   csrValA,
                                             const int*                csrColIndA,
                                             const int*                csrRowPtrA,
                                             int                       nnzA,
                                             const int*                nnzPerRow,
                                             hipDoubleComplex*         csrValC,
                                             int*                      csrColIndC,
                                             int*                      csrRowPtrC,
                                             hipDoubleComplex          tol);
/**@}*/

/*! \ingroup conv_module
 *  \brief Convert and prune sparse CSR matrix into a sparse CSR matrix
 *
 *  \details
 *  \p hipsparseXpruneCsr2csr_bufferSize returns the size of the temporary buffer that
 *  is required by \p hipsparseXpruneCsr2csrNnz and hipsparseXpruneCsr2csr. The
 *  temporary storage buffer must be allocated by the user.
 *
 *  @param[in]
 *  handle             handle to the hipsparse library context queue.
 *  @param[in]
 *  m                  number of rows in the sparse CSR matrix.
 *  @param[in]
 *  n                  number of columns in the sparse CSR matrix.
 *  @param[in]
 *  nnzA               number of non-zeros in the sparse CSR matrix A.
 *  @param[in]
 *  descrA             descriptor of the sparse CSR matrix A. Currently, only
 *                     \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
 *  @param[in]
 *  csrValA            array of \p nnzA elements containing the values of the sparse CSR matrix A.
 *  @param[in]
 *  csrRowPtrA         array of \p m+1 elements that point to the start of every row of the
 *                     sparse CSR matrix A.
 *  @param[in]
 *  csrColIndA         array of \p nnzA elements containing the column indices of the sparse CSR matrix A.
 *  @param[in]
 *  threshold          pointer to the non-negative pruning threshold which can exist in either host or device memory.
 *  @param[in]
 *  descrC             descriptor of the sparse CSR matrix C. Currently, only
 *                     \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
 *  @param[in]
 *  csrValC            array of \p nnzC elements containing the values of the sparse CSR matrix C.
 *  @param[in]
 *  csrRowPtrC         array of \p m+1 elements that point to the start of every row of the
 *                     sparse CSR matrix C.
 *  @param[in]
 *  csrColIndC         array of \p nnzC elements containing the column indices of the sparse CSR matrix C.
 *  @param[out]
 *  pBufferSizeInBytes number of bytes of the temporary storage buffer required by hipsparseSpruneCsr2csrNnz(),
 *                     hipsparseDpruneCsr2csrNnz(), hipsparseSpruneCsr2csr(), and hipsparseDpruneCsr2csr().
 *
 *  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
 *  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle or \p pBufferSizeInBytes pointer is invalid.
 *  \retval     HIPSPARSE_STATUS_INTERNAL_ERROR an internal error occurred.
 */
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpruneCsr2csr_bufferSize(hipsparseHandle_t         handle,
                                                    int                       m,
                                                    int                       n,
                                                    int                       nnzA,
                                                    const hipsparseMatDescr_t descrA,
                                                    const float*              csrValA,
                                                    const int*                csrRowPtrA,
                                                    const int*                csrColIndA,
                                                    const float*              threshold,
                                                    const hipsparseMatDescr_t descrC,
                                                    const float*              csrValC,
                                                    const int*                csrRowPtrC,
                                                    const int*                csrColIndC,
                                                    size_t*                   pBufferSizeInBytes);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDpruneCsr2csr_bufferSize(hipsparseHandle_t         handle,
                                                    int                       m,
                                                    int                       n,
                                                    int                       nnzA,
                                                    const hipsparseMatDescr_t descrA,
                                                    const double*             csrValA,
                                                    const int*                csrRowPtrA,
                                                    const int*                csrColIndA,
                                                    const double*             threshold,
                                                    const hipsparseMatDescr_t descrC,
                                                    const double*             csrValC,
                                                    const int*                csrRowPtrC,
                                                    const int*                csrColIndC,
                                                    size_t*                   pBufferSizeInBytes);
/**@}*/

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
/*! \ingroup conv_module
 *  \brief Convert and prune sparse CSR matrix into a sparse CSR matrix
 *
 *  \details
 *  \p hipsparseXpruneCsr2csr_bufferSizeExt returns the size of the temporary buffer that
 *  is required by \ref hipsparseSpruneCsr2csrNnz "hipsparseXpruneCsr2csrNnz()" and 
 *  \ref hipsparseSpruneCsr2csr "hipsparseXpruneCsr2csr()". The temporary storage buffer 
 *  must be allocated by the user.
 *
 *  @param[in]
 *  handle             handle to the hipsparse library context queue.
 *  @param[in]
 *  m                  number of rows in the sparse CSR matrix.
 *  @param[in]
 *  n                  number of columns in the sparse CSR matrix.
 *  @param[in]
 *  nnzA               number of non-zeros in the sparse CSR matrix A.
 *  @param[in]
 *  descrA             descriptor of the sparse CSR matrix A. Currently, only
 *                     \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
 *  @param[in]
 *  csrValA            array of \p nnzA elements containing the values of the sparse CSR matrix A.
 *  @param[in]
 *  csrRowPtrA         array of \p m+1 elements that point to the start of every row of the
 *                     sparse CSR matrix A.
 *  @param[in]
 *  csrColIndA         array of \p nnzA elements containing the column indices of the sparse CSR matrix A.
 *  @param[in]
 *  threshold          pointer to the non-negative pruning threshold which can exist in either host or device memory.
 *  @param[in]
 *  descrC             descriptor of the sparse CSR matrix C. Currently, only
 *                     \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
 *  @param[in]
 *  csrValC            array of \p nnzC elements containing the values of the sparse CSR matrix C.
 *  @param[in]
 *  csrRowPtrC         array of \p m+1 elements that point to the start of every row of the
 *                     sparse CSR matrix C.
 *  @param[in]
 *  csrColIndC         array of \p nnzC elements containing the column indices of the sparse CSR matrix C.
 *  @param[out]
 *  pBufferSizeInBytes number of bytes of the temporary storage buffer required by hipsparseSpruneCsr2csrNnz(),
 *                     hipsparseDpruneCsr2csrNnz(), hipsparseSpruneCsr2csr(), and hipsparseDpruneCsr2csr().
 *
 *  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
 *  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle or \p pBufferSizeInBytes pointer is invalid.
 *  \retval     HIPSPARSE_STATUS_INTERNAL_ERROR an internal error occurred.
 */
/**@{*/
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpruneCsr2csr_bufferSizeExt(hipsparseHandle_t         handle,
                                                       int                       m,
                                                       int                       n,
                                                       int                       nnzA,
                                                       const hipsparseMatDescr_t descrA,
                                                       const float*              csrValA,
                                                       const int*                csrRowPtrA,
                                                       const int*                csrColIndA,
                                                       const float*              threshold,
                                                       const hipsparseMatDescr_t descrC,
                                                       const float*              csrValC,
                                                       const int*                csrRowPtrC,
                                                       const int*                csrColIndC,
                                                       size_t* pBufferSizeInBytes);

DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDpruneCsr2csr_bufferSizeExt(hipsparseHandle_t         handle,
                                                       int                       m,
                                                       int                       n,
                                                       int                       nnzA,
                                                       const hipsparseMatDescr_t descrA,
                                                       const double*             csrValA,
                                                       const int*                csrRowPtrA,
                                                       const int*                csrColIndA,
                                                       const double*             threshold,
                                                       const hipsparseMatDescr_t descrC,
                                                       const double*             csrValC,
                                                       const int*                csrRowPtrC,
                                                       const int*                csrColIndC,
                                                       size_t* pBufferSizeInBytes);
/**@}*/
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
/*! \ingroup conv_module
 *  \brief Convert and prune sparse CSR matrix into a sparse CSR matrix
 *
 *  \details
 *  \p hipsparseXpruneCsr2csrNnz computes the number of nonzero elements per row and the total
 *  number of nonzero elements in a sparse CSR matrix once elements less than the threshold are
 *  pruned from the matrix.
 *
 *  \note The routine does support asynchronous execution if the pointer mode is set to device.
 *
 *  @param[in]
 *  handle             handle to the hipsparse library context queue.
 *  @param[in]
 *  m                  number of rows in the sparse CSR matrix.
 *  @param[in]
 *  n                  number of columns in the sparse CSR matrix.
 *  @param[in]
 *  nnzA               number of non-zeros in the sparse CSR matrix A.
 *  @param[in]
 *  descrA             descriptor of the sparse CSR matrix A. Currently, only
 *                     \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
 *  @param[in]
 *  csrValA            array of \p nnzA elements containing the values of the sparse CSR matrix A.
 *  @param[in]
 *  csrRowPtrA         array of \p m+1 elements that point to the start of every row of the
 *                     sparse CSR matrix A.
 *  @param[in]
 *  csrColIndA         array of \p nnzA elements containing the column indices of the sparse CSR matrix A.
 *  @param[in]
 *  threshold          pointer to the non-negative pruning threshold which can exist in either host or device memory.
 *  @param[in]
 *  descrC             descriptor of the sparse CSR matrix C. Currently, only
 *                     \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
 *  @param[out]
 *  csrRowPtrC         array of \p m+1 elements that point to the start of every row of the
 *                     sparse CSR matrix C.
 *  @param[out]
 *  nnzTotalDevHostPtr total number of nonzero elements in device or host memory.
 *  @param[out]
 *  buffer             buffer allocated by the user whose size is determined by calling \ref hipsparseSpruneCsr2csr_bufferSize 
 *                     "hipsparseXpruneCsr2csr_bufferSize()".
 *
 *  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
 *  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p m, \p n, \p nnzA, \p threshold, \p descrA, 
 *              \p descrC, \p csrValA, \p csrRowPtrA, \p csrColIndA, \p csrRowPtrC, \p nnzTotalDevHostPtr
 *              or \p buffer pointer is invalid.
*/
/**@{*/
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpruneCsr2csrNnz(hipsparseHandle_t         handle,
                                            int                       m,
                                            int                       n,
                                            int                       nnzA,
                                            const hipsparseMatDescr_t descrA,
                                            const float*              csrValA,
                                            const int*                csrRowPtrA,
                                            const int*                csrColIndA,
                                            const float*              threshold,
                                            const hipsparseMatDescr_t descrC,
                                            int*                      csrRowPtrC,
                                            int*                      nnzTotalDevHostPtr,
                                            void*                     buffer);

DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDpruneCsr2csrNnz(hipsparseHandle_t         handle,
                                            int                       m,
                                            int                       n,
                                            int                       nnzA,
                                            const hipsparseMatDescr_t descrA,
                                            const double*             csrValA,
                                            const int*                csrRowPtrA,
                                            const int*                csrColIndA,
                                            const double*             threshold,
                                            const hipsparseMatDescr_t descrC,
                                            int*                      csrRowPtrC,
                                            int*                      nnzTotalDevHostPtr,
                                            void*                     buffer);
/**@}*/
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
/*! \ingroup conv_module
 *  \brief Convert and prune sparse CSR matrix into a sparse CSR matrix
 *
 *  \details
 *  This function converts the sparse CSR matrix A into a sparse CSR matrix C by pruning values in A
 *  that are less than the threshold. All the parameters are assumed to have been pre-allocated by the user.
 *  The user first calls \ref hipsparseSpruneCsr2csr_bufferSize "hipsparseXpruneCsr2csr_bufferSize()" to 
 *  determine the size of the buffer used by \ref hipsparseSpruneCsr2csrNnz "hipsparseXpruneCsr2csrNnz()" 
 *  and \p hipsparseXpruneCsr2csr() which the user then allocates. The user then allocates \p csrRowPtrC to 
 *  have \p m+1 elements and then calls hipsparseXpruneCsr2csrNnz() which fills in the \p csrRowPtrC array 
 *  stores then number of elements that are larger than the pruning \p threshold in \p nnzTotalDevHostPtr. 
 *  The user then calls \p hipsparseXpruneCsr2csr() to complete the conversion. It is executed asynchronously 
 *  with respect to the host and may return control to the application on the host before the entire result is ready.
 *
 *  @param[in]
 *  handle        handle to the hipsparse library context queue.
 *  @param[in]
 *  m             number of rows in the sparse CSR matrix.
 *  @param[in]
 *  n             number of columns in the sparse CSR matrix.
 *  @param[in]
 *  nnzA          number of non-zeros in the sparse CSR matrix A.
 *  @param[in]
 *  descrA        descriptor of the sparse CSR matrix A. Currently, only
 *                \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
 *  @param[in]
 *  csrValA       array of \p nnzA elements containing the values of the sparse CSR matrix A.
 *  @param[in]
 *  csrRowPtrA    array of \p m+1 elements that point to the start of every row of the
 *                sparse CSR matrix A.
 *  @param[in]
 *  csrColIndA    array of \p nnzA elements containing the column indices of the sparse CSR matrix A.
 *  @param[in]
 *  threshold     pointer to the non-negative pruning threshold which can exist in either host or device memory.
 *  @param[in]
 *  descrC        descriptor of the sparse CSR matrix C. Currently, only
 *                \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
 *  @param[out]
 *  csrValC       array of \p nnzC elements containing the values of the sparse CSR matrix C.
 *  @param[in]
 *  csrRowPtrC    array of \p m+1 elements that point to the start of every row of the
 *                sparse CSR matrix C.
 *  @param[out]
 *  csrColIndC    array of \p nnzC elements containing the column indices of the sparse CSR matrix C.
 *  @param[in]
 *  buffer        buffer allocated by the user whose size is determined by calling \ref hipsparseSpruneCsr2csr_bufferSize 
 *                "hipsparseXpruneCsr2csr_bufferSize()".
 *
 *  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
 *  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p m, \p n, \p nnzA, \p threshold, \p descrA, \p descrC, \p csrValA,
 *              \p csrRowPtrA, \p csrcolindA, \p csrvalC, \p csrrowptrC, \p csrcolIndC or \p buffer pointer is invalid.
 */
/**@{*/
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpruneCsr2csr(hipsparseHandle_t         handle,
                                         int                       m,
                                         int                       n,
                                         int                       nnzA,
                                         const hipsparseMatDescr_t descrA,
                                         const float*              csrValA,
                                         const int*                csrRowPtrA,
                                         const int*                csrColIndA,
                                         const float*              threshold,
                                         const hipsparseMatDescr_t descrC,
                                         float*                    csrValC,
                                         const int*                csrRowPtrC,
                                         int*                      csrColIndC,
                                         void*                     buffer);

DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDpruneCsr2csr(hipsparseHandle_t         handle,
                                         int                       m,
                                         int                       n,
                                         int                       nnzA,
                                         const hipsparseMatDescr_t descrA,
                                         const double*             csrValA,
                                         const int*                csrRowPtrA,
                                         const int*                csrColIndA,
                                         const double*             threshold,
                                         const hipsparseMatDescr_t descrC,
                                         double*                   csrValC,
                                         const int*                csrRowPtrC,
                                         int*                      csrColIndC,
                                         void*                     buffer);
/**@}*/
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
/*! \ingroup conv_module
 *  \brief Convert and prune by percentage a sparse CSR matrix into a sparse CSR matrix
 *
 *  \details
 *  \p hipsparseXpruneCsr2csrByPercentage_bufferSize returns the size of the temporary buffer that
 *  is required by \ref hipsparseSpruneCsr2csrNnzByPercentage "hipsparseXpruneCsr2csrNnzByPercentage()".
 *  The temporary storage buffer must be allocated by the user.
 *
 *  @param[in]
 *  handle              handle to the hipsparse library context queue.
 *  @param[in]
 *  m                   number of rows in the sparse CSR matrix.
 *  @param[in]
 *  n                   number of columns in the sparse CSR matrix.
 *  @param[in]
 *  nnzA                number of non-zeros in the sparse CSR matrix A.
 *  @param[in]
 *  descrA              descriptor of the sparse CSR matrix A. Currently, only
 *                      \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
 *  @param[in]
 *  csrValA             array of \p nnzA elements containing the values of the sparse CSR matrix A.
 *  @param[in]
 *  csrRowPtrA          array of \p m+1 elements that point to the start of every row of the
 *                      sparse CSR matrix A.
 *  @param[in]
 *  csrColIndA          array of \p nnzA elements containing the column indices of the sparse CSR matrix A.
 *  @param[in]
 *  percentage          \p percentage>=0 and \p percentage<=100.
 *  @param[in]
 *  descrC              descriptor of the sparse CSR matrix C. Currently, only
 *                      \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
 *  @param[in]
 *  csrValC             array of \p nnzC elements containing the values of the sparse CSR matrix C.
 *  @param[in]
 *  csrRowPtrC          array of \p m+1 elements that point to the start of every row of the
 *                      sparse CSR matrix C.
 *  @param[in]
 *  csrColIndC          array of \p nnzC elements containing the column indices of the sparse CSR matrix C.
 *  @param[in]
 *  info                prune info structure.
 *  @param[out]
 *  pBufferSizeInBytes  number of bytes of the temporary storage buffer required by hipsparseSpruneCsr2csrNnzByPercentage(),
 *                      hipsparseDpruneCsr2csrNnzByPercentage(), hipsparseSpruneCsr2csrByPercentage(),
 *                      and hipsparseDpruneCsr2csrByPercentage().
 *
 *  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
 *  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle or \p pBufferSizeInBytes pointer is invalid.
 *  \retval     HIPSPARSE_STATUS_INTERNAL_ERROR an internal error occurred.
 */
/**@{*/
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpruneCsr2csrByPercentage_bufferSize(hipsparseHandle_t         handle,
                                                                int                       m,
                                                                int                       n,
                                                                int                       nnzA,
                                                                const hipsparseMatDescr_t descrA,
                                                                const float*              csrValA,
                                                                const int* csrRowPtrA,
                                                                const int* csrColIndA,
                                                                float      percentage,
                                                                const hipsparseMatDescr_t descrC,
                                                                const float*              csrValC,
                                                                const int*  csrRowPtrC,
                                                                const int*  csrColIndC,
                                                                pruneInfo_t info,
                                                                size_t*     pBufferSizeInBytes);
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDpruneCsr2csrByPercentage_bufferSize(hipsparseHandle_t         handle,
                                                                int                       m,
                                                                int                       n,
                                                                int                       nnzA,
                                                                const hipsparseMatDescr_t descrA,
                                                                const double*             csrValA,
                                                                const int* csrRowPtrA,
                                                                const int* csrColIndA,
                                                                double     percentage,
                                                                const hipsparseMatDescr_t descrC,
                                                                const double*             csrValC,
                                                                const int*  csrRowPtrC,
                                                                const int*  csrColIndC,
                                                                pruneInfo_t info,
                                                                size_t*     pBufferSizeInBytes);
/**@}*/
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
/*! \ingroup conv_module
 *  \brief Convert and prune by percentage a sparse CSR matrix into a sparse CSR matrix
 *
 *  \details
 *  \p hipsparseXpruneCsr2csrByPercentage_bufferSizeExt returns the size of the temporary buffer that
 *  is required by \ref hipsparseSpruneCsr2csrNnzByPercentage "hipsparseXpruneCsr2csrNnzByPercentage()".
 *  The temporary storage buffer must be allocated by the user.
 *
 *  @param[in]
 *  handle              handle to the hipsparse library context queue.
 *  @param[in]
 *  m                   number of rows in the sparse CSR matrix.
 *  @param[in]
 *  n                   number of columns in the sparse CSR matrix.
 *  @param[in]
 *  nnzA                number of non-zeros in the sparse CSR matrix A.
 *  @param[in]
 *  descrA              descriptor of the sparse CSR matrix A. Currently, only
 *                      \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
 *  @param[in]
 *  csrValA             array of \p nnzA elements containing the values of the sparse CSR matrix A.
 *  @param[in]
 *  csrRowPtrA          array of \p m+1 elements that point to the start of every row of the
 *                      sparse CSR matrix A.
 *  @param[in]
 *  csrColIndA          array of \p nnzA elements containing the column indices of the sparse CSR matrix A.
 *  @param[in]
 *  percentage          \p percentage>=0 and \p percentage<=100.
 *  @param[in]
 *  descrC              descriptor of the sparse CSR matrix C. Currently, only
 *                      \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
 *  @param[in]
 *  csrValC             array of \p nnzC elements containing the values of the sparse CSR matrix C.
 *  @param[in]
 *  csrRowPtrC          array of \p m+1 elements that point to the start of every row of the
 *                      sparse CSR matrix C.
 *  @param[in]
 *  csrColIndC          array of \p nnzC elements containing the column indices of the sparse CSR matrix C.
 *  @param[in]
 *  info                prune info structure.
 *  @param[out]
 *  pBufferSizeInBytes  number of bytes of the temporary storage buffer required by hipsparseSpruneCsr2csrNnzByPercentage(),
 *                      hipsparseDpruneCsr2csrNnzByPercentage(), hipsparseSpruneCsr2csrByPercentage(),
 *                      and hipsparseDpruneCsr2csrByPercentage().
 *
 *  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
 *  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle or \p pBufferSizeInBytes pointer is invalid.
 *  \retval     HIPSPARSE_STATUS_INTERNAL_ERROR an internal error occurred.
 */
/**@{*/
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpruneCsr2csrByPercentage_bufferSizeExt(hipsparseHandle_t         handle,
                                                                   int                       m,
                                                                   int                       n,
                                                                   int                       nnzA,
                                                                   const hipsparseMatDescr_t descrA,
                                                                   const float* csrValA,
                                                                   const int*   csrRowPtrA,
                                                                   const int*   csrColIndA,
                                                                   float        percentage,
                                                                   const hipsparseMatDescr_t descrC,
                                                                   const float* csrValC,
                                                                   const int*   csrRowPtrC,
                                                                   const int*   csrColIndC,
                                                                   pruneInfo_t  info,
                                                                   size_t*      pBufferSizeInBytes);

DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDpruneCsr2csrByPercentage_bufferSizeExt(hipsparseHandle_t         handle,
                                                                   int                       m,
                                                                   int                       n,
                                                                   int                       nnzA,
                                                                   const hipsparseMatDescr_t descrA,
                                                                   const double* csrValA,
                                                                   const int*    csrRowPtrA,
                                                                   const int*    csrColIndA,
                                                                   double        percentage,
                                                                   const hipsparseMatDescr_t descrC,
                                                                   const double* csrValC,
                                                                   const int*    csrRowPtrC,
                                                                   const int*    csrColIndC,
                                                                   pruneInfo_t   info,
                                                                   size_t* pBufferSizeInBytes);
/**@}*/
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
/*! \ingroup conv_module
 *  \brief Convert and prune by percentage a sparse CSR matrix into a sparse CSR matrix
 *
 *  \details
 *  \p hipsparseXpruneCsr2csrNnzByPercentage computes the number of nonzero elements per row and the total
 *  number of nonzero elements in a sparse CSR matrix once elements less than the threshold are
 *  pruned from the matrix.
 *
 *  \note The routine does support asynchronous execution if the pointer mode is set to device.
 *
 *  @param[in]
 *  handle             handle to the hipsparse library context queue.
 *  @param[in]
 *  m                  number of rows in the sparse CSR matrix.
 *  @param[in]
 *  n                  number of columns in the sparse CSR matrix.
 *  @param[in]
 *  nnzA               number of non-zeros in the sparse CSR matrix A.
 *  @param[in]
 *  descrA             descriptor of the sparse CSR matrix A. Currently, only
 *                     \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
 *  @param[in]
 *  csrValA            array of \p nnzA elements containing the values of the sparse CSR matrix A.
 *  @param[in]
 *  csrRowPtrA         array of \p m+1 elements that point to the start of every row of the
 *                     sparse CSR matrix A.
 *  @param[in]
 *  csrColIndA         array of \p nnzA elements containing the column indices of the sparse CSR matrix A.
 *  @param[in]
 *  percentage         \p percentage>=0 and \p percentage<=100.
 *  @param[in]
 *  descrC             descriptor of the sparse CSR matrix C. Currently, only
 *                     \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
 *  @param[out]
 *  csrRowPtrC         array of \p m+1 elements that point to the start of every row of the
 *                     sparse CSR matrix C.
 *  @param[out]
 *  nnzTotalDevHostPtr total number of nonzero elements in device or host memory.
 *  @param[in]
 *  info               prune info structure.
 *  @param[out]
 *  buffer             buffer allocated by the user whose size is determined by calling
 *                     \ref hipsparseSpruneCsr2csrByPercentage_bufferSize "hipsparseXpruneCsr2csrByPercentage_bufferSize()".
 *
 *  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
 *  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p m, \p n, \p nnzA, \p percentage, \p descrA, \p descrC, 
 *              \p info, \p csrValA, \p csrRowPtrA, \p csrColIndA, \p csrRowPtrC, \p nnzTotalDevHostPtr or \p buffer 
 *              pointer is invalid.
 */
/**@{*/
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpruneCsr2csrNnzByPercentage(hipsparseHandle_t         handle,
                                                        int                       m,
                                                        int                       n,
                                                        int                       nnzA,
                                                        const hipsparseMatDescr_t descrA,
                                                        const float*              csrValA,
                                                        const int*                csrRowPtrA,
                                                        const int*                csrColIndA,
                                                        float                     percentage,
                                                        const hipsparseMatDescr_t descrC,
                                                        int*                      csrRowPtrC,
                                                        int*        nnzTotalDevHostPtr,
                                                        pruneInfo_t info,
                                                        void*       buffer);

DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDpruneCsr2csrNnzByPercentage(hipsparseHandle_t         handle,
                                                        int                       m,
                                                        int                       n,
                                                        int                       nnzA,
                                                        const hipsparseMatDescr_t descrA,
                                                        const double*             csrValA,
                                                        const int*                csrRowPtrA,
                                                        const int*                csrColIndA,
                                                        double                    percentage,
                                                        const hipsparseMatDescr_t descrC,
                                                        int*                      csrRowPtrC,
                                                        int*        nnzTotalDevHostPtr,
                                                        pruneInfo_t info,
                                                        void*       buffer);
/**@}*/
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
/*! \ingroup conv_module
 *  \brief Convert and prune by percentage a sparse CSR matrix into a sparse CSR matrix
 *
 *  \details
 *  This function converts the sparse CSR matrix A into a sparse CSR matrix C by pruning values in A
 *  that are less than the threshold. All the parameters are assumed to have been pre-allocated by the user.
 *  The user first calls \ref hipsparseSpruneCsr2csr_bufferSize "hipsparseXpruneCsr2csr_bufferSize()" to 
 *  determine the size of the buffer used by \ref hipsparseSpruneCsr2csrNnz "hipsparseXpruneCsr2csrNnz()" and 
 *  \p hipsparseXpruneCsr2csr() which the user then allocates. The user then allocates \p csrRowPtrC to have 
 *  \p m+1 elements and then calls \ref hipsparseSpruneCsr2csrNnz "hipsparseXpruneCsr2csrNnz()" which fills
 *  in the \p csrRowPtrC array stores then number of elements that are larger than the pruning \p threshold
 *  in \p nnzTotalDevHostPtr. The user then calls \p hipsparseXpruneCsr2csr() to complete the conversion. It
 *  is executed asynchronously with respect to the host and may return control to the application on the host
 *  before the entire result is ready.
 *
 *  @param[in]
 *  handle        handle to the hipsparse library context queue.
 *  @param[in]
 *  m             number of rows in the sparse CSR matrix.
 *  @param[in]
 *  n             number of columns in the sparse CSR matrix.
 *  @param[in]
 *  nnzA          number of non-zeros in the sparse CSR matrix A.
 *  @param[in]
 *  descrA        descriptor of the sparse CSR matrix A. Currently, only
 *                \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
 *  @param[in]
 *  csrValA       array of \p nnzA elements containing the values of the sparse CSR matrix A.
 *  @param[in]
 *  csrRowPtrA    array of \p m+1 elements that point to the start of every row of the
 *                sparse CSR matrix A.
 *  @param[in]
 *  csrColIndA    array of \p nnzA elements containing the column indices of the sparse CSR matrix A.
 *  @param[in]
 *  percentage    \p percentage>=0 and \p percentage<=100.
 *  @param[in]
 *  descrC        descriptor of the sparse CSR matrix C. Currently, only
 *                \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
 *  @param[out]
 *  csrValC       array of \p nnz_C elements containing the values of the sparse CSR matrix C.
 *  @param[in]
 *  csrRowPtrC    array of \p m+1 elements that point to the start of every row of the
 *                sparse CSR matrix C.
 *  @param[out]
 *  csrColIndC    array of \p nnz_C elements containing the column indices of the sparse CSR matrix C.
 *  @param[in]
 *  info          prune info structure.
 *  @param[in]
 *  buffer        buffer allocated by the user whose size is determined by calling 
 *                \ref hipsparseSpruneCsr2csrByPercentage_bufferSize "hipsparseXpruneCsr2csrByPercentage_bufferSize()".
 *
 *  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
 *  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p m, \p n, \p nnzA, \p percentage, \p descrA, \p descrC, \p info, 
 *              \p csrValA, \p csrRowPtrA, \p csrColIndA, \p csrValC, \p csrRowPtrC, \p csrColIndC or \p buffer pointer is 
 *              invalid.
 */
/**@{*/
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpruneCsr2csrByPercentage(hipsparseHandle_t         handle,
                                                     int                       m,
                                                     int                       n,
                                                     int                       nnzA,
                                                     const hipsparseMatDescr_t descrA,
                                                     const float*              csrValA,
                                                     const int*                csrRowPtrA,
                                                     const int*                csrColIndA,
                                                     float                     percentage,
                                                     const hipsparseMatDescr_t descrC,
                                                     float*                    csrValC,
                                                     const int*                csrRowPtrC,
                                                     int*                      csrColIndC,
                                                     pruneInfo_t               info,
                                                     void*                     buffer);

DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDpruneCsr2csrByPercentage(hipsparseHandle_t         handle,
                                                     int                       m,
                                                     int                       n,
                                                     int                       nnzA,
                                                     const hipsparseMatDescr_t descrA,
                                                     const double*             csrValA,
                                                     const int*                csrRowPtrA,
                                                     const int*                csrColIndA,
                                                     double                    percentage,
                                                     const hipsparseMatDescr_t descrC,
                                                     double*                   csrValC,
                                                     const int*                csrRowPtrC,
                                                     int*                      csrColIndC,
                                                     pruneInfo_t               info,
                                                     void*                     buffer);
/**@}*/
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 11000)
/*! \ingroup conv_module
*  \brief Convert a sparse HYB matrix into a sparse CSR matrix
*
*  \details
*  \p hipsparseXhyb2csr converts a HYB matrix into a CSR matrix.
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*
*  @param[in]
*  handle            handle to the hipsparse library context queue.
*  @param[in]
*  descrA            descriptor of the sparse HYB matrix. Currently, only
*                    \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*  @param[in]
*  hybA              sparse matrix in HYB format.
*  @param[out]
*  csrSortedValA     array containing the values of the sparse CSR matrix.
*  @param[out]
*  csrSortedRowPtrA  array of \p m+1 elements that point to the start of every row of the
*                    sparse CSR matrix.
*  @param[out]
*  csrSortedColIndA  array containing the column indices of the sparse CSR matrix.
*
*  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p descrA, \p hybA, \p csrSortedValA,
*              \p csrSortedRowPtrA or \p csrSortedColIndA pointer is invalid.
*  \retval     HIPSPARSE_STATUS_INTERNAL_ERROR an internal error occurred.
*  \retval     HIPSPARSE_STATUS_NOT_SUPPORTED
*              \ref hipsparseMatrixType_t != \ref HIPSPARSE_MATRIX_TYPE_GENERAL.
*/
/**@{*/
DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseShyb2csr(hipsparseHandle_t         handle,
                                    const hipsparseMatDescr_t descrA,
                                    const hipsparseHybMat_t   hybA,
                                    float*                    csrSortedValA,
                                    int*                      csrSortedRowPtrA,
                                    int*                      csrSortedColIndA);
DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDhyb2csr(hipsparseHandle_t         handle,
                                    const hipsparseMatDescr_t descrA,
                                    const hipsparseHybMat_t   hybA,
                                    double*                   csrSortedValA,
                                    int*                      csrSortedRowPtrA,
                                    int*                      csrSortedColIndA);
DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseChyb2csr(hipsparseHandle_t         handle,
                                    const hipsparseMatDescr_t descrA,
                                    const hipsparseHybMat_t   hybA,
                                    hipComplex*               csrSortedValA,
                                    int*                      csrSortedRowPtrA,
                                    int*                      csrSortedColIndA);
DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZhyb2csr(hipsparseHandle_t         handle,
                                    const hipsparseMatDescr_t descrA,
                                    const hipsparseHybMat_t   hybA,
                                    hipDoubleComplex*         csrSortedValA,
                                    int*                      csrSortedRowPtrA,
                                    int*                      csrSortedColIndA);
/**@}*/
#endif

/*! \ingroup conv_module
*  \brief Convert a sparse COO matrix into a sparse CSR matrix
*
*  \details
*  \p hipsparseXcoo2csr converts the COO array containing the row indices into a
*  CSR array of row offsets, that point to the start of every row.
*  It is assumed that the COO row index array is sorted and that all arrays have been allocated 
*  prior to calling hipsparseXcoo2csr.
*
*  For example, given the COO row indices array:
*  \f[
*    \begin{align}
*    \text{cooRowInd} &= \begin{bmatrix} 0 & 0 & 1 & 2 & 2 & 4 & 4 & 4 \end{bmatrix}
*    \end{align}
*  \f]
*
*  the resulting CSR row pointer array after calling \p hipsparseXcoo2csr is:
*  \f[
*    \begin{align}
*    \text{csrRowPtr} &= \begin{bmatrix} 0 & 2 & 3 & 5 & 8 \end{bmatrix}
*    \end{align}
*  \f]
*
*  \note It can also be used, to convert a COO array containing the column indices into
*  a CSC array of column offsets, that point to the start of every column. Then, it is
*  assumed that the COO column index array is sorted, instead.
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*
*  @param[in]
*  handle      handle to the hipsparse library context queue.
*  @param[in]
*  cooRowInd   array of \p nnz elements containing the row indices of the sparse COO
*              matrix.
*  @param[in]
*  nnz         number of non-zero entries of the sparse CSR matrix.
*  @param[in]
*  m           number of rows of the sparse CSR matrix.
*  @param[out]
*  csrRowPtr   array of \p m+1 elements that point to the start of every row of the
*              sparse CSR matrix.
*  @param[in]
*  idxBase    \ref HIPSPARSE_INDEX_BASE_ZERO or \ref HIPSPARSE_INDEX_BASE_ONE.
*
*  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p m, \p nnz, \p cooRowInd or \p csrRowPtr
*              pointer is invalid.
*
*  \par Example
*  \code{.c}
*    // hipSPARSE handle
*    hipsparseHandle_t handle;
*    hipsparseCreate(&handle);
*
*    // Sparse matrix in COO format
*    //     1 2 0 3 0
*    // A = 0 4 5 0 0
*    //     6 0 0 7 8
*    int hcooRowInd[8] = {0, 0, 0, 1, 1, 2, 2, 2};
*    int hcooColInd[8] = {0, 1, 3, 1, 2, 0, 3, 4};
*    float hcooVal[8]   = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f}; 
*
*    int m         = 3;
*    int n         = 5;
*    int nnz       = 8;
*    hipsparseIndexBase_t base = HIPSPARSE_INDEX_BASE_ZERO;
*
*    int* dcooRowInd = nullptr;
*    int* dcooColInd = nullptr;
*    hipMalloc((void**)&dcooRowInd, sizeof(int) * nnz);
*    hipMalloc((void**)&dcooColInd, sizeof(int) * nnz);
*
*    hipMemcpy(dcooRowInd, hcooRowInd, sizeof(int) * nnz, hipMemcpyHostToDevice);
*    hipMemcpy(dcooColInd, hcooColInd, sizeof(int) * nnz, hipMemcpyHostToDevice);
*
*    int* dcsrRowPtr = nullptr;
*    hipMalloc((void**)&dcsrRowPtr, sizeof(int) * (m + 1));
*
*    hipsparseXcoo2csr(handle, dcooRowInd, nnz, m, dcsrRowPtr, base);
*
*    hipFree(dcooRowInd);
*    hipFree(dcooColInd);
*    
*    hipFree(dcsrRowPtr);
*
*    hipsparseDestroy(handle);
*  \endcode
*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseXcoo2csr(hipsparseHandle_t    handle,
                                    const int*           cooRowInd,
                                    int                  nnz,
                                    int                  m,
                                    int*                 csrRowPtr,
                                    hipsparseIndexBase_t idxBase);

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
/*! \ingroup conv_module
*  \brief Create the identity map
*
*  \details
*  \p hipsparseCreateIdentityPermutation stores the identity map in \p p, such that
*  \f$p = 0:1:(n-1)\f$.
*
*  \code{.c}
*      for(i = 0; i < n; ++i)
*      {
*          p[i] = i;
*      }
*  \endcode
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*
*  @param[in]
*  handle      handle to the hipsparse library context queue.
*  @param[in]
*  n           size of the map \p p.
*  @param[out]
*  p           array of \p n integers containing the map.
*
*  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p n or \p p pointer is invalid.
*
*  \par Example
*  \code{.c}
*    // hipSPARSE handle
*    hipsparseHandle_t handle;
*    hipsparseCreate(&handle);
*
*    int n = 10;
*
*    int* dperm = nullptr;
*    hipMalloc((void**)&dperm, sizeof(int) * n);
*
*    hipsparseCreateIdentityPermutation(handle, n, dperm);
*
*    hipsparseDestroy(handle);
*  \endcode
*/
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCreateIdentityPermutation(hipsparseHandle_t handle, int n, int* p);
#endif

/*! \ingroup conv_module
*  \brief Sort a sparse CSR matrix
*
*  \details
*  \p hipsparseXcsrsort_bufferSizeExt returns the size of the temporary storage buffer
*  in bytes required by hipsparseXcsrsort(). The temporary storage buffer must be allocated by
*  the user.
*
*  @param[in]
*  handle              handle to the hipsparse library context queue.
*  @param[in]
*  m                   number of rows of the sparse CSR matrix.
*  @param[in]
*  n                   number of columns of the sparse CSR matrix.
*  @param[in]
*  nnz                 number of non-zero entries of the sparse CSR matrix.
*  @param[in]
*  csrRowPtr           array of \p m+1 elements that point to the start of every row of the
*                      sparse CSR matrix.
*  @param[in]
*  csrColInd           array of \p nnz elements containing the column indices of the sparse
*                      CSR matrix.
*  @param[out]
*  pBufferSizeInBytes  number of bytes of the temporary storage buffer required by
*                      \ref hipsparseXcsrsort().
*
*  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p m, \p n, \p nnz, \p csrRowPtr, \p csrColInd or
*              \p pBufferSizeInBytes pointer is invalid.
*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseXcsrsort_bufferSizeExt(hipsparseHandle_t handle,
                                                  int               m,
                                                  int               n,
                                                  int               nnz,
                                                  const int*        csrRowPtr,
                                                  const int*        csrColInd,
                                                  size_t*           pBufferSizeInBytes);

/*! \ingroup conv_module
*  \brief Sort a sparse CSR matrix
*
*  \details
*  \p hipsparseXcsrsort sorts a matrix in CSR format. The sorted permutation vector
*  \p P can be used to obtain sorted \p csrVal array. In this case, \p P must be
*  initialized as the identity permutation, see \ref hipsparseCreateIdentityPermutation(). To 
*  apply the permutation vector to the CSR values, see hipsparse \ref hipsparseSgthr 
*  "hipsparseXgthr()".
*
*  \p hipsparseXcsrsort requires extra temporary storage buffer that has to be allocated by
*  the user. Storage buffer size can be determined by hipsparseXcsrsort_bufferSizeExt().
*
*  \note
*  \p P can be \p NULL if a sorted permutation vector is not required.
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*
*  @param[in]
*  handle          handle to the hipsparse library context queue.
*  @param[in]
*  m               number of rows of the sparse CSR matrix.
*  @param[in]
*  n               number of columns of the sparse CSR matrix.
*  @param[in]
*  nnz             number of non-zero entries of the sparse CSR matrix.
*  @param[in]
*  descrA          descriptor of the sparse CSR matrix. Currently, only
*                  \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*  @param[in]
*  csrRowPtr       array of \p m+1 elements that point to the start of every row of the
*                  sparse CSR matrix.
*  @param[inout]
*  csrColInd       array of \p nnz elements containing the column indices of the sparse
*                  CSR matrix.
*  @param[inout]
*  P               array of \p nnz integers containing the unsorted map indices, can be
*                  \p NULL.
*  @param[in]
*  pBuffer         temporary storage buffer allocated by the user, size is returned by
*                  \ref hipsparseXcsrsort_bufferSizeExt().
*
*  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p m, \p n, \p nnz, \p descrA, \p csrRowPtr, 
*              \p csrColInd or \p pBuffer pointer is invalid.
*  \retval     HIPSPARSE_STATUS_INTERNAL_ERROR an internal error occurred.
*  \retval     HIPSPARSE_STATUS_NOT_SUPPORTED
*              \ref hipsparseMatrixType_t != \ref HIPSPARSE_MATRIX_TYPE_GENERAL.
*
*  \par Example
*  \code{.c}
*    // hipSPARSE handle
*    hipsparseHandle_t handle;
*    hipsparseCreate(&handle);
*
*    hipsparseMatDescr_t descr;
*    hipsparseCreateMatDescr(&descr);
*
*    // Sparse matrix in CSR format (columns unsorted)
*    //     1 2 0 3 0
*    // A = 0 4 5 0 0
*    //     6 0 0 7 8
*    int hcsrRowPtr[4] = {0, 3, 5, 8};
*    int hcsrColInd[8] = {3, 1, 0, 2, 1, 0, 4, 3};
*    float hcsrVal[8]  = {3.0f, 2.0f, 1.0f, 5.0f, 4.0f, 6.0f, 8.0f, 7.0f}; 
*
*    int m         = 3;
*    int n         = 5;
*    int nnz       = 8;
*
*    int* dcsrRowPtr = nullptr;
*    int* dcsrColInd = nullptr;
*    float* dcsrVal = nullptr;
*    hipMalloc((void**)&dcsrRowPtr, sizeof(int) * (m + 1));
*    hipMalloc((void**)&dcsrColInd, sizeof(int) * nnz);
*    hipMalloc((void**)&dcsrVal, sizeof(float) * nnz);
*
*    hipMemcpy(dcsrRowPtr, hcsrRowPtr, sizeof(int) * (m + 1), hipMemcpyHostToDevice);
*    hipMemcpy(dcsrColInd, hcsrColInd, sizeof(int) * nnz, hipMemcpyHostToDevice);
*    hipMemcpy(dcsrVal, hcsrVal, sizeof(float) * nnz, hipMemcpyHostToDevice);
*
*    size_t bufferSize;
*    hipsparseXcsrsort_bufferSizeExt(handle,
*                                    m,
*                                    n,
*                                    nnz,
*                                    dcsrRowPtr,
*                                    dcsrColInd,
*                                    &bufferSize);
*
*    void* dbuffer = nullptr;
*    hipMalloc((void**)&dbuffer, bufferSize);
*
*    int* dperm = nullptr;
*    hipMalloc((void**)&dperm, sizeof(int) * nnz);
*    hipsparseCreateIdentityPermutation(handle, nnz, dperm);
*
*    hipsparseXcsrsort(handle,
*                      m,
*                      n,
*                      nnz,
*                      descr,
*                      dcsrRowPtr,
*                      dcsrColInd,
*                      dperm,
*                      dbuffer);
*
*    float* dcsrValSorted = nullptr;
*    hipMalloc((void**)&dcsrValSorted, sizeof(float) * nnz);
*    hipsparseSgthr(handle, nnz, dcsrVal, dcsrValSorted, dperm, HIPSPARSE_INDEX_BASE_ZERO);
*
*    hipFree(dcsrRowPtr);
*    hipFree(dcsrColInd);
*    hipFree(dcsrVal);
*    hipFree(dcsrValSorted);
*
*    hipFree(dbuffer);
*    hipFree(dperm);
*
*    hipsparseDestroyMatDescr(descr);
*    hipsparseDestroy(handle);
*  \endcode
*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseXcsrsort(hipsparseHandle_t         handle,
                                    int                       m,
                                    int                       n,
                                    int                       nnz,
                                    const hipsparseMatDescr_t descrA,
                                    const int*                csrRowPtr,
                                    int*                      csrColInd,
                                    int*                      P,
                                    void*                     pBuffer);

/*! \ingroup conv_module
*  \brief Sort a sparse CSC matrix
*
*  \details
*  \p hipsparseXcscsort_bufferSizeExt returns the size of the temporary storage buffer
*  in bytes required by hipsparseXcscsort(). The temporary storage buffer must be 
*  allocated by the user.
*
*  @param[in]
*  handle              handle to the hipsparse library context queue.
*  @param[in]
*  m                   number of rows of the sparse CSC matrix.
*  @param[in]
*  n                   number of columns of the sparse CSC matrix.
*  @param[in]
*  nnz                 number of non-zero entries of the sparse CSC matrix.
*  @param[in]
*  cscColPtr           array of \p n+1 elements that point to the start of every column of
*                      the sparse CSC matrix.
*  @param[in]
*  cscRowInd           array of \p nnz elements containing the row indices of the sparse
*                      CSC matrix.
*  @param[out]
*  pBufferSizeInBytes  number of bytes of the temporary storage buffer required by
*                      \ref hipsparseXcscsort().
*
*  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p m, \p n, \p nnz, \p cscColPtr, \p cscRowInd or
*              \p pBufferSizeInBytes pointer is invalid.
*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseXcscsort_bufferSizeExt(hipsparseHandle_t handle,
                                                  int               m,
                                                  int               n,
                                                  int               nnz,
                                                  const int*        cscColPtr,
                                                  const int*        cscRowInd,
                                                  size_t*           pBufferSizeInBytes);

/*! \ingroup conv_module
*  \brief Sort a sparse CSC matrix
*
*  \details
*  \p hipsparseXcscsort sorts a matrix in CSC format. The sorted permutation vector
*  \p P can be used to obtain sorted \p cscVal array. In this case, \p P must be
*  initialized as the identity permutation, see \ref hipsparseCreateIdentityPermutation(). To 
*  apply the permutation vector to the CSC values, see hipsparse \ref hipsparseSgthr 
*  "hipsparseXgthr()".
*
*  \p hipsparseXcscsort requires extra temporary storage buffer that has to be allocated by
*  the user. Storage buffer size can be determined by \ref hipsparseXcscsort_bufferSizeExt().
*
*  \note
*  \p P can be \p NULL if a sorted permutation vector is not required.
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*
*  @param[in]
*  handle          handle to the hipsparse library context queue.
*  @param[in]
*  m               number of rows of the sparse CSC matrix.
*  @param[in]
*  n               number of columns of the sparse CSC matrix.
*  @param[in]
*  nnz             number of non-zero entries of the sparse CSC matrix.
*  @param[in]
*  descrA          descriptor of the sparse CSC matrix. Currently, only
*                  \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*  @param[in]
*  cscColPtr       array of \p n+1 elements that point to the start of every column of
*                  the sparse CSC matrix.
*  @param[inout]
*  cscRowInd       array of \p nnz elements containing the row indices of the sparse
*                  CSC matrix.
*  @param[inout]
*  P               array of \p nnz integers containing the unsorted map indices, can be
*                  \p NULL.
*  @param[in]
*  pBuffer         temporary storage buffer allocated by the user, size is returned by
*                  \ref hipsparseXcscsort_bufferSizeExt().
*
*  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p m, \p n, \p nnz, \p descrA, \p cscColPtr, 
*              \p cscRowInd or \p pBuffer pointer is invalid.
*  \retval     HIPSPARSE_STATUS_INTERNAL_ERROR an internal error occurred.
*  \retval     HIPSPARSE_STATUS_NOT_SUPPORTED
*              \ref hipsparseMatrixType_t != \ref HIPSPARSE_MATRIX_TYPE_GENERAL.
*
*  \par Example
*  \code{.c}
*    // hipSPARSE handle
*    hipsparseHandle_t handle;
*    hipsparseCreate(&handle);
*
*    hipsparseMatDescr_t descr;
*    hipsparseCreateMatDescr(&descr);
*
*    // Sparse matrix in CSC format (unsorted row indices)
*    //     1 2 0 3 0
*    // A = 0 4 5 0 0
*    //     6 0 0 7 8
*    int hcscRowInd[8] = {2, 0, 1, 0, 1, 2, 0, 2};
*    int hcscColPtr[6] = {0, 2, 4, 5, 7, 8};
*    float hcscVal[8]  = {6.0f, 1.0f, 4.0f, 2.0f, 5.0f, 7.0f, 3.0f, 8.0f}; 
*
*    int m         = 3;
*    int n         = 5;
*    int nnz       = 8;
*
*    int* dcscRowInd = nullptr;
*    int* dcscColPtr = nullptr;
*    float* dcscVal = nullptr;
*    hipMalloc((void**)&dcscRowInd, sizeof(int) * nnz);
*    hipMalloc((void**)&dcscColPtr, sizeof(int) * (n + 1));
*    hipMalloc((void**)&dcscVal, sizeof(float) * nnz);
*
*    hipMemcpy(dcscRowInd, hcscRowInd, sizeof(int) * nnz, hipMemcpyHostToDevice);
*    hipMemcpy(dcscColPtr, hcscColPtr, sizeof(int) * (n + 1), hipMemcpyHostToDevice);
*    hipMemcpy(dcscVal, hcscVal, sizeof(float) * nnz, hipMemcpyHostToDevice);
*
*    size_t bufferSize;
*    hipsparseXcscsort_bufferSizeExt(handle,
*                                    m,
*                                    n,
*                                    nnz,
*                                    dcscColPtr,
*                                    dcscRowInd,
*                                    &bufferSize);
*
*    void* dbuffer = nullptr;
*    hipMalloc((void**)&dbuffer, bufferSize);
*
*    int* dperm = nullptr;
*    hipMalloc((void**)&dperm, sizeof(int) * nnz);
*    hipsparseCreateIdentityPermutation(handle, nnz, dperm);
*
*    hipsparseXcscsort(handle,
*                      m,
*                      n,
*                      nnz,
*                      descr,
*                      dcscColPtr,
*                      dcscRowInd,
*                      dperm,
*                      dbuffer);
*
*    float* dcscValSorted = nullptr;
*    hipMalloc((void**)&dcscValSorted, sizeof(float) * nnz);
*    hipsparseSgthr(handle, nnz, dcscVal, dcscValSorted, dperm, HIPSPARSE_INDEX_BASE_ZERO);
*
*    hipFree(dcscRowInd);
*    hipFree(dcscColPtr);
*    hipFree(dcscVal);
*    hipFree(dcscValSorted);
*
*    hipFree(dbuffer);
*    hipFree(dperm);
*
*    hipsparseDestroyMatDescr(descr);
*    hipsparseDestroy(handle);
*  \endcode
*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseXcscsort(hipsparseHandle_t         handle,
                                    int                       m,
                                    int                       n,
                                    int                       nnz,
                                    const hipsparseMatDescr_t descrA,
                                    const int*                cscColPtr,
                                    int*                      cscRowInd,
                                    int*                      P,
                                    void*                     pBuffer);

/*! \ingroup conv_module
*  \brief Sort a sparse COO matrix
*
*  \details
*  \p hipsparseXcoosort_bufferSizeExt returns the size of the temporary storage buffer
*  in bytes required by \ref hipsparseXcoosortByRow() and \ref hipsparseXcoosortByColumn(). 
*  The temporary storage buffer must be allocated by the user.
*
*  @param[in]
*  handle              handle to the hipsparse library context queue.
*  @param[in]
*  m                   number of rows of the sparse COO matrix.
*  @param[in]
*  n                   number of columns of the sparse COO matrix.
*  @param[in]
*  nnz                 number of non-zero entries of the sparse COO matrix.
*  @param[in]
*  cooRows             array of \p nnz elements containing the row indices of the sparse
*                      COO matrix.
*  @param[in]
*  cooCols             array of \p nnz elements containing the column indices of the sparse
*                      COO matrix.
*  @param[out]
*  pBufferSizeInBytes  number of bytes of the temporary storage buffer required by
*                      hipsparseXcoosortByRow() and hipsparseXcoosortByColumn().
*
*  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p m, \p n, \p nnz, \p cooRows, 
*              \p cooCols or \p pBufferSizeInBytes pointer is invalid.
*  \retval     HIPSPARSE_STATUS_INTERNAL_ERROR an internal error occurred.
*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseXcoosort_bufferSizeExt(hipsparseHandle_t handle,
                                                  int               m,
                                                  int               n,
                                                  int               nnz,
                                                  const int*        cooRows,
                                                  const int*        cooCols,
                                                  size_t*           pBufferSizeInBytes);

/*! \ingroup conv_module
*  \brief Sort a sparse COO matrix by row
*
*  \details
*  \p hipsparseXcoosortByRow sorts a matrix in COO format by row. The sorted
*  permutation vector \p P can be used to obtain sorted \p cooVal array. In this
*  case, \p P must be initialized as the identity permutation, see
*  \ref hipsparseCreateIdentityPermutation(). To apply the permutation vector to the COO 
*  values, see hipsparse \ref hipsparseSgthr "hipsparseXgthr()".
*
*  \p hipsparseXcoosortByRow requires extra temporary storage buffer that has to be
*  allocated by the user. Storage buffer size can be determined by
*  \ref hipsparseXcoosort_bufferSizeExt().
*
*  \note
*  \p P can be \p NULL if a sorted permutation vector is not required.
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*
*  @param[in]
*  handle          handle to the hipsparse library context queue.
*  @param[in]
*  m               number of rows of the sparse COO matrix.
*  @param[in]
*  n               number of columns of the sparse COO matrix.
*  @param[in]
*  nnz             number of non-zero entries of the sparse COO matrix.
*  @param[inout]
*  cooRows         array of \p nnz elements containing the row indices of the sparse
*                  COO matrix.
*  @param[inout]
*  cooCols         array of \p nnz elements containing the column indices of the sparse
*                  COO matrix.
*  @param[inout]
*  P               array of \p nnz integers containing the unsorted map indices, can be
*                  \p NULL.
*  @param[in]
*  pBuffer         temporary storage buffer allocated by the user, size is returned by
*                  \ref hipsparseXcoosort_bufferSizeExt().
*
*  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p m, \p n, \p nnz, \p cooRows, 
*              \p cooCols or \p pBuffer pointer is invalid.
*  \retval     HIPSPARSE_STATUS_INTERNAL_ERROR an internal error occurred.
*
*  \par Example
*  \code{.c}
*    // hipSPARSE handle
*    hipsparseHandle_t handle;
*    hipsparseCreate(&handle);
*
*    // Sparse matrix in COO format (with unsorted row indices)
*    //     1 2 0 3 0
*    // A = 0 4 5 0 0
*    //     6 0 0 7 8
*    int hcooRowInd[8] = {0, 2, 0, 1, 1, 0, 2, 2};
*    int hcooColInd[8] = {0, 0, 1, 1, 2, 3, 3, 4};
*    float hcooVal[8]   = {1.0f, 6.0f, 2.0f, 4.0f, 5.0f, 3.0f, 7.0f, 8.0f}; 
*
*    int m         = 3;
*    int n         = 5;
*    int nnz       = 8;
*    hipsparseIndexBase_t base = HIPSPARSE_INDEX_BASE_ZERO;
*
*    int* dcooRowInd = nullptr;
*    int* dcooColInd = nullptr;
*    float* dcooVal = nullptr;
*    hipMalloc((void**)&dcooRowInd, sizeof(int) * nnz);
*    hipMalloc((void**)&dcooColInd, sizeof(int) * nnz);
*    hipMalloc((void**)&dcooVal, sizeof(float) * nnz);
*
*    hipMemcpy(dcooRowInd, hcooRowInd, sizeof(int) * nnz, hipMemcpyHostToDevice);
*    hipMemcpy(dcooColInd, hcooColInd, sizeof(int) * nnz, hipMemcpyHostToDevice);
*    hipMemcpy(dcooVal, hcooVal, sizeof(float) * nnz, hipMemcpyHostToDevice);
*
*    size_t bufferSize;
*    hipsparseXcoosort_bufferSizeExt(handle, m, n, nnz, dcooRowInd, dcooColInd, &bufferSize);
*
*    void* dbuffer = nullptr;
*    hipMalloc((void**)&dbuffer, bufferSize);
*
*    int* dperm = nullptr;
*    hipMalloc((void**)&dperm, sizeof(int) * nnz);
*    hipsparseCreateIdentityPermutation(handle, nnz, dperm);
*
*    hipsparseXcoosortByRow(handle, m, n, nnz, dcooRowInd, dcooColInd, dperm, dbuffer);
*
*    float* dcooValSorted = nullptr;
*    hipMalloc((void**)&dcooValSorted, sizeof(float) * nnz);
*    hipsparseSgthr(handle, nnz, dcooVal, dcooValSorted, dperm, base);
*
*    hipFree(dcooRowInd);
*    hipFree(dcooColInd);
*    hipFree(dcooVal);
*    hipFree(dcooValSorted);
*    hipFree(dperm);
*   
*    hipFree(dbuffer);
*
*    hipsparseDestroy(handle);
*  \endcode
*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseXcoosortByRow(hipsparseHandle_t handle,
                                         int               m,
                                         int               n,
                                         int               nnz,
                                         int*              cooRows,
                                         int*              cooCols,
                                         int*              P,
                                         void*             pBuffer);

/*! \ingroup conv_module
*  \brief Sort a sparse COO matrix by column
*
*  \details
*  \p hipsparseXcoosortByColumn sorts a matrix in COO format by column. The sorted
*  permutation vector \p P can be used to obtain sorted \p cooVal array. In this
*  case, \p P must be initialized as the identity permutation, see
*  \ref hipsparseCreateIdentityPermutation(). To apply the permutation vector to the COO 
*  values, see hipsparse \ref hipsparseSgthr "hipsparseXgthr()".
*
*  \p hipsparseXcoosortByColumn requires extra temporary storage buffer that has to be
*  allocated by the user. Storage buffer size can be determined by
*  \ref hipsparseXcoosort_bufferSizeExt().
*
*  \note
*  \p P can be \p NULL if a sorted permutation vector is not required.
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*
*  @param[in]
*  handle          handle to the hipsparse library context queue.
*  @param[in]
*  m               number of rows of the sparse COO matrix.
*  @param[in]
*  n               number of columns of the sparse COO matrix.
*  @param[in]
*  nnz             number of non-zero entries of the sparse COO matrix.
*  @param[inout]
*  cooRows         array of \p nnz elements containing the row indices of the sparse
*                  COO matrix.
*  @param[inout]
*  cooCols         array of \p nnz elements containing the column indices of the sparse
*                  COO matrix.
*  @param[inout]
*  P               array of \p nnz integers containing the unsorted map indices, can be
*                  \p NULL.
*  @param[in]
*  pBuffer         temporary storage buffer allocated by the user, size is returned by
*                  \ref hipsparseXcoosort_bufferSizeExt().
*
*  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p m, \p n, \p nnz, \p cooRows, 
*              \p cooCols or \p pBuffer pointer is invalid.
*  \retval     HIPSPARSE_STATUS_INTERNAL_ERROR an internal error occurred.
*
*  \par Example
*  \code{.c}
*    // hipSPARSE handle
*    hipsparseHandle_t handle;
*    hipsparseCreate(&handle);
*
*    // Sparse matrix in COO format (with unsorted column indices)
*    //     1 2 0 3 0
*    // A = 0 4 5 0 0
*    //     6 0 0 7 8
*    int hcooRowInd[8] = {0, 0, 0, 1, 1, 2, 2, 2};
*    int hcooColInd[8] = {0, 1, 3, 1, 2, 0, 3, 4};
*    float hcooVal[8]   = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f}; 
*
*    int m         = 3;
*    int n         = 5;
*    int nnz       = 8;
*    hipsparseIndexBase_t base = HIPSPARSE_INDEX_BASE_ZERO;
*
*    int* dcooRowInd = nullptr;
*    int* dcooColInd = nullptr;
*    float* dcooVal = nullptr;
*    hipMalloc((void**)&dcooRowInd, sizeof(int) * nnz);
*    hipMalloc((void**)&dcooColInd, sizeof(int) * nnz);
*    hipMalloc((void**)&dcooVal, sizeof(float) * nnz);
*
*    hipMemcpy(dcooRowInd, hcooRowInd, sizeof(int) * nnz, hipMemcpyHostToDevice);
*    hipMemcpy(dcooColInd, hcooColInd, sizeof(int) * nnz, hipMemcpyHostToDevice);
*    hipMemcpy(dcooVal, hcooVal, sizeof(float) * nnz, hipMemcpyHostToDevice);
*
*    size_t bufferSize;
*    hipsparseXcoosort_bufferSizeExt(handle, m, n, nnz, dcooRowInd, dcooColInd, &bufferSize);
*
*    void* dbuffer = nullptr;
*    hipMalloc((void**)&dbuffer, bufferSize);
*
*    int* dperm = nullptr;
*    hipMalloc((void**)&dperm, sizeof(int) * nnz);
*    hipsparseCreateIdentityPermutation(handle, nnz, dperm);
*
*    hipsparseXcoosortByColumn(handle, m, n, nnz, dcooRowInd, dcooColInd, dperm, dbuffer);
*
*    float* dcooValSorted = nullptr;
*    hipMalloc((void**)&dcooValSorted, sizeof(float) * nnz);
*    hipsparseSgthr(handle, nnz, dcooVal, dcooValSorted, dperm, base);
*
*    hipFree(dcooRowInd);
*    hipFree(dcooColInd);
*    hipFree(dcooVal);
*    hipFree(dcooValSorted);
*    hipFree(dperm);
*   
*    hipFree(dbuffer);
*
*    hipsparseDestroy(handle);
*  \endcode
*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseXcoosortByColumn(hipsparseHandle_t handle,
                                            int               m,
                                            int               n,
                                            int               nnz,
                                            int*              cooRows,
                                            int*              cooCols,
                                            int*              P,
                                            void*             pBuffer);

/*! \ingroup conv_module
*  \brief
*  This function computes the the size of the user allocated temporary storage buffer used when converting a sparse
*  GEBSR matrix to another sparse GEBSR matrix.
*
*  \details
*  \p hipsparseXgebsr2gebsr_bufferSize returns the size of the temporary storage buffer that is required by 
*  \ref hipsparseXgebsr2gebsrNnz() and \ref hipsparseSgebsr2gebsr "hipsparseXgebsr2gebsr()". The temporary storage 
*  buffer must be allocated by the user.
*
*  @param[in]
*  handle             handle to the hipsparse library context queue.
*  @param[in]
*  dirA               the storage format of the blocks, \ref HIPSPARSE_DIRECTION_ROW or \ref HIPSPARSE_DIRECTION_COLUMN
*  @param[in]
*  mb                 number of block rows of the general BSR sparse matrix \f$A\f$.
*  @param[in]
*  nb                 number of block columns of the general BSR sparse matrix \f$A\f$.
*  @param[in]
*  nnzb               number of blocks in the general BSR sparse matrix \f$A\f$.
*  @param[in]
*  descrA             the descriptor of the general BSR sparse matrix \f$A\f$, the supported matrix type is 
*                     \ref HIPSPARSE_MATRIX_TYPE_GENERAL and also any valid value of the \ref hipsparseIndexBase_t.
*  @param[in]
*  bsrValA            array of \p nnzb*rowBlockDimA*colBlockDimA containing the values of the sparse general BSR matrix \f$A\f$.
*  @param[in]
*  bsrRowPtrA         array of \p mb+1 elements that point to the start of every block row of the
*                     sparse general BSR matrix \f$A\f$.
*  @param[in]
*  bsrColIndA         array of \p nnzb elements containing the block column indices of the sparse general BSR matrix \f$A\f$.
*  @param[in]
*  rowBlockDimA       row size of the blocks in the sparse general BSR matrix \f$A\f$.
*  @param[in]
*  colBlockDimA       column size of the blocks in the sparse general BSR matrix \f$A\f$.
*  @param[in]
*  rowBlockDimC       row size of the blocks in the sparse general BSR matrix \f$C\f$.
*  @param[in]
*  colBlockDimC       column size of the blocks in the sparse general BSR matrix \f$C\f$.
*  @param[out]
*  pBufferSizeInBytes number of bytes of the temporary storage buffer required by hipsparseXgebsr2gebsrNnz(),
*                     hipsparseSgebsr2gebsr(), hipsparseDgebsr2gebsr(), hipsparseCgebsr2gebsr(), and 
*                     hipsparseZgebsr2gebsr().
*
*  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p mb, \p nb, \p nnzb, \p rowBlockDimA, \p colBlockDimA, 
*              \p rowBlockDimC, \p colBlockDimC, \p bsrRowPtrA, \p bsrColIndA, \p descrA or \p pBufferSizeInBytes pointer 
*              is invalid.
*/
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSgebsr2gebsr_bufferSize(hipsparseHandle_t         handle,
                                                   hipsparseDirection_t      dirA,
                                                   int                       mb,
                                                   int                       nb,
                                                   int                       nnzb,
                                                   const hipsparseMatDescr_t descrA,
                                                   const float*              bsrValA,
                                                   const int*                bsrRowPtrA,
                                                   const int*                bsrColIndA,
                                                   int                       rowBlockDimA,
                                                   int                       colBlockDimA,
                                                   int                       rowBlockDimC,
                                                   int                       colBlockDimC,
                                                   int*                      pBufferSizeInBytes);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDgebsr2gebsr_bufferSize(hipsparseHandle_t         handle,
                                                   hipsparseDirection_t      dirA,
                                                   int                       mb,
                                                   int                       nb,
                                                   int                       nnzb,
                                                   const hipsparseMatDescr_t descrA,
                                                   const double*             bsrValA,
                                                   const int*                bsrRowPtrA,
                                                   const int*                bsrColIndA,
                                                   int                       rowBlockDimA,
                                                   int                       colBlockDimA,
                                                   int                       rowBlockDimC,
                                                   int                       colBlockDimC,
                                                   int*                      pBufferSizeInBytes);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCgebsr2gebsr_bufferSize(hipsparseHandle_t         handle,
                                                   hipsparseDirection_t      dirA,
                                                   int                       mb,
                                                   int                       nb,
                                                   int                       nnzb,
                                                   const hipsparseMatDescr_t descrA,
                                                   const hipComplex*         bsrValA,
                                                   const int*                bsrRowPtrA,
                                                   const int*                bsrColIndA,
                                                   int                       rowBlockDimA,
                                                   int                       colBlockDimA,
                                                   int                       rowBlockDimC,
                                                   int                       colBlockDimC,
                                                   int*                      pBufferSizeInBytes);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZgebsr2gebsr_bufferSize(hipsparseHandle_t         handle,
                                                   hipsparseDirection_t      dirA,
                                                   int                       mb,
                                                   int                       nb,
                                                   int                       nnzb,
                                                   const hipsparseMatDescr_t descrA,
                                                   const hipDoubleComplex*   bsrValA,
                                                   const int*                bsrRowPtrA,
                                                   const int*                bsrColIndA,
                                                   int                       rowBlockDimA,
                                                   int                       colBlockDimA,
                                                   int                       rowBlockDimC,
                                                   int                       colBlockDimC,
                                                   int*                      pBufferSizeInBytes);
/**@}*/

/*! \ingroup conv_module
*  \brief This function is used when converting a GEBSR sparse matrix \f$A\f$ to another GEBSR sparse matrix \f$C\f$.
*  Specifically, this function determines the number of non-zero blocks that will exist in \f$C\f$ (stored using either a host
*  or device pointer), and computes the row pointer array for \f$C\f$.
*
*  \details
*  The routine does support asynchronous execution.
*
*  @param[in]
*  handle             handle to the hipsparse library context queue.
*  @param[in]
*  dirA               the storage format of the blocks, \ref HIPSPARSE_DIRECTION_ROW or \ref HIPSPARSE_DIRECTION_COLUMN
*  @param[in]
*  mb                 number of block rows of the general BSR sparse matrix \f$A\f$.
*  @param[in]
*  nb                 number of block columns of the general BSR sparse matrix \f$A\f$.
*  @param[in]
*  nnzb               number of blocks in the general BSR sparse matrix \f$A\f$.
*  @param[in]
*  descrA             the descriptor of the general BSR sparse matrix \f$A\f$, the supported matrix type is 
*                     \ref HIPSPARSE_MATRIX_TYPE_GENERAL and also any valid value of the \ref hipsparseIndexBase_t.
*  @param[in]
*  bsrRowPtrA         array of \p mb+1 elements that point to the start of every block row of the
*                     sparse general BSR matrix \f$A\f$.
*  @param[in]
*  bsrColIndA         array of \p nnzb elements containing the block column indices of the sparse general BSR matrix \p A.
*  @param[in]
*  rowBlockDimA       row size of the blocks in the sparse general BSR matrix \f$A\f$.
*  @param[in]
*  colBlockDimA       column size of the blocks in the sparse general BSR matrix \f$A\f$.
*  @param[in]
*  descrC             the descriptor of the general BSR sparse matrix \f$C\f$, the supported matrix type is 
*                     \ref HIPSPARSE_MATRIX_TYPE_GENERAL and also any valid value of the \ref hipsparseIndexBase_t.
*  @param[in]
*  bsrRowPtrC         array of \p mbC+1 elements that point to the start of every block row of the
*                     sparse general BSR matrix \f$C\f$ where \p mbC = ( \p m+rowBlockDimC-1 ) / \p rowBlockDimC.
*  @param[in]
*  rowBlockDimC       row size of the blocks in the sparse general BSR matrix \f$C\f$.
*  @param[in]
*  colBlockDimC       column size of the blocks in the sparse general BSR matrix \f$C\f$.
*  @param[out]
*  nnzTotalDevHostPtr total number of nonzero blocks in general BSR sparse matrix \f$C\f$ stored using device or host memory.
*  @param[out]
*  buffer             buffer allocated by the user whose size is determined by calling \ref hipsparseSgebsr2gebsr_bufferSize 
*                     "hipsparseXgebsr2gebsr_bufferSize()".
*
*  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p mb, \p nb, \p nnzb, \p rowBlockDimA, \p colBlockDimA, \p rowBlockDimC,
*              \p colBlockDimC, \p bsrRowPtrA, \p bsrColIndA, \p bsrRowPtrC, \p descrA, \p descrC, \p buffer pointer is invalid.
*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseXgebsr2gebsrNnz(hipsparseHandle_t         handle,
                                           hipsparseDirection_t      dirA,
                                           int                       mb,
                                           int                       nb,
                                           int                       nnzb,
                                           const hipsparseMatDescr_t descrA,
                                           const int*                bsrRowPtrA,
                                           const int*                bsrColIndA,
                                           int                       rowBlockDimA,
                                           int                       colBlockDimA,
                                           const hipsparseMatDescr_t descrC,
                                           int*                      bsrRowPtrC,
                                           int                       rowBlockDimC,
                                           int                       colBlockDimC,
                                           int*                      nnzTotalDevHostPtr,
                                           void*                     buffer);

/*! \ingroup conv_module
*  \brief
*  This function converts the GEBSR sparse matrix \f$A\f$ to another GEBSR sparse matrix \f$C\f$.
*
*  \details
*  The conversion uses three steps. First, the user calls \ref hipsparseSgebsr2gebsr_bufferSize 
*  "hipsparseXgebsr2gebsr_bufferSize()" to determine the size of the required temporary storage buffer. 
*  The user then allocates this buffer. Secondly, the user then allocates \p mbC+1 integers for the row 
*  pointer array for \f$C\f$ where:
*  \f[
*    \begin{align}
*    \text{mbC} &= \text{(m - 1) / rowBlockDimC + 1} \\
*    \text{nbC} &= \text{(n - 1) / colBlockDimC + 1}
*    \end{align}
*  \f]
*  The user then calls hipsparseXgebsr2gebsrNnz() to fill in the row pointer array for \f$C\f$ ( \p bsrRowPtrC ) and 
*  determine the number of non-zero blocks that will exist in \f$C\f$. Finally, the user allocates space for the column 
*  indices array of \f$C\f$ to have \p nnzbC elements and space for the values array of \f$C\f$ to have 
*  \p nnzbC*rowBlockDimC*colBlockDimC and then calls \p hipsparseXgebsr2gebsr to complete the conversion.
*
*  It may be the case that \p rowBlockDimC does not divide evenly into \p m and/or \p colBlockDim does not divide evenly 
*  into \p n. In these cases, the GEBSR matrix is expanded in size in order to fit full GEBSR blocks. For example, if 
*  the original GEBSR matrix A (using \p rowBlockDimA=2, \p colBlockDimA=3) looks like: 
*
*  \f[
*   \left[ 
*    \begin{array}{c | c} 
*      \begin{array}{c c c} 
*       1 & 0 & 0 \\ 
*       3 & 4 & 0
*      \end{array} & 
*      \begin{array}{c c c} 
*       2 & 0 & 0 \\ 
*       4 & 5 & 6
*      \end{array} \\ 
*    \hline 
*      \begin{array}{c c c} 
*       1 & 2 & 3 \\ 
*       1 & 2 & 0
*      \end{array} & 
*      \begin{array}{c c c} 
*       4 & 0 & 0 \\ 
*       3 & 0 & 1
*      \end{array} \\ 
*   \end{array} 
*  \right] 
*  \f]
*
*  then if we specify \p rowBlockDimC=3 and \p colBlockDimC=2, our output GEBSR matrix C would be:
*
*  \f[
*   \left[ 
*    \begin{array}{c | c | c} 
*      \begin{array}{c c} 
*       1 & 0 \\ 
*       3 & 4 \\
*       1 & 2
*      \end{array} & 
*      \begin{array}{c c} 
*       0 & 2 \\ 
*       0 & 4 \\
*       3 & 4
*      \end{array} & 
*      \begin{array}{c c} 
*       0 & 0 \\ 
*       5 & 6 \\
*       0 & 0
*      \end{array} \\
*    \hline 
*      \begin{array}{c c} 
*       1 & 2 \\ 
*       0 & 0 \\
*       0 & 0
*      \end{array} & 
*      \begin{array}{c c} 
*       0 & 3 \\ 
*       0 & 0 \\
*       0 & 0
*      \end{array} & 
*      \begin{array}{c c} 
*       0 & 1 \\ 
*       0 & 0 \\
*       0 & 0
*      \end{array} \\
*   \end{array} 
*  \right] 
*  \f]
*
*  @param[in]
*  handle        handle to the hipsparse library context queue.
*  @param[in]
*  dirA          the storage format of the blocks, \ref HIPSPARSE_DIRECTION_ROW or \ref HIPSPARSE_DIRECTION_COLUMN
*  @param[in]
*  mb            number of block rows of the general BSR sparse matrix \f$A\f$.
*  @param[in]
*  nb            number of block columns of the general BSR sparse matrix \f$A\f$.
*  @param[in]
*  nnzb          number of blocks in the general BSR sparse matrix \f$A\f$.
*  @param[in]
*  descrA        the descriptor of the general BSR sparse matrix \f$A\f$, the supported matrix type is 
*                \ref HIPSPARSE_MATRIX_TYPE_GENERAL and also any valid value of the \ref hipsparseIndexBase_t.
*  @param[in]
*  bsrValA       array of \p nnzb*rowBlockDimA*colBlockDimA containing the values of the sparse general BSR matrix \f$A\f$.
*  @param[in]
*  bsrRowPtrA    array of \p mb+1 elements that point to the start of every block row of the
*                sparse general BSR matrix \f$A\f$.
*  @param[in]
*  bsrColIndA    array of \p nnzb elements containing the block column indices of the sparse general BSR matrix \f$A\f$.
*  @param[in]
*  rowBlockDimA  row size of the blocks in the sparse general BSR matrix \f$A\f$.
*  @param[in]
*  colBlockDimA  column size of the blocks in the sparse general BSR matrix \f$A\f$.
*  @param[in]
*  descrC        the descriptor of the general BSR sparse matrix \f$C\f$, the supported matrix type is 
*                \ref HIPSPARSE_MATRIX_TYPE_GENERAL and also any valid value of the \ref hipsparseIndexBase_t.
*  @param[in]
*  bsrValC       array of \p nnzbC*rowBlockDimC*colBlockDimC containing the values of the sparse general BSR matrix \f$C\f$.
*  @param[in]
*  bsrRowPtrC    array of \p mbC+1 elements that point to the start of every block row of the
*                sparse general BSR matrix \f$C\f$.
*  @param[in]
*  bsrColIndC    array of \p nnzbC elements containing the block column indices of the sparse general BSR matrix \f$C\f$.
*  @param[in]
*  rowBlockDimC  row size of the blocks in the sparse general BSR matrix \f$C\f$.
*  @param[in]
*  colBlockDimC  column size of the blocks in the sparse general BSR matrix \f$C\f$.
*  @param[out]
*  buffer        buffer allocated by the user whose size is determined by calling \ref hipsparseSgebsr2gebsr_bufferSize 
*                "hipsparseXgebsr2gebsr_bufferSize()".
*
*  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p mb, \p nb, \p nnzb, \p rowBlockDimA, \p colBlockDimA,
*              \p rowBlockDimC, \p colBlockDimC, \p bsrRowPtrA, \p bsrColIndA, \p bsrValA, \p bsrRowPtrC, \p bsrColIndC,
*              \p bsrValC, \p descrA, \p descrC or \p buffer pointer is invalid.
*
*  \par Example
*  \code{.c}
*    // hipSPARSE handle
*    hipsparseHandle_t handle;
*    hipsparseCreate(&handle);
*
*    hipsparseMatDescr_t descrA;
*    hipsparseCreateMatDescr(&descrA);
*
*    hipsparseMatDescr_t descrC;
*    hipsparseCreateMatDescr(&descrC);
*
*    // Sparse matrix in BSR format
*    //     1 2 | 0 3 | 0 0 
*    //     0 4 | 5 0 | 0 1
*    // A = 6 0 | 0 7 | 8 0
*    //     --------------- 
*    //     0 0 | 3 0 | 2 2
*    //     1 0 | 0 0 | 4 3 
*    //     7 2 | 0 0 | 1 4
*    int hbsrRowPtrA[3] = {0, 3, 6};
*    int hbsrColIndA[6] = {0, 1, 2, 0, 1, 2};
*    float hbsrValA[36]  = {1.0f, 2.0f, 0.0f, 4.0f, 6.0f, 0.0f, 
*                           0.0f, 3.0f, 5.0f, 0.0f, 0.0f, 7.0f, 
*                           0.0f, 0.0f, 0.0f, 1.0f, 8.0f, 0.0f, 
*                           0.0f, 0.0f, 1.0f, 0.0f, 7.0f, 2.0f,
*                           3.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
*                           2.0f, 2.0f, 4.0f, 3.0f, 1.0f, 4.0f}; 
*
*    int m           = 6;
*    int n           = 6;
*    int rowBlockDimA = 3;
*    int colBlockDimA = 2;
*    int rowBlockDimC = 2;
*    int colBlockDimC = 2;
*    hipsparseDirection_t dirA = HIPSPARSE_DIRECTION_ROW;
*
*    int mbA   = (m + rowBlockDimA - 1) / rowBlockDimA;
*    int nbA   = (n + colBlockDimA - 1) / colBlockDimA;
*    int nnzbA = 6;
*
*    int mbC   = (m + rowBlockDimC - 1) / rowBlockDimC;
*    int nbC   = (n + colBlockDimC - 1) / colBlockDimC;
*
*    int* dbsrRowPtrA = nullptr;
*    int* dbsrColIndA = nullptr;
*    float* dbsrValA = nullptr;
*    hipMalloc((void**)&dbsrRowPtrA, sizeof(int) * (mbA + 1));
*    hipMalloc((void**)&dbsrColIndA, sizeof(int) * nnzbA);
*    hipMalloc((void**)&dbsrValA, sizeof(float) * rowBlockDimA * colBlockDimA * nnzbA);
*
*    hipMemcpy(dbsrRowPtrA, hbsrRowPtrA, sizeof(int) * (mbA + 1), hipMemcpyHostToDevice);
*    hipMemcpy(dbsrColIndA, hbsrColIndA, sizeof(int) * nnzbA, hipMemcpyHostToDevice);
*    hipMemcpy(dbsrValA, hbsrValA, sizeof(float) * rowBlockDimA * colBlockDimA * nnzbA, hipMemcpyHostToDevice);
*
*    int* dbsrRowPtrC = nullptr;
*    hipMalloc((void**)&dbsrRowPtrC, sizeof(int) * (mbC + 1));
*
*    size_t bufferSize;
*    hipsparseSgebsr2gebsr_bufferSize(handle, 
*                                     dirA, 
*                                     mbA, 
*                                     nbA, 
*                                     nnzbA, 
*                                     descrA, 
*                                     dbsrValA, 
*                                     dbsrRowPtrA, 
*                                     dbsrColIndA, 
*                                     rowBlockDimA, 
*                                     colBlockDimA, 
*                                     rowBlockDimC, 
*                                     colBlockDimC, 
*                                     &bufferSize);
*
*    void* dbuffer = nullptr;
*    hipMalloc((void**)&dbuffer, bufferSize);
*
*    int nnzbC;
*    hipsparseXgebsr2gebsrNnz(handle,
*                             dirA,
*                             mbA,
*                             nbA,
*                             nnzbA,
*                             descrA,
*                             dbsrRowPtrA,
*                             dbsrColIndA,
*                             rowBlockDimA,
*                             colBlockDimA,
*                             descrC,
*                             dbsrRowPtrC,
*                             rowBlockDimC,
*                             colBlockDimC,
*                             &nnzbC,
*                             dbuffer);
*
*    hipDeviceSynchronize();
*
*    int* dbsrColIndC = nullptr;
*    float* dbsrValC = nullptr;
*    hipMalloc((void**)&dbsrColIndC, sizeof(int) * nnzbC);
*    hipMalloc((void**)&dbsrValC, sizeof(float) * rowBlockDimC * colBlockDimC * nnzbC);
*
*    hipsparseSgebsr2gebsr(handle, 
*                          dirA, 
*                          mbA, 
*                          nbA, 
*                          nnzbA, 
*                          descrA, 
*                          dbsrValA, 
*                          dbsrRowPtrA, 
*                          dbsrColIndA, 
*                          rowBlockDimA, 
*                          colBlockDimA, 
*                          descrC, 
*                          dbsrValC, 
*                          dbsrRowPtrC, 
*                          dbsrColIndC, 
*                          rowBlockDimC, 
*                          colBlockDimC, 
*                          dbuffer);
*
*    hipFree(dbsrRowPtrA);
*    hipFree(dbsrColIndA);
*    hipFree(dbsrValA);
*    
*    hipFree(dbsrRowPtrC);
*    hipFree(dbsrColIndC);
*    hipFree(dbsrValC);
*
*    hipFree(dbuffer);
*
*    hipsparseDestroyMatDescr(descrA);
*    hipsparseDestroyMatDescr(descrC);
*    hipsparseDestroy(handle);
*  \endcode
*/
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSgebsr2gebsr(hipsparseHandle_t         handle,
                                        hipsparseDirection_t      dirA,
                                        int                       mb,
                                        int                       nb,
                                        int                       nnzb,
                                        const hipsparseMatDescr_t descrA,
                                        const float*              bsrValA,
                                        const int*                bsrRowPtrA,
                                        const int*                bsrColIndA,
                                        int                       rowBlockDimA,
                                        int                       colBlockDimA,
                                        const hipsparseMatDescr_t descrC,
                                        float*                    bsrValC,
                                        int*                      bsrRowPtrC,
                                        int*                      bsrColIndC,
                                        int                       rowBlockDimC,
                                        int                       colBlockDimC,
                                        void*                     buffer);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDgebsr2gebsr(hipsparseHandle_t         handle,
                                        hipsparseDirection_t      dirA,
                                        int                       mb,
                                        int                       nb,
                                        int                       nnzb,
                                        const hipsparseMatDescr_t descrA,
                                        const double*             bsrValA,
                                        const int*                bsrRowPtrA,
                                        const int*                bsrColIndA,
                                        int                       rowBlockDimA,
                                        int                       colBlockDimA,
                                        const hipsparseMatDescr_t descrC,
                                        double*                   bsrValC,
                                        int*                      bsrRowPtrC,
                                        int*                      bsrColIndC,
                                        int                       rowBlockDimC,
                                        int                       colBlockDimC,
                                        void*                     buffer);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCgebsr2gebsr(hipsparseHandle_t         handle,
                                        hipsparseDirection_t      dirA,
                                        int                       mb,
                                        int                       nb,
                                        int                       nnzb,
                                        const hipsparseMatDescr_t descrA,
                                        const hipComplex*         bsrValA,
                                        const int*                bsrRowPtrA,
                                        const int*                bsrColIndA,
                                        int                       rowBlockDimA,
                                        int                       colBlockDimA,
                                        const hipsparseMatDescr_t descrC,
                                        hipComplex*               bsrValC,
                                        int*                      bsrRowPtrC,
                                        int*                      bsrColIndC,
                                        int                       rowBlockDimC,
                                        int                       colBlockDimC,
                                        void*                     buffer);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZgebsr2gebsr(hipsparseHandle_t         handle,
                                        hipsparseDirection_t      dirA,
                                        int                       mb,
                                        int                       nb,
                                        int                       nnzb,
                                        const hipsparseMatDescr_t descrA,
                                        const hipDoubleComplex*   bsrValA,
                                        const int*                bsrRowPtrA,
                                        const int*                bsrColIndA,
                                        int                       rowBlockDimA,
                                        int                       colBlockDimA,
                                        const hipsparseMatDescr_t descrC,
                                        hipDoubleComplex*         bsrValC,
                                        int*                      bsrRowPtrC,
                                        int*                      bsrColIndC,
                                        int                       rowBlockDimC,
                                        int                       colBlockDimC,
                                        void*                     buffer);
/**@}*/

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
/*! \ingroup conv_module
*  \brief
*  This function calculates the amount of temporary storage in bytes required for
*  hipsparseXcsru2csr() and hipsparseXcsr2csru().
*/
/**@{*/
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseScsru2csr_bufferSizeExt(hipsparseHandle_t handle,
                                                   int               m,
                                                   int               n,
                                                   int               nnz,
                                                   float*            csrVal,
                                                   const int*        csrRowPtr,
                                                   int*              csrColInd,
                                                   csru2csrInfo_t    info,
                                                   size_t*           pBufferSizeInBytes);

DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDcsru2csr_bufferSizeExt(hipsparseHandle_t handle,
                                                   int               m,
                                                   int               n,
                                                   int               nnz,
                                                   double*           csrVal,
                                                   const int*        csrRowPtr,
                                                   int*              csrColInd,
                                                   csru2csrInfo_t    info,
                                                   size_t*           pBufferSizeInBytes);

DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCcsru2csr_bufferSizeExt(hipsparseHandle_t handle,
                                                   int               m,
                                                   int               n,
                                                   int               nnz,
                                                   hipComplex*       csrVal,
                                                   const int*        csrRowPtr,
                                                   int*              csrColInd,
                                                   csru2csrInfo_t    info,
                                                   size_t*           pBufferSizeInBytes);

DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZcsru2csr_bufferSizeExt(hipsparseHandle_t handle,
                                                   int               m,
                                                   int               n,
                                                   int               nnz,
                                                   hipDoubleComplex* csrVal,
                                                   const int*        csrRowPtr,
                                                   int*              csrColInd,
                                                   csru2csrInfo_t    info,
                                                   size_t*           pBufferSizeInBytes);
/**@}*/
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
/*! \ingroup conv_module
*  \brief
*  This function converts unsorted CSR format to sorted CSR format. The required
*  temporary storage has to be allocated by the user.
*/
/**@{*/
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseScsru2csr(hipsparseHandle_t         handle,
                                     int                       m,
                                     int                       n,
                                     int                       nnz,
                                     const hipsparseMatDescr_t descrA,
                                     float*                    csrVal,
                                     const int*                csrRowPtr,
                                     int*                      csrColInd,
                                     csru2csrInfo_t            info,
                                     void*                     pBuffer);

DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDcsru2csr(hipsparseHandle_t         handle,
                                     int                       m,
                                     int                       n,
                                     int                       nnz,
                                     const hipsparseMatDescr_t descrA,
                                     double*                   csrVal,
                                     const int*                csrRowPtr,
                                     int*                      csrColInd,
                                     csru2csrInfo_t            info,
                                     void*                     pBuffer);

DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCcsru2csr(hipsparseHandle_t         handle,
                                     int                       m,
                                     int                       n,
                                     int                       nnz,
                                     const hipsparseMatDescr_t descrA,
                                     hipComplex*               csrVal,
                                     const int*                csrRowPtr,
                                     int*                      csrColInd,
                                     csru2csrInfo_t            info,
                                     void*                     pBuffer);

DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZcsru2csr(hipsparseHandle_t         handle,
                                     int                       m,
                                     int                       n,
                                     int                       nnz,
                                     const hipsparseMatDescr_t descrA,
                                     hipDoubleComplex*         csrVal,
                                     const int*                csrRowPtr,
                                     int*                      csrColInd,
                                     csru2csrInfo_t            info,
                                     void*                     pBuffer);
/**@}*/
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
/*! \ingroup conv_module
*  \brief
*  This function converts sorted CSR format to unsorted CSR format. The required
*  temporary storage has to be allocated by the user.
*/
/**@{*/
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseScsr2csru(hipsparseHandle_t         handle,
                                     int                       m,
                                     int                       n,
                                     int                       nnz,
                                     const hipsparseMatDescr_t descrA,
                                     float*                    csrVal,
                                     const int*                csrRowPtr,
                                     int*                      csrColInd,
                                     csru2csrInfo_t            info,
                                     void*                     pBuffer);

DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDcsr2csru(hipsparseHandle_t         handle,
                                     int                       m,
                                     int                       n,
                                     int                       nnz,
                                     const hipsparseMatDescr_t descrA,
                                     double*                   csrVal,
                                     const int*                csrRowPtr,
                                     int*                      csrColInd,
                                     csru2csrInfo_t            info,
                                     void*                     pBuffer);

DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCcsr2csru(hipsparseHandle_t         handle,
                                     int                       m,
                                     int                       n,
                                     int                       nnz,
                                     const hipsparseMatDescr_t descrA,
                                     hipComplex*               csrVal,
                                     const int*                csrRowPtr,
                                     int*                      csrColInd,
                                     csru2csrInfo_t            info,
                                     void*                     pBuffer);

DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZcsr2csru(hipsparseHandle_t         handle,
                                     int                       m,
                                     int                       n,
                                     int                       nnz,
                                     const hipsparseMatDescr_t descrA,
                                     hipDoubleComplex*         csrVal,
                                     const int*                csrRowPtr,
                                     int*                      csrColInd,
                                     csru2csrInfo_t            info,
                                     void*                     pBuffer);
/**@}*/
#endif

/*
* ===========================================================================
*    reordering SPARSE
* ===========================================================================
*/

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
/*! \ingroup reordering_module
*  \brief Coloring of the adjacency graph of the matrix \f$A\f$ stored in the CSR format.
*
*  \details
*  \p hipsparseXcsrcolor performs the coloring of the undirected graph represented by the (symmetric) sparsity 
*  pattern of the matrix \f$A\f$ stored in CSR format. Graph coloring is a way of coloring the nodes of a graph 
*  such that no two adjacent nodes are of the same color. The \p fractionToColor is a parameter to only color
*  a given percentage of the graph nodes, the remaining uncolored nodes receive distinct new colors. The optional 
*  \p reordering array is a permutation array such that unknowns of the same color are grouped. The matrix \f$A\f$ 
*  must be stored as a general matrix with a symmetric sparsity pattern, and if the matrix \f$A\f$ is non-symmetric 
*  then the user is responsible to provide the symmetric part \f$\frac{A+A^T}{2}\f$.
*
*  @param[in]
*  handle          handle to the hipsparse library context queue.
*  @param[in]
*  m               number of rows of sparse matrix \f$A\f$.
*  @param[in]
*  nnz             number of non-zero entries of sparse matrix \f$A\f$.
*  @param[in]
*  descrA          sparse matrix descriptor.
*  @param[in]
*  csrValA         array of \p nnz elements of the sparse CSR matrix.
*  @param[in]
*  csrRowPtrA      array of \p m+1 elements that point to the start of every row of the
*                  sparse CSR matrix.
*  @param[in]
*  csrColIndA      array of \p nnz elements containing the column indices of the sparse
*                  CSR matrix.
*  @param[in]
*  fractionToColor fraction of nodes to be colored, which should be in the interval \f$[0.0,1.0]\f$, for example \f$0.8\f$ implies that 
*                  \f$80\f$ percent of nodes will be colored.
*  @param[out]
*  ncolors         resulting number of distinct colors.
*  @param[out]
*  coloring        resulting mapping of colors.
*  @param[out]
*  reordering      optional resulting reordering permutation if \p reordering is a non-null pointer.
*  @param[inout]
*  info            structure that holds the information collected during the coloring algorithm.
*
*  \retval HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p m, \p nnz, \p descrA, \p csrValA, \p csrRowPtrA, \p csrColIndA, 
*                                         \p fractionToColor, \p ncolors, \p coloring or \p info pointer is invalid.
*/
/**@{*/
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseScsrcolor(hipsparseHandle_t         handle,
                                     int                       m,
                                     int                       nnz,
                                     const hipsparseMatDescr_t descrA,
                                     const float*              csrValA,
                                     const int*                csrRowPtrA,
                                     const int*                csrColIndA,
                                     const float*              fractionToColor,
                                     int*                      ncolors,
                                     int*                      coloring,
                                     int*                      reordering,
                                     hipsparseColorInfo_t      info);

DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDcsrcolor(hipsparseHandle_t         handle,
                                     int                       m,
                                     int                       nnz,
                                     const hipsparseMatDescr_t descrA,
                                     const double*             csrValA,
                                     const int*                csrRowPtrA,
                                     const int*                csrColIndA,
                                     const double*             fractionToColor,
                                     int*                      ncolors,
                                     int*                      coloring,
                                     int*                      reordering,
                                     hipsparseColorInfo_t      info);

DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCcsrcolor(hipsparseHandle_t         handle,
                                     int                       m,
                                     int                       nnz,
                                     const hipsparseMatDescr_t descrA,
                                     const hipComplex*         csrValA,
                                     const int*                csrRowPtrA,
                                     const int*                csrColIndA,
                                     const float*              fractionToColor,
                                     int*                      ncolors,
                                     int*                      coloring,
                                     int*                      reordering,
                                     hipsparseColorInfo_t      info);

DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZcsrcolor(hipsparseHandle_t         handle,
                                     int                       m,
                                     int                       nnz,
                                     const hipsparseMatDescr_t descrA,
                                     const hipDoubleComplex*   csrValA,
                                     const int*                csrRowPtrA,
                                     const int*                csrColIndA,
                                     const double*             fractionToColor,
                                     int*                      ncolors,
                                     int*                      coloring,
                                     int*                      reordering,
                                     hipsparseColorInfo_t      info);
/**@}*/
#endif

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

/* Generic API functions */

/*! \ingroup generic_module
*  \brief Scale a sparse vector and add it to a scaled dense vector.
*
*  \details
*  \ref hipsparseAxpby multiplies the sparse vector \f$x\f$ with scalar \f$\alpha\f$ and
*  adds the result to the dense vector \f$y\f$ that is multiplied with scalar
*  \f$\beta\f$, such that
*
*  \f[
*      y := \alpha \cdot x + \beta \cdot y
*  \f]
*
*  \code{.c}
*      for(i = 0; i < size; ++i)
*      {
*          y[i] = beta * y[i]
*      }
*      for(i = 0; i < nnz; ++i)
*      {
*          y[xInd[i]] += alpha * xVal[i]
*      }
*  \endcode
*
*  @param[in]
*  handle      handle to the hipsparse library context queue.
*  @param[in]
*  alpha       scalar \f$\alpha\f$.
*  @param[in]
*  vecX        sparse matrix descriptor.
*  @param[in]
*  beta        scalar \f$\beta\f$.
*  @param[inout]
*  vecY        dense matrix descriptor.
*
*  \retval HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p alpha, \p vecX, \p beta or \p vecY pointer is
*          invalid.
*
*  \par Example
*  \code{.c}
*    // Number of non-zeros of the sparse vector
*    int nnz = 3;
*
*    // Size of sparse and dense vector
*    int size = 9;
*
*    // Sparse index vector
*    std::vector<int> hxInd = {0, 3, 5};
*
*    // Sparse value vector
*    std::vector<float> hxVal = {1.0f, 2.0f, 3.0f};
*
*    // Dense vector
*    std::vector<float> hy = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
*
*    // Scalar alpha
*    float alpha = 3.7f;
*
*    // Scalar beta
*    float beta = 1.2f;
*
*    // Offload data to device
*    int* dxInd;
*    float* dxVal;
*    float* dy;
*    hipMalloc((void**)&dxInd, sizeof(int) * nnz);
*    hipMalloc((void**)&dxVal, sizeof(float) * nnz);
*    hipMalloc((void**)&dy, sizeof(float) * size);
*
*    hipMemcpy(dxInd, hxInd.data(), sizeof(int) * nnz, hipMemcpyHostToDevice);
*    hipMemcpy(dxVal, hxVal.data(), sizeof(float) * nnz, hipMemcpyHostToDevice);
*    hipMemcpy(dy, hy.data(), sizeof(float) * size, hipMemcpyHostToDevice);
*
*    hipsparseHandle_t handle;
*    hipsparseCreate(&handle);
*
*    // Create sparse vector X
*    hipsparseSpVecDescr_t vecX;
*    hipsparseCreateSpVec(&vecX,
*                        size,
*                        nnz,
*                        dxInd,
*                        dxVal,
*                        HIPSPARSE_INDEX_32I,
*                        HIPSPARSE_INDEX_BASE_ZERO,
*                        HIP_R_32F);
*
*    // Create dense vector Y
*    hipsparseDnVecDescr_t vecY;
*    hipsparseCreateDnVec(&vecY, size, dy, HIP_R_32F);
*
*    // Call axpby to perform y = beta * y + alpha * x
*    hipsparseAxpby(handle, &alpha, vecX, &beta, vecY);
*
*    hipsparseDnVecGetValues(vecY, (void**)&dy);
*
*    // Copy result back to host
*    hipMemcpy(hy.data(), dy, sizeof(float) * size, hipMemcpyDeviceToHost);
*
*
*    // Clear hipSPARSE
*    hipsparseDestroySpVec(vecX);
*    hipsparseDestroyDnVec(vecY);
*    hipsparseDestroy(handle);
*
*    // Clear device memory
*    hipFree(dxInd);
*    hipFree(dxVal);
*    hipFree(dy);
*  \endcode
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseAxpby(hipsparseHandle_t          handle,
                                 const void*                alpha,
                                 hipsparseConstSpVecDescr_t vecX,
                                 const void*                beta,
                                 hipsparseDnVecDescr_t      vecY);
#elif(CUDART_VERSION >= 11000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseAxpby(hipsparseHandle_t     handle,
                                 const void*           alpha,
                                 hipsparseSpVecDescr_t vecX,
                                 const void*           beta,
                                 hipsparseDnVecDescr_t vecY);
#endif

/*! \ingroup generic_module
*  \brief Gather elements from a dense vector and store them into a sparse vector.
*
*  \details
*  \ref hipsparseGather gathers the elements from the dense vector \f$y\f$ and stores
*  them in the sparse vector \f$x\f$.
*
*  \code{.c}
*      for(i = 0; i < nnz; ++i)
*      {
*          xVal[i] = y[xInd[i]];
*      }
*  \endcode
*
*  @param[in]
*  handle       handle to the hipsparse library context queue.
*  @param[in]
*  vecY         dense vector descriptor \f$y\f$.
*  @param[out]
*  vecX         sparse vector descriptor \f$x\f$.
*
*  \retval      HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval      HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p vecX or \p vecY pointer is invalid.
*
*  \par Example
*  \code{.c}
*    // Number of non-zeros of the sparse vector
*    int nnz = 3;
*
*    // Size of sparse and dense vector
*    int size = 9;
*
*    // Sparse index vector
*    std::vector<int> hxInd = {0, 3, 5};
*
*    // Dense vector
*    std::vector<float> hy = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
*
*    // Offload data to device
*    int* dxInd;
*    float* dxVal;
*    float* dy;
*    hipMalloc((void**)&dxInd, sizeof(int) * nnz);
*    hipMalloc((void**)&dxVal, sizeof(float) * nnz);
*    hipMalloc((void**)&dy, sizeof(float) * size);
*
*    hipMemcpy(dxInd, hxInd.data(), sizeof(int) * nnz, hipMemcpyHostToDevice);
*    hipMemcpy(dy, hy.data(), sizeof(float) * size, hipMemcpyHostToDevice);
*
*    hipsparseHandle_t handle;
*    hipsparseCreate(&handle);
*
*    // Create sparse vector X
*    hipsparseSpVecDescr_t vecX;
*    hipsparseCreateSpVec(&vecX,
*                         size,
*                         nnz,
*                         dxInd,
*                         dxVal,
*                         HIPSPARSE_INDEX_32I,
*                         HIPSPARSE_INDEX_BASE_ZERO,
*                         HIP_R_32F);
*
*    // Create dense vector Y
*    hipsparseDnVecDescr_t vecY;
*    hipsparseCreateDnVec(&vecY, size, dy, HIP_R_32F);
*
*    // Perform gather
*    hipsparseGather(handle, vecY, vecX);
*
*    hipsparseSpVecGetValues(vecX, (void**)&dxVal);
*
*    // Copy result back to host
*    std::vector<float> hxVal(nnz, 0.0f);
*    hipMemcpy(hxVal.data(), dxVal, sizeof(float) * nnz, hipMemcpyDeviceToHost);
*
*    // Clear hipSPARSE
*    hipsparseDestroySpVec(vecX);
*    hipsparseDestroyDnVec(vecY);
*    hipsparseDestroy(handle);
*
*    // Clear device memory
*    hipFree(dxInd);
*    hipFree(dxVal);
*    hipFree(dy);
*  \endcode
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseGather(hipsparseHandle_t          handle,
                                  hipsparseConstDnVecDescr_t vecY,
                                  hipsparseSpVecDescr_t      vecX);
#elif(CUDART_VERSION >= 11000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseGather(hipsparseHandle_t     handle,
                                  hipsparseDnVecDescr_t vecY,
                                  hipsparseSpVecDescr_t vecX);
#endif

/*! \ingroup generic_module
*  \brief Scatter elements from a sparse vector into a dense vector.
*
*  \details
*  \ref hipsparseScatter scatters the elements from the sparse vector \f$x\f$ in the dense
*  vector \f$y\f$.
*
*  \code{.c}
*      for(i = 0; i < nnz; ++i)
*      {
*          y[xInd[i]] = xVal[i];
*      }
*  \endcode
*
*  @param[in]
*  handle       handle to the hipsparse library context queue.
*  @param[in]
*  vecX         sparse vector descriptor \f$x\f$.
*  @param[out]
*  vecY         dense vector descriptor \f$y\f$.
*
*  \retval      HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval      HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p vecX or \p vecY pointer is invalid.
*
*  \par Example
*  \code{.c}
*    // Number of non-zeros of the sparse vector
*    int nnz = 3;
*
*    // Size of sparse and dense vector
*    int size = 9;
*
*    // Sparse index vector
*    std::vector<int> hxInd = {0, 3, 5};
*
*    // Sparse value vector
*    std::vector<float> hxVal = {1.0f, 2.0f, 3.0f};
*
*    // Dense vector
*    std::vector<float> hy = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
*
*    // Offload data to device
*    int* dxInd;
*    float* dxVal;
*    float* dy;
*    hipMalloc((void**)&dxInd, sizeof(int) * nnz);
*    hipMalloc((void**)&dxVal, sizeof(float) * nnz);
*    hipMalloc((void**)&dy, sizeof(float) * size);
*
*    hipMemcpy(dxInd, hxInd.data(), sizeof(int) * nnz, hipMemcpyHostToDevice);
*    hipMemcpy(dxVal, hxVal.data(), sizeof(float) * nnz, hipMemcpyHostToDevice);
*    hipMemcpy(dy, hy.data(), sizeof(float) * size, hipMemcpyHostToDevice);
*
*    hipsparseHandle_t handle;
*    hipsparseCreate(&handle);
*
*    // Create sparse vector X
*    hipsparseSpVecDescr_t vecX;
*    hipsparseCreateSpVec(&vecX,
*                                size,
*                                nnz,
*                                dxInd,
*                                dxVal,
*                                HIPSPARSE_INDEX_32I,
*                                HIPSPARSE_INDEX_BASE_ZERO,
*                                HIP_R_32F);
*
*    // Create dense vector Y
*    hipsparseDnVecDescr_t vecY;
*    hipsparseCreateDnVec(&vecY, size, dy, HIP_R_32F);
*
*    // Perform scatter
*    hipsparseScatter(handle, vecX, vecY);
*
*    hipsparseDnVecGetValues(vecY, (void**)&dy);
*
*    // Copy result back to host
*    hipMemcpy(hy.data(), dy, sizeof(float) * size, hipMemcpyDeviceToHost);
*
*    // Clear hipSPARSE
*    hipsparseDestroySpVec(vecX);
*    hipsparseDestroyDnVec(vecY);
*    hipsparseDestroy(handle);
*
*    // Clear device memory
*    hipFree(dxInd);
*    hipFree(dxVal);
*    hipFree(dy);
*  \endcode
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseScatter(hipsparseHandle_t          handle,
                                   hipsparseConstSpVecDescr_t vecX,
                                   hipsparseDnVecDescr_t      vecY);
#elif(CUDART_VERSION >= 11000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseScatter(hipsparseHandle_t     handle,
                                   hipsparseSpVecDescr_t vecX,
                                   hipsparseDnVecDescr_t vecY);
#endif

/*! \ingroup generic_module
*  \brief Apply Givens rotation to a dense and a sparse vector.
*
*  \details
*  \ref hipsparseRot applies the Givens rotation matrix \f$G\f$ to the sparse vector
*  \f$x\f$ and the dense vector \f$y\f$, where
*  \f[
*    G = \begin{pmatrix} c & s \\ -s & c \end{pmatrix}
*  \f]
*
*  \code{.c}
*      for(i = 0; i < nnz; ++i)
*      {
*          x_tmp = xVal[i];
*          y_tmp = y[xInd[i]];
*
*          xVal[i]    = c * x_tmp + s * y_tmp;
*          y[xInd[i]] = c * y_tmp - s * x_tmp;
*      }
*  \endcode
*
*  @param[in]
*  handle      handle to the hipsparse library context queue.
*  @param[in]
*  c_coeff     pointer to the cosine element of \f$G\f$, can be on host or device.
*  @param[in]
*  s_coeff     pointer to the sine element of \f$G\f$, can be on host or device.
*  @param[inout]
*  vecX        sparse vector descriptor \f$x\f$.
*  @param[inout]
*  vecY        dense vector descriptor \f$y\f$.
*
*  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p c_coeff, \p s_coeff, \p vecX or \p vecY pointer is
*              invalid.
*
*  \par Example
*  \code{.c}
*    // Number of non-zeros of the sparse vector
*    int nnz = 3;
*
*    // Size of sparse and dense vector
*    int size = 9;
*
*    // Sparse index vector
*    std::vector<int> hxInd = {0, 3, 5};
*
*    // Sparse value vector
*    std::vector<float> hxVal = {1.0f, 2.0f, 3.0f};
*
*    // Dense vector
*    std::vector<float> hy = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
*
*    // Scalar c
*    float c = 3.7f;
*
*    // Scalar s
*    float s = 1.2f;
*
*    // Offload data to device
*    int* dxInd;
*    float* dxVal;
*    float* dy;
*    hipMalloc((void**)&dxInd, sizeof(int) * nnz);
*    hipMalloc((void**)&dxVal, sizeof(float) * nnz);
*    hipMalloc((void**)&dy, sizeof(float) * size);
*
*    hipMemcpy(dxInd, hxInd.data(), sizeof(int) * nnz, hipMemcpyHostToDevice);
*    hipMemcpy(dxVal, hxVal.data(), sizeof(float) * nnz, hipMemcpyHostToDevice);
*    hipMemcpy(dy, hy.data(), sizeof(float) * size, hipMemcpyHostToDevice);
*
*    hipsparseHandle_t handle;
*    hipsparseCreate(&handle);
*
*    // Create sparse vector X
*    hipsparseSpVecDescr_t vecX;
*    hipsparseCreateSpVec(&vecX,
*                                size,
*                                nnz,
*                                dxInd,
*                                dxVal,
*                                HIPSPARSE_INDEX_32I,
*                                HIPSPARSE_INDEX_BASE_ZERO,
*                                HIP_R_32F);
*
*    // Create dense vector Y
*    hipsparseDnVecDescr_t vecY;
*    hipsparseCreateDnVec(&vecY, size, dy, HIP_R_32F);
*
*    // Call rot
*    hipsparseRot(handle, (void*)&c, (void*)&s, vecX, vecY);
*
*    hipsparseSpVecGetValues(vecX, (void**)&dxVal);
*    hipsparseDnVecGetValues(vecY, (void**)&dy);
*
*    // Copy result back to host
*    hipMemcpy(hxVal.data(), dxVal, sizeof(float) * nnz, hipMemcpyDeviceToHost);
*    hipMemcpy(hy.data(), dy, sizeof(float) * size, hipMemcpyDeviceToHost);
*
*    // Clear hipSPARSE
*    hipsparseDestroySpVec(vecX);
*    hipsparseDestroyDnVec(vecY);
*    hipsparseDestroy(handle);
*
*    // Clear device memory
*    hipFree(dxInd);
*    hipFree(dxVal);
*    hipFree(dy);
*  \endcode
*/
#if(!defined(CUDART_VERSION) || (CUDART_VERSION >= 11000 && CUDART_VERSION < 13000))
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseRot(hipsparseHandle_t     handle,
                               const void*           c_coeff,
                               const void*           s_coeff,
                               hipsparseSpVecDescr_t vecX,
                               hipsparseDnVecDescr_t vecY);
#endif

/*! \ingroup generic_module
*  \brief Sparse matrix to dense matrix conversion
*
*  \details
*  \p hipsparseSparseToDense_bufferSize computes the required user allocated buffer size needed when converting 
*  a sparse matrix to a dense matrix.
*
*  @param[in]
*  handle              handle to the hipsparse library context queue.
*  @param[in]
*  matA                sparse matrix descriptor.
*  @param[in]
*  matB                dense matrix descriptor.
*  @param[in]
*  alg                 algorithm for the sparse to dense computation.
*  @param[out]
*  pBufferSizeInBytes  number of bytes of the temporary storage buffer.
*
*  \retval      HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval      HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p matA, \p matB, or \p pBufferSizeInBytes
*               pointer is invalid.
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSparseToDense_bufferSize(hipsparseHandle_t           handle,
                                                    hipsparseConstSpMatDescr_t  matA,
                                                    hipsparseDnMatDescr_t       matB,
                                                    hipsparseSparseToDenseAlg_t alg,
                                                    size_t*                     pBufferSizeInBytes);
#elif(CUDART_VERSION >= 11020)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSparseToDense_bufferSize(hipsparseHandle_t           handle,
                                                    hipsparseSpMatDescr_t       matA,
                                                    hipsparseDnMatDescr_t       matB,
                                                    hipsparseSparseToDenseAlg_t alg,
                                                    size_t*                     pBufferSizeInBytes);
#endif

/*! \ingroup generic_module
*  \brief Sparse matrix to dense matrix conversion
*
*  \details
*  \p hipsparseSparseToDense converts a sparse matrix to a dense matrix. This routine takes a user allocated buffer 
*  whose size must first be computed by calling \ref hipsparseSparseToDense_bufferSize
*
*  @param[in]
*  handle          handle to the hipsparse library context queue.
*  @param[in]
*  matA            sparse matrix descriptor.
*  @param[in]
*  matB            dense matrix descriptor.
*  @param[in]
*  alg             algorithm for the sparse to dense computation.
*  @param[in]
*  externalBuffer  temporary storage buffer allocated by the user.
*
*  \retval      HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval      HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p matA, \p matB, or \p externalBuffer
*               pointer is invalid.
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSparseToDense(hipsparseHandle_t           handle,
                                         hipsparseConstSpMatDescr_t  matA,
                                         hipsparseDnMatDescr_t       matB,
                                         hipsparseSparseToDenseAlg_t alg,
                                         void*                       externalBuffer);
#elif(CUDART_VERSION >= 11020)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSparseToDense(hipsparseHandle_t           handle,
                                         hipsparseSpMatDescr_t       matA,
                                         hipsparseDnMatDescr_t       matB,
                                         hipsparseSparseToDenseAlg_t alg,
                                         void*                       externalBuffer);
#endif

/*! \ingroup generic_module
*  \brief Dense matrix to sparse matrix conversion
*
*  \details
*  \p hipsparseDenseToSparse_bufferSize computes the required user allocated buffer size needed when converting 
*  a dense matrix to a sparse matrix.
*
*  @param[in]
*  handle              handle to the hipsparse library context queue.
*  @param[in]
*  matA                dense matrix descriptor.
*  @param[in]
*  matB                sparse matrix descriptor.
*  @param[in]
*  alg                 algorithm for the dense to sparse computation.
*  @param[out]
*  pBufferSizeInBytes  number of bytes of the temporary storage buffer. 
*
*  \retval      HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval      HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p matA, \p matB, or \p pBufferSizeInBytes
*               pointer is invalid.
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDenseToSparse_bufferSize(hipsparseHandle_t           handle,
                                                    hipsparseConstDnMatDescr_t  matA,
                                                    hipsparseSpMatDescr_t       matB,
                                                    hipsparseDenseToSparseAlg_t alg,
                                                    size_t*                     pBufferSizeInBytes);
#elif(CUDART_VERSION >= 11020)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDenseToSparse_bufferSize(hipsparseHandle_t           handle,
                                                    hipsparseDnMatDescr_t       matA,
                                                    hipsparseSpMatDescr_t       matB,
                                                    hipsparseDenseToSparseAlg_t alg,
                                                    size_t*                     pBufferSizeInBytes);
#endif

/*! \ingroup generic_module
*  \brief Dense matrix to sparse matrix conversion
*
*  \details
*  \p hipsparseDenseToSparse_analysis performs analysis that is later used in \ref hipsparseDenseToSparse_convert when 
*  converting a dense matrix to sparse matrix. This routine takes a user allocated buffer whose size must first be computed 
*  using \ref hipsparseDenseToSparse_bufferSize.
*
*  @param[in]
*  handle          handle to the hipsparse library context queue.
*  @param[in]
*  matA            dense matrix descriptor.
*  @param[in]
*  matB            sparse matrix descriptor.
*  @param[in]
*  alg             algorithm for the dense to sparse computation.
*  @param[out]
*  externalBuffer  temporary storage buffer allocated by the user. 
*
*  \retval      HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval      HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p matA, \p matB, or \p externalBuffer
*               pointer is invalid.
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDenseToSparse_analysis(hipsparseHandle_t           handle,
                                                  hipsparseConstDnMatDescr_t  matA,
                                                  hipsparseSpMatDescr_t       matB,
                                                  hipsparseDenseToSparseAlg_t alg,
                                                  void*                       externalBuffer);
#elif(CUDART_VERSION >= 11020)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDenseToSparse_analysis(hipsparseHandle_t           handle,
                                                  hipsparseDnMatDescr_t       matA,
                                                  hipsparseSpMatDescr_t       matB,
                                                  hipsparseDenseToSparseAlg_t alg,
                                                  void*                       externalBuffer);
#endif

/*! \ingroup generic_module
*  \brief Dense matrix to sparse matrix conversion
*
*  \details
*  \p hipsparseDenseToSparse_convert converts a dense matrix to a sparse matrix. This routine requires a user allocated buffer
*  whose size must be determined by first calling \ref hipsparseDenseToSparse_bufferSize.
*
*  @param[in]
*  handle          handle to the hipsparse library context queue.
*  @param[in]
*  matA            dense matrix descriptor.
*  @param[in]
*  matB            sparse matrix descriptor.
*  @param[in]
*  alg             algorithm for the dense to sparse computation.
*  @param[out]
*  externalBuffer  temporary storage buffer allocated by the user. 
*
*  \retval      HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval      HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p matA, \p matB, or \p externalBuffer
*               pointer is invalid.
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDenseToSparse_convert(hipsparseHandle_t           handle,
                                                 hipsparseConstDnMatDescr_t  matA,
                                                 hipsparseSpMatDescr_t       matB,
                                                 hipsparseDenseToSparseAlg_t alg,
                                                 void*                       externalBuffer);
#elif(CUDART_VERSION >= 11020)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDenseToSparse_convert(hipsparseHandle_t           handle,
                                                 hipsparseDnMatDescr_t       matA,
                                                 hipsparseSpMatDescr_t       matB,
                                                 hipsparseDenseToSparseAlg_t alg,
                                                 void*                       externalBuffer);
#endif

/*! \ingroup generic_module
*  \brief Compute the inner dot product of a sparse vector with a dense vector
*
*  \details
*  \p hipsparseSpVV_bufferSize computes the required user allocated buffer size needed when computing the 
*  inner dot product of a sparse vector with a dense vector
*
*  See \ref hipsparseSpVV for full code example.
*
*  @param[in]
*  handle              handle to the hipsparse library context queue.
*  @param[in]
*  opX                 sparse vector operation type.
*  @param[in]
*  vecX                sparse vector descriptor.
*  @param[in]
*  vecY                dense vector descriptor.
*  @param[out]
*  result              pointer to the result, can be host or device memory
*  @param[in]
*  computeType         floating point precision for the SpVV computation.
*  @param[out]
*  pBufferSizeInBytes  number of bytes of the temporary storage buffer. 
*
*  \retval      HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval      HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p vecX, \p vecY, \p result or \p pBufferSizeInBytes
*               pointer is invalid.
*  \retval      HIPSPARSE_STATUS_NOT_SUPPORTED \p computeType is currently not
*               supported.
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpVV_bufferSize(hipsparseHandle_t          handle,
                                           hipsparseOperation_t       opX,
                                           hipsparseConstSpVecDescr_t vecX,
                                           hipsparseConstDnVecDescr_t vecY,
                                           void*                      result,
                                           hipDataType                computeType,
                                           size_t*                    pBufferSizeInBytes);
#elif(CUDART_VERSION > 10010 || (CUDART_VERSION == 10010 && CUDART_10_1_UPDATE_VERSION == 1))
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpVV_bufferSize(hipsparseHandle_t     handle,
                                           hipsparseOperation_t  opX,
                                           hipsparseSpVecDescr_t vecX,
                                           hipsparseDnVecDescr_t vecY,
                                           void*                 result,
                                           hipDataType           computeType,
                                           size_t*               pBufferSizeInBytes);
#endif

/*! \ingroup generic_module
*  \brief Compute the inner dot product of a sparse vector with a dense vector
*
*  \details
*  \p hipsparseSpVV computes the inner dot product of a sparse vector with a dense vector. This routine takes a user 
*  allocated buffer whose size must first be computed by calling \ref hipsparseSpVV_bufferSize
*
*  @param[in]
*  handle          handle to the hipsparse library context queue.
*  @param[in]
*  opX             sparse vector operation type.
*  @param[in]
*  vecX            sparse vector descriptor.
*  @param[in]
*  vecY            dense vector descriptor.
*  @param[out]
*  result          pointer to the result, can be host or device memory
*  @param[in]
*  computeType     floating point precision for the SpVV computation.
*  @param[out]
*  externalBuffer  temporary storage buffer allocated by the user.
*
*  \retval      HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval      HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p vecX, \p vecY, \p result or \p externalBuffer
*               pointer is invalid.
*  \retval      HIPSPARSE_STATUS_NOT_SUPPORTED \p computeType is currently not
*               supported.
*
*  \par Example
*  \code{.c}
*    // Number of non-zeros of the sparse vector
*    int nnz = 3;
*
*    // Size of sparse and dense vector
*    int size = 9;
*
*    // Sparse index vector
*    std::vector<int> hxInd = {0, 3, 5};
*
*    // Sparse value vector
*    std::vector<float> hxVal = {1.0f, 2.0f, 3.0f};
*
*    // Dense vector
*    std::vector<float> hy = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
*
*    // Offload data to device
*    int* dxInd;
*    float* dxVal;
*    float* dy;
*    hipMalloc((void**)&dxInd, sizeof(int) * nnz);
*    hipMalloc((void**)&dxVal, sizeof(float) * nnz);
*    hipMalloc((void**)&dy, sizeof(float) * size);
*
*    hipMemcpy(dxInd, hxInd.data(), sizeof(int) * nnz, hipMemcpyHostToDevice);
*    hipMemcpy(dxVal, hxVal.data(), sizeof(float) * nnz, hipMemcpyHostToDevice);
*    hipMemcpy(dy, hy.data(), sizeof(float) * size, hipMemcpyHostToDevice);
*
*    hipsparseHandle_t handle;
*    hipsparseCreate(&handle);
*
*    // Create sparse vector X
*    hipsparseSpVecDescr_t vecX;
*    hipsparseCreateSpVec(&vecX,
*                        size,
*                        nnz,
*                        dxInd,
*                        dxVal,
*                        HIPSPARSE_INDEX_32I,
*                        HIPSPARSE_INDEX_BASE_ZERO,
*                        HIP_R_32F);
*
*    // Create dense vector Y
*    hipsparseDnVecDescr_t vecY;
*    hipsparseCreateDnVec(&vecY, size, dy, HIP_R_32F);
*
*    // Obtain buffer size
*    float hresult = 0.0f;
*    size_t buffer_size;
*    hipsparseSpVV_bufferSize(handle,
*                HIPSPARSE_OPERATION_NON_TRANSPOSE,
*                vecX,
*                vecY,
*                &hresult,
*                HIP_R_32F,
*                &buffer_size);
*
*    void* temp_buffer;
*    hipMalloc(&temp_buffer, buffer_size);
*
*    // SpVV
*    hipsparseSpVV(handle,
*                HIPSPARSE_OPERATION_NON_TRANSPOSE,
*                vecX,
*                vecY,
*                &hresult,
*                HIP_R_32F,
*                temp_buffer);
*
*    hipDeviceSynchronize();
*
*    std::cout << "hresult: " << hresult << std::endl;
*
*    // Clear hipSPARSE
*    hipsparseDestroySpVec(vecX);
*    hipsparseDestroyDnVec(vecY);
*    hipsparseDestroy(handle);
*
*    // Clear device memory
*    hipFree(dxInd);
*    hipFree(dxVal);
*    hipFree(dy);
*    hipFree(temp_buffer);
*  \endcode
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpVV(hipsparseHandle_t          handle,
                                hipsparseOperation_t       opX,
                                hipsparseConstSpVecDescr_t vecX,
                                hipsparseConstDnVecDescr_t vecY,
                                void*                      result,
                                hipDataType                computeType,
                                void*                      externalBuffer);
#elif(CUDART_VERSION > 10010 || (CUDART_VERSION == 10010 && CUDART_10_1_UPDATE_VERSION == 1))
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpVV(hipsparseHandle_t     handle,
                                hipsparseOperation_t  opX,
                                hipsparseSpVecDescr_t vecX,
                                hipsparseDnVecDescr_t vecY,
                                void*                 result,
                                hipDataType           computeType,
                                void*                 externalBuffer);
#endif

/*! \ingroup generic_module
*  \brief Buffer size step of the sparse matrix multiplication with a dense vector
*
*  \details
*  \p hipsparseSpMV_bufferSize computes the required user allocated buffer size needed when computing the 
*  sparse matrix multiplication with a dense vector:
*  \f[
*    y := \alpha \cdot op(A) \cdot x + \beta \cdot y,
*  \f]
*  where \f$op(A)\f$ is a sparse \f$m \times n\f$ matrix in CSR format, \f$x\f$ is a dense vector of length \f$n\f$ and 
*  \f$y\f$ is a dense vector of length \f$m\f$.
*
*  \ref hipsparseSpMV_bufferSize supports multiple combinations of data types and compute types. See \ref hipsparseSpMV for a complete 
*  listing of all the data type and compute type combinations available.
*
*  See \ref hipsparseSpMV for full code example.
*
*  @param[in]
*  handle              handle to the hipsparse library context queue.
*  @param[in]
*  opA                 matrix operation type.
*  @param[in]
*  alpha               scalar \f$\alpha\f$.
*  @param[in]
*  matA                matrix descriptor.
*  @param[in]
*  vecX                vector descriptor.
*  @param[in]
*  beta                scalar \f$\beta\f$.
*  @param[inout]
*  vecY                vector descriptor.
*  @param[in]
*  computeType         floating point precision for the SpMV computation.
*  @param[in]
*  alg                 SpMV algorithm for the SpMV computation.
*  @param[out]
*  pBufferSizeInBytes  number of bytes of the temporary storage buffer.
*
*  \retval      HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval      HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p alpha, \p matA, \p x, \p beta, \p y or
*               \p pBufferSizeInBytes pointer is invalid or if \p opA, \p computeType, \p alg is incorrect.
*  \retval      HIPSPARSE_STATUS_NOT_SUPPORTED \p computeType or \p alg is
*               currently not supported.
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpMV_bufferSize(hipsparseHandle_t           handle,
                                           hipsparseOperation_t        opA,
                                           const void*                 alpha,
                                           hipsparseConstSpMatDescr_t  matA,
                                           hipsparseConstDnVecDescr_t  vecX,
                                           const void*                 beta,
                                           const hipsparseDnVecDescr_t vecY,
                                           hipDataType                 computeType,
                                           hipsparseSpMVAlg_t          alg,
                                           size_t*                     pBufferSizeInBytes);
#elif(CUDART_VERSION > 10010 || (CUDART_VERSION == 10010 && CUDART_10_1_UPDATE_VERSION == 1))
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpMV_bufferSize(hipsparseHandle_t           handle,
                                           hipsparseOperation_t        opA,
                                           const void*                 alpha,
                                           const hipsparseSpMatDescr_t matA,
                                           const hipsparseDnVecDescr_t vecX,
                                           const void*                 beta,
                                           const hipsparseDnVecDescr_t vecY,
                                           hipDataType                 computeType,
                                           hipsparseSpMVAlg_t          alg,
                                           size_t*                     pBufferSizeInBytes);
#endif

/*! \ingroup generic_module
*  \brief Preprocess step of the sparse matrix multiplication with a dense vector (optional)
*
*  \details
*  \p hipsparseSpMV_preprocess performs analysis on the sparse matrix \f$A\f$ when computing the 
*  sparse matrix multiplication with a dense vector:
*  \f[
*    y := \alpha \cdot op(A) \cdot x + \beta \cdot y,
*  \f]
*  where \f$op(A)\f$ is a sparse \f$m \times n\f$ matrix in CSR format, \f$x\f$ is a dense vector of length \f$n\f$ and 
*  \f$y\f$ is a dense vector of length \f$m\f$.
*
*  This step is optional but if used may results in better performance.
*
*  \ref hipsparseSpMV_preprocess supports multiple combinations of data types and compute types. See \ref hipsparseSpMV for a complete 
*  listing of all the data type and compute type combinations available.
*
*  See \ref hipsparseSpMV for full code example.
*
*  @param[in]
*  handle          handle to the hipsparse library context queue.
*  @param[in]
*  opA             matrix operation type.
*  @param[in]
*  alpha           scalar \f$\alpha\f$.
*  @param[in]
*  matA            matrix descriptor.
*  @param[in]
*  vecX            vector descriptor.
*  @param[in]
*  beta            scalar \f$\beta\f$.
*  @param[inout]
*  vecY            vector descriptor.
*  @param[in]
*  computeType     floating point precision for the SpMV computation.
*  @param[in]
*  alg             SpMV algorithm for the SpMV computation.
*  @param[out]
*  externalBuffer  temporary storage buffer allocated by the user.
*
*  \retval      HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval      HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p alpha, \p matA, \p x, \p beta, \p y or
*               \p externalBuffer pointer is invalid or if \p opA, \p computeType, \p alg is incorrect.
*  \retval      HIPSPARSE_STATUS_NOT_SUPPORTED \p computeType or \p alg is
*               currently not supported.
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpMV_preprocess(hipsparseHandle_t           handle,
                                           hipsparseOperation_t        opA,
                                           const void*                 alpha,
                                           hipsparseConstSpMatDescr_t  matA,
                                           hipsparseConstDnVecDescr_t  vecX,
                                           const void*                 beta,
                                           const hipsparseDnVecDescr_t vecY,
                                           hipDataType                 computeType,
                                           hipsparseSpMVAlg_t          alg,
                                           void*                       externalBuffer);
#elif(CUDART_VERSION > 10010 || (CUDART_VERSION == 10010 && CUDART_10_1_UPDATE_VERSION == 1))
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpMV_preprocess(hipsparseHandle_t           handle,
                                           hipsparseOperation_t        opA,
                                           const void*                 alpha,
                                           const hipsparseSpMatDescr_t matA,
                                           const hipsparseDnVecDescr_t vecX,
                                           const void*                 beta,
                                           const hipsparseDnVecDescr_t vecY,
                                           hipDataType                 computeType,
                                           hipsparseSpMVAlg_t          alg,
                                           void*                       externalBuffer);
#endif

/*! \ingroup generic_module
*  \brief Compute the sparse matrix multiplication with a dense vector
*
*  \details
*  \p hipsparseSpMV computes sparse matrix multiplication with a dense vector:
*  \f[
*    y := \alpha \cdot op(A) \cdot x + \beta \cdot y,
*  \f]
*  where \f$op(A)\f$ is a sparse \f$m \times n\f$ matrix in CSR format, \f$x\f$ is a dense vector of length \f$n\f$ and 
*  \f$y\f$ is a dense vector of length \f$m\f$.
*
*  \ref hipsparseSpMV supports multiple combinations of data types and compute types. The tables below indicate the currently
*  supported different data types that can be used for for the sparse matrix A and the dense vectors X and Y and the compute
*  type for \f$\alpha\f$ and \f$\beta\f$. The advantage of using different data types is to save on memory bandwidth and storage
*  when a user application allows while performing the actual computation in a higher precision.
*
*  \par Uniform Precisions:
*  <table>
*  <caption id="spmv_uniform">Uniform Precisions</caption>
*  <tr><th>A / X / Y / compute_type
*  <tr><td>HIP_R_32F
*  <tr><td>HIP_R_64F
*  <tr><td>HIP_C_32F
*  <tr><td>HIP_C_64F
*  </table>
*
*  \par Mixed precisions:
*  <table>
*  <caption id="spmv_mixed">Mixed Precisions</caption>
*  <tr><th>A / X    <th>Y         <th>compute_type
*  <tr><td>HIP_R_8I <td>HIP_R_32I <td>HIP_R_32I
*  <tr><td>HIP_R_8I <td>HIP_R_32F <td>HIP_R_32F
*  </table>
*
*  \par Mixed-regular real precisions
*  <table>
*  <caption id="spmv_mixed_regular_real">Mixed-regular real precisions</caption>
*  <tr><th>A         <th>X / Y / compute_type
*  <tr><td>HIP_R_32F <td>HIP_R_64F
*  <tr><td>HIP_C_32F <td>HIP_C_64F
*  </table>
*
*  \par Mixed-regular Complex precisions
*  <table>
*  <caption id="spmv_mixed_regular_complex">Mixed-regular Complex precisions</caption>
*  <tr><th>A         <th>X / Y / compute_type
*  <tr><td>HIP_R_32F <td>HIP_C_32F
*  <tr><td>HIP_R_64F <td>HIP_C_64F
*  </table>
*
*  @param[in]
*  handle          handle to the hipsparse library context queue.
*  @param[in]
*  opA             matrix operation type.
*  @param[in]
*  alpha           scalar \f$\alpha\f$.
*  @param[in]
*  matA            matrix descriptor.
*  @param[in]
*  vecX            vector descriptor.
*  @param[in]
*  beta            scalar \f$\beta\f$.
*  @param[inout]
*  vecY            vector descriptor.
*  @param[in]
*  computeType     floating point precision for the SpMV computation.
*  @param[in]
*  alg             SpMV algorithm for the SpMV computation.
*  @param[out]
*  externalBuffer  temporary storage buffer allocated by the user.
*
*  \retval      HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval      HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p alpha, \p matA, \p x, \p beta, \p y or
*               \p externalBuffer pointer is invalid or if \p opA, \p computeType, \p alg is incorrect.
*  \retval      HIPSPARSE_STATUS_NOT_SUPPORTED \p computeType or \p alg is
*               currently not supported.
*
*  \par Example
*  \code{.c}
*    // A, x, and y are m×k, k×1, and m×1
*    int m = 3, k = 4;
*    int nnz_A = 8;
*    hipsparseOperation_t transA = HIPSPARSE_OPERATION_NON_TRANSPOSE;
*
*    // alpha and beta
*    float alpha = 0.5f;
*    float beta  = 0.25f;
*
*    std::vector<int> hcsrRowPtr = {0, 3, 5, 8};
*    std::vector<int> hcsrColInd = {0, 1, 3, 1, 2, 0, 2, 3}; 
*    std::vector<float> hcsrVal     = {1, 2, 3, 4, 5, 6, 7, 8}; 
*
*    std::vector<float> hx(k, 1.0f);
*    std::vector<float> hy(m, 1.0f);
*
*    int *dcsrRowPtr;
*    int *dcsrColInd;
*    float *dcsrVal;
*    hipMalloc((void**)&dcsrRowPtr, sizeof(int) * (m + 1));
*    hipMalloc((void**)&dcsrColInd, sizeof(int) * nnz_A);
*    hipMalloc((void**)&dcsrVal, sizeof(float) * nnz_A);
*
*    hipMemcpy(dcsrRowPtr, hcsrRowPtr.data(), sizeof(int) * (m + 1), hipMemcpyHostToDevice);
*    hipMemcpy(dcsrColInd, hcsrColInd.data(), sizeof(int) * nnz_A, hipMemcpyHostToDevice);
*    hipMemcpy(dcsrVal, hcsrVal.data(), sizeof(float) * nnz_A, hipMemcpyHostToDevice);
*
*    hipsparseHandle_t handle;
*    hipsparseCreate(&handle);
*
*    hipsparseSpMatDescr_t matA;
*    hipsparseCreateCsr(&matA, m, k, nnz_A,
*                        dcsrRowPtr, dcsrColInd, dcsrVal,
*                        HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_32I,
*                        HIPSPARSE_INDEX_BASE_ZERO, HIP_R_32F);
*
*    // Allocate memory for the vector x
*    float* dx;
*    hipMalloc((void**)&dx, sizeof(float) * k);
*    hipMemcpy(dx, hx.data(), sizeof(float) * k, hipMemcpyHostToDevice);
*
*    hipsparseDnVecDescr_t vecX;
*    hipsparseCreateDnVec(&vecX, k, dx, HIP_R_32F);
*
*    // Allocate memory for the resulting vector y
*    float* dy;
*    hipMalloc((void**)&dy, sizeof(float) * m);
*    hipMemcpy(dy, hy.data(), sizeof(float) * m, hipMemcpyHostToDevice);
*
*    hipsparseDnMatDescr_t vecY;
*    hipsparseCreateDnVec(&vecY, m, dy, HIP_R_32F);
*
*    // Compute buffersize
*    size_t bufferSize;
*    hipsparseSpMV_bufferSize(handle,
*                             transA,
*                             &alpha,
*                             matA,
*                             vecX,
*                             &beta,
*                             vecY,
*                             HIP_R_32F,
*                             HIPSPARSE_MV_ALG_DEFAULT,
*                             &bufferSize);
*
*    void* buffer;
*    hipMalloc(&buffer, bufferSize);
*
*    // Preprocess operation (Optional)
*    hipsparseSpMV_preprocess(handle,
*                            transA,
*                            &alpha,
*                            matA,
*                            vecX,
*                            &beta,
*                            vecY,
*                            HIP_R_32F,
*                            HIPSPARSE_MV_ALG_DEFAULT,
*                            &buffer);
*
*    // Perform operation
*    hipsparseSpMV(handle,
*                 transA,
*                 &alpha,
*                 matA,
*                 vecX,
*                 &beta,
*                 vecY,
*                 HIP_R_32F,
*                 HIPSPARSE_MV_ALG_DEFAULT,
*                 &buffer);
*
*    // Copy device to host
*    hipMemcpy(hy.data(), dy, sizeof(float) * m, hipMemcpyDeviceToHost);
*
*    // Destroy matrix descriptors and handles
*    hipsparseDestroySpMat(matA);
*    hipsparseDestroyDnVec(vecX);
*    hipsparseDestroyDnVec(vecY);
*    hipsparseDestroy(handle);
*
*    hipFree(buffer);
*    hipFree(dcsrRowPtr);
*    hipFree(dcsrColInd);
*    hipFree(dcsrVal);
*    hipFree(dx);
*    hipFree(dy);
*  \endcode
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpMV(hipsparseHandle_t           handle,
                                hipsparseOperation_t        opA,
                                const void*                 alpha,
                                hipsparseConstSpMatDescr_t  matA,
                                hipsparseConstDnVecDescr_t  vecX,
                                const void*                 beta,
                                const hipsparseDnVecDescr_t vecY,
                                hipDataType                 computeType,
                                hipsparseSpMVAlg_t          alg,
                                void*                       externalBuffer);
#elif(CUDART_VERSION > 10010 || (CUDART_VERSION == 10010 && CUDART_10_1_UPDATE_VERSION == 1))
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpMV(hipsparseHandle_t           handle,
                                hipsparseOperation_t        opA,
                                const void*                 alpha,
                                const hipsparseSpMatDescr_t matA,
                                const hipsparseDnVecDescr_t vecX,
                                const void*                 beta,
                                const hipsparseDnVecDescr_t vecY,
                                hipDataType                 computeType,
                                hipsparseSpMVAlg_t          alg,
                                void*                       externalBuffer);
#endif

/*! \ingroup generic_module
*  \brief Calculate the buffer size required for the sparse matrix multiplication with a dense matrix
*
*  \details
*  \p hipsparseSpMM_bufferSize computes the required user allocated buffer size needed when computing the 
*  sparse matrix multiplication with a dense matrix:
*  \f[
*    C := \alpha \cdot op(A) \cdot op(B) + \beta \cdot C,
*  \f]
*  where \f$op(A)\f$ is a sparse \f$m \times k\f$ matrix in CSR format, \f$B\f$ is a dense matrix of size \f$k \times n\f$ and 
*  \f$C\f$ is a dense matrix of size \f$m \times n\f$.
*
*  \ref hipsparseSpMM_bufferSize supports multiple combinations of data types and compute types. See \ref hipsparseSpMM for a complete 
*  listing of all the data type and compute type combinations available.
*
*  See \ref hipsparseSpMM for full code example.
*
*  @param[in]
*  handle              handle to the hipsparse library context queue.
*  @param[in]
*  opA                 matrix operation type.
*  @param[in]
*  opB                 matrix operation type.
*  @param[in]
*  alpha               scalar \f$\alpha\f$.
*  @param[in]
*  matA                matrix descriptor.
*  @param[in]
*  matB                matrix descriptor.
*  @param[in]
*  beta                scalar \f$\beta\f$.
*  @param[in]
*  matC                matrix descriptor.
*  @param[in]
*  computeType         floating point precision for the SpMM computation.
*  @param[in]
*  alg                 SpMM algorithm for the SpMM computation.
*  @param[out]
*  pBufferSizeInBytes  number of bytes of the temporary storage buffer.
*
*  \retval      HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval      HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p alpha, \p matA, \p matB, \p matC, \p beta, or
*               \p pBufferSizeInBytes pointer is invalid.
*  \retval      HIPSPARSE_STATUS_NOT_SUPPORTED \p opA, \p opB, \p computeType or \p alg is
*               currently not supported.
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpMM_bufferSize(hipsparseHandle_t           handle,
                                           hipsparseOperation_t        opA,
                                           hipsparseOperation_t        opB,
                                           const void*                 alpha,
                                           hipsparseConstSpMatDescr_t  matA,
                                           hipsparseConstDnMatDescr_t  matB,
                                           const void*                 beta,
                                           const hipsparseDnMatDescr_t matC,
                                           hipDataType                 computeType,
                                           hipsparseSpMMAlg_t          alg,
                                           size_t*                     pBufferSizeInBytes);
#elif(CUDART_VERSION >= 10010)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpMM_bufferSize(hipsparseHandle_t           handle,
                                           hipsparseOperation_t        opA,
                                           hipsparseOperation_t        opB,
                                           const void*                 alpha,
                                           const hipsparseSpMatDescr_t matA,
                                           const hipsparseDnMatDescr_t matB,
                                           const void*                 beta,
                                           const hipsparseDnMatDescr_t matC,
                                           hipDataType                 computeType,
                                           hipsparseSpMMAlg_t          alg,
                                           size_t*                     pBufferSizeInBytes);
#endif

/*! \ingroup generic_module
*  \brief Preprocess step of the sparse matrix multiplication with a dense matrix.
*
*  \details
*  \p hipsparseSpMM_preprocess performs the required preprocessing used when computing the 
*  sparse matrix multiplication with a dense matrix:
*  \f[
*    C := \alpha \cdot op(A) \cdot op(B) + \beta \cdot C,
*  \f]
*  where \f$op(A)\f$ is a sparse \f$m \times k\f$ matrix in CSR format, \f$B\f$ is a dense matrix of size \f$k \times n\f$ and 
*  \f$C\f$ is a dense matrix of size \f$m \times n\f$.
*
*  \ref hipsparseSpMM_preprocess supports multiple combinations of data types and compute types. See \ref hipsparseSpMM for a complete 
*  listing of all the data type and compute type combinations available.
*
*  See \ref hipsparseSpMM for full code example.
*
*  @param[in]
*  handle          handle to the hipsparse library context queue.
*  @param[in]
*  opA             matrix operation type.
*  @param[in]
*  opB             matrix operation type.
*  @param[in]
*  alpha           scalar \f$\alpha\f$.
*  @param[in]
*  matA            matrix descriptor.
*  @param[in]
*  matB            matrix descriptor.
*  @param[in]
*  beta            scalar \f$\beta\f$.
*  @param[in]
*  matC            matrix descriptor.
*  @param[in]
*  computeType     floating point precision for the SpMM computation.
*  @param[in]
*  alg             SpMM algorithm for the SpMM computation.
*  @param[out]
*  externalBuffer  temporary storage buffer allocated by the user.
*
*  \retval      HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval      HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p alpha, \p matA, \p matB, \p matC, \p beta, or
*               \p externalBuffer pointer is invalid.
*  \retval      HIPSPARSE_STATUS_NOT_SUPPORTED \p opA, \p opB, \p computeType or \p alg is
*               currently not supported.
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpMM_preprocess(hipsparseHandle_t           handle,
                                           hipsparseOperation_t        opA,
                                           hipsparseOperation_t        opB,
                                           const void*                 alpha,
                                           hipsparseConstSpMatDescr_t  matA,
                                           hipsparseConstDnMatDescr_t  matB,
                                           const void*                 beta,
                                           const hipsparseDnMatDescr_t matC,
                                           hipDataType                 computeType,
                                           hipsparseSpMMAlg_t          alg,
                                           void*                       externalBuffer);
#elif(CUDART_VERSION >= 11021)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpMM_preprocess(hipsparseHandle_t           handle,
                                           hipsparseOperation_t        opA,
                                           hipsparseOperation_t        opB,
                                           const void*                 alpha,
                                           const hipsparseSpMatDescr_t matA,
                                           const hipsparseDnMatDescr_t matB,
                                           const void*                 beta,
                                           const hipsparseDnMatDescr_t matC,
                                           hipDataType                 computeType,
                                           hipsparseSpMMAlg_t          alg,
                                           void*                       externalBuffer);
#endif

/*! \ingroup generic_module
*  \brief Compute the sparse matrix multiplication with a dense matrix
*
*  \details
*  \p hipsparseSpMM computes sparse matrix multiplication with a dense matrix:
*  \f[
*    C := \alpha \cdot op(A) \cdot op(B) + \beta \cdot C,
*  \f]
*  where \f$op(A)\f$ is a sparse \f$m \times k\f$ matrix in CSR format, \f$B\f$ is a dense matrix of size \f$k \times n\f$ and 
*  \f$C\f$ is a dense matrix of size \f$m \times n\f$.
*
*  \ref hipsparseSpMM supports multiple combinations of data types and compute types. The tables below indicate the currently
*  supported different data types that can be used for for the sparse matrix A and the dense matrices B and C and the compute
*  type for \f$\alpha\f$ and \f$\beta\f$. The advantage of using different data types is to save on memory bandwidth and storage
*  when a user application allows while performing the actual computation in a higher precision.
*
*  \par Uniform Precisions:
*  <table>
*  <caption id="spmm_uniform">Uniform Precisions</caption>
*  <tr><th>A / B / C / compute_type
*  <tr><td>HIP_R_32F
*  <tr><td>HIP_R_64F
*  <tr><td>HIP_C_32F
*  <tr><td>HIP_C_64F
*  </table>
*
*  \par Mixed precisions:
*  <table>
*  <caption id="spmm_mixed">Mixed Precisions</caption>
*  <tr><th>A / B    <th>C         <th>compute_type
*  <tr><td>HIP_R_8I <td>HIP_R_32I <td>HIP_R_32I
*  <tr><td>HIP_R_8I <td>HIP_R_32F <td>HIP_R_32F
*  </table>
*
*  @param[in]
*  handle          handle to the hipsparse library context queue.
*  @param[in]
*  opA             matrix operation type.
*  @param[in]
*  opB             matrix operation type.
*  @param[in]
*  alpha           scalar \f$\alpha\f$.
*  @param[in]
*  matA            matrix descriptor.
*  @param[in]
*  matB            matrix descriptor.
*  @param[in]
*  beta            scalar \f$\beta\f$.
*  @param[in]
*  matC            matrix descriptor.
*  @param[in]
*  computeType     floating point precision for the SpMM computation.
*  @param[in]
*  alg             SpMM algorithm for the SpMM computation.
*  @param[out]
*  externalBuffer  temporary storage buffer allocated by the user.
*
*  \retval      HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval      HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p alpha, \p matA, \p matB, \p matC, \p beta, or
*               \p externalBuffer pointer is invalid.
*  \retval      HIPSPARSE_STATUS_NOT_SUPPORTED \p opA, \p opB, \p computeType or \p alg is
*               currently not supported.
*
*  \par Example
*  \code{.c}
*    // A, B, and C are m×k, k×n, and m×n
*    int m = 3, n = 5, k = 4;
*    int ldb = n, ldc = n;
*    int nnz_A = 8, nnz_B = 20, nnz_C = 15;
*    hipsparseOperation_t transA = HIPSPARSE_OPERATION_NON_TRANSPOSE;
*    hipsparseOperation_t transB = HIPSPARSE_OPERATION_NON_TRANSPOSE;
*    hipsparseOperation_t transC = HIPSPARSE_OPERATION_NON_TRANSPOSE;
*    hipsparseOrder_t order = HIPSPARSE_ORDER_ROW;
*
*    // alpha and beta
*    float alpha = 0.5f;
*    float beta  = 0.25f;
*
*    std::vector<int> hcsrRowPtr = {0, 3, 5, 8};
*    std::vector<int> hcsrColInd = {0, 1, 3, 1, 2, 0, 2, 3}; 
*    std::vector<float> hcsrVal     = {1, 2, 3, 4, 5, 6, 7, 8}; 
*
*    std::vector<float> hB(nnz_B, 1.0f);
*    std::vector<float> hC(nnz_C, 1.0f);
*
*    int *dcsrRowPtr;
*    int *dcsrColInd;
*    float *dcsrVal;
*    hipMalloc((void**)&dcsrRowPtr, sizeof(int) * (m + 1));
*    hipMalloc((void**)&dcsrColInd, sizeof(int) * nnz_A);
*    hipMalloc((void**)&dcsrVal, sizeof(float) * nnz_A);
*
*    hipMemcpy(dcsrRowPtr, hcsrRowPtr.data(), sizeof(int) * (m + 1), hipMemcpyHostToDevice);
*    hipMemcpy(dcsrColInd, hcsrColInd.data(), sizeof(int) * nnz_A, hipMemcpyHostToDevice);
*    hipMemcpy(dcsrVal, hcsrVal.data(), sizeof(float) * nnz_A, hipMemcpyHostToDevice);
*
*    hipsparseHandle_t handle;
*    hipsparseCreate(&handle);
*
*    hipsparseSpMatDescr_t matA;
*    hipsparseCreateCsr(&matA, m, k, nnz_A,
*                        dcsrRowPtr, dcsrColInd, dcsrVal,
*                        HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_32I,
*                        HIPSPARSE_INDEX_BASE_ZERO, HIP_R_32F);
*
*    // Allocate memory for the matrix B
*    float* dB;
*    hipMalloc((void**)&dB, sizeof(float) * nnz_B);
*    hipMemcpy(dB, hB.data(), sizeof(float) * nnz_B, hipMemcpyHostToDevice);
*
*    hipsparseDnMatDescr_t matB;
*    hipsparseCreateDnMat(&matB, k, n, ldb, dB, HIP_R_32F, order);
*
*    // Allocate memory for the resulting matrix C
*    float* dC;
*    hipMalloc((void**)&dC, sizeof(float) * nnz_C);
*    hipMemcpy(dC, hC.data(), sizeof(float) * nnz_C, hipMemcpyHostToDevice);
*
*    hipsparseDnMatDescr_t matC;
*    hipsparseCreateDnMat(&matC, m, n, ldc, dC, HIP_R_32F, HIPSPARSE_ORDER_ROW);
*
*    // Compute buffersize
*    size_t bufferSize;
*    hipsparseSpMM_bufferSize(handle,
*                             transA,
*                             transB,
*                             &alpha,
*                             matA,
*                             matB,
*                             &beta,
*                             matC,
*                             HIP_R_32F,
*                             HIPSPARSE_MM_ALG_DEFAULT,
*                             &bufferSize);
*
*    void* buffer;
*    hipMalloc(&buffer, bufferSize);
*
*    // Preprocess operation (Optional)
*    hipsparseSpMM_preprocess(handle,
*                            transA,
*                            transB,
*                            &alpha,
*                            matA,
*                            matB,
*                            &beta,
*                            matC,
*                            HIP_R_32F,
*                            HIPSPARSE_MM_ALG_DEFAULT,
*                            &buffer);
*
*    // Perform operation
*    hipsparseSpMM(handle,
*                 transA,
*                 transB,
*                 &alpha,
*                 matA,
*                 matB,
*                 &beta,
*                 matC,
*                 HIP_R_32F,
*                 HIPSPARSE_MM_ALG_DEFAULT,
*                 &buffer);
*
*    // Copy device to host
*    hipMemcpy(hC.data(), dC, sizeof(float) * nnz_C, hipMemcpyDeviceToHost);
*
*    // Destroy matrix descriptors and handles
*    hipsparseDestroySpMat(matA);
*    hipsparseDestroyDnMat(matB);
*    hipsparseDestroyDnMat(matC);
*    hipsparseDestroy(handle);
*
*    hipFree(buffer);
*    hipFree(dcsrRowPtr);
*    hipFree(dcsrColInd);
*    hipFree(dcsrVal);
*    hipFree(dB);
*    hipFree(dC);
*  \endcode
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpMM(hipsparseHandle_t           handle,
                                hipsparseOperation_t        opA,
                                hipsparseOperation_t        opB,
                                const void*                 alpha,
                                hipsparseConstSpMatDescr_t  matA,
                                hipsparseConstDnMatDescr_t  matB,
                                const void*                 beta,
                                const hipsparseDnMatDescr_t matC,
                                hipDataType                 computeType,
                                hipsparseSpMMAlg_t          alg,
                                void*                       externalBuffer);
#elif(CUDART_VERSION >= 10010)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpMM(hipsparseHandle_t           handle,
                                hipsparseOperation_t        opA,
                                hipsparseOperation_t        opB,
                                const void*                 alpha,
                                const hipsparseSpMatDescr_t matA,
                                const hipsparseDnMatDescr_t matB,
                                const void*                 beta,
                                const hipsparseDnMatDescr_t matC,
                                hipDataType                 computeType,
                                hipsparseSpMMAlg_t          alg,
                                void*                       externalBuffer);
#endif

/*! \ingroup generic_module
*  \brief Create sparse matrix sparse matrix product descriptor
*  \details
*  \p hipsparseSpGEMM_createDescr creates a sparse matrix sparse matrix product descriptor. It should be
*  destroyed at the end using hipsparseSpGEMM_destroyDescr().
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpGEMM_createDescr(hipsparseSpGEMMDescr_t* descr);
#endif

/*! \ingroup generic_module
*  \brief Destroy sparse matrix sparse matrix product descriptor
*  \details
*  \p hipsparseSpGEMM_destroyDescr destroys a sparse matrix sparse matrix product descriptor and releases all
*  resources used by the descriptor.
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpGEMM_destroyDescr(hipsparseSpGEMMDescr_t descr);
#endif

/*! \ingroup generic_module
*  \brief Work estimation step of the sparse matrix sparse matrix product:
*  \f[
*    C' := \alpha \cdot op(A) \cdot op(B) + \beta \cdot C,
*  \f]
*  where \f$C'\f$, \f$A\f$, \f$B\f$, \f$C\f$ are sparse matrices and \f$C'\f$ and \f$C\f$ have the same sparsity pattern.
*
*  \details
*  \p hipsparseSpGEMM_workEstimation is called twice. We call it to compute the size of the first required user allocated
*  buffer. After this buffer size is determined, the user allocates it and calls \p hipsparseSpGEMM_workEstimation
*  a second time with the newly allocated buffer passed in. This second call inspects the matrices \f$A\f$ and \f$B\f$ to 
*  determine the number of intermediate products that will result from multipltying \f$A\f$ and \f$B\f$ together.
*  
*  @param[in]
*  handle           handle to the hipsparse library context queue.
*  @param[in]
*  opA              sparse matrix \f$A\f$ operation type.
*  @param[in]
*  opB              sparse matrix \f$B\f$ operation type.
*  @param[in]
*  alpha            scalar \f$\alpha\f$.
*  @param[in]
*  matA             sparse matrix \f$A\f$ descriptor.
*  @param[in]
*  matB             sparse matrix \f$B\f$ descriptor.
*  @param[in]
*  beta             scalar \f$\beta\f$.
*  @param[out]
*  matC             sparse matrix \f$C\f$ descriptor.
*  @param[in]
*  computeType      floating point precision for the SpGEMM computation.
*  @param[in]
*  alg              SpGEMM algorithm for the SpGEMM computation.
*  @param[in]
*  spgemmDescr      SpGEMM descriptor.
*  @param[out]
*  bufferSize1      number of bytes of the temporary storage buffer. 
*  @param[in]
*  externalBuffer1  temporary storage buffer allocated by the user.
*
*  \retval HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p alpha, \p beta, \p matA, \p matB, \p matC 
*                                         or \p bufferSize1 pointer is invalid.
*  \retval HIPSPARSE_STATUS_ALLOC_FAILED additional buffer for long rows could not be
*          allocated.
*  \retval HIPSPARSE_STATUS_NOT_SUPPORTED
*          \p opA != \ref HIPSPARSE_OPERATION_NON_TRANSPOSE or
*          \p opB != \ref HIPSPARSE_OPERATION_NON_TRANSPOSE.
*
*  \par Example (See full example below)
*  \code{.c}
*    void*  dBuffer1  = NULL; 
*    size_t bufferSize1 = 0;
*
*    hipsparseSpGEMMDescr_t spgemmDesc;
*    hipsparseSpGEMM_createDescr(&spgemmDesc);
*
*    size_t bufferSize1 = 0;
*    hipsparseSpGEMM_workEstimation(handle, opA, opB,
*                                  &alpha, matA, matB, &beta, matC,
*                                  computeType, HIPSPARSE_SPGEMM_DEFAULT,
*                                  spgemmDesc, &bufferSize1, NULL);
*    hipMalloc((void**) &dBuffer1, bufferSize1);
*
*    // Determine number of intermediate product when computing A * B
*    hipsparseSpGEMM_workEstimation(handle, opA, opB,
*                                    &alpha, matA, matB, &beta, matC,
*                                    computeType, HIPSPARSE_SPGEMM_DEFAULT,
*                                    spgemmDesc, &bufferSize1, dBuffer1);
*  \endcode
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpGEMM_workEstimation(hipsparseHandle_t          handle,
                                                 hipsparseOperation_t       opA,
                                                 hipsparseOperation_t       opB,
                                                 const void*                alpha,
                                                 hipsparseConstSpMatDescr_t matA,
                                                 hipsparseConstSpMatDescr_t matB,
                                                 const void*                beta,
                                                 hipsparseSpMatDescr_t      matC,
                                                 hipDataType                computeType,
                                                 hipsparseSpGEMMAlg_t       alg,
                                                 hipsparseSpGEMMDescr_t     spgemmDescr,
                                                 size_t*                    bufferSize1,
                                                 void*                      externalBuffer1);
#elif(CUDART_VERSION >= 11000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpGEMM_workEstimation(hipsparseHandle_t      handle,
                                                 hipsparseOperation_t   opA,
                                                 hipsparseOperation_t   opB,
                                                 const void*            alpha,
                                                 hipsparseSpMatDescr_t  matA,
                                                 hipsparseSpMatDescr_t  matB,
                                                 const void*            beta,
                                                 hipsparseSpMatDescr_t  matC,
                                                 hipDataType            computeType,
                                                 hipsparseSpGEMMAlg_t   alg,
                                                 hipsparseSpGEMMDescr_t spgemmDescr,
                                                 size_t*                bufferSize1,
                                                 void*                  externalBuffer1);
#endif

/*! \ingroup generic_module
*  \brief Compute step of the sparse matrix sparse matrix product:
*  \f[
*    C' := \alpha \cdot op(A) \cdot op(B) + \beta \cdot C,
*  \f]
*  where \f$C'\f$, \f$A\f$, \f$B\f$, \f$C\f$ are sparse matrices and \f$C'\f$ and \f$C\f$ have the same sparsity pattern.
*
*  \details
*  \p hipsparseSpGEMM_compute is called twice. First to compute the size of the second required user allocated
*  buffer. After this buffer size is determined, the user allocates it and calls \p hipsparseSpGEMM_compute
*  a second time with the newly allocated buffer passed in. This second call performs the actual computation 
*  of \f$C' = \alpha \cdot A \cdot B\f$ (the result is stored in the temporary buffers).
*  
*  @param[in]
*  handle           handle to the hipsparse library context queue.
*  @param[in]
*  opA              sparse matrix \f$A\f$ operation type.
*  @param[in]
*  opB              sparse matrix \f$B\f$ operation type.
*  @param[in]
*  alpha            scalar \f$\alpha\f$.
*  @param[in]
*  matA             sparse matrix \f$A\f$ descriptor.
*  @param[in]
*  matB             sparse matrix \f$B\f$ descriptor.
*  @param[in]
*  beta             scalar \f$\beta\f$.
*  @param[out]
*  matC             sparse matrix \f$C\f$ descriptor.
*  @param[in]
*  computeType      floating point precision for the SpGEMM computation.
*  @param[in]
*  alg              SpGEMM algorithm for the SpGEMM computation.
*  @param[in]
*  spgemmDescr      SpGEMM descriptor.
*  @param[out]
*  bufferSize2      number of bytes of the temporary storage buffer. 
*  @param[in]
*  externalBuffer2  temporary storage buffer allocated by the user.
*
*  \retval HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p alpha, \p beta, \p matA, \p matB, \p matC 
*                                         or \p bufferSize2 pointer is invalid.
*  \retval HIPSPARSE_STATUS_ALLOC_FAILED additional buffer for long rows could not be
*          allocated.
*  \retval HIPSPARSE_STATUS_NOT_SUPPORTED
*          \p opA != \ref HIPSPARSE_OPERATION_NON_TRANSPOSE or
*          \p opB != \ref HIPSPARSE_OPERATION_NON_TRANSPOSE.
*
*  \par Example (See full example below)
*  \code{.c}
*    void*  dBuffer2  = NULL; 
*    size_t bufferSize2 = 0;
*
*    size_t bufferSize2 = 0;
*    hipsparseSpGEMM_compute(handle, opA, opB,
*                            &alpha, matA, matB, &beta, matC,
*                            computeType, HIPSPARSE_SPGEMM_DEFAULT,
*                            spgemmDesc, &bufferSize2, NULL);
*    hipMalloc((void**) &dBuffer2, bufferSize2);
*
*    // compute the intermediate product of A * B
*    hipsparseSpGEMM_compute(handle, opA, opB,
*                            &alpha, matA, matB, &beta, matC,
*                            computeType, HIPSPARSE_SPGEMM_DEFAULT,
*                            spgemmDesc, &bufferSize2, dBuffer2);
*  \endcode
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpGEMM_compute(hipsparseHandle_t          handle,
                                          hipsparseOperation_t       opA,
                                          hipsparseOperation_t       opB,
                                          const void*                alpha,
                                          hipsparseConstSpMatDescr_t matA,
                                          hipsparseConstSpMatDescr_t matB,
                                          const void*                beta,
                                          hipsparseSpMatDescr_t      matC,
                                          hipDataType                computeType,
                                          hipsparseSpGEMMAlg_t       alg,
                                          hipsparseSpGEMMDescr_t     spgemmDescr,
                                          size_t*                    bufferSize2,
                                          void*                      externalBuffer2);
#elif(CUDART_VERSION >= 11000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpGEMM_compute(hipsparseHandle_t      handle,
                                          hipsparseOperation_t   opA,
                                          hipsparseOperation_t   opB,
                                          const void*            alpha,
                                          hipsparseSpMatDescr_t  matA,
                                          hipsparseSpMatDescr_t  matB,
                                          const void*            beta,
                                          hipsparseSpMatDescr_t  matC,
                                          hipDataType            computeType,
                                          hipsparseSpGEMMAlg_t   alg,
                                          hipsparseSpGEMMDescr_t spgemmDescr,
                                          size_t*                bufferSize2,
                                          void*                  externalBuffer2);
#endif

/*! \ingroup generic_module
*  \brief Copy step of the sparse matrix sparse matrix product: 
*  \f[
*    C' := \alpha \cdot op(A) \cdot op(B) + \beta \cdot C,
*  \f]
*  where \f$C'\f$, \f$A\f$, \f$B\f$, \f$C\f$ are sparse matrices and \f$C'\f$ and \f$C\f$ have the same sparsity pattern.
*
*  \details
*  \p hipsparseSpGEMM_copy is called once to copy the results (that are currently stored in the temporary arrays) 
*  to the output sparse matrix. If \f$\beta != 0\f$, then the \f$beta \cdot C\f$ portion of the computation: 
*  \f$C' = \alpha \cdot A \cdot B + \beta * C\f$ is handled. This is possible because \f$C'\f$ and \f$C\f$ must have 
*  the same sparsity pattern.
*
*  \note The two user allocated temporary buffers can only be freed after the call to \p hipsparseSpGEMM_copy
*
*  @param[in]
*  handle           handle to the hipsparse library context queue.
*  @param[in]
*  opA              sparse matrix \f$A\f$ operation type.
*  @param[in]
*  opB              sparse matrix \f$B\f$ operation type.
*  @param[in]
*  alpha            scalar \f$\alpha\f$.
*  @param[in]
*  matA             sparse matrix \f$A\f$ descriptor.
*  @param[in]
*  matB             sparse matrix \f$B\f$ descriptor.
*  @param[in]
*  beta             scalar \f$\beta\f$.
*  @param[out]
*  matC             sparse matrix \f$C\f$ descriptor.
*  @param[in]
*  computeType      floating point precision for the SpGEMM computation.
*  @param[in]
*  alg              SpGEMM algorithm for the SpGEMM computation.
*  @param[in]
*  spgemmDescr      SpGEMM descriptor.
*
*  \retval HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p alpha, \p beta, \p matA, \p matB, \p matC pointer is invalid.
*  \retval HIPSPARSE_STATUS_ALLOC_FAILED additional buffer for long rows could not be
*          allocated.
*  \retval HIPSPARSE_STATUS_NOT_SUPPORTED
*          \p opA != \ref HIPSPARSE_OPERATION_NON_TRANSPOSE or
*          \p opB != \ref HIPSPARSE_OPERATION_NON_TRANSPOSE.
*  
*  \par Example (Full example)
*  \code{.c}
*    hipsparseHandle_t     handle = NULL;
*    hipsparseSpMatDescr_t matA, matB, matC;
*    void*  dBuffer1  = NULL; 
*    void*  dBuffer2  = NULL;
*    size_t bufferSize1 = 0;  
*    size_t bufferSize2 = 0;
*
*    hipsparseCreate(&handle);
*
*    // Create sparse matrix A in CSR format
*    hipsparseCreateCsr(&matA, m, k, nnzA,
*                                        dcsrRowPtrA, dcsrColIndA, dcsrValA,
*                                        HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_32I,
*                                        HIPSPARSE_INDEX_BASE_ZERO, HIP_R_32F);
*    hipsparseCreateCsr(&matB, k, n, nnzB,
*                                        dcsrRowPtrB, dcsrColIndB, dcsrValB,
*                                        HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_32I,
*                                        HIPSPARSE_INDEX_BASE_ZERO, HIP_R_32F);
*    hipsparseCreateCsr(&matC, m, n, 0,
*                                        dcsrRowPtrC, NULL, NULL,
*                                        HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_32I,
*                                        HIPSPARSE_INDEX_BASE_ZERO, HIP_R_32F);
*
*    hipsparseSpGEMMDescr_t spgemmDesc;
*    hipsparseSpGEMM_createDescr(&spgemmDesc);
*
*    // Determine size of first user allocated buffer
*    hipsparseSpGEMM_workEstimation(handle, opA, opB,
*                                        &alpha, matA, matB, &beta, matC,
*                                        computeType, HIPSPARSE_SPGEMM_DEFAULT,
*                                        spgemmDesc, &bufferSize1, NULL);
*    hipMalloc((void**) &dBuffer1, bufferSize1);
*
*    // Inspect the matrices A and B to determine the number of intermediate product in 
*    // C = alpha * A * B
*    hipsparseSpGEMM_workEstimation(handle, opA, opB,
*                                        &alpha, matA, matB, &beta, matC,
*                                        computeType, HIPSPARSE_SPGEMM_DEFAULT,
*                                        spgemmDesc, &bufferSize1, dBuffer1);
*
*    // Determine size of second user allocated buffer
*    hipsparseSpGEMM_compute(handle, opA, opB,
*                                &alpha, matA, matB, &beta, matC,
*                                computeType, HIPSPARSE_SPGEMM_DEFAULT,
*                                spgemmDesc, &bufferSize2, NULL);
*    hipMalloc((void**) &dBuffer2, bufferSize2);
*
*    // Compute C = alpha * A * B and store result in temporary buffers
*    hipsparseSpGEMM_compute(handle, opA, opB,
*                                        &alpha, matA, matB, &beta, matC,
*                                        computeType, HIPSPARSE_SPGEMM_DEFAULT,
*                                        spgemmDesc, &bufferSize2, dBuffer2);
*
*    // Get matrix C non-zero entries C_nnz1
*    int64_t C_num_rows1, C_num_cols1, C_nnz1;
*    hipsparseSpMatGetSize(matC, &C_num_rows1, &C_num_cols1, &C_nnz1);
*
*    // Allocate the CSR structures for the matrix C
*    hipMalloc((void**) &dcsrColIndC, C_nnz1 * sizeof(int));
*    hipMalloc((void**) &dcsrValC,  C_nnz1 * sizeof(float));
*
*    // Update matC with the new pointers
*    hipsparseCsrSetPointers(matC, dcsrRowPtrC, dcsrColIndC, dcsrValC);
*
*    // Copy the final products to the matrix C
*    hipsparseSpGEMM_copy(handle, opA, opB,
*                            &alpha, matA, matB, &beta, matC,
*                            computeType, HIPSPARSE_SPGEMM_DEFAULT, spgemmDesc);
*
*    // Destroy matrix descriptors and handles
*    hipsparseSpGEMM_destroyDescr(spgemmDesc);
*    hipsparseDestroySpMat(matA);
*    hipsparseDestroySpMat(matB);
*    hipsparseDestroySpMat(matC);
*    hipsparseDestroy(handle);
* 
*    // Free device memory
*    hipFree(dBuffer1);
*    hipFree(dBuffer2);
*  \endcode
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpGEMM_copy(hipsparseHandle_t          handle,
                                       hipsparseOperation_t       opA,
                                       hipsparseOperation_t       opB,
                                       const void*                alpha,
                                       hipsparseConstSpMatDescr_t matA,
                                       hipsparseConstSpMatDescr_t matB,
                                       const void*                beta,
                                       hipsparseSpMatDescr_t      matC,
                                       hipDataType                computeType,
                                       hipsparseSpGEMMAlg_t       alg,
                                       hipsparseSpGEMMDescr_t     spgemmDescr);
#elif(CUDART_VERSION >= 11000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpGEMM_copy(hipsparseHandle_t      handle,
                                       hipsparseOperation_t   opA,
                                       hipsparseOperation_t   opB,
                                       const void*            alpha,
                                       hipsparseSpMatDescr_t  matA,
                                       hipsparseSpMatDescr_t  matB,
                                       const void*            beta,
                                       hipsparseSpMatDescr_t  matC,
                                       hipDataType            computeType,
                                       hipsparseSpGEMMAlg_t   alg,
                                       hipsparseSpGEMMDescr_t spgemmDescr);
#endif

/*! \ingroup generic_module
*  \brief Work estimation step of the sparse matrix sparse matrix product:
*  \f[
*    C' := \alpha \cdot op(A) \cdot op(B) + \beta \cdot C,
*  \f]
*  where \f$C'\f$, \f$A\f$, \f$B\f$, \f$C\f$ are sparse matrices and \f$C'\f$ and \f$C\f$ have the same sparsity pattern.
*
*  \details
*  \p hipsparseSpGEMMreuse_workEstimation is called twice. We call it to compute the size of the first required user allocated
*  buffer. After this buffer size is determined, the user allocates it and calls \p hipsparseSpGEMMreuse_workEstimation
*  a second time with the newly allocated buffer passed in. This second call inspects the matrices \f$A\f$ and \f$B\f$ to 
*  determine the number of intermediate products that will result from multipltying \f$A\f$ and \f$B\f$ together.
*  
*  @param[in]
*  handle           handle to the hipsparse library context queue.
*  @param[in]
*  opA              sparse matrix \f$A\f$ operation type.
*  @param[in]
*  opB              sparse matrix \f$B\f$ operation type.
*  @param[in]
*  matA             sparse matrix \f$A\f$ descriptor.
*  @param[in]
*  matB             sparse matrix \f$B\f$ descriptor.
*  @param[out]
*  matC             sparse matrix \f$C\f$ descriptor.
*  @param[in]
*  alg              SpGEMM algorithm for the SpGEMM computation.
*  @param[in]
*  spgemmDescr      SpGEMM descriptor.
*  @param[out]
*  bufferSize1      number of bytes of the temporary storage buffer. 
*  @param[in]
*  externalBuffer1  temporary storage buffer allocated by the user.
*
*  \retval HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p matA, \p matB, \p matC or \p bufferSize1 pointer is invalid.
*  \retval HIPSPARSE_STATUS_ALLOC_FAILED additional buffer for long rows could not be
*          allocated.
*  \retval HIPSPARSE_STATUS_NOT_SUPPORTED
*          \p opA != \ref HIPSPARSE_OPERATION_NON_TRANSPOSE or
*          \p opB != \ref HIPSPARSE_OPERATION_NON_TRANSPOSE.
*
*  \par Example (See full example below)
*  \code{.c}
*    void*  dBuffer1  = NULL; 
*    size_t bufferSize1 = 0;
*
*    hipsparseSpGEMMDescr_t spgemmDesc;
*    hipsparseSpGEMM_createDescr(&spgemmDesc);
*
*    size_t bufferSize1 = 0;
*    hipsparseSpGEMMreuse_workEstimation(handle, opA, opB, matA, matB, matC,
*                                        HIPSPARSE_SPGEMM_DEFAULT, spgemmDesc, 
*                                        &bufferSize1, NULL);
*    hipMalloc((void**) &dBuffer1, bufferSize1);
*
*    // Determine number of intermediate product when computing A * B
*    hipsparseSpGEMMreuse_workEstimation(handle, opA, opB, matA, matB, matC,
*                                        HIPSPARSE_SPGEMM_DEFAULT, spgemmDesc, 
*                                        &bufferSize1, dBuffer1);
*  \endcode
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpGEMMreuse_workEstimation(hipsparseHandle_t          handle,
                                                      hipsparseOperation_t       opA,
                                                      hipsparseOperation_t       opB,
                                                      hipsparseConstSpMatDescr_t matA,
                                                      hipsparseConstSpMatDescr_t matB,
                                                      hipsparseSpMatDescr_t      matC,
                                                      hipsparseSpGEMMAlg_t       alg,
                                                      hipsparseSpGEMMDescr_t     spgemmDescr,
                                                      size_t*                    bufferSize1,
                                                      void*                      externalBuffer1);
#elif(CUDART_VERSION >= 11031)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpGEMMreuse_workEstimation(hipsparseHandle_t      handle,
                                                      hipsparseOperation_t   opA,
                                                      hipsparseOperation_t   opB,
                                                      hipsparseSpMatDescr_t  matA,
                                                      hipsparseSpMatDescr_t  matB,
                                                      hipsparseSpMatDescr_t  matC,
                                                      hipsparseSpGEMMAlg_t   alg,
                                                      hipsparseSpGEMMDescr_t spgemmDescr,
                                                      size_t*                bufferSize1,
                                                      void*                  externalBuffer1);
#endif

/*! \ingroup generic_module
*  \brief Nnz calculation step of the sparse matrix sparse matrix product:
*  \f[
*    C' := \alpha \cdot op(A) \cdot op(B) + \beta \cdot C,
*  \f]
*  where \f$C'\f$, \f$A\f$, \f$B\f$, \f$C\f$ are sparse matrices and \f$C'\f$ and \f$C\f$ have the same sparsity pattern.
*
*  @param[in]
*  handle           handle to the hipsparse library context queue.
*  @param[in]
*  opA              sparse matrix \f$A\f$ operation type.
*  @param[in]
*  opB              sparse matrix \f$B\f$ operation type.
*  @param[in]
*  matA             sparse matrix \f$A\f$ descriptor.
*  @param[in]
*  matB             sparse matrix \f$B\f$ descriptor.
*  @param[out]
*  matC             sparse matrix \f$C\f$ descriptor.
*  @param[in]
*  alg              SpGEMM algorithm for the SpGEMM computation.
*  @param[in]
*  spgemmDescr      SpGEMM descriptor.
*  @param[out]
*  bufferSize2      number of bytes of the temporary storage \p externalBuffer2. 
*  @param[in]
*  externalBuffer2  temporary storage buffer allocated by the user.
*  @param[out]
*  bufferSize3      number of bytes of the temporary storage \p externalBuffer3. 
*  @param[in]
*  externalBuffer3  temporary storage buffer allocated by the user.
*  @param[out]
*  bufferSize4      number of bytes of the temporary storage \p externalBuffer4. 
*  @param[in]
*  externalBuffer4  temporary storage buffer allocated by the user.
*
*  \retval HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p matA, \p matB, \p matC, \p bufferSize2, \p bufferSize3 
*                                         or \p bufferSize4 pointer is invalid.
*  \retval HIPSPARSE_STATUS_ALLOC_FAILED additional buffer for long rows could not be
*          allocated.
*  \retval HIPSPARSE_STATUS_NOT_SUPPORTED
*          \p opA != \ref HIPSPARSE_OPERATION_NON_TRANSPOSE or
*          \p opB != \ref HIPSPARSE_OPERATION_NON_TRANSPOSE.
*
*  \par Example (See full example below)
*  \code{.c}
*    // Determine size of second, third, and fourth user allocated buffer
*    hipsparseSpGEMMreuse_nnz(handle, opA, opB, matA, matB,
*                                matC, HIPSPARSE_SPGEMM_DEFAULT, spgemmDesc,
*                                &bufferSize2, NULL, &bufferSize3, NULL,
*                                &bufferSize4, NULL);
*
*    hipMalloc((void**) &dBuffer2, bufferSize2);
*    hipMalloc((void**) &dBuffer3, bufferSize3);
*    hipMalloc((void**) &dBuffer4, bufferSize4);
*
*    // COmpute sparsity pattern of C matrix and store in temporary buffers
*    hipsparseSpGEMMreuse_nnz(handle, opA, opB, matA, matB,
*                                matC, HIPSPARSE_SPGEMM_DEFAULT, spgemmDesc,
*                                &bufferSize2, dBuffer2, &bufferSize3, dBuffer3,
*                                &bufferSize4, dBuffer4);
*
*    // We can now free buffer 1 and 2
*    hipFree(dBuffer1);
*    hipFree(dBuffer2);
*  \endcode
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpGEMMreuse_nnz(hipsparseHandle_t          handle,
                                           hipsparseOperation_t       opA,
                                           hipsparseOperation_t       opB,
                                           hipsparseConstSpMatDescr_t matA,
                                           hipsparseConstSpMatDescr_t matB,
                                           hipsparseSpMatDescr_t      matC,
                                           hipsparseSpGEMMAlg_t       alg,
                                           hipsparseSpGEMMDescr_t     spgemmDescr,
                                           size_t*                    bufferSize2,
                                           void*                      externalBuffer2,
                                           size_t*                    bufferSize3,
                                           void*                      externalBuffer3,
                                           size_t*                    bufferSize4,
                                           void*                      externalBuffer4);

#elif(CUDART_VERSION >= 11031)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpGEMMreuse_nnz(hipsparseHandle_t      handle,
                                           hipsparseOperation_t   opA,
                                           hipsparseOperation_t   opB,
                                           hipsparseSpMatDescr_t  matA,
                                           hipsparseSpMatDescr_t  matB,
                                           hipsparseSpMatDescr_t  matC,
                                           hipsparseSpGEMMAlg_t   alg,
                                           hipsparseSpGEMMDescr_t spgemmDescr,
                                           size_t*                bufferSize2,
                                           void*                  externalBuffer2,
                                           size_t*                bufferSize3,
                                           void*                  externalBuffer3,
                                           size_t*                bufferSize4,
                                           void*                  externalBuffer4);

#endif

/*! \ingroup generic_module
*  \brief Copy step of the sparse matrix sparse matrix product: 
*  \f[
*    C' := \alpha \cdot op(A) \cdot op(B) + \beta \cdot C,
*  \f]
*  where \f$C'\f$, \f$A\f$, \f$B\f$, \f$C\f$ are sparse matrices and \f$C'\f$ and \f$C\f$ have the same sparsity pattern.
*
*  @param[in]
*  handle           handle to the hipsparse library context queue.
*  @param[in]
*  opA              sparse matrix \f$A\f$ operation type.
*  @param[in]
*  opB              sparse matrix \f$B\f$ operation type.
*  @param[in]
*  matA             sparse matrix \f$A\f$ descriptor.
*  @param[in]
*  matB             sparse matrix \f$B\f$ descriptor.
*  @param[out]
*  matC             sparse matrix \f$C\f$ descriptor.
*  @param[in]
*  alg              SpGEMM algorithm for the SpGEMM computation.
*  @param[in]
*  spgemmDescr      SpGEMM descriptor.
*  @param[out]
*  bufferSize5      number of bytes of the temporary storage \p externalBuffer5. 
*  @param[in]
*  externalBuffer5  temporary storage buffer allocated by the user.
*
*  \retval HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p matA, \p matB, \p matC, or \p bufferSize5 pointer is invalid.
*  \retval HIPSPARSE_STATUS_ALLOC_FAILED additional buffer for long rows could not be
*          allocated.
*  \retval HIPSPARSE_STATUS_NOT_SUPPORTED
*          \p opA != \ref HIPSPARSE_OPERATION_NON_TRANSPOSE or
*          \p opB != \ref HIPSPARSE_OPERATION_NON_TRANSPOSE.
*
*  \par Example (See full example below)
*  \code{.c}
*    // Get matrix C non-zero entries nnzC
*    int64_t rowsC, colsC, nnzC;
*    hipsparseSpMatGetSize(matC, &rowsC, &colsC, &nnzC);
*
*    // Allocate matrix C
*    hipMalloc((void**) &dcsrColIndC, sizeof(int) * nnzC);
*    hipMalloc((void**) &dcsrValC,  sizeof(float) * nnzC);
*    
*    // Update matC with the new pointers. The C values array can be filled with data here
*    // which is used if beta != 0.
*    hipsparseCsrSetPointers(matC, dcsrRowPtrC, dcsrColIndC, dcsrValC);
*
*    // Determine size of fifth user allocated buffer
*    hipsparseSpGEMMreuse_copy(handle, opA, opB, matA, matB, matC,
*                                 HIPSPARSE_SPGEMM_DEFAULT, spgemmDesc,
*                                 &bufferSize5, NULL);
*
*    hipMalloc((void**) &dBuffer5, bufferSize5);
*    
*    // Copy data from temporary buffers to the newly allocated C matrix
*    hipsparseSpGEMMreuse_copy(handle, opA, opB, matA, matB, matC,
*                                 HIPSPARSE_SPGEMM_DEFAULT, spgemmDesc,
*                                 &bufferSize5, dBuffer5);
*    
*    // We can now free buffer 3
*    hipFree(dBuffer3);
*  \endcode
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpGEMMreuse_copy(hipsparseHandle_t          handle,
                                            hipsparseOperation_t       opA,
                                            hipsparseOperation_t       opB,
                                            hipsparseConstSpMatDescr_t matA,
                                            hipsparseConstSpMatDescr_t matB,
                                            hipsparseSpMatDescr_t      matC,
                                            hipsparseSpGEMMAlg_t       alg,
                                            hipsparseSpGEMMDescr_t     spgemmDescr,
                                            size_t*                    bufferSize5,
                                            void*                      externalBuffer5);
#elif(CUDART_VERSION >= 11031)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpGEMMreuse_copy(hipsparseHandle_t      handle,
                                            hipsparseOperation_t   opA,
                                            hipsparseOperation_t   opB,
                                            hipsparseSpMatDescr_t  matA,
                                            hipsparseSpMatDescr_t  matB,
                                            hipsparseSpMatDescr_t  matC,
                                            hipsparseSpGEMMAlg_t   alg,
                                            hipsparseSpGEMMDescr_t spgemmDescr,
                                            size_t*                bufferSize5,
                                            void*                  externalBuffer5);
#endif

/*! \ingroup generic_module
*  \brief Copy step of the sparse matrix sparse matrix product: 
*  \f[
*    C' := \alpha \cdot op(A) \cdot op(B) + \beta \cdot C,
*  \f]
*  where \f$C'\f$, \f$A\f$, \f$B\f$, \f$C\f$ are sparse matrices and \f$C'\f$ and \f$C\f$ have the same sparsity pattern.
*
*  @param[in]
*  handle           handle to the hipsparse library context queue.
*  @param[in]
*  opA              sparse matrix \f$A\f$ operation type.
*  @param[in]
*  opB              sparse matrix \f$B\f$ operation type.
*  @param[in]
*  alpha            scalar \f$\alpha\f$.
*  @param[in]
*  matA             sparse matrix \f$A\f$ descriptor.
*  @param[in]
*  matB             sparse matrix \f$B\f$ descriptor.
*  @param[in]
*  beta             scalar \f$\beta\f$.
*  @param[out]
*  matC             sparse matrix \f$C\f$ descriptor.
*  @param[in]
*  computeType      floating point precision for the SpGEMM computation.
*  @param[in]
*  alg              SpGEMM algorithm for the SpGEMM computation.
*  @param[in]
*  spgemmDescr      SpGEMM descriptor.
*
*  \retval HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p alpha, \p beta, \p matA, \p matB, or \p matC 
*                                         pointer is invalid.
*  \retval HIPSPARSE_STATUS_ALLOC_FAILED additional buffer for long rows could not be
*          allocated.
*  \retval HIPSPARSE_STATUS_NOT_SUPPORTED
*          \p opA != \ref HIPSPARSE_OPERATION_NON_TRANSPOSE or
*          \p opB != \ref HIPSPARSE_OPERATION_NON_TRANSPOSE.
*
*  \par Full example
*  \code{.c}
*    int m = 2;
*    int k = 2;
*    int n = 3;
*    int nnzA = 4;
*    int nnzB = 4;
*    
*    float alpha{1.0f};
*    float beta{0.0f};
*
*    hipsparseOperation_t opA        = HIPSPARSE_OPERATION_NON_TRANSPOSE;
*    hipsparseOperation_t opB        = HIPSPARSE_OPERATION_NON_TRANSPOSE;
*    hipDataType computeType         = HIP_R_32F;
*
*    // A, B, and C are m×k, k×n, and m×n
*
*    // A
*    std::vector<int> hcsrRowPtrA = {0, 2, 4};
*    std::vector<int> hcsrColIndA = {0, 1, 0, 1};
*    std::vector<float> hcsrValA = {1.0f, 2.0f, 3.0f, 4.0f};
*
*    // B
*    std::vector<int> hcsrRowPtrB = {0, 2, 4};
*    std::vector<int> hcsrColIndB = {1, 2, 0, 2};
*    std::vector<float> hcsrValB = {5.0f , 6.0f, 7.0f, 8.0f};
*
*    // Device memory management: Allocate and copy A, B
*    int* dcsrRowPtrA;
*    int* dcsrColIndA;
*    float* dcsrValA;
*    int* dcsrRowPtrB;
*    int* dcsrColIndB;
*    float* dcsrValB;
*    int* dcsrRowPtrC;
*    int* dcsrColIndC;
*    float* dcsrValC;
*    hipMalloc((void**)&dcsrRowPtrA, (m + 1) * sizeof(int));
*    hipMalloc((void**)&dcsrColIndA, nnzA * sizeof(int));
*    hipMalloc((void**)&dcsrValA, nnzA * sizeof(float));
*    hipMalloc((void**)&dcsrRowPtrB, (k + 1) * sizeof(int));
*    hipMalloc((void**)&dcsrColIndB, nnzB * sizeof(int));
*    hipMalloc((void**)&dcsrValB, nnzB * sizeof(float));
*    hipMalloc((void**)&dcsrRowPtrC, (m + 1) * sizeof(int));
*
*    hipMemcpy(dcsrRowPtrA, hcsrRowPtrA.data(), (m + 1) * sizeof(int), hipMemcpyHostToDevice);
*    hipMemcpy(dcsrColIndA, hcsrColIndA.data(), nnzA * sizeof(int), hipMemcpyHostToDevice);
*    hipMemcpy(dcsrValA, hcsrValA.data(), nnzA * sizeof(float), hipMemcpyHostToDevice);
*    hipMemcpy(dcsrRowPtrB, hcsrRowPtrB.data(), (k + 1) * sizeof(int), hipMemcpyHostToDevice);
*    hipMemcpy(dcsrColIndB, hcsrColIndB.data(), nnzB * sizeof(int), hipMemcpyHostToDevice);
*    hipMemcpy(dcsrValB, hcsrValB.data(), nnzB * sizeof(float), hipMemcpyHostToDevice);
*
*    hipsparseHandle_t     handle = NULL;
*    hipsparseSpMatDescr_t matA, matB, matC;
*    void*  dBuffer1  = NULL; 
*    void*  dBuffer2  = NULL;
*    void*  dBuffer3  = NULL;
*    void*  dBuffer4  = NULL;
*    void*  dBuffer5  = NULL;
*    size_t bufferSize1 = 0;  
*    size_t bufferSize2 = 0;
*    size_t bufferSize3 = 0;
*    size_t bufferSize4 = 0;
*    size_t bufferSize5 = 0;
*
*    hipsparseCreate(&handle);
*
*    // Create sparse matrix A in CSR format
*    hipsparseCreateCsr(&matA, m, k, nnzA,
*                        dcsrRowPtrA, dcsrColIndA, dcsrValA,
*                        HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_32I,
*                        HIPSPARSE_INDEX_BASE_ZERO, HIP_R_32F);
*    hipsparseCreateCsr(&matB, k, n, nnzB,
*                        dcsrRowPtrB, dcsrColIndB, dcsrValB,
*                        HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_32I,
*                        HIPSPARSE_INDEX_BASE_ZERO, HIP_R_32F);
*    hipsparseCreateCsr(&matC, m, n, 0,
*                        dcsrRowPtrC, NULL, NULL,
*                        HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_32I,
*                        HIPSPARSE_INDEX_BASE_ZERO, HIP_R_32F);
*
*    hipsparseSpGEMMDescr_t spgemmDesc;
*    hipsparseSpGEMM_createDescr(&spgemmDesc);
*
*    // Determine size of first user allocated buffer
*    hipsparseSpGEMMreuse_workEstimation(handle, opA, opB, matA, matB, matC,
*                                        HIPSPARSE_SPGEMM_DEFAULT, spgemmDesc, 
*                                        &bufferSize1, NULL);
*
*    hipMalloc((void**) &dBuffer1, bufferSize1);
*
*    // Inspect the matrices A and B to determine the number of intermediate product in 
*    // C = alpha * A * B
*    hipsparseSpGEMMreuse_workEstimation(handle, opA, opB, matA, matB, matC,
*                                        HIPSPARSE_SPGEMM_DEFAULT, spgemmDesc, 
*                                        &bufferSize1, dBuffer1);
*
*    // Determine size of second, third, and fourth user allocated buffer
*    hipsparseSpGEMMreuse_nnz(handle, opA, opB, matA, matB,
*                                matC, HIPSPARSE_SPGEMM_DEFAULT, spgemmDesc,
*                                &bufferSize2, NULL, &bufferSize3, NULL,
*                                &bufferSize4, NULL);
*
*    hipMalloc((void**) &dBuffer2, bufferSize2);
*    hipMalloc((void**) &dBuffer3, bufferSize3);
*    hipMalloc((void**) &dBuffer4, bufferSize4);
*
*    // COmpute sparsity pattern of C matrix and store in temporary buffers
*    hipsparseSpGEMMreuse_nnz(handle, opA, opB, matA, matB,
*                                matC, HIPSPARSE_SPGEMM_DEFAULT, spgemmDesc,
*                                &bufferSize2, dBuffer2, &bufferSize3, dBuffer3,
*                                &bufferSize4, dBuffer4);
*    
*    // We can now free buffer 1 and 2
*    hipFree(dBuffer1);
*    hipFree(dBuffer2);
*
*    // Get matrix C non-zero entries nnzC
*    int64_t rowsC, colsC, nnzC;
*    hipsparseSpMatGetSize(matC, &rowsC, &colsC, &nnzC);
*
*    // Allocate matrix C
*    hipMalloc((void**) &dcsrColIndC, sizeof(int) * nnzC);
*    hipMalloc((void**) &dcsrValC,  sizeof(float) * nnzC);
*    
*    // Update matC with the new pointers. The C values array can be filled with data here
*    // which is used if beta != 0.
*    hipsparseCsrSetPointers(matC, dcsrRowPtrC, dcsrColIndC, dcsrValC);
*
*    // Determine size of fifth user allocated buffer
*    hipsparseSpGEMMreuse_copy(handle, opA, opB, matA, matB, matC,
*                                 HIPSPARSE_SPGEMM_DEFAULT, spgemmDesc,
*                                 &bufferSize5, NULL);
*
*    hipMalloc((void**) &dBuffer5, bufferSize5);
*    
*    // Copy data from temporary buffers to the newly allocated C matrix
*    hipsparseSpGEMMreuse_copy(handle, opA, opB, matA, matB, matC,
*                                 HIPSPARSE_SPGEMM_DEFAULT, spgemmDesc,
*                                 &bufferSize5, dBuffer5);
*    
*    // We can now free buffer 3
*    hipFree(dBuffer3);
*
*    // Compute C' = alpha * A * B + beta * C
*    hipsparseSpGEMMreuse_compute(handle, opA, opB, &alpha, matA, matB, &beta,
*                                    matC, computeType, HIPSPARSE_SPGEMM_DEFAULT,
*                                    spgemmDesc);
*
*    // Copy results back to host if required using hipsparseCsrGet...
*
*    // Update dcsrValA, dcsrValB with new values
*    for(size_t i = 0; i < hcsrValA.size(); i++){ hcsrValA[i] = 1.0f; }
*    for(size_t i = 0; i < hcsrValB.size(); i++){ hcsrValB[i] = 2.0f; }
*
*    hipMemcpy(dcsrValA, hcsrValA.data(), sizeof(float) * nnzA, hipMemcpyHostToDevice);
*    hipMemcpy(dcsrValB, hcsrValB.data(), sizeof(float) * nnzB, hipMemcpyHostToDevice);
*    
*    // Compute C' = alpha * A * B + beta * C again with the new A and B values
*    hipsparseSpGEMMreuse_compute(handle, opA, opB, &alpha, matA, matB, &beta,
*                                    matC, computeType, HIPSPARSE_SPGEMM_DEFAULT,
*                                    spgemmDesc);
*
*    // Copy results back to host if required using hipsparseCsrGet...
*
*    // Destroy matrix descriptors and handles
*    hipsparseSpGEMM_destroyDescr(spgemmDesc);
*    hipsparseDestroySpMat(matA);
*    hipsparseDestroySpMat(matB);
*    hipsparseDestroySpMat(matC);
*    hipsparseDestroy(handle);
*
*    // Free device memory
*    hipFree(dBuffer4);
*    hipFree(dBuffer5);
*    hipFree(dcsrRowPtrA);
*    hipFree(dcsrColIndA);
*    hipFree(dcsrValA);
*    hipFree(dcsrRowPtrB);
*    hipFree(dcsrColIndB);
*    hipFree(dcsrValB);
*    hipFree(dcsrRowPtrC);
*    hipFree(dcsrColIndC);
*    hipFree(dcsrValC);
*  \endcode
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpGEMMreuse_compute(hipsparseHandle_t          handle,
                                               hipsparseOperation_t       opA,
                                               hipsparseOperation_t       opB,
                                               const void*                alpha,
                                               hipsparseConstSpMatDescr_t matA,
                                               hipsparseConstSpMatDescr_t matB,
                                               const void*                beta,
                                               hipsparseSpMatDescr_t      matC,
                                               hipDataType                computeType,
                                               hipsparseSpGEMMAlg_t       alg,
                                               hipsparseSpGEMMDescr_t     spgemmDescr);
#elif(CUDART_VERSION >= 11031)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpGEMMreuse_compute(hipsparseHandle_t      handle,
                                               hipsparseOperation_t   opA,
                                               hipsparseOperation_t   opB,
                                               const void*            alpha,
                                               hipsparseSpMatDescr_t  matA,
                                               hipsparseSpMatDescr_t  matB,
                                               const void*            beta,
                                               hipsparseSpMatDescr_t  matC,
                                               hipDataType            computeType,
                                               hipsparseSpGEMMAlg_t   alg,
                                               hipsparseSpGEMMDescr_t spgemmDescr);
#endif

/*! \ingroup generic_module
*  \brief Calculate the buffer size required for the sampled dense dense matrix multiplication:
*  \f[
*    C := \alpha (op(A) \cdot op(B)) \circ spy(C) + \beta \cdot C,
*  \f]
*  where \f$C\f$ is a sparse matrix and \f$A\f$ and \f$B\f$ are dense matrices.
*
*  \details
*  \p hipsparseSDDMM_bufferSize computes the required user allocated buffer size needed when computing the 
*  sampled dense dense matrix multiplication
*
*  @param[in]
*  handle              handle to the hipsparse library context queue.
*  @param[in]
*  opA                 dense matrix \f$A\f$ operation type.
*  @param[in]
*  opB                 dense matrix \f$B\f$ operation type.
*  @param[in]
*  alpha               scalar \f$\alpha\f$.
*  @param[in]
*  A                   dense matrix \f$A\f$ descriptor.
*  @param[in]
*  B                   dense matrix \f$B\f$ descriptor.
*  @param[in]
*  beta                scalar \f$\beta\f$.
*  @param[inout]
*  C                   sparse matrix \f$C\f$ descriptor.
*  @param[in]
*  computeType         floating point precision for the SDDMM computation.
*  @param[in]
*  alg                 specification of the algorithm to use.
*  @param[out]
*  pBufferSizeInBytes  number of bytes of the temporary storage buffer.
*
*  \retval HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p alpha, \p beta, \p A, \p B, \p D, \p C or 
*          \p pBufferSizeInBytes pointer is invalid or the value of \p opA or \p opB is incorrect
*  \retval HIPSPARSE_STATUS_NOT_SUPPORTED
*          \p opA == \ref HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE or
*          \p opB == \ref HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE.
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSDDMM_bufferSize(hipsparseHandle_t          handle,
                                            hipsparseOperation_t       opA,
                                            hipsparseOperation_t       opB,
                                            const void*                alpha,
                                            hipsparseConstDnMatDescr_t A,
                                            hipsparseConstDnMatDescr_t B,
                                            const void*                beta,
                                            hipsparseSpMatDescr_t      C,
                                            hipDataType                computeType,
                                            hipsparseSDDMMAlg_t        alg,
                                            size_t*                    pBufferSizeInBytes);
#elif(CUDART_VERSION >= 11022)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSDDMM_bufferSize(hipsparseHandle_t           handle,
                                            hipsparseOperation_t        opA,
                                            hipsparseOperation_t        opB,
                                            const void*                 alpha,
                                            const hipsparseDnMatDescr_t A,
                                            const hipsparseDnMatDescr_t B,
                                            const void*                 beta,
                                            hipsparseSpMatDescr_t       C,
                                            hipDataType                 computeType,
                                            hipsparseSDDMMAlg_t         alg,
                                            size_t*                     pBufferSizeInBytes);
#endif

/*! \ingroup generic_module
*  \brief Preprocess step of the sampled dense dense matrix multiplication:
*  \f[
*    C := \alpha (op(A) \cdot op(B)) \circ spy(C) + \beta \cdot C,
*  \f]
*  where \f$C\f$ is a sparse matrix and \f$A\f$ and \f$B\f$ are dense matrices.
*
*  \details
*  \p hipsparseSDDMM_preprocess performs the required preprocessing used when computing the 
*  sampled dense dense matrix multiplication
*
*  @param[in]
*  handle       handle to the hipsparse library context queue.
*  @param[in]
*  opA          dense matrix \f$A\f$ operation type.
*  @param[in]
*  opB          dense matrix \f$B\f$ operation type.
*  @param[in]
*  alpha        scalar \f$\alpha\f$.
*  @param[in]
*  A            dense matrix \f$A\f$ descriptor.
*  @param[in]
*  B            dense matrix \f$B\f$ descriptor.
*  @param[in]
*  beta         scalar \f$\beta\f$.
*  @param[inout]
*  C            sparse matrix \f$C\f$ descriptor.
*  @param[in]
*  computeType  floating point precision for the SDDMM computation.
*  @param[in]
*  alg          specification of the algorithm to use.
*  @param[in]
*  tempBuffer   temporary storage buffer allocated by the user. The size must be greater or equal to 
*               the size obtained with \ref hipsparseSDDMM_bufferSize.
*
*  \retval HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p alpha, \p beta, \p A, \p B, \p C or 
*          \p tempBuffer pointer is invalid or the value of \p opA or \p opB is incorrect.
*  \retval HIPSPARSE_STATUS_NOT_SUPPORTED
*          \p opA == \ref HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE or
*          \p opB == \ref HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE.
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSDDMM_preprocess(hipsparseHandle_t          handle,
                                            hipsparseOperation_t       opA,
                                            hipsparseOperation_t       opB,
                                            const void*                alpha,
                                            hipsparseConstDnMatDescr_t A,
                                            hipsparseConstDnMatDescr_t B,
                                            const void*                beta,
                                            hipsparseSpMatDescr_t      C,
                                            hipDataType                computeType,
                                            hipsparseSDDMMAlg_t        alg,
                                            void*                      tempBuffer);
#elif(CUDART_VERSION >= 11022)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSDDMM_preprocess(hipsparseHandle_t           handle,
                                            hipsparseOperation_t        opA,
                                            hipsparseOperation_t        opB,
                                            const void*                 alpha,
                                            const hipsparseDnMatDescr_t A,
                                            const hipsparseDnMatDescr_t B,
                                            const void*                 beta,
                                            hipsparseSpMatDescr_t       C,
                                            hipDataType                 computeType,
                                            hipsparseSDDMMAlg_t         alg,
                                            void*                       tempBuffer);
#endif

/*! \ingroup generic_module
*  \brief Description: Sampled Dense-Dense Matrix Multiplication.
*
*  \details
*  \ref hipsparseSDDMM multiplies the scalar \f$\alpha\f$ with the dense
*  \f$m \times k\f$ matrix \f$A\f$, the dense \f$k \times n\f$ matrix \f$B\f$, filtered by the sparsity pattern of the \f$m \times n\f$ sparse matrix \f$C\f$ and
*  adds the result to \f$C\f$ scaled by
*  \f$\beta\f$. The final result is stored in the sparse \f$m \times n\f$ matrix \f$C\f$,
*  such that
*  \f[
*    C := \alpha ( op(A) \cdot op(B) ) \circ spy(C) + \beta C,
*  \f]
*  with
*  \f[
*    op(A) = \left\{
*    \begin{array}{ll}
*        A,   & \text{if opA == HIPSPARSE_OPERATION_NON_TRANSPOSE} \\
*        A^T,   & \text{if opA == HIPSPARSE_OPERATION_TRANSPOSE} \\
*    \end{array}
*    \right.
*  \f],
*  \f[
*    op(B) = \left\{
*    \begin{array}{ll}
*        B,   & \text{if opB == HIPSPARSE_OPERATION_NON_TRANSPOSE} \\
*        B^T,   & \text{if opB == HIPSPARSE_OPERATION_TRANSPOSE} \\
*    \end{array}
*    \right.
*  \f]
*   and
*  \f[
*    spy(C)_{ij} = \left\{
*    \begin{array}{ll}
*        1 \text{  if i == j},   & 0 \text{  if i != j} \\
*    \end{array}
*    \right.
*  \f]
*
*  @param[in]
*  handle       handle to the hipsparse library context queue.
*  @param[in]
*  opA          dense matrix \f$A\f$ operation type.
*  @param[in]
*  opB          dense matrix \f$B\f$ operation type.
*  @param[in]
*  alpha        scalar \f$\alpha\f$.
*  @param[in]
*  A            dense matrix \f$A\f$ descriptor.
*  @param[in]
*  B            dense matrix \f$B\f$ descriptor.
*  @param[in]
*  beta         scalar \f$\beta\f$.
*  @param[inout]
*  C            sparse matrix \f$C\f$ descriptor.
*  @param[in]
*  computeType  floating point precision for the SDDMM computation.
*  @param[in]
*  alg          specification of the algorithm to use.
*  @param[in]
*  tempBuffer   temporary storage buffer allocated by the user. The size must be greater or equal to 
*               the size obtained with \ref hipsparseSDDMM_bufferSize.
*
*  \retval HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p alpha, \p beta, \p A, \p B, \p C or 
*          \p tempBuffer pointer is invalid or the value of \p opA or \p opB is incorrect.
*  \retval HIPSPARSE_STATUS_NOT_SUPPORTED
*          \p opA == \ref HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE or
*          \p opB == \ref HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE.
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSDDMM(hipsparseHandle_t          handle,
                                 hipsparseOperation_t       opA,
                                 hipsparseOperation_t       opB,
                                 const void*                alpha,
                                 hipsparseConstDnMatDescr_t A,
                                 hipsparseConstDnMatDescr_t B,
                                 const void*                beta,
                                 hipsparseSpMatDescr_t      C,
                                 hipDataType                computeType,
                                 hipsparseSDDMMAlg_t        alg,
                                 void*                      tempBuffer);
#elif(CUDART_VERSION >= 11022)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSDDMM(hipsparseHandle_t           handle,
                                 hipsparseOperation_t        opA,
                                 hipsparseOperation_t        opB,
                                 const void*                 alpha,
                                 const hipsparseDnMatDescr_t A,
                                 const hipsparseDnMatDescr_t B,
                                 const void*                 beta,
                                 hipsparseSpMatDescr_t       C,
                                 hipDataType                 computeType,
                                 hipsparseSDDMMAlg_t         alg,
                                 void*                       tempBuffer);
#endif

/*! \ingroup generic_module
*  \brief Create sparse matrix triangular solve descriptor
*  \details
*  \p hipsparseSpSV_createDescr creates a sparse matrix triangular solve descriptor. It should be
*  destroyed at the end using hipsparseSpSV_destroyDescr().
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11030)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpSV_createDescr(hipsparseSpSVDescr_t* descr);
#endif

/*! \ingroup generic_module
*  \brief Destroy sparse matrix triangular solve descriptor
*  \details
*  \p hipsparseSpSV_destroyDescr destroys a sparse matrix triangular solve descriptor and releases all
*  resources used by the descriptor.
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11030)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpSV_destroyDescr(hipsparseSpSVDescr_t descr);
#endif

/*! \ingroup generic_module
*  \brief Buffer size step of solution of triangular linear system: 
*  \f[
*    op(A) \cdot y := \alpha \cdot x,
*  \f]
*  where \f$A\f$ is a sparse matrix in CSR storage format, \f$x\f$ and \f$y\f$ are dense vectors.
*
*  \details
*  \p hipsparseSpSV_bufferSize computes the required user allocated buffer size needed when computing the 
*  solution of triangular linear system \f$op(A) \cdot y = \alpha \cdot x\f$, where \f$A\f$ is a sparse matrix in CSR storage 
*  format, \f$x\f$ and \f$y\f$ are dense vectors.
*
*  @param[in]
*  handle              handle to the hipsparse library context queue.
*  @param[in]
*  opA                 matrix operation type.
*  @param[in]
*  alpha               scalar \f$\alpha\f$.
*  @param[in]
*  matA                matrix descriptor.
*  @param[in]
*  x                   vector descriptor.
*  @param[inout]
*  y                   vector descriptor.
*  @param[in]
*  computeType         floating point precision for the SpSV computation.
*  @param[in]
*  alg                 SpSV algorithm for the SpSV computation.
*  @param[in]
*  spsvDescr           SpSV descriptor.
*  @param[out]
*  pBufferSizeInBytes  number of bytes of the temporary storage buffer.
*
*  \retval      HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval      HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p alpha, \p matA, \p x, \p y, \p spsvDescr or
*               \p pBufferSizeInBytes pointer is invalid.
*  \retval      HIPSPARSE_STATUS_NOT_SUPPORTED \p opA, \p computeType or \p alg is
*               currently not supported.
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpSV_bufferSize(hipsparseHandle_t           handle,
                                           hipsparseOperation_t        opA,
                                           const void*                 alpha,
                                           hipsparseConstSpMatDescr_t  matA,
                                           hipsparseConstDnVecDescr_t  x,
                                           const hipsparseDnVecDescr_t y,
                                           hipDataType                 computeType,
                                           hipsparseSpSVAlg_t          alg,
                                           hipsparseSpSVDescr_t        spsvDescr,
                                           size_t*                     pBufferSizeInBytes);
#elif(CUDART_VERSION >= 11030)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpSV_bufferSize(hipsparseHandle_t           handle,
                                           hipsparseOperation_t        opA,
                                           const void*                 alpha,
                                           const hipsparseSpMatDescr_t matA,
                                           const hipsparseDnVecDescr_t x,
                                           const hipsparseDnVecDescr_t y,
                                           hipDataType                 computeType,
                                           hipsparseSpSVAlg_t          alg,
                                           hipsparseSpSVDescr_t        spsvDescr,
                                           size_t*                     pBufferSizeInBytes);
#endif

/*! \ingroup generic_module
*  \brief Analysis step of solution of triangular linear system:
*  \f[
*    op(A) \cdot y := \alpha \cdot x,
*  \f]
*  where \f$A\f$ is a sparse matrix in CSR storage format, \f$x\f$ and \f$y\f$ are dense vectors.
*
*  \details
*  \p hipsparseSpSV_analysis performs the required analysis used when computing the 
*  solution of triangular linear system \f$op(A) \cdot y = \alpha \cdot x\f$,
*  where \f$A\f$ is a sparse matrix in CSR storage format, \f$x\f$ and \f$y\f$ are dense vectors.
*
*  @param[in]
*  handle          handle to the hipsparse library context queue.
*  @param[in]
*  opA             matrix operation type.
*  @param[in]
*  alpha           scalar \f$\alpha\f$.
*  @param[in]
*  matA            matrix descriptor.
*  @param[in]
*  x               vector descriptor.
*  @param[inout]
*  y               vector descriptor.
*  @param[in]
*  computeType     floating point precision for the SpSV computation.
*  @param[in]
*  alg             SpSV algorithm for the SpSV computation.
*  @param[in]
*  spsvDescr       SpSV descriptor.
*  @param[out]
*  externalBuffer  temporary storage buffer allocated by the user.
*
*  \retval      HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval      HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p alpha, \p matA, \p x, \p y, \p spsvDescr or
*               \p externalBuffer pointer is invalid.
*  \retval      HIPSPARSE_STATUS_NOT_SUPPORTED \p opA, \p computeType or \p alg is
*               currently not supported.
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpSV_analysis(hipsparseHandle_t           handle,
                                         hipsparseOperation_t        opA,
                                         const void*                 alpha,
                                         hipsparseConstSpMatDescr_t  matA,
                                         hipsparseConstDnVecDescr_t  x,
                                         const hipsparseDnVecDescr_t y,
                                         hipDataType                 computeType,
                                         hipsparseSpSVAlg_t          alg,
                                         hipsparseSpSVDescr_t        spsvDescr,
                                         void*                       externalBuffer);
#elif(CUDART_VERSION >= 11030)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpSV_analysis(hipsparseHandle_t           handle,
                                         hipsparseOperation_t        opA,
                                         const void*                 alpha,
                                         const hipsparseSpMatDescr_t matA,
                                         const hipsparseDnVecDescr_t x,
                                         const hipsparseDnVecDescr_t y,
                                         hipDataType                 computeType,
                                         hipsparseSpSVAlg_t          alg,
                                         hipsparseSpSVDescr_t        spsvDescr,
                                         void*                       externalBuffer);
#endif

/*! \ingroup generic_module
*  \brief Sparse triangular solve
*
*  \details
*  \p hipsparseSpSV_solve solves a sparse triangular linear system of a sparse
*  \f$m \times m\f$ matrix, defined in CSR or COO storage format, a dense solution vector
*  \f$y\f$ and the right-hand side \f$x\f$ that is multiplied by \f$\alpha\f$, such that
*  \f[
*    op(A) \cdot y = \alpha \cdot x,
*  \f]
*  with
*  \f[
*    op(A) = \left\{
*    \begin{array}{ll}
*        A,   & \text{if trans == HIPSPARSE_OPERATION_NON_TRANSPOSE} \\
*        A^T, & \text{if trans == HIPSPARSE_OPERATION_TRANSPOSE} \\
*        A^H, & \text{if trans == HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE}
*    \end{array}
*    \right.
*  \f]
*
*  @param[in]
*  handle          handle to the hipsparse library context queue.
*  @param[in]
*  opA             matrix operation type.
*  @param[in]
*  alpha           scalar \f$\alpha\f$.
*  @param[in]
*  matA            matrix descriptor.
*  @param[in]
*  x               vector descriptor.
*  @param[inout]
*  y               vector descriptor.
*  @param[in]
*  computeType     floating point precision for the SpSV computation.
*  @param[in]
*  alg             SpSV algorithm for the SpSV computation.
*  @param[in]
*  spsvDescr       SpSV descriptor.
*
*  \retval      HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval      HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p alpha, \p matA, \p x, \p y, or \p spsvDescr 
*               pointer is invalid.
*  \retval      HIPSPARSE_STATUS_NOT_SUPPORTED \p opA, \p computeType or \p alg is
*               currently not supported.
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpSV_solve(hipsparseHandle_t           handle,
                                      hipsparseOperation_t        opA,
                                      const void*                 alpha,
                                      hipsparseConstSpMatDescr_t  matA,
                                      hipsparseConstDnVecDescr_t  x,
                                      const hipsparseDnVecDescr_t y,
                                      hipDataType                 computeType,
                                      hipsparseSpSVAlg_t          alg,
                                      hipsparseSpSVDescr_t        spsvDescr);
#elif(CUDART_VERSION >= 11030)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpSV_solve(hipsparseHandle_t           handle,
                                      hipsparseOperation_t        opA,
                                      const void*                 alpha,
                                      const hipsparseSpMatDescr_t matA,
                                      const hipsparseDnVecDescr_t x,
                                      const hipsparseDnVecDescr_t y,
                                      hipDataType                 computeType,
                                      hipsparseSpSVAlg_t          alg,
                                      hipsparseSpSVDescr_t        spsvDescr);
#endif

/*! \ingroup generic_module
*  \brief Create sparse matrix triangular solve with multiple rhs descriptor
*  \details
*  \p hipsparseSpSM_createDescr creates a sparse matrix triangular solve with multiple rhs descriptor. It should be
*  destroyed at the end using hipsparseSpSM_destroyDescr().
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11031)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpSM_createDescr(hipsparseSpSMDescr_t* descr);
#endif

/*! \ingroup generic_module
*  \brief Destroy sparse matrix triangular solve with multiple rhs descriptor
*  \details
*  \p hipsparseSpSM_destroyDescr destroys a sparse matrix triangular solve with multiple rhs descriptor and releases all
*  resources used by the descriptor.
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11031)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpSM_destroyDescr(hipsparseSpSMDescr_t descr);
#endif

/*! \ingroup generic_module
*  \brief Buffer size step of solution of triangular linear system:
*  \f[
*    op(A) \cdot C := \alpha \cdot op(B),
*  \f]
*  where \f$A\f$ is a sparse matrix in CSR storage format, \f$B\f$ and \f$C\f$ are dense matrices.
*
*  \details
*  \p hipsparseSpSM_bufferSize computes the required user allocated buffer size needed when computing the 
*  solution of triangular linear system \f$op(A) \cdot C = \alpha \cdot op(B)\f$, where \f$A\f$ is a sparse matrix in CSR storage 
*  format, \f$B\f$ and \f$C\f$ are dense matrices.
*
*  @param[in]
*  handle              handle to the hipsparse library context queue.
*  @param[in]
*  opA                 matrix operation type for the sparse matrix \f$A\f$.
*  @param[in]
*  opB                 matrix operation type for the dense matrix \f$B\f$.
*  @param[in]
*  alpha               scalar \f$\alpha\f$.
*  @param[in]
*  matA                sparse matrix descriptor.
*  @param[in]
*  matB                dense matrix descriptor.
*  @param[inout]
*  matC                dense matrix descriptor.
*  @param[in]
*  computeType         floating point precision for the SpSM computation.
*  @param[in]
*  alg                 SpSM algorithm for the SpSM computation.
*  @param[in]
*  spsmDescr           SpSM descriptor.
*  @param[out]
*  pBufferSizeInBytes  number of bytes of the temporary storage buffer.
*
*  \retval      HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval      HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p alpha, \p matA, \p matB, \p matC, \p spsmDescr or
*               \p pBufferSizeInBytes pointer is invalid.
*  \retval      HIPSPARSE_STATUS_NOT_SUPPORTED \p opA, \p opB, \p computeType or \p alg is
*               currently not supported.
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpSM_bufferSize(hipsparseHandle_t           handle,
                                           hipsparseOperation_t        opA,
                                           hipsparseOperation_t        opB,
                                           const void*                 alpha,
                                           hipsparseConstSpMatDescr_t  matA,
                                           hipsparseConstDnMatDescr_t  matB,
                                           const hipsparseDnMatDescr_t matC,
                                           hipDataType                 computeType,
                                           hipsparseSpSMAlg_t          alg,
                                           hipsparseSpSMDescr_t        spsmDescr,
                                           size_t*                     pBufferSizeInBytes);
#elif(CUDART_VERSION >= 11031)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpSM_bufferSize(hipsparseHandle_t           handle,
                                           hipsparseOperation_t        opA,
                                           hipsparseOperation_t        opB,
                                           const void*                 alpha,
                                           const hipsparseSpMatDescr_t matA,
                                           const hipsparseDnMatDescr_t matB,
                                           const hipsparseDnMatDescr_t matC,
                                           hipDataType                 computeType,
                                           hipsparseSpSMAlg_t          alg,
                                           hipsparseSpSMDescr_t        spsmDescr,
                                           size_t*                     pBufferSizeInBytes);
#endif

/*! \ingroup generic_module
*  \brief Analysis step of solution of triangular linear system:
*  \f[
*    op(A) \cdot C := \alpha \cdot op(B),
*  \f]
*  where \f$A\f$ is a sparse matrix in CSR storage format, \f$B\f$ and \f$C\f$ are dense vectors.
*
*  \details
*  \p hipsparseSpSM_analysis performs the required analysis used when computing the 
*  solution of triangular linear system \f$op(A) \cdot C = \alpha \cdot op(B)\f$,
*  where \f$A\f$ is a sparse matrix in CSR storage format, \f$B\f$ and \f$C\f$ are dense vectors.
*
*  @param[in]
*  handle          handle to the hipsparse library context queue.
*  @param[in]
*  opA             matrix operation type for the sparse matrix \f$A\f$.
*  @param[in]
*  opB             matrix operation type for the dense matrix \f$B\f$.
*  @param[in]
*  alpha           scalar \f$\alpha\f$.
*  @param[in]
*  matA            sparse matrix descriptor.
*  @param[in]
*  matB            dense matrix descriptor.
*  @param[inout]
*  matC            dense matrix descriptor.
*  @param[in]
*  computeType     floating point precision for the SpSM computation.
*  @param[in]
*  alg             SpSM algorithm for the SpSM computation.
*  @param[in]
*  spsmDescr       SpSM descriptor.
*  @param[out]
*  externalBuffer  temporary storage buffer allocated by the user.
*
*  \retval      HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval      HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p alpha, \p matA, \p matB, \p matC, \p spsmDescr or
*               \p externalBuffer pointer is invalid.
*  \retval      HIPSPARSE_STATUS_NOT_SUPPORTED \p opA, \p opB, \p computeType or \p alg is
*               currently not supported.
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpSM_analysis(hipsparseHandle_t           handle,
                                         hipsparseOperation_t        opA,
                                         hipsparseOperation_t        opB,
                                         const void*                 alpha,
                                         hipsparseConstSpMatDescr_t  matA,
                                         hipsparseConstDnMatDescr_t  matB,
                                         const hipsparseDnMatDescr_t matC,
                                         hipDataType                 computeType,
                                         hipsparseSpSMAlg_t          alg,
                                         hipsparseSpSMDescr_t        spsmDescr,
                                         void*                       externalBuffer);
#elif(CUDART_VERSION >= 11031)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpSM_analysis(hipsparseHandle_t           handle,
                                         hipsparseOperation_t        opA,
                                         hipsparseOperation_t        opB,
                                         const void*                 alpha,
                                         const hipsparseSpMatDescr_t matA,
                                         const hipsparseDnMatDescr_t matB,
                                         const hipsparseDnMatDescr_t matC,
                                         hipDataType                 computeType,
                                         hipsparseSpSMAlg_t          alg,
                                         hipsparseSpSMDescr_t        spsmDescr,
                                         void*                       externalBuffer);
#endif

/*! \ingroup generic_module
*  \brief Sparse triangular system solve
*
*  \details
*  \p hipsparseSpSM_solve solves a sparse triangular linear system of a sparse
*  \f$m \times m\f$ matrix, defined in CSR or COO storage format, a dense solution matrix
*  \f$C\f$ and the right-hand side \f$B\f$ that is multiplied by \f$\alpha\f$, such that
*  \f[
*    op(A) \cdot C = \alpha \cdot op(B),
*  \f]
*  with
*  \f[
*    op(A) = \left\{
*    \begin{array}{ll}
*        A,   & \text{if trans == HIPSPARSE_OPERATION_NON_TRANSPOSE} \\
*        A^T, & \text{if trans == HIPSPARSE_OPERATION_TRANSPOSE} \\
*        A^H, & \text{if trans == HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE}
*    \end{array}
*    \right.
*  \f]
*  and
*  \f[
*    op(B) = \left\{
*    \begin{array}{ll}
*        B,   & \text{if transB == HIPSPARSE_OPERATION_NON_TRANSPOSE} \\
*        B^T, & \text{if transB == HIPSPARSE_OPERATION_TRANSPOSE} \\
*        B^H, & \text{if transB == HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE}
*    \end{array}
*    \right.
*  \f]
*
*  @param[in]
*  handle          handle to the hipsparse library context queue.
*  @param[in]
*  opA             matrix operation type for the sparse matrix \f$A\f$.
*  @param[in]
*  opB             matrix operation type for the dense matrix \f$B\f$.
*  @param[in]
*  alpha           scalar \f$\alpha\f$.
*  @param[in]
*  matA            sparse matrix descriptor.
*  @param[in]
*  matB            dense matrix descriptor.
*  @param[inout]
*  matC            dense matrix descriptor.
*  @param[in]
*  computeType     floating point precision for the SpSM computation.
*  @param[in]
*  alg             SpSM algorithm for the SpSM computation.
*  @param[in]
*  spsmDescr       SpSM descriptor.
*  @param[out]
*  externalBuffer  temporary storage buffer allocated by the user.
*
*  \retval      HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval      HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p alpha, \p matA, \p matB, \p matC, \p spsmDescr or
*               \p externalBuffer pointer is invalid.
*  \retval      HIPSPARSE_STATUS_NOT_SUPPORTED \p opA, \p opB, \p computeType or \p alg is
*               currently not supported.
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpSM_solve(hipsparseHandle_t           handle,
                                      hipsparseOperation_t        opA,
                                      hipsparseOperation_t        opB,
                                      const void*                 alpha,
                                      hipsparseConstSpMatDescr_t  matA,
                                      hipsparseConstDnMatDescr_t  matB,
                                      const hipsparseDnMatDescr_t matC,
                                      hipDataType                 computeType,
                                      hipsparseSpSMAlg_t          alg,
                                      hipsparseSpSMDescr_t        spsmDescr,
                                      void*                       externalBuffer);
#elif(CUDART_VERSION >= 11031)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpSM_solve(hipsparseHandle_t           handle,
                                      hipsparseOperation_t        opA,
                                      hipsparseOperation_t        opB,
                                      const void*                 alpha,
                                      const hipsparseSpMatDescr_t matA,
                                      const hipsparseDnMatDescr_t matB,
                                      const hipsparseDnMatDescr_t matC,
                                      hipDataType                 computeType,
                                      hipsparseSpSMAlg_t          alg,
                                      hipsparseSpSMDescr_t        spsmDescr,
                                      void*                       externalBuffer);
#endif

#ifdef __cplusplus
}
#endif

#endif // HIPSPARSE_H
