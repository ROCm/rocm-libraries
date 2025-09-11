/*! \file */
/* ************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights Reserved.
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
#ifndef HIPSPARSE_TYPES_H
#define HIPSPARSE_TYPES_H

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

// clang-format on

#endif /* HIPSPARSE_TYPES_H */
