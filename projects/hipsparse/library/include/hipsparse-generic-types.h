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
#ifndef HIPSPARSE_GENERIC_TYPES_H
#define HIPSPARSE_GENERIC_TYPES_H

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
typedef struct hipsparseSpMatDescr_st* hipsparseSpMatDescr_t;
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
typedef struct hipsparseSpMatDescr_st const* hipsparseConstSpMatDescr_t;
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
    HIPSPARSE_FORMAT_CSR         = 1, /**< Compressed Sparse Row */
    HIPSPARSE_FORMAT_CSC         = 2, /**< Compressed Sparse Column */
    HIPSPARSE_FORMAT_COO         = 3, /**< Coordinate - Structure of Arrays */
    HIPSPARSE_FORMAT_COO_AOS     = 4, /**< Coordinate - Array of Structures */
    HIPSPARSE_FORMAT_BLOCKED_ELL = 5 /**< Blocked ELL */
} hipsparseFormat_t;
#else
#if(CUDART_VERSION >= 12000)
typedef enum
{
    HIPSPARSE_FORMAT_CSR         = 1, /**< Compressed Sparse Row */
    HIPSPARSE_FORMAT_CSC         = 2, /**< Compressed Sparse Column */
    HIPSPARSE_FORMAT_COO         = 3, /**< Coordinate - Structure of Arrays */
    HIPSPARSE_FORMAT_BLOCKED_ELL = 5 /**< Blocked ELL */
} hipsparseFormat_t;
#elif(CUDART_VERSION >= 11021 && CUDART_VERSION < 12000)
typedef enum
{
    HIPSPARSE_FORMAT_CSR         = 1, /**< Compressed Sparse Row */
    HIPSPARSE_FORMAT_CSC         = 2, /**< Compressed Sparse Column */
    HIPSPARSE_FORMAT_COO         = 3, /**< Coordinate - Structure of Arrays */
    HIPSPARSE_FORMAT_COO_AOS     = 4, /**< Coordinate - Array of Structures */
    HIPSPARSE_FORMAT_BLOCKED_ELL = 5 /**< Blocked ELL */
} hipsparseFormat_t;
#elif(CUDART_VERSION >= 10010 && CUDART_VERSION < 11021)
typedef enum
{
    HIPSPARSE_FORMAT_CSR     = 1, /**< Compressed Sparse Row */
    HIPSPARSE_FORMAT_COO     = 3, /**< Coordinate - Structure of Arrays */
    HIPSPARSE_FORMAT_COO_AOS = 4, /**< Coordinate - Array of Structures */
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
    HIPSPARSE_MM_ALG_DEFAULT        = 0, /**< Default algorithm */
    HIPSPARSE_COOMM_ALG1            = 1, /**< COO algorithm 1 */
    HIPSPARSE_COOMM_ALG2            = 2, /**< COO algorithm 2 */
    HIPSPARSE_COOMM_ALG3            = 3, /**< COO algorithm 3 */
    HIPSPARSE_CSRMM_ALG1            = 4, /**< CSR algorithm 1 */
    HIPSPARSE_SPMM_ALG_DEFAULT      = 0, /**< Default algorithm */
    HIPSPARSE_SPMM_COO_ALG1         = 1, /**< COO algorithm 1 */
    HIPSPARSE_SPMM_COO_ALG2         = 2, /**< COO algorithm 2 */
    HIPSPARSE_SPMM_COO_ALG3         = 3, /**< COO algorithm 3 */
    HIPSPARSE_SPMM_COO_ALG4         = 5, /**< COO algorithm 4 */
    HIPSPARSE_SPMM_CSR_ALG1         = 4, /**< CSR algorithm 1 */
    HIPSPARSE_SPMM_CSR_ALG2         = 6, /**< CSR algorithm 2 */
    HIPSPARSE_SPMM_CSR_ALG3         = 12, /**< CSR algorithm 3 */
    HIPSPARSE_SPMM_BLOCKED_ELL_ALG1 = 13 /**< Blocked ELL algorithm 1 */
} hipsparseSpMMAlg_t;
#else
#if(CUDART_VERSION >= 12000)
typedef enum
{
    HIPSPARSE_SPMM_ALG_DEFAULT      = 0, /**< Default algorithm */
    HIPSPARSE_SPMM_COO_ALG1         = 1, /**< COO algorithm 1 */
    HIPSPARSE_SPMM_COO_ALG2         = 2, /**< COO algorithm 2 */
    HIPSPARSE_SPMM_COO_ALG3         = 3, /**< COO algorithm 3 */
    HIPSPARSE_SPMM_COO_ALG4         = 5, /**< COO algorithm 4 */
    HIPSPARSE_SPMM_CSR_ALG1         = 4, /**< CSR algorithm 1 */
    HIPSPARSE_SPMM_CSR_ALG2         = 6, /**< CSR algorithm 2 */
    HIPSPARSE_SPMM_CSR_ALG3         = 12, /**< CSR algorithm 3 */
    HIPSPARSE_SPMM_BLOCKED_ELL_ALG1 = 13 /**< Blocked ELL algorithm 1 */
} hipsparseSpMMAlg_t;
#elif(CUDART_VERSION >= 11021 && CUDART_VERSION < 12000)
typedef enum
{
    HIPSPARSE_MM_ALG_DEFAULT        = 0, /**< Default algorithm */
    HIPSPARSE_COOMM_ALG1            = 1, /**< COO algorithm 1 */
    HIPSPARSE_COOMM_ALG2            = 2, /**< COO algorithm 2 */
    HIPSPARSE_COOMM_ALG3            = 3, /**< COO algorithm 3 */
    HIPSPARSE_CSRMM_ALG1            = 4, /**< CSR algorithm 1 */
    HIPSPARSE_SPMM_ALG_DEFAULT      = 0, /**< Default algorithm */
    HIPSPARSE_SPMM_COO_ALG1         = 1, /**< COO algorithm 1 */
    HIPSPARSE_SPMM_COO_ALG2         = 2, /**< COO algorithm 2 */
    HIPSPARSE_SPMM_COO_ALG3         = 3, /**< COO algorithm 3 */
    HIPSPARSE_SPMM_COO_ALG4         = 5, /**< COO algorithm 4 */
    HIPSPARSE_SPMM_CSR_ALG1         = 4, /**< CSR algorithm 1 */
    HIPSPARSE_SPMM_CSR_ALG2         = 6, /**< CSR algorithm 2 */
    HIPSPARSE_SPMM_CSR_ALG3         = 12, /**< CSR algorithm 3 */
    HIPSPARSE_SPMM_BLOCKED_ELL_ALG1 = 13 /**< Blocked ELL algorithm 1 */
} hipsparseSpMMAlg_t;
#elif(CUDART_VERSION >= 11003 && CUDART_VERSION < 11021)
typedef enum
{
    HIPSPARSE_MM_ALG_DEFAULT        = 0, /**< Default algorithm */
    HIPSPARSE_COOMM_ALG1            = 1, /**< COO algorithm 1 */
    HIPSPARSE_COOMM_ALG2            = 2, /**< COO algorithm 2 */
    HIPSPARSE_COOMM_ALG3            = 3, /**< COO algorithm 3 */
    HIPSPARSE_CSRMM_ALG1            = 4, /**< CSR algorithm 1 */
    HIPSPARSE_SPMM_ALG_DEFAULT      = 0, /**< Default algorithm */
    HIPSPARSE_SPMM_COO_ALG1         = 1, /**< COO algorithm 1 */
    HIPSPARSE_SPMM_COO_ALG2         = 2, /**< COO algorithm 2 */
    HIPSPARSE_SPMM_COO_ALG3         = 3, /**< COO algorithm 3 */
    HIPSPARSE_SPMM_COO_ALG4         = 5, /**< COO algorithm 4 */
    HIPSPARSE_SPMM_CSR_ALG1         = 4, /**< CSR algorithm 1 */
    HIPSPARSE_SPMM_CSR_ALG2         = 6, /**< CSR algorithm 2 */
    HIPSPARSE_SPMM_BLOCKED_ELL_ALG1 = 13 /**< Blocked ELL algorithm 1 */
} hipsparseSpMMAlg_t;
#elif(CUDART_VERSION >= 10010 && CUDART_VERSION < 11003)
typedef enum
{
    HIPSPARSE_MM_ALG_DEFAULT = 0, /**< Default algorithm */
    HIPSPARSE_COOMM_ALG1     = 1, /**< COO algorithm 1 */
    HIPSPARSE_COOMM_ALG2     = 2, /**< COO algorithm 2 */
    HIPSPARSE_COOMM_ALG3     = 3, /**< COO algorithm 3 */
    HIPSPARSE_CSRMM_ALG1     = 4 /**< CSR algorithm 1 */
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

#endif /* HIPSPARSE_GENERIC_TYPES_H */
