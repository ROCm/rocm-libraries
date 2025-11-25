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
#ifndef HIPSPARSE_GENERIC_AUXILIARY_H
#define HIPSPARSE_GENERIC_AUXILIARY_H

#ifdef __cplusplus
extern "C" {
#endif

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

#ifdef __cplusplus
}
#endif

#endif /* HIPSPARSE_GENERIC_AUXILIARY_H */
