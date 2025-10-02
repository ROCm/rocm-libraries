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
#ifndef HIPSPARSE_SPGEMM_REUSE_H
#define HIPSPARSE_SPGEMM_REUSE_H

#ifdef __cplusplus
extern "C" {
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
*  \par Example
*  \snippet example_hipsparse_spgemm_reuse.cpp doc example
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

#ifdef __cplusplus
}
#endif

#endif /* HIPSPARSE_SPGEMM_REUSE_H */
