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
#ifndef HIPSPARSE_CSRU2CSR_H
#define HIPSPARSE_CSRU2CSR_H

#ifdef __cplusplus
extern "C" {
#endif

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

#ifdef __cplusplus
}
#endif

#endif /* HIPSPARSE_CSRU2CSR_H */
