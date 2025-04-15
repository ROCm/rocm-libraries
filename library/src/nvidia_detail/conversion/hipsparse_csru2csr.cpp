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
#include "hipsparse.h"

#include <cuda_runtime_api.h>
#include <cusparse_v2.h>
#include <hip/hip_runtime_api.h>

#include "../utility.h"

hipsparseStatus_t hipsparseCreateCsru2csrInfo(csru2csrInfo_t* info)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(cusparseCreateCsru2csrInfo(info));
}

hipsparseStatus_t hipsparseDestroyCsru2csrInfo(csru2csrInfo_t info)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(cusparseDestroyCsru2csrInfo(info));
}

#if CUDART_VERSION < 13000
hipsparseStatus_t hipsparseScsru2csr_bufferSizeExt(hipsparseHandle_t handle,
                                                   int               m,
                                                   int               n,
                                                   int               nnz,
                                                   float*            csrVal,
                                                   const int*        csrRowPtr,
                                                   int*              csrColInd,
                                                   csru2csrInfo_t    info,
                                                   size_t*           pBufferSizeInBytes)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseScsru2csr_bufferSizeExt((cusparseHandle_t)handle,
                                        m,
                                        n,
                                        nnz,
                                        csrVal,
                                        csrRowPtr,
                                        csrColInd,
                                        info,
                                        pBufferSizeInBytes));
}

hipsparseStatus_t hipsparseDcsru2csr_bufferSizeExt(hipsparseHandle_t handle,
                                                   int               m,
                                                   int               n,
                                                   int               nnz,
                                                   double*           csrVal,
                                                   const int*        csrRowPtr,
                                                   int*              csrColInd,
                                                   csru2csrInfo_t    info,
                                                   size_t*           pBufferSizeInBytes)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseDcsru2csr_bufferSizeExt((cusparseHandle_t)handle,
                                        m,
                                        n,
                                        nnz,
                                        csrVal,
                                        csrRowPtr,
                                        csrColInd,
                                        info,
                                        pBufferSizeInBytes));
}

hipsparseStatus_t hipsparseCcsru2csr_bufferSizeExt(hipsparseHandle_t handle,
                                                   int               m,
                                                   int               n,
                                                   int               nnz,
                                                   hipComplex*       csrVal,
                                                   const int*        csrRowPtr,
                                                   int*              csrColInd,
                                                   csru2csrInfo_t    info,
                                                   size_t*           pBufferSizeInBytes)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseCcsru2csr_bufferSizeExt((cusparseHandle_t)handle,
                                        m,
                                        n,
                                        nnz,
                                        (cuComplex*)csrVal,
                                        csrRowPtr,
                                        csrColInd,
                                        info,
                                        pBufferSizeInBytes));
}

hipsparseStatus_t hipsparseZcsru2csr_bufferSizeExt(hipsparseHandle_t handle,
                                                   int               m,
                                                   int               n,
                                                   int               nnz,
                                                   hipDoubleComplex* csrVal,
                                                   const int*        csrRowPtr,
                                                   int*              csrColInd,
                                                   csru2csrInfo_t    info,
                                                   size_t*           pBufferSizeInBytes)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseZcsru2csr_bufferSizeExt((cusparseHandle_t)handle,
                                        m,
                                        n,
                                        nnz,
                                        (cuDoubleComplex*)csrVal,
                                        csrRowPtr,
                                        csrColInd,
                                        info,
                                        pBufferSizeInBytes));
}
#endif

#if CUDART_VERSION < 13000
hipsparseStatus_t hipsparseScsru2csr(hipsparseHandle_t         handle,
                                     int                       m,
                                     int                       n,
                                     int                       nnz,
                                     const hipsparseMatDescr_t descrA,
                                     float*                    csrVal,
                                     const int*                csrRowPtr,
                                     int*                      csrColInd,
                                     csru2csrInfo_t            info,
                                     void*                     pBuffer)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseScsru2csr((cusparseHandle_t)handle,
                          m,
                          n,
                          nnz,
                          (const cusparseMatDescr_t)descrA,
                          csrVal,
                          csrRowPtr,
                          csrColInd,
                          info,
                          pBuffer));
}

hipsparseStatus_t hipsparseDcsru2csr(hipsparseHandle_t         handle,
                                     int                       m,
                                     int                       n,
                                     int                       nnz,
                                     const hipsparseMatDescr_t descrA,
                                     double*                   csrVal,
                                     const int*                csrRowPtr,
                                     int*                      csrColInd,
                                     csru2csrInfo_t            info,
                                     void*                     pBuffer)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseDcsru2csr((cusparseHandle_t)handle,
                          m,
                          n,
                          nnz,
                          (const cusparseMatDescr_t)descrA,
                          csrVal,
                          csrRowPtr,
                          csrColInd,
                          info,
                          pBuffer));
}

hipsparseStatus_t hipsparseCcsru2csr(hipsparseHandle_t         handle,
                                     int                       m,
                                     int                       n,
                                     int                       nnz,
                                     const hipsparseMatDescr_t descrA,
                                     hipComplex*               csrVal,
                                     const int*                csrRowPtr,
                                     int*                      csrColInd,
                                     csru2csrInfo_t            info,
                                     void*                     pBuffer)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseCcsru2csr((cusparseHandle_t)handle,
                          m,
                          n,
                          nnz,
                          (const cusparseMatDescr_t)descrA,
                          (cuComplex*)csrVal,
                          csrRowPtr,
                          csrColInd,
                          info,
                          pBuffer));
}

hipsparseStatus_t hipsparseZcsru2csr(hipsparseHandle_t         handle,
                                     int                       m,
                                     int                       n,
                                     int                       nnz,
                                     const hipsparseMatDescr_t descrA,
                                     hipDoubleComplex*         csrVal,
                                     const int*                csrRowPtr,
                                     int*                      csrColInd,
                                     csru2csrInfo_t            info,
                                     void*                     pBuffer)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseZcsru2csr((cusparseHandle_t)handle,
                          m,
                          n,
                          nnz,
                          (const cusparseMatDescr_t)descrA,
                          (cuDoubleComplex*)csrVal,
                          csrRowPtr,
                          csrColInd,
                          info,
                          pBuffer));
}
#endif
