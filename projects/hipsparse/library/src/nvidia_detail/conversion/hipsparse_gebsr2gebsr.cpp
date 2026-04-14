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
                                                   int*                      pBufferSizeInBytes)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseSgebsr2gebsr_bufferSize((cusparseHandle_t)handle,
                                        hipsparse::hipDirectionToCudaDirection(dirA),
                                        mb,
                                        nb,
                                        nnzb,
                                        (const cusparseMatDescr_t)descrA,
                                        bsrValA,
                                        bsrRowPtrA,
                                        bsrColIndA,
                                        rowBlockDimA,
                                        colBlockDimA,
                                        rowBlockDimC,
                                        colBlockDimC,
                                        pBufferSizeInBytes));
}

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
                                                   int*                      pBufferSizeInBytes)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseDgebsr2gebsr_bufferSize((cusparseHandle_t)handle,
                                        hipsparse::hipDirectionToCudaDirection(dirA),
                                        mb,
                                        nb,
                                        nnzb,
                                        (const cusparseMatDescr_t)descrA,
                                        bsrValA,
                                        bsrRowPtrA,
                                        bsrColIndA,
                                        rowBlockDimA,
                                        colBlockDimA,
                                        rowBlockDimC,
                                        colBlockDimC,
                                        pBufferSizeInBytes));
}

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
                                                   int*                      pBufferSizeInBytes)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseCgebsr2gebsr_bufferSize((cusparseHandle_t)handle,
                                        hipsparse::hipDirectionToCudaDirection(dirA),
                                        mb,
                                        nb,
                                        nnzb,
                                        (const cusparseMatDescr_t)descrA,
                                        (const cuComplex*)bsrValA,
                                        bsrRowPtrA,
                                        bsrColIndA,
                                        rowBlockDimA,
                                        colBlockDimA,
                                        rowBlockDimC,
                                        colBlockDimC,
                                        pBufferSizeInBytes));
}

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
                                                   int*                      pBufferSizeInBytes)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseZgebsr2gebsr_bufferSize((cusparseHandle_t)handle,
                                        hipsparse::hipDirectionToCudaDirection(dirA),
                                        mb,
                                        nb,
                                        nnzb,
                                        (const cusparseMatDescr_t)descrA,
                                        (const cuDoubleComplex*)bsrValA,
                                        bsrRowPtrA,
                                        bsrColIndA,
                                        rowBlockDimA,
                                        colBlockDimA,
                                        rowBlockDimC,
                                        colBlockDimC,
                                        pBufferSizeInBytes));
}

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
                                           void*                     buffer)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseXgebsr2gebsrNnz((cusparseHandle_t)handle,
                                hipsparse::hipDirectionToCudaDirection(dirA),
                                mb,
                                nb,
                                nnzb,
                                (const cusparseMatDescr_t)descrA,
                                bsrRowPtrA,
                                bsrColIndA,
                                rowBlockDimA,
                                colBlockDimA,
                                (cusparseMatDescr_t)descrC,
                                bsrRowPtrC,
                                rowBlockDimC,
                                colBlockDimC,
                                nnzTotalDevHostPtr,
                                buffer));
}

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
                                        void*                     buffer)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseSgebsr2gebsr((cusparseHandle_t)handle,
                             hipsparse::hipDirectionToCudaDirection(dirA),
                             mb,
                             nb,
                             nnzb,
                             (const cusparseMatDescr_t)descrA,
                             bsrValA,
                             bsrRowPtrA,
                             bsrColIndA,
                             rowBlockDimA,
                             colBlockDimA,
                             (const cusparseMatDescr_t)descrC,
                             bsrValC,
                             bsrRowPtrC,
                             bsrColIndC,
                             rowBlockDimC,
                             colBlockDimC,
                             buffer));
}

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
                                        void*                     buffer)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseDgebsr2gebsr((cusparseHandle_t)handle,
                             hipsparse::hipDirectionToCudaDirection(dirA),
                             mb,
                             nb,
                             nnzb,
                             (const cusparseMatDescr_t)descrA,
                             bsrValA,
                             bsrRowPtrA,
                             bsrColIndA,
                             rowBlockDimA,
                             colBlockDimA,
                             (const cusparseMatDescr_t)descrC,
                             bsrValC,
                             bsrRowPtrC,
                             bsrColIndC,
                             rowBlockDimC,
                             colBlockDimC,
                             buffer));
}

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
                                        void*                     buffer)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseCgebsr2gebsr((cusparseHandle_t)handle,
                             hipsparse::hipDirectionToCudaDirection(dirA),
                             mb,
                             nb,
                             nnzb,
                             (const cusparseMatDescr_t)descrA,
                             (const cuComplex*)bsrValA,
                             bsrRowPtrA,
                             bsrColIndA,
                             rowBlockDimA,
                             colBlockDimA,
                             (const cusparseMatDescr_t)descrC,
                             (cuComplex*)bsrValC,
                             bsrRowPtrC,
                             bsrColIndC,
                             rowBlockDimC,
                             colBlockDimC,
                             buffer));
}

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
                                        void*                     buffer)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseZgebsr2gebsr((cusparseHandle_t)handle,
                             hipsparse::hipDirectionToCudaDirection(dirA),
                             mb,
                             nb,
                             nnzb,
                             (const cusparseMatDescr_t)descrA,
                             (const cuDoubleComplex*)bsrValA,
                             bsrRowPtrA,
                             bsrColIndA,
                             rowBlockDimA,
                             colBlockDimA,
                             (const cusparseMatDescr_t)descrC,
                             (cuDoubleComplex*)bsrValC,
                             bsrRowPtrC,
                             bsrColIndC,
                             rowBlockDimC,
                             colBlockDimC,
                             buffer));
}
