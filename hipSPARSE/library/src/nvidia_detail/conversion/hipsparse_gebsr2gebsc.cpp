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

hipsparseStatus_t hipsparseSgebsr2gebsc_bufferSize(hipsparseHandle_t handle,
                                                   int               mb,
                                                   int               nb,
                                                   int               nnzb,
                                                   const float*      bsrVal,
                                                   const int*        bsrRowPtr,
                                                   const int*        bsrColInd,
                                                   int               rowBlockDim,
                                                   int               colBlockDim,
                                                   size_t*           pBufferSizeInBytes)
{
    int               cu_buffer_size;
    hipsparseStatus_t status = hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseSgebsr2gebsc_bufferSize((cusparseHandle_t)handle,
                                        mb,
                                        nb,
                                        nnzb,
                                        bsrVal,
                                        bsrRowPtr,
                                        bsrColInd,
                                        rowBlockDim,
                                        colBlockDim,
                                        &cu_buffer_size));
    pBufferSizeInBytes[0] = cu_buffer_size;
    return status;
}

hipsparseStatus_t hipsparseDgebsr2gebsc_bufferSize(hipsparseHandle_t handle,
                                                   int               mb,
                                                   int               nb,
                                                   int               nnzb,
                                                   const double*     bsrVal,
                                                   const int*        bsrRowPtr,
                                                   const int*        bsrColInd,
                                                   int               rowBlockDim,
                                                   int               colBlockDim,
                                                   size_t*           pBufferSizeInBytes)
{
    int               cu_buffer_size;
    hipsparseStatus_t status = hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseDgebsr2gebsc_bufferSize((cusparseHandle_t)handle,
                                        mb,
                                        nb,
                                        nnzb,
                                        bsrVal,
                                        bsrRowPtr,
                                        bsrColInd,
                                        rowBlockDim,
                                        colBlockDim,
                                        &cu_buffer_size));
    pBufferSizeInBytes[0] = cu_buffer_size;
    return status;
}

hipsparseStatus_t hipsparseCgebsr2gebsc_bufferSize(hipsparseHandle_t handle,
                                                   int               mb,
                                                   int               nb,
                                                   int               nnzb,
                                                   const hipComplex* bsrVal,
                                                   const int*        bsrRowPtr,
                                                   const int*        bsrColInd,
                                                   int               rowBlockDim,
                                                   int               colBlockDim,
                                                   size_t*           pBufferSizeInBytes)
{
    int               cu_buffer_size;
    hipsparseStatus_t status = hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseCgebsr2gebsc_bufferSize((cusparseHandle_t)handle,
                                        mb,
                                        nb,
                                        nnzb,
                                        (const cuComplex*)bsrVal,
                                        bsrRowPtr,
                                        bsrColInd,
                                        rowBlockDim,
                                        colBlockDim,
                                        &cu_buffer_size));
    pBufferSizeInBytes[0] = cu_buffer_size;
    return status;
}

hipsparseStatus_t hipsparseZgebsr2gebsc_bufferSize(hipsparseHandle_t       handle,
                                                   int                     mb,
                                                   int                     nb,
                                                   int                     nnzb,
                                                   const hipDoubleComplex* bsrVal,
                                                   const int*              bsrRowPtr,
                                                   const int*              bsrColInd,
                                                   int                     rowBlockDim,
                                                   int                     colBlockDim,
                                                   size_t*                 pBufferSizeInBytes)
{
    int               cu_buffer_size;
    hipsparseStatus_t status = hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseZgebsr2gebsc_bufferSize((cusparseHandle_t)handle,
                                        mb,
                                        nb,
                                        nnzb,
                                        (const cuDoubleComplex*)bsrVal,
                                        bsrRowPtr,
                                        bsrColInd,
                                        rowBlockDim,
                                        colBlockDim,
                                        &cu_buffer_size));
    pBufferSizeInBytes[0] = cu_buffer_size;
    return status;
}

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
                                        void*                temp_buffer)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseSgebsr2gebsc((cusparseHandle_t)handle,
                             mb,
                             nb,
                             nnzb,
                             bsrVal,
                             bsrRowPtr,
                             bsrColInd,
                             rowBlockDim,
                             colBlockDim,
                             bscVal,
                             bscRowInd,
                             bscColPtr,
                             hipsparse::hipActionToCudaAction(copyValues),
                             hipsparse::hipIndexBaseToCudaIndexBase(idxBase),
                             temp_buffer));
}

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
                                        void*                temp_buffer)
{

    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseDgebsr2gebsc((cusparseHandle_t)handle,
                             mb,
                             nb,
                             nnzb,
                             bsrVal,
                             bsrRowPtr,
                             bsrColInd,
                             rowBlockDim,
                             colBlockDim,
                             bscVal,
                             bscRowInd,
                             bscColPtr,
                             hipsparse::hipActionToCudaAction(copyValues),
                             hipsparse::hipIndexBaseToCudaIndexBase(idxBase),
                             temp_buffer));
}

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
                                        void*                temp_buffer)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseCgebsr2gebsc((cusparseHandle_t)handle,
                             mb,
                             nb,
                             nnzb,
                             (const cuComplex*)bsrVal,
                             bsrRowPtr,
                             bsrColInd,
                             rowBlockDim,
                             colBlockDim,
                             (cuComplex*)bscVal,
                             bscRowInd,
                             bscColPtr,
                             hipsparse::hipActionToCudaAction(copyValues),
                             hipsparse::hipIndexBaseToCudaIndexBase(idxBase),
                             temp_buffer));
}

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
                                        void*                   temp_buffer)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseZgebsr2gebsc((cusparseHandle_t)handle,
                             mb,
                             nb,
                             nnzb,
                             (const cuDoubleComplex*)bsrVal,
                             bsrRowPtr,
                             bsrColInd,
                             rowBlockDim,
                             colBlockDim,
                             (cuDoubleComplex*)bscVal,
                             bscRowInd,
                             bscColPtr,
                             hipsparse::hipActionToCudaAction(copyValues),
                             hipsparse::hipIndexBaseToCudaIndexBase(idxBase),
                             temp_buffer));
}
