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
                                      int*                      csrColIndC)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseSgebsr2csr((cusparseHandle_t)handle,
                           hipsparse::hipDirectionToCudaDirection(dirA),
                           mb,
                           nb,
                           (const cusparseMatDescr_t)descrA,
                           bsrValA,
                           bsrRowPtrA,
                           bsrColIndA,
                           rowBlockDim,
                           colBlockDim,
                           (const cusparseMatDescr_t)descrC,
                           csrValC,
                           csrRowPtrC,
                           csrColIndC));
}

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
                                      int*                      csrColIndC)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseDgebsr2csr((cusparseHandle_t)handle,
                           hipsparse::hipDirectionToCudaDirection(dirA),
                           mb,
                           nb,
                           (const cusparseMatDescr_t)descrA,
                           bsrValA,
                           bsrRowPtrA,
                           bsrColIndA,
                           rowBlockDim,
                           colBlockDim,
                           (const cusparseMatDescr_t)descrC,
                           csrValC,
                           csrRowPtrC,
                           csrColIndC));
}

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
                                      int*                      csrColIndC)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseCgebsr2csr((cusparseHandle_t)handle,
                           hipsparse::hipDirectionToCudaDirection(dirA),
                           mb,
                           nb,
                           (const cusparseMatDescr_t)descrA,
                           (const cuComplex*)bsrValA,
                           bsrRowPtrA,
                           bsrColIndA,
                           rowBlockDim,
                           colBlockDim,
                           (const cusparseMatDescr_t)descrC,
                           (cuComplex*)csrValC,
                           csrRowPtrC,
                           csrColIndC));
}

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
                                      int*                      csrColIndC)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseZgebsr2csr((cusparseHandle_t)handle,
                           hipsparse::hipDirectionToCudaDirection(dirA),
                           mb,
                           nb,
                           (const cusparseMatDescr_t)descrA,
                           (const cuDoubleComplex*)bsrValA,
                           bsrRowPtrA,
                           bsrColIndA,
                           rowBlockDim,
                           colBlockDim,
                           (const cusparseMatDescr_t)descrC,
                           (cuDoubleComplex*)csrValC,
                           csrRowPtrC,
                           csrColIndC));
}
