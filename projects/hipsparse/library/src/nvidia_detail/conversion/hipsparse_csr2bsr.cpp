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
                                       int*                      bsrNnzb)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseXcsr2bsrNnz((cusparseHandle_t)handle,
                            hipsparse::hipDirectionToCudaDirection(dirA),
                            m,
                            n,
                            (const cusparseMatDescr_t)descrA,
                            csrRowPtrA,
                            csrColIndA,
                            blockDim,
                            (const cusparseMatDescr_t)descrC,
                            bsrRowPtrC,
                            bsrNnzb));
}

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
                                    int*                      bsrColIndC)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseScsr2bsr((cusparseHandle_t)handle,
                         hipsparse::hipDirectionToCudaDirection(dirA),
                         m,
                         n,
                         (const cusparseMatDescr_t)descrA,
                         csrValA,
                         csrRowPtrA,
                         csrColIndA,
                         blockDim,
                         (const cusparseMatDescr_t)descrC,
                         bsrValC,
                         bsrRowPtrC,
                         bsrColIndC));
}

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
                                    int*                      bsrColIndC)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseDcsr2bsr((cusparseHandle_t)handle,
                         hipsparse::hipDirectionToCudaDirection(dirA),
                         m,
                         n,
                         (const cusparseMatDescr_t)descrA,
                         csrValA,
                         csrRowPtrA,
                         csrColIndA,
                         blockDim,
                         (const cusparseMatDescr_t)descrC,
                         bsrValC,
                         bsrRowPtrC,
                         bsrColIndC));
}

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
                                    int*                      bsrColIndC)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseCcsr2bsr((cusparseHandle_t)handle,
                         hipsparse::hipDirectionToCudaDirection(dirA),
                         m,
                         n,
                         (const cusparseMatDescr_t)descrA,
                         (const cuComplex*)csrValA,
                         csrRowPtrA,
                         csrColIndA,
                         blockDim,
                         (const cusparseMatDescr_t)descrC,
                         (cuComplex*)bsrValC,
                         bsrRowPtrC,
                         bsrColIndC));
}

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
                                    int*                      bsrColIndC)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseZcsr2bsr((cusparseHandle_t)handle,
                         hipsparse::hipDirectionToCudaDirection(dirA),
                         m,
                         n,
                         (const cusparseMatDescr_t)descrA,
                         (const cuDoubleComplex*)csrValA,
                         csrRowPtrA,
                         csrColIndA,
                         blockDim,
                         (const cusparseMatDescr_t)descrC,
                         (cuDoubleComplex*)bsrValC,
                         bsrRowPtrC,
                         bsrColIndC));
}
