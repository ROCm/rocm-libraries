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

hipsparseStatus_t hipsparseScsr2csr_compress(hipsparseHandle_t         handle,
                                             int                       m,
                                             int                       n,
                                             const hipsparseMatDescr_t descrA,
                                             const float*              csrValA,
                                             const int*                csrColIndA,
                                             const int*                csrRowPtrA,
                                             int                       nnzA,
                                             const int*                nnzPerRow,
                                             float*                    csrValC,
                                             int*                      csrColIndC,
                                             int*                      csrRowPtrC,
                                             float                     tol)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseScsr2csr_compress((cusparseHandle_t)handle,
                                  m,
                                  n,
                                  (const cusparseMatDescr_t)descrA,
                                  csrValA,
                                  csrColIndA,
                                  csrRowPtrA,
                                  nnzA,
                                  nnzPerRow,
                                  csrValC,
                                  csrColIndC,
                                  csrRowPtrC,
                                  tol));
}

hipsparseStatus_t hipsparseDcsr2csr_compress(hipsparseHandle_t         handle,
                                             int                       m,
                                             int                       n,
                                             const hipsparseMatDescr_t descrA,
                                             const double*             csrValA,
                                             const int*                csrColIndA,
                                             const int*                csrRowPtrA,
                                             int                       nnzA,
                                             const int*                nnzPerRow,
                                             double*                   csrValC,
                                             int*                      csrColIndC,
                                             int*                      csrRowPtrC,
                                             double                    tol)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseDcsr2csr_compress((cusparseHandle_t)handle,
                                  m,
                                  n,
                                  (const cusparseMatDescr_t)descrA,
                                  csrValA,
                                  csrColIndA,
                                  csrRowPtrA,
                                  nnzA,
                                  nnzPerRow,
                                  csrValC,
                                  csrColIndC,
                                  csrRowPtrC,
                                  tol));
}

hipsparseStatus_t hipsparseCcsr2csr_compress(hipsparseHandle_t         handle,
                                             int                       m,
                                             int                       n,
                                             const hipsparseMatDescr_t descrA,
                                             const hipComplex*         csrValA,
                                             const int*                csrColIndA,
                                             const int*                csrRowPtrA,
                                             int                       nnzA,
                                             const int*                nnzPerRow,
                                             hipComplex*               csrValC,
                                             int*                      csrColIndC,
                                             int*                      csrRowPtrC,
                                             hipComplex                tol)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseCcsr2csr_compress((cusparseHandle_t)handle,
                                  m,
                                  n,
                                  (const cusparseMatDescr_t)descrA,
                                  (const cuComplex*)csrValA,
                                  csrColIndA,
                                  csrRowPtrA,
                                  nnzA,
                                  nnzPerRow,
                                  (cuComplex*)csrValC,
                                  csrColIndC,
                                  csrRowPtrC,
                                  {cuCrealf(tol), cuCimagf(tol)}));
}

hipsparseStatus_t hipsparseZcsr2csr_compress(hipsparseHandle_t         handle,
                                             int                       m,
                                             int                       n,
                                             const hipsparseMatDescr_t descrA,
                                             const hipDoubleComplex*   csrValA,
                                             const int*                csrColIndA,
                                             const int*                csrRowPtrA,
                                             int                       nnzA,
                                             const int*                nnzPerRow,
                                             hipDoubleComplex*         csrValC,
                                             int*                      csrColIndC,
                                             int*                      csrRowPtrC,
                                             hipDoubleComplex          tol)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseZcsr2csr_compress((cusparseHandle_t)handle,
                                  m,
                                  n,
                                  (const cusparseMatDescr_t)descrA,
                                  (const cuDoubleComplex*)csrValA,
                                  csrColIndA,
                                  csrRowPtrA,
                                  nnzA,
                                  nnzPerRow,
                                  (cuDoubleComplex*)csrValC,
                                  csrColIndC,
                                  csrRowPtrC,
                                  {cuCreal(tol), cuCimag(tol)}));
}
