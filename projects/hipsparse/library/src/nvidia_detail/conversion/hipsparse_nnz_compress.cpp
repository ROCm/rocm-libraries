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

#if CUDART_VERSION < 13000
hipsparseStatus_t hipsparseSnnz_compress(hipsparseHandle_t         handle,
                                         int                       m,
                                         const hipsparseMatDescr_t descrA,
                                         const float*              csrValA,
                                         const int*                csrRowPtrA,
                                         int*                      nnzPerRow,
                                         int*                      nnzC,
                                         float                     tol)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseSnnz_compress((cusparseHandle_t)handle,
                              m,
                              (const cusparseMatDescr_t)descrA,
                              csrValA,
                              csrRowPtrA,
                              nnzPerRow,
                              nnzC,
                              tol));
}

hipsparseStatus_t hipsparseDnnz_compress(hipsparseHandle_t         handle,
                                         int                       m,
                                         const hipsparseMatDescr_t descrA,
                                         const double*             csrValA,
                                         const int*                csrRowPtrA,
                                         int*                      nnzPerRow,
                                         int*                      nnzC,
                                         double                    tol)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseDnnz_compress((cusparseHandle_t)handle,
                              m,
                              (const cusparseMatDescr_t)descrA,
                              csrValA,
                              csrRowPtrA,
                              nnzPerRow,
                              nnzC,
                              tol));
}

hipsparseStatus_t hipsparseCnnz_compress(hipsparseHandle_t         handle,
                                         int                       m,
                                         const hipsparseMatDescr_t descrA,
                                         const hipComplex*         csrValA,
                                         const int*                csrRowPtrA,
                                         int*                      nnzPerRow,
                                         int*                      nnzC,
                                         hipComplex                tol)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseCnnz_compress((cusparseHandle_t)handle,
                              m,
                              (const cusparseMatDescr_t)descrA,
                              (const cuComplex*)csrValA,
                              csrRowPtrA,
                              nnzPerRow,
                              nnzC,
                              {cuCrealf(tol), cuCimagf(tol)}));
}

hipsparseStatus_t hipsparseZnnz_compress(hipsparseHandle_t         handle,
                                         int                       m,
                                         const hipsparseMatDescr_t descrA,
                                         const hipDoubleComplex*   csrValA,
                                         const int*                csrRowPtrA,
                                         int*                      nnzPerRow,
                                         int*                      nnzC,
                                         hipDoubleComplex          tol)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseZnnz_compress((cusparseHandle_t)handle,
                              m,
                              (const cusparseMatDescr_t)descrA,
                              (const cuDoubleComplex*)csrValA,
                              csrRowPtrA,
                              nnzPerRow,
                              nnzC,
                              {cuCreal(tol), cuCimag(tol)}));
}
#endif
