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

#if CUDART_VERSION < 11000
hipsparseStatus_t hipsparseShyb2csr(hipsparseHandle_t         handle,
                                    const hipsparseMatDescr_t descrA,
                                    const hipsparseHybMat_t   hybA,
                                    float*                    csrSortedValA,
                                    int*                      csrSortedRowPtrA,
                                    int*                      csrSortedColIndA)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseShyb2csr((cusparseHandle_t)handle,
                         (const cusparseMatDescr_t)descrA,
                         (const cusparseHybMat_t)hybA,
                         csrSortedValA,
                         csrSortedRowPtrA,
                         csrSortedColIndA));
}

hipsparseStatus_t hipsparseDhyb2csr(hipsparseHandle_t         handle,
                                    const hipsparseMatDescr_t descrA,
                                    const hipsparseHybMat_t   hybA,
                                    double*                   csrSortedValA,
                                    int*                      csrSortedRowPtrA,
                                    int*                      csrSortedColIndA)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseDhyb2csr((cusparseHandle_t)handle,
                         (const cusparseMatDescr_t)descrA,
                         (const cusparseHybMat_t)hybA,
                         csrSortedValA,
                         csrSortedRowPtrA,
                         csrSortedColIndA));
}

hipsparseStatus_t hipsparseChyb2csr(hipsparseHandle_t         handle,
                                    const hipsparseMatDescr_t descrA,
                                    const hipsparseHybMat_t   hybA,
                                    hipComplex*               csrSortedValA,
                                    int*                      csrSortedRowPtrA,
                                    int*                      csrSortedColIndA)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseChyb2csr((cusparseHandle_t)handle,
                         (const cusparseMatDescr_t)descrA,
                         (const cusparseHybMat_t)hybA,
                         csrSortedValA,
                         csrSortedRowPtrA,
                         csrSortedColIndA));
}

hipsparseStatus_t hipsparseZhyb2csr(hipsparseHandle_t         handle,
                                    const hipsparseMatDescr_t descrA,
                                    const hipsparseHybMat_t   hybA,
                                    hipDoubleComplex*         csrSortedValA,
                                    int*                      csrSortedRowPtrA,
                                    int*                      csrSortedColIndA)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseZhyb2csr((cusparseHandle_t)handle,
                         (const cusparseMatDescr_t)descrA,
                         (const cusparseHybMat_t)hybA,
                         csrSortedValA,
                         csrSortedRowPtrA,
                         csrSortedColIndA));
}
#endif
