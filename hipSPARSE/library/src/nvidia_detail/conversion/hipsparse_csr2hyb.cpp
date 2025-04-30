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
hipsparseStatus_t hipsparseScsr2hyb(hipsparseHandle_t         handle,
                                    int                       m,
                                    int                       n,
                                    const hipsparseMatDescr_t descrA,
                                    const float*              csrSortedValA,
                                    const int*                csrSortedRowPtrA,
                                    const int*                csrSortedColIndA,
                                    hipsparseHybMat_t         hybA,
                                    int                       userEllWidth,
                                    hipsparseHybPartition_t   partitionType)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseScsr2hyb((cusparseHandle_t)handle,
                         m,
                         n,
                         (const cusparseMatDescr_t)descrA,
                         csrSortedValA,
                         csrSortedRowPtrA,
                         csrSortedColIndA,
                         (const cusparseHybMat_t)hybA,
                         userEllWidth,
                         hipsparse::hipHybPartitionToCudaHybPartition(partitionType)));
}

hipsparseStatus_t hipsparseDcsr2hyb(hipsparseHandle_t         handle,
                                    int                       m,
                                    int                       n,
                                    const hipsparseMatDescr_t descrA,
                                    const double*             csrSortedValA,
                                    const int*                csrSortedRowPtrA,
                                    const int*                csrSortedColIndA,
                                    hipsparseHybMat_t         hybA,
                                    int                       userEllWidth,
                                    hipsparseHybPartition_t   partitionType)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseDcsr2hyb((cusparseHandle_t)handle,
                         m,
                         n,
                         (const cusparseMatDescr_t)descrA,
                         csrSortedValA,
                         csrSortedRowPtrA,
                         csrSortedColIndA,
                         (const cusparseHybMat_t)hybA,
                         userEllWidth,
                         hipsparse::hipHybPartitionToCudaHybPartition(partitionType)));
}

hipsparseStatus_t hipsparseCcsr2hyb(hipsparseHandle_t         handle,
                                    int                       m,
                                    int                       n,
                                    const hipsparseMatDescr_t descrA,
                                    const hipComplex*         csrSortedValA,
                                    const int*                csrSortedRowPtrA,
                                    const int*                csrSortedColIndA,
                                    hipsparseHybMat_t         hybA,
                                    int                       userEllWidth,
                                    hipsparseHybPartition_t   partitionType)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseCcsr2hyb((cusparseHandle_t)handle,
                         m,
                         n,
                         (const cusparseMatDescr_t)descrA,
                         (const cuComplex*)csrSortedValA,
                         csrSortedRowPtrA,
                         csrSortedColIndA,
                         (const cusparseHybMat_t)hybA,
                         userEllWidth,
                         hipsparse::hipHybPartitionToCudaHybPartition(partitionType)));
}

hipsparseStatus_t hipsparseZcsr2hyb(hipsparseHandle_t         handle,
                                    int                       m,
                                    int                       n,
                                    const hipsparseMatDescr_t descrA,
                                    const hipDoubleComplex*   csrSortedValA,
                                    const int*                csrSortedRowPtrA,
                                    const int*                csrSortedColIndA,
                                    hipsparseHybMat_t         hybA,
                                    int                       userEllWidth,
                                    hipsparseHybPartition_t   partitionType)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseZcsr2hyb((cusparseHandle_t)handle,
                         m,
                         n,
                         (const cusparseMatDescr_t)descrA,
                         (const cuDoubleComplex*)csrSortedValA,
                         csrSortedRowPtrA,
                         csrSortedColIndA,
                         (const cusparseHybMat_t)hybA,
                         userEllWidth,
                         hipsparse::hipHybPartitionToCudaHybPartition(partitionType)));
}
#endif
