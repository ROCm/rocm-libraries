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

#include "hipsparse.h"

#include <hip/hip_complex.h>
#include <hip/hip_runtime_api.h>
#include <rocsparse/rocsparse.h>

#include "../utility.h"

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
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_scsr2hyb((rocsparse_handle)handle,
                           m,
                           n,
                           (rocsparse_mat_descr)descrA,
                           csrSortedValA,
                           csrSortedRowPtrA,
                           csrSortedColIndA,
                           (rocsparse_hyb_mat)hybA,
                           userEllWidth,
                           hipsparse::hipHybPartToHCCHybPart(partitionType)));
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
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_dcsr2hyb((rocsparse_handle)handle,
                           m,
                           n,
                           (rocsparse_mat_descr)descrA,
                           csrSortedValA,
                           csrSortedRowPtrA,
                           csrSortedColIndA,
                           (rocsparse_hyb_mat)hybA,
                           userEllWidth,
                           hipsparse::hipHybPartToHCCHybPart(partitionType)));
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
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_ccsr2hyb((rocsparse_handle)handle,
                           m,
                           n,
                           (rocsparse_mat_descr)descrA,
                           (const rocsparse_float_complex*)csrSortedValA,
                           csrSortedRowPtrA,
                           csrSortedColIndA,
                           (rocsparse_hyb_mat)hybA,
                           userEllWidth,
                           hipsparse::hipHybPartToHCCHybPart(partitionType)));
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
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_zcsr2hyb((rocsparse_handle)handle,
                           m,
                           n,
                           (rocsparse_mat_descr)descrA,
                           (const rocsparse_double_complex*)csrSortedValA,
                           csrSortedRowPtrA,
                           csrSortedColIndA,
                           (rocsparse_hyb_mat)hybA,
                           userEllWidth,
                           hipsparse::hipHybPartToHCCHybPart(partitionType)));
}
