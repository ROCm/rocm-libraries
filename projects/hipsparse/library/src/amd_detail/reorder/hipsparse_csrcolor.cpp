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

hipsparseStatus_t hipsparseScsrcolor(hipsparseHandle_t         handle,
                                     int                       m,
                                     int                       nnz,
                                     const hipsparseMatDescr_t descrA,
                                     const float*              csrValA,
                                     const int*                csrRowPtrA,
                                     const int*                csrColIndA,
                                     const float*              fractionToColor,
                                     int*                      ncolors,
                                     int*                      coloring,
                                     int*                      reordering,
                                     hipsparseColorInfo_t      info)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_scsrcolor((rocsparse_handle)handle,
                            m,
                            nnz,
                            (const rocsparse_mat_descr)descrA,
                            csrValA,
                            csrRowPtrA,
                            csrColIndA,
                            fractionToColor,
                            ncolors,
                            coloring,
                            reordering,
                            (rocsparse_mat_info)info));
}

hipsparseStatus_t hipsparseDcsrcolor(hipsparseHandle_t         handle,
                                     int                       m,
                                     int                       nnz,
                                     const hipsparseMatDescr_t descrA,
                                     const double*             csrValA,
                                     const int*                csrRowPtrA,
                                     const int*                csrColIndA,
                                     const double*             fractionToColor,
                                     int*                      ncolors,
                                     int*                      coloring,
                                     int*                      reordering,
                                     hipsparseColorInfo_t      info)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_dcsrcolor((rocsparse_handle)handle,
                            m,
                            nnz,
                            (const rocsparse_mat_descr)descrA,
                            csrValA,
                            csrRowPtrA,
                            csrColIndA,
                            fractionToColor,
                            ncolors,
                            coloring,
                            reordering,
                            (rocsparse_mat_info)info));
}

hipsparseStatus_t hipsparseCcsrcolor(hipsparseHandle_t         handle,
                                     int                       m,
                                     int                       nnz,
                                     const hipsparseMatDescr_t descrA,
                                     const hipComplex*         csrValA,
                                     const int*                csrRowPtrA,
                                     const int*                csrColIndA,
                                     const float*              fractionToColor,
                                     int*                      ncolors,
                                     int*                      coloring,
                                     int*                      reordering,
                                     hipsparseColorInfo_t      info)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_ccsrcolor((rocsparse_handle)handle,
                            m,
                            nnz,
                            (const rocsparse_mat_descr)descrA,
                            (const rocsparse_float_complex*)csrValA,
                            csrRowPtrA,
                            csrColIndA,
                            fractionToColor,
                            ncolors,
                            coloring,
                            reordering,
                            (rocsparse_mat_info)info));
}

hipsparseStatus_t hipsparseZcsrcolor(hipsparseHandle_t         handle,
                                     int                       m,
                                     int                       nnz,
                                     const hipsparseMatDescr_t descrA,
                                     const hipDoubleComplex*   csrValA,
                                     const int*                csrRowPtrA,
                                     const int*                csrColIndA,
                                     const double*             fractionToColor,
                                     int*                      ncolors,
                                     int*                      coloring,
                                     int*                      reordering,
                                     hipsparseColorInfo_t      info)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_zcsrcolor((rocsparse_handle)handle,
                            m,
                            nnz,
                            (const rocsparse_mat_descr)descrA,
                            (const rocsparse_double_complex*)csrValA,
                            csrRowPtrA,
                            csrColIndA,
                            fractionToColor,
                            ncolors,
                            coloring,
                            reordering,
                            (rocsparse_mat_info)info));
}
