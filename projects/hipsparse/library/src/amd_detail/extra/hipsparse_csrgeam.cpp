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

hipsparseStatus_t hipsparseXcsrgeamNnz(hipsparseHandle_t         handle,
                                       int                       m,
                                       int                       n,
                                       const hipsparseMatDescr_t descrA,
                                       int                       nnzA,
                                       const int*                csrRowPtrA,
                                       const int*                csrColIndA,
                                       const hipsparseMatDescr_t descrB,
                                       int                       nnzB,
                                       const int*                csrRowPtrB,
                                       const int*                csrColIndB,
                                       const hipsparseMatDescr_t descrC,
                                       int*                      csrRowPtrC,
                                       int*                      nnzTotalDevHostPtr)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_csrgeam_nnz((rocsparse_handle)handle,
                              m,
                              n,
                              (const rocsparse_mat_descr)descrA,
                              nnzA,
                              csrRowPtrA,
                              csrColIndA,
                              (const rocsparse_mat_descr)descrB,
                              nnzB,
                              csrRowPtrB,
                              csrColIndB,
                              (const rocsparse_mat_descr)descrC,
                              csrRowPtrC,
                              nnzTotalDevHostPtr));
}

hipsparseStatus_t hipsparseScsrgeam(hipsparseHandle_t         handle,
                                    int                       m,
                                    int                       n,
                                    const float*              alpha,
                                    const hipsparseMatDescr_t descrA,
                                    int                       nnzA,
                                    const float*              csrValA,
                                    const int*                csrRowPtrA,
                                    const int*                csrColIndA,
                                    const float*              beta,
                                    const hipsparseMatDescr_t descrB,
                                    int                       nnzB,
                                    const float*              csrValB,
                                    const int*                csrRowPtrB,
                                    const int*                csrColIndB,
                                    const hipsparseMatDescr_t descrC,
                                    float*                    csrValC,
                                    int*                      csrRowPtrC,
                                    int*                      csrColIndC)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_scsrgeam((rocsparse_handle)handle,
                           m,
                           n,
                           alpha,
                           (const rocsparse_mat_descr)descrA,
                           nnzA,
                           csrValA,
                           csrRowPtrA,
                           csrColIndA,
                           beta,
                           (const rocsparse_mat_descr)descrB,
                           nnzB,
                           csrValB,
                           csrRowPtrB,
                           csrColIndB,
                           (const rocsparse_mat_descr)descrC,
                           csrValC,
                           csrRowPtrC,
                           csrColIndC));
}

hipsparseStatus_t hipsparseDcsrgeam(hipsparseHandle_t         handle,
                                    int                       m,
                                    int                       n,
                                    const double*             alpha,
                                    const hipsparseMatDescr_t descrA,
                                    int                       nnzA,
                                    const double*             csrValA,
                                    const int*                csrRowPtrA,
                                    const int*                csrColIndA,
                                    const double*             beta,
                                    const hipsparseMatDescr_t descrB,
                                    int                       nnzB,
                                    const double*             csrValB,
                                    const int*                csrRowPtrB,
                                    const int*                csrColIndB,
                                    const hipsparseMatDescr_t descrC,
                                    double*                   csrValC,
                                    int*                      csrRowPtrC,
                                    int*                      csrColIndC)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_dcsrgeam((rocsparse_handle)handle,
                           m,
                           n,
                           alpha,
                           (const rocsparse_mat_descr)descrA,
                           nnzA,
                           csrValA,
                           csrRowPtrA,
                           csrColIndA,
                           beta,
                           (const rocsparse_mat_descr)descrB,
                           nnzB,
                           csrValB,
                           csrRowPtrB,
                           csrColIndB,
                           (const rocsparse_mat_descr)descrC,
                           csrValC,
                           csrRowPtrC,
                           csrColIndC));
}

hipsparseStatus_t hipsparseCcsrgeam(hipsparseHandle_t         handle,
                                    int                       m,
                                    int                       n,
                                    const hipComplex*         alpha,
                                    const hipsparseMatDescr_t descrA,
                                    int                       nnzA,
                                    const hipComplex*         csrValA,
                                    const int*                csrRowPtrA,
                                    const int*                csrColIndA,
                                    const hipComplex*         beta,
                                    const hipsparseMatDescr_t descrB,
                                    int                       nnzB,
                                    const hipComplex*         csrValB,
                                    const int*                csrRowPtrB,
                                    const int*                csrColIndB,
                                    const hipsparseMatDescr_t descrC,
                                    hipComplex*               csrValC,
                                    int*                      csrRowPtrC,
                                    int*                      csrColIndC)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_ccsrgeam((rocsparse_handle)handle,
                           m,
                           n,
                           (const rocsparse_float_complex*)alpha,
                           (const rocsparse_mat_descr)descrA,
                           nnzA,
                           (const rocsparse_float_complex*)csrValA,
                           csrRowPtrA,
                           csrColIndA,
                           (const rocsparse_float_complex*)beta,
                           (const rocsparse_mat_descr)descrB,
                           nnzB,
                           (const rocsparse_float_complex*)csrValB,
                           csrRowPtrB,
                           csrColIndB,
                           (const rocsparse_mat_descr)descrC,
                           (rocsparse_float_complex*)csrValC,
                           csrRowPtrC,
                           csrColIndC));
}

hipsparseStatus_t hipsparseZcsrgeam(hipsparseHandle_t         handle,
                                    int                       m,
                                    int                       n,
                                    const hipDoubleComplex*   alpha,
                                    const hipsparseMatDescr_t descrA,
                                    int                       nnzA,
                                    const hipDoubleComplex*   csrValA,
                                    const int*                csrRowPtrA,
                                    const int*                csrColIndA,
                                    const hipDoubleComplex*   beta,
                                    const hipsparseMatDescr_t descrB,
                                    int                       nnzB,
                                    const hipDoubleComplex*   csrValB,
                                    const int*                csrRowPtrB,
                                    const int*                csrColIndB,
                                    const hipsparseMatDescr_t descrC,
                                    hipDoubleComplex*         csrValC,
                                    int*                      csrRowPtrC,
                                    int*                      csrColIndC)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_zcsrgeam((rocsparse_handle)handle,
                           m,
                           n,
                           (const rocsparse_double_complex*)alpha,
                           (const rocsparse_mat_descr)descrA,
                           nnzA,
                           (const rocsparse_double_complex*)csrValA,
                           csrRowPtrA,
                           csrColIndA,
                           (const rocsparse_double_complex*)beta,
                           (const rocsparse_mat_descr)descrB,
                           nnzB,
                           (const rocsparse_double_complex*)csrValB,
                           csrRowPtrB,
                           csrColIndB,
                           (const rocsparse_mat_descr)descrC,
                           (rocsparse_double_complex*)csrValC,
                           csrRowPtrC,
                           csrColIndC));
}

hipsparseStatus_t hipsparseScsrgeam2_bufferSizeExt(hipsparseHandle_t         handle,
                                                   int                       m,
                                                   int                       n,
                                                   const float*              alpha,
                                                   const hipsparseMatDescr_t descrA,
                                                   int                       nnzA,
                                                   const float*              csrSortedValA,
                                                   const int*                csrSortedRowPtrA,
                                                   const int*                csrSortedColIndA,
                                                   const float*              beta,
                                                   const hipsparseMatDescr_t descrB,
                                                   int                       nnzB,
                                                   const float*              csrSortedValB,
                                                   const int*                csrSortedRowPtrB,
                                                   const int*                csrSortedColIndB,
                                                   const hipsparseMatDescr_t descrC,
                                                   const float*              csrSortedValC,
                                                   const int*                csrSortedRowPtrC,
                                                   const int*                csrSortedColIndC,
                                                   size_t*                   pBufferSizeInBytes)
{
    *pBufferSizeInBytes = 4;

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseDcsrgeam2_bufferSizeExt(hipsparseHandle_t         handle,
                                                   int                       m,
                                                   int                       n,
                                                   const double*             alpha,
                                                   const hipsparseMatDescr_t descrA,
                                                   int                       nnzA,
                                                   const double*             csrSortedValA,
                                                   const int*                csrSortedRowPtrA,
                                                   const int*                csrSortedColIndA,
                                                   const double*             beta,
                                                   const hipsparseMatDescr_t descrB,
                                                   int                       nnzB,
                                                   const double*             csrSortedValB,
                                                   const int*                csrSortedRowPtrB,
                                                   const int*                csrSortedColIndB,
                                                   const hipsparseMatDescr_t descrC,
                                                   const double*             csrSortedValC,
                                                   const int*                csrSortedRowPtrC,
                                                   const int*                csrSortedColIndC,
                                                   size_t*                   pBufferSizeInBytes)
{
    *pBufferSizeInBytes = 4;

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseCcsrgeam2_bufferSizeExt(hipsparseHandle_t         handle,
                                                   int                       m,
                                                   int                       n,
                                                   const hipComplex*         alpha,
                                                   const hipsparseMatDescr_t descrA,
                                                   int                       nnzA,
                                                   const hipComplex*         csrSortedValA,
                                                   const int*                csrSortedRowPtrA,
                                                   const int*                csrSortedColIndA,
                                                   const hipComplex*         beta,
                                                   const hipsparseMatDescr_t descrB,
                                                   int                       nnzB,
                                                   const hipComplex*         csrSortedValB,
                                                   const int*                csrSortedRowPtrB,
                                                   const int*                csrSortedColIndB,
                                                   const hipsparseMatDescr_t descrC,
                                                   const hipComplex*         csrSortedValC,
                                                   const int*                csrSortedRowPtrC,
                                                   const int*                csrSortedColIndC,
                                                   size_t*                   pBufferSizeInBytes)
{
    *pBufferSizeInBytes = 4;

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseZcsrgeam2_bufferSizeExt(hipsparseHandle_t         handle,
                                                   int                       m,
                                                   int                       n,
                                                   const hipDoubleComplex*   alpha,
                                                   const hipsparseMatDescr_t descrA,
                                                   int                       nnzA,
                                                   const hipDoubleComplex*   csrSortedValA,
                                                   const int*                csrSortedRowPtrA,
                                                   const int*                csrSortedColIndA,
                                                   const hipDoubleComplex*   beta,
                                                   const hipsparseMatDescr_t descrB,
                                                   int                       nnzB,
                                                   const hipDoubleComplex*   csrSortedValB,
                                                   const int*                csrSortedRowPtrB,
                                                   const int*                csrSortedColIndB,
                                                   const hipsparseMatDescr_t descrC,
                                                   const hipDoubleComplex*   csrSortedValC,
                                                   const int*                csrSortedRowPtrC,
                                                   const int*                csrSortedColIndC,
                                                   size_t*                   pBufferSizeInBytes)
{
    *pBufferSizeInBytes = 4;

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseXcsrgeam2Nnz(hipsparseHandle_t         handle,
                                        int                       m,
                                        int                       n,
                                        const hipsparseMatDescr_t descrA,
                                        int                       nnzA,
                                        const int*                csrSortedRowPtrA,
                                        const int*                csrSortedColIndA,
                                        const hipsparseMatDescr_t descrB,
                                        int                       nnzB,
                                        const int*                csrSortedRowPtrB,
                                        const int*                csrSortedColIndB,
                                        const hipsparseMatDescr_t descrC,
                                        int*                      csrSortedRowPtrC,
                                        int*                      nnzTotalDevHostPtr,
                                        void*                     workspace)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_csrgeam_nnz((rocsparse_handle)handle,
                              m,
                              n,
                              (const rocsparse_mat_descr)descrA,
                              nnzA,
                              csrSortedRowPtrA,
                              csrSortedColIndA,
                              (const rocsparse_mat_descr)descrB,
                              nnzB,
                              csrSortedRowPtrB,
                              csrSortedColIndB,
                              (const rocsparse_mat_descr)descrC,
                              csrSortedRowPtrC,
                              nnzTotalDevHostPtr));
}

hipsparseStatus_t hipsparseScsrgeam2(hipsparseHandle_t         handle,
                                     int                       m,
                                     int                       n,
                                     const float*              alpha,
                                     const hipsparseMatDescr_t descrA,
                                     int                       nnzA,
                                     const float*              csrSortedValA,
                                     const int*                csrSortedRowPtrA,
                                     const int*                csrSortedColIndA,
                                     const float*              beta,
                                     const hipsparseMatDescr_t descrB,
                                     int                       nnzB,
                                     const float*              csrSortedValB,
                                     const int*                csrSortedRowPtrB,
                                     const int*                csrSortedColIndB,
                                     const hipsparseMatDescr_t descrC,
                                     float*                    csrSortedValC,
                                     int*                      csrSortedRowPtrC,
                                     int*                      csrSortedColIndC,
                                     void*                     pBuffer)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_scsrgeam((rocsparse_handle)handle,
                           m,
                           n,
                           alpha,
                           (const rocsparse_mat_descr)descrA,
                           nnzA,
                           csrSortedValA,
                           csrSortedRowPtrA,
                           csrSortedColIndA,
                           beta,
                           (const rocsparse_mat_descr)descrB,
                           nnzB,
                           csrSortedValB,
                           csrSortedRowPtrB,
                           csrSortedColIndB,
                           (const rocsparse_mat_descr)descrC,
                           csrSortedValC,
                           csrSortedRowPtrC,
                           csrSortedColIndC));
}

hipsparseStatus_t hipsparseDcsrgeam2(hipsparseHandle_t         handle,
                                     int                       m,
                                     int                       n,
                                     const double*             alpha,
                                     const hipsparseMatDescr_t descrA,
                                     int                       nnzA,
                                     const double*             csrSortedValA,
                                     const int*                csrSortedRowPtrA,
                                     const int*                csrSortedColIndA,
                                     const double*             beta,
                                     const hipsparseMatDescr_t descrB,
                                     int                       nnzB,
                                     const double*             csrSortedValB,
                                     const int*                csrSortedRowPtrB,
                                     const int*                csrSortedColIndB,
                                     const hipsparseMatDescr_t descrC,
                                     double*                   csrSortedValC,
                                     int*                      csrSortedRowPtrC,
                                     int*                      csrSortedColIndC,
                                     void*                     pBuffer)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_dcsrgeam((rocsparse_handle)handle,
                           m,
                           n,
                           alpha,
                           (const rocsparse_mat_descr)descrA,
                           nnzA,
                           csrSortedValA,
                           csrSortedRowPtrA,
                           csrSortedColIndA,
                           beta,
                           (const rocsparse_mat_descr)descrB,
                           nnzB,
                           csrSortedValB,
                           csrSortedRowPtrB,
                           csrSortedColIndB,
                           (const rocsparse_mat_descr)descrC,
                           csrSortedValC,
                           csrSortedRowPtrC,
                           csrSortedColIndC));
}

hipsparseStatus_t hipsparseCcsrgeam2(hipsparseHandle_t         handle,
                                     int                       m,
                                     int                       n,
                                     const hipComplex*         alpha,
                                     const hipsparseMatDescr_t descrA,
                                     int                       nnzA,
                                     const hipComplex*         csrSortedValA,
                                     const int*                csrSortedRowPtrA,
                                     const int*                csrSortedColIndA,
                                     const hipComplex*         beta,
                                     const hipsparseMatDescr_t descrB,
                                     int                       nnzB,
                                     const hipComplex*         csrSortedValB,
                                     const int*                csrSortedRowPtrB,
                                     const int*                csrSortedColIndB,
                                     const hipsparseMatDescr_t descrC,
                                     hipComplex*               csrSortedValC,
                                     int*                      csrSortedRowPtrC,
                                     int*                      csrSortedColIndC,
                                     void*                     pBuffer)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_ccsrgeam((rocsparse_handle)handle,
                           m,
                           n,
                           (const rocsparse_float_complex*)alpha,
                           (const rocsparse_mat_descr)descrA,
                           nnzA,
                           (const rocsparse_float_complex*)csrSortedValA,
                           csrSortedRowPtrA,
                           csrSortedColIndA,
                           (const rocsparse_float_complex*)beta,
                           (const rocsparse_mat_descr)descrB,
                           nnzB,
                           (const rocsparse_float_complex*)csrSortedValB,
                           csrSortedRowPtrB,
                           csrSortedColIndB,
                           (const rocsparse_mat_descr)descrC,
                           (rocsparse_float_complex*)csrSortedValC,
                           csrSortedRowPtrC,
                           csrSortedColIndC));
}

hipsparseStatus_t hipsparseZcsrgeam2(hipsparseHandle_t         handle,
                                     int                       m,
                                     int                       n,
                                     const hipDoubleComplex*   alpha,
                                     const hipsparseMatDescr_t descrA,
                                     int                       nnzA,
                                     const hipDoubleComplex*   csrSortedValA,
                                     const int*                csrSortedRowPtrA,
                                     const int*                csrSortedColIndA,
                                     const hipDoubleComplex*   beta,
                                     const hipsparseMatDescr_t descrB,
                                     int                       nnzB,
                                     const hipDoubleComplex*   csrSortedValB,
                                     const int*                csrSortedRowPtrB,
                                     const int*                csrSortedColIndB,
                                     const hipsparseMatDescr_t descrC,
                                     hipDoubleComplex*         csrSortedValC,
                                     int*                      csrSortedRowPtrC,
                                     int*                      csrSortedColIndC,
                                     void*                     pBuffer)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_zcsrgeam((rocsparse_handle)handle,
                           m,
                           n,
                           (const rocsparse_double_complex*)alpha,
                           (const rocsparse_mat_descr)descrA,
                           nnzA,
                           (const rocsparse_double_complex*)csrSortedValA,
                           csrSortedRowPtrA,
                           csrSortedColIndA,
                           (const rocsparse_double_complex*)beta,
                           (const rocsparse_mat_descr)descrB,
                           nnzB,
                           (const rocsparse_double_complex*)csrSortedValB,
                           csrSortedRowPtrB,
                           csrSortedColIndB,
                           (const rocsparse_mat_descr)descrC,
                           (rocsparse_double_complex*)csrSortedValC,
                           csrSortedRowPtrC,
                           csrSortedColIndC));
}
