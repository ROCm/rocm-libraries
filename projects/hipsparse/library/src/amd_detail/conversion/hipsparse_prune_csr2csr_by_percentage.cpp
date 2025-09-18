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

hipsparseStatus_t hipsparseSpruneCsr2csrByPercentage_bufferSize(hipsparseHandle_t         handle,
                                                                int                       m,
                                                                int                       n,
                                                                int                       nnzA,
                                                                const hipsparseMatDescr_t descrA,
                                                                const float*              csrValA,
                                                                const int* csrRowPtrA,
                                                                const int* csrColIndA,
                                                                float      percentage,
                                                                const hipsparseMatDescr_t descrC,
                                                                const float*              csrValC,
                                                                const int*  csrRowPtrC,
                                                                const int*  csrColIndC,
                                                                pruneInfo_t info,
                                                                size_t*     pBufferSizeInBytes)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_sprune_csr2csr_by_percentage_buffer_size((rocsparse_handle)handle,
                                                           m,
                                                           n,
                                                           nnzA,
                                                           (const rocsparse_mat_descr)descrA,
                                                           csrValA,
                                                           csrRowPtrA,
                                                           csrColIndA,
                                                           percentage,
                                                           (const rocsparse_mat_descr)descrC,
                                                           csrValC,
                                                           csrRowPtrC,
                                                           csrColIndC,
                                                           (rocsparse_mat_info)info,
                                                           pBufferSizeInBytes));
}

hipsparseStatus_t hipsparseDpruneCsr2csrByPercentage_bufferSize(hipsparseHandle_t         handle,
                                                                int                       m,
                                                                int                       n,
                                                                int                       nnzA,
                                                                const hipsparseMatDescr_t descrA,
                                                                const double*             csrValA,
                                                                const int* csrRowPtrA,
                                                                const int* csrColIndA,
                                                                double     percentage,
                                                                const hipsparseMatDescr_t descrC,
                                                                const double*             csrValC,
                                                                const int*  csrRowPtrC,
                                                                const int*  csrColIndC,
                                                                pruneInfo_t info,
                                                                size_t*     pBufferSizeInBytes)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_dprune_csr2csr_by_percentage_buffer_size((rocsparse_handle)handle,
                                                           m,
                                                           n,
                                                           nnzA,
                                                           (const rocsparse_mat_descr)descrA,
                                                           csrValA,
                                                           csrRowPtrA,
                                                           csrColIndA,
                                                           percentage,
                                                           (const rocsparse_mat_descr)descrC,
                                                           csrValC,
                                                           csrRowPtrC,
                                                           csrColIndC,
                                                           (rocsparse_mat_info)info,
                                                           pBufferSizeInBytes));
}

hipsparseStatus_t hipsparseSpruneCsr2csrByPercentage_bufferSizeExt(hipsparseHandle_t         handle,
                                                                   int                       m,
                                                                   int                       n,
                                                                   int                       nnzA,
                                                                   const hipsparseMatDescr_t descrA,
                                                                   const float* csrValA,
                                                                   const int*   csrRowPtrA,
                                                                   const int*   csrColIndA,
                                                                   float        percentage,
                                                                   const hipsparseMatDescr_t descrC,
                                                                   const float* csrValC,
                                                                   const int*   csrRowPtrC,
                                                                   const int*   csrColIndC,
                                                                   pruneInfo_t  info,
                                                                   size_t*      pBufferSizeInBytes)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_sprune_csr2csr_by_percentage_buffer_size((rocsparse_handle)handle,
                                                           m,
                                                           n,
                                                           nnzA,
                                                           (const rocsparse_mat_descr)descrA,
                                                           csrValA,
                                                           csrRowPtrA,
                                                           csrColIndA,
                                                           percentage,
                                                           (const rocsparse_mat_descr)descrC,
                                                           csrValC,
                                                           csrRowPtrC,
                                                           csrColIndC,
                                                           (rocsparse_mat_info)info,
                                                           pBufferSizeInBytes));
}

hipsparseStatus_t hipsparseDpruneCsr2csrByPercentage_bufferSizeExt(hipsparseHandle_t         handle,
                                                                   int                       m,
                                                                   int                       n,
                                                                   int                       nnzA,
                                                                   const hipsparseMatDescr_t descrA,
                                                                   const double* csrValA,
                                                                   const int*    csrRowPtrA,
                                                                   const int*    csrColIndA,
                                                                   double        percentage,
                                                                   const hipsparseMatDescr_t descrC,
                                                                   const double* csrValC,
                                                                   const int*    csrRowPtrC,
                                                                   const int*    csrColIndC,
                                                                   pruneInfo_t   info,
                                                                   size_t*       pBufferSizeInBytes)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_dprune_csr2csr_by_percentage_buffer_size((rocsparse_handle)handle,
                                                           m,
                                                           n,
                                                           nnzA,
                                                           (const rocsparse_mat_descr)descrA,
                                                           csrValA,
                                                           csrRowPtrA,
                                                           csrColIndA,
                                                           percentage,
                                                           (const rocsparse_mat_descr)descrC,
                                                           csrValC,
                                                           csrRowPtrC,
                                                           csrColIndC,
                                                           (rocsparse_mat_info)info,
                                                           pBufferSizeInBytes));
}

hipsparseStatus_t hipsparseSpruneCsr2csrNnzByPercentage(hipsparseHandle_t         handle,
                                                        int                       m,
                                                        int                       n,
                                                        int                       nnzA,
                                                        const hipsparseMatDescr_t descrA,
                                                        const float*              csrValA,
                                                        const int*                csrRowPtrA,
                                                        const int*                csrColIndA,
                                                        float                     percentage,
                                                        const hipsparseMatDescr_t descrC,
                                                        int*                      csrRowPtrC,
                                                        int*        nnzTotalDevHostPtr,
                                                        pruneInfo_t info,
                                                        void*       buffer)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_sprune_csr2csr_nnz_by_percentage((rocsparse_handle)handle,
                                                   m,
                                                   n,
                                                   nnzA,
                                                   (const rocsparse_mat_descr)descrA,
                                                   csrValA,
                                                   csrRowPtrA,
                                                   csrColIndA,
                                                   percentage,
                                                   (const rocsparse_mat_descr)descrC,
                                                   csrRowPtrC,
                                                   nnzTotalDevHostPtr,
                                                   (rocsparse_mat_info)info,
                                                   buffer));
}

hipsparseStatus_t hipsparseDpruneCsr2csrNnzByPercentage(hipsparseHandle_t         handle,
                                                        int                       m,
                                                        int                       n,
                                                        int                       nnzA,
                                                        const hipsparseMatDescr_t descrA,
                                                        const double*             csrValA,
                                                        const int*                csrRowPtrA,
                                                        const int*                csrColIndA,
                                                        double                    percentage,
                                                        const hipsparseMatDescr_t descrC,
                                                        int*                      csrRowPtrC,
                                                        int*        nnzTotalDevHostPtr,
                                                        pruneInfo_t info,
                                                        void*       buffer)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_dprune_csr2csr_nnz_by_percentage((rocsparse_handle)handle,
                                                   m,
                                                   n,
                                                   nnzA,
                                                   (const rocsparse_mat_descr)descrA,
                                                   csrValA,
                                                   csrRowPtrA,
                                                   csrColIndA,
                                                   percentage,
                                                   (const rocsparse_mat_descr)descrC,
                                                   csrRowPtrC,
                                                   nnzTotalDevHostPtr,
                                                   (rocsparse_mat_info)info,
                                                   buffer));
}

hipsparseStatus_t hipsparseSpruneCsr2csrByPercentage(hipsparseHandle_t         handle,
                                                     int                       m,
                                                     int                       n,
                                                     int                       nnzA,
                                                     const hipsparseMatDescr_t descrA,
                                                     const float*              csrValA,
                                                     const int*                csrRowPtrA,
                                                     const int*                csrColIndA,
                                                     float                     percentage,
                                                     const hipsparseMatDescr_t descrC,
                                                     float*                    csrValC,
                                                     const int*                csrRowPtrC,
                                                     int*                      csrColIndC,
                                                     pruneInfo_t               info,
                                                     void*                     buffer)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_sprune_csr2csr_by_percentage((rocsparse_handle)handle,
                                               m,
                                               n,
                                               nnzA,
                                               (const rocsparse_mat_descr)descrA,
                                               csrValA,
                                               csrRowPtrA,
                                               csrColIndA,
                                               percentage,
                                               (const rocsparse_mat_descr)descrC,
                                               csrValC,
                                               csrRowPtrC,
                                               csrColIndC,
                                               (rocsparse_mat_info)info,
                                               buffer));
}

hipsparseStatus_t hipsparseDpruneCsr2csrByPercentage(hipsparseHandle_t         handle,
                                                     int                       m,
                                                     int                       n,
                                                     int                       nnzA,
                                                     const hipsparseMatDescr_t descrA,
                                                     const double*             csrValA,
                                                     const int*                csrRowPtrA,
                                                     const int*                csrColIndA,
                                                     double                    percentage,
                                                     const hipsparseMatDescr_t descrC,
                                                     double*                   csrValC,
                                                     const int*                csrRowPtrC,
                                                     int*                      csrColIndC,
                                                     pruneInfo_t               info,
                                                     void*                     buffer)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_dprune_csr2csr_by_percentage((rocsparse_handle)handle,
                                               m,
                                               n,
                                               nnzA,
                                               (const rocsparse_mat_descr)descrA,
                                               csrValA,
                                               csrRowPtrA,
                                               csrColIndA,
                                               percentage,
                                               (const rocsparse_mat_descr)descrC,
                                               csrValC,
                                               csrRowPtrC,
                                               csrColIndC,
                                               (rocsparse_mat_info)info,
                                               buffer));
}
