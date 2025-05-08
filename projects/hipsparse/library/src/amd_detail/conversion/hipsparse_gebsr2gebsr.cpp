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

hipsparseStatus_t hipsparseSgebsr2gebsr_bufferSize(hipsparseHandle_t         handle,
                                                   hipsparseDirection_t      dirA,
                                                   int                       mb,
                                                   int                       nb,
                                                   int                       nnzb,
                                                   const hipsparseMatDescr_t descrA,
                                                   const float*              bsrValA,
                                                   const int*                bsrRowPtrA,
                                                   const int*                bsrColIndA,
                                                   int                       rowBlockDimA,
                                                   int                       colBlockDimA,
                                                   int                       rowBlockDimC,
                                                   int                       colBlockDimC,
                                                   int*                      pBufferSizeInBytes)
{
    size_t bufSize;
    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse_sgebsr2gebsr_buffer_size((rocsparse_handle)handle,
                                           hipsparse::hipDirectionToHCCDirection(dirA),
                                           mb,
                                           nb,
                                           nnzb,
                                           (const rocsparse_mat_descr)descrA,
                                           bsrValA,
                                           bsrRowPtrA,
                                           bsrColIndA,
                                           rowBlockDimA,
                                           colBlockDimA,
                                           rowBlockDimC,
                                           colBlockDimC,
                                           pBufferSizeInBytes != nullptr ? &bufSize : nullptr));

    if(pBufferSizeInBytes != nullptr)
    {
        *pBufferSizeInBytes = bufSize;
    }
    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseDgebsr2gebsr_bufferSize(hipsparseHandle_t         handle,
                                                   hipsparseDirection_t      dirA,
                                                   int                       mb,
                                                   int                       nb,
                                                   int                       nnzb,
                                                   const hipsparseMatDescr_t descrA,
                                                   const double*             bsrValA,
                                                   const int*                bsrRowPtrA,
                                                   const int*                bsrColIndA,
                                                   int                       rowBlockDimA,
                                                   int                       colBlockDimA,
                                                   int                       rowBlockDimC,
                                                   int                       colBlockDimC,
                                                   int*                      pBufferSizeInBytes)
{
    size_t bufSize;
    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse_dgebsr2gebsr_buffer_size((rocsparse_handle)handle,
                                           hipsparse::hipDirectionToHCCDirection(dirA),
                                           mb,
                                           nb,
                                           nnzb,
                                           (const rocsparse_mat_descr)descrA,
                                           bsrValA,
                                           bsrRowPtrA,
                                           bsrColIndA,
                                           rowBlockDimA,
                                           colBlockDimA,
                                           rowBlockDimC,
                                           colBlockDimC,
                                           pBufferSizeInBytes != nullptr ? &bufSize : nullptr));
    if(pBufferSizeInBytes != nullptr)
    {
        *pBufferSizeInBytes = bufSize;
    }
    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseCgebsr2gebsr_bufferSize(hipsparseHandle_t         handle,
                                                   hipsparseDirection_t      dirA,
                                                   int                       mb,
                                                   int                       nb,
                                                   int                       nnzb,
                                                   const hipsparseMatDescr_t descrA,
                                                   const hipComplex*         bsrValA,
                                                   const int*                bsrRowPtrA,
                                                   const int*                bsrColIndA,
                                                   int                       rowBlockDimA,
                                                   int                       colBlockDimA,
                                                   int                       rowBlockDimC,
                                                   int                       colBlockDimC,
                                                   int*                      pBufferSizeInBytes)
{
    size_t bufSize;
    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse_cgebsr2gebsr_buffer_size((rocsparse_handle)handle,
                                           hipsparse::hipDirectionToHCCDirection(dirA),
                                           mb,
                                           nb,
                                           nnzb,
                                           (const rocsparse_mat_descr)descrA,
                                           (const rocsparse_float_complex*)bsrValA,
                                           bsrRowPtrA,
                                           bsrColIndA,
                                           rowBlockDimA,
                                           colBlockDimA,
                                           rowBlockDimC,
                                           colBlockDimC,
                                           pBufferSizeInBytes != nullptr ? &bufSize : nullptr));
    if(pBufferSizeInBytes != nullptr)
    {
        *pBufferSizeInBytes = bufSize;
    }
    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseZgebsr2gebsr_bufferSize(hipsparseHandle_t         handle,
                                                   hipsparseDirection_t      dirA,
                                                   int                       mb,
                                                   int                       nb,
                                                   int                       nnzb,
                                                   const hipsparseMatDescr_t descrA,
                                                   const hipDoubleComplex*   bsrValA,
                                                   const int*                bsrRowPtrA,
                                                   const int*                bsrColIndA,
                                                   int                       rowBlockDimA,
                                                   int                       colBlockDimA,
                                                   int                       rowBlockDimC,
                                                   int                       colBlockDimC,
                                                   int*                      pBufferSizeInBytes)
{
    size_t bufSize;
    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse_zgebsr2gebsr_buffer_size((rocsparse_handle)handle,
                                           hipsparse::hipDirectionToHCCDirection(dirA),
                                           mb,
                                           nb,
                                           nnzb,
                                           (const rocsparse_mat_descr)descrA,
                                           (const rocsparse_double_complex*)bsrValA,
                                           bsrRowPtrA,
                                           bsrColIndA,
                                           rowBlockDimA,
                                           colBlockDimA,
                                           rowBlockDimC,
                                           colBlockDimC,
                                           pBufferSizeInBytes != nullptr ? &bufSize : nullptr));
    if(pBufferSizeInBytes != nullptr)
    {
        *pBufferSizeInBytes = bufSize;
    }
    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseXgebsr2gebsrNnz(hipsparseHandle_t         handle,
                                           hipsparseDirection_t      dirA,
                                           int                       mb,
                                           int                       nb,
                                           int                       nnzb,
                                           const hipsparseMatDescr_t descrA,
                                           const int*                bsrRowPtrA,
                                           const int*                bsrColIndA,
                                           int                       rowBlockDimA,
                                           int                       colBlockDimA,
                                           const hipsparseMatDescr_t descrC,
                                           int*                      bsrRowPtrC,
                                           int                       rowBlockDimC,
                                           int                       colBlockDimC,
                                           int*                      nnzTotalDevHostPtr,
                                           void*                     buffer)
{
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_gebsr2gebsr_nnz((rocsparse_handle)handle,
                                                        hipsparse::hipDirectionToHCCDirection(dirA),
                                                        mb,
                                                        nb,
                                                        nnzb,
                                                        (const rocsparse_mat_descr)descrA,
                                                        bsrRowPtrA,
                                                        bsrColIndA,
                                                        rowBlockDimA,
                                                        colBlockDimA,
                                                        (const rocsparse_mat_descr)descrC,
                                                        bsrRowPtrC,
                                                        rowBlockDimC,
                                                        colBlockDimC,
                                                        nnzTotalDevHostPtr,
                                                        buffer));
    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseSgebsr2gebsr(hipsparseHandle_t         handle,
                                        hipsparseDirection_t      dirA,
                                        int                       mb,
                                        int                       nb,
                                        int                       nnzb,
                                        const hipsparseMatDescr_t descrA,
                                        const float*              bsrValA,
                                        const int*                bsrRowPtrA,
                                        const int*                bsrColIndA,
                                        int                       rowBlockDimA,
                                        int                       colBlockDimA,
                                        const hipsparseMatDescr_t descrC,
                                        float*                    bsrValC,
                                        int*                      bsrRowPtrC,
                                        int*                      bsrColIndC,
                                        int                       rowBlockDimC,
                                        int                       colBlockDimC,
                                        void*                     buffer)
{
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_sgebsr2gebsr((rocsparse_handle)handle,
                                                     hipsparse::hipDirectionToHCCDirection(dirA),
                                                     mb,
                                                     nb,
                                                     nnzb,
                                                     (const rocsparse_mat_descr)descrA,
                                                     bsrValA,
                                                     bsrRowPtrA,
                                                     bsrColIndA,
                                                     rowBlockDimA,
                                                     colBlockDimA,
                                                     (const rocsparse_mat_descr)descrC,
                                                     bsrValC,
                                                     bsrRowPtrC,
                                                     bsrColIndC,
                                                     rowBlockDimC,
                                                     colBlockDimC,
                                                     buffer));
    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseDgebsr2gebsr(hipsparseHandle_t         handle,
                                        hipsparseDirection_t      dirA,
                                        int                       mb,
                                        int                       nb,
                                        int                       nnzb,
                                        const hipsparseMatDescr_t descrA,
                                        const double*             bsrValA,
                                        const int*                bsrRowPtrA,
                                        const int*                bsrColIndA,
                                        int                       rowBlockDimA,
                                        int                       colBlockDimA,
                                        const hipsparseMatDescr_t descrC,
                                        double*                   bsrValC,
                                        int*                      bsrRowPtrC,
                                        int*                      bsrColIndC,
                                        int                       rowBlockDimC,
                                        int                       colBlockDimC,
                                        void*                     buffer)
{
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_dgebsr2gebsr((rocsparse_handle)handle,
                                                     hipsparse::hipDirectionToHCCDirection(dirA),
                                                     mb,
                                                     nb,
                                                     nnzb,
                                                     (const rocsparse_mat_descr)descrA,
                                                     bsrValA,
                                                     bsrRowPtrA,
                                                     bsrColIndA,
                                                     rowBlockDimA,
                                                     colBlockDimA,
                                                     (const rocsparse_mat_descr)descrC,
                                                     bsrValC,
                                                     bsrRowPtrC,
                                                     bsrColIndC,
                                                     rowBlockDimC,
                                                     colBlockDimC,
                                                     buffer));
    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseCgebsr2gebsr(hipsparseHandle_t         handle,
                                        hipsparseDirection_t      dirA,
                                        int                       mb,
                                        int                       nb,
                                        int                       nnzb,
                                        const hipsparseMatDescr_t descrA,
                                        const hipComplex*         bsrValA,
                                        const int*                bsrRowPtrA,
                                        const int*                bsrColIndA,
                                        int                       rowBlockDimA,
                                        int                       colBlockDimA,
                                        const hipsparseMatDescr_t descrC,
                                        hipComplex*               bsrValC,
                                        int*                      bsrRowPtrC,
                                        int*                      bsrColIndC,
                                        int                       rowBlockDimC,
                                        int                       colBlockDimC,
                                        void*                     buffer)
{
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_cgebsr2gebsr((rocsparse_handle)handle,
                                                     hipsparse::hipDirectionToHCCDirection(dirA),
                                                     mb,
                                                     nb,
                                                     nnzb,
                                                     (const rocsparse_mat_descr)descrA,
                                                     (const rocsparse_float_complex*)bsrValA,
                                                     bsrRowPtrA,
                                                     bsrColIndA,
                                                     rowBlockDimA,
                                                     colBlockDimA,
                                                     (const rocsparse_mat_descr)descrC,
                                                     (rocsparse_float_complex*)bsrValC,
                                                     bsrRowPtrC,
                                                     bsrColIndC,
                                                     rowBlockDimC,
                                                     colBlockDimC,
                                                     buffer));
    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseZgebsr2gebsr(hipsparseHandle_t         handle,
                                        hipsparseDirection_t      dirA,
                                        int                       mb,
                                        int                       nb,
                                        int                       nnzb,
                                        const hipsparseMatDescr_t descrA,
                                        const hipDoubleComplex*   bsrValA,
                                        const int*                bsrRowPtrA,
                                        const int*                bsrColIndA,
                                        int                       rowBlockDimA,
                                        int                       colBlockDimA,
                                        const hipsparseMatDescr_t descrC,
                                        hipDoubleComplex*         bsrValC,
                                        int*                      bsrRowPtrC,
                                        int*                      bsrColIndC,
                                        int                       rowBlockDimC,
                                        int                       colBlockDimC,
                                        void*                     buffer)
{
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_zgebsr2gebsr((rocsparse_handle)handle,
                                                     hipsparse::hipDirectionToHCCDirection(dirA),
                                                     mb,
                                                     nb,
                                                     nnzb,
                                                     (const rocsparse_mat_descr)descrA,
                                                     (const rocsparse_double_complex*)bsrValA,
                                                     bsrRowPtrA,
                                                     bsrColIndA,
                                                     rowBlockDimA,
                                                     colBlockDimA,
                                                     (const rocsparse_mat_descr)descrC,
                                                     (rocsparse_double_complex*)bsrValC,
                                                     bsrRowPtrC,
                                                     bsrColIndC,
                                                     rowBlockDimC,
                                                     colBlockDimC,
                                                     buffer));
    return HIPSPARSE_STATUS_SUCCESS;
}
