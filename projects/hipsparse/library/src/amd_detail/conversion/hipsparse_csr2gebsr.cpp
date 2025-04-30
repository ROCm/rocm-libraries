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

hipsparseStatus_t hipsparseScsr2gebsr_bufferSize(hipsparseHandle_t         handle,
                                                 hipsparseDirection_t      dir,
                                                 int                       m,
                                                 int                       n,
                                                 const hipsparseMatDescr_t csr_descr,
                                                 const float*              csrVal,
                                                 const int*                csrRowPtr,
                                                 const int*                csrColInd,
                                                 int                       rowBlockDim,
                                                 int                       colBlockDim,
                                                 size_t*                   pBufferSizeInBytes)
{
    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse_scsr2gebsr_buffer_size((rocsparse_handle)handle,
                                         hipsparse::hipDirectionToHCCDirection(dir),
                                         m,
                                         n,
                                         (const rocsparse_mat_descr)csr_descr,
                                         csrVal,
                                         csrRowPtr,
                                         csrColInd,
                                         rowBlockDim,
                                         colBlockDim,
                                         pBufferSizeInBytes));
    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseDcsr2gebsr_bufferSize(hipsparseHandle_t         handle,
                                                 hipsparseDirection_t      dir,
                                                 int                       m,
                                                 int                       n,
                                                 const hipsparseMatDescr_t csr_descr,
                                                 const double*             csrVal,
                                                 const int*                csrRowPtr,
                                                 const int*                csrColInd,
                                                 int                       rowBlockDim,
                                                 int                       colBlockDim,
                                                 size_t*                   pBufferSizeInBytes)
{
    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse_dcsr2gebsr_buffer_size((rocsparse_handle)handle,
                                         hipsparse::hipDirectionToHCCDirection(dir),
                                         m,
                                         n,
                                         (const rocsparse_mat_descr)csr_descr,
                                         csrVal,
                                         csrRowPtr,
                                         csrColInd,
                                         rowBlockDim,
                                         colBlockDim,
                                         pBufferSizeInBytes));
    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseCcsr2gebsr_bufferSize(hipsparseHandle_t         handle,
                                                 hipsparseDirection_t      dir,
                                                 int                       m,
                                                 int                       n,
                                                 const hipsparseMatDescr_t csr_descr,
                                                 const hipComplex*         csrVal,
                                                 const int*                csrRowPtr,
                                                 const int*                csrColInd,
                                                 int                       rowBlockDim,
                                                 int                       colBlockDim,
                                                 size_t*                   pBufferSizeInBytes)
{
    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse_ccsr2gebsr_buffer_size((rocsparse_handle)handle,
                                         hipsparse::hipDirectionToHCCDirection(dir),
                                         m,
                                         n,
                                         (const rocsparse_mat_descr)csr_descr,
                                         (const rocsparse_float_complex*)csrVal,
                                         csrRowPtr,
                                         csrColInd,
                                         rowBlockDim,
                                         colBlockDim,
                                         pBufferSizeInBytes));
    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseZcsr2gebsr_bufferSize(hipsparseHandle_t         handle,
                                                 hipsparseDirection_t      dir,
                                                 int                       m,
                                                 int                       n,
                                                 const hipsparseMatDescr_t csr_descr,
                                                 const hipDoubleComplex*   csrVal,
                                                 const int*                csrRowPtr,
                                                 const int*                csrColInd,
                                                 int                       rowBlockDim,
                                                 int                       colBlockDim,
                                                 size_t*                   pBufferSizeInBytes)
{
    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse_zcsr2gebsr_buffer_size((rocsparse_handle)handle,
                                         hipsparse::hipDirectionToHCCDirection(dir),
                                         m,
                                         n,
                                         (const rocsparse_mat_descr)csr_descr,
                                         (const rocsparse_double_complex*)csrVal,
                                         csrRowPtr,
                                         csrColInd,
                                         rowBlockDim,
                                         colBlockDim,
                                         pBufferSizeInBytes));
    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseXcsr2gebsrNnz(hipsparseHandle_t         handle,
                                         hipsparseDirection_t      dir,
                                         int                       m,
                                         int                       n,
                                         const hipsparseMatDescr_t csr_descr,
                                         const int*                csrRowPtr,
                                         const int*                csrColInd,
                                         const hipsparseMatDescr_t bsr_descr,
                                         int*                      bsrRowPtr,
                                         int                       rowBlockDim,
                                         int                       colBlockDim,
                                         int*                      bsrNnzDevhost,
                                         void*                     pbuffer)
{

    RETURN_IF_ROCSPARSE_ERROR(rocsparse_csr2gebsr_nnz((rocsparse_handle)handle,
                                                      hipsparse::hipDirectionToHCCDirection(dir),
                                                      m,
                                                      n,
                                                      (const rocsparse_mat_descr)csr_descr,
                                                      csrRowPtr,
                                                      csrColInd,
                                                      (const rocsparse_mat_descr)bsr_descr,
                                                      bsrRowPtr,
                                                      rowBlockDim,
                                                      colBlockDim,
                                                      bsrNnzDevhost,
                                                      pbuffer));
    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseScsr2gebsr(hipsparseHandle_t         handle,
                                      hipsparseDirection_t      dir,
                                      int                       m,
                                      int                       n,
                                      const hipsparseMatDescr_t csr_descr,
                                      const float*              csrVal,
                                      const int*                csrRowPtr,
                                      const int*                csrColInd,
                                      const hipsparseMatDescr_t bsr_descr,
                                      float*                    bsrVal,
                                      int*                      bsrRowPtr,
                                      int*                      bsrColInd,
                                      int                       rowBlockDim,
                                      int                       colBlockDim,
                                      void*                     pbuffer)
{
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_scsr2gebsr((rocsparse_handle)handle,
                                                   hipsparse::hipDirectionToHCCDirection(dir),
                                                   m,
                                                   n,
                                                   (const rocsparse_mat_descr)csr_descr,
                                                   csrVal,
                                                   csrRowPtr,
                                                   csrColInd,
                                                   (const rocsparse_mat_descr)bsr_descr,
                                                   bsrVal,
                                                   bsrRowPtr,
                                                   bsrColInd,
                                                   rowBlockDim,
                                                   colBlockDim,
                                                   pbuffer));
    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseDcsr2gebsr(hipsparseHandle_t         handle,
                                      hipsparseDirection_t      dir,
                                      int                       m,
                                      int                       n,
                                      const hipsparseMatDescr_t csr_descr,
                                      const double*             csrVal,
                                      const int*                csrRowPtr,
                                      const int*                csrColInd,
                                      const hipsparseMatDescr_t bsr_descr,
                                      double*                   bsrVal,
                                      int*                      bsrRowPtr,
                                      int*                      bsrColInd,
                                      int                       rowBlockDim,
                                      int                       colBlockDim,
                                      void*                     pbuffer)
{
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_dcsr2gebsr((rocsparse_handle)handle,
                                                   hipsparse::hipDirectionToHCCDirection(dir),
                                                   m,
                                                   n,
                                                   (const rocsparse_mat_descr)csr_descr,
                                                   csrVal,
                                                   csrRowPtr,
                                                   csrColInd,
                                                   (const rocsparse_mat_descr)bsr_descr,
                                                   bsrVal,
                                                   bsrRowPtr,
                                                   bsrColInd,
                                                   rowBlockDim,
                                                   colBlockDim,
                                                   pbuffer));
    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseCcsr2gebsr(hipsparseHandle_t         handle,
                                      hipsparseDirection_t      dir,
                                      int                       m,
                                      int                       n,
                                      const hipsparseMatDescr_t csr_descr,
                                      const hipComplex*         csrVal,
                                      const int*                csrRowPtr,
                                      const int*                csrColInd,
                                      const hipsparseMatDescr_t bsr_descr,
                                      hipComplex*               bsrVal,
                                      int*                      bsrRowPtr,
                                      int*                      bsrColInd,
                                      int                       rowBlockDim,
                                      int                       colBlockDim,
                                      void*                     pbuffer)
{
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_ccsr2gebsr((rocsparse_handle)handle,
                                                   hipsparse::hipDirectionToHCCDirection(dir),
                                                   m,
                                                   n,
                                                   (const rocsparse_mat_descr)csr_descr,
                                                   (const rocsparse_float_complex*)csrVal,
                                                   csrRowPtr,
                                                   csrColInd,
                                                   (const rocsparse_mat_descr)bsr_descr,
                                                   (rocsparse_float_complex*)bsrVal,
                                                   bsrRowPtr,
                                                   bsrColInd,
                                                   rowBlockDim,
                                                   colBlockDim,
                                                   pbuffer));
    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseZcsr2gebsr(hipsparseHandle_t         handle,
                                      hipsparseDirection_t      dir,
                                      int                       m,
                                      int                       n,
                                      const hipsparseMatDescr_t csr_descr,
                                      const hipDoubleComplex*   csrVal,
                                      const int*                csrRowPtr,
                                      const int*                csrColInd,
                                      const hipsparseMatDescr_t bsr_descr,
                                      hipDoubleComplex*         bsrVal,
                                      int*                      bsrRowPtr,
                                      int*                      bsrColInd,
                                      int                       rowBlockDim,
                                      int                       colBlockDim,
                                      void*                     pbuffer)
{
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_zcsr2gebsr((rocsparse_handle)handle,
                                                   hipsparse::hipDirectionToHCCDirection(dir),
                                                   m,
                                                   n,
                                                   (const rocsparse_mat_descr)csr_descr,
                                                   (const rocsparse_double_complex*)csrVal,
                                                   csrRowPtr,
                                                   csrColInd,
                                                   (const rocsparse_mat_descr)bsr_descr,
                                                   (rocsparse_double_complex*)bsrVal,
                                                   bsrRowPtr,
                                                   bsrColInd,
                                                   rowBlockDim,
                                                   colBlockDim,
                                                   pbuffer));
    return HIPSPARSE_STATUS_SUCCESS;
}
