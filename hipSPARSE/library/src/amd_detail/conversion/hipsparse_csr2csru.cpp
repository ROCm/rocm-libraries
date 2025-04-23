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

// csru2csr struct - to hold permutation array
struct csru2csrInfo
{
    int  size = 0;
    int* P    = nullptr;
};

hipsparseStatus_t hipsparseScsr2csru(hipsparseHandle_t         handle,
                                     int                       m,
                                     int                       n,
                                     int                       nnz,
                                     const hipsparseMatDescr_t descrA,
                                     float*                    csrVal,
                                     const int*                csrRowPtr,
                                     int*                      csrColInd,
                                     csru2csrInfo_t            info,
                                     void*                     pBuffer)
{
    // Test for bad args
    if(handle == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    // Invalid sizes
    if(m < 0 || n < 0 || nnz < 0)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    // Quick return
    if(m == 0 || n == 0 || nnz == 0)
    {
        // nnz must be 0
        if(nnz != 0)
        {
            return HIPSPARSE_STATUS_INVALID_VALUE;
        }

        return HIPSPARSE_STATUS_SUCCESS;
    }

    // Invalid pointers
    if(descrA == nullptr || csrVal == nullptr || csrRowPtr == nullptr || csrColInd == nullptr
       || info == nullptr || pBuffer == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    // Check for valid permutation array
    if(info->P == nullptr || info->size != nnz)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    // Get stream
    hipStream_t stream;
    RETURN_IF_HIPSPARSE_ERROR(hipsparseGetStream(handle, &stream));

    // Unsort CSR column indices based on the given permutation
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_isctr((rocsparse_handle)handle,
                                              nnz,
                                              csrColInd,
                                              info->P,
                                              (int*)pBuffer,
                                              rocsparse_index_base_zero));

    // Copy unsorted column indices back to csrColInd
    RETURN_IF_HIP_ERROR(
        hipMemcpyAsync(csrColInd, pBuffer, sizeof(int) * nnz, hipMemcpyDeviceToDevice, stream));

    // Unsort CSR values based on the given permutation
    RETURN_IF_HIPSPARSE_ERROR(
        hipsparseSsctr(handle, nnz, csrVal, info->P, (float*)pBuffer, HIPSPARSE_INDEX_BASE_ZERO));

    // Copy unsorted values back to csrVal
    RETURN_IF_HIP_ERROR(
        hipMemcpyAsync(csrVal, pBuffer, sizeof(float) * nnz, hipMemcpyDeviceToDevice, stream));

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseDcsr2csru(hipsparseHandle_t         handle,
                                     int                       m,
                                     int                       n,
                                     int                       nnz,
                                     const hipsparseMatDescr_t descrA,
                                     double*                   csrVal,
                                     const int*                csrRowPtr,
                                     int*                      csrColInd,
                                     csru2csrInfo_t            info,
                                     void*                     pBuffer)
{
    // Test for bad args
    if(handle == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    // Invalid sizes
    if(m < 0 || n < 0 || nnz < 0)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    // Quick return
    if(m == 0 || n == 0 || nnz == 0)
    {
        // nnz must be 0
        if(nnz != 0)
        {
            return HIPSPARSE_STATUS_INVALID_VALUE;
        }

        return HIPSPARSE_STATUS_SUCCESS;
    }

    // Invalid pointers
    if(descrA == nullptr || csrVal == nullptr || csrRowPtr == nullptr || csrColInd == nullptr
       || info == nullptr || pBuffer == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    // Check for valid permutation array
    if(info->P == nullptr || info->size != nnz)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    // Get stream
    hipStream_t stream;
    RETURN_IF_HIPSPARSE_ERROR(hipsparseGetStream(handle, &stream));

    // Unsort CSR column indices based on the given permutation
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_isctr((rocsparse_handle)handle,
                                              nnz,
                                              csrColInd,
                                              info->P,
                                              (int*)pBuffer,
                                              rocsparse_index_base_zero));

    // Copy unsorted column indices back to csrColInd
    RETURN_IF_HIP_ERROR(
        hipMemcpyAsync(csrColInd, pBuffer, sizeof(int) * nnz, hipMemcpyDeviceToDevice, stream));

    // Unsort CSR values based on the given permutation
    RETURN_IF_HIPSPARSE_ERROR(
        hipsparseDsctr(handle, nnz, csrVal, info->P, (double*)pBuffer, HIPSPARSE_INDEX_BASE_ZERO));

    // Copy unsorted values back to csrVal
    RETURN_IF_HIP_ERROR(
        hipMemcpyAsync(csrVal, pBuffer, sizeof(double) * nnz, hipMemcpyDeviceToDevice, stream));

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseCcsr2csru(hipsparseHandle_t         handle,
                                     int                       m,
                                     int                       n,
                                     int                       nnz,
                                     const hipsparseMatDescr_t descrA,
                                     hipComplex*               csrVal,
                                     const int*                csrRowPtr,
                                     int*                      csrColInd,
                                     csru2csrInfo_t            info,
                                     void*                     pBuffer)
{
    // Test for bad args
    if(handle == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    // Invalid sizes
    if(m < 0 || n < 0 || nnz < 0)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    // Quick return
    if(m == 0 || n == 0 || nnz == 0)
    {
        // nnz must be 0
        if(nnz != 0)
        {
            return HIPSPARSE_STATUS_INVALID_VALUE;
        }

        return HIPSPARSE_STATUS_SUCCESS;
    }

    // Invalid pointers
    if(descrA == nullptr || csrVal == nullptr || csrRowPtr == nullptr || csrColInd == nullptr
       || info == nullptr || pBuffer == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    // Check for valid permutation array
    if(info->P == nullptr || info->size != nnz)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    // Get stream
    hipStream_t stream;
    RETURN_IF_HIPSPARSE_ERROR(hipsparseGetStream(handle, &stream));

    // Unsort CSR column indices based on the given permutation
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_isctr((rocsparse_handle)handle,
                                              nnz,
                                              csrColInd,
                                              info->P,
                                              (int*)pBuffer,
                                              rocsparse_index_base_zero));

    // Copy unsorted column indices back to csrColInd
    RETURN_IF_HIP_ERROR(
        hipMemcpyAsync(csrColInd, pBuffer, sizeof(int) * nnz, hipMemcpyDeviceToDevice, stream));

    // Unsort CSR values based on the given permutation
    RETURN_IF_HIPSPARSE_ERROR(hipsparseCsctr(
        handle, nnz, csrVal, info->P, (hipComplex*)pBuffer, HIPSPARSE_INDEX_BASE_ZERO));

    // Copy unsorted values back to csrVal
    RETURN_IF_HIP_ERROR(
        hipMemcpyAsync(csrVal, pBuffer, sizeof(hipComplex) * nnz, hipMemcpyDeviceToDevice, stream));

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseZcsr2csru(hipsparseHandle_t         handle,
                                     int                       m,
                                     int                       n,
                                     int                       nnz,
                                     const hipsparseMatDescr_t descrA,
                                     hipDoubleComplex*         csrVal,
                                     const int*                csrRowPtr,
                                     int*                      csrColInd,
                                     csru2csrInfo_t            info,
                                     void*                     pBuffer)
{
    // Test for bad args
    if(handle == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    // Invalid sizes
    if(m < 0 || n < 0 || nnz < 0)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    // Quick return
    if(m == 0 || n == 0 || nnz == 0)
    {
        // nnz must be 0
        if(nnz != 0)
        {
            return HIPSPARSE_STATUS_INVALID_VALUE;
        }

        return HIPSPARSE_STATUS_SUCCESS;
    }

    // Invalid pointers
    if(descrA == nullptr || csrVal == nullptr || csrRowPtr == nullptr || csrColInd == nullptr
       || info == nullptr || pBuffer == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    // Check for valid permutation array
    if(info->P == nullptr || info->size != nnz)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    // Get stream
    hipStream_t stream;
    RETURN_IF_HIPSPARSE_ERROR(hipsparseGetStream(handle, &stream));

    // Unsort CSR column indices based on the given permutation
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_isctr((rocsparse_handle)handle,
                                              nnz,
                                              csrColInd,
                                              info->P,
                                              (int*)pBuffer,
                                              rocsparse_index_base_zero));

    // Copy unsorted column indices back to csrColInd
    RETURN_IF_HIP_ERROR(
        hipMemcpyAsync(csrColInd, pBuffer, sizeof(int) * nnz, hipMemcpyDeviceToDevice, stream));

    // Unsort CSR values based on the given permutation
    RETURN_IF_HIPSPARSE_ERROR(hipsparseZsctr(
        handle, nnz, csrVal, info->P, (hipDoubleComplex*)pBuffer, HIPSPARSE_INDEX_BASE_ZERO));

    // Copy unsorted values back to csrVal
    RETURN_IF_HIP_ERROR(hipMemcpyAsync(
        csrVal, pBuffer, sizeof(hipDoubleComplex) * nnz, hipMemcpyDeviceToDevice, stream));

    return HIPSPARSE_STATUS_SUCCESS;
}
