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
#include <cassert>

// csru2csr struct - to hold permutation array
struct csru2csrInfo
{
    int  size = 0;
    int* P    = nullptr;
};

hipsparseStatus_t hipsparseCreateCsru2csrInfo(csru2csrInfo_t* info)
{
    if(info == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    *info = new csru2csrInfo;

    // Initialize permutation array with nullptr
    (*info)->size = 0;
    (*info)->P    = nullptr;

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseDestroyCsru2csrInfo(csru2csrInfo_t info)
{
    // Check if info structure has been created
    if(info != nullptr)
    {
        // Check if permutation array is allocated
        if(info->P != nullptr)
        {
            RETURN_IF_HIP_ERROR(hipFree(info->P));
            info->size = 0;
        }

        delete info;
    }

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseScsru2csr_bufferSizeExt(hipsparseHandle_t handle,
                                                   int               m,
                                                   int               n,
                                                   int               nnz,
                                                   float*            csrVal,
                                                   const int*        csrRowPtr,
                                                   int*              csrColInd,
                                                   csru2csrInfo_t    info,
                                                   size_t*           pBufferSizeInBytes)
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
        // nnz must be 0 and pBufferSizeInBytes must be valid
        if(nnz != 0 || pBufferSizeInBytes == nullptr)
        {
            return HIPSPARSE_STATUS_INVALID_VALUE;
        }

        *pBufferSizeInBytes = 4;

        return HIPSPARSE_STATUS_SUCCESS;
    }

    // Invalid pointers
    if(csrVal == nullptr || csrRowPtr == nullptr || csrColInd == nullptr || info == nullptr
       || pBufferSizeInBytes == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    // Determine required buffer size for CSR sort
    RETURN_IF_HIPSPARSE_ERROR(hipsparseXcsrsort_bufferSizeExt(
        handle, m, n, nnz, csrRowPtr, csrColInd, pBufferSizeInBytes));

    // We need a buffer of at least nnz * sizeof(float)
    size_t min_size     = nnz * sizeof(float);
    *pBufferSizeInBytes = std::max(*pBufferSizeInBytes, min_size);

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseDcsru2csr_bufferSizeExt(hipsparseHandle_t handle,
                                                   int               m,
                                                   int               n,
                                                   int               nnz,
                                                   double*           csrVal,
                                                   const int*        csrRowPtr,
                                                   int*              csrColInd,
                                                   csru2csrInfo_t    info,
                                                   size_t*           pBufferSizeInBytes)
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
        // nnz must be 0 and pBufferSizeInBytes must be valid
        if(nnz != 0 || pBufferSizeInBytes == nullptr)
        {
            return HIPSPARSE_STATUS_INVALID_VALUE;
        }

        *pBufferSizeInBytes = 4;

        return HIPSPARSE_STATUS_SUCCESS;
    }

    // Invalid pointers
    if(csrVal == nullptr || csrRowPtr == nullptr || csrColInd == nullptr || info == nullptr
       || pBufferSizeInBytes == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    // Determine required buffer size for CSR sort
    RETURN_IF_HIPSPARSE_ERROR(hipsparseXcsrsort_bufferSizeExt(
        handle, m, n, nnz, csrRowPtr, csrColInd, pBufferSizeInBytes));

    // We need a buffer of at least nnz * sizeof(double)
    size_t min_size     = nnz * sizeof(double);
    *pBufferSizeInBytes = std::max(*pBufferSizeInBytes, min_size);

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseCcsru2csr_bufferSizeExt(hipsparseHandle_t handle,
                                                   int               m,
                                                   int               n,
                                                   int               nnz,
                                                   hipComplex*       csrVal,
                                                   const int*        csrRowPtr,
                                                   int*              csrColInd,
                                                   csru2csrInfo_t    info,
                                                   size_t*           pBufferSizeInBytes)
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
        // nnz must be 0 and pBufferSizeInBytes must be valid
        if(nnz != 0 || pBufferSizeInBytes == nullptr)
        {
            return HIPSPARSE_STATUS_INVALID_VALUE;
        }

        *pBufferSizeInBytes = 4;

        return HIPSPARSE_STATUS_SUCCESS;
    }

    // Invalid pointers
    if(csrVal == nullptr || csrRowPtr == nullptr || csrColInd == nullptr || info == nullptr
       || pBufferSizeInBytes == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    // Determine required buffer size for CSR sort
    RETURN_IF_HIPSPARSE_ERROR(hipsparseXcsrsort_bufferSizeExt(
        handle, m, n, nnz, csrRowPtr, csrColInd, pBufferSizeInBytes));

    // We need a buffer of at least nnz * sizeof(hipComplex)
    size_t min_size     = nnz * sizeof(hipComplex);
    *pBufferSizeInBytes = std::max(*pBufferSizeInBytes, min_size);

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseZcsru2csr_bufferSizeExt(hipsparseHandle_t handle,
                                                   int               m,
                                                   int               n,
                                                   int               nnz,
                                                   hipDoubleComplex* csrVal,
                                                   const int*        csrRowPtr,
                                                   int*              csrColInd,
                                                   csru2csrInfo_t    info,
                                                   size_t*           pBufferSizeInBytes)
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
        // nnz must be 0 and pBufferSizeInBytes must be valid
        if(nnz != 0 || pBufferSizeInBytes == nullptr)
        {
            return HIPSPARSE_STATUS_INVALID_VALUE;
        }

        *pBufferSizeInBytes = 4;

        return HIPSPARSE_STATUS_SUCCESS;
    }

    // Invalid pointers
    if(csrVal == nullptr || csrRowPtr == nullptr || csrColInd == nullptr || info == nullptr
       || pBufferSizeInBytes == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    // Determine required buffer size for CSR sort
    RETURN_IF_HIPSPARSE_ERROR(hipsparseXcsrsort_bufferSizeExt(
        handle, m, n, nnz, csrRowPtr, csrColInd, pBufferSizeInBytes));

    // We need a buffer of at least nnz * sizeof(hipDoubleComplex)
    size_t min_size     = nnz * sizeof(hipDoubleComplex);
    *pBufferSizeInBytes = std::max(*pBufferSizeInBytes, min_size);

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseScsru2csr(hipsparseHandle_t         handle,
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

    // De-allocate permutation array, if already allocated but sizes do not match
    if(info->P != nullptr && info->size != nnz)
    {
        RETURN_IF_HIP_ERROR(hipFree(info->P));
        info->size = 0;
    }

    // Allocate memory inside info structure to keep track of the permutation
    // if it has not yet been allocated with matching size
    if(info->P == nullptr)
    {
        // size must be 0
        assert(info->size == 0);

        RETURN_IF_HIP_ERROR(hipMalloc((void**)&info->P, sizeof(int) * nnz));

        info->size = nnz;
    }

    // Initialize permutation with identity
    RETURN_IF_HIPSPARSE_ERROR(hipsparseCreateIdentityPermutation(handle, nnz, info->P));

    // Sort CSR columns
    RETURN_IF_HIPSPARSE_ERROR(
        hipsparseXcsrsort(handle, m, n, nnz, descrA, csrRowPtr, csrColInd, info->P, pBuffer));

    // Sort CSR values
    RETURN_IF_HIPSPARSE_ERROR(
        hipsparseSgthr(handle, nnz, csrVal, (float*)pBuffer, info->P, HIPSPARSE_INDEX_BASE_ZERO));

    // Get stream
    hipStream_t stream;
    RETURN_IF_HIPSPARSE_ERROR(hipsparseGetStream(handle, &stream));

    // Copy sorted values back to csrVal
    RETURN_IF_HIP_ERROR(
        hipMemcpyAsync(csrVal, pBuffer, sizeof(float) * nnz, hipMemcpyDeviceToDevice, stream));

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseDcsru2csr(hipsparseHandle_t         handle,
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

    // De-allocate permutation array, if already allocated but sizes do not match
    if(info->P != nullptr && info->size != nnz)
    {
        RETURN_IF_HIP_ERROR(hipFree(info->P));
        info->size = 0;
    }

    // Allocate memory inside info structure to keep track of the permutation
    // if it has not yet been allocated with matching size
    if(info->P == nullptr)
    {
        // size must be 0
        assert(info->size == 0);

        RETURN_IF_HIP_ERROR(hipMalloc((void**)&info->P, sizeof(int) * nnz));

        info->size = nnz;
    }

    // Initialize permutation with identity
    RETURN_IF_HIPSPARSE_ERROR(hipsparseCreateIdentityPermutation(handle, nnz, info->P));

    // Sort CSR columns
    RETURN_IF_HIPSPARSE_ERROR(
        hipsparseXcsrsort(handle, m, n, nnz, descrA, csrRowPtr, csrColInd, info->P, pBuffer));

    // Sort CSR values
    RETURN_IF_HIPSPARSE_ERROR(
        hipsparseDgthr(handle, nnz, csrVal, (double*)pBuffer, info->P, HIPSPARSE_INDEX_BASE_ZERO));

    // Get stream
    hipStream_t stream;
    RETURN_IF_HIPSPARSE_ERROR(hipsparseGetStream(handle, &stream));

    // Copy sorted values back to csrVal
    RETURN_IF_HIP_ERROR(
        hipMemcpyAsync(csrVal, pBuffer, sizeof(double) * nnz, hipMemcpyDeviceToDevice, stream));

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseCcsru2csr(hipsparseHandle_t         handle,
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

    // De-allocate permutation array, if already allocated but sizes do not match
    if(info->P != nullptr && info->size != nnz)
    {
        RETURN_IF_HIP_ERROR(hipFree(info->P));
        info->size = 0;
    }

    // Allocate memory inside info structure to keep track of the permutation
    // if it has not yet been allocated with matching size
    if(info->P == nullptr)
    {
        // size must be 0
        assert(info->size == 0);

        RETURN_IF_HIP_ERROR(hipMalloc((void**)&info->P, sizeof(int) * nnz));

        info->size = nnz;
    }

    // Initialize permutation with identity
    RETURN_IF_HIPSPARSE_ERROR(hipsparseCreateIdentityPermutation(handle, nnz, info->P));

    // Sort CSR columns
    RETURN_IF_HIPSPARSE_ERROR(
        hipsparseXcsrsort(handle, m, n, nnz, descrA, csrRowPtr, csrColInd, info->P, pBuffer));

    // Sort CSR values
    RETURN_IF_HIPSPARSE_ERROR(hipsparseCgthr(
        handle, nnz, csrVal, (hipComplex*)pBuffer, info->P, HIPSPARSE_INDEX_BASE_ZERO));

    // Get stream
    hipStream_t stream;
    RETURN_IF_HIPSPARSE_ERROR(hipsparseGetStream(handle, &stream));

    // Copy sorted values back to csrVal
    RETURN_IF_HIP_ERROR(
        hipMemcpyAsync(csrVal, pBuffer, sizeof(hipComplex) * nnz, hipMemcpyDeviceToDevice, stream));

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseZcsru2csr(hipsparseHandle_t         handle,
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

    // De-allocate permutation array, if already allocated but sizes do not match
    if(info->P != nullptr && info->size != nnz)
    {
        RETURN_IF_HIP_ERROR(hipFree(info->P));
        info->size = 0;
    }

    // Allocate memory inside info structure to keep track of the permutation
    // if it has not yet been allocated with matching size
    if(info->P == nullptr)
    {
        // size must be 0
        assert(info->size == 0);

        RETURN_IF_HIP_ERROR(hipMalloc((void**)&info->P, sizeof(int) * nnz));

        info->size = nnz;
    }

    // Initialize permutation with identity
    RETURN_IF_HIPSPARSE_ERROR(hipsparseCreateIdentityPermutation(handle, nnz, info->P));

    // Sort CSR columns
    RETURN_IF_HIPSPARSE_ERROR(
        hipsparseXcsrsort(handle, m, n, nnz, descrA, csrRowPtr, csrColInd, info->P, pBuffer));

    // Sort CSR values
    RETURN_IF_HIPSPARSE_ERROR(hipsparseZgthr(
        handle, nnz, csrVal, (hipDoubleComplex*)pBuffer, info->P, HIPSPARSE_INDEX_BASE_ZERO));

    // Get stream
    hipStream_t stream;
    RETURN_IF_HIPSPARSE_ERROR(hipsparseGetStream(handle, &stream));

    // Copy sorted values back to csrVal
    RETURN_IF_HIP_ERROR(hipMemcpyAsync(
        csrVal, pBuffer, sizeof(hipDoubleComplex) * nnz, hipMemcpyDeviceToDevice, stream));

    return HIPSPARSE_STATUS_SUCCESS;
}
