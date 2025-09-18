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

hipsparseStatus_t hipsparseXcsrgemmNnz(hipsparseHandle_t         handle,
                                       hipsparseOperation_t      transA,
                                       hipsparseOperation_t      transB,
                                       int                       m,
                                       int                       n,
                                       int                       k,
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
    // Create matrix info
    rocsparse_mat_info info;
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_create_mat_info(&info));

    // Buffer
    size_t buffer_size;
    void*  temp_buffer;

    // Initialize alpha = 1.0
    hipDoubleComplex  one = make_hipDoubleComplex(1.0, 0.0);
    hipDoubleComplex* alpha;

    hipsparseStatus_t status;

    // Get pointer mode
    rocsparse_pointer_mode pointer_mode;
    status = hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_get_pointer_mode((rocsparse_handle)handle, &pointer_mode));

    if(status != HIPSPARSE_STATUS_SUCCESS)
    {
        rocsparse_destroy_mat_info(info);

        return status;
    }

    if(pointer_mode == rocsparse_pointer_mode_host)
    {
        alpha  = (hipDoubleComplex*)malloc(sizeof(hipDoubleComplex));
        *alpha = one;
    }
    else
    {
        RETURN_IF_HIP_ERROR(hipMalloc((void**)&alpha, sizeof(hipDoubleComplex)));
        RETURN_IF_HIP_ERROR(
            hipMemcpy(alpha, &one, sizeof(hipDoubleComplex), hipMemcpyHostToDevice));
    }

    // Obtain temporary buffer size
    status = hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_zcsrgemm_buffer_size((rocsparse_handle)handle,
                                       hipsparse::hipOperationToHCCOperation(transA),
                                       hipsparse::hipOperationToHCCOperation(transB),
                                       m,
                                       n,
                                       k,
                                       (const rocsparse_double_complex*)alpha,
                                       (rocsparse_mat_descr)descrA,
                                       nnzA,
                                       csrRowPtrA,
                                       csrColIndA,
                                       (rocsparse_mat_descr)descrB,
                                       nnzB,
                                       csrRowPtrB,
                                       csrColIndB,
                                       nullptr,
                                       nullptr,
                                       0,
                                       nullptr,
                                       nullptr,
                                       info,
                                       &buffer_size));

    if(status != HIPSPARSE_STATUS_SUCCESS)
    {
        if(pointer_mode == rocsparse_pointer_mode_host)
        {
            free(alpha);
        }
        else
        {
            RETURN_IF_HIP_ERROR(hipFree(alpha));
        }

        rocsparse_destroy_mat_info(info);

        return status;
    }

    RETURN_IF_HIP_ERROR(hipMalloc(&temp_buffer, buffer_size));

    // Determine nnz
    status = hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_csrgemm_nnz((rocsparse_handle)handle,
                              hipsparse::hipOperationToHCCOperation(transA),
                              hipsparse::hipOperationToHCCOperation(transB),
                              m,
                              n,
                              k,
                              (rocsparse_mat_descr)descrA,
                              nnzA,
                              csrRowPtrA,
                              csrColIndA,
                              (rocsparse_mat_descr)descrB,
                              nnzB,
                              csrRowPtrB,
                              csrColIndB,
                              nullptr,
                              0,
                              nullptr,
                              nullptr,
                              (rocsparse_mat_descr)descrC,
                              csrRowPtrC,
                              nnzTotalDevHostPtr,
                              info,
                              temp_buffer));

    if(pointer_mode == rocsparse_pointer_mode_host)
    {
        free(alpha);
    }
    else
    {
        RETURN_IF_HIP_ERROR(hipFree(alpha));
    }

    RETURN_IF_HIP_ERROR(hipFree(temp_buffer));

    if(status != HIPSPARSE_STATUS_SUCCESS)
    {
        rocsparse_destroy_mat_info(info);

        return status;
    }

    RETURN_IF_ROCSPARSE_ERROR(rocsparse_destroy_mat_info(info));

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseScsrgemm(hipsparseHandle_t         handle,
                                    hipsparseOperation_t      transA,
                                    hipsparseOperation_t      transB,
                                    int                       m,
                                    int                       n,
                                    int                       k,
                                    const hipsparseMatDescr_t descrA,
                                    int                       nnzA,
                                    const float*              csrValA,
                                    const int*                csrRowPtrA,
                                    const int*                csrColIndA,
                                    const hipsparseMatDescr_t descrB,
                                    int                       nnzB,
                                    const float*              csrValB,
                                    const int*                csrRowPtrB,
                                    const int*                csrColIndB,
                                    const hipsparseMatDescr_t descrC,
                                    float*                    csrValC,
                                    const int*                csrRowPtrC,
                                    int*                      csrColIndC)
{
    // Create matrix info
    rocsparse_mat_info info;
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_create_mat_info(&info));

    // Buffer
    size_t buffer_size;
    void*  temp_buffer;

    // Initialize alpha = 1.0
    float  one = 1.0f;
    float* alpha;

    hipsparseStatus_t status;

    // Get pointer mode
    rocsparse_pointer_mode pointer_mode;
    status = hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_get_pointer_mode((rocsparse_handle)handle, &pointer_mode));

    if(status != HIPSPARSE_STATUS_SUCCESS)
    {
        rocsparse_destroy_mat_info(info);

        return status;
    }

    if(pointer_mode == rocsparse_pointer_mode_host)
    {
        alpha  = (float*)malloc(sizeof(float));
        *alpha = one;
    }
    else
    {
        RETURN_IF_HIP_ERROR(hipMalloc((void**)&alpha, sizeof(float)));
        RETURN_IF_HIP_ERROR(hipMemcpy(alpha, &one, sizeof(float), hipMemcpyHostToDevice));
    }

    // Obtain temporary buffer size
    status = hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_scsrgemm_buffer_size((rocsparse_handle)handle,
                                       hipsparse::hipOperationToHCCOperation(transA),
                                       hipsparse::hipOperationToHCCOperation(transB),
                                       m,
                                       n,
                                       k,
                                       alpha,
                                       (rocsparse_mat_descr)descrA,
                                       nnzA,
                                       csrRowPtrA,
                                       csrColIndA,
                                       (rocsparse_mat_descr)descrB,
                                       nnzB,
                                       csrRowPtrB,
                                       csrColIndB,
                                       nullptr,
                                       nullptr,
                                       0,
                                       nullptr,
                                       nullptr,
                                       info,
                                       &buffer_size));

    if(status != HIPSPARSE_STATUS_SUCCESS)
    {
        if(pointer_mode == rocsparse_pointer_mode_host)
        {
            free(alpha);
        }
        else
        {
            RETURN_IF_HIP_ERROR(hipFree(alpha));
        }

        rocsparse_destroy_mat_info(info);

        return status;
    }

    RETURN_IF_HIP_ERROR(hipMalloc(&temp_buffer, buffer_size));

    // Perform csrgemm computation
    status = hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_scsrgemm((rocsparse_handle)handle,
                           hipsparse::hipOperationToHCCOperation(transA),
                           hipsparse::hipOperationToHCCOperation(transB),
                           m,
                           n,
                           k,
                           alpha,
                           (rocsparse_mat_descr)descrA,
                           nnzA,
                           csrValA,
                           csrRowPtrA,
                           csrColIndA,
                           (rocsparse_mat_descr)descrB,
                           nnzB,
                           csrValB,
                           csrRowPtrB,
                           csrColIndB,
                           nullptr,
                           nullptr,
                           0,
                           nullptr,
                           nullptr,
                           nullptr,
                           (rocsparse_mat_descr)descrC,
                           csrValC,
                           csrRowPtrC,
                           csrColIndC,
                           info,
                           temp_buffer));

    if(pointer_mode == rocsparse_pointer_mode_host)
    {
        free(alpha);
    }
    else
    {
        RETURN_IF_HIP_ERROR(hipFree(alpha));
    }

    RETURN_IF_HIP_ERROR(hipFree(temp_buffer));

    if(status != HIPSPARSE_STATUS_SUCCESS)
    {
        rocsparse_destroy_mat_info(info);

        return status;
    }

    RETURN_IF_ROCSPARSE_ERROR(rocsparse_destroy_mat_info(info));

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseDcsrgemm(hipsparseHandle_t         handle,
                                    hipsparseOperation_t      transA,
                                    hipsparseOperation_t      transB,
                                    int                       m,
                                    int                       n,
                                    int                       k,
                                    const hipsparseMatDescr_t descrA,
                                    int                       nnzA,
                                    const double*             csrValA,
                                    const int*                csrRowPtrA,
                                    const int*                csrColIndA,
                                    const hipsparseMatDescr_t descrB,
                                    int                       nnzB,
                                    const double*             csrValB,
                                    const int*                csrRowPtrB,
                                    const int*                csrColIndB,
                                    const hipsparseMatDescr_t descrC,
                                    double*                   csrValC,
                                    const int*                csrRowPtrC,
                                    int*                      csrColIndC)
{
    // Create matrix info
    rocsparse_mat_info info;
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_create_mat_info(&info));

    // Buffer
    size_t buffer_size;
    void*  temp_buffer;

    // Initialize alpha = 1.0
    double  one = 1.0;
    double* alpha;

    hipsparseStatus_t status;

    // Get pointer mode
    rocsparse_pointer_mode pointer_mode;
    status = hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_get_pointer_mode((rocsparse_handle)handle, &pointer_mode));

    if(status != HIPSPARSE_STATUS_SUCCESS)
    {
        rocsparse_destroy_mat_info(info);

        return status;
    }

    if(pointer_mode == rocsparse_pointer_mode_host)
    {
        alpha  = (double*)malloc(sizeof(double));
        *alpha = one;
    }
    else
    {
        RETURN_IF_HIP_ERROR(hipMalloc((void**)&alpha, sizeof(double)));
        RETURN_IF_HIP_ERROR(hipMemcpy(alpha, &one, sizeof(double), hipMemcpyHostToDevice));
    }

    // Obtain temporary buffer size
    status = hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_dcsrgemm_buffer_size((rocsparse_handle)handle,
                                       hipsparse::hipOperationToHCCOperation(transA),
                                       hipsparse::hipOperationToHCCOperation(transB),
                                       m,
                                       n,
                                       k,
                                       alpha,
                                       (rocsparse_mat_descr)descrA,
                                       nnzA,
                                       csrRowPtrA,
                                       csrColIndA,
                                       (rocsparse_mat_descr)descrB,
                                       nnzB,
                                       csrRowPtrB,
                                       csrColIndB,
                                       nullptr,
                                       nullptr,
                                       0,
                                       nullptr,
                                       nullptr,
                                       info,
                                       &buffer_size));

    if(status != HIPSPARSE_STATUS_SUCCESS)
    {
        if(pointer_mode == rocsparse_pointer_mode_host)
        {
            free(alpha);
        }
        else
        {
            RETURN_IF_HIP_ERROR(hipFree(alpha));
        }

        rocsparse_destroy_mat_info(info);

        return status;
    }

    RETURN_IF_HIP_ERROR(hipMalloc(&temp_buffer, buffer_size));

    // Perform csrgemm computation
    status = hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_dcsrgemm((rocsparse_handle)handle,
                           hipsparse::hipOperationToHCCOperation(transA),
                           hipsparse::hipOperationToHCCOperation(transB),
                           m,
                           n,
                           k,
                           alpha,
                           (rocsparse_mat_descr)descrA,
                           nnzA,
                           csrValA,
                           csrRowPtrA,
                           csrColIndA,
                           (rocsparse_mat_descr)descrB,
                           nnzB,
                           csrValB,
                           csrRowPtrB,
                           csrColIndB,
                           nullptr,
                           nullptr,
                           0,
                           nullptr,
                           nullptr,
                           nullptr,
                           (rocsparse_mat_descr)descrC,
                           csrValC,
                           csrRowPtrC,
                           csrColIndC,
                           info,
                           temp_buffer));

    if(pointer_mode == rocsparse_pointer_mode_host)
    {
        free(alpha);
    }
    else
    {
        RETURN_IF_HIP_ERROR(hipFree(alpha));
    }

    RETURN_IF_HIP_ERROR(hipFree(temp_buffer));

    if(status != HIPSPARSE_STATUS_SUCCESS)
    {
        rocsparse_destroy_mat_info(info);

        return status;
    }

    RETURN_IF_ROCSPARSE_ERROR(rocsparse_destroy_mat_info(info));

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseCcsrgemm(hipsparseHandle_t         handle,
                                    hipsparseOperation_t      transA,
                                    hipsparseOperation_t      transB,
                                    int                       m,
                                    int                       n,
                                    int                       k,
                                    const hipsparseMatDescr_t descrA,
                                    int                       nnzA,
                                    const hipComplex*         csrValA,
                                    const int*                csrRowPtrA,
                                    const int*                csrColIndA,
                                    const hipsparseMatDescr_t descrB,
                                    int                       nnzB,
                                    const hipComplex*         csrValB,
                                    const int*                csrRowPtrB,
                                    const int*                csrColIndB,
                                    const hipsparseMatDescr_t descrC,
                                    hipComplex*               csrValC,
                                    const int*                csrRowPtrC,
                                    int*                      csrColIndC)
{
    // Create matrix info
    rocsparse_mat_info info;
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_create_mat_info(&info));

    // Buffer
    size_t buffer_size;
    void*  temp_buffer;

    // Initialize alpha = 1.0
    hipComplex  one = make_hipComplex(1.0f, 0.0f);
    hipComplex* alpha;

    hipsparseStatus_t status;

    // Get pointer mode
    rocsparse_pointer_mode pointer_mode;
    status = hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_get_pointer_mode((rocsparse_handle)handle, &pointer_mode));

    if(status != HIPSPARSE_STATUS_SUCCESS)
    {
        rocsparse_destroy_mat_info(info);

        return status;
    }

    if(pointer_mode == rocsparse_pointer_mode_host)
    {
        alpha  = (hipComplex*)malloc(sizeof(hipComplex));
        *alpha = one;
    }
    else
    {
        RETURN_IF_HIP_ERROR(hipMalloc((void**)&alpha, sizeof(hipComplex)));
        RETURN_IF_HIP_ERROR(hipMemcpy(alpha, &one, sizeof(hipComplex), hipMemcpyHostToDevice));
    }

    // Obtain temporary buffer size
    status = hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_ccsrgemm_buffer_size((rocsparse_handle)handle,
                                       hipsparse::hipOperationToHCCOperation(transA),
                                       hipsparse::hipOperationToHCCOperation(transB),
                                       m,
                                       n,
                                       k,
                                       (const rocsparse_float_complex*)alpha,
                                       (rocsparse_mat_descr)descrA,
                                       nnzA,
                                       csrRowPtrA,
                                       csrColIndA,
                                       (rocsparse_mat_descr)descrB,
                                       nnzB,
                                       csrRowPtrB,
                                       csrColIndB,
                                       nullptr,
                                       nullptr,
                                       0,
                                       nullptr,
                                       nullptr,
                                       info,
                                       &buffer_size));

    if(status != HIPSPARSE_STATUS_SUCCESS)
    {
        if(pointer_mode == rocsparse_pointer_mode_host)
        {
            free(alpha);
        }
        else
        {
            RETURN_IF_HIP_ERROR(hipFree(alpha));
        }

        rocsparse_destroy_mat_info(info);

        return status;
    }

    RETURN_IF_HIP_ERROR(hipMalloc(&temp_buffer, buffer_size));

    // Perform csrgemm computation
    status = hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_ccsrgemm((rocsparse_handle)handle,
                           hipsparse::hipOperationToHCCOperation(transA),
                           hipsparse::hipOperationToHCCOperation(transB),
                           m,
                           n,
                           k,
                           (const rocsparse_float_complex*)alpha,
                           (rocsparse_mat_descr)descrA,
                           nnzA,
                           (const rocsparse_float_complex*)csrValA,
                           csrRowPtrA,
                           csrColIndA,
                           (rocsparse_mat_descr)descrB,
                           nnzB,
                           (const rocsparse_float_complex*)csrValB,
                           csrRowPtrB,
                           csrColIndB,
                           nullptr,
                           nullptr,
                           0,
                           nullptr,
                           nullptr,
                           nullptr,
                           (rocsparse_mat_descr)descrC,
                           (rocsparse_float_complex*)csrValC,
                           csrRowPtrC,
                           csrColIndC,
                           info,
                           temp_buffer));

    if(pointer_mode == rocsparse_pointer_mode_host)
    {
        free(alpha);
    }
    else
    {
        RETURN_IF_HIP_ERROR(hipFree(alpha));
    }

    RETURN_IF_HIP_ERROR(hipFree(temp_buffer));

    if(status != HIPSPARSE_STATUS_SUCCESS)
    {
        rocsparse_destroy_mat_info(info);

        return status;
    }

    RETURN_IF_ROCSPARSE_ERROR(rocsparse_destroy_mat_info(info));

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseZcsrgemm(hipsparseHandle_t         handle,
                                    hipsparseOperation_t      transA,
                                    hipsparseOperation_t      transB,
                                    int                       m,
                                    int                       n,
                                    int                       k,
                                    const hipsparseMatDescr_t descrA,
                                    int                       nnzA,
                                    const hipDoubleComplex*   csrValA,
                                    const int*                csrRowPtrA,
                                    const int*                csrColIndA,
                                    const hipsparseMatDescr_t descrB,
                                    int                       nnzB,
                                    const hipDoubleComplex*   csrValB,
                                    const int*                csrRowPtrB,
                                    const int*                csrColIndB,
                                    const hipsparseMatDescr_t descrC,
                                    hipDoubleComplex*         csrValC,
                                    const int*                csrRowPtrC,
                                    int*                      csrColIndC)
{
    // Create matrix info
    rocsparse_mat_info info;
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_create_mat_info(&info));

    // Buffer
    size_t buffer_size;
    void*  temp_buffer;

    // Initialize alpha = 1.0
    hipDoubleComplex  one = make_hipDoubleComplex(1.0, 0.0);
    hipDoubleComplex* alpha;

    hipsparseStatus_t status;

    // Get pointer mode
    rocsparse_pointer_mode pointer_mode;
    status = hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_get_pointer_mode((rocsparse_handle)handle, &pointer_mode));

    if(status != HIPSPARSE_STATUS_SUCCESS)
    {
        rocsparse_destroy_mat_info(info);

        return status;
    }

    if(pointer_mode == rocsparse_pointer_mode_host)
    {
        alpha  = (hipDoubleComplex*)malloc(sizeof(hipDoubleComplex));
        *alpha = one;
    }
    else
    {
        RETURN_IF_HIP_ERROR(hipMalloc((void**)&alpha, sizeof(hipDoubleComplex)));
        RETURN_IF_HIP_ERROR(
            hipMemcpy(alpha, &one, sizeof(hipDoubleComplex), hipMemcpyHostToDevice));
    }

    // Obtain temporary buffer size
    status = hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_zcsrgemm_buffer_size((rocsparse_handle)handle,
                                       hipsparse::hipOperationToHCCOperation(transA),
                                       hipsparse::hipOperationToHCCOperation(transB),
                                       m,
                                       n,
                                       k,
                                       (const rocsparse_double_complex*)alpha,
                                       (rocsparse_mat_descr)descrA,
                                       nnzA,
                                       csrRowPtrA,
                                       csrColIndA,
                                       (rocsparse_mat_descr)descrB,
                                       nnzB,
                                       csrRowPtrB,
                                       csrColIndB,
                                       nullptr,
                                       nullptr,
                                       0,
                                       nullptr,
                                       nullptr,
                                       info,
                                       &buffer_size));

    if(status != HIPSPARSE_STATUS_SUCCESS)
    {
        if(pointer_mode == rocsparse_pointer_mode_host)
        {
            free(alpha);
        }
        else
        {
            RETURN_IF_HIP_ERROR(hipFree(alpha));
        }

        rocsparse_destroy_mat_info(info);

        return status;
    }

    RETURN_IF_HIP_ERROR(hipMalloc(&temp_buffer, buffer_size));

    // Perform csrgemm computation
    status = hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_zcsrgemm((rocsparse_handle)handle,
                           hipsparse::hipOperationToHCCOperation(transA),
                           hipsparse::hipOperationToHCCOperation(transB),
                           m,
                           n,
                           k,
                           (const rocsparse_double_complex*)alpha,
                           (rocsparse_mat_descr)descrA,
                           nnzA,
                           (const rocsparse_double_complex*)csrValA,
                           csrRowPtrA,
                           csrColIndA,
                           (rocsparse_mat_descr)descrB,
                           nnzB,
                           (const rocsparse_double_complex*)csrValB,
                           csrRowPtrB,
                           csrColIndB,
                           nullptr,
                           nullptr,
                           0,
                           nullptr,
                           nullptr,
                           nullptr,
                           (rocsparse_mat_descr)descrC,
                           (rocsparse_double_complex*)csrValC,
                           csrRowPtrC,
                           csrColIndC,
                           info,
                           temp_buffer));

    if(pointer_mode == rocsparse_pointer_mode_host)
    {
        free(alpha);
    }
    else
    {
        RETURN_IF_HIP_ERROR(hipFree(alpha));
    }

    RETURN_IF_HIP_ERROR(hipFree(temp_buffer));

    if(status != HIPSPARSE_STATUS_SUCCESS)
    {
        rocsparse_destroy_mat_info(info);

        return status;
    }

    RETURN_IF_ROCSPARSE_ERROR(rocsparse_destroy_mat_info(info));

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseScsrgemm2_bufferSizeExt(hipsparseHandle_t         handle,
                                                   int                       m,
                                                   int                       n,
                                                   int                       k,
                                                   const float*              alpha,
                                                   const hipsparseMatDescr_t descrA,
                                                   int                       nnzA,
                                                   const int*                csrRowPtrA,
                                                   const int*                csrColIndA,
                                                   const hipsparseMatDescr_t descrB,
                                                   int                       nnzB,
                                                   const int*                csrRowPtrB,
                                                   const int*                csrColIndB,
                                                   const float*              beta,
                                                   const hipsparseMatDescr_t descrD,
                                                   int                       nnzD,
                                                   const int*                csrRowPtrD,
                                                   const int*                csrColIndD,
                                                   csrgemm2Info_t            info,
                                                   size_t*                   pBufferSizeInBytes)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_scsrgemm_buffer_size((rocsparse_handle)handle,
                                       rocsparse_operation_none,
                                       rocsparse_operation_none,
                                       m,
                                       n,
                                       k,
                                       alpha,
                                       (rocsparse_mat_descr)descrA,
                                       nnzA,
                                       csrRowPtrA,
                                       csrColIndA,
                                       (rocsparse_mat_descr)descrB,
                                       nnzB,
                                       csrRowPtrB,
                                       csrColIndB,
                                       beta,
                                       (rocsparse_mat_descr)descrD,
                                       nnzD,
                                       csrRowPtrD,
                                       csrColIndD,
                                       (rocsparse_mat_info)info,
                                       pBufferSizeInBytes));
}

hipsparseStatus_t hipsparseDcsrgemm2_bufferSizeExt(hipsparseHandle_t         handle,
                                                   int                       m,
                                                   int                       n,
                                                   int                       k,
                                                   const double*             alpha,
                                                   const hipsparseMatDescr_t descrA,
                                                   int                       nnzA,
                                                   const int*                csrRowPtrA,
                                                   const int*                csrColIndA,
                                                   const hipsparseMatDescr_t descrB,
                                                   int                       nnzB,
                                                   const int*                csrRowPtrB,
                                                   const int*                csrColIndB,
                                                   const double*             beta,
                                                   const hipsparseMatDescr_t descrD,
                                                   int                       nnzD,
                                                   const int*                csrRowPtrD,
                                                   const int*                csrColIndD,
                                                   csrgemm2Info_t            info,
                                                   size_t*                   pBufferSizeInBytes)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_dcsrgemm_buffer_size((rocsparse_handle)handle,
                                       rocsparse_operation_none,
                                       rocsparse_operation_none,
                                       m,
                                       n,
                                       k,
                                       alpha,
                                       (rocsparse_mat_descr)descrA,
                                       nnzA,
                                       csrRowPtrA,
                                       csrColIndA,
                                       (rocsparse_mat_descr)descrB,
                                       nnzB,
                                       csrRowPtrB,
                                       csrColIndB,
                                       beta,
                                       (rocsparse_mat_descr)descrD,
                                       nnzD,
                                       csrRowPtrD,
                                       csrColIndD,
                                       (rocsparse_mat_info)info,
                                       pBufferSizeInBytes));
}

hipsparseStatus_t hipsparseCcsrgemm2_bufferSizeExt(hipsparseHandle_t         handle,
                                                   int                       m,
                                                   int                       n,
                                                   int                       k,
                                                   const hipComplex*         alpha,
                                                   const hipsparseMatDescr_t descrA,
                                                   int                       nnzA,
                                                   const int*                csrRowPtrA,
                                                   const int*                csrColIndA,
                                                   const hipsparseMatDescr_t descrB,
                                                   int                       nnzB,
                                                   const int*                csrRowPtrB,
                                                   const int*                csrColIndB,
                                                   const hipComplex*         beta,
                                                   const hipsparseMatDescr_t descrD,
                                                   int                       nnzD,
                                                   const int*                csrRowPtrD,
                                                   const int*                csrColIndD,
                                                   csrgemm2Info_t            info,
                                                   size_t*                   pBufferSizeInBytes)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_ccsrgemm_buffer_size((rocsparse_handle)handle,
                                       rocsparse_operation_none,
                                       rocsparse_operation_none,
                                       m,
                                       n,
                                       k,
                                       (const rocsparse_float_complex*)alpha,
                                       (rocsparse_mat_descr)descrA,
                                       nnzA,
                                       csrRowPtrA,
                                       csrColIndA,
                                       (rocsparse_mat_descr)descrB,
                                       nnzB,
                                       csrRowPtrB,
                                       csrColIndB,
                                       (const rocsparse_float_complex*)beta,
                                       (rocsparse_mat_descr)descrD,
                                       nnzD,
                                       csrRowPtrD,
                                       csrColIndD,
                                       (rocsparse_mat_info)info,
                                       pBufferSizeInBytes));
}

hipsparseStatus_t hipsparseZcsrgemm2_bufferSizeExt(hipsparseHandle_t         handle,
                                                   int                       m,
                                                   int                       n,
                                                   int                       k,
                                                   const hipDoubleComplex*   alpha,
                                                   const hipsparseMatDescr_t descrA,
                                                   int                       nnzA,
                                                   const int*                csrRowPtrA,
                                                   const int*                csrColIndA,
                                                   const hipsparseMatDescr_t descrB,
                                                   int                       nnzB,
                                                   const int*                csrRowPtrB,
                                                   const int*                csrColIndB,
                                                   const hipDoubleComplex*   beta,
                                                   const hipsparseMatDescr_t descrD,
                                                   int                       nnzD,
                                                   const int*                csrRowPtrD,
                                                   const int*                csrColIndD,
                                                   csrgemm2Info_t            info,
                                                   size_t*                   pBufferSizeInBytes)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_zcsrgemm_buffer_size((rocsparse_handle)handle,
                                       rocsparse_operation_none,
                                       rocsparse_operation_none,
                                       m,
                                       n,
                                       k,
                                       (const rocsparse_double_complex*)alpha,
                                       (rocsparse_mat_descr)descrA,
                                       nnzA,
                                       csrRowPtrA,
                                       csrColIndA,
                                       (rocsparse_mat_descr)descrB,
                                       nnzB,
                                       csrRowPtrB,
                                       csrColIndB,
                                       (const rocsparse_double_complex*)beta,
                                       (rocsparse_mat_descr)descrD,
                                       nnzD,
                                       csrRowPtrD,
                                       csrColIndD,
                                       (rocsparse_mat_info)info,
                                       pBufferSizeInBytes));
}

hipsparseStatus_t hipsparseXcsrgemm2Nnz(hipsparseHandle_t         handle,
                                        int                       m,
                                        int                       n,
                                        int                       k,
                                        const hipsparseMatDescr_t descrA,
                                        int                       nnzA,
                                        const int*                csrRowPtrA,
                                        const int*                csrColIndA,
                                        const hipsparseMatDescr_t descrB,
                                        int                       nnzB,
                                        const int*                csrRowPtrB,
                                        const int*                csrColIndB,
                                        const hipsparseMatDescr_t descrD,
                                        int                       nnzD,
                                        const int*                csrRowPtrD,
                                        const int*                csrColIndD,
                                        const hipsparseMatDescr_t descrC,
                                        int*                      csrRowPtrC,
                                        int*                      nnzTotalDevHostPtr,
                                        const csrgemm2Info_t      info,
                                        void*                     pBuffer)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(rocsparse_csrgemm_nnz((rocsparse_handle)handle,
                                                                       rocsparse_operation_none,
                                                                       rocsparse_operation_none,
                                                                       m,
                                                                       n,
                                                                       k,
                                                                       (rocsparse_mat_descr)descrA,
                                                                       nnzA,
                                                                       csrRowPtrA,
                                                                       csrColIndA,
                                                                       (rocsparse_mat_descr)descrB,
                                                                       nnzB,
                                                                       csrRowPtrB,
                                                                       csrColIndB,
                                                                       (rocsparse_mat_descr)descrD,
                                                                       nnzD,
                                                                       csrRowPtrD,
                                                                       csrColIndD,
                                                                       (rocsparse_mat_descr)descrC,
                                                                       csrRowPtrC,
                                                                       nnzTotalDevHostPtr,
                                                                       (rocsparse_mat_info)info,
                                                                       pBuffer));
}

hipsparseStatus_t hipsparseScsrgemm2(hipsparseHandle_t         handle,
                                     int                       m,
                                     int                       n,
                                     int                       k,
                                     const float*              alpha,
                                     const hipsparseMatDescr_t descrA,
                                     int                       nnzA,
                                     const float*              csrValA,
                                     const int*                csrRowPtrA,
                                     const int*                csrColIndA,
                                     const hipsparseMatDescr_t descrB,
                                     int                       nnzB,
                                     const float*              csrValB,
                                     const int*                csrRowPtrB,
                                     const int*                csrColIndB,
                                     const float*              beta,
                                     const hipsparseMatDescr_t descrD,
                                     int                       nnzD,
                                     const float*              csrValD,
                                     const int*                csrRowPtrD,
                                     const int*                csrColIndD,
                                     const hipsparseMatDescr_t descrC,
                                     float*                    csrValC,
                                     const int*                csrRowPtrC,
                                     int*                      csrColIndC,
                                     const csrgemm2Info_t      info,
                                     void*                     pBuffer)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(rocsparse_scsrgemm((rocsparse_handle)handle,
                                                                    rocsparse_operation_none,
                                                                    rocsparse_operation_none,
                                                                    m,
                                                                    n,
                                                                    k,
                                                                    alpha,
                                                                    (rocsparse_mat_descr)descrA,
                                                                    nnzA,
                                                                    csrValA,
                                                                    csrRowPtrA,
                                                                    csrColIndA,
                                                                    (rocsparse_mat_descr)descrB,
                                                                    nnzB,
                                                                    csrValB,
                                                                    csrRowPtrB,
                                                                    csrColIndB,
                                                                    beta,
                                                                    (rocsparse_mat_descr)descrD,
                                                                    nnzD,
                                                                    csrValD,
                                                                    csrRowPtrD,
                                                                    csrColIndD,
                                                                    (rocsparse_mat_descr)descrC,
                                                                    csrValC,
                                                                    csrRowPtrC,
                                                                    csrColIndC,
                                                                    (rocsparse_mat_info)info,
                                                                    pBuffer));
}

hipsparseStatus_t hipsparseDcsrgemm2(hipsparseHandle_t         handle,
                                     int                       m,
                                     int                       n,
                                     int                       k,
                                     const double*             alpha,
                                     const hipsparseMatDescr_t descrA,
                                     int                       nnzA,
                                     const double*             csrValA,
                                     const int*                csrRowPtrA,
                                     const int*                csrColIndA,
                                     const hipsparseMatDescr_t descrB,
                                     int                       nnzB,
                                     const double*             csrValB,
                                     const int*                csrRowPtrB,
                                     const int*                csrColIndB,
                                     const double*             beta,
                                     const hipsparseMatDescr_t descrD,
                                     int                       nnzD,
                                     const double*             csrValD,
                                     const int*                csrRowPtrD,
                                     const int*                csrColIndD,
                                     const hipsparseMatDescr_t descrC,
                                     double*                   csrValC,
                                     const int*                csrRowPtrC,
                                     int*                      csrColIndC,
                                     const csrgemm2Info_t      info,
                                     void*                     pBuffer)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(rocsparse_dcsrgemm((rocsparse_handle)handle,
                                                                    rocsparse_operation_none,
                                                                    rocsparse_operation_none,
                                                                    m,
                                                                    n,
                                                                    k,
                                                                    alpha,
                                                                    (rocsparse_mat_descr)descrA,
                                                                    nnzA,
                                                                    csrValA,
                                                                    csrRowPtrA,
                                                                    csrColIndA,
                                                                    (rocsparse_mat_descr)descrB,
                                                                    nnzB,
                                                                    csrValB,
                                                                    csrRowPtrB,
                                                                    csrColIndB,
                                                                    beta,
                                                                    (rocsparse_mat_descr)descrD,
                                                                    nnzD,
                                                                    csrValD,
                                                                    csrRowPtrD,
                                                                    csrColIndD,
                                                                    (rocsparse_mat_descr)descrC,
                                                                    csrValC,
                                                                    csrRowPtrC,
                                                                    csrColIndC,
                                                                    (rocsparse_mat_info)info,
                                                                    pBuffer));
}

hipsparseStatus_t hipsparseCcsrgemm2(hipsparseHandle_t         handle,
                                     int                       m,
                                     int                       n,
                                     int                       k,
                                     const hipComplex*         alpha,
                                     const hipsparseMatDescr_t descrA,
                                     int                       nnzA,
                                     const hipComplex*         csrValA,
                                     const int*                csrRowPtrA,
                                     const int*                csrColIndA,
                                     const hipsparseMatDescr_t descrB,
                                     int                       nnzB,
                                     const hipComplex*         csrValB,
                                     const int*                csrRowPtrB,
                                     const int*                csrColIndB,
                                     const hipComplex*         beta,
                                     const hipsparseMatDescr_t descrD,
                                     int                       nnzD,
                                     const hipComplex*         csrValD,
                                     const int*                csrRowPtrD,
                                     const int*                csrColIndD,
                                     const hipsparseMatDescr_t descrC,
                                     hipComplex*               csrValC,
                                     const int*                csrRowPtrC,
                                     int*                      csrColIndC,
                                     const csrgemm2Info_t      info,
                                     void*                     pBuffer)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_ccsrgemm((rocsparse_handle)handle,
                           rocsparse_operation_none,
                           rocsparse_operation_none,
                           m,
                           n,
                           k,
                           (const rocsparse_float_complex*)alpha,
                           (rocsparse_mat_descr)descrA,
                           nnzA,
                           (const rocsparse_float_complex*)csrValA,
                           csrRowPtrA,
                           csrColIndA,
                           (rocsparse_mat_descr)descrB,
                           nnzB,
                           (const rocsparse_float_complex*)csrValB,
                           csrRowPtrB,
                           csrColIndB,
                           (const rocsparse_float_complex*)beta,
                           (rocsparse_mat_descr)descrD,
                           nnzD,
                           (const rocsparse_float_complex*)csrValD,
                           csrRowPtrD,
                           csrColIndD,
                           (rocsparse_mat_descr)descrC,
                           (rocsparse_float_complex*)csrValC,
                           csrRowPtrC,
                           csrColIndC,
                           (rocsparse_mat_info)info,
                           pBuffer));
}

hipsparseStatus_t hipsparseZcsrgemm2(hipsparseHandle_t         handle,
                                     int                       m,
                                     int                       n,
                                     int                       k,
                                     const hipDoubleComplex*   alpha,
                                     const hipsparseMatDescr_t descrA,
                                     int                       nnzA,
                                     const hipDoubleComplex*   csrValA,
                                     const int*                csrRowPtrA,
                                     const int*                csrColIndA,
                                     const hipsparseMatDescr_t descrB,
                                     int                       nnzB,
                                     const hipDoubleComplex*   csrValB,
                                     const int*                csrRowPtrB,
                                     const int*                csrColIndB,
                                     const hipDoubleComplex*   beta,
                                     const hipsparseMatDescr_t descrD,
                                     int                       nnzD,
                                     const hipDoubleComplex*   csrValD,
                                     const int*                csrRowPtrD,
                                     const int*                csrColIndD,
                                     const hipsparseMatDescr_t descrC,
                                     hipDoubleComplex*         csrValC,
                                     const int*                csrRowPtrC,
                                     int*                      csrColIndC,
                                     const csrgemm2Info_t      info,
                                     void*                     pBuffer)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_zcsrgemm((rocsparse_handle)handle,
                           rocsparse_operation_none,
                           rocsparse_operation_none,
                           m,
                           n,
                           k,
                           (const rocsparse_double_complex*)alpha,
                           (rocsparse_mat_descr)descrA,
                           nnzA,
                           (const rocsparse_double_complex*)csrValA,
                           csrRowPtrA,
                           csrColIndA,
                           (rocsparse_mat_descr)descrB,
                           nnzB,
                           (const rocsparse_double_complex*)csrValB,
                           csrRowPtrB,
                           csrColIndB,
                           (const rocsparse_double_complex*)beta,
                           (rocsparse_mat_descr)descrD,
                           nnzD,
                           (const rocsparse_double_complex*)csrValD,
                           csrRowPtrD,
                           csrColIndD,
                           (rocsparse_mat_descr)descrC,
                           (rocsparse_double_complex*)csrValC,
                           csrRowPtrC,
                           csrColIndC,
                           (rocsparse_mat_info)info,
                           pBuffer));
}
