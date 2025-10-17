/* ************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
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

#include "utility.hpp"

#include <hip/hip_runtime_api.h>
#include <hipsparse/hipsparse.h>
#include <stdio.h>
#include <stdlib.h>

#define HIP_CHECK(stat)                                               \
    {                                                                 \
        if(stat != hipSuccess)                                        \
        {                                                             \
            fprintf(stderr, "Error: hip error in line %d", __LINE__); \
            return -1;                                                \
        }                                                             \
    }

#define HIPSPARSE_CHECK(stat)                                               \
    {                                                                       \
        if(stat != HIPSPARSE_STATUS_SUCCESS)                                \
        {                                                                   \
            fprintf(stderr, "Error: hipsparse error in line %d", __LINE__); \
            return -1;                                                      \
        }                                                                   \
    }

//! [doc example]
int main(int argc, char* argv[])
{
    // hipSPARSE handle
    hipsparseHandle_t handle;
    HIPSPARSE_CHECK(hipsparseCreate(&handle));

    // A sample symmetric positive definite matrix A (4x4)
    // with a block size of 1. This example effectively uses BSR format
    // for a CSR-like matrix.
    // Matrix A:
    // ( 4  1  0  0 )
    // ( 1  5  2  0 )
    // ( 0  2  3  1 )
    // ( 0  0  1  2 )

    const int m    = 4; // Number of rows
    const int n    = 4; // Number of columns
    const int bs   = 1; // Block size
    const int mb   = m / bs; // Number of block rows
    const int nb   = n / bs; // Number of block columns
    const int nnzb = 10; // Number of non-zero blocks

    // BSR row pointers
    int hbsrRowPtr[mb + 1] = {0, 2, 5, 8, 10};

    // BSR column indices
    int hbsrColInd[nnzb] = {0, 1, 0, 1, 2, 1, 2, 3, 2, 3};

    // BSR values (single precision float for 'S'bsric02)
    // Values are stored column-major within each block, but with bs=1, this is simple.
    // The values correspond to the upper triangular part of the matrix.
    float hbsrVal[nnzb * bs * bs] = {4.0f, 1.0f, 1.0f, 5.0f, 2.0f, 2.0f, 3.0f, 1.0f, 1.0f, 2.0f};

    // Matrix descriptor
    hipsparseMatDescr_t descr;
    HIPSPARSE_CHECK(hipsparseCreateMatDescr(&descr));

    // Set index base on descriptor
    HIPSPARSE_CHECK(hipsparseSetMatIndexBase(descr, HIPSPARSE_INDEX_BASE_ZERO));

    // Set fill mode to lower and diagonal type to unit (required for IC02)
    HIPSPARSE_CHECK(hipsparseSetMatFillMode(descr, HIPSPARSE_FILL_MODE_LOWER));
    HIPSPARSE_CHECK(hipsparseSetMatDiagType(descr, HIPSPARSE_DIAG_TYPE_UNIT));

    // BSRIC02 info
    bsric02Info_t info;
    HIPSPARSE_CHECK(hipsparseCreateBsric02Info(&info));

    // Offload data to device
    int*   dbsrRowPtr;
    int*   dbsrColInd;
    float* dbsrVal;

    HIP_CHECK(hipMalloc((void**)&dbsrRowPtr, sizeof(int) * (mb + 1)));
    HIP_CHECK(hipMalloc((void**)&dbsrColInd, sizeof(int) * nnzb));
    HIP_CHECK(hipMalloc((void**)&dbsrVal, sizeof(float) * nnzb * bs * bs));

    HIP_CHECK(hipMemcpy(dbsrRowPtr, hbsrRowPtr, sizeof(int) * (mb + 1), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dbsrColInd, hbsrColInd, sizeof(int) * nnzb, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dbsrVal, hbsrVal, sizeof(float) * nnzb * bs * bs, hipMemcpyHostToDevice));

    // 1. Get buffer size
    int bufferSize = 0;
    HIPSPARSE_CHECK(hipsparseSbsric02_bufferSize(handle,
                                                 HIPSPARSE_DIRECTION_COLUMN,
                                                 mb,
                                                 nnzb,
                                                 descr,
                                                 dbsrVal,
                                                 dbsrRowPtr,
                                                 dbsrColInd,
                                                 bs,
                                                 info,
                                                 &bufferSize));

    void* dbuffer = nullptr;
    HIP_CHECK(hipMalloc((void**)&dbuffer, bufferSize));

    // 2. Perform analysis (symbolic factorization)
    HIPSPARSE_CHECK(hipsparseSbsric02_analysis(handle,
                                               HIPSPARSE_DIRECTION_COLUMN,
                                               mb,
                                               nnzb,
                                               descr,
                                               dbsrVal,
                                               dbsrRowPtr,
                                               dbsrColInd,
                                               bs,
                                               info,
                                               HIPSPARSE_SOLVE_POLICY_USE_LEVEL,
                                               dbuffer));

    // 3. Perform factorization (numerical computation)
    HIPSPARSE_CHECK(hipsparseSbsric02(handle,
                                      HIPSPARSE_DIRECTION_COLUMN,
                                      mb,
                                      nnzb,
                                      descr,
                                      dbsrVal,
                                      dbsrRowPtr,
                                      dbsrColInd,
                                      bs,
                                      info,
                                      HIPSPARSE_SOLVE_POLICY_USE_LEVEL,
                                      dbuffer));

    // 4. Check for zero pivots
    int zeroPivot = 0;
    HIPSPARSE_CHECK(hipsparseXbsric02_zeroPivot(handle, info, &zeroPivot));
    if(zeroPivot != -1)
    {
        printf("Error: Zero pivot detected at index %d\n", zeroPivot);
        // Handle error, e.g., by returning an error code
    }

    // Copy the factorized values back to host
    float* hbsrVal_result = new float[nnzb * bs * bs];
    HIP_CHECK(
        hipMemcpy(hbsrVal_result, dbsrVal, sizeof(float) * nnzb * bs * bs, hipMemcpyDeviceToHost));

    // Print the result (the values of the factorized matrix)
    printf("Successfully computed incomplete Cholesky factorization.\n");
    printf("Factorized BSR values:\n");
    for(int i = 0; i < nnzb * bs * bs; ++i)
    {
        printf("val[%d] = %f\n", i, hbsrVal_result[i]);
    }

    // Clean up
    delete[] hbsrVal_result;

    HIPSPARSE_CHECK(hipsparseDestroyBsric02Info(info));
    HIPSPARSE_CHECK(hipsparseDestroyMatDescr(descr));
    HIPSPARSE_CHECK(hipsparseDestroy(handle));

    HIP_CHECK(hipFree(dbsrRowPtr));
    HIP_CHECK(hipFree(dbsrColInd));
    HIP_CHECK(hipFree(dbsrVal));
    HIP_CHECK(hipFree(dbuffer));

    return 0;
}
//! [doc example]