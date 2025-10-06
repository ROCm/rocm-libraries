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

    //     1 2 0 3 0 0
    // A = 0 4 5 0 0 0
    //     0 0 0 7 8 0
    //     0 0 1 2 4 1

    const int                  blockDim = 2;
    const int                  mb       = 2;
    const int                  kb       = 3;
    const int                  nnzb     = 4;
    const hipsparseDirection_t dir      = HIPSPARSE_DIRECTION_ROW;

    int   hbsrRowPtr[mb + 1]                  = {0, 2, 4};
    int   hbsrColInd[nnzb]                    = {0, 1, 1, 2};
    float hbsrVal[nnzb * blockDim * blockDim] = {1, 2, 0, 4, 0, 3, 5, 0, 0, 7, 1, 2, 8, 0, 4, 1};

    // Set dimension n of B
    const int n = 3;
    const int m = mb * blockDim;
    const int k = kb * blockDim;

    // Allocate and generate dense matrix B (k x n)
    float hB[k * n] = {1.0f,
                       2.0f,
                       3.0f,
                       4.0f,
                       5.0f,
                       6.0f,
                       7.0f,
                       8.0f,
                       9.0f,
                       10.0f,
                       11.0f,
                       12.0f,
                       13.0f,
                       14.0f,
                       15.0f,
                       16.0f,
                       17.0f,
                       18.0f};

    int*   dbsrRowPtr = NULL;
    int*   dbsrColInd = NULL;
    float* dbsrVal    = NULL;
    HIP_CHECK(hipMalloc((void**)&dbsrRowPtr, sizeof(int) * (mb + 1)));
    HIP_CHECK(hipMalloc((void**)&dbsrColInd, sizeof(int) * nnzb));
    HIP_CHECK(hipMalloc((void**)&dbsrVal, sizeof(float) * nnzb * blockDim * blockDim));
    HIP_CHECK(hipMemcpy(dbsrRowPtr, hbsrRowPtr, sizeof(int) * (mb + 1), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dbsrColInd, hbsrColInd, sizeof(int) * nnzb, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(
        dbsrVal, hbsrVal, sizeof(float) * nnzb * blockDim * blockDim, hipMemcpyHostToDevice));

    // Copy B to the device
    float* dB;
    HIP_CHECK(hipMalloc((void**)&dB, sizeof(float) * k * n));
    HIP_CHECK(hipMemcpy(dB, hB, sizeof(float) * k * n, hipMemcpyHostToDevice));

    // alpha and beta
    float alpha = 1.0f;
    float beta  = 0.0f;

    // Allocate memory for the resulting matrix C
    float* dC;
    HIP_CHECK(hipMalloc((void**)&dC, sizeof(float) * m * n));

    // Matrix descriptor
    hipsparseMatDescr_t descr;
    HIPSPARSE_CHECK(hipsparseCreateMatDescr(&descr));

    // Perform the matrix multiplication
    HIPSPARSE_CHECK(hipsparseSbsrmm(handle,
                                    dir,
                                    HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                    HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                    mb,
                                    n,
                                    kb,
                                    nnzb,
                                    &alpha,
                                    descr,
                                    dbsrVal,
                                    dbsrRowPtr,
                                    dbsrColInd,
                                    blockDim,
                                    dB,
                                    k,
                                    &beta,
                                    dC,
                                    m));

    // Copy results to host
    float hC[m * n];
    HIP_CHECK(hipMemcpy(hC, dC, sizeof(float) * m * n, hipMemcpyDeviceToHost));

    std::cout << "hC" << std::endl;
    for(int i = 0; i < m * n; i++)
    {
        std::cout << hC[i] << " ";
    }
    std::cout << "" << std::endl;

    HIP_CHECK(hipFree(dbsrRowPtr));
    HIP_CHECK(hipFree(dbsrColInd));
    HIP_CHECK(hipFree(dbsrVal));
    HIP_CHECK(hipFree(dB));
    HIP_CHECK(hipFree(dC));

    return 0;
}
//! [doc example]
