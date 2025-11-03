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

#define HIP_CHECK(stat)                                                 \
    {                                                                   \
        if(stat != hipSuccess)                                          \
        {                                                               \
            fprintf(stderr, "Error: hip error in line %d\n", __LINE__); \
            return -1;                                                  \
        }                                                               \
    }

#define HIPSPARSE_CHECK(stat)                                                 \
    {                                                                         \
        if(stat != HIPSPARSE_STATUS_SUCCESS)                                  \
        {                                                                     \
            fprintf(stderr, "Error: hipsparse error in line %d\n", __LINE__); \
            return -1;                                                        \
        }                                                                     \
    }

/*! [doc example start] */
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

    int   hbsrRowPtr[] = {0, 2, 4};
    int   hbsrColInd[] = {0, 1, 1, 2};
    float hbsrVal[]    = {1, 2, 0, 4, 0, 3, 5, 0, 0, 7, 1, 2, 8, 0, 4, 1};

    // Set dimension n of B
    const int n = 3;
    const int m = mb * blockDim;
    const int k = kb * blockDim;

    // Allocate and generate dense matrix B (k x n)
    float hB[] = {1.0,
                  2.0,
                  3.0,
                  4.0,
                  5.0,
                  6.0,
                  7.0,
                  8.0,
                  9.0,
                  10.0,
                  11.0,
                  12.0,
                  13.0,
                  14.0,
                  15.0,
                  16.0,
                  17.0,
                  18.0};

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
    float alpha = 1.0;
    float beta  = 0.0;

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

    printf("hC\n");
    for(int i = 0; i < m * n; i++)
    {
        printf("%f ", hC[i]);
    }
    printf("\n");

    HIP_CHECK(hipFree(dbsrRowPtr));
    HIP_CHECK(hipFree(dbsrColInd));
    HIP_CHECK(hipFree(dbsrVal));
    HIP_CHECK(hipFree(dB));
    HIP_CHECK(hipFree(dC));

    return 0;
}
/*! [doc example end] */
