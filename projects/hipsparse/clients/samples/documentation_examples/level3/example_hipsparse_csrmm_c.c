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

    const int                  m   = 4;
    const int                  k   = 6;
    const int                  nnz = 11;
    const hipsparseDirection_t dir = HIPSPARSE_DIRECTION_ROW;

    int   hcsrRowPtr[] = {0, 3, 5, 7, 11};
    int   hcsrColInd[] = {0, 1, 3, 1, 2, 3, 4, 2, 3, 4, 5};
    float hcsrVal[]    = {1, 2, 3, 4, 5, 7, 8, 1, 2, 4, 1};

    // Set dimension n of B
    const int n = 3;

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

    int*   dcsrRowPtr = NULL;
    int*   dcsrColInd = NULL;
    float* dcsrVal    = NULL;
    HIP_CHECK(hipMalloc((void**)&dcsrRowPtr, sizeof(int) * (m + 1)));
    HIP_CHECK(hipMalloc((void**)&dcsrColInd, sizeof(int) * nnz));
    HIP_CHECK(hipMalloc((void**)&dcsrVal, sizeof(float) * nnz));
    HIP_CHECK(hipMemcpy(dcsrRowPtr, hcsrRowPtr, sizeof(int) * (m + 1), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dcsrColInd, hcsrColInd, sizeof(int) * nnz, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dcsrVal, hcsrVal, sizeof(float) * nnz, hipMemcpyHostToDevice));

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
    HIPSPARSE_CHECK(hipsparseScsrmm(handle,
                                    HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                    m,
                                    n,
                                    k,
                                    nnz,
                                    &alpha,
                                    descr,
                                    dcsrVal,
                                    dcsrRowPtr,
                                    dcsrColInd,
                                    dB,
                                    k,
                                    &beta,
                                    dC,
                                    m));

    // Copy results to host
    float hC[6 * 3];
    HIP_CHECK(hipMemcpy(hC, dC, sizeof(float) * m * n, hipMemcpyDeviceToHost));

    printf("hC\n");
    for(int i = 0; i < m * n; i++)
    {
        printf("%f ", hC[i]);
    }
    printf("\n");

    HIP_CHECK(hipFree(dcsrRowPtr));
    HIP_CHECK(hipFree(dcsrColInd));
    HIP_CHECK(hipFree(dcsrVal));
    HIP_CHECK(hipFree(dB));
    HIP_CHECK(hipFree(dC));

    HIPSPARSE_CHECK(hipsparseDestroy(handle));

    return 0;
}
/*! [doc example end] */
