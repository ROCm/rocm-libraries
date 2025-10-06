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
#include <vector>

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
    // A, B, and C are m×k, k×n, and m×n
    const int m = 3, n = 5, k = 4;
    const int lda = m, ldc = m;
    const int nnz_A = m * k, nnz_B = 10, nnz_C = m * n;

    // alpha and beta
    float alpha = 0.5f;
    float beta  = 0.25f;

    std::vector<int>   hcscColPtr = {0, 2, 5, 7, 8, 10};
    std::vector<int>   hcscRowInd = {0, 2, 0, 1, 3, 1, 3, 2, 0, 2};
    std::vector<float> hcsc_val   = {1, 6, 2, 4, 9, 5, 2, 7, 3, 8};

    std::vector<float> hA(nnz_A, 1.0f);
    std::vector<float> hC(nnz_C, 1.0f);

    int*   dcscColPtr;
    int*   dcscRowInd;
    float* dcsc_val;
    HIP_CHECK(hipMalloc((void**)&dcscColPtr, sizeof(int) * (n + 1)));
    HIP_CHECK(hipMalloc((void**)&dcscRowInd, sizeof(int) * nnz_B));
    HIP_CHECK(hipMalloc((void**)&dcsc_val, sizeof(float) * nnz_B));

    HIP_CHECK(
        hipMemcpy(dcscColPtr, hcscColPtr.data(), sizeof(int) * (n + 1), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dcscRowInd, hcscRowInd.data(), sizeof(int) * nnz_B, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dcsc_val, hcsc_val.data(), sizeof(float) * nnz_B, hipMemcpyHostToDevice));

    hipsparseHandle_t handle;
    HIPSPARSE_CHECK(hipsparseCreate(&handle));

    // Allocate memory for the matrix A
    float* dA;
    HIP_CHECK(hipMalloc((void**)&dA, sizeof(float) * nnz_A));
    HIP_CHECK(hipMemcpy(dA, hA.data(), sizeof(float) * nnz_A, hipMemcpyHostToDevice));

    // Allocate memory for the resulting matrix C
    float* dC;
    HIP_CHECK(hipMalloc((void**)&dC, sizeof(float) * nnz_C));
    HIP_CHECK(hipMemcpy(dC, hC.data(), sizeof(float) * nnz_C, hipMemcpyHostToDevice));

    // Perform operation
    HIPSPARSE_CHECK(hipsparseSgemmi(
        handle, m, n, k, nnz_B, &alpha, dA, lda, dcsc_val, dcscColPtr, dcscRowInd, &beta, dC, ldc));

    // Copy device to host
    HIP_CHECK(hipMemcpy(hC.data(), dC, sizeof(float) * nnz_C, hipMemcpyDeviceToHost));

    std::cout << "hC" << std::endl;
    for(int i = 0; i < nnz_C; i++)
    {
        std::cout << hC[i] << " ";
    }
    std::cout << "" << std::endl;

    // Destroy matrix descriptors and handles
    HIPSPARSE_CHECK(hipsparseDestroy(handle));

    HIP_CHECK(hipFree(dcscColPtr));
    HIP_CHECK(hipFree(dcscRowInd));
    HIP_CHECK(hipFree(dcsc_val));
    HIP_CHECK(hipFree(dA));
    HIP_CHECK(hipFree(dC));

    return 0;
}
//! [doc example]
