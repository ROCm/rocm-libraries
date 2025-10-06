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
    // // hipSPARSE handle
    // hipsparseHandle_t handle;
    // HIPSPARSE_CHECK(hipsparseCreate(&handle));

    // // Define a sparse matrix A (CSR format) and vectors x and y
    // // For sgemvi: y = alpha * A * x + beta * y
    // // M x N matrix, where M is number of rows, N is number of columns.
    // // In this example, we'll assume a square matrix for simplicity, M = N = 4.
    // // However, hipsparseSgemvi supports M != N.

    // const int m   = 4; // Number of rows of A
    // const int n   = 4; // Number of columns of A
    // const int nnz = 13; // Number of non-zero elements in A

    // hipsparseOperation_t op = HIPSPARSE_OPERATION_NON_TRANSPOSE;

    // // CSR row pointers
    // int hcsrRowPtr[m + 1] = {0, 2, 6, 10, 13};

    // // CSR column indices
    // int hcsrColInd[nnz] = {0, 2, 0, 1, 2, 3, 0, 1, 2, 3, 0, 2, 3};

    // // CSR values (single precision float for 'S'gemvi)
    // float hcsrVal[nnz]
    //     = {1.0f, 2.0f, 3.0f, 2.0f, 4.0f, 1.0f, 5.0f, 6.0f, 1.0f, 3.0f, 7.0f, 8.0f, 0.6f};

    // // Scalar alpha and beta
    // float alpha = 1.0f;
    // float beta  = 0.0f; // For y = A * x (effectively)

    // // Input vector x and initial y (output vector)
    // float hx[n] = {1.0f, 2.0f, 3.0f, 4.0f}; // Example input x
    // float hy[m] = {0.0f, 0.0f, 0.0f, 0.0f}; // Initialize y to zeros for y = A * x

    // // Matrix descriptor
    // hipsparseMatDescr_t descr;
    // HIPSPARSE_CHECK(hipsparseCreateMatDescr(&descr));

    // // Set index base on descriptor
    // HIPSPARSE_CHECK(hipsparseSetMatIndexBase(descr, HIPSPARSE_INDEX_BASE_ZERO));

    // // Offload data to device
    // int*   dcsrRowPtr;
    // int*   dcsrColInd;
    // float* dcsrVal;
    // float* dx;
    // float* dy;

    // HIP_CHECK(hipMalloc((void**)&dcsrRowPtr, sizeof(int) * (m + 1)));
    // HIP_CHECK(hipMalloc((void**)&dcsrColInd, sizeof(int) * nnz));
    // HIP_CHECK(hipMalloc((void**)&dcsrVal, sizeof(float) * nnz));
    // HIP_CHECK(hipMalloc((void**)&dx, sizeof(float) * n)); // x has 'n' elements
    // HIP_CHECK(hipMalloc((void**)&dy, sizeof(float) * m)); // y has 'm' elements

    // HIP_CHECK(hipMemcpy(dcsrRowPtr, hcsrRowPtr, sizeof(int) * (m + 1), hipMemcpyHostToDevice));
    // HIP_CHECK(hipMemcpy(dcsrColInd, hcsrColInd, sizeof(int) * nnz, hipMemcpyHostToDevice));
    // HIP_CHECK(hipMemcpy(dcsrVal, hcsrVal, sizeof(float) * nnz, hipMemcpyHostToDevice));
    // HIP_CHECK(hipMemcpy(dx, hx, sizeof(float) * n, hipMemcpyHostToDevice));
    // HIP_CHECK(hipMemcpy(dy, hy, sizeof(float) * m, hipMemcpyHostToDevice));

    // // Call hipsparseSgemvi to perform y = alpha * A * x + beta * y
    // // hipsparseSgemvi does not require a bufferSize or analysis phase.

    // HIPSPARSE_CHECK(hipsparseSgemvi(
    //     handle, op, m, n, &alpha, A, lda, nnz, dx, dxInd, &beta, dy, indBase, buffer));

    // // Copy result back to host
    // HIP_CHECK(hipMemcpy(hy, dy, sizeof(float) * m, hipMemcpyDeviceToHost));

    // // Print the result (optional)
    // std::cout << "hy" << std::endl;
    // for(int i = 0; i < m; i++)
    // {
    //     std::cout << hy[i] << " ";
    // }
    // std::cout << "" << std::endl;

    // // Clear hipSPARSE
    // HIPSPARSE_CHECK(hipsparseDestroyMatDescr(descr));
    // HIPSPARSE_CHECK(hipsparseDestroy(handle));

    // // Clear device memory
    // HIP_CHECK(hipFree(dcsrRowPtr));
    // HIP_CHECK(hipFree(dcsrColInd));
    // HIP_CHECK(hipFree(dcsrVal));
    // HIP_CHECK(hipFree(dx));
    // HIP_CHECK(hipFree(dy));

    return 0;
}
//! [doc example]
