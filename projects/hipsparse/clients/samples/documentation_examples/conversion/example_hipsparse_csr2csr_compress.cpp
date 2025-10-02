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

    // Matrix descriptor
    hipsparseMatDescr_t descr;
    HIPSPARSE_CHECK(hipsparseCreateMatDescr(&descr));

    // Sparse matrix in CSR format
    //     1 2 0 3 0
    // A = 0 4 5 0 0
    //     6 0 0 7 8
    int   hcsrRowPtrA[4] = {0, 3, 5, 8};
    int   hcsrColIndA[8] = {0, 1, 3, 1, 2, 0, 3, 4};
    float hcsrValA[8]    = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};

    int m    = 3;
    int n    = 5;
    int nnzA = 8;

    float tol = 5.9f;

    int*   dcsrRowPtrA = nullptr;
    int*   dcsrColIndA = nullptr;
    float* dcsrValA    = nullptr;
    HIP_CHECK(hipMalloc((void**)&dcsrRowPtrA, sizeof(int) * (m + 1)));
    HIP_CHECK(hipMalloc((void**)&dcsrColIndA, sizeof(int) * nnzA));
    HIP_CHECK(hipMalloc((void**)&dcsrValA, sizeof(float) * nnzA));

    HIP_CHECK(hipMemcpy(dcsrRowPtrA, hcsrRowPtrA, sizeof(int) * (m + 1), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dcsrColIndA, hcsrColIndA, sizeof(int) * nnzA, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dcsrValA, hcsrValA, sizeof(float) * nnzA, hipMemcpyHostToDevice));

    // Allocate memory for the nnz_per_row array
    int* dnnz_per_row;
    HIP_CHECK(hipMalloc((void**)&dnnz_per_row, sizeof(int) * m));

    // Call snnz_compress() which fills in nnz_per_row array and finds the number
    // of entries that will be in the compressed CSR matrix
    int nnzC;
    HIPSPARSE_CHECK(
        hipsparseSnnz_compress(handle, m, descr, dcsrValA, dcsrRowPtrA, dnnz_per_row, &nnzC, tol));

    int*   dcsrRowPtrC = nullptr;
    int*   dcsrColIndC = nullptr;
    float* dcsrValC    = nullptr;
    HIP_CHECK(hipMalloc((void**)&dcsrRowPtrC, sizeof(int) * (m + 1)));
    HIP_CHECK(hipMalloc((void**)&dcsrColIndC, sizeof(int) * nnzC));
    HIP_CHECK(hipMalloc((void**)&dcsrValC, sizeof(float) * nnzC));

    HIPSPARSE_CHECK(hipsparseScsr2csr_compress(handle,
                                               m,
                                               n,
                                               descr,
                                               dcsrValA,
                                               dcsrColIndA,
                                               dcsrRowPtrA,
                                               nnzA,
                                               dnnz_per_row,
                                               dcsrValC,
                                               dcsrColIndC,
                                               dcsrRowPtrC,
                                               tol));

    HIP_CHECK(hipFree(dcsrRowPtrA));
    HIP_CHECK(hipFree(dcsrColIndA));
    HIP_CHECK(hipFree(dcsrValA));

    HIP_CHECK(hipFree(dcsrRowPtrC));
    HIP_CHECK(hipFree(dcsrColIndC));
    HIP_CHECK(hipFree(dcsrValC));

    HIP_CHECK(hipFree(dnnz_per_row));

    HIPSPARSE_CHECK(hipsparseDestroyMatDescr(descr));
    HIPSPARSE_CHECK(hipsparseDestroy(handle));

    return 0;
}
//! [doc example]