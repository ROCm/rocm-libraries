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

    hipsparseMatDescr_t descr;
    HIPSPARSE_CHECK(hipsparseCreateMatDescr(&descr));

    // Sparse matrix in CSR format (columns unsorted)
    //     1 2 0 3 0
    // A = 0 4 5 0 0
    //     6 0 0 7 8
    int   hcsrRowPtr[4] = {0, 3, 5, 8};
    int   hcsrColInd[8] = {3, 1, 0, 2, 1, 0, 4, 3};
    float hcsrVal[8]    = {3.0f, 2.0f, 1.0f, 5.0f, 4.0f, 6.0f, 8.0f, 7.0f};

    int m   = 3;
    int n   = 5;
    int nnz = 8;

    int*   dcsrRowPtr = nullptr;
    int*   dcsrColInd = nullptr;
    float* dcsrVal    = nullptr;
    HIP_CHECK(hipMalloc((void**)&dcsrRowPtr, sizeof(int) * (m + 1)));
    HIP_CHECK(hipMalloc((void**)&dcsrColInd, sizeof(int) * nnz));
    HIP_CHECK(hipMalloc((void**)&dcsrVal, sizeof(float) * nnz));

    HIP_CHECK(hipMemcpy(dcsrRowPtr, hcsrRowPtr, sizeof(int) * (m + 1), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dcsrColInd, hcsrColInd, sizeof(int) * nnz, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dcsrVal, hcsrVal, sizeof(float) * nnz, hipMemcpyHostToDevice));

    size_t bufferSize;
    HIPSPARSE_CHECK(
        hipsparseXcsrsort_bufferSizeExt(handle, m, n, nnz, dcsrRowPtr, dcsrColInd, &bufferSize));

    void* dbuffer = nullptr;
    HIP_CHECK(hipMalloc((void**)&dbuffer, bufferSize));

    int* dperm = nullptr;
    HIP_CHECK(hipMalloc((void**)&dperm, sizeof(int) * nnz));
    HIPSPARSE_CHECK(hipsparseCreateIdentityPermutation(handle, nnz, dperm));

    HIPSPARSE_CHECK(
        hipsparseXcsrsort(handle, m, n, nnz, descr, dcsrRowPtr, dcsrColInd, dperm, dbuffer));

    float* dcsrValSorted = nullptr;
    HIP_CHECK(hipMalloc((void**)&dcsrValSorted, sizeof(float) * nnz));
    HIPSPARSE_CHECK(
        hipsparseSgthr(handle, nnz, dcsrVal, dcsrValSorted, dperm, HIPSPARSE_INDEX_BASE_ZERO));

    HIP_CHECK(hipFree(dcsrRowPtr));
    HIP_CHECK(hipFree(dcsrColInd));
    HIP_CHECK(hipFree(dcsrVal));
    HIP_CHECK(hipFree(dcsrValSorted));

    HIP_CHECK(hipFree(dbuffer));
    HIP_CHECK(hipFree(dperm));

    HIPSPARSE_CHECK(hipsparseDestroyMatDescr(descr));
    HIPSPARSE_CHECK(hipsparseDestroy(handle));

    return 0;
}
//! [doc example]