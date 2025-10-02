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

    // Sparse matrix in CSC format (unsorted row indices)
    //     1 2 0 3 0
    // A = 0 4 5 0 0
    //     6 0 0 7 8
    int   hcscRowInd[8] = {2, 0, 1, 0, 1, 2, 0, 2};
    int   hcscColPtr[6] = {0, 2, 4, 5, 7, 8};
    float hcscVal[8]    = {6.0f, 1.0f, 4.0f, 2.0f, 5.0f, 7.0f, 3.0f, 8.0f};

    int m   = 3;
    int n   = 5;
    int nnz = 8;

    int*   dcscRowInd = nullptr;
    int*   dcscColPtr = nullptr;
    float* dcscVal    = nullptr;
    HIP_CHECK(hipMalloc((void**)&dcscRowInd, sizeof(int) * nnz));
    HIP_CHECK(hipMalloc((void**)&dcscColPtr, sizeof(int) * (n + 1)));
    HIP_CHECK(hipMalloc((void**)&dcscVal, sizeof(float) * nnz));

    HIP_CHECK(hipMemcpy(dcscRowInd, hcscRowInd, sizeof(int) * nnz, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dcscColPtr, hcscColPtr, sizeof(int) * (n + 1), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dcscVal, hcscVal, sizeof(float) * nnz, hipMemcpyHostToDevice));

    size_t bufferSize;
    HIPSPARSE_CHECK(
        hipsparseXcscsort_bufferSizeExt(handle, m, n, nnz, dcscColPtr, dcscRowInd, &bufferSize));

    void* dbuffer = nullptr;
    HIP_CHECK(hipMalloc((void**)&dbuffer, bufferSize));

    int* dperm = nullptr;
    HIP_CHECK(hipMalloc((void**)&dperm, sizeof(int) * nnz));
    HIPSPARSE_CHECK(hipsparseCreateIdentityPermutation(handle, nnz, dperm));

    HIPSPARSE_CHECK(
        hipsparseXcscsort(handle, m, n, nnz, descr, dcscColPtr, dcscRowInd, dperm, dbuffer));

    float* dcscValSorted = nullptr;
    HIP_CHECK(hipMalloc((void**)&dcscValSorted, sizeof(float) * nnz));
    HIPSPARSE_CHECK(
        hipsparseSgthr(handle, nnz, dcscVal, dcscValSorted, dperm, HIPSPARSE_INDEX_BASE_ZERO));

    HIP_CHECK(hipFree(dcscRowInd));
    HIP_CHECK(hipFree(dcscColPtr));
    HIP_CHECK(hipFree(dcscVal));
    HIP_CHECK(hipFree(dcscValSorted));

    HIP_CHECK(hipFree(dbuffer));
    HIP_CHECK(hipFree(dperm));

    HIPSPARSE_CHECK(hipsparseDestroyMatDescr(descr));
    HIPSPARSE_CHECK(hipsparseDestroy(handle));

    return 0;
}
//! [doc example]