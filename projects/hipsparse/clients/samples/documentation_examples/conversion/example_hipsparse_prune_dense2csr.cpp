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

    // Dense matrix in column order
    //     1 2 0 3 0
    // A = 0 4 5 0 0
    //     6 0 0 7 8
    float hdense_A[15] = {
        1.0f, 0.0f, 6.0f, 2.0f, 4.0f, 0.0f, 0.0f, 5.0f, 0.0f, 3.0f, 0.0f, 7.0f, 0.0f, 0.0f, 8.0f};

    int   m         = 3;
    int   n         = 5;
    int   lda       = m;
    float threshold = 4.0f;

    float* ddense_A = nullptr;
    HIP_CHECK(hipMalloc((void**)&ddense_A, sizeof(float) * lda * n));
    HIP_CHECK(hipMemcpy(ddense_A, hdense_A, sizeof(float) * lda * n, hipMemcpyHostToDevice));

    // Allocate sparse CSR matrix
    int* dcsrRowPtr = nullptr;
    HIP_CHECK(hipMalloc((void**)&dcsrRowPtr, sizeof(int) * (m + 1)));

    size_t bufferSize;
    HIPSPARSE_CHECK(hipsparseSpruneDense2csr_bufferSize(
        handle, m, n, ddense_A, lda, &threshold, descr, nullptr, dcsrRowPtr, nullptr, &bufferSize));

    void* dbuffer = nullptr;
    HIP_CHECK(hipMalloc((void**)&dbuffer, bufferSize));

    int nnz_A;
    HIPSPARSE_CHECK(hipsparseSpruneDense2csrNnz(
        handle, m, n, ddense_A, lda, &threshold, descr, dcsrRowPtr, &nnz_A, dbuffer));

    int*   dcsrColInd = nullptr;
    float* dcsrVal    = nullptr;
    HIP_CHECK(hipMalloc((void**)&dcsrColInd, sizeof(int) * nnz_A));
    HIP_CHECK(hipMalloc((void**)&dcsrVal, sizeof(float) * nnz_A));

    HIPSPARSE_CHECK(hipsparseSpruneDense2csr(
        handle, m, n, ddense_A, lda, &threshold, descr, dcsrVal, dcsrRowPtr, dcsrColInd, dbuffer));

    HIP_CHECK(hipFree(dcsrRowPtr));
    HIP_CHECK(hipFree(dcsrColInd));
    HIP_CHECK(hipFree(dcsrVal));
    HIP_CHECK(hipFree(ddense_A));
    HIP_CHECK(hipFree(dbuffer));

    HIPSPARSE_CHECK(hipsparseDestroyMatDescr(descr));
    HIPSPARSE_CHECK(hipsparseDestroy(handle));

    return 0;
}
//! [doc example]