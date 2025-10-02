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

    // Sparse matrix in COO format
    //     1 2 0 3 0
    // A = 0 4 5 0 0
    //     6 0 0 7 8
    int   hcooRowInd[8] = {0, 0, 0, 1, 1, 2, 2, 2};
    int   hcooColInd[8] = {0, 1, 3, 1, 2, 0, 3, 4};
    float hcooVal[8]    = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};

    int                  m    = 3;
    int                  n    = 5;
    int                  nnz  = 8;
    hipsparseIndexBase_t base = HIPSPARSE_INDEX_BASE_ZERO;

    int* dcooRowInd = nullptr;
    int* dcooColInd = nullptr;
    HIP_CHECK(hipMalloc((void**)&dcooRowInd, sizeof(int) * nnz));
    HIP_CHECK(hipMalloc((void**)&dcooColInd, sizeof(int) * nnz));

    HIP_CHECK(hipMemcpy(dcooRowInd, hcooRowInd, sizeof(int) * nnz, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dcooColInd, hcooColInd, sizeof(int) * nnz, hipMemcpyHostToDevice));

    int* dcsrRowPtr = nullptr;
    HIP_CHECK(hipMalloc((void**)&dcsrRowPtr, sizeof(int) * (m + 1)));

    HIPSPARSE_CHECK(hipsparseXcoo2csr(handle, dcooRowInd, nnz, m, dcsrRowPtr, base));

    HIP_CHECK(hipFree(dcooRowInd));
    HIP_CHECK(hipFree(dcooColInd));

    HIP_CHECK(hipFree(dcsrRowPtr));

    HIPSPARSE_CHECK(hipsparseDestroy(handle));

    return 0;
}
//! [doc example]