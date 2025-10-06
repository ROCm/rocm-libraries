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
    // Number of non-zeros of the sparse vector
    const int nnz = 3;

    // Number of entries in the dense vector
    const int size = 9;

    // Sparse index vector
    int hxInd[nnz] = {0, 3, 5};

    // Sparse value vector
    float hxVal[nnz] = {1.0f, 2.0f, 3.0f};

    // Dense vector
    float hy[size] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};

    // c and s
    float c = 3.7;
    float s = 1.3;

    // Index base
    hipsparseIndexBase_t idxBase = HIPSPARSE_INDEX_BASE_ZERO;

    // Offload data to device
    int*   dxInd;
    float* dxVal;
    float* dy;

    HIP_CHECK(hipMalloc((void**)&dxInd, sizeof(int) * nnz));
    HIP_CHECK(hipMalloc((void**)&dxVal, sizeof(float) * nnz));
    HIP_CHECK(hipMalloc((void**)&dy, sizeof(float) * size));

    HIP_CHECK(hipMemcpy(dxInd, hxInd, sizeof(int) * nnz, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dxVal, hxVal, sizeof(float) * nnz, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dy, hy, sizeof(float) * size, hipMemcpyHostToDevice));

    // hipSPARSE handle
    hipsparseHandle_t handle;
    HIPSPARSE_CHECK(hipsparseCreate(&handle));

    // Call sroti
    HIPSPARSE_CHECK(hipsparseSroti(handle, nnz, dxVal, dxInd, dy, &c, &s, idxBase));

    // Copy result back to host
    HIP_CHECK(hipMemcpy(hxVal, dxVal, sizeof(float) * nnz, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(hy, dy, sizeof(float) * size, hipMemcpyDeviceToHost));

    std::cout << "hxVal" << std::endl;
    for(int i = 0; i < nnz; i++)
    {
        std::cout << hxVal[i] << " ";
    }
    std::cout << "" << std::endl;

    std::cout << "hy" << std::endl;
    for(int i = 0; i < size; i++)
    {
        std::cout << hy[i] << " ";
    }
    std::cout << "" << std::endl;

    // Clear hipSPARSE
    HIPSPARSE_CHECK(hipsparseDestroy(handle));

    // Clear device memory
    HIP_CHECK(hipFree(dxInd));
    HIP_CHECK(hipFree(dxVal));
    HIP_CHECK(hipFree(dy));

    return 0;
}
//! [doc example]
