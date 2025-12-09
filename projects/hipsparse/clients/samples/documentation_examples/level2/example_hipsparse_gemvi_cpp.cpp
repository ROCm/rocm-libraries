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
#include <iostream>
#include <vector>

#define HIP_CHECK(stat)                                                        \
    {                                                                          \
        if(stat != hipSuccess)                                                 \
        {                                                                      \
            std::cerr << "Error: hip error in line " << __LINE__ << std::endl; \
            return -1;                                                         \
        }                                                                      \
    }

#define HIPSPARSE_CHECK(stat)                                                        \
    {                                                                                \
        if(stat != HIPSPARSE_STATUS_SUCCESS)                                         \
        {                                                                            \
            std::cerr << "Error: hipsparse error in line " << __LINE__ << std::endl; \
            return -1;                                                               \
        }                                                                            \
    }

//! [doc example start]
int main(int argc, char* argv[])
{
    hipsparseOperation_t opA     = HIPSPARSE_OPERATION_NON_TRANSPOSE;
    hipsparseIndexBase_t idxBase = HIPSPARSE_INDEX_BASE_ZERO;

    // Scalar alpha and beta
    float alpha = 1.0f;
    float beta  = 1.0f;

    const int m   = 4; // Number of rows of A
    const int n   = 4; // Number of columns of A
    const int lda = m; // leading dimension of A

    // A = 1 2 3 4
    //     5 6 7 8
    //     2 4 6 8
    //     4 3 2 1
    std::vector<float> hA = {1.0f,
                             5.0f,
                             2.0f,
                             4.0f,
                             2.0f,
                             6.0f,
                             4.0f,
                             3.0f,
                             3.0f,
                             7.0f,
                             6.0f,
                             2.0f,
                             4.0f,
                             8.0f,
                             8.0f,
                             1.0f};

    // Sparse vector x
    int                nnz   = 2;
    std::vector<int>   hxInd = {0, 2};
    std::vector<float> hx    = {10.0f, 11.0f};

    // Dense vector y
    std::vector<float> hy = {1.0f, 2.0f, 3.0f, 4.0f};

    // Device data
    float* dA = nullptr;
    HIP_CHECK(hipMalloc((void**)&dA, sizeof(float) * m * n));

    int*   dxInd = nullptr;
    float* dx    = nullptr;
    HIP_CHECK(hipMalloc((void**)&dxInd, sizeof(int) * nnz));
    HIP_CHECK(hipMalloc((void**)&dx, sizeof(float) * nnz));

    float* dy = nullptr;
    HIP_CHECK(hipMalloc((void**)&dy, sizeof(float) * m));

    // Copy data from host to device
    HIP_CHECK(hipMemcpy(dA, hA.data(), sizeof(float) * m * n, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dxInd, hxInd.data(), sizeof(int) * nnz, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dx, hx.data(), sizeof(float) * nnz, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dy, hy.data(), sizeof(float) * m, hipMemcpyHostToDevice));

    // hipSPARSE handle
    hipsparseHandle_t handle;
    HIPSPARSE_CHECK(hipsparseCreate(&handle));

    // Call hipsparseSgemvi to perform y = alpha * A * x + beta * y
    int bufferSize = 0;
    HIPSPARSE_CHECK(hipsparseSgemvi_bufferSize(handle, opA, m, n, nnz, &bufferSize));

    void* dbuffer = nullptr;
    HIP_CHECK(hipMalloc((void**)&dbuffer, bufferSize));

    HIPSPARSE_CHECK(hipsparseSgemvi(
        handle, opA, m, n, &alpha, dA, lda, nnz, dx, dxInd, &beta, dy, idxBase, dbuffer));

    // Copy result back to host
    HIP_CHECK(hipMemcpy(hy.data(), dy, sizeof(float) * m, hipMemcpyDeviceToHost));

    // Print the result (optional)
    std::cout << "hy" << std::endl;
    for(int i = 0; i < m; i++)
    {
        std::cout << hy[i] << " ";
    }
    std::cout << "" << std::endl;

    // Clear hipSPARSE
    HIPSPARSE_CHECK(hipsparseDestroy(handle));

    // Clear device memory
    HIP_CHECK(hipFree(dA));
    HIP_CHECK(hipFree(dxInd));
    HIP_CHECK(hipFree(dx));
    HIP_CHECK(hipFree(dy));
    HIP_CHECK(hipFree(dbuffer));

    return 0;
}
//! [doc example end]