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
    // hipSPARSE handle
    hipsparseHandle_t handle;
    HIPSPARSE_CHECK(hipsparseCreate(&handle));

    // Size of each square tridiagonal matrix
    int m = 6;

    // Number of batches
    int batchCount = 4;

    // Can be Thomas algorithm (0), LU (1), or QR (2)
    int algo = 1;

    // Host tridiagonal matrix
    std::vector<float> hdl(m * batchCount);
    std::vector<float> hd(m * batchCount);
    std::vector<float> hdu(m * batchCount);

    // Solve multiple tridiagonal matrix systems by interleaving matrices for better memory access:
    //
    //      4 2 0 0 0 0        5 3 0 0 0 0        6 4 0 0 0 0        7 5 0 0 0 0
    //      2 4 2 0 0 0        3 5 3 0 0 0        4 6 4 0 0 0        5 7 5 0 0 0
    // A1 = 0 2 4 2 0 0   A2 = 0 3 5 3 0 0   A3 = 0 4 6 4 0 0   A4 = 0 5 7 5 0 0
    //      0 0 2 4 2 0        0 0 3 5 3 0        0 0 4 6 4 0        0 0 5 7 5 0
    //      0 0 0 2 4 2        0 0 0 3 5 3        0 0 0 4 6 4        0 0 0 5 7 5
    //      0 0 0 0 2 4        0 0 0 0 3 5        0 0 0 0 4 6        0 0 0 0 5 7
    //
    // hdl = 0 0 0 0 2 3 4 5 2 3 4 5 2 3 4 5 2 3 4 5 2 3 4 5
    // hd  = 4 5 6 7 4 5 6 7 4 5 6 7 4 5 6 7 4 5 6 7 4 5 6 7
    // hdu = 2 3 4 5 2 3 4 5 2 3 4 5 2 3 4 5 2 3 4 5 0 0 0 0
    for(int b = 0; b < batchCount; ++b)
    {
        for(int i = 0; i < m; ++i)
        {
            hdl[batchCount * i + b] = 2 + b;
            hd[batchCount * i + b]  = 4 + b;
            hdu[batchCount * i + b] = 2 + b;
        }

        hdl[batchCount * 0 + b]       = 0.0f;
        hdu[batchCount * (m - 1) + b] = 0.0f;
    }

    // Host dense rhs
    std::vector<float> hx(m * batchCount);

    for(int b = 0; b < batchCount; ++b)
    {
        for(int i = 0; i < m; ++i)
        {
            hx[batchCount * i + b] = static_cast<float>(b + 1);
        }
    }

    float* ddl = nullptr;
    float* dd  = nullptr;
    float* ddu = nullptr;
    float* dx  = nullptr;
    HIP_CHECK(hipMalloc((void**)&ddl, sizeof(float) * m * batchCount));
    HIP_CHECK(hipMalloc((void**)&dd, sizeof(float) * m * batchCount));
    HIP_CHECK(hipMalloc((void**)&ddu, sizeof(float) * m * batchCount));
    HIP_CHECK(hipMalloc((void**)&dx, sizeof(float) * m * batchCount));

    HIP_CHECK(hipMemcpy(ddl, hdl.data(), sizeof(float) * m * batchCount, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dd, hd.data(), sizeof(float) * m * batchCount, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(ddu, hdu.data(), sizeof(float) * m * batchCount, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dx, hx.data(), sizeof(float) * m * batchCount, hipMemcpyHostToDevice));

    // 1. Get buffer size
    size_t bufferSize = 0;
    HIPSPARSE_CHECK(hipsparseSgtsvInterleavedBatch_bufferSizeExt(
        handle, algo, m, ddl, dd, ddu, dx, batchCount, &bufferSize));

    void* dbuffer = nullptr;
    HIP_CHECK(hipMalloc((void**)&dbuffer, bufferSize));

    // 2. Perform batched tridiagonal solve
    HIPSPARSE_CHECK(
        hipsparseSgtsvInterleavedBatch(handle, algo, m, ddl, dd, ddu, dx, batchCount, dbuffer));

    // Copy solution back to host
    HIP_CHECK(hipMemcpy(hx.data(), dx, sizeof(float) * m * batchCount, hipMemcpyDeviceToHost));

    // Print the solutions
    printf("Solutions for batched tridiagonal systems:\n");
    for(int b = 0; b < batchCount; ++b)
    {
        printf("  Batch %d:\n", b);
        for(int i = 0; i < m; ++i)
        {
            printf("    x[%d] = %f\n", i, hx[i * batchCount + b]);
        }
    }

    // Clean up
    HIP_CHECK(hipFree(ddl));
    HIP_CHECK(hipFree(dd));
    HIP_CHECK(hipFree(ddu));
    HIP_CHECK(hipFree(dx));
    HIP_CHECK(hipFree(dbuffer));

    HIPSPARSE_CHECK(hipsparseDestroy(handle));

    return 0;
}
//! [doc example]