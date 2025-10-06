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

#include "utility.hpp" // Assuming this provides basic utility functions, remove if not needed for standalone example

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

    // Size of square tridiagonal matrix
    int m = 5;

    // Number of columns in right-hand side (column ordered) matrix
    int n = 3;

    // Leading dimension of right-hand side (column ordered) matrix
    int ldb = m;

    // Host tri-diagonal matrix
    //  2 -1  0  0  0
    // -1  2 -1  0  0
    //  0 -1  2 -1  0
    //  0  0 -1  2 -1
    //  0  0  0 -1  2
    std::vector<float> hdl = {0.0f, -1.0f, -1.0f, -1.0f, -1.0f};
    std::vector<float> hd  = {2.0f, 2.0f, 2.0f, 2.0f, 2.0f};
    std::vector<float> hdu = {-1.0f, -1.0f, -1.0f, -1.0f, 0.0f};

    // Host right-hand side column vectors
    std::vector<float> hB(ldb * n, 1.0f);

    float* ddl = nullptr;
    float* dd  = nullptr;
    float* ddu = nullptr;
    float* dB  = nullptr;
    HIP_CHECK(hipMalloc((void**)&ddl, sizeof(float) * m));
    HIP_CHECK(hipMalloc((void**)&dd, sizeof(float) * m));
    HIP_CHECK(hipMalloc((void**)&ddu, sizeof(float) * m));
    HIP_CHECK(hipMalloc((void**)&dB, sizeof(float) * ldb * n));

    HIP_CHECK(hipMemcpy(ddl, hdl.data(), sizeof(float) * m, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dd, hd.data(), sizeof(float) * m, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(ddu, hdu.data(), sizeof(float) * m, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dB, hB.data(), sizeof(float) * ldb * n, hipMemcpyHostToDevice));

    // Obtain required buffer size
    size_t bufferSize;
    HIPSPARSE_CHECK(
        hipsparseSgtsv2_nopivot_bufferSizeExt(handle, m, n, ddl, dd, ddu, dB, ldb, &bufferSize));

    void* dbuffer;
    HIP_CHECK(hipMalloc(&dbuffer, bufferSize));

    HIPSPARSE_CHECK(hipsparseSgtsv2_nopivot(handle, m, n, ddl, dd, ddu, dB, ldb, dbuffer));

    // Copy right-hand side to host
    HIP_CHECK(hipMemcpy(hB.data(), dB, sizeof(float) * ldb * n, hipMemcpyDeviceToHost));

    // Print the solution
    printf("Solution for the tridiagonal system:\n");
    for(int i = 0; i < m; ++i)
    {
        printf("  x[%d] = %f\n", i, hB[i]);
    }

    // Clean up
    HIP_CHECK(hipFree(ddl));
    HIP_CHECK(hipFree(dd));
    HIP_CHECK(hipFree(ddu));
    HIP_CHECK(hipFree(dB));
    HIP_CHECK(hipFree(dbuffer));

    HIPSPARSE_CHECK(hipsparseDestroy(handle));

    return 0;
}
//! [doc example]
