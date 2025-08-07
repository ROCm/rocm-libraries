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

    // Define the size of the tridiagonal system
    int m = 4; // Dimension of the square matrix

    // A sample tridiagonal linear system Ax = f, where:
    // A is not necessarily diagonally dominant, so a solver with pivoting is a safer choice.
    // Matrix:
    // ( 1  2  0  0 )
    // ( 3  4  5  0 )
    // ( 0  6  7  8 )
    // ( 0  0  9 10 )
    // Right-hand side vector: f = [1, 2, 3, 4]

    // Host arrays for lower diagonal (l), main diagonal (d), upper diagonal (u),
    // and right-hand side (f) vectors.
    std::vector<float> h_dl(m - 1); // Lower diagonal (m-1 elements)
    std::vector<float> h_d(m); // Main diagonal (m elements)
    std::vector<float> h_du(m - 1); // Upper diagonal (m-1 elements)
    std::vector<float> h_f(m); // Right-hand side vector (m elements)
    std::vector<float> h_x(m); // Solution vector (m elements)

    // Populate host data
    h_dl = {3.0f, 6.0f, 9.0f};
    h_d  = {1.0f, 4.0f, 7.0f, 10.0f};
    h_du = {2.0f, 5.0f, 8.0f};
    h_f  = {1.0f, 2.0f, 3.0f, 4.0f};

    // Device memory pointers
    float* d_dl;
    float* d_d;
    float* d_du;
    float* d_f;
    float* d_x; // This is not strictly needed as result is written into d_f, but good for clarity.

    HIP_CHECK(hipMalloc((void**)&d_dl, sizeof(float) * (m - 1)));
    HIP_CHECK(hipMalloc((void**)&d_d, sizeof(float) * m));
    HIP_CHECK(hipMalloc((void**)&d_du, sizeof(float) * (m - 1)));
    HIP_CHECK(hipMalloc((void**)&d_f, sizeof(float) * m));
    HIP_CHECK(hipMalloc((void**)&d_x, sizeof(float) * m));

    // Copy host data to device
    HIP_CHECK(hipMemcpy(d_dl, h_dl.data(), sizeof(float) * (m - 1), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_d, h_d.data(), sizeof(float) * m, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_du, h_du.data(), sizeof(float) * (m - 1), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_f, h_f.data(), sizeof(float) * m, hipMemcpyHostToDevice));

    // 1. Get buffer size
    size_t bufferSize = 0;
    HIPSPARSE_CHECK(hipsparseSgtsv2_bufferSizeExt(handle, m, d_dl, d_d, d_du, d_f, &bufferSize));

    void* dbuffer = nullptr;
    HIP_CHECK(hipMalloc((void**)&dbuffer, bufferSize));

    // 2. Perform tridiagonal solve with pivoting
    // The solution is computed and stored in the d_f vector.
    HIPSPARSE_CHECK(hipsparseSgtsv2(handle, m, d_dl, d_d, d_du, d_f, dbuffer));

    // Copy solution back to host from d_f
    HIP_CHECK(hipMemcpy(h_x.data(), d_f, sizeof(float) * m, hipMemcpyDeviceToHost));

    // Print the solution
    printf("Solution for the tridiagonal system:\n");
    for(int i = 0; i < m; ++i)
    {
        printf("  x[%d] = %f\n", i, h_x[i]);
    }

    // Clean up
    HIP_CHECK(hipFree(d_dl));
    HIP_CHECK(hipFree(d_d));
    HIP_CHECK(hipFree(d_du));
    HIP_CHECK(hipFree(d_f));
    HIP_CHECK(hipFree(d_x));
    HIP_CHECK(hipFree(dbuffer));

    HIPSPARSE_CHECK(hipsparseDestroy(handle));

    return 0;
}
//! [doc example]
