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

    // Define parameters for the batched tridiagonal systems
    int m          = 4; // Dimension of each tridiagonal system (number of rows/columns)
    int batchCount = 2; // Number of tridiagonal systems to solve in batch

    // Host arrays for lower diagonal (dl), main diagonal (d), upper diagonal (du),
    // and right-hand side (f) vectors.
    // These arrays will be prepared in interleaved storage format.
    // Total elements:
    // (m-1) * batchCount for dl_interleaved and du_interleaved
    // m * batchCount for d_interleaved, f_interleaved, and x_interleaved

    // Example System 1 (4x4 tridiagonal matrix):
    // ( 2 -1  0  0 ) (x0)   (1)
    // (-1  2 -1  0 ) (x1) = (2)
    // ( 0 -1  2 -1 ) (x2)   (3)
    // ( 0  0 -1  2 ) (x3)   (4)

    // Example System 2 (4x4 tridiagonal matrix):
    // ( 3 -1  0  0 ) (x0)   (5)
    // (-1  3 -1  0 ) (x1) = (6)
    // ( 0 -1  3 -1 ) (x2)   (7)
    // ( 0  0 -1  3 ) (x3)   (8)

    // Prepare host data in interleaved storage
    // hdl_interleaved: (m-1) * batchCount elements
    // hd_interleaved: m * batchCount elements
    // hdu_interleaved: (m-1) * batchCount elements
    // hf_interleaved: m * batchCount elements
    // hx_interleaved: m * batchCount elements (for solutions)

    std::vector<float> hdl_interleaved((m - 1) * batchCount);
    std::vector<float> hd_interleaved(m * batchCount);
    std::vector<float> hdu_interleaved((m - 1) * batchCount);
    std::vector<float> hf_interleaved(m * batchCount);
    std::vector<float> hx_interleaved(m * batchCount); // Solution vector

    // Populate interleaved data
    // Indexing: element `i` of batch `b` is at `(i * batchCount) + b`
    for(int i = 0; i < m; ++i)
    {
        // Main diagonal (d)
        hd_interleaved[i * batchCount + 0] = 2.0f; // System 1
        hd_interleaved[i * batchCount + 1] = 3.0f; // System 2

        // Right-hand side (f)
        hf_interleaved[i * batchCount + 0] = (float)(i + 1); // System 1
        hf_interleaved[i * batchCount + 1] = (float)(i + 5); // System 2
    }

    for(int i = 0; i < m - 1; ++i)
    {
        // Lower diagonal (dl)
        hdl_interleaved[i * batchCount + 0] = -1.0f; // System 1
        hdl_interleaved[i * batchCount + 1] = -1.0f; // System 2

        // Upper diagonal (du)
        hdu_interleaved[i * batchCount + 0] = -1.0f; // System 1
        hdu_interleaved[i * batchCount + 1] = -1.0f; // System 2
    }

    // Device memory pointers
    float* d_dl_interleaved;
    float* d_d_interleaved;
    float* d_du_interleaved;
    float* d_x_interleaved;
    float* d_f_interleaved;

    HIP_CHECK(hipMalloc((void**)&d_dl_interleaved, sizeof(float) * (m - 1) * batchCount));
    HIP_CHECK(hipMalloc((void**)&d_d_interleaved, sizeof(float) * m * batchCount));
    HIP_CHECK(hipMalloc((void**)&d_du_interleaved, sizeof(float) * (m - 1) * batchCount));
    HIP_CHECK(hipMalloc((void**)&d_x_interleaved, sizeof(float) * m * batchCount));
    HIP_CHECK(hipMalloc((void**)&d_f_interleaved, sizeof(float) * m * batchCount));

    // Copy host data to device
    HIP_CHECK(hipMemcpy(d_dl_interleaved,
                        hdl_interleaved.data(),
                        sizeof(float) * (m - 1) * batchCount,
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_d_interleaved,
                        hd_interleaved.data(),
                        sizeof(float) * m * batchCount,
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_du_interleaved,
                        hdu_interleaved.data(),
                        sizeof(float) * (m - 1) * batchCount,
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_f_interleaved,
                        hf_interleaved.data(),
                        sizeof(float) * m * batchCount,
                        hipMemcpyHostToDevice));

    // 1. Get buffer size
    size_t bufferSize = 0;
    HIPSPARSE_CHECK(hipsparseSgtsvInterleavedBatch_bufferSizeExt(handle,
                                                                 m,
                                                                 batchCount,
                                                                 d_dl_interleaved,
                                                                 d_d_interleaved,
                                                                 d_du_interleaved,
                                                                 d_x_interleaved,
                                                                 d_f_interleaved,
                                                                 &bufferSize));

    void* dbuffer = nullptr;
    HIP_CHECK(hipMalloc((void**)&dbuffer, bufferSize));

    // 2. Perform batched tridiagonal solve
    HIPSPARSE_CHECK(hipsparseSgtsvInterleavedBatch(handle,
                                                   m,
                                                   batchCount,
                                                   d_dl_interleaved,
                                                   d_d_interleaved,
                                                   d_du_interleaved,
                                                   d_x_interleaved,
                                                   d_f_interleaved,
                                                   dbuffer));

    // Copy solution back to host
    HIP_CHECK(hipMemcpy(hx_interleaved.data(),
                        d_x_interleaved,
                        sizeof(float) * m * batchCount,
                        hipMemcpyDeviceToHost));

    // Print the solutions
    printf("Solutions for batched tridiagonal systems:\n");
    for(int b = 0; b < batchCount; ++b)
    {
        printf("  Batch %d:\n", b);
        for(int i = 0; i < m; ++i)
        {
            printf("    x[%d] = %f\n", i, hx_interleaved[i * batchCount + b]);
        }
    }

    // Clean up
    HIP_CHECK(hipFree(d_dl_interleaved));
    HIP_CHECK(hipFree(d_d_interleaved));
    HIP_CHECK(hipFree(d_du_interleaved));
    HIP_CHECK(hipFree(d_x_interleaved));
    HIP_CHECK(hipFree(d_f_interleaved));
    HIP_CHECK(hipFree(dbuffer));

    HIPSPARSE_CHECK(hipsparseDestroy(handle));

    return 0;
}
//! [doc example]