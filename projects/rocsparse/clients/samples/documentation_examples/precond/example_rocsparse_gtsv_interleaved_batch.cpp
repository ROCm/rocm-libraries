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

#include <iostream>
#include <vector>

#include <rocsparse/rocsparse.h>

#define HIP_CHECK(stat)                                                                       \
    {                                                                                         \
        if(stat != hipSuccess)                                                                \
        {                                                                                     \
            std::cerr << "Error: hip error " << stat << " in line " << __LINE__ << std::endl; \
            return -1;                                                                        \
        }                                                                                     \
    }

#define ROCSPARSE_CHECK(stat)                                                         \
    {                                                                                 \
        if(stat != rocsparse_status_success)                                          \
        {                                                                             \
            std::cerr << "Error: rocsparse error " << stat << " in line " << __LINE__ \
                      << std::endl;                                                   \
            return -1;                                                                \
        }                                                                             \
    }

//! [doc example]
int main()
{
    // Size of each square tridiagonal matrix
    rocsparse_int m = 6;

    // Number of batches
    rocsparse_int batch_count = 4;

    // Batch stride
    rocsparse_int batch_stride = batch_count;

    // Host tridiagonal matrix
    std::vector<float> hdl(m * batch_stride);
    std::vector<float> hd(m * batch_stride);
    std::vector<float> hdu(m * batch_stride);

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
    for(int b = 0; b < batch_count; ++b)
    {
        for(rocsparse_int i = 0; i < m; ++i)
        {
            hdl[batch_stride * i + b] = 2 + b;
            hd[batch_stride * i + b]  = 4 + b;
            hdu[batch_stride * i + b] = 2 + b;
        }

        hdl[batch_stride * 0 + b]       = 0.0f;
        hdu[batch_stride * (m - 1) + b] = 0.0f;
    }

    // Host dense rhs
    std::vector<float> hx(m * batch_stride);

    for(int b = 0; b < batch_count; ++b)
    {
        for(int i = 0; i < m; ++i)
        {
            hx[batch_stride * i + b] = static_cast<float>(b + 1);
        }
    }

    float* ddl;
    float* dd;
    float* ddu;
    float* dx;
    HIP_CHECK(hipMalloc(&ddl, sizeof(float) * m * batch_stride));
    HIP_CHECK(hipMalloc(&dd, sizeof(float) * m * batch_stride));
    HIP_CHECK(hipMalloc(&ddu, sizeof(float) * m * batch_stride));
    HIP_CHECK(hipMalloc(&dx, sizeof(float) * m * batch_stride));

    HIP_CHECK(hipMemcpy(ddl, hdl.data(), sizeof(float) * m * batch_stride, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dd, hd.data(), sizeof(float) * m * batch_stride, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(ddu, hdu.data(), sizeof(float) * m * batch_stride, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dx, hx.data(), sizeof(float) * m * batch_stride, hipMemcpyHostToDevice));

    // rocSPARSE handle
    rocsparse_handle handle;
    ROCSPARSE_CHECK(rocsparse_create_handle(&handle));

    // Obtain required buffer size
    size_t buffer_size;
    ROCSPARSE_CHECK(
        rocsparse_sgtsv_interleaved_batch_buffer_size(handle,
                                                      rocsparse_gtsv_interleaved_alg_default,
                                                      m,
                                                      ddl,
                                                      dd,
                                                      ddu,
                                                      dx,
                                                      batch_count,
                                                      batch_stride,
                                                      &buffer_size));

    void* dbuffer;
    HIP_CHECK(hipMalloc(&dbuffer, buffer_size));

    ROCSPARSE_CHECK(rocsparse_sgtsv_interleaved_batch(handle,
                                                      rocsparse_gtsv_interleaved_alg_default,
                                                      m,
                                                      ddl,
                                                      dd,
                                                      ddu,
                                                      dx,
                                                      batch_count,
                                                      batch_stride,
                                                      dbuffer));

    // Copy right-hand side to host
    HIP_CHECK(hipMemcpy(hx.data(), dx, sizeof(float) * m * batch_stride, hipMemcpyDeviceToHost));

    std::cout << "hx" << std::endl;
    for(size_t i = 0; i < hx.size(); i++)
    {
        std::cout << hx[i] << " ";
    }
    std::cout << "" << std::endl;

    // Clear rocSPARSE
    ROCSPARSE_CHECK(rocsparse_destroy_handle(handle));

    // Clear device memory
    HIP_CHECK(hipFree(ddl));
    HIP_CHECK(hipFree(dd));
    HIP_CHECK(hipFree(ddu));
    HIP_CHECK(hipFree(dx));
    HIP_CHECK(hipFree(dbuffer));

    return 0;
}
//! [doc example]
