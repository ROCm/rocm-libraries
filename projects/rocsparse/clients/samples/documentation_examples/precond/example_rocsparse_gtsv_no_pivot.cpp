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
    // Size of square tridiagonal matrix
    rocsparse_int m = 5;

    // Number of columns in right-hand side (column ordered) matrix
    rocsparse_int n = 3;

    // Leading dimension of right-hand side (column ordered) matrix
    rocsparse_int ldb = m;

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

    float* ddl;
    float* dd;
    float* ddu;
    float* dB;
    HIP_CHECK(hipMalloc(&ddl, sizeof(float) * m));
    HIP_CHECK(hipMalloc(&dd, sizeof(float) * m));
    HIP_CHECK(hipMalloc(&ddu, sizeof(float) * m));
    HIP_CHECK(hipMalloc(&dB, sizeof(float) * ldb * n));

    HIP_CHECK(hipMemcpy(ddl, hdl.data(), sizeof(float) * m, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dd, hd.data(), sizeof(float) * m, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(ddu, hdu.data(), sizeof(float) * m, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dB, hB.data(), sizeof(float) * ldb * n, hipMemcpyHostToDevice));

    // rocSPARSE handle
    rocsparse_handle handle;
    ROCSPARSE_CHECK(rocsparse_create_handle(&handle));

    // Obtain required buffer size
    size_t buffer_size;
    ROCSPARSE_CHECK(
        rocsparse_sgtsv_no_pivot_buffer_size(handle, m, n, ddl, dd, ddu, dB, ldb, &buffer_size));

    void* dbuffer;
    HIP_CHECK(hipMalloc(&dbuffer, buffer_size));

    ROCSPARSE_CHECK(rocsparse_sgtsv_no_pivot(handle, m, n, ddl, dd, ddu, dB, ldb, dbuffer));

    // Copy right-hand side to host
    HIP_CHECK(hipMemcpy(hB.data(), dB, sizeof(float) * ldb * n, hipMemcpyDeviceToHost));

    std::cout << "hB" << std::endl;
    for(size_t i = 0; i < hB.size(); i++)
    {
        std::cout << hB[i] << " ";
    }
    std::cout << "" << std::endl;

    // Clear rocSPARSE
    ROCSPARSE_CHECK(rocsparse_destroy_handle(handle));

    // Clear device memory
    HIP_CHECK(hipFree(ddl));
    HIP_CHECK(hipFree(dd));
    HIP_CHECK(hipFree(ddu));
    HIP_CHECK(hipFree(dB));
    HIP_CHECK(hipFree(dbuffer));

    return 0;
}
//! [doc example]
