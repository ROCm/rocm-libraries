/*! \file */
/* ************************************************************************
 * Copyright (C) 2021 Advanced Micro Devices, Inc. All rights Reserved.
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
#include <iostream>
#include <rocsparse/rocsparse.h>

#define HIP_CHECK(stat)                                                        \
    {                                                                          \
        if(stat != hipSuccess)                                                 \
        {                                                                      \
            std::cerr << "Error: hip error in line " << __LINE__ << std::endl; \
            return -1;                                                         \
        }                                                                      \
    }

#define ROCSPARSE_CHECK(stat)                                                        \
    {                                                                                \
        if(stat != rocsparse_status_success)                                         \
        {                                                                            \
            std::cerr << "Error: rocsparse error in line " << __LINE__ << std::endl; \
            return -1;                                                               \
        }                                                                            \
    }

int main(int argc, char* argv[])
{
    // Query device
    int ndev;
    HIP_CHECK(hipGetDeviceCount(&ndev));

    if(ndev < 1)
    {
        std::cerr << "No HIP device found" << std::endl;
        return -1;
    }

    // Query device properties
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));

    std::cout << "Device: " << prop.name << std::endl;

    // rocSPARSE handle
    rocsparse_handle handle;
    ROCSPARSE_CHECK(rocsparse_create_handle(&handle));

    // Print rocSPARSE version and revision
    int  ver;
    char rev[64];

    ROCSPARSE_CHECK(rocsparse_get_version(handle, &ver));
    ROCSPARSE_CHECK(rocsparse_get_git_rev(handle, rev));

    std::cout << "rocSPARSE version: " << ver / 100000 << "." << ver / 100 % 1000 << "."
              << ver % 100 << "-" << rev << std::endl;

    // Input data

    // Number of unknowns
    const rocsparse_int m = 4;

    // Number of batches and stride
    const rocsparse_int batch_count  = 2;
    const rocsparse_int batch_stride = 543;

    double ds[m * batch_stride];
    double dl[m * batch_stride];
    double d[m * batch_stride];
    double du[m * batch_stride];
    double dw[m * batch_stride];
    double B[m * batch_stride];

    // Penta-diagonal matrix A
    ds[batch_stride * 0 + 0] = 0;
    ds[batch_stride * 1 + 0] = 0;
    ds[batch_stride * 2 + 0] = 8;
    ds[batch_stride * 3 + 0] = 12;
    dl[batch_stride * 0 + 0] = 0;
    dl[batch_stride * 1 + 0] = 4;
    dl[batch_stride * 2 + 0] = 9;
    dl[batch_stride * 3 + 0] = 13;
    d[batch_stride * 0 + 0]  = 1;
    d[batch_stride * 1 + 0]  = 5;
    d[batch_stride * 2 + 0]  = 10;
    d[batch_stride * 3 + 0]  = 14;
    du[batch_stride * 0 + 0] = 2;
    du[batch_stride * 1 + 0] = 6;
    du[batch_stride * 2 + 0] = 11;
    du[batch_stride * 3 + 0] = 0;
    dw[batch_stride * 0 + 0] = 3;
    dw[batch_stride * 1 + 0] = 7;
    dw[batch_stride * 2 + 0] = 0;
    dw[batch_stride * 3 + 0] = 0;
    B[batch_stride * 0 + 0]  = 1;
    B[batch_stride * 1 + 0]  = 2;
    B[batch_stride * 2 + 0]  = 3;
    B[batch_stride * 3 + 0]  = 4;

    // Penta-diagonal matrix B
    ds[batch_stride * 0 + 1] = 0;
    ds[batch_stride * 1 + 1] = 0;
    ds[batch_stride * 2 + 1] = 22;
    ds[batch_stride * 3 + 1] = 26;
    dl[batch_stride * 0 + 1] = 0;
    dl[batch_stride * 1 + 1] = 18;
    dl[batch_stride * 2 + 1] = 23;
    dl[batch_stride * 3 + 1] = 27;
    d[batch_stride * 0 + 1]  = 15;
    d[batch_stride * 1 + 1]  = 19;
    d[batch_stride * 2 + 1]  = 24;
    d[batch_stride * 3 + 1]  = 28;
    du[batch_stride * 0 + 1] = 16;
    du[batch_stride * 1 + 1] = 20;
    du[batch_stride * 2 + 1] = 25;
    du[batch_stride * 3 + 1] = 0;
    dw[batch_stride * 0 + 1] = 17;
    dw[batch_stride * 1 + 1] = 21;
    dw[batch_stride * 2 + 1] = 0;
    dw[batch_stride * 3 + 1] = 0;
    B[batch_stride * 0 + 1]  = 5;
    B[batch_stride * 1 + 1]  = 6;
    B[batch_stride * 2 + 1]  = 7;
    B[batch_stride * 3 + 1]  = 8;

    double G[m * batch_stride];

    // Offload data to device
    double* d_ds;
    double* d_dl;
    double* d_d;
    double* d_du;
    double* d_dw;
    double* d_X;

    HIP_CHECK(hipMalloc((void**)&d_ds, sizeof(double) * m * batch_stride));
    HIP_CHECK(hipMalloc((void**)&d_dl, sizeof(double) * m * batch_stride));
    HIP_CHECK(hipMalloc((void**)&d_d, sizeof(double) * m * batch_stride));
    HIP_CHECK(hipMalloc((void**)&d_du, sizeof(double) * m * batch_stride));
    HIP_CHECK(hipMalloc((void**)&d_dw, sizeof(double) * m * batch_stride));
    HIP_CHECK(hipMalloc((void**)&d_X, sizeof(double) * m * batch_stride));

    HIP_CHECK(hipMemcpy(d_ds, ds, sizeof(double) * m * batch_stride, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_dl, dl, sizeof(double) * m * batch_stride, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_d, d, sizeof(double) * m * batch_stride, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_du, du, sizeof(double) * m * batch_stride, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_dw, dw, sizeof(double) * m * batch_stride, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_X, B, sizeof(double) * m * batch_stride, hipMemcpyHostToDevice));

    // Query buffer size
    size_t buffer_size;

    ROCSPARSE_CHECK(rocsparse_dgpsv_interleaved_batch_buffer_size(handle,
                                                                  rocsparse_gpsv_interleaved_alg_qr,
                                                                  m,
                                                                  d_ds,
                                                                  d_dl,
                                                                  d_d,
                                                                  d_du,
                                                                  d_dw,
                                                                  d_X,
                                                                  batch_count,
                                                                  batch_stride,
                                                                  &buffer_size));

    // Allocate temporary buffer
    void* temp_buffer;
    HIP_CHECK(hipMalloc(&temp_buffer, buffer_size));

    // Run gpsv
    ROCSPARSE_CHECK(rocsparse_dgpsv_interleaved_batch(handle,
                                                      rocsparse_gpsv_interleaved_alg_qr,
                                                      m,
                                                      d_ds,
                                                      d_dl,
                                                      d_d,
                                                      d_du,
                                                      d_dw,
                                                      d_X,
                                                      batch_count,
                                                      batch_stride,
                                                      temp_buffer));

    // Free temporary buffer
    HIP_CHECK(hipFree(temp_buffer));

    // Copy result back to the host
    HIP_CHECK(hipMemcpy(G, d_X, sizeof(double) * m * batch_stride, hipMemcpyDeviceToHost));

    // Print result
    for(int b = 0; b < batch_count; ++b)
    {
        printf("Batch #%d:", b);
        for(int i = 0; i < m; ++i)
        {
            printf(" %lf", G[batch_stride * i + b]);
        }
        printf("\n");
    }

    // Clear rocSPARSE
    ROCSPARSE_CHECK(rocsparse_destroy_handle(handle));

    // Clear device memory
    HIP_CHECK(hipFree(d_ds));
    HIP_CHECK(hipFree(d_dl));
    HIP_CHECK(hipFree(d_d));
    HIP_CHECK(hipFree(d_du));
    HIP_CHECK(hipFree(d_dw));
    HIP_CHECK(hipFree(d_X));

    return 0;
}
