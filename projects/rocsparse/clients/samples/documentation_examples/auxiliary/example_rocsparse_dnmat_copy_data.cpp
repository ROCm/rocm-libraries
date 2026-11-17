/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights Reserved.
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
    // X is a 3 x 2 matrix stored in column order:
    //
    //     X = [ 1 4 ]
    //         [ 2 5 ]
    //         [ 3 6 ]
    int64_t            m     = 3;
    int64_t            n     = 2;
    std::vector<float> hX    = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    int64_t            ld_X  = m;
    rocsparse_order    ord_X = rocsparse_order_column;

    // Y is a 3 x 2 matrix stored in row order
    std::vector<float> hY(m * n);
    int64_t            ld_Y  = n;
    rocsparse_order    ord_Y = rocsparse_order_row;

    // Scaling factor, in host memory
    float alpha = 2.0f;

    // Offload data to device
    float* dX;
    float* dY;
    HIP_CHECK(hipMalloc(&dX, sizeof(float) * m * n));
    HIP_CHECK(hipMalloc(&dY, sizeof(float) * m * n));
    HIP_CHECK(hipMemcpy(dX, hX.data(), sizeof(float) * m * n, hipMemcpyHostToDevice));

    rocsparse_handle handle;
    ROCSPARSE_CHECK(rocsparse_create_handle(&handle));

    // Create the dense matrices X and Y
    rocsparse_dnmat_descr X;
    rocsparse_dnmat_descr Y;
    ROCSPARSE_CHECK(
        rocsparse_create_dnmat_descr(&X, m, n, ld_X, dX, rocsparse_datatype_f32_r, ord_X));
    ROCSPARSE_CHECK(
        rocsparse_create_dnmat_descr(&Y, m, n, ld_Y, dY, rocsparse_datatype_f32_r, ord_Y));

    // Create the scaling factor
    rocsparse_dnvec_descr scale;
    ROCSPARSE_CHECK(rocsparse_dnvec_descr_create_scalar(handle,
                                                        &scale,
                                                        rocsparse_pointer_mode_host,
                                                        rocsparse_datatype_f32_r,
                                                        &alpha,
                                                        nullptr,
                                                        nullptr));

    // Compute Y = alpha * X
    ROCSPARSE_CHECK(rocsparse_dnmat_copy_data(handle, scale, X, Y, nullptr));

    // Copy result back to host
    HIP_CHECK(hipMemcpy(hY.data(), dY, sizeof(float) * m * n, hipMemcpyDeviceToHost));

    // Y = [ 2  8 ]
    //     [ 4 10 ]
    //     [ 6 12 ]
    std::cout << "Y" << std::endl;
    for(int64_t i = 0; i < m; ++i)
    {
        for(int64_t j = 0; j < n; ++j)
        {
            std::cout << hY[i * ld_Y + j] << " ";
        }
        std::cout << std::endl;
    }

    // Clear rocSPARSE
    ROCSPARSE_CHECK(rocsparse_destroy_dnvec_descr(scale));
    ROCSPARSE_CHECK(rocsparse_destroy_dnmat_descr(X));
    ROCSPARSE_CHECK(rocsparse_destroy_dnmat_descr(Y));
    ROCSPARSE_CHECK(rocsparse_destroy_handle(handle));

    // Clear device memory
    HIP_CHECK(hipFree(dX));
    HIP_CHECK(hipFree(dY));

    return 0;
}
//! [doc example]
