/* ************************************************************************
 * Copyright (C) 20256-2026 Advanced Micro Devices, Inc. All rights Reserved.
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
    const int64_t            M        = 3;
    const int64_t            N        = 2;
    const rocsparse_order    order    = rocsparse_order_column;
    const rocsparse_datatype datatype = rocsparse_datatype_f32_r;
    const size_t             nbytes   = sizeof(float) * M * N;

    const int64_t x_ld = (order == rocsparse_order_column) ? M : N;
    const int64_t y_ld = (order == rocsparse_order_column) ? M : N;

    float* x_host = (float*)malloc(nbytes);
    float* y_host = (float*)malloc(nbytes);

    memset(x_host, 255 - 1, nbytes);
    memset(y_host, 255 - 1, nbytes);

    for(int64_t j = 0; j < N; ++j)
        for(int64_t i = 0; i < M; ++i)
            x_host[j * x_ld + i] = i + M * j;

    rocsparse_error*      p_error = NULL;
    rocsparse_dnmat_descr Y       = NULL;
    rocsparse_dnmat_descr X       = NULL;
    rocsparse_handle      handle  = NULL;
    void*                 x       = NULL;
    void*                 y       = NULL;

    hipStream_t stream = NULL;

    HIP_CHECK(hipStreamCreate(&stream));
    ROCSPARSE_CHECK(rocsparse_handle_create(&handle, stream, p_error));

    //
    // Create Y.
    //
    HIP_CHECK(hipMallocAsync(&y, nbytes, stream));
    ROCSPARSE_CHECK(rocsparse_create_dnmat_descr(
        &Y, M, N, (order == rocsparse_order_column) ? M : N, y, datatype, order));

    //
    // Create x.
    //
    HIP_CHECK(hipMallocAsync(&x, nbytes, stream));
    ROCSPARSE_CHECK(rocsparse_create_dnmat_descr(
        &X, M, N, (order == rocsparse_order_column) ? M : N, x, datatype, order));

    //
    // Init x.
    //
    HIP_CHECK(hipMemcpyAsync(x, x_host, nbytes, hipMemcpyHostToDevice, stream));

    //
    // Copy data with no scalar.
    //
    rocsparse_const_dnvec_descr no_scale = nullptr;
    ROCSPARSE_CHECK(rocsparse_dnmat_copy_data(handle, no_scale, X, Y, p_error));

    //
    // Verify.
    //

    HIP_CHECK(hipMemcpyAsync(y_host, y, nbytes, hipMemcpyDeviceToHost, stream));

    HIP_CHECK(hipStreamSynchronize(stream));

    ROCSPARSE_CHECK(rocsparse_destroy_dnmat_descr(Y));
    ROCSPARSE_CHECK(rocsparse_destroy_dnmat_descr(X));
    HIP_CHECK(hipFreeAsync(y, stream));
    HIP_CHECK(hipFreeAsync(x, stream));
    ROCSPARSE_CHECK(rocsparse_destroy_handle(handle));
    HIP_CHECK(hipStreamDestroy(stream));
    free(x_host);

    int failed = 0;
    for(int64_t j = 0; j < N; ++j)
    {
        for(int64_t i = 0; i < M; ++i)
        {
            if(y_host[j * y_ld + i] != i + M * j)
            {
                failed = 1;
                break;
            }
        }
    }

    free(y_host);
    return failed;
}
//! [doc example]
