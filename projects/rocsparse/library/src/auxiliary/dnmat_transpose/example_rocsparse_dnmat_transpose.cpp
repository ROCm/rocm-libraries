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
    const int64_t M = 3;
    const int64_t N = 2;

    const rocsparse_order order = rocsparse_order_column;

    const int64_t x_ld = (order == rocsparse_order_column) ? M : N;
    const int64_t y_ld = (order == rocsparse_order_column) ? N : M;

    const rocsparse_datatype datatype = rocsparse_datatype_f32_r;
    const size_t             nbytes   = sizeof(float) * M * N;

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

    HIP_CHECK(hipMallocAsync(&y, nbytes, stream));
    ROCSPARSE_CHECK(rocsparse_create_dnmat_descr(&Y, N, M, y_ld, y, datatype, order));

    HIP_CHECK(hipMallocAsync(&x, nbytes, stream));
    HIP_CHECK(hipMemcpyAsync(x, x_host, nbytes, hipMemcpyHostToDevice, stream));
    ROCSPARSE_CHECK(rocsparse_create_dnmat_descr(&X, M, N, x_ld, x, datatype, order));

    //
    // Copy data with no scalar.
    //
    rocsparse_const_dnvec_descr no_scale = nullptr;
    ROCSPARSE_CHECK(rocsparse_dnmat_transpose(handle, no_scale, X, Y, p_error));

    //
    // Verify.
    //

    HIP_CHECK(hipMemcpyAsync(y_host, y, nbytes, hipMemcpyDeviceToHost, stream));

    HIP_CHECK(hipStreamSynchronize(stream));

    int failed = 0;
    for(int64_t j = 0; j < M; ++j)
    {
        for(int64_t i = 0; i < N; ++i)
        {
            if(y_host[j * y_ld + i] != j + M * i)
            {
                failed = 1;
                break;
            }
        }
    }

    ROCSPARSE_CHECK(rocsparse_destroy_dnmat_descr(Y));
    HIP_CHECK(hipFreeAsync(y, stream));

    ROCSPARSE_CHECK(rocsparse_destroy_dnmat_descr(X));
    HIP_CHECK(hipFreeAsync(x, stream));

    free(y_host);
    free(x_host);

    ROCSPARSE_CHECK(rocsparse_destroy_handle(handle));
    if(stream)
    {
        HIP_CHECK(hipStreamDestroy(stream));
    }
    return failed;
}
//! [doc example]
