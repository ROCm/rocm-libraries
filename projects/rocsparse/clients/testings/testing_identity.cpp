/*! \file */
/* ************************************************************************
 * Copyright (C) 2019-2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "testing.hpp"

template <typename T>
void testing_identity_bad_arg(const Arguments& arg)
{
    rocsparse_local_handle local_handle;
    rocsparse_handle       handle = local_handle;
    rocsparse_int          n      = 100;
    rocsparse_int*         p      = (rocsparse_int*)0x4;

    bad_arg_analysis(rocsparse_create_identity_permutation, handle, n, p);
}

template <typename T>
void testing_identity(const Arguments& arg)
{
    rocsparse_int N = arg.N;

    // Create rocsparse handle
    rocsparse_local_handle handle(arg);

    // Allocate host memory
    host_vector<rocsparse_int> hp(N);
    host_vector<rocsparse_int> hp_gold(N);

    // Allocate device memory
    device_vector<rocsparse_int> dp(N);

    if(arg.unit_check)
    {
        CHECK_ROCSPARSE_ERROR(testing::rocsparse_create_identity_permutation(handle, N, dp));

        // Copy output to host
        CHECK_HIP_ERROR(hipMemcpy(hp, dp, sizeof(rocsparse_int) * N, hipMemcpyDeviceToHost));

        // CPU identity
        for(rocsparse_int i = 0; i < N; ++i)
        {
            hp_gold[i] = i;
        }

        hp_gold.unit_check(hp);
    }

    if(arg.timing)
    {

        const double gpu_time_used = rocsparse_clients::run_benchmark(
            arg, rocsparse_create_identity_permutation, handle, N, dp);

        double gbyte_count = identity_gbyte_count<T>(N);
        double gpu_gbyte   = get_gpu_gbyte(gpu_time_used, gbyte_count);
        display_timing_info(display_key_t::N,
                            N,
                            display_key_t::bandwidth,
                            gpu_gbyte,
                            display_key_t::time_ms,
                            get_gpu_time_msec(gpu_time_used));
    }
}

#define INSTANTIATE(TYPE)                                               \
    template void testing_identity_bad_arg<TYPE>(const Arguments& arg); \
    template void testing_identity<TYPE>(const Arguments& arg)
INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex);
void testing_identity_extra(const Arguments& arg) {}
