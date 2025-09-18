/*! \file */
/* ************************************************************************
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc. All rights Reserved.
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
void testing_inverse_permutation_bad_arg(const Arguments& arg)
{
    rocsparse_local_handle local_handle;
    rocsparse_handle       handle = local_handle;
    rocsparse_int          n      = 1;
    rocsparse_int*         p      = (rocsparse_int*)0x4;
    rocsparse_int*         q      = (rocsparse_int*)0x4;
    rocsparse_index_base   base   = rocsparse_index_base_zero;
    bad_arg_analysis(rocsparse_inverse_permutation, handle, n, p, q, base);
}

template <typename T>
void testing_inverse_permutation(const Arguments& arg)
{
    rocsparse_int        N    = arg.N;
    rocsparse_index_base base = arg.baseA;

    // Create rocsparse handle
    rocsparse_local_handle handle;

    // Allocate host memory
    host_dense_vector<rocsparse_int> hp(N);

    for(rocsparse_int i = 0; i < N; ++i)
    {
        hp[i] = i + base;
    }

    for(rocsparse_int k = 0; k < std::min(static_cast<rocsparse_int>(20), N / 2); ++k)
    {
        rocsparse_int i = random_generator_exact<rocsparse_int>(0, N - 1);
        rocsparse_int j = random_generator_exact<rocsparse_int>(0, N - 1);
        if(i != j)
        {
            std::swap(hp[i], hp[j]);
        }
    }
    device_dense_vector<rocsparse_int> dp(hp);
    device_dense_vector<rocsparse_int> dq(N);

    if(arg.unit_check)
    {
        host_dense_vector<rocsparse_int> hq(N);

        for(rocsparse_int i = 0; i < N; ++i)
        {
            hq[hp[i] - base] = i + base;
        }
        CHECK_ROCSPARSE_ERROR(testing::rocsparse_inverse_permutation(handle, N, dp, dq, base));
        hq.unit_check(dq);
    }

    if(arg.timing)
    {

        const double gpu_time_used = rocsparse_clients::run_benchmark(
            arg, rocsparse_inverse_permutation, handle, N, dp, dq, base);

        double gbyte_count = inverse_permutation_gbyte_count<T>(N);
        double gpu_gbyte   = get_gpu_gbyte(gpu_time_used, gbyte_count);
        display_timing_info(display_key_t::N,
                            N,
                            display_key_t::bandwidth,
                            gpu_gbyte,
                            display_key_t::time_ms,
                            get_gpu_time_msec(gpu_time_used));
    }
}

#define INSTANTIATE(TYPE)                                                          \
    template void testing_inverse_permutation_bad_arg<TYPE>(const Arguments& arg); \
    template void testing_inverse_permutation<TYPE>(const Arguments& arg)
INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex);
void testing_inverse_permutation_extra(const Arguments& arg) {}
