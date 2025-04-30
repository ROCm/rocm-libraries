/* ************************************************************************
 * Copyright (C) 2018-2019 Advanced Micro Devices, Inc. All rights Reserved.
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

#pragma once
#ifndef TESTING_IDENTITY_HPP
#define TESTING_IDENTITY_HPP

#include "display.hpp"
#include "flops.hpp"
#include "gbyte.hpp"
#include "hipsparse.hpp"
#include "hipsparse_arguments.hpp"
#include "hipsparse_test_unique_ptr.hpp"
#include "unit.hpp"
#include "utility.hpp"

#include <algorithm>
#include <hipsparse.h>

using namespace hipsparse;
using namespace hipsparse_test;

void testing_identity_bad_arg(void)
{
#if(!defined(CUDART_VERSION))
    int n         = 100;
    int safe_size = 100;

    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t              handle = unique_ptr_handle->handle;

    auto p_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};

    int* p = (int*)p_managed.get();

    verify_hipsparse_status_invalid_pointer(
        hipsparseCreateIdentityPermutation(handle, n, (int*)nullptr), "Error: p is nullptr");
    verify_hipsparse_status_invalid_handle(hipsparseCreateIdentityPermutation(nullptr, n, p));
#endif
}

hipsparseStatus_t testing_identity(Arguments argus)
{
#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
    int n = argus.N;

    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t              handle = unique_ptr_handle->handle;

    // Host structures
    std::vector<int> hp(n);
    std::vector<int> hp_gold(n);

    // create_identity_permutation on host
    for(int i = 0; i < n; ++i)
    {
        hp_gold[i] = i;
    }

    // Allocate memory on the device
    auto dp_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * n), device_free};

    int* dp = (int*)dp_managed.get();

    if(argus.unit_check)
    {
        CHECK_HIPSPARSE_ERROR(hipsparseCreateIdentityPermutation(handle, n, dp));

        // Copy output from device to host
        CHECK_HIP_ERROR(hipMemcpy(hp.data(), dp, sizeof(int) * n, hipMemcpyDeviceToHost));

        // Unit check
        unit_check_general(1, n, 1, hp_gold.data(), hp.data());
    }

    if(argus.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = argus.iters;

        // Warm up
        for(int iter = 0; iter < number_cold_calls; ++iter)
        {
            CHECK_HIPSPARSE_ERROR(hipsparseCreateIdentityPermutation(handle, n, dp));
        }

        double gpu_time_used = get_time_us();

        // Performance run
        for(int iter = 0; iter < number_hot_calls; ++iter)
        {
            CHECK_HIPSPARSE_ERROR(hipsparseCreateIdentityPermutation(handle, n, dp));
        }

        gpu_time_used = (get_time_us() - gpu_time_used) / number_hot_calls;

        double gbyte_count = identity_gbyte_count(n);
        double gpu_gbyte   = get_gpu_gbyte(gpu_time_used, gbyte_count);

        display_timing_info(display_key_t::N,
                            n,
                            display_key_t::bandwidth,
                            gpu_gbyte,
                            display_key_t::time_ms,
                            get_gpu_time_msec(gpu_time_used));
    }
#endif

    return HIPSPARSE_STATUS_SUCCESS;
}

#endif // TESTING_IDENTITY_HPP
