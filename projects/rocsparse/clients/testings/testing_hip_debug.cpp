/*! \file */
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

#include "testing.hpp"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

#include <pthread.h>
#include <shared_mutex>
#include <stdio.h>
#include <stdlib.h>
#include <thread>

#include <map>

static std::shared_mutex                  s_shared_mutex{};
static std::map<std::thread::id, int32_t> s_tid2local_id{};
static int32_t                            get_local_id()
{
    std::unique_lock lock(s_shared_mutex);
    const int32_t    size                      = s_tid2local_id.size();
    s_tid2local_id[std::this_thread::get_id()] = size;
    return size;
}

template <typename T>
void testing_hip_debug_task_thread(void* usr)
{
    if(1)
    {
        int32_t           local_id = get_local_id();
        std::stringstream s;
        s << "// rocsparse_clients: " << __FUNCTION__ << ", local_id = " << local_id
          << ", tid = " << std::this_thread::get_id() << std::endl;
        std::cout << s.str();
    }

    const Arguments& arg = *(const Arguments*)usr;
    //    printf("Thread #%ld is working...\n", tid);

    rocsparse_int        M    = arg.M;
    rocsparse_int        nnz  = arg.nnz;
    rocsparse_index_base base = arg.baseA;
    std::cout << "nnz " << nnz << std::endl;
    std::cout << "M " << M << std::endl;
    T h_alpha = arg.get_alpha<T>();
    // Create rocsparse handle
    rocsparse_local_handle handle(arg);
    // Allocate host memory
    std::cout << "aaaaaaa " << __LINE__ << std::endl;
    host_vector<rocsparse_int> hx_ind(nnz);
    host_vector<T>             hx_val(nnz);
    host_vector<T>             hy_1(M);
    host_vector<T>             hy_2(M);
    host_vector<T>             hy_gold(M);
    std::cout << "aaaaaaa " << __LINE__ << std::endl;
    rocsparse_init_index(hx_ind, nnz, base, M + base);

#if 0

    // Initialize data on CPU
    // rocsparse_seedrand();
    std::cout << "aaaaaaa " << __LINE__ <<  std::endl;
    rocsparse_init<T>(hx_val, 1, nnz, 1, arg.convert_to_int);
    rocsparse_init<T>(hy_1, 1, M, 1, arg.convert_to_int);
    hy_2    = hy_1;
    std::cout << "aaaaaaa " << __LINE__ <<  std::endl;
    hy_gold = hy_1;

    // Allocate device memory
    std::cout << "aaaaaaa " << __LINE__ <<  std::endl;
    device_vector<rocsparse_int> dx_ind(nnz);
    device_vector<T>             dx_val(nnz);
    device_vector<T>             dy_1(M);
    device_vector<T>             dy_2(M);
    device_vector<T>             d_alpha(1);
    std::cout << "aaaaaaa " << __LINE__ <<  std::endl;

    // Copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dx_ind, hx_ind, sizeof(rocsparse_int) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dx_val, hx_val, sizeof(T) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dy_1, hy_1, sizeof(T) * M, hipMemcpyHostToDevice));
    std::cout << "aaaaaaa " << __LINE__ <<  std::endl;
    if(arg.unit_check)
    {
        // Copy data from CPU to device
        CHECK_HIP_ERROR(hipMemcpy(dy_2, hy_2, sizeof(T) * M, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));
    std::cout << "aaaaaaa " << __LINE__ <<  std::endl;

        // Pointer mode host
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
        CHECK_ROCSPARSE_ERROR(
			      rocsparse_axpyi<T>(handle, nnz, &h_alpha, dx_val, dx_ind, dy_1, base));
    std::cout << "aaaaaaa " << __LINE__ <<  std::endl;

        // Pointer mode device
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));
        CHECK_ROCSPARSE_ERROR(
            rocsparse_axpyi<T>(handle, nnz, d_alpha, dx_val, dx_ind, dy_2, base));

    std::cout << "aaaaaaa " << __LINE__ <<  std::endl;
        // Copy output to host
        CHECK_HIP_ERROR(hipMemcpy(hy_1, dy_1, sizeof(T) * M, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hy_2, dy_2, sizeof(T) * M, hipMemcpyDeviceToHost));

    std::cout << "aaaaaaa " << __LINE__ <<  std::endl;

        // CPU axpyi
        host_axpby<T, rocsparse_int, T, T>(M, nnz, h_alpha, hx_val, hx_ind, (T)1.0, hy_gold, base);

    std::cout << "aaaaaaa " << __LINE__ <<  std::endl;
        hy_gold.unit_check(hy_1);
        hy_gold.unit_check(hy_2);
    }
#endif
}

template <typename T>
void* testing_hip_debug_task(void* usr)
{
    testing_hip_debug_task_thread<T>(usr);
    pthread_exit(NULL);
}

template <typename T>
void testing_hip_debug_bad_arg(const Arguments& arg)
{
}

template <typename T>
void testing_hip_debug(const Arguments& arg)
{
    const int32_t          num_threads = 8;
    std::vector<pthread_t> threads(num_threads);

    // 1. Create threads in a loop
    for(int32_t t = 0; t < num_threads; t++)
    {
        const auto rc = pthread_create(&threads[t], NULL, testing_hip_debug_task<T>, ((void*)&arg));
        CHECK_ROCSPARSE_ERROR((rc) ? rocsparse_status_internal_error : rocsparse_status_success);
    }
    // 2. Join threads in a loop
    for(int32_t t = 0; t < num_threads; t++)
    {
        pthread_join(threads[t], NULL);
    }
}

#define INSTANTIATE(TYPE)                                                \
    template void testing_hip_debug_bad_arg<TYPE>(const Arguments& arg); \
    template void testing_hip_debug<TYPE>(const Arguments& arg)
INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex);
void testing_hip_debug_extra(const Arguments& arg) {}
