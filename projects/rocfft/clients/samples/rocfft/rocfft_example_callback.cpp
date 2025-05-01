/******************************************************************************
* Copyright (C) 2021 - 2022 Advanced Micro Devices, Inc. All rights reserved.
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
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
* THE SOFTWARE.
*******************************************************************************/

#include "rocfft/rocfft.h"
#include <hip/hip_complex.h>
#include <hip/hip_runtime.h>
#include <hip/hip_vector_types.h>
#include <iostream>
#include <math.h>
#include <stdexcept>
#include <vector>

// example of using load/store callbacks with rocfft

struct load_cbdata
{
    double2* filter;
    double   scale;
};

__device__ double2 load_callback(double2* input, size_t offset, void* cbdata, void* sharedMem)
{
    auto data = static_cast<load_cbdata*>(cbdata);

    // multiply each element by filter element and scale
    return hipCmul(hipCmul(input[offset], data->filter[offset]),
                   make_hipDoubleComplex(data->scale, data->scale));
}
__device__ auto load_callback_dev = load_callback;

int main()
{

    const size_t N = 8;

    std::vector<double2> cx(N), filter(N);

    // initialize data and filter
    for(size_t i = 0; i < N; i++)
    {
        cx[i].x     = i;
        cx[i].y     = i;
        filter[i].x = rand() / static_cast<double>(RAND_MAX);
        filter[i].y = 0;
    }

    // rocfft gpu compute
    // ==================

    if(rocfft_setup() != rocfft_status_success)
        throw std::runtime_error("rocfft_setup failed.");

    size_t Nbytes = N * sizeof(double2);

    // Create HIP device object.
    double2 *x, *filter_dev;

    // create buffers
    if(hipMalloc(&x, Nbytes) != hipSuccess)
        throw std::runtime_error("hipMalloc failed.");

    if(hipMalloc(&filter_dev, Nbytes) != hipSuccess)
        throw std::runtime_error("hipMalloc failed.");

    //  Copy data to device
    hipError_t hip_status = hipMemcpy(x, cx.data(), Nbytes, hipMemcpyHostToDevice);
    if(hip_status != hipSuccess)
        throw std::runtime_error("hipMemcpy failed.");

    hip_status = hipMemcpy(filter_dev, filter.data(), Nbytes, hipMemcpyHostToDevice);
    if(hip_status != hipSuccess)
        throw std::runtime_error("hipMemcpy failed.");

    // Create plan
    rocfft_plan plan   = nullptr;
    size_t      length = N;
    if(rocfft_plan_create(&plan,
                          rocfft_placement_inplace,
                          rocfft_transform_type_complex_forward,
                          rocfft_precision_double,
                          1,
                          &length,
                          1,
                          nullptr)
       != rocfft_status_success)
        throw std::runtime_error("rocfft_plan_create failed.");

    // Check if the plan requires a work buffer
    size_t work_buf_size = 0;
    if(rocfft_plan_get_work_buffer_size(plan, &work_buf_size) != rocfft_status_success)
        throw std::runtime_error("rocfft_plan_get_work_buffer_size failed.");
    void*                 work_buf = nullptr;
    rocfft_execution_info info     = nullptr;
    if(rocfft_execution_info_create(&info) != rocfft_status_success)
        throw std::runtime_error("rocfft_execution_info_create failed.");
    if(work_buf_size)
    {
        if(hipMalloc(&work_buf, work_buf_size) != hipSuccess)
            throw std::runtime_error("hipMalloc failed.");

        if(rocfft_execution_info_set_work_buffer(info, work_buf, work_buf_size)
           != rocfft_status_success)
            throw std::runtime_error("rocfft_execution_info_set_work_buffer failed.");
    }

    // Prepare callback
    load_cbdata cbdata_host;
    cbdata_host.filter = filter_dev;
    cbdata_host.scale  = 1.0 / static_cast<double>(N);

    void* cbdata_dev;
    if(hipMalloc(&cbdata_dev, sizeof(load_cbdata)) != hipSuccess)
        throw std::runtime_error("hipMalloc failed.");

    hip_status = hipMemcpy(cbdata_dev, &cbdata_host, sizeof(load_cbdata), hipMemcpyHostToDevice);
    if(hip_status != hipSuccess)
        throw std::runtime_error("hipMemcpy failed.");

    // Get a properly-typed host pointer to the device function, as
    // rocfft_execution_info_set_load_callback expects void*.
    void* cbptr_host = nullptr;
    hip_status = hipMemcpyFromSymbol(&cbptr_host, HIP_SYMBOL(load_callback_dev), sizeof(void*));
    if(hip_status != hipSuccess)
        throw std::runtime_error("hipMemcpyFromSymbol failed.");

    // set callback
    if(rocfft_execution_info_set_load_callback(info, &cbptr_host, &cbdata_dev, 0)
       != rocfft_status_success)
        throw std::runtime_error("rocfft_execution_info_set_load_callback failed.");

    // Execute plan
    if(rocfft_execute(plan, (void**)&x, nullptr, info) != rocfft_status_success)
        throw std::runtime_error("rocfft_execute failed.");

    // Clean up work buffer
    if(work_buf_size)
    {
        if(hipFree(work_buf) != hipSuccess)
            throw std::runtime_error("hipFree failed.");
        if(rocfft_execution_info_destroy(info) != rocfft_status_success)
            throw std::runtime_error("rocfft_execution_info_destroy failed.");
        info = nullptr;
    }

    // Destroy plan
    if(rocfft_plan_destroy(plan) != rocfft_status_success)
        throw std::runtime_error("rocfft_plan_destroy failed.");
    plan = nullptr;

    // Copy result back to host
    std::vector<double2> y(N);
    hip_status = hipMemcpy(&y[0], x, Nbytes, hipMemcpyDeviceToHost);
    if(hip_status != hipSuccess)
        throw std::runtime_error("hipMemcpy failed.");

    for(size_t i = 0; i < N; i++)
    {
        std::cout << "element " << i << " input:  (" << cx[i].x << "," << cx[i].y << ")"
                  << " output: (" << y[i].x << "," << y[i].y << ")" << std::endl;
    }

    if(hipFree(cbdata_dev) != hipSuccess)
        throw std::runtime_error("hipFree failed.");
    if(hipFree(filter_dev) != hipSuccess)
        throw std::runtime_error("hipFree failed.");
    if(hipFree(x) != hipSuccess)
        throw std::runtime_error("hipFree failed.");

    if(rocfft_cleanup() != rocfft_status_success)
        throw std::runtime_error("rocfft_cleanup failed.");

    return 0;
}
