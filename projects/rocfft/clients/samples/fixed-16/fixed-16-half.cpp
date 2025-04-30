/******************************************************************************
* Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
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
#include <hip/hip_fp16.h>
#include <hip/hip_runtime_api.h>
#include <iostream>
#include <vector>

int main()
{

    const size_t N = 16;

    std::vector<_Float16_2> cx(N);

    for(size_t i = 0; i < N; i++)
    {
        cx[i].x = static_cast<_Float16>(i + (i % 3) - (i % 7));
        cx[i].y = 0;
    }

    // rocfft gpu compute
    // ========================================

    if(rocfft_setup() != rocfft_status_success)
        throw std::runtime_error("rocfft_setup failed.");

    size_t Nbytes = N * sizeof(_Float16_2);

    // Create HIP device object.
    _Float16_2* x = nullptr;
    if(hipMalloc(&x, Nbytes) != hipSuccess)
        throw std::runtime_error("hipMalloc failed.");

    //  Copy data to device
    if(hipMemcpy(x, &cx[0], Nbytes, hipMemcpyHostToDevice) != hipSuccess)
        throw std::runtime_error("hipMemcpy failed.");

    // Create plan
    rocfft_plan plan   = NULL;
    size_t      length = N;
    if(rocfft_plan_create(&plan,
                          rocfft_placement_inplace,
                          rocfft_transform_type_complex_forward,
                          rocfft_precision_half,
                          1,
                          &length,
                          1,
                          NULL)
       != rocfft_status_success)
        throw std::runtime_error("rocfft_plan_create failed.");

    // Check if the plan requires a work buffer
    size_t work_buf_size = 0;
    if(rocfft_plan_get_work_buffer_size(plan, &work_buf_size) != rocfft_status_success)
        throw std::runtime_error("rocfft_plan_get_work_buffer_size failed.");
    void*                 work_buf = nullptr;
    rocfft_execution_info info     = nullptr;
    if(work_buf_size)
    {
        if(rocfft_execution_info_create(&info) != rocfft_status_success)
            throw std::runtime_error("rocfft_execution_info_create failed.");
        if(hipMalloc(&work_buf, work_buf_size) != hipSuccess)
            throw std::runtime_error("hipMalloc failed.");
        if(rocfft_execution_info_set_work_buffer(info, work_buf, work_buf_size)
           != rocfft_status_success)
            throw std::runtime_error("rocfft_execution_info_set_work_buffer failed.");
    }

    // Execute plan
    if(rocfft_execute(plan, (void**)&x, NULL, info) != rocfft_status_success)
        throw std::runtime_error("rocfft_execute failed.");
    if(hipDeviceSynchronize() != hipSuccess)
        throw std::runtime_error("hipDeviceSynchronize failed.");

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
    std::vector<_Float16_2> y(N);
    if(hipMemcpy(&y[0], x, Nbytes, hipMemcpyDeviceToHost) != hipSuccess)
        throw std::runtime_error("hipMemcpy failed.");

    for(size_t i = 0; i < N; i++)
    {
        std::cout << "element " << i << " input:  (" << static_cast<double>(cx[i].x) << ","
                  << static_cast<double>(cx[i].y) << ")"
                  << " output: (" << static_cast<double>(y[i].x) << ","
                  << static_cast<double>(y[i].y) << ")" << std::endl;
    }

    if(hipFree(x) != hipSuccess)
        throw std::runtime_error("hipFree failed.");

    if(rocfft_cleanup() != rocfft_status_success)
        throw std::runtime_error("rocfft_cleanup failed.");

    return 0;
}
