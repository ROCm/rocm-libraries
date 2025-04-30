/******************************************************************************
* Copyright (C) 2016 - 2022 Advanced Micro Devices, Inc. All rights reserved.
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

#include <iostream>
#include <math.h>
#include <vector>

#include "rocfft/rocfft.h"
#include <hip/hip_runtime_api.h>
#include <hip/hip_vector_types.h>

int main()
{
    // For size N >= 8192, temporary buffer is required to allocated
    const size_t N = 64 * 2048;

    std::vector<float2> cx(N);

    for(size_t i = 0; i < N; i++)
    {
        cx[i].x = i + (i % 3) - (i % 7);
        cx[i].y = 0;
    }

    // rocfft gpu compute
    // ========================================

    if(rocfft_setup() != rocfft_status_success)
        throw std::runtime_error("rocfft_setup failed.");

    size_t Nbytes = N * sizeof(float2);

    // Create HIP device object.
    float2* x;
    if(hipMalloc(&x, Nbytes) != hipSuccess)
        throw std::runtime_error("hipMalloc failed.");

    //  Copy data to device
    if(hipMemcpy(x, &cx[0], Nbytes, hipMemcpyHostToDevice) != hipSuccess)
        throw std::runtime_error("hipMemcpy failed.");

    // Create plan
    rocfft_plan plan   = nullptr;
    size_t      length = N;
    if(rocfft_plan_create(&plan,
                          rocfft_placement_inplace,
                          rocfft_transform_type_complex_forward,
                          rocfft_precision_single,
                          1,
                          &length,
                          1,
                          nullptr)
       != rocfft_status_success)
        throw std::runtime_error("rocfft_plan_create failed.");

    // Setup work buffer
    void*  workBuffer     = nullptr;
    size_t workBufferSize = 0;
    if(rocfft_plan_get_work_buffer_size(plan, &workBufferSize) != rocfft_status_success)
        throw std::runtime_error("rocfft_plan_get_work_buffer_size failed.");

    // Setup exec info to pass work buffer to the library
    rocfft_execution_info info = nullptr;
    if(rocfft_execution_info_create(&info) != rocfft_status_success)
        throw std::runtime_error("rocfft_execution_info_create failed.");

    if(workBufferSize > 0)
    {
        printf("size of workbuffer=%d\n", (int)workBufferSize);
        if(hipMalloc(&workBuffer, workBufferSize) != hipSuccess)
            throw std::runtime_error("hipMalloc failed.");
        if(rocfft_execution_info_set_work_buffer(info, workBuffer, workBufferSize)
           != rocfft_status_success)
            throw std::runtime_error("rocfft_execution_info_set_work_buffer failed.");
    }

    // Execute plan
    if(rocfft_execute(plan, (void**)&x, nullptr, info) != rocfft_status_success)
        throw std::runtime_error("rocfft_execute failed.");
    if(hipDeviceSynchronize() != hipSuccess)
        throw std::runtime_error("hipDeviceSynchronize failed.");

    // Destroy plan
    if(rocfft_plan_destroy(plan) != rocfft_status_success)
        throw std::runtime_error("rocfft_plan_destroy failed.");
    plan = nullptr;

    if(workBuffer)
        if(hipFree(workBuffer) != hipSuccess)
            throw std::runtime_error("hipFree failed.");

    if(rocfft_execution_info_destroy(info) != rocfft_status_success)
        throw std::runtime_error("rocfft_execution_info_destroy failed.");
    info = nullptr;

    // Copy result back to host
    std::vector<float2> y(N);
    if(hipMemcpy(&y[0], x, Nbytes, hipMemcpyDeviceToHost) != hipSuccess)
        throw std::runtime_error("hipMemcpy failed.");

    for(size_t i = 0; i < N; i++)
    {
        std::cout << "element " << i << " input:  (" << cx[i].x << "," << cx[i].y << ")"
                  << " output: (" << y[i].x << "," << y[i].y << ")" << std::endl;
    }

    if(hipFree(x) != hipSuccess)
        throw std::runtime_error("hipFree failed.");

    if(rocfft_cleanup() != rocfft_status_success)
        throw std::runtime_error("rocfft_cleanup failed.");

    return 0;
}
