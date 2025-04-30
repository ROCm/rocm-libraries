// Copyright (C) 2020 - 2023 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "rocfft/rocfft.h"
#include <cassert>
#include <hip/hip_runtime_api.h>
#include <iostream>
#include <stdexcept>
#include <vector>

struct fft_fixture_t
{
    std::vector<double2>  cpu_buf;
    double2*              gpu_buf = nullptr;
    hipStream_t           stream  = nullptr;
    rocfft_execution_info info    = nullptr;
    rocfft_plan           plan    = nullptr;
};

int main(int argc, char* argv[])
{
    std::cout << "rocfft example of 2 inplace transforms with 2 streams.\n" << std::endl;

    size_t        length      = 8;
    size_t        total_bytes = length * sizeof(double2);
    hipError_t    hip_status;
    rocfft_status fft_status;

    fft_fixture_t ffts[2];

    /// preparation
    if(rocfft_setup() != rocfft_status_success)
        throw std::runtime_error("rocfft_setup failed.");
    for(auto& it : ffts)
    {
        // create cpu buffer
        it.cpu_buf.resize(length);

        // init cpu buffer...

        // create gpu buffer
        if(hipMalloc(&(it.gpu_buf), total_bytes) != hipSuccess)
            throw std::runtime_error("hipMalloc failed.");

        // copy host to device
        if(hipMemcpy(it.gpu_buf, it.cpu_buf.data(), total_bytes, hipMemcpyHostToDevice)
           != hipSuccess)
            throw std::runtime_error("hipMemcpy failed.");

        // create stream
        if(hipStreamCreate(&(it.stream)) != hipSuccess)
            throw std::runtime_error("hipStreamCreate failed.");

        // create execution info
        fft_status = rocfft_execution_info_create(&(it.info));
        if(fft_status != rocfft_status_success)
            throw std::runtime_error("rocfft_execution_info_create failed.");

        // set stream
        // NOTE: The stream must be of type hipStream_t.
        // It is an error to pass the address of a hipStream_t object.
        fft_status = rocfft_execution_info_set_stream(it.info, it.stream);
        if(fft_status != rocfft_status_success)
            throw std::runtime_error("rocfft_execution_info_set_stream failed.");

        // create plan
        fft_status = rocfft_plan_create(&it.plan,
                                        rocfft_placement_inplace,
                                        rocfft_transform_type_complex_forward,
                                        rocfft_precision_double,
                                        1,
                                        &length,
                                        1,
                                        nullptr);
        if(fft_status != rocfft_status_success)
            throw std::runtime_error("rocfft_plan_create failed.");

        size_t work_buf_size = 0;
        fft_status           = rocfft_plan_get_work_buffer_size(it.plan, &work_buf_size);
        if(fft_status != rocfft_status_success)
            throw std::runtime_error("rocfft_plan_get_work_buffer_size failed.");

        assert(work_buf_size == 0); // simple 1D inplace fft doesn't need extra working buffer
    }

    /// execution
    for(auto& it : ffts)
    {
        fft_status = rocfft_execute(it.plan, (void**)&(it.gpu_buf), (void**)&(it.gpu_buf), nullptr);
        if(fft_status != rocfft_status_success)
            throw std::runtime_error("rocfft_execute failed.");
    }

    /// wait and copy back
    for(auto& it : ffts)
    {
        if(hipStreamSynchronize(it.stream) != hipSuccess)
            throw std::runtime_error("hipStreamSynchronize failed.");
        hip_status = hipMemcpy(it.cpu_buf.data(), it.gpu_buf, total_bytes, hipMemcpyDeviceToHost);
        if(hip_status != hipSuccess)
            throw std::runtime_error("hipMemcpy failed.");
    }

    /// clean up
    for(auto& it : ffts)
    {
        fft_status = rocfft_plan_destroy(it.plan);
        if(fft_status != rocfft_status_success)
            throw std::runtime_error("rocfft_plan_destroy failed.");

        fft_status = rocfft_execution_info_destroy(it.info);
        if(fft_status != rocfft_status_success)
            throw std::runtime_error("rocfft_execution_info_destroy failed.");

        if(hipStreamDestroy(it.stream) != hipSuccess)
            throw std::runtime_error("hipStreamDestroy failed.");
        if(hipFree(it.gpu_buf) != hipSuccess)
            throw std::runtime_error("hipFree failed.");
    }

    if(rocfft_cleanup() != rocfft_status_success)
        throw std::runtime_error("rocfft_cleanup failed.");
    return 0;
}
