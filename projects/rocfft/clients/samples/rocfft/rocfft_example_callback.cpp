/******************************************************************************
* Copyright (C) 2021 - 2026 Advanced Micro Devices, Inc. All rights reserved.
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

// The static function-pointer callback path is unusable under the
// SPIR-V JIT-link flow, so skip it in the device pass. Gate on
// __HIP_DEVICE_COMPILE__ because when amdgcnspirv is the sole target
// __SPIRV__ is also set in the host pass, which must still compile.
#if !(defined(__HIP_DEVICE_COMPILE__) && defined(__SPIRV__))

#include "rocfft/rocfft.h"
#include <hip/hip_complex.h>
#include <hip/hip_runtime.h>
#include <hip/hip_vector_types.h>
#include <hip/hiprtc.h>
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

const char* callback_src{
    R"_CALLBACK_SRC_(
struct load_cbdata
{
    double2* filter;
    double   scale;
};

extern "C"
__device__ double2 load_callback(double2* input, size_t offset, void* cbdata, void* sharedMem)
{
    auto data = static_cast<load_cbdata*>(cbdata);

    // multiply each element by filter element and scale
    return hipCmul(hipCmul(input[offset], data->filter[offset]),
                   make_hipDoubleComplex(data->scale, data->scale));
}
)_CALLBACK_SRC_"};

// compile the callback with hiprtc
std::vector<char> compile_callback()
{
    hiprtcProgram prog;
    if(hiprtcCreateProgram(&prog, callback_src, "callback.hip", 0, nullptr, nullptr)
       != HIPRTC_SUCCESS)
        throw std::runtime_error("unable to create program");

    std::vector<const char*> options;
    options.push_back("--offload-arch=amdgcnspirv");
    options.push_back("-c");

    if(hiprtcCompileProgram(prog, options.size(), options.data()) != HIPRTC_SUCCESS)
    {
        size_t logSize = 0;
        hiprtcGetProgramLogSize(prog, &logSize);

        if(logSize)
        {
            std::vector<char> log(logSize, '\0');
            if(hiprtcGetProgramLog(prog, log.data()) == HIPRTC_SUCCESS)
                throw std::runtime_error(std::string(log.begin(), log.end()));
        }
        throw std::runtime_error("compile failed without log");
    }

    size_t codeSize;
    if(hiprtcGetBitcodeSize(prog, &codeSize) != HIPRTC_SUCCESS)
        throw std::runtime_error("failed to get code size");

    std::vector<char> code(codeSize);
    if(hiprtcGetBitcode(prog, code.data()) != HIPRTC_SUCCESS)
        throw std::runtime_error("failed to get code");
    hiprtcDestroyProgram(&prog);
    return code;
}

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

    std::vector<void*> cbdatas(1);
    cbdatas[0] = cbdata_dev;

    // Add callback to plan description
    auto                    code = compile_callback();
    rocfft_plan_description desc = nullptr;
    if(rocfft_plan_description_create(&desc) != rocfft_status_success)
        throw std::runtime_error("failed to create plan description");

    if(rocfft_plan_description_set_load_callback(desc, "load_callback", code.data(), code.size(), 0)
       != rocfft_status_success)
        throw std::runtime_error("failed to set load callback");

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
                          desc)
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

    if(rocfft_execution_info_set_load_callback_data(info, cbdatas.data(), cbdatas.size())
       != rocfft_status_success)
        throw std::runtime_error("rocfft_execution_info_set_load_callback_data failed");

    // Execute plan
    if(rocfft_execute(plan, (void**)&x, nullptr, info) != rocfft_status_success)
        throw std::runtime_error("rocfft_execute failed.");

    // Read the result back before tearing down the plan and its resources.
    std::vector<double2> y(N);
    hip_status = hipMemcpy(y.data(), x, Nbytes, hipMemcpyDeviceToHost);
    if(hip_status != hipSuccess)
        throw std::runtime_error("hipMemcpy failed.");

    for(size_t i = 0; i < N; i++)
    {
        std::cout << "element " << i << " input:  (" << cx[i].x << "," << cx[i].y << ")"
                  << " output: (" << y[i].x << "," << y[i].y << ")" << std::endl;
    }

    // Clean up work buffer
    if(work_buf_size)
    {
        if(hipFree(work_buf) != hipSuccess)
            throw std::runtime_error("hipFree failed.");
    }

    // execution info is always created above; destroy it unconditionally
    if(rocfft_execution_info_destroy(info) != rocfft_status_success)
        throw std::runtime_error("rocfft_execution_info_destroy failed.");
    info = nullptr;

    // Destroy plan
    if(rocfft_plan_destroy(plan) != rocfft_status_success)
        throw std::runtime_error("rocfft_plan_destroy failed.");
    plan = nullptr;

    // Destroy plan description
    if(rocfft_plan_description_destroy(desc) != rocfft_status_success)
        throw std::runtime_error("rocfft_plan_description_destroy failed.");

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

#endif
