// Copyright (C) 2019 - 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include <complex>
#include <functional>
#include <iostream>
#include <numeric>
#include <vector>

#include <hip/hip_runtime_api.h>
#include <hip/hip_vector_types.h>
#include <rocfft/rocfft.h>

#include "../../../shared/CLI11.hpp"
#include "examplekernels.h"
#include "exampleutils.h"
#include <stdexcept>

int main(int argc, char* argv[])
{
    std::cout << "rocfft double-precision complex-to-complex transform\n" << std::endl;

    // Length of transform:
    std::vector<size_t> length = {8};

    // Gpu device id:
    size_t deviceId = 0;

    // Command-line options:
    CLI::App app{"rocfft sample command line options"};
    app.add_option("--device", deviceId, "Select a specific device id")->default_val(0);
    CLI::Option* opt_outofplace
        = app.add_flag("-o, --outofplace", "Perform an out-of-place transform");
    CLI::Option* opt_inverse = app.add_flag("-i, --inverse", "Perform an inverse transform");
    app.add_option(
        "--length", length, "Lengths of the transform separated by spaces (eg: --length 4 4)");

    try
    {
        app.parse(argc, argv);
    }
    catch(const CLI::ParseError& e)
    {
        return app.exit(e);
    }

    // Placeness for the transform
    if(rocfft_setup() != rocfft_status_success)
        throw std::runtime_error("rocfft_setup failed.");
    const rocfft_result_placement place
        = *opt_outofplace ? rocfft_placement_notinplace : rocfft_placement_inplace;
    const bool inplace = place == rocfft_placement_inplace;

    // Direction of transform
    const rocfft_transform_type direction = *opt_inverse ? rocfft_transform_type_complex_forward
                                                         : rocfft_transform_type_complex_inverse;

    // Set up the strides and buffer size for the input:
    std::vector<size_t> istride = {1};
    for(unsigned int i = 1; i < length.size(); ++i)
    {
        istride.push_back(length[i - 1] * istride[i - 1]);
    }
    const size_t isize = length[length.size() - 1] * istride[istride.size() - 1];

    // Set up the strides and buffer size for the output:
    std::vector<size_t> ostride = {1};
    for(unsigned int i = 1; i < length.size(); ++i)
    {
        ostride.push_back(length[i - 1] * ostride[i - 1]);
    }
    const size_t osize = length[length.size() - 1] * ostride[ostride.size() - 1];

    // Print information about the transform:
    std::cout << "direction: ";
    if(direction == rocfft_transform_type_complex_forward)
        std::cout << "forward\n";
    else
        std::cout << "inverse\n";
    std::cout << "length:";
    for(const auto i : length)
        std::cout << " " << i;
    std::cout << "\n";
    if(inplace)
        std::cout << "in-place transform\n";
    else
        std::cout << "out-of-place transform\n";
    std::cout << "deviceID: " << deviceId << "\n";
    std::cout << "input strides:";
    for(auto i : istride)
        std::cout << " " << i;
    std::cout << "\n";
    std::cout << "output strides:";
    for(auto i : ostride)
        std::cout << " " << i;
    std::cout << "\n";
    std::cout << "input size: " << isize << "\n";
    std::cout << "output size: " << isize << "\n";
    std::cout << std::endl;

    // Set the device:
    if(hipSetDevice(deviceId) != hipSuccess)
        throw std::runtime_error("hipSetDevice failed.");

    // Create HIP device object and allocate data
    hipDoubleComplex* gpu_in = nullptr;
    if(hipMalloc(&gpu_in, isize * sizeof(hipDoubleComplex)) != hipSuccess)
        throw std::runtime_error("hipMalloc failed.");

    // Inititalize the data on the device
    initcomplex_cm(length, istride, gpu_in);
    if(hipDeviceSynchronize() != hipSuccess)
        throw std::runtime_error("hipDeviceSynchronize failed.");

    hipError_t hip_status = hipGetLastError();
    if(hip_status != hipSuccess)
        throw std::runtime_error("device error");

    std::cout << "input:\n";
    std::vector<hipDoubleComplex> idata(isize);
    hip_status
        = hipMemcpy(idata.data(), gpu_in, isize * sizeof(hipDoubleComplex), hipMemcpyDefault);
    if(hip_status != hipSuccess)
        throw std::runtime_error("hipMemcpy failed.");

    printbuffer_cm(idata, length, istride, 1, isize);

    // Create the a descrition struct to set data layout:
    rocfft_plan_description gpu_description = nullptr;
    // rocfft_status can be used to capture API status info
    rocfft_status rc = rocfft_plan_description_create(&gpu_description);
    if(rc != rocfft_status_success)
        throw std::runtime_error("failed to create plan description");
    rc = rocfft_plan_description_set_data_layout(gpu_description,
                                                 rocfft_array_type_complex_interleaved,
                                                 rocfft_array_type_complex_interleaved,
                                                 nullptr,
                                                 nullptr,
                                                 istride.size(), // input stride length
                                                 istride.data(), // input stride data
                                                 0, // input batch distance
                                                 ostride.size(), // output stride length
                                                 ostride.data(), // output stride data
                                                 0); // ouptut batch distance
    if(rc != rocfft_status_success)
        throw std::runtime_error("failed to set data layout");
    // We can also pass "nullptr" instead of a description; rocFFT will use reasonable
    // default parameters.  If the data isn't contiguous, we need to set strides, etc,
    // using the description.

    // Create the plan
    rocfft_plan gpu_plan = nullptr;
    rc                   = rocfft_plan_create(&gpu_plan,
                            place,
                            direction,
                            rocfft_precision_double,
                            length.size(), // Dimension
                            length.data(), // lengths
                            1, // Number of transforms
                            gpu_description); // Description
    if(rc != rocfft_status_success)
        throw std::runtime_error("failed to create plan");

    // Get the execution info for the fft plan (in particular, work memory requirements):
    rocfft_execution_info planinfo = nullptr;
    rc                             = rocfft_execution_info_create(&planinfo);
    if(rc != rocfft_status_success)
        throw std::runtime_error("failed to create execution info");
    size_t workbuffersize = 0;
    rc                    = rocfft_plan_get_work_buffer_size(gpu_plan, &workbuffersize);
    if(rc != rocfft_status_success)
        throw std::runtime_error("failed to get work buffer size");

    // If the transform requires work memory, allocate a work buffer:
    void* wbuffer = nullptr;
    if(workbuffersize > 0)
    {
        hip_status = hipMalloc(&wbuffer, workbuffersize);
        if(hip_status != hipSuccess)
            throw std::runtime_error("hipMalloc failed.");
        rc = rocfft_execution_info_set_work_buffer(planinfo, wbuffer, workbuffersize);
        if(rc != rocfft_status_success)
            throw std::runtime_error("failed to set work buffer.");
    }

    // If the transform is out-of-place, allocate the output buffer as well:
    double2* gpu_out = inplace ? gpu_in : nullptr;
    if(!inplace)
    {
        hip_status = hipMalloc(&gpu_out, osize * sizeof(hipDoubleComplex));
        if(hip_status != hipSuccess)
            throw std::runtime_error("hipMalloc failed.");
    }

    // Execute the GPU transform:
    rc = rocfft_execute(gpu_plan, // plan
                        (void**)&gpu_in, // in_buffer
                        (void**)&gpu_out, // out_buffer
                        planinfo); // execution info
    if(rc != rocfft_status_success)
        throw std::runtime_error("failed to execute.");

    // Get the output from the device and print to cout:
    std::cout << "output:\n";
    std::vector<hipDoubleComplex> odata(osize);
    hip_status
        = hipMemcpy(odata.data(), gpu_out, osize * sizeof(hipDoubleComplex), hipMemcpyDeviceToHost);
    if(hip_status != hipSuccess)
        throw std::runtime_error("hipMemcpy failed.");

    printbuffer_cm(odata, length, istride, 1, isize);

    // Clean up: free GPU memory:
    if(hipFree(gpu_in) != hipSuccess)
        throw std::runtime_error("hipFree failed.");

    if(!inplace)
    {
        if(hipFree(gpu_out) != hipSuccess)
            throw std::runtime_error("hipFree failed.");
    }
    if(wbuffer != nullptr)
    {
        if(hipFree(wbuffer) != hipSuccess)
            throw std::runtime_error("hipFree failed.");
    }

    // Clean up: destroy plans:
    if(rocfft_execution_info_destroy(planinfo) != rocfft_status_success)
        throw std::runtime_error("rocfft_execution_info_destroy failed.");
    planinfo = nullptr;
    if(rocfft_plan_description_destroy(gpu_description) != rocfft_status_success)
        throw std::runtime_error("rocfft_plan_description_destroy failed.");
    gpu_description = nullptr;
    if(rocfft_plan_destroy(gpu_plan) != rocfft_status_success)
        throw std::runtime_error("rocfft_plan_destroy failed.");
    gpu_plan = nullptr;

    if(rocfft_cleanup() != rocfft_status_success)
        throw std::runtime_error("rocfft_cleanup failed.");
    return 0;
}
