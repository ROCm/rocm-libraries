// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include "../../../shared/CLI11.hpp"
#include "rocfft/rocfft.h"
#include <hip/hip_runtime_api.h>
#include <hip/hip_vector_types.h>

#include <stdexcept>

int main(int argc, char* argv[])
{
    std::cout << "rocfft single-node multi-gpu complex-to-complex 3D FFT example\n";

    // Length of transform, first dimension must be greather than number of GPU devices
    std::vector<size_t> length = {8, 8};

    // Gpu device ids:
    std::vector<size_t> devices = {0, 1};

    // Command-line options:
    CLI::App app{"rocfft sample command line options"};
    app.add_option("--length", length, "2-D FFT size (eg: --length 256 256)");
    app.add_option(
        "--devices", devices, "List of devices to use separated by spaces (eg: --devices 1 3)");

    try
    {
        app.parse(argc, argv);
    }
    catch(const CLI::ParseError& e)
    {
        return app.exit(e);
    }

    int deviceCount = devices.size();
    std::cout << "Using " << deviceCount << " device(s)\n";
    int nDevices;
    (void)hipGetDeviceCount(&nDevices);

    std::cout << "Number of available GPUs: " << nDevices << " \n";
    if(nDevices <= static_cast<int>(*std::max_element(devices.begin(), devices.end())))
        throw std::runtime_error("device ID greater than number of available devices");

    // Placeness for the transform
    auto fftrc = rocfft_status_success;
    fftrc      = rocfft_setup();
    if(fftrc != rocfft_status_success)
        throw std::runtime_error("rocfft_setup failed.");
    const rocfft_result_placement place = rocfft_placement_notinplace;

    // Direction of transform
    const rocfft_transform_type direction = rocfft_transform_type_complex_forward;

    rocfft_plan_description description = nullptr;
    rocfft_plan_description_create(&description);
    // Do not set stride information via the descriptor, they are to be defined during field
    // creation below
    rocfft_plan_description_set_data_layout(description,
                                            rocfft_array_type_complex_interleaved,
                                            rocfft_array_type_complex_interleaved,
                                            nullptr,
                                            nullptr,
                                            0,
                                            nullptr,
                                            0,
                                            0,
                                            nullptr,
                                            0);

    auto hiprc = hipSuccess;

    std::cout << "input data decomposition:\n";
    std::vector<void*> gpu_in(devices.size());
    {
        // Row-major stride for brick data layout in memory
        std::vector<size_t> inbrick_stride = {1, length[1]};

        rocfft_field infield = nullptr;
        rocfft_field_create(&infield);

        std::vector<std::vector<size_t>> inbrick_lower(gpu_in.size());
        std::vector<std::vector<size_t>> inbrick_upper(gpu_in.size());

        for(size_t idx = 0; idx < gpu_in.size(); ++idx)
        {
            const size_t inbrick_length1
                = length[1] / gpu_in.size() + (idx < length[1] % gpu_in.size() ? 1 : 0);
            const size_t inbrick_lower1
                = idx * (length[1] / gpu_in.size()) + std::min(idx, length[1] % gpu_in.size());
            const size_t inbrick_upper1 = inbrick_lower1 + inbrick_length1;
            inbrick_lower[idx]          = {0, inbrick_lower1};
            inbrick_upper[idx]          = {length[0], inbrick_upper1};

            rocfft_brick inbrick = nullptr;
            rocfft_brick_create(&inbrick,
                                inbrick_lower[idx].data(),
                                inbrick_upper[idx].data(),
                                inbrick_stride.data(),
                                inbrick_lower[idx].size(),
                                devices[idx]);
            rocfft_field_add_brick(infield, inbrick);
            rocfft_brick_destroy(inbrick);
            inbrick = nullptr;

            const size_t memSize = length[0] * inbrick_length1 * sizeof(std::complex<double>);

            std::cout << "in-brick " << idx;
            std::cout << "\n\tlower indices:";
            for(const auto val : inbrick_lower[idx])
                std::cout << " " << val;
            std::cout << "\n\tupper indices:";
            for(const auto val : inbrick_upper[idx])
                std::cout << " " << val;
            std::cout << "\n\tstrides:";
            for(const auto val : inbrick_stride)
                std::cout << " " << val;
            std::cout << "\n";
            std::cout << "\tbuffer size: " << memSize << "\n";

            hiprc = hipSetDevice(devices[idx]);
            if(hiprc != hipSuccess)
                throw std::runtime_error("hipSetDevice failed");

            hiprc = hipMalloc(&gpu_in[idx], memSize);
            if(hiprc != hipSuccess)
                throw std::runtime_error("hipMalloc failed");
            std::vector<std::complex<double>> host_in(length[0] * inbrick_length1);
            for(auto idx0 = inbrick_lower[idx][0]; idx0 < inbrick_upper[idx][0]; ++idx0)
            {
                for(auto idx1 = inbrick_lower[idx][1]; idx1 < inbrick_upper[idx][1]; ++idx1)
                {
                    const auto pos = (idx0 - inbrick_lower[idx][0]) * inbrick_stride[0]
                                     + (idx1 - inbrick_lower[idx][1]) * inbrick_stride[1];
                    host_in[pos] = std::complex<double>(idx0, idx1);
                    std::cout << host_in[pos] << " ";
                }
                std::cout << "\n";
            }

            hiprc = hipMemcpy(gpu_in[idx], host_in.data(), memSize, hipMemcpyHostToDevice);
            if(hiprc != hipSuccess)
                throw std::runtime_error("hipMemcpy failed");
        }

        rocfft_plan_description_add_infield(description, infield);

        fftrc = rocfft_field_destroy(infield);
        if(fftrc != rocfft_status_success)
            throw std::runtime_error("failed destroy infield");
    }

    std::cout << "output data decomposition:\n";
    std::vector<void*>               gpu_out(devices.size());
    std::vector<std::vector<size_t>> outbrick_lower(gpu_out.size());
    std::vector<std::vector<size_t>> outbrick_upper(gpu_out.size());
    std::vector<size_t>              outbrick_stride = {1, length[1]};
    {
        rocfft_field outfield = nullptr;
        rocfft_field_create(&outfield);

        for(size_t idx = 0; idx < gpu_out.size(); ++idx)
        {
            const size_t outbrick_length1
                = length[1] / gpu_out.size() + (idx < length[1] % gpu_in.size() ? 1 : 0);
            const size_t outbrick_lower1
                = idx * (length[1] / gpu_out.size()) + std::min(idx, length[1] % gpu_out.size());

            rocfft_brick outbrick = nullptr;
            outbrick_lower[idx]   = {0, outbrick_lower1};
            outbrick_upper[idx]   = {length[0], outbrick_lower1 + outbrick_length1};
            rocfft_brick_create(&outbrick,
                                outbrick_lower[idx].data(),
                                outbrick_upper[idx].data(),
                                outbrick_stride.data(),
                                outbrick_lower[idx].size(),
                                devices[idx]);
            rocfft_field_add_brick(outfield, outbrick);
            rocfft_brick_destroy(outbrick);
            outbrick = nullptr;

            const size_t memSize = length[0] * outbrick_length1 * sizeof(std::complex<double>);

            std::cout << "out-brick " << idx;
            std::cout << "\n\tlower indices:";
            for(const auto val : outbrick_lower[idx])
                std::cout << " " << val;
            std::cout << "\n\tupper indices:";
            for(const auto val : outbrick_upper[idx])
                std::cout << " " << val;
            std::cout << "\n\tstrides:";
            for(const auto val : outbrick_stride)
                std::cout << " " << val;
            std::cout << "\n";
            std::cout << "\tbuffer size: " << memSize << "\n";

            (void)hipSetDevice(devices[idx]);

            if(hipMalloc(&gpu_out[idx], memSize) != hipSuccess)
                throw std::runtime_error("hipMalloc failed");
        }

        rocfft_plan_description_add_outfield(description, outfield);

        fftrc = rocfft_field_destroy(outfield);
        if(fftrc != rocfft_status_success)
            throw std::runtime_error("failed destroy outfield");
    }

    // Create a multi-gpu plan:
    (void)hipSetDevice(devices[0]);
    rocfft_plan gpu_plan = nullptr;
    fftrc                = rocfft_plan_create(&gpu_plan,
                               place,
                               direction,
                               rocfft_precision_double,
                               length.size(), // Dimension
                               length.data(), // lengths
                               1, // Number of transforms
                               description); // Description
    if(fftrc != rocfft_status_success)
        throw std::runtime_error("failed to create plan");

    // Get execution information and allocate work buffer
    rocfft_execution_info planinfo      = nullptr;
    size_t                work_buf_size = 0;
    if(rocfft_plan_get_work_buffer_size(gpu_plan, &work_buf_size) != rocfft_status_success)
        throw std::runtime_error("rocfft_plan_get_work_buffer_size failed.");
    void* work_buf = nullptr;

    if(work_buf_size)
    {
        if(rocfft_execution_info_create(&planinfo) != rocfft_status_success)
            throw std::runtime_error("failed to create execution info");
        if(hipMalloc(&work_buf, work_buf_size) != hipSuccess)
            throw std::runtime_error("hipMalloc failed");
        if(rocfft_execution_info_set_work_buffer(planinfo, work_buf, work_buf_size)
           != rocfft_status_success)
            throw std::runtime_error("rocfft_execution_info_set_work_buffer failed.");
    }

    // Execute plan:
    fftrc = rocfft_execute(gpu_plan, (void**)gpu_in.data(), (void**)gpu_out.data(), planinfo);
    if(fftrc != rocfft_status_success)
        throw std::runtime_error("failed to execute.");

    // Output the data.
    for(size_t idx = 0; idx < gpu_out.size(); ++idx)
    {
        std::cout << "out brick " << idx << "\n";

        const auto nbrick = (outbrick_upper[idx][0] - outbrick_lower[idx][0])
                            * (outbrick_upper[idx][1] - outbrick_lower[idx][1]);
        std::vector<std::complex<double>> host_out(nbrick);
        hiprc = hipMemcpy(host_out.data(),
                          gpu_out[idx],
                          nbrick * sizeof(std::complex<double>),
                          hipMemcpyDeviceToHost);
        if(hiprc != hipSuccess)
            throw std::runtime_error("hipMemcpy failed");

        for(auto idx0 = outbrick_lower[idx][0]; idx0 < outbrick_upper[idx][0]; ++idx0)
        {
            for(auto idx1 = outbrick_lower[idx][1]; idx1 < outbrick_upper[idx][1]; ++idx1)
            {
                const auto pos = (idx0 - outbrick_lower[idx][0]) * outbrick_stride[0]
                                 + (idx1 - outbrick_lower[idx][1]) * outbrick_stride[1];
                std::cout << host_out[pos] << " ";
            }
            std::cout << "\n";
        }
    }

    // Destroy plan
    if(planinfo != nullptr)
    {
        if(rocfft_execution_info_destroy(planinfo) != rocfft_status_success)
            throw std::runtime_error("rocfft_execution_info_destroy failed.");
        planinfo = nullptr;
    }
    if(rocfft_plan_description_destroy(description) != rocfft_status_success)
        throw std::runtime_error("rocfft_plan_description_destroy failed.");
    description = nullptr;
    if(rocfft_plan_destroy(gpu_plan) != rocfft_status_success)
        throw std::runtime_error("rocfft_plan_destroy failed.");
    gpu_plan = nullptr;

    if(rocfft_cleanup() != rocfft_status_success)
        throw std::runtime_error("rocfft_cleanup failed.");

    for(size_t idx = 0; idx < gpu_in.size(); ++idx)
    {
        (void)hipFree(gpu_in[idx]);
    }
    for(size_t idx = 0; idx < gpu_out.size(); ++idx)
    {
        (void)hipFree(gpu_out[idx]);
    }

    return 0;
}
