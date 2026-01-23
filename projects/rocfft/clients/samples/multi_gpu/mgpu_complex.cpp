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
#include "../../../shared/ptrdiff.h"
#include "rocfft/rocfft.h"
#include <hip/hip_runtime_api.h>
#include <hip/hip_vector_types.h>

#include <stdexcept>

void initbrick(const std::vector<size_t>&         lower,
               const std::vector<size_t>&         upper,
               const std::vector<size_t>&         stride,
               std::vector<std::complex<double>>& hostbrick)
{
    // We assume that the batch size is 1.
    switch(lower.size())
    {
    case 3:
        // 2D data + 1 batch
        for(auto idx1 = lower[1]; idx1 < upper[1]; ++idx1)
        {
            for(auto idx0 = lower[0]; idx0 < upper[0]; ++idx0)
            {
                const auto pos = (idx0 - lower[0]) * stride[0] + (idx1 - lower[1]) * stride[1];
                hostbrick[pos] = std::complex<double>(idx0, idx1);
            }
        }
        break;
    case 4:
        // 3D data + 1 batch
        for(auto idx2 = lower[2]; idx2 < upper[2]; ++idx2)
        {
            for(auto idx1 = lower[1]; idx1 < upper[1]; ++idx1)
            {
                for(auto idx0 = lower[0]; idx0 < upper[0]; ++idx0)
                {
                    const auto pos = (idx0 - lower[0]) * stride[0] + (idx1 - lower[1]) * stride[1]
                                     + (idx2 - lower[2]) * stride[2];
                    hostbrick[pos] = std::complex<double>(idx0 + idx1, idx2);
                }
            }
        }
        break;
    default:
        throw std::runtime_error("Unspported dimension");
    }
}

template <typename Tval>
void printbrick(const std::vector<size_t>& lower,
                const std::vector<size_t>& upper,
                const std::vector<size_t>& stride,
                const std::vector<Tval>&   hostbrick)
{
    // We assume that the batch size is 1.
    switch(lower.size())
    {
    case 3:
        // 2D data + 1 batch
        for(auto idx1 = lower[1]; idx1 < upper[1]; ++idx1)
        {
            for(auto idx0 = lower[0]; idx0 < upper[0]; ++idx0)
            {
                const auto pos = (idx0 - lower[0]) * stride[0] + (idx1 - lower[1]) * stride[1];
                std::cout << hostbrick[pos] << " ";
            }
            std::cout << "\n";
        }
        break;
    case 4:
        // 3D data + 1 batch
        for(auto idx2 = lower[2]; idx2 < upper[2]; ++idx2)
        {
            for(auto idx1 = lower[1]; idx1 < upper[1]; ++idx1)
            {
                for(auto idx0 = lower[0]; idx0 < upper[0]; ++idx0)
                {
                    const auto pos = (idx0 - lower[0]) * stride[0] + (idx1 - lower[1]) * stride[1]
                                     + (idx2 - lower[2]) * stride[2];
                    std::cout << hostbrick[pos] << " ";
                }
                std::cout << "\n";
            }
            std::cout << "\n";
        }
        break;
    default:
        throw std::runtime_error("Unspported dimension");
    }
}

int main(int argc, char* argv[])
{
    std::cout << "rocfft single-node multi-gpu complex-to-complex 2D/3D FFT example\n";

    // Length of transform, first dimension must be greather than number of GPU devices
    std::vector<size_t> length = {8, 8};

    // Gpu device ids:
    std::vector<size_t> devices = {0, 1};

    // Is the transform in-place or out-of-place:
    rocfft_result_placement place = rocfft_placement_notinplace;

    // Command-line options:
    CLI::App app{"rocfft sample command line options"};
    app.add_option("--length", length, "FFT size (eg: --length 256 256)");
    app.add_option(
        "--devices", devices, "List of devices to use separated by spaces (eg: --devices 1 3)");
    app.add_flag("-o, --notInPlace", "")->each([&](const std::string&) {
        place = rocfft_placement_notinplace;
    });
    app.add_flag("-i, --inPlace", "")->each([&](const std::string&) {
        place = rocfft_placement_inplace;
    });

    try
    {
        app.parse(argc, argv);
    }
    catch(const CLI::ParseError& e)
    {
        return app.exit(e);
    }

    if(length.size() != 2 && length.size() != 3)
        throw std::invalid_argument("This sample is restricted to 2D and 3D cases.");

    int deviceCount = devices.size();
    std::cout << "Using " << deviceCount << " device(s)\n";
    int  nDevices;
    auto hiprc = hipGetDeviceCount(&nDevices);
    if(hiprc != hipSuccess || nDevices == -1)
        throw std::runtime_error("hipGetDeviceCount failed");

    std::cout << "Number of available GPUs: " << nDevices << " \n";
    if(nDevices <= static_cast<int>(*std::max_element(devices.begin(), devices.end())))
        throw std::runtime_error("device ID greater than number of available devices");

    // Placeness for the transform
    auto fftrc = rocfft_status_success;
    fftrc      = rocfft_setup();
    if(fftrc != rocfft_status_success)
        throw std::runtime_error("rocfft_setup failed.");

    // Direction of transform
    const rocfft_transform_type direction = rocfft_transform_type_complex_forward;

    rocfft_plan_description description = nullptr;
    fftrc                               = rocfft_plan_description_create(&description);
    if(fftrc != rocfft_status_success)
        throw std::runtime_error("rocfft_plan_description_create failed.");
    // Do not set stride information via the descriptor, they are to be defined during field
    // creation below
    fftrc = rocfft_plan_description_set_data_layout(description,
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
    if(fftrc != rocfft_status_success)
        throw std::runtime_error("rocfft_plan_description_set_data_layout failed.");

    std::cout << "\nInput data decomposition:\n";
    std::vector<std::vector<size_t>> inbrick_lower(devices.size());
    std::vector<std::vector<size_t>> inbrick_upper(devices.size());
    std::vector<std::vector<size_t>> inbrick_stride(devices.size());
    std::vector<size_t>              inbufsizes(devices.size());
    {
        rocfft_field infield = nullptr;
        fftrc                = rocfft_field_create(&infield);
        if(fftrc != rocfft_status_success)
            throw std::runtime_error("rocfft_field_create failed.");

        for(size_t idx = 0; idx < devices.size(); ++idx)
        {
            // Divide the data in dimension 1:
            const size_t inbrick_length1
                = length[1] / devices.size() + (idx < length[1] % devices.size() ? 1 : 0);
            const size_t inbrick_lower1
                = idx * (length[1] / devices.size()) + std::min(idx, length[1] % devices.size());
            const size_t inbrick_upper1 = inbrick_lower1 + inbrick_length1;
            for(size_t dim = 0; dim < length.size(); ++dim)
            {
                inbrick_stride[idx].push_back(
                    compute_ptrdiff(inbrick_lower[idx], inbrick_upper[idx], inbrick_stride[idx]));

                if(dim != 1)
                {
                    inbrick_lower[idx].push_back(0);
                    inbrick_upper[idx].push_back(length[dim]);
                }
                else
                {
                    inbrick_lower[idx].push_back(inbrick_lower1);
                    inbrick_upper[idx].push_back(inbrick_upper1);
                }
            }
            // We must also include the batch dimension:
            inbrick_stride[idx].push_back(
                compute_ptrdiff(inbrick_lower[idx], inbrick_upper[idx], inbrick_stride[idx]));
            inbrick_lower[idx].push_back(0);
            inbrick_upper[idx].push_back(1);

            rocfft_brick inbrick = nullptr;
            fftrc                = rocfft_brick_create(&inbrick,
                                        inbrick_lower[idx].data(),
                                        inbrick_upper[idx].data(),
                                        inbrick_stride[idx].data(),
                                        inbrick_lower[idx].size(),
                                        devices[idx]);
            if(fftrc != rocfft_status_success)
                throw std::runtime_error("rocfft_brick_create failed (inbrick["
                                         + std::to_string(idx) + "].");
            fftrc = rocfft_field_add_brick(infield, inbrick);
            if(fftrc != rocfft_status_success)
                throw std::runtime_error("rocfft_field_add_brick failed (inbrick["
                                         + std::to_string(idx) + "].");
            fftrc = rocfft_brick_destroy(inbrick);
            if(fftrc != rocfft_status_success)
                throw std::runtime_error("rocfft_brick_destroy failed (inbrick["
                                         + std::to_string(idx) + "].");

            inbufsizes[idx] = inbrick_stride[idx].back() * sizeof(std::complex<double>);
            std::cout << "Input brick " << idx;
            std::cout << "\n\tlower indices:";
            for(const auto val : inbrick_lower[idx])
                std::cout << " " << val;
            std::cout << "\n\tupper indices:";
            for(const auto val : inbrick_upper[idx])
                std::cout << " " << val;
            std::cout << "\n\tstrides:";
            for(const auto val : inbrick_stride[idx])
                std::cout << " " << val;
            std::cout << "\n";
            std::cout << "\tbuffer size: " << inbufsizes[idx] << "\n";
        }

        fftrc = rocfft_plan_description_add_infield(description, infield);
        if(fftrc != rocfft_status_success)
            throw std::runtime_error("rocfft_plan_description_add_infield failed");

        fftrc = rocfft_field_destroy(infield);
        if(fftrc != rocfft_status_success)
            throw std::runtime_error("failed destroy infield");
    }

    std::cout << "\nOutput data decomposition:\n";
    std::vector<std::vector<size_t>> outbrick_lower(devices.size());
    std::vector<std::vector<size_t>> outbrick_upper(devices.size());
    std::vector<std::vector<size_t>> outbrick_stride(devices.size());
    std::vector<size_t>              outbufsizes(devices.size());
    {
        rocfft_field outfield = nullptr;
        fftrc                 = rocfft_field_create(&outfield);
        if(fftrc != rocfft_status_success)
            throw std::runtime_error("rocfft_field_create failed (outfield)");

        // 2D output is split in the same direction as input; 3D output is split in the slowest
        // dimension.
        const size_t splitdim = length.size() == 2 ? 1 : 2;

        for(size_t idx = 0; idx < devices.size(); ++idx)
        {
            const size_t outbrick_length_split
                = length[splitdim] / devices.size()
                  + (idx < length[splitdim] % devices.size() ? 1 : 0);
            const size_t outbrick_lower_split = idx * (length[splitdim] / devices.size())
                                                + std::min(idx, length[splitdim] % devices.size());
            const size_t outbrick_upper_split = outbrick_lower_split + outbrick_length_split;

            for(size_t dim = 0; dim < length.size(); ++dim)
            {
                outbrick_stride[idx].push_back(compute_ptrdiff(
                    outbrick_lower[idx], outbrick_upper[idx], outbrick_stride[idx]));
                if(dim != splitdim)
                {
                    outbrick_lower[idx].push_back(0);
                    outbrick_upper[idx].push_back(length[dim]);
                }
                else
                {
                    outbrick_lower[idx].push_back(outbrick_lower_split);
                    outbrick_upper[idx].push_back(outbrick_upper_split);
                }
            }
            // We must also include the batch dimension:
            outbrick_stride[idx].push_back(
                compute_ptrdiff(outbrick_lower[idx], outbrick_upper[idx], outbrick_stride[idx]));
            outbrick_lower[idx].push_back(0);
            outbrick_upper[idx].push_back(1);

            rocfft_brick outbrick = nullptr;

            fftrc = rocfft_brick_create(&outbrick,
                                        outbrick_lower[idx].data(),
                                        outbrick_upper[idx].data(),
                                        outbrick_stride[idx].data(),
                                        outbrick_lower[idx].size(),
                                        devices[idx]);
            if(fftrc != rocfft_status_success)
                throw std::runtime_error("rocfft_brick_create failed (outbrick["
                                         + std::to_string(idx) + "].");
            fftrc = rocfft_field_add_brick(outfield, outbrick);
            if(fftrc != rocfft_status_success)
                throw std::runtime_error("rocfft_field_add_brick failed (outbrick["
                                         + std::to_string(idx) + "].");
            fftrc = rocfft_brick_destroy(outbrick);
            if(fftrc != rocfft_status_success)
                throw std::runtime_error("rocfft_brick_destroy failed (outbrick["
                                         + std::to_string(idx) + "].");

            outbufsizes[idx] = outbrick_stride[idx].back() * sizeof(std::complex<double>);
            std::cout << "Output brick " << idx;
            std::cout << "\n\tlower indices:";
            for(const auto val : outbrick_lower[idx])
                std::cout << " " << val;
            std::cout << "\n\tupper indices:";
            for(const auto val : outbrick_upper[idx])
                std::cout << " " << val;
            std::cout << "\n\tstrides:";
            for(const auto val : outbrick_stride[idx])
                std::cout << " " << val;
            std::cout << "\n";
            std::cout << "\tbuffer size: " << outbufsizes[idx] << "\n";
        }

        fftrc = rocfft_plan_description_add_outfield(description, outfield);
        if(fftrc != rocfft_status_success)
            throw std::runtime_error("rocfft_plan_description_add_outfield failed");

        fftrc = rocfft_field_destroy(outfield);
        if(fftrc != rocfft_status_success)
            throw std::runtime_error("failed destroy outfield");
    }

    // Allocation and initialization of gpu buffers:
    std::cout << "\nInput data:\n";
    std::vector<void*> gpu_in(devices.size(), nullptr);
    std::vector<void*> gpu_out(devices.size(), nullptr);
    for(size_t idx = 0; idx < gpu_in.size(); ++idx)
    {
        std::cout << "Input brick " << idx << "\n";
        hiprc = hipSetDevice(devices[idx]);
        if(hiprc != hipSuccess)
            throw std::runtime_error("hipSetDevice failed");

        const size_t memsize = place == rocfft_placement_notinplace
                                   ? inbufsizes[idx]
                                   : std::max(inbufsizes[idx], outbufsizes[idx]);
        hiprc                = hipMalloc(&gpu_in[idx], memsize);
        if(hiprc != hipSuccess)
            throw std::runtime_error("hipMalloc failed");
        std::vector<std::complex<double>> host_in(memsize / sizeof(std::complex<double>));
        initbrick(inbrick_lower[idx], inbrick_upper[idx], inbrick_stride[idx], host_in);
        printbrick(inbrick_lower[idx], inbrick_upper[idx], inbrick_stride[idx], host_in);

        hiprc = hipMemcpy(gpu_in[idx], host_in.data(), inbufsizes[idx], hipMemcpyHostToDevice);
        if(hiprc != hipSuccess)
            throw std::runtime_error("hipMemcpy failed");

        if(place == rocfft_placement_notinplace)
            if(hipMalloc(&gpu_out[idx], outbufsizes[idx]) != hipSuccess)
                throw std::runtime_error("hipMalloc failed");
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

    // Execute plan:
    fftrc = rocfft_execute(gpu_plan,
                           (void**)gpu_in.data(),
                           place == rocfft_placement_notinplace ? (void**)gpu_out.data()
                                                                : (void**)nullptr,
                           nullptr // no execution info
    );
    if(fftrc != rocfft_status_success)
        throw std::runtime_error("failed to execute.");

    std::cout << "\nOutput data:\n";
    const auto& out_data = place == rocfft_placement_inplace ? gpu_in : gpu_out;
    for(size_t idx = 0; idx < out_data.size(); ++idx)
    {
        std::cout << "Output brick " << idx << "\n";
        std::vector<std::complex<double>> host_out(outbufsizes[idx] / sizeof(std::complex<double>));
        hiprc = hipMemcpy(host_out.data(), out_data[idx], outbufsizes[idx], hipMemcpyDeviceToHost);
        if(hiprc != hipSuccess)
            throw std::runtime_error("hipMemcpy failed");

        printbrick(outbrick_lower[idx], outbrick_upper[idx], outbrick_stride[idx], host_out);
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
