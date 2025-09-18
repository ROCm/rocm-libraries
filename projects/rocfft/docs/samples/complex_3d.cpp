// Copyright (C) 2019 - 2022 Advanced Micro Devices, Inc. All rights reserved.
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

#include <cassert>
#include <complex>
#include <iostream>
#include <vector>

#include <hip/hip_runtime_api.h>

#include <rocfft/rocfft.h>

int main(int argc, char* argv[])
{
    std::cout << "rocFFT complex 3d FFT example\n";

    // The problem size
    const size_t Nx      = (argc < 2) ? 4 : atoi(argv[1]);
    const size_t Ny      = (argc < 3) ? 4 : atoi(argv[2]);
    const size_t Nz      = (argc < 4) ? 2 : atoi(argv[3]);
    const bool   inplace = (argc < 5) ? false : atoi(argv[4]);
    std::cout << "Nx: " << Nx << "\tNy: " << Ny << "\tNz: " << Nz << "\tin-place: " << inplace
              << std::endl;

    // Initialize data on the host
    std::cout << "Input:\n";
    std::vector<std::complex<float>> cx(Nx * Ny * Nz);
    for(size_t i = 0; i < Nx; i++)
    {
        for(size_t j = 0; j < Ny; j++)
        {
            for(size_t k = 0; k < Nz; k++)
            {
                const size_t pos = (i * Ny + j) * Nz + k;
                cx[pos]          = std::complex<float>(i + j + k, 0);
            }
        }
    }
    for(size_t i = 0; i < Nx; i++)
    {
        for(size_t j = 0; j < Ny; j++)
        {
            for(size_t k = 0; k < Nz; k++)
            {
                const size_t pos = (i * Ny + j) * Nz + k;
                std::cout << cx[pos] << " ";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }
    std::cout << "\n";

    rocfft_setup();

    // Create HIP device object and copy data:
    float2* x = NULL;
    hipMalloc(&x, cx.size() * sizeof(decltype(cx)::value_type));
    hipMemcpy(x, cx.data(), cx.size() * sizeof(decltype(cx)::value_type), hipMemcpyHostToDevice);
    float2* y = inplace ? (float2*)x : NULL;
    if(!inplace)
    {
        hipMalloc(&y, cx.size() * sizeof(decltype(cx)::value_type));
    }

    // Length are in reverse order because rocfft is column-major.
    const size_t lengths[3] = {Nz, Ny, Nx};

    rocfft_status status = rocfft_status_success;

    // Create plans
    rocfft_plan forward = NULL;
    status              = rocfft_plan_create(&forward,
                                inplace ? rocfft_placement_inplace : rocfft_placement_notinplace,
                                rocfft_transform_type_complex_forward,
                                rocfft_precision_single,
                                3, // Dimensions
                                lengths, // lengths
                                1, // Number of transforms
                                NULL); // Description
    assert(status == rocfft_status_success);

    // We may need work memory, which is passed via rocfft_execution_info
    rocfft_execution_info forwardinfo = NULL;
    status                            = rocfft_execution_info_create(&forwardinfo);
    assert(status == rocfft_status_success);
    size_t fbuffersize = 0;
    status             = rocfft_plan_get_work_buffer_size(forward, &fbuffersize);
    assert(status == rocfft_status_success);
    void* fbuffer = NULL;
    if(fbuffersize > 0)
    {
        hipMalloc(&fbuffer, fbuffersize);
        status = rocfft_execution_info_set_work_buffer(forwardinfo, fbuffer, fbuffersize);
        assert(status == rocfft_status_success);
    }

    // Create plans
    rocfft_plan backward = NULL;
    status               = rocfft_plan_create(&backward,
                                inplace ? rocfft_placement_inplace : rocfft_placement_notinplace,
                                rocfft_transform_type_complex_inverse,
                                rocfft_precision_single,
                                3, // Dimensions
                                lengths, // lengths
                                1, // Number of transforms
                                NULL); // Description
    assert(status == rocfft_status_success);

    rocfft_execution_info backwardinfo = NULL;
    status                             = rocfft_execution_info_create(&backwardinfo);
    assert(status == rocfft_status_success);
    size_t bbuffersize = 0;
    status             = rocfft_plan_get_work_buffer_size(backward, &bbuffersize);
    assert(status == rocfft_status_success);
    void* bbuffer = NULL;
    if(bbuffersize > 0)
    {
        hipMalloc(&bbuffer, bbuffersize);
        status = rocfft_execution_info_set_work_buffer(backwardinfo, bbuffer, bbuffersize);
        assert(status == rocfft_status_success);
    }

    // Execute the forward transform
    status = rocfft_execute(forward,
                            (void**)&x, // in_buffer
                            (void**)&y, // out_buffer
                            forwardinfo); // execution info
    assert(status == rocfft_status_success);

    // Copy result back to host
    std::vector<std::complex<float>> cy(cx.size());
    hipMemcpy(cy.data(), y, cy.size() * sizeof(decltype(cy)::value_type), hipMemcpyDeviceToHost);

    std::cout << "Transformed:\n";
    for(size_t i = 0; i < Nx; i++)
    {
        for(size_t j = 0; j < Ny; j++)
        {
            for(size_t k = 0; k < Nz; k++)
            {
                const size_t pos = (i * Ny + j) * Nz + k;
                std::cout << cy[pos] << " ";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }
    std::cout << "\n";

    // Execute the backward transform
    status = rocfft_execute(backward,
                            (void**)&y, // in_buffer
                            (void**)&x, // out_buffer
                            backwardinfo); // execution info
    assert(status == rocfft_status_success);

    std::cout << "Transformed back:\n";
    hipMemcpy(cy.data(), x, cy.size() * sizeof(decltype(cy)::value_type), hipMemcpyDeviceToHost);
    for(size_t i = 0; i < Nx; i++)
    {
        for(size_t j = 0; j < Ny; j++)
        {
            for(size_t k = 0; k < Nz; k++)
            {
                const size_t pos = (i * Ny + j) * Nz + k;
                std::cout << cy[pos] << " ";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }
    std::cout << "\n";

    const float overN = 1.0f / cx.size();
    float       error = 0.0f;
    for(size_t i = 0; i < cx.size(); i++)
    {
        float diff = std::max(std::abs(cx[i].real() - cy[i].real() * overN),
                              std::abs(cx[i].imag() - cy[i].imag() * overN));
        if(diff > error)
        {
            error = diff;
        }
    }
    std::cout << "Maximum error: " << error << "\n";

    hipFree(x);
    if(!inplace)
    {
        hipFree(y);
    }
    hipFree(fbuffer);
    hipFree(bbuffer);

    // Destroy plans
    rocfft_plan_destroy(forward);
    rocfft_plan_destroy(backward);

    rocfft_cleanup();
}
