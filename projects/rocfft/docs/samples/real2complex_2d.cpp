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
    std::cout << "rocFFT real/complex 2d FFT example\n";

    // The problem size
    const size_t Nx      = (argc < 2) ? 8 : atoi(argv[1]);
    const size_t Ny      = (argc < 3) ? 8 : atoi(argv[2]);
    const bool   inplace = (argc < 4) ? true : atoi(argv[3]);

    std::cout << "Nx: " << Nx << "\tNy: " << Ny << "\tin-place: " << inplace << std::endl;

    const size_t Nycomplex = Ny / 2 + 1;

    const size_t Nystride = inplace ? 2 * Nycomplex : Ny;

    // Initialize data on the host
    std::cout << "Input:\n";
    std::vector<float> cx(Nx * Nystride);
    std::fill(cx.begin(), cx.end(), 0.0);
    for(size_t i = 0; i < Nx; i++)
    {
        for(size_t j = 0; j < Ny; j++)
        {
            cx[i * Nystride + j] = i + j;
        }
    }
    for(size_t i = 0; i < Nx; i++)
    {
        for(size_t j = 0; j < Nystride; j++)
        {
            std::cout << cx[i * Nystride + j] << "  ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";

    // Output buffer
    std::vector<std::complex<float>> cy(Nx * Nycomplex);

    rocfft_setup();

    // Create HIP device objects:
    float* x = NULL;
    hipMalloc(&x, cx.size() * sizeof(decltype(cx)::value_type));
    hipMemcpy(x, cx.data(), cx.size() * sizeof(decltype(cx)::value_type), hipMemcpyHostToDevice);
    float2* y = inplace ? (float2*)x : NULL;
    if(!inplace)
    {
        hipMalloc(&y, cy.size() * sizeof(decltype(cy)::value_type));
    }

    // Length are in reverse order because rocfft is column-major.
    const size_t lengths[2] = {Ny, Nx};

    rocfft_status status = rocfft_status_success;

    // Create plans
    rocfft_plan forward = NULL;
    status              = rocfft_plan_create(&forward,
                                inplace ? rocfft_placement_inplace : rocfft_placement_notinplace,
                                rocfft_transform_type_real_forward,
                                rocfft_precision_single,
                                2, // Dimensions
                                lengths, // lengths
                                1, // Number of transforms
                                NULL); // Description
    assert(status == rocfft_status_success);

    // The real-to-complex transform uses work memory, which is passed
    // via a rocfft_execution_info struct.
    rocfft_execution_info forwardinfo;
    status = rocfft_execution_info_create(&forwardinfo);
    assert(status == rocfft_status_success);
    size_t forwardworkbuffersize = 0;
    status = rocfft_plan_get_work_buffer_size(forward, &forwardworkbuffersize);
    assert(status == rocfft_status_success);
    void* forwardwbuffer = NULL;
    if(forwardworkbuffersize > 0)
    {
        hipMalloc(&forwardwbuffer, forwardworkbuffersize);
        status = rocfft_execution_info_set_work_buffer(
            forwardinfo, forwardwbuffer, forwardworkbuffersize);
        assert(status == rocfft_status_success);
    }

    // Execute the forward transform
    status = rocfft_execute(forward, // plan
                            (void**)&x, // in_buffer
                            (void**)&y, // out_buffer
                            forwardinfo); // execution info
    assert(status == rocfft_status_success);

    std::cout << "Transformed:\n";

    hipMemcpy(cy.data(), y, cy.size() * sizeof(decltype(cy)::value_type), hipMemcpyDeviceToHost);
    for(size_t i = 0; i < Nx; i++)
    {
        for(size_t j = 0; j < Nycomplex; ++j)
        {
            const size_t pos = i * Nycomplex + j;
            std::cout << cy[pos] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";

    // Create plans
    rocfft_plan backward = NULL;
    status               = rocfft_plan_create(&backward,
                                inplace ? rocfft_placement_inplace : rocfft_placement_notinplace,
                                rocfft_transform_type_real_inverse,
                                rocfft_precision_single,
                                2, // Dimensions
                                lengths, // lengths
                                1, // Number of transforms
                                NULL); // Description
    assert(status == rocfft_status_success);

    // The real-to-complex transform uses work memory, which is passed
    // via a rocfft_execution_info struct.
    rocfft_execution_info backwardinfo;
    status = rocfft_execution_info_create(&backwardinfo);
    assert(status == rocfft_status_success);
    size_t backwardworkbuffersize = 0;
    status = rocfft_plan_get_work_buffer_size(backward, &backwardworkbuffersize);
    assert(status == rocfft_status_success);
    void* backwardwbuffer = NULL;
    if(backwardworkbuffersize > 0)
    {
        hipMalloc(&backwardwbuffer, backwardworkbuffersize);
        status = rocfft_execution_info_set_work_buffer(
            backwardinfo, backwardwbuffer, backwardworkbuffersize);
        assert(status == rocfft_status_success);
    }

    // Execute the backward transform
    status = rocfft_execute(backward, // plan
                            (void**)&y, // in_buffer
                            (void**)&x, // out_buffer
                            backwardinfo); // execution info
    assert(status == rocfft_status_success);

    std::cout << "Transformed back:\n";
    std::vector<float> backx(cx.size());
    std::fill(backx.begin(), backx.end(), 0.0);
    hipMemcpy(
        backx.data(), x, backx.size() * sizeof(decltype(backx)::value_type), hipMemcpyDeviceToHost);
    for(size_t i = 0; i < Nx; i++)
    {
        for(size_t j = 0; j < Nystride; ++j)
        {
            const size_t pos = i * Nystride + j;
            std::cout << backx[pos] << "  ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";

    const float overN = 1.0f / (Nx * Ny);
    float       error = 0.0f;
    for(size_t i = 0; i < Nx; i++)
    {
        for(size_t j = 0; j < Ny; j++)
        {
            float diff = std::abs(backx[i] * overN - cx[i]);
            if(diff > error)
            {
                error = diff;
            }
        }
    }
    std::cout << "Maximum error: " << error << "\n";

    hipFree(x);
    if(!inplace)
    {
        hipFree(y);
    }
    hipFree(forwardwbuffer);
    hipFree(backwardwbuffer);

    // Destroy plans
    rocfft_plan_destroy(forward);
    rocfft_plan_destroy(backward);

    rocfft_cleanup();
}
