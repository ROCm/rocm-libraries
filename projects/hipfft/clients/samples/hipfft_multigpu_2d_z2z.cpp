// Copyright (C) 2023 Advanced Micro Devices, Inc. All rights
// reserved.
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
#include <iostream>
#include <vector>

#include <hipfft/hipfft.h>

DISABLE_WARNING_PUSH
DISABLE_WARNING_DEPRECATED_DECLARATIONS
DISABLE_WARNING_RETURN_TYPE
#include <hip/hip_runtime_api.h>
DISABLE_WARNING_POP

#include "../hipfft_params.h"

int main()
{
    std::cout << "Multi-gpu hipFFT 2D double-precision complex-to-complex transform\n";

    // 2D FFTs are encountered in diverse applications of image processing,
    // examples range from image denoising to RTM seismic imaging.
    // In this example we compare the 2D FFT computation using single vs multiple GPUs.

    // Note that when using cuFFTXt with two or more GPUs, its latest version requires
    // a minimum size per dimension greater or equal than 32 and less equal than 4096
    // for single precision, and 2048 for double precision.
    const int Nx        = 512;
    const int Ny        = 512;
    int       direction = HIPFFT_FORWARD; // forward=-1, backward=1

    int verbose = 0;

    // Initialize reference data
    std::vector<std::complex<double>> cinput(Nx * Ny);
    for(size_t i = 0; i < Nx * Ny; i++)
    {
        cinput[i] = i;
    }

    if(verbose)
    {
        std::cout << "Input:\n";
        for(int i = 0; i < Nx; i++)
        {
            for(int j = 0; j < Ny; j++)
            {
                int pos = i * Ny + j;
                std::cout << cinput[pos] << " ";
            }
            std::cout << "\n";
        }
        std::cout << std::endl;
    }

    // Define list of GPUs to use
    std::vector<int> gpus = {0, 1};

    // Create the multi-gpu plan
    hipLibXtDesc* desc; // input descriptor

    hipfftHandle plan = hipfft_params::INVALID_PLAN_HANDLE;
    if(hipfftCreate(&plan) != HIPFFT_SUCCESS)
        throw std::runtime_error("failed to create plan");

    // Create a GPU stream and assign it to the plan
    hipStream_t stream{};
    if(hipStreamCreate(&stream) != hipSuccess)
        throw std::runtime_error("hipStreamCreate failed.");
    if(hipfftSetStream(plan, stream) != HIPFFT_SUCCESS)
        throw std::runtime_error("hipfftSetStream failed.");

    // Assign GPUs to the plan
    hipfftResult hipfft_rt = hipfftXtSetGPUs(plan, gpus.size(), gpus.data());
    if(hipfft_rt != HIPFFT_SUCCESS)
        throw std::runtime_error("hipfftXtSetGPUs failed.");

    // Make the 2D plan
    size_t workSize[gpus.size()];
    hipfft_rt = hipfftMakePlan2d(plan, Nx, Ny, HIPFFT_Z2Z, workSize);
    if(hipfft_rt != HIPFFT_SUCCESS)
        throw std::runtime_error("hipfftMakePlan2d failed.");

    // Copy input data to GPUs
    hipfftXtSubFormat_t format = HIPFFT_XT_FORMAT_INPLACE_SHUFFLED;
    hipfft_rt                  = hipfftXtMalloc(plan, &desc, format);
    if(hipfft_rt != HIPFFT_SUCCESS)
        throw std::runtime_error("hipfftXtMalloc failed.");

    hipfft_rt = hipfftXtMemcpy(plan,
                               reinterpret_cast<void*>(desc),
                               reinterpret_cast<void*>(cinput.data()),
                               HIPFFT_COPY_HOST_TO_DEVICE);
    if(hipfft_rt != HIPFFT_SUCCESS)
        throw std::runtime_error("hipfftXtMemcpy failed.");

    // Execute the plan
    hipfft_rt = hipfftXtExecDescriptor(plan, desc, desc, direction);
    if(hipfft_rt != HIPFFT_SUCCESS)
        throw std::runtime_error("hipfftXtMemcpy failed.");

    // Print output
    if(verbose)
    {
        // Move result to the host
        hipfft_rt = hipfftXtMemcpy(plan,
                                   reinterpret_cast<void*>(cinput.data()),
                                   reinterpret_cast<void*>(desc),
                                   HIPFFT_COPY_DEVICE_TO_HOST);
        if(hipfft_rt != HIPFFT_SUCCESS)
            throw std::runtime_error("hipfftXtMemcpy D2H failed.");

        std::cout << "Output:\n";
        for(size_t i = 0; i < Nx; i++)
        {
            for(size_t j = 0; j < Ny; j++)
            {
                auto pos = i * Ny + j;
                std::cout << cinput[pos] << " ";
            }
            std::cout << "\n";
        }
        std::cout << std::endl;
    }

    // Clean up
    if(hipfftXtFree(desc) != HIPFFT_SUCCESS)
        throw std::runtime_error("hipfftXtFree failed.");

    if(hipfftDestroy(plan) != HIPFFT_SUCCESS)
        throw std::runtime_error("hipfftDestroy failed.");

    if(hipStreamDestroy(stream) != hipSuccess)
        throw std::runtime_error("hipStreamDestroy failed.");

    return 0;
}
