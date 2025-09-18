// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights
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
#include <numeric>
#include <vector>

#include <hipfft/hipfft.h>
#include <hipfft/hipfftXt.h>

DISABLE_WARNING_PUSH
DISABLE_WARNING_DEPRECATED_DECLARATIONS
DISABLE_WARNING_RETURN_TYPE
#include <hip/hip_runtime_api.h>
DISABLE_WARNING_POP

int main()
{
    std::cout << "Multi-gpu hipFFT in-place 2D double-precision complex-to-complex transform\n";

    // 2D FFTs are encountered in diverse applications of image processing,
    // examples range from image denoising to RTM seismic imaging.
    // In this example we compare the 2D FFT computation using single vs multiple GPUs.

    // Note that when using cuFFTXt with two or more GPUs, its latest version requires
    // a minimum size per dimension greater or equal than 32 and less equal than 4096
    // for single precision, and 2048 for double precision.
    const int  Nx             = 25;
    const int  Ny             = 16;
    int        direction      = HIPFFT_FORWARD; // forward=-1, backward=1
    hipfftType transform_type = HIPFFT_Z2Z; // std::complex<double> to std::complex<double>
    size_t     ngpus          = 2;

    // We only want to print a subset of the data:
    const int printlimit = 4;

    int deviceCount;
    if(hipGetDeviceCount(&deviceCount) != hipSuccess)
        throw std::runtime_error("hipGetDeviceCount failed.");
    std::cout << "Number of available devices: " << deviceCount << std::endl;
    if(deviceCount < ngpus)
    {
        std::cout << "Sample needs at least " << ngpus << "GPUs\n";
        return 0;
    }
    std::cout << "\n";

    // Initialize reference data
    std::vector<std::complex<double>> cinput(Nx * Ny);
    for(size_t idx = 0; idx < Nx * Ny; ++idx)
    {
        cinput[idx] = idx;
    }

    std::cout << "Input:\n";
    for(int xidx = 0; xidx < Nx; ++xidx)
    {
        if(xidx > printlimit)
        {
            std::cout << "...\n";
            xidx = Nx - 1;
        }
        for(int yidx = 0; yidx < Ny; ++yidx)
        {
            if(yidx > printlimit)
            {
                std::cout << "... ";
                yidx = Ny - 1;
            }
            int pos = xidx * Ny + yidx;
            std::cout << cinput[pos] << " ";
        }
        std::cout << "\n";
    }

    // Define list of GPUs to use
    std::vector<int> gpus = {0, 1};

    // Create the multi-gpu plan
    hipLibXtDesc* desc; // input descriptor

    hipfftHandle plan;
    if(hipfftCreate(&plan) != HIPFFT_SUCCESS)
        throw std::runtime_error("failed to create plan");

    // Create a GPU stream and assign it to the plan
    hipStream_t stream{};
    if(hipStreamCreate(&stream) != hipSuccess)
        throw std::runtime_error("hipStreamCreate failed.");
    if(hipfftSetStream(plan, stream) != HIPFFT_SUCCESS)
        throw std::runtime_error("hipfftSetStream failed.");

    // Assign GPUs to the plan
    std::vector<int> gpus(ngpus);
    std::iota(gpus.begin(), gpus.end(), 0);
    hipfftResult hipfft_rt = hipfftXtSetGPUs(plan, gpus.size(), gpus.data());
    if(hipfft_rt != HIPFFT_SUCCESS)
        throw std::runtime_error("hipfftXtSetGPUs failed.");

    // Make the 2D plan

    std::vector<size_t> workSize(ngpus);
    hipfft_rt = hipfftMakePlan2d(plan, Nx, Ny, transform_type, workSize.data());
    if(hipfft_rt != HIPFFT_SUCCESS)
        throw std::runtime_error("hipfftMakePlan2d failed.");

    // Copy input data to GPUs
    hipfftXtSubFormat_t format    = HIPFFT_XT_FORMAT_INPUT;
    hipLibXtDesc*       inoutdesc = nullptr;
    hipfft_rt                     = hipfftXtMalloc(plan, &inoutdesc, format);
    if(hipfft_rt != HIPFFT_SUCCESS)
    {
        std::stringstream ss;
        ss << "hipfftXtMalloc failed with error " << hipfft_rt;
        throw std::runtime_error(ss.str());
    }

    std::cout << "The descriptor is now allocated:\n";
    for(size_t idx = 0; idx < ngpus; ++idx)
    {
        const size_t vsize
            = inoutdesc->descriptor->size[idx] / sizeof(decltype(cinput)::value_type);
        std::cout << "\tbuffer " << idx << ": " << inoutdesc->descriptor->size[idx] << " bytes, "
                  << vsize << " values\n";
    }
    std::cout << "\n";

    hipfft_rt = hipfftXtMemcpy(plan,
                               reinterpret_cast<void*>(inoutdesc),
                               reinterpret_cast<void*>(cinput.data()),
                               HIPFFT_COPY_HOST_TO_DEVICE);
    if(hipfft_rt != HIPFFT_SUCCESS)
        throw std::runtime_error("hipfftXtMemcpy failed.");

    std::cout << "Distributed input data on the GPUs:\n";
    for(size_t idx = 0; idx < ngpus; ++idx)
    {
        std::cout << "buffer " << idx << "\n";
        const size_t vsize
            = inoutdesc->descriptor->size[idx] / sizeof(decltype(cinput)::value_type);
        std::vector<decltype(cinput)::value_type> hbuf(vsize);
        if(hipMemcpy(hbuf.data(),
                     inoutdesc->descriptor->data[idx],
                     inoutdesc->descriptor->size[idx],
                     hipMemcpyDeviceToHost)
           != hipSuccess)
        {
            throw std::runtime_error("hipMemcpy failed.");
        }
        const int Nxmax = Nx / ngpus + ((idx < Nx % ngpus) ? 1 : 0);
        for(int xidx = 0; xidx < Nxmax; ++xidx)
        {
            if(xidx > printlimit)
            {
                std::cout << "...\n";
                xidx = Nxmax - 1;
            }
            for(int yidx = 0; yidx < Ny; ++yidx)
            {
                if(yidx > printlimit)
                {
                    std::cout << "... ";
                    yidx = Ny - 1;
                }
                int pos = xidx * Ny + yidx;
                std::cout << hbuf[pos] << " ";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }

    // Execute the plan
    hipfft_rt = hipfftXtExecDescriptor(plan, inoutdesc, inoutdesc, direction);
    if(hipfft_rt != HIPFFT_SUCCESS)
        throw std::runtime_error("hipfftXtMemcpy failed.");

    std::cout << "Distributed output data on the GPUs:\n";
    for(size_t idx = 0; idx < ngpus; ++idx)
    {
        std::cout << "buffer " << idx << "\n";
        const size_t vsize
            = inoutdesc->descriptor->size[idx] / sizeof(decltype(cinput)::value_type);

        std::vector<decltype(cinput)::value_type> hbuf(vsize);
        if(hipMemcpy(hbuf.data(),
                     inoutdesc->descriptor->data[idx],
                     inoutdesc->descriptor->size[idx],
                     hipMemcpyDeviceToHost)
           != hipSuccess)
        {
            throw std::runtime_error("hipMemcpy failed.");
        }
        const int Nxmax = Nx / ngpus + ((idx < Nx % ngpus) ? 1 : 0);
        for(int xidx = 0; xidx < Nxmax; ++xidx)
        {
            if(xidx > printlimit)
            {
                std::cout << "...\n";
                xidx = Nxmax - 1;
            }

            for(int yidx = 0; yidx < Ny; ++yidx)
            {
                if(yidx > printlimit)
                {
                    std::cout << "... ";
                    yidx = Ny - 1;
                }
                int pos = xidx * Ny + yidx;
                std::cout << hbuf[pos] << " ";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }

    // Move result to the host
    hipfft_rt = hipfftXtMemcpy(plan,
                               reinterpret_cast<void*>(cinput.data()),
                               reinterpret_cast<void*>(inoutdesc),
                               HIPFFT_COPY_DEVICE_TO_HOST);
    if(hipfft_rt != HIPFFT_SUCCESS)
        throw std::runtime_error("hipfftXtMemcpy D2H failed.");

    std::cout << "Collected output:\n";
    for(size_t xidx = 0; xidx < Nx; ++xidx)
    {
        if(xidx > printlimit)
        {
            std::cout << "...\n";
            xidx = Nx - 1;
        }
        for(size_t yidx = 0; yidx < Ny; ++yidx)
        {
            if(yidx > printlimit)
            {
                std::cout << "... ";
                yidx = Ny - 1;
            }
            auto pos = xidx * Ny + yidx;
            std::cout << cinput[pos] << " ";
        }
        std::cout << "\n";
    }
    std::cout << std::endl;

    // Clean up
    if(hipfftXtFree(inoutdesc) != HIPFFT_SUCCESS)
        throw std::runtime_error("hipfftXtFree failed.");

    if(hipfftDestroy(plan) != HIPFFT_SUCCESS)
        throw std::runtime_error("hipfftDestroy failed.");

    if(hipStreamDestroy(stream) != hipSuccess)
        throw std::runtime_error("hipStreamDestroy failed.");

    return 0;
}
