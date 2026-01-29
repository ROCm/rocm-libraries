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

#include "sample_utils.hpp"

DISABLE_WARNING_PUSH
DISABLE_WARNING_DEPRECATED_DECLARATIONS
DISABLE_WARNING_RETURN_TYPE
#include <hip/hip_runtime_api.h>
DISABLE_WARNING_POP

int main()
{
    std::cout << "Multi-gpu hipFFT in-place 3D double-precision real-to-complex transform\n";

    // 2D FFTs are encountered in diverse applications of image processing,
    // examples range from image denoising to RTM seismic imaging.
    // In this example we compare the 2D FFT computation using single vs multiple GPUs.

    // Note that when using cuFFTXt with two or more GPUs, its latest version requires
    // a minimum size per dimension greater or equal than 32 and less equal than 4096
    // for single precision, and 2048 for double precision.
    const int  Nx              = 32;
    const int  Ny              = 32;
    const int  Nz              = 32;
    int        direction       = HIPFFT_FORWARD; // forward=-1, backward=1
    hipfftType transform_type  = HIPFFT_D2Z;     // double to std::complex<double>
    hipfftXtSubFormat_t format = HIPFFT_XT_FORMAT_INPLACE;
    size_t     ngpus           = 2;

    const int Nzp = Nz / 2 + 1;
    
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
    std::vector<double> rinput(Nx * Ny * (Nz + 2));
    for(size_t xidx = 0; xidx < Nx; ++xidx)
    {
        for(size_t yidx = 0; yidx < Ny; ++yidx)
        {
            for(size_t zidx = 0; zidx < Nz; ++zidx)
            {
                const size_t pos = (xidx * Ny + yidx) * (Nz + 2) + zidx;
                const size_t idx = (xidx * Ny + yidx) * Nz + zidx;
                rinput[pos] = idx;
            }
        }
    }

    std::cout << "Input:\n";
    printarraylimit(rinput, Nx, Ny, (Nz / 2 + 1) * 2, printlimit);
    std::cout << "\n";
        
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
        throw std::runtime_error("hipfftXtSetGPUs failed with code" + std::to_string(hipfft_rt));

    // Make the 3D plan

    std::vector<size_t> workSize(ngpus);
    hipfft_rt = hipfftMakePlan3d(plan, Nx, Ny, Nz, transform_type, workSize.data());
    if(hipfft_rt != HIPFFT_SUCCESS)
        throw std::runtime_error("hipfftMakePlan3d failed.");

    // Copy input data to GPUs
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
            = inoutdesc->descriptor->size[idx] / sizeof(decltype(rinput)::value_type);
        std::cout << "\tbuffer " << idx << ": " << inoutdesc->descriptor->size[idx] << " bytes, "
                  << vsize << " values\n";
    }
    std::cout << "\n";

    hipfft_rt = hipfftXtMemcpy(plan,
                               reinterpret_cast<void*>(inoutdesc),
                               reinterpret_cast<void*>(rinput.data()),
                               HIPFFT_COPY_HOST_TO_DEVICE);
    if(hipfft_rt != HIPFFT_SUCCESS)
    {
        std::stringstream ss;
        ss << "hipfftXtMemcpy host-to-device failed with code ";
        ss << std::to_string(hipfft_rt) << " : " << hipfftResult_to_name(hipfft_rt);
        throw std::runtime_error(ss.str());
    }
   
    std::cout << "Distributed input data on the GPUs:\n";
    for(size_t idx = 0; idx < ngpus; ++idx)
    {
        const int Nxmax = (Nx / ngpus)  + ((idx < Nx % ngpus) ? 1 : 0);
        const int Nymax = Ny;
        const int Nzmax = Nz + 2;
        const size_t vsize
            = inoutdesc->descriptor->size[idx] / sizeof(decltype(rinput)::value_type);
        std::vector<decltype(rinput)::value_type> hbuf(vsize);
        std::cout << "buffer " << idx << ": "
                  << Nxmax << " x " << Nymax <<" x " << Nzmax << ": "
                  << Nxmax * Nymax * Nzmax <<" elements, buffer holds " << vsize << " elements\n";
        if(hipMemcpy(hbuf.data(),
                     inoutdesc->descriptor->data[idx],
                     inoutdesc->descriptor->size[idx],
                     hipMemcpyDeviceToHost)
           != hipSuccess)
        {
            throw std::runtime_error("hipMemcpy failed.");
        }
        printarraylimit(hbuf, Nxmax, Nymax, Nzmax, printlimit);
        std::cout << "\n";
        
        //if(idx == 0)
        {
            for(size_t hidx = 0; hidx < hbuf.size(); ++ hidx)
            {
                //std::cout << hidx << "\t" << hbuf[hidx] << "\t" << (hbuf[hidx] - hidx) << "\n";
            }
        }
        std::cout << "\n";
    }

    std::cout << "inoutdesc->subFormat: " << inoutdesc->subFormat << "\n";
    
    // Execute the plan
    std::cout << "Executing the plan...\n";
    hipfft_rt = hipfftXtExecDescriptor(plan, inoutdesc, inoutdesc, direction);
    if(hipfft_rt != HIPFFT_SUCCESS)
    {
        std::stringstream ss;
        ss << "hipfftXtExecDescriptor failed with code ";
        ss << std::to_string(hipfft_rt) << " : " << hipfftResult_to_name(hipfft_rt);
        throw std::runtime_error(ss.str());
    }

    std::cout << "inoutdesc->subFormat: " << inoutdesc->subFormat << "\n";

    std::vector<std::complex<double>> coutput(Nx * Ny * (Nz / 2 + 1));
    
    std::cout << "Distributed output data on the GPUs:\n";
    for(size_t idx = 0; idx < ngpus; ++idx)
    {
        const int Nxmax = Nx;
        const int Nymax = Ny / ngpus + ((idx < Ny % ngpus) ? 1 : 0);
        const int Nzmax = Nzp;
        const size_t vsize
            = inoutdesc->descriptor->size[idx] / sizeof(decltype(coutput)::value_type);
        std::cout << "buffer " << idx << ": "
                  << Nxmax << " x " << Nymax<< " x " << Nzmax << ": "
                  << Nxmax * Nymax * Nzmax <<" elements, buffer holds " << vsize << " elements\n";

        std::vector<decltype(coutput)::value_type> hbuf(vsize);
        if(hipMemcpy(hbuf.data(),
                     inoutdesc->descriptor->data[idx],
                     inoutdesc->descriptor->size[idx],
                     hipMemcpyDeviceToHost)
           != hipSuccess)
        {
            throw std::runtime_error("hipMemcpy failed.");
        }
        printarraylimit(hbuf, Nxmax, Nymax, Nzmax, printlimit);
        std::cout << "\n";
    }
    
    // Move result to the host
    hipfft_rt = hipfftXtMemcpy(plan,
                               reinterpret_cast<void*>(coutput.data()),
                               reinterpret_cast<void*>(inoutdesc),
                               HIPFFT_COPY_DEVICE_TO_HOST);
    if(hipfft_rt != HIPFFT_SUCCESS)
        throw std::runtime_error("hipfftXtMemcpy D2H failed.");

    std::cout << "Collected output:\n";
    printarraylimit(coutput, Nx, Ny, Nzp, printlimit);
    
    // Clean up
    if(hipfftXtFree(inoutdesc) != HIPFFT_SUCCESS)
        throw std::runtime_error("hipfftXtFree failed.");

    if(hipfftDestroy(plan) != HIPFFT_SUCCESS)
        throw std::runtime_error("hipfftDestroy failed.");

    if(hipStreamDestroy(stream) != hipSuccess)
        throw std::runtime_error("hipStreamDestroy failed.");

    return 0;
}
